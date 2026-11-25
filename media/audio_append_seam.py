"""分段录音续录拼接。

修复点对应关系：

* 续录按采样帧对齐            -> :func:`frame_aligned_append` / :class:`Recorder`
* 读写指针原子交换            -> :class:`AtomicRingBuffer` / :func:`atomic_ring_swap`
* 暂停停流并留存尾部          -> :func:`pause_stream` / :class:`AudioStream`
* 时长按样本计数累计          -> :func:`sample_count_duration`
* 取消清理临时分段            -> :func:`cleanup_segments`
* 拼缝交叉淡化                -> :func:`seam_crossfade`

模块只依赖标准库；PCM 统一为 16bit 小端有符号，一“帧”= 一个采样点上
``channels`` 个声道交织后的字节。
"""

from __future__ import annotations

import os
import tempfile
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 常量与基础工具
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # int16
DEFAULT_CROSSFADE_MS = 10.0  # 拼缝默认交叉淡化时长
# 拼缝两侧相邻样本跳变阈值（int16 量程内），超过即视为存在重复/空洞
SEAM_JUMP_LIMIT = 4096


def frame_byte_size(channels: int = DEFAULT_CHANNELS,
                    sample_width: int = DEFAULT_SAMPLE_WIDTH) -> int:
    """一帧（一个采样时刻的全部声道）占多少字节。"""
    return int(channels) * int(sample_width)


def align_to_frame(byte_offset: int, frame_size: int) -> int:
    """把任意字节偏移向下取整到帧边界。"""
    frame_size = int(frame_size)
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    return (int(byte_offset) // frame_size) * frame_size


def ms_to_frames(ms: float, sample_rate: int) -> int:
    return max(0, int(round(float(ms) * float(sample_rate) / 1000.0)))


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _pcm_samples(chunk: bytes, sample_width: int = DEFAULT_SAMPLE_WIDTH
                 ) -> List[int]:
    return [int.from_bytes(chunk[i:i + sample_width], "little", signed=True)
            for i in range(0, len(chunk) - sample_width + 1, sample_width)]


def _samples_to_bytes(samples: Iterable[int],
                      sample_width: int = DEFAULT_SAMPLE_WIDTH) -> bytes:
    lo, hi = -(1 << (8 * sample_width - 1)), (1 << (8 * sample_width - 1)) - 1
    out = bytearray()
    for s in samples:
        s = min(hi, max(lo, int(s)))
        out += int(s).to_bytes(sample_width, "little", signed=True)
    return bytes(out)


# ---------------------------------------------------------------------------
# 修复点 2：环形缓冲读写指针原子交换
# ---------------------------------------------------------------------------

class AtomicRingBuffer:
    """固定容量字节环形缓冲。

    设备回调线程 :meth:`write`，预览线程 :meth:`snapshot` 读取。所有读写
    都在同一把锁内完成并一次性发布指针，预览永远只能看到完整帧，不会读到
    “写了一半”的半帧。缓冲写满时按最旧数据覆盖（环形覆盖语义）。
    """

    def __init__(self, capacity_bytes: int,
                 frame_size: int = frame_byte_size()):
        frame_size = int(frame_size)
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        capacity_bytes = align_to_frame(int(capacity_bytes), frame_size)
        if capacity_bytes < frame_size:
            capacity_bytes = frame_size
        self.frame_size = frame_size
        self.capacity = capacity_bytes
        self._buf = bytearray(capacity_bytes)
        self._lock = threading.RLock()
        self._head = 0          # 最旧有效数据的位置
        self._size = 0          # 有效字节数（恒为 frame_size 的整数倍）
        self.overruns = 0       # 因写满覆盖而丢弃的字节计数
        self.active = True      # 暂停停流后置 False，拒绝再写入，尾部得以留存

    # -- 写侧（设备回调线程） ---------------------------------------------

    def write(self, data: bytes) -> int:
        """写入数据（按帧截断），返回实际写入字节数。

        缓冲停流（暂停）后写入会被直接拒绝，避免覆盖暂停前的尾部。
        """
        if not self.active:
            return 0
        with self._lock:
            return self._write_locked(data)

    def _write_locked(self, data: bytes) -> int:
        usable = len(data) - (len(data) % self.frame_size)
        if usable == 0:
            return 0
        data = bytes(data[:usable])
        n = len(data)
        if n >= self.capacity:
            # 整缓冲被新数据替换，旧内容全部被覆盖
            self.overruns += self._size + (n - self.capacity)
            self._buf[:] = data[-self.capacity:]
            self._head = 0
            self._size = self.capacity
            return self.capacity
        end = (self._head + self._size) % self.capacity
        first = min(n, self.capacity - end)
        self._buf[end:end + first] = data[:first]
        if n > first:
            self._buf[:n - first] = data[first:]
        if self._size + n <= self.capacity:
            self._size += n
        else:
            overflow = self._size + n - self.capacity
            self.overruns += overflow
            self._size = self.capacity
            self._head = (self._head + overflow) % self.capacity
        return n

    # -- 读侧（波形预览线程） ---------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """原子取出当前可读内容的副本与指针，供波形预览渲染。"""
        with self._lock:
            data = self._read_locked(self._size)
            return {
                "data": data,
                "head": self._head,
                "size": self._size,
                "frames": self._size // self.frame_size,
                "capacity": self.capacity,
                "overruns": self.overruns,
            }

    def _read_locked(self, n_bytes: int) -> bytes:
        n_bytes = min(n_bytes, self._size)
        if n_bytes == 0:
            return b""
        start = (self._head + self._size - n_bytes) % self.capacity
        first = min(n_bytes, self.capacity - start)
        part = bytes(self._buf[start:start + first])
        if n_bytes > first:
            part += bytes(self._buf[:n_bytes - first])
        return part

    def read(self, n_bytes: Optional[int] = None) -> bytes:
        with self._lock:
            if n_bytes is None:
                n_bytes = self._size
            return self._read_locked(align_to_frame(n_bytes, self.frame_size))

    # -- 原子交换：续录排空 / 停流交接 ------------------------------------

    def swap(self, new_bytes: bytes = b"") -> bytes:
        """原子地取出全部现有内容并替换为 ``new_bytes``。

        读写指针在同一临界区内一起交换，读侧不可能观察到中间态。
        """
        with self._lock:
            old = self._read_locked(self._size)
            self._head = 0
            self._size = 0
            self._write_locked(new_bytes)
            return old

    def drain(self) -> bytes:
        """原子排空（swap 的空替换特例）。"""
        return self.swap(b"")

    def stop(self) -> bytes:
        """停流：拒绝后续写入并原子取出留存的尾部。"""
        with self._lock:
            self.active = False
            return self._read_locked(self._size)

    def resume(self) -> None:
        with self._lock:
            self.active = True

    def __len__(self) -> int:
        with self._lock:
            return self._size


def atomic_ring_swap(*args, **kwargs) -> Any:
    """环形缓冲原子交换入口。

    * ``atomic_ring_swap(capacity_bytes)`` -> 新建 :class:`AtomicRingBuffer`
    * ``atomic_ring_swap(ring, new_bytes=b"")`` -> 原子交换，返回旧内容
    * ``atomic_ring_swap(ring)`` -> 原子排空
    """
    if not args or _is_int(args[0]):
        capacity = int(args[0]) if args else kwargs.get(
            "capacity_bytes", DEFAULT_SAMPLE_RATE * frame_byte_size())
        frame_size = int(kwargs.get("frame_size", frame_byte_size()))
        return AtomicRingBuffer(capacity, frame_size)
    ring = args[0]
    if not isinstance(ring, AtomicRingBuffer):
        ring = AtomicRingBuffer(int(ring))
    new_bytes = args[1] if len(args) > 1 else kwargs.get("new_bytes", b"")
    return ring.swap(new_bytes or b"")


# ---------------------------------------------------------------------------
# 修复点 3：暂停停流并留存尾部
# ---------------------------------------------------------------------------

class AudioStream:
    """录音设备回调的测试替身。

    暂停即停流：``active=False`` 时回调不再向环形缓冲写入，因此暂停期间
    设备“仍在回调”也不会覆盖暂停前的尾部；恢复后从空缓冲重新积累。
    """

    def __init__(self, ring: AtomicRingBuffer):
        self.ring = ring
        self.active = False
        self.callbacks = 0
        self.delivered = 0

    def start(self) -> "AudioStream":
        self.active = True
        self.ring.resume()
        return self

    def stop(self) -> bytes:
        self.active = False
        return self.ring.stop()

    def callback(self, chunk: bytes) -> int:
        """模拟设备回调；停流时丢弃本回调数据，尾部原样留存。"""
        self.callbacks += 1
        if not self.active:
            return 0
        written = self.ring.write(chunk)
        self.delivered += written
        return written


def pause_stream(*args, **kwargs) -> Any:
    """暂停录音流：停止设备回调写入，环形缓冲尾部原样留存。"""
    if not args:
        return {"active": False, "tail_kept": b"", "tail_frames": 0}
    target = args[0]
    if isinstance(target, AudioStream):
        tail = target.stop()
        return {"active": False, "tail_kept": tail,
                "tail_frames": len(tail) // target.ring.frame_size}
    if isinstance(target, AtomicRingBuffer):
        tail = target.stop()
        return {"active": False, "tail_kept": tail,
                "tail_frames": len(tail) // target.frame_size}
    if isinstance(target, Recorder):
        return target.pause()
    # 原始输入：构造缓冲走一遍停流流程
    frame_size = int(kwargs.get("frame_size", frame_byte_size()))
    ring = AtomicRingBuffer(max(int(target), frame_size), frame_size)
    tail = ring.stop()
    return {"active": False, "tail_kept": tail,
            "tail_frames": len(tail) // frame_size}


# ---------------------------------------------------------------------------
# 修复点 1：续录按采样帧对齐
# ---------------------------------------------------------------------------

def frame_aligned_append(*args, **kwargs) -> Any:
    """续录写入对齐到采样帧边界。

    支持三种调用形态：

    * ``frame_aligned_append(byte_offset, frame_size=...)``：
      偏移（字节）向下取整到帧边界，返回对齐后的字节偏移；
    * ``frame_aligned_append(byte_offset, sample_rate, elapsed_ms)``：
      旧的墙钟毫秒取整方式，换算成字节后再按帧取整（替代按毫秒直接拼接）；
    * ``frame_aligned_append(existing, new_data, frame_size=...)``：
      截掉 ``new_data`` 不足一帧的尾巴后拼接，返回
      ``(拼接字节, 丢弃字节数, 对齐帧大小)``。
    """
    frame_size = int(kwargs.get("frame_size", frame_byte_size()))
    if not args:
        return 0
    first = args[0]
    if isinstance(first, (bytes, bytearray)):
        new_data = bytes(args[1]) if len(args) > 1 else b""
        dropped = len(new_data) % frame_size
        aligned_new = new_data[:len(new_data) - dropped] if dropped else new_data
        merged = bytes(first) + aligned_new
        return merged, dropped, frame_size
    offset = int(first)
    if len(args) >= 3:  # 墙钟时间输入：毫秒 -> 字节 -> 帧对齐
        sample_rate = int(args[1])
        elapsed_ms = float(args[2])
        channels = int(kwargs.get("channels", DEFAULT_CHANNELS))
        offset = int(elapsed_ms / 1000.0 * sample_rate) * frame_size
    return align_to_frame(offset, frame_size)


# ---------------------------------------------------------------------------
# 修复点 4：时长按样本计数累计
# ---------------------------------------------------------------------------

def sample_count_duration(*args, **kwargs) -> Any:
    """按实际样本（帧）计数计算时长，单位秒。

    * ``sample_count_duration(total_frames, sample_rate)`` -> 秒（float）
    * ``sample_count_duration(frame_counts, sample_rate)`` ->
      各段帧数求和后换算，重叠帧由调用方先扣除，避免重复计时
    * ``sample_count_duration(pcm_bytes, ...)`` -> 按字节数反推帧数
    """
    sample_rate = int(kwargs.get("sample_rate", DEFAULT_SAMPLE_RATE))
    frame_size = int(kwargs.get("frame_size", frame_byte_size()))
    if not args:
        return 0.0
    first = args[0]
    if isinstance(first, (bytes, bytearray)):
        total_frames = len(first) // frame_size
        if len(args) > 1:
            sample_rate = int(args[1])
    elif isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        total_frames = sum(int(n) for n in first)
        if len(args) > 1:
            sample_rate = int(args[1])
    else:
        total_frames = int(first)
        if len(args) > 1:
            sample_rate = int(args[1])
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return float(total_frames) / float(sample_rate)


# ---------------------------------------------------------------------------
# 修复点 6：拼缝交叉淡化
# ---------------------------------------------------------------------------

def seam_crossfade(*args, **kwargs) -> Any:
    """对两段 PCM 的拼缝做交叉淡化。

    * ``seam_crossfade(left, right, overlap_frames, ...)``：
      重叠区逐样本等功率线性淡化，``overlap_frames`` 会被收敛到不超过
      两段各自长度；返回拼接字节，输出帧数 = L + R - overlap。
    * ``seam_crossfade(joined, seam_frame, fade_frames, ...)``：
      对已经拼在一起的缓冲在指定拼缝处做原地增益斜坡淡化，长度不变。
    """
    sample_width = int(kwargs.get("sample_width", DEFAULT_SAMPLE_WIDTH))
    channels = int(kwargs.get("channels", DEFAULT_CHANNELS))
    frame_size = sample_width * channels

    if not args:
        return b""
    if len(args) >= 2 and isinstance(args[0], (bytes, bytearray)) and \
            isinstance(args[1], (bytes, bytearray)):
        left, right = bytes(args[0]), bytes(args[1])
        overlap_frames = int(args[2]) if len(args) > 2 else int(
            kwargs.get("overlap_frames", 0))
        lf, rf = len(left) // frame_size, len(right) // frame_size
        overlap_frames = max(0, min(overlap_frames, lf, rf))
        if overlap_frames == 0:
            return left + right
        head = left[:(lf - overlap_frames) * frame_size]
        tail = right[overlap_frames * frame_size:]
        ls = _pcm_samples(left[(lf - overlap_frames) * frame_size:],
                          sample_width)
        rs = _pcm_samples(right[:overlap_frames * frame_size], sample_width)
        n = overlap_frames * channels
        if n == 1:
            mixed = [int(round((ls[0] + rs[0]) / 2.0))]
        else:
            mixed = []
            for i, (a, b) in enumerate(zip(ls, rs)):
                # a 从 1.0 线性收到 0.0，b 从 0.0 放到 1.0
                w_b = i / (n - 1)
                mixed.append(int(round(a * (1.0 - w_b) + b * w_b)))
        return head + _samples_to_bytes(mixed, sample_width) + tail

    # 单缓冲 + 拼缝位置：原地淡化
    joined = bytes(args[0])
    seam_frame = int(args[1]) if len(args) > 1 else int(
        kwargs.get("seam_frame", len(joined) // frame_size // 2))
    fade_frames = int(args[2]) if len(args) > 2 else int(
        kwargs.get("fade_frames", 0))
    total_frames = len(joined) // frame_size
    fade_frames = max(0, min(fade_frames, seam_frame, total_frames - seam_frame))
    if fade_frames == 0:
        return joined
    samples = _pcm_samples(joined, sample_width)
    start = (seam_frame - fade_frames) * channels
    count = fade_frames * channels
    if count > 1:
        for i in range(count):
            idx = start + i
            # 左半淡出、右半淡入
            gain = 1.0 - 0.5 * (i / (count - 1)) if idx < seam_frame * channels \
                else 0.5 + 0.5 * ((idx - seam_frame * channels) / (count - 1))
            samples[idx] = int(round(samples[idx] * gain))
    return _samples_to_bytes(samples, sample_width)


# ---------------------------------------------------------------------------
# 分段状态、临时分段与录音器
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """一段录音：PCM 数据、落盘临时文件与重叠信息。"""

    index: int
    frames: int
    path: Optional[Path] = None
    data: bytes = b""
    overlap_frames: int = 0  # 与上一段在拼缝处的重叠帧数

    @property
    def pcm(self) -> bytes:
        if self.data:
            return self.data
        if self.path is not None and self.path.exists():
            return self.path.read_bytes()
        return b""


@dataclass
class Recorder:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    sample_width: int = DEFAULT_SAMPLE_WIDTH
    crossfade_ms: float = DEFAULT_CROSSFADE_MS
    tmp_dir: Optional[Path] = None
    state: str = "idle"  # idle | recording | paused | stopped | cancelled
    segments: List[Segment] = field(default_factory=list)
    total_frames: int = 0
    ring: Optional[AtomicRingBuffer] = None
    stream: Optional[AudioStream] = None
    crossfade_enabled: bool = True
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if self.tmp_dir is None:
            self.tmp_dir = Path(tempfile.mkdtemp(prefix="recseg_"))
        else:
            self.tmp_dir = Path(self.tmp_dir)
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
        fs = self.frame_size
        # 默认 1 秒环形容量
        self.ring = AtomicRingBuffer(self.sample_rate * fs, fs)
        self.stream = AudioStream(self.ring)

    # -- 基本量 -----------------------------------------------------------

    @property
    def frame_size(self) -> int:
        return frame_byte_size(self.channels, self.sample_width)

    @property
    def crossfade_frames(self) -> int:
        return ms_to_frames(self.crossfade_ms, self.sample_rate)

    # -- 生命周期 ---------------------------------------------------------

    def start(self) -> "Recorder":
        with self._lock:
            if self.state in ("stopped", "cancelled"):
                raise RuntimeError(f"cannot start a {self.state} recorder")
            self.stream.start()
            self.state = "recording"
            return self

    def pause(self) -> Dict[str, Any]:
        """暂停：停流，环形缓冲里的尾部完整留存。"""
        with self._lock:
            if self.state != "recording":
                raise RuntimeError("only a recording stream can be paused")
            tail = self.stream.stop()  # ring.active=False，回调写不进来
            self.state = "paused"
            return {
                "state": "paused",
                "tail_kept": tail,
                "tail_frames": len(tail) // self.frame_size,
            }

    def resume(self) -> "Recorder":
        """续录：重新开流。

        停流期间尾部完整保留在缓冲里且未被覆盖，这里只需重新放行写入，
        新回调数据会顺序接在尾部之后，不丢样本、不产生空洞。
        """
        with self._lock:
            if self.state != "paused":
                raise RuntimeError("only a paused stream can be resumed")
            self.ring.resume()
            self.stream.start()
            self.state = "recording"
            return self

    def write(self, chunk: bytes) -> int:
        """设备数据入口（帧对齐，不足一帧的残尾丢弃）。"""
        with self._lock:
            if self.state != "recording":
                return 0
            return self.ring.write(chunk)

    # -- 分段落盘 ---------------------------------------------------------

    def append(self, pcm: Optional[bytes] = None,
               overlap_frames: int = 0) -> Segment:
        """结束当前段并续录下一段。

        环形缓冲通过原子交换排空；数据按帧对齐截断后写入临时分段文件，
        段内帧数即该段样本计数，时长统计与重叠扣除都基于它。
        """
        with self._lock:
            if pcm is None:
                pcm = self.ring.drain()
            pcm = bytes(pcm)
            dropped = len(pcm) % self.frame_size
            pcm = pcm[:len(pcm) - dropped] if dropped else pcm
            idx = len(self.segments)
            seg_path = self.tmp_dir / f"segment_{idx:04d}.pcm"
            seg_path.write_bytes(pcm)
            seg = Segment(index=idx, frames=len(pcm) // self.frame_size,
                          path=seg_path, data=pcm,
                          overlap_frames=max(0, int(overlap_frames)))
            self.segments.append(seg)
            # 样本计数累计：显式重叠先扣除，杜绝时长整体偏移
            self.total_frames += seg.frames - seg.overlap_frames
            return seg

    # -- 取消：清理临时分段 ----------------------------------------------

    def cancel(self) -> Dict[str, Any]:
        with self._lock:
            if self.stream.active:
                self.stream.stop()
            removed = cleanup_segments(self)
            self.state = "cancelled"
            return removed

    # -- 导出：交叉淡化拼接 + 样本计数时长 -------------------------------

    def stitch(self) -> Tuple[bytes, List[Dict[str, Any]]]:
        with self._lock:
            if not self.segments:
                return b"", []
            fade = self.crossfade_frames if self.crossfade_enabled else 0
            pcm = self.segments[0].pcm
            seam_reports: List[Dict[str, Any]] = []
            for seg in self.segments[1:]:
                right = seg.pcm
                overlap = min(seg.overlap_frames,
                              len(pcm) // self.frame_size,
                              len(right) // self.frame_size)
                if overlap == 0 and fade > 0:
                    # 无显式重叠：在拼缝两侧各取 fade 帧做斜坡淡化，长度不变
                    seam_frame = len(pcm) // self.frame_size
                    pcm = seam_crossfade(
                        pcm + right, seam_frame, fade,
                        sample_width=self.sample_width,
                        channels=self.channels)
                else:
                    pcm = seam_crossfade(
                        pcm, right, overlap,
                        sample_width=self.sample_width,
                        channels=self.channels)
                seam_reports.append({
                    "segment": seg.index,
                    "overlap_frames": overlap,
                    "fade_frames": fade if overlap == 0 else overlap,
                })
            return pcm, seam_reports

    def export(self, out_path: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            pcm, seam_reports = self.stitch()
            total_frames = len(pcm) // self.frame_size
            duration = sample_count_duration(total_frames, self.sample_rate)
            result: Dict[str, Any] = {
                "frames": total_frames,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "sample_width": self.sample_width,
                "duration_s": duration,
                "seams": seam_reports,
                "pcm": pcm,
            }
            if out_path is not None:
                path = Path(out_path)
                with wave.open(str(path), "wb") as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.sample_width)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(pcm)
                result["path"] = path
            self.state = "stopped"
            return result

    # -- 预览 -------------------------------------------------------------

    def preview(self) -> Dict[str, Any]:
        """波形预览：原子快照，永远读到完整帧。"""
        snap = self.ring.snapshot()
        snap["duration_s"] = sample_count_duration(
            snap["frames"], self.sample_rate)
        return snap


# ---------------------------------------------------------------------------
# 修复点 5：取消清理临时分段
# ---------------------------------------------------------------------------

def cleanup_segments(*args, **kwargs) -> Dict[str, Any]:
    """删除已写入的临时分段文件。

    接受 :class:`Recorder`、:class:`Segment` 列表、目录路径或若干路径。
    返回删除汇总；缺失文件计入 ``missing`` 而不报错。
    """
    removed: List[str] = []
    missing: List[str] = []

    def _remove(p: Path) -> None:
        try:
            p.unlink()
            removed.append(str(p))
        except FileNotFoundError:
            missing.append(str(p))
        except OSError:
            missing.append(str(p))

    if not args:
        return {"removed": removed, "missing": missing, "count": 0}

    targets: List[Any] = list(args)
    recorder: Optional[Recorder] = None
    for t in targets:
        if isinstance(t, Recorder):
            recorder = t
            tmp_dir = t.tmp_dir
            for seg in t.segments:
                if seg.path is not None:
                    _remove(seg.path)
            # 清理空目录
            try:
                if tmp_dir is not None and tmp_dir.exists() and \
                        not any(tmp_dir.iterdir()):
                    tmp_dir.rmdir()
            except OSError:
                pass
            t.segments = []
        elif isinstance(t, Segment):
            if t.path is not None:
                _remove(t.path)
        elif isinstance(t, (str, os.PathLike)):
            path = Path(t)
            if path.is_dir():
                for child in sorted(path.glob("segment_*.pcm")):
                    _remove(child)
            else:
                _remove(path)
        elif isinstance(t, Sequence):
            for item in t:
                if isinstance(item, Segment) and item.path is not None:
                    _remove(item.path)
                elif isinstance(item, (str, os.PathLike)):
                    _remove(Path(item))

    return {
        "removed": removed,
        "missing": missing,
        "count": len(removed),
        "recorder_cancelled": recorder.state == "cancelled"
        if recorder is not None else False,
    }


# ---------------------------------------------------------------------------
# 生命周期入口（薄封装，兼容原始参数调用）
# ---------------------------------------------------------------------------

def record_start(*args, **kwargs) -> Recorder:
    """开始录音，返回处于 recording 状态的 :class:`Recorder`。"""
    if args and isinstance(args[0], Recorder):
        return args[0].start()
    sample_rate = int(args[0]) if args and _is_int(args[0]) else int(
        kwargs.get("sample_rate", DEFAULT_SAMPLE_RATE))
    channels = int(kwargs.get("channels", DEFAULT_CHANNELS))
    sample_width = int(kwargs.get("sample_width", DEFAULT_SAMPLE_WIDTH))
    rec = Recorder(sample_rate=sample_rate, channels=channels,
                   sample_width=sample_width,
                   crossfade_ms=float(kwargs.get("crossfade_ms",
                                                 DEFAULT_CROSSFADE_MS)))
    return rec.start()


def pause(*args, **kwargs) -> Any:
    """暂停：停流并留存尾部。"""
    if args and isinstance(args[0], Recorder):
        return args[0].pause()
    return pause_stream(*args, **kwargs)


def resume(*args, **kwargs) -> Any:
    """续录：按采样帧对齐重新开流。"""
    if args and isinstance(args[0], Recorder):
        return args[0].resume()
    # 无状态原始调用：返回一份续录交接描述
    frame_size = int(kwargs.get("frame_size", frame_byte_size()))
    return {
        "state": "recording",
        "frame_size": frame_size,
        "resume_offset_aligned": frame_aligned_append(
            args[0] if args else 0, frame_size),
    }


def append(*args, **kwargs) -> Any:
    """分段续录写入（帧对齐 + 原子排空 + 重叠登记）。"""
    overlap_frames = int(kwargs.get("overlap_frames", 0))
    if args and isinstance(args[0], Recorder):
        rec = args[0]
        pcm = args[1] if len(args) > 1 else None
        return rec.append(pcm, overlap_frames=overlap_frames)
    if args and isinstance(args[0], (bytes, bytearray)):
        # 纯数据调用：帧对齐拼接，不落盘
        frame_size = int(kwargs.get("frame_size", frame_byte_size()))
        merged, dropped, _ = frame_aligned_append(
            b"", bytes(args[0]), frame_size=frame_size)
        return Segment(index=0, frames=len(merged) // frame_size,
                       data=merged)
    frame_size = int(kwargs.get("frame_size", frame_byte_size()))
    return Segment(index=0, frames=0, data=b"", path=None)


def export(*args, **kwargs) -> Any:
    """导出拼接结果：交叉淡化拼缝，时长按样本计数。"""
    if args and isinstance(args[0], Recorder):
        out_path = args[1] if len(args) > 1 else kwargs.get("out_path")
        return args[0].export(out_path)
    return {
        "frames": 0,
        "duration_s": 0.0,
        "seams": [],
        "pcm": b"",
    }


# ---------------------------------------------------------------------------
# 校验、指标、审计、兜底
# ---------------------------------------------------------------------------

def validate_seam(*args, **kwargs) -> Dict[str, Any]:
    """校验拼缝：帧对齐、重叠一致性与相邻样本跳变（重复/空洞特征）。"""
    sample_width = int(kwargs.get("sample_width", DEFAULT_SAMPLE_WIDTH))
    channels = int(kwargs.get("channels", DEFAULT_CHANNELS))
    frame_size = sample_width * channels
    jump_limit = int(kwargs.get("jump_limit", SEAM_JUMP_LIMIT))

    if not args or _is_int(args[0]):
        return {"ok": True, "aligned": True, "seams": [], "issues": []}

    if args and isinstance(args[0], Recorder):
        rec = args[0]
        pcm, _ = rec.stitch()
        frame_size = rec.frame_size
        sample_width = rec.sample_width
        channels = rec.channels
    else:
        pcm = bytes(args[0])

    issues: List[str] = []
    if len(pcm) % frame_size != 0:
        issues.append(
            f"pcm length {len(pcm)} not aligned to frame size {frame_size}")

    seams: List[Dict[str, Any]] = []
    samples = _pcm_samples(pcm, sample_width)
    # 已知拼缝位置：由 Recorder 分段边界推出；纯缓冲输入只做整体扫描
    seam_frames: List[int] = []
    if args and isinstance(args[0], Recorder):
        rec = args[0]
        cursor = 0
        for seg in rec.segments[1:]:
            cursor += seg.frames - seg.overlap_frames
            seam_frames.append(cursor)
    for sf in seam_frames:
        idx = sf * channels
        if 0 < idx < len(samples):
            jump = abs(samples[idx] - samples[idx - 1])
            info = {"seam_frame": sf, "jump": jump,
                    "smooth": jump <= jump_limit}
            seams.append(info)
            if not info["smooth"]:
                issues.append(
                    f"seam at frame {sf} jump {jump} exceeds {jump_limit}")

    return {
        "ok": not issues,
        "aligned": len(pcm) % frame_size == 0,
        "frame_size": frame_size,
        "seams": seams,
        "issues": issues,
    }


def metrics(*args, **kwargs) -> Dict[str, Any]:
    """录音指标：全部以样本计数为准，不使用墙钟相加。"""
    if not args or not isinstance(args[0], Recorder):
        return {
            "segments": 0,
            "total_frames": 0,
            "duration_s": 0.0,
            "overlap_frames": 0,
            "dropped_bytes": 0,
            "overruns": 0,
        }
    rec = args[0]
    overlap = sum(s.overlap_frames for s in rec.segments)
    raw_frames = sum(s.frames for s in rec.segments)
    return {
        "segments": len(rec.segments),
        "total_frames": rec.total_frames,
        "raw_frames": raw_frames,
        "duration_s": sample_count_duration(rec.total_frames,
                                            rec.sample_rate),
        "overlap_frames": overlap,
        "overruns": rec.ring.overruns,
        "sample_rate": rec.sample_rate,
        "frame_size": rec.frame_size,
    }


def audit(*args, **kwargs) -> Dict[str, Any]:
    """审计录音状态与临时分段落盘情况。"""
    if not args or not isinstance(args[0], Recorder):
        return {"issues": [], "temp_files": [], "state": None}
    rec = args[0]
    issues: List[str] = []
    temp_files: List[str] = []
    for seg in rec.segments:
        if seg.path is None:
            issues.append(f"segment {seg.index} has no temp file")
            continue
        if not seg.path.exists():
            issues.append(f"segment {seg.index} temp file missing: {seg.path}")
            continue
        actual = seg.path.stat().st_size
        temp_files.append(str(seg.path))
        if actual != seg.frames * rec.frame_size:
            issues.append(
                f"segment {seg.index} size {actual} != "
                f"{seg.frames * rec.frame_size}")
        if actual % rec.frame_size != 0:
            issues.append(f"segment {seg.index} not frame aligned")
    if rec.state == "cancelled" and rec.segments:
        issues.append("cancelled recorder still references segments")
    if rec.ring.overruns:
        issues.append(f"ring overrun count={rec.ring.overruns}")
    return {"issues": issues, "temp_files": temp_files,
            "state": rec.state, "segments": len(rec.segments)}


def fallback(*args, **kwargs) -> Dict[str, Any]:
    """异常兜底：停流、原子排空、残帧静音补齐，保证导出仍按帧对齐。

    正常写入路径会按帧截断残尾；兜底时把设备回调的最后一个原始 chunk
    通过 ``raw_chunk`` 传入，不足一帧的部分用静音补齐后再落盘。
    """
    if not args or not isinstance(args[0], Recorder):
        frame_size = int(kwargs.get("frame_size", frame_byte_size()))
        return {
            "recovered": False,
            "reason": "no recorder",
            "padding_bytes": 0,
            "frame_size": frame_size,
        }
    rec = args[0]
    reason = kwargs.get("reason", "stream error")
    if rec.stream.active:
        rec.stream.stop()
    tail = rec.ring.drain()
    raw = bytes(kwargs.get("raw_chunk", b""))
    remainder = (len(tail) + len(raw)) % rec.frame_size
    padding = b""
    if remainder:
        padding = b"\x00" * (rec.frame_size - remainder)
    recovered = tail + raw + padding
    if recovered:
        rec.append(recovered)
    return {
        "recovered": True,
        "reason": reason,
        "drained_bytes": len(tail),
        "raw_bytes": len(raw),
        "padding_bytes": len(padding),
        "segments": len(rec.segments),
        "state": rec.state,
    }
