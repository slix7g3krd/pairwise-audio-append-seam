"""六个修复点的行为验证（契约测试之外的端到端行为）。"""

import threading
from pathlib import Path

import pytest

import media.audio_append_seam as mod
from media.audio_append_seam import (
    AtomicRingBuffer,
    AudioStream,
    Recorder,
    frame_aligned_append,
    frame_byte_size,
    sample_count_duration,
    seam_crossfade,
)

FS = frame_byte_size()  # int16 mono -> 2 字节/帧
SR = 8000


def pcm(frames, start=0):
    return b"".join(int(start + i).to_bytes(2, "little", signed=True)
                    for i in range(frames))


# 修复点 1：续录按采样帧对齐 ---------------------------------------------

def test_frame_aligned_append_offsets_snap_to_frame():
    assert frame_aligned_append(0, FS) == 0
    assert frame_aligned_append(FS + 1, FS) == FS          # 残尾向下取整
    assert frame_aligned_append(3 * FS - 1, FS) == 2 * FS


def test_frame_aligned_append_drops_partial_tail():
    merged, dropped, frame_size = frame_aligned_append(
        pcm(10), pcm(5) + b"\x01", frame_size=FS)
    assert dropped == 1
    assert len(merged) == 15 * FS
    assert frame_size == FS


def test_wall_clock_resume_is_frame_aligned_not_millisecond_rounded():
    # 墙钟 1.0001ms 在 8kHz 下不是整数帧，必须收敛到帧边界
    aligned = frame_aligned_append(0, SR, 1.0001)
    assert aligned % FS == 0
    assert aligned == int(round(1.0001 / 1000 * SR)) * FS


# 修复点 2：读写指针原子交换 ---------------------------------------------

def test_ring_snapshot_never_returns_half_frame():
    ring = AtomicRingBuffer(1024, FS)
    errors = []

    def reader():
        for _ in range(2000):
            snap = ring.snapshot()
            if len(snap["data"]) % FS != 0 or snap["size"] % FS != 0:
                errors.append("half frame observed")

    t = threading.Thread(target=reader)
    t.start()
    for i in range(2000):
        ring.write(pcm(3, start=i))
    t.join()
    assert not errors


def test_ring_swap_is_atomic_and_returns_old_content():
    ring = AtomicRingBuffer(1024, FS)
    ring.write(pcm(10))
    old = ring.swap(pcm(4))
    assert len(old) == 10 * FS
    assert len(ring) == 4 * FS
    assert ring.drain() == pcm(4)


# 修复点 3：暂停停流并留存尾部，回调不覆盖 -------------------------------

def test_pause_stops_stream_and_preserves_tail():
    rec = Recorder(sample_rate=SR)
    rec.start()
    rec.write(pcm(100))
    info = rec.pause()
    assert rec.state == "paused"
    assert info["tail_frames"] == 100
    # 暂停期间设备仍在回调，但数据写不进来，尾部不被覆盖
    for i in range(50):
        assert rec.stream.callback(pcm(7, start=900 + i)) == 0
    assert rec.ring.overruns == 0
    snap = rec.ring.snapshot()
    assert snap["frames"] == 100
    assert snap["data"] == pcm(100)
    rec.resume()
    rec.write(pcm(20))
    assert rec.ring.snapshot()["frames"] == 120  # 尾部 + 续录自然衔接


# 修复点 4：时长按样本计数累计，扣除重叠 ---------------------------------

def test_duration_uses_sample_count_and_deducts_overlap():
    rec = Recorder(sample_rate=SR)
    rec.start()
    rec.write(pcm(1000))
    seg1 = rec.append()
    rec.write(pcm(1000))
    seg2 = rec.append(overlap_frames=100)  # 与第一段重叠 100 帧
    assert seg1.frames == 1000 and seg2.frames == 1000
    assert rec.total_frames == 1900
    assert mod.sample_count_duration(rec.total_frames, SR) == \
        pytest.approx(1900 / SR)
    # 简单相加的错误算法会得到 2000/SR
    assert mod.sample_count_duration([1000, 1000 - 100], SR) == \
        pytest.approx(1900 / SR)


# 修复点 5：取消清理临时分段 ---------------------------------------------

def test_cancel_deletes_temp_segments(tmp_path):
    rec = Recorder(sample_rate=SR, tmp_dir=tmp_path)
    rec.start()
    rec.write(pcm(50))
    rec.append()
    rec.write(pcm(30))
    rec.append()
    temp_files = list(tmp_path.glob("segment_*.pcm"))
    assert len(temp_files) == 2 and all(p.exists() for p in temp_files)
    report = rec.cancel()
    assert rec.state == "cancelled"
    assert report["count"] == 2
    assert not list(tmp_path.glob("segment_*.pcm"))


# 修复点 6：拼缝交叉淡化 -------------------------------------------------

def test_crossfade_blends_overlap_and_deducts_it():
    left, right = pcm(100), pcm(100, start=1000)
    joined = seam_crossfade(left, right, overlap_frames=20)
    # 输出帧数 = L + R - overlap
    assert len(joined) == 180 * FS
    # 淡化区两端与原始信号一致（权重 1.0 / 0.0）
    samples = mod._pcm_samples(joined)
    assert samples[0] == 0
    assert samples[-1] == 1000 + 99


def test_crossfade_smooths_seam_jump():
    # 无交叉淡化时拼缝处有巨大跳变（重复/空洞的听感特征）
    left = b"".join((10000).to_bytes(2, "little", signed=True)
                    for _ in range(50))
    right = b"".join((-10000).to_bytes(2, "little", signed=True)
                     for _ in range(50))
    hard = left + right
    hard_jump = abs(mod._pcm_samples(hard)[50]
                    - mod._pcm_samples(hard)[49])
    faded = seam_crossfade(hard, seam_frame=50, fade_frames=20)
    faded_samples = mod._pcm_samples(faded)
    faded_jump = abs(faded_samples[50] - faded_samples[49])
    assert faded_jump < hard_jump


# 端到端：暂停-续录-导出全流程 -------------------------------------------

def test_end_to_end_pause_resume_export(tmp_path):
    rec = Recorder(sample_rate=SR, crossfade_ms=5.0)
    rec.start()
    rec.write(pcm(500))
    rec.pause()
    rec.stream.callback(pcm(10, start=9999))  # 暂停期回调被丢弃
    rec.resume()
    rec.write(pcm(500))
    rec.append()
    out = rec.export(str(tmp_path / "out.wav"))
    assert out["frames"] == 1000
    assert out["duration_s"] == pytest.approx(1000 / SR)
    assert Path(out["path"]).exists()
    report = mod.validate_seam(rec)
    assert report["aligned"]
    audit = mod.audit(rec)
    assert audit["state"] == "stopped"
    # 兜底：残帧静音补齐
    fb = mod.fallback(Recorder(sample_rate=SR, tmp_dir=tmp_path / "fb"),
                      reason="test")
    assert fb["recovered"] is True
