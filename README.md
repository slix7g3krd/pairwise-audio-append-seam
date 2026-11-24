# pairwise-audio-append-seam

## 领域

游戏、图形与多媒体

## 主题

分段录音续录拼接

## 说明

采样帧对齐、原子读写指针与拼缝交叉淡化。

初始状态下 `media/audio_append_seam.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
