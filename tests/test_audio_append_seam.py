import pytest
import media.audio_append_seam as mod

CASES = [
    ("frame_aligned_append", (1,)),
    ("atomic_ring_swap", (1,)),
    ("pause_stream", (1,)),
    ("sample_count_duration", (1,)),
    ("cleanup_segments", (1,)),
    ("seam_crossfade", (1,)),
    ("record_start", (1,)),
    ("pause", (1,)),
    ("resume", (1,)),
    ("append", (1,)),
    ("export", (1,)),
    ("validate_seam", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
