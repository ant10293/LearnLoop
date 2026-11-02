"""OpenCV's bundled libavdevice must not reuse PyAV's AVFoundation class names."""

import pytest
from app.native_compat import renamed_avfoundation_classes


def test_class_names_are_renamed_without_shifting_the_image():
    original = b"xxAVFFrameReceiver\x00AVFAudioReceiver\x00yy"
    updated = renamed_avfoundation_classes(original)
    assert updated is not None
    assert len(updated) == len(original)
    assert updated == b"xxOcvFrameReceiver\x00OcvAudioReceiver\x00yy"
    assert renamed_avfoundation_classes(updated) is None


def test_ambiguous_class_name_is_left_alone():
    packed = b"AVFFrameReceiver\x00AVFFrameReceiver\x00"
    with pytest.raises(ValueError, match="AVFFrameReceiver"):
        renamed_avfoundation_classes(packed)
