"""
Unit Tests for VideoProcessor Module.

Validates webcam/file ingestion, metadata extraction, graceful error handling
for invalid sources, rate pacing, frame retrieval, and resource release.
"""

from pathlib import Path
import tempfile
from typing import Generator
import cv2
import numpy as np
import pytest

from modules.video_processor import VideoMetadata, VideoProcessor, VideoStatus


@pytest.fixture
def synthetic_video_file() -> Generator[Path, None, None]:
    """
    Creates a temporary valid .avi video file (10 frames, 320x240, 10 FPS)
    for reproducible offline video ingestion testing.
    """
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    width, height, fps = 320, 240, 10
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))

    try:
        for i in range(10):
            # Generate solid color frames with a moving bright rectangle
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add a moving box to simulate motion
            cv2.rectangle(frame, (20 + i * 5, 30), (70 + i * 5, 90), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        yield tmp_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_invalid_video_file_handling() -> None:
    """Test that non-existent video files are handled gracefully without exceptions."""
    non_existent = "/non_existent/path/to/lecture.mp4"
    vp = VideoProcessor(source=non_existent)
    
    success = vp.open_source()
    assert success is False
    assert vp.status == VideoStatus.ERROR
    assert vp.is_opened() is False
    assert vp.error_message is not None
    assert "not found" in vp.error_message.lower()


def test_unavailable_webcam_handling() -> None:
    """Test that unavailable camera hardware indexes fail gracefully with clear status."""
    # Index 9999 is guaranteed unavailable on typical test environments
    vp = VideoProcessor(source=9999)
    
    success = vp.open_source()
    assert success is False
    assert vp.status == VideoStatus.ERROR
    assert vp.is_opened() is False
    assert vp.error_message is not None


def test_synthetic_video_metadata_and_reading(synthetic_video_file: Path) -> None:
    """Test opening a valid video file, inspecting metadata, and reading all frames to EOF."""
    vp = VideoProcessor(source=synthetic_video_file, target_fps=10)
    
    opened = vp.open_source()
    assert opened is True
    assert vp.is_opened() is True
    assert vp.status == VideoStatus.OPENED
    
    # Verify metadata
    meta = vp.metadata
    assert meta.width == 320
    assert meta.height == 240
    assert meta.resolution_str == "320x240"
    assert meta.total_frames == 10
    assert meta.is_live is False
    assert meta.duration_seconds > 0.0

    # Read and process all frames
    frames_read = 0
    while True:
        success, frame = vp.get_next_frame(apply_pacing=False)
        if not success:
            break
        assert frame is not None
        assert frame.shape == (240, 320, 3)
        frames_read += 1

        # Process frame
        annotated, tracks, motion = vp.process_frame(frame)
        assert annotated.shape == (240, 320, 3)
        assert isinstance(motion, float)

    assert frames_read == 10
    assert vp.status == VideoStatus.COMPLETED

    # Clean release
    vp.release()
    assert vp.is_opened() is False


def test_context_manager_cleanup(synthetic_video_file: Path) -> None:
    """Test that VideoProcessor works as a context manager and releases resources on exit."""
    with VideoProcessor(source=synthetic_video_file) as vp:
        assert vp.is_opened() is True
        frame = vp.read_next_frame()
        assert frame is not None

    # After exiting with block, capture must be released
    assert vp.is_opened() is False


def test_privacy_frame_saving_guardrail() -> None:
    """Test that saving frames to disk is prevented by default to preserve privacy."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    vp = VideoProcessor(allow_save_frames=False)

    with pytest.raises(PermissionError) as exc_info:
        vp.save_frame_diagnostic(dummy_frame, "/tmp/should_fail.jpg")
    assert "Privacy Policy Violation" in str(exc_info.value)
