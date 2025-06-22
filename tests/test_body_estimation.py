from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from easy_dwpose.body_estimation import Wholebody, resize_image
from easy_dwpose.body_estimation.detector import multiclass_nms, nms, preprocess


class TestResizeImage:
    """Test the resize_image utility function."""

    @pytest.mark.parametrize(
        "input_size,target_resolution",
        [
            ((480, 640, 3), 512),
            ((720, 1280, 3), 256),
            ((100, 200, 3), 1024),
        ],
    )
    def test_resize_image_basic(self, input_size, target_resolution):
        """Test basic resize_image functionality."""
        input_image = np.random.randint(0, 255, input_size, dtype=np.uint8)

        result = resize_image(input_image, target_resolution=target_resolution)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # RGB channels

        # Check that the minimum dimension matches target resolution
        min_dim = min(result.shape[:2])
        assert min_dim == target_resolution

    def test_resize_image_preserves_aspect_ratio(self):
        """Test that resize_image preserves aspect ratio."""
        input_image = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)  # 1:2 ratio

        result = resize_image(input_image, target_resolution=512)

        height, width = result.shape[:2]
        original_ratio = 600 / 300  # width / height
        result_ratio = width / height

        # Allow small tolerance due to rounding
        assert abs(original_ratio - result_ratio) < 0.1


class TestDetectorFunctions:
    """Test individual detector functions."""

    def test_nms_basic(self):
        """Test basic NMS functionality."""
        boxes = np.array(
            [
                [10, 10, 50, 50],  # Box 1
                [15, 15, 55, 55],  # Box 2 (overlapping with Box 1)
                [100, 100, 150, 150],  # Box 3 (separate)
            ],
            dtype=np.float32,
        )

        scores = np.array([0.9, 0.8, 0.95])
        nms_thr = 0.5

        keep_indices = nms(boxes, scores, nms_thr)

        assert isinstance(keep_indices, list)
        assert len(keep_indices) > 0
        assert len(keep_indices) <= len(boxes)

    def test_multiclass_nms_basic(self):
        """Test basic multiclass NMS functionality."""
        boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32)

        scores = np.array(
            [
                [0.9, 0.1],  # Box 1: high score for class 0
                [0.1, 0.95],  # Box 2: high score for class 1
            ]
        )

        result = multiclass_nms(boxes, scores, nms_thr=0.5, score_thr=0.5)

        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.shape[1] == 6  # [x1, y1, x2, y2, score, class]

    def test_preprocess_basic(self):
        """Test basic preprocessing functionality."""
        input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        input_size = (640, 640)

        padded_img, ratio = preprocess(input_image, input_size)

        assert isinstance(padded_img, np.ndarray)
        assert isinstance(ratio, float)
        assert padded_img.shape == (3, 640, 640)  # Should be CHW format
        assert padded_img.dtype == np.float32
        assert ratio > 0


class TestWholebodyClass:
    """Test the Wholebody detector class."""

    def test_wholebody_initialization_cpu(self):
        """Test Wholebody initialization with CPU device."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session:
            mock_session.return_value = MagicMock()

            detector = Wholebody("model_det.onnx", "model_pose.onnx", device="cpu")

            assert detector is not None
            assert hasattr(detector, "session_det")
            assert hasattr(detector, "session_pose")

    def test_wholebody_call_basic(self):
        """Test basic Wholebody call functionality."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session:
            mock_det_session = MagicMock()
            mock_det_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_det_session.run.return_value = [np.random.rand(1, 100, 6)]

            mock_pose_session = MagicMock()
            mock_pose_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_pose_session.run.return_value = [np.random.rand(1, 384, 192)]

            mock_session.side_effect = [mock_det_session, mock_pose_session]

            with patch("easy_dwpose.body_estimation.wholebody.inference_detector") as mock_inf_det:
                mock_inf_det.return_value = np.array([[100, 100, 200, 200]])

                with patch("easy_dwpose.body_estimation.wholebody.inference_pose") as mock_inf_pose:
                    mock_keypoints = np.random.rand(1, 133, 2)
                    mock_scores = np.random.rand(1, 133)
                    mock_inf_pose.return_value = (mock_keypoints, mock_scores)

                    detector = Wholebody("model_det.onnx", "model_pose.onnx")
                    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                    keypoints, scores = detector(test_image)

                    assert isinstance(keypoints, np.ndarray)
                    assert isinstance(scores, np.ndarray)
                    assert keypoints.shape[1] == 134  # DWPose format (full keypoint set)
                    assert scores.shape[1] == 134
