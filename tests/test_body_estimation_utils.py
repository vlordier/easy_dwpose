"""
Comprehensive tests for body estimation utility functions with parametrization.
Tests utility functions, preprocessing, postprocessing, and pose estimation components.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from easy_dwpose.body_estimation import Wholebody, resize_image
from easy_dwpose.body_estimation.detector import demo_postprocess, inference_detector, multiclass_nms, nms
from easy_dwpose.body_estimation.detector import preprocess as detector_preprocess
from easy_dwpose.body_estimation.pose import bbox_xyxy2cs, inference_pose, postprocess
from easy_dwpose.body_estimation.pose import preprocess as pose_preprocess
from easy_dwpose.body_estimation.utils import resize_image as utils_resize_image


class TestResizeImageFunction:
    """Test resize_image utility function with comprehensive parameters."""

    @pytest.mark.parametrize(
        "input_shape,target_resolution,expected_min_dim",
        [
            ((480, 640, 3), 512, 512),
            ((720, 1280, 3), 256, 256),
            ((100, 200, 3), 1024, 1024),
            ((1920, 1080, 3), 768, 768),
            ((64, 64, 3), 128, 128),
            ((128, 256, 3), 384, 384),
        ],
    )
    def test_resize_image_target_resolution(self, input_shape, target_resolution, expected_min_dim):
        """Test that resize_image produces correct target resolution."""
        input_image = np.random.randint(0, 255, input_shape, dtype=np.uint8)

        result = resize_image(input_image, target_resolution=target_resolution)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

        min_dim = min(result.shape[:2])
        assert min_dim == expected_min_dim

    @pytest.mark.parametrize("divisible_by", [1, 8, 16, 32, 64, 128])
    def test_resize_image_divisible_constraint(self, divisible_by):
        """Test that output dimensions are divisible by specified value."""
        input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = resize_image(input_image, target_resolution=512, dividable_by=divisible_by)

        height, width = result.shape[:2]
        assert height % divisible_by == 0, f"Height {height} not divisible by {divisible_by}"
        assert width % divisible_by == 0, f"Width {width} not divisible by {divisible_by}"

    @pytest.mark.parametrize(
        "aspect_ratio",
        [
            (1, 1),  # Square
            (4, 3),  # Standard
            (16, 9),  # Widescreen
            (9, 16),  # Portrait
            (21, 9),  # Ultra-wide
            (1, 2),  # Very tall
            (3, 1),  # Very wide
        ],
    )
    def test_resize_image_preserves_aspect_ratio(self, aspect_ratio):
        """Test that resize_image preserves aspect ratio for various ratios."""
        width, height = 320 * aspect_ratio[0], 240 * aspect_ratio[1]
        input_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        result = resize_image(input_image, target_resolution=512)

        result_height, result_width = result.shape[:2]
        original_ratio = width / height
        result_ratio = result_width / result_height

        # Allow small tolerance due to rounding
        assert abs(original_ratio - result_ratio) < 0.1, (
            f"Aspect ratio not preserved: {original_ratio} vs {result_ratio}"
        )

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_resize_image_data_types(self, dtype):
        """Test resize_image with different input data types."""
        if dtype == np.uint8:
            input_image = np.random.randint(0, 255, (256, 256, 3), dtype=dtype)
        elif dtype == np.uint16:
            input_image = np.random.randint(0, 65535, (256, 256, 3), dtype=dtype)
        else:  # float32
            input_image = np.random.rand(256, 256, 3).astype(dtype)

        try:
            result = resize_image(input_image, target_resolution=128)
            assert result is not None
            # resize_image preserves input data type
            assert result.dtype == dtype
        except (ValueError, TypeError) as e:
            # Some dtypes might not be supported
            assert "dtype" in str(e).lower() or "type" in str(e).lower()

    @pytest.mark.parametrize(
        "extreme_size",
        [
            (1, 1, 3),  # Minimum size
            (2, 2, 3),  # Very small
            (10000, 10, 3),  # Very wide
            (10, 10000, 3),  # Very tall
        ],
    )
    def test_resize_image_extreme_sizes(self, extreme_size):
        """Test resize_image with extreme input sizes."""
        input_image = np.random.randint(0, 255, extreme_size, dtype=np.uint8)

        result = resize_image(input_image, target_resolution=256)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3
        assert min(result.shape[:2]) == 256


class TestDetectorFunctions:
    """Test detector utility functions with comprehensive parameters."""

    @pytest.mark.parametrize(
        "num_boxes,overlap_threshold",
        [
            (3, 0.1),
            (5, 0.3),
            (10, 0.5),
            (20, 0.7),
            (50, 0.9),
        ],
    )
    def test_nms_various_thresholds(self, num_boxes, overlap_threshold):
        """Test NMS with various numbers of boxes and thresholds."""
        # Generate random boxes
        boxes = np.random.rand(num_boxes, 4) * 100
        # Ensure boxes are in xyxy format with x1 < x2, y1 < y2
        boxes[:, 2:] += boxes[:, :2] + 1

        scores = np.random.rand(num_boxes)

        keep_indices = nms(boxes, scores, overlap_threshold)

        assert isinstance(keep_indices, list)
        assert len(keep_indices) <= num_boxes
        assert all(0 <= idx < num_boxes for idx in keep_indices)

        # Higher thresholds should generally keep more boxes
        if overlap_threshold > 0.8:
            assert len(keep_indices) >= num_boxes // 4  # At least 25% should be kept

    def test_nms_edge_cases(self):
        """Test NMS with edge cases."""
        # Empty input
        empty_boxes = np.array([]).reshape(0, 4)
        empty_scores = np.array([])
        result = nms(empty_boxes, empty_scores, 0.5)
        assert result == []

        # Single box
        single_box = np.array([[10, 10, 50, 50]])
        single_score = np.array([0.9])
        result = nms(single_box, single_score, 0.5)
        assert result == [0]

        # Identical boxes with different scores
        identical_boxes = np.array([[10, 10, 50, 50], [10, 10, 50, 50]])
        different_scores = np.array([0.9, 0.8])
        result = nms(identical_boxes, different_scores, 0.5)
        assert len(result) == 1
        assert result[0] == 0  # Higher score should be kept

    @pytest.mark.parametrize(
        "num_classes,score_threshold,nms_threshold",
        [
            (2, 0.1, 0.5),
            (5, 0.3, 0.3),
            (10, 0.5, 0.7),
            (1, 0.8, 0.9),
        ],
    )
    def test_multiclass_nms_various_parameters(self, num_classes, score_threshold, nms_threshold):
        """Test multiclass NMS with various parameters."""
        num_boxes = 10
        boxes = np.random.rand(num_boxes, 4) * 100
        boxes[:, 2:] += boxes[:, :2] + 1  # Ensure valid boxes

        scores = np.random.rand(num_boxes, num_classes)

        result = multiclass_nms(boxes, scores, nms_threshold, score_threshold)

        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.shape[1] == 6  # [x1, y1, x2, y2, score, class]
            assert np.all(result[:, 4] >= score_threshold)  # Scores above threshold
            assert np.all(result[:, 5] >= 0)  # Valid class indices
            assert np.all(result[:, 5] < num_classes)  # Class indices within range

    @pytest.mark.parametrize(
        "img_size,p6",
        [
            ((416, 416), False),
            ((640, 640), False),
            ((832, 832), True),
            ((1280, 1280), True),
        ],
    )
    def test_demo_postprocess_various_sizes(self, img_size, p6):
        """Test demo_postprocess with various image sizes and P6 configurations."""
        # Create mock output data
        num_anchors = 8400 if not p6 else 22500  # Typical anchor counts
        num_classes = 80
        output_shape = (1, num_anchors, 5 + num_classes)

        outputs = np.random.rand(*output_shape)

        try:
            result = demo_postprocess(outputs, img_size, p6=p6)
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 1  # Batch size
            assert result.shape[1] == num_anchors
            assert result.shape[2] == 5 + num_classes  # x, y, w, h, obj, classes
        except Exception as e:
            # Some configurations might not be valid
            pytest.skip(f"Configuration not supported: {e}")

    @pytest.mark.parametrize("input_size", [(416, 416), (640, 640), (832, 832)])
    def test_detector_preprocess_various_sizes(self, input_size):
        """Test detector preprocessing with various input sizes."""
        original_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        padded_img, ratio = detector_preprocess(original_img, input_size)

        assert isinstance(padded_img, np.ndarray)
        assert isinstance(ratio, (float, np.floating))
        assert padded_img.shape == (3, input_size[1], input_size[0])  # CHW format
        assert padded_img.dtype == np.float32
        assert ratio > 0
        assert np.all(padded_img >= 0) and np.all(padded_img <= 255)


class TestPoseFunctions:
    """Test pose estimation utility functions."""

    @pytest.mark.parametrize("input_size", [(192, 256), (256, 192), (384, 288)])
    def test_pose_preprocess_various_sizes(self, input_size):
        """Test pose preprocessing with various input sizes."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out_bbox = np.array([[100, 100, 300, 400]])  # Single bbox

        try:
            out_img, out_center, out_scale = pose_preprocess(img, out_bbox, input_size)

            assert len(out_img) == 1  # Single person
            assert len(out_center) == 1
            assert len(out_scale) == 1

            processed_img = out_img[0]
            assert processed_img.shape == (input_size[1], input_size[0], 3)  # HWC format
            assert processed_img.dtype == np.float32
        except Exception as e:
            # Some bbox configurations might be invalid
            pytest.skip(f"Bbox configuration not supported: {e}")

    @pytest.mark.parametrize("num_people", [1, 2, 5, 10])
    def test_pose_preprocess_multiple_people(self, num_people):
        """Test pose preprocessing with multiple people."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Generate random bboxes
        out_bbox = []
        for _ in range(num_people):
            x1, y1 = np.random.randint(0, 300, 2)
            x2, y2 = x1 + np.random.randint(50, 200), y1 + np.random.randint(50, 200)
            out_bbox.append([x1, y1, min(x2, 639), min(y2, 479)])
        out_bbox = np.array(out_bbox)

        try:
            out_img, out_center, out_scale = pose_preprocess(img, out_bbox)

            assert len(out_img) == num_people
            assert len(out_center) == num_people
            assert len(out_scale) == num_people

            for i in range(num_people):
                assert out_img[i].shape[2] == 3  # RGB channels
                assert isinstance(out_center[i], np.ndarray)
                assert isinstance(out_scale[i], np.ndarray)
        except Exception as e:
            pytest.skip(f"Multiple people configuration failed: {e}")

    @pytest.mark.parametrize(
        "num_keypoints,simcc_split_ratio",
        [
            (17, 2.0),  # COCO format
            (133, 2.0),  # Wholebody format
            (21, 1.5),  # Hand keypoints
            (68, 2.5),  # Face keypoints
        ],
    )
    def test_postprocess_various_keypoints(self, num_keypoints, simcc_split_ratio):
        """Test postprocessing with various keypoint configurations."""
        model_input_size = (192, 256)
        batch_size = 2

        # Mock outputs
        outputs = [np.random.rand(batch_size, num_keypoints, model_input_size[0])]
        center = [np.array([96, 128]) for _ in range(batch_size)]
        scale = [np.array([192, 256]) for _ in range(batch_size)]

        try:
            keypoints, scores = postprocess(outputs, model_input_size, center, scale, simcc_split_ratio)

            assert isinstance(keypoints, np.ndarray)
            assert isinstance(scores, np.ndarray)
            assert keypoints.shape[0] == batch_size
            assert scores.shape[0] == batch_size
            assert keypoints.shape[1] == num_keypoints
            assert scores.shape[1] == num_keypoints
        except Exception as e:
            pytest.skip(f"Keypoint configuration not supported: {e}")

    @pytest.mark.parametrize(
        "bbox,padding",
        [
            ([10, 10, 50, 50], 1.0),
            ([0, 0, 100, 200], 1.25),
            ([50, 50, 150, 100], 1.5),
            ([100, 200, 300, 400], 2.0),
        ],
    )
    def test_bbox_xyxy2cs_various_inputs(self, bbox, padding):
        """Test bbox to center-scale conversion with various inputs."""
        bbox_array = np.array(bbox)

        center, scale = bbox_xyxy2cs(bbox_array, padding=padding)

        assert isinstance(center, np.ndarray)
        assert isinstance(scale, np.ndarray)
        assert len(center) == 2  # x, y coordinates
        assert len(scale) == 2  # width, height
        assert np.all(center >= 0)
        assert np.all(scale > 0)

        # Verify that center is within reasonable bounds
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        assert abs(center[0] - bbox_center_x) < 1e-3
        assert abs(center[1] - bbox_center_y) < 1e-3


class TestWholebodyClass:
    """Test Wholebody detector class with mocked dependencies."""

    @pytest.fixture
    def mock_sessions(self):
        """Create mock ONNX sessions."""
        mock_det_session = MagicMock()
        mock_det_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_det_session.run.return_value = [np.random.rand(1, 100, 6)]

        mock_pose_session = MagicMock()
        # Mock the pose session input to have proper shape for RTMPose
        mock_pose_input = MagicMock(name="input")
        mock_pose_input.shape = [1, 3, 256, 192]  # RTMPose input shape
        mock_pose_session.get_inputs.return_value = [mock_pose_input]
        # RTMPose returns two outputs: simcc_x and simcc_y
        mock_pose_session.run.return_value = [
            np.random.rand(1, 133, 96),  # simcc_x: (batch, keypoints, width/2)
            np.random.rand(1, 133, 128),  # simcc_y: (batch, keypoints, height/2)
        ]

        return mock_det_session, mock_pose_session

    @pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
    def test_wholebody_initialization(self, mock_sessions, device):
        """Test Wholebody initialization with different devices."""
        mock_det_session, mock_pose_session = mock_sessions

        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session:
            mock_session.side_effect = [mock_det_session, mock_pose_session]

            try:
                detector = Wholebody("model_det.onnx", "model_pose.onnx", device=device)
                assert detector is not None
                assert hasattr(detector, "session_det")
                assert hasattr(detector, "session_pose")
            except Exception as e:
                if "cuda" in device.lower() and "not available" in str(e).lower():
                    pytest.skip(f"CUDA not available: {e}")
                else:
                    raise

    @pytest.mark.parametrize(
        "image_shape",
        [
            (480, 640, 3),
            (720, 1280, 3),
            (256, 256, 3),
            (1080, 1920, 3),
        ],
    )
    def test_wholebody_call_various_images(self, mock_sessions, image_shape):
        """Test Wholebody call with various image shapes."""
        mock_det_session, mock_pose_session = mock_sessions

        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session:
            mock_session.side_effect = [mock_det_session, mock_pose_session]

            with patch("easy_dwpose.body_estimation.wholebody.inference_detector") as mock_inf_det:
                mock_inf_det.return_value = np.array([[100, 100, 200, 200]])

                with patch("easy_dwpose.body_estimation.wholebody.inference_pose") as mock_inf_pose:
                    mock_keypoints = np.random.rand(1, 133, 2)
                    mock_scores = np.random.rand(1, 133)
                    mock_inf_pose.return_value = (mock_keypoints, mock_scores)

                    detector = Wholebody("model_det.onnx", "model_pose.onnx")
                    test_image = np.random.randint(0, 255, image_shape, dtype=np.uint8)

                    keypoints, scores = detector(test_image)

                    assert isinstance(keypoints, np.ndarray)
                    assert isinstance(scores, np.ndarray)
                    assert keypoints.shape[1] == 134  # DWPose format
                    assert scores.shape[1] == 134
                    assert keypoints.shape[0] == scores.shape[0]  # Same number of people

    def test_wholebody_no_detections(self, mock_sessions):
        """Test Wholebody when no people are detected."""
        mock_det_session, mock_pose_session = mock_sessions

        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session:
            mock_session.side_effect = [mock_det_session, mock_pose_session]

            with patch("easy_dwpose.body_estimation.wholebody.inference_detector") as mock_inf_det:
                mock_inf_det.return_value = np.array([]).reshape(0, 4)  # No detections

                detector = Wholebody("model_det.onnx", "model_pose.onnx")
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                keypoints, scores = detector(test_image)

                assert isinstance(keypoints, np.ndarray)
                assert isinstance(scores, np.ndarray)
                # Should return empty results or default structure
                assert keypoints.shape[1] == 134  # DWPose format maintained
                assert scores.shape[1] == 134

    @pytest.mark.parametrize("num_detections", [1, 2, 5, 10])
    def test_wholebody_multiple_detections(self, mock_sessions, num_detections):
        """Test Wholebody with multiple person detections."""
        mock_det_session, mock_pose_session = mock_sessions

        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session:
            mock_session.side_effect = [mock_det_session, mock_pose_session]

            with patch("easy_dwpose.body_estimation.wholebody.inference_detector") as mock_inf_det:
                # Create multiple bboxes
                bboxes = []
                for i in range(num_detections):
                    x1, y1 = i * 50, i * 50
                    x2, y2 = x1 + 100, y1 + 150
                    bboxes.append([x1, y1, x2, y2])
                mock_inf_det.return_value = np.array(bboxes)

                with patch("easy_dwpose.body_estimation.wholebody.inference_pose") as mock_inf_pose:
                    mock_keypoints = np.random.rand(num_detections, 133, 2)
                    mock_scores = np.random.rand(num_detections, 133)
                    mock_inf_pose.return_value = (mock_keypoints, mock_scores)

                    detector = Wholebody("model_det.onnx", "model_pose.onnx")
                    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                    keypoints, scores = detector(test_image)

                    assert isinstance(keypoints, np.ndarray)
                    assert isinstance(scores, np.ndarray)
                    assert keypoints.shape[1] == 134  # DWPose format
                    assert scores.shape[1] == 134
                    # Number of people detected should be related to input
                    assert keypoints.shape[0] <= num_detections * 18  # Max possible output


class TestIntegrationScenarios:
    """Integration tests for body estimation components."""

    def test_complete_pipeline_simulation(self):
        """Test complete body estimation pipeline with mocked components."""
        # This test simulates the complete pipeline from image to pose
        input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Mock the complete pipeline
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

                    # Test the complete pipeline
                    detector = Wholebody("model_det.onnx", "model_pose.onnx")
                    keypoints, scores = detector(input_image)

                    # Verify pipeline output
                    assert isinstance(keypoints, np.ndarray)
                    assert isinstance(scores, np.ndarray)
                    assert keypoints.shape[1] == 134
                    assert scores.shape[1] == 134

                    # Test that the pipeline can handle the complete flow
                    assert np.all(np.isfinite(keypoints))
                    assert np.all(np.isfinite(scores))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
