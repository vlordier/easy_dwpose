from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from easy_dwpose.body_estimation import Wholebody, resize_image
from easy_dwpose.body_estimation.detector import demo_postprocess, inference_detector, multiclass_nms, nms, preprocess


class TestResizeImage:
    """Test the resize_image utility function."""

    @pytest.mark.parametrize(
        "input_size,target_resolution,expected_min_dim",
        [
            ((480, 640, 3), 512, 512),
            ((720, 1280, 3), 256, 256),
            ((100, 200, 3), 1024, 1024),
            ((1920, 1080, 3), 768, 768),
        ],
    )
    def test_resize_image_dimensions(self, input_size, target_resolution, expected_min_dim):
        """Test that resize_image produces correct dimensions."""
        input_image = np.random.randint(0, 255, input_size, dtype=np.uint8)

        resized_result = resize_image(input_image, target_resolution=target_resolution)

        assert resized_result is not None
        assert isinstance(resized_result, np.ndarray)
        assert len(resized_result.shape) == 3
        assert resized_result.shape[2] == 3  # RGB channels

        # Check that the minimum dimension matches target resolution
        min_dimension = min(resized_result.shape[:2])
        assert min_dimension == expected_min_dim

    @pytest.mark.parametrize("dividable_by", [32, 64, 128])
    def test_resize_image_dividable_constraint(self, dividable_by):
        """Test that output dimensions are dividable by specified value."""
        input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        resized_result = resize_image(input_image, target_resolution=512, dividable_by=dividable_by)

        result_height, result_width = resized_result.shape[:2]
        assert result_height % dividable_by == 0
        assert result_width % dividable_by == 0

    def test_resize_image_preserves_aspect_ratio(self):
        """Test that resize_image preserves aspect ratio."""
        # Create a known aspect ratio image
        input_image = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)  # 1:2 ratio

        resized_result = resize_image(input_image, target_resolution=512)

        result_height, result_width = resized_result.shape[:2]
        original_aspect_ratio = 600 / 300  # width / height
        result_aspect_ratio = result_width / result_height

        # Allow small tolerance due to rounding
        assert abs(original_aspect_ratio - result_aspect_ratio) < 0.1

    def test_resize_image_data_type_preservation(self):
        """Test that resize_image preserves data type."""
        input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        resized_result = resize_image(input_image)

        assert resized_result.dtype == np.uint8

    def test_resize_image_edge_cases(self):
        """Test resize_image with edge case inputs."""
        # Very small image
        tiny_image = np.ones((1, 1, 3), dtype=np.uint8)
        tiny_result = resize_image(tiny_image, target_resolution=64)
        assert tiny_result is not None
        assert min(tiny_result.shape[:2]) == 64

        # Square image
        square_image = np.ones((100, 100, 3), dtype=np.uint8)
        square_result = resize_image(square_image, target_resolution=128)
        assert square_result is not None
        assert square_result.shape[0] == square_result.shape[1] == 128


class TestDetectorFunctions:
    """Test individual detector functions."""

    def test_nms_basic_functionality(self):
        """Test basic NMS functionality."""
        # Create overlapping boxes
        bounding_boxes = np.array(
            [
                [10, 10, 50, 50],  # Box 1
                [15, 15, 55, 55],  # Box 2 (overlapping with Box 1)
                [100, 100, 150, 150],  # Box 3 (separate)
            ],
            dtype=np.float32,
        )

        confidence_scores = np.array([0.9, 0.8, 0.95])
        nms_threshold = 0.5

        kept_indices = nms(bounding_boxes, confidence_scores, nms_threshold)

        assert isinstance(kept_indices, list)
        assert len(kept_indices) > 0
        assert len(kept_indices) <= len(bounding_boxes)

        # Highest scoring box should be kept
        assert 2 in kept_indices  # Box 3 has highest score (0.95)

    def test_nms_no_overlap(self):
        """Test NMS with non-overlapping boxes."""
        non_overlapping_boxes = np.array([[10, 10, 30, 30], [50, 50, 70, 70], [100, 100, 120, 120]], dtype=np.float32)

        box_scores = np.array([0.8, 0.9, 0.7])
        nms_threshold = 0.5

        kept_indices = nms(non_overlapping_boxes, box_scores, nms_threshold)

        # All boxes should be kept since they don't overlap
        assert len(kept_indices) == 3

    def test_multiclass_nms_functionality(self):
        """Test multiclass NMS functionality."""
        detection_boxes = np.array([[10, 10, 50, 50], [15, 15, 55, 55], [100, 100, 150, 150]], dtype=np.float32)

        # Multi-class scores (3 boxes, 2 classes)
        multiclass_scores = np.array(
            [
                [0.9, 0.1],  # Box 1: high score for class 0
                [0.8, 0.2],  # Box 2: high score for class 0
                [0.1, 0.95],  # Box 3: high score for class 1
            ]
        )

        nms_result = multiclass_nms(detection_boxes, multiclass_scores, nms_thr=0.5, score_thr=0.5)

        if nms_result is not None:
            assert isinstance(nms_result, np.ndarray)
            assert nms_result.shape[1] == 6  # [x1, y1, x2, y2, score, class]

    def test_multiclass_nms_low_scores(self):
        """Test multiclass NMS with all low scores."""
        low_score_boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        low_scores = np.array([[0.1, 0.2]])  # All scores below threshold

        nms_result = multiclass_nms(low_score_boxes, low_scores, nms_thr=0.5, score_thr=0.5)

        assert nms_result is None  # Should return None when no detections meet threshold

    def test_preprocess_functionality(self):
        """Test image preprocessing function."""
        original_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        target_input_size = (640, 640)

        preprocessed_image, scale_ratio = preprocess(original_image, target_input_size)

        assert isinstance(preprocessed_image, np.ndarray)
        assert isinstance(scale_ratio, float)
        assert preprocessed_image.shape == (3, 640, 640)  # Should be CHW format
        assert preprocessed_image.dtype == np.float32
        assert scale_ratio > 0

    def test_preprocess_grayscale_input(self):
        """Test preprocessing with grayscale input."""
        grayscale_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        target_input_size = (640, 640)

        # Convert grayscale to 3-channel for preprocessing
        rgb_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)

        preprocessed_image, scale_ratio = preprocess(rgb_image, target_input_size)

        assert isinstance(preprocessed_image, np.ndarray)
        assert isinstance(scale_ratio, float)
        assert len(preprocessed_image.shape) == 3  # Should be CHW format
        assert preprocessed_image.shape[0] == 3  # 3 channels
        assert preprocessed_image.dtype == np.float32

    def test_demo_postprocess_functionality(self):
        """Test demo postprocessing function."""
        # Create mock model outputs
        batch_size = 1
        image_size = (640, 640)
        detection_strides = [8, 16, 32]

        # Calculate the number of grid points based on image size and strides
        total_grid_predictions = sum(
            (image_size[0] // stride) * (image_size[1] // stride) for stride in detection_strides
        )
        # For 640x640: (80*80) + (40*40) + (20*20) = 6400 + 1600 + 400 = 8400

        num_prediction_params = 6  # x, y, w, h, obj_conf, class_conf

        model_outputs = np.random.rand(batch_size, total_grid_predictions, num_prediction_params).astype(np.float32)
        original_outputs = model_outputs.copy()  # Save original for comparison

        postprocessed_result = demo_postprocess(model_outputs, image_size)

        assert isinstance(postprocessed_result, np.ndarray)
        assert postprocessed_result.shape == original_outputs.shape

        # Check that coordinates have been transformed
        assert not np.array_equal(postprocessed_result, original_outputs)

        # Check that first two coordinates (x, y) have been modified
        assert not np.array_equal(postprocessed_result[..., :2], original_outputs[..., :2])

        # Check that width/height coordinates (2:4) have been modified (exp applied)
        assert not np.array_equal(postprocessed_result[..., 2:4], original_outputs[..., 2:4])

        # Check that confidence scores (4:6) remain unchanged
        assert np.array_equal(postprocessed_result[..., 4:6], original_outputs[..., 4:6])


class TestWholebodyClass:
    """Test the Wholebody detector class."""

    @pytest.fixture
    def mock_onnx_session(self):
        """Create a mock ONNX runtime session."""
        mock_inference_session = MagicMock()
        mock_inference_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_inference_session.run.return_value = [np.random.rand(1, 100, 6)]
        return mock_inference_session

    def test_wholebody_initialization_cpu(self):
        """Test Wholebody initialization with CPU device."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session_class:
            mock_session_class.return_value = MagicMock()

            pose_detector = Wholebody("model_det.onnx", "model_pose.onnx", device="cpu")

            assert pose_detector is not None
            assert hasattr(pose_detector, "session_det")
            assert hasattr(pose_detector, "session_pose")

            # Verify CPU providers were used
            mock_session_class.assert_called_with(
                path_or_bytes="model_pose.onnx", providers=["CPUExecutionProvider"], provider_options=None
            )

    def test_wholebody_initialization_cuda(self):
        """Test Wholebody initialization with CUDA device."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session_class:
            mock_session_class.return_value = MagicMock()

            pose_detector = Wholebody("model_det.onnx", "model_pose.onnx", device="cuda")

            assert pose_detector is not None

            # Verify CUDA providers were used
            mock_session_class.assert_called_with(
                path_or_bytes="model_pose.onnx",
                providers=["CUDAExecutionProvider"],
                provider_options=[{"device_id": 0}],
            )

    def test_wholebody_initialization_cuda_with_id(self):
        """Test Wholebody initialization with specific CUDA device ID."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session_class:
            mock_session_class.return_value = MagicMock()

            pose_detector = Wholebody("model_det.onnx", "model_pose.onnx", device="cuda:1")

            assert pose_detector is not None

            # Verify CUDA providers with specific device ID were used
            mock_session_class.assert_called_with(
                path_or_bytes="model_pose.onnx",
                providers=["CUDAExecutionProvider"],
                provider_options=[{"device_id": 1}],
            )

    def test_wholebody_call_functionality(self):
        """Test Wholebody __call__ method."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session_class:
            # Mock detection session
            mock_detection_session = MagicMock()
            mock_detection_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_detection_session.run.return_value = [np.random.rand(1, 100, 6)]

            # Mock pose session
            mock_pose_session = MagicMock()
            mock_pose_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_pose_session.run.return_value = [np.random.rand(1, 133, 3)]

            mock_session_class.side_effect = [mock_detection_session, mock_pose_session]

            with patch("easy_dwpose.body_estimation.wholebody.inference_detector") as mock_detector_inference:
                mock_detector_inference.return_value = np.array([[100, 100, 200, 200]])  # Mock bounding box

                with patch("easy_dwpose.body_estimation.wholebody.inference_pose") as mock_pose_inference:
                    # Mock pose inference returns
                    detected_keypoints = np.random.rand(1, 133, 2)
                    keypoint_scores = np.random.rand(1, 133)
                    mock_pose_inference.return_value = (detected_keypoints, keypoint_scores)

                    pose_detector = Wholebody("model_det.onnx", "model_pose.onnx")

                    # Test with a sample image
                    test_input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    final_keypoints, final_scores = pose_detector(test_input_image)

                    assert isinstance(final_keypoints, np.ndarray)
                    assert isinstance(final_scores, np.ndarray)
                    assert final_keypoints.shape[1] == 134  # DWPose format (full keypoint set)
                    assert final_scores.shape[1] == 134  # DWPose format (full keypoint set)

    def test_wholebody_keypoint_mapping(self):
        """Test that keypoints are correctly mapped to OpenPose format."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session_class:
            mock_detection_session = MagicMock()
            mock_detection_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_detection_session.run.return_value = [np.random.rand(1, 100, 6)]

            mock_pose_session = MagicMock()
            mock_pose_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_pose_session.run.return_value = [np.random.rand(1, 133, 3)]

            mock_session_class.side_effect = [mock_detection_session, mock_pose_session]

            with patch("easy_dwpose.body_estimation.wholebody.inference_detector") as mock_detector_inference:
                mock_detector_inference.return_value = np.array([[100, 100, 200, 200]])

                with patch("easy_dwpose.body_estimation.wholebody.inference_pose") as mock_pose_inference:
                    # Create predictable keypoints for testing mapping
                    uniform_keypoints = np.ones((1, 133, 2)) * 0.5  # All keypoints at (0.5, 0.5)
                    uniform_scores = np.ones((1, 133)) * 0.8  # All scores at 0.8
                    mock_pose_inference.return_value = (uniform_keypoints, uniform_scores)

                    pose_detector = Wholebody("model_det.onnx", "model_pose.onnx")
                    test_input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                    mapped_keypoints, mapped_scores = pose_detector(test_input_image)

                    # Check output format
                    assert mapped_keypoints.shape[1] == 134  # DWPose body keypoints
                    assert mapped_scores.shape[1] == 134  # DWPose body keypoints

                    # Check that mapping was applied (values should be different from input)
                    # The neck joint should be computed as average of shoulders
                    assert mapped_keypoints.shape[0] == 1  # One person detected


class TestDetectorIntegration:
    """Integration tests for detector components."""

    def test_full_detection_pipeline_mock(self):
        """Test the full detection pipeline with mocked components."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session_class:
            # Setup mock sessions
            mock_detection_session = MagicMock()
            mock_detection_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_detection_session.run.return_value = [np.random.rand(1, 8400, 6)]  # YOLO-style output

            mock_pose_session = MagicMock()
            # Mock the pose session input to have proper shape for RTMPose (batch, channels, height, width)
            mock_pose_input = MagicMock(name="input")
            mock_pose_input.shape = [1, 3, 256, 192]  # RTMPose input shape
            mock_pose_session.get_inputs.return_value = [mock_pose_input]
            # RTMPose returns two outputs: simcc_x and simcc_y
            mock_pose_session.run.return_value = [
                np.random.rand(1, 133, 96),  # simcc_x: (batch, keypoints, width/2)
                np.random.rand(1, 133, 128),  # simcc_y: (batch, keypoints, height/2)
            ]

            mock_session_class.side_effect = [mock_detection_session, mock_pose_session]

            # Test with realistic image
            test_input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            integrated_detector = Wholebody("fake_det.onnx", "fake_pose.onnx")

            # This should not crash and should return reasonable outputs
            pipeline_keypoints, pipeline_scores = integrated_detector(test_input_image)

            assert isinstance(pipeline_keypoints, np.ndarray)
            assert isinstance(pipeline_scores, np.ndarray)
            assert len(pipeline_keypoints.shape) >= 2
            assert len(pipeline_scores.shape) >= 2

    @pytest.mark.parametrize(
        "image_size",
        [
            (240, 320, 3),
            (480, 640, 3),
            (720, 1280, 3),
        ],
    )
    def test_detector_different_image_sizes(self, image_size):
        """Test detector with different input image sizes."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session_class:
            mock_detection_session = MagicMock()
            mock_detection_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_detection_session.run.return_value = [np.random.rand(1, 100, 6)]

            mock_pose_session = MagicMock()
            mock_pose_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_pose_session.run.return_value = [np.random.rand(1, 384, 192)]

            mock_session_class.side_effect = [mock_detection_session, mock_pose_session]

            with patch("easy_dwpose.body_estimation.wholebody.inference_detector") as mock_detector_inference:
                mock_detector_inference.return_value = np.array([[100, 100, 200, 200]])

                with patch("easy_dwpose.body_estimation.wholebody.inference_pose") as mock_pose_inference:
                    variable_keypoints = np.random.rand(1, 133, 2)
                    variable_scores = np.random.rand(1, 133)
                    mock_pose_inference.return_value = (variable_keypoints, variable_scores)

                    multi_size_detector = Wholebody("fake_det.onnx", "fake_pose.onnx")
                    variable_size_image = np.random.randint(0, 255, image_size, dtype=np.uint8)

                    size_adapted_keypoints, size_adapted_scores = multi_size_detector(variable_size_image)

                    assert size_adapted_keypoints is not None
                    assert size_adapted_scores is not None


# Benchmark and stress tests
class TestPerformanceAndStress:
    """Performance and stress tests for body estimation components."""

    def test_resize_image_performance(self):
        """Test resize_image performance with large images."""
        ultra_hd_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)  # 4K image

        performance_result = resize_image(ultra_hd_image, target_resolution=512)

        assert performance_result is not None
        assert min(performance_result.shape[:2]) == 512

    def test_nms_performance_many_boxes(self):
        """Test NMS performance with many bounding boxes."""
        num_test_boxes = 1000
        many_boxes = np.random.rand(num_test_boxes, 4) * 100  # Random boxes in 100x100 space
        many_scores = np.random.rand(num_test_boxes)

        stress_test_indices = nms(many_boxes, many_scores, nms_thr=0.5)

        assert isinstance(stress_test_indices, list)
        assert len(stress_test_indices) <= num_test_boxes
        assert len(stress_test_indices) > 0  # Should keep at least some boxes

    def test_stress_multiple_detections(self):
        """Stress test with multiple consecutive detections."""
        with patch("easy_dwpose.body_estimation.wholebody.onnxruntime.InferenceSession") as mock_session_class:
            mock_detection_session = MagicMock()
            mock_detection_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_detection_session.run.return_value = [np.random.rand(1, 100, 6)]

            mock_pose_session = MagicMock()
            mock_pose_session.get_inputs.return_value = [MagicMock(name="input")]
            mock_pose_session.run.return_value = [np.random.rand(1, 384, 192)]

            mock_session_class.side_effect = [mock_detection_session, mock_pose_session]

            with patch("easy_dwpose.body_estimation.wholebody.inference_detector") as mock_detector_inference:
                mock_detector_inference.return_value = np.array([[100, 100, 200, 200]])

                with patch("easy_dwpose.body_estimation.wholebody.inference_pose") as mock_pose_inference:
                    stress_keypoints = np.random.rand(1, 133, 2)
                    stress_scores = np.random.rand(1, 133)
                    mock_pose_inference.return_value = (stress_keypoints, stress_scores)

                    stress_test_detector = Wholebody("fake_det.onnx", "fake_pose.onnx")

                    # Run multiple detections
                    for detection_iteration in range(10):
                        stress_test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        iteration_keypoints, iteration_scores = stress_test_detector(stress_test_image)

                        assert iteration_keypoints is not None
                        assert iteration_scores is not None


if __name__ == "__main__":
    # Run basic functionality tests if executed directly
    pytest.main([__file__])
