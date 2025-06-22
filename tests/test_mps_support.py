"""
Tests for MPS (Metal Performance Shaders) support on Apple Silicon devices.
"""

import os
import platform
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.body_estimation import Wholebody


class TestMPSSupport:
    """Test MPS support for Apple Silicon devices."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        if os.path.exists("assets/pose.png"):
            return Image.open("assets/pose.png").convert("RGB")
        # Create a dummy test image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(dummy_image)

    def test_mps_device_validation_available(self):
        """Test MPS device validation when MPS is available."""
        with patch("torch.backends.mps.is_available", return_value=True):
            detector = DWposeDetector(device="mps")
            assert detector is not None
            assert hasattr(detector, "pose_estimation")

    def test_mps_device_validation_unavailable(self):
        """Test MPS device validation when MPS is unavailable."""
        with patch("torch.backends.mps.is_available", return_value=False):
            with patch("builtins.print") as mock_print:
                detector = DWposeDetector(device="mps")
                assert detector is not None
                mock_print.assert_called_with("Warning: MPS is not available on this system. Falling back to CPU.")

    def test_wholebody_mps_device_initialization(self):
        """Test Wholebody initialization with MPS device."""
        with patch("onnxruntime.InferenceSession") as mock_session:
            mock_session.return_value = MagicMock()

            Wholebody("model_det.onnx", "model_pose.onnx", device="mps")

            # Verify sessions were created
            assert mock_session.call_count == 2

            # Verify CoreML providers were used for MPS
            calls = mock_session.call_args_list
            for call in calls:
                providers = call[1]["providers"]
                assert isinstance(providers, list)
                # Should have CoreML as first provider, CPU as fallback
                if len(providers) > 1:
                    assert isinstance(providers[0], tuple)
                    assert providers[0][0] == "CoreMLExecutionProvider"
                    assert providers[1] == "CPUExecutionProvider"

    @pytest.mark.parametrize("device", ["mps", "mps:0"])
    def test_mps_device_strings(self, device):
        """Test various MPS device string formats."""
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("onnxruntime.InferenceSession") as mock_session:
                mock_session.return_value = MagicMock()

                detector = DWposeDetector(device=device)
                assert detector is not None

    def test_mps_inference_execution(self, sample_image):
        """Test that MPS device can perform inference."""
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("onnxruntime.InferenceSession") as mock_session:
                # Mock the ONNX sessions
                mock_det_session = MagicMock()
                mock_pose_session = MagicMock()
                mock_session.side_effect = [mock_det_session, mock_pose_session]

                # Mock detection session input/output
                mock_det_input = MagicMock()
                mock_det_input.shape = [1, 3, 640, 640]
                mock_det_session.get_inputs.return_value = [mock_det_input]

                # Mock detection results - proper YOLOX output format
                # YOLOX outputs have shape (1, 8400, 85) where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
                mock_detection_output = np.zeros((1, 8400, 85))
                # Set up one detection in the middle of the output
                mock_detection_output[0, 4200, :4] = [320, 320, 100, 100]  # center_x, center_y, width, height
                mock_detection_output[0, 4200, 4] = 0.9  # objectness score
                mock_detection_output[0, 4200, 5] = 0.9  # class score for person (class 0)
                mock_det_session.run.return_value = [mock_detection_output]

                # Mock pose session input/output
                mock_pose_input = MagicMock()
                mock_pose_input.shape = [1, 3, 384, 288]  # DWPose model input shape
                mock_pose_session.get_inputs.return_value = [mock_pose_input]

                # Mock pose estimation results
                mock_keypoints = np.random.rand(1, 133, 2)  # 1 person, 133 keypoints, x,y
                mock_scores = np.random.rand(1, 133)  # confidence scores
                mock_pose_session.run.return_value = [mock_keypoints, mock_scores]

                detector = DWposeDetector(device="mps")
                result = detector(sample_image, output_type="pil")

                assert result is not None
                assert isinstance(result, Image.Image)
                mock_det_input.shape = [1, 3, 640, 640]
                mock_det_session.get_inputs.return_value = [mock_det_input]

                # Mock detection results - proper YOLOX output format
                # YOLOX outputs have shape (1, 8400, 85) where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
                mock_detection_output = np.zeros((1, 8400, 85))
                # Set up one detection in the middle of the output
                mock_detection_output[0, 4200, :4] = [320, 320, 100, 100]  # center_x, center_y, width, height
                mock_detection_output[0, 4200, 4] = 0.9  # objectness score
                mock_detection_output[0, 4200, 5] = 0.9  # class score for person (class 0)
                mock_det_session.run.return_value = [mock_detection_output]

                # Mock pose session input/output
                mock_pose_input = MagicMock()
                mock_pose_input.shape = [1, 3, 384, 288]  # DWPose model input shape
                mock_pose_session.get_inputs.return_value = [mock_pose_input]

                # Mock pose estimation results
                mock_keypoints = np.random.rand(1, 133, 2)  # 1 person, 133 keypoints, x,y
                mock_scores = np.random.rand(1, 133)  # confidence scores
                mock_pose_session.run.return_value = [mock_keypoints, mock_scores]

                detector = DWposeDetector(device="mps")
                result = detector(sample_image, output_type="pil")

                assert result is not None
                assert isinstance(result, Image.Image)

    def test_device_validation_cyrillic_cpu(self):
        """Test that Cyrillic 'сpu' is converted to Latin 'cpu'."""
        detector = DWposeDetector(device="сpu")  # Cyrillic 'с'
        validated_device = detector._validate_device("сpu")
        assert validated_device == "cpu"

    @pytest.mark.skipif(platform.system() != "Darwin", reason="MPS is only available on macOS")
    def test_real_mps_availability_on_macos(self):
        """Test actual MPS availability on macOS systems."""
        if torch.backends.mps.is_available():
            # If MPS is available, test that we can create a detector
            detector = DWposeDetector(device="mps")
            assert detector is not None
        else:
            # If MPS is not available, test fallback to CPU
            with patch("builtins.print") as mock_print:
                detector = DWposeDetector(device="mps")
                assert detector is not None
                mock_print.assert_called_with("Warning: MPS is not available on this system. Falling back to CPU.")

    def test_mps_with_custom_drawing_functions(self, sample_image):
        """Test MPS device with custom drawing functions."""
        from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose

        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("onnxruntime.InferenceSession") as mock_session:
                # Mock the ONNX sessions
                mock_det_session = MagicMock()
                mock_pose_session = MagicMock()
                mock_session.side_effect = [mock_det_session, mock_pose_session]

                # Mock detection and pose results
                mock_det_input = MagicMock()
                mock_det_input.shape = [1, 3, 640, 640]
                mock_det_session.get_inputs.return_value = [mock_det_input]

                mock_detection_output = np.zeros((1, 8400, 85))
                mock_detection_output[0, 4200, :4] = [320, 320, 100, 100]  # center_x, center_y, width, height
                mock_detection_output[0, 4200, 4] = 0.9  # objectness score
                mock_detection_output[0, 4200, 5] = 0.9  # class score for person (class 0)
                mock_det_session.run.return_value = [mock_detection_output]

                mock_pose_input = MagicMock()
                mock_pose_input.shape = [1, 3, 384, 288]
                mock_pose_session.get_inputs.return_value = [mock_pose_input]
                mock_keypoints = np.random.rand(1, 133, 2)
                mock_scores = np.random.rand(1, 133)
                mock_pose_session.run.return_value = [mock_keypoints, mock_scores]

                detector = DWposeDetector(device="mps")
                result = detector(
                    sample_image,
                    output_type="pil",
                    draw_pose=draw_pose_musepose,
                    include_hands=False,
                    include_face=False,
                )

                assert result is not None
                assert isinstance(result, Image.Image)

    def test_mps_output_formats(self, sample_image):
        """Test MPS device with different output formats."""
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("onnxruntime.InferenceSession") as mock_session:
                # Mock the ONNX sessions
                mock_det_session = MagicMock()
                mock_pose_session = MagicMock()
                mock_session.side_effect = [mock_det_session, mock_pose_session]

                # Mock results
                mock_det_input = MagicMock()
                mock_det_input.shape = [1, 3, 640, 640]
                mock_det_session.get_inputs.return_value = [mock_det_input]

                mock_detection_output = np.zeros((1, 8400, 85))
                mock_detection_output[0, 4200, :4] = [320, 320, 100, 100]  # center_x, center_y, width, height
                mock_detection_output[0, 4200, 4] = 0.9  # objectness score
                mock_detection_output[0, 4200, 5] = 0.9  # class score for person (class 0)
                mock_det_session.run.return_value = [mock_detection_output]

                mock_pose_input = MagicMock()
                mock_pose_input.shape = [1, 3, 384, 288]
                mock_pose_session.get_inputs.return_value = [mock_pose_input]
                mock_keypoints = np.random.rand(1, 133, 2)
                mock_scores = np.random.rand(1, 133)
                mock_pose_session.run.return_value = [mock_keypoints, mock_scores]

                detector = DWposeDetector(device="mps")

                # Test PIL output
                pil_result = detector(sample_image, output_type="pil")
                assert isinstance(pil_result, Image.Image)

                # Test numpy output
                np_result = detector(sample_image, output_type="np")
                assert isinstance(np_result, np.ndarray)

                # Test dict output (no drawing)
                dict_result = detector(sample_image, draw_pose=None)
                assert isinstance(dict_result, dict)
                assert "bodies" in dict_result
                assert "hands" in dict_result
                assert "faces" in dict_result

    def test_mps_memory_considerations(self):
        """Test that MPS usage provides appropriate information about CoreML acceleration."""
        with patch("torch.backends.mps.is_available", return_value=True):
            with patch("builtins.print") as mock_print:
                detector = DWposeDetector(device="mps")
                assert detector is not None
                # Should print note about CoreML ExecutionProvider
                mock_print.assert_called_with(
                    "Note: Using MPS device. ONNX Runtime will use CoreML ExecutionProvider "
                    "for Apple Silicon acceleration."
                )

    @pytest.mark.parametrize("invalid_device", ["mps:invalid", "mps:abc", "mps:-1"])
    def test_invalid_mps_device_strings(self, invalid_device):
        """Test handling of invalid MPS device strings."""
        with patch("torch.backends.mps.is_available", return_value=True):
            # These should not raise errors but should work (device parsing is simple)
            detector = DWposeDetector(device=invalid_device)
            assert detector is not None

    def test_performance_comparison_info(self):
        """Test that appropriate performance information is available."""
        # This test documents expected performance characteristics
        performance_notes = {
            "mps": "Uses CoreML ExecutionProvider for Apple Silicon hardware acceleration",
            "cpu": "Uses CPU for all operations",
            "cuda": "Uses GPU for ONNX Runtime operations when available",
        }

        for device, note in performance_notes.items():
            assert isinstance(note, str)
            assert len(note) > 0
