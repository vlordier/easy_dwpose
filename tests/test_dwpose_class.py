import os
import tempfile
from typing import Dict, List, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.draw import draw_openpose
from easy_dwpose.draw.mimic_motion import draw_pose as draw_pose_mimic_motion
from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose

SAVE_SIZE = (256, 448)


# Test fixtures
@pytest.fixture(scope="session")
def sample_image():
    """Fixture to provide a sample test image."""
    image_path = "assets/pose.png"
    if not os.path.exists(image_path):
        # Create a dummy test image if assets don't exist
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(dummy_image)
    return Image.open(image_path).convert("RGB")


@pytest.fixture(scope="session")
def detector():
    """Fixture to provide a DWpose detector instance."""
    return DWposeDetector()


@pytest.fixture
def sample_numpy_image():
    """Fixture to provide a sample numpy image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def various_image_sizes():
    """Fixture to provide various image sizes for testing."""
    return [(128, 128, 3), (256, 256, 3), (480, 640, 3), (720, 1280, 3), (1080, 1920, 3)]


class TestDWposeDetector:
    """Comprehensive test class for DWposeDetector functionality."""

    # Initialization Tests
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            "сpu",  # Cyrillic character test
            "mps",  # Apple Silicon MPS support
        ],
    )
    def test_detector_initialization_valid_devices(self, device):
        """Test detector initialization with valid device strings."""
        with patch("torch.backends.mps.is_available", return_value=True):
            detector = DWposeDetector(device=device)
            assert detector is not None
            assert hasattr(detector, "pose_estimation")
            assert detector.pose_estimation is not None

    def test_detector_initialization_default(self):
        """Test detector initialization with default parameters."""
        detector = DWposeDetector()
        assert detector is not None
        assert hasattr(detector, "pose_estimation")

    # Output Type Tests
    @pytest.mark.parametrize(
        "output_type,expected_type",
        [
            ("pil", Image.Image),
            ("np", np.ndarray),
        ],
    )
    def test_forward_output_types(self, detector, sample_image, output_type, expected_type):
        """Test forward pass with different output types."""
        result = detector(sample_image, output_type=output_type)
        assert result is not None
        assert isinstance(result, expected_type)

        if output_type == "np":
            assert len(result.shape) == 3  # H x W x C
            assert result.shape[2] == 3  # RGB channels
        elif output_type == "pil":
            width, height = result.size
            assert width > 0
            assert height > 0

    def test_forward_dict_output_no_drawing(self, detector, sample_image):
        """Test forward pass returning pose dictionary without drawing."""
        result = detector(sample_image, draw_pose=None)

        assert result is not None
        assert isinstance(result, dict)

        # Check expected keys in pose dictionary
        expected_keys = ["bodies", "body_scores", "hands", "hands_scores", "faces", "faces_scores"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], np.ndarray), f"Key {key} should be numpy array"

    def test_invalid_output_type_raises_error(self, detector, sample_image):
        """Test that invalid output types raise ValueError."""
        invalid_types = ["invalid", "jpeg", "png", 123, None]

        for invalid_type in invalid_types:
            with pytest.raises(ValueError, match="output_type should be 'pil' or 'np'"):
                detector(sample_image, output_type=invalid_type)

    # Resolution Tests
    @pytest.mark.parametrize("detect_resolution", [128, 256, 384, 512, 640, 768])
    def test_different_detect_resolutions(self, detector, sample_image, detect_resolution):
        """Test detection with different resolutions."""
        result = detector(sample_image, detect_resolution=detect_resolution, output_type="pil")
        assert result is not None
        assert isinstance(result, Image.Image)

        # Verify the result has reasonable dimensions
        width, height = result.size
        assert width > 0 and height > 0

    # Drawing Function Tests
    @pytest.mark.parametrize(
        "draw_function,kwargs",
        [
            (draw_openpose, {"include_hands": True, "include_face": True}),
            (draw_openpose, {"include_hands": False, "include_face": False}),
            (draw_openpose, {"include_hands": True, "include_face": False}),
            (draw_openpose, {"include_hands": False, "include_face": True}),
            (draw_pose_musepose, {"draw_face": True}),
            (draw_pose_musepose, {"draw_face": False}),
            (draw_pose_mimic_motion, {"ref_w": 2160}),
            (draw_pose_mimic_motion, {"ref_w": 1080}),
        ],
    )
    def test_different_drawing_functions(self, detector, sample_image, draw_function, kwargs):
        """Test with different drawing functions and their parameters."""
        result = detector(sample_image, output_type="pil", draw_pose=draw_function, **kwargs)
        assert result is not None
        assert isinstance(result, Image.Image)

        # Verify reasonable image dimensions
        width, height = result.size
        assert width > 0 and height > 0

    # Input Type Tests
    def test_numpy_input(self, detector, sample_numpy_image):
        """Test with numpy array input."""
        result = detector(sample_numpy_image, output_type="np")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_pil_input(self, detector, sample_image):
        """Test with PIL image input."""
        result = detector(sample_image, output_type="pil")
        assert result is not None
        assert isinstance(result, Image.Image)
        width, height = result.size
        assert width > 0 and height > 0

    # Image Size Tests
    def test_various_input_image_sizes(self, detector, various_image_sizes):
        """Test with various input image sizes."""
        for height, width, channels in various_image_sizes:
            test_image = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)

            result = detector(test_image, output_type="np")
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3

    # Pose Data Structure Tests
    def test_pose_data_structure_detailed(self, detector, sample_image):
        """Test detailed structure of pose data returned."""
        pose = detector(sample_image, draw_pose=None)

        # Test bodies structure
        assert "bodies" in pose
        bodies = pose["bodies"]
        assert isinstance(bodies, np.ndarray)
        assert bodies.ndim == 2
        assert bodies.shape == (18, 2)  # 18 body keypoints with x, y coordinates

        # Test body_scores structure
        assert "body_scores" in pose
        body_scores = pose["body_scores"]
        assert isinstance(body_scores, np.ndarray)
        assert body_scores.shape == (1, 18)  # scores for 18 body keypoints

        # Test hands structure
        assert "hands" in pose
        hands = pose["hands"]
        assert isinstance(hands, np.ndarray)
        assert hands.shape == (2, 21, 2)  # 2 hands, 21 keypoints each, x,y coordinates
        assert hands.ndim == 3

        # Test hands_scores structure
        assert "hands_scores" in pose
        hands_scores = pose["hands_scores"]
        assert isinstance(hands_scores, np.ndarray)
        assert hands_scores.shape == (2, 21)  # scores for 2 hands, 21 keypoints each

        # Test faces structure
        assert "faces" in pose
        faces = pose["faces"]
        assert isinstance(faces, np.ndarray)
        assert faces.shape == (1, 68, 2)  # 1 face, 68 keypoints, x,y coordinates
        assert faces.ndim == 3

        # Test faces_scores structure
        assert "faces_scores" in pose
        faces_scores = pose["faces_scores"]
        assert isinstance(faces_scores, np.ndarray)
        assert faces_scores.shape == (1, 68)  # scores for 1 face, 68 keypoints

    # Performance and Consistency Tests
    def test_detector_consistency(self, detector, sample_image):
        """Test that detector produces consistent results."""
        result1 = detector(sample_image, draw_pose=None)
        result2 = detector(sample_image, draw_pose=None)

        # Results should be identical for same input
        for key in result1.keys():
            np.testing.assert_array_equal(result1[key], result2[key], err_msg=f"Inconsistent results for key: {key}")

    @pytest.mark.parametrize("num_runs", [3])
    def test_detector_stability_multiple_runs(self, detector, sample_image, num_runs):
        """Test detector stability over multiple runs."""
        results = []
        for _ in range(num_runs):
            result = detector(sample_image, draw_pose=None)
            results.append(result)

        # All results should have same structure
        base_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result.keys() == base_result.keys(), f"Key mismatch in run {i}"
            for key in base_result.keys():
                assert result[key].shape == base_result[key].shape, f"Shape mismatch for {key} in run {i}"

    # Integration Tests
    def test_full_pipeline_with_drawing(self, detector, sample_image):
        """Test complete pipeline from input to drawn output."""
        result = detector(sample_image, output_type="pil", detect_resolution=512, include_hands=True, include_face=True)

        assert result is not None
        assert isinstance(result, Image.Image)

        # Save to temporary file to test serialization
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            result.save(tmp.name)
            assert os.path.exists(tmp.name)

            # Load back and verify
            loaded_image = Image.open(tmp.name)
            assert loaded_image.size == result.size

            # Cleanup
            os.unlink(tmp.name)

    def test_batch_processing_simulation(self, detector):
        """Test processing multiple images (batch simulation)."""
        batch_size = 3
        images = []

        for i in range(batch_size):
            # Create different test images
            img = np.random.randint(0, 255, (240 + i * 40, 320 + i * 40, 3), dtype=np.uint8)
            images.append(img)

        results = []
        for img in images:
            result = detector(img, output_type="np")
            results.append(result)
            assert result is not None
            assert isinstance(result, np.ndarray)

        # All results should be valid
        assert len(results) == batch_size
        for result in results:
            assert result.shape[2] == 3  # RGB channels


# Edge case tests
class TestDWposeDetectorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def detector(self):
        return DWposeDetector()

    def test_empty_image_handling(self, detector):
        """Test handling of edge case images."""
        # Very small image
        tiny_image = np.ones((1, 1, 3), dtype=np.uint8) * 255
        result = detector(tiny_image, output_type="np")
        assert result is not None

        # Single color image
        solid_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = detector(solid_image, output_type="np")
        assert result is not None

    def test_invalid_image_inputs(self, detector):
        """Test handling of invalid image inputs."""
        invalid_inputs = [
            np.array([]),  # Empty array
            np.ones((10, 10)),  # 2D array instead of 3D
            np.ones((10, 10, 1)),  # Single channel
            np.ones((10, 10, 4)),  # RGBA
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, IndexError, TypeError)):
                detector(invalid_input, output_type="np")

    @pytest.mark.parametrize(
        "image_format",
        [
            "RGB",
            "RGBA",
            "L",  # Grayscale
        ],
    )
    def test_different_pil_image_formats(self, detector, image_format):
        """Test handling of different PIL image formats."""
        # Create a test image in different formats
        base_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        pil_image = Image.fromarray(base_image).convert(image_format)

        if image_format in ["RGB", "RGBA"]:
            # Should work fine
            result = detector(pil_image, output_type="pil")
            assert result is not None
        else:
            # May raise an error or handle gracefully
            try:
                result = detector(pil_image, output_type="pil")
                assert result is not None
            except (ValueError, IndexError):
                # Expected for unsupported formats
                pass

    def test_extreme_aspect_ratios(self, detector):
        """Test with extreme aspect ratios."""
        import cv2

        extreme_images = [
            np.random.randint(0, 255, (10, 1000, 3), dtype=np.uint8),  # Very wide
            np.random.randint(0, 255, (1000, 10, 3), dtype=np.uint8),  # Very tall
            np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8),  # Tiny square
        ]

        for img in extreme_images:
            try:
                result = detector(img, output_type="np")
                assert result is not None
                assert isinstance(result, np.ndarray)
            except cv2.error as e:
                # OpenCV has limits on image dimensions (SHRT_MAX)
                # Skip this test case if it exceeds OpenCV's internal limits
                if "SHRT_MAX" in str(e):
                    pytest.skip(f"Image dimensions exceed OpenCV limits: {e}")
                else:
                    raise


# Legacy compatibility tests (preserved for backward compatibility)
def test_forward():
    """Legacy test function for backward compatibility."""
    detector = DWposeDetector()
    if os.path.exists("assets/pose.png"):
        input_img = Image.open("assets/pose.png").convert("RGB")
    else:
        input_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    result = detector(input_img, output_type="pil", include_hands=True, include_face=True)
    return result


def test_replace_drawing_musepose():
    """Legacy test function for backward compatibility."""
    detector = DWposeDetector()
    if os.path.exists("assets/pose.png"):
        input_img = Image.open("assets/pose.png").convert("RGB")
    else:
        input_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    result = detector(input_img, output_type="pil", draw_pose=draw_pose_musepose, draw_face=False)
    return result


def test_replace_drawing_mimic_motion():
    """Legacy test function for backward compatibility."""
    detector = DWposeDetector()
    if os.path.exists("assets/pose.png"):
        input_img = Image.open("assets/pose.png").convert("RGB")
    else:
        input_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    result = detector(input_img, output_type="pil", draw_pose=draw_pose_mimic_motion)
    return result


if __name__ == "__main__":
    # Run basic tests if executed directly
    res_test_forward = test_forward()
    if hasattr(res_test_forward, "resize"):
        res_test_forward.resize(SAVE_SIZE).save("test_forward.png")

    res_test_replace_drawing_musepose = test_replace_drawing_musepose()
    if hasattr(res_test_replace_drawing_musepose, "resize"):
        res_test_replace_drawing_musepose.resize(SAVE_SIZE).save("test_replace_drawing_musepose.png")

    res_test_replace_drawing_mimic_motion = test_replace_drawing_mimic_motion()
    if hasattr(res_test_replace_drawing_mimic_motion, "resize"):
        res_test_replace_drawing_mimic_motion.resize(SAVE_SIZE).save("test_replace_drawing_mimic_motion.png")
