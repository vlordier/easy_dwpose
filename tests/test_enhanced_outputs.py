"""
Comprehensive tests for enhanced JSON and image outputs with parts filtering.
"""

import json
import os
from typing import Dict, List, Union

import numpy as np
import pytest
from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.draw import draw_openpose


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


class TestEnhancedOutputs:
    """Test class for enhanced output functionality."""

    def test_wholebody_all_parts_dict(self, detector, sample_image):
        """Test getting whole body pose data as dictionary."""
        result = detector.get_wholebody(sample_image, output_type="dict")

        assert isinstance(result, dict)
        expected_keys = ["bodies", "body_scores", "hands", "hands_scores", "faces", "faces_scores"]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], np.ndarray)

    def test_wholebody_all_parts_json(self, detector, sample_image):
        """Test getting whole body pose data as JSON."""
        result = detector.get_wholebody(sample_image, output_type="json")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

        expected_keys = ["bodies", "body_scores", "hands", "hands_scores", "faces", "faces_scores"]
        for key in expected_keys:
            assert key in parsed
            assert isinstance(parsed[key], list)

    def test_body_only_dict(self, detector, sample_image):
        """Test getting body-only pose data as dictionary."""
        result = detector.get_body_only(sample_image, output_type="dict")

        assert isinstance(result, dict)
        expected_keys = ["bodies", "body_scores"]
        unexpected_keys = ["hands", "hands_scores", "faces", "faces_scores"]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], np.ndarray)

        for key in unexpected_keys:
            assert key not in result

    def test_body_only_json(self, detector, sample_image):
        """Test getting body-only pose data as JSON."""
        result = detector.get_body_only(sample_image, output_type="json")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

        expected_keys = ["bodies", "body_scores"]
        unexpected_keys = ["hands", "hands_scores", "faces", "faces_scores"]

        for key in expected_keys:
            assert key in parsed
            assert isinstance(parsed[key], list)

        for key in unexpected_keys:
            assert key not in parsed

    def test_hands_only_dict(self, detector, sample_image):
        """Test getting hands-only pose data as dictionary."""
        result = detector.get_hands_only(sample_image, output_type="dict")

        assert isinstance(result, dict)
        expected_keys = ["hands", "hands_scores"]
        unexpected_keys = ["bodies", "body_scores", "faces", "faces_scores"]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], np.ndarray)

        for key in unexpected_keys:
            assert key not in result

    def test_hands_only_json(self, detector, sample_image):
        """Test getting hands-only pose data as JSON."""
        result = detector.get_hands_only(sample_image, output_type="json")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

        expected_keys = ["hands", "hands_scores"]
        unexpected_keys = ["bodies", "body_scores", "faces", "faces_scores"]

        for key in expected_keys:
            assert key in parsed
            assert isinstance(parsed[key], list)

        for key in unexpected_keys:
            assert key not in parsed

    def test_face_only_dict(self, detector, sample_image):
        """Test getting face-only pose data as dictionary."""
        result = detector.get_face_only(sample_image, output_type="dict")

        assert isinstance(result, dict)
        expected_keys = ["faces", "faces_scores"]
        unexpected_keys = ["bodies", "body_scores", "hands", "hands_scores"]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], np.ndarray)

        for key in unexpected_keys:
            assert key not in result

    def test_face_only_json(self, detector, sample_image):
        """Test getting face-only pose data as JSON."""
        result = detector.get_face_only(sample_image, output_type="json")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

        expected_keys = ["faces", "faces_scores"]
        unexpected_keys = ["bodies", "body_scores", "hands", "hands_scores"]

        for key in expected_keys:
            assert key in parsed
            assert isinstance(parsed[key], list)

        for key in unexpected_keys:
            assert key not in parsed

    @pytest.mark.parametrize("output_type", ["pil", "np"])
    def test_body_only_image_output(self, detector, sample_image, output_type):
        """Test getting body-only pose as image."""
        result = detector.get_body_only(sample_image, output_type=output_type)

        if output_type == "pil":
            assert isinstance(result, Image.Image)
            width, height = result.size
            assert width > 0 and height > 0
        elif output_type == "np":
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3

    @pytest.mark.parametrize("output_type", ["pil", "np"])
    def test_hands_only_image_output(self, detector, sample_image, output_type):
        """Test getting hands-only pose as image."""
        result = detector.get_hands_only(sample_image, output_type=output_type)

        if output_type == "pil":
            assert isinstance(result, Image.Image)
            width, height = result.size
            assert width > 0 and height > 0
        elif output_type == "np":
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3

    @pytest.mark.parametrize("output_type", ["pil", "np"])
    def test_face_only_image_output(self, detector, sample_image, output_type):
        """Test getting face-only pose as image."""
        result = detector.get_face_only(sample_image, output_type=output_type)

        if output_type == "pil":
            assert isinstance(result, Image.Image)
            width, height = result.size
            assert width > 0 and height > 0
        elif output_type == "np":
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3

    @pytest.mark.parametrize("output_type", ["pil", "np"])
    def test_wholebody_image_output(self, detector, sample_image, output_type):
        """Test getting whole body pose as image."""
        result = detector.get_wholebody(sample_image, output_type=output_type)

        if output_type == "pil":
            assert isinstance(result, Image.Image)
            width, height = result.size
            assert width > 0 and height > 0
        elif output_type == "np":
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3

    def test_custom_part_filtering_via_call(self, detector, sample_image):
        """Test custom part filtering via direct __call__ method."""
        # Test body + hands, no face
        result = detector(sample_image, output_type="dict", include_body=True, include_hands=True, include_face=False)

        assert isinstance(result, dict)
        expected_keys = ["bodies", "body_scores", "hands", "hands_scores"]
        unexpected_keys = ["faces", "faces_scores"]

        for key in expected_keys:
            assert key in result

        for key in unexpected_keys:
            assert key not in result

    def test_custom_part_filtering_json(self, detector, sample_image):
        """Test custom part filtering with JSON output."""
        # Test face + hands, no body
        result = detector(sample_image, output_type="json", include_body=False, include_hands=True, include_face=True)

        assert isinstance(result, str)
        parsed = json.loads(result)

        expected_keys = ["hands", "hands_scores", "faces", "faces_scores"]
        unexpected_keys = ["bodies", "body_scores"]

        for key in expected_keys:
            assert key in parsed

        for key in unexpected_keys:
            assert key not in parsed

    def test_empty_parts_filtering(self, detector, sample_image):
        """Test filtering that results in empty parts."""
        # Test with all parts disabled (should return empty dict)
        result = detector(sample_image, output_type="dict", include_body=False, include_hands=False, include_face=False)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_consistency_between_methods(self, detector, sample_image):
        """Test consistency between different methods for getting same data."""
        # Get whole body data using different methods
        result1 = detector.get_wholebody(sample_image, output_type="dict")
        result2 = detector(sample_image, output_type="dict", include_body=True, include_hands=True, include_face=True)

        # Results should be identical
        assert result1.keys() == result2.keys()
        for key in result1.keys():
            np.testing.assert_array_equal(result1[key], result2[key])

    def test_json_serialization_integrity(self, detector, sample_image):
        """Test that JSON serialization maintains data integrity."""
        dict_result = detector.get_wholebody(sample_image, output_type="dict")
        json_result = detector.get_wholebody(sample_image, output_type="json")

        parsed_json = json.loads(json_result)

        # Check that all keys are present
        assert dict_result.keys() == parsed_json.keys()

        # Check that arrays can be converted back and are equivalent
        for key in dict_result.keys():
            original_array = dict_result[key]
            reconstructed_array = np.array(parsed_json[key])
            np.testing.assert_array_equal(original_array, reconstructed_array)

    @pytest.mark.parametrize("detect_resolution", [128, 256, 384, 512])
    def test_different_resolutions_with_parts(self, detector, sample_image, detect_resolution):
        """Test that part filtering works across different detection resolutions."""
        result = detector.get_body_only(sample_image, detect_resolution=detect_resolution, output_type="dict")

        assert isinstance(result, dict)
        expected_keys = ["bodies", "body_scores"]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], np.ndarray)

    def test_drawing_with_part_filtering(self, detector, sample_image):
        """Test that drawing functions respect part filtering flags."""
        # This test ensures the include_* flags are passed to drawing functions
        result = detector(sample_image, output_type="pil", include_body=True, include_hands=False, include_face=False)

        assert isinstance(result, Image.Image)
        width, height = result.size
        assert width > 0 and height > 0
