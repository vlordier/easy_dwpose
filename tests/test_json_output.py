"""
Tests for JSON output functionality in DWposeDetector.
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from easy_dwpose import DWposeDetector


class TestJSONOutput:
    """Test JSON output functionality."""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return DWposeDetector(device="cpu")

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        if os.path.exists("assets/pose.png"):
            return Image.open("assets/pose.png").convert("RGB")
        # Create a dummy test image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(dummy_image)

    def test_json_output_type(self, detector, sample_image):
        """Test that JSON output returns a valid JSON string."""
        result = detector(sample_image, output_type="json", draw_pose=None)

        assert isinstance(result, str)

        # Validate that it's proper JSON
        parsed_json = json.loads(result)
        assert isinstance(parsed_json, dict)

    def test_dict_output_type(self, detector, sample_image):
        """Test that dict output returns a dictionary."""
        result = detector(sample_image, output_type="dict", draw_pose=None)

        assert isinstance(result, dict)

    def test_json_contains_expected_keys(self, detector, sample_image):
        """Test that JSON output contains expected pose data keys."""
        result = detector(sample_image, output_type="json", draw_pose=None)
        parsed_json = json.loads(result)

        expected_keys = ["bodies", "body_scores", "hands", "hands_scores", "faces", "faces_scores"]
        for key in expected_keys:
            assert key in parsed_json, f"Missing key: {key}"

    def test_dict_contains_expected_keys(self, detector, sample_image):
        """Test that dict output contains expected pose data keys."""
        result = detector(sample_image, output_type="dict", draw_pose=None)

        expected_keys = ["bodies", "body_scores", "hands", "hands_scores", "faces", "faces_scores"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_json_serializable_data(self, detector, sample_image):
        """Test that JSON output contains only serializable data types."""
        result = detector(sample_image, output_type="json", draw_pose=None)
        parsed_json = json.loads(result)

        def check_serializable(obj):
            """Recursively check if object is JSON serializable."""
            if isinstance(obj, (str, int, float, bool, type(None))):
                return True
            elif isinstance(obj, (list, tuple)):
                return all(check_serializable(item) for item in obj)
            elif isinstance(obj, dict):
                return all(check_serializable(v) for v in obj.values())
            else:
                return False

        assert check_serializable(parsed_json), "JSON contains non-serializable data"

    def test_json_data_structure(self, detector, sample_image):
        """Test the structure of JSON data."""
        result = detector(sample_image, output_type="json", draw_pose=None)
        parsed_json = json.loads(result)

        # Bodies should be a list of lists (coordinates)
        assert isinstance(parsed_json["bodies"], list)
        if len(parsed_json["bodies"]) > 0:
            assert isinstance(parsed_json["bodies"][0], list)

        # Scores should be lists
        assert isinstance(parsed_json["body_scores"], list)
        assert isinstance(parsed_json["hands_scores"], list)
        assert isinstance(parsed_json["faces_scores"], list)

        # Hands and faces should be lists
        assert isinstance(parsed_json["hands"], list)
        assert isinstance(parsed_json["faces"], list)

    def test_json_vs_dict_consistency(self, detector, sample_image):
        """Test that JSON and dict outputs contain the same data."""
        json_result = detector(sample_image, output_type="json", draw_pose=None)
        dict_result = detector(sample_image, output_type="dict", draw_pose=None)

        parsed_json = json.loads(json_result)

        # Convert numpy arrays in dict to lists for comparison
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            return obj

        dict_converted = convert_numpy_to_list(dict_result)

        # Compare keys
        assert set(parsed_json.keys()) == set(dict_converted.keys())

        # Compare data structure (not exact values due to possible numerical precision)
        for key in parsed_json.keys():
            assert type(parsed_json[key]) == type(dict_converted[key])

    @pytest.mark.parametrize("output_type", ["json", "dict"])
    def test_data_output_with_drawing_disabled(self, detector, sample_image, output_type):
        """Test that data outputs work when drawing is explicitly disabled."""
        result = detector(sample_image, output_type=output_type, draw_pose=None)

        if output_type == "json":
            assert isinstance(result, str)
            parsed_result = json.loads(result)
            assert isinstance(parsed_result, dict)
        else:  # dict
            assert isinstance(result, dict)

    def test_invalid_output_type_raises_error(self, detector, sample_image):
        """Test that invalid output types raise ValueError."""
        invalid_types = ["invalid", "jpeg", "png", 123, None]

        for invalid_type in invalid_types:
            with pytest.raises(ValueError, match="output_type should be"):
                detector(sample_image, output_type=invalid_type)

    def test_json_formatting(self, detector, sample_image):
        """Test that JSON output is properly formatted with indentation."""
        result = detector(sample_image, output_type="json", draw_pose=None)

        # Check that JSON is formatted with indentation
        assert "\n" in result, "JSON should be formatted with newlines"
        assert "  " in result, "JSON should be indented"

    def test_pose_to_json_serializable_method(self, detector):
        """Test the _pose_to_json_serializable method directly."""
        # Create sample pose data with numpy arrays
        sample_pose = {
            "bodies": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "body_scores": np.array([0.8, 0.9]),
            "hands": np.array([[[0.1, 0.2], [0.3, 0.4]]]),
            "hands_scores": np.array([[0.7, 0.8]]),
            "faces": np.array([[[0.5, 0.6]]]),
            "faces_scores": np.array([[0.9]]),
        }

        result = detector._pose_to_json_serializable(sample_pose)

        # Check that numpy arrays are converted to lists
        assert isinstance(result["bodies"], list)
        assert isinstance(result["body_scores"], list)
        assert isinstance(result["hands"], list)
        assert isinstance(result["hands_scores"], list)
        assert isinstance(result["faces"], list)
        assert isinstance(result["faces_scores"], list)

        # Verify the actual conversion
        assert result["bodies"] == [[1.0, 2.0], [3.0, 4.0]]
        assert result["body_scores"] == [0.8, 0.9]

    @pytest.mark.parametrize("resolution", [256, 384, 512])
    def test_json_output_different_resolutions(self, detector, sample_image, resolution):
        """Test JSON output at different resolutions."""
        result = detector(sample_image, output_type="json", detect_resolution=resolution, draw_pose=None)

        assert isinstance(result, str)
        parsed_json = json.loads(result)
        assert isinstance(parsed_json, dict)

        # Data should be valid regardless of resolution
        expected_keys = ["bodies", "body_scores", "hands", "hands_scores", "faces", "faces_scores"]
        for key in expected_keys:
            assert key in parsed_json


class TestJSONWithDrawing:
    """Test JSON output when combined with drawing functions."""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return DWposeDetector(device="cpu")

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        if os.path.exists("assets/pose.png"):
            return Image.open("assets/pose.png").convert("RGB")
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return Image.fromarray(dummy_image)

    def test_json_output_ignores_drawing_function(self, detector, sample_image):
        """Test that JSON output ignores drawing function even if provided."""
        from easy_dwpose.draw.openpose import draw_pose as draw_openpose

        result = detector(sample_image, output_type="json", draw_pose=draw_openpose)

        # Should still return JSON string, not an image
        assert isinstance(result, str)
        parsed_json = json.loads(result)
        assert isinstance(parsed_json, dict)

    def test_dict_output_ignores_drawing_function(self, detector, sample_image):
        """Test that dict output ignores drawing function even if provided."""
        from easy_dwpose.draw.openpose import draw_pose as draw_openpose

        result = detector(sample_image, output_type="dict", draw_pose=draw_openpose)

        # Should still return dict, not an image
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
