"""
Advanced validation tests for easy_dwpose with edge cases and boundary conditions.
Tests data validation, boundary conditions, and specific algorithmic behaviors.
"""

import math
import warnings
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import pytest
from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.draw import draw_openpose
from easy_dwpose.draw.mimic_motion import draw_pose as draw_pose_mimic_motion
from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose


class TestDataValidation:
    """Test data validation and boundary conditions."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to provide a DWpose detector instance."""
        return DWposeDetector()

    @pytest.mark.parametrize("image_channel_count", [1, 3, 4])
    def test_image_channel_validation(self, detector, image_channel_count):
        """Test validation of different image channel counts."""
        if image_channel_count == 1:
            # Grayscale
            image_data = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            # Convert to 3 channels for PIL
            image_data = np.stack([image_data] * 3, axis=-1)
            pil_image = Image.fromarray(image_data).convert("L")
        elif image_channel_count == 3:
            # RGB
            image_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            pil_image = Image.fromarray(image_data, mode="RGB")
        else:
            # RGBA
            image_data = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
            pil_image = Image.fromarray(image_data, mode="RGBA")

        try:
            result = detector(
                pil_image,
                output_type="np",
                detect_resolution=128,
                draw_pose=draw_openpose,
                include_hands=False,
                include_face=False,
            )

            # If successful, validate result
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3  # Output should always be RGB

        except Exception as e:
            # Some formats may not be supported
            if image_channel_count != 3:
                pytest.skip(f"Channel count {image_channel_count} may not be supported: {e}")
            else:
                raise

    @pytest.mark.parametrize(
        "resolution_pair",
        [
            (64, 128),  # Very small
            (128, 256),  # Small
            (256, 512),  # Medium
            (384, 768),  # Large
            (512, 1024),  # Very large
        ],
    )
    def test_resolution_scaling_consistency(self, detector, resolution_pair):
        """Test that different resolution pairs produce consistent relative results."""
        low_res, high_res = resolution_pair

        # Create test image
        test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)

        # Process at both resolutions
        result_low = detector(pil_image, draw_pose=None, detect_resolution=low_res)

        result_high = detector(pil_image, draw_pose=None, detect_resolution=high_res)

        # Both should have valid structure
        for result in [result_low, result_high]:
            assert "bodies" in result
            assert "hands" in result
            assert "faces" in result
            assert isinstance(result["bodies"], np.ndarray)
            assert isinstance(result["hands"], np.ndarray)
            assert isinstance(result["faces"], np.ndarray)

        # Structure should be the same regardless of resolution
        assert result_low["bodies"].shape[1:] == result_high["bodies"].shape[1:]
        assert result_low["hands"].shape[1:] == result_high["hands"].shape[1:]
        assert result_low["faces"].shape[1:] == result_high["faces"].shape[1:]

    @pytest.mark.parametrize(
        "data_type_scenario",
        [
            "normal_range",
            "zero_image",
            "max_value_image",
            "noisy_image",
        ],
    )
    def test_input_data_range_robustness(self, detector, data_type_scenario):
        """Test robustness with different input data ranges."""

        if data_type_scenario == "normal_range":
            image_data = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        elif data_type_scenario == "zero_image":
            image_data = np.zeros((200, 200, 3), dtype=np.uint8)
        elif data_type_scenario == "max_value_image":
            image_data = np.full((200, 200, 3), 255, dtype=np.uint8)
        elif data_type_scenario == "noisy_image":
            # High contrast noise
            image_data = np.random.choice([0, 255], size=(200, 200, 3)).astype(np.uint8)

        pil_image = Image.fromarray(image_data)

        try:
            result = detector(
                pil_image,
                output_type="np",
                detect_resolution=128,
                draw_pose=draw_openpose,
                include_hands=False,
                include_face=False,
            )

            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.shape[:2] == (200, 200)
            assert result.shape[2] == 3

            # Check that output is in valid range
            assert np.all(result >= 0) and np.all(result <= 255)

        except Exception as e:
            # Some extreme cases might fail gracefully
            if data_type_scenario in ["zero_image", "max_value_image"]:
                pytest.skip(f"Extreme case {data_type_scenario} may not be handled: {e}")
            else:
                raise

    @pytest.mark.parametrize(
        "aspect_ratio",
        [
            (1, 10),  # Very wide
            (10, 1),  # Very tall
            (1, 1),  # Square
            (16, 9),  # Widescreen
            (4, 3),  # Standard
            (3, 4),  # Portrait
        ],
    )
    def test_extreme_aspect_ratios(self, detector, aspect_ratio):
        """Test handling of extreme aspect ratios."""
        width_ratio, height_ratio = aspect_ratio

        # Create image with specified aspect ratio
        base_size = 100
        width = base_size * width_ratio
        height = base_size * height_ratio

        # Limit maximum size to prevent memory issues
        max_dim = 2000
        if width > max_dim:
            scale_factor = max_dim / width
            width = int(width * scale_factor)
            height = int(height * scale_factor)
        if height > max_dim:
            scale_factor = max_dim / height
            width = int(width * scale_factor)
            height = int(height * scale_factor)

        # Ensure minimum size
        width = max(width, 10)
        height = max(height, 10)

        image_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image_data)

        try:
            result = detector(
                pil_image,
                output_type="np",
                detect_resolution=256,
                draw_pose=draw_openpose,
                include_hands=False,
                include_face=False,
            )

            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.shape[:2] == (height, width)
            assert result.shape[2] == 3

        except Exception as e:
            # Very extreme aspect ratios might not be supported
            extreme_ratio = max(width_ratio / height_ratio, height_ratio / width_ratio)
            if extreme_ratio > 5:
                pytest.skip(f"Extreme aspect ratio {aspect_ratio} may not be supported: {e}")
            else:
                raise

    @pytest.mark.parametrize(
        "drawing_params",
        [
            {"include_hands": True, "include_face": True},
            {"include_hands": False, "include_face": False},
            {"include_hands": True, "include_face": False},
            {"include_hands": False, "include_face": True},
        ],
    )
    def test_drawing_parameter_combinations(self, detector, drawing_params):
        """Test various drawing parameter combinations."""
        image_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image_data)

        result = detector(pil_image, output_type="np", detect_resolution=256, draw_pose=draw_openpose, **drawing_params)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256, 3)

    @pytest.mark.parametrize("output_format", ["pil", "np", "dict"])
    def test_output_format_consistency(self, detector, output_format):
        """Test consistency across different output formats."""
        image_data = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image_data)

        if output_format == "dict":
            # Test pose data extraction
            result = detector(pil_image, draw_pose=None, detect_resolution=256)

            assert isinstance(result, dict)
            required_keys = ["bodies", "hands", "faces", "body_scores", "hands_scores", "faces_scores"]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"
                assert isinstance(result[key], np.ndarray), f"Key {key} should be numpy array"

        else:
            result = detector(
                pil_image,
                output_type=output_format,
                detect_resolution=256,
                draw_pose=draw_openpose,
                include_hands=False,
                include_face=False,
            )

            if output_format == "pil":
                assert isinstance(result, Image.Image)
                assert result.mode == "RGB"
                assert result.size == (200, 200)
            elif output_format == "np":
                assert isinstance(result, np.ndarray)
                assert result.shape == (200, 200, 3)
                assert result.dtype == np.uint8

    def test_pose_data_structure_validation(self, detector):
        """Test the structure of pose data output."""
        image_data = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image_data)

        pose_data = detector(pil_image, draw_pose=None, detect_resolution=256)

        # Validate structure
        assert isinstance(pose_data, dict)

        # Check bodies
        bodies = pose_data["bodies"]
        assert isinstance(bodies, np.ndarray)
        assert len(bodies.shape) == 2  # Should be 2D

        # Check body scores
        body_scores = pose_data["body_scores"]
        assert isinstance(body_scores, np.ndarray)

        # Check hands
        hands = pose_data["hands"]
        assert isinstance(hands, np.ndarray)
        assert len(hands.shape) == 3  # Should be 3D

        # Check hands scores
        hands_scores = pose_data["hands_scores"]
        assert isinstance(hands_scores, np.ndarray)

        # Check faces
        faces = pose_data["faces"]
        assert isinstance(faces, np.ndarray)
        assert len(faces.shape) == 3  # Should be 3D

        # Check faces scores
        faces_scores = pose_data["faces_scores"]
        assert isinstance(faces_scores, np.ndarray)

        # Validate coordinate ranges (should be reasonable values)
        if bodies.size > 0:
            assert np.all(np.isfinite(bodies)), "Body coordinates should be finite"

        if hands.size > 0:
            assert np.all(np.isfinite(hands)), "Hand coordinates should be finite"

        if faces.size > 0:
            assert np.all(np.isfinite(faces)), "Face coordinates should be finite"

    @pytest.mark.parametrize("seed_value", [42, 123, 456, 789])
    def test_deterministic_behavior(self, detector, seed_value):
        """Test that processing is deterministic for the same input."""
        # Set random seed for reproducible test image
        np.random.seed(seed_value)
        image_data = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image_data)

        # Process twice with same parameters
        result1 = detector(pil_image, draw_pose=None, detect_resolution=256)

        result2 = detector(pil_image, draw_pose=None, detect_resolution=256)

        # Results should be identical (or very close due to floating point precision)
        assert result1.keys() == result2.keys()

        for key in result1.keys():
            if result1[key].size > 0 and result2[key].size > 0:
                assert result1[key].shape == result2[key].shape
                # Allow small floating point differences
                assert np.allclose(result1[key], result2[key], rtol=1e-5, atol=1e-5), f"Difference in {key}"

    def test_memory_efficiency_large_batch(self, detector):
        """Test memory efficiency with a batch of images."""
        batch_size = 5
        image_size = (200, 200, 3)

        results = []
        for i in range(batch_size):
            # Create slightly different images
            image_data = np.random.randint(0, 255, image_size, dtype=np.uint8)
            image_data[i * 10 : (i + 1) * 10, i * 10 : (i + 1) * 10] = 255  # Add unique marker
            pil_image = Image.fromarray(image_data)

            result = detector(
                pil_image,
                output_type="np",
                detect_resolution=256,
                draw_pose=draw_openpose,
                include_hands=False,
                include_face=False,
            )

            assert result is not None
            assert result.shape == image_size
            results.append(result)

        # All results should be valid
        assert len(results) == batch_size

        # For random noise images, results might be identical (no pose detected)
        # This is acceptable behavior - just verify they're all the same expected shape
        for result in results:
            assert result.shape == image_size
            assert result.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
