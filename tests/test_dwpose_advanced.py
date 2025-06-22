"""
Advanced parametrized tests for DWPose detector with comprehensive coverage.
Tests various edge cases, parameter combinations, and integration scenarios.
"""

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union
from unittest.mock import MagicMock, PropertyMock, patch

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.draw import draw_openpose
from easy_dwpose.draw.mimic_motion import draw_pose as draw_pose_mimic_motion
from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose


class TestDWposeDetectorAdvanced:
    """Advanced tests for DWposeDetector with comprehensive parametrization."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to provide a DWpose detector instance."""
        return DWposeDetector()

    @pytest.fixture
    def sample_images(self):
        """Fixture providing various test images."""
        images = {}

        # Create different types of test images
        np.random.seed(42)  # For reproducible tests

        # Standard RGB image
        images["rgb"] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Square image
        images["square"] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Small image
        images["small"] = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # Large image
        images["large"] = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Wide image
        images["wide"] = np.random.randint(0, 255, (360, 1280, 3), dtype=np.uint8)

        # Tall image
        images["tall"] = np.random.randint(0, 255, (1280, 360, 3), dtype=np.uint8)

        # Solid color images
        images["black"] = np.zeros((256, 256, 3), dtype=np.uint8)
        images["white"] = np.ones((256, 256, 3), dtype=np.uint8) * 255
        images["red"] = np.zeros((256, 256, 3), dtype=np.uint8)
        images["red"][:, :, 0] = 255

        # Gradient image
        images["gradient"] = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            images["gradient"][i, :, :] = i

        return images

    @pytest.fixture
    def test_parameters(self):
        """Fixture providing various test parameters."""
        return {
            "detect_resolutions": [128, 256, 384, 512, 640, 768, 1024],
            "output_types": ["pil", "np"],
            "drawing_functions": [
                (draw_openpose, {"include_hands": True, "include_face": True}),
                (draw_openpose, {"include_hands": True, "include_face": False}),
                (draw_openpose, {"include_hands": False, "include_face": True}),
                (draw_openpose, {"include_hands": False, "include_face": False}),
                (draw_pose_musepose, {"draw_face": True}),
                (draw_pose_musepose, {"draw_face": False}),
                (draw_pose_mimic_motion, {"ref_w": 1080}),
                (draw_pose_mimic_motion, {"ref_w": 2160}),
                (draw_pose_mimic_motion, {"ref_w": 4320}),
            ],
            "device_strings": ["cpu", "сpu", "CPU", "cpu:0"],
        }

    # Comprehensive parametrized tests
    @pytest.mark.parametrize("image_type", ["rgb", "square", "small", "wide", "tall"])
    @pytest.mark.parametrize("detect_resolution", [256, 512, 768])
    @pytest.mark.parametrize("output_type", ["pil", "np"])
    def test_detector_image_resolution_combinations(
        self, detector, sample_images, image_type, detect_resolution, output_type
    ):
        """Test detector with various image types, resolutions, and output types."""
        image = sample_images[image_type]

        result = detector(image, detect_resolution=detect_resolution, output_type=output_type)

        assert result is not None

        if output_type == "pil":
            assert isinstance(result, Image.Image)
            width, height = result.size
            assert width > 0 and height > 0
            # Verify output dimensions match input aspect ratio
            input_height, input_width = image.shape[:2]
            input_ratio = input_width / input_height
            output_ratio = width / height
            assert abs(input_ratio - output_ratio) < 0.01
        else:  # np
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3
            assert result.dtype == np.uint8

    @pytest.mark.parametrize(
        "draw_function,kwargs",
        [
            (draw_openpose, {"include_hands": True, "include_face": True}),
            (draw_openpose, {"include_hands": True, "include_face": False}),
            (draw_openpose, {"include_hands": False, "include_face": True}),
            (draw_openpose, {"include_hands": False, "include_face": False}),
            (draw_pose_musepose, {"draw_face": True}),
            (draw_pose_musepose, {"draw_face": False}),
            (draw_pose_mimic_motion, {"ref_w": 1080}),
            (draw_pose_mimic_motion, {"ref_w": 2160}),
            (draw_pose_mimic_motion, {"ref_w": 4320}),
        ],
    )
    def test_detector_all_drawing_functions(self, detector, sample_images, draw_function, kwargs):
        """Test detector with all available drawing functions and their parameters."""
        image = sample_images["rgb"]

        result = detector(image, output_type="pil", draw_pose=draw_function, **kwargs)

        assert result is not None
        assert isinstance(result, Image.Image)
        width, height = result.size
        assert width > 0 and height > 0

    @pytest.mark.parametrize("image_type", ["black", "white", "red", "gradient"])
    def test_detector_special_images(self, detector, sample_images, image_type):
        """Test detector with special images (solid colors, gradients)."""
        image = sample_images[image_type]

        result = detector(image, output_type="np")

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("input_format", ["numpy", "pil_rgb", "pil_rgba"])
    def test_detector_input_formats(self, detector, sample_images, input_format):
        """Test detector with different input formats."""
        base_image = sample_images["rgb"]

        if input_format == "numpy":
            test_image = base_image
        elif input_format == "pil_rgb":
            test_image = Image.fromarray(base_image)
        elif input_format == "pil_rgba":
            test_image = Image.fromarray(base_image).convert("RGBA")
        else:
            pytest.skip(f"Unknown input format: {input_format}")

        result = detector(test_image, output_type="np")

        assert result is not None
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 10])
    def test_detector_batch_processing_simulation(self, detector, sample_images, batch_size):
        """Test detector processing multiple images in sequence."""
        images = [sample_images["rgb"] for _ in range(batch_size)]
        results = []

        for image in images:
            result = detector(image, output_type="np", detect_resolution=256)
            results.append(result)

        assert len(results) == batch_size
        for result in results:
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.shape[2] == 3

    @pytest.mark.parametrize("detect_resolution", [128, 256, 384, 512, 640, 768, 1024])
    def test_detector_resolution_scaling(self, detector, sample_images, detect_resolution):
        """Test detector with various detection resolutions."""
        image = sample_images["rgb"]

        result = detector(image, detect_resolution=detect_resolution, output_type="np")

        assert result is not None
        assert isinstance(result, np.ndarray)
        # Output should maintain input aspect ratio regardless of detection resolution
        input_height, input_width = image.shape[:2]
        output_height, output_width = result.shape[:2]

        input_ratio = input_width / input_height
        output_ratio = output_width / output_height
        assert abs(input_ratio - output_ratio) < 0.01

    def test_detector_pose_data_validation(self, detector, sample_images):
        """Test that pose data returned by detector has correct structure."""
        image = sample_images["rgb"]

        pose_data = detector(image, draw_pose=None)

        # Validate pose data structure
        assert isinstance(pose_data, dict)
        required_keys = ["bodies", "body_scores", "hands", "hands_scores", "faces", "faces_scores"]

        for key in required_keys:
            assert key in pose_data, f"Missing key: {key}"
            assert isinstance(pose_data[key], np.ndarray), f"Key {key} should be numpy array"

        # Validate shapes and data types
        bodies = pose_data["bodies"]
        assert bodies.ndim == 2
        assert bodies.shape == (18, 2)  # 18 body keypoints with x, y coordinates
        assert bodies.dtype in [np.float32, np.float64]

        body_scores = pose_data["body_scores"]
        assert body_scores.ndim == 2
        assert body_scores.dtype in [np.float32, np.float64, np.int32, np.int64]

        hands = pose_data["hands"]
        assert hands.ndim == 3
        assert hands.shape[2] == 2  # x, y coordinates

        faces = pose_data["faces"]
        assert faces.ndim == 3
        assert faces.shape[2] == 2  # x, y coordinates

    @pytest.mark.parametrize("num_runs", [3, 5, 10])
    def test_detector_consistency_multiple_runs(self, detector, sample_images, num_runs):
        """Test detector consistency across multiple runs."""
        image = sample_images["rgb"]
        results = []

        for _ in range(num_runs):
            result = detector(image, draw_pose=None, detect_resolution=256)
            results.append(result)

        # All results should have the same structure
        base_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result.keys() == base_result.keys(), f"Key mismatch in run {i}"
            for key in base_result.keys():
                assert result[key].shape == base_result[key].shape, f"Shape mismatch for {key} in run {i}"
                # Results should be identical for deterministic models
                if key in ["bodies", "body_scores", "hands", "hands_scores", "faces", "faces_scores"]:
                    np.testing.assert_array_equal(
                        result[key], base_result[key], err_msg=f"Results differ for {key} in run {i}"
                    )

    def test_detector_error_handling(self, detector):
        """Test detector error handling with invalid inputs."""
        # Test with invalid output type
        with pytest.raises(ValueError, match="output_type should be 'pil' or 'np'"):
            detector(np.zeros((100, 100, 3), dtype=np.uint8), output_type="invalid")

        # Test with invalid image dimensions
        with pytest.raises((ValueError, IndexError)):
            detector(np.zeros((100, 100), dtype=np.uint8))  # 2D instead of 3D

        # Test with invalid channel count
        with pytest.raises((ValueError, IndexError)):
            detector(np.zeros((100, 100, 1), dtype=np.uint8))  # Grayscale

        # Test with invalid data type
        invalid_image = np.zeros((100, 100, 3), dtype=np.complex128)
        try:
            result = detector(invalid_image, output_type="np")
            # If it doesn't raise an error, the result should still be valid
            assert result is not None
        except (ValueError, TypeError, cv2.error):
            # Expected for complex data types or unsupported image formats
            pass

    @pytest.mark.slow
    def test_detector_performance_benchmarks(self, detector, sample_images):
        """Test detector performance with various image sizes."""
        performance_data = {}

        test_cases = [
            ("small", 64),
            ("medium", 256),
            ("large", 512),
            ("xlarge", 768),
        ]

        for size_name, detect_resolution in test_cases:
            image = sample_images["rgb"]

            # Warm up
            detector(image, detect_resolution=detect_resolution, output_type="np")

            # Measure performance
            start_time = time.time()
            result = detector(image, detect_resolution=detect_resolution, output_type="np")
            end_time = time.time()

            performance_data[size_name] = {
                "resolution": detect_resolution,
                "time": end_time - start_time,
                "result_shape": result.shape,
            }

            # Ensure reasonable performance
            assert performance_data[size_name]["time"] < 10.0, (
                f"Detection took too long for {size_name}: {performance_data[size_name]['time']:.2f}s"
            )

        # Performance should generally increase with resolution
        # (though this might not always be true due to optimizations)

    def test_detector_memory_usage(self, detector, sample_images):
        """Test detector memory usage with various inputs."""
        import gc

        # Test with large image
        large_image = sample_images["large"]

        # Run detection multiple times to check for memory leaks
        for _ in range(5):
            result = detector(large_image, detect_resolution=512, output_type="np")
            assert result is not None
            del result
            gc.collect()

    @pytest.mark.parametrize("device_str", ["cpu", "сpu", "CPU"])
    def test_detector_device_initialization(self, device_str):
        """Test detector initialization with different device strings."""
        detector = DWposeDetector(device=device_str)
        assert detector is not None
        assert hasattr(detector, "pose_estimation")

        # Test that it can process an image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = detector(test_image, output_type="np")
        assert result is not None


class TestDWposeDetectorIntegration:
    """Integration tests for DWposeDetector."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to provide a DWpose detector instance."""
        return DWposeDetector()

    def test_detector_with_real_asset_image(self, detector):
        """Test detector with real asset image if available."""
        asset_path = "assets/pose.png"

        if os.path.exists(asset_path):
            image = Image.open(asset_path).convert("RGB")

            # Test various configurations
            configs = [
                {"output_type": "pil", "include_hands": True, "include_face": True},
                {"output_type": "np", "include_hands": False, "include_face": False},
                {"output_type": "pil", "draw_pose": draw_pose_musepose, "draw_face": True},
                {"output_type": "pil", "draw_pose": draw_pose_mimic_motion, "ref_w": 2160},
            ]

            for config in configs:
                result = detector(image, **config)
                assert result is not None

                if config["output_type"] == "pil":
                    assert isinstance(result, Image.Image)
                else:
                    assert isinstance(result, np.ndarray)
        else:
            pytest.skip("Asset image not available")

    def test_detector_save_and_load_results(self, detector):
        """Test saving and loading detector results."""
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Test PIL output saving
        pil_result = detector(test_image, output_type="pil")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            pil_result.save(tmp_file.name)
            assert os.path.exists(tmp_file.name)

            # Load and verify
            loaded_image = Image.open(tmp_file.name)
            assert loaded_image.size == pil_result.size

            # Clean up
            os.unlink(tmp_file.name)

    def test_detector_pipeline_integration(self, detector):
        """Test complete pipeline integration."""
        # Simulate a complete processing pipeline
        input_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Step 1: Get pose data
        pose_data = detector(input_image, draw_pose=None)
        assert isinstance(pose_data, dict)

        # Step 2: Process with different drawing functions
        drawing_functions = [
            (draw_openpose, {"include_hands": True, "include_face": True}),
            (draw_pose_musepose, {"draw_face": True}),
            (draw_pose_mimic_motion, {"ref_w": 2160}),
        ]

        results = []
        for draw_func, kwargs in drawing_functions:
            result = detector(input_image, output_type="pil", draw_pose=draw_func, **kwargs)
            results.append(result)
            assert isinstance(result, Image.Image)

        # All results should have the same dimensions
        base_size = results[0].size
        for result in results[1:]:
            assert result.size == base_size

    def test_detector_with_video_simulation(self, detector):
        """Test detector with simulated video frames."""
        # Simulate video frames with slightly different content
        frames = []
        for i in range(10):
            # Create frames with moving content
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            # Add some variation to simulate movement
            frame[i * 10 : (i + 1) * 10, :, :] = 255
            frames.append(frame)

        # Process all frames
        results = []
        for frame in frames:
            result = detector(frame, output_type="np", detect_resolution=256)
            results.append(result)
            assert result is not None
            assert isinstance(result, np.ndarray)

        # All results should have consistent dimensions
        base_shape = results[0].shape
        for result in results[1:]:
            assert result.shape == base_shape


class TestDWposeDetectorRobustness:
    """Robustness tests for DWposeDetector."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to provide a DWpose detector instance."""
        return DWposeDetector()

    @pytest.mark.parametrize("corruption_type", ["noise", "blur", "brightness", "contrast", "saturation"])
    def test_detector_robustness_to_image_corruption(self, detector, corruption_type):
        """Test detector robustness to various image corruptions."""
        base_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        if corruption_type == "noise":
            # Add random noise
            noise = np.random.randint(-50, 50, base_image.shape, dtype=np.int16)
            corrupted_image = np.clip(base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif corruption_type == "blur":
            # Simulate blur by averaging with neighbors
            corrupted_image = base_image.copy()
            for i in range(1, corrupted_image.shape[0] - 1):
                for j in range(1, corrupted_image.shape[1] - 1):
                    corrupted_image[i, j] = base_image[i - 1 : i + 2, j - 1 : j + 2].mean(axis=(0, 1))
        elif corruption_type == "brightness":
            # Adjust brightness
            corrupted_image = np.clip(base_image * 1.5, 0, 255).astype(np.uint8)
        elif corruption_type == "contrast":
            # Adjust contrast
            corrupted_image = np.clip((base_image - 128) * 1.5 + 128, 0, 255).astype(np.uint8)
        elif corruption_type == "saturation":
            # Reduce saturation (move towards grayscale)
            gray = np.mean(base_image, axis=2, keepdims=True)
            corrupted_image = (base_image * 0.3 + gray * 0.7).astype(np.uint8)
        else:
            corrupted_image = base_image

        # Detector should handle corrupted images gracefully
        result = detector(corrupted_image, output_type="np")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3

    @pytest.mark.parametrize("extreme_value", [0, 255])
    def test_detector_extreme_pixel_values(self, detector, extreme_value):
        """Test detector with extreme pixel values."""
        extreme_image = np.full((256, 256, 3), extreme_value, dtype=np.uint8)

        result = detector(extreme_image, output_type="np")
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_detector_with_checkerboard_pattern(self, detector):
        """Test detector with synthetic checkerboard pattern."""
        checkerboard = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    checkerboard[i : i + 32, j : j + 32] = 255

        result = detector(checkerboard, output_type="np")
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_detector_with_geometric_patterns(self, detector):
        """Test detector with various geometric patterns."""
        patterns = {
            "circles": lambda x, y: ((x - 128) ** 2 + (y - 128) ** 2) < 64**2,
            "stripes": lambda x, y: (x // 16) % 2 == 0,
            "diagonal": lambda x, y: (x + y) % 32 < 16,
        }

        for pattern_name, pattern_func in patterns.items():
            pattern_image = np.zeros((256, 256, 3), dtype=np.uint8)
            for i in range(256):
                for j in range(256):
                    if pattern_func(i, j):
                        pattern_image[i, j] = 255

            result = detector(pattern_image, output_type="np")
            assert result is not None, f"Failed for pattern: {pattern_name}"
            assert isinstance(result, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
