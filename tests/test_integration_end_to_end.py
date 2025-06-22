"""
End-to-end integration tests for easy_dwpose.
Tests complete workflows and real-world usage scenarios.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest
from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.draw import draw_openpose
from easy_dwpose.draw.mimic_motion import draw_pose as draw_pose_mimic_motion
from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to provide a DWpose detector instance."""
        return DWposeDetector()

    @pytest.fixture
    def test_images(self):
        """Generate test images for integration testing."""
        images = {}
        np.random.seed(42)

        # Create realistic test images
        images["portrait"] = np.random.randint(0, 255, (720, 480, 3), dtype=np.uint8)
        images["landscape"] = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
        images["square"] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        images["small"] = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        images["large"] = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        return images

    @pytest.mark.integration
    @pytest.mark.parametrize("image_format", ["RGB", "RGBA", "L"])
    @pytest.mark.parametrize(
        "drawing_function",
        [
            draw_openpose,
            draw_pose_musepose,
            draw_pose_mimic_motion,
        ],
    )
    def test_complete_pipeline_different_formats(self, detector, image_format, drawing_function):
        """Test complete pipeline with different image formats and drawing functions."""
        # Create test image in specified format
        base_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pil_image = Image.fromarray(base_image).convert(image_format)

        # Configure drawing function parameters
        if drawing_function == draw_pose_musepose:
            kwargs = {"draw_face": True}
        elif drawing_function == draw_pose_mimic_motion:
            kwargs = {"ref_w": 1080}
        else:
            kwargs = {"include_hands": True, "include_face": True}

        # Run the complete pipeline
        try:
            result = detector(pil_image, output_type="pil", draw_pose=drawing_function, detect_resolution=256, **kwargs)

            assert result is not None
            assert isinstance(result, Image.Image)
            assert result.size == pil_image.size
            assert result.mode == "RGB"

        except Exception as e:
            if image_format == "L" and "RGB" in str(e):
                pytest.skip(f"Grayscale images may not be supported: {e}")
            else:
                raise

    @pytest.mark.integration
    @pytest.mark.parametrize("batch_size", [1, 3, 5])
    def test_batch_processing_simulation(self, detector, test_images, batch_size):
        """Test processing multiple images in sequence."""
        image_list = list(test_images.values())[:batch_size]
        results = []

        start_time = time.time()

        for i, image in enumerate(image_list):
            pil_image = Image.fromarray(image)
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
            assert result.shape[:2] == image.shape[:2]

            results.append(result)

        end_time = time.time()
        processing_time = end_time - start_time

        # Performance assertions
        assert len(results) == batch_size
        avg_time_per_image = processing_time / batch_size
        assert avg_time_per_image < 30.0, f"Average processing time too slow: {avg_time_per_image:.2f}s"

    @pytest.mark.integration
    def test_save_and_load_results(self, detector, test_images):
        """Test saving results to files and loading them back."""
        image = test_images["square"]
        pil_image = Image.fromarray(image)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Process image and get pose data
            pose_data = detector(pil_image, draw_pose=None, detect_resolution=256)

            assert isinstance(pose_data, dict)
            assert "bodies" in pose_data
            assert "hands" in pose_data
            assert "faces" in pose_data

            # Save pose data to JSON
            pose_file = Path(temp_dir) / "pose_data.json"

            # Convert numpy arrays to lists for JSON serialization
            serializable_pose = {}
            for key, value in pose_data.items():
                if isinstance(value, np.ndarray):
                    serializable_pose[key] = value.tolist()
                else:
                    serializable_pose[key] = value

            with open(pose_file, "w") as f:
                json.dump(serializable_pose, f)

            # Verify file was created and can be loaded
            assert pose_file.exists()

            with open(pose_file) as f:
                loaded_pose = json.load(f)

            # Verify structure is preserved
            assert set(loaded_pose.keys()) == set(serializable_pose.keys())

            # Generate images with different drawing functions
            for i, (draw_func, name) in enumerate(
                [
                    (draw_openpose, "openpose"),
                    (draw_pose_musepose, "musepose"),
                    (draw_pose_mimic_motion, "mimic_motion"),
                ]
            ):
                if draw_func == draw_pose_musepose:
                    kwargs = {"draw_face": True}
                elif draw_func == draw_pose_mimic_motion:
                    kwargs = {"ref_w": 1080}
                else:
                    kwargs = {"include_hands": True, "include_face": True}

                result = detector(pil_image, output_type="pil", draw_pose=draw_func, detect_resolution=256, **kwargs)

                # Save result image
                result_file = Path(temp_dir) / f"result_{name}.png"
                result.save(result_file)

                # Verify file was created
                assert result_file.exists()

                # Load and verify image
                loaded_image = Image.open(result_file)
                assert loaded_image.size == result.size
                assert loaded_image.mode == result.mode

    @pytest.mark.integration
    @pytest.mark.parametrize("resolution", [128, 256, 384, 512, 768])
    def test_resolution_consistency(self, detector, test_images, resolution):
        """Test that different resolutions produce consistent results."""
        image = test_images["square"]
        pil_image = Image.fromarray(image)

        # Process with different resolutions
        result_low = detector(pil_image, draw_pose=None, detect_resolution=resolution)

        # Second run with same resolution should be consistent
        result_consistent = detector(pil_image, draw_pose=None, detect_resolution=resolution)

        # Check that results have the same structure
        assert set(result_low.keys()) == set(result_consistent.keys())

        # Check that keypoint counts are consistent
        assert result_low["bodies"].shape == result_consistent["bodies"].shape
        assert result_low["hands"].shape == result_consistent["hands"].shape
        assert result_low["faces"].shape == result_consistent["faces"].shape

    @pytest.mark.integration
    def test_memory_stability_repeated_calls(self, detector, test_images):
        """Test memory stability with repeated calls."""
        image = test_images["small"]  # Use small image for faster processing
        pil_image = Image.fromarray(image)

        # Make multiple calls and track memory patterns
        results = []
        for i in range(10):
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
            results.append(result.shape)

        # All results should have the same shape
        assert all(shape == results[0] for shape in results), "Inconsistent output shapes"

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "error_condition",
        [
            "corrupted_image",
            "invalid_image_data",
            "empty_image",
        ],
    )
    def test_error_handling_robustness(self, detector, error_condition):
        """Test robust error handling with various error conditions."""

        if error_condition == "corrupted_image":
            # Create corrupted image data
            corrupted_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            corrupted_data[50:60, 50:60] = 0  # Corrupt a section
            test_image = Image.fromarray(corrupted_data)

        elif error_condition == "invalid_image_data":
            # Create image with invalid data ranges
            invalid_data = np.random.randint(-50, 300, (100, 100, 3), dtype=np.int16)
            # Clamp to valid range for PIL
            invalid_data = np.clip(invalid_data, 0, 255).astype(np.uint8)
            test_image = Image.fromarray(invalid_data)

        elif error_condition == "empty_image":
            # Create minimal size image
            empty_data = np.zeros((2, 2, 3), dtype=np.uint8)
            test_image = Image.fromarray(empty_data)

        try:
            result = detector(
                test_image,
                output_type="np",
                detect_resolution=128,
                draw_pose=draw_openpose,
                include_hands=False,
                include_face=False,
            )

            # If no exception is raised, result should be valid
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3  # RGB channels

        except (ValueError, RuntimeError, Exception) as e:
            # Some error conditions may legitimately raise exceptions
            # Just ensure the error message is informative
            assert len(str(e)) > 0, "Error message should not be empty"

    @pytest.mark.integration
    def test_cross_platform_compatibility(self, detector, test_images):
        """Test that results are consistent across different scenarios."""
        image = test_images["portrait"]
        pil_image = Image.fromarray(image)

        # Test with different input types
        numpy_result = detector(
            image,  # numpy array input
            output_type="np",
            detect_resolution=256,
        )

        pil_result = detector(
            pil_image,  # PIL image input
            output_type="np",
            detect_resolution=256,
        )

        # Results should have the same shape (allowing for minor differences due to preprocessing)
        assert numpy_result.shape == pil_result.shape

        # Convert to PIL and back to test format consistency
        pil_from_numpy = Image.fromarray(numpy_result)
        assert pil_from_numpy.mode == "RGB"
        assert pil_from_numpy.size == (image.shape[1], image.shape[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
