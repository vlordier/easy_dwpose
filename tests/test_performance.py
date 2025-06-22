"""
Performance and benchmark tests for easy_dwpose.
Tests performance, memory usage, and scalability with various configurations.
"""

import gc
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import pytest
from PIL import Image

from easy_dwpose import DWposeDetector
from easy_dwpose.draw import draw_openpose
from easy_dwpose.draw.mimic_motion import draw_pose as draw_pose_mimic_motion
from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose


@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics."""

    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    result_size: Tuple[int, ...]
    success: bool
    error_message: str = ""


class TestPerformanceBenchmarks:
    """Performance and benchmark tests for DWpose detector."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to provide a DWpose detector instance."""
        return DWposeDetector()

    @pytest.fixture
    def performance_images(self):
        """Generate images for performance testing."""
        images = {}
        np.random.seed(42)  # For reproducible results

        # Different image sizes for performance testing
        sizes = {
            "tiny": (64, 64, 3),
            "small": (256, 256, 3),
            "medium": (512, 512, 3),
            "large": (1024, 1024, 3),
            "hd": (720, 1280, 3),
            "fhd": (1080, 1920, 3),
        }

        for name, size in sizes.items():
            images[name] = np.random.randint(0, 255, size, dtype=np.uint8)

        return images

    def measure_performance(self, func, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance metrics for a function call."""
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Measure execution time
        start_time = time.time()
        cpu_times_before = process.cpu_times()

        try:
            result = func(*args, **kwargs)
            success = True
            error_message = ""
            result_size = getattr(result, "shape", (0,)) if hasattr(result, "shape") else (0,)
            if hasattr(result, "size"):  # PIL Image
                result_size = result.size
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
            result_size = (0,)

        end_time = time.time()
        cpu_times_after = process.cpu_times()

        # Calculate metrics
        execution_time = end_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory

        # CPU usage calculation (simplified)
        cpu_time_used = (cpu_times_after.user - cpu_times_before.user) + (
            cpu_times_after.system - cpu_times_before.system
        )
        cpu_usage_percent = (cpu_time_used / execution_time * 100) if execution_time > 0 else 0

        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage_percent,
            result_size=result_size,
            success=success,
            error_message=error_message,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("image_size", ["tiny", "small", "medium", "large"])
    @pytest.mark.parametrize("detect_resolution", [256, 512, 768])
    def test_detection_performance_by_size(self, detector, performance_images, image_size, detect_resolution):
        """Test detection performance with different image sizes and resolutions."""
        image = performance_images[image_size]

        # Warm up
        detector(image, detect_resolution=detect_resolution, output_type="np")

        # Measure performance
        metrics = self.measure_performance(detector, image, detect_resolution=detect_resolution, output_type="np")

        assert metrics.success, f"Detection failed: {metrics.error_message}"
        assert metrics.execution_time < 30.0, f"Detection too slow: {metrics.execution_time:.2f}s"
        assert metrics.memory_usage_mb < 1000, f"Memory usage too high: {metrics.memory_usage_mb:.2f}MB"

        # Log performance metrics

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "drawing_function",
        [
            draw_openpose,
            draw_pose_musepose,
            draw_pose_mimic_motion,
        ],
    )
    def test_drawing_performance(self, detector, performance_images, drawing_function):
        """Test performance of different drawing functions."""
        image = performance_images["medium"]

        # Test drawing performance
        if drawing_function == draw_pose_musepose:
            kwargs = {"draw_face": True}
        elif drawing_function == draw_pose_mimic_motion:
            kwargs = {"ref_w": 2160}
        else:
            kwargs = {"include_hands": True, "include_face": True}

        # Warm up
        detector(image, output_type="np", draw_pose=drawing_function, **kwargs)

        # Measure performance
        metrics = self.measure_performance(detector, image, output_type="np", draw_pose=drawing_function, **kwargs)

        assert metrics.success, f"Drawing failed: {metrics.error_message}"
        assert metrics.execution_time < 15.0, f"Drawing too slow: {metrics.execution_time:.2f}s"

    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
    def test_batch_processing_performance(self, detector, performance_images, batch_size):
        """Test performance of batch processing simulation."""
        image = performance_images["small"]

        def process_batch():
            results = []
            for _ in range(batch_size):
                result = detector(image, output_type="np", detect_resolution=256)
                results.append(result)
            return results

        # Warm up
        process_batch()

        # Measure performance
        metrics = self.measure_performance(process_batch)

        assert metrics.success, f"Batch processing failed: {metrics.error_message}"

        # Calculate per-image metrics
        time_per_image = metrics.execution_time / batch_size
        metrics.memory_usage_mb / batch_size

        assert time_per_image < 5.0, f"Per-image time too slow: {time_per_image:.2f}s"

    @pytest.mark.slow
    def test_memory_leak_detection(self, detector, performance_images):
        """Test for potential memory leaks in repeated processing."""
        image = performance_images["small"]

        # Record initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Process multiple images
        num_iterations = 20
        memory_measurements = []

        for i in range(num_iterations):
            result = detector(image, output_type="np", detect_resolution=256)
            del result
            gc.collect()

            if i % 5 == 0:  # Measure every 5 iterations
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory - initial_memory)

        # Check for memory growth
        if len(memory_measurements) > 2:
            memory_growth = memory_measurements[-1] - memory_measurements[0]
            assert memory_growth < 500, f"Potential memory leak detected: {memory_growth:.2f}MB growth"

    @pytest.mark.slow
    @pytest.mark.parametrize("num_concurrent", [1, 2, 4])
    def test_concurrent_processing_simulation(self, detector, performance_images, num_concurrent):
        """Test concurrent processing simulation."""
        image = performance_images["small"]

        def concurrent_process():
            # Simulate concurrent processing by running multiple detections
            results = []
            for _ in range(num_concurrent):
                result = detector(image, output_type="np", detect_resolution=256)
                results.append(result)
            return results

        # Measure performance
        metrics = self.measure_performance(concurrent_process)

        assert metrics.success, f"Concurrent processing failed: {metrics.error_message}"

        # Performance should not degrade linearly with concurrency
        expected_max_time = num_concurrent * 3.0  # Allow 3s per concurrent process
        assert metrics.execution_time < expected_max_time, (
            f"Concurrent processing too slow: {metrics.execution_time:.2f}s for {num_concurrent} concurrent"
        )

    @pytest.mark.slow
    def test_scalability_with_image_complexity(self, detector):
        """Test performance scalability with different image complexities."""
        complexities = {
            "simple": lambda: np.ones((512, 512, 3), dtype=np.uint8) * 128,  # Solid color
            "noise": lambda: np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),  # Random noise
            "gradient": lambda: self._create_gradient_image(512, 512),  # Gradient pattern
            "checkerboard": lambda: self._create_checkerboard_image(512, 512),  # Pattern
        }

        performance_results = {}

        for complexity_name, image_generator in complexities.items():
            image = image_generator()

            # Warm up
            detector(image, output_type="np", detect_resolution=256)

            # Measure performance
            metrics = self.measure_performance(detector, image, output_type="np", detect_resolution=256)

            performance_results[complexity_name] = metrics

            assert metrics.success, f"Detection failed for {complexity_name}: {metrics.error_message}"

        # Performance should be relatively consistent across complexities
        times = [m.execution_time for m in performance_results.values()]
        time_variance = max(times) - min(times)
        assert time_variance < 10.0, f"Performance varies too much with complexity: {time_variance:.2f}s"

    def _create_gradient_image(self, height: int, width: int) -> np.ndarray:
        """Create a gradient test image."""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                image[i, j, 0] = int(255 * i / height)  # Red gradient
                image[i, j, 1] = int(255 * j / width)  # Green gradient
                image[i, j, 2] = int(255 * (i + j) / (height + width))  # Blue gradient
        return image

    def _create_checkerboard_image(self, height: int, width: int, square_size: int = 32) -> np.ndarray:
        """Create a checkerboard test image."""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i : i + square_size, j : j + square_size] = 255
        return image


class TestResourceUsage:
    """Test resource usage and efficiency."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to provide a DWpose detector instance."""
        return DWposeDetector()

    @pytest.mark.slow
    def test_gpu_memory_usage(self, detector):
        """Test GPU memory usage if available."""
        try:
            import torch

            if torch.cuda.is_available():
                # Test with GPU device
                gpu_detector = DWposeDetector(device="cuda:0")
                test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

                # Monitor GPU memory
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()

                result = gpu_detector(test_image, output_type="np")

                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = (peak_memory - initial_memory) / 1024 / 1024  # MB

                assert result is not None
                assert memory_used < 2000, f"GPU memory usage too high: {memory_used:.2f}MB"

                # Clean up
                torch.cuda.empty_cache()
            else:
                pytest.skip("CUDA not available")
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_cpu_efficiency(self, detector):
        """Test CPU efficiency with monitoring."""
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        process = psutil.Process(os.getpid())

        # Monitor CPU usage during detection
        process.cpu_percent()
        start_time = time.time()

        result = detector(test_image, output_type="np", detect_resolution=256)

        end_time = time.time()
        process.cpu_percent()

        execution_time = end_time - start_time

        assert result is not None
        assert execution_time < 10.0, f"CPU processing too slow: {execution_time:.2f}s"

    @pytest.mark.slow
    def test_memory_efficiency_large_images(self, detector):
        """Test memory efficiency with large images."""
        # Test with progressively larger images
        sizes = [(512, 512), (1024, 1024), (1536, 1536)]

        process = psutil.Process(os.getpid())

        for width, height in sizes:
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Create large test image
            large_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            result = detector(large_image, output_type="np", detect_resolution=512)

            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_used = peak_memory - initial_memory

            assert result is not None
            # Memory usage should not grow excessively with image size
            assert memory_used < 1000, f"Memory usage too high for {width}x{height}: {memory_used:.2f}MB"

            # Clean up
            del large_image, result
            gc.collect()


class TestScalabilityBenchmarks:
    """Scalability and stress tests."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture to provide a DWpose detector instance."""
        return DWposeDetector()

    @pytest.mark.slow
    @pytest.mark.parametrize("stress_level", ["light", "medium", "heavy"])
    def test_stress_testing(self, detector, stress_level):
        """Test system under various stress levels."""
        stress_configs = {
            "light": {"num_images": 10, "image_size": (256, 256), "detect_res": 256},
            "medium": {"num_images": 50, "image_size": (512, 512), "detect_res": 512},
            "heavy": {"num_images": 100, "image_size": (768, 768), "detect_res": 768},
        }

        config = stress_configs[stress_level]

        # Generate test images
        images = []
        for i in range(config["num_images"]):
            img = np.random.randint(0, 255, (*config["image_size"], 3), dtype=np.uint8)
            images.append(img)

        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        successful_detections = 0
        failed_detections = 0

        for i, image in enumerate(images):
            try:
                result = detector(image, output_type="np", detect_resolution=config["detect_res"])
                if result is not None:
                    successful_detections += 1
                else:
                    failed_detections += 1
            except Exception:
                failed_detections += 1

            # Monitor memory every 10 images
            if (i + 1) % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                assert memory_growth < 2000, f"Memory growth too high: {memory_growth:.2f}MB"

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertions
        success_rate = successful_detections / config["num_images"]
        assert success_rate > 0.95, f"Success rate too low: {success_rate:.2%}"

        avg_time_per_image = total_time / config["num_images"]
        assert avg_time_per_image < 10.0, f"Average time per image too high: {avg_time_per_image:.2f}s"

    @pytest.mark.slow
    def test_long_running_stability(self, detector):
        """Test long-running stability and consistency."""
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Run for extended period
        num_iterations = 100
        execution_times = []

        for i in range(num_iterations):
            start_time = time.time()
            result = detector(test_image, output_type="np", detect_resolution=256)
            end_time = time.time()

            assert result is not None, f"Detection failed at iteration {i}"

            execution_time = end_time - start_time
            execution_times.append(execution_time)

            # Check for performance degradation
            if i > 10:  # After warm-up
                recent_avg = np.mean(execution_times[-5:])
                initial_avg = np.mean(execution_times[5:15])
                degradation = (recent_avg - initial_avg) / initial_avg

                assert degradation < 0.5, f"Performance degraded by {degradation:.1%} at iteration {i}"

        # Overall stability metrics
        avg_time = np.mean(execution_times[10:])  # Exclude warm-up
        std_time = np.std(execution_times[10:])

        # Performance should be consistent
        assert std_time / avg_time < 0.3, "Performance too inconsistent over time"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
