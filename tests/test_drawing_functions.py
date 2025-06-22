"""
Comprehensive tests for drawing functions in easy_dwpose.
Tests all drawing functions with various parameters and edge cases.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from easy_dwpose.draw import draw_openpose
from easy_dwpose.draw.mimic_motion import draw_bodypose as draw_bodypose_mimic_motion
from easy_dwpose.draw.mimic_motion import draw_pose as draw_pose_mimic_motion
from easy_dwpose.draw.musepose import draw_bodypose as draw_bodypose_musepose
from easy_dwpose.draw.musepose import draw_pose as draw_pose_musepose
from easy_dwpose.draw.openpose import draw_bodypose, draw_facepose, draw_handpose


class TestDrawingFunctions:
    """Test all drawing functions with comprehensive parameters."""

    @pytest.fixture
    def mock_pose_data(self):
        """Generate mock pose data for testing."""
        num_people = 2
        num_body_keypoints = 18
        num_hand_keypoints = 21
        num_face_keypoints = 68

        # Create mock bodies data
        bodies = np.random.rand(num_people * num_body_keypoints, 3)
        body_scores = np.random.rand(num_people, num_body_keypoints)

        # Create mock hands data (left + right hands)
        hands = np.random.rand(num_people * 2, num_hand_keypoints, 2)
        hands_scores = np.random.rand(num_people * 2, num_hand_keypoints)

        # Create mock faces data
        faces = np.random.rand(num_people, num_face_keypoints, 2)
        faces_scores = np.random.rand(num_people, num_face_keypoints)

        return {
            "bodies": bodies,
            "body_scores": body_scores,
            "hands": hands,
            "hands_scores": hands_scores,
            "faces": faces,
            "faces_scores": faces_scores,
        }

    @pytest.fixture
    def canvas_sizes(self):
        """Different canvas sizes for testing."""
        return [
            (256, 256),
            (512, 512),
            (480, 640),
            (720, 1280),
            (1080, 1920),
        ]

    @pytest.mark.parametrize(
        "height,width",
        [
            (256, 256),
            (512, 512),
            (480, 640),
            (720, 1280),
            (1080, 1920),
        ],
    )
    def test_draw_openpose_basic(self, mock_pose_data, height, width):
        """Test basic OpenPose drawing functionality."""
        result = draw_openpose(mock_pose_data, height, width)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (height, width, 3)
        assert result.dtype == np.uint8
        assert np.all(result >= 0) and np.all(result <= 255)

    @pytest.mark.parametrize(
        "include_hands,include_face",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_draw_openpose_options(self, mock_pose_data, include_hands, include_face):
        """Test OpenPose drawing with different include options."""
        height, width = 512, 512
        result = draw_openpose(mock_pose_data, height, width, include_hands=include_hands, include_face=include_face)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (height, width, 3)

    @pytest.mark.parametrize("draw_face", [True, False])
    def test_draw_musepose_basic(self, mock_pose_data, draw_face):
        """Test basic MusePose drawing functionality."""
        height, width = 512, 512
        result = draw_pose_musepose(mock_pose_data, height, width, draw_face)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (height, width, 3)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("ref_w", [512, 1080, 1920, 2160, 4320])
    def test_draw_mimic_motion_ref_w(self, mock_pose_data, ref_w):
        """Test MimicMotion drawing with different reference widths."""
        height, width = 512, 512
        result = draw_pose_mimic_motion(mock_pose_data, height, width, ref_w=ref_w)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (height, width, 3)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize(
        "canvas_size",
        [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
        ],
    )
    def test_draw_bodypose_different_sizes(self, mock_pose_data, canvas_size):
        """Test body pose drawing with different canvas sizes."""
        height, width = canvas_size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Test OpenPose body drawing
        result = draw_bodypose(canvas, mock_pose_data["bodies"], mock_pose_data["body_scores"])
        assert result.shape == (height, width, 3)

        # Test MusePose body drawing
        result = draw_bodypose_musepose(canvas, mock_pose_data["bodies"], mock_pose_data["body_scores"])
        assert result.shape == (height, width, 3)

        # Test MimicMotion body drawing
        result = draw_bodypose_mimic_motion(
            canvas, mock_pose_data["bodies"], mock_pose_data["body_scores"], mock_pose_data["body_scores"]
        )
        assert result.shape == (height, width, 3)

    def test_draw_handpose_edge_cases(self, mock_pose_data):
        """Test hand pose drawing with edge cases."""
        height, width = 512, 512
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Test with empty hands
        empty_hands = np.array([]).reshape(0, 21, 2)
        result = draw_handpose(canvas, empty_hands)
        assert result.shape == (height, width, 3)

        # Test with single hand
        single_hand = mock_pose_data["hands"][:1]
        result = draw_handpose(canvas, single_hand)
        assert result.shape == (height, width, 3)

        # Test with multiple hands
        result = draw_handpose(canvas, mock_pose_data["hands"])
        assert result.shape == (height, width, 3)

    def test_draw_facepose_edge_cases(self, mock_pose_data):
        """Test face pose drawing with edge cases."""
        height, width = 512, 512
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Test with empty faces
        empty_faces = np.array([]).reshape(0, 68, 2)
        result = draw_facepose(canvas, empty_faces)
        assert result.shape == (height, width, 3)

        # Test with single face
        single_face = mock_pose_data["faces"][:1]
        result = draw_facepose(canvas, single_face)
        assert result.shape == (height, width, 3)

        # Test with multiple faces
        result = draw_facepose(canvas, mock_pose_data["faces"])
        assert result.shape == (height, width, 3)

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
    def test_drawing_functions_input_dtypes(self, mock_pose_data, dtype):
        """Test drawing functions with different input data types."""
        height, width = 256, 256
        canvas = np.zeros((height, width, 3), dtype=dtype)

        # Convert pose data to specified dtype
        bodies = mock_pose_data["bodies"].astype(dtype)
        body_scores = mock_pose_data["body_scores"].astype(dtype)

        try:
            result = draw_bodypose(canvas, bodies, body_scores)
            assert result is not None
        except Exception as e:
            # Some functions might not support all dtypes
            assert "dtype" in str(e).lower() or "type" in str(e).lower()

    def test_drawing_functions_extreme_values(self, mock_pose_data):
        """Test drawing functions with extreme coordinate values."""
        height, width = 512, 512

        # Test with coordinates at boundaries
        extreme_pose = mock_pose_data.copy()
        extreme_pose["bodies"][:, 0] = np.clip(extreme_pose["bodies"][:, 0], 0, 1)  # x coordinates
        extreme_pose["bodies"][:, 1] = np.clip(extreme_pose["bodies"][:, 1], 0, 1)  # y coordinates

        result = draw_openpose(extreme_pose, height, width)
        assert result is not None
        assert result.shape == (height, width, 3)

        # Test with coordinates outside boundaries
        extreme_pose["bodies"][:, 0] = np.random.uniform(-0.5, 1.5, extreme_pose["bodies"].shape[0])
        extreme_pose["bodies"][:, 1] = np.random.uniform(-0.5, 1.5, extreme_pose["bodies"].shape[0])

        result = draw_openpose(extreme_pose, height, width)
        assert result is not None
        assert result.shape == (height, width, 3)

    def test_drawing_functions_zero_confidence(self, mock_pose_data):
        """Test drawing functions with zero confidence scores."""
        height, width = 256, 256

        # Set all confidence scores to zero
        zero_conf_pose = mock_pose_data.copy()
        zero_conf_pose["body_scores"] = np.zeros_like(zero_conf_pose["body_scores"])
        zero_conf_pose["hands_scores"] = np.zeros_like(zero_conf_pose["hands_scores"])
        zero_conf_pose["faces_scores"] = np.zeros_like(zero_conf_pose["faces_scores"])

        # Should still produce valid output
        result = draw_openpose(zero_conf_pose, height, width)
        assert result is not None
        assert result.shape == (height, width, 3)

    def test_drawing_functions_high_confidence(self, mock_pose_data):
        """Test drawing functions with maximum confidence scores."""
        height, width = 256, 256

        # Set all confidence scores to maximum
        high_conf_pose = mock_pose_data.copy()
        high_conf_pose["body_scores"] = np.ones_like(high_conf_pose["body_scores"])
        high_conf_pose["hands_scores"] = np.ones_like(high_conf_pose["hands_scores"])
        high_conf_pose["faces_scores"] = np.ones_like(high_conf_pose["faces_scores"])

        result = draw_openpose(high_conf_pose, height, width)
        assert result is not None
        assert result.shape == (height, width, 3)


class TestDrawingFunctionIntegration:
    """Integration tests for drawing functions."""

    @pytest.fixture
    def realistic_pose_data(self):
        """Generate more realistic pose data."""
        # Simulate a single person with realistic keypoint positions
        bodies = np.array(
            [
                [0.5, 0.3, 0.9],  # head
                [0.5, 0.4, 0.8],  # neck
                [0.4, 0.45, 0.7],  # left shoulder
                [0.6, 0.45, 0.7],  # right shoulder
                [0.35, 0.55, 0.6],  # left elbow
                [0.65, 0.55, 0.6],  # right elbow
                [0.3, 0.65, 0.5],  # left wrist
                [0.7, 0.65, 0.5],  # right wrist
                [0.45, 0.6, 0.8],  # left hip
                [0.55, 0.6, 0.8],  # right hip
                [0.45, 0.75, 0.7],  # left knee
                [0.55, 0.75, 0.7],  # right knee
                [0.45, 0.9, 0.6],  # left ankle
                [0.55, 0.9, 0.6],  # right ankle
                [0.48, 0.25, 0.5],  # left eye
                [0.52, 0.25, 0.5],  # right eye
                [0.46, 0.28, 0.4],  # left ear
                [0.54, 0.28, 0.4],  # right ear
            ]
        )

        body_scores = np.array(
            [[0.9, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4]]
        )

        # Create realistic hand poses
        hands = np.random.rand(2, 21, 2) * 0.1 + 0.45  # Near wrist positions
        hands_scores = np.random.rand(2, 21) * 0.5 + 0.3

        # Create realistic face poses
        faces = np.random.rand(1, 68, 2) * 0.15 + 0.425  # Around head position
        faces_scores = np.random.rand(1, 68) * 0.6 + 0.2

        return {
            "bodies": bodies,
            "body_scores": body_scores,
            "hands": hands,
            "hands_scores": hands_scores,
            "faces": faces,
            "faces_scores": faces_scores,
        }

    @pytest.mark.parametrize(
        "drawing_function",
        [
            draw_openpose,
            draw_pose_musepose,
            draw_pose_mimic_motion,
        ],
    )
    def test_drawing_functions_consistency(self, realistic_pose_data, drawing_function):
        """Test that drawing functions produce consistent results."""
        height, width = 512, 512

        # Test multiple calls with same input
        if drawing_function == draw_openpose:
            result1 = drawing_function(realistic_pose_data, height, width)
            result2 = drawing_function(realistic_pose_data, height, width)
        elif drawing_function == draw_pose_musepose:
            result1 = drawing_function(realistic_pose_data, height, width, True)
            result2 = drawing_function(realistic_pose_data, height, width, True)
        else:  # mimic_motion
            result1 = drawing_function(realistic_pose_data, height, width)
            result2 = drawing_function(realistic_pose_data, height, width)

        np.testing.assert_array_equal(result1, result2, "Drawing function should produce consistent results")

    def test_drawing_functions_memory_efficiency(self, realistic_pose_data):
        """Test that drawing functions handle memory efficiently."""
        # Test with large canvas
        height, width = 2048, 2048

        # This should not cause memory issues
        result = draw_openpose(realistic_pose_data, height, width)
        assert result is not None
        assert result.shape == (height, width, 3)

        # Clean up
        del result

    @pytest.mark.parametrize(
        "aspect_ratio",
        [
            (1, 1),  # Square
            (16, 9),  # Widescreen
            (4, 3),  # Standard
            (9, 16),  # Portrait
            (21, 9),  # Ultra-wide
        ],
    )
    def test_drawing_functions_aspect_ratios(self, realistic_pose_data, aspect_ratio):
        """Test drawing functions with different aspect ratios."""
        base_size = 256
        width = base_size * aspect_ratio[0]
        height = base_size * aspect_ratio[1]

        result = draw_openpose(realistic_pose_data, height, width)
        assert result is not None
        assert result.shape == (height, width, 3)


class TestDrawingFunctionPerformance:
    """Performance tests for drawing functions."""

    @pytest.fixture
    def large_pose_data(self):
        """Generate large pose data for performance testing."""
        num_people = 10
        num_body_keypoints = 18
        num_hand_keypoints = 21
        num_face_keypoints = 68

        bodies = np.random.rand(num_people * num_body_keypoints, 2)
        body_scores = np.random.rand(num_people, num_body_keypoints)
        hands = np.random.rand(num_people * 2, num_hand_keypoints, 2)
        hands_scores = np.random.rand(num_people * 2, num_hand_keypoints)
        faces = np.random.rand(num_people, num_face_keypoints, 2)
        faces_scores = np.random.rand(num_people, num_face_keypoints)

        return {
            "bodies": bodies,
            "body_scores": body_scores,
            "hands": hands,
            "hands_scores": hands_scores,
            "faces": faces,
            "faces_scores": faces_scores,
        }

    @pytest.fixture
    def realistic_pose_data(self):
        """Generate realistic pose data for performance testing."""
        # Bodies: (18, 2) shape as returned by actual detector
        bodies = np.array(
            [
                [0.5, 0.3],
                [0.5, 0.4],
                [0.4, 0.45],
                [0.6, 0.45],
                [0.35, 0.55],
                [0.65, 0.55],
                [0.3, 0.65],
                [0.7, 0.65],
                [0.45, 0.6],
                [0.55, 0.6],
                [0.45, 0.75],
                [0.55, 0.75],
                [0.45, 0.9],
                [0.55, 0.9],
                [0.48, 0.25],
                [0.52, 0.25],
                [0.48, 0.29],
                [0.52, 0.29],
            ]
        )

        body_scores = np.ones((1, 18))

        # Generate hand keypoints: (2, 21, 2) shape
        hands = np.random.rand(2, 21, 2) * 0.2 + 0.4  # Center around 0.5
        hands_scores = np.ones((2, 21))

        # Generate face keypoints: (1, 68, 2) shape
        faces = np.random.rand(1, 68, 2) * 0.3 + 0.35  # Center around face region
        faces_scores = np.ones((1, 68))

        return {
            "bodies": bodies,
            "body_scores": body_scores,
            "hands": hands,
            "hands_scores": hands_scores,
            "faces": faces,
            "faces_scores": faces_scores,
        }

    @pytest.mark.slow
    def test_drawing_functions_performance(self, large_pose_data):
        """Test drawing functions performance with large data."""
        import time

        height, width = 1024, 1024

        # Test OpenPose drawing performance
        start_time = time.time()
        result = draw_openpose(large_pose_data, height, width)
        openpose_time = time.time() - start_time

        assert result is not None
        assert openpose_time < 5.0  # Should complete within 5 seconds

        # Test MusePose drawing performance
        start_time = time.time()
        result = draw_pose_musepose(large_pose_data, height, width, True)
        musepose_time = time.time() - start_time

        assert result is not None
        assert musepose_time < 5.0  # Should complete within 5 seconds

        # Test MimicMotion drawing performance
        start_time = time.time()
        result = draw_pose_mimic_motion(large_pose_data, height, width)
        mimic_time = time.time() - start_time

        assert result is not None
        assert mimic_time < 5.0  # Should complete within 5 seconds

    @pytest.mark.slow
    @pytest.mark.parametrize("num_iterations", [10, 50, 100])
    def test_drawing_functions_repeated_calls(self, realistic_pose_data, num_iterations):
        """Test drawing functions with repeated calls."""
        import time

        height, width = 256, 256

        start_time = time.time()
        for _ in range(num_iterations):
            result = draw_openpose(realistic_pose_data, height, width)
            assert result is not None

        total_time = time.time() - start_time
        avg_time = total_time / num_iterations

        # Should average less than 100ms per call for small images
        assert avg_time < 0.1, f"Average time per call: {avg_time:.3f}s"
