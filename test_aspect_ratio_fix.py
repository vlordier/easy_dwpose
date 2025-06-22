#!/usr/bin/env python3
"""
Simple test to verify that the aspect ratio fix is working correctly.
"""

import numpy as np

from easy_dwpose.body_estimation.pose import _fix_aspect_ratio


def test_aspect_ratio_fix():
    """Test the _fix_aspect_ratio function with various scenarios."""

    # Test case 1: Square bbox, target aspect ratio 3:4 (0.75)
    square_scale = np.array([100, 100])  # w, h
    target_aspect_ratio = 192 / 256  # 0.75 (typical RTMPose model input)

    _fix_aspect_ratio(square_scale, target_aspect_ratio)

    # Test case 2: Wide bbox, target aspect ratio 3:4
    wide_scale = np.array([200, 100])  # w, h
    _fix_aspect_ratio(wide_scale, target_aspect_ratio)

    # Test case 3: Tall bbox, target aspect ratio 3:4
    tall_scale = np.array([100, 200])  # w, h
    _fix_aspect_ratio(tall_scale, target_aspect_ratio)


if __name__ == "__main__":
    test_aspect_ratio_fix()
