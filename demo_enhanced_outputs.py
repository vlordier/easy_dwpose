#!/usr/bin/env python3
"""
Demonstration script for enhanced DWPose functionality.
Shows how to get JSON outputs and/or images for hands, face, wholebody.
"""

import json

import torch
from PIL import Image

from easy_dwpose import DWposeDetector


def main():
    # Initialize detector
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    detector = DWposeDetector(device=device)

    # Load sample image
    try:
        input_image = Image.open("assets/pose.png").convert("RGB")
    except FileNotFoundError:
        import numpy as np

        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        input_image = Image.fromarray(dummy_image)

    # 1. Whole body detection with different output types

    # Get as dictionary
    pose_dict = detector.get_wholebody(input_image, output_type="dict")
    for key, value in pose_dict.items():
        pass

    # Get as JSON
    detector.get_wholebody(input_image, output_type="json")

    # Get as image
    detector.get_wholebody(input_image, output_type="pil")

    # 2. Body-only detection
    detector.get_body_only(input_image, output_type="dict")

    # 3. Hands-only detection
    detector.get_hands_only(input_image, output_type="dict")

    # 4. Face-only detection
    detector.get_face_only(input_image, output_type="dict")

    # 5. Custom combinations

    # Body + hands (no face)
    detector(input_image, output_type="dict", include_body=True, include_hands=True, include_face=False)

    # Face + hands (no body)
    detector(input_image, output_type="dict", include_body=False, include_hands=True, include_face=True)

    # 6. JSON serialization examples

    # Get body-only as JSON
    body_json = detector.get_body_only(input_image, output_type="json")
    json.loads(body_json)

    # Verify JSON can be loaded back
    hands_json = detector.get_hands_only(input_image, output_type="json")
    json.loads(hands_json)


if __name__ == "__main__":
    main()
