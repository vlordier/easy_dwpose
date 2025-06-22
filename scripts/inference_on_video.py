import argparse
import json
from typing import Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from easy_dwpose import DWposeDetector


def load_video(input_path: str, max_video_len: Optional[int] = None) -> Tuple[list[np.ndarray], int]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Can't open video file {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if max_video_len and len(frames) >= max_video_len:
            break

    return frames, fps


def save_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", output_params=["-pix_fmt", "yuv420p"])

    for frame in frames:
        writer.append_data(frame)

    writer.close()


def save_pose_data_sequence(pose_data_list: list[str], output_path: str) -> None:
    """Save sequence of pose data as JSON array."""
    # Convert JSON strings back to dicts for proper array serialization
    pose_dicts = [json.loads(pose_data) for pose_data in pose_data_list]

    with open(output_path, "w") as f:
        json.dump(pose_dicts, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="assets/dance.mp4")
    parser.add_argument("--output_path", type=str, default="assets/skeleton.mp4")
    parser.add_argument("--max_video_len", type=int, default=None)
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["image", "json", "both"],
        default="image",
        help="Output type: 'image' for skeleton video, 'json' for pose data, 'both' for both",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detector = DWposeDetector(device=device)
    logger.info("Loaded detector")

    logger.info(f"Starting inference on video {args.input}")
    video, fps = load_video(args.input, args.max_video_len)

    if args.output_type in ["image", "both"]:
        result = []
        for frame in tqdm(video, desc="Processing frames for video"):
            result.append(detector(frame, output_type="np", include_hands=True, include_face=True))

        logger.info(f"Saving output video to {args.output_path}")
        save_video(result, args.output_path, fps)

    if args.output_type in ["json", "both"]:
        pose_data_list = []
        for frame in tqdm(video, desc="Processing frames for JSON"):
            pose_data = detector(frame, output_type="json", draw_pose=None)
            pose_data_list.append(pose_data)

        json_output_path = args.output_path.rsplit(".", 1)[0] + "_poses.json"
        logger.info(f"Saving pose data to {json_output_path}")
        save_pose_data_sequence(pose_data_list, json_output_path)

    logger.info("Done")
