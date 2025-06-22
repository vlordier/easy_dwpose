import json
from typing import Any, Callable, Dict, Literal, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import torch
from huggingface_hub import hf_hub_download

from easy_dwpose.body_estimation import Wholebody, resize_image
from easy_dwpose.draw import draw_openpose


class DWposeDetector:
    def __init__(self, device: str = "cpu") -> None:
        # Validate and potentially adjust device
        device = self._validate_device(device)

        hf_hub_download("RedHash/DWPose", "yolox_l.onnx", local_dir="./checkpoints")
        hf_hub_download("RedHash/DWPose", "dw-ll_ucoco_384.onnx", local_dir="./checkpoints")
        self.pose_estimation = Wholebody(
            device=device,
            model_det="checkpoints/yolox_l.onnx",
            model_pose="checkpoints/dw-ll_ucoco_384.onnx",
        )

    def _validate_device(self, device: str) -> str:
        """Validate and potentially adjust the device string."""
        # Fix common typo: 'сpu' (Cyrillic) -> 'cpu' (Latin)
        if device == "сpu":
            device = "cpu"

        # Handle MPS availability for Apple Silicon
        if device == "mps":
            if not torch.backends.mps.is_available():
                return "cpu"

        return device

    def _format_pose(self, candidate_keypoints, keypoint_scores, image_width, image_height):
        num_persons, _, coordinate_dims = candidate_keypoints.shape

        candidate_keypoints[..., 0] /= float(image_width)
        candidate_keypoints[..., 1] /= float(image_height)

        body_keypoints = candidate_keypoints[:, :18].copy()
        body_keypoints = body_keypoints.reshape(num_persons * 18, coordinate_dims)

        body_confidence_scores = keypoint_scores[:, :18]
        for person_index in range(len(body_confidence_scores)):
            for joint_index in range(len(body_confidence_scores[person_index])):
                if body_confidence_scores[person_index][joint_index] > 0.3:
                    body_confidence_scores[person_index][joint_index] = int(18 * person_index + joint_index)
                else:
                    body_confidence_scores[person_index][joint_index] = -1

        face_keypoints = candidate_keypoints[:, 24:92]
        face_confidence_scores = keypoint_scores[:, 24:92]

        hand_keypoints = np.vstack([candidate_keypoints[:, 92:113], candidate_keypoints[:, 113:]])
        hand_confidence_scores = np.vstack([keypoint_scores[:, 92:113], keypoint_scores[:, 113:]])

        formatted_pose = dict(
            bodies=body_keypoints,
            body_scores=body_confidence_scores,
            hands=hand_keypoints,
            hands_scores=hand_confidence_scores,
            faces=face_keypoints,
            faces_scores=face_confidence_scores,
        )

        return formatted_pose

    def _filter_pose_parts(
        self,
        pose_data: Dict[str, Any],
        include_hands: bool = True,
        include_face: bool = True,
        include_body: bool = True,
    ) -> Dict[str, Any]:
        """Filter pose data to include only specified parts."""
        filtered_pose = {}

        if include_body:
            filtered_pose["bodies"] = pose_data["bodies"]
            filtered_pose["body_scores"] = pose_data["body_scores"]

        if include_hands:
            filtered_pose["hands"] = pose_data["hands"]
            filtered_pose["hands_scores"] = pose_data["hands_scores"]

        if include_face:
            filtered_pose["faces"] = pose_data["faces"]
            filtered_pose["faces_scores"] = pose_data["faces_scores"]

        return filtered_pose

    def _pose_to_json_serializable(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert pose data to JSON serializable format."""

        def convert_numpy(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        serializable_pose = {}
        for key, value in pose_data.items():
            serializable_pose[key] = convert_numpy(value)

        return serializable_pose

    def get_body_only(
        self,
        input_image: Union[PIL.Image.Image, np.ndarray],
        detect_resolution: int = 512,
        output_type: Literal["pil", "np", "json", "dict"] = "pil",
        draw_pose: Optional[Callable] = draw_openpose,
        **kwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict, str]:
        """Get pose detection results with body only."""
        return self(
            input_image=input_image,
            detect_resolution=detect_resolution,
            output_type=output_type,
            draw_pose=draw_pose,
            include_hands=False,
            include_face=False,
            include_body=True,
            **kwargs,
        )

    def get_hands_only(
        self,
        input_image: Union[PIL.Image.Image, np.ndarray],
        detect_resolution: int = 512,
        output_type: Literal["pil", "np", "json", "dict"] = "pil",
        draw_pose: Optional[Callable] = draw_openpose,
        **kwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict, str]:
        """Get pose detection results with hands only."""
        return self(
            input_image=input_image,
            detect_resolution=detect_resolution,
            output_type=output_type,
            draw_pose=draw_pose,
            include_hands=True,
            include_face=False,
            include_body=False,
            **kwargs,
        )

    def get_face_only(
        self,
        input_image: Union[PIL.Image.Image, np.ndarray],
        detect_resolution: int = 512,
        output_type: Literal["pil", "np", "json", "dict"] = "pil",
        draw_pose: Optional[Callable] = draw_openpose,
        **kwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict, str]:
        """Get pose detection results with face only."""
        return self(
            input_image=input_image,
            detect_resolution=detect_resolution,
            output_type=output_type,
            draw_pose=draw_pose,
            include_hands=False,
            include_face=True,
            include_body=False,
            **kwargs,
        )

    def get_wholebody(
        self,
        input_image: Union[PIL.Image.Image, np.ndarray],
        detect_resolution: int = 512,
        output_type: Literal["pil", "np", "json", "dict"] = "pil",
        draw_pose: Optional[Callable] = draw_openpose,
        **kwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict, str]:
        """Get pose detection results with whole body (all parts)."""
        return self(
            input_image=input_image,
            detect_resolution=detect_resolution,
            output_type=output_type,
            draw_pose=draw_pose,
            include_hands=True,
            include_face=True,
            include_body=True,
            **kwargs,
        )

    @torch.inference_mode()
    def __call__(
        self,
        input_image: Union[PIL.Image.Image, np.ndarray],
        detect_resolution: int = 512,
        draw_pose: Optional[Callable] = draw_openpose,
        output_type: Literal["pil", "np", "json", "dict"] = "pil",
        **kwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict, str]:
        """
        Detect poses in an image and return results in specified format.

        Args:
            input_image: Input image as PIL Image or numpy array
            detect_resolution: Resolution for pose detection
            draw_pose: Drawing function to use (None for data-only output)
            output_type: Output format - "pil", "np", "json", or "dict"
            **kwargs: Additional arguments including:
                include_hands (bool): Whether to include hand keypoints (default: True)
                include_face (bool): Whether to include face keypoints (default: True)
                include_body (bool): Whether to include body keypoints (default: True)
                Other arguments passed to drawing function

        Returns:
            Pose results in specified format
        """
        # Extract include parameters from kwargs with defaults
        include_hands = kwargs.pop("include_hands", True)
        include_face = kwargs.pop("include_face", True)
        include_body = kwargs.pop("include_body", True)
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image.convert("RGB"))

        processed_image = input_image.copy()
        original_image_height, original_image_width, _ = processed_image.shape

        resized_image = resize_image(processed_image, target_resolution=detect_resolution)
        resized_height, resized_width, _ = resized_image.shape

        detected_keypoints, keypoint_scores = self.pose_estimation(resized_image)

        formatted_pose_data = self._format_pose(detected_keypoints, keypoint_scores, resized_width, resized_height)

        # Filter pose data based on what parts to include
        filtered_pose_data = self._filter_pose_parts(
            formatted_pose_data, include_hands=include_hands, include_face=include_face, include_body=include_body
        )

        # Handle JSON output type
        if output_type == "json":
            serializable_pose = self._pose_to_json_serializable(filtered_pose_data)
            return json.dumps(serializable_pose, indent=2)
        elif output_type == "dict":
            return filtered_pose_data

        # If no drawing function specified, return data
        if not draw_pose:
            return filtered_pose_data

        # For image outputs, always use the full pose data for drawing,
        # but pass only the relevant include flags to the drawing function
        drawing_kwargs = kwargs.copy()  # Since we already popped the include parameters

        # Only pass include_hands and include_face to drawing function if they exist in the function signature
        if include_hands and "include_hands" in draw_pose.__code__.co_varnames:
            drawing_kwargs["include_hands"] = include_hands
        if include_face and "include_face" in draw_pose.__code__.co_varnames:
            drawing_kwargs["include_face"] = include_face

        rendered_pose_image = draw_pose(
            formatted_pose_data, height=resized_height, width=resized_width, **drawing_kwargs
        )
        final_pose_image = cv2.resize(
            rendered_pose_image, (original_image_width, original_image_height), cv2.INTER_LANCZOS4
        )

        if output_type == "pil":
            final_pose_image = PIL.Image.fromarray(final_pose_image)
        elif output_type == "np":
            pass
        else:
            raise ValueError("output_type should be 'pil', 'np', 'json', or 'dict'")

        return final_pose_image
