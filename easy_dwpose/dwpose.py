from typing import Callable, Dict, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import torch
from huggingface_hub import hf_hub_download

from easy_dwpose.body_estimation import Wholebody, resize_image
from easy_dwpose.draw import draw_openpose


class DWposeDetector:
    def __init__(self, device: str = "сpu") -> None:
        hf_hub_download("RedHash/DWPose", "yolox_l.onnx", local_dir="./checkpoints")
        hf_hub_download("RedHash/DWPose", "dw-ll_ucoco_384.onnx", local_dir="./checkpoints")
        self.pose_estimation = Wholebody(
            device=device,
            model_det="checkpoints/yolox_l.onnx",
            model_pose="checkpoints/dw-ll_ucoco_384.onnx",
        )

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

    @torch.inference_mode()
    def __call__(
        self,
        input_image: Union[PIL.Image.Image, np.ndarray],
        detect_resolution: int = 512,
        draw_pose: Optional[Callable] = draw_openpose,
        output_type: str = "pil",
        **kwargs,
    ) -> Union[PIL.Image.Image, np.ndarray, Dict]:
        if type(input_image) != np.ndarray:
            input_image = np.array(input_image.convert("RGB"))

        processed_image = input_image.copy()
        original_image_height, original_image_width, _ = processed_image.shape

        resized_image = resize_image(processed_image, target_resolution=detect_resolution)
        resized_height, resized_width, _ = resized_image.shape

        detected_keypoints, keypoint_scores = self.pose_estimation(resized_image)

        formatted_pose_data = self._format_pose(detected_keypoints, keypoint_scores, resized_width, resized_height)

        if not draw_pose:
            return formatted_pose_data

        rendered_pose_image = draw_pose(formatted_pose_data, height=resized_height, width=resized_width, **kwargs)
        final_pose_image = cv2.resize(
            rendered_pose_image, (original_image_width, original_image_height), cv2.INTER_LANCZOS4
        )

        if output_type == "pil":
            final_pose_image = PIL.Image.fromarray(final_pose_image)
        elif output_type == "np":
            pass
        else:
            raise ValueError("output_type should be 'pil' or 'np'")

        return final_pose_image
