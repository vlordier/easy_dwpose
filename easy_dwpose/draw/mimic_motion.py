"""
Reference drawing function from the MimicMotion
https://github.com/Tencent/MimicMotion/blob/main/mimicmotion/dwpose/util.py
"""

import math

import cv2
import matplotlib.colors
import numpy as np

eps = 0.01


def alpha_blend_color(color, alpha):
    """blend color according to point conf"""
    return [int(c * alpha) for c in color]


def draw_bodypose(canvas, candidate, subset, score):
    canvas_height, canvas_width, canvas_channels = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for limb_index in range(17):
        for person_index in range(len(subset)):
            joint_indices = subset[person_index][np.array(limbSeq[limb_index]) - 1]
            joint_confidences = score[person_index][np.array(limbSeq[limb_index]) - 1]
            if joint_confidences[0] < 0.3 or joint_confidences[1] < 0.3:
                continue
            y_coordinates = candidate[joint_indices.astype(int), 0] * float(canvas_width)
            x_coordinates = candidate[joint_indices.astype(int), 1] * float(canvas_height)
            center_x = np.mean(x_coordinates)
            center_y = np.mean(y_coordinates)
            limb_length = (
                (x_coordinates[0] - x_coordinates[1]) ** 2 + (y_coordinates[0] - y_coordinates[1]) ** 2
            ) ** 0.5
            limb_angle = math.degrees(
                math.atan2(x_coordinates[0] - x_coordinates[1], y_coordinates[0] - y_coordinates[1])
            )
            limb_polygon = cv2.ellipse2Poly(
                (int(center_y), int(center_x)), (int(limb_length / 2), stickwidth), int(limb_angle), 0, 360, 1
            )
            cv2.fillConvexPoly(
                canvas, limb_polygon, alpha_blend_color(colors[limb_index], joint_confidences[0] * joint_confidences[1])
            )

    canvas = (canvas * 0.6).astype(np.uint8)

    for keypoint_index in range(18):
        for person_index in range(len(subset)):
            joint_index = int(subset[person_index][keypoint_index])
            if joint_index == -1:
                continue
            x_coord, y_coord = candidate[joint_index][0:2]
            confidence = score[person_index][keypoint_index]
            pixel_x = int(x_coord * canvas_width)
            pixel_y = int(y_coord * canvas_height)
            cv2.circle(
                canvas,
                (int(pixel_x), int(pixel_y)),
                4,
                alpha_blend_color(colors[keypoint_index], confidence),
                thickness=-1,
            )

    return canvas


def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
    canvas_height, canvas_width, canvas_channels = canvas.shape

    hand_edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for hand_peaks, hand_scores in zip(all_hand_peaks, all_hand_scores):
        for edge_index, edge_connection in enumerate(hand_edges):
            x1_coord, y1_coord = hand_peaks[edge_connection[0]]
            x2_coord, y2_coord = hand_peaks[edge_connection[1]]
            x1_pixel = int(x1_coord * canvas_width)
            y1_pixel = int(y1_coord * canvas_height)
            x2_pixel = int(x2_coord * canvas_width)
            y2_pixel = int(y2_coord * canvas_height)
            edge_score = int(hand_scores[edge_connection[0]] * hand_scores[edge_connection[1]] * 255)
            if x1_pixel > eps and y1_pixel > eps and x2_pixel > eps and y2_pixel > eps:
                cv2.line(
                    canvas,
                    (x1_pixel, y1_pixel),
                    (x2_pixel, y2_pixel),
                    matplotlib.colors.hsv_to_rgb([edge_index / float(len(hand_edges)), 1.0, 1.0]) * edge_score,
                    thickness=2,
                )

        for keypoint_index, keypoint_coords in enumerate(hand_peaks):
            x_coord, y_coord = keypoint_coords
            x_pixel = int(x_coord * canvas_width)
            y_pixel = int(y_coord * canvas_height)
            keypoint_score = int(hand_scores[keypoint_index] * 255)
            if x_pixel > eps and y_pixel > eps:
                cv2.circle(canvas, (x_pixel, y_pixel), 4, (0, 0, keypoint_score), thickness=-1)
    return canvas


def draw_facepose(canvas, all_face_landmarks, all_face_scores):
    canvas_height, canvas_width, canvas_channels = canvas.shape
    for face_landmarks, face_scores in zip(all_face_landmarks, all_face_scores):
        for landmark_coords, landmark_score in zip(face_landmarks, face_scores):
            x_coord, y_coord = landmark_coords
            x_pixel = int(x_coord * canvas_width)
            y_pixel = int(y_coord * canvas_height)
            confidence_value = int(landmark_score * 255)
            if x_pixel > eps and y_pixel > eps:
                cv2.circle(
                    canvas, (x_pixel, y_pixel), 3, (confidence_value, confidence_value, confidence_value), thickness=-1
                )
    return canvas


def draw_pose(pose, height, width, ref_w=2160):
    """vis dwpose outputs

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    bodies = pose["bodies"]
    body_scores = pose["body_scores"]
    # candidate = bodies['candidate']
    # subset = bodies['subset']
    faces = pose["faces"]
    hands = pose["hands"]

    sz = min(height, width)
    sr = (ref_w / sz) if sz != ref_w else 1

    ########################################## create zero canvas ##################################################
    canvas = np.zeros(shape=(int(height * sr), int(width * sr), 3), dtype=np.uint8)

    ########################################### draw body pose #####################################################
    canvas = draw_bodypose(canvas, bodies, body_scores, score=body_scores)

    ########################################### draw hand pose #####################################################
    canvas = draw_handpose(canvas, hands, pose["hands_scores"])

    ########################################### draw face pose #####################################################
    canvas = draw_facepose(canvas, faces, pose["faces_scores"])

    return cv2.cvtColor(cv2.resize(canvas, (width, height)), cv2.COLOR_BGR2RGB)
