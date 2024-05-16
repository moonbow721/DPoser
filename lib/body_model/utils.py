import json
import os

import numpy as np

#
# From https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
# Please see license for usage restrictions.
#
# SMPLX body poses (exclude "left_hand" and "right_hand" in SMPL, 21 local poses)
BODY_JOINT_NAMES = [
    "pelvis",  # global_orient actually
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

body_name_to_index = {name: index - 1 for index, name in enumerate(BODY_JOINT_NAMES)}


class BodyPartIndices:
    left_leg = sorted([body_name_to_index[name] for name in ["left_hip", "left_knee", "left_ankle", "left_foot"]])
    right_leg = sorted([body_name_to_index[name] for name in ["right_hip", "right_knee", "right_ankle", "right_foot"]])
    left_arm = sorted(
        [body_name_to_index[name] for name in ["left_collar", "left_shoulder", "left_elbow", "left_wrist"]])
    right_arm = sorted(
        [body_name_to_index[name] for name in ["right_collar", "right_shoulder", "right_elbow", "right_wrist"]])
    trunk = sorted(
        [body_name_to_index[name] for name in ["spine1", "spine2", "spine3", "left_shoulder", "right_shoulder"]])
    hands = sorted([body_name_to_index[name] for name in ["left_wrist", "right_wrist"]])
    legs = sorted(left_leg + right_leg)
    arms = sorted(left_arm + right_arm)
    left_body = sorted(left_leg + left_arm)
    right_body = sorted(right_leg + right_arm)

    @staticmethod
    def get_indices(part_name):
        return getattr(BodyPartIndices, part_name)


class BodySegIndices:
    current_file_path = os.path.abspath(__file__)
    segmentdata = json.load(
        open(os.path.join(os.path.dirname(current_file_path), '../data/smplx_vert_segmentation.json')))

    left_leg = sorted(list(
        set(segmentdata['leftLeg'] + segmentdata['leftUpLeg'] + segmentdata['leftFoot'] + segmentdata['leftToeBase'])))
    right_leg = sorted(list(set(
        segmentdata['rightLeg'] + segmentdata['rightUpLeg'] + segmentdata['rightFoot'] + segmentdata['rightToeBase'])))
    left_arm = sorted(list(set(segmentdata['leftArm'] + segmentdata['leftForeArm'])))
    right_arm = sorted(list(set(segmentdata['rightArm'] + segmentdata['rightForeArm'])))
    trunk = sorted(list(set(
        segmentdata['spine1'] + segmentdata['spine2'] + segmentdata['leftShoulder'] + segmentdata['rightShoulder'])))
    hands = sorted(list(set(segmentdata['leftHand'] + segmentdata['rightHand'])))
    legs = sorted(list(set(left_leg + right_leg)))
    arms = sorted(list(set(left_arm + right_arm)))


# MANO right hand poses (15+1 poses)
HAND_JOINT_NAMES = [
    "right_wrist",  # global_orient actually
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
]

hand_name_to_index = {name: index - 1 for index, name in enumerate(HAND_JOINT_NAMES)}


class HandPartIndices:
    thumb = sorted([hand_name_to_index[name] for name in ['right_thumb1', 'right_thumb2', 'right_thumb3']])
    index_finger = sorted([hand_name_to_index[name] for name in ['right_index1', 'right_index2', 'right_index3']])
    middle_finger = sorted([hand_name_to_index[name] for name in ['right_middle1', 'right_middle2', 'right_middle3']])
    ring_finger = sorted([hand_name_to_index[name] for name in ['right_ring1', 'right_ring2', 'right_ring3']])
    pinky_finger = sorted([hand_name_to_index[name] for name in ['right_pinky1', 'right_pinky2', 'right_pinky3']])

    @staticmethod
    def get_indices(part_name):
        # e.g. part_name = 'index_middle_finger'
        finger_names = part_name.split('_')
        indices = []

        for finger in finger_names:
            if finger == 'index':
                indices.extend(HandPartIndices.index_finger)
            elif finger == 'middle':
                indices.extend(HandPartIndices.middle_finger)
            elif finger == 'pinky':
                indices.extend(HandPartIndices.pinky_finger)
            elif finger == 'ring':
                indices.extend(HandPartIndices.ring_finger)
            elif finger == 'thumb':
                indices.extend(HandPartIndices.thumb)

        return sorted(set(indices))


def get_smpl_skeleton():
    return np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 7],
            [5, 8],
            [6, 9],
            [7, 10],
            [8, 11],
            [9, 12],
            [9, 13],
            [9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21]
        ]
    )


def get_openpose_skeleton():
    return np.array(
        [
            [0, 1],
            [1, 2],
            [1, 5],
            [2, 3],
            [3, 4],
            [5, 6],
            [6, 7],
            [1, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 24],
            [11, 22],
            [22, 23],
            [8, 12],
            [12, 13],
            [13, 14],
            [14, 21],
            [14, 19],
            [19, 20],
            [0, 15],
            [0, 16],
            [15, 17],
            [16, 18]
        ]
    )


def get_mano_skeleton():
    skeleton = []
    wrist_index = 0

    for finger_start in range(1, len(HAND_JOINT_NAMES), 3):
        skeleton.append([wrist_index, finger_start])
        skeleton.append([finger_start, finger_start + 1])
        skeleton.append([finger_start + 1, finger_start + 2])

    return np.array(skeleton)


def get_openpose_hand_skeleton():
    skeleton = []
    wrist_index = 0

    for finger_start in range(1, 21, 4):
        skeleton.append([wrist_index, finger_start])
        skeleton.append([finger_start, finger_start + 1])
        skeleton.append([finger_start + 1, finger_start + 2])
        skeleton.append([finger_start + 2, finger_start + 3])

    return np.array(skeleton)


def get_openpose_face_skeleton():
    skeleton = []
    current_index = 0

    # Face contour
    face_contour = 17
    for i in range(face_contour - 1):
        skeleton.append([current_index + i, current_index + i + 1])

    current_index += face_contour

    # Left eyebrow
    left_eyebrow = 5
    for i in range(left_eyebrow - 1):
        skeleton.append([current_index + i, current_index + i + 1])

    current_index += left_eyebrow

    # Right eyebrow
    right_eyebrow = 5
    for i in range(right_eyebrow - 1):
        skeleton.append([current_index + i, current_index + i + 1])

    current_index += right_eyebrow

    # Nose bridge to nose tip
    nose = 4
    for i in range(nose - 1):
        skeleton.append([current_index + i, current_index + i + 1])

    current_index += nose

    # upper lip
    lip = 5
    for i in range(lip - 1):
        skeleton.append([current_index + i, current_index + i + 1])

    current_index += lip

    # Left eye, closed contour
    left_eye = 6
    for i in range(left_eye):
        skeleton.append([current_index + i, current_index + (i + 1) % left_eye])

    current_index += left_eye

    # Right eye, closed contour
    right_eye = 6
    for i in range(right_eye):
        skeleton.append([current_index + i, current_index + (i + 1) % right_eye])

    current_index += right_eye

    # Outer mouth, closed contour
    outer_mouth = 12
    for i in range(outer_mouth):
        skeleton.append([current_index + i, current_index + (i + 1) % outer_mouth])

    current_index += outer_mouth

    # Inner mouth, closed contour
    inner_mouth = 8
    for i in range(inner_mouth):
        skeleton.append([current_index + i, current_index + (i + 1) % inner_mouth])

    return np.array(skeleton)


def merge_skeleton(skeletons):
    """
    Merges multiple skeletons by adjusting the indices of keypoints in subsequent skeletons
    to ensure continuity and correctness of the merged skeleton.

    :param skeletons: A list of skeletons, where each skeleton is an array of [pointA, pointB] pairs.
    :return: A merged skeleton as an array of [pointA, pointB] pairs.
    """
    merged_skeleton = []
    offset = 0  # Initialize offset to adjust keypoints indices for each skeleton

    for skeleton in skeletons:
        # Adjust each skeleton's keypoints indices by the current offset
        adjusted_skeleton = [[point[0] + offset, point[1] + offset] for point in skeleton]
        # Update the merged skeleton
        merged_skeleton.extend(adjusted_skeleton)
        # Update the offset for the next skeleton, assuming the last keypoint index in the current skeleton
        # represents the highest value
        if skeleton.size > 0:  # Check if skeleton is not empty
            offset += max(skeleton.max(), 0) + 1  # Adjust offset based on the highest keypoint index in the current skeleton

    return np.array(merged_skeleton)

