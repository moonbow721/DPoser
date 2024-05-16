import os

import numpy as np


# We convert all other versions of joints to OpenPose 135 format
# From https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py
# Please see license for usage restrictions.
def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps SMPL to OpenPose

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'
    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))


# The OpJtr in output is already in the order of openpose
def mano_to_openpose():
    pass


# The Jtr in output is already in the order of openpose (smplx == openpose)
def flame_to_openpose():
    pass


# slicing the whole-body results to body, lhand, rhand, face
def get_openpose_part(part='body'):
    if part == 'body':
        return np.arange(0, 25)
    elif part == 'lhand':
        return np.arange(25, 25+21)
    elif part == 'rhand':
        return np.arange(25+21, 25+21+21)
    elif part == 'face':
        return np.arange(25+21+21, 25+21+21+68)
    else:
        raise ValueError(f"Invalid part: {part}")


joint_set_coco = {
    'joint_num': 135,  # body 25 (23 + pelvis + neck), lhand 21, rhand 21, face 68
    'joints_name': \
        ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
         'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck',
         'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # body part
         'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2',
         'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1',
         'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand
         'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2',
         'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1',
         'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand
         *['Face_' + str(i) for i in range(56, 73)],  # face contour
         *['Face_' + str(i) for i in range(5, 36)],  # eyebrow, nose, eyes
         *['Face_' + str(i) for i in range(36, 56)],  # outer mouth, inner mouth
         ),
    'flip_pairs': \
        ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (18, 21), (19, 22), (20, 23),
         # body part
         (24, 45), (25, 46), (26, 47), (27, 48), (28, 49), (29, 50), (30, 51), (31, 52), (32, 53), (33, 54),
         (34, 55), (35, 56), (36, 57), (37, 58), (38, 59), (39, 60), (40, 61), (41, 62), (42, 63), (43, 64),
         (44, 65),  # hand part
         (66, 82), (67, 81), (68, 80), (69, 79), (70, 78), (71, 77), (72, 76), (73, 75),  # face contour
         (83, 92), (84, 91), (85, 90), (86, 89), (87, 88),  # face eyebrow
         (97, 101), (98, 100),  # face below nose
         (102, 111), (103, 110), (104, 109), (105, 108), (106, 113), (107, 112),  # face eyes
         (114, 120), (115, 119), (116, 118), (121, 125), (122, 124),  # face mouth
         (126, 130), (127, 129), (131, 133)  # face lip
         )
}


joint_set_openpose = {
    'joint_num': 135,  # body 25, lhand 21, rhand 21, face 68
    'joints_name':
        ("Nose", "Neck", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Shoulder", "L_Elbow", "L_Wrist",
         "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "R_Eye", "L_Eye", "R_Ear", "L_Ear",
         'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # body part
         'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2',
         'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1',
         'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand
         'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2',
         'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1',
         'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand
         *['Face_' + str(i) for i in range(56, 73)],  # face contour
         *['Face_' + str(i) for i in range(5, 36)],  # eyebrow, nose, eyes
         *['Face_' + str(i) for i in range(36, 56)],  # outer mouth, inner mouth
         # 'L_Eyeball', 'R_Eyeball',  # eye ball, abandoned in practice
         ),
}


# creating index map from coco to openpose
def coco_to_openpose():
    # Initialize a list with -1 for each OpenPose keypoint
    mapping = [-1] * len(joint_set_openpose['joints_name'])

    # Loop through the COCO keypoints for every keypoints in OpenPose
    for i, openpose_keypoint_name in enumerate(joint_set_openpose['joints_name']):
        if openpose_keypoint_name in joint_set_coco['joints_name']:
            # Find the index of the COCO keypoint in the OpenPose list
            coco_index = joint_set_coco['joints_name'].index(openpose_keypoint_name)
            # Update the mapping for the found keypoint
            mapping[i] = coco_index
        else:
            # Handle the case where the COCO keypoint doesn't have a direct mapping in OpenPose
            print(f"No direct mapping for OpenPose keypoint: {openpose_keypoint_name}")

    return np.array(mapping)


def openpose_to_coco():
    return reverse_mapping(coco_to_openpose())


# processing coco whole-body results from mmpose (133 joints)
# check mmpose/mmpose/datasets/datasets/wholebody/coco_wholebody_dataset.py
def mmpose_to_openpose(keypoints, keypoint_scores):
    # array: [133, 2], [133] -> array: [135, 3]
    assert len(keypoints) == 133, f"Expected 133 keypoints, got {len(keypoints)}"
    keypoints = np.hstack((keypoints, keypoint_scores.reshape(-1, 1)))

    # Now, split this array into parts for body, feet, left hand, right hand, and face
    body_kpts = keypoints[:17]  # 17 body keypoints
    feet_kpts = keypoints[17:23]  # 6 foot keypoints
    face_kpts = keypoints[23:91]  # 68 face keypoints
    left_hand_kpts = keypoints[91:112]  # 21 left hand keypoints
    right_hand_kpts = keypoints[112:133]  # 21 right hand keypoints

    # Call merge_joint with the prepared parts
    merged_joints = merge_joint(body_kpts, feet_kpts, left_hand_kpts, right_hand_kpts, face_kpts)
    idx_mapping = coco_to_openpose()

    return merged_joints[idx_mapping]
    

def vitpose_to_openpose(keypoints):
    # array: [133, 3] -> array: [135, 3]
    assert len(keypoints) == 133, f"Expected 133 keypoints, got {len(keypoints)}"

    # Now, split this array into parts for body, feet, left hand, right hand, and face
    body_kpts = keypoints[:17]  # 17 body keypoints
    feet_kpts = keypoints[17:23]  # 6 foot keypoints
    face_kpts = keypoints[23:91]  # 68 face keypoints
    left_hand_kpts = keypoints[91:112]  # 21 left hand keypoints
    right_hand_kpts = keypoints[112:133]  # 21 right hand keypoints

    # Call merge_joint with the prepared parts
    merged_joints = merge_joint(body_kpts, feet_kpts, left_hand_kpts, right_hand_kpts, face_kpts)
    idx_mapping = coco_to_openpose()

    return merged_joints[idx_mapping]


# processing original coco whole-body annotations (133 joints)
def merge_joint(joint_img, feet_img, lhand_img, rhand_img, face_img):
    # pelvis
    lhip_idx = joint_set_coco['joints_name'].index('L_Hip')
    rhip_idx = joint_set_coco['joints_name'].index('R_Hip')
    pelvis = (joint_img[lhip_idx, :] + joint_img[rhip_idx, :]) * 0.5
    pelvis[2] = joint_img[lhip_idx, 2] * joint_img[rhip_idx, 2]  # joint_valid
    pelvis = pelvis.reshape(1, 3)

    lshoud_idx = joint_set_coco['joints_name'].index('L_Shoulder')
    rshoud_idx = joint_set_coco['joints_name'].index('R_Shoulder')
    neck = (joint_img[lshoud_idx, :] + joint_img[rshoud_idx, :]) * 0.5
    neck[2] = joint_img[lshoud_idx, 2] * joint_img[rshoud_idx, 2]  # joint_valid
    neck = neck.reshape(1, 3)

    # feet
    lfoot = feet_img[:3, :]
    rfoot = feet_img[3:, :]

    joint_img = np.concatenate((joint_img, pelvis, neck, lfoot, rfoot, lhand_img, rhand_img, face_img)).astype(
        np.float32)  # [135, 3]
    return joint_img


def reverse_mapping(original_mapping):
    reversed_mapping = np.full(original_mapping.shape, -1, dtype=int)  # Initialize with -1 for unmapped values
    for source_index, target_index in enumerate(original_mapping):
        if target_index != -1:  # Ensure that the mapping exists
            reversed_mapping[target_index] = source_index
    return reversed_mapping


# Not used
smplx_joints_name = \
    ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot',
     'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
     'R_Wrist',  # body
     'Jaw', 'L_Eye_SMPLH', 'R_Eye_SMPLH',  # SMPLH
     'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2',
     'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',  # fingers
     'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2',
     'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',  # fingers
     'Nose', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear',  # face in body
     'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # feet
     'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4',  # finger tips
     'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4',  # finger tips
     *['Face_' + str(i) for i in range(5, 56)],  # face
     *['Face_' + str(i) for i in range(56, 73)],  # face contour
     )

smplx_joint_part = {
    'body': list(range(smplx_joints_name.index('Pelvis'), smplx_joints_name.index('R_Eye_SMPLH') + 1)) + list(
        range(smplx_joints_name.index('Nose'), smplx_joints_name.index('R_Heel') + 1)),
    'lhand': list(range(smplx_joints_name.index('L_Index_1'), smplx_joints_name.index('L_Thumb_3') + 1)) + list(
        range(smplx_joints_name.index('L_Thumb_4'), smplx_joints_name.index('L_Pinky_4') + 1)),
    'rhand': list(range(smplx_joints_name.index('R_Index_1'), smplx_joints_name.index('R_Thumb_3') + 1)) + list(
        range(smplx_joints_name.index('R_Thumb_4'), smplx_joints_name.index('R_Pinky_4') + 1)),
    'face': list(range(smplx_joints_name.index('Face_5'), smplx_joints_name.index('Face_55') + 1)) + list(
        range(smplx_joints_name.index('Face_56'), smplx_joints_name.index('Face_72') + 1))}
