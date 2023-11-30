import numpy as np
import torch
import torchgeometry as tgm
from torch.nn import functional as F


# ======================== 3D =======================================

def world_to_camera_frame(P, R, T):
    """
    Convert points from world to camera coordinates

    Args
        P: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        X_cam: Nx3 3d points in camera coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.dot(P.T - T)  # rotate and translate

    return X_cam.T


def camera_to_world_frame(P, R, T):
    """Inverse of world_to_camera_frame

    Args
        P: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        X_cam: Nx3 points in world coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.T.dot(P.T) + T  # rotate and translate

    return X_cam.T


def procrustes(A, B, scaling=True, reflection='best'):
    """ A port of MATLAB's `procrustes` function to Numpy.

    $$ \min_{R, T, S} \sum_i^N || A_i - R B_i + T ||^2. $$
    Use notation from [course note]
    (https://fling.seas.upenn.edu/~cis390/dynamic/slides/CIS390_Lecture11.pdf).

    Args:
        A: Matrices of target coordinates.
        B: Matrices of input coordinates. Must have equal numbers of  points
            (rows), but B may have fewer dimensions (columns) than A.
        scaling: if False, the scaling component of the transformation is forced
            to 1
        reflection:
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

    Returns:
        d: The residual sum of squared errors, normalized according to a measure
            of the scale of A, ((A - A.mean(0))**2).sum().
        Z: The matrix of transformed B-values.
        tform: A dict specifying the rotation, translation and scaling that
            maps A --> B.
    """
    assert A.shape[0] == B.shape[0]
    n, dim_x = A.shape
    _, dim_y = B.shape

    # remove translation
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    A0 = A - A_bar
    B0 = B - B_bar

    # remove scale
    ssX = (A0 ** 2).sum()
    ssY = (B0 ** 2).sum()
    A_norm = np.sqrt(ssX)
    B_norm = np.sqrt(ssY)
    A0 /= A_norm
    B0 /= B_norm

    if dim_y < dim_x:
        B0 = np.concatenate((B0, np.zeros(n, dim_x - dim_y)), 0)

    # optimum rotation matrix of B
    A = np.dot(A0.T, B0)
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T
    R = np.dot(V, U.T)

    if reflection != 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(R) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            R = np.dot(V, U.T)

    S_trace = s.sum()
    if scaling:
        # optimum scaling of B
        scale = S_trace * A_norm / B_norm

        # standarised distance between A and scale*B*R + c
        d = 1 - S_trace ** 2

        # transformed coords
        Z = A_norm * S_trace * np.dot(B0, R) + A_bar
    else:
        scale = 1
        d = 1 + ssY / ssX - 2 * S_trace * B_norm / A_norm
        Z = B_norm * np.dot(B0, R) + A_bar

    # transformation matrix
    if dim_y < dim_x:
        R = R[:dim_y, :]
    translation = A_bar - scale * np.dot(B_bar, R)

    # transformation values
    tform = {'rotation': R, 'scale': scale, 'translation': translation}
    return d, Z, tform


def image_to_camera_frame(pose3d_image_frame, box, camera, rootIdx, root_depth):
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame = pose3d_image_frame.copy()
    pose3d_image_frame[:, 2] = pose3d_image_frame[:, 2] / ratio + root_depth

    cx, cy, fx, fy = camera['cx'], camera['cy'], camera['fx'], camera['fy']
    pose3d_image_frame[:, 0] = (pose3d_image_frame[:, 0] - cx) / fx
    pose3d_image_frame[:, 1] = (pose3d_image_frame[:, 1] - cy) / fy
    pose3d_image_frame[:, 0] *= pose3d_image_frame[:, 2]
    pose3d_image_frame[:, 1] *= pose3d_image_frame[:, 2]
    return pose3d_image_frame


def align_to_gt(pose, pose_gt):
    """Align pose to ground truth pose.

    Use MLE.
    """
    return procrustes(pose_gt, pose)[1]


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    return np.stack((x, y, z), 1)


def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree


def rot6d_to_axis_angle(rot6d):
    """Convert 6d rotation representation to 3d vector of axis-angle rotation.

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 3d vector of axis-angle rotation.

    Shape:
        - Input: :math:`(N, 6)`
        - Output: :math:`(N, 3)`
    """
    batch_size = rot6d.shape[0]

    rot6d = rot6d.view(batch_size, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1), device=rot_mat.device).float()],
                        2)  # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle


def rot6d_to_mat3x3(rot6d):
    rot6d = rot6d.view(-1, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix
    return rot_mat


def axis_angle_to_rot6d(angle_axis):
    """Convert 3d vector of axis-angle rotation to 6d rotation representation.

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 6d rotation representation.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 6)`
    """
    rot_mat = tgm.angle_axis_to_rotation_matrix(angle_axis)  # 4x4 rotation matrix
    rot6d = rot_mat[:, :3, :2]
    rot6d = rot6d.reshape(-1, 6)

    return rot6d


def axis_angle_to_mat3x3(angle_axis):
    rot_mat = tgm.angle_axis_to_rotation_matrix(angle_axis)  # 4x4 rotation matrix

    return rot_mat[:, :3, :3]


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def rotate_points(points, rotation_matrix):
    return np.dot(points, rotation_matrix.T)


def get_rotation_matrix_x(angle):
    """
    Return rotation matrix for rotation around X-axis.
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])


def get_rotation_matrix_y(angle):
    """
    Return rotation matrix for rotation around Y-axis.
    """
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])