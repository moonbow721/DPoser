import torch

from . import constants
from .utils import BODY_JOINT_NAMES


def perspective_projection(points,
                           rotation,
                           translation,
                           focal_length,
                           camera_center):
    """
    This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """

    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose, part="body"):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    if part == 'body':
        return torch.exp(
            pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2
    # elif part == 'rhand':  # this seems useless?
    #     indices = [3 * i + 2 for i in range(15)]
    #     return torch.exp(pose[:, indices] * torch.tensor(-1., device=pose.device)) ** 2
    # elif part == 'lhand':
    #     indices = [3 * i + 2 for i in range(15)]
    #     return torch.exp(pose[:, indices] * torch.tensor(1., device=pose.device)) ** 2
    elif part == 'face':
        return torch.exp(pose[:, :3] * torch.tensor([1., 10., 10.], device=pose.device)) ** 2
    else:
        return torch.zeros(pose.shape[0], 0, device=pose.device)


def body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior, t,
                      focal_length=5000, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='mean', verbose=True, **kwargs):
    """
    Loss function for body fitting
    """
    part = kwargs.get('part', 'none')
    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)  # sum along x-y
    if 'hand' in part:
        # To ensure proper hand, we use root-relative coordinates
        projected_hand = projected_joints[:,] - projected_joints[:, [0]]
        gt_hand = joints_2d[:,] - joints_2d[:, [0]]
        reprojection_hand_loss = (joints_conf ** 2) * gmof(projected_hand - gt_hand, sigma).sum(dim=-1)
        fidelity_loss = 0.8*reprojection_loss.sum(dim=-1) + 0.2*reprojection_hand_loss.sum(dim=-1)
    else:
        fidelity_loss = reprojection_loss.sum(dim=-1)

    # Pose prior loss
    if pose_prior is not None:
        pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas, t)
    else:
        pose_prior_loss = 0.0

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose, part).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    # sum along different joints
    total_loss = fidelity_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss
    if verbose:
        print(f"Reprojection Loss: {reprojection_loss.sum(dim=-1).mean().item():.2f}")
        print(f"Angle Prior Loss: {angle_prior_loss.mean().item():.2f}")
        print(f"Shape Prior Loss: {shape_prior_loss.mean().item():.2f}")
        if pose_prior is not None:
            print(f"Pose Prior Loss: {pose_prior_loss.mean().item():.2f}")

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss.sum(dim=-1)
    else:
        return total_loss.mean()  # mean along batch


def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100, part="body"):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)
    if part == "body":
        op_joints = ['RShoulder', 'LShoulder', 'R_Hip', 'L_Hip',]
        op_joints_ind = [2, 5, 9, 12,]
    elif part == "rhand":
        op_joints = ['R_Wrist_Hand', 'R_Thumb_1', 'R_Index_1', 'R_Ring_1', 'R_Pinky_1',]
        op_joints_ind = [0, 1, 5, 9, 13, 17]
    elif part == "lhand":
        op_joints = ['L_Wrist_Hand', 'L_Thumb_1', 'L_Index_1', 'L_Ring_1', 'L_Pinky_1',]
        op_joints_ind = [0, 1, 5, 9, 13, 17]
    else:
        raise ValueError(f"Unknown part: {part}")
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2

    reprojection_loss = (joints_conf[:, op_joints_ind] ** 2) * reprojection_error_op.sum(dim=-1)  # sum along x-y

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss.sum(dim=-1) + depth_loss  # sum along different joints
    return total_loss.sum()


# adapted From https://github.com/vchoutas/smplify-x/blob/master/smplifyx/fitting.py
def guess_init(joints_3d,
               joints_2d,
               focal_length=5000,
               part="body",
               ):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        joints_3d: torch.tensor 1xJx3
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    if part == "body":
        edge_idxs = [(5, 12), (2, 9)]
    elif 'hand' in part:
        edge_idxs = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)]
    else:
        raise ValueError(f"Unknown part: {part}")

    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=joints_3d.dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t