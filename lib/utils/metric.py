import sys

import numpy as np
import pymeshlab as pyml
import torch
from scipy import linalg

from lib.utils.transforms import axis_angle_to_quaternion
from lib.prdc import compute_prdc_torch


def generate_dataset_statistics(pose_dataloader, key='body_pose', output_filepath=None):
    """
    Generate statistics for a dataset of poses and save them as npz file.
    Args:
        pose_dataloader: PyTorch DataLoader containing the dataset of poses.

    Returns:
        mu: mean of the poses
        sigma: covariance of the poses
    """
    all_poses = []

    # Loop through the DataLoader to accumulate all pose data
    for batch in pose_dataloader:
        poses = batch[key]  # Adjust this key based on how your data is structured
        if isinstance(poses, torch.Tensor):
            poses = poses.numpy()  # Convert to numpy if it's still a tensor
        all_poses.append(poses)

    # Concatenate all pose data into a single NumPy array
    all_poses = np.vstack(all_poses)

    # Compute the mean and covariance of the pose vectors
    mu = np.mean(all_poses, axis=0)
    sigma = np.cov(all_poses, rowvar=False)  # Ensure each column represents a variable

    # Save the statistics to an npz file
    if output_filepath is not None:
        np.savez(output_filepath, mu=mu, sigma=sigma)
        print(f'Statistics saved to {output_filepath}')
    return mu, sigma  # Optionally return these values for immediate use


def compute_batch_statistics(pose_batch):
    """
    Compute the mean and covariance of a batch of poses for testing.

    Args:
        pose_batch: A tensor of shape (batch_size, pose_dim)

    Returns:
        mu: mean of the poses
        sigma: covariance of the poses
    """
    # Convert the tensor to a NumPy array
    if isinstance(pose_batch, torch.Tensor):
        poses = pose_batch.detach().cpu().numpy()
    else:
        poses = pose_batch

    # Compute the mean and covariance of the pose vectors
    mu = np.mean(poses, axis=0)
    sigma = np.cov(poses, rowvar=False)  # Ensure each column represents a variable

    return mu, sigma


def evaluate_fid(pose_batch, dataset_stat_path):
    """
    Compute the Frechet Inception Distance (FID) between a batch of poses and a dataset.

    Args:
        pose_batch: A tensor of shape (batch_size, pose_dim)
        dataset_stat_path: Path to the npz file containing the dataset statistics

    Returns:
        fid: The Frechet Inception Distance
    """
    # Load the dataset statistics
    dataset_stats = np.load(dataset_stat_path)
    mu, sigma = dataset_stats['mu'], dataset_stats['sigma']

    # Compute the mean and covariance of the batch of poses
    batch_mu, batch_sigma = compute_batch_statistics(pose_batch)

    # Compute the Frechet Distance
    fid = calculate_frechet_distance(mu, sigma, batch_mu, batch_sigma)

    return fid


def evaluate_prdc(pose_batch, reference_batch, nearest_k=3):
    """
    Compute the {Precision, Recall, Density, Coverage} between a batch of poses and a dataset.
    from: https://github.com/clovaai/generative-evaluation-prdc
    Args:
        pose_batch: A tensor of shape (batch_size, pose_dim)
        reference_batch: A tensor of shape (num_samples, pose_dim) containing the reference data
            Or a path to the .pt file containing the reference batch
        nearest_k: The number of nearest neighbors to consider

    Returns:
        prdc: The Pose-Respecting Distance Correlation
    """
    # Load the dataset tensor if a path is provided
    if isinstance(reference_batch, str):
        reference_batch = torch.load(reference_batch).to(pose_batch.device)
    assert pose_batch.size(1) == reference_batch.shape[1], "Pose dimension mismatch."
    results_dict = compute_prdc_torch(reference_batch, pose_batch, nearest_k=nearest_k)

    return results_dict


@torch.no_grad()
def evaluate_dnn(pose_batch, dataset_tensor, reduce='mean', batch_size=100000, measure='rotation'):
    """
    Measure the distance between the generated pose and its nearest neighbor from the training data,
    processing the dataset in mini-batches to conserve memory.

    Args:
        pose_batch: A tensor of shape (batch_size, pose_dim), axis-angle representation
        dataset_tensor: A tensor of shape (num_samples, pose_dim) containing the training data
            Or a path to the .pt file containing the dataset
        reduce: The reduction operation to apply to the distances ('mean', 'none')
        batch_size: Size of mini-batches to use when processing the dataset tensor
        measure (str): The distance measure to use ('absolute' or 'rotation'). Default: 'rotation'

    Returns:
        dnn: The distance to the nearest neighbor for each pose in the batch (reduced if specified)
    """
    # Load the dataset tensor if a path is provided
    if isinstance(dataset_tensor, str):
        dataset_tensor = torch.load(dataset_tensor).to(pose_batch.device)
    assert pose_batch.size(1) == dataset_tensor.size(1), "Pose dimension mismatch."
    # Initialize a tensor to hold the minimum distances found so far
    min_distances = torch.full((pose_batch.size(0),), float('inf'), device=pose_batch.device)

    # Flatten pose_batch and dataset_tensor for easier processing if measure is 'absolute'
    if measure == 'rotation':
        pose_batch = axis_angle_to_quaternion(pose_batch.reshape(pose_batch.size(0), -1, 3))

    # Process the dataset_tensor in mini-batches
    for start_idx in range(0, dataset_tensor.size(0), batch_size):
        end_idx = min(start_idx + batch_size, dataset_tensor.size(0))
        batch = dataset_tensor[start_idx:end_idx]

        if measure == 'absolute':
            distances = torch.cdist(pose_batch[None], batch[None], p=2)[0]
        elif measure == 'rotation':
            # Calculate rotational distances for each joint separately and sum them up
            batch = axis_angle_to_quaternion(batch.reshape(batch.size(0), -1, 3))
            distances = torch.zeros(pose_batch.size(0), batch.size(0), device=pose_batch.device)
            for j in range(pose_batch.size(1)):
                joint_distances = quaternion_angular_distance(pose_batch[:, j, :], batch[:, j, :])
                distances += joint_distances
        else:
            raise ValueError("Invalid distance measure. Use 'absolute' or 'rotation'.")
        # Update the minimum distances found so far
        min_distances, _ = torch.min(torch.stack([min_distances, torch.min(distances, dim=1).values]), dim=0)

    # Apply the reduction operation
    if reduce == 'mean':
        dnn = torch.mean(min_distances)
    elif reduce == 'none':
        dnn = min_distances
    else:
        raise ValueError("Invalid reduction operation. Use 'mean' or 'none'.")

    return dnn


def quaternion_angular_distance(q1, q2, eps=1e-12, check_unit=True):
    """
    Compute the angular distances between two batches of quaternions.
    Args:
        q1 (Tensor): A batch of quaternions with shape [N, 4], where N is the number of quaternions.
        q2 (Tensor): Another batch of quaternions with shape [M, 4], where M is the number of quaternions.
        eps (float): A small value to avoid numerical issues when clamping the dot product.
        check_unit (bool): Whether to check if the quaternions are unit quaternions before computing the distances.
    Returns:
        Tensor: A matrix of angular distances with shape [N, M].
    """
    # Normalize quaternions to ensure they are unit quaternions
    if check_unit:
        q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
        q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)

    # Compute dot products between all combinations of quaternions from q1 and q2
    dot_product = torch.mm(q1, q2.transpose(0, 1))  # Resulting shape is [N, M]

    # Clamp dot product values to avoid numerical issues outside arccos range
    dot_product = torch.clamp(dot_product, -1.0+eps, 1.0-eps)

    # Compute angular distances
    angular_distances = 2 * torch.acos(torch.abs(dot_product))

    return angular_distances


# from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def average_pairwise_distance(joints3d):
    """
    Calculate Average Pairwise Distance (APD) for a batch of poses.

    Parameters:
    - joints3d (torch.Tensor): A tensor of shape (batch_size, num_joints, 3)

    Returns:
    - APD (torch.Tensor): Average Pairwise Distance (unit: meter)
    """
    batch_size, num_joints, _ = joints3d.shape

    # Initialize tensor to store pairwise distances between samples in the batch
    pairwise_distances = torch.zeros(batch_size, batch_size)

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            # Calculate the pairwise distance between sample i and sample j
            dist = torch.mean(torch.norm(joints3d[i, :, :] - joints3d[j, :, :], dim=1))
            pairwise_distances[i, j] = dist
            pairwise_distances[j, i] = dist  # Distance is symmetric

    # The diagonal is zero as the distance between a sample and itself is zero
    pairwise_distances.fill_diagonal_(0)

    # Calculate the mean over all the pairwise distances in the batch to get APD
    APD = torch.sum(pairwise_distances) / (batch_size * (batch_size - 1))

    return APD


# from https://bitbucket.csiro.au/projects/CRCPMAX/repos/corticalflow/browse/src/metrics.py
def self_intersections_percentage(vertices, faces):
    """
    Calculate the average percentage of self-intersecting faces for a batch of 3D meshes.

    Parameters:
    - vertices (numpy.ndarray or torch.Tensor): A tensor or array of shape (batch_size, num_vertices, 3).
                                                Contains the vertices for each mesh in the batch.
    - faces (numpy.ndarray or torch.Tensor): A tensor or array of shape (num_faces, 3).
                                             Contains the indices of vertices that make up each face.
                                             The same faces are used for each mesh in the batch.

   Returns:
    - fracSI_array (numpy.ndarray): An array containing the percentage of self-intersecting faces for each mesh in the batch.

    Note: If PyMeshLab is not installed, this function returns an array of NaNs.
    """

    # Type check and conversion for vertices
    if isinstance(vertices, torch.Tensor):
        if vertices.is_cuda:
            vertices = vertices.cpu()
        vertices = vertices.detach().numpy()

    # Type check and conversion for faces
    if isinstance(faces, torch.Tensor):
        if faces.is_cuda:
            faces = faces.cpu()
        faces = faces.detach().numpy()

    if 'pymeshlab' not in sys.modules:
        return np.ones(len(vertices)) * np.nan  # Assuming vertices has a batch dimension

    fracSI_array = np.zeros(len(vertices))

    for i, vert in enumerate(vertices):
        ms = pyml.MeshSet()
        ms.add_mesh(pyml.Mesh(vert, faces))

        # Use updated function names as per the warning messages
        total_faces = ms.get_topological_measures()['faces_number']
        ms.compute_selection_by_self_intersections_per_face()
        ms.meshing_remove_selected_faces()

        non_SI_faces = ms.get_topological_measures()['faces_number']
        SI_faces = total_faces - non_SI_faces
        fracSI = (SI_faces / total_faces) * 100
        fracSI_array[i] = fracSI

    return fracSI_array
