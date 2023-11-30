import sys

import numpy as np
import pymeshlab as pyml
import torch


def average_pairwise_distance(joints3d):
    """
    Calculate Average Pairwise Distance (APD) for a batch of poses.

    Parameters:
    - joints3d (torch.Tensor): A tensor of shape (batch_size, num_joints, 3)

    Returns:
    - APD (torch.Tensor): Average Pairwise Distance
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

