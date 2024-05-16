import torch

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: torch.Tensor([N, feature_dim], dtype=torch.float32)
        data_y: torch.Tensor([N, feature_dim], dtype=torch.float32)
    Returns:
        torch.Tensor([N, N], dtype=torch.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    # Use broadcasting to calculate pairwise Euclidean distance
    dists = torch.cdist(data_x, data_y, p=2)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: torch.Tensor of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    # Using topk to find the kth values directly
    kth_values, _ = torch.kthvalue(unsorted, k, dim=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    # Find k+1 because the smallest distance is to the point itself (0)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc_torch(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        fake_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'.format(real_features.size(0), fake_features.size(0)))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)).any(dim=0).float().mean()

    recall = (distance_real_fake < fake_nearest_neighbour_distances.unsqueeze(0)).any(dim=1).float().mean()

    density = (1. / float(nearest_k)) * (distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)).sum(dim=0).float().mean()

    coverage = (distance_real_fake.min(dim=1).values <real_nearest_neighbour_distances).float().mean()

    return dict(precision=precision.item(), recall=recall.item(), density=density.item(), coverage=coverage.item())
