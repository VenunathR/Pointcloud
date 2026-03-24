"""
Utility functions for point cloud data augmentation and preprocessing.
"""

import numpy as np


def rotate_point_cloud(batch_data):
    """
    Randomly rotate point clouds around the Z-axis.

    Args:
        batch_data: (B, N, 3) array of point clouds
    Returns:
        rotated_data: (B, N, 3)
    """
    B = batch_data.shape[0]
    angles = np.random.uniform(0, 2 * np.pi, B).astype(np.float32)
    cos_a = np.cos(angles)   # (B,)
    sin_a = np.sin(angles)   # (B,)
    zeros = np.zeros(B, dtype=np.float32)
    ones  = np.ones(B,  dtype=np.float32)
    # Build (B, 3, 3) rotation matrices
    R = np.stack([
        cos_a, -sin_a, zeros,
        sin_a,  cos_a, zeros,
        zeros,  zeros, ones,
    ], axis=1).reshape(B, 3, 3)  # (B, 3, 3)
    # batch_data: (B, N, 3), R: (B, 3, 3)
    rotated_data = np.einsum('bni,bji->bnj', batch_data, R)
    return rotated_data.astype(np.float32)


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """
    Randomly perturb point clouds by small rotations around all axes.

    Args:
        batch_data:   (B, N, 3)
        angle_sigma:  std dev of rotation angles
        angle_clip:   maximum rotation angle
    Returns:
        rotated_data: (B, N, 3)
    """
    rotated_data = np.zeros_like(batch_data)
    for i in range(batch_data.shape[0]):
        angles = np.clip(
            angle_sigma * np.random.randn(3), -angle_clip, angle_clip
        ).astype(np.float32)
        Rx = np.array([
            [1,              0,               0],
            [0,  np.cos(angles[0]), -np.sin(angles[0])],
            [0,  np.sin(angles[0]),  np.cos(angles[0])],
        ], dtype=np.float32)
        Ry = np.array([
            [ np.cos(angles[1]), 0, np.sin(angles[1])],
            [0,                  1,               0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ], dtype=np.float32)
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]),  np.cos(angles[2]), 0],
            [0,                  0,                 1],
        ], dtype=np.float32)
        R = Rz @ Ry @ Rx
        rotated_data[i] = batch_data[i] @ R.T
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """
    Add random Gaussian noise to each point.

    Args:
        batch_data: (B, N, 3)
        sigma:      noise std dev
        clip:       noise clip value
    Returns:
        jittered_data: (B, N, 3)
    """
    assert clip > 0
    noise = np.clip(
        sigma * np.random.randn(*batch_data.shape).astype(np.float32),
        -clip, clip
    )
    return batch_data + noise


def normalize_point_cloud(batch_data):
    """
    Normalize each point cloud to zero mean and unit sphere.

    Args:
        batch_data: (B, N, 3)
    Returns:
        normalized: (B, N, 3)
    """
    normalized = np.zeros_like(batch_data)
    for i in range(batch_data.shape[0]):
        xyz = batch_data[i]
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        normalized[i] = xyz
    return normalized


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """
    Randomly scale point clouds.

    Args:
        batch_data:  (B, N, 3)
        scale_low:   minimum scale factor
        scale_high:  maximum scale factor
    Returns:
        scaled_data: (B, N, 3)
    """
    B = batch_data.shape[0]
    scales = np.random.uniform(scale_low, scale_high, B).astype(np.float32)
    return batch_data * scales[:, np.newaxis, np.newaxis]


def shift_point_cloud(batch_data, shift_range=0.1):
    """
    Randomly shift point clouds.

    Args:
        batch_data:  (B, N, 3)
        shift_range: max absolute shift
    Returns:
        shifted_data: (B, N, 3)
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 1, C)).astype(np.float32)
    return batch_data + shifts
