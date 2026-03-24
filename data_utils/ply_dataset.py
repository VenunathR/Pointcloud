"""
Custom PLY dataset loader for PointNet++ semantic segmentation.
Loads .ply files with x, y, z coordinates and scalar_Classification labels.
"""

import os
import numpy as np
from torch.utils.data import Dataset

from provider import rotate_point_cloud, jitter_point_cloud

try:
    from plyfile import PlyData
except ImportError:
    raise ImportError("plyfile is required. Install with: pip install plyfile")


NUM_CLASSES = 9
NUM_POINTS = 4096


def load_ply_file(filepath):
    """Load a .ply file and return xyz coordinates and classification labels."""
    ply_data = PlyData.read(filepath)
    vertex = ply_data['vertex']

    x = np.array(vertex['x'], dtype=np.float32)
    y = np.array(vertex['y'], dtype=np.float32)
    z = np.array(vertex['z'], dtype=np.float32)
    xyz = np.stack([x, y, z], axis=1)  # (N, 3)

    labels = np.array(vertex['scalar_Classification'], dtype=np.int64)

    return xyz, labels


def normalize_point_cloud(xyz):
    """Center and scale point cloud to unit sphere."""
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if max_dist > 0:
        xyz = xyz / max_dist
    return xyz


def sample_or_pad(xyz, labels, num_points=NUM_POINTS):
    """Sample or pad point cloud to a fixed number of points."""
    n = xyz.shape[0]
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
    else:
        # Pad by repeating existing points
        pad_indices = np.random.choice(n, num_points - n, replace=True)
        indices = np.concatenate([np.arange(n), pad_indices])
        np.random.shuffle(indices)
    return xyz[indices], labels[indices]


class PLYDataset(Dataset):
    """
    Dataset for .ply point cloud files.

    Args:
        data_root (str): Root directory containing train/test/val subdirectories.
        split (str): One of 'train', 'test', or 'val'.
        num_points (int): Number of points to sample per file.
        augment (bool): Whether to apply data augmentation (rotation + jitter).
    """

    def __init__(self, data_root, split='train', num_points=NUM_POINTS, augment=True):
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.augment = augment and (split == 'train')

        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Expected structure: {data_root}/{{train,test,val}}/*.ply"
            )

        self.file_paths = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith('.ply')
        ])

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .ply files found in {split_dir}")

        print(f"[PLYDataset] {split}: {len(self.file_paths)} files found")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        xyz, labels = load_ply_file(self.file_paths[idx])

        # Sample or pad to fixed size
        xyz, labels = sample_or_pad(xyz, labels, self.num_points)

        # Normalize
        xyz = normalize_point_cloud(xyz)

        # Augmentation (provider functions expect (B, N, 3))
        if self.augment:
            xyz = rotate_point_cloud(xyz[np.newaxis])[0]
            xyz = jitter_point_cloud(xyz[np.newaxis])[0]

        # (N, 3) -> (3, N) for model input
        xyz = xyz.T.astype(np.float32)

        return xyz, labels
