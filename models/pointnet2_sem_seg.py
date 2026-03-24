"""
PointNet++ semantic segmentation model.
Input: (B, 3, N) point cloud (x, y, z only)
Output: (B, N, num_classes) per-point class scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def square_distance(src, dst):
    """
    Compute squared Euclidean distance between every pair of points.

    Args:
        src: (B, N, C)
        dst: (B, M, C)
    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1, keepdim=True).permute(0, 2, 1)
    return dist


def index_points(points, idx):
    """
    Index into a point cloud tensor.

    Args:
        points: (B, N, C)
        idx:    (B, S) or (B, S, K)
    Returns:
        indexed: same shape as idx with last dim C
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """
    Farthest-point sampling.

    Args:
        xyz:    (B, N, 3)
        npoint: number of samples
    Returns:
        centroids: (B, npoint) indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball-query neighbourhood search.

    Args:
        radius:  search radius
        nsample: max points per ball
        xyz:     (B, N, 3) all points
        new_xyz: (B, S, 3) query centres
    Returns:
        group_idx: (B, S, nsample)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqr_dists = square_distance(new_xyz, xyz)
    group_idx[sqr_dists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Combine FPS + ball query + feature grouping.

    Args:
        npoint:  number of centres (FPS)
        radius:  ball radius
        nsample: points per ball
        xyz:     (B, N, 3)
        points:  (B, N, C) or None
    Returns:
        new_xyz:    (B, npoint, 3)
        new_points: (B, npoint, nsample, 3+C)
    """
    B, N, C = xyz.shape
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)                             # (B, npoint, nsample, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, 3)  # relative coords

    if points is not None:
        grouped_points = index_points(points, idx)                   # (B, npoint, nsample, C)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Group all points into a single group (for the global abstraction layer).

    Args:
        xyz:    (B, N, 3)
        points: (B, N, C) or None
    Returns:
        new_xyz:    (B, 1, 3) – origin
        new_points: (B, 1, N, 3+C)
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PointNetSetAbstraction(nn.Module):
    """Set Abstraction layer (MSG disabled – single radius)."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers += [
                nn.Conv2d(last_channel, out_channel, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            last_channel = out_channel
        self.mlp_convs = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        Args:
            xyz:    (B, C, N)
            points: (B, D, N) or None
        Returns:
            new_xyz:    (B, C, S)
            new_points: (B, mlp[-1], S)
        """
        xyz = xyz.permute(0, 2, 1)                              # (B, N, 3)
        if points is not None:
            points = points.permute(0, 2, 1)                    # (B, N, D)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )

        new_points = new_points.permute(0, 3, 2, 1)             # (B, 3+D, nsample, S)
        new_points = self.mlp_convs(new_points)                  # (B, mlp[-1], nsample, S)
        new_points = torch.max(new_points, dim=2)[0]             # (B, mlp[-1], S)

        new_xyz = new_xyz.permute(0, 2, 1)                      # (B, 3, S)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """Feature Propagation layer (interpolation + skip connection)."""

    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers += [
                nn.Conv1d(last_channel, out_channel, 1, bias=False),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(inplace=True),
            ]
            last_channel = out_channel
        self.mlp_convs = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Interpolate features from xyz2 to xyz1.

        Args:
            xyz1:    (B, 3, N) – denser point set
            xyz2:    (B, 3, S) – sparser point set
            points1: (B, D1, N) skip features (or None)
            points2: (B, D2, S) features to propagate
        Returns:
            new_points: (B, mlp[-1], N)
        """
        xyz1 = xyz1.permute(0, 2, 1)  # (B, N, 3)
        xyz2 = xyz2.permute(0, 2, 1)  # (B, S, 3)

        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)                  # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]         # 3 nearest neighbours

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm                            # (B, N, 3)

            interpolated = torch.sum(
                index_points(points2.permute(0, 2, 1), idx) * weight.unsqueeze(-1),
                dim=2,
            ).permute(0, 2, 1)                                   # (B, D2, N)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=1)
        else:
            new_points = interpolated

        new_points = self.mlp_convs(new_points)
        return new_points


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PointNet2SemSeg(nn.Module):
    """
    PointNet++ semantic segmentation network.

    Input:  (B, 3, N)  – x, y, z point cloud
    Output: (B, N, num_classes)
    """

    def __init__(self, num_classes=9):
        super().__init__()

        # Encoder (Set Abstraction layers)
        # SA1: 1024 points, r=0.1, 32 neighbours
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            in_channel=3, mlp=[32, 32, 64]
        )
        # SA2: 256 points, r=0.2, 32 neighbours
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            in_channel=64 + 3, mlp=[64, 64, 128]
        )
        # SA3: 64 points, r=0.4, 32 neighbours
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        # SA4: global abstraction
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32,
            in_channel=256 + 3, mlp=[256, 256, 512]
        )

        # Decoder (Feature Propagation layers)
        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp2 = PointNetFeaturePropagation(in_channel=128 + 64,  mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128,        mlp=[128, 128])

        # Output head
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz):
        """
        Args:
            xyz: (B, 3, N)
        Returns:
            logits: (B, N, num_classes)
        """
        # Encoder
        l0_xyz, l0_points = xyz, None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None,      l1_points)

        # Per-point prediction
        logits = self.head(l0_points)              # (B, num_classes, N)
        logits = logits.permute(0, 2, 1)           # (B, N, num_classes)
        return logits
