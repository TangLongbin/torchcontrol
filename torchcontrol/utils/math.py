"""
math.py
Utility functions for mathematical operations in PyTorch.
"""

import torch
from torch import Tensor

def skew_symmetric(vec: Tensor) -> Tensor:
    assert isinstance(vec, Tensor), "vec must be a torch.Tensor"
    assert vec.ndim == 2 and vec.shape[1] == 3, "vec must have shape (num_envs, 3)"
    """
    Compute the skew-symmetric matrix for a batch of 3D vectors.
    Args:
        vec: Tensor of shape (num_envs, 3)
    Returns:
        Tensor of shape (num_envs, 3, 3)
    """
    zero = torch.zeros(vec.shape[0], device=vec.device)
    return torch.stack([
        torch.stack([zero, -vec[:, 2], vec[:, 1]], dim=1),
        torch.stack([vec[:, 2], zero, -vec[:, 0]], dim=1),
        torch.stack([-vec[:, 1], vec[:, 0], zero], dim=1)
    ], dim=1)

def quaternion_to_dcm(q: Tensor) -> Tensor:
    assert isinstance(q, Tensor), "q must be a torch.Tensor"
    assert q.ndim == 2 and q.shape[1] == 4, "q must have shape (num_envs, 4)"
    """
    Convert a batch of quaternions to direction cosine matrices (DCM).
    Args:
        q: Tensor of shape (num_envs, 4), format [qw, qx, qy, qz]
    Returns:
        Tensor of shape (num_envs, 3, 3)
    """
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    C_B_I = torch.stack([
        torch.stack([1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)], dim=1),
        torch.stack([2 * (qx*qy + qw*qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qw*qx)], dim=1),
        torch.stack([2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx**2 + qy**2)], dim=1)
    ], dim=1)
    return C_B_I

def omega_quat_matrix(omega: Tensor) -> Tensor:
    assert isinstance(omega, Tensor), "omega must be a torch.Tensor"
    assert omega.ndim == 2 and omega.shape[1] == 3, "omega must have shape (num_envs, 3)"
    """
    Compute the Omega matrix for quaternion kinematics for a batch of angular velocities.
    Args:
        omega: Tensor of shape (num_envs, 3), angular velocity in body frame
    Returns:
        Tensor of shape (num_envs, 4, 4)
    """
    zero = torch.zeros(omega.shape[0], device=omega.device)
    return torch.stack([
        torch.stack([zero, -omega[:, 0], -omega[:, 1], -omega[:, 2]], dim=1),
        torch.stack([omega[:, 0], zero, omega[:, 2], -omega[:, 1]], dim=1),
        torch.stack([omega[:, 1], -omega[:, 2], zero, omega[:, 0]], dim=1),
        torch.stack([omega[:, 2], omega[:, 1], -omega[:, 0], zero], dim=1),
    ], dim=1)
