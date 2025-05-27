"""
math.py
Utility functions for mathematical operations in PyTorch.
"""

import torch
from torch import Tensor

@torch.jit.script
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

@torch.jit.script
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

@torch.jit.script
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

@torch.jit.script
def quaternion_error(q: Tensor, q_ref: Tensor) -> Tensor:
    """
    Compute the orientation error between two quaternions as a rotation vector (axis-angle).
    The result is a vector whose direction is the rotation axis and whose norm is the rotation angle (in radians).
    Args:
        q: Tensor of shape (..., 4), current quaternion(s), scalar-first (qw, qx, qy, qz)
        q_ref: Tensor of shape (..., 4), reference quaternion(s), scalar-first (qw, qx, qy, qz)
    Returns:
        rotvec: Tensor of shape (..., 3), rotation vector (axis * angle)
    Raises:
        AssertionError: if input shapes or types are invalid
    """
    assert isinstance(q, Tensor) and isinstance(q_ref, Tensor), "Inputs must be torch.Tensor"
    assert q.shape == q_ref.shape, f"Shape mismatch: q {q.shape}, q_ref {q_ref.shape}"
    assert q.shape[-1] == 4, f"Last dimension must be 4 (quaternion), got {q.shape[-1]}"
    # Quaternion multiplication: q_err = q * q_ref_conj
    # For batch, use Hamilton product
    # q = (..., 4), q_ref = (..., 4)
    qw, qx, qy, qz = q.unbind(-1)
    rw, rx, ry, rz = q_ref.unbind(-1)
    # Conjugate of q_ref
    rw, rx, ry, rz = rw, -rx, -ry, -rz
    # Hamilton product (q * q_ref_conj)
    err_w = qw * rw - qx * rx - qy * ry - qz * rz
    err_x = qw * rx + qx * rw + qy * rz - qz * ry
    err_y = qw * ry - qx * rz + qy * rw + qz * rx
    err_z = qw * rz + qx * ry - qy * rx + qz * rw
    quat_err = torch.stack([err_w, err_x, err_y, err_z], dim=-1)
    # Normalize
    quat_err = quat_err / quat_err.norm(p=2, dim=-1, keepdim=True)
    # Compute rotation angle and axis
    theta = 2 * torch.acos(torch.clamp(quat_err[..., 0], -1.0, 1.0))  # (...,)
    sin_half_theta = torch.sqrt(1 - quat_err[..., 0] ** 2)
    axis = quat_err[..., 1:4] / (sin_half_theta.unsqueeze(-1) + 1e-8)  # (..., 3)
    rotvec = axis * theta.unsqueeze(-1)  # (..., 3)
    return rotvec
