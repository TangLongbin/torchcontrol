"""
Unit tests for torchcontrol.utils.math
Covers quaternion_error for identity, 180-degree, and arbitrary rotation cases.
"""

import torch
from torchcontrol.utils.math import quaternion_error

def test_quaternion_error_identity():
    # Identity quaternion error should be zero
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_ref = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    err = quaternion_error(q, q_ref)
    assert torch.allclose(err, torch.zeros_like(err), atol=1e-6)

def test_quaternion_error_180deg():
    # 180 degree rotation about x axis
    q = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    q_ref = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    err = quaternion_error(q, q_ref)
    assert torch.allclose(err.abs(), torch.tensor([[torch.pi, 0.0, 0.0]]), atol=1e-5)

def test_quaternion_error_arbitrary():
    # 90 degree rotation about z axis
    angle = torch.tensor(torch.pi / 2)
    q = torch.tensor([[torch.cos(angle/2), 0.0, 0.0, torch.sin(angle/2)]])
    q_ref = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    err = quaternion_error(q, q_ref)
    assert torch.allclose(err.abs(), torch.tensor([[0.0, 0.0, angle]]), atol=1e-5)
