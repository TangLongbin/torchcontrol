"""
parameters.py
Parameters dataclass for dynamics and output functions.

This module defines a Parameters config class that holds simulation or plant
parameters as tensors. Scalars, vectors, matrices or 3D tensors are supported,
with optional batch dimension enforcement.
"""
from __future__ import annotations

import torch
from .system_cfg import configclass

@configclass
class Parameters:
    """Parameters dataclass for the dynamics/output functions.
    
    Example:
    ```
    # Parameters for different environments (num_envs=8)
    k = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # spring constant
    c = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # damping coefficient
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # nonlinear coefficient
    
    # Parameters for the nonlinear oscillator example
    params = Parameters(k=k, c=c, alpha=alpha)
    
    # Accessing parameters
    params.k  # tensor([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    params.c  # tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    params.alpha  # tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ```
    """
    
    force_batch: bool = False
    """If True, add a batch dimension to the parameters, even if the first dimension is equal to num_envs."""
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            v = torch.as_tensor(v, dtype=torch.float32)
            # Only allow scalar, 1D, 2D or 3D tensors, and 3D tensors must have first dimension equal to num_envs
            assert v.dim() <= 3, f"Parameter {k} must be a scalar, 1D, 2D or 3D tensor, but got {v.shape}"
            setattr(self, k, v)
    
    def __repr__(self):
        return f"Parameters({', '.join(f'{k}' for k in self.__dict__.keys())}, force_batch={self.force_batch})"
