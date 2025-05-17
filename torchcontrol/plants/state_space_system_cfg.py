"""
state_space_system_cfg.py
Configuration for StateSpaceSystem plant (torch version).
"""
from __future__ import annotations

import torch
from torch import Tensor
from .plant_cfg import PlantCfg, configclass
from .state_space_system import StateSpaceSystem

@configclass
class StateSpaceSystemCfg(PlantCfg):
    """Configuration for StateSpaceSystem plant."""
    
    class_type: type = StateSpaceSystem

    A: list[list[float]] | list[list[list[float]]] | Tensor = None
    B: list[list[float]] | list[list[list[float]]] | Tensor = None
    C: list[list[float]] | list[list[list[float]]] | Tensor = None
    D: list[list[float]] | list[list[list[float]]] | Tensor = None
    """System matrices"""
    
    def __repr__(self):
        """String representation of the configuration."""
        return f"{super().__repr__()}A: {self.A}\nB: {self.B}\nC: {self.C}\nD: {self.D}\n"

    def __post_init__(self):
        """Post-initialization"""
        # Convert to torch tensors
        self.A = torch.as_tensor(self.A, dtype=torch.float32)
        self.B = torch.as_tensor(self.B, dtype=torch.float32)
        self.C = torch.as_tensor(self.C, dtype=torch.float32)
        self.D = torch.as_tensor(self.D, dtype=torch.float32)
        # Set state_dim and action_dim
        self.state_dim = self.A.shape[-2]
        self.action_dim = self.B.shape[-1]
        # Dimension checks
        assert self.A.dim() == 2 or self.A.dim() == 3, f"A must be 2D or 3D, got {self.A.shape}"
        assert self.B.dim() == 2 or self.B.dim() == 3, f"B must be 2D or 3D, got {self.B.shape}"
        assert self.C.dim() == 2 or self.C.dim() == 3, f"C must be 2D or 3D, got {self.C.shape}"
        assert self.D.dim() == 2 or self.D.dim() == 3, f"D must be 2D or 3D, got {self.D.shape}"
        # Check for matrix dimensions
        assert self.A.shape[-2] == self.A.shape[-1], f"A must be square, got {self.A.shape}"
        assert self.A.shape[-2] == self.B.shape[-2], f"A rows ({self.A.shape[-2]}) must match B rows ({self.B.shape[-2]})"
        assert self.A.shape[-1] == self.C.shape[-1], f"A cols ({self.A.shape[-1]}) must match C cols ({self.C.shape[-1]})"
        assert self.C.shape[-2] == self.D.shape[-2], f"C rows ({self.C.shape[-2]}) must match D rows ({self.D.shape[-2]})"
        # Check for batch dimension
        if self.A.dim() == 3:
            assert self.A.shape[0] == self.num_envs, f"A must have first dimension {self.num_envs} for batch, got {self.A.shape[0]}"
        if self.B.dim() == 3:
            assert self.B.shape[0] == self.num_envs, f"B must have first dimension {self.num_envs} for batch, got {self.B.shape[0]}"
        if self.C.dim() == 3:
            assert self.C.shape[0] == self.num_envs, f"C must have first dimension {self.num_envs} for batch, got {self.C.shape[0]}"
        if self.D.dim() == 3:
            assert self.D.shape[0] == self.num_envs, f"D must have first dimension {self.num_envs} for batch, got {self.D.shape[0]}"
        # Call parent class post_init
        super().__post_init__()