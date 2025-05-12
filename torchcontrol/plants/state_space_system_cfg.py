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
        if self.A.dim() == 3:
            # Check for batch dimension
            assert self.A.shape[0] == self.num_envs, f"A must have first dimension {self.num_envs}, got {self.A.shape[0]}"
            assert self.B.shape[0] == self.num_envs, f"B must have first dimension {self.num_envs}, got {self.B.shape[0]}"
            assert self.C.shape[0] == self.num_envs, f"C must have first dimension {self.num_envs}, got {self.C.shape[0]}"
            assert self.D.shape[0] == self.num_envs, f"D must have first dimension {self.num_envs}, got {self.D.shape[0]}"
        # Check for matrix dimensions
        assert self.A.shape[-2] == self.A.shape[-1], f"A must be square, got {self.A.shape}"
        assert self.A.shape[-2] == self.B.shape[-2], f"A rows ({self.A.shape[-2]}) must match B rows ({self.B.shape[-2]})"
        assert self.A.shape[-1] == self.C.shape[-1], f"A cols ({self.A.shape[-1]}) must match C cols ({self.C.shape[-1]})"
        assert self.C.shape[-2] == self.D.shape[-2], f"C rows ({self.C.shape[-2]}) must match D rows ({self.D.shape[-2]})"
        # Call parent class post_init
        super().__post_init__()