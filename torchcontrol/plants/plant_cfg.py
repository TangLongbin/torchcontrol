"""
plant_cfg.py
Base configuration class for PlantBase.
"""
from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import MISSING
from torchcontrol.system import SystemCfg, configclass
from .plant_base import PlantBase

@configclass
class PlantCfg(SystemCfg):
    """Base configuration class for PlantBase."""
    
    class_type: type[PlantBase] = MISSING
    
    initial_state: list[float] | list[list[float]] | Tensor = None
    
    ode_method: str = "rk4"
    """ODE integration method to use. Options: "rk4" or method from torchdiffeq."""
    
    ode_options: dict = None
    """Options for ODE integration method. Used only if ode_method is "rk4" or "dopri5"."""
    
    def __repr__(self):
        """String representation of the configuration."""
        initial_state_description = "manually set" if self.initial_state.count_nonzero().item() != 0 else "all zero"
        return (
            f"{super().__repr__()}"
            f"initial_state: {initial_state_description}\n"
            f"ode_method: {self.ode_method}\n"
            f"ode_options: {self.ode_options}\n"
        )
    
    def __post_init__(self):
        """Post-initialization"""
        # Convert initial_state to tensor if not None, otherwise set to zero
        if self.initial_state is None:
            self.initial_state = torch.zeros(self.state_dim)
        else:
            self.initial_state = torch.as_tensor(self.initial_state, dtype=torch.float32)
        # Dimension checks
        assert self.initial_state.shape == (self.state_dim,) or self.initial_state.shape == (self.num_envs, self.state_dim), \
            f"initial_state shape {self.initial_state.shape} must be ({self.state_dim},) or ({self.num_envs}, {self.state_dim})"
        # Call parent class post_init
        super().__post_init__()