"""
plant_cfg.py
Base configuration class for PlantBase.
"""
import torch
from dataclasses import MISSING
from torchcontrol.system import SystemCfg, configclass
from .plant_base import PlantBase

@configclass
class PlantCfg(SystemCfg):
    """Base configuration class for PlantBase."""
    
    class_type: type[PlantBase] = MISSING
    
    state_dim: int = 1
    """Dimension of the state space."""
    
    action_dim: int = 1
    """Dimension of the action space."""
    
    initial_state: list[float] = None
    
    ode_method: str = "rk4"
    """ODE integration method to use. Options: "rk4" or method from torchdiffeq."""
    
    ode_options: dict = None
    """Options for ODE integration method. Used only if ode_method is "rk4" or "dopri5"."""
    
    def __post_init__(self):
        """Post-initialization"""
        # Convert initial_state to tensor if not None, otherwise set to zero
        if self.initial_state is None:
            self.initial_state = torch.zeros(self.state_dim)
        else:
            self.initial_state = torch.as_tensor(self.initial_state, dtype=torch.float32)
        # Dimension checks
        assert self.state_dim > 0, "state_dim must be greater than 0"
        assert self.action_dim > 0, "action_dim must be greater than 0"
        assert self.initial_state.shape[0] == self.state_dim, f"initial_state dim ({self.initial_state.shape[0]}) must match state_dim ({self.state_dim})"