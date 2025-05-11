"""
state_space_system_cfg.py
Configuration for StateSpaceSystem plant (torch version).
"""
import torch
from torch import Tensor
from .plant_cfg import PlantCfg, configclass
from .state_space_system import StateSpaceSystem

@configclass
class StateSpaceSystemCfg(PlantCfg):
    """Configuration for StateSpaceSystem plant."""
    
    class_type: type = StateSpaceSystem

    A: list[float] | Tensor = None
    B: list[float] | Tensor = None
    C: list[float] | Tensor = None
    D: list[float] | Tensor = None
    """System matrices"""

    def __post_init__(self):
        """Post-initialization"""
        # Convert to torch tensors
        self.A = torch.as_tensor(self.A, dtype=torch.float32)
        self.B = torch.as_tensor(self.B, dtype=torch.float32)
        self.C = torch.as_tensor(self.C, dtype=torch.float32)
        self.D = torch.as_tensor(self.D, dtype=torch.float32)
        self.initial_state = torch.as_tensor(self.initial_state, dtype=torch.float32)
        # Dimension checks
        assert self.A.shape[0] == self.A.shape[1], f"A must be square, got {self.A.shape}"
        assert self.A.shape[0] == self.B.shape[0], f"A rows ({self.A.shape[0]}) must match B rows ({self.B.shape[0]})"
        assert self.A.shape[1] == self.C.shape[1], f"A cols ({self.A.shape[1]}) must match C cols ({self.C.shape[1]})"
        assert self.C.shape[0] == self.D.shape[0], f"C rows ({self.C.shape[0]}) must match D rows ({self.D.shape[0]})"
        assert self.initial_state.shape[0] == self.A.shape[1], f"initial_state dim ({self.initial_state.shape[0]}) must match A cols ({self.A.shape[1]})"
        # Set state_dim and action_dim
        self.state_dim = self.A.shape[0]
        self.action_dim = self.B.shape[1]