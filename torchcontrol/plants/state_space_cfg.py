"""
state_space_cfg.py
Configuration for StateSpace plant (torch version).
"""
from .plant_cfg import PlantCfg, configclass
import torch

@configclass
class StateSpaceCfg(PlantCfg):
    """
    Configuration for StateSpace plant.
    Args:
        A, B, C, D: system matrices (list/array/torch.Tensor)
        x0: initial state
        dt: time step
    """
    def __init__(self, A, B, C, D, x0=None, dt=0.01):
        super().__init__()
        self.A = torch.tensor(A, dtype=torch.float32)
        self.B = torch.tensor(B, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        self.D = torch.tensor(D, dtype=torch.float32)
        self.x0 = torch.zeros(self.A.shape[0], dtype=torch.float32) if x0 is None else torch.tensor(x0, dtype=torch.float32)
        self.dt = dt
