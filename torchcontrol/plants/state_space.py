"""
state_space.py
StateSpace plant: standard state space model, torch version.
"""
import torch
from .plant_base import PlantBase
from .state_space_cfg import StateSpaceCfg

class StateSpace(PlantBase):
    """
    StateSpace plant: standard state space model (torch version).
    Args:
        cfg: StateSpaceCfg
    """
    def __init__(self, cfg: StateSpaceCfg):
        super().__init__(cfg)
        self.A = cfg.A
        self.B = cfg.B
        self.C = cfg.C
        self.D = cfg.D
        self.dt = cfg.dt
        self.x = cfg.x0.clone()
        self.reset()

    def forward(self, u):
        return self.step(u)

    def reset(self, x0=None):
        if x0 is not None:
            self.x = torch.as_tensor(x0, dtype=torch.float32)
        else:
            self.x = torch.zeros_like(self.x)
        return self.x

    def step(self, u):
        u = torch.as_tensor(u, dtype=torch.float32)
        self.x = self.A @ self.x + self.B * u
        y = self.C @ self.x + self.D * u
        return y

    def update(self, *args, **kwargs):
        pass
