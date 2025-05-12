"""
state_space_system.py
StateSpaceSystem plant: standard state space model, torch version.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from .plant_base import PlantBase

if TYPE_CHECKING:
    from .state_space_system_cfg import StateSpaceSystemCfg

class StateSpaceSystem(PlantBase):
    """
    StateSpaceSystem plant: standard state space model (torch version).
    Args:
        cfg: StateSpaceSystemCfg
    """
    cfg: StateSpaceSystemCfg

    def __init__(self, cfg: StateSpaceSystemCfg):
        super().__init__(cfg)
        self.A = cfg.A.to(self.device)
        self.B = cfg.B.to(self.device)
        self.C = cfg.C.to(self.device)
        self.D = cfg.D.to(self.device)
        if self.A.dim() == 2:
            # Add batch dimension if not present
            self.A = self.A.repeat(self.num_envs, 1, 1)
            self.B = self.B.repeat(self.num_envs, 1, 1)
            self.C = self.C.repeat(self.num_envs, 1, 1)
            self.D = self.D.repeat(self.num_envs, 1, 1)
        self.reset()

    def forward(self, t, x, u):
        """
        State space model dx/dt = Ax + Bu
        Args:
            t: time
            x: state
            u: input
        Returns:
            dx/dt: state derivative
        """
        # A.shape == (num_envs, n, n)
        # B.shape == (num_envs, n, m)
        # x.shape == (num_envs, n) --unsqueeze(-1)--> (num_envs, n, 1)
        # u.shape == (num_envs, m) --unsqueeze(-1)--> (num_envs, m, 1)
        # dx/dt.shape == (num_envs, n) <--squeeze(-1)-- (num_envs, n, 1)
        Ax = torch.bmm(self.A, x.unsqueeze(-1)).squeeze(-1)
        Bu = torch.bmm(self.B, u.unsqueeze(-1)).squeeze(-1)
        return Ax + Bu

    def output(self, x, u):
        """
        State space model y = Cx + Du
        Args:
            x: state
            u: input
        Returns:
            y: output
        """
        # C.shape == (num_envs, p, n)
        # D.shape == (num_envs, p, m)
        # x.shape == (num_envs, n) --unsqueeze(-1)--> (num_envs, n, 1)
        # u.shape == (num_envs, m) --unsqueeze(-1)--> (num_envs, m, 1)
        # y.shape == (num_envs, p) <--squeeze(-1)-- (num_envs, p, 1)
        Cx = torch.bmm(self.C, x.unsqueeze(-1)).squeeze(-1)
        Du = torch.bmm(self.D, u.unsqueeze(-1)).squeeze(-1)
        return Cx + Du

    def update(self, *args, **kwargs):
        """
        Update the state space model with new parameters or initial state.
        Args:
            *args: new parameters
            **kwargs: new parameters
        """
        super().update(*args, **kwargs) # Call parent class update method
        for key in ['A', 'B', 'C', 'D']:
            if key in kwargs:
                assert getattr(self, key).shape == kwargs[key].shape, \
                    f"Shape mismatch for {key}: {getattr(self, key).shape} != {kwargs[key].shape}"
                setattr(self, key, kwargs[key])