"""
state_space_system.py
StateSpaceSystem plant: standard state space model, torch version.
"""
from __future__ import annotations

import torch
from torch import Tensor
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
        # Add batch dimension if not present
        self.A = self.A.repeat(self.num_envs, 1, 1) if self.A.dim() == 2 else self.A
        self.B = self.B.repeat(self.num_envs, 1, 1) if self.B.dim() == 2 else self.B
        self.C = self.C.repeat(self.num_envs, 1, 1) if self.C.dim() == 2 else self.C
        self.D = self.D.repeat(self.num_envs, 1, 1) if self.D.dim() == 2 else self.D
        # Reset state to initial state
        self.reset()

    def forward(self, x, u, t):
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

    def output(self, x: Tensor | None = None, u: Tensor | None = None, t: float | None = None) -> Tensor:
        """
        State space model y = Cx + Du
        Args:
            x: state
            u: input
            t: time
        Returns:
            y: output
        """
        # If x is not provided, use the current state
        x = x if x is not None else self.state
        # u is optional, if not provided, use zero input
        u = u if u is not None else torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # C.shape == (num_envs, p, n)
        # x.shape == (num_envs, n) --unsqueeze(-1)--> (num_envs, n, 1)
        Cx = torch.bmm(self.C, x.unsqueeze(-1)).squeeze(-1)
        # D.shape == (num_envs, p, m)
        # u.shape == (num_envs, m) --unsqueeze(-1)--> (num_envs, m, 1)
        Du = torch.bmm(self.D, u.unsqueeze(-1)).squeeze(-1)
        # y.shape == (num_envs, p) <--squeeze(-1)-- (num_envs, p, 1)
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