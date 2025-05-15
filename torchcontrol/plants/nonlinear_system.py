"""
nonlinear_system.py
NonlinearSystem plant: general nonlinear system using user-defined dynamics and output functions.
"""
from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING
from .plant_base import PlantBase

if TYPE_CHECKING:
    from .nonlinear_system_cfg import NonlinearSystemCfg

class NonlinearSystem(PlantBase):
    """
    NonlinearSystem plant: general nonlinear system using user-defined dynamics and output functions.
    Args:
        cfg: NonlinearSystemCfg
    """
    cfg: NonlinearSystemCfg

    def __init__(self, cfg: NonlinearSystemCfg):
        super().__init__(cfg)
        self.params = cfg.params
        # Move parameters to the device
        for k, v in self.params.__dict__.items():
            setattr(self.params, k, v.to(self.device))
        self.reset()

    def forward(self, x, u, t):
        """
        Compute the state derivative using the user-defined dynamics function.
        Args:
            x: state
            u: input
            t: time
        Returns:
            dx/dt: state derivative
        """
        return self.cfg.dynamics(x, u, t, self.params)

    def output(self, x, u, t):
        """
        Compute the output using the user-defined output function, or return state if not provided.
        Args:
            x: state
            u: input
            t: time
        Returns:
            y: output
        """
        return self.cfg.output(x, u, t, self.params)

    def update(self, *args, **kwargs):
        """
        Update the plant parameters or internal state.
        """
        super().update(*args, **kwargs)
        if 'params' in kwargs:
            self.params = kwargs['params']