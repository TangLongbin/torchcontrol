"""
nonlinear_system.py
NonlinearSystem plant: general nonlinear system using user-defined dynamics and output functions.
"""
from __future__ import annotations

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
        # Move parameters to the device and add batch dimension if needed
        for k, v in self.params.__dict__.items():
            v = v.to(self.device) # Move to device
            if v.dim() in [1, 2]:
                if v.shape[0] == self.num_envs:
                    if self.params.force_batch:
                        v = v.repeat(self.num_envs, *([1] * v.dim()))
                else:
                    v = v.repeat(self.num_envs,  *([1] * v.dim()))
            # Set the parameter in the params object
            setattr(self.params, k, v)
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