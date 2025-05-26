"""
controller_base.py
Base class for controllers, inherits from SystemBase.
Provides general interfaces for controllers and acts as a middle layer between SystemBase and specific controllers.
"""
from __future__ import annotations

import abc
import torch
from typing import Sequence, TYPE_CHECKING
from ..plants import PlantBase
from ..system import SystemBase

if TYPE_CHECKING:
    from .controller_cfg import ControllerCfg

class ControllerBase(SystemBase, metaclass=abc.ABCMeta):
    """
    Abstract base class for controllers. Inherits SystemBase.
    Overwrites abstract methods and provides controller-specific interfaces.
    """
    cfg: ControllerCfg

    def __init__(self, cfg: ControllerCfg):
        super().__init__(cfg)

    @abc.abstractmethod
    def forward(self, x, r, t = None):
        """
        Forward computation of the controller.
        Args:
            x: State variable
            r: Reference variable
            t: Time variable
        Returns:
            u: Control output
        """
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._ALL_INDICES # Reset all environments
        self.plant.reset(env_ids) if self.plant is not None else None
    
    def step(self, r, x = None):
        """
        Step the controller forward in time using the provided reference and state.
        Args:
            r: Reference variable
            x: State variable
        Returns:
            u: Control output
        """
        # Ensure the reference is a tensor and the shape is correct
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        if r.dim() == 0:
            r = r.repeat(self.num_envs, self.state_dim)
        if r.dim() == 1:
            r = r.unsqueeze(0).repeat(self.num_envs, 1)
        assert r.shape == (self.num_envs, self.state_dim), \
            f"Reference shape {r.shape} must be [{self.num_envs}, {self.state_dim}]"
        
        # Ensure the state is a tensor and the shape is correct
        if x is not None:
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            assert x.shape == (self.num_envs, self.state_dim), \
                f"State shape {x.shape} must be [{self.num_envs}, {self.state_dim}]"
        else:
            assert self.plant is not None, \
                "State must be provided or plant must be set in the controller."
            # Get the current output from the plant
            x = self.plant.output(x=self.plant.state)
            
        # Call the forward method with the reference and current output
        return self.forward(x, r, t=self.dt)

    @property
    def plant(self) -> PlantBase | None:
        """
        Get the plant associated with the controller.
        Returns:
            The plant object.
        """
        return self.cfg.plant