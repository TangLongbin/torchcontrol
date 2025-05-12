"""
plant_base.py
Base class for plant (system) models. Inherits from SystemBase.
"""
from __future__ import annotations

import abc
import torch
from torchdiffeq import odeint
from typing import TYPE_CHECKING
from torchcontrol.system import SystemBase

if TYPE_CHECKING:
    from .plant_cfg import PlantCfg

class PlantBase(SystemBase, metaclass=abc.ABCMeta):
    """
    Abstract base class for plant (system) models.
    """
    cfg: PlantCfg

    def __init__(self, cfg: PlantCfg):
        super().__init__(cfg)
        self.ode_method = cfg.ode_method
        self.ode_options = cfg.ode_options
        self.reset()

    @abc.abstractmethod
    def forward(self, t, x, u):
        """
        Plant dynamics function to be implemented by subclasses.
        Args:
            t: Time variable
            x: State variable
            u: Input variable
        Returns:
            dx/dt: Derivative of state variable
        """
        pass

    def reset(self):
        self.state = self.cfg.initial_state

    def step(self, u):
        """
        Step the plant forward in time using the provided input.
        Args:
            u: Input variable
        Returns:
            y: Output variable
        """
        # Ensure the input is a tensor
        u = torch.as_tensor(u, dtype=torch.float32)
        # odeint requires f(t, x) as dynamics function
        def dynamics(t, x):
            return self.forward(t, x, u)
        # Integrate the ODE using odeint
        state_trajectory = odeint(
            dynamics,
            self.state,
            torch.tensor([0, self.dt]),
            method=self.ode_method,
            options=self.ode_options
        ) # Integrate the ODE
        self.state = state_trajectory[-1] # Get the last state
        return self.output(self.state, u)

    @abc.abstractmethod
    def output(self, x, u):
        """
        Plant output function to be implemented by subclasses.
        Args:
            x: State variable
            u: Input variable
        Returns:
            y: Output variable
        """
        pass
    
    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the plant parameters or internal state. Should be implemented by subclasses.
        Args:
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        pass