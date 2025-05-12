"""
plant_base.py
Base class for plant (system) models. Inherits from SystemBase.
"""
from __future__ import annotations

import abc
import torch
from torchdiffeq import odeint
from typing import TYPE_CHECKING
from collections.abc import Sequence
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
        self.initial_state = cfg.initial_state.to(self.device)
        if self.initial_state.dim() == 1:
            # Add batch dimension if not present
            self.initial_state = self.initial_state.repeat(cfg.num_envs, 1)
        self.state = self.initial_state.clone()
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

    def reset(self, env_ids: Sequence[int] | None = None):
        """
        Reset all or part of the environments to their initial state.
        Args:
            env_ids: sequence of environment indices, None or all indices means reset all
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._ALL_INDICES # Reset all environments
        self.state[env_ids] = self.initial_state[env_ids].clone()

    def step(self, u):
        """
        Step the plant forward in time using the provided input.
        Args:
            u: Input variable
        Returns:
            y: Output variable
        """
        # Ensure the input is a tensor and the shape is correct
        u = torch.as_tensor(u, dtype=torch.float32, device=self.device)
        if u.dim() == 0:
            u = u.repeat(self.num_envs, self.action_dim)
        if u.dim() == 1:
            u = u.unsqueeze(0).repeat(self.num_envs, 1)
        assert u.shape == (self.num_envs, self.action_dim), \
            f"Input shape {u.shape} must be [{self.num_envs}, {self.action_dim}]"
        # odeint requires f(t, x) as dynamics function
        def dynamics(t, x):
            return self.forward(t, x, u)
        # Integrate the ODE using odeint, shape (len(t), num_envs, state_dim)
        state_trajectory = odeint(
            func=dynamics,
            y0=self.state,
            t=torch.tensor([0, self.dt], device=self.device),
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
    
    def update(self, *args, **kwargs):
        """
        Update the plant parameters or internal state.
        Args:
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        for key in ['init_state']:
            if key in kwargs:
                assert getattr(self, key).shape == kwargs[key].shape, \
                    f"Shape mismatch for {key}: {getattr(self, key).shape} != {kwargs[key].shape}"
                setattr(self, key, kwargs[key])
    
    @property
    def state_dim(self):
        """
        State dimension of the plant.
        Returns:
            int: State dimension
        """
        return self.cfg.state_dim
    
    @property
    def action_dim(self):
        """
        Action dimension of the plant.
        Returns:
            int: Action dimension
        """
        return self.cfg.action_dim
    
    @property
    def ode_method(self):
        """
        ODE integration method.
        Returns:
            str: ODE integration method
        """
        return self.cfg.ode_method
    
    @property
    def ode_options(self):
        """
        ODE integration options.
        Returns:
            dict: ODE integration options
        """
        return self.cfg.ode_options