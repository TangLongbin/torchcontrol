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
from ..system import SystemBase

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
    def forward(self, x, u, t):
        """
        Plant dynamics function to be implemented by subclasses.
        Args:
            x: State variable
            u: Input variable
            t: Time variable
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
        # odeint requires dx/dt = f(t, x) as Ordinary Differential Equation (ODE)
        def ode_func(t, x):
            return self.forward(x, u, t)
        # Integrate the ODE using odeint, shape (len(t), num_envs, state_dim)
        state_trajectory = odeint(
            func=ode_func,
            y0=self.state,
            t=torch.tensor([0, self.dt], device=self.device),
            method=self.ode_method,
            options=self.ode_options
        ) # Integrate the ODE
        self.state = state_trajectory[-1] # Get the last state
        return self.output(self.state, u, self.dt)

    def rollout(self, u, step_by_step=True):
        """
        Rollout the plant dynamics for a given control input over `T` time steps.
        Args:
            u (Tensor[num_envs, `T`, action_dim]): Control input over the rollout
            step_by_step (bool): If True, simulate the dynamics step by step, otherwise use simulate the dynamics for the entire horizon at once.
        Returns:
            state_rollouts (Tensor[num_envs, `T`, state_dim]): state rollouts over the rollout
        """
        # Check dimensions of the control input
        assert u.dim() == 3, \
            f"Input must be a 3D tensor, got {u.dim()}D tensor"
        assert u.shape[0] == self.num_envs and u.shape[2] == self.action_dim, \
            f"Input shape {u.shape} must be [{self.num_envs}, T, {self.action_dim}], got {u.shape}"
        
        # Initialize the state rollouts tensor
        T = u.shape[1]  # rollout horizon length
        state_rollouts = torch.zeros(self.num_envs, T, self.state_dim, device=self.device)
        
        # Two methods to simulate the dynamics
        if step_by_step:
            # Method No.1 : Simulate the dynamics step by step, slower but more accurate
            # Store current state
            current_state = self.state.clone() # store the current state
            
            # Simulate dynamics for T steps
            for k in range(T):
                state_rollouts[:, k, :] = self.state # Store the current state
                u_k = u[:, k, :]                     # Get the control input
                self.step(u_k)                      # Step the plant forward
            
            # Restore current state
            self.state = current_state.clone()
        else:
            # Method No.2 : Use odeint to simulate the dynamics, faster but less accurate
            # define the ODE function
            def ode_func(t, x):
                # Get the current input
                k = int(t.item() // self.dt)  # Convert time to step index
                u_k = u[:, k, :] if k < T else u[:, -1, :]
                return self.forward(x, u_k, t)
            
            # Rollout the plant dynamics for T time steps
            y = odeint(
                func=ode_func,
                y0=self.state,
                t=torch.arange(0, T*self.dt, self.dt, device=self.device),
                method=self.ode_method,
                options=self.ode_options
            ) # shape (T, num_envs, state_dim)
            state_rollouts = y.transpose(0, 1) # Transpose to shape (num_envs, T, state_dim)
        
        return state_rollouts # shape (num_envs, T, state_dim)

    @abc.abstractmethod
    def output(self, x, u, t):
        """
        Plant output function to be implemented by subclasses.
        Args:
            x: State variable
            u: Input variable
            t: Time variable
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