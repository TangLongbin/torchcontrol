"""
pid.py
PID controller implementation for torchcontrol. Inherits ControllerBase.
Supports both continuous and discrete PID control, with feedforward term. All computation uses torch.Tensor.
"""
from __future__ import annotations

import torch
from typing import Sequence, TYPE_CHECKING
from .controller_base import ControllerBase

if TYPE_CHECKING:
    from .pid_cfg import PIDCfg

class PID(ControllerBase):
    """
    General PID controller class. Supports discrete PID with feedforward.
    Args:
        cfg: PIDCfg
    """
    cfg: PIDCfg
    
    def __init__(self, cfg: PIDCfg):
        super().__init__(cfg)
        # PID Parameters
        self.Kp = cfg.Kp.to(self.device)
        self.Ki = cfg.Ki.to(self.device)
        self.Kd = cfg.Kd.to(self.device)
        self.u_ff = cfg.u_ff.to(self.device)
        # Add batch dimension if not present
        if self.Kp.dim() == 1:
            self.Kp = self.Kp.repeat(self.num_envs, 1)
        if self.Ki.dim() == 1:
            self.Ki = self.Ki.repeat(self.num_envs, 1)
        if self.Kd.dim() == 1:
            self.Kd = self.Kd.repeat(self.num_envs, 1)
        if self.u_ff.dim() == 1:
            self.u_ff = self.u_ff.repeat(self.num_envs, 1)
        # Error and control output initialization
        self.e_k_1 = torch.zeros(self.num_envs, self.state_dim, dtype=torch.float32, device=self.device)
        self.e_k_2 = torch.zeros(self.num_envs, self.state_dim, dtype=torch.float32, device=self.device)
        self.u_k_1 = torch.zeros(self.num_envs, self.action_dim, dtype=torch.float32, device=self.device)
        # Reset
        self.reset()

    def forward(self, x, r, t=None):
        """
        Forward computation of the PID controller.
        Args:
            x: State variable
            r: Reference variable
            t: Time variable (not used)
        Returns:
            u: Control output
        """
        # Ensure the reference is a tensor and the shape is correct
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        assert r.shape == (self.num_envs, self.state_dim) and x.shape == (self.num_envs, self.state_dim), \
            f"Reference shape {r.shape} must be [{self.num_envs}, {self.state_dim}] and state shape {x.shape} must be [{self.num_envs}, {self.state_dim}]"
        
        # Compute the error at current time step
        e_k = r - x # shape: (num_envs, state_dim)
        # Compute the PID control output
        u = self.u_k_1 \
            + self.Kp * (e_k - self.e_k_1) \
            + self.Ki * e_k * self.dt \
            + self.Kd * (e_k - 2 * self.e_k_1 + self.e_k_2) / self.dt \
            + self.u_ff # shape: (num_envs, action_dim)
        # Update the previous error and control output
        self.e_k_2 = self.e_k_1.clone()
        self.e_k_1 = e_k.clone()
        self.u_k_1 = u.clone()
        # Return the control output
        return u

    def update(self, *args, **kwargs):
        """
        Update the PID controller parameters.
        Args:
            *args: new parameters
            **kwargs: new parameters
        """
        super().update(*args, **kwargs) # Call parent class update method
        for key in ['Kp', 'Ki', 'Kd', 'u_ff']:
            if key in kwargs:
                assert getattr(self, key).shape == kwargs[key].shape, \
                    f"Shape mismatch for {key}: {getattr(self, key).shape} != {kwargs[key].shape}"
                setattr(self, key, kwargs[key])

    def reset(self, env_ids: Sequence[int] | None = None):
        """
        Reset all or part of the environments to their initial state.
        Args:
            env_ids: sequence of environment indices, None or all indices means reset all
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._ALL_INDICES # Reset all environments
        # Call parent class reset method
        super().reset(env_ids)
        # Reset the error and control output
        self.e_k_1[env_ids] = torch.zeros_like(self.e_k_1[env_ids])
        self.e_k_2[env_ids] = torch.zeros_like(self.e_k_2[env_ids])
        self.u_k_1[env_ids] = torch.zeros_like(self.u_k_1[env_ids])