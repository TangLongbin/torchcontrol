"""
pid_cfg.py
PIDCfg provides specific parameter values for PID controller.
"""
from __future__ import annotations

import torch
from torch import Tensor
from .pid import PID
from .controller_cfg import ControllerCfg, configclass

@configclass
class PIDCfg(ControllerCfg):
    """
    PID configuration class. Holds Kp, Ki, Kd, u_ff.
    """
    
    class_type: type = PID
    
    Kp: float | list[float] | Tensor = 1.0
    """Proportional gain"""
    
    Ki: float | list[float] | Tensor = 0.0
    """Integral gain"""
    
    Kd: float | list[float] | Tensor = 0.0
    """Derivative gain"""
    
    u_ff: float | list[float] | Tensor = 0.0
    """Feedforward term"""
    
    def __repr__(self):
        """String representation of the configuration."""
        return f"{super().__repr__()}Kp: {self.Kp}\nKi: {self.Ki}\nKd: {self.Kd}\nu_ff: {self.u_ff}\n"
    
    def __post_init__(self):
        """Post-initialization"""
        # Convert to torch tensors
        self.Kp = torch.as_tensor(self.Kp, dtype=torch.float32)
        self.Ki = torch.as_tensor(self.Ki, dtype=torch.float32)
        self.Kd = torch.as_tensor(self.Kd, dtype=torch.float32)
        self.u_ff = torch.as_tensor(self.u_ff, dtype=torch.float32)
        # Dimension checks
        assert self.Kp.shape == (self.action_dim,) or self.Kp.shape == (self.num_envs, self.action_dim) or self.Kp.dim() == 0, \
            f"Kp shape {self.Kp.shape} must be ({self.action_dim},) or ({self.num_envs}, {self.action_dim}), or scalar"
        assert self.Ki.shape == (self.action_dim,) or self.Ki.shape == (self.num_envs, self.action_dim) or self.Ki.dim() == 0, \
            f"Ki shape {self.Ki.shape} must be ({self.action_dim},) or ({self.num_envs}, {self.action_dim}), or scalar"
        assert self.Kd.shape == (self.action_dim,) or self.Kd.shape == (self.num_envs, self.action_dim) or self.Kd.dim() == 0, \
            f"Kd shape {self.Kd.shape} must be ({self.action_dim},) or ({self.num_envs}, {self.action_dim}), or scalar"
        assert self.u_ff.shape == (self.action_dim,) or self.u_ff.shape == (self.num_envs, self.action_dim) or self.u_ff.dim() == 0, \
            f"u_ff shape {self.u_ff.shape} must be ({self.action_dim},) or ({self.num_envs}, {self.action_dim}), or scalar"
        # Call parent class post_init
        super().__post_init__()
