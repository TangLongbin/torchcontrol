"""
system_base.py
Base class for input/output systems with gym-like interfaces for RL and control applications.
Inherits from torch.nn.Module for GPU parallel computing support.
All main methods are abstract and should be implemented by subclasses.
"""
from __future__ import annotations

import abc
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system_cfg import SystemCfg

class SystemBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for input/output systems. Provides gym-like interfaces for RL/control.
    Methods: __init__, forward, update, reset, step (all abstract).
    """
    
    cfg: SystemCfg
    """The configuration parameters."""
    
    def __init__(self, cfg: SystemCfg):
        """
        Initialize the system with optional configuration.
        Args:
            cfg: SystemCfg
        """
        super().__init__()
        self.cfg = cfg
        self._ALL_INDICES = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward computation of the system. Should be implemented by subclass.
        """
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """
        Update system parameters or internal state. Should be implemented by subclass.
        """
        pass

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        """
        Reset the system to initial state. Should be implemented by subclass.
        """
        pass

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        """
        Step the system forward (like gym.Env.step). Should be implemented by subclass.
        """
        pass
    
    @property
    def num_envs(self):
        """
        Number of environments in the system.
        Returns:
            int: Number of environments.
        """
        return self.cfg.num_envs

    @property
    def dt(self):
        """
        Time step for the system.
        Returns:
            float: Time step.
        """
        return self.cfg.dt
    
    @property
    def device(self):
        """
        Device for the system (CPU or GPU).
        Returns:
            str: Device type.
        """
        return self.cfg.device
