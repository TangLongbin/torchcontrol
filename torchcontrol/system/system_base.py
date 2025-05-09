"""
system_base.py
Base class for input/output systems with gym-like interfaces for RL and control applications.
Inherits from torch.nn.Module for GPU parallel computing support.
All main methods are abstract and should be implemented by subclasses.
"""
import abc
import torch.nn as nn

class SystemBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for input/output systems. Provides gym-like interfaces for RL/control.
    Methods: __init__, forward, update, reset, step (all abstract).
    """
    def __init__(self, cfg=None):
        """
        Initialize the system with optional configuration.
        Args:
            cfg: Optional configuration object.
        """
        super().__init__()
        self.cfg = cfg

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
