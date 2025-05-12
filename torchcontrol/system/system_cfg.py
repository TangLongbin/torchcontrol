"""
system_cfg.py
Base configuration class for SystemBase. Uses configclass decorator for style consistency.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass
from .system_base import SystemBase

def configclass(cls):
    """Decorator to mark config classes (for style consistency)."""
    cls._is_configclass = True
    return dataclass(cls)

@configclass
class SystemCfg:
    """Base configuration class for SystemBase."""
    
    class_type: type[SystemBase] = MISSING
    """The associated system class.
    
    The class should inherit from :class:`torchcontrol.system.SystemBase`.
    """
    
    num_envs: int = 1
    """Number of environments to create."""
    
    dt: float = 0.01
    """Time step for simulation."""
    
    device: str = "cpu"
    """Device to run the simulation on. Can be "cpu" or "cuda"."""
    
    def __repr__(self):
        """String representation of the configuration."""
        return f"num_envs: {self.num_envs}\ndt: {self.dt}\ndevice: {self.device}\n"
    
    def __post_init__(self):
        """Post-initialization checks."""
        assert self.num_envs > 0, "num_envs must be greater than 0"
        assert self.dt > 0, "dt must be greater than 0"
        assert self.device in ["cpu", "cuda"], "device must be 'cpu' or 'cuda'"