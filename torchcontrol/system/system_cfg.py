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
    
    num_envs: int | None = None
    """Number of environments to create."""
    
    dt: float = 0.01
    """Time step for simulation."""