"""
observer_cfg.py
Configuration for ObserverBase class.
"""
from __future__ import annotations

from dataclasses import MISSING
from torchcontrol.system import SystemCfg, configclass
from .observer_base import ObserverBase

@configclass
class ObserverCfg(SystemCfg):
    """Configuration for ObserverBase class."""
    
    class_type: type[ObserverBase] = MISSING
    """Type of the observer class."""