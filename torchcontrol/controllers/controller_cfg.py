"""
controller_cfg.py
ControllerCfg inherits SystemCfg and provides a middle layer for controller configuration.
"""
from __future__ import annotations

from dataclasses import MISSING
from torchcontrol.plants import PlantBase
from torchcontrol.system import SystemCfg, configclass
from .controller_base import ControllerBase

@configclass
class ControllerCfg(SystemCfg):
    """
    Controller configuration base class.
    """
    
    class_type: type[ControllerBase] = MISSING
    
    plant: PlantBase = None
    """Plant to be controlled."""
    
    def __repr__(self):
        """String representation of the configuration."""
        return f"{super().__repr__()}plant: {type(self.plant)}\n"
    
    def __post_init__(self):
        """Post-initialization checks."""
        # Call parent class post_init
        super().__post_init__()
