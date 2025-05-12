"""
controller_cfg.py
ControllerCfg inherits SystemCfg and provides a middle layer for controller configuration.
"""
from __future__ import annotations

from torchcontrol.system import SystemCfg, configclass

@configclass
class ControllerCfg(SystemCfg):
    """
    Controller configuration base class.
    """
    def __init__(self):
        super().__init__()
