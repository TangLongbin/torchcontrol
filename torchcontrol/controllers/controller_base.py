"""
controller_base.py
Base class for controllers, inherits from SystemBase.
Provides general interfaces for controllers and acts as a middle layer between SystemBase and specific controllers.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from torchcontrol.system import SystemBase

if TYPE_CHECKING:
    from .controller_cfg import ControllerCfg

class ControllerBase(SystemBase, metaclass=abc.ABCMeta):
    """
    Abstract base class for controllers. Inherits SystemBase.
    Overwrites abstract methods and provides controller-specific interfaces.
    """
    cfg: ControllerCfg

    def __init__(self, cfg: ControllerCfg):
        self.cfg = cfg
        super().__init__(cfg)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        pass
