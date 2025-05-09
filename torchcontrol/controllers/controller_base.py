"""
controller_base.py
Base class for controllers, inherits from SystemBase.
Provides general interfaces for controllers and acts as a middle layer between SystemBase and specific controllers.
"""
import abc
from torchcontrol.system import SystemBase

class ControllerBase(SystemBase, metaclass=abc.ABCMeta):
    """
    Abstract base class for controllers. Inherits SystemBase.
    Overwrites abstract methods and provides controller-specific interfaces.
    """
    def __init__(self, cfg=None):
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
