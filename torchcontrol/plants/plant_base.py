"""
plant_base.py
Base class for plant (system) models. Inherits from SystemBase.
"""
import abc
from torchcontrol.system import SystemBase

class PlantBase(SystemBase, metaclass=abc.ABCMeta):
    """
    Abstract base class for plant (system) models.
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)

    @abc.abstractmethod
    def forward(self, u):
        pass

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, u):
        pass
