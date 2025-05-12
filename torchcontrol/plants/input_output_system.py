"""
input_output_system.py
InputOutputSystem plant: n-order system described by input/output difference equation, using torch.Tensor.
"""
from __future__ import annotations

from scipy.signal import tf2ss
from typing import TYPE_CHECKING
from .state_space_system import StateSpaceSystem

if TYPE_CHECKING:
    from .input_output_system_cfg import InputOutputSystemCfg

class InputOutputSystem(StateSpaceSystem):
    """
    InputOutputSystem plant: n-order system (transfer function form, linear system), implemented via state-space canonical form.
    Args:
        cfg: InputOutputSystemCfg
    """
    cfg: InputOutputSystemCfg

    def __init__(self, cfg: InputOutputSystemCfg):
        super().__init__(cfg)
        # num/den are handled in cfg and converted to A, B, C, D via tf2ss
        # All state, step, forward, output logic is inherited from StateSpaceSystem
        # No additional attributes or methods are needed unless you want to extend functionality
        pass
    
    def update(self, *args, **kwargs):
        """
        Update the numerator and denominator coefficients of the transfer function.
        Args:
            *args: new numerator and denominator coefficients
            **kwargs: new numerator and denominator coefficients
        """
        super().update(*args, **kwargs)
        is_updated = False
        for key in ['numerator', 'denominator']:
            if key in kwargs:
                assert getattr(self, key).shape == kwargs[key].shape, \
                    f"Shape mismatch for {key}: {getattr(self, key).shape} != {kwargs[key].shape}"
                setattr(self, key, kwargs[key])
                is_updated = True
        if is_updated:
            # Convert transfer function to state space representation
            self.cfg._tf2ss()