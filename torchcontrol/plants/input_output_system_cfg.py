"""
input_output_system_cfg.py
Configuration for InputOutputSystem plant (n-order system).
"""
from scipy.signal import tf2ss
from .input_output_system import InputOutputSystem
from .state_space_system_cfg import StateSpaceSystemCfg, configclass

@configclass
class InputOutputSystemCfg(StateSpaceSystemCfg):
    """Configuration for InputOutputSystem plant."""
    
    class_type: type = InputOutputSystem
        
    numerator: list[float] = None
    """Numerator coefficients of the transfer function."""
    
    denominator: list[float] = None
    """Denominator coefficients of the transfer function."""
    
    def __post_init__(self):
        A, B, C, D = tf2ss(self.numerator, self.denominator)
        self.A, self.B, self.C, self.D = A, B, C, D
        super().__post_init__()
