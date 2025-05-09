"""
input_output_cfg.py
Configuration for InputOutput plant (n-order system).
"""
import numpy as np
from scipy.signal import cont2discrete
from .plant_cfg import PlantCfg, configclass

@configclass
class InputOutputCfg(PlantCfg):
    """
    Configuration for InputOutput plant.
    Args:
        num: list or array of numerator coefficients (continuous, highest order first)
        den: list or array of denominator coefficients (continuous, highest order first)
        dt: time step
    """
    def __init__(self, num, den, dt=None, method="zoh"):
        super().__init__()
        # Convert to numpy arrays
        self.num = np.array(num, dtype=np.float32)
        self.den = np.array(den, dtype=np.float32)
        self.discrete = False
        
        # Convert to discrete if dt is provided
        self.dt = dt
        self.method = method
        if self.dt is not None:
            self.to_discrete(self.dt, self.method)
        
    def to_discrete(self, dt=0.01, method="zoh"):
        """
        Convert continuous system to discrete system with dt and method.
        Args:
            dt: time step
            method: discretization method (default: "zoh")
        """
        if self.discrete:
            raise ValueError("System is already discrete.")
        else:
            sysd = cont2discrete((self.num, self.den), dt, method=method)
            self.num = sysd[0].flatten().tolist()
            self.den = sysd[1].flatten().tolist()
            self.discrete = True
