"""
input_output_system_cfg.py
Configuration for InputOutputSystem plant (n-order system).
"""
from __future__ import annotations

import torch
from torch import Tensor
from scipy.signal import tf2ss
from .input_output_system import InputOutputSystem
from .state_space_system_cfg import StateSpaceSystemCfg, configclass

@configclass
class InputOutputSystemCfg(StateSpaceSystemCfg):
    """Configuration for InputOutputSystem plant."""
    
    class_type: type = InputOutputSystem
        
    numerator: list[float] | list[list[float]] | Tensor = None
    """Numerator coefficients of the transfer function."""
    
    denominator: list[float] | list[list[float]] | Tensor = None
    """Denominator coefficients of the transfer function."""
    
    def __repr__(self):
        """String representation of the configuration."""
        return f"{super().__repr__()}numerator: {self.numerator}\ndenominator: {self.denominator}\n"
    
    def __post_init__(self):
        """Post-initialization"""
        # Convert to torch tensors
        self.numerator = torch.as_tensor(self.numerator, dtype=torch.float32)
        self.denominator = torch.as_tensor(self.denominator, dtype=torch.float32)
        # Convert transfer function to state space representation
        self._tf2ss()
        # If initial_state is given as output y (1D or [num_envs, 1]), extend to state_dim
        if self.initial_state is not None:
            initial_output = torch.as_tensor(self.initial_state, dtype=torch.float32)
            self.initial_state = torch.zeros(self.num_envs, self.denominator.shape[-1]-1, dtype=torch.float32)
            self.initial_state[:, -initial_output.shape[-1]:] = initial_output
        # Call parent class post_init
        super().__post_init__()

    def _tf2ss(self):
        """
        Convert transfer function to state space representation.
        
        self.numerator & self.denominator --> self.A, self.B, self.C, self.D
        """
        # For-loop to do tf2ss, because tf2ss does not support batch processing
        if self.numerator.dim() == 2 and self.denominator.dim() == 2:
            self.A, self.B, self.C, self.D = [], [], [], []
            for num, den in zip(self.numerator, self.denominator):
                A, B, C, D = tf2ss(num.tolist(), den.tolist())
                self.A.append(A)
                self.B.append(B)
                self.C.append(C)
                self.D.append(D)
            # Stack the lists into tensors
            self.A = torch.stack(self.A, dim=0)
            self.B = torch.stack(self.B, dim=0)
            self.C = torch.stack(self.C, dim=0)
            self.D = torch.stack(self.D, dim=0)
        else:
            A, B, C, D = tf2ss(self.numerator.tolist(), self.denominator.tolist())
            self.A = torch.as_tensor(A, dtype=torch.float32)
            self.B = torch.as_tensor(B, dtype=torch.float32)
            self.C = torch.as_tensor(C, dtype=torch.float32)
            self.D = torch.as_tensor(D, dtype=torch.float32)