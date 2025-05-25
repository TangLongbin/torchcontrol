"""
mppi_cfg.py
Configuration for Model Predictive Path Integral (MPPI) controller.
"""
from __future__ import annotations

import torch
from torch import Tensor
from .mppi import MPPI
from .controller_cfg import ControllerCfg, configclass


@configclass
class MPPICfg(ControllerCfg):
    """
    Configuration dataclass for the MPPI controller.

    Attributes:
        K (int): Number of rollouts (samples) per control step.
        T (int): Prediction horizon length (number of time steps).
        sigma (float | Tensor): Standard deviation for control noise.
            Can be a scalar (isotropic noise) or a tensor of shape (action_dim,)
            for different noise levels per action.
        alpha (float): Temperature parameter for trajectory weighting.
                         Controls the "softness" of the softmax for weights.
        u_min (float | Tensor): Minimum control input limits.
            Scalar or tensor of shape (action_dim,).
        u_max (float | Tensor): Maximum control input limits.
            Scalar or tensor of shape (action_dim,).
        cost_function (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            Function to compute the cost of rollouts.
            Input: `state_rollouts, shape=(num_envs, K, T, state_dim)` and `action_rollouts, shape=(num_envs, K, T, action_dim)`
            Output: `cost_rollouts, shape=(num_envs, K)`
    """
    class_type: type = MPPI  # Type of the controller class
    
    # MPPI specific parameters
    K: int = 100                   # Default number of samples
    T: int = 20                    # Default prediction horizon
    sigma: float | Tensor = 0.1    # Standard deviation for control noise, can be scalar or tensor
    alpha: float = 0.05            # Temperature parameter for weighting

    # Control input limits
    u_min: float | Tensor = -float('inf') # Minimum control input limits, can be scalar or tensor
    u_max: float | Tensor = float('inf')  # Maximum control input limits, can be scalar or tensor

    # Mandatory parameters that need to be set by the user
    cost_function: callable = None # Must be overridden by user

    def __repr__(self):
        """String representation of the configuration."""
        return (f"{super().__repr__()}"
                f"K: {self.K}\n"
                f"T: {self.T}\n"
                f"sigma: {self.sigma}\n"
                f"alpha: {self.alpha}\n"
                f"u_min: {self.u_min}\n"
                f"u_max: {self.u_max}\n"
                f"cost_function: {self.cost_function.__name__ if self.cost_function else None}\n")

    def __post_init__(self):
        """Post-initialization"""
        # Convert to torch tensors
        self.sigma = torch.as_tensor(self.sigma, dtype=torch.float32)
        self.u_min = torch.as_tensor(self.u_min, dtype=torch.float32)
        self.u_max = torch.as_tensor(self.u_max, dtype=torch.float32)
        # Check plant and cost function
        assert self.plant is not None, "MPPICfg: 'plant' must be specified and configured."
        assert self.cost_function is not None, "MPPICfg: 'cost_function' must be specified."
        assert callable(self.cost_function), "MPPICfg: 'cost_function' must be a callable."
        # Set action_dim and state_dim from plant
        self.action_dim = self.plant.action_dim
        self.state_dim = self.plant.state_dim
        # Check parameters
        assert isinstance(self.K, int) and self.K > 0, f"K must be a positive integer, got {self.K}"
        assert isinstance(self.T, int) and self.T > 0, f"T must be a positive integer, got {self.T}"
        assert torch.all(self.sigma > 0), f"sigma must be positive, got {self.sigma[self.sigma <= 0]}"
        assert self.alpha > 0, f"alpha must be positive, got {self.alpha}"
        assert torch.all(self.u_min <= self.u_max), f"u_min ({self.u_min[self.u_min > self.u_max]}) \
            must be less than or equal to u_max ({self.u_max[self.u_min > self.u_max]})"
        # Check tensor dimensions
        assert self.sigma.dim() == 0 or self.sigma.shape == (self.action_dim,), \
            f"sigma must be a scalar or a tensor of shape ({self.action_dim},), got {self.sigma.shape}"
        assert self.u_min.dim() == 0 or self.u_min.shape == (self.action_dim,), \
            f"u_min must be a scalar or a tensor of shape ({self.action_dim},), got {self.u_min.shape}"
        assert self.u_max.dim() == 0 or self.u_max.shape == (self.action_dim,), \
            f"u_max must be a scalar or a tensor of shape ({self.action_dim},), got {self.u_max.shape}"
        # Call parent's post_init
        super().__post_init__()
