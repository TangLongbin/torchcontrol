"""
nonlinear_system_cfg.py
Configuration for NonlinearSystem plant (general nonlinear system).
"""
from __future__ import annotations

import torch
from dataclasses import fields
from torchcontrol.plants.plant_cfg import PlantCfg, configclass
from .nonlinear_system import NonlinearSystem

@configclass
class Parameters:
    """Parameters dataclass for the dynamics/output functions.
    
    Example:
    ```
    # Parameters for different environments (num_envs=8)
    k = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # spring constant
    c = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # damping coefficient
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # nonlinear coefficient
    
    # Parameters for the nonlinear oscillator example
    params = Parameters(k=k, c=c, alpha=alpha)
    
    # Accessing parameters
    params.k  # tensor([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    params.c  # tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    params.alpha  # tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ```
    """
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            v = torch.as_tensor(v, dtype=torch.float32)
            # TODO: the parameter may be a 2D matrix
            assert v.dim() == 1 or v.dim() == 0, f"Parameter {k} must be a 1D tensor or a scalar, but got {v.dim()}D"
            setattr(self, k, v)
    
    def __repr__(self):
        return f"Parameters({', '.join(f'{k}' for k in self.__dict__.keys())})\n"

@configclass
class NonlinearSystemCfg(PlantCfg):
    """Configuration for NonlinearSystem plant."""
    
    class_type: type = NonlinearSystem
    
    dynamics: callable = None
    """Callable for system dynamics: `dx/dt = f(x, u, t, params)`
    ```
    def dynamics(x: Tensor, u: Tensor, t: float, params: Parameters) -> Tensor:
        # x: state, shape=(num_envs, state_dim)
        # u: input, shape=(num_envs, action_dim)
        # t: time, float
        # params: parameters
        return dx/dt # shape=(num_envs, state_dim)
    ```
    """
    
    output: callable = None
    """Optional callable for system output: `y = g(x, u, t, params)`. If None, `y = x`.
    ```
    def output(x: Tensor, u: Tensor, t: float, params: Parameters) -> Tensor:
        # x: state, shape=(num_envs, state_dim)
        # u: input, shape=(num_envs, action_dim)
        # t: time, float
        # params: parameters
        return y # shape=(num_envs, output_dim)
    ```
    """
    
    params: Parameters = None
    """Parameters dataclass for the dynamics/output functions."""

    def __repr__(self):
        return f"{super().__repr__()}dynamics: {self.dynamics}\noutput: {self.output}\nparams: {self.params}\n"

    def __post_init__(self):
        """Post-initialization"""
        # dynamics and output must be callable functions
        assert callable(self.dynamics), "dynamics must be a callable function"
        if self.output is not None:
            assert callable(self.output), "output must be a callable function if provided"
        else:
            self.output = lambda x, u, t, params: x # all state feedback if output is None
        # every parameter must be a 1D tensor with shape (num_envs,) or a scalar
        if self.params is None:
            self.params = Parameters()
        else:
            for k, v in self.params.__dict__.items():
                if isinstance(v, torch.Tensor) and v.dim() == 1:
                    # TODO: the parameter may be a 2D matrix
                    assert v.shape[0] == self.num_envs, f"Parameter {k} must have shape (num_envs,) but got {v.shape}"
        # Call the parent post_init
        super().__post_init__()