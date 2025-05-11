"""
plant_cfg.py
Base configuration class for PlantBase.
"""
from torchcontrol.system import SystemCfg, configclass
from .plant_base import PlantBase

@configclass
class PlantCfg(SystemCfg):
    """Base configuration class for PlantBase."""
    
    class_type: type = PlantBase
    
    state_dim: int = 1
    """Dimension of the state space."""
    
    action_dim: int = 1
    """Dimension of the action space."""
    
    initial_state: list[float] = None
    
    ode_method: str = "rk4"
    """ODE integration method to use. Options: "rk4" or method from torchdiffeq."""
    
    ode_options: dict = {}
    """Options for ODE integration method. Used only if ode_method is "rk4" or "dopri5"."""
