"""
state_space_system.py
StateSpaceSystem plant: standard state space model, torch version.
"""
from .plant_base import PlantBase
from .state_space_system_cfg import StateSpaceSystemCfg

class StateSpaceSystem(PlantBase):
    """
    StateSpaceSystem plant: standard state space model (torch version).
    Args:
        cfg: StateSpaceSystemCfg
    """
    def __init__(self, cfg: StateSpaceSystemCfg):
        super().__init__(cfg)
        self.A = cfg.A
        self.B = cfg.B
        self.C = cfg.C
        self.D = cfg.D
        self.reset()

    def forward(self, t, x, u):
        """
        State space model dx/dt = Ax + Bu
        Args:
            t: time
            x: state
            u: input
        Returns:
            dx/dt: state derivative
        """
        return self.A @ x + self.B @ u

    def output(self, x, u):
        """
        State space model y = Cx + Du
        Args:
            x: state
            u: input
        Returns:
            y: output
        """
        return self.C @ x + self.D @ u

    def update(self, *args, **kwargs):
        """
        Update the state space model with new parameters.
        Args:
            *args: new parameters
            **kwargs: new parameters
        """
        for key in ['A', 'B', 'C', 'D']:
            if key in kwargs:
                setattr(self, key, kwargs[key])