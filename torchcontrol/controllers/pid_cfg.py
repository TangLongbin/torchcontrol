"""
pid_cfg.py
PIDCfg provides specific parameter values for PID controller.
"""
from .controller_cfg import ControllerCfg, configclass

@configclass
class PIDCfg(ControllerCfg):
    """
    PID configuration class. Holds Kp, Ki, Kd, dt, u_ff.
    """
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, dt=1.0, u_ff=0.0):
        super().__init__()
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_ff = u_ff
