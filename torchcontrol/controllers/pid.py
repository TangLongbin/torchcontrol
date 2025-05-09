"""
pid.py
PID controller implementation for torchControl. Inherits ControllerBase.
Supports both continuous and discrete PID control, with feedforward term. All computation uses torch.Tensor.
"""
import torch
from .controller_base import ControllerBase

class PID(ControllerBase):
    """
    General PID controller class. Supports discrete PID with feedforward.
    Args:
        cfg: PIDCfg object or dict with keys 'Kp', 'Ki', 'Kd', 'dt', 'u_ff' (optional)
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.Kp = torch.tensor(getattr(cfg, 'Kp', 1.0), dtype=torch.float32) if cfg else torch.tensor(1.0)
        self.Ki = torch.tensor(getattr(cfg, 'Ki', 0.0), dtype=torch.float32) if cfg else torch.tensor(0.0)
        self.Kd = torch.tensor(getattr(cfg, 'Kd', 0.0), dtype=torch.float32) if cfg else torch.tensor(0.0)
        self.dt = torch.tensor(getattr(cfg, 'dt', 1.0), dtype=torch.float32) if cfg else torch.tensor(1.0)
        self.u_ff = torch.tensor(getattr(cfg, 'u_ff', 0.0), dtype=torch.float32) if cfg else torch.tensor(0.0)
        self.reset()

    def forward(self, error, u_ff=None):
        if u_ff is None:
            u_ff = self.u_ff
        e_k = torch.as_tensor(error, dtype=torch.float32)
        e_k_1 = self.e_k_1
        e_k_2 = self.e_k_2
        dt = self.dt
        du = self.Kp * (e_k - e_k_1) \
            + self.Ki * e_k * dt \
            + self.Kd * ((e_k - 2 * e_k_1 + e_k_2) / dt)
        self.u_k = self.u_k_1 + du + (u_ff - self.u_ff_1)
        self.e_k_2 = self.e_k_1
        self.e_k_1 = e_k
        self.u_k_1 = self.u_k
        self.u_ff_1 = u_ff
        return self.u_k

    def update(self, Kp=None, Ki=None, Kd=None, dt=None):
        if Kp is not None:
            self.Kp = torch.tensor(Kp, dtype=torch.float32)
        if Ki is not None:
            self.Ki = torch.tensor(Ki, dtype=torch.float32)
        if Kd is not None:
            self.Kd = torch.tensor(Kd, dtype=torch.float32)
        if dt is not None:
            self.dt = torch.tensor(dt, dtype=torch.float32)

    def reset(self, u0=0.0, e0=0.0):
        self.e_k_1 = torch.tensor(0.0, dtype=torch.float32)
        self.e_k_2 = torch.tensor(0.0, dtype=torch.float32)
        self.u_k_1 = torch.tensor(u0, dtype=torch.float32)
        self.u_k = torch.tensor(u0, dtype=torch.float32)
        self.u_ff_1 = self.u_ff

    def step(self, error, u_ff=None):
        return self.forward(error, u_ff)
