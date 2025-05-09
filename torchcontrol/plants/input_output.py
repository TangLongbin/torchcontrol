"""
input_output.py
InputOutput plant: n-order system described by input/output difference equation, using torch.Tensor.
"""
import torch
from .plant_base import PlantBase
from .input_output_cfg import InputOutputCfg

class InputOutput(PlantBase):
    """
    InputOutput plant: n-order system (difference equation, like transfer function), torch version.
    Args:
        cfg: InputOutputCfg
    """
    def __init__(self, cfg: InputOutputCfg):
        super().__init__(cfg)
        self.num = torch.tensor(cfg.num, dtype=torch.float32)
        self.den = torch.tensor(cfg.den, dtype=torch.float32)
        self.dt = cfg.dt
        self.n = self.den.shape[0] - 1
        self.reset()

    def forward(self, u):
        return self.step(u)

    def reset(self):
        # Reset input and output history buffers
        self.u_hist = torch.zeros(self.num.shape[0], dtype=torch.float32)
        self.y_hist = torch.zeros(self.den.shape[0] - 1, dtype=torch.float32)
        return torch.tensor(0.0, dtype=torch.float32)

    def step(self, u):
        # u: scalar or tensor
        u = torch.as_tensor(u, dtype=torch.float32)
        # Update input history
        self.u_hist = torch.cat([u.view(1), self.u_hist[:-1]])
        # Compute output using difference equation (all torch)
        y = (torch.dot(self.num, self.u_hist) - torch.dot(self.den[1:], self.y_hist)) / self.den[0]
        # Update output history
        self.y_hist = torch.cat([y.view(1), self.y_hist[:-1]])
        return y

    def update(self, *args, **kwargs):
        pass
