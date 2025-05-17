"""
uav_geometric_hover.py
Example: Geometric hover control of a batch quadrotor UAV using NonlinearSystem and a simple geometric controller.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchcontrol.plants.nonlinear_system import NonlinearSystem
from torchcontrol.plants.nonlinear_system_cfg import NonlinearSystemCfg, Parameters

def uav_dynamics(x, u, t, params):
    # x: [num_envs, 7] (p[3], v[3], q[1])
    # u: [num_envs, 4] (F, omega_x, omega_y, omega_z)
    g = params.g  # [num_envs] or scalar
    m = params.m  # [num_envs] or scalar
    p = x[:, 0:3]  # position [num_envs, 3]
    v = x[:, 3:6]  # velocity [num_envs, 3]
    q = x[:, 6]    # yaw (not used)
    F = u[:, 0]    # thrust [num_envs]
    g = g if g.shape[0] == x.shape[0] else g.expand(x.shape[0])
    m = m if m.shape[0] == x.shape[0] else m.expand(x.shape[0])
    dp = v
    dv = torch.zeros_like(v)
    dv[:, 2] = F / m - g  # only z-axis affected by thrust
    dq = torch.zeros_like(q)  # ignore rotation for hover
    dx = torch.cat([dp, dv, dq.unsqueeze(1)], dim=1)
    return dx

def uav_output(x, u, t, params):
    return x

class GeometricHoverController:
    def __init__(self, m, g, dt, num_envs, device):
        self.m = m
        self.g = g
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        self.z_ref = torch.ones(num_envs, device=device) * 1.0  # hover at z=1.0
        self.kp = 8.0
        self.kd = 4.0
    def step(self, x):
        # x: [num_envs, 7]
        z = x[:, 2]
        vz = x[:, 5]
        e = self.z_ref - z
        de = -vz
        u_thrust = self.m * (self.g + self.kp * e + self.kd * de)
        u = torch.zeros(self.num_envs, 4, device=self.device)
        u[:, 0] = u_thrust
        return u

if __name__ == "__main__":
    height, width = 4, 4
    num_envs = height * width
    dt = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.full((num_envs,), 1.0, device=device)
    g = torch.full((num_envs,), 9.81, device=device)
    params = Parameters(m=m, g=g)
    initial_state = torch.zeros(num_envs, 7, device=device)
    cfg = NonlinearSystemCfg(
        dynamics=uav_dynamics,
        output=uav_output,
        dt=dt,
        num_envs=num_envs,
        state_dim=7,
        action_dim=4,
        initial_state=initial_state,
        params=params,
        device=device,
    )
    plant = NonlinearSystem(cfg)
    controller = GeometricHoverController(m, g, dt, num_envs, device)
    T = 5
    steps = int(T / dt)
    t_arr = [0.0]
    y = [initial_state]
    for k in range(steps):
        u = controller.step(y[-1])
        output = plant.step(u)
        y.append(output)
        t_arr.append(t_arr[-1] + dt)
    y = torch.stack(y, dim=1).cpu().numpy()  # [num_envs, steps+1, state_dim]
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(height, width, figsize=(12, 10))
    for idx in range(num_envs):
        i, j = np.unravel_index(idx, (height, width))
        ax = axes[i, j]
        ax.plot(t_arr, y[idx, :, 2], label='z (pos)')
        ax.plot(t_arr, y[idx, :, 5], label='vz (vel)', linestyle='--')
        ax.axhline(1.0, color='r', linestyle=':', label='z_ref')
        ax.set_title(f'Env {idx} Hover')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('State')
        ax.grid()
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "uav_geometric_hover.png")
    plt.savefig(fig_path)
    print("Geometric hover plot saved to:", fig_path)
    print("\033[1;32mTest completed successfully.\033[0m")
