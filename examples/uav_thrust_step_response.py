"""
uav_thrust_step_response.py
Example: Step response of a batch quadrotor UAV using NonlinearSystem.
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
    # params: Parameters
    g = params.g  # [num_envs] or scalar
    m = params.m  # [num_envs] or scalar
    p = x[:, 0:3]  # position [num_envs, 3]
    v = x[:, 3:6]  # velocity [num_envs, 3]
    q = x[:, 6]    # yaw (for now, ignore)
    F = u[:, 0]    # thrust [num_envs]
    # Batch gravity and mass
    g = g if g.shape[0] == x.shape[0] else g.expand(x.shape[0])
    m = m if m.shape[0] == x.shape[0] else m.expand(x.shape[0])
    dp = v
    dv = torch.zeros_like(v)
    dv[:, 2] = F / m - g  # only z-axis affected by thrust
    dq = torch.zeros_like(q)  # ignore rotation for step response
    dx = torch.cat([dp, dv, dq.unsqueeze(1)], dim=1)
    return dx

def uav_output(x, u, t, params):
    return x  # full state output

if __name__ == "__main__":
    height, width = 4, 4
    num_envs = height * width
    dt = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # UAV parameters (batch)
    m = torch.full((num_envs,), 1.0, device=device)  # kg
    g = torch.full((num_envs,), 9.81, device=device) # m/s^2
    params = Parameters(m=m, g=g)
    # Initial state: [p(3), v(3), q(1)]
    initial_state = torch.zeros(num_envs, 7, device=device)
    # Config
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
    print(f"\033[1;33mSystem configuration:\n{cfg}\033[0m")
    plant = NonlinearSystem(cfg)
    # Step response: thrust step from 0 to hover thrust
    T = 5
    steps = int(T / dt)
    hover_thrust = m * g  # [num_envs]
    u = torch.zeros(num_envs, 4, device=device)
    u[:, 0] = hover_thrust  # step input on thrust
    t_arr = [0.0]
    y = [initial_state]
    for k in range(steps):
        output = plant.step(u)
        y.append(output)
        t_arr.append(t_arr[-1] + dt)
    y = torch.stack(y, dim=1).cpu().numpy()  # [num_envs, steps+1, state_dim]
    # Plot z position for all envs
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(height, width, figsize=(12, 10))
    for idx in range(num_envs):
        i, j = np.unravel_index(idx, (height, width))
        ax = axes[i, j]
        ax.plot(t_arr, y[idx, :, 2], label='z (pos)')
        ax.plot(t_arr, y[idx, :, 5], label='vz (vel)', linestyle='--')
        ax.set_title(f'Env {idx} Step Response')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('State')
        ax.grid()
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "uav_thrust_step_response.png")
    plt.savefig(fig_path)
    print("Step response plot saved to:", fig_path)
    print("\033[1;32mTest completed successfully.\033[0m")
