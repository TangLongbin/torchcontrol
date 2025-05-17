"""
nonlinear_plant_step_response.py
Example: Step response of a batch nonlinear system using NonlinearSystem.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchcontrol.plants.nonlinear_system import NonlinearSystem
from torchcontrol.plants.nonlinear_system_cfg import NonlinearSystemCfg, Parameters

def nonlinear_oscillator(x, u, t, params):
    # x: [num_envs, 2], u: [num_envs, 1], t: scalar or [num_envs], params: Parameters
    k = params.k  # [num_envs] or scalar
    c = params.c
    alpha = params.alpha
    x1 = x[:, 0]
    x2 = x[:, 1]
    u = u.squeeze(-1) if u.dim() > 1 else u
    dx1 = x2
    dx2 = -k * x1 - c * x2 + alpha * x1 ** 3 + u
    dx = torch.stack([dx1, dx2], dim=1)
    return dx

def nonlinear_output(x, u, t, params):
    # x: [num_envs, 2], u: [num_envs, 1], t: scalar or [num_envs], params: Parameters
    return x # full state output

if __name__ == "__main__":
    # Batch size and device
    height, width = 4, 4
    num_envs = height * width
    dt = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters (same for all envs, but could be different)
    k = torch.full((num_envs,), 1.0) # Spring constant
    c = torch.full((num_envs,), 0.7) # Damping coefficient
    alpha = torch.full((num_envs,), 0.1) # Nonlinear coefficient, should be less than 0.15
    params = Parameters(k=k, c=c, alpha=alpha)

    # Initial states: random in [0,2] for x1, zero for x2
    torch.manual_seed(42)
    x1_0 = torch.rand(num_envs, device=device) * 2
    x2_0 = torch.zeros(num_envs, device=device)
    initial_states = torch.stack([x1_0, x2_0], dim=1)  # [num_envs, state_dim]

    # Config
    cfg = NonlinearSystemCfg(
        dynamics=nonlinear_oscillator,
        output=nonlinear_output,
        dt=dt,
        num_envs=num_envs,
        state_dim=2,
        action_dim=1,
        initial_state=initial_states,
        params=params,
        device=device,
    )
    print(f"\033[1;33mSystem configuration:\n{cfg}\033[0m")

    # Create system
    plant = NonlinearSystem(cfg)

    # Step response
    T = 20
    u = torch.ones(num_envs, 1, device=device)  # Step input for all envs
    t = [0.0]
    y = [initial_states]
    for k in range(int(T / dt)):
        output = plant.step(u)  # [num_envs, state_dim]
        y.append(output)
        t.append(t[-1] + dt)
    y = torch.stack(y, dim=1).cpu().numpy()  # [num_envs, steps+1, state_dim]

    # Visualize x1 (position)
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(height, width, figsize=(12, 10))
    for idx in range(num_envs):
        i, j = np.unravel_index(idx, (height, width))
        ax = axes[i, j]
        ax.plot(t, y[idx, :, 0], label='x1 (pos)')
        ax.plot(t, y[idx, :, 1], label='x2 (vel)', linestyle='--', alpha=0.7)
        ax.plot(t, np.ones_like(t), 'r--', label='Input')
        ax.set_title(f'Env {idx} Step Response')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('State')
        ax.grid()
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "nonlinear_plant_step_response.png")
    plt.savefig(fig_path)
    print("Step response plot saved to:", fig_path)
    print("\033[1;32mTest completed successfully.\033[0m")
