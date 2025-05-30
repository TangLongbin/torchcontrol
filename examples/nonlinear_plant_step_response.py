"""
nonlinear_plant_step_response.py
Example: Step response of a batch nonlinear system using NonlinearSystem.
"""
import os
import torch
import numpy as np

from tqdm import tqdm

from torchcontrol.system import Parameters
from torchcontrol.plants import NonlinearSystem, NonlinearSystemCfg
from torchcontrol.utils.visualization import render_batch_gif

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
    for k in tqdm(range(int(T / dt)), desc="Simulating step response"):
        output = plant.step(u)  # [num_envs, state_dim]
        y.append(output)
        t.append(t[-1] + dt)
    u = u.repeat(1, len(y)) # shape: [num_envs, num_steps]
    y = torch.stack(y, dim=1) # shape: [num_envs, num_steps, state_dim]


    """
    Visualize x1 (position), x2 (velocity), and input (u) as animated GIF
    """
    # Save directory for results
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, "nonlinear_plant_step_response.gif")
    
    # Title for each environment
    titles = [f"Env {i} Step Response" for i in range(num_envs)]
    
    # Let time be the x-axis
    t = np.array(t)  # shape: [num_steps]
    x_hist = np.tile(t[np.newaxis, :, np.newaxis], (num_envs, 1, 3))  # [num_envs, num_steps, 3]
    xlim = [0, t[-1]]  # x-axis limits
    xlabel = "Time (s)"
    
    # Let x1 (pos), x2 (vel) and u (input) be the y-axis curves
    x1 = y[:, :, 0].cpu().numpy()  # Position, shape: [num_envs, num_steps]
    x2 = y[:, :, 1].cpu().numpy()  # Velocity, shape: [num_envs, num_steps]
    u = u.cpu().numpy()  # Input, shape: [num_envs, num_steps]
    # Stack y to include reference for visualization
    y_hist = np.stack([x1, x2, u], axis=-1)  # [num_envs, num_steps, 3]
    labels = ["x1 (pos)", "x2 (vel)", "u (input)"]  # Labels for each curve
    line_styles = ['-', '--', 'r--'] # Line styles for each curve
    ylabel = "Value"

    # Use render_batch_gif utility for multi-curve GIF rendering
    render_batch_gif(
        gif_path=gif_path,
        x_hist=x_hist, # [num_envs, num_steps, 1]
        y_hist=y_hist, # [num_envs, num_steps, 3]
        width=width,
        height=height,
        labels=labels,
        line_styles=line_styles,
        titles=titles,
        frame_stride=10,
        duration=0.04,
        xlim=xlim,
        ylabel=ylabel,
        xlabel=xlabel,
    )
    print("\033[1;32mTest completed successfully.\033[0m")
