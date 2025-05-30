"""
pid_with_external_plant.py
Example script to test the PID controller in torchcontrol.
"""
import os
import torch
import numpy as np

from tqdm import tqdm

from torchcontrol.controllers import PID, PIDCfg
from torchcontrol.utils.visualization import render_batch_gif

if __name__ == "__main__":
    # Create PID config and controller
    Kp = 5.0
    Ki = 0.1
    Kd = 0.05
    u_ff = 0.0
    dt = 0.01
    height, width = 4, 4
    num_envs = height * width
    torch.manual_seed(42)  # Set seed for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # External plant dynamics
    plant = lambda x, u: x + dt * u # Simple first-order plant dynamics
    print(f"\033[1;33mExternal Plant: First-order system\nx(t+1) = x(t) + dt * u\n\033[0m")

    # PID controller configuration
    cfg = PIDCfg(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        u_ff=u_ff,
        dt=dt,
        num_envs=num_envs,
        state_dim=1,
        action_dim=1,
        device=device
    )
    print(f"\033[1;33mPID Controller configuration:\n{cfg}\033[0m")
    
    # Create PID controller
    pid = PID(cfg)

    # Simulate a step response
    T = 10.0  # Total time

    # Multiple environment initial outputs
    y = torch.rand((num_envs, pid.state_dim), device=device) * 2  # Each environment has a different initial output
    setpoint = torch.ones((num_envs, pid.state_dim), device=device)  # All environments have a setpoint of 1
    r_hist = [setpoint.clone()]  # Initialize reference history
    y_hist = [y.clone()]
    u_hist = [torch.zeros((num_envs, pid.action_dim), device=device)]  # Initialize control output history
    e_hist = [torch.zeros((num_envs, pid.state_dim), device=device)]  # Initialize error history
    for k in tqdm(range(int(T / dt)), desc="Simulating PID control"):
        e = setpoint - y  # [num_envs]
        u = pid.step(x=y, r=setpoint)  # [num_envs]
        y = plant(y, u)  # Update plant state
        r_hist.append(setpoint.clone())  # Append reference
        y_hist.append(y.clone())
        u_hist.append(u.clone())
        e_hist.append(e.clone())
    r_hist = torch.cat(r_hist, dim=1).tolist()  # Concatenate references and convert to list
    y_hist = torch.cat(y_hist, dim=1).tolist()  # Concatenate outputs and convert to list
    u_hist = torch.cat(u_hist, dim=1).tolist()  # Concatenate control outputs and convert to list
    e_hist = torch.cat(e_hist, dim=1).tolist()  # Concatenate errors and convert to list

    # Visualization
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, 'pid_with_external_plant.gif')

    # Prepare time axis for x_hist
    t_arr = np.array([k * dt for k in range(int(T / dt) + 1)])
    x_hist = np.tile(t_arr[np.newaxis, :, np.newaxis], (num_envs, 1, 2))  # [num_envs, num_steps, 2]
    xlim = [0, t_arr[-1]]
    xlabel = "Time (s)"

    # Prepare y_hist for output and reference
    y_hist_np = np.stack([
        np.array(y_hist),  # Output (y), shape [num_envs, num_steps]
        np.array(r_hist),  # Reference (r), shape [num_envs, num_steps]
    ], axis=-1)  # [num_envs, num_steps, 2]
    labels = ["Output (y)", "Reference (r)"]
    line_styles = ['-', 'r--']
    ylabel = "Value"
    titles = [f"Env {i}" for i in range(num_envs)]

    # Use render_batch_gif utility for batch GIF rendering
    render_batch_gif(
        gif_path=gif_path,
        x_hist=x_hist,
        y_hist=y_hist_np,
        width=width,
        height=height,
        labels=labels,
        line_styles=line_styles,
        titles=titles,
        frame_stride=5,
        duration=0.04,
        xlim=xlim,
        ylabel=ylabel,
        xlabel=xlabel,
    )
    print("\033[1;32mTest completed successfully.\033[0m")
