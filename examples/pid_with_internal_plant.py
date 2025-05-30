"""
pid_with_internal_plant.py
Example script to test the PID controller in torchcontrol with an internal second-order plant (InputOutputSystem).
"""
import os
import torch
import numpy as np

from tqdm import tqdm

from torchcontrol.controllers import PID, PIDCfg
from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg
from torchcontrol.utils.visualization import render_batch_gif

if __name__ == "__main__":
    # System and simulation parameters
    Kp = 120.0
    Ki = 600.0
    Kd = 30.0
    u_ff = 0.0
    dt = 0.01
    height, width = 4, 4
    num_envs = height * width
    torch.manual_seed(42)  # Set seed for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Second-order system: G(s) = 1 / (s^2 + 2s + 1)
    num = [1.0]
    den = [1.0, 2.0, 1.0]
    # Random initial states for each environment (output y0 in [0,2])
    initial_states = torch.rand(num_envs, 1, device=device) * 2

    # Create plant config and plant
    plant_cfg = InputOutputSystemCfg(
        numerator=num,
        denominator=den,
        dt=dt,
        num_envs=num_envs,
        initial_state=initial_states,
        device=device,
    )
    plant = InputOutputSystem(plant_cfg)
    print(f"\033[1;33mInternal Plant: Second-order system\n{plant_cfg}\033[0m")

    # PID controller configuration (attach plant)
    pid_cfg = PIDCfg(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        u_ff=u_ff,
        dt=dt,
        num_envs=num_envs,
        state_dim=1,
        action_dim=1,
        device=device,
        plant=plant
    )
    pid = PID(pid_cfg)
    print(f"\033[1;33mPID Controller configuration:\n{pid_cfg}\033[0m")

    # Simulate a step response
    T = 10.0  # Total time
    setpoint = torch.ones((num_envs, 1), device=device)  # Step reference
    r_hist = [setpoint.clone()]
    y_hist = [initial_states.clone()]
    u_hist = [torch.zeros((num_envs, 1), device=device)]
    e_hist = [torch.zeros((num_envs, 1), device=device)]
    pid.reset()
    for k in tqdm(range(int(T / dt)), desc="Simulating PID control"):
        u = pid.step(r=setpoint)  # PID computes new control based on reference
        y = plant.step(u)  # Plant computes new output based on control
        e = setpoint - y  # Compute the error
        r_hist.append(setpoint.clone())
        y_hist.append(y.clone())
        u_hist.append(u.clone())
        e_hist.append(e.clone())
    r_hist = torch.cat(r_hist, dim=1).tolist()
    y_hist = torch.cat(y_hist, dim=1).tolist()
    u_hist = torch.cat(u_hist, dim=1).tolist()
    e_hist = torch.cat(e_hist, dim=1).tolist()

    # Visualization
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, 'pid_with_internal_plant.gif')

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
