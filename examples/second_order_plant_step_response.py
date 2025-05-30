"""
second_order_plant_step_response.py
This script demonstrates how to create a second-order system using the InputOutputSystem class from the torchcontrol library.
It simulates the step response of the system with different initial states and visualizes the results.
"""
import os
import torch
import numpy as np

from tqdm import tqdm

from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg
from torchcontrol.utils.visualization import render_batch_gif

if __name__ == "__main__":
    # Example usage
    omega_n = 1.0 # Natural frequency
    zeta = 0.7 # Damping ratio
    num = [omega_n**2]
    den = [1.0, 2.0 * zeta * omega_n, omega_n**2]
    height, width = 4, 4
    num_envs = height * width
    dt = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 16 different initial states for each env (random values in [0,2])
    torch.manual_seed(42) # Set seed for reproducibility
    initial_states = torch.rand(num_envs, 1, device=device)*2 # shape: [num_envs, 1]

    # Create a configuration object
    cfg = InputOutputSystemCfg(
        numerator=num,
        denominator=den,
        dt=dt,
        num_envs=num_envs,
        initial_state=initial_states,
        device=device,
    )
    print(f"\033[1;33mSystem configuration:\n{cfg}\033[0m")

    # Create a plant object using the configuration
    plant = InputOutputSystem(cfg)
    
    # Step response
    T = 20
    u = [1.0]
    t = [0.0]
    y = [initial_states]
    for k in tqdm(range(int(T / dt)), desc="Simulating step response"):
        # Simulate a step input
        output = plant.step(u)  # output: [num_envs, output_dim]
        y.append(output)
        t.append(t[-1] + dt)
    y = torch.cat(y, dim=1).cpu().numpy()  # [num_envs, num_steps]
    u = np.ones_like(y)  # Step input u=1 for all time, [num_envs, num_steps]

    # Visualization
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, 'second_order_plant_step_response.gif')

    # Prepare time axis for x_hist
    t_arr = np.array([k * dt for k in range(int(T / dt) + 1)])
    x_hist = np.tile(t_arr[np.newaxis, :, np.newaxis], (num_envs, 1, 2))  # [num_envs, num_steps, 2]
    xlim = [0, t_arr[-1]]
    xlabel = "Time (s)"

    # Prepare y_hist for output and input
    y_hist = np.stack([
        y,           # Output (y), shape [num_envs, num_steps]
        u,            # Input (u), shape [num_envs, num_steps]
    ], axis=-1)  # [num_envs, num_steps, 2]
    labels = ["Output (y)", "Input (u)"]
    line_styles = ['-', 'r--']
    ylabel = "Value"
    titles = [f"Env {i} Step Response" for i in range(num_envs)]

    # Use render_batch_gif utility for batch GIF rendering
    render_batch_gif(
        gif_path=gif_path,
        x_hist=x_hist,
        y_hist=y_hist,
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