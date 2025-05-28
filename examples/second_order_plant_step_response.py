"""
second_order_plant_step_response.py
This script demonstrates how to create a second-order system using the InputOutputSystem class from the torchcontrol library.
It simulates the step response of the system with different initial states and visualizes the results.
"""
import os
import io
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg

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
    for k in range(int(T / dt)):
        # Simulate a step input
        output = plant.step(u)  # output: [num_envs, output_dim]
        y.append(output)
        t.append(t[-1] + dt)
    y = torch.cat(y, dim=1).tolist()  # Concatenate outputs and convert to list
    y_hist = y  # For compatibility with render_frame
    # Reference is a step input of 1.0 for all time
    r_hist = [[[1.0] for _ in range(len(y_hist[0]))] for _ in range(num_envs)]
    # Visualization
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, 'second_order_plant_step_response.gif')
    frames = []
    t_arr = [k * dt for k in range(int(T / dt) + 1)]

    def render_frame(frame_idx):
        fig = plt.figure(figsize=(4 * width, 3 * height))
        axes = []
        for idx in range(num_envs):
            i, j = np.unravel_index(idx, (height, width))
            ax = fig.add_subplot(height, width, idx + 1)
            axes.append(ax)
            ax.clear()
            ax.plot(t_arr[:frame_idx+1], y_hist[idx][:frame_idx+1], label='Output (y)')
            ax.plot(t_arr[:frame_idx+1], r_hist[idx][:frame_idx+1], 'r--', label='Reference (r)')
            ax.set_title(f'Env {idx} Step Response')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            ax.set_xlim([0, t_arr[-1]])  # Fix x-axis length
            ax.grid()
            ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = imageio.imread(buf)
        buf.close()
        return img

    print("Rendering frames and creating GIF with multiprocessing, this may take a while...")
    # Save every 10th frame for speed (dt=0.01 -> 100Hz, so 10 frames per second)
    frame_stride = 10
    frame_indices = list(range(0, len(t_arr), frame_stride))
    with ProcessPoolExecutor() as executor:
        frames = list(tqdm(executor.map(render_frame, frame_indices), total=len(frame_indices), desc="Rendering GIF frames"))
    imageio.mimsave(gif_path, frames, duration=0.04)
    print(f"Step response GIF saved to {gif_path}")
    print("\033[1;32mTest completed successfully.\033[0m")