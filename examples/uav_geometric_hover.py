"""
uav_geometric_hover.py
Example: Geometric hover control of a batch quadrotor UAV using NonlinearSystem and a simple geometric controller.
"""
import os
import io
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from torchcontrol.system import Parameters
from torchcontrol.plants import NonlinearSystem, NonlinearSystemCfg
from uav_thrust_descent_to_hover import uav_dynamics, uav_output  # Use absolute import for script execution

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
        # x: [num_envs, 10]
        z = x[:, 2]
        vz = x[:, 5]
        e = self.z_ref - z
        de = -vz
        # Use only z-axis gravity for thrust control
        u_thrust = self.m * (self.g[:, 2].abs() + self.kp * e + self.kd * de)
        u = torch.zeros(self.num_envs, 4, device=self.device)
        u[:, 0] = u_thrust
        return u


if __name__ == "__main__":
    # Batch grid size
    height, width = 4, 4
    num_envs = height * width
    dt = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # UAV parameters (consistent with uav_thrust_descent_to_hover.py)
    m = torch.full((num_envs,), 1.0, device=device)
    # Define gravity as a 3D vector and broadcast to (num_envs, 3)
    g = torch.tensor([0.0, 0.0, -9.81], device=device).expand(num_envs, 3)
    params = Parameters(m=m, g=g)

    # Initial state: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
    initial_state = torch.zeros(num_envs, 10, device=device)
    initial_state[:, 6] = 1.0  # Set quaternion to [1,0,0,0] for all envs
    initial_state[:, 5] = 1.0  # Set initial vz = 1.0 m/s for all envs (to see response)

    # System configuration
    cfg = NonlinearSystemCfg(
        dynamics=uav_dynamics,
        output=uav_output,
        dt=dt,
        num_envs=num_envs,
        state_dim=10,
        action_dim=4,
        initial_state=initial_state,
        params=params,
        device=device,
    )
    print(f"\033[1;33mSystem configuration:\n{cfg}\033[0m")

    plant = NonlinearSystem(cfg)
    controller = GeometricHoverController(m, g, dt, num_envs, device)

    # Simulation parameters
    T = 5
    steps = int(T / dt)
    t_arr = [0.0]
    y = [initial_state]

    # Main simulation loop
    for k in range(steps):
        u = controller.step(y[-1])
        output = plant.step(u)
        y.append(output)
        t_arr.append(t_arr[-1] + dt)

    y = torch.stack(y, dim=1).cpu().numpy()  # [num_envs, steps+1, state_dim]

    # Visualize tracking as animated GIF
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, "uav_geometric_hover.gif")
    frames = []
    t_arr = [k * dt for k in range(steps + 1)]

    def render_frame(frame_idx):
        fig = plt.figure(figsize=(4 * width, 3 * height))
        axes = []
        for idx in range(num_envs):
            i, j = np.unravel_index(idx, (height, width))
            ax = fig.add_subplot(height, width, idx + 1)  # 2D subplot, no projection
            axes.append(ax)
            ax.clear()
            ax.plot(t_arr[:frame_idx+1], y[idx, :frame_idx+1, 2], label='z (altitude)')
            ax.plot(t_arr[:frame_idx+1], np.ones_like(t_arr[:frame_idx+1]), 'r--', label='Reference (z=1)')
            ax.plot(t_arr[:frame_idx+1], y[idx, :frame_idx+1, 5], label='vz (vertical speed)', linestyle='--', alpha=0.7)
            ax.set_title(f'Env {idx} Hover')
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
    print(f"Hover GIF saved to {gif_path}")
    print("\033[1;32mTest completed successfully.\033[0m")
