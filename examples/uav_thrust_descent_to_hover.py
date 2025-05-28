"""
uav_thrust_descent_to_hover.py
Example: Batch quadrotor UAV descent and hover using NonlinearSystem.
This simulates a UAV starting with downward velocity and 90% hover thrust,
switching to 100% hover thrust when vertical speed reaches zero, to achieve hover.
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
from torchcontrol.utils.math import quaternion_to_dcm, omega_quat_matrix

def uav_dynamics(x, u, t, params):
    """
    UAV batch nonlinear dynamics for 3D quadrotor with quaternion attitude.
    Args:
        x: State tensor (num_envs, 10) [pos(3), vel(3), quat(4)]
        u: Action tensor (num_envs, 4) [thrust, omega_x, omega_y, omega_z]
        t: Time (unused)
        params: Parameters object with mass m and gravity g (num_envs, 3)
    Returns:
        dx: State derivative (num_envs, 10)
    """
    m = params.m # mass (num_envs,)
    g = params.g # gravity vector (num_envs, 3)
    p = x[:, 0:3]  # position
    v = x[:, 3:6]  # velocity
    q = x[:, 6:10] # quaternion
    omega = u[:, 1:4]  # body angular velocity
    F = u[:, 0].unsqueeze(-1)  # thrust (N)

    # Thrust vector in body frame (z axis)
    thrust_vec = torch.cat([torch.zeros_like(F), torch.zeros_like(F), F], dim=-1)

    # Rotation matrix from quaternion
    C_B_I = quaternion_to_dcm(q)
    C_I_B = C_B_I.transpose(1, 2)

    # State derivatives
    dp = v
    dv = g + torch.matmul(C_I_B, thrust_vec.unsqueeze(-1)).squeeze(-1) / m.unsqueeze(-1)
    dq = 0.5 * torch.matmul(omega_quat_matrix(omega), q.unsqueeze(-1)).squeeze(-1)
    dx = torch.cat([dp, dv, dq], dim=1)
    return dx

def uav_output(x, u, t, params):
    """Output function (identity)."""
    return x

if __name__ == "__main__":
    # Batch grid size
    height, width = 4, 4
    num_envs = height * width
    dt = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # UAV parameters
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

    # Simulation parameters
    T = 5
    steps = int(T / dt)
    # Compute hover thrust as norm of gravity * mass (per env)
    hover_thrust = m * g[:, 2].abs()  # (num_envs,)
    u = torch.zeros(num_envs, 4, device=device)
    t_arr = [0.0]
    y = [initial_state]
    vz_crossed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Main simulation loop
    for k in range(steps):
        last_vz = y[-1][:, 5]
        vz_now_crossed = (~vz_crossed) & (last_vz <= 0)
        vz_crossed = vz_crossed | vz_now_crossed
        u[:, 0] = hover_thrust * 0.9  # 90% thrust for descent
        u[vz_crossed, 0] = hover_thrust[vz_crossed]  # Switch to hover thrust
        output = plant.step(u)
        y.append(output)
        t_arr.append(t_arr[-1] + dt)

    y = torch.stack(y, dim=1).cpu().numpy()
    x = y  # For compatibility with render_frame
    # Visualize descent as animated GIF
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, "uav_thrust_descent_to_hover.gif")
    frames = []
    t_arr = [k * dt for k in range(steps + 1)]

    def render_frame(frame_idx):
        fig = plt.figure(figsize=(4 * width, 3 * height))
        axes = []
        for idx in range(num_envs):
            i, j = np.unravel_index(idx, (height, width))
            ax = fig.add_subplot(height, width, idx + 1)
            axes.append(ax)
            ax.clear()
            ax.plot(t_arr[:frame_idx+1], x[idx, :frame_idx+1, 2], label='z (altitude)')
            ax.plot(t_arr[:frame_idx+1], x[idx, :frame_idx+1, 5], label='vz (velocity)', linestyle='--', alpha=0.7)
            ax.set_title(f'Env {idx} Descent -> Hover')
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
    print(f"Descent GIF saved to {gif_path}")
    print("\033[1;32mTest completed successfully.\033[0m")
