"""
uav_mppi_tracking.py
Example: MPPI control of a batch quadrotor UAV using NonlinearSystem and MPPI controller.
Tracks a circular trajectory in the xy-plane for all batch environments.
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
from torchcontrol.controllers import MPPI, MPPICfg
from torchcontrol.utils.math import quaternion_error
from uav_thrust_descent_to_hover import uav_dynamics, uav_output

def cost_function(state_trajs, action_trajs, reference):
    """
    MPPI cost function for UAV tracking (position and orientation), with discount factor.
    Args:
        state_trajs: (num_envs, K, T, state_dim)
        action_trajs: (num_envs, K, T, action_dim)
        reference: (num_envs, T, state_dim)
    Returns:
        cost: (num_envs, K)
    """
    gamma = 0.95  # discount factor
    # Extract position and orientation from state
    pos = state_trajs[..., 0:3]   # (num_envs, K, T, 3)
    quat = state_trajs[..., 6:10]  # (num_envs, K, T, 4)
    # Reference position
    pos_ref = reference[..., 0:3] # (num_envs, T, 3)
    quat_ref = reference[..., 6:10] # (num_envs, T, 4)
    # Expand reference to match rollouts
    pos_ref = pos_ref.unsqueeze(1).expand_as(pos)
    quat_ref = quat_ref.unsqueeze(1).expand_as(quat)
    # Position error
    pos_err = pos - pos_ref
    # Orientation error (quaternion)
    orientation_err = quaternion_error(quat, quat_ref)  # (num_envs, K, T, 3)
    # Control effort (thrust and angular rates)
    u = action_trajs
    thrust = u[..., 0]
    omega = u[..., 1:4]
    # Weights for cost terms
    w_pos = 100.0
    w_orientation = 1.0
    w_thrust = 0.01
    w_omega = 0.01
    # Discount weights for each time step
    T = pos.shape[-2]
    discounts = torch.tensor([gamma ** t for t in range(T)], device=pos.device).view(1, 1, T)
    # Compute cost
    cost = (
        w_pos * (pos_err ** 2).sum(-1) +
        w_orientation * (orientation_err ** 2).sum(-1) +
        w_thrust * (thrust ** 2) +
        w_omega * (omega ** 2).sum(-1)
    )  # (num_envs, K, T)
    cost = cost * discounts  # apply discount
    # Sum over time horizon
    total_cost = cost.sum(-1)  # (num_envs, K)
    # If the height is less than 0, add a large penalty
    height_penalty = (pos[..., 2] < 0).float() * 1e9  # large penalty if z < 0
    total_cost += height_penalty.sum(-1)  # (num_envs, K)

    return total_cost

if __name__ == "__main__":
    # Batch grid size
    height, width = 4, 4
    num_envs = height * width
    dt = 0.02
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.full((num_envs,), 1.0, device=device)
    g = torch.tensor([0.0, 0.0, -9.81], device=device).expand(num_envs, 3)
    params = Parameters(m=m, g=g)
    # Initial state: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
    initial_state = torch.zeros(num_envs, 10, device=device)
    initial_state[:, 6] = 1.0  # Set quaternion to [1,0,0,0] for all envs
    # System configuration
    plant_cfg = NonlinearSystemCfg(
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
    print(f"\033[1;33mSystem configuration:\n{plant_cfg}\033[0m")
    plant = NonlinearSystem(plant_cfg)
    # Trajectory: circle in xy-plane, z=1.0
    T_total = 10
    steps = int(T_total / dt)
    t_arr = torch.linspace(0, T_total, steps+1, device=device)
    radius = 2.0
    num_laps = 3
    omega_traj = 2 * np.pi * num_laps / T_total  # n laps in T seconds
    # Generate different circle centers for each environment
    cx = torch.linspace(-2, 2, width).repeat(height)
    cy = torch.linspace(-2, 2, height).repeat_interleave(width)
    centers = torch.stack([cx, cy], dim=1).to(device)  # (num_envs, 2)
    # Generate reference trajectory for each environment (batch circles)
    x_ref = torch.zeros(num_envs, steps+1, 3, device=device)
    for i in range(num_envs):
        x_ref[i, :, 0] = centers[i, 0] + radius * torch.cos(omega_traj * t_arr)
        x_ref[i, :, 1] = centers[i, 1] + radius * torch.sin(omega_traj * t_arr)
        k = torch.randint(0, 4, (1,), device=device).item()  # Randomly vary height
        x_ref[i, :, 2] = 3.0 + 0.5 * torch.sin(omega_traj*k * t_arr)  # Varying height
    # MPPI controller configuration
    mppi_horizon = 25
    mppi_samples = 1024
    mppi_sigma = torch.tensor([10.0, 2.0, 2.0, 2.0], device=device)  # std for actions
    mppi_alpha = 2.0  # MPPI exploration parameter
    mppi_cfg = MPPICfg(
        K=mppi_samples,
        T=mppi_horizon,
        sigma=mppi_sigma,
        alpha=mppi_alpha,
        u_min=torch.tensor([0.0, -4.0, -4.0, -4.0], device=device),
        u_max=torch.tensor([20.0, 4.0, 4.0, 4.0], device=device),
        cost_function=cost_function,
        dt=dt,
        num_envs=num_envs,
        state_dim=10,
        action_dim=4,
        device=device,
        plant=plant
    )
    print(f"\033[1;33mMPPI configuration:\n{mppi_cfg}\033[0m")
    # Initialize MPPI controller
    controller = MPPI(mppi_cfg)
    # Main simulation loop
    y = [initial_state]
    for k in tqdm(range(steps), desc='Simulating MPPI'):  # Add tqdm progress bar
        # Reference trajectory for the next mppi_horizon steps
        idx_end = min(k + mppi_horizon, steps)
        ref_traj = x_ref[:, k:idx_end, :]
        # Pad if at the end
        if ref_traj.shape[1] < mppi_horizon:
            pad = ref_traj[:, -1:, :].repeat(1, mppi_horizon - ref_traj.shape[1], 1)
            ref_traj = torch.cat([ref_traj, pad], dim=1)
        # Reference state
        ref_state_traj = torch.zeros(num_envs, mppi_horizon, 10, device=device)
        ref_state_traj[:, :, 0:3] = ref_traj
        ref_state_traj[:, :, 6] = 1.0  # Set quaternion to [1,0,0,0] for all envs
        # Current state
        x = y[-1]
        # MPPI step
        u = controller.step(r=ref_state_traj, x=x)
        output = plant.step(u)
        y.append(output)
    y = torch.stack(y, dim=1).cpu().numpy()  # [num_envs, steps+1, state_dim]
    # Plotting results as GIF
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)

    y_cpu = y.copy()  # y is already numpy
    x_ref_cpu = x_ref.cpu().numpy()  # convert to numpy on CPU

    # Efficient GIF creation using multiprocessing (no disk I/O)
    def render_frame(frame):
        fig = plt.figure(figsize=(4 * width, 3 * height))
        axes = []
        for idx in range(num_envs):
            i, j = np.unravel_index(idx, (height, width))
            ax = fig.add_subplot(height, width, idx + 1, projection='3d')
            axes.append(ax)
            ax.plot(y_cpu[idx, :frame, 0], y_cpu[idx, :frame, 1], y_cpu[idx, :frame, 2], label='UAV track')
            ax.plot(x_ref_cpu[idx, :, 0], x_ref_cpu[idx, :, 1], x_ref_cpu[idx, :, 2], 'r--', alpha=0.3, label='Reference')
            ax.set_xlim(np.min(y_cpu[..., 0]), np.max(y_cpu[..., 0]))
            ax.set_ylim(np.min(y_cpu[..., 1]), np.max(y_cpu[..., 1]))
            ax.set_zlim(np.min(y_cpu[..., 2]), np.max(y_cpu[..., 2]))
            ax.set_title(f'Env {idx}')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
            ax.grid()
            ax.legend(fontsize=8)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = imageio.imread(buf)
        buf.close()
        return img

    print("Rendering frames and creating GIF with multiprocessing, this may take a while...")
    frame_indices = list(range(0, y_cpu.shape[1], 5))  # Save every 5th frame for speed
    with ProcessPoolExecutor() as executor:
        images = list(tqdm(executor.map(render_frame, frame_indices), total=len(frame_indices), desc="Rendering frames"))
    gif_path = os.path.join(save_dir, "uav_mppi_tracking.gif")
    imageio.mimsave(gif_path, images, duration=0.04)
    print("MPPI tracking gif saved to:", gif_path)
    print("\033[1;32mMPPI tracking test completed successfully.\033[0m")
