"""
uav_geometric_tracking.py
Example: Geometric tracking control of a batch quadrotor UAV using NonlinearSystem and a geometric controller (PID-based outer loop).
Tracks a circular trajectory in the xy-plane for all batch environments.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchcontrol.plants.nonlinear_system import NonlinearSystem
from torchcontrol.plants.nonlinear_system_cfg import NonlinearSystemCfg, Parameters
from torchcontrol.utils.math import quaternion_to_dcm
from torchcontrol.controllers.pid import PID
from torchcontrol.controllers.pid_cfg import PIDCfg
from uav_thrust_descent_to_hover import uav_dynamics, uav_output

class GeometricTrackingController(PID):
    def __init__(self, m, g, dt, num_envs, device, kp=8.0, kd=4.0):
        # PID config for position+velocity (outer loop)
        state_dim = 3  # x, y, z
        action_dim = 3  # a_des (x, y, z)
        cfg = PIDCfg(
            Kp=kp * torch.ones(action_dim),
            Ki=torch.zeros(action_dim),
            Kd=kd * torch.ones(action_dim),
            u_ff=torch.zeros(action_dim),
            num_envs=num_envs,
            state_dim=state_dim,
            action_dim=action_dim,
            dt=dt,
        )
        super().__init__(cfg)
        self.m = m
        self.g = g
        # Remove redundant .to(device) calls for Kp, Ki, Kd, u_ff
        self.e3 = torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_envs, 3)
        self.kR = 4.0  # attitude gain

    def step(self, x, v, q, x_ref, v_ref, a_ref, yaw_ref, yaw_rate_ref):
        # Ensure all tensors are on the correct device
        device = self.device
        x = x.to(device)
        v = v.to(device)
        q = q.to(device)
        x_ref = x_ref.to(device)
        v_ref = v_ref.to(device)
        a_ref = a_ref.to(device)
        yaw_ref = yaw_ref.to(device)
        yaw_rate_ref = yaw_rate_ref.to(device)
        # 1. Outer loop: position+velocity PID to get a_des
        a_fb = self.forward(x, x_ref) + self.Kd[0, 0] * (v_ref - v)  # batch PD
        a_des = a_fb + self.g + a_ref  # (num_envs, 3)
        # 2. Desired body z axis
        b3_des = a_des / (a_des.norm(dim=1, keepdim=True) + 1e-6)
        # 3. Desired body x axis from yaw
        b1_des = torch.stack([torch.cos(yaw_ref), torch.sin(yaw_ref), torch.zeros_like(yaw_ref)], dim=1)
        # 4. Desired body y axis
        b2_des = torch.cross(b3_des, b1_des, dim=1)
        b2_des = b2_des / (b2_des.norm(dim=1, keepdim=True) + 1e-6)
        b1_des = torch.cross(b2_des, b3_des, dim=1)
        # 5. Desired rotation matrix
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=2)  # (num_envs, 3, 3)
        # 6. Current rotation matrix
        R = quaternion_to_dcm(q)
        # 7. Attitude error (SO(3) vee map)
        e_R_mat = 0.5 * (torch.matmul(R_des.transpose(1,2), R) - torch.matmul(R.transpose(1,2), R_des))
        e_R = torch.stack([e_R_mat[:,2,1], e_R_mat[:,0,2], e_R_mat[:,1,0]], dim=1)
        # 8. Thrust (projected to body z)
        F = (self.m * (a_des * (R @ self.e3.unsqueeze(2)).squeeze(2)).sum(dim=1)).clamp(min=0.0)
        # 9. Omega_cmd (no feedforward, only feedback for hover/tracking)
        omega_cmd = self.kR * e_R  # (num_envs, 3)
        # 10. Output action
        u = torch.zeros(self.num_envs, 4, device=self.device)
        u[:, 0] = F
        u[:, 1:4] = omega_cmd
        return u

if __name__ == "__main__":
    # Batch grid size
    height, width = 4, 4
    num_envs = height * width
    dt = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.full((num_envs,), 1.0, device=device)
    g = torch.tensor([0.0, 0.0, -9.81], device=device).expand(num_envs, 3)
    params = Parameters(m=m, g=g)
    # Initial state: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
    initial_state = torch.zeros(num_envs, 10, device=device)
    initial_state[:, 6] = 1.0  # Set quaternion to [1,0,0,0] for all envs
    initial_state[:, 2] = 0.0  # z=0
    initial_state[:, 0] = 1.0  # x=1 (start on circle)
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
    controller = GeometricTrackingController(m, g, dt, num_envs, device)
    # Trajectory: circle in xy-plane, z=1.0
    T = 10
    steps = int(T / dt)
    t_arr = torch.linspace(0, T, steps+1, device=device)
    radius = 1.0
    omega_traj = 2 * np.pi / T  # one circle in T seconds
    x_ref = torch.stack([
        radius * torch.cos(omega_traj * t_arr),
        radius * torch.sin(omega_traj * t_arr),
        torch.ones_like(t_arr)
    ], dim=1)  # (steps+1, 3)
    v_ref = torch.stack([
        -radius * omega_traj * torch.sin(omega_traj * t_arr),
        radius * omega_traj * torch.cos(omega_traj * t_arr),
        torch.zeros_like(t_arr)
    ], dim=1)
    a_ref = torch.stack([
        -radius * omega_traj**2 * torch.cos(omega_traj * t_arr),
        -radius * omega_traj**2 * torch.sin(omega_traj * t_arr),
        torch.zeros_like(t_arr)
    ], dim=1)
    yaw_ref = omega_traj * t_arr  # optional: face along tangent
    yaw_rate_ref = torch.full_like(t_arr, omega_traj)
    # Main simulation loop
    y = [initial_state]
    for k in range(steps):
        x = y[-1][:, 0:3]
        v = y[-1][:, 3:6]
        q = y[-1][:, 6:10]
        # batch reference for all envs
        x_ref_batch = x_ref[k].unsqueeze(0).expand(num_envs, 3)
        v_ref_batch = v_ref[k].unsqueeze(0).expand(num_envs, 3)
        a_ref_batch = a_ref[k].unsqueeze(0).expand(num_envs, 3)
        yaw_ref_batch = yaw_ref[k].repeat(num_envs)
        yaw_rate_ref_batch = yaw_rate_ref[k].repeat(num_envs)
        u = controller.step(x, v, q, x_ref_batch, v_ref_batch, a_ref_batch, yaw_ref_batch, yaw_rate_ref_batch)
        output = plant.step(u)
        y.append(output)
    y = torch.stack(y, dim=1).cpu().numpy()  # [num_envs, steps+1, state_dim]
    # Plotting results
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(height, width, figsize=(12, 10))
    for idx in range(num_envs):
        i, j = np.unravel_index(idx, (height, width))
        ax = axes[i, j]
        ax.plot(y[idx, :, 0], y[idx, :, 1], label='xy (track)')
        ax.plot(x_ref[:, 0].cpu(), x_ref[:, 1].cpu(), 'r--', label='xy_ref')
        ax.set_title(f'Env {idx} Circle Track')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.axis('equal')
        ax.grid()
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "uav_geometric_tracking.png")
    plt.savefig(fig_path)
    print("Geometric tracking plot saved to:", fig_path)
    print("\033[1;32mTracking test completed successfully.\033[0m")
