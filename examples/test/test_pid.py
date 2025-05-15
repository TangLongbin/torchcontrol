"""
test_pid.py
Example script to test the PID controller in torchcontrol.
"""
import os
import torch
import matplotlib.pyplot as plt
from torchcontrol.controllers import PID, PIDCfg

if __name__ == "__main__":
    # Create PID config and controller
    Kp = 1.0
    Ki = 0.01
    Kd = 0.001
    u_ff = 0.0
    dt = 0.01
    height, width = 4, 4
    num_envs = height * width
    torch.manual_seed(42)  # Set seed for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a PID controller
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
    for k in range(int(T / dt)):
        e = setpoint - y  # [num_envs]
        u = pid.step(x=y, r=setpoint)  # [num_envs]
        y = y + dt * u  # Simple first-order plant
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
    fig, axes = plt.subplots(height, width, figsize=(12, 10))
    t_arr = [k * dt for k in range(int(T / dt) + 1)]
    for k in range(num_envs):
        i, j = divmod(k, width)
        ax = axes[i, j]
        ax.plot(t_arr, r_hist[k], label='Reference (r)')
        ax.plot(t_arr, y_hist[k], label='Output (y)')
        # ax.plot(t_arr, u_hist[k], label='Control (u)')
        # ax.plot(t_arr, e_hist[k], label='Error')
        ax.set_title(f'Env {k}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.grid()
        ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'pid_test_plot.png')
    plt.savefig(fig_path)
    print(f"PID test plot saved to {fig_path}")
