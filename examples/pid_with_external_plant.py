"""
pid_with_external_plant.py
Example script to test the PID controller in torchcontrol.
"""
import os
import torch
import matplotlib.pyplot as plt
from torchcontrol.controllers import PID, PIDCfg

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
    for k in range(int(T / dt)):
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
    fig, axes = plt.subplots(height, width, figsize=(12, 10))
    t_arr = [k * dt for k in range(int(T / dt) + 1)]
    for k in range(num_envs):
        i, j = divmod(k, width)
        ax = axes[i, j]
        ax.plot(t_arr, y_hist[k], label='Output (y)')
        ax.plot(t_arr, r_hist[k], 'r--', label='Reference (r)')
        # ax.plot(t_arr, u_hist[k], label='Control (u)')
        # ax.plot(t_arr, e_hist[k], label='Error')
        ax.set_title(f'Env {k}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.grid()
        ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'pid_with_external_plant.png')
    plt.savefig(fig_path)
    print(f"PID test plot saved to {fig_path}")
    print("\033[1;32mTest completed successfully.\033[0m")
