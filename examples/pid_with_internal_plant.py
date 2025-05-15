"""
pid_with_internal_plant.py
Example script to test the PID controller in torchcontrol with an internal second-order plant (InputOutputSystem).
"""
import os
import torch
import matplotlib.pyplot as plt
from torchcontrol.controllers import PID, PIDCfg
from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg

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
    for k in range(int(T / dt)):
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
    fig_path = os.path.join(save_dir, 'pid_with_internal_plant.png')
    plt.savefig(fig_path)
    print(f"PID test plot saved to {fig_path}")
    print("\033[1;32mTest completed successfully.\033[0m")
