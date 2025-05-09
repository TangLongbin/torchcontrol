"""
pid_control_second_order_cuda.py
Test PID control of a second-order system (InputOutput model) on GPU (CUDA).
"""
from torchcontrol.controllers import PID, PIDCfg
from torchcontrol.plants import InputOutput, InputOutputCfg
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dt = 0.01
    num = [1.0]
    den = [1.0, 2.0, 1.0]
    plant_cfg = InputOutputCfg(num=num, den=den, dt=dt)
    plant = InputOutput(plant_cfg)
    # Move plant parameters to device
    plant.num = plant.num.to(device)
    plant.den = plant.den.to(device)
    plant.u_hist = plant.u_hist.to(device)
    plant.y_hist = plant.y_hist.to(device)

    pid_cfg = PIDCfg(Kp=3.0, Ki=6.0, Kd=0.05, dt=dt, u_ff=0.0)
    pid = PID(pid_cfg)
    # Move PID parameters to device
    pid.Kp = pid.Kp.to(device)
    pid.Ki = pid.Ki.to(device)
    pid.Kd = pid.Kd.to(device)
    pid.dt = pid.dt.to(device)
    pid.u_ff = pid.u_ff.to(device)
    pid.e_k_1 = pid.e_k_1.to(device)
    pid.e_k_2 = pid.e_k_2.to(device)
    pid.u_k_1 = pid.u_k_1.to(device)
    pid.u_k = pid.u_k.to(device)
    pid.u_ff_1 = pid.u_ff_1.to(device)

    n_steps = 500
    setpoint = torch.tensor(1.0, device=device)
    y = torch.tensor(0.0, device=device)
    y_hist = []
    u_hist = []
    r_hist = []
    for i in range(n_steps):
        error = setpoint - y
        u = pid.step(error)
        # Ensure u is on device
        u = u.to(device)
        y = plant.step(u)
        y_hist.append(y.item())
        u_hist.append(u.item())
        r_hist.append(setpoint.item())

    # Visualization (move to cpu for plotting)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
    os.makedirs(output_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8,4))
    t = np.arange(n_steps) * dt
    ax1.plot(t, y_hist, label='System Output y', color='b')
    ax1.plot(t, r_hist, '--', label='Setpoint r', color='g')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Output')
    ax2 = ax1.twinx()
    ax2.plot(t, u_hist, label='Control u', color='r', alpha=0.5)
    ax2.set_ylabel('Control Signal')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('PID Control of Second-Order System (CUDA)')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'pid_control_second_order_cuda.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Simulation complete. Figure saved to {fig_path}")
