"""
pid_control_second_order.py
PID control of a second-order system (InputOutputSystem model).
Visualizes system output, setpoint, and control signal.
"""
from torchcontrol.controllers import PID, PIDCfg
from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg
import numpy as np
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    # Second-order system: G(s) = 1 / (s^2 + 2s + 1)
    # Discretize with dt=0.01, num=[1], den=[1,2,1]
    dt = 0.01
    num = [1.0]
    den = [1.0, 2.0, 1.0]
    plant_cfg = InputOutputSystemCfg(num=num, den=den, dt=dt)
    plant = InputOutputSystem(plant_cfg)

    # PID controller parameters (tuned for this plant, smaller values to avoid overflow)
    pid_cfg = PIDCfg(Kp=3.0, Ki=6.0, Kd=0.05, dt=dt, u_ff=0.0)
    pid = PID(pid_cfg)
    pid.reset()
    plant.reset()

    n_steps = 500
    setpoint = 1.0
    y = torch.tensor(0.0)
    y_hist = []
    u_hist = []
    r_hist = []
    for i in range(n_steps):
        error = torch.tensor(setpoint) - y
        u = pid.step(error)
        y = plant.step(u)
        y_hist.append(y.item())
        u_hist.append(u.item())
        r_hist.append(setpoint)

    # Visualization
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
    plt.title('PID Control of Second-Order System')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'pid_control_second_order.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Simulation complete. Figure saved to {fig_path}")
