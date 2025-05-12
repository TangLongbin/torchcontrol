"""
test_pid.py
Example script to test the PID controller in torchcontrol.
Saves results to examples/results/pid_test_output.txt
"""
from torchcontrol.controllers import PID, PIDCfg
import torch
import os

if __name__ == "__main__":
    # Create PID config and controller
    cfg = PIDCfg(Kp=2.0, Ki=0.5, Kd=0.1, dt=0.1, u_ff=0.0)
    pid = PID(cfg)
    pid.reset()

    # Simulate a step response
    n_steps = 50
    setpoint = torch.tensor(1.0)
    y = torch.tensor(0.0)
    y_hist = []
    u_hist = []
    e_hist = []
    for i in range(n_steps):
        error = setpoint - y
        u = pid.step(error)
        # Simple first-order plant: y_{k+1} = y_k + 0.1*u
        y = y + 0.1 * u
        y_hist.append(y.item())
        u_hist.append(u.item())
        e_hist.append(error.item())

    # Save results
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'pid_test_output.txt')
    with open(output_path, "w") as f:
        f.write("Step\tOutput(y)\tControl(u)\tError\n")
        for i in range(n_steps):
            f.write(f"{i}\t{y_hist[i]:.4f}\t{u_hist[i]:.4f}\t{e_hist[i]:.4f}\n")
    print(f"PID test complete. Results saved to {output_path}")
