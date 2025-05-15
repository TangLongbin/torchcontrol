"""
second_order_plant_step_response.py
This script demonstrates how to create a second-order system using the InputOutputSystem class from the torchcontrol library.
It simulates the step response of the system with different initial states and visualizes the results.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg

if __name__ == "__main__":
    # Example usage
    omega_n = 1.0 # Natural frequency
    zeta = 0.7 # Damping ratio
    num = [omega_n**2]
    den = [1.0, 2.0 * zeta * omega_n, omega_n**2]
    height, width = 4, 4
    num_envs = height * width
    dt = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 16 different initial states for each env (random values in [0,2])
    torch.manual_seed(42) # Set seed for reproducibility
    initial_states = torch.rand(num_envs, 1, device=device)*2 # shape: [num_envs, 1]

    # Create a configuration object
    cfg = InputOutputSystemCfg(
        numerator=num,
        denominator=den,
        dt=dt,
        num_envs=num_envs,
        initial_state=initial_states,
        device=device,
    )
    print(f"\033[1;33mSystem configuration:\n{cfg}\033[0m")

    # Create a plant object using the configuration
    plant = InputOutputSystem(cfg)
    
    # Step response
    T = 20
    u = [1.0]
    t = [0.0]
    y = [initial_states]
    for k in range(int(T / dt)):
        # Simulate a step input
        output = plant.step(u)  # output: [num_envs, output_dim]
        y.append(output)
        t.append(t[-1] + dt)
    y = torch.cat(y, dim=1).tolist()  # Concatenate outputs and convert to list
    
    # Visualize the output
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(height, width, figsize=(12, 10))
    for k in range(num_envs):
        # index to xy
        i, j = np.unravel_index(k, (height, width))
        ax = axes[i, j]
        ax.plot(t, y[k], label='Output')
        ax.plot(t, u * len(t), 'r--', label='Input')
        ax.set_title(f'Env {k} Step Response')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Output')
        ax.grid()
        ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'second_order_plant_step_response.png')
    plt.savefig(fig_path)
    print("Step response plot saved to:", fig_path)
    print("\033[1;32mTest completed successfully.\033[0m")