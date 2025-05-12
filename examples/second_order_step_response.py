import os
import matplotlib.pyplot as plt
from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg

if __name__ == "__main__":
    # Example usage
    omega_n = 1.0 # Natural frequency
    zeta = 0.7 # Damping ratio
    num = [omega_n**2]
    den = [1.0, 2.0 * zeta * omega_n, omega_n**2]
    dt = 0.01

    # Create a configuration object
    cfg = InputOutputSystemCfg(numerator=num, denominator=den, dt=dt)

    # Create a plant object using the configuration
    plant = InputOutputSystem(cfg)
    
    # Step response
    T = 20
    u = [1.0]
    y = list()
    for t in range(int(T / dt)):
        # Simulate a step input
        output = plant.step(u)
        y.append(output.item())
        
    # Visualize the output
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    plt.plot([t * dt for t in range(int(T / dt))], y, label='Output')
    plt.plot([t * dt for t in range(int(T / dt))], [u] * int(T / dt), 'r--', label='Input')
    plt.title("Step Response of InputOutputSystem Plant")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.grid()
    plt.savefig(os.path.join(save_dir, "step_response.png"))
    print("Step response plot saved to:", os.path.join(save_dir, "step_response.png"))
    print("Test completed successfully.")
