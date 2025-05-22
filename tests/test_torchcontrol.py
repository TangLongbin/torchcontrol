"""
test_torchcontrol.py
Test the modules of the torchcontrol package.
This script will be tested by github actions with pytest.
All test functions should be named with the prefix "test_".
"""
import torch
from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg

def test_second_order_plant_step_response():
    """
    Test the step response of a second-order system using the InputOutputSystem class.
    This function creates a second-order system with given numerator and denominator coefficients,
    simulates the step response with different initial states.
    """
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