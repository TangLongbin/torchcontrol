# torchcontrol

## Introduction

torchcontrol is a parallel control system simulation and control library based on PyTorch. It supports batch simulation, classical and modern control, nonlinear systems, GPU acceleration, and visualization. The library is modular and extensible, suitable for both research and teaching.

## Features

- **Batch simulation**: Simulate multiple environments in parallel (vectorized, GPU-friendly)
- **Classical & modern control**: Built-in PID controller, state-space, transfer function, and nonlinear system support
- **Custom plants**: Easily define linear or nonlinear plants (systems) with custom dynamics
- **GPU acceleration**: All computations support CUDA (if available)
- **Visualization**: Example scripts for step response, PID control, and nonlinear systems with matplotlib output
- **Extensible**: Modular design for adding new controllers, observers, or plants

## Installation

From the project root, run:

```bash
pip install .
```

Or for development mode (auto-reload on code change):

```bash
pip install -e .
```

## Directory Structure

- torchcontrol/           # Main package (controllers, plants, system, observers)
- examples/               # Example scripts (PID, nonlinear, batch, visualization)
  - results/              # Output results (figures, logs, etc.)
- README.md
- setup.py
- pyproject.toml

## How to Run Examples

After installation, run example scripts from the project root:

```bash
python3 examples/pid_with_internal_plant.py
python3 examples/pid_with_external_plant.py
python3 examples/second_order_plant_step_response.py
python3 examples/nonlinear_plant_step_response.py
```

- All import paths use package-level imports (e.g., `from torchcontrol.controllers import PID`).
- Output files will be automatically saved in `examples/results/`.
- All examples support both CPU and GPU (CUDA) if available.

## Example: Batch PID Control of Second-Order System

```python
from torchcontrol.controllers import PID, PIDCfg
from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg
import torch

# System: G(s) = 1/(s^2 + 2s + 1)
dt = 0.01
num_envs = 16
num = [1.0]
den = [1.0, 2.0, 1.0]
initial_states = torch.rand(num_envs, 1) * 2
plant_cfg = InputOutputSystemCfg(numerator=num, denominator=den, dt=dt, num_envs=num_envs, initial_state=initial_states)
plant = InputOutputSystem(plant_cfg)
pid_cfg = PIDCfg(Kp=120.0, Ki=600.0, Kd=30.0, dt=dt, num_envs=num_envs, state_dim=1, action_dim=1, plant=plant)
pid = PID(pid_cfg)
# ...simulate and visualize...
```

## Dependencies

- Python 3.8+
- torch
- torchdiffeq
- scipy
- numpy
- matplotlib

For GPU support, please ensure you have installed the CUDA version of PyTorch.

## Customization & Extension

- To add new controllers, inherit from `ControllerBase` and register a config.
- To add new plants, inherit from `PlantBase` or use `InputOutputSystem`/`NonlinearSystem`.
- See `examples/` for advanced usage and batch simulation.

---

For batch simulation, RL interface, or custom controllers/systems, please refer to the code structure in the torchcontrol/ directory.
