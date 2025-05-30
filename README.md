# torchcontrol

[![PyPI](https://img.shields.io/pypi/v/torchcontrol?color=blue)](https://pypi.org/project/torchcontrol/) ![CI](https://github.com/TangLongbin/torchcontrol/actions/workflows/python-publish.yml/badge.svg)

**torchcontrol** is a modern, parallel control system simulation and control library built on PyTorch. It supports batch simulation, classical and modern control, nonlinear systems, GPU acceleration, and rich visualization. Designed for both research and teaching, torchcontrol is modular, extensible, and easy to use for rapid prototyping and large-scale experiments.

---

## 🚀 Features

- **Batch simulation**: Simulate many environments in parallel (vectorized, GPU-friendly)
- **Classical & modern control**: Built-in PID, state-space, transfer function, and nonlinear system support
- **Custom plants**: Easily define linear or nonlinear plants (systems) with custom dynamics
- **GPU acceleration**: All computations support CUDA (if available)
- **Visualization**: Example scripts for step response, PID control, nonlinear and UAV systems with matplotlib and animation output
- **Extensible**: Modular design for adding new controllers, observers, or plants
- **Research-ready**: RL-style interfaces, batched rollouts, and reproducible results

---

## 📦 Installation

Install the latest release from PyPI:

```bash
pip install torchcontrol
```

Or, for the latest development version from source (from the project root):

```bash
pip install .
```

Or for development mode (auto-reload on code change):

```bash
pip install -e .
```

---

## 🗂️ Directory Structure

- `torchcontrol/`           — Main package (controllers, plants, system, observers, utils)
- `examples/`               — Example scripts (PID, nonlinear, batch, UAV, visualization)
  - `results/`              — Output results (figures, GIFs, logs, etc.)
- `assets/`                 — Project images and GIFs for documentation
- `tests/`                  — Unit tests (pytest)
- `README.md`, `setup.py`, `pyproject.toml`, `LICENSE`

---

## 🖼️ Visual Examples

### MPPI UAV Trajectory Tracking (Batch, 3D, Animated)

![UAV MPPI Tracking](assets/uav_mppi_tracking.gif)

### Batch PID Control (Internal Plant)

![PID with Internal Plant](assets/pid_with_internal_plant.gif)

### Batch Step Response (Second-Order System)

![Second Order Step Response](assets/second_order_plant_step_response.gif)

### Nonlinear System Step Response (Batch)

![Nonlinear Plant Step Response](assets/nonlinear_plant_step_response.gif)

---

## 📖 Quick Start

### Batch PID Control of a Second-Order System

```python
from torchcontrol.controllers import PID, PIDCfg
from torchcontrol.plants import InputOutputSystem, InputOutputSystemCfg
import torch

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

### Nonlinear System Example

```python
from torchcontrol.system import Parameters
from torchcontrol.plants import NonlinearSystem, NonlinearSystemCfg
import torch

def nonlinear_oscillator(x, u, t, params):
    k, c, alpha = params.k, params.c, params.alpha
    x1, x2 = x[:, 0], x[:, 1]
    dx1 = x2
    dx2 = -k * x1 - c * x2 + alpha * x1 ** 3 + u.squeeze(-1)
    return torch.stack([dx1, dx2], dim=1)

params = Parameters(k=1.0, c=0.7, alpha=0.1)
initial_states = torch.rand(16, 2)
cfg = NonlinearSystemCfg(dynamics=nonlinear_oscillator, output=None, dt=0.01, num_envs=16, state_dim=2, action_dim=1, initial_state=initial_states, params=params)
plant = NonlinearSystem(cfg)
# ...simulate and visualize...
```

---

## 🏗️ Architecture Overview

- **SystemBase**: Abstract base for all systems (plants, controllers, observers)
- **PlantBase**: Base for all plant (system) models (linear, nonlinear, batch)
- **ControllerBase**: Base for all controllers (PID, MPPI, custom)
- **Config Classes**: All systems/controllers/plants use dataclass configs for reproducibility
- **Batching**: All classes support `num_envs` for parallel simulation
- **Device**: All tensors and computation can run on CPU or CUDA

---

## 📚 Example Scripts

Run from the project root:

```bash
python3 examples/pid_with_internal_plant.py
python3 examples/pid_with_external_plant.py
python3 examples/second_order_plant_step_response.py
python3 examples/nonlinear_plant_step_response.py
python3 examples/uav_geometric_hover.py
python3 examples/uav_geometric_tracking.py
python3 examples/uav_mppi_tracking.py
python3 examples/uav_thrust_descent_to_hover.py
```

- All import paths use package-level imports (e.g., `from torchcontrol.controllers import PID`).
- Output files are saved in `examples/results/`.
- All examples support both CPU and GPU (CUDA) if available.

---

## 🧪 Testing

Run all tests with:

```bash
pytest tests/
```

---

## 🛠️ Customization & Extension

- Add new controllers by inheriting from `ControllerBase` and registering a config
- Add new plants by inheriting from `PlantBase` or using `InputOutputSystem`/`NonlinearSystem`
- See `examples/` for advanced usage, batch simulation, and RL-style rollouts

---

## 🤝 Contributing

Pull requests, issues, and suggestions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) if available, or open an issue to discuss your ideas.

---

## 📄 License

MIT License © 2025 Tang Longbin
