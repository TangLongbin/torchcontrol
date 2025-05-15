# torchcontrol

## Introduction

torchcontrol is a parallel control system simulation and control library based on PyTorch, supporting RL, classical control, and GPU parallelism.

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

- torchcontrol/           # Main package (controllers, plants, system)
- examples/               # Example scripts
  - results/            # Output results (figures, logs, etc.)
- README.md
- setup.py

## How to Run Examples

After installation, run example scripts from the project root:

```bash
python3 examples/test_pid.py
python3 examples/pid_control_second_order.py
python3 examples/pid_control_second_order_cuda.py
```

- All import paths use package-level imports (e.g., from torchcontrol.controllers import PID).
- Output files will be automatically saved in examples/results/.

## Dependencies

- Python 3.8+
- torch
- numpy
- matplotlib

For GPU support, please ensure you have installed the CUDA version of PyTorch.

---
For batch simulation, RL interface, or custom controllers/systems, please refer to the code structure in the torchcontrol/ directory.
