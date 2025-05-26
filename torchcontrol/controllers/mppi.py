"""
mppi.py
Model Predictive Path Integral (MPPI) controller implementation.
"""
from __future__ import annotations

import copy
import torch
from typing import TYPE_CHECKING
from .controller_base import ControllerBase

if TYPE_CHECKING:
    from .mppi_cfg import MPPICfg

class MPPI(ControllerBase):
    """
    Model Predictive Path Integral (MPPI) controller.

    This controller uses a sampling-based approach to find an optimal control sequence
    by minimizing a cost function over a finite prediction horizon.
    It iteratively samples control noise, simulates rollouts, computes costs,
    and updates the nominal control sequence based on weighted averages of the sampled controls.

    The controller is designed to work with batched environments, meaning it can
    process multiple states and compute actions for multiple environments in parallel.
    """

    cfg: MPPICfg # Type hint for the config for this specific controller

    def __init__(self, cfg: MPPICfg):
        """
        Initializes the MPPI controller.

        Args:
            cfg (MPPICfg): Configuration object for MPPI. This should include:
                - K (int): Number of rollouts.
                - T (int): Prediction horizon.
                - sigma (Union[float, torch.Tensor]): Control noise standard deviation.
                - alpha (float): Temperature for trajectory weighting.
                - u_min (Union[float, torch.Tensor]): Minimum control limits.
                - u_max (Union[float, torch.Tensor]): Maximum control limits.
                - cost_function (Callable): Function to compute rollout costs.
                - num_envs (int): Number of parallel environments (from SystemBaseCfg).
                - action_dim (int): Dimensionality of the action space (from SystemBaseCfg).
                - device (torch.device): Device for computations (from SystemBaseCfg).
                - plant (PlantBase): The plant model providing dynamics (from ControllerCfg).
        """
        super().__init__(cfg)
        self.T = cfg.T # int
        self.K = cfg.K # int
        self.alpha = cfg.alpha # float
        self.sigma = cfg.sigma.to(self.device)
        self.u_min = cfg.u_min.to(self.device)
        self.u_max = cfg.u_max.to(self.device)
        self.cost_function = cfg.cost_function # Callable for computing costs
        
        # Initialize nominal control sequence (mean), shape: (num_envs, T, action_dim)
        self.u_nominal = torch.zeros(self.num_envs, self.T, self.action_dim, device=self.device)

        # If sigma is a scalar, convert to tensor of shape (action_dim,)
        if self.sigma.dim() == 0:
            self.sigma = torch.full((self.action_dim,), float(self.sigma), device=self.device)
        
        # If u_min and u_max are scalars, convert to tensors of shape (action_dim,)
        if self.u_min.dim() == 0:
            self.u_min = torch.full((self.action_dim,), float(self.u_min), device=self.device)
        if self.u_max.dim() == 0:
            self.u_max = torch.full((self.action_dim,), float(self.u_max), device=self.device)

        # For rollout simulation, create a new plant with batch_size = num_envs * K using plant's cfg
        rollout_plant_cfg = copy.deepcopy(self.plant.cfg) # Use deepcopy to avoid modifying the original cfg
        rollout_plant_cfg.num_envs = self.num_envs * self.K
        rollout_plant_cfg.initial_state = self.plant.initial_state.repeat_interleave(self.K, dim=0) # Repeat each env's initial_state for K rollouts
        # extend the parameters batch size if they are tensors
        if hasattr(rollout_plant_cfg, "params") and rollout_plant_cfg.params is not None:
            for k, v in rollout_plant_cfg.params.__dict__.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == self.num_envs:
                    setattr(rollout_plant_cfg.params, k, v.repeat_interleave(self.K, dim=0))
        self._rollout_plant = rollout_plant_cfg.class_type(rollout_plant_cfg)  # Create a new plant instance for rollouts

    # Note: The `forward` method signature in `ControllerBase` is `forward(self, x, r, t=None)`.
    # MPPI typically only needs the current state `x` (current_state).
    # The reference `r` and time `t` might not be directly used in the core MPPI logic,
    # but the signature should match the base class.
    # We can choose to ignore `r` and `t` if not needed by MPPI's logic.
    # For this implementation, `current_state` corresponds to `x`.
    def forward(self, current_state: torch.Tensor, reference: torch.Tensor | None = None, t: torch.Tensor | None = None) -> torch.Tensor:
        """
        Computes the optimal action for the current state.

        Args:
            current_state (torch.Tensor): The current state of the system for each environment.
                                        Corresponds to `x` in `ControllerBase`.
                                        Shape: (num_envs, state_dim)
            reference (torch.Tensor | None): Reference signal or trajectory for tracking tasks.
                                             Corresponds to `r` in `ControllerBase`.
                                             Shape: (num_envs, T, state_dim) or None.
            t (torch.Tensor | None): Time signal. May not be used by MPPI directly.
                                     Corresponds to `t` in `ControllerBase`.

        Returns:
            torch.Tensor: The optimal action for the current state for each environment.
                          Shape: (num_envs, action_dim)
        """
        assert current_state.shape == (self.num_envs, self.state_dim), \
            f"Expected current_state shape to be ({self.num_envs}, {self.state_dim}), got {current_state.shape}"
        current_state = current_state.to(self.device) # Ensure current_state is on the correct device

        # 1. Sample control noise
        noise = self._sample_control_noise() # Shape: (num_envs, K, T, action_dim)

        # 2. Create perturbed control sequences
        u_perturbed = self.u_nominal.unsqueeze(1) + noise # Shape: (num_envs, K, T, action_dim)

        # Apply control limits
        u_perturbed = torch.clamp(u_perturbed,
                                  self.u_min.view(1, 1, 1, -1),
                                  self.u_max.view(1, 1, 1, -1))

        # 3. Simulate rollouts and 4. Compute costs
        total_costs = self._compute_rollout_cost(current_state, u_perturbed, reference) # Shape: (num_envs, K)

        # 5. Compute weights
        weights = self._compute_weights(total_costs) # Shape: (num_envs, K)

        # 6. Update nominal control sequence
        weights = weights.unsqueeze(-1).unsqueeze(-1) # Shape: (num_envs, K, 1, 1)
        self.u_nominal = torch.sum(weights * u_perturbed, dim=1) # Shape: (num_envs, T, action_dim)

        # 7. Get the first action from the updated nominal sequence
        action = self.u_nominal[:, 0, :].clone() # Shape: (num_envs, action_dim)

        # 8. Receding horizon: shift nominal control sequence for next time step
        self.u_nominal[:, :-1, :] = self.u_nominal[:, 1:, :].clone()
        self.u_nominal[:, -1, :] = 0 # Re-initialize the last action

        return action

    def update(self, *args, **kwargs):
        """
        Update the MPPI controller parameters online.

        Args:
            *args: new parameters (not used)
            **kwargs: new parameters (alpha, sigma, u_min, u_max)
        """
        super().update(*args, **kwargs)  # Call parent class update method
        for key in ['alpha', 'sigma', 'u_min', 'u_max']:
            if key in kwargs:
                val = kwargs[key]
                # For tensors, check shape if possible
                if key in ['sigma', 'u_min', 'u_max']:
                    val = torch.as_tensor(val, dtype=torch.float32, device=self.device)
                    if val.dim() == 0:
                        val = torch.full((self.action_dim,), float(val), device=self.device)
                    assert val.shape == (self.action_dim,), \
                        f"Shape mismatch for {key}: {getattr(self, key).shape} != {val.shape}"
                    setattr(self, key, val)
                elif key == 'alpha':
                    assert isinstance(val, (float, int)) and val > 0, \
                        f"alpha must be a positive float or int, got {val}"
                    setattr(self, key, float(val))
        # After all updates, check u_min <= u_max
        assert torch.all(self.u_min <= self.u_max), \
            f"u_min ({self.u_min}) must be less than or equal to u_max ({self.u_max}) after update."

    def reset(self, env_ids: list[int] | None = None):
        """
        Resets the internal state of the controller, primarily the nominal control sequence.
        This should be called when an episode ends or the environment is reset.

        Args:
            env_ids (list[int] | None): Optional list of environment IDs to reset.
                                        If None, resets all environments.
        """
        # Resetting the plant is handled by the ControllerBase's reset method if called.
        super().reset(env_ids) # Call base class reset, which handles plant.reset if plant is set.
        
        # This reset focuses on MPPI's internal state (u_nominal).
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._ALL_INDICES # Reset all environments
        
        self.u_nominal[env_ids] = torch.zeros(len(env_ids), self.T, self.action_dim, device=self.device)

    def _sample_control_noise(self) -> torch.Tensor:
        """
        Samples control noise for K rollouts across all environments.

        The noise is sampled from a normal distribution with mean 0 and standard deviation sigma.

        Returns:
            torch.Tensor: Sampled control noise.
                          Shape: (num_envs, K, T, action_dim)
        """
        # Sample noise: (num_envs, K, T, action_dim)
        # Each environment gets K independent noise sequences for its T-step horizon.
        # self.sigma is (action_dim,), needs to be broadcast.
        noise = torch.randn(self.num_envs, self.K, self.T, self.action_dim, device=self.device) * \
                self.sigma.view(1, 1, 1, -1) # Broadcast sigma
        return noise

    def _compute_rollout_cost(self, current_state: torch.Tensor, u_perturbed: torch.Tensor, reference: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the total cost for each rollout trajectory.

        This involves simulating the system dynamics for T steps using the perturbed control sequences
        and then evaluating the cost of the resulting state-action trajectories.
        The cost function is expected to take inputs of shape (num_envs, K, T, state_dim) and (num_envs, K, T, action_dim)
        and return costs of shape (num_envs, K).
        Optionally, a reference trajectory can be provided for tracking tasks.

        Args:
            current_state (torch.Tensor): The current state of the system for each environment.
                                        Shape: (num_envs, state_dim)
            u_perturbed (torch.Tensor): Perturbed control sequences for all K rollouts and all environments.
                                        Shape: (num_envs, K, T, action_dim)
            reference (torch.Tensor, optional): Reference trajectory for tracking tasks.
                Shape: (num_envs, T, state_dim) or None.

        Returns:
            torch.Tensor: Total cost for each rollout.
                          Shape: (num_envs, K)
        """
        K = self.K
        T = self.T
        num_envs = self.num_envs
        action_dim = self.action_dim
        state_dim = self.state_dim

        # Flatten num_envs and K dimensions for efficient batch simulation with the plant
        # u_perturbed shape: (num_envs, K, T, action_dim) -> u_for_sim shape: (num_envs * K, T, action_dim)
        u_for_sim = u_perturbed.reshape(num_envs * K, T, action_dim)

        # Initialize states for simulation
        # current_state shape: (num_envs, state_dim)
        # s_t shape: (num_envs * K, state_dim)
        s_t = current_state.unsqueeze(1).repeat(1, K, 1).reshape(num_envs * K, state_dim)

        # Use a plant with correct batch size for rollouts
        plant = self._rollout_plant

        # Store simulated state trajectories
        # state_rollouts_sim shape: (num_envs * K, T, state_dim)
        state_rollouts_sim = torch.zeros(num_envs * K, T, state_dim, device=self.device)

        # Simulate dynamics for T steps
        for t_step in range(T):
            state_rollouts_sim[:, t_step, :] = s_t
            a_t = u_for_sim[:, t_step, :]  # Shape: (num_envs * K, action_dim)
            plant.state = s_t  # Update plant state for the current step
            s_t = plant.step(a_t)  # Step the plant forward, returns new state, shape: (num_envs * K, state_dim)
        # Reshape simulated states to (num_envs, K, T, state_dim) for the cost function
        state_rollouts_for_cost_fn = state_rollouts_sim.reshape(num_envs, K, T, state_dim)

        # u_perturbed is already (num_envs, K, T, action_dim), which is the expected input for the cost function.

        # Call the cost function (defined in mppi_cfg.py)
        # Expected input shapes:
        #   state_trajs: (num_envs, K, T, state_dim)
        #   action_trajs: (num_envs, K, T, action_dim)
        #   reference: (num_envs, T, state_dim) or None
        # Expected output shape: (num_envs, K)
        total_costs = self.cost_function(state_rollouts_for_cost_fn, u_perturbed, reference)

        assert total_costs.shape == (num_envs, K), \
            f"Expected total_costs shape to be ({num_envs}, {K}), got {total_costs.shape}"
        return total_costs

    def _compute_weights(self, total_costs: torch.Tensor) -> torch.Tensor:
        """
        Computes the weights for each rollout based on their costs.

        Uses the softmax function with a temperature parameter alpha.
        Lower costs result in higher weights.

        Args:
            total_costs (torch.Tensor): Total cost for each rollout.
                                      Shape: (num_envs, K)

        Returns:
            torch.Tensor: Normalized weights for each rollout.
                          Shape: (num_envs, K)
        """
        # total_costs shape: (num_envs, K)
        min_costs = torch.min(total_costs, dim=1, keepdim=True).values # Shape: (num_envs, 1)
        exp_terms = torch.exp(-1.0 / self.alpha * (total_costs - min_costs))
        sum_exp = torch.sum(exp_terms, dim=1, keepdim=True) # Shape: (num_envs, 1)
        weights = exp_terms / (sum_exp + 1e-9) # Add epsilon for numerical stability
        return weights
