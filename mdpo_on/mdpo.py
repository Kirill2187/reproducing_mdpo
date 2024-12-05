import warnings
from copy import deepcopy
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

SelfMDPO = TypeVar("SelfMDPO", bound="MDPO")


class MDPO(OnPolicyAlgorithm):
    """
    Mirror Descent Policy Optimization algorithm (MDPO)

    :param policy: The policy model to use (e.g., MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate (float or callable)
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epochs (full passes over the data)
    :param sgd_steps: Number of gradient ascent steps per minibatch (m in the pseudocode)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for GAE
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param kl_coeff: Coefficient for the KL divergence term in the policy loss (1 / t_k in the pseudocode)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    :param rollout_buffer_class: Rollout buffer class to use. If None, it will be automatically selected
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param stats_window_size: Window size for the rollout logging
    :param tensorboard_log: The log location for tensorboard (if None, no logging)
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level (0: no output, 1: info, 2: debug)
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 128,
        n_epochs: int = 10,  # analogous to m in paper
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        kl_coeff: Union[float, Schedule] = 0.1,  # 1 / t_k
        normalize_advantage: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(MDPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        if normalize_advantage:
            assert batch_size > 1, "batch_size must be greater than 1."

        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), "n_steps * n_envs must be greater than 1."
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"Mini-batch size {batch_size} may cause truncated batches."
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.normalize_advantage = normalize_advantage
        self.kl_coeff = get_schedule_fn(kl_coeff)
        self._current_kl_coeff = self.kl_coeff(1.0)  # Initial KL coefficient

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def get_old_policy(self):
        old_policy = deepcopy(self.policy)
        for param in old_policy.parameters():
            param.requires_grad = False
        return old_policy

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        learning_rate = self.policy.optimizer.param_groups[0]["lr"]

        old_policy = self.get_old_policy()

        # Prepare data
        advantages = self.rollout_buffer.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.rollout_buffer.advantages = advantages

        # For logging
        entropy_losses = []
        pg_losses, value_losses = [], []
        kl_divs = []
        losses = []

        # Update KL coefficient
        self._current_kl_coeff = self.kl_coeff(1.0 - self._current_progress_remaining)

        # Training loop
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                # Evaluate current policy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Evaluate old policy
                with th.no_grad():
                    old_distribution = old_policy.get_distribution(
                        rollout_data.observations
                    )
                    old_log_prob = old_distribution.log_prob(actions)

                # Compute ratio
                ratio = th.exp(log_prob - old_log_prob.detach())

                # Compute KL divergence
                distribution = self.policy.get_distribution(
                    rollout_data.observations
                )
                old_pytorch_dist = old_distribution.distribution
                new_pytorch_dist = distribution.distribution

                kl_div = th.distributions.kl_divergence(
                    old_pytorch_dist, new_pytorch_dist
                )
                kl_div = kl_div.mean()

                # Compute policy loss
                policy_loss = -(
                    (ratio * rollout_data.advantages).mean()
                ) + self._current_kl_coeff * kl_div

                # Value loss
                value_loss = F.mse_loss(rollout_data.returns, values)

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                )

                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )

                # Optimizer step
                self.policy.optimizer.step()

                # Logging
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                kl_divs.append(kl_div.item())
                losses.append(loss.item())

            self._n_updates += 1

        # Record logs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence", np.mean(kl_divs))
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/learning_rate", learning_rate)
        self.logger.record("train/kl_coefficient", self._current_kl_coeff)

    def learn(
        self: SelfMDPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MDPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfMDPO:
        return super(MDPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
