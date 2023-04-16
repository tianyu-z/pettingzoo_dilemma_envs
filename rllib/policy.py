from collections import OrderedDict
import gymnasium as gym
import numpy as np
import random
from ray.rllib.policy.policy import Policy
from utils import get_action_idx, get_idx_action, get_state_idx, get_idx_state


class TitForTat(Policy):
    """
    This policy always defects on the first round, and then copies the opponent's
    previous action on all subsequent rounds.
    """

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        config=None,
    ):
        if not config:
            config = {}
        config["framework"] = "torch"

        Policy.__init__(self, observation_space, action_space, config)

    def action_to_state(self, obs):
        state = np.argmax(obs)

        if state == get_state_idx("CC"):
            return get_action_idx("C")
        elif state == get_state_idx("CD"):
            return get_action_idx("D")
        elif state == get_state_idx("DC"):
            return get_action_idx("C")
        elif state == get_state_idx("DD"):
            return get_action_idx("D")
        else:
            raise ValueError("Invalid state")

    def create_action(self, obs):
        action = self.action_space.sample()
        action["defect"] = self.action_to_state(obs)
        return action

    def compute_single_action(
        self,
        obs=None,
        state=None,
        *,
        prev_action=None,
        prev_reward=None,
        info=None,
        input_dict=None,
        episode=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        return (
            self.create_action(obs),
            [],
            {},
        )
        return self.action_to_state(obs)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        return (
            [self.create_action(obs) for obs in obs_batch],
            [],
            {},
        )

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


class Pavlov(Policy):
    """
    win-stay, lose-shift
    """

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        config=None,
    ):
        if not config:
            config = {}
        config["framework"] = "torch"

        Policy.__init__(self, observation_space, action_space, config)

    def action_to_state(self, obs):
        state = np.argmax(obs)
        # the first is the opponent's action, the second is the agent's action
        if state == get_state_idx("CC"):
            return get_action_idx("C")
        elif state == get_state_idx("CD"):
            return get_action_idx("D")
        elif state == get_state_idx("DC"):
            return get_action_idx("D")
        elif state == get_state_idx("DD"):
            return get_action_idx("C")
        else:
            raise ValueError("Invalid state")

    def create_action(self, obs):
        action = self.action_space.sample()
        action["defect"] = self.action_to_state(obs)
        return action

    def compute_single_action(
        self,
        obs=None,
        state=None,
        *,
        prev_action=None,
        prev_reward=None,
        info=None,
        input_dict=None,
        episode=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        return (
            self.create_action(obs),
            [],
            {},
        )
        return self.action_to_state(obs)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        return (
            [self.create_action(obs) for obs in obs_batch],
            [],
            {},
        )

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


class AlwaysSame(Policy):
    """This policy always do a specific action in config."""

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        config=None,
    ):
        if not config:
            config = {}
        config["framework"] = "torch"
        self.always_action = config["always_action"]
        Policy.__init__(self, observation_space, action_space, config)

    def action_to_state(self, obs):
        state = np.argmax(obs)

        if state == get_state_idx("CC"):
            return self.always_action
        elif state == get_state_idx("CD"):
            return self.always_action
        elif state == get_state_idx("DC"):
            return self.always_action
        elif state == get_state_idx("DD"):
            return self.always_action
        else:
            raise ValueError("Invalid state")

    def create_action(self, obs):
        action = self.action_space.sample()
        action["defect"] = self.action_to_state(obs)
        return action

    def compute_single_action(
        self,
        obs=None,
        state=None,
        *,
        prev_action=None,
        prev_reward=None,
        info=None,
        input_dict=None,
        episode=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        return (
            self.create_action(obs),
            [],
            {},
        )
        return self.action_to_state(obs)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        return (
            [self.create_action(obs) for obs in obs_batch],
            [],
            {},
        )

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


class RandomPolicy(Policy):
    def __init__(
        self,
        observation_space=None,
        action_space=None,
        config=None,
    ):
        if not config:
            config = {}
        config["framework"] = "torch"
        self.always_action = config["always_action"]
        Policy.__init__(self, observation_space, action_space, config)

    def action_to_state(self, obs):
        return np.random.randint(0, 2)

    def create_action(self, obs):
        action = self.action_space.sample()
        action["defect"] = self.action_to_state(obs)
        return action

    def compute_single_action(
        self,
        obs=None,
        state=None,
        *,
        prev_action=None,
        prev_reward=None,
        info=None,
        input_dict=None,
        episode=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        return (
            self.create_action(obs),
            [],
            {},
        )
        return self.action_to_state(obs)

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        return (
            [self.create_action(obs) for obs in obs_batch],
            [],
            {},
        )

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
