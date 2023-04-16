from collections import OrderedDict
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

from utils import (
    action_idx_mapping,
    state_idx_mapping,
)


class RepeatedPrisonnerDilemma(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()

        self.add_messaging_step = env_config.get("add_messaging_step", False)
        self.include_timestep_in_state = env_config.get(
            "include_timestep_in_state", False
        )
        self.reward_matrix = env_config.get(
            "reward_matrix",
            [
                [4, 4],
                [0, 5],
                [5, 0],
                [2, 2],
            ],
        )

        self.num_rounds_cfg = env_config.get("num_rounds", 250)

        self.max_rounds = env_config.get("max_rounds", self.num_rounds_cfg)

        self._agent_ids = [0, 1]

        # Init state
        self.reset()

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    def _build_action_space(self):
        action_space = {
            "defect": spaces.Discrete(2),
        }

        if self.add_messaging_step:
            action_space["message"] = spaces.Discrete(2)

        return spaces.Dict(action_space)

    def action_space_sample(self):
        return {agent_id: self.action_space.sample() for agent_id in self._agent_ids}

    def _build_observation_space(self):
        space_dict = {
            "last_outcome": spaces.Discrete(4),
        }

        if self.add_messaging_step:
            space_dict["message"] = spaces.Discrete(2)
            space_dict["stage"]: spaces.Discrete(2)

        if self.include_timestep_in_state:
            space_dict["timestep"] = spaces.Discrete(self.max_rounds + 1)

        return spaces.Dict(space_dict)

    def _get_obs(self):
        obs = {}

        for agent_id in self._agent_ids:
            outcome = self.state["last_outcome"]
            if agent_id == 1:
                if self.state["last_outcome"] == 1:
                    outcome = 2
                if self.state["last_outcome"] == 2:
                    outcome = 1

            obs[agent_id] = [
                ("last_outcome", outcome),
            ]

            if self.add_messaging_step:
                obs.append(("message", self.state["message"][1 - agent_id]))
                obs.append(("stage", 0 if self.state["stage"] == "communicate" else 1))
            if self.include_timestep_in_state:
                obs[agent_id].append(("timestep", self.state["timestep"]))

            obs[agent_id] = OrderedDict(obs[agent_id])

        return obs

    def _get_individual_rew(self, agent_id):
        return self.reward_matrix[self.state["last_outcome"]][agent_id]

    def _get_rew(self, action_dict):
        return {
            agent_id: self._get_individual_rew(agent_id)
            for agent_id, action in action_dict.items()
        }

    def _get_info(self, action_dict):
        return {}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if isinstance(self.num_rounds_cfg, int):
            self.num_rounds = self.num_rounds_cfg
        else:
            self.num_rounds = self.num_rounds_cfg()

        self.state = {}
        self.state["timestep"] = 0

        self.state["last_outcome"] = np.random.choice(4)

        if self.add_messaging_step:
            self.state["stage"] = "communicate"
            self.state["message"] = {agent_id: 0 for agent_id in self._agent_ids}

        self.history = [self.state["last_outcome"]]

        return self._get_obs(), {}  # Return obs, info

    def actions_to_state(self, action_dict):
        action_1, action_2 = (
            action_dict[self._agent_ids[0]]["defect"],
            action_dict[self._agent_ids[1]]["defect"],
        )

        if action_1 == 0 and action_2 == 0:
            return 0
        elif action_1 == 0 and action_2 == 1:
            return 1
        elif action_1 == 1 and action_2 == 0:
            return 2
        elif action_1 == 1 and action_2 == 1:
            return 3

    def communicate_step(self, action_dict):
        self.state["timestep"] += 1
        self.state["stage"] = "act"

        for agent_id, action in action_dict.items():
            self.state["message"][agent_id] = action["message"]

        terminateds = {"__all__": False}
        truncateds = {"__all__": False}

        rew = {agent_id: 0 for agent_id in self._agent_ids}
        obs = self._get_obs()
        infos = self._get_info(action_dict)

        return obs, rew, terminateds, truncateds, infos

    def outcome_step(self, action_dict):
        outcome = self.actions_to_state(action_dict)
        self.state["last_outcome"] = outcome

        if self.add_messaging_step:
            self.state["stage"] = "communicate"
        else:
            self.state["timestep"] += 1

        self.history.append(outcome)

        rew = self._get_rew(action_dict)
        obs = self._get_obs()
        infos = self._get_info(action_dict)

        terminated = self.state["timestep"] == self.num_rounds
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": terminated}

        return obs, rew, terminateds, truncateds, infos

    def step(self, action_dict):
        if self.add_messaging_step and self.state["stage"] == "communicate":
            return self.communicate_step(action_dict)
        else:
            return self.outcome_step(action_dict)

    def estimate_policy(self, algo, policy_id, num_samples=10000):
        """
        估计策略。根据给定的算法和策略ID，生成策略样本并计算各种状态和时间步下的背叛百分比。
        Estimate the policy. Based on the given algorithm and policy ID, generate policy samples and calculate the percentage of defection for various states and timesteps.
        """
        policy_samples = {}
        for state, state_id in state_idx_mapping.items():
            for j in range(self.max_rounds):
                for i in range(
                    num_samples // (len(state_idx_mapping) * self.max_rounds)
                ):
                    obs = self.observation_space.sample()
                    obs["last_outcome"] = state_id

                    if self.include_timestep_in_state:
                        obs["timestep"] = j

                    if self.add_messaging_step:
                        message = np.random.choice(2)
                        obs["message"] = message

                    action = algo.compute_single_action(obs, policy_id=policy_id)

                    past_state_name = state
                    if self.include_timestep_in_state:
                        past_state_name += str(j)

                    if self.add_messaging_step:
                        past_state_name = state + str(message)

                    samples = policy_samples.get(
                        past_state_name, {"defect_count": 0, "total": 0}
                    )
                    samples["defect_count"] += action["defect"]
                    samples["total"] += 1

                    policy_samples[past_state_name] = samples

        policy_samples = pd.DataFrame(policy_samples).T
        policy_samples["pct_D"] = (
            policy_samples["defect_count"] / policy_samples["total"]
        ) * 100
        return policy_samples

    def get_episode_samples(self, algo, first_policy, second_policy):
        """
        获取单集样本。使用给定的算法和两个策略ID，在多轮游戏中模拟代理之间的互动。
        Get episode samples. Using the given algorithm and two policy IDs, simulate the interaction between agents in multiple rounds of the game.
        """
        episode_samples = []
        for i in range(250 // self.max_rounds):
            terminated = {"__all__": False}
            obs, info = self.reset()

            while not terminated["__all__"]:
                actions = {
                    0: algo.compute_single_action(obs[0], policy_id=first_policy),
                    1: algo.compute_single_action(obs[1], policy_id=second_policy),
                }
                obs, reward, terminated, truncated, info = self.step(actions)

            episode_samples.append(self.history)

        return episode_samples
