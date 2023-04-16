import random

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from utils.config_utils import ConfigObjectFactory
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
from policy.qmix import QMix
from policy.centralized_ppo import CentralizedPPO
from policy.independent_ppo import IndependentPPO
from policy.decentralized_ppo import DecentralizedPPO


class Baseline_Agents:
    def __init__(self, env_info: dict):
        self.env_info = env_info
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.n_agents = self.env_info["n_agents"]

        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # 下面三个算法作为baseline
        if self.env_config.learn_policy == "qmix":
            self.n_actions = self.env_info["n_actions"]
            self.policy = QMix(self.env_info)

        elif self.env_config.learn_policy == "centralized_ppo":
            self.action_space = self.env_info["action_space"]
            self.policy = CentralizedPPO(self.env_info)

        elif self.env_config.learn_policy == "independent_ppo":
            self.action_space = self.env_info["action_space"]
            self.policy = IndependentPPO(self.env_info)

        else:
            raise ValueError(
                "learn_policy error, just support, " "qmix, centralized_ppo"
            )
        self.discrete_action = isinstance(self.action_space, Discrete)
        self.continuous_action = isinstance(self.action_space, Box)
        type_action_space = type(self.env_info["action_space"])
        assert (
            self.discrete_action or self.continuous_action
        ), f"Action space error. The type is {type_action_space}. We except Discrete or Box."

        if self.discrete_action:
            self.action_space_low = 0
            self.action_space_high = self.env_info["action_space"].n - 1
        elif self.continuous_action:
            self.action_space_low = self.env_info["action_space"].low
            self.action_space_high = self.env_info["action_space"].high
        else:
            raise ValueError(
                f"Action space error. The type is {type_action_space}. We except Discrete or Box."
            )

    def learn(self, batch_data: dict, episode_num: int = 0):
        self.policy.learn(batch_data, episode_num)

    def choose_actions(self, obs: dict) -> tuple:
        actions_with_name = {}
        actions = []
        log_probs = []
        obs = torch.stack([torch.Tensor(value) for value in obs.values()], dim=0)
        self.policy.init_hidden(1)
        if isinstance(self.policy, QMix):
            actions_ind = [i for i in range(self.n_actions)]
            for i, agent in enumerate(self.env_info["agents_name"]):
                inputs = list()
                inputs.append(obs[i, :])
                inputs.append(torch.zeros(self.n_actions))
                agent_id = torch.zeros(self.n_agents)
                agent_id[i] = 1
                inputs.append(agent_id)
                inputs = torch.cat(inputs).unsqueeze(dim=0).to(self.device)
                with torch.no_grad():
                    hidden_state = self.policy.eval_hidden[:, i, :]
                    q_value, _ = self.policy.rnn_eval(inputs, hidden_state)
                if random.uniform(0, 1) > self.train_config.epsilon:
                    action = random.sample(actions_ind, 1)[0]
                else:
                    action = int(torch.argmax(q_value.squeeze()))
                actions_with_name[agent] = action
                actions.append(action)
        elif isinstance(self.policy, CentralizedPPO):
            obs = obs.reshape(1, -1).to(self.device)
            with torch.no_grad():
                action_means, _ = self.policy.ppo_actor(obs, self.policy.rnn_hidden)
            for i, agent_name in enumerate(self.env_info["agents_name"]):
                action_mean = action_means[:, i].squeeze()
                dist = MultivariateNormal(action_mean, self.policy.get_cov_mat())
                action = np.clip(
                    dist.sample().cpu().numpy(),
                    self.action_space_low,
                    self.action_space_high,
                ).astype(dtype=np.float32)
                log_probs.append(dist.log_prob(torch.Tensor(action).to(self.device)))
                actions_with_name[agent_name] = action
                actions.append(action)
        elif isinstance(self.policy, IndependentPPO):
            obs = obs.to(self.device)
            for i, agent_name in enumerate(self.env_info["agents_name"]):
                with torch.no_grad():
                    action_mean, _ = self.policy.ppo_actor(
                        obs[i].unsqueeze(dim=0), self.policy.rnn_hidden[i]
                    )
                action_mean = action_mean.squeeze()
                dist = MultivariateNormal(action_mean, self.policy.get_cov_mat())
                if self.discrete_action:
                    # action = np.round(action).astype(int)
                    float_action = dist.sample()
                    log_probs.append(
                        dist.log_prob(torch.Tensor(float_action).to(self.device))
                    )
                    _, action = float_action.max(dim=-1)
                    action = action.item()
                elif self.continuous_action:
                    action = np.clip(
                        dist.sample().cpu().numpy(),
                        self.action_space_low,
                        self.action_space_high,
                    ).astype(dtype=np.float32)
                    log_probs.append(
                        dist.log_prob(torch.Tensor(action).to(self.device))
                    )
                else:
                    raise ValueError("Action type error.")

                actions_with_name[agent_name] = action
                actions.append(action)
        return actions_with_name, actions, log_probs

    def save_model(self):
        self.policy.save_model()

    def load_model(self):
        self.policy.load_model()

    def del_model(self):
        self.policy.del_model()

    def is_saved_model(self) -> bool:
        return self.policy.is_saved_model()

    def get_results_path(self):
        return self.policy.result_path
