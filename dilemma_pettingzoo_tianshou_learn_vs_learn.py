import os
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

import dilemma_pettingzoo


def _get_agents(
    agent_learn1: Optional[BasePolicy] = None,
    agent_learn2: Optional[BasePolicy] = None,
    optim_1: Optional[torch.optim.Optimizer] = None,
    optim_2: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    if agent_learn1 is None:
        # model
        state_shape = (
            observation_space["observation"].shape or observation_space["observation"].n
        )
        action_shape = env.action_space.shape or env.action_space.n
        print("state_shape", state_shape)
        print("action_shape", action_shape)
        net_1 = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=[128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim_1 is None:
            optim_1 = torch.optim.Adam(net_1.parameters(), lr=1e-4)
        agent_learn1 = DQNPolicy(
            model=net_1,
            optim=optim_1,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )

    if agent_learn2 is None:
        state_shape = (
            observation_space["observation"].shape or observation_space["observation"].n
        )
        action_shape = env.action_space.shape or env.action_space.n
        print("state_shape", state_shape)
        print("action_shape", action_shape)
        net_2 = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=[128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim_2 is None:
            optim_2 = torch.optim.Adam(net_2.parameters(), lr=1e-4)
        agent_learn2 = DQNPolicy(
            model=net_2,
            optim=optim_2,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )

    agents = [agent_learn1, agent_learn2]
    optims = [optim_1, optim_2]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optims, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(dilemma_pettingzoo.env(max_cycles=10000))


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )

    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "rps", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "rps", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    # def stop_fn(mean_rewards):
    #     return mean_rewards >= 0.6
    stop_fn = None

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=10,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
