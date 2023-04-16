import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from gymnasium import spaces
from collections import OrderedDict
import ray
import random
from tqdm import tqdm
import numpy as np
from gym.utils.env_checker import check_env
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from configs import configs

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec

from policy import AlwaysSame, RandomPolicy
import utils

from RepeatedPrisonnerDilemma import RepeatedPrisonnerDilemma
from policy import TitForTat, Pavlov

from ray.rllib.algorithms.a2c.a2c import A2CConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from policy import TitForTat

""" TRAINING """


def train(policy_id, env_config, second_agent, num_iter=500):
    if ray.is_initialized():
        ray.shutdown()

    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True)

    results = []

    def select_policy(agent_id, episode, **kwargs):
        if agent_id == 0:
            return policy_id
        else:
            if second_agent == "self_play":
                return policy_id
            else:
                return second_agent

    algo = (
        PPOConfig()
        .resources(num_gpus=1)
        .rollouts(num_rollout_workers=4, num_envs_per_worker=100)
        .framework("torch")
        .environment(env=RepeatedPrisonnerDilemma, env_config=env_config)
        .multi_agent(
            policies={
                "ppo": PolicySpec(config=PPOConfig.overrides(framework_str="torch")),
                "a2c": PolicySpec(
                    config=A2CConfig.overrides(
                        framework_str="torch", entropy_coeff=0.02
                    )
                ),
                "dqn": PolicySpec(config=DQNConfig.overrides(framework_str="torch")),
                "always_cooperate": PolicySpec(
                    policy_class=AlwaysSame, config={"always_action": 0}
                ),
                "always_defect": PolicySpec(
                    policy_class=AlwaysSame, config={"always_action": 1}
                ),
                "tit_for_tat": PolicySpec(policy_class=TitForTat, config={}),
                "pavlov": PolicySpec(policy_class=Pavlov, config={}),
            },
            policy_mapping_fn=select_policy,
            policies_to_train=[policy_id],
        )
        .training(train_batch_size=1000, lr=5e-5, gamma=0.98)
        .build()
    )

    for i in tqdm(range(num_iter)):
        result = algo.train()
        # if i > 0:
        #     print(result["policy_reward_mean"])

        if i % 5 == 0 and i > 0:
            env = RepeatedPrisonnerDilemma(env_config)
            policy_estimate = env.estimate_policy(algo, policy_id, num_samples=2000)
            episode_samples = env.get_episode_samples(
                algo,
                policy_id,
                policy_id if second_agent == "self_play" else second_agent,
            )
            rewards = result["policy_reward_mean"]

            curr_results = {
                "policy_estimate": policy_estimate,
                "episodes": episode_samples,
                "rewards": rewards,
            }
            print(curr_results)

            results.append(curr_results)

    return algo, results


if __name__ == "__main__":
    # config = configs[sys.argv[1]]
    config = configs["low_benefit_nonzero"]
    print(config)

    experiment_name = config["experiment_name"]
    env_config = config["env_config"]
    num_iter = config.get("num_iter", 1000)
    second_agent = config.get(
        "second_agent", "self_play"
    )  # get second agent key from config, default to self_play

    # a2c_algo, a2c_results = train("a2c", env_config, second_agent, num_iter)
    a2c_algo, a2c_results = train("a2c", env_config, "always_cooperate", num_iter)

    results = {
        "a2c": a2c_results,
    }

    import pickle

    with open(f"{experiment_name}.pkl", "wb") as f:
        pickle.dump(results, f)
