# adapted from https://pettingzoo.farama.org/content/tutorials/

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.test import parallel_api_test
import gymnasium
from gymnasium.spaces import Discrete

from games import (
    Game,
    Prisoners_Dilemma,
    Samaritans_Dilemma,
    Stag_Hunt,
    Chicken,
)
from dilemma_pettingzoo import raw_env, env, parallel_env
from config import parse_args
import tqdm
from utils.visualizing_utils import save_pt, dict_addition_by_key
from policy.independent_ppo import IndependentPPO
from agents.agents import Baseline_Agents
from common.replay_buffer import CommBatchEpisodeMemory, CommMemory
from utils.config_utils import ConfigObjectFactory
import csv, pickle

args = parse_args()
print("Exp setting: ", args)
if not args.local:
    import wandb
end_step = args.max_cycles
import threading

"""ALGORITHM PARAMS"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["SDL_VIDEODRIVER"] = "dummy"
from collections import defaultdict


def init_env():
    env = parallel_env(render_mode=None, max_cycles=args.max_cycles)
    # parallel_api_test(env, num_cycles=1000)
    # obs = env.reset()
    return env


def get_env_info(env):
    env_info = {
        "n_agents": len(env.possible_agents),
        "action_dim": env.action_space(env.possible_agents[0]).n,
        "obs_space": 2,
        "action_space": env.action_space(env.possible_agents[0]),
        "agents_name": env.possible_agents,
    }
    return env_info


def get_agents(env_info, verbose=False):
    agent = Baseline_Agents(env_info)
    if verbose:
        print("train_config: ", agent.train_config)
        print("env_config: ", agent.env_config)
    return agent


class Runner(object):
    def __init__(self, env):
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.env = env
        self.current_epoch = 0
        self.result_buffer = []
        self.env_info = get_env_info(env)
        self.agents = get_agents(self.env_info, verbose=True)
        if self.env_config.learn_policy in ["centralized_ppo", "independent_ppo"]:
            self.memory = None
            self.batch_episode_memory = CommBatchEpisodeMemory(
                continuous_actions=False,
                n_actions=self.env_info["action_dim"],
                n_agents=self.env_info["n_agents"],
            )
        elif self.env_config.learn_policy in ["qmix"]:
            self.memory = CommMemory()
            self.batch_episode_memory = CommBatchEpisodeMemory(
                continuous_actions=False,
                n_actions=self.env_info["action_dim"],
                n_agents=self.env_info["n_agents"],
            )
        else:
            raise ValueError(
                f"learn_policy should be in [centralized_ppo, independent_ppo, qmix], but it is {self.env_config.learn_policy}."
            )
        self.lock = threading.Lock()
        # init paths
        self.results_path = self.agents.get_results_path()
        self.memory_path = os.path.join(self.results_path, "memory.txt")
        self.result_path = os.path.join(self.results_path, "result.csv")

    def train(self):
        self.obs = self.env.reset()
        self.finish_game = False
        self.cycle = 0
        while not self.finish_game and self.cycle < self.env_config.max_cycles:
            self.state = self.obs2states_array()
            # self.state = self.obs
            (
                self.actions_with_name,
                self.actions,
                self.log_probs,
            ) = self.agents.choose_actions(self.obs)
            (
                self.obs_next,
                self.rewards,
                self.finish_game,
                _,
                self.infos,
            ) = self.env.step(self.actions_with_name)
            self.state_next = self.obs2states_array()
            # self.state_next = self.obs_next
            if "ppo" in self.env_config.learn_policy:
                self.batch_episode_memory.store_one_episode(
                    one_obs=self.obs,
                    one_state=self.state,
                    action=self.actions,
                    reward=self.rewards,
                    log_probs=self.log_probs,
                )
            else:
                self.batch_episode_memory.store_one_episode(
                    one_obs=self.obs,
                    one_state=self.state,
                    action=self.actions,
                    reward=self.rewards,
                    one_obs_next=self.obs_next,
                    one_state_next=self.state_next,
                )
            self.total_reward = dict_addition_by_key(self.total_reward, self.rewards)
            self.obs = self.obs_next
            self.cycle += 1
        self.batch_episode_memory.set_per_episode_len(self.cycle)
        return

    def obs2states_array(self):
        return np.concatenate(tuple(v for v in self.obs.values()), axis=None)

    def run_marl(self):
        self.init_saved_model()
        run_episode = (
            self.train_config.run_episode_before_train
            if "ppo" in self.env_config.learn_policy
            else 1
        )
        for epoch in range(self.current_epoch, self.train_config.epochs + 1):
            # 在正式开始训练之前做一些动作并将信息存进记忆单元中 # before training, do some actions and store the info in memory
            # ppo 属于on policy算法，训练数据要是同策略的 # ppo is on policy algorithm, the training data should be the same policy
            self.total_reward = defaultdict(
                int, {k: 0 for k in self.env_info["agents_name"]}
            )
            if isinstance(self.batch_episode_memory, CommBatchEpisodeMemory):
                for i in range(run_episode):
                    self.train()
            else:
                raise NotImplementedError
            if "ppo" in self.env_config.learn_policy:
                # 可以用一个policy跑一个batch的数据来收集，由于性能问题假设batch=1，后续来优化
                # can use one policy to run a batch of data to collect, because of the performance problem, assume batch=1, and optimize later
                batch_data = self.batch_episode_memory.get_batch_data()
                self.agents.learn(batch_data)
                self.batch_episode_memory.clear_memories()
            else:
                self.memory.store_episode(self.batch_episode_memory)
                self.batch_episode_memory.clear_memories()
                if self.memory.get_memory_real_size() >= 10:
                    for i in range(self.train_config.learn_num):
                        batch = self.memory.sample(self.train_config.memory_batch)
                        self.agents.learn(batch, epoch)
            # avg_reward = self.evaluate()
            avg_reward = self.total_reward / run_episode
            one_result_buffer = (
                [avg_reward] if isinstance(avg_reward, (float, int)) else avg_reward
            )
            self.result_buffer.append(one_result_buffer)
            if epoch % self.train_config.save_epoch == 0 and epoch != 0:
                self.save_model_and_result(epoch)
            print("episode_{} over,avg_reward {}".format(epoch, avg_reward))

    def init_saved_model(self):
        if (
            os.path.exists(self.result_path)
            and (
                os.path.exists(self.memory_path)
                or "ppo" in self.env_config.learn_policy
            )
            and self.agents.is_saved_model()
        ):  # 如果有存储的结果，就加载结果 # 如果有存储的模型，就加载模型
            if "ppo" not in self.env_config.learn_policy:
                with open(self.memory_path, "rb") as f:
                    self.memory = pickle.load(f)
                    self.current_epoch = self.memory.episode + 1
                self.result_buffer.clear()
            else:
                with open(self.result_path, "r") as f:
                    count = 0
                    for _ in csv.reader(f):
                        count += 1
                    self.current_epoch = count
                self.result_buffer.clear()
            self.agents.load_model()
        else:
            self.agents.del_model()
            file_list = os.listdir(self.results_path)
            for file in file_list:
                os.remove(os.path.join(self.results_path, file))

    def save_model_and_result(self, episode: int):
        self.agents.save_model()
        with open(self.result_path, "a", newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerows(self.result_buffer)
            self.result_buffer.clear()
        if "ppo" not in self.env_config.learn_policy:
            with open(self.memory_path, "wb") as f:
                self.memory.episode = episode
                pickle.dump(self.memory, f)

    def evaluate(self):
        pass


def main():
    env = init_env()
    # d_args = vars(args)
    # if not args.local:
    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="mediator",
    #         # track hyperparameters and run metadata
    #         config=d_args,
    #     )
    runner = Runner(env)
    runner.run_marl()
    pass


def run(args):
    pass


#####################

if __name__ == "__main__":
    # # test init_env
    # env = init_env()

    # # test get_env_info
    # env_info = get_env_info(env)
    # print("env_info: ", env_info)

    # # test get_agents
    # agent = get_agents(env_info, verbose=True)

    # test main
    main()
    # """ENV SETUP"""
    # env = parallel_env(render_mode=None, max_cycles=args.max_cycles)
    # # parallel_api_test(env, num_cycles=1000)
    # obs = env.reset()
    # num_agents = len(env.possible_agents)
    # num_actions = env.action_space(env.possible_agents[0]).n
    # # observation_size = env.observation_space(env.possible_agents[0]).shape
    # num_observations = 2

    # """ LEARNER SETUP """
    # # separate policies and optimizers
    # # agent = Agent(num_actions=num_actions).to(device)
    # agents = {agent: Agent(num_actions=num_actions).to(device) for agent in env.agents}
    # optimizers = {
    #     agent: optim.Adam(agents[agent].parameters(), lr=args.lr, eps=args.eps)
    #     for agent in env.agents
    # }

    # """ TRAINING STORAGE """
    # """ ALGO LOGIC: EPISODE STORAGE"""
    # """ TRAINING LOGIC """
    # """ RENDER THE POLICY """
