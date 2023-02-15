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
from agents.agent import Agent
from agents.utils import batchify_obs, batchify, unbatchify
from config import parse_args

"""ALGORITHM PARAMS"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["SDL_VIDEODRIVER"] = "dummy"

#####################

if __name__ == "__main__":

    args = parse_args()
    print("Exp setting: ", args)
    if not args.local:
        import wandb
    end_step = args.max_cycles

    """ENV SETUP"""
    env = parallel_env(render_mode=None, max_cycles=args.max_cycles)
    # parallel_api_test(env, num_cycles=1000)
    obs = env.reset()
    d_args = vars(args)
    if not args.local:
        wandb.init(
            # set the wandb project where this run will be logged
            project="mediator",
            # track hyperparameters and run metadata
            config=d_args,
        )
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    # observation_size = env.observation_space(env.possible_agents[0]).shape
    num_observations = 2

    """ LEARNER SETUP """
    # separate policies and optimizers
    # agent = Agent(num_actions=num_actions).to(device)
    agents = {
        agent: Agent(num_actions=num_actions).to(device) for agent in env.agents
    }
    optimizers = {
        agent: optim.Adam(agents[agent].parameters(), lr=args.lr, eps=args.eps)
        for agent in env.agents
    }

    """ TRAINING STORAGE """
    all_obs = {}
    all_actions = {}
    all_logprobs = {}
    all_rewards = {}
    all_terms = {}
    all_values = {}
    all_returns = {}
    all_advantages = {}
    all_loss = {}
    all_loss_pi = {}
    all_loss_v = {}
    all_loss_ent = {}
    all_loss_info = {}
    all_loss_info["entropy"] = {}
    all_loss_info["kl"] = {}
    all_loss_info["policy_loss"] = {}
    all_loss_info["value_loss"] = {}
    all_loss_info["approxkl"] = {}
    all_loss_info["clipfrac"] = {}

    """ ALGO LOGIC: EPISODE STORAGE"""
    total_episodic_return = 0
    rb_obs = torch.zeros((args.max_cycles, num_agents, num_observations)).to(
        device
    )  # stores stacked observations for each agent
    rb_actions = torch.zeros((args.max_cycles, num_agents)).to(
        device
    )  # stores actions taken by each agent
    rb_logprobs = torch.zeros((args.max_cycles, num_agents)).to(
        device
    )  # stores log probabilities of actions taken by each agent
    rb_rewards = torch.zeros((args.max_cycles, num_agents)).to(
        device
    )  # stores rewards received by each agent
    rb_terms = torch.zeros((args.max_cycles, num_agents)).to(
        device
    )  # stores indicators for terminal states encountered by each agent
    rb_values = torch.zeros((args.max_cycles, num_agents)).to(
        device
    )  # stores values predicted by the value function for each state and each agent

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(args.total_episodes):

        # collect an episode
        with torch.no_grad():

            # collect observations and convert to batch of torch tensors
            next_obs = env.reset()
            # reset the episodic return
            total_episodic_return = np.zeros(num_agents)

            # each episode has num_steps
            for step in range(0, args.max_cycles):

                # rollover the observation
                obs = batchify_obs(next_obs, device)

                policy_outputs = {}
                for idx, agent in enumerate(agents):
                    obs_agent = obs[idx]
                    obs_agent = (
                        obs_agent.float()
                    )  # hotpatch, TODO: fix batchify to return floats
                    agent_actions, agent_logprobs, _, agent_values = agents[
                        agent
                    ].get_action_and_value(obs_agent, action=None)
                    policy_outputs[agent] = {
                        "actions": agent_actions,
                        "logprobs": agent_logprobs,
                        "_": _,
                        "values": agent_values,
                    }

                # join separate tensors from each agent
                actions = torch.cat(
                    [
                        policy_outputs[agent]["actions"].view(1)
                        for agent in agents
                    ]
                )
                logprobs = torch.cat(
                    [
                        policy_outputs[agent]["logprobs"].view(1)
                        for agent in agents
                    ]
                )
                values = torch.cat(
                    [
                        policy_outputs[agent]["values"].view(1)
                        for agent in agents
                    ]
                )
                # execute the environment and log data
                actions_dict = unbatchify(actions, env)

                next_obs, rewards, terms, _, _ = env.step(
                    actions_dict
                )  # TODO : why does this return (2,2) for each agent's observations, instead of the previous actions?
                print("next_obs: ", next_obs)
                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step - 1)):
                delta = (
                    rb_rewards[t]
                    + args.gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = (
                    delta + args.gamma * args.gamma * rb_advantages[t + 1]
                )
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        # b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        # b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        # b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        # b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        # b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        # b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        b_obs = rb_obs[:end_step]
        b_logprobs = rb_logprobs[:end_step]
        b_actions = rb_actions[:end_step]
        b_returns = rb_returns[:end_step]
        b_values = rb_values[:end_step]
        b_advantages = rb_advantages[:end_step]

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            len_b_obs = len(b_obs)
            for start in range(0, len(b_obs), args.batch_size):

                for idx, agent in enumerate(agents):
                    # select the indices we want to train on
                    end = start + args.batch_size
                    batch_index = b_index[start:end]

                    _, newlogprob, entropy, value = agents[
                        agent
                    ].get_action_and_value(
                        b_obs[:, idx, :][batch_index],
                        b_actions[:, idx].long()[batch_index],
                    )
                    logratio = newlogprob - b_logprobs[:, idx][batch_index]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > args.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    # normalize advantaegs
                    advantages = b_advantages[:, idx][batch_index]
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -b_advantages[:, idx][batch_index] * ratio
                    pg_loss2 = -b_advantages[:, idx][batch_index] * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.flatten()
                    v_loss_unclipped = (
                        value - b_returns[:, idx][batch_index]
                    ) ** 2
                    v_clipped = b_values[:, idx][batch_index] + torch.clamp(
                        value - b_values[:, idx][batch_index],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (
                        v_clipped - b_returns[:, idx][batch_index]
                    ) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - args.ent_coef * entropy_loss
                        + v_loss * args.vf_coef
                    )

                    optimizers[agent].zero_grad()
                    loss.backward()
                    optimizers[agent].step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

        """ TRAINING STORAGE """
        # store the training data
        all_obs[episode] = b_obs
        all_actions[episode] = b_actions
        all_logprobs[episode] = b_logprobs
        all_rewards[episode] = rb_rewards[:end_step]
        all_terms[episode] = rb_terms[:end_step]
        all_values[episode] = b_values
        all_returns[episode] = b_returns
        all_advantages[episode] = b_advantages
        all_loss[episode] = loss.item()
        all_loss_pi[episode] = pg_loss.item()
        all_loss_v[episode] = v_loss.item()
        all_loss_ent[episode] = entropy_loss.item()
        all_loss_info[episode] = {}
        all_loss_info[episode]["entropy"] = entropy_loss.item()
        all_loss_info[episode]["kl"] = approx_kl.item()
        all_loss_info[episode]["policy_loss"] = pg_loss.item()
        all_loss_info[episode]["value_loss"] = v_loss.item()
        all_loss_info[episode]["approxkl"] = approx_kl.item()
        all_loss_info[episode]["clipfrac"] = np.mean(clip_fracs)
        if not args.local:
            wandb.log(
                {
                    "episode": episode,
                    "Episodic Return:": np.mean(total_episodic_return),
                    "Episode Length": end_step,
                    "Value Loss": v_loss.item(),
                    "Policy Loss": pg_loss.item(),
                    "Old Approx KL": old_approx_kl.item(),
                    "Approx KL": approx_kl.item(),
                    "Clip Fraction": np.mean(clip_fracs),
                    "Explained Variance": explained_var.item(),
                    # "all_obs": all_obs[episode],
                    # "all_actions": all_actions[episode],
                    # "all_logprobs": all_logprobs[episode],
                    # "all_rewards": all_rewards[episode],
                    # "all_terms": all_terms[episode],
                    # "all_values": all_values[episode],
                    # "all_returns": all_returns[episode],
                    # "all_advantages": all_advantages[episode],
                    # "all_loss": all_loss[episode],
                    # "all_loss_pi": all_loss_pi[episode],
                    # "all_loss_v": all_loss_v[episode],
                    # "all_loss_ent": all_loss_ent[episode],
                }
            )
    """ RENDER THE POLICY """
    # env = env()
    # env = color_reduction_v0(env)
    # env = resize_v1(env, 64, 64)
    # env = frame_stack_v1(env, args.stack_size=4)

    for agent in agents:
        agents[agent].eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs = batchify_obs(env.reset(seed=None), device)
            # obs = obs
            for step in range(0, args.max_cycles):
                actions = {}
                for idx, agent in enumerate(agents):
                    agent_obs = obs[idx]
                    agent_actions, logprobs, _, values = agents[
                        agent
                    ].get_action_and_value(agent_obs.float(), action=None)
                    actions[agent] = agent_actions

                actions = torch.cat(
                    [actions[agent].view(1) for agent in agents]
                )
                obs, rewards, terms, _, _ = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
