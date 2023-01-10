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

from games import Game, Prisoners_Dilemma, Samaritans_Dilemma, Stag_Hunt, Chicken
from dilemma_pettingzoo import raw_env, env, parallel_env
from agents.agent import Agent
from agents.utils import batchify_obs, batchify, unbatchify

"""ALGORITHM PARAMS"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ent_coef = 0.1 # Coefficient for the entropy term in the loss function
vf_coef = 0.1 # Coefficient for the value function term in the loss function
clip_coef = 0.1 # Coefficient for the gradient clipping term in the loss function
gamma = 0.99 # Discount factor used in the Bellman equation
batch_size = 32 # Number of experiences to sample in each training batch
stack_size = 4 # Number of frames to stack together in a state
frame_size = (64, 64) # Height and width of each frame in the stack
max_cycles = 100 # Maximum number of cycles to run the training for
end_step   = max_cycles
total_episodes = 2 # Number of episodes to run the trained model for during evaluation
os.environ["SDL_VIDEODRIVER"] = "dummy"

#####################

if __name__ == "__main__":


    """ ENV SETUP """
    env = parallel_env(render_mode="human", max_cycles=max_cycles)
    # parallel_api_test(env, num_cycles=1000)
    obs = env.reset()

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    # observation_size = env.observation_space(env.possible_agents[0]).shape
    num_observations = 2

    """ LEARNER SETUP """
    # separate policies and optimizers
    # agent = Agent(num_actions=num_actions).to(device)
    agents = {agent : Agent(num_actions=num_actions).to(device) for agent in env.agents}
    optimizers = {agent : optim.Adam(agents[agent].parameters(), lr=0.001, eps=1e-5) for agent in env.agents}

    """ ALGO LOGIC: EPISODE STORAGE"""
    # end_step = 0
    total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, num_observations)).to(device) # stores stacked observations for each agent
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device) # stores actions taken by each agent
    rb_logprobs = torch.zeros((max_cycles, num_agents, num_actions)).to(device) # stores log probabilities of actions taken by each agent
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device) # stores rewards received by each agent
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device) # stores indicators for terminal states encountered by each agent
    rb_values = torch.zeros((max_cycles, num_agents)).to(device) # stores values predicted by the value function for each state and each agent

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):

        # collect an episode
        with torch.no_grad():

            # collect observations and convert to batch of torch tensors
            next_obs = env.reset()
            # reset the episodic return
            total_episodic_return = np.zeros(num_agents)

            # each episode has num_steps
            for step in range(0, max_cycles):

                # rollover the observation

                policy_outputs = {}
                for agent in agents:
                    obs = batchify_obs(next_obs[agent], device)
                    obs = obs.float() #hotpatch, TODO: fix batchify to return floats
                    actions, logprobs, _, values = agents[agent].get_action_and_value(obs, action=None)
                    policy_outputs[agent] ={
                      "actions": actions, 
                      "logprobs": logprobs, 
                      "_": _, 
                      "values": values
                    }

                # join separate action tensors from each agent
                actions = torch.cat([policy_outputs[agent]["actions"].view(1) for agent in agents])

                # execute the environment and log data
                actions_dict = unbatchify(actions, env)
                next_obs, rewards, terms, _, _ = env.step(
                    actions_dict
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

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
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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

    """ RENDER THE POLICY """
    # env = env()
    # env = color_reduction_v0(env)
    # env = resize_v1(env, 64, 64)
    # env = frame_stack_v1(env, stack_size=4)

    agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs = batchify_obs(env.reset(seed=None), device)
            terms = [False]
            while not any(terms):
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                obs, rewards, terms, _ = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]