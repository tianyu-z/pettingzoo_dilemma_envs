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
from agents.mediator import Mediator
from agents.utils import batchify_obs, batchify, unbatchify
from config import parse_args
from utils import save_pt, LimitedStack, tokenize_actions, load_pt
import tqdm
from gfn_dev import init, sample

"""ALGORITHM PARAMS"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["SDL_VIDEODRIVER"] = "dummy"

med_message_size = 4  # 8 bits

#####################

if __name__ == "__main__":
    args = parse_args()
    print("Exp setting: ", args)
    if not args.local:
        import wandb
    end_step = args.max_cycles

    """ENV SETUP"""
    actionspair2int, _, _, _, _ = tokenize_actions()
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
    num_observations = 2 + med_message_size  # 2 = Coop and defect

    """ LEARNER SETUP """
    # separate policies and optimizers
    # agent = Agent(num_actions=num_actions).to(device)
    agents = {
        agent: Agent(num_input=num_observations, num_actions=num_actions).to(device)
        for agent in env.agents
    }
    optimizers = {
        agent: optim.Adam(agents[agent].parameters(), lr=args.lr, eps=args.eps)
        for agent in env.agents
    }

    """ GFN SAMPLPER SETUP """
    GFN_sample_len_without_bos = 6
    """ MEDIATOR SETUP"""
    mediator = Mediator(
        num_actions=med_message_size,
        num_additional_info=GFN_sample_len_without_bos + 1,
    ).to(device)

    mediator_optimizer = optim.Adam(mediator.parameters(), lr=args.lr, eps=args.eps)

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

    """ MEDIATOR STORAGE """
    all_med_obs = {}
    all_med_actions = {}
    all_med_logprobs = {}
    all_med_rewards = {}
    all_med_terms = {}
    all_med_values = {}
    all_med_returns = {}
    all_med_advantages = {}
    all_med_loss = {}
    all_med_loss_pi = {}
    all_med_loss_v = {}
    all_med_loss_ent = {}
    all_med_loss_info = {}
    all_med_loss_info["entropy"] = {}
    all_med_loss_info["kl"] = {}
    all_med_loss_info["policy_loss"] = {}
    all_med_loss_info["value_loss"] = {}
    all_med_loss_info["approxkl"] = {}
    all_med_loss_info["clipfrac"] = {}

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

    """ MEDIATOR LOGIC: EPISODE STORAGE"""
    med_total_episodic_return = 0
    med_rb_obs = torch.zeros(
        (args.max_cycles, num_agents, num_observations - med_message_size)
    ).to(
        device
    )  # stores stacked observations for each agent
    med_rb_actions = torch.zeros(
        (args.max_cycles, num_agents, num_observations - med_message_size)
    ).to(
        device
    )  # stores actions taken by each agent
    med_rb_logprobs = torch.zeros((args.max_cycles)).to(
        device
    )  # stores log probabilities of actions taken by each agent
    med_rb_rewards = torch.zeros((args.max_cycles)).to(
        device
    )  # stores rewards received by each agent
    med_rb_terms = torch.zeros((args.max_cycles)).to(
        device
    )  # stores indicators for terminal states encountered by each agent
    med_rb_values = torch.zeros((args.max_cycles)).to(
        device
    )  # stores values predicted by the value function for each state and each agent

    """
    gfn_model_loading
    """
    args_gfn, game, logZ, model, optim, device = init()
    checkpoint_dict = load_pt(
        "/home/tiany/pettingzoo_dilemma_envs/checkpoints/63ecf3d0/checkpoints_1400.pt"
    )
    model.load_state_dict(checkpoint_dict["GFN_model_state_dict"])

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(args.total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs = env.reset()
            # reset the episodic return
            total_episodic_return = np.zeros(num_agents)
            med_total_episodic_return = 0

            # init a stack for history recording
            history = LimitedStack(args.max_history)
            # each episode has num_steps
            for step in tqdm.trange(0, args.max_cycles):
                # run GFN_sampler
                history_ = history.lst
                # _, _, sample_and_reward = load_and_sample(
                #     ,
                #     num_batches=1,
                #     batch_size=8192,
                #     start_from=history_,
                # )  # get the samples and the reward
                samples, samples_R, samples_and_reward, _ = sample(
                    args_gfn,
                    model,
                    device,
                    game,
                    num_batches=1,
                    batch_size=8192,
                    start_from=history_,
                    condition_sample_size=1,
                )
                input_sample = samples_and_reward[0][1:]  # [1:] is to remove the bos
                # rollover the observation
                obs = batchify_obs(next_obs, device)
                obs_togo = torch.cat(
                    (obs[0], torch.tensor(input_sample).to(device)), dim=0
                )
                # run the mediator
                (
                    med_actions,
                    med_logprobs,
                    _,
                    med_values,
                    med_logits,
                ) = mediator.get_action_and_value(obs_togo.float(), action=None)
                med_actions = med_actions.to(device)
                # run the agents
                policy_outputs = {}
                for idx, agent in enumerate(agents):
                    agent_obs = obs[idx]
                    # add the mediator action to the observation
                    agent_obs = torch.cat((agent_obs, med_logits.squeeze()), dim=0)
                    agent_obs = agent_obs.float()  # hotpatch
                    (
                        agent_actions,
                        agent_logprobs,
                        _,
                        agent_values,
                        agent_logits,
                    ) = agents[agent].get_action_and_value(agent_obs, action=None)
                    policy_outputs[agent] = {
                        "actions": agent_actions,
                        "logprobs": agent_logprobs,
                        "_": _,
                        "values": agent_values,
                        "agent_logits": agent_logits,
                    }

                # join separate tensors from each agent
                agent_logits = torch.stack(
                    [policy_outputs[agent]["agent_logits"] for agent in agents]
                )
                actions = torch.cat(
                    [policy_outputs[agent]["actions"].view(1) for agent in agents]
                )
                onedigit_actions = actionspair2int[tuple(actions.cpu().numpy())]
                history.push(onedigit_actions)
                logprobs = torch.cat(
                    [policy_outputs[agent]["logprobs"].view(1) for agent in agents]
                )
                values = torch.cat(
                    [policy_outputs[agent]["values"].view(1) for agent in agents]
                )
                # execute the environment and log data
                actions_dict = unbatchify(actions, env)

                next_obs, rewards, terms, _, _ = env.step(actions_dict)
                # print("next_obs: ", next_obs)
                # add to episode storage
                # concatenate obs with mediator action and add to rb_obs
                # stack mediator action twice to match agent action shape

                # TODO
                med_actions = torch.stack((med_logits, med_logits))

                rb_obs[step] = torch.cat((obs, med_actions), dim=1)
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values

                med_rb_obs[step] = obs
                med_rb_rewards[step] = torch.mean(rb_rewards[step])
                med_rb_actions[step] = agent_logits
                med_rb_logprobs[step] = med_logprobs
                med_rb_values[step] = med_values

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()
                # TODO : aggregate rewards from all agents
                med_total_episodic_return += (
                    torch.mean(med_rb_rewards[step]).cpu().numpy()
                )

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            med_rb_advantages = torch.zeros_like(med_rb_rewards).to(device)
            for t in reversed(range(end_step - 1)):
                delta = (
                    rb_rewards[t]
                    + args.gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = (
                    delta + args.gamma * args.gamma * rb_advantages[t + 1]
                )
                # compute advantages for mediator
                med_delta = (
                    med_rb_rewards[t]
                    + args.gamma * med_rb_values[t + 1] * med_rb_terms[t + 1]
                    - med_rb_values[t]
                )
                med_rb_advantages[t] = (
                    med_delta + args.gamma * args.gamma * med_rb_advantages[t + 1]
                )

            rb_returns = rb_advantages + rb_values
            med_rb_returns = med_rb_advantages + med_rb_values

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

        med_b_obs = med_rb_obs[:end_step]
        med_b_logprobs = med_rb_logprobs[:end_step]
        med_b_actions = med_rb_actions[:end_step]
        med_b_returns = med_rb_returns[:end_step]
        med_b_values = med_rb_values[:end_step]
        med_b_advantages = med_rb_advantages[:end_step]

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            len_b_obs = len(b_obs)
            for start in range(0, len(b_obs), args.batch_size):
                # select the indices we want to train on
                end = start + args.batch_size
                batch_index = b_index[start:end]

                for idx, agent in enumerate(agents):
                    _, newlogprob, entropy, value = agents[agent].get_action_and_value(
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
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
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
                    v_loss_unclipped = (value - b_returns[:, idx][batch_index]) ** 2
                    v_clipped = b_values[:, idx][batch_index] + torch.clamp(
                        value - b_values[:, idx][batch_index],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[:, idx][batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    optimizers[agent].zero_grad()
                    loss.backward()
                    optimizers[agent].step()

                # train the mediator
                obs_batch = med_b_obs[batch_index][:, 0, :]
                action_batch = med_b_actions.long()[batch_index][:, 0, :]
                _, newlogprob, entropy, value = mediator.get_action_and_value(
                    obs_batch,
                    action_batch,
                )
                logratio = newlogprob - med_b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = med_b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -med_b_advantages[batch_index] * ratio
                pg_loss2 = -med_b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - med_b_returns[batch_index]) ** 2
                v_clipped = med_b_values[batch_index] + torch.clamp(
                    value - med_b_values[batch_index],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - med_b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                mediator_optimizer.zero_grad()
                loss.backward()
                mediator_optimizer.step()

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
    mediator.eval()

    all_loggings = []
    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs = batchify_obs(env.reset(seed=None), device)
            # run the mediator
            if args.save_local_logging_pts:
                logging_actions = {a + "_actions": [] for a in agents}
                logging_rewards = {a + "_rewards": [] for a in agents}
                loggings = {**logging_actions, **logging_rewards}
            for step in tqdm.trange(0, args.max_cycles):
                (
                    med_actions,
                    med_logprobs,
                    _,
                    med_values,
                ) = mediator.get_action_and_value(obs[0].float(), action=None)
                med_actions = med_actions.to(device)
                actions = {}
                for idx, agent in enumerate(agents):
                    agent_obs = obs[idx]
                    agent_obs = torch.cat((agent_obs, med_actions), dim=0)
                    agent_obs = agent_obs.float()  # hotpatch
                    agent_actions, logprobs, _, values = agents[
                        agent
                    ].get_action_and_value(agent_obs.float(), action=None)
                    actions[agent] = agent_actions

                actions = torch.cat([actions[agent].view(1) for agent in agents])
                obs, rewards, terms, _, _ = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                if args.save_local_logging_pts:
                    for idx, agent in enumerate(agents):
                        loggings[agent + "_actions"].append(actions[idx].item())
                        loggings[agent + "_rewards"].append(
                            rewards["player_" + str(idx)]
                        )
            if args.save_local_logging_pts:
                all_loggings.append(loggings)
    env.close()
    if args.save_local_logging_pts:
        save_pt(
            all_loggings, "agents_loggings/two_agent_simple_mediator_loggings_eval.pt"
        )
