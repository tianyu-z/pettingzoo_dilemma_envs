# adapted from https://pettingzoo.farama.org/content/tutorials/

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical

from pettingzoo.butterfly import pistonball_v6

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

# https://pettingzoo.farama.org/content/tutorials/

def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    # obs = np.stack([obs[a] for a in obs], axis=0)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(actions, env):
    """Converts np array to PZ style arguments."""
    print("actions batchified : ", actions)
    actions = actions.cpu().numpy()
    actions = {a: actions[i] for i, a in enumerate(env.possible_agents)}

    return actions
