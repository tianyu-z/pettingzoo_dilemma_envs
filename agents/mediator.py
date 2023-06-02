# adapted from https://pettingzoo.farama.org/content/tutorials/

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
from torch.distributions.multinomial import Multinomial

from pettingzoo.butterfly import pistonball_v6

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"


def decimal_to_binary(integers, num_bits):
    """Converts an integer to a binary vector."""
    # if integers is 0-d tensor, convert to 1-d tensor
    if integers.dim() == 0:
        integers = integers.unsqueeze(0)
    binary = np.zeros((len(integers), num_bits))
    for i, integer in enumerate(integers):
        binary[i, :integer] = 1
    binary = torch.tensor(binary)
    return binary


def map_binary_to_integer(binary):
    """Converts a tensor of binary vectors to integers."""

    integers = binary[:, 0] * 2 + binary[:, 1]
    return integers


class Mediator(nn.Module):
    def __init__(
        self, num_actions=4, num_input=2, num_hidden=512, num_additional_info=0
    ):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(num_input + num_additional_info, num_hidden)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(num_hidden, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(num_hidden, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

            action_binary = decimal_to_binary(action, 2)
            return (
                action_binary.squeeze(),
                probs.log_prob(action),
                probs.entropy(),
                self.critic(hidden),
                logits,
            )
        else:
            decimal_actions = map_binary_to_integer(action)
            return (
                action,
                probs.log_prob(decimal_actions),
                probs.entropy(),
                self.critic(hidden),
                logits,
            )
