import torch
import numpy as np
from games import (
    Game,
    Prisoners_Dilemma,
    Samaritans_Dilemma,
    Stag_Hunt,
    Chicken,
)
from copy import deepcopy
from itertools import product
from visualization import create_gif, plot_dict, get_top_k
from gfn_config import get_merged_args


def tokenize_actions(game):
    actionspair2int = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    # key: (action1, action2), value: a token to represent the pair
    actionint2pair = {v: k for k, v in actionspair2int.items()}
    # key: a token to represent the pair, value: (action1, action2)
    actionspair2str = {(0, 0): "CC", (0, 1): "CD", (1, 0): "DC", (1, 1): "DD"}
    # key: (action1, action2), value: a string to represent the pair C: cooperate, D: defect
    actionstr2pair = {v: k for k, v in actionspair2str.items()}
    # key: a string to represent the pair C: cooperate, D: defect, value: (action1, action2)
    int_payoff = {actionspair2int[k]: v for k, v in game.payoff.items()}
    # key: a token to represent the pair, value: (reward1, reward2)
    int_payoff[4] = (0, 0)  # None, None is a tie
    int_payoff[5] = (0, 0)
    return actionspair2int, actionint2pair, actionspair2str, actionstr2pair, int_payoff


def reward_func(game, actions, is_sum_agent_rewards=False, only_last=False):
    """
    discription: calculate the reward of a sequence of actions
    input:
        game: a game object
        actions: a list of actions [4,0,1,2,3,...] or a 2D array of actions [[0,0],[0,1],[1,1],...]
        is_sum_agent_rewards: if True, sum the rewards of two agents
        only_last: if True, only return the reward of the last action
    """
    (
        actionspair2int,
        actionint2pair,
        actionspair2str,
        actionstr2pair,
        int_payoff,
    ) = tokenize_actions(game)

    if torch.is_tensor(actions):
        assert len(actions.shape) <= 2, "actions must be a 1D or 2D tensor"
        assert max(actions).item() <= 4, "actions must be in [0, 1, 2, 3, 4]"
        if len(actions.shape) == 2:
            T, _ = actions.shape  # time, nb_actions
            actions_ = []
            for t in range(T):
                actions_.append(actionspair2int[tuple(actions[t].numpy())])
        else:
            actions_ = tuple(actions.numpy())
    elif isinstance(actions, (list, tuple)):
        actions_ = actions
    acc_reward = np.zeros(2)
    acc_rewards_ts = []

    for a in actions_:
        acc_reward += np.array(int_payoff[a])
        if not is_sum_agent_rewards:
            acc_rewards_ts.append(deepcopy(acc_reward))
        else:
            acc_rewards_ts.append(deepcopy(acc_reward).sum())
    if only_last:
        return acc_rewards_ts[-1]
    else:
        return acc_rewards_ts


def batch_reward(game, list_of_actions, is_sum_agent_rewards=False, only_last=True):
    return [
        x
        for x in map(
            lambda x: reward_func(
                game, x, is_sum_agent_rewards=only_last, only_last=only_last
            ),
            list_of_actions,
        )
    ]


def list2string(l):
    """convert a list of integers to a string"""
    return "".join([str(i) for i in l])


def string2list(s):
    """convert a string to a list of integers"""
    return [int(i) for i in list(s)]


def get_true_dist(args):
    if args.game_type == "PD":
        game = Prisoners_Dilemma()
    elif args.game_type == "SD":
        game = Samaritans_Dilemma()
    elif args.game_type == "SH":
        game = Stag_Hunt()
    elif args.game_type == "CH":
        game = Chicken()

    xs = list(product([0, 1, 2, 3], repeat=args.max_len - 1))
    xs = [[args.bos_index] + list(x) for x in xs]  # begin with <EOS>
    xs_string = [list2string(x) for x in xs]
    all_rewards = batch_reward(game, xs, is_sum_agent_rewards=True)
    true_dist = torch.tensor(all_rewards).softmax(0).cpu().numpy()
    true_dist_dict = {k: v for k, v in zip(xs_string, true_dist)}
    return true_dist, true_dist_dict, xs_string


if __name__ == "__main__":
    args = get_merged_args()
    true_dist, true_dist_dict, xs_string = get_true_dist(args)
    print(true_dist)
    print(true_dist.shape)
    plot_dict(get_top_k(true_dist_dict, 20))
