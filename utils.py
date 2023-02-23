import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import os
import glob
import time
from collections import Counter


def tokenize_actions(game=None):
    actionspair2int = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    # key: (action1, action2), value: a token to represent the pair
    actionint2pair = {v: k for k, v in actionspair2int.items()}
    # key: a token to represent the pair, value: (action1, action2)
    actionspair2str = {(0, 0): "CC", (0, 1): "CD", (1, 0): "DC", (1, 1): "DD"}
    # key: (action1, action2), value: a string to represent the pair C: cooperate, D: defect
    actionstr2pair = {v: k for k, v in actionspair2str.items()}
    # key: a string to represent the pair C: cooperate, D: defect, value: (action1, action2)
    if game is None:
        int_payoff = None
    else:
        int_payoff = {actionspair2int[k]: v for k, v in game.payoff.items()}
        int_payoff[4] = (0, 0)  # None, None is a tie
        int_payoff[5] = (0, 0)
    # key: a token to represent the pair, value: (reward1, reward2)
    return actionspair2int, actionint2pair, actionspair2str, actionstr2pair, int_payoff


# Sample data
def create_gif(
    emp_ts, true_ts, x_limit=None, y_limit=None, title=None, filename=None, T=None
):  # Create a figure and axis
    """
    input:
        emp_ts: list of numpy array, each numpy array is the empirical distribution at time t
        true_ts: list of numpy array, each numpy array is the true distribution at time t
        x_limit: tuple, (min, max) of x-axis
        y_limit: tuple, (min, max) of y-axis
        title: str or list of str, title of each frame
        filename: str, filename of the gif
        T: int, number of frames
    discrpition:
        create a gif of the comparison between the empirical distribution and the true distribution
    """
    fig, ax = plt.subplots()
    A = np.array(emp_ts)
    B = np.array(true_ts)
    isConstantSeries = len(B.shape) == 1
    if T is None and A.shape == B.shape:
        # both A and B are series that vary over time
        # T , n = A.shape
        T = min(len(A), len(B))
    elif T is None and isConstantSeries:
        # B is a constant series
        T = len(A)
    # Loop through the range of i
    if x_limit is None:
        n = max(A.shape[-1], B.shape[-1])
        x_limit = (0, n)

    for i in range(T):
        if y_limit is None:
            if isConstantSeries:
                y_limit = (min(A[i].min(), B.min()), max(A[i].max(), B.max()))
            else:
                y_limit = (min(A[i].min(), B[i].min()), max(A[i].max(), B[i].max()))
        # Plot the data
        ax.cla()
        # Fixing x-axis and y-axis bounds
        if x_limit is not None:
            ax.set_xlim(*x_limit)
        if y_limit is not None:
            ax.set_ylim(*y_limit)
        ax.plot(A[i], label="Estimated_distribution")
        if isConstantSeries:
            ax.plot(B, label="True_distribution")
        else:
            ax.plot(B[i], label="True_distribution")
        # Add a title
        if title is not None:
            if isinstance(title, list):
                ax.set_title(title[i] + f"frame {i}")
            elif isinstance(title, str):
                ax.set_title(title + f"frame {i}")
        else:
            ax.set_title(f"frame {i}")
        # Save the current frame as an image
        fig.savefig(f"frame{i}.png")

    # Generate the gif using pillow
    import os

    frames = []
    for i in range(T):
        frames.append(Image.open(f"frame{i}.png"))
    if filename is None:
        filename = "animation.gif"
    if not filename.endswith(".gif"):
        filename += ".gif"
    frames[0].save(
        filename, save_all=True, append_images=frames[1:], duration=500, loop=0
    )

    # Cleaning up
    for i in range(T):
        os.remove(f"frame{i}.png")
    return Image.open(filename)


def create_gif_by_dicts(
    emp_ts_dict,
    true_ts_dict,
    x_limit=None,
    y_limit=None,
    title=None,
    filename=None,
    T=None,
):  # Create a figure and axis
    """
    def plot_dict(d):
        x = list(d.keys())
        y = list(d.values())

        plt.bar(x, y)
        plt.show()
        return plt
    input:
        emp_ts_dict: list of dict, each dict is the empirical distribution at time t
        true_ts_dict: list of dict, each dict is the true distribution at time t
        x_limit: tuple, (min, max) of x-axis
        y_limit: tuple, (min, max) of y-axis
        title: str or list of str, title of each frame
        filename: str, filename of the gif
        T: int, number of frames
    discrpition:
        create a gif of the comparison between the empirical distribution and the true distribution
    """
    A, B = emp_ts_dict, true_ts_dict
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    isConstantSeries = isinstance(true_ts_dict, dict)
    for i in range(len(emp_ts_dict)):
        if y_limit is None:
            y_limit = (
                min(min(emp_ts_dict[i].values()), min(true_ts_dict.values())),
                max(max(emp_ts_dict[i].values()), max(true_ts_dict.values())),
            )
        ax.cla()
        ax.set_ylim(*y_limit)
        xA = list(A[i].keys())
        yA = list(A[i].values())
        ax.bar(xA, yA, label="Estimated_distribution", alpha=0.5)
        if isConstantSeries:
            xB = list(B.keys())
            yB = list(B.values())
            ax.bar(xB, yB, label="True_distribution", alpha=0.5)
        else:
            xB = list(B[i].keys())
            yB = list(B[i].values())
            ax.bar(xB, yB, label="True_distribution", alpha=0.5)
        # Add a title
        if title is not None:
            if isinstance(title, list):
                ax.set_title(title[i] + f"frame {i}")
            elif isinstance(title, str):
                ax.set_title(title + f"frame {i}")
        else:
            ax.set_title(f"frame {i}")
        # Save the current frame as an image
        fig.savefig(f"frame{i}.png")

    # Generate the gif using pillow
    import os

    frames = []
    if T is None:
        T = len(emp_ts_dict)
    for i in range(T):
        frames.append(Image.open(f"frame{i}.png"))
    if filename is None:
        filename = "animation.gif"
    if not filename.endswith(".gif"):
        filename += ".gif"
    frames[0].save(
        filename, save_all=True, append_images=frames[1:], duration=500, loop=0
    )

    # Cleaning up
    for i in range(T):
        os.remove(f"frame{i}.png")
    return Image.open(filename)


def plot_dict(d):
    """
    input:
        d: dict
    description:
        plot the dict
    """
    x = list(d.keys())
    y = list(d.values())

    plt.bar(x, y)
    plt.show()
    return plt


def get_top_k(d, k, return_keys=False):
    """
    d = {'apple': 10, 'banana': 20, 'cherry': 30, 'dates': 40, 'figs': 15}
    top_3 = get_top_k(d, 3)
    print(top_3)
    # Output: {'dates': 40, 'cherry': 30, 'banana': 20}
    """
    sorted_d = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
    out = dict(list(sorted_d.items())[:k])
    if return_keys:
        return out, out.keys()
    else:
        return out


def filter_dict_by_keys(subkeys, d):
    """
    d = {'apple': 10, 'banana': 20, 'cherry': 30, 'dates': 40, 'figs': 15}
    subkeys = ['dates', 'figs']
    filtered = filter_dict_by_keys(subkeys, d)
    print(filtered)
    # Output: {'dates': 40, 'figs': 15}
    """
    return {k: d[k] for k in subkeys if k in d}


def normalize_dict_values(d):
    """
    d = {'apple': 10, 'banana': 20, 'cherry': 30, 'dates': 40, 'figs': 15}
    normalized = normalize_dict_values(d)
    print(normalized)
    # Output: {'apple': 0.1, 'banana': 0.2, 'cherry': 0.3, 'dates': 0.4, 'figs': 0.15}
    """
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}


def save_pt(pt, filename):
    """
    input:
        pt: torch.tensor or dictionary
        filename: str
    description:
        save a torch.tensor to a file
    """
    torch.save(pt, filename)


def load_pt(filename):
    """
    input:
        filename: str
    description:
        load a torch.tensor from a file
    """
    return torch.load(filename)


def get_hex_time():
    """
    Description:
        get the current time in the format "DD/MM/YY HH:MM:SS" and convert it to a hexadecimal string
    """
    current_time = time.strftime("%d/%m/%y %H:%M:%S", time.localtime())
    # convert the timestamp string to a Unix timestamp
    unix_time = int(time.mktime(time.strptime(current_time, "%d/%m/%y %H:%M:%S")))

    # convert the Unix timestamp to a hexadecimal string
    hex_time = hex(unix_time)[2:]

    return hex_time


def hex_to_time(hex_time):
    """
    input:
        hex_time: str
    description:
        convert a hexadecimal string to a timestamp string in the format "DD/MM/YY HH:MM:SS"
    """
    # convert the hexadecimal string to a Unix timestamp
    unix_time = int(hex_time, 16)

    # convert the Unix timestamp to a timestamp string in the format "DD/MM/YY HH:MM:SS"
    time_str = time.strftime("%d/%m/%y %H:%M:%S", time.localtime(unix_time))

    return time_str


def delete_oldest_files(directory, max_count, prefix="checkpoint"):
    # Get a list of files in the directory that match the prefix
    file_list = sorted(glob.glob(os.path.join(directory, f"{prefix}*")))

    # If the number of files is less than or equal to the maximum count, do nothing
    if len(file_list) <= max_count:
        return

    # Otherwise, sort the files by modification time (oldest first) and delete the oldest ones
    file_list = sorted(file_list, key=lambda f: os.stat(f).st_mtime)
    for i in range(len(file_list) - max_count):
        os.remove(file_list[i])


(
    actionspair2int,
    actionint2pair,
    actionspair2str,
    actionstr2pair,
    int_payoff,
) = tokenize_actions(None)


def tokenize_actions_dict(loaded):
    """
    Args:
        loaded: a dict of loaded data
        e.g. loaded = {"player_0_actions": [1,1,1], "player_1_actions": [0,0,0],
                        "player_0_rewards": [...], "player_1_rewards": [...]}
    Return:
        str_tokenized_actions_list: a list of string tokenized actions e.g. ["DD", "CC", ...]
        num_tokenized_actions_list: a list of number tokenized actions e.g. [3, 0 , ...]
        action_summary: a Counter object of the action summary e.g. {"CC": 10, "CD": 20, "DC": 30, "DD": 40}
    """
    actions = []
    action_counts = 0
    for k in loaded.keys():
        if "action" in k:
            action_counts += 1
    for i in range(action_counts):
        actions.append(loaded[f"player_{i}_actions"])
    str_tokenized_actions_list = [
        actionspair2str[action_pair] for action_pair in zip(*actions)
    ]
    num_tokenized_actions_list = [
        actionspair2int[action_pair] for action_pair in zip(*actions)
    ]
    action_summary = Counter(str_tokenized_actions_list)
    return str_tokenized_actions_list, num_tokenized_actions_list, action_summary


def get_acc_freq(loaded, interested_action):
    """
    Find the accumulative frequency of a specific pairs of actions (like "CC", "CD", "DC", "DD")
    Args:
        loaded: a list of loaded data
        interested_action: if not None, only plot the accumulative frequency of this action
    Return:
        acc_freqs: a dict of accumulative frequency of actions
    e.g. input: loaded = {"player_0_actions": [1,1,1], "player_1_actions": [0,0,0],
                        "player_0_rewards": [...], "player_1_rewards": [...]}

    """
    (
        str_tokenized_actions_list,
        num_tokenized_actions_list,
        action_summary,
    ) = tokenize_actions_dict(loaded)
    acc_freq = []
    if interested_action is None or interested_action == "":
        interested_actions = action_summary.keys()
    else:
        interested_actions = [interested_action]
    acc_freqs = {}
    for i_a in interested_actions:
        acc_freq = []
        for i in range(len(str_tokenized_actions_list)):
            window = str_tokenized_actions_list[0 : i + 1]
            count = window.count(i_a)
            acc_freq.append(count)
        acc_freqs[i_a] = acc_freq
    return acc_freqs, interested_actions


def plot_action_accumulative_frequency(
    loaded, plot_bounds=False, interested_action=None, beta=1.96, include_y_eq_x=False
):
    """Plot the accumulative frequency of actions
    Args:
        loaded: a list of loaded data, it is from the loaded checkpoints from the two_player_game.py
        plot_bounds: if True, plot the upper and lower bounds of the mean
        interested_action: if not None, only plot the accumulative frequency of this action
        beta: the beta value for the upper and lower bounds (lower = mean - beta * std; upper = mean + beta * std),
             default to 1.96 for 95% confidence interval
    """
    if plot_bounds:
        tmp = []
        for l in loaded:
            tmp_acc_freqs, interested_actions = get_acc_freq(l, interested_action)
            tmp.append(tmp_acc_freqs)
        acc_freqs, acc_freqs_upper, acc_freqs_lower = compute_mean_std_dict(tmp, beta)
    else:
        acc_freqs, interested_actions = get_acc_freq(loaded, interested_action)
    # Plot
    for i_a in interested_actions:
        plt.plot(list(range(len(acc_freqs[i_a]))), acc_freqs[i_a], label=i_a)
        plt.plot(list(range(len(acc_freqs_lower[i_a]))), acc_freqs_lower[i_a])
        plt.plot(list(range(len(acc_freqs_upper[i_a]))), acc_freqs_upper[i_a])
        if plot_bounds:
            plt.fill_between(
                list(range(len(acc_freqs[i_a]))),
                acc_freqs_lower[i_a],
                acc_freqs_upper[i_a],
                alpha=0.2,
            )
    if include_y_eq_x:
        plt.plot(
            list(range(len(acc_freqs[i_a]))),
            list(range(len(acc_freqs[i_a]))),
            label="y=x",
        )
    plt.legend()
    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel(f"accumulative frequency of action {interested_action}")
    plt.title("accumulative frequency")
    plt.show()
    if plot_bounds:
        return acc_freqs, acc_freqs_lower, acc_freqs_upper
    else:
        return acc_freqs, None, None


def compute_mean_std_dict(input_dicts, beta=1.96):
    """
    Compute the mean and standard deviation of a list of dictionaries
    Args:
        input_dicts: a list of dictionaries
        beta: the beta value for the upper and lower bounds (lower = mean - beta * std; upper = mean + beta * std),
             default to 1.96 for 95% confidence interval
    Returns:
        mean_dicts: a dictionary of the mean of each key
        upper_dicts: a dictionary of the upper bound of each key
        lower_dicts: a dictionary of the lower bound of each key
    """
    # Initialize an empty dictionary to accumulate the lists
    accumulated_lists = {}

    # Iterate over the dictionaries in A and accumulate the lists by key
    for dict_elem in input_dicts:
        for key, val in dict_elem.items():
            if key not in accumulated_lists:
                accumulated_lists[key] = []
            accumulated_lists[key].append(val)

    # Compute the mean of each accumulated list using numpy
    mean_dicts = {}
    upper_dicts = {}
    lower_dicts = {}
    for key, lists in accumulated_lists.items():
        mean_dicts[key] = np.mean(lists, axis=0).tolist()
        upper_dicts[key] = (
            np.mean(lists, axis=0) + beta * np.std(lists, axis=0)
        ).tolist()
        lower_dicts[key] = (
            np.mean(lists, axis=0) - beta * np.std(lists, axis=0)
        ).tolist()
    return mean_dicts, upper_dicts, lower_dicts


def get_error_bounds(loaded, beta=1.96):
    """
    Compute the mean and standard deviation of a list of list
    Args:
        loaded: a list of list
        beta: the beta value for the upper and lower bounds (lower = mean - beta * std; upper = mean + beta * std),
    Returns:
        mean_ts: a list of the mean of each key
        upper_error_bounds: a list of the upper bound of each key
        lower_error_bounds: a list of the lower bound of each key
    """
    loaded = np.array(loaded)
    assert len(loaded.shape) == 2, "loaded must be a 2D array"
    mean_ts = np.mean(loaded, axis=0)
    std_ts = np.std(loaded, axis=0)
    upper_error_bounds = mean_ts + beta * std_ts
    lower_error_bounds = mean_ts - beta * std_ts
    return mean_ts, upper_error_bounds, lower_error_bounds


if __name__ == "__main__":
    # test create gif
    # A = np.random.rand(100, 10)
    # B = np.random.rand(10)
    # create_gif(A, B, title="test", filename="test.gif")
    # test plot accumulative graph
    loaded = load_pt(
        "loggings.pt"
    )  # this is from the two_player_game.py, we save the evaluation results in this file
    plot_action_accumulative_frequency(loaded, plot_bounds=True)
