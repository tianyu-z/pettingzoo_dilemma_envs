import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


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


if __name__ == "__main__":
    A = np.random.rand(100, 10)
    B = np.random.rand(10)
    create_gif(A, B, title="test", filename="test.gif")
