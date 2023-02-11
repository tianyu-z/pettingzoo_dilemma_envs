import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Sample data
def create_gif(
    emp_ts, true_ts, x_limit=None, y_limit=None, title=None, filename=None, T=None
):  # Create a figure and axis
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
    fig, ax = plt.subplots()
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
        yA = list(B[i].values())
        ax.plt.bar(xA, yA, label="Estimated_distribution")
        if isConstantSeries:
            xB = list(B.keys())
            yB = list(B.values())
            ax.plt.bar(xB, yB, label="True_distribution")
        else:
            xB = list(B[i].keys())
            yB = list(B[i].values())
            ax.plt.bar(xB, yB, label="True_distribution")
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


if __name__ == "__main__":
    A = np.random.rand(100, 10)
    B = np.random.rand(10)
    create_gif(A, B, title="test", filename="test.gif")
