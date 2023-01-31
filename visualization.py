import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Sample data
def create_gif(
    A, B, x_limit=None, y_limit=None, title=None, filename=None, T=None
):  # Create a figure and axis
    fig, ax = plt.subplots()
    A = np.array(A)
    B = np.array(B)
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
    if y_limit is None:
        y_limit = (min(A.min(), B.min()), max(A.max(), B.max()))
    for i in range(T):
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
            ax.set_title(title + f"epoch {i}")
        else:
            ax.set_title(f"epoch {i}")
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
