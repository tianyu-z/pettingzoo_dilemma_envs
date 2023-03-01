from gfn_dev import sample, init
from utils import load_pt
from agents.GFN_example_from_tutorial.transformer import TransformerModel, make_mlp
from games import (
    Prisoners_Dilemma,
    Samaritans_Dilemma,
    Stag_Hunt,
    Chicken,
)
from gfn_config import get_merged_args


def load_and_sample(filename, num_batches, batch_size=1, start_from=None):
    """Load a model and sample from it.
    Args:
        filename (str): Path to the model file.
        num_batches (int): Number of batches to sample.
        batch_size (int): Batch size.
        start_from (list): List of strings to start from.
    Returns:
        samples (list): List of samples.
        samples_R (list): List of rewards.
        samples_and_Reward (list): List of samples with rewards.
    """
    args, game, logZ, model, optim, device = init()

    # Load the model
    checkpoint_dict = load_pt(filename)
    model.load_state_dict(checkpoint_dict["GFN_model_state_dict"])
    samples, samples_R, samples_and_reward, _ = sample(
        args,
        model,
        device,
        game,
        num_batches,
        batch_size=batch_size,
        start_from=start_from,
        condition_sample_size=128,
    )
    return samples, samples_R, samples_and_reward


if __name__ == "__main__":
    print(
        load_and_sample(
            "/home/tiany/pettingzoo_dilemma_envs/checkpoints/63ecf3d0/checkpoints_1400.pt",
            num_batches=100,
            batch_size=1024,
            start_from=["1", "2", "3"],
        )[0]
    )
