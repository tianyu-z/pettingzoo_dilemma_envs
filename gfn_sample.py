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


def load_and_sample(filename, num_batches, batch_size=1, no_bos=True, start_from=None):
    args, game, logZ, model, optim, device = init()

    # Load the model
    checkpoint_dict = load_pt(filename)
    model.load_state_dict(checkpoint_dict["GFN_model_state_dict"])
    samples, samples_R = sample(
        args, model, device, game, num_batches, batch_size, start_from
    )
    if no_bos:
        if samples[0][0] == 5:
            samples = [x[1:] for x in samples]
    return samples, samples_R, [x + [y] for x, y in zip(samples, samples_R)]


if __name__ == "__main__":
    print(
        load_and_sample(
            "/home/tiany/pettingzoo_dilemma_envs/checkpoints/63ecf3d0/checkpoints_1400.pt",
            num_batches=100,
            start_from=["1", "2", "3"],
        )[0]
    )
