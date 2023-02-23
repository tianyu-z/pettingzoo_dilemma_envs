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


def load_and_sample(filename, num_samples):
    args, game, logZ, model, optim, device = init()

    # Load the model
    checkpoint_dict = load_pt(filename)
    model.load_state_dict(checkpoint_dict["GFN_model_state_dict"])
    samples, samples_R = sample(args, model, device, game, num_samples)
    return samples, samples_R


if __name__ == "__main__":
    print(
        load_and_sample(
            "/home/tiany/pettingzoo_dilemma_envs/checkpoints/63ecf3d0/checkpoints_1400.pt",
            num_samples=100,
        )[0]
    )
