import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        action="store_true",
        help="Whether to use wandb for logging",
    )
    parser.add_argument(
        "--nb_hidden",
        type=int,
        default=128,
        help="Number of hidden units in the network",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.1,
        help="Entropy coefficient for the loss calculation",
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=0.1,
        help="Value function coefficient for the loss calculation",
    )
    parser.add_argument(
        "--clip_coef",
        type=float,
        default=0.1,
        help="Clipping coefficient for PPO",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--stack_size",
        type=int,
        default=4,
        help="Number of frames to stack together in a state",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        default=64,
        help="Height and width of each frame in the stack",
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=1000,
        help="Maximum number of cycles to train for",
    )
    parser.add_argument(
        "--total_episodes",
        type=int,
        default=2,
        help="Total number of episodes to train for",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Epsilon value for the optimizer",
    )

    args = parser.parse_args()

    return args
