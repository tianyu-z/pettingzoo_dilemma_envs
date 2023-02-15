import argparse
import yaml


def merge_config(args, config_file):
    """
    Merge config file with args
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    for key in config:
        setattr(args, key, config[key])
    return args


def merge_args(args1, args2, default_args=None):
    """
    Merge two argparse.Namespace objects
    """
    args1_dict = vars(args1)
    args2_dict = vars(args2)
    merged_args = argparse.Namespace()
    for key in args1_dict:
        if key not in args2_dict:
            setattr(merged_args, key, args1_dict[key])
        else:
            setattr(merged_args, key, default_args[key])
    for key in args2_dict:
        if key not in args1_dict:
            setattr(merged_args, key, args2_dict[key])
    return merged_args


def get_merged_args():
    args = parse_args()
    if args.config is not None or args.config != "":
        merge_config(args, args.config)
    return args


def parse_args(return_parser=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_type", type=str, default="PD")
    parser.add_argument(
        "--vocab",
        nargs="+",
        default=None,
    )
    parser.add_argument("--pad_index", type=int, default=4)
    parser.add_argument("--bos_index", type=int, default=5)
    parser.add_argument("--eos_index", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=3)
    parser.add_argument("--config", type=str, default="gfn_game.yaml")
    parser.add_argument(
        "--resume", type=int, default=0, help="whether to resume training"
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="./",
        help="path to resume checkpoint",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=128,
        help="dimensionality of the embedding",
    )
    parser.add_argument("--n_hid", type=int, default=256, help="number of hidden units")
    parser.add_argument("--n_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--n_train_steps",
        type=int,
        default=1500,
        help="number of training steps",
    )
    parser.add_argument(
        "--is_detach_form_TB",
        type=int,
        default=1,
        help="whether to detach hidden states from the computation graph",
    )
    parser.add_argument(
        "--visualize_last",
        type=int,
        default=1,
        help="whether to visualize the last checkpoint during training",
    )
    parser.add_argument(
        "--visualize_every_eval",
        type=int,
        default=1,
        help="whether to visualize every checkpoint during evaluation",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=100,
        help="evaluate every N steps",
    )
    parser.add_argument(
        "--save_when_eval",
        type=int,
        default=1,
        help="whether to save a checkpoint when evaluation score improves",
    )
    parser.add_argument(
        "--save_max",
        type=int,
        default=3,
        help="maximum number of checkpoints to save",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="",
        help="name of the folder to save checkpoints",
    )
    parser.add_argument(
        "--top_k_param_for_vis",
        type=int,
        default=15,
        help="number of instances to visualize in the distribution visualization",
    )

    if return_parser:
        return parser
    else:
        args = parser.parse_args()
        return args
