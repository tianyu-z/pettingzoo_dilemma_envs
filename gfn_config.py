import argparse
import yaml


def merge_config(args, config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    for key in config:
        setattr(args, key, config[key])
    return args


def merge_args(args1, args2, default_args=None):
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
    if args.config is not None:
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
    parser.add_argument("--bos_index", type=int, default=4)
    parser.add_argument("--eos_index", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=3)
    parser.add_argument("--config", type=str, default="gfn_game.yaml")

    if return_parser:
        return parser
    else:
        args = parser.parse_args()
        return args
