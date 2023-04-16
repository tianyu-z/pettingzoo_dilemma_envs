from collections import OrderedDict

action_idx_mapping = {
    "C": 0,
    "D": 1,
}
idx_action_mapping = {
    0: "C",
    1: "D",
}

state_idx_mapping = {
    "CC": 0,
    "CD": 1,
    "DC": 2,
    "DD": 3,
}
idx_state_mapping = {
    0: "CC",
    1: "CD",
    2: "DC",
    3: "DD",
}


def get_action_idx(action):
    return action_idx_mapping[action]


def get_idx_action(idx):
    return idx_action_mapping[idx]


def get_state_idx(state):
    return state_idx_mapping[state]


def get_idx_state(idx):
    return idx_state_mapping[idx]
