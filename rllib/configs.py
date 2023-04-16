import numpy as np

configs = {
    "default": {"experiment_name": "default", "env_config": {}, "num_iter": 1000},
    "low_benefit": {
        "experiment_name": "low_benefit",
        "env_config": {
            "reward_matrix": [
                [4.9, 4.9],
                [0, 5],
                [5, 0],
                [0.1, 0.1],
            ],
        },
        "num_iter": 1000,
    },
    "low_benefit_nonzero": {
        "experiment_name": "low_benefit_nonzero",
        "env_config": {
            "reward_matrix": [
                [4.5, 4.5],
                [3, 5],
                [5, 3],
                [3.5, 3.5],
            ],
        },
        "num_iter": 1000,
    },
    "vs_tit_for_tat": {
        "experiment_name": "vs_tit_for_tat",
        "env_config": {},
        "second_agent": "tit_for_tat",
        "num_iter": 1000,
    },
    "vs_pavlov": {
        "experiment_name": "vs_pavlov",
        "env_config": {},
        "second_agent": "pavlov",
        "num_iter": 1000,
    },
}
