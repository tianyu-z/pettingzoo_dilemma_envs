import torch


def reward_function(x, a, lambda_=2, beta=1, R_0=1e-6):
    q = x // a
    return lambda_ ** -(beta * min(x - a * q, a * (q + 1) - x)) + R_0


def reward_function11(state, a, lambda_=2, beta=1):
    """state : (bs, slen)"""
    r = torch.Tensor(
        [
            reward_function(s, a, lambda_, beta)
            for s in [float("".join([str(s_i) for s_i in s])) for s in state.tolist()]
        ]
    )
    return r


def reward_function22(state, a, lambda_=2, beta=1):
    """state : (bs, slen)"""
    r = torch.Tensor([reward_function(s, a, lambda_, beta) for s in state])
    return r


reward_coef = 81

vocab = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ".",
    ",",
]  # eos (.) and pad (,) tokens
pad_index = 10
eos_index = 10 + 1
bos_index = eos_index  # we will use <EOS> as <BOS> (start token) everywhere
vocab_size = len(vocab)
max_length = 3
lambda_ = 2
beta = 1
xs = torch.arange(0, int("".join(["9"] * max_length)) + 1)
all_rewards = reward_function22(xs, reward_coef, lambda_, beta)
true_dist = all_rewards.softmax(0).cpu().numpy()
if __name__ == "__main__":
    print("all_rewards", all_rewards)
    tk = torch.tensor(all_rewards).topk(k=20)
    modes = tk.indices
    print("modes", modes)
    print("tk.values", tk.values)
