import torch
import tqdm
import torch.nn.functional as F

import numpy as np

from agents.GFN_WIP.transformer import TransformerModel, make_mlp
from reward import (
    batch_reward,
    vocab_size,
    pad_index,
    eos_index,
    bos_index,
    max_length,
    true_dist,
    true_dist_dict,
    list2string,
    string2list,
    xs_string,
)

from collections import Counter
from games import (
    Game,
    Prisoners_Dilemma,
    Samaritans_Dilemma,
    Stag_Hunt,
    Chicken,
)

from utils import (
    create_gif,
    create_gif_by_dicts,
    get_top_k,
    filter_dict_by_keys,
    normalize_dict_values,
)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


params = AttrDict(
    {
        "n_words": vocab_size,
        "pad_index": pad_index,
        "eos_index": eos_index,
        "bos_index": bos_index,
        "emb_dim": 128,
    }
)
game = Prisoners_Dilemma()
# game.payoff[(2, 2)] = (0, 0)  # None, None is a tie
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logZ = torch.zeros((1,)).to(device)

n_hid = 256
n_layers = 2

mlp = make_mlp([params.emb_dim] + [n_hid] * n_layers + [params.n_words]).to(device)
model = TransformerModel(params, mlp).to(device)
P_B = 1  # DAG & sequence generation => tree


optim = torch.optim.Adam(
    [{"params": model.parameters(), "lr": 0.001}, {"params": [logZ], "lr": 0.01}]
)
logZ.requires_grad_()

losses_TB = []
zs_TB = []
rewards_TB = []
all_visited_TB = []
first_visit_TB = {}
# first_visit_TB = -1 * np.ones_like(true_dist)
l1log_TB = []

batch_size = 256
max_len = max_length + 0  # +1 for <BOS>

# n_train_steps = 1000
n_train_steps = 10000

emp_dist_ts = []
emp_dist_dict_ts = []  # for visualization
losses = []
top_k_param_for_vis = 15
for it in tqdm.trange(n_train_steps):
    generated = torch.LongTensor(batch_size, max_len)  # upcoming output
    generated.fill_(params.pad_index)  # fill upcoming ouput with <PAD>
    generated[:, 0].fill_(params.bos_index)  # <BOS> (start token), initial state

    # Length of already generated sequences : 1 because of <BOS>
    gen_len = torch.LongTensor(
        batch_size,
    ).fill_(
        1
    )  # (batch_size,)
    # 1 (True) if the generation of the sequence is not yet finished, 0 (False) otherwise
    unfinished_sents = gen_len.clone().fill_(1)  # (batch_size,)
    # Length of already generated sequences : 1 because of <BOS>
    cur_len = 1

    Z = logZ.exp()

    is_detach_form_TB = True
    if is_detach_form_TB:
        # detached form  of TB
        ll_diff = torch.zeros((batch_size,)).to(device)
        ll_diff += logZ
    else:
        # non-detached form of TB ojective, where we multiply everything before doing the logarithm
        in_probs = torch.ones(batch_size, dtype=torch.float, requires_grad=True).to(
            device
        )

    while cur_len < max_len:
        state = generated[:, :cur_len] + 0  # (bs, cur_len)
        tensor = model(
            state.to(device), lengths=gen_len.to(device)
        )  # (bs, cur_len, vocab_size)
        # scores = tensor[:,0] # (bs, vocab_size) : use last word for prediction
        scores = tensor.sum(dim=1)  # (bs, vocab_size)
        scores[:, pad_index] = -1e8  # we don't want to generate pad_token
        scores[
            :, eos_index
        ] = (
            -1e8
        )  # if we don't want to generate eos_token : don't allow generation of sentences with differents lengths
        scores = scores.log_softmax(dim=1)
        sample_temperature = 1
        probs = F.softmax(scores / sample_temperature, dim=1)  # softmax of log softmax?
        next_words = torch.multinomial(probs, 1).squeeze(1)

        # update generations / lengths / finished sentences / current length
        generated[
            :, cur_len
        ] = next_words.cpu() * unfinished_sents + params.pad_index * (
            1 - unfinished_sents
        )  # if the sentence is finished, we don't update it, just sent it to <PAD>
        gen_len.add_(
            unfinished_sents
        )  # add 1 to the length of the unfinished sentences: 1(init)+1(unfinished_sents)+1+1...+1+0(finished)
        unfinished_sents.mul_(
            next_words.cpu()
            .ne(params.eos_index)
            .long()  # ne: A boolean tensor that is True where input is not equal to other and False elsewhere
        )  # as soon as we generate <EOS>, set unfinished_sents to 0
        cur_len = cur_len + 1

        # loss
        if is_detach_form_TB:
            # sample_in_probs = probs.gather(1, next_words.unsqueeze(-1)).squeeze(1)
            # sample_in_probs[unfinished_sents == 0] = 1.
            # ll_diff += sample_in_probs.log()

            ll_diff += scores.gather(1, next_words.unsqueeze(-1)).squeeze(1)
        else:
            sample_in_probs = probs.gather(1, next_words.unsqueeze(-1)).squeeze(1)
            sample_in_probs[unfinished_sents == 0] = 1.0
            in_probs = in_probs * sample_in_probs

        # stop when there is a <EOS> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

    generated = generated.apply_(
        lambda index: 5 if index == pad_index or index == eos_index else index
    )
    # R = reward_function(generated, reward_coef, lambda_, beta).to(device)
    generated = generated.tolist()
    R = torch.tensor(
        batch_reward(game, generated, is_sum_agent_rewards=True, only_last=True)
    ).to(device)

    optim.zero_grad()
    if is_detach_form_TB:
        ll_diff -= R.log()
        loss = (ll_diff**2).sum() / batch_size
    else:
        loss = ((Z * in_probs / R).log() ** 2).sum() / batch_size

    loss.backward()
    optim.step()

    losses_TB.append(loss.item())
    zs_TB.append(Z.item())
    rewards_TB.append(R.mean().cpu())
    all_visited_TB.extend(generated)
    # for state_idx in range(len(all_visited_TB)):
    #     if first_visit_TB[state_idx] < 0:
    #         first_visit_TB[state_idx] = it
    for state in all_visited_TB:  #
        state = list2string(state)
        if state not in first_visit_TB:
            first_visit_TB[int(state)] = it
        elif first_visit_TB[state] < 0:
            first_visit_TB[int(state)] = it

    if it % 100 == 0:
        print(
            "\nloss =",
            np.array(losses_TB[-100:]).mean(),
            "Z =",
            Z.item(),
            "R =",
            np.array(rewards_TB[-100:]).mean(),
        )
        all_visited_TB_string = [list2string(x) for x in all_visited_TB]
        Counter_TB = Counter(all_visited_TB_string)
        Counter_TB = normalize_dict_values(Counter_TB)
        Counter_TB_ = []
        for x in xs_string:
            if x not in Counter_TB:
                Counter_TB_.append(0)
            else:
                Counter_TB_.append(Counter_TB[x])
        emp_dist = np.array(Counter_TB_).astype(float)

        # emp_dist = np.bincount(
        #     all_visited_TB_string[-len(true_dist) :], minlength=len(true_dist)
        # ).astype(float)
        emp_dist /= emp_dist.sum()
        l1 = np.abs(true_dist - emp_dist).mean()
        print("L1 =", l1)
        l1log_TB.append((len(all_visited_TB), l1))
        # print("gen", generated[-100:])
        emp_dist_ts.append(emp_dist)
        emp_dist_dict_ts.append(Counter_TB)
        losses.append(l1)
a = emp_dist_ts[0][-1000:]
print(a)
losses = ["loss: " + str(losses[i]) for i in range(len(losses))]
top_k_true_dist_dict, top_k_keys = get_top_k(
    true_dist_dict, top_k_param_for_vis, "true"
)
top_k_emp_dist_dict_ts = [
    filter_dict_by_keys(top_k_keys, emp_dist_dict_ts[i])
    for i in range(len(emp_dist_dict_ts))
]

create_gif_by_dicts(
    emp_ts_dict=top_k_emp_dist_dict_ts,
    true_ts_dict=top_k_true_dist_dict,
    title=losses,
    filename="1000.gif",
)
