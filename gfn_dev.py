from gfn_config import get_merged_args
import torch
import tqdm
import torch.nn.functional as F
import numpy as np
from agents.GFN_example_from_tutorial.transformer import TransformerModel, make_mlp

from reward import batch_reward, get_true_dist, list2string
import os
from collections import Counter
from games import (
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
    save_pt,
    load_pt,
    get_hex_time,
    delete_oldest_files,
)


def main():
    args = get_merged_args()
    if args.game_type == "PD":
        game = Prisoners_Dilemma()
    elif args.game_type == "SD":
        game = Samaritans_Dilemma()
    elif args.game_type == "SH":
        game = Stag_Hunt()
    elif args.game_type == "CH":
        game = Chicken()

    setattr(args, "n_words", len(args.vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logZ = torch.zeros((1,)).to(device)
    mlp = make_mlp([args.emb_dim] + [args.n_hid] * args.n_layers + [args.n_words]).to(
        device
    )

    model = TransformerModel(args, mlp).to(device)
    optim = torch.optim.Adam(
        [{"params": model.parameters(), "lr": 0.001}, {"params": [logZ], "lr": 0.01}]
    )

    if args.resume:
        checkpoint_dict = load_pt(args.resume_path)
        model.load_state_dict(checkpoint_dict["GFN_model_state_dict"])
        optim.load_state_dict(checkpoint_dict["GFN_optimizer_state_dict"])
        logZ += checkpoint_dict["logZ"]
        print("Loaded model from checkpoint")
    if args.save_when_eval:
        if args.save_name is None or args.save_name == "":
            saving_folder_name = get_hex_time()
        else:
            saving_folder_name = args.save_name
        if not os.path.exists(f"./checkpoints/{saving_folder_name}"):
            os.makedirs(f"./checkpoints/{saving_folder_name}")
    logZ.requires_grad_()
    true_dist, true_dist_dict, xs_string = get_true_dist(args)
    # for training
    losses_TB = []
    zs_TB = []
    rewards_TB = []
    all_visited_TB = []
    first_visit_TB = {}
    # first_visit_TB = -1 * np.ones_like(true_dist)
    l1log_TB = []

    # for evaluation
    emp_dist_ts = []
    emp_dist_dict_ts = []  # for visualization
    eval_metrics = []

    for it in tqdm.trange(args.n_train_steps):
        # train one step
        loss_item, Z_item, R_mean, generated, model, optim = train_one_step(
            args, logZ, model, optim, game, device, true_dist, true_dist_dict, xs_string
        )
        # logging stuff
        losses_TB.append(loss_item)
        zs_TB.append(Z_item)
        rewards_TB.append(R_mean)
        all_visited_TB.extend(generated)
        for state in all_visited_TB:  #
            state = list2string(state)
            if state not in first_visit_TB:
                first_visit_TB[int(state)] = it
            elif first_visit_TB[state] < 0:
                first_visit_TB[int(state)] = it
        # start evaluation
        if it % args.eval_every == 0:
            print(
                "\nloss =",
                np.array(losses_TB[-100:]).mean(),
                "Z =",
                Z_item,
                "R =",
                np.array(rewards_TB[-100:]).mean(),
            )
            l1log, emp_dist, Counter_TB, l1 = evaluate_TB(
                all_visited_TB, true_dist, true_dist_dict, xs_string
            )
            l1log_TB.append(l1log)
            emp_dist_ts.append(emp_dist)
            emp_dist_dict_ts.append(Counter_TB)
            eval_metrics.append(l1)

            if args.save_when_eval:
                save_pt(
                    {
                        "eval_metrics": eval_metrics,
                        "emp_dist_dict_ts": emp_dist_dict_ts,
                        "true_dist_dict": true_dist_dict,
                        "log_Z_item": np.log(Z_item),
                        "GFN_model_state_dict": model.state_dict(),
                        "GFN_optimizer_state_dict": optim.state_dict(),
                    },
                    f"./checkpoints/{saving_folder_name}/checkpoints_{it}.pt",
                )
                # only save the lastest ones (number of saved files = args.save_max)
                assert isinstance(args.save_max, int), "save_max must be an integer"
                if args.save_max > 0:
                    delete_oldest_files(
                        f"./checkpoints/{saving_folder_name}", args.save_max
                    )
            if args.visualize_every_eval:
                visualize_evaluation(
                    args, eval_metrics, emp_dist_dict_ts, true_dist_dict
                )
    if args.visualize_last and (not args.visualize_every_eval):
        visualize_evaluation(args, eval_metrics, emp_dist_dict_ts, true_dist_dict)


def train_one_step(
    args, logZ, model, optim, game, device, true_dist, true_dist_dict, xs_string
):
    generated = torch.LongTensor(args.batch_size, args.max_len)  # upcoming output
    generated.fill_(args.pad_index)  # fill upcoming ouput with <PAD>
    generated[:, 0].fill_(args.bos_index)  # <BOS> (start token), initial state

    # Length of already generated sequences : 1 because of <BOS>
    gen_len = torch.LongTensor(
        args.batch_size,
    ).fill_(
        1
    )  # (batch_size,)
    # 1 (True) if the generation of the sequence is not yet finished, 0 (False) otherwise
    unfinished_sents = gen_len.clone().fill_(1)  # (batch_size,)
    # Length of already generated sequences : 1 because of <BOS>
    cur_len = 1

    Z = logZ.exp()

    if args.is_detach_form_TB:
        # detached form  of TB
        ll_diff = torch.zeros((args.batch_size,)).to(device)
        ll_diff += logZ
    else:
        # non-detached form of TB ojective, where we multiply everything before doing the logarithm
        in_probs = torch.ones(
            args.batch_size, dtype=torch.float, requires_grad=True
        ).to(device)
    while cur_len < args.max_len:
        state = generated[:, :cur_len] + 0  # (bs, cur_len)
        tensor = model(
            state.to(device), lengths=gen_len.to(device)
        )  # (bs, cur_len, vocab_size)
        # scores = tensor[:,0] # (bs, vocab_size) : use last word for prediction
        scores = tensor.sum(dim=1)  # (bs, vocab_size)
        scores[:, args.pad_index] = -1e8  # we don't want to generate pad_token
        scores[
            :, args.eos_index
        ] = (
            -1e8
        )  # if we don't want to generate eos_token : don't allow generation of sentences with differents lengths
        scores = scores.log_softmax(dim=1)
        sample_temperature = 1
        probs = F.softmax(scores / sample_temperature, dim=1)  # softmax of log softmax?
        next_words = torch.multinomial(probs, 1).squeeze(1)

        # update generations / lengths / finished sentences / current length
        generated[:, cur_len] = next_words.cpu() * unfinished_sents + args.pad_index * (
            1 - unfinished_sents
        )  # if the sentence is finished, we don't update it, just sent it to <PAD>
        gen_len.add_(
            unfinished_sents
        )  # add 1 to the length of the unfinished sentences: 1(init)+1(unfinished_sents)+1+1...+1+0(finished)
        unfinished_sents.mul_(
            next_words.cpu()
            .ne(args.eos_index)
            .long()  # ne: A boolean tensor that is True where input is not equal to other and False elsewhere
        )  # as soon as we generate <EOS>, set unfinished_sents to 0
        cur_len = cur_len + 1

        # loss
        if args.is_detach_form_TB:
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
        lambda index: 5 if index == args.pad_index or index == args.eos_index else index
    )
    # R = reward_function(generated, reward_coef, lambda_, beta).to(device)
    generated = generated.tolist()
    R = torch.tensor(
        batch_reward(game, generated, is_sum_agent_rewards=True, only_last=True)
    ).to(device)

    optim.zero_grad()
    if args.is_detach_form_TB:
        ll_diff -= R
        loss = (ll_diff**2).sum() / args.batch_size
    else:
        loss = ((Z * in_probs / R).log() ** 2).sum() / args.batch_size

    loss.backward()
    optim.step()
    return loss.item(), Z.item(), R.mean().cpu(), generated, model, optim


def evaluate_TB(all_visited_TB, true_dist, true_dist_dict, xs_string):
    # for evaluation
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
    print("emp_dist: ", emp_dist)
    return (len(all_visited_TB), l1), emp_dist, Counter_TB, l1

    # visualization of the evaluation


def visualize_evaluation(args, eval_metrics, emp_dist_dict_ts, true_dist_dict):
    eval_metrics_ = [
        "eval_metrics: " + str(eval_metrics[i]) for i in range(len(eval_metrics))
    ]
    top_k_true_dist_dict, top_k_keys = get_top_k(
        true_dist_dict, args.top_k_param_for_vis, "true"
    )
    top_k_emp_dist_dict_ts = [
        filter_dict_by_keys(top_k_keys, emp_dist_dict_ts[i])
        for i in range(len(emp_dist_dict_ts))
    ]

    create_gif_by_dicts(
        emp_ts_dict=top_k_emp_dist_dict_ts,
        true_ts_dict=top_k_true_dist_dict,
        title=eval_metrics_,
        filename="1000.gif",
    )


if __name__ == "__main__":
    main()
