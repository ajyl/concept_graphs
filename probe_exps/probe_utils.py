import os
import torch


def load_acts(acts_dir, batch_size, concepts, t_index):
    inner_batch_size = int(batch_size / len(concepts))
    all_acts = {}
    print("Loading cached activations..")

    for _act_dir in os.listdir(acts_dir):
        _act_dir = os.path.join(acts_dir, _act_dir)
        print(f"  {_act_dir}")

        for concept_label in concepts:
            print(f"  {concept_label}")
            acts = torch.load(
                os.path.join(_act_dir, f"activations_{concept_label}.pt")
            )
            acts = acts[:, 1::t_index, :]
            if concept_label not in all_acts:
                all_acts[concept_label] = acts
            else:
                all_acts[concept_label] = torch.cat(
                    [all_acts[concept_label], acts], dim=0
                )

    total_samples = all_acts["000"].shape[0]
    train_size = int(total_samples * 0.95)
    valid_size = total_samples - train_size
    train_acts = {}
    valid_acts = {}
    for concept_label in concepts:
        train_acts[concept_label] = all_acts[concept_label][:train_size]
        valid_acts[concept_label] = all_acts[concept_label][train_size:]
    return train_acts, valid_acts
