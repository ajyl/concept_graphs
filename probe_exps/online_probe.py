"""
Probing hidden states.
"""
import os
import json
from tqdm import tqdm
import numpy as np
import einops
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from fancy_einsum import einsum
from celeba_dataset import celeba_dataset
from train import DDPM
from record_utils import record_activations
from cg_constants import ROOT_DIR


def main():
    """ Probe hidden representations """
    run_config = {
        "exp_name": "up1_first_run",
        "is_text": False,
        "nn_model_path": os.path.join(
            ROOT_DIR, "ckpts/nonlinear_contextembed_rerun/epoch_99.pt"
        ),
        "betas": (1e-4, 0.02),
        "n_T": 500,
        "drop_prob": 0.1,
        "n_classes": [2, 2, 2],
        "device": "cuda:0",
        "batch_size": 128,
        "pixel_size": 28,
        "total_samples": 1000,
        "cache_dir": os.path.join(ROOT_DIR, "probe_exps/cached_acts"),
        "config_category_filepath": os.path.join(
            ROOT_DIR, "config_category.json"
        ),
        "concepts": ["000", "001", "010", "100", "011", "101", "110", "111"],
        "activations_dir": os.path.join(
            ROOT_DIR, "probe_exps/cached_acts/first_run"
        ),
    }

    is_text = run_config["is_text"]
    device = run_config["device"]
    n_samples = run_config["batch_size"]
    pixel_size = run_config["pixel_size"]
    total_samples = run_config["total_samples"]
    config_category_filepath = run_config["config_category_filepath"]
    concepts = run_config["concepts"]
    batch_size = run_config["batch_size"]
    acts_dir = run_config["activations_dir"]
    n_feat = 256

    in_channels = 3
    pixel_size = 28

    num_concepts = 3
    num_labels_per_concept = 2
    timesteps = run_config["n_T"]
    probe = (
        torch.randn(
            n_feat * 2,
            int(timesteps / 2),
            num_concepts,
            num_labels_per_concept,
            requires_grad=False,
            device=device,
        )
        / np.sqrt(n_feat * 2)
    )
    lr = 1e-4
    wd = 0.01
    probe.requires_grad = True
    optimizer = torch.optim.AdamW(
        [probe], lr=lr, betas=(0.9, 0.99), weight_decay=wd
    )
    torch.manual_seed(42)

    train_size = int(total_samples * 0.9)
    valid_size = total_samples - train_size
    n_epochs = 5

    inner_batch_size = int(batch_size / len(concepts))
    all_acts = {}
    print("Loading cached activations..")
    for concept_label in concepts:
        print(f"  {concept_label}")
        all_acts[concept_label] = torch.load(
            os.path.join(acts_dir, f"activations_{concept_label}.pt")
        )

    for ep in range(n_epochs):
        print(f"Epoch {ep}")

        for idx in range(0, train_size, inner_batch_size):

            acts = None
            labels = None
            # Construct batch
            for concept_label in concepts:

                # [batch, timestep, d_model]
                # curr_acts_btd = torch.load(
                #    os.path.join(acts_dir, f"activations_{concept_label}.pt")
                # )[idx : idx + batch_size]
                curr_acts_btd = all_acts[concept_label][idx : idx + batch_size]

                curr_batchsize = curr_acts_btd.shape[0]
                curr_label = torch.zeros(
                    (
                        curr_batchsize,
                        int(timesteps / 2),
                        num_concepts,
                        num_labels_per_concept,
                    )
                )
                for c_idx, c_val in enumerate(concept_label):
                    curr_label[:, :, c_idx] = float(c_val)

                if acts is None:
                    acts = curr_acts_btd
                else:
                    acts = torch.cat([acts, curr_acts_btd], dim=0)

                if labels is None:
                    labels = curr_label
                else:
                    labels = torch.cat([labels, curr_label], dim=0)

            # [batch timesteps concepts labels]

            acts = acts[:, int(timesteps / 2):, :]
            probe_out = einsum(
                "batch timesteps d_model, d_model timesteps concepts labels -> batch timesteps concepts labels",
                acts.to(probe.device),
                probe,
            )
            probe_log_probs = probe_out.log_softmax(-1)
            probe_correct_log_probs = (
                einops.reduce(
                    probe_log_probs * labels.to(probe.device),
                    "batch timesteps concepts labels -> timesteps concepts",
                    "mean",
                )
                * num_labels_per_concept
            )
            train_loss = -probe_correct_log_probs.sum(dim=1).mean()
            train_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            print(train_loss)

        torch.save(probe, f"probe_{ep}.pt")


if __name__ == "__main__":
    main()
