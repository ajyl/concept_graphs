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
from probe_exps.probe_utils import load_acts
from cg_constants import ROOT_DIR


def _build_batch(
    concepts,
    acts,
    idx,
    batch_size,
    timesteps,
    num_concepts,
    num_labels_per_concept,
):
    """
    Build batch of activations and labels.
    """
    batch_acts = None
    labels = None
    for concept_label in concepts:
        curr_acts_btd = acts[concept_label][idx : idx + batch_size]
        curr_label = torch.zeros(
            (
                curr_acts_btd.shape[0],
                curr_acts_btd.shape[1],
                num_concepts,
                num_labels_per_concept,
            )
        )
        for c_idx, c_val in enumerate(concept_label):
            curr_label[:, :, c_idx, int(c_val)] = 1

        if batch_acts is None:
            batch_acts = curr_acts_btd
        else:
            batch_acts = torch.cat([batch_acts, curr_acts_btd], dim=0)

        if labels is None:
            labels = curr_label
        else:
            labels = torch.cat([labels, curr_label], dim=0)
    return batch_acts, labels


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
        "total_samples": 5056,
        "cache_dir": os.path.join(ROOT_DIR, "probe_exps/cached_acts"),
        "config_category_filepath": os.path.join(
            ROOT_DIR, "config_category.json"
        ),
        "concepts": ["000", "001", "010", "100", "011", "101", "110", "111"],
        "activations_dir": os.path.join(ROOT_DIR, "probe_exps/cached_acts/"),
        "valid_every": 200,
        "probe_layer": "bottleneck",
        "probe_dir": os.path.join(ROOT_DIR, "probe_exps/probe_ckpts"),
        "T_index_every": 10,
    }

    is_text = run_config["is_text"]
    device = run_config["device"]
    n_samples = run_config["batch_size"]
    pixel_size = run_config["pixel_size"]
    total_samples = run_config["total_samples"]
    config_category_filepath = run_config["config_category_filepath"]
    concepts = run_config["concepts"]
    batch_size = run_config["batch_size"]
    acts_dir = os.path.join(
        run_config["activations_dir"], run_config["probe_layer"]
    )

    output_dir = os.path.join(
        run_config["probe_dir"], run_config["probe_layer"]
    )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    valid_every = run_config["valid_every"]
    t_index = run_config["T_index_every"]
    n_feat = 256

    in_channels = 3
    pixel_size = 28

    num_concepts = 3
    num_labels_per_concept = 2
    timesteps = run_config["n_T"]
    probe = (
        torch.randn(
            n_feat * 2,
            int(timesteps / t_index),
            num_concepts,
            num_labels_per_concept,
            requires_grad=False,
            device=device,
        )
        / np.sqrt(n_feat * 2)
    )
    lr = 1e-3
    wd = 0.01
    probe.requires_grad = True
    optimizer = torch.optim.AdamW(
        [probe], lr=lr, betas=(0.9, 0.99), weight_decay=wd
    )
    torch.manual_seed(42)

    n_epochs = 20
    train_acts, valid_acts = load_acts(acts_dir, batch_size, concepts, t_index)

    for ep in range(n_epochs):
        print(f"Epoch {ep}")

        for idx in range(0, train_size, inner_batch_size):

            acts, labels = _build_batch(
                concepts,
                train_acts,
                idx,
                inner_batch_size,
                timesteps,
                num_concepts,
                num_labels_per_concept,
            )
            # [batch timesteps concepts labels]
            probe_out = einsum(
                "batch timesteps d_model, d_model timesteps concepts options -> batch timesteps concepts options",
                acts.to(probe.device),
                probe,
            )
            probe_log_probs = probe_out.log_softmax(-1)
            probe_correct_log_probs = (
                einops.reduce(
                    probe_log_probs * labels.to(probe.device),
                    "batch timesteps concepts options -> timesteps concepts",
                    "mean",
                )
                * num_labels_per_concept
            )
            train_loss = -probe_correct_log_probs.mean(0).sum()
            train_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if idx % valid_every == 0:

                val_losses = []
                val_accuracies = []
                for valid_idx in range(0, valid_size, inner_batch_size):
                    acts_valid, labels_valid = _build_batch(
                        concepts,
                        valid_acts,
                        valid_idx,
                        inner_batch_size,
                        timesteps,
                        num_concepts,
                        num_labels_per_concept,
                    )
                    curr_valid_batch_size = acts_valid.shape[0]
                    # [batch timesteps concepts labels]
                    probe_out = einsum(
                        "batch timesteps d_model, d_model timesteps concepts options -> batch timesteps concepts options",
                        acts_valid.to(probe.device),
                        probe,
                    )
                    probe_log_probs = probe_out.log_softmax(-1)
                    probe_correct_log_probs = (
                        einops.reduce(
                            probe_log_probs * labels_valid.to(probe.device),
                            "batch timesteps concepts options -> timesteps concepts",
                            "mean",
                        )
                        * num_labels_per_concept
                    )
                    valid_loss = -probe_correct_log_probs.mean(0).sum().item()
                    val_losses.append(valid_loss * curr_valid_batch_size)

                    # [batch, timesteps, concepts]
                    val_preds = probe_log_probs.argmax(-1)
                    val_gold = labels_valid.argmax(-1).to(val_preds.device)
                    val_results = val_preds == val_gold

                    acc_per_timestep = (
                        val_results.sum(dim=-1) / val_results.shape[-1]
                    ).mean(dim=0)
                    valid_acc = (
                        val_results.sum() / val_results.numel()
                    ).item()
                    val_accuracies.append(valid_acc * curr_valid_batch_size)

                print(f"Train loss: {train_loss.item()}")
                print(f"Val loss: {sum(val_losses) / (valid_size * 8)}")
                print(f"Val acc: {sum(val_accuracies) / (valid_size * 8)}")
                # print(acc_per_timestep)

        torch.save(
            probe,
            os.path.join(
                output_dir,
                f"epoch_{ep}.pt",
            ),
        )


if __name__ == "__main__":
    main()
