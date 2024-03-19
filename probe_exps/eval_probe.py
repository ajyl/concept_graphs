"""
Module Doc String
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
import plotly.express as px

from celeba_dataset import celeba_dataset
from train import DDPM
from record_utils import record_activations
from probe_exps.linear_probe import _build_batch
from probe_exps.probe_utils import load_acts
from cg_constants import ROOT_DIR


def run_eval():
    """ Driver """
    run_config = {
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
        # "concepts": ["000", "001", "010", "100"],
        # "concepts": ["011", "101", "110", "111"],
        "activations_dir": os.path.join(ROOT_DIR, "probe_exps/cached_acts/"),
        "T_index_every": 10,
        "probe_dir": os.path.join(ROOT_DIR, "probe_exps/probe_ckpts"),
        "probe_layer": "bottleneck",
    }

    is_text = run_config["is_text"]
    device = run_config["device"]
    pixel_size = run_config["pixel_size"]
    total_samples = run_config["total_samples"]
    config_category_filepath = run_config["config_category_filepath"]
    concepts = run_config["concepts"]
    batch_size = run_config["batch_size"]

    probe_dir = run_config["probe_dir"]
    probe_layer = run_config["probe_layer"]
    acts_dir = os.path.join(run_config["activations_dir"], probe_layer)
    timesteps = run_config["n_T"]
    t_index = run_config["T_index_every"]
    n_feat = 256

    num_concepts = 3
    num_labels_per_concept = 2
    probe_dir = os.path.join(probe_dir, probe_layer)

    probe = torch.load(os.path.join(probe_dir, "epoch_19.pt"))

    inner_batch_size = int(batch_size / len(concepts))
    train_acts, valid_acts = load_acts(acts_dir, batch_size, concepts, t_index)

    val_losses = []
    val_accuracies = []
    all_results = None
    valid_size = valid_acts["000"].shape[0]
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

        if all_results is None:
            all_results = val_results
        else:
            all_results = torch.cat([all_results, val_results], dim=0)

        acc_per_timestep = (
            val_results.sum(dim=-1) / val_results.shape[-1]
        ).mean(dim=0)
        valid_acc = (val_results.sum() / val_results.numel()).item()
        val_accuracies.append(valid_acc * curr_valid_batch_size)

    fig = px.line(all_results.float().mean(dim=0).cpu())
    fig.write_image("zxcv_20.png")
    print(f"Val loss: {sum(val_losses) / (valid_size * 8)}")
    print(f"Val acc: {sum(val_accuracies) / (valid_size * 8)}")
    print(acc_per_timestep)
    breakpoint()
    print("z")


def plot():

    run_config = {
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
    }
    probe = torch.load("probe_4.pt")


if __name__ == "__main__":
    # plot()
    run_eval()
