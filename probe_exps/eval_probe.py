"""
Module Doc String
"""
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import einops
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from fancy_einsum import einsum
import plotly.express as px

from train import DDPM
from record_utils import record_activations
from probe_exps.linear_probe import _build_batch
from probe_exps.probe_utils import load_acts
from cg_constants import ROOT_DIR


def run_eval():
    """ Driver """
    run_config = {
        "is_text": False,
        "n_T": 500,
        "drop_prob": 0.1,
        "batch_size": 128,
        "concepts": ["000", "001", "010", "100", "011", "101", "110", "111"],
        "activations_dir": os.path.join(ROOT_DIR, "probe_exps/cached_acts/"),
        "T_index_every": 10,
        "probe_dir": os.path.join(ROOT_DIR, "probe_exps/probe_ckpts"),
        "probe_layer": "bottleneck",
        "dataset": "shapes",
    }
    concepts = run_config["concepts"]
    batch_size = run_config["batch_size"]

    probe_layer = run_config["probe_layer"]
    dataset = run_config["dataset"]
    probe_dir = os.path.join(
        run_config["probe_dir"], f"{dataset}/{probe_layer}"
    )
    acts_dir = os.path.join(
        run_config["activations_dir"],
        f"{dataset}/{probe_layer}/",
    )
    timesteps = run_config["n_T"]
    t_index = run_config["T_index_every"]
    n_feat = 256

    num_concepts = 3
    num_labels_per_concept = 2

    probe = torch.load(os.path.join(probe_dir, "epoch_19.pt"))

    inner_batch_size = int(batch_size / len(concepts))
    _, valid_acts = load_acts(acts_dir, batch_size, concepts, t_index)

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
    fig.write_image("zxcv_21.png")
    print(f"Val loss: {sum(val_losses) / (valid_size * 8)}")
    print(f"Val acc: {sum(val_accuracies) / (valid_size * 8)}")
    print(acc_per_timestep)
    breakpoint()
    print("z")


def cos_sim():
    """ Driver """
    run_config = {
        "concepts": ["000", "001", "010", "100", "011", "101", "110", "111"],
        "probe_dir": os.path.join(ROOT_DIR, "probe_exps/probe_ckpts"),
        "probe_layer": "bottleneck",
        "dataset": "shapes",
    }
    probe_layer = run_config["probe_layer"]
    dataset = run_config["dataset"]
    probe_dir = os.path.join(
        run_config["probe_dir"], f"{dataset}/{probe_layer}"
    )
    probe = torch.load(os.path.join(probe_dir, "epoch_19.pt"))
    cos_sim = F.cosine_similarity
    print("z")

    probes = probe[:, -1, :, 0]
    probes_norm = probes / probes.norm(dim=0)
    cos_scores = torch.mm(probes_norm.transpose(0, 1), probes_norm)

    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(
        cos_scores.detach().cpu(),
        cmap="magma_r",
        vmax=1.0,
        vmin=0,
        annot=True,
        fmt=".2f",
    )
    fig.savefig("heatmap_shapes.png")

    breakpoint()

if __name__ == "__main__":
    #run_eval()
    cos_sim()
