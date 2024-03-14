"""
Probing hidden states.
"""
import os
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
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
        "batch_size": 32,
        "pixel_size": 28,
        "total_samples": 1000,
        "cache_dir": os.path.join(ROOT_DIR, "probe_exps/cached_acts"),
        "config_category_filepath": os.path.join(
            ROOT_DIR, "config_category.json"
        ),
        "concepts": ["000", "001", "010", "100", "011", "101", "110", "111"],
    }

    output_dir = os.path.join(run_config["cache_dir"], run_config["exp_name"])
    if os.path.isdir(output_dir):
        raise RuntimeError(f"Directory {output_dir} exists.")

    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "run_config.json"), "w") as file_p:
        json.dump(run_config, file_p, indent=4)

    is_text = run_config["is_text"]
    device = run_config["device"]
    n_samples = run_config["batch_size"]
    pixel_size = run_config["pixel_size"]
    total_samples = run_config["total_samples"]
    config_category_filepath = run_config["config_category_filepath"]
    concepts = run_config["concepts"]
    n_feat = 256

    nn_model = torch.load(run_config["nn_model_path"]).to(device)
    ddpm = DDPM(
        is_text,
        nn_model,
        run_config["betas"],
        run_config["n_T"],
        device,
        run_config["drop_prob"],
        run_config["n_classes"],
    )

    transform = transforms.Compose(
        [
            transforms.Resize((pixel_size, pixel_size)),
            transforms.ToTensor(),
        ]
    )

    with open(config_category_filepath, "r") as file_p:
        concept_configs = json.load(file_p)["H32-train1"]

    in_channels = 3
    pixel_size = 28

    # num_concepts = 3
    # num_labels_per_concept = 2
    # probe = (
    #    torch.randn(
    #        n_feat * 2,
    #        num_concepts,
    #        num_labels_per_concept,
    #        requires_grad=False,
    #        device=device,
    #    )
    #    / np.sqrt(n_feat * 2)
    # )

    for _config in tqdm(concepts):
        acts = None
        concept_label = {
            0: torch.tensor([int(_config[0])]),
            1: torch.tensor([int(_config[1])]),
            2: torch.tensor([int(_config[2])]),
        }
        remaining = total_samples
        while remaining > 0:
            print(f"Number remaining: {remaining}")
            with torch.no_grad():
                with record_activations(
                    ddpm,
                    layers_to_hook=["nn_model.up1"],
                ) as recorded_acts:
                    ddpm.sample(
                        n_samples,
                        concept_label,
                        (in_channels, pixel_size, pixel_size),
                        device,
                        guide_w=0.0,
                    )

            # to_vec: [batch, timesteps, d_model]
            # curr_acts = (
            #    torch.stack(recorded_acts["nn_model.to_vec"], dim=1)
            #    .squeeze(-1)
            #    .squeeze(-1)
            #    .cpu()
            # )

            # [batch, timesteps, d_model, pixel/2, pixel/2]
            curr_acts = torch.stack(recorded_acts["nn_model.up1"], dim=1).cpu()
            if acts is None:
                acts = curr_acts
            else:
                acts = torch.cat([acts, curr_acts], dim=0)
            remaining -= n_samples

        torch.save(acts, os.path.join(output_dir, f"activations_{_config}.pt"))


if __name__ == "__main__":
    main()
