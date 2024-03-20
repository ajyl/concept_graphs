"""
Probing hidden states.
"""
import os
import json
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from cg_datasets.shapes_dataset import shapes_dataset
from train import DDPM
from record_utils import record_activations
from cg_constants import ROOT_DIR


def main():
    """ Probe hidden representations """
    run_config = {
        "dataset": "shapes",
        "exp_name": "debug",
        "is_text": False,
        "nn_model_path": os.path.join(ROOT_DIR, "ckpts/shapes_v1/epoch_99.pt"),
        "betas": (1e-4, 0.02),
        "n_T": 500,
        "drop_prob": 0.1,
        "n_classes": [2, 3, 1],
        "device": "cuda:0",
        "pixel_size": 28,
        "total_samples": 5000,
        "n_samples_per_clabel": 32,
        "cache_dir": os.path.join(ROOT_DIR, "probe_exps/cached_acts"),
        "config_category_filepath": os.path.join(
            ROOT_DIR, "config_category.json"
        ),
        "concepts": ["000", "001", "010", "100", "011", "101", "110", "111"],
    }

    output_dir = os.path.join(
        run_config["cache_dir"],
        f"{run_config['dataset']}/{run_config['exp_name']}",
    )
    if os.path.isdir(output_dir):
        raise RuntimeError(f"Directory {output_dir} exists.")

    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "run_config.json"), "w") as file_p:
        json.dump(run_config, file_p, indent=4)

    is_text = run_config["is_text"]
    device = run_config["device"]
    n_samples_per_clabel = run_config["n_samples_per_clabel"]
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

    in_channels = 4
    pixel_size = 28
    tf = transforms.Compose(
        [
            transforms.Resize((pixel_size, pixel_size)),
            transforms.ToTensor(),
        ]
    )

    for _config in tqdm(concepts):
        dataset = shapes_dataset(
            False,
            transform=tf,
            num_samples=total_samples,
            configs=_config,
            training=False,
            test_size=1.6,
        )
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1
        )
        acts = None
        remaining = total_samples
        with torch.no_grad():
            while remaining > 0:
                print(f"Number remaining: {remaining}")
                _, _c_label = next(iter(dataloader))

                with record_activations(
                    ddpm,
                    layers_to_hook=["nn_model.to_vec"],
                ) as recorded_acts:

                    ddpm.sample(
                        n_samples_per_clabel,
                        _c_label,
                        (in_channels, pixel_size, pixel_size),
                        device,
                        guide_w=0.0,
                    )

                # to_vec: [batch, timesteps, d_model]
                curr_acts = (
                    torch.stack(recorded_acts["nn_model.to_vec"], dim=1)
                    .squeeze(-1)
                    .squeeze(-1)
                    .cpu()
                )

                # [batch, timesteps, d_model, pixel/2, pixel/2]
                if acts is None:
                    acts = curr_acts
                else:
                    acts = torch.cat([acts, curr_acts], dim=0)

                remaining -= n_samples_per_clabel

        torch.save(acts, os.path.join(output_dir, f"activations_{_config}.pt"))


if __name__ == "__main__":
    main()
