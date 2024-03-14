"""
Module Doc String
"""
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from celeba_dataset import celeba_dataset
from train import DDPM
from record_utils import record_activations


def main():
    """ Testing recorder """
    run_config = {
        "is_text": False,
        "nn_model_path": "ckpts/first/epoch_99.pt",
        "betas": (1e-4, 0.02),
        "n_T": 500,
        "drop_prob": 0.1,
        "n_classes": [2, 2, 2],
        "device": "cuda:0",
        "batch_size": 1,
        "pixel_size": 28,
    }

    is_text = run_config["is_text"]
    device = run_config["device"]
    batch_size = run_config["batch_size"]
    pixel_size = run_config["pixel_size"]

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

    with open("config_category.json", "r") as file_p:
        concept_configs = json.load(file_p)["H32-train1"]

    for _config in concept_configs["test"]:
        test_dataset = celeba_dataset(
            is_text,
            transform=transform,
            num_samples=100,
            configs=_config,
            training=False,
            test_size=100,
        )

        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )

        for image, concept_label in test_dataloader:
            image = image.to(device)
            concept_label = (
                concept_label
                if run_config["is_text"]
                else [
                    tmptest_c.to(device)
                    for tmptest_c in concept_label.values()
                ]
            )
            with record_activations(
                ddpm,
                layers_to_hook=["nn_model.to_vec"],
            ) as recorded_acts:
                ddpm(image, concept_label)

            recorded_acts = {
                layer_name: torch.stack(acts, dim=0)
                for layer_name, acts in recorded_acts.items()
            }
            breakpoint()
            print("z")


if __name__ == "__main__":
    main()
