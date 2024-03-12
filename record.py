"""
Utility function to record activations.
"""
from typing import (
    Any,
    Callable,
    Literal,
    Sequence,
    Generator,
    cast,
)
from collections import defaultdict
from contextlib import contextmanager
import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


def get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


def _create_read_hook(
    layer_name: str, records: dict[int, list[Tensor]], clone_activations: bool
) -> Any:
    """Create a hook function that records the model activation at :layer_name:"""

    def hook_fn(_m: Any, _inputs: Any, outputs: Any) -> Any:
        activation = untuple_tensor(outputs)
        if not isinstance(cast(Any, activation), Tensor):
            raise ValueError(
                f"Expected a Tensor reading model activations, got {type(activation)}"
            )
        if clone_activations:
            activation = activation.clone().detach()
        records[layer_name].append(activation)
        return outputs

    return hook_fn


@contextmanager
def record_activations(
    model: nn.Module,
    layers_to_hook: list[str],
    clone_activations: bool = True,
) -> Generator[dict[int, list[Tensor]], None, None]:
    """
    Record the model activations at each layer type `layer_types`.
    This function will record every forward pass through the model
    at all layers of the given layer_type.

    Args:
        model: The model to record activations from.
        layer_types: A list of types of layers to record activations from.
        layer_config: A dictionary mapping layer types to layer matching functions.
            If not provided, this will be inferred automatically.
        clone_activations: If True, clone the activations before recording them. Default True.
        layer_nums: A list of layer numbers to record activations from. If None, record
            activations from all matching layers
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
        >>> with record_activations(model, layer_type="decoder_block") as recorded_activations:
        >>>     model.forward(...)
        >>> # recorded_activations is a dictionary mapping layer names to lists of activations
    """
    recorded_activations: dict[int, list[Tensor]] = defaultdict(list)
    hooks: list[RemovableHandle] = []

    for layer_name in layers_to_hook:

        module = get_module(model, layer_name)
        hook_fn = _create_read_hook(
            layer_name,
            recorded_activations,
            clone_activations=clone_activations,
        )
        hooks.append(module.register_forward_hook(hook_fn))

    try:
        yield recorded_activations

    finally:
        for hook in hooks:
            hook.remove()


def main():
    """ Testing recorder """
    import json
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from celeba_dataset import celeba_dataset
    from train import DDPM

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
