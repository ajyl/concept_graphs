"""
Utility function to record activations.
"""
from typing import (
    Any,
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
