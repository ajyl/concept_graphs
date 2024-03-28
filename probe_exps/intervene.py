"""
Module Doc String
"""
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# import seaborn as sns
import seaborn_image as isns
import plotly.express as px
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from fancy_einsum import einsum
from train import DDPM
from record_utils import record_activations
from model_components import UnetUp, EmbedFC
from probe_exps.probe_utils import load_acts
from DDPM import DDPM
from cg_datasets.shapes_dataset import shapes_dataset
from cg_constants import ROOT_DIR


class HookedDDPM(DDPM):
    def __init__(
        self,
        text,
        nn_model,
        betas,
        n_T,
        device,
        probe_path,
        drop_prob=0.1,
        n_classes=None,
        flag_weight=0,
    ):
        super(HookedDDPM, self).__init__(
            text,
            nn_model,
            betas,
            n_T,
            device,
            drop_prob,
            n_classes,
            flag_weight,
        )
        self.probe_path = probe_path
        self.probe = torch.load(probe_path)
        self.probe.requires_grad = False

    def sample_intervene(
        self,
        n_sample,
        c_gen,
        orig_concept,
        target_concept,
        size,
        device,
        guide_w=0.0,
    ):
        """
        concept[0]: shape
        concept[1]: color
        concept[2]: size
        """

        def add(add_vector, subtract_vector, _scale):
            def hook(module, input, output):
                # output: [batch, d_model, 1, 1]

                _output = output[:, :, 0, 0]
                # _add_vector = add_vector / add_vector.norm()
                _add_vector = add_vector
                _add_vector = _add_vector.unsqueeze(0).repeat(
                    (output.shape[0], 1)
                )
                add_coeff = _output.norm(dim=1) * F.cosine_similarity(
                    _output, _add_vector
                )
                add_component = einsum(
                    "batch, batch d_model -> batch d_model",
                    add_coeff,
                    _add_vector,
                )

                updated = _output + _scale * _add_vector
                # updated = _output + _scale * add_component

                # _subtract_vector = subtract_vector / subtract_vector.norm()

                # _subtract_vector = _subtract_vector.unsqueeze(0).repeat(
                #    (output.shape[0], 1)
                # )
                # subtract_coeff = updated.norm(dim=1) * F.cosine_similarity(
                #    updated, _subtract_vector
                # )
                # print(subtract_coeff)
                # subtract_component = einsum(
                #    "batch, batch d_model -> batch d_model",
                #    F.relu(subtract_coeff),
                #    _subtract_vector,
                # )
                # print(subtract_component.norm())
                # updated = updated - subtract_component

                # scaled = (output.norm() / updated.norm()) * updated
                # return scaled
                updated = updated.unsqueeze(-1).unsqueeze(-1)
                return updated

            return hook

        scale = 2
        # self.probe.shape: [d_model, t, concepts, options]
        shape_dir = self.probe[:, :, 0, target_concept[0]]
        color_dir = self.probe[:, :, 1, target_concept[1]]
        size_dir = self.probe[:, :, 2, target_concept[2]]

        probe_map = {
            0: shape_dir,
            1: color_dir,
            2: size_dir,
        }

        x_i = torch.randn(n_sample, *size).to(
            device
        )  # x_T ~ N(0, 1), sample initial noise

        _c_gen = (
            c_gen[:n_sample]
            if self.text
            else [
                tmpc_gen[:n_sample].to(device) for tmpc_gen in c_gen.values()
            ]
        )
        # context_mask = torch.zeros(len(_c_gen)).to(self.device) if self.text else torch.zeros_like(_c_gen[0]).to(device)

        condition_concepts = []
        for idx in range(len(orig_concept)):
            if orig_concept[idx] == target_concept[idx]:
                condition_concepts.append(idx)

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            # print(f"sampling timestep {i}", end="\r")
            print(f"sampling timestep {i}")
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            _hooks = []
            for _c_idx, _c_label in enumerate(orig_concept):
                if _c_label != target_concept[_c_idx]:
                    _add_vec = probe_map[_c_idx][
                        :, self.n_T - i  # , target_concept[_c_idx]
                    ]
                    _subtract_vec = probe_map[_c_idx][
                        :, self.n_T - i  # , orig_concept[_c_idx]
                    ]
                    # _add_vec = probe_map[_c_idx][
                    #    :, 400, target_concept[_c_idx]
                    # ]
                    # _subtract_vec = probe_map[_c_idx][
                    #    :, 400, orig_concept[_c_idx]
                    # ]
                    # _intervene_vec = probe_map[_c_idx][:, i - 1]
                    _hooks.append(
                        self.nn_model.to_vec.register_forward_hook(
                            add(_add_vec, _subtract_vec, scale)
                        )
                    )

            eps = self.nn_model(
                x_i,
                _c_gen,
                t_is,
                condition_concepts=condition_concepts
            )  # , context_mask)
            for _hook in _hooks:
                _hook.remove()

            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 10 == 0:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


def plot(img, out_filepath):
    breakpoint()
    img = img[0].detach().cpu().numpy()
    processed = np.transpose(np.clip(img, 0, 1), (0, 2, 3, 1))

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4)
    for _idx in range(16):
        curr_row = _idx // 4
        curr_col = _idx % 4

        ax = fig.add_subplot(gs[curr_row, curr_col])
        isns.imshow(processed[_idx], ax=ax)

    fig.savefig(out_filepath)


def plot_animate(img, out_filepath):
    processed = np.transpose(np.clip(img, 0, 1), (0, 1, 3, 4, 2))

    breakpoint()
    fig = px.imshow(processed[0])
    breakpoint()


def plot_series(img, out_filepath):

    fig = plt.figure(figsize=(20, 10))
    num_row = 5
    num_col = 10
    gs = GridSpec(num_row, num_col)
    processed = np.transpose(np.clip(img, 0, 1), (0, 1, 3, 4, 2))
    for _idx in range(num_row * num_col):
        curr_row = _idx // num_col
        curr_col = _idx % num_col

        try:
            ax = fig.add_subplot(gs[curr_row, curr_col])
        except:
            breakpoint()

        isns.imshow(processed[_idx, 0], ax=ax)

    fig.savefig(out_filepath)


def sample_unhooked(model, c_label, concept, size, device):
    with torch.no_grad():
        gen_x = model.sample(16, c_label, size, device)

    plot_series(gen_x[1], f"null_intervene_{concept}.pdf")


def sample_hooked(model, c_label, orig_concept, target_concept, size, device):
    """
    z
    """
    orig = (int(orig_concept[0]), int(orig_concept[1]), int(orig_concept[2]))
    target = (
        int(target_concept[0]),
        int(target_concept[1]),
        int(target_concept[2]),
    )
    with torch.no_grad():
        intervened_x = model.sample_intervene(
            16, c_label, orig, target, size, device
        )
    plot_series(
        intervened_x[1],
        f"intervened_from_{orig_concept}_to_{target_concept}.pdf",
    )


def main():
    """ Driver """
    run_config = {
        "is_text": False,
        "nn_model_path": os.path.join(ROOT_DIR, "ckpts/shapes_v1/epoch_99.pt"),
        "betas": (1e-4, 0.02),
        "n_T": 500,
        "in_channels": 4,
        "n_feat": 256,
        "drop_prob": 0.1,
        # "n_classes": [2, 2, 2],
        "n_classes": [2, 3, 1],
        "device": "cuda:0",
        "batch_size": 128,
        "pixel_size": 28,
        "concepts": ["000", "001", "010", "100", "011", "101", "110", "111"],
        "activations_dir": os.path.join(
            ROOT_DIR, "probe_exps/cached_acts/shapes/bottleneck"
        ),
        "n_samples": 32,
        "probe_path": os.path.join(
            ROOT_DIR, "probe_exps/probe_ckpts/shapes/bottleneck/epoch_19.pt"
        ),
        "total_samples": 64,
    }
    is_text = run_config["is_text"]
    device = run_config["device"]
    n_T = run_config["n_T"]
    in_channels = run_config["in_channels"]
    pixel_size = run_config["pixel_size"]
    size = (in_channels, pixel_size, pixel_size)

    nn_model = torch.load(run_config["nn_model_path"]).to(device)

    # load_partial_model(run_config)
    hooked = HookedDDPM(
        is_text,
        nn_model,
        run_config["betas"],
        n_T,
        device,
        run_config["probe_path"],
        drop_prob=run_config["drop_prob"],
        n_classes=run_config["n_classes"],
        flag_weight=0,
    )

    bottleneck_samples = 16
    total_samples = run_config["total_samples"]
    tf = transforms.Compose(
        [
            transforms.Resize((pixel_size, pixel_size)),
            transforms.ToTensor(),
        ]
    )
    test_concepts = [
        # ["010", "000"],
        ["000", "010"],
        # ["000", "100"],
        # ["100", "000"],
        # ["010", "110"],
        # ["110", "010"]
    ]
    for concept in test_concepts:
        orig_concept = concept[0]
        target_concept = concept[1]

        dataset = shapes_dataset(
            False,
            transform=tf,
            num_samples=total_samples,
            configs=orig_concept,
            training=False,
            test_size=1.6,
        )
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1
        )
        _, c_label = next(iter(dataloader))
        # sample_unhooked(hooked, c_label, orig_concept, size, device)
        sample_hooked(
            hooked, c_label, orig_concept, target_concept, size, device
        )


if __name__ == "__main__":
    main()
