"""
Module Doc String
"""
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
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

# %%


def plot(img, out_filepath):
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


def plot_series(img, out_filepath):

    fig = plt.figure(figsize=(20, 10))
    num_row = 5
    num_col = 10
    gs = GridSpec(num_row, num_col)
    processed = np.transpose(np.clip(img, 0, 1), (0, 1, 3, 4, 2))
    for _idx in range(num_row * num_col):
        curr_row = _idx // num_col
        curr_col = _idx % num_col

        ax = fig.add_subplot(gs[curr_row, curr_col])

        isns.imshow(processed[_idx, 0], ax=ax)

    fig.savefig(out_filepath)


class HookedDDPM(DDPM):
    def __init__(
        self,
        text,
        nn_model,
        betas,
        n_T,
        device,
        probes,
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
        self.probes = probes
        shape_dir = self.probes[:, :, 0]
        color_dir = self.probes[:, :, 1]
        size_dir = self.probes[:, :, 2]
        self.probes_map = {
            0: shape_dir,
            1: color_dir,
            2: size_dir,
        }

    def sample_intervene(
        self,
        n_sample,
        target_vector,
        scales,
        beta,
        c_gen,
        orig_concept,
        target_concept,
        size,
        device,
        guide_w=0.0,
        average_concepts=None,
        gamma=False,
    ):
        """
        concept[0]: shape
        concept[1]: color
        concept[2]: size
        """

        def add(add_vector, subtract_vector, _scale):
            def hook(module, input, output):

                # _scale: [batch]
                # output: [batch, d_model, 1, 1]
                _output = output[:, :, 0, 0]
                _add_vector = add_vector / add_vector.norm()
                _add_vector = _add_vector.unsqueeze(0).repeat(
                    (output.shape[0], 1)
                )
                scaled = einsum(
                    "batch, batch d_model -> batch d_model",
                    beta * _scale,
                    _add_vector,
                )

                updated = _output + scaled
                updated = updated.unsqueeze(-1).unsqueeze(-1)
                return updated

            return hook

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
                    _add_vec = target_vector[:, self.n_T - i]
                    # _subtract_vec = self.probes_map[_c_idx][
                    #    :, self.n_T - i  # , orig_concept[_c_idx]
                    # ]
                    _hooks.append(
                        self.nn_model.to_vec.register_forward_hook(
                            add(_add_vec, None, scales[:, self.n_T - i])
                        )
                    )

            eps = self.nn_model(
                x_i,
                _c_gen,
                t_is,
                #condition_concepts=condition_concepts,
                average_concepts=average_concepts,
                gamma=gamma,
            )
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


def sample_unhooked(model, c_label, concept, size, device):
    with torch.no_grad():
        gen_x = model.sample(16, c_label, size, device)

    plot_series(gen_x[1], f"null_intervene_{concept}.png")


def sample_unhooked_unconditioned(
    model, c_label, concept, condition_concepts, average_concepts, size, device
):
    with torch.no_grad():
        gen_x = model.sample(
            16,
            c_label,
            size,
            device,
            condition_concepts=condition_concepts,
            average_concepts=average_concepts,
        )

    _condition_concepts = "".join([str(x) for x in condition_concepts])
    plot_series(
        gen_x[1],
        f"null_intervene_conditioned_{_condition_concepts}_concepts_{concept}.png",
    )


def sample_hooked(
    model,
    target_vec,
    orig_concept_vec,
    scales,
    beta,
    c_label,
    orig_concept,
    target_concept,
    size,
    device,
    average_concepts=None,
    gamma=1,
):
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
        with record_activations(
            model, layers_to_hook=["nn_model.to_vec"]
        ) as recorded_acts:
            intervened_x = model.sample_intervene(
                16,
                target_vec,
                scales,
                beta,
                c_label,
                orig,
                target,
                size,
                device,
                average_concepts=average_concepts,
                gamma=gamma,
            )

    # batch time d_model]
    # acts is ordered from timestep 500 to 0.
    acts = (
        torch.stack(recorded_acts["nn_model.to_vec"], dim=1)
        .squeeze(-1)
        .squeeze(-1)
    )

    target_vec = target_vec.transpose(0, 1)
    target_comp_vec_normalized = einsum(
        "time d_model, time -> time d_model",
        target_vec,
        1 / target_vec.norm(dim=1),
    )
    target_comps = einsum(
        "batch time d_model, time d_model -> batch time",
        acts,
        target_comp_vec_normalized,
    )

    orig_concept_vec = orig_concept_vec.transpose(0, 1)
    orig_comp_vec_normalized = einsum(
        "time d_model, time -> time d_model",
        orig_concept_vec,
        1 / orig_concept_vec.norm(dim=1),
    )
    orig_comps = einsum(
        "batch time d_model, time d_model -> batch time",
        acts,
        orig_comp_vec_normalized,
    )

    _avged = False
    if average_concepts is not None:
        _avged = True

    plot_series(
        intervened_x[1],
        f"intervened_from_{orig_concept}_to_{target_concept}_avg_{_avged}_gamma_{gamma}_beta_{beta}_bleh.png",
    )
    return target_comps, orig_comps, acts.norm(dim=-1)


def get_clabel(concept):
    """
    Build clabel object.
    """
    c_shape = int(concept[0])
    c_color = int(concept[1])
    c_size = int(concept[2])

    colors_map = {
        0: [0.9, 0.1, 0.1],
        1: [0.1, 0.1, 0.9],
        2: [0.1, 0.9, 0.1],
    }

    if c_size == 0:
        size = 2.6
    else:
        size = 1.6

    return {
        0: torch.tensor([c_shape]),
        1: torch.tensor(colors_map[c_color]),
        2: torch.tensor([size]),
    }


def get_components(model, comp_vec, target_vec, c_label, size, device):
    """
    z
    """
    # After transpose: [time, d_model]
    comp_vec = comp_vec.transpose(0, 1)
    target_vec = target_vec.transpose(0, 1)
    with torch.no_grad():
        with record_activations(
            model, layers_to_hook=["nn_model.to_vec"]
        ) as recorded_acts:
            model.sample(16, c_label, size, device)

    # batch time d_model]
    # acts is ordered from timestep 500 to 0.
    acts = (
        torch.stack(recorded_acts["nn_model.to_vec"], dim=1)
        .squeeze(-1)
        .squeeze(-1)
    )

    comp_vec_normalized = einsum(
        "time d_model, time -> time d_model",
        comp_vec,
        1 / comp_vec.norm(dim=1),
    )
    comps = einsum(
        "batch time d_model, time d_model -> batch time",
        acts,
        comp_vec_normalized,
    )

    target_comp_vec_normalized = einsum(
        "time d_model, time -> time d_model",
        target_vec,
        1 / target_vec.norm(dim=1),
    )
    target_comps = einsum(
        "batch time d_model, time d_model -> batch time",
        acts,
        target_comp_vec_normalized,
    )
    return comps, target_comps, acts.norm(dim=-1)


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
    probes = torch.load(run_config["probe_path"])
    probes.requires_grad = False

    # load_partial_model(run_config)
    hooked = HookedDDPM(
        is_text,
        nn_model,
        run_config["betas"],
        n_T,
        device,
        probes,
        drop_prob=run_config["drop_prob"],
        n_classes=run_config["n_classes"],
        flag_weight=0,
    )

    probes_map = hooked.probes_map
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
        changed_concepts = [
            idx
            for idx, concept in enumerate(orig_concept)
            if concept != target_concept[idx]
        ]
        # For now, assume only single concept changes.
        assert len(changed_concepts) == 1
        changed_concept = changed_concepts[0]

        orig_clabel = get_clabel(orig_concept)
        target_clabel = get_clabel(target_concept)

        orig_concept_value = int(orig_concept[changed_concept])
        target_concept_value = int(target_concept[changed_concept])

        orig_concept_vec_tT = probes_map[changed_concept][
            :, :, orig_concept_value
        ]
        target_concept_vec_tT = probes_map[changed_concept][
            :, :, target_concept_value
        ]

        color_poss_values = {
            1: [
                torch.tensor([0.9, 0.1, 0.1]),
                torch.tensor([0.1, 0.1, 0.9]),
                # torch.tensor([0.1, 0.9, 0.1]),
            ]
        }

        # sample_unhooked_unconditioned(
        #    hooked,
        #    target_clabel,
        #    target_concept,
        #    [0, 2],
        #    color_poss_values,
        #    size,
        #    device,
        # )
        print("Getting original concept components...")
        # orig_comps, target_comps_in_orig, orig_acts_norms = get_components(
        #    hooked,
        #    orig_concept_vec_tT,
        #    target_concept_vec_tT,
        #    orig_clabel,
        #    size,
        #    device,
        # )
        target_comps, orig_comps_in_target, target_acts_norms = get_components(
            hooked,
            target_concept_vec_tT,
            orig_concept_vec_tT,
            target_clabel,
            size,
            device,
        )
        # _tmp = [
        #    orig_comps.mean(dim=0).cpu().numpy(),
        #    target_comps_in_orig.mean(dim=0).cpu().numpy(),
        #    orig_acts_norms.mean(dim=0).cpu().numpy(),
        # ]
        # df = pd.DataFrame(_tmp).transpose()
        # df.index = torch.arange(500).numpy()
        # zxcv = px.line(df)
        # zxcv.write_image("hmm.png")

        # _tmp = [
        #    target_comps.mean(dim=0).cpu().numpy(),
        #    orig_comps_in_target.mean(dim=0).cpu().numpy(),
        #    target_acts_norms.mean(dim=0).cpu().numpy(),
        # ]
        # df = pd.DataFrame(_tmp).transpose()
        # df.index = torch.arange(500).numpy()
        # zxcv = px.line(df)
        # zxcv.write_image("hmm2.png")

        # sample_unhooked(hooked, orig_clabel, orig_concept, size, device)
        # sample_unhooked(hooked, target_clabel, target_concept, size, device)
        for beta in [1, 10, 50, 100, 150, 200]:
            (
                target_comps_intervened,
                orig_comps_intervened,
                act_norms_intervened,
            ) = sample_hooked(
                hooked,
                target_concept_vec_tT,
                orig_concept_vec_tT,
                target_comps,
                beta,
                orig_clabel,
                orig_concept,
                target_concept,
                size,
                device,
                # average_concepts=color_poss_values,
                gamma=0.5,
            )

            _tmp = [
                target_comps_intervened.mean(dim=0).cpu().numpy(),
                orig_comps_intervened.mean(dim=0).cpu().numpy(),
                act_norms_intervened.mean(dim=0).cpu().numpy(),
            ]
            df = pd.DataFrame(_tmp).transpose()
            df.index = torch.arange(500).numpy()
            zxcv = px.line(df)
            zxcv.write_image(f"hmm_{beta}.png")


if __name__ == "__main__":
    main()
