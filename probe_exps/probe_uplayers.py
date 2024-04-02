"""
Module Doc String
"""
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
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
from cg_constants import ROOT_DIR


class Unembed(nn.Module):
    def __init__(
        self, text, in_channels, n_feat, n_classes, pixel_size, device=None
    ):
        """
        Partial network.
        """
        super(Unembed, self).__init__()

        self.device = device
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.n_out1 = 2 * n_feat
        self.n_out2 = n_feat
        self.n_feat = 2 * n_feat
        self.n_classes = n_classes
        self.in_channels = in_channels

        n_conv = pixel_size // 4
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, n_conv, n_conv),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat, "", False)
        self.up2 = UnetUp(2 * n_feat, n_feat, "", False)

        if text:
            raise RuntimeError("Not implemented yet.")

        else:
            self.timeembed1 = EmbedFC(1, 2 * n_feat)
            self.timeembed2 = EmbedFC(1, 1 * n_feat)
            self.contextembed1 = nn.ModuleList(
                [
                    EmbedFC(self.n_classes[iclass], self.n_out1)
                    for iclass in range(len(self.n_classes))
                ]
            )
            self.contextembed2 = nn.ModuleList(
                [
                    EmbedFC(self.n_classes[iclass], self.n_out2)
                    for iclass in range(len(self.n_classes))
                ]
            )

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, bottleneck, timestep) -> torch.Tensor:
        """
        Forward pass starting from the bottleneck.
        bottleneck: [batch, d_model, 1, 1]
        """

        temb1 = self.timeembed1(timestep).view(-1, int(self.n_feat), 1, 1)
        temb2 = self.timeembed2(timestep).view(-1, int(self.n_feat / 2), 1, 1)

        # DESIGN DECISION 1: Set cemb to 1.
        cemb1 = 1
        cemb2 = 1

        # TODO: if self.is_res:
        # [b, 2*f, w/4, h/4]
        up1 = self.up0(bottleneck)

        # DESIGN DECISION 2: set down2 to 0.
        down1 = torch.zeros_like(up1)
        # [b, f, w/2, h/2]
        up2 = self.up1(cemb1 * up1 + temb1, down1)

        # DESIGN DECISION 3: set down1 to 0.
        down2 = torch.zeros_like(up2)
        # [b, f, w, h]
        up3 = self.up2(cemb2 * up2 + temb2, down2)

        # DESIGN DECISION 4: set x to 0.
        x = torch.zeros_like(up3)
        # [b, channel, w, h]
        out = self.out(torch.cat((up3, x), 1))
        return out

    def init_weights(self, weights_path, device):
        """
        Initialize network.
        """
        keep = [
            "timeembed1",
            "timeembed2",
            "contextembed1",
            "contextembed2",
            "up0",
            "up1",
            "up2",
            "out",
        ]
        state_dict = torch.load(weights_path).state_dict()
        keep_dict = {}
        for param_name, param_weights in state_dict.items():
            component = param_name.split(".")[0]
            if component in keep:
                param_weights.requires_grad = False
                keep_dict[param_name] = param_weights

        self.load_state_dict(keep_dict)
        self.to(device)


def huh(x, timestep):
    beta1 = 1e-4
    beta2 = 0.02
    T = 500

    beta_t = (beta2 - beta1) * torch.arange(
        0, T + 1, dtype=torch.float32
    ) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / sqrtmab

    size = (4, 28, 28)
    device = "cuda:0"
    z = torch.randn(x.shape[0], *size).to(device) if timestep > 1 else 0
    x_i = torch.randn(x.shape[0], *size).to(device)

    x_i = (
        oneover_sqrta[timestep] * (x * mab_over_sqrtmab[timestep])
    )
    return x_i


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
    }
    device = run_config["device"]
    n_T = run_config["n_T"]

    # load_partial_model(run_config)
    unembed = Unembed(
        run_config["is_text"],
        run_config["in_channels"],
        run_config["n_feat"],
        run_config["n_classes"],
        run_config["pixel_size"],
        run_config["device"],
    )
    unembed.init_weights(run_config["nn_model_path"], device)

    concepts = run_config["concepts"]
    concepts = ["000", "001", "010"]
    _, acts = load_acts(run_config["activations_dir"], concepts, 1)
    bottleneck_samples = 16
    random_idx = 42

    colors_map = {
        "0": [0.9, 0.1, 0.1],
        "1": [0.1, 0.1, 0.9],
        "2": [0.1, 0.9, 0.1],
    }
    for concept in concepts:
        timestep = 450

        # [batch, d_model]
        _acts = (
            acts[concept][
                random_idx : random_idx + bottleneck_samples, timestep
            ]
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).to(device)

        # shape = int(concept[0])
        # size = 2.6 if concept[2] == 0 else 1.6
        # concept_pt = {
        #    0: F.one_hot(shape, num_classes=2),
        #    1: torch.tensor(colors_map[concept[1]]).to(device),
        #    2: torch.tensor([size]).to(device),
        # }

        t_is = torch.tensor([timestep / n_T]).to(device)
        t_is = t_is.repeat(_acts.shape[0], 1, 1, 1)
        out = unembed(_acts, t_is)

        zxcv = huh(out, timestep)
        zxcv = zxcv.detach().cpu().numpy()

        zxcv = np.clip(zxcv, 0, 1)
        zxcv = np.transpose(zxcv, (0, 2, 3, 1))
        fig, axes = plt.subplots(
            ncols=4, nrows=1
        )
        axes[0].imshow(zxcv[0])
        plt.savefig("zxcvzxcv.png")

        breakpoint()
        print("z")


if __name__ == "__main__":
    main()
