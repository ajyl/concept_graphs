"""
DDPM module.
"""

import numpy as np
import torch
import torch.nn as nn


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(
        0, T + 1, dtype=torch.float32
    ) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(
        self,
        text,
        nn_model,
        betas,
        n_T,
        device,
        drop_prob=0.1,
        n_classes=None,
        flag_weight=0,
    ):
        super(DDPM, self).__init__()
        self.text = text
        self.nn_model = nn_model.to(device)
        self.n_classes = n_classes

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v.to(device))

        self.betas = torch.linspace(betas[0], betas[1], n_T).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.n_T = n_T
        self.device = device
        self.flag_weight = flag_weight
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(
            self.device
        )  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        return self.loss_mse(
            noise, self.nn_model(x_t, c, _ts / self.n_T)
        )  # , context_mask))

    def sample(
        self,
        n_sample,
        c_gen,
        size,
        device,
        guide_w=0.0,
        condition_concepts=None,
        average_concepts=None,
    ):

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

        x_i_store = []
        for i in range(self.n_T, 0, -1):
            print(f"sampling timestep {i}", end="\r")
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(
                x_i,
                _c_gen,
                t_is,
                condition_concepts=condition_concepts,
                average_concepts=average_concepts,
            )
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 10 == 0:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
