"""
Train script
"""
import os
import json
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.utils import save_image, make_grid
from einops import repeat
from load_dataset import my_dataset
from cg_datasets.celeba_dataset import celeba_dataset
from cg_datasets.shapes_dataset import shapes_dataset
from DDPM import DDPM
from model_components import ContextUnet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str)
parser.add_argument("--sample-every", type=int, default=50)
parser.add_argument("--save-ckpt-every", type=int, default=1)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--text", action="store_true")
parser.add_argument("--lrate", default=1e-4, type=float)
parser.add_argument("--test_size", default=1.6, type=float)
parser.add_argument("--alpha", default=1500, type=int)
parser.add_argument("--beta", default=2.0, type=float)
parser.add_argument("--num_samples", default=5000, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--n_T", default=500, type=int)
parser.add_argument("--n_feat", default=256, type=int)
parser.add_argument("--n_sample", default=64, type=int)
parser.add_argument("--n_epoch", default=200, type=int)
parser.add_argument("--experiment", default="H32-train1", type=str)
parser.add_argument("--remove_node", default="None", type=str)
parser.add_argument("--type_attention", default="", type=str)
parser.add_argument("--pixel_size", default=28, type=int)
parser.add_argument("--dataset", default="single-body_2d_3classes", type=str)
parser.add_argument("--scheduler", default="", type=str)
parser.add_argument("--seed", type=int, default=1)


class ConvertToRGB(object):
    def __call__(self, img):
        if img.shape[0] == 1:  # Grayscale image
            img = torch.cat([img, img, img], dim=0)
        elif img.shape[0] == 4:
            img = img[:3]  # Drop the alpha channel
        return img

    def ddim_step(self, x_t, t, noise_pred):
        """
        DDIM step to predict the next state of the image.
        """
        alpha_t = self.alphas_cumprod[t]
        alpha_t_1 = torch.where(
            t > 0,
            self.alphas_cumprod[t - 1],
            torch.tensor(1.0).to(self.device),
        )
        sigma_t = torch.sqrt(
            (1 - alpha_t_1) / (1 - alpha_t) * (1 - alpha_t / alpha_t_1)
        )
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        alpha_t_1 = alpha_t_1.view(-1, 1, 1, 1)

        x_0_pred = (x_t - sigma_t * noise_pred) / torch.sqrt(alpha_t)
        x_t_1 = torch.sqrt(alpha_t_1) * x_0_pred + sigma_t * torch.randn_like(
            x_t
        )
        return x_t_1

    def sample_ddim(self, n_sample, c_gen, size, device):
        """
        Sample using the DDIM scheduler.
        """
        x_t = torch.randn(n_sample, *size).to(device)  # Initialize with noise

        _c_gen = {k: v.to(device) for k, v in c_gen.items()}

        x_i_store = []
        for i in reversed(range(0, self.n_T)):
            print(f"sampling timestep {i}", end="\r")
            t = torch.full((n_sample,), i, device=device, dtype=torch.long)
            noise_pred = self.nn_model(x_t, _c_gen, t.float() / self.n_T)
            x_t = self.ddim_step(x_t, t, noise_pred)

            if i % 20 == 0:
                x_i_store.append(x_t.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_t, x_i_store


def training(args):
    print("start")
    start_time = time.time()
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    n_T = args.n_T
    n_feat = args.n_feat
    lrate = args.lrate
    alpha = args.alpha
    beta = args.beta
    test_size = args.test_size
    dataset = args.dataset
    num_samples = args.num_samples
    pixel_size = args.pixel_size
    experiment = args.experiment
    n_sample = args.n_sample
    type_attention = args.type_attention
    remove_node = args.remove_node
    seed = args.seed
    scheduler = args.scheduler
    in_channels = (
        3 if any([x in dataset for x in ["celeba", "astronaut"]]) else 4
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    with open("config_category.json", "r") as f:
        configs = json.load(f)[experiment]
    print(configs)

    experiment_classes = {
        "H42-train1": [2, 3, 1, 1],
        "H22-train1": [2, 2],
        # "H32-train1": [2, 2, 2],
        "default": [2, 3, 1],
    }
    n_classes = experiment_classes.get(
        experiment, experiment_classes["default"]
    )

    if "celeba" in dataset:
        n_classes = [2, 2, 2]

    if "astronaut" in dataset:
        tf = transforms.Compose(
            [
                transforms.Resize((pixel_size, pixel_size)),
                transforms.ToTensor(),
                ConvertToRGB(),
            ]
        )
    else:
        tf = transforms.Compose(
            [
                transforms.Resize((pixel_size, pixel_size)),
                transforms.ToTensor(),
            ]
        )

    save_dir = (
        f'./output{"_dbg" if args.debug else ""}/'
        + f'{dataset}{"_txt" if args.text else ""}'
        + "/"
        + args.exp_name
        + "/"
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_dir = (
        save_dir
        + str(pixel_size)
        + str(num_samples)
        + "_"
        + str(test_size)
        + "_"
        + str(n_feat)
        + "_"
        + str(n_T)
        + "_"
        + str(n_epoch)
        + "_"
        + str(lrate)
        + "_"
        + remove_node
        + "_"
        + str(alpha)
        + "_"
        + str(beta)
        + "_"
        + str(seed)
        + "/"
    )  # + str(type_attention) + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    ddpm = DDPM(
        text=args.text,
        nn_model=ContextUnet(
            text=args.text,
            in_channels=in_channels,
            n_feat=n_feat,
            n_classes=n_classes,
            dataset=dataset,
            type_attention=type_attention,
            pixel_size=pixel_size,
            device=device,
        ),
        betas=(lrate, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1,
        n_classes=n_classes,
    )
    ddpm.to(device)
    print("model", time.time() - start_time)

    if dataset.startswith("celeba"):
        train_dataset = celeba_dataset(
            args.text,
            tf,
            num_samples,
            configs=configs["train"],
            training=True,
            alpha=alpha,
            remove_node=remove_node,
        )

    elif dataset.startswith("single-body_2d_3classes"):
        train_dataset = shapes_dataset(
            args.text,
            transform=tf,
            num_samples=num_samples,
            configs=configs["train"],
            training=True,
            alpha=alpha,
            remove_node=remove_node,
        )

    else:
        train_dataset = my_dataset(
            args.text,
            tf,
            num_samples,
            dataset,
            configs=configs["train"],
            training=True,
            alpha=alpha,
            remove_node=remove_node,
        )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    print("train", time.time() - start_time)
    test_dataloaders = {}
    log_dict = {
        "train_loss_per_batch": [],
        "test_loss_per_batch": {key: [] for key in configs["test"]},
    }
    output_configs = list(set(configs["test"] + configs["train"]))
    for config in output_configs:

        if dataset.startswith("celeba"):
            test_dataset = celeba_dataset(
                args.text,
                tf,
                n_sample,
                configs=config,
                training=False,
                test_size=test_size,
            )

        elif dataset.startswith("single-body_2d_3classes"):
            test_dataset = shapes_dataset(
                args.text,
                transform=tf,
                num_samples=n_sample,
                configs=config,
                training=False,
                test_size=test_size,
            )

        else:
            test_dataset = my_dataset(
                args.text,
                tf,
                n_sample,
                dataset,
                configs=config,
                training=False,
                test_size=test_size,
            )

        test_dataloaders[config] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )

    print("test", time.time() - start_time)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    for ep in range(n_epoch):
        print(f"epoch {ep}")

        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(train_dataloader)
        for x, c in pbar:
            # print('train batch', time.time() - start_time)
            optim.zero_grad()
            x = x.to(device)
            _c = c if args.text else [tmpc.to(device) for tmpc in c.values()]
            loss = ddpm(x, _c)
            log_dict["train_loss_per_batch"].append(loss.item())
            loss.backward()
            loss_ema = loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():

            for test_config in configs["test"]:
                for test_x, test_c in test_dataloaders[test_config]:
                    test_x = test_x.to(device)
                    _test_c = (
                        test_c
                        if args.text
                        else [
                            tmptest_c.to(device)
                            for tmptest_c in test_c.values()
                        ]
                    )
                    test_loss = ddpm(test_x, _test_c)
                    log_dict["test_loss_per_batch"][test_config].append(
                        test_loss.item()
                    )

            if (ep + 1) % args.save_ckpt_every == 0:

                ckpt_dir = os.path.join("ckpts/", args.exp_name)
                if not os.path.isdir(ckpt_dir):
                    os.makedirs(ckpt_dir)
                ckpt_filepath = os.path.join(ckpt_dir, f"epoch_{ep}.pt")
                torch.save(ddpm.nn_model, ckpt_filepath)

            if (ep + 1) % args.sample_every == 0 or ep >= (n_epoch - 5):

                for test_config in output_configs:
                    x_real, c_gen = next(iter(test_dataloaders[test_config]))
                    x_real = x_real[:n_sample].to(device)
                    if scheduler == "DDIM":
                        x_gen, x_gen_store = ddpm.sample_ddim(
                            n_sample,
                            c_gen,
                            (in_channels, pixel_size, pixel_size),
                            device,
                        )
                    else:
                        x_gen, x_gen_store = ddpm.sample(
                            n_sample,
                            c_gen,
                            (in_channels, pixel_size, pixel_size),
                            guide_w=0.0,
                        )
                    np.savez_compressed(
                        save_dir
                        + f"image_"
                        + test_config
                        + "_ep"
                        + str(ep)
                        + ".npz",
                        x_gen=x_gen.detach().cpu().numpy(),
                    )
                    print(
                        "saved image at "
                        + save_dir
                        + f"image_"
                        + test_config
                        + "_ep"
                        + str(ep)
                        + ".png"
                    )

                    if ep + 1 == n_epoch:
                        np.savez_compressed(
                            save_dir
                            + f"gen_store_"
                            + test_config
                            + "_ep"
                            + str(ep)
                            + ".npz",
                            x_gen_store=x_gen_store,
                        )
                        print(
                            "saved image file at "
                            + save_dir
                            + f"gen_store_"
                            + test_config
                            + "_ep"
                            + str(ep)
                            + ".npz"
                        )

            if (ep + 1) == n_epoch:
                with open(
                    save_dir + f"training_log_" + str(ep) + ".json", "w"
                ) as outfile:
                    json.dump(log_dict, outfile)


if __name__ == "__main__":
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    training(args)
