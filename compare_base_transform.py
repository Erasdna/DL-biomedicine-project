import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

from methods.protonet import ProtoNet
import numpy as np
from math import ceil

from utils.io_utils import (
    get_resume_file,
    hydra_setup,
    fix_seed,
    model_to_dict,
    opt_to_dict,
    get_model_file,
)


def initialize_dataset_model(cfg):
    # Instantiate test dataset
    if cfg.method.eval_type == "simple":
        test_dataset = instantiate(
            cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode="test"
        )
    else:
        test_dataset = instantiate(
            cfg.dataset.set_cls, n_episode=cfg.iter_num, mode="test"
        )

    # For MAML (and other optimization-based methods), need to instantiate backbone layers with fast weight
    if cfg.method.fast_weight:
        backbone = instantiate(cfg.backbone, x_dim=test_dataset.dim, fast_weight=True)
    else:
        backbone = instantiate(cfg.backbone, x_dim=test_dataset.dim)

    # Instantiate few-shot method class
    model = instantiate(cfg.method.cls, backbone=backbone)

    if torch.cuda.is_available():
        model = model.cuda()

    return test_dataset, model


@hydra.main(version_base=None, config_path="conf", config_name="main")
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.method.name not in ["feat", "feads", "fealstm"]:
        raise ValueError("You should run this script with FEAT/FEADS/FEALSTM method.")

    if "name" not in cfg.exp:
        raise ValueError("The 'exp.name' argument is required!")

    if cfg.mode not in ["train", "test"]:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    fix_seed(cfg.exp.seed)

    test_dataset, model = initialize_dataset_model(cfg)

    model_file = get_model_file(cfg)

    model.load_state_dict(torch.load(model_file)["state"])
    model.eval()

    model_base = ProtoNet(model.feature, cfg.n_way, cfg.n_shot)
    model_base.eval()

    print(f"Base embeddings (acc, std): {test(model_base, test_dataset, cfg)}")
    print(f"Transformed embeddings (acc, std): {test(model, test_dataset, cfg)}")


def test(model, test_dataset, cfg):
    test_loader = test_dataset.get_data_loader()

    if cfg.method.eval_type == "simple":
        acc_all = []

        num_iters = ceil(cfg.iter_num / len(test_dataset.get_data_loader()))
        cfg.iter_num = num_iters * len(test_dataset.get_data_loader())
        print("num_iters", num_iters)
        for i in range(num_iters):
            acc_mean, acc_std = model.test_loop(test_loader, return_std=True)
            acc_all.append(acc_mean)

        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

    else:
        # Don't need to iterate, as this is accounted for in num_episodes of set data-loader
        acc_mean, acc_std = model.test_loop(test_loader, return_std=True)

    return acc_mean, acc_std


if __name__ == "__main__":
    hydra_setup()
    run()
