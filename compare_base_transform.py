import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

from methods.protonet import ProtoNet
import numpy as np

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

    test_loader = test_dataset.get_data_loader()

    # For MAML (and other optimization-based methods), need to instantiate backbone layers with fast weight
    if cfg.method.fast_weight:
        backbone = instantiate(cfg.backbone, x_dim=test_dataset.dim, fast_weight=True)
    else:
        backbone = instantiate(cfg.backbone, x_dim=test_dataset.dim)

    # Instantiate few-shot method class
    model = instantiate(cfg.method.cls, backbone=backbone)

    if torch.cuda.is_available():
        model = model.cuda()

    return test_loader, model


@hydra.main(version_base=None, config_path="conf", config_name="main")
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if "name" not in cfg.exp:
        raise ValueError("The 'exp.name' argument is required!")

    if cfg.mode not in ["train", "test"]:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    fix_seed(cfg.exp.seed)

    test_loader, model = initialize_dataset_model(cfg)

    model_file = get_model_file(cfg)

    model.load_state_dict(torch.load(model_file)["state"])
    model.eval()

    model_base = ProtoNet(model.feature, cfg.n_way, cfg.n_shot)
    model_base.eval()

    print(f"Base embeddings: {model_base.test_loop(test_loader)}")
    print(f"Transformed embeddings: {model.test_loop(test_loader)}")


if __name__ == "__main__":
    hydra_setup()
    run()
