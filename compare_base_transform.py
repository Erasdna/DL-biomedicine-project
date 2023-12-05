import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

from methods.protonet import ProtoNet
import numpy as np

from utils.io_utils import get_resume_file, hydra_setup, fix_seed, model_to_dict, opt_to_dict, get_model_file


def initialize_dataset_model(cfg):
    # Instantiate train dataset as specified in dataset config under simple_cls or set_cls
    if cfg.method.type == "baseline":
        train_dataset = instantiate(cfg.dataset.simple_cls, batch_size=cfg.method.train_batch, mode='train')
    elif cfg.method.type == "meta":
        train_dataset = instantiate(cfg.dataset.set_cls, mode='train')
    else:
        raise ValueError(f"Unknown method type: {cfg.method.type}")
    train_loader = train_dataset.get_data_loader()

    # Instantiate val dataset as specified in dataset config under simple_cls or set_cls
    # Eval type (simple or set) is specified in method config, rather than dataset config
    if cfg.method.eval_type == 'simple':
        val_dataset = instantiate(cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode='val')
    else:
        val_dataset = instantiate(cfg.dataset.set_cls, mode='val')
    val_loader = val_dataset.get_data_loader()

    # For MAML (and other optimization-based methods), need to instantiate backbone layers with fast weight
    if cfg.method.fast_weight:
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim, fast_weight=True)
    else:
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim)

    # Instantiate few-shot method class
    model = instantiate(cfg.method.cls, backbone=backbone)

    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.method.name == 'maml':
        cfg.method.stop_epoch *= model.n_task  # maml use multiple tasks in one update

    return train_loader, val_loader, model



@hydra.main(version_base=None, config_path='conf', config_name='main')
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if "name" not in cfg.exp:
        raise ValueError("The 'exp.name' argument is required!")

    if cfg.mode not in ["train", "test"]:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    fix_seed(cfg.exp.seed)

    train_loader, val_loader, model = initialize_dataset_model(cfg)

    model_file = get_model_file(cfg)

    model.load_state_dict(torch.load(model_file)['state'])
    model.eval()

    model_base = ProtoNet(model.feature, cfg.n_way, cfg.n_shot)
    model_base.eval()

    print(f"Base embeddings: {model_base.test_loop(val_loader)}")
    print(f"Transformed embeddings: {model.test_loop(val_loader)}")

if __name__ == '__main__':
    hydra_setup()
    run()
