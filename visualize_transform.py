import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from itertools import islice

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

    # Check that method used is one of our implemented methods
    if cfg.method.name not in ["feat", "feads", "fealstm"]:
        raise ValueError("You should run this script with FEAT/FEADS/FEALSTM method.")

    if "name" not in cfg.exp:
        raise ValueError("The 'exp.name' argument is required!")

    if cfg.mode not in ["train", "test"]:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    # Fix the seed
    fix_seed(cfg.exp.seed)

    # Get the dataloader and model
    train_loader, val_loader, model = initialize_dataset_model(cfg)

    # Initialize model
    model_file = get_model_file(cfg)

    model.load_state_dict(torch.load(model_file, map_location="cuda" if torch.cuda.is_available() else "cpu")['state'])
    model.eval()

    # Visualize
    fig, ax = visualize(model, val_loader, cfg.get("dataloader_index", 0))

    # Output the visualization to file or display the visualization
    if "output" in cfg:
        fig.savefig(cfg.output, bbox_inches='tight')
    else:
        plt.show()


def visualize(model, dataloader, index):
    # Extract data
    data, labels = next(islice(dataloader, index, None))
    W, N, D = data.shape
    query = data.reshape(-1, D)
    if torch.cuda.is_available():
        query = query.cuda()
    query = model.feature(query).cpu().detach()
    labels = labels.reshape(-1)
    labels = np.unique(labels, return_inverse=True)[1] + 1

    # Calculate prototypes
    means = torch.Tensor(query).reshape(W, N, -1).mean(1)
    if torch.cuda.is_available():
        means = means.cuda()
    
    # Transform prototypes
    means_transformed = model.transform(means).cpu().detach()

    # Perform PCA
    pca = PCA(2, random_state=1)
    pca = pca.fit(query)

    # Use PCA on the data
    query_pca = pca.transform(query)
    old_means_pca = pca.transform(query.reshape(W, N, -1).mean(1))
    new_means_pca = pca.transform(means_transformed)
    
    # Construct dataframes for visualization
    query_df = pd.DataFrame(query_pca, columns=["x", "y"])
    query_df["Class"] = labels
    query_df["Type"] = "Data"
    query_df["alpha"] = 0.6

    old_means_df = pd.DataFrame(old_means_pca, columns=["x", "y"])
    old_means_df["Class"] = query_df["Class"].unique()
    old_means_df["Type"] = "Old"
    old_means_df["alpha"] = 1

    new_means_df = pd.DataFrame(new_means_pca, columns=["x", "y"])
    new_means_df["Class"] = query_df["Class"].unique()
    new_means_df["Type"] = "New"
    new_means_df["alpha"] = 1

    # Visualize data
    sns.set_theme()
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
   
    data = pd.concat([query_df, old_means_df, new_means_df])
    sns.scatterplot(data, x="x", y="y", size="Type", sizes=[100, 300, 300], alpha=data["alpha"], markers=[".", "^", "*"], style="Type", hue="Class", palette="tab10", legend="brief")
    
    # Fix layout (move legend outside, remove x and y labels)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)
    ax.set(yticklabels=[])
    ax.tick_params(left=False)
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    fig.tight_layout()

    return fig, ax


if __name__ == '__main__':
    hydra_setup()
    run()