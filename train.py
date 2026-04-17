"""
train.py -- Training script for Step 2: Epistemic GNN

Usage (mock data, for development):
    python train.py

Usage (Step 1 JSON outputs, once Gorang's pipeline has run):
    python train.py --dataset step1 --data_root data/scene_graphs/

The mock dataset generates synthetic scene graphs in the correct Step 1 output
format.  Swap it out by implementing a real loader in get_dataset() — the rest
of the training loop is dataset-agnostic.

Key hyperparameters (see --help for full list):
    --hidden_dim       256     GNN hidden dimension
    --lambda_metric    1.0     weight of Huber loss relative to CE
    --lr               1e-3    Adam learning rate
    --epochs           50
    --sem_dim          384     must match SEM_DIM in scene_graph_to_pyg.py
    --num_pred_classes 14      must match NUM_PRED_CLASSES in scene_graph_to_pyg.py
"""

import argparse
import os

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from step2_epistemic_gnn.epistemic_gnn import QuantEpiGNN, DualStreamLoss, build_scene_graph_data
from step2_epistemic_gnn.scene_graph_to_pyg import (
    load_scene_graph_dataset, load_embed_model, NUM_PRED_CLASSES, SEM_DIM,
)


# ---------------------------------------------------------------------------
# Mock dataset  (used until Step 1 outputs are available)
# ---------------------------------------------------------------------------

def make_mock_graph(N: int, E: int, sem_dim: int, num_pred_classes: int):
    """Builds one synthetic scene graph in the Step 1 → Step 2 interface format."""
    edge_index = torch.unique(torch.randint(0, N, (2, E)), dim=1)
    E = edge_index.size(1)
    data = build_scene_graph_data(
        node_sem        = torch.randn(N, sem_dim),
        node_bbox       = torch.rand(N, 4) * 100.0,
        node_depth      = torch.rand(N, 1) * 10.0,
        edge_index      = edge_index,
        edge_dist       = torch.rand(E, 1) * 5.0,
        edge_conf       = torch.rand(E, 1),
        edge_angle      = torch.rand(E, 1) * 3.14159,
        edge_depth_diff = torch.randn(E, 1),
    )
    data.target_classes = torch.randint(0, num_pred_classes, (E,))
    data.target_dist    = torch.rand(E, 1) * 5.0
    return data


def make_mock_dataset(n_graphs, N, E, sem_dim, num_pred_classes):
    return [make_mock_graph(N, E, sem_dim, num_pred_classes) for _ in range(n_graphs)]


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def get_dataset(args) -> tuple[list, list]:
    """
    Returns (train_dataset, val_dataset) as lists of PyG Data objects.

    To add a real dataset add an elif branch here.
    """
    if args.dataset == "mock":
        train = make_mock_dataset(
            args.mock_train_size, args.mock_N, args.mock_E,
            args.sem_dim, args.num_pred_classes,
        )
        val = make_mock_dataset(
            args.mock_val_size, args.mock_N, args.mock_E,
            args.sem_dim, args.num_pred_classes,
        )
        return train, val

    if args.dataset == "step1":
        # Load Step 1 JSON scene graphs produced by step1_scene_graph/run_pipeline.py
        embed_model = load_embed_model()
        train_dir = os.path.join(args.data_root, "train")
        val_dir   = os.path.join(args.data_root, "val")
        train = load_scene_graph_dataset(train_dir, embed_model=embed_model)
        val   = load_scene_graph_dataset(val_dir,   embed_model=embed_model)
        if not train:
            raise RuntimeError(f"No JSON files found in {train_dir}")
        return train, val

    raise NotImplementedError(
        f"Dataset '{args.dataset}' not implemented. Use --dataset mock or step1."
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "huber": 0.0}
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss, L_CE, L_Huber = loss_fn(
            out["sem_logits"], data.target_classes,
            out["pred_dist"],  data.target_dist,
        )
        loss.backward()
        optimizer.step()
        totals["loss"]  += loss.item()
        totals["ce"]    += L_CE.item()
        totals["huber"] += L_Huber.item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    totals = {"loss": 0.0, "ce": 0.0, "huber": 0.0}
    for data in loader:
        data = data.to(device)
        out  = model(data)
        loss, L_CE, L_Huber = loss_fn(
            out["sem_logits"], data.target_classes,
            out["pred_dist"],  data.target_dist,
        )
        totals["loss"]  += loss.item()
        totals["ce"]    += L_CE.item()
        totals["huber"] += L_Huber.item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, epoch, model, optimizer, best_val_loss, args):
    torch.save({
        "epoch":         epoch,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "args":          vars(args),
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_val_loss"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Step 2 Epistemic GNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("dataset")
    g.add_argument("--dataset",         type=str, default="mock",
                   help="'mock' or 'step1' (real Step 1 JSON outputs)")
    g.add_argument("--data_root",       type=str, default="data/scene_graphs/",
                   help="root dir for step1 dataset (expects train/ and val/ subdirs)")
    g.add_argument("--mock_train_size", type=int, default=200)
    g.add_argument("--mock_val_size",   type=int, default=50)
    g.add_argument("--mock_N",          type=int, default=12, help="nodes per mock graph")
    g.add_argument("--mock_E",          type=int, default=24, help="edges per mock graph")

    g = p.add_argument_group("model")
    g.add_argument("--sem_dim",          type=int, default=SEM_DIM,
                   help="VLM semantic embedding dimension (384 for all-MiniLM-L6-v2)")
    g.add_argument("--hidden_dim",       type=int, default=256)
    g.add_argument("--num_pred_classes", type=int, default=NUM_PRED_CLASSES,
                   help="number of predicate classes (14, matches Step 1 vocab)")

    g = p.add_argument_group("loss")
    g.add_argument("--lambda_metric", type=float, default=1.0,
                   help="weight of Huber loss relative to CE")
    g.add_argument("--huber_delta",   type=float, default=1.0,
                   help="Huber transition point (metres)")

    g = p.add_argument_group("training")
    g.add_argument("--epochs",     type=int,   default=50)
    g.add_argument("--lr",         type=float, default=1e-3)
    g.add_argument("--batch_size", type=int,   default=16)

    g = p.add_argument_group("io")
    g.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    g.add_argument("--resume",         type=str, default=None,
                   help="path to checkpoint to resume training from")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model  : hidden_dim={args.hidden_dim}, sem_dim={args.sem_dim}, "
          f"num_pred_classes={args.num_pred_classes}")
    print(f"Loss   : lambda_metric={args.lambda_metric}, huber_delta={args.huber_delta}")
    print(f"Train  : lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}\n")

    train_data, val_data = get_dataset(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False)
    print(f"Train graphs: {len(train_data)}  |  Val graphs: {len(val_data)}\n")

    model = QuantEpiGNN(
        sem_dim          = args.sem_dim,
        hidden_dim       = args.hidden_dim,
        num_pred_classes = args.num_pred_classes,
    ).to(device)

    loss_fn   = DualStreamLoss(lambda_metric=args.lambda_metric,
                               huber_delta=args.huber_delta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch   = 1
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, device
        )
        start_epoch += 1
        print(f"Resumed from {args.resume}  (starting at epoch {start_epoch})\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(args.checkpoint_dir, "best.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        tr = train_epoch(model, train_loader, loss_fn, optimizer, device)
        va = eval_epoch(model,  val_loader,   loss_fn,            device)

        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  train  loss={tr['loss']:.4f}  CE={tr['ce']:.4f}  Huber={tr['huber']:.4f}"
            f"  |  val  loss={va['loss']:.4f}  CE={va['ce']:.4f}  Huber={va['huber']:.4f}",
            end="",
        )

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            save_checkpoint(best_ckpt, epoch, model, optimizer, best_val_loss, args)
            print("  *")
        else:
            print()

    print(f"\nDone. Best val loss: {best_val_loss:.4f}  →  {best_ckpt}")


if __name__ == "__main__":
    main()
