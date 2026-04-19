"""
train.py -- Training script for QuantEpiGNN

Usage:
    cd 03_code/
    export PYTHONPATH=$(pwd)/src
    python scripts/train.py --dataset spatial457_20k --data_root data/spatial457_20k/scenes

Key hyperparameters (see --help for full list):
    --hidden_dim       256     GNN hidden dimension
    --lambda_metric    1.0     weight of Huber loss relative to CE
    --lr               1e-3    Adam learning rate
    --epochs           100
    --sem_dim          384     must match SEM_DIM in scene_graph_to_pyg.py
    --num_pred_classes 14      must match NUM_PRED_CLASSES in scene_graph_to_pyg.py
"""

import argparse
import json
import math
import os
from glob import glob
from itertools import combinations
from pathlib import Path

import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from step2_epistemic_gnn.epistemic_gnn import QuantEpiGNN, DualStreamLoss, build_scene_graph_data
from step2_epistemic_gnn.ablation_gnn import AblationGNN
from step2_epistemic_gnn.scene_graph_to_pyg import (
    load_scene_graph_dataset, load_embed_model, NUM_PRED_CLASSES, SEM_DIM,
    PRED_TO_IDX,
)


# ---------------------------------------------------------------------------
# Spatial457-20k loader (no images required — uses 3d_coords only)
# ---------------------------------------------------------------------------

def _scene_json_to_pyg(scene: dict, embed_cache: dict, embed_fn) -> Data | None:
    """Convert one superCLEVR per-scene JSON dict to a PyG Data object."""
    objects  = scene.get("objects", [])
    mask_box = scene.get("obj_mask_box", {})
    N = len(objects)
    if N < 2:
        return None

    W, H = 640, 480  # superCLEVR canonical resolution
    img_diag = math.sqrt(W ** 2 + H ** 2)

    node_sem, node_bbox, node_depth = [], [], []
    for i, obj in enumerate(objects):
        mb = mask_box.get(str(i), {}).get("obj", None)
        if mb is None or not mb:
            return None
        x, y, w, h = mb[0]
        if w <= 0 or h <= 0:
            return None
        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)

        pcoords = obj.get("pixel_coords", None)
        camera_z = float(pcoords[0][2]) if pcoords else (y1 + y2) / 2.0

        label = f"{obj.get('size','?')}_{obj.get('color','?')}_{obj.get('shape','?')}"
        if label not in embed_cache:
            embed_cache[label] = embed_fn([label])[0]
        node_sem.append(embed_cache[label])
        node_bbox.append([x1 / W, y1 / H, x2 / W, y2 / H])
        node_depth.append([camera_z])

    coords_3d = [o["3d_coords"] for o in objects]

    src_list, dst_list = [], []
    dist_list, conf_list, angle_list, ddiff_list = [], [], [], []
    cls_list, target_list = [], []

    for i, j in combinations(range(N), 2):
        for (s, t) in [(i, j), (j, i)]:
            dx = coords_3d[s][0] - coords_3d[t][0]
            dy = coords_3d[s][1] - coords_3d[t][1]
            dz = coords_3d[s][2] - coords_3d[t][2]
            dist_m = math.sqrt(dx * dx + dy * dy + dz * dz)

            cx_s = (node_bbox[s][0] + node_bbox[s][2]) / 2.0
            cy_s = (node_bbox[s][1] + node_bbox[s][3]) / 2.0
            cx_t = (node_bbox[t][0] + node_bbox[t][2]) / 2.0
            cy_t = (node_bbox[t][1] + node_bbox[t][3]) / 2.0
            angle = math.atan2(cy_t - cy_s, cx_t - cx_s)
            depth_diff = node_depth[s][0] - node_depth[t][0]

            src_list.append(s); dst_list.append(t)
            dist_list.append(dist_m)
            conf_list.append(1.0)
            angle_list.append(angle)
            ddiff_list.append(depth_diff)
            cls_list.append(PRED_TO_IDX.get("near" if dist_m < 2.0 else "far", 0))
            target_list.append(dist_m)

    if not src_list:
        return None

    data = Data(
        node_sem        = torch.tensor(node_sem, dtype=torch.float32),
        node_bbox       = torch.tensor(node_bbox, dtype=torch.float32),
        node_depth      = torch.tensor(node_depth, dtype=torch.float32),
        edge_index      = torch.tensor([src_list, dst_list], dtype=torch.long),
        edge_dist       = torch.tensor(dist_list, dtype=torch.float32).unsqueeze(1),
        edge_conf       = torch.tensor(conf_list, dtype=torch.float32).unsqueeze(1),
        edge_angle      = torch.tensor(angle_list, dtype=torch.float32).unsqueeze(1),
        edge_depth_diff = torch.tensor(ddiff_list, dtype=torch.float32).unsqueeze(1),
        target_classes  = torch.tensor(cls_list, dtype=torch.long),
        target_dist     = torch.tensor(target_list, dtype=torch.float32).unsqueeze(1),
    )
    return data


def load_spatial457_20k(scenes_dir: str, val_split: float = 0.05) -> tuple[list, list]:
    """
    Load up to 24k individual per-scene JSONs from scenes_dir.
    Returns (train_graphs, val_graphs) as PyG Data lists.
    No images required — uses 3d_coords for GT distances.
    """
    import numpy as np

    print(f"Loading Spatial457-20k from {scenes_dir} ...")
    scene_files = sorted(glob(os.path.join(scenes_dir, "*.json")))
    if not scene_files:
        raise RuntimeError(f"No JSON files found in {scenes_dir}")
    print(f"  Found {len(scene_files)} scene JSON files")

    embed_model = load_embed_model()

    def embed_fn(texts):
        return embed_model.encode(texts, convert_to_numpy=True).tolist()

    embed_cache: dict = {}
    graphs, skipped = [], 0
    for i, fpath in enumerate(scene_files):
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{len(scene_files)}  "
                  f"({len(graphs)} ok, {skipped} skipped)")
        with open(fpath) as f:
            scene = json.load(f)
        g = _scene_json_to_pyg(scene, embed_cache, embed_fn)
        if g is None:
            skipped += 1
        else:
            graphs.append(g)

    print(f"  Done: {len(graphs)} graphs ({skipped} skipped)")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(graphs))
    n_val = max(1, int(len(graphs) * val_split))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]


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
        # Load Step 1 JSON scene graphs produced by step1_scene_graph/run_pipeline.py.
        # Accepts either:
        #   (a) a flat directory of JSONs → auto-split 80/20 into train/val
        #   (b) a directory with train/ and val/ subdirs → use them directly
        embed_model = load_embed_model()
        train_dir = os.path.join(args.data_root, "train")
        val_dir   = os.path.join(args.data_root, "val")

        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            train = load_scene_graph_dataset(train_dir, embed_model=embed_model)
            val   = load_scene_graph_dataset(val_dir,   embed_model=embed_model)
        else:
            all_graphs = load_scene_graph_dataset(args.data_root, embed_model=embed_model)
            if not all_graphs:
                raise RuntimeError(f"No JSON files found in {args.data_root}")
            if len(all_graphs) < 2:
                raise RuntimeError(
                    f"Need at least 2 scene graphs for train/val split, "
                    f"found {len(all_graphs)} in {args.data_root}"
                )
            split = max(1, int(len(all_graphs) * (1 - args.val_split)))
            split = min(split, len(all_graphs) - 1)  # guarantee at least 1 val graph
            train, val = all_graphs[:split], all_graphs[split:]
            print(f"Auto-split: {len(train)} train / {len(val)} val "
                  f"({args.val_split:.0%} val)")

        if not train:
            raise RuntimeError(f"No training graphs found in {args.data_root}")
        return train, val

    if args.dataset == "spatial457_20k":
        return load_spatial457_20k(args.data_root, val_split=args.val_split)

    raise NotImplementedError(
        f"Dataset '{args.dataset}' not implemented. Use --dataset mock, step1, or spatial457_20k."
    )


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _apply_vlm_noise(data, sigma: float):
    """
    Corrupt edge_dist with log-normal multiplicative noise to simulate VLM errors.

    Uses a mixture: 80% moderate noise (sigma) + 20% severe noise (2*sigma),
    which covers both good VLMs (InternVL2 ≈ factor 2 errors) and bad ones
    (LLaVA ≈ factor 10 errors) in the same training distribution.
    """
    import torch
    E = data.edge_dist.shape[0]
    severe_mask = torch.rand(E, 1, device=data.edge_dist.device) < 0.20
    noise_sigma = torch.where(severe_mask,
                              torch.full_like(data.edge_dist, 2.0 * sigma),
                              torch.full_like(data.edge_dist, sigma))
    log_noise = torch.randn_like(data.edge_dist) * noise_sigma
    data.edge_dist = (data.edge_dist * torch.exp(log_noise)).clamp(min=0.01)
    return data


def train_epoch(model, loader, loss_fn, optimizer, device, vlm_noise_sigma: float = 0.0):
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "huber": 0.0}
    for data in loader:
        data = data.to(device)
        if vlm_noise_sigma > 0:
            data = _apply_vlm_noise(data, vlm_noise_sigma)
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
                   help="flat dir of JSONs, or dir with train/ and val/ subdirs")
    g.add_argument("--val_split",       type=float, default=0.2,
                   help="val fraction when using a flat data_root (default 0.2)")
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
    g.add_argument("--no_geom_constraint", action="store_true",
                   help="ablation: disable triangle-inequality residual weighting")
    g.add_argument("--no_epistemic",       action="store_true",
                   help="ablation: disable epistemic sigma uncertainty")

    g = p.add_argument_group("loss")
    g.add_argument("--lambda_metric", type=float, default=1.0,
                   help="weight of Huber loss relative to CE")
    g.add_argument("--huber_delta",   type=float, default=1.0,
                   help="Huber transition point (metres)")

    g = p.add_argument_group("training")
    g.add_argument("--epochs",          type=int,   default=50)
    g.add_argument("--lr",              type=float, default=1e-3)
    g.add_argument("--batch_size",      type=int,   default=16)
    g.add_argument("--vlm_noise_sigma", type=float, default=0.8,
                   help="Log-normal σ for VLM dist noise augmentation. "
                        "0=disabled, 0.8=default (covers factor-2 to factor-10 VLM errors)")

    g = p.add_argument_group("io")
    g.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    g.add_argument("--resume",         type=str, default=None,
                   help="path to checkpoint to resume training from")
    g.add_argument("--device",         type=str, default=None,
                   help="device override e.g. cuda:0, cuda:1, cpu (default: auto)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_geom = not args.no_geom_constraint
    use_epi  = not args.no_epistemic
    variant  = ("full" if (use_geom and use_epi)
                else f"{'no_geom' if not use_geom else ''}{'_' if not use_geom and not use_epi else ''}{'no_epi' if not use_epi else ''}")

    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Variant: {variant}  (geom_constraint={use_geom}  epistemic={use_epi})")
    print(f"Model  : hidden_dim={args.hidden_dim}, sem_dim={args.sem_dim}, "
          f"num_pred_classes={args.num_pred_classes}")
    print(f"Loss   : lambda_metric={args.lambda_metric}, huber_delta={args.huber_delta}")
    print(f"Train  : lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}\n")

    train_data, val_data = get_dataset(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False)
    print(f"Train graphs: {len(train_data)}  |  Val graphs: {len(val_data)}\n")

    model = AblationGNN(
        sem_dim              = args.sem_dim,
        hidden_dim           = args.hidden_dim,
        num_pred_classes     = args.num_pred_classes,
        use_geom_constraint  = use_geom,
        use_epistemic        = use_epi,
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
        if start_epoch > args.epochs:
            print(f"Checkpoint epoch {start_epoch - 1} >= --epochs {args.epochs}. Nothing to train.")
            return

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(args.checkpoint_dir, f"best_{variant}.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        tr = train_epoch(model, train_loader, loss_fn, optimizer, device,
                         vlm_noise_sigma=args.vlm_noise_sigma)
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
