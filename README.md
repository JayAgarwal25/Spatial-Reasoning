# Detecting Quantitative Spatial Hallucinations in Vision-Language Models via Geometric Consistency Verification and Selective Visual Grounding

> **Status:** Active development — Step 1 (zero-shot scene graph extraction) and Step 2 (epistemic GNN + geometric consistency residuals) complete. Step 3 (visual grounding feedback loop) and Step 4 (evaluation) in progress.

---

## Motivation

Vision-Language Models (VLMs) fail systematically at quantitative spatial reasoning. The failure mode is specific: VLMs produce metric estimates (distances, relative positions) that are not merely imprecise — they are often **globally geometrically inconsistent**. A model might claim `d(A,C) = 15cm` while separately estimating `d(A,B) = 50cm` and `d(B,C) = 10cm`. Each claim looks plausible in isolation; together they violate the triangle inequality.

Standard uncertainty estimation (MC dropout, per-node variance) cannot catch this because it operates locally. The proposed approach is **selective visual grounding via geometric consistency verification**: propagate triangle-inequality constraints across the full scene graph to identify which specific VLM claims are mutually contradictory, then trigger targeted regrounding only on those edges. The signal that drives this — the consistency residual — is deterministic, requires no stochastic sampling, and pinpoints *which* edges to requery rather than flagging the whole scene as uncertain.

---

## Architecture Overview

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│  Step 1: Zero-Shot Scene    │
│  Graph Extraction           │
│  (Florence-2 / DINO +       │
│   MiDaS + VLM predicates)  │
└─────────────┬───────────────┘
              │  G = (V, E) with edge_dist, edge_conf
              ▼
┌─────────────────────────────┐
│  Step 2: Epistemic GNN      │
│  - Epistemic Node Encoder   │
│    (mu, sigma) per node     │
│  - Geometric Constraint     │
│    Message Passing          │
│  - Consistency Residual r   │
│    per edge                 │
│  - Dual-Stream Loss         │
│    (CE + Huber)             │
└─────────────┬───────────────┘
              │  residuals r per edge
              ▼
         r > ε ?
        /       \
      YES        NO
       │          │
       ▼          ▼
┌───────────┐  Final spatial
│  Step 3:  │  reasoning output
│  Visual   │
│  Grounding│
│  Agent    │
│  (typed   │
│   actions)│
└─────┬─────┘
      │  annotated image
      └──────────────────► back to Step 1
```

---

## The Core Contribution: Geometric Consistency Residuals

For every direct edge `(A→C)` in the scene graph, the GNN finds all two-hop paths `A→B→C` and computes:

```
r_AC = | d_AC_direct  −  mean_B( d_AB + d_BC ) |
```

A high residual means the VLM's direct claim about `(A, C)` is geometrically irreconcilable with its surrounding edge estimates. This is the hallucination signal.

Message passing weights incoming messages by `exp(−r)` — consistent edges contribute more, contradictory edges are down-weighted. The residual is deterministic; no stochastic sampling is required.

---

## Repository Structure

```
quant_epignn/
├── step1_scene_graph/
│   ├── node_localization.py       # Florence-2 / GroundingDINO
│   ├── depth_estimation.py        # MiDaS depth maps
│   ├── geometric_filter.py        # PRISM-0 inspired candidate pruning
│   ├── predicate_parser.py        # VLM captioning + LLM parsing
│   └── graph_builder.py           # Assembles PyG Data object
│
├── step2_epistemic_gnn/
│   ├── epistemic_gnn.py           # Main: encoder + message passing + loss
│   └── epistemic_gnn_v1.py        # Archive: original MC DropEdge version
│
├── step3_visual_agent/
│   ├── trigger.py                 # Residual thresholding
│   ├── actions.py                 # draw_bbox, draw_line, depth_crop
│   └── feedback_loop.py           # Recursive iteration manager
│
├── step4_evaluation/
│   ├── spatialqa_eval.py
│   ├── nuscenes_eval.py
│   └── ablation.py
│
├── data/
├── configs/
│   └── default.yaml
├── requirements.txt
└── README.md
```

---

## Installation

```bash
conda create -n quant_epignn python=3.10
conda activate quant_epignn

pip install torch torchvision
pip install torch-geometric
pip install transformers sentence-transformers
pip install networkx opencv-python numpy scipy shapely matplotlib tqdm
```

---

## Data Format (Step 1 → Step 2 Interface)

Step 2 expects a PyTorch Geometric `Data` object with named fields. No flat `x` blob.

```python
Data(
    node_sem       = FloatTensor(N, sem_dim),   # VLM semantic embedding
    node_bbox      = FloatTensor(N, 4),          # [x, y, w, h]
    node_depth     = FloatTensor(N, 1),          # mean MiDaS depth in bbox
    edge_index     = LongTensor(2, E),
    edge_dist      = FloatTensor(E, 1),          # VLM predicted metric distance
    edge_conf      = FloatTensor(E, 1),          # VLM confidence score ∈ [0,1]
    edge_angle     = FloatTensor(E, 1),          # relative angle between centroids
    edge_depth_diff= FloatTensor(E, 1),          # depth difference
)
```

---

## Step 2 Interface Contract

**Input:** `Data` object as above

**Output:**

```python
{
    "pred_classes":  LongTensor(E,),     # predicted predicate class per edge
    "pred_dist":     FloatTensor(E, 1),  # refined metric distance per edge
    "residuals":     FloatTensor(E, 1),  # geometric consistency residual r per edge
    "mu":            FloatTensor(N, H),  # node mean embeddings
    "sigma":         FloatTensor(N, H),  # node uncertainty embeddings
}
```

Step 3 reads `residuals` directly. No other coupling between steps.

---

## Training

```bash
python train.py \
  --dataset spatialqa \
  --lambda_metric 1.0 \
  --residual_threshold 0.3 \
  --max_iters 3 \
  --epochs 50
```

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_metric` | 1.0 | Weight of Huber loss relative to CE |
| `residual_threshold ε` | calibrated | Calibrated per dataset on a validation set via precision-recall curve for re-grounding decisions; 0.3 is a starting point only |
| `max_iters` | 3 | Max feedback loop iterations |
| `hidden_dim` | 256 | GNN hidden dimension |

---

## Evaluation

```bash
# SpatiaLQA multi-step reasoning
python step4_evaluation/spatialqa_eval.py --model checkpoints/best.pt

# NuScenes-SpatialQA distance MAE
python step4_evaluation/nuscenes_eval.py --model checkpoints/best.pt

# Ablation suite
python step4_evaluation/ablation.py --model checkpoints/best.pt
```

**Ablations:**

| Ablation | What it isolates |
|----------|-----------------|
| A1: Residual integrated into GNN message passing vs. residual as post-hoc filter (applied to raw VLM outputs, no GNN aggregation) | Does GNN integration do real work, or is the raw residual useful standalone without propagation? |
| A2: No feedback loop | Contribution of active visual grounding |
| A3: Standard GNN vs EpiGNN (deterministic node embeddings vs. mu/sigma) | Does sigma routing in message passing change outcomes? If not, simplify the encoder |
| A4: CE only vs CE + Huber | Contribution of metric loss stream |
| A5: exp(-r) consistency weighting vs. learned attention gating on edges | Is the fixed geometric prior for edge weighting adequate, or does learned routing improve performance? |

---

## Benchmarks

### Primary Evaluation Datasets

| Dataset | Year | Size | Metric GT | Domain | Access |
|---------|------|------|-----------|--------|--------|
| **Spatial457** | CVPR 2025 | 457 synthetic scenes | Exact 3D coords per object | Indoor/outdoor synthetic | HuggingFace (`RyanWW/Spatial457`) |
| **3DSRBench** | ICCV 2025 | 2,772 QA pairs | RGB-D depth (real sensor) | Indoor real + synthetic | HuggingFace (`ccvl/3DSRBench`, CC BY 4.0) |

These two are the primary benchmarks. Both provide ground-truth metric distances necessary to verify triangle inequality violations — the core claim of this system.

**Spatial457** uses synthetic scenes with perfect 3D object coordinates, making it ideal for controlled evaluation: given exact positions A, B, C we can compute the ground-truth d(A,C) and check whether the VLM's claim violates `|d(A,C) − (d(A,B) + d(B,C))| > ε`. No measurement noise; any residual spike is unambiguously attributable to the VLM.

**3DSRBench** uses real RGB-D images with depth-sensor ground truth. This tests the system under realistic depth noise — directly relevant to the source-ambiguity limitation described below.

### Secondary / Future Benchmarks

| Dataset | Year | Size | Notes |
|---------|------|------|-------|
| **SpatialBench** | CVPR 2026 | 1,347 QA / 50 videos | LiDAR GT distances; real egocentric video; strongest real-world test but requires frame extraction |
| **NuScenes-SpatialQA** | 2025 | 3.5M QA pairs | LiDAR GT; autonomous driving; HuggingFace repo currently empty, base dataset 700 GB |
| **SpatiaLQA** | 2026 | 9,605 QA pairs | No metric distances (logical/categorical only); not suited for triangle-inequality evaluation |

### Evaluation Metrics

| Dataset | Primary Metric | Notes |
|---------|---------------|-------|
| Spatial457 | MAE on predicted distances | Exact GT → direct MAE computation |
| 3DSRBench | Accuracy + MAE | CircularEval and FlipEval protocols |
| SpatialBench | MRA (mean relative accuracy) | Numerical tolerance thresholds; MAE computed on top |

---

## Known Limitations & Open Questions

**Source ambiguity in the consistency residual.**
The residual measures internal disagreement among the VLM and depth stack outputs. A high residual can arise from two distinct causes: (a) the VLM hallucinated a metric distance, or (b) the MiDaS depth estimate is noisy and corrupted the edge features that feed into the residual computation. The current system does not separate these. A high residual is correctly interpreted as "something is wrong here" — not specifically as "the VLM hallucinated."

**The key publication-strength validation needed:**
On a dataset with ground-truth hallucination labels, measure whether the consistency residual correlates specifically with hallucination rate versus depth noise rate. If residuals spike equally on noisy depth estimates as on VLM hallucinations, the source-bias problem weakens the claim. This experiment is the gating condition for a strong paper claim about hallucination detection.

**ε threshold generalization.**
The re-grounding threshold ε is calibrated per dataset. Its transferability across scene types (indoor Spatial457 vs. real-world 3DSRBench vs. egocentric SpatialBench) is untested and should be treated as a hyperparameter requiring per-domain tuning.

---

## Team

| Step | Lead | Component |
|------|------|-----------|
| 1 | Gorang | Zero-shot scene graph extraction |
| 2 | Jay | Epistemic GNN + geometric consistency residuals |
| 3 | Harsimar | Visual grounding feedback loop |
| 4 | Mayukh | Evaluation, benchmarking, ablation |

---

## Key Design Decisions

**Why geometric consistency residuals over MC dropout?**
MC dropout variance measures local prediction instability under parameter perturbation. Consistency residuals measure whether the VLM's joint spatial claims are mutually compatible — a fundamentally different and more interpretable signal for detecting metric hallucinations.

**Why directed incoming-edge uncertainty seeding?**
Incoming edges to a node represent VLM claims about that node as a spatial target. Low confidence on those edges directly implies the node is poorly grounded. Outgoing edges, reflecting this node's role as an anchor for others, influence message passing weights — not the node's own uncertainty seed.

**Why (mu, sigma) node encoding?**
This is a design choice, not a claimed contribution. The encoder only earns its place if sigma influences message passing routing in a way that meaningfully changes residuals and downstream re-grounding decisions. Ablation A3 directly tests this: if deterministic embeddings produce equivalent residuals, the encoder should be simplified to a standard projection. Do not cite the encoder as a contribution until A3 confirms it.

**Why exp(-r) consistency weighting?**
It is a sensible geometric prior — edges with low residuals are down-weighted less, so geometrically coherent claims propagate more. However, this weighting function is fixed and makes assumptions about the residual scale. Ablation A5 compares it against learned attention gating on edges to determine whether a data-driven routing function improves over the geometric prior.

**Why Huber over MAE for metric loss?**
Huber is less sensitive to outlier distance predictions during early training while converging to MAE behavior for small errors. This matters when the scene graph initially contains noisy VLM distance estimates.
