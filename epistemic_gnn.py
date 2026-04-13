"""
Step 2: Quant-EpiGNN — Data Structure & Epistemic Node Encoder

Expected PyG Data fields (produced by Step 1):

  Node features (per node v):
    node_sem        (N, sem_dim)  — VLM semantic embedding
    node_bbox       (N, 4)        — bounding box [x, y, w, h]
    node_depth      (N, 1)        — monocular depth estimate

  Edge features (directed: u → v):
    edge_index      (2, E)        — [src_nodes; dst_nodes]
    edge_dist       (E, 1)        — VLM-predicted metric distance d_uv
    edge_conf       (E, 1)        — VLM confidence score c_uv ∈ [0, 1]
    edge_angle      (E, 1)        — relative angle (radians)
    edge_depth_diff (E, 1)        — signed depth difference (depth_u - depth_v)

  Named fields are kept separate (not flattened into x / edge_attr) so that
  geometric constraint message passing can index edge_dist and edge_conf directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import scatter


# Geometric node feature dimension: bbox(4) + depth(1)
GEOM_DIM = 5
# Edge feature dimension: edge_dist(1) + edge_conf(1) + edge_angle(1) + edge_depth_diff(1)
EDGE_FEAT_DIM = 4


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

def build_scene_graph_data(
    node_sem:        torch.Tensor,   # (N, sem_dim)
    node_bbox:       torch.Tensor,   # (N, 4)
    node_depth:      torch.Tensor,   # (N, 1)
    edge_index:      torch.Tensor,   # (2, E)
    edge_dist:       torch.Tensor,   # (E, 1)  VLM metric distance estimate
    edge_conf:       torch.Tensor,   # (E, 1)  VLM confidence ∈ [0, 1]
    edge_angle:      torch.Tensor,   # (E, 1)  relative angle in radians
    edge_depth_diff: torch.Tensor,   # (E, 1)  depth_u - depth_v
) -> Data:
    """
    Packages Step 1 outputs into a named-field PyG Data object.

    Returned Data object is the contract between Step 1 and Step 2.
    All downstream modules in this file consume this exact schema.
    """
    assert node_bbox.size(1) == 4,  "node_bbox must be (N, 4)"
    assert node_depth.size(1) == 1, "node_depth must be (N, 1)"
    assert edge_dist.size(1) == 1,  "edge_dist must be (E, 1)"
    assert edge_conf.size(1) == 1,  "edge_conf must be (E, 1)"

    return Data(
        node_sem        = node_sem,
        node_bbox       = node_bbox,
        node_depth      = node_depth,
        edge_index      = edge_index,
        edge_dist       = edge_dist,
        edge_conf       = edge_conf,
        edge_angle      = edge_angle,
        edge_depth_diff = edge_depth_diff,
        num_nodes       = node_sem.size(0),
    )


# ---------------------------------------------------------------------------
# Epistemic Node Encoder
# ---------------------------------------------------------------------------

class EpistemicNodeEncoder(nn.Module):
    """
    Encodes each node as a (mu, sigma) pair.

    mu
        Learned projection of the node's concatenated semantic + geometric
        features.  No uncertainty conditioning — a clean deterministic mean.

    sigma
        Learned projection that receives node features AND a scalar
        uncertainty seed derived from *incoming* edge confidences.

        uncertainty_seed_v = mean_{u: (u→v) ∈ E} (1 − c_{u→v})

        Rationale: incoming edges are the VLM's claims about this node as a
        spatial target.  Low confidence on those claims → poorly grounded
        node → high initial uncertainty.  Outgoing edges reflect this node's
        reliability as a reference anchor; that signal belongs in message
        passing weights, not here.

        Nodes with no incoming edges receive seed = 1.0 (maximally uncertain).
        softplus is applied at output so every element of sigma is strictly > 0.

    Args:
        sem_dim    : dimension of VLM semantic node embeddings
        hidden_dim : output dimension for both mu and sigma
    """

    def __init__(self, sem_dim: int, hidden_dim: int):
        super().__init__()
        node_in = sem_dim + GEOM_DIM  # sem + bbox(4) + depth(1)

        self.mu_proj = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # +1 for the scalar uncertainty seed prepended before the MLP.
        # The MLP learns how to spread that scalar signal across hidden_dim dims.
        self.sigma_proj = nn.Sequential(
            nn.Linear(node_in + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_sem:   torch.Tensor,   # (N, sem_dim)
        node_bbox:  torch.Tensor,   # (N, 4)
        node_depth: torch.Tensor,   # (N, 1)
        edge_index: torch.Tensor,   # (2, E)
        edge_conf:  torch.Tensor,   # (E, 1)  VLM confidence ∈ [0, 1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mu    : (N, hidden_dim)  deterministic mean embedding
        sigma : (N, hidden_dim)  strictly positive uncertainty vector
        """
        N = node_sem.size(0)
        x = torch.cat([node_sem, node_bbox, node_depth], dim=-1)  # (N, node_in)

        # ---- mu: clean feature projection, no uncertainty conditioning ----
        mu = self.mu_proj(x)  # (N, hidden_dim)

        # ---- sigma: conditioned on incoming-edge uncertainty seed ---------

        dst = edge_index[1]  # destination index of each directed edge, shape (E,)

        # Mean VLM confidence across all edges pointing *at* each node
        # reduce='mean' zero-fills entries with no contributions
        avg_incoming_conf = scatter(
            edge_conf, dst, dim=0, dim_size=N, reduce='mean'
        )  # (N, 1)

        # Detect nodes that received no incoming edges
        incoming_count = scatter(
            torch.ones_like(edge_conf), dst, dim=0, dim_size=N, reduce='sum'
        )  # (N, 1)
        isolated = (incoming_count == 0)  # (N, 1) bool

        uncertainty_seed = 1.0 - avg_incoming_conf   # (N, 1), ↑ when VLM unconfident
        uncertainty_seed[isolated] = 1.0             # maximally uncertain if unreferenced

        sigma_input = torch.cat([x, uncertainty_seed], dim=-1)  # (N, node_in + 1)
        sigma = F.softplus(self.sigma_proj(sigma_input))         # (N, hidden_dim), > 0

        return mu, sigma


# ---------------------------------------------------------------------------
# Geometric Constraint Message Passing
# ---------------------------------------------------------------------------

def compute_consistency_residuals(
    edge_index: torch.Tensor,   # (2, E)
    edge_dist:  torch.Tensor,   # (E, 1)
    num_nodes:  int,
) -> torch.Tensor:              # (E, 1)
    """
    For each directed edge (A→C), compute the geometric consistency residual:

        r_AC = | d_AC  −  mean_B( d_AB + d_BC ) |

    where the mean is over all intermediate nodes B such that both edges
    (A→B) and (B→C) exist in the graph (two-hop paths A→B→C).

    Interpretation
    --------------
    d_AB + d_BC is the triangle-inequality upper bound on d_AC.
    If the VLM's direct claim d_AC disagrees strongly with what multi-hop
    geometry demands, r_AC is large → likely hallucinated distance.

    Boundary conditions
    -------------------
    - No two-hop path exists for (A→C): r_AC = 0  (unconstrained; no signal).
    - Self-loops (A == C) are handled naturally; d_AA + d_AC = d_AC when valid.

    Complexity
    ----------
    O(E × N) time and memory.  Appropriate for scene graphs (N < 200, E < 1000).
    The (E, N) matrices are built once per forward pass and not stored.
    """
    src   = edge_index[0]    # (E,)
    dst   = edge_index[1]    # (E,)
    d_AC  = edge_dist[:, 0]  # (E,)

    # Adjacency distance matrix: adj[u, v] = d_uv if (u→v) exists, else -1
    adj = edge_dist.new_full((num_nodes, num_nodes), -1.0)
    adj[src, dst] = d_AC

    # d_AB[e, B]: distance from src[e] to every B  (-1 if edge absent)
    d_AB = adj[src]           # (E, N)
    # d_BC_T[e, B]: distance from every B to dst[e]  (-1 if edge absent)
    d_BC_T = adj[:, dst].T    # (N, E) → (E, N)

    # Two-hop paths: both legs must exist
    valid      = (d_AB >= 0) & (d_BC_T >= 0)          # (E, N) bool
    two_hop    = (d_AB + d_BC_T).masked_fill(~valid, 0.0)  # zero-out invalid
    path_count = valid.float().sum(dim=1)              # (E,)

    # Mean two-hop distance; fall back to d_AC when no path exists (→ r = 0)
    mean_two_hop = torch.where(
        path_count > 0,
        two_hop.sum(dim=1) / path_count.clamp(min=1.0),
        d_AC,
    )  # (E,)

    residuals = (d_AC - mean_two_hop).abs()  # (E,)
    return residuals.unsqueeze(1)            # (E, 1)


class GeometricConstraintMessagePassing(nn.Module):
    """
    One round of geometry-aware message passing over the epistemic graph.

    Messages
    --------
    For each directed edge (u→v), a message is computed from:
        [ mu_u  ||  sigma_u  ||  edge_features ]
    and weighted by exp(−r_{u→v}).  Edges with low consistency residuals
    (geometrically coherent VLM claims) contribute more to the aggregation.

    Node update
    -----------
    mu_new    = mu  +  MLP( weighted_mean_message )
                Residual connection: geometric context refines, not replaces.

    sigma_new = softplus( MLP([ weighted_mean_message  ||  mean_r_incoming ]) )
                Directly grounds epistemic uncertainty in geometric consistency:
                many conflicting incoming claims → high mean_r → high sigma.

    Output
    ------
    Returns updated (mu, sigma) AND the per-edge residuals r.
    The residuals are passed verbatim to Step 3 as the uncertainty trigger
    signal — they are deterministic geometric quantities, not learned.

    Args:
        hidden_dim : must match the output dim of EpistemicNodeEncoder
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Message MLP: [mu_src || sigma_src || edge_feats] → hidden_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + EDGE_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # mu update: maps aggregated message → residual delta
        self.mu_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # sigma update: [aggregated message || mean_incoming_residual(1)] → hidden_dim
        # The scalar residual term directly grounds uncertainty in geometry.
        self.sigma_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        mu:             torch.Tensor,   # (N, hidden_dim)
        sigma:          torch.Tensor,   # (N, hidden_dim)
        edge_index:     torch.Tensor,   # (2, E)
        edge_dist:      torch.Tensor,   # (E, 1)
        edge_conf:      torch.Tensor,   # (E, 1)
        edge_angle:     torch.Tensor,   # (E, 1)
        edge_depth_diff: torch.Tensor,  # (E, 1)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mu_new    : (N, hidden_dim)  updated mean embeddings
        sigma_new : (N, hidden_dim)  updated uncertainty vectors (> 0)
        residuals : (E, 1)           per-edge consistency residuals r  (→ Step 3)
        """
        N   = mu.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        # 1. Geometric consistency residuals  (no learned parameters involved)
        residuals = compute_consistency_residuals(edge_index, edge_dist, N)  # (E, 1)

        # 2. Consistency weights: high residual → low weight
        weights = torch.exp(-residuals)  # (E, 1), ∈ (0, 1]

        # 3. Build messages from source epistemic state + edge features
        edge_feat = torch.cat(
            [edge_dist, edge_conf, edge_angle, edge_depth_diff], dim=-1
        )  # (E, EDGE_FEAT_DIM)
        msg_input = torch.cat([mu[src], sigma[src], edge_feat], dim=-1)
        messages  = self.msg_mlp(msg_input)  # (E, hidden_dim)

        # 4. Weight and aggregate (weighted mean at each destination)
        weighted_msgs = messages * weights                                          # (E, hidden_dim)
        agg_msg   = scatter(weighted_msgs, dst, dim=0, dim_size=N, reduce='sum')   # (N, hidden_dim)
        weight_sum = scatter(weights,      dst, dim=0, dim_size=N, reduce='sum')   # (N, 1)
        agg_msg   = agg_msg / weight_sum.clamp(min=1e-8)                           # weighted mean

        # 5. Update mu via residual connection
        mu_new = mu + self.mu_update_mlp(agg_msg)  # (N, hidden_dim)

        # 6. Update sigma conditioned on mean incoming residual
        #    Nodes receiving geometrically inconsistent claims → high mean_r → high sigma
        mean_incoming_r = scatter(
            residuals, dst, dim=0, dim_size=N, reduce='mean'
        )  # (N, 1)
        sigma_input = torch.cat([agg_msg, mean_incoming_r], dim=-1)  # (N, hidden_dim+1)
        sigma_new   = F.softplus(self.sigma_update_mlp(sigma_input)) # (N, hidden_dim)

        return mu_new, sigma_new, residuals


# ---------------------------------------------------------------------------
# Dual-Stream Prediction Heads + Top-Level Model
# ---------------------------------------------------------------------------

class QuantEpiGNN(nn.Module):
    """
    Full Step 2 model.  Wraps encoder, message passing, and both output heads.

    Edge representation
    -------------------
    After message passing produces updated (mu, sigma), each edge (u→v) is
    represented as [ mu_u || mu_v ] — the epistemic means of its two endpoints.
    sigma is not included in the edge repr; it flows to the output dict for
    Step 3 to inspect, but the prediction heads operate on mu only.

    Semantic head  (E, 2*H) → (E, num_pred_classes)
        Cross-Entropy target: integer predicate class per edge.

    Metric head    (E, 2*H + 1) → (E, 1)
        The "+1" is edge_dist — the VLM's prior distance estimate.
        The head learns a *correction* to that prior rather than predicting
        distance from scratch.  Softplus at output ensures pred_dist > 0.
        Huber target: ground-truth metric distance per edge.

    Output dict (matches README interface contract)
    -----------------------------------------------
        sem_logits   (E, num_pred_classes)  — raw logits for loss computation
        pred_classes (E,)                   — argmax for inference / Step 3
        pred_dist    (E, 1)                 — refined metric distance
        residuals    (E, 1)                 — geometric consistency residual r
        mu           (N, hidden_dim)        — node mean embeddings
        sigma        (N, hidden_dim)        — node uncertainty embeddings

    Args:
        sem_dim         : VLM semantic embedding dimension
        hidden_dim      : GNN hidden dimension (default 256 per config)
        num_pred_classes: number of predicate classes for semantic head
    """

    def __init__(self, sem_dim: int, hidden_dim: int, num_pred_classes: int):
        super().__init__()
        self.encoder = EpistemicNodeEncoder(sem_dim, hidden_dim)
        self.mp      = GeometricConstraintMessagePassing(hidden_dim)

        # Semantic head: edge repr → predicate class logits
        self.sem_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_pred_classes),
        )

        # Metric head: edge repr + VLM prior → refined distance
        # edge_dist (+1) gives the head a prior to refine, not predict cold
        self.metric_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> dict:
        """
        Args:
            data : PyG Data object produced by build_scene_graph_data()

        Returns:
            dict with keys: sem_logits, pred_classes, pred_dist, residuals, mu, sigma
        """
        # 1. Epistemic node encoding
        mu, sigma = self.encoder(
            data.node_sem, data.node_bbox, data.node_depth,
            data.edge_index, data.edge_conf,
        )

        # 2. Geometric constraint message passing
        mu, sigma, residuals = self.mp(
            mu, sigma,
            data.edge_index, data.edge_dist, data.edge_conf,
            data.edge_angle, data.edge_depth_diff,
        )

        # 3. Build edge representations from updated node means
        src, dst   = data.edge_index
        edge_repr  = torch.cat([mu[src], mu[dst]], dim=-1)  # (E, 2*H)

        # 4. Semantic head
        sem_logits = self.sem_head(edge_repr)                # (E, num_pred_classes)

        # 5. Metric head (with VLM prior distance as additional input)
        metric_input = torch.cat([edge_repr, data.edge_dist], dim=-1)  # (E, 2H+1)
        pred_dist    = F.softplus(self.metric_head(metric_input))       # (E, 1), > 0

        return {
            "sem_logits"  : sem_logits,              # for loss
            "pred_classes": sem_logits.argmax(dim=1),# for inference / Step 3
            "pred_dist"   : pred_dist,
            "residuals"   : residuals,               # → Step 3 trigger
            "mu"          : mu,
            "sigma"       : sigma,
        }


# ---------------------------------------------------------------------------
# Dual-Stream Loss
# ---------------------------------------------------------------------------

class DualStreamLoss(nn.Module):
    """
    L = L_CE  +  lambda_metric * L_Huber

    L_CE   : CrossEntropy on predicate class logits  (semantic stream)
    L_Huber: Huber loss on refined distance predictions  (metric stream)

    Huber over MAE: less sensitive to noisy VLM distance outliers during
    early training; converges to MAE behaviour once predictions are close.

    Args:
        lambda_metric : weight on the Huber term (default 1.0, tunable)
        huber_delta   : Huber transition point in metres (default 1.0)
    """

    def __init__(self, lambda_metric: float = 1.0, huber_delta: float = 1.0):
        super().__init__()
        self.lambda_metric = lambda_metric
        self.ce    = nn.CrossEntropyLoss()
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(
        self,
        sem_logits:    torch.Tensor,   # (E, num_pred_classes)
        target_classes: torch.Tensor,  # (E,) long
        pred_dist:     torch.Tensor,   # (E, 1)
        target_dist:   torch.Tensor,   # (E, 1)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total  : scalar total loss
        L_CE   : scalar semantic loss component
        L_Huber: scalar metric loss component
        """
        L_CE    = self.ce(sem_logits, target_classes)
        L_Huber = self.huber(pred_dist, target_dist)
        total   = L_CE + self.lambda_metric * L_Huber
        return total, L_CE, L_Huber


# ---------------------------------------------------------------------------
# Quick shape-correctness smoke test (not a training harness)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    N, E           = 8, 15
    SEM_DIM        = 512
    HIDDEN_DIM     = 128   # 256 in production; 128 here to keep the test fast
    NUM_PRED_CLASSES = 10

    # --- Simulate Step 1 outputs ---
    node_sem        = torch.randn(N, SEM_DIM)
    node_bbox       = torch.rand(N, 4) * 100
    node_depth      = torch.rand(N, 1) * 10
    edge_index      = torch.randint(0, N, (2, E))
    edge_dist       = torch.rand(E, 1) * 5.0
    edge_conf       = torch.rand(E, 1)
    edge_angle      = torch.rand(E, 1) * 3.14
    edge_depth_diff = torch.randn(E, 1)

    data = build_scene_graph_data(
        node_sem, node_bbox, node_depth,
        edge_index, edge_dist, edge_conf, edge_angle, edge_depth_diff,
    )
    print(f"Data fields : {sorted(data.keys())}\n")

    # ---- 1. Encoder ----
    encoder = EpistemicNodeEncoder(sem_dim=SEM_DIM, hidden_dim=HIDDEN_DIM)
    mu, sigma = encoder(
        data.node_sem, data.node_bbox, data.node_depth,
        data.edge_index, data.edge_conf,
    )
    print(f"[Encoder]")
    print(f"  mu    : {tuple(mu.shape)}   (expect ({N}, {HIDDEN_DIM}))")
    print(f"  sigma : {tuple(sigma.shape)}   (expect ({N}, {HIDDEN_DIM}))")
    print(f"  sigma min = {sigma.min().item():.6f}  (must be > 0)\n")

    # ---- 2. Residual correctness on a controlled triangle ----
    # 0->1 (3m), 1->2 (4m), 0->2 (10m)
    # Only edge 0->2 has a two-hop path (0->1->2): mean = 3+4 = 7 => r = |10-7| = 3
    ei_tri = torch.tensor([[0, 1, 0], [1, 2, 2]])
    ed_tri = torch.tensor([[3.0], [4.0], [10.0]])
    r_tri  = compute_consistency_residuals(ei_tri, ed_tri, num_nodes=3)
    print(f"[Residual check -- triangle 0->1 (3m), 1->2 (4m), 0->2 (10m)]")
    print(f"  r_01 = {r_tri[0,0].item():.4f}  (expect 0.0 -- no two-hop path)")
    print(f"  r_12 = {r_tri[1,0].item():.4f}  (expect 0.0 -- no two-hop path)")
    print(f"  r_02 = {r_tri[2,0].item():.4f}  (expect 3.0 -- |10 - (3+4)|)\n")

    # ---- 3. Full forward pass ----
    model  = QuantEpiGNN(SEM_DIM, HIDDEN_DIM, NUM_PRED_CLASSES)
    out    = model(data)

    print(f"[QuantEpiGNN forward pass]")
    print(f"  sem_logits   : {tuple(out['sem_logits'].shape)}  (expect ({E}, {NUM_PRED_CLASSES}))")
    print(f"  pred_classes : {tuple(out['pred_classes'].shape)}  (expect ({E},))")
    print(f"  pred_dist    : {tuple(out['pred_dist'].shape)}  (expect ({E}, 1))")
    print(f"  pred_dist min = {out['pred_dist'].min().item():.6f}  (must be > 0 -- softplus)")
    print(f"  residuals    : {tuple(out['residuals'].shape)}  (expect ({E}, 1))")
    print(f"  mu           : {tuple(out['mu'].shape)}  (expect ({N}, {HIDDEN_DIM}))")
    print(f"  sigma        : {tuple(out['sigma'].shape)}  (expect ({N}, {HIDDEN_DIM}))\n")

    # ---- 4. Loss ----
    target_classes = torch.randint(0, NUM_PRED_CLASSES, (E,))
    target_dist    = torch.rand(E, 1) * 5.0

    loss_fn               = DualStreamLoss(lambda_metric=1.0)
    total, L_CE, L_Huber  = loss_fn(
        out["sem_logits"], target_classes,
        out["pred_dist"],  target_dist,
    )
    print(f"[DualStreamLoss]")
    print(f"  L_CE    = {L_CE.item():.4f}")
    print(f"  L_Huber = {L_Huber.item():.4f}")
    print(f"  Total   = {total.item():.4f}  (expect L_CE + 1.0 * L_Huber)\n")

    # ---- 5. Backward / gradient check ----
    total.backward()
    enc_grad   = model.encoder.mu_proj[0].weight.grad is not None
    sem_grad   = model.sem_head[0].weight.grad is not None
    metr_grad  = model.metric_head[0].weight.grad is not None
    print(f"[Backward]")
    print(f"  encoder grad  : {enc_grad}")
    print(f"  sem_head grad : {sem_grad}")
    print(f"  metric_head   : {metr_grad}")
    assert enc_grad and sem_grad and metr_grad, "Missing gradients"
    print("\nAll smoke tests passed.")
