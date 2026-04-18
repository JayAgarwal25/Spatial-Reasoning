"""
ablation_gnn.py
---------------
Ablation variants of QuantEpiGNN produced by disabling specific components.
These share all weights with QuantEpiGNN — the ablation flags are injected
at forward-pass time, not at construction time, so checkpoints are interchangeable.

Variants
--------
  use_geom_constraint : if False, consistency residuals are set to zero so
                        all message weights equal exp(0)=1 (uniform aggregation).
                        This tests whether the triangle-inequality residual
                        weighting actually helps.

  use_epistemic       : if False, sigma is zeroed out so the node representation
                        collapses to mu-only (deterministic embeddings).
                        This tests whether epistemic uncertainty modulation helps.

The full QuantEpiGNN is use_geom_constraint=True, use_epistemic=True.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

from step2_epistemic_gnn.epistemic_gnn import (
    QuantEpiGNN,
    EpistemicNodeEncoder,
    GeometricConstraintMessagePassing,
    EDGE_FEAT_DIM,
)


class AblationGNN(QuantEpiGNN):
    """
    QuantEpiGNN with runtime ablation flags.

    Args:
        use_geom_constraint : enable triangle-inequality residual weights
        use_epistemic       : enable epistemic sigma uncertainty modulation
        (all other args forwarded to QuantEpiGNN)
    """

    def __init__(
        self,
        sem_dim:             int,
        hidden_dim:          int,
        num_pred_classes:    int,
        use_geom_constraint: bool = True,
        use_epistemic:       bool = True,
    ):
        super().__init__(sem_dim, hidden_dim, num_pred_classes)
        self.use_geom_constraint = use_geom_constraint
        self.use_epistemic       = use_epistemic

    def forward(self, data) -> dict:
        # ---- Epistemic node encoding ----------------------------------------
        mu, sigma = self.encoder(
            data.node_sem, data.node_bbox, data.node_depth,
            data.edge_index, data.edge_conf,
        )
        if not self.use_epistemic:
            sigma = torch.zeros_like(sigma)   # collapse to deterministic mu

        # ---- Geometric constraint message passing (potentially ablated) -----
        mu, sigma, residuals = self._mp_forward(
            mu, sigma, data, use_geom_constraint=self.use_geom_constraint
        )

        # ---- Edge representations & heads -----------------------------------
        src, dst  = data.edge_index
        edge_repr = torch.cat([mu[src], mu[dst]], dim=-1)

        sem_logits = self.sem_head(edge_repr)

        metric_input = torch.cat([edge_repr, data.edge_dist], dim=-1)
        pred_dist    = F.softplus(self.metric_head(metric_input))

        return {
            "sem_logits"  : sem_logits,
            "pred_classes": sem_logits.argmax(dim=1),
            "pred_dist"   : pred_dist,
            "residuals"   : residuals,
            "mu"          : mu,
            "sigma"       : sigma,
        }

    def _mp_forward(self, mu, sigma, data, use_geom_constraint: bool):
        """
        Runs GeometricConstraintMessagePassing with optional residual weighting.
        When use_geom_constraint=False, residuals are zeroed before weighting
        so exp(-r)=1 for all edges.
        """
        mp: GeometricConstraintMessagePassing = self.mp
        N   = mu.size(0)
        src = data.edge_index[0]
        dst = data.edge_index[1]

        from step2_epistemic_gnn.epistemic_gnn import compute_consistency_residuals
        residuals = compute_consistency_residuals(data.edge_index, data.edge_dist, N)

        if not use_geom_constraint:
            weights = torch.ones_like(residuals)          # uniform weights
        else:
            weights = torch.exp(-residuals)               # geometry-aware weights

        edge_feat = torch.cat([
            data.edge_dist, data.edge_conf,
            data.edge_angle, data.edge_depth_diff,
        ], dim=-1)
        msg_input = torch.cat([mu[src], sigma[src], edge_feat], dim=-1)
        messages  = mp.msg_mlp(msg_input)

        weighted_msgs = messages * weights
        agg_msg   = scatter(weighted_msgs, dst, dim=0, dim_size=N, reduce='sum')
        weight_sum = scatter(weights,      dst, dim=0, dim_size=N, reduce='sum')
        agg_msg   = agg_msg / weight_sum.clamp(min=1e-8)

        mu_new = mu + mp.mu_update_mlp(agg_msg)

        mean_incoming_r = scatter(residuals, dst, dim=0, dim_size=N, reduce='mean')
        sigma_input = torch.cat([agg_msg, mean_incoming_r], dim=-1)
        sigma_new   = F.softplus(mp.sigma_update_mlp(sigma_input))

        return mu_new, sigma_new, residuals
