# Project Workplan & Team Brief

## Overview

This document outlines the four major milestones for this project and assigns responsibilities across the team. Each phase corresponds to a foundational step in the proposed methodology.

The core claim of the project: **VLM spatial hallucinations are often not just locally uncertain — they are globally geometrically inconsistent.** A pair of distances can each look plausible in isolation but violate basic triangle-inequality constraints when viewed jointly. The GNN's job is to detect this inconsistency via geometric consistency residuals and trigger targeted visual grounding on the conflicting edges.

---

## 👨‍💻 Step 1: Zero-Shot Scene Graph Extraction (Perception Pipeline)

**Lead:** Gorang

**Objective:** Bypass heavily annotated SGG datasets by implementing a modular, zero-shot extraction pipeline.

**Key Responsibilities:**

1. **Node Localization:** Implement a lightweight vision model (e.g., GroundingDINO or Florence-2) to extract 2D bounding boxes and classify objects.

2. **Geometric Candidate Filtering:** Develop depth and geometry-based filters (inspired by PRISM-0) to dynamically prune spatially impossible object pairs using IoU, center-distance thresholds, and depth difference thresholds.

3. **LLM Predicate Parsing:** Route viable object pairs through a frozen VLM for spatial captioning. Pass results to an LLM to extract fine-grained, open-vocabulary predicates. Assign a VLM confidence score `c ∈ [0,1]` to each edge — this confidence is a first-class output consumed directly by Step 2.

**Interface contract (output to Step 2):**

A scene graph `G = (V, E)` in PyTorch Geometric `Data` format with named fields:

- `node_sem` (N, sem_dim) — semantic embedding per node
- `node_bbox` (N, 4) — bounding box `[x, y, w, h]`
- `node_depth` (N, 1) — mean depth estimate from MiDaS
- `edge_index` (2, E)
- `edge_dist` (E, 1) — VLM-predicted metric distance
- `edge_conf` (E, 1) — VLM confidence score
- `edge_angle` (E, 1) — relative angle between object centroids
- `edge_depth_diff` (E, 1) — depth difference between objects

---

## 👨‍💻 Step 2: Epistemic GNN and Geometric Consistency Loss (Core ML)

**Lead:** Jay

**Objective:** Enable selective visual grounding via geometric consistency verification — detect which specific VLM spatial claims are mutually contradictory across the scene graph, and pass per-edge consistency residuals to Step 3 to trigger targeted regrounding only on conflicting edges.

### Core novelty

Standard epistemic uncertainty (MC dropout, per-node variance) measures *local* prediction confidence and cannot identify which edges conflict with each other. The key insight here is that VLM metric hallucinations are often *globally* inconsistent — `d(A,C)` as directly claimed by the VLM conflicts with what `d(A,B) + d(B,C)` demands geometrically. The consistency residual is a deterministic, interpretable signal that identifies *which specific claims* are contradictory, enabling targeted regrounding rather than uniform re-querying.

**Limitation to validate:** A high residual can arise from a VLM hallucination *or* from a noisy depth estimate corrupting edge features. These are not yet separated. The key publication-strength experiment is: on images with ground-truth hallucination labels, does residual correlate with hallucination rate specifically, or equally with depth noise?

**Key Responsibilities:**

1. **Epistemic Node Encoder**

   Encode each node as a `(mu, sigma)` pair — both are vectors, not scalars.

   - `mu`: learned projection of `[node_sem ∥ node_bbox ∥ node_depth]` through a 2-layer MLP.
   - `sigma`: same input features concatenated with an `uncertainty_seed` scalar, passed through a 2-layer MLP with softplus output.
   - `uncertainty_seed` for node `v` = `mean(1 - c)` over **incoming** edges to `v` (directed: edges where `v` is the target). Nodes with no incoming edges default to seed = 1.0 (maximally uncertain).

   Rationale for incoming-only: incoming edges represent VLM claims *about* this node as a spatial target. Outgoing edges reflect this node's reliability as an anchor for others — that signal belongs in message passing weights, not the seed.

   **Design choice, not claimed contribution:** The (mu, sigma) encoder only earns its place if sigma influences message passing routing in a way that changes residuals and re-grounding outcomes. Ablation A3 tests this directly. If deterministic embeddings produce equivalent residuals, the encoder should be simplified.

2. **Geometric Constraint Message Passing**

   For every direct edge `(A→C)` in the graph, enumerate all two-hop paths `A→B→C` through intermediate nodes `B`. Compute the **consistency residual**:

   ```
   r_AC = | d_AC_direct  −  mean_B( d_AB + d_BC ) |
   ```

   This residual measures how much the VLM's direct claim about `(A, C)` conflicts with what triangle-inequality geometry demands given the surrounding edges.

   Message aggregation weights incoming messages by `exp(−r)` — geometrically consistent edges (low residual) contribute more; conflicting edges are down-weighted automatically. This is a fixed geometric prior; Ablation A5 compares it against learned attention gating to determine whether a data-driven routing function is needed.

   For edges with no two-hop path, `r = 0` (no inconsistency detectable from graph structure alone).

3. **Uncertainty Output Head**

   After message passing, each edge exposes its consistency residual `r` as the uncertainty signal. This is passed directly to Step 3 as the trigger. No MC dropout, no stochastic forward passes — the residual is deterministic and geometrically grounded.

4. **Dual-Stream Prediction and Loss**

   Two output heads on the final node/edge representations:

   - **Semantic head:** predicts predicate class → Cross-Entropy loss `L_CE`
   - **Metric head:** predicts refined distance estimate → Huber loss `L_Huber`

   Total loss: `L = L_CE + λ * L_Huber`, where `λ = 1.0` as default (tunable).

   Note: this is a standard multi-task loss. The novelty is not in the loss form but in the GNN's consistency residual mechanism that produces geometrically-grounded uncertainty.

**Interface contract (output to Step 3):**

- Refined distance predictions per edge
- Per-edge consistency residuals `r` (E, 1)
- Step 3 reads `r` directly to decide which edges to trigger visual grounding on

---

## 👨‍💻 Step 3: Recursive Visual Manipulation (The Feedback Loop)

**Lead:** Harsimar

**Objective:** Create an agentic visual grounding loop activated by the geometric consistency residuals from Step 2, with a typed action mapping from uncertainty type to grounding operation.

**Key Responsibilities:**

1. **Uncertainty Trigger Mechanism:** Read per-edge residuals `r` from Step 2. If `r > ε` for any edge, halt the forward pass and identify the conflicting edge(s). The threshold `ε` is not a fixed default — it must be calibrated per dataset on a validation set using a precision-recall curve for re-grounding decisions.

2. **Typed Visual Agent Actions:** Map uncertainty type to a specific drawing action — not undifferentiated "draw something":

   - High residual on a **positional edge** → `draw_bbox` to re-anchor object location
   - High residual on a **relational edge** → `draw_line` between object centroids to make the geometry explicit
   - High residual on a **depth-sensitive edge** → request a depth-re-estimation crop

3. **Reflective State Management:** Re-run the scene graph extractor on the newly annotated image. Pass the updated graph back to Step 2. Iterate until all residuals fall below `ε` or `max_iters` is reached.

---

## 👨‍💻 Step 4: Evaluation, Benchmarking, and Ablation

**Lead:** Mayukh

**Objective:** Validate the model quantitatively and establish rigorous comparison against VLM baselines.

**Key Responsibilities:**

1. **SpatiaLQA Evaluation:** Multi-step logical dependency tests across 9,605 indoor-scene QA pairs; evaluate dependency-aware precondition inference.

2. **NuScenes-SpatialQA Evaluation:** Calculate Mean Absolute Error (MAE) for numerical distance predictions against ground-truth sensor data, comparing our approach against VLM baselines.

3. **Ablation Studies (mandatory per eval scheme):**

   - **A1:** Residual integrated into GNN message passing (current system) vs. residual computed as a post-hoc filter applied directly to raw VLM outputs without GNN aggregation. Key question: does the GNN propagation do real work, or is the raw residual sufficient standalone?
   - **A2:** Disable the visual feedback loop entirely. How much does the grounding loop contribute?
   - **A3:** Replace EpiGNN with a standard GNN (deterministic node embeddings, no sigma). Does sigma routing in message passing change residuals and re-grounding outcomes? If not, simplify the encoder.
   - **A4:** Remove the Huber metric loss — use CE only. Validates the multi-task loss contribution.
   - **A5:** Replace exp(-r) consistency weighting with learned attention gating on edges. Tests whether the fixed geometric prior for edge weighting is adequate or whether a data-driven routing function improves performance.
