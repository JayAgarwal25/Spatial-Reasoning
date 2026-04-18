# Prior Work Basis

## Core Papers

### 1. Spatial457: A Benchmark for Spatial Reasoning in Vision-Language Models
**Authors:** Ryan W. et al., CVPR 2025
**Influence:**
- Primary benchmark and dataset used (Spatial457-20k for training, Spatial457 for eval)
- Provided the VLM comparison table (Qwen2-VL-7B, InternVL2-8B, LLaVA-NeXT baselines)
- Motivated the quantitative spatial reasoning problem: VLMs perform well on semantic
  tasks but fail on exact distance/ordering/consistency tasks
- Justified our evaluation metrics (MAE, MRA, triangle-inequality violation rate)

### 2. Graph Neural Networks for Relational Reasoning (Gilmer et al., 2017 — MPNN)
**Influence:**
- Foundational message passing framework used in QuantEpiGNN
- Motivated residual-weighted message passing: unreliable edges should contribute less

### 3. Scene Graph Generation (Johnson et al., 2015; Xu et al., 2017)
**Influence:**
- Motivated the scene graph as intermediate representation between image and reasoning
- Informed our modular Step 1 pipeline (detection → depth → pruning → relation extraction)

### 4. Epistemic Uncertainty in Neural Networks (Kendall & Gal, 2017)
**Influence:**
- Motivated the dual mean/uncertainty (μ, σ) node representation in QuantEpiGNN
- Informed the design choice to represent node uncertainty separately from node content

### 5. Visual Grounding and Active Perception (various)
**Influence:**
- Motivated Step 3: rather than a single forward pass, iteratively annotate and re-query
  the VLM for geometrically uncertain edges
- Drew on the idea that explicit visual context (bounding boxes, distance lines) can
  focus model attention on specific spatial relationships

### 6. Triangle Inequality as a Geometric Consistency Constraint
**Influence:**
- The geometric consistency residual (Eq. 1 in paper) is directly derived from the
  triangle inequality: d(A,C) ≤ d(A,B) + d(B,C)
- Large residuals indicate edges that violate multi-hop path consistency
