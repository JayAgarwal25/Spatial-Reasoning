# Project Workplan & Team Brief

## Overview
This document outlines the four major milestones for building **Quant-EpiGNN** and assigns responsibilities across the team. Each phase corresponds to a foundational step in the proposed methodology.

---

## 👨‍💻 Step 1: Zero-Shot Scene Graph Extraction (Perception Pipeline)
**Lead:** Gorang

**Objective:** Bypass heavily annotated SGG datasets by implementing a modular, zero-shot extraction pipeline.
**Key Responsibilities:**
1. **Node Localization:** Implement lightweight vision models (e.g., GroundingDINO or Florence-2) to extract 2D bounding boxes and classify objects.
2. **Geometric Candidate Filtering:** Develop depth and geometry-based filters (inspired by PRISM-0) to dynamically prune spatially impossible paired candidates (e.g., using IoU, center distance, depth difference).
3. **LLM Predicate Parsing:** Route viable object pairs through a frozen VLM for spatial captioning, passing results to an LLM to extract fine-grained, open-vocabulary predicates.

---

## 👨‍💻 Step 2: Epistemic GNN and Structured Loss (Core ML)
**Lead:** Jay

**Objective:** Handle the graph representations using an Epistemic Graph Neural Network equipped with a tailored numerical loss function.
**Key Responsibilities:**
1. **Epistemic Embeddings Formulation:** Construct the GNN framework where node features are parameterized mathematically as epistemic states (mean and variants/uncertainties) rather than deterministic vectors.
2. **Message Passing Modulation:** Implement multi-path reasoning that aggregates signals based on relational confidence logic.
3. **Structured Dual-Stream Loss Design:** Implement and tune the novel structured loss function that strictly diverges semantic variables (optimized via Cross-Entropy) from numerical and coordinate variables (optimized via Huber or Mean Absolute Error).

---

## 👨‍💻 Step 3: Recursive Visual Manipulation (The Feedback Loop)
**Lead:** Harsimar

**Objective:** Create an agentic visual grounding loop activated by structural or quantitative uncertainty.
**Key Responsibilities:**
1. **Uncertainty Trigger Mechanism:** Create a dynamic thresholding mechanism reading Epistemic GNN variance to halt the standard forward pass when uncertainty metrics are critical.
2. **Visual Agent Actions (“Drawing to Reason”):** Allow the architecture to prompt a vision model to perform drawing operations directly in visual space (e.g., plot trajectories, annotate bounding boxes, draw auxiliary geometric lines).
3. **Reflective State Management:** Establish the recursive iteration block that re-validates the newly annotated image and continuously grounds inference.

---

## 👨‍💻 Step 4: Evaluation, Benchmarking, and Ablation Complete testing
**Lead:** Mayukh

**Objective:** Validate the model computationally and establish rigorous quantitative success against existing VLMs.
**Key Responsibilities:**
1. **SpatiaLQA Evaluation:** Formulate multi-step logic/indoor reasoning tests on the SpatiaLQA benchmark; evaluate dependency-aware inference.
2. **NuScenes-SpatialQA Evaluation:** Calculate baseline vs. implemented Mean Absolute Error (MAE) for purely numerical distance predictions in open-environment domains.
3. **Ablation Studies:** Conduct robust analyses disabling specific systems (i.e., turn off the Dual-Stream loss vs regular Cross Entropy, disable the Visual Feedback drawing agent, use regular GNN vs EpiGNN) and quantify their exact impacts to demonstrate the validity of the contributions.
