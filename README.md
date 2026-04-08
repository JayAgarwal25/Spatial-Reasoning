Hello

Research Proposal: Quant-EpiGNN: Correcting Quantitative Spatial Hallucinations via Structured Loss and Active Visual Grounding
1. Introduction and Objective While Vision-Language Models (VLMs) have achieved remarkable success in generalized semantic understanding, their deployment in complex physical environments reveals a profound limitation: they consistently fail at quantitative spatial reasoning. Recent evaluations show that while models perform moderately well on qualitative questions, standard cross-entropy loss and token-level objectives are fundamentally misaligned with geometric and numerical correctness. This project proposes "Quant-EpiGNN," a hybrid neuro-symbolic architecture that addresses this exact gap by combining zero-shot scene graph generation, epistemic graph reasoning, a novel structured loss function, and an active visual feedback loop.
2. Literature Review
Zero-Shot Scene Graph Generation: Fully supervised Scene Graph Generation (SGG) models are plagued by biased datasets containing a long-tail distribution of simplistic predicates. To overcome this, recent frameworks like PRISM-0 utilize a bottom-up, zero-shot approach, leveraging frozen foundation models and geometric filters to extract open-vocabulary predicates without relying on manual annotations.
Neuro-Symbolic Graph Reasoning: Standard Graph Neural Networks (GNNs) suffer from oversmoothing and fail to systematically generalize over long inference chains. The Epistemic GNN (EpiGNN) introduces an epistemic inductive bias where node embeddings are treated mathematically as epistemic states, modulating message passing based on confidence and effectively handling multi-path relational data.
The Quantitative Spatial Bottleneck: Modern benchmarks like SpatiaLQA (containing 9,605 QA pairs for multi-step indoor reasoning) and NuScenes-SpatialQA (autonomous driving scenarios) expose the flaws of purely text-centric VLMs. NuScenes-SpatialQA explicitly demonstrates that even spatially enhanced VLMs fail heavily on quantitative spatial tasks (e.g., calculating exact distance metrics evaluated via Mean Absolute Error) and that standard Supervised Fine-Tuning (SFT) can actively degrade numerical accuracy due to loss misalignment.
Active Visual Grounding: To solve complex spatial interactions, models like Spark introduce a "drawing to reason in space" paradigm. By allowing the model to perform elementary drawing operations directly in the visual space (e.g., drawing auxiliary lines or bounding boxes), the architecture grounds its reasoning iteratively, outperforming standard static reasoning methods.
3. Proposed Methodology (The Four Major Steps)
Step 1: Zero-Shot Scene Graph Extraction (Perception Pipeline) The system will bypass heavily annotated 3D Semantic Scene Graph datasets by implementing a modular, zero-shot extraction pipeline inspired by PRISM-0.
Node Localization: A lightweight, state-of-the-art vision model like Florence-2 will identify object nodes and generate precise bounding box coordinates.
Geometric Candidate Filtering: To prevent exponential computational overhead, a geometric and knowledge filter will dynamically prune spatially impossible or semantically improbable object pairings.
Edge Generation: The remaining viable pairs will be passed to a frozen VLM to generate descriptive, localized captions, which an LLM will subsequently parse into fine-grained, open-vocabulary predicates.
Step 2: The Epistemic GNN and Structured Loss (Core ML) The extracted scene graph will be routed into an Epistemic Graph Neural Network (EpiGNN) reasoning head.
Epistemic Embeddings: Node features will not be deterministic vectors; they will be parameterized as epistemic states, encoding the VLM's semantic feature, the structural relations, and the network's uncertainty regarding specific edges.
Structured Loss Function (Novelty): To address the catastrophic failure of VLMs on quantitative tasks, the team will design a novel structured loss function. This function will explicitly decouple the processing of non-numerical semantic tokens from strict numerical and geometric data during the GNN's message-passing phase, resolving the token-level cross-entropy misalignment identified in recent literature.
Step 3: Recursive Visual Manipulation (The Feedback Loop) Static graphs are insufficient for highly ambiguous spatial tasks. The system will feature an agentic feedback loop triggered by the EpiGNN's uncertainty metrics.
Active Grounding: If the EpiGNN registers high epistemic uncertainty regarding a specific quantitative distance or structural dependency, it will recursively query the VLM.
Visual Drawing Operations: Leveraging the "drawing to reason" paradigm, the agent will prompt the VLM to actively manipulate the visual crop by annotating bounding boxes, plotting trajectories, or drawing auxiliary geometric lines to establish explicit spatial anchors before the GNN re-evaluates the relationship.
Step 4: Evaluation, Benchmarking, and Ablation The architecture must be validated against modern, rigorous benchmarks designed to test multi-step and quantitative logic, bypassing older perception-only datasets.
SpatiaLQA Evaluation: The system will be tested on the SpatiaLQA benchmark to evaluate its ability to handle ordered, dependency-aware spatial logic and precondition inference across complex indoor scenes.
SITE Evaluation:
NuScenes-SpatialQA Evaluation: The model's quantitative accuracy will be rigorously evaluated against the NuScenes-SpatialQA benchmark, calculating the Mean Absolute Error (MAE) for numerical distance predictions to prove that the structured loss function statistically reduces quantitative hallucinations compared to baseline models.
Ablation Studies: The team will ablate the structured loss module and the active drawing feedback loop to isolate and quantify their specific contributions to the overall performance increase.






































Project 1: Quant-EpiGNN: Correcting Quantitative Spatial Hallucinations via Structured Loss and Active Visual Grounding
1. Introduction and Literature Review
While Vision-Language Models (VLMs) demonstrate remarkable semantic understanding, they systematically fail at quantitative spatial reasoning tasks, such as calculating exact distances or geometric dependencies [1]. Evaluations on the NuScenes-SpatialQA benchmark reveal that while spatially-enhanced VLMs perform adequately on qualitative tasks, their quantitative accuracy (measured via Mean Absolute Error) degrades significantly, partially because standard token-level cross-entropy objectives are fundamentally misaligned with numerical geometry [1].
Concurrently, Scene Graph Generation (SGG) has evolved from biased, fully-supervised paradigms to zero-shot frameworks like PRISM-0, which leverage frozen foundation models and geometric filters to bypass computational bottlenecks [2]. However, static graphs are insufficient for multi-step logic. Recent innovations like the Epistemic GNN (EpiGNN) resolve the oversmoothing limitations of regular GNNs by treating node embeddings as epistemic states, excelling at multi-path relational reasoning . Furthermore, the "drawing to reason in space" paradigm (e.g., the Spark and VILASR models) demonstrates that allowing VLMs to actively manipulate visual spaces—such as drawing auxiliary lines and bounding boxes—drastically improves spatial comprehension when trained via reflective rejection sampling `[3]`, .
2. Granular Methodology (The Four Major Steps)
Step 1: Zero-Shot Scene Graph Extraction (Perception Pipeline)
Node Localization: The pipeline initializes with a lightweight vision model (e.g., Florence-2) to extract 2D bounding boxes and object categories.
Bottom-Up Candidate Filtering: Instead of evaluating all combinatorial pairs, the system uses PRISM-0's geometry filter to prune spatially impossible triplet candidates based on depth heuristics and bounding-box overlap thresholds [2].
LLM Predicate Parsing: Validated object pairs are passed to a VLM to generate localized textual captions. An LLM parses these into fine-grained, open-vocabulary predicates. An uncertainty-based VQA module assigns a confidence score to each edge [2].
Step 2: Epistemic GNN and Structured Loss (Core ML)
Epistemic Message Passing: The extracted graph is processed by an EpiGNN, where node embeddings encode both semantic features and structural uncertainty. Message passing is explicitly calculated across multiple relational paths. For a head node $h$ and tail node $t$, the relation vector $s$ is aggregated as $s=\psi(\{t\rightarrow,h\leftarrow\}\cup\{\phi(e\rightarrow,e\leftarrow)\mid e\in E_{h,t}\})$, where $E_{h,t}$ represents entities on the shortest paths between $h$ and $t$ ``.
Dual-Stream Structured Loss: To solve the quantitative failure, the loss function separates categorical semantic tokens from numerical coordinate data. While semantic tokens use standard cross-entropy loss, bounding box coordinates and distance estimations are explicitly penalized using a Huber or Mean Absolute Error (MAE) loss, directly addressing the misalignment identified in NuScenes-SpatialQA [1].
Step 3: Recursive Visual Manipulation (The Feedback Loop)
Uncertainty Trigger: If the EpiGNN's epistemic uncertainty for a critical numerical edge exceeds a dynamic threshold $\epsilon$, the GNN halts its forward pass.
Drawing to Reason: The system prompts the VLM to perform elementary drawing operations directly on the visual crop (e.g., annotating exact bounding boxes or drawing auxiliary geometric lines to calculate support structures) [3].
Reflective Rejection Sampling: The VLM re-evaluates this newly annotated image. During training, the system uses reflective rejection sampling to filter out hallucinations and guide the model toward optimal reasoning paths ``.
Step 4: Evaluation, Benchmarking, and Ablation
SpatiaLQA Evaluation: The system evaluates multi-step logical dependencies, specifically testing step content generation and precondition inference across 9,605 QA pairs from indoor scenes [4], [4].
NuScenes-SpatialQA Evaluation: Quantitative distance measurements and multi-view fusion are evaluated against ground-truth sensor data to validate the efficacy of the new structured loss [1], [5].



Quant-EpiGNN: Full Implementation Blueprint
1. System Overview
Pipeline:
Input Image → Object Detection → Scene Graph → Epistemic GNN → Uncertainty Detection → Visual Grounding → Final Spatial Reasoning Output
2. Environment Setup
conda create -n quant_epignn python=3.10
conda activate quant_epignn
 
pip install torch torchvision transformers sentence-transformers
pip install networkx pytorch-geometric opencv-python numpy scipy shapely matplotlib tqdm
3. Project Folder Structure
quant_epignn/
data/
models/
scene_graph/
gnn/
agent/
training/
evaluation/
utils/
4. Node Localization
Use object detection (GroundingDINO / Florence‑2).
Input: image
Output: list of objects with bounding boxes.
5. Depth Estimation
Use MiDaS to generate a depth map.
Compute mean depth within each bounding box.
6. Geometry Candidate Filtering
Filter object pairs using:
• IoU overlap
• center distance threshold
• depth difference threshold
7. Caption Generation
Crop region covering object pair.
Use a VLM (BLIP2 / LLaVA) to describe spatial relation.
Example prompt:
Describe the spatial relationship between a {object1} and {object2}.
8. Predicate Parsing
Use an LLM to convert caption → structured predicate.
Example output:
{ subject: 'cup', predicate: 'on', object: 'table' }
9. Scene Graph Representation
Graph G = (V,E)
V = detected objects
E = spatial relationships between objects
10. Node Feature Encoding
Features:
semantic embedding
bbox coordinates
depth
object size
11. Edge Feature Encoding
distance between centers
relative angle
depth difference
relation embedding
12. Epistemic Embedding
Represent node state as Gaussian distribution:
h = (mu, sigma)
mu = mean feature embedding
sigma = uncertainty estimate
13. Message Passing
Messages weighted by confidence:
confidence = 1 / variance
Higher confidence edges contribute more during aggregation.
14. Multi‑Path Reasoning
Perform multi‑hop message passing across relational paths
to infer relationships between distant objects.
15. Structured Dual‑Stream Loss
Total loss = Semantic Loss + λ * Numeric Loss
 
Semantic Loss: CrossEntropy
Numeric Loss: Huber or MAE for distance prediction
16. Uncertainty Detection
Compute mean variance per node.
If uncertainty > threshold → trigger visual grounding.
17. Visual Grounding Agent
Allow model to perform drawing actions:
• draw_bbox
• draw_line
• highlight_region
 
Re‑evaluate scene graph after annotation.
18. Recursive Loop
for iteration in range(max_iters):
    run_epignn()
    if uncertainty < threshold:
        break
    perform_visual_grounding()
19. Training Procedure
Train Epistemic GNN while perception modules remain frozen.
Use spatial reasoning datasets such as SpatiaLQA and NuScenes‑SpatialQA.
20. Evaluation Metrics
SpatiaLQA: accuracy on multi‑step reasoning tasks
NuScenes‑SpatialQA: Mean Absolute Error for distance predictions
21. Ablation Experiments
1. Remove structured loss
2. Remove drawing feedback loop
3. Replace epistemic embeddings with standard GNN
Compare performance differences.
22. Hardware
Recommended:
1× A100 or RTX4090 GPU
23. Development Timeline
Week 1: scene graph pipeline
Week 2: Epistemic GNN
Week 3: structured loss
Week 4: visual grounding
Week 5: training + evaluation

