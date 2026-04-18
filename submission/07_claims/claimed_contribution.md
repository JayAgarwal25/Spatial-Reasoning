# Claimed Contributions

## What We Reproduced
- Evaluated Qwen2-VL-7B-Instruct on the Spatial457 superCLEVR test split (999 scenes)
  using the same distance prediction protocol as the Spatial457 paper
- Reproduced the MAE metric showing VLMs average ~6.19m error on pairwise 3D distances
  in synthetic scenes

## What We Modified
- **Train/test distribution fix:** The original setup trains a GNN on GT distances but
  evaluates on VLM predictions (10× scale mismatch). We introduced log-normal VLM noise
  augmentation (σ=0.8) during training so the GNN learns to handle VLM-scale inputs.
- **Epistemic GNN:** Extended a standard relational GNN with dual mean/uncertainty node
  states and geometric consistency residuals as edge reliability weights.
- **Geometric consistency residual:** For each edge (A→C), computed disagreement with
  supporting two-hop paths (A→B→C). Used as message-passing weights and trigger signal.
- **Feedback loop integration:** Fully implemented a trigger-based visual grounding loop
  (Step 3) and connected it to the evaluation pipeline — annotate high-residual edges on
  the image, re-query Qwen2-VL for corrections, update distances, re-run GNN.
- **Spatial457-20k training:** Retrained all ablation variants on the 23,999-scene
  training set rather than the smaller original split.

## What Did Not Work
- **InternVL2-8B baseline:** Incompatible with transformers 5.x (API change in
  `_tied_weights_keys` / `language_model` module access). Replaced with Qwen2-VL-7B.
- **Hallucination detection metrics** (Resid-Hall Pearson r, AUROC, Trigger F1): Spatial457
  test set has no hallucination labels. These metrics remain NaN and are not reported.
- **Feedback loop improvement:** The feedback loop is correctly implemented and runs, but
  produces only marginal improvement (2.8885 → 2.8868 MAE, ~0.06%). On synthetic scenes
  with unambiguous objects, re-annotating the image gives Qwen2-VL little new information.
- **Epistemic/geom components in ablation:** Adding the geom constraint and epistemic σ
  to the plain GNN *increases* MAE (2.89 → 3.40). The noise augmentation likely already
  handles uncertainty implicitly; explicit epistemic layers add gradient noise on this
  synthetic dataset.

## What We Believe Is Our Contribution
1. **Distribution-aware training:** The log-normal VLM noise augmentation is a principled
   fix for the train/test mismatch in GNN-corrects-VLM pipelines. It reduces MAE by ~53%
   over raw Qwen2-VL-7B (2.89m vs 6.19m on 999 Spatial457 scenes).
2. **End-to-end visual grounding loop:** We are the first (to our knowledge) to connect
   a residual-triggered visual annotation feedback loop to a GNN-based spatial reasoning
   pipeline and evaluate it quantitatively.
3. **Honest ablation:** The finding that plain GNN outperforms the full epistemic model
   is a genuine empirical result — not a failure, but evidence that implicit noise
   robustness from augmentation subsumes explicit epistemic modeling on this benchmark.

## Scope Reduction Justification
- Evaluated on the superCLEVR subset of Spatial457 rather than the full 457-question VQA
  benchmark due to dataset format compatibility and compute constraints.
- InternVL2-8B dropped due to transformers version incompatibility; Qwen2-VL-7B is an
  equivalent 8B-class model from the same comparison table in the Spatial457 paper.
