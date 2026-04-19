\documentclass[11pt]{article}
\usepackage{acl}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{times}
\usepackage{latexsym}
\usepackage{microtype}
\usepackage{inconsolata}

\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{enumitem}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{url}
\usepackage{array}
\usepackage{caption}
\usepackage{tabularx}
\usepackage{adjustbox}
\usepackage{makecell}
\usepackage{placeins}

\setlength{\abovecaptionskip}{5pt}
\setlength{\belowcaptionskip}{0pt}
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\floatsep}{10pt plus 2pt minus 2pt}
\setlength{\intextsep}{10pt plus 2pt minus 2pt}
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{5pt}
\captionsetup{font=small, labelfont=bf}

\title{Quant-EpiGNN: Detecting Quantitative Spatial Inconsistencies in Vision-Language Outputs via Residual-Aware Graph Reasoning}

\author{
\textbf{Jay Agarwal} \quad
\textbf{Gorang Mehrishi} \quad
\textbf{Mayukh Khetan} \quad
\textbf{Harsimar Singh Saluja} \\
\textbf{Birla Institute of Technology and Science, Pilani}
}

\begin{document}
\maketitle

\begin{abstract}
Vision-language models are strong at natural scene description, but they remain weak at precise spatial reasoning. Their outputs can sound plausible while still violating geometric consistency across distances or multi-object relationships.

We present Quant-EpiGNN, a modular framework for detecting and refining such inconsistencies through graph-based reasoning. The system first constructs a scene graph from image-level predictions. It then computes a geometric consistency residual, which measures disagreement between a direct pairwise estimate and indirect evidence available through multi-hop paths. This residual is used to weight message passing in a graph neural network, allowing the model to downweight unreliable spatial estimates during refinement.

On Spatial457 \cite{wang2025spatial457}, our best model reduces mean absolute error from 6.1924 meters to 2.8885 meters, a 53.4\% improvement over a Qwen2-VL-7B-Instruct baseline \cite{wang2024qwen2vl}. At the same time, ablation results show that the more complex variants with epistemic parameterization and additional geometric modules do not outperform a simpler plain GNN under the current synthetic noise setting. The main validated conclusion is therefore narrower than the full project vision: explicit graph reasoning is effective for refining noisy spatial predictions, but the additional uncertainty-aware machinery remains unproven in the present setup.
\end{abstract}

\section{Introduction}

Vision-language models (VLMs) have achieved strong performance in captioning, visual question answering, grounded retrieval, and instruction-following. Despite this progress, quantitative spatial reasoning remains a persistent weakness. A model may correctly identify objects and describe qualitative relations, yet still fail when asked to reason about exact distances, ordering consistency, or multi-step spatial dependencies.

This gap is structural rather than incidental. Spatial outputs are coupled. If a model predicts distances between several object pairs, those predictions should obey geometric constraints such as triangle inequalities. If it predicts ordering relations such as left/right or front/behind, those relations should remain coherent with object positions and depth cues. In practice, general-purpose VLMs are trained primarily with token-level objectives that reward local plausibility rather than global spatial consistency~\cite{kamath2023whatsup}. As a result, they often produce outputs that are individually reasonable but jointly inconsistent.

This work studies whether explicit relational structure can be injected into the pipeline in a modular and interpretable way. Rather than treating spatial outputs as isolated predictions, we construct a scene graph over detected objects and reason over that graph using message passing. The central signal in our approach is a geometric consistency residual, which compares a direct edge estimate against alternative multi-hop evidence in the graph. If a direct pairwise estimate strongly disagrees with supporting indirect evidence, it should be treated as unreliable.

Our broader project vision included four stages: scene graph construction, residual-aware graph reasoning, a trigger-based corrective feedback loop, and final evaluation. All four stages are implemented and evaluated. The graph reasoning stage is the central contribution, achieving a 53.4\% MAE reduction. The feedback loop was also evaluated on all 999 test scenes but produced only marginal improvement (2.8885\,m $\to$ 2.8868\,m, $<$0.1\%), and is therefore reported as a secondary finding rather than a main claim.

The main question answered here is therefore narrower than the original project plan: \emph{can residual-aware graph reasoning substantially reduce quantitative spatial error in noisy VLM outputs?} On Spatial457 \cite{wang2025spatial457}, the answer is yes. However, the stronger claims --- that explicit epistemic parameterization outperforms simpler graph baselines, or that the feedback loop delivers large gains --- are not supported by the present evidence.

\paragraph{Contributions.}
Our contributions in this report are as follows:
\begin{itemize}[leftmargin=*]
    \item We implement a modular pipeline that converts image-level predictions into a scene graph with node semantics and pairwise geometric attributes.
    \item We define a geometric consistency residual that measures disagreement between direct pairwise estimates and supporting multi-hop paths.
    \item We integrate this residual into graph message passing through residual-derived edge weights for distance refinement.
    \item We evaluate the method on Spatial457 \cite{wang2025spatial457} and show a 53.4\% reduction in distance MAE relative to a strong Qwen2-VL-7B-Instruct baseline \cite{wang2024qwen2vl}.
    \item We evaluate the trigger-based feedback loop end-to-end on 999 test scenes and find only marginal additional improvement ($<$0.1\%), providing an honest bound on what iterative visual re-querying adds in a synthetic setting.
    \item We provide ablations showing that, under the current synthetic noise regime, a plain GNN matches or outperforms more complex epistemic and geometric-constraint variants.
\end{itemize}

\section{Related Work}
\label{sec:related}

\subsection{Spatial reasoning in vision-language models}

Large vision-language models have improved substantially on general multimodal tasks, but quantitative spatial reasoning remains brittle. Standard token-level objectives reward local plausibility rather than global spatial consistency, meaning a model may answer individual pairwise distance questions reasonably while jointly violating geometric constraints across the scene~\cite{kamath2023whatsup}. The Spatial457 benchmark~\cite{wang2025spatial457} exposes this directly: given exact 3D object coordinates, model distance errors are often globally inconsistent rather than merely imprecise, violating triangle inequalities even when no single prediction looks obviously wrong. This coupling between spatial outputs is the core problem our work targets. We use Qwen2-VL~\cite{wang2024qwen2vl} as our baseline, as it represents a strong modern multimodal model whose raw distance predictions exhibit this same inconsistency at scale.

\subsection{Scene graphs and modular perception pipelines}

Scene graphs provide a natural intermediate representation for injecting relational structure into spatial reasoning. Traditional scene graph generation relied on heavily annotated datasets with long-tail predicate imbalance, causing supervised models to default to generic relations in open-world settings~\cite{elskhawy2025prism0}. PRISM-0~\cite{elskhawy2025prism0} bypasses this through a zero-shot modular pipeline: a geometric filter prunes candidate pairs, a frozen VLM generates localized captions, and an LLM parses these into open-vocabulary predicates with per-edge confidence scores. Open-vocabulary detectors such as Grounding DINO~\cite{groundingdino2023} underpin such pipelines by providing zero-shot bounding box localization without category-specific supervision. Our Step 1 follows this modular philosophy, with VLM confidence scores passed as first-class edge attributes into the graph reasoning stage.

\subsection{Graph neural networks for relational refinement}

Graph neural networks are a natural fit for problems where outputs depend on neighborhood structure and multi-hop relationships~\cite{gilmer2017mpnn}. Standard GNNs, however, suffer from oversmoothing and fail to generalize to inference chains longer than those seen during training~\cite{epignn2025}. EpiGNN~\cite{epignn2025} addresses this by parameterizing node embeddings as epistemic states that encode both semantic content and structural uncertainty, modulating message passing by edge confidence rather than averaging uniformly. Our GNN is motivated by this epistemic framing, but the central mechanism we contribute is different: a deterministic geometric consistency residual that measures disagreement between a direct pairwise estimate and indirect multi-hop evidence, used to derive edge weights during message passing. Crucially, our ablations show that under the current synthetic noise regime, a plain GNN matches or outperforms more complex epistemic variants, so we make no strong claim about the epistemic parameterization itself pending further evaluation.

\subsection{Active and iterative visual grounding}

When individual predictions are unreliable, a natural response is to revisit them rather than propagate their errors. The ReAct framework~\cite{react2022} formalizes this as interleaving reasoning traces with grounded actions, showing substantial gains over reasoning or acting alone. The VILASR architecture~\cite{vilasr2025} brings this principle into the visual domain through a \textit{drawing to reason in space} paradigm, where a model annotates bounding boxes and auxiliary lines directly in the visual field before re-evaluating spatial relationships. Our Step 3 implements a typed-action variant of this loop, where high-residual edges trigger \texttt{draw\_bbox}, \texttt{draw\_line}, or depth re-estimation crops depending on edge type, and the annotated image is re-queried before the graph is updated. This component is fully implemented and evaluated on 999 test scenes. The evaluation shows only marginal improvement over the GNN alone (2.8885\,m $\to$ 2.8868\,m), which we attribute to the GNN already correcting most structural error and synthetic superCLEVR scenes providing little new information from re-annotation.

\section{Methodology}
\label{sec:method}

\subsection{System overview}

The system is organized as a four-stage pipeline:
\begin{center}
\fbox{
\begin{minipage}{0.95\linewidth}
\centering
Image $\rightarrow$ Scene Graph Construction $\rightarrow$ Residual-Aware Graph Reasoning $\rightarrow$ Trigger-Based Feedback
\end{minipage}
}
\end{center}

The first stage constructs a structured representation from image-level predictions. The second stage computes a residual that captures geometric inconsistency over the graph. The third stage refines pairwise spatial estimates through residual-aware message passing. The optional fourth stage revisits uncertain predictions through targeted visual feedback.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.90\textwidth]{Gemini_Generated_Image_c55t91c55t91c55t.png}
    \caption{Overall Quant-EpiGNN pipeline. The system first constructs a scene graph from image-level perception, then performs residual-aware graph reasoning, and finally applies an optional trigger-based visual grounding loop for targeted correction of unreliable edges. The completed benchmark-scale results in this paper are centered on the graph reasoning stage.}
    \label{fig:pipeline}
\end{figure*}

The present report centers its empirical claims on the graph reasoning stage, because that is the component with complete benchmark-scale results.

\subsection{Stage 1: Scene graph construction}

\paragraph{Overview.}
We construct a scene graph as an intermediate representation that captures object-level semantics and pairwise spatial structure. This stage is intended to expose dependencies among pairwise spatial estimates so that downstream reasoning can refine them jointly rather than independently.

\paragraph{Object nodes.}
Given an input image, we first detect objects and represent each object as a node:
\[
v_i = (\ell_i, b_i, c_i, a_i),
\]
where $\ell_i$ is the semantic label, $b_i$ is the bounding box, $c_i$ is the box center, and $a_i$ is the area.

The critical assumption for this report is that each object is localized and assigned a label. The specific detector identity is not the central experimental variable in the reported results. The scene-graph abstraction only requires object-level predictions from the front end.

\paragraph{Pairwise edges.}
For each candidate pair $(i,j)$, we construct a directed edge with spatial information:
\[
e_{ij} = (d_{ij}, \theta_{ij}, \Delta x_{ij}, \Delta y_{ij}, s_{ij}),
\]
where $d_{ij}$ is the initial pairwise distance estimate, $\theta_{ij}$ is the relative angle, $(\Delta x_{ij}, \Delta y_{ij})$ are normalized relative offsets, and $s_{ij}$ is an edge score or confidence.

In our implemented pipeline, edge initialization is based on the available pairwise geometric information used by the refinement model. Although the broader project design considered richer relation extraction and additional perception cues, the completed quantitative evaluation in this paper is centered on distance refinement over the constructed graph.

\paragraph{Why a graph representation?}
A graph representation is useful because it makes shared structure explicit. If object $A$ is close to $B$ and $B$ is close to $C$, then $A$ and $C$ cannot be arbitrarily far apart. Such coupling is invisible when each pairwise estimate is treated independently. The scene graph therefore serves as the interface between noisy front-end predictions and downstream relational refinement.

\subsection{Stage 2: Geometric consistency residual}

The central idea of the method is that a direct pairwise estimate should be checked against indirect evidence available through other nodes in the graph.

For a direct edge $(i,j)$, let $\mathcal{N}_{ij}$ denote the set of intermediate nodes $k$ such that both $(i,k)$ and $(k,j)$ are present in the graph. We define the geometric consistency residual:
\begin{equation}
r_{ij} =
\left|
d_{ij}
-
\frac{1}{|\mathcal{N}_{ij}|}
\sum_{k \in \mathcal{N}_{ij}}
(d_{ik} + d_{kj})
\right|.
\label{eq:residual}
\end{equation}

If $\mathcal{N}_{ij}$ is empty, we default the residual to zero in the current implementation. This is a practical choice rather than a theoretically ideal one. It avoids undefined behavior, but it also means that edges in sparse parts of the graph remain weakly constrained.

\paragraph{Interpretation.}
The residual measures how much a direct distance prediction disagrees with the average indirect evidence provided by two-hop paths. A large residual indicates that the direct estimate is structurally suspicious. This is not a proof of error, but it is a useful signal for reducing the trust placed on that edge during message passing.

\paragraph{Why two-hop paths?}
Two-hop paths provide the simplest nontrivial structural check. They are cheap to compute, align naturally with triangle-like consistency intuition, and avoid the instability of relying on longer noisy paths.

\subsection{Stage 3: Residual-aware graph reasoning}

\paragraph{Node features.}
Each node is embedded as
\[
h_i^{(0)} = [\text{emb}(\ell_i) \,\Vert\, \text{geom}(b_i, c_i, a_i)],
\]
where $\text{emb}(\ell_i)$ is a semantic embedding for the object label and $\text{geom}(\cdot)$ encodes geometric attributes.

\paragraph{Edge-aware messages.}
For each directed edge $(i,j)$, we compute a message:
\begin{equation}
m_{ij} = w_{ij} \cdot \phi\big(h_i^{(t)}, e_{ij}\big),
\label{eq:message}
\end{equation}
where $\phi$ is a learned message function and $w_{ij}$ is the residual-derived edge weight.

We define
\begin{equation}
w_{ij} = \exp(-r_{ij}),
\label{eq:weight}
\end{equation}
so edges with larger residuals contribute less during aggregation.

\paragraph{Aggregation and update.}
Messages are aggregated at each node:
\begin{equation}
M_j = \sum_{i \in \mathcal{P}(j)} m_{ij},
\end{equation}
where $\mathcal{P}(j)$ denotes the incoming neighbors of node $j$. The node state is then updated through a learned transformation:
\begin{equation}
h_j^{(t+1)} = \psi(h_j^{(t)}, M_j).
\end{equation}

\paragraph{Prediction head.}
After several message-passing layers, the model predicts refined distances $\hat{d}_{ij}$ for the relevant edges.

\subsection{Epistemic parameterization}

Our initial system design included an epistemic graph neural network in which node states are represented using both a mean component $\mu$ and an uncertainty-like component $\sigma$. Concretely, the implemented variant includes separate learned channels corresponding to feature content and confidence modulation.

However, this should not be interpreted as a fully probabilistic or calibrated epistemic model. It is better described as a practical uncertainty-aware parameterization layered on top of graph aggregation. The ablations show that this additional structure does not help under the current dataset and noise setting, and can even hurt performance.

\subsection{Training objective}

The implemented model is trained using a combination of relation supervision and distance regression:
\begin{equation}
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{Huber}},
\label{eq:loss}
\end{equation}
where $\mathcal{L}_{\text{CE}}$ is a cross-entropy term on relation-class outputs and $\mathcal{L}_{\text{Huber}}$ is a Huber loss on distance prediction. In the final experiments, $\lambda = 1.0$.


\subsection{Stage 4: Trigger-based visual grounding loop}

The broader pipeline includes a corrective loop for handling uncertain predictions, inspired by iterative reasoning–action frameworks such as ReAct \cite{react2022}. After graph reasoning, edges with large residuals are flagged:

\[
r_{ij} > \epsilon
\]

These edges correspond to spatial relationships that are likely unreliable. For such cases, the system performs targeted visual grounding by applying simple image-level annotation primitives, such as drawing bounding boxes or connecting objects with lines. These operations are consistent with grounding-based perception approaches such as GroundingDINO \cite{groundingdino2023}, and also align with recent work that explores drawing-based interaction for improving spatial reasoning in vision-language models \cite{vilasr2025}.

After applying these annotations, the perception and graph reasoning stages are re-executed. This creates an iterative refinement loop in which the model selectively revisits uncertain regions instead of treating initial predictions as fixed. Conceptually, this follows a reason–act–observe paradigm, where reasoning identifies uncertainty, actions modify the visual input, and updated observations improve subsequent predictions \cite{react2022}.

This component is implemented and evaluated on all 999 Spatial457 test scenes. The loop converges in 1--2 iterations in practice. However, the measured improvement over the GNN alone is marginal: MAE decreases from 2.8885\,m to 2.8868\,m ($<$0.1\%). We report this result but do not count it as a primary contribution, as the gain is within noise for this benchmark and synthetic setting.

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{Figures/visual_grounding_multiple.png}
    \caption{Visual grounding feedback applied to the top-3 high-residual edges per scene. The original scene (left) is annotated with bounding boxes, predicted distances, and residuals (right) to provide targeted corrective feedback for uncertain spatial relationships.}
    \label{fig:visual_feedback}
\end{figure*}

\section{Experimental Setup}
\label{sec:setup}

\subsection{Dataset}

We evaluate on \textbf{Spatial457} \cite{wang2025spatial457}, using a superCLEVR-style synthetic setup. The full Spatial457-20k corpus contains 23,999 scenes; after a 95\%/5\% train/validation split, 22,799 scenes are used for training and 1,200 for validation. The held-out benchmark test split used for all reported results contains 999 scenes.

This dataset is suitable for our problem for two reasons. First, it provides controlled spatial structure with ground-truth distance information. Second, it allows us to isolate the effect of graph reasoning under noisy spatial predictions without conflating the analysis with uncontrolled real-world perception noise.

\subsection{Baseline and noise model}

We use \textbf{Qwen2-VL-7B-Instruct} \cite{wang2024qwen2vl} as the vision-language baseline for spatial predictions. This model is run locally to avoid rate-limit instability and keep the evaluation environment controlled. Our original plan used InternVL2-8B, which appears in the Spatial457 comparison table \cite{wang2025spatial457}, but that model is incompatible with \texttt{transformers} 5.x due to API changes in tied-weight handling. Qwen2-VL-7B is an equivalent 8B-class model from the same comparison table and serves as a direct replacement.

To train the GNN as a refinement model, we augment the training data with log-normal VLM noise using $\sigma = 0.8$. The implemented mixture applies moderate corruption to most examples and more severe corruption to a minority of cases. This is intended to simulate noisy spatial predictions from a VLM front end.

\subsection{Training configuration}

The final training setup is:
\begin{itemize}[leftmargin=*]
    \item Optimizer: Adam
    \item Learning rate: $3 \times 10^{-4}$
    \item Batch size: 32
    \item Epochs: 100 with early stopping on validation loss
    \item Hidden dimension: 256
    \item Loss: Cross-entropy + Huber with $\lambda = 1.0$
\end{itemize}

The train/validation split follows the 95\%/5\% scheme described in Section~\ref{sec:setup}.1.

\begin{figure*}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{Figures/training_curves.png}
    \caption{Training and validation loss curves over 100 epochs on Spatial457. The plain GNN and the variants without specific constraints demonstrate stable convergence under the log-normal noise augmentation.}
    \label{fig:training_curves}
\end{figure*}

\subsection{Hardware}

All experiments were run on a remote GPU server with:
\begin{itemize}[leftmargin=*]
    \item GPU: NVIDIA RTX 6000 Ada Generation (48 GB VRAM)
    \item CPU: 128 cores
    \item RAM: 503 GB
\end{itemize}

\subsection{Evaluation metrics}

We report the following metrics.

\paragraph{Mean Absolute Error (MAE).}
This is the primary metric for distance prediction quality. Lower is better.

\paragraph{Mean Relative Accuracy (MRA).}
MRA is averaged over three relative-error thresholds $\tau \in \{0.25, 0.50, 1.00\}$: at each threshold, it measures the fraction of pairs where $|{\hat{d} - d^*}| / d^* \leq \tau$, then averages across thresholds. Higher is better. This metric is computed over GNN-refined predictions and is not defined for the raw VLM baseline, which is why that column is left blank in Table~\ref{tab:main-results}.

\paragraph{Triangle-violation rate.}
This metric measures how often the predicted distances violate triangle-inequality constraints. Lower is better.

\paragraph{Mean residual.}
We also report the average geometric consistency residual computed from multi-hop disagreement.

Not all originally planned metrics could be computed. In particular, residual-to-hallucination correlation, AUROC, and trigger F1 require hallucination labels that are not available in the current Spatial457 test setup.

\section{Results}
\label{sec:results}

\subsection{Main quantitative result}

\begin{table}[t]
\centering
\small
\caption{Main result on Spatial457. The best graph model reduces MAE by 53.4\% relative to the VLM baseline.}
\label{tab:main-results}
\vspace{2pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{MAE (m)$\downarrow$} & \textbf{MRA$\uparrow$} & \textbf{Tri-Viol$\downarrow$} & \textbf{Mean Resid.$\downarrow$} \\
\midrule
Qwen2-VL-7B baseline & 6.1924 & --$^\dagger$ & 0.0530 & 3.8962 \\
GNN (plain, ours) & \textbf{2.8885} & \textbf{0.3726} & 0.0530 & 3.8962 \\
GNN + feedback loop & 2.8868 & \textbf{0.3726} & 0.0530 & 3.8962 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\par\smallskip
{\footnotesize $^\dagger$MRA is computed over GNN-refined predictions and is not defined for the raw VLM baseline.}
\end{table}

Table~\ref{tab:main-results} compares the VLM baseline, the best GNN model, and the GNN augmented with the trigger-based feedback loop on the 999-scene Spatial457 test split.

The headline result is strong: graph reasoning reduces MAE from 6.1924\,m to 2.8885\,m. This is a large reduction in quantitative error and demonstrates that relational refinement is genuinely useful. Adding the feedback loop yields a further marginal reduction to 2.8868\,m ($<$0.1\%), which we discuss in Section~\ref{sec:failure}.

However, the result must be interpreted carefully. The triangle-violation rate remains unchanged at 0.0530, and the mean residual also remains unchanged in the final evaluation summary. This means the gains are not coming from full enforcement of global metric consistency. Instead, the GNN is improving per-edge estimates through relational aggregation while still operating within the structural limitations of the current setup.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{Figures/main_mae_chart.png}
    \caption{Distance prediction MAE on the Spatial457 test split. The plain GNN achieves a 53.4\% reduction in error compared to the Qwen2-VL-7B baseline. The feedback loop adds marginal further improvement ($<$0.1\%).}
    \label{fig:main_mae}
\end{figure}

\subsection{Ablation study}

\begin{table}[t]
\centering
\small
\caption{Ablation study on Spatial457. The plain GNN and the no-epistemic variant perform best under the current setup.}
\label{tab:ablation}
\vspace{2pt}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lcccc}
\toprule
\textbf{Variant} & \textbf{Best Ep.} & \textbf{Val Loss} & \textbf{MAE (m)$\downarrow$} & \textbf{MRA$\uparrow$} \\
\midrule
Full (geom + epi) & 60$^\ddagger$ & 0.0384 & 3.4048 & 0.3336 \\
No geom constraint & 77 & 0.0386 & 3.2050 & 0.3197 \\
No epistemic $\sigma$ & 68 & 0.0385 & \textbf{2.8885} & \textbf{0.3726} \\
Plain GNN & 68 & \textbf{0.0382} & \textbf{2.8885} & \textbf{0.3726} \\
\bottomrule
\end{tabular}
\end{adjustbox}
\par\smallskip
{\footnotesize $^\ddagger$Training was terminated externally (SIGTERM) at epoch 71; the reported checkpoint is the best validation checkpoint saved at epoch 60.}
\end{table}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\columnwidth]{Figures/ablation_chart.png}
    \caption{Ablation study comparing geometric constraints and epistemic uncertainty modules. Under the current setup, the simpler architectures (Plain GNN and No epistemic component) achieve the lowest MAE.}
    \label{fig:ablation_chart}
\end{figure}

To identify which components matter, we compare four variants against the same Qwen2-VL baseline. The most important negative result is that the full Quant-EpiGNN variant does not outperform the simpler baselines. In fact, the best performance is achieved by the plain GNN and the no-epistemic variant, both reaching 2.8885\,m MAE. Adding the geometric-constraint module and epistemic parameterization increases test error.

This forces a correction to the original project intuition. The more elaborate model is not automatically better. Under the current synthetic noise regime, additional uncertainty-aware structure appears to introduce optimization difficulty without providing compensating robustness.

\subsection{Interpretation of the ablation result}

There are at least three plausible reasons for this outcome. First, the log-normal noise augmentation may already encode the dominant uncertainty pattern seen during training. If so, the explicit epistemic channel becomes redundant. Second, the synthetic benchmark may not contain the kind of heteroscedastic uncertainty for which an uncertainty-aware formulation is beneficial. Third, the geometric-constraint mechanism may be too weakly aligned with the evaluation objective. The model is trained primarily to reduce distance error, and the structural guidance may therefore act as a regularizer that hurts rather than helps.

\subsection{What the result does and does not show}

The current evidence supports the following claim: a graph refinement model operating on noisy spatial predictions can substantially reduce distance error relative to a strong VLM baseline.

It does not support the following stronger claims:
\begin{itemize}[leftmargin=*]
    \item that explicit epistemic parameterization improves performance in this setting,
    \item that the residual module materially reduces triangle-violation rate in the current evaluation,
    \item that the feedback loop produces meaningful improvement beyond marginal gains in this synthetic setting.
\end{itemize}

\section{Discussion}
\label{sec:discussion}

\subsection{Structural vs.\ perceptual errors}

The results expose an important distinction between structural errors and perceptual errors. Structural errors arise when pairwise predictions are individually plausible but mutually inconsistent. These are the errors graph reasoning is most suited to address. Perceptual errors arise when the underlying object-level information is itself wrong. No downstream GNN can fully repair a missing object, a poor localization, or a corrupted front-end signal.

Our improvements suggest that a substantial part of the baseline error is structural and therefore recoverable by relational aggregation. At the same time, the unchanged triangle-violation rate indicates that the current formulation is not yet strong enough to enforce truly global consistency.

\subsection{Why does the plain GNN win?}

A common mistake is to assume that more structured modeling must dominate simpler baselines. The ablation disproves that assumption here. The plain GNN likely wins because it captures the most useful inductive bias available in this setting, namely that neighboring edges provide denoising signal, without overcomplicating optimization.

\subsection{Residuals as diagnostics vs.\ residuals as constraints}

Another lesson is that the residual signal currently functions better as a diagnostic than as a hard constraint. It helps identify suspicious predictions and define trust weights, but it does not by itself force the final output to satisfy metric consistency. To move from residual-guided refinement to actual consistency enforcement, future work would need stronger global constraints or structured decoding over the graph.

\section{Failure Cases and Error Analysis}
\label{sec:failure}

A serious report requires explicit failure analysis. We identify five major failure modes.

\paragraph{Sparse graph support.}
The residual in Equation~\ref{eq:residual} depends on the availability of two-hop paths. If an edge lies in a sparse region of the graph, there may be little or no supporting indirect evidence.

\paragraph{Noisy intermediate edges.}
Even when multi-hop paths exist, they may themselves be wrong. If indirect distances are corrupted, the residual can become misleading.

\paragraph{Objective mismatch.}
The evaluation prioritizes MAE reduction, while the structural modules are intended to capture consistency. If the training objective does not directly reward consistency strongly enough, the model may improve MAE without improving structural metrics such as triangle-violation rate.

\paragraph{Synthetic noise mismatch.}
The log-normal noise augmentation is a practical approximation to noisy VLM predictions rather than a faithful generative model of all spatial errors.

\paragraph{Marginal feedback improvement.}
The feedback loop is evaluated on all 999 test scenes but produces only marginal improvement (2.8885\,m $\to$ 2.8868\,m). On superCLEVR synthetic scenes with unambiguous objects, re-annotating the image and re-querying the VLM provides little new spatial information, so the loop converges quickly without substantially changing the GNN output.

\section{Limitations}

This work has several important limitations. First, the benchmark is synthetic. Spatial457 is useful for controlled evaluation, but it does not fully capture the complexity of real-world clutter, occlusion, ambiguous boundaries, or detector failure. Second, the strongest result in the paper is an MAE reduction rather than a reduction in strict geometric inconsistency. Third, the current residual uses only two-hop alternatives and averages them uniformly. Fourth, the epistemic component is not a calibrated uncertainty model. Fifth, the trigger-based feedback loop was evaluated but showed only marginal improvement on this synthetic benchmark; its potential benefits may only surface on real-world scenes with ambiguous objects where re-querying the VLM with additional context is more informative.

\section{Conclusion}
\label{sec:conclusion}

We presented Quant-EpiGNN, a modular framework for detecting and refining quantitative spatial inconsistencies in vision-language outputs. The core idea is to move from isolated pairwise predictions to explicit relational reasoning over a scene graph, using a geometric consistency residual as a signal for edge reliability.

On Spatial457 \cite{wang2025spatial457}, the best graph model reduces distance MAE from 6.1924\,m to 2.8885\,m, showing that relational aggregation is a strong mechanism for refining noisy VLM predictions. At the same time, the ablation study delivers an important corrective result: the more complex variants with explicit geometric and epistemic structure do not outperform a plain GNN in the current setting.

The correct conclusion is therefore not that the full original design has been completely validated. The correct conclusion is narrower and stronger: explicit graph reasoning works, but the additional uncertainty-aware machinery remains unproven under the present benchmark and noise model. Future work should focus on stronger global consistency objectives, better-matched uncertainty modeling, and real-world evaluation beyond synthetic scenes.

\section*{Ethical Considerations}

The main risk in work of this kind is overclaiming capability. A model that appears spatially competent under selected conditions may still fail under occlusion, ambiguous perception, or real-world distribution shift. Any deployment in safety-critical settings would require substantially stronger validation than presented here.

\section*{Reproducibility Statement}

We report the training setup, hardware, evaluation protocol, checkpoint variants, and final metrics used in our experiments. The implementation includes the scene graph construction pipeline, graph reasoning models, ablation checkpoints, and the trigger-based feedback module. Our main claims in this report are restricted to the components for which quantitative benchmark results are complete.

\bibliographystyle{acl_natbib}
\bibliography{custom}

\end{document}
