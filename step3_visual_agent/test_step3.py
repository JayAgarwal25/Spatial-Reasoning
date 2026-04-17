"""
tests/test_step3.py — Syntax, logic, and integration tests for Step 3.

Run with:
    pytest tests/test_step3.py -v

Tests are grouped into three classes:
    TestTrigger      — trigger.py logic
    TestActions      — actions.py drawing primitives
    TestFeedbackLoop — feedback_loop.py orchestration
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch

# Allow imports from step3_visual_agent/ and root schemas.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "step3_visual_agent"))
sys.path.insert(0, os.path.join(REPO_ROOT, "step1_scene_graph"))
sys.path.insert(0, REPO_ROOT)

from step1_scene_graph.schemas import ObjectNode, RelationEdge, SceneGraph
from step3_visual_agent.trigger import residual_trigger, rank_flagged_edges, TriggerResult
from step3_visual_agent.actions import draw_bbox, draw_line, depth_crop, ground_flagged_edge, ActionResult
from step3_visual_agent.feedback_loop import run_feedback_loop, LoopResult, save_loop_images


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_node(node_id: int, label: str = "obj",
               x=10.0, y=10.0, w=50.0, h=50.0, depth=1.0) -> ObjectNode:
    return ObjectNode(
        id=node_id, label=label,
        bbox=[x, y, w, h],
        confidence=0.9,
        center=[x + w / 2, y + h / 2],
        width=w, height=h, area=w * h,
        depth=depth,
    )


def _make_edge(src: int, dst: int, predicate: str = "near",
               confidence: float = 0.8, dist_m: float = 1.5) -> RelationEdge:
    return RelationEdge(
        subject_id=src, object_id=dst,
        predicate=predicate, confidence=confidence,
        features={"dist_m": dist_m, "angle_rad": 0.3, "depth_diff": 0.2},
    )


def _make_scene_graph(n_nodes: int = 3, n_edges: int = 3) -> SceneGraph:
    nodes = [_make_node(i, f"obj_{i}", x=float(i * 60)) for i in range(n_nodes)]
    edges = [
        _make_edge(i, (i + 1) % n_nodes)
        for i in range(min(n_edges, n_nodes))
    ]
    return SceneGraph(
        image_path="test.jpg",
        image_width=640, image_height=480,
        nodes=nodes, edges=edges,
    )


def _make_residuals(values: list[float]) -> dict:
    """Build a minimal gnn_output dict with given residual values."""
    return {"residuals": torch.tensor(values, dtype=torch.float32).unsqueeze(1)}


def _blank_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# TestTrigger
# ---------------------------------------------------------------------------

class TestTrigger:

    def test_no_edges_flagged_when_all_below_epsilon(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = _make_residuals([0.1, 0.2, 0.05])
        result = residual_trigger(gnn_out, sg, epsilon=0.3)

        assert result.needs_regrounding is False
        assert result.flagged_edge_indices == []
        assert result.flagged_edges == []

    def test_correct_edges_flagged_above_epsilon(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = _make_residuals([0.1, 0.8, 0.05])   # edge 1 exceeds 0.3
        result = residual_trigger(gnn_out, sg, epsilon=0.3)

        assert result.needs_regrounding is True
        assert result.flagged_edge_indices == [1]
        assert len(result.flagged_edges) == 1
        assert result.flagged_edges[0] is sg.edges[1]

    def test_all_edges_flagged_when_all_exceed_epsilon(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = _make_residuals([1.0, 2.0, 0.5])
        result = residual_trigger(gnn_out, sg, epsilon=0.3)

        assert result.needs_regrounding is True
        assert len(result.flagged_edge_indices) == 3

    def test_epsilon_boundary_is_exclusive(self):
        """r == epsilon should NOT be flagged; only r > epsilon."""
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = _make_residuals([0.3, 0.3, 0.3])
        result = residual_trigger(gnn_out, sg, epsilon=0.3)
        assert result.needs_regrounding is False

    def test_raises_on_missing_residuals_key(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        with pytest.raises(KeyError, match="residuals"):
            residual_trigger({"pred_dist": torch.zeros(3, 1)}, sg)

    def test_raises_on_edge_count_mismatch(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)    # 3 edges
        gnn_out = _make_residuals([0.1, 0.2])            # only 2 residuals
        with pytest.raises(ValueError, match="Edge count mismatch"):
            residual_trigger(gnn_out, sg, epsilon=0.3)

    def test_stats_fields_present(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = _make_residuals([0.1, 0.5, 0.05])
        result = residual_trigger(gnn_out, sg, epsilon=0.3)

        for key in ("num_edges", "num_flagged", "flagged_fraction",
                    "residual_mean", "residual_max", "residual_min", "epsilon"):
            assert key in result.stats, f"Missing stat: {key}"

    def test_stats_values_correct(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = _make_residuals([0.1, 0.5, 0.05])
        result = residual_trigger(gnn_out, sg, epsilon=0.3)

        assert result.stats["num_edges"]    == 3
        assert result.stats["num_flagged"]  == 1
        assert abs(result.stats["residual_max"]  - 0.5)  < 1e-5
        assert abs(result.stats["residual_min"]  - 0.05) < 1e-5

    def test_rank_flagged_edges_descending(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = _make_residuals([0.1, 0.9, 0.5])
        result = residual_trigger(gnn_out, sg, epsilon=0.05)
        ranked = rank_flagged_edges(result)

        # edge 1 (r=0.9) should come before edge 2 (r=0.5) before edge 0 (r=0.1)
        assert ranked == [1, 2, 0]

    def test_rank_flagged_edges_empty_when_none_flagged(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = _make_residuals([0.0, 0.0, 0.0])
        result = residual_trigger(gnn_out, sg, epsilon=0.3)
        assert rank_flagged_edges(result) == []

    def test_trigger_result_is_dataclass_with_correct_types(self):
        sg = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = _make_residuals([0.1, 0.2, 0.05])
        result = residual_trigger(gnn_out, sg, epsilon=0.3)

        assert isinstance(result, TriggerResult)
        assert isinstance(result.flagged_edge_indices, list)
        assert isinstance(result.flagged_edges, list)
        assert isinstance(result.residuals, torch.Tensor)
        assert isinstance(result.needs_regrounding, bool)
        assert isinstance(result.epsilon, float)


# ---------------------------------------------------------------------------
# TestActions
# ---------------------------------------------------------------------------

class TestActions:

    def setup_method(self):
        self.image    = _blank_image()
        self.src_node = _make_node(0, "chair", x=50,  y=50,  w=80, h=80)
        self.dst_node = _make_node(1, "table", x=300, y=200, w=100, h=60)
        self.edge     = _make_edge(0, 1, predicate="next_to", dist_m=2.0)

    # ---- draw_bbox ----

    def test_draw_bbox_returns_action_result(self):
        result = draw_bbox(self.image, self.src_node, edge_index=0, residual=0.42)
        assert isinstance(result, ActionResult)

    def test_draw_bbox_does_not_mutate_input(self):
        original = self.image.copy()
        draw_bbox(self.image, self.src_node, edge_index=0, residual=0.42)
        np.testing.assert_array_equal(self.image, original)

    def test_draw_bbox_output_same_shape(self):
        result = draw_bbox(self.image, self.src_node, edge_index=0, residual=0.42)
        assert result.image.shape == self.image.shape

    def test_draw_bbox_output_differs_from_input(self):
        result = draw_bbox(self.image, self.src_node, edge_index=0, residual=0.42)
        assert not np.array_equal(result.image, self.image), \
            "draw_bbox should modify the output image"

    def test_draw_bbox_action_type(self):
        result = draw_bbox(self.image, self.src_node, edge_index=0, residual=0.42)
        assert result.action_type == "draw_bbox"

    def test_draw_bbox_metadata_fields(self):
        result = draw_bbox(self.image, self.src_node, edge_index=2, residual=0.77)
        for key in ("node_id", "node_label", "bbox", "residual"):
            assert key in result.metadata
        assert result.metadata["node_id"]    == self.src_node.id
        assert result.metadata["node_label"] == self.src_node.label
        assert abs(result.metadata["residual"] - 0.77) < 1e-6
        assert result.edge_index == 2

    # ---- draw_line ----

    def test_draw_line_returns_action_result(self):
        result = draw_line(
            self.image, self.src_node, self.dst_node,
            self.edge, edge_index=0, residual=0.55, pred_dist=2.1,
        )
        assert isinstance(result, ActionResult)

    def test_draw_line_does_not_mutate_input(self):
        original = self.image.copy()
        draw_line(
            self.image, self.src_node, self.dst_node,
            self.edge, edge_index=0, residual=0.55, pred_dist=2.1,
        )
        np.testing.assert_array_equal(self.image, original)

    def test_draw_line_output_same_shape(self):
        result = draw_line(
            self.image, self.src_node, self.dst_node,
            self.edge, edge_index=0, residual=0.55, pred_dist=2.1,
        )
        assert result.image.shape == self.image.shape

    def test_draw_line_action_type(self):
        result = draw_line(
            self.image, self.src_node, self.dst_node,
            self.edge, edge_index=0, residual=0.55, pred_dist=2.1,
        )
        assert result.action_type == "draw_line"

    def test_draw_line_metadata_fields(self):
        result = draw_line(
            self.image, self.src_node, self.dst_node,
            self.edge, edge_index=1, residual=0.55, pred_dist=2.1,
        )
        for key in ("src_node_id", "dst_node_id", "predicate",
                    "pred_dist_m", "residual", "confidence",
                    "src_centre", "dst_centre"):
            assert key in result.metadata, f"Missing metadata key: {key}"
        assert result.metadata["predicate"] == self.edge.predicate
        assert abs(result.metadata["pred_dist_m"] - 2.1) < 1e-6

    # ---- depth_crop ----

    def test_depth_crop_returns_action_result(self):
        result = depth_crop(self.image, self.src_node, edge_index=0, residual=0.3)
        assert isinstance(result, ActionResult)

    def test_depth_crop_does_not_mutate_input(self):
        original = self.image.copy()
        depth_crop(self.image, self.src_node, edge_index=0, residual=0.3)
        np.testing.assert_array_equal(self.image, original)

    def test_depth_crop_smaller_than_original(self):
        result = depth_crop(self.image, self.src_node, edge_index=0, residual=0.3)
        H, W = self.image.shape[:2]
        ch, cw = result.image.shape[:2]
        assert ch <= H and cw <= W

    def test_depth_crop_action_type(self):
        result = depth_crop(self.image, self.src_node, edge_index=0, residual=0.3)
        assert result.action_type == "depth_crop"

    def test_depth_crop_metadata_fields(self):
        result = depth_crop(self.image, self.src_node, edge_index=0, residual=0.3)
        for key in ("node_id", "node_label", "crop_rect", "original_size", "residual"):
            assert key in result.metadata

    def test_depth_crop_pad_clamps_to_image_bounds(self):
        """Node at corner with huge pad should not go out of bounds."""
        corner_node = _make_node(99, "corner", x=0, y=0, w=20, h=20)
        result = depth_crop(self.image, corner_node, edge_index=0, residual=0.1, pad_px=9999)
        h, w = result.image.shape[:2]
        assert h > 0 and w > 0

    # ---- ground_flagged_edge ----

    def test_ground_flagged_edge_returns_five_results(self):
        results = ground_flagged_edge(
            self.image, self.edge, edge_index=0,
            src_node=self.src_node, dst_node=self.dst_node,
            residual=0.7, pred_dist=3.0,
        )
        assert len(results) == 5

    def test_ground_flagged_edge_action_types(self):
        results = ground_flagged_edge(
            self.image, self.edge, edge_index=0,
            src_node=self.src_node, dst_node=self.dst_node,
            residual=0.7, pred_dist=3.0,
        )
        types = [r.action_type for r in results]
        assert types[0] == "draw_bbox"
        assert types[1] == "draw_bbox"
        assert types[2] == "draw_line"
        assert types[3] == "depth_crop"
        assert types[4] == "depth_crop"

    def test_ground_flagged_edge_final_annotation_differs_from_input(self):
        results = ground_flagged_edge(
            self.image, self.edge, edge_index=0,
            src_node=self.src_node, dst_node=self.dst_node,
            residual=0.7, pred_dist=3.0,
        )
        assert not np.array_equal(results[2].image, self.image)


# ---------------------------------------------------------------------------
# TestFeedbackLoop
# ---------------------------------------------------------------------------

class TestFeedbackLoop:
    """
    Feedback loop tests use mock run_step1 / run_step2 callables so they
    do not require any GPU or real model weights.
    """

    def _make_gnn_output(self, n_edges: int, residual_val: float = 0.0) -> dict:
        return {
            "residuals":    torch.full((n_edges, 1), residual_val),
            "pred_dist":    torch.ones(n_edges, 1),
            "pred_classes": torch.zeros(n_edges, dtype=torch.long),
            "mu":           torch.zeros(3, 64),
            "sigma":        torch.ones(3, 64),
        }

    def test_converges_immediately_when_no_edges_flagged(self):
        sg      = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = self._make_gnn_output(n_edges=3, residual_val=0.0)
        image   = _blank_image()

        result = run_feedback_loop(
            image=image, scene_graph=sg, gnn_output=gnn_out,
            run_step1=lambda img, **kw: (_ for _ in ()).throw(AssertionError("Step1 should not be called")),
            run_step2=lambda sg: (_ for _ in ()).throw(AssertionError("Step2 should not be called")),
            epsilon=0.3, max_iters=3,
        )

        assert result.converged is True
        assert result.iterations_run == 1

    def test_loop_stops_at_max_iters(self):
        """Even with persistently high residuals, loop must not exceed max_iters."""
        sg      = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = self._make_gnn_output(n_edges=3, residual_val=1.0)
        image   = _blank_image()

        call_counts = {"step1": 0, "step2": 0}

        def fake_step1(img, **kw):
            call_counts["step1"] += 1
            return _make_scene_graph(n_nodes=3, n_edges=3)

        def fake_step2(sg):
            call_counts["step2"] += 1
            return self._make_gnn_output(n_edges=3, residual_val=1.0)

        result = run_feedback_loop(
            image=image, scene_graph=sg, gnn_output=gnn_out,
            run_step1=fake_step1, run_step2=fake_step2,
            epsilon=0.3, max_iters=3,
        )

        assert result.converged is False
        assert call_counts["step1"] <= 3
        assert call_counts["step2"] <= 3

    def test_loop_converges_when_residuals_drop_after_one_iter(self):
        """Simulate: iteration 0 has high residuals, iteration 1 drops to zero."""
        sg       = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out  = self._make_gnn_output(n_edges=3, residual_val=1.0)
        image    = _blank_image()
        call_n   = {"n": 0}

        def fake_step1(img, **kw):
            return _make_scene_graph(n_nodes=3, n_edges=3)

        def fake_step2(sg):
            call_n["n"] += 1
            # First re-run returns 0 residuals → triggers convergence
            return self._make_gnn_output(n_edges=3, residual_val=0.0)

        result = run_feedback_loop(
            image=image, scene_graph=sg, gnn_output=gnn_out,
            run_step1=fake_step1, run_step2=fake_step2,
            epsilon=0.3, max_iters=5,
        )

        assert result.converged is True
        assert result.iterations_run <= 5

    def test_loop_result_has_all_fields(self):
        sg      = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = self._make_gnn_output(n_edges=3, residual_val=0.0)
        image   = _blank_image()

        result = run_feedback_loop(
            image=image, scene_graph=sg, gnn_output=gnn_out,
            run_step1=lambda img, **kw: sg,
            run_step2=lambda sg: gnn_out,
            epsilon=0.3, max_iters=3,
        )

        assert isinstance(result, LoopResult)
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations_run, int)
        assert isinstance(result.final_scene_graph, SceneGraph)
        assert isinstance(result.final_gnn_output, dict)
        assert isinstance(result.history, list)
        assert isinstance(result.final_image, np.ndarray)

    def test_history_records_per_iteration(self):
        sg      = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = self._make_gnn_output(n_edges=3, residual_val=0.0)
        image   = _blank_image()

        result = run_feedback_loop(
            image=image, scene_graph=sg, gnn_output=gnn_out,
            run_step1=lambda img, **kw: sg,
            run_step2=lambda sg: gnn_out,
            epsilon=0.3, max_iters=3,
        )

        assert len(result.history) == result.iterations_run
        for rec in result.history:
            assert hasattr(rec, "iteration")
            assert hasattr(rec, "trigger_result")
            assert hasattr(rec, "action_results")
            assert hasattr(rec, "annotated_image")
            assert hasattr(rec, "residual_max")
            assert hasattr(rec, "num_flagged")

    def test_final_image_is_numpy_array(self):
        sg      = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = self._make_gnn_output(n_edges=3, residual_val=0.0)
        image   = _blank_image()

        result = run_feedback_loop(
            image=image, scene_graph=sg, gnn_output=gnn_out,
            run_step1=lambda img, **kw: sg,
            run_step2=lambda sg: gnn_out,
            epsilon=0.3, max_iters=3,
        )

        assert isinstance(result.final_image, np.ndarray)
        assert result.final_image.dtype == np.uint8

    def test_save_loop_images(self, tmp_path):
        sg      = _make_scene_graph(n_nodes=3, n_edges=3)
        gnn_out = self._make_gnn_output(n_edges=3, residual_val=0.0)
        image   = _blank_image()

        result = run_feedback_loop(
            image=image, scene_graph=sg, gnn_output=gnn_out,
            run_step1=lambda img, **kw: sg,
            run_step2=lambda sg: gnn_out,
            epsilon=0.3, max_iters=3,
        )

        paths = save_loop_images(result, output_dir=str(tmp_path))
        for p in paths:
            assert os.path.isfile(p), f"Expected saved file: {p}"
