import unittest
import torch
import numpy as np
from pathlib import Path

# Importing the modules from your project structure
from step4_evaluation.metrics import compute_residuals_from_distances, triangle_violation_rate
from step4_evaluation.baseline import build_baseline
from step4_evaluation.spatialqa_eval import build_pyg_data

class TestSpatialEvaluation(unittest.TestCase):
    
    def setUp(self):
        """Set up a mock scene for testing."""
        self.mock_scene = {
            "scene_id": "test_scene_001",
            "objects": [
                {"id": "chair", "position": [0, 0, 0]},
                {"id": "table", "position": [1, 0, 0]},
                {"id": "lamp",  "position": [0, 1, 0]}
            ],
            "gt_distances": {
                "chair__table": 100.0, # 1 meter in cm
                "table__lamp":  141.4, # approx sqrt(2) in cm
                "chair__lamp":  100.0
            },
            "hallucination_labels": {
                "chair__table": 0,
                "table__lamp": 1 # Simulate a hallucination
            }
        }
        self.baseline = build_baseline("mock", seed=42)

    def test_baseline_standardization(self):
        """Check if the mock baseline returns expected types."""
        pairs = [("chair", "table")]
        preds = self.baseline.predict_distances("__none__", pairs)
        self.assertEqual(len(preds), 1)
        self.assertIsInstance(preds[0], float)

    def test_geometric_residual_logic(self):
        """Verify that residuals identify triangle inequality violations."""
        # Create a violation: d(AC) > d(AB) + d(BC)
        # Edge index: 0->1, 1->2, 0->2
        edge_index = np.array([[0, 1, 0], 
                               [1, 2, 2]])
        # 0->1 is 10, 1->2 is 10, but 0->2 is 50 (Impossible!)
        edge_dist = np.array([10.0, 10.0, 50.0])
        
        residuals = compute_residuals_from_distances(edge_index, edge_dist)
        
        # The residual for the 0->2 edge should be high
        self.assertGreater(residuals[2], 0)
        # The residual should be |50 - (10+10)| = 30
        self.assertEqual(residuals[2], 30.0)

    def test_pyg_data_construction(self):
        """Ensure the PyG Data object is built correctly for the EpiGNN."""
        baseline_preds = {"chair__table": 105.0, "table__lamp": 150.0, "chair__lamp": 95.0}
        data = build_pyg_data(self.mock_scene, baseline_preds, sem_dim=384)
        
        self.assertIsNotNone(data)
        self.assertEqual(data.num_nodes, 3)
        self.assertEqual(data.edge_index.shape[1], 3) # 3 pairs in combinations(3, 2)
        self.assertEqual(data.node_sem.shape, (3, 384))

    def test_triangle_violation_rate(self):
        """Check the fraction calculation for TI violations."""
        distances = {
            (0, 1): 10.0,
            (1, 2): 10.0,
            (0, 2): 50.0  # Violation
        }
        rate, mag = triangle_violation_rate(distances, [0, 1, 2])
        self.assertEqual(rate, 1.0) # 1 out of 1 triple violates
        self.assertEqual(mag, 30.0)

if __name__ == "__main__":
    unittest.main()