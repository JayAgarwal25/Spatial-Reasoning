import torch
from torch_geometric.data import Data
import random

def generate_mock_scene_graph(num_nodes=10, num_edges=20, feature_dim=64):
    x = torch.randn(num_nodes, feature_dim)
    sem_target = torch.randint(0, 10, (num_nodes,))
    num_target = torch.rand(num_nodes, 5) * 100.0
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, feature_dim)
    
    # NEW: VLM Confidence Score (0.0 to 1.0) for each extracted spatial edge
    vlm_confidence = torch.rand(num_edges, 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                vlm_confidence=vlm_confidence, sem_target=sem_target, num_target=num_target)
    return data

def run_mock_training_step():
    from epistemic_gnn import SelectiveVerificationGNN, RiskAwareVerificationLoss

    print("\n--- Running Selective Verification Mock Training Step ---")
    feature_dim = 64
    model = SelectiveVerificationGNN(node_in_channels=feature_dim, edge_in_channels=feature_dim, hidden_channels=128, num_classes=10)
    # The cost to verify visually instead of trusting the regression model
    loss_fn = RiskAwareVerificationLoss(verification_cost=15.0) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    mock_graph = generate_mock_scene_graph(num_nodes=15, num_edges=30, feature_dim=feature_dim)
    
    # Standard Forward Pass
    model.train()
    optimizer.zero_grad()
    sem_logits, num_preds, verify_logits = model(mock_graph.x, mock_graph.edge_index, mock_graph.edge_attr, mock_graph.vlm_confidence)
    
    total_loss, sem_loss, num_loss, verify_loss = loss_fn(sem_logits, mock_graph.sem_target, num_preds, mock_graph.num_target, verify_logits)
    total_loss.backward()
    optimizer.step()
    
    print(f"Forward Step Complete. Total Loss: {total_loss.item():.4f}")
    print(f" -> Semantic CE: {sem_loss.item():.4f}")
    print(f" -> Numerical MAE: {num_loss.item():.4f}")
    print(f" -> Verification Abstain Loss: {verify_loss.item():.4f}")
    
    # Step 3 Trigger Test (Structural Sensitivity via Spatial DropEdge)
    print("\n--- Running Structural Variance Estimation (Step 3 Trigger) ---")
    mean_pred, variance_pred = model.estimate_structural_sensitivity(
        mock_graph.x, mock_graph.edge_index, mock_graph.edge_attr, mock_graph.vlm_confidence, num_samples=5, drop_prob=0.3
    )
    print(f"Calculated Structural Sensitivity across {mock_graph.x.shape[0]} nodes using Spatial DropEdge.")
    print(f"Node 0 Depth Variance: {variance_pred[0, 4].item():.4f} (If High -> Predicts Failure -> Requery VLM Crop!)")

if __name__ == "__main__":
    run_mock_training_step()
