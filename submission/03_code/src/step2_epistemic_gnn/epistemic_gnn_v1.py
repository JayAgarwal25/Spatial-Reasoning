import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Data

class TemperatureScaledConfidence(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        # Learnable temperature parameter for Platt/Temperature Scaling
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, raw_logits):
        # Scale the raw VLM logits to prevent confidence laundering
        return torch.sigmoid(raw_logits / self.temperature)

class ConfidenceConditionedMessagePassing(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Map concatenated source, target, and edge features to a message
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 128, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        # Attention gate modulated by CALIBRATED VLM confidence
        self.attn_mlp = nn.Sequential(
            nn.Linear(out_channels + 1, 1), # Message + Calibrated VLM Confidence
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, calibrated_vlm_conf):
        row, col = edge_index
        src, tgt = x[row], x[col]
        
        # Compute raw messages
        msg_input = torch.cat([src, tgt, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        
        # Cross-Modal Uncertainty Routing
        # GNN modulates message passing explicitly based on calibrated language model confidence
        attn_input = torch.cat([msg, calibrated_vlm_conf], dim=-1)
        attn_weight = self.attn_mlp(attn_input)
        
        gated_msg = msg * attn_weight
        
        # Aggregate messages
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, msg.size(1), device=x.device)
        out.index_add_(0, col, gated_msg)
        
        return out

class SelectiveVerificationGNN(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, num_classes):
        super().__init__()
        self.node_proj = nn.Linear(node_in_channels, hidden_channels)
        self.edge_proj = nn.Linear(edge_in_channels, 128)
        
        self.temp_scaler = TemperatureScaledConfidence()
        
        self.conv1 = ConfidenceConditionedMessagePassing(hidden_channels, hidden_channels)
        self.conv2 = ConfidenceConditionedMessagePassing(hidden_channels, hidden_channels)
        
        self.semantic_head = nn.Linear(hidden_channels, num_classes)
        self.numeric_head = nn.Linear(hidden_channels, 5) # [x, y, w, h, depth]
        self.verify_head = nn.Linear(hidden_channels, 1) # Probability of requesting visual verification

    def forward(self, x, edge_index, edge_attr, raw_vlm_confidence, drop_prob=0.0):
        x = F.relu(self.node_proj(x))
        e = F.relu(self.edge_proj(edge_attr))
        
        calibrated_vlm_conf = self.temp_scaler(raw_vlm_confidence)
        
        # Spatial DropEdge for Structural Sensitivity
        if drop_prob > 0.0 and self.training:
            edge_index_dropped, edge_mask = dropout_edge(edge_index, p=drop_prob, force_undirected=False, training=self.training)
            e_dropped = e[edge_mask]
            vlm_conf_dropped = calibrated_vlm_conf[edge_mask]
        else:
            edge_index_dropped, edge_mask = edge_index, None
            e_dropped, vlm_conf_dropped = e, calibrated_vlm_conf
            
        x = self.conv1(x, edge_index_dropped, e_dropped, vlm_conf_dropped)
        x = F.relu(x)
        
        if drop_prob > 0.0 and self.training:
            edge_index_dropped, edge_mask = dropout_edge(edge_index, p=drop_prob, force_undirected=False, training=self.training)
            e_dropped = e[edge_mask]
            vlm_conf_dropped = calibrated_vlm_conf[edge_mask]
        else:
            edge_index_dropped, edge_mask = edge_index, None
            e_dropped, vlm_conf_dropped = e, calibrated_vlm_conf
            
        x = self.conv2(x, edge_index_dropped, e_dropped, vlm_conf_dropped)
        x = F.relu(x)
        
        sem_logits = self.semantic_head(x)
        num_preds = self.numeric_head(x)
        verify_logits = self.verify_head(x)
        
        return sem_logits, num_preds, verify_logits

    def estimate_structural_sensitivity(self, x, edge_index, edge_attr, raw_vlm_confidence, num_samples=5, drop_prob=0.3):
        '''
        Trigger Mechanism for Step 3:
        Runs multiple Stochastic Forward Passes using Spatial DropEdge.
        High prediction variance = High Structural Sensitivity = Vulnerable to visual-symbolic mismatches.
        '''
        self.train() # Force spatial dropout to be active for MC Sampling
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                _, num_pred, _ = self.forward(x, edge_index, edge_attr, raw_vlm_confidence, drop_prob=drop_prob)
                preds.append(num_pred)
                
        preds = torch.stack(preds) # [num_samples, num_nodes, 5]
        mean_pred = preds.mean(dim=0)
        var_pred = preds.var(dim=0) # This is the Structural Sensitivity (Variance)!
        
        return mean_pred, var_pred

class RiskAwareVerificationLoss(nn.Module):
    def __init__(self, semantic_weight=1.0, numeric_weight=1.0, verify_weight=0.5, verification_cost=0.3):
        super().__init__()
        self.semantic_weight = semantic_weight
        self.numeric_weight = numeric_weight
        self.verify_weight = verify_weight
        self.verification_cost = verification_cost
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, sem_pred, sem_target, num_pred, num_target, verify_logits):
        loss_semantic = self.ce_loss(sem_pred, sem_target)
        
        loss_numerical_node = self.mae_loss(num_pred, num_target).mean(dim=1) 
        verify_prob = torch.sigmoid(verify_logits).squeeze(-1)
        
        loss_verify_node = verify_prob * self.verification_cost + (1 - verify_prob) * loss_numerical_node
        
        loss_numerical = loss_numerical_node.mean()
        loss_verify = loss_verify_node.mean()
        
        total_loss = (
            self.semantic_weight * loss_semantic + 
            self.numeric_weight * loss_numerical + 
            self.verify_weight * loss_verify
        )
        return total_loss, loss_semantic, loss_numerical, loss_verify
