# models/gnn_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data

class GNNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 model_type: str = "GCN", num_heads: int = 8):
        super(GNNClassifier, self).__init__()
        self.model_type = model_type
        
        if model_type == "GCN":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif model_type == "GAT":
            self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
            self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
        elif model_type == "GraphSAGE":
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class InfluencePredictor(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int):
        super(InfluencePredictor, self).__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.influence_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_features: torch.Tensor = None) -> torch.Tensor:
        node_embeddings = self.node_encoder(node_features)
        
        if edge_features is None:
            edge_features = torch.ones(edge_index.size(1), 1)
        
        edge_embeddings = self.edge_encoder(edge_features)
        
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        source_embeddings = node_embeddings[source_nodes]
        target_embeddings = node_embeddings[target_nodes]
        
        combined_features = torch.cat([source_embeddings, target_embeddings, edge_embeddings], dim=1)
        influence_scores = self.influence_predictor(combined_features)
        
        return influence_scores