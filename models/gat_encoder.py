import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    """
    Module 2: Graph Attention Network (GAT) Encoder
    Replaces DRL-PCR's uniform GRU-based aggregation with an attention-weighted encoder.
    Generates node embeddings taking topology and node features into account.
    """
    def __init__(self, in_channels: int = 2, hidden_channels: int = 32, out_channels: int = 32, heads: int = 4):
        """
        in_channels: Usually 2 (capacity, balance) or similar node features.
        """
        super().__init__()
        # GATv2Conv is generally preferred over original GATConv for dynamic attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # Output layer aggregates heads via mean (concat=False)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.activation = nn.ELU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        return x
