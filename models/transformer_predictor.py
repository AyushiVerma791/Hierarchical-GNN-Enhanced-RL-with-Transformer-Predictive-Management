import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class TransformerPredictor(nn.Module):
    """
    Module 3: Transformer-Based Predictor
    Maintains a rolling buffer of T GAT-embedding snapshots to predict near-future state.
    """
    def __init__(self, d_model: int = 32, nhead: int = 4, num_layers: int = 2, seq_len: int = 10, num_nodes: int = 100):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Prediction heads
        # Outputs: 
        # (a) channel liquidity imbalance (num_nodes)
        # (b) traffic demand (num_nodes)
        # (c) depletion risk (num_nodes)
        # (d) congestion prob (1)
        self.head_imbalance = nn.Linear(d_model, 1)
        self.head_demand = nn.Linear(d_model, 1)
        self.head_depletion = nn.Linear(d_model, 1)
        self.head_congestion = nn.Linear(d_model * num_nodes, 1)
        
        self.active = True # For ablation toggling
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, num_nodes, d_model]
        Returns:
            Dict with predictions.
        """
        if not self.active:
            # Ablation mode: output zeros
            batch_size = x.size(1)
            return {
                'imbalance': torch.zeros(batch_size, self.num_nodes, 1).to(x.device),
                'demand': torch.zeros(batch_size, self.num_nodes, 1).to(x.device),
                'depletion': torch.zeros(batch_size, self.num_nodes, 1).to(x.device),
                'congestion': torch.zeros(batch_size, 1).to(x.device)
            }
            
        seq_len, batch_size, num_nodes, d_model = x.size()
        
        # We need to process temporal sequence. 
        # Reshape to [seq_len, batch_size * num_nodes, d_model]
        x_reshaped = x.view(seq_len, batch_size * num_nodes, d_model)
        
        # Add positional encoding
        x_reshaped = self.pos_encoder(x_reshaped)
        
        # Transformer forward
        output = self.transformer_encoder(x_reshaped)
        
        # Take the last time step's output [batch_size * num_nodes, d_model]
        last_out = output[-1, :, :]
        
        # Predict node-level metrics
        imbalance = self.head_imbalance(last_out).view(batch_size, num_nodes, 1)
        demand = self.head_demand(last_out).view(batch_size, num_nodes, 1)
        depletion = torch.sigmoid(self.head_depletion(last_out)).view(batch_size, num_nodes, 1)
        
        # Predict graph-level congestion
        # Flatten across nodes
        last_out_flat = last_out.view(batch_size, num_nodes * d_model)
        congestion = torch.sigmoid(self.head_congestion(last_out_flat))
        
        return {
            'imbalance': imbalance,
            'demand': demand,
            'depletion': depletion,
            'congestion': congestion
        }
