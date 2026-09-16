import os
import argparse
import yaml
import numpy as np
import torch
import sys

from shared_env.pcn_env import PCNEnvironment
from shared_env.transaction_generator import TransactionGenerator
from models.gat_encoder import GATEncoder
from models.transformer_predictor import TransformerPredictor
from models.hierarchical_agents import HighLevelAgent, LowLevelAgent

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    env = PCNEnvironment(num_nodes=config['env']['num_nodes'], seed=config['seed'])
    
    # Initialize models
    print("Initializing HGRL-TPM modules...")
    gat = GATEncoder(in_channels=2, hidden_channels=32, out_channels=32, heads=4)
    predictor = TransformerPredictor(d_model=32, seq_len=10, num_nodes=config['env']['num_nodes'])
    
    # We mock num_regions=5 for Louvain clustering
    high_agent = HighLevelAgent(state_dim=32, num_regions=5) 
    # State dim for low level would be local + high level outputs (budget, region embedding)
    low_agent = LowLevelAgent(state_dim=64, num_paths=config['env']['K'])
    
    optimizer = torch.optim.Adam(
        list(gat.parameters()) + 
        list(predictor.parameters()) + 
        list(high_agent.parameters()) + 
        list(low_agent.parameters()), 
        lr=float(config['training']['lr'])
    )
    
    epochs = config['training']['epochs']
    
    print("Starting hierarchical offline training loop...")
    for epoch in range(epochs):
        # Mocking forward pass for structural completeness
        b_size = config['training']['batch_size']
        n_nodes = config['env']['num_nodes']
        
        # 1. GAT Forward
        # Mock node features [N, 2] and edge index [2, E]
        x = torch.randn(n_nodes, 2)
        edge_index = torch.randint(0, n_nodes, (2, n_nodes * 4))
        node_embeddings = gat(x, edge_index) # [N, 32]
        
        # 2. Predictor Forward
        # Buffer of past T snapshots
        buffer = torch.randn(10, b_size, n_nodes, 32)
        preds = predictor(buffer)
        
        # 3. High-level Agent
        global_state = torch.randn(b_size, 32)
        hl_logits, hl_budget, hl_fee, hl_val = high_agent(global_state)
        
        # 4. Low-level Agent
        local_state = torch.randn(b_size, 64)
        ll_amount, ll_fee, ll_val = low_agent(local_state)
        
        # 5. Compute losses (Mock)
        loss = hl_val.mean() + ll_val.mean() + preds['imbalance'].mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Total Loss: {loss.item():.4f}")
            
    print("Training complete.")
    os.makedirs('eval', exist_ok=True)
    torch.save(high_agent.state_dict(), 'eval/hgrl_high_agent.pth')
    torch.save(low_agent.state_dict(), 'eval/hgrl_low_agent.pth')
    print("Models saved to eval/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    train(args.config)
