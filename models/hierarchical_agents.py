import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

class HighLevelAgent(nn.Module):
    """
    Strategic Agent (PPO).
    Input: global liquidity status, network-wide congestion, Transformer predictions, global pooled GAT embeddings.
    Output: which region/subgraph to target (discrete), liquidity budget (continuous), global fee adjustment (continuous).
    """
    def __init__(self, state_dim: int, num_regions: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor heads
        self.region_logits = nn.Linear(128, num_regions)
        self.budget_mean = nn.Linear(128, 1)
        self.budget_std = nn.Linear(128, 1)
        self.fee_mean = nn.Linear(128, 1)
        self.fee_std = nn.Linear(128, 1)
        
        # Critic head
        self.value = nn.Linear(128, 1)
        
    def forward(self, state):
        features = self.net(state)
        
        # Region selection (Discrete)
        logits = self.region_logits(features)
        
        # Budget and Fee (Continuous)
        b_mean = self.budget_mean(features)
        b_std = torch.nn.functional.softplus(self.budget_std(features)) + 1e-5
        
        f_mean = self.fee_mean(features)
        f_std = torch.nn.functional.softplus(self.fee_std(features)) + 1e-5
        
        val = self.value(features)
        
        return logits, (b_mean, b_std), (f_mean, f_std), val

class LowLevelAgent(nn.Module):
    """
    Operational Agent (PPO).
    Input: local channel states, local subgraph topology, predicted local demand, local GAT embeddings, 
           PLUS high-level agent's chosen region and budget.
    Output: Which circular path to use, amount to transfer, local fee adjustment.
    """
    def __init__(self, state_dim: int, num_paths: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Action is a vector of length K (num_paths) representing amount routed through each candidate path
        # plus a scalar for fee adjustment
        self.amount_mean = nn.Linear(128, num_paths)
        self.amount_std = nn.Linear(128, num_paths)
        
        self.fee_mean = nn.Linear(128, 1)
        self.fee_std = nn.Linear(128, 1)
        
        # Critic
        self.value = nn.Linear(128, 1)
        
    def forward(self, state, action_mask=None):
        """
        action_mask: boolean mask constraining actions to the high-level agent's chosen region.
        """
        features = self.net(state)
        
        a_mean = self.amount_mean(features)
        
        # Masking: zero out means for paths outside the chosen region
        if action_mask is not None:
            a_mean = a_mean * action_mask
            
        a_std = torch.nn.functional.softplus(self.amount_std(features)) + 1e-5
        
        f_mean = self.fee_mean(features)
        f_std = torch.nn.functional.softplus(self.fee_std(features)) + 1e-5
        
        val = self.value(features)
        
        return (a_mean, a_std), (f_mean, f_std), val
