import numpy as np
import networkx as nx
from typing import Tuple, List, Dict
from .pcn_env import PCNEnvironment

class TransactionGenerator:
    """
    Generates transactions and routes them. Triggers rebalancing requests on failure.
    """
    def __init__(self, env: PCNEnvironment, rebalance_threshold: int = 25):
        self.env = env
        self.rebalance_threshold = rebalance_threshold
        self.failed_requests = [] # List of (u, v) tuples that failed
        self.total_transactions = 0
        self.successful_transactions = 0
        self.total_volume = 0
        
        # Exponential distribution parameters for senders and receivers to create hubs
        self.nodes = list(self.env.graph.nodes())
        self.weights = np.random.exponential(scale=1.0, size=len(self.nodes))
        self.weights /= self.weights.sum()
        
    def sample_transaction(self, min_amount=10, max_amount=1000) -> Tuple[int, int, int]:
        """Samples sender, receiver, and amount."""
        u, v = np.random.choice(self.nodes, size=2, replace=False, p=self.weights)
        amount = np.random.randint(min_amount, max_amount)
        return u, v, amount
        
    def route_shortest_path(self, u: int, v: int, amount: int) -> bool:
        """Attempts to route transaction using shortest path first."""
        try:
            # We just use unweighted shortest path for routing
            path = nx.shortest_path(self.env.graph, source=u, target=v)
            
            # Check if all edges in path have enough balance
            for i in range(len(path) - 1):
                src, dst = path[i], path[i+1]
                if self.env.get_balance(src, dst) < amount:
                    # Failed due to insufficient balance on src->dst
                    self.failed_requests.append((src, dst))
                    return False
                    
            # If we get here, path is valid, execute the transaction
            for i in range(len(path) - 1):
                src, dst = path[i], path[i+1]
                self.env.update_balance(src, dst, amount)
                
            return True
            
        except nx.NetworkXNoPath:
            # No topological path exists
            return False

    def step(self) -> bool:
        """
        Executes one transaction.
        Returns True if a rebalancing trigger was hit.
        """
        u, v, amount = self.sample_transaction()
        self.total_transactions += 1
        
        success = self.route_shortest_path(u, v, amount)
        if success:
            self.successful_transactions += 1
            self.total_volume += amount
            
        # Check if we should trigger rebalancing
        if len(self.failed_requests) >= self.rebalance_threshold:
            return True
            
        return False
        
    def clear_requests(self):
        self.failed_requests = []
        
    def get_metrics(self) -> Dict:
        return {
            'total_transactions': self.total_transactions,
            'successful_transactions': self.successful_transactions,
            'success_ratio': self.successful_transactions / max(1, self.total_transactions),
            'throughput': self.total_volume
        }
