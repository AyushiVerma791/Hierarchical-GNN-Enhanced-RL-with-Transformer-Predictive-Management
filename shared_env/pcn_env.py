import networkx as nx
import numpy as np
import random
from typing import List, Tuple, Dict, Any

class PCNEnvironment:
    """
    Payment Channel Network (PCN) Environment.
    Represents the PCN as a directed graph G=(V,E) with bidirectional channels and balance matrix.
    """
    def __init__(self, num_nodes: int = 100, seed: int = 42, topology: str = 'barabasi_albert'):
        self.num_nodes = num_nodes
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize graph
        self.graph = self._generate_topology(topology)
        self._initialize_balances()
        self.initial_balances = nx.get_edge_attributes(self.graph, 'balance').copy()

    def _generate_topology(self, topology: str) -> nx.DiGraph:
        """Generates the PCN topology."""
        if topology == 'barabasi_albert':
            # Barabasi-Albert generates scale-free graphs (similar to Lightning network hubs)
            base_graph = nx.barabasi_albert_graph(n=self.num_nodes, m=2, seed=self.seed)
        elif topology == 'small_world':
            base_graph = nx.watts_strogatz_graph(n=self.num_nodes, k=4, p=0.1, seed=self.seed)
        else:
            raise ValueError(f"Unknown topology: {topology}")
            
        # Convert to directed graph with bidirectional edges
        di_graph = nx.DiGraph()
        for u, v in base_graph.edges():
            di_graph.add_edge(u, v)
            di_graph.add_edge(v, u)
            
        return di_graph
        
    def _initialize_balances(self, min_cap: int = 1000, max_cap: int = 10000):
        """Initializes balances evenly across both directions of each channel."""
        for u, v in self.graph.edges():
            if not 'balance' in self.graph[u][v]:
                # Total capacity of the undirected channel
                total_capacity = random.randint(min_cap, max_cap)
                # Split evenly according to DRL-PCR paper initialization approach
                self.graph[u][v]['capacity'] = total_capacity
                self.graph[u][v]['balance'] = total_capacity // 2
                self.graph[v][u]['capacity'] = total_capacity
                self.graph[v][u]['balance'] = total_capacity - (total_capacity // 2)

    def get_balance(self, u: int, v: int) -> int:
        """Get available balance from u to v."""
        if self.graph.has_edge(u, v):
            return self.graph[u][v]['balance']
        return 0
        
    def get_capacity(self, u: int, v: int) -> int:
        """Get total capacity of channel u->v."""
        if self.graph.has_edge(u, v):
            return self.graph[u][v]['capacity']
        return 0

    def update_balance(self, u: int, v: int, amount: int) -> bool:
        """
        Transfers amount from u to v. 
        Returns True if successful, False if insufficient balance.
        """
        if self.get_balance(u, v) >= amount:
            self.graph[u][v]['balance'] -= amount
            self.graph[v][u]['balance'] += amount
            return True
        return False

    def reset(self):
        """Resets network to initial balances."""
        for u, v in self.graph.edges():
            self.graph[u][v]['balance'] = self.initial_balances[(u, v)]
            
    def compute_channel_imbalance(self, u: int, v: int) -> float:
        """
        Computes degree of channel imbalance: |C(u,v) - C(v,u)| / (C(u,v) + C(v,u))
        Returns 0 if total capacity is 0 to handle div-by-zero.
        """
        c_uv = self.get_balance(u, v)
        c_vu = self.get_balance(v, u)
        total = c_uv + c_vu
        if total == 0:
            return 0.0
        return abs(c_uv - c_vu) / total
