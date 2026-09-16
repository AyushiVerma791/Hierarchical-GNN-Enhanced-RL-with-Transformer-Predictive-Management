import networkx as nx
from typing import List, Tuple

def get_candidate_paths(graph: nx.DiGraph, source: int, target: int, 
                        K: int = 5, max_length: int = 6, penalty: float = 10.0) -> List[List[int]]:
    """
    Candidate path generation: Dijkstra-based, iterative.
    Starts with unit edge weights, repeatedly finds shortest path, 
    then adds a penalty weight to used edges so next search finds a different path.
    Stops once K candidate paths are found or no more valid paths exist under max_length.
    
    In the context of circular rebalancing, the source and target are the nodes of the depleted channel.
    Wait, if depleted channel is u->v, we want a path v->...->u to send funds back to u, 
    which effectively completes the circular transaction u->v->...->u.
    """
    temp_graph = graph.copy()
    
    # Initialize all edge weights to 1.0
    for u, v in temp_graph.edges():
        temp_graph[u][v]['weight'] = 1.0
        
    candidate_paths = []
    
    for _ in range(K):
        try:
            # Find shortest path using Dijkstra
            path = nx.shortest_path(temp_graph, source=source, target=target, weight='weight')
            
            # Filter by max_length (number of edges is len(path) - 1)
            if len(path) - 1 <= max_length:
                # Only add if it's not a direct channel between source and target 
                # (since we are trying to find alternative routes)
                if len(path) > 2:
                    candidate_paths.append(path)
                    
                    # Apply penalty to edges used in this path
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        temp_graph[u][v]['weight'] += penalty
            else:
                # If shortest path is longer than max_length, 
                # subsequent penalized paths will likely be longer too.
                # We apply penalty to avoid infinite loop on the same path, 
                # but we don't add it to candidates.
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    temp_graph[u][v]['weight'] += penalty
                    
        except nx.NetworkXNoPath:
            break
            
    return candidate_paths

def precompute_candidate_paths(graph: nx.DiGraph, K: int = 5, max_length: int = 6) -> dict:
    """
    Precomputes candidate paths for all directed edges in the graph.
    Returns a dict mapping edge (u, v) to a list of candidate circular paths (represented as lists of nodes).
    For a channel u->v, we look for paths from v to u, to form a circle.
    """
    all_candidates = {}
    for u, v in graph.edges():
        # To rebalance channel u->v, we send funds from u to v directly, and route back v to u.
        # So we need a path from v to u.
        paths = get_candidate_paths(graph, source=v, target=u, K=K, max_length=max_length)
        all_candidates[(u, v)] = paths
    return all_candidates
