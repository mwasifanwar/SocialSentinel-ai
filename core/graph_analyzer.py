# core/graph_analyzer.py
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class GraphAnalyzer:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_features = {}
        self.centrality_measures = {}
        self.communities = {}
    
    def build_graph_from_edges(self, edges: List[Tuple[int, int, Dict]]):
        self.graph.clear()
        self.graph.add_edges_from(edges)
    
    def add_node_features(self, node_features: Dict[int, np.ndarray]):
        self.node_features = node_features
    
    def compute_centrality(self):
        self.centrality_measures = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'closeness': nx.closeness_centrality(self.graph),
            'eigenvector': nx.eigenvector_centrality(self.graph, max_iter=1000)
        }
        return self.centrality_measures
    
    def detect_communities(self, method: str = "louvain"):
        if method == "louvain":
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            self.communities = partition
        elif method == "label_propagation":
            communities = list(nx.algorithms.community.label_propagation_communities(self.graph))
            self.communities = {node: i for i, comm in enumerate(communities) for node in comm}
        
        return self.communities
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_degree': np.mean([d for n, d in self.graph.degree()]),
            'clustering_coefficient': nx.average_clustering(self.graph),
            'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else float('inf'),
            'connected_components': nx.number_connected_components(self.graph)
        }
        return stats
    
    def get_influential_nodes(self, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.centrality_measures:
            self.compute_centrality()
        
        combined_scores = {}
        for node in self.graph.nodes():
            score = np.mean([
                self.centrality_measures['degree'][node],
                self.centrality_measures['betweenness'][node],
                self.centrality_measures['eigenvector'][node]
            ])
            combined_scores[node] = score
        
        influential = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return influential