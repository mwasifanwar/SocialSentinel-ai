# core/influence_detector.py
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple
from sklearn.cluster import DBSCAN
from collections import defaultdict

class InfluenceDetector:
    def __init__(self):
        self.influence_scores = {}
        self.community_influencers = {}
    
    def calculate_influence_metrics(self, graph: nx.Graph, content_scores: Dict[int, float] = None) -> Dict[int, float]:
        centrality_measures = {
            'degree': nx.degree_centrality(graph),
            'betweenness': nx.betweenness_centrality(graph),
            'closeness': nx.closeness_centrality(graph),
            'eigenvector': nx.eigenvector_centrality(graph, max_iter=1000)
        }
        
        influence_scores = {}
        for node in graph.nodes():
            structural_score = np.mean([
                centrality_measures['degree'][node],
                centrality_measures['betweenness'][node],
                centrality_measures['eigenvector'][node]
            ])
            
            if content_scores and node in content_scores:
                content_score = content_scores[node]
            else:
                content_score = 0.5
            
            final_score = 0.7 * structural_score + 0.3 * content_score
            influence_scores[node] = final_score
        
        self.influence_scores = influence_scores
        return influence_scores
    
    def detect_influence_clusters(self, graph: nx.Graph, eps: float = 0.1, min_samples: int = 2) -> Dict[int, int]:
        nodes = list(graph.nodes())
        if len(nodes) == 0:
            return {}
        
        influence_values = np.array([self.influence_scores[node] for node in nodes]).reshape(-1, 1)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(influence_values)
        labels = clustering.labels_
        
        clusters = {}
        for i, node in enumerate(nodes):
            clusters[node] = labels[i]
        
        return clusters
    
    def identify_community_leaders(self, graph: nx.Graph, communities: Dict[int, int]) -> Dict[int, List[Tuple[int, float]]]:
        community_leaders = {}
        
        for community_id in set(communities.values()):
            community_nodes = [node for node, comm in communities.items() if comm == community_id]
            
            if not community_nodes:
                continue
            
            community_scores = [(node, self.influence_scores[node]) for node in community_nodes]
            top_influencers = sorted(community_scores, key=lambda x: x[1], reverse=True)[:5]
            
            community_leaders[community_id] = top_influencers
        
        self.community_influencers = community_leaders
        return community_leaders
    
    def analyze_influence_spread(self, graph: nx.Graph, start_nodes: List[int], max_steps: int = 10) -> Dict[int, List[int]]:
        spread_patterns = {}
        
        for start_node in start_nodes:
            if start_node not in graph:
                continue
            
            visited = set([start_node])
            current_frontier = [start_node]
            spread_sequence = [start_node]
            
            for step in range(max_steps):
                next_frontier = []
                
                for node in current_frontier:
                    neighbors = list(graph.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.append(neighbor)
                            spread_sequence.append(neighbor)
                
                if not next_frontier:
                    break
                
                current_frontier = next_frontier
            
            spread_patterns[start_node] = spread_sequence
        
        return spread_patterns
    
    def get_influence_network_properties(self, graph: nx.Graph) -> Dict[str, Any]:
        if not self.influence_scores:
            return {}
        
        scores = list(self.influence_scores.values())
        
        return {
            'mean_influence': np.mean(scores),
            'std_influence': np.std(scores),
            'max_influence': max(scores),
            'min_influence': min(scores),
            'influence_gini': self._calculate_gini_coefficient(scores),
            'high_influence_nodes': len([s for s in scores if s > np.mean(scores) + np.std(scores)])
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumulative = np.cumsum(sorted_vals)
        total = cumulative[-1]
        
        if total == 0:
            return 0.0
        
        gini_sum = sum((2 * i - n - 1) * sorted_vals[i] for i in range(n))
        gini = gini_sum / (n * total)
        return gini