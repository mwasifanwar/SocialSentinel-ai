# utils/metrics_calculator.py
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import networkx as nx

class MetricsCalculator:
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_moderation_metrics(self, ground_truth: List[bool], 
                                   predictions: List[bool]) -> Dict[str, float]:
        if len(ground_truth) != len(predictions):
            raise ValueError("Ground truth and predictions must have same length")
        
        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall = recall_score(ground_truth, predictions, zero_division=0)
        f1 = f1_score(ground_truth, predictions, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(ground_truth, predictions)
        except:
            auc_roc = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'accuracy': np.mean(np.array(ground_truth) == np.array(predictions))
        }
    
    def calculate_network_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        metrics = {}
        
        try:
            metrics['average_degree'] = np.mean([d for n, d in graph.degree()])
            metrics['density'] = nx.density(graph)
            metrics['clustering_coefficient'] = nx.average_clustering(graph)
            
            if nx.is_connected(graph):
                metrics['diameter'] = nx.diameter(graph)
                metrics['average_path_length'] = nx.average_shortest_path_length(graph)
            else:
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                metrics['diameter'] = nx.diameter(subgraph)
                metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
            
            metrics['assortativity'] = nx.degree_assortativity_coefficient(graph)
            metrics['modularity'] = self._estimate_modularity(graph)
            
        except Exception as e:
            print(f"Error calculating network metrics: {e}")
        
        return metrics
    
    def _estimate_modularity(self, graph: nx.Graph) -> float:
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
            modularity = community_louvain.modularity(partition, graph)
            return modularity
        except:
            return 0.0
    
    def calculate_influence_metrics(self, actual_influence: Dict[int, float],
                                  predicted_influence: Dict[int, float]) -> Dict[str, float]:
        common_nodes = set(actual_influence.keys()) & set(predicted_influence.keys())
        
        if not common_nodes:
            return {}
        
        actual_scores = [actual_influence[node] for node in common_nodes]
        predicted_scores = [predicted_influence[node] for node in common_nodes]
        
        correlation = np.corrcoef(actual_scores, predicted_scores)[0, 1]
        mae = np.mean(np.abs(np.array(actual_scores) - np.array(predicted_scores)))
        rmse = np.sqrt(np.mean((np.array(actual_scores) - np.array(predicted_scores)) ** 2))
        
        return {
            'pearson_correlation': correlation,
            'mae': mae,
            'rmse': rmse,
            'r_squared': 1 - (np.sum((np.array(actual_scores) - np.array(predicted_scores)) ** 2) / 
                             np.sum((np.array(actual_scores) - np.mean(actual_scores)) ** 2))
        }
    
    def calculate_cascade_metrics(self, cascades: Dict[str, Any]) -> Dict[str, float]:
        if not cascades:
            return {}
        
        sizes = [cascade['size'] for cascade in cascades.values()]
        durations = [cascade['duration'].total_seconds() for cascade in cascades.values() 
                    if hasattr(cascade['duration'], 'total_seconds')]
        
        metrics = {
            'avg_cascade_size': np.mean(sizes) if sizes else 0,
            'max_cascade_size': max(sizes) if sizes else 0,
            'avg_cascade_duration': np.mean(durations) if durations else 0,
            'cascade_count': len(cascades),
            'virality_coefficient': np.std(sizes) / np.mean(sizes) if sizes and np.mean(sizes) > 0 else 0
        }
        
        return metrics
    
    def track_metrics_over_time(self, metrics: Dict[str, float], timestamp: str):
        if timestamp not in self.metrics_history:
            self.metrics_history[timestamp] = {}
        
        self.metrics_history[timestamp].update(metrics)
    
    def get_metrics_trend(self, metric_name: str) -> List[Tuple[str, float]]:
        trend = []
        for timestamp, metrics in sorted(self.metrics_history.items()):
            if metric_name in metrics:
                trend.append((timestamp, metrics[metric_name]))
        return trend