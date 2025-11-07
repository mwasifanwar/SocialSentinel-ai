# core/network_dynamics.py
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict

class NetworkDynamics:
    def __init__(self):
        self.temporal_graphs = {}
        self.information_cascades = {}
        self.network_evolution = {}
    
    def build_temporal_network(self, temporal_edges: List[Tuple[int, int, datetime]]):
        time_windows = defaultdict(list)
        
        for source, target, timestamp in temporal_edges:
            time_key = timestamp.replace(minute=0, second=0, microsecond=0)
            time_windows[time_key].append((source, target))
        
        for time_key, edges in time_windows.items():
            graph = nx.Graph()
            graph.add_edges_from(edges)
            self.temporal_graphs[time_key] = graph
    
    def track_information_cascade(self, source_node: int, start_time: datetime, 
                                content_analysis: Dict[int, Any]) -> Dict[str, Any]:
        cascade_nodes = set([source_node])
        cascade_timeline = [(source_node, start_time)]
        current_frontier = [source_node]
        
        time_sorted_graphs = sorted(self.temporal_graphs.items())
        
        for time_key, graph in time_sorted_graphs:
            if time_key < start_time:
                continue
            
            next_frontier = []
            for node in current_frontier:
                if node not in graph:
                    continue
                
                neighbors = list(graph.neighbors(node))
                for neighbor in neighbors:
                    if (neighbor not in cascade_nodes and 
                        self._should_adopt_content(neighbor, content_analysis)):
                        cascade_nodes.add(neighbor)
                        cascade_timeline.append((neighbor, time_key))
                        next_frontier.append(neighbor)
            
            current_frontier = next_frontier
            if not current_frontier:
                break
        
        cascade_id = f"cascade_{source_node}_{start_time.strftime('%Y%m%d%H%M')}"
        self.information_cascades[cascade_id] = {
            'source': source_node,
            'start_time': start_time,
            'nodes': list(cascade_nodes),
            'timeline': cascade_timeline,
            'size': len(cascade_nodes),
            'duration': (cascade_timeline[-1][1] - start_time) if cascade_timeline else timedelta(0)
        }
        
        return self.information_cascades[cascade_id]
    
    def _should_adopt_content(self, node: int, content_analysis: Dict[int, Any]) -> bool:
        if node not in content_analysis:
            return True
        
        analysis = content_analysis[node]
        harmful_threshold = 0.7
        
        if analysis.get('is_harmful', False) and analysis.get('confidence', 0) > harmful_threshold:
            return np.random.random() < 0.3
        else:
            return np.random.random() < 0.8
    
    def analyze_network_evolution(self, metric: str = 'density') -> Dict[datetime, float]:
        evolution_data = {}
        
        for time_key, graph in sorted(self.temporal_graphs.items()):
            if metric == 'density':
                value = nx.density(graph)
            elif metric == 'average_degree':
                value = np.mean([d for n, d in graph.degree()])
            elif metric == 'clustering':
                value = nx.average_clustering(graph)
            elif metric == 'components':
                value = nx.number_connected_components(graph)
            else:
                value = graph.number_of_nodes()
            
            evolution_data[time_key] = value
        
        self.network_evolution[metric] = evolution_data
        return evolution_data
    
    def detect_critical_periods(self, threshold_std: float = 2.0) -> List[Tuple[datetime, str]]:
        critical_periods = []
        
        for metric, evolution in self.network_evolution.items():
            values = list(evolution.values())
            if len(values) < 2:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            for time_key, value in evolution.items():
                if abs(value - mean_val) > threshold_std * std_val:
                    if value > mean_val:
                        event_type = f"high_{metric}"
                    else:
                        event_type = f"low_{metric}"
                    critical_periods.append((time_key, event_type))
        
        return critical_periods
    
    def predict_network_growth(self, days: int = 7) -> Dict[str, Any]:
        if not self.temporal_graphs:
            return {}
        
        time_points = sorted(self.temporal_graphs.keys())
        node_counts = [graph.number_of_nodes() for graph in self.temporal_graphs.values()]
        
        if len(node_counts) < 2:
            return {}
        
        x = np.arange(len(node_counts))
        coefficients = np.polyfit(x, node_counts, 1)
        future_growth = np.polyval(coefficients, np.arange(len(node_counts) + days))
        
        return {
            'current_nodes': node_counts[-1],
            'predicted_nodes': int(future_growth[-1]),
            'growth_rate': coefficients[0],
            'prediction_confidence': self._calculate_prediction_confidence(node_counts, future_growth[:len(node_counts)])
        }
    
    def _calculate_prediction_confidence(self, actual: List[int], predicted: List[float]) -> float:
        if len(actual) != len(predicted):
            return 0.0
        
        mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))
        max_val = max(actual)
        confidence = max(0, 1 - (mae / max_val)) if max_val > 0 else 0.0
        return confidence