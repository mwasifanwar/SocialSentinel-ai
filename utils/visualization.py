# utils/visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class NetworkVisualizer:
    def __init__(self):
        self.color_palette = plt.cm.Set3
        self.layout_algorithms = {
            'spring': nx.spring_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout
        }
    
    def plot_network(self, graph: nx.Graph, node_colors: Dict[int, str] = None,
                    node_sizes: Dict[int, float] = None, layout: str = 'spring',
                    title: str = "Social Network") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        pos = self.layout_algorithms[layout](graph)
        
        if node_colors is None:
            node_colors = ['lightblue' for _ in graph.nodes()]
        else:
            node_colors = [node_colors.get(node, 'lightblue') for node in graph.nodes()]
        
        if node_sizes is None:
            node_sizes = [300 for _ in graph.nodes()]
        else:
            node_sizes = [node_sizes.get(node, 300) for node in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        return fig
    
    def plot_community_structure(self, graph: nx.Graph, communities: Dict[int, int]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        pos = nx.spring_layout(graph)
        
        unique_communities = set(communities.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        
        community_colors = {comm: colors[i] for i, comm in enumerate(unique_communities)}
        node_colors = [community_colors[communities[node]] for node in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=300, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)
        
        ax.set_title("Network Community Structure", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        return fig
    
    def plot_influence_distribution(self, influence_scores: Dict[int, float]) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scores = list(influence_scores.values())
        
        ax1.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Influence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Influence Scores')
        ax1.grid(True, alpha=0.3)
        
        sorted_scores = sorted(scores, reverse=True)
        ax2.plot(range(len(sorted_scores)), sorted_scores, 'b-', linewidth=2)
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Influence Score')
        ax2.set_title('Influence Score Rank Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_network(self, graph: nx.Graph, node_attributes: Dict[int, Dict] = None) -> go.Figure:
        pos = nx.spring_layout(graph)
        
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node_attributes and node in node_attributes:
                attrs = node_attributes[node]
                node_text.append(f"Node {node}<br>" + "<br>".join([f"{k}: {v}" for k, v in attrs.items()]))
                node_color.append(attrs.get('influence', 0.5))
            else:
                node_text.append(f"Node {node}")
                node_color.append(0.5)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Influence',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        node_trace.text = node_text
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive Social Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="SocialSentinel Network Visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 ) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig