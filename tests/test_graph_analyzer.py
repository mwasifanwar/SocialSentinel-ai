# tests/test_graph_analyzer.py
import unittest
import sys
import os
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.graph_analyzer import GraphAnalyzer

class TestGraphAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = GraphAnalyzer()
        self.sample_edges = [(1, 2, {}), (2, 3, {}), (3, 4, {}), (4, 1, {})]
        self.sample_features = {1: [1.0, 2.0], 2: [2.0, 3.0], 3: [3.0, 4.0], 4: [4.0, 5.0]}
    
    def test_build_graph(self):
        self.analyzer.build_graph_from_edges(self.sample_edges)
        self.assertEqual(self.analyzer.graph.number_of_nodes(), 4)
        self.assertEqual(self.analyzer.graph.number_of_edges(), 4)
    
    def test_centrality_computation(self):
        self.analyzer.build_graph_from_edges(self.sample_edges)
        centrality = self.analyzer.compute_centrality()
        
        self.assertIn('degree', centrality)
        self.assertIn('betweenness', centrality)
        self.assertEqual(len(centrality['degree']), 4)
    
    def test_community_detection(self):
        self.analyzer.build_graph_from_edges(self.sample_edges)
        communities = self.analyzer.detect_communities()
        
        self.assertEqual(len(communities), 4)
        self.assertTrue(all(node in communities for node in [1, 2, 3, 4]))

if __name__ == '__main__':
    unittest.main()