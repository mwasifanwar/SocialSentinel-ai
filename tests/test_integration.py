# tests/test_integration.py
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.graph_analyzer import GraphAnalyzer
from src.core.content_moderator import ContentModerator
from src.core.influence_detector import InfluenceDetector

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.graph_analyzer = GraphAnalyzer()
        self.content_moderator = ContentModerator()
        self.influence_detector = InfluenceDetector()
        
        self.sample_edges = [(1, 2, {}), (2, 3, {}), (3, 4, {}), (4, 1, {}), (1, 3, {})]
        self.sample_texts = ["Good content", "Bad content with threats", "Neutral message"]
    
    def test_end_to_end_analysis(self):
        self.graph_analyzer.build_graph_from_edges(self.sample_edges)
        
        content_analyses = self.content_moderator.batch_analyze(self.sample_texts)
        content_scores = {i: analysis['confidence'] for i, analysis in enumerate(content_analyses)}
        
        influence_scores = self.influence_detector.calculate_influence_metrics(
            self.graph_analyzer.graph, content_scores
        )
        
        self.assertEqual(len(influence_scores), 4)
        self.assertTrue(all(0 <= score <= 1 for score in influence_scores.values()))

if __name__ == '__main__':
    unittest.main()