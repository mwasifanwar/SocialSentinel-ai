# tests/test_content_moderator.py
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.content_moderator import ContentModerator

class TestContentModerator(unittest.TestCase):
    def setUp(self):
        self.moderator = ContentModerator()
    
    def test_text_analysis(self):
        test_text = "This is a normal message without harmful content."
        analysis = self.moderator.analyze_text(test_text)
        
        self.assertIn('predicted_class', analysis)
        self.assertIn('confidence', analysis)
        self.assertIn('is_harmful', analysis)
        self.assertIsInstance(analysis['confidence'], float)
    
    def test_harmful_pattern_detection(self):
        test_text = "I will kill everyone who disagrees with me"
        analysis = self.moderator.analyze_text(test_text)
        
        self.assertTrue(len(analysis['harmful_patterns']) > 0)
    
    def test_batch_analysis(self):
        texts = [
            "Hello world!",
            "This is great!",
            "I hate everyone"
        ]
        analyses = self.moderator.batch_analyze(texts)
        
        self.assertEqual(len(analyses), 3)
        self.assertIsInstance(analyses[0], dict)

if __name__ == '__main__':
    unittest.main()