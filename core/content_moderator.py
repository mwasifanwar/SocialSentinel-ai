# core/content_moderator.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import Counter

class ContentModerator:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-offensive"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.harmful_patterns = self._initialize_harmful_patterns()
    
    def _initialize_harmful_patterns(self) -> Dict[str, List[str]]:
        return {
            'hate_speech': [
                r'\b(kill|hurt|harm)\s+(all|every)\s+\w+',
                r'\b(worthless|stupid|idiot)\s+\w+',
                r'\b(should\s+die|must\s+die)',
                r'\b(racial|racist)\s+slur',
                r'\b(discriminate\s+against)'
            ],
            'harassment': [
                r'\b(stalk|follow|harass)\s+\w+',
                r'\b(threaten|intimidate)\s+\w+',
                r'\b(unwanted\s+attention)',
                r'\b(creepy|stalker)'
            ],
            'violence': [
                r'\b(bomb|shoot|attack)\s+\w+',
                r'\b(violent|assault)\s+\w+',
                r'\b(weapon|gun|knife)\s+\w+',
                r'\b(fight|beat)\s+\w+'
            ]
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        pattern_matches = self._check_harmful_patterns(text)
        sentiment_score = self._analyze_sentiment(text)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'harmful_patterns': pattern_matches,
            'sentiment_score': sentiment_score,
            'is_harmful': confidence > 0.7 or len(pattern_matches) > 0
        }
    
    def _check_harmful_patterns(self, text: str) -> Dict[str, List[str]]:
        matches = {}
        for category, patterns in self.harmful_patterns.items():
            category_matches = []
            for pattern in patterns:
                found = re.findall(pattern, text.lower())
                if found:
                    category_matches.extend(found)
            if category_matches:
                matches[category] = category_matches
        return matches
    
    def _analyze_sentiment(self, text: str) -> float:
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst'}
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = pos_count + neg_count
        if total_sentiment_words == 0:
            return 0.0
        
        sentiment_score = (pos_count - neg_count) / total_sentiment_words
        return sentiment_score
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.analyze_text(text) for text in texts]
    
    def get_content_metrics(self, texts: List[str]) -> Dict[str, Any]:
        analyses = self.batch_analyze(texts)
        
        harmful_count = sum(1 for analysis in analyses if analysis['is_harmful'])
        avg_confidence = np.mean([analysis['confidence'] for analysis in analyses])
        pattern_counts = Counter()
        
        for analysis in analyses:
            for category in analysis['harmful_patterns']:
                pattern_counts[category] += 1
        
        return {
            'total_texts': len(texts),
            'harmful_content_count': harmful_count,
            'harmful_percentage': (harmful_count / len(texts)) * 100,
            'average_confidence': avg_confidence,
            'pattern_distribution': dict(pattern_counts)
        }