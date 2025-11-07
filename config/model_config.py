# config/model_config.py
from typing import Dict, Any, List

class ModelConfig:
    CONTENT_MODERATION_MODELS = {
        "offensive": {
            "name": "cardiffnlp/twitter-roberta-base-offensive",
            "type": "hate_speech",
            "description": "RoBERTa base model for offensive language detection",
            "max_length": 512
        },
        "sentiment": {
            "name": "cardiffnlp/twitter-roberta-base-sentiment",
            "type": "sentiment",
            "description": "RoBERTa base model for sentiment analysis",
            "max_length": 512
        }
    }
    
    GNN_MODELS = {
        "GCN": {
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.3
        },
        "GAT": {
            "hidden_dim": 64,
            "num_heads": 8,
            "dropout": 0.2
        },
        "GraphSAGE": {
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.3
        }
    }
    
    ANALYSIS_PARAMS = {
        "community_detection": {
            "louvain_resolution": 1.0,
            "label_propagation_iterations": 100
        },
        "influence_detection": {
            "dbscan_eps": 0.1,
            "min_samples": 2
        },
        "network_dynamics": {
            "time_window_hours": 1,
            "cascade_max_steps": 10
        }
    }
    
    @classmethod
    def get_content_model_config(cls, model_type: str) -> Dict[str, Any]:
        return cls.CONTENT_MODERATION_MODELS.get(model_type, {})
    
    @classmethod
    def get_gnn_model_config(cls, model_type: str) -> Dict[str, Any]:
        return cls.GNN_MODELS.get(model_type, {})
    
    @classmethod
    def get_analysis_params(cls, analysis_type: str) -> Dict[str, Any]:
        return cls.ANALYSIS_PARAMS.get(analysis_type, {})