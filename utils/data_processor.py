# utils/data_processor.py
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
import networkx as nx

class DataProcessor:
    def __init__(self):
        self.raw_data = {}
        self.processed_data = {}
    
    def load_social_media_data(self, file_path: str, platform: str = "twitter") -> Dict[str, Any]:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                df = pd.DataFrame(json.load(f))
        else:
            raise ValueError("Unsupported file format")
        
        if platform == "twitter":
            return self._process_twitter_data(df)
        elif platform == "reddit":
            return self._process_reddit_data(df)
        else:
            return self._process_generic_data(df)
    
    def _process_twitter_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        edges = []
        node_features = {}
        content_data = {}
        
        for _, row in df.iterrows():
            user_id = row.get('user_id', hash(row.get('user_name', '')))
            if 'retweeted_user' in row and pd.notna(row['retweeted_user']):
                retweeted_id = hash(row['retweeted_user'])
                edges.append((user_id, retweeted_id, {'type': 'retweet'}))
            
            if 'mentioned_users' in row and pd.notna(row['mentioned_users']):
                mentions = eval(row['mentioned_users']) if isinstance(row['mentioned_users'], str) else row['mentioned_users']
                for mentioned_user in mentions:
                    mentioned_id = hash(mentioned_user)
                    edges.append((user_id, mentioned_id, {'type': 'mention'}))
            
            node_features[user_id] = np.array([
                row.get('followers_count', 0),
                row.get('friends_count', 0),
                row.get('statuses_count', 0),
                len(str(row.get('description', '')))
            ])
            
            content_data[user_id] = {
                'text': row.get('text', ''),
                'created_at': row.get('created_at', datetime.now()),
                'sentiment': row.get('sentiment', 0.0)
            }
        
        return {
            'edges': edges,
            'node_features': node_features,
            'content_data': content_data,
            'platform': 'twitter'
        }
    
    def _process_reddit_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        edges = []
        node_features = {}
        content_data = {}
        
        for _, row in df.iterrows():
            author = row.get('author', '')
            if author == '[deleted]':
                continue
            
            author_id = hash(author)
            
            if 'parent_author' in row and pd.notna(row['parent_author']):
                parent_author = row['parent_author']
                if parent_author != '[deleted]':
                    parent_id = hash(parent_author)
                    edges.append((author_id, parent_id, {'type': 'reply'}))
            
            node_features[author_id] = np.array([
                row.get('score', 0),
                len(str(row.get('body', ''))),
                row.get('controversiality', 0),
                row.get('gilded', 0)
            ])
            
            content_data[author_id] = {
                'text': row.get('body', ''),
                'created_at': row.get('created_utc', datetime.now()),
                'subreddit': row.get('subreddit', '')
            }
        
        return {
            'edges': edges,
            'node_features': node_features,
            'content_data': content_data,
            'platform': 'reddit'
        }
    
    def _process_generic_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        edges = []
        node_features = {}
        content_data = {}
        
        for _, row in df.iterrows():
            if 'source' in row and 'target' in row:
                source_id = row['source']
                target_id = row['target']
                edge_attrs = {k: v for k, v in row.items() if k not in ['source', 'target']}
                edges.append((source_id, target_id, edge_attrs))
            
            if 'node_id' in row:
                node_id = row['node_id']
                feature_columns = [col for col in df.columns if col.startswith('feature_')]
                features = [row[col] for col in feature_columns if col in row]
                if features:
                    node_features[node_id] = np.array(features)
                
                if 'content' in row:
                    content_data[node_id] = {
                        'text': row['content'],
                        'timestamp': row.get('timestamp', datetime.now())
                    }
        
        return {
            'edges': edges,
            'node_features': node_features,
            'content_data': content_data,
            'platform': 'generic'
        }
    
    def create_pyg_data(self, edges: List[Tuple], node_features: Dict[int, np.ndarray]) -> Any:
        try:
            import torch
            from torch_geometric.data import Data
            
            node_ids = list(node_features.keys())
            node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
            
            edge_index = []
            for source, target, _ in edges:
                if source in node_id_to_idx and target in node_id_to_idx:
                    edge_index.append([node_id_to_idx[source], node_id_to_idx[target]])
            
            if not edge_index:
                raise ValueError("No valid edges found")
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            feature_matrix = []
            for node_id in node_ids:
                feature_matrix.append(node_features[node_id])
            
            x = torch.tensor(np.array(feature_matrix), dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index)
        
        except ImportError:
            print("PyTorch Geometric not available")
            return None
    
    def normalize_features(self, features: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        if not features:
            return {}
        
        feature_matrix = np.array(list(features.values()))
        normalized_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
        
        normalized_features = {}
        for i, node_id in enumerate(features.keys()):
            normalized_features[node_id] = normalized_matrix[i]
        
        return normalized_features