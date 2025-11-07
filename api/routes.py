# api/routes.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.graph_analyzer import GraphAnalyzer
from src.core.content_moderator import ContentModerator
from src.core.influence_detector import InfluenceDetector
from src.core.network_dynamics import NetworkDynamics
from src.utils.data_processor import DataProcessor
from src.utils.metrics_calculator import MetricsCalculator

api_router = APIRouter()

class NetworkAnalysisRequest(BaseModel):
    edges: List[Tuple[int, int, Dict]]
    node_features: Optional[Dict[int, List[float]]] = None

class ContentAnalysisRequest(BaseModel):
    texts: List[str]
    language: str = "en"

class InfluenceDetectionRequest(BaseModel):
    graph_data: NetworkAnalysisRequest
    content_scores: Optional[Dict[int, float]] = None

graph_analyzer = GraphAnalyzer()
content_moderator = ContentModerator()
influence_detector = InfluenceDetector()
network_dynamics = NetworkDynamics()
data_processor = DataProcessor()
metrics_calculator = MetricsCalculator()

@api_router.post("/analyze-network")
async def analyze_network(request: NetworkAnalysisRequest):
    try:
        graph_analyzer.build_graph_from_edges(request.edges)
        
        if request.node_features:
            graph_analyzer.add_node_features(request.node_features)
        
        centrality = graph_analyzer.compute_centrality()
        communities = graph_analyzer.detect_communities()
        statistics = graph_analyzer.get_graph_statistics()
        influential_nodes = graph_analyzer.get_influential_nodes()
        
        return {
            "centrality_measures": centrality,
            "communities": communities,
            "graph_statistics": statistics,
            "influential_nodes": influential_nodes,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/moderate-content")
async def moderate_content(request: ContentAnalysisRequest):
    try:
        analyses = content_moderator.batch_analyze(request.texts)
        metrics = content_moderator.get_content_metrics(request.texts)
        
        return {
            "analyses": analyses,
            "metrics": metrics,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/detect-influence")
async def detect_influence(request: InfluenceDetectionRequest):
    try:
        graph_analyzer.build_graph_from_edges(request.graph_data.edges)
        
        if request.graph_data.node_features:
            graph_analyzer.add_node_features(request.graph_data.node_features)
        
        influence_scores = influence_detector.calculate_influence_metrics(
            graph_analyzer.graph, request.content_scores
        )
        
        influence_clusters = influence_detector.detect_influence_clusters(graph_analyzer.graph)
        communities = graph_analyzer.detect_communities()
        community_leaders = influence_detector.identify_community_leaders(
            graph_analyzer.graph, communities
        )
        
        network_properties = influence_detector.get_influence_network_properties(graph_analyzer.graph)
        
        return {
            "influence_scores": influence_scores,
            "influence_clusters": influence_clusters,
            "community_leaders": community_leaders,
            "network_properties": network_properties,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/upload-network-data")
async def upload_network_data(file: UploadFile = File(...), platform: str = "twitter"):
    try:
        contents = await file.read()
        
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        processed_data = data_processor.load_social_media_data(temp_path, platform)
        
        os.remove(temp_path)
        
        return {
            "processed_data": processed_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "SocialSentinel",
        "version": "1.0.0"
    }

@api_router.get("/metrics")
async def get_system_metrics():
    try:
        if hasattr(graph_analyzer, 'graph') and graph_analyzer.graph:
            network_metrics = metrics_calculator.calculate_network_metrics(graph_analyzer.graph)
        else:
            network_metrics = {}
        
        return {
            "network_metrics": network_metrics,
            "metrics_history": metrics_calculator.metrics_history,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))