# config/settings.py
import os
from typing import Dict, Any, List

class Settings:
    def __init__(self):
        self.API_HOST = os.getenv("SOCIAL_SENTINEL_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("SOCIAL_SENTINEL_PORT", "8000"))
        self.DEBUG = os.getenv("SOCIAL_SENTINEL_DEBUG", "False").lower() == "true"
        
        self.MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")
        self.DATA_STORAGE_DIR = os.getenv("DATA_STORAGE_DIR", "./data_storage")
        
        self.SUPPORTED_PLATFORMS = ["twitter", "reddit", "generic"]
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "100000000"))
        
        self.SECURITY_ENABLED = os.getenv("SECURITY_ENABLED", "True").lower() == "true"
        self.RATE_LIMITING_ENABLED = os.getenv("RATE_LIMITING_ENABLED", "True").lower() == "true"
        
        self.DEFAULT_COMMUNITY_METHOD = os.getenv("DEFAULT_COMMUNITY_METHOD", "louvain")
        self.INFLUENCE_THRESHOLD = float(os.getenv("INFLUENCE_THRESHOLD", "0.7"))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'API_HOST': self.API_HOST,
            'API_PORT': self.API_PORT,
            'DEBUG': self.DEBUG,
            'MODEL_CACHE_DIR': self.MODEL_CACHE_DIR,
            'DATA_STORAGE_DIR': self.DATA_STORAGE_DIR,
            'SUPPORTED_PLATFORMS': self.SUPPORTED_PLATFORMS,
            'MAX_FILE_SIZE': self.MAX_FILE_SIZE,
            'SECURITY_ENABLED': self.SECURITY_ENABLED,
            'RATE_LIMITING_ENABLED': self.RATE_LIMITING_ENABLED,
            'DEFAULT_COMMUNITY_METHOD': self.DEFAULT_COMMUNITY_METHOD,
            'INFLUENCE_THRESHOLD': self.INFLUENCE_THRESHOLD
        }