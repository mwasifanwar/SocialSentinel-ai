# api/middleware.py
from fastapi import Request
import time
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityMiddleware:
    def __init__(self):
        self.request_counts: Dict[str, int] = {}
    
    async def __call__(self, request: Request, call_next):
        client_ip = request.client.host
        
        start_time = time.time()
        
        if self._is_rate_limited(client_ip):
            from fastapi import HTTPException
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
        
        self._update_rate_limit(client_ip)
        
        return response
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        current_time = time.time()
        window_start = current_time - 60
        
        self.request_counts = {ip: count for ip, count in self.request_counts.items() 
                              if self.request_counts.get('timestamp', 0) > window_start}
        
        client_data = self.request_counts.get(client_ip, {'count': 0, 'timestamp': current_time})
        
        return client_data['count'] > 100
    
    def _update_rate_limit(self, client_ip: str):
        current_time = time.time()
        
        if client_ip in self.request_counts:
            self.request_counts[client_ip]['count'] += 1
            self.request_counts[client_ip]['timestamp'] = current_time
        else:
            self.request_counts[client_ip] = {'count': 1, 'timestamp': current_time}