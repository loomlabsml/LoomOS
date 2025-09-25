"""
LoomOS Python SDK

Enterprise-grade Python SDK for LoomOS with comprehensive functionality:
- Async and sync job management
- Real-time log streaming and monitoring
- Cluster management and worker coordination
- Marketplace integration
- WebSocket support for real-time updates
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, AsyncIterator, Iterator, Union, Callable
from datetime import datetime, timezone
import httpx
import websockets
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)

class LoomOSError(Exception):
    """Base exception for LoomOS SDK"""
    pass

class JobNotFoundError(LoomOSError):
    """Job not found error"""
    pass

class AuthenticationError(LoomOSError):
    """Authentication error"""
    pass

class APIError(LoomOSError):
    """API error"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class JobSpec:
    """Job specification builder"""
    
    def __init__(self, job_type: str, name: str, image: str):
        self.spec = {
            "job_type": job_type,
            "name": name,
            "image": image
        }
    
    def command(self, command: List[str]) -> 'JobSpec':
        """Set command to execute"""
        self.spec["command"] = command
        return self
    
    def environment(self, env: Dict[str, str]) -> 'JobSpec':
        """Set environment variables"""
        self.spec["environment"] = env
        return self
    
    def resources(self, resource_type: str, amount: float, unit: str = "cores") -> 'JobSpec':
        """Add resource requirement"""
        if "resources" not in self.spec:
            self.spec["resources"] = []
        self.spec["resources"].append({
            "resource_type": resource_type,
            "amount": amount,
            "unit": unit
        })
        return self
    
    def priority(self, priority: int) -> 'JobSpec':
        """Set job priority (1-10)"""
        self.spec["priority"] = priority
        return self
    
    def timeout(self, timeout_seconds: int) -> 'JobSpec':
        """Set job timeout"""
        self.spec["timeout_seconds"] = timeout_seconds
        return self
    
    def retry_limit(self, retry_limit: int) -> 'JobSpec':
        """Set retry limit"""
        self.spec["retry_limit"] = retry_limit
        return self
    
    def distributed(self, world_size: int, compression_ratio: float = 0.01) -> 'JobSpec':
        """Configure for distributed training"""
        self.spec["world_size"] = world_size
        self.spec["compression_ratio"] = compression_ratio
        return self
    
    def tags(self, tags: Dict[str, str]) -> 'JobSpec':
        """Set job tags"""
        self.spec["tags"] = tags
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the job specification"""
        return self.spec.copy()

class AsyncLoomClient:
    """Async LoomOS client"""
    
    def __init__(self, api_url: str, token: Optional[str] = None, timeout: float = 300.0):
        self.api_url = api_url.rstrip('/')
        self.token = token
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Initialize the HTTP client"""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        self._client = httpx.AsyncClient(
            base_url=self.api_url,
            headers=headers,
            timeout=self.timeout
        )
    
    async def disconnect(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _ensure_client(self):
        """Ensure client is initialized"""
        if not self._client:
            raise LoomOSError("Client not connected. Use async with or call connect() first.")
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        self._ensure_client()
        
        try:
            response = await self._client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid authentication token")
            elif e.response.status_code == 404:
                raise JobNotFoundError("Resource not found")
            else:
                try:
                    error_detail = e.response.json().get("detail", str(e))
                except:
                    error_detail = str(e)
                raise APIError(f"API error ({e.response.status_code}): {error_detail}", e.response.status_code)
        except httpx.RequestError as e:
            raise LoomOSError(f"Request failed: {e}")
    
    # Job management
    async def submit_job(self, job_spec: Union[Dict[str, Any], JobSpec]) -> str:
        """Submit a job and return job ID"""
        if isinstance(job_spec, JobSpec):
            job_spec = job_spec.build()
        
        result = await self._request("POST", "/v1/jobs", json=job_spec)
        return result["job_id"]
    
    async def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status and details"""
        return await self._request("GET", f"/v1/jobs/{job_id}")
    
    async def list_jobs(self, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List jobs with optional filtering"""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        
        return await self._request("GET", "/v1/jobs", params=params)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        await self._request("DELETE", f"/v1/jobs/{job_id}")
        return True
    
    async def wait_for_job(self, job_id: str, poll_interval: float = 5.0, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for job completion"""
        start_time = time.time()
        
        while True:
            job = await self.get_job(job_id)
            status = job.get("status")
            
            if status in ["completed", "failed", "cancelled"]:
                return job
            
            if timeout and (time.time() - start_time) > timeout:
                raise LoomOSError(f"Timeout waiting for job {job_id}")
            
            await asyncio.sleep(poll_interval)
    
    async def stream_logs(self, job_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Stream job logs"""
        self._ensure_client()
        
        try:
            async with self._client.stream("GET", f"/v1/jobs/{job_id}/logs") as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            log_entry = json.loads(line[6:])  # Remove "data: " prefix
                            yield log_entry
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse log line: {line}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise JobNotFoundError(f"Job {job_id} not found")
            raise APIError(f"Failed to stream logs: {e}")
    
    # Cluster management
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics"""
        return await self._request("GET", "/v1/cluster/stats")
    
    async def list_workers(self) -> List[Dict[str, Any]]:
        """List cluster workers"""
        return await self._request("GET", "/v1/workers")
    
    # Health and monitoring
    async def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        return await self._request("GET", "/v1/health")
    
    # Marketplace
    async def list_marketplace_listings(self, listing_type: Optional[str] = None) -> Dict[str, Any]:
        """List marketplace listings"""
        params = {}
        if listing_type:
            params["listing_type"] = listing_type
        
        return await self._request("GET", "/v1/marketplace/listings", params=params)
    
    async def purchase_listing(self, listing_id: str, quantity: int = 1) -> Dict[str, Any]:
        """Purchase a marketplace listing"""
        return await self._request("POST", "/v1/marketplace/purchase", json={
            "listing_id": listing_id,
            "quantity": quantity
        })
    
    # AI operations
    async def execute_agent_task(self, prompt: str, tools: Optional[List[str]] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute AI agent task"""
        payload = {"prompt": prompt}
        if tools:
            payload["tools"] = tools
        if context:
            payload["context"] = context
        
        return await self._request("POST", "/v1/ai/agents/execute", json=payload)
    
    async def adapt_model(self, model_id: str, adaptation_type: str, training_data: Dict, config: Dict) -> Dict[str, Any]:
        """Adapt a model using Forge"""
        return await self._request("POST", "/v1/ai/models/adapt", json={
            "model_id": model_id,
            "adaptation_type": adaptation_type,
            "training_data": training_data,
            "config": config
        })
    
    async def verify_content(self, content: str, verification_type: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Verify content using Prism"""
        payload = {
            "content": content,
            "type": verification_type
        }
        if context:
            payload["context"] = context
        
        return await self._request("POST", "/v1/ai/verify", json=payload)

class LoomClient:
    """Synchronous LoomOS client (wrapper around AsyncLoomClient)"""
    
    def __init__(self, api_url: str, token: Optional[str] = None, timeout: float = 300.0):
        self.api_url = api_url
        self.token = token
        self.timeout = timeout
        self._async_client = AsyncLoomClient(api_url, token, timeout)
    
    def __enter__(self):
        asyncio.run(self._async_client.connect())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self._async_client.disconnect())
    
    def _run_async(self, coro):
        """Run async coroutine in sync context"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coro)
    
    # Job management
    def submit_job(self, job_spec: Union[Dict[str, Any], JobSpec]) -> str:
        """Submit a job and return job ID"""
        return self._run_async(self._async_client.submit_job(job_spec))
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status and details"""
        return self._run_async(self._async_client.get_job(job_id))
    
    def list_jobs(self, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List jobs with optional filtering"""
        return self._run_async(self._async_client.list_jobs(status, limit, offset))
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        return self._run_async(self._async_client.cancel_job(job_id))
    
    def wait_for_job(self, job_id: str, poll_interval: float = 5.0, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for job completion"""
        return self._run_async(self._async_client.wait_for_job(job_id, poll_interval, timeout))
    
    def stream_logs(self, job_id: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Iterator[Dict[str, Any]]:
        """Stream job logs (sync generator)"""
        async def _stream():
            async for log_entry in self._async_client.stream_logs(job_id):
                if callback:
                    callback(log_entry)
                yield log_entry
        
        # Convert async generator to sync
        async_gen = _stream()
        try:
            while True:
                yield self._run_async(async_gen.__anext__())
        except StopAsyncIteration:
            pass
    
    # Cluster management
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics"""
        return self._run_async(self._async_client.get_cluster_stats())
    
    def list_workers(self) -> List[Dict[str, Any]]:
        """List cluster workers"""
        return self._run_async(self._async_client.list_workers())
    
    # Health and monitoring
    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        return self._run_async(self._async_client.health_check())
    
    # Marketplace
    def list_marketplace_listings(self, listing_type: Optional[str] = None) -> Dict[str, Any]:
        """List marketplace listings"""
        return self._run_async(self._async_client.list_marketplace_listings(listing_type))
    
    def purchase_listing(self, listing_id: str, quantity: int = 1) -> Dict[str, Any]:
        """Purchase a marketplace listing"""
        return self._run_async(self._async_client.purchase_listing(listing_id, quantity))
    
    # AI operations
    def execute_agent_task(self, prompt: str, tools: Optional[List[str]] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute AI agent task"""
        return self._run_async(self._async_client.execute_agent_task(prompt, tools, context))
    
    def adapt_model(self, model_id: str, adaptation_type: str, training_data: Dict, config: Dict) -> Dict[str, Any]:
        """Adapt a model using Forge"""
        return self._run_async(self._async_client.adapt_model(model_id, adaptation_type, training_data, config))
    
    def verify_content(self, content: str, verification_type: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Verify content using Prism"""
        return self._run_async(self._async_client.verify_content(content, verification_type, context))

# Convenience functions
def create_training_job(name: str, image: str, command: List[str], world_size: int = 1) -> JobSpec:
    """Create a training job specification"""
    return (JobSpec("training", name, image)
            .command(command)
            .distributed(world_size)
            .resources("gpu", world_size)
            .resources("memory", 32 * world_size, "GB"))

def create_inference_job(name: str, image: str, command: List[str]) -> JobSpec:
    """Create an inference job specification"""
    return (JobSpec("inference", name, image)
            .command(command)
            .resources("gpu", 1)
            .resources("memory", 16, "GB"))

def create_evaluation_job(name: str, image: str, command: List[str]) -> JobSpec:
    """Create an evaluation job specification"""
    return (JobSpec("evaluation", name, image)
            .command(command)
            .resources("cpu", 4)
            .resources("memory", 8, "GB"))

# Export public API
__all__ = [
    "LoomClient",
    "AsyncLoomClient", 
    "JobSpec",
    "LoomOSError",
    "JobNotFoundError",
    "AuthenticationError",
    "APIError",
    "create_training_job",
    "create_inference_job", 
    "create_evaluation_job"
]