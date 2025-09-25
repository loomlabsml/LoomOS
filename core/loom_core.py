"""
LoomOS Core Runtime Engine

This module provides the central orchestration engine for LoomOS, handling
request lifecycle, resource management, and component coordination.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import uuid
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestStatus(Enum):
    """Request processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ExecutionContext:
    """Execution context for request processing"""
    request_id: str
    user_id: str
    tenant_id: str
    priority: RequestPriority = RequestPriority.NORMAL
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

class Request(BaseModel):
    """Enhanced request model with full validation and metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = Field(..., description="Request type identifier")
    payload: Dict[str, Any] = Field(..., description="Request payload data")
    context: ExecutionContext = Field(default_factory=lambda: ExecutionContext(
        request_id=str(uuid.uuid4()),
        user_id="system",
        tenant_id="default"
    ))
    
    class Config:
        arbitrary_types_allowed = True

@dataclass
class ProcessingResult:
    """Result of request processing"""
    request_id: str
    status: RequestStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

class LoomCore:
    """
    LoomOS Core Runtime Engine
    
    Provides centralized orchestration for AI workloads with:
    - Request lifecycle management
    - Resource allocation and monitoring
    - Component coordination
    - Metrics collection
    - Error handling and recovery
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LoomCore with configuration"""
        self.config = config or {}
        self.active_requests: Dict[str, Request] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.result_cache: Dict[str, ProcessingResult] = {}
        self.metrics: Dict[str, float] = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "avg_processing_time_ms": 0.0,
            "queue_depth": 0
        }
        self._shutdown_event = asyncio.Event()
        self._worker_tasks: List[asyncio.Task] = []
        
        # Initialize component integrations
        self._init_components()
        
        logger.info("LoomCore initialized with config: %s", json.dumps(self.config, indent=2))

    def _init_components(self):
        """Initialize connections to other LoomOS components"""
        from .weaver import Weaver
        from .spinner import Spinner
        from .tapestry import Tapestry
        from .loom_agent import LoomAgent
        from .prism import Prism
        from .forge import Forge
        
        self.weaver = Weaver()
        self.spinner = Spinner()
        self.tapestry = Tapestry()
        self.agent = LoomAgent()
        self.prism = Prism()
        self.forge = Forge()
        
        logger.info("Core components initialized")

    async def start(self) -> None:
        """Start the LoomCore processing engine"""
        logger.info("Starting LoomCore processing engine...")
        
        # Start worker tasks
        worker_count = self.config.get("worker_count", 4)
        for i in range(worker_count):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        # Start metrics collection
        asyncio.create_task(self._metrics_loop())
        
        logger.info("LoomCore started with %d workers", worker_count)

    async def stop(self) -> None:
        """Gracefully stop the LoomCore engine"""
        logger.info("Stopping LoomCore...")
        
        self._shutdown_event.set()
        
        # Wait for workers to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        logger.info("LoomCore stopped")

    async def submit_request(self, request: Request) -> str:
        """Submit a request for processing"""
        self.active_requests[request.id] = request
        await self.processing_queue.put(request)
        
        self.metrics["requests_total"] += 1
        self.metrics["queue_depth"] = self.processing_queue.qsize()
        
        logger.info("Request submitted: %s (type: %s)", request.id, request.type)
        return request.id

    async def get_result(self, request_id: str) -> Optional[ProcessingResult]:
        """Get processing result for a request"""
        return self.result_cache.get(request_id)

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or processing request"""
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            result = ProcessingResult(
                request_id=request_id,
                status=RequestStatus.CANCELLED,
                error="Request cancelled by user"
            )
            self.result_cache[request_id] = result
            del self.active_requests[request_id]
            
            logger.info("Request cancelled: %s", request_id)
            return True
        return False

    async def process_request(self, request: Request) -> ProcessingResult:
        """
        Process a single request through the LoomOS pipeline
        
        This is the main processing pipeline that coordinates all components
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info("Processing request %s (type: %s)", request.id, request.type)
            
            # Step 1: Trace initialization with Weaver
            trace_id = await self._init_trace(request)
            
            # Step 2: Memory lookup with Tapestry
            context_data = await self._lookup_context(request)
            
            # Step 3: Route request based on type
            if request.type == "inference":
                result_data = await self._process_inference(request, context_data)
            elif request.type == "training":
                result_data = await self._process_training(request, context_data)
            elif request.type == "agent_task":
                result_data = await self._process_agent_task(request, context_data)
            elif request.type == "verification":
                result_data = await self._process_verification(request, context_data)
            else:
                raise ValueError(f"Unknown request type: {request.type}")
            
            # Step 4: Verification with Prism
            verification_result = await self._verify_result(request, result_data)
            
            # Step 5: Store results and update traces
            await self._store_result(request, result_data, trace_id)
            
            # Calculate metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = ProcessingResult(
                request_id=request.id,
                status=RequestStatus.COMPLETED,
                result={
                    "data": result_data,
                    "verification": verification_result,
                    "trace_id": trace_id
                },
                duration_ms=duration,
                resource_usage=await self._collect_resource_usage(),
                metrics={"processing_time_ms": duration}
            )
            
            self.metrics["requests_success"] += 1
            logger.info("Request completed: %s (%.2fms)", request.id, duration)
            
            return result
            
        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            result = ProcessingResult(
                request_id=request.id,
                status=RequestStatus.FAILED,
                error=str(e),
                duration_ms=duration
            )
            
            self.metrics["requests_failed"] += 1
            logger.error("Request failed: %s - %s", request.id, str(e), exc_info=True)
            
            return result

    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop for processing requests"""
        logger.info("Worker %s started", worker_id)
        
        while not self._shutdown_event.is_set():
            try:
                # Get request from queue with timeout
                request = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=1.0
                )
                
                # Process the request
                result = await self.process_request(request)
                
                # Store result and cleanup
                self.result_cache[request.id] = result
                if request.id in self.active_requests:
                    del self.active_requests[request.id]
                
                self.metrics["queue_depth"] = self.processing_queue.qsize()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Worker %s error: %s", worker_id, str(e), exc_info=True)
        
        logger.info("Worker %s stopped", worker_id)

    async def _metrics_loop(self) -> None:
        """Metrics collection loop"""
        while not self._shutdown_event.is_set():
            try:
                # Update average processing time
                completed_requests = [r for r in self.result_cache.values() 
                                    if r.status == RequestStatus.COMPLETED]
                if completed_requests:
                    avg_time = sum(r.duration_ms for r in completed_requests) / len(completed_requests)
                    self.metrics["avg_processing_time_ms"] = avg_time
                
                # Additional metrics can be collected here
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error("Metrics collection error: %s", str(e))

    async def _init_trace(self, request: Request) -> str:
        """Initialize trace for request"""
        trace_data = {
            "request_id": request.id,
            "type": request.type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": request.context.__dict__
        }
        return await self.weaver.create_trace(trace_data)

    async def _lookup_context(self, request: Request) -> Dict[str, Any]:
        """Lookup relevant context for request"""
        return await self.tapestry.retrieve_context(
            request.context.user_id,
            request.type,
            request.payload
        )

    async def _process_inference(self, request: Request, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference request"""
        return await self.spinner.execute_inference(request.payload, context)

    async def _process_training(self, request: Request, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process training request"""
        return await self.forge.execute_training(request.payload, context)

    async def _process_agent_task(self, request: Request, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent task request"""
        return await self.agent.execute_task(request.payload, context)

    async def _process_verification(self, request: Request, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process verification request"""
        return await self.prism.verify_content(request.payload, context)

    async def _verify_result(self, request: Request, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify processing result"""
        return await self.prism.verify_result(result_data, request.context.metadata)

    async def _store_result(self, request: Request, result_data: Dict[str, Any], trace_id: str) -> None:
        """Store result and update traces"""
        await self.tapestry.store_result(request.id, result_data)
        await self.weaver.update_trace(trace_id, {
            "result": result_data,
            "completed_at": datetime.now(timezone.utc).isoformat()
        })

    async def _collect_resource_usage(self) -> Dict[str, Any]:
        """Collect current resource usage metrics"""
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        return self.metrics.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "active_requests": len(self.active_requests),
            "queue_depth": self.processing_queue.qsize(),
            "cached_results": len(self.result_cache),
            "worker_count": len(self._worker_tasks),
            "uptime_seconds": (datetime.now(timezone.utc) - 
                             datetime.now(timezone.utc)).total_seconds(),
            "metrics": self.get_metrics()
        }