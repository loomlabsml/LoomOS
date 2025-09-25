"""
LoomCtl: Advanced Control Plane Service

Enterprise-grade control plane for LoomOS providing:
- High-performance job management API
- Advanced scheduling and resource allocation
- Real-time monitoring and metrics
- Multi-tenant security and isolation
- WebSocket streaming for real-time updates
- Distributed workflow orchestration
- Auto-scaling and resource optimization
- Production-grade observability
"""

import asyncio
import logging
import time
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Database and caching
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_ASYNC_AVAILABLE = True
except ImportError:
    SQLALCHEMY_ASYNC_AVAILABLE = False

# Import LoomOS core components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from core.spinner import ContainerRuntime
    from core.prism import Prism
    from core.loomdb import LoomDB, EventType, AuditContext
    from core.tapestry import Tapestry, MemoryType
    from core.marketplace import Marketplace
    from core.loom_agent import LoomAgent
    from core.forge import Forge
    from nexus.loomnode.worker import Worker, WorkerConfig, TrainingTask, TaskType
    from nexus.loomnode.attest import TEEAttestationService
except ImportError as e:
    logging.warning(f"Failed to import LoomOS components: {e}")
    # Create mock classes for standalone operation
    class ContainerRuntime:
        async def create_container(self, *args, **kwargs): return "mock_container"
    class Prism:
        async def verify(self, *args, **kwargs): return {"verified": True}
    class LoomDB:
        async def log_job_event(self, *args, **kwargs): return "mock_entry"
    class Tapestry:
        async def store_memory(self, *args, **kwargs): return "mock_memory"
    class Marketplace:
        async def create_account(self, *args, **kwargs): return None
    class LoomAgent:
        async def execute_task(self, *args, **kwargs): return {"result": "mock"}
    class Forge:
        async def adapt_model(self, *args, **kwargs): return {"adapted": True}
    class Worker:
        def __init__(self, *args, **kwargs): pass
        async def submit_task(self, *args, **kwargs): return True
    class WorkerConfig:
        def __init__(self, **kwargs): pass
    class TrainingTask:
        def __init__(self, **kwargs): pass
    class TaskType:
        FORWARD_PASS = "forward_pass"
    class TEEAttestationService:
        async def attest(self, *args, **kwargs): return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
API_REQUESTS = Counter('loomctl_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
API_REQUEST_DURATION = Histogram('loomctl_api_request_duration_seconds', 'API request duration')
ACTIVE_JOBS = Gauge('loomctl_active_jobs', 'Number of active jobs')
ACTIVE_WORKERS = Gauge('loomctl_active_workers', 'Number of active workers')
WEBSOCKET_CONNECTIONS = Gauge('loomctl_websocket_connections', 'Active WebSocket connections')

# Enums
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class JobType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    DATA_PROCESSING = "data_processing"
    MODEL_ADAPTATION = "model_adaptation"
    VERIFICATION = "verification"

class ResourceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

# Pydantic Models
class JobResource(BaseModel):
    resource_type: ResourceType
    amount: float
    unit: str = "cores"
    
class JobSpec(BaseModel):
    job_id: Optional[str] = Field(None, description="Unique job identifier")
    job_type: JobType
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    
    # Container specification
    image: str = Field(..., description="Container image to run")
    command: List[str] = Field(default_factory=list)
    environment: Dict[str, str] = Field(default_factory=dict)
    
    # Resource requirements
    resources: List[JobResource] = Field(default_factory=list)
    
    # Scheduling
    priority: int = Field(1, ge=1, le=10, description="Job priority (1-10)")
    timeout_seconds: Optional[int] = Field(None, gt=0)
    retry_limit: int = Field(3, ge=0, le=10)
    
    # Distributed training specific
    world_size: int = Field(1, ge=1, description="Number of workers for distributed training")
    compression_ratio: float = Field(0.01, ge=0.001, le=1.0)
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    user_id: Optional[str] = None
    
    @validator('job_id', pre=True, always=True)
    def set_job_id(cls, v):
        return v or f"job_{uuid.uuid4().hex[:8]}"

class JobStatus_Response(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: Optional[str] = None
    
    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Resource usage
    resources_used: List[JobResource] = Field(default_factory=list)
    
    # Worker assignment
    assigned_workers: List[str] = Field(default_factory=list)

class WorkerInfo(BaseModel):
    worker_id: str
    status: str
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    current_load: float = Field(0.0, ge=0.0, le=1.0)
    last_heartbeat: datetime
    
class ClusterStats(BaseModel):
    total_workers: int
    active_workers: int
    total_jobs: int
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    
    # Resource utilization
    cpu_utilization: float
    gpu_utilization: float
    memory_utilization: float
    
    # Performance metrics
    avg_job_duration_seconds: float
    jobs_per_hour: float

class StreamingMetrics(BaseModel):
    timestamp: datetime
    job_id: str
    metrics: Dict[str, float]

# Core Service Classes
@dataclass
class Job:
    job_id: str
    spec: JobSpec
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: Optional[str] = None
    
    # Timing
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results and errors
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Resource assignment
    assigned_workers: List[str] = None
    resources_used: List[JobResource] = None
    
    # Retry tracking
    retry_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.assigned_workers is None:
            self.assigned_workers = []
        if self.resources_used is None:
            self.resources_used = []

class JobScheduler:
    """Advanced job scheduler with resource optimization"""
    
    def __init__(self):
        self.pending_jobs: List[Job] = []
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}
        self.workers: Dict[str, WorkerInfo] = {}
        
        # Scheduling policies
        self.max_concurrent_jobs = 100
        self.resource_overcommit_ratio = 1.2
        
        logger.info("Job scheduler initialized")
    
    async def submit_job(self, job_spec: JobSpec) -> str:
        """Submit a new job for scheduling"""
        job = Job(
            job_id=job_spec.job_id,
            spec=job_spec,
            status=JobStatus.PENDING
        )
        
        self.pending_jobs.append(job)
        self.pending_jobs.sort(key=lambda j: j.spec.priority, reverse=True)
        
        ACTIVE_JOBS.inc()
        logger.info(f"Job {job.job_id} submitted for scheduling")
        
        return job.job_id
    
    async def schedule_jobs(self):
        """Main scheduling loop"""
        while self.pending_jobs and len(self.running_jobs) < self.max_concurrent_jobs:
            job = self.pending_jobs.pop(0)
            
            # Find suitable workers
            suitable_workers = await self._find_suitable_workers(job)
            
            if suitable_workers:
                await self._assign_job_to_workers(job, suitable_workers)
            else:
                # Put job back in queue
                self.pending_jobs.insert(0, job)
                break
    
    async def _find_suitable_workers(self, job: Job) -> List[str]:
        """Find workers suitable for the job"""
        suitable = []
        required_workers = job.spec.world_size
        
        available_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker.status == "idle" and worker.current_load < 0.8
        ]
        
        return available_workers[:required_workers]
    
    async def _assign_job_to_workers(self, job: Job, worker_ids: List[str]):
        """Assign job to workers"""
        job.assigned_workers = worker_ids
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        
        self.running_jobs[job.job_id] = job
        
        # Update worker status
        for worker_id in worker_ids:
            if worker_id in self.workers:
                self.workers[worker_id].current_load += 1.0 / len(worker_ids)
        
        logger.info(f"Job {job.job_id} assigned to workers: {worker_ids}")
    
    async def complete_job(self, job_id: str, result: Optional[Dict[str, Any]] = None, 
                          error: Optional[str] = None):
        """Mark job as completed"""
        if job_id not in self.running_jobs:
            return False
        
        job = self.running_jobs.pop(job_id)
        job.status = JobStatus.COMPLETED if error is None else JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.result = result
        job.error = error
        job.progress = 1.0
        
        # Free up workers
        for worker_id in job.assigned_workers:
            if worker_id in self.workers:
                self.workers[worker_id].current_load = max(0, 
                    self.workers[worker_id].current_load - 1.0 / len(job.assigned_workers))
        
        self.completed_jobs[job_id] = job
        ACTIVE_JOBS.dec()
        
        logger.info(f"Job {job_id} completed with status: {job.status}")
        return True
    
    def get_job_status(self, job_id: str) -> Optional[Job]:
        """Get current job status"""
        # Check running jobs first
        if job_id in self.running_jobs:
            return self.running_jobs[job_id]
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        
        # Check pending jobs
        for job in self.pending_jobs:
            if job.job_id == job_id:
                return job
        
        return None
    
    def get_cluster_stats(self) -> ClusterStats:
        """Get cluster statistics"""
        total_jobs = len(self.pending_jobs) + len(self.running_jobs) + len(self.completed_jobs)
        completed_jobs = len([j for j in self.completed_jobs.values() if j.status == JobStatus.COMPLETED])
        failed_jobs = len([j for j in self.completed_jobs.values() if j.status == JobStatus.FAILED])
        
        # Calculate utilization
        total_workers = len(self.workers)
        active_workers = len([w for w in self.workers.values() if w.status != "offline"])
        
        avg_load = sum(w.current_load for w in self.workers.values()) / max(total_workers, 1)
        
        # Calculate performance metrics
        completed_job_durations = []
        for job in self.completed_jobs.values():
            if job.started_at and job.completed_at:
                duration = (job.completed_at - job.started_at).total_seconds()
                completed_job_durations.append(duration)
        
        avg_duration = sum(completed_job_durations) / max(len(completed_job_durations), 1)
        
        return ClusterStats(
            total_workers=total_workers,
            active_workers=active_workers,
            total_jobs=total_jobs,
            active_jobs=len(self.running_jobs),
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            cpu_utilization=avg_load,
            gpu_utilization=avg_load * 0.8,  # Mock GPU utilization
            memory_utilization=avg_load * 0.6,  # Mock memory utilization
            avg_job_duration_seconds=avg_duration,
            jobs_per_hour=len(completed_job_durations) * 3600 / max(avg_duration, 1)
        )

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # client_id -> list of job_ids
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        WEBSOCKET_CONNECTIONS.inc()
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Handle WebSocket disconnection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            WEBSOCKET_CONNECTIONS.dec()
        
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        
        logger.info(f"WebSocket client {client_id} disconnected")
    
    async def subscribe_to_job(self, client_id: str, job_id: str):
        """Subscribe client to job updates"""
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = []
        
        if job_id not in self.subscriptions[client_id]:
            self.subscriptions[client_id].append(job_id)
    
    async def broadcast_job_update(self, job_id: str, update: Dict[str, Any]):
        """Broadcast job update to subscribed clients"""
        message = json.dumps({
            "type": "job_update",
            "job_id": job_id,
            "data": update,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        disconnected_clients = []
        
        for client_id, subscribed_jobs in self.subscriptions.items():
            if job_id in subscribed_jobs and client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_text(message)
                except Exception:
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

# Main Application Class
class LoomCtlService:
    """Main LoomCtl service orchestrating all components"""
    
    def __init__(self):
        # Core LoomOS components
        self.container_runtime = ContainerRuntime()
        self.verifier = Prism()
        self.database = LoomDB()
        self.memory_system = Tapestry()
        self.marketplace = Marketplace()
        self.agent_system = LoomAgent()
        self.model_forge = Forge()
        self.attestation_service = TEEAttestationService()
        
        # LoomCtl components
        self.scheduler = JobScheduler()
        self.websocket_manager = WebSocketManager()
        self.workers: Dict[str, Worker] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("LoomCtl service initialized")
    
    async def start(self):
        """Start the LoomCtl service"""
        logger.info("Starting LoomCtl service...")
        
        # Initialize core components
        await self.attestation_service.initialize()
        
        # Start background tasks
        self.background_tasks.append(
            asyncio.create_task(self._scheduler_loop())
        )
        self.background_tasks.append(
            asyncio.create_task(self._worker_health_monitor())
        )
        self.background_tasks.append(
            asyncio.create_task(self._metrics_collector())
        )
        
        logger.info("LoomCtl service started successfully")
    
    async def stop(self):
        """Stop the LoomCtl service"""
        logger.info("Stopping LoomCtl service...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("LoomCtl service stopped")
    
    async def _scheduler_loop(self):
        """Background scheduler loop"""
        while True:
            try:
                await self.scheduler.schedule_jobs()
                await asyncio.sleep(1.0)  # Schedule every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _worker_health_monitor(self):
        """Monitor worker health"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                stale_workers = []
                
                for worker_id, worker_info in self.scheduler.workers.items():
                    # Check if worker hasn't sent heartbeat in 30 seconds
                    if (current_time - worker_info.last_heartbeat).total_seconds() > 30:
                        stale_workers.append(worker_id)
                
                # Remove stale workers
                for worker_id in stale_workers:
                    del self.scheduler.workers[worker_id]
                    ACTIVE_WORKERS.dec()
                    logger.warning(f"Removed stale worker: {worker_id}")
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker health monitor error: {e}")
                await asyncio.sleep(10.0)
    
    async def _metrics_collector(self):
        """Collect and update metrics"""
        while True:
            try:
                # Update cluster metrics
                stats = self.scheduler.get_cluster_stats()
                ACTIVE_WORKERS.set(stats.active_workers)
                ACTIVE_JOBS.set(stats.active_jobs)
                
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(5.0)

# Global service instance
loomctl_service = LoomCtlService()

# FastAPI Application
app = FastAPI(
    title="LoomCtl API",
    description="Advanced Control Plane for LoomOS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API token"""
    # Mock token verification - in production, verify JWT tokens
    if credentials.credentials == "loom_admin_token":
        return "admin"
    elif credentials.credentials.startswith("loom_user_"):
        return credentials.credentials[10:]  # Extract user ID
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    API_REQUEST_DURATION.observe(duration)
    API_REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

# Event handlers
@app.on_event("startup")
async def startup_event():
    await loomctl_service.start()

@app.on_event("shutdown")
async def shutdown_event():
    await loomctl_service.stop()

# API Endpoints

# Health and Status
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Check core components
        db_health = await loomctl_service.database.health_check()
        memory_health = await loomctl_service.memory_system.health_check()
        marketplace_health = await loomctl_service.marketplace.health_check()
        
        # Check cluster status
        cluster_stats = loomctl_service.scheduler.get_cluster_stats()
        
        overall_status = "healthy"
        if (db_health.get("status") != "healthy" or 
            memory_health.get("status") != "healthy" or
            marketplace_health.get("status") != "healthy"):
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "database": db_health,
                "memory_system": memory_health,
                "marketplace": marketplace_health,
                "cluster": asdict(cluster_stats)
            },
            "version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Job Management
@app.post("/jobs", response_model=Dict[str, str])
async def submit_job(
    job_spec: JobSpec,
    user_id: str = Depends(verify_token)
):
    """Submit a new job for execution"""
    try:
        # Set user context
        job_spec.user_id = user_id
        
        # Submit to scheduler
        job_id = await loomctl_service.scheduler.submit_job(job_spec)
        
        # Log job submission
        await loomctl_service.database.log_job_event(
            job_id=job_id,
            event_type=EventType.JOB_CREATED,
            event_data=job_spec.dict(),
            context=AuditContext(user_id=user_id, service_name="loomctl")
        )
        
        # Broadcast to WebSocket subscribers
        await loomctl_service.websocket_manager.broadcast_job_update(
            job_id=job_id,
            update={"status": "submitted", "message": "Job submitted successfully"}
        )
        
        return {"job_id": job_id, "status": "submitted"}
        
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobStatus_Response)
async def get_job_status(
    job_id: str,
    user_id: str = Depends(verify_token)
):
    """Get status of a specific job"""
    job = loomctl_service.scheduler.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check user authorization
    if job.spec.user_id != user_id and user_id != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return JobStatus_Response(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result,
        error=job.error,
        resources_used=job.resources_used or [],
        assigned_workers=job.assigned_workers or []
    )

@app.get("/jobs", response_model=List[JobStatus_Response])
async def list_jobs(
    user_id: str = Depends(verify_token),
    status: Optional[JobStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """List jobs for the authenticated user"""
    all_jobs = []
    
    # Collect jobs from all sources
    all_jobs.extend(loomctl_service.scheduler.pending_jobs)
    all_jobs.extend(loomctl_service.scheduler.running_jobs.values())
    all_jobs.extend(loomctl_service.scheduler.completed_jobs.values())
    
    # Filter by user and status
    filtered_jobs = []
    for job in all_jobs:
        if job.spec.user_id == user_id or user_id == "admin":
            if status is None or job.status == status:
                filtered_jobs.append(job)
    
    # Sort by creation time (newest first)
    filtered_jobs.sort(key=lambda j: j.created_at, reverse=True)
    
    # Apply pagination
    paginated_jobs = filtered_jobs[offset:offset + limit]
    
    # Convert to response format
    return [
        JobStatus_Response(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            message=job.message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            result=job.result,
            error=job.error,
            resources_used=job.resources_used or [],
            assigned_workers=job.assigned_workers or []
        )
        for job in paginated_jobs
    ]

@app.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    user_id: str = Depends(verify_token)
):
    """Cancel a running job"""
    job = loomctl_service.scheduler.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check user authorization
    if job.spec.user_id != user_id and user_id != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    # Cancel the job
    await loomctl_service.scheduler.complete_job(
        job_id=job_id,
        error="Job cancelled by user"
    )
    
    # Log cancellation
    await loomctl_service.database.log_job_event(
        job_id=job_id,
        event_type=EventType.JOB_FAILED,
        event_data={"reason": "cancelled_by_user"},
        context=AuditContext(user_id=user_id, service_name="loomctl")
    )
    
    return {"message": "Job cancelled successfully"}

# Worker Management
@app.get("/workers", response_model=List[WorkerInfo])
async def list_workers(user_id: str = Depends(verify_token)):
    """List all active workers"""
    workers = []
    for worker_id, worker_info in loomctl_service.scheduler.workers.items():
        workers.append(WorkerInfo(
            worker_id=worker_id,
            status=worker_info.status,
            capabilities=worker_info.capabilities,
            current_load=worker_info.current_load,
            last_heartbeat=worker_info.last_heartbeat
        ))
    
    return workers

@app.post("/workers/{worker_id}/heartbeat")
async def worker_heartbeat(
    worker_id: str,
    worker_status: Dict[str, Any],
    user_id: str = Depends(verify_token)
):
    """Receive heartbeat from worker"""
    current_time = datetime.now(timezone.utc)
    
    # Update or create worker info
    if worker_id not in loomctl_service.scheduler.workers:
        ACTIVE_WORKERS.inc()
    
    loomctl_service.scheduler.workers[worker_id] = WorkerInfo(
        worker_id=worker_id,
        status=worker_status.get("status", "unknown"),
        capabilities=worker_status.get("capabilities", {}),
        current_load=worker_status.get("current_load", 0.0),
        last_heartbeat=current_time
    )
    
    return {"message": "Heartbeat received", "timestamp": current_time.isoformat()}

# Cluster Management
@app.get("/cluster/stats", response_model=ClusterStats)
async def get_cluster_stats(user_id: str = Depends(verify_token)):
    """Get cluster statistics"""
    return loomctl_service.scheduler.get_cluster_stats()

@app.get("/cluster/nodes")
async def get_cluster_nodes(user_id: str = Depends(verify_token)):
    """Get cluster node information"""
    return {
        "total_nodes": len(loomctl_service.scheduler.workers),
        "nodes": [
            {
                "worker_id": worker_id,
                "status": worker_info.status,
                "load": worker_info.current_load,
                "last_seen": worker_info.last_heartbeat.isoformat()
            }
            for worker_id, worker_info in loomctl_service.scheduler.workers.items()
        ]
    }

# AI and ML Specific Endpoints
@app.post("/ai/agents/execute")
async def execute_agent_task(
    task_spec: Dict[str, Any],
    user_id: str = Depends(verify_token)
):
    """Execute an AI agent task"""
    try:
        result = await loomctl_service.agent_system.execute_task(
            task_spec.get("prompt", ""),
            tools=task_spec.get("tools", []),
            context=task_spec.get("context", {})
        )
        
        return {
            "status": "completed",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/models/adapt")
async def adapt_model(
    adaptation_spec: Dict[str, Any],
    user_id: str = Depends(verify_token)
):
    """Adapt a model using the Forge system"""
    try:
        result = await loomctl_service.model_forge.adapt_model(
            model_id=adaptation_spec.get("model_id"),
            adaptation_type=adaptation_spec.get("adaptation_type", "lora"),
            training_data=adaptation_spec.get("training_data"),
            config=adaptation_spec.get("config", {})
        )
        
        return {
            "status": "completed",
            "adapted_model_id": result.get("adapted_model_id"),
            "metrics": result.get("metrics", {}),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/verify")
async def verify_content(
    verification_request: Dict[str, Any],
    user_id: str = Depends(verify_token)
):
    """Verify content using Prism verification system"""
    try:
        result = await loomctl_service.verifier.verify(
            content=verification_request.get("content"),
            verification_type=verification_request.get("type", "comprehensive"),
            context=verification_request.get("context", {})
        )
        
        return {
            "verification_result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Marketplace Integration
@app.get("/marketplace/listings")
async def get_marketplace_listings(
    listing_type: Optional[str] = None,
    user_id: str = Depends(verify_token)
):
    """Get marketplace listings"""
    try:
        # Convert string to enum if provided
        type_filter = None
        if listing_type:
            type_filter = getattr(loomctl_service.marketplace.ListingType, listing_type.upper(), None)
        
        listings = await loomctl_service.marketplace.search_listings(
            listing_type=type_filter,
            max_price=None,
            min_quality=0.0
        )
        
        return {
            "listings": [
                {
                    "listing_id": listing.listing_id,
                    "title": listing.title,
                    "description": listing.description,
                    "price": str(listing.price),
                    "quality_score": listing.quality_score,
                    "provider_id": listing.provider_id
                }
                for listing in listings
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/marketplace/purchase")
async def purchase_listing(
    purchase_request: Dict[str, Any],
    user_id: str = Depends(verify_token)
):
    """Purchase a marketplace listing"""
    try:
        transaction_id = await loomctl_service.marketplace.purchase_listing(
            buyer_id=user_id,
            listing_id=purchase_request.get("listing_id"),
            quantity=purchase_request.get("quantity", 1)
        )
        
        return {
            "transaction_id": transaction_id,
            "status": "processing",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket Endpoints
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await loomctl_service.websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe_job":
                job_id = message.get("job_id")
                if job_id:
                    await loomctl_service.websocket_manager.subscribe_to_job(client_id, job_id)
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "job_id": job_id
                    }))
            
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
    
    except WebSocketDisconnect:
        loomctl_service.websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        loomctl_service.websocket_manager.disconnect(client_id)

# Streaming Endpoints
@app.get("/jobs/{job_id}/logs")
async def stream_job_logs(
    job_id: str,
    user_id: str = Depends(verify_token)
):
    """Stream job logs in real-time"""
    job = loomctl_service.scheduler.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.spec.user_id != user_id and user_id != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    async def log_generator():
        # Mock log streaming - in production, stream from actual containers
        for i in range(100):
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                break
            
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "message": f"Processing step {i+1}/100",
                "progress": (i + 1) / 100
            }
            
            yield f"data: {json.dumps(log_entry)}\n\n"
            await asyncio.sleep(0.1)
        
        # Final log entry
        yield f"data: {json.dumps({'message': 'Job completed', 'final': True})}\n\n"
    
    return StreamingResponse(
        log_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/metrics/stream")
async def stream_metrics(user_id: str = Depends(verify_token)):
    """Stream real-time metrics"""
    async def metrics_generator():
        while True:
            cluster_stats = loomctl_service.scheduler.get_cluster_stats()
            
            metrics_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cluster_stats": asdict(cluster_stats),
                "active_jobs": len(loomctl_service.scheduler.running_jobs),
                "pending_jobs": len(loomctl_service.scheduler.pending_jobs),
                "worker_count": len(loomctl_service.scheduler.workers)
            }
            
            yield f"data: {json.dumps(metrics_data)}\n\n"
            await asyncio.sleep(5.0)  # Send metrics every 5 seconds
    
    return StreamingResponse(
        metrics_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Admin Endpoints
@app.get("/admin/system/status")
async def get_system_status(user_id: str = Depends(verify_token)):
    """Get comprehensive system status (admin only)"""
    if user_id != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return {
        "system_info": {
            "uptime_seconds": time.time() - loomctl_service.start_time if hasattr(loomctl_service, 'start_time') else 0,
            "version": "1.0.0",
            "environment": "production"
        },
        "component_status": {
            "scheduler": "healthy",
            "database": "healthy",
            "memory_system": "healthy",
            "marketplace": "healthy",
            "attestation": "healthy"
        },
        "resource_usage": {
            "active_jobs": len(loomctl_service.scheduler.running_jobs),
            "total_workers": len(loomctl_service.scheduler.workers),
            "websocket_connections": len(loomctl_service.websocket_manager.active_connections)
        }
    }

@app.post("/admin/system/maintenance")
async def trigger_maintenance(
    maintenance_request: Dict[str, Any],
    user_id: str = Depends(verify_token)
):
    """Trigger system maintenance tasks (admin only)"""
    if user_id != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    maintenance_type = maintenance_request.get("type", "cleanup")
    
    if maintenance_type == "cleanup":
        # Clean up completed jobs older than 24 hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        cleaned_jobs = []
        
        for job_id, job in list(loomctl_service.scheduler.completed_jobs.items()):
            if job.completed_at and job.completed_at < cutoff_time:
                del loomctl_service.scheduler.completed_jobs[job_id]
                cleaned_jobs.append(job_id)
        
        return {
            "maintenance_type": maintenance_type,
            "cleaned_jobs": len(cleaned_jobs),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    return {"message": "Maintenance task completed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
from sqlalchemy.orm import sessionmaker

# Import LoomOS components
from core.loom_core import LoomCore, Request, RequestPriority, ExecutionContext
from core.scheduler import WeaverScheduler
from core.loomdb import LoomDB
from .schemas import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('loomctl_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('loomctl_request_duration_seconds', 'Request duration')
ACTIVE_JOBS = Gauge('loomctl_active_jobs', 'Active jobs')
QUEUE_DEPTH = Gauge('loomctl_queue_depth', 'Job queue depth')

# Security
security = HTTPBearer()

class LoomCtlConfig:
    """Configuration for LoomCtl service"""
    def __init__(self):
        self.database_url = "postgresql+asyncpg://loomos:loomos@localhost/loomos"
        self.redis_url = "redis://localhost:6379"
        self.max_jobs_per_user = 100
        self.job_timeout_seconds = 3600
        self.enable_metrics = True
        self.enable_websockets = True
        self.cors_origins = ["*"]
        self.worker_pool_size = 10

class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.job_subscriptions: Dict[str, List[str]] = {}  # job_id -> [connection_ids]
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        if connection_id not in self.active_connections:
            self.active_connections[connection_id] = []
        self.active_connections[connection_id].append(websocket)
    
    def disconnect(self, connection_id: str, websocket: WebSocket):
        if connection_id in self.active_connections:
            self.active_connections[connection_id].remove(websocket)
            if not self.active_connections[connection_id]:
                del self.active_connections[connection_id]
    
    async def subscribe_to_job(self, connection_id: str, job_id: str):
        if job_id not in self.job_subscriptions:
            self.job_subscriptions[job_id] = []
        if connection_id not in self.job_subscriptions[job_id]:
            self.job_subscriptions[job_id].append(connection_id)
    
    async def broadcast_job_update(self, job_id: str, update: Dict[str, Any]):
        if job_id in self.job_subscriptions:
            for connection_id in self.job_subscriptions[job_id]:
                if connection_id in self.active_connections:
                    for websocket in self.active_connections[connection_id]:
                        try:
                            await websocket.send_json({
                                "type": "job_update",
                                "job_id": job_id,
                                "data": update
                            })
                        except Exception as e:
                            logger.error(f"Failed to send update to {connection_id}: {e}")

class LoomCtlService:
    """Main LoomCtl service class"""
    
    def __init__(self, config: LoomCtlConfig):
        self.config = config
        self.loom_core = LoomCore()
        self.scheduler = WeaverScheduler()
        self.loomdb = LoomDB()
        self.connection_manager = ConnectionManager()
        
        # Job tracking
        self.active_jobs: Dict[str, JobStatus] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        
        # Database and Redis connections
        self.db_engine = None
        self.redis_client = None
        self.db_session = None
    
    async def startup(self):
        """Initialize service connections and start background tasks"""
        logger.info("Starting LoomCtl service...")
        
        # Initialize database
        self.db_engine = create_async_engine(self.config.database_url)
        self.db_session = sessionmaker(
            self.db_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Initialize Redis
        self.redis_client = redis.from_url(self.config.redis_url)
        
        # Start LoomCore
        await self.loom_core.start()
        
        # Start background tasks
        for i in range(self.config.worker_pool_size):
            asyncio.create_task(self._job_worker(f"worker-{i}"))
        
        asyncio.create_task(self._metrics_updater())
        asyncio.create_task(self._job_monitor())
        
        logger.info("LoomCtl service started successfully")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down LoomCtl service...")
        
        await self.loom_core.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.db_engine:
            await self.db_engine.dispose()
        
        logger.info("LoomCtl service shut down")
    
    async def submit_job(self, job_request: JobSubmitRequest, user_id: str) -> JobSubmitResponse:
        """Submit a new job for processing"""
        # Validate user job limits
        user_jobs = len([j for j in self.active_jobs.values() if j.user_id == user_id])
        if user_jobs >= self.config.max_jobs_per_user:
            raise HTTPException(
                status_code=429, 
                detail=f"User job limit exceeded ({self.config.max_jobs_per_user})"
            )
        
        # Create job
        job_id = str(uuid.uuid4())
        
        # Parse and validate manifest
        manifest = job_request.manifest
        
        # Create execution context
        context = ExecutionContext(
            request_id=job_id,
            user_id=user_id,
            tenant_id=job_request.tenant_id or "default",
            priority=RequestPriority(job_request.priority or 2),
            timeout_seconds=manifest.get("timeout", self.config.job_timeout_seconds),
            metadata={
                "manifest": manifest,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "source_ip": job_request.source_ip or "unknown"
            }
        )
        
        # Create LoomCore request
        request = Request(
            id=job_id,
            type=manifest.get("type", "inference"),
            payload=manifest,
            context=context
        )
        
        # Schedule job
        allocation = await self.scheduler.schedule_job(job_id, manifest)
        
        # Create job status
        job_status = JobStatus(
            job_id=job_id,
            status="submitted",
            user_id=user_id,
            tenant_id=context.tenant_id,
            manifest=manifest,
            allocation=allocation,
            created_at=datetime.now(timezone.utc),
            priority=context.priority.value
        )
        
        self.active_jobs[job_id] = job_status
        
        # Queue for processing
        await self.job_queue.put(request)
        
        # Store in database
        await self._store_job_in_db(job_status)
        
        # Update metrics
        ACTIVE_JOBS.set(len(self.active_jobs))
        QUEUE_DEPTH.set(self.job_queue.qsize())
        
        logger.info(f"Job submitted: {job_id} for user {user_id}")
        
        return JobSubmitResponse(
            job_id=job_id,
            status="submitted",
            allocation=allocation,
            estimated_start_time=datetime.now(timezone.utc),
            estimated_completion_time=None
        )
    
    async def get_job_status(self, job_id: str, user_id: str) -> JobStatusResponse:
        """Get status of a specific job"""
        if job_id not in self.active_jobs:
            # Try to load from database
            job_status = await self._load_job_from_db(job_id)
            if not job_status:
                raise HTTPException(status_code=404, detail="Job not found")
        else:
            job_status = self.active_jobs[job_id]
        
        # Check authorization
        if job_status.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get processing result if available
        result = await self.loom_core.get_result(job_id)
        
        return JobStatusResponse(
            job_id=job_id,
            status=job_status.status,
            progress=job_status.progress,
            created_at=job_status.created_at,
            started_at=job_status.started_at,
            completed_at=job_status.completed_at,
            result=result.result if result else None,
            error=result.error if result else job_status.error,
            resource_usage=result.resource_usage if result else {},
            metrics=result.metrics if result else {}
        )
    
    async def cancel_job(self, job_id: str, user_id: str) -> CancelJobResponse:
        """Cancel a running or queued job"""
        if job_id not in self.active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_status = self.active_jobs[job_id]
        
        # Check authorization
        if job_status.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if job_status.status in ["completed", "failed", "cancelled"]:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")
        
        # Cancel in LoomCore
        cancelled = await self.loom_core.cancel_request(job_id)
        
        if cancelled:
            job_status.status = "cancelled"
            job_status.completed_at = datetime.now(timezone.utc)
            job_status.error = "Cancelled by user"
            
            # Update database
            await self._update_job_in_db(job_status)
            
            # Broadcast update
            await self.connection_manager.broadcast_job_update(job_id, {
                "status": "cancelled",
                "message": "Job cancelled by user"
            })
            
            logger.info(f"Job cancelled: {job_id}")
            
            return CancelJobResponse(
                job_id=job_id,
                status="cancelled",
                cancelled_at=job_status.completed_at
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel job")
    
    async def list_jobs(self, user_id: str, limit: int = 100, offset: int = 0, 
                       status_filter: Optional[str] = None) -> JobListResponse:
        """List jobs for a user"""
        # Get jobs from active memory and database
        user_jobs = []
        
        # Add active jobs
        for job in self.active_jobs.values():
            if job.user_id == user_id:
                if not status_filter or job.status == status_filter:
                    user_jobs.append(job)
        
        # Add completed jobs from database
        db_jobs = await self._load_user_jobs_from_db(user_id, limit, offset, status_filter)
        user_jobs.extend(db_jobs)
        
        # Sort by creation time
        user_jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        total = len(user_jobs)
        user_jobs = user_jobs[offset:offset + limit]
        
        job_summaries = [
            JobSummary(
                job_id=job.job_id,
                status=job.status,
                type=job.manifest.get("type", "unknown"),
                created_at=job.created_at,
                completed_at=job.completed_at,
                duration_seconds=((job.completed_at or datetime.now(timezone.utc)) - job.created_at).total_seconds()
            )
            for job in user_jobs
        ]
        
        return JobListResponse(
            jobs=job_summaries,
            total=total,
            offset=offset,
            limit=limit
        )
    
    async def stream_job_logs(self, job_id: str, user_id: str):
        """Stream job logs in real-time"""
        if job_id not in self.active_jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_status = self.active_jobs[job_id]
        if job_status.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        async def log_generator():
            # Stream logs from Redis
            log_key = f"job_logs:{job_id}"
            
            # Get existing logs
            logs = await self.redis_client.lrange(log_key, 0, -1)
            for log_entry in logs:
                yield f"data: {log_entry.decode()}\n\n"
            
            # Stream new logs
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe(f"job_logs_stream:{job_id}")
            
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        yield f"data: {message['data'].decode()}\n\n"
            finally:
                await pubsub.unsubscribe(f"job_logs_stream:{job_id}")
                await pubsub.close()
        
        return StreamingResponse(
            log_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    async def get_system_metrics(self) -> SystemMetricsResponse:
        """Get system metrics and health status"""
        loom_core_metrics = self.loom_core.get_metrics()
        scheduler_metrics = await self.scheduler.get_metrics()
        
        return SystemMetricsResponse(
            active_jobs=len(self.active_jobs),
            queue_depth=self.job_queue.qsize(),
            worker_utilization=loom_core_metrics.get("worker_utilization", 0.0),
            success_rate=loom_core_metrics.get("success_rate", 0.0),
            avg_job_duration=loom_core_metrics.get("avg_processing_time_ms", 0.0),
            resource_usage=await self._get_resource_usage(),
            scheduler_metrics=scheduler_metrics
        )
    
    async def _job_worker(self, worker_id: str):
        """Background job processing worker"""
        logger.info(f"Job worker {worker_id} started")
        
        while True:
            try:
                # Get job from queue
                request = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                
                job_id = request.id
                if job_id not in self.active_jobs:
                    continue
                
                job_status = self.active_jobs[job_id]
                
                # Update status
                job_status.status = "running"
                job_status.started_at = datetime.now(timezone.utc)
                
                await self._update_job_in_db(job_status)
                await self.connection_manager.broadcast_job_update(job_id, {
                    "status": "running",
                    "started_at": job_status.started_at.isoformat()
                })
                
                # Process job
                logger.info(f"Worker {worker_id} processing job {job_id}")
                
                # Submit to LoomCore
                await self.loom_core.submit_request(request)
                
                # Wait for completion
                result = None
                while not result:
                    await asyncio.sleep(1)
                    result = await self.loom_core.get_result(job_id)
                
                # Update job status
                if result.status.value == "completed":
                    job_status.status = "completed"
                    job_status.result = result.result
                elif result.status.value == "failed":
                    job_status.status = "failed"
                    job_status.error = result.error
                
                job_status.completed_at = datetime.now(timezone.utc)
                job_status.duration_seconds = (job_status.completed_at - job_status.started_at).total_seconds()
                
                await self._update_job_in_db(job_status)
                await self.connection_manager.broadcast_job_update(job_id, {
                    "status": job_status.status,
                    "completed_at": job_status.completed_at.isoformat(),
                    "result": job_status.result,
                    "error": job_status.error
                })
                
                # Clean up
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
                
                logger.info(f"Worker {worker_id} completed job {job_id}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
    
    async def _metrics_updater(self):
        """Update Prometheus metrics"""
        while True:
            try:
                ACTIVE_JOBS.set(len(self.active_jobs))
                QUEUE_DEPTH.set(self.job_queue.qsize())
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
    
    async def _job_monitor(self):
        """Monitor jobs for timeouts and cleanup"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                for job_id, job_status in list(self.active_jobs.items()):
                    # Check for timeout
                    if job_status.started_at:
                        runtime = (current_time - job_status.started_at).total_seconds()
                        if runtime > job_status.timeout_seconds:
                            logger.warning(f"Job {job_id} timed out after {runtime}s")
                            
                            job_status.status = "failed"
                            job_status.error = f"Job timed out after {runtime}s"
                            job_status.completed_at = current_time
                            
                            await self.loom_core.cancel_request(job_id)
                            await self._update_job_in_db(job_status)
                            
                            del self.active_jobs[job_id]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Job monitor error: {e}")
    
    async def _store_job_in_db(self, job_status: JobStatus):
        """Store job in database"""
        # Implementation would use SQLAlchemy to store job
        pass
    
    async def _update_job_in_db(self, job_status: JobStatus):
        """Update job in database"""
        # Implementation would use SQLAlchemy to update job
        pass
    
    async def _load_job_from_db(self, job_id: str) -> Optional[JobStatus]:
        """Load job from database"""
        # Implementation would use SQLAlchemy to load job
        return None
    
    async def _load_user_jobs_from_db(self, user_id: str, limit: int, offset: int, 
                                     status_filter: Optional[str]) -> List[JobStatus]:
        """Load user jobs from database"""
        # Implementation would use SQLAlchemy to load jobs
        return []
    
    async def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict()
        }

# Create service instance
config = LoomCtlConfig()
service = LoomCtlService(config)

# Create FastAPI app
app = FastAPI(
    title="LoomCtl API",
    description="Advanced Control Plane for LoomOS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Dependency injection
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract user ID from JWT token (simplified)"""
    # In production, validate JWT token and extract user ID
    return "user_123"  # Placeholder

async def get_tenant_id(user_id: str = Depends(get_current_user)) -> str:
    """Get tenant ID for user"""
    # In production, lookup tenant for user
    return "tenant_123"  # Placeholder

# Event handlers
@app.on_event("startup")
async def startup_event():
    await service.startup()

@app.on_event("shutdown")
async def shutdown_event():
    await service.shutdown()

# API Routes

@app.post("/v1/jobs", response_model=JobSubmitResponse)
async def submit_job(
    job_request: JobSubmitRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user)
):
    """Submit a new job for processing"""
    REQUEST_COUNT.labels(method="POST", endpoint="/v1/jobs").inc()
    
    with REQUEST_DURATION.time():
        return await service.submit_job(job_request, user_id)

@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get status of a specific job"""
    REQUEST_COUNT.labels(method="GET", endpoint="/v1/jobs/{job_id}").inc()
    
    with REQUEST_DURATION.time():
        return await service.get_job_status(job_id, user_id)

@app.delete("/v1/jobs/{job_id}", response_model=CancelJobResponse)
async def cancel_job(
    job_id: str,
    user_id: str = Depends(get_current_user)
):
    """Cancel a running or queued job"""
    REQUEST_COUNT.labels(method="DELETE", endpoint="/v1/jobs/{job_id}").inc()
    
    with REQUEST_DURATION.time():
        return await service.cancel_job(job_id, user_id)

@app.get("/v1/jobs", response_model=JobListResponse)
async def list_jobs(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """List jobs for the current user"""
    REQUEST_COUNT.labels(method="GET", endpoint="/v1/jobs").inc()
    
    with REQUEST_DURATION.time():
        return await service.list_jobs(user_id, limit, offset, status)

@app.get("/v1/jobs/{job_id}/logs")
async def stream_job_logs(
    job_id: str,
    user_id: str = Depends(get_current_user)
):
    """Stream job logs in real-time"""
    REQUEST_COUNT.labels(method="GET", endpoint="/v1/jobs/{job_id}/logs").inc()
    
    return await service.stream_job_logs(job_id, user_id)

@app.get("/v1/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    user_id: str = Depends(get_current_user)
):
    """Get system metrics and health status"""
    REQUEST_COUNT.labels(method="GET", endpoint="/v1/system/metrics").inc()
    
    with REQUEST_DURATION.time():
        return await service.get_system_metrics()

@app.get("/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

# WebSocket endpoint
@app.websocket("/v1/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """WebSocket endpoint for real-time updates"""
    await service.connection_manager.connect(websocket, connection_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe_job":
                job_id = data.get("job_id")
                if job_id:
                    await service.connection_manager.subscribe_to_job(connection_id, job_id)
                    await websocket.send_json({
                        "type": "subscribed",
                        "job_id": job_id
                    })
    
    except WebSocketDisconnect:
        service.connection_manager.disconnect(connection_id, websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)