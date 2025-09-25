"""
LoomOS Nexus Master Node - Centralized Training Coordinator

The Master Node provides centralized coordination for distributed training while
preserving Nexus's revolutionary low-communication architecture:

Key Responsibilities:
- Worker discovery, registration, and health monitoring
- Training job orchestration and task distribution
- Global state synchronization and checkpoint coordination
- Fault tolerance and dynamic worker scaling
- Performance optimization and load balancing
- Security policy enforcement and attestation management

Architecture Benefits:
- Single point of control for complex distributed jobs
- Centralized monitoring and debugging capabilities
- Simplified worker coordination protocols
- Enhanced fault tolerance and recovery
- Better resource utilization and scheduling
- Unified security and compliance management

The Master Node acts as the "brain" while workers remain the "muscle",
maintaining Nexus's 3-4 orders of magnitude communication reduction.
"""

import asyncio
import json
import logging
import time
import uuid
import sys
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import aiohttp
from aiohttp import web, WSMsgType
import socket
import psutil
import hashlib
from pathlib import Path
import ssl
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# LoomOS imports
from ..loomnode.worker import WorkerConfig, WorkerStatus, TaskType, TrainingTask, CompressionMethod
from ..loomnode.attest import TEEAttestationService
from ..failover.election import FailoverCoordinator, NodeRole, NodeInfo

logger = logging.getLogger(__name__)

# Master Node Metrics
MASTER_CONNECTED_WORKERS = Gauge('nexus_master_connected_workers', 'Number of connected workers')
MASTER_ACTIVE_JOBS = Gauge('nexus_master_active_jobs', 'Number of active training jobs')
MASTER_TASKS_SCHEDULED = Counter('nexus_master_tasks_scheduled', 'Total tasks scheduled')
MASTER_WORKER_FAILURES = Counter('nexus_master_worker_failures', 'Worker failure events')
MASTER_JOB_COMPLETION_TIME = Histogram('nexus_master_job_completion_seconds', 'Job completion time')

class MasterStatus(Enum):
    """Master node status states"""
    INITIALIZING = "initializing"
    READY = "ready"
    COORDINATING = "coordinating"
    SCALING = "scaling"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"

class JobStatus(Enum):
    """Training job status states"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkerInfo:
    """Information about a connected worker"""
    worker_id: str
    worker_rank: int
    status: WorkerStatus
    capabilities: Dict[str, Any]
    last_heartbeat: datetime
    active_tasks: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    network_latency_ms: float = 0.0
    attestation_status: str = "unknown"
    connection: Optional[web.WebSocketResponse] = None

@dataclass
class TrainingJob:
    """Distributed training job configuration"""
    job_id: str
    name: str
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    distributed_config: Dict[str, Any]
    
    # Job lifecycle
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Resource allocation
    required_workers: int = 1
    assigned_workers: List[str] = field(default_factory=list)
    
    # Progress tracking
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Performance metrics
    throughput_samples_per_sec: float = 0.0
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Error handling
    max_retries: int = 3
    retry_count: int = 0
    error_message: Optional[str] = None

@dataclass
class MasterConfig:
    """Master node configuration"""
    master_id: str
    bind_address: str = "0.0.0.0"
    bind_port: int = 29500
    
    # Worker management
    worker_timeout_seconds: float = 30.0
    heartbeat_interval_seconds: float = 10.0
    max_workers: int = 1000
    
    # Job scheduling
    max_concurrent_jobs: int = 10
    task_allocation_strategy: str = "round_robin"  # round_robin, load_balanced, locality_aware
    
    # Performance optimization
    enable_adaptive_scaling: bool = True
    enable_load_balancing: bool = True
    enable_fault_tolerance: bool = True
    
    # Security
    enable_tls: bool = True
    enable_worker_attestation: bool = True
    attestation_timeout_seconds: float = 60.0
    
    # Monitoring
    metrics_port: int = 8090
    enable_metrics: bool = True
    log_level: str = "INFO"

class MasterNode:
    """Nexus Master Node - Centralized Training Coordinator"""
    
    def __init__(self, config: Optional[MasterConfig] = None):
        self.config = config or MasterConfig(master_id=f"master_{uuid.uuid4().hex[:8]}")
        self.status = MasterStatus.INITIALIZING
        
        # Worker management
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_connections: Dict[str, web.WebSocketResponse] = {}
        
        # Job management
        self.jobs: Dict[str, TrainingJob] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.task_scheduler: Optional[asyncio.Task] = None
        
        # Security and attestation
        self.attestation_service = TEEAttestationService() if self.config.enable_worker_attestation else None
        
        # Performance monitoring
        self.performance_tracker = PerformanceTracker()
        self.fault_detector = FaultDetector()
        
        # Master failover and election
        self.failover_coordinator: Optional[FailoverCoordinator] = None
        self.is_primary_master = True  # Assumes this starts as primary
        self.backup_masters: List[str] = []
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Web application
        self.app = web.Application()
        self.setup_routes()
        
        logger.info(f"Master node {self.config.master_id} initialized")
    
    def setup_routes(self):
        """Setup HTTP and WebSocket routes"""
        # Worker management endpoints
        self.app.router.add_post('/api/v1/workers/register', self.register_worker)
        self.app.router.add_get('/api/v1/workers', self.list_workers)
        self.app.router.add_get('/api/v1/workers/{worker_id}', self.get_worker)
        self.app.router.add_delete('/api/v1/workers/{worker_id}', self.remove_worker)
        
        # Job management endpoints
        self.app.router.add_post('/api/v1/jobs', self.create_job)
        self.app.router.add_get('/api/v1/jobs', self.list_jobs)
        self.app.router.add_get('/api/v1/jobs/{job_id}', self.get_job)
        self.app.router.add_post('/api/v1/jobs/{job_id}/start', self.start_job)
        self.app.router.add_post('/api/v1/jobs/{job_id}/pause', self.pause_job)
        self.app.router.add_post('/api/v1/jobs/{job_id}/cancel', self.cancel_job)
        
        # Monitoring endpoints
        self.app.router.add_get('/api/v1/status', self.get_master_status)
        self.app.router.add_get('/api/v1/metrics', self.get_metrics)
        
        # Failover and election endpoints
        self.app.router.add_post('/api/v1/failover/initiate', self.initiate_failover)
        self.app.router.add_post('/api/v1/failover/promote', self.promote_to_master)
        self.app.router.add_post('/api/v1/failover/demote', self.demote_from_master)
        self.app.router.add_get('/api/v1/failover/status', self.get_failover_status)
        
        # Raft consensus endpoints for election
        self.app.router.add_post('/raft/vote', self.handle_vote_request)
        self.app.router.add_post('/raft/heartbeat', self.handle_heartbeat)
        self.app.router.add_post('/raft/master_announcement', self.handle_master_announcement)
        
        # WebSocket endpoints
        self.app.router.add_get('/ws/worker/{worker_id}', self.worker_websocket)
        self.app.router.add_get('/ws/client', self.client_websocket)
        
        # Health check
        self.app.router.add_get('/health', self.health_check)
    
    async def run(self):
        """Run the master node"""
        logger.info(f"Starting Nexus Master Node {self.config.master_id}")
        
        try:
            # Initialize failover coordination if configured
            await self._initialize_failover_system()
            
            # Start metrics server
            if self.config.enable_metrics:
                start_http_server(self.config.metrics_port)
                logger.info(f"Metrics server started on port {self.config.metrics_port}")
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Set status to ready
            self.status = MasterStatus.READY
            logger.info(f"Master node ready on {self.config.bind_address}:{self.config.bind_port}")
            
            # Start web server
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(
                runner, 
                self.config.bind_address, 
                self.config.bind_port,
                ssl_context=self._create_ssl_context() if self.config.enable_tls else None
            )
            await site.start()
            
            # Keep running until shutdown
            while self.status != MasterStatus.SHUTDOWN:
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Master node failed: {e}")
            self.status = MasterStatus.SHUTDOWN
        finally:
            await self._cleanup()
    
    async def _start_background_tasks(self):
        """Start background monitoring and coordination tasks"""
        # Worker health monitoring
        health_monitor = asyncio.create_task(self._worker_health_monitor())
        self.background_tasks.append(health_monitor)
        
        # Job scheduler
        job_scheduler = asyncio.create_task(self._job_scheduler())
        self.background_tasks.append(job_scheduler)
        
        # Performance optimizer
        perf_optimizer = asyncio.create_task(self._performance_optimizer())
        self.background_tasks.append(perf_optimizer)
        
        # Fault detector
        fault_detector = asyncio.create_task(self._fault_detector())
        self.background_tasks.append(fault_detector)
        
        logger.info("Background tasks started")
    
    async def _worker_health_monitor(self):
        """Monitor worker health and connectivity"""
        while self.status != MasterStatus.SHUTDOWN:
            try:
                current_time = datetime.now(timezone.utc)
                timeout_threshold = timedelta(seconds=self.config.worker_timeout_seconds)
                
                # Check for timed-out workers
                timed_out_workers = []
                for worker_id, worker_info in self.workers.items():
                    if current_time - worker_info.last_heartbeat > timeout_threshold:
                        timed_out_workers.append(worker_id)
                
                # Remove timed-out workers
                for worker_id in timed_out_workers:
                    await self._handle_worker_failure(worker_id, "timeout")
                
                # Update metrics
                MASTER_CONNECTED_WORKERS.set(len(self.workers))
                
                await asyncio.sleep(self.config.heartbeat_interval_seconds)
                
            except Exception as e:
                logger.error(f"Worker health monitor error: {e}")
                await asyncio.sleep(5.0)
    
    async def _job_scheduler(self):
        """Schedule and coordinate training jobs"""
        while self.status != MasterStatus.SHUTDOWN:
            try:
                # Process job queue
                if not self.job_queue.empty():
                    job = await self.job_queue.get()
                    await self._schedule_job(job)
                
                # Check running jobs
                for job_id, job in list(self.jobs.items()):
                    if job.status == JobStatus.RUNNING:
                        await self._monitor_job_progress(job)
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Job scheduler error: {e}")
                await asyncio.sleep(5.0)
    
    async def _schedule_job(self, job: TrainingJob):
        """Schedule a training job across available workers"""
        logger.info(f"Scheduling job {job.job_id}")
        
        # Find available workers
        available_workers = [
            worker_id for worker_id, worker_info in self.workers.items()
            if worker_info.status == WorkerStatus.IDLE
        ]
        
        if len(available_workers) < job.required_workers:
            logger.warning(f"Insufficient workers for job {job.job_id}: need {job.required_workers}, have {len(available_workers)}")
            await asyncio.sleep(5.0)  # Retry later
            await self.job_queue.put(job)
            return
        
        # Select workers based on strategy
        selected_workers = await self._select_workers(available_workers, job)
        
        # Assign workers to job
        job.assigned_workers = selected_workers[:job.required_workers]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        
        # Create and distribute training tasks
        tasks = await self._create_training_tasks(job)
        for i, task in enumerate(tasks):
            worker_id = job.assigned_workers[i % len(job.assigned_workers)]
            await self._send_task_to_worker(worker_id, task)
        
        job.total_tasks = len(tasks)
        MASTER_TASKS_SCHEDULED.inc(len(tasks))
        MASTER_ACTIVE_JOBS.inc()
        
        logger.info(f"Job {job.job_id} scheduled to workers: {job.assigned_workers}")
    
    async def _select_workers(self, available_workers: List[str], job: TrainingJob) -> List[str]:
        """Select optimal workers for a job based on strategy"""
        if self.config.task_allocation_strategy == "round_robin":
            return available_workers
        elif self.config.task_allocation_strategy == "load_balanced":
            # Sort by current load (ascending)
            workers_by_load = sorted(
                available_workers,
                key=lambda w: len(self.workers[w].active_tasks)
            )
            return workers_by_load
        elif self.config.task_allocation_strategy == "locality_aware":
            # Consider network latency and geographical proximity
            workers_by_latency = sorted(
                available_workers,
                key=lambda w: self.workers[w].network_latency_ms
            )
            return workers_by_latency
        else:
            return available_workers
    
    async def _create_training_tasks(self, job: TrainingJob) -> List[TrainingTask]:
        """Create training tasks for a job"""
        tasks = []
        
        # Example: Create tasks for different phases of training
        task_types = [
            TaskType.FORWARD_PASS,
            TaskType.BACKWARD_PASS,
            TaskType.GRADIENT_EXCHANGE,
            TaskType.PARAMETER_UPDATE
        ]
        
        for i, task_type in enumerate(task_types):
            task = TrainingTask(
                task_id=f"{job.job_id}_task_{i}",
                task_type=task_type,
                model_config=job.model_config,
                dataset_path=job.dataset_config.get("path", ""),
                batch_indices=list(range(i * 32, (i + 1) * 32)),  # Example batch allocation
                learning_rate=job.training_config.get("learning_rate", 0.001),
                gradient_compression=job.distributed_config.get("compression"),
                priority=1
            )
            tasks.append(task)
        
        return tasks
    
    async def _send_task_to_worker(self, worker_id: str, task: TrainingTask):
        """Send a training task to a specific worker"""
        if worker_id not in self.worker_connections:
            logger.error(f"No connection to worker {worker_id}")
            return
        
        connection = self.worker_connections[worker_id]
        
        message = {
            "type": "task_assignment",
            "task": {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "model_config": task.model_config,
                "dataset_path": task.dataset_path,
                "batch_indices": task.batch_indices,
                "learning_rate": task.learning_rate,
                "gradient_compression": task.gradient_compression,
                "priority": task.priority
            }
        }
        
        try:
            await connection.send_str(json.dumps(message))
            
            # Track task assignment
            if worker_id in self.workers:
                self.workers[worker_id].active_tasks.append(task.task_id)
            
            logger.debug(f"Task {task.task_id} sent to worker {worker_id}")
            
        except Exception as e:
            logger.error(f"Failed to send task to worker {worker_id}: {e}")
    
    async def _monitor_job_progress(self, job: TrainingJob):
        """Monitor progress of a running job"""
        # Check if all tasks are completed
        total_completed = sum(
            1 for worker_info in self.workers.values()
            for task_id in worker_info.active_tasks
            if task_id.startswith(job.job_id) and task_id not in worker_info.active_tasks
        )
        
        if total_completed >= job.total_tasks:
            # Job completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.completed_tasks = total_completed
            
            # Calculate job duration
            if job.started_at:
                duration = (job.completed_at - job.started_at).total_seconds()
                MASTER_JOB_COMPLETION_TIME.observe(duration)
            
            # Release workers
            for worker_id in job.assigned_workers:
                if worker_id in self.workers:
                    # Clear job-related tasks
                    self.workers[worker_id].active_tasks = [
                        task_id for task_id in self.workers[worker_id].active_tasks
                        if not task_id.startswith(job.job_id)
                    ]
                    self.workers[worker_id].status = WorkerStatus.IDLE
            
            MASTER_ACTIVE_JOBS.dec()
            logger.info(f"Job {job.job_id} completed in {duration:.2f} seconds")
    
    async def _handle_worker_failure(self, worker_id: str, reason: str):
        """Handle worker failure and implement recovery"""
        logger.warning(f"Worker {worker_id} failed: {reason}")
        
        if worker_id in self.workers:
            worker_info = self.workers[worker_id]
            
            # Reassign active tasks to other workers
            if worker_info.active_tasks:
                await self._reassign_tasks(worker_info.active_tasks)
            
            # Remove worker
            del self.workers[worker_id]
            
            # Close connection if exists
            if worker_id in self.worker_connections:
                try:
                    await self.worker_connections[worker_id].close()
                except:
                    pass
                del self.worker_connections[worker_id]
            
            MASTER_WORKER_FAILURES.inc()
            MASTER_CONNECTED_WORKERS.dec()
    
    async def _reassign_tasks(self, failed_tasks: List[str]):
        """Reassign tasks from failed worker to available workers"""
        available_workers = [
            worker_id for worker_id, worker_info in self.workers.items()
            if worker_info.status == WorkerStatus.IDLE
        ]
        
        if not available_workers:
            logger.error("No available workers for task reassignment")
            return
        
        # Simple round-robin reassignment
        for i, task_id in enumerate(failed_tasks):
            target_worker = available_workers[i % len(available_workers)]
            # In production, recreate and send the task
            logger.info(f"Reassigning task {task_id} to worker {target_worker}")
    
    async def _performance_optimizer(self):
        """Optimize cluster performance based on metrics"""
        while self.status != MasterStatus.SHUTDOWN:
            try:
                if self.config.enable_adaptive_scaling:
                    await self._adaptive_scaling()
                
                if self.config.enable_load_balancing:
                    await self._load_balancing()
                
                await asyncio.sleep(30.0)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance optimizer error: {e}")
                await asyncio.sleep(30.0)
    
    async def _adaptive_scaling(self):
        """Implement adaptive worker scaling"""
        # Simple scaling logic based on job queue and worker utilization
        pending_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.PENDING)
        avg_worker_utilization = sum(len(w.active_tasks) for w in self.workers.values()) / max(len(self.workers), 1)
        
        if pending_jobs > 0 and avg_worker_utilization > 3:
            logger.info("High load detected - consider scaling up workers")
        elif pending_jobs == 0 and avg_worker_utilization < 1:
            logger.info("Low load detected - consider scaling down workers")
    
    async def _load_balancing(self):
        """Implement dynamic load balancing"""
        # Identify overloaded and underloaded workers
        overloaded_workers = [
            worker_id for worker_id, worker_info in self.workers.items()
            if len(worker_info.active_tasks) > 5
        ]
        
        underloaded_workers = [
            worker_id for worker_id, worker_info in self.workers.items()
            if len(worker_info.active_tasks) < 2 and worker_info.status == WorkerStatus.IDLE
        ]
        
        if overloaded_workers and underloaded_workers:
            logger.info(f"Load balancing opportunity: {len(overloaded_workers)} overloaded, {len(underloaded_workers)} underloaded")
    
    async def _fault_detector(self):
        """Detect and respond to system faults"""
        while self.status != MasterStatus.SHUTDOWN:
            try:
                # Monitor system health metrics
                for worker_id, worker_info in self.workers.items():
                    # Check for performance degradation
                    if worker_info.performance_metrics.get('task_completion_rate', 1.0) < 0.1:
                        logger.warning(f"Performance degradation detected in worker {worker_id}")
                    
                    # Check for high error rates
                    error_rate = worker_info.performance_metrics.get('error_rate', 0.0)
                    if error_rate > 0.1:  # 10% error rate threshold
                        logger.warning(f"High error rate in worker {worker_id}: {error_rate}")
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except Exception as e:
                logger.error(f"Fault detector error: {e}")
                await asyncio.sleep(60.0)
    
    # HTTP API Handlers
    
    async def register_worker(self, request: web.Request) -> web.Response:
        """Register a new worker"""
        try:
            data = await request.json()
            worker_id = data.get('worker_id')
            
            if not worker_id:
                return web.json_response({'error': 'worker_id required'}, status=400)
            
            if worker_id in self.workers:
                return web.json_response({'error': 'Worker already registered'}, status=409)
            
            # Create worker info
            worker_info = WorkerInfo(
                worker_id=worker_id,
                worker_rank=data.get('worker_rank', 0),
                status=WorkerStatus.IDLE,
                capabilities=data.get('capabilities', {}),
                last_heartbeat=datetime.now(timezone.utc)
            )
            
            # Perform attestation if enabled
            if self.attestation_service:
                attestation_result = await self._attest_worker(data.get('attestation_data'))
                worker_info.attestation_status = "verified" if attestation_result else "failed"
                
                if not attestation_result:
                    return web.json_response({'error': 'Worker attestation failed'}, status=403)
            
            self.workers[worker_id] = worker_info
            MASTER_CONNECTED_WORKERS.inc()
            
            logger.info(f"Worker {worker_id} registered successfully")
            
            return web.json_response({
                'status': 'registered',
                'worker_id': worker_id,
                'master_id': self.config.master_id
            })
            
        except Exception as e:
            logger.error(f"Worker registration error: {e}")
            return web.json_response({'error': 'Registration failed'}, status=500)
    
    async def worker_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle worker WebSocket connections"""
        worker_id = request.match_info['worker_id']
        
        if worker_id not in self.workers:
            return web.json_response({'error': 'Worker not registered'}, status=404)
        
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Store connection
        self.worker_connections[worker_id] = ws
        logger.info(f"Worker {worker_id} connected via WebSocket")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_worker_message(worker_id, json.loads(msg.data))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error from worker {worker_id}: {ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"Worker WebSocket error: {e}")
        finally:
            # Clean up connection
            if worker_id in self.worker_connections:
                del self.worker_connections[worker_id]
            logger.info(f"Worker {worker_id} disconnected")
        
        return ws
    
    async def _handle_worker_message(self, worker_id: str, message: Dict[str, Any]):
        """Handle messages from workers"""
        msg_type = message.get('type')
        
        if msg_type == 'heartbeat':
            await self._handle_worker_heartbeat(worker_id, message)
        elif msg_type == 'task_completed':
            await self._handle_task_completion(worker_id, message)
        elif msg_type == 'task_failed':
            await self._handle_task_failure(worker_id, message)
        elif msg_type == 'performance_update':
            await self._handle_performance_update(worker_id, message)
        else:
            logger.warning(f"Unknown message type from worker {worker_id}: {msg_type}")
    
    async def _handle_worker_heartbeat(self, worker_id: str, message: Dict[str, Any]):
        """Handle worker heartbeat"""
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = datetime.now(timezone.utc)
            self.workers[worker_id].status = WorkerStatus(message.get('status', 'idle'))
    
    async def _handle_task_completion(self, worker_id: str, message: Dict[str, Any]):
        """Handle task completion notification"""
        task_id = message.get('task_id')
        
        if worker_id in self.workers and task_id in self.workers[worker_id].active_tasks:
            self.workers[worker_id].active_tasks.remove(task_id)
            logger.info(f"Task {task_id} completed by worker {worker_id}")
    
    async def _handle_task_failure(self, worker_id: str, message: Dict[str, Any]):
        """Handle task failure notification"""
        task_id = message.get('task_id')
        error = message.get('error', 'Unknown error')
        
        if worker_id in self.workers and task_id in self.workers[worker_id].active_tasks:
            self.workers[worker_id].active_tasks.remove(task_id)
            logger.error(f"Task {task_id} failed on worker {worker_id}: {error}")
            
            # Implement retry logic
            await self._retry_failed_task(task_id)
    
    async def _handle_performance_update(self, worker_id: str, message: Dict[str, Any]):
        """Handle performance metrics update"""
        if worker_id in self.workers:
            self.workers[worker_id].performance_metrics.update(message.get('metrics', {}))
    
    async def _retry_failed_task(self, task_id: str):
        """Retry a failed task on another worker"""
        # Find an available worker
        available_workers = [
            worker_id for worker_id, worker_info in self.workers.items()
            if worker_info.status == WorkerStatus.IDLE and len(worker_info.active_tasks) < 3
        ]
        
        if available_workers:
            # Retry on first available worker
            target_worker = available_workers[0]
            logger.info(f"Retrying task {task_id} on worker {target_worker}")
            # In production, recreate and send the task
    
    async def create_job(self, request: web.Request) -> web.Response:
        """Create a new training job"""
        try:
            data = await request.json()
            
            job = TrainingJob(
                job_id=f"job_{uuid.uuid4().hex[:8]}",
                name=data.get('name', 'Unnamed Job'),
                model_config=data.get('model_config', {}),
                dataset_config=data.get('dataset_config', {}),
                training_config=data.get('training_config', {}),
                distributed_config=data.get('distributed_config', {}),
                required_workers=data.get('required_workers', 1)
            )
            
            self.jobs[job.job_id] = job
            await self.job_queue.put(job)
            
            logger.info(f"Job {job.job_id} created: {job.name}")
            
            return web.json_response({
                'job_id': job.job_id,
                'status': job.status.value,
                'created_at': job.created_at.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Job creation error: {e}")
            return web.json_response({'error': 'Job creation failed'}, status=500)
    
    async def get_master_status(self, request: web.Request) -> web.Response:
        """Get master node status"""
        status_info = {
            'master_id': self.config.master_id,
            'status': self.status.value,
            'workers': {
                'total': len(self.workers),
                'by_status': {}
            },
            'jobs': {
                'total': len(self.jobs),
                'by_status': {}
            },
            'performance': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'uptime_seconds': time.time() - getattr(self, 'start_time', time.time())
            }
        }
        
        # Worker status breakdown
        for worker_info in self.workers.values():
            status = worker_info.status.value
            status_info['workers']['by_status'][status] = status_info['workers']['by_status'].get(status, 0) + 1
        
        # Job status breakdown
        for job in self.jobs.values():
            status = job.status.value
            status_info['jobs']['by_status'][status] = status_info['jobs']['by_status'].get(status, 0) + 1
        
        return web.json_response(status_info)
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'master_id': self.config.master_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def _attest_worker(self, attestation_data: Optional[Dict[str, Any]]) -> bool:
        """Perform worker attestation"""
        if not self.attestation_service or not attestation_data:
            return True  # Skip attestation if not configured
        
        try:
            # Perform TEE attestation
            result = await self.attestation_service.verify_attestation(
                attestation_data.get('quote', ''),
                attestation_data.get('measurement', ''),
                attestation_data.get('signature', '')
            )
            return result.get('valid', False)
        except Exception as e:
            logger.error(f"Worker attestation failed: {e}")
            return False
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure connections"""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        # In production, load proper certificates
        context.load_cert_chain('master.crt', 'master.key')
        return context
    
    async def _cleanup(self):
        """Cleanup master node resources"""
        logger.info(f"Cleaning up master node {self.config.master_id}")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close worker connections
        for connection in self.worker_connections.values():
            try:
                await connection.close()
            except:
                pass
        
        self.status = MasterStatus.SHUTDOWN
        logger.info("Master node shutdown complete")
    
    async def _initialize_failover_system(self):
        """Initialize master failover and election system"""
        # Check if failover is configured
        cluster_config = getattr(self.config, 'cluster_config', None)
        if not cluster_config:
            logger.info("No cluster configuration - running as standalone master")
            return
        
        # Initialize failover coordinator
        self.failover_coordinator = FailoverCoordinator(
            self.config.master_id, 
            cluster_config
        )
        
        # Start coordination
        await self.failover_coordinator.start_coordination()
        
        logger.info("Failover system initialized")
    
    # Failover API endpoints
    
    async def initiate_failover(self, request: web.Request) -> web.Response:
        """Initiate planned failover to another master"""
        try:
            data = await request.json()
            target_master_id = data.get('target_master_id')
            reason = data.get('reason', 'planned maintenance')
            
            if not target_master_id:
                return web.json_response({'error': 'target_master_id required'}, status=400)
            
            if not self.failover_coordinator:
                return web.json_response({'error': 'Failover not configured'}, status=400)
            
            # Initiate graceful failover
            await self.failover_coordinator.initiate_graceful_failover(target_master_id, reason)
            
            return web.json_response({
                'status': 'failover_initiated',
                'target_master': target_master_id,
                'reason': reason,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failover initiation error: {e}")
            return web.json_response({'error': 'Failover initiation failed'}, status=500)
    
    async def promote_to_master(self, request: web.Request) -> web.Response:
        """Promote this node to master (emergency failover)"""
        try:
            if self.status == MasterStatus.READY and self.is_primary_master:
                return web.json_response({'error': 'Already primary master'}, status=400)
            
            if not self.failover_coordinator:
                return web.json_response({'error': 'Failover not configured'}, status=400)
            
            # Force promotion to master
            await self.failover_coordinator._handle_master_promotion()
            
            self.is_primary_master = True
            self.status = MasterStatus.READY
            
            return web.json_response({
                'status': 'promoted_to_master',
                'master_id': self.config.master_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Master promotion error: {e}")
            return web.json_response({'error': 'Master promotion failed'}, status=500)
    
    async def demote_from_master(self, request: web.Request) -> web.Response:
        """Demote this node from master role"""
        try:
            data = await request.json()
            reason = data.get('reason', 'manual demotion')
            
            if not self.is_primary_master:
                return web.json_response({'error': 'Not primary master'}, status=400)
            
            if not self.failover_coordinator:
                return web.json_response({'error': 'Failover not configured'}, status=400)
            
            # Demote from master
            await self.failover_coordinator.failover_manager.demote_from_master(reason)
            
            self.is_primary_master = False
            self.status = MasterStatus.MAINTENANCE
            
            return web.json_response({
                'status': 'demoted_from_master',
                'reason': reason,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Master demotion error: {e}")
            return web.json_response({'error': 'Master demotion failed'}, status=500)
    
    async def get_failover_status(self, request: web.Request) -> web.Response:
        """Get failover system status"""
        try:
            if not self.failover_coordinator:
                return web.json_response({
                    'failover_enabled': False,
                    'reason': 'Failover not configured'
                })
            
            election_state = self.failover_coordinator.election
            
            status_info = {
                'failover_enabled': True,
                'is_primary_master': self.is_primary_master,
                'master_id': self.config.master_id,
                'current_master': election_state.current_master,
                'election_state': election_state.state.value,
                'current_term': election_state.current_term,
                'cluster_nodes': len(election_state.cluster_nodes),
                'backup_masters': self.backup_masters,
                'is_coordinating_failover': self.failover_coordinator.is_coordinating_failover,
                'last_election': getattr(election_state, 'last_election_time', None)
            }
            
            return web.json_response(status_info)
            
        except Exception as e:
            logger.error(f"Failover status error: {e}")
            return web.json_response({'error': 'Failed to get failover status'}, status=500)
    
    # Raft consensus handlers
    
    async def handle_vote_request(self, request: web.Request) -> web.Response:
        """Handle Raft vote request"""
        try:
            if not self.failover_coordinator:
                return web.json_response({'error': 'Election not configured'}, status=400)
            
            vote_request = await request.json()
            response = await self.failover_coordinator.election.handle_vote_request(vote_request)
            
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Vote request error: {e}")
            return web.json_response({'error': 'Vote request failed'}, status=500)
    
    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle Raft heartbeat"""
        try:
            if not self.failover_coordinator:
                return web.json_response({'error': 'Election not configured'}, status=400)
            
            heartbeat_request = await request.json()
            response = await self.failover_coordinator.election.handle_heartbeat(heartbeat_request)
            
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return web.json_response({'error': 'Heartbeat failed'}, status=500)
    
    async def handle_master_announcement(self, request: web.Request) -> web.Response:
        """Handle master announcement"""
        try:
            if not self.failover_coordinator:
                return web.json_response({'error': 'Election not configured'}, status=400)
            
            announcement = await request.json()
            response = await self.failover_coordinator.election.handle_master_announcement(announcement)
            
            # Update our master status based on announcement
            if response.get('status') == 'acknowledged':
                announced_master = announcement.get('master_id')
                if announced_master != self.config.master_id:
                    self.is_primary_master = False
                    logger.info(f"Acknowledged new master: {announced_master}")
            
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Master announcement error: {e}")
            return web.json_response({'error': 'Master announcement failed'}, status=500)

class PerformanceTracker:
    """Track and analyze cluster performance"""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append((datetime.now(timezone.utc), value))
        
        # Keep only last 1000 measurements
        if len(self.metrics_history[metric_name]) > 1000:
            self.metrics_history[metric_name] = self.metrics_history[metric_name][-1000:]
    
    def get_trend(self, metric_name: str, window_minutes: int = 10) -> str:
        """Get trend for a metric over time window"""
        if metric_name not in self.metrics_history:
            return "unknown"
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent_values = [
            value for timestamp, value in self.metrics_history[metric_name]
            if timestamp > cutoff_time
        ]
        
        if len(recent_values) < 2:
            return "insufficient_data"
        
        if recent_values[-1] > recent_values[0] * 1.1:
            return "increasing"
        elif recent_values[-1] < recent_values[0] * 0.9:
            return "decreasing"
        else:
            return "stable"

class FaultDetector:
    """Detect and classify system faults"""
    
    def __init__(self):
        self.fault_patterns: Dict[str, int] = {}
    
    def detect_fault(self, metrics: Dict[str, float]) -> Optional[str]:
        """Detect potential faults from metrics"""
        # Simple fault detection rules
        if metrics.get('cpu_percent', 0) > 95:
            return "high_cpu_usage"
        
        if metrics.get('memory_percent', 0) > 90:
            return "high_memory_usage"
        
        if metrics.get('error_rate', 0) > 0.1:
            return "high_error_rate"
        
        if metrics.get('network_latency_ms', 0) > 1000:
            return "high_network_latency"
        
        return None
    
    def classify_fault(self, fault_type: str) -> str:
        """Classify fault severity"""
        severity_map = {
            'high_cpu_usage': 'warning',
            'high_memory_usage': 'critical',
            'high_error_rate': 'critical',
            'high_network_latency': 'warning'
        }
        
        return severity_map.get(fault_type, 'unknown')

# CLI entry point for master node
async def main():
    """Main entry point for master node"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nexus Master Node")
    parser.add_argument('--master-id', type=str, help='Master node identifier')
    parser.add_argument('--bind-address', type=str, default='0.0.0.0', help='Bind address')
    parser.add_argument('--bind-port', type=int, default=29500, help='Bind port')
    parser.add_argument('--max-workers', type=int, default=1000, help='Maximum workers')
    parser.add_argument('--enable-tls', action='store_true', help='Enable TLS')
    parser.add_argument('--enable-attestation', action='store_true', help='Enable worker attestation')
    parser.add_argument('--metrics-port', type=int, default=8090, help='Metrics port')
    parser.add_argument('--log-level', type=str, default='INFO', help='Log level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run master node
    config = MasterConfig(
        master_id=args.master_id or f"master_{uuid.uuid4().hex[:8]}",
        bind_address=args.bind_address,
        bind_port=args.bind_port,
        max_workers=args.max_workers,
        enable_tls=args.enable_tls,
        enable_worker_attestation=args.enable_attestation,
        metrics_port=args.metrics_port,
        log_level=args.log_level
    )
    
    master = MasterNode(config)
    master.start_time = time.time()
    
    try:
        await master.run()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Master node failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())