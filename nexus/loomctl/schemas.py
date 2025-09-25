"""
Advanced Pydantic Schemas for LoomCtl API

Comprehensive data models with validation, serialization, and documentation
for all API endpoints and internal data structures.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import re

# Base Models

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class PaginationMixin(BaseModel):
    """Mixin for pagination"""
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of items to return")
    offset: int = Field(0, ge=0, description="Number of items to skip")

# Enums

class JobStatus(str, Enum):
    """Job status enumeration"""
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class JobPriority(int, Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ResourceType(str, Enum):
    """Resource types"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

class NodeStatus(str, Enum):
    """Node status enumeration"""
    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

# Resource Models

class GPUResource(BaseModel):
    """GPU resource specification"""
    type: str = Field(..., description="GPU type (e.g., t4, v100, a100)")
    count: int = Field(1, ge=1, le=8, description="Number of GPUs required")
    memory_gb: Optional[int] = Field(None, ge=1, description="GPU memory in GB")
    compute_capability: Optional[float] = Field(None, ge=3.0, description="Minimum compute capability")

class ResourceRequirements(BaseModel):
    """Resource requirements specification"""
    cpu: int = Field(1, ge=1, le=128, description="Number of CPU cores")
    memory_gb: int = Field(1, ge=1, le=1024, description="Memory in GB")
    storage_gb: int = Field(10, ge=1, le=10000, description="Storage in GB")
    gpu: Optional[GPUResource] = Field(None, description="GPU requirements")
    network_mbps: Optional[int] = Field(None, ge=1, description="Network bandwidth in Mbps")

class ResourceAllocation(BaseModel):
    """Allocated resources for a job"""
    node_id: str = Field(..., description="Allocated node ID")
    resources: ResourceRequirements = Field(..., description="Allocated resources")
    lease_duration_minutes: int = Field(..., description="Resource lease duration")
    cost_estimate_usd: Optional[float] = Field(None, description="Estimated cost in USD")

# Job Models

class JobConstraints(BaseModel):
    """Job execution constraints"""
    allowed_node_tags: List[str] = Field(default_factory=list, description="Required node tags")
    disallowed_nodes: List[str] = Field(default_factory=list, description="Excluded node IDs")
    locality_preference: Optional[str] = Field(None, description="Locality preference")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    timeout_minutes: int = Field(60, ge=1, le=1440, description="Job timeout in minutes")

class JobPolicy(BaseModel):
    """Job execution policy"""
    attestation_required: bool = Field(False, description="Require node attestation")
    privacy_level: str = Field("standard", description="Privacy protection level")
    max_cost_usd: Optional[float] = Field(None, ge=0, description="Maximum allowed cost")
    verifier: Optional[str] = Field(None, description="Verifier to use")
    audit_level: str = Field("standard", description="Audit logging level")

class ArtifactSpec(BaseModel):
    """Artifact specification"""
    name: str = Field(..., description="Artifact name")
    type: str = Field(..., description="Artifact type (s3, http, local)")
    uri: str = Field(..., description="Artifact URI")
    checksum: Optional[str] = Field(None, description="SHA256 checksum")
    size_bytes: Optional[int] = Field(None, ge=0, description="Artifact size")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class CommandSpec(BaseModel):
    """Command specification"""
    type: str = Field(..., description="Command type (container, binary, python, wasm)")
    image: Optional[str] = Field(None, description="Container image (for container type)")
    entrypoint: Optional[List[str]] = Field(None, description="Command entrypoint")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    working_dir: Optional[str] = Field(None, description="Working directory")

class OutputSpec(BaseModel):
    """Output specification"""
    path: str = Field(..., description="Output path")
    upload: bool = Field(True, description="Whether to upload output")
    store: Optional[str] = Field(None, description="Storage destination")
    compress: bool = Field(False, description="Whether to compress output")

class MonitoringSpec(BaseModel):
    """Monitoring specification"""
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    logs_enabled: bool = Field(True, description="Enable log collection")
    traces_enabled: bool = Field(False, description="Enable distributed tracing")
    custom_metrics: Dict[str, str] = Field(default_factory=dict, description="Custom metrics")

class JobManifest(BaseModel):
    """Complete job manifest"""
    name: str = Field(..., min_length=1, max_length=64, description="Job name")
    description: Optional[str] = Field(None, max_length=500, description="Job description")
    type: str = Field("inference", description="Job type")
    
    resources: ResourceRequirements = Field(..., description="Resource requirements")
    constraints: JobConstraints = Field(default_factory=JobConstraints, description="Execution constraints")
    policy: JobPolicy = Field(default_factory=JobPolicy, description="Execution policy")
    
    artifacts: List[ArtifactSpec] = Field(default_factory=list, description="Input artifacts")
    command: CommandSpec = Field(..., description="Command to execute")
    outputs: List[OutputSpec] = Field(default_factory=list, description="Output specifications")
    
    monitoring: MonitoringSpec = Field(default_factory=MonitoringSpec, description="Monitoring settings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Job name must contain only alphanumeric characters, underscores, and hyphens')
        return v

# Request/Response Models

class JobSubmitRequest(BaseModel):
    """Job submission request"""
    manifest: JobManifest = Field(..., description="Job manifest")
    priority: Optional[int] = Field(2, ge=1, le=4, description="Job priority")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    tags: List[str] = Field(default_factory=list, description="Job tags")

class JobSubmitResponse(BaseModel):
    """Job submission response"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    allocation: Optional[ResourceAllocation] = Field(None, description="Resource allocation")
    estimated_start_time: Optional[datetime] = Field(None, description="Estimated start time")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    cost_estimate_usd: Optional[float] = Field(None, description="Cost estimate in USD")

class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Completion progress (0-1)")
    
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    result: Optional[Dict[str, Any]] = Field(None, description="Job result")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource usage statistics")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    
    allocation: Optional[ResourceAllocation] = Field(None, description="Resource allocation")
    logs_url: Optional[str] = Field(None, description="Logs streaming URL")

class CancelJobResponse(BaseModel):
    """Job cancellation response"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="New status (should be cancelled)")
    cancelled_at: datetime = Field(..., description="Cancellation timestamp")
    message: str = Field("Job cancelled successfully", description="Cancellation message")

class JobSummary(BaseModel):
    """Job summary for listings"""
    job_id: str = Field(..., description="Job identifier")
    name: str = Field(..., description="Job name")
    status: JobStatus = Field(..., description="Current status")
    type: str = Field(..., description="Job type")
    priority: int = Field(..., description="Job priority")
    
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    duration_seconds: Optional[float] = Field(None, description="Job duration in seconds")
    cost_usd: Optional[float] = Field(None, description="Actual cost in USD")
    
    tags: List[str] = Field(default_factory=list, description="Job tags")

class JobListResponse(BaseModel):
    """Job list response"""
    jobs: List[JobSummary] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    offset: int = Field(..., description="Offset used")
    limit: int = Field(..., description="Limit used")
    has_more: bool = Field(..., description="Whether more jobs are available")

# System Models

class SystemMetricsResponse(BaseModel):
    """System metrics response"""
    active_jobs: int = Field(..., description="Number of active jobs")
    queue_depth: int = Field(..., description="Job queue depth")
    
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Job success rate")
    avg_job_duration: float = Field(..., description="Average job duration in seconds")
    
    resource_usage: Dict[str, float] = Field(..., description="Current resource usage")
    cost_metrics: Dict[str, float] = Field(default_factory=dict, description="Cost metrics")
    
    scheduler_metrics: Dict[str, Any] = Field(default_factory=dict, description="Scheduler metrics")
    node_health: Dict[str, Any] = Field(default_factory=dict, description="Node health statistics")

# Internal Models (for service implementation)

class JobStatus(BaseModel):
    """Internal job status tracking"""
    job_id: str
    status: str
    user_id: str
    tenant_id: str
    manifest: Dict[str, Any]
    allocation: Optional[ResourceAllocation] = None
    
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    priority: int = 2
    timeout_seconds: int = 3600
    duration_seconds: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True