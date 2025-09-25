"""
LoomOS Job Manifest

Advanced job manifest system for LoomOS with comprehensive validation and features.
Supports complex job specifications including distributed training, resource allocation,
artifact management, and advanced scheduling constraints.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator
from enum import Enum

class JobType(str, Enum):
    """Supported job types"""
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    DATA_PROCESSING = "data_processing"
    MODEL_ADAPTATION = "model_adaptation"
    VERIFICATION = "verification"

class ResourceType(str, Enum):
    """Supported resource types"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

class JobPriority(int, Enum):
    """Job priority levels"""
    LOWEST = 1
    LOW = 3
    NORMAL = 5
    HIGH = 7
    HIGHEST = 10

class ResourceSpec(BaseModel):
    """Resource specification"""
    resource_type: ResourceType
    amount: float = Field(gt=0, description="Amount of resource required")
    unit: str = Field(default="cores", description="Unit of measurement")
    
    class Config:
        use_enum_values = True

class ArtifactSpec(BaseModel):
    """Artifact specification"""
    name: str = Field(description="Artifact name")
    type: str = Field(description="Artifact type (model, dataset, code, etc.)")
    source: str = Field(description="Artifact source (URL, S3 path, etc.)")
    destination: Optional[str] = Field(None, description="Destination path in container")
    required: bool = Field(True, description="Whether artifact is required")
    checksum: Optional[str] = Field(None, description="Artifact checksum for verification")

class CommandSpec(BaseModel):
    """Command specification"""
    executable: str = Field(description="Command to execute")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    working_dir: Optional[str] = Field(None, description="Working directory")
    shell: bool = Field(False, description="Whether to run in shell")

class ConstraintSpec(BaseModel):
    """Job constraints"""
    node_selector: Optional[Dict[str, str]] = Field(None, description="Node selection constraints")
    tolerations: Optional[List[Dict[str, Any]]] = Field(None, description="Node tolerations")
    affinity: Optional[Dict[str, Any]] = Field(None, description="Pod affinity rules")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    timeout_seconds: Optional[int] = Field(None, gt=0, description="Job timeout")
    deadline_seconds: Optional[int] = Field(None, gt=0, description="Job deadline")

class PolicySpec(BaseModel):
    """Job execution policy"""
    restart_policy: str = Field("OnFailure", description="Restart policy")
    cleanup_policy: str = Field("Always", description="Cleanup policy")
    security_context: Optional[Dict[str, Any]] = Field(None, description="Security context")
    network_policy: Optional[Dict[str, Any]] = Field(None, description="Network policy")
    resource_quotas: Optional[Dict[str, Any]] = Field(None, description="Resource quotas")

class CheckpointSpec(BaseModel):
    """Checkpointing configuration"""
    enabled: bool = Field(True, description="Enable checkpointing")
    interval_seconds: int = Field(300, gt=0, description="Checkpoint interval")
    storage_path: Optional[str] = Field(None, description="Checkpoint storage path")
    max_checkpoints: int = Field(5, gt=0, description="Maximum checkpoints to keep")
    compression: bool = Field(True, description="Compress checkpoints")

class OutputSpec(BaseModel):
    """Output specification"""
    name: str = Field(description="Output name")
    type: str = Field(description="Output type")
    path: str = Field(description="Output path")
    required: bool = Field(True, description="Whether output is required")
    format: Optional[str] = Field(None, description="Output format")

class NotificationSpec(BaseModel):
    """Notification specification"""
    type: str = Field(description="Notification type (email, webhook, slack)")
    target: str = Field(description="Notification target")
    events: List[str] = Field(description="Events to notify on")
    template: Optional[str] = Field(None, description="Notification template")

class MonitoringSpec(BaseModel):
    """Monitoring configuration"""
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    logs_enabled: bool = Field(True, description="Enable log collection")
    profiling_enabled: bool = Field(False, description="Enable performance profiling")
    alerts: Optional[List[Dict[str, Any]]] = Field(None, description="Alert configurations")
    dashboards: Optional[List[str]] = Field(None, description="Dashboard configurations")

class DistributedSpec(BaseModel):
    """Distributed training configuration"""
    world_size: int = Field(1, gt=0, description="Number of workers")
    master_port: int = Field(29500, gt=1024, lt=65536, description="Master port")
    backend: str = Field("nccl", description="Communication backend")
    compression_ratio: float = Field(0.01, gt=0, le=1.0, description="Gradient compression ratio")
    fault_tolerance: bool = Field(True, description="Enable fault tolerance")

class JobManifest(BaseModel):
    """
    Comprehensive job manifest for LoomOS
    
    Defines all aspects of a job including resources, execution parameters,
    constraints, monitoring, and lifecycle management.
    """
    
    # Core job information
    job_id: Optional[str] = Field(None, description="Unique job identifier (auto-generated if not provided)")
    job_type: JobType = Field(description="Type of job to execute")
    name: str = Field(min_length=1, max_length=255, description="Human-readable job name")
    description: Optional[str] = Field(None, description="Job description")
    owner: Optional[str] = Field(None, description="Job owner")
    
    # Container and execution
    image: str = Field(description="Container image to run")
    command: Optional[CommandSpec] = Field(None, description="Command specification")
    environment: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    
    # Resources and scheduling
    resources: List[ResourceSpec] = Field(description="Resource requirements")
    priority: JobPriority = Field(JobPriority.NORMAL, description="Job priority")
    constraints: Optional[ConstraintSpec] = Field(None, description="Scheduling constraints")
    
    # Distributed training
    distributed: Optional[DistributedSpec] = Field(None, description="Distributed training configuration")
    
    # Data and artifacts
    artifacts: Optional[List[ArtifactSpec]] = Field(None, description="Input artifacts")
    outputs: Optional[List[OutputSpec]] = Field(None, description="Expected outputs")
    
    # Policies and lifecycle
    policy: Optional[PolicySpec] = Field(None, description="Execution policies")
    checkpointing: Optional[CheckpointSpec] = Field(None, description="Checkpointing configuration")
    
    # Monitoring and notifications
    monitoring: Optional[MonitoringSpec] = Field(None, description="Monitoring configuration")
    notifications: Optional[List[NotificationSpec]] = Field(None, description="Notification settings")
    
    # Metadata
    tags: Optional[Dict[str, str]] = Field(None, description="User-defined tags")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    
    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        """Set creation timestamp if not provided"""
        return v or datetime.now(timezone.utc)
    
    @validator('job_id', pre=True, always=True)
    def generate_job_id(cls, v, values):
        """Generate job ID if not provided"""
        if not v and 'name' in values:
            import uuid
            name_part = values['name'].lower().replace(' ', '_')[:20]
            uuid_part = str(uuid.uuid4())[:8]
            return f"job_{name_part}_{uuid_part}"
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API submission"""
        return self.dict(exclude_none=True, by_alias=True)
    
    def to_yaml(self) -> str:
        """Convert to YAML format"""
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def to_json(self) -> str:
        """Convert to JSON format"""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobManifest':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_yaml_file(cls, file_path: str) -> 'JobManifest':
        """Load from YAML file"""
        import yaml
        from pathlib import Path
        
        with open(file_path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'JobManifest':
        """Load from JSON file"""
        import json
        from pathlib import Path
        
        with open(file_path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"

# Convenience builders
class JobManifestBuilder:
    """Builder pattern for creating job manifests"""
    
    def __init__(self, job_type: JobType, name: str, image: str):
        self.manifest_data = {
            "job_type": job_type,
            "name": name,
            "image": image,
            "resources": []
        }
    
    def description(self, description: str) -> 'JobManifestBuilder':
        """Set job description"""
        self.manifest_data["description"] = description
        return self
    
    def owner(self, owner: str) -> 'JobManifestBuilder':
        """Set job owner"""
        self.manifest_data["owner"] = owner
        return self
    
    def command(self, executable: str, args: List[str] = None, working_dir: str = None) -> 'JobManifestBuilder':
        """Set command"""
        self.manifest_data["command"] = CommandSpec(
            executable=executable,
            args=args or [],
            working_dir=working_dir
        )
        return self
    
    def environment(self, env_vars: Dict[str, str]) -> 'JobManifestBuilder':
        """Set environment variables"""
        self.manifest_data["environment"] = env_vars
        return self
    
    def add_resource(self, resource_type: ResourceType, amount: float, unit: str = "cores") -> 'JobManifestBuilder':
        """Add resource requirement"""
        self.manifest_data["resources"].append(
            ResourceSpec(resource_type=resource_type, amount=amount, unit=unit)
        )
        return self
    
    def priority(self, priority: JobPriority) -> 'JobManifestBuilder':
        """Set job priority"""
        self.manifest_data["priority"] = priority
        return self
    
    def distributed(self, world_size: int, compression_ratio: float = 0.01) -> 'JobManifestBuilder':
        """Configure distributed training"""
        self.manifest_data["distributed"] = DistributedSpec(
            world_size=world_size,
            compression_ratio=compression_ratio
        )
        return self
    
    def add_artifact(self, name: str, artifact_type: str, source: str, destination: str = None) -> 'JobManifestBuilder':
        """Add input artifact"""
        if "artifacts" not in self.manifest_data:
            self.manifest_data["artifacts"] = []
        
        self.manifest_data["artifacts"].append(
            ArtifactSpec(name=name, type=artifact_type, source=source, destination=destination)
        )
        return self
    
    def add_output(self, name: str, output_type: str, path: str) -> 'JobManifestBuilder':
        """Add expected output"""
        if "outputs" not in self.manifest_data:
            self.manifest_data["outputs"] = []
        
        self.manifest_data["outputs"].append(
            OutputSpec(name=name, type=output_type, path=path)
        )
        return self
    
    def timeout(self, timeout_seconds: int) -> 'JobManifestBuilder':
        """Set job timeout"""
        if "constraints" not in self.manifest_data:
            self.manifest_data["constraints"] = {}
        self.manifest_data["constraints"]["timeout_seconds"] = timeout_seconds
        return self
    
    def max_retries(self, retries: int) -> 'JobManifestBuilder':
        """Set maximum retries"""
        if "constraints" not in self.manifest_data:
            self.manifest_data["constraints"] = {}
        self.manifest_data["constraints"]["max_retries"] = retries
        return self
    
    def enable_checkpointing(self, interval_seconds: int = 300) -> 'JobManifestBuilder':
        """Enable checkpointing"""
        self.manifest_data["checkpointing"] = CheckpointSpec(
            enabled=True,
            interval_seconds=interval_seconds
        )
        return self
    
    def add_notification(self, notification_type: str, target: str, events: List[str]) -> 'JobManifestBuilder':
        """Add notification"""
        if "notifications" not in self.manifest_data:
            self.manifest_data["notifications"] = []
        
        self.manifest_data["notifications"].append(
            NotificationSpec(type=notification_type, target=target, events=events)
        )
        return self
    
    def tags(self, tags: Dict[str, str]) -> 'JobManifestBuilder':
        """Set job tags"""
        self.manifest_data["tags"] = tags
        return self
    
    def build(self) -> JobManifest:
        """Build the job manifest"""
        return JobManifest(**self.manifest_data)

def create_training_manifest(name: str, image: str, world_size: int = 1) -> JobManifestBuilder:
    """Create a training job manifest builder"""
    return (JobManifestBuilder(JobType.TRAINING, name, image)
            .add_resource(ResourceType.GPU, world_size)
            .add_resource(ResourceType.MEMORY, 32 * world_size, "GB")
            .distributed(world_size))

def create_inference_manifest(name: str, image: str) -> JobManifestBuilder:
    """Create an inference job manifest builder"""
    return (JobManifestBuilder(JobType.INFERENCE, name, image)
            .add_resource(ResourceType.GPU, 1)
            .add_resource(ResourceType.MEMORY, 16, "GB"))

def create_evaluation_manifest(name: str, image: str) -> JobManifestBuilder:
    """Create an evaluation job manifest builder"""
    return (JobManifestBuilder(JobType.EVALUATION, name, image)
            .add_resource(ResourceType.CPU, 4)
            .add_resource(ResourceType.MEMORY, 8, "GB"))

# Export public API
__all__ = [
    "JobManifest",
    "JobManifestBuilder",
    "JobType",
    "ResourceType", 
    "JobPriority",
    "ResourceSpec",
    "ArtifactSpec",
    "CommandSpec",
    "ConstraintSpec",
    "PolicySpec",
    "CheckpointSpec",
    "OutputSpec",
    "NotificationSpec",
    "MonitoringSpec",
    "DistributedSpec",
    "create_training_manifest",
    "create_inference_manifest",
    "create_evaluation_manifest"
]