"""
LoomOS Nexus Node - Distributed Training Coordinator

Main entry point for Nexus distributed training nodes that implement
ultra-low communication overhead distributed optimization.

Features:
- Automatic worker discovery and coordination
- Dynamic resource allocation and scaling
- Fault tolerance and recovery
- Performance monitoring and optimization
- Multi-GPU and multi-node support
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .worker import Worker, WorkerConfig, CompressionMethod
from .attest import TEEAttestationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nexus_worker.log')
    ]
)

logger = logging.getLogger(__name__)

async def create_worker_from_args(args) -> Worker:
    """Create worker from command line arguments"""
    config = WorkerConfig(
        worker_id=args.worker_id or f"nexus_worker_{os.getpid()}",
        worker_rank=args.rank,
        world_size=args.world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        backend=args.backend,
        compression_method=CompressionMethod(args.compression_method),
        compression_ratio=args.compression_ratio,
        batch_size=args.batch_size,
        max_concurrent_tasks=args.max_tasks,
        max_memory_gb=args.max_memory,
        max_cpu_cores=args.max_cpus
    )
    
    return Worker(config)

async def run_worker(args):
    """Run the Nexus worker"""
    logger.info("Starting Nexus distributed training worker")
    
    try:
        # Create and configure worker
        worker = await create_worker_from_args(args)
        
        # Initialize TEE attestation if enabled
        if args.enable_tee:
            attestation_service = TEEAttestationService()
            # In production, perform attestation here
            logger.info("TEE attestation enabled")
        
        # Start worker
        await worker.run()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Nexus Distributed Training Worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Worker identification
    parser.add_argument(
        '--worker-id', 
        type=str, 
        help='Unique worker identifier'
    )
    parser.add_argument(
        '--rank', 
        type=int, 
        default=0,
        help='Worker rank in distributed training'
    )
    parser.add_argument(
        '--world-size', 
        type=int, 
        default=1,
        help='Total number of workers'
    )
    
    # Network configuration
    parser.add_argument(
        '--master-addr', 
        type=str, 
        default='localhost',
        help='Master node address'
    )
    parser.add_argument(
        '--master-port', 
        type=int, 
        default=29500,
        help='Master node port'
    )
    parser.add_argument(
        '--backend', 
        type=str, 
        choices=['nccl', 'gloo', 'mpi'],
        default='gloo',
        help='Distributed training backend'
    )
    
    # Compression settings
    parser.add_argument(
        '--compression-method', 
        type=str, 
        choices=['none', 'top_k', 'random_k', 'threshold', 'quantization', 'sparsification'],
        default='top_k',
        help='Gradient compression method'
    )
    parser.add_argument(
        '--compression-ratio', 
        type=float, 
        default=0.01,
        help='Compression ratio (0.0-1.0)'
    )
    
    # Training configuration
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--max-tasks', 
        type=int, 
        default=4,
        help='Maximum concurrent tasks'
    )
    
    # Resource limits
    parser.add_argument(
        '--max-memory', 
        type=float, 
        default=8.0,
        help='Maximum memory usage (GB)'
    )
    parser.add_argument(
        '--max-cpus', 
        type=int, 
        default=4,
        help='Maximum CPU cores'
    )
    
    # Security
    parser.add_argument(
        '--enable-tee', 
        action='store_true',
        help='Enable TEE attestation'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--log-file', 
        type=str, 
        default='nexus_worker.log',
        help='Log file path'
    )
    
    return parser.parse_args()

def setup_logging(args):
    """Setup logging configuration"""
    level = getattr(logging, args.log_level)
    
    # Configure root logger
    logging.getLogger().setLevel(level)
    
    # Add file handler if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

async def main():
    """Main entry point"""
    args = parse_arguments()
    setup_logging(args)
    
    logger.info(f"Nexus Worker starting with config: {vars(args)}")
    
    # Environment validation
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logger.info(f"CUDA devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Set process title for monitoring
    try:
        import setproctitle
        setproctitle.setproctitle(f"nexus-worker-{args.rank}")
    except ImportError:
        pass
    
    # Run worker
    await run_worker(args)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Nexus worker terminated by user")
    except Exception as e:
        logger.error(f"Nexus worker failed: {e}", exc_info=True)
        sys.exit(1)