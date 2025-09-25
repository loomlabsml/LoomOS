/**
 * LoomOS JavaScript/TypeScript SDK
 * 
 * Enterprise-grade SDK for LoomOS - The Iron Suit for AI Models
 * 
 * Features:
 * - Complete job management and monitoring
 * - Real-time log streaming via WebSocket
 * - Cluster management and worker coordination
 * - AI agent execution and model adaptation
 * - Marketplace integration
 * - TypeScript support with full type safety
 * - Promise-based async API
 * - Event-driven architecture
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import WebSocket from 'ws';
import { EventEmitter } from 'eventemitter3';

// Core types
export interface JobResource {
  resource_type: 'cpu' | 'gpu' | 'memory' | 'storage' | 'network';
  amount: number;
  unit?: string;
}

export interface JobSpec {
  job_id?: string;
  job_type: 'training' | 'inference' | 'evaluation' | 'data_processing' | 'model_adaptation' | 'verification';
  name: string;
  description?: string;
  image: string;
  command?: string[];
  environment?: Record<string, string>;
  resources?: JobResource[];
  priority?: number;
  timeout_seconds?: number;
  retry_limit?: number;
  world_size?: number;
  compression_ratio?: number;
  tags?: Record<string, string>;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';
  progress?: number;
  message?: string;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
  result?: any;
  error?: string;
  resources_used?: JobResource[];
  assigned_workers?: string[];
}

export interface WorkerInfo {
  worker_id: string;
  status: 'idle' | 'busy' | 'offline' | 'maintenance';
  capabilities?: Record<string, any>;
  current_load?: number;
  last_heartbeat?: string;
}

export interface ClusterStats {
  total_workers: number;
  active_workers: number;
  total_jobs: number;
  active_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  cpu_utilization: number;
  gpu_utilization: number;
  memory_utilization: number;
  avg_job_duration_seconds: number;
  jobs_per_hour: number;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

export interface MarketplaceListing {
  listing_id: string;
  title: string;
  description: string;
  price: string;
  quality_score: number;
  provider_id: string;
}

export interface AITaskRequest {
  prompt: string;
  tools?: string[];
  context?: Record<string, any>;
}

export interface ModelAdaptationRequest {
  model_id: string;
  adaptation_type: 'lora' | 'qlora' | 'full_finetune';
  training_data: Record<string, any>;
  config: Record<string, any>;
}

export interface ContentVerificationRequest {
  content: string;
  type: 'factual' | 'safety' | 'quality' | 'bias';
  context?: Record<string, any>;
}

// Exceptions
export class LoomOSError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LoomOSError';
  }
}

export class JobNotFoundError extends LoomOSError {
  constructor(jobId: string) {
    super(`Job not found: ${jobId}`);
    this.name = 'JobNotFoundError';
  }
}

export class AuthenticationError extends LoomOSError {
  constructor(message: string = 'Authentication failed') {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class APIError extends LoomOSError {
  public statusCode?: number;
  
  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = 'APIError';
    this.statusCode = statusCode;
  }
}

// Job specification builder
export class JobSpecBuilder {
  private spec: JobSpec;

  constructor(jobType: JobSpec['job_type'], name: string, image: string) {
    this.spec = {
      job_type: jobType,
      name,
      image,
      resources: []
    };
  }

  command(command: string[]): JobSpecBuilder {
    this.spec.command = command;
    return this;
  }

  environment(env: Record<string, string>): JobSpecBuilder {
    this.spec.environment = env;
    return this;
  }

  addResource(resourceType: JobResource['resource_type'], amount: number, unit: string = 'cores'): JobSpecBuilder {
    if (!this.spec.resources) {
      this.spec.resources = [];
    }
    this.spec.resources.push({ resource_type: resourceType, amount, unit });
    return this;
  }

  priority(priority: number): JobSpecBuilder {
    this.spec.priority = priority;
    return this;
  }

  timeout(timeoutSeconds: number): JobSpecBuilder {
    this.spec.timeout_seconds = timeoutSeconds;
    return this;
  }

  retryLimit(retryLimit: number): JobSpecBuilder {
    this.spec.retry_limit = retryLimit;
    return this;
  }

  distributed(worldSize: number, compressionRatio: number = 0.01): JobSpecBuilder {
    this.spec.world_size = worldSize;
    this.spec.compression_ratio = compressionRatio;
    return this;
  }

  tags(tags: Record<string, string>): JobSpecBuilder {
    this.spec.tags = tags;
    return this;
  }

  build(): JobSpec {
    return { ...this.spec };
  }
}

// Main client class
export class LoomClient extends EventEmitter {
  private axios: AxiosInstance;
  private websockets: Map<string, WebSocket> = new Map();

  constructor(
    private apiUrl: string,
    private token?: string,
    private timeout: number = 300000
  ) {
    super();
    
    this.axios = axios.create({
      baseURL: this.apiUrl.replace(/\/$/, ''),
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
        ...(this.token && { Authorization: `Bearer ${this.token}` })
      }
    });

    // Add response interceptor for error handling
    this.axios.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error: any) => {
        if (error.response?.status === 401) {
          throw new AuthenticationError('Invalid authentication token');
        } else if (error.response?.status === 404) {
          throw new JobNotFoundError('Resource not found');
        } else if (error.response) {
          const message = error.response.data?.detail || error.message;
          throw new APIError(`API error (${error.response.status}): ${message}`, error.response.status);
        } else {
          throw new LoomOSError(`Request failed: ${error.message}`);
        }
      }
    );
  }

  // Job management
  async submitJob(jobSpec: JobSpec): Promise<string> {
    const response = await this.axios.post<{ job_id: string }>('/v1/jobs', jobSpec);
    return response.data.job_id;
  }

  async getJob(jobId: string): Promise<JobStatus> {
    const response = await this.axios.get<JobStatus>(`/v1/jobs/${jobId}`);
    return response.data;
  }

  async listJobs(options: {
    status?: JobStatus['status'];
    limit?: number;
    offset?: number;
  } = {}): Promise<JobStatus[]> {
    const params: Record<string, string> = {};
    if (options.status) params.status = options.status;
    if (options.limit !== undefined) params.limit = options.limit.toString();
    if (options.offset !== undefined) params.offset = options.offset.toString();

    const response = await this.axios.get<JobStatus[]>('/v1/jobs', { params });
    return response.data;
  }

  async cancelJob(jobId: string): Promise<void> {
    await this.axios.delete(`/v1/jobs/${jobId}`);
  }

  async waitForJob(
    jobId: string,
    options: {
      pollInterval?: number;
      timeout?: number;
      onProgress?: (job: JobStatus) => void;
    } = {}
  ): Promise<JobStatus> {
    const { pollInterval = 5000, timeout, onProgress } = options;
    const startTime = Date.now();

    while (true) {
      const job = await this.getJob(jobId);
      
      if (onProgress) {
        onProgress(job);
      }

      if (['completed', 'failed', 'cancelled'].includes(job.status)) {
        return job;
      }

      if (timeout && (Date.now() - startTime) > timeout) {
        throw new LoomOSError(`Timeout waiting for job ${jobId}`);
      }

      await this.sleep(pollInterval);
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Log streaming (simplified for browser compatibility)
  async streamLogs(
    jobId: string,
    onLog: (log: LogEntry) => void,
    onError?: (error: Error) => void
  ): Promise<() => void> {
    let isStreaming = true;
    
    const streamLoop = async () => {
      while (isStreaming) {
        try {
          // For browser compatibility, we'll poll the logs endpoint
          // In a real implementation, you might use Server-Sent Events
          const response = await this.axios.get(`/v1/jobs/${jobId}/logs`);
          
          if (response.data) {
            // Parse server-sent events format
            const lines = response.data.split('\n');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const logEntry = JSON.parse(line.slice(6)) as LogEntry;
                  onLog(logEntry);
                } catch (parseError) {
                  console.warn('Failed to parse log line:', line);
                }
              }
            }
          }
          
          await this.sleep(1000); // Poll every second
        } catch (error) {
          if (onError) {
            onError(error as Error);
          } else {
            this.emit('error', error);
          }
          break;
        }
      }
    };

    streamLoop();

    // Return cleanup function
    return () => {
      isStreaming = false;
    };
  }

  // WebSocket streaming
  async connectWebSocket(clientId: string): Promise<WebSocket> {
    const wsUrl = this.apiUrl.replace(/^http/, 'ws') + `/ws/${clientId}`;
    const ws = new WebSocket(wsUrl, {
      headers: this.token ? { Authorization: `Bearer ${this.token}` } : {}
    });

    return new Promise((resolve, reject) => {
      ws.on('open', () => {
        this.websockets.set(clientId, ws);
        this.emit('websocket:connected', clientId);
        resolve(ws);
      });

      ws.on('error', (error: any) => {
        this.emit('websocket:error', clientId, error);
        reject(error);
      });

      ws.on('message', (data: any) => {
        try {
          const message = JSON.parse(data.toString());
          this.emit('websocket:message', clientId, message);
        } catch (error) {
          console.warn('Failed to parse WebSocket message:', data.toString());
        }
      });

      ws.on('close', () => {
        this.websockets.delete(clientId);
        this.emit('websocket:disconnected', clientId);
      });
    });
  }

  disconnectWebSocket(clientId: string): void {
    const ws = this.websockets.get(clientId);
    if (ws) {
      ws.close();
    }
  }

  // Cluster management
  async getClusterStats(): Promise<ClusterStats> {
    const response = await this.axios.get<ClusterStats>('/v1/cluster/stats');
    return response.data;
  }

  async listWorkers(): Promise<WorkerInfo[]> {
    const response = await this.axios.get<WorkerInfo[]>('/v1/workers');
    return response.data;
  }

  // Health monitoring
  async healthCheck(): Promise<any> {
    const response = await this.axios.get('/v1/health');
    return response.data;
  }

  // Marketplace
  async listMarketplaceListings(listingType?: string): Promise<{ listings: MarketplaceListing[] }> {
    const params = listingType ? { listing_type: listingType } : undefined;
    const response = await this.axios.get('/v1/marketplace/listings', { params });
    return response.data;
  }

  async purchaseListing(listingId: string, quantity: number = 1): Promise<any> {
    const response = await this.axios.post('/v1/marketplace/purchase', {
      listing_id: listingId,
      quantity
    });
    return response.data;
  }

  // AI operations
  async executeAgentTask(request: AITaskRequest): Promise<any> {
    const response = await this.axios.post('/v1/ai/agents/execute', request);
    return response.data;
  }

  async adaptModel(request: ModelAdaptationRequest): Promise<any> {
    const response = await this.axios.post('/v1/ai/models/adapt', request);
    return response.data;
  }

  async verifyContent(request: ContentVerificationRequest): Promise<any> {
    const response = await this.axios.post('/v1/ai/verify', request);
    return response.data;
  }

  // Cleanup
  async disconnect(): Promise<void> {
    // Close all WebSocket connections
    for (const [clientId, ws] of this.websockets) {
      ws.close();
    }
    this.websockets.clear();
    
    this.removeAllListeners();
  }
}

// Convenience functions
export function createTrainingJob(name: string, image: string, command: string[], worldSize: number = 1): JobSpecBuilder {
  return new JobSpecBuilder('training', name, image)
    .command(command)
    .distributed(worldSize)
    .addResource('gpu', worldSize)
    .addResource('memory', 32 * worldSize, 'GB');
}

export function createInferenceJob(name: string, image: string, command: string[]): JobSpecBuilder {
  return new JobSpecBuilder('inference', name, image)
    .command(command)
    .addResource('gpu', 1)
    .addResource('memory', 16, 'GB');
}

export function createEvaluationJob(name: string, image: string, command: string[]): JobSpecBuilder {
  return new JobSpecBuilder('evaluation', name, image)
    .command(command)
    .addResource('cpu', 4)
    .addResource('memory', 8, 'GB');
}

// Export everything
export default LoomClient;