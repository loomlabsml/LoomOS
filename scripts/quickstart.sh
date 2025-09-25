#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting LoomOS Quickstart Demo"
echo "=================================="

# Check dependencies
echo "📋 Checking dependencies..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed. Aborting." >&2; exit 1; }

echo "✅ All dependencies found"

# Stop any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose down --remove-orphans || true

# Build and start infrastructure
echo "🏗️  Building and starting infrastructure..."
docker-compose up -d postgres minio redis nats

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are healthy
echo "🔍 Checking service health..."
for service in postgres minio redis nats; do
    if docker-compose ps $service | grep -q "Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed to start"
        docker-compose logs $service
        exit 1
    fi
done

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -e . --quiet || {
    echo "❌ Failed to install dependencies. Trying with --user flag..."
    pip3 install -e . --user --quiet
}

# Start LoomOS services
echo "🚀 Starting LoomOS services..."
docker-compose up -d loomctl loomnode

# Wait for LoomOS services
echo "⏳ Waiting for LoomOS services..."
sleep 15

# Check LoomOS service health
echo "🔍 Checking LoomOS service health..."
if curl -s http://localhost:8000/v1/health > /dev/null; then
    echo "✅ LoomCtl is running"
else
    echo "❌ LoomCtl failed to start"
    docker-compose logs loomctl
    exit 1
fi

# Run the demo
echo "🎯 Running demo workload..."
python3 examples/demo/submit_demo.py

echo ""
echo "🎉 LoomOS Demo Complete!"
echo "========================"
echo ""
echo "📊 Next steps:"
echo "  • View logs: docker-compose logs -f"
echo "  • API docs: http://localhost:8000/docs"
echo "  • Metrics: http://localhost:9090 (Prometheus)"
echo "  • MinIO: http://localhost:9001 (admin/admin123)"
echo ""
echo "🛑 To stop: docker-compose down"