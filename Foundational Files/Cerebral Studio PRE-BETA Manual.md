# Cerebral Studio - Complete Assembly Manual

## Overview

Cerebral Studio is a unified web application that wraps the Cerebral SDK neuromorphic engine with a modern frontend interface, Ollama integration, and comprehensive MCP server management. This platform provides SillyTavern-like functionality with advanced AI orchestration capabilities.

---

## System Architecture

```
Cerebral Studio Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│   React Frontend│◄──►│   FastAPI Backend│◄──►│  Cerebral SDK Core │
└─────────────────┘    └──────────────────┘    └─────────┬──────────┘
                                                         │
                                                 ┌───────┴──────────┐
                                                 │   MCP Adapters   │
                                                 ├──────────────────┤
                                         ┌───────┤ OpenAI           │
                                         │       ├──────────────────┤
                                 ┌───────┤       │ Anthropic        │
                                 │       │       ├──────────────────┤
                         ┌───────┤       │       │ Google           │
                         │       │       │       ├──────────────────┤
                 ┌───────┤       │       │       │ DeepSeek         │
                 │       │       │       │       ├──────────────────┤
         ┌───────┤       │       │       │       │ Ollama (Turbo!)  │
         │       │       │       │       │       └──────────────────┘
         │       │       │       │       │
┌────────────────┐ ┌─────────────┐ ┌─────────────┐
│  Model Manager │ │ Port Manager│ │ MCP Manager │
└────────────────┘ └─────────────┘ └─────────────┘
```

---

## Prerequisites

### Hardware Requirements
- CPU: 4+ cores
- RAM: 8GB+ (16GB recommended for Ollama)
- Storage: 20GB+ free space
- GPU: Optional but recommended for local models (NVIDIA with 8GB+ VRAM)

### Software Requirements
- Docker and Docker Compose
- Python 3.9+
- Node.js 16+ (for development)
- Git

---

## Complete Installation Guide

### Method 1: Docker Deployment (Recommended)

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cerebral-studio.git
cd cerebral-studio
```

#### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys and settings
nano .env
```

#### 3. Configure .env File
```env
# Core Settings
LITE_MODE=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Provider API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-your-anthropic-key-here
GOOGLE_PROJECT=your-google-project-id
GOOGLE_LOCATION=us-central1

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2-turbo
ENABLE_OLLAMA=true
OLLAMA_NUM_GPU=1  # Set to 0 for CPU-only

# MCP Server Defaults
DEEPSEEK_MCP_URL=http://localhost:8080/generate

# Advanced Settings
MAX_CONTEXT_TOKENS=128000
RESERVED_OUTPUT_TOKENS=2048
```

#### 4. Start with Docker Compose
```bash
docker-compose up -d
```

#### 5. Access the Application
Open your browser to: http://localhost:8000

### Method 2: Manual Installation

#### 1. Backend Setup
```bash
# Create virtual environment
python -m venv cerebral-env
source cerebral-env/bin/activate  # Linux/Mac
# or
cerebral-env\Scripts\activate    # Windows

# Install backend dependencies
pip install -r backend/requirements.txt

# Install Ollama separately
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2-turbo

# Start backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup (Development)
```bash
# Install frontend dependencies
cd frontend
npm install

# Start development server
npm run dev
```

---

## Ollama Turbo Configuration

### 1. Using Your Turbo Membership
```bash
# Pull your turbo model
ollama pull your-turbo-model-name

# Verify model is available
ollama list

# Set model in .env
OLLAMA_MODEL=your-turbo-model-name
```

### 2. GPU Acceleration
```bash
# Check if Ollama detects GPU
ollama ps

# For NVIDIA GPUs, ensure drivers are installed
nvidia-smi

# Enable GPU in Docker (add to docker-compose.yml)
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 3. Model Configuration
Create a Modelfile for custom settings:
```bash
# Create Modelfile
cat > Modelfile << EOL
FROM your-turbo-model-name
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
EOL

# Create custom model
ollama create my-turbo-optimized -f Modelfile
```

---

## MCP Server Management

### 1. Default MCP Servers
The system includes support for:
- DeepSeek Coder (localhost:8080)
- Custom MCP servers
- Ollama integration

### 2. Adding Custom MCP Servers

#### Through UI:
1. Navigate to Settings → MCP Servers
2. Click "Add New Server"
3. Fill in configuration:
   - Name: Custom Server
   - Host: localhost or remote IP
   - Port: 8080 (or custom)
   - Model: Specific model name
   - Enabled: ✓

#### Through API:
```bash
curl -X POST "http://localhost:8000/api/mcp/servers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Custom Server",
    "host": "localhost",
    "port": 9090,
    "model": "custom-model",
    "enabled": true
  }'
```

### 3. Port Management
The system automatically manages ports for multiple MCP servers:
- Default range: 8080-8100
- Configurable in settings
- Port conflict detection and resolution

---

## Configuration Files

### 1. Docker Compose (docker-compose.yml)
```yaml
version: '3.8'

services:
  cerebral-studio:
    build: .
    ports:
      - "8000:8000"
      - "11434:11434"  # Ollama
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./data:/app/data
      - ollama_data:/root/.ollama
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

### 2. Backend Configuration (config.py)
```python
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = None
    ANTHROPIC_API_KEY: str = None
    GOOGLE_PROJECT: str = None
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2-turbo"
    OLLAMA_NUM_GPU: int = 1
    
    # MCP Settings
    MCP_SERVER_PORT_RANGE: tuple = (8080, 8100)
    
    class Config:
        env_file = ".env"
```

### 3. Frontend Configuration (vite.config.js)
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  }
})
```

---

## Usage Guide

### 1. Starting the System
```bash
# Production
docker-compose up -d

# Development
docker-compose -f docker-compose.dev.yml up

# Monitor logs
docker-compose logs -f
```

### 2. Accessing the Web Interface
1. Open http://localhost:8000
2. Configure your API keys in Settings
3. Select preferred providers and models
4. Start chatting!

### 3. Model Management
- Switch between providers in real-time
- Monitor token usage and costs
- Adjust model parameters on the fly
- Save and load conversation templates

### 4. Advanced Features
- **Semantic Memory**: Enable/disable in settings
- **Dynamic Routing**: Let the AI choose the best provider
- **Context Management**: Adjust token budgets and summarization
- **Multi-session Support**: Run multiple conversations simultaneously

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Not Starting
```bash
# Check Ollama status
docker-compose logs ollama

# Restart Ollama service
docker-compose restart ollama

# Pull model again if needed
docker-compose exec ollama ollama pull llama2-turbo
```

#### 2. GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Rebuild with GPU support
docker-compose build --no-cache

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### 3. Port Conflicts
```bash
# Check used ports
netstat -tulpn | grep :8080

# Change default ports in .env
MCP_PORT_RANGE=8100-8200
```

#### 4. API Key Issues
- Verify keys in .env file
- Check provider status pages
- Ensure proper billing setup

### Logs and Monitoring
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs cerebral-studio
docker-compose logs ollama

# Monitor resource usage
docker stats
```

---

## Performance Optimization

### 1. For GPU Systems
```bash
# Enable GPU acceleration
OLLAMA_NUM_GPU=1
CUDA_VISIBLE_DEVICES=0

# Use quantized models for better performance
ollama pull llama2-turbo:7b-q4_0
```

### 2. For CPU-Only Systems
```bash
# Use smaller models
OLLAMA_MODEL=llama2:7b

# Enable lite mode
LITE_MODE=true

# Reduce context size
MAX_CONTEXT_TOKENS=4096
```

### 3. Network Optimization
```bash
# Use CDN for models if available
OLLAMA_BASE_URL=https://cdn.ollama.ai

# Enable compression
GZIP_COMPRESSION=true
```

---

## Backup and Migration

### 1. Backup Data
```bash
# Backup conversations and settings
docker-compose exec cerebral-studio tar -czf backup.tar.gz /app/data

# Backup Ollama models
docker-compose exec ollama tar -czf models.tar.gz /root/.ollama
```

### 2. Restore Data
```bash
# Restore from backup
docker cp backup.tar.gz cerebral-studio:/app/
docker-compose exec cerebral-studio tar -xzf backup.tar.gz

# Restore models
docker cp models.tar.gz ollama:/root/
docker-compose exec ollama tar -xzf models.tar.gz
```

### 3. Migration Between Systems
1. Backup data and models
2. Copy .env file to new system
3. Restore backups
4. Update any host-specific settings

---

## Security Considerations

### 1. Environment Security
```bash
# Secure .env file
chmod 600 .env

# Use secrets management for production
docker secret create openai_key ./secrets/openai_key
```

### 2. Network Security
```bash
# Enable HTTPS
SSL_CERT_FILE=/path/to/cert.pem
SSL_KEY_FILE=/path/to/key.pem

# Restrict access
ALLOWED_HOSTS=yourdomain.com,localhost
```

### 3. API Security
- Rotate API keys regularly
- Use provider-specific rate limiting
- Monitor usage for anomalies

---

## Support and Resources

### 1. Documentation
- GitHub Repository: https://github.com/your-username/cerebral-studio
- API Documentation: http://localhost:8000/docs
- Swagger UI: http://localhost:8000/redoc

### 2. Community
- Discord Channel: [Link to Discord]
- GitHub Issues: For bug reports and feature requests
- Wiki: Comprehensive usage guides

### 3. Monitoring
```bash
# Health check endpoint
curl http://localhost:8000/health

# Metrics endpoint (Prometheus format)
curl http://localhost:8000/metrics
```

---

## Conclusion

Cerebral Studio provides a powerful, flexible platform for advanced AI interactions with neuromorphic capabilities. The Docker-based deployment makes it easy to get started, while the comprehensive configuration options allow for fine-tuning to specific needs.

---
Notes:
1. Start with the Docker deployment for easiest setup
2. Configure Ollama Turbo model for optimal performance
3. Monitor resource usage, especially when running local models
4. Regularly back up your conversations and settings