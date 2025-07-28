"""
Cloud Deployment Configuration
Deploy your trading system to AWS/GCP/Azure for 24/7 operation
"""

import os
from pathlib import Path

def create_dockerfile():
    """Create Dockerfile for containerized deployment"""
    
    dockerfile_content = """# Meme Coin Trading System - Production Container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

# Expose port for monitoring dashboard
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Default command
CMD ["python", "trading_main.py"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile created")

def create_docker_compose():
    """Create docker-compose for local development and testing"""
    
    compose_content = """version: '3.8'

services:
  trading-system:
    build: .
    container_name: meme-trader
    restart: unless-stopped
    environment:
      - PAPER_TRADING=false
      - LOG_LEVEL=INFO
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./backups:/app/backups
    ports:
      - "8080:8080"
    networks:
      - trading-network
    
  redis:
    image: redis:alpine
    container_name: trading-redis
    restart: unless-stopped
    volumes:
      - redis-data:/data
    networks:
      - trading-network
    
  postgres:
    image: postgres:13
    container_name: trading-db
    restart: unless-stopped
    environment:
      - POSTGRES_DB=trading
      - POSTGRES_USER=trader
      - POSTGRES_PASSWORD=secure_password_here
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - trading-network
    
  monitoring:
    image: grafana/grafana:latest
    container_name: trading-monitor
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - trading-network

volumes:
  redis-data:
  postgres-data:
  grafana-data:

networks:
  trading-network:
    driver: bridge
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(compose_content)
    
    print("✅ docker-compose.yml created")

def create_kubernetes_deployment():
    """Create Kubernetes deployment for production scaling"""
    
    k8s_content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: meme-trading-system
  labels:
    app: meme-trader
spec:
  replicas: 2
  selector:
    matchLabels:
      app: meme-trader
  template:
    metadata:
      labels:
        app: meme-trader
    spec:
      containers:
      - name: trading-system
        image: your-registry/meme-trader:latest
        ports:
        - containerPort: 8080
        env:
        - name: PAPER_TRADING
          value: "false"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: trading-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: trading-service
spec:
  selector:
    app: meme-trader
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
"""
    
    with open('k8s-deployment.yaml', 'w') as f:
        f.write(k8s_content)
    
    print("✅ Kubernetes deployment created")

def create_terraform_aws():
    """Create Terraform configuration for AWS deployment"""
    
    terraform_content = """# AWS Infrastructure for Meme Trading System
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

# VPC
resource "aws_vpc" "trading_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "trading-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "trading_igw" {
  vpc_id = aws_vpc.trading_vpc.id
  
  tags = {
    Name = "trading-igw"
  }
}

# Public Subnet
resource "aws_subnet" "trading_public" {
  vpc_id                  = aws_vpc.trading_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true
  
  tags = {
    Name = "trading-public-subnet"
  }
}

# Route Table
resource "aws_route_table" "trading_public_rt" {
  vpc_id = aws_vpc.trading_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.trading_igw.id
  }
  
  tags = {
    Name = "trading-public-rt"
  }
}

resource "aws_route_table_association" "trading_public_rta" {
  subnet_id      = aws_subnet.trading_public.id
  route_table_id = aws_route_table.trading_public_rt.id
}

# Security Group
resource "aws_security_group" "trading_sg" {
  name        = "trading-security-group"
  description = "Security group for trading system"
  vpc_id      = aws_vpc.trading_vpc.id
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "trading-sg"
  }
}

# EC2 Instance
resource "aws_instance" "trading_server" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = "t3.medium"
  key_name              = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.trading_sg.id]
  subnet_id             = aws_subnet.trading_public.id
  
  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install -y docker
    systemctl start docker
    systemctl enable docker
    usermod -a -G docker ec2-user
    
    # Install docker-compose
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    
    # Clone and start trading system
    # (Add your deployment commands here)
  EOF
  
  tags = {
    Name = "trading-server"
    Environment = var.environment
  }
}

# RDS Database
resource "aws_db_instance" "trading_db" {
  identifier = "trading-database"
  
  engine            = "postgres"
  engine_version    = "13.7"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  
  db_name  = "trading"
  username = "trader"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.trading_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.trading_db_subnet_group.name
  
  skip_final_snapshot = true
  
  tags = {
    Name = "trading-db"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

# Outputs
output "instance_public_ip" {
  description = "Public IP of the trading server"
  value       = aws_instance.trading_server.public_ip
}

output "database_endpoint" {
  description = "Database endpoint"
  value       = aws_db_instance.trading_db.endpoint
  sensitive   = true
}
"""
    
    with open('main.tf', 'w') as f:
        f.write(terraform_content)
    
    print("✅ Terraform AWS configuration created")

def create_deployment_scripts():
    """Create deployment automation scripts"""
    
    # Production deployment script
    deploy_script = """#!/bin/bash
# Production Deployment Script for Meme Trading System

set -e

echo "🚀 Starting production deployment..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -t meme-trader:latest .

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

# Push to registry
echo "📤 Pushing to container registry..."
docker tag meme-trader:latest your-registry/meme-trader:latest
docker push your-registry/meme-trader:latest

# Deploy to production
echo "🌟 Deploying to production..."
kubectl apply -f k8s-deployment.yaml

# Wait for rollout
echo "⏳ Waiting for deployment..."
kubectl rollout status deployment/meme-trading-system

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -l app=meme-trader

echo "🎉 Deployment complete!"
echo "🔗 Access your trading system at: http://your-domain.com"
"""
    
    with open('deploy.sh', 'w') as f:
        f.write(deploy_script)
    
    os.chmod('deploy.sh', 0o755)
    print("✅ Deployment script created")

if __name__ == "__main__":
    print("🏗️ Creating cloud deployment configurations...")
    
    create_dockerfile()
    create_docker_compose()
    create_kubernetes_deployment()
    create_terraform_aws()
    create_deployment_scripts()
    
    print("\n✅ Cloud deployment setup complete!")
    print("\n🎯 Next steps:")
    print("1. docker-compose up -d  # Test locally")
    print("2. terraform init && terraform apply  # Deploy to AWS")
    print("3. ./deploy.sh  # Deploy to Kubernetes")