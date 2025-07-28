#!/bin/bash

set -e

ENVIRONMENT=${1:-staging}
REGION=${2:-us-east-1}

echo "Deploying to $ENVIRONMENT environment in $REGION..."

# Export environment variables
export AWS_DEFAULT_REGION=$REGION
export ENVIRONMENT=$ENVIRONMENT

# Initialize Terraform
cd terraform
terraform init

# Plan deployment
terraform plan -var="environment=$ENVIRONMENT" -out=tfplan

# Apply if approved
read -p "Do you want to apply this plan? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    terraform apply tfplan
    
    # Get outputs
    DB_HOST=$(terraform output -raw db_host)
    DB_NAME=$(terraform output -raw db_name)
    
    # Export database variables
    export DB_HOST=$DB_HOST
    export DB_NAME=$DB_NAME
    export DB_USERNAME=$DB_USERNAME
    export DB_PASSWORD=$DB_PASSWORD
    
    # Run database migrations
    cd ../scripts
    python migrate-db.py
    
    # Deploy application
    cd ..
    docker-compose -f docker-compose.prod.yml up -d
    
    echo "Deployment completed successfully!"
    echo "Application URL: https://memetradingpro.com"
    echo "Health check: https://memetradingpro.com/health"
else
    echo "Deployment cancelled"
fi