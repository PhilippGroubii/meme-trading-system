#!/bin/bash
set -e

echo "🚀 Starting deployment to AWS using Systems Manager..."

# Configuration
INSTANCE_IP="54.162.71.225"
APP_PATH="/opt/meme-trading"
GITHUB_REPO="git@github.com:PhilippGroubii/meme-trading-system.git"

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=ip-address,Values=$INSTANCE_IP" --query "Reservations[].Instances[].InstanceId" --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Could not find instance with IP $INSTANCE_IP"
    exit 1
fi

echo "📡 Found instance: $INSTANCE_ID"

# Push to GitHub first
echo "📤 Pushing to GitHub..."
git add .
git commit -m "Update: $(date)" || echo "No changes to commit"
git push origin main

# Create deployment commands
COMMANDS=$(cat << 'EOF'
cd /opt
if [ ! -d "meme-trading" ]; then
    sudo mkdir -p meme-trading
    sudo chown ubuntu:ubuntu meme-trading
fi
cd meme-trading

# Clone or pull latest code
if [ ! -d ".git" ]; then
    git clone https://github.com/PhilippGroubii/meme-trading-system.git .
else
    git pull origin main
fi

# Set up virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Set environment
export ENVIRONMENT=production

# Start the web server (simple approach)
pkill -f "python.*web_server.py" || true
nohup python web_server.py > app.log 2>&1 &

echo "Deployment completed"
ps aux | grep web_server
EOF
)

# Execute commands on EC2 via Systems Manager
echo "🔧 Executing deployment commands..."
aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"$COMMANDS\"]" \
    --output text

echo "🎉 Deployment initiated!"
echo "🌐 Your app should be available at: http://$INSTANCE_IP"
echo "📊 Check status at: http://$INSTANCE_IP/health"
