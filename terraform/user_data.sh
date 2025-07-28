#!/bin/bash
yum update -y
yum install -y docker git

# Start Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /opt/meme-trading
cd /opt/meme-trading

# Create environment file
cat > .env << ENVEOF
DATABASE_URL=postgresql://${db_username}:${db_password}@${db_host}:5432/${db_name}
ENVIRONMENT=staging
ENVEOF

echo "Server ready for deployment" > /tmp/setup-complete