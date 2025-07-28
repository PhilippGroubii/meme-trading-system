#!/bin/bash

# Setup CloudWatch monitoring
echo "Setting up CloudWatch monitoring..."

# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
    --dashboard-name "MemeTrading-Dashboard" \
    --dashboard-body file://cloudwatch-dashboard.json

# Create CloudWatch alarms
aws cloudwatch put-metric-alarm \
    --alarm-name "MemeTrading-HighCPU" \
    --alarm-description "High CPU utilization" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:us-east-1:ACCOUNT:meme-trading-alerts

# Setup log groups
aws logs create-log-group --log-group-name /aws/ec2/meme-trading
aws logs create-log-group --log-group-name /aws/rds/meme-trading

echo "Monitoring setup complete"