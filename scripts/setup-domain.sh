#!/bin/bash

DOMAIN="memetradingpro.com"
EMAIL="admin@memetradingpro.com"

# Register domain via Route53 (if not already registered)
echo "Setting up Route53 hosted zone..."
aws route53 create-hosted-zone \
    --name $DOMAIN \
    --caller-reference $(date +%s) \
    --hosted-zone-config Comment="Meme Trading System"

# Get hosted zone ID
HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name \
    --dns-name $DOMAIN \
    --query 'HostedZones[0].Id' \
    --output text | cut -d'/' -f3)

echo "Hosted Zone ID: $HOSTED_ZONE_ID"

# Create SSL certificate
echo "Requesting SSL certificate..."
CERT_ARN=$(aws acm request-certificate \
    --domain-name $DOMAIN \
    --domain-name "*.$DOMAIN" \
    --validation-method DNS \
    --query 'CertificateArn' \
    --output text)

echo "Certificate ARN: $CERT_ARN"

# Wait for certificate validation records
echo "Waiting for DNS validation records..."
sleep 30

# Get validation records
aws acm describe-certificate \
    --certificate-arn $CERT_ARN \
    --query 'Certificate.DomainValidationOptions[*].[DomainName,ResourceRecord.Name,ResourceRecord.Value]' \
    --output table

echo "Add the DNS validation records to your domain's DNS settings"
echo "Certificate will be validated automatically once DNS records are added"