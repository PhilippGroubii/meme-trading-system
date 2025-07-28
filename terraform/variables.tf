variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "staging"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "meme-trading"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "memetradingdb"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "memetrader"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
  default     = "MemeTrading2024!"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "groubiiconsulting.com"
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