output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "db_endpoint" {
  description = "Database endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "app_instance_ip" {
  description = "Application instance public IP"
  value       = aws_instance.app.public_ip
}

output "app_instance_dns" {
  description = "Application instance public DNS"
  value       = aws_instance.app.public_dns
}