#!/bin/bash
# Install Docker if not present (for Amazon Linux 2)
if ! command -v docker &> /dev/null; then
    sudo yum update -y
    sudo amazon-linux-extras install docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user
fi
# Login to ECR
aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
# Make scripts executable
chmod +x scripts/*.sh