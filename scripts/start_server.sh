#!/bin/bash
# Run the Docker container
docker run -d --name ml-project -p 80:5000 $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG