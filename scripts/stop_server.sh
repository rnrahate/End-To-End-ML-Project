#!/bin/bash
# Stop the running container if it exists
docker stop ml-project || true
docker rm ml-project || true