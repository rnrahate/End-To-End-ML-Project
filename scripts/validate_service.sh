#!/bin/bash
# Validate the service is running
if curl -f http://localhost; then
    echo "Service is running"
else
    echo "Service failed to start"
    exit 1
fi