#!/bin/sh

# Start the application
uvicorn apps.agent_flow.src.main:app --host 0.0.0.0 --port ${AGENT_FLOW_PORT} --workers ${WORKERS}
