#!/usr/bin/env bash
# Manage the pgvector container for the RAG pipeline.
#
# Usage:
#   ./pgvector.sh start    — start the container (idempotent)
#   ./pgvector.sh stop     — stop the container
#   ./pgvector.sh status   — show running status
#   ./pgvector.sh logs     — tail container logs

set -euo pipefail

CONTAINER_NAME="agentomics_pgvector"
POSTGRES_USER="rag"
POSTGRES_PASSWORD="rag"
POSTGRES_DB="rag"
POSTGRES_PORT=54320

cmd="${1:-status}"

case "$cmd" in
  start)
    if docker ps --filter "name=^${CONTAINER_NAME}$" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
      echo "Container '$CONTAINER_NAME' is already running."
    else
      echo "Starting pgvector container..."
      docker run -d \
        --name "$CONTAINER_NAME" \
        -e POSTGRES_USER="$POSTGRES_USER" \
        -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        -e POSTGRES_DB="$POSTGRES_DB" \
        -p ${POSTGRES_PORT}:5432 \
        pgvector/pgvector:pg17
      echo "Waiting for PostgreSQL to be ready..."
      until docker exec "$CONTAINER_NAME" pg_isready -U "$POSTGRES_USER" -q; do
        sleep 1
      done
      echo "pgvector is ready on port ${POSTGRES_PORT}."
    fi
    ;;

  stop)
    echo "Stopping and container '$CONTAINER_NAME'..."
    docker stop "$CONTAINER_NAME"
    echo "Done."
    ;;

  remove)
    echo "Removing container '$CONTAINER_NAME'..."
    docker rm "$CONTAINER_NAME"
    echo "Done."
    ;;

  status)
    docker ps --filter "name=^${CONTAINER_NAME}$" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    ;;

  logs)
    docker logs -f "$CONTAINER_NAME"
    ;;

  *)
    echo "Usage: $0 {start|stop|status|logs}"
    exit 1
    ;;
esac
