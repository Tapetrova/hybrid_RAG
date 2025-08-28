#!/usr/bin/env python3
"""Health check script for content scraper services."""

import sys
import requests
import redis
import psycopg2
from urllib.parse import quote
import os


def check_api(host="localhost", port=8099):
    """Check if API is responsive."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"API health check failed: {e}")
        return False


def check_redis(host="localhost", port=6379, password=None):
    """Check if Redis is accessible."""
    try:
        r = redis.Redis(host=host, port=port, password=password, decode_responses=True)
        return r.ping()
    except Exception as e:
        print(f"Redis health check failed: {e}")
        return False


def check_postgres(
    host="localhost",
    port=5432,
    dbname="content_scraper",
    user="postgres",
    password="password",
):
    """Check if PostgreSQL is accessible."""
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        )
        conn.close()
        return True
    except Exception as e:
        print(f"PostgreSQL health check failed: {e}")
        return False


def main():
    """Run all health checks."""
    # Get configuration from environment
    api_host = os.getenv("API_HOST", "localhost")
    api_port = int(os.getenv("API_PORT", "8099"))

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD_LLM")

    db_host = os.getenv("DATABASE_HOST", "localhost")
    db_port = int(os.getenv("DATABASE_PORT", "5432"))
    db_name = os.getenv("DATABASE_NAME", "content_scraper")
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_password = os.getenv("DATABASE_PASSWORD", "password")

    # Run checks
    checks = {
        "API": check_api(api_host, api_port),
        "Redis": check_redis(redis_host, redis_port, redis_password),
        "PostgreSQL": check_postgres(db_host, db_port, db_name, db_user, db_password),
    }

    # Print results
    print("Health Check Results:")
    print("-" * 30)
    all_healthy = True
    for service, status in checks.items():
        status_str = "✓ Healthy" if status else "✗ Unhealthy"
        print(f"{service:12} {status_str}")
        if not status:
            all_healthy = False

    print("-" * 30)

    # Exit with appropriate code
    sys.exit(0 if all_healthy else 1)


if __name__ == "__main__":
    main()
