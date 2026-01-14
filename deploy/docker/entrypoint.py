#!/usr/bin/env python3
"""
PyFlameRT Model Server Entry Point

Starts the PyFlameRT model server with configuration from environment variables.
"""

import os
import sys
import signal
import logging
from pathlib import Path

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pyflame-rt")


def get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def get_env_bool(name: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.environ.get(name, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def main():
    """Main entry point."""
    # Import PyFlameRT
    try:
        from pyflame_rt import serving
    except ImportError as e:
        logger.error(f"Failed to import pyflame_rt: {e}")
        sys.exit(1)

    # Read configuration from environment
    http_host = os.environ.get("HTTP_HOST", "0.0.0.0")
    http_port = get_env_int("HTTP_PORT", 8080)
    http_workers = get_env_int("HTTP_WORKERS", 4)
    request_timeout = get_env_int("HTTP_REQUEST_TIMEOUT_MS", 30000)

    grpc_host = os.environ.get("GRPC_HOST", "0.0.0.0")
    grpc_port = get_env_int("GRPC_PORT", 9090)

    metrics_enabled = get_env_bool("METRICS_ENABLED", True)
    metrics_port = get_env_int("METRICS_PORT", 9091)

    model_dir = os.environ.get("MODEL_DIR", "/models")
    max_memory_mb = get_env_int("MAX_MEMORY_MB", 8192)

    enable_batching = get_env_bool("ENABLE_BATCHING", True)
    max_batch_size = get_env_int("MAX_BATCH_SIZE", 32)
    batch_timeout_us = get_env_int("BATCH_TIMEOUT_US", 5000)

    # Log configuration
    logger.info("PyFlameRT Model Server starting...")
    logger.info(f"HTTP: {http_host}:{http_port} (workers: {http_workers})")
    logger.info(f"gRPC: {grpc_host}:{grpc_port}")
    logger.info(f"Metrics: {'enabled' if metrics_enabled else 'disabled'} (port: {metrics_port})")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Max memory: {max_memory_mb} MB")
    logger.info(f"Batching: {'enabled' if enable_batching else 'disabled'} (max: {max_batch_size}, timeout: {batch_timeout_us}us)")

    # Build server configuration
    config = serving.ServerConfig()

    # HTTP config
    config.http.host = http_host
    config.http.port = http_port
    config.http.num_workers = http_workers
    config.http.request_timeout_ms = request_timeout

    # gRPC config
    config.grpc.host = grpc_host
    config.grpc.port = grpc_port

    # Other config
    config.model_dir = model_dir
    config.enable_metrics = metrics_enabled
    config.metrics_port = metrics_port
    config.max_memory = max_memory_mb * 1024 * 1024

    # Load models from model directory
    model_path = Path(model_dir)
    if model_path.exists():
        for model_file in model_path.glob("**/*.pfm"):
            model_name = model_file.stem
            model_config = serving.ModelConfig()
            model_config.name = model_name
            model_config.model_path = str(model_file)
            model_config.version = "1"
            model_config.enable_batching = enable_batching
            model_config.max_batch_size = max_batch_size
            model_config.batch_timeout_us = batch_timeout_us
            config.models.append(model_config)
            logger.info(f"Found model: {model_name} at {model_file}")

        # Also check for ONNX models
        for model_file in model_path.glob("**/*.onnx"):
            model_name = model_file.stem
            model_config = serving.ModelConfig()
            model_config.name = model_name
            model_config.model_path = str(model_file)
            model_config.version = "1"
            model_config.enable_batching = enable_batching
            model_config.max_batch_size = max_batch_size
            model_config.batch_timeout_us = batch_timeout_us
            config.models.append(model_config)
            logger.info(f"Found model: {model_name} at {model_file}")
    else:
        logger.warning(f"Model directory does not exist: {model_dir}")

    if not config.models:
        logger.warning("No models found to load")

    # Create and start server
    server = serving.ModelServer(config)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Setup callbacks
    def on_ready():
        logger.info("Server is ready to accept requests")

    def on_error(error):
        logger.error(f"Server error: {error}")

    server.on_ready(on_ready)
    server.on_error(on_error)

    # Start server
    try:
        logger.info("Starting server...")
        server.start()
        logger.info(f"Server started on port {server.http_port()}")

        # Wait for server to stop
        server.wait()

    except Exception as e:
        logger.exception(f"Server failed: {e}")
        sys.exit(1)

    logger.info("Server stopped")


if __name__ == "__main__":
    main()
