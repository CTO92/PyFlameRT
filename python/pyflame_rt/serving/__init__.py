"""
PyFlameRT Serving - Client SDK for model inference serving.

This module provides clients for interacting with PyFlameRT model servers:

- ModelClient: Synchronous HTTP client
- AsyncModelClient: Asynchronous HTTP client
- InferenceRequest: Request builder for inference
- InferenceResponse: Response from inference

Example usage:

    from pyflame_rt.serving import ModelClient

    client = ModelClient("http://localhost:8080")

    # Check server health
    if client.is_ready():
        # Run inference
        response = client.infer(
            model="my_model",
            inputs={"input": np.array([1, 2, 3], dtype=np.float32)}
        )
        print(response.outputs["output"])
"""

from .client import (
    ModelClient,
    AsyncModelClient,
    InferenceRequest,
    InferenceResponse,
    ModelMetadata,
    ModelInfo,
    ServerError,
)

__all__ = [
    "ModelClient",
    "AsyncModelClient",
    "InferenceRequest",
    "InferenceResponse",
    "ModelMetadata",
    "ModelInfo",
    "ServerError",
]
