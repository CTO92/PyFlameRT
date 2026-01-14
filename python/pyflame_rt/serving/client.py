"""
PyFlameRT Model Server Client SDK

Provides synchronous and asynchronous clients for interacting with
PyFlameRT model servers over HTTP.
"""

from __future__ import annotations

import json
import base64
import time
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Try to import async libraries (optional)
try:
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False


class ServerError(Exception):
    """Exception raised when server returns an error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Server error {status_code}: {message}")


@dataclass
class IOSpec:
    """Specification for model input or output."""
    name: str
    dtype: str
    shape: List[int]


@dataclass
class ModelMetadata:
    """Metadata about a model."""
    name: str
    version: str
    platform: str
    ready: bool
    inputs: List[IOSpec] = field(default_factory=list)
    outputs: List[IOSpec] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Basic model information."""
    name: str
    ready: bool
    versions: List[str] = field(default_factory=list)


@dataclass
class ModelStats:
    """Model statistics."""
    model: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


@dataclass
class InferenceRequest:
    """Request for model inference."""
    model: str
    inputs: Dict[str, np.ndarray]
    outputs: Optional[List[str]] = None
    request_id: Optional[str] = None
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to JSON-serializable dict."""
        result = {
            "inputs": {}
        }

        if self.request_id:
            result["request_id"] = self.request_id
        if self.outputs:
            result["outputs"] = self.outputs
        if self.priority != 0:
            result["priority"] = self.priority

        for name, tensor in self.inputs.items():
            result["inputs"][name] = {
                "shape": list(tensor.shape),
                "dtype": _numpy_dtype_to_string(tensor.dtype),
                "data": _encode_tensor_data(tensor)
            }

        return result


@dataclass
class InferenceResponse:
    """Response from model inference."""
    request_id: str
    model_name: str
    model_version: str
    outputs: Dict[str, np.ndarray]
    success: bool
    error_message: Optional[str] = None
    latency_us: int = 0
    queue_time_us: int = 0

    @property
    def latency_ms(self) -> float:
        """Get latency in milliseconds."""
        return self.latency_us / 1000.0

    @property
    def queue_time_ms(self) -> float:
        """Get queue time in milliseconds."""
        return self.queue_time_us / 1000.0


def _numpy_dtype_to_string(dtype: np.dtype) -> str:
    """Convert numpy dtype to string representation."""
    mapping = {
        np.float32: "float32",
        np.float64: "float64",
        np.float16: "float16",
        np.int32: "int32",
        np.int64: "int64",
        np.int16: "int16",
        np.int8: "int8",
        np.uint8: "uint8",
        np.bool_: "bool",
    }
    return mapping.get(dtype.type, "float32")


def _string_to_numpy_dtype(dtype_str: str) -> np.dtype:
    """Convert string to numpy dtype."""
    mapping = {
        "float32": np.float32,
        "float64": np.float64,
        "float16": np.float16,
        "bfloat16": np.float32,  # No native bfloat16 in numpy
        "int32": np.int32,
        "int64": np.int64,
        "int16": np.int16,
        "int8": np.int8,
        "uint8": np.uint8,
        "bool": np.bool_,
    }
    return np.dtype(mapping.get(dtype_str, np.float32))


def _encode_tensor_data(tensor: np.ndarray) -> str:
    """Encode tensor data as base64 string."""
    return base64.b64encode(tensor.tobytes()).decode("ascii")


def _decode_tensor_data(data: str, dtype: np.dtype, shape: List[int]) -> np.ndarray:
    """Decode base64 tensor data to numpy array."""
    raw_bytes = base64.b64decode(data)
    return np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)


class ModelClient:
    """
    Synchronous HTTP client for PyFlameRT model server.

    Example:
        client = ModelClient("http://localhost:8080")

        # Check health
        if client.is_ready():
            # Get model info
            models = client.list_models()

            # Run inference
            response = client.infer(
                model="resnet50",
                inputs={"input": image_tensor}
            )
            predictions = response.outputs["output"]
    """

    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 0.5
    ):
        """
        Initialize the client.

        Args:
            url: Base URL of the model server (e.g., "http://localhost:8080")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        retry: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request to server."""
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}

        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")

        last_error = None
        attempts = self.max_retries if retry else 1

        for attempt in range(attempts):
            try:
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers=headers,
                    method=method
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))

            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8")
                try:
                    error_json = json.loads(error_body)
                    message = error_json.get("error", error_body)
                except json.JSONDecodeError:
                    message = error_body
                raise ServerError(e.code, message)

            except urllib.error.URLError as e:
                last_error = e
                if attempt < attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise ConnectionError(f"Failed to connect to server: {last_error}")

    def is_alive(self) -> bool:
        """Check if server is alive (liveness probe)."""
        try:
            self._request("GET", "/health/live", retry=False)
            return True
        except Exception:
            return False

    def is_ready(self) -> bool:
        """Check if server is ready to accept requests (readiness probe)."""
        try:
            result = self._request("GET", "/health/ready", retry=False)
            return result.get("status") == "ready"
        except ServerError as e:
            return False
        except Exception:
            return False

    def wait_for_ready(self, timeout: float = 60.0, poll_interval: float = 1.0) -> bool:
        """
        Wait for server to become ready.

        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: Time between health checks

        Returns:
            True if server became ready, False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.is_ready():
                return True
            time.sleep(poll_interval)
        return False

    def list_models(self) -> List[ModelInfo]:
        """Get list of all models on the server."""
        result = self._request("GET", "/v1/models")
        models = []
        for m in result.get("models", []):
            models.append(ModelInfo(
                name=m["name"],
                ready=m.get("ready", False),
                versions=m.get("versions", [])
            ))
        return models

    def get_model_metadata(self, model: str) -> ModelMetadata:
        """Get metadata for a specific model."""
        result = self._request("GET", f"/v1/models/{model}")

        inputs = [
            IOSpec(name=i["name"], dtype=i["dtype"], shape=i["shape"])
            for i in result.get("inputs", [])
        ]
        outputs = [
            IOSpec(name=o["name"], dtype=o["dtype"], shape=o["shape"])
            for o in result.get("outputs", [])
        ]

        return ModelMetadata(
            name=result["name"],
            version=result.get("version", "1"),
            platform=result.get("platform", "pyflame_rt"),
            ready=result.get("ready", False),
            inputs=inputs,
            outputs=outputs
        )

    def get_model_stats(self, model: str) -> ModelStats:
        """Get statistics for a specific model."""
        result = self._request("GET", f"/v1/models/{model}/stats")
        return ModelStats(
            model=result["model"],
            total_requests=result.get("total_requests", 0),
            successful_requests=result.get("successful_requests", 0),
            failed_requests=result.get("failed_requests", 0),
            avg_latency_ms=result.get("avg_latency_ms", 0.0),
            p50_latency_ms=result.get("p50_latency_ms", 0.0),
            p95_latency_ms=result.get("p95_latency_ms", 0.0),
            p99_latency_ms=result.get("p99_latency_ms", 0.0)
        )

    def infer(
        self,
        model: str,
        inputs: Dict[str, np.ndarray],
        outputs: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        priority: int = 0
    ) -> InferenceResponse:
        """
        Run inference on a model.

        Args:
            model: Model name
            inputs: Dictionary mapping input names to numpy arrays
            outputs: Optional list of output names to return (None = all)
            request_id: Optional request ID for tracking
            priority: Request priority (higher = more important)

        Returns:
            InferenceResponse with output tensors

        Raises:
            ServerError: If server returns an error
            ConnectionError: If unable to connect to server
        """
        request = InferenceRequest(
            model=model,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
            priority=priority
        )

        result = self._request(
            "POST",
            f"/v1/models/{model}/infer",
            data=request.to_dict()
        )

        # Parse outputs
        output_tensors = {}
        for name, tensor_data in result.get("outputs", {}).items():
            dtype = _string_to_numpy_dtype(tensor_data["dtype"])
            shape = tensor_data["shape"]
            data = tensor_data["data"]
            output_tensors[name] = _decode_tensor_data(data, dtype, shape)

        return InferenceResponse(
            request_id=result.get("request_id", ""),
            model_name=result.get("model_name", model),
            model_version=result.get("model_version", "1"),
            outputs=output_tensors,
            success=result.get("success", True),
            error_message=result.get("error_message"),
            latency_us=result.get("latency_us", 0),
            queue_time_us=result.get("queue_time_us", 0)
        )

    def infer_batch(
        self,
        model: str,
        batch_inputs: List[Dict[str, np.ndarray]],
        max_workers: int = 4
    ) -> List[InferenceResponse]:
        """
        Run inference on multiple inputs in parallel.

        Args:
            model: Model name
            batch_inputs: List of input dictionaries
            max_workers: Maximum parallel requests

        Returns:
            List of InferenceResponse objects
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.infer, model, inputs)
                for inputs in batch_inputs
            ]
            return [f.result() for f in futures]


class AsyncModelClient:
    """
    Asynchronous HTTP client for PyFlameRT model server.

    Requires aiohttp: pip install aiohttp

    Example:
        async with AsyncModelClient("http://localhost:8080") as client:
            if await client.is_ready():
                response = await client.infer(
                    model="resnet50",
                    inputs={"input": image_tensor}
                )
    """

    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        max_connections: int = 100
    ):
        """
        Initialize the async client.

        Args:
            url: Base URL of the model server
            timeout: Request timeout in seconds
            max_connections: Maximum concurrent connections
        """
        if not ASYNC_AVAILABLE:
            raise ImportError(
                "aiohttp is required for AsyncModelClient. "
                "Install with: pip install aiohttp"
            )

        self.base_url = url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._connector = aiohttp.TCPConnector(limit=max_connections)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AsyncModelClient":
        """Enter async context manager."""
        self._session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=self._connector
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if self._session:
            await self._session.close()

    async def _ensure_session(self):
        """Ensure session is created."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=self._connector
            )

    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make async HTTP request."""
        await self._ensure_session()
        url = f"{self.base_url}{path}"

        async with self._session.request(
            method,
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        ) as response:
            body = await response.text()

            if response.status >= 400:
                try:
                    error_json = json.loads(body)
                    message = error_json.get("error", body)
                except json.JSONDecodeError:
                    message = body
                raise ServerError(response.status, message)

            return json.loads(body)

    async def is_alive(self) -> bool:
        """Check if server is alive."""
        try:
            await self._request("GET", "/health/live")
            return True
        except Exception:
            return False

    async def is_ready(self) -> bool:
        """Check if server is ready."""
        try:
            result = await self._request("GET", "/health/ready")
            return result.get("status") == "ready"
        except Exception:
            return False

    async def list_models(self) -> List[ModelInfo]:
        """Get list of all models."""
        result = await self._request("GET", "/v1/models")
        models = []
        for m in result.get("models", []):
            models.append(ModelInfo(
                name=m["name"],
                ready=m.get("ready", False),
                versions=m.get("versions", [])
            ))
        return models

    async def get_model_metadata(self, model: str) -> ModelMetadata:
        """Get metadata for a model."""
        result = await self._request("GET", f"/v1/models/{model}")

        inputs = [
            IOSpec(name=i["name"], dtype=i["dtype"], shape=i["shape"])
            for i in result.get("inputs", [])
        ]
        outputs = [
            IOSpec(name=o["name"], dtype=o["dtype"], shape=o["shape"])
            for o in result.get("outputs", [])
        ]

        return ModelMetadata(
            name=result["name"],
            version=result.get("version", "1"),
            platform=result.get("platform", "pyflame_rt"),
            ready=result.get("ready", False),
            inputs=inputs,
            outputs=outputs
        )

    async def infer(
        self,
        model: str,
        inputs: Dict[str, np.ndarray],
        outputs: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        priority: int = 0
    ) -> InferenceResponse:
        """Run inference asynchronously."""
        request = InferenceRequest(
            model=model,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
            priority=priority
        )

        result = await self._request(
            "POST",
            f"/v1/models/{model}/infer",
            data=request.to_dict()
        )

        # Parse outputs
        output_tensors = {}
        for name, tensor_data in result.get("outputs", {}).items():
            dtype = _string_to_numpy_dtype(tensor_data["dtype"])
            shape = tensor_data["shape"]
            data = tensor_data["data"]
            output_tensors[name] = _decode_tensor_data(data, dtype, shape)

        return InferenceResponse(
            request_id=result.get("request_id", ""),
            model_name=result.get("model_name", model),
            model_version=result.get("model_version", "1"),
            outputs=output_tensors,
            success=result.get("success", True),
            error_message=result.get("error_message"),
            latency_us=result.get("latency_us", 0),
            queue_time_us=result.get("queue_time_us", 0)
        )

    async def infer_batch(
        self,
        model: str,
        batch_inputs: List[Dict[str, np.ndarray]],
        max_concurrent: int = 10
    ) -> List[InferenceResponse]:
        """
        Run inference on multiple inputs concurrently.

        Args:
            model: Model name
            batch_inputs: List of input dictionaries
            max_concurrent: Maximum concurrent requests

        Returns:
            List of InferenceResponse objects
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_infer(inputs):
            async with semaphore:
                return await self.infer(model, inputs)

        tasks = [bounded_infer(inputs) for inputs in batch_inputs]
        return await asyncio.gather(*tasks)
