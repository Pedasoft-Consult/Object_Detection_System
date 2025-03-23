import time
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

# Define metrics
HTTP_REQUEST_COUNTER = Counter(
    'http_requests_total',
    'Total count of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint']
)

HTTP_REQUEST_ERRORS = Counter(
    'http_request_errors_total',
    'Total count of HTTP request errors',
    ['method', 'endpoint', 'exception']
)

MODEL_INFERENCE_TIME = Summary(
    'model_inference_time_seconds',
    'Time spent on model inference',
    ['model_name']
)

OBJECT_DETECTIONS = Counter(
    'object_detections_total',
    'Total count of objects detected',
    ['class_name']  # Changed from 'class' to 'class_name' to avoid Python reserved keyword
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to capture HTTP request metrics for Prometheus"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        endpoint = request.url.path

        # Skip metrics endpoint to avoid infinite recursion
        if endpoint == "/metrics":
            return await call_next(request)

        # Track request in progress
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()

        # Track request duration
        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code

            # Update metrics
            HTTP_REQUEST_COUNTER.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()

            return response

        except Exception as e:
            # Track exceptions
            HTTP_REQUEST_ERRORS.labels(
                method=method,
                endpoint=endpoint,
                exception=type(e).__name__
            ).inc()
            raise

        finally:
            # Request duration and in-progress tracking
            HTTP_REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(time.time() - start_time)

            HTTP_REQUESTS_IN_PROGRESS.labels(
                method=method,
                endpoint=endpoint
            ).dec()


def track_inference_time(func):
    """Decorator to track model inference time"""

    def wrapper(*args, **kwargs):
        # Start timer
        start_time = time.time()

        # Call the original function
        result = func(*args, **kwargs)

        # Record time
        MODEL_INFERENCE_TIME.labels(
            model_name="YOLOv5"  # Or dynamically extract model name
        ).observe(time.time() - start_time)

        return result

    return wrapper


def track_detections(detections, class_names):
    """Track detected objects by class"""
    for detection in detections:
        class_name = detection.get('class_name', 'unknown')
        OBJECT_DETECTIONS.labels(class_name=class_name).inc()  # Using class_name parameter instead of class


def setup_monitoring(app: FastAPI):
    """Set up monitoring for the FastAPI application"""

    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)

    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Add health endpoint (for Prometheus scraping)
    @app.get("/health")
    async def health():
        return {"status": "healthy"}