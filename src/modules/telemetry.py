"""Rich telemetry, tracing, and metrics export helpers.

This module centralizes observability for the Resonance Transformer stack. It
provides:

* Structured JSON logging with sampling and field redaction
* Per-operation tracing context managers that capture runtime metadata
* Optional Prometheus / OpenTelemetry exporters (best-effort, optional deps)
* Rolling latency quantiles (p50/p95/p99) per op/device signature
* Simple metric counters/gauges for success/failure tracking
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, MutableMapping, Optional


try:  # Optional Prometheus support
    from prometheus_client import Histogram, Counter, Gauge, CollectorRegistry, start_http_server

    PROM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PROM_AVAILABLE = False
    Histogram = Counter = Gauge = CollectorRegistry = None  # type: ignore
    start_http_server = None  # type: ignore


try:  # Optional OpenTelemetry support
    from opentelemetry import trace as otel_trace  # type: ignore
    from opentelemetry.sdk.resources import Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import (  # type: ignore
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    OTEL_AVAILABLE = False
    otel_trace = Resource = TracerProvider = BatchSpanProcessor = ConsoleSpanExporter = None  # type: ignore


def _default_redactions() -> List[str]:
    return ["token", "secret", "password", "key", "credential"]


@dataclass
class TelemetryConfig:
    """Configuration for telemetry writer."""

    enabled: bool = True
    output_path: Optional[str] = None
    structured_logging: bool = True
    log_to_stdout: bool = True
    log_level: int = logging.INFO
    sample_rate: float = 1.0
    latency_sample_size: int = 512
    redact_fields: Iterable[str] = field(default_factory=_default_redactions)
    max_field_length: int = 4096
    service_name: str = "resonance-transformer"
    service_version: Optional[str] = None
    default_device: Optional[str] = None
    environment: Optional[str] = None
    prometheus_enabled: bool = False
    prometheus_port: Optional[int] = None
    prometheus_addr: str = "0.0.0.0"
    prometheus_namespace: str = "resonance_transformer"
    opentelemetry_enabled: bool = False
    otel_endpoint: Optional[str] = None
    otel_exporter: str = "console"
    redact_values_with: str = "<redacted>"
    random_seed: Optional[int] = None


class _PrometheusExporter:
    """Thin wrapper that owns Prometheus metric objects."""

    def __init__(self, config: TelemetryConfig):
        if not PROM_AVAILABLE:
            raise RuntimeError("prometheus_client is not available")

        self.registry = CollectorRegistry()
        namespace = config.prometheus_namespace
        label_names = ("operation", "device")
        self.latency = Histogram(
            "op_latency_seconds",
            "Latency per operation",
            labelnames=label_names,
            namespace=namespace,
            registry=self.registry,
        )
        self.success = Counter(
            "op_success_total",
            "Successful operation count",
            labelnames=label_names,
            namespace=namespace,
            registry=self.registry,
        )
        self.failures = Counter(
            "op_failure_total",
            "Failed operation count",
            labelnames=label_names,
            namespace=namespace,
            registry=self.registry,
        )
        self.retries = Counter(
            "op_retries_total",
            "Retry count per operation",
            labelnames=label_names,
            namespace=namespace,
            registry=self.registry,
        )
        self.device_headroom = Gauge(
            "device_headroom",
            "Reported device headroom (0-1)",
            labelnames=("device",),
            namespace=namespace,
            registry=self.registry,
        )

        if config.prometheus_port is not None and start_http_server is not None:
            start_http_server(config.prometheus_port, addr=config.prometheus_addr, registry=self.registry)

    def observe(self, op: str, device: str, duration_s: float, success: bool, retries: int) -> None:
        labels = {"operation": op, "device": device or "unknown"}
        self.latency.labels(**labels).observe(max(duration_s, 0.0))
        counter = self.success if success else self.failures
        counter.labels(**labels).inc()
        if retries:
            self.retries.labels(**labels).inc(retries)

    def report_headroom(self, device: str, headroom: float) -> None:
        self.device_headroom.labels(device=device).set(max(min(headroom, 1.0), 0.0))


class _OpenTelemetryBridge:
    """Simple OpenTelemetry span factory (console exporter by default)."""

    def __init__(self, config: TelemetryConfig):
        if not OTEL_AVAILABLE:
            raise RuntimeError("opentelemetry SDK not available")

        resource = Resource.create({"service.name": config.service_name, "service.version": config.service_version or "dev"})
        provider = TracerProvider(resource=resource)

        if config.otel_exporter == "console" or config.otel_endpoint is None:
            exporter = ConsoleSpanExporter()
        else:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("OTLP exporter requested but not available") from exc

            exporter = OTLPSpanExporter(endpoint=config.otel_endpoint)

        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        otel_trace.set_tracer_provider(provider)
        self._tracer = otel_trace.get_tracer(config.service_name)

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        return self._tracer.start_as_current_span(name, attributes=attributes or {})


class _NullOperationTrace(contextlib.AbstractContextManager):
    def record_retry(self, count: int = 1) -> None:  # pragma: no cover - trivial
        return None

    def __exit__(self, exc_type, exc, exc_tb) -> bool:  # pragma: no cover - trivial
        return False


class OperationTrace(contextlib.AbstractContextManager):
    """Context manager returned by TelemetryWriter.trace_operation."""

    def __init__(self, writer: "TelemetryWriter", op_name: str, device: Optional[str], metadata: Optional[Dict[str, Any]]):
        self.writer = writer
        self.op_name = op_name
        self.device = device
        self.metadata = metadata or {}
        self.retries = 0
        self.start_time = time.perf_counter()
        self.event_id = str(uuid.uuid4())
        self._otel_ctx = None
        if self.writer._otel_bridge is not None:
            self._otel_ctx = self.writer._otel_bridge.start_span(
                op_name,
                attributes={**self.metadata, "device": device or writer.config.default_device or "unknown"},
            )

    def record_retry(self, count: int = 1) -> None:
        self.retries += max(0, int(count))

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        duration_s = time.perf_counter() - self.start_time
        success = exc_type is None
        if self._otel_ctx is not None:
            # Context manager ensures span closure
            self._otel_ctx.__exit__(exc_type, exc, exc_tb)

        self.writer._finalize_operation(
            op_name=self.op_name,
            duration_s=duration_s,
            success=success,
            device=self.device,
            retries=self.retries,
            event_id=self.event_id,
            metadata=self.metadata,
            exception=exc,
        )
        return False  # Do not suppress exceptions


class TelemetryWriter:
    """Emits structured telemetry, metrics, and traces."""

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig()
        self.enabled = bool(self.config.enabled)
        self._rng = random.Random(self.config.random_seed)
        self._logger = logging.getLogger("resonance.telemetry")
        self._logger.setLevel(self.config.log_level)
        if not self._logger.handlers and self.config.log_to_stdout:
            handler = logging.StreamHandler()
            handler.setLevel(self.config.log_level)
            fmt = logging.Formatter("%(message)s")
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)

        self._prometheus: Optional[_PrometheusExporter] = None
        if self.config.prometheus_enabled:
            try:
                self._prometheus = _PrometheusExporter(self.config)
            except Exception as exc:  # pragma: no cover - best-effort init
                self._logger.warning("Failed to initialize Prometheus exporter: %s", exc)

        self._otel_bridge: Optional[_OpenTelemetryBridge] = None
        if self.config.opentelemetry_enabled:
            try:
                self._otel_bridge = _OpenTelemetryBridge(self.config)
            except Exception as exc:  # pragma: no cover - best-effort init
                self._logger.warning("Failed to initialize OpenTelemetry bridge: %s", exc)

        self._op_latency: MutableMapping[str, deque] = defaultdict(lambda: deque(maxlen=self.config.latency_sample_size))
        self._lock = threading.Lock()

        self._output_path = self.config.output_path
        if self._output_path:
            output_dir = os.path.dirname(self._output_path) or "."
            os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def emit(self, data: Dict[str, Any], event_type: str = "event", severity: str = "INFO") -> None:
        """Emit a structured telemetry event."""

        if not self.enabled or not self._should_sample():
            return

        payload = self._prepare_payload(data, event_type=event_type, severity=severity)
        self._write(payload)

    def trace_operation(
        self,
        op_name: str,
        *,
        device: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> contextlib.AbstractContextManager:
        """Return a context manager for per-operation tracing."""

        if not self.enabled:
            return _NullOperationTrace()

        device_resolved = device or self.config.default_device or "unknown"
        meta = metadata or {}
        return OperationTrace(self, op_name, device_resolved, meta)

    def observe_device_headroom(self, device: str, headroom: float) -> None:
        if self._prometheus is not None:
            self._prometheus.report_headroom(device, headroom)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _should_sample(self) -> bool:
        if self.config.sample_rate >= 1.0:
            return True
        if self.config.sample_rate <= 0.0:
            return False
        return self._rng.random() <= self.config.sample_rate

    def _prepare_payload(self, data: Dict[str, Any], *, event_type: str, severity: str) -> Dict[str, Any]:
        timestamp = time.time()
        base = {
            "event_id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "event_type": event_type,
            "service": self.config.service_name,
            "version": self.config.service_version or "dev",
            "severity": severity,
            "environment": self.config.environment,
        }
        sanitized = self._sanitize_dict(data)
        base.update(sanitized)
        return base

    def _write(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, default=str, sort_keys=False)
        if self.config.structured_logging and self.config.log_to_stdout:
            self._logger.info(line)
        if self._output_path:
            with open(self._output_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        redact_fields = set(self.config.redact_fields)
        for key, value in data.items():
            if key in redact_fields:
                sanitized[key] = self.config.redact_values_with
                continue
            if isinstance(value, str) and len(value) > self.config.max_field_length:
                sanitized[key] = value[: self.config.max_field_length] + "…"
            else:
                sanitized[key] = value
        return sanitized

    def _finalize_operation(
        self,
        *,
        op_name: str,
        duration_s: float,
        success: bool,
        device: Optional[str],
        retries: int,
        event_id: str,
        metadata: Dict[str, Any],
        exception: Optional[BaseException],
    ) -> None:
        label = self._signature_key(op_name, device)
        with self._lock:
            bucket = self._op_latency[label]
            bucket.append(duration_s)
            quantiles = self._compute_quantiles(list(bucket))

        payload = {
            "event_id": event_id,
            "event_type": "operation",
            "op": op_name,
            "device": device,
            "success": success,
            "duration_ms": duration_s * 1000.0,
            "p50_ms": quantiles["p50"] * 1000.0,
            "p95_ms": quantiles["p95"] * 1000.0,
            "p99_ms": quantiles["p99"] * 1000.0,
            "retries": retries,
            **metadata,
        }
        if exception is not None:
            payload["error"] = type(exception).__name__
            payload["error_message"] = str(exception)

        self.emit(payload, event_type="operation", severity="ERROR" if not success else "INFO")

        if self._prometheus is not None:
            self._prometheus.observe(op_name, device or "unknown", duration_s, success, retries)

    def _signature_key(self, op_name: str, device: Optional[str]) -> str:
        return f"{op_name}:{device or self.config.default_device or 'unknown'}"

    @staticmethod
    def _compute_quantiles(samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        ordered = sorted(samples)

        def percentile(p: float) -> float:
            if len(ordered) == 1:
                return ordered[0]
            k = (len(ordered) - 1) * p
            f = int(k)
            c = min(f + 1, len(ordered) - 1)
            if f == c:
                return ordered[int(k)]
            d0 = ordered[f] * (c - k)
            d1 = ordered[c] * (k - f)
            return d0 + d1

        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }


__all__ = [
    "TelemetryConfig",
    "TelemetryWriter",
    "OperationTrace",
]
