"""Tests for stress test Pydantic models."""

from warpt.models.stress_models import (
    CPUSystemResult,
    MixedPrecisionResults,
    PrecisionResult,
    ThrottleEvent,
)


def test_throttle_event_validation():
    """Test ThrottleEvent model validation."""
    event = ThrottleEvent(
        timestamp=1600000000.0,
        device_id="gpu_0",
        reasons=["thermal", "power_limit"],
    )
    assert event.device_id == "gpu_0"
    assert "thermal" in event.reasons


def test_cpu_system_result_validation():
    """Test CPUSystemResult model validation."""
    result = CPUSystemResult(
        cpu_model="Test CPU",
        cpu_architecture="x86_64",
        tflops=1.5,
        duration=30.0,
        total_operations=1000000,
        burnin_seconds=5,
        sockets_used=1,
        physical_cores=8,
        logical_cores=16,
    )
    assert result.cpu_model == "Test CPU"
    assert result.tflops == 1.5


def test_mixed_precision_results_validation():
    """Test MixedPrecisionResults model validation."""
    fp32_res = PrecisionResult(
        supported=True,
        dtype="float32",
        tflops=10.0,
        matrix_size=1024,
    )
    fp16_res = PrecisionResult(
        supported=True,
        dtype="float16",
        tflops=20.0,
        matrix_size=1024,
    )

    mixed = MixedPrecisionResults(
        fp32=fp32_res,
        fp16=fp16_res,
        fp16_speedup=2.0,
        mixed_precision_ready=True,
        tf32_enabled=False,
    )
    assert mixed.fp16_speedup == 2.0
    assert mixed.fp32.tflops == 10.0
