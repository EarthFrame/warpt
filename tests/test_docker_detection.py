"""Tests for Docker detection utilities."""

from __future__ import annotations

import types

import pytest

from warpt.backends.software.docker import (
    DockerDetectionResult,
    DockerDetector,
)


def test_detect_returns_none_when_executable_missing(monkeypatch: pytest.MonkeyPatch):
    """Detector should return None when docker is not on PATH."""
    monkeypatch.setattr("warpt.backends.software.docker.shutil.which", lambda _: None)

    detector = DockerDetector()
    assert detector.detect() is None


def test_detect_success(monkeypatch: pytest.MonkeyPatch):
    """Detector should return Docker path and parsed version."""

    def fake_which(_: str) -> str:
        return "/usr/local/bin/docker"

    def fake_run(*_args, **_kwargs):
        return types.SimpleNamespace(
            returncode=0,
            stdout="Docker version 25.0.3, build deadbeef",
            stderr="",
        )

    monkeypatch.setattr(
        "warpt.backends.software.docker.shutil.which",
        fake_which,
    )
    monkeypatch.setattr(
        "warpt.backends.software.docker.subprocess.run",
        fake_run,
    )

    detector = DockerDetector()
    result = detector.detect()

    assert result is not None
    assert result.path == "/usr/local/bin/docker"
    assert result.version == "25.0.3"


def test_is_installed_relies_on_detect(monkeypatch: pytest.MonkeyPatch):
    """is_installed should reflect the detect result."""
    detector = DockerDetector()

    monkeypatch.setattr(
        DockerDetector,
        "detect",
        lambda _: DockerDetectionResult(path="/tmp/docker", version="1.0"),
    )
    assert detector.is_installed() is True

    monkeypatch.setattr(DockerDetector, "detect", lambda _: None)
    assert detector.is_installed() is False
