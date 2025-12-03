"""Tests for Docker detection utilities."""

from __future__ import annotations

import types

import pytest

from warpt.backends.software.docker import DockerDetector
from warpt.models.list_models import DockerInfo


def test_detect_returns_none_when_executable_missing(monkeypatch: pytest.MonkeyPatch):
    """Detector should return None when docker is not on PATH."""
    monkeypatch.setattr("shutil.which", lambda _: None)

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

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    detector = DockerDetector()
    result = detector.detect()

    assert result is not None
    assert isinstance(result, DockerInfo)
    assert result.path == "/usr/local/bin/docker"
    assert result.version == "25.0.3"
    assert result.installed is True


def test_is_installed_relies_on_detect(monkeypatch: pytest.MonkeyPatch):
    """is_installed should reflect the detect result."""
    detector = DockerDetector()

    monkeypatch.setattr(
        DockerDetector,
        "detect",
        lambda _: DockerInfo(installed=True, path="/tmp/docker", version="1.0"),
    )
    assert detector.is_installed() is True

    monkeypatch.setattr(DockerDetector, "detect", lambda _: None)
    assert detector.is_installed() is False


def test_software_name():
    """software_name property should return 'docker'."""
    detector = DockerDetector()
    assert detector.software_name == "docker"


def test_to_dict(monkeypatch: pytest.MonkeyPatch):
    """to_dict should return dictionary representation."""
    detector = DockerDetector()
    monkeypatch.setattr(
        DockerDetector,
        "detect",
        lambda _: DockerInfo(installed=True, path="/usr/bin/docker", version="25.0.3"),
    )

    result = detector.to_dict()
    assert result is not None
    assert result["installed"] is True
    assert result["path"] == "/usr/bin/docker"
    assert result["version"] == "25.0.3"


def test_to_json(monkeypatch: pytest.MonkeyPatch):
    """to_json should return JSON string."""
    detector = DockerDetector()
    monkeypatch.setattr(
        DockerDetector,
        "detect",
        lambda _: DockerInfo(installed=True, path="/usr/bin/docker", version="25.0.3"),
    )

    result = detector.to_json()
    assert result is not None
    assert '"installed": true' in result
    assert '"path": "/usr/bin/docker"' in result
