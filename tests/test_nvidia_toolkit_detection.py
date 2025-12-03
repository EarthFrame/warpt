"""Tests for NVIDIA Container Toolkit detection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warpt.backends.software.nvidia_toolkit import (
    NvidiaContainerToolkitDetector,
    NvidiaContainerToolkitInfo,
)


def test_detect_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch):
    """Detector should return None when no binaries are on PATH."""

    def fake_which(_name: str) -> None:
        return None

    monkeypatch.setattr("shutil.which", fake_which)

    detector = NvidiaContainerToolkitDetector()
    assert detector.detect() is None


def test_detect_cli_only(monkeypatch: pytest.MonkeyPatch):
    """Detector captures CLI metadata even without the runtime."""
    paths = {
        "nvidia-container-cli": "/usr/bin/nvidia-container-cli",
    }

    def fake_which(name: str) -> str | None:
        return paths.get(name)

    def fake_run(cmd, **_kwargs):
        assert cmd[0] == "/usr/bin/nvidia-container-cli"
        return SimpleNamespace(
            returncode=0,
            stdout="NVIDIA Container Runtime - version 1.13.5",
            stderr="",
        )

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    detector = NvidiaContainerToolkitDetector()
    result = detector.detect()

    assert result is not None
    assert result.cli_path == "/usr/bin/nvidia-container-cli"
    assert result.cli_version == "1.13.5"
    assert result.runtime_path is None
    assert result.installed is True


def test_detect_with_runtime_and_docker(monkeypatch: pytest.MonkeyPatch):
    """Detector captures runtime path and Docker runtime availability."""
    paths = {
        "nvidia-container-cli": "/usr/bin/nvidia-container-cli",
        "nvidia-container-runtime": "/usr/bin/nvidia-container-runtime",
        "docker": "/usr/bin/docker",
    }

    def fake_which(name: str) -> str | None:
        return paths.get(name)

    def fake_run(cmd, **_kwargs):
        executable = cmd[0]
        if executable == "/usr/bin/nvidia-container-cli":
            return SimpleNamespace(
                returncode=0,
                stdout="NVIDIA Container Runtime - version 1.14.0",
                stderr="",
            )
        if executable == "/usr/bin/docker":
            return SimpleNamespace(
                returncode=0,
                stdout='{"nvidia": {"path": "/usr/bin/nvidia-container-runtime"}}',
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    detector = NvidiaContainerToolkitDetector()
    result = detector.detect()

    assert result is not None
    assert result.runtime_path == "/usr/bin/nvidia-container-runtime"
    assert result.cli_version == "1.14.0"
    assert result.docker_runtime_ready is True
    assert result.installed is True


def test_is_installed_relies_on_detect(monkeypatch: pytest.MonkeyPatch):
    """is_installed() mirrors the detect() outcome."""
    detector = NvidiaContainerToolkitDetector()

    monkeypatch.setattr(
        NvidiaContainerToolkitDetector,
        "detect",
        lambda _self: NvidiaContainerToolkitInfo(
            installed=True,
            cli_path=None,
            cli_version=None,
            runtime_path="/usr/bin/nvidia-container-runtime",
            docker_runtime_ready=None,
        ),
    )
    assert detector.is_installed() is True

    monkeypatch.setattr(NvidiaContainerToolkitDetector, "detect", lambda _self: None)
    assert detector.is_installed() is False


def test_serialization_methods(monkeypatch: pytest.MonkeyPatch):
    """Verify inherited serialization methods work correctly."""
    paths = {
        "nvidia-container-cli": "/usr/bin/nvidia-container-cli",
    }

    def fake_which(name: str) -> str | None:
        return paths.get(name)

    def fake_run(_cmd, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="NVIDIA Container Runtime - version 1.13.5",
            stderr="",
        )

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    detector = NvidiaContainerToolkitDetector()

    # Test to_dict
    data = detector.to_dict()
    assert data is not None
    assert data["installed"] is True
    assert data["cli_path"] == "/usr/bin/nvidia-container-cli"
    assert data["cli_version"] == "1.13.5"

    # Test to_json
    json_str = detector.to_json()
    assert json_str is not None
    assert '"installed": true' in json_str
    assert '"cli_version": "1.13.5"' in json_str
