"""Tests for the DaemonProcess — orchestration, PID file, lifecycle."""

from __future__ import annotations

import threading
from unittest.mock import patch

from warpt.daemon.daemon_process import DaemonProcess, send_stop


def test_start_wires_vitals_nurse_to_charge_nurse(tmp_path) -> None:
    """Starting the daemon wires VitalsNurse threshold callback to ChargeNurse."""
    pid_dir = tmp_path / ".warpt"
    pid_dir.mkdir()

    with (
        patch("warpt.daemon.daemon_process.VitalsNurse") as mock_vn_cls,
        patch("warpt.daemon.daemon_process.ChargeNurse") as mock_cn_cls,
        patch("warpt.daemon.daemon_process.CaseFile"),
    ):
        dp = DaemonProcess(warpt_dir=str(pid_dir))

        # Run in thread so we can stop it
        t = threading.Thread(target=dp.run, daemon=True)
        t.start()
        dp.stop()
        t.join(timeout=2)

        mock_vn = mock_vn_cls.return_value
        mock_cn = mock_cn_cls.return_value
        mock_vn.set_on_threshold_breach.assert_called_once_with(mock_cn.handle_breach)
        mock_vn.start.assert_called_once()


def test_pid_file_written_on_start_removed_on_stop(tmp_path) -> None:
    """PID file is created on start and removed after shutdown."""
    pid_dir = tmp_path / ".warpt"
    pid_dir.mkdir()

    with (
        patch("warpt.daemon.daemon_process.VitalsNurse"),
        patch("warpt.daemon.daemon_process.ChargeNurse"),
        patch("warpt.daemon.daemon_process.CaseFile"),
    ):
        dp = DaemonProcess(warpt_dir=str(pid_dir))
        pid_path = pid_dir / "daemon.pid"

        t = threading.Thread(target=dp.run, daemon=True)
        t.start()

        # PID file should exist while running
        import time

        time.sleep(0.05)
        assert pid_path.exists()
        pid_content = pid_path.read_text().strip()
        assert pid_content.isdigit()

        dp.stop()
        t.join(timeout=2)

        # PID file removed after shutdown
        assert not pid_path.exists()


def test_double_start_detected(tmp_path) -> None:
    """Starting when a daemon is already running raises RuntimeError."""
    pid_dir = tmp_path / ".warpt"
    pid_dir.mkdir()

    import os

    # Write a PID file with our own PID (so is_running() returns True)
    pid_path = pid_dir / "daemon.pid"
    pid_path.write_text(str(os.getpid()))

    dp = DaemonProcess(warpt_dir=str(pid_dir))
    assert dp.is_running() is True

    import pytest

    with pytest.raises(RuntimeError, match="already running"):
        dp.run()


def test_stop_when_not_running(tmp_path) -> None:
    """Stopping when no daemon is running returns a clear message."""
    pid_dir = tmp_path / ".warpt"
    pid_dir.mkdir()

    result = send_stop(warpt_dir=str(pid_dir))
    assert "not running" in result.lower()


def test_status_reports_stopped_when_no_pid(tmp_path) -> None:
    """Status shows not running when no PID file exists."""
    pid_dir = tmp_path / ".warpt"
    pid_dir.mkdir()

    dp = DaemonProcess(warpt_dir=str(pid_dir))
    status = dp.get_status()
    assert status["running"] is False
    assert "pid" not in status


def test_status_reports_running_with_valid_pid(tmp_path) -> None:
    """Status shows running when PID file points to a live process."""
    import os

    pid_dir = tmp_path / ".warpt"
    pid_dir.mkdir()
    (pid_dir / "daemon.pid").write_text(str(os.getpid()))

    dp = DaemonProcess(warpt_dir=str(pid_dir))
    status = dp.get_status()
    assert status["running"] is True
    assert status["pid"] == os.getpid()


def test_missing_duckdb_shows_clear_error() -> None:
    """When duckdb is not installed, check_duckdb returns a helpful message."""
    from warpt.commands.daemon_cmd import check_duckdb

    with patch(
        "warpt.commands.daemon_cmd.importlib.import_module",
        side_effect=ImportError("No module named 'duckdb'"),
    ):
        result = check_duckdb()
        assert result is not None
        assert "pip install warpt[daemon]" in result


def test_cli_daemon_status_when_not_running(tmp_path) -> None:
    """'warpt daemon status' reports not running when no daemon exists."""
    from click.testing import CliRunner

    from warpt.cli import warpt

    pid_dir = tmp_path / ".warpt"
    pid_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(warpt, ["daemon", "status"], env={"WARPT_DIR": str(pid_dir)})
    assert result.exit_code == 0
    assert "not running" in result.output.lower() or "stopped" in result.output.lower()
