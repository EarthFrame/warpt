"""Tests for the warpt version information."""

from datetime import datetime

from warpt.version.warpt_version import Version


def test_version_methods():
    """Test Version class methods."""
    v = Version(
        major=1,
        minor=2,
        patch=3,
        hash="abcdef123456",
        date=datetime(2023, 1, 1),
    )

    assert str(v) == "1.2.3"
    assert v.semver() == (1, 2, 3)
    assert v.hash_short(4) == "abcd"
    assert v.date_string("%Y") == "2023"
    assert "1.2.3" in v.full_version()
    assert "abcd" in v.full_version()


def test_warpt_version_instance():
    """Test the global WARPT_VERSION instance."""
    from warpt.version.warpt_version import WARPT_VERSION

    assert isinstance(WARPT_VERSION, Version)
    assert WARPT_VERSION.major >= 0
    assert len(WARPT_VERSION.hash) > 0
