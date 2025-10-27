"""
Warpt - Performance monitoring and system utilities
"""

from warpt.version.warpt_version import Version, WARPT_VERSION

__version__ = str(WARPT_VERSION)
__version_info__ = WARPT_VERSION

__all__ = [
    "Version",
    "WARPT_VERSION",
    "__version__",
    "__version_info__",
]
