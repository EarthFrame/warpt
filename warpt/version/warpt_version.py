from dataclasses import dataclass
from datetime import datetime
from typing import Tuple
import hashlib
import os


@dataclass(frozen=True)
class Version:
    """
    Semantic version information for warpt.
    
    Includes major, minor, and patch version numbers following semver,
    plus a package hash and build date.
    """
    major: int
    minor: int
    patch: int
    hash: str
    date: datetime
    
    def __str__(self) -> str:
        """Return the semantic version string (e.g., '0.1.0')."""
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def full_version(self) -> str:
        """Return full version info including hash and date."""
        return (
            f"{self} (hash: {self.hash[:8]}, "
            f"date: {self.date.strftime('%Y-%m-%d')})"
        )
    
    def semver(self) -> Tuple[int, int, int]:
        """Return semantic version as tuple (major, minor, patch)."""
        return (self.major, self.minor, self.patch)
    
    def hash_short(self, length: int = 8) -> str:
        """Return shortened hash (default 8 characters)."""
        return self.hash[:length]
    
    def date_string(self, fmt: str = "%Y-%m-%d") -> str:
        """Return formatted date string."""
        return self.date.strftime(fmt)


def _compute_package_hash() -> str:
    """
    Compute a hash of the warpt package.
    
    Returns a SHA256 hash of the main package directory contents.
    """
    warpt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hasher = hashlib.sha256()
    
    for root, dirs, files in os.walk(warpt_dir):
        # Skip __pycache__ and other non-essential directories
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', '.pytest_cache')]
        
        for file in sorted(files):
            if file.endswith(('.pyc', '.pyo', '.pyd')):
                continue
            
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())
            except (IOError, OSError):
                pass
    
    return hasher.hexdigest()


# Current version instance
WARPT_VERSION = Version(
    major=0,
    minor=1,
    patch=0,
    hash=_compute_package_hash(),
    date=datetime(2025, 10, 27),
)

