"""Version command - displays warpt version information."""

from warpt.version import WARPT_VERSION


def run_version(verbose: bool = False) -> None:
    """Display warpt version information.

    Args:
        verbose: If True, show additional details like full hash and date
    """
    if verbose:
        print(f"warpt version {WARPT_VERSION.full_version()}")
        print("\nDetailed version information:")
        major = WARPT_VERSION.major
        minor = WARPT_VERSION.minor
        patch = WARPT_VERSION.patch
        print(f"  Semantic Version: {major}.{minor}.{patch}")
        print(f"  Build Date:       {WARPT_VERSION.date_string()}")
        print(f"  Package Hash:     {WARPT_VERSION.hash}")
    else:
        print(f"warpt {WARPT_VERSION}")
