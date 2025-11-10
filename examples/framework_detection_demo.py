#!/usr/bin/env python3
"""Demo script showing framework detection capabilities."""

import json

from warpt.backends.software.frameworks import detect_all_frameworks, detect_framework


def main():
    """Demonstrate framework detection and structured output."""
    print("=" * 60)
    print("Framework Detection Demo")
    print("=" * 60)
    print()

    # Detect all frameworks
    print("Detecting all installed frameworks...")
    frameworks = detect_all_frameworks()

    if not frameworks:
        print("No ML frameworks detected.")
        return

    # Display results in human-readable format
    print(f"\nFound {len(frameworks)} framework(s):\n")
    for name, info in frameworks.items():
        print(f"  {name}:")
        print(f"    Version: {info.version}")
        print(f"    CUDA Support: {'Yes' if info.cuda_support else 'No'}")
        print()

    # Show JSON output
    print("=" * 60)
    print("JSON Output (for integration with other tools):")
    print("=" * 60)

    # Convert to dict for JSON serialization
    frameworks_dict = {name: info.model_dump() for name, info in frameworks.items()}

    print(json.dumps(frameworks_dict, indent=2))
    print()

    # Demonstrate individual framework detection
    print("=" * 60)
    print("Individual Framework Detection Example:")
    print("=" * 60)

    pytorch_info = detect_framework("pytorch")
    if pytorch_info:
        print("\nPyTorch is installed:")
        print(f"  Version: {pytorch_info.version}")
        print(f"  CUDA Support: {'Yes' if pytorch_info.cuda_support else 'No'}")
    else:
        print("\nPyTorch is not installed.")


if __name__ == "__main__":
    main()
