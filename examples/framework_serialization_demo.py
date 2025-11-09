#!/usr/bin/env python3
"""Demo script showing framework serialization methods."""

from warpt.backends.software.frameworks import PyTorchDetector


def demo_basic_serialization():
    """Demonstrate basic serialization methods."""
    print("=" * 60)
    print("Basic Serialization Demo")
    print("=" * 60)
    print()

    detector = PyTorchDetector()

    print(f"Framework: {detector.framework_name}")
    print()

    # Dict format
    print("1. Dictionary format:")
    data = detector.to_dict()
    if data:
        for key, value in data.items():
            print(f"   {key}: {value}")
    else:
        print("   Framework not installed")
    print()

    # JSON format
    print("2. JSON format:")
    json_str = detector.to_json()
    if json_str:
        print(json_str)
    else:
        print("   Framework not installed")
    print()

    # Compact JSON
    print("3. Compact JSON (no indentation):")
    compact_json = detector.to_json(indent=None)
    if compact_json:
        print(compact_json)
    else:
        print("   Framework not installed")
    print()

    # YAML format (if PyYAML is installed)
    print("4. YAML format:")
    try:
        yaml_str = detector.to_yaml()
        if yaml_str:
            print(yaml_str)
        else:
            print("   Framework not installed")
    except ImportError as e:
        print(f"   Skipped: {e}")
    print()

    # TOML format (if tomli_w is installed)
    print("5. TOML format:")
    try:
        toml_str = detector.to_toml()
        if toml_str:
            print(toml_str)
        else:
            print("   Framework not installed")
    except ImportError as e:
        print(f"   Skipped: {e}")
    print()

    # HUML format (if pyhuml is installed)
    print("6. HUML format:")
    try:
        huml_str = detector.to_huml()
        if huml_str:
            print(huml_str)
        else:
            print("   Framework not installed")
    except ImportError as e:
        print(f"   Skipped: {e}")
    print()


def demo_custom_override():
    """Demonstrate how to override serialization methods."""
    print("=" * 60)
    print("Custom Override Demo")
    print("=" * 60)
    print()

    # Create a custom detector with overridden serialization
    class CustomPyTorchDetector(PyTorchDetector):
        """Custom PyTorch detector with enhanced output."""

        def to_dict(self) -> dict[str, str | bool] | None:
            """Override to add custom metadata."""
            base_dict = super().to_dict()
            if base_dict is None:
                return None

            # Add custom fields
            base_dict["framework_name"] = self.framework_name
            base_dict["detector_type"] = "custom"
            base_dict["supports_distributed"] = True

            return base_dict

    detector = CustomPyTorchDetector()

    print("Custom detector with additional fields:")
    print()

    # Show enhanced dict
    data = detector.to_dict()
    if data:
        print("Dictionary with custom fields:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    else:
        print("Framework not installed")
    print()

    # JSON automatically uses the overridden to_dict()
    json_str = detector.to_json()
    if json_str:
        print("JSON with custom fields:")
        print(json_str)
    else:
        print("Framework not installed")


def demo_multiple_formats():
    """Show how to export the same data in multiple formats."""
    print()
    print("=" * 60)
    print("Multiple Format Export Demo")
    print("=" * 60)
    print()

    detector = PyTorchDetector()

    formats = {
        "dict": lambda: detector.to_dict(),
        "json": lambda: detector.to_json(indent=None),
    }

    # Try YAML if available
    try:
        formats["yaml"] = lambda: detector.to_yaml()
    except ImportError:
        pass

    # Try TOML if available
    try:
        formats["toml"] = lambda: detector.to_toml()
    except ImportError:
        pass

    # Try HUML if available
    try:
        formats["huml"] = lambda: detector.to_huml()
    except ImportError:
        pass

    print("Exporting framework info in multiple formats:\n")

    for format_name, format_func in formats.items():
        try:
            result = format_func()
            if result:
                print(f"{format_name.upper()}:")
                print(result)
                print()
        except ImportError as e:
            print(f"{format_name.upper()}: Skipped ({e})")
            print()


if __name__ == "__main__":
    demo_basic_serialization()
    demo_custom_override()
    demo_multiple_formats()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
