"""Build the system prompt for the integration agent.

Assembles the prompt from static templates (integration guide,
coding standards) and runtime-read source files (ABC, NVIDIA
reference, factory pattern, test patterns).
"""

from __future__ import annotations

from pathlib import Path

# Root of the warpt package
_WARPT_ROOT = Path(__file__).resolve().parent.parent
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _read_file(path: Path) -> str:
    """Read a file, returning empty string on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return f"[Could not read {path}]"


def build_system_prompt(
    vendor: str,
    sdk_docs_text: str,  # noqa: ARG001
    vendor_context: str | None = None,
) -> str:
    """Assemble the full system prompt for the agent.

    Reads warpt source files at runtime so the prompt always
    reflects the current state of the codebase.

    Parameters
    ----------
    vendor : str
        Vendor name (e.g., "tenstorrent").
    sdk_docs_text : str
        Pre-loaded SDK documentation text.
    vendor_context : str | None
        Optional additional context about the hardware.

    Returns
    -------
    str
        The complete system prompt.
    """
    sections: list[str] = []

    # 1. Integration guide (the "constitution")
    guide = _read_file(_PROMPTS_DIR / "integration_guide.md")
    sections.append(guide)

    # 2. Coding standards
    standards = _read_file(_PROMPTS_DIR / "coding_standards.md")
    sections.append(
        f"# Coding Standards Reference\n\n{standards}"
    )

    # 3. ABC definition
    abc_text = _read_file(_WARPT_ROOT / "backends" / "base.py")
    sections.append(
        "# AcceleratorBackend ABC (warpt/backends/base.py)\n"
        "```python\n"
        f"{abc_text}\n"
        "```"
    )

    # 4. NVIDIA reference implementation
    nvidia_text = _read_file(
        _WARPT_ROOT / "backends" / "nvidia.py"
    )
    sections.append(
        "# NVIDIA Reference Implementation "
        "(warpt/backends/nvidia.py)\n"
        "```python\n"
        f"{nvidia_text}\n"
        "```"
    )

    # 5. Factory pattern
    factory_text = _read_file(
        _WARPT_ROOT / "backends" / "factory.py"
    )
    sections.append(
        "# Factory Pattern (warpt/backends/factory.py)\n"
        "```python\n"
        f"{factory_text}\n"
        "```"
    )

    # 6. Power backend ABC
    power_base_text = _read_file(
        _WARPT_ROOT / "backends" / "power" / "base.py"
    )
    sections.append(
        "# PowerBackend ABC "
        "(warpt/backends/power/base.py)\n"
        "```python\n"
        f"{power_base_text}\n"
        "```"
    )

    # 7. NVIDIA power reference
    nvidia_power_text = _read_file(
        _WARPT_ROOT / "backends" / "power" / "nvidia_power.py"
    )
    sections.append(
        "# NVIDIA Power Reference "
        "(warpt/backends/power/nvidia_power.py)\n"
        "```python\n"
        f"{nvidia_power_text}\n"
        "```"
    )

    # 8. Test patterns
    test_factory_text = _read_file(
        _WARPT_ROOT.parent / "tests" / "test_backends_factory.py"
    )
    sections.append(
        "# Test Patterns "
        "(tests/test_backends_factory.py)\n"
        "```python\n"
        f"{test_factory_text}\n"
        "```"
    )

    # 9. GPUInfo model
    list_models_text = _read_file(
        _WARPT_ROOT / "models" / "list_models.py"
    )
    sections.append(
        "# GPUInfo Model (warpt/models/list_models.py)\n"
        "```python\n"
        f"{list_models_text}\n"
        "```"
    )

    # 10. Power models
    power_models_text = _read_file(
        _WARPT_ROOT / "models" / "power_models.py"
    )
    sections.append(
        "# Power Models (warpt/models/power_models.py)\n"
        "```python\n"
        f"{power_models_text}\n"
        "```"
    )

    # 11. Question protocol
    sections.append(
        "# Question Protocol\n\n"
        "As you work, log questions directly to "
        "`questions.yaml` in the repo root.\n\n"
        "**IMPORTANT: You MUST use this exact YAML schema. "
        "Do not invent your own fields.**\n\n"
        "The file has this structure:\n"
        "```yaml\n"
        "metadata:\n"
        "  vendor: vendorname\n"
        "  sdk_source: cli-provided\n"
        '  created: "2026-01-01T00:00:00+00:00"\n'
        '  session_id: "..."\n'
        "  pass_number: 1\n"
        "questions:\n"
        "  - id: 1\n"
        "    tier: blocking\n"
        "    status: open\n"
        '    title: "Short description"\n'
        '    finding: "What you found in the SDK docs"\n'
        '    decision: "What you decided, or why you '
        'couldn\'t"\n'
        '    alternatives: "Available options"\n'
        '    impact: "Downstream impact on warpt"\n'
        '    code_reference: "file:line"\n'
        "    answer: null\n"
        "    notes: null\n"
        "```\n\n"
        "Required fields for each question:\n"
        "- `id`: integer, auto-increment (1, 2, 3...)\n"
        "- `tier`: one of: blocking, defaulted, "
        "clarification_needed, informational\n"
        "- `status`: one of: open, answered, implemented, "
        "verified\n"
        "- `title`: short description string\n"
        "- `finding`: what you found in the SDK docs\n"
        "- `decision`: what you decided or why you "
        "couldn't\n"
        "- `alternatives`: available options\n"
        "- `impact`: downstream impact on warpt\n"
        "- `code_reference`: file:line (optional, can be "
        "null)\n"
        "- `answer`: null (filled by human later)\n"
        "- `notes`: null (optional)\n\n"
        "Tier meanings:\n"
        "- **blocking**: Cannot proceed without answer\n"
        "- **defaulted**: Made a reasonable assumption\n"
        "- **clarification_needed**: Multiple valid "
        "approaches, need vendor input\n"
        "- **informational**: FYI for the vendor\n\n"
        "Do NOT modify the metadata section — it is "
        "pre-populated. Only add entries to the questions "
        "list."
    )

    # 12. Note that SDK docs will be in the user prompt
    sections.append(
        f"# Vendor SDK Documentation ({vendor})\n\n"
        "The vendor's SDK documentation is provided in the "
        "user prompt below (not in this system prompt). "
        "Refer to it when implementing the backend."
    )

    # 13. Vendor context (if provided)
    if vendor_context:
        sections.append(
            f"# Additional Vendor Context\n\n{vendor_context}"
        )

    # 14. Task directive
    sections.append(
        f"# Your Task\n\n"
        f"Implement the **{vendor}** backend for warpt.\n\n"
        f"Vendor name: `{vendor}`\n"
        f"Backend class: `{vendor.capitalize()}Backend`\n"
        f"Backend file: `warpt/backends/{vendor}.py`\n"
        f"Power file: `warpt/backends/power/{vendor}_power.py`\n"
        f"Test file: `tests/test_{vendor}_backend.py`\n\n"
        "Follow the integration guide above. "
        "Log questions as you encounter them. "
        "Run tests and linting when done."
    )

    return "\n\n---\n\n".join(sections)
