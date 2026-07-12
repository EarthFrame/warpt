"""Read-only agent tools — the evidence-gathering seam for the Attending.

Every tool here is read-only with one policy-gated exception: the diagnostic
probe, which generates load and is therefore allowed only when config enables
it AND the node is idle. The remediation seam reuses this interface later.
"""

from warpt.daemon.agents.tools.base import Tool, ToolDeniedError, ToolError
from warpt.daemon.agents.tools.builtin import build_default_registry
from warpt.daemon.agents.tools.registry import ToolRegistry

__all__ = [
    "Tool",
    "ToolDeniedError",
    "ToolError",
    "ToolRegistry",
    "build_default_registry",
]
