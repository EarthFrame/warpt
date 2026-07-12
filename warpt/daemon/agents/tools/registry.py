"""Tool registry — lookup, LLM tool defs, and never-raising execution."""

from __future__ import annotations

from typing import Any

from warpt.daemon.agents.tools.base import Tool, ToolDeniedError, ToolError
from warpt.daemon.llm.base import ToolDef
from warpt.utils.logger import Logger


class ToolRegistry:
    """Holds the tools available to an agent loop."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._log = Logger.get("daemon.agents.tools")

    def register(self, tool: Tool) -> None:
        """Add *tool*; duplicate names are a programming error.

        Raises
        ------
        ValueError
            When a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool named *name*.

        Raises
        ------
        ToolError
            When no such tool is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(
                f"Unknown tool {name!r} (available: {', '.join(self.names())})"
            ) from None

    def names(self) -> list[str]:
        """Return registered tool names in registration order."""
        return list(self._tools)

    def tool_defs(self) -> list[ToolDef]:
        """Return the registered tools as LLM ``ToolDef`` objects."""
        return [
            ToolDef(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
            )
            for t in self._tools.values()
        ]

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a tool by name; never raises.

        Returns
        -------
            ``{"status": "ok", "result": ...}`` on success,
            ``{"status": "denied", "error": ...}`` when policy/safety refused,
            ``{"status": "error", "error": ...}`` on any failure.
        """
        try:
            tool = self.get(name)
            result = tool.run(args or {})
            return {"status": "ok", "result": result}
        except ToolDeniedError as e:
            self._log.info("Tool %s denied: %s", name, e)
            return {"status": "denied", "error": str(e)}
        except ToolError as e:
            self._log.warning("Tool %s failed: %s", name, e)
            return {"status": "error", "error": str(e)}
        except Exception as e:
            self._log.exception("Tool %s crashed", name)
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}
