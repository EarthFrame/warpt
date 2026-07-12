"""Prompt-driven tool-call emulation for backends without native tool support.

Some backends (older Ollama models, bare inference clusters) can't take a
``tools`` parameter. This module renders the tool definitions into a prompt
instruction and constrains the reply to ``EMULATION_SCHEMA`` via the
provider's existing structured-output machinery, so the agent loop sees the
same ``ToolCall`` seam regardless of backend capability.
"""

from __future__ import annotations

import json
from typing import Any

from warpt.daemon.llm.base import LLMResponse, ToolCall, ToolDef

# The model replies with exactly one of: a tool invocation (tool non-null)
# or a final answer (tool null + response text).
EMULATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": ["string", "null"]},
        "arguments": {"type": "object"},
        "response": {"type": ["string", "null"]},
    },
    "required": ["tool"],
}

_EMULATION_INSTRUCTION = """\

You have access to the following tools. You may call ONE tool per turn.
Reply with ONLY a JSON object of this shape (no prose, no code fences):
{{"tool": "<tool name or null>", "arguments": {{...}},
 "response": "<final answer when tool is null>"}}

Available tools:
{tool_lines}"""


def build_emulation_messages(
    messages: list[dict[str, Any]], tools: list[ToolDef]
) -> tuple[list[dict[str, Any]], str]:
    """Flatten canonical messages and render tools into a prompt suffix.

    Canonical assistant-with-``tool_calls`` and ``role: "tool"`` messages are
    flattened to plain text so the transcript works on backends that only
    understand ``system``/``user``/``assistant`` strings.

    Parameters
    ----------
    messages
        Canonical message list (see ``llm.base`` module docstring).
    tools
        Tool definitions to expose to the model.

    Returns
    -------
        ``(flattened_messages, system_suffix)`` — append the suffix to the
        system prompt before generating.
    """
    tool_lines = "\n".join(
        f"- {t.name}: {t.description}\n  arguments schema: "
        f"{json.dumps(t.input_schema)}"
        for t in tools
    )
    suffix = _EMULATION_INSTRUCTION.format(tool_lines=tool_lines)

    flattened: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            calls = ", ".join(
                f"{c['name']}({json.dumps(c.get('arguments', {}))})"
                for c in msg["tool_calls"]
            )
            text = msg.get("content") or ""
            prefix = f"{text}\n" if text else ""
            flattened.append(
                {"role": "assistant", "content": prefix + f"[called tools: {calls}]"}
            )
        elif role == "tool":
            flattened.append(
                {
                    "role": "user",
                    "content": (
                        f"[tool {msg.get('name', '?')} returned: "
                        f"{msg.get('content', '')}]"
                    ),
                }
            )
        else:
            flattened.append({"role": role, "content": msg.get("content", "")})
    return flattened, suffix


def parse_emulation_result(
    structured: dict[str, Any], *, model: str, provider: str
) -> LLMResponse:
    """Convert an ``EMULATION_SCHEMA``-shaped object into an ``LLMResponse``.

    Parameters
    ----------
    structured
        Validated object matching ``EMULATION_SCHEMA``.
    model
        Model identifier for the response.
    provider
        Provider name for the response.
    """
    tool = structured.get("tool")
    if tool:
        return LLMResponse(
            text="",
            model=model,
            provider=provider,
            tool_calls=[
                ToolCall(
                    id="call_0", name=tool, arguments=structured.get("arguments") or {}
                )
            ],
            stop_reason="tool_use",
        )
    return LLMResponse(
        text=structured.get("response") or "",
        model=model,
        provider=provider,
        stop_reason=None,
    )
