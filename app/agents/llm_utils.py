"""Shared helpers for invoking the chat model and parsing JSON output."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage


def invoke_llm(model: BaseChatModel, messages: list[BaseMessage]) -> AIMessage:
    """Invoke the chat model and return its ``AIMessage``.

    Args:
        model: Any LangChain chat model.
        messages: Prompt messages.

    Returns:
        The model's response message.
    """
    return model.invoke(messages)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response.

    Tolerates markdown fences, leading prose and trailing commentary —
    common LLM behaviors that break naive ``json.loads`` calls.

    Args:
        text: Raw model output.

    Returns:
        The parsed dict, or ``None`` when no JSON object is present.
    """
    if not text:
        return None
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        candidates.extend(fenced)
    # Balance-scan for an outermost {...} block.
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in starts:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None