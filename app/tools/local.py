"""Local utility tools: safe arithmetic and a persistent research notebook."""

from __future__ import annotations

import ast
import json
import operator as op
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("tools.local")

_NOTE_LOCK = threading.Lock()

# Supported AST operators for the calculator. Excludes attribute access,
# subscripting and names so arbitrary code cannot be evaluated.
_ALLOWED_BINOPS: dict[type, Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_ALLOWED_UNARYOPS: dict[type, Any] = {ast.USub: op.neg, ast.UAdd: op.pos}


@tool
def calculator(expression: str) -> str:
    """Evaluate a pure-arithmetic expression safely.

    Supports +, -, *, /, **, %, //, parentheses and unary +/-.
    Use this instead of mental math whenever numeric precision matters.

    Args:
        expression: A numeric expression, e.g. ``(2.5 * 3) / 0.5``.

    Returns:
        The numeric result, or an error message for invalid input.
    """
    logger.info("calculator: %r", expression)
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
        return f"Calculator error: {exc.__class__.__name__}: {exc}"
    return f"{expression} = {result}"


def _safe_eval(node: ast.AST) -> float | int:
    """Evaluate an arithmetic AST node, refusing anything non-numeric."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        binop_type = _ALLOWED_BINOPS.get(type(node.op))
        if binop_type is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return binop_type(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        unaryop_type = _ALLOWED_UNARYOPS.get(type(node.op))
        if unaryop_type is None:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
        return unaryop_type(_safe_eval(node.operand))
    raise ValueError(f"Expression element not allowed: {type(node).__name__}")


@tool
def save_note(content: str, topic: str = "general") -> str:
    """Save a research note to the persistent shared notebook.

    Use this to record important intermediate findings worth keeping for
    later questions in the session or for the final report.

    Args:
        content: The note text (keep it concise and self-contained).
        topic: Short topic tag used for filtering later.

    Returns:
        A confirmation message.
    """
    settings = get_settings()
    entry = {
        "topic": topic,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _NOTE_LOCK:
        _append_jsonl(settings.notes_path, entry)
    logger.info("save_note: %s (%s)", content[:60], topic)
    return f"Note saved under topic '{topic}'."


@tool
def list_notes(topic: str = "") -> str:
    """List previously saved research notes, optionally by topic.

    Args:
        topic: Optional topic tag filter; empty lists all notes.

    Returns:
        The formatted notes, or a message when none exist.
    """
    settings = get_settings()
    entries = _read_jsonl(settings.notes_path)
    if topic:
        entries = [e for e in entries if e.get("topic") == topic]
    if not entries:
        return "No notes found." if topic else "No notes found."
    lines = [
        f"- [{e.get('topic', '?')}] {e.get('content', '')} ({e.get('created_at', '?')})"
        for e in entries[-20:]
    ]
    return f"Saved notes ({len(entries)} total):\n" + "\n".join(lines)


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file, creating directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file (missing file -> empty list)."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed note line: %r", line[:80])
    return records