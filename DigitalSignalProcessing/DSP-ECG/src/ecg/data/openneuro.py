"""OpenNeuro command helpers."""

from __future__ import annotations


def format_shell_command(command: list[str]) -> str:
    if len(command) <= 3:
        return " ".join(command)
    return " ".join(command[:3]) + " \\\n  " + " \\\n  ".join(command[3:])

