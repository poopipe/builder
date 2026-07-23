"""Command type definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from builder.commands.command_context import CommandContext

CommandId = str


@dataclass(frozen=True)
class Command:
    """A named, invokable action."""

    id: CommandId
    label: str
    run: Callable[[CommandContext], None]
