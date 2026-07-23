"""Command type definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from builder.commands.command_context import CommandContext


@dataclass(frozen=True)
class Command:
    """A labelled action; ``run`` may close over any arguments it needs."""

    label: str
    run: Callable[[CommandContext], None]
