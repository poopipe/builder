"""Command type definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from builder.commands.command_context import CommandContext


@dataclass(frozen=True)
class Command[P]:
    """A labelled action; ``run`` always receives ``(context, params)``."""

    label: str
    run: Callable[[CommandContext, P], None]


@dataclass(frozen=True)
class CommandEntry[P]:
    """A command paired with the params used when the UI invokes it."""

    command: Command[P]
    params: P


# Heterogeneous menus erase ``P`` at the collection boundary.
type CommandItem = CommandEntry[Any]
