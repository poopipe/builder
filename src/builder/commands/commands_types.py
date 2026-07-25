"""Command type definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from builder.commands.command_context import CommandContext


@dataclass(frozen=True)
class Command[P]:
    """generic. labelled action, run always receives context, params"""

    label: str
    run: Callable[[CommandContext, P], None]


@dataclass(frozen=True)
class CommandEntry[P]:
    """generic. command plus params(of any type) used when invoked"""

    command: Command[P]
    params: P


# alias that stops the type checker getting upset when you eg. have a list of CommandEntry that use different types
type CommandItem = CommandEntry[Any]
