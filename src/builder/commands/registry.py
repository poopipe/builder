"""Command registration and binding."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import CommandItem, Command, CommandEntry


@dataclass
class CommandRegistry:
    """Ordered list of available command entries."""

    entries: list[CommandItem] = field(default_factory=list)

    def register(self, entry: CommandItem) -> None:
        """Append a command entry."""
        self.entries.append(entry)


def register_commands(entries: Iterable[CommandItem]) -> CommandRegistry:
    """Register command entries into a new registry and return it."""
    registry: CommandRegistry = CommandRegistry()
    entry: CommandItem
    for entry in entries:
        registry.register(entry)
    return registry


def bind_command[P](
    context: CommandContext,
    command: Command[P],
    params: P,
) -> Callable[[], None]:
    """Return a zero-arg callback that runs ``command`` with ``context`` and ``params``."""
    return lambda: command.run(context, params)


def bind_entry(context: CommandContext, entry: CommandEntry[Any]) -> Callable[[], None]:
    """Bind a ``CommandEntry`` (command + params) to a zero-arg UI callback."""
    return bind_command(context, entry.command, entry.params)
