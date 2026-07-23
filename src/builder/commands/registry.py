"""Command registration and binding."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command


@dataclass
class CommandRegistry:
    """Ordered list of available commands."""

    commands: list[Command] = field(default_factory=list)

    def register(self, command: Command) -> None:
        """Append a command."""
        self.commands.append(command)


def register_commands(commands: Iterable[Command]) -> CommandRegistry:
    """Register commands into a new registry and return it."""
    registry = CommandRegistry()
    for command in commands:
        registry.register(command)
    return registry


def bind_command(
    context: CommandContext, command: Command
) -> Callable[[], None]:
    """Return a zero-arg callback that runs ``command`` with ``context``."""
    return lambda: command.run(context)
