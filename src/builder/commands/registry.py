"""Command registration and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command, CommandId


@dataclass
class CommandRegistry:
    """Per-application map of command id → handler."""

    commands: dict[CommandId, Command] = field(default_factory=dict)

    def register(self, command: Command) -> None:
        """Add or replace a command."""
        self.commands[command.id] = command

    def get(self, command_id: CommandId) -> Command | None:
        """Look up a command by id."""
        return self.commands.get(command_id)

    def dispatch(self, context: CommandContext, command_id: CommandId) -> None:
        """Execute a command, or set an error status if missing."""
        command = self.commands.get(command_id)
        if command is None:
            context.ui.status = f"Unknown command: {command_id}"
            return
        command.run(context)

    def bind(
        self, context: CommandContext, command_id: CommandId
    ) -> Callable[[], None]:
        """Return a zero-arg callback suitable for UI widgets."""
        return lambda: self.dispatch(context, command_id)
