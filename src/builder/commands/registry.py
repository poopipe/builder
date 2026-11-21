"""command registration and binding"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import CommandItem, CommandEntry
from builder.session_journal import log_exception


@dataclass
class CommandRegistry:
    """ordered list of available command entries"""

    entries: list[CommandItem] = field(default_factory=list)

    def register(self, entry: CommandItem) -> None:
        """append a command entry"""
        self.entries.append(entry)


def register_commands(entries: Iterable[CommandItem]) -> CommandRegistry:
    """register command entries into a new registry and return it"""
    registry: CommandRegistry = CommandRegistry()
    entry: CommandItem
    for entry in entries:
        registry.register(entry)
    return registry


def bind_entry(context: CommandContext, entry: CommandEntry[Any]) -> Callable[[], None]:
    """return a zero-arg callback that runs the entry's command with context and params"""

    def run_bound() -> None:
        context.application.journal.record(
            "command",
            label=entry.command.label,
            params=entry.params,
        )
        try:
            entry.command.run(context, entry.params)
        except Exception as exc:
            log_exception(f"command failed: {entry.command.label}", exc)
            context.ui.status = f"Error: {exc}"

    return run_bound
