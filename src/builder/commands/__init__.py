"""Commands package."""

from builder.commands.command_context import (
    ApplicationContext,
    CommandContext,
    SceneContext,
    UiContext,
)
from builder.commands.commands_types import Command, CommandEntry, CommandItem
from builder.commands.menus import MENU_COMMANDS, PANEL_COMMANDS, all_commands
from builder.commands.registry import (
    CommandRegistry,
    bind_command,
    bind_entry,
    register_commands,
)

__all__ = [
    "CommandItem",
    "ApplicationContext",
    "Command",
    "CommandContext",
    "CommandEntry",
    "CommandRegistry",
    "MENU_COMMANDS",
    "PANEL_COMMANDS",
    "SceneContext",
    "UiContext",
    "all_commands",
    "bind_command",
    "bind_entry",
    "register_commands",
]
