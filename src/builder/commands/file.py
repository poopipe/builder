"""File / session commands."""

from __future__ import annotations

from builder.commands.command_context import CommandContext
from builder.commands.commands_types import Command


def quit_app(context: CommandContext, _: None) -> None:
    context.application.should_close = True


def open_scene(context: CommandContext, _: None) -> None:
    context.ui.status = "Open: not implemented"


def about(context: CommandContext, _: None) -> None:
    context.ui.status = "Builder framework — pyray / raylib 6"


def import_mesh(context: CommandContext, _: None) -> None:
    context.ui.status = (
        "Import mesh: no file dialog yet. "
        + context.application.importer.status_message()
    )


cmd_quit: Command[None] = Command("Quit", quit_app)
cmd_open: Command[None] = Command("Open", open_scene)
cmd_about: Command[None] = Command("About", about)
cmd_import_mesh: Command[None] = Command("Import mesh", import_mesh)
