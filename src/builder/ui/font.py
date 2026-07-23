"""Load CaskaydiaCove Nerd Font as the application default."""

from __future__ import annotations

from pathlib import Path

from pyray import (
    Font,
    TextureFilter,
    get_font_default,
    load_font_ex,
    set_texture_filter,
    unload_font,
)

_FONT_NAME = "CaskaydiaCoveNerdFont-Regular.ttf"


def assets_fonts_dir() -> Path:
    """Return the packaged fonts directory."""
    return Path(__file__).resolve().parent.parent / "assets" / "fonts"


def load_app_font(size: int = 18) -> Font:
    """Load CaskaydiaCove, or fall back to the raylib default font."""
    path = assets_fonts_dir() / _FONT_NAME
    if not path.is_file():
        print(f"warning: font not found at {path}; using default font")
        return get_font_default()

    font = load_font_ex(str(path), size, None, 0)
    if getattr(font, "glyphCount", getattr(font, "glyph_count", 0)) == 0:
        print(f"warning: failed to load font {path}; using default font")
        return get_font_default()

    set_texture_filter(font.texture, TextureFilter.TEXTURE_FILTER_BILINEAR)
    return font


def unload_app_font(font: Font) -> None:
    """Unload a font if it is not the built-in default."""
    default = get_font_default()
    if font.texture.id != default.texture.id:
        unload_font(font)
