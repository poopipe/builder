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

FONT_NAME: str = "CaskaydiaCoveNerdFont-Regular.ttf"


def assets_fonts_dir() -> Path:
    """Return the packaged fonts directory."""
    return Path(__file__).resolve().parent.parent / "assets" / "fonts"


def load_app_font(size: int = 18) -> Font:
    """Load CaskaydiaCove. Raises if the file is missing or fails to load."""
    path: Path = assets_fonts_dir() / FONT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"application font not found: {path}")

    font: Font = load_font_ex(str(path), size, None, 0)
    glyph_count: int = int(getattr(font, "glyphCount", getattr(font, "glyph_count", 0)))
    if glyph_count == 0:
        raise RuntimeError(f"failed to load application font: {path}")

    set_texture_filter(font.texture, TextureFilter.TEXTURE_FILTER_BILINEAR)
    return font


def unload_app_font(font: Font) -> None:
    """Unload a font if it is not the built-in default."""
    default: Font = get_font_default()
    if font.texture.id != default.texture.id:
        unload_font(font)
