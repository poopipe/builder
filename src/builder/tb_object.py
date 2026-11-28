"""terrain builder object-import row and file parse/write"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TbObject:
    """one terrain builder object-import row (no height; snaps on import)"""

    model: str
    x: float
    y: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0
    scale: float = 1.0
    reserved: float = 0.0
    underground: int = 0

    def to_line(self) -> str:
        """tb object export line; template name only, no z"""
        return (
            f'"{self.model}";{self.x:.6f};{self.y:.6f};'
            f"{self.yaw:.6f};{self.pitch:.6f};{self.roll:.6f};{self.scale:.6f};"
            f"{self.reserved:.6f};{int(self.underground)};"
        )


def parse_tb_object_line(line: str) -> TbObject | None:
    """parse one import line; none when blank or malformed"""
    stripped: str = line.strip()
    if stripped == "":
        return None
    parts: list[str] = stripped.split(";")
    if len(parts) < 9:
        return None
    model: str = parts[0].strip().strip('"').strip("'")
    if model == "":
        return None
    try:
        x: float = float(parts[1])
        y: float = float(parts[2])
        yaw: float = float(parts[3])
        pitch: float = float(parts[4])
        roll: float = float(parts[5])
        scale: float = float(parts[6])
        reserved: float = float(parts[7])
        underground: int = int(float(parts[8]))
    except ValueError:
        return None
    return TbObject(
        model=model,
        x=x,
        y=y,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        scale=scale,
        reserved=reserved,
        underground=underground,
    )


def parse_tb_object_import(path: Path) -> list[TbObject]:
    """load every valid object-import row from a txt file"""
    text: str = Path(path).read_text(encoding="utf-8")
    objects: list[TbObject] = []
    line: str
    for line in text.splitlines():
        parsed: TbObject | None = parse_tb_object_line(line)
        if parsed is not None:
            objects.append(parsed)
    return objects


def write_object_import(path: Path, objects: list[TbObject]) -> None:
    """write one import_*.txt for terrain builder"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [obj.to_line() for obj in objects]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
