# Builder

Procedural mesh placement tool (framework scaffold).

## Run

```bat
cd C:\Users\malcolm\src\github\poopipe\builder
.venv\Scripts\python.exe -m builder
```

Or with Poetry:

```bat
poetry run python -m builder
```

## Controls

- **Right-drag** — orbit camera
- **Middle-drag** — pan
- **Scroll** — zoom
- Top menu — application actions
- Left panel — context actions (clipped when the window is short)

## Mesh import

FBX import is stubbed. Prefer **ufbx** bindings (`pyufbx` / `pufbx`) later for Windows / Linux / macOS — not the Autodesk FBX SDK. Assimp is a fallback if wheels lag your Python version.

## Font

UI text uses **CaskaydiaCove Nerd Font** (OFL), vendored under `src/builder/assets/fonts/`.
