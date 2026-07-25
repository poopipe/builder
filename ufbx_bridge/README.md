# ufbx bridge

This folder builds a tiny DLL that wraps [ufbx](https://github.com/ufbx/ufbx).
No numpy, no Assimp, no Autodesk SDK.

## Build (Windows, cmd.exe)

```bat
cd ufbx_bridge
build.cmd
```

That downloads `ufbx.c` / `ufbx.h` if missing and produces `ufbx_bridge/build/ufbx_bridge.dll`.

Needs either:

- **MSVC**: open *x64 Native Tools Command Prompt for VS*, then run `build.cmd`
- **or MinGW `gcc`** on PATH

## Runtime

Python loads `ufbx_bridge/build/ufbx_bridge.dll` via ctypes (see `builder.io.ufbx_bridge`).
