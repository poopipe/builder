General:
- responses must be concise and deliver only the information requested. if you have something other than a direct answer to a question to say, confirm that i want to hear it before wasting tokens.
- comments should be concise and describe intent rather than behaviour
- I will be editing the code manually outside of your cursor environment- it is essential that you ensure you are working off the most recently saved on-disk file at all times.
- never edit from chat history or memory: before every change, re-read that file from disk in the same turn; base the edit only on what you just read; do not rewrite a whole file from an earlier version in the conversation
- never use PowerShell. on this machine the Cursor Shell tool always wraps commands in PowerShell, so do not use the Shell tool at all — not even with `cmd /c`. use Read / Write / StrReplace / Grep / Glob / Delete instead. if a command must be run (pyright, app launch, build), ask me to run it and paste the output
- no mutable module-level globals; pass state explicitly (immutable constants are fine)
- do not obfuscate ownership with forwarding properties or thin wrappers that re-expose nested state under a flatter name (eg. `selected_ids` that only returns `self.nodes.selected_ids`). access the owning object directly so it is obvious what the data belongs to
- no redundant indirection: do not invent aliases, parallel catalogs, or normalize helpers that only restate something already usable (eg. building `axis_choices` from `Axis`, or `param_options` that unpacks an enum the caller could use). use the source value at the call site; if the API cannot accept it, change the API
- no magic-name / sniffing designs: do not wire behaviour by `hasattr`, string field names, or silent conventions that guess which data applies (eg. `if role == "point" and hasattr(params, "point_orient")`). the link between data and behaviour must be explicit and typed at the call site — if the shape is wrong, fix the types, do not invent a helper that papers over it

Style:
- the _method convention is stupid - it's not actually private so there is no point pretending it is.  just name the methods
- do not name modules types.py (collides / confuses tooling); use modulename_types.py
- US spelling
- docstrings and comments: lowercase, no trailing period — eg. `"""return transforms for grid in XZ centered at origin"""`

Imports:
- do not shorten library names eg. pyray should be imported as pyray, not pr
- in general do not import full modules. instead import types/functions etc. individually eg. from module import function, otherfunction, a_type
- for pyray enum-like constants, import the enum type and use members (eg. `ConfigFlags.FLAG_WINDOW_RESIZABLE`, `ShaderLocationIndex.SHADER_LOC_MATRIX_MVP`, `ShaderUniformDataType.SHADER_UNIFORM_FLOAT`) — bare names may work at runtime but fail the type checker
- in general i dislike object oriented design patterns, i prefer a functional style using dataclasses and libraries of pure (where possible) functions (exceptions are allowed with discussion)
- I insist on strict type annotations

Architecture / data model:
- do not force a feature into the wrong abstraction. generators are instance-placement recipes (transforms → meshed children of shared catalog meshes). continuous surfaces, terrains, and other unique geometry are first-class node recipes (eg. `Node.heightfield`), not another generator kind
- recipe fields that drive UI/build use typed param wrappers (`IntParam`, `FloatParam`, …); do not revive ParamField maps or string-field sniffing for recipes
- prefer IntEnum (or similar) over string/`Literal` discriminators for roles, modes, kinds; serialize enums by **name** in scene JSON, decode name-only — no int fallbacks
- no backwards compatibility in scene load unless explicitly requested: no migrators, legacy key aliases, soft “load anyway” version paths, or dual shapes. bump `scene_format_version` when the format changes; reject mismatches
- regenerated children (generator slots, heightfield tiles) are omitted on save and rebuilt on load; keep mesh/tile ids deterministic from parent id + indices so later LOD/streaming can swap residency without renaming the recipe
- separate catalogs for different asset kinds (meshes vs heightmaps); do not overload `MeshCatalog` for non-mesh assets
- design for large scale early when the user says so (tiling, sample-per-tile, no full-resolution buffers) but ship the smallest working slice they pick; leave hooks, do not build LOD/streaming until asked
- keep package `__init__` slim; domain code must not import `command_context` / command modules — use a local Protocol for the host surface needed (register mesh, unload, shader). match exhaustively when adding enum members (file browser titles, etc.); mirror existing UI action enums (`TextAction.editing`, not invented variants)
