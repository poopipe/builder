General:
- responses must be concise and deliver only the information requested. if you have something other than a direct answer to a question to say, confirm that i want to hear it before wasting tokens.
- comments should be concise and describe intent rather than behaviour
- I will be editing the code manually outside of your cursor environment- it is essential that you ensure you are working off the most recently saved on-disk file at all times.
- never use PowerShell for shell commands or file I/O; use cmd, git, python, or other direct tools instead
- no mutable module-level globals; pass state explicitly (immutable constants are fine)
- do not refer to ui decoration as chrome - it makes me quite angry. I also dislike other jargon such as 'sidecar', simple english is readable and doesn't assume you spend your life reading blogs about react

Style:
- the _method convention is stupid - it's not actually private so there is no point pretending it is.  just name the methods
- do not name modules types.py (collides / confuses tooling); use modulename_types.py
- methods/functions that return a bool should be prefixed with is_, can_, has_ eg. is_point_in_rect(), can_import(), has_valid_suffix()  

Imports:
- do not shorten library names eg. pyray should be imported as pyray, not pr
- in general do not import full modules. instead import types/functions etc. individually eg. from module import function, otherfunction, a_type
- in general i dislike object oriented design patterns, i prefer a functional style using dataclasses and libraries of pure (where possible) functions (exceptions are allowed with discussion)
- I insist on strict type annotations
