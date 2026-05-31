# Python readability, documentation, and typing rules

When editing Python code:

1. Preserve behavior unless the task explicitly asks for behavior changes.
2. Prefer clear human-readable names over short clever names.
3. Use `snake_case` for functions, methods, and variables.
4. Use `PascalCase` for classes.
5. Avoid vague names such as `data`, `item`, `result`, `x`, `tmp`, `thing`, `stuff`, `manager`, `helper`, unless the scope is tiny and obvious.
6. Add or improve type annotations for public functions, methods, constructors, dataclasses, and non-obvious internal helpers.
7. Add class and function docstrings using Google-style docstrings.
8. Docstrings must explain purpose, inputs, outputs, side effects, raised exceptions, and important constraints.
9. Do not write comments that simply repeat the code.
10. Prefer comments only for why a non-obvious choice exists.
11. Do not rename public symbols without listing the rename and checking call sites.
12. After editing, run the smallest relevant test first.
13. Provide a documentation report listing files changed, symbols documented, symbols renamed, tests run, and any skipped items.

## Project-specific additions

- Keep sorting code source-name blind. File paths are allowed for I/O and reports, not classification evidence.
- Do not add broad category power without a decoy test.
- Keep FX behind valid Drum and Instrument evidence unless measured concrete FX authority exists.
- Prefer small, reviewable architecture changes over one-file hacks.
