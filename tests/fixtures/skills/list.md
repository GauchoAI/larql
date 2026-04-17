# Skill: list

When the user asks to list files, directories, or contents, follow this procedure:

1. Run the appropriate bash command (ls, find, etc.)
2. Present the output in TWO formats:

## Raw output

Show the raw command output in a ```raw``` block:

```raw
(paste the exact terminal output here)
```

## Summary

Analyze the output and provide a human-friendly summary in a ```summary``` block:

```summary
Found N files in /path:
- X source files (.py, .rs, .js)
- Y documentation files (.md)
- Z configuration files (.toml, .yaml, .json)
Total size: approximately S
```

## Example

User: "list the files in the project"

Response:

```bash
ls -la ~/project
```

Then after seeing the output:

```raw
total 48
drwxr-xr-x  8 user staff  256 Apr 17 src/
-rw-r--r--  1 user staff 1234 Apr 17 Cargo.toml
-rw-r--r--  1 user staff  567 Apr 17 README.md
```

```summary
Found 3 items in ~/project:
- 1 directory (src/)
- 1 build config (Cargo.toml)
- 1 documentation file (README.md)
```
