# Skill: search

When the user asks to search for something in files or code, follow this procedure:

1. Run `grep -rn` with the search term
2. Present results in TWO formats:

## Raw output

```raw
(paste the exact grep output — file:line:match)
```

## Summary

```summary
Found N matches across M files:
- file1.py (lines 10, 25, 42): brief context of what matched
- file2.rs (line 7): brief context
Most matches are in: <most common file>
```

Always use `grep -rn --include="*.{py,rs,js,md}"` to limit to code/doc files.
