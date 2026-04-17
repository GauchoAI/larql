# Skill: list

You have a tool called `list`. When the user asks to list files, see directory contents, or explore a folder, call it by outputting EXACTLY:

```tool
list <path>
```

Where `<path>` is the directory to list (e.g., ~/Desktop, ., /tmp).

The tool returns structured output. You will receive a summary of the results.
Use the summary to answer the user's question or provide commentary.

Do NOT run `ls` yourself. Always use the `list` tool.
