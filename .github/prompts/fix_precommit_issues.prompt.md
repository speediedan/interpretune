---
mode: "agent"
tools: ["githubRepo", " codebase", "file_search", "semantic_search", "read_file", "insert_edit_into_file", "create_file"]
---
Fix the pre-commit issues (especially 'Line too long' problems (by wrapping code so it is < 120 characters per line)) in
 the files and at the lines identified in this pre-commit message: `${input:pre_commit_message}`.
