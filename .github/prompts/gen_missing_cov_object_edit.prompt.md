---
mode: "edit"
description: "Generate code to add missing test coverage for a source object to a test module"
---

### What to change

- We only want to add or extend tests in the target test module to fully cover the missing lines annotated in the provided object definition with '# MISSING' comments at the end of the relevant lines of the current implementation of our source code module.
- We do not want to make any changes to implementations in the source code module or object at this point,

### Target Source and Test Modules with starting Coverage State

Following this prompt, the user will provide following additional information to guide the generation of the test code:

- Test code module to update: `${input:test_code_module_name}`

- Fully qualified source code object reference for object with missing lines: `${input:source_code_object_name}`

- Annotated source code object with "# MISSING" lines for `${source_code_object_name}`

`${input:annotated_source_code_object}`
