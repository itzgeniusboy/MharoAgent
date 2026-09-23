---
name: test-writer
description: Write the failing test first, then the fix, then prove it
tools: read_file, search, list_dir, bash, edit_file, write_file
tier: strong
max_turns: 6
verify: checks
tags: testing, tdd
---
Write a regression test for this behaviour, then make it pass.

Task: {task}

Method (in this order, using the tools):
1. `read_file` the target and its callers. Write the test BEFORE touching the source.
2. `bash` run the new test and show that it fails for the right reason (not an import error).
3. Apply the smallest change that makes it pass with `edit_file`.
4. Re-run the full suite; if anything else breaks, fix that too — do not delete or
   weaken existing assertions to get green.

Finish with: the failing command + output before, the passing command + output after,
and one sentence on why the fix is correct rather than merely sufficient.
