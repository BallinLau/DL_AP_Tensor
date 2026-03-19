---
name: update-docs
description: Keep project documentation aligned with implementation changes. Use when code has been modified and README, API docs, or usage examples may be stale, especially after interface, behavior, configuration, dependency, or workflow updates.
---

# Update Docs

## Goal

Detect and fix documentation drift after code changes so README, API docs, and usage examples remain accurate in the same update.

## Workflow

1. Identify change scope.
- Inspect modified files and commits first.
- Prioritize public-facing changes: function signatures, endpoints, CLI flags, config keys, data format changes, and behavior changes.

2. Map code changes to doc targets.
- Update [README.md](/Users/ballinliu/Desktop/PHD/Project1/DL-APSSH/README.md) when setup, quickstart, configuration, feature list, or caveats changed.
- Update API docs for any public contract changes.
- Update usage examples for every changed public API or workflow.

3. Update README.
- Reflect current installation and execution steps.
- Sync option names, defaults, and output descriptions with source code.
- Keep at least one end-to-end quickstart path that is directly runnable.

4. Update API docs.
- Rewrite signatures, parameters, return values, error behavior, side effects, and constraints.
- Mark breaking changes explicitly and add migration notes when needed.
- Use exact names and argument order from code.

5. Update usage examples.
- Keep examples minimal but complete.
- Ensure imports, parameters, and expected outputs match current behavior.
- Prefer runnable examples over pseudo code; if pseudo code is required, label it clearly.

6. Verify consistency.
- Cross-check identifiers across code, README, API docs, and examples.
- Remove stale references to deleted options or APIs.
- Run available tests or doc checks when present.

7. Report changes clearly.
- Summarize which files were updated and why.
- Call out unresolved ambiguities and required follow-up.

## Output Checklist

- README reflects current behavior.
- API docs match public interfaces exactly.
- Examples run with current code paths.
- Breaking changes and migration notes are documented.
- No stale names, flags, or parameter defaults remain.

## Reference

Load [update-checklist.md](/Users/ballinliu/Desktop/PHD/Project1/DL-APSSH/skills/update-docs/references/update-checklist.md) when you need a detailed per-section review checklist or response template.
