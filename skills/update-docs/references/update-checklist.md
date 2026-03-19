# Update Checklist

Use this checklist when synchronizing documentation after code updates.

## 1) README Sync

- Update project purpose or feature bullets if capability changed.
- Update install/run commands if dependencies, entrypoints, or flags changed.
- Update configuration section for new/removed keys and default values.
- Update outputs, metrics, and known limitations if behavior changed.
- Keep one quickstart path fully runnable.

## 2) API Doc Sync

- Update public signatures (name, parameters, defaults, return type/value).
- Update request/response schemas for endpoint or data contract changes.
- Document validation, error cases, and side effects.
- Add versioning or migration note for breaking changes.

## 3) Example Sync

- Update imports, constructors, and call signatures.
- Update sample inputs/outputs to match current behavior.
- Remove examples for deleted APIs and add replacements.
- Prefer executable snippets over pseudo code.

## 4) Final Cross-Check

- Ensure terms and names are consistent across code, README, API docs, examples.
- Ensure no stale flags, parameter names, or old paths remain.
- Run available tests or lint/doc checks when practical.

## Response Template

Use this concise structure when reporting work:

1. Files updated.
2. What changed in README.
3. What changed in API docs.
4. What changed in examples.
5. Validation run and remaining risks.
