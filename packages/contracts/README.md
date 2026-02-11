# Contracts Index

## Purpose
This directory is the canonical source of shared schemas and contracts.

## Ownership
- Changes require explicit planner tasks and tester coverage.
- Do not edit without updating affected tests and documentation.

## Structure
- `events/`: event payload schemas
- `features/`: feature flag or capability definitions
- `generated/`: generated artifacts (read-only)
- `prediction/`: prediction request/response schemas
- `session/`: session-related schemas
- `summary/`: summary/report schemas
- `tools/`: tool invocation schemas

## Versioning Policy
- Contracts must be backward compatible by default.
- Breaking changes require explicit approval and migration plan.
- Additive changes should include version notes in commit messages or run logs.
