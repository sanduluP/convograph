# 0001: Monorepo over git submodules

## Status

Accepted (2026-08-28).

## Context

Three modules are authored by three different people, delivered incrementally, often as a zip export rather than a live git repo. We need a structure that makes folding in a new/updated module low-friction.

## Decision

Use a single monorepo with one top-level folder per module (`modules/<name>/`), rather than git submodules or separate repos per module.

## Rationale

- Contributors don't yet have independent repos with their own release cadence — most handoffs are zip files, not git remotes. Submodules add detached-HEAD / `--init --recursive` friction without a matching benefit at this stage.
- A single `git clone` gets everyone the whole pipeline; no separate auth/permissions per module.
- Module boundaries are enforced by convention (self-contained folder, own README/deps) and by the schemas in `schemas/`, not by git repo boundaries.
- Large binary data in module 2 is handled with Git LFS scoped to that module's `.gitattributes`, so it doesn't bloat clones of the whole repo disproportionately more than a submodule would.

## Revisit if

A module's author starts iterating independently at a pace that makes copy-in updates painful — at that point, `git subtree` (already documented in `scripts/import_module.sh`) or a switch to submodules for that specific module is the escape hatch, without needing to restructure the rest of the repo.
