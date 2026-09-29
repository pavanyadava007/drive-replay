#!/usr/bin/env bash
# Bazel --workspace_status_command: stamps drop builds with the commit they were built from.
echo "STABLE_GIT_COMMIT $(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then echo "STABLE_GIT_DIRTY 1"; else echo "STABLE_GIT_DIRTY 0"; fi
