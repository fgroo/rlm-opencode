# Git Workflow — Dual Remote Setup

## Remotes

| Remote | Repo | Visibility | Purpose |
|--------|------|------------|---------|
| `private` / `origin` | `fgroo/rlm-opencode-dev` | 🔒 Private | Default. Daily dev, experiments |
| `public` | `fgroo/rlm-opencode` | 🌍 Public | Releases, public-facing code |

> `origin` = `private` (both point to the same private repo). This makes `git push` default to private.

## Daily Usage

```bash
# Regular push (goes to private by default)
git push

# Push to public when ready for release
git push public main
```

## Quick Reference

```bash
# Push all branches to private
git push private --all

# Push a specific branch to public
git push public main

# Pull from either
git pull private main
git pull public main

# Check remotes
git remote -v
```

## Rules

1. **`git push` always goes to private** (safe default)
2. **Only push to public when code is clean** — no API keys, no sensitive data
3. **Feature branches stay private** until merged to main and reviewed
4. Before public push, double-check: `git grep -i "api.key\|secret\|password"`
