#!/usr/bin/env bash
set -euo pipefail

# One-way mirror of the Windows-side checkout of this repo into the WSL
# clone used for actually running training (see README: RocketSim is ~10x
# slower under /mnt/c, so the two checkouts are kept separate on purpose).
# Run this from WSL. Only copies tracked-looking source files -- never
# touches .git, the venv, or run-generated artifacts on either side.

WIN_SRC="${WIN_SRC:-/mnt/c/Developer/rocket-league-bot}"
WSL_DST="${WSL_DST:-$HOME/rocket-league-bot}"

rsync -av --no-perms \
  --exclude='.git/' \
  --exclude='.claude/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='wandb/' \
  --exclude='agent_controller_checkpoints/' \
  --exclude='shmem_flinks/' \
  --exclude='config.json' \
  --exclude='rlviser' \
  --exclude='rlviser.disabled' \
  --exclude='rlviser.exe' \
  --exclude='rlviser_windows_fixed.exe' \
  --exclude='*.dll' \
  --exclude='settings.txt' \
  "$WIN_SRC"/ "$WSL_DST"/

# The Windows checkout has core.autocrlf on, so anything copied from it
# arrives with CRLF line endings -- which makes every line of every text
# file show up as changed to a Linux git repo expecting LF. Strip CRLF from
# whatever git actually sees as touched (tracked-and-modified or untracked
# but not gitignored) -- scoped this way instead of a repo-wide `find` so it
# doesn't crawl .venv/wandb/checkpoints, which are large and irrelevant.
cd "$WSL_DST"
{ git diff --name-only; git ls-files --others --exclude-standard; } \
  | sort -u \
  | while IFS= read -r f; do
      [ -f "$f" ] && file "$f" | grep -q "CRLF" && sed -i 's/\r$//' "$f"
    done
true

# --no-perms above means new files land non-executable regardless of
# whatever drvfs reported for them on the Windows side; fix up the ones
# that actually need +x.
chmod +x scripts/*.sh 2>/dev/null || true
