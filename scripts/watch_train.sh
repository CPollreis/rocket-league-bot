#!/usr/bin/env bash
set -euo pipefail

# Watches training via a *native* Windows rlviser.exe instead of one launched
# inside WSL. Confirmed by hand: on this project's WSL2 setup there is no
# GPU-accelerated graphics path at all (no /dev/dri render node, no working
# Vulkan ICD), so a WSL-launched rlviser either crashes or renders a solid
# black window through WSLg's "[WARN:COPY MODE]" RDP fallback -- see
# scripts/watch_train_wsl_software.sh for that (unreliable) path.
#
# Running rlviser.exe natively on Windows sidesteps WSLg's graphics stack
# entirely: rlviser only needs to exchange UDP packets with this process.
#
# Two separate bugs previously made this render only a bare ball with no
# stadium/goals/cars, both now worked around by this script:
#
#   1. The official Windows rlviser build silently loses its one UDP receive
#      thread to a Windows-specific WSAECONNRESET a couple ms after every
#      startup (see scripts/launch_rlviser_windows.ps1 for the full
#      explanation), so no packet is ever processed again. Fixed by using a
#      custom-built rlviser_windows_fixed.exe instead -- see that ps1 script.
#   2. On this machine, WSL2's UDP loopback forwarding (WSL -> Windows via
#      127.0.0.1, which rlviser_py always targets) isn't relaying traffic --
#      confirmed with a raw packet test: identical bytes sent to the WSL
#      host's real gateway IP arrive at a Windows listener instantly, the
#      same bytes sent to 127.0.0.1 never do. Worked around at the
#      application layer (not a kernel/iptables NAT rule -- that hits a
#      separate Linux route_localnet restriction on DNAT-ing loopback-
#      destined traffic) with scripts/wsl_udp_relay.py: it listens on
#      127.0.0.1 (which works fine for purely-local delivery inside WSL) and
#      forwards every packet on to the real host IP (which we confirmed
#      works). No sudo needed for the relay itself.
#
# This script launches rlviser_windows_fixed.exe on Windows itself (via WSL
# interop, which is on by default) if it isn't already running, and starts
# the relay, so there's nothing to run manually first -- just run this
# script.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Don't depend on the caller having activated .venv themselves -- use its
# python directly if present, so this works from a plain fresh shell too.
if [ -x .venv/bin/python ]; then
    PYTHON_BIN="$(pwd)/.venv/bin/python"
elif command -v python3 > /dev/null 2>&1; then
    PYTHON_BIN=python3
else
    PYTHON_BIN=python
fi

# Move the local Linux rlviser binary out of the way so rlviser_py doesn't
# also try to spawn a redundant (broken) local copy. A missing binary just
# makes that spawn attempt fail non-fatally; training and rendering to the
# native Windows instance continue normally.
if [ -f rlviser ]; then
    mv rlviser rlviser.disabled
    echo "[watch_train] moved local ./rlviser to ./rlviser.disabled (rendering via native Windows rlviser.exe instead)"
fi

# Launch (or confirm already running) the native Windows rlviser.exe. The
# ps1 script itself checks for an existing process and skips launching a
# second one, so this is safe to run every time. WIN_REPO points at the
# Windows-side checkout of this repo (see scripts/sync_to_wsl.sh) -- that's
# where launch_rlviser_windows.ps1 downloads/keeps rlviser.exe.
WIN_REPO="${WIN_REPO:-/mnt/c/Developer/rocket-league-bot}"
if [ -d "$WIN_REPO" ] && command -v powershell.exe > /dev/null 2>&1; then
    # -ExecutionPolicy Bypass is scoped to just this one process -- it does
    # not change the user's persistent PowerShell execution policy, unlike
    # Set-ExecutionPolicy. Needed because the default policy on this machine
    # blocks running local .ps1 files at all.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$WIN_REPO/scripts/launch_rlviser_windows.ps1")"
    sleep 2  # give rlviser.exe a moment to open its UDP socket on first launch
else
    echo "[watch_train] couldn't reach $WIN_REPO via WSL interop -- launch it yourself:" >&2
    echo "  powershell -ExecutionPolicy Bypass -File scripts\\launch_rlviser_windows.ps1" >&2
fi

export RL_RENDER=1
export RL_N_PROC="${RL_N_PROC:-1}"
unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER 2>/dev/null || true

# Start the WSL-side relay (see the top-of-file explanation) so rlviser_py's
# 127.0.0.1 traffic actually reaches the Windows rlviser process.
"$PYTHON_BIN" scripts/wsl_udp_relay.py 45243 > /tmp/wsl_udp_relay.log 2>&1 &
RELAY_PID=$!

# Close the rlviser window and relay process once training exits -- whether
# that's a normal finish or Ctrl+C (which quick_start.py itself already
# handles gracefully, saving a checkpoint first; this only runs after that
# finishes). Not `exec`ing python above is what lets this trap fire.
cleanup() {
    kill "$RELAY_PID" > /dev/null 2>&1 || true
    command -v powershell.exe > /dev/null 2>&1 || return 0
    powershell.exe -NoProfile -Command \
        "Get-Process rlviser_windows_fixed -ErrorAction SilentlyContinue | Stop-Process -Force" \
        > /dev/null 2>&1 || true
    echo "[watch_train] closed rlviser_windows_fixed.exe and the UDP relay"
}
trap cleanup EXIT

"$PYTHON_BIN" quick_start.py "$@"
