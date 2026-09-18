# Launches the native Windows build of rlviser so training running in WSL2
# can render to it over localhost UDP, with real GPU acceleration.
#
# Why: WSLg has no GPU-accelerated graphics path on this project's WSL2 setup
# (no /dev/dri render node, no working Vulkan ICD), so an rlviser launched
# *inside* WSL either crashes or renders a solid black window. rlviser only
# exchanges UDP packets with the training process, so running the identical
# tool natively here sidesteps WSLg's graphics stack entirely.
#
# This launches rlviser_windows_fixed.exe, NOT the official upstream release,
# because the official Windows build of rlviser has a real bug: an unconnected
# UDP socket that sends to a peer with nobody listening yet gets an ICMP
# "port unreachable" back, and Windows (unlike Linux) surfaces that as
# WSAECONNRESET (10054) on the socket's *next* recv() call. That silently
# kills rlviser's one long-lived UDP receive thread within milliseconds of
# every startup -- before the training process has even connected -- so no
# GameState packet is ever processed again for the rest of the run. Visible
# symptom: the ball renders (from whatever default state happened to load),
# but the stadium/goals/cars never spawn, because rlviser's game_mode never
# gets a chance to change from its TheVoid default. rlviser_windows_fixed.exe
# is a from-source build (source: ~/rlviser-debug-build in WSL, cross-compiled
# via cargo-xwin) with one addition: disabling that behavior via the
# documented SIO_UDP_CONNRESET socket ioctl, right after the UDP socket is
# bound. No other logic changed. Confirmed fix via a log-instrumented build:
# without it, the receive thread dies ~2ms after every startup; with it, the
# thread survives indefinitely and the full stadium/goals/cars spawn.
#
# This alone isn't sufficient on this machine, separately: WSL2's UDP
# loopback forwarding (WSL -> Windows via 127.0.0.1) doesn't currently work
# here, which rlviser_py relies on. scripts/watch_train.sh also starts
# scripts/wsl_udp_relay.py in WSL to work around that (see that script for
# details) -- this launcher alone is not enough, always go through
# watch_train.sh.
#
# Usage: run this first, leave the window open, then in WSL run
#   ./scripts/watch_train.sh

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ExePath = Join-Path $RepoRoot "rlviser_windows_fixed.exe"

if (-not (Test-Path $ExePath)) {
    throw "rlviser_windows_fixed.exe not found at $ExePath -- this is a custom from-source build (see comments above), not something to download. Rebuild it from ~/rlviser-debug-build in WSL via cargo-xwin, or see the project's rendering notes for the exact steps."
}

if (Get-Process rlviser_windows_fixed -ErrorAction SilentlyContinue) {
    Write-Host "rlviser_windows_fixed.exe is already running."
} else {
    Start-Process -FilePath $ExePath -WorkingDirectory $RepoRoot
    Write-Host "Launched rlviser_windows_fixed.exe. Leave it open, then run ./scripts/watch_train.sh in WSL."
}
