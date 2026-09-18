#!/usr/bin/env bash
set -euo pipefail

# Last-resort fallback for watching training when there's no Windows GUI to
# run rlviser.exe natively (see scripts/watch_train.sh, the preferred path).
#
# On WSL2 setups where rlviser can't get a hardware-accelerated GPU context
# (no /dev/dri render node, no working d3d12/nvidia Vulkan ICD), EGL falls
# back to dri2 then zink and fails ("ZINK: failed to choose pdev"), so the
# render window never opens at all. Forcing Mesa's software rasterizer
# sidesteps that crash -- but confirmed by hand: even when it doesn't crash,
# WSLg's "[WARN:COPY MODE]" RDP fallback often just shows a solid black
# window, and rlviser can still segfault after a few minutes under sustained
# software-rendering load. Treat this as a "does it stay alive" smoke test,
# not a reliable way to actually watch a match.
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe

export RL_RENDER=1
export RL_N_PROC="${RL_N_PROC:-1}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python quick_start.py "$@"
