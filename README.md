# PILOT

**P**roximal-**I**teration **L**earning **O**ver **T**ime

Reinforcement learning bot for Rocket League. Uses [RLGym v2](https://rlgym.org/)
with the [RocketSim](https://github.com/ZealanL/RocketSim) engine for the
environment, and [`rlgym-learn`](https://github.com/JPK314/rlgym-learn) +
`rlgym-learn-algos` for PPO training.

Background and rationale for the choices below live in
[design_choices.md](design_choices.md).

> **Linux only.** `rlgym-learn` 2.0.0's Rust env-process backend does not run on
> native Windows — it dies at startup with `FileExistsError: entity already
> exists` (details [below](#why-not-native-windows)). On Windows, run it inside
> **WSL2**; the setup is in [Windows (WSL2)](#windows-wsl2).

## Requirements

- Linux x86_64 (native, or **WSL2** on Windows — see below)
- Python 3.12
- An NVIDIA GPU (CUDA) for training. CPU works but the gradient step is slower.

## Windows (WSL2)

Native Windows is not supported (see [Why not native Windows](#why-not-native-windows)).
Run everything from a WSL2 Ubuntu shell instead. One-time setup:

1. **Install WSL2.** In an **elevated** PowerShell (Run as administrator):

   ```powershell
   wsl --install
   ```

   This enables the WSL2 + Virtual Machine Platform features and installs Ubuntu.
   **Reboot** when prompted. On the next boot Ubuntu opens and asks you to create
   a UNIX username and password. (If WSL was already partly installed, run
   `wsl --install -d Ubuntu-24.04` then `wsl --update`.)

2. **Confirm it's WSL version 2.** Back in PowerShell:

   ```powershell
   wsl -l -v          # the VERSION column must say 2
   ```

3. **Check CUDA reaches the VM.** The Windows NVIDIA driver exposes the GPU to
   WSL2 automatically — do **not** install an NVIDIA driver inside Ubuntu (it
   breaks the passthrough). From the Ubuntu shell:

   ```sh
   nvidia-smi         # should list your GPU
   ```

4. **Install build prerequisites** in Ubuntu:

   ```sh
   sudo apt update && sudo apt install -y git python3.12-venv build-essential
   ```

5. **Clone the repo into the Linux filesystem** — under `~`, *not*
   `/mnt/c/...`. RocketSim hammers the disk and the `/mnt/c` Windows bridge is
   an order of magnitude slower.

   ```sh
   cd ~
   git clone https://github.com/CPollreis/rocket-league-bot.git
   cd rocket-league-bot
   ```

Then follow [Setup](#setup) in that same Ubuntu shell.

### Why not native Windows

`rlgym-learn` 2.0.0 spins up its env processes through a Rust backend that uses
[`mio`](https://github.com/tokio-rs/mio) for socket multiplexing. It registers
each env's parent UDP socket with a short-lived handshake `mio::Poll`, then
re-registers that same socket with the main multiplexer `Poll` **without
deregistering it first**. `mio`'s Windows (IOCP) backend rejects a double
registration:

```rust
// mio/src/sys/windows/mod.rs
if self.inner.is_some() {
    Err(io::ErrorKind::AlreadyExists.into())
}
```

which surfaces in Python as:

```
FileExistsError: entity already exists
```

`mio`'s Linux (epoll) backend has no such guard, so the same code path works
there. This is independent of Python version, CUDA, and the `shmem_flinks`
directory. There is no released fix (`2.0.0` is the latest and matches `main`),
so WSL2 is the path on Windows.

## Setup

Run these on Linux (a WSL2 Ubuntu shell counts):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `torch==2.13.0+cu126` pin in `requirements.txt` pulls the CUDA 12.6 build
from the PyTorch download channel (via the `--extra-index-url` line in that
file), so an NVIDIA GPU with a driver new enough for CUDA 12.6 is required.

The training device is auto-detected: **CUDA** (`cuda:0`) if
`torch.cuda.is_available()`, otherwise **CPU**. Override with `RL_DEVICE` (see
[Environment variables](#environment-variables)). Verify CUDA is visible:

```sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## How to use

Activate the venv first (see [Setup](#setup)).

### Train

```sh
python quick_start.py    # full run: 2v2 self-play, 1B steps
python speed_test.py     # throughput benchmark: 1v1, short timeout, 10M step cap
```

Both regenerate `config.json` on startup, then start the `LearningCoordinator`.
Training progress prints to the terminal.

### Watch the sim in real time

`RL_RENDER=1` opens an `rlviser` 3D window on the first collection step, showing
one live match from env process 0. The other `RL_N_PROC - 1` processes stay
headless at full speed, so total throughput barely moves. Mouse and keys are
camera controls. Ctrl+C stops training and closes the window.

```sh
RL_RENDER=1 python quick_start.py              # 2v2, one rendered env + 7 headless
RL_RENDER=1 RL_N_PROC=1 python quick_start.py  # single real-time env, clean view, no training throughput
```

For the window to appear:

- **A local display.** Not over SSH; from inside a container only via the
  host-viewer workaround in
  [Watching a containerized run](#watching-a-containerized-run). Under **WSL2**
  the window goes through WSLg — it works out of the box on Windows 11 and
  recent Windows 10; if nothing appears, run `wsl --update` and confirm
  `echo $DISPLAY` is non-empty in the Ubuntu shell.

- **The `rlviser` window's system libraries.** `rlviser` is a Bevy app; a
  minimal Ubuntu (including a fresh WSL2 install) is missing at least
  `libxkbcommon-x11-0`, without which it panics at startup with `Library
  libxkbcommon-x11.so could not be loaded` before any window opens:

  ```sh
  sudo apt install -y libxkbcommon-x11-0
  ```

  Test the viewer on its own — `./rlviser` from the repo root should open a
  window (a bare scene, since nothing is feeding it yet); `Ctrl+C` to close.
  If that still panics on a different `lib*.so`, add the usual Vulkan/GL set:
  `sudo apt install -y libvulkan1 mesa-vulkan-drivers libgl1-mesa-dri libegl1`.

- **The `rlviser` binary present, and matching `rlviser-py`.** The viewer's UDP
  protocol changed after **v0.8.2**; `requirements.txt` pins
  `rlviser-py==0.6.13` (released the same day as `rlviser` v0.8.2). A newer
  viewer (v0.9.x) makes `rlviser-py` panic with `memory allocation of
  72057594037927944 bytes failed`. Get the v0.8.2 Linux build (this includes
  WSL2) and put it on `PATH` or in the repo root:

  ```sh
  curl -L -o rlviser https://github.com/VirxEC/rlviser/releases/download/v0.8.2/rlviser
  chmod +x rlviser
  ```

  `rlviser-py` launches it on the first render call. `Failed to launch RLViser
  (./rlviser)` means it is not finding it; the repo root is the most reliable
  place. A repo-root `rlviser` is git-ignored. Newer builds and other platforms
  are on the [RLViser releases](https://github.com/VirxEC/rlviser/releases) page
  (the v0.9.x Linux asset is `rlviser-x86_64-unknown-linux-gnu`, not `rlviser`).
- **A training device that does not crash.** CPU or CUDA. See
  [`RL_DEVICE`](#rl_device-where-the-gradient-step-runs).

RLViser draws the arena and cars from an `assets/` folder it builds from a
Rocket League installation; without one it falls back to a bare scene (ball and
boost pads, box-shaped cars). See [design_choices.md](design_choices.md) for how
rendering is wired and how to write a clean checkpoint-eval script.

### Get logs from the sim

Weights and Biases logging is on by default. Pick one:

```sh
wandb login                        # once, interactive
export WANDB_API_KEY=...            # non-interactive
export WANDB_MODE=offline           # log locally, no account
RL_WANDB=0 python quick_start.py    # disable entirely
```

`quick_start.py` runs land under the `cpollreis` entity by default; export
`WANDB_ENTITY` to override. Local run data is written to `wandb/`.

### Environment variables

All of these are read by both `quick_start.py` and `speed_test.py`. Set them
inline (`RL_N_PROC=12 python quick_start.py`) or `export` them for the shell
session.

#### `RL_DEVICE`: where the gradient step runs

Default: auto-detect (CUDA if present, else CPU).

| Value | Use it for |
| --- | --- |
| `cpu` | Any box with no NVIDIA GPU, or to compare. For this repo's tiny 3x256 MLPs CPU is often within noise of the GPU anyway (the bottleneck is RocketSim). |
| `cuda:0` | First NVIDIA GPU. What auto-detect picks when CUDA is available. |
| `cuda:1`, `cuda:2`, ... | A specific GPU on a multi-GPU box. Only one GPU is used; there is no multi-GPU training here. |

The string is passed straight to `torch.device(...)`, so anything torch accepts
works.

#### `RL_N_PROC`: parallel env processes

Default: `8`. Integer. Each process runs RocketSim (CPU physics) for one env and
streams timesteps back to the learner.

- Rough starting point: **(physical cores - 2)**.
- Higher = more steps/sec (up to the point the learner or RAM becomes the
  limit) and more RAM.
- `RL_N_PROC=1` with `RL_RENDER=1` gives one clean real-time match to watch, but
  collection then runs at wall-clock speed. Fine for eyeballing a policy, not
  for training.

#### `RL_RENDER`: RLViser window

Default: `0`. `1` renders env process 0 in an `rlviser` window (real-time
playback via `render_delay = 8/120`). See
[Watch the sim in real time](#watch-the-sim-in-real-time) for the binary and
display requirements. Any other value is treated as `0`.

#### `RL_WANDB`: Weights and Biases logging

Default: `1` (on). Set to exactly `0` to disable W&B entirely and fall back to
the plain console `PPOMetricsLogger`. Any value other than `0` leaves it on.

#### W&B configuration (only read when `RL_WANDB` != `0`)

| Variable | Default | Effect |
| --- | --- | --- |
| `WANDB_API_KEY` | unset | API key for non-interactive auth. Alternative to running `wandb login` once. |
| `WANDB_MODE` | unset | `online` (default behavior), `offline` (log to `wandb/` only, sync later with `wandb sync`), or `disabled` (W&B calls become no-ops). |
| `WANDB_ENTITY` | `cpollreis` in `quick_start.py`, account default in `speed_test.py` | Team / user the run lands under. `quick_start.py` only sets it if you have not, so exporting always wins. |

Both scripts put runs in the W&B group `rlgym-learn-testing`. The project and
run names come from the config in the script (`quick_start.py` logs to project
`rlgym-learn`).

#### Set automatically by the scripts (you should not need to touch these)

| Variable | Set to | Why |
| --- | --- | --- |
| `OPENBLAS_NUM_THREADS` | `1` | Stops numpy in each env process from spawning a thread pool and having the processes throttle each other. |

#### Examples

```sh
# Watch a single live match, no W&B noise
RL_RENDER=1 RL_N_PROC=1 RL_WANDB=0 python quick_start.py

# Full-speed throughput benchmark, log locally only
RL_N_PROC=12 WANDB_MODE=offline python speed_test.py

# Train on the second GPU with 24 env processes
RL_DEVICE=cuda:1 RL_N_PROC=24 WANDB_API_KEY=... python quick_start.py
```

## Docker image selection

The container is for **model training plus sim data collection** on a Linux box
with a CUDA GPU. RocketSim is CPU physics and runs fine inside a container. The
RLViser *window* cannot open inside the container (no display), but you can run
the viewer on the Linux host and have the containerized trainer stream game
state to it over loopback UDP, see
[Watching a containerized run](#watching-a-containerized-run).

Three reasonable base images, from least to most assembly:

| Base image | Use it when | Rough size | Notes |
| --- | --- | --- | --- |
| `pytorch/pytorch:<tag>-runtime` (premade) | You want GPU training with the least setup. **This repo's `Dockerfile` uses it.** | ~7 GB | torch + CUDA + cuDNN preinstalled and version-matched. Pick a tag whose torch matches `requirements.txt`. |
| `nvidia/cuda:<ver>-cudnn-runtime-ubuntu22.04` | You need system CUDA to compile CUDA kernels, or a CUDA / cuDNN combo PyTorch does not publish. | ~3 GB | You install Python and torch yourself. Most flexible, most work. |
| `python:3.12-slim` + CUDA torch wheel | Dev, CI, CPU-only sim rollouts, or one Dockerfile that also happens to run on GPU. | ~1.5 GB built | `pip install torch` pulls a wheel that bundles its own CUDA libs. Runs on GPU with `--gpus all` and a recent host driver. Cannot build CUDA extensions. |

Build and run the repo image:

```sh
docker build -t rl-bot .

# GPU box (NVIDIA Container Toolkit installed):
docker run --rm -it --gpus all -v "$(pwd)":/app \
  -e WANDB_API_KEY="$WANDB_API_KEY" rl-bot python quick_start.py

# No GPU (sim throughput check, dev):
docker run --rm -it -v "$(pwd)":/app \
  -e RL_DEVICE=cpu -e RL_WANDB=0 rl-bot python speed_test.py
```

Override the base tag (must match `requirements.txt`'s torch):

```sh
docker build --build-arg PYTORCH_TAG=2.13.0-cuda12.4-cudnn9-runtime -t rl-bot .
```

The bind mount (`-v "$(pwd)":/app`) keeps code edits, `config.json`,
`agent_controller_checkpoints/`, and `wandb/` on the host so they survive
`--rm`. Without it, mount at least a volume for the checkpoints folder.

### Watching a containerized run

**Requires a real Linux host** (not Docker Desktop). `rlviser-py` in the
container sends game state as UDP to `127.0.0.1:45243` and listens for viewer
commands on `127.0.0.1:34254`, both hardcoded to loopback. `--network host` puts
the container on the host's network namespace, so that loopback *is* the host's,
and an `rlviser` process running natively on the host receives the stream.
Docker Desktop routes `--network host` to its own Linux VM's loopback rather
than the real host, so it does not work there.

Two terminals **on the Linux GPU host**:

```sh
# Terminal 1 (host, outside the container): start the viewer, leave it running.
# It binds 127.0.0.1:45243 and waits for packets.
rlviser

# Terminal 2 (host, outside the container): start the trainer with host
# networking so its loopback UDP reaches the viewer above.
docker run --rm -it --network host --gpus all \
  -v "$(pwd)":/app \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e RL_RENDER=1 \
  -e RL_N_PROC=1 \
  rl-bot python quick_start.py
```

The window populates on the first collection step. Notes:

- `RL_N_PROC=1` gives one clean real-time match. Drop it (back to the `8`
  default) to train at full speed with env process 0 rendered and the other 7
  headless.
- The container prints `Failed to launch RLViser (...)` because `rlviser-py`
  still tries to spawn its own viewer binary and there is none on the image.
  Harmless: the render UDP packets are sent regardless, and Terminal 1's viewer
  is what draws them. To silence the message, add a no-op `rlviser` to the
  image (`printf '#!/bin/sh\nexec sleep infinity\n' > /usr/local/bin/rlviser &&
  chmod +x /usr/local/bin/rlviser` in a derived Dockerfile).
- Keep training on the GPU here: `RL_DEVICE` is left unset so it auto-detects
  `cuda:0`. Rendering does not change the device story.
- If the host has no physical display (headless server), run `rlviser` under a
  virtual framebuffer (`xvfb-run rlviser`) and view it over VNC, or copy
  checkpoints to a machine with a display and use a checkpoint-eval script
  instead (see [design_choices.md](design_choices.md)).

### More

See [design_choices.md](design_choices.md) for the Dockerfile internals, the
CPU-only slim variant, and what does and does not run in the container.

## Repo layout

| Path | Purpose |
| --- | --- |
| `quick_start.py` | Main training entry point. 2v2 self-play, trains for 1B steps. |
| `speed_test.py` | Throughput benchmark. Smaller 1v1 env, short timeout, 10M step cap. |
| `config.json` | Regenerated on every run from the config in the script. Safe to delete. |
| `agent_controller_checkpoints/` | Saved checkpoints. The trainer may wipe anything inside this folder. |
| `wandb/` | Local Weights and Biases run data. Git keeps the folder, ignores the contents. |
| `Dockerfile` | MVP training + sim container. |

## Checkpoints

Checkpoints land in `agent_controller_checkpoints/<run>/<step>/`. The trainer
owns that folder and may delete anything in it. To resume from one, set
`checkpoint_load_folder` in the PPO config in the script.
