# rocket-league-bot

RL bot for Rocket League. [RLGym v2](https://rlgym.org/) +
[RocketSim](https://github.com/ZealanL/RocketSim) for the env,
[`rlgym-learn`](https://github.com/JPK314/rlgym-learn) + `rlgym-learn-algos` for
PPO.

> **Linux only.** `rlgym-learn` 2.0.0's env-process backend crashes on startup on
> native Windows (`FileExistsError: entity already exists`). On Windows, run it in
> WSL2 — setup below.

## Quickstart

Every command, in order. Run it all in a Linux shell (a WSL2 Ubuntu shell
counts). The rest of this README is reference and troubleshooting for when a step
misbehaves.

```sh
# Windows only, first: run `wsl --install` in an admin PowerShell, reboot, then
# open the Ubuntu shell and run everything below there. Native Linux: start here.

# 1. System deps: build tools + the libs the rlviser window needs
sudo apt update && sudo apt install -y \
  git python3.12-venv build-essential \
  libxkbcommon-x11-0 libvulkan1 mesa-vulkan-drivers libgl1-mesa-dri libegl1

# 2. Clone into the Linux filesystem (NOT /mnt/c — RocketSim is ~10x slower there)
cd ~ && git clone <your-repo-url> rocket-league-bot && cd rocket-league-bot

# 3. Python env + deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Confirm CUDA is visible (expect: True <your GPU name>)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 5. rlviser viewer binary — only needed to watch the sim (RL_RENDER=1)
curl -L -o rlviser https://github.com/VirxEC/rlviser/releases/download/v0.8.2/rlviser
chmod +x rlviser

# 6. Weights & Biases login, once (or skip with: export WANDB_MODE=offline)
wandb login

# 7. Train
python quick_start.py

# ...or train while watching one match in the rlviser window
RL_RENDER=1 RL_N_PROC=1 python quick_start.py
```

Coming back in a new shell later: `cd ~/rocket-league-bot && source .venv/bin/activate`
before running anything.

## Requirements

- Linux x86_64 (native or WSL2)
- Python 3.12
- NVIDIA GPU for training (CPU works, the gradient step is just slower)

## Windows: WSL2 setup

Native Windows won't work ([why](#why-not-native-windows)). Do everything from a
WSL2 Ubuntu shell.

1. **Install WSL2** in an admin PowerShell, then reboot:

   ```powershell
   wsl --install
   ```

   Ubuntu opens on reboot and asks for a UNIX username/password. If WSL was
   already half-installed: `wsl --install -d Ubuntu-24.04 && wsl --update`.

2. **Check it's version 2:**

   ```powershell
   wsl -l -v          # VERSION column says 2
   ```

3. **Check the GPU reaches WSL.** The Windows NVIDIA driver handles this — don't
   install a driver inside Ubuntu, it breaks passthrough.

   ```sh
   nvidia-smi         # lists your GPU
   ```

4. **Install build deps:**

   ```sh
   sudo apt update && sudo apt install -y git python3.12-venv build-essential
   ```

5. **Clone into the Linux filesystem** — under `~`, not `/mnt/c/...`. RocketSim
   hammers the disk and `/mnt/c` is ~10x slower.

   ```sh
   cd ~ && git clone <your-repo-url> rocket-league-bot && cd rocket-league-bot
   ```

Then do [Setup](#setup) in that same shell.

### Why not native Windows

`rlgym-learn` 2.0.0's Rust backend registers an env's UDP socket with one poll
instance, then re-registers the same socket with another without deregistering
first. Windows' IOCP backend rejects the double registration; Linux's epoll
doesn't care. You see `FileExistsError: entity already exists`. No fix released
(2.0.0 is latest), so WSL2 it is.

## Setup

On Linux (a WSL2 shell counts):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch==2.13.0+cu126` pulls the CUDA 12.6 build, so you need an NVIDIA driver new
enough for CUDA 12.6. The device is auto-detected — CUDA if available, else CPU
(override with `RL_DEVICE`). Check CUDA:

```sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

CPU-only box? Install the CPU wheel instead of the default:

```sh
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

## Running

Activate the venv first.

### Train

```sh
python quick_start.py    # full run: 2v2 self-play, 1B steps
python speed_test.py     # throughput benchmark: 1v1, short timeout, 10M steps
```

Both rewrite `config.json` on startup, then start the `LearningCoordinator`.
Progress prints to the terminal.

### Watch the sim

`RL_RENDER=1` opens an `rlviser` window on the first collection step showing env
process 0. The other `RL_N_PROC - 1` envs stay headless at full speed, so
training barely slows. Mouse and keys are the camera; `rlviser`'s own
speed/pause buttons only affect local playback, not the trainer. Ctrl+C stops
everything.

```sh
RL_RENDER=1 RL_N_PROC=1 python quick_start.py                   # one clean env, no training throughput
RL_RENDER=1 python quick_start.py                               # env 0 rendered, still trains
RL_RENDER=1 RL_N_PROC=1 RL_RENDER_FPS=15 python quick_start.py  # real-time playback
```

**`RL_RENDER_FPS` (default 60)** — frames/sec sent to `rlviser`. One env step is
8/120 s of game time, so playback runs at `RL_RENDER_FPS / 15` x real time (15 =
real time, 60 = 4x and smoother-looking). It's a target — env 0 only hits it if
it's not fighting the other envs for CPU, so watch with `RL_N_PROC=1`. `rlviser`
interpolates between frames, so a steady feed beats a high one. If it stutters
instead of just running slow, drop `RL_RENDER_FPS` or `RL_N_PROC`.

**Still stuck at ~12 fps no matter what?** That's `rlviser` drawing in software,
not the feed. It's a Bevy/`wgpu` app — if `wgpu` can't get a GPU it falls back to
`llvmpipe` and any 3D scene crawls at ~12 fps. Common in WSL2 (WSLg's Vulkan is
`dzn` over D3D12 and `wgpu` often rejects it). Fixes:

- Confirm it: `sudo apt install -y mesa-utils vulkan-tools`, then `glxinfo -B`.
  `renderer string: llvmpipe` means software.
- Force the GL backend instead of Vulkan: `WGPU_BACKEND=gl ./rlviser` and
  `WGPU_BACKEND=gl RL_RENDER=1 python quick_start.py`. WSLg's GL path (the
  `d3d12` Mesa driver) is GPU-backed and enough for this scene.
- On Windows, the best option is the native `rlviser.exe` on the Windows GPU with
  training in WSL2 — see [Watching from another machine](#watching-from-another-machine).
- `rlviser` writes `settings.txt` (vsync, fps_limit, shadows, msaa). Shadows and
  msaa are already off; the software renderer is the ceiling, not these.

**First-time `rlviser` setup:**

- **The binary.** `rlviser-py==0.6.13` (pinned) only talks to `rlviser`
  **v0.8.2**. Newer viewers (v0.9.x) make it panic with a nonsense `memory
  allocation of 72057594037927944 bytes failed`. Grab v0.8.2 and drop it in the
  repo root (git-ignored there):

  ```sh
  curl -L -o rlviser https://github.com/VirxEC/rlviser/releases/download/v0.8.2/rlviser
  chmod +x rlviser
  ```

  `rlviser-py` launches it on the first render. `Failed to launch RLViser
  (./rlviser)` means it can't find it; the repo root is the safe spot.

- **System libs.** A fresh WSL2 / minimal Ubuntu is missing `libxkbcommon-x11-0`
  — without it `rlviser` panics with `Library libxkbcommon-x11.so could not be
  loaded` before the window opens:

  ```sh
  sudo apt install -y libxkbcommon-x11-0
  ```

  If it then panics on a different `lib*.so`:
  `sudo apt install -y libvulkan1 mesa-vulkan-drivers libgl1-mesa-dri libegl1`.

- **A display.** Not over plain `ssh` (X forwarding won't work — see
  [Watching from another machine](#watching-from-another-machine)). WSLg gives
  you one out of the box on Win11 / recent Win10; if nothing shows, run
  `wsl --update` and check `echo $DISPLAY` is set.

Test the viewer alone: `./rlviser` from the repo root opens a bare scene (ball,
boost pads, box-shaped cars — it needs a Rocket League `assets/` folder for real
models). Ctrl+C to close.

### Evaluating a checkpoint

`RL_RENDER=1` shows env 0 mid-collection — stochastic actions, a reset on every
goal or timeout. For a clean read on how good the bot actually is, write a short
script that loads the newest `agent_controller_checkpoints/<run>/<step>/`,
rebuilds the env with `RLViserRenderer`, and steps it with greedy actions. It
runs on its own, separate from any training run.

### Logs (Weights & Biases)

On by default. Pick one:

```sh
wandb login                        # once, interactive
export WANDB_API_KEY=...            # non-interactive
export WANDB_MODE=offline           # local only, no account
RL_WANDB=0 python quick_start.py    # off entirely
```

`quick_start.py` runs go under the `cpollreis` entity (export `WANDB_ENTITY` to
change). Local data lands in `wandb/`.

## Environment variables

Read by both scripts. Set inline (`RL_N_PROC=12 python quick_start.py`) or
`export` them for the shell session.

| Var | Default | What it does |
| --- | --- | --- |
| `RL_DEVICE` | auto | `cpu`, `cuda:0`, `cuda:1`, ... — passed straight to `torch.device()`. Auto picks `cuda:0` if CUDA is there, else `cpu`. The nets are tiny 3x256 MLPs, so CPU is often within noise of the GPU (RocketSim is the bottleneck). No multi-GPU. |
| `RL_N_PROC` | `8` | Parallel env processes. Start around **(physical cores − 2)**. Higher = more steps/sec and more RAM, up to where the learner or RAM caps it. Use `1` when watching. |
| `RL_RENDER` | `0` | `1` renders env 0 in an `rlviser` window. Anything else is off. |
| `RL_RENDER_FPS` | `60` | Only with `RL_RENDER=1`. Frames/sec to `rlviser`; sets `render_delay` and the renderer's `tick_rate`. Playback speed = `RL_RENDER_FPS / 15` x real time. Doesn't fix software-rendering choppiness (see [Watch the sim](#watch-the-sim)). |
| `RL_WANDB` | `1` | Exactly `0` disables W&B and uses the plain console logger. Anything else is on. |

**W&B vars** (only when `RL_WANDB` ≠ `0`): `WANDB_API_KEY` for non-interactive
auth, `WANDB_MODE` (`online` / `offline` / `disabled`), `WANDB_ENTITY` for the
team/user — `cpollreis` in `quick_start.py`, account default in `speed_test.py`,
and exporting always wins. Both scripts use W&B group `rlgym-learn-testing`;
`quick_start.py` logs to project `rlgym-learn`.

The scripts also set `OPENBLAS_NUM_THREADS=1` themselves so numpy in each env
process doesn't spawn a thread pool and throttle the others. Leave it alone.

### Examples

```sh
# Watch one match, no W&B
RL_RENDER=1 RL_N_PROC=1 RL_WANDB=0 python quick_start.py

# Throughput benchmark, log locally
RL_N_PROC=12 WANDB_MODE=offline python speed_test.py

# Second GPU, 24 envs
RL_DEVICE=cuda:1 RL_N_PROC=24 WANDB_API_KEY=... python quick_start.py
```

## Docker

For training + sim collection on a Linux CUDA box. The `Dockerfile` uses the
premade `pytorch/pytorch:<tag>-runtime` image (torch + CUDA + cuDNN already
version-matched) — pick a tag whose torch matches `requirements.txt`. Build it on
a Linux CUDA host or in CI; cross-arch builds run the pip step under emulation
and are slow.

```sh
docker build -t rl-bot .

# GPU box (NVIDIA Container Toolkit installed):
docker run --rm -it --gpus all -v "$(pwd)":/app \
  -e WANDB_API_KEY="$WANDB_API_KEY" rl-bot python quick_start.py

# No GPU (sim throughput check):
docker run --rm -it -v "$(pwd)":/app \
  -e RL_DEVICE=cpu -e RL_WANDB=0 rl-bot python speed_test.py
```

Override the base tag (must match `requirements.txt`'s torch):

```sh
docker build --build-arg PYTORCH_TAG=2.13.0-cuda12.4-cudnn9-runtime -t rl-bot .
```

The bind mount (`-v "$(pwd)":/app`) keeps code, `config.json`, checkpoints, and
`wandb/` on the host so they survive `--rm`. Without it, at least mount a volume
for the checkpoints folder.

Check what a tag actually ships, and that the GPU reaches the container:

```sh
docker run --rm pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime \
  python -c "import torch, sys; print(sys.version); print(torch.__version__, torch.version.cuda)"
docker run --rm -it --gpus all rl-bot \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### CPU-only / slim image

Smaller base, no premade PyTorch. Save as `Dockerfile.slim`, build with
`docker build -f Dockerfile.slim -t rl-bot .`. On `linux/amd64` the pip step
pulls the CUDA torch wheel (works with `--gpus all` and a recent host driver);
add `--index-url https://download.pytorch.org/whl/cpu` to the torch install for a
pure-CPU image.

```dockerfile
FROM python:3.12-slim-bookworm
ENV PIP_NO_CACHE_DIR=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
CMD ["python", "quick_start.py"]
```

### Watching a containerized run

The training command runs **inside** the container fine. The `rlviser` **window**
can't — no display. So run the viewer on the host and let the container stream to
it.

**Real Linux host only** (not Docker Desktop). `rlviser-py` sends state to
`127.0.0.1:45243` and takes viewer commands on `127.0.0.1:34254`, both hardcoded
to loopback. `--network host` puts the container on the host's network namespace
so that loopback *is* the host's, and a native `rlviser` on the host picks up the
stream. Docker Desktop routes `--network host` to its own VM, so it doesn't work
there.

Two terminals on the host:

```sh
# 1: start the viewer, leave it running
rlviser

# 2: trainer with host networking
docker run --rm -it --network host --gpus all -v "$(pwd)":/app \
  -e WANDB_API_KEY="$WANDB_API_KEY" -e RL_RENDER=1 -e RL_N_PROC=1 \
  rl-bot python quick_start.py
```

Notes:

- The container logs `Failed to launch RLViser` because `rlviser-py` still tries
  to spawn its own binary and there's none on the image. Harmless — the UDP
  packets go out anyway and terminal 1's viewer draws them.
- Training stays on GPU (`RL_DEVICE` unset → `cuda:0`). Rendering doesn't change
  that.
- Headless host: run `xvfb-run rlviser` and view over VNC, or copy checkpoints to
  a machine with a display and eval there
  ([Evaluating a checkpoint](#evaluating-a-checkpoint)).

### Watching from another machine

The trainer and the `rlviser` window don't have to be on the same machine or OS.
`rlviser-py` just emits a UDP stream to `127.0.0.1:45243` — get that to an
`rlviser` with a GPU and it renders.

**Plain X forwarding (`ssh -X` / `-Y`) won't work.** `rlviser` needs a local
Vulkan / GL-core context; `ssh -X` gives indirect GLX, which `wgpu` can't use —
it errors or falls back to network software rendering. Don't bother.

Run `rlviser` where the GPU is and move the stream to it instead:

- **Native `rlviser.exe` on Windows + trainer in WSL2** — renders on the Windows
  GPU, the smoothest local option. WSL2's loopback isn't Windows' loopback, so
  either turn on mirrored networking (Win11 22H2+: add `networkingMode=mirrored`
  under `[wsl2]` in `.wslconfig`, which shares `127.0.0.1`), or bridge the port
  with `socat` like below.

- **Remote GPU box → laptop.** SSH only forwards TCP, so bridge UDP↔TCP at both
  ends (view-only, no camera control back):

  ```sh
  # laptop
  socat -u TCP-LISTEN:45243,reuseaddr,fork UDP:127.0.0.1:45243 &
  rlviser
  ssh -R 45243:localhost:45243 user@gpubox

  # gpubox, in that ssh session
  socat -u UDP-RECV:45243 TCP:localhost:45243 &
  RL_RENDER=1 RL_N_PROC=1 python quick_start.py
  ```

- **Or just don't stream.** `rsync` the latest
  `agent_controller_checkpoints/<run>/<step>/` down and eval it locally
  ([Evaluating a checkpoint](#evaluating-a-checkpoint)). Smooth, local GPU, no
  tunnel, and you watch a clean greedy policy instead of mid-collection env 0.

## Repo layout

| Path | What |
| --- | --- |
| `quick_start.py` | Main training entry point. 2v2 self-play, 1B steps. |
| `speed_test.py` | Throughput benchmark. 1v1, short timeout, 10M steps. |
| `config.json` | Rewritten from the script every run. Safe to delete. |
| `agent_controller_checkpoints/` | Saved checkpoints. The trainer may wipe anything in here. |
| `wandb/` | Local W&B data. Folder tracked, contents ignored. |
| `Dockerfile` | Training + sim container. |

## Checkpoints

They land in `agent_controller_checkpoints/<run>/<step>/`. The trainer owns that
folder and may delete anything in it. To resume from one, set
`checkpoint_load_folder` in the PPO config in the script.
