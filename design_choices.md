# Design choices

Background and rationale that does not need to be in the README. Placeholder
name for now.

## Device and platform

RocketSim environments are CPU only. The GPU (CUDA) is only used for the actor /
critic gradient step, and the networks here are small 3x256 MLPs, so the sim is
usually the bottleneck rather than the device. `cuda` and `cpu` can be within
noise of each other here; benchmark both with `RL_DEVICE`.

### CUDA

The device is auto-detected as `cuda:0` when `torch.cuda.is_available()`,
otherwise `cpu`. Verify CUDA:

```sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On Windows and Linux x86_64, `torch==2.13.0` resolves to a CUDA wheel. For a
CPU-only box, install the CPU wheel instead:

```sh
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

### Other notes

- Observations are cast to `float32` (from `DefaultObs`'s `float64`) to match the
  network dtype and halve obs memory / bandwidth.
- The scripts set `OPENBLAS_NUM_THREADS=1` in-process to stop numpy from
  over-allocating in env processes and throttling each other.
- The RLViser render window needs a local display: run training natively (not
  over SSH, not inside a container). See "Watching a containerized run" in the
  README for the host-viewer workaround on a Linux host.

## Watching the sim during training

Rendering is undocumented upstream, so here is the whole picture.

### The moving parts

| Piece | What it is |
| --- | --- |
| `RLViserRenderer` | An rlgym class (`rlgym/rocket_league/rlviser/`). Each frame it packs the game state into a UDP packet and sends it to localhost. It is passed to `RLGym(renderer=RLViserRenderer())` in both scripts. |
| `rlviser-py` | Thin Python / Rust shim that owns the UDP socket and launches the viewer binary the first time a render function is called. |
| `rlviser` | The standalone 3D viewer window (a Bevy app by VirxEC). This is what you actually watch. Download it from <https://github.com/VirxEC/rlviser/releases>. |

`rlviser-py` looks for a binary named `rlviser` (`rlviser.exe` on Windows) on
your `PATH` first, then in the working directory. If you see
`Failed to launch RLViser`, put the binary in the repo root.

The binary and `rlviser-py` must agree on the UDP wire format. `requirements.txt`
pins `rlviser-py==0.6.13`, which matches `rlviser` **v0.8.2**. A newer viewer
(v0.9.x) changed the protocol and makes `rlviser-py` panic with a nonsense
`memory allocation of 72057594037927944 bytes failed`. On a minimal Linux /
WSL2 the viewer also needs `libxkbcommon-x11-0` (panics with `Library
libxkbcommon-x11.so could not be loaded` without it), and sometimes
`libvulkan1` / `mesa-vulkan-drivers` / `libgl1-mesa-dri`.

RLViser draws the arena and car models from an `assets/` folder it builds from a
Rocket League installation. Without one it renders a bare scene: the ball, the
boost pads, and box-shaped cars.

### How training drives it

`RL_RENDER=1` sets `render=True` and `render_delay=8/120` on the process config.
Inside `rlgym-learn`:

- Only **env process 0** renders (`render_this_proc = proc_idx == 0 and render`).
  The other `RL_N_PROC - 1` processes stay headless at full speed.
- Process 0 calls `env.render()` every env step and sleeps `render_delay`
  (about 66 ms), so it plays back at roughly real time. It collects timesteps
  slower than its siblings, but it is 1 of `RL_N_PROC`, so total throughput
  barely moves.
- You are watching **one 2v2 match** out of the pool, live, while the learner
  trains on experience pooled from all of them.

In the `rlviser` window, mouse and keys are camera controls. Its speed / pause
buttons only affect local playback interpolation; they do not pause the trainer
(the renderer never reads `get_game_speed()` / `get_game_paused()`).

### Evaluating a checkpoint cleanly

Watching process 0 means watching a policy mid-data-collection (stochastic
actions, resets on goal / timeout). To evaluate "how good is the model right
now" cleanly, write a separate eval script that loads the latest folder under
`agent_controller_checkpoints/<run>/<step>/`, rebuilds the env with
`RLViserRenderer`, and steps it with greedy actions. That runs independently of
training.

## Docker

### The image in this repo (MVP)

`Dockerfile` is the minimum that gives good support for both training and the
sim:

- `ARG PYTORCH_TAG` / `FROM pytorch/pytorch:...-runtime` gets a working
  torch + CUDA + cuDNN with no version juggling.
- `apt-get install build-essential cmake ...` plus a Rust toolchain covers
  source builds of `rocketsim` and `rlgym-learn` when pip has no wheel for the
  platform.
- The pip step strips the `torch==` line from `requirements.txt` so it never
  touches the base image's matched torch build, then prints the resolved
  torch / CUDA versions so a mismatch fails the build loudly.
- `COPY . .` is only a fallback for `docker run` with no mount; normal use bind
  mounts the working tree.

Check what a premade PyTorch tag actually ships before committing to it:

```sh
docker run --rm pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime \
  python -c "import torch, sys; print(sys.version); print(torch.__version__, torch.version.cuda)"
```

Tag list: <https://hub.docker.com/r/pytorch/pytorch/tags>. The base image's
Python (3.11 on recent tags) is fine; nothing here hard-requires 3.12.

### Building the image

Build on the Linux CUDA host or in CI, where the base image is native. Building
for `linux/amd64` on another architecture works but runs the pip step under
emulation and is slow.

### What runs in the container

| Workload | In a container |
| --- | --- |
| PPO training on GPU | Yes, with `--gpus all`. |
| RocketSim environments / data collection | Yes, CPU, full speed. |
| RLViser window (`RL_RENDER=1`) | No. Needs a host display and GPU-backed windowing. Run the viewer on the Linux host (see "Watching a containerized run" in the README), or copy checkpoints to a machine with a display and run an eval script there. |

Confirm the GPU is visible inside the container:

```sh
docker run --rm -it --gpus all rl-bot \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### CPU-only / slim variant

If you do not need the premade PyTorch base, this smaller Dockerfile also works
(save it as `Dockerfile.slim`, build with `docker build -f Dockerfile.slim ...`):

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

On `linux/amd64`, `pip install -r requirements.txt` here pulls the CUDA torch
wheel (works with `--gpus all` and a recent host driver). Add
`--index-url https://download.pytorch.org/whl/cpu` to the torch install for a
pure CPU image.
