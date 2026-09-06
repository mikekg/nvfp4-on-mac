# nvfp4-on-mac

NVFP4 is NVIDIA's native 4-bit floating point format, and it's fast becoming a
shared format across the industry: hardware and software providers alike are
adopting and supporting it because it gets the best compression with the best
quality, thanks to adaptive per-block scaling. Nemotron 3 Super 120B is the
first model in the Nemotron 3 family pretrained natively in NVFP4, rather than
quantized down after the fact. I wanted to see it run on a Mac, with the real
NVFP4 weights, no requantizing, no GGUF conversion. This repo streams the
original `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` ModelOpt checkpoint
straight off disk on Apple Silicon. If you want to see NVFP4 running on a GPU
instead, including for free on the T4 tier, see
[nvfp4-on-turing](https://github.com/mikekg/nvfp4-on-turing) for a Google Colab
version.

## What this is

`nemotron_nvfp4_stream` is a narrow adapter for one checkpoint layout: it
reads Nemotron 3 Super's NVFP4 ModelOpt shards directly from the source
safetensors files, decoding routed experts on demand instead of holding the
full 59 GB of expert weights resident. No model file is rewritten, no weight
is requantized. It runs on a patched MLX Metal kernel that adds per-expert
NVFP4 `global_scale` support, from an as-yet-unmerged MLX pull request
([ml-explore/mlx#4458](https://github.com/ml-explore/mlx/pull/4458)) — `make
install` builds MLX from that PR commit, not from a release, since the
required kernel isn't in one yet.

## Prerequisites

- macOS on Apple Silicon (tested on an M3 Pro, 36 GB unified memory)
- Xcode Command Line Tools (`xcode-select --install`) — provides `git`,
  `make`, `clang`, and `cmake`'s build toolchain
- [Homebrew](https://brew.sh)
- `cmake`, needed to build MLX from source:
  ```sh
  brew install cmake
  ```
- `make`, if for some reason it isn't already on your system (Xcode Command
  Line Tools normally provide it):
  ```sh
  brew install make
  ```
- Python 3.10+
- ~80 GB free disk for the checkpoint, ~25 GB free unified memory to run it

## Install and run

```sh
git clone https://github.com/mikekg/nvfp4-on-mac.git
cd nvfp4-on-mac
make all
```

`make all` downloads the 80 GB checkpoint, builds the patched MLX and
installs the adapter, applies the mlx-lm patches this needs, asks the model
for the capital of Austria as a sanity check, then runs the two benchmarks
below.

To run the benchmarks on their own:

```sh
make metal-bench   # 500-token essay, Metal (GPU) backend
make cpu-bench     # 500-token essay, CPU-only backend
```

Both print the run's stats block (tokens/sec, peak memory, load time) after
generation.

## Makefile targets

| Target | Does |
|---|---|
| `make download` / `make model` | Downloads the NVFP4 checkpoint from Hugging Face |
| `make install` | Creates a venv, builds patched MLX from the PR commit, installs mlx-lm and this adapter |
| `make patch` | Applies the mlx-lm patches this adapter needs |
| `make test` | Asks the model for the capital of Austria |
| `make metal-bench` | Times a 500-token essay on Metal |
| `make cpu-bench` | Times the same essay CPU-only |
| `make all` | Runs all of the above in order |
| `make clean` | Removes the venv |
