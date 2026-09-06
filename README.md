# nvfp4-on-mac

*Copyright (c) 2026 the nvfp4-stream authors. SPDX-License-Identifier: Apache-2.0*

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

`nvfp4_stream` is an adapter for the Nemotron-H checkpoint layout (`model_type:
nemotron_h` in `config.json`) — it isn't restricted to one specific model
size, but it is restricted to that architecture family. It reads a model's
NVFP4 ModelOpt shards directly from the source safetensors files, decoding
routed experts on demand instead of holding the full expert weight set
resident (59 GB for Super 120B). No model file is rewritten, no weight is
requantized. It runs on a patched MLX Metal kernel that adds per-expert NVFP4
`global_scale` support, from an as-yet-unmerged MLX pull request
([ml-explore/mlx#4458](https://github.com/ml-explore/mlx/pull/4458)) — `make
install` builds MLX from that PR commit, not from a release, since the
required kernel isn't in one yet.

It's named `nvfp4-stream`, not `nemotron-nvfp4-stream`, on purpose: the
Nemotron-H family is the first target, not the only intended one. Today
`index.py` hard-checks `model_type == "nemotron_h"` and its tensor paths are
Nemotron-H's own naming, so it does not read other architectures yet.
Extending it to other NVFP4 checkpoint layouts is future work, not a claim
about what it does now.

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

By default this runs Nemotron 3 Super 120B. `MODEL_ID` is an overridable
variable, so another NVFP4 checkpoint works the same way:

```sh
MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 make all
```

Nano 30B is also `model_type: nemotron_h`, so the adapter reads its expert
layout the same way it reads Super's. It has not been run end to end through
this adapter yet — Super 120B is the one these benchmarks are from.

`make all` downloads the 80 GB checkpoint, builds the patched MLX and
installs the adapter, applies the mlx-lm patches this needs, asks the model
for the capital of Austria as a sanity check, then runs the two benchmarks
below.

To run the benchmarks on their own:

```sh
make metal-bench   # 500-token essay, Metal (GPU) backend
make cpu-bench     # short capital-of-Austria prompt, CPU-only backend (CPU
                   # dequant is slow enough that a 500-token essay isn't practical)
```

Both print the run's stats block (tokens/sec, peak memory, load time) after
generation.

## Calling the CLI directly

The Makefile targets are thin wrappers around one command,
`nvfp4-stream`. There's no separate benchmark mode — every run
streams generated text to stdout as it's produced, then prints a JSON stats
block (prompt/generation tok/sec, peak memory, load and total time) once
generation finishes:

```sh
.venv/bin/nvfp4-stream \
  --model models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --device metal \
  --expert-budget-gib 8 \
  --prompt "Write a 500-word essay about the beauty of Austria and its capital city." \
  --max-tokens 500
```

Key options:

| Flag | Does |
|---|---|
| `--model` | Local checkpoint directory (required) |
| `--device {metal,cpu}` | Run on the Metal GPU backend or CPU-only |
| `--prompt` | The user message |
| `--max-tokens` | How many tokens to generate |
| `--expert-budget-gib` | Unified memory given to the resident expert cache |
| `--workers` | Parallel readers for streaming expert weights off disk |
| `--temp`, `--top-p` | Sampling parameters |
| `--output <file>` | Also write the generated text to a file |
| `--raw-prompt` | Skip the chat template, send the prompt as-is |
| `--trust-remote-code` | Needed for the model's custom `modeling_nemotron_h.py` |
| `--check-only` | Validate the checkpoint's expert tensors without loading or generating anything |

## Makefile targets

| Target | Does |
|---|---|
| `make download` / `make model` | Downloads the NVFP4 checkpoint from Hugging Face |
| `make install` | Creates a venv, builds patched MLX from the PR commit, installs mlx-lm and this adapter |
| `make patch` | Applies the mlx-lm patches this adapter needs |
| `make test` | Asks the model for the capital of Austria |
| `make metal-bench` | Times a 500-token essay on Metal |
| `make cpu-bench` | Times the short capital-of-Austria prompt CPU-only |
| `make all` | Runs all of the above in order |
| `make clean` | Removes the venv |
