# Run NVFP4 models on a Mac

NVFP4 is an adaptive 4-bit floating point format, and it's fast becoming a
shared format across the AI community: hardware and software providers alike are
adopting and supporting it because it gets the best compression with the best
quality, thanks to adaptive per-block scaling. Nemotron 3 Super 120B is the
first model in the Nemotron 3 family pretrained natively in NVFP4, rather than
quantized down after the fact. I wanted to see it run on a Mac, with NVFP4 
weights, without no requantizing, GGUF conversion or other preprocessing -- 
because that means you can run any NVFP4 model (subject to operator support) 
on your Mac. nvfp4-stream runs the original 
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` NVFP4 checkpoint in the
ModelOpt layout
straight off disk on your Apple Silicon Mac. If you want to see NVFP4 running 
on a GPU instead, including for free on the Colab T4 tier, see
[nvfp4-on-turing](https://github.com/mikekg/nvfp4-on-turing) for a Google Colab
version.

## What this is

`nvfp4_stream` is an adapter for the Nemotron-H checkpoint layout (`model_type:
nemotron_h` in `config.json`) — it isn't restricted to one specific model
size, but it is restricted to that architecture family. It reads a model's
NVFP4 shards in the ModelOpt layout directly from the source safetensors files.
When the checkpoint fits in unified memory, all experts are loaded before
generation; larger models use an SSD-backed expert cache (Super 120B has 59 GB
of routed experts). No model file is rewritten, no weight is requantized. It
runs on the MLX Metal kernel's per-expert NVFP4 `global_scale` support from
[ml-explore/mlx#4458](https://github.com/ml-explore/mlx/pull/4458). `make
install` builds MLX from that merge commit until the required kernel appears
in a release.

The Nemotron-H family is the first target, not the only intended one and adding
support for other models should be straightforward: Today, `index.py` hard-checks 
`model_type == "nemotron_h"` and its tensor paths are Nemotron-H's own naming, so 
it does not read other architectures yet.  Extending it to other NVFP4 checkpoint 
layouts should be straightforward, though.

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
- Free disk for the selected checkpoint (~19 GB for Nano, ~80 GB for Super)
- ~25 GB free unified memory to run it

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
layout the same way it reads Super's. On the tested 36 GB M3 Pro it runs fully
resident, using 20.07 GB peak memory and generating at 32.78 tokens/second.

`make all` downloads the selected checkpoint, builds MLX and
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
| `--expert-mode {auto,resident,stream}` | Load every expert, stream experts, or choose from checkpoint and memory size |
| `--expert-budget-gib` | Unified memory given to the expert cache in stream mode |
| `--workers` | Parallel checkpoint readers for resident preload and streamed cache fills |
| `--temp`, `--top-p` | Sampling parameters |
| `--output <file>` | Also write the generated text to a file |
| `--raw-prompt` | Skip the chat template, send the prompt as-is |
| `--trust-remote-code` | Needed for the model's custom `modeling_nemotron_h.py` |
| `--check-only` | Validate the checkpoint's expert tensors without loading or generating anything |

## Makefile targets

| Target | Does |
|---|---|
| `make download` / `make model` | Downloads the NVFP4 checkpoint from Hugging Face |
| `make install` | Creates a venv, builds MLX from the #4458 merge commit, and installs mlx-lm and this adapter |
| `make patch` | Applies the mlx-lm patches this adapter needs |
| `make test` | Asks the model for the capital of Austria |
| `make metal-bench` | Times a 500-token essay on Metal |
| `make cpu-bench` | Times the short capital-of-Austria prompt CPU-only |
| `make all` | Runs all of the above in order |
| `make clean` | Removes the venv |
