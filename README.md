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

`nvfp4_stream` reads Hugging Face safetensors directly for Nemotron-H and dense
Llama and Qwen2 checkpoints. It keeps NVFP4 packed, expands FP8 to BF16 in memory,
and loads BF16 unchanged. Routed expert caches support packed NVFP4 and BF16.
It is not restricted to one specific model size.
When the checkpoint fits in unified memory, all experts are loaded before
generation; larger models use an SSD-backed expert cache (Super 120B has 59 GB
of routed experts). No model file is rewritten, no weight is requantized. It
runs on the MLX Metal kernel's per-expert NVFP4 `global_scale` support from
[ml-explore/mlx#4458](https://github.com/ml-explore/mlx/pull/4458), included in
MLX 0.32.3. `make install` installs a release wheel. See [DESIGN.md](DESIGN.md)
for the loader, resident mode, and streaming expert-cache design.

## Prerequisites

- macOS on Apple Silicon (tested on an M3 Pro, 36 GB unified memory)
- Xcode Command Line Tools (`xcode-select --install`) — provides `git` and
  `make`; no local Metal compiler is needed for the MLX release wheel.
- [Homebrew](https://brew.sh)
- `make`, if for some reason it isn't already on your system (Xcode Command
  Line Tools normally provide it):
  ```sh
  brew install make
  ```
- Python 3.11+
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

Dense Llama checkpoints use the same resident path:

```sh
MODEL_ID=nvidia/Llama-3.1-8B-Instruct-NVFP4 make all
```

Nano 30B is also `model_type: nemotron_h`, so the adapter reads its expert
layout the same way it reads Super's. On the tested 36 GB M3 Pro it runs fully
resident, using 20.07 GB peak memory and generating at 32.78 tokens/second.

`make all` downloads the selected checkpoint, installs MLX and
installs the adapter, applies the mlx-lm final-lookahead patch, asks the model
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
`nvfp4-stream`. There's no separate benchmark mode. Every run streams
generated text to stdout as it's produced, then prints a JSON stats block
(prompt/generation tok/sec, peak memory, load and total time):

```sh
.venv/bin/nvfp4-stream \
  --model models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --device metal \
  --expert-budget-gib 8 \
  --prompt "Write a 500-word essay about the beauty of Austria and its capital city." \
  --max-tokens 500
```

Options:

| Flag | Does |
|---|---|
| `-h`, `--help` | Show CLI help and exit |
| `--model DIR` | Local Hugging Face checkpoint directory (required) |
| `--prompt TEXT` | User message; default `Hello` |
| `--output FILE` | Write generated text; multiple runs add `-runN` to the name |
| `--stats-output FILE` | Write all run metrics as one JSON list |
| `--quiet-inference` | Suppress generated text but still print statistics |
| `--expert-stats` | Add per-run expert-cache cold/capacity misses and residency |
| `--max-tokens N` | Maximum generated tokens per run; default `1` |
| `--prefill-chunk N` | Tokens per prefill step; defaults to `2048` resident or slots divided by top-K streaming |
| `--expert-budget-gib GIB` | Total streaming expert-cache budget; default `8` GiB |
| `--expert-mode {auto,resident,stream}` | Select full residency, SSD streaming, or automatic memory-based selection |
| `--workers N` | Parallel checkpoint readers; default `6` |
| `--device {metal,cpu}` | MLX device; default `metal` |
| `--temp FLOAT` | Sampling temperature; default `0` |
| `--top-p FLOAT` | Nucleus-sampling probability; default `1` |
| `--raw-prompt` | Bypass the checkpoint chat template |
| `--trust-remote-code` | Allow Hugging Face remote code while loading |
| `--check-only` | Validate and summarize the checkpoint without loading the model |
| `--mlx-cache-gib GIB` | MLX cache limit; default `0.5` GiB |
| `--compile` | Compile routed-expert compute with MLX |
| `--runs N` | Sequential runs sharing one loaded model and expert cache; default `1` |

## Makefile targets

| Target | Does |
|---|---|
| `make download` / `make model` | Downloads the selected checkpoint from Hugging Face |
| `make install` | Creates a venv and installs release MLX, mlx-lm, and this adapter |
| `make patch` | Applies the mlx-lm final-lookahead patch |
| `make test` | Asks the model for the capital of Austria |
| `make test-compile` | Runs the same test with MLX compilation |
| `make run` | Runs the interpreted Metal essay `RERUN_INTERPRETED` times and writes the metrics to `interpreted-runs.json` |
| `make run-compile` | Runs the compiled Metal essay `RERUN_COMPILED` times and writes the metrics to `compile-runs.json` |
| `make metal-bench` | Times a 500-token essay on Metal |
| `make cpu-bench` | Times the short capital-of-Austria prompt CPU-only |
| `make all` | Runs all of the above in order |
| `make clean` | Removes the venv |

Override the rerun counts with `RERUN_INTERPRETED` and `RERUN_COMPILED`.
