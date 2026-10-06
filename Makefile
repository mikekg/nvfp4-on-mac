# MODEL_ID ?= nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
# MODEL_ID ?= nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8
# MODEL_ID ?= Qwen/Qwen2.5-3B-Instruct
# MODEL_ID ?= nvidia/Qwen3.6-35B-A3B-NVFP4
# MODEL_ID ?= nvidia/Llama-3.1-8B-Instruct-NVFP4
# MODEL_ID ?= nvidia/Llama-3.1-8B-Instruct-FP8
# MODEL_ID ?= meta-llama/Llama-3.1-8B-Instruct
MODEL_ID   ?= nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
MODEL_DIR  := models/$(notdir $(MODEL_ID))

VENV       := .venv
PY         := $(VENV)/bin/python3
PIP        := $(VENV)/bin/pip
RUN        := $(VENV)/bin/nvfp4-stream

MLX_LM_DIR := mlx-lm
EXPERT_BUDGET_GIB := 8
RERUN_COMPILED ?= 2
RERUN_INTERPRETED ?= 2

PROMPT_TEST  := What is the capital of Austria?
PROMPT_ESSAY := Write a 500-word essay about the beauty of Austria and its capital city.

.PHONY: download model install patch test test-compile run run-compile all metal-bench cpu-bench clean

# `download` is the real fetch step; `model` is an alias so `make model`
# and `make download` both do the same thing.
download: $(VENV)/bin/python3
	@mkdir -p $(dir $(MODEL_DIR))
	$(VENV)/bin/hf download $(MODEL_ID) --local-dir $(MODEL_DIR)

model: download

$(VENV)/bin/python3:
	python3 -m venv $(VENV)
	$(PIP) install -q -U pip huggingface_hub

install: $(VENV)/bin/python3 download
	@if [ ! -d "$(MLX_LM_DIR)" ]; then \
		git clone https://github.com/ml-explore/mlx-lm.git $(MLX_LM_DIR); \
	fi
	$(PIP) install -q "mlx>=0.32.3"
	$(PIP) install -q -e $(MLX_LM_DIR)
	$(PIP) install -q -e . --no-deps

patch:
	@patch="$(abspath patches/mlx-lm-skip-final-lookahead.patch)"; \
		if git -C "$(MLX_LM_DIR)" apply --check "$$patch" 2>/dev/null; then \
			git -C "$(MLX_LM_DIR)" apply "$$patch" || exit 1; \
		elif git -C "$(MLX_LM_DIR)" apply --reverse --check "$$patch" 2>/dev/null; then \
			echo "$$patch already applied, skipping"; \
		else \
			git -C "$(MLX_LM_DIR)" apply --check "$$patch"; \
			exit 1; \
		fi

test: install patch
	$(RUN) --model $(MODEL_DIR) --expert-budget-gib $(EXPERT_BUDGET_GIB) \
		--device metal --prompt "$(PROMPT_TEST)" --max-tokens 50

test-compile: install patch
	$(RUN) --model $(MODEL_DIR) --expert-budget-gib $(EXPERT_BUDGET_GIB) \
		--device metal --compile --prompt "$(PROMPT_TEST)" --max-tokens 50

run: install patch
	$(RUN) --model $(MODEL_DIR) --expert-budget-gib $(EXPERT_BUDGET_GIB) \
		--device metal --quiet-inference --expert-stats \
		--runs $(RERUN_INTERPRETED) \
		--stats-output interpreted-runs.json \
		--prompt "$(PROMPT_ESSAY)" --max-tokens 500

run-compile: install patch
	$(RUN) --model $(MODEL_DIR) --expert-budget-gib $(EXPERT_BUDGET_GIB) \
		--device metal --compile --quiet-inference --expert-stats \
		--runs $(RERUN_COMPILED) \
		--stats-output compile-runs.json \
		--prompt "$(PROMPT_ESSAY)" --max-tokens 500

metal-bench: install patch
	$(RUN) --model $(MODEL_DIR) --expert-budget-gib $(EXPERT_BUDGET_GIB) \
		--device metal --prompt "$(PROMPT_ESSAY)" --max-tokens 500

cpu-bench: install patch
	$(RUN) --model $(MODEL_DIR) --expert-budget-gib $(EXPERT_BUDGET_GIB) \
		--device cpu --prompt "$(PROMPT_TEST)" --max-tokens 50

all: install patch test metal-bench cpu-bench

clean:
	rm -rf $(VENV)
