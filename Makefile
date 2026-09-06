# Override on the command line or in the environment, e.g.:
#   MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 make all
MODEL_ID   ?= nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
MODEL_DIR  := models/$(notdir $(MODEL_ID))

VENV       := .venv
PY         := $(VENV)/bin/python3
PIP        := $(VENV)/bin/pip
RUN        := $(VENV)/bin/nvfp4-stream

MLX_LM_DIR := mlx-lm
EXPERT_BUDGET_GIB := 8

# ml-explore/mlx#4458 adds the per-expert NVFP4 global_scale support that
# gather_qmm needs for this model. It is not in any mlx release yet, so we
# build it from the PR commit instead of `pip install mlx`.
MLX_PR_SHA := 285ab899087ecfbae9390c1e96b1571f86a5ef8a

PROMPT_TEST  := What is the capital of Austria?
PROMPT_ESSAY := Write a 500-word essay about the beauty of Austria and its capital city.

.PHONY: download model install patch test all metal-bench cpu-bench clean

# `download` is the real fetch step; `model` is an alias so `make model`
# and `make download` both do the same thing.
download: $(VENV)/bin/python3
	@mkdir -p $(dir $(MODEL_DIR))
	@if [ -f "$(MODEL_DIR)/model.safetensors.index.json" ]; then \
		echo "Checkpoint already present at $(MODEL_DIR), skipping download."; \
	else \
		HF_HUB_ENABLE_HF_TRANSFER=0 $(VENV)/bin/hf download $(MODEL_ID) --local-dir $(MODEL_DIR); \
	fi

model: download

$(VENV)/bin/python3:
	python3 -m venv $(VENV)
	$(PIP) install -q -U pip huggingface_hub

install: $(VENV)/bin/python3 download
	@if [ ! -d "$(MLX_LM_DIR)" ]; then \
		git clone https://github.com/ml-explore/mlx-lm.git $(MLX_LM_DIR); \
	fi
	$(PIP) install -q -e $(MLX_LM_DIR) --no-deps
	@echo "Building MLX from ml-explore/mlx#4458 (not in any release yet, this takes a few minutes)..."
	$(PIP) install -q "git+https://github.com/ml-explore/mlx.git@$(MLX_PR_SHA)"
	$(PIP) install -q -e . --no-deps

patch:
	git -C $(MLX_LM_DIR) apply --check patches/mlx-lm-skip-final-lookahead.patch 2>/dev/null && \
		git -C $(MLX_LM_DIR) apply patches/mlx-lm-skip-final-lookahead.patch || \
		echo "skip-final-lookahead patch already applied, skipping"
	git -C $(MLX_LM_DIR) apply --check patches/mlx-lm-cpu-wired-limit-fix.patch 2>/dev/null && \
		git -C $(MLX_LM_DIR) apply patches/mlx-lm-cpu-wired-limit-fix.patch || \
		echo "cpu-wired-limit-fix patch already applied, skipping"

test: install patch
	$(RUN) --model $(MODEL_DIR) --expert-budget-gib $(EXPERT_BUDGET_GIB) \
		--device metal --prompt "$(PROMPT_TEST)" --max-tokens 50

metal-bench: install patch
	$(RUN) --model $(MODEL_DIR) --expert-budget-gib $(EXPERT_BUDGET_GIB) \
		--device metal --prompt "$(PROMPT_ESSAY)" --max-tokens 500

cpu-bench: install patch
	$(RUN) --model $(MODEL_DIR) --expert-budget-gib $(EXPERT_BUDGET_GIB) \
		--device cpu --prompt "$(PROMPT_TEST)" --max-tokens 50

all: install patch test metal-bench cpu-bench

clean:
	rm -rf $(VENV)
