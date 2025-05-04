PYTHON := python3

# --------------------------------------------------------------------
# PIPELINE
# --------------------------------------------------------------------

download:
	$(PYTHON) download_script.py

# --------------------------------------------------------------------

install-cpu:
	@echo "Installing CPU-only stack…"
	pip install -r requirements.txt
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

install-gpu:
	@echo "Installing CUDA-enabled stack"
	pip install -r requirements.txt
	pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# Auto-detect: prefer GPU if nvidia-smi exists
install: 
	@if command -v nvidia-smi >/dev/null 2>&1; then \
	  echo "NVIDIA GPU detected → installing CUDA build…"; \
	  $(MAKE) gpu; \
	else \
	  echo "No NVIDIA GPU found → installing CPU build…"; \
	  $(MAKE) cpu; \
	fi

clear_all:
	rm -rf __pycache__ *.pyc .pytest_cache

# --------------------------------------------------------------------

.PHONY: install-cpu install-gpu install clear_all download