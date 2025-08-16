# bin/bash

uv sync --no-install-package flash-attn
uv sync
uv run pytest

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install transformers
pip install datasets
pip install math-verify[antlr4_13_2]
