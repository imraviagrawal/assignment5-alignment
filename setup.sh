# bin/bash
pip install uv
uv sync --no-install-package flash-attn
uv sync
uv run pytest

pip install typing-extensions==4.12.2
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install transformers
pip install datasets
pip install math-verify[antlr4_13_2]
