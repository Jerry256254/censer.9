.PHONY: setup data train eval bench test cli repl clean

TEXT ?=

setup:
	bash scripts/setup_env.sh

data:
	.venv/bin/python -m kuclab.prepare_data

train:
	TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/bin/python -m kuclab.train

eval:
	.venv/bin/python -m kuclab.evaluate --json-out reports/eval_test.json

bench:
	.venv/bin/python scripts/benchmark.py

test:
	.venv/bin/python -m pytest tests -q

cli:
	.venv/bin/python -m kuclab.cli "$(TEXT)"

repl:
	.venv/bin/python -m kuclab.cli --repl

clean:
	rm -rf checkpoints reports data/processed
