.PHONY: install train clean

install:
	uv venv --prompt swiftroute
	uv pip install -r requirements.txt
	uv pip install -e .

train:
	uv run --no-sync python -m swiftroute.modeling.train

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
