.PHONY: env

# Make environment for the first time
env:
	chmod u+x bootstrap.sh && ./bootstrap.sh
	uv venv --seed
	uv sync
