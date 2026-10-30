_:
    just --list

# Install the pre-commit hooks so that they will be run automatically.
pre-commit-install:
    uv run --group dev prek install

# Manually run all pre-commit hooks.
pre-commit:
    uv run --group dev prek run --all-files

# Lint and type check the python package.
lint:
    uv run --group dev ruff check src tests
    uv run --group dev mypy src tests
    uv run --group dev ruff format --check src tests
    uv run --group lint pydoclint src

# Fix lint and formatting issues in the python package.
fix:
    uv run --group dev ruff check --fix src tests
    uv run --group dev ruff format src tests

# Start a python REPL with all dependencies installed.
repl:
    uv run --all-groups --all-extras --with ipython ipython

# Test the python package.
test:
    uv run pytest -v

# Generate documentation, by default under `./doc/html`.
doc docdir='doc/html':
    rm -rf {{docdir}}
    mkdir -p {{docdir}}
    uv run --group doc pdoc \
        --output-directory {{docdir}} \
        --no-include-undocumented \
        --docformat numpy \
        --search \
        zarr_ext_jpeg
