# toolkit
My data science tool kit

## Documentation

Documentation will be available at [https://clementcome-toolkit.readthedocs.io/en/latest/](https://clementcome-toolkit.readthedocs.io/en/latest/).

## Installation

```bash
pip install clementcome-toolkit
```

## Locally work with the toolkit

If you want to work locally with the toolkit, you can clone the repository and execute `uv sync`

## Development

This project uses mainly uv, pytest and ruff for development.

If you cloned this project and want to start developing, you can install the package locally within a virtual environment.
```
uv sync
```
by default, it will create a virtual environment if you have no virtual environment activate.
My current setup is to first create a virtual environment (pyenv is my preferred choice but feel free) and then install the package locally.

For development you can add dependency groups specified in pyproject.toml especially the following ones:
```
uv sync --all-groups
```

Perform ruff checks with
```
uv run ruff check
```

Perform ruff formatting with
```
uv run ruff format
```

Execute tests with pytest
```
uv run pytest
```
