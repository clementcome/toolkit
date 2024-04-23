# toolkit
My data science tool kit

## Documentation

Documentation will be available at [https://clementcome-toolkit.readthedocs.io/en/latest/](https://clementcome-toolkit.readthedocs.io/en/latest/).

## Installation

```bash
pip install clementcome-toolkit
```

## Locally work with the toolkit

If you want to work locally with the toolkit, you can clone the repository and execute `pip install --editable .`

## Development

This project uses mainly poetry, pytest and ruff for development.

If you cloned this project and want to start developing, you can install the package locally within a virtual environment.
```
poetry install
```
by default, it will create a virtual environment if you have no virtual environment activate.
My current setup is to first create a virtual environment (pyenv is my preferred choice but feel free) and then install the package locally.

For development you can add dependency groups specified in pyproject.toml especially the following ones:
```
poetry install --with dev,lint,test
```

Perform ruff checks with
```
ruff check
```

Perform ruff formatting with
```
ruff format
```

Execute tests with pytest
```
pytest
```
