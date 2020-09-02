.PHONY: install

install:
	poetry install

.PHONY: test

test:
	poetry run pytest --pyargs funpymodeling

.PHONY: check_style

check_style:
	poetry run flake8 --exclude=__init__.py
	poetry run flake8 --ignore F401 funpymodeling/__init__.py