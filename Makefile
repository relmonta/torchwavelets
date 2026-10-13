# torchwavelets developer shortcuts

.PHONY: docs docs-serve docs-deploy test lint

## Build the documentation site into site/ (verify before pushing)
docs:
	mkdocs build

## Serve the documentation locally with live reload
docs-serve:
	mkdocs serve

## Build and publish to GitHub Pages (needs the gh-pages branch)
docs-deploy:
	mkdocs gh-deploy

## Run the test suite
test:
	python -m pytest tests/ -q

## Lint everything (src, tests, examples)
lint:
	ruff check src tests examples
