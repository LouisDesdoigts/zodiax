# Contributing Guide

Thanks for your interest in contributing to `zodiax`.

---

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Create a feature branch.

```bash
git clone https://github.com/<your-username>/zodiax.git
cd zodiax
git checkout -b my-feature
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

The editable install (`-e`) is recommended so local changes in `src/zodiax` are immediately used when running tests.

---

## Pre-commit (Formatting and Linting)

Install hooks once:

```bash
pre-commit install
```

Run them manually on all files at any time:

```bash
pre-commit run --all-files
```

Current hooks include `black` and `ruff` (with autofix where possible).

---

## Testing

Run all tests:

```bash
pytest -q tests
```

Run one file:

```bash
pytest -q tests/test_base.py
```

Run one test:

```bash
pytest -q tests/test_base.py::TestBase::test_get
```

If you are not using an editable install, you can run with:

```bash
PYTHONPATH=src pytest -q tests
```

---

## Code Coverage

Run tests with coverage:

```bash
pytest --cov=zodiax --cov-report=term-missing --cov-report=xml --cov-report=html tests -q
```

This produces:
- terminal coverage summary
- `coverage.xml` (used by Codecov in CI)
- `htmlcov/` (local HTML report)

Open the HTML report locally:

```bash
open htmlcov/index.html
```

Note: `coverage.xml` and `htmlcov/` are generated artifacts and are ignored by git.

---

## Documentation

Docstrings own API contracts. Authored guides and source-package overviews teach
workflows. The pages in `docs/API/` render docstrings directly with mkdocstrings.
Build the documentation with:

```bash
zensical build
```

Tutorial notebooks live in the separate zodiax_tutorials repository. Edit and execute
the notebook there, export it with export.py, then synchronise Markdown and assets;
do not hand-edit exported tutorial pages. Run affected examples before publishing.

Pull requests build documentation without publishing. Main updates only the `main`
version. A validated release tag publishes its version and advances `latest` after
PyPI publication. Publishing requires the repository's PyPI trusted publisher and
GitHub environment to be configured. No release is created by merging a branch.

---

## CI Expectations

GitHub Actions runs tests across a Python version matrix and uploads coverage.

Before opening a PR, aim to have:
- pre-commit checks passing
- relevant tests added/updated
- local tests passing
- local coverage run for significant changes

---

## Pull Requests

When opening a PR, please include:
- a short summary of what changed
- why the change was needed
- test coverage for new behavior
- any doc updates (if applicable)

Keep PRs focused and small when possible; this makes review much easier.

---

## Questions / Help

If you are unsure about implementation details, test strategy, or API direction, open a draft PR early and ask for feedback.

## Archiving the pre-numerics documentation

Merge this workflow-only branch before the numerics changes. Run **Archive current
main documentation** with version `0.5.0`; it verifies main still reports that version,
archives its docs and sets the default to that archive. If the version directory
already exists, it registers those existing pages without overwriting them. This does not create a tag,
publish to PyPI or recreate a GitHub release. Verify the deployed version before
merging numerics. Normal main pushes update only `/main/` thereafter.
Existing archives remain available; old releases are not automatically rebuilt.
