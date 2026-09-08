# Contributing

Bug reports and pull requests are welcome at
<https://github.com/dpiccolomd/qubosel>.

## Development setup

```bash
uv sync --extra dev --extra qaoa --extra docs
uv run pre-commit install
uv run pytest                       # core suite (< 30 s)
uv run pytest -m qaoa -o addopts="" tests/qaoa
uv run mkdocs build --strict
```

## Guidelines

- Keep the numerical conventions in `qubosel/qubo.py` unchanged; add tests for
  any new solver against `BruteForceSolver` on random instances.
- Every solver must be seed-deterministic and must return
  `energy == qubo.energy(x)`.
- Never commit data. Scripts that use non-public data must read them from a
  path outside the repository and write outside it.
- Run `ruff check` and `ruff format` before pushing (pre-commit does it).
