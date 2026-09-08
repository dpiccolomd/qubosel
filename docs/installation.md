# Installation

```bash
pip install qubosel                  # core: numpy, scipy, scikit-learn>=1.6, pandas, joblib
pip install "qubosel[qaoa]"          # + pennylane (Python >= 3.11)
pip install "qubosel[sb-torch]"      # + simulated-bifurcation (torch) as an alternative SB backend
pip install "qubosel[dwave]"         # + dwave-samplers as an alternative SA backend
```

Python 3.10–3.13 are supported (3.14 is tested on a best-effort basis).

## Development

```bash
git clone https://github.com/dpiccolomd/qubosel
cd qubosel
uv sync --extra dev --extra qaoa --extra docs
uv run pytest                        # core suite
uv run pytest -m qaoa -o addopts="" tests/qaoa
uv run mkdocs serve
```
