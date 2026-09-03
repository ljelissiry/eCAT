# Contributing

Thanks for helping improve eCAT. The package is still in beta, so small, focused
pull requests are easiest to review and merge.

## Before Opening A PR

Please rebase onto the latest `main` before opening a PR:

```bash
git fetch origin
git rebase origin/main
```

After rebasing, run the relevant tests. For broad package changes, run:

```bash
pytest -q
```

Install the development tools, including pytest, notebook runners, package-build
tools, and Ruff, with:

```bash
python -m pip install -e ".[dev]"
```

Ruff performs static code analysis without executing eCAT. The required lint
profile intentionally contains only high-confidence correctness checks while the
existing broader lint debt is reduced incrementally:

```bash
python -m ruff check src apps/workbench/src tests packaging
```

CI also prints a non-blocking statistical summary of broader maintainability and
style findings. Do not mass-fix that advisory report: broad exception handling and
unused calculations should be reviewed with focused behavior tests, while purely
mechanical formatting can be handled separately.

Run that same advisory summary locally with:

```bash
python -m ruff check src apps/workbench/src tests packaging \
    --config ruff-advisory.toml --statistics --exit-zero
```

If your branch was already pushed before the rebase, update it with:

```bash
git push --force-with-lease
```

Use `--force-with-lease` instead of `--force`; it avoids overwriting someone
else's newer remote work by accident. If multiple people are working on the
same branch, coordinate before rebasing.

## PR Checklist

- Keep changes scoped to one bug, feature, or documentation update when possible.
- Add or update focused tests for behavior changes.
- Update documentation when public APIs, options, parser behavior, plotting, or
  analysis outputs change.
- Follow the [analysis output contract](docs/analysis_output_contract.md) when
  adding or changing notebook-facing analysis reports.
- Include notebook screenshots or short notes when changing notebook-facing plots
  or printed output.
- Keep the required Linux version matrix, Windows core suite, installed-wheel app
  smoke test, and Ruff correctness check passing.

Maintainers preparing a package release should follow
[the release guide](docs/releasing.md). It keeps TestPyPI validation separate
from production publication and uses short-lived Trusted Publishing credentials.
