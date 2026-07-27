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
- Include notebook screenshots or short notes when changing notebook-facing plots
  or printed output.
