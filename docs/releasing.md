# Releasing eCAT

The project is branded **eCAT**, imported as `ecat`, and published under the
PyPI distribution name `ecat-electrochemistry`.

## One-Time Account Setup

PyPI and TestPyPI are separate services. Create an account on both, verify both
email addresses, and enable two-factor authentication. No account or API token
is needed for local builds and checks, but an account on each service is needed
to configure and manage its Trusted Publisher.

Register a pending GitHub Trusted Publisher on both services. Pending
publishers create the corresponding project on first successful publication;
they do not reserve the distribution name beforehand.

Use these values:

| Setting | PyPI | TestPyPI |
| --- | --- | --- |
| Project name | `ecat-electrochemistry` | `ecat-electrochemistry` |
| GitHub owner | `ljelissiry` | `ljelissiry` |
| Repository | `eCAT` | `eCAT` |
| Workflow | `publish.yml` | `publish.yml` |
| Environment | `pypi` | `testpypi` |

Configure the publishers at:

- <https://pypi.org/manage/account/publishing/>
- <https://test.pypi.org/manage/account/publishing/>

In the GitHub repository, create environments named `pypi` and `testpypi`.
Require manual approval for every `pypi` deployment. TestPyPI may also require
approval if the maintainers prefer a conservative beta workflow. No long-lived
PyPI tokens should be stored as GitHub secrets.

## Prepare A Release Candidate

1. Update `src/ecat/_version.py`. The package, command-line app, source macOS
   shortcut, and standalone build all derive or verify their version against
   that source.
2. Update `CHANGELOG.md`, `README.md`, `docs/beta_scope.md`, and public notebook
   installation/output text.
3. Run the required Ruff check and full test suite.
4. Build and validate both distribution artifacts:

   ```bash
   python -m pip install -e ".[dev]"
   python -m ruff check src apps/workbench/src tests packaging
   pytest -q
   python -m build
   python -m twine check dist/*
   ```

5. Install the wheel into a new virtual environment and verify the import,
   package version, examples, app command, app HTTP response, and optional
   simulation contract.

## TestPyPI Trial

After the release candidate is merged into the default branch, open **Actions**,
select **Publish distributions**, and choose **Run workflow**. Manual dispatch
builds and checks one wheel/source-archive pair, then publishes those exact
artifacts to TestPyPI.

Install the exact candidate in a new environment. TestPyPI does not mirror all
runtime dependencies, so allow dependency resolution from PyPI:

```bash
python -m venv .testpypi-venv
source .testpypi-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "ecat-electrochemistry==0.1.0b6"
python -c "import ecat as e; print(e.__version__)"
```

Repeat with `[app]` and `[simulation]`, then run the packaged-example, notebook
import, app command, and simulation checks before publishing the same version to
PyPI. Published filenames cannot be replaced; fix a bad candidate by advancing
the beta version.

## Production Publication

Create and publish a GitHub Release whose tag exactly matches the package
version, for example `v0.1.0b6`. The release event rebuilds and checks the
artifacts, verifies the tag, waits for approval in the protected `pypi`
environment, and publishes through Trusted Publishing.

The workflow follows the Python Packaging Authority guidance:

- <https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>
- <https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/>
- <https://packaging.python.org/guides/using-testpypi/>
