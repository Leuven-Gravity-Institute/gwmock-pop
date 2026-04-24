# Troubleshooting

This guide covers common issues you might encounter when using this package and
how to resolve them.

## Setup Issues

### Pre-commit Hook Installation Fails

**Problem:** `pre-commit install` returns an error or hooks don't run on commit.

**Solutions:**

<!-- prettier-ignore-start -->

1. Ensure you're in the project root directory
2. Verify Python virtual environment is activated
3. Reinstall pre-commit:

    ```bash
    uv pip uninstall pre-commit
    uv pip install pre-commit
    uv run pre-commit install
    ```

4. Check if `.git` directory exists (must be a git repository)
5. Try running manually: `pre-commit run --all-files`

<!-- prettier-ignore-end -->

### Virtual Environment Issues

**Problem:** Packages can't be found or dependencies conflict.

**Solutions:**

<!-- prettier-ignore-start -->

1. Create a fresh virtual environment:

    ```bash
    rm -rf .venv
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Install dependencies:

    ```bash
    uv sync
    ```

3. Verify installation:

    ```bash
    python -c "import gwmock_pop; print(gwmock_pop.__version__)"
    ```

<!-- prettier-ignore-end -->

<<<<<<< HEAD

## Development Issues

=======

## Testing Issues

> > > > > > > 65d49bc (docs: Update the docs of troubleshooting)

### "Unstaged Changes" After Running Hooks

**Problem:** Pre-commit modified files but they're not staged.

**Solutions:**

<!-- prettier-ignore-start -->

1. This is expected - review changes:

    ```bash
    git diff
    ```

2. Stage the changes:

    ```bash
    git add .
    ```

3. Try committing again
4. Or use `git add -A` to stage all changes before commit

<!-- prettier-ignore-end -->

### Dependency Conflicts

**Problem:** `uv pip install` fails with conflict messages.

**Solutions:**

<!-- prettier-ignore-start -->

1. Check Python version:

    ```bash
    python --version
    ```

2. Create fresh virtual environment:

    ```bash
    rm -rf .venv && uv venv
    source .venv/bin/activate
    ```

3. Install with verbose output to see conflict:

    ```bash
    uv sync -vv
    ```

4. Check `pyproject.toml` for overly restrictive version constraints

<!-- prettier-ignore-end -->

### Newer Version of Tool Breaks Things

**Problem:** Pre-commit hooks or tools updated and now fail.

**Solutions:**

<!-- prettier-ignore-start -->

1. Check what changed:

    ```bash
    pre-commit autoupdate --dry-run
    ```

2. Update individual tool:

    ```bash
    pre-commit autoupdate --repo https://github.com/tool-repo
    ```

3. Test changes:

    ```bash
   pre-commit run --all-files
    ```

4. Pin to known-good version in `.pre-commit-config.yaml`:

    ```yaml
    rev: v1.0.0 # Specific version instead of latest
    ```

<!-- prettier-ignore-end -->

## Getting Help

If you encounter issues not listed here:

<!-- prettier-ignore-start -->

1. **Check existing issues**: Search GitHub Issues for your problem
2. **Review logs carefully**: Error messages usually point to the root cause
3. **Search documentation**: Many issues are covered in specific tool docs
4. **Try minimal reproduction**: Isolate the problem to a single file/command
5. **Ask for help**: Open an [issue](https://github.com/Leuven-Gravity-Institute/gwmock_pop/issues/new/choose) with:
    - Your environment (Python version, OS)
    - Steps to reproduce
    - Full error message/logs
    - What you've already tried

<!-- prettier-ignore-end -->
