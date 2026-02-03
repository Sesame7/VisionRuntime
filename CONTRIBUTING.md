# Contributing Guide

Thanks for your interest in contributing! This project is maintained in English,
and all contributions and discussions should be in English.

## What We Welcome

We welcome all kinds of contributions:

- Bug reports
- Feature requests
- Documentation improvements
- Code changes and refactors

## Reporting Issues

Please use the issue templates when possible. Include:

- A clear summary
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (OS, Python, version/commit)

Security vulnerabilities should be reported privately. See `SECURITY.md`.

## Development Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Optional Checks

If you can, please run one or more of the following and note the results in your PR:

- `ruff check .`
- `ruff format .`
- `python -m compileall -q .`

## Pull Requests

- Keep PRs focused and small.
- Describe what changed and why.
- Testing is optional, but please state what you ran (or that you did not run tests).
