# LLM Engineering Tutorial

### Project Setup

#### Initial Setup
With uv, you don't need a requirements.txt. `uv` uses `pyproject.toml` (what deps you want) + `uv.lock` (exact versions resolved) instead. That combo is more reliable than `requirements.txt` because the lock file pins exact versions for everyone.

Setup for git + fresh machine:

On your machine (once):
```bash
uv init                     # creates pyproject.toml (skip if you already have one)
uv add openai python-dotenv # adds deps, writes to pyproject.toml + uv.lock
```

On a fresh PC, after `git clone`:
```bash
uv sync
```

That one command reads `uv.lock`, creates `.venv`, and installs the exact same dependency versions you had — no manual uv add needed there. Then just recreate the .env file with the API key (never commit that).
