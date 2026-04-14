# Utilities Directory

This directory contains standalone utility scripts that are not part of the main workflow but provide helpful tools.

---

## Scripts

### `check_environment.py`

**Purpose**: Check current environment and suggest workflow.

**What it checks:**
- Whether you're in Colab or local environment
- GPU availability
- PyTorch installation
- Recommended workflow for current setup

**Usage:**
```bash
python scripts/utils/check_environment.py
```

---

### `sync_for_colab.py`

**Purpose**: Sync local changes to GitHub for Colab use.

**What it does:**
- Checks for uncommitted changes
- Commits changes with message
- Pushes to GitHub
- Helps keep Colab in sync with local development

**Usage:**
```bash
# With default message
python scripts/utils/sync_for_colab.py

# With custom message
python scripts/utils/sync_for_colab.py "Updated training script"
```

---

## When to Use

These utilities are helpful for:
- **Environment setup**: Check what you have available
- **Colab workflow**: Sync local changes to use in Colab
- **Development**: Quick environment checks

They are **not required** for the main workflow but provide convenience.

---

**Last Updated**: December 2024

