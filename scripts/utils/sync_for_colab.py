#!/usr/bin/env python
"""
Colab Sync Utility

Sync local changes to GitHub for Colab use.

This script helps you quickly sync your local changes to GitHub
so you can use them in Google Colab.

Usage:
    python scripts/utils/sync_for_colab.py [commit_message]
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_git_status():
    """Check if there are uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=project_root
    )
    return result.stdout.strip()


def sync_to_colab(commit_message: str = "Sync for Colab"):
    """
    Commit and push changes to GitHub for Colab use.

    Parameters
    ----------
    commit_message : str
        Commit message (default: "Sync for Colab")
    """
    print("🔄 Syncing changes to GitHub for Colab...")
    print(f"   Project root: {project_root}")
    print()

    # Check for uncommitted changes
    changes = check_git_status()
    if not changes:
        print("✅ No uncommitted changes. Already in sync!")
        return

    print("📝 Uncommitted changes found:")
    print(changes)
    print()

    # Ask for confirmation
    response = input("Commit and push these changes? (y/n): ").strip().lower()
    if response != "y":
        print("❌ Cancelled.")
        return

    try:
        # Add all changes
        print("📦 Adding changes...")
        subprocess.run(["git", "add", "."], check=True, cwd=project_root)

        # Commit
        print(f"💾 Committing: {commit_message}")
        subprocess.run(["git", "commit", "-m", commit_message], check=True, cwd=project_root)

        # Push
        print("🚀 Pushing to GitHub...")
        subprocess.run(["git", "push", "origin", "main"], check=True, cwd=project_root)

        print()
        print("✅ Successfully synced to GitHub!")
        print()
        print("📋 Next steps in Colab:")
        print("   !cd /content/PINN_project_using_andes && git pull origin main")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    commit_message = sys.argv[1] if len(sys.argv) > 1 else "Sync for Colab"
    sync_to_colab(commit_message)


if __name__ == "__main__":
    main()
