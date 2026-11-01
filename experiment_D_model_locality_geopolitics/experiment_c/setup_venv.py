"""Create the Experiment C virtual environment and install laptop packages.

The environment is ``.venv`` in the repository, unless that path is too long
for Windows. On this OneDrive project it is created at
``~/.venvs/cs612-experiment-c``. It is for ``build_dataset``, the tests, the
judge, and ``analyze`` on your machine. Colab does not use this environment.
The notebook installs its own packages on the A100 runtime.

From the repository root:

    python -m experiment_c.setup_venv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"
REQUIREMENTS_COLAB = Path(__file__).resolve().parent / "requirements-colab.txt"
# numpy's license tree is nested deeply. On Windows, a .venv inside this
# OneDrive/iCloud project path exceeds MAX_PATH and pip fails with WinError 206.
_WINDOWS_PATH_BUDGET = 240
_SITE_PACKAGES_TAIL = 120


def default_venv_dir() -> Path:
    local = REPO_ROOT / ".venv"
    if os.name == "nt" and len(str(local)) + _SITE_PACKAGES_TAIL > _WINDOWS_PATH_BUDGET:
        return Path.home() / ".venvs" / "cs612-experiment-c"
    return local


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def activate_hint(venv_dir: Path) -> str:
    if os.name == "nt":
        script = venv_dir / "Scripts" / "Activate.ps1"
        return f"& '{script}'"
    return f"source '{venv_dir / 'bin' / 'activate'}'"


def create_venv(venv_dir: Path, *, recreate: bool, with_generation: bool) -> Path:
    if recreate and venv_dir.exists():
        import shutil

        shutil.rmtree(venv_dir)
    if not venv_dir.exists():
        print(f"Creating {venv_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    else:
        print(f"Using existing {venv_dir}")

    python = venv_python(venv_dir)
    if not python.exists():
        raise FileNotFoundError(f"Expected the venv interpreter at {python}")

    subprocess.check_call([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    if with_generation:
        print(
            "Installing local generation packages. This downloads PyTorch and "
            "is only needed if you generate outside Colab."
        )
        subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS_COLAB)])
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Experiment C virtual environment.")
    parser.add_argument("--venv", type=Path, default=None, help="Defaults to .venv, or a short path under your home directory on Windows when the project path is too long.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete an existing .venv and create it again.",
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also install the Colab/Hugging Face packages from requirements-colab.txt.",
    )
    args = parser.parse_args()
    chosen = default_venv_dir() if args.venv is None else args.venv
    venv_dir = chosen if chosen.is_absolute() else REPO_ROOT / chosen
    python = create_venv(venv_dir, recreate=args.recreate, with_generation=args.with_generation)
    print(f"Ready: {python}")
    print(f"Activate it with: {activate_hint(venv_dir)}")
    print("Then run commands from the repository root, for example:")
    print("  python -m unittest experiment_c.test_pipeline")


if __name__ == "__main__":
    main()
