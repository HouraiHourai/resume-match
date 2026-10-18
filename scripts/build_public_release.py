"""从白名单生成公开源码快照，排除 Git 历史、docs 和本地资料。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "public-release-ready"

FILES = [
    ".dockerignore",
    ".env.example",
    ".gitignore",
    ".python-version",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "docker-compose.yml",
    "pyproject.toml",
    "uv.lock",
    "evaluation/README.md",
    "evaluation/synthetic_jd_pilot.json",
    "scripts/build_public_release.py",
    "scripts/benchmark_embedding_cache.py",
    "scripts/fallback_smoke.py",
    "scripts/public_job_benchmark.py",
    "scripts/rebuild_bm25_index.py",
    "scripts/rebuild_vector_index.py",
    "scripts/run_server.py",
]

DIRECTORIES = ["src", "tests"]


def copy_file(relative: str, destination: Path) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build_release(destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {destination}")
    for relative in DIRECTORIES:
        if destination.is_relative_to((ROOT / relative).resolve()):
            raise ValueError("Output directory must be outside the source and test directories")
    for relative in FILES:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)

    destination.mkdir(parents=True)
    for relative in FILES:
        copy_file(relative, destination)
    for relative in DIRECTORIES:
        shutil.copytree(
            ROOT / relative,
            destination / relative,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache",
                ".git", ".env", ".env.*", ".venv",
            ),
        )

    print(f"Created clean public snapshot at {destination}")
    print("Excluded: .git, docs, private evaluation data, local resumes, and personal notes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST, help="New directory; existing files are never overwritten")
    args = parser.parse_args()
    build_release(args.output)


if __name__ == "__main__":
    main()
