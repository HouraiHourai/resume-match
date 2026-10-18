"""公开快照必须排除本地资料，并保护已有发布目录。"""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "public_release_builder",
    Path(__file__).resolve().parents[1] / "scripts/build_public_release.py",
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_release_excludes_private_material_and_caches(tmp_path, monkeypatch):
    source = tmp_path / "source"
    monkeypatch.setattr(builder, "ROOT", source)
    for relative in builder.FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public fixture", encoding="utf-8")
    public_files = ["src/resume_match/__init__.py", "tests/test_example.py"]
    private_files = [
        ".env", "docs/notes.md", ".git/config", "backups/old.py",
        "evaluation/private/input.json", "evaluation/runs/result.json",
        "public-release/README.md", ".resume_xml/word/document.xml",
        "src/resume_match/__pycache__/module.pyc", "src/resume_match/.env",
        "tests/.pytest_cache/results", "tests/.ruff_cache/results",
    ]
    for relative in public_files + private_files:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture", encoding="utf-8")

    destination = source / "public-release-ready"
    builder.build_release(destination)

    assert all((destination / p).is_file() for p in public_files)
    assert (destination / ".env.example").is_file()
    assert all(not (destination / p).exists() for p in private_files)
    assert not (destination / "docs").exists()
    assert not (destination / ".git").exists()


def test_release_refuses_to_overwrite_existing_directory(tmp_path):
    destination = tmp_path / "release"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("existing release", encoding="utf-8")

    with pytest.raises(FileExistsError):
        builder.build_release(destination)

    assert marker.read_text(encoding="utf-8") == "existing release"


def test_release_rejects_output_inside_source_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    destination = tmp_path / "src" / "release"

    with pytest.raises(ValueError):
        builder.build_release(destination)

    assert not destination.exists()
