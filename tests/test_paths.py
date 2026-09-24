"""路徑正規化與 Google Drive 偵測測試。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from envs.config import (
    MY_DRIVE_FOLDER_NAMES,
    find_google_drive_roots,
    normalize_user_path,
    resolve_path,
)

windows_only = pytest.mark.skipif(os.name != "nt", reason="僅 Windows 需要這些正規化規則")


@windows_only
def test_normalize_collapses_duplicate_separators():
    assert str(normalize_user_path(r"G:\\MyDrive")) == r"G:\MyDrive"
    assert str(normalize_user_path(r"G:\\\\MyDrive\\\\tetrio-ai")) == r"G:\MyDrive\tetrio-ai"
    assert str(normalize_user_path(r"C:\Users\\Walla\\\tetr")) == r"C:\Users\Walla\tetr"


@windows_only
def test_normalize_handles_mixed_separators_and_quotes():
    assert str(normalize_user_path("G:/MyDrive/tetrio-ai")) == r"G:\MyDrive\tetrio-ai"
    assert str(normalize_user_path('"G:\\\\MyDrive\\\\tetrio-ai"')) == r"G:\MyDrive\tetrio-ai"
    assert str(normalize_user_path("  C:\\Users\\Walla\\tetr  ")) == r"C:\Users\Walla\tetr"


@windows_only
def test_normalize_keeps_unc_prefix_and_drive_root():
    assert str(normalize_user_path(r"\\server\share\x")) == r"\\server\share\x"
    assert str(normalize_user_path(r"\\\\server\\share\\x")) == r"\\server\share\x"
    assert str(normalize_user_path("C:\\")) == "C:\\"
    assert str(normalize_user_path(r"C:\Users\Walla\tetr\\")) == r"C:\Users\Walla\tetr"


def test_normalize_expands_user_home():
    result = normalize_user_path("~/tetrio-ai")
    assert "tetrio-ai" in str(result)
    assert "~" not in str(result)


def test_normalize_empty_string():
    assert str(normalize_user_path("   ")) == "."


def test_find_google_drive_roots_contract():
    roots = find_google_drive_roots()
    assert isinstance(roots, list)
    for root in roots:
        assert isinstance(root, Path)
        assert root.name in MY_DRIVE_FOLDER_NAMES
        assert root.is_dir()


def test_resolve_path_accepts_existing_project_file():
    resolved = resolve_path("configs/default.yaml")
    assert resolved.is_absolute()
    assert resolved.exists()
