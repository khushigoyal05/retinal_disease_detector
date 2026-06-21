"""
Tests for FileService — real filesystem operations via tmp_path, since
this class has no hardware dependency.
"""

from pathlib import Path

import pytest
from PIL import Image

from services.file_service import FileService
import config


@pytest.fixture
def fake_upload(tmp_path: Path) -> Path:
    """A throwaway JPEG simulating a file picked via a dialog."""
    path = tmp_path / "my_retina_photo.jpg"
    Image.new("RGB", (100, 100), color="red").save(path)
    return path


def test_get_image_copies_file_into_upload_dir(fake_upload: Path, monkeypatch, tmp_path: Path):
    fake_upload_dir = tmp_path / "uploads"
    fake_upload_dir.mkdir()
    monkeypatch.setattr(config, "UPLOAD_DIR", fake_upload_dir)

    service = FileService()
    service.select(fake_upload)
    result_path = service.get_image()

    assert result_path.exists()
    assert result_path.parent == fake_upload_dir
    assert result_path.suffix == ".jpg"


def test_get_image_without_select_raises():
    service = FileService()
    with pytest.raises(RuntimeError):
        service.get_image()


def test_select_rejects_missing_file(tmp_path: Path):
    service = FileService()
    with pytest.raises(FileNotFoundError):
        service.select(tmp_path / "does_not_exist.jpg")


def test_select_rejects_unsupported_extension(tmp_path: Path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not an image")
    service = FileService()
    with pytest.raises(ValueError):
        service.select(bad_file)