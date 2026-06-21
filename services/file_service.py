"""
FileService — ImageSource implementation backed by a file the user
selected through the UI's native file picker.

The dialog itself lives in the UI layer (it's a widget, it needs a
parent window). This class only handles what happens after a path is
chosen: validate it, copy it into managed storage, hand back a path the
rest of the pipeline can trust.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

from core.interfaces import ImageSource
import config


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class FileService(ImageSource):
    """
    Ingests a user-selected image file into app-managed upload storage.
    The UI must call select(path) with a file-picker result before
    calling get_image().
    """

    def __init__(self):
        self._selected_path: Optional[Path] = None

    def select(self, source_path: Path) -> None:
        """Validate and record the file the user picked."""
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Selected file does not exist: {source_path}")

        if source_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{source_path.suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        try:
            with Image.open(source_path) as img:
                img.verify()  # raises if the file isn't actually a valid image
        except Exception as exc:
            raise ValueError(f"File is not a valid image: {source_path}") from exc

        self._selected_path = source_path

    def get_image(self) -> Path:
        """Copy the selected file into config.UPLOAD_DIR and return the new path."""
        if self._selected_path is None:
            raise RuntimeError("No file selected — call select() before get_image().")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = config.UPLOAD_DIR / f"upload_{timestamp}{self._selected_path.suffix.lower()}"

        shutil.copy2(self._selected_path, destination)
        return destination