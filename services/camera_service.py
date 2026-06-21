"""Concrete ImageSource implementation backed by the Pi Camera Module via picamera2.

picamera2 only exists on Raspberry Pi OS (it wraps libcamera, the Pi's native
camera stack) -- unlike tflite-runtime/tensorflow, there is no Windows
equivalent. This module is structured now so that once you're on real Pi
hardware, only the body of get_image() needs finishing -- the class shape,
file naming, and error handling are already decided.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None  # Expected on any non-Pi machine, e.g. this Windows dev box

import config
from core.interfaces import ImageSource


class CameraService(ImageSource):
    """Captures a retinal image from the Pi Camera Module and saves it to disk."""

    def __init__(self) -> None:
        if Picamera2 is None:
            raise RuntimeError(
                "CameraService requires the picamera2 package, which is only "
                "available on Raspberry Pi OS. Are you running this on a dev "
                "machine? Use FileService for the upload path instead."
            )
        self._camera: Optional[Picamera2] = None

    def _ensure_started(self) -> None:
        if self._camera is None:
            self._camera = Picamera2()
            still_config = self._camera.create_still_configuration(
                main={"size": config.CAMERA_RESOLUTION}
            )
            self._camera.configure(still_config)
            self._camera.start()

    def get_image(self) -> Path:
        self._ensure_started()

        # TODO (needs real hardware): confirm whether this camera module
        # requires an explicit autofocus trigger before capture, or whether
        # focus is fixed by the 3D-printed lens mount. Fill in once known.

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.CAPTURE_DIR / f"capture_{timestamp}.jpg"
        self._camera.capture_file(str(output_path))
        return output_path

    def close(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera = None