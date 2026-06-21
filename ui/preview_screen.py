from pathlib import Path
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap, QImage

import numpy as np


class ThumbnailWidget(QWidget):
    """
    A single frame thumbnail with a sharpness label below it.
    Highlighted with a black border if it's the best frame.
    """
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._image_label = QLabel()
        self._image_label.setFixedSize(100, 80)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "background-color: #222222; border-radius: 6px;"
        )

        self._score_label = QLabel(f"Frame {self.index + 1}")
        self._score_label.setFont(QFont("Inter", 9))
        self._score_label.setStyleSheet("color: #888888;")
        self._score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._image_label)
        layout.addWidget(self._score_label)

    def set_image(self, image_path: Path, sharpness: float) -> None:
        pixmap = QPixmap(str(image_path))
        self._image_label.setPixmap(
            pixmap.scaled(100, 80,
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        )
        self._score_label.setText(f"Frame {self.index + 1} — {sharpness:.1f}")

    def set_best(self, is_best: bool) -> None:
        """Highlight with black border if this is the best frame."""
        if is_best:
            self._image_label.setStyleSheet("""
                background-color: #222222;
                border-radius: 6px;
                border: 3px solid #111111;
            """)
            self._score_label.setStyleSheet(
                "color: #111111; font-weight: bold;"
            )
        else:
            self._image_label.setStyleSheet(
                "background-color: #222222; border-radius: 6px;"
            )
            self._score_label.setStyleSheet("color: #888888;")


class PreviewScreen(QWidget):
    """
    Screen 3 — shows the best captured frame and lets user
    proceed to analysis or retake.

    Signals:
        analyze_requested(Path): emitted with the best frame path
        retake_requested:        emitted when user wants to go back
    """
    analyze_requested = Signal(Path)
    retake_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: List[Path] = []
        self._sharpness_scores: List[float] = []
        self._best_index: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(16)

        # ── Top bar ───────────────────────────────────────────────
        top_bar = QHBoxLayout()

        btn_retake = QPushButton("← Retake")
        btn_retake.setFixedWidth(110)
        btn_retake.setFont(QFont("Inter", 12))
        btn_retake.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 1.5px solid #CCCCCC;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #F5F5F5; }
        """)
        btn_retake.clicked.connect(self.retake_requested)

        self._title_label = QLabel("Best frame selected")
        self._title_label.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._frame_counter = QLabel("")
        self._frame_counter.setFont(QFont("Inter", 11))
        self._frame_counter.setStyleSheet("color: #888888;")
        self._frame_counter.setFixedWidth(110)
        self._frame_counter.setAlignment(Qt.AlignmentFlag.AlignRight)

        top_bar.addWidget(btn_retake)
        top_bar.addStretch()
        top_bar.addWidget(self._title_label)
        top_bar.addStretch()
        top_bar.addWidget(self._frame_counter)
        root.addLayout(top_bar)

        # ── Main image display ────────────────────────────────────
        self._main_image = QLabel()
        self._main_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main_image.setStyleSheet(
            "background-color: #111111; border-radius: 10px;"
        )
        self._main_image.setMinimumHeight(240)
        root.addWidget(self._main_image, stretch=1)

        # ── Metrics row ───────────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)

        self._metric_frames = {}
        for key, label in [
            ("frames",   "Frames captured"),
            ("sharpness","Best sharpness"),
            ("quality",  "Quality"),
        ]:
            box = QFrame()
            box.setStyleSheet(
                "background-color: #F7F7F7; border-radius: 8px;"
            )
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(12, 8, 12, 8)
            box_layout.setSpacing(2)

            lbl = QLabel(label)
            lbl.setFont(QFont("Inter", 9))
            lbl.setStyleSheet("color: #888888; background: transparent;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            val = QLabel("—")
            val.setFont(QFont("Inter", 14, QFont.Weight.Bold))
            val.setStyleSheet("color: #111111; background: transparent;")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)

            box_layout.addWidget(lbl)
            box_layout.addWidget(val)
            self._metric_frames[key] = val
            metrics_row.addWidget(box)

        root.addLayout(metrics_row)

        # ── Thumbnail strip ───────────────────────────────────────
        thumb_row = QHBoxLayout()
        thumb_row.setSpacing(12)
        self._thumbnails: List[ThumbnailWidget] = []

        for i in range(3):
            thumb = ThumbnailWidget(i)
            self._thumbnails.append(thumb)
            thumb_row.addWidget(thumb)

        thumb_row.addStretch()
        root.addLayout(thumb_row)

        # ── Analyze button ────────────────────────────────────────
        self._btn_analyze = QPushButton("Analyze Retina →")
        self._btn_analyze.setFixedHeight(56)
        self._btn_analyze.setFont(QFont("Inter", 15, QFont.Weight.Bold))
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(self._on_analyze)
        root.addWidget(self._btn_analyze)

    def load_frames(self, frame_paths: List[Path],
                    sharpness_scores: List[float]) -> None:
        """
        Called after capture. Receives all frame paths + their
        sharpness scores, picks the best one, updates the UI.
        """
        self._frames = frame_paths
        self._sharpness_scores = sharpness_scores
        self._best_index = int(np.argmax(sharpness_scores))

        best_path = frame_paths[self._best_index]
        best_score = sharpness_scores[self._best_index]

        # Main image
        pixmap = QPixmap(str(best_path))
        self._main_image.setPixmap(
            pixmap.scaled(
                self._main_image.width(),
                self._main_image.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        # Metrics
        self._metric_frames["frames"].setText(str(len(frame_paths)))
        self._metric_frames["sharpness"].setText(f"{best_score:.1f}")
        quality = (
            "Excellent" if best_score > 100 else
            "Good"      if best_score > 50  else
            "Fair"      if best_score > 20  else
            "Poor"
        )
        self._metric_frames["quality"].setText(quality)

        # Frame counter
        self._frame_counter.setText(
            f"Frame {self._best_index + 1} / {len(frame_paths)}"
        )

        # Thumbnails
        for i, thumb in enumerate(self._thumbnails):
            if i < len(frame_paths):
                thumb.set_image(frame_paths[i], sharpness_scores[i])
                thumb.set_best(i == self._best_index)

        self._btn_analyze.setEnabled(True)

    def _on_analyze(self) -> None:
        """Emit the best frame path for the results screen."""
        if self._frames:
            self.analyze_requested.emit(self._frames[self._best_index])