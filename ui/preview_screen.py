from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap

import numpy as np


class PreviewScreen(QWidget):
    """
    Shows the auto-selected best frame from the capture burst.
    User either retakes or proceeds to analysis.
    Sized for 480x320 landscape touchscreen.

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
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        # ── Top bar: Retake + Quality badge (title dropped — implied by screen) ──
        top_bar = QHBoxLayout()

        btn_retake = QPushButton("← Retake")
        btn_retake.setFixedSize(80, 32)
        btn_retake.setFont(QFont("Inter", 10))
        btn_retake.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 1.5px solid #CCCCCC;
                border-radius: 6px;
                padding: 2px;
            }
            QPushButton:hover { background-color: #F5F5F5; }
        """)
        btn_retake.clicked.connect(self.retake_requested)

        self._quality_label = QLabel("—")
        self._quality_label.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        self._quality_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        top_bar.addWidget(btn_retake)
        top_bar.addStretch()
        top_bar.addWidget(self._quality_label)
        root.addLayout(top_bar)

        # ── Main image — takes almost all the space ──
        self._main_image = QLabel()
        self._main_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main_image.setStyleSheet(
            "background-color: #111111; border-radius: 8px;"
        )
        root.addWidget(self._main_image, stretch=1)

        # ── Analyze button ──
        self._btn_analyze = QPushButton("Analyze Retina →")
        self._btn_analyze.setFixedHeight(48)
        self._btn_analyze.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(self._on_analyze)
        root.addWidget(self._btn_analyze)

    def load_frames(self, frame_paths: List[Path],
                    sharpness_scores: List[float]) -> None:
        """
        Called after capture. Picks the best frame internally
        (same logic as before) but only displays that one frame.
        """
        self._frames = frame_paths
        self._sharpness_scores = sharpness_scores
        self._best_index = int(np.argmax(sharpness_scores))

        best_path = frame_paths[self._best_index]
        best_score = sharpness_scores[self._best_index]

        pixmap = QPixmap(str(best_path))
        self._main_image.setPixmap(
            pixmap.scaled(
                self._main_image.width(),
                self._main_image.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        quality = (
            "Excellent" if best_score > 100 else
            "Good"      if best_score > 50  else
            "Fair"      if best_score > 20  else
            "Poor"
        )
        self._quality_label.setText(f"Quality: {quality}")
        self._quality_label.setStyleSheet(
            "color: #111111;" if quality in ("Excellent", "Good")
            else "color: #B00020;"  # red-ish warning for Fair/Poor
        )

        self._btn_analyze.setEnabled(True)

    def _on_analyze(self) -> None:
        """Emit the best frame path for the results screen."""
        if self._frames:
            self.analyze_requested.emit(self._frames[self._best_index])