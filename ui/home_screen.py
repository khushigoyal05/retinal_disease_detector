from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QBrush, QColor


class EyeLogo(QWidget):
    """Small retinal eye icon, drawn with QPainter — no image file needed."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(32, 32)  # was 52 — shrunk for small header

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#111111"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawEllipse(2, 2, 28, 28)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#111111")))
        painter.drawEllipse(10, 10, 12, 12)


class HomeScreen(QWidget):
    """
    Landing screen — two entry points: Capture or Upload.
    Laid out for a 320x480 PORTRAIT touchscreen. Buttons stacked
    vertically — portrait has generous height (480px) but narrow
    width (320px), so stacking keeps each button full-width and
    easy to tap instead of squeezing them side by side.
    """
    capture_requested = Signal()
    upload_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── Header: logo + title only, no subtitle ──
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(EyeLogo())

        title = QLabel("Retinal Disease Detector")
        title.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()
        root.addLayout(header)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #EEEEEE;")
        root.addWidget(divider)

        root.addStretch(1)

        instruction = QLabel("Select an option to begin")
        instruction.setFont(QFont("Inter", 10))
        instruction.setStyleSheet("color: #888888;")
        instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(instruction)

        # ── Two big touch-friendly buttons, stacked vertically ──
        btn_col = QVBoxLayout()
        btn_col.setSpacing(14)

        self._btn_capture = QPushButton("Capture Image")
        self._btn_capture.setFixedHeight(110)  # tall for easy finger tap
        self._btn_capture.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        self._btn_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_capture.clicked.connect(self.capture_requested)

        self._btn_upload = QPushButton("Upload Image")
        self._btn_upload.setFixedHeight(110)
        self._btn_upload.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        self._btn_upload.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_upload.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 2px solid #111111;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #F5F5F5; }
            QPushButton:pressed { background-color: #EEEEEE; }
        """)
        self._btn_upload.clicked.connect(self.upload_requested)

        btn_col.addWidget(self._btn_capture)
        btn_col.addWidget(self._btn_upload)
        root.addLayout(btn_col)

        root.addStretch(1)