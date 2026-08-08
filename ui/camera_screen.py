from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush


class CameraPreviewWidget(QWidget):
    """
    Live camera feed area.
    On Pi: replaced with real picamera2 preview.
    On Windows: black placeholder with alignment guide circle.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #111111; border-radius: 8px;")
        self._sharpness = 0.0

    def set_sharpness(self, value: float) -> None:
        self._sharpness = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 3

        # Guide circle — helps user center the eye
        pen = QPen(QColor("#FFFFFF"))
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Sharpness score, top-right corner (small, out of the way)
        painter.setPen(QPen(QColor("#FFFFFF")))
        painter.setFont(QFont("Inter", 9))
        painter.drawText(
            w - 100, 6, 94, 18,
            Qt.AlignmentFlag.AlignRight,
            f"Sharp: {self._sharpness:.1f}"
        )


class CameraScreen(QWidget):
    """
    Live camera preview — single Capture button, no manual controls.
    Sized for 480x320 landscape touchscreen.
    """
    capture_requested = Signal()
    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        # ── Top bar: Back + title, no separate LIVE badge (saves space) ──
        top_bar = QHBoxLayout()

        btn_back = QPushButton("← Back")
        btn_back.setFixedSize(70, 32)
        btn_back.setFont(QFont("Inter", 10))
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 1.5px solid #CCCCCC;
                border-radius: 6px;
                padding: 2px;
            }
            QPushButton:hover { background-color: #F5F5F5; }
        """)
        btn_back.clicked.connect(self.back_requested)

        screen_title = QLabel("Live Preview")
        screen_title.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        screen_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top_bar.addWidget(btn_back)
        top_bar.addStretch()
        top_bar.addWidget(screen_title)
        top_bar.addStretch()
        top_bar.addSpacing(70)  # balances the back button's width so title stays centered
        root.addLayout(top_bar)

        # ── Camera preview — takes almost all remaining space ──
        self._preview = CameraPreviewWidget()
        root.addWidget(self._preview, stretch=1)

        # ── Capture button ──
        btn_capture = QPushButton("Capture  ⬤")
        btn_capture.setFixedHeight(48)
        btn_capture.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        btn_capture.clicked.connect(self.capture_requested)
        root.addWidget(btn_capture)