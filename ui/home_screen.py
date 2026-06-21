from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QBrush, QColor


class EyeLogo(QWidget):
    """
    A simple retinal eye icon drawn with QPainter.
    Outer circle = iris, inner filled circle = pupil.
    Clean, medical, no external image files needed.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(52, 52)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Outer circle (iris)
        pen = QPen(QColor("#111111"))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawEllipse(4, 4, 44, 44)

        # Inner filled circle (pupil)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#111111")))
        painter.drawEllipse(16, 16, 20, 20)

class HomeScreen(QWidget):
    """
    Landing screen — two entry points:
    1. Capture a new image with the Pi camera
    2. Upload an existing image from storage
    """

    # Signals — emitted when user taps a button
    # MainWindow listens to these and calls show_screen()
    capture_requested = Signal()
    upload_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 40, 48, 40)
        root.setSpacing(0)
# ── Header ───────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(14)

        # Eye logo drawn with QPainter
        logo = EyeLogo()
        header.addWidget(logo)

        # Title + subtitle stacked vertically
        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        title = QLabel("Retinal Disease Detector")
        title.setFont(QFont("Inter", 18, QFont.Weight.Bold))

        subtitle = QLabel("AI-powered fundus screening")
        subtitle.setFont(QFont("Inter", 11))
        subtitle.setStyleSheet("color: #888888;")

        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        header.addLayout(title_col)
        header.addStretch()
        root.addLayout(header)

        # ── Divider ──────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #EEEEEE;")
        root.addWidget(divider)

        root.addStretch(1)  # pushes buttons to vertical center

        # ── Centre instruction text ───────────────────────────────
        instruction = QLabel("Select an option to begin")
        instruction.setFont(QFont("Inter", 14))
        instruction.setStyleSheet("color: #888888;")
        instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(instruction)

        root.addSpacing(20)

        # ── Two main action buttons ───────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)

        self._btn_capture = QPushButton("Capture Image")
        self._btn_capture.setFixedHeight(80)
        self._btn_capture.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        self._btn_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_capture.clicked.connect(self.capture_requested)

        self._btn_upload = QPushButton("Upload Image")
        self._btn_upload.setFixedHeight(80)
        self._btn_upload.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        self._btn_upload.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_upload.setProperty("class", "outline")
        self._btn_upload.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 2px solid #111111;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                padding: 14px 24px;
            }
            QPushButton:hover { background-color: #F5F5F5; }
            QPushButton:pressed { background-color: #EEEEEE; }
        """)
        self._btn_upload.clicked.connect(self.upload_requested)

        btn_row.addWidget(self._btn_capture)
        btn_row.addWidget(self._btn_upload)
        root.addLayout(btn_row)

        root.addStretch(1)

        # ── Disease class tags at bottom ──────────────────────────
        tags_label = QLabel("Detectable conditions:")
        tags_label.setFont(QFont("Inter", 11))
        tags_label.setStyleSheet("color: #888888;")
        root.addWidget(tags_label)

        root.addSpacing(8)

        tags_row = QHBoxLayout()
        tags_row.setSpacing(10)

        diseases = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]
        for i, name in enumerate(diseases):
            tag = QLabel(name)
            tag.setFont(QFont("Inter", 11))
            tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tag.setContentsMargins(12, 6, 12, 6)
            if i == 0:
                # First tag filled black (represents "Normal" as default)
                tag.setStyleSheet("""
                    background-color: #111111;
                    color: #FFFFFF;
                    border-radius: 6px;
                """)
            else:
                tag.setStyleSheet("""
                    background-color: #FFFFFF;
                    color: #111111;
                    border: 1.5px solid #CCCCCC;
                    border-radius: 6px;
                """)
            tags_row.addWidget(tag)

        tags_row.addStretch()
        root.addLayout(tags_row)