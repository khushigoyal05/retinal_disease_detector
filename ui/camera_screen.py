from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QGridLayout, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush


class CameraPreviewWidget(QWidget):
    """
    The live camera feed area.
    On Pi: will be replaced with real picamera2 preview.
    On Windows: shows a black placeholder with guide circle.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setStyleSheet("background-color: #111111; border-radius: 10px;")
        self._sharpness = 0.0

    def set_sharpness(self, value: float) -> None:
        self._sharpness = value
        self.update()  # triggers repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 3

        # Outer guide circle (dashed)
        pen = QPen(QColor("#FFFFFF"))
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Crosshair lines
        pen.setStyle(Qt.PenStyle.SolidLine)
        pen.setColor(QColor("#444444"))
        painter.setPen(pen)
        painter.drawLine(cx, cy - radius, cx, cy + radius)
        painter.drawLine(cx - radius, cy, cx + radius, cy)

        # Placeholder text
        painter.setPen(QPen(QColor("#555555")))
        painter.setFont(QFont("Inter", 11))
        painter.drawText(
            0, h - 30, w, 24,
            Qt.AlignmentFlag.AlignCenter,
            "Camera preview — live on Pi"
        )

        # Sharpness score top-right
        painter.setPen(QPen(QColor("#FFFFFF")))
        painter.setFont(QFont("Inter", 11))
        painter.drawText(
            w - 140, 10, 130, 28,
            Qt.AlignmentFlag.AlignRight,
            f"Sharpness: {self._sharpness:.1f}"
        )


class CameraScreen(QWidget):
    """
    Screen 2 — live camera preview with adjustment sliders.
    Emits capture_requested when user taps the capture button.
    Emits back_requested when user taps back.
    """
    capture_requested = Signal()
    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(16)

        # ── Top bar ───────────────────────────────────────────────
        top_bar = QHBoxLayout()

        btn_back = QPushButton("← Back")
        btn_back.setFixedWidth(100)
        btn_back.setFont(QFont("Inter", 12))
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 1.5px solid #CCCCCC;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #F5F5F5; }
        """)
        btn_back.clicked.connect(self.back_requested)

        screen_title = QLabel("Live Preview")
        screen_title.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        screen_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # LIVE indicator
        live_row = QHBoxLayout()
        live_row.setSpacing(6)
        live_dot = QLabel("●")
        live_dot.setStyleSheet("color: #111111; font-size: 10px;")
        live_text = QLabel("LIVE")
        live_text.setFont(QFont("Inter", 10))
        live_text.setStyleSheet("color: #111111;")
        live_row.addStretch()
        live_row.addWidget(live_dot)
        live_row.addWidget(live_text)

        live_container = QWidget()
        live_container.setFixedWidth(100)
        live_container.setLayout(live_row)

        top_bar.addWidget(btn_back)
        top_bar.addStretch()
        top_bar.addWidget(screen_title)
        top_bar.addStretch()
        top_bar.addWidget(live_container)
        root.addLayout(top_bar)

        # ── Camera preview ────────────────────────────────────────
        self._preview = CameraPreviewWidget()
        root.addWidget(self._preview, stretch=1)

        # ── Sliders 2x2 grid ──────────────────────────────────────
        sliders_frame = QFrame()
        sliders_frame.setStyleSheet("""
            QFrame {
                background-color: #F7F7F7;
                border-radius: 10px;
            }
        """)
        sliders_layout = QGridLayout(sliders_frame)
        sliders_layout.setContentsMargins(16, 12, 16, 12)
        sliders_layout.setHorizontalSpacing(32)
        sliders_layout.setVerticalSpacing(10)

        self._sliders = {}
        controls = [
            ("Brightness", 0, 100, 60),
            ("Contrast",   0, 100, 50),
            ("Saturation", 0, 100, 45),
            ("Sharpness",  0, 100, 55),
        ]

        for i, (name, min_val, max_val, default) in enumerate(controls):
            row, col = divmod(i, 2)

            # Label + value on same line
            label_row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFont(QFont("Inter", 11))
            lbl.setStyleSheet("color: #888888; background: transparent;")

            val_lbl = QLabel(str(default))
            val_lbl.setFont(QFont("Inter", 11, QFont.Weight.Bold))
            val_lbl.setStyleSheet("color: #111111; background: transparent;")
            val_lbl.setFixedWidth(30)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

            label_row.addWidget(lbl)
            label_row.addStretch()
            label_row.addWidget(val_lbl)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)

            # Update value label live as slider moves
            slider.valueChanged.connect(
                lambda v, vl=val_lbl: vl.setText(str(v))
            )

            self._sliders[name] = slider

            cell = QVBoxLayout()
            cell.setSpacing(4)
            cell.addLayout(label_row)
            cell.addWidget(slider)

            cell_widget = QWidget()
            cell_widget.setLayout(cell)
            sliders_layout.addWidget(cell_widget, row, col)

        root.addWidget(sliders_frame)

        # ── Capture button ────────────────────────────────────────
        btn_capture = QPushButton("Capture  ⬤")
        btn_capture.setFixedHeight(56)
        btn_capture.setFont(QFont("Inter", 15, QFont.Weight.Bold))
        btn_capture.clicked.connect(self.capture_requested)
        root.addWidget(btn_capture)

        hint = QLabel("Captures 3 frames — best one selected automatically")
        hint.setFont(QFont("Inter", 10))
        hint.setStyleSheet("color: #AAAAAA;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(hint)

    def get_camera_settings(self) -> dict:
        """Returns current slider values — CameraService will apply these."""
        return {name: slider.value() for name, slider in self._sliders.items()}