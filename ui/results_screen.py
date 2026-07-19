from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PySide6.QtWidgets import QFileDialog, QMessageBox
from utils.report_generator import generate_report
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor

from core.models import PredictionResult
import config


def _compute_extra_metrics(image_path: Path) -> dict:
    """
    Computes CDR approximation, lesion count, vessel density,
    haemorrhage count from the retinal image using OpenCV.
    These are estimates — not lab-grade measurements.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"cdr": 0.0, "lesion_area": 0.0,
                "vessel_density": 0.0, "haemorrhage_count": 0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── CDR approximation ─────────────────────────────────────────
    # Brightest region ≈ optic disc, slightly less bright ≈ cup
    _, disc_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    _, cup_mask  = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    disc_area = np.sum(disc_mask > 0) + 1   # +1 avoids division by zero
    cup_area  = np.sum(cup_mask  > 0)
    cdr = round(min(float(cup_area / disc_area), 0.99), 2)

    # ── Lesion area % ─────────────────────────────────────────────
    # Bright yellowish spots (exudates) in the red channel
    red_channel = img[:, :, 2]
    _, lesion_mask = cv2.threshold(red_channel, 220, 255, cv2.THRESH_BINARY)
    lesion_area = round(float(np.sum(lesion_mask > 0)) / (h * w) * 100, 2)

    # ── Vessel density % ──────────────────────────────────────────
    # Dark, thin structures detected with Frangi-like approach via
    # adaptive threshold + morphological operations
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    vessel_mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )
    vessel_density = round(float(np.sum(vessel_mask > 0)) / (h * w) * 100, 1)

    # ── Haemorrhage count ─────────────────────────────────────────
    # Dark red blobs in the image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark_red = cv2.inRange(hsv, (0, 50, 20), (15, 255, 120))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(dark_red, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    haemorrhage_count = len([c for c in contours if cv2.contourArea(c) > 30])

    return {
        "cdr":               cdr,
        "lesion_area":       lesion_area,
        "vessel_density":    vessel_density,
        "haemorrhage_count": haemorrhage_count,
    }


def _annotate_image(image_path: Path, metrics: dict) -> QPixmap:
    """
    Draws CDR circle and lesion highlights onto the retinal image.
    Returns a QPixmap ready to display.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return QPixmap()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2

    # CDR circle — red ring around estimated optic disc
    disc_radius = int(min(h, w) * 0.18)
    cv2.circle(img_rgb, (cx, cy), disc_radius, (255, 60, 60), 2)
    cv2.putText(img_rgb, f"CDR:{metrics['cdr']:.2f}",
                (cx - disc_radius, cy - disc_radius - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 60, 60), 1)

    # Lesion highlights — find bright spots and circle them
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, lesion_mask = cv2.threshold(
        img[:, :, 2], 220, 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours[:5]:  # max 5 annotations
        if cv2.contourArea(cnt) > 20:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cv2.circle(img_rgb,
                       (x + cw // 2, y + ch // 2),
                       max(cw, ch) // 2 + 4,
                       (255, 140, 0), 1)

    # Convert to QPixmap
    qimg = QImage(
        img_rgb.data, w, h, 3 * w,
        QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(qimg)


class ResultsScreen(QWidget):
    """
    Screen 4 — annotated image + 7 metrics + recommendation.
    """
    back_requested = Signal()
    new_scan_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: Optional[PredictionResult] = None
        self._last_metrics: dict = {}
        self._last_annotated_path: Optional[Path] = None
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

        title = QLabel("Analysis results")
        title.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_new = QPushButton("New scan")
        btn_new.setFixedWidth(100)
        btn_new.setFont(QFont("Inter", 12))
        btn_new.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 1.5px solid #CCCCCC;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #F5F5F5; }
        """)
        btn_new.clicked.connect(self.new_scan_requested)

        self._btn_save = QPushButton("Save report ↓")
        self._btn_save.setFixedWidth(130)
        self._btn_save.setFont(QFont("Inter", 12))
        self._btn_save.setEnabled(False)  # enabled after results load
        self._btn_save.setStyleSheet("""
            QPushButton {
                background-color: #111111;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #333333; }
            QPushButton:disabled { background-color: #CCCCCC; }
        """)
        self._btn_save.clicked.connect(self._on_save)

        top_bar.addWidget(btn_back)
        top_bar.addStretch()
        top_bar.addWidget(title)
        top_bar.addStretch()
        top_bar.addWidget(btn_new)
        top_bar.addSpacing(8)
        top_bar.addWidget(self._btn_save)
        root.addLayout(top_bar)

        # ── Main content: image left, metrics right ───────────────
        content = QHBoxLayout()
        content.setSpacing(20)

        # Annotated image
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "background-color: #111111; border-radius: 10px;"
        )
        self._image_label.setMinimumWidth(340)
        content.addWidget(self._image_label, stretch=1)

        # Right panel: diagnosis + metrics list
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # Diagnosis box
        diag_box = QFrame()
        diag_box.setStyleSheet(
            "background-color: #F7F7F7; border-radius: 8px;"
        )
        diag_layout = QVBoxLayout(diag_box)
        diag_layout.setContentsMargins(14, 10, 14, 10)
        diag_layout.setSpacing(4)

        diag_header = QLabel("Primary diagnosis")
        diag_header.setFont(QFont("Inter", 9))
        diag_header.setStyleSheet("color: #888888; background: transparent;")

        diag_row = QHBoxLayout()
        self._diagnosis_label = QLabel("—")
        self._diagnosis_label.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        self._diagnosis_label.setStyleSheet(
            "color: #111111; background: transparent;"
        )

        self._confidence_badge = QLabel("")
        self._confidence_badge.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        self._confidence_badge.setStyleSheet("""
            background-color: #111111;
            color: #FFFFFF;
            border-radius: 6px;
            padding: 3px 10px;
        """)

        diag_row.addWidget(self._diagnosis_label)
        diag_row.addStretch()
        diag_row.addWidget(self._confidence_badge)

        diag_layout.addWidget(diag_header)
        diag_layout.addLayout(diag_row)
        right_panel.addWidget(diag_box)

        # Metrics list box
        metrics_box = QFrame()
        metrics_box.setStyleSheet(
            "background-color: #F7F7F7; border-radius: 8px;"
        )
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(14, 10, 14, 10)
        metrics_layout.setSpacing(0)

        metrics_title = QLabel("Metrics")
        metrics_title.setFont(QFont("Inter", 9, QFont.Weight.Bold))
        metrics_title.setStyleSheet(
            "color: #888888; background: transparent; margin-bottom: 6px;"
        )
        metrics_layout.addWidget(metrics_title)

        # Each metric row: label on left, value on right, divider below
        self._metric_labels = {}
        metric_defs = [
            ("cdr",               "CDR value"),
            ("lesion_area",       "Lesion area"),
            ("vessel_density",    "Vessel density"),
            ("haemorrhage_count", "Haemorrhage count"),
            ("sharpness",         "Image quality score"),
        ]

        for i, (key, display) in enumerate(metric_defs):
            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 6, 0, 6)

            lbl = QLabel(display)
            lbl.setFont(QFont("Inter", 11))
            lbl.setStyleSheet("color: #888888; background: transparent;")

            val = QLabel("—")
            val.setFont(QFont("Inter", 11, QFont.Weight.Bold))
            val.setStyleSheet("color: #111111; background: transparent;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)

            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            self._metric_labels[key] = val
            metrics_layout.addWidget(row_widget)

            # Divider between rows except last
            if i < len(metric_defs) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet(
                    "color: #EEEEEE; background: transparent;"
                )
                metrics_layout.addWidget(line)

        right_panel.addWidget(metrics_box, stretch=1)
        content.addLayout(right_panel, stretch=1)
        root.addLayout(content, stretch=1)

        # ── Recommendation bar ────────────────────────────────────
        self._rec_bar = QFrame()
        self._rec_bar.setStyleSheet(
            "background-color: #F7F7F7; border-radius: 8px;"
        )
        rec_layout = QHBoxLayout(self._rec_bar)
        rec_layout.setContentsMargins(14, 10, 14, 10)
        rec_layout.setSpacing(12)

        self._rec_icon = QLabel("!")
        self._rec_icon.setFixedSize(28, 28)
        self._rec_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rec_icon.setStyleSheet("""
            background-color: #111111;
            color: #FFFFFF;
            border-radius: 14px;
            font-weight: bold;
            font-size: 14px;
        """)

        rec_text_col = QVBoxLayout()
        rec_text_col.setSpacing(2)
        self._rec_title = QLabel("—")
        self._rec_title.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        self._rec_title.setStyleSheet("background: transparent;")

        self._rec_detail = QLabel("")
        self._rec_detail.setFont(QFont("Inter", 10))
        self._rec_detail.setStyleSheet(
            "color: #888888; background: transparent;"
        )

        rec_text_col.addWidget(self._rec_title)
        rec_text_col.addWidget(self._rec_detail)

        rec_layout.addWidget(self._rec_icon)
        rec_layout.addLayout(rec_text_col)
        root.addWidget(self._rec_bar)

    def load_results(self, result: PredictionResult,
                     sharpness: float) -> None:
        """
        Called after inference completes.
        Populates all UI elements with real data.
        """
        # Compute extra metrics from the image
        metrics = _compute_extra_metrics(result.image_path)
        metrics["sharpness"] = sharpness

        # Annotate and display image
        pixmap = _annotate_image(result.image_path, metrics)
        if not pixmap.isNull():
            self._image_label.setPixmap(
                pixmap.scaled(
                    340, 400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

        # Diagnosis
        top = result.top_prediction
        self._diagnosis_label.setText(top.label)
        self._confidence_badge.setText(f"{top.confidence * 100:.0f}%")

        # Metrics
        self._metric_labels["cdr"].setText(str(metrics["cdr"]))
        self._metric_labels["lesion_area"].setText(
            f"{metrics['lesion_area']}%"
        )
        self._metric_labels["vessel_density"].setText(
            f"{metrics['vessel_density']}%"
        )
        self._metric_labels["haemorrhage_count"].setText(
            str(metrics["haemorrhage_count"])
        )
        self._metric_labels["sharpness"].setText(
            f"{sharpness:.1f}"
        )

        # Recommendation
        self._set_recommendation(top.label, metrics["cdr"],
                                 metrics["haemorrhage_count"])
        # Store for save report
        self._last_result = result
        self._last_metrics = metrics
        self._last_annotated_path = result.image_path
        self._btn_save.setEnabled(True)

    def _set_recommendation(self, diagnosis: str,
                             cdr: float,
                             haemorrhage_count: int) -> None:
        """Rule-based recommendation from diagnosis + key metrics."""
        if diagnosis == "Normal" and cdr < 0.65:
            title = "No action needed"
            detail = "No signs of retinal disease detected. Routine check-up in 12 months."
        elif diagnosis == "Glaucoma" or cdr >= 0.65:
            title = "Refer to ophthalmologist"
            cdr_status = "elevated" if cdr >= 0.65 else "within normal range"
            detail = f"CDR: {cdr:.2f} ({cdr_status}). Glaucoma screening recommended."
        elif diagnosis == "Diabetic Retinopathy" or haemorrhage_count > 2:
            title = "Urgent referral recommended"
            detail = f"{haemorrhage_count} haemorrhages detected. Immediate ophthalmology review needed."
        elif diagnosis in ("Cataract", "AMD"):
            title = "Specialist consultation advised"
            detail = f"{diagnosis} indicators found. Refer to ophthalmologist for confirmation."
        else:
            title = "Follow-up recommended"
            detail = "Findings detected. Please consult an eye care professional."

        self._rec_title.setText(title)
        self._rec_detail.setText(detail)

    def _on_save(self) -> None:
        """Opens a save dialog and generates the PDF report."""
        if not self._last_result:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"retinal_report_{timestamp}.pdf"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report",
            default_name,
            "PDF Files (*.pdf)"
        )

        if not save_path:
            return  # user cancelled

        try:
            generate_report(
                result=self._last_result,
                metrics=self._last_metrics,
                annotated_image_path=self._last_annotated_path,
                output_path=Path(save_path)
            )
            QMessageBox.information(
                self,
                "Report saved",
                f"Report saved to:\n{save_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not save report:\n{str(e)}"
            )