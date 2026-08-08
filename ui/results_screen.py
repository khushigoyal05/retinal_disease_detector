from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PySide6.QtWidgets import QFileDialog, QMessageBox
from utils.report_generator import generate_report
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap, QImage

from core.models import PredictionResult
import config

from services.cdr_service import compute_cdr
from services.cup_segmentation_engine import CupSegmentationEngine

from services.vessel_segmentation_engine import VesselSegmentationEngine
from PIL import Image

_cup_engine = CupSegmentationEngine()
_cup_engine.load()

_vessel_engine = VesselSegmentationEngine(config.VESSEL_MODEL_PATH)


def _compute_extra_metrics(image_path: Path) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        return {"cdr": 0.0, "vessel_density": 0.0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── CDR (real disc detection + trained cup model) ──────────────
    try:
        cdr_result = compute_cdr(img, _cup_engine)
        cdr = cdr_result.cdr
    except ValueError:
        cdr = None
        cdr_result = None

    # ── Vessel density % (trained model) ─────────────────────────
    pil_img = Image.open(image_path)
    vessel_mask = _vessel_engine.predict_mask(pil_img)
    vessel_density = round(float(np.sum(vessel_mask > 0)) / vessel_mask.size * 100, 1)

    return {
        "cdr":               cdr,
        "cdr_result":        cdr_result,
        "vessel_density":    vessel_density,
        "vessel_mask":       vessel_mask,
    }


def _annotate_image(image_path: Path, metrics: dict) -> QPixmap:
    """
    Draws vessel overlay, CDR circle and lesion highlights onto the
    retinal image. Backend logic — vessel overlay is new, rest UNCHANGED.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return QPixmap()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # ── Vessel overlay (green wash over detected vessel pixels) ────
    vessel_mask = metrics.get("vessel_mask")
    if vessel_mask is not None:
        mask_resized = cv2.resize(vessel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay_color = np.array([0, 255, 140], dtype=np.uint8)
        alpha = 0.45
        hit = mask_resized > 0
        img_rgb[hit] = (img_rgb[hit] * (1 - alpha) + overlay_color * alpha).astype(np.uint8)

    cdr_result = metrics.get("cdr_result")
    if cdr_result is not None:
        disc = cdr_result.disc
        cv2.circle(img_rgb, (disc.center_x, disc.center_y), disc.radius, (255, 60, 60), 2)
        cv2.putText(img_rgb, f"CDR:{metrics['cdr']:.2f}",
                    (disc.center_x - disc.radius, disc.center_y - disc.radius - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 60, 60), 1)


    qimg = QImage(
        img_rgb.data, w, h, 3 * w,
        QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(qimg)


def _load_original_pixmap(image_path: Path) -> QPixmap:
    """Loads the raw, unmodified fundus image for the 'Original' panel."""
    return QPixmap(str(image_path))


class ResultsScreen(QWidget):
    """
    Results screen — sized for 320x480 PORTRAIT touchscreen.

    Layout: only the top bar (Back/New scan/Save) is pinned. Everything
    else — diagnosis card, both images, recommendation, and metrics —
    scrolls together as one screen. This keeps things simple and lets
    the operator scroll straight from the images down to the numbers,
    rather than having images pinned separately from their metrics.
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
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(6)

        # ── Top bar: Back, New scan, Save — the only pinned element ──
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)

        btn_back = QPushButton("← Back")
        btn_back.setFixedSize(64, 28)
        btn_back.setFont(QFont("Inter", 9))
        btn_back.setStyleSheet(self._outline_btn_style())
        btn_back.clicked.connect(self.back_requested)

        btn_new = QPushButton("New scan")
        btn_new.setFixedSize(76, 28)
        btn_new.setFont(QFont("Inter", 9))
        btn_new.setStyleSheet(self._outline_btn_style())
        btn_new.clicked.connect(self.new_scan_requested)

        self._btn_save = QPushButton("Save ↓")
        self._btn_save.setFixedSize(64, 28)
        self._btn_save.setFont(QFont("Inter", 9))
        self._btn_save.setEnabled(False)
        self._btn_save.setStyleSheet("""
            QPushButton {
                background-color: #111111;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #333333; }
            QPushButton:disabled { background-color: #CCCCCC; }
        """)
        self._btn_save.clicked.connect(self._on_save)

        top_bar.addWidget(btn_back)
        top_bar.addStretch()
        top_bar.addWidget(btn_new)
        top_bar.addWidget(self._btn_save)
        root.addLayout(top_bar)

        # ── SCROLLABLE: everything below the top bar ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 4)
        scroll_layout.setSpacing(8)

        # ── Diagnosis card ──
        diag_box = QFrame()
        diag_box.setStyleSheet("background-color: #F7F7F7; border-radius: 8px;")
        diag_layout = QHBoxLayout(diag_box)
        diag_layout.setContentsMargins(12, 8, 12, 8)

        diag_col = QVBoxLayout()
        diag_col.setSpacing(1)
        diag_header = QLabel("Primary diagnosis")
        diag_header.setFont(QFont("Inter", 8))
        diag_header.setStyleSheet("color: #888888; background: transparent;")
        self._diagnosis_label = QLabel("—")
        self._diagnosis_label.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        self._diagnosis_label.setStyleSheet("color: #111111; background: transparent;")
        diag_col.addWidget(diag_header)
        diag_col.addWidget(self._diagnosis_label)

        self._confidence_badge = QLabel("")
        self._confidence_badge.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        self._confidence_badge.setStyleSheet("""
            background-color: #111111;
            color: #FFFFFF;
            border-radius: 6px;
            padding: 4px 10px;
        """)

        diag_layout.addLayout(diag_col)
        diag_layout.addStretch()
        diag_layout.addWidget(self._confidence_badge)
        scroll_layout.addWidget(diag_box)

        # ── Two images STACKED (Original above, AI Analysis below) ──
        orig_caption = QLabel("Original")
        orig_caption.setFont(QFont("Inter", 8))
        orig_caption.setStyleSheet("color: #888888;")
        self._image_label_original = QLabel()
        self._image_label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label_original.setStyleSheet("background-color: #111111; border-radius: 8px;")
        self._image_label_original.setFixedHeight(160)

        annot_caption = QLabel("AI Analysis")
        annot_caption.setFont(QFont("Inter", 8))
        annot_caption.setStyleSheet("color: #888888;")
        self._image_label_annotated = QLabel()
        self._image_label_annotated.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label_annotated.setStyleSheet("background-color: #111111; border-radius: 8px;")
        self._image_label_annotated.setFixedHeight(160)

        scroll_layout.addWidget(orig_caption)
        scroll_layout.addWidget(self._image_label_original)
        scroll_layout.addWidget(annot_caption)
        scroll_layout.addWidget(self._image_label_annotated)

        # ── Recommendation bar ──
        self._rec_bar = QFrame()
        self._rec_bar.setStyleSheet("background-color: #F7F7F7; border-radius: 8px;")
        rec_layout = QHBoxLayout(self._rec_bar)
        rec_layout.setContentsMargins(10, 6, 10, 6)
        rec_layout.setSpacing(8)

        self._rec_icon = QLabel("!")
        self._rec_icon.setFixedSize(22, 22)
        self._rec_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._rec_icon.setStyleSheet("""
            background-color: #111111;
            color: #FFFFFF;
            border-radius: 11px;
            font-weight: bold;
            font-size: 11px;
        """)

        rec_text_col = QVBoxLayout()
        rec_text_col.setSpacing(0)
        self._rec_title = QLabel("—")
        self._rec_title.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        self._rec_title.setStyleSheet("background: transparent;")
        self._rec_title.setWordWrap(True)

        self._rec_detail = QLabel("")
        self._rec_detail.setFont(QFont("Inter", 8))
        self._rec_detail.setStyleSheet("color: #888888; background: transparent;")
        self._rec_detail.setWordWrap(True)

        rec_text_col.addWidget(self._rec_title)
        rec_text_col.addWidget(self._rec_detail)

        rec_layout.addWidget(self._rec_icon)
        rec_layout.addLayout(rec_text_col, stretch=1)
        scroll_layout.addWidget(self._rec_bar)

        # ── Metrics box — compact rows ──
        metrics_box = QFrame()
        metrics_box.setStyleSheet("background-color: #F7F7F7; border-radius: 8px;")
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(12, 8, 12, 6)
        metrics_layout.setSpacing(0)

        self._metric_labels = {}
        metric_defs = [
            ("cdr",               "CDR value"),
            ("vessel_density",    "Vessel density"),
            ("sharpness",         "Image quality score"),
        ]

        for i, (key, display) in enumerate(metric_defs):
            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 5, 0, 5)

            lbl = QLabel(display)
            lbl.setFont(QFont("Inter", 9))
            lbl.setStyleSheet("color: #888888; background: transparent;")

            val = QLabel("—")
            val.setFont(QFont("Inter", 9, QFont.Weight.Bold))
            val.setStyleSheet("color: #111111; background: transparent;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)

            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            self._metric_labels[key] = val
            metrics_layout.addWidget(row_widget)

            if i < len(metric_defs) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet("color: #EEEEEE; background: transparent;")
                metrics_layout.addWidget(line)

        scroll_layout.addWidget(metrics_box)
        scroll.setWidget(scroll_content)
        root.addWidget(scroll, stretch=1)

    @staticmethod
    def _outline_btn_style() -> str:
        return """
            QPushButton {
                background-color: #FFFFFF;
                color: #111111;
                border: 1.5px solid #CCCCCC;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #F5F5F5; }
        """

    def load_results(self, result: PredictionResult,
                     sharpness: float) -> None:
        """
        Called after inference completes. Loads both the original
        and the AI-annotated image into their own stacked panels.
        """
        metrics = _compute_extra_metrics(result.image_path)
        metrics["sharpness"] = sharpness

        original_pixmap = _load_original_pixmap(result.image_path)
        if not original_pixmap.isNull():
            self._image_label_original.setPixmap(
                original_pixmap.scaled(
                    296, 160,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

        annotated_pixmap = _annotate_image(result.image_path, metrics)
        if not annotated_pixmap.isNull():
            self._image_label_annotated.setPixmap(
                annotated_pixmap.scaled(
                    296, 160,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

        top = result.top_prediction
        self._diagnosis_label.setText(top.label)
        self._confidence_badge.setText(f"{top.confidence * 100:.0f}%")

        self._metric_labels["cdr"].setText(
            str(metrics["cdr"]) if metrics["cdr"] is not None else "N/A"
        )
        self._metric_labels["vessel_density"].setText(f"{metrics['vessel_density']}%")
        self._metric_labels["sharpness"].setText(f"{sharpness:.1f}")

        self._set_recommendation(top.label, metrics["cdr"])

        self._last_result = result
        self._last_metrics = metrics
        self._last_annotated_path = result.image_path
        self._btn_save.setEnabled(True)

    def _set_recommendation(self, diagnosis: str, cdr: Optional[float]) -> None:
        cdr_known = cdr is not None
        if diagnosis == "Normal" and (not cdr_known or cdr < 0.65):
            title = "No action needed"
            detail = "No signs of retinal disease detected. Routine check-up in 12 months."
        elif diagnosis == "Glaucoma" or (cdr_known and cdr >= 0.65):
            title = "Refer to ophthalmologist"
            cdr_status = "elevated" if cdr >= 0.65 else "within normal range"
            detail = f"CDR: {cdr:.2f} ({cdr_status}). Glaucoma screening recommended."
        elif diagnosis == "Diabetic Retinopathy":
            title = "Urgent referral recommended"
            detail = "Diabetic retinopathy indicators detected. Immediate ophthalmology review needed."
        elif diagnosis in ("Cataract", "AMD"):
            title = "Specialist consultation advised"
            detail = f"{diagnosis} indicators found. Refer to ophthalmologist for confirmation."
        else:
            title = "Follow-up recommended"
            detail = "Findings detected. Please consult an eye care professional."

        self._rec_title.setText(title)
        self._rec_detail.setText(detail)

        
    def _on_save(self) -> None:
        """UNCHANGED LOGIC from original."""
        if not self._last_result:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"retinal_report_{timestamp}.pdf"

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default_name, "PDF Files (*.pdf)"
        )

        if not save_path:
            return

        try:
            generate_report(
                result=self._last_result,
                metrics=self._last_metrics,
                annotated_image_path=self._last_annotated_path,
                output_path=Path(save_path)
            )
            QMessageBox.information(self, "Report saved", f"Report saved to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save report:\n{str(e)}")