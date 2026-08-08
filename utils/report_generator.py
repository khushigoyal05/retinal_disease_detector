from pathlib import Path
from datetime import datetime
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib import colors

from core.models import PredictionResult


def generate_report(
    result: PredictionResult,
    metrics: dict,
    annotated_image_path: Path,
    output_path: Path
) -> Path:
    """
    Generates a clean black/white medical PDF report.
    Returns the path to the saved PDF.
    """
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    # ── Styles ────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#111111"),
        spaceAfter=2 * mm,
    )
    subtitle_style = ParagraphStyle(
        "subtitle",
        fontName="Helvetica",
        fontSize=11,
        textColor=colors.HexColor("#888888"),
        spaceAfter=6 * mm,
    )
    section_style = ParagraphStyle(
        "section",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#111111"),
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
    )
    body_style = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#111111"),
        spaceAfter=2 * mm,
    )
    muted_style = ParagraphStyle(
        "muted",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#888888"),
        spaceAfter=2 * mm,
    )

    story = []
    top = result.top_prediction
    timestamp = result.timestamp.strftime("%d %B %Y, %H:%M")

    # ── Header ────────────────────────────────────────────────────
    story.append(Paragraph("Retinal Disease Detector", title_style))
    story.append(Paragraph("AI-Powered Fundus Screening Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#111111")))
    story.append(Spacer(1, 4 * mm))

    # Scan info row
    info_data = [
        ["Date & Time", timestamp],
        ["Inference time", f"{result.inference_time_ms:.1f} ms"],
        ["Image path", str(result.image_path.name)],
    ]
    info_table = Table(info_data, colWidths=[50 * mm, 120 * mm])
    info_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#888888")),
        ("TEXTCOLOR",   (1, 0), (1, -1), colors.HexColor("#111111")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 6 * mm))

    # ── Annotated image ───────────────────────────────────────────
    story.append(Paragraph("Fundus Image", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#DDDDDD")))
    story.append(Spacer(1, 3 * mm))

    if annotated_image_path.exists():
        img = RLImage(str(annotated_image_path),
                      width=90 * mm, height=90 * mm)
        story.append(img)

    story.append(Spacer(1, 6 * mm))

    # ── Primary diagnosis ─────────────────────────────────────────
    story.append(Paragraph("Primary Diagnosis", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#DDDDDD")))
    story.append(Spacer(1, 3 * mm))

    diag_data = [
        ["Diagnosis", top.label],
        ["Confidence", f"{top.confidence * 100:.1f}%"],
    ]
    diag_table = Table(diag_data, colWidths=[50 * mm, 120 * mm])
    diag_table.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 11),
        ("TEXTCOLOR",     (0, 0), (0, -1), colors.HexColor("#888888")),
        ("TEXTCOLOR",     (1, 0), (1, -1), colors.HexColor("#111111")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 6 * mm))

    # ── Metrics table ─────────────────────────────────────────────
    story.append(Paragraph("Clinical Metrics", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#DDDDDD")))
    story.append(Spacer(1, 3 * mm))

    metric_rows = [
        ["Metric", "Value", "Notes"],
        ["CDR value",
         f"{metrics['cdr']:.2f}" if metrics.get("cdr") is not None else "N/A",
         "Normal < 0.65"],
        ["Vessel density",
         f"{metrics.get('vessel_density', 0):.1f}%",
         "Blood vessel coverage"],
        ["Image quality score",
         f"{metrics.get('sharpness', 0):.1f}",
         "Variance of Laplacian"],
    ]

    metric_table = Table(
        metric_rows,
        colWidths=[60 * mm, 40 * mm, 70 * mm]
    )
    metric_table.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#111111")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        # Data rows
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR",     (0, 1), (-1, -1), colors.HexColor("#111111")),
        ("TEXTCOLOR",     (2, 1), (2, -1), colors.HexColor("#888888")),
        # Alternating row background
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F7F7F7"), colors.white]),
        # Grid
        ("GRID",          (0, 0), (-1, -1), 0.5,
         colors.HexColor("#DDDDDD")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 6 * mm))

    # ── All predictions ───────────────────────────────────────────
    story.append(Paragraph("All Class Predictions", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#DDDDDD")))
    story.append(Spacer(1, 3 * mm))

    pred_rows = [["Disease Class", "Confidence"]]
    for pred in result.predictions:
        pred_rows.append([
            pred.label,
            f"{pred.confidence * 100:.1f}%"
        ])

    pred_table = Table(pred_rows, colWidths=[100 * mm, 70 * mm])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#111111")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",     (0, 1), (-1, -1), colors.HexColor("#111111")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F7F7F7"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.5,
         colors.HexColor("#DDDDDD")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 6 * mm))

    # ── Recommendation ────────────────────────────────────────────
    story.append(Paragraph("Recommendation", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#DDDDDD")))
    story.append(Spacer(1, 3 * mm))

    rec = _get_recommendation(
        top.label,
        metrics.get("cdr")
    )
    story.append(Paragraph(rec["title"], body_style))
    story.append(Paragraph(rec["detail"], muted_style))
    story.append(Spacer(1, 6 * mm))

    # ── Disclaimer ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#DDDDDD")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI screening tool "
        "and is intended for research and educational purposes only. "
        "It does not constitute a medical diagnosis. Always consult a "
        "qualified ophthalmologist for clinical decisions.",
        muted_style
    ))

    doc.build(story)
    return output_path


def _get_recommendation(diagnosis: str, cdr: Optional[float]) -> dict:
    cdr_known = cdr is not None
    if diagnosis == "Normal" and (not cdr_known or cdr < 0.65):
        return {
            "title": "No action needed",
            "detail": "No signs of retinal disease detected. "
                      "Routine check-up in 12 months recommended."
        }
    elif diagnosis == "Glaucoma" or (cdr_known and cdr >= 0.65):
        cdr_status = "elevated" if cdr >= 0.65 else "within normal range"
        cdr_text = f"{cdr:.2f}" if cdr_known else "unavailable"
        return {
            "title": "Refer to ophthalmologist",
            "detail": f"CDR: {cdr_text} ({cdr_status if cdr_known else 'not measured'}). Glaucoma screening strongly recommended."
        }
    elif diagnosis == "Diabetic Retinopathy":
        return {
            "title": "Urgent referral recommended",
            "detail": "Diabetic retinopathy indicators detected. Immediate ophthalmology review needed."
        }
    elif diagnosis in ("Cataract", "AMD"):
        return {
            "title": "Specialist consultation advised",
            "detail": f"{diagnosis} indicators found. "
                      "Refer to ophthalmologist for confirmation."
        }
    return {
        "title": "Follow-up recommended",
        "detail": "Findings detected. Please consult an eye care professional."
    }