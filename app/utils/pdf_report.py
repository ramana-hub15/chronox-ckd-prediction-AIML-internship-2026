"""PDF report generator for ChronoX screening results using ReportLab."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

if TYPE_CHECKING:
    from app.models import Prediction, User


def build_prediction_pdf(prediction: "Prediction", user: "User") -> bytes:
    """Render a clean clinical-style PDF summary for a stored prediction."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=f"ChronoX CKD Screening Report #{prediction.id}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChronoXTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#0B3C5D"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ChronoXSub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "ChronoXSection",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#0B3C5D"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ChronoXBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1F2937"),
    )

    story = [
        Paragraph("ChronoX — Chronic Kidney Disease Screening Report", title_style),
        Paragraph(
            f"Report ID: #{prediction.id} &nbsp;|&nbsp; Label: {prediction.patient_label} "
            f"&nbsp;|&nbsp; Generated for: {user.full_name} ({user.email}) "
            f"&nbsp;|&nbsp; Date: {prediction.created_at.strftime('%Y-%m-%d %H:%M UTC')}",
            subtitle_style,
        ),
    ]

    summary_data = [
        ["Overall Risk Score", f"{prediction.risk_score:.1f}% ({prediction.risk_band} Risk)"],
        ["Selected Model", prediction.model_used.upper()],
        ["ANN / Logistic Breakdown", f"ANN: {prediction.ann_score:.1f}%  |  Logistic: {prediction.logistic_score:.1f}%"],
        ["CKD Stage Hint", f"{prediction.stage_hint} (Approx. eGFR: {prediction.egfr_approx} mL/min/1.73m²)"],
    ]
    summary_table = Table(summary_data, colWidths=[2.2 * inch, 4.9 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E6F7F5")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0B3C5D")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Clinical Parameters & Risk Contributions", section_style))
    rows = [["Parameter", "Input Value", "Healthy Reference", "Risk Impact"]]
    for item in prediction.contributions:
        val = "Yes" if item["feature"] == "htn" and item["value"] == 1 else (
            "No" if item["feature"] == "htn" else f"{item['value']} {item['unit']}".strip()
        )
        ref = "No" if item["feature"] == "htn" else f"{item['baseline']} {item['unit']}".strip()
        sign = "+" if item["impact_pct"] > 0 else ""
        rows.append([item["label"], val, ref, f"{sign}{item['impact_pct']:.2f}%"])

    param_table = Table(rows, colWidths=[2.5 * inch, 1.5 * inch, 1.7 * inch, 1.4 * inch])
    param_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3C5D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(param_table)

    story.append(Paragraph("Diagnostic & Preventive Guidance", section_style))
    category_titles = {
        "nephrologist": "When to See a Nephrologist",
        "tests": "Recommended Follow-up Tests",
        "diet": "Dietary Guidance",
        "lifestyle": "Lifestyle Habits",
    }
    for key, heading in category_titles.items():
        items = prediction.suggestions.get(key, [])
        if items:
            story.append(Paragraph(f"<b>{heading}:</b>", body_style))
            for tip in items:
                story.append(Paragraph(f"&bull; {tip}", body_style))
            story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.HexColor("#6B7280"),
        backColor=colors.HexColor("#FEF2F2"),
        borderPadding=8,
    )
    story.append(
        Paragraph(
            "<b>Medical Disclaimer:</b> This tool is for educational and screening support only "
            "and is not a substitute for professional medical diagnosis, clinical judgment, or "
            "laboratory confirmation.",
            disclaimer_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()
