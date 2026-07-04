"""
report_generator.py — Phase 3: Automated PDF Candidate Report

Generates a professional PDF report for each planet candidate containing:
  - Cover page with target metadata
  - Light curve plot
  - Phase-folded transit
  - BLS periodogram
  - MCMC corner plot (if available)
  - Parameter table with uncertainties
  - Classification confidence breakdown
  - Data quality metrics
  - Sector summary table (for batch reports)

Usage:
    from report_generator import generate_candidate_report, generate_sector_report
    pdf_bytes = generate_candidate_report(candidate_data, mcmc_data)
"""

import io, base64, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect
from PIL import Image as PILImage


# ── Colour palette (matches the dark UI) ──────────────────────────────────

C_BG     = colors.HexColor("#050B1A")
C_PANEL  = colors.HexColor("#0A1628")
C_CYAN   = colors.HexColor("#00D4FF")
C_AMBER  = colors.HexColor("#FFB347")
C_GREEN  = colors.HexColor("#39FF14")
C_RED    = colors.HexColor("#FF4757")
C_PURPLE = colors.HexColor("#A78BFA")
C_SLATE  = colors.HexColor("#8899BB")
C_WHITE  = colors.HexColor("#E8F0FF")
C_BORDER = colors.HexColor("#0E2040")

# Light versions for print legibility on white background
C_DARK_BLUE  = colors.HexColor("#0A2456")
C_MID_BLUE   = colors.HexColor("#1A3A6E")
C_LIGHT_BLUE = colors.HexColor("#EBF4FF")
C_TEXT       = colors.HexColor("#1A202C")
C_SUBTEXT    = colors.HexColor("#4A5568")
C_ACCENT     = colors.HexColor("#0070CC")


# ── Matplotlib plot helpers ────────────────────────────────────────────────

def _mpl_to_rl_image(fig, width_cm=16):
    """Convert a matplotlib figure to a ReportLab Image object."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = RLImage(buf, width=width_cm * cm,
                  height=width_cm * cm * 0.45)
    return img


def _b64_to_rl_image(b64_str, width_cm=16, aspect=0.7):
    """Convert a base64 PNG string to a ReportLab Image."""
    buf = io.BytesIO(base64.b64decode(b64_str))
    img = RLImage(buf, width=width_cm * cm,
                  height=width_cm * cm * aspect)
    return img


def _plot_light_curve(lc_data, model_flux=None):
    """Generate light curve matplotlib figure."""
    time = np.array(lc_data["time"])
    flux = np.array(lc_data["flux"])

    fig, ax = plt.subplots(figsize=(12, 3.5))
    stride = max(1, len(time) // 3000)
    ax.plot(time[::stride], flux[::stride],
            color="#378ADD", linewidth=0.6, alpha=0.8, label="TESS data")

    if model_flux is not None and len(model_flux) == len(time):
        ax.plot(time[::stride], np.array(model_flux)[::stride],
                color="#E89820", linewidth=1.8, label="batman model", zorder=5)
        ax.legend(fontsize=8)

    ax.set_xlabel("Time [BTJD]", fontsize=10)
    ax.set_ylabel("Normalized Flux", fontsize=10)
    ax.set_title("TESS Light Curve (PDCSAP)", fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


def _plot_phase_folded(phase_data, period):
    """Generate phase-folded plot."""
    phase = np.array(phase_data["phase"])
    flux  = np.array(phase_data["flux"])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    stride = max(1, len(phase) // 2000)
    ax.scatter(phase[::stride], flux[::stride],
               s=1.5, alpha=0.5, color="#378ADD", rasterized=True)
    ax.axvline(0, color="#E89820", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Orbital Phase", fontsize=10)
    ax.set_ylabel("Normalized Flux", fontsize=10)
    ax.set_title(f"Phase-folded Transit  (P = {period:.4f} d)", fontsize=11)
    ax.set_xlim(-0.5, 0.5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig


def _plot_attention(attn_weights, global_len=201, patch_size=3):
    """Plot attention heatmap as a bar chart."""
    if attn_weights is None:
        return None
    attn  = np.array(attn_weights)
    n_pat = len(attn)
    # Map patch indices to phase bins
    phase = np.linspace(-0.5, 0.5, n_pat)

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.bar(phase, attn, width=1.0 / n_pat, color="#7C3AED", alpha=0.85,
           label="Attention weight")
    ax.axvline(0, color="#E89820", linewidth=1, linestyle="--")
    ax.set_xlabel("Orbital Phase", fontsize=10)
    ax.set_ylabel("Attention", fontsize=10)
    ax.set_title("TransitFormer Attention Weights (XAI)", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 0.5)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")
    fig.tight_layout()
    return fig


# ── Style helpers ──────────────────────────────────────────────────────────

def _styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title2", fontSize=22, fontName="Helvetica-Bold",
            textColor=C_DARK_BLUE, spaceAfter=4, alignment=TA_CENTER),
        "subtitle": ParagraphStyle(
            "Subtitle2", fontSize=13, fontName="Helvetica",
            textColor=C_SUBTEXT, spaceAfter=2, alignment=TA_CENTER),
        "h2": ParagraphStyle(
            "H2", fontSize=13, fontName="Helvetica-Bold",
            textColor=C_DARK_BLUE, spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle(
            "H3", fontSize=11, fontName="Helvetica-Bold",
            textColor=C_MID_BLUE, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle(
            "Body2", fontSize=9, fontName="Helvetica",
            textColor=C_TEXT, leading=14),
        "small": ParagraphStyle(
            "Small", fontSize=8, fontName="Helvetica",
            textColor=C_SUBTEXT, leading=11),
        "mono": ParagraphStyle(
            "Mono", fontSize=8, fontName="Courier",
            textColor=C_TEXT, leading=12),
        "badge": ParagraphStyle(
            "Badge", fontSize=12, fontName="Helvetica-Bold",
            textColor=colors.white, alignment=TA_CENTER),
    }


def _class_color(class_name):
    return {
        "Exoplanet Transit": C_ACCENT,
        "Eclipsing Binary":  colors.HexColor("#C05621"),
        "Stellar Blend":     colors.HexColor("#C53030"),
        "Starspot":          colors.HexColor("#276749"),
    }.get(class_name, C_DARK_BLUE)


def _param_table(params, mcmc_params=None):
    """Build a styled parameter comparison table."""
    header = ["Parameter", "BLS Estimate", "MCMC Median", "−1σ", "+1σ", "Unit"]
    rows   = [header]

    bls_map = {
        "Period":   (str(params.get("period_days", "—")),    "days"),
        "Depth":    (str(params.get("depth_pct", "—")) + "%",""),
        "Duration": (str(params.get("duration_hours", "—")), "hours"),
        "T₀":       (str(params.get("t0_bjd", "—")),         "BTJD"),
        "Rp/R★":    (str(params.get("rp_rs", "—")),          ""),
        "SNR":      (str(params.get("snr", "—")),             "σ"),
    }

    mcmc_key_map = {
        "Period":   "Period (d)",
        "Depth":    "Depth (ppm)",
        "Duration": None,
        "T₀":       "T₀ (BTJD)",
        "Rp/R★":    "Rp/R★",
        "SNR":      None,
    }

    for param, (bls_val, unit) in bls_map.items():
        mcmc_med = mcmc_elo = mcmc_ehi = "—"
        if mcmc_params:
            mk = mcmc_key_map.get(param)
            if mk and mk in mcmc_params:
                d = mcmc_params[mk]
                mcmc_med = f"{d['median']:.5f}"
                mcmc_elo = f"{d['err_low']:.5f}"
                mcmc_ehi = f"{d['err_high']:.5f}"
        rows.append([param, bls_val, mcmc_med, mcmc_elo, mcmc_ehi, unit])

    col_widths = [3.2*cm, 3.2*cm, 3.2*cm, 2.4*cm, 2.4*cm, 2.0*cm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  C_DARK_BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  8),
        ("FONTSIZE",    (0,1), (-1,-1), 8),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LIGHT_BLUE, colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, C_BORDER),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    return t


def _classification_table(probs):
    rows = [["Classification", "Probability", "Interpretation"]]
    interp = {
        "Exoplanet Transit": "Periodic transit by orbiting planet",
        "Eclipsing Binary":  "Stellar eclipsing binary system",
        "Stellar Blend":     "Diluted signal from background EB",
        "Starspot":          "Rotational modulation / starspot",
    }
    for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
        rows.append([cls, f"{prob:.1f}%", interp.get(cls, "")])

    t = Table(rows, colWidths=[5*cm, 3*cm, 8.4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  C_MID_BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LIGHT_BLUE, colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, C_BORDER),
        ("ALIGN",       (1,1), (1,-1),  "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    return t


# ── Main report builder ────────────────────────────────────────────────────

def generate_candidate_report(scan_data: dict, mcmc_data: dict = None) -> bytes:
    """
    Generate a PDF report for a single transit candidate.

    scan_data : output of fetch_and_analyse() from server_phase2.py
    mcmc_data : output of run_mcmc() (optional, adds MCMC section)

    Returns raw PDF bytes.
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    s      = _styles()
    story  = []
    now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    target = scan_data.get("target", "Unknown")
    cls    = scan_data.get("classification", {})
    tp     = scan_data.get("transit_params", {})
    dq     = scan_data.get("data_quality", {})

    # ── Cover ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("◎ EXODETECT", s["title"]))
    story.append(Paragraph("Automated Transit Candidate Report", s["subtitle"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT))
    story.append(Spacer(1, 0.4*cm))

    # Target info
    top_class = cls.get("top_class", "Unknown")
    conf      = cls.get("confidence", 0)
    story.append(Paragraph(f"Target: <b>{target}</b>", s["h2"]))
    story.append(Paragraph(
        f"Classification: <b>{top_class}</b> &nbsp; ({conf:.1f}% confidence) &nbsp;|&nbsp; "
        f"Generated: {now} &nbsp;|&nbsp; "
        f"TESS Sector {scan_data.get('sector', '?')} · "
        f"{scan_data.get('n_cadences', 0):,} cadences &nbsp;|&nbsp; "
        f"Classifier: {scan_data.get('classifier', 'CNN')}",
        s["body"]))
    story.append(Spacer(1, 0.5*cm))

    # Quick-stats row
    qs_data = [
        ["Period", "Depth", "Duration", "SNR", "Rp/R★", "N Transits"],
        [
            f"{tp.get('period_days','—')} d",
            f"{tp.get('depth_pct','—')} %",
            f"{tp.get('duration_hours','—')} hr",
            f"{tp.get('snr','—')} σ",
            str(tp.get('rp_rs','—')),
            str(tp.get('n_transits','—')),
        ]
    ]
    qs_table = Table(qs_data, colWidths=[2.7*cm]*6)
    qs_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  C_DARK_BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,1), (-1,1),  C_LIGHT_BLUE),
        ("FONTNAME",      (0,1), (-1,1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,1), (-1,1),  C_ACCENT),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("BOX",           (0,0), (-1,-1), 1, C_ACCENT),
        ("INNERGRID",     (0,0), (-1,-1), 0.5, C_BORDER),
    ]))
    story.append(qs_table)
    story.append(Spacer(1, 0.6*cm))

    # ── Light curve ────────────────────────────────────────────────────────
    story.append(Paragraph("1. Light Curve", s["h2"]))
    if scan_data.get("light_curve"):
        model_flux = mcmc_data.get("model_flux") if mcmc_data else None
        fig = _plot_light_curve(scan_data["light_curve"], model_flux)
        story.append(_mpl_to_rl_image(fig, width_cm=16))
        if model_flux:
            story.append(Paragraph(
                "Orange line: best-fit batman transit model from MCMC sampling.", s["small"]))
    story.append(Spacer(1, 0.4*cm))

    # ── Phase-folded + attention ───────────────────────────────────────────
    story.append(Paragraph("2. Phase-folded Transit", s["h2"]))
    pf = scan_data.get("phase_folded")
    if pf:
        fig = _plot_phase_folded(pf, tp.get("period_days", 1.0))
        story.append(_mpl_to_rl_image(fig, width_cm=16))
    story.append(Spacer(1, 0.4*cm))

    # ── XAI Attention heatmap ──────────────────────────────────────────────
    if scan_data.get("attention_weights"):
        story.append(Paragraph("3. TransitFormer Attention Heatmap (XAI)", s["h2"]))
        fig = _plot_attention(scan_data["attention_weights"])
        if fig:
            story.append(_mpl_to_rl_image(fig, width_cm=16))
        story.append(Paragraph(
            "Bars show which orbital-phase bins the TransitFormer model attended to most "
            "strongly when making its classification decision. High attention near phase=0 "
            "is consistent with a real transit signal.", s["small"]))
        story.append(Spacer(1, 0.4*cm))

    # ── Parameter table ────────────────────────────────────────────────────
    story.append(Paragraph("4. Parameter Estimates", s["h2"]))
    mcmc_pct = mcmc_data.get("percentiles") if mcmc_data else None
    story.append(_param_table(tp, mcmc_pct))
    story.append(Spacer(1, 0.4*cm))

    # ── MCMC section ───────────────────────────────────────────────────────
    if mcmc_data:
        story.append(Paragraph("5. MCMC Posterior (batman + emcee)", s["h2"]))
        story.append(Paragraph(
            f"Walkers: 32 &nbsp;|&nbsp; Steps: 2000 &nbsp;|&nbsp; "
            f"Burn-in: 500 &nbsp;|&nbsp; "
            f"Acceptance: {mcmc_data.get('acceptance_frac', 0)*100:.1f}% &nbsp;|&nbsp; "
            f"Elapsed: {mcmc_data.get('elapsed_sec', '—')}s",
            s["body"]))
        story.append(Spacer(1, 0.3*cm))

        if mcmc_data.get("plots", {}).get("posterior"):
            story.append(Paragraph("Phase-folded posterior model:", s["h3"]))
            story.append(_b64_to_rl_image(
                mcmc_data["plots"]["posterior"], width_cm=16, aspect=0.55))
            story.append(Spacer(1, 0.4*cm))

        if mcmc_data.get("plots", {}).get("corner"):
            story.append(Paragraph("Posterior corner plot:", s["h3"]))
            story.append(_b64_to_rl_image(
                mcmc_data["plots"]["corner"], width_cm=16, aspect=0.85))
            story.append(Spacer(1, 0.3*cm))

    # ── Classification ─────────────────────────────────────────────────────
    sec_num = 6 if mcmc_data else 5
    story.append(Paragraph(f"{sec_num}. AI Classification", s["h2"]))
    story.append(_classification_table(cls.get("probabilities", {})))
    story.append(Spacer(1, 0.4*cm))

    # ── Data quality ───────────────────────────────────────────────────────
    story.append(Paragraph(f"{sec_num+1}. Data Quality", s["h2"]))
    dq_rows = [
        ["Metric", "Value", "Assessment"],
        ["Completeness", f"{dq.get('completeness_pct','—')}%",
         "Good" if dq.get("completeness_pct", 0) > 90 else "Fair"],
        ["CDPP Noise",   f"{dq.get('cdpp_ppm','—')} ppm",
         "Low" if dq.get("cdpp_ppm", 999) < 300 else "High"],
        ["Systematic",   f"{dq.get('sys_noise_ppm','—')} ppm", "Estimated 15% of CDPP"],
    ]
    dq_table = Table(dq_rows, colWidths=[5*cm, 4*cm, 7.4*cm])
    dq_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), C_DARK_BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LIGHT_BLUE, colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, C_BORDER),
        ("ALIGN",       (1,1), (1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story.append(dq_table)
    story.append(Spacer(1, 0.6*cm))

    # ── Footer ─────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated by ExoDetect Phase 3 · {now} · "
        "Data from NASA/MIT TESS (MAST archive) · "
        "This is an automated analysis — follow-up observations recommended for candidates.",
        s["small"]))

    doc.build(story)
    return buf.getvalue()


def generate_sector_report(sector_number: int, results: list) -> bytes:
    """
    Generate a sector-level summary PDF with a ranked candidate table.

    results : list of dicts from batch_pipeline.run_sector()
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    s     = _styles()
    story = []
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    story.append(Paragraph("◎ EXODETECT", s["title"]))
    story.append(Paragraph(f"TESS Sector {sector_number} — Batch Scan Report", s["subtitle"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT))
    story.append(Spacer(1, 0.4*cm))

    ok      = [r for r in results if r.get("status") == "ok"]
    cands   = [r for r in ok if r.get("flag_mcmc")]
    story.append(Paragraph(
        f"Generated: {now} &nbsp;|&nbsp; "
        f"Targets processed: {len(ok)} &nbsp;|&nbsp; "
        f"Candidates flagged: {len(cands)}",
        s["body"]))
    story.append(Spacer(1, 0.6*cm))

    # Ranked candidate table
    story.append(Paragraph("Planet Candidates (ranked by P(planet))", s["h2"]))
    header = ["TIC ID", "P(planet)", "Period (d)", "Depth (%)", "SNR", "CNN", "TF", "Flag"]
    rows   = [header]
    for r in sorted(ok, key=lambda x: -x.get("p_planet", 0))[:30]:
        rows.append([
            r.get("tic_id", "—"),
            f"{r.get('p_planet', 0)*100:.1f}%",
            f"{r.get('period_days', 0):.3f}",
            f"{r.get('depth_pct', 0):.3f}",
            f"{r.get('snr', 0):.1f}σ",
            f"{r.get('p_cnn', 0)*100:.0f}%",
            f"{r.get('p_transformer', 0)*100:.0f}%",
            "★ MCMC" if r.get("flag_mcmc") else "",
        ])

    col_w = [2.8*cm, 2.2*cm, 2.4*cm, 2.4*cm, 1.8*cm, 1.6*cm, 1.6*cm, 1.6*cm]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), C_DARK_BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 7.5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [C_LIGHT_BLUE, colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.4, C_BORDER),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"ExoDetect Phase 3 · Sector {sector_number} · {now}", s["small"]))

    doc.build(story)
    return buf.getvalue()