"""
PDF Report Generator for Accident Detection Results.
Produces a professional incident report containing:
  - Analysis metadata (video name, model, thresholds, duration)
  - Executive summary statistics
  - Accident interval table
  - Embedded accident vehicle images (all saved frames)
  - Full frame-by-frame timeline table (optional)

Unicode note: fpdf2's built-in Helvetica only covers Latin-1.
Every string passed to pdf.cell() is filtered through _safe() which:
  1. Replaces known emoji with ASCII equivalents.
  2. Strips any remaining character outside the Latin-1 range.
"""

import os
import re
from pathlib import Path
from datetime import datetime

try:
    from fpdf import FPDF, XPos, YPos
    FPDF2_AVAILABLE = True
except ImportError:
    FPDF2_AVAILABLE = False

# ── Colour constants (R, G, B) ──────────────────────────────────────────────
C_RED    = (220, 38,  38)
C_GREEN  = (22,  163, 74)
C_DARK   = (17,  24,  39)
C_GREY   = (107, 114, 128)
C_LIGHT  = (243, 244, 246)
C_WHITE  = (255, 255, 255)
C_ACCENT = (30,  58,  138)

# ── Emoji / special-char substitution table ──────────────────────────────────
_EMOJI_MAP = {
    "🚨": "[ALERT]",
    "✅": "[OK]",
    "❌": "[X]",
    "📋": "",
    "📈": "",
    "🖼️": "",
    "🖼": "",
    "📥": "",
    "📄": "",
    "📊": "",
    "📂": "",
    "📹": "",
    "🎥": "",
    "📷": "",
    "🚗": "",
    "⭐": "*",
    "🏆": "[BEST]",
    "🌊": "",
    "🔥": "",
    "⬇️": "",
    "⬇": "",
    "🖨️": "",
    "🖨": "",
    "ℹ️": "[i]",
    "ℹ": "[i]",
    "–":  "-",
    "—":  "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
}

# Matches any char outside Latin-1 (0x00-0xFF)
_NON_LATIN1 = re.compile(r"[^\x00-\xff]")


def _safe(text: str) -> str:
    """Return a Latin-1-safe version of *text* suitable for Helvetica/fpdf2."""
    if not isinstance(text, str):
        text = str(text)
    for src, dst in _EMOJI_MAP.items():
        text = text.replace(src, dst)
    # Drop any character still outside Latin-1
    text = _NON_LATIN1.sub("", text)
    return text


# ── PDF class ────────────────────────────────────────────────────────────────
class IncidentReportPDF(FPDF):
    """Custom FPDF subclass with branded header and footer."""

    def __init__(self, video_name: str, analysis_ts: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.video_name  = _safe(video_name)
        self.analysis_ts = _safe(analysis_ts)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(15, 15, 15)

    def header(self):
        self.set_fill_color(*C_ACCENT)
        self.rect(0, 0, 210, 14, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_WHITE)
        self.set_y(3)
        self.cell(0, 8, "AI Traffic Accident Detection System  |  Incident Report", align="C")
        self.set_text_color(*C_DARK)
        self.ln(10)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*C_GREY)
        self.cell(
            0, 10,
            f"Page {self.page_no()} | Generated: {self.analysis_ts} | Video: {self.video_name}",
            align="C"
        )

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*C_ACCENT)
        self.ln(4)
        self.cell(0, 8, _safe(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.5)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)
        self.set_text_color(*C_DARK)

    def kv_row(self, key: str, value: str, shade: bool = False):
        fill = shade
        if fill:
            self.set_fill_color(*C_LIGHT)
        self.set_font("Helvetica", "B", 10)
        self.cell(60, 7, _safe(key), border=0, fill=fill)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, _safe(value), border=0, fill=fill,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ── Main report builder ──────────────────────────────────────────────────────
def generate_pdf_report(video_res: dict, timeline_df=None) -> bytes:
    """
    Build a full incident PDF report and return the raw bytes.

    Parameters
    ----------
    video_res   : dict returned by HybridInferenceEngine.process_video_stream()
    timeline_df : optional pandas DataFrame of frame-level telemetry
    """
    if not FPDF2_AVAILABLE:
        raise ImportError("fpdf2 is required. Run: pip install fpdf2")

    video_stem      = video_res.get("video_stem", "unknown")
    analysis_ts     = video_res.get("analysis_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    accident_frames = video_res.get("accident_frames", [])
    intervals       = video_res.get("accident_intervals", [])
    model_key       = video_res.get("model_key", "ensemble")
    conf_threshold  = video_res.get("conf_threshold", 0.5)
    frame_skip      = video_res.get("frame_skip", 5)
    total_frames    = video_res.get("total_frames", 0)
    processed       = video_res.get("processed_frames", 0)
    fps             = video_res.get("fps", 25.0)
    duration_s      = video_res.get("duration_s", 0.0)
    elapsed         = video_res.get("elapsed_time_s", 0.0)
    fps_speed       = video_res.get("fps_speed", 0.0)
    accident_dir    = video_res.get("accident_images_dir", "")

    accident_count = len(accident_frames)
    accident_ratio = (accident_count / processed * 100) if processed > 0 else 0.0
    peak_conf      = max((af["confidence"] for af in accident_frames), default=0.0)
    duration_str   = f"{int(duration_s // 60):02d}:{duration_s % 60:05.2f}"
    report_dt      = datetime.strptime(analysis_ts, "%Y%m%d_%H%M%S").strftime("%d %b %Y  %H:%M:%S")

    # ── Init ──────────────────────────────────────────────────────────────────
    pdf = IncidentReportPDF(video_name=video_stem, analysis_ts=report_dt)
    pdf.add_page()

    # ── Title ─────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*C_ACCENT)
    pdf.cell(0, 12, "TRAFFIC INCIDENT ANALYSIS REPORT",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*C_GREY)
    pdf.cell(0, 7, _safe(f"Multi-Modal Hybrid Deep Learning  |  {report_dt}"),
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Verdict banner ────────────────────────────────────────────────────────
    if accident_count > 0:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_draw_color(*C_RED)
        banner_txt = f"  [ALERT]  ACCIDENT DETECTED  -  {accident_count} incident frame(s) identified"
        clr = C_RED
    else:
        pdf.set_fill_color(220, 252, 231)
        pdf.set_draw_color(*C_GREEN)
        banner_txt = "  [OK]  NO ACCIDENT DETECTED  -  Normal traffic flow throughout"
        clr = C_GREEN
    pdf.set_line_width(1.0)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*clr)
    pdf.rect(15, pdf.get_y(), 180, 14, "FD")
    pdf.cell(0, 14, banner_txt, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_line_width(0.2)
    pdf.set_text_color(*C_DARK)
    pdf.ln(6)

    # ── Section 1: Metadata ───────────────────────────────────────────────────
    pdf.section_title("1. Analysis Metadata")
    rows = [
        ("Video File",          video_stem),
        ("Report Generated",    report_dt),
        ("Model Architecture",  model_key.upper()),
        ("Detection Threshold", f"{conf_threshold * 100:.0f}%"),
        ("Frame Sampling (K)",  f"Every {frame_skip} frames"),
        ("Video Duration",      duration_str),
        ("Total Frames",        f"{total_frames:,}"),
        ("Frames Analysed",     f"{processed:,}"),
        ("Video FPS",           f"{fps:.2f}"),
        ("Processing Time",     f"{elapsed:.1f} s  ({fps_speed:.1f} fps)"),
        ("Accident Images Dir", accident_dir if accident_dir else "N/A"),
    ]
    for i, (k, v) in enumerate(rows):
        pdf.kv_row(k + ":", v, shade=(i % 2 == 0))
    pdf.ln(4)

    # ── Section 2: Executive Summary ──────────────────────────────────────────
    pdf.section_title("2. Executive Summary")
    summary_cols = [
        ("Frames Analysed",    str(processed)),
        ("Accident Frames",    str(accident_count)),
        ("Accident Ratio",     f"{accident_ratio:.1f}%"),
        ("Peak Confidence",    f"{peak_conf:.1f}%"),
        ("Incident Intervals", str(len(intervals))),
    ]
    col_w = 180 / len(summary_cols)
    # header
    pdf.set_fill_color(*C_ACCENT)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font("Helvetica", "B", 10)
    for lbl, _ in summary_cols:
        pdf.cell(col_w, 8, lbl, border=1, align="C", fill=True)
    pdf.ln()
    # values
    pdf.set_fill_color(*C_LIGHT)
    pdf.set_text_color(*C_DARK)
    pdf.set_font("Helvetica", "B", 12)
    for _, val in summary_cols:
        pdf.cell(col_w, 10, val, border=1, align="C", fill=True)
    pdf.ln(6)

    # ── Section 3: Incident Intervals ─────────────────────────────────────────
    pdf.section_title("3. Incident Interval Table")
    if intervals:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(*C_ACCENT)
        pdf.set_text_color(*C_WHITE)
        for w, h in [(15, "#"), (80, "Start Timestamp"), (80, "End Timestamp")]:
            pdf.cell(w, 8, h, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*C_DARK)
        for idx, (start, end) in enumerate(intervals, 1):
            fill = idx % 2 == 0
            pdf.set_fill_color(*C_LIGHT)
            pdf.cell(15, 7, str(idx),       border=1, align="C", fill=fill)
            pdf.cell(80, 7, _safe(start),   border=1, align="C", fill=fill)
            pdf.cell(80, 7, _safe(end),     border=1, align="C", fill=fill)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*C_GREY)
        pdf.cell(0, 8, "No accident intervals detected in this video.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*C_DARK)
    pdf.ln(4)

    # ── Section 4: Accident Frame Log ─────────────────────────────────────────
    pdf.section_title("4. Accident Frame Log")
    if accident_frames:
        col_widths = [12, 30, 28, 110]
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(*C_ACCENT)
        pdf.set_text_color(*C_WHITE)
        for w, h in zip(col_widths, ["#", "Frame #", "Timestamp", "Saved Image Filename"]):
            pdf.cell(w, 8, h, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*C_DARK)
        for idx, af in enumerate(accident_frames, 1):
            fill = idx % 2 == 0
            pdf.set_fill_color(*C_LIGHT)
            pdf.cell(col_widths[0], 7, str(idx),                             border=1, align="C", fill=fill)
            pdf.cell(col_widths[1], 7, str(af["frame_idx"]),                 border=1, align="C", fill=fill)
            pdf.cell(col_widths[2], 7, _safe(af["timestamp_str"]),           border=1, align="C", fill=fill)
            pdf.cell(col_widths[3], 7, _safe(af.get("image_filename", "")),  border=1,            fill=fill)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*C_GREY)
        pdf.cell(0, 8, "No accident frames were captured.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*C_DARK)
    pdf.ln(4)

    # ── Section 5: Accident Vehicle Image Gallery ──────────────────────────────
    if accident_frames:
        pdf.section_title("5. Accident Vehicle Image Gallery")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*C_GREY)
        pdf.cell(
            0, 6,
            _safe(f"Showing {len(accident_frames)} accident frame(s).  "
                  f"Images saved to: {accident_dir}"),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        pdf.set_text_color(*C_DARK)
        pdf.ln(3)

        images_per_row = 2
        img_w = 85
        img_h = 55
        x_positions = [15, 110]
        col_idx = 0

        for af in accident_frames:
            img_path = af.get("image_path", "")

            # Fallback: regenerate from PIL image if path missing
            if img_path and not Path(img_path).exists():
                pil = af.get("image")
                if pil:
                    pil.save(img_path)

            if not (img_path and Path(img_path).exists()):
                continue

            x = x_positions[col_idx]
            y = pdf.get_y()

            # New page if not enough room
            if y + img_h + 20 > pdf.h - pdf.b_margin:
                pdf.add_page()
                y = pdf.get_y()

            # Caption
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*C_RED)
            pdf.set_x(x)
            caption = _safe(
                f"Frame {af['frame_idx']}  |  {af['timestamp_str']}  |  "
                f"Conf: {af['confidence']:.1f}%"
            )
            pdf.cell(img_w, 5, caption)
            pdf.ln(5)
            y_img = pdf.get_y()

            # Image
            try:
                pdf.image(img_path, x=x, y=y_img, w=img_w, h=img_h)
            except Exception:
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(*C_GREY)
                pdf.set_x(x)
                pdf.cell(img_w, img_h, "[Image unavailable]", border=1, align="C")

            col_idx += 1
            if col_idx >= images_per_row:
                col_idx = 0
                pdf.set_y(y_img + img_h + 8)
                pdf.set_text_color(*C_DARK)
            else:
                pdf.set_xy(x_positions[col_idx], y)

        if col_idx != 0:
            pdf.ln(img_h + 8)
        pdf.set_text_color(*C_DARK)
        pdf.ln(4)

    # ── Section 6: Telemetry table (up to 200 rows) ───────────────────────────
    if timeline_df is not None and len(timeline_df) > 0:
        pdf.add_page()
        pdf.section_title("6. Frame-by-Frame Telemetry (Summary)")
        display_df = timeline_df.head(200)
        col_ws     = [25, 35, 40, 30]
        col_labels = ["Frame #", "Timestamp", "Acc. Prob (%)", "Accident?"]

        pdf.set_fill_color(*C_ACCENT)
        pdf.set_text_color(*C_WHITE)
        pdf.set_font("Helvetica", "B", 8)
        for cw, cl in zip(col_ws, col_labels):
            pdf.cell(cw, 7, cl, border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*C_DARK)
        for _, row in display_df.iterrows():
            is_acc = bool(row["is_accident"])
            if is_acc:
                pdf.set_fill_color(254, 226, 226)
            else:
                pdf.set_fill_color(*C_WHITE)
            pdf.cell(col_ws[0], 6, str(int(row["frame_idx"])),        border=1, align="C", fill=is_acc)
            pdf.cell(col_ws[1], 6, _safe(str(row["timestamp_str"])),  border=1, align="C", fill=is_acc)
            pdf.cell(col_ws[2], 6, f"{row['accident_prob']:.2f}%",    border=1, align="C", fill=is_acc)
            pdf.cell(col_ws[3], 6, "ACCIDENT" if is_acc else "Normal",border=1, align="C", fill=is_acc)
            pdf.ln()

        if len(timeline_df) > 200:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*C_GREY)
            pdf.cell(
                0, 7,
                f"... {len(timeline_df) - 200} more rows in the exported CSV file.",
                new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )

    return bytes(pdf.output())
