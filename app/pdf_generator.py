"""
pdf_generator.py
Converts Executive Summary & Charts → Full PDF Report
Uses pdfkit + wkhtmltopdf (Windows compatible)
"""

from pathlib import Path
from markdown import markdown
import pdfkit
import datetime


REPORT_DIR = Path("reports")
IMG_DIR = REPORT_DIR / "img"
EXEC_MD = REPORT_DIR / "executive_summary.md"
EXEC_PDF = REPORT_DIR / "executive_summary.pdf"

# Path to wkhtmltopdf (update if installed somewhere else)
WKHTMLTOPDF_PATH = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"



def embed_images_html():
    """Generate HTML blocks for all persona charts."""
    if not IMG_DIR.exists():
        return "<p>No charts found.</p>"

    html_blocks = ""
    for img in sorted(IMG_DIR.glob("*.png")):
        rel_path = img.as_posix()

        html_blocks += f"""
        <div style="page-break-inside: avoid; margin-bottom: 25px;">
            <h3 style="color:#0A84FF;">{img.stem.replace('_',' ').title()}</h3>
            <img src="{rel_path}" style="width:100%; border:1px solid #ddd; border-radius:8px; margin-top:8px;">
        </div>
        """

    return html_blocks


def generate_pdf_summary():
    """Convert executive_summary.md + charts → styled PDF."""
    if not EXEC_MD.exists():
        raise FileNotFoundError("executive_summary.md missing. Run Generate Reports first.")

    # Load markdown
    md_text = EXEC_MD.read_text(encoding="utf-8")
    html_body = markdown(md_text)

    # Embed charts
    charts_html = embed_images_html()

    # Final styled HTML template
    html_template = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                margin: 40px 45px;
                color: #222;
                line-height: 1.6;
            }}

            h1, h2, h3 {{
                color: #0A84FF;
                font-weight: 700;
            }}

            h1 {{
                font-size: 28px;
                margin-bottom: 15px;
            }}

            h2 {{
                font-size: 22px;
                margin-top: 35px;
            }}

            h3 {{
                font-size: 18px;
                margin-top: 20px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }}

            th {{
                background:#0A84FF;
                color:white;
                padding:8px;
                border:1px solid #ccc;
            }}

            td {{
                border:1px solid #ccc;
                padding:8px;
            }}

            .section-title {{
                margin-top: 45px;
                font-size: 24px;
                color: #0A84FF;
                border-bottom: 2px solid #0A84FF;
                padding-bottom: 6px;
            }}

            img {{
                margin-top: 10px;
                margin-bottom: 22px;
            }}

            .footer {{
                margin-top: 40px;
                text-align: center;
                font-size: 12px;
                color: #777;
            }}

            hr {{
                margin: 35px 0;
                border: none;
                border-top: 1px solid #ccc;
            }}
        </style>
    </head>

    <body>
        <h1>Executive Summary Report</h1>
        <p><i>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i></p>

        {html_body}

        <h2 class="section-title">📊 Embedded Charts</h2>

        {charts_html}

        <div class="footer">
            <hr/>
            Customer Lifetime Value & Retention Dashboard — Auto Generated PDF Report
        </div>

    </body>
    </html>
    """

    # PDF configuration
    config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
    

    pdfkit.from_string(
        html_template,
        str(EXEC_PDF),
        configuration=config,
        options={
            "enable-local-file-access": "",
            "page-size": "A4",
            "margin-top": "10mm",
            "margin-bottom": "10mm",
        }
    )

    return EXEC_PDF