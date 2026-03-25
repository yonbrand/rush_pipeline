"""Convert a Markdown file to a styled PDF.

Usage:
    python make_pdf.py                                    # converts all known docs
    python make_pdf.py extraction/PIPELINE_OVERVIEW.md    # convert a specific file
"""
import sys
import os
import markdown
from xhtml2pdf import pisa

# Default documents to convert when no argument is given
DEFAULT_DOCS = [
    os.path.join("extraction", "PIPELINE_OVERVIEW.md"),
    os.path.join("modeling", "MODELING_OVERVIEW.md"),
]

CSS = """
@page {
    size: A4;
    margin: 2.2cm 2.4cm 2.2cm 2.4cm;
}

body {
    font-family: Arial, sans-serif;
    font-size: 10pt;
    color: #1a1a1a;
    line-height: 1.55;
}

h1 {
    font-size: 20pt;
    color: #003087;
    border-bottom: 3px solid #003087;
    padding-bottom: 6pt;
    margin-top: 0;
    margin-bottom: 6pt;
}

h2 {
    font-size: 14pt;
    color: #003087;
    border-bottom: 1.5px solid #c8d8f0;
    padding-bottom: 3pt;
    margin-top: 18pt;
    margin-bottom: 6pt;
}

h3 {
    font-size: 11pt;
    color: #005a9c;
    margin-top: 12pt;
    margin-bottom: 4pt;
}

p {
    margin: 4pt 0 8pt 0;
}

em {
    color: #555;
    font-style: italic;
}

strong {
    color: #111;
}

/* Code / monospace */
code {
    font-family: "Courier New", monospace;
    font-size: 8.5pt;
    background-color: #f4f6fb;
    padding: 1pt 3pt;
    border-radius: 2pt;
    color: #c0392b;
}

pre {
    background-color: #f4f6fb;
    border-left: 4px solid #003087;
    padding: 8pt 10pt;
    font-family: "Courier New", monospace;
    font-size: 8.5pt;
    color: #003087;
    margin: 8pt 0;
    white-space: pre-wrap;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0 12pt 0;
    font-size: 9pt;
}

thead {
    background-color: #003087;
    color: #ffffff;
}

thead th {
    padding: 5pt 7pt;
    text-align: left;
    font-weight: bold;
}

tbody tr:nth-child(even) {
    background-color: #eef3fb;
}

tbody tr:nth-child(odd) {
    background-color: #ffffff;
}

tbody td {
    padding: 4pt 7pt;
    border-bottom: 0.5pt solid #d0d8e8;
    vertical-align: top;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #005a9c;
    margin: 8pt 0;
    padding: 4pt 10pt;
    background-color: #f0f5ff;
    color: #333;
    font-size: 9.5pt;
}

/* Horizontal rule */
hr {
    border: none;
    border-top: 1.5pt solid #c8d8f0;
    margin: 14pt 0;
}

/* Lists */
ul, ol {
    margin: 4pt 0 8pt 16pt;
    padding: 0;
}

li {
    margin-bottom: 2pt;
}
"""


def convert(md_path):
    pdf_path = md_path.rsplit(".", 1)[0] + ".pdf"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"]
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
{CSS}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    with open(pdf_path, "wb") as out_f:
        result = pisa.CreatePDF(full_html, dest=out_f)

    if result.err:
        print(f"ERROR: {md_path} -> PDF failed with {result.err} errors.")
    else:
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"PDF created: {pdf_path}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = [d for d in DEFAULT_DOCS if os.path.exists(d)]

    if not targets:
        print("No Markdown files found to convert.")
    for t in targets:
        convert(t)
