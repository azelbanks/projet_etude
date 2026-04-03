#!/usr/bin/env python3
"""
Generateur de PDF pour la documentation projet Thumalien.
Convertit les fichiers Markdown en PDF via fpdf2.

Usage:
    python3 docs/generate_pdf.py
"""

import re
import sys
from pathlib import Path
from fpdf import FPDF


DOCS_DIR = Path(__file__).parent
OUTPUT_DIR = DOCS_DIR / "pdf"

DOCUMENTS = [
    "01_cahier_des_charges_techniques.md",
    "02_conformite_RGPD_AI_Act.md",
    "03_methodologie_projet.md",
    "04_revue_challenge_equipe.md",
    "roles_et_competences_projet.md",
    "rapport_projet_thumalien.md",
    "guide_utilisateur.md",
]

# Colors
NAVY = (10, 36, 99)
BLUE = (30, 82, 136)
LIGHT_BLUE = (42, 122, 181)
BLACK = (26, 26, 26)
GRAY = (100, 100, 100)
LIGHT_GRAY = (240, 243, 247)
WHITE = (255, 255, 255)
TABLE_HEADER_BG = (10, 36, 99)
TABLE_ALT_BG = (245, 247, 250)
CODE_BG = (240, 243, 247)


class ThumalienPDF(FPDF):
    """PDF generator with professional styling for Thumalien docs."""

    def __init__(self, title=""):
        super().__init__()
        self.doc_title = title
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 5, f"Thumalien - {self.doc_title}", align="L")
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, text, level=1):
        text = self._strip_markdown_inline(text)
        if level == 1:
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(*NAVY)
            self.ln(5)
            self.multi_cell(0, 9, text)
            # Underline
            y = self.get_y()
            self.set_draw_color(*NAVY)
            self.set_line_width(0.8)
            self.line(10, y, 200, y)
            self.set_line_width(0.2)
            self.ln(6)
        elif level == 2:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*BLUE)
            self.ln(4)
            self.multi_cell(0, 8, text)
            y = self.get_y()
            self.set_draw_color(200, 200, 200)
            self.line(10, y, 200, y)
            self.ln(4)
        elif level == 3:
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(*LIGHT_BLUE)
            self.ln(3)
            self.multi_cell(0, 7, text)
            self.ln(2)
        else:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*LIGHT_BLUE)
            self.ln(2)
            self.multi_cell(0, 6, text)
            self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*BLACK)
        # Handle bold markers
        text = self._strip_markdown_inline(text)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def code_block(self, text):
        self.set_fill_color(*CODE_BG)
        self.set_font("Courier", "", 8)
        self.set_text_color(30, 30, 30)
        text = self._strip_markdown_inline(text)
        lines = text.split("\n")
        for line in lines:
            # Truncate very long lines
            if len(line) > 100:
                line = line[:97] + "..."
            self.cell(0, 4.5, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def render_table(self, headers, rows):
        """Render a table with headers and rows."""
        if not headers:
            return

        n_cols = len(headers)
        available_width = 190
        col_width = available_width / n_cols
        # Adjust column widths: first column gets more space if few columns
        if n_cols <= 3:
            col_widths = [available_width / n_cols] * n_cols
        else:
            col_widths = [available_width / n_cols] * n_cols

        row_height = 6

        # Header
        self.set_fill_color(*TABLE_HEADER_BG)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8)
        for i, h in enumerate(headers):
            w = col_widths[i]
            h_text = self._strip_markdown_inline(h.strip())
            if len(h_text) > int(w / 2):
                h_text = h_text[:int(w / 2) - 1] + "."
            self.cell(w, row_height, h_text, border=1, fill=True, align="C")
        self.ln(row_height)

        # Rows
        self.set_font("Helvetica", "", 7.5)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 1:
                self.set_fill_color(*TABLE_ALT_BG)
            else:
                self.set_fill_color(*WHITE)
            self.set_text_color(*BLACK)

            # Calculate max height needed
            max_lines = 1
            cell_texts = []
            for i, cell in enumerate(row):
                w = col_widths[i] if i < len(col_widths) else col_widths[-1]
                text = self._strip_markdown_inline(cell.strip())
                # Estimate lines needed
                char_per_line = max(int(w / 1.8), 10)
                n_lines = max(1, (len(text) + char_per_line - 1) // char_per_line)
                max_lines = max(max_lines, n_lines)
                cell_texts.append(text)

            cell_h = row_height * min(max_lines, 4)

            for i, text in enumerate(cell_texts):
                w = col_widths[i] if i < len(col_widths) else col_widths[-1]
                x = self.get_x()
                y = self.get_y()
                self.rect(x, y, w, cell_h, style="DF")
                self.set_xy(x + 1, y + 1)
                self.multi_cell(w - 2, row_height - 1.5, text[:200], border=0)
                self.set_xy(x + w, y)

            self.ln(cell_h)

            # Page break check
            if self.get_y() > 265:
                self.add_page()

        self.ln(3)

    def horizontal_rule(self):
        self.ln(3)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_line_width(0.2)
        self.ln(5)

    def bullet_item(self, text, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*BLACK)
        text = self._strip_markdown_inline(text)
        x = 15 + indent * 5
        self.set_x(x)
        self.cell(3, 5.5, "-")  # bullet char
        self.multi_cell(190 - x, 5.5, " " + text)
        self.ln(1)

    @staticmethod
    def _strip_markdown_inline(text):
        """Remove markdown bold/italic/code markers and sanitize unicode."""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        # Sanitize unicode chars not in latin-1
        text = text.replace("\u2014", "-")   # em dash
        text = text.replace("\u2013", "-")   # en dash
        text = text.replace("\u2018", "'")   # left single quote
        text = text.replace("\u2019", "'")   # right single quote
        text = text.replace("\u201c", '"')   # left double quote
        text = text.replace("\u201d", '"')   # right double quote
        text = text.replace("\u2026", "...")  # ellipsis
        text = text.replace("\u2022", "-")   # bullet
        text = text.replace("\u00ab", '"')   # guillemet left
        text = text.replace("\u00bb", '"')   # guillemet right
        text = text.replace("\u2192", "->")  # arrow right
        text = text.replace("\u2190", "<-")  # arrow left
        text = text.replace("\u2265", ">=")  # >=
        text = text.replace("\u2264", "<=")  # <=
        # Remove emojis and other non-latin1 chars
        text = text.encode("latin-1", errors="replace").decode("latin-1")
        return text


def parse_markdown(md_text):
    """Parse markdown into structured blocks."""
    lines = md_text.split("\n")
    blocks = []
    i = 0
    in_code = False
    code_lines = []
    in_table = False
    table_headers = []
    table_rows = []

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith("```"):
            if in_code:
                blocks.append(("code", "\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                # Flush table if any
                if in_table:
                    blocks.append(("table", table_headers, table_rows))
                    in_table = False
                    table_headers = []
                    table_rows = []
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Table detection
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not in_table:
                # Check if next line is separator
                if i + 1 < len(lines) and re.match(r'\s*\|[\s\-:|]+\|', lines[i + 1]):
                    in_table = True
                    table_headers = cells
                    i += 2  # skip header + separator
                    continue
            else:
                table_rows.append(cells)
                i += 1
                continue
        else:
            if in_table:
                blocks.append(("table", table_headers, table_rows))
                in_table = False
                table_headers = []
                table_rows = []

        # Headers
        if line.startswith("####"):
            blocks.append(("h4", line.lstrip("#").strip()))
        elif line.startswith("###"):
            blocks.append(("h3", line.lstrip("#").strip()))
        elif line.startswith("##"):
            blocks.append(("h2", line.lstrip("#").strip()))
        elif line.startswith("#"):
            blocks.append(("h1", line.lstrip("#").strip()))
        elif line.strip().startswith("---") and len(line.strip()) >= 3 and all(c in "-" for c in line.strip()):
            blocks.append(("hr",))
        elif line.strip().startswith("- ") or line.strip().startswith("* "):
            indent = (len(line) - len(line.lstrip())) // 2
            text = re.sub(r'^[\s]*[-*]\s+', '', line)
            blocks.append(("bullet", text, indent))
        elif re.match(r'^\s*\d+\.\s+', line):
            text = re.sub(r'^\s*\d+\.\s+', '', line)
            blocks.append(("bullet", text, 0))
        elif line.strip():
            blocks.append(("text", line.strip()))
        # Empty lines are skipped

        i += 1

    # Flush remaining
    if in_code and code_lines:
        blocks.append(("code", "\n".join(code_lines)))
    if in_table:
        blocks.append(("table", table_headers, table_rows))

    return blocks


def convert_md_to_pdf(md_path: Path, pdf_path: Path):
    """Convert a markdown file to PDF."""
    md_text = md_path.read_text(encoding="utf-8")
    title = md_path.stem.replace("_", " ").title()

    pdf = ThumalienPDF(title=title)
    pdf.alias_nb_pages()
    pdf.add_page()

    blocks = parse_markdown(md_text)

    for block in blocks:
        btype = block[0]

        # Check if we need a new page
        if pdf.get_y() > 270 and btype not in ("hr",):
            pdf.add_page()

        if btype == "h1":
            pdf.chapter_title(block[1], 1)
        elif btype == "h2":
            pdf.chapter_title(block[1], 2)
        elif btype == "h3":
            pdf.chapter_title(block[1], 3)
        elif btype == "h4":
            pdf.chapter_title(block[1], 4)
        elif btype == "text":
            pdf.body_text(block[1])
        elif btype == "code":
            pdf.code_block(block[1])
        elif btype == "table":
            pdf.render_table(block[1], block[2])
        elif btype == "hr":
            pdf.horizontal_rule()
        elif btype == "bullet":
            pdf.bullet_item(block[1], block[2])

    pdf.output(str(pdf_path))


def main():
    print("=" * 60)
    print("THUMALIEN - Generateur de documentation PDF")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nDossier de sortie: {OUTPUT_DIR}\n")

    existing = []
    for doc in DOCUMENTS:
        path = DOCS_DIR / doc
        if path.exists():
            existing.append(path)
            print(f"  [OK] {doc}")
        else:
            print(f"  [--] {doc} (non trouve)")

    if not existing:
        print("\nAucun document trouve.")
        sys.exit(1)

    print(f"\n{len(existing)} documents a convertir.\n")

    success = 0
    for md_path in existing:
        pdf_path = OUTPUT_DIR / (md_path.stem + ".pdf")
        try:
            convert_md_to_pdf(md_path, pdf_path)
            size_kb = pdf_path.stat().st_size / 1024
            print(f"  [PDF] {md_path.name} -> {pdf_path.name} ({size_kb:.0f} KB)")
            success += 1
        except Exception as e:
            print(f"  [ECHEC] {md_path.name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Resultat: {success}/{len(existing)} documents convertis")
    if success > 0:
        print(f"PDF disponibles dans: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
