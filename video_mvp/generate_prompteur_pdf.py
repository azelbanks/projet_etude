#!/usr/bin/env python3
"""Genere le PDF prompteur depuis PROMPTEUR_v3.md — format lisible pour enregistrement."""

from fpdf import FPDF
from pathlib import Path
import re

SCRIPT_DIR = Path(__file__).parent
INPUT = SCRIPT_DIR / "PROMPTEUR_v3.md"
OUTPUT = SCRIPT_DIR / "PROMPTEUR_v3.pdf"


ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ARIAL_ITALIC = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"


class PrompterPDF(FPDF):
    def setup_fonts(self):
        self.add_font("arial", "", ARIAL, uni=True)
        self.add_font("arial", "B", ARIAL_BOLD, uni=True)
        self.add_font("arial", "I", ARIAL_ITALIC, uni=True)

    def header(self):
        self.set_font("arial", "I", 9)
        self.set_text_color(140, 140, 140)
        self.cell(0, 6, "PROMPTEUR -- ThumaCheck -- Niamato Consulting", align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("arial", "I", 9)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def build_pdf():
    md_text = INPUT.read_text(encoding="utf-8")

    pdf = PrompterPDF(orientation="P", unit="mm", format="A4")
    pdf.setup_fonts()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)

    pdf.add_page()

    # Title page
    pdf.set_font("arial", "B", 28)
    pdf.set_text_color(20, 60, 140)
    pdf.ln(40)
    pdf.cell(0, 15, "PROMPTEUR", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("arial", "", 18)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 12, "ThumaCheck -- Niamato Consulting pour Thumalien", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("arial", "I", 14)
    pdf.cell(0, 10, "Niamato Consulting", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("arial", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Version prompteur — texte aere pour lecture pendant enregistrement", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Police grande, retours a la ligne frequents, indications techniques en gris", align="C", new_x="LMARGIN", new_y="NEXT")

    # Parse and render
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip title and first metadata lines
        if i < 5 and (line.startswith("# PROMPTEUR") or line.startswith("## Version")):
            i += 1
            continue

        # Triple separator = new section page
        if line == "---" and i + 1 < len(lines) and lines[i + 1].strip() == "---":
            # Skip all consecutive ---
            while i < len(lines) and lines[i].strip() == "---":
                i += 1
            continue

        # Single --- = visual pause
        if line == "---":
            pdf.ln(4)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.get_x() + 30, pdf.get_y(), pdf.get_x() + 140, pdf.get_y())
            pdf.ln(4)
            i += 1
            continue

        # Section header (# ===...)
        if line.startswith("# =="):
            i += 1
            continue

        # Speaker section header
        if re.match(r"^# (AZELIE|SEBASTIEN)", line):
            pdf.add_page()
            speaker = line.replace("# ", "")
            # Color: blue for Azelie, green for Sebastien
            if "AZELIE" in speaker:
                pdf.set_fill_color(20, 60, 140)
            else:
                pdf.set_fill_color(30, 120, 60)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("arial", "B", 16)
            pdf.cell(0, 14, f"  {speaker}", fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            pdf.set_text_color(0, 0, 0)
            i += 1
            continue

        # Technical instruction (# [...])
        if line.startswith("# [") or line.startswith("# FIN"):
            pdf.ln(3)
            pdf.set_font("arial", "B", 11)
            pdf.set_text_color(100, 100, 100)
            instruction = line.replace("# ", "")
            pdf.multi_cell(0, 7, instruction)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
            i += 1
            continue

        # Empty line
        if not line:
            pdf.ln(3)
            i += 1
            continue

        # Regular text (voix off)
        pdf.set_font("arial", "", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 9, line)
        pdf.ln(1)
        i += 1

    pdf.output(str(OUTPUT))
    print(f"PDF genere : {OUTPUT}")
    print(f"Pages : {pdf.page_no()}")


if __name__ == "__main__":
    build_pdf()
