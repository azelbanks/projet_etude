#!/usr/bin/env python3
"""
Script de packaging du livrable conforme a la nomenclature Sup de Vinci.

Nomenclature :
  ZIP  : PE_2526_M1BDIA_BernardAzelie_LazcanoteguiSebastien.zip
  PDF groupe : PE-2526_M1BDIA_BernardAzelie_LazcanoteguiSebastien.pdf
  PDF individuel Azelie : PE-2526_M1BDIA_BernardAzelie.pdf
  PDF individuel Sebastien : PE-2526_M1BDIA_LazcanoteguiSebastien.pdf

Usage:
    python scripts/package_livrable.py

Le script :
1. Genere les PDF depuis les MD (via generate_pdf.py)
2. Assemble le rendu groupe (rapport + tous les docs) en un seul PDF
3. Copie les rendus individuels avec la bonne nomenclature
4. Cree le ZIP final conforme
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
PDF_DIR = DOCS_DIR / "pdf"
OUTPUT_DIR = ROOT / "livrable"

# Nomenclature
CODE_PROMO = "M1BDIA"
ANNEE = "2526"
ETUDIANT_1 = "BernardAzelie"
ETUDIANT_2 = "LazcanoteguiSebastien"

ZIP_NAME = f"PE_{ANNEE}_{CODE_PROMO}_{ETUDIANT_1}_{ETUDIANT_2}"
PDF_GROUPE = f"PE-{ANNEE}_{CODE_PROMO}_{ETUDIANT_1}_{ETUDIANT_2}.pdf"
PDF_INDIV_1 = f"PE-{ANNEE}_{CODE_PROMO}_{ETUDIANT_1}.pdf"
PDF_INDIV_2 = f"PE-{ANNEE}_{CODE_PROMO}_{ETUDIANT_2}.pdf"

# PDF qui constituent le rendu groupe (dans l'ordre)
GROUPE_PDFS = [
    "00_executive_summary.pdf",
    "rapport_projet_thumalien.pdf",
    "01_cahier_des_charges_techniques.pdf",
    "02_conformite_RGPD_AI_Act.pdf",
    "03_methodologie_projet.pdf",
    "04_revue_challenge_equipe.pdf",
    "05_analyse_erreurs_qualitative.pdf",
    "06_analyse_modele_par_longueur.pdf",
    "07_evolution_modeles_comparatif.pdf",
    "08_planification_gantt.pdf",
    "09_PRA_PCA.pdf",
    "10_veille_technologique.pdf",
    "11_accessibilite_handicap.pdf",
    "12_model_card.pdf",
    "guide_utilisateur.pdf",
    "roles_et_competences_projet.pdf",
]


def step1_generate_pdfs():
    """Regenere tous les PDF depuis les MD."""
    print("[1/4] Regeneration des PDF...")
    result = subprocess.run(
        [sys.executable, str(DOCS_DIR / "generate_pdf.py")],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERREUR: {result.stderr}")
        sys.exit(1)
    print("  OK — PDF regeneres")


def step2_assemble_groupe():
    """Assemble les PDF du groupe en un seul fichier."""
    print(f"[2/4] Assemblage du rendu groupe -> {PDF_GROUPE}")
    try:
        from pypdf import PdfMerger
    except ImportError:
        try:
            from PyPDF2 import PdfMerger
        except ImportError:
            print("  pypdf/PyPDF2 non disponible — copie du rapport seul")
            shutil.copy2(PDF_DIR / "rapport_projet_thumalien.pdf",
                         OUTPUT_DIR / PDF_GROUPE)
            return

    merger = PdfMerger()
    for pdf_name in GROUPE_PDFS:
        pdf_path = PDF_DIR / pdf_name
        if pdf_path.exists():
            merger.append(str(pdf_path))
            print(f"  + {pdf_name}")
        else:
            print(f"  - {pdf_name} (manquant)")

    merger.write(str(OUTPUT_DIR / PDF_GROUPE))
    merger.close()
    print(f"  OK — {PDF_GROUPE}")


def step3_copy_individuels():
    """Copie les rendus individuels avec la nomenclature."""
    print("[3/4] Copie des rendus individuels...")

    src1 = PDF_DIR / "rendu_individuel_azelie_bernard.pdf"
    src2 = PDF_DIR / "rendu_individuel_sebastien_lazcanotegui.pdf"

    if src1.exists():
        shutil.copy2(src1, OUTPUT_DIR / PDF_INDIV_1)
        print(f"  {src1.name} -> {PDF_INDIV_1}")
    else:
        print(f"  MANQUANT: {src1.name}")

    if src2.exists():
        shutil.copy2(src2, OUTPUT_DIR / PDF_INDIV_2)
        print(f"  {src2.name} -> {PDF_INDIV_2}")
    else:
        print(f"  MANQUANT: {src2.name}")


def step4_create_zip():
    """Cree le ZIP final."""
    print(f"[4/4] Creation du ZIP -> {ZIP_NAME}.zip")
    zip_path = ROOT / f"{ZIP_NAME}"
    shutil.make_archive(str(zip_path), 'zip', str(OUTPUT_DIR))
    final_size = (ROOT / f"{ZIP_NAME}.zip").stat().st_size / (1024 * 1024)
    print(f"  OK — {ZIP_NAME}.zip ({final_size:.1f} MB)")


def main():
    print("=" * 60)
    print("PACKAGING LIVRABLE — Nomenclature Sup de Vinci")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    step1_generate_pdfs()
    step2_assemble_groupe()
    step3_copy_individuels()
    step4_create_zip()

    print(f"\n{'=' * 60}")
    print(f"Contenu du ZIP :")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")
    print(f"\nZIP pret : {ROOT / ZIP_NAME}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
