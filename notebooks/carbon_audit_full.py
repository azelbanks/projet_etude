#!/usr/bin/env python3
"""
carbon_audit_full.py — Audit carbone complet ThumaCheck
========================================================

Objectif :
    Reentrainer chaque version du modele ThumaCheck (V3 → V8 + Stage1 +
    CamemBERT) avec CodeCarbon active, et consolider les emissions dans
    un seul fichier logs/carbon_audit_full.csv.

    RoBERTa EN (V18/V19) : deja traque — on importe emissions.csv existant.

Execution :
    cd /chemin/vers/projet_etude
    python3 notebooks/carbon_audit_full.py

Duree estimee :
    Modeles sklearn (V3-V8, Stage1) : ~15-30 min total
    CamemBERT V1 + V2              : ~2-6 heures (GPU/MPS dependant)
    Total                          : variable selon la machine

Auteur : Thumalien Team
"""

import sys
import os
import subprocess
import csv
import time
from datetime import datetime

try:
    from codecarbon import EmissionsTracker
except ImportError:
    print("ERREUR : codecarbon non installe. Lancer : pip install codecarbon")
    sys.exit(1)

# ============================================================
#  CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
AUDIT_CSV = os.path.join(LOGS_DIR, 'carbon_audit_full.csv')
ROBERTA_EXISTING = os.path.join(PROJECT_ROOT, 'emissions.csv')

os.makedirs(LOGS_DIR, exist_ok=True)

# Ordre d'execution des scripts (nom, fichier, description)
TRAINING_SCRIPTS = [
    ("V3_Expert",       "11_Retraining_V3.py",              "TF-IDF+LogReg V3 — features linguistiques corrigees"),
    ("V4_Expert",       "12_Retraining_V4.py",              "TF-IDF+LogReg V4 — enrichissement dataset"),
    ("V5_Expert",       "14_Retraining_V5_Social.py",       "TF-IDF+LogReg V5 — social media oversample"),
    ("V5_Bluesky",      "26_V5_Finetune_Bluesky.py",        "V5 fine-tune Bluesky posts reels"),
    ("V6_Style",        "23_Style_Only_V6.py",              "Modele style-only V6 — 30 features stylistiques"),
    ("CamemBERT_V1",    "13_FineTune_CamemBERT_FR.py",      "Fine-tuning CamemBERT V1 — textes courts FR"),
    ("CamemBERT_V2",    "16_FineTune_CamemBERT_V2_Social.py","Fine-tuning CamemBERT V2 — social media"),
    ("V7_Hybrid",       "24_Hybrid_Ensemble_V7_SHAP.py",    "Meta-learner V7 — V5+V6+SHAP"),
    ("V8_Hybrid",       "25_V8_Hybrid_Extended_CamemBERT.py","Meta-learner V8 — V5+V6+CamemBERT"),
    ("Stage1_Cascade",  "27_Pipeline_2_Etapes.py",          "Pipeline cascade fait/opinion Stage1"),
]

# ============================================================
#  UTILITAIRES
# ============================================================
def fmt_g(kg):
    return f"{kg * 1000:.4f} g"

def fmt_m(kg):
    m = (kg / 0.18) * 1000
    return f"{m:.1f} m"

def read_last_emission(csv_path):
    """Lire la derniere ligne d'un fichier emissions CodeCarbon."""
    try:
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            return float(rows[-1]['emissions'])
    except Exception:
        pass
    return 0.0

def write_audit_row(writer, name, desc, kg, duration_s, note=""):
    writer.writerow({
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model_version': name,
        'description': desc,
        'duration_min': f"{duration_s / 60:.1f}",
        'emissions_kg': f"{kg:.8f}",
        'emissions_g': f"{kg * 1000:.4f}",
        'equivalent_voiture_m': f"{(kg / 0.18) * 1000:.1f}",
        'note': note,
    })

# ============================================================
#  INITIALISATION DU FICHIER AUDIT
# ============================================================
FIELDNAMES = ['timestamp', 'model_version', 'description',
              'duration_min', 'emissions_kg', 'emissions_g',
              'equivalent_voiture_m', 'note']

with open(AUDIT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()

print("=" * 70)
print("AUDIT CARBONE COMPLET — ThumaCheck")
print(f"Resultats : {AUDIT_CSV}")
print("=" * 70)

total_kg = 0.0
results = []

# ============================================================
#  1. IMPORTER ROBERTA EN (deja tracke)
# ============================================================
print("\n[0/11] RoBERTa EN V1+V2 — import depuis emissions.csv existant...")
roberta_kg = 0.0
if os.path.exists(ROBERTA_EXISTING):
    try:
        with open(ROBERTA_EXISTING, newline='') as f:
            roberta_rows = list(csv.DictReader(f))
        for row in roberta_rows:
            roberta_kg += float(row['emissions'])
            results.append({
                'name': row.get('project_name', 'RoBERTa_EN'),
                'desc': 'Fine-tuning RoBERTa EN (deja tracke)',
                'kg': float(row['emissions']),
                'duration_s': float(row['duration']),
                'note': 'Importe depuis emissions.csv original',
            })
        print(f"  RoBERTa EN total : {fmt_g(roberta_kg)} CO2 ({len(roberta_rows)} runs)")
    except Exception as e:
        print(f"  Impossible de lire emissions.csv : {e}")
else:
    print("  emissions.csv non trouve — RoBERTa EN non inclus")

total_kg += roberta_kg

# ============================================================
#  2. REENTRAINER CHAQUE MODELE AVEC CODECARBON
# ============================================================
for i, (name, script_file, desc) in enumerate(TRAINING_SCRIPTS, start=1):
    script_path = os.path.join(NOTEBOOKS_DIR, script_file)

    if not os.path.exists(script_path):
        print(f"\n[{i}/{len(TRAINING_SCRIPTS)}] SKIP {name} — script non trouve : {script_path}")
        continue

    print(f"\n[{i}/{len(TRAINING_SCRIPTS)}] {name} — {desc}")
    print(f"  Script : {script_file}")

    # Fichier emissions temporaire pour ce run
    tmp_emissions_dir = LOGS_DIR
    tmp_emissions_file = f"audit_{name}_emissions.csv"

    tracker = EmissionsTracker(
        project_name=f"ThumaCheck_{name}",
        output_dir=tmp_emissions_dir,
        output_file=tmp_emissions_file,
        log_level='warning',
        save_to_file=True,
        tracking_mode='machine',
    )

    t0 = time.time()
    try:
        tracker.start()
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_ROOT,
            capture_output=False,  # laisser le stdout passer pour voir la progression
            timeout=7200,  # 2h max par script
        )
        kg = tracker.stop()
        duration_s = time.time() - t0

        if kg is None or kg == 0.0:
            # Fallback : lire depuis le fichier CSV si tracker.stop() retourne 0
            tmp_path = os.path.join(tmp_emissions_dir, tmp_emissions_file)
            kg = read_last_emission(tmp_path)

        total_kg += kg
        status = "OK" if result.returncode == 0 else f"ERREUR (code={result.returncode})"
        print(f"  {status} | {duration_s/60:.1f} min | {fmt_g(kg)} CO2 | ~{fmt_m(kg)} en voiture")

        results.append({
            'name': name,
            'desc': desc,
            'kg': kg,
            'duration_s': duration_s,
            'note': status,
        })

    except subprocess.TimeoutExpired:
        try:
            tracker.stop()
        except Exception:
            pass
        duration_s = time.time() - t0
        print(f"  TIMEOUT apres {duration_s/60:.0f} min — script trop long, non inclus dans le total")
        results.append({
            'name': name,
            'desc': desc,
            'kg': 0.0,
            'duration_s': duration_s,
            'note': 'TIMEOUT — non mesure',
        })

    except Exception as e:
        try:
            tracker.stop()
        except Exception:
            pass
        duration_s = time.time() - t0
        print(f"  ERREUR : {e}")
        results.append({
            'name': name,
            'desc': desc,
            'kg': 0.0,
            'duration_s': duration_s,
            'note': f'ERREUR: {e}',
        })

# ============================================================
#  3. ECRIRE LE RAPPORT FINAL
# ============================================================
with open(AUDIT_CSV, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    for r in results:
        write_audit_row(w, r['name'], r['desc'], r['kg'], r['duration_s'], r.get('note', ''))

    # Ligne totale
    total_duration = sum(r['duration_s'] for r in results)
    write_audit_row(w, 'TOTAL', 'Bilan carbone complet projet ThumaCheck',
                    total_kg, total_duration, 'Tous modeles V3-V8 + RoBERTa EN')

# ============================================================
#  4. AFFICHER LE RESUME
# ============================================================
print("\n" + "=" * 70)
print("BILAN CARBONE FINAL — ThumaCheck")
print("=" * 70)
print(f"\n  {'Modele':<25} {'Emissions':>12} {'Voiture':>12}  Note")
print(f"  {'-'*65}")
for r in results:
    note = r.get('note', '')[:25]
    print(f"  {r['name']:<25} {fmt_g(r['kg']):>12} {fmt_m(r['kg']):>12}  {note}")

print(f"\n  {'─'*65}")
print(f"  {'TOTAL':25} {fmt_g(total_kg):>12} {fmt_m(total_kg):>12}")
print(f"\n  Equivalent voiture : {fmt_m(total_kg)}")
km_total = total_kg / 0.18
print(f"  ({km_total*1000:.1f} m = {km_total:.4f} km)")
print(f"\n  Rapport complet : {AUDIT_CSV}")
print("=" * 70)
