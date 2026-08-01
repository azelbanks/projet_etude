#!/usr/bin/env python3
"""
carbon_audit_retry.py — Relance uniquement les scripts en erreur du premier audit
et fusionne les resultats dans carbon_audit_full.csv
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
    print("ERREUR : codecarbon non installe.")
    sys.exit(1)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
AUDIT_CSV = os.path.join(LOGS_DIR, 'carbon_audit_full.csv')

RETRY_SCRIPTS = [
    ("V4_Expert",    "12_Retraining_V4.py",              "TF-IDF+LogReg V4 — enrichissement dataset"),
    ("V5_Expert",    "14_Retraining_V5_Social.py",       "TF-IDF+LogReg V5 — social media oversample"),
    ("CamemBERT_V1", "13_FineTune_CamemBERT_FR.py",     "Fine-tuning CamemBERT V1 — textes courts FR"),
    ("CamemBERT_V2", "16_FineTune_CamemBERT_V2_Social.py","Fine-tuning CamemBERT V2 — social media"),
]

FIELDNAMES = ['timestamp', 'model_version', 'description',
              'duration_min', 'emissions_kg', 'emissions_g',
              'equivalent_voiture_m', 'note']

def fmt_g(kg): return f"{kg * 1000:.4f} g"
def fmt_m(kg): return f"{(kg / 0.170) * 1000:.1f} m"

def read_last_emission(csv_path):
    try:
        rows = list(csv.DictReader(open(csv_path, newline='')))
        return float(rows[-1]['emissions']) if rows else 0.0
    except Exception:
        return 0.0

print("=" * 70)
print("AUDIT CARBONE — RETRY scripts en erreur")
print("=" * 70)

retry_results = []

for i, (name, script_file, desc) in enumerate(RETRY_SCRIPTS, 1):
    script_path = os.path.join(NOTEBOOKS_DIR, script_file)
    print(f"\n[{i}/{len(RETRY_SCRIPTS)}] {name} — {desc}")

    tmp_file = f"audit_retry_{name}_emissions.csv"
    tracker = EmissionsTracker(
        project_name=f"ThumaCheck_{name}",
        output_dir=LOGS_DIR,
        output_file=tmp_file,
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
            timeout=14400,  # 4h max (CamemBERT)
        )
        kg = tracker.stop()
        duration_s = time.time() - t0

        if not kg:
            kg = read_last_emission(os.path.join(LOGS_DIR, tmp_file))

        status = "OK" if result.returncode == 0 else f"ERREUR code={result.returncode}"
        print(f"  {status} | {duration_s/60:.1f} min | {fmt_g(kg)} | ~{fmt_m(kg)} en voiture")
        retry_results.append({'name': name, 'desc': desc, 'kg': kg, 'duration_s': duration_s, 'note': status})

    except Exception as e:
        try: tracker.stop()
        except Exception: pass
        duration_s = time.time() - t0
        print(f"  ERREUR : {e}")
        retry_results.append({'name': name, 'desc': desc, 'kg': 0.0, 'duration_s': duration_s, 'note': str(e)})

# Lire l'audit existant pour le total consolidé
existing_rows = []
try:
    with open(AUDIT_CSV, newline='') as f:
        existing_rows = list(csv.DictReader(f))
except Exception:
    pass

# Supprimer les anciennes lignes ERREUR pour ces modeles + ligne TOTAL
names_to_replace = {r['name'] for r in retry_results} | {'TOTAL'}
kept_rows = [r for r in existing_rows if r['model_version'] not in names_to_replace]

# Recalculer le total
all_results = kept_rows + [
    {'model_version': r['name'], 'description': r['desc'],
     'emissions_kg': str(r['kg']), 'emissions_g': fmt_g(r['kg']),
     'equivalent_voiture_m': fmt_m(r['kg']),
     'duration_min': f"{r['duration_s']/60:.1f}",
     'timestamp': datetime.now().isoformat(timespec='seconds'),
     'note': r['note']}
    for r in retry_results
]

total_kg = sum(float(r['emissions_kg']) for r in all_results)
total_dur = sum(float(r['duration_min']) * 60 for r in all_results)

# Reecrire le fichier audit complet
with open(AUDIT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    for r in all_results:
        w.writerow(r)
    # Ligne TOTAL
    w.writerow({
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model_version': 'TOTAL',
        'description': 'Bilan carbone complet projet ThumaCheck',
        'duration_min': f"{total_dur/60:.1f}",
        'emissions_kg': f"{total_kg:.8f}",
        'emissions_g': fmt_g(total_kg),
        'equivalent_voiture_m': fmt_m(total_kg),
        'note': 'Tous modeles V3-V8 + RoBERTa EN + CamemBERT',
    })

print("\n" + "=" * 70)
print("BILAN CONSOLIDE FINAL")
print("=" * 70)
for r in all_results:
    print(f"  {r['model_version']:<25} {r['emissions_g']:>12}  {r['note'][:30]}")
print(f"\n  {'TOTAL':25} {fmt_g(total_kg):>12}  ~{fmt_m(total_kg)} en voiture")
print(f"\n  Fichier mis a jour : {AUDIT_CSV}")
print("=" * 70)
