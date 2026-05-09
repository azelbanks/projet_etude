#!/usr/bin/env python3
"""
Script pour capturer les screenshots des 5 pages du dashboard Streamlit.

Pre-requis :
  pip install selenium webdriver-manager

Usage :
  1. Lancer le dashboard : streamlit run dashboard/app.py
  2. Executer : python scripts/capture_dashboard.py

Les captures sont sauvegardees dans docs/figures/screenshots/
"""

import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures" / "screenshots"

PAGES = {
    "page_dashboard": "http://localhost:8501/",
    "page_analyse_ia": "http://localhost:8501/?page=analyse",
    "page_explorateur": "http://localhost:8501/?page=explorateur",
    "page_performance": "http://localhost:8501/?page=performance",
    "page_about": "http://localhost:8501/?page=about",
}


def capture_with_selenium():
    """Capture screenshots using Selenium."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except ImportError:
        print("ERREUR: selenium non installe.")
        print("  pip install selenium webdriver-manager")
        return False

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"ERREUR: Impossible de lancer Chrome: {e}")
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in PAGES.items():
        try:
            driver.get(url)
            time.sleep(3)  # Wait for Streamlit to render
            filepath = OUTPUT_DIR / f"{name}.png"
            driver.save_screenshot(str(filepath))
            print(f"  [OK] {name}.png ({filepath.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"  [ERREUR] {name}: {e}")

    driver.quit()
    return True


def main():
    print("=" * 60)
    print("CAPTURE SCREENSHOTS — Dashboard Thumalien")
    print("=" * 60)
    print(f"Dossier de sortie: {OUTPUT_DIR}\n")

    # Check if dashboard is running
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:8501/", timeout=3)
    except Exception:
        print("ATTENTION: Le dashboard ne semble pas en cours d'execution.")
        print("  Lancez d'abord: streamlit run dashboard/app.py")
        print("  Puis relancez ce script.\n")
        print("Creation de placeholders a la place...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name in PAGES:
            placeholder = OUTPUT_DIR / f"{name}.png"
            if not placeholder.exists():
                # Create a minimal placeholder text file
                placeholder.write_text(f"[Placeholder — lancer le dashboard puis re-executer capture_dashboard.py]")
                print(f"  [PLACEHOLDER] {name}.png")
        return

    success = capture_with_selenium()
    if not success:
        print("\nCapture Selenium echouee.")
        print("Alternative manuelle :")
        print("  1. Ouvrir http://localhost:8501 dans le navigateur")
        print("  2. Capturer chaque page avec Cmd+Shift+4 (Mac)")
        print("  3. Sauvegarder dans docs/figures/screenshots/")
        for name in PAGES:
            print(f"     - {name}.png")

    print("=" * 60)


if __name__ == "__main__":
    main()
