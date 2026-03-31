"""Generate clean architecture diagram for Thumalien pipeline V1.5."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('#FFFFFF')

# ---- Colors ----
C_BLUE = '#00D4FF'
C_DARK = '#1A1F2E'
C_MID = '#2A3040'
C_GREEN = '#00E676'
C_ORANGE = '#FF9100'
C_GOLD = '#FFD600'
C_RED = '#FF1744'
C_TEXT = '#FFFFFF'
C_GRAY = '#666666'


def box(x, y, w, h, label, sublabel='', color=C_DARK, text_color=C_TEXT, fontsize=10):
    """Draw a rounded box with label."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.15',
        facecolor=color, edgecolor=C_BLUE, linewidth=1.5, alpha=0.95,
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.15, label,
                ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color)
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel,
                ha='center', va='center', fontsize=fontsize - 2, color=text_color, alpha=0.7)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color)


def arrow(x1, y1, x2, y2, color=C_BLUE):
    """Draw an arrow between two points."""
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.8, connectionstyle='arc3,rad=0'),
    )


def section_label(x, y, text, color=C_GRAY):
    ax.text(x, y, text, fontsize=8, color=color, fontstyle='italic', ha='center')


# ==== TITLE ====
ax.text(8, 10.6, 'THUMALIEN — Architecture Pipeline V1.5',
        ha='center', va='center', fontsize=18, fontweight='bold', color=C_DARK,
        family='sans-serif')
ax.text(8, 10.25, 'Detection de Fake News Bilingue FR/EN',
        ha='center', va='center', fontsize=11, color=C_GRAY)

# ==== ROW 1: Data Sources ====
section_label(3, 9.8, 'SOURCES')
box(0.5, 9.0, 2.8, 0.7, 'Bluesky API', 'AT Protocol', color='#0066FF')
box(4.2, 9.0, 2.8, 0.7, 'ISOT Fake News', '44 898 articles EN', color='#0066FF')
box(7.9, 9.0, 2.8, 0.7, 'Kaggle FR', '9 494 articles FR', color='#0066FF')
box(11.6, 9.0, 2.8, 0.7, 'Emotions Dataset', '25 800 textes FR+EN', color='#0066FF')

# ==== ROW 2: Storage ====
arrow(1.9, 9.0, 4.5, 8.35)
box(2.5, 7.7, 3.5, 0.6, 'MongoDB', 'raw_posts + enriched_posts', color='#2E7D32')

# Training arrow
arrow(5.6, 9.0, 8, 8.35)
arrow(9.3, 9.0, 8, 8.35)
box(6.5, 7.7, 3.0, 0.6, 'DatasetCleaner', 'Biais Reuters supprime', color=C_MID)

arrow(12.9, 9.0, 12.9, 8.35)
box(11.3, 7.7, 3.2, 0.6, 'Emotion Training', 'MLP PyTorch', color=C_MID)

# ==== ROW 3: Feature Extraction ====
section_label(8, 7.15, 'EXTRACTION DE FEATURES')

arrow(4.25, 7.7, 2.5, 6.95)
box(1.0, 6.3, 3.0, 0.6, 'langdetect', 'FR / EN / other', color='#4A148C')

arrow(8, 7.7, 8, 6.95)
box(5.5, 6.3, 3.0, 0.6, 'TF-IDF', '30 000 n-grams', color='#4A148C')

box(9.0, 6.3, 3.0, 0.6, 'Features Ling.', '12 indicateurs', color='#4A148C')

arrow(12.9, 7.7, 12.9, 6.95)
box(11.5, 6.3, 3.0, 0.6, 'MLP Emotions', '7 probas PyTorch', color='#4A148C')

# Feature detail labels
section_label(7, 5.85, 'sublinear_tf, bigrams, trigrams')
section_label(10.5, 5.85, 'majuscules, ponctuation, sensationnalisme...')
section_label(13, 5.85, 'colere, degout, joie, neutre, peur, surprise, tristesse')

# ==== ROW 4: Concatenation ====
arrow(7, 6.3, 8, 5.45)
arrow(10.5, 6.3, 8, 5.45)
arrow(13, 6.3, 9.8, 5.45)
box(5.5, 4.8, 5.0, 0.6, 'hstack( TF-IDF + Ling. + Emotions )', '30 019 features', color='#BF360C')

# ==== ROW 5: Classifier ====
arrow(8, 4.8, 8, 4.25)
box(5.5, 3.5, 5.0, 0.7, 'LogisticRegression', 'calibree, class_weight=balanced', color=C_DARK)

# ==== ROW 6: Output ====
arrow(8, 3.5, 5.5, 2.85)
arrow(8, 3.5, 10.5, 2.85)

box(3.0, 2.2, 5.0, 0.6, 'Prediction', 'label + score credibilite + langue', color='#1B5E20')

box(8.5, 2.2, 5.0, 0.6, 'Dashboard Streamlit', 'Glassmorphism dark UI', color='#1B5E20')

# ==== ROW 7: Details ====
arrow(2.5, 6.3, 5.5, 2.85)  # langdetect -> prediction
section_label(5.5, 1.75, 'FIABLE (score > 0.5) / SUSPECT (score < 0.5)')
section_label(11, 1.75, 'Vue Globale | Analyse temps reel | Metriques')

# ==== Legend ====
legend_y = 0.7
ax.text(1, legend_y + 0.35, 'Legende :', fontsize=9, fontweight='bold', color=C_DARK)
legend_items = [
    ('#0066FF', 'Sources de donnees'),
    ('#2E7D32', 'Stockage / Sortie'),
    (C_MID, 'Preprocessing'),
    ('#4A148C', 'Feature Extraction'),
    ('#BF360C', 'Concatenation'),
    (C_DARK, 'Classification'),
]
for i, (color, label) in enumerate(legend_items):
    x = 1 + i * 2.4
    rect = FancyBboxPatch((x, legend_y - 0.15), 0.3, 0.25, boxstyle='round,pad=0.05',
                          facecolor=color, edgecolor=C_BLUE, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + 0.45, legend_y, label, fontsize=8, va='center', color=C_DARK)

plt.tight_layout()
plt.savefig('docs/architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Architecture diagram saved to docs/architecture.png')
plt.close()
