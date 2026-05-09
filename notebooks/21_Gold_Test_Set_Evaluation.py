"""
Notebook 21 - Evaluation sur Gold Test Set (posts reels Bluesky annotes)

Etapes :
1. Charger les annotations des 2 annotateurs
2. Calculer l'accord inter-annotateur (Cohen's Kappa)
3. Resoudre les desaccords → label_final
4. Evaluer les modeles (CamemBERT V2, RoBERTa V2) sur le gold set
5. Comparer predictions IA vs annotations humaines

Auteur: Azelie Bernard - Thumalien Team
"""

# %% [markdown]
# # 21. Evaluation sur Gold Test Set
# ## Posts reels Bluesky annotes par 2 annotateurs

# %% Imports
import pandas as pd
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, classification_report,
    confusion_matrix, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# %% 1. Charger les annotations
GOLD_PATH = "../data/gold_test_set.csv"

df = pd.read_csv(GOLD_PATH)
print(f"Gold test set: {len(df)} posts")
print(f"Colonnes: {list(df.columns)}")

# %% 2. Verifier la completude des annotations
n_annot1 = df["annotateur_1"].notna().sum()
n_annot2 = df["annotateur_2"].notna().sum()
print(f"Annotateur 1: {n_annot1}/{len(df)} annotes")
print(f"Annotateur 2: {n_annot2}/{len(df)} annotes")

both = df.dropna(subset=["annotateur_1", "annotateur_2"])
print(f"Annotes par les 2: {len(both)}/{len(df)}")

# %% 3. Cohen's Kappa - Accord inter-annotateur
# Convertir en numerique : fiable=0, suspect=1
label_map = {"fiable": 0, "suspect": 1}
y1 = both["annotateur_1"].map(label_map)
y2 = both["annotateur_2"].map(label_map)

kappa = cohen_kappa_score(y1, y2)
accord_pct = (y1 == y2).mean() * 100

print(f"\n=== Accord Inter-Annotateur ===")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Accord brut: {accord_pct:.1f}%")
print()

# Interpretation du Kappa
if kappa < 0.20:
    interp = "Faible (slight)"
elif kappa < 0.40:
    interp = "Correct (fair)"
elif kappa < 0.60:
    interp = "Modere (moderate)"
elif kappa < 0.80:
    interp = "Bon (substantial)"
else:
    interp = "Excellent (almost perfect)"
print(f"Interpretation: {interp}")

# Seuil acceptable pour un gold set : Kappa >= 0.60
if kappa >= 0.60:
    print("✓ Kappa suffisant pour un gold test set fiable")
else:
    print("⚠ Kappa insuffisant - revoir les guidelines d'annotation")

# %% 4. Matrice de confusion inter-annotateurs
cm = confusion_matrix(y1, y2, labels=[0, 1])
print(f"\nMatrice de confusion (Annotateur 1 vs 2):")
print(f"               Annot.2=fiable  Annot.2=suspect")
print(f"Annot.1=fiable      {cm[0,0]:>5}         {cm[0,1]:>5}")
print(f"Annot.1=suspect     {cm[1,0]:>5}         {cm[1,1]:>5}")

# %% 5. Analyse des desaccords
desaccords = both[y1 != y2].copy()
print(f"\n=== Desaccords: {len(desaccords)} posts ({100-accord_pct:.1f}%) ===")

if len(desaccords) > 0:
    print("\nDesaccords par strate:")
    print(desaccords["strate"].value_counts().to_string())
    print("\nDesaccords par langue:")
    print(desaccords["langue"].value_counts().to_string())
    print("\nDesaccords par longueur (mots):")
    print(f"  Moyenne: {desaccords['nb_mots'].mean():.1f}")
    print(f"  vs global: {both['nb_mots'].mean():.1f}")

    # Afficher quelques exemples de desaccords
    print("\n--- Exemples de desaccords (5 premiers) ---")
    for _, row in desaccords.head(5).iterrows():
        print(f"\n  [{row['id']}] ({row['langue']}, {row['nb_mots']} mots)")
        print(f"  Texte: {row['text'][:120]}...")
        print(f"  Annot.1: {row['annotateur_1']} (conf={row['confiance_1']})")
        print(f"  Annot.2: {row['annotateur_2']} (conf={row['confiance_2']})")

# %% 6. Resolution des desaccords → label_final
# Strategie : si les deux sont d'accord, label_final = leur choix
# Si desaccord : utiliser la colonne label_final (remplie apres discussion)

df_resolved = both.copy()
df_resolved["accord"] = (y1 == y2)

# Cas d'accord : label_final = annotateur_1 (= annotateur_2)
mask_accord = df_resolved["accord"]
df_resolved.loc[mask_accord, "label_final"] = df_resolved.loc[mask_accord, "annotateur_1"]

# Cas de desaccord : verifier que label_final a ete rempli
n_desaccords_resolus = df_resolved.loc[~mask_accord, "label_final"].notna().sum()
n_desaccords_total = (~mask_accord).sum()
print(f"\nDesaccords resolus: {n_desaccords_resolus}/{n_desaccords_total}")

if n_desaccords_total > n_desaccords_resolus:
    print(f"⚠ Il reste {n_desaccords_total - n_desaccords_resolus} desaccords a resoudre")
    print("  Remplir la colonne 'label_final' pour les lignes en desaccord")

# %% 7. Evaluation des modeles sur le gold set
gold = df_resolved.dropna(subset=["label_final"]).copy()
y_true = gold["label_final"].map(label_map).values

# Predictions IA
y_pred_ia = gold["ia_prediction"].values

print(f"\n=== Evaluation IA vs Gold Set ({len(gold)} posts) ===")
print(classification_report(y_true, y_pred_ia, target_names=["fiable", "suspect"]))

f1_ia = f1_score(y_true, y_pred_ia, average="weighted")
acc_ia = accuracy_score(y_true, y_pred_ia)
print(f"F1 weighted: {f1_ia:.4f}")
print(f"Accuracy: {acc_ia:.4f}")

# %% 8. Evaluation par segment
for segment, label in [("fr", "Francais"), ("en", "Anglais")]:
    mask = gold["langue"] == segment
    if mask.sum() > 0:
        f1 = f1_score(y_true[mask], y_pred_ia[mask], average="weighted")
        print(f"\n{label}: F1={f1:.4f} (n={mask.sum()})")

for seuil, label in [(15, "ultra-court <15"), (30, "court 15-30")]:
    if seuil == 15:
        mask = gold["nb_mots"] < 15
    else:
        mask = (gold["nb_mots"] >= 15) & (gold["nb_mots"] < 30)
    if mask.sum() > 0:
        f1 = f1_score(y_true[mask], y_pred_ia[mask], average="weighted")
        print(f"{label} mots: F1={f1:.4f} (n={mask.sum()})")

mask_long = gold["nb_mots"] >= 30
if mask_long.sum() > 0:
    f1 = f1_score(y_true[mask_long], y_pred_ia[mask_long], average="weighted")
    print(f"long >=30 mots: F1={f1:.4f} (n={mask_long.sum()})")

# %% 9. Cas frontieres : ou l'IA se trompe-t-elle ?
erreurs = gold[y_true != y_pred_ia].copy()
print(f"\n=== Erreurs IA: {len(erreurs)}/{len(gold)} ({100*len(erreurs)/len(gold):.1f}%) ===")

if len(erreurs) > 0:
    print("\nErreurs par type:")
    faux_positifs = erreurs[erreurs["ia_prediction"] == 1]
    faux_negatifs = erreurs[erreurs["ia_prediction"] == 0]
    print(f"  Faux positifs (fiable classe suspect): {len(faux_positifs)}")
    print(f"  Faux negatifs (suspect classe fiable): {len(faux_negatifs)}")

    print("\nErreurs par langue:")
    print(erreurs["langue"].value_counts().to_string())
    print("\nErreurs par strate:")
    print(erreurs["strate"].value_counts().to_string())

    print("\nScore IA moyen sur les erreurs vs correct:")
    print(f"  Erreurs: {erreurs['ia_score'].mean():.4f}")
    correct = gold[y_true == y_pred_ia]
    print(f"  Correct: {correct['ia_score'].mean():.4f}")

# %% 10. Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matrice de confusion IA vs Gold
cm_ia = confusion_matrix(y_true, y_pred_ia, labels=[0, 1])
sns.heatmap(cm_ia, annot=True, fmt="d", cmap="Blues",
            xticklabels=["fiable", "suspect"],
            yticklabels=["fiable", "suspect"], ax=axes[0])
axes[0].set_xlabel("Prediction IA")
axes[0].set_ylabel("Gold (humain)")
axes[0].set_title(f"IA vs Gold Set (F1={f1_ia:.3f})")

# Distribution des scores IA par label gold
for label_val, label_name, color in [(0, "fiable", "green"), (1, "suspect", "red")]:
    mask = y_true == label_val
    axes[1].hist(gold.loc[mask, "ia_score"], bins=20, alpha=0.5,
                 label=f"Gold={label_name}", color=color)
axes[1].set_xlabel("Score credibilite IA")
axes[1].set_ylabel("Nombre de posts")
axes[1].set_title("Distribution scores IA par label humain")
axes[1].legend()

plt.tight_layout()
plt.savefig("../outputs/gold_test_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n=== Gold Test Set Evaluation terminee ===")
