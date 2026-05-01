# Analyse de l'echec du Self-Training V5 sur donnees Bluesky

**Notebook** : `26_V5_Finetune_Bluesky.py`
**Date** : Avril 2026
**Statut** : Echec confirme — technique abandonnee

---

## 1. Contexte

Le modele V5 (TF-IDF + LogReg) est entraine sur des articles de presse (Reuters, ISOT, Kaggle) mais deploye sur des posts Bluesky courts et informels. Ce **domain shift** provoque un F1 suspect de seulement 0.087 sur le gold test set (200 posts Bluesky annotes).

**Hypothese testee** : Ajouter des posts Bluesky au dataset d'entrainement via pseudo-labeling (self-training) pour adapter le vocabulaire TF-IDF au domaine Bluesky.

## 2. Protocole experimental

1. Exporter les posts Bluesky a haute confiance depuis MongoDB :
   - 17 868 posts avec score V5 <= 0.15 (etiquetes "suspect")
   - 37 141 posts avec score V5 >= 0.85 (etiquetes "fiable")
2. Echantillonner 5 000 suspect + 5 000 fiable
3. Les ajouter au dataset d'entrainement original (~68K textes)
4. Re-entrainer V5 sur le dataset augmente (~78K textes)
5. Evaluer sur le gold test set

## 3. Resultats

| Modele                  | Accuracy | F1 macro | F1 suspect | FP  | FN |
|-------------------------|----------|----------|------------|-----|-----|
| V5 original             | 0.685    | 0.087    | 0.087      | 57  | 3   |
| V5-Bluesky (self-train) | 0.645    | 0.078    | 0.078      | 65  | 3   |

**V5-Bluesky est PIRE que V5 original** : +8 faux positifs, F1 suspect en baisse.

## 4. Pourquoi ca ne marche pas

### 4.1 Le probleme fondamental : circularite

Le self-training utilise les predictions de V5 comme labels d'entrainement. Or V5 est precisement le modele dont on veut corriger les erreurs. C'est circulaire :

```
V5 fait des erreurs sur Bluesky
    → On utilise les predictions de V5 comme labels
    → V5 apprend a reproduire ses propres erreurs
    → Les erreurs sont RENFORCEES, pas corrigees
```

Meme en ne prenant que les posts a haute confiance (score <= 0.15 ou >= 0.85), V5 est systematiquement biaise sur le domaine Bluesky. Ses posts "haute confiance" contiennent les memes biais que ses predictions normales.

### 4.2 Le biais de V5 sur Bluesky

V5 est entraine sur des articles de presse longs et formels. Son TF-IDF a appris que :
- Vocabulaire formel + long = fiable
- Vocabulaire informel + court = suspect

Sur Bluesky, TOUS les posts sont courts et informels. Donc V5 surclasse en "suspect" (57 FP sur 191 fiables = 30% de faux positifs). Ajouter plus de posts Bluesky etiquetes par V5 ne fait qu'amplifier ce biais.

### 4.3 Analogie

C'est comme demander a un etudiant qui confond systematiquement deux concepts de corriger ses propres copies d'examen, puis de reviser a partir de ses corrections. Il va renforcer sa confusion, pas la corriger.

## 5. Techniques alternatives qui POURRAIENT fonctionner

| Technique | Principe | Faisabilite |
|-----------|----------|-------------|
| **Annotation manuelle** | Annoter 500+ posts Bluesky par des humains | **En cours** (fichier `bluesky_500_annotation.xlsx`) |
| **Weak supervision multi-source** | Combiner V5 + CamemBERT + heuristiques comme fonctions de labeling (Snorkel) | Moyenne — necessite 3+ signaux independants |
| **Domain-Adversarial Training** | Entrainer un modele qui ne distingue pas presse/social media | Complexe — necessite un modele neural |
| **Few-shot avec LLM** | Utiliser GPT-4/Claude pour annoter avec des exemples | Cout API, mais tres efficace |
| **Active Learning** | Selectionner les posts les plus informatifs a annoter | Bon complement a l'annotation manuelle |

## 6. Recommandation

**L'annotation manuelle est la seule voie fiable.** Un dataset de 500 posts annotes par des humains (stratifie par score et langue) permettra de :

1. Evaluer precisement les performances reelles de V5/V7/V8 sur Bluesky
2. Identifier les types d'erreurs systematiques (FP sur humour, FP sur opinions, etc.)
3. Entrainer un modele adapte au domaine Bluesky avec des labels fiables
4. Servir de gold standard pour toutes les evaluations futures

Le fichier `data/bluesky_500_annotation.xlsx` est pret pour l'annotation.

## 7. Lecon apprise

> **Ne jamais utiliser un modele pour generer les labels d'entrainement de lui-meme.**
> Le self-training ne fonctionne que si le modele-source est deja performant sur le domaine cible,
> ce qui est exactement le probleme qu'on cherche a resoudre.
