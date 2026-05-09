"""
Thumalien — Module d'IA explicable (XAI)
=========================================

Couvre les 4 niveaux d'explicabilité du pipeline V9 :

1. **Global / Modèle** : SHAP beeswarm + dependence plots sur V6
   (`shap_global.GlobalShapExplainer`)
2. **Local / Instance** :
     - Coefficients LogReg pour V5 (déjà dans `expert_detector.explain_prediction`)
     - SHAP TreeExplainer pour V6 (déjà dans `dashboard.app`)
     - Attention CamemBERT (`attention_viz.CamembertAttentionExplainer`)
     - Layer Integrated Gradients via Captum (`integrated_gradients.IGExplainer`)
3. **Méta-learner** : décomposition exacte des coefficients V7/V8
   (`meta_decomposition.MetaLearnerDecomposer`)
4. **Validation / Faithfulness** : AOPC, Comprehensiveness, Sufficiency
   (`faithfulness.FaithfulnessEvaluator`)

Toutes les méthodes produisent (a) un objet de résultat sérialisable JSON,
(b) une figure matplotlib/plotly pour rapport, (c) des métriques quantitatives.

Conformité AI Act : ces méthodes alimentent l'art. 13 (transparence) et
l'art. 14 (supervision humaine) en exposant le « pourquoi » de chaque
décision automatisée.
"""

from __future__ import annotations

__all__ = [
    "GlobalShapExplainer",
    "CamembertAttentionExplainer",
    "IGExplainer",
    "MetaLearnerDecomposer",
    "FaithfulnessEvaluator",
]


def __getattr__(name):  # lazy imports — évite de charger torch si pas utilisé
    if name == "GlobalShapExplainer":
        from .shap_global import GlobalShapExplainer
        return GlobalShapExplainer
    if name == "CamembertAttentionExplainer":
        from .attention_viz import CamembertAttentionExplainer
        return CamembertAttentionExplainer
    if name == "IGExplainer":
        from .integrated_gradients import IGExplainer
        return IGExplainer
    if name == "MetaLearnerDecomposer":
        from .meta_decomposition import MetaLearnerDecomposer
        return MetaLearnerDecomposer
    if name == "FaithfulnessEvaluator":
        from .faithfulness import FaithfulnessEvaluator
        return FaithfulnessEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
