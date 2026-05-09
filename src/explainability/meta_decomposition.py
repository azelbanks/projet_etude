"""
Décomposition exacte du méta-learner V7 / V8
=============================================

Le méta-learner est un `LogisticRegression` à 4 ou 7 features (selon V7
ou V8). Sa décision est :

    z = β_0 + Σ_i β_i × x_i
    P(suspect) = σ(z)

Cette décomposition expose `β_i × x_i` pour chaque feature, c'est la
forme la plus rigoureuse d'explicabilité pour un modèle linéaire — pas
une approximation, c'est la formule fermée exacte.

Sortie typique :
    contribution_v5_fiable    = +0.42    (pousse vers SUSPECT car score V5 fiable bas)
    contribution_v6_suspect   = +0.31    (pousse vers SUSPECT car style suspect)
    contribution_disagreement = -0.05
    contribution_interaction  = +0.02
    intercept                 = -0.61
    z = +0.09  →  P(suspect) = σ(0.09) = 0.522

C'est ce qu'on veut afficher dans le dashboard V9 pour expliquer
**l'ensemble** (et pas seulement V6 via SHAP).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class MetaDecomposition:
    """Décomposition exacte d'une prédiction du méta-learner."""

    feature_names: List[str]
    feature_values: List[float]
    coefficients: List[float]
    contributions: List[float]  # β_i * x_i
    intercept: float
    logit: float  # z = intercept + Σ contributions
    proba_suspect: float  # σ(z)
    label: str  # "SUSPECT" ou "FIABLE"
    threshold: float = 0.5
    figures: dict = field(default_factory=dict)

    def top_drivers(self, k: int = 3) -> List[dict]:
        """Top-k features par valeur absolue de contribution."""
        idx = np.argsort(np.abs(self.contributions))[::-1][:k]
        return [
            {
                "feature": self.feature_names[i],
                "value": self.feature_values[i],
                "coefficient": self.coefficients[i],
                "contribution": self.contributions[i],
                "direction": "SUSPECT" if self.contributions[i] > 0 else "FIABLE",
            }
            for i in idx
        ]

    def to_json(self) -> str:
        return json.dumps(
            {
                "feature_names": self.feature_names,
                "feature_values": self.feature_values,
                "coefficients": self.coefficients,
                "contributions": self.contributions,
                "intercept": self.intercept,
                "logit": self.logit,
                "proba_suspect": self.proba_suspect,
                "label": self.label,
                "threshold": self.threshold,
                "top_drivers": self.top_drivers(5),
                "figures": self.figures,
            },
            ensure_ascii=False,
            indent=2,
        )


class MetaLearnerDecomposer:
    """
    Décompose une prédiction du méta-learner V7 ou V8.

    Parameters
    ----------
    meta_data : dict
        Contenu de `joblib.load('models/model_hybrid_v8.joblib')`.
        Doit exposer `meta_model` (LogisticRegression), `feature_names` ou
        `uses_camembert`. Le format de notre projet est inféré.
    threshold : float
        Seuil de classification (défaut 0.5).

    Examples
    --------
    >>> import joblib
    >>> v8 = joblib.load('models/model_hybrid_v8.joblib')
    >>> decomposer = MetaLearnerDecomposer(v8)
    >>> X_meta = np.array([[0.82, 0.41, 0.57, 0.21, 0.05, 0.34, 0.57]])
    >>> result = decomposer.decompose(X_meta[0])
    >>> result.proba_suspect
    0.34
    >>> result.top_drivers(3)
    [{'feature': 'score_v5_fiable', 'contribution': -0.51, ...}, ...]
    """

    DEFAULT_FEATURE_NAMES_V7 = [
        "score_v5_fiable",
        "score_v6_suspect",
        "disagreement",
        "interaction",
    ]
    DEFAULT_FEATURE_NAMES_V8 = [
        "score_v5_fiable",
        "score_v6_suspect",
        "score_camembert_fiable",
        "disagreement_v5_v6",
        "disagreement_v5_cam",
        "interaction_v5_v6",
        "min_fiable",
    ]

    def __init__(self, meta_data: dict, threshold: float = 0.5):
        self.meta_data = meta_data
        self.threshold = threshold

        self.meta_model = (
            meta_data.get("meta_model")
            or meta_data.get("model")
            or meta_data
        )
        if not hasattr(self.meta_model, "coef_"):
            raise ValueError(
                "Le méta-learner ne semble pas linéaire (pas de `coef_`). "
                "Cette décomposition n'est valide que pour LogReg / linéaire."
            )

        # Inférence du nombre de features
        n = self.meta_model.coef_.shape[1]
        self.feature_names = (
            meta_data.get("feature_names")
            or (self.DEFAULT_FEATURE_NAMES_V8 if n == 7
                else self.DEFAULT_FEATURE_NAMES_V7 if n == 4
                else [f"f_{i}" for i in range(n)])
        )

        self.coef = self.meta_model.coef_[0].copy()
        self.intercept = float(self.meta_model.intercept_[0])

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def decompose(self, x: Sequence[float]) -> MetaDecomposition:
        """
        Décompose une prédiction. `x` doit avoir la même dimension que
        `self.feature_names`.
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != len(self.feature_names):
            raise ValueError(
                f"x a {x.shape[0]} features, attendu {len(self.feature_names)}"
            )

        contribs = self.coef * x  # β_i * x_i
        z = float(self.intercept + contribs.sum())
        p_suspect = float(self._sigmoid(z))
        label = "SUSPECT" if p_suspect >= self.threshold else "FIABLE"

        return MetaDecomposition(
            feature_names=list(self.feature_names),
            feature_values=x.tolist(),
            coefficients=self.coef.tolist(),
            contributions=contribs.tolist(),
            intercept=self.intercept,
            logit=z,
            proba_suspect=p_suspect,
            label=label,
            threshold=self.threshold,
        )

    # ------------------------------------------------------------------
    #  Figure dashboard (Plotly) — réutilisée dans dashboard/app.py
    # ------------------------------------------------------------------

    @staticmethod
    def to_plotly_bar(
        decomposition: MetaDecomposition,
        labels_fr: Optional[dict] = None,
    ):
        """
        Bar plot horizontal Plotly des contributions.
        Retourne un `plotly.graph_objects.Figure`.
        """
        import plotly.graph_objects as go

        labels_fr = labels_fr or {
            "score_v5_fiable": "Score V5 (P fiable, TF-IDF)",
            "score_v6_suspect": "Score V6 (P suspect, style)",
            "score_camembert_fiable": "Score CamemBERT (P fiable)",
            "disagreement": "Désaccord V5/V6",
            "disagreement_v5_v6": "Désaccord V5/V6",
            "disagreement_v5_cam": "Désaccord V5/CamemBERT",
            "interaction": "Interaction V5×V6",
            "interaction_v5_v6": "Interaction V5×V6",
            "min_fiable": "Min(V5,CamemBERT)",
        }

        contribs = np.asarray(decomposition.contributions)
        # Ordonner par magnitude
        order = np.argsort(np.abs(contribs))[::-1]
        names = [labels_fr.get(decomposition.feature_names[i],
                               decomposition.feature_names[i]) for i in order]
        vals = contribs[order]
        colors = ["#FF1744" if v > 0 else "#00E676" for v in vals]

        # Inverser pour barh top-down lisible
        names = names[::-1]
        vals = vals[::-1]
        colors = colors[::-1]

        hover = [
            f"{n}<br>x={decomposition.feature_values[order[len(order)-1-i]]:.3f}"
            f"<br>β={decomposition.coefficients[order[len(order)-1-i]]:+.3f}"
            f"<br>β·x={v:+.4f}"
            for i, (n, v) in enumerate(zip(names, vals))
        ]

        fig = go.Figure(
            go.Bar(
                y=names,
                x=vals,
                orientation="h",
                marker_color=colors,
                hovertext=hover,
                hoverinfo="text",
            )
        )
        fig.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            title=dict(
                text=(
                    f"Décomposition de la décision V8 — "
                    f"P(suspect)={decomposition.proba_suspect:.2f} → "
                    f"{decomposition.label}"
                ),
                x=0.5, font=dict(color="#E0E0E0", size=14),
            ),
            xaxis=dict(
                title="Contribution β·x au logit (+ = pousse vers SUSPECT)",
                zeroline=True, zerolinecolor="rgba(255,255,255,0.2)",
            ),
            margin=dict(t=60, b=40, l=200, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E0E0"),
            height=max(280, len(names) * 38),
        )
        return fig
