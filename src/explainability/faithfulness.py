"""
Validation de l'explicabilité — Faithfulness / Fidélité
========================================================

Une explication n'a de valeur que si elle reflète **réellement** le
comportement du modèle. Ce module implémente les métriques standard de
faithfulness en XAI (DeYoung et al. 2020, *ERASER benchmark*) :

* **AOPC (Area Over the Perturbation Curve)** : on masque progressivement
  les top-k features (par importance descendante), on mesure la chute de
  P(classe). AOPC élevé = explication fidèle.
* **Comprehensiveness** : 1 - P(classe | features importantes masquées).
  Élevé = les features expliquées portent vraiment la décision.
* **Sufficiency** : P(classe | seules les features importantes
  conservées). Élevé = les features expliquées suffisent à reproduire la
  décision.

Le module est **agnostique du modèle** : il prend une fonction
`predict_proba_fn(X) -> P(classe)` et un tableau `X` + ses scores
d'importance par échantillon.

Utilisation typique :
- expliquer SHAP sur V6 (35 features), masquer = mettre à 0
- expliquer Integrated Gradients sur CamemBERT, masquer = remplacer
  l'embedding par <pad>

Pour la version transformer, voir `mask_with_pad_for_transformer` qui
gère le masquage en espace tokens plutôt que features tabulaires.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class FaithfulnessResult:
    """Résultat agrégé d'une évaluation de fidélité."""

    n_samples: int
    n_features: int
    k_values: List[int]  # nombre de features masquées à chaque pas
    proba_curve_mean: List[float]  # P(classe) moyenne après k masquages
    proba_curve_std: List[float]
    aopc: float
    comprehensiveness_at_k: dict  # k -> mean delta
    sufficiency_at_k: dict
    figures: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "k_values": self.k_values,
                "proba_curve_mean": self.proba_curve_mean,
                "proba_curve_std": self.proba_curve_std,
                "aopc": self.aopc,
                "comprehensiveness_at_k": self.comprehensiveness_at_k,
                "sufficiency_at_k": self.sufficiency_at_k,
                "figures": self.figures,
            },
            ensure_ascii=False,
            indent=2,
        )


class FaithfulnessEvaluator:
    """
    Évalue la fidélité d'attributions sur un modèle de classification.

    Parameters
    ----------
    predict_proba_fn : Callable[[np.ndarray], np.ndarray]
        Fonction de scoring : `X -> probas[batch, n_classes]`.
        Pour V6 : `lambda X: model.predict_proba(scaler.transform(X))`.
    class_index : int
        Index de la classe pour laquelle on calcule la chute (1=suspect).
    output_dir : str

    Notes
    -----
    Le **masquage par 0** est neutre pour les features stylistiques
    (compte de mots, ratios). Pour des features où 0 est une valeur
    significative (ex: emotions), il vaut mieux masquer par la **moyenne**
    de la feature sur le dataset d'entraînement (param `mask_value`).
    """

    def __init__(
        self,
        predict_proba_fn: Callable[[np.ndarray], np.ndarray],
        class_index: int = 1,
        output_dir: str = "docs/figures/xai",
    ):
        self.predict = predict_proba_fn
        self.class_index = class_index
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    #  Curve & métriques
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        attributions: np.ndarray,
        max_k: Optional[int] = None,
        mask_value: float | np.ndarray = 0.0,
        comprehensiveness_ks: Sequence[int] = (1, 3, 5, 10),
    ) -> FaithfulnessResult:
        """
        Calcule la courbe de perturbation et les métriques agrégées.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        attributions : np.ndarray, shape (n, d)
            Importance par feature et par échantillon (ex: SHAP values).
            Le signe importe peu — on classe par |attributions|.
        max_k : int
            Nombre maximum de features à masquer. Défaut : d.
        mask_value : float ou np.ndarray
            Valeur de remplacement (scalaire ou par-feature).
        comprehensiveness_ks : iterable
            Valeurs de k pour lesquelles calculer comprehensiveness/sufficiency.
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        X = np.asarray(X, dtype=float)
        attributions = np.asarray(attributions, dtype=float)
        n, d = X.shape
        if max_k is None:
            max_k = d
        max_k = min(max_k, d)

        # Probas baseline
        p_base = self.predict(X)[:, self.class_index]  # (n,)

        # Pour chaque sample, ordre décroissant des features par importance
        order = np.argsort(-np.abs(attributions), axis=1)  # (n, d)

        proba_curve = np.zeros((n, max_k + 1))
        proba_curve[:, 0] = p_base

        # Masque cumulatif
        for k in range(1, max_k + 1):
            X_masked = X.copy()
            for i in range(n):
                top_feats = order[i, :k]
                if np.isscalar(mask_value):
                    X_masked[i, top_feats] = mask_value
                else:
                    X_masked[i, top_feats] = mask_value[top_feats]
            proba_curve[:, k] = self.predict(X_masked)[:, self.class_index]

        # AOPC = (1 / (max_k+1)) * Σ (p_base - p_k)
        aopc = float(np.mean(p_base[:, None] - proba_curve, axis=1).mean())

        # Comprehensiveness = p_base - p_k (haut = bonne explication)
        # Sufficiency = p(garder seulement top-k) - p_base (proche de 0 = top-k suffit)
        comp = {}
        suf = {}
        for k in comprehensiveness_ks:
            if k > max_k:
                continue
            comp[int(k)] = float(np.mean(p_base - proba_curve[:, k]))

            # Sufficiency : ne garder que les top-k, masquer le reste
            X_only_top = np.full_like(X, mask_value if np.isscalar(mask_value) else 0.0)
            if not np.isscalar(mask_value):
                X_only_top = np.broadcast_to(mask_value, X.shape).copy()
            for i in range(n):
                keep = order[i, :k]
                X_only_top[i, keep] = X[i, keep]
            p_only = self.predict(X_only_top)[:, self.class_index]
            suf[int(k)] = float(np.mean(p_base - p_only))

        result = FaithfulnessResult(
            n_samples=int(n),
            n_features=int(d),
            k_values=list(range(max_k + 1)),
            proba_curve_mean=proba_curve.mean(axis=0).tolist(),
            proba_curve_std=proba_curve.std(axis=0).tolist(),
            aopc=aopc,
            comprehensiveness_at_k=comp,
            sufficiency_at_k=suf,
        )
        result.figures["aopc_curve"] = self._plot_curve(result)
        return result

    # ------------------------------------------------------------------
    #  Comparaison aléatoire (sanity check)
    # ------------------------------------------------------------------

    def compare_with_random(
        self,
        X: np.ndarray,
        attributions: np.ndarray,
        n_random_seeds: int = 5,
        **kwargs,
    ) -> dict:
        """
        Compare la courbe d'attribution avec des courbes aléatoires.
        Une attribution faithful doit donner AOPC > AOPC_random significativement.
        """
        result_attr = self.evaluate(X, attributions, **kwargs)
        rng_aopcs = []
        for seed in range(n_random_seeds):
            rng = np.random.default_rng(seed)
            random_attr = rng.standard_normal(size=attributions.shape)
            r = self.evaluate(X, random_attr, **kwargs)
            rng_aopcs.append(r.aopc)
        return {
            "aopc_attribution": result_attr.aopc,
            "aopc_random_mean": float(np.mean(rng_aopcs)),
            "aopc_random_std": float(np.std(rng_aopcs)),
            "aopc_uplift": float(result_attr.aopc - np.mean(rng_aopcs)),
            "result": result_attr,
        }

    # ------------------------------------------------------------------
    #  Figure
    # ------------------------------------------------------------------

    def _plot_curve(self, result: FaithfulnessResult) -> str:
        import os
        import matplotlib.pyplot as plt

        ks = np.asarray(result.k_values)
        mean = np.asarray(result.proba_curve_mean)
        std = np.asarray(result.proba_curve_std)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ks, mean, color="#FF1744", linewidth=2,
                label="P(classe expliquée)")
        ax.fill_between(ks, mean - std, mean + std, color="#FF1744", alpha=0.18,
                        label="±1σ")
        ax.set_xlabel("k = nombre de top features masquées")
        ax.set_ylabel(f"P(classe={self.class_index}) moyenne")
        ax.set_title(
            f"Courbe de perturbation — Faithfulness Test\n"
            f"AOPC = {result.aopc:.4f} sur n={result.n_samples} échantillons "
            f"(plus haut = meilleur)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        path = os.path.join(self.output_dir, "faithfulness_aopc_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
