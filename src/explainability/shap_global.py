"""
Explicabilité globale via SHAP — V6 Style GradientBoosting
==========================================================

Produit les figures de référence pour un rapport ML senior :

* **Beeswarm summary plot** : distribution complète des SHAP values pour les
  top-N features. Permet de voir d'un coup la direction (suspect/fiable),
  l'amplitude et la dispersion de chaque feature.
* **Dependence plots** : effet marginal d'une feature sur la sortie du
  modèle, coloré par la feature qui interagit le plus (auto-détection).
* **Bar plot global** : importance moyenne `mean(|SHAP|)` — déjà couvert
  par le notebook 24, on le ré-expose ici pour cohérence.

Le module fonctionne sur **n'importe quel modèle tree-based** chargé depuis
`models/model_style_v6.joblib` ou `model_style_v6_extended.joblib`.

Référence : Lundberg & Lee (2017), "A Unified Approach to Interpreting
Model Predictions", NeurIPS.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GlobalShapResult:
    """Résultat sérialisable d'une analyse SHAP globale."""

    feature_names: list
    mean_abs_shap: list  # importance globale, len == n_features
    sorted_indices: list  # indices triés par importance décroissante
    n_samples: int
    n_features: int
    model_name: str
    figures: dict = field(default_factory=dict)  # nom -> chemin figure

    def to_json(self) -> str:
        return json.dumps(
            {
                "feature_names": self.feature_names,
                "mean_abs_shap": self.mean_abs_shap,
                "sorted_indices": self.sorted_indices,
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "model_name": self.model_name,
                "figures": self.figures,
            },
            ensure_ascii=False,
            indent=2,
        )


class GlobalShapExplainer:
    """
    Explicabilité globale d'un modèle tree-based (GradientBoosting,
    RandomForest, XGBoost, LightGBM, CatBoost) ou linéaire.

    Parameters
    ----------
    model : object
        Modèle sklearn-compatible avec `predict_proba` ou `predict`.
    feature_names : Sequence[str]
        Noms des features (longueur égale à `X.shape[1]`).
    output_dir : str
        Dossier de sortie pour les figures. Créé si inexistant.
    class_index : int
        Index de la classe à expliquer (1 = suspect dans nos modèles).

    Examples
    --------
    >>> import joblib
    >>> v6 = joblib.load('models/model_style_v6.joblib')
    >>> explainer = GlobalShapExplainer(
    ...     model=v6['model'],
    ...     feature_names=v6['feature_names'],
    ...     output_dir='docs/figures/xai',
    ... )
    >>> result = explainer.explain(X_gold)
    >>> result.figures['beeswarm']
    'docs/figures/xai/shap_beeswarm_v6.png'
    """

    def __init__(
        self,
        model,
        feature_names: Sequence[str],
        output_dir: str = "docs/figures/xai",
        class_index: int = 1,
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.output_dir = output_dir
        self.class_index = class_index
        os.makedirs(output_dir, exist_ok=True)

        self._shap_values = None
        self._X = None

    # ------------------------------------------------------------------
    #  Calcul des SHAP values
    # ------------------------------------------------------------------

    def _build_explainer(self, X: np.ndarray):
        """Choisit le bon explainer SHAP selon le type de modèle."""
        import shap

        model_type = type(self.model).__name__
        # Tree-based — TreeExplainer (exact, fast)
        tree_models = {
            "GradientBoostingClassifier",
            "RandomForestClassifier",
            "XGBClassifier",
            "LGBMClassifier",
            "CatBoostClassifier",
        }
        if model_type in tree_models:
            return shap.TreeExplainer(self.model)
        # Linéaire — LinearExplainer
        if hasattr(self.model, "coef_"):
            return shap.LinearExplainer(self.model, X)
        # Fallback générique — KernelExplainer (lent, à n'utiliser qu'en dernier recours)
        logger.warning(
            "Modèle %s : fallback KernelExplainer (lent). "
            "Privilégier un modèle tree-based ou linéaire.",
            model_type,
        )
        background = shap.sample(X, min(50, len(X)), random_state=42)
        return shap.KernelExplainer(self.model.predict_proba, background)

    def _compute_shap(self, X: np.ndarray) -> np.ndarray:
        """Calcule les SHAP values pour la classe `class_index`."""
        explainer = self._build_explainer(X)
        sv = explainer.shap_values(X)
        # SHAP renvoie soit (n, d) soit list de [n, d] par classe selon la version
        if isinstance(sv, list):
            return np.asarray(sv[self.class_index])
        if sv.ndim == 3:  # (n, d, n_classes) — versions récentes de shap
            return sv[:, :, self.class_index]
        return sv

    # ------------------------------------------------------------------
    #  API publique
    # ------------------------------------------------------------------

    def explain(self, X: np.ndarray) -> GlobalShapResult:
        """
        Calcule les SHAP values et produit toutes les figures globales.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Matrice de features (déjà scalée si le modèle l'exige).

        Returns
        -------
        GlobalShapResult
        """
        X = np.asarray(X)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Mismatch features: X a {X.shape[1]} colonnes, "
                f"feature_names en a {len(self.feature_names)}"
            )

        logger.info("Calcul SHAP sur %d échantillons, %d features…", *X.shape)
        self._shap_values = self._compute_shap(X)
        self._X = X

        mean_abs = np.abs(self._shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]

        result = GlobalShapResult(
            feature_names=self.feature_names,
            mean_abs_shap=mean_abs.tolist(),
            sorted_indices=sorted_idx.tolist(),
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            model_name=type(self.model).__name__,
        )

        # Figures
        result.figures["beeswarm"] = self._plot_beeswarm()
        result.figures["bar_global"] = self._plot_bar_global(mean_abs, sorted_idx)
        # Top 3 dependence plots
        for rank in range(min(3, len(sorted_idx))):
            idx = int(sorted_idx[rank])
            path = self._plot_dependence(idx)
            result.figures[f"dependence_{self.feature_names[idx]}"] = path

        return result

    # ------------------------------------------------------------------
    #  Figures
    # ------------------------------------------------------------------

    def _plot_beeswarm(self, max_display: int = 20) -> str:
        """Beeswarm summary plot (figure de référence d'un rapport SHAP)."""
        import matplotlib.pyplot as plt
        import shap

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self._shap_values,
            self._X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title(
            f"SHAP Beeswarm — {self.model_name_for_title()}\n"
            f"top {max_display} features par mean(|SHAP|), n={self._X.shape[0]}",
            fontsize=11,
        )
        plt.tight_layout()
        path = os.path.join(self.output_dir, "shap_beeswarm_v6.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def _plot_bar_global(self, mean_abs: np.ndarray, sorted_idx: np.ndarray,
                        max_display: int = 20) -> str:
        """Bar plot d'importance globale (figure 'classique')."""
        import matplotlib.pyplot as plt

        top = sorted_idx[:max_display][::-1]  # bas vers haut
        names = [self.feature_names[i] for i in top]
        values = mean_abs[top]

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
        ax.barh(range(len(top)), values, color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("mean(|SHAP value|)", fontsize=10)
        ax.set_title(
            f"Importance globale des features — {self.model_name_for_title()}",
            fontsize=11,
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "shap_global_bar_v6.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_dependence(self, feature_idx: int) -> str:
        """Dependence plot pour une feature, interaction auto-détectée."""
        import matplotlib.pyplot as plt
        import shap

        fname = self.feature_names[feature_idx]
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feature_idx,
            self._shap_values,
            self._X,
            feature_names=self.feature_names,
            interaction_index="auto",
            show=False,
        )
        plt.title(
            f"Dependence — {fname}\n"
            f"effet marginal + interaction la plus forte (auto-détectée)",
            fontsize=10,
        )
        plt.tight_layout()
        safe = "".join(c if c.isalnum() else "_" for c in fname)
        path = os.path.join(self.output_dir, f"shap_dependence_{safe}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def model_name_for_title(self) -> str:
        return type(self.model).__name__

    # ------------------------------------------------------------------
    #  Accesseurs
    # ------------------------------------------------------------------

    @property
    def shap_values(self) -> Optional[np.ndarray]:
        return self._shap_values
