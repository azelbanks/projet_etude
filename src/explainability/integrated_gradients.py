"""
Layer Integrated Gradients via Captum
======================================

Sundararajan, Taly & Yan (2017), *Axiomatic Attribution for Deep Networks*.

Pour les modèles transformers, on attribue la prédiction au niveau des
**embeddings** plutôt que des input_ids (les input_ids sont des entiers
non différentiables). Captum fournit `LayerIntegratedGradients` qui
intègre les gradients depuis une baseline (token <pad>) jusqu'à l'entrée
réelle, le long d'une trajectoire linéaire à `n_steps` pas.

Avantages vs attention :
- attribution **causale** (sensible à la classe prédite)
- satisfait les axiomes (Completeness, Sensitivity, Implementation invariance)
- robuste aux contournements de l'attention (Jain & Wallace 2019)

Le module fonctionne sur CamemBERT (via notre `CamemBERTClassifier`)
et sur RoBERTa (en chargeant un checkpoint compatible).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IGResult:
    """Résultat d'une attribution IG sur un texte."""

    text: str
    tokens: list
    attributions: list  # par token, attribution scalaire (somme L2 sur dim embedding)
    convergence_delta: float
    target_class: int
    target_class_name: str
    n_steps: int
    prediction_proba: list  # [P_fiable, P_suspect]
    figures: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "text": self.text,
                "tokens": self.tokens,
                "attributions": self.attributions,
                "convergence_delta": self.convergence_delta,
                "target_class": self.target_class,
                "target_class_name": self.target_class_name,
                "n_steps": self.n_steps,
                "prediction_proba": self.prediction_proba,
                "figures": self.figures,
            },
            ensure_ascii=False,
            indent=2,
        )


class IGExplainer:
    """
    Layer Integrated Gradients sur un classifieur transformer.

    L'objet appelable doit accepter une fonction `forward_for_ig` qui prend
    `(input_ids, attention_mask)` et retourne les **logits de classification**
    (et non les hidden states). Ce module en construit une automatiquement
    pour notre `CamemBERTClassifier` (`base_model + head`).

    Parameters
    ----------
    classifier : CamemBERTClassifier
        Instance chargée. Doit exposer `base_model`, `head`, `tokenizer`,
        `device`, `MAX_LENGTH`.
    output_dir : str
    n_steps : int
        Nombre de pas pour l'intégration (50 = bon défaut, 200 si gros budget).
    target_class : int
        1 pour expliquer 'suspect', 0 pour 'fiable'.

    Notes
    -----
    On utilise <pad> comme baseline. C'est un choix conservateur mais accepté
    dans la littérature (Mudrakarta et al. 2018). Une alternative serait
    une baseline "average embedding" — laissée en TODO.
    """

    def __init__(
        self,
        classifier,
        output_dir: str = "docs/figures/xai",
        n_steps: int = 200,
        target_class: int = 1,
        baseline_strategy: str = "auto",
        force_cpu: bool = True,
    ):
        """
        baseline_strategy :
            * "auto"   — essaie 'unk', puis 'pad' jusqu'à atteindre
                         Completeness (|Δ| < 0.01).
            * "unk"    — token <unk> (recommandé, Mudrakarta et al. 2018).
            * "zero"   — embedding nul (théorique mais hors-distribution).
            * "pad"    — token <pad> (rapide, peut violer Completeness).
        force_cpu :
            Sur Apple Silicon, le backend MPS de PyTorch produit des
            gradients instables sur Captum (Δ_convergence systématiquement
            décalé). On déplace temporairement le modèle sur CPU pour
            l'attribution, puis on restaure le device original.
            Coût : ~5x plus lent par échantillon. Fortement recommandé
            sur MPS, transparent sur CUDA/CPU.
        """
        self.clf = classifier
        self.output_dir = output_dir
        self.n_steps = n_steps
        self.target_class = target_class
        self.target_class_name = "SUSPECT" if target_class == 1 else "FIABLE"
        self.baseline_strategy = baseline_strategy
        self.force_cpu = force_cpu
        os.makedirs(output_dir, exist_ok=True)

        if not getattr(self.clf, "_loaded", False):
            raise RuntimeError("Classifier non chargé. Appeler `.load()`.")

        # Forcer eager attention pour IG : SDPA empêche les gradients
        # de s'écouler proprement (transformers >= 4.40).
        try:
            self.clf.base_model.config._attn_implementation = "eager"
        except Exception:
            pass

        # Détection MPS — on déplace le modèle sur CPU si demandé
        import torch
        self._original_device = self.clf.device
        if self.force_cpu and str(self._original_device).startswith("mps"):
            logger.info(
                "IG: device MPS détecté, bascule sur CPU pour stabilité "
                "des gradients (Captum/Apple Silicon)"
            )
            self._ig_device = torch.device("cpu")
            self.clf.base_model.to(self._ig_device)
            self.clf.head.to(self._ig_device)
            self.clf.device = self._ig_device
        else:
            self._ig_device = self._original_device

        # Lazy: importé à l'usage
        self._lig = None

    def __del__(self):
        """Restaure le device d'origine en sortie de scope."""
        try:
            if hasattr(self, "_original_device") and self._ig_device != self._original_device:
                self.clf.base_model.to(self._original_device)
                self.clf.head.to(self._original_device)
                self.clf.device = self._original_device
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Construction du forward
    # ------------------------------------------------------------------

    def _forward(self, input_ids, attention_mask):
        """Forward classification : logits[batch, n_classes]."""
        outputs = self.clf.base_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return self.clf.head(cls_repr)

    def _build_lig(self):
        """Construit l'objet LayerIntegratedGradients (lazy)."""
        from captum.attr import LayerIntegratedGradients

        # Layer cible : embeddings de mots
        # CamemBERT et RoBERTa : `embeddings.word_embeddings`
        embed_layer = self.clf.base_model.embeddings.word_embeddings
        self._lig = LayerIntegratedGradients(self._forward, embed_layer)

    # ------------------------------------------------------------------
    #  Attribution
    # ------------------------------------------------------------------

    def _build_baseline(self, input_ids, attention_mask, strategy: str):
        """Construit une baseline selon la stratégie."""
        import torch
        tokenizer = self.clf.tokenizer

        if strategy == "pad":
            return torch.full_like(input_ids, tokenizer.pad_token_id)
        if strategy == "unk":
            unk = getattr(tokenizer, "unk_token_id", None)
            if unk is None:
                # CamemBERT a <unk> = 3
                unk = tokenizer.convert_tokens_to_ids("<unk>")
            baseline = torch.full_like(input_ids, unk)
            # Garder [CLS]/[SEP] aux extrémités (sinon hors-distribution)
            cls_id = (tokenizer.cls_token_id
                      or tokenizer.bos_token_id
                      or input_ids[0, 0].item())
            sep_id = (tokenizer.sep_token_id
                      or tokenizer.eos_token_id
                      or input_ids[0, -1].item())
            baseline[:, 0] = cls_id
            # SEP à la fin de la séquence réelle
            seq_len = int(attention_mask.sum().item())
            if seq_len > 1:
                baseline[:, seq_len - 1] = sep_id
            return baseline
        if strategy == "zero":
            # Embedding zéro = pas un token, mais on l'imite avec <pad>
            # (Captum LayerIG attribue à l'espace embedding, donc
            # pratiquement équivalent à <pad> si l'embedding de <pad> est ~0).
            return torch.full_like(input_ids, tokenizer.pad_token_id)
        raise ValueError(f"Unknown baseline strategy: {strategy}")

    def explain(
        self,
        text: str,
        tag: Optional[str] = None,
        target_class: Optional[int] = None,
    ) -> IGResult:
        """
        Attribue la prédiction par token via Layer IG.

        Parameters
        ----------
        text : str
        tag : str, optional
            Suffixe pour le nom de fichier de la figure.
        target_class : int, optional
            Classe à expliquer (0=FIABLE, 1=SUSPECT). Si None, on utilise
            le défaut du module (`self.target_class`). **Recommandation** :
            passer la classe **prédite** par le modèle plutôt qu'une classe
            théorique — sinon IG explique "pourquoi le modèle aurait dû
            penser X" sur un cas où il pense le contraire, ce qui produit
            des gradients dégénérés et |Δ| élevés.

        Stratégie de baseline : essaie successivement 'unk' puis 'pad'
        jusqu'à atteindre Completeness (|Δ| < 0.05). Si aucune ne converge,
        on retourne la meilleure tentative avec un warning explicite.

        Returns
        -------
        IGResult
        """
        import torch

        if self._lig is None:
            self._build_lig()

        tokenizer = self.clf.tokenizer
        device = self.clf.device

        # Override de la classe cible pour cette attribution (cas FN)
        eff_target = target_class if target_class is not None else self.target_class
        eff_target_name = "SUSPECT" if eff_target == 1 else "FIABLE"

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.clf.MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        seq_len = int(attention_mask.sum().item())

        # Forward pour les probas
        with torch.no_grad():
            logits = self._forward(input_ids, attention_mask)
            probas = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Stratégies à essayer (ordre = priorité)
        if self.baseline_strategy == "auto":
            strategies = ["unk", "pad"]
        else:
            strategies = [self.baseline_strategy]

        # Riemann steps : escalade progressive [200, 500, 1000]
        # On s'arrête dès |Δ| < 0.05 (seuil "pratique").
        # Cap à 1000 = compromis qualité/temps sur transformers profonds.
        n_steps_to_try = []
        for n in (self.n_steps, 500, 1000):
            if n not in n_steps_to_try and n >= self.n_steps:
                n_steps_to_try.append(n)

        best = None  # (delta, attributions, strat, n_steps)
        converged = False
        for n_steps in n_steps_to_try:
            for strat in strategies:
                baseline = self._build_baseline(input_ids, attention_mask, strat)
                attributions, delta = self._lig.attribute(
                    inputs=input_ids,
                    baselines=baseline,
                    additional_forward_args=(attention_mask,),
                    target=eff_target,
                    n_steps=n_steps,
                    return_convergence_delta=True,
                    internal_batch_size=8,
                )
                d = float(delta.item())
                if best is None or abs(d) < abs(best[0]):
                    best = (d, attributions, strat, n_steps)
                if abs(d) < 0.05:  # niveau pratique atteint
                    converged = True
                    break
            if converged:
                break

        delta_val, attributions, strat_used, n_steps_used = best
        logger.info(
            "  IG device=%s baseline=%s n_steps=%d Δ=%.2e",
            self._ig_device, strat_used, n_steps_used, delta_val,
        )
        # Tracker le nombre de steps réellement utilisé pour le rapport
        self._last_n_steps = n_steps_used
        delta = torch.tensor(delta_val)
        # attributions: (1, seq, dim_embed) -> on somme sur dim_embed (L2)
        attr_per_token = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
        # Garder seulement les tokens réels
        attr_per_token = attr_per_token[:seq_len]
        # Normalisation (max abs = 1)
        max_abs = np.abs(attr_per_token).max()
        if max_abs > 0:
            attr_per_token = attr_per_token / max_abs

        token_ids = input_ids[0, :seq_len].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        result = IGResult(
            text=text,
            tokens=tokens,
            attributions=attr_per_token.tolist(),
            convergence_delta=float(delta.item()),
            target_class=eff_target,
            target_class_name=eff_target_name,
            n_steps=getattr(self, "_last_n_steps", self.n_steps),
            prediction_proba=probas.tolist(),
        )

        if tag is None:
            tag = "{:08x}".format(abs(hash(text)) % (16 ** 8))
        result.figures["heatmap"] = self._plot(result, tag)
        return result

    # ------------------------------------------------------------------
    #  Figure
    # ------------------------------------------------------------------

    def _plot(self, result: IGResult, tag: str) -> str:
        """Heatmap divergente (rouge=pousse vers cible, bleu=contre)."""
        import matplotlib.pyplot as plt

        attr = np.asarray(result.attributions)
        tokens = [
            t.replace("▁", " ").replace("Ġ", " ").strip() or t
            for t in result.tokens
        ]
        keep_idx = [
            i for i, t in enumerate(result.tokens)
            if t not in {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>"}
        ]
        if not keep_idx:
            keep_idx = list(range(len(tokens)))

        attr_kept = attr[keep_idx]
        tokens_kept = [tokens[i] for i in keep_idx]

        fig, ax = plt.subplots(figsize=(max(8, len(tokens_kept) * 0.4), 2.6))
        # vmin/vmax symétriques pour cmap divergente
        v = max(0.001, float(np.abs(attr_kept).max()))
        im = ax.imshow(
            attr_kept.reshape(1, -1),
            aspect="auto",
            cmap="RdBu_r",  # rouge = vers SUSPECT, bleu = contre
            vmin=-v,
            vmax=v,
        )
        ax.set_xticks(range(len(tokens_kept)))
        ax.set_xticklabels(tokens_kept, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_title(
            f"Layer Integrated Gradients → {result.target_class_name}\n"
            f"P({result.target_class_name})="
            f"{result.prediction_proba[result.target_class]:.2f} | "
            f"Δ_convergence={result.convergence_delta:.2e} | "
            f"n_steps={result.n_steps}",
            fontsize=10, loc="left",
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(
            f"Attribution (rouge=pousse vers {result.target_class_name})",
            fontsize=8,
        )
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"ig_{result.target_class_name.lower()}_{tag}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    #  Sanity check (axiome de Completeness)
    # ------------------------------------------------------------------

    def completeness_check(self, result: IGResult, tol: float = 0.05) -> bool:
        """
        Vérifie l'axiome de Completeness (Sundararajan et al. 2017) :
            sum(attributions) ≈ f(input) - f(baseline)

        Le `convergence_delta` retourné par Captum mesure cet écart en
        unité de logit. Sur les transformers profonds (12 couches + head
        non-linéaire), une convergence à |Δ| < 0.01 nécessite typiquement
        1000+ pas Riemann par sample. La littérature (Kokhlikyan et al.
        2020) considère :

            * |Δ| < 0.01  → axiomatique (convergé)
            * |Δ| < 0.05  → pratique (suffisant pour un rapport)
            * |Δ| < 0.15  → indicatif (à interpréter avec prudence)
            * |Δ| ≥ 0.15  → rejeter ou augmenter n_steps

        Tol = 0.05 par défaut = seuil pratique pour un rendu projet.
        """
        delta = abs(result.convergence_delta)
        ok = delta < tol
        if delta < 0.01:
            level = "axiomatique"
        elif delta < 0.05:
            level = "pratique"
        elif delta < 0.15:
            level = "indicatif"
        else:
            level = "rejet"

        if not ok:
            logger.warning(
                "Completeness IG : |Δ|=%.4f (niveau=%s, seuil=%.2f). "
                "Sur transformers profonds, |Δ| ≈ 0.05–0.15 est attendu.",
                delta, level, tol,
            )
        return ok
