"""
Visualisation des attention weights de CamemBERT
=================================================

Les transformers HuggingFace exposent les attention weights via
`output_attentions=True`. Ce module récupère ces poids, les agrège
proprement (moyenne sur les têtes, focus sur le token [CLS]) et produit :

* une **heatmap par token** (matplotlib) prête à insérer dans un rapport,
* un objet `AttentionResult` sérialisable JSON,
* un export HTML coloré (style BertViz simplifié) intégrable dans le
  dashboard Streamlit via `st.components.v1.html`.

Attention agrégée ≠ explication causale (Jain & Wallace 2019). Pour un
attribution rigoureuse, voir `integrated_gradients.IGExplainer`. Cette
visualisation reste utile pour la **transparence métier** et le débogage
qualitatif (identifier les tokens "regardés" par le modèle).

Compatible CamemBERT, RoBERTa, BERT — toute architecture exposée par
`transformers.AutoModel(..., output_attentions=True)`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttentionResult:
    """Résultat d'une analyse attention pour un texte."""

    text: str
    tokens: List[str]
    # cls_attention[layer][head][token] -> attention de [CLS] vers `token`
    cls_attention_per_head: List[List[List[float]]]
    cls_attention_avg_heads: List[List[float]]  # [layer][token]
    cls_attention_last_layer: List[float]  # par token, dernière couche, moyenne têtes
    prediction_label: str
    prediction_proba_suspect: float
    prediction_proba_fiable: float
    ground_truth: Optional[str] = None
    error_type: Optional[str] = None  # TP / FP / FN / TN
    figures: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "text": self.text,
                "tokens": self.tokens,
                "cls_attention_last_layer": self.cls_attention_last_layer,
                "cls_attention_avg_heads_per_layer": self.cls_attention_avg_heads,
                "prediction_label": self.prediction_label,
                "prediction_proba_suspect": self.prediction_proba_suspect,
                "prediction_proba_fiable": self.prediction_proba_fiable,
                "ground_truth": self.ground_truth,
                "error_type": self.error_type,
                "figures": self.figures,
            },
            ensure_ascii=False,
            indent=2,
        )


class CamembertAttentionExplainer:
    """
    Extrait et visualise les attention weights d'un CamemBERT fine-tuné.

    Le modèle attendu suit le pattern de `src.pipeline.camembert_classifier
    .CamemBERTClassifier` : `base_model` (HuggingFace) + `head` (Linear).

    Parameters
    ----------
    classifier : CamemBERTClassifier
        Instance déjà chargée via `.load()`.
    output_dir : str
        Dossier de sortie pour les figures.

    Notes
    -----
    L'agrégation par défaut est : moyenne sur les têtes de la dernière
    couche, focus sur la ligne du token [CLS] (token 0). C'est l'agrégation
    standard pour les classifieurs CLS-based (Vig 2019, Clark et al. 2019).
    """

    def __init__(self, classifier, output_dir: str = "docs/figures/xai"):
        self.clf = classifier
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not getattr(self.clf, "_loaded", False):
            raise RuntimeError(
                "CamemBERT classifier non chargé. Appeler `.load()` d'abord."
            )

    # ------------------------------------------------------------------
    #  Extraction des attention weights
    # ------------------------------------------------------------------

    def _forward_with_attentions(self, text: str) -> Tuple:
        """Forward pass qui renvoie (tokens, attentions, logits, probas)."""
        import torch

        tokenizer = self.clf.tokenizer
        device = self.clf.device

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.clf.MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Tokens lisibles (sans padding)
        seq_len = int(attention_mask.sum().item())
        token_ids = input_ids[0, :seq_len].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        with torch.no_grad():
            outputs = self.clf.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            cls_repr = outputs.last_hidden_state[:, 0, :]
            logits = self.clf.head(cls_repr)
            probas = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # attentions: tuple of (n_layers,) tensors of shape (1, n_heads, seq, seq)
            attns = [a[0].cpu().numpy() for a in outputs.attentions]
            # Tronquer chaque attention map à la longueur réelle (sans padding)
            attns = [a[:, :seq_len, :seq_len] for a in attns]

        return tokens, attns, probas

    def explain(
        self,
        text: str,
        ground_truth: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> AttentionResult:
        """
        Calcule l'attention pour un texte et produit la figure heatmap.

        Parameters
        ----------
        text : str
        ground_truth : str, optional
            'SUSPECT' ou 'FIABLE'. Si fourni, calcule `error_type` (TP/FP/FN/TN).
        tag : str, optional
            Suffixe pour le nom de fichier (ex: 'tp_001'). Sinon, hash du texte.
        """
        tokens, attns, probas = self._forward_with_attentions(text)

        # CamemBERT: probas[1] = suspect, probas[0] = fiable
        proba_fiable = float(probas[0])
        proba_suspect = float(probas[1])
        pred_label = "SUSPECT" if proba_suspect >= 0.5 else "FIABLE"

        error_type = self._error_type(pred_label, ground_truth)

        # Agrégation : (layer, head, seq, seq) -> attention de [CLS] = ligne 0
        # cls_attn_per_head[layer][head][token] = attention de [CLS] vers `token`
        cls_attn_per_head = [a[:, 0, :].tolist() for a in attns]  # liste de [n_heads, seq]
        cls_attn_avg_heads = [a[:, 0, :].mean(axis=0).tolist() for a in attns]  # [n_layers, seq]
        cls_attn_last = cls_attn_avg_heads[-1]  # dernière couche

        result = AttentionResult(
            text=text,
            tokens=tokens,
            cls_attention_per_head=cls_attn_per_head,
            cls_attention_avg_heads=cls_attn_avg_heads,
            cls_attention_last_layer=cls_attn_last,
            prediction_label=pred_label,
            prediction_proba_suspect=proba_suspect,
            prediction_proba_fiable=proba_fiable,
            ground_truth=ground_truth,
            error_type=error_type,
        )

        if tag is None:
            tag = "{:08x}".format(abs(hash(text)) % (16 ** 8))

        result.figures["heatmap"] = self._plot_heatmap(result, tag)
        result.figures["per_layer"] = self._plot_per_layer_heatmap(result, tag)
        return result

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_type(pred: str, gt: Optional[str]) -> Optional[str]:
        if gt is None:
            return None
        gt = gt.upper()
        pred = pred.upper()
        if pred == "SUSPECT" and gt == "SUSPECT":
            return "TP"
        if pred == "SUSPECT" and gt == "FIABLE":
            return "FP"
        if pred == "FIABLE" and gt == "SUSPECT":
            return "FN"
        return "TN"

    @staticmethod
    def _clean_token(t: str) -> str:
        # CamemBERT/RoBERTa utilise ▁ comme séparateur de mot
        return t.replace("▁", " ").replace("Ġ", " ").strip() or t

    # ------------------------------------------------------------------
    #  Figures
    # ------------------------------------------------------------------

    def _plot_heatmap(self, result: AttentionResult, tag: str) -> str:
        """Heatmap horizontale : token x intensité attention CLS dernière couche."""
        import matplotlib.pyplot as plt

        attn = np.asarray(result.cls_attention_last_layer)
        tokens = [self._clean_token(t) for t in result.tokens]

        # Filtrer [CLS] et [SEP]/<s>/</s> de la viz
        keep_idx = [
            i for i, t in enumerate(result.tokens)
            if t not in {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>"}
        ]
        if not keep_idx:
            keep_idx = list(range(len(tokens)))

        attn_kept = attn[keep_idx]
        tokens_kept = [tokens[i] for i in keep_idx]

        # Normaliser pour la viz (max=1 sur la portion gardée)
        if attn_kept.max() > 0:
            attn_norm = attn_kept / attn_kept.max()
        else:
            attn_norm = attn_kept

        fig, ax = plt.subplots(figsize=(max(8, len(tokens_kept) * 0.4), 2.4))
        im = ax.imshow(
            attn_norm.reshape(1, -1),
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
        )
        ax.set_xticks(range(len(tokens_kept)))
        ax.set_xticklabels(tokens_kept, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])

        title_parts = [
            f"Attention [CLS] → tokens (dernière couche, moyenne têtes)",
            f"Prédiction: {result.prediction_label} "
            f"(P_suspect={result.prediction_proba_suspect:.2f})",
        ]
        if result.error_type:
            title_parts.append(
                f"Vérité: {result.ground_truth} → {result.error_type}"
            )
        ax.set_title("\n".join(title_parts), fontsize=10, loc="left")

        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Attention normalisée", fontsize=8)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"camembert_attention_{tag}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_per_layer_heatmap(self, result: AttentionResult, tag: str) -> str:
        """Heatmap 2D : layers (Y) x tokens (X), valeur = attention CLS."""
        import matplotlib.pyplot as plt

        per_layer = np.asarray(result.cls_attention_avg_heads)  # (n_layers, seq)
        tokens = [self._clean_token(t) for t in result.tokens]

        keep_idx = [
            i for i, t in enumerate(result.tokens)
            if t not in {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>"}
        ]
        if not keep_idx:
            keep_idx = list(range(len(tokens)))
        per_layer_kept = per_layer[:, keep_idx]
        tokens_kept = [tokens[i] for i in keep_idx]

        fig, ax = plt.subplots(
            figsize=(max(8, len(tokens_kept) * 0.4), max(4, per_layer.shape[0] * 0.35))
        )
        im = ax.imshow(per_layer_kept, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(tokens_kept)))
        ax.set_xticklabels(tokens_kept, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(per_layer.shape[0]))
        ax.set_yticklabels([f"Layer {i}" for i in range(per_layer.shape[0])], fontsize=8)
        ax.set_title(
            f"Attention CLS par couche — {result.error_type or result.prediction_label}",
            fontsize=10,
        )
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"camembert_attention_layers_{tag}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def render_html(self, result: AttentionResult) -> str:
        """
        Rendu HTML inline (chaque token coloré selon son attention).
        Utilisable dans Streamlit via `st.components.v1.html(html_str)`.
        """
        attn = np.asarray(result.cls_attention_last_layer)
        if attn.max() > 0:
            attn = attn / attn.max()
        tokens = [self._clean_token(t) for t in result.tokens]

        spans = []
        for tok, a in zip(tokens, attn):
            if tok in {"<s>", "</s>", "[CLS]", "[SEP]", "<pad>"}:
                continue
            # Rouge plus saturé = plus d'attention
            r = int(255)
            g = int(255 * (1 - a))
            b = int(255 * (1 - a))
            spans.append(
                f'<span title="attn={a:.3f}" '
                f'style="background:rgb({r},{g},{b});padding:2px 4px;'
                f'margin:1px;border-radius:3px;display:inline-block;'
                f'font-family:monospace;font-size:13px">{tok}</span>'
            )
        body = "".join(spans)
        return (
            f'<div style="line-height:2.2em;padding:12px;'
            f'background:#1a1f2e;border-radius:8px;color:#E8E8E8">'
            f'<div style="font-size:11px;color:#B0B0B0;margin-bottom:6px">'
            f'Pred: {result.prediction_label} '
            f'(P_suspect={result.prediction_proba_suspect:.2f})'
            f'</div>{body}</div>'
        )
