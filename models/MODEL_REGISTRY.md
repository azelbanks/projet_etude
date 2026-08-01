# Model Registry

All trained models are stored locally in this directory. Large PyTorch models
(CamemBERT, RoBERTa) are excluded from git via `.gitignore` — only lightweight
scikit-learn/joblib models are tracked.

## Production Models (V9 Cascade)

| Model | File | Type | Metrics | Dataset | Date |
|-------|------|------|---------|---------|------|
| **V5 Expert** | `model_expert_v5.pkl` | LogReg + TF-IDF | F1: 0.913 | 197K texts (FR+EN) | Apr 2026 |
| **V6 Style** | `model_style_v6.joblib` | LogReg (style features only) | F1: 0.88 | 197K texts | Apr 2026 |
| **V7 Hybrid** | `model_hybrid_v7.joblib` | Ensemble (V5+V6) | F1: 0.92 | 197K texts | May 2026 |
| **V8 Hybrid+CamemBERT** | `model_hybrid_v8.joblib` | Meta-learner (V5+V6+CamemBERT) | F1: 0.94 | 197K texts | May 2026 |
| **V9 Stage 1** | `stage1_fact_opinion.joblib` | Fact/opinion gate | FP -67% | 197K texts | May 2026 |
| **Emotion MLP** | `emotion_bilingual.pt` | PyTorch MLP (7 classes) | Accuracy: 0.82 | 50K labeled | Mar 2026 |

## CamemBERT & RoBERTa (not in git — too large)

| Model | Performance | Training |
|-------|------------|----------|
| CamemBERT FR V2 | F1: 0.957 (ultra-short) | 3 epochs, lr=2e-5, batch=16 |
| RoBERTa EN V2 | F1: 0.874 (ultra-short) | 3 epochs, lr=2e-5, batch=16 |

To reproduce: see `notebooks/training/13_FineTune_CamemBERT_FR.py` and
`notebooks/training/18_FineTune_RoBERTa_EN.py`.

## Version History

| Version | Date | Innovation | Global F1 |
|---------|------|-----------|-----------|
| V1.0 | Dec 2025 | TF-IDF baseline (Reuters-biased) | 0.996* |
| V2.0 | Jan 2026 | Reuters bias removed | 0.89 |
| V3.0 | Feb 2026 | Linguistic features added | 0.90 |
| V4.0 | Mar 2026 | Bilingual pipeline | 0.91 |
| V5.0 | Apr 2026 | +10K synthetic FR social texts | 0.913 |
| V6.0 | Apr 2026 | Style-only features | 0.88 |
| V7.0 | May 2026 | Hybrid ensemble (V5+V6) | 0.92 |
| V8.0 | May 2026 | Meta-learner + CamemBERT | 0.94 |
| **V9.0** | **May 2026** | **2-stage cascade (FP -67%)** | **N/A** |

*V1 F1 of 0.996 was artificial due to Reuters attribution bias.

## How to add a new model version

1. Train using the notebook in `notebooks/training/`
2. Track experiment: `from monitoring.mlflow_tracker import track_experiment`
3. Save model to `models/` with version suffix
4. Update this registry
5. Run `make test` to verify nothing breaks
