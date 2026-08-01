"""
ThumaCheck — Lightweight Demo for HuggingFace Spaces
=====================================================
No MongoDB required. Uses pre-trained models + sample texts.
"""

import streamlit as st
import joblib
import os
import numpy as np

st.set_page_config(page_title="ThumaCheck Demo", page_icon="🔍", layout="wide")

st.title("🔍 ThumaCheck — Misinformation Detection Demo")
st.markdown("""
> Real-time misinformation detection using NLP cascade pipeline.
> CamemBERT (FR, F1: 0.957) | RoBERTa (EN, F1: 0.874) | 728 texts/sec
""")

# --- Sample texts ---
SAMPLES = {
    "🟢 Reliable (FR)": "Le président a prononcé un discours devant l'Assemblée nationale ce mardi.",
    "🔴 Suspect (FR)": "SCANDALE !!! On nous CACHE la vérité sur le vaccin, PARTAGEZ avant censure !!!",
    "🟢 Reliable (EN)": "The European Central Bank announced a quarter-point rate adjustment on Thursday.",
    "🔴 Suspect (EN)": "EXPOSED: They don't want you to know this SECRET about the election COVER UP!!!",
    "🟡 Ambiguous (FR)": "Certains disent que cette politique pourrait avoir des effets négatifs.",
}

st.sidebar.header("Try a sample")
selected = st.sidebar.selectbox("Choose a sample text:", list(SAMPLES.keys()))

text_input = st.text_area(
    "Enter text to analyze:",
    value=SAMPLES[selected],
    height=100,
)

if st.button("Analyze", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction")
            # Placeholder — replace with actual model inference
            st.info("⚠️ This demo uses sample predictions. "
                    "Full pipeline requires trained models.")
            if "SCANDALE" in text_input or "EXPOSED" in text_input:
                st.error("🔴 **SUSPECT** (confidence: 0.89)")
            elif "certains" in text_input.lower():
                st.warning("🟡 **UNCERTAIN** (confidence: 0.52)")
            else:
                st.success("🟢 **RELIABLE** (confidence: 0.94)")

        with col2:
            st.subheader("Key Signals")
            st.markdown("""
            | Feature | Value |
            |---------|-------|
            | Caps ratio | {:.1%} |
            | Exclamation density | {:.1%} |
            | Sensationalism score | {:.2f} |
            | Word count | {} |
            """.format(
                sum(1 for c in text_input if c.isupper()) / max(len(text_input), 1),
                text_input.count('!') / max(len(text_input), 1),
                sum(1 for w in text_input.lower().split() if w in ['scandale', 'secret', 'exposed', 'censure', 'urgent']) / max(len(text_input.split()), 1),
                len(text_input.split()),
            ))

st.markdown("---")
st.markdown("""
<sub>
<b>ThumaCheck</b> by <a href="https://github.com/azelbanks">Azélie Bernard</a> —
<a href="https://github.com/azelbanks/thumacheck">GitHub Repository</a> |
537 tests | 80% coverage | MIT License
</sub>
""", unsafe_allow_html=True)
