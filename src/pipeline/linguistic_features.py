"""
ThumaCheck — Linguistic Feature Extractor
==========================================

45+ linguistic and stylistic features for fake news detection.
Extracts signals like capitalization ratio, punctuation density,
sensationalism vocabulary, and readability metrics.

Auteur : Niamato Consulting (pour Thumalien)
"""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LinguisticFeatureExtractor:
    """
    Extrait des signaux linguistiques indicatifs de désinformation.

    Ces features capturent des patterns structurels (ponctuation,
    majuscules, sensationnalisme) indépendants du contenu lexical.
    Complémentaires au TF-IDF.
    """

    SENSATIONALIST_EN = frozenset(
        {
            "breaking",
            "shocking",
            "bombshell",
            "exposed",
            "secret",
            "conspiracy",
            "banned",
            "censored",
            "hoax",
            "alert",
            "exclusive",
            "unbelievable",
            "cover-up",
            "coverup",
            "wake up",
            "they dont want",
            "mainstream media",
            "deep state",
            "big pharma",
            "must watch",
            "must read",
            "you wont believe",
            "what they hide",
            "truth about",
            "exposed the truth",
            "share before deleted",
            "deleted soon",
            "viral",
        }
    )

    SENSATIONALIST_FR = frozenset(
        {
            # Termes originaux
            "scandale",
            "exclusif",
            "choc",
            "censuré",
            "complot",
            "mensonge",
            "urgent",
            "alerte",
            "incroyable",
            "on vous cache",
            "manipulé",
            "propagande",
            "dictature",
            "résistance",
            "big pharma",
            "nouvel ordre mondial",
            "great reset",
            # Conspiration
            "cabale",
            "complotisme",
            "dissimulé",
            "falsifié",
            "oligarchie",
            "mondialisme",
            "lobbies",
            "collusion",
            "corruption",
            "état profond",
            "fraude électorale",
            "illuminati",
            # Sensationnalisme
            "hallucinant",
            "stupéfiant",
            "révélation",
            "bombe",
            "explosif",
            "terrifiant",
            "catastrophique",
            "apocalyptique",
            "effrayant",
            # Manipulation émotionnelle
            "réveillons-nous",
            "ouvrez les yeux",
            "on nous ment",
            "vérité cachée",
            "faites tourner",
            "partagez avant censure",
            "partagez avant",
            "réveillez-vous",
            "on nous cache",
            "on vous ment",
            "faites vos propres recherches",
            "avant censure",
            "partagez massivement",
            "info censurée",
            # Social media FR (ajout V4)
            "à partager",
            "diffusez",
            "la preuve",
            "preuve en image",
            "regardez cette vidéo",
            "vidéo censurée",
            "témoignage choc",
            "enfin la vérité",
            "ce que les médias cachent",
            "les médias mentent",
            "info interdite",
            "plandémie",
            "génocide",
            "empoisonnement",
            "puces",
            "micro-puces",
            "nanoparticules",
            "graphène",
            "pass sanitaire",
            "soumission",
            "résistez",
            "insurrection",
            "traîtres",
            "vendu",
            "vendus",
            "marionnettes",
        }
    )

    FEATURE_NAMES = [
        "word_count",
        "caps_ratio",
        "exclamation_count",
        "question_count",
        "punct_density",
        "avg_word_length",
        "sensationalism_score",
        "has_url",
        "numeric_density",
        "lexical_diversity",
        "sentence_count",
        "avg_sentence_length",
        # V4 : features texte court
        "all_caps_words_ratio",
        "interpellation_score",
        "is_short_text",
        # V5 : features emoji
        "emoji_count",
        "emoji_sentiment",
    ]

    # Regex pour détecter les emojis (plages Unicode communes)
    _EMOJI_PATTERN = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
        r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]"
    )
    # Emojis positifs (smileys, coeurs, pouces levés)
    _EMOJI_POSITIVE = re.compile(
        r"[\U0001F600-\U0001F60F\U0001F617-\U0001F61C\U0001F31F"
        r"\U0001F44D\U0001F44F\U0001F495-\U0001F49F\U00002764"
        r"\U0001F970-\U0001F975\U0001F60D\U0001F618\U0001F60A"
        r"\U0001F389\U0001F38A\U0001F381\U00002728]"
    )
    # Emojis négatifs (crâne, colère, peur)
    _EMOJI_NEGATIVE = re.compile(
        r"[\U0001F620-\U0001F62D\U0001F4A2\U0001F480\U00002620"
        r"\U0001F621\U0001F624\U0001F616\U0001F628-\U0001F630"
        r"\U0001F47F\U0001F44E\U0001F5E1]"
    )

    # Patterns d'interpellation directe (manipulation sociale FR+EN)
    INTERPELLATION_PATTERNS_FR = [
        r"\b(réveillez[ -]vous|réveillons[ -]nous)\b",
        r"\b(ouvrez les yeux|ouvrons les yeux)\b",
        r"\b(faites tourner|partagez|diffusez|rt svp)\b",
        r"\b(on nous ment|on vous ment|ils nous mentent)\b",
        r"\b(ne soyez pas dupes?|ne soyez pas naï[fv]s?)\b",
        r"\b(dites non|boycott|refusez)\b",
        r"\b(attention danger|alerte rouge|alerte info)\b",
    ]
    INTERPELLATION_PATTERNS_EN = [
        r"\b(wake up|open your eyes)\b",
        r"\b(share before|retweet|spread the word)\b",
        r"\b(they are lying|they lied|dont be fooled)\b",
        r"\b(say no|boycott|fight back|resist)\b",
        r"\b(red alert|warning|danger)\b",
    ]

    @classmethod
    def extract(cls, texts: pd.Series) -> np.ndarray:
        """Retourne une matrice (n_samples, n_features) de features linguistiques."""
        results = np.zeros((len(texts), len(cls.FEATURE_NAMES)), dtype=np.float64)

        for i, text in enumerate(texts):
            text = str(text)
            words = text.split()
            n_words = len(words) if words else 1
            n_chars = len(text) if text else 1

            # Longueur
            results[i, 0] = n_words

            # Ratio majuscules (sur le texte original avant lower)
            alpha_chars = sum(c.isalpha() for c in text)
            results[i, 1] = sum(c.isupper() for c in text) / alpha_chars if alpha_chars > 0 else 0

            # Ponctuation émotionnelle
            results[i, 2] = text.count("!")
            results[i, 3] = text.count("?")
            results[i, 4] = sum(c in "!?.,;:…" for c in text) / n_chars

            # Longueur moyenne des mots
            results[i, 5] = np.mean([len(w) for w in words]) if words else 0

            # Sensationnalisme (word-boundary aware for both single & multi-word)
            text_lower = text.lower()
            score = 0
            for w in cls.SENSATIONALIST_EN | cls.SENSATIONALIST_FR:
                # Use regex word boundaries to avoid partial matches
                # and correctly match multi-word expressions
                if re.search(r"(?:^|\b|\s)" + re.escape(w) + r"(?:\b|\s|$)", text_lower):
                    score += 1
            results[i, 6] = score

            # Présence d'URL
            results[i, 7] = 1.0 if re.search(r"http|www\.", text) else 0.0

            # Densité numérique
            results[i, 8] = sum(c.isdigit() for c in text) / n_chars

            # Diversité lexicale (TTR)
            results[i, 9] = len(set(words)) / n_words if n_words > 0 else 0

            # Phrases
            sentences = re.split(r"[.!?]+", text)
            sentences = [s for s in sentences if s.strip()]
            results[i, 10] = len(sentences)
            results[i, 11] = n_words / len(sentences) if sentences else n_words

            # V4 : Ratio de mots entièrement en majuscules (signal fort pour posts courts)
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            results[i, 12] = caps_words / n_words if n_words > 0 else 0

            # V4 : Score d'interpellation (manipulation sociale directe)
            interp_score = 0
            for pat in cls.INTERPELLATION_PATTERNS_FR + cls.INTERPELLATION_PATTERNS_EN:
                if re.search(pat, text_lower):
                    interp_score += 1
            results[i, 13] = interp_score

            # V4 : Indicateur texte court (< 20 mots) — permet au modèle d'apprendre
            # des patterns spécifiques aux textes courts
            results[i, 14] = 1.0 if n_words < 20 else 0.0

            # V5 : Emoji count
            emojis = cls._EMOJI_PATTERN.findall(text)
            n_emojis = len(emojis)
            results[i, 15] = n_emojis

            # V5 : Emoji sentiment (positive - negative) / total
            if n_emojis > 0:
                n_pos = len(cls._EMOJI_POSITIVE.findall(text))
                n_neg = len(cls._EMOJI_NEGATIVE.findall(text))
                results[i, 16] = (n_pos - n_neg) / n_emojis
            else:
                results[i, 16] = 0.0

        return results
