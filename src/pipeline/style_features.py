"""Extracteur de features stylistiques V6 (topic-agnostic, 28 features)."""

import re

import numpy as np

from pipeline.expert_detector import LinguisticFeatureExtractor


class StyleFeatureExtractorV6:
    """Extracteur de features stylistiques -- topic-agnostic (28 features)."""

    SENSATIONALIST_EN = LinguisticFeatureExtractor.SENSATIONALIST_EN
    SENSATIONALIST_FR = LinguisticFeatureExtractor.SENSATIONALIST_FR

    CALL_TO_ACTION_FR = [
        r'\b(partagez|diffusez|faites tourner|rt svp|a partager)\b',
        r'\b(likez|abonnez|suivez|inscrivez)\b',
        r'\b(signez la petition|mobilisons|reagissez)\b',
        r'\b(avant (la )?censure|avant suppression|avant qu.?ils? suppriment)\b',
    ]
    CALL_TO_ACTION_EN = [
        r'\b(share|retweet|spread the word|pass it on)\b',
        r'\b(subscribe|follow|like|sign the petition)\b',
        r'\b(before (they|it gets?) deleted?|before censored)\b',
        r'\b(act now|do something|fight back|resist)\b',
    ]
    HEDGING_FR = [
        r'\b(selon|d.?apr[e\u00e8]s|il para[i\u00ee]t que|il semblerait)\b',
        r'\b(certains disent|on dit que|des sources)\b',
        r'\b(apparemment|soi-?disant|pr[e\u00e9]tendument)\b',
    ]
    HEDGING_EN = [
        r'\b(allegedly|reportedly|according to|sources say)\b',
        r'\b(it is said|some say|rumor has it|unconfirmed)\b',
        r'\b(supposedly|purportedly|claimed)\b',
    ]
    AUTHORITY_CLAIM_FR = [
        r'\b(un (m[e\u00e9]decin|scientifique|expert|chercheur|professeur) (affirme|confirme|r[e\u00e9]v[e\u00e8]le))\b',
        r'\b(etude (prouve|montre|confirme))\b',
        r'\b(c.?est prouv[e\u00e9]|la science dit|les chiffres parlent)\b',
    ]
    AUTHORITY_CLAIM_EN = [
        r'\b(doctor|scientist|expert|professor|researcher) (says|confirms|reveals)\b',
        r'\b(study (proves|shows|confirms))\b',
        r'\b(science says|the data shows|proven)\b',
    ]
    SOURCE_CITATION_PATTERNS = [
        r'\b(reuters|afp|ap news|associated press)\b',
        r'\b(selon (le |la |l.?)?[A-Z])',
        r'\b(source[s]?\s*:)',
        r'\b(d.?apr[e\u00e8]s (le |la |l.?)?[A-Z])',
        r'\b(published in|peer.?reviewed|journal)\b',
        r'\b(lib[e\u00e9]ration|le monde|figaro|bbc|cnn|nyt|washington post)\b',
    ]

    FEATURE_NAMES = [
        'word_count', 'sentence_count', 'avg_sentence_length',
        'avg_word_length', 'is_short_text', 'paragraph_count',
        'exclamation_count', 'question_count', 'punct_density',
        'ellipsis_count', 'repeated_punct_ratio', 'emoji_count',
        'caps_ratio', 'all_caps_words_ratio', 'caps_lock_words_count',
        'sensationalism_score', 'interpellation_score',
        'call_to_action_score', 'hedging_score', 'authority_claim_score',
        'has_url', 'has_source_citation', 'numeric_density',
        'quote_count', 'named_entity_density',
        'lexical_diversity', 'repeated_char_ratio', 'spelling_anomaly_score',
    ]

    FEATURE_LABELS_FR = {
        'word_count': 'Nombre de mots', 'sentence_count': 'Nombre de phrases',
        'avg_sentence_length': 'Longueur moy. phrases', 'avg_word_length': 'Longueur moy. mots',
        'is_short_text': 'Texte court (<20 mots)', 'paragraph_count': 'Nombre de paragraphes',
        'exclamation_count': "Points d'exclamation", 'question_count': "Points d'interrogation",
        'punct_density': 'Densit\u00e9 ponctuation', 'ellipsis_count': 'Points de suspension',
        'repeated_punct_ratio': 'Ponctuation r\u00e9p\u00e9t\u00e9e', 'emoji_count': 'Emojis',
        'caps_ratio': 'Ratio majuscules', 'all_caps_words_ratio': 'Mots tout en MAJUSCULES',
        'caps_lock_words_count': 'Nb mots CAPS LOCK', 'sensationalism_score': 'Sensationnalisme',
        'interpellation_score': 'Interpellation', 'call_to_action_score': "Appel \u00e0 l'action",
        'hedging_score': 'Langage \u00e9vasif', 'authority_claim_score': "Appel \u00e0 l'autorit\u00e9",
        'has_url': 'Pr\u00e9sence URL', 'has_source_citation': 'Citation de source',
        'numeric_density': 'Densit\u00e9 num\u00e9rique', 'quote_count': 'Citations/guillemets',
        'named_entity_density': 'Densit\u00e9 entit\u00e9s nomm\u00e9es', 'lexical_diversity': 'Diversit\u00e9 lexicale (TTR)',
        'repeated_char_ratio': 'Caract\u00e8res r\u00e9p\u00e9t\u00e9s', 'spelling_anomaly_score': 'Anomalies orthographiques',
        'emo_anger': '\u00c9motion : Col\u00e8re', 'emo_disgust': '\u00c9motion : D\u00e9go\u00fbt',
        'emo_joy': '\u00c9motion : Joie', 'emo_neutral': '\u00c9motion : Neutre',
        'emo_fear': '\u00c9motion : Peur', 'emo_surprise': '\u00c9motion : Surprise',
        'emo_sadness': '\u00c9motion : Tristesse',
    }

    _LFE = LinguisticFeatureExtractor

    @classmethod
    def extract(cls, texts) -> np.ndarray:
        """Extraire 28 features stylistiques (sans emotions)."""
        n = len(texts)
        results = np.zeros((n, len(cls.FEATURE_NAMES)), dtype=np.float64)
        for i, text in enumerate(texts):
            text = str(text)
            text_lower = text.lower()
            words = text.split()
            n_words = len(words) if words else 1
            n_chars = len(text) if text else 1
            alpha_chars = sum(c.isalpha() for c in text)

            results[i, 0] = n_words
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            n_sentences = len(sentences) if sentences else 1
            results[i, 1] = n_sentences
            results[i, 2] = n_words / n_sentences
            results[i, 3] = np.mean([len(w) for w in words]) if words else 0
            results[i, 4] = 1.0 if n_words < 20 else 0.0
            paragraphs = [p for p in text.split('\n') if p.strip()]
            results[i, 5] = len(paragraphs)
            results[i, 6] = text.count('!')
            results[i, 7] = text.count('?')
            results[i, 8] = sum(c in '!?.,;:\u2026' for c in text) / n_chars
            results[i, 9] = text.count('...') + text.count('\u2026')
            repeated = len(re.findall(r'([!?.])\1{1,}', text))
            results[i, 10] = repeated / n_chars if n_chars > 0 else 0
            emoji_count = len(re.findall(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
                r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
                r'\U00002702-\U000027B0\U0001F900-\U0001F9FF'
                r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
                r'\U00002600-\U000026FF]', text))
            results[i, 11] = emoji_count
            results[i, 12] = sum(c.isupper() for c in text) / alpha_chars if alpha_chars > 0 else 0
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            results[i, 13] = caps_words / n_words if n_words > 0 else 0
            results[i, 14] = caps_words
            sens_score = 0
            for w in cls.SENSATIONALIST_EN | cls.SENSATIONALIST_FR:
                if re.search(r'(?:^|\b|\s)' + re.escape(w) + r'(?:\b|\s|$)', text_lower):
                    sens_score += 1
            results[i, 15] = sens_score
            interp_score = 0
            for pat in (cls._LFE.INTERPELLATION_PATTERNS_FR + cls._LFE.INTERPELLATION_PATTERNS_EN):
                if re.search(pat, text_lower):
                    interp_score += 1
            results[i, 16] = interp_score
            cta_score = 0
            for pat in cls.CALL_TO_ACTION_FR + cls.CALL_TO_ACTION_EN:
                if re.search(pat, text_lower):
                    cta_score += 1
            results[i, 17] = cta_score
            hedge_score = 0
            for pat in cls.HEDGING_FR + cls.HEDGING_EN:
                if re.search(pat, text_lower):
                    hedge_score += 1
            results[i, 18] = hedge_score
            auth_score = 0
            for pat in cls.AUTHORITY_CLAIM_FR + cls.AUTHORITY_CLAIM_EN:
                if re.search(pat, text_lower):
                    auth_score += 1
            results[i, 19] = auth_score
            results[i, 20] = 1.0 if re.search(r'http|www\.', text) else 0.0
            source_score = 0
            for pat in cls.SOURCE_CITATION_PATTERNS:
                if re.search(pat, text_lower):
                    source_score += 1
            results[i, 21] = source_score
            results[i, 22] = sum(c.isdigit() for c in text) / n_chars
            results[i, 23] = text.count('"') + text.count('\u201c') + text.count('\u00ab')
            if len(words) > 1:
                ne_count = sum(1 for j, w in enumerate(words[1:], 1) if w[0].isupper() and w.isalpha())
                results[i, 24] = ne_count / n_words
            else:
                results[i, 24] = 0
            words_lower = [w.lower() for w in words]
            results[i, 25] = len(set(words_lower)) / n_words if n_words > 0 else 0
            repeated_chars = len(re.findall(r'(.)\1{2,}', text_lower))
            results[i, 26] = repeated_chars / n_words if n_words > 0 else 0
            common_short = {'je', 'tu', 'il', 'on', 'le', 'la', 'de', 'a', 'i', '\u00e0',
                            'y', 'or', 'et', 'en', 'du', 'un', 'au', 'ne', 'se', 'me',
                            'te', 'ce', 'ma', 'sa', 'ta', 'is', 'am', 'an', 'as', 'at',
                            'be', 'by', 'do', 'go', 'he', 'if', 'in', 'it', 'me', 'my',
                            'no', 'of', 'on', 'or', 'so', 'to', 'up', 'us', 'we'}
            anomalous = sum(1 for w in words_lower if 1 <= len(w) <= 2 and w not in common_short)
            results[i, 27] = anomalous / n_words if n_words > 0 else 0
        return results
