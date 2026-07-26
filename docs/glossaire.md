# Glossaire Technique — ThumaCheck

Document de reference pour le vocabulaire technique utilise dans le projet ThumaCheck.
Les definitions sont volontairement concises et contextualisees au projet.

---

**AI Act** — Reglement europeen (UE 2024/1689) encadrant les systemes d'intelligence artificielle selon leur niveau de risque. ThumaCheck est classe en risque limite et applique les obligations de transparence (art. 13), notamment via la Model Card et l'explicabilite XAI.

**AIPD (Analyse d'Impact relative a la Protection des Donnees)** — Evaluation formalisee des risques qu'un traitement de donnees personnelles fait peser sur les droits des personnes. Requise par le RGPD pour les traitements a risque eleve ; documentee dans ThumaCheck bien que le risque soit juge limite.

**API (Application Programming Interface)** — Interface permettant a des logiciels de communiquer entre eux. ThumaCheck expose une API REST via FastAPI avec les endpoints `/predict`, `/explain`, `/health` et `/energy`.

**AT Protocol** — Protocole decentralise sous-jacent au reseau social Bluesky. Le collecteur ThumaCheck l'utilise pour ingerer les posts en temps reel.

**Batch** — Mode de traitement par lots, ou les donnees sont accumulees puis traitees en une seule passe, par opposition au traitement unitaire. Utilise dans le prototype Kafka pour l'inference groupee.

**Biais** — Distorsion systematique dans les donnees ou le modele qui conduit a des predictions inegalement fiables selon les sous-populations ou les contextes. ThumaCheck documente un biais lexical (sensibilite aux marqueurs de surface) et un biais temporel (drift sur evenements recents).

**Bluesky** — Reseau social decentralise base sur l'AT Protocol. Source de donnees principale de ThumaCheck, avec plus de 537 000 posts collectes depuis decembre 2025.

**CamemBERT** — Modele Transformer pre-entraine sur du francais, derive de RoBERTa. Fine-tune dans ThumaCheck pour la detection de desinformation sur textes ultra-courts (F1 = 0.957).

**CI/CD (Continuous Integration / Continuous Deployment)** — Pratique d'automatisation des tests et du deploiement a chaque modification du code. ThumaCheck utilise pytest (537 tests) et une couverture de 80 %.

**Classification binaire** — Tache de machine learning consistant a affecter chaque observation a l'une de deux classes. Dans ThumaCheck : fiable ou suspect.

**CodeCarbon** — Bibliotheque Python mesurant les emissions de CO2 des calculs informatiques. Utilisee dans ThumaCheck pour le suivi de l'empreinte carbone a l'entrainement et en inference temps reel via l'endpoint `/energy`.

**CORS (Cross-Origin Resource Sharing)** — Mecanisme HTTP permettant a un navigateur d'acceder a des ressources hebergees sur un domaine different. Configure dans l'API FastAPI de ThumaCheck.

**Cross-validation (Validation croisee)** — Technique d'evaluation d'un modele en le testant sur plusieurs sous-ensembles distincts des donnees. ThumaCheck utilise une validation croisee a 5 folds pour mesurer le F1 du modele V5 (0.913).

**Dashboard** — Interface de visualisation interactive. ThumaCheck propose un dashboard Streamlit de 5 pages : Dashboard, Analyse IA, Explorateur, Performance, A propos.

**Desinformation** — Diffusion deliberee d'informations fausses ou trompeuses. ThumaCheck detecte les signaux de forme (style, lexique) et non la veracite factuelle des contenus.

**Docker** — Plateforme de conteneurisation permettant d'executer des applications dans des environnements isoles et reproductibles. ThumaCheck utilise Docker Compose pour orchestrer MongoDB, le collecteur, Jupyter et Streamlit.

**Drift (derive)** — Evolution dans le temps des caracteristiques des donnees ou des performances d'un modele. ThumaCheck documente un drift temporel sur les evenements post-2024.

**Embedding** — Representation numerique d'un texte sous forme de vecteur dense dans un espace de grande dimension. Les Transformers (CamemBERT, RoBERTa) produisent des embeddings contextuels utilises par le meta-learner.

**Emotion (7 classes)** — Classification emotionnelle des textes selon 7 categories : colere, degout, joie, neutre, peur, surprise, tristesse. Realisee dans ThumaCheck par un reseau MLP PyTorch.

**Explicabilite (XAI)** — Capacite a rendre comprehensibles les decisions d'un modele d'IA. ThumaCheck deploie 8 mecanismes d'explicabilite : SHAP, attention, Integrated Gradients, decomposition du meta-learner, et validation de fidelite (AOPC).

**F1-score** — Moyenne harmonique de la precision et du rappel, mesurant la performance d'un classifieur. Metrique de reference de ThumaCheck ; F1 macro sur le gold test set pour comparer les versions V5 a V9.

**Fairness (equite)** — Propriete d'un modele qui produit des predictions equitables independamment des caracteristiques sensibles des individus. ThumaCheck inclut un audit de fairness documente.

**FastAPI** — Framework Python pour la creation d'API REST performantes et auto-documentees. Utilise pour l'API ThumaCheck avec rate limiting (60 req/min).

**Feature engineering** — Processus de creation manuelle de variables d'entree pour un modele. ThumaCheck utilise 17 features linguistiques (dont emojis) et 28 features stylistiques dans ses modeles V5 et V6.

**Few-shot** — Technique d'apprentissage ou un modele realise une tache a partir de quelques exemples seulement, fournis dans le prompt. Distingue du fine-tuning qui modifie les poids du modele.

**Fine-tuning** — Reapprentissage partiel d'un modele pre-entraine sur des donnees specifiques a une tache. CamemBERT et RoBERTa sont fine-tunes dans ThumaCheck sur des corpus de fake news.

**Gold test set** — Jeu de test de reference annote manuellement par des experts. ThumaCheck utilise 500 posts Bluesky annotes par 2 annotateurs independants (kappa de Cohen = 0.498).

**Green IT** — Demarche visant a reduire l'empreinte environnementale du numerique. ThumaCheck privilegie des modeles frugaux (LogReg en production) et mesure ses emissions via CodeCarbon (6.14 g CO2 total).

**Integrated Gradients** — Methode d'attribution qui mesure la contribution de chaque feature d'entree a la prediction d'un modele en integrant les gradients le long d'un chemin de reference. Implementee via Captum dans ThumaCheck pour CamemBERT.

**Kafka** — Plateforme de streaming evenementiel distribue. ThumaCheck inclut un prototype Kafka consumer pour le traitement asynchrone par batch des predictions.

**KPI (Key Performance Indicator)** — Indicateur cle de performance permettant de mesurer l'atteinte des objectifs. Les KPI de ThumaCheck incluent le F1-score, la latence, le taux de faux positifs et l'empreinte CO2.

**Latence** — Delai entre l'envoi d'une requete et la reception de la reponse. ThumaCheck atteint 1.5 ms par texte en inference (728 textes/sec).

**LLM (Large Language Model)** — Modele de langage de grande taille entraine sur de vastes corpus textuels. CamemBERT et RoBERTa sont des LLM specialises ; ThumaCheck les utilise en complement de modeles classiques plus frugaux.

**LogisticRegression (Regression logistique)** — Modele de classification lineaire estimant la probabilite d'appartenance a une classe. Modele principal de ThumaCheck (V5, meta-learner V8) pour sa rapidite, sa frugalite et son interpretabilite.

**Meta-learner** — Modele de second niveau qui combine les predictions de plusieurs modeles de base. Dans ThumaCheck V8, un LogReg meta-learner fusionne les sorties de V5 (TF-IDF), V6 (style) et CamemBERT.

**Model Card** — Document standardise (Mitchell et al., 2019) decrivant un modele de machine learning : architecture, performances, limites, biais et considerations ethiques. Disponible dans `docs/12_model_card.md`.

**MongoDB** — Base de donnees NoSQL orientee documents. Stocke les posts Bluesky collectes et les resultats d'analyse dans ThumaCheck.

**Mutation testing** — Technique de test logiciel consistant a introduire des modifications (mutants) dans le code pour verifier que les tests les detectent. ThumaCheck atteint un taux de destruction de mutants de 80.3 %.

**NLP (Natural Language Processing)** — Sous-domaine de l'IA traitant du langage humain. ThumaCheck est un pipeline NLP bilingue de detection de desinformation.

**Notebook** — Document interactif combinant code, visualisations et commentaires. ThumaCheck comprend 27 notebooks Jupyter documentant l'exploration, l'entrainement et l'analyse.

**Open-weight** — Qualifie un modele dont les poids sont publiquement accessibles, permettant l'inspection et la reproductibilite. CamemBERT et RoBERTa sont des modeles open-weight.

**Overfitting (surapprentissage)** — Situation ou un modele apprend les particularites du jeu d'entrainement au detriment de sa capacite de generalisation. Controle dans ThumaCheck par la cross-validation et le gold test set independant.

**Pipeline** — Enchainement automatise d'etapes de traitement. ThumaCheck V9 est un pipeline cascade en 2 etapes : filtre fait/opinion puis ensemble hybride multi-modeles.

**Precision** — Proportion de predictions positives qui sont effectivement correctes. Complementaire du rappel ; les deux sont synthetises par le F1-score.

**Prompt engineering** — Conception et optimisation des instructions textuelles fournies a un LLM pour guider sa reponse. Utilise dans les benchmarks LLM du projet (zero-shot, few-shot).

**Recall (Rappel)** — Proportion des positifs reels qui sont correctement identifies par le modele. Un rappel eleve minimise les faux negatifs (contenus suspects non detectes).

**RGPD (Reglement General sur la Protection des Donnees)** — Reglement europeen encadrant le traitement des donnees personnelles. ThumaCheck invoque l'art. 6.1.f (interet legitime) pour le traitement des posts publics Bluesky. Conformite detaillee dans `docs/02_conformite_RGPD_AI_Act.md`.

**RoBERTa** — Variante optimisee de BERT (Liu et al., 2019), modele Transformer pre-entraine pour l'anglais. Fine-tune dans ThumaCheck pour les textes EN ultra-courts (F1 = 0.874).

**ROC-AUC** — Aire sous la courbe ROC, mesurant la capacite d'un classifieur a distinguer les classes a travers tous les seuils de decision. Metrique complementaire au F1-score.

**SHAP (SHapley Additive exPlanations)** — Methode d'explicabilite attribuant a chaque feature sa contribution marginale a une prediction, fondee sur les valeurs de Shapley. Utilisee dans ThumaCheck en global (beeswarm V6), par instance, et sur les emotions (KernelExplainer MLP).

**Souverainete numerique** — Capacite d'un Etat ou d'une organisation a maitriser ses systemes et donnees numeriques. ThumaCheck y contribue par le choix de modeles open-weight et l'hebergement local des donnees.

**Streamlit** — Framework Python pour la creation de dashboards web interactifs. Utilise pour les 5 pages du dashboard ThumaCheck.

**TF-IDF (Term Frequency — Inverse Document Frequency)** — Methode de ponderation statistique mesurant l'importance d'un mot dans un document par rapport a un corpus. Feature principale du modele V5 de ThumaCheck.

**Threshold (seuil de decision)** — Valeur a partir de laquelle une probabilite predite est convertie en classe. Le seuil de ThumaCheck V5 est fixe a 0.44, optimise pour equilibrer precision et rappel.

**Tokenization** — Decoupage d'un texte en unites elementaires (tokens : mots, sous-mots ou caracteres) exploitables par un modele. Les Transformers utilisent des tokenizers en sous-mots (BPE, SentencePiece).

**Transformer** — Architecture de reseau de neurones basee sur le mecanisme d'attention (Vaswani et al., 2017). Fondement de CamemBERT et RoBERTa utilises dans ThumaCheck.

**WCAG (Web Content Accessibility Guidelines)** — Recommandations internationales pour l'accessibilite des contenus web. Le dashboard ThumaCheck vise la conformite WCAG pour les contrastes et la lisibilite.

**Zero-shot** — Capacite d'un modele a realiser une tache sans aucun exemple prealable, en s'appuyant uniquement sur ses connaissances pre-entrainees. Evalue dans le benchmark LLM de ThumaCheck.
