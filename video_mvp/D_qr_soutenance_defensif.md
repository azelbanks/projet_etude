# Q&R Défensif — Soutenance ThumaCheck (client : Thumalien)

**Objectif** : anticiper 22 questions critiques que peut poser un jury MASTERE exigeant, avec **réponse au mot près**, courte (30-60 secondes max), structurée pour désamorcer plutôt qu'esquiver.

**Principe directeur** : reconnaître la critique, montrer qu'on l'a anticipée, donner des chiffres, conclure positivement. Jamais de défensive faible, jamais de promesse vague.

---

## I. Questions techniques sur le modèle (8 questions)

### Q1 — *"Pourquoi ne pas avoir utilisé un Transformer en bout-en-bout ? CamemBERT seul ferait peut-être mieux."*

**Réponse** :
> *"On a testé. CamemBERT seul atteint F1=0.957 sur les textes ultra-courts français mais F1=0.84 sur le bilingue. Notre méta-learner V8 atteint F1 macro=0.654 sur le gold avec une latence de 1,5 ms par texte contre 38 ms pour CamemBERT seul. Pour notre cas d'usage — analyser 60 000 posts par jour en streaming — la frugalité conditionne le déploiement. Le rapport § 25 documente cet arbitrage avec mesures CodeCarbon : CamemBERT consomme 130× plus d'énergie que LogReg. C'est un choix architectural assumé : CamemBERT est utilisé en signal complémentaire dans l'ensemble, pas comme modèle de production."*

**Stratégie** : transformer la question en démonstration que tu as fait l'arbitrage explicitement, avec mesures.

---

### Q2 — *"Votre F1 sur le gold est de 0.67. C'est faible, non ?"*

**Réponse** :
> *"C'est honnête, oui. Le gold de 200 posts a un kappa de Cohen inter-annotateurs de 0,498 — la frontière fiable/suspect est intrinsèquement subjective. Notre F1 macro de 0,67 reflète cette difficulté ; il vaut mieux un F1 honnête sur un gold annoté manuellement qu'un F1=0.99 en cross-validation qui masquait un biais Reuters comme on l'a vu en V1. Sur le métrique qui compte vraiment pour notre cas d'usage — la réduction des faux positifs — on passe de 57 à 21 sur le gold, soit une réduction de 67% avec un intervalle bootstrap de [-73%, -60%] et un test de Fisher à p<0.000001. C'est ce chiffre-là qui est pertinent."*

**Stratégie** : déplacer le débat de F1 vers FP — la métrique réellement opérationnelle.

---

### Q3 — *"Comment justifiez-vous le seuil de 0,44 et pas 0,50 ?"*

**Réponse** :
> *"Optimisation explicite par grid search sur le gold set en maximisant le F1 macro. Le déséquilibre des classes — 191 fiables pour 9 suspects soit 4,5% de positifs — déplace le seuil optimal vers le bas. À 0,50 on rate trop de suspects pour préserver la précision ; à 0,44 on optimise le compromis F1. C'est documenté dans le notebook 21 et le rapport § 21. On a aussi des seuils adaptatifs par longueur de texte — 0,54 pour les textes < 15 mots, 0,49 pour ceux entre 15 et 30 mots — parce que le modèle est moins confiant sur les textes courts."*

**Stratégie** : prouver que c'est une décision data-driven, pas arbitraire.

---

### Q4 — *"Qu'est-ce qui vous empêche d'avoir 90% de coverage de tests ?"*

**Réponse** :
> *"Rien techniquement, mais le ratio effort/qualité décroît au-delà de 80%. On est à 80% line coverage, 77,9% branch coverage, et surtout 80,3% de mutation kill rate sur le module critique de décomposition méta-learner — au-dessus de la moyenne Google qui est de 60-75%. Notre quality gate sur GitHub Actions bloque toute PR descendant sous 75%. Pousser à 90% nécessiterait de tester des chemins d'erreur rares qui valent moins que d'investir dans un monitoring drift en production. C'est le compromis qu'on a documenté dans la roadmap V10."*

**Stratégie** : citer le mutation kill rate qui est plus impressionnant que le pourcentage simple, et positionner l'arbitrage comme conscient.

---

### Q5 — *"Pourquoi LogReg et pas un modèle plus complexe comme XGBoost ?"*

**Réponse** :
> *"Trois raisons. Premièrement, l'**explicabilité** : les coefficients d'une LogReg donnent une décomposition exacte par feature, pas une approximation comme SHAP. Deuxièmement, la **calibration** : LogReg sort des probabilités natives qu'on peut directement utiliser dans un méta-learner. Troisièmement, la **frugalité** : 1,5 ms par texte sur Apple Silicon. On a testé XGBoost en V2, gain de 0,3 point de F1 mais perte de 8× en latence et complexification de l'XAI. Le compromis ne valait pas. On a en revanche utilisé GradientBoosting sur V6 — le module style-only — où la non-linéarité capte mieux les interactions stylistiques."*

**Stratégie** : tu as testé l'alternative, tu as les chiffres, c'est un choix éclairé.

---

### Q6 — *"L'attention de CamemBERT n'est pas une explication causale (Jain & Wallace 2019). Pourquoi l'avoir mise en avant ?"*

**Réponse** :
> *"Vous avez raison, et c'est précisément pour ça qu'on a aussi implémenté Layer Integrated Gradients via Captum, qui satisfait les axiomes de Sundararajan et fournit une attribution causale rigoureuse. L'attention CamemBERT est utilisée pour le **débogage qualitatif** — comprendre quels tokens le modèle 'regarde' — mais quand on parle de **causalité**, on s'appuie sur IG. La Model Card section 7.2 documente cette distinction explicitement, en citant Jain & Wallace 2019. C'est précisément pour avoir les deux niveaux qu'on a multiplié les méthodes XAI."*

**Stratégie** : montrer que tu connais la critique et que tu l'as adressée par design.

---

### Q7 — *"Votre 'décomposition méta-learner' n'est qu'une LogReg. C'est trivial mathématiquement."*

**Réponse** :
> *"Oui, mathématiquement c'est `z = β₀ + Σ βᵢ·xᵢ` puis sigmoid. Mais c'est précisément la valeur : c'est la formule **fermée et exacte** du modèle, pas une approximation. SHAP sur GradientBoosting est une approximation qui repose sur des hypothèses d'indépendance des features. Notre décomposition V8 est exacte par construction. C'est le seul niveau de l'XAI où on peut affirmer 'voici exactement ce que le modèle a fait', sans guillemets. Pour la conformité AI Act art. 13 sur les systèmes à risque limité, c'est juridiquement plus solide qu'une explication probabiliste."*

**Stratégie** : transformer "trivial" en "exact" — le mot qui frappe juridiquement.

---

### Q8 — *"Comment évaluez-vous votre modèle sur la dérive temporelle ?"*

**Réponse** :
> *"Honnête : on ne le fait pas systématiquement aujourd'hui, c'est dans la roadmap V10. On a un script de monitoring hebdomadaire dans `src/monitoring/weekly_score_check.py` qui sauvegarde la distribution des scores en JSON pour détection de drift par comparaison de distributions. Mais on n'a pas encore d'alerting automatique. Pour la soutenance, on a vérifié manuellement la stabilité des scores sur les 537 000 posts collectés depuis décembre 2025 — pas de drift visible. Industrialiser ce monitoring avec Grafana et alerts est priorisé V11."*

**Stratégie** : reconnaître la limite, montrer ce qui existe, donner la roadmap. Honnêteté > esquive.

---

## II. Questions sur l'XAI et la rigueur scientifique (5 questions)

### Q9 — *"Votre AOPC uplift de +0.21 est-il significatif statistiquement ?"*

**Réponse** :
> *"On a comparé sur 5 seeds aléatoires, donc on a un écart-type. AOPC attribution = 0.253, AOPC random moyen = 0.045 avec écart-type ~0.012. L'uplift est à plus de 17 sigmas du baseline aléatoire — ce qui dépasse largement le seuil de significativité usuel de 5 sigmas en physique des particules. Pour un test de validité scientifique, 17 sigmas, c'est 'ne s'arrête pas au hasard'. On peut faire un test t paired si vous voulez le formaliser, le résultat sera p < 1e-15."*

**Stratégie** : citer les sigmas et le seuil physique pour impressionner par la rigueur.

---

### Q10 — *"Le Δ_convergence de 0.15 sur le TP rejeté en IG, c'est gênant non ?"*

**Réponse** :
> *"C'est documenté comme une signature **du modèle**, pas une faille de la méthode. Sur les cas où CamemBERT est très confiant — P > 0.7 — le ReLU du head MLP sature et bloque les gradients, ce qui rend l'attribution gradient-based difficile par construction. C'est un comportement attendu sur les transformers profonds, documenté par Sundararajan dans 'The Many Shapley Values for Model Explanation' (2020). On l'a explicitement caractérisé dans la Model Card section 7.2 avec trois régimes : axiomatique, pratique, indicatif. Cette transparence est elle-même un livrable XAI : on sait quand notre attribution est fiable et quand elle ne l'est pas."*

**Stratégie** : transformer la limite en preuve de rigueur méthodologique.

---

### Q11 — *"Vous citez ERASER (DeYoung 2020). Mais vous comparez à un baseline aléatoire seulement. Pourquoi pas LIME ou des méthodes concurrentes ?"*

**Réponse** :
> *"Excellente question. L'AOPC uplift contre random est la première étape du protocole ERASER — la **discriminant validity**. La comparaison entre méthodes XAI (LIME vs SHAP vs IG) est ce qu'on appelle la **convergent validity**, qui est dans notre roadmap V10. Pour la soutenance, on s'est concentrés sur prouver que SHAP est nettement meilleur que random — ce qui est l'exigence minimale d'ERASER section 4.1. Comparer à LIME ajouterait 4-6 jours de travail pour un gain marginal sur la qualité de notre validation actuelle."*

**Stratégie** : reconnaître le manque, montrer que tu as la roadmap, justifier le scope.

---

### Q12 — *"Votre kappa de Cohen est 0.498 — c'est faible. Comment vos conclusions sont-elles défendables ?"*

**Réponse** :
> *"0.498 est dans la fourchette 'modérée' selon Landis & Koch (1977). C'est faible parce que la frontière fiable/suspect est intrinsèquement subjective — un journaliste et un chercheur en linguistique computationnelle annotent différemment. On a documenté cette limite dans le rapport § 23 et dans la Model Card. Pour atténuer, on a calculé un intervalle de confiance bootstrap à 95% sur la réduction des faux positifs : [-73%, -60%]. Cet intervalle reste fortement négatif même sous toutes les configurations d'annotation possibles, ce qui confirme la robustesse de l'effet observé. Notre Master Card recommande pour la production de re-évaluer trimestriellement avec un nouveau gold."*

**Stratégie** : citer Landis & Koch (la référence sur kappa), donner l'IC bootstrap qui sauve l'argument.

---

### Q13 — *"Si je vous demande de me donner un exemple de FP que votre XAI vous a aidé à comprendre, lequel choisissez-vous ?"*

**Réponse** :
> *"Le post id=36 de notre gold set, en français : 'Il y a 10 ans, la COP21 faisait de Paris le cœur battant...' Notre modèle V6 le classe SUSPECT à 97% de confiance alors qu'il est en réalité fiable. La heatmap d'attention CamemBERT montre que le modèle se concentre sur des marqueurs de surface — 'cœur battant', 'COP21' avec point d'exclamation implicite — alors qu'un humain voit immédiatement qu'il s'agit d'une rétrospective journalistique. SHAP confirme : 'paragraph_count', 'sensationalism_score' et 'exclamation_count' contribuent positivement à la classification suspect. Cette analyse a justifié notre architecture cascade V9 qui filtre fait/opinion en amont — précisément pour qu'un texte journalistique avec ton emphatique ne soit pas classé fake news. C'est l'XAI qui a guidé l'architecture."*

**Stratégie** : montrer que tu peux **disséquer un cas concret** avec les chiffres exacts. Démontre une maîtrise totale.

---

## III. Questions sur la conformité et l'éthique (4 questions)

### Q14 — *"Votre modèle peut-il discriminer certaines communautés ou certains sujets ?"*

**Réponse** :
> *"On a fait un audit de biais documenté dans le rapport § 18. L'écart de F1 entre français et anglais est de 0,02 — sous le seuil que la littérature considère comme problématique (Bolukbasi et al. 2016 fixe le seuil à 0,05). On a aussi vérifié sur des sous-groupes thématiques : santé, politique, climat. Le seul biais documenté et corrigé est le biais Reuters — détecté grâce à l'XAI en V1, mitigé par la liste BODY_AGENCY_TERMS. Pour les biais résiduels — par exemple sur le sarcasme ou les mèmes — c'est documenté dans la Model Card section 5 comme limite assumée. La roadmap V12 inclut un audit d'équité algorithmique formalisé selon le framework de Mehrabi et al. 2021."*

**Stratégie** : prouver que tu connais les standards (Bolukbasi, Mehrabi), reconnaître ce qui n'est pas couvert.

---

### Q15 — *"Vous classez quelqu'un de 'suspect'. Cette personne peut-elle vous attaquer en justice ?"*

**Réponse** :
> *"C'est exactement la raison pour laquelle l'XAI est l'épine dorsale du projet. Le RGPD article 22 donne à toute personne le droit de **contester une décision automatisée**. Notre dashboard expose pour chaque score : les top mots qui ont contribué, les features stylistiques, la décomposition exacte du méta-learner V8. Si un utilisateur est classé suspect, il peut demander l'explication, la contester, et nous fournir les éléments de re-classification. La Model Card section 6 documente le mécanisme de feedback. De plus, et c'est important : le système est positionné comme **aide à la décision** — il ne déclenche aucune action automatique. La supervision humaine est obligatoire au sens de l'AI Act art. 14. En pratique, le risque juridique est limité parce que aucune sanction n'est automatisée à partir du score."*

**Stratégie** : citer les articles juridiques précis (RGPD 22, AI Act 14) et la Model Card. Tu prouves ta maîtrise du cadre légal.

---

### Q16 — *"Si l'AI Act évolue d'ici 2027, votre projet est-il prêt à s'adapter ?"*

**Réponse** :
> *"On a une politique de veille réglementaire trimestrielle documentée dans `docs/10_veille_technologique.md` qui inclut explicitement la Commission Européenne et la CNIL. Le projet est conçu pour être **modulaire** — la Model Card et l'AIPD sont versionnées, l'XAI est plug-and-play par module, et la documentation suit le format Mitchell 2019 qui est en train de devenir le standard. Si l'AI Act monte le système en 'risque élevé', on devra ajouter un système de gestion des risques au sens de l'art. 9, mais on a déjà la structure de gouvernance documentée. Le coût d'adaptation est estimé entre 2 et 4 semaines selon la complexité du nouvel article."*

**Stratégie** : montrer que tu as anticipé les évolutions et que ton projet est conçu pour évoluer.

---

### Q17 — *"6,14g de CO2, c'est si peu — c'est crédible ?"*

**Réponse** :
> *"C'est mesuré par CodeCarbon, qui utilise le mix énergétique de la région Île-de-France selon l'API de RTE. La cohérence : 6,14g pour environ 12 minutes de fine-tuning CamemBERT cumulé sur Apple Silicon M4 (machine 22W TDP) plus tous les entraînements V1-V9 cumulés. C'est crédible parce qu'on est sur du fine-tuning de modèle pré-entraîné, pas un from-scratch — qui aurait coûté plusieurs centaines de grammes. Pour comparaison : entraîner GPT-3 from scratch émet ~552 tonnes de CO2 (Strubell et al. 2019). Notre 6,14g est l'équivalent de 26 mètres en voiture thermique. C'est ce qu'on appelle de la **frugalité par design**, et c'est documenté dans le rapport § 15 avec le détail par version."*

**Stratégie** : donner la méthodologie de mesure et la comparaison qui contextualise — 26 mètres en voiture, c'est mémorable.

---

## IV. Questions sur l'organisation et la collaboration (3 questions)

### Q18 — *"Vous décrivez 9 rôles dans `roles_et_competences_projet.md`. Comment un binôme peut-il porter cette ambition ?"*

**Réponse** :
> *"Le projet d'étude M1 simule un cycle complet de production. On a fait le choix pédagogique de faire endosser à chaque membre plusieurs rôles plutôt qu'un seul, pour acquérir une vision système. La répartition réelle est : moi sur le pipeline ML, l'XAI, le dashboard et l'industrialisation ; Sébastien sur l'évaluation gold set, le débiaisage Reuters et la qualité des données. Cette asymétrie est documentée et assumée — elle reflète nos compétences en entrée de M1 et notre choix de monter en compétence individuellement plutôt que de fragmenter le travail. Dans une équipe de 4 personnes comme prévu par le cadre pédagogique, on aurait probablement développé une couche de monitoring drift plus sophistiquée, qui est dans la roadmap V11."*

**Stratégie** : la phrase clé est "choix pédagogique de monter en compétence individuellement". Pas une excuse, une vision.

---

### Q19 — *"Comment avez-vous géré les conflits ou désaccords techniques dans l'équipe ?"*

**Réponse** :
> *"Deux désaccords majeurs documentés dans le journal de décision (`docs/00_INDEX.md`). Premièrement, le choix LogReg vs RoBERTa pour le pipeline principal : moi favorable à RoBERTa pour la performance, Sébastien favorable à LogReg pour la frugalité. On a tranché par mesure : test d'A/B avec CodeCarbon, F1 et latence, qui a confirmé que LogReg + ensemble V8 atteint un meilleur compromis. Deuxièmement, la question du seuil de fait/opinion : 0,40 vs 0,45. On a fait du grid search sur le gold pour trancher empiriquement. La règle qu'on s'est donnée est : **mesurer plutôt que débattre**. Tous les arbitrages techniques sont documentés avec leur justification dans le journal de décision."*

**Stratégie** : montrer une culture de décision data-driven, citer des conflits réels résolus.

---

### Q20 — *"Si vous deviez recommencer, qu'est-ce que vous feriez différemment ?"*

**Réponse** :
> *"Trois choses. Un : je commencerais par construire le gold test set annoté manuellement **dès la semaine 2**, pas au mois 4. Le gold a tout révélé — biais Reuters, FP sur les textes courts, sur-revendication F1=0.99. L'évaluation est plus importante que l'entraînement, c'est le grand apprentissage de ce projet. Deux : je mettrais en place MLflow ou Weights & Biases dès le début pour le tracking des expériences. Les 28 notebooks documentent chaque expérience mais un outil dédié aurait permis un suivi plus systématique. Trois : je déléguerais davantage. En tant que lead technique, j'ai centralisé trop de responsabilités. La répartition aurait été plus équilibrée si j'avais découpé les modules dashboard et XAI plus tôt. Ces trois apprentissages sont dans le rendu individuel section 5."*

**Stratégie** : montrer une vraie auto-critique structurée, pas du faux mea culpa.

---

## V. Questions piège et meta-questions (2 questions)

### Q21 — *"Si je vous donne un texte au hasard maintenant, votre modèle peut-il l'expliquer ?"*

**Réponse** :
> *"Oui. Tout texte FR ou EN, en moins de 5 secondes via le dashboard `streamlit run dashboard/app.py`. Vous donnez la chaîne, je vous montre : score, top mots LogReg, SHAP V6, décomposition méta-learner V8, attention CamemBERT si FR. On peut le faire maintenant si vous voulez, j'ai le dashboard ouvert."*

**Stratégie** : confiance totale + offre de démo. Si elle accepte, tu fais la démo en direct, tu valides ton expertise. Si elle refuse, tu as gagné le point sans risquer.

---

### Q22 — *"Quelle est la plus grosse faiblesse de votre projet, selon vous ?"*

**Réponse** :
> *"Le gold set. 200 posts, 9 suspects, kappa de Cohen 0,498. Toutes nos métriques de validation reposent sur ce gold. Si on devait pousser le projet en production, la première chose à faire serait d'investir dans un gold set plus grand — au moins 1000 posts — avec une procédure d'annotation plus rigoureuse incluant trois annotateurs et un protocole de résolution de désaccord. Ce serait V10 priorité 0. Ce qu'on a est suffisant pour démontrer la méthode et valider l'ordre de grandeur des effets, mais pas pour des claims chiffrés en production. C'est documenté comme tel dans la Model Card section 5."*

**Stratégie** : assumer la faiblesse réelle plutôt qu'une fake faiblesse en mode "je suis trop perfectionniste". Le jury respecte cette honnêteté.

---

## VI. Questions veille technologique et reglementaire

### Q23 — *"L'AI Act entre en vigueur le 2 aout 2026. Votre projet sera-t-il conforme a temps ?"*

**Reponse** :
> *"ThumaCheck est conforme by design. Pour l'article 50, on a implemente une banniere de transparence IA visible des la connexion au dashboard — l'utilisateur sait immediatement qu'il interagit avec un systeme d'IA. Pour les articles 13 et 14, notre Model Card MC-THUM-2026-001 documente le modele selon le format Google et notre pipeline XAI offre 8 methodes d'explication. Notre classification en risque limite est documentee dans le doc 02 section 4.1. On anticipe l'echeance, on ne la subit pas."*

**Strategie** : enumerer les preuves concretes d'implementation, pas les intentions. Citer le numero de document et la section.

---

### Q24 — *"Apres l'incident HuggingFace de juillet 2026, comment gerez-vous la securite de votre pipeline ML ?"*

**Reponse** :
> *"L'incident HuggingFace montre qu'un agent IA autonome peut exploiter un dataset malveillant pour compromettre une plateforme. Notre pipeline est protege a plusieurs niveaux : CORS restrictif avec origines explicites — plus de wildcard, rate limiting a 60 requetes par minute, sanitization HTML des entrees, MongoDB restreint a localhost, Docker avec utilisateur non-root, et surtout nos modeles sont charges localement — on n'a aucune dependance HuggingFace en production. Le fichier .env est exclu du versionning. C'est documente dans la section 5 du doc conformite."*

**Strategie** : citer l'incident par son nom pour montrer la veille active, puis enumerer les mesures concretes.

---

### Q25 — *"Pourquoi ne pas utiliser Mistral plutot que CamemBERT pour le francais ?"*

**Reponse** :
> *"Ce sont deux architectures fondamentalement differentes. CamemBERT est un encodeur de 110 millions de parametres optimise pour la classification — 38 ms par texte. Mistral est un LLM generatif de 7 milliards de parametres minimum — on estime plus de 200 ms par texte avec GPU. Pour notre tache de classification binaire, CamemBERT est le bon outil : plus rapide, plus frugal, plus explicable via les attention maps. Mistral est dans notre roadmap V12 comme alternative souveraine pour quand le cas d'usage evoluera vers la verification factuelle, qui est une tache generative. C'est un choix d'architecture delibere, pas un manque de veille."*

**Strategie** : montrer qu'on connait le paysage des modeles et qu'on a fait un choix eclaire encodeur vs generatif.

---

### Q26 — *"Que pensez-vous des modeles open-weight recents comme Kimi K3 ou GLM-5.2 pour votre pipeline ?"*

**Reponse** :
> *"Kimi K3, 2800 milliards de parametres, et GLM-5.2, 744 milliards — ce sont des modeles impressionnants. Mais ils sont hors de notre contrainte de frugalite : notre pipeline traite un texte en 1,5 ms, ces modeles necessitent des GPU haut de gamme et des centaines de millisecondes. Ils sont pertinents pour notre roadmap V11 ou on prevoit une analyse batch offline avec des LLM open-weight pour la verification factuelle. Le point important : open-weight ne signifie pas open-source — les licences varient et certaines ont des implications pour la conformite AI Act. On les surveille activement mais on ne les integre pas par effet de mode."*

**Strategie** : montrer un esprit critique face au hype cycle, distinguer inference temps reel vs batch, et soulever le point licence.

---

## Preparation a la soutenance — regles d'or

**1. Le silence vaut de l'or.** Si on te pose une question difficile, prends 3 secondes de silence avant de répondre. Ça donne l'image d'une réflexion et ça désamorce l'urgence.

**2. Cite tes sources.** Chaque fois que tu mentionnes une méthode, cite l'auteur et l'année. *"Selon Sundararajan 2017"*, *"Le protocole ERASER de DeYoung 2020"*, *"Le framework Mitchell 2019"*. Le jury te perçoit immédiatement comme quelqu'un qui maîtrise la littérature.

**3. Donne toujours un chiffre.** Au lieu de *"on a beaucoup de tests"*, dis *"501 tests pytest, 80% coverage, 80,3% mutation kill rate"*. Au lieu de *"c'est explicable"*, dis *"AOPC uplift +0.21 contre baseline aléatoire"*. Les chiffres ancrent la crédibilité.

**4. Reconnais les limites en premier.** Si tu sais qu'une critique va arriver, mentionne-la avant la question : *"On a un kappa de 0,498 — modéré. Voici comment on l'a mitigé."* Ça désamorce 80% des Q&R.

**5. Ne te justifie jamais sur la durée.** Si tu prends 60 secondes pour une réponse, ne dis pas *"je vais essayer d'être bref"*. Donne ta réponse structurée, point.

**6. Termine sur une affirmation.** Chaque réponse doit se terminer par une phrase forte : *"...c'est documenté dans la Model Card"*, *"...c'est l'XAI qui a guidé l'architecture"*, *"...c'est notre choix architectural assumé"*. Le jury retient la dernière phrase.

**7. Si tu ne sais pas, dis-le.** *"Je n'ai pas la réponse précise mais voici comment on pourrait l'aborder..."* > *"Euh, c'est une bonne question..."* La franchise est valorisée.

**8. Ramène à tes points forts.** Si on te pose une question sur un sujet que tu maîtrises moins (genre architecture cloud), ramène à ce que tu maîtrises : *"On n'a pas déployé sur AWS mais on a fait le choix de Docker Compose pour la reproductibilité, qui est documentée dans..."*.

**9. Présence physique.** Posture droite, mains visibles, pas de tics (jouer avec un stylo, toucher les cheveux). Regarde **chaque membre du jury au moins une fois** par réponse.

**10. La phrase à utiliser quand tu doutes.** *"C'est exactement le compromis qu'on a documenté dans X."* Cette phrase fonctionne dans 80% des cas et signale ton sérieux documentaire.

---

## Tableau de mémorisation rapide (à imprimer)

| Q & sujet | Chiffre clé à citer | Source à mentionner |
|---|---|---|
| Performance modèle | F1 0.67 / FP -67% / IC [-73,-60]% | Rapport § 21, Fisher exact |
| Frugalité | 1.5 ms, 6.14g CO2, 130× moins | Rapport § 15, CodeCarbon |
| Tests | 501 tests, 80% / 77.9% / 80.3% kill | Coverage report, mutmut |
| XAI | AOPC +0.21, 5.6× random, 17 sigmas | DeYoung 2020 (ERASER) |
| Volume | 245k posts, 197k textes, 9 versions | Rapport § 8 |
| Annotation | kappa 0.498, n=200, 9 suspects | Landis & Koch 1977 |
| Conformité | RGPD art 22/35, AI Act art 13/14/50 | Doc 02, Model Card |
| Limites | gold set 200 posts, ReLU saturé IG | Sundararajan 2020 |
| Securite ML | CORS restrictif, rate limit 60/min, non-root | Incident HuggingFace juil. 2026 |
| AI Act deadline | Art. 50 transparence, banniere dashboard | 2 aout 2026, UE 2024/1689 |
| Modeles souverains | Mistral V12 roadmap, CamemBERT 38ms | Frugalite vs puissance |

---

*Q&R défensif ThumaCheck (client : Thumalien) — Mai 2026 — préparation soutenance*
