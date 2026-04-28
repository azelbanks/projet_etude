"""Generates the video MVP guide as a PDF using fpdf2."""
import re
from fpdf import FPDF


def _clean(text):
    """Replace unicode chars unsupported by Helvetica with ASCII equivalents."""
    return (
        text.replace("\u2014", "-")   # em dash
        .replace("\u2013", "-")       # en dash
        .replace("\u2018", "'")       # left single quote
        .replace("\u2019", "'")       # right single quote
        .replace("\u201c", '"')       # left double quote
        .replace("\u201d", '"')       # right double quote
        .replace("\u2026", "...")     # ellipsis
        .replace("\u00e9", "e")      # e accent aigu
        .replace("\u00e8", "e")      # e accent grave
        .replace("\u00ea", "e")      # e accent circ
        .replace("\u00e0", "a")      # a accent grave
        .replace("\u00f4", "o")      # o accent circ
        .replace("\u00ee", "i")      # i accent circ
        .replace("\u00e7", "c")      # c cedille
        .replace("\u00fb", "u")      # u accent circ
    )

class GuidePDF(FPDF):
    def cell(self, *args, **kwargs):
        # Clean text arg (3rd positional or 'text'/'txt' kwarg)
        args = list(args)
        if len(args) > 2 and isinstance(args[2], str):
            args[2] = _clean(args[2])
        for k in ("text", "txt"):
            if k in kwargs and isinstance(kwargs[k], str):
                kwargs[k] = _clean(kwargs[k])
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        args = list(args)
        if len(args) > 2 and isinstance(args[2], str):
            args[2] = _clean(args[2])
        for k in ("text", "txt"):
            if k in kwargs and isinstance(kwargs[k], str):
                kwargs[k] = _clean(kwargs[k])
        return super().multi_cell(*args, **kwargs)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, "Thumalien — Guide Production Video MVP", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(0, 51, 102)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 51, 102)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(51, 51, 51)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def sub_sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(33, 33, 33)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(33, 33, 33)
        x = self.get_x()
        self.cell(indent, 5.5, "")
        self.multi_cell(0, 5.5, f"  {text}")
        self.ln(0.5)

    def bold_text(self, label, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(33, 33, 33)
        self.cell(self.get_string_width(label) + 1, 5.5, label)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            n = len(headers)
            col_widths = [190 / n] * n
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(33, 33, 33)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(240, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            max_h = 7
            for i, cell in enumerate(row):
                self.cell(col_widths[i], max_h, str(cell)[:60], border=1, fill=True)
            self.ln()
            fill = not fill
        self.ln(3)

    def script_block(self, speaker, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(0, 102, 51)
        self.cell(0, 6, f"[{speaker}]", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Courier", "", 9)
        self.set_text_color(60, 60, 60)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(2)

    def highlight_box(self, text, color=(255, 243, 205)):
        self.set_fill_color(*color)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(33, 33, 33)
        y = self.get_y()
        self.rect(10, y, 190, 12, "F")
        self.set_xy(14, y + 2)
        self.multi_cell(182, 5.5, text)
        self.ln(4)


def build_pdf():
    pdf = GuidePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title page
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "Guide de Production Video MVP", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 12, "Thumalien — Social Media Intelligence & AI Monitor", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "VI.2 - VIDEO & MVP - SAVOIR CONVAINCRE", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(33, 33, 33)
    info = [
        "Equipe : Azelie Bernard (Lead Technique) / Sebastien Lazcanotegui (Optimisation ML & Doc)",
        "Formation : Master Big Data — Sup de Vinci",
        "Duree cible : 15-18 minutes",
        "Livrable : MP4 1920x1080 ou lien YouTube non repertorie",
    ]
    for line in info:
        pdf.cell(0, 7, line, align="C", new_x="LMARGIN", new_y="NEXT")

    # Section 1 — Strategie
    pdf.add_page()
    pdf.section_title("1. Strategie de differenciation")

    pdf.sub_title("Ce que font les autres groupes")
    for item in ["PowerPoint filme avec voix off monotone", "Demo basique en fin de video", "Pas de storytelling, pas d'emotion"]:
        pdf.bullet(item)
    pdf.ln(2)

    pdf.sub_title("Ce que NOUS allons faire")
    strategies = [
        ("Storytelling 'enquete'", "commencer par le probleme (vrais posts suspects Bluesky), pas par la solution"),
        ("Plot twist V1", "raconter comment notre V1 a 99.6% etait en realite cassee — montre la maturite"),
        ("Demo live interactive", "taper des textes en direct (FR et EN) et voir le verdict en temps reel"),
        ("Radar emotionnel", "exploiter le radar chart 7 emotions du dashboard (visuellement tres fort)"),
        ("Chiffres d'impact", "188K posts, 11 versions, 0.55g CO2 — des metriques concretes"),
    ]
    for label, desc in strategies:
        pdf.bold_text(f"- {label} : ", desc)

    # Section 2 — Moyens techniques
    pdf.add_page()
    pdf.section_title("2. Moyens techniques de production")

    pdf.sub_title("2.1 Capture video")
    pdf.table(
        ["Outil", "Usage", "Prix", "Plateforme"],
        [
            ["OBS Studio", "Ecran + webcam simultanee", "Gratuit", "Mac/Win/Linux"],
            ["QuickTime Player", "Capture ecran simple", "Gratuit", "Mac"],
            ["Loom", "Ecran + webcam + partage", "Gratuit (5min) / 15$/m", "Web"],
        ],
        [45, 60, 45, 40],
    )
    pdf.highlight_box("Recommandation : OBS Studio — picture-in-picture, scenes pre-configurees, bandeau nom")

    pdf.body_text(
        "Configuration OBS recommandee :\n"
        "- Resolution : 1920x1080 @ 30fps\n"
        "- Encodeur : x264, CRF 18 (qualite elevee)\n"
        "- Audio : micro USB ou AirPods Pro (pas le micro integre du laptop)\n"
        "- 4 scenes : Slide (plein ecran), Demo (ecran+webcam), Speaker (webcam+bandeau), Split (50/50)"
    )

    pdf.sub_title("2.2 Montage video")
    pdf.table(
        ["Outil", "Niveau", "Prix", "Avantage"],
        [
            ["iMovie", "Debutant", "Gratuit (Mac)", "Simple, couper/assembler"],
            ["DaVinci Resolve", "Intermediaire/Pro", "Gratuit", "Etalonnage, effets, titrage pro"],
            ["CapCut Desktop", "Debutant/Intermed.", "Gratuit", "Sous-titres auto, templates"],
        ],
        [45, 45, 50, 50],
    )
    pdf.highlight_box("Recommandation : CapCut Desktop pour sous-titres auto + templates bandeaux nom")

    pdf.sub_title("2.3 Slides / Visuels")
    pdf.table(
        ["Outil", "Usage", "Prix"],
        [
            ["Canva", "Slides, schemas, bandeaux", "Gratuit / Pro etudiant"],
            ["Google Slides", "Slides collaboratifs", "Gratuit"],
            ["Figma", "Schemas sur mesure", "Gratuit"],
            ["Mermaid Live Editor", "Diagrammes depuis le code", "Gratuit"],
        ],
        [50, 80, 60],
    )

    pdf.sub_title("2.4 Son")
    for item in [
        "Micro-casque USB ou AirPods Pro — jamais le micro integre du laptop",
        "Enregistrer dans une piece calme, porte fermee",
        "Test de 30 secondes + reecoute AVANT de filmer",
        "Post-traitement si necessaire : Audacity (gratuit) — reduction de bruit",
    ]:
        pdf.bullet(item)

    # Section 3 — IA
    pdf.add_page()
    pdf.section_title("3. Comment l'IA peut nous aider")

    pdf.sub_title("3.1 Generation de visuels")
    pdf.table(
        ["Tache", "Outil IA", "Comment"],
        [
            ["Slides professionnelles", "Canva AI (Magic Design)", "Generer slides theme dark/tech"],
            ["Schema d'architecture", "Claude (artefact)", "Schema SVG ou Mermaid du pipeline"],
            ["Illustrations", "DALL-E / Midjourney", "Visuels cybersecurity, data analysis"],
            ["Logo Thumalien", "Canva AI / DALL-E", "Logo style tech/surveillance"],
        ],
        [50, 55, 85],
    )

    pdf.sub_title("3.2 Script et voix")
    pdf.table(
        ["Tache", "Outil IA", "Comment"],
        [
            ["Script detaille", "Claude", "Texte mot-a-mot (inclus dans ce doc)"],
            ["Sous-titres auto", "CapCut AI", "Genere les sous-titres FR"],
            ["Corrections audio", "Adobe Podcast AI", "Ameliore qualite, supprime bruit"],
        ],
        [50, 55, 85],
    )

    pdf.sub_title("3.3 Remotion via Claude — Video programmatique")
    pdf.body_text(
        "Remotion (remotion.dev) est un framework React pour creer des videos par le code. "
        "Claude peut generer le code pour :\n"
        "- Animations de graphiques (courbe F1 qui monte version apres version)\n"
        "- Bandeaux nom animes\n"
        "- Transitions entre slides\n"
        "- Affichage progressif du tableau des 11 versions"
    )
    pdf.bold_text("Avantages : ", "rendu pixel-perfect, reproductible, animations fluides, Claude ecrit le code")
    pdf.bold_text("Inconvenients : ", "necessite Node.js + React, ~2h de setup")

    pdf.highlight_box(
        "Verdict : approche HYBRIDE recommandee — Remotion pour les animations, "
        "OBS pour les screencasts, assemblage dans CapCut/DaVinci",
        color=(220, 237, 255),
    )

    pdf.sub_title("3.4 Alternative sans code : Canva + CapCut")
    pdf.body_text(
        "1. Canva : creer toutes les slides avec animations, exporter en MP4\n"
        "2. CapCut : assembler slides + screencasts OBS + sous-titres auto\n"
        "3. Claude : generer le script, textes des slides, talking points"
    )

    # Section 4 — Script detaille
    pdf.add_page()
    pdf.section_title("4. Script detaille — 15 minutes")

    # Acte 1
    pdf.sub_title("ACTE 1 — 'Le Probleme' (Sebastien, 3 min)")

    pdf.sub_sub_title("[0:00-0:30] Accroche")
    pdf.body_text("Ecran noir + texte : 'Sur Bluesky, 1 post sur 4 que nous analysons est suspect.'")
    pdf.script_block("Sebastien",
        "Chaque jour, des milliers de posts circulent sur Bluesky.\n"
        "Certains informent. D'autres manipulent.\n"
        "A l'oeil nu, la difference est parfois invisible.\n"
        "Comment faire le tri dans 188 000 posts, en francais ET en anglais,\n"
        "quand chaque texte ne fait que 10 mots ?"
    )

    pdf.sub_sub_title("[0:30-1:30] La problematique")
    pdf.script_block("Sebastien",
        "Bluesky est un reseau social decentralise base sur le protocole AT.\n"
        "Contrairement a Twitter, il n'y a pas de moderation centralisee.\n"
        "C'est un terrain fertile pour la desinformation.\n\n"
        "Notre defi :\n"
        "- Des textes ultra-courts : 5 a 20 mots en moyenne\n"
        "- Deux langues : francais et anglais\n"
        "- Un volume massif : plus de 188 000 posts collectes\n"
        "- Et zero budget cloud : tout tourne en local.\n\n"
        "Un humain met 3 minutes pour verifier un seul post.\n"
        "Pour 188 000 posts, c'est plus d'un an de travail non-stop.\n"
        "C'est pour ca qu'on a cree Thumalien."
    )

    pdf.sub_sub_title("[1:30-2:30] Presentation de l'equipe")
    pdf.script_block("Sebastien",
        "Je suis Sebastien Lazcanotegui. Mon role : l'optimisation\n"
        "du machine learning et la coherence documentaire.\n"
        "J'ai travaille sur le debiaisage, le GridSearch, et la consolidation.\n\n"
        "Ma collegue Azelie Bernard est la lead technique.\n"
        "C'est elle qui a concu l'ensemble du pipeline :\n"
        "de la collecte Bluesky au dashboard, en passant par\n"
        "les 11 versions du modele et les fine-tunings Transformer."
    )

    pdf.sub_sub_title("[2:30-3:00] Teaser solution")
    pdf.script_block("Sebastien",
        "Voici Thumalien en un schema : Bluesky, collecteur, MongoDB,\n"
        "pipeline NLP, dashboard. Tout conteneurise avec Docker.\n"
        "Une commande, et le systeme entier demarre."
    )

    # Acte 2
    pdf.add_page()
    pdf.sub_title("ACTE 2 — 'Le Parcours' (Sebastien, 4 min)")

    pdf.sub_sub_title("[3:00-4:00] Methodologie")
    pdf.script_block("Sebastien",
        "On ne construit pas un modele d'IA en une fois.\n"
        "On suit le cycle CRISP-DM adapte a l'IA :\n"
        "comprendre, explorer, preparer, modeliser, evaluer, deployer.\n"
        "Et surtout : iterer. Encore et encore."
    )

    pdf.sub_sub_title("[4:00-5:30] Le plot twist — La V1 a 99.6%")
    pdf.highlight_box("MOMENT CLE DE LA VIDEO — Montre la maturite intellectuelle du projet", color=(255, 220, 220))
    pdf.script_block("Sebastien",
        "Notre premier modele affichait 99.6% de precision.\n"
        "(pause)\n"
        "99.6%. On pourrait se feliciter et rentrer chez nous.\n\n"
        "Sauf que quand on a regarde les mots les plus predictifs,\n"
        "on a trouve... 'reuters', 'reporting by', 'editing by'.\n\n"
        "Le modele ne detectait pas les fake news.\n"
        "Il detectait les articles Reuters. C'etait un biais de donnees.\n\n"
        "C'est LA lecon de ce projet : des metriques parfaites\n"
        "peuvent cacher un modele completement inutile.\n"
        "Il faut toujours tester sur des donnees reelles."
    )

    pdf.sub_sub_title("[5:30-7:00] 11 versions en 5 mois")
    pdf.script_block("Sebastien",
        "A partir de ce constat, on a itere. 11 fois.\n\n"
        "V1.5 : debiaisage Reuters, ajout du francais.\n"
        "V2 : tweets et titres courts. Bluesky passe de 23% a 73% fiable.\n"
        "V3 : 5 features sur 12 etaient a zero — bug de preprocessing.\n"
        "V4 : augmentation FR court, +32% de F1.\n"
        "V5 : 10 000 posts sociaux synthetiques, test 12/12.\n\n"
        "Puis les transformers :\n"
        "CamemBERT V2 : F1 0.957 sur ultra-courts FR.\n"
        "RoBERTa V2 : F1 0.874 sur ultra-courts EN, test 16/18.\n\n"
        "De mon cote, j'ai optimise les hyperparametres par GridSearch\n"
        "et ajoute le debiaisage residuel des agences et annees artefacts.\n\n"
        "Le tout pour 0.55 gramme de CO2."
    )

    # Acte 3
    pdf.add_page()
    pdf.sub_title("ACTE 3 — 'La Demo' (Azelie, 6 min)")

    pdf.sub_sub_title("[7:00-8:00] Architecture technique")
    pdf.script_block("Azelie",
        "4 containers Docker. Le collecteur se connecte a Bluesky\n"
        "via le protocole AT, stocke dans MongoDB, le pipeline NLP\n"
        "analyse chaque texte, et Streamlit affiche tout en temps reel.\n"
        "On lance tout avec une seule commande.\n"
        "(montrer docker-compose up dans le terminal)"
    )

    pdf.sub_sub_title("[8:00-9:30] Collecte + MongoDB")
    pdf.script_block("Azelie",
        "Le collecteur tourne en continu. En 5 mois : 188 553 posts.\n"
        "(montrer MongoDB : les documents, le count)\n"
        "Indexes, validation de schema, monitoring qualite."
    )

    pdf.sub_sub_title("[9:30-11:30] Demo Dashboard — Le moment cle")
    pdf.highlight_box("MOMENT WOW — Le design glassmorphism dark va impressionner", color=(220, 255, 220))
    pdf.script_block("Azelie",
        "Voici le dashboard Thumalien.\n"
        "(ouvrir Streamlit — KPIs, distribution, emotions)\n\n"
        "Test en direct :\n\n"
        "1. 'SCANDALE ! Le gouvernement cache la verite sur les vaccins !'\n"
        "   -> Score 0.02 -> SUSPECT (verdict rouge) + radar colere/peur\n\n"
        "2. 'Le CNRS publie une etude sur le changement climatique.'\n"
        "   -> Score 0.95 -> FIABLE (verdict vert) + radar neutre\n\n"
        "3. 'SHARE before they DELETE this!! The truth about 5G!'\n"
        "   -> Score 0.02 -> SUSPECT — 'Le modele detecte aussi en anglais.'\n\n"
        "4. 'The city council approved the new budget.'\n"
        "   -> Score 0.81 -> FIABLE\n\n"
        "'Le modele ne fait pas que classer vrai ou faux.\n"
        "Il analyse l'emotion derriere chaque texte.'"
    )

    pdf.sub_sub_title("[11:30-13:00] Sous le capot")
    pdf.script_block("Azelie",
        "Le pipeline combine 3 types de features :\n"
        "- TF-IDF : 30 000 n-grammes\n"
        "- 12 features linguistiques : majuscules, ponctuation, sensationnalisme\n"
        "- 7 probabilites emotionnelles : MLP PyTorch\n\n"
        "Le tout dans une regression logistique. Pourquoi pas un Transformer ?\n"
        "Parce qu'on peut EXPLIQUER chaque prediction.\n"
        "Ce n'est pas une boite noire."
    )

    # Acte 4
    pdf.sub_title("ACTE 4 — 'L'Impact' (2 min)")

    pdf.sub_sub_title("[13:00-14:00] ROI (Azelie)")
    pdf.script_block("Azelie",
        "188 000 posts analyses automatiquement — 1 an de travail humain.\n"
        "Zero euro de cloud GPU. 0.55g de CO2.\n"
        "100% open source, 100% reproductible.\n"
        "Scalable : Docker Compose vers Kubernetes."
    )

    pdf.sub_sub_title("[14:00-15:00] Conclusion (Sebastien)")
    pdf.script_block("Sebastien",
        "Prochaines etapes : API REST FastAPI, detection multimodale, plus de langues.\n\n"
        "Ce projet montre qu'avec les bonnes donnees, la bonne methodologie,\n"
        "et un esprit critique sur ses propres resultats,\n"
        "on peut construire un outil de lutte contre la desinformation\n"
        "efficace, interpretable, et eco-responsable.\n\n"
        "Merci pour votre attention."
    )

    # Section 5 — Planning
    pdf.add_page()
    pdf.section_title("5. Planning de production")
    pdf.table(
        ["Etape", "Qui", "Duree", "Livrable"],
        [
            ["1. Creer slides (Canva)", "Les deux", "3h", "10 slides PNG/MP4"],
            ["2. Repeter le script", "Chacun", "2x 30min", "Fluide, < 15 min"],
            ["3. Configurer OBS", "Azelie", "1h", "4 scenes configurees"],
            ["4. Enregistrer Actes 1+2", "Sebastien", "1-2h", "Rushes Sebastien"],
            ["5. Enregistrer Actes 3+4", "Azelie", "1-2h", "Rushes + screencasts"],
            ["6. Montage + sous-titres", "Les deux", "3h", "Video assemblee"],
            ["7. Relecture + corrections", "Les deux", "1h", "Version finale"],
            ["8. Export + upload", "Azelie", "30min", "Livrable final"],
        ],
        [60, 35, 35, 60],
    )
    pdf.highlight_box("TOTAL ESTIME : 12 a 15 heures de travail", color=(220, 237, 255))

    # Section 6 — Checklist
    pdf.section_title("6. Checklist avant tournage")
    checks = [
        "Dashboard Streamlit demarre et fonctionnel (docker-compose up)",
        "MongoDB avec des donnees (188K posts)",
        "OBS Studio installe + 4 scenes configurees",
        "Micro teste (30s + reecoute)",
        "Slides Canva finalisees et exportees",
        "Script imprime ou sur ecran secondaire",
        "Piece calme, porte fermee, notifications desactivees",
        "Webcam cadree (visage centre, fond neutre)",
        "Bandeau nom prepare dans OBS",
    ]
    for c in checks:
        pdf.bullet(f"[ ]  {c}")

    # Section 7 — Nomenclature
    pdf.ln(5)
    pdf.section_title("7. Nomenclature du livrable")
    pdf.sub_title("Solution 1 — Fichier ZIP")
    pdf.body_text("PE_2526_[codepromo]_Bernard_Lazcanotegui.zip\n  -> PE-2526_[codepromo]_BernardAzelie.mp4")
    pdf.sub_title("Solution 2 — Lien YouTube")
    pdf.body_text("PE_2526_[codepromo]_Bernard_Lazcanotegui.txt\n  -> contient le lien YouTube (non repertorie)")

    # Save
    pdf.output("guide_production_video.pdf")
    print("PDF genere : video_mvp/guide_production_video.pdf")


if __name__ == "__main__":
    build_pdf()
