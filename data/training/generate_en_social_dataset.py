#!/usr/bin/env python3
"""
Generator for EN social media synthetic dataset for Thumalien.
================================================================

Generates realistic short posts in English (Bluesky/Twitter style)
covering disinformation and reliable information patterns absent
from the ISOT/FakeNewsNet datasets (long articles and political titles).

Strategy:
    - Templates with combinatorial variations (subject x formulation x style)
    - Suspect posts: conspiracies, sensationalism, emotional manipulation
    - Reliable posts: news, science, sports, daily life, measured opinions
    - Balanced distribution ~50/50 reliable/suspect
    - Target length: 5-30 words (Bluesky use case)

Author: Thumalien Team
"""

import csv
import os
import random

random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'en_social_media_synthetic.csv')

# ================================================================
#  SUSPECT TEMPLATES (label=1)
# ================================================================

# --- Classic conspiracies ---
CONSPIRACIES = [
    "Vaccines contain {substance} to {objective} !!",
    "BREAKING: they're hiding the truth about {conspiracy_subject} !",
    "The government is using {technology} to {tech_objective} !!!",
    "{institution} is covering up the real numbers on {stats_subject}",
    "EXPOSED: {personality} admitted that {revelation} !",
    "The {media} refuses to cover {censored_subject}",
    "WARNING: {alert_subject} is actually {shocking_revelation}",
    "What {institution} doesn't want you to know about {conspiracy_subject}",
    "The truth about {conspiracy_subject} that nobody dares to say",
    "LEAKED: {personality} involved in {scandal}",
    "{institution} has been lying to us for {duration} about {conspiracy_subject}",
    "They're poisoning us with {substance} in {food}",
    "PROOF that {conspiracy_subject} is a government lie !",
    "Look what {institution} is doing in secret: {revelation}",
    "BOMBSHELL: documents prove that {revelation} since {duration}",
]

# --- Emotional manipulation ---
MANIPULATION = [
    "Share before they censor this !! {manipulation_subject}",
    "SPREAD THIS: {manipulation_subject} !!!",
    "Open your eyes ! {manipulation_subject}",
    "Wake up people: {manipulation_subject} !!",
    "They think we're stupid: {manipulation_subject}",
    "RT before deletion: {manipulation_subject}",
    "SHARE EVERYWHERE: {manipulation_subject}",
    "They want to censor this: {manipulation_subject} !!",
    "Send this to everyone you know ! {manipulation_subject}",
    "DON'T BE FOOLED: {manipulation_subject}",
    "CENSORED info: {manipulation_subject}",
    "BEFORE THEY DELETE THIS: {manipulation_subject} !!",
    "Do your own research on {conspiracy_subject} !",
    "Share urgently: {manipulation_subject}",
    "Who still denies that {manipulation_subject} ?",
]

# --- Health / pseudo-science ---
PSEUDO_HEALTH = [
    "This natural remedy cures {disease} in {short_duration} !",
    "Doctors don't want you to know: {remedy} cures {disease}",
    "STOP taking {treatment}: {remedy} is 10x more effective !",
    "Big Pharma is hiding this {remedy} that naturally cures {disease}",
    "{remedy}: the miracle cure for {disease} that labs are hiding",
    "Stop {treatment} NOW: {side_effect} proven !",
    "Testimony: I cured my {disease} with {remedy} in {short_duration}",
    "DANGER: {food} causes {disease}, authorities know about it !",
    "CENSORED study proves that {treatment} causes {side_effect}",
    "Your doctor will never tell you that {remedy} cures {disease}",
]

# --- Political misinformation ---
POLITICAL_SUSPECT = [
    "ELECTION FRAUD: {party} rigged the {election} !",
    "{political_personality} is a puppet of {organization}",
    "The {party} sold out to {organization}: here's the proof",
    "CORRUPTION: {political_personality} received {amount} from {organization}",
    "{political_personality} betrayed America for {organization}",
    "Elections are rigged: {party} controls everything",
    "SCANDAL: {political_personality} is lying to Americans about {political_subject}",
    "CLASSIFIED document proves that {political_personality} {accusation}",
    "{political_personality} = {organization}: same fight against the people",
    "Who's funding {political_personality}? Follow the money...",
]

# --- Variables for suspect templates ---
SUBSTANCES = [
    "nanoparticles", "graphene", "microchips", "heavy metals",
    "aluminum", "mercury", "GMOs", "endocrine disruptors",
    "GPS trackers", "chemical agents",
]

OBJECTIVES = [
    "control you", "reduce the population", "make you sick",
    "track you", "alter your DNA", "weaken your immune system",
    "sterilize you", "control your thoughts",
]

CONSPIRACY_SUBJECTS = [
    "vaccines", "5G", "chemtrails", "climate change",
    "COVID", "GMOs", "fluoride in water", "electromagnetic waves",
    "processed food", "pesticides", "nuclear energy",
    "pharmaceuticals", "artificial intelligence", "mass surveillance",
]

TECHNOLOGIES = [
    "5G", "Starlink satellites", "smart meters",
    "surveillance cameras", "facial recognition",
    "RFID chips", "cell towers", "drones",
]

TECH_OBJECTIVES = [
    "spy on us", "control the population", "manipulate the weather",
    "irradiate citizens", "harvest our data", "manipulate us",
]

INSTITUTIONS = [
    "The government", "The WHO", "Big Pharma", "The EU", "Pfizer",
    "The White House", "The labs", "The Fed", "The IMF", "Davos",
    "Big Tech", "Bill Gates", "Soros", "The Rothschilds",
]

PERSONALITIES = [
    "a president", "a senator", "the Pfizer CEO", "a whistleblower",
    "a former intelligence agent", "a repentant scientist", "an insider",
]

REVELATIONS = [
    "vaccines are useless", "5G is dangerous",
    "the virus was created in a lab", "the numbers are fake",
    "everything was planned since 2010", "it's a global conspiracy",
    "elections are rigged", "they're poisoning us on purpose",
]

STATS_SUBJECTS = [
    "mortality rates", "side effects", "unemployment",
    "immigration", "the national debt", "pollution", "crime rates",
]

CENSORED_SUBJECTS = [
    "vaccine side effects", "corruption in Washington",
    "the real COVID numbers", "the dangers of 5G",
    "the failed immigration policy", "pesticides in drinking water",
]

ALERT_SUBJECTS = [
    "the health pass", "universal basic income", "digital currency",
    "the social credit system", "the new tax reform", "the trade deal",
]

SHOCKING_REVELATIONS = [
    "a tool for social control", "a global scam",
    "a surveillance system", "a legal poison", "an organized fraud",
]

SCANDALS = [
    "a corruption ring", "a lobbying scandal",
    "illegal financing", "a health scandal",
    "a government cover-up", "embezzlement",
]

DURATIONS = ["years", "20 years", "the beginning", "2020", "forever"]

MEDIA = ["mainstream media", "journalists", "news channels", "big newsrooms"]

MANIPULATION_SUBJECTS = [
    "they're poisoning us and nobody reacts",
    "vaccines are killing people and they hide it",
    "5G is a deadly threat",
    "the government is setting up a health dictatorship",
    "our freedoms are in danger",
    "they're stealing our future",
    "the truth about COVID is finally coming out",
    "the evidence of the conspiracy is right there",
    "the people must rise up now",
    "it's now or never",
    "they're going to censor everything",
    "the system is collapsing and they're hiding it",
]

DISEASES = ["cancer", "diabetes", "autism", "Alzheimer's", "depression", "arthritis"]
REMEDIES = [
    "turmeric", "CBD oil", "lemon juice", "baking soda",
    "intermittent fasting", "high-dose vitamin C", "colloidal silver",
    "essential oils", "ivermectin",
]
TREATMENTS = ["chemotherapy", "vaccines", "antibiotics", "antidepressants"]
SIDE_EFFECTS = [
    "cancer", "infertility", "neurological disorders",
    "autoimmune diseases", "autism", "strokes",
]
FOODS = ["gluten", "dairy", "refined sugar", "aspartame", "food additives"]
SHORT_DURATIONS = ["3 days", "a week", "48 hours", "a month", "a few days"]

PARTIES = ["the Democrats", "the Republicans", "the GOP", "the DNC", "Congress"]
POLITICAL_PERSONALITIES = [
    "Biden", "Trump", "a senator", "a congressman",
    "the Vice President", "the Secretary of State", "the CDC director",
]
ORGANIZATIONS = [
    "Goldman Sachs", "BlackRock", "NATO", "the WEF",
    "pharmaceutical lobbies", "multinational corporations",
]
ELECTIONS = ["the presidential election", "the midterms", "the primaries", "the vote"]
AMOUNTS = ["$500,000", "$2 million", "a bribe", "millions in dark money"]
POLITICAL_SUBJECTS = ["the reform", "the budget", "foreign policy", "healthcare"]
ACCUSATIONS = [
    "has hidden offshore accounts", "received illegal funding",
    "intentionally broke campaign promises", "secretly negotiated with lobbyists",
]


# ================================================================
#  RELIABLE TEMPLATES (label=0)
# ================================================================

NEWS = [
    "The {reliable_institution} announces {announcement} for {period}",
    "{reliable_institution} publishes its report on {reliable_subject}",
    "The results of the {event} are in: {result}",
    "{city} hosts the {city_event} this weekend",
    "The weather forecast calls for {weather} next week",
    "The unemployment rate {trend} to {percentage} last quarter",
    "New bus route inaugurated in {city}",
    "The {reliable_institution} approves {reliable_decision}",
    "{sports_team} beats {opponent} {score} in the {competition}",
    "The holiday market in {city} opens its doors Saturday",
    "Registration for {edu_event} is now open",
    "Record attendance at the {venue} with {number} visitors",
    "The {reliable_institution} launches a public consultation on {reliable_subject}",
    "{reliable_personality} visits {city} as part of {context}",
    "Road work on {road}: detours expected until {month}",
]

SCIENCE_RELIABLE = [
    "A study by {lab} shows that {science_discovery}",
    "Researchers at {lab} discover {science_discovery}",
    "{science_journal} publishes a study on {science_subject}",
    "New data on {science_subject}: {science_conclusion}",
    "{lab} confirms {science_conclusion}",
    "International conference on {science_subject} in {city}",
    "Major breakthrough in {science_field} according to {lab}",
    "The Nobel Prize in {science_field} awarded to {reliable_personality}",
    "Phase 3 clinical trial: {clinical_result} results for {reliable_treatment}",
    "The WHO recommends {health_recommendation} based on latest data",
]

DAILY_LIFE = [
    "Beautiful day in {city}, perfect for a walk",
    "Anyone know a good {shop} in {city}?",
    "Sales start next week, any good deals?",
    "Amazing concert by {artist} last night at the {venue}",
    "First day of vacation, heading to {destination}!",
    "Happy birthday to my little one, already 5!",
    "Back to school Monday, good luck to all the parents",
    "Intense game tonight, go {sports_team}!",
    "Terrible traffic on {road} this morning",
    "The new {product} is really great, highly recommend",
    "Heavy rain in {city} today, grab an umbrella",
    "Just started my new job in {city}, so excited",
    "The farmers market in {city} is amazing this morning",
    "Must-see movie: {movie} in theaters this week",
    "Just finished a great book, anyone need recommendations?",
    "Coffee and sunshine, perfect Saturday morning",
    "The local library has a great new collection",
    "Trying a new recipe tonight, wish me luck",
    "The park in {city} is gorgeous this time of year",
    "Great podcast episode about {reliable_subject} today",
]

MEASURED_OPINION = [
    "I think {opinion_subject} deserves more attention",
    "In my opinion, {opinion_subject} is a complex topic that needs nuance",
    "Interesting article about {opinion_subject} in {reliable_newspaper}",
    "The debate about {opinion_subject} is necessary but not simple",
    "I disagree with {political_personality} on {opinion_subject} but respect the position",
    "We need more data before drawing conclusions about {opinion_subject}",
    "Good summary of the situation on {opinion_subject} by {reliable_newspaper}",
    "The {reliable_institution} report on {opinion_subject} is really informative",
    "We should listen more to experts on {opinion_subject}",
    "The question of {opinion_subject} has no simple answer",
]

# --- Variables for reliable templates ---
RELIABLE_INSTITUTIONS = [
    "Census Bureau", "NIH", "Senate", "Supreme Court", "GAO",
    "Department of Health", "Department of Education", "FDA",
    "CDC", "NASA", "NOAA", "EPA", "city council",
    "state legislature", "county board", "school board",
]

ANNOUNCEMENTS = [
    "a budget increase", "a new investment plan",
    "support measures", "enhanced oversight",
    "a system reform", "the creation of 500 jobs",
    "an energy renovation plan", "the launch of a pilot program",
]

PERIODS = ["2026", "next quarter", "the fall semester", "the next 5 years", "this summer"]

RELIABLE_SUBJECTS = [
    "employment", "public health", "education", "housing",
    "transportation", "the environment", "the economy", "public safety",
    "arts and culture", "technology", "renewable energy",
]

EVENTS = [
    "SAT 2026", "the state championship", "the Super Bowl",
    "the US Open", "the NBA finals", "the midterm elections",
    "the film festival", "the science fair",
]

RESULTS = [
    "pass rates are up", "a new record",
    "encouraging results", "higher participation",
]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Austin",
    "Seattle", "Denver", "Boston", "Portland", "Nashville",
    "San Francisco", "Atlanta", "Miami", "Philadelphia", "Minneapolis",
]

WEATHER = [
    "sunshine", "rain", "cloudy skies", "thunderstorms",
    "mild temperatures", "a cold snap", "beautiful spring weather",
]

TRENDS = ["dropped", "remained stable", "rose slightly", "declined"]
PERCENTAGES = ["3.8%", "4.2%", "3.5%", "4.0%", "3.6%"]

SPORTS_TEAMS = [
    "the Lakers", "the Yankees", "the Patriots", "the Warriors",
    "Team USA", "the Eagles", "the Dodgers", "the Chiefs",
]
OPPONENTS = [
    "the Celtics", "the Red Sox", "the 49ers", "the Nets",
    "the Mavericks", "the Cubs", "the Bucks", "the Rams",
]
SCORES = ["112-98", "3-1", "24-17", "4-2", "2-1", "28-21"]
COMPETITIONS = ["NBA", "NFL", "MLB", "World Series", "Champions League"]

CITY_EVENTS = ["book fair", "music festival", "holiday market", "science expo"]
EDU_EVENTS = ["evening classes", "the entrance exam", "the summer program"]
VENUES = ["museum", "convention center", "arena", "state park", "stadium"]
NUMBERS = ["50,000", "100,000", "25,000", "200,000"]

RELIABLE_DECISIONS = [
    "the new health protocol", "the infrastructure bill",
    "the climate action plan 2030", "the annual budget",
]

RELIABLE_PERSONALITIES = [
    "a research team", "the NIH director", "the governor",
    "the mayor", "a professor of medicine", "the university president",
]

CONTEXTS = ["a national tour", "the G7 summit", "the awareness campaign"]
ROADS = ["I-95", "Route 66", "the highway", "Interstate 405", "the expressway", "Main Street"]
MONTHS = ["June", "September", "December", "March", "January"]

LABS = ["MIT", "Stanford", "Johns Hopkins", "Caltech", "Harvard", "Mayo Clinic"]
SCIENCE_JOURNALS = ["Nature", "The Lancet", "PNAS", "Science", "NEJM"]
SCIENCE_SUBJECTS = [
    "climate change", "gene therapy", "immunology",
    "neuroscience", "quantum physics", "biodiversity",
    "renewable energy", "artificial intelligence",
]
SCIENCE_DISCOVERIES = [
    "a new cellular mechanism", "a significant correlation",
    "a promising biomarker", "a new species",
    "a revolutionary material", "a more efficient algorithm",
]
SCIENCE_CONCLUSIONS = [
    "encouraging results", "significant progress",
    "an important breakthrough", "promising data",
]
SCIENCE_FIELDS = ["Medicine", "Physics", "Chemistry", "Biology", "Economics", "Computer Science"]
CLINICAL_RESULTS = ["positive", "encouraging", "promising", "significant"]
RELIABLE_TREATMENTS = [
    "the new vaccine", "immunotherapy", "the antiviral treatment",
    "gene therapy", "the experimental protocol",
]
HEALTH_RECOMMENDATIONS = [
    "annual vaccination", "30 minutes of daily exercise",
    "reducing sodium intake", "regular screening",
]

SHOPS = ["restaurant", "barber", "bakery", "mechanic", "dentist"]
ARTISTS = [
    "Taylor Swift", "Beyonce", "Kendrick Lamar", "Billie Eilish", "Ed Sheeran", "Adele",
]
DESTINATIONS = ["the mountains", "the coast", "Florida", "Colorado", "the lake"]
PRODUCTS = ["MacBook", "Galaxy S26", "robot vacuum", "electric bike"]
MOVIES = ["the new Nolan film", "the Marvel movie", "the Pixar movie", "the Oscar winner"]

OPINION_SUBJECTS = [
    "Social Security reform", "nuclear energy", "immigration policy",
    "remote work", "public education", "the energy transition",
    "social media regulation", "housing prices", "income inequality",
]
RELIABLE_NEWSPAPERS = ["The New York Times", "The Washington Post", "NPR", "Reuters", "AP News"]


def fill_suspect():
    """Generate a random suspect post."""
    templates = CONSPIRACIES + MANIPULATION + PSEUDO_HEALTH + POLITICAL_SUSPECT
    template = random.choice(templates)

    mappings = {
        '{substance}': random.choice(SUBSTANCES),
        '{objective}': random.choice(OBJECTIVES),
        '{conspiracy_subject}': random.choice(CONSPIRACY_SUBJECTS),
        '{technology}': random.choice(TECHNOLOGIES),
        '{tech_objective}': random.choice(TECH_OBJECTIVES),
        '{institution}': random.choice(INSTITUTIONS),
        '{personality}': random.choice(PERSONALITIES),
        '{revelation}': random.choice(REVELATIONS),
        '{stats_subject}': random.choice(STATS_SUBJECTS),
        '{censored_subject}': random.choice(CENSORED_SUBJECTS),
        '{alert_subject}': random.choice(ALERT_SUBJECTS),
        '{shocking_revelation}': random.choice(SHOCKING_REVELATIONS),
        '{scandal}': random.choice(SCANDALS),
        '{duration}': random.choice(DURATIONS),
        '{media}': random.choice(MEDIA),
        '{manipulation_subject}': random.choice(MANIPULATION_SUBJECTS),
        '{disease}': random.choice(DISEASES),
        '{remedy}': random.choice(REMEDIES),
        '{treatment}': random.choice(TREATMENTS),
        '{side_effect}': random.choice(SIDE_EFFECTS),
        '{food}': random.choice(FOODS),
        '{short_duration}': random.choice(SHORT_DURATIONS),
        '{party}': random.choice(PARTIES),
        '{political_personality}': random.choice(POLITICAL_PERSONALITIES),
        '{organization}': random.choice(ORGANIZATIONS),
        '{election}': random.choice(ELECTIONS),
        '{amount}': random.choice(AMOUNTS),
        '{political_subject}': random.choice(POLITICAL_SUBJECTS),
        '{accusation}': random.choice(ACCUSATIONS),
    }

    text = template
    for k, v in mappings.items():
        text = text.replace(k, v)
    return text


def fill_reliable():
    """Generate a random reliable post."""
    templates = NEWS + SCIENCE_RELIABLE + DAILY_LIFE + MEASURED_OPINION
    template = random.choice(templates)

    mappings = {
        '{reliable_institution}': random.choice(RELIABLE_INSTITUTIONS),
        '{announcement}': random.choice(ANNOUNCEMENTS),
        '{period}': random.choice(PERIODS),
        '{reliable_subject}': random.choice(RELIABLE_SUBJECTS),
        '{event}': random.choice(EVENTS),
        '{result}': random.choice(RESULTS),
        '{city}': random.choice(CITIES),
        '{city_event}': random.choice(CITY_EVENTS),
        '{weather}': random.choice(WEATHER),
        '{trend}': random.choice(TRENDS),
        '{percentage}': random.choice(PERCENTAGES),
        '{sports_team}': random.choice(SPORTS_TEAMS),
        '{opponent}': random.choice(OPPONENTS),
        '{score}': random.choice(SCORES),
        '{competition}': random.choice(COMPETITIONS),
        '{reliable_decision}': random.choice(RELIABLE_DECISIONS),
        '{reliable_personality}': random.choice(RELIABLE_PERSONALITIES),
        '{context}': random.choice(CONTEXTS),
        '{road}': random.choice(ROADS),
        '{month}': random.choice(MONTHS),
        '{edu_event}': random.choice(EDU_EVENTS),
        '{venue}': random.choice(VENUES),
        '{number}': random.choice(NUMBERS),
        '{lab}': random.choice(LABS),
        '{science_journal}': random.choice(SCIENCE_JOURNALS),
        '{science_subject}': random.choice(SCIENCE_SUBJECTS),
        '{science_discovery}': random.choice(SCIENCE_DISCOVERIES),
        '{science_conclusion}': random.choice(SCIENCE_CONCLUSIONS),
        '{science_field}': random.choice(SCIENCE_FIELDS),
        '{clinical_result}': random.choice(CLINICAL_RESULTS),
        '{reliable_treatment}': random.choice(RELIABLE_TREATMENTS),
        '{health_recommendation}': random.choice(HEALTH_RECOMMENDATIONS),
        '{shop}': random.choice(SHOPS),
        '{artist}': random.choice(ARTISTS),
        '{destination}': random.choice(DESTINATIONS),
        '{product}': random.choice(PRODUCTS),
        '{movie}': random.choice(MOVIES),
        '{opinion_subject}': random.choice(OPINION_SUBJECTS),
        '{reliable_newspaper}': random.choice(RELIABLE_NEWSPAPERS),
        '{political_personality}': random.choice(POLITICAL_PERSONALITIES),
    }

    text = template
    for k, v in mappings.items():
        text = text.replace(k, v)
    return text


# --- Social media style variations ---
def add_social_style(text, label):
    """Add social media style variations for realism."""
    r = random.random()

    if label == 1:  # suspect
        if r < 0.2:
            text = text.upper()
        elif r < 0.4:
            words = text.split()
            words = [w.upper() if random.random() < 0.3 else w for w in words]
            text = ' '.join(words)
        if random.random() < 0.3:
            text = text.rstrip('!.') + random.choice([' !!!', ' !!!!', ' !!!!!'])
        if random.random() < 0.15:
            text = random.choice(['THREAD: ', 'INFO: ', 'ALERT: ', 'URGENT: ']) + text
    else:  # reliable
        if r < 0.1:
            text = text.lower()
        if random.random() < 0.1:
            text += random.choice([' :)', ' !', ' <3', '...'])
        if random.random() < 0.05:
            text = random.choice(['BTW, ', 'Also, ', 'Hey, ']) + text[0].lower() + text[1:]

    return text


def main():
    """Generate the dataset and save to CSV."""
    n_suspect = 5000
    n_reliable = 5000

    rows = []

    # Generate suspect posts
    print(f"Generating {n_suspect} suspect posts...")
    for _ in range(n_suspect):
        text = fill_suspect()
        text = add_social_style(text, label=1)
        rows.append({'text': text, 'label': 1, 'source': 'synthetic_en_social'})

    # Generate reliable posts
    print(f"Generating {n_reliable} reliable posts...")
    for _ in range(n_reliable):
        text = fill_reliable()
        text = add_social_style(text, label=0)
        rows.append({'text': text, 'label': 0, 'source': 'synthetic_en_social'})

    # Shuffle
    random.shuffle(rows)

    # Stats
    texts = [r['text'] for r in rows]
    word_counts = [len(t.split()) for t in texts]
    short = sum(1 for wc in word_counts if wc < 15)
    medium = sum(1 for wc in word_counts if 15 <= wc < 30)
    long_ = sum(1 for wc in word_counts if wc >= 30)

    print(f"\nDataset generated: {len(rows)} posts")
    print(f"  Suspect  : {sum(1 for r in rows if r['label'] == 1)}")
    print(f"  Reliable : {sum(1 for r in rows if r['label'] == 0)}")
    print(f"  < 15 words  : {short} ({100*short/len(rows):.1f}%)")
    print(f"  15-30 words : {medium} ({100*medium/len(rows):.1f}%)")
    print(f"  > 30 words  : {long_} ({100*long_/len(rows):.1f}%)")
    print(f"  Average length: {sum(word_counts)/len(word_counts):.1f} words")

    # Save
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label', 'source'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Size: {os.path.getsize(OUTPUT_FILE) / 1024:.0f} KB")

    # Examples
    print("\n--- Suspect examples ---")
    for r in [r for r in rows if r['label'] == 1][:5]:
        print(f"  [SUSPECT]  {r['text'][:80]}")

    print("\n--- Reliable examples ---")
    for r in [r for r in rows if r['label'] == 0][:5]:
        print(f"  [RELIABLE] {r['text'][:80]}")


if __name__ == '__main__':
    main()
