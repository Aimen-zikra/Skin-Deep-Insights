# Skin-Deep Insights
### Aspect-Based Sentiment Analysis of Sephora Skincare Reviews

**Business Problem:** Many highly rated beauty products (4★+) still experience hidden dissatisfaction. Star ratings alone do not reveal which specific product aspects — packaging, scent, skin reactions, texture — are causing problems. This project surfaces those patterns using NLP on 1,033,710 reviews.

---

## Core Finding

> **4.19% of reviews rated 4★ or higher contain negative written sentiment** — representing approximately **37,650 hidden complaints** inside praised products across the full dataset. These complaints are structurally invisible to any analysis relying on star ratings alone.

---

## Key Findings

| Finding | Value | Business Implication |
|---|---|---|
| Positive rating skew | 81.4% are 4★–5★ | Ratings are unreliable quality signals |
| Hidden dissatisfaction | 4.19% of 4★+ reviews negative | ~37,650 complaints invisible to rating analysis |
| Packaging complaint rate | 11.5% negative (highest) | 1 in 9 packaging mentions is a complaint |
| Skin reaction hidden rate | 5.4% in 4★+ reviews | Most dangerous signal — health-adjacent |
| Texture hidden volume | 3,488 complaints in 4★+ | Highest raw count — affects all skin types |
| Oily skin segment | 12% of users, outsized complaints | Commercially underserved segment |
| Scent complaint rate | 5.7% negative (lowest) | Confirmed brand strength — amplify in marketing |
| Price vs satisfaction | r = 0.0036 | Price tier does not predict product quality |

---

## Technical Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | pandas · numpy |
| Visualization | matplotlib · seaborn |
| NLP | VADER Sentiment (vaderSentiment) · scikit-learn (TF-IDF, CountVectorizer) |
| Analysis | Aspect-Based Sentiment Analysis · keyword extraction |
| Validation | Pearson correlation · sample representativeness testing |
| Environment | Jupyter Notebook |

---

## Project Structure

```
beauty-absa/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb    Dataset exploration & ABSA viability
│   ├── 02_data_extraction.ipynb       Column selection & working dataset creation
│   ├── 03_data_cleaning.ipynb         NLP-specific cleaning pipeline
│   ├── 04_eda.ipynb                   Exploratory data analysis
│   ├── 05_deep_analysis.ipynb         ABSA core analysis (VADER + TF-IDF + aspects)
│   ├── 06_validation.ipynb            5-check validation framework
│   ├── 07_insights.ipynb              Business insight generation
│   ├── 08_recommendations.ipynb       Evidence-backed recommendations
│   └── 09_visualization.ipynb         12-chart visualization suite
│
├── outputs/
│   ├── charts/                        All 12 chart PNGs
│   └── insights/
│
├── requirements.txt
└── README.md
```

> **Note:** Notebooks 01–09 correspond to Stages 3–11 of the analytics framework. Stages 1–2 (business problem definition and success metrics) were completed as pre-analysis planning documents before notebook work began.

---

## Methodology

### 12-Stage Analytics Framework

| Stage | Description |
|---|---|
| 1 | Define business problem |
| 2 | Define success metrics |
| 3 | Data understanding |
| 4 | Data extraction |
| 5 | Data cleaning |
| 6 | Exploratory data analysis |
| **7** | **Deep analysis — ABSA (core stage)** |
| 8 | Validation |
| 9 | Insight generation |
| 10 | Recommendations |
| 11 | Visualization |
| 12 | Presentation |

### Aspect Extraction — How It Works

**Step 1 — TF-IDF complaint word discovery**

Run on 1★ vs 5★ reviews to find statistically distinctive complaint words from data — not assumptions.

Top TF-IDF findings: `break` (+0.026) · `smell` (+0.017) · `sensitive` (+0.012) · `acne` (+0.009) · `sticky` (+0.009) · `bottle` (+0.007)

**Step 2 — Data-driven keyword dictionary**

Built directly from TF-IDF findings above, not manually defined:

```python
aspect_keywords = {
    'packaging':     ['pump', 'bottle', 'break', 'broke', 'leaked', 'cap', 'tube'],
    'skin_reaction': ['breakout', 'acne', 'irritation', 'rash', 'sensitive', 'reaction'],
    'scent':         ['smell', 'scent', 'fragrance', 'odor', 'perfume'],
    'texture':       ['texture', 'thick', 'greasy', 'sticky', 'watery', 'heavy'],
    'hydration':     ['hydrating', 'moisturizing', 'dry', 'moisture', 'hydration'],
    'price':         ['expensive', 'cheap', 'worth', 'overpriced', 'value']
}
```

**Step 3 — VADER sentiment scoring**

Every review is scored independently of aspect detection. Validated: Pearson r = 0.4544 between VADER compound scores and star ratings — confirming a healthy independent signal range (neither duplicating ratings nor uncorrelated noise).

**VADER limitation:** As a lexicon-based model, VADER can mishandle negation ("not irritating") and sarcasm. This affects an estimated small minority of reviews. For this project's purpose — identifying complaint patterns at scale — the aggregate findings are robust, as validated in Stage 8.

**Sample note:** ABSA was run on a 300,000-review stratified sample. At 1M+ rows, full-dataset NLP processing requires significant compute. The sample was validated for representativeness (max deviation from full dataset: 0.09%) before analysis.

---

## Data Cleaning Pipeline

| Step | Action | Records Removed | Remaining |
|---|---|---|---|
| 1 | Remove null `review_text` | 1,444 | 1,092,967 |
| 2 | Remove ingestion duplicates | 650 | 1,092,317 |
| 3 | Remove short reviews (<5 words) | 1,680 | 1,090,637 |
| 4 | Remove incentivized reviews | ~56,927 | 1,033,710 |
| **Total** | | **60,701 (5.5%)** | **1,033,710** |

**Deduplication logic:** Only records where `product_id + review_text + rating + review_date` were all identical were removed. Repeated review text across different users or products was preserved as authentic sentiment signal. This conservative approach avoids the common mistake of treating natural language repetition as a data error.

---

## Validation Results

| Check | Test | Result | Status |
|---|---|---|---|
| VADER alignment | Negative% decreases monotonically as rating rises | 44.3% (1★) → 3.3% (5★) | PASS |
| Aspect coverage | Reviews with at least one aspect detected | 80.8% (threshold: >75%) | PASS |
| Sentiment correlation | Pearson r vs star rating | r = 0.4544 (range: 0.40–0.70) | PASS |
| Sample representativeness | Max deviation vs full dataset distribution | 0.09% (threshold: <1%) | PASS |
| Business logic | Manual review of flagged aspect-sentiment records | Labels accurate | PASS |

---

## Core Results

### Aspect Sentiment (300k sample)

| Aspect | Mentions | Negative % | Positive % |
|---|---|---|---|
| packaging | 61,315 | **11.5%** | 86.5% |
| skin_reaction | 74,723 | 10.1% | 88.0% |
| price | 46,752 | 7.8% | 90.7% |
| texture | 125,969 | 7.8% | 90.3% |
| hydration | 101,187 | 6.2% | 91.7% |
| scent | 74,495 | 5.7% | 93.5% |

### Hidden Dissatisfaction in 4★+ Reviews

| Aspect | Hidden Rate | Hidden Count | Why It Matters |
|---|---|---|---|
| skin_reaction | 5.4% | 3,292 | Health-adjacent — most dangerous signal |
| packaging | 4.8% | 2,201 | Operational fix available without reformulation |
| texture | 3.5% | **3,488** | Highest raw volume — universal across skin types |
| hydration | 3.4% | 2,881 | Core product promise — if failing, retention risk |
| price | 3.5% | 1,249 | Luxury tier perception gap |
| scent | 2.3% | 1,394 | Lowest — confirmed strength |

---

## 6 Recommendations

| # | Recommendation | Evidence | Priority | Timeline |
|---|---|---|---|---|
| R1 | Packaging quality control audit — focus on pump and dispenser mechanisms | 11.5% complaint rate · 2,201 hidden complaints in 4★+ reviews | HIGH | 30–90 days |
| R2 | Oily skin formulation review — reformulate mid-tier products for oily/combination skin | 12% of users, disproportionate complaint volume across skin_reaction and texture | HIGH | 60–180 days |
| R3 | Texture alignment — reconcile formula texture with marketing claims | 3,488 hidden texture complaints in 4★+ — highest raw volume of any aspect | HIGH | 30–60 days |
| R4 | Luxury value differentiation — strengthen value narrative for $100+ products | Price complaint rate 7.8% · r = 0.0036 (price doesn't predict satisfaction) | MEDIUM | 45–90 days |
| R5 | Repeat ABSA quarterly to track complaint trend direction | Baseline established — only trend data will show if interventions work | HIGH | Ongoing |
| R6 | Amplify scent and hydration messaging in marketing | Lowest complaint rates (5.7% / 6.2%) — these are confirmed product strengths | MEDIUM | 14 days |

---

## Dataset

**Source:** [Sephora Products and Skincare Reviews — Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

- Raw reviews: 1,094,411 rows · 21 columns
- After cleaning: 1,033,710 rows · 8 columns
- Products: 2,351 · Brands: 142 · Date range: 2008–2023

Data files are not tracked due to size. Download from Kaggle and place in `data/raw/`.

---

## Installation

```bash
git clone https://github.com/Aimen-zikra/Skin-Deep-Insights.git
cd Skin-Deep-Insights
pip install -r requirements.txt
```

**requirements.txt**
```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
vaderSentiment>=3.3.2
jupyter>=1.0
```

Run notebooks in order: `01_data_understanding` through `09_visualization`.

---

## Business Context

This project mirrors the kind of analysis performed internally at beauty and skincare companies — using review text to surface product signals that aggregate ratings cannot detect.

**Stakeholders addressed:** Product Development · Customer Experience · Brand Managers · Marketing Strategy

**Decisions supported:** Reformulate? Redesign packaging? Adjust marketing claims? Which segments are underserved? Are premium products delivering on their price promise?

The methodology is transferable to any review-based product domain: e-commerce, SaaS, hospitality, consumer electronics.

---

## Author

**Aimen Zikra** — Data Analyst  
Python · SQL · NLP · Data Visualization  
[GitHub](https://github.com/Aimen-zikra) 
