# Skin-Deep Insights
## Aspect-Based Sentiment Analysis of Sephora Skincare Reviews

> **Business Problem:** Many highly rated beauty products (4★+) still experience high return rates and hidden dissatisfaction. Star ratings alone do not reveal which specific product aspects — packaging, scent, irritation — are causing problems.

---

## Project Overview

This project builds an end-to-end **Aspect-Based Sentiment Analysis (ABSA)** pipeline on 1,033,710 Sephora skincare reviews. Using a 12-stage consulting analytics framework, it surfaces hidden dissatisfaction patterns that aggregate star ratings cannot detect — and translates those patterns into specific, evidence-backed business recommendations.

**The core finding:** 4.19% of reviews rated 4★ or higher contain negative written sentiment — representing approximately **37,650 hidden complaints** inside praised products across the full dataset.

---

## Key Findings

| Finding | Value | Business Implication |
|---|---|---|
| Positive rating skew | 81.4% are 4★–5★ | Ratings are unreliable quality signals |
| Hidden dissatisfaction | 4.19% of 4★+ reviews | ~37,650 complaints invisible to rating analysis |
| Packaging complaint rate | 11.5% negative | Highest rate — 1 in 9 mentions is a complaint |
| Skin reaction hidden rate | 5.4% in 4★+ | Most dangerous signal — health-adjacent |
| Texture hidden volume | 3,488 complaints in 4★+ | Highest raw count — universal across skin types |
| Oily skin | 12% of data, outsized complaints | Commercially underserved segment |
| Scent complaint rate | 5.7% negative | Lowest — confirmed brand strength |
| Price vs rating correlation | r = 0.0036 | Price does not predict satisfaction |

---

## Technical Stack

```
Python 3.13     pandas · numpy · matplotlib · seaborn
NLP             VADER Sentiment · scikit-learn (TF-IDF, CountVectorizer)
Analysis        Aspect-Based Sentiment Analysis · keyword extraction
Validation      Pearson correlation · sample representativeness testing
```

---

## Project Structure

```
beauty-absa/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb   Stage 3  — Dataset exploration & viability
│   ├── 02_data_extraction.ipynb      Stage 4  — Column selection & working dataset
│   ├── 03_data_cleaning.ipynb        Stage 5  — NLP-specific cleaning pipeline
│   ├── 04_eda.ipynb                  Stage 6  — Exploratory data analysis
│   ├── 05_deep_analysis.ipynb        Stage 7  — ABSA core analysis
│   ├── 06_validation.ipynb           Stage 8  — 5-check validation framework
│   ├── 07_insights.ipynb             Stage 9  — Business insight generation
│   ├── 08_recommendations.ipynb      Stage 10 — Evidence-backed recommendations
│   └── 09_visualization.ipynb        Stage 11 — 12-chart visualization suite
│
├── outputs/
│   ├── charts/                       All 12 chart PNGs
│   └── insights/
│
└── requirements.txt
```

---

## Methodology

### 12-Stage Analytics Framework

```
Stage 1   Define business problem
Stage 2   Define success metrics
Stage 3   Data understanding
Stage 4   Data extraction
Stage 5   Data cleaning
Stage 6   EDA
Stage 7   Deep analysis (ABSA)   ← core
Stage 8   Validation
Stage 9   Insight generation
Stage 10  Recommendations
Stage 11  Visualization
Stage 12  Presentation
```

### Aspect Extraction — How It Works

**Step 1 — TF-IDF complaint discovery**
Run on 1★ vs 5★ reviews to find statistically distinctive complaint words from data, not assumptions.

Top findings: `break (+0.026)` · `smell (+0.017)` · `sensitive (+0.012)` · `acne (+0.009)` · `sticky (+0.009)` · `bottle (+0.007)`

**Step 2 — Data-driven keyword dictionary**
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
Validated: Pearson r = 0.4544 between VADER scores and star ratings — healthy independent signal range.

---

## Data Cleaning Pipeline

| Step | Action | Removed | Remaining |
|---|---|---|---|
| 1 | Remove null review_text | 1,444 | 1,092,967 |
| 2 | Remove ingestion duplicates | 650 | 1,092,317 |
| 3 | Remove short reviews (<5 words) | 1,680 | 1,090,637 |
| 4 | Remove incentivized reviews | ~56,927 | **1,033,710** |

Total reduction: 5.5% — surgical, not aggressive.

**Deduplication logic:** Only records where `product_id + review_text + rating + review_date` were all identical were removed. Repeated review text across different users was preserved as authentic sentiment signal.

---

## Validation Results

| Check | Test | Result | Status |
|---|---|---|---|
| VADER Alignment | Neg% decreases as rating rises | 44.3% (1★) → 3.3% (5★) | PASS |
| Aspect Coverage | Reviews with no aspect detected | 19.2% | PASS (<25% threshold) |
| Sentiment Correlation | Pearson r vs star rating | r = 0.4544 | PASS (0.40–0.70) |
| Sample Representativeness | Max deviation vs full dataset | 0.09% | PASS (<1%) |
| Business Logic | Manual review of flagged records | Labels accurate | PASS |

---

## Core Results

### Aspect Sentiment (300k sample)

| Aspect | Mentions | Negative % | Positive % |
|---|---|---|---|
| packaging | 61,315 | **11.5%** | 86.5% |
| skin_reaction | 74,723 | **10.1%** | 88.0% |
| price | 46,752 | 7.8% | 90.7% |
| texture | 125,969 | 7.8% | 90.3% |
| hydration | 101,187 | 6.2% | 91.7% |
| scent | 74,495 | 5.7% | **93.5%** |

### Hidden Dissatisfaction in 4★+ Reviews

| Aspect | Hidden Rate | Hidden Count |
|---|---|---|
| skin_reaction | **5.4%** | 3,292 |
| packaging | **4.8%** | 2,201 |
| texture | 3.5% | **3,488** (highest volume) |
| hydration | 3.4% | 2,881 |
| price | 3.5% | 1,249 |
| scent | 2.3% | 1,394 |

---

## 6 Recommendations

| # | Recommendation | Priority | Timeline |
|---|---|---|---|
| R1 | Packaging quality control audit | HIGH | 30–90 days |
| R2 | Oily skin formulation review | HIGH | 60–180 days |
| R3 | Texture alignment — formula + marketing | HIGH | 30–60 days |
| R4 | Luxury value differentiation | MEDIUM | 45–90 days |
| R5 | Repeat ABSA quarterly to track complaint trends | HIGH | Ongoing |
| R6 | Amplify scent + hydration in marketing | MEDIUM | 14 days |

---

## Dataset

**Source:** [Sephora Products and Skincare Reviews — Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

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
```

Run notebooks in order: `01` through `09`.

---

## Business Context

This project mirrors the kind of analysis companies like Sephora, L'Oreal, and Perfect Corp perform internally — using review text to surface product signals that aggregate ratings cannot detect.

**Stakeholders:** Product Development · Customer Experience · Brand Managers · Marketing Strategy

**Decisions supported:** Reformulate? Redesign packaging? Adjust marketing claims? Which segments are underserved? Are premium products delivering on their price?
