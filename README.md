# Expedia Hotel Recommendations

**MSc Data Science — University of Nicosia**

## About

This project builds a hotel recommendation system that predicts which hotel a user will click next based on their browsing session. The idea is simple: instead of just looking at the last hotel someone clicked (which is what standard Word2Vec approaches do), we look at the last few hotels they browsed and average their embeddings to get a better sense of what they're looking for.

The dataset comes from the [Expedia PKDD 2022 Challenge](https://github.com/ExpediaGroup/pkdd22-challenge-expediagroup/releases) — real user session data from Expedia's platform with ~1M click sequences across thousands of hotels.

## The Approach

We train Word2Vec (skip-gram) on hotel click sequences, treating each hotel ID like a "word" and each user session like a "sentence." This gives us embeddings where similar hotels end up close together in vector space.

Then instead of recommending hotels similar to just the last click, we average the embeddings of the last *w* hotels (context window) to capture the user's overall intent within a session. A user browsing luxury beachfront properties probably wants more of the same — not a random budget downtown option.

### Methods Compared

- **Popularity Baseline** — just recommend the most-clicked hotels globally
- **Markov Chain** — predict based on single-step transition probabilities
- **Word2Vec-Last** — recommend hotels most similar to the last clicked hotel
- **Word2Vec-Context (ours)** — average last *w* hotel embeddings, then find similar hotels

## Results

| Method | Hit Rate @10 | Hit Rate @100 |
|--------|-------------|---------------|
| Popularity | 0.32% | 1.85% |
| Markov Chain | 4.72% | 6.42% |
| Word2Vec-Last (dim=32) | 3.22% | 12.14% |
| **Word2Vec-Context (dim=32, w=3)** | **4.56%** | **14.22%** |

The context window approach gives a **+2.08% absolute improvement** over the standard single-hotel baseline at K=100. Dimension 32 with a window of 3 hotels turned out to be the sweet spot — larger dimensions (64, 100) overfit on the training data, and larger windows (5+) start adding noise from older browsing history.

### Why It Works

Users tend to be locally consistent within a session. If someone is looking at 4-star hotels near the beach, averaging those embeddings pulls the recommendation vector toward that "cluster" of similar hotels. It smooths out random clicks while keeping the signal about what the user actually wants.

## Dataset

- **Source:** Expedia PKDD 2022 Challenge
- **Size:** 100K sessions (10% sample), 67,322 unique hotels
- **Avg session length:** 4.2 hotels
- **Split:** 80K train / 20K test

## How to Run

1. Download `train.tsv.gz` from the [dataset releases page](https://github.com/ExpediaGroup/pkdd22-challenge-expediagroup/releases)
2. Place it in the same directory as the script
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the full experiment:
   ```bash
   python recommendations.py
   ```
   This trains all model configurations and evaluates them. Takes about 40 minutes.

## Files

| File | What it is |
|------|-----------|
| `recommendations.py` | All the code — data loading, model training, evaluation, results |
| `hotel_recommendations.tex` | LaTeX source for the project report |
| `Project.pdf` | The compiled report |
| `presentation.pptx` | Presentation slides |
| `requirements.txt` | Python dependencies |

## Configuration

The script tests all combinations of:
- **Embedding dimensions:** 16, 32, 64, 100
- **Context window sizes:** 1 (baseline), 2, 3, 5
- **K values:** 10, 100

## Requirements

- Python 3.8+
- pandas
- gensim (Word2Vec)
- numpy
- tqdm
