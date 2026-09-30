# 🍽️ Customer Recommendation Engine

> Predicts which restaurants customers are most likely to order from — Random Forest with feature engineering. **84.84% validation accuracy.**

## The Problem
Given customer profiles, locations and order history, recommend the vendor (restaurant) each customer is most likely to order from next.

## The Data
- **34,674** customers in train, **9,768** in test
- Tables: customers (demographics, account status), locations (masked lat/long, home/work/other), orders (items, totals, promos, ratings, delivery timings), vendors (100 vendors with tags and masked locations)

## Approach
1. **EDA** — order frequency by vendor, customer-location patterns, promo and rating effects.
2. **Feature engineering** — customer × location × vendor interaction features (`CID X LOC_NUM X VENDOR`), aggregated order history per customer-vendor pair, distance proxies from masked coordinates, recency/frequency features.
3. **Modeling** — Random Forest classifier; train/validation split with stratification.
4. **Evaluation** — validation accuracy on held-out customers.

## Key Results
- **Validation accuracy: 84.84%**
- Strongest signals came from customer–vendor order history features and location affinity.

## Tech Stack
Python · Pandas · NumPy · Scikit-learn (Random Forest) · Matplotlib · Seaborn · Jupyter

## Project Structure
```
├── Main.ipynb        # Full workflow: EDA → features → model → evaluation
├── Train/            # Training data
├── Test/             # Test data
└── submission.csv    # Sample submission format
```

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Main.ipynb
```
