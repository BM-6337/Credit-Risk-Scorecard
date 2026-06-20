# Credit Scorecard — Lending Club Loan Default Model

A classic credit-risk scorecard built on the [Lending Club loan dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club). The notebook bins every predictor, computes Weight of Evidence (WOE) and Information Value (IV), transforms variables into their WOE-encoded form, and fits both a logistic regression and a decision tree to predict loan default.

## What it does

1. Loads raw Lending Club loan data and keeps only loans with a resolved outcome (`Fully Paid`, `Charged Off`, `Default`)
2. Engineers a binary target, `Late_Loan` (1 = defaulted, 0 = fully paid)
3. Drops sparse columns (>10% missing) and any column that leaks the outcome (e.g. `total_pymnt`, `recoveries`) or carries no predictive intuition (e.g. `id`, `url`)
4. Bins every remaining predictor and computes WOE / IV per bin
5. Substitutes each raw variable with its WOE-encoded equivalent
6. Fits a logistic regression and a decision tree on the WOE features
7. Evaluates both models on a held-out test set — accuracy, confusion matrix, classification report, and ROC curves

## Getting started

### Requirements

- Python ≥ 3.10
- The Lending Club `loan.csv` file ([download from Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club))

### Install dependencies

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn jupyter
```

### Run

1. Open `credit_scorecard_fixed.ipynb` in Jupyter
2. Update `DATA_PATH` near the top of the notebook to point at your local `loan.csv`
3. Run all cells

```bash
jupyter notebook credit_scorecard_fixed.ipynb
```

## Project structure

```
.
├── credit_scorecard_fixed.ipynb   # main notebook — WOE/IV binning, model fitting, evaluation
└── README.md
```

## Notes on the WOE/IV pipeline

- `mono_bin` bins numeric predictors into monotonic quantile buckets (falling back to a fixed number of bins if a monotonic relationship can't be found)
- `char_bin` bins categorical predictors one bucket per category
- `data_vars` runs both across every column in the dataset and returns a per-bin detail table plus a per-variable IV summary
- Variables are ranked by IV to gauge how predictive each one is before being fed into the model


## License

MIT
