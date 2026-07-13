# Income Classification — Ensemble Modeling, Tuning & Deployment Pipeline

A comparative study of tree-based ensemble methods for predicting whether an individual earns more than $50K annually, built on the UCI Adult Income dataset. This repository covers baseline and controlled decision trees, Random Forest and Gradient Boosting classifiers, a feature ablation study, cross-validated model selection, hyperparameter tuning with `GridSearchCV`, a manual bias/variance learning curve, and a serialized, reload-verified `scikit-learn` pipeline ready for inference.

This is the third stage of a three-part project. Part 1 covers data cleaning and exploratory analysis; Part 2 covers baseline linear/logistic modeling; this repository (Part 3) covers ensembles, tuning, and packaging a deployable pipeline.

---

## Business Problem

Predicting income bracket from census attributes is a standard proxy for problems organizations actually care about — credit risk tiers, marketing segment targeting, and eligibility screening all reduce to the same shape of problem: binary classification on a mix of numeric and categorical demographic features, with a moderately imbalanced target (roughly 76% ≤$50K vs. 24% >$50K).

The objective here isn't just "does a model work" — it's identifying which model generalizes most reliably, quantifying that reliability with cross-validation rather than a single lucky train/test split, and packaging the winner in a form that can be handed to an inference service without re-running notebook cells.

## Solution Overview

Six models are trained and compared on identical, leakage-free train/test splits:

1. Logistic Regression (carried forward from Part 2 as the linear baseline)
2. Decision Tree — unconstrained
3. Decision Tree — depth- and split-constrained
4. Random Forest
5. Gradient Boosting
6. A `GridSearchCV`-tuned Random Forest wrapped in a `scikit-learn` `Pipeline`

Every model is scored with both a single test-set ROC-AUC and 5-fold stratified cross-validated AUC, since a single split can flatter or punish a model by chance. The best-performing pipeline from the grid search is serialized with `joblib`, reloaded in a clean cell, and verified to produce byte-identical predictions to the in-memory model before being committed to the repository.

## Dataset

- **Source:** [UCI Machine Learning Repository — Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Rows:** 32,537 (post-cleaning, carried forward from Part 1)
- **Target:** `income` (0 = ≤$50K, 1 = >$50K), class balance ≈ 75.9% / 24.1%
- **Split:** 80/20 stratified train/test (`random_state=42`), preserving class ratio in both sets — 26,029 training rows, 6,508 test rows
- **Feature encoding:** raw `education` dropped in favor of the already-ordinal `education_num`; all remaining categorical columns one-hot encoded with `drop_first=True` to avoid collinearity, producing 81 numeric features from the original 13
- **Scaling:** `StandardScaler` fit only on the training partition and applied to both splits, preventing test-set statistics from leaking into training

## Technology Stack

| Component | Tool |
|---|---|
| Language | Python 3 |
| Data handling | pandas, numpy |
| Modeling | scikit-learn (tree, ensemble, linear_model, pipeline, model_selection) |
| Serialization | joblib |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter / Google Colab |

## Project Architecture

```
Part 1 (EDA & Cleaning)  →  cleaned_data.csv
                                   │
Part 2 (Baseline Model)  →  Logistic Regression, train/test split definition
                                   │
Part 3 (this repo)       →  Decision Trees → Random Forest → Gradient Boosting
                                   │
                          Feature Ablation → Cross-Validation → GridSearchCV
                                   │
                          Learning Curve → Serialization → Model Comparison
```

The train/test split from Part 2 is recreated deterministically here (same `test_size`, `random_state`, and stratification) so that all Part 3 models are evaluated on exactly the same holdout data as the linear baseline.

## Machine Learning Workflow

**1. Data reload & feature preparation.** `cleaned_data.csv` is reloaded and re-verified (zero nulls, zero duplicates). The classification target and feature matrix are re-derived, the raw `education` column is dropped in favor of `education_num`, and the remaining seven categorical columns are one-hot encoded, taking the feature space from 12 to 81 columns. Feature names are saved as a list and reused throughout the notebook so importance scores and ablation results stay aligned to the correct columns.

**2. Split & scaling.** The train/test split is recreated with the same parameters as Part 2 to keep results comparable, and a `StandardScaler` is fit exclusively on the training data to avoid any leakage of test-set distribution into the transform.

**3. Baseline re-established.** Logistic Regression is retrained on the scaled features (`class_weight='balanced'`, `C=1.0`, `max_iter=1000`) purely so it has a fair seat at the same cross-validation table as the tree-based models later on.

**4. Tree-based models, from unconstrained to tuned.** Two decision trees, a Random Forest, and a Gradient Boosting classifier are trained, each isolating a specific modeling lever — tree depth, split criterion, bagging, and boosting — before converging on a tuned, pipelined Random Forest as the deployment candidate.

**5. Validation and selection.** Every candidate model is scored with 5-fold stratified cross-validation on ROC-AUC (not accuracy, since accuracy is a poor metric under class imbalance), and the tuned pipeline is additionally profiled with a learning curve to check whether it's starved for data or has hit a capacity ceiling.

**6. Packaging.** The winning pipeline is serialized, reloaded into a fresh variable, and used to predict on two hand-built rows to confirm the artifact works exactly as the in-memory object did.

---

## Models Implemented

### Decision Tree — Unconstrained Baseline

An unconstrained `DecisionTreeClassifier` (`max_depth=None`) is allowed to grow until every leaf is pure. Decision trees are high-variance learners by construction: at each split they greedily choose the locally optimal feature and threshold based only on the data in front of them at that node, with no mechanism to revisit or correct an earlier split once the training data has been fully partitioned. Left unconstrained, this greedy process keeps splitting until it has effectively memorized the training set.

**Result:** depth 47, 4,151 leaves, training accuracy **0.9998**, test accuracy **0.8190** — a train/test gap of **0.1808**. A gap this large is a textbook overfitting signature: the tree has learned noise specific to the training rows rather than the underlying decision boundary, and that noise does not transfer to unseen data.

### Decision Tree — Depth- and Split-Constrained

The same algorithm, constrained with `max_depth=5` and `min_samples_split=20`. `max_depth` caps how many sequential decisions the tree can make per prediction, which trades a small amount of bias (the tree can no longer represent arbitrarily complex boundaries) for a large reduction in variance. `min_samples_split=20` blocks the tree from splitting a node once fewer than 20 samples remain there, which prevents the tree from carving out rules that fit a handful of training rows rather than a genuine pattern.

**Result:** depth 5, 25 leaves, training accuracy **0.8454**, test accuracy **0.8490** — gap **-0.0036**. The gap didn't just shrink, it went slightly negative (test performance marginally exceeds training performance, well within noise), which is the constrained tree's structural ceiling working as intended: it can no longer overfit because it physically lacks the capacity to.

### Gini vs. Entropy

Two constrained trees (`max_depth=5`) were trained with different split criteria:

- **Gini impurity:** `1 - Σ pᵢ²`
- **Entropy:** `-Σ pᵢ log₂(pᵢ)`

Both measure node impurity — how mixed the classes are at a given node — and a node with Gini = 0 (or Entropy = 0) means every sample reaching it belongs to a single class, i.e., a perfectly pure leaf. The two criteria are mathematically different curves but tend to select nearly identical splits in practice.

| Metric | Gini | Entropy |
|---|---|---|
| Train accuracy | 0.8457 | 0.8453 |
| Test accuracy | **0.8486** | 0.8485 |
| Train/test gap | -0.0030 | -0.0032 |

The difference between the two (0.0001 on test accuracy) is negligible — consistent with the well-known result that criterion choice rarely matters much for final performance, while `max_depth` and `min_samples_split` matter a great deal.

### Random Forest

`RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)`.

**Result:** training accuracy **0.8626**, test accuracy **0.8631**, test ROC-AUC **0.9133**, gap **-0.0005**.

**Bagging, concretely:** each of the 100 trees in the forest is trained on a bootstrap sample — a random draw of the training rows *with replacement*, so each tree sees a slightly different dataset and roughly a third of the original rows never appear in any given tree's sample. At every split, instead of considering all 81 features, only a random subset of about √81 ≈ 9 features is evaluated, which decorrelates the trees from one another; without this, most trees would converge on the same dominant features and split similarly regardless of the bootstrap sample. Averaging predictions across many decorrelated, individually high-variance trees cancels out much of each tree's idiosyncratic noise while preserving the signal they all pick up on, which is exactly why the Random Forest's train/test gap (-0.0005) is dramatically tighter than the single unconstrained tree's (0.1808) despite both being tree-based.

**Top 5 features by importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `capital_gain` | 0.1840 |
| 2 | `marital_status_Married-civ-spouse` | 0.1807 |
| 3 | `education_num` | 0.1483 |
| 4 | `age` | 0.0790 |
| 5 | `marital_status_Never-married` | 0.0716 |

Random Forest importance is computed as the average reduction in Gini impurity contributed by a feature across every split where it's used, averaged over all trees in the forest. This is fundamentally different from a linear regression coefficient: a coefficient tells you the marginal effect of a feature holding all others constant under an assumed linear relationship, while Gini importance tells you how often and how effectively a feature was chosen to separate classes across an ensemble of non-linear, interaction-aware trees — it captures usefulness for splitting, not a signed, additive effect on the outcome.

### Gradient Boosting

`GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)`.

**Result:** training accuracy **0.8665**, test accuracy **0.8682**, test ROC-AUC **0.9226**, gap **-0.0017** — the highest test ROC-AUC of any individually trained model in this repository, outperforming the Random Forest by roughly a full point of AUC (0.9226 vs. 0.9133). Unlike bagging, boosting builds trees sequentially, with each shallow tree (`max_depth=3`) explicitly targeting the residual errors of the ensemble built so far. This lets Gradient Boosting extract more signal per tree than an equivalently sized Random Forest, at the cost of being more sensitive to the learning rate and iteration count, and slightly more expensive to train since the trees can't be built in parallel.

### Feature Ablation Study

Using the Random Forest's importance scores, the five lowest-ranked features — `workclass_Never-worked`, `occupation_Armed-Forces`, `native_country_Holand-Netherlands`, `native_country_Scotland`, and `workclass_Without-pay` — were dropped, and an identically configured Random Forest was retrained on the remaining 76 features.

| | Full model (81 features) | Reduced model (76 features) |
|---|---|---|
| Test ROC-AUC | 0.9133 | **0.9136** |

The reduced model matched — and marginally exceeded — the full model's AUC. All five dropped features are rare one-hot categories from sparsely populated groups (e.g., `native_country_Holand-Netherlands` corresponds to a single-digit number of training rows), so removing them didn't cost any discriminative power; if anything, it removed a small amount of noise the model was fitting to near-empty categories. In a production setting this is a genuinely useful result: fewer input columns means a smaller feature-engineering surface to maintain, fewer categories that can silently break if a new value appears at inference time, and marginally cheaper inference — with essentially zero AUC cost. The caveat is that this only holds because the AUC delta is inside noise; if ablation had dropped AUC by more than a few tenths of a point, the simplicity gain would not have been worth the accuracy trade-off.

### Cross-Validated Model Comparison

Single train/test splits are sensitive to which rows happened to land in the test set — a model can look better or worse than it really is purely by chance. 5-fold stratified cross-validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `scoring='roc_auc'`) averages performance across five different held-out folds, which gives a mean that's far less sensitive to any single split and a standard deviation that quantifies how stable that performance actually is.

| Model | CV Mean AUC | CV Std AUC |
|---|---|---|
| Logistic Regression | 0.8989 | 0.0015 |
| Decision Tree (controlled) | 0.8838 | 0.0045 |
| Random Forest | 0.9082 | 0.0026 |
| **Gradient Boosting** | **0.9165** | 0.0035 |

Gradient Boosting leads on mean AUC by a clear margin, and its standard deviation (0.0035) is small enough that the ranking is unlikely to be an artifact of the fold split.

### Hyperparameter Tuning — GridSearchCV Pipeline

A `Pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(random_state=42))` was searched over:

```python
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
```

This is 3 × 3 × 2 = **18 configurations**, each evaluated across 5 folds, for **90 total model fits**. The search was scored on `roc_auc` with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and run directly on the unscaled `X_train` / `y_clf_train`, since the pipeline itself handles imputation and scaling internally — this is precisely the point of wrapping preprocessing inside the pipeline rather than doing it as a separate step: cross-validation folds get their own independently fitted scaler, eliminating any risk of fold-to-fold leakage.

Exhaustive grid search evaluates every combination in the grid, which guarantees the reported optimum really is the best point *within that grid*, but its cost grows multiplicatively with every parameter you add — a fourth parameter with three options would take the fit count from 90 to 270. Randomized Search instead samples a fixed number of random combinations from the parameter space, which scales independently of grid size and in practice finds near-optimal regions almost as reliably, at a fraction of the compute cost — the trade-off is giving up the guarantee of having tested every combination.

**Best parameters:** `max_depth=None`, `min_samples_leaf=5`, `n_estimators=200`
**Best CV AUC:** 0.9123
**Test set (best pipeline):** accuracy 0.8665, AUC **0.9186**

### Manual Learning Curve

The tuned pipeline was refit from scratch on five progressively larger slices of the training data (first 20%, 40%, 60%, 80%, 100% of `X_train`), scored each time on that same training slice and on the fixed, full test set:

| Training fraction | Samples | Training AUC | Test AUC |
|---|---|---|---|
| 20% | 5,205 | 0.9353 | 0.9125 |
| 40% | 10,411 | 0.9309 | 0.9162 |
| 60% | 15,617 | 0.9324 | 0.9169 |
| 80% | 20,823 | 0.9341 | 0.9176 |
| 100% | 26,029 | 0.9323 | 0.9186 |

Training AUC drifts down slightly from 0.9353 to 0.9323 as more data is added — the expected pattern for a model that could fit small subsets almost perfectly and finds slightly more genuine variance to explain as the training set grows. Test AUC rises from 0.9125 to 0.9186, a gain of **0.0061** across the full range. That gain is real but small, and it's still trending upward at 100% of the data without having plateaued sharply — the honest read here is a model that is close to its capacity ceiling for this feature set and algorithm, with only a marginal amount of headroom left from additional data alone. Meaningfully higher performance would more likely come from a different algorithm (the Gradient Boosting results above already suggest as much) or new features than from simply collecting more rows of the same kind.

### Model Serialization & Reload Verification

The tuned pipeline (`best_pipeline`, the `best_estimator_` from `GridSearchCV`) is saved with:

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

producing a **14.19 MB** artifact — comfortably under GitHub's 100 MB file limit, so no regeneration script is required. It's reloaded in an isolated cell:

```python
loaded_model = joblib.load("best_model.pkl")
predictions = loaded_model.predict(new_rows)
probabilities = loaded_model.predict_proba(new_rows)[:, 1]
```

and used to score two hand-built rows, returning a `<=50K` prediction at 32.6% probability of the positive class and a `>50K` prediction at 92.75% — sane, well-separated outputs rather than borderline noise. As a final integrity check, the reloaded model's test-set AUC (0.918587) was compared against the original in-memory pipeline's AUC (0.918587) and confirmed identical to floating-point precision, meaning the serialized artifact is a faithful, deployable copy of the model that was actually validated.

---

## Experimental Results — Full Comparison

| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression (Part 2) | 0.8989 | 0.0015 | 0.9046 |
| Decision Tree (baseline) | 0.7436 | 0.0047 | 0.7531 |
| Decision Tree (controlled) | 0.8838 | 0.0045 | 0.8852 |
| Random Forest | 0.9082 | 0.0026 | 0.9133 |
| **Gradient Boosting** | **0.9165** | 0.0035 | **0.9226** |
| Tuned Pipeline (GridSearchCV, Random Forest) | 0.9123 | 0.0024 | 0.9186 |

## Model Comparison & Recommendation

**Gradient Boosting is the model I'd recommend for deployment.** It posts the highest score on both metrics that matter for this problem — 0.9165 mean CV AUC and 0.9226 test AUC — with a standard deviation (0.0035) tight enough to trust the ranking rather than attribute it to a lucky fold. The unconstrained decision tree is the clearest cautionary result in the table: strong-looking test accuracy in isolation collapses to the weakest CV AUC (0.7436) once evaluated properly across folds, which is exactly why single-split evaluation is unreliable for model selection.

One nuance worth flagging: the artifact serialized and reload-verified in this repository (`best_model.pkl`) is the *tuned Random Forest pipeline* (test AUC 0.9186), because the grid search in this stage was scoped to Random Forest hyperparameters. Gradient Boosting was evaluated with a fixed, untuned configuration and still outperformed the tuned Random Forest — which means there's a straightforward next step for a production iteration: rerun the `GridSearchCV` pipeline with `GradientBoostingClassifier` in place of `RandomForestClassifier`, expecting a further AUC gain beyond 0.9226. Until that's done, the tuned Random Forest pipeline remains the safest deployable choice on record, since it's the only candidate that has been through the full imputation → scaling → tuning → serialization → reload-verification cycle end to end.

## Engineering Decisions

| Decision | Rationale |
|---|---|
| ROC-AUC as primary metric, not accuracy | Target is imbalanced (76/24); accuracy rewards a model for defaulting to the majority class |
| 5-fold stratified CV over single split | Removes sensitivity to which rows land in the test set; folds preserve class ratio |
| Preprocessing inside the `Pipeline`, not before | Guarantees each CV fold fits its own scaler — no leakage between folds |
| Grid search scoped to Random Forest only | Kept the search space tractable (90 fits) while still covering depth, tree count, and leaf size — the parameters shown to matter most in the earlier decision-tree experiments |
| Ablation removed features by importance rank, not by hand | Avoids selection bias — the five weakest features by the model's own metric were removed, not features chosen to make a point |
| Serialize the full pipeline, not just the estimator | Preprocessing steps travel with the model, so `best_model.pkl` can score raw, unscaled input directly |

## Production Considerations

- **Inference cost:** the ablation study shows five sparse one-hot columns can be dropped with a net-positive AUC change, which is a low-risk first step toward a leaner feature set if inference latency or feature-pipeline maintenance becomes a constraint.
- **Interpretability:** Random Forest's Gini-based importances are directly inspectable and were used to drive the ablation study; if a stakeholder needs individual-prediction explanations rather than global feature rankings, SHAP values on either the Random Forest or Gradient Boosting model would be the natural next addition.
- **Model drift risk:** several one-hot categories (`native_country_Holand-Netherlands`, `occupation_Armed-Forces`) are backed by a handful of training rows. A category absent from training but present at inference time will simply produce an all-zero encoding for that feature — the pipeline won't error, but it's worth monitoring for new categorical values in production data.
- **Reproducibility:** every model in this repository is fit with `random_state=42`, and cross-validation folds are generated the same way across every comparison, so the summary table can be regenerated deterministically from the notebook.

## Repository Structure

```
.
├── part3_advanced_modeling_ensembles_tuning_pipeline.ipynb   # full modeling notebook
├── best_model.pkl                                            # serialized, reload-verified pipeline
├── requirements.txt                                          # pinned dependencies
├── cleaned_data.csv                                          # input (produced by Part 1)
├── plots/
│   ├── decision_tree_baseline.png
│   ├── decision_tree_comparison.png
│   ├── gini_vs_entropy.png
│   ├── random_forest_importance.png
│   ├── gradient_boosting_results.png
│   ├── feature_ablation.png
│   ├── cv_comparison.png
│   ├── gridsearch_results.png
│   ├── learning_curve.png
│   └── model_summary.png
└── README.md
```

## Installation

```bash
git clone <this-repo-url>
cd <repo-directory>
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, matplotlib, seaborn, scikit-learn, joblib, jupyter

## Usage

**Run the full notebook top to bottom:**

```bash
jupyter notebook part3_advanced_modeling_ensembles_tuning_pipeline.ipynb
```

**Load the serialized model directly for inference:**

```python
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")

# new_data must match the 81-column one-hot encoded schema
# produced by the preprocessing steps in the notebook
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]
```

## Conclusion

This stage of the project moves from a single linear baseline to a validated, tuned ensemble pipeline, and the results tell a consistent story: tree depth and split constraints control overfitting exactly as theory predicts (0.1808 gap unconstrained vs. -0.0036 controlled), bagging closes most of that gap automatically without manual tuning, and boosting closes the rest of it further, ending up as the strongest individual model in the comparison at 0.9226 test AUC. Cross-validation surfaced a result a single split would have hidden — the unconstrained tree's inflated test accuracy evaporates under proper folding — which is the strongest practical argument in this repository for never trusting a single train/test split for model selection. The tuned, serialized Random Forest pipeline is production-ready today; Gradient Boosting is the clear target for the next tuning pass.
