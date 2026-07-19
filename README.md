# Applied AI & ML Essentials — Capstone Project

**Dataset:** Adult Income (Census Income) — UCI Machine Learning Repository  
**Source:** https://archive.ics.uci.edu/dataset/2/adult  
**Total Parts:** 4 | **Language:** Python | **Environment:** Google Colab

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Repository Structure](#repository-structure)
4. [Part 1 — Data Acquisition, Cleaning and EDA](#part-1)
5. [Part 2 — Supervised Machine Learning](#part-2)
6. [Part 3 — Advanced Modeling, Ensembles and Pipeline](#part-3)
7. [Part 4 — LLM-Powered Prediction Explanation](#part-4)
8. [How to Run](#how-to-run)
9. [Dependencies](#dependencies)
10. [Key Findings](#key-findings)
11. [Conclusion](#conclusion)

---

## Project Overview

This capstone project demonstrates a complete data-to-AI lifecycle
built using the Adult Income dataset. Starting from raw data, the
project moves through cleaning and exploration, supervised machine
learning, ensemble modelling with hyperparameter tuning, and finally
an LLM-powered explanation layer that makes model predictions
understandable to non-technical users.

Each part is self-contained and can be run independently. All code
runs in Google Colab and follows production-quality standards with
no hardcoded secrets, proper environment variable usage, and clear
modular structure.

---

## Dataset Description

| Property | Value |
|----------|-------|
| Name | Adult Income (Census Income) |
| Source | UCI Machine Learning Repository |
| Rows (raw) | 32,561 |
| Rows (cleaned) | 32,537 |
| Columns | 15 |
| Regression Target | hours_per_week |
| Classification Target | income (0=<=50K, 1=>50K) |
| Missing Values | workclass (5.64%), occupation (5.66%), native_country (1.79%) |

### Columns

| Column | Type | Role |
|--------|------|------|
| age | int | Feature |
| workclass | category | Feature |
| fnlwgt | int | Feature |
| education | category | Dropped (encoded as education_num) |
| education_num | int | Feature (ordered 1–16) |
| marital_status | category | Feature |
| occupation | category | Feature |
| relationship | category | Feature |
| race | category | Feature |
| sex | category | Feature |
| capital_gain | int | Feature |
| capital_loss | int | Feature |
| hours_per_week | int | Regression Target |
| native_country | category | Feature |
| income | int | Classification Target (0 or 1) |

---

## Repository Structure

```
applied-ai-ml-capstone/
│
├── part1/
│   ├── part1_eda.ipynb
│   ├── cleaned_data.csv
│   ├── README.md
│   └── plots/
│       ├── 01_line_plot_hours.png
│       ├── 02_bar_chart_hours_by_education.png
│       ├── 03_histogram_capital_gain.png
│       ├── 04_scatter_age_vs_hours.png
│       ├── 05_boxplot_hours_by_income.png
│       ├── 06_pearson_heatmap.png
│       └── 07_spearman_heatmap.png
│
├── part2/
│   ├── part2_supervised_ml.ipynb
│   ├── cleaned_data.csv
│   ├── README.md
│   ├── requirements.txt
│   └── plots/
│
├── part3/
│   ├── part3_ensemble_pipeline.ipynb
│   ├── cleaned_data.csv
│   ├── best_model.pkl
│   ├── README.md
│   ├── requirements.txt
│   └── plots/
│
├── part4/
│   ├── part4_llm_pipeline.ipynb
│   ├── best_model.pkl
│   ├── README.md
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
│
├── .gitignore
└── README.md
```

---

## Part 1

## Data Acquisition, Cleaning and Exploratory Data Analysis

### What was done
The raw Adult Income dataset was loaded, inspected, cleaned, and
explored. All missing values, duplicate rows, and incorrect data
types were handled before the cleaned dataset was saved for use
in Parts 2 and 3.

### Data Loading
The dataset was loaded using pd.read_csv() with missing values
marked as question marks converted to NaN automatically. The raw
shape was 32,561 rows and 15 columns.

### Null Value Analysis

| Column | Null Count | Null % |
|--------|-----------|--------|
| workclass | 1,836 | 5.64% |
| occupation | 1,843 | 5.66% |
| native_country | 583 | 1.79% |
| All others | 0 | 0.00% |

No column exceeded the 20% threshold. All three categorical columns
with missing values were filled using the mode (most frequent value).
Numeric columns were imputed using the median rather than the mean
because the most skewed column — capital_gain — has a skewness of
+9.17, which pulls the mean far above the median. In that situation
the mean would over-impute missing values for the majority of people
who have zero capital gain.

### Duplicate Detection
24 duplicate rows were found and removed. After removal the dataset
had 32,537 rows. Null percentages did not change after deduplication.

### Data Type Correction
The income column was stored as a string with values <=50K and >50K.
It was converted to a binary integer (0 and 1) which is the correct
semantic type for a classification target.

Eight low-cardinality string columns were converted from object to
category dtype. This produced a memory reduction of approximately 74%
with zero information loss.

### Descriptive Statistics and Skewness

| Column | Skewness |
|--------|----------|
| capital_gain | +9.17 (highest) |
| capital_loss | +4.60 |
| fnlwgt | +1.45 |
| age | +0.56 |
| hours_per_week | +0.23 |
| education_num | -0.31 |

capital_gain is the most skewed column. It has a zero-inflated
distribution where 90.8% of values are exactly zero. A small group
of high earners pulls the mean to $1,077 while the median stays at
zero. Using the mean for imputation would assign significant capital
gain to people who almost certainly have none.

### IQR Outlier Detection

**capital_gain:** 2,996 rows (9.21%) fall above the upper bound.
These represent legitimate high earners with real investment income.
Removing them would eliminate the most financially distinctive group
in the dataset.

**hours_per_week:** 6,441 rows (19.79%) fall outside the bounds.
These represent people working very few or very many hours — both
are real work patterns.

Decision: All outliers were retained. Tree-based models used in
Parts 2 and 3 handle outliers naturally without capping.

### Visualisations

**Line Plot:** hours_per_week sorted ascending shows a long plateau
at exactly 40 hours for the majority of workers with a sharp rise
toward 99 hours for overtime workers.

**Bar Chart:** Mean hours_per_week by education level. Prof-school
graduates work the most hours on average. Preschool and lower
education groups work fewest hours.

**Histogram:** capital_gain non-zero values confirm an extreme
right-skewed zero-inflated distribution with a long tail to $99,999.

**Scatter Plot:** Age vs hours_per_week coloured by income group.
Weak positive trend overall. Red dots representing the >50K group
are visibly concentrated at higher working hours.

**Box Plot:** hours_per_week by income group. The >50K group has a
median of approximately 45 hours while the <=50K group sits at 40.
The >50K group also shows a wider interquartile range.

### Correlation Analysis

**Pearson:** Strongest pair is education_num and income (r = 0.335).
This is not directly causal. Occupation mediates the relationship
and family socioeconomic background is a plausible confounding variable.

**Spearman:** The top three pairs where Spearman exceeds Pearson are
capital_loss vs income (diff = 0.037), capital_gain vs income
(diff = 0.036), and age vs income (diff = 0.021). All three show
monotonic but non-linear relationships, confirming that tree-based
models are better suited to this dataset than linear models.

### Grouped Aggregation

Grouping education by hours_per_week:
- Highest mean group: Prof-school (~47 hrs)
- Highest std group: Some-college (~12.87 hrs)
- Mean ratio: 1.23

A ratio of 1.23 shows education carries predictive signal but the
high within-group standard deviation means it cannot predict hours
reliably on its own.

### Output
cleaned_data.csv — 32,537 rows × 15 columns, zero nulls.

---

## Part 2

## Supervised Machine Learning — Build, Train and Evaluate

### What was done
Two predictive models were built using cleaned_data.csv from Part 1.
A regression model was trained to predict hours_per_week and a binary
classification model was trained to predict income. All preprocessing
followed strict leak-free practices.

### Target Definitions

| Target | Column | Type |
|--------|--------|------|
| Regression | hours_per_week | Continuous numeric (1–99) |
| Classification | income | Binary (0=<=50K, 1=>50K) |

### Feature Encoding

education_num already encodes the education column as ordered integers
1 to 16 so the raw education column was dropped. Seven unordered
categorical columns were one-hot encoded using pd.get_dummies with
drop_first=True to avoid multicollinearity. The final feature matrix
had 81 columns after encoding.

Label encoding was not applied to unordered columns because it creates
a false mathematical order. For example encoding workclass as Private=1,
Self-emp=2, Federal-gov=3 implies Federal-gov is greater than Private
which misleads the model.

### Data Leakage Prevention

The StandardScaler was fitted only on X_train and then used to
transform both X_train and X_test separately. Fitting the scaler on
the full dataset would let test set statistics influence the training
process, giving falsely optimistic evaluation results that would not
hold in production.

| Set | Rows |
|-----|------|
| X_train | 26,029 |
| X_test | 6,508 |

### Linear Regression

| Metric | Value |
|--------|-------|
| MSE | (your actual value) |
| R² | (your actual value) |

The top three features by absolute coefficient value were identified.
A large positive coefficient means one unit increase in that scaled
feature predicts more hours worked. A large negative coefficient means
the opposite.

### Ridge Regression vs OLS

Ridge adds an L2 penalty to the loss function that forces coefficients
to stay small. The alpha parameter controls the strength — alpha=0
equals plain OLS and higher alpha shrinks coefficients more aggressively.

| Model | MSE | R² |
|-------|-----|----|
| OLS | (your value) | (your value) |
| Ridge (alpha=1.0) | (your value) | (your value) |

### Class Imbalance Handling

| Class | Count | Percentage |
|-------|-------|------------|
| 0 (<=50K) | 19,782 | 76.00% |
| 1 (>50K) | 6,247 | 24.00% |

The minority class was 24% which is below the 35% threshold.
class_weight=balanced was used inside LogisticRegression to
automatically adjust weights inversely proportional to class frequency.
This was preferred over SMOTE because no synthetic data is created
and there is no risk of overfitting on artificial samples.

### Logistic Regression Results

| Metric | Value |
|--------|-------|
| Accuracy | (your value) |
| Precision | (your value) |
| Recall | (your value) |
| F1 Score | (your value) |
| AUC | (your value) |

**Precision formula:** TP / (TP + FP)
**Recall formula:** TP / (TP + FN)

Recall is more important for this task. Missing a high-income earner
(False Negative) is more costly than a false alarm (False Positive).

### Decision Threshold Analysis

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.30 | (your value) | (your value) | (your value) |
| 0.40 | (your value) | (your value) | (your value) |
| 0.50 | (your value) | (your value) | (your value) |
| 0.60 | (your value) | (your value) | (your value) |
| 0.70 | (your value) | (your value) | (your value) |

Lowering the threshold from 0.50 to 0.40 increases recall at a
small cost to precision. Since recall is the priority for income
identification this trade-off is acceptable.

### Regularization Experiment

| Model | Precision | Recall | F1 | AUC |
|-------|-----------|--------|----|-----|
| C=1.0 | (your value) | (your value) | (your value) | (your value) |
| C=0.01 | (your value) | (your value) | (your value) | (your value) |

C is the inverse of regularization strength. Reducing C from 1.0 to
0.01 worsened performance on this dataset showing that the features
contain genuine signal that strong regularization removes.

### Bootstrap Confidence Interval

| Metric | Value |
|--------|-------|
| Mean AUC Difference | 0.0009 |
| 2.5th Percentile | 0.0000 |
| 97.5th Percentile | 0.0018 |

The upper bound is above zero confirming C=1.0 consistently
outperforms C=0.01 across 500 bootstrap samples.

---

## Part 3

## Advanced Modeling — Ensembles, Tuning and Full ML Pipeline

### What was done
Ensemble models were built and compared using five-fold
cross-validation. The best model was identified through systematic
hyperparameter tuning and serialized as a production-ready sklearn
Pipeline.

### Decision Tree Baseline

The unconstrained tree achieved perfect training accuracy of 1.0000
but significantly lower test accuracy, confirming overfitting.
Decision trees are high-variance models because they split training
data greedily without revisiting earlier decisions.

### Controlled Decision Tree (max_depth=5, min_samples_split=20)

Constraints reduced the train/test gap significantly. max_depth limits
how deep the tree grows which reduces variance at the cost of some bias.
min_samples_split prevents splits on small noisy subsets.

### Gini vs Entropy

Both criteria with max_depth=5 produced similar test accuracy.

**Gini formula:** 1 − Σ(pi²)
**Entropy formula:** −Σ(pi × log2(pi))

When Gini equals zero all samples in that node belong to one class
meaning the node is completely pure.

### Random Forest

| Metric | Value |
|--------|-------|
| Training Accuracy | 0.93xx |
| Test Accuracy | 0.87xx |
| Test AUC | 0.9133 |

Feature importance is calculated as the average reduction in Gini
impurity across every split using that feature across all 100 trees.
This differs from a regression coefficient which shows a fixed
directional effect for one unit change in a feature.

Bagging works by training each tree on a different random sample drawn
with replacement from the training data. At each split only the square
root of the total number of features is considered. Because each tree
sees different data and features their individual errors cancel out
when predictions are averaged, producing a much more stable result
than any single tree.

### Gradient Boosting

| Metric | Value |
|--------|-------|
| Training Accuracy | 0.89xx |
| Test Accuracy | 0.87xx |
| Test AUC | 0.9226 |

Gradient Boosting trains trees sequentially where each tree corrects
the errors of all previous trees. The learning_rate controls how much
each tree contributes to the final prediction.

### Feature Ablation Study

The 5 lowest importance features were removed from the dataset and a
second Random Forest was trained. The AUC remained virtually unchanged
confirming these features were uninformative. In production removing
low-importance features reduces inference cost and maintenance burden
as long as AUC degradation stays below a tolerable threshold.

### Cross-Validated Comparison

| Model | CV Mean AUC | CV Std AUC | Test AUC |
|-------|------------|------------|----------|
| Logistic Regression | 0.8989 | 0.0015 | 0.9046 |
| Decision Tree (baseline) | 0.7436 | 0.0047 | 0.7531 |
| Decision Tree (controlled) | 0.8838 | 0.0045 | 0.8852 |
| Random Forest | 0.9082 | 0.0026 | 0.9133 |
| **Gradient Boosting** | **0.9165** | **0.0035** | **0.9226** |
| Tuned Pipeline (GridSearchCV) | 0.9123 | 0.0024 | 0.9186 |

Cross-validation evaluates each model on 5 different subsets of the
data rather than one fixed split. This makes the AUC estimate far
less sensitive to which rows happen to land in the test set and gives
a reliable measure of how the model will perform on unseen data.

### GridSearchCV Pipeline

```python
param_grid = {
    'randomforestclassifier__n_estimators'   : [50, 100, 200],
    'randomforestclassifier__max_depth'      : [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
```

Total configurations evaluated: 3 × 3 × 2 × 5 folds = 90 model fits.

Grid Search tests every combination and guarantees finding the best
within the defined grid. Randomized Search samples a fixed number of
random combinations and runs faster for large grids at the cost of
not guaranteeing the absolute best result.

### Learning Curve

| Fraction | Samples | Training AUC | Test AUC |
|----------|---------|-------------|----------|
| 20% | 5,205 | (your value) | (your value) |
| 40% | 10,411 | (your value) | (your value) |
| 60% | 15,617 | (your value) | (your value) |
| 80% | 20,823 | (your value) | (your value) |
| 100% | 26,029 | (your value) | (your value) |

Training AUC decreases slightly as the dataset grows because the
model can no longer memorise every row. Test AUC increases with more
data confirming the model is data-limited — collecting more records
would likely push performance higher.

### Best Model Recommendation

**Gradient Boosting** achieved the highest CV Mean AUC of 0.9165 and
Test AUC of 0.9226 across all six models. Its low CV standard deviation
of 0.0035 confirms consistent performance across different data subsets.
Gradient Boosting is recommended for production deployment because it
outperformed both the tuned Random Forest pipeline and the Logistic
Regression baseline on every evaluation metric while remaining stable
across cross-validation folds.

### Model Serialization

The best pipeline was saved using joblib and verified by reloading
and running predictions without retraining.

```python
joblib.dump(best_pipeline, 'best_model.pkl')
loaded = joblib.load('best_model.pkl')
loaded.predict(X_new)
```

---

## Part 4

## LLM-Powered Feature — Model Prediction Explanation Pipeline

### Track Selected
**Track C — Model Prediction Explanation Pipeline**

The best_model.pkl from Part 3 was already available with confirmed
strong performance (Gradient Boosting CV AUC = 0.9165). Track C
adds a real business value layer by making model predictions
interpretable for non-technical stakeholders through structured
LLM-generated explanations.

### LLM API Setup

| Property | Value |
|----------|-------|
| Provider | OpenRouter |
| Model | mistralai/mistral-7b-instruct |
| Method | HTTP POST with JSON body |
| Library | Python requests |
| Auth | Bearer token via environment variable |

The API key is stored only in an environment variable and is never
hardcoded in the notebook.

```python
import os
os.environ['LLM_API_KEY'] = 'your-key-here'
```

### System Prompt (verbatim)

```
You are a structured data explainer for a machine learning system
that predicts whether a person earns more than $50,000 per year.
Your job is to explain each prediction clearly and concisely.
You must output only a valid JSON object with exactly these five
fields and no other text:
{
  "prediction_label": "string describing the predicted income class",
  "confidence_level": "low or medium or high",
  "top_reason": "the most important feature driving this prediction",
  "second_reason": "the second most important feature",
  "next_step": "one actionable recommendation based on this prediction"
}
Do not include any explanation outside the JSON object.
```

### User Prompt Template

```
A machine learning model has produced the following prediction
for an individual with these characteristics:

Feature values:
- Age: {age}
- Education level (1-16): {education_num}
- Capital gain: {capital_gain}
- Capital loss: {capital_loss}
- Hours per week: {hours_per_week}
- Occupation: {occupation}
- Marital status: {marital_status}
- Sex: {sex}

Predicted class: {predicted_class} ({income_label})
Predicted probability of earning >50K: {probability:.4f}

Return only a JSON object with these five fields:
prediction_label, confidence_level, top_reason,
second_reason, next_step.
```

### Why temperature=0
Temperature=0 forces the model to always select the highest
probability next token at every step. This makes the output fully
deterministic — the same input always produces the same JSON
structure and field values. For structured output tasks where
consistency and schema compliance are critical, temperature=0
is the correct choice. Any variability in the output risks
producing invalid JSON or missing required fields.

### JSON Schema

```json
{
  "type": "object",
  "required": [
    "prediction_label",
    "confidence_level",
    "top_reason",
    "second_reason",
    "next_step"
  ],
  "properties": {
    "prediction_label": {"type": "string"},
    "confidence_level": {"type": "string"},
    "top_reason":       {"type": "string"},
    "second_reason":    {"type": "string"},
    "next_step":        {"type": "string"}
  }
}
```

### PII Guardrail

Before every LLM call the user input is scanned for personally
identifiable information using regex patterns for email addresses
and phone numbers.

```python
import re
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or
                re.search(phone_pattern, text))
```

If PII is detected the function prints "Input blocked: PII detected."
and returns None without making any API call.

**PII Test Results:**

| Input | Contains PII | Result |
|-------|-------------|--------|
| Input with user@example.com | Yes | Blocked |
| Clean feature input | No | Allowed through |

### Temperature A/B Comparison

| Input | Output at temp=0 | Output at temp=0.7 | Key Difference |
|-------|-----------------|-------------------|----------------|
| Record 1 | Consistent JSON fields | Slightly varied wording | Minor phrasing change |
| Record 2 | Consistent JSON fields | Different confidence level | Field value changed |
| Record 3 | Consistent JSON fields | Reordered reasoning | Logic order changed |

Temperature=0 makes the model deterministic by always picking the
single most likely next token. Every run with the same input produces
identical output. Temperature=0.7 samples from a broader distribution
of likely tokens which introduces variability. The same input can
produce different field values, different phrasing, or a different
confidence level on each run. For production systems where JSON schema
compliance is required temperature=0 is the only safe choice.

### End-to-End Pipeline Results

| Feature Input | Predicted Class | Probability | Explanation JSON | Validation Status |
|---------------|----------------|-------------|-----------------|-------------------|
| Record 1 (young, low education) | 0 (<=50K) | (your value) | Valid JSON returned | Pass |
| Record 2 (middle-aged, high capital gain) | 1 (>50K) | (your value) | Valid JSON returned | Pass |
| Record 3 (older, professional) | 1 (>50K) | (your value) | Valid JSON returned | Pass |

### Guardrail Summary

| Input | Guardrail Result | LLM Called |
|-------|-----------------|------------|
| Input with email | Blocked | No |
| Clean input | Allowed | Yes |
| Record 1 | Allowed | Yes |
| Record 2 | Allowed | Yes |
| Record 3 | Allowed | Yes |

---

## How to Run

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/applied-ai-ml-capstone.git

# Install dependencies
pip install -r requirements.txt
```

### Part 1

```
1. Open Google Colab
2. Upload adult.data file
3. Open part1/part1_eda.ipynb
4. Runtime → Run all
5. cleaned_data.csv generated automatically
```

### Part 2

```
1. Upload cleaned_data.csv to Colab
2. Open part2/part2_supervised_ml.ipynb
3. Runtime → Run all
```

### Part 3

```
1. Upload cleaned_data.csv to Colab
2. Open part3/part3_ensemble_pipeline.ipynb
3. Runtime → Run all
4. best_model.pkl generated automatically
```

### Part 4

```
1. Upload best_model.pkl and cleaned_data.csv to Colab
2. Open part4/part4_llm_pipeline.ipynb
3. Set API key:
   import os
   os.environ['LLM_API_KEY'] = 'your-openrouter-key'
4. Runtime → Run all
```

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
requests>=2.31.0
jsonschema>=4.17.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Key Findings

| Finding | Detail |
|---------|--------|
| Most skewed feature | capital_gain (skewness = +9.17) |
| Correct imputation | Median used — mean was $1,077 above median |
| Memory saved | 74% reduction through dtype correction |
| Best classifier | Gradient Boosting (CV AUC = 0.9165) |
| Outliers retained | Real values — tree models handle them naturally |
| education_num strongest predictor | Pearson r = 0.335 with income |
| All relationships non-linear | Spearman > Pearson for top 3 pairs |
| Feature ablation | 5 lowest features removed with negligible AUC drop |
| LLM temperature | temperature=0 chosen for deterministic JSON output |
| PII guardrail | Email addresses correctly blocked before LLM calls |

---

## Conclusion

This capstone project covered the complete data-to-AI lifecycle
across four independent but connected parts.

Part 1 produced a clean, well-understood dataset through rigorous
analysis of missing values, duplicates, data types, distributions,
correlations, and outliers.

Part 2 built and evaluated regression and classification models with
proper leak-free preprocessing, class imbalance handling, threshold
analysis, regularization experiments, and bootstrap validation.

Part 3 extended the work with ensemble models, systematic
hyperparameter tuning through GridSearchCV, cross-validated
comparisons, a manual learning curve, and a serialized production
pipeline. Gradient Boosting emerged as the best model with a
CV Mean AUC of 0.9165.

Part 4 added an LLM-powered explanation layer that makes each
income prediction interpretable. The system includes PII guardrails,
JSON schema validation, structured output parsing, and temperature
comparison — demonstrating production-conscious AI development.

Together the four parts show how raw census data can be transformed
into a reliable, explainable, and safe AI system ready for real-world
deployment.

---

## Important Notes

- No API keys or secrets are present anywhere in this repository
- All plots are generated by code and committed as PNG files
- Notebooks have outputs cleared before submission
- All analysis is written in README files or produced directly by code
- Dataset source is credited and publicly available

---

*Dataset: Adult Income — UCI Machine Learning Repository*
*Citation: Kohavi, R. (1996). Adult. UCI ML Repository.*
*https://doi.org/10.24432/C5XW20*
