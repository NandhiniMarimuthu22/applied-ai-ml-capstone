# Part 1 — Data Cleaning & Exploratory Data Analysis

## Dataset
- **Name:** Adult Income Dataset
- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/2/adult
- **Rows:** 32,537 (after cleaning)
- **Columns:** 15
- **Regression Target:** hours_per_week
- **Classification Target:** income (0 = <=50K, 1 = >50K)

## How to Run
1. Upload adult.data to Google Colab
2. Run all cells top to bottom
3. All plots saved to plots/ folder
4. cleaned_data.csv saved automatically

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn

---

## Task Findings

### Task 1 — Load Dataset
- Shape: 32,561 rows × 15 columns
- Mix of numeric and categorical columns
- Missing values found in workclass, occupation, native_country

### Task 2 — Null Value Analysis
| Column | Null % |
|--------|--------|
| workclass | 5.64% |
| occupation | 5.66% |
| native_country | 1.79% |
| All others | 0.00% |

- No column exceeded 20% null rate
- Categorical nulls filled with mode
- Numeric nulls filled with median

**Why median over mean?**
The median is resistant to outliers. For skewed columns like
capital_gain (skewness = 9.17), the mean ($1,077) is pulled
far above the median ($0) by extreme values. Using the mean
would over-impute for most people who have zero capital gain.
The median is the correct and honest choice.

### Task 3 — Duplicates
- Duplicates found: 24
- Rows after removal: 32,537
- Null % unchanged after removal

### Task 4 — Data Type Correction
- income column converted from string to binary integer (0/1)
- 8 categorical columns converted to category dtype
- Memory before: ~3,748,682 bytes
- Memory after: ~957,312 bytes
- Memory saved: 74.5% reduction

### Task 5 — Skewness
| Column | Skewness |
|--------|----------|
| capital_gain | +9.17 (highest) |
| capital_loss | +4.60 |
| fnlwgt | +1.45 |
| age | +0.56 |
| hours_per_week | +0.23 |
| education_num | -0.31 |

**Most skewed: capital_gain (+9.17)**
Positive skew means a long right tail. 90.8% of values
are zero but a few extreme values pull the mean upward.
Using the mean for imputation would be misleading.
The median (0) is the correct choice.

### Task 6 — Outlier Detection (IQR)

**capital_gain**
- Q1 = 0, Q3 = 0, IQR = 0
- Outliers = 2,996 rows (9.21%)

**hours_per_week**
- Q1 = 40, Q3 = 45, IQR = 5
- Lower bound = 32.5, Upper bound = 52.5
- Outliers = 6,441 rows (19.79%)

**Decision: Retain all outliers**
These are real values — not errors. capital_gain outliers
are genuine high earners. hours_per_week outliers are real
work patterns. Tree-based models in Part 2 handle outliers
naturally so no capping is needed.

### Task 7 — Visualisations

**Line Plot (plots/01_line_plot_hours.png)**
Shows hours_per_week sorted ascending. Long flat section
at 40 hours for most workers. Sharp rise toward 99 hours
for overtime workers.

**Bar Chart (plots/02_bar_chart_hours_by_education.png)**
Prof-school group works most hours (~47 hrs).
Preschool group works fewest hours (~38 hrs).
Higher education linked to more hours worked.

**Histogram (plots/03_histogram_capital_gain.png)**
Extreme right skew confirmed. 90.8% values are zero.
Long tail stretching to $99,999.
Classic zero-inflated distribution shape.

**Scatter Plot (plots/04_scatter_age_vs_hours.png)**
Direction: Weak positive relationship.
Strength: Very weak (r ≈ 0.07).
Red dots (>50K) concentrated at higher hours.
Age alone is not a strong predictor of hours worked.

**Box Plot (plots/05_boxplot_hours_by_income.png)**
>50K group median ≈ 45 hours per week.
<=50K group median ≈ 40 hours per week.
>50K group has wider spread (more variation).
Higher earners consistently work longer hours.

### Task 8 — Pearson Correlation Heatmap
- Highest correlation: education_num ↔ income (r = 0.335)
- This is not directly causal
- Occupation mediates the relationship
- Alternative explanation: family wealth drives both
  higher education AND higher income independently

### Task 9a — Imputation Strategy Comparison
| Column | Mean | Median | Skewness | Chosen |
|--------|------|--------|----------|--------|
| capital_gain | 1077.65 | 0.00 | +9.17 | Median |
| capital_loss | 87.50 | 0.00 | +4.60 | Median |

Both columns are positively skewed. The mean is inflated
by extreme values. The median correctly represents the
majority experience (zero for both columns).
isnull().sum() confirmed 0 nulls after imputation.

### Task 9b — Spearman Correlation
| Pair | Spearman | Pearson | Diff |
|------|----------|---------|------|
| capital_loss ↔ income | 0.187 | 0.150 | 0.037 |
| capital_gain ↔ income | 0.259 | 0.223 | 0.036 |
| age ↔ income | 0.255 | 0.234 | 0.021 |

All 3 pairs show Spearman > Pearson meaning
monotonic but non-linear relationships.
Spearman will guide feature selection in Part 2
because it better captures these non-linear patterns.

### Task 9c — Grouped Aggregation
education → hours_per_week

- Highest mean group: Prof-school (~47.18 hrs)
- Highest std group: Some-college (~12.87 hrs)
- Mean ratio: 1.23

A ratio of 1.23 shows education carries predictive signal.
However high within-group std means education alone is
not enough to predict hours reliably. Other features like
occupation and workclass are needed alongside it.

---

## Design Decisions
| Decision | Reason |
|----------|--------|
| Median imputation | Robust to skewed distributions |
| Mode for categoricals | Most representative for low-null columns |
| Retain outliers | Real values — tree models handle them |
| Category dtype | 74.5% memory saving |
| Spearman over Pearson | Non-linear relationships detected |

## Plot Files
| File | Description |
|------|-------------|
| plots/01_line_plot_hours.png | Line plot — hours per week |
| plots/02_bar_chart_hours_by_education.png | Bar chart — hours by education |
| plots/03_histogram_capital_gain.png | Histogram — capital gain |
| plots/04_scatter_age_vs_hours.png | Scatter — age vs hours |
| plots/05_boxplot_hours_by_income.png | Box plot — hours by income |
| plots/06_pearson_heatmap.png | Pearson heatmap |
| plots/07_spearman_heatmap.png | Spearman heatmap |

---

## Conclusion

Part 1 successfully completed all required steps of the
data acquisition, cleaning, and exploratory analysis phase.

### What was done:
- Loaded the Adult Income dataset (32,561 rows, 15 columns)
- Found and filled missing values in 3 columns
- Removed 24 duplicate rows
- Fixed incorrect data types and saved 74.5% memory
- Identified capital_gain as most skewed column (9.17)
- Detected outliers in capital_gain and hours_per_week
- Created all 5 required visualisations
- Computed Pearson and Spearman correlation matrices
- Compared mean vs median imputation strategies
- Performed grouped aggregation by education level

### Key Findings:
- capital_gain is extremely right-skewed (9.17)
  → median is the only correct imputation choice
- Outliers are real values — retained for Part 2
- education_num is strongest predictor of income (r = 0.335)
- All top correlation pairs are non-linear
  → tree-based models will work better than linear models
- Education carries predictive signal (mean ratio = 1.23)
  but needs other features to predict reliably

### Ready for Part 2:
- cleaned_data.csv saved with 32,537 rows × 15 columns
- All nulls removed (confirmed with isnull().sum() = 0)
- All duplicates removed
- Data types corrected
- Dataset is clean and ready for machine learning