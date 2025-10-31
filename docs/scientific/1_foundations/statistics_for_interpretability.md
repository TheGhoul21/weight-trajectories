# Statistics for Interpretability

A practical guide to statistical concepts and methods essential for rigorous interpretability research, focusing on hypothesis testing, uncertainty quantification, and experimental design.

---

## Why Statistics for Interpretability?

Interpretability research makes claims about what models learn and how they work. Statistics provides tools to:

- **Validate findings**: Is this pattern real or random?
- **Quantify uncertainty**: How confident are we in this interpretation?
- **Design experiments**: How to test hypotheses rigorously?
- **Avoid pitfalls**: Multiple testing, p-hacking, spurious correlations

This primer covers essential statistical concepts with interpretability applications.

---

## Fundamental Concepts

### Distributions

**Distribution**: Describes how values are spread out.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Common distributions in interpretability work

# Normal (Gaussian): Most common, due to Central Limit Theorem
normal_samples = np.random.normal(loc=0, scale=1, size=1000)

# Uniform: Random baseline, null hypothesis
uniform_samples = np.random.uniform(low=-1, high=1, size=1000)

# Bernoulli: Binary outcomes (correct/incorrect classification)
bernoulli_samples = np.random.binomial(n=1, p=0.7, size=1000)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].hist(normal_samples, bins=50, density=True, alpha=0.7)
axes[0].set_title('Normal Distribution')
axes[1].hist(uniform_samples, bins=50, density=True, alpha=0.7)
axes[1].set_title('Uniform Distribution')
axes[2].hist(bernoulli_samples, bins=50, density=True, alpha=0.7)
axes[2].set_title('Bernoulli Distribution')
```

**Interpretability relevance**:
- Activation distributions reveal network behavior
- Baseline comparisons use uniform/random distributions
- Probe accuracy is binomial/multinomial

### Expected Value and Variance

**Expected value** (mean): Center of distribution

```python
def expected_value(samples):
    """Estimate expected value from samples."""
    return np.mean(samples)

# Example: Average probe accuracy
probe_accuracies = np.array([0.75, 0.78, 0.72, 0.76, 0.74])
mean_accuracy = expected_value(probe_accuracies)
print(f"Mean accuracy: {mean_accuracy:.3f}")
```

**Variance**: Spread around mean

```python
def variance(samples):
    """Measure of spread."""
    return np.var(samples, ddof=1)  # ddof=1 for unbiased estimate

def standard_deviation(samples):
    """Square root of variance (same units as data)."""
    return np.std(samples, ddof=1)

# Example: Consistency across runs
accuracies = np.array([0.75, 0.78, 0.72, 0.76, 0.74])
print(f"Std dev: {standard_deviation(accuracies):.3f}")
# Low std dev = consistent, high std dev = variable
```

**Standard error**: Uncertainty in the mean estimate

```python
def standard_error(samples):
    """Uncertainty in mean estimate."""
    return standard_deviation(samples) / np.sqrt(len(samples))

# Example: Report mean ± SE
mean_acc = expected_value(accuracies)
se_acc = standard_error(accuracies)
print(f"Accuracy: {mean_acc:.3f} ± {se_acc:.3f}")
```

### Confidence Intervals

**Confidence interval**: Range likely to contain true value

```python
def confidence_interval(samples, confidence=0.95):
    """Compute confidence interval for mean.

    Returns: (lower_bound, upper_bound)
    """
    n = len(samples)
    mean = np.mean(samples)
    se = stats.sem(samples)  # Standard error

    # Use t-distribution for small samples
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)

    margin = t_critical * se
    return (mean - margin, mean + margin)

# Example
accuracies = np.array([0.75, 0.78, 0.72, 0.76, 0.74])
ci_low, ci_high = confidence_interval(accuracies)
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
```

**Interpretation**: We're 95% confident the true mean lies in this interval.

---

## Hypothesis Testing

### The Basic Framework

**Null hypothesis (H₀)**: Default assumption (e.g., "no effect", "random chance")

**Alternative hypothesis (H₁)**: What we're trying to show

**p-value**: Probability of seeing this result (or more extreme) if H₀ is true

**Decision rule**: If p < α (typically 0.05), reject H₀

### T-Test: Comparing Means

**One-sample t-test**: Is mean different from a value?

```python
def one_sample_ttest(samples, null_value=0.5):
    """Test if sample mean differs from null_value.

    Example: Is probe accuracy > 50% (chance)?
    """
    t_stat, p_value = stats.ttest_1samp(samples, null_value)

    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Reject null: mean is significantly different")
    else:
        print("Fail to reject null: not significant")

    return t_stat, p_value

# Example: Is probe better than chance?
probe_accuracies = np.array([0.65, 0.68, 0.62, 0.67, 0.64])
one_sample_ttest(probe_accuracies, null_value=0.5)
# If p < 0.05, probe learns something beyond chance
```

**Two-sample t-test**: Are two means different?

```python
def two_sample_ttest(samples1, samples2):
    """Test if two groups have different means.

    Example: Does layer 5 encode more than layer 3?
    """
    t_stat, p_value = stats.ttest_ind(samples1, samples2)

    print(f"Group 1 mean: {np.mean(samples1):.3f}")
    print(f"Group 2 mean: {np.mean(samples2):.3f}")
    print(f"p-value: {p_value:.4f}")

    return t_stat, p_value

# Example: Compare probe accuracies across layers
layer3_accuracies = np.array([0.65, 0.68, 0.62, 0.67, 0.64])
layer5_accuracies = np.array([0.75, 0.78, 0.72, 0.76, 0.74])

two_sample_ttest(layer3_accuracies, layer5_accuracies)
```

**Paired t-test**: Compare matched pairs

```python
def paired_ttest(before, after):
    """Test difference in matched pairs.

    Example: Same model before/after fine-tuning
    """
    t_stat, p_value = stats.ttest_rel(before, after)

    differences = after - before
    print(f"Mean improvement: {np.mean(differences):.3f}")
    print(f"p-value: {p_value:.4f}")

    return t_stat, p_value

# Example: Probe accuracy before/after intervention
before_intervention = np.array([0.65, 0.68, 0.62, 0.67, 0.64])
after_intervention = np.array([0.70, 0.72, 0.68, 0.71, 0.69])

paired_ttest(before_intervention, after_intervention)
```

### Permutation Tests: Non-Parametric Alternative

**Advantage**: No distributional assumptions needed

```python
def permutation_test(group1, group2, n_permutations=10000):
    """Non-parametric test via permutation.

    More robust than t-test when distributions are non-normal.
    """
    # Observed difference
    observed_diff = np.mean(group1) - np.mean(group2)

    # Pool all data
    pooled = np.concatenate([group1, group2])
    n1 = len(group1)

    # Generate permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        # Shuffle and split
        np.random.shuffle(pooled)
        perm_group1 = pooled[:n1]
        perm_group2 = pooled[n1:]

        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    print(f"Observed difference: {observed_diff:.3f}")
    print(f"p-value: {p_value:.4f}")

    return p_value

# Example: Compare representations (no assumption of normality)
repr1_scores = np.random.randn(50) + 0.5
repr2_scores = np.random.randn(50)

permutation_test(repr1_scores, repr2_scores)
```

---

## Multiple Testing Correction

**Problem**: Testing many hypotheses inflates false positive rate.

**Example**: Test 100 probes, 5 will appear significant by chance (α = 0.05)

### Bonferroni Correction

**Simple and conservative**: Divide α by number of tests

```python
def bonferroni_correction(p_values, alpha=0.05):
    """Conservative correction for multiple tests."""
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant = p_values < corrected_alpha

    print(f"Testing {n_tests} hypotheses")
    print(f"Corrected α: {corrected_alpha:.6f}")
    print(f"Significant: {np.sum(significant)} / {n_tests}")

    return significant

# Example: Test many probes
p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.10])
bonferroni_correction(p_values)
```

### Benjamini-Hochberg (FDR Control)

**Less conservative**: Controls false discovery rate

```python
def benjamini_hochberg(p_values, fdr=0.05):
    """Control false discovery rate.

    More powerful than Bonferroni when many tests.
    """
    n = len(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH critical values
    bh_critical = fdr * np.arange(1, n+1) / n

    # Find largest i where p_i ≤ (i/n) * FDR
    significant_mask = sorted_p <= bh_critical
    if np.any(significant_mask):
        max_significant_idx = np.where(significant_mask)[0][-1]
        threshold = sorted_p[max_significant_idx]
    else:
        threshold = 0

    significant = p_values <= threshold

    print(f"FDR threshold: {fdr}")
    print(f"Significant: {np.sum(significant)} / {n}")

    return significant

# Example
p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.10])
benjamini_hochberg(p_values, fdr=0.05)
```

**When to use**:
- Bonferroni: When false positives are very costly
- BH: When you want more power and can tolerate some false positives

---

## Effect Sizes

**Problem**: Statistical significance ≠ practical importance

**Effect size**: Magnitude of difference, independent of sample size

### Cohen's d

**Standardized mean difference**

```python
def cohens_d(group1, group2):
    """Effect size for difference between means.

    Interpretation:
    - d < 0.2: negligible
    - 0.2 ≤ d < 0.5: small
    - 0.5 ≤ d < 0.8: medium
    - d ≥ 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    # Effect size
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    print(f"Cohen's d: {d:.3f}")

    if abs(d) < 0.2:
        print("Effect size: negligible")
    elif abs(d) < 0.5:
        print("Effect size: small")
    elif abs(d) < 0.8:
        print("Effect size: medium")
    else:
        print("Effect size: large")

    return d

# Example: Compare layer representations
layer1_scores = np.random.randn(100)
layer10_scores = np.random.randn(100) + 0.8

cohens_d(layer1_scores, layer10_scores)
```

### R-squared (Variance Explained)

**Proportion of variance explained by model**

```python
def r_squared(y_true, y_pred):
    """Variance explained: R² ∈ [0, 1]

    R² = 1: perfect prediction
    R² = 0: no better than mean
    """
    ss_res = np.sum((y_true - y_pred)**2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true))**2)  # Total sum of squares

    r2 = 1 - (ss_res / ss_tot)

    print(f"R²: {r2:.3f}")
    print(f"Variance explained: {r2*100:.1f}%")

    return r2

# Example: How well does probe predict?
y_true = np.random.randn(100)
y_pred = 0.7 * y_true + np.random.randn(100) * 0.3

r_squared(y_true, y_pred)
```

---

## Bootstrap and Resampling

**Bootstrap**: Estimate uncertainty by resampling data

```python
def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=10000, confidence=0.95):
    """Compute CI for any statistic via bootstrap.

    statistic_func: Function that computes statistic from data
    """
    n = len(data)
    bootstrap_statistics = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        stat = statistic_func(resample)
        bootstrap_statistics.append(stat)

    bootstrap_statistics = np.array(bootstrap_statistics)

    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_statistics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_statistics, 100 * (1 - alpha / 2))

    observed_stat = statistic_func(data)

    print(f"Observed statistic: {observed_stat:.3f}")
    print(f"{confidence*100}% CI: [{lower:.3f}, {upper:.3f}]")

    return lower, upper, bootstrap_statistics

# Example: CI for median probe accuracy
probe_accuracies = np.array([0.65, 0.68, 0.62, 0.67, 0.64, 0.70, 0.61])

lower, upper, boot_dist = bootstrap_confidence_interval(
    probe_accuracies,
    statistic_func=np.median,
    n_bootstrap=10000
)
```

**Advantages**: Works for any statistic, no distributional assumptions

### Bootstrap Hypothesis Test

```python
def bootstrap_difference_test(group1, group2, n_bootstrap=10000):
    """Test if two groups differ using bootstrap.

    Null: groups have same distribution
    """
    observed_diff = np.mean(group1) - np.mean(group2)

    # Pool under null hypothesis
    pooled = np.concatenate([group1, group2])
    n1, n2 = len(group1), len(group2)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample from pooled distribution
        boot_sample1 = np.random.choice(pooled, size=n1, replace=True)
        boot_sample2 = np.random.choice(pooled, size=n2, replace=True)

        boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-tailed p-value
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

    print(f"Observed difference: {observed_diff:.3f}")
    print(f"p-value: {p_value:.4f}")

    return p_value

# Example
group1 = np.random.randn(30) + 0.5
group2 = np.random.randn(30)

bootstrap_difference_test(group1, group2)
```

---

## Correlation and Regression

### Correlation Coefficients

**Pearson**: Linear correlation

```python
def pearson_correlation(x, y):
    """Measure linear relationship: r ∈ [-1, 1]

    r = 1: perfect positive correlation
    r = -1: perfect negative correlation
    r = 0: no linear correlation
    """
    r, p_value = stats.pearsonr(x, y)

    print(f"Pearson r: {r:.3f}")
    print(f"p-value: {p_value:.4f}")

    return r, p_value

# Example: Do deeper layers encode more?
layer_depth = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
probe_accuracy = np.array([0.55, 0.58, 0.62, 0.68, 0.72, 0.75, 0.76, 0.74, 0.73, 0.72])

pearson_correlation(layer_depth, probe_accuracy)
```

**Spearman**: Rank correlation (non-parametric)

```python
def spearman_correlation(x, y):
    """Correlation of ranks, robust to outliers."""
    rho, p_value = stats.spearmanr(x, y)

    print(f"Spearman ρ: {rho:.3f}")
    print(f"p-value: {p_value:.4f}")

    return rho, p_value

# Use when relationship is monotonic but not necessarily linear
spearman_correlation(layer_depth, probe_accuracy)
```

### Linear Regression

**Fit**: y = β₀ + β₁x + ε

```python
from scipy.stats import linregress

def simple_linear_regression(x, y):
    """Fit y = intercept + slope * x"""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    print(f"Equation: y = {intercept:.3f} + {slope:.3f} * x")
    print(f"R²: {r_value**2:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Standard error: {std_err:.4f}")

    return slope, intercept

# Example: Probe accuracy vs layer depth
slope, intercept = simple_linear_regression(layer_depth, probe_accuracy)

# Predict
predicted = intercept + slope * layer_depth
```

**Residual analysis**: Check assumptions

```python
def analyze_residuals(y_true, y_pred):
    """Check if residuals are well-behaved."""
    residuals = y_true - y_pred

    print(f"Mean residual: {np.mean(residuals):.6f}")  # Should be ~0
    print(f"Std residual: {np.std(residuals):.3f}")

    # Normality test
    _, p_norm = stats.shapiro(residuals)
    print(f"Shapiro-Wilk p-value: {p_norm:.4f}")  # p > 0.05 = normal

    # Heteroscedasticity check (visual)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    # Should show random scatter around zero

analyze_residuals(probe_accuracy, predicted)
```

---

## Experimental Design for Interpretability

### Randomization and Controls

**Always include random baselines**

```python
def random_baseline_probe(hidden_states, labels, n_shuffles=100):
    """Test if probe beats random labeling.

    Null hypothesis: probe just memorizes random patterns
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Actual performance
    probe = LogisticRegression(max_iter=1000)
    actual_scores = cross_val_score(probe, hidden_states, labels, cv=5)
    actual_mean = np.mean(actual_scores)

    # Random label performance
    random_scores = []
    for _ in range(n_shuffles):
        shuffled_labels = np.random.permutation(labels)
        probe_random = LogisticRegression(max_iter=1000)
        scores = cross_val_score(probe_random, hidden_states, shuffled_labels, cv=5)
        random_scores.append(np.mean(scores))

    random_scores = np.array(random_scores)

    # Compare
    p_value = np.mean(random_scores >= actual_mean)

    print(f"Actual probe accuracy: {actual_mean:.3f}")
    print(f"Random baseline mean: {np.mean(random_scores):.3f}")
    print(f"p-value: {p_value:.4f}")

    return actual_mean, random_scores, p_value
```

### Cross-Validation

**Prevent overfitting, estimate generalization**

```python
from sklearn.model_selection import KFold

def cross_validated_performance(X, y, model, n_folds=5):
    """Robust performance estimate via k-fold CV."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores = []
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        fold_scores.append(score)

    fold_scores = np.array(fold_scores)

    print(f"Mean score: {np.mean(fold_scores):.3f}")
    print(f"Std score: {np.std(fold_scores):.3f}")

    return fold_scores

# Example
from sklearn.linear_model import LogisticRegression

X = np.random.randn(200, 128)
y = np.random.binomial(1, 0.7, size=200)

model = LogisticRegression(max_iter=1000)
scores = cross_validated_performance(X, y, model, n_folds=5)
```

### Power Analysis

**How many samples do you need?**

```python
from statsmodels.stats.power import TTestIndPower

def sample_size_for_ttest(effect_size, alpha=0.05, power=0.8):
    """Compute required sample size.

    effect_size: Expected Cohen's d
    alpha: Significance level
    power: Probability of detecting effect if it exists
    """
    analysis = TTestIndPower()
    n_per_group = analysis.solve_power(effect_size=effect_size,
                                       alpha=alpha,
                                       power=power)

    print(f"Effect size (d): {effect_size}")
    print(f"Required n per group: {int(np.ceil(n_per_group))}")

    return n_per_group

# Example: Plan experiment
sample_size_for_ttest(effect_size=0.5, alpha=0.05, power=0.8)
# Need ~64 samples per group to detect medium effect with 80% power
```

---

## Common Pitfalls

### Pitfall 1: p-Hacking

**Problem**: Testing many hypotheses, reporting only significant ones

**Solutions**:
- Pre-register hypotheses
- Report all tests performed
- Use multiple testing correction
- Report effect sizes alongside p-values

### Pitfall 2: Ignoring Sample Size

**Problem**: Large n makes tiny effects "significant"

```python
# Example: Statistically significant but practically meaningless
np.random.seed(42)
large_group1 = np.random.normal(100, 15, size=10000)
large_group2 = np.random.normal(100.5, 15, size=10000)  # Tiny difference

t_stat, p_value = stats.ttest_ind(large_group1, large_group2)
print(f"p-value: {p_value:.6f}")  # Likely < 0.05

d = cohens_d(large_group1, large_group2)
print(f"Cohen's d: {d:.3f}")  # But very small effect size!
```

**Solution**: Always report effect sizes

### Pitfall 3: Assuming Independence

**Problem**: Multiple measurements from same model are not independent

**Example**: Testing 100 sentences through same model - observations are correlated

**Solutions**:
- Use mixed-effects models
- Bootstrap at the appropriate level (e.g., resample models, not sentences)
- Report clustering in data

### Pitfall 4: Confusing Correlation with Causation

**Problem**: X correlates with Y doesn't mean X causes Y

**Interpretability example**: Layer depth correlates with probe accuracy - but is it depth or capacity?

**Solutions**:
- Controlled experiments (interventions)
- Causal inference methods
- Acknowledge limitations

---

## Reporting Results

### Checklist for Statistical Reporting

```python
def statistical_report_template():
    """What to report in interpretability papers."""
    report = {
        "sample_size": "n = ? (observations, models, runs)",
        "descriptive_stats": "mean ± std or median [IQR]",
        "effect_size": "Cohen's d, R², or domain-appropriate metric",
        "confidence_interval": "95% CI: [lower, upper]",
        "test_used": "t-test, permutation test, etc.",
        "p_value": "exact p-value (not just p < 0.05)",
        "multiple_testing": "correction method if applicable",
        "assumptions": "checked and reported",
        "code_availability": "analysis scripts shared",
        "random_seed": "for reproducibility"
    }
    return report
```

### Example Results Section

```python
# Good reporting example
def report_probe_results(layer_accuracies, baseline_accuracy=0.5):
    """Report probe results with full statistics."""

    n = len(layer_accuracies)
    mean_acc = np.mean(layer_accuracies)
    std_acc = np.std(layer_accuracies, ddof=1)
    ci_low, ci_high = confidence_interval(layer_accuracies)

    # Test against baseline
    t_stat, p_value = stats.ttest_1samp(layer_accuracies, baseline_accuracy)

    # Effect size
    pooled_std = np.sqrt((std_acc**2 + 0**2) / 2)  # Assuming no variance in baseline
    d = (mean_acc - baseline_accuracy) / pooled_std

    print("=" * 50)
    print("PROBE RESULTS")
    print("=" * 50)
    print(f"Sample size: n = {n}")
    print(f"Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"\nComparison to baseline ({baseline_accuracy}):")
    print(f"t({n-1}) = {t_stat:.3f}, p = {p_value:.4f}")
    print(f"Cohen's d = {d:.3f}")

    if p_value < 0.001:
        print("Result: Highly significant (p < 0.001)")
    elif p_value < 0.01:
        print("Result: Very significant (p < 0.01)")
    elif p_value < 0.05:
        print("Result: Significant (p < 0.05)")
    else:
        print("Result: Not significant (p ≥ 0.05)")

    return {
        'mean': mean_acc,
        'std': std_acc,
        'ci': (ci_low, ci_high),
        'p_value': p_value,
        'effect_size': d
    }

# Usage
results = report_probe_results(
    layer_accuracies=np.array([0.72, 0.75, 0.73, 0.76, 0.74]),
    baseline_accuracy=0.5
)
```

---

## Advanced Topics

### Bayesian Statistics (Brief Introduction)

**Difference from frequentist**: Incorporates prior beliefs, provides probability distributions over parameters

```python
# Conceptual example (requires PyMC3 or similar)
# Rather than "is accuracy > 0.5?" (yes/no)
# Ask "what is distribution of accuracy?" (full posterior)

# Simplified pseudo-code
def bayesian_probe_analysis(accuracies):
    """
    prior ~ Beta(α=1, β=1)  # Uniform prior
    likelihood ~ Binomial(n, p)
    posterior ~ Beta(α + successes, β + failures)

    Returns full distribution over accuracy
    """
    # This gives you probability statements like:
    # "95% probability that accuracy is between 0.65 and 0.75"
    # vs frequentist "95% CI" which is different
    pass
```

### Mixed Effects Models

**For hierarchical data**: Multiple measurements per model, model, or dataset

```python
# Requires statsmodels
# Example: probe accuracy across layers and models
# Fixed effect: layer depth
# Random effect: which model
# Controls for model-to-model variation
```

### False Discovery Rate (FDR)

**Control proportion of false positives among discoveries**

Already covered in multiple testing section - use Benjamini-Hochberg procedure.

---

## Interpretability-Specific Statistical Tests

### Representational Similarity Analysis (RSA)

```python
def rsa_significance_test(rdm1, rdm2, n_permutations=10000):
    """Test if two representational dissimilarity matrices correlate.

    rdm1, rdm2: Representational dissimilarity matrices
    Returns: Spearman correlation and p-value
    """
    # Flatten upper triangles
    triu_idx = np.triu_indices(rdm1.shape[0], k=1)
    rdm1_flat = rdm1[triu_idx]
    rdm2_flat = rdm2[triu_idx]

    # Observed correlation
    observed_rho, _ = stats.spearmanr(rdm1_flat, rdm2_flat)

    # Permutation test
    # Permute rows/columns of one RDM
    perm_rhos = []
    for _ in range(n_permutations):
        perm_idx = np.random.permutation(rdm2.shape[0])
        rdm2_perm = rdm2[perm_idx][:, perm_idx]
        rdm2_perm_flat = rdm2_perm[triu_idx]

        rho, _ = stats.spearmanr(rdm1_flat, rdm2_perm_flat)
        perm_rhos.append(rho)

    perm_rhos = np.array(perm_rhos)
    p_value = np.mean(np.abs(perm_rhos) >= np.abs(observed_rho))

    print(f"Observed Spearman ρ: {observed_rho:.3f}")
    print(f"p-value: {p_value:.4f}")

    return observed_rho, p_value
```

### Selectivity Index

```python
def compute_selectivity_index(responses, conditions):
    """Measure neuron selectivity across conditions.

    responses: (n_neurons, n_samples) activation matrix
    conditions: (n_samples,) condition labels

    Returns: Selectivity index for each neuron
    """
    unique_conditions = np.unique(conditions)
    n_neurons = responses.shape[0]

    selectivity = np.zeros(n_neurons)

    for i in range(n_neurons):
        # Mean response per condition
        means = [np.mean(responses[i, conditions == c])
                 for c in unique_conditions]

        # Selectivity: (max - mean) / (max + mean)
        # 0 = non-selective, 1 = highly selective
        max_response = np.max(means)
        mean_response = np.mean(means)

        if max_response + mean_response > 0:
            selectivity[i] = (max_response - mean_response) / (max_response + mean_response)

    return selectivity
```

---

## Practical Workflow

### Complete Analysis Example

```python
def complete_probe_analysis(hidden_states, labels, random_seed=42):
    """Full statistical pipeline for probe analysis.

    Returns comprehensive results with all statistics.
    """
    np.random.seed(random_seed)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    print("=" * 60)
    print("COMPLETE PROBE ANALYSIS")
    print("=" * 60)

    # 1. Descriptive statistics
    print("\n1. DATA DESCRIPTION")
    print(f"Samples: {len(labels)}")
    print(f"Features: {hidden_states.shape[1]}")
    print(f"Classes: {np.unique(labels)}")
    print(f"Class balance: {np.bincount(labels) / len(labels)}")

    # 2. Cross-validated performance
    print("\n2. CROSS-VALIDATION")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    probe = LogisticRegression(max_iter=1000, random_state=random_seed)
    cv_scores = cross_val_score(probe, hidden_states, labels, cv=cv)

    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores, ddof=1)
    ci_low, ci_high = confidence_interval(cv_scores)

    print(f"Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    # 3. Significance test vs baseline
    print("\n3. SIGNIFICANCE TESTING")
    baseline = 1.0 / len(np.unique(labels))  # Chance level
    t_stat, p_value = stats.ttest_1samp(cv_scores, baseline)

    print(f"Baseline (chance): {baseline:.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.6f}")

    # 4. Effect size
    d = (mean_acc - baseline) / std_acc
    print(f"Cohen's d: {d:.3f}")

    # 5. Random baseline comparison
    print("\n4. RANDOM BASELINE")
    n_random = 100
    random_accs = []
    for _ in range(n_random):
        shuffled_labels = np.random.permutation(labels)
        random_scores = cross_val_score(probe, hidden_states, shuffled_labels, cv=cv)
        random_accs.append(np.mean(random_scores))

    random_accs = np.array(random_accs)
    print(f"Random baseline: {np.mean(random_accs):.3f} ± {np.std(random_accs):.3f}")

    # Permutation test
    perm_p = np.mean(random_accs >= mean_acc)
    print(f"Permutation test p-value: {perm_p:.4f}")

    # 6. Bootstrap confidence interval
    print("\n5. BOOTSTRAP")
    boot_ci_low, boot_ci_high, _ = bootstrap_confidence_interval(
        cv_scores,
        statistic_func=np.mean,
        n_bootstrap=10000
    )

    # 7. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Probe accuracy: {mean_acc:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")
    print(f"vs Baseline {baseline:.3f}: p = {p_value:.4f}, d = {d:.3f}")

    if p_value < 0.001 and d > 0.8:
        print("Conclusion: Strong evidence of learning (p < 0.001, large effect)")
    elif p_value < 0.05 and d > 0.5:
        print("Conclusion: Moderate evidence of learning (p < 0.05, medium effect)")
    elif p_value < 0.05:
        print("Conclusion: Weak evidence (p < 0.05 but small effect size)")
    else:
        print("Conclusion: Insufficient evidence of learning above chance")

    return {
        'accuracy': mean_acc,
        'ci': (ci_low, ci_high),
        'p_value': p_value,
        'effect_size': d,
        'random_baseline': np.mean(random_accs),
        'permutation_p': perm_p
    }

# Example usage
# hidden_states = extract_hidden_states(model, data)
# labels = get_labels(data)
# results = complete_probe_analysis(hidden_states, labels)
```

---

## Further Reading

**Textbooks**:
- Wasserman, *All of Statistics* - Concise overview
- Efron & Tibshirani, *An Introduction to the Bootstrap* - Resampling methods
- Gelman et al., *Bayesian Data Analysis* - Bayesian approach

**For ML/interpretability**:
- Bouthillier et al., "Accounting for Variance in ML Experiments"
- Dodge et al., "Show Your Work: Improved Reporting of Experimental Results"
- Dror et al., "Deep Dominance: How to Properly Compare Deep Neural Models"

**This handbook**:
- [What is Interpretability?](what_is_interpretability.md)
- [Linear Probes](../2_methods/probing/linear_probes.md)
- [Experimental Design](../6_tutorials/) (when available)

**Full bibliography**: [References](../references/bibliography.md)

---

**Return to**: [Foundations](README.md) | [Main Handbook](../0_start_here/README.md)
