import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
import os

warnings.filterwarnings("ignore")

print("Step 18: Year 5 Patient Characterization")
print("Profile, dropout comparison, trajectory, and retention prediction")
print("=" * 70)

IMPROVEMENT_THRESHOLD = 0.5

# ==========================================
# LOAD DATA
# ==========================================
longitudinal = pd.read_csv("step15_sadda_style_longitudinal_results.csv", dtype=str)
demographics = pd.read_csv("patient_demographics.csv", dtype=str)

longitudinal["patient_id"] = longitudinal["patient_id"].astype(str).str.strip()
demographics["patient_id"] = demographics["patient_id"].astype(str).str.strip()

# ==========================================
# NUMERIC CONVERSION
# ==========================================
year_cols = ["baseline_a1c","year1_a1c","year2_a1c","year3_a1c","year4_a1c","year5_a1c"]
for col in year_cols:
    if col in longitudinal.columns:
        longitudinal[col] = pd.to_numeric(longitudinal[col], errors="coerce")

demographics["approximate_age"] = pd.to_numeric(demographics["approximate_age"], errors="coerce")

# ==========================================
# CHECK PROCEDURE CODE
# ==========================================
print(f"\nprocedure_code in longitudinal: {'procedure_code' in longitudinal.columns}")

# ==========================================
# MERGE DEMOGRAPHICS
# ==========================================
demo_cols = [c for c in [
    "patient_id","sex","race","ethnicity",
    "approximate_age","patient_regional_location","diabetes_type"
] if c in demographics.columns]

df = longitudinal.merge(demographics[demo_cols], on="patient_id", how="left")

# ==========================================
# BMI
# ==========================================
if os.path.exists("preop_bmi.csv"):
    bmi = pd.read_csv("preop_bmi.csv", dtype=str)
    bmi["patient_id"] = bmi["patient_id"].astype(str).str.strip()
    bmi["preop_bmi"] = pd.to_numeric(bmi["bmi_value"], errors="coerce")
    df = df.merge(bmi[["patient_id","preop_bmi"]], on="patient_id", how="left")
    print(f"BMI data: {df['preop_bmi'].notna().sum()} patients")
else:
    df["preop_bmi"] = np.nan
    print("preop_bmi.csv not found — BMI excluded")

# ==========================================
# PROCEDURE CODE — COLLAPSE G-POEM
# ==========================================
if "procedure_code" in df.columns:
    df["procedure_code"] = df["procedure_code"].astype(str).str.strip()
    df["is_gpoem"] = df["procedure_code"].isin(["43659","43999"]).astype(int)
    df["procedure_label"] = df["procedure_code"].map({
        "43659": "G-POEM",
        "43999": "G-POEM",
        "43800": "Pyloroplasty"
    }).fillna("Unknown")
else:
    df["is_gpoem"] = np.nan
    df["procedure_label"] = "Unknown"

# ==========================================
# CALCULATE CHANGE SCORES
# ==========================================
for label in ["year1","year2","year3","year4","year5"]:
    df[f"{label}_change"] = df[f"{label}_a1c"] - df["baseline_a1c"]

# ==========================================
# AGE DECADE
# ==========================================
df["age_decade"] = df["approximate_age"] / 10

# ==========================================
# DEFINE GROUPS
# ==========================================
year5 = df[df["year5_a1c"].notna()].copy()
year1_all = df[df["year1_a1c"].notna()].copy()
year1_only = df[
    (df["year1_a1c"].notna()) &
    (df["year5_a1c"].isna())
].copy()

# Dropout verification
y5_missing_y1 = ((df["year5_a1c"].notna()) & (df["year1_a1c"].isna())).sum()

print(f"\nTotal cohort: {len(df):,}")
print(f"Patients with baseline A1c: {df['baseline_a1c'].notna().sum()}")
print(f"Patients with Year 1 data: {len(year1_all)}")
print(f"Patients with Year 5 data: {len(year5)}")
print(f"Year 5 patients missing Year 1 data: {y5_missing_y1}")
print(f"Year 1 patients without Year 5 (dropouts): {len(year1_only)}")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def describe_col(grp, col, label):
    vals = pd.to_numeric(grp[col], errors="coerce").dropna()
    if len(vals) == 0:
        return f"  {label}: no data"
    return (f"  {label}: mean {vals.mean():.2f} ± {vals.std():.2f}, "
            f"median {vals.median():.2f} "
            f"(IQR {vals.quantile(0.25):.2f}–{vals.quantile(0.75):.2f}, n={len(vals)})")

def profile_group(grp, label):
    print(f"\n--- {label} (n={len(grp)}) ---")
    print(describe_col(grp, "baseline_a1c", "Baseline A1c"))
    print(describe_col(grp, "approximate_age", "Age"))
    if "preop_bmi" in grp.columns:
        print(describe_col(grp, "preop_bmi", "Pre-op BMI"))
    print(f"  Female: {grp['sex'].eq('F').sum()} ({100*grp['sex'].eq('F').mean():.1f}%)")
    print(f"  Type 1: {grp['diabetes_type'].eq('Type 1').sum()} ({100*grp['diabetes_type'].eq('Type 1').mean():.1f}%)")
    print(f"  Type 2: {grp['diabetes_type'].eq('Type 2').sum()} ({100*grp['diabetes_type'].eq('Type 2').mean():.1f}%)")
    if "is_gpoem" in grp.columns:
        print(f"  G-POEM: {int(grp['is_gpoem'].sum())} ({100*grp['is_gpoem'].mean():.1f}%)")
        print(f"  Pyloroplasty: {grp['procedure_label'].eq('Pyloroplasty').sum()} ({100*grp['procedure_label'].eq('Pyloroplasty').mean():.1f}%)")
    if "race" in grp.columns:
        print("  Race:")
        for val, cnt in grp["race"].value_counts(dropna=False).head(6).items():
            print(f"    {val}: {cnt} ({100*cnt/len(grp):.1f}%)")
    if "ethnicity" in grp.columns:
        print("  Ethnicity:")
        for val, cnt in grp["ethnicity"].value_counts(dropna=False).items():
            print(f"    {val}: {cnt} ({100*cnt/len(grp):.1f}%)")
    if "patient_regional_location" in grp.columns:
        print("  Top 5 regions:")
        for val, cnt in grp["patient_regional_location"].value_counts(dropna=False).head(5).items():
            print(f"    {val}: {cnt} ({100*cnt/len(grp):.1f}%)")

def compare_continuous(label, col, g1, g2, g1_label="Year5", g2_label="Dropout"):
    r = pd.to_numeric(g1[col], errors="coerce").dropna()
    nr = pd.to_numeric(g2[col], errors="coerce").dropna()
    if len(r) < 3 or len(nr) < 3:
        print(f"  {label:<38} insufficient data")
        return
    u, p = stats.mannwhitneyu(r, nr, alternative="two-sided")
    print(f"  {label:<38} "
          f"{g1_label}: {r.median():.2f} (mean {r.mean():.2f}±{r.std():.2f})  "
          f"{g2_label}: {nr.median():.2f} (mean {nr.mean():.2f}±{nr.std():.2f})  "
          f"p={p:.4f}{'*' if p<0.05 else ''} (Mann-Whitney)")

def compare_binary(label, col, val, g1, g2, g1_label="Year5", g2_label="Dropout"):
    """Binary comparison using Fisher's exact test."""
    if col not in g1.columns or col not in g2.columns:
        print(f"  {label:<38} missing column")
        return
    combined = pd.concat([g1[[col]].assign(group=0), g2[[col]].assign(group=1)])
    ct = pd.crosstab(combined[col], combined["group"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print(f"  {label:<38} insufficient variation")
        return
    try:
        if ct.shape == (2, 2):
            odds, p = stats.fisher_exact(ct)
        else:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
        g1_pct = 100 * g1[col].eq(val).mean()
        g2_pct = 100 * g2[col].eq(val).mean()
        print(f"  {label:<38} {g1_label}: {g1_pct:.1f}%  "
              f"{g2_label}: {g2_pct:.1f}%  p={p:.4f}{'*' if p<0.05 else ''} (Fisher)")
    except Exception:
        print(f"  {label:<38} test failed")

def compare_distribution(label, col, g1, g2, g1_label="Year5", g2_label="Dropout"):
    """Global chi-square test for categorical distribution."""
    if col not in g1.columns or col not in g2.columns:
        print(f"  {label:<38} missing column")
        return
    combined = pd.concat([g1[[col]].assign(group=0), g2[[col]].assign(group=1)])
    ct = pd.crosstab(combined[col], combined["group"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print(f"  {label:<38} insufficient variation")
        return
    try:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        print(f"  {label:<38} p={p:.4f}{'*' if p<0.05 else ''} (Chi-square, {ct.shape[0]} categories)")
        # Print distribution
        for val in ct.index:
            g1_pct = 100 * g1[col].eq(val).mean()
            g2_pct = 100 * g2[col].eq(val).mean()
            print(f"    {str(val):<30} {g1_label}: {g1_pct:.1f}%  {g2_label}: {g2_pct:.1f}%")
    except Exception:
        print(f"  {label:<38} test failed")

# ==========================================
# PART A — PROFILE OF YEAR 5 PATIENTS
# ==========================================
print("\n" + "="*70)
print("PART A — PROFILE OF YEAR 5 PATIENTS")
print("="*70)

profile_group(year5, "Year 5 patients")

print("\n--- Year 5 by diabetes type ---")
for dtype in ["Type 1","Type 2"]:
    grp = year5[year5["diabetes_type"] == dtype]
    print(f"\n  {dtype} (n={len(grp)})")
    print(f"    Baseline A1c: mean {grp['baseline_a1c'].dropna().mean():.2f} ± {grp['baseline_a1c'].dropna().std():.2f}")
    print(f"    Age: mean {grp['approximate_age'].dropna().mean():.1f} ± {grp['approximate_age'].dropna().std():.1f}")

print("\n--- Year 5 by procedure type ---")
for plabel in ["G-POEM","Pyloroplasty"]:
    grp = year5[year5["procedure_label"] == plabel]
    if len(grp) > 0:
        print(f"\n  {plabel} (n={len(grp)})")
        print(f"    Baseline A1c: mean {grp['baseline_a1c'].dropna().mean():.2f} ± {grp['baseline_a1c'].dropna().std():.2f}")
        print(f"    Age: mean {grp['approximate_age'].dropna().mean():.1f} ± {grp['approximate_age'].dropna().std():.1f}")

# ==========================================
# PART B — YEAR 5 vs DROPOUT COMPARISON
# ==========================================
print("\n" + "="*70)
print("PART B — YEAR 5 PATIENTS vs DROPOUTS")
print("(Dropouts = had Year 1 data but NOT Year 5)")
print("="*70)

print(f"\nYear 5 patients: {len(year5)}")
print(f"Dropout patients: {len(year1_only)}")
print()

# Continuous — Mann-Whitney
compare_continuous("Baseline A1c", "baseline_a1c", year5, year1_only)
compare_continuous("Age", "approximate_age", year5, year1_only)
if "preop_bmi" in df.columns:
    compare_continuous("Pre-op BMI", "preop_bmi", year5, year1_only)

# Binary — Fisher's exact
compare_binary("Female (%)", "sex", "F", year5, year1_only)
compare_binary("Type 1 diabetes (%)", "diabetes_type", "Type 1", year5, year1_only)
compare_binary("Type 2 diabetes (%)", "diabetes_type", "Type 2", year5, year1_only)
if "is_gpoem" in df.columns and df["is_gpoem"].notna().sum() > 0:
    compare_binary("G-POEM (%)", "is_gpoem", 1, year5, year1_only)

# Categorical distributions — global chi-square
compare_distribution("Race distribution", "race", year5, year1_only)
compare_distribution("Ethnicity distribution", "ethnicity", year5, year1_only)

print("\n* p < 0.05 (unadjusted)")

# ==========================================
# PART C — A1c TRAJECTORY OF YEAR 5 PATIENTS
# ==========================================
print("\n" + "="*70)
print("PART C — A1c TRAJECTORY OF YEAR 5 PATIENTS")
print(f"Threshold: Improved ≤ -{IMPROVEMENT_THRESHOLD}, "
      f"Stable ±{IMPROVEMENT_THRESHOLD}, Worsened ≥ +{IMPROVEMENT_THRESHOLD}")
print("="*70)

print(f"\nBaseline A1c: mean {year5['baseline_a1c'].dropna().mean():.2f} ± "
      f"{year5['baseline_a1c'].dropna().std():.2f}, "
      f"median {year5['baseline_a1c'].dropna().median():.2f}")

for label, col, change_col in [
    ("Year 1","year1_a1c","year1_change"),
    ("Year 2","year2_a1c","year2_change"),
    ("Year 3","year3_a1c","year3_change"),
    ("Year 4","year4_a1c","year4_change"),
    ("Year 5","year5_a1c","year5_change"),
]:
    paired = year5.dropna(subset=["baseline_a1c", col])
    if len(paired) == 0:
        continue
    mean_change = paired[change_col].mean()
    direction = ("↓ Improved" if mean_change < -0.1
                 else "↑ Worsened" if mean_change > 0.1
                 else "→ Flat")
    print(f"\n--- {label} (n={len(paired)}) ---")
    print(f"  Mean A1c: {paired[col].mean():.2f} ± {paired[col].std():.2f}")
    print(f"  Mean change: {mean_change:+.2f}  "
          f"Median: {paired[change_col].median():+.2f}  {direction}")
    print(f"  Improved (≤-{IMPROVEMENT_THRESHOLD}): "
          f"{(paired[change_col] <= -IMPROVEMENT_THRESHOLD).sum()}")
    print(f"  Stable   (±{IMPROVEMENT_THRESHOLD}):  "
          f"{((paired[change_col] > -IMPROVEMENT_THRESHOLD) & (paired[change_col] < IMPROVEMENT_THRESHOLD)).sum()}")
    print(f"  Worsened (≥+{IMPROVEMENT_THRESHOLD}): "
          f"{(paired[change_col] >= IMPROVEMENT_THRESHOLD).sum()}")

# Overall Year 5 summary
year5_paired = year5.dropna(subset=["baseline_a1c","year5_a1c"]).copy()
print(f"\n--- Overall Year 5 net change summary (n={len(year5_paired)}) ---")
print(f"  Improved (≤-{IMPROVEMENT_THRESHOLD}): "
      f"{(year5_paired['year5_change'] <= -IMPROVEMENT_THRESHOLD).sum()}")
print(f"  Stable   (±{IMPROVEMENT_THRESHOLD}):  "
      f"{((year5_paired['year5_change'] > -IMPROVEMENT_THRESHOLD) & (year5_paired['year5_change'] < IMPROVEMENT_THRESHOLD)).sum()}")
print(f"  Worsened (≥+{IMPROVEMENT_THRESHOLD}): "
      f"{(year5_paired['year5_change'] >= IMPROVEMENT_THRESHOLD).sum()}")
print(f"  Net mean change: {year5_paired['year5_change'].mean():+.2f}")
print(f"  Net median change: {year5_paired['year5_change'].median():+.2f}")

print("\n--- Year 5 trajectory by diabetes type ---")
for dtype in ["Type 1","Type 2"]:
    grp = year5_paired[year5_paired["diabetes_type"] == dtype]
    if len(grp) == 0:
        continue
    print(f"\n  {dtype} (n={len(grp)})")
    print(f"    Mean baseline A1c: {grp['baseline_a1c'].mean():.2f}")
    print(f"    Mean Year 5 A1c:   {grp['year5_a1c'].mean():.2f}")
    print(f"    Mean change:       {grp['year5_change'].mean():+.2f}")
    print(f"    Improved: {(grp['year5_change'] <= -IMPROVEMENT_THRESHOLD).sum()}")
    print(f"    Stable:   {((grp['year5_change'] > -IMPROVEMENT_THRESHOLD) & (grp['year5_change'] < IMPROVEMENT_THRESHOLD)).sum()}")
    print(f"    Worsened: {(grp['year5_change'] >= IMPROVEMENT_THRESHOLD).sum()}")

print("\n--- Year 5 trajectory by procedure type ---")
for plabel in ["G-POEM","Pyloroplasty"]:
    grp = year5_paired[year5_paired["procedure_label"] == plabel]
    if len(grp) == 0:
        continue
    print(f"\n  {plabel} (n={len(grp)})")
    print(f"    Mean baseline A1c: {grp['baseline_a1c'].mean():.2f}")
    print(f"    Mean Year 5 A1c:   {grp['year5_a1c'].mean():.2f}")
    print(f"    Mean change:       {grp['year5_change'].mean():+.2f}")

# ==========================================
# PART D — LOGISTIC REGRESSION FOR RETENTION
# ==========================================
print("\n" + "="*70)
print("PART D — LOGISTIC REGRESSION: PREDICTORS OF 5-YEAR RETENTION")
print("Outcome: Has Year 5 data (1=Yes, 0=No)")
print("Population: Patients with Year 1 data")
print("="*70)

lr_df = year1_all.copy()
lr_df["has_year5"] = lr_df["year5_a1c"].notna().astype(int)
lr_df["is_female"] = lr_df["sex"].eq("F").astype(int)
lr_df["is_type1"] = lr_df["diabetes_type"].eq("Type 1").astype(int)

# Base predictors — always included
predictors = ["baseline_a1c","age_decade","is_female","is_type1"]

# Conditionally add is_gpoem
if "is_gpoem" in lr_df.columns and lr_df["is_gpoem"].notna().sum() > 0:
    predictors.append("is_gpoem")

# Conditionally add BMI
if "preop_bmi" in lr_df.columns and lr_df["preop_bmi"].notna().sum() > 10:
    predictors.append("preop_bmi")

lr_clean = lr_df[["has_year5","patient_id"] + predictors].dropna()

print(f"\nPatients in model: {len(lr_clean)}")
print(f"With Year 5:    {lr_clean['has_year5'].sum()}")
print(f"Without Year 5: {(lr_clean['has_year5']==0).sum()}")
print(f"Predictors: {predictors}")
print("\nNOTE: Continuous predictor ORs are per 1-unit increase.")
print("      Age OR is per 10-year increase (age_decade).")
print("      Baseline A1c OR is per 1% increase.")
if "preop_bmi" in predictors:
    print("      BMI OR is per 1 kg/m² increase.")

X = sm.add_constant(lr_clean[predictors].astype(float))
y = lr_clean["has_year5"].astype(float)

results_rows = []
uni_results = []

try:
    model = sm.Logit(y, X).fit(disp=0)

    print("\n--- Multivariate logistic regression ---")
    print(f"\n{'Predictor':<25}{'OR':>10}{'95% CI':>20}{'p-value':>12}")
    print("-" * 70)

    OR = np.exp(model.params)
    CI = np.exp(model.conf_int())
    pvals = model.pvalues

    for pred in predictors:
        or_val = OR[pred]
        ci_low = CI.loc[pred, 0]
        ci_high = CI.loc[pred, 1]
        p = pvals[pred]
        ci_str = f"({ci_low:.3f}–{ci_high:.3f})"
        print(f"  {pred:<23}{or_val:>10.3f}{ci_str:>20}{p:>12.4f}{'*' if p<0.05 else ''}")
        results_rows.append({
            "predictor": pred,
            "OR": round(or_val, 3),
            "CI_low": round(ci_low, 3),
            "CI_high": round(ci_high, 3),
            "p_value": round(p, 4),
            "significant": p < 0.05
        })

    print(f"\n  N: {int(model.nobs)}")
    print(f"  Pseudo R² (McFadden): {model.prsquared:.3f}")
    print(f"  AIC: {model.aic:.2f}")

    print("\n--- Univariate logistic regression ---")
    print(f"\n{'Predictor':<25}{'OR':>10}{'95% CI':>20}{'p-value':>12}")
    print("-" * 70)

    for pred in predictors:
        X_uni = sm.add_constant(lr_clean[[pred]].astype(float))
        try:
            m_uni = sm.Logit(y, X_uni).fit(disp=0)
            or_val = np.exp(m_uni.params[pred])
            ci_low = np.exp(m_uni.conf_int().loc[pred, 0])
            ci_high = np.exp(m_uni.conf_int().loc[pred, 1])
            p = m_uni.pvalues[pred]
            ci_str = f"({ci_low:.3f}–{ci_high:.3f})"
            print(f"  {pred:<23}{or_val:>10.3f}{ci_str:>20}{p:>12.4f}{'*' if p<0.05 else ''}")
            uni_results.append({
                "predictor": pred,
                "OR_univariate": round(or_val, 3),
                "CI_low": round(ci_low, 3),
                "CI_high": round(ci_high, 3),
                "p_value": round(p, 4),
                "significant": p < 0.05
            })
        except Exception:
            print(f"  {pred:<23} model failed")

    print("\n* p < 0.05 (unadjusted)")

    # Save regression results
    pd.DataFrame(results_rows).to_csv(
        "step18_retention_regression_results.csv", index=False
    )
    pd.DataFrame(uni_results).to_csv(
        "step18_retention_univariate_results.csv", index=False
    )

except Exception as e:
    print(f"\nLogistic regression failed: {e}")
    print("Check for perfect separation or insufficient variation.")

# ==========================================
# SAVE
# ==========================================
year5.to_csv("step18_year5_patients.csv", index=False)
year5_paired.to_csv("step18_year5_trajectory.csv", index=False)
lr_clean.to_csv("step18_retention_model_data.csv", index=False)

print("\nSaved: step18_year5_patients.csv")
print("Saved: step18_year5_trajectory.csv")
print("Saved: step18_retention_model_data.csv")
print("Saved: step18_retention_regression_results.csv")
print("Saved: step18_retention_univariate_results.csv")
print("\nStep 18 complete.")
