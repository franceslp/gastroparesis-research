import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

print("Step 16.5: Z-Score Normalization Sensitivity Analysis")
print("Mean A1c change across all available years (Y1-Y5)")
print("Minimum data: baseline + Y1 + Y2 or Y3")
print("Z-score responder: Z <= -2 AND mean change <= -0.5")
print("=" * 70)

# ==========================================
# LOAD DATA
# ==========================================
longitudinal = pd.read_csv("step15_sadda_style_longitudinal_results.csv", dtype=str)
trajectory = pd.read_csv("step16_trajectory_groups.csv", dtype=str)
demographics = pd.read_csv("patient_demographics.csv", dtype=str)

for d in [longitudinal, trajectory, demographics]:
    d["patient_id"] = d["patient_id"].astype(str).str.strip()

# ==========================================
# CONVERT NUMERIC
# ==========================================
year_cols = ["baseline_a1c","year1_a1c","year2_a1c","year3_a1c","year4_a1c","year5_a1c"]
for col in year_cols:
    if col in longitudinal.columns:
        longitudinal[col] = pd.to_numeric(longitudinal[col], errors="coerce")

# ==========================================
# CALCULATE CHANGE SCORES
# ==========================================
longitudinal["y1_change"] = longitudinal["year1_a1c"] - longitudinal["baseline_a1c"]
longitudinal["y2_change"] = longitudinal["year2_a1c"] - longitudinal["baseline_a1c"]
longitudinal["y3_change"] = longitudinal["year3_a1c"] - longitudinal["baseline_a1c"]
longitudinal["y4_change"] = longitudinal["year4_a1c"] - longitudinal["baseline_a1c"]
longitudinal["y5_change"] = longitudinal["year5_a1c"] - longitudinal["baseline_a1c"]

# ==========================================
# APPLY SAME INCLUSION CRITERIA AS STEP 16
# Require: baseline + Y1 + (Y2 or Y3)
# ==========================================
def has_sufficient_data(row):
    if pd.isna(row["baseline_a1c"]) or pd.isna(row["y1_change"]):
        return False
    if pd.notna(row["y2_change"]) or pd.notna(row["y3_change"]):
        return True
    return False

longitudinal["sufficient_data"] = longitudinal.apply(has_sufficient_data, axis=1)

df = longitudinal[longitudinal["sufficient_data"]].copy()
print(f"\nPatients meeting inclusion criteria: {len(df)}")

# ==========================================
# CALCULATE MEAN CHANGE ACROSS ALL AVAILABLE YEARS
# ==========================================
change_cols = ["y1_change","y2_change","y3_change","y4_change","y5_change"]

df["mean_change"] = df[change_cols].mean(axis=1)

print(f"Patients with mean change calculated: {df['mean_change'].notna().sum()}")
print(f"Cohort mean change: {df['mean_change'].mean():.3f}")
print(f"Cohort SD of mean change: {df['mean_change'].std():.3f}")

# ==========================================
# Z-SCORE NORMALIZATION
# ==========================================
mu = df["mean_change"].mean()
sigma = df["mean_change"].std()

print(f"\n=== Z-SCORE NORMALIZATION ===")
print(f"Mean ΔA1c (across all years): {mu:.3f}")
print(f"SD ΔA1c:                      {sigma:.3f}")

if sigma == 0 or pd.isna(sigma):
    raise ValueError("Invalid SD for mean ΔA1c — cannot compute Z-scores")

df["a1c_z"] = (df["mean_change"] - mu) / sigma

# ==========================================
# DEFINE Z-SCORE RESPONDER
# Z <= -2 AND mean change <= -0.5
# ==========================================
df["z_responder"] = (
    (df["a1c_z"] <= -2) &
    (df["mean_change"] <= -0.5)
).astype(int)

print(f"\n=== Z-SCORE RESPONDER SUMMARY ===")
print(f"Z-score responders:     {df['z_responder'].sum()} ({100*df['z_responder'].mean():.1f}%)")
print(f"Z-score non-responders: {(df['z_responder']==0).sum()} ({100*(df['z_responder']==0).mean():.1f}%)")

# ==========================================
# MERGE RULE-BASED RESPONDER FROM STEP 16
# ==========================================
traj_cols = ["patient_id","trajectory","responder","procedure_code","diabetes_type"]
traj_cols = [c for c in traj_cols if c in trajectory.columns]
trajectory["responder"] = pd.to_numeric(trajectory["responder"], errors="coerce")

df = df.merge(trajectory[traj_cols], on="patient_id", how="left")

# ==========================================
# COMPARE TWO DEFINITIONS
# ==========================================
print(f"\n=== COMPARISON: RULE-BASED vs Z-SCORE ===")
print(f"Rule-based responders (1.1% threshold): {df['responder'].sum()} ({100*df['responder'].mean():.1f}%)")
print(f"Z-score responders (Z <= -2):           {df['z_responder'].sum()} ({100*df['z_responder'].mean():.1f}%)")

# Overlap
both = ((df["responder"] == 1) & (df["z_responder"] == 1)).sum()
rule_only = ((df["responder"] == 1) & (df["z_responder"] == 0)).sum()
z_only = ((df["responder"] == 0) & (df["z_responder"] == 1)).sum()
neither = ((df["responder"] == 0) & (df["z_responder"] == 0)).sum()

print(f"\nOverlap analysis:")
print(f"  Both responder definitions:     {both}")
print(f"  Rule-based only:                {rule_only}")
print(f"  Z-score only:                   {z_only}")
print(f"  Neither:                        {neither}")

# Agreement rate
agreement = (both + neither) / len(df)
print(f"\nAgreement between definitions:  {100*agreement:.1f}%")

# ==========================================
# MERGE DEMOGRAPHICS
# ==========================================
demo_cols = ["patient_id","sex","race","ethnicity","approximate_age","patient_regional_location"]
demo_cols = [c for c in demo_cols if c in demographics.columns]
demographics["approximate_age"] = pd.to_numeric(demographics["approximate_age"], errors="coerce")
df = df.merge(demographics[demo_cols], on="patient_id", how="left")

# ==========================================
# DERIVED VARIABLES
# ==========================================
gpoem_codes = ["43999", "43659"]
df["is_gpoem"] = df["procedure_code"].isin(gpoem_codes).astype(int) if "procedure_code" in df.columns else 0
df["high_baseline"] = (pd.to_numeric(df["baseline_a1c"], errors="coerce") >= 7.0).astype(int)

# ==========================================
# FILTER TO VALID Z-RESPONDER VALUES
# ==========================================
df_z = df[df["z_responder"].isin([0, 1])].copy()
z_resp = df_z[df_z["z_responder"] == 1]
z_non_resp = df_z[df_z["z_responder"] == 0]

# ==========================================
# UNIVARIATE ANALYSIS — Z-SCORE RESPONDER
# ==========================================
print(f"\n=== UNIVARIATE ANALYSIS (Z-SCORE RESPONDER) ===")
print(f"Z-score responders:     {len(z_resp)}")
print(f"Z-score non-responders: {len(z_non_resp)}")
print(f"\n{'Predictor':<35}{'Z-Resp':>12}{'Z-NonResp':>12}{'p-value':>12}")
print("-" * 75)

def mannwhitney_row(label, r_col, nr_col):
    r = pd.to_numeric(r_col, errors="coerce").dropna()
    nr = pd.to_numeric(nr_col, errors="coerce").dropna()
    if len(r) < 3 or len(nr) < 3:
        print(f"{label:<35}{'insufficient data':>38}")
        return
    try:
        u, p = stats.mannwhitneyu(r, nr, alternative="two-sided")
        print(f"{label:<35}{r.median():>12.2f}{nr.median():>12.2f}{p:>12.4f}{'*' if p<0.05 else ''}")
    except Exception:
        print(f"{label:<35}{'test failed':>38}")

def chisq_binary_row(label, col):
    if col not in df_z.columns:
        print(f"{label:<35}{'missing column':>38}")
        return
    ct = pd.crosstab(df_z[col], df_z["z_responder"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print(f"{label:<35}{'insufficient variation':>38}")
        return
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        r_pct = 100 * df_z.loc[df_z["z_responder"]==1, col].mean()
        nr_pct = 100 * df_z.loc[df_z["z_responder"]==0, col].mean()
        print(f"{label:<35}{r_pct:>11.1f}%{nr_pct:>11.1f}%{p:>12.4f}{'*' if p<0.05 else ''}")
    except Exception:
        print(f"{label:<35}{'test failed':>38}")

def chisq_row(label, col, resp_val):
    if col not in df_z.columns:
        print(f"{label:<35}{'missing column':>38}")
        return
    if df_z[col].dropna().nunique() < 2:
        print(f"{label:<35}{'insufficient variation':>38}")
        return
    ct = pd.crosstab(df_z[col], df_z["z_responder"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print(f"{label:<35}{'insufficient variation':>38}")
        return
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        r_pct = 100 * df_z.loc[df_z["z_responder"]==1, col].eq(resp_val).mean()
        nr_pct = 100 * df_z.loc[df_z["z_responder"]==0, col].eq(resp_val).mean()
        print(f"{label:<35}{r_pct:>11.1f}%{nr_pct:>11.1f}%{p:>12.4f}{'*' if p<0.05 else ''}")
    except Exception:
        print(f"{label:<35}{'test failed':>38}")

# Continuous
mannwhitney_row("Baseline A1c (median)", z_resp["baseline_a1c"], z_non_resp["baseline_a1c"])
mannwhitney_row("Age (median)", z_resp["approximate_age"], z_non_resp["approximate_age"])
mannwhitney_row("Mean A1c change", z_resp["mean_change"], z_non_resp["mean_change"])

# Categorical
chisq_row("Female (%)", "sex", "F")
chisq_row("Type 2 diabetes (%)", "diabetes_type", "Type 2")
chisq_binary_row("G-POEM (%)", "is_gpoem")
chisq_binary_row("Baseline A1c >= 7.0 (%)", "high_baseline")

print("\n* p < 0.05 (UNADJUSTED P-VALUES)")
print("NOTE: Multiple comparisons not corrected (exploratory analysis).")

# ==========================================
# Z-SCORE DISTRIBUTION SUMMARY
# ==========================================
print(f"\n=== Z-SCORE DISTRIBUTION ===")
print(f"Min Z:    {df['a1c_z'].min():.2f}")
print(f"Max Z:    {df['a1c_z'].max():.2f}")
print(f"Mean Z:   {df['a1c_z'].mean():.2f}")
print(f"Z <= -1:  {(df['a1c_z'] <= -1).sum()} patients")
print(f"Z <= -2:  {(df['a1c_z'] <= -2).sum()} patients")
print(f"Z <= -3:  {(df['a1c_z'] <= -3).sum()} patients")

# ==========================================
# SAVE
# ==========================================
df.to_csv("step16_5_zscore_results.csv", index=False)
print(f"\nSaved: step16_5_zscore_results.csv")
print("Step 16.5 complete.")
