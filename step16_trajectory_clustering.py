import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

print("Step 16: Rule-Based Trajectory Group Analysis")
print("Response threshold: 1.1% A1c change (2 SD median within-patient)")
print("Y2 requirement relaxed: Y3 accepted as substitute if Y2 missing")
print("=" * 70)

THRESHOLD = 1.1
PARTIAL_IMPROVEMENT_THRESHOLD = 0.5

longitudinal = pd.read_csv("step15_sadda_style_longitudinal_results.csv", dtype=str)
demographics = pd.read_csv("patient_demographics.csv", dtype=str)

longitudinal["patient_id"] = longitudinal["patient_id"].astype(str).str.strip()
demographics["patient_id"] = demographics["patient_id"].astype(str).str.strip()

long_dup = longitudinal["patient_id"].duplicated().sum()
demo_dup = demographics["patient_id"].duplicated().sum()
print(f"\nDuplicate longitudinal patients: {long_dup:,}")
print(f"Duplicate demographic patients: {demo_dup:,}")

if long_dup > 0:
    longitudinal = longitudinal.drop_duplicates(subset="patient_id", keep="first")
    print(f"Dropped {long_dup:,} duplicate longitudinal rows")

if demo_dup > 0:
    demographics = demographics.drop_duplicates(subset="patient_id", keep="first")
    print(f"Dropped {demo_dup:,} duplicate demographic rows")

year_cols = ["baseline_a1c","year1_a1c","year2_a1c","year3_a1c","year4_a1c","year5_a1c"]
for col in year_cols:
    if col in longitudinal.columns:
        longitudinal[col] = pd.to_numeric(longitudinal[col], errors="coerce")

demographics["approximate_age"] = pd.to_numeric(demographics["approximate_age"], errors="coerce")

print(f"\ndiabetes_type in longitudinal: {'diabetes_type' in longitudinal.columns}")
print(f"procedure_code in longitudinal: {'procedure_code' in longitudinal.columns}")

demo_cols = ["patient_id","sex","race","ethnicity","patient_regional_location","approximate_age"]
if "diabetes_type" not in longitudinal.columns:
    demo_cols.append("diabetes_type")

df = longitudinal.merge(demographics[demo_cols], on="patient_id", how="left")

df["y1_change"] = df["year1_a1c"] - df["baseline_a1c"]
df["y2_change"] = df["year2_a1c"] - df["baseline_a1c"]
df["y3_change"] = df["year3_a1c"] - df["baseline_a1c"]
df["y4_change"] = df["year4_a1c"] - df["baseline_a1c"]
df["y5_change"] = df["year5_a1c"] - df["baseline_a1c"]

def get_confirmation_change(row):
    if pd.notna(row["y2_change"]):
        return row["y2_change"], "Y2"
    elif pd.notna(row["y3_change"]):
        return row["y3_change"], "Y3"
    else:
        return None, None

def assign_trajectory(row):
    baseline = row["baseline_a1c"]
    y1 = row["y1_change"]
    y2 = row["y2_change"]
    y3 = row["y3_change"]
    y4 = row["y4_change"]
    y5 = row["y5_change"]

    if pd.isna(baseline) or pd.isna(y1):
        return "Insufficient data"

    confirm_change, confirm_label = get_confirmation_change(row)

    if confirm_change is None:
        return "Insufficient data"

    all_changes = [x for x in [y1, y2, y3, y4, y5] if pd.notna(x)]

    if y1 <= -THRESHOLD and confirm_change <= -THRESHOLD:
        return "Early sustained improver"

    if y1 > -THRESHOLD:
        if confirm_change <= -THRESHOLD:
            return "Late improver"
        if pd.notna(y3) and y3 <= -THRESHOLD:
            if pd.isna(y4) or y4 <= -THRESHOLD:
                return "Late improver"

    if y1 >= THRESHOLD and confirm_change >= THRESHOLD:
        return "Worsener"

    partial_hits = sum(c <= -PARTIAL_IMPROVEMENT_THRESHOLD for c in all_changes)
    if partial_hits >= 2:
        return "Partial improver"

    return "Stable non-responder"

df["trajectory"] = df.apply(assign_trajectory, axis=1)

print("\n=== TRAJECTORY GROUP SUMMARY ===")
traj_counts = df["trajectory"].value_counts()
for traj, count in traj_counts.items():
    pct = 100 * count / len(df)
    print(f"  {traj}: {count} ({pct:.1f}%)")

df_analysis = df[df["trajectory"] != "Insufficient data"].copy()
print(f"\nPatients with sufficient data (baseline + Y1 + Y2 or Y3): {len(df_analysis)}")

gpoem_codes = ["43999", "43659"]
df_analysis["is_gpoem"] = df_analysis["procedure_code"].isin(gpoem_codes).astype(int)
df_analysis["high_baseline"] = (df_analysis["baseline_a1c"] >= 7.0).astype(int)
df_analysis["responder"] = df_analysis["trajectory"].isin(
    ["Early sustained improver", "Late improver"]
).astype(int)

resp = df_analysis[df_analysis["responder"] == 1]
non_resp = df_analysis[df_analysis["responder"] == 0]

print("\n=== TRAJECTORY GROUP PROFILES ===")
trajectory_order = [
    "Early sustained improver",
    "Late improver",
    "Partial improver",
    "Stable non-responder",
    "Worsener"
]

for traj in trajectory_order:
    grp = df_analysis[df_analysis["trajectory"] == traj]
    if len(grp) == 0:
        continue
    print(f"\n{traj} (n={len(grp)})")
    print(f"  Mean baseline A1c:   {grp['baseline_a1c'].mean():.2f}")
    print(f"  Median baseline A1c: {grp['baseline_a1c'].median():.2f}")
    print(f"  Mean age:            {grp['approximate_age'].mean():.1f}")
    print(f"  Mean Y1 change:      {grp['y1_change'].mean():.2f}")
    print(f"  Mean Y2 change:      {grp['y2_change'].mean():.2f}")
    print(f"  Female:              {grp['sex'].eq('F').sum()} ({100*grp['sex'].eq('F').mean():.1f}%)")
    print(f"  Type 1 diabetes:     {grp['diabetes_type'].eq('Type 1').sum()} ({100*grp['diabetes_type'].eq('Type 1').mean():.1f}%)")
    print(f"  Type 2 diabetes:     {grp['diabetes_type'].eq('Type 2').sum()} ({100*grp['diabetes_type'].eq('Type 2').mean():.1f}%)")
    print(f"  G-POEM:              {grp['is_gpoem'].sum()} ({100*grp['is_gpoem'].mean():.1f}%)")
    print(f"  Pyloroplasty:        {grp['procedure_code'].eq('43800').sum()} ({100*grp['procedure_code'].eq('43800').mean():.1f}%)")

print("\n=== UNIVARIATE ANALYSIS ===")
print("Responders vs non-responders\n")

if len(df_analysis) > 0:
    print(f"Responders:     {len(resp)} ({100*len(resp)/len(df_analysis):.1f}%)")
    print(f"Non-responders: {len(non_resp)} ({100*len(non_resp)/len(df_analysis):.1f}%)")
else:
    print("No analyzable patients")

print(f"\n{'Predictor':<35}{'Responder':>12}{'Non-Resp':>12}{'p-value':>12}")
print("-" * 75)

r = resp["baseline_a1c"].dropna()
nr = non_resp["baseline_a1c"].dropna()
if len(r) > 0 and len(nr) > 0:
    u, p = stats.mannwhitneyu(r, nr, alternative="two-sided")
    print(f"{'Baseline A1c (median)':<35}{r.median():>12.2f}{nr.median():>12.2f}{p:>12.4f}{'*' if p<0.05 else ''}")

r = resp["approximate_age"].dropna()
nr = non_resp["approximate_age"].dropna()
if len(r) > 0 and len(nr) > 0:
    u, p = stats.mannwhitneyu(r, nr, alternative="two-sided")
    print(f"{'Age (median)':<35}{r.median():>12.1f}{nr.median():>12.1f}{p:>12.4f}{'*' if p<0.05 else ''}")

ct = pd.crosstab(df_analysis["sex"], df_analysis["responder"])
if ct.shape == (2, 2):
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    r_pct = 100 * resp["sex"].eq("F").mean()
    nr_pct = 100 * non_resp["sex"].eq("F").mean()
    print(f"{'Female (%)':<35}{r_pct:>11.1f}%{nr_pct:>11.1f}%{p:>12.4f}{'*' if p<0.05 else ''}")

ct = pd.crosstab(df_analysis["diabetes_type"], df_analysis["responder"])
if ct.shape == (2, 2):
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    r_pct = 100 * resp["diabetes_type"].eq("Type 2").mean()
    nr_pct = 100 * non_resp["diabetes_type"].eq("Type 2").mean()
    print(f"{'Type 2 diabetes (%)':<35}{r_pct:>11.1f}%{nr_pct:>11.1f}%{p:>12.4f}{'*' if p<0.05 else ''}")

ct = pd.crosstab(df_analysis["is_gpoem"], df_analysis["responder"])
if ct.shape == (2, 2):
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    r_pct = 100 * df_analysis.loc[df_analysis["responder"]==1, "is_gpoem"].mean()
    nr_pct = 100 * df_analysis.loc[df_analysis["responder"]==0, "is_gpoem"].mean()
    print(f"{'G-POEM (%)':<35}{r_pct:>11.1f}%{nr_pct:>11.1f}%{p:>12.4f}{'*' if p<0.05 else ''}")

ct = pd.crosstab(df_analysis["high_baseline"], df_analysis["responder"])
if ct.shape == (2, 2):
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    r_pct = 100 * df_analysis.loc[df_analysis["responder"]==1, "high_baseline"].mean()
    nr_pct = 100 * df_analysis.loc[df_analysis["responder"]==0, "high_baseline"].mean()
    print(f"{'Baseline A1c >= 7.0 (%)':<35}{r_pct:>11.1f}%{nr_pct:>11.1f}%{p:>12.4f}{'*' if p<0.05 else ''}")

print("\n* p < 0.05")

df_analysis.to_csv("step16_trajectory_groups.csv", index=False)

df_with_baseline = df[df["baseline_a1c"].notna()].copy()
df_with_baseline.to_csv("step16_all_patients_with_trajectories.csv", index=False)
print(f"\nPatients with baseline A1c saved: {len(df_with_baseline)}")

print("\nSaved: step16_trajectory_groups.csv")
print("Saved: step16_all_patients_with_trajectories.csv")
print("Step 16 complete.")
