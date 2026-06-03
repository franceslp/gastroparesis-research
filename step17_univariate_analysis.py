import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

print("Step 17: Full Univariate Analysis")
print("Predictors of sustained A1c response after pyloric intervention")
print("=" * 70)

df = pd.read_csv("step16_trajectory_groups.csv", dtype=str)
demographics = pd.read_csv("patient_demographics.csv", dtype=str)
proc_flags = pd.read_csv("procedure_dates.csv", dtype=str)
confounders = pd.read_csv("step6_diabetes_confounders_final.csv", dtype=str)

for d in [df, demographics, proc_flags, confounders]:
    if "patient_id" in d.columns:
        d["patient_id"] = d["patient_id"].astype(str).str.strip()

for col in ["baseline_a1c", "approximate_age", "responder"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "procedure_date" in proc_flags.columns:
    proc_flags["procedure_date"] = pd.to_datetime(proc_flags["procedure_date"], errors="coerce")

demo_cols = [c for c in ["patient_id","race","ethnicity","marital_status"] if c in demographics.columns]
df = df.merge(demographics[demo_cols], on="patient_id", how="left", suffixes=("", "_demo"))

flag_cols = ["flag_any_stimulator","flag_combined_procedure","flag_pre_stimulator","flag_post_stimulator"]
for col in flag_cols:
    if col in proc_flags.columns:
        proc_flags[col] = pd.to_numeric(proc_flags[col], errors="coerce")

if "procedure_date" in proc_flags.columns:
    proc_flags_first = (
        proc_flags.dropna(subset=["procedure_date"])
        .sort_values("procedure_date")
        .groupby("patient_id").first()
        .reset_index()
    )
else:
    proc_flags_first = proc_flags.copy()

merge_cols = [c for c in ["patient_id"] + flag_cols if c in proc_flags_first.columns]
df = df.merge(proc_flags_first[merge_cols], on="patient_id", how="left")

med_cols = ["metformin_baseline","num_secondary_meds","num_secondary_classes"]
for col in med_cols:
    if col in confounders.columns:
        confounders[col] = pd.to_numeric(confounders[col], errors="coerce")

conf_cols = [c for c in ["patient_id"] + med_cols if c in confounders.columns]
df = df.merge(confounders[conf_cols], on="patient_id", how="left")

gpoem_codes = ["43999", "43659", "43210"]
df["is_gpoem"] = df["procedure_code"].isin(gpoem_codes).astype(int) if "procedure_code" in df.columns else 0
df["high_baseline"] = (df["baseline_a1c"] >= 7.0).astype(int) if "baseline_a1c" in df.columns else 0

age_bins = [(0,40,"<40"),(40,55,"40-54"),(55,65,"55-64"),(65,120,"65+")]
df["age_group"] = "Unknown"
if "approximate_age" in df.columns:
    for low, high, label in age_bins:
        df.loc[(df["approximate_age"] >= low) & (df["approximate_age"] < high), "age_group"] = label

for col in flag_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

df["metformin_baseline"] = df["metformin_baseline"].fillna(0).astype(int) if "metformin_baseline" in df.columns else 0

race_col = "race_demo" if "race_demo" in df.columns else "race"
if race_col in df.columns:
    df["race_simplified"] = df[race_col].apply(
        lambda x: x if x in ["White","Black or African American","Asian"] else "Other/Unknown"
    )

eth_col = "ethnicity_demo" if "ethnicity_demo" in df.columns else "ethnicity"
if eth_col in df.columns:
    df["is_hispanic"] = df[eth_col].eq("Hispanic or Latino").astype(int)

df = df[df["responder"].isin([0, 1])].copy()
resp = df[df["responder"] == 1]
non_resp = df[df["responder"] == 0]

print(f"\nTotal analysed: {len(df)}")
print(f"Responders:     {len(resp)} ({100*len(resp)/len(df):.1f}%)")
print(f"Non-responders: {len(non_resp)} ({100*len(non_resp)/len(df):.1f}%)")
print(f"\n{'Predictor':<45}{'Responder':>12}{'Non-Resp':>12}{'p-value':>12}")
print("-" * 85)

def mannwhitney_row(label, r_col, nr_col):
    r = pd.to_numeric(r_col, errors="coerce").dropna()
    nr = pd.to_numeric(nr_col, errors="coerce").dropna()
    if len(r) < 3 or len(nr) < 3:
        print(f"{label:<45}{'insufficient data':>38}")
        return
    try:
        u, p = stats.mannwhitneyu(r, nr, alternative="two-sided")
        print(f"{label:<45}{r.median():>12.2f}{nr.median():>12.2f}{p:>12.4f}{'*' if p<0.05 else ''}")
    except Exception:
        print(f"{label:<45}{'test failed':>38}")

def chisq_row(label, col, resp_val):
    if col not in df.columns:
        print(f"{label:<45}{'missing column':>38}")
        return
    ct = pd.crosstab(df[col], df["responder"])
    if ct.shape[1] != 2 or ct.shape[0] < 2:
        print(f"{label:<45}{'insufficient variation':>38}")
        return
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        r_pct = 100 * resp[col].eq(resp_val).mean()
        nr_pct = 100 * non_resp[col].eq(resp_val).mean()
        print(f"{label:<45}{r_pct:>11.1f}%{nr_pct:>11.1f}%{p:>12.4f}{'*' if p<0.05 else ''}")
    except Exception:
        print(f"{label:<45}{'test failed':>38}")

def chisq_binary_row(label, col):
    if col not in df.columns:
        print(f"{label:<45}{'missing column':>38}")
        return
    ct = pd.crosstab(df[col], df["responder"])
    if ct.shape != (2, 2):
        print(f"{label:<45}{'insufficient variation':>38}")
        return
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        r_pct = 100 * df.loc[df["responder"]==1, col].mean()
        nr_pct = 100 * df.loc[df["responder"]==0, col].mean()
        print(f"{label:<45}{r_pct:>11.1f}%{nr_pct:>11.1f}%{p:>12.4f}{'*' if p<0.05 else ''}")
    except Exception:
        print(f"{label:<45}{'test failed':>38}")

print("\n--- CONTINUOUS PREDICTORS ---")
mannwhitney_row("Baseline A1c (median)", resp["baseline_a1c"], non_resp["baseline_a1c"])
mannwhitney_row("Age (median)", resp["approximate_age"], non_resp["approximate_age"])
if "num_secondary_meds" in df.columns:
    mannwhitney_row("Number of diabetes meds", resp["num_secondary_meds"], non_resp["num_secondary_meds"])
if "num_secondary_classes" in df.columns:
    mannwhitney_row("Number of med classes", resp["num_secondary_classes"], non_resp["num_secondary_classes"])

print("\n--- AGE GROUPS ---")
for low, high, label in age_bins:
    r_pct = 100 * ((resp["approximate_age"] >= low) & (resp["approximate_age"] < high)).mean()
    nr_pct = 100 * ((non_resp["approximate_age"] >= low) & (non_resp["approximate_age"] < high)).mean()
    print(f"  {'Age '+label:<43}{r_pct:>11.1f}%{nr_pct:>11.1f}%")
ct = pd.crosstab(df["age_group"], df["responder"])
if ct.shape[1] == 2 and ct.shape[0] >= 2:
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"  {'Age group overall (chi-square)':<43}{'':>12}{'':>12}{p:>12.4f}{'*' if p<0.05 else ''}")

print("\n--- SEX ---")
chisq_row("Female (%)", "sex", "F")

print("\n--- DIABETES TYPE ---")
chisq_row("Type 2 diabetes (%)", "diabetes_type", "Type 2")

print("\n--- PROCEDURE TYPE ---")
chisq_binary_row("G-POEM vs Pyloroplasty (%G-POEM)", "is_gpoem")

print("\n--- BASELINE A1c THRESHOLD ---")
chisq_binary_row("Baseline A1c >= 7.0 (%)", "high_baseline")

print("\n--- STIMULATOR HISTORY ---")
chisq_binary_row("Any gastric stimulator (%)", "flag_any_stimulator")
chisq_binary_row("Pre-op stimulator (%)", "flag_pre_stimulator")
chisq_binary_row("Post-op stimulator (%)", "flag_post_stimulator")

print("\n--- COMBINED PROCEDURE ---")
chisq_binary_row("Combined procedure (%)", "flag_combined_procedure")

print("\n--- METFORMIN AT BASELINE ---")
chisq_binary_row("Metformin at baseline (%)", "metformin_baseline")

print("\n--- RACE ---")
if race_col in df.columns:
    for race in ["White","Black or African American","Asian","Other Race","Unknown"]:
        r_pct = 100 * resp[race_col].eq(race).mean()
        nr_pct = 100 * non_resp[race_col].eq(race).mean()
        print(f"  {race:<43}{r_pct:>11.1f}%{nr_pct:>11.1f}%")
    ct = pd.crosstab(df["race_simplified"], df["responder"])
    if ct.shape[1] == 2 and ct.shape[0] >= 2:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        print(f"  {'Race overall (chi-square)':<43}{'':>12}{'':>12}{p:>12.4f}{'*' if p<0.05 else ''}")

print("\n--- ETHNICITY ---")
if eth_col in df.columns:
    for eth in ["Hispanic or Latino","Not Hispanic or Latino","Unknown"]:
        r_pct = 100 * resp[eth_col].eq(eth).mean()
        nr_pct = 100 * non_resp[eth_col].eq(eth).mean()
        print(f"  {eth:<43}{r_pct:>11.1f}%{nr_pct:>11.1f}%")
    chisq_binary_row("Hispanic or Latino (%)", "is_hispanic")

print("\n--- REGIONAL LOCATION ---")
if "patient_regional_location" in df.columns:
    for region in sorted(df["patient_regional_location"].dropna().unique()):
        r_pct = 100 * resp["patient_regional_location"].eq(region).mean()
        nr_pct = 100 * non_resp["patient_regional_location"].eq(region).mean()
        print(f"  {region:<43}{r_pct:>11.1f}%{nr_pct:>11.1f}%")
    ct = pd.crosstab(df["patient_regional_location"].fillna("Unknown"), df["responder"])
    if ct.shape[1] == 2 and ct.shape[0] >= 2:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        print(f"  {'Region overall (chi-square)':<43}{'':>12}{'':>12}{p:>12.4f}{'*' if p<0.05 else ''}")

print("\n--- MARITAL STATUS ---")
if "marital_status" in df.columns:
    for status in sorted(df["marital_status"].dropna().unique()):
        r_pct = 100 * resp["marital_status"].eq(status).mean()
        nr_pct = 100 * non_resp["marital_status"].eq(status).mean()
        print(f"  {status:<43}{r_pct:>11.1f}%{nr_pct:>11.1f}%")
    ct = pd.crosstab(df["marital_status"].fillna("Unknown"), df["responder"])
    if ct.shape[1] == 2 and ct.shape[0] >= 2:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        print(f"  {'Marital status overall (chi-square)':<43}{'':>12}{'':>12}{p:>12.4f}{'*' if p<0.05 else ''}")

print("\n* p < 0.05")
print("\nNote: BMI and diabetes duration excluded due to VM disk space constraints.")
ENDOFFILE
