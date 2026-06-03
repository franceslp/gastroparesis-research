import pandas as pd
import numpy as np

print("Step 15: Sadda-Style Annual A1c Analysis")
print("Most recent A1c value per year for up to 5 years")

# ==========================================
# LOAD DATA
# ==========================================
final_clean = pd.read_csv("final_clean_patients.csv", dtype=str)
a1c_results = pd.read_csv("a1c_results.csv", dtype=str)
procedure_dates = pd.read_csv("procedure_dates.csv", dtype=str)

# Standardize patient IDs
for df in [final_clean, a1c_results, procedure_dates]:
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

# ==========================================
# CLEAN PROCEDURE DATES
# Keep FIRST procedure only
# ==========================================
procedure_dates = procedure_dates.dropna(
    subset=["patient_id", "procedure_date"]
)

procedure_dates["procedure_date"] = pd.to_datetime(
    procedure_dates["procedure_date"],
    errors="coerce"
)

procedure_dates = procedure_dates.dropna(
    subset=["procedure_date"]
)

procedure_dates = procedure_dates.sort_values(
    ["patient_id", "procedure_date"]
)

# KEEP EARLIEST PROCEDURE ONLY
procedure_dates = procedure_dates.drop_duplicates(
    "patient_id",
    keep="first"
)

procedure_dates["procedure_code"] = (
    procedure_dates["procedure_code"]
    .fillna("")
    .astype(str)
    .str.strip()
)

procedure_dates = procedure_dates[
    procedure_dates["patient_id"].isin(final_clean["patient_id"])
]

# ==========================================
# CLEAN A1C DATA
# ==========================================
a1c_results = a1c_results[
    a1c_results["patient_id"].isin(final_clean["patient_id"])
].copy()

a1c_results["date"] = pd.to_datetime(
    a1c_results["date"],
    errors="coerce"
)

a1c_results["lab_result_num_val"] = pd.to_numeric(
    a1c_results["lab_result_num_val"],
    errors="coerce"
)

# Physiologic filter
a1c_results = a1c_results[
    (a1c_results["lab_result_num_val"] >= 4) &
    (a1c_results["lab_result_num_val"] <= 15)
]

a1c_results = a1c_results.dropna(
    subset=["date", "lab_result_num_val"]
)

# Drop old procedure date if present
a1c_results = a1c_results.drop(
    columns=["procedure_date"],
    errors="ignore"
)

# Merge procedure data
a1c_results = a1c_results.merge(
    procedure_dates[
        ["patient_id", "procedure_code", "procedure_date"]
    ],
    on="patient_id",
    how="left"
)

# ==========================================
# CALCULATE DAYS FROM PROCEDURE
# ==========================================
a1c_results["days_from_procedure"] = (
    a1c_results["date"] -
    a1c_results["procedure_date"]
).dt.days

# ==========================================
# SADDA-STYLE ANNUAL WINDOWS
# Skip first month post-op
# ==========================================
ANNUAL_WINDOWS = {
    "baseline": (-365, -1),
    "year1": (30, 365),
    "year2": (366, 730),
    "year3": (731, 1095),
    "year4": (1096, 1460),
    "year5": (1461, 1825)
}

# ==========================================
# FUNCTION TO GET MOST RECENT VALUE
# ==========================================
def get_most_recent(df, start_day, end_day, label):

    subset = df[
        (df["days_from_procedure"] >= start_day) &
        (df["days_from_procedure"] <= end_day)
    ].copy()

    if subset.empty:
        return pd.DataFrame(columns=[
            "patient_id",
            f"{label}_a1c",
            f"{label}_date",
            f"{label}_days"
        ])

    # MOST RECENT VALUE IN WINDOW
    subset = subset.sort_values(
        "days_from_procedure",
        ascending=False
    )

    subset = subset.drop_duplicates(
        subset="patient_id",
        keep="first"
    )

    subset = subset[[
        "patient_id",
        "lab_result_num_val",
        "date",
        "days_from_procedure"
    ]].copy()

    subset.columns = [
        "patient_id",
        f"{label}_a1c",
        f"{label}_date",
        f"{label}_days"
    ]

    return subset

# ==========================================
# EXTRACT EACH TIMEPOINT
# ==========================================
print("\nExtracting annual A1c values...")

baseline = get_most_recent(
    a1c_results,
    ANNUAL_WINDOWS["baseline"][0],
    ANNUAL_WINDOWS["baseline"][1],
    "baseline"
)

year1 = get_most_recent(
    a1c_results,
    ANNUAL_WINDOWS["year1"][0],
    ANNUAL_WINDOWS["year1"][1],
    "year1"
)

year2 = get_most_recent(
    a1c_results,
    ANNUAL_WINDOWS["year2"][0],
    ANNUAL_WINDOWS["year2"][1],
    "year2"
)

year3 = get_most_recent(
    a1c_results,
    ANNUAL_WINDOWS["year3"][0],
    ANNUAL_WINDOWS["year3"][1],
    "year3"
)

year4 = get_most_recent(
    a1c_results,
    ANNUAL_WINDOWS["year4"][0],
    ANNUAL_WINDOWS["year4"][1],
    "year4"
)

year5 = get_most_recent(
    a1c_results,
    ANNUAL_WINDOWS["year5"][0],
    ANNUAL_WINDOWS["year5"][1],
    "year5"
)

# ==========================================
# BUILD LONGITUDINAL DATASET
# ==========================================
longitudinal = pd.DataFrame({
    "patient_id": list(set(final_clean["patient_id"]))
})

for timepoint in [
    baseline,
    year1,
    year2,
    year3,
    year4,
    year5
]:
    longitudinal = longitudinal.merge(
        timepoint,
        on="patient_id",
        how="left"
    )

# ==========================================
# FORCE NUMERIC A1C COLUMNS
# ==========================================
a1c_cols = [
    "baseline_a1c",
    "year1_a1c",
    "year2_a1c",
    "year3_a1c",
    "year4_a1c",
    "year5_a1c"
]

for col in a1c_cols:
    longitudinal[col] = pd.to_numeric(
        longitudinal[col],
        errors="coerce"
    )

# ==========================================
# ADD PROCEDURE CODE
# ==========================================
proc_map = procedure_dates[
    ["patient_id", "procedure_code"]
].drop_duplicates("patient_id")

proc_map["procedure_code"] = (
    proc_map["procedure_code"]
    .astype(str)
)

longitudinal = longitudinal.merge(
    proc_map,
    on="patient_id",
    how="left"
)

# ==========================================
# CALCULATE CHANGES FROM BASELINE
# ==========================================
for label in [
    "year1",
    "year2",
    "year3",
    "year4",
    "year5"
]:
    longitudinal[f"change_{label}"] = (
        longitudinal[f"{label}_a1c"] -
        longitudinal["baseline_a1c"]
    )

# ==========================================
# SUMMARY FUNCTION
# ==========================================
def summarize_year(df, label, year_name):

    temp = df.dropna(
        subset=["baseline_a1c", f"{label}_a1c"]
    ).copy()

    print(f"\n--- {year_name} ---")
    print(f"Patients with data: {len(temp)}")

    if len(temp) == 0:
        return None

    change_col = f"change_{label}"

    mean_change = temp[change_col].mean()

    print(
        f"Mean baseline A1c: "
        f"{temp['baseline_a1c'].mean():.2f}"
    )

    print(
        f"Mean {year_name} A1c: "
        f"{temp[f'{label}_a1c'].mean():.2f}"
    )

    print(f"Mean Change: {mean_change:.2f}")

    print(
        f"Median Change: "
        f"{temp[change_col].median():.2f}"
    )

    print(
        f"Median follow-up day: "
        f"{temp[f'{label}_days'].median():.0f}"
    )

    print(
        f"Improved: "
        f"{(temp[change_col] < 0).sum()}"
    )

    print(
        f"Worsened: "
        f"{(temp[change_col] > 0).sum()}"
    )

    print(
        f"Unchanged: "
        f"{(temp[change_col] == 0).sum()}"
    )

    return mean_change

# ==========================================
# OVERALL RESULTS
# ==========================================
print("\n===================================")
print("OVERALL ANNUAL RESULTS (Sadda-Style)")
print("===================================")

print(f"Total patients: {len(longitudinal):,}")

print(
    f"Patients with baseline A1c: "
    f"{longitudinal['baseline_a1c'].notna().sum()}"
)

if longitudinal["baseline_a1c"].notna().sum() > 0:
    print(
        f"Mean baseline A1c: "
        f"{longitudinal['baseline_a1c'].dropna().mean():.2f}"
    )

y1 = summarize_year(longitudinal, "year1", "YEAR 1")
y2 = summarize_year(longitudinal, "year2", "YEAR 2")
y3 = summarize_year(longitudinal, "year3", "YEAR 3")
y4 = summarize_year(longitudinal, "year4", "YEAR 4")
y5 = summarize_year(longitudinal, "year5", "YEAR 5")

# ==========================================
# PYLOROPLASTY RESULTS
# ==========================================
print("\n===================================")
print("PYLOROPLASTY ANNUAL RESULTS")
print("===================================")

pylo = longitudinal[
    longitudinal["procedure_code"] == "43800"
]

print(f"Total Pyloroplasty patients: {len(pylo)}")

print(
    f"Patients with baseline A1c: "
    f"{pylo['baseline_a1c'].notna().sum()}"
)

if pylo["baseline_a1c"].notna().sum() > 0:
    print(
        f"Mean baseline A1c: "
        f"{pylo['baseline_a1c'].dropna().mean():.2f}"
    )

summarize_year(pylo, "year1", "YEAR 1")
summarize_year(pylo, "year2", "YEAR 2")
summarize_year(pylo, "year3", "YEAR 3")
summarize_year(pylo, "year4", "YEAR 4")
summarize_year(pylo, "year5", "YEAR 5")

# ==========================================
# G-POEM RESULTS
# ==========================================
print("\n===================================")
print("G-POEM ANNUAL RESULTS")
print("===================================")

gpoem = longitudinal[
    longitudinal["procedure_code"].isin([
        "43999",
        "43659"
    ])
]

print(f"Total G-POEM patients: {len(gpoem)}")

print(
    f"Patients with baseline A1c: "
    f"{gpoem['baseline_a1c'].notna().sum()}"
)

if gpoem["baseline_a1c"].notna().sum() > 0:
    print(
        f"Mean baseline A1c: "
        f"{gpoem['baseline_a1c'].dropna().mean():.2f}"
    )

summarize_year(gpoem, "year1", "YEAR 1")
summarize_year(gpoem, "year2", "YEAR 2")
summarize_year(gpoem, "year3", "YEAR 3")
summarize_year(gpoem, "year4", "YEAR 4")
summarize_year(gpoem, "year5", "YEAR 5")

# ==========================================
# TYPE 1 VS TYPE 2
# ==========================================
demographics = pd.read_csv(
    "patient_demographics.csv",
    dtype=str
)

diabetes_type = demographics[
    ["patient_id", "diabetes_type"]
].drop_duplicates("patient_id")

longitudinal = longitudinal.merge(
    diabetes_type,
    on="patient_id",
    how="left"
)

type1 = longitudinal[
    longitudinal["diabetes_type"] == "Type 1"
]

type2 = longitudinal[
    longitudinal["diabetes_type"] == "Type 2"
]

print("\n===================================")
print("TYPE 1 ANNUAL RESULTS")
print("===================================")

print(f"Total Type 1 patients: {len(type1)}")

summarize_year(type1, "year1", "YEAR 1")
summarize_year(type1, "year2", "YEAR 2")
summarize_year(type1, "year3", "YEAR 3")
summarize_year(type1, "year4", "YEAR 4")
summarize_year(type1, "year5", "YEAR 5")

print("\n===================================")
print("TYPE 2 ANNUAL RESULTS")
print("===================================")

print(f"Total Type 2 patients: {len(type2)}")

summarize_year(type2, "year1", "YEAR 1")
summarize_year(type2, "year2", "YEAR 2")
summarize_year(type2, "year3", "YEAR 3")
summarize_year(type2, "year4", "YEAR 4")
summarize_year(type2, "year5", "YEAR 5")

# ==========================================
# COMPARISON SUMMARY
# ==========================================
print("\n===================================")
print("ANNUAL TREND SUMMARY")
print("===================================")

print(
    f"{'Year':<10} "
    f"{'N':<8} "
    f"{'Mean Change':<15} "
    f"{'Direction'}"
)

print("-" * 50)

for label, name, change in [
    ("year1", "Year 1", y1),
    ("year2", "Year 2", y2),
    ("year3", "Year 3", y3),
    ("year4", "Year 4", y4),
    ("year5", "Year 5", y5)
]:

    temp = longitudinal.dropna(
        subset=["baseline_a1c", f"{label}_a1c"]
    )

    n = len(temp)

    if change is None:
        continue

    direction = (
        "Improved"
        if change < 0
        else "Worsened"
    )

    print(
        f"{name:<10} "
        f"{n:<8} "
        f"{change:<15.2f} "
        f"{direction}"
    )

# ==========================================
# SAVE OUTPUT
# ==========================================
longitudinal.to_csv(
    "step15_sadda_style_longitudinal_results.csv",
    index=False
)

print("\nSaved:")
print("step15_sadda_style_longitudinal_results.csv")

print("\nStep 15 complete.")
