import subprocess
import pandas as pd
from datetime import datetime

BUCKET = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"

print("Step 9: Demographics Analysis")

final_clean = pd.read_csv("final_clean_patients.csv")
final_clean = final_clean.drop_duplicates("patient_id")
our_patients = set(final_clean["patient_id"])
print(f"Final clean patients: {len(our_patients)}")

procedure_dates = pd.read_csv("procedure_dates.csv")
procedure_dates = procedure_dates.dropna(subset=["patient_id", "procedure_date"])
procedure_dates["procedure_date"] = pd.to_datetime(procedure_dates["procedure_date"], errors="coerce")
procedure_dates = procedure_dates.sort_values(["patient_id", "procedure_date"])
procedure_dates = procedure_dates.drop_duplicates("patient_id", keep="first")
procedure_dates["procedure_code"] = procedure_dates["procedure_code"].fillna("").astype(str).str.strip()
procedure_dates = procedure_dates[procedure_dates["patient_id"].isin(our_patients)]

print("\nReading patient demographics file...")
proc = subprocess.Popen(["gsutil", "cat", f"{BUCKET}/patient.csv"], stdout=subprocess.PIPE)
patient_df = pd.read_csv(proc.stdout, dtype=str)
print(f"Total patient rows loaded: {len(patient_df):,}")

our_demographics = patient_df[patient_df["patient_id"].isin(our_patients)].copy()
our_demographics = our_demographics.drop_duplicates("patient_id")
print(f"Demographics found for {len(our_demographics)} patients")

our_demographics = our_demographics.merge(
    procedure_dates[["patient_id", "procedure_code", "procedure_date"]],
    on="patient_id", how="left"
)

our_demographics["year_of_birth"] = pd.to_numeric(our_demographics["year_of_birth"], errors="coerce")
our_demographics["procedure_year"] = pd.to_datetime(our_demographics["procedure_date"], errors="coerce").dt.year
our_demographics["approximate_age"] = our_demographics["procedure_year"] - our_demographics["year_of_birth"]
our_demographics.loc[
    (our_demographics["approximate_age"] < 18) | (our_demographics["approximate_age"] > 100),
    "approximate_age"
] = pd.NA

for col in ["sex", "race", "ethnicity", "patient_regional_location", "marital_status"]:
    if col in our_demographics.columns:
        our_demographics[col] = our_demographics[col].fillna("Unknown").astype(str).str.strip()

print("\nDetermining diabetes type from diagnosis file...")
diag_records = []
chunk_number = 0
proc2 = subprocess.Popen(["gsutil", "cat", f"{BUCKET}/diagnosis.csv"], stdout=subprocess.PIPE)
for chunk in pd.read_csv(proc2.stdout, chunksize=100000, dtype=str):
    chunk_number += 1
    chunk = chunk[chunk["patient_id"].isin(our_patients)]
    if not chunk.empty:
        type1 = chunk[chunk["code"].str.startswith("E10", na=False)][["patient_id", "code"]]
        type2 = chunk[chunk["code"].str.startswith("E11", na=False)][["patient_id", "code"]]
        if len(type1) > 0:
            diag_records.append(type1.assign(diabetes_type="Type 1"))
        if len(type2) > 0:
            diag_records.append(type2.assign(diabetes_type="Type 2"))
    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * 100000:,} diagnosis rows so far...")

print("Finished reading diagnosis file!")

if diag_records:
    all_diag = pd.concat(diag_records)
    type1_patients = set(all_diag[all_diag["diabetes_type"] == "Type 1"]["patient_id"])
    type2_patients = set(all_diag[all_diag["diabetes_type"] == "Type 2"]["patient_id"])
    def assign_diabetes_type(patient_id):
        if patient_id in type1_patients:
            return "Type 1"
        elif patient_id in type2_patients:
            return "Type 2"
        return "Unknown"
    our_demographics["diabetes_type"] = our_demographics["patient_id"].apply(assign_diabetes_type)
else:
    our_demographics["diabetes_type"] = "Unknown"

pylo = our_demographics[our_demographics["procedure_code"] == "43800"].copy()
gpoem = our_demographics[our_demographics["procedure_code"].isin(["43999", "43659"])].copy()

def print_counts_pct(series, label):
    counts = series.value_counts(dropna=False)
    pct = series.value_counts(normalize=True, dropna=False).mul(100).round(1)
    print(f"\n{label}")
    for val in counts.index:
        print(f"  {val}: {counts[val]} ({pct[val]}%)")

def print_age_stats(series, label):
    valid = series.dropna()
    print(f"\n{label}")
    if len(valid) > 0:
        print(f"  Patients with valid age: {len(valid)}")
        print(f"  Mean age: {valid.mean():.1f}")
        print(f"  Median age: {valid.median():.1f}")
        print(f"  Min age: {valid.min():.0f}")
        print(f"  Max age: {valid.max():.0f}")
    else:
        print("  No valid age data")

def print_section(df, label):
    print(f"\n===================================")
    print(f"{label} (n={len(df)})")
    print(f"===================================")
    print_age_stats(df["approximate_age"], "AGE AT PROCEDURE")
    print_counts_pct(df["sex"], "SEX")
    print_counts_pct(df["race"], "RACE")
    print_counts_pct(df["ethnicity"], "ETHNICITY")
    print_counts_pct(df["patient_regional_location"], "REGIONAL LOCATION")
    if "marital_status" in df.columns:
        print_counts_pct(df["marital_status"], "MARITAL STATUS")
    print_counts_pct(df["diabetes_type"], "DIABETES TYPE")

print_section(our_demographics, "OVERALL DEMOGRAPHICS")

print("\n===================================")
print("AGE COMPARISON BY PROCEDURE")
print("===================================")
print(f"\nMean age Pyloroplasty: {pylo['approximate_age'].mean():.1f}")
print(f"Mean age G-POEM: {gpoem['approximate_age'].mean():.1f}")
print(f"Median age Pyloroplasty: {pylo['approximate_age'].median():.1f}")
print(f"Median age G-POEM: {gpoem['approximate_age'].median():.1f}")

print_section(pylo, "PYLOROPLASTY DEMOGRAPHICS")
print_section(gpoem, "G-POEM DEMOGRAPHICS")

print("\n===================================")
print("TYPE 1 DIABETES SUBSET")
print("===================================")
type1_df = our_demographics[our_demographics["diabetes_type"] == "Type 1"]
print(f"Total Type 1 patients: {len(type1_df)}")
print(f"Pyloroplasty: {len(type1_df[type1_df['procedure_code'] == '43800'])}")
print(f"G-POEM: {len(type1_df[type1_df['procedure_code'].isin(['43999', '43659'])])}")
print_age_stats(type1_df["approximate_age"], "AGE AT PROCEDURE")
print_counts_pct(type1_df["sex"], "SEX")

print("\n===================================")
print("TYPE 2 DIABETES SUBSET")
print("===================================")
type2_df = our_demographics[our_demographics["diabetes_type"] == "Type 2"]
print(f"Total Type 2 patients: {len(type2_df)}")
print(f"Pyloroplasty: {len(type2_df[type2_df['procedure_code'] == '43800'])}")
print(f"G-POEM: {len(type2_df[type2_df['procedure_code'].isin(['43999', '43659'])])}")
print_age_stats(type2_df["approximate_age"], "AGE AT PROCEDURE")
print_counts_pct(type2_df["sex"], "SEX")

our_demographics.to_csv("patient_demographics.csv", index=False)
print("\nSaved: patient_demographics.csv")
print("Done with Step 9!")
