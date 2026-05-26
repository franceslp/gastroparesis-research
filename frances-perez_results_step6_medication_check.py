import subprocess
import pandas as pd
import re

BUCKET = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"

print("Step 6: Diabetes confounders + post-op sensitivity analysis (CLINICAL-GRADE)")

procedure_dates = pd.read_csv("procedure_dates.csv")
procedure_dates = procedure_dates.dropna(subset=["patient_id", "procedure_date"])
procedure_dates["procedure_date"] = pd.to_datetime(
    procedure_dates["procedure_date"], errors="coerce"
)
procedure_dates = procedure_dates.sort_values(["patient_id", "procedure_date"])
procedure_dates = procedure_dates.drop_duplicates("patient_id", keep="first")
our_patients = set(procedure_dates["patient_id"])

PRE_WINDOW = 365
POST_WINDOW = 365

def make_pattern(keywords):
    return re.compile(
        r"\b(" + "|".join(map(re.escape, keywords)) + r")\b",
        re.IGNORECASE
    )

insulin_keywords = [
    "insulin", "lantus", "basaglar", "toujeo", "tresiba",
    "humalog", "novolog", "levemir", "fiasp", "humulin", "novolin"
]

glp1_keywords = [
    "ozempic", "wegovy", "mounjaro", "victoza", "trulicity",
    "rybelsus", "semaglutide", "tirzepatide", "liraglutide"
]

secondary_keywords = [
    "metformin", "glucophage",
    "jardiance", "farxiga", "invokana",
    "glipizide", "glimepiride", "glyburide",
    "januvia", "tradjenta", "onglyza",
    "actos", "avandia", "pioglitazone", "rosiglitazone"
]

insulin_pat = make_pattern(insulin_keywords)
glp1_pat = make_pattern(glp1_keywords)
secondary_pat = make_pattern(secondary_keywords)

device_codes = set(["95249", "95250", "95251", "E0784", "K0553", "K0554"])

print("\nReading medication file...")
proc = subprocess.Popen(
    ["gsutil", "cat", f"{BUCKET}/medication_drug.csv"],
    stdout=subprocess.PIPE
)

insulin, glp1, secondary = [], [], []
chunk_number = 0

for chunk in pd.read_csv(proc.stdout, chunksize=100000, dtype=str):
    chunk_number += 1
    chunk = chunk.dropna(subset=["patient_id"])
    chunk = chunk[chunk["patient_id"].isin(our_patients)]
    if chunk.empty:
        continue

    chunk["brand"] = chunk["brand"].fillna("").str.lower()

    insulin_mask = chunk["brand"].str.contains(insulin_pat.pattern, na=False, regex=True)
    glp1_mask = chunk["brand"].str.contains(glp1_pat.pattern, na=False, regex=True)
    secondary_mask = chunk["brand"].str.contains(secondary_pat.pattern, na=False, regex=True)
    secondary_mask = secondary_mask & ~insulin_mask & ~glp1_mask

    if insulin_mask.any():
        insulin.append(chunk[insulin_mask][["patient_id", "brand", "start_date"]])
    if glp1_mask.any():
        glp1.append(chunk[glp1_mask][["patient_id", "brand", "start_date"]])
    if secondary_mask.any():
        secondary.append(chunk[secondary_mask][["patient_id", "brand", "start_date"]])

    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * 100000:,} medication rows so far...")

print("Medication data loaded.")

print("\nReading procedure file for devices...")
proc2 = subprocess.Popen(
    ["gsutil", "cat", f"{BUCKET}/procedure.csv"],
    stdout=subprocess.PIPE
)

devices = []
chunk_number = 0

for chunk in pd.read_csv(proc2.stdout, chunksize=100000, dtype=str):
    chunk_number += 1
    chunk = chunk.dropna(subset=["patient_id"])
    chunk = chunk[chunk["patient_id"].isin(our_patients)]
    if chunk.empty:
        continue

    match = chunk[chunk["code"].isin(device_codes)]
    if not match.empty:
        devices.append(match[["patient_id", "code", "date"]])

    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * 100000:,} procedure rows so far...")

print("Device data loaded.")

def get_window_patients(records, proc_dates, start, end, date_col="start_date"):
    if not records:
        return set()
    df = pd.concat(records)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.merge(proc_dates[["patient_id", "procedure_date"]], on="patient_id")
    df["delta"] = (df[date_col] - df["procedure_date"]).dt.days
    window_df = df[(df["delta"] >= start) & (df["delta"] <= end)]
    return set(window_df["patient_id"].unique())

print("\nCalculating pre-op confounders...")
insulin_pre = get_window_patients(insulin, procedure_dates, -PRE_WINDOW, 0)
glp1_pre = get_window_patients(glp1, procedure_dates, -PRE_WINDOW, 0)
device_pre = get_window_patients(devices, procedure_dates, -PRE_WINDOW, 0, date_col="date")
major_preop_change = insulin_pre | glp1_pre | device_pre

print(f"Insulin pre-op changes: {len(insulin_pre)} patients")
print(f"GLP-1 pre-op changes: {len(glp1_pre)} patients")
print(f"Device pre-op changes: {len(device_pre)} patients")

print("\nCalculating post-op sensitivity...")
insulin_post = get_window_patients(insulin, procedure_dates, 0, POST_WINDOW)
glp1_post = get_window_patients(glp1, procedure_dates, 0, POST_WINDOW)
device_post = get_window_patients(devices, procedure_dates, 0, POST_WINDOW, date_col="date")
major_postop_change = insulin_post | glp1_post | device_post

print(f"Insulin post-op changes: {len(insulin_post)} patients")
print(f"GLP-1 post-op changes: {len(glp1_post)} patients")
print(f"Device post-op changes: {len(device_post)} patients")

secondary_df = pd.concat(secondary) if secondary else pd.DataFrame()

if not secondary_df.empty:
    secondary_df["start_date"] = pd.to_datetime(secondary_df["start_date"], errors="coerce")
    secondary_df = secondary_df.merge(
        procedure_dates[["patient_id", "procedure_date"]], on="patient_id"
    )
    secondary_df["delta"] = (
        secondary_df["start_date"] - secondary_df["procedure_date"]
    ).dt.days
    cov = secondary_df[
        (secondary_df["delta"] >= -PRE_WINDOW) &
        (secondary_df["delta"] <= 0)
    ]
    covariates = cov.groupby("patient_id").agg(
        num_secondary_meds=("brand", "count"),
        num_secondary_classes=("brand", "nunique")
    ).reset_index()
else:
    covariates = pd.DataFrame(columns=[
        "patient_id", "num_secondary_meds", "num_secondary_classes"
    ])

final_df = pd.DataFrame({"patient_id": list(our_patients)})
final_df["preop_major_change"] = final_df["patient_id"].isin(major_preop_change).astype(int)
final_df["postop_major_change"] = final_df["patient_id"].isin(major_postop_change).astype(int)
final_df = final_df.merge(covariates, on="patient_id", how="left")
final_df.fillna(0, inplace=True)

analysis_cohort = final_df[final_df["preop_major_change"] == 0].copy()

final_df.to_csv("step6_diabetes_confounders_final.csv", index=False)
analysis_cohort.to_csv("analysis_cohort_only.csv", index=False)

print("\n--- FINAL RESULTS ---")
print(f"Total patients: {len(final_df)}")
print(f"Pre-op excluded (Group B): {final_df['preop_major_change'].sum()}")
print(f"Post-op changes (sensitivity): {final_df['postop_major_change'].sum()}")
print(f"Analytic cohort Group A: {len(analysis_cohort)}")
print("\nSaved: step6_diabetes_confounders_final.csv")
print("Saved: analysis_cohort_only.csv")
print("Done with Step 6!")
