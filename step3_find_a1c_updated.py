import subprocess
import pandas as pd

BUCKET = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"

print("Step 3: Extracting ALL A1c lab values for cohort...")

final_clean = pd.read_csv("final_clean_patients.csv", dtype=str)
final_clean["patient_id"] = final_clean["patient_id"].astype(str).str.strip()

procedure_dates = pd.read_csv("procedure_dates.csv", dtype=str)
procedure_dates["patient_id"] = procedure_dates["patient_id"].astype(str).str.strip()
procedure_dates["procedure_code"] = procedure_dates["procedure_code"].astype(str).str.strip()
procedure_dates["procedure_date"] = pd.to_datetime(procedure_dates["procedure_date"], errors="coerce")
procedure_dates = procedure_dates.dropna(subset=["patient_id", "procedure_date"])
procedure_dates = (
    procedure_dates
    .sort_values(["patient_id", "procedure_date"])
    .drop_duplicates("patient_id", keep="first")
)
procedure_dates = procedure_dates[
    procedure_dates["patient_id"].isin(final_clean["patient_id"])
]

our_patients = set(procedure_dates["patient_id"])
print(f"Patients: {len(our_patients):,}", flush=True)

a1c_codes = {"4548-4", "17856-6", "4549-2"}

proc = subprocess.Popen(
    ["gsutil", "cat", f"{BUCKET}/lab_result.csv"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

a1c_records = []
chunk_number = 0

try:
    for chunk in pd.read_csv(proc.stdout, chunksize=100000, dtype=str):
        chunk_number += 1

        required = {"patient_id", "code", "date", "lab_result_num_val"}
        if not required.issubset(chunk.columns):
            print(f"Warning: missing columns in chunk {chunk_number}", flush=True)
            continue

        chunk["patient_id"] = chunk["patient_id"].astype(str).str.strip()
        chunk = chunk[chunk["patient_id"].isin(our_patients)]
        if chunk.empty:
            continue

        chunk["code"] = chunk["code"].astype(str).str.strip()
        chunk = chunk[chunk["code"].isin(a1c_codes)]
        if chunk.empty:
            continue

        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["lab_result_num_val"] = pd.to_numeric(chunk["lab_result_num_val"], errors="coerce")
        chunk = chunk.dropna(subset=["date", "lab_result_num_val"])
        chunk = chunk[chunk["lab_result_num_val"].between(3, 20)]

        if not chunk.empty:
            a1c_records.append(chunk)

        if chunk_number % 10 == 0:
            print(f"Processed ~{chunk_number * 100000:,} rows...", flush=True)

finally:
    proc.stdout.close()
    stderr_output = proc.stderr.read().decode("utf-8", errors="replace").strip()
    proc.wait()
    if stderr_output:
        print(f"\ngsutil stderr output:\n{stderr_output}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gsutil exited with code {proc.returncode}")

print("\nFinished streaming lab data.", flush=True)

if not a1c_records:
    raise ValueError("No A1c records found for cohort.")

all_a1c = pd.concat(a1c_records, ignore_index=True)

all_a1c = all_a1c.merge(
    procedure_dates[["patient_id", "procedure_code", "procedure_date"]],
    on="patient_id",
    how="left"
)

print(f"Total A1c records: {len(all_a1c):,}", flush=True)
print(f"Patients: {all_a1c['patient_id'].nunique():,}", flush=True)

all_a1c.to_csv("a1c_results.csv", index=False)
print("Saved: a1c_results.csv")
print("Step 3 complete.")
