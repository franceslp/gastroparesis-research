import subprocess
import pandas as pd

BUCKET = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"

print("Step 7: Surgical confounder analysis")

analysis_cohort = pd.read_csv("analysis_cohort_only.csv")
procedure_dates = pd.read_csv("procedure_dates.csv")
procedure_dates = procedure_dates.dropna(subset=["patient_id", "procedure_date"])
procedure_dates["procedure_date"] = pd.to_datetime(procedure_dates["procedure_date"], errors="coerce")
procedure_dates = procedure_dates.sort_values(["patient_id", "procedure_date"])
procedure_dates = procedure_dates.drop_duplicates("patient_id", keep="first")
procedure_dates = procedure_dates[procedure_dates["patient_id"].isin(analysis_cohort["patient_id"])]
our_patients = set(procedure_dates["patient_id"])
print(f"Checking {len(our_patients)} patients")

PRE_WINDOW = 365
POST_WINDOW = 365

major_surgery_codes = set([
    "43644", "43645", "43770", "43771", "43772", "43773", "43774", "43775",
    "43846", "43847", "43848",
    "48150", "48152", "48153", "48154", "48155", "48146", "48148",
    "43620", "43621", "43622", "43631", "43632", "43633", "43634",
    "50360", "50365", "50370", "48554", "48556",
    "44140", "44141", "44143", "44144", "44145", "44146", "44147"
])

print("\nReading procedure file...")
proc = subprocess.Popen(
    ["gsutil", "cat", f"{BUCKET}/procedure.csv"],
    stdout=subprocess.PIPE
)

surgery_records = []
chunk_number = 0

for chunk in pd.read_csv(proc.stdout, chunksize=100000, dtype=str):
    chunk_number += 1
    chunk = chunk.dropna(subset=["patient_id"])
    chunk = chunk[chunk["patient_id"].isin(our_patients)]
    if chunk.empty:
        continue
    surgeries = chunk[chunk["code"].isin(major_surgery_codes)]
    if not surgeries.empty:
        surgery_records.append(surgeries[["patient_id", "code", "date"]])
    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * 100000:,} rows")

print("Procedure data loaded.")

def get_window_patients(records, proc_dates, start_window, end_window, date_col="date"):
    if not records:
        return set()
    df = pd.concat(records)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.merge(proc_dates[["patient_id", "procedure_date"]], on="patient_id")
    df["delta"] = (df[date_col] - df["procedure_date"]).dt.days
    df = df[(df["delta"] >= start_window) & (df["delta"] <= end_window)]
    return set(df["patient_id"].unique())

print("\nCalculating surgery exclusions...")
surgery_pre = get_window_patients(surgery_records, procedure_dates, -PRE_WINDOW, 0)
surgery_post = get_window_patients(surgery_records, procedure_dates, 0, POST_WINDOW)

# Exclude both pre and post op for now
# Post-op patients can be added back after PI discussion
surgery_exclusions = surgery_pre | surgery_post

print(f"Pre-op surgery exclusions: {len(surgery_pre)}")
print(f"Post-op surgery exclusions: {len(surgery_post)}")
print(f"Total surgery exclusions: {len(surgery_exclusions)}")

# Final clean cohort
final_clean = analysis_cohort[
    ~analysis_cohort["patient_id"].isin(surgery_exclusions)
].copy()

# Save exclusion details
exclude_df = pd.DataFrame({"patient_id": list(surgery_exclusions)})
exclude_df["preop_surgery"] = exclude_df["patient_id"].isin(surgery_pre).astype(int)
exclude_df["postop_surgery"] = exclude_df["patient_id"].isin(surgery_post).astype(int)

final_clean.to_csv("final_clean_patients.csv", index=False)
exclude_df.to_csv("step7_major_surgery_exclusions.csv", index=False)

if surgery_records:
    pd.concat(surgery_records).to_csv("all_major_surgery_records.csv", index=False)

print("\n--- FINAL RESULTS ---")
print(f"Starting analytic cohort: {len(analysis_cohort)}")
print(f"Pre-op surgery exclusions: {len(surgery_pre)}")
print(f"Post-op surgery exclusions: {len(surgery_post)}")
print(f"Total excluded: {len(surgery_exclusions)}")
print(f"Final clean cohort: {len(final_clean)}")
print("\nSaved: final_clean_patients.csv")
print("Saved: step7_major_surgery_exclusions.csv")
print("Saved: all_major_surgery_records.csv")
print("\nDone with Step 7!")
