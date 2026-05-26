import subprocess
import pandas as pd

BUCKET = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"

print("Validating all procedure codes with gastroparesis within 30 days before...")

proc = pd.read_csv('procedure_patients.csv')
proc['procedure_code'] = proc['procedure_code'].astype(str)
proc['procedure_date'] = pd.to_datetime(proc['procedure_date'], format='mixed')
our_patients = proc['patient_id'].tolist()
print(f"Checking {len(our_patients)} patients")

print("\nReading diagnosis file...")
diag_records = []
chunk_number = 0

proc2 = subprocess.Popen(
    ['gsutil', 'cat', f'{BUCKET}/diagnosis.csv'],
    stdout=subprocess.PIPE
)
for chunk in pd.read_csv(proc2.stdout, chunksize=100000, dtype=str):
    chunk_number += 1
    chunk = chunk[chunk['patient_id'].isin(our_patients)]
    gp_diag = chunk[chunk['code'] == 'K31.84'][['patient_id', 'date']]
    if len(gp_diag) > 0:
        diag_records.append(gp_diag)
    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * 100000:,} rows so far...")

print("Finished reading diagnosis file!")

if diag_records:
    all_diag = pd.concat(diag_records)
    all_diag['date'] = pd.to_datetime(all_diag['date'], format='%Y%m%d')

    merged = proc.merge(all_diag, on='patient_id')
    merged['days_before_procedure'] = (
        merged['procedure_date'] - merged['date']
    ).dt.days

    # 30 day pre-op window
    preop_diag = merged[
        (merged['days_before_procedure'] >= 0) &
        (merged['days_before_procedure'] <= 30)
    ]

    confirmed_patients = set(preop_diag['patient_id'].unique())
    all_patients = set(proc['patient_id'].unique())
    excluded_patients = all_patients - confirmed_patients

    print(f"\nTotal patients checked: {len(all_patients)}")
    print(f"Confirmed (K31.84 within 30 days before): {len(confirmed_patients)}")
    print(f"Not confirmed (excluded): {len(excluded_patients)}")

    print("\n--- BREAKDOWN BY CODE ---")
    for code in ['43800', '43999', '43659']:
        code_patients = set(proc[proc['procedure_code'] == code]['patient_id'].unique())
        code_confirmed = code_patients & confirmed_patients
        code_excluded = code_patients - confirmed_patients
        print(f"{code}: {len(code_confirmed)} confirmed, {len(code_excluded)} excluded")

    # Save confirmed patients
    confirmed_proc = proc[proc['patient_id'].isin(confirmed_patients)]
    confirmed_proc.to_csv('procedure_patients_validated.csv', index=False)
    print(f"\nFinal validated breakdown:")
    print(confirmed_proc['procedure_code'].value_counts())
    print(f"Total validated patients: {len(confirmed_proc)}")

print("Done!")
