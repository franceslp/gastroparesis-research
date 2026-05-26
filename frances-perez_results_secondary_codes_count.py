import subprocess
import pandas as pd

BUCKET = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"

print("Counting secondary procedure codes across diabetic gastroparesis patients...")

our_patients = pd.read_csv('diabetic_gastroparesis_patients.csv')['patient_id'].tolist()
print(f"Checking {len(our_patients):,} diabetic gastroparesis patients")

# Secondary codes to count
secondary_codes = ['43210', '43239', '43620']

code_records = []

chunk_number = 0
proc = subprocess.Popen(
    ['gsutil', 'cat', f'{BUCKET}/procedure.csv'],
    stdout=subprocess.PIPE
)
for chunk in pd.read_csv(proc.stdout, chunksize=100000, dtype=str):
    chunk_number += 1
    chunk = chunk[chunk['patient_id'].isin(our_patients)]
    procedures = chunk[chunk['code'].isin(secondary_codes)][['patient_id', 'code', 'date']]
    if len(procedures) > 0:
        code_records.append(procedures)
    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * 100000:,} rows so far...")

if code_records:
    all_codes = pd.concat(code_records)
    all_codes.columns = ['patient_id', 'procedure_code', 'procedure_date']

    print(f"\n--- SECONDARY CODE BREAKDOWN ---")
    print(f"43210 (G-POEM endoscopic): {len(all_codes[all_codes['procedure_code'] == '43210']['patient_id'].unique())} patients")
    print(f"43239 (G-POEM possible): {len(all_codes[all_codes['procedure_code'] == '43239']['patient_id'].unique())} patients")
    print(f"43620 (Pyloroplasty with vagotomy): {len(all_codes[all_codes['procedure_code'] == '43620']['patient_id'].unique())} patients")

    all_codes.to_csv('secondary_codes_patients.csv', index=False)
    print(f"\nSaved secondary codes patient list!")
else:
    print("No secondary code patients found!")

print("Done!")
