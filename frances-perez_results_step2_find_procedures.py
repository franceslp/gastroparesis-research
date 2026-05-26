import subprocess
import pandas as pd

BUCKET = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"

print("Step 2: Finding patients who had G-POEM or Pyloroplasty...")
print("Reading in chunks to save memory...")

our_patients = pd.read_csv('diabetic_gastroparesis_patients.csv')['patient_id'].tolist()
print(f"Starting with {len(our_patients):,} diabetic gastroparesis patients")

# PI approved codes only
pyloroplasty_codes = ['43800']
gpoem_codes = ['43999', '43659']
all_codes = pyloroplasty_codes + gpoem_codes

procedure_records = []

chunk_number = 0
proc = subprocess.Popen(
    ['gsutil', 'cat', f'{BUCKET}/procedure.csv'],
    stdout=subprocess.PIPE
)
for chunk in pd.read_csv(proc.stdout, chunksize=100000, dtype=str):
    chunk_number += 1
    chunk = chunk[chunk['patient_id'].isin(our_patients)]
    procedures = chunk[chunk['code'].isin(all_codes)][['patient_id', 'code', 'date']]
    if len(procedures) > 0:
        procedure_records.append(procedures)
    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * 100000:,} rows so far...")

all_procedures = pd.concat(procedure_records)
all_procedures.columns = ['patient_id', 'procedure_code', 'procedure_date']
all_procedures['procedure_date'] = pd.to_datetime(
    all_procedures['procedure_date'], format='%Y%m%d'
)
all_procedures = all_procedures.sort_values('procedure_date')
first_procedures = all_procedures.drop_duplicates(subset='patient_id', keep='first')

print(f"\nTotal procedure records found: {len(all_procedures)}")
print(f"Unique patients after keeping first procedure only: {len(first_procedures)}")

pylo = first_procedures[first_procedures['procedure_code'] == '43800']
gpoem = first_procedures[first_procedures['procedure_code'].isin(['43999', '43659'])]
print(f"\nPyloroplasty (43800): {len(pylo)}")
print(f"G-POEM (43999 + 43659): {len(gpoem)}")

first_procedures.to_csv('procedure_patients.csv', index=False)
first_procedures.to_csv('procedure_dates.csv', index=False)
print("\nSaved!")
print("Done with Step 2!")
