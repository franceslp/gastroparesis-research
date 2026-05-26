import subprocess
import pandas as pd

BUCKET = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"

print("Step 3: Finding A1c values for our 710 patients...")
print("Reading in chunks to save memory...")

# Load our procedure patients from step 2
procedure_patients = pd.read_csv('procedure_patients.csv')
our_patients = procedure_patients['patient_id'].tolist()
print(f"Looking for A1c values for {len(our_patients)} patients")

# A1c LOINC codes
a1c_codes = ['4548-4', '4549-2', '17856-6']

# Store all A1c results
a1c_results = []

chunk_number = 0
proc = subprocess.Popen(['gsutil', 'cat', f'{BUCKET}/lab_result.csv'], stdout=subprocess.PIPE)
for chunk in pd.read_csv(proc.stdout, chunksize=100000, dtype=str):
    chunk_number += 1

    # Only look at our patients
    chunk = chunk[chunk['patient_id'].isin(our_patients)]

    # Find A1c results
    a1c_chunk = chunk[chunk['code'].isin(a1c_codes)]

    if len(a1c_chunk) > 0:
        a1c_results.append(a1c_chunk)

    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * 100000:,} rows so far...")

# Combine all results
if a1c_results:
    all_a1c = pd.concat(a1c_results)
    
    # Merge with procedure info to get procedure dates
    all_a1c = all_a1c.merge(procedure_patients, on='patient_id')
    
    # Save results
    all_a1c.to_csv('a1c_results.csv', index=False)
    print(f"Total A1c records found: {len(all_a1c)}")
    print(f"Unique patients with A1c data: {all_a1c['patient_id'].nunique()}")
else:
    print("No A1c results found!")

print("Done with Step 3!")
