import subprocess
import pandas as pd

BUCKET = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"

print("Step 1: Finding diabetic gastroparesis patients...")
print("Reading in chunks to save memory...")

gastroparesis_code = 'K31.84'

gp_patients = set()
dm_patients = set()

chunk_number = 0
proc = subprocess.Popen(
    ['gsutil', 'cat', f'{BUCKET}/diagnosis.csv'],
    stdout=subprocess.PIPE
)
for chunk in pd.read_csv(proc.stdout, chunksize=100000, dtype=str):
    chunk_number += 1

    # Find gastroparesis patients
    gp_patients.update(
        chunk[chunk['code'] == gastroparesis_code]['patient_id'].tolist()
    )

    # Find ALL Type 1 and Type 2 diabetes patients
    # Using startswith to capture all E10.x and E11.x codes
    dm_mask = chunk['code'].str.startswith('E10') | chunk['code'].str.startswith('E11')
    dm_patients.update(
        chunk[dm_mask]['patient_id'].tolist()
    )

    if chunk_number % 10 == 0:
        print(f"Processed {chunk_number * 100000:,} rows so far...")

print(f"Patients with gastroparesis: {len(gp_patients)}")
print(f"Patients with diabetes (E10.x or E11.x): {len(dm_patients)}")

diabetic_gp_patients = gp_patients & dm_patients
print(f"Patients with BOTH: {len(diabetic_gp_patients)}")

pd.DataFrame(
    list(diabetic_gp_patients),
    columns=['patient_id']
).to_csv('diabetic_gastroparesis_patients.csv', index=False)

print("Saved diabetic gastroparesis patient list!")
print("Done with Step 1!")
