import pandas as pd

print("Step 5: A1c Before vs After Analysis...")
print("Using final clean cohort of 1,053 patients")

# Load final clean patients
final_clean = pd.read_csv('final_clean_patients.csv')
print(f"Final clean patients: {len(final_clean)}")

# Load A1c results - already has procedure info!
a1c_data = pd.read_csv('a1c_results.csv')

# Filter to final clean patients only
a1c_data = a1c_data[a1c_data['patient_id'].isin(final_clean['patient_id'])]
print(f"A1c records for final clean patients: {len(a1c_data)}")

# Fix dates
a1c_data['date'] = pd.to_datetime(a1c_data['date'], format='%Y%m%d')
a1c_data['procedure_date'] = pd.to_datetime(a1c_data['procedure_date'])
a1c_data['procedure_code'] = a1c_data['procedure_code'].astype(str)

# Convert A1c values
a1c_data['lab_result_num_val'] = pd.to_numeric(a1c_data['lab_result_num_val'], errors='coerce')
a1c_data = a1c_data.dropna(subset=['lab_result_num_val'])
a1c_data = a1c_data[
    (a1c_data['lab_result_num_val'] >= 4) &
    (a1c_data['lab_result_num_val'] <= 15)
]
print(f"Valid A1c records: {len(a1c_data)}")

# Calculate days from procedure
a1c_data['days_from_procedure'] = (
    a1c_data['date'] - a1c_data['procedure_date']
).dt.days

# Split before and after
before = a1c_data[
    (a1c_data['days_from_procedure'] >= -365) &
    (a1c_data['days_from_procedure'] < 0)
]
after = a1c_data[
    (a1c_data['days_from_procedure'] > 0) &
    (a1c_data['days_from_procedure'] <= 365)
]

print(f"\n--- OVERALL RESULTS ---")
print(f"Patients with A1c BEFORE: {before['patient_id'].nunique()}")
print(f"Patients with A1c AFTER: {after['patient_id'].nunique()}")
print(f"Average A1c BEFORE: {before['lab_result_num_val'].mean():.2f}")
print(f"Average A1c AFTER: {after['lab_result_num_val'].mean():.2f}")
print(f"A1c Change: {(after['lab_result_num_val'].mean() - before['lab_result_num_val'].mean()):.2f}")

print(f"\n--- PYLOROPLASTY (43800) ---")
pylo_before = before[before['procedure_code'] == '43800']
pylo_after = after[after['procedure_code'] == '43800']
print(f"Patients with BEFORE A1c: {pylo_before['patient_id'].nunique()}")
print(f"Patients with AFTER A1c: {pylo_after['patient_id'].nunique()}")
if len(pylo_before) > 0:
    print(f"Average A1c BEFORE: {pylo_before['lab_result_num_val'].mean():.2f}")
if len(pylo_after) > 0:
    print(f"Average A1c AFTER: {pylo_after['lab_result_num_val'].mean():.2f}")
if len(pylo_before) > 0 and len(pylo_after) > 0:
    print(f"A1c Change: {(pylo_after['lab_result_num_val'].mean() - pylo_before['lab_result_num_val'].mean()):.2f}")

print(f"\n--- G-POEM (43999 + 43659) ---")
gpoem_before = before[before['procedure_code'].isin(['43999', '43659'])]
gpoem_after = after[after['procedure_code'].isin(['43999', '43659'])]
print(f"Patients with BEFORE A1c: {gpoem_before['patient_id'].nunique()}")
print(f"Patients with AFTER A1c: {gpoem_after['patient_id'].nunique()}")
if len(gpoem_before) > 0:
    print(f"Average A1c BEFORE: {gpoem_before['lab_result_num_val'].mean():.2f}")
if len(gpoem_after) > 0:
    print(f"Average A1c AFTER: {gpoem_after['lab_result_num_val'].mean():.2f}")
if len(gpoem_before) > 0 and len(gpoem_after) > 0:
    print(f"A1c Change: {(gpoem_after['lab_result_num_val'].mean() - gpoem_before['lab_result_num_val'].mean()):.2f}")

# Save results
before.to_csv('a1c_before.csv', index=False)
after.to_csv('a1c_after.csv', index=False)
print("\nSaved before and after files!")
print("Done with Step 5!")
