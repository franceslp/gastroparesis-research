import pandas as pd
import numpy as np

print("Step 10: A1c Analysis by Diabetes Type")

final_clean = pd.read_csv("final_clean_patients.csv")
demographics = pd.read_csv("patient_demographics.csv")
longitudinal = pd.read_csv("longitudinal_a1c_results.csv")

# Fix procedure code to string
longitudinal["procedure_code"] = longitudinal["procedure_code"].astype(str).str.strip()

diabetes_type = demographics[["patient_id", "diabetes_type"]].drop_duplicates("patient_id")
longitudinal = longitudinal.merge(diabetes_type, on="patient_id", how="left")

print(f"\nPatients with diabetes type info: {longitudinal['diabetes_type'].notna().sum()}")
print(f"Type 1: {(longitudinal['diabetes_type'] == 'Type 1').sum()}")
print(f"Type 2: {(longitudinal['diabetes_type'] == 'Type 2').sum()}")

def summarize_timepoint(df, label, timepoint_name):
    temp = df.dropna(subset=["preop_a1c", f"{label}_a1c"])
    print(f"\n--- {timepoint_name} ---")
    print(f"Paired patients: {len(temp)}")
    if len(temp) == 0:
        return
    change_col = f"change_{label}"
    print(f"Mean PRE-OP A1c: {temp['preop_a1c'].mean():.2f}")
    print(f"Mean {timepoint_name} A1c: {temp[f'{label}_a1c'].mean():.2f}")
    print(f"Mean Change: {temp[change_col].mean():.2f}")
    print(f"Improved: {(temp[change_col] < 0).sum()}")
    print(f"Worsened: {(temp[change_col] > 0).sum()}")
    print(f"Unchanged: {(temp[change_col] == 0).sum()}")

type1 = longitudinal[longitudinal["diabetes_type"] == "Type 1"]
type2 = longitudinal[longitudinal["diabetes_type"] == "Type 2"]

print("\n===================================")
print(f"TYPE 1 DIABETES (n={len(type1)})")
print("===================================")
print(f"Patients with pre-op A1c: {type1['preop_a1c'].notna().sum()}")
if type1['preop_a1c'].notna().sum() > 0:
    print(f"Mean pre-op A1c: {type1['preop_a1c'].dropna().mean():.2f}")
summarize_timepoint(type1, "m3", "3 MONTH")
summarize_timepoint(type1, "m6", "6 MONTH")
summarize_timepoint(type1, "m12", "12 MONTH")

print("\n--- TYPE 1 BY PROCEDURE ---")
type1_pylo = type1[type1["procedure_code"] == "43800"]
type1_gpoem = type1[type1["procedure_code"].isin(["43999", "43659"])]
print(f"Pyloroplasty: {len(type1_pylo)}")
print(f"G-POEM: {len(type1_gpoem)}")
print("\nType 1 — Pyloroplasty:")
summarize_timepoint(type1_pylo, "m3", "3 MONTH")
summarize_timepoint(type1_pylo, "m6", "6 MONTH")
summarize_timepoint(type1_pylo, "m12", "12 MONTH")
print("\nType 1 — G-POEM:")
summarize_timepoint(type1_gpoem, "m3", "3 MONTH")
summarize_timepoint(type1_gpoem, "m6", "6 MONTH")
summarize_timepoint(type1_gpoem, "m12", "12 MONTH")

print("\n===================================")
print(f"TYPE 2 DIABETES (n={len(type2)})")
print("===================================")
print(f"Patients with pre-op A1c: {type2['preop_a1c'].notna().sum()}")
if type2['preop_a1c'].notna().sum() > 0:
    print(f"Mean pre-op A1c: {type2['preop_a1c'].dropna().mean():.2f}")
summarize_timepoint(type2, "m3", "3 MONTH")
summarize_timepoint(type2, "m6", "6 MONTH")
summarize_timepoint(type2, "m12", "12 MONTH")

print("\n--- TYPE 2 BY PROCEDURE ---")
type2_pylo = type2[type2["procedure_code"] == "43800"]
type2_gpoem = type2[type2["procedure_code"].isin(["43999", "43659"])]
print(f"Pyloroplasty: {len(type2_pylo)}")
print(f"G-POEM: {len(type2_gpoem)}")
print("\nType 2 — Pyloroplasty:")
summarize_timepoint(type2_pylo, "m3", "3 MONTH")
summarize_timepoint(type2_pylo, "m6", "6 MONTH")
summarize_timepoint(type2_pylo, "m12", "12 MONTH")
print("\nType 2 — G-POEM:")
summarize_timepoint(type2_gpoem, "m3", "3 MONTH")
summarize_timepoint(type2_gpoem, "m6", "6 MONTH")
summarize_timepoint(type2_gpoem, "m12", "12 MONTH")

print("\n===================================")
print("TYPE 1 vs TYPE 2 COMPARISON SUMMARY")
print("===================================")
for label, name in [("m3", "3 MONTH"), ("m6", "6 MONTH"), ("m12", "12 MONTH")]:
    t1 = type1.dropna(subset=["preop_a1c", f"{label}_a1c"])
    t2 = type2.dropna(subset=["preop_a1c", f"{label}_a1c"])
    print(f"\n{name}:")
    print(f"  Type 1 — n={len(t1)}, Change: {t1[f'change_{label}'].mean():.2f}")
    print(f"  Type 2 — n={len(t2)}, Change: {t2[f'change_{label}'].mean():.2f}")

longitudinal.to_csv("a1c_by_diabetes_type.csv", index=False)
print("\nSaved: a1c_by_diabetes_type.csv")
print("Done with Step 10!")
