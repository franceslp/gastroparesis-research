import pandas as pd
import numpy as np

print("Step 8: Longitudinal Paired A1c Analysis")
print("Timepoints: Pre-op, 3 month, 6 month, 12 month")

final_clean = pd.read_csv("final_clean_patients.csv")
print(f"Final clean cohort: {len(final_clean)} patients")

procedure_dates = pd.read_csv("procedure_dates.csv")
procedure_dates = procedure_dates.dropna(subset=["patient_id", "procedure_date"])
procedure_dates["procedure_date"] = pd.to_datetime(procedure_dates["procedure_date"], errors="coerce")
procedure_dates = procedure_dates.sort_values(["patient_id", "procedure_date"])
procedure_dates = procedure_dates.drop_duplicates("patient_id", keep="first")
procedure_dates = procedure_dates[procedure_dates["patient_id"].isin(final_clean["patient_id"])].copy()

a1c_data = pd.read_csv("a1c_results.csv")
a1c_data = a1c_data[a1c_data["patient_id"].isin(final_clean["patient_id"])].copy()
a1c_data["date"] = pd.to_datetime(a1c_data["date"], format="%Y%m%d", errors="coerce")
a1c_data["lab_result_num_val"] = pd.to_numeric(a1c_data["lab_result_num_val"], errors="coerce")
a1c_data = a1c_data.drop(columns=["procedure_date"], errors="ignore")
a1c_data = a1c_data.merge(
    procedure_dates[["patient_id", "procedure_date"]], on="patient_id", how="left"
)
a1c_data = a1c_data[
    (a1c_data["lab_result_num_val"] >= 4) &
    (a1c_data["lab_result_num_val"] <= 15)
].copy()
a1c_data = a1c_data.dropna(subset=["date", "procedure_date", "lab_result_num_val"])
a1c_data["days_from_procedure"] = (a1c_data["date"] - a1c_data["procedure_date"]).dt.days

WINDOWS = {
    "preop": (-365, -1),
    "m3": (60, 120),
    "m6": (150, 210),
    "m12": (300, 395)
}

TARGET_DAYS = {
    "preop": -1,
    "m3": 90,
    "m6": 180,
    "m12": 365
}

def get_closest_measurement(df, start_day, end_day, target_day, label):
    subset = df[
        (df["days_from_procedure"] >= start_day) &
        (df["days_from_procedure"] <= end_day)
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=["patient_id", f"{label}_a1c", f"{label}_date", f"{label}_days"])
    subset["distance_to_target"] = (subset["days_from_procedure"] - target_day).abs()
    subset = subset.sort_values("distance_to_target")
    subset = subset.drop_duplicates(subset="patient_id", keep="first")
    subset = subset[["patient_id", "lab_result_num_val", "date", "days_from_procedure"]].copy()
    subset.columns = ["patient_id", f"{label}_a1c", f"{label}_date", f"{label}_days"]
    return subset

print("\nExtracting longitudinal A1c values...")
preop = get_closest_measurement(a1c_data, WINDOWS["preop"][0], WINDOWS["preop"][1], TARGET_DAYS["preop"], "preop")
m3 = get_closest_measurement(a1c_data, WINDOWS["m3"][0], WINDOWS["m3"][1], TARGET_DAYS["m3"], "m3")
m6 = get_closest_measurement(a1c_data, WINDOWS["m6"][0], WINDOWS["m6"][1], TARGET_DAYS["m6"], "m6")
m12 = get_closest_measurement(a1c_data, WINDOWS["m12"][0], WINDOWS["m12"][1], TARGET_DAYS["m12"], "m12")

longitudinal = final_clean[["patient_id"]].copy()
longitudinal = longitudinal.merge(preop, on="patient_id", how="left")
longitudinal = longitudinal.merge(m3, on="patient_id", how="left")
longitudinal = longitudinal.merge(m6, on="patient_id", how="left")
longitudinal = longitudinal.merge(m12, on="patient_id", how="left")

longitudinal["change_m3"] = longitudinal["m3_a1c"] - longitudinal["preop_a1c"]
longitudinal["change_m6"] = longitudinal["m6_a1c"] - longitudinal["preop_a1c"]
longitudinal["change_m12"] = longitudinal["m12_a1c"] - longitudinal["preop_a1c"]

if "procedure_code" in procedure_dates.columns:
    proc_map = procedure_dates[["patient_id", "procedure_code"]].drop_duplicates("patient_id")
    proc_map["procedure_code"] = proc_map["procedure_code"].astype(str)
    longitudinal = longitudinal.merge(proc_map, on="patient_id", how="left")

def summarize_timepoint(df, label, timepoint_name):
    temp = df.dropna(subset=["preop_a1c", f"{label}_a1c"])
    print(f"\n--- {timepoint_name} RESULTS ---")
    print(f"Paired patients: {len(temp)}")
    if len(temp) == 0:
        return
    change_col = f"change_{label}"
    print(f"Mean PRE-OP A1c: {temp['preop_a1c'].mean():.2f}")
    print(f"Mean {timepoint_name} A1c: {temp[f'{label}_a1c'].mean():.2f}")
    print(f"Mean Change: {temp[change_col].mean():.2f}")
    print(f"Improved (A1c decreased): {(temp[change_col] < 0).sum()}")
    print(f"Worsened (A1c increased): {(temp[change_col] > 0).sum()}")
    print(f"Unchanged: {(temp[change_col] == 0).sum()}")

print("\n===================================")
print("BASELINE PRE-OP SUMMARY")
print("===================================")
print(f"Patients with pre-op A1c: {longitudinal['preop_a1c'].notna().sum()}")
print(f"Mean pre-op A1c: {longitudinal['preop_a1c'].dropna().mean():.2f}")
print(f"Min pre-op A1c: {longitudinal['preop_a1c'].dropna().min():.2f}")
print(f"Max pre-op A1c: {longitudinal['preop_a1c'].dropna().max():.2f}")

print("\n===================================")
print("OVERALL LONGITUDINAL RESULTS")
print("===================================")
summarize_timepoint(longitudinal, "m3", "3 MONTH")
summarize_timepoint(longitudinal, "m6", "6 MONTH")
summarize_timepoint(longitudinal, "m12", "12 MONTH")

print("\n===================================")
print("PYLOROPLASTY RESULTS (43800)")
print("===================================")
pylo = longitudinal[longitudinal["procedure_code"] == "43800"]
print(f"Total Pyloroplasty patients: {len(pylo)}")
print(f"Patients with pre-op A1c: {pylo['preop_a1c'].notna().sum()}")
if len(pylo) > 0:
    print(f"Mean pre-op A1c: {pylo['preop_a1c'].dropna().mean():.2f}")
summarize_timepoint(pylo, "m3", "3 MONTH")
summarize_timepoint(pylo, "m6", "6 MONTH")
summarize_timepoint(pylo, "m12", "12 MONTH")

print("\n===================================")
print("G-POEM RESULTS (43659 + 43999)")
print("===================================")
gpoem = longitudinal[longitudinal["procedure_code"].isin(["43659", "43999"])]
print(f"Total G-POEM patients: {len(gpoem)}")
print(f"Patients with pre-op A1c: {gpoem['preop_a1c'].notna().sum()}")
if len(gpoem) > 0:
    print(f"Mean pre-op A1c: {gpoem['preop_a1c'].dropna().mean():.2f}")
summarize_timepoint(gpoem, "m3", "3 MONTH")
summarize_timepoint(gpoem, "m6", "6 MONTH")
summarize_timepoint(gpoem, "m12", "12 MONTH")

longitudinal.to_csv("longitudinal_a1c_results.csv", index=False)
print("\nSaved: longitudinal_a1c_results.csv")
print("Done with Step 8!")
