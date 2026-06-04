# Data Extraction Notes

## A1c Extraction — Step 3 Update

### Change
Date parsing method updated from strict format to flexible parsing.

### Old code (step3_find_a1c.py)
```python
a1c["date"] = pd.to_datetime(
    a1c["date"],
    format="%Y%m%d",
    errors="coerce"
)
```

### New code (step3_find_a1c_updated.py)
```python
chunk["date"] = pd.to_datetime(
    chunk["date"],
    errors="coerce"
)
```

### Impact
- Old extraction: 491 patients with A1c data
- New extraction: 793 patients with A1c data
- Difference: +302 patients

### Reason
Old code only accepted dates in YYYYMMDD format (e.g. 20220614).
Any date formatted differently (e.g. 2022-06-14) was silently dropped.
New code uses pandas auto-detection which handles multiple date formats correctly.

### Conclusion
New extraction is more complete and accurate. The additional 302 patients
had valid A1c records that were previously excluded due to date format


----- 
Update 6/4/2026
"Investigation confirmed that original a1c_results.csv (491 patients) was produced by step8_pull_all_data.py for an earlier cohort, NOT by step3_find_a1c.py. The old step3 script used format='%Y%m%d' which drops ALL records since TriNetX lab dates are in YYYY-MM-DD format. New extraction (793 patients) is the correct and complete extraction for the final n=1,041 cohort."

### Date of change
Step 3 rerun with updated extraction script after cohort cleaning.
