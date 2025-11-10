# Data Pre-Processing: Cleaning

The goal of the cleaning step is to turn raw platform response data into the formats required for IRT modelling.

The process produces two key outputs:

1. **Long-format data** for initial inspection and validation  
2. **Wide-format data (`mirt_in.csv`)** for use in R's `mirt` package

---

## Long-Format Output

Given a response data with item-level responses, raw assessment responses are first transformed into a long-format table:

| participant_id | item_id | response |
|----------------|---------|----------|
| 1028           | 1       | 1        |
| 1028           | 2       | 0        |
| ...            | ...     | ...      |

---

## Assumptions Used During Cleaning

1. **One attempt per participant**  
   - If a participant has multiple test attempts, the *earliest completed attempt* (based on timestamp) is selected.

2. **Single assessment version**  
   - Only rows from a chosen `AssessmentVersionID` are included to avoid mixing items across versions.

3. **Progressive difficulty**  
   - Items are ordered by level.  
   - If a participant did not reach a given item, it is treated as **0** (incorrect) because the task was never attempted.

If any of these assumptions change (e.g., non-sequential items, optional items, multiple branching paths), the cleaning logic must be updated accordingly.

---

## Wide-Format Output (for IRT)

After cleaning, the data is pivoted into the wide structure used by **mirt**:

| participant_id | item 1 | item 2 | ... |
|----------------|--------|--------|-----|
| 1028           | 1      | 0      | ... |
| 1097           | 1      | 1      | ... |
| ...            | ...    | ...    | ... |

This file is saved as: data/<assessment_name>/mirt_in.csv

