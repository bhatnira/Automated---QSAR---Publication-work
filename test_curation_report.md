# QSAR Data Curation Report

**Generated:** 2026-02-10 11:16:08

## Summary

| Metric | Value |
|--------|-------|
| Initial samples | 10 |
| Invalid SMILES | 1 |
| Duplicates removed | 1 |
| Invalid activities | 1 |
| Final samples | 7 |
| Task type | MULTICLASS_CLASSIFICATION |
| Transform applied | none |

## Curation Log

- Starting QSAR data curation pipeline
- Initial rows: 10
- SMILES column: SMILES
- Activity column: IC50
- Unit: nM
- SMILES validation: 9 valid, 1 invalid
- Duplicate molecules removed: 1
- Censored values: 1 left (<), 1 right (>)
- Invalid activity values removed: 1
- Inferred task type: MULTICLASS_CLASSIFICATION
- Classification task - no log transform applied
- Target statistics:
-   N: 7
-   Range: 1.000 - 10000.000
-   Mean ± Std: 1675.214 ± 3414.805
- ✓ Final dataset size: 7
- ✓ Pipeline completed successfully

## Target Statistics

| Statistic | Value |
|-----------|-------|
| N | 7 |
| Mean | 1675.2143 |
| Std | 3414.8050 |
| Min | 1.0000 |
| Max | 10000.0000 |
| Median | 100.0000 |
