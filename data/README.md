## Custom Dataset Format

You can provide your own dataset to train and evaluate the model for your specific use case.

The dataset must contain:
*	SMILES string of each compound
*	pIC50 values against multiple isoforms (one column per isoform)

Required Format:

| smiles | pIC50_isoformA | pIC50_isoformB | pIC50_isoformC |
|---|---|---|---|
| CCOc1ccc2nc(S(N)(=O)=O)sc2c1 | 7.85 | 6.32 | 5.91 |
| CCN(CC)CCOc1ccc2nc(S(N)(=O)=O)sc2c1 | NaN | 7.45 |6.10 |
| Cc1ccc2nc(S(N)(=O)=O)sc2c1 | 6.95 | 5.88 | NaN |


*	Missing values can be left blank or set as NaN
*	Column names should clearly indicate isoform identity
*	Units must be pIC50 (not IC50)

**Notes**

*	Ensure SMILES strings are valid and sanitized (RDKit-compatible).
*	For multi-task training, each isoform column is treated as an independent regression task.
*	Selectivity losses (if enabled) are computed based on pairwise differences between isoform activities.
