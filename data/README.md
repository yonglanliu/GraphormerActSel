## Custom Dataset Format

You can provide your own dataset to train and evaluate the model for your specific use case.

The dataset must contain:
*	SMILES string of each compound
*	pIC50 values against multiple subtypes (one column per subtype)

Required Format:

| SMILES | pIC50_subtypeA | pIC50_subtypeB | pIC50_subtypeC |
|---|---|---|---|
| CCOc1ccc2nc(S(N)(=O)=O)sc2c1 | 7.85 | 6.32 | 5.91 |
| CCN(CC)CCOc1ccc2nc(S(N)(=O)=O)sc2c1 | NaN | 7.45 |6.10 |
| Cc1ccc2nc(S(N)(=O)=O)sc2c1 | 6.95 | 5.88 | NaN |


*	Missing values can be left blank or set as NaN
*	Column names should clearly indicate subtype identity
*	Units must be pIC50 (not IC50)

**Notes**

*	Ensure SMILES strings are valid and sanitized (RDKit-compatible).
*	For multi-task training, each subtype column is treated as an independent regression task.
*	Selectivity losses (if enabled) are computed based on pairwise differences between subtype activities.
