# Data source and citation

## Source

`data/raw/F-F_Research_Data_5_Factors_2x3.csv` is the monthly
**Fama/French 5 Factors (2x3)** file from the **Kenneth R. French Data Library**:

- Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- Factor definitions: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html
- Vintage used here: file created from the **202511 CRSP database** (header line of the file);
  monthly observations **July 1963 – November 2025**.
- The data is redistributed in this repository unchanged except for the removal of an
  accidental duplicated block (see below), solely so the results can be reproduced.
  Copyright remains with Eugene F. Fama and Kenneth R. French (as stated in the file's
  footer); it is **not** covered by this repository's MIT licence.

## Variables

| Column | Meaning |
|---|---|
| `Mkt-RF` | Market excess return: value-weighted US market return minus the one-month T-bill rate |
| `SMB` | Size: small minus big |
| `HML` | Value: high minus low book-to-market |
| `RMW` | Profitability: robust minus weak operating profitability |
| `CMA` | Investment: conservative minus aggressive investment |
| `RF` | One-month Treasury bill rate (not used as a feature) |

Values are monthly percentage returns; the loader converts them to decimals.

## Processing in this project

- Only the monthly table (`YYYYMM` rows) is used; the annual table and notes are ignored.
- Features: the five factors at month *t*. Target: `Mkt-RF` at month *t + 1*
  (or its absolute value for the risk-forecasting experiment).
- The raw file is validated before use (`lrd/core/data/validation.py`).
- **Correction:** the copy of the file originally committed to this repository began with a
  duplicated copy of its header and the first 68 months (Jul 1963 – Feb 1969). That block
  was removed; the original corrupted state is preserved in the git tag `v1-standard`.

## How to cite

Data:

> Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model.
> *Journal of Financial Economics*, 116(1), 1–22. https://doi.org/10.1016/j.jfineco.2014.10.010

> French, K. R. (2025). *Fama/French 5 Factors (2x3)* [Data set]. Kenneth R. French Data Library,
> Tuck School of Business at Dartmouth. Retrieved 2025, CRSP vintage 202511.

Methods used in the evaluation:

> Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample:
> Can anything beat the historical average? *Review of Financial Studies*, 21(4), 1509–1531.

> Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy
> in nested models. *Journal of Econometrics*, 138(1), 291–311.

> Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
> *Journal of Business & Economic Statistics*, 13(3), 253–263.

> Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity
> premium prediction. *Review of Financial Studies*, 21(4), 1455–1508.
