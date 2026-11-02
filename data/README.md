# Data

**No data files are committed to this repository.** Everything under `raw/`, `interim/` and `processed/` is gitignored; only the empty folders (`.gitkeep`), this README and `data_register.csv` are tracked.

## Rule
Only Indian, Pakistani or Bangladeshi datasets are used. Every dataset must have an entry in `data_register.csv` (licence or agreement, date obtained, evidence) **before** it is used for anything that goes into the product.

## Folders
| Folder | Contents |
|---|---|
| `raw/` | Original downloads, untouched. One subfolder per dataset. |
| `interim/` | Cleaned and de-duplicated data. |
| `processed/` | Derived files, e.g. `oof_tabular.csv` (held-out predictions written by `scripts/train_tabular.py`). Patient-level: never commit. |

## Datasets and where they go
| Folder | Dataset | Get it from | Licence / access |
|---|---|---|---|
| `raw/mendeley_india_cvd/` | Cardiovascular_Disease_Dataset (India, 1,000 patients) | https://doi.org/10.17632/dzz48mvjht.1 | CC BY 4.0 |
| `raw/bangladesh_cardiac/` | Dataset of clinical, anthropometric, and biochemical parameters in cardiac patients from Northern Bangladesh (1,048 patients) | https://doi.org/10.17632/vf6yrn2jjt.1 | CC BY 4.0 |
| `raw/ecg_multan_pk/` | ECG Images dataset of Cardiac Patients (Pakistan; 928 images but only 491 unique ECGs, many images are copies; see the data register) | https://doi.org/10.17632/gwbz3fsgp8.2 | CC BY 4.0 |
| `raw/ecg_bhopal_in/` | ECG Image dataset for Myocardial Infarction Complication (India) | https://doi.org/10.17632/5dhyd9t26m.2 | CC BY 4.0 |

Requested / pending (add a folder when access is granted):
- **CARRS** via NIH BioLINCC ("GHCoE New Delhi"): data-use agreement required.
- **South Asia Biobank** via the GHRU Data Access Committee: eligibility for commercial use to be confirmed.
- **ICADD** (Indian CAD dataset): via the authors.

## Steps when adding a dataset
1. Download into its `raw/<dataset>/` folder. Do not edit files in `raw/`.
2. Screenshot the licence or agreement page with the date visible; store it in the shared drive (not in git).
3. Add or update its row in `data_register.csv`.
4. Add the CC BY credit to `ATTRIBUTIONS.md` if applicable.

## Handling
- Data is de-identified; never attempt to re-identify anyone.
- Don't upload data to third-party tools or share it outside the team.
- Delete local copies when the project ends or if an agreement requires it.
