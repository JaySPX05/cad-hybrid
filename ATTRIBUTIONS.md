# Dataset attributions

The datasets below are used under the Creative Commons Attribution 4.0 International licence (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

CC BY 4.0 requires us to (1) credit the creators, (2) link to the licence and (3) say what we changed. It must not suggest that the dataset creators endorse this project.

Fill in the author names exactly as shown on each dataset page, and keep the dated licence screenshot referenced in `data/data_register.csv`.

| Dataset | Creators | Source | Licence | Changes we made |
|---|---|---|---|---|
| Cardiovascular_Disease_Dataset (India) | _TODO: authors from dataset page_ | Mendeley Data, https://doi.org/10.17632/dzz48mvjht.1 | CC BY 4.0 | Columns renamed; cholesterol 0 and undocumented slope 0 set to missing; chest pain type mapped to yes/no. Used for pipeline testing only. |
| Clinical, anthropometric and biochemical parameters in cardiac patients from Northern Bangladesh (Zia Heart Foundation Hospital) | _TODO: authors from dataset page_ | Mendeley Data, https://data.mendeley.com/datasets/vf6yrn2jjt/1 | CC BY 4.0 | Columns renamed; typos parsed; BMI recomputed; out-of-range values set to missing; under-18s removed; UNIT, Troponin-I and MaxHR excluded. |
| ECG Images dataset of Cardiac Patients (Pakistan) | _TODO_ | Mendeley Data, https://data.mendeley.com/datasets/gwbz3fsgp8/2 | CC BY 4.0 | _TODO: e.g. cropped, resized, labels merged into CAD-positive/negative_ |
| ECG Image Dataset for Myocardial Infarction Complication (India) | _TODO_ | Mendeley Data, https://data.mendeley.com/datasets/5dhyd9t26m/2 | CC BY 4.0 | _TODO_ |

Datasets obtained under data-use agreements (e.g. CARRS via BioLINCC, South Asia Biobank) are governed by those agreements, not CC BY. Record their terms in `data/data_register.csv`.
