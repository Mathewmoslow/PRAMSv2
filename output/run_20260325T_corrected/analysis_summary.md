# PCHE Analysis Summary (v2 — IRB-corrected)

**Run ID**: run_20260325T_corrected
**Timestamp**: 2026-03-25T19:46:22.716716+00:00
**Random Seed**: 42
**Panel SHA-256**: 03fd9f42e1626b63...

## Terminology
- Outcome: PDS (Postpartum Depressive Symptoms)
- Construct: PCHE (Pre-Conception Health Engagement)
- Never use: PPD prevalence, PCHA, Pre-Conception Health Agency

## Data
- Panel: 317 state-year rows x 233 columns
- PDS panel: 180 rows with outcome
- Locations: 35, Years: 2004-2011
- Era breakdown: {'2004-2008': 97, '2009-2011': 83}

## PCHE Index (PRIMARY: 7 common components)
- Components: QUO41, QUO65, QUO257, QUO4, QUO5, QUO44, QUO101
- Valid rows: 180
- Cronbach's alpha: 0.870
- Mean: 0.000, SD: 0.747

## FE-Only Baselines
- State-only FE R2: 0.5978
- State+Year FE R2: 0.8015
- Incremental year R2: 0.2037

## Parallel Analysis
- Retained components: 2

## Interaction Results

### QUO197: Partner argued more than usual (12 mo before birth) (Primary)
- n = 180, clusters = 35
- **OLS interaction beta: -0.3614**, parametric p=0.0038
- CR1 p=0.0697
- CR1-FE p=0.0365
- Wild cluster bootstrap p=0.1906
- Cluster permutation p=0.1770
- Cluster bootstrap 95% CI: [-0.7064, 0.1039]
- Benefit at +1 SD risk (PCHE -1 to +1): 2.77 pp PDS reduction
- Attenuation signature: True

### QUO210: Any partner-related stressors (Primary)
- n = 180, clusters = 35
- **OLS interaction beta: -0.3512**, parametric p=0.0038
- CR1 p=0.0585
- CR1-FE p=0.0327
- Wild cluster bootstrap p=0.1814
- Cluster permutation p=0.1868
- Cluster bootstrap 95% CI: [-0.6935, 0.0808]
- Benefit at +1 SD risk (PCHE -1 to +1): 2.31 pp PDS reduction
- Attenuation signature: True

### QUO313: Physical IPV by ex-partner (before pregnancy) (Secondary)
- n = 103, clusters = 27
- **OLS interaction beta: -0.0608**, parametric p=0.6267
- CR1 p=0.7271
- CR1-FE p=0.8032
- Wild cluster bootstrap p=0.7094
- Cluster permutation p=0.8122
- Cluster bootstrap 95% CI: [-0.8242, 0.3939]
- Benefit at +1 SD risk (PCHE -1 to +1): 2.61 pp PDS reduction
- Attenuation signature: True

### QUO315: Physical IPV by ex-partner (during pregnancy) (Secondary)
- n = 103, clusters = 27
- **OLS interaction beta: -0.0692**, parametric p=0.5839
- CR1 p=0.6538
- CR1-FE p=0.3989
- Wild cluster bootstrap p=0.6716
- Cluster permutation p=0.7626
- Cluster bootstrap 95% CI: [-0.5253, 0.5533]
- Benefit at +1 SD risk (PCHE -1 to +1): 2.53 pp PDS reduction
- Attenuation signature: True