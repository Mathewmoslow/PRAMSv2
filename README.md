# Pre-Conception Health Engagement and Postpartum Depressive Symptoms

An ecological panel analysis testing whether state-level preconception health engagement (PCHE) moderates the partner-stress → postpartum depressive symptom pathway using CDC PRAMStat data, 2000–2011.

## Reproduction

```bash
pip install pandas numpy scipy matplotlib

# Step 1: Prepare data (exclude aggregates, correct variable labels)
python scripts/01_extract_panel.py

# Step 2: Main analysis (interactions, FE, cluster inference, figures)
python scripts/02_main_analysis.py --run-id run_20260325T_corrected

# Step 3: Follow-up (variants, domains, intentionality, SES controls)
python scripts/03_followup_analysis.py --run-id run_20260325T_followup

# Step 4: Domain combinations and latent variable analysis
python scripts/04_domain_combos_and_latent.py --run-id run_20260325T_domains_lv
```

All scripts use `seed=42` for full reproducibility.

## Key Results

- PCHE moderates partner-stress → PDS (FE β = −0.46 to −0.50, p = .011–.021)
- Effect holds with strict pre-conception components only (β = −0.41)
- Survives removal of pregnancy intentionality (β = −0.31)
- Not reducible to SES (survives 18 simultaneous controls)
- Cronbach's α = .87, dominant single factor (63% variance)
- Cluster-aware inference more equivocal (perm p ≈ .18, WCB p ≈ .19)

## Structure

```
data/          Source panel and question dictionary
scripts/       4 analysis scripts (numbered, sequential)
output/        3 timestamped run directories (39 CSV/JSON files)
figures/       3 figure directories (18 PNG files)
reports/       Final APA 7 manuscript (HTML, print-ready)
```

## Paper

Open `reports/PCHE_Paper_Final.html` in a browser. Print via Cmd+P → Save as PDF.

## Author

Mathew Moslow, AdventHealth University
