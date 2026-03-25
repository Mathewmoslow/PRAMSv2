# PCHE Latent Variable Analysis Summary
Run ID: run_20260325T_domains_lv
Date: 2026-03-25 20:54 UTC

## B1: Factor Analysis on 7 Common PCHE Components

Components: QUO41, QUO65, QUO257, QUO4, QUO5, QUO44, QUO101 (7 items)
Complete cases: 180

### Eigenvalues and Parallel Analysis
  PC1: eigenvalue=4.4018, PA threshold=1.3983, var explained=62.9% *RETAIN*
  PC2: eigenvalue=1.3835, PA threshold=1.2388, var explained=19.8% *RETAIN*
  PC3: eigenvalue=0.9595, PA threshold=1.1327, var explained=13.7%
  PC4: eigenvalue=0.1981, PA threshold=1.0463, var explained=2.8%
  PC5: eigenvalue=0.0272, PA threshold=0.9692, var explained=0.4%
  PC6: eigenvalue=0.0256, PA threshold=0.8946, var explained=0.4%
  PC7: eigenvalue=0.0043, PA threshold=0.8076, var explained=0.1%
Factors retained: 2

### Rotated Loadings (VARIMAX)
QID | Factor1 | Factor2
-----------------------
QUO41 (Took multivitamins >4x/week month before pregnancy) | -0.3656 | 0.8919
QUO65 (Took daily multivitamin in month before pregnancy) | -0.2082 | 0.9471
QUO257 (Was trying to become pregnant) | -0.5211 | 0.7067
QUO4 (Still breastfeeding at 4 weeks postpartum) | -0.9472 | 0.2551
QUO5 (Ever breastfed or pumped breast milk) | -0.9330 | 0.2271
QUO44 (Still breastfeeding at 8 weeks postpartum) | -0.9411 | 0.2883
QUO101 (Baby had checkup/exam within first week) | 0.3311 | 0.4266

### Factor Interpretation
Factor1: QUO4 (-0.947), QUO44 (-0.941), QUO5 (-0.933), QUO257 (-0.521), QUO41 (-0.366), QUO101 (0.331)
Factor2: QUO65 (0.947), QUO41 (0.892), QUO257 (0.707), QUO101 (0.427)

### Factor Score vs Simple Composite Comparison
  QUO197: Composite adj-R2=0.7553, Factor adj-R2=0.7603
  QUO210: Composite adj-R2=0.7574, Factor adj-R2=0.7623

## B2: Factor Analysis on Full 12 PCHE Components

Components: QUO41, QUO65, QUO257, QUO4, QUO5, QUO44, QUO101, QUO179, QUO249, QUO75, QUO296, QUO297 (12 items)
Complete cases: 0
  SKIP: Only 0 complete cases (< 30). Not enough data for 12-component factor analysis.

## B3: Does the Factor Structure Support a Single Composite?

Eigenvalue ratio (PC1/PC2): 3.182
  (> 3 suggests dominant single factor)
Variance explained by PC1 alone: 62.9%
Factors retained by parallel analysis: 2

Cross-loading items (|loading| > 0.3 on >1 factor): QUO41, QUO257, QUO101

### Conclusion
The factor structure DOES support treating PCHE as a unitary composite because the first eigenvalue (4.402) is 3.2x the second (1.384), explaining 62.9% of variance. This strong first-factor dominance indicates that the 7 indicators share substantial common variance captured by a single latent dimension. The VARIMAX rotation produced 2 factors with cross-loadings, suggesting the factors are not cleanly separable and a composite may still be reasonable.
