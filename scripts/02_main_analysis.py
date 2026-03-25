#!/usr/bin/env python3
"""
Pre-Conception Health Engagement (PCHE) Analysis — v2
=====================================================
Ecological panel analysis testing whether a latent "Pre-Conception Health
Engagement" (PCHE) construct moderates the partner-stress -> postpartum
depressive symptoms (PDS) pathway at the US state-year level using
PRAMStat 2004-2011 data.

All corrections from IRB review are implemented:
  1.  PDS/PCHE terminology throughout (never PPD prevalence / PCHA / Agency)
  2.  Aggregate "PRAMS Total" rows excluded (verified)
  3.  PRIMARY PCHE index = 7 common-component set; full 12 = sensitivity
  4.  State + year fixed effects in ALL FE models
  5.  Cluster-aware bootstrap (resample entire state blocks)
  6.  Cluster-aware permutation (permute PCHE across state blocks)
  7.  Wild cluster bootstrap for inference (Rademacher weights)
  8.  CR1 reported for ALL models (no selective reporting)
  9.  Benefit estimates at +1 SD partner stress (not at mean)
  10. Correct figure labels (PCHE, Pre-Conception Health Engagement, PDS)
  11. FE-only baseline reports state-only AND state+year R-squared
  12. Panel accounting table with cluster counts

Usage:
    python scripts/02_main_analysis.py [--panel PATH] [--run-id ID]

Outputs: output/<run_id>/  and  figures/<run_id>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_PERMUTATIONS = 5000
N_BOOTSTRAP = 5000
MIN_N = 30
MIN_SD = 0.25

# PDS outcome question IDs (harmonized across eras)
PDS_QUO_IDS = {"QUO74", "QUO219"}
BLUES_PROXY_QUO = "QUO97"
LEAKAGE_EXCLUDE = PDS_QUO_IDS | {BLUES_PROXY_QUO}

# Partner-stress risk anchors
RISK_ANCHORS = {
    "QUO197": "Partner argued more than usual (12 mo before birth)",
    "QUO210": "Any partner-related stressors",
    "QUO313": "Physical IPV by ex-partner (before pregnancy)",
    "QUO315": "Physical IPV by ex-partner (during pregnancy)",
}
PRIMARY_RISK = ["QUO197", "QUO210"]
SECONDARY_RISK = ["QUO313", "QUO315"]

# PCHE theory-driven indicator sets — COMMON-COMPONENT (7 items, all 180 rows)
PCHE_COMMON = {
    "QUO41":  "Took multivitamins >4x/week month before pregnancy",
    "QUO65":  "Took daily multivitamin in month before pregnancy",
    "QUO257": "Was trying to become pregnant",
    "QUO4":   "Still breastfeeding at 4 weeks postpartum",
    "QUO5":   "Ever breastfed or pumped breast milk",
    "QUO44":  "Still breastfeeding at 8 weeks postpartum",
    "QUO101": "Baby had checkup/exam within first week",
}
PCHE_COMMON_ORDER = ["QUO41", "QUO65", "QUO257", "QUO4", "QUO5", "QUO44", "QUO101"]

# Full 12-component set (sensitivity only — era-specific items not in all rows)
PCHE_FULL = {
    **PCHE_COMMON,
    "QUO179": "Exercised 3+ days/week in 3 months before pregnancy",
    "QUO249": "Had teeth cleaned by dentist before pregnancy",
    "QUO75":  "Had teeth cleaned during pregnancy",
    "QUO16":  "Pregnancy was intended",
    "QUO296": "Got prenatal care as early as wanted (2000-2008)",
}

# Structural access indicators (discriminant validity)
STRUCTURAL_ACCESS = {
    "QUO53":  "Medicaid recipient at any time",
    "QUO267": "Medicaid paid for delivery",
    "QUO317": "Private insurance paid for delivery",
    "QUO322": "Private insurance paid for prenatal care",
    "QUO310": "Insurance before pregnancy (excl. Medicaid)",
    "QUO25":  "Medicaid covered prenatal care",
    "QUO227": "Insurance coverage month before pregnancy",
    "QUO318": "Personal income paid for delivery",
}

# Counseling indicators (paradox analysis)
COUNSELING_INDICATORS = {
    "QUO38":  "PNC discussed alcohol effects",
    "QUO215": "PNC discussed illegal drug effects",
    "QUO40":  "PNC discussed smoking effects",
    "QUO66":  "PNC discussed early labor signs",
    "QUO67":  "PNC discussed birth defect screening",
    "QUO37":  "PNC discussed breastfeeding",
    "QUO36":  "PNC discussed seatbelt use",
    "QUO39":  "PNC discussed HIV testing",
    "QUO35":  "PNC discussed partner violence",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_pche")


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardize to z-scores. Returns NaN array if SD ~ 0."""
    s = float(np.nanstd(x, ddof=1))
    if not math.isfinite(s) or s < 1e-12:
        return np.full_like(x, np.nan, dtype=float)
    return (x - float(np.nanmean(x))) / s


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------
@dataclass
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    pvals: np.ndarray
    n: int
    k: int
    r2: float
    adj_r2: float
    rmse: float
    aic: float
    bic: float
    rss: float
    tss: float
    col_names: list


def fit_ols(y: np.ndarray, X_cols: list[np.ndarray],
            col_names: list[str] = None,
            weights: np.ndarray = None) -> Optional[OLSResult]:
    """Fit OLS (or WLS) regression y ~ intercept + X_cols."""
    X = np.column_stack([np.ones(len(y)), *X_cols])
    n, k = X.shape
    if n <= k or n < 8:
        return None
    if col_names is None:
        col_names = ["intercept"] + [f"x{i}" for i in range(len(X_cols))]
    else:
        col_names = ["intercept"] + list(col_names)

    if weights is not None:
        W = np.diag(np.sqrt(weights))
        Xw, yw = W @ X, W @ y
    else:
        Xw, yw = X, y

    try:
        beta, residuals, rank, sv = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < k:
        return None

    resid = y - X @ beta
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if tss <= 0:
        return None
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * ((n - 1) / max(n - k, 1))
    rmse = math.sqrt(rss / n)
    aic = n * math.log(max(rss / n, 1e-12)) + 2 * k
    bic = n * math.log(max(rss / n, 1e-12)) + k * math.log(n)

    s2 = rss / max(n - k, 1)
    try:
        XtX_inv = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        return None
    cov = s2 * XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    tvals = np.where(se > 0, beta / se, 0)
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=max(n - k, 1)))

    return OLSResult(
        beta=beta, se=se, pvals=pvals, n=n, k=k,
        r2=r2, adj_r2=adj_r2, rmse=rmse, aic=aic, bic=bic,
        rss=rss, tss=tss, col_names=col_names,
    )


def fit_ols_cluster_robust(y: np.ndarray, X_cols: list[np.ndarray],
                           cluster_ids: np.ndarray,
                           col_names: list[str] = None) -> Optional[OLSResult]:
    """OLS with CR1 cluster-robust standard errors."""
    X = np.column_stack([np.ones(len(y)), *X_cols])
    n, k = X.shape
    if n <= k or n < 8:
        return None
    if col_names is None:
        col_names = ["intercept"] + [f"x{i}" for i in range(len(X_cols))]
    else:
        col_names = ["intercept"] + list(col_names)

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    resid = y - X @ beta
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if tss <= 0:
        return None
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * ((n - 1) / max(n - k, 1))
    rmse = math.sqrt(rss / n)
    aic = n * math.log(max(rss / n, 1e-12)) + 2 * k
    bic = n * math.log(max(rss / n, 1e-12)) + k * math.log(n)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None

    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = cluster_ids == c
        Xc = X[mask]
        ec = resid[mask]
        score_c = Xc.T @ ec
        meat += np.outer(score_c, score_c)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V_cr = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V_cr), 0))
    tvals = np.where(se > 0, beta / se, 0)
    df_cr = max(G - 1, 1)
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=df_cr))

    return OLSResult(
        beta=beta, se=se, pvals=pvals, n=n, k=k,
        r2=r2, adj_r2=adj_r2, rmse=rmse, aic=aic, bic=bic,
        rss=rss, tss=tss, col_names=col_names,
    )


# ---------------------------------------------------------------------------
# Fixed-effects helpers
# ---------------------------------------------------------------------------
def make_state_fe_dummies(panel: pd.DataFrame) -> tuple[list[np.ndarray], list[str]]:
    """State-only fixed effects (drop-one reference encoding)."""
    states = panel["location_abbr"].values
    u_states = sorted(set(states))
    dummies, names = [], []
    for s in u_states[1:]:
        dummies.append((states == s).astype(float))
        names.append(f"FE_state_{s}")
    return dummies, names


def make_fe_dummies(panel: pd.DataFrame) -> tuple[list[np.ndarray], list[str]]:
    """State + year fixed effects (drop-one reference encoding)."""
    states = panel["location_abbr"].values
    years = panel["year"].values.astype(int)
    u_states = sorted(set(states))
    u_years = sorted(set(years))
    dummies, names = [], []
    for s in u_states[1:]:
        dummies.append((states == s).astype(float))
        names.append(f"FE_state_{s}")
    for yr in u_years[1:]:
        dummies.append((years == yr).astype(float))
        names.append(f"FE_year_{yr}")
    return dummies, names


# ---------------------------------------------------------------------------
# Parallel analysis (Horn 1965)
# ---------------------------------------------------------------------------
def parallel_analysis(data_matrix: np.ndarray, n_iter: int = 1000,
                      seed: int = RANDOM_SEED) -> np.ndarray:
    """Horn's parallel analysis: 95th-percentile eigenvalue thresholds."""
    rng = np.random.RandomState(seed)
    n, p = data_matrix.shape
    all_eigs = np.zeros((n_iter, p))
    for i in range(n_iter):
        rand_data = rng.standard_normal((n, p))
        corr = np.corrcoef(rand_data, rowvar=False)
        eigs = np.linalg.eigvalsh(corr)[::-1]
        all_eigs[i] = eigs
    return np.percentile(all_eigs, 95, axis=0)


# ---------------------------------------------------------------------------
# Wild cluster bootstrap (Rademacher weights)
# ---------------------------------------------------------------------------
def wild_cluster_bootstrap(y, X_full, X_restricted, cluster_ids,
                           test_col_idx, n_boot=5000, seed=42):
    """
    Wild cluster bootstrap for a single coefficient test.

    Parameters
    ----------
    y : array (n,)
    X_full : array (n, k_full) — includes intercept
    X_restricted : array (n, k_restr) — model without the tested variable
    cluster_ids : array (n,)
    test_col_idx : int — column index in X_full for the coefficient being tested
    n_boot : int
    seed : int

    Returns
    -------
    p_value : float — proportion of bootstrap |t*| >= observed |t|
    """
    rng = np.random.RandomState(seed)
    n = len(y)

    # Fit restricted model
    try:
        beta_r, *_ = np.linalg.lstsq(X_restricted, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    resid_r = y - X_restricted @ beta_r
    y_fitted_r = X_restricted @ beta_r

    # Fit full model to get observed t-stat
    try:
        beta_f, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    resid_f = y - X_full @ beta_f

    # CR1 variance for observed t-stat
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    k_f = X_full.shape[1]
    try:
        XtX_inv_f = np.linalg.inv(X_full.T @ X_full)
    except np.linalg.LinAlgError:
        return np.nan

    meat_obs = np.zeros((k_f, k_f))
    for c in unique_clusters:
        m = cluster_ids == c
        sc = X_full[m].T @ resid_f[m]
        meat_obs += np.outer(sc, sc)
    correction = (G / (G - 1)) * ((n - 1) / (n - k_f))
    V_obs = correction * XtX_inv_f @ meat_obs @ XtX_inv_f
    se_obs = math.sqrt(max(V_obs[test_col_idx, test_col_idx], 0))
    if se_obs < 1e-15:
        return np.nan
    t_obs = abs(beta_f[test_col_idx] / se_obs)

    # Bootstrap iterations
    # Map each observation to its cluster index
    cluster_map = {c: i for i, c in enumerate(unique_clusters)}
    obs_cluster_idx = np.array([cluster_map[c] for c in cluster_ids])

    count_exceed = 0
    for b in range(n_boot):
        # Rademacher weights: +1 or -1 per cluster
        w_g = rng.choice([-1.0, 1.0], size=G)
        w_i = w_g[obs_cluster_idx]

        # Bootstrap residuals and y*
        e_star = w_i * resid_r
        y_star = y_fitted_r + e_star

        # Fit full model on y*
        try:
            beta_star, *_ = np.linalg.lstsq(X_full, y_star, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid_star = y_star - X_full @ beta_star

        # CR1 for bootstrap
        meat_star = np.zeros((k_f, k_f))
        for c in unique_clusters:
            m = cluster_ids == c
            sc = X_full[m].T @ resid_star[m]
            meat_star += np.outer(sc, sc)
        V_star = correction * XtX_inv_f @ meat_star @ XtX_inv_f
        se_star = math.sqrt(max(V_star[test_col_idx, test_col_idx], 0))
        if se_star < 1e-15:
            continue
        t_star = abs(beta_star[test_col_idx] / se_star)
        if t_star >= t_obs:
            count_exceed += 1

    return count_exceed / n_boot


# ---------------------------------------------------------------------------
# Cluster-aware permutation
# ---------------------------------------------------------------------------
def cluster_permutation_test(y, risk_z, pche_z, cluster_ids,
                             n_perm=5000, seed=42):
    """
    Permute PCHE trajectories across state blocks (whole-state shuffle).
    Returns permutation p-value for the interaction coefficient.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    interaction = risk_z * pche_z

    # Observed interaction beta
    m_obs = fit_ols(y, [risk_z, pche_z, interaction],
                    col_names=["risk_z", "pche_z", "interaction"])
    if m_obs is None:
        return None, None
    obs_beta = float(m_obs.beta[m_obs.col_names.index("interaction")])

    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    # Build cluster->index mapping
    cluster_indices = {}
    for c in unique_clusters:
        cluster_indices[c] = np.where(cluster_ids == c)[0]

    # For each cluster, store the pche_z values
    cluster_pche = {}
    for c in unique_clusters:
        cluster_pche[c] = pche_z[cluster_indices[c]].copy()

    perm_betas = np.zeros(n_perm)
    cluster_list = list(unique_clusters)
    for i in range(n_perm):
        # Shuffle which cluster gets which PCHE trajectory
        perm_order = rng.permutation(G)
        pche_perm = np.empty(n, dtype=float)
        for j, c in enumerate(cluster_list):
            source_cluster = cluster_list[perm_order[j]]
            target_idx = cluster_indices[c]
            source_vals = cluster_pche[source_cluster]
            # If cluster sizes differ, sample with replacement
            if len(source_vals) == len(target_idx):
                pche_perm[target_idx] = source_vals
            else:
                pche_perm[target_idx] = rng.choice(source_vals, size=len(target_idx), replace=True)

        int_perm = risk_z * pche_perm
        m_perm = fit_ols(y, [risk_z, pche_perm, int_perm],
                         col_names=["risk_z", "pche_z", "interaction"])
        if m_perm:
            perm_betas[i] = float(m_perm.beta[m_perm.col_names.index("interaction")])
        else:
            perm_betas[i] = 0.0

    perm_p = float(np.mean(np.abs(perm_betas) >= np.abs(obs_beta)))
    return perm_p, perm_betas


# ---------------------------------------------------------------------------
# Cluster-aware bootstrap CI
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(y, risk_z, pche_z, cluster_ids,
                         n_boot=5000, seed=42):
    """
    Resample entire state blocks (all years for a state together).
    Returns bootstrap betas for the interaction coefficient.
    """
    rng = np.random.RandomState(seed)
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    cluster_indices = {}
    for c in unique_clusters:
        cluster_indices[c] = np.where(cluster_ids == c)[0]
    cluster_list = list(unique_clusters)

    boot_betas = np.zeros(n_boot)
    for i in range(n_boot):
        # Resample G clusters with replacement
        sampled = rng.choice(cluster_list, size=G, replace=True)
        boot_idx = np.concatenate([cluster_indices[c] for c in sampled])

        y_b = y[boot_idx]
        r_b = risk_z[boot_idx]
        p_b = pche_z[boot_idx]
        int_b = r_b * p_b

        m_b = fit_ols(y_b, [r_b, p_b, int_b],
                      col_names=["risk_z", "pche_z", "interaction"])
        if m_b:
            boot_betas[i] = float(m_b.beta[m_b.col_names.index("interaction")])
        else:
            boot_betas[i] = np.nan

    return boot_betas


# ---------------------------------------------------------------------------
# Helper to extract a named coefficient from OLSResult
# ---------------------------------------------------------------------------
def extract_coef(m: Optional[OLSResult], name: str):
    """Return (beta, se, p) for a named coefficient, or (None, None, None)."""
    if m is None or name not in m.col_names:
        return None, None, None
    idx = m.col_names.index(name)
    return float(m.beta[idx]), float(m.se[idx]), float(m.pvals[idx])


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PCHE Analysis Pipeline v2")
    parser.add_argument(
        "--panel",
        default=str(Path(__file__).resolve().parent.parent / "data" / "panel_clean.csv"),
        help="Path to panel_clean.csv (317 state-year rows)",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    run_id = args.run_id or default_run_id()
    seed = args.seed
    rng = np.random.RandomState(seed)

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "output" / run_id
    fig_dir = base_dir / "figures" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(args.panel)
    print(f"[PCHE] Run ID: {run_id}")
    print(f"[PCHE] Seed: {seed}")
    print(f"[PCHE] Panel: {panel_path}")
    print(f"[PCHE] Output: {out_dir}")
    print(f"[PCHE] Figures: {fig_dir}")

    # ===================================================================
    # PHASE 0: LOAD PANEL AND VERIFY
    # ===================================================================
    print("\n=== PHASE 0: DATA LOADING ===")
    panel_hash = sha256_file(panel_path)
    panel = pd.read_csv(panel_path, low_memory=False)
    print(f"  Panel loaded: {len(panel)} rows x {panel.shape[1]} columns")
    print(f"  SHA-256: {panel_hash[:16]}...")

    # Verify no aggregate rows
    if "location_abbr" in panel.columns:
        agg_check = panel["location_abbr"].str.contains("Total|PRAMS Total", case=False, na=False)
        n_agg = int(agg_check.sum())
        print(f"  Aggregate row check: {n_agg} found {'(OK - none)' if n_agg == 0 else '*** WARNING ***'}")
        if n_agg > 0:
            panel = panel[~agg_check].copy()
            print(f"    Removed {n_agg} aggregate rows. Now {len(panel)} rows.")

    # PDS working sample: rows where outcome_ppd is non-null
    pds_panel = panel[panel["outcome_ppd"].notna()].copy()
    pds_panel["outcome_ppd"] = pds_panel["outcome_ppd"].astype(float)
    n_locations = pds_panel["location_abbr"].nunique()
    year_range = f"{int(pds_panel['year'].min())}-{int(pds_panel['year'].max())}"
    print(f"  PDS panel: {len(pds_panel)} rows with outcome")
    print(f"    {n_locations} unique locations, years {year_range}")
    print(f"    Era breakdown: {pds_panel['ppd_era'].value_counts().to_dict()}")

    # Save metadata
    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "panel_path": str(panel_path),
        "panel_sha256": panel_hash,
        "panel_rows": int(len(panel)),
        "pds_panel_rows": int(len(pds_panel)),
        "n_locations": n_locations,
        "year_range": year_range,
        "random_seed": seed,
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "min_n": MIN_N,
        "terminology": {
            "outcome": "PDS (postpartum depressive symptoms)",
            "construct": "PCHE (Pre-Conception Health Engagement)",
            "never_use": ["PPD prevalence", "PCHA", "Pre-Conception Health Agency"],
        },
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ===================================================================
    # PHASE 1: FIXED-EFFECTS-ONLY BASELINES
    # ===================================================================
    print("\n=== PHASE 1: FIXED-EFFECTS-ONLY BASELINES ===")
    y_pds = pds_panel["outcome_ppd"].values

    # 1a: State-only FE
    state_fe_d, state_fe_n = make_state_fe_dummies(pds_panel)
    fe_state_only = fit_ols(y_pds, state_fe_d, col_names=state_fe_n)
    if fe_state_only:
        print(f"  State-only FE: R2={fe_state_only.r2:.4f}, adj-R2={fe_state_only.adj_r2:.4f}, "
              f"n={fe_state_only.n}, k={fe_state_only.k}")
    else:
        print("  WARNING: State-only FE model failed to fit")

    # 1b: State + Year FE
    full_fe_d, full_fe_n = make_fe_dummies(pds_panel)
    fe_state_year = fit_ols(y_pds, full_fe_d, col_names=full_fe_n)
    if fe_state_year:
        print(f"  State+Year FE: R2={fe_state_year.r2:.4f}, adj-R2={fe_state_year.adj_r2:.4f}, "
              f"n={fe_state_year.n}, k={fe_state_year.k}")
    else:
        print("  WARNING: State+Year FE model failed to fit")

    # Incremental year contribution
    if fe_state_only and fe_state_year:
        incr_r2 = fe_state_year.r2 - fe_state_only.r2
        print(f"  Incremental R2 from year FE: {incr_r2:.4f}")

    fe_baseline = {
        "state_only_r2": fe_state_only.r2 if fe_state_only else None,
        "state_only_adj_r2": fe_state_only.adj_r2 if fe_state_only else None,
        "state_year_r2": fe_state_year.r2 if fe_state_year else None,
        "state_year_adj_r2": fe_state_year.adj_r2 if fe_state_year else None,
        "incremental_year_r2": incr_r2 if (fe_state_only and fe_state_year) else None,
        "n": fe_state_year.n if fe_state_year else 0,
        "k_state_only": fe_state_only.k if fe_state_only else 0,
        "k_state_year": fe_state_year.k if fe_state_year else 0,
    }

    # ===================================================================
    # PHASE 2: PCHE INDEX CONSTRUCTION (PRIMARY = 7 common components)
    # ===================================================================
    print("\n=== PHASE 2: PCHE INDEX CONSTRUCTION (PRIMARY: 7 COMMON COMPONENTS) ===")

    pche_avail = {}
    for qid in PCHE_COMMON_ORDER:
        desc = PCHE_COMMON[qid]
        if qid in pds_panel.columns:
            n_valid = int(pds_panel[qid].notna().sum())
            pche_avail[qid] = {"description": desc, "n_valid": n_valid}
            print(f"  {qid}: {desc[:60]:60s} n={n_valid}")
        else:
            pche_avail[qid] = {"description": desc, "n_valid": 0}
            print(f"  {qid}: {desc[:60]:60s} NOT IN PANEL")

    # Use only components present with sufficient data
    pche_components_primary = [q for q in PCHE_COMMON_ORDER
                               if q in pds_panel.columns
                               and pds_panel[q].notna().sum() >= MIN_N]
    print(f"\n  Primary PCHE components with >= {MIN_N} obs: {len(pche_components_primary)}")
    print(f"  Components: {pche_components_primary}")

    # Z-score each component
    pche_z_cols = {}
    for qid in pche_components_primary:
        vals = pds_panel[qid].values.astype(float)
        z = zscore(vals)
        pche_z_cols[qid] = z
        pds_panel[f"{qid}_z"] = z

    # PCHE composite: mean of available z-scores per row (require >= 3)
    z_matrix = np.column_stack([pche_z_cols[q] for q in pche_components_primary])
    n_valid_per_row = np.sum(~np.isnan(z_matrix), axis=1)
    pche_index = np.nanmean(z_matrix, axis=1)
    pche_index[n_valid_per_row < 3] = np.nan
    pds_panel["pche_index"] = pche_index
    pds_panel["pche_n_components"] = n_valid_per_row.astype(int)

    n_pche_valid = int(np.sum(~np.isnan(pche_index)))
    print(f"  PCHE index (primary): {n_pche_valid} valid rows (of {len(pds_panel)})")
    print(f"  PCHE mean={np.nanmean(pche_index):.3f}, SD={np.nanstd(pche_index):.3f}")

    # Component correlation matrix
    z_df = pd.DataFrame(pche_z_cols, index=pds_panel.index)
    comp_corr = z_df.corr()
    comp_corr.to_csv(out_dir / "pche_component_correlations.csv")
    print(f"  Component correlation matrix saved")

    # Cronbach's alpha
    z_clean = z_df.dropna()
    if len(z_clean) >= 10 and z_clean.shape[1] >= 2:
        k_items = z_clean.shape[1]
        item_vars = z_clean.var(ddof=1).values
        total_var = z_clean.sum(axis=1).var(ddof=1)
        alpha = (k_items / (k_items - 1)) * (1 - item_vars.sum() / total_var) if total_var > 0 else float("nan")
        print(f"  Cronbach's alpha (PCHE, 7 items): {alpha:.3f} (k={k_items}, n={len(z_clean)})")
    else:
        alpha = float("nan")
        print(f"  Cronbach's alpha: insufficient data")

    # ===================================================================
    # PHASE 2B: PARALLEL ANALYSIS (Horn 1965, 1000 iterations)
    # ===================================================================
    print("\n=== PHASE 2B: PARALLEL ANALYSIS (PCHE COMPONENTS) ===")
    z_for_pca = z_df.dropna()
    n_retain = 0
    pa_df = pd.DataFrame()
    if len(z_for_pca) >= 20:
        data_mat = z_for_pca.values
        corr_mat = np.corrcoef(data_mat, rowvar=False)
        actual_eigs = np.sort(np.linalg.eigvalsh(corr_mat))[::-1]
        pa_thresholds = parallel_analysis(data_mat, n_iter=1000, seed=seed)

        pa_rows = []
        for i in range(len(actual_eigs)):
            pa_rows.append({
                "component": i + 1,
                "actual_eigenvalue": float(actual_eigs[i]),
                "pa_95th_threshold": float(pa_thresholds[i]),
                "retain": actual_eigs[i] > pa_thresholds[i],
            })
        pa_df = pd.DataFrame(pa_rows)
        pa_df.to_csv(out_dir / "pche_parallel_analysis.csv", index=False)
        n_retain = int(pa_df["retain"].sum())
        print(f"  Parallel analysis: retain {n_retain} components")
        for _, r in pa_df.iterrows():
            marker = "***" if r["retain"] else ""
            print(f"    PC{int(r['component'])}: eigenvalue={r['actual_eigenvalue']:.3f} "
                  f"vs threshold={r['pa_95th_threshold']:.3f} {marker}")

        if n_retain >= 1:
            evals, evecs = np.linalg.eigh(corr_mat)
            idx = np.argsort(evals)[::-1]
            evals, evecs = evals[idx], evecs[:, idx]
            loadings = evecs[:, :n_retain] * np.sqrt(evals[:n_retain])
            loading_df = pd.DataFrame(
                loadings,
                index=pche_components_primary,
                columns=[f"PC{i+1}" for i in range(n_retain)],
            )
            loading_df.to_csv(out_dir / "pche_pca_loadings.csv")
            print(f"  PCA loadings saved ({n_retain} components)")
    else:
        print(f"  Insufficient data for parallel analysis (n={len(z_for_pca)})")

    # ===================================================================
    # PHASE 2C: STRUCTURAL ACCESS INDEX
    # ===================================================================
    print("\n=== PHASE 2C: STRUCTURAL ACCESS INDEX ===")
    struct_avail = [q for q in STRUCTURAL_ACCESS if q in pds_panel.columns
                    and pds_panel[q].notna().sum() >= MIN_N]
    print(f"  Structural access components available: {len(struct_avail)}")

    struct_z_cols = {}
    for qid in struct_avail:
        struct_z_cols[qid] = zscore(pds_panel[qid].values.astype(float))

    if struct_avail:
        struct_matrix = np.column_stack([struct_z_cols[q] for q in struct_avail])
        struct_n_valid = np.sum(~np.isnan(struct_matrix), axis=1)
        struct_index = np.nanmean(struct_matrix, axis=1)
        struct_index[struct_n_valid < 3] = np.nan
        pds_panel["structural_access_index"] = struct_index
        print(f"  Structural access index: {int(np.sum(~np.isnan(struct_index)))} valid rows")

    # ===================================================================
    # PHASE 2D: COUNSELING INTENSITY INDEX
    # ===================================================================
    print("\n=== PHASE 2D: COUNSELING INTENSITY INDEX ===")
    couns_avail = [q for q in COUNSELING_INDICATORS if q in pds_panel.columns
                   and pds_panel[q].notna().sum() >= MIN_N]
    print(f"  Counseling components available: {len(couns_avail)}")

    couns_z_cols = {}
    for qid in couns_avail:
        couns_z_cols[qid] = zscore(pds_panel[qid].values.astype(float))

    if couns_avail:
        couns_matrix = np.column_stack([couns_z_cols[q] for q in couns_avail])
        couns_n_valid = np.sum(~np.isnan(couns_matrix), axis=1)
        couns_index = np.nanmean(couns_matrix, axis=1)
        couns_index[couns_n_valid < 3] = np.nan
        pds_panel["counseling_index"] = couns_index
        print(f"  Counseling index: {int(np.sum(~np.isnan(couns_index)))} valid rows")

    # ===================================================================
    # PHASE 3: PCHE x PARTNER-STRESS INTERACTION MODELS (ALL 4 ANCHORS)
    # ===================================================================
    print("\n=== PHASE 3: PCHE x PARTNER-STRESS INTERACTION MODELS ===")

    interaction_results = []
    for risk_qid, risk_desc in RISK_ANCHORS.items():
        if risk_qid not in pds_panel.columns:
            print(f"\n  {risk_qid}: NOT IN PANEL, skipping")
            continue

        mask = (
            pds_panel["outcome_ppd"].notna() &
            pds_panel[risk_qid].notna() &
            pds_panel["pche_index"].notna()
        )
        work = pds_panel[mask].copy()
        n_work = len(work)
        if n_work < MIN_N:
            print(f"\n  {risk_qid}: n={n_work} < {MIN_N}, skipping")
            continue

        anchor_type = "Primary" if risk_qid in PRIMARY_RISK else "Secondary"
        print(f"\n  --- {risk_qid} ({anchor_type}): {risk_desc} (n={n_work}) ---")

        y = work["outcome_ppd"].values
        risk_z = zscore(work[risk_qid].values)
        pche_z = zscore(work["pche_index"].values)
        interaction = risk_z * pche_z
        clusters = work["location_abbr"].values
        n_clusters = len(np.unique(clusters))

        # Model 1: Risk-only OLS
        m_risk = fit_ols(y, [risk_z], col_names=["risk_z"])

        # Model 2: Risk + PCHE additive OLS
        m_add = fit_ols(y, [risk_z, pche_z], col_names=["risk_z", "pche_z"])

        # Model 3: Risk + PCHE + interaction OLS
        m_int = fit_ols(y, [risk_z, pche_z, interaction],
                        col_names=["risk_z", "pche_z", "risk_z_x_pche_z"])

        # Model 4: State+Year FE interaction model
        fe_d, fe_n = make_fe_dummies(work)
        m_fe_int = fit_ols(y, fe_d + [risk_z, pche_z, interaction],
                           col_names=fe_n + ["risk_z", "pche_z", "risk_z_x_pche_z"])

        # Model 5: CR1 cluster-robust for ALL models (not just significant)
        m_cr_risk = fit_ols_cluster_robust(y, [risk_z], clusters, col_names=["risk_z"])
        m_cr_add = fit_ols_cluster_robust(y, [risk_z, pche_z], clusters,
                                          col_names=["risk_z", "pche_z"])
        m_cr_int = fit_ols_cluster_robust(y, [risk_z, pche_z, interaction], clusters,
                                          col_names=["risk_z", "pche_z", "risk_z_x_pche_z"])
        m_cr_fe = fit_ols_cluster_robust(y, fe_d + [risk_z, pche_z, interaction], clusters,
                                         col_names=fe_n + ["risk_z", "pche_z", "risk_z_x_pche_z"])

        # Extract coefficients
        int_beta_ols, int_se_ols, int_p_ols = extract_coef(m_int, "risk_z_x_pche_z")
        int_beta_fe, int_se_fe, int_p_fe = extract_coef(m_fe_int, "risk_z_x_pche_z")
        int_beta_cr, int_se_cr, int_p_cr = extract_coef(m_cr_int, "risk_z_x_pche_z")
        int_beta_cr_fe, int_se_cr_fe, int_p_cr_fe = extract_coef(m_cr_fe, "risk_z_x_pche_z")

        # CR1 for risk-only and additive models too (no selective reporting)
        risk_beta_cr, risk_se_cr, risk_p_cr = extract_coef(m_cr_risk, "risk_z")
        pche_beta_cr_add, pche_se_cr_add, pche_p_cr_add = extract_coef(m_cr_add, "pche_z")

        if int_beta_ols is not None:
            print(f"    OLS interaction: beta={int_beta_ols:.4f}, SE={int_se_ols:.4f}, p={int_p_ols:.4f}")
        if int_beta_fe is not None:
            print(f"    FE interaction:  beta={int_beta_fe:.4f}, SE={int_se_fe:.4f}, p={int_p_fe:.4f}")
        if int_beta_cr is not None:
            print(f"    CR1 interaction: beta={int_beta_cr:.4f}, SE={int_se_cr:.4f}, p={int_p_cr:.4f}")
        if int_beta_cr_fe is not None:
            print(f"    CR1-FE interaction: beta={int_beta_cr_fe:.4f}, SE={int_se_cr_fe:.4f}, p={int_p_cr_fe:.4f}")

        # --- Wild cluster bootstrap (Rademacher) ---
        wild_p = np.nan
        if int_beta_ols is not None and n_clusters >= 5:
            print(f"    Running wild cluster bootstrap ({N_BOOTSTRAP} iterations)...")
            # Full model: intercept + risk + pche + interaction
            X_full = np.column_stack([np.ones(n_work), risk_z, pche_z, interaction])
            # Restricted: no interaction
            X_restr = np.column_stack([np.ones(n_work), risk_z, pche_z])
            wild_p = wild_cluster_bootstrap(
                y, X_full, X_restr, clusters,
                test_col_idx=3,  # interaction is column 3
                n_boot=N_BOOTSTRAP, seed=seed,
            )
            print(f"    Wild cluster bootstrap p={wild_p:.4f}")

        # --- Cluster-aware permutation ---
        perm_p = None
        if int_beta_ols is not None:
            print(f"    Running cluster permutation ({N_PERMUTATIONS} iterations)...")
            perm_p, perm_betas = cluster_permutation_test(
                y, risk_z, pche_z, clusters,
                n_perm=N_PERMUTATIONS, seed=seed,
            )
            if perm_p is not None:
                print(f"    Cluster permutation p={perm_p:.4f}")

        # --- Cluster-aware bootstrap CI ---
        boot_ci_lo = boot_ci_hi = None
        if int_beta_ols is not None:
            print(f"    Running cluster bootstrap CI ({N_BOOTSTRAP} iterations)...")
            boot_betas = cluster_bootstrap_ci(
                y, risk_z, pche_z, clusters,
                n_boot=N_BOOTSTRAP, seed=seed,
            )
            valid_boots = boot_betas[~np.isnan(boot_betas)]
            if len(valid_boots) > 100:
                boot_ci_lo = float(np.percentile(valid_boots, 2.5))
                boot_ci_hi = float(np.percentile(valid_boots, 97.5))
                print(f"    Cluster bootstrap 95% CI: [{boot_ci_lo:.4f}, {boot_ci_hi:.4f}]")

        # --- Benefit estimate at +1 SD partner stress ---
        # Correct: at risk_z=+1, compare pche_z=-1 vs pche_z=+1
        # y(+1,-1) = b0 + b1(1) + b2(-1) + b3(1)(-1) = b0 + b1 - b2 - b3
        # y(+1,+1) = b0 + b1(1) + b2(+1) + b3(1)(+1) = b0 + b1 + b2 + b3
        # Benefit = y(+1,-1) - y(+1,+1) = -2*b2 - 2*b3
        # (positive = PDS reduction when PCHE goes from -1 to +1)
        benefit_pp = None
        if m_int and int_beta_ols is not None:
            b2 = float(m_int.beta[m_int.col_names.index("pche_z")])
            b3 = float(m_int.beta[m_int.col_names.index("risk_z_x_pche_z")])
            # PDS at high risk, low PCHE minus PDS at high risk, high PCHE
            benefit_pp = -2 * b2 - 2 * b3  # positive = lower PDS with higher PCHE
            print(f"    Benefit at +1 SD risk (PCHE -1->+1): {benefit_pp:.2f} pp PDS reduction")

        # Attenuation signature: risk+, PCHE-, interaction-
        if m_int:
            b1 = float(m_int.beta[m_int.col_names.index("risk_z")])
            b2 = float(m_int.beta[m_int.col_names.index("pche_z")])
            b3 = float(m_int.beta[m_int.col_names.index("risk_z_x_pche_z")])
            attenuation = (b1 > 0) and (b2 < 0) and (b3 < 0)
        else:
            b1 = b2 = b3 = None
            attenuation = False

        result = {
            "risk_qid": risk_qid,
            "risk_description": risk_desc,
            "anchor_type": anchor_type,
            "n": n_work,
            "n_clusters": n_clusters,
            # Risk-only
            "risk_only_r2": m_risk.r2 if m_risk else None,
            "risk_only_adj_r2": m_risk.adj_r2 if m_risk else None,
            "risk_beta_cr": risk_beta_cr,
            "risk_se_cr": risk_se_cr,
            "risk_p_cr": risk_p_cr,
            # Additive
            "additive_r2": m_add.r2 if m_add else None,
            "additive_adj_r2": m_add.adj_r2 if m_add else None,
            "pche_additive_beta_cr": pche_beta_cr_add,
            "pche_additive_se_cr": pche_se_cr_add,
            "pche_additive_p_cr": pche_p_cr_add,
            # OLS interaction
            "interaction_r2": m_int.r2 if m_int else None,
            "interaction_adj_r2": m_int.adj_r2 if m_int else None,
            "interaction_beta": int_beta_ols,
            "interaction_se": int_se_ols,
            "interaction_p_parametric": int_p_ols,
            # CR1 interaction (OLS)
            "cr1_interaction_beta": int_beta_cr,
            "cr1_interaction_se": int_se_cr,
            "cr1_interaction_p": int_p_cr,
            # FE interaction
            "fe_interaction_beta": int_beta_fe,
            "fe_interaction_se": int_se_fe,
            "fe_interaction_p": int_p_fe,
            "fe_r2": m_fe_int.r2 if m_fe_int else None,
            "fe_adj_r2": m_fe_int.adj_r2 if m_fe_int else None,
            # CR1-FE interaction
            "cr1_fe_interaction_beta": int_beta_cr_fe,
            "cr1_fe_interaction_se": int_se_cr_fe,
            "cr1_fe_interaction_p": int_p_cr_fe,
            # Wild cluster bootstrap
            "wild_cluster_p": float(wild_p) if not np.isnan(wild_p) else None,
            # Cluster permutation
            "cluster_permutation_p": perm_p,
            # Cluster bootstrap CI
            "cluster_boot_ci_lo": boot_ci_lo,
            "cluster_boot_ci_hi": boot_ci_hi,
            # Main effects
            "risk_beta": b1,
            "pche_beta": b2,
            "interaction_beta_raw": b3,
            "attenuation_signature": attenuation,
            # Benefit at +1 SD risk
            "benefit_pp_at_plus1sd_risk": benefit_pp,
            # FE baselines
            "fe_state_only_r2": fe_baseline["state_only_r2"],
            "fe_state_year_r2": fe_baseline["state_year_r2"],
        }
        interaction_results.append(result)

    int_df = pd.DataFrame(interaction_results)
    int_df.to_csv(out_dir / "pche_interaction_models.csv", index=False)
    print(f"\n  Interaction models saved: {len(int_df)} risk anchors tested")

    # ===================================================================
    # PHASE 4: DISCRIMINANT VALIDITY (PCHE vs STRUCTURAL ACCESS)
    # ===================================================================
    print("\n=== PHASE 4: DISCRIMINANT VALIDITY (report cautiously) ===")

    discrim_results = []
    if "structural_access_index" in pds_panel.columns:
        both_valid = pds_panel[["pche_index", "structural_access_index"]].dropna()
        if len(both_valid) >= 10:
            r_val, p_val = stats.pearsonr(
                both_valid["pche_index"], both_valid["structural_access_index"]
            )
            print(f"  PCHE vs structural access: r={r_val:.3f}, p={p_val:.4f}, n={len(both_valid)}")
            print(f"  NOTE: Interpret cautiously — ecological correlations may not reflect individual-level associations")
            discrim_results.append({
                "comparison": "pche_vs_structural_access",
                "r": r_val, "p": p_val, "n": len(both_valid),
            })

    for risk_qid in PRIMARY_RISK:
        if risk_qid not in pds_panel.columns:
            continue
        mask = (
            pds_panel["outcome_ppd"].notna() &
            pds_panel[risk_qid].notna() &
            pds_panel["pche_index"].notna() &
            pds_panel["structural_access_index"].notna()
        )
        work = pds_panel[mask].copy()
        if len(work) < MIN_N:
            continue

        y = work["outcome_ppd"].values
        risk_z = zscore(work[risk_qid].values)
        pche_z_w = zscore(work["pche_index"].values)
        struct_z = zscore(work["structural_access_index"].values)

        m_pche = fit_ols(y, [risk_z, pche_z_w, risk_z * pche_z_w],
                         col_names=["risk_z", "pche_z", "risk_x_pche"])
        m_struct = fit_ols(y, [risk_z, struct_z, risk_z * struct_z],
                           col_names=["risk_z", "struct_z", "risk_x_struct"])
        m_both = fit_ols(y, [risk_z, pche_z_w, struct_z, risk_z * pche_z_w, risk_z * struct_z],
                         col_names=["risk_z", "pche_z", "struct_z", "risk_x_pche", "risk_x_struct"])

        print(f"\n  Head-to-head for {risk_qid} (n={len(work)}):")
        if m_pche:
            b, s, p = extract_coef(m_pche, "risk_x_pche")
            print(f"    PCHE-only: interaction beta={b:.4f}, p={p:.4f}, adj-R2={m_pche.adj_r2:.4f}")
        if m_struct:
            b, s, p = extract_coef(m_struct, "risk_x_struct")
            print(f"    Struct-only: interaction beta={b:.4f}, p={p:.4f}, adj-R2={m_struct.adj_r2:.4f}")
        if m_both:
            bp, sp, pp = extract_coef(m_both, "risk_x_pche")
            bs, ss, ps = extract_coef(m_both, "risk_x_struct")
            print(f"    Both: PCHE int beta={bp:.4f} p={pp:.4f}, struct int beta={bs:.4f} p={ps:.4f}")

        row = {"comparison": f"head_to_head_{risk_qid}", "n": len(work)}
        if m_pche:
            row["pche_int_beta"], _, row["pche_int_p"] = extract_coef(m_pche, "risk_x_pche")
        if m_struct:
            row["struct_int_beta"], _, row["struct_int_p"] = extract_coef(m_struct, "risk_x_struct")
        if m_both:
            row["both_pche_int_beta"], _, row["both_pche_int_p"] = extract_coef(m_both, "risk_x_pche")
            row["both_struct_int_beta"], _, row["both_struct_int_p"] = extract_coef(m_both, "risk_x_struct")
        discrim_results.append(row)

    pd.DataFrame(discrim_results).to_csv(out_dir / "discriminant_validity.csv", index=False)

    # ===================================================================
    # PHASE 5: COUNSELING PARADOX
    # ===================================================================
    print("\n=== PHASE 5: COUNSELING PARADOX ANALYSIS ===")

    paradox_results = []
    if "counseling_index" in pds_panel.columns:
        for risk_qid in PRIMARY_RISK:
            if risk_qid not in pds_panel.columns:
                continue
            mask = (
                pds_panel["outcome_ppd"].notna() &
                pds_panel[risk_qid].notna() &
                pds_panel["counseling_index"].notna()
            )
            work = pds_panel[mask].copy()
            if len(work) < MIN_N:
                continue

            y = work["outcome_ppd"].values
            risk_z = zscore(work[risk_qid].values)
            couns_z = zscore(work["counseling_index"].values)

            m_couns = fit_ols(y, [couns_z], col_names=["counseling_z"])
            fe_d, fe_n = make_fe_dummies(work)
            m_couns_fe = fit_ols(y, fe_d + [couns_z], col_names=fe_n + ["counseling_z"])
            m_couns_int = fit_ols(y, [risk_z, couns_z, risk_z * couns_z],
                                  col_names=["risk_z", "counseling_z", "risk_x_counseling"])

            result = {"risk_qid": risk_qid, "n": len(work)}
            if m_couns:
                b, s, p = extract_coef(m_couns, "counseling_z")
                result["counseling_simple_beta"] = b
                result["counseling_simple_p"] = p
                print(f"  {risk_qid}: Counseling simple beta={b:.4f}, p={p:.4f}")
            if m_couns_fe:
                b, s, p = extract_coef(m_couns_fe, "counseling_z")
                result["counseling_fe_beta"] = b
                result["counseling_fe_p"] = p
                print(f"  {risk_qid}: Counseling FE beta={b:.4f}, p={p:.4f}")
            if m_couns_int:
                b, s, p = extract_coef(m_couns_int, "risk_x_counseling")
                result["counseling_interaction_beta"] = b
                result["counseling_interaction_p"] = p
                print(f"  {risk_qid}: Counseling interaction beta={b:.4f}, p={p:.4f}")

            # Simultaneous PCHE + counseling
            if pds_panel["pche_index"].notna().sum() >= MIN_N:
                mask2 = mask & pds_panel["pche_index"].notna()
                work2 = pds_panel[mask2].copy()
                if len(work2) >= MIN_N:
                    y2 = work2["outcome_ppd"].values
                    r2 = zscore(work2[risk_qid].values)
                    p2 = zscore(work2["pche_index"].values)
                    c2 = zscore(work2["counseling_index"].values)
                    m_both = fit_ols(y2, [r2, p2, c2, r2*p2, r2*c2],
                                     col_names=["risk_z", "pche_z", "counseling_z",
                                                 "risk_x_pche", "risk_x_counseling"])
                    if m_both:
                        bp, _, pp = extract_coef(m_both, "risk_x_pche")
                        bc, _, pc = extract_coef(m_both, "risk_x_counseling")
                        result["simultaneous_pche_int_beta"] = bp
                        result["simultaneous_pche_int_p"] = pp
                        result["simultaneous_couns_int_beta"] = bc
                        result["simultaneous_couns_int_p"] = pc

            paradox_results.append(result)

    pd.DataFrame(paradox_results).to_csv(out_dir / "counseling_paradox.csv", index=False)

    # ===================================================================
    # PHASE 6: CONSTANT-SAMPLE SENSITIVITY
    # ===================================================================
    print("\n=== PHASE 6: CONSTANT-SAMPLE SENSITIVITY ===")

    key_vars = ["outcome_ppd", "pche_index"] + [q for q in RISK_ANCHORS if q in pds_panel.columns]
    key_vars_avail = [v for v in key_vars if v in pds_panel.columns]
    const_mask = pds_panel[key_vars_avail].notna().all(axis=1)
    const_panel = pds_panel[const_mask].copy()
    print(f"  Constant sample: {len(const_panel)} rows (all key vars non-missing)")

    const_results = []
    for risk_qid in RISK_ANCHORS:
        if risk_qid not in const_panel.columns or len(const_panel) < MIN_N:
            continue
        y = const_panel["outcome_ppd"].values
        risk_z = zscore(const_panel[risk_qid].values)
        pche_z_c = zscore(const_panel["pche_index"].values)
        if np.all(np.isnan(risk_z)) or np.all(np.isnan(pche_z_c)):
            continue

        m = fit_ols(y, [risk_z, pche_z_c, risk_z * pche_z_c],
                    col_names=["risk_z", "pche_z", "risk_x_pche"])
        if m:
            b, s, p = extract_coef(m, "risk_x_pche")
            print(f"  {risk_qid}: constant-sample interaction beta={b:.4f}, p={p:.4f}, n={m.n}")
            const_results.append({
                "risk_qid": risk_qid, "n": m.n,
                "interaction_beta": b, "interaction_p": p, "adj_r2": m.adj_r2,
            })

    pd.DataFrame(const_results).to_csv(out_dir / "constant_sample_sensitivity.csv", index=False)

    # ===================================================================
    # PHASE 7: ERA-SPECIFIC SENSITIVITY
    # ===================================================================
    print("\n=== PHASE 7: ERA-SPECIFIC SENSITIVITY ===")

    era_results = []
    for era in sorted(pds_panel["ppd_era"].dropna().unique()):
        era_panel = pds_panel[pds_panel["ppd_era"] == era].copy()
        print(f"\n  Era {era}: {len(era_panel)} rows")
        for risk_qid in RISK_ANCHORS:
            if risk_qid not in era_panel.columns:
                continue
            mask = (
                era_panel["outcome_ppd"].notna() &
                era_panel[risk_qid].notna() &
                era_panel["pche_index"].notna()
            )
            work = era_panel[mask].copy()
            if len(work) < MIN_N:
                print(f"    {risk_qid}: n={len(work)} < {MIN_N}, skipping")
                continue

            y = work["outcome_ppd"].values
            risk_z = zscore(work[risk_qid].values)
            pche_z_e = zscore(work["pche_index"].values)

            m = fit_ols(y, [risk_z, pche_z_e, risk_z * pche_z_e],
                        col_names=["risk_z", "pche_z", "risk_x_pche"])
            if m:
                b, s, p = extract_coef(m, "risk_x_pche")
                print(f"    {risk_qid}: interaction beta={b:.4f}, p={p:.4f}")
                era_results.append({
                    "era": era, "risk_qid": risk_qid, "n": m.n,
                    "interaction_beta": b, "interaction_p": p, "adj_r2": m.adj_r2,
                })

    pd.DataFrame(era_results).to_csv(out_dir / "era_sensitivity.csv", index=False)

    # ===================================================================
    # PHASE 8: LEAVE-ONE-STATE-OUT (states with >= 2 obs)
    # ===================================================================
    print("\n=== PHASE 8: LEAVE-ONE-STATE-OUT STABILITY ===")

    loso_results = []
    for risk_qid in RISK_ANCHORS:
        if risk_qid not in pds_panel.columns:
            continue
        mask = (
            pds_panel["outcome_ppd"].notna() &
            pds_panel[risk_qid].notna() &
            pds_panel["pche_index"].notna()
        )
        work = pds_panel[mask].copy()
        if len(work) < MIN_N:
            continue

        states = work["location_abbr"].unique()
        state_counts = work["location_abbr"].value_counts()
        eligible_states = state_counts[state_counts >= 2].index.tolist()
        print(f"  {risk_qid}: {len(eligible_states)} states with >= 2 obs "
              f"(of {len(states)} total)")

        for st in sorted(eligible_states):
            sub = work[work["location_abbr"] != st].copy()
            if len(sub) < MIN_N:
                continue
            y = sub["outcome_ppd"].values
            risk_z = zscore(sub[risk_qid].values)
            pche_z_l = zscore(sub["pche_index"].values)
            m = fit_ols(y, [risk_z, pche_z_l, risk_z * pche_z_l],
                        col_names=["risk_z", "pche_z", "risk_x_pche"])
            if m:
                b, s, p = extract_coef(m, "risk_x_pche")
                loso_results.append({
                    "risk_qid": risk_qid, "excluded_state": st,
                    "n": m.n, "interaction_beta": b, "interaction_p": p,
                })

    loso_df = pd.DataFrame(loso_results)
    loso_df.to_csv(out_dir / "leave_one_state_out.csv", index=False)
    if len(loso_df) > 0:
        for rq in loso_df["risk_qid"].unique():
            sub = loso_df[loso_df["risk_qid"] == rq]
            betas = sub["interaction_beta"].values
            print(f"  {rq} LOSO: beta range [{betas.min():.4f}, {betas.max():.4f}], "
                  f"mean={betas.mean():.4f}, SD={betas.std():.4f}")

    # ===================================================================
    # PHASE 9: INDIVIDUAL PCHE COMPONENT ANALYSIS WITH FDR
    # ===================================================================
    print("\n=== PHASE 9: INDIVIDUAL PCHE COMPONENT ANALYSIS ===")

    component_results = []
    for risk_qid in RISK_ANCHORS:
        if risk_qid not in pds_panel.columns:
            continue
        for comp_qid in pche_components_primary:
            mask = (
                pds_panel["outcome_ppd"].notna() &
                pds_panel[risk_qid].notna() &
                pds_panel[comp_qid].notna()
            )
            work = pds_panel[mask].copy()
            if len(work) < MIN_N:
                continue

            y = work["outcome_ppd"].values
            risk_z = zscore(work[risk_qid].values)
            comp_z = zscore(work[comp_qid].values)

            m = fit_ols(y, [risk_z, comp_z, risk_z * comp_z],
                        col_names=["risk_z", "comp_z", "risk_x_comp"])
            if m:
                b, s, p = extract_coef(m, "risk_x_comp")
                component_results.append({
                    "risk_qid": risk_qid,
                    "component_qid": comp_qid,
                    "component_description": PCHE_COMMON.get(comp_qid, ""),
                    "n": m.n,
                    "interaction_beta": b,
                    "interaction_se": s,
                    "interaction_p": p,
                    "adj_r2": m.adj_r2,
                })

    comp_df = pd.DataFrame(component_results)
    if len(comp_df) > 0:
        comp_df["interaction_fdr"] = bh_fdr(comp_df["interaction_p"].values)
        comp_df = comp_df.sort_values("interaction_p")
    comp_df.to_csv(out_dir / "pche_component_interactions.csv", index=False)
    print(f"  Component-level results: {len(comp_df)} models")

    # ===================================================================
    # PHASE 10: LEAVE-ONE-COMPONENT-OUT
    # ===================================================================
    print("\n=== PHASE 10: LEAVE-ONE-COMPONENT-OUT ===")

    loco_results = []
    for risk_qid in RISK_ANCHORS:
        if risk_qid not in pds_panel.columns:
            continue
        for drop_qid in pche_components_primary:
            remaining = [q for q in pche_components_primary if q != drop_qid]
            if len(remaining) < 3:
                continue

            z_mat_drop = np.column_stack([pche_z_cols[q] for q in remaining])
            n_valid_drop = np.sum(~np.isnan(z_mat_drop), axis=1)
            pche_drop = np.nanmean(z_mat_drop, axis=1)
            pche_drop[n_valid_drop < 3] = np.nan

            mask = (
                pds_panel["outcome_ppd"].notna() &
                pds_panel[risk_qid].notna() &
                pd.Series(~np.isnan(pche_drop), index=pds_panel.index)
            )
            work_idx = pds_panel.index[mask]
            if len(work_idx) < MIN_N:
                continue

            y = pds_panel.loc[work_idx, "outcome_ppd"].values
            risk_z = zscore(pds_panel.loc[work_idx, risk_qid].values)
            pche_z_drop = zscore(pche_drop[mask.values])

            m = fit_ols(y, [risk_z, pche_z_drop, risk_z * pche_z_drop],
                        col_names=["risk_z", "pche_z", "risk_x_pche"])
            if m:
                b, s, p = extract_coef(m, "risk_x_pche")
                loco_results.append({
                    "risk_qid": risk_qid,
                    "dropped_component": drop_qid,
                    "dropped_description": PCHE_COMMON.get(drop_qid, ""),
                    "n": m.n, "interaction_beta": b,
                    "interaction_p": p, "adj_r2": m.adj_r2,
                })

    loco_df = pd.DataFrame(loco_results)
    loco_df.to_csv(out_dir / "leave_one_component_out.csv", index=False)
    print(f"  Leave-one-component-out results: {len(loco_df)} models")

    # ===================================================================
    # PHASE 11: FIGURES (correct labels: PCHE, PDS)
    # ===================================================================
    print("\n=== PHASE 11: GENERATING FIGURES ===")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # --- Figure 1: PCHE index distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pche_valid = pds_panel["pche_index"].dropna()
    axes[0].hist(pche_valid, bins=25, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("PCHE Index (z-score composite)")
    axes[0].set_ylabel("Frequency (state-year rows)")
    axes[0].set_title("Distribution of Pre-Conception Health Engagement Index")
    axes[0].axvline(pche_valid.mean(), color="red", linestyle="--",
                    label=f"Mean={pche_valid.mean():.2f}")
    axes[0].legend()

    comp_counts = pds_panel.loc[pds_panel["pche_index"].notna(), "pche_n_components"]
    axes[1].hist(comp_counts, bins=range(3, int(comp_counts.max())+2), color="#55A868",
                 edgecolor="white", alpha=0.8, align="left")
    axes[1].set_xlabel("Number of PCHE Components Available")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Component Coverage per State-Year")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_pche_distribution.png")
    plt.close(fig)
    print("  Figure 1: PCHE distribution saved")

    # --- Figure 2: Interaction scatter plots ---
    for risk_qid in RISK_ANCHORS:
        if risk_qid not in pds_panel.columns:
            continue
        mask = (
            pds_panel["outcome_ppd"].notna() &
            pds_panel[risk_qid].notna() &
            pds_panel["pche_index"].notna()
        )
        work = pds_panel[mask].copy()
        if len(work) < MIN_N:
            continue

        pche_median = work["pche_index"].median()
        work["pche_group"] = np.where(work["pche_index"] >= pche_median,
                                      "High PCHE", "Low PCHE")

        fig, ax = plt.subplots(figsize=(8, 6))
        for grp, color, marker in [("Low PCHE", "#C44E52", "o"), ("High PCHE", "#4C72B0", "s")]:
            g = work[work["pche_group"] == grp]
            ax.scatter(g[risk_qid], g["outcome_ppd"], c=color, marker=marker,
                       alpha=0.6, s=40, label=grp, edgecolors="white", linewidths=0.5)
            if len(g) >= 5:
                z = np.polyfit(g[risk_qid].values, g["outcome_ppd"].values, 1)
                p = np.poly1d(z)
                x_range = np.linspace(g[risk_qid].min(), g[risk_qid].max(), 50)
                ax.plot(x_range, p(x_range), color=color, linewidth=2, alpha=0.8)

        risk_label = RISK_ANCHORS.get(risk_qid, risk_qid)
        ax.set_xlabel(f"Partner Stress ({risk_qid}): {risk_label[:50]}")
        ax.set_ylabel("Postpartum Depressive Symptoms (%)")
        ax.set_title(f"Partner Stress x PCHE Interaction\n({risk_qid}, n={len(work)})")
        ax.legend(loc="upper left")
        fig.savefig(fig_dir / f"fig2_interaction_{risk_qid}.png")
        plt.close(fig)
        print(f"  Figure 2: Interaction plot for {risk_qid} saved")

    # --- Figure 3: LOSO stability forest plot ---
    if len(loso_df) > 0:
        for risk_qid in loso_df["risk_qid"].unique():
            sub = loso_df[loso_df["risk_qid"] == risk_qid].sort_values("interaction_beta")
            fig, ax = plt.subplots(figsize=(8, max(6, len(sub) * 0.25)))
            y_pos = range(len(sub))
            colors = ["#C44E52" if p < 0.05 else "#8C8C8C" for p in sub["interaction_p"]]
            ax.barh(list(y_pos), sub["interaction_beta"], color=colors, height=0.7,
                    edgecolor="white", alpha=0.8)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels([f"Drop {s}" for s in sub["excluded_state"]], fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            full_beta = int_df.loc[int_df["risk_qid"] == risk_qid, "interaction_beta"].values
            if len(full_beta) > 0 and full_beta[0] is not None:
                ax.axvline(full_beta[0], color="#4C72B0", linewidth=2, linestyle="--",
                           label=f"Full sample: {full_beta[0]:.3f}")
            ax.set_xlabel("Interaction Beta (Risk x PCHE)")
            ax.set_title(f"Leave-One-State-Out Stability ({risk_qid})")
            ax.legend()
            plt.tight_layout()
            fig.savefig(fig_dir / f"fig3_loso_{risk_qid}.png")
            plt.close(fig)
            print(f"  Figure 3: LOSO forest plot for {risk_qid} saved")

    # --- Figure 4: Component-level interaction heatmap ---
    if len(comp_df) > 0:
        pivot = comp_df.pivot_table(index="component_qid", columns="risk_qid",
                                     values="interaction_beta", aggfunc="first")
        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.4)))
        vmax = max(abs(pivot.values[np.isfinite(pivot.values)].min()),
                   abs(pivot.values[np.isfinite(pivot.values)].max())) if np.any(np.isfinite(pivot.values)) else 1
        im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ylabels = [f"{q} ({PCHE_COMMON.get(q, '')[:30]})" for q in pivot.index]
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(ylabels, fontsize=8)
        plt.colorbar(im, ax=ax, label="Interaction Beta")
        ax.set_title("PCHE Component x Risk Anchor Interaction Betas")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if abs(val) > 0.3 else "black")
        plt.tight_layout()
        fig.savefig(fig_dir / "fig4_component_heatmap.png")
        plt.close(fig)
        print("  Figure 4: Component heatmap saved")

    # --- Figure 5: Parallel analysis scree plot ---
    if len(pa_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(pa_df["component"], pa_df["actual_eigenvalue"], "o-", color="#4C72B0",
                label="Actual eigenvalues", markersize=8)
        ax.plot(pa_df["component"], pa_df["pa_95th_threshold"], "s--", color="#C44E52",
                label="95th percentile (random)", markersize=6)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Kaiser criterion (=1)")
        ax.set_xlabel("Component Number")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Parallel Analysis: PCHE Component Dimensionality\n(Horn 1965, 1000 iterations)")
        ax.legend()
        ax.set_xticks(pa_df["component"].values)
        plt.tight_layout()
        fig.savefig(fig_dir / "fig5_parallel_analysis.png")
        plt.close(fig)
        print("  Figure 5: Parallel analysis scree plot saved")

    # --- Figure 6: Summary model comparison ---
    if len(int_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        risk_labels = [f"{r[:6]}" for r in int_df["risk_qid"]]
        x = np.arange(len(int_df))
        w = 0.2

        axes[0].bar(x - w, int_df["risk_only_adj_r2"].fillna(0), w,
                    label="Risk-only", color="#8C8C8C")
        axes[0].bar(x, int_df["additive_adj_r2"].fillna(0), w,
                    label="Risk + PCHE", color="#55A868")
        axes[0].bar(x + w, int_df["interaction_adj_r2"].fillna(0), w,
                    label="Risk x PCHE", color="#4C72B0")
        if fe_baseline["state_year_adj_r2"] is not None:
            axes[0].axhline(fe_baseline["state_year_adj_r2"], color="red", linestyle="--",
                            label=f"State+Year FE: {fe_baseline['state_year_adj_r2']:.3f}")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(risk_labels)
        axes[0].set_ylabel("Adjusted R-squared")
        axes[0].set_title("Model Fit Comparison")
        axes[0].legend(fontsize=9)

        betas = int_df["interaction_beta"].fillna(0).values
        ci_lo = int_df["cluster_boot_ci_lo"].fillna(0).values
        ci_hi = int_df["cluster_boot_ci_hi"].fillna(0).values
        colors = ["#4C72B0" if p is not None and p < 0.05 else "#C44E52" if p is not None and p < 0.10 else "#8C8C8C"
                  for p in int_df["cluster_permutation_p"]]
        axes[1].barh(x, betas, color=colors, height=0.5, edgecolor="white")
        axes[1].errorbar(betas, x, xerr=[betas - ci_lo, ci_hi - betas],
                         fmt="none", ecolor="black", capsize=3)
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(risk_labels)
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Interaction Beta (Risk x PCHE)")
        axes[1].set_title("PCHE Moderation Effects with Cluster Bootstrap 95% CI")
        plt.tight_layout()
        fig.savefig(fig_dir / "fig6_model_comparison.png")
        plt.close(fig)
        print("  Figure 6: Model comparison saved")

    # ===================================================================
    # PHASE 12: DESCRIPTIVE STATISTICS + PANEL ACCOUNTING TABLE
    # ===================================================================
    print("\n=== PHASE 12: DESCRIPTIVE STATISTICS + PANEL ACCOUNTING ===")

    # Descriptive stats
    desc_vars = (
        ["outcome_ppd", "pche_index"]
        + [q for q in RISK_ANCHORS if q in pds_panel.columns]
        + list(pche_components_primary)
    )
    desc_rows = []
    for var in desc_vars:
        if var not in pds_panel.columns:
            continue
        vals = pds_panel[var].dropna()
        desc_rows.append({
            "variable": var,
            "description": (
                PCHE_COMMON.get(var, "") or
                RISK_ANCHORS.get(var, "") or
                ("PDS rate (harmonized)" if var == "outcome_ppd" else
                 "PCHE composite index" if var == "pche_index" else "")
            ),
            "n": len(vals),
            "mean": float(vals.mean()),
            "sd": float(vals.std()),
            "min": float(vals.min()),
            "p25": float(vals.quantile(0.25)),
            "median": float(vals.median()),
            "p75": float(vals.quantile(0.75)),
            "max": float(vals.max()),
        })
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(out_dir / "descriptive_statistics.csv", index=False)
    print(f"  Descriptive statistics: {len(desc_df)} variables")

    # Panel accounting table
    acct_rows = []
    for loc in sorted(pds_panel["location_abbr"].unique()):
        loc_data = pds_panel[pds_panel["location_abbr"] == loc]
        years_covered = sorted(loc_data["year"].astype(int).unique())
        eras_covered = sorted(loc_data["ppd_era"].dropna().unique())
        n_pche_comp = 0
        for q in pche_components_primary:
            if q in loc_data.columns and loc_data[q].notna().any():
                n_pche_comp += 1
        acct_rows.append({
            "location": loc,
            "n_obs": len(loc_data),
            "years_covered": ", ".join(str(y) for y in years_covered),
            "year_min": min(years_covered),
            "year_max": max(years_covered),
            "pche_components_available": n_pche_comp,
            "era_coverage": ", ".join(eras_covered),
        })

    acct_df = pd.DataFrame(acct_rows)
    acct_df.to_csv(out_dir / "panel_accounting.csv", index=False)
    print(f"  Panel accounting table: {len(acct_df)} locations")

    # Cluster counts for FE and CR models
    cluster_info = {
        "pds_panel_rows": len(pds_panel),
        "n_clusters_total": int(pds_panel["location_abbr"].nunique()),
        "cluster_sizes": pds_panel["location_abbr"].value_counts().describe().to_dict(),
    }
    with open(out_dir / "cluster_counts.json", "w") as f:
        json.dump(cluster_info, f, indent=2, default=str)
    print(f"  Cluster counts: {cluster_info['n_clusters_total']} clusters")

    # Bivariate correlation matrix
    corr_vars = ["outcome_ppd", "pche_index"]
    if "structural_access_index" in pds_panel.columns:
        corr_vars.append("structural_access_index")
    if "counseling_index" in pds_panel.columns:
        corr_vars.append("counseling_index")
    corr_vars += [q for q in RISK_ANCHORS if q in pds_panel.columns]
    corr_data = pds_panel[corr_vars].dropna()
    if len(corr_data) >= 10:
        corr_matrix = corr_data.corr()
        corr_matrix.to_csv(out_dir / "bivariate_correlations.csv")
        print(f"  Bivariate correlations saved (n={len(corr_data)})")

    # ===================================================================
    # PHASE 12B: FULL 12-COMPONENT SENSITIVITY INDEX
    # ===================================================================
    print("\n=== PHASE 12B: FULL 12-COMPONENT PCHE (SENSITIVITY ONLY) ===")

    pche_full_avail = [q for q in PCHE_FULL if q in pds_panel.columns
                       and pds_panel[q].notna().sum() >= MIN_N]
    print(f"  Full PCHE components available: {len(pche_full_avail)}")

    if len(pche_full_avail) > len(pche_components_primary):
        full_z_cols = {}
        for qid in pche_full_avail:
            full_z_cols[qid] = zscore(pds_panel[qid].values.astype(float))

        full_z_mat = np.column_stack([full_z_cols[q] for q in pche_full_avail])
        full_n_valid = np.sum(~np.isnan(full_z_mat), axis=1)
        pche_full_index = np.nanmean(full_z_mat, axis=1)
        pche_full_index[full_n_valid < 3] = np.nan

        n_full_valid = int(np.sum(~np.isnan(pche_full_index)))
        print(f"  Full PCHE index: {n_full_valid} valid rows")

        # Sensitivity: repeat primary risk anchors with full index
        full_sens_results = []
        for risk_qid in PRIMARY_RISK:
            if risk_qid not in pds_panel.columns:
                continue
            mask = (
                pds_panel["outcome_ppd"].notna() &
                pds_panel[risk_qid].notna() &
                ~np.isnan(pche_full_index)
            )
            work_idx = pds_panel.index[mask]
            if len(work_idx) < MIN_N:
                continue

            y = pds_panel.loc[work_idx, "outcome_ppd"].values
            risk_z = zscore(pds_panel.loc[work_idx, risk_qid].values)
            pche_z_full = zscore(pche_full_index[mask.values])

            m = fit_ols(y, [risk_z, pche_z_full, risk_z * pche_z_full],
                        col_names=["risk_z", "pche_z", "risk_x_pche"])
            if m:
                b, s, p = extract_coef(m, "risk_x_pche")
                print(f"  {risk_qid} full-12: interaction beta={b:.4f}, p={p:.4f}, n={m.n}")
                full_sens_results.append({
                    "risk_qid": risk_qid, "n_components": len(pche_full_avail),
                    "n": m.n, "interaction_beta": b, "interaction_p": p,
                    "adj_r2": m.adj_r2,
                })

        pd.DataFrame(full_sens_results).to_csv(out_dir / "full_pche_sensitivity.csv", index=False)
    else:
        print("  No additional components beyond primary set; skipping")

    # ===================================================================
    # PHASE 13: SUMMARY
    # ===================================================================
    print("\n=== PHASE 13: GENERATING SUMMARY ===")

    summary_lines = [
        "# PCHE Analysis Summary (v2 — IRB-corrected)",
        "",
        f"**Run ID**: {run_id}",
        f"**Timestamp**: {datetime.now(timezone.utc).isoformat()}",
        f"**Random Seed**: {seed}",
        f"**Panel SHA-256**: {panel_hash[:16]}...",
        "",
        "## Terminology",
        "- Outcome: PDS (Postpartum Depressive Symptoms)",
        "- Construct: PCHE (Pre-Conception Health Engagement)",
        "- Never use: PPD prevalence, PCHA, Pre-Conception Health Agency",
        "",
        "## Data",
        f"- Panel: {len(panel)} state-year rows x {panel.shape[1]} columns",
        f"- PDS panel: {len(pds_panel)} rows with outcome",
        f"- Locations: {n_locations}, Years: {year_range}",
        f"- Era breakdown: {pds_panel['ppd_era'].value_counts().to_dict()}",
        "",
        "## PCHE Index (PRIMARY: 7 common components)",
        f"- Components: {', '.join(pche_components_primary)}",
        f"- Valid rows: {n_pche_valid}",
        f"- Cronbach's alpha: {alpha:.3f}",
        f"- Mean: {np.nanmean(pche_index):.3f}, SD: {np.nanstd(pche_index):.3f}",
        "",
        "## FE-Only Baselines",
        f"- State-only FE R2: {fe_baseline['state_only_r2']:.4f}" if fe_baseline['state_only_r2'] else "- State-only FE R2: N/A",
        f"- State+Year FE R2: {fe_baseline['state_year_r2']:.4f}" if fe_baseline['state_year_r2'] else "- State+Year FE R2: N/A",
        f"- Incremental year R2: {fe_baseline['incremental_year_r2']:.4f}" if fe_baseline.get('incremental_year_r2') else "- Incremental year R2: N/A",
        "",
        "## Parallel Analysis",
        f"- Retained components: {n_retain}",
        "",
        "## Interaction Results",
    ]

    for _, row in int_df.iterrows():
        lines = [
            "",
            f"### {row['risk_qid']}: {row['risk_description']} ({row['anchor_type']})",
            f"- n = {row['n']}, clusters = {row['n_clusters']}",
        ]
        if row.get('interaction_beta') is not None:
            lines.append(f"- **OLS interaction beta: {row['interaction_beta']:.4f}**, parametric p={row['interaction_p_parametric']:.4f}")
        if row.get('cr1_interaction_p') is not None:
            lines.append(f"- CR1 p={row['cr1_interaction_p']:.4f}")
        if row.get('cr1_fe_interaction_p') is not None:
            lines.append(f"- CR1-FE p={row['cr1_fe_interaction_p']:.4f}")
        if row.get('wild_cluster_p') is not None:
            lines.append(f"- Wild cluster bootstrap p={row['wild_cluster_p']:.4f}")
        if row.get('cluster_permutation_p') is not None:
            lines.append(f"- Cluster permutation p={row['cluster_permutation_p']:.4f}")
        if row.get('cluster_boot_ci_lo') is not None:
            lines.append(f"- Cluster bootstrap 95% CI: [{row['cluster_boot_ci_lo']:.4f}, {row['cluster_boot_ci_hi']:.4f}]")
        if row.get('benefit_pp_at_plus1sd_risk') is not None:
            lines.append(f"- Benefit at +1 SD risk (PCHE -1 to +1): {row['benefit_pp_at_plus1sd_risk']:.2f} pp PDS reduction")
        lines.append(f"- Attenuation signature: {row['attenuation_signature']}")
        summary_lines.extend(lines)

    summary_text = "\n".join([l for l in summary_lines if l is not None])
    with open(out_dir / "analysis_summary.md", "w") as f:
        f.write(summary_text)

    # Save working panel
    pds_panel.to_csv(out_dir / "working_panel.csv", index=False)

    print(f"\n{'='*60}")
    print(f"PCHE ANALYSIS COMPLETE (v2 — IRB-corrected)")
    print(f"Run ID: {run_id}")
    print(f"Output: {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
