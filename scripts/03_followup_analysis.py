#!/usr/bin/env python3
"""
PCHE Follow-Up Analysis — Addressing Remaining IRB Items
=========================================================
Reads: data/panel_clean.csv (317 rows, 180 with PDS outcome, 35 locations)

Items implemented:
  1. Domain-specific sub-index analysis (Nutritional, Intentionality, Postpartum)
  2. PCHE with and without intentionality (QUO257)
  3. Strict pre-conception variants (8-item and 5-item)
  4. Strict pre-conception without intentionality (4-item)
  5. Era-specific analysis for strict variants
  6. LOSO for strict variants (QUO197 and QUO210)
  7. Question wording appendix CSV
  8. Variant comparison figure
  9. VIF for primary models
  10. Summary of all results

Terminology: PDS (postpartum depressive symptoms), PCHE (Pre-Conception Health
Engagement). Never PCHA, Agency, or PPD prevalence.

Usage:
    python scripts/03_followup_analysis.py [--panel PATH] [--run-id ID]

Outputs: output/<run_id>/  and  figures/<run_id>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_PERMUTATIONS = 5000
N_BOOTSTRAP = 5000
MIN_N = 30

# PDS outcome columns (harmonized across eras)
PDS_QUO_IDS = {"QUO74", "QUO219"}

# Primary PCHE — 7 common components (available for all 180 PDS rows)
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

# Full 12-component set (sensitivity — from script 02)
PCHE_FULL_12 = {
    **PCHE_COMMON,
    "QUO179": "Exercised 3+ days/week in 3 months before pregnancy",
    "QUO249": "Had teeth cleaned by dentist before pregnancy",
    "QUO75":  "Had teeth cleaned during pregnancy",
    "QUO16":  "Pregnancy was intended",
    "QUO296": "Got prenatal care as early as wanted (2000-2008)",
}

# Partner-stress risk anchors
RISK_ANCHORS = {
    "QUO197": "Partner argued more than usual (12 mo before birth)",
    "QUO210": "Any partner-related stressors",
    "QUO313": "Physical IPV by ex-partner (before pregnancy)",
    "QUO315": "Physical IPV by ex-partner (during pregnancy)",
}

# ---------------------------------------------------------------------------
# Domain sub-indices (Item 1)
# ---------------------------------------------------------------------------
DOMAIN_NUTRITIONAL = {"QUO41": "vitamins 4x/wk", "QUO65": "daily multivitamin"}
DOMAIN_INTENTIONALITY = {"QUO257": "trying to become pregnant"}
DOMAIN_POSTPARTUM = {
    "QUO4":   "breastfeeding 4wk",
    "QUO5":   "ever breastfed",
    "QUO44":  "breastfeeding 8wk",
    "QUO101": "baby checkup",
}

# ---------------------------------------------------------------------------
# PCHE without intentionality (Item 2) — 6 items
# ---------------------------------------------------------------------------
PCHE_NO_INTENT = {k: v for k, v in PCHE_COMMON.items() if k != "QUO257"}

# ---------------------------------------------------------------------------
# Strict pre-conception variants (Items 3 & 4)
# ---------------------------------------------------------------------------
# Strict pre-conception: 8 items from full 12, excluding postpartum
PCHE_STRICT_PRECONCEPTION = {
    "QUO179": "Exercised 3+ days/week in 3 months before pregnancy",
    "QUO41":  "Took multivitamins >4x/week month before pregnancy",
    "QUO65":  "Took daily multivitamin in month before pregnancy",
    "QUO249": "Had teeth cleaned by dentist before pregnancy",
    "QUO75":  "Had teeth cleaned during pregnancy",
    "QUO257": "Was trying to become pregnant",
    "QUO296": "Got prenatal care as early as wanted (2000-2008)",
    "QUO297": "Prenatal care began in first trimester (2009-2011)",
}

# Narrow pre-pregnancy: 5 items strictly before pregnancy
PCHE_NARROW_PREPREG = {
    "QUO179": "Exercised 3+ days/week in 3 months before pregnancy",
    "QUO41":  "Took multivitamins >4x/week month before pregnancy",
    "QUO65":  "Took daily multivitamin in month before pregnancy",
    "QUO249": "Had teeth cleaned by dentist before pregnancy",
    "QUO257": "Was trying to become pregnant",
}

# Narrow without intentionality: 4 items (Item 4)
PCHE_NARROW_NO_INTENT = {
    "QUO179": "Exercised 3+ days/week in 3 months before pregnancy",
    "QUO41":  "Took multivitamins >4x/week month before pregnancy",
    "QUO65":  "Took daily multivitamin in month before pregnancy",
    "QUO249": "Had teeth cleaned by dentist before pregnancy",
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
def fit_ols(y, X_cols, col_names=None, weights=None):
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

    return {
        "beta": beta, "se": se, "pvals": pvals, "n": n, "k": k,
        "r2": r2, "adj_r2": adj_r2, "rmse": rmse, "aic": aic, "bic": bic,
        "rss": rss, "tss": tss, "col_names": col_names, "XtX_inv": XtX_inv,
    }


def fit_ols_cluster_robust(y, X_cols, cluster_ids, col_names=None):
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

    return {
        "beta": beta, "se": se, "pvals": pvals, "n": n, "k": k,
        "r2": r2, "adj_r2": adj_r2, "rmse": rmse, "aic": aic, "bic": bic,
        "rss": rss, "tss": tss, "col_names": col_names,
    }


# ---------------------------------------------------------------------------
# Fixed-effects helpers
# ---------------------------------------------------------------------------
def make_fe_dummies(panel):
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
# Psychometric helpers
# ---------------------------------------------------------------------------
def compute_cronbach_alpha(df_items):
    """Cronbach's alpha on a DataFrame of items (rows=observations, cols=items)."""
    df_clean = df_items.dropna()
    if len(df_clean) < 10 or df_clean.shape[1] < 2:
        return float("nan"), len(df_clean), df_clean.shape[1]
    k = df_clean.shape[1]
    item_vars = df_clean.var(ddof=1).values
    total_var = df_clean.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan"), len(df_clean), k
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha), len(df_clean), k


def compute_vif(X_cols, col_names):
    """Compute VIF for each predictor column."""
    X = np.column_stack(X_cols)
    n, p = X.shape
    vifs = {}
    for j in range(p):
        y_j = X[:, j]
        others = [X[:, i] for i in range(p) if i != j]
        if not others:
            vifs[col_names[j]] = 1.0
            continue
        Xo = np.column_stack([np.ones(n)] + others)
        try:
            beta, *_ = np.linalg.lstsq(Xo, y_j, rcond=None)
        except np.linalg.LinAlgError:
            vifs[col_names[j]] = float("inf")
            continue
        resid = y_j - Xo @ beta
        rss = float(np.sum(resid ** 2))
        tss = float(np.sum((y_j - np.mean(y_j)) ** 2))
        if tss <= 0:
            vifs[col_names[j]] = float("inf")
            continue
        r2_j = 1.0 - rss / tss
        vifs[col_names[j]] = 1.0 / max(1.0 - r2_j, 1e-12)
    return vifs


# ---------------------------------------------------------------------------
# PCHE index builder
# ---------------------------------------------------------------------------
def build_pche_index(panel, component_dict, min_n=MIN_N, min_components=2):
    """Build a z-score composite index from a dict of QUO->description.

    Returns (index_array, available_qids, z_cols_dict).
    """
    avail = [q for q in component_dict if q in panel.columns
             and panel[q].notna().sum() >= min_n]
    z_cols = {}
    for qid in avail:
        z_cols[qid] = zscore(panel[qid].values.astype(float))
    if not z_cols:
        return np.full(len(panel), np.nan), avail, z_cols
    z_matrix = np.column_stack([z_cols[q] for q in avail])
    n_valid = np.sum(~np.isnan(z_matrix), axis=1)
    index = np.nanmean(z_matrix, axis=1)
    index[n_valid < min(min_components, len(avail))] = np.nan
    return index, avail, z_cols


# ---------------------------------------------------------------------------
# Extract helper
# ---------------------------------------------------------------------------
def extract_coef(model, name):
    """Extract beta, se, p for a named coefficient from a model dict."""
    if model is None:
        return None, None, None
    if name not in model["col_names"]:
        return None, None, None
    i = model["col_names"].index(name)
    return float(model["beta"][i]), float(model["se"][i]), float(model["pvals"][i])


# ---------------------------------------------------------------------------
# Cluster-aware permutation test (state blocks)
# ---------------------------------------------------------------------------
def cluster_permutation_test(y, risk_z, pche_z, cluster_ids, rng,
                             n_perm=N_PERMUTATIONS):
    """Permute PCHE across state blocks (all years for a state move together).

    Returns (observed_beta, permutation_p, perm_betas_array).
    """
    interaction = risk_z * pche_z
    m = fit_ols(y, [risk_z, pche_z, interaction],
                col_names=["risk_z", "pche_z", "risk_x_pche"])
    if m is None:
        return None, None, None

    int_idx = m["col_names"].index("risk_x_pche")
    obs_beta = float(m["beta"][int_idx])

    # Identify state blocks
    unique_states = np.unique(cluster_ids)
    block_indices = [np.where(cluster_ids == s)[0] for s in unique_states]

    perm_betas = np.zeros(n_perm)
    for i in range(n_perm):
        # Shuffle pche_z values by permuting state blocks
        perm_order = rng.permutation(len(unique_states))
        pche_perm = np.empty_like(pche_z)
        for orig_idx, new_idx in enumerate(perm_order):
            orig_block = block_indices[orig_idx]
            new_block = block_indices[new_idx]
            # Map values from new_block positions to orig_block positions
            if len(orig_block) == len(new_block):
                pche_perm[orig_block] = pche_z[new_block]
            else:
                # If blocks differ in size, use simple permutation fallback
                pche_perm[orig_block] = rng.choice(
                    pche_z[new_block], size=len(orig_block), replace=True
                )
        int_perm = risk_z * pche_perm
        m_perm = fit_ols(y, [risk_z, pche_perm, int_perm],
                         col_names=["risk_z", "pche_z", "interaction"])
        if m_perm:
            perm_betas[i] = float(m_perm["beta"][3])

    perm_p = float(np.mean(np.abs(perm_betas) >= np.abs(obs_beta)))
    return obs_beta, perm_p, perm_betas


# ---------------------------------------------------------------------------
# Cluster-aware bootstrap CI (state blocks)
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(y, risk_z, pche_z, cluster_ids, rng,
                         n_boot=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap CI by resampling entire state blocks.

    Returns (boot_lo, boot_hi, boot_betas_array).
    """
    interaction = risk_z * pche_z
    m = fit_ols(y, [risk_z, pche_z, interaction],
                col_names=["risk_z", "pche_z", "risk_x_pche"])
    if m is None:
        return None, None, None

    int_idx = m["col_names"].index("risk_x_pche")
    obs_beta = float(m["beta"][int_idx])

    unique_states = np.unique(cluster_ids)
    G = len(unique_states)
    block_indices = {s: np.where(cluster_ids == s)[0] for s in unique_states}

    boot_betas = np.zeros(n_boot)
    for i in range(n_boot):
        # Resample state blocks with replacement
        sampled_states = rng.choice(unique_states, size=G, replace=True)
        boot_idx = np.concatenate([block_indices[s] for s in sampled_states])
        y_b = y[boot_idx]
        risk_b = risk_z[boot_idx]
        pche_b = pche_z[boot_idx]
        int_b = risk_b * pche_b
        m_b = fit_ols(y_b, [risk_b, pche_b, int_b],
                      col_names=["risk_z", "pche_z", "interaction"])
        if m_b:
            boot_betas[i] = float(m_b["beta"][3])
        else:
            boot_betas[i] = obs_beta

    lo = float(np.percentile(boot_betas, 100 * alpha / 2))
    hi = float(np.percentile(boot_betas, 100 * (1 - alpha / 2)))
    return lo, hi, boot_betas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PCHE Follow-Up Analysis — IRB items"
    )
    parser.add_argument(
        "--panel",
        default=str(Path(__file__).resolve().parent.parent / "data" / "panel_clean.csv"),
    )
    parser.add_argument("--run-id", default="run_20260325T_followup")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    run_id = args.run_id
    seed = args.seed
    rng = np.random.RandomState(seed)

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "output" / run_id
    fig_dir = base_dir / "figures" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(args.panel)
    panel_hash = sha256_file(panel_path)
    panel = pd.read_csv(panel_path, low_memory=False)

    # Build PDS panel (rows with outcome)
    pds_panel = panel[panel["outcome_ppd"].notna()].copy()
    pds_panel["outcome_ppd"] = pds_panel["outcome_ppd"].astype(float)

    print(f"[PCHE followup] Run ID: {run_id}")
    print(f"[PCHE followup] Seed: {seed}")
    print(f"[PCHE followup] Panel: {len(panel)} rows, PDS panel: {len(pds_panel)} rows")
    print(f"[PCHE followup] SHA-256: {panel_hash[:16]}...")

    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "panel_sha256": panel_hash,
        "panel_rows": len(panel),
        "pds_panel_rows": len(pds_panel),
        "random_seed": seed,
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "purpose": "Follow-up addressing remaining IRB review items",
    }

    all_results = {}  # Collect for summary

    # ===================================================================
    # ITEM 1: DOMAIN-SPECIFIC SUB-INDEX ANALYSIS
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 1: DOMAIN-SPECIFIC SUB-INDEX ANALYSIS")
    print("=" * 70)

    domain_results = []
    domains = {
        "nutritional": DOMAIN_NUTRITIONAL,
        "intentionality": DOMAIN_INTENTIONALITY,
        "postpartum_engagement": DOMAIN_POSTPARTUM,
    }

    for domain_name, domain_dict in domains.items():
        print(f"\n--- Domain: {domain_name} ---")
        avail = [q for q in domain_dict if q in pds_panel.columns
                 and pds_panel[q].notna().sum() >= MIN_N]
        if not avail:
            print(f"  No available components, skipping")
            continue

        # Build domain sub-index
        if len(avail) == 1:
            # Single indicator: use raw z-score
            domain_z = zscore(pds_panel[avail[0]].values.astype(float))
            print(f"  Single indicator ({avail[0]}): using raw z-score")
        else:
            z_matrix = np.column_stack([
                zscore(pds_panel[q].values.astype(float)) for q in avail
            ])
            n_valid = np.sum(~np.isnan(z_matrix), axis=1)
            domain_z = np.nanmean(z_matrix, axis=1)
            domain_z[n_valid < 1] = np.nan
            print(f"  Components: {avail}, composite z-score")

        pds_panel[f"domain_{domain_name}_z"] = domain_z

        for risk_qid in ["QUO197", "QUO210"]:
            if risk_qid not in pds_panel.columns:
                continue
            mask = (pds_panel["outcome_ppd"].notna() &
                    pds_panel[risk_qid].notna() &
                    pd.notna(domain_z))
            work = pds_panel[mask].copy()
            if len(work) < MIN_N:
                print(f"  {risk_qid}: n={len(work)} < {MIN_N}, skip")
                continue

            y = work["outcome_ppd"].values
            risk_z = zscore(work[risk_qid].values)
            mod_z = zscore(work[f"domain_{domain_name}_z"].values)
            interaction = risk_z * mod_z

            # OLS
            m_ols = fit_ols(y, [risk_z, mod_z, interaction],
                            col_names=["risk_z", "domain_z", "risk_x_domain"])

            # State+year FE
            fe_d, fe_n = make_fe_dummies(work)
            m_fe = fit_ols(y, fe_d + [risk_z, mod_z, interaction],
                           col_names=fe_n + ["risk_z", "domain_z", "risk_x_domain"])

            ols_b, ols_se, ols_p = extract_coef(m_ols, "risk_x_domain")
            fe_b, fe_se, fe_p = extract_coef(m_fe, "risk_x_domain")

            print(f"  {risk_qid} x {domain_name} (n={len(work)}):")
            if ols_b is not None:
                print(f"    OLS: beta={ols_b:.4f}, se={ols_se:.4f}, p={ols_p:.4f}")
            if fe_b is not None:
                print(f"    FE:  beta={fe_b:.4f}, se={fe_se:.4f}, p={fe_p:.4f}")

            domain_results.append({
                "domain": domain_name,
                "components": ", ".join(avail),
                "n_components": len(avail),
                "risk_qid": risk_qid,
                "n": len(work),
                "ols_interaction_beta": ols_b,
                "ols_interaction_se": ols_se,
                "ols_interaction_p": ols_p,
                "ols_adj_r2": m_ols["adj_r2"] if m_ols else None,
                "fe_interaction_beta": fe_b,
                "fe_interaction_se": fe_se,
                "fe_interaction_p": fe_p,
                "fe_adj_r2": m_fe["adj_r2"] if m_fe else None,
            })

    domain_df = pd.DataFrame(domain_results)
    domain_df.to_csv(out_dir / "domain_subindex_analysis.csv", index=False)
    all_results["item1_domain"] = domain_results

    # ===================================================================
    # ITEM 2: PCHE WITH AND WITHOUT INTENTIONALITY
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 2: PCHE WITH AND WITHOUT INTENTIONALITY (QUO257)")
    print("=" * 70)

    # Build both indices
    pche_7_idx, pche_7_avail, pche_7_zcols = build_pche_index(
        pds_panel, PCHE_COMMON, min_components=2
    )
    pds_panel["pche_7"] = pche_7_idx

    pche_no_int_idx, pche_no_int_avail, pche_no_int_zcols = build_pche_index(
        pds_panel, PCHE_NO_INTENT, min_components=2
    )
    pds_panel["pche_no_intent"] = pche_no_int_idx

    # Cronbach's alpha for 6-item version
    no_int_z_df = pd.DataFrame(pche_no_int_zcols, index=pds_panel.index)
    alpha_6, n_alpha_6, k_alpha_6 = compute_cronbach_alpha(no_int_z_df)
    print(f"  Cronbach's alpha (6-item, without intentionality): {alpha_6:.4f} "
          f"(k={k_alpha_6}, n={n_alpha_6})")

    # Also compute alpha for 7-item for comparison
    full_z_df = pd.DataFrame(pche_7_zcols, index=pds_panel.index)
    alpha_7, n_alpha_7, k_alpha_7 = compute_cronbach_alpha(full_z_df)
    print(f"  Cronbach's alpha (7-item, primary PCHE): {alpha_7:.4f} "
          f"(k={k_alpha_7}, n={n_alpha_7})")

    intent_results = []
    for risk_qid in ["QUO197", "QUO210"]:
        if risk_qid not in pds_panel.columns:
            continue
        for label, col in [("pche_7_primary", "pche_7"),
                           ("pche_6_no_intent", "pche_no_intent")]:
            mask = (pds_panel["outcome_ppd"].notna() &
                    pds_panel[risk_qid].notna() &
                    pds_panel[col].notna())
            work = pds_panel[mask].copy()
            if len(work) < MIN_N:
                continue

            y = work["outcome_ppd"].values
            risk_z = zscore(work[risk_qid].values)
            pche_z = zscore(work[col].values)
            interaction = risk_z * pche_z

            m_ols = fit_ols(y, [risk_z, pche_z, interaction],
                            col_names=["risk_z", "pche_z", "risk_x_pche"])

            fe_d, fe_n = make_fe_dummies(work)
            m_fe = fit_ols(y, fe_d + [risk_z, pche_z, interaction],
                           col_names=fe_n + ["risk_z", "pche_z", "risk_x_pche"])

            ols_b, ols_se, ols_p = extract_coef(m_ols, "risk_x_pche")
            fe_b, fe_se, fe_p = extract_coef(m_fe, "risk_x_pche")

            print(f"  {risk_qid} x {label} (n={len(work)}):")
            if ols_b is not None:
                print(f"    OLS: beta={ols_b:.4f}, se={ols_se:.4f}, p={ols_p:.4f}")
            if fe_b is not None:
                print(f"    FE:  beta={fe_b:.4f}, se={fe_se:.4f}, p={fe_p:.4f}")

            intent_results.append({
                "pche_variant": label,
                "risk_qid": risk_qid,
                "n": len(work),
                "ols_interaction_beta": ols_b,
                "ols_interaction_se": ols_se,
                "ols_interaction_p": ols_p,
                "ols_adj_r2": m_ols["adj_r2"] if m_ols else None,
                "fe_interaction_beta": fe_b,
                "fe_interaction_se": fe_se,
                "fe_interaction_p": fe_p,
                "fe_adj_r2": m_fe["adj_r2"] if m_fe else None,
            })

    intent_df = pd.DataFrame(intent_results)
    intent_df.to_csv(out_dir / "intentionality_sensitivity.csv", index=False)
    all_results["item2_intent"] = {
        "alpha_7_primary": alpha_7,
        "alpha_6_no_intent": alpha_6,
        "n_alpha_7": n_alpha_7,
        "n_alpha_6": n_alpha_6,
        "models": intent_results,
    }

    # ===================================================================
    # ITEM 3: STRICT PRE-CONCEPTION VARIANTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 3: STRICT PRE-CONCEPTION PCHE VARIANTS")
    print("=" * 70)

    strict_results = []
    strict_variants = [
        ("strict_preconception_8", PCHE_STRICT_PRECONCEPTION),
        ("narrow_prepreg_5", PCHE_NARROW_PREPREG),
    ]

    for label, component_dict in strict_variants:
        print(f"\n--- {label} ---")
        idx, avail, z_cols = build_pche_index(pds_panel, component_dict, min_components=2)
        pds_panel[f"pche_{label}"] = idx
        n_valid = int(np.sum(~np.isnan(idx)))
        print(f"  Components available: {avail}")
        print(f"  Valid rows: {n_valid}")

        # Cronbach's alpha
        if len(z_cols) >= 2:
            z_df_sub = pd.DataFrame(z_cols, index=pds_panel.index)
            alpha_sub, n_sub, k_sub = compute_cronbach_alpha(z_df_sub)
            print(f"  Cronbach's alpha: {alpha_sub:.4f} (k={k_sub}, n={n_sub})")
        else:
            alpha_sub = float("nan")

        if n_valid < MIN_N:
            print(f"  SKIP: insufficient valid rows ({n_valid} < {MIN_N})")
            continue

        for risk_qid in ["QUO197", "QUO210"]:
            if risk_qid not in pds_panel.columns:
                continue
            mask = (pds_panel["outcome_ppd"].notna() &
                    pds_panel[risk_qid].notna() &
                    pds_panel[f"pche_{label}"].notna())
            work = pds_panel[mask].copy()
            if len(work) < MIN_N:
                print(f"  {risk_qid}: n={len(work)} < {MIN_N}, skip")
                continue

            y = work["outcome_ppd"].values
            risk_z = zscore(work[risk_qid].values)
            pche_z = zscore(work[f"pche_{label}"].values)
            interaction = risk_z * pche_z
            cluster_ids = work["location_abbr"].values

            # --- OLS interaction ---
            m_ols = fit_ols(y, [risk_z, pche_z, interaction],
                            col_names=["risk_z", "pche_z", "risk_x_pche"])
            ols_b, ols_se, ols_p = extract_coef(m_ols, "risk_x_pche")

            # --- State+year FE interaction ---
            fe_d, fe_n = make_fe_dummies(work)
            m_fe = fit_ols(y, fe_d + [risk_z, pche_z, interaction],
                           col_names=fe_n + ["risk_z", "pche_z", "risk_x_pche"])
            fe_b, fe_se, fe_p = extract_coef(m_fe, "risk_x_pche")

            # --- Cluster-robust CR1 ---
            m_cr = fit_ols_cluster_robust(
                y, fe_d + [risk_z, pche_z, interaction],
                cluster_ids=cluster_ids,
                col_names=fe_n + ["risk_z", "pche_z", "risk_x_pche"],
            )
            cr_b, cr_se, cr_p = extract_coef(m_cr, "risk_x_pche")

            # --- Cluster-aware permutation (state blocks) ---
            print(f"  {risk_qid} x {label}: running {N_PERMUTATIONS} cluster permutations...")
            obs_beta_perm, perm_p, _ = cluster_permutation_test(
                y, risk_z, pche_z, cluster_ids, rng, n_perm=N_PERMUTATIONS
            )

            # --- Cluster-aware bootstrap CI (state blocks) ---
            print(f"  {risk_qid} x {label}: running {N_BOOTSTRAP} cluster bootstrap...")
            boot_lo, boot_hi, _ = cluster_bootstrap_ci(
                y, risk_z, pche_z, cluster_ids, rng, n_boot=N_BOOTSTRAP
            )

            print(f"  {risk_qid} x {label} (n={len(work)}):")
            if ols_b is not None:
                print(f"    OLS: beta={ols_b:.4f}, p={ols_p:.4f}")
            if perm_p is not None:
                print(f"    Cluster perm p={perm_p:.4f}")
            if boot_lo is not None:
                print(f"    Cluster boot CI=[{boot_lo:.4f}, {boot_hi:.4f}]")
            if fe_b is not None:
                print(f"    FE: beta={fe_b:.4f}, p={fe_p:.4f}")
            if cr_b is not None:
                print(f"    CR1: beta={cr_b:.4f}, p={cr_p:.4f}")

            strict_results.append({
                "pche_variant": label,
                "n_components": len(avail),
                "components": ", ".join(avail),
                "cronbach_alpha": alpha_sub,
                "risk_qid": risk_qid,
                "n": len(work),
                "ols_interaction_beta": ols_b,
                "ols_interaction_se": ols_se,
                "ols_interaction_p": ols_p,
                "ols_adj_r2": m_ols["adj_r2"] if m_ols else None,
                "cluster_permutation_p": perm_p,
                "cluster_bootstrap_ci_lo": boot_lo,
                "cluster_bootstrap_ci_hi": boot_hi,
                "fe_interaction_beta": fe_b,
                "fe_interaction_se": fe_se,
                "fe_interaction_p": fe_p,
                "fe_adj_r2": m_fe["adj_r2"] if m_fe else None,
                "cr1_interaction_beta": cr_b,
                "cr1_interaction_se": cr_se,
                "cr1_interaction_p": cr_p,
            })

    strict_df = pd.DataFrame(strict_results)
    strict_df.to_csv(out_dir / "strict_preconception_sensitivity.csv", index=False)
    all_results["item3_strict"] = strict_results

    # ===================================================================
    # ITEM 4: STRICT PRE-CONCEPTION WITHOUT INTENTIONALITY
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 4: NARROW PRE-PREGNANCY WITHOUT INTENTIONALITY (4-item)")
    print("=" * 70)

    narrow_no_int_results = []
    idx_nni, avail_nni, z_cols_nni = build_pche_index(
        pds_panel, PCHE_NARROW_NO_INTENT, min_components=2
    )
    pds_panel["pche_narrow_no_intent"] = idx_nni
    n_valid_nni = int(np.sum(~np.isnan(idx_nni)))
    print(f"  Components: {avail_nni}")
    print(f"  Valid rows: {n_valid_nni}")

    if len(z_cols_nni) >= 2:
        z_df_nni = pd.DataFrame(z_cols_nni, index=pds_panel.index)
        alpha_nni, n_alpha_nni, k_alpha_nni = compute_cronbach_alpha(z_df_nni)
        print(f"  Cronbach's alpha: {alpha_nni:.4f} (k={k_alpha_nni}, n={n_alpha_nni})")
    else:
        alpha_nni = float("nan")

    if n_valid_nni >= MIN_N:
        for risk_qid in ["QUO197", "QUO210"]:
            if risk_qid not in pds_panel.columns:
                continue
            mask = (pds_panel["outcome_ppd"].notna() &
                    pds_panel[risk_qid].notna() &
                    pds_panel["pche_narrow_no_intent"].notna())
            work = pds_panel[mask].copy()
            if len(work) < MIN_N:
                print(f"  {risk_qid}: n={len(work)} < {MIN_N}, skip")
                continue

            y = work["outcome_ppd"].values
            risk_z = zscore(work[risk_qid].values)
            pche_z = zscore(work["pche_narrow_no_intent"].values)
            interaction = risk_z * pche_z
            cluster_ids = work["location_abbr"].values

            m_ols = fit_ols(y, [risk_z, pche_z, interaction],
                            col_names=["risk_z", "pche_z", "risk_x_pche"])
            ols_b, ols_se, ols_p = extract_coef(m_ols, "risk_x_pche")

            fe_d, fe_n = make_fe_dummies(work)
            m_fe = fit_ols(y, fe_d + [risk_z, pche_z, interaction],
                           col_names=fe_n + ["risk_z", "pche_z", "risk_x_pche"])
            fe_b, fe_se, fe_p = extract_coef(m_fe, "risk_x_pche")

            m_cr = fit_ols_cluster_robust(
                y, fe_d + [risk_z, pche_z, interaction],
                cluster_ids=cluster_ids,
                col_names=fe_n + ["risk_z", "pche_z", "risk_x_pche"],
            )
            cr_b, cr_se, cr_p = extract_coef(m_cr, "risk_x_pche")

            print(f"  {risk_qid} x narrow_no_intent: running cluster permutation...")
            obs_beta_perm, perm_p, _ = cluster_permutation_test(
                y, risk_z, pche_z, cluster_ids, rng, n_perm=N_PERMUTATIONS
            )
            print(f"  {risk_qid} x narrow_no_intent: running cluster bootstrap...")
            boot_lo, boot_hi, _ = cluster_bootstrap_ci(
                y, risk_z, pche_z, cluster_ids, rng, n_boot=N_BOOTSTRAP
            )

            print(f"  {risk_qid} x narrow_no_intent (n={len(work)}):")
            if ols_b is not None:
                print(f"    OLS: beta={ols_b:.4f}, p={ols_p:.4f}")
            if perm_p is not None:
                print(f"    Cluster perm p={perm_p:.4f}")
            if boot_lo is not None:
                print(f"    Cluster boot CI=[{boot_lo:.4f}, {boot_hi:.4f}]")

            narrow_no_int_results.append({
                "pche_variant": "narrow_no_intent_4",
                "n_components": len(avail_nni),
                "components": ", ".join(avail_nni),
                "cronbach_alpha": alpha_nni,
                "risk_qid": risk_qid,
                "n": len(work),
                "ols_interaction_beta": ols_b,
                "ols_interaction_se": ols_se,
                "ols_interaction_p": ols_p,
                "ols_adj_r2": m_ols["adj_r2"] if m_ols else None,
                "cluster_permutation_p": perm_p,
                "cluster_bootstrap_ci_lo": boot_lo,
                "cluster_bootstrap_ci_hi": boot_hi,
                "fe_interaction_beta": fe_b,
                "fe_interaction_se": fe_se,
                "fe_interaction_p": fe_p,
                "fe_adj_r2": m_fe["adj_r2"] if m_fe else None,
                "cr1_interaction_beta": cr_b,
                "cr1_interaction_se": cr_se,
                "cr1_interaction_p": cr_p,
            })

    narrow_no_int_df = pd.DataFrame(narrow_no_int_results)
    narrow_no_int_df.to_csv(out_dir / "narrow_no_intent_sensitivity.csv", index=False)
    all_results["item4_narrow_no_intent"] = narrow_no_int_results

    # ===================================================================
    # ITEM 5: ERA-SPECIFIC FOR STRICT VARIANTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 5: ERA-SPECIFIC ANALYSIS FOR STRICT PCHE VARIANTS")
    print("=" * 70)

    era_strict_results = []
    all_strict_variants = [
        ("strict_preconception_8", PCHE_STRICT_PRECONCEPTION),
        ("narrow_prepreg_5", PCHE_NARROW_PREPREG),
        ("narrow_no_intent_4", PCHE_NARROW_NO_INTENT),
    ]

    # Determine era column
    era_col = "ppd_era" if "ppd_era" in pds_panel.columns else None
    if era_col is None:
        # Build from year
        pds_panel["ppd_era"] = pds_panel["year"].apply(
            lambda yr: "2004-2008" if yr <= 2008 else "2009-2011"
        )
        era_col = "ppd_era"

    for label, component_dict in all_strict_variants:
        for era in sorted(pds_panel[era_col].unique()):
            era_panel = pds_panel[pds_panel[era_col] == era].copy()
            era_idx, era_avail, _ = build_pche_index(era_panel, component_dict, min_components=2)
            era_panel[f"pche_{label}"] = era_idx

            for risk_qid in ["QUO197", "QUO210"]:
                if risk_qid not in era_panel.columns:
                    continue
                mask = (era_panel["outcome_ppd"].notna() &
                        era_panel[risk_qid].notna() &
                        era_panel[f"pche_{label}"].notna())
                work = era_panel[mask].copy()
                if len(work) < MIN_N:
                    print(f"  {era} {label} x {risk_qid}: n={len(work)} < {MIN_N}, skip")
                    continue

                y = work["outcome_ppd"].values
                risk_z = zscore(work[risk_qid].values)
                pche_z = zscore(work[f"pche_{label}"].values)

                m = fit_ols(y, [risk_z, pche_z, risk_z * pche_z],
                            col_names=["risk_z", "pche_z", "risk_x_pche"])
                if m:
                    b, se, p = extract_coef(m, "risk_x_pche")
                    print(f"  {era} {label} x {risk_qid}: beta={b:.4f}, p={p:.4f}, n={m['n']}")
                    era_strict_results.append({
                        "era": era,
                        "pche_variant": label,
                        "risk_qid": risk_qid,
                        "n": m["n"],
                        "interaction_beta": b,
                        "interaction_se": se,
                        "interaction_p": p,
                        "adj_r2": m["adj_r2"],
                    })
                else:
                    print(f"  {era} {label} x {risk_qid}: OLS failed")

    era_strict_df = pd.DataFrame(era_strict_results)
    era_strict_df.to_csv(out_dir / "era_strict_variants.csv", index=False)
    all_results["item5_era"] = era_strict_results

    # ===================================================================
    # ITEM 6: LOSO FOR STRICT VARIANTS (QUO197 and QUO210 only)
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 6: LOSO FOR STRICT PCHE VARIANTS")
    print("=" * 70)

    loso_results = []
    for label, component_dict in all_strict_variants:
        idx, avail, _ = build_pche_index(pds_panel, component_dict, min_components=2)
        pds_panel[f"pche_{label}"] = idx

        for risk_qid in ["QUO197", "QUO210"]:
            if risk_qid not in pds_panel.columns:
                continue
            mask = (pds_panel["outcome_ppd"].notna() &
                    pds_panel[risk_qid].notna() &
                    pds_panel[f"pche_{label}"].notna())
            work = pds_panel[mask].copy()
            if len(work) < MIN_N:
                continue

            states = work["location_abbr"].unique()
            for st in states:
                sub = work[work["location_abbr"] != st]
                if len(sub) < MIN_N:
                    continue
                y = sub["outcome_ppd"].values
                rz = zscore(sub[risk_qid].values)
                pz = zscore(sub[f"pche_{label}"].values)
                m = fit_ols(y, [rz, pz, rz * pz],
                            col_names=["risk_z", "pche_z", "risk_x_pche"])
                if m:
                    b, se, p = extract_coef(m, "risk_x_pche")
                    loso_results.append({
                        "pche_variant": label,
                        "risk_qid": risk_qid,
                        "excluded_state": st,
                        "n": m["n"],
                        "interaction_beta": b,
                        "interaction_se": se,
                        "interaction_p": p,
                    })

    loso_df = pd.DataFrame(loso_results)
    loso_df.to_csv(out_dir / "loso_strict_variants.csv", index=False)

    if len(loso_df) > 0:
        for variant in loso_df["pche_variant"].unique():
            for rq in loso_df["risk_qid"].unique():
                sub = loso_df[(loso_df["pche_variant"] == variant) &
                              (loso_df["risk_qid"] == rq)]
                if len(sub) == 0:
                    continue
                betas = sub["interaction_beta"].values
                print(f"  {variant} x {rq} LOSO: range=[{betas.min():.4f}, {betas.max():.4f}], "
                      f"mean={betas.mean():.4f}, SD={betas.std():.4f}")
    all_results["item6_loso"] = "loso_strict_variants.csv"

    # ===================================================================
    # ITEM 7: QUESTION WORDING APPENDIX
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 7: QUESTION WORDING APPENDIX")
    print("=" * 70)

    QUESTION_WORDING = {
        "QUO4": {
            "exact_prams_wording": "Indicator of whether mother was still breastfeeding 4 weeks after delivery",
            "domain": "postpartum_engagement",
            "timing": "postpartum",
        },
        "QUO5": {
            "exact_prams_wording": "Did you ever breastfeed or pump breast milk to feed your new baby after delivery?",
            "domain": "postpartum_engagement",
            "timing": "postpartum",
        },
        "QUO41": {
            "exact_prams_wording": "Indicator of whether mother took vitamins more than 4 times a week during the month prior to pregnancy",
            "domain": "nutritional",
            "timing": "pre-pregnancy",
        },
        "QUO44": {
            "exact_prams_wording": "Indicator of whether mother was still breastfeeding 8 weeks after delivery",
            "domain": "postpartum_engagement",
            "timing": "postpartum",
        },
        "QUO65": {
            "exact_prams_wording": "During the month before you got pregnant with your new baby, did you take a daily multivitamin?",
            "domain": "nutritional",
            "timing": "pre-pregnancy",
        },
        "QUO75": {
            "exact_prams_wording": "During your most recent pregnancy, did you have your teeth cleaned?",
            "domain": "preventive_health",
            "timing": "prenatal",
        },
        "QUO101": {
            "exact_prams_wording": "Was your baby seen by a doctor, nurse or other health care provider in the first week after leaving the hospital?",
            "domain": "postpartum_engagement",
            "timing": "postpartum",
        },
        "QUO179": {
            "exact_prams_wording": "Indicator of pre-pregnancy exercise 3 or more days a week",
            "domain": "physical_activity",
            "timing": "pre-pregnancy",
        },
        "QUO249": {
            "exact_prams_wording": "Indicator of mother having her teeth cleaned in 12 months prior to pregnancy (years 2009-2011)",
            "domain": "preventive_health",
            "timing": "pre-pregnancy",
        },
        "QUO257": {
            "exact_prams_wording": "When you got pregnant with your new baby were you trying to become pregnant?",
            "domain": "intentionality",
            "timing": "pre-pregnancy",
        },
        "QUO296": {
            "exact_prams_wording": "Did you get prenatal care as early in your pregnancy as you wanted? (years 2000-2008)",
            "domain": "prenatal_care",
            "timing": "prenatal",
        },
        "QUO297": {
            "exact_prams_wording": "Did you get prenatal care as early in your pregnancy as you wanted? (years 2009-2011)",
            "domain": "prenatal_care",
            "timing": "prenatal",
        },
        "QUO74": {
            "exact_prams_wording": "Indicator of whether mother reported frequent postpartum depressive symptoms (years 2004-2008)",
            "domain": "outcome_PDS",
            "timing": "postpartum",
        },
        "QUO219": {
            "exact_prams_wording": "Indicator of whether mother reported frequent postpartum depressive symptoms (years 2009-2011)",
            "domain": "outcome_PDS",
            "timing": "postpartum",
        },
        "QUO197": {
            "exact_prams_wording": "In the 12 months before your baby was born, you argued with your husband or partner more than usual",
            "domain": "risk_anchor_partner_stress",
            "timing": "12 months before birth",
        },
        "QUO210": {
            "exact_prams_wording": "Indicator of any partner-related stressors reported",
            "domain": "risk_anchor_partner_stress",
            "timing": "12 months before birth",
        },
        "QUO313": {
            "exact_prams_wording": "Physical IPV by ex-partner (before pregnancy)",
            "domain": "risk_anchor_IPV",
            "timing": "before pregnancy",
        },
        "QUO315": {
            "exact_prams_wording": "Physical IPV by ex-partner (during pregnancy)",
            "domain": "risk_anchor_IPV",
            "timing": "during pregnancy",
        },
        "QUO16": {
            "exact_prams_wording": "Pregnancy was intended",
            "domain": "intentionality",
            "timing": "pre-pregnancy",
        },
    }

    # Determine which sets each question belongs to
    primary_pche_set = set(PCHE_COMMON.keys())
    strict_pche_set = set(PCHE_STRICT_PRECONCEPTION.keys())
    narrow_pche_set = set(PCHE_NARROW_PREPREG.keys())

    wording_rows = []
    for qid, info in sorted(QUESTION_WORDING.items()):
        # Check years available from panel
        if qid in panel.columns:
            valid_years = sorted(panel.loc[panel[qid].notna(), "year"].unique())
            years_str = ", ".join(str(int(y)) for y in valid_years) if len(valid_years) > 0 else "none"
        else:
            years_str = "not in panel"

        wording_rows.append({
            "question_id": qid,
            "domain": info["domain"],
            "timing": info["timing"],
            "exact_prams_wording": info["exact_prams_wording"],
            "years_available": years_str,
            "in_primary_pche": qid in primary_pche_set,
            "in_strict_pche": qid in strict_pche_set,
            "in_narrow_pche": qid in narrow_pche_set,
        })

    wording_df = pd.DataFrame(wording_rows)
    wording_df.to_csv(out_dir / "question_wording_appendix.csv", index=False)
    print(f"  Saved question wording appendix: {len(wording_rows)} questions")
    all_results["item7_wording"] = "question_wording_appendix.csv"

    # ===================================================================
    # ITEM 8: VARIANT COMPARISON FIGURE
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 8: VARIANT COMPARISON FIGURE")
    print("=" * 70)

    # Build all variant indices and run OLS for comparison
    variant_defs = [
        ("Primary 7-component", PCHE_COMMON),
        ("Full 12-component", PCHE_FULL_12),
        ("Without intent (6)", PCHE_NO_INTENT),
        ("Strict pre-con (8)", PCHE_STRICT_PRECONCEPTION),
        ("Narrow pre-preg (5)", PCHE_NARROW_PREPREG),
        ("Narrow no intent (4)", PCHE_NARROW_NO_INTENT),
    ]

    comparison_rows = []
    for var_label, comp_dict in variant_defs:
        idx_v, avail_v, _ = build_pche_index(pds_panel, comp_dict, min_components=2)

        for risk_qid in ["QUO197", "QUO210"]:
            if risk_qid not in pds_panel.columns:
                continue
            mask = (pds_panel["outcome_ppd"].notna() &
                    pds_panel[risk_qid].notna() &
                    pd.notna(idx_v))
            work_idx = np.where(mask.values)[0]
            if len(work_idx) < MIN_N:
                continue

            y = pds_panel.iloc[work_idx]["outcome_ppd"].values
            risk_z = zscore(pds_panel.iloc[work_idx][risk_qid].values)
            pche_z = zscore(idx_v[work_idx])
            interaction = risk_z * pche_z
            cluster_ids = pds_panel.iloc[work_idx]["location_abbr"].values

            m = fit_ols(y, [risk_z, pche_z, interaction],
                        col_names=["risk_z", "pche_z", "risk_x_pche"])
            if m is None:
                continue

            b, se, p = extract_coef(m, "risk_x_pche")

            # Quick cluster bootstrap for CI
            boot_lo, boot_hi, _ = cluster_bootstrap_ci(
                y, risk_z, pche_z, cluster_ids, rng, n_boot=N_BOOTSTRAP
            )

            # Quick cluster permutation for p
            _, perm_p, _ = cluster_permutation_test(
                y, risk_z, pche_z, cluster_ids, rng, n_perm=N_PERMUTATIONS
            )

            comparison_rows.append({
                "variant": var_label,
                "n_components": len(avail_v),
                "risk_qid": risk_qid,
                "n": m["n"],
                "ols_interaction_beta": b,
                "ols_interaction_se": se,
                "ols_interaction_p": p,
                "cluster_perm_p": perm_p,
                "cluster_boot_ci_lo": boot_lo,
                "cluster_boot_ci_hi": boot_hi,
                "adj_r2": m["adj_r2"],
            })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(out_dir / "variant_comparison.csv", index=False)

    # Generate the figure
    if len(comparison_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        plt.rcParams.update({"font.family": "serif", "font.size": 11})

        for ax_idx, risk_qid in enumerate(["QUO197", "QUO210"]):
            ax = axes[ax_idx]
            sub = comparison_df[comparison_df["risk_qid"] == risk_qid].copy()
            sub = sub[sub["ols_interaction_beta"].notna()]
            if len(sub) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            variants = sub["variant"].values
            betas = sub["ols_interaction_beta"].values
            ci_lo = sub["cluster_boot_ci_lo"].fillna(0).values
            ci_hi = sub["cluster_boot_ci_hi"].fillna(0).values
            perm_ps = sub["cluster_perm_p"].fillna(1).values

            y_pos = np.arange(len(variants))
            colors = []
            for p_val in perm_ps:
                if p_val < 0.05:
                    colors.append("#4C72B0")
                elif p_val < 0.10:
                    colors.append("#55A868")
                else:
                    colors.append("#8C8C8C")

            ax.barh(y_pos, betas, color=colors, height=0.55, edgecolor="white", alpha=0.85)
            ax.errorbar(betas, y_pos,
                        xerr=[betas - ci_lo, ci_hi - betas],
                        fmt="none", ecolor="black", capsize=3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(variants, fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Interaction Beta (Risk x PCHE)")
            risk_label = RISK_ANCHORS.get(risk_qid, risk_qid)
            ax.set_title(f"{risk_qid}: PCHE Variant Comparison\n({risk_label})",
                         fontsize=10)

            # Annotate p-values
            for i, (b_val, p_val) in enumerate(zip(betas, perm_ps)):
                if np.isfinite(p_val) and p_val < 1:
                    label = f"p={p_val:.3f}"
                    offset = 0.001 if b_val < 0 else -0.001
                    ha = "left" if b_val < 0 else "right"
                    ax.text(b_val + offset, i, label,
                            va="center", ha=ha, fontsize=7.5)

        fig.suptitle("PCHE Variant Comparison: Interaction with Partner Stress on PDS",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        fig.savefig(fig_dir / "fig_variant_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  Figure: variant comparison saved")
    else:
        print("  No comparison data available for figure")

    all_results["item8_comparison"] = "variant_comparison.csv"

    # ===================================================================
    # ITEM 9: VIF FOR PRIMARY MODELS
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 9: VIF FOR PRIMARY MODELS")
    print("=" * 70)

    vif_results = []
    pche_7_idx, _, _ = build_pche_index(pds_panel, PCHE_COMMON, min_components=2)
    pds_panel["pche_7"] = pche_7_idx

    for risk_qid in ["QUO197", "QUO210", "QUO313", "QUO315"]:
        if risk_qid not in pds_panel.columns:
            continue
        mask = (pds_panel["outcome_ppd"].notna() &
                pds_panel[risk_qid].notna() &
                pds_panel["pche_7"].notna())
        work = pds_panel[mask].copy()
        if len(work) < MIN_N:
            continue

        risk_z = zscore(work[risk_qid].values)
        pche_z = zscore(work["pche_7"].values)
        interaction = risk_z * pche_z

        vifs = compute_vif([risk_z, pche_z, interaction],
                           ["risk_z", "pche_z", "risk_x_pche"])

        print(f"  {risk_qid}:")
        for name, vif_val in vifs.items():
            flag = " (HIGH)" if vif_val > 5 else ""
            print(f"    VIF({name}) = {vif_val:.2f}{flag}")
            vif_results.append({
                "risk_qid": risk_qid,
                "variable": name,
                "vif": vif_val,
            })

    vif_df = pd.DataFrame(vif_results)
    vif_df.to_csv(out_dir / "vif_primary_models.csv", index=False)
    all_results["item9_vif"] = vif_results

    # ===================================================================
    # ITEM 10: SUMMARY
    # ===================================================================
    print("\n" + "=" * 70)
    print("ITEM 10: GENERATING SUMMARY")
    print("=" * 70)

    summary_lines = [
        "# PCHE Follow-Up Analysis Summary",
        "",
        f"**Run ID**: {run_id}",
        f"**Timestamp**: {datetime.now(timezone.utc).isoformat()}",
        f"**Seed**: {seed}",
        f"**Panel SHA-256**: {panel_hash[:16]}...",
        f"**Panel rows**: {len(panel)}, PDS panel: {len(pds_panel)}",
        f"**N_PERMUTATIONS**: {N_PERMUTATIONS}, **N_BOOTSTRAP**: {N_BOOTSTRAP}",
        "",
    ]

    # Item 1: Domain sub-index
    summary_lines.extend([
        "## Item 1: Domain-Specific Sub-Index Analysis",
        "",
        "Tests whether PCHE moderation is driven by one domain.",
        "",
    ])
    for r in domain_results:
        sig = " *" if r.get("ols_interaction_p") is not None and r["ols_interaction_p"] < 0.05 else ""
        p_str = f"{r['ols_interaction_p']:.4f}" if r["ols_interaction_p"] is not None else "N/A"
        b_str = f"{r['ols_interaction_beta']:.4f}" if r["ols_interaction_beta"] is not None else "N/A"
        summary_lines.append(
            f"- {r['domain']} x {r['risk_qid']} (n={r['n']}): "
            f"OLS beta={b_str}, p={p_str}{sig}"
        )
    summary_lines.append("")

    # Item 2: Intentionality
    summary_lines.extend([
        "## Item 2: PCHE With and Without Intentionality (QUO257)",
        "",
        f"- Cronbach's alpha (7-item primary): {alpha_7:.4f} (n={n_alpha_7})",
        f"- Cronbach's alpha (6-item without QUO257): {alpha_6:.4f} (n={n_alpha_6})",
        "",
    ])
    for r in intent_results:
        b_str = f"{r['ols_interaction_beta']:.4f}" if r["ols_interaction_beta"] is not None else "N/A"
        p_str = f"{r['ols_interaction_p']:.4f}" if r["ols_interaction_p"] is not None else "N/A"
        summary_lines.append(
            f"- {r['pche_variant']} x {r['risk_qid']} (n={r['n']}): "
            f"OLS beta={b_str}, p={p_str}"
        )
    summary_lines.append("")

    # Item 3: Strict pre-conception
    summary_lines.extend([
        "## Item 3: Strict Pre-Conception PCHE Variants",
        "",
    ])
    for r in strict_results:
        b_str = f"{r['ols_interaction_beta']:.4f}" if r["ols_interaction_beta"] is not None else "N/A"
        p_str = f"{r['ols_interaction_p']:.4f}" if r["ols_interaction_p"] is not None else "N/A"
        perm_str = f"{r['cluster_permutation_p']:.4f}" if r.get("cluster_permutation_p") is not None else "N/A"
        ci_str = ""
        if r.get("cluster_bootstrap_ci_lo") is not None:
            ci_str = f", boot CI=[{r['cluster_bootstrap_ci_lo']:.4f}, {r['cluster_bootstrap_ci_hi']:.4f}]"
        summary_lines.append(
            f"- {r['pche_variant']} x {r['risk_qid']} (n={r['n']}, k={r['n_components']}): "
            f"OLS beta={b_str}, p={p_str}, perm p={perm_str}{ci_str}"
        )
    summary_lines.append("")

    # Item 4: Narrow without intentionality
    summary_lines.extend([
        "## Item 4: Narrow Pre-Pregnancy Without Intentionality (4-item)",
        "",
        "The most conservative test: only unambiguous pre-pregnancy health behaviors.",
        "",
    ])
    for r in narrow_no_int_results:
        b_str = f"{r['ols_interaction_beta']:.4f}" if r["ols_interaction_beta"] is not None else "N/A"
        p_str = f"{r['ols_interaction_p']:.4f}" if r["ols_interaction_p"] is not None else "N/A"
        perm_str = f"{r['cluster_permutation_p']:.4f}" if r.get("cluster_permutation_p") is not None else "N/A"
        summary_lines.append(
            f"- narrow_no_intent x {r['risk_qid']} (n={r['n']}): "
            f"OLS beta={b_str}, p={p_str}, perm p={perm_str}"
        )
    summary_lines.append("")

    # Item 5: Era-specific
    summary_lines.extend([
        "## Item 5: Era-Specific for Strict Variants",
        "",
    ])
    for r in era_strict_results:
        summary_lines.append(
            f"- {r['era']} {r['pche_variant']} x {r['risk_qid']} (n={r['n']}): "
            f"beta={r['interaction_beta']:.4f}, p={r['interaction_p']:.4f}"
        )
    summary_lines.append("")

    # Item 6: LOSO
    summary_lines.extend([
        "## Item 6: LOSO for Strict Variants",
        "",
    ])
    if len(loso_df) > 0:
        for variant in loso_df["pche_variant"].unique():
            for rq in loso_df["risk_qid"].unique():
                sub = loso_df[(loso_df["pche_variant"] == variant) &
                              (loso_df["risk_qid"] == rq)]
                if len(sub) == 0:
                    continue
                betas = sub["interaction_beta"].values
                summary_lines.append(
                    f"- {variant} x {rq}: range=[{betas.min():.4f}, {betas.max():.4f}], "
                    f"mean={betas.mean():.4f}, SD={betas.std():.4f}, "
                    f"n_states={len(sub)}"
                )
    summary_lines.append("")

    # Item 7: Wording
    summary_lines.extend([
        "## Item 7: Question Wording Appendix",
        "",
        f"Saved {len(wording_rows)} question wordings to question_wording_appendix.csv",
        "",
    ])

    # Item 8: Variant comparison
    summary_lines.extend([
        "## Item 8: Variant Comparison Figure",
        "",
    ])
    if len(comparison_df) > 0:
        for _, r in comparison_df.iterrows():
            perm_str = f"{r['cluster_perm_p']:.4f}" if pd.notna(r.get("cluster_perm_p")) else "N/A"
            summary_lines.append(
                f"- {r['variant']} x {r['risk_qid']} (n={int(r['n'])}): "
                f"beta={r['ols_interaction_beta']:.4f}, cluster perm p={perm_str}"
            )
    summary_lines.append("")

    # Item 9: VIF
    summary_lines.extend([
        "## Item 9: VIF for Primary Models",
        "",
    ])
    for r in vif_results:
        flag = " **HIGH**" if r["vif"] > 5 else ""
        summary_lines.append(f"- {r['risk_qid']} {r['variable']}: VIF={r['vif']:.2f}{flag}")
    summary_lines.append("")

    # BH FDR correction across all OLS p-values
    summary_lines.extend([
        "## FDR Correction (Benjamini-Hochberg)",
        "",
    ])
    all_pvals = []
    all_plabels = []
    for r in strict_results:
        if r.get("ols_interaction_p") is not None:
            all_pvals.append(r["ols_interaction_p"])
            all_plabels.append(f"{r['pche_variant']} x {r['risk_qid']}")
    for r in narrow_no_int_results:
        if r.get("ols_interaction_p") is not None:
            all_pvals.append(r["ols_interaction_p"])
            all_plabels.append(f"narrow_no_intent x {r['risk_qid']}")
    for r in domain_results:
        if r.get("ols_interaction_p") is not None:
            all_pvals.append(r["ols_interaction_p"])
            all_plabels.append(f"{r['domain']} x {r['risk_qid']}")
    for r in intent_results:
        if r.get("ols_interaction_p") is not None:
            all_pvals.append(r["ols_interaction_p"])
            all_plabels.append(f"{r['pche_variant']} x {r['risk_qid']}")

    if all_pvals:
        q_vals = bh_fdr(np.array(all_pvals))
        for label, pv, qv in zip(all_plabels, all_pvals, q_vals):
            sig = " *" if qv < 0.10 else ""
            summary_lines.append(f"- {label}: p={pv:.4f}, q={qv:.4f}{sig}")
    summary_lines.append("")

    # Write summary
    with open(out_dir / "followup_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  Summary written to followup_summary.txt")

    # Write metadata
    metadata["outputs"] = sorted(str(p.name) for p in out_dir.iterdir())
    metadata["figures"] = sorted(str(p.name) for p in fig_dir.iterdir()) if fig_dir.exists() else []
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"FOLLOW-UP ANALYSIS COMPLETE")
    print(f"Run ID: {run_id}")
    print(f"Output: {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
