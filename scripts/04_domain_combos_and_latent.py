#!/usr/bin/env python3
"""
PCHE Domain Combinations & Latent Variable Analysis — Script 4
================================================================
Reads: data/panel_clean.csv (317 rows, 180 with PDS outcome, 35 locations,
       years 2004-2011)

Two major sections:
  A. Pairwise domain combination analysis — tests every 2-domain and
     single-domain PCHE sub-index as a moderator of partner-stress -> PDS.
  B. Latent variable / factor analysis of PCHE components (7 and 12) —
     PCA, parallel analysis, VARIMAX rotation, factor-score moderation tests.

Terminology: PDS (postpartum depressive symptoms), PCHE (Pre-Conception
Health Engagement). Never PCHA, Agency, or PPD prevalence.

Usage:
    python scripts/04_domain_combos_and_latent.py [--panel PATH] [--run-id ID]

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

# Full 12-component set (sensitivity — era-specific items not in all rows)
PCHE_FULL_12 = {
    **PCHE_COMMON,
    "QUO179": "Exercised 3+ days/week in 3 months before pregnancy",
    "QUO249": "Had teeth cleaned by dentist before pregnancy",
    "QUO75":  "Had teeth cleaned during pregnancy",
    "QUO296": "Got prenatal care as early as wanted (2000-2008)",
    "QUO297": "Prenatal care began in first trimester (2009-2011)",
}
PCHE_FULL_12_ORDER = PCHE_COMMON_ORDER + ["QUO179", "QUO249", "QUO75", "QUO296", "QUO297"]

# Partner-stress risk anchors (primary)
PRIMARY_RISK = ["QUO197", "QUO210"]
RISK_LABELS = {
    "QUO197": "Partner argued more than usual (12 mo before birth)",
    "QUO210": "Any partner-related stressors",
}

# Three domains within the 7 common components
DOMAIN_NUTRITIONAL = {"QUO41": "vitamins 4x/wk", "QUO65": "daily multivitamin"}
DOMAIN_INTENTIONALITY = {"QUO257": "trying to become pregnant"}
DOMAIN_POSTPARTUM = {
    "QUO4":   "breastfeeding 4wk",
    "QUO5":   "ever breastfed",
    "QUO44":  "breastfeeding 8wk",
    "QUO101": "baby checkup",
}

# Pairwise domain combinations
DOMAIN_COMBOS = {
    "Intentionality + Nutritional": {
        "items": ["QUO257", "QUO41", "QUO65"],
        "domains": ["Intentionality", "Nutritional"],
    },
    "Intentionality + Postpartum Engagement": {
        "items": ["QUO257", "QUO4", "QUO5", "QUO44", "QUO101"],
        "domains": ["Intentionality", "Postpartum Engagement"],
    },
    "Nutritional + Postpartum Engagement": {
        "items": ["QUO41", "QUO65", "QUO4", "QUO5", "QUO44", "QUO101"],
        "domains": ["Nutritional", "Postpartum Engagement"],
    },
}

# Single-domain definitions (for comparison)
SINGLE_DOMAINS = {
    "Nutritional only": {
        "items": ["QUO41", "QUO65"],
        "domains": ["Nutritional"],
    },
    "Intentionality only": {
        "items": ["QUO257"],
        "domains": ["Intentionality"],
    },
    "Postpartum Engagement only": {
        "items": ["QUO4", "QUO5", "QUO44", "QUO101"],
        "domains": ["Postpartum Engagement"],
    },
    "Full 7-component primary": {
        "items": PCHE_COMMON_ORDER,
        "domains": ["Nutritional", "Intentionality", "Postpartum Engagement"],
    },
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
    """Fit OLS (or WLS) regression y ~ intercept + X_cols.

    Returns dict with beta, se, pvals, n, k, r2, adj_r2, rmse, aic, bic,
    rss, tss, col_names, XtX_inv; or None on failure.
    """
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


# ---------------------------------------------------------------------------
# PCHE index builder
# ---------------------------------------------------------------------------
def build_pche_index(panel, component_dict_or_list, min_n=MIN_N, min_components=2):
    """Build a z-score composite index.

    component_dict_or_list: dict {QUO->desc} or list [QUO, ...].
    Returns (index_array, available_qids, z_cols_dict).
    """
    if isinstance(component_dict_or_list, dict):
        candidates = list(component_dict_or_list.keys())
    else:
        candidates = list(component_dict_or_list)

    avail = [q for q in candidates if q in panel.columns
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
# Parallel analysis (Horn 1965)
# ---------------------------------------------------------------------------
def parallel_analysis(data_matrix, n_iter=1000, seed=RANDOM_SEED):
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
# VARIMAX rotation (Kaiser)
# ---------------------------------------------------------------------------
def varimax_rotation(loadings, max_iter=100, tol=1e-6):
    """Kaiser varimax rotation."""
    n, k = loadings.shape
    rotation = np.eye(k)
    for _ in range(max_iter):
        old_rotation = rotation.copy()
        for i in range(k):
            for j in range(i + 1, k):
                # Compute rotation angle for columns i and j
                x = loadings @ rotation
                u = x[:, i] ** 2 - x[:, j] ** 2
                v = 2 * x[:, i] * x[:, j]
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)
                num = D - 2 * A * B / n
                den = C - (A ** 2 - B ** 2) / n
                angle = 0.25 * np.arctan2(num, den)
                # Apply rotation
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot = np.eye(k)
                rot[i, i] = cos_a
                rot[j, j] = cos_a
                rot[i, j] = -sin_a
                rot[j, i] = sin_a
                rotation = rotation @ rot
        if np.max(np.abs(rotation - old_rotation)) < tol:
            break
    return loadings @ rotation, rotation


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PCHE Domain Combinations & Latent Variable Analysis"
    )
    parser.add_argument(
        "--panel",
        default=str(Path(__file__).resolve().parent.parent / "data" / "panel_clean.csv"),
    )
    parser.add_argument("--run-id", default="run_20260325T_domains_lv")
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

    print(f"[Script 04] Run ID: {run_id}")
    print(f"[Script 04] Seed: {seed}")
    print(f"[Script 04] Panel: {len(panel)} rows, PDS panel: {len(pds_panel)} rows")
    print(f"[Script 04] SHA-256: {panel_hash[:16]}...")

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
        "purpose": "Domain combinations and latent variable analysis for PCHE",
    }

    # ===================================================================
    # SECTION A: PAIRWISE DOMAIN COMBINATION ANALYSIS
    # ===================================================================
    print("\n" + "=" * 70)
    print("SECTION A: PAIRWISE DOMAIN COMBINATION ANALYSIS")
    print("=" * 70)

    # Merge all combos into one dict for iteration
    all_combos = {}
    all_combos.update(DOMAIN_COMBOS)
    all_combos.update(SINGLE_DOMAINS)

    combo_results = []

    for combo_name, combo_info in all_combos.items():
        items = combo_info["items"]
        n_items = len(items)
        print(f"\n--- {combo_name} ({n_items} items: {', '.join(items)}) ---")

        # Build sub-index
        if n_items == 1:
            # Single indicator: raw z-score
            qid = items[0]
            if qid not in pds_panel.columns or pds_panel[qid].notna().sum() < MIN_N:
                print(f"  SKIP: {qid} not available or < {MIN_N} valid")
                continue
            sub_index = zscore(pds_panel[qid].values.astype(float))
            cronbach = float("nan")
            n_alpha = int(pds_panel[qid].notna().sum())
            print(f"  Single indicator ({qid}): raw z-score, alpha=N/A")
        else:
            sub_index, avail, z_cols = build_pche_index(
                pds_panel, items, min_components=2
            )
            if not avail or len(avail) < 2:
                print(f"  SKIP: fewer than 2 components available")
                continue
            # Cronbach alpha
            z_df = pd.DataFrame(z_cols, index=pds_panel.index)
            cronbach, n_alpha, k_alpha = compute_cronbach_alpha(z_df)
            print(f"  Cronbach alpha = {cronbach:.4f} (k={k_alpha}, n={n_alpha})")

        col_name = f"combo_{combo_name.replace(' ', '_').replace('+', 'plus')}"
        pds_panel[col_name] = sub_index

        for risk_qid in PRIMARY_RISK:
            if risk_qid not in pds_panel.columns:
                continue
            mask = (pds_panel["outcome_ppd"].notna() &
                    pds_panel[risk_qid].notna() &
                    pd.notna(sub_index))
            work = pds_panel[mask].copy()
            if len(work) < MIN_N:
                print(f"  {risk_qid}: n={len(work)} < {MIN_N}, skip")
                continue

            y = work["outcome_ppd"].values
            risk_z = zscore(work[risk_qid].values)
            mod_z = zscore(work[col_name].values)
            interaction = risk_z * mod_z
            cluster_ids = work["location_abbr"].values

            # OLS
            m_ols = fit_ols(y, [risk_z, mod_z, interaction],
                            col_names=["risk_z", "pche_z", "risk_x_pche"])
            ols_b, ols_se, ols_p = extract_coef(m_ols, "risk_x_pche")

            # State + year FE
            fe_d, fe_n = make_fe_dummies(work)
            m_fe = fit_ols(y, fe_d + [risk_z, mod_z, interaction],
                           col_names=fe_n + ["risk_z", "pche_z", "risk_x_pche"])
            fe_b, fe_se, fe_p = extract_coef(m_fe, "risk_x_pche")

            # CR1 cluster-robust FE
            m_cr = fit_ols_cluster_robust(
                y, fe_d + [risk_z, mod_z, interaction],
                cluster_ids=cluster_ids,
                col_names=fe_n + ["risk_z", "pche_z", "risk_x_pche"],
            )
            cr_b, cr_se, cr_p = extract_coef(m_cr, "risk_x_pche")

            print(f"  {risk_qid} (n={len(work)}):")
            if ols_b is not None:
                print(f"    OLS: beta={ols_b:.4f}, se={ols_se:.4f}, p={ols_p:.4f}")
            if fe_b is not None:
                print(f"    FE:  beta={fe_b:.4f}, se={fe_se:.4f}, p={fe_p:.4f}")
            if cr_b is not None:
                print(f"    CR1: beta={cr_b:.4f}, se={cr_se:.4f}, p={cr_p:.4f}")

            combo_results.append({
                "combination": combo_name,
                "n_items": n_items,
                "components": ", ".join(items),
                "cronbach_alpha": cronbach,
                "risk_qid": risk_qid,
                "n": len(work),
                "ols_beta": ols_b,
                "ols_se": ols_se if ols_b is not None else None,
                "ols_p": ols_p,
                "fe_beta": fe_b,
                "fe_se": fe_se if fe_b is not None else None,
                "fe_p": fe_p,
                "cr_beta": cr_b,
                "cr_se": cr_se if cr_b is not None else None,
                "cr_p": cr_p,
            })

    combo_df = pd.DataFrame(combo_results)
    combo_df.to_csv(out_dir / "domain_combination_analysis.csv", index=False)
    print(f"\n  Saved domain_combination_analysis.csv ({len(combo_df)} rows)")

    # --- Figure: Domain combination comparison bar chart ---
    print("\n  Creating domain combination comparison figure...")
    for risk_qid in PRIMARY_RISK:
        sub = combo_df[combo_df["risk_qid"] == risk_qid].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("cr_beta", ascending=True, na_position="first")

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(sub))
        labels = sub["combination"].values
        betas_ols = sub["ols_beta"].values.astype(float)
        betas_fe = sub["fe_beta"].values.astype(float)
        betas_cr = sub["cr_beta"].values.astype(float)

        bar_height = 0.25
        ax.barh(y_pos - bar_height, betas_ols, bar_height, label="OLS",
                color="#4c72b0", alpha=0.85)
        ax.barh(y_pos, betas_fe, bar_height, label="State+Year FE",
                color="#55a868", alpha=0.85)
        ax.barh(y_pos + bar_height, betas_cr, bar_height, label="CR1 (FE)",
                color="#c44e52", alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Interaction beta (PCHE sub-index x Risk -> PDS)", fontsize=10)
        ax.set_title(f"Domain Combination Moderation of {risk_qid} -> PDS", fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()

        fname = f"fig_domain_combination_comparison_{risk_qid}.png"
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

    # Also save a single combined figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax_idx, risk_qid in enumerate(PRIMARY_RISK):
        ax = axes[ax_idx]
        sub = combo_df[combo_df["risk_qid"] == risk_qid].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("cr_beta", ascending=True, na_position="first")
        y_pos = np.arange(len(sub))
        labels = sub["combination"].values
        bar_height = 0.25

        ax.barh(y_pos - bar_height, sub["ols_beta"].values.astype(float),
                bar_height, label="OLS", color="#4c72b0", alpha=0.85)
        ax.barh(y_pos, sub["fe_beta"].values.astype(float),
                bar_height, label="State+Year FE", color="#55a868", alpha=0.85)
        ax.barh(y_pos + bar_height, sub["cr_beta"].values.astype(float),
                bar_height, label="CR1 (FE)", color="#c44e52", alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Interaction beta", fontsize=10)
        ax.set_title(f"{risk_qid}", fontsize=11)
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("PCHE Domain Combination Moderation: Risk x PCHE Sub-Index -> PDS",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_domain_combination_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_domain_combination_comparison.png")

    # ===================================================================
    # SECTION B: LATENT VARIABLE / FACTOR ANALYSIS OF PCHE COMPONENTS
    # ===================================================================
    print("\n" + "=" * 70)
    print("SECTION B: LATENT VARIABLE / FACTOR ANALYSIS")
    print("=" * 70)

    summary_lines = []
    summary_lines.append("# PCHE Latent Variable Analysis Summary")
    summary_lines.append(f"Run ID: {run_id}")
    summary_lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    summary_lines.append("")

    # ---------------------------------------------------------------
    # B1: Factor analysis on the 7 common components (n=180)
    # ---------------------------------------------------------------
    print("\n--- B1: Factor Analysis — 7 Common Components ---")
    summary_lines.append("## B1: Factor Analysis on 7 Common PCHE Components")
    summary_lines.append("")

    # Build z-scored matrix for the 7 components (complete cases)
    avail_7 = [q for q in PCHE_COMMON_ORDER if q in pds_panel.columns
               and pds_panel[q].notna().sum() >= MIN_N]
    z_matrix_7 = np.column_stack([
        zscore(pds_panel[q].values.astype(float)) for q in avail_7
    ])
    # Complete cases only
    complete_mask_7 = ~np.isnan(z_matrix_7).any(axis=1)
    z_complete_7 = z_matrix_7[complete_mask_7]
    n_complete_7 = z_complete_7.shape[0]
    p_7 = z_complete_7.shape[1]
    print(f"  Available components: {avail_7}")
    print(f"  Complete cases: {n_complete_7} (of {len(pds_panel)})")
    summary_lines.append(f"Components: {', '.join(avail_7)} ({p_7} items)")
    summary_lines.append(f"Complete cases: {n_complete_7}")
    summary_lines.append("")

    # Correlation matrix
    corr_7 = np.corrcoef(z_complete_7, rowvar=False)
    print(f"  Correlation matrix ({p_7}x{p_7}) computed.")

    # PCA: eigenvalues and eigenvectors from correlation matrix
    eigenvalues_7_raw, eigenvectors_7_raw = np.linalg.eigh(corr_7)
    # Sort descending
    sort_idx = np.argsort(eigenvalues_7_raw)[::-1]
    eigenvalues_7 = eigenvalues_7_raw[sort_idx]
    eigenvectors_7 = eigenvectors_7_raw[:, sort_idx]

    var_explained_7 = eigenvalues_7 / eigenvalues_7.sum()
    cum_var_7 = np.cumsum(var_explained_7)

    print(f"  Eigenvalues: {np.round(eigenvalues_7, 4)}")
    print(f"  Variance explained: {np.round(var_explained_7 * 100, 2)}%")
    print(f"  Cumulative: {np.round(cum_var_7 * 100, 2)}%")

    eig_df_7 = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(p_7)],
        "qid": avail_7,
        "eigenvalue": eigenvalues_7,
        "variance_explained": var_explained_7,
        "cumulative_variance": cum_var_7,
    })
    eig_df_7.to_csv(out_dir / "latent_eigenvalues_7comp.csv", index=False)

    # Parallel analysis
    print("  Running parallel analysis (1000 iterations)...")
    pa_thresholds_7 = parallel_analysis(z_complete_7, n_iter=1000, seed=seed)
    print(f"  PA thresholds: {np.round(pa_thresholds_7, 4)}")

    n_retain_7 = int(np.sum(eigenvalues_7 > pa_thresholds_7))
    n_retain_7 = max(n_retain_7, 1)  # Retain at least 1
    print(f"  Factors to retain (eigenvalue > PA threshold): {n_retain_7}")

    pa_df_7 = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(p_7)],
        "eigenvalue": eigenvalues_7,
        "parallel_threshold_95": pa_thresholds_7,
        "retain": eigenvalues_7 > pa_thresholds_7,
    })
    pa_df_7.to_csv(out_dir / "latent_parallel_analysis_7comp.csv", index=False)

    summary_lines.append("### Eigenvalues and Parallel Analysis")
    for i in range(p_7):
        retain_str = " *RETAIN*" if eigenvalues_7[i] > pa_thresholds_7[i] else ""
        summary_lines.append(
            f"  PC{i+1}: eigenvalue={eigenvalues_7[i]:.4f}, "
            f"PA threshold={pa_thresholds_7[i]:.4f}, "
            f"var explained={var_explained_7[i]*100:.1f}%{retain_str}"
        )
    summary_lines.append(f"Factors retained: {n_retain_7}")
    summary_lines.append("")

    # Extract retained factor loadings (unrotated)
    # Loadings = eigenvectors * sqrt(eigenvalues)
    loadings_unrotated_7 = eigenvectors_7[:, :n_retain_7] * np.sqrt(eigenvalues_7[:n_retain_7])

    loadings_unrot_df_7 = pd.DataFrame(
        loadings_unrotated_7,
        index=avail_7,
        columns=[f"Factor{i+1}" for i in range(n_retain_7)],
    )
    loadings_unrot_df_7.index.name = "qid"
    loadings_unrot_df_7.to_csv(out_dir / "latent_loadings_7comp.csv")

    # VARIMAX rotation (only if > 1 factor retained)
    if n_retain_7 > 1:
        print(f"  Applying VARIMAX rotation to {n_retain_7} factors...")
        loadings_rotated_7, rotation_matrix_7 = varimax_rotation(loadings_unrotated_7)
    else:
        print("  Only 1 factor retained; no rotation needed.")
        loadings_rotated_7 = loadings_unrotated_7.copy()
        rotation_matrix_7 = np.eye(1)

    loadings_rot_df_7 = pd.DataFrame(
        loadings_rotated_7,
        index=avail_7,
        columns=[f"Factor{i+1}" for i in range(n_retain_7)],
    )
    loadings_rot_df_7.index.name = "qid"
    loadings_rot_df_7.to_csv(out_dir / "latent_loadings_rotated_7comp.csv")

    print("  Rotated loadings:")
    for i, qid in enumerate(avail_7):
        vals = ", ".join([f"{loadings_rotated_7[i, j]:.3f}" for j in range(n_retain_7)])
        print(f"    {qid}: [{vals}]")

    summary_lines.append("### Rotated Loadings (VARIMAX)")
    header = "QID | " + " | ".join([f"Factor{i+1}" for i in range(n_retain_7)])
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    for i, qid in enumerate(avail_7):
        vals = " | ".join([f"{loadings_rotated_7[i, j]:.4f}" for j in range(n_retain_7)])
        desc = PCHE_COMMON.get(qid, "")
        summary_lines.append(f"{qid} ({desc}) | {vals}")
    summary_lines.append("")

    # Factor interpretation
    summary_lines.append("### Factor Interpretation")
    for f_idx in range(n_retain_7):
        high_loaders = [(avail_7[i], loadings_rotated_7[i, f_idx])
                        for i in range(p_7) if abs(loadings_rotated_7[i, f_idx]) > 0.3]
        high_loaders.sort(key=lambda x: abs(x[1]), reverse=True)
        items_str = ", ".join([f"{q} ({v:.3f})" for q, v in high_loaders])
        summary_lines.append(f"Factor{f_idx+1}: {items_str}")
    summary_lines.append("")

    # Compute factor scores for each state-year (complete cases)
    # Factor scores via regression method: F = Z * R^-1 * L
    try:
        corr_inv_7 = np.linalg.inv(corr_7)
        factor_scores_7 = z_complete_7 @ corr_inv_7 @ loadings_rotated_7
    except np.linalg.LinAlgError:
        # Fallback: simple scoring (Z * loadings)
        factor_scores_7 = z_complete_7 @ loadings_rotated_7

    # Map back to pds_panel indices
    pds_complete_idx = pds_panel.index[complete_mask_7]
    fs_df_7 = pd.DataFrame(
        factor_scores_7,
        index=pds_complete_idx,
        columns=[f"Factor{i+1}_score" for i in range(n_retain_7)],
    )
    # Add location and year for context
    fs_df_7["location_abbr"] = pds_panel.loc[pds_complete_idx, "location_abbr"].values
    fs_df_7["year"] = pds_panel.loc[pds_complete_idx, "year"].values
    fs_df_7.to_csv(out_dir / "latent_factor_scores_7comp.csv", index=False)

    # Assign factor scores back to pds_panel
    for f_idx in range(n_retain_7):
        col = f"factor{f_idx+1}_score"
        pds_panel[col] = np.nan
        pds_panel.loc[pds_complete_idx, col] = factor_scores_7[:, f_idx]

    # Test each factor score as a moderator of QUO197 and QUO210 -> PDS
    print("\n  Testing factor scores as moderators...")
    factor_mod_results = []
    for f_idx in range(n_retain_7):
        f_col = f"factor{f_idx+1}_score"
        for risk_qid in PRIMARY_RISK:
            if risk_qid not in pds_panel.columns:
                continue
            mask = (pds_panel["outcome_ppd"].notna() &
                    pds_panel[risk_qid].notna() &
                    pds_panel[f_col].notna())
            work = pds_panel[mask].copy()
            if len(work) < MIN_N:
                continue

            y = work["outcome_ppd"].values
            risk_z = zscore(work[risk_qid].values)
            f_z = zscore(work[f_col].values)
            interaction = risk_z * f_z

            # OLS
            m_ols = fit_ols(y, [risk_z, f_z, interaction],
                            col_names=["risk_z", "factor_z", "risk_x_factor"])
            ols_b, ols_se, ols_p = extract_coef(m_ols, "risk_x_factor")

            # FE
            fe_d, fe_n = make_fe_dummies(work)
            m_fe = fit_ols(y, fe_d + [risk_z, f_z, interaction],
                           col_names=fe_n + ["risk_z", "factor_z", "risk_x_factor"])
            fe_b, fe_se, fe_p = extract_coef(m_fe, "risk_x_factor")

            print(f"    Factor{f_idx+1} x {risk_qid} (n={len(work)}):")
            if ols_b is not None:
                print(f"      OLS: beta={ols_b:.4f}, p={ols_p:.4f}")
            if fe_b is not None:
                print(f"      FE:  beta={fe_b:.4f}, p={fe_p:.4f}")

            factor_mod_results.append({
                "factor": f"Factor{f_idx+1}",
                "risk_qid": risk_qid,
                "n": len(work),
                "ols_beta": ols_b,
                "ols_se": ols_se,
                "ols_p": ols_p,
                "ols_r2": m_ols["r2"] if m_ols else None,
                "fe_beta": fe_b,
                "fe_se": fe_se,
                "fe_p": fe_p,
                "fe_r2": m_fe["r2"] if m_fe else None,
            })

    factor_mod_df_7 = pd.DataFrame(factor_mod_results)
    factor_mod_df_7.to_csv(out_dir / "latent_factor_moderation_7comp.csv", index=False)

    # Compare factor scores vs simple composite mean
    print("\n  Comparing factor scores vs simple composite for prediction...")
    summary_lines.append("### Factor Score vs Simple Composite Comparison")
    pche_7_idx, _, _ = build_pche_index(pds_panel, PCHE_COMMON, min_components=2)
    pds_panel["pche_7_composite"] = pche_7_idx

    for risk_qid in PRIMARY_RISK:
        if risk_qid not in pds_panel.columns:
            continue
        # Simple composite model
        mask_c = (pds_panel["outcome_ppd"].notna() &
                  pds_panel[risk_qid].notna() &
                  pds_panel["pche_7_composite"].notna())
        work_c = pds_panel[mask_c].copy()
        if len(work_c) < MIN_N:
            continue
        y_c = work_c["outcome_ppd"].values
        risk_z_c = zscore(work_c[risk_qid].values)
        pche_z_c = zscore(work_c["pche_7_composite"].values)
        int_c = risk_z_c * pche_z_c
        fe_d_c, fe_n_c = make_fe_dummies(work_c)
        m_comp = fit_ols(y_c, fe_d_c + [risk_z_c, pche_z_c, int_c],
                         col_names=fe_n_c + ["risk_z", "pche_z", "risk_x_pche"])
        comp_r2 = m_comp["adj_r2"] if m_comp else None

        # Factor model (all factors + interactions)
        factor_cols = [f"factor{f_idx+1}_score" for f_idx in range(n_retain_7)]
        mask_f = (pds_panel["outcome_ppd"].notna() &
                  pds_panel[risk_qid].notna())
        for fc in factor_cols:
            mask_f = mask_f & pds_panel[fc].notna()
        work_f = pds_panel[mask_f].copy()
        if len(work_f) < MIN_N:
            continue

        y_f = work_f["outcome_ppd"].values
        risk_z_f = zscore(work_f[risk_qid].values)
        fe_d_f, fe_n_f = make_fe_dummies(work_f)
        factor_z_list = []
        factor_names = []
        for f_idx in range(n_retain_7):
            fc = f"factor{f_idx+1}_score"
            fz = zscore(work_f[fc].values)
            factor_z_list.append(fz)
            factor_names.append(f"F{f_idx+1}")
            factor_z_list.append(risk_z_f * fz)
            factor_names.append(f"risk_x_F{f_idx+1}")

        m_factor = fit_ols(
            y_f,
            fe_d_f + [risk_z_f] + factor_z_list,
            col_names=fe_n_f + ["risk_z"] + factor_names,
        )
        factor_r2 = m_factor["adj_r2"] if m_factor else None

        line = (f"  {risk_qid}: Composite adj-R2={comp_r2:.4f}, "
                f"Factor adj-R2={factor_r2:.4f}" if comp_r2 is not None and factor_r2 is not None
                else f"  {risk_qid}: comparison not available")
        print(line)
        summary_lines.append(line)

    summary_lines.append("")

    # --- Scree plot ---
    print("\n  Creating scree plot...")
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(1, p_7 + 1)
    ax.plot(x_pos, eigenvalues_7, "bo-", linewidth=2, markersize=8, label="Observed eigenvalues")
    ax.plot(x_pos, pa_thresholds_7, "r^--", linewidth=1.5, markersize=7,
            label="Parallel analysis (95th pctile)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="Kaiser criterion (=1)")
    ax.set_xlabel("Component", fontsize=11)
    ax.set_ylabel("Eigenvalue", fontsize=11)
    ax.set_title("Scree Plot: PCHE 7 Common Components with Parallel Analysis", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"PC{i}" for i in x_pos])
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_scree_7comp.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_scree_7comp.png")

    # --- Loading heatmap ---
    print("  Creating loading heatmap...")
    fig, ax = plt.subplots(figsize=(max(4, n_retain_7 * 2 + 2), 6))
    im = ax.imshow(loadings_rotated_7, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(p_7))
    ax.set_yticklabels([f"{q}\n({PCHE_COMMON.get(q, '')[:30]})" for q in avail_7],
                       fontsize=8)
    ax.set_xticks(range(n_retain_7))
    ax.set_xticklabels([f"Factor{i+1}" for i in range(n_retain_7)], fontsize=10)
    ax.set_title("VARIMAX Rotated Loadings: 7 PCHE Components", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Loading", fontsize=10)
    # Annotate cells
    for i in range(p_7):
        for j in range(n_retain_7):
            val = loadings_rotated_7[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if abs(val) > 0.3 else "normal")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_loading_heatmap_7comp.png", dpi=150)
    plt.close(fig)
    print("  Saved fig_loading_heatmap_7comp.png")

    # ---------------------------------------------------------------
    # B2: Factor analysis on the FULL 12 components (complete-case)
    # ---------------------------------------------------------------
    print("\n--- B2: Factor Analysis — 12 Full Components ---")
    summary_lines.append("## B2: Factor Analysis on Full 12 PCHE Components")
    summary_lines.append("")

    avail_12 = [q for q in PCHE_FULL_12_ORDER if q in pds_panel.columns
                and pds_panel[q].notna().sum() >= MIN_N]
    print(f"  Available components: {avail_12} ({len(avail_12)} items)")

    if len(avail_12) >= 3:
        z_matrix_12 = np.column_stack([
            zscore(pds_panel[q].values.astype(float)) for q in avail_12
        ])
        complete_mask_12 = ~np.isnan(z_matrix_12).any(axis=1)
        z_complete_12 = z_matrix_12[complete_mask_12]
        n_complete_12 = z_complete_12.shape[0]
        p_12 = z_complete_12.shape[1]
        print(f"  Complete cases: {n_complete_12}")
        summary_lines.append(f"Components: {', '.join(avail_12)} ({p_12} items)")
        summary_lines.append(f"Complete cases: {n_complete_12}")

        if n_complete_12 < MIN_N:
            msg = (f"  SKIP: Only {n_complete_12} complete cases (< {MIN_N}). "
                   "Not enough data for 12-component factor analysis.")
            print(msg)
            summary_lines.append(msg)
            summary_lines.append("")
        else:
            corr_12 = np.corrcoef(z_complete_12, rowvar=False)
            eigenvalues_12_raw, eigenvectors_12_raw = np.linalg.eigh(corr_12)
            sort_idx_12 = np.argsort(eigenvalues_12_raw)[::-1]
            eigenvalues_12 = eigenvalues_12_raw[sort_idx_12]
            eigenvectors_12 = eigenvectors_12_raw[:, sort_idx_12]
            var_explained_12 = eigenvalues_12 / eigenvalues_12.sum()
            cum_var_12 = np.cumsum(var_explained_12)

            print(f"  Eigenvalues: {np.round(eigenvalues_12, 4)}")

            eig_df_12 = pd.DataFrame({
                "component": [f"PC{i+1}" for i in range(p_12)],
                "qid": avail_12,
                "eigenvalue": eigenvalues_12,
                "variance_explained": var_explained_12,
                "cumulative_variance": cum_var_12,
            })
            eig_df_12.to_csv(out_dir / "latent_eigenvalues_12comp.csv", index=False)

            # Parallel analysis
            print("  Running parallel analysis for 12 components...")
            pa_thresholds_12 = parallel_analysis(z_complete_12, n_iter=1000, seed=seed)
            n_retain_12 = int(np.sum(eigenvalues_12 > pa_thresholds_12))
            n_retain_12 = max(n_retain_12, 1)
            print(f"  Factors to retain: {n_retain_12}")

            # Loadings
            loadings_unrot_12 = eigenvectors_12[:, :n_retain_12] * np.sqrt(eigenvalues_12[:n_retain_12])

            if n_retain_12 > 1:
                loadings_rot_12, _ = varimax_rotation(loadings_unrot_12)
            else:
                loadings_rot_12 = loadings_unrot_12.copy()

            loadings_df_12 = pd.DataFrame(
                loadings_rot_12,
                index=avail_12,
                columns=[f"Factor{i+1}" for i in range(n_retain_12)],
            )
            loadings_df_12.index.name = "qid"
            loadings_df_12.to_csv(out_dir / "latent_loadings_12comp.csv")

            print("  Rotated loadings (12 comp):")
            for i, qid in enumerate(avail_12):
                vals = ", ".join([f"{loadings_rot_12[i, j]:.3f}" for j in range(n_retain_12)])
                print(f"    {qid}: [{vals}]")

            summary_lines.append(f"Factors retained: {n_retain_12}")
            summary_lines.append("")
            summary_lines.append("### 12-Component Rotated Loadings")
            for i, qid in enumerate(avail_12):
                vals = ", ".join([f"{loadings_rot_12[i, j]:.4f}" for j in range(n_retain_12)])
                desc = PCHE_FULL_12.get(qid, "")
                summary_lines.append(f"  {qid} ({desc}): [{vals}]")
            summary_lines.append("")

            # Compare factor structures
            summary_lines.append("### Comparison: 7-Component vs 12-Component Factor Structures")
            summary_lines.append(f"  7-component: {n_retain_7} factors retained, "
                                 f"PC1 variance = {var_explained_7[0]*100:.1f}%")
            summary_lines.append(f"  12-component: {n_retain_12} factors retained, "
                                 f"PC1 variance = {var_explained_12[0]*100:.1f}%")
            summary_lines.append("")
    else:
        msg = f"  SKIP: Only {len(avail_12)} of 12 components available in data."
        print(msg)
        summary_lines.append(msg)
        summary_lines.append("")

    # ---------------------------------------------------------------
    # B3: Confirmatory check — single composite support
    # ---------------------------------------------------------------
    print("\n--- B3: Confirmatory Check — Single Composite Support ---")
    summary_lines.append("## B3: Does the Factor Structure Support a Single Composite?")
    summary_lines.append("")

    # Ratio of first to second eigenvalue
    if p_7 >= 2:
        eig_ratio = eigenvalues_7[0] / eigenvalues_7[1] if eigenvalues_7[1] > 0 else float("inf")
    else:
        eig_ratio = float("inf")
    pc1_var = var_explained_7[0] * 100

    print(f"  Eigenvalue ratio (PC1/PC2): {eig_ratio:.3f}")
    print(f"  PC1 variance explained: {pc1_var:.1f}%")
    print(f"  Factors retained by parallel analysis: {n_retain_7}")

    summary_lines.append(f"Eigenvalue ratio (PC1/PC2): {eig_ratio:.3f}")
    summary_lines.append(f"  (> 3 suggests dominant single factor)")
    summary_lines.append(f"Variance explained by PC1 alone: {pc1_var:.1f}%")
    summary_lines.append(f"Factors retained by parallel analysis: {n_retain_7}")
    summary_lines.append("")

    # Check for cross-loadings vs clean simple structure
    if n_retain_7 > 1:
        # Check if any item loads > 0.3 on more than one factor
        cross_loading_items = []
        for i, qid in enumerate(avail_7):
            high_factors = [j for j in range(n_retain_7)
                           if abs(loadings_rotated_7[i, j]) > 0.3]
            if len(high_factors) > 1:
                cross_loading_items.append(qid)

        if cross_loading_items:
            summary_lines.append(f"Cross-loading items (|loading| > 0.3 on >1 factor): "
                                 f"{', '.join(cross_loading_items)}")
            structure_type = "complex (cross-loadings present)"
        else:
            summary_lines.append("No cross-loading items found (clean simple structure).")
            structure_type = "clean simple structure"
    else:
        summary_lines.append("Single factor extracted; simple structure assessment not applicable.")
        structure_type = "single factor"

    summary_lines.append("")

    # Generate explicit conclusion
    summary_lines.append("### Conclusion")

    if eig_ratio >= 3 and pc1_var >= 30:
        conclusion = (
            f"The factor structure DOES support treating PCHE as a unitary composite "
            f"because the first eigenvalue ({eigenvalues_7[0]:.3f}) is {eig_ratio:.1f}x "
            f"the second ({eigenvalues_7[1]:.3f}), explaining {pc1_var:.1f}% of variance. "
            f"This strong first-factor dominance indicates that the 7 indicators share "
            f"substantial common variance captured by a single latent dimension."
        )
    elif eig_ratio >= 2 and pc1_var >= 25:
        conclusion = (
            f"The factor structure provides MODERATE support for treating PCHE as a "
            f"unitary composite. The eigenvalue ratio (PC1/PC2 = {eig_ratio:.2f}) and "
            f"PC1 variance ({pc1_var:.1f}%) suggest a reasonably dominant first factor, "
            f"though some multidimensionality exists. The composite index is defensible "
            f"but domain-specific analyses should be reported alongside."
        )
    else:
        conclusion = (
            f"The factor structure does NOT strongly support treating PCHE as a unitary "
            f"composite. The eigenvalue ratio (PC1/PC2 = {eig_ratio:.2f}) is below 3 "
            f"and PC1 explains only {pc1_var:.1f}% of variance, suggesting the 7 indicators "
            f"are multidimensional. Domain-specific indices may be more appropriate."
        )

    if n_retain_7 > 1 and structure_type == "clean simple structure":
        conclusion += (
            f" The VARIMAX rotation produced {n_retain_7} factors with clean simple "
            f"structure (no cross-loadings), supporting a multidimensional interpretation."
        )
    elif n_retain_7 > 1 and "cross-loadings" in structure_type:
        conclusion += (
            f" The VARIMAX rotation produced {n_retain_7} factors with cross-loadings, "
            f"suggesting the factors are not cleanly separable and a composite may still "
            f"be reasonable."
        )

    print(f"\n  CONCLUSION: {conclusion}")
    summary_lines.append(conclusion)
    summary_lines.append("")

    # Save summary
    with open(out_dir / "latent_summary.md", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\n  Saved latent_summary.md")

    # Save metadata
    metadata["section_a_n_combinations"] = len(combo_df)
    metadata["section_b_n_retain_7comp"] = n_retain_7
    metadata["section_b_pc1_variance_7comp"] = float(pc1_var)
    metadata["section_b_eigenvalue_ratio"] = float(eig_ratio)

    with open(out_dir / "metadata_04.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SCRIPT 04 COMPLETE")
    print("=" * 70)
    print(f"  Output dir: {out_dir}")
    print(f"  Figures dir: {fig_dir}")
    print(f"  Domain combination analysis: {len(combo_df)} rows")
    print(f"  Factors retained (7-comp): {n_retain_7}")
    print(f"  PC1/PC2 eigenvalue ratio: {eig_ratio:.3f}")
    print(f"  PC1 variance explained: {pc1_var:.1f}%")


if __name__ == "__main__":
    main()
