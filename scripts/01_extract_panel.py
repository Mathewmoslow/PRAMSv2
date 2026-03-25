#!/usr/bin/env python3
"""
Script 1: Panel Extraction
===========================
Copies and validates the pre-built analysis panel. Excludes aggregate rows
(e.g., "PRAMS Total") that are not state-level observations. Produces a
corrected question dictionary with actual PRAMS question wording.

This is a standalone project. All data flows from this script forward.
"""
import argparse
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Corrected PRAMS variable descriptions from actual question text
CORRECTED_DESCRIPTIONS = {
    "QUO4": "Indicator of whether mother was still breastfeeding 4 weeks after delivery",
    "QUO5": "Did you ever breastfeed or pump breast milk to feed your new baby after delivery?",
    "QUO44": "Indicator of whether mother was still breastfeeding 8 weeks after delivery",
    "QUO101": "Was your baby seen by a doctor, nurse or other health care provider in the first week after leaving the hospital?",
    "QUO179": "Indicator of pre-pregnancy exercise 3 or more days a week",
    "QUO41": "Indicator of whether mother took vitamins more than 4 times a week during the month prior to pregnancy",
    "QUO65": "During the month before you got pregnant with your new baby, did you take a daily multivitamin?",
    "QUO249": "Indicator of mother having her teeth cleaned in 12 months prior to pregnancy (years 2009-2011)",
    "QUO75": "During your most recent pregnancy, did you have your teeth cleaned?",
    "QUO257": "When you got pregnant with your new baby were you trying to become pregnant?",
    "QUO296": "Did you get prenatal care as early in your pregnancy as you wanted? (years 2000-2008)",
    "QUO297": "Did you get prenatal care as early in your pregnancy as you wanted? (years 2009-2011)",
    "QUO74": "Indicator of whether mother reported frequent postpartum depressive symptoms (years 2004-2008)",
    "QUO219": "Indicator of whether mother reported frequent postpartum depressive symptoms (years 2009-2011)",
    "QUO197": "In the 12 months before your baby was born, you argued with your husband or partner more than usual",
    "QUO210": "Indicator of any partner-related stressors reported",
    "QUO313": "During the 12 months before you got pregnant, did your ex-husband or ex-partner push, hit, slap, kick, choke, or physically hurt you in any other way?",
    "QUO315": "During your most recent pregnancy, did an ex-husband or ex-partner push, hit, slap, kick, choke, or physically hurt you in any other way?",
    "QUO97": "Did a doctor, nurse, or other health care worker talk with you about baby blues or postpartum depression during pregnancy or after your delivery?",
}

# Rows to EXCLUDE (aggregate rows that are not individual states)
EXCLUDE_LOCATIONS = {"PRAMS Total", "Total", "US"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-panel",
        default=str(Path(__file__).resolve().parent.parent / "data" / "analysis_panel.csv"))
    parser.add_argument("--source-qdict",
        default=str(Path(__file__).resolve().parent.parent / "data" / "question_dictionary.csv"))
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "data"
    source_panel = Path(args.source_panel)
    source_qdict = Path(args.source_qdict)

    print(f"[Panel Extract] Source: {source_panel}")
    print(f"[Panel Extract] SHA-256: {sha256_file(source_panel)[:16]}...")

    # Load panel
    panel = pd.read_csv(source_panel, low_memory=False)
    print(f"  Raw panel: {len(panel)} rows, {panel.shape[1]} cols")
    print(f"  Locations: {panel['location_abbr'].nunique()} unique")

    # Exclude aggregate rows
    before = len(panel)
    panel = panel[~panel["location_abbr"].isin(EXCLUDE_LOCATIONS)].copy()
    excluded = before - len(panel)
    print(f"  Excluded {excluded} aggregate rows ({EXCLUDE_LOCATIONS})")
    print(f"  Clean panel: {len(panel)} rows, {panel['location_abbr'].nunique()} locations")

    # Save cleaned panel
    panel.to_csv(out_dir / "panel_clean.csv", index=False)
    print(f"  Saved: panel_clean.csv")

    # Load and correct question dictionary
    qdict = pd.read_csv(source_qdict)
    # Apply corrected descriptions
    for qid, desc in CORRECTED_DESCRIPTIONS.items():
        mask = qdict["question_id"] == qid
        if mask.any():
            qdict.loc[mask, "question_text_corrected"] = desc
        else:
            qdict = pd.concat([qdict, pd.DataFrame([{
                "question_id": qid, "question_text": desc,
                "question_text_corrected": desc,
            }])], ignore_index=True)
    # For items without correction, copy original
    qdict["question_text_corrected"] = qdict["question_text_corrected"].fillna(qdict["question_text"])
    qdict.to_csv(out_dir / "question_dictionary_corrected.csv", index=False)
    print(f"  Saved: question_dictionary_corrected.csv ({len(qdict)} entries)")

    # Panel accounting summary
    ppd_panel = panel[panel["outcome_ppd"].notna()]
    print(f"\n  === PANEL ACCOUNTING ===")
    print(f"  Full panel: {len(panel)} state-years, {panel['location_abbr'].nunique()} locations, "
          f"years {int(panel['year'].min())}-{int(panel['year'].max())}")
    print(f"  PDS panel (outcome non-missing): {len(ppd_panel)} state-years, "
          f"{ppd_panel['location_abbr'].nunique()} locations, "
          f"years {int(ppd_panel['year'].min())}-{int(ppd_panel['year'].max())}")
    print(f"  Obs per location: min={ppd_panel.groupby('location_abbr').size().min()}, "
          f"max={ppd_panel.groupby('location_abbr').size().max()}, "
          f"mean={ppd_panel.groupby('location_abbr').size().mean():.1f}")

    # PCHE component coverage
    pche_all = ["QUO179","QUO41","QUO65","QUO249","QUO75","QUO257",
                "QUO296","QUO297","QUO4","QUO5","QUO44","QUO101"]
    print(f"\n  === PCHE COMPONENT COVERAGE (PDS panel) ===")
    common = []
    for q in pche_all:
        if q in ppd_panel.columns:
            n = int(ppd_panel[q].notna().sum())
            pct = 100 * n / len(ppd_panel)
            full = "FULL" if n == len(ppd_panel) else ""
            print(f"    {q}: {n}/{len(ppd_panel)} ({pct:.0f}%) {full}")
            if n == len(ppd_panel):
                common.append(q)
    print(f"  Common-component set (all {len(ppd_panel)} rows): {common} ({len(common)} items)")


if __name__ == "__main__":
    main()
