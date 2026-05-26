# module_2_5_piecewise_linear_dg_high_boundary.py
#
# =============================================================================
# PURPOSE
# =============================================================================
# Module 2.5 sits between:
#   - Module 2: Global Gating
#   - Module 3: MMB Threshold Finder
#   - Module 4: GMM Probabilistic Clustering
#
# The purpose of this module is to generate a patient-specific adaptive upper
# fluorescence boundary for DG-positive events using a piecewise-linear 2D gate.
#
# RATIONALE
# =============================================================================
# Module 2 provides:
#   - a global MMB FSC window
#   - a global DG low_cut
#   - a DG peak fluorescence anchor, currently set to 0.8
#
# The DG high-fluorescence tail becomes exaggerated in patient/plasma samples.
# Therefore, a single control-derived DG high_cut is not reliable enough.
#
# Instead, this module:
#   1) Opens each patient .fcs file.
#   2) Applies the same Logicle/asinh transformation.
#   3) Applies the global MMB FSC window from Module 2.
#   4) Divides the MMB FSC window into 10 FSC bins.
#   5) Assigns a progressively increasing max fluorescence limit across the bins.
#
# For an anchor of 0.8:
#   step = 2.5% of 0.8 = 0.02
#
# Therefore:
#   Bin 1  max fluorescence = 0.8 + 1(0.02)  = 0.82
#   Bin 2  max fluorescence = 0.8 + 2(0.02)  = 0.84
#   Bin 3  max fluorescence = 0.8 + 3(0.02)  = 0.86
#   ...
#   Bin 10 max fluorescence = 0.8 + 10(0.02) = 1.00
#
# This creates a set of paired points:
#   (FSC bin centre, max fluorescence)
#
# Downstream, Module 4 can interpolate between these points and keep events where:
#   fluor >= dg_low_cut
#   fluor <= patient_specific_max_fluorescence_at_that_FSC
#
# =============================================================================
# OUTPUTS
# =============================================================================
# For each patient/sample .fcs file:
#   - one patient-specific JSON containing:
#       - mmb_window
#       - dg_low_cut
#       - dg_peak_fluorescence
#       - fsc_bin_edges
#       - fsc_bin_centres
#       - max_fluorescence
#       - piecewise_boundary_points
#
# Also:
#   - one combined JSON containing all patient-specific boundaries
#   - one QC plot per patient showing the piecewise boundary
#
# =============================================================================

from __future__ import annotations

from pathlib import Path
import sys, os, json, math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fcsparser


# =============================================================================
# PATHS
# =============================================================================

# Folder containing Module 2 output: gate_summary_by_type.json
GATES_JSON_DIR = r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Controls Refined"
GATES_BASENAME = "gate_summary_by_type"

# Folder containing patient/sample .fcs files
PATIENT_FCS_DIR = r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Raw Flow Data\06.16.2025_20samples_optimized protocol"

# Output root for Module 2.5 results
OUTPUT_ROOT = r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Controls Refined"
OUTPUT_SUBFOLDER = "Module_2_5_Piecewise_Linear_DG_High_Boundary"


# =============================================================================
# PIECEWISE GATING SETTINGS
# =============================================================================

N_FSC_BINS = 10

# The final fluorescence limit is anchor × 1.25.
# For anchor = 0.8, this gives final upper limit = 1.0.
UPPER_MULTIPLIER = 1.25

# This means the full fluorescence expansion from anchor to anchor × 1.25
# is divided across 10 bins.
# For anchor = 0.8:
#   total increase = 0.8 × 0.25 = 0.2
#   step = 0.2 / 10 = 0.02
STEP_FRACTION_PER_BIN = (UPPER_MULTIPLIER - 1.0) / N_FSC_BINS

MAX_POINTS_PER_QC_PLOT = 40000


# =============================================================================
# TRANSFORMATION: Logicle preferred, with asinh fallback
# =============================================================================

try:
    from flowutils.transforms import logicle as logicle_transform
    LOGICLE_SOURCE = "flowutils.transforms.logicle"
except Exception:
    logicle_transform = None
    LOGICLE_SOURCE = "asinh-fallback"


def apply_logicle_or_asinh(arr: np.ndarray) -> np.ndarray:
    """
    Apply Logicle if available, otherwise apply asinh fallback.

    This mirrors the transformation logic used in the previous modules:
      T = max(262144, observed vmax)
      M = 4.5
      W = 0.5
      A = 0.5 if vmin < 0 else 0.0
    """
    arr = arr.astype(float, copy=False)
    v = arr[np.isfinite(arr)]

    if logicle_transform is not None and v.size:
        vmax = float(np.max(v))
        vmin = float(np.min(v))
        T = max(262144.0, vmax)
        M = 4.5
        W = 0.5
        A = 0.5 if vmin < 0 else 0.0
        return logicle_transform(arr, channel_indices=None, t=T, m=M, w=W, a=A)

    return np.arcsinh(arr / 150.0)


# =============================================================================
# CHANNEL PICKING
# =============================================================================

def first_match(columns, patterns):
    ups = [c.upper() for c in columns]

    for pat in patterns:
        P = pat.upper()

        for i, cu in enumerate(ups):
            if P in cu:
                return columns[i]

    return None


def pick_channels(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Auto-pick fluorescence and FSC channels.

    Returns:
      fluor_channel, fsc_channel
    """
    cols = list(df.columns)

    fsc = first_match(cols, ["FSC-H"]) or first_match(cols, ["FSC-A", "FSC"])

    fluor = (
        first_match(cols, ["FL1-H"])
        or first_match(cols, ["B525-H", "BL1-H", "FITC-H", "GFP-H"])
        or first_match(cols, ["B530-H", "B515-H", "530/30", "525/50"])
        or first_match(cols, ["FL1-A", "BL1-A", "B525-A", "FITC-A", "GFP-A"])
    )

    if fsc is None:
        raise ValueError(f"No FSC channel found in columns: {cols}")

    if fluor is None:
        h_cands = [
            c for c in cols
            if c.upper().endswith("-H")
            and not any(k in c.upper() for k in ["FSC", "SSC", "TIME", "WIDTH"])
        ]

        if h_cands:
            fluor = h_cands[0]
        else:
            a_cands = [
                c for c in cols
                if c.upper().endswith("-A")
                and not any(k in c.upper() for k in ["FSC", "SSC", "TIME", "WIDTH"])
            ]

            if not a_cands:
                raise ValueError(f"No fluorescence channel found in columns: {cols}")

            fluor = a_cands[0]

    return fluor, fsc


# =============================================================================
# JSON HELPERS
# =============================================================================

def safe_filename(name: str) -> str:
    """
    Make filename Windows-safe.
    """
    return "".join(c if c not in r'<>:"/\|?*' else "_" for c in name)


def find_json_by_basename(folder: Path, basename: str) -> Path:
    """
    Find a JSON file by exact or partial basename.
    """
    if not folder.exists():
        raise FileNotFoundError(f"JSON folder does not exist:\n  {folder}")

    exact = folder / f"{basename}.json"

    if exact.exists():
        return exact

    cands = sorted(folder.glob("*.json"))

    if not cands:
        raise FileNotFoundError(f"No .json files found in:\n  {folder}")

    for p in cands:
        if p.stem.lower().startswith(basename.lower()):
            return p

    for p in cands:
        if basename.lower() in p.stem.lower():
            return p

    raise FileNotFoundError(
        f"Could not find JSON matching basename '{basename}' in {folder}.\n"
        f"Available JSON files: {[p.name for p in cands]}"
    )


def load_module2_gate_summary(gates_json_path: Path):
    """
    Load Module 2 gate_summary_by_type.json.

    Expected flattened structure:
      {
        ".*": {
          "dg_low_cut": 0.12846260309997892,
          "dg_peak_fluorescence": 0.8,
          "mmb_window": [0.7141938599636604, 0.8972652021227275],
          "channels": {
            "fluor": "B525-H",
            "fsc": "FSC-H"
          }
        }
      }

    This function is written defensively so it can also tolerate older names
    if needed.
    """
    d = json.loads(gates_json_path.read_text(encoding="utf-8"))

    if not isinstance(d, dict) or len(d) == 0:
        raise ValueError(f"Unexpected or empty JSON structure in {gates_json_path}")

    first_key = next(iter(d.keys()))
    block = d[first_key]

    if not isinstance(block, dict):
        raise ValueError(f"Unexpected block under key '{first_key}' in {gates_json_path.name}")

    # Current expected fields
    mmb_window = block.get("mmb_window")
    dg_low_cut = block.get("dg_low_cut")
    dg_peak_fluorescence = block.get("dg_peak_fluorescence")

    # Backward compatibility if an older Module 2 JSON still has dg_window
    if dg_low_cut is None and block.get("dg_window") is not None:
        dg_window = block.get("dg_window")
        dg_low_cut = dg_window[0]

    if dg_peak_fluorescence is None:
        raise KeyError(
            "Could not find 'dg_peak_fluorescence' in Module 2 JSON.\n"
            "Please ensure Module 2 exports dg_peak_fluorescence, e.g. 0.8."
        )

    if mmb_window is None:
        raise KeyError("Could not find 'mmb_window' in Module 2 JSON.")

    if dg_low_cut is None:
        raise KeyError("Could not find 'dg_low_cut' in Module 2 JSON.")

    if len(mmb_window) != 2:
        raise ValueError(f"mmb_window should have two values, got: {mmb_window}")

    channels = block.get("channels", {})

    return (
        (float(mmb_window[0]), float(mmb_window[1])),
        float(dg_low_cut),
        float(dg_peak_fluorescence),
        channels,
        first_key
    )


# =============================================================================
# PIECEWISE BOUNDARY CONSTRUCTION
# =============================================================================

def build_piecewise_boundary(
    mmb_window: Tuple[float, float],
    dg_peak_fluorescence: float,
    n_bins: int = N_FSC_BINS,
    upper_multiplier: float = UPPER_MULTIPLIER,
):
    """
    Build a piecewise-linear upper fluorescence boundary.

    The FSC axis is split into n_bins between mmb_low and mmb_high.

    For each FSC bin, the maximum allowed fluorescence increases linearly from:
      anchor + 1 step
    to:
      anchor + n_bins steps = anchor × upper_multiplier

    For anchor = 0.8 and n_bins = 10:
      max fluorescence values are:
        0.82, 0.84, 0.86, ..., 1.00

    Returns:
      fsc_bin_edges
      fsc_bin_centres
      max_fluorescence
      boundary_points
    """
    mmb_low, mmb_high = mmb_window

    if not np.isfinite(mmb_low) or not np.isfinite(mmb_high) or mmb_low >= mmb_high:
        raise ValueError(f"Invalid MMB window: {mmb_window}")

    if not np.isfinite(dg_peak_fluorescence) or dg_peak_fluorescence <= 0:
        raise ValueError(f"Invalid DG peak fluorescence: {dg_peak_fluorescence}")

    fsc_bin_edges = np.linspace(mmb_low, mmb_high, n_bins + 1)
    fsc_bin_centres = 0.5 * (fsc_bin_edges[:-1] + fsc_bin_edges[1:])

    step_absolute = dg_peak_fluorescence * ((upper_multiplier - 1.0) / n_bins)

    max_fluorescence = np.array([
        dg_peak_fluorescence + (i + 1) * step_absolute
        for i in range(n_bins)
    ], dtype=float)

    boundary_points = [
        {
            "fsc": float(fsc_bin_centres[i]),
            "max_fluorescence": float(max_fluorescence[i])
        }
        for i in range(n_bins)
    ]

    return (
        fsc_bin_edges.astype(float),
        fsc_bin_centres.astype(float),
        max_fluorescence.astype(float),
        boundary_points
    )


def interpolate_piecewise_high_cut(
    fsc_values: np.ndarray,
    fsc_bin_centres: np.ndarray,
    max_fluorescence: np.ndarray,
):
    """
    Interpolate the piecewise-linear fluorescence boundary.

    For each event's FSC value, this returns the maximum allowed fluorescence
    at that FSC position.
    """
    return np.interp(
        fsc_values,
        fsc_bin_centres,
        max_fluorescence,
        left=max_fluorescence[0],
        right=max_fluorescence[-1]
    )


# =============================================================================
# QC PLOTTING
# =============================================================================

def save_piecewise_qc_plot(
    out_path: Path,
    sample_name: str,
    fluor_values: np.ndarray,
    fsc_values: np.ndarray,
    dg_low_cut: float,
    fsc_bin_centres: np.ndarray,
    max_fluorescence: np.ndarray,
    fluor_label: str,
    fsc_label: str,
    transform_label: str,
):
    """
    Save QC scatter plot showing:
      - patient events after global MMB window
      - DG low_cut vertical line
      - patient-specific piecewise high boundary
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = fluor_values
    y = fsc_values

    if x.size > MAX_POINTS_PER_QC_PLOT:
        rng = np.random.default_rng(0)
        pick = rng.choice(x.size, size=MAX_POINTS_PER_QC_PLOT, replace=False)
        x_plot = x[pick]
        y_plot = y[pick]
    else:
        x_plot = x
        y_plot = y

    plt.figure(figsize=(7.5, 5.2))
    plt.scatter(x_plot, y_plot, s=2, alpha=0.4)

    # DG low_cut is a vertical fluorescence boundary.
    plt.axvline(dg_low_cut, color="crimson", ls="--", lw=1, label="DG low_cut")

    # Piecewise high boundary is max fluorescence as a function of FSC.
    # Since the scatter plot has fluorescence on x-axis and FSC on y-axis,
    # we plot x = max_fluorescence, y = fsc_bin_centres.
    plt.plot(
        max_fluorescence,
        fsc_bin_centres,
        color="black",
        ls="-",
        lw=1.5,
        marker="o",
        markersize=3,
        label="Piecewise high boundary"
    )

    plt.xlabel(f"{fluor_label} ({transform_label})")
    plt.ylabel(f"{fsc_label} ({transform_label})")
    plt.title(f"{sample_name}: Module 2.5 piecewise DG high boundary")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved QC plot: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    gates_dir = Path(GATES_JSON_DIR)
    patient_dir = Path(PATIENT_FCS_DIR)
    out_root = Path(OUTPUT_ROOT) / OUTPUT_SUBFOLDER

    patient_json_dir = out_root / "per_patient_piecewise_json"
    qc_plot_dir = out_root / "qc_piecewise_plots"

    if not patient_dir.exists():
        raise SystemExit(f"Patient FCS directory does not exist:\n  {patient_dir}")

    fcs_files = sorted(patient_dir.glob("*.fcs"))

    if not fcs_files:
        raise SystemExit(f"No .fcs files found in:\n  {patient_dir}")

    gates_json = find_json_by_basename(gates_dir, GATES_BASENAME)

    (
        mmb_window,
        dg_low_cut,
        dg_peak_fluorescence,
        channel_prefs,
        gate_json_top_key
    ) = load_module2_gate_summary(gates_json)

    transform_label = "logicle" if LOGICLE_SOURCE != "asinh-fallback" else "asinh"

    print(f"Python: {sys.executable}")
    print(f"CWD:    {os.getcwd()}")
    print(f"Transform source: {LOGICLE_SOURCE}")
    print(f"Module 2 gates JSON: {gates_json}")
    print(f"Patient FCS dir:     {patient_dir}")
    print(f"Output dir:          {out_root}\n")

    print("Loaded Module 2 gates:")
    print(f"  JSON top key:              {gate_json_top_key}")
    print(f"  MMB window:                {mmb_window}")
    print(f"  DG low_cut:                {dg_low_cut}")
    print(f"  DG peak fluorescence:      {dg_peak_fluorescence}")
    print(f"  N FSC bins:                {N_FSC_BINS}")
    print(f"  Upper multiplier:          {UPPER_MULTIPLIER}")
    print(f"  Step fraction per bin:     {STEP_FRACTION_PER_BIN}")
    print()

    (
        fsc_bin_edges,
        fsc_bin_centres,
        max_fluorescence,
        boundary_points
    ) = build_piecewise_boundary(
        mmb_window=mmb_window,
        dg_peak_fluorescence=dg_peak_fluorescence,
        n_bins=N_FSC_BINS,
        upper_multiplier=UPPER_MULTIPLIER,
    )

    combined = {
        "script": "module_2_5_piecewise_linear_dg_high_boundary.py",
        "purpose": (
            "Generate patient-specific piecewise-linear upper DG fluorescence "
            "boundaries using the global MMB window and DG low_cut from Module 2."
        ),
        "module2_gates_json": str(gates_json),
        "patient_fcs_dir": str(patient_dir),
        "output_dir": str(out_root),
        "transform_source": LOGICLE_SOURCE,
        "global_inputs_from_module2": {
            "mmb_window": [float(mmb_window[0]), float(mmb_window[1])],
            "dg_low_cut": float(dg_low_cut),
            "dg_peak_fluorescence": float(dg_peak_fluorescence),
        },
        "piecewise_settings": {
            "n_fsc_bins": int(N_FSC_BINS),
            "upper_multiplier": float(UPPER_MULTIPLIER),
            "step_fraction_per_bin": float(STEP_FRACTION_PER_BIN),
            "step_absolute": float(dg_peak_fluorescence * STEP_FRACTION_PER_BIN),
            "final_max_fluorescence": float(dg_peak_fluorescence * UPPER_MULTIPLIER),
            "interpretation": (
                "The MMB FSC window is divided into 10 bins. Each bin receives "
                "a progressively increasing maximum allowed fluorescence, starting "
                "from anchor + one step and ending at anchor × 1.25."
            )
        },
        "patients": {}
    }

    fluor_pref = channel_prefs.get("fluor") if isinstance(channel_prefs, dict) else None
    fsc_pref = channel_prefs.get("fsc") if isinstance(channel_prefs, dict) else None

    for f in fcs_files:
        stem = f.stem
        print(f"Processing: {f.name}")

        meta, df = fcsparser.parse(str(f), reformat_meta=True)

        fluor_name = fluor_pref if (fluor_pref in df.columns) else None
        fsc_name = fsc_pref if (fsc_pref in df.columns) else None

        if fluor_name is None or fsc_name is None:
            fluor_auto, fsc_auto = pick_channels(df)
            fluor_name = fluor_name or fluor_auto
            fsc_name = fsc_name or fsc_auto

        if fluor_name not in df.columns or fsc_name not in df.columns:
            raise ValueError(
                f"Could not find required channels in {f.name}.\n"
                f"Needed: fluor={fluor_name}, fsc={fsc_name}\n"
                f"Available: {list(df.columns)}"
            )

        fluor_t = apply_logicle_or_asinh(df[fluor_name].to_numpy(dtype=float))
        fsc_t = apply_logicle_or_asinh(df[fsc_name].to_numpy(dtype=float))

        finite = np.isfinite(fluor_t) & np.isfinite(fsc_t)

        fluor_t = fluor_t[finite]
        fsc_t = fsc_t[finite]

        mmb_low, mmb_high = mmb_window

        # -------------------------------------------------------------
        # First-stage ROI for Module 2.5:
        # Apply only the global MMB FSC window.
        #
        # We do NOT apply the old DG high_cut here, because the whole purpose
        # of this module is to create a patient-specific high boundary.
        # -------------------------------------------------------------
        mmb_roi = (
            (fsc_t >= mmb_low)
            & (fsc_t <= mmb_high)
        )

        fluor_mmb_roi = fluor_t[mmb_roi]
        fsc_mmb_roi = fsc_t[mmb_roi]

        if fluor_mmb_roi.size == 0:
            print("  ⚠️ No events inside global MMB window.")

        # -------------------------------------------------------------
        # Interpolate patient-specific high cut for each event.
        # This allows us to estimate how many events would pass the final
        # Module 2.5 adaptive gate.
        # -------------------------------------------------------------
        patient_high_cut = interpolate_piecewise_high_cut(
            fsc_values=fsc_mmb_roi,
            fsc_bin_centres=fsc_bin_centres,
            max_fluorescence=max_fluorescence
        )

        adaptive_gate_mask = (
            (fluor_mmb_roi >= dg_low_cut)
            & (fluor_mmb_roi <= patient_high_cut)
        )

        n_total_events = int(df.shape[0])
        n_finite_events = int(finite.sum())
        n_mmb_roi_events = int(fluor_mmb_roi.size)
        n_after_piecewise_gate = int(np.sum(adaptive_gate_mask))

        patient_record = {
            "file": f.name,
            "sample": stem,
            "channels": {
                "fluor": fluor_name,
                "fsc": fsc_name
            },
            "transform_source": LOGICLE_SOURCE,
            "module2_inputs": {
                "mmb_window": [float(mmb_window[0]), float(mmb_window[1])],
                "dg_low_cut": float(dg_low_cut),
                "dg_peak_fluorescence": float(dg_peak_fluorescence),
            },
            "piecewise_boundary": {
                "n_fsc_bins": int(N_FSC_BINS),
                "upper_multiplier": float(UPPER_MULTIPLIER),
                "step_fraction_per_bin": float(STEP_FRACTION_PER_BIN),
                "step_absolute": float(dg_peak_fluorescence * STEP_FRACTION_PER_BIN),
                "fsc_bin_edges": [float(x) for x in fsc_bin_edges],
                "fsc_bin_centres": [float(x) for x in fsc_bin_centres],
                "max_fluorescence": [float(x) for x in max_fluorescence],
                "boundary_points": boundary_points,
                "rule": (
                    "For each event, interpolate max_fluorescence from the "
                    "piecewise boundary using its FSC value. Keep event if "
                    "fluorescence >= dg_low_cut and fluorescence <= interpolated "
                    "max_fluorescence."
                )
            },
            "event_counts": {
                "n_total_events": n_total_events,
                "n_finite_events": n_finite_events,
                "n_mmb_roi_events": n_mmb_roi_events,
                "n_after_piecewise_gate": n_after_piecewise_gate,
            },
        }

        combined["patients"][f.name] = patient_record

        patient_json_dir.mkdir(parents=True, exist_ok=True)

        patient_json_path = patient_json_dir / f"{safe_filename(stem)}_piecewise_dg_boundary.json"
        patient_json_path.write_text(
            json.dumps(patient_record, indent=2),
            encoding="utf-8"
        )

        print(f"  Saved patient JSON: {patient_json_path}")
        print(f"  MMB ROI events: {n_mmb_roi_events}")
        print(f"  After piecewise DG gate: {n_after_piecewise_gate}")

        qc_plot_path = qc_plot_dir / f"{safe_filename(stem)}_piecewise_dg_boundary_qc.png"

        save_piecewise_qc_plot(
            out_path=qc_plot_path,
            sample_name=stem,
            fluor_values=fluor_mmb_roi,
            fsc_values=fsc_mmb_roi,
            dg_low_cut=dg_low_cut,
            fsc_bin_centres=fsc_bin_centres,
            max_fluorescence=max_fluorescence,
            fluor_label=fluor_name,
            fsc_label=fsc_name,
            transform_label=transform_label,
        )

        print("  Done.\n")

    out_root.mkdir(parents=True, exist_ok=True)

    combined_json_path = out_root / "piecewise_dg_boundaries_by_patient.json"
    combined_json_path.write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8"
    )

    print(f"\nSaved combined JSON summary:")
    print(f"  {combined_json_path}")

    print(f"\nAll Module 2.5 outputs written to:")
    print(f"  {out_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)