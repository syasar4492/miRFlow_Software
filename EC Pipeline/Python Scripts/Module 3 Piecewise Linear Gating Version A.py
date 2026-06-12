# Module 3_Piecewsie Linear Gating Version A.py
#
# =============================================================================
# PURPOSE
# =============================================================================
# Module 3 sits between:
#   - Module 2: Global Gating
#   - Module 4: MMB Threshold Finder
#   - Module 5: GMM Probabilistic Clustering
#
# This module creates a patient-specific upper fluorescence boundary for the
# two-biomarker EC assay. The boundary is built as a simple piecewise-linear
# gate across the MMB FSC window.
#
# Module 2 provides:
#   - the global MMB FSC window,
#   - the fixed lower DG fluorescence cut,
#   - the DG fluorescence anchor used to scale the upper boundary.
#
# For each patient file, this script:
#   1) Reads the .fcs file and applies the same signal transformation used
#      elsewhere in the pipeline.
#   2) Keeps events within the global MMB FSC window.
#   3) Splits that FSC window into 10 bins.
#   4) Assigns a gradually increasing fluorescence ceiling across the bins.
#   5) Saves patient-level JSON outputs and QC plots.
#
# With a fluorescence anchor of 0.8 and 10 FSC bins, the upper boundary runs
# from 0.82 to 1.00 in steps of 0.02. Module 5 then interpolates along this
# boundary and keeps events that sit between the lower DG cut and the
# patient-specific upper fluorescence boundary.
#
# OUTPUTS
# =============================================================================
# For each patient/sample .fcs file:
#   - one patient-specific JSON file containing the piecewise boundary,
#   - one QC plot showing the boundary on the transformed scatter plot.
#
# The module also exports:
#   - one combined JSON file containing all patient-specific boundaries,
#   - one collated QC overview PNG containing all patient/sample QC plots.
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
GATES_JSON_DIR = r" "
GATES_BASENAME = "gate_summary_by_type"

# Folder containing patient/sample .fcs files
PATIENT_FCS_DIR = r" "

# Output root for Module 3 results
OUTPUT_ROOT = r" "
OUTPUT_SUBFOLDER = "Piecewise Linear Plots"


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
MAX_POINTS_PER_OVERVIEW_PANEL = 12000
MAX_PANELS_PER_OVERVIEW_ROW = 4


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
    Apply Logicle if available, otherwise use an asinh fallback.

    The same transformation settings are used across the pipeline:
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

    The loader expects the flattened JSON structure exported by Module 2.
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

    # Accept dg_window as a fallback if the JSON contains that field.
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
    Build the piecewise-linear upper fluorescence boundary.

    The MMB FSC window is split into equal bins. Each bin receives a maximum
    fluorescence value, increasing from anchor + one step to anchor × multiplier.

    Returns the bin edges, bin centres, upper fluorescence values and paired
    boundary points for JSON export.
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
    Return the maximum allowed fluorescence for each event FSC value.
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
    Save one patient-level QC scatter plot with the lower DG cut and upper boundary.
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

    plt.axvline(dg_low_cut, color="crimson", ls="--", lw=1, label="DG low_cut")

    # x-axis is fluorescence and y-axis is FSC.
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
    plt.title(f"{sample_name}: Piecewise DG high boundary")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved QC plot: {out_path}")


def save_piecewise_qc_overview_png(
    out_path: Path,
    panels: List[Dict],
    dg_low_cut: float,
    fsc_bin_centres: np.ndarray,
    max_fluorescence: np.ndarray,
    fluor_label: str,
    fsc_label: str,
    transform_label: str,
):
    """
    Save one collated PNG containing all per-patient Module 3/3 QC plots.

    Each panel shows:
      - patient events after applying the global MMB FSC window
      - the global DG low_cut as a vertical dashed red line
      - the shared piecewise DG high boundary

    This gives a single overview image for quick visual review across all
    patients/samples.
    """
    if not panels:
        print("No QC panels available for overview plot. Skipping collated PNG.")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(panels)
    ncols = min(MAX_PANELS_PER_OVERVIEW_ROW, n)
    nrows = int(math.ceil(n / ncols))

    fig_w = max(12, 4.2 * ncols)
    fig_h = max(8, 3.6 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        squeeze=False
    )

    for idx, panel in enumerate(panels):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        x = panel["fluor"]
        y = panel["fsc"]

        ax.scatter(x, y, s=1.4, alpha=0.35)
        ax.axvline(dg_low_cut, color="crimson", ls="--", lw=1)
        ax.plot(
            max_fluorescence,
            fsc_bin_centres,
            color="black",
            ls="-",
            lw=1.2,
            marker="o",
            markersize=2,
        )

        ax.set_title(panel["sample"], fontsize=9)
        ax.set_xlabel(f"{fluor_label} ({transform_label})", fontsize=8)
        ax.set_ylabel(f"{fsc_label} ({transform_label})", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    fig.suptitle("All patients: Piecewise DG high-boundary QC overview", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved collated QC overview plot: {out_path}")
    return out_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    gates_dir = Path(GATES_JSON_DIR)
    patient_dir = Path(PATIENT_FCS_DIR)
    out_root = Path(OUTPUT_ROOT) / OUTPUT_SUBFOLDER

    patient_json_dir = out_root / "per_patient_piecewise_json"
    qc_plot_dir = out_root / "qc_piecewise_plots"
    qc_overview_panels = []

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
        "script": "Module 3_Piecewsie Linear Gating Version A.py",
        "purpose": (
            "Generate patient-specific piecewise-linear upper DG fluorescence "
            "boundaries for the two-biomarker EC assay."
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
                "a progressively increasing upper fluorescence limit."
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
        # Initial ROI for this module: keep events inside the global MMB
        # FSC window. The upper fluorescence boundary is created below.
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
        # Interpolate the patient-specific high cut for each event.
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
                    "For each event, interpolate max_fluorescence from its FSC "
                    "value. Keep the event if fluorescence is between dg_low_cut "
                    "and the interpolated upper boundary."
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

        # Store downsampled data for the collated QC overview PNG.
        if fluor_mmb_roi.size > MAX_POINTS_PER_OVERVIEW_PANEL:
            rng = np.random.default_rng(0)
            pick = rng.choice(
                fluor_mmb_roi.size,
                size=MAX_POINTS_PER_OVERVIEW_PANEL,
                replace=False
            )
            fluor_panel = fluor_mmb_roi[pick]
            fsc_panel = fsc_mmb_roi[pick]
        else:
            fluor_panel = fluor_mmb_roi
            fsc_panel = fsc_mmb_roi

        qc_overview_panels.append({
            "sample": stem,
            "fluor": fluor_panel,
            "fsc": fsc_panel,
        })

        print("  Done.\n")

    # Save one collated PNG containing all QC plots.
    if qc_overview_panels:
        overview_plot_path = qc_plot_dir / "ALL_PATIENTS_piecewise_dg_boundary_qc_overview.png"
        save_piecewise_qc_overview_png(
            out_path=overview_plot_path,
            panels=qc_overview_panels,
            dg_low_cut=dg_low_cut,
            fsc_bin_centres=fsc_bin_centres,
            max_fluorescence=max_fluorescence,
            fluor_label=fluor_pref or "Fluor",
            fsc_label=fsc_pref or "FSC",
            transform_label=transform_label,
        )

    out_root.mkdir(parents=True, exist_ok=True)

    combined_json_path = out_root / "piecewise_dg_boundaries_by_patient.json"
    combined_json_path.write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8"
    )

    print(f"\nSaved combined JSON summary:")
    print(f"  {combined_json_path}")

    print(f"\nAll Module 3 outputs written to:")
    print(f"  {out_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)