# gmm_patient_clustering_4state_per_bead.py
#
# =============================================================================
# WHAT THIS SCRIPT DOES
# =============================================================================
# For each patient .fcs file:
#   0) Load + auto-pick channels
#   1) Apply Logicle transform with asinh fallback
#   2) Apply ROI gating using control-derived global windows
#   3) Assign bead size deterministically using FSC thresholds
#   4) Fit 1D GMM on fluorescence within each bead population:
#        - q20 = only
#        - q40 = sandwich_v1
#        - q60 = sandwich_v2
#        - q80 = sandwich_v3
#   5) Outputs:
#        - per-patient 2D cluster plots
#        - per-patient JSON summaries
#        - one overview grid PNG
#        - one Excel workbook with:
#             Sheet 1: patient_gmm_summary_12clusters
#             Sheet 2: Diagnostic Metrics
#
# IMPORTANT UPDATE: NORMALISED DIAGNOSTIC COFACTORS
# =============================================================================
# The raw expected_sandwich_total values are useful, but absolute event counts can
# be influenced by how many events entered each bead population. Therefore this
# version also calculates, for each bead population:
#
#   normalised_total_sandwich_events (%) =
#       100 * expected_sandwich_total / (expected_sandwich_total + expected_only)
#
# This expresses the sandwich-positive signal as a percentage of the informative
# events within that bead population. The Diagnostic Metrics sheet then calculates
# MMB ratios using these normalised values instead of raw expected_sandwich_total
# counts.
#
# Example:
#   MMB 3:6 Diagnostic_Ratio =
#       normalised_sandwich_events (%) for MMB-3 /
#       normalised_sandwich_events (%) for MMB-6
#
# This makes the diagnostic cofactor less dependent on bead abundance/event-count
# differences and more reflective of relative sandwich formation per bead class.
# =============================================================================

from __future__ import annotations

# =============================================================================
# PATHS
# =============================================================================

JSON_DIR = r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Piecewise Testing\JSON Files"
GATES_BASENAME = "gate_summary_by_type"
THRESH_BASENAME = "mmb_fsc_thresholds"
PIECEWISE_BASENAME = "piecewise_dg_boundaries_by_patient"
PATIENT_FCS_DIR = r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Raw Flow Data\06.16.2025_20samples_optimized protocol"

OUTPUT_ROOT = r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Visual Flow Data\20 samples optimised protocol"
OUTPUT_SUBFOLDER = "GMM_Clustering_12clusters_piecewise DG Window"

# =============================================================================
# TUNABLE HYPERPARAMETERS
# =============================================================================

LOW_Q = 20       # only
LOWMID_Q = 40    # sandwich_v1
MIDHIGH_Q = 60   # sandwich_v2
HIGH_Q = 80      # sandwich_v3

MIN_EVENTS_4COMP = 1000
MIN_EVENTS_2COMP = 300

MAX_POINTS_PER_PLOT = 30000
MAX_POINTS_PER_PANEL_GRID = 12000

# =============================================================================
# DIAGNOSTIC THRESHOLD PLACEHOLDERS
# =============================================================================
# Fill these in later when you know the diagnostic ratio thresholds.
#
# The PTB/TB pipeline now uses one three-way ratio structure per patient,
# calculated in three variants using the normalised sandwich_v1, sandwich_v2,
# and sandwich_v3 event percentages.
#
# Format:
#   "3:6:8 using normalised_sandwich_v1_events (%)": {"mean": 1.20, "sd": 0.15}
#
# Diagnostic_Ratio formula for each variant:
#   MMB-3 / (MMB-6 + MMB-8)
#
# where MMB-3, MMB-6 and MMB-8 are the relevant normalised sandwich-state
# percentages for that row.
#
# Triage recommendation logic:
#   within ±2 SD  = "Recommend triaging to secondary care"     -> red   -> score 1
#   outside ±2 SD = "Do not recommend triaging to secondary care" -> green -> score 0
#
# Keep mean/sd as None for now. The Diagnostic Metrics sheet will leave
# EC_threshold, EC_probability and score blank until values are entered.
# =============================================================================

EC_THRESHOLDS = {
    "3:6:8 using normalised_sandwich_v1_events (%)": {"mean": 0.9, "sd": 0.15},
    "3:6:8 using normalised_sandwich_v2_events (%)": {"mean": 0.76, "sd": 0.1},
    "3:6:8 using normalised_sandwich_v3_events (%)": {"mean": 1.2, "sd": 0.25},
}

RATIO_ORDER = [
    ("3:6:8 using normalised_sandwich_v1_events (%)", "normalised_sandwich_v1_events (%)"),
    ("3:6:8 using normalised_sandwich_v2_events (%)", "normalised_sandwich_v2_events (%)"),
    ("3:6:8 using normalised_sandwich_v3_events (%)", "normalised_sandwich_v3_events (%)"),
]

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import os, sys, json, math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fcsparser

try:
    from sklearn.mixture import GaussianMixture
except Exception as e:
    raise SystemExit("scikit-learn is required. Install with: pip install scikit-learn") from e

try:
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter
except Exception as e:
    raise SystemExit("openpyxl is required. Install with: pip install openpyxl") from e

try:
    from flowutils.transforms import logicle as logicle_transform
    LOGICLE_SOURCE = "flowutils.transforms.logicle"
except Exception:
    logicle_transform = None
    LOGICLE_SOURCE = "asinh-fallback"


# =============================================================================
# TRANSFORM
# =============================================================================

def apply_logicle_or_asinh(arr: np.ndarray) -> np.ndarray:
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
    cols = list(df.columns)

    fsc = first_match(cols, ["FSC-H"]) or first_match(cols, ["FSC-A", "FSC"])

    fluor = (
        first_match(cols, ["FL1-H"])
        or first_match(cols, ["B525-H", "BL1-H", "FITC-H", "GFP-H"])
        or first_match(cols, ["B530-H", "B515-H", "530/30", "525/50"])
        or first_match(cols, ["FL1-A", "BL1-A", "B525-A", "FITC-A", "GFP-A"])
    )

    if fsc is None:
        raise ValueError(f"No FSC channel found in {cols}")

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
                raise ValueError(f"No fluorescence channel found in {cols}")
            fluor = a_cands[0]

    return fluor, fsc


# =============================================================================
# JSON LOADING
# =============================================================================

def find_json_by_basename(folder: Path, basename: str) -> Path:
    if not folder.exists():
        raise FileNotFoundError(f"JSON folder does not exist: {folder}")

    exact = folder / f"{basename}.json"
    if exact.exists():
        return exact

    cands = sorted(folder.glob("*.json"))
    if not cands:
        raise FileNotFoundError(f"No .json files found in: {folder}")

    for p in cands:
        if p.stem.lower().startswith(basename.lower()):
            return p

    for p in cands:
        if basename.lower() in p.stem.lower():
            return p

    raise FileNotFoundError(
        f"Could not find a JSON matching basename '{basename}' in {folder}.\n"
        f"Available: {[p.name for p in cands]}"
    )


def load_global_windows(gates_json_path: Path) -> Tuple[Tuple[float, float], float, Dict[str, str]]:
    """
    Load Module 2 gate_summary_by_type.json.

    Updated piecewise-gating structure:
      - Module 2 now exports:
          mmb_window
          dg_low_cut
          dg_peak_fluorescence
          channels
      - Module 2 no longer exports a global dg_high_cut because the high
        fluorescence boundary is now patient-specific and loaded from
        piecewise_dg_boundaries_by_patient.json.

    Backward compatibility:
      - If an older JSON contains dg_window = [low, high], this function will
        use dg_window[0] as dg_low_cut.
    """
    d = json.loads(gates_json_path.read_text(encoding="utf-8"))

    if not isinstance(d, dict) or len(d) == 0:
        raise ValueError(f"Unexpected gates JSON structure in {gates_json_path.name}")

    first_key = next(iter(d.keys()))
    block = d[first_key]

    if "mmb_window" not in block:
        raise KeyError(f"Missing mmb_window under top key '{first_key}' in {gates_json_path.name}")

    mmb = block["mmb_window"]

    if mmb is None:
        raise ValueError(f"mmb_window is None in {gates_json_path.name}")

    dg_low_cut = block.get("dg_low_cut", None)

    # Backward compatibility for older Module 2 JSON files.
    if dg_low_cut is None and block.get("dg_window") is not None:
        dg_window = block.get("dg_window")
        dg_low_cut = dg_window[0]

    if dg_low_cut is None:
        raise KeyError(
            f"Missing dg_low_cut under top key '{first_key}' in {gates_json_path.name}.\n"
            "Module 4 now expects Module 2 to export dg_low_cut only; the high cut "
            "is loaded from the piecewise patient-specific JSON."
        )

    ch = block.get("channels", {})

    return (float(mmb[0]), float(mmb[1])), float(dg_low_cut), ch


def load_piecewise_boundaries(piecewise_json_path: Path) -> Dict:
    """
    Load the combined Module 2.5 JSON file:
      piecewise_dg_boundaries_by_patient.json

    Expected structure:
      {
        "patients": {
          "Patient_A.fcs": {
            "piecewise_boundary": {
              "fsc_bin_centres": [...],
              "max_fluorescence": [...]
            }
          }
        }
      }

    The fsc_bin_centres and max_fluorescence arrays are used to interpolate
    a patient-specific DG high fluorescence limit for each event in Module 4.
    """
    d = json.loads(piecewise_json_path.read_text(encoding="utf-8"))

    if not isinstance(d, dict) or "patients" not in d:
        raise ValueError(
            f"Unexpected piecewise JSON structure in {piecewise_json_path.name}. "
            "Expected a top-level 'patients' dictionary."
        )

    patients = d["patients"]

    if not isinstance(patients, dict) or len(patients) == 0:
        raise ValueError(f"No patient records found in {piecewise_json_path.name}")

    return d


def get_piecewise_record_for_file(piecewise_data: Dict, fcs_file: Path) -> Dict:
    """
    Retrieve the patient-specific piecewise boundary record for a given .fcs file.

    Matching is intentionally flexible:
      1) exact filename match, e.g. PTB_A1.fcs
      2) exact stem match, e.g. PTB_A1
      3) case-insensitive filename/stem match

    This avoids failure if the combined JSON uses slightly different key formats.
    """
    patients = piecewise_data.get("patients", {})

    if fcs_file.name in patients:
        return patients[fcs_file.name]

    if fcs_file.stem in patients:
        return patients[fcs_file.stem]

    name_upper = fcs_file.name.upper()
    stem_upper = fcs_file.stem.upper()

    for key, record in patients.items():
        key_upper = str(key).upper()
        key_stem_upper = Path(str(key)).stem.upper()

        if key_upper == name_upper or key_stem_upper == stem_upper:
            return record

    raise KeyError(
        f"Could not find piecewise DG boundary record for {fcs_file.name}.\n"
        f"Available patient keys include: {list(patients.keys())[:10]}"
    )


def interpolate_piecewise_high_cut(
    fsc_values: np.ndarray,
    fsc_bin_centres: np.ndarray,
    max_fluorescence: np.ndarray,
) -> np.ndarray:
    """
    Interpolate the patient-specific DG high fluorescence boundary.

    For each event's FSC value, this returns the maximum allowed fluorescence
    at that FSC position based on the Module 2.5 piecewise boundary.

    Events are then retained if:
      fluor >= dg_low_cut
      fluor <= interpolated_piecewise_high_cut
    """
    return np.interp(
        fsc_values,
        fsc_bin_centres,
        max_fluorescence,
        left=max_fluorescence[0],
        right=max_fluorescence[-1],
    )


def load_bead_boundaries(thresh_json_path: Path) -> Tuple[float, float, str]:
    d = json.loads(thresh_json_path.read_text(encoding="utf-8"))

    rb = d.get("recommended_boundaries", {})
    b36 = rb.get("boundary_3_6", None)
    b68 = rb.get("boundary_6_8", None)

    if b36 is None or b68 is None:
        raise KeyError(f"Could not find boundary_3_6 / boundary_6_8 in {thresh_json_path.name}")

    fsc_channel = d.get("fsc_channel", "FSC-H")

    return float(b36), float(b68), str(fsc_channel)


# =============================================================================
# GMM UTILITIES
# =============================================================================

def ensure_strictly_increasing(means: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = means.astype(float).copy()
    for i in range(1, len(m)):
        if m[i] <= m[i - 1]:
            m[i] = m[i - 1] + eps
    return m


@dataclass
class BeadFitResult:
    bead: int
    n: int
    model_kind: str
    means: List[float]
    p_only: Optional[np.ndarray] = None
    p_sandwich_v1: Optional[np.ndarray] = None
    p_sandwich_v2: Optional[np.ndarray] = None
    p_sandwich_v3: Optional[np.ndarray] = None
    warn: Optional[str] = None


def fit_gmm_4state_or_fallback(fluor: np.ndarray, bead: int) -> BeadFitResult:
    """
    Fit 4-component 1D GMM if enough events; else fallback to 2-component; else skip.

    4-component state order by ascending mean:
      0 = only
      1 = sandwich_v1
      2 = sandwich_v2
      3 = sandwich_v3

    2-component fallback:
      0 = only
      1 = sandwich_v1
      sandwich_v2 = 0
      sandwich_v3 = 0
    """
    x = fluor[np.isfinite(fluor)].astype(float)
    n = int(x.size)

    if n < MIN_EVENTS_2COMP:
        return BeadFitResult(
            bead=bead,
            n=n,
            model_kind="skipped",
            means=[],
            warn=f"Bead {bead}: too few events for GMM (n={n} < {MIN_EVENTS_2COMP})."
        )

    X = x.reshape(-1, 1)

    if n < MIN_EVENTS_4COMP:
        q33 = float(np.percentile(x, 33))
        q67 = float(np.percentile(x, 67))
        init_means = ensure_strictly_increasing(np.array([q33, q67], dtype=float))

        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            reg_covar=1e-6,
            max_iter=300,
            n_init=1,
            means_init=init_means.reshape(-1, 1),
            random_state=0
        )

        gmm.fit(X)
        probs = gmm.predict_proba(X)
        means = gmm.means_.flatten()

        order = np.argsort(means)
        means_sorted = means[order]

        p_only = probs[:, order[0]]
        p_sandwich_v1 = probs[:, order[1]]
        p_sandwich_v2 = np.zeros_like(p_only)
        p_sandwich_v3 = np.zeros_like(p_only)

        return BeadFitResult(
            bead=bead,
            n=n,
            model_kind="gmm2",
            means=means_sorted.tolist(),
            p_only=p_only,
            p_sandwich_v1=p_sandwich_v1,
            p_sandwich_v2=p_sandwich_v2,
            p_sandwich_v3=p_sandwich_v3,
            warn=(
                f"Bead {bead}: used 2-comp fallback "
                f"(n={n} < {MIN_EVENTS_4COMP}). sandwich_v2 and sandwich_v3 set to 0."
            )
        )

    q1 = float(np.percentile(x, LOW_Q))
    q2 = float(np.percentile(x, LOWMID_Q))
    q3 = float(np.percentile(x, MIDHIGH_Q))
    q4 = float(np.percentile(x, HIGH_Q))

    init_means = ensure_strictly_increasing(np.array([q1, q2, q3, q4], dtype=float))

    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        reg_covar=1e-6,
        max_iter=500,
        n_init=1,
        means_init=init_means.reshape(-1, 1),
        random_state=0
    )

    gmm.fit(X)
    probs = gmm.predict_proba(X)
    means = gmm.means_.flatten()

    order = np.argsort(means)
    means_sorted = means[order]

    p_only = probs[:, order[0]]
    p_sandwich_v1 = probs[:, order[1]]
    p_sandwich_v2 = probs[:, order[2]]
    p_sandwich_v3 = probs[:, order[3]]

    return BeadFitResult(
        bead=bead,
        n=n,
        model_kind="gmm4",
        means=means_sorted.tolist(),
        p_only=p_only,
        p_sandwich_v1=p_sandwich_v1,
        p_sandwich_v2=p_sandwich_v2,
        p_sandwich_v3=p_sandwich_v3,
        warn=None
    )


# =============================================================================
# PLOTTING
# =============================================================================

def safe_filename(name: str) -> str:
    return "".join(c if c not in r'<>:"/\|?*' else "_" for c in name)


def downsample_xy(x: np.ndarray, y: np.ndarray, labels: np.ndarray, max_points: int, seed: int = 0):
    n = x.size
    if n <= max_points:
        return x, y, labels

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)

    return x[idx], y[idx], labels[idx]


def make_color_map(cluster_names: List[str]) -> Dict[str, str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    cmap = {}
    for i, name in enumerate(cluster_names):
        cmap[name] = cycle[i % len(cycle)]

    return cmap


def get_cluster_names() -> List[str]:
    states = ["only", "sandwich_v1", "sandwich_v2", "sandwich_v3"]
    return [f"MMB-{b} {s}" for b in (3, 6, 8) for s in states]


def plot_patient_clusters(
    out_path: Path,
    fluor: np.ndarray,
    fsc: np.ndarray,
    cluster_labels: np.ndarray,
    fluor_label: str,
    fsc_label: str,
    title: str,
    dg_low_cut: float,
    piecewise_fsc_bin_centres: np.ndarray,
    piecewise_max_fluorescence: np.ndarray,
    mmb_window: Tuple[float, float],
    b36: float,
    b68: float,
):
    x, y, lab = downsample_xy(fluor, fsc, cluster_labels, MAX_POINTS_PER_PLOT, seed=0)

    cluster_names = get_cluster_names()
    cmap = make_color_map(cluster_names)

    plt.figure(figsize=(7.5, 5.2))

    for cn in cluster_names:
        m = (lab == cn)
        if np.any(m):
            plt.scatter(x[m], y[m], s=2, alpha=0.45, label=cn, color=cmap[cn])

    mmb_low, mmb_high = mmb_window

    # Global lower DG fluorescence boundary from Module 2.
    plt.axvline(dg_low_cut, ls="--", lw=1, label="DG low_cut")

    # Patient-specific piecewise DG high boundary from Module 2.5.
    # Plot orientation:
    #   x-axis = fluorescence
    #   y-axis = FSC
    # Therefore we plot x = max_fluorescence and y = FSC bin centres.
    plt.plot(
        piecewise_max_fluorescence,
        piecewise_fsc_bin_centres,
        ls="-",
        lw=1.5,
        marker="o",
        markersize=2.5,
        label="Piecewise DG high boundary"
    )

    plt.axhline(mmb_low, ls="--", lw=1)
    plt.axhline(mmb_high, ls="--", lw=1)
    plt.axhline(b36, ls=":", lw=1)
    plt.axhline(b68, ls=":", lw=1)

    plt.xlabel(fluor_label)
    plt.ylabel(fsc_label)
    plt.title(title)
    plt.tight_layout()

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        markerscale=2
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_grid_overview(out_path: Path, panels: List[Dict], fluor_label: str, fsc_label: str, suptitle: str):
    n = len(panels)
    if n == 0:
        return

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig_w = max(10, 3.4 * ncols)
    fig_h = max(8, 3.0 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    cluster_names = get_cluster_names()
    cmap = make_color_map(cluster_names)

    for idx, p in enumerate(panels):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        x = p["x"]
        y = p["y"]
        lab = p["labels"]

        if x.size > MAX_POINTS_PER_PANEL_GRID:
            rng = np.random.default_rng(0)
            pick = rng.choice(x.size, size=MAX_POINTS_PER_PANEL_GRID, replace=False)
            x = x[pick]
            y = y[pick]
            lab = lab[pick]

        for cn in cluster_names:
            m = (lab == cn)
            if np.any(m):
                ax.scatter(x[m], y[m], s=1.3, alpha=0.45, color=cmap[cn])

        ax.set_title(p["name"], fontsize=9)
        ax.set_xlabel(fluor_label, fontsize=8)
        ax.set_ylabel(fsc_label, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# =============================================================================
# DIAGNOSTIC METRICS
# =============================================================================

def classify_triage_recommendation(ratio_label: str, diagnostic_ratio):
    """
    Returns:
      threshold_display, EC_probability/recommendation, risk_score

    If EC_THRESHOLDS are not yet filled in, returns blanks.

    Binary triage logic once thresholds are provided:
      - within ±2 SD  -> Recommend triaging to secondary care, score 1
      - outside ±2 SD -> Do not recommend triaging to secondary care, score 0
    """
    block = EC_THRESHOLDS.get(ratio_label, {"mean": None, "sd": None})
    mean = block.get("mean")
    sd = block.get("sd")

    if mean is None or sd is None:
        return "", "", ""

    try:
        mean = float(mean)
        sd = float(sd)
        diagnostic_ratio = float(diagnostic_ratio)
    except Exception:
        return "", "", ""

    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        return f"{mean} ± {sd}", "", ""

    z = abs(diagnostic_ratio - mean) / sd
    threshold_display = f"{mean} ± {sd}"

    if z <= 2:
        return threshold_display, "Recommend triaging to secondary care", 1

    return threshold_display, "Do not recommend triaging to secondary care", 0


def calculate_normalised_percent(numerator, expected_sandwich_total, expected_only):
    """
    Calculate a normalised sandwich signal for one bead population.

    For state-specific columns, the denominator must be state-specific:

      normalised_sandwich_v1_events (%) =
          100 * expected_sandwich_v1 / (expected_sandwich_v1 + expected_only)

      normalised_sandwich_v2_events (%) =
          100 * expected_sandwich_v2 / (expected_sandwich_v2 + expected_only)

      normalised_sandwich_v3_events (%) =
          100 * expected_sandwich_v3 / (expected_sandwich_v3 + expected_only)

    For the total sandwich column, numerator = expected_sandwich_total, so this
    same function gives:

      normalised_total_sandwich_events (%) =
          100 * expected_sandwich_total / (expected_sandwich_total + expected_only)

    The expected_sandwich_total argument is retained for compatibility with the
    existing function calls, but it is not used in the denominator for the
    state-specific calculations.

    Returns np.nan if the calculation is not possible.
    """
    try:
        numerator = float(numerator)
        expected_only = float(expected_only)
    except Exception:
        return np.nan

    denom = numerator + expected_only

    if (
        not np.isfinite(numerator)
        or not np.isfinite(expected_only)
        or denom <= 0
    ):
        return np.nan

    return 100.0 * numerator / denom

def build_diagnostic_rows(df_master: pd.DataFrame, sample_order: List[str]) -> pd.DataFrame:
    """
    Builds Diagnostic Metrics sheet.

    For each patient:
      patient header row
      three 3:6:8 Diagnostic_Ratio rows
      overall risk score row
      blank spacer row

    The three rows are:
      1) 3:6:8 using normalised_sandwich_v1_events (%)
      2) 3:6:8 using normalised_sandwich_v2_events (%)
      3) 3:6:8 using normalised_sandwich_v3_events (%)

    For each row:
      Diagnostic_Ratio = MMB-3 / (MMB-6 + MMB-8)

    using the relevant state-specific normalised sandwich percentage.
    """
    rows = []

    for sample in sample_order:
        sample_df = df_master[df_master["sample"] == sample].copy()

        if sample_df.empty:
            continue

        rows.append({
            "file name": sample,
            "Diagnostic_Ratio": "",
            "EC_threshold": "",
            "EC_probability": "",
            "Overall Risk Score (3)": "",
            "_row_type": "patient_header"
        })

        risk_scores = []

        for ratio_label, normalised_column in RATIO_ORDER:
            bead_values = {}

            for bead in (3, 6, 8):
                bead_row = sample_df[sample_df["bead"].astype(str) == str(bead)]
                if bead_row.empty or normalised_column not in bead_row.columns:
                    bead_values[bead] = np.nan
                else:
                    bead_values[bead] = float(bead_row.iloc[0][normalised_column])

            numerator = bead_values.get(3, np.nan)
            denominator = bead_values.get(6, np.nan) + bead_values.get(8, np.nan)

            if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
                diagnostic_ratio = ""
            else:
                diagnostic_ratio = numerator / denominator

            threshold_display, probability, score = classify_triage_recommendation(
                ratio_label,
                diagnostic_ratio
            )

            if score != "":
                risk_scores.append(int(score))

            rows.append({
                "file name": ratio_label,
                "Diagnostic_Ratio": diagnostic_ratio,
                "EC_threshold": threshold_display,
                "EC_probability": probability,
                "Overall Risk Score (3)": score,
                "_row_type": "ratio"
            })

        overall_score = sum(risk_scores) if risk_scores else ""

        rows.append({
            "file name": "",
            "Diagnostic_Ratio": "",
            "EC_threshold": "",
            "EC_probability": "Overall Risk Score",
            "Overall Risk Score (3)": overall_score,
            "_row_type": "overall_score"
        })

        rows.append({
            "file name": "",
            "Diagnostic_Ratio": "",
            "EC_threshold": "",
            "EC_probability": "",
            "Overall Risk Score (3)": "",
            "_row_type": "blank"
        })

    return pd.DataFrame(rows)


def save_excel_workbook(out_path: Path, df_master: pd.DataFrame, df_diag: pd.DataFrame):
    """
    Saves one .xlsx workbook with:
      1) patient_gmm_summary_12clusters
      2) Diagnostic Metrics

    Applies colour formatting to EC_probability:
      Recommend triaging to secondary care = red
      Do not recommend triaging to secondary care = green
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    diag_export = df_diag.drop(columns=["_row_type"], errors="ignore")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_master.to_excel(writer, index=False, sheet_name="patient_gmm_summary_12clusters")
        diag_export.to_excel(writer, index=False, sheet_name="Diagnostic Metrics")

    wb = load_workbook(out_path)

    header_fill = PatternFill(start_color="D9EAF7", end_color="D9EAF7", fill_type="solid")
    grey_fill = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")

    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")

        for col_idx in range(1, ws.max_column + 1):
            col_letter = get_column_letter(col_idx)
            max_len = 12
            for cell in ws[col_letter]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 35)

    ws_diag = wb["Diagnostic Metrics"]

    for excel_row_idx, row in enumerate(df_diag.itertuples(index=False), start=2):
        row_type = getattr(row, "_row_type", "")

        if row_type in ["patient_header", "overall_score"]:
            for col_idx in range(1, ws_diag.max_column + 1):
                cell = ws_diag.cell(row=excel_row_idx, column=col_idx)
                cell.font = Font(bold=True)
                cell.fill = grey_fill

        probability = str(ws_diag.cell(row=excel_row_idx, column=4).value).lower()

        prob_cell = ws_diag.cell(row=excel_row_idx, column=4)
        score_cell = ws_diag.cell(row=excel_row_idx, column=5)

        if probability == "recommend triaging to secondary care":
            prob_cell.fill = red_fill
            score_cell.fill = red_fill
        elif probability == "do not recommend triaging to secondary care":
            prob_cell.fill = green_fill
            score_cell.fill = green_fill

    wb.save(out_path)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    json_dir = Path(JSON_DIR)

    gates_json = find_json_by_basename(json_dir, GATES_BASENAME)
    thresh_json = find_json_by_basename(json_dir, THRESH_BASENAME)
    piecewise_json = find_json_by_basename(json_dir, PIECEWISE_BASENAME)

    mmb_window, dg_low_cut, ch = load_global_windows(gates_json)
    b36, b68, fsc_ch_from_thresh = load_bead_boundaries(thresh_json)
    piecewise_data = load_piecewise_boundaries(piecewise_json)

    patient_dir = Path(PATIENT_FCS_DIR)

    if not patient_dir.exists():
        raise SystemExit(f"Patient directory does not exist:\n  {patient_dir}")

    fcs_files = sorted(patient_dir.glob("*.fcs"))

    if not fcs_files:
        raise SystemExit(f"No .fcs files found in:\n  {patient_dir}")

    out_root = Path(OUTPUT_ROOT) / OUTPUT_SUBFOLDER
    plots_dir = out_root / "plots_per_patient"
    per_patient_json_dir = out_root / "per_patient_json"

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Python: {sys.executable}")
    print(f"CWD:    {os.getcwd()}")
    print(f"Transform source: {LOGICLE_SOURCE}")
    print(f"Gates JSON: {gates_json}")
    print(f"Thresh JSON: {thresh_json}")
    print(f"Piecewise JSON: {piecewise_json}")
    print(f"Patient dir: {patient_dir}")
    print(f"Output dir:  {out_root}\n")

    print("Loaded global windows:")
    print(f"  MMB window (FSC): {mmb_window}")
    print(f"  DG low_cut (FL):  {dg_low_cut}")
    print("Loaded bead boundaries:")
    print(f"  boundary_3_6: {b36}")
    print(f"  boundary_6_8: {b68}\n")

    fluor_pref = ch.get("fluor") if isinstance(ch, dict) else None
    fsc_pref = ch.get("fsc") if isinstance(ch, dict) else None

    master_rows = []
    grid_panels = []
    sample_order = []

    for f in fcs_files:
        stem = f.stem
        print(f"Processing: {f.name}")

        meta, df = fcsparser.parse(str(f), reformat_meta=True)

        fluor_name = fluor_pref if (fluor_pref in df.columns) else None
        fsc_name = fsc_pref if (fsc_pref in df.columns) else None

        if fluor_name is None or fsc_name is None:
            fluor_name_auto, fsc_name_auto = pick_channels(df)
            fluor_name = fluor_name or fluor_name_auto
            fsc_name = fsc_name or fsc_name_auto

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
        # Patient-specific DG high boundary from Module 2.5
        # -------------------------------------------------------------
        # Module 2 now provides only:
        #   - global MMB FSC window
        #   - global DG low_cut
        #
        # The upper DG fluorescence boundary is no longer a single global value.
        # Instead, Module 2.5 provides fsc_bin_centres and max_fluorescence for
        # each patient. We interpolate those points so every event receives its
        # own allowed DG high cut based on its FSC value.
        # -------------------------------------------------------------
        piecewise_record = get_piecewise_record_for_file(piecewise_data, f)
        piecewise_boundary = piecewise_record.get("piecewise_boundary", {})

        fsc_bin_centres = np.array(
            piecewise_boundary.get("fsc_bin_centres", []),
            dtype=float
        )
        max_fluorescence = np.array(
            piecewise_boundary.get("max_fluorescence", []),
            dtype=float
        )

        if fsc_bin_centres.size == 0 or max_fluorescence.size == 0:
            raise ValueError(
                f"Missing fsc_bin_centres/max_fluorescence in piecewise JSON for {f.name}"
            )

        if fsc_bin_centres.size != max_fluorescence.size:
            raise ValueError(
                f"Piecewise boundary length mismatch for {f.name}: "
                f"{fsc_bin_centres.size} FSC centres vs {max_fluorescence.size} fluorescence limits"
            )

        piecewise_dg_high_cut = interpolate_piecewise_high_cut(
            fsc_values=fsc_t,
            fsc_bin_centres=fsc_bin_centres,
            max_fluorescence=max_fluorescence,
        )

        roi = (
            (fsc_t >= mmb_low)
            & (fsc_t <= mmb_high)
            & (fluor_t >= dg_low_cut)
            & (fluor_t <= piecewise_dg_high_cut)
        )

        fluor_roi = fluor_t[roi]
        fsc_roi = fsc_t[roi]

        if fluor_roi.size == 0:
            warn = "ROI empty after applying global gates."
            print(f"  ⚠️ {warn}")

            per_summary = {
                "file": f.name,
                "sample": stem,
                "channels": {"fluor": fluor_name, "fsc": fsc_name},
                "windows": {
                    "mmb_window": list(mmb_window),
                    "dg_low_cut": float(dg_low_cut),
                    "piecewise_dg_high_boundary": {
                        "fsc_bin_centres": [float(x) for x in fsc_bin_centres],
                        "max_fluorescence": [float(x) for x in max_fluorescence],
                    }
                },
                "boundaries": {"boundary_3_6": b36, "boundary_6_8": b68},
                "n_total_events": int(df.shape[0]),
                "n_finite": int(finite.sum()),
                "n_roi_events": 0,
                "warnings": [warn],
                "per_bead": {},
            }

            per_patient_json_dir.mkdir(parents=True, exist_ok=True)
            (per_patient_json_dir / f"{safe_filename(stem)}_summary.json").write_text(
                json.dumps(per_summary, indent=2),
                encoding="utf-8"
            )
            continue

        sample_order.append(stem)

        bead = np.full(fsc_roi.shape, -1, dtype=int)
        bead[fsc_roi < b36] = 3
        bead[(fsc_roi >= b36) & (fsc_roi < b68)] = 6
        bead[fsc_roi >= b68] = 8

        per_bead_results: Dict[int, BeadFitResult] = {}
        warnings = []

        cluster_labels = np.empty(fluor_roi.shape, dtype=object)

        for b in (3, 6, 8):
            idx = bead == b
            fx = fluor_roi[idx]

            if fx.size == 0:
                per_bead_results[b] = BeadFitResult(
                    bead=b,
                    n=0,
                    model_kind="skipped",
                    means=[],
                    warn=f"Bead {b}: no ROI events."
                )
                warnings.append(per_bead_results[b].warn)
                continue

            res = fit_gmm_4state_or_fallback(fx, bead=b)
            per_bead_results[b] = res

            if res.warn:
                warnings.append(res.warn)

            if res.model_kind == "skipped":
                cluster_labels[idx] = f"MMB-{b} only"
                continue

            P = np.vstack([
                res.p_only,
                res.p_sandwich_v1,
                res.p_sandwich_v2,
                res.p_sandwich_v3
            ]).T

            hard = np.argmax(P, axis=1)
            states = np.array(["only", "sandwich_v1", "sandwich_v2", "sandwich_v3"], dtype=object)

            cluster_labels[idx] = np.array([f"MMB-{b} {states[i]}" for i in hard], dtype=object)

        missing = np.equal(cluster_labels, None)
        if np.any(missing):
            cluster_labels[missing] = "MMB-3 only"

        clusters = get_cluster_names()
        hard_counts = {c: 0 for c in clusters}
        expected_counts = {c: 0.0 for c in clusters}

        u, cts = np.unique(cluster_labels, return_counts=True)
        for name, ct in zip(u.tolist(), cts.tolist()):
            if name in hard_counts:
                hard_counts[name] = int(ct)

        per_bead_summary = {}

        for b in (3, 6, 8):
            idx = bead == b
            res = per_bead_results[b]

            if res.model_kind == "skipped" or res.p_only is None:
                per_bead_summary[str(b)] = {
                    "n_roi_events": int(idx.sum()),
                    "model": res.model_kind,
                    "means": res.means,
                    "expected": {
                        "only": None,
                        "sandwich_v1": None,
                        "sandwich_v2": None,
                        "sandwich_v3": None,
                        "sandwich_total": None,
                        "normalised_sandwich_v1_events_percent": None,
                        "normalised_sandwich_v2_events_percent": None,
                        "normalised_sandwich_v3_events_percent": None,
                        "normalised_total_sandwich_events_percent": None,
                    },
                    "warning": res.warn,
                }
                continue

            e_only = float(np.sum(res.p_only))
            e_sandwich_v1 = float(np.sum(res.p_sandwich_v1))
            e_sandwich_v2 = float(np.sum(res.p_sandwich_v2))
            e_sandwich_v3 = float(np.sum(res.p_sandwich_v3))
            e_sandwich_total = e_sandwich_v1 + e_sandwich_v2 + e_sandwich_v3

            # Normalised sandwich signals for this bead population.
            # The total normalised value is retained for summary purposes, while
            # the v1/v2/v3-specific normalised values form the new 3:6:8
            # Diagnostic_Ratios in the Diagnostic Metrics sheet.
            normalised_sandwich_v1_percent = calculate_normalised_percent(
                numerator=e_sandwich_v1,
                expected_sandwich_total=e_sandwich_total,
                expected_only=e_only
            )

            normalised_sandwich_v2_percent = calculate_normalised_percent(
                numerator=e_sandwich_v2,
                expected_sandwich_total=e_sandwich_total,
                expected_only=e_only
            )

            normalised_sandwich_v3_percent = calculate_normalised_percent(
                numerator=e_sandwich_v3,
                expected_sandwich_total=e_sandwich_total,
                expected_only=e_only
            )

            normalised_total_sandwich_percent = calculate_normalised_percent(
                numerator=e_sandwich_total,
                expected_sandwich_total=e_sandwich_total,
                expected_only=e_only
            )

            expected_counts[f"MMB-{b} only"] += e_only
            expected_counts[f"MMB-{b} sandwich_v1"] += e_sandwich_v1
            expected_counts[f"MMB-{b} sandwich_v2"] += e_sandwich_v2
            expected_counts[f"MMB-{b} sandwich_v3"] += e_sandwich_v3

            per_bead_summary[str(b)] = {
                "n_roi_events": int(idx.sum()),
                "model": res.model_kind,
                "means": res.means,
                "expected": {
                    "only": e_only,
                    "sandwich_v1": e_sandwich_v1,
                    "sandwich_v2": e_sandwich_v2,
                    "sandwich_v3": e_sandwich_v3,
                    "sandwich_total": e_sandwich_total,
                    "normalised_sandwich_v1_events_percent": normalised_sandwich_v1_percent,
                    "normalised_sandwich_v2_events_percent": normalised_sandwich_v2_percent,
                    "normalised_sandwich_v3_events_percent": normalised_sandwich_v3_percent,
                    "normalised_total_sandwich_events_percent": normalised_total_sandwich_percent,
                },
                "warning": res.warn,
            }

            master_rows.append({
                "file": f.name,
                "sample": stem,
                "bead": b,
                "n_roi_events": int(idx.sum()),
                "model": res.model_kind,
                "mean_only": res.means[0] if len(res.means) >= 1 else None,
                "mean_sandwich_v1": res.means[1] if len(res.means) >= 2 else None,
                "mean_sandwich_v2": res.means[2] if len(res.means) >= 3 else None,
                "mean_sandwich_v3": res.means[3] if len(res.means) >= 4 else None,
                "expected_only": e_only,
                "expected_sandwich_v1": e_sandwich_v1,
                "normalised_sandwich_v1_events (%)": normalised_sandwich_v1_percent,
                "expected_sandwich_v2": e_sandwich_v2,
                "normalised_sandwich_v2_events (%)": normalised_sandwich_v2_percent,
                "expected_sandwich_v3": e_sandwich_v3,
                "normalised_sandwich_v3_events (%)": normalised_sandwich_v3_percent,
                "expected_sandwich_total": e_sandwich_total,
                "normalised_total_sandwich_events (%)": normalised_total_sandwich_percent,
            })

        total_only = (
            expected_counts["MMB-3 only"]
            + expected_counts["MMB-6 only"]
            + expected_counts["MMB-8 only"]
        )

        total_sandwich_v1 = (
            expected_counts["MMB-3 sandwich_v1"]
            + expected_counts["MMB-6 sandwich_v1"]
            + expected_counts["MMB-8 sandwich_v1"]
        )

        total_sandwich_v2 = (
            expected_counts["MMB-3 sandwich_v2"]
            + expected_counts["MMB-6 sandwich_v2"]
            + expected_counts["MMB-8 sandwich_v2"]
        )

        total_sandwich_v3 = (
            expected_counts["MMB-3 sandwich_v3"]
            + expected_counts["MMB-6 sandwich_v3"]
            + expected_counts["MMB-8 sandwich_v3"]
        )

        total_sandwich = total_sandwich_v1 + total_sandwich_v2 + total_sandwich_v3

        # Overall normalised sandwich signals across all bead populations in the sample.
        # These are included for completeness in the ALL row. Diagnostic_Ratios
        # are calculated from the per-bead normalised values above.
        total_normalised_sandwich_v1_percent = calculate_normalised_percent(
            numerator=total_sandwich_v1,
            expected_sandwich_total=total_sandwich,
            expected_only=total_only
        )

        total_normalised_sandwich_v2_percent = calculate_normalised_percent(
            numerator=total_sandwich_v2,
            expected_sandwich_total=total_sandwich,
            expected_only=total_only
        )

        total_normalised_sandwich_v3_percent = calculate_normalised_percent(
            numerator=total_sandwich_v3,
            expected_sandwich_total=total_sandwich,
            expected_only=total_only
        )

        total_normalised_total_sandwich_percent = calculate_normalised_percent(
            numerator=total_sandwich,
            expected_sandwich_total=total_sandwich,
            expected_only=total_only
        )

        master_rows.append({
            "file": f.name,
            "sample": stem,
            "bead": "ALL",
            "n_roi_events": int(fluor_roi.size),
            "model": "mix",
            "mean_only": None,
            "mean_sandwich_v1": None,
            "mean_sandwich_v2": None,
            "mean_sandwich_v3": None,
            "expected_only": float(total_only),
            "expected_sandwich_v1": float(total_sandwich_v1),
            "normalised_sandwich_v1_events (%)": total_normalised_sandwich_v1_percent,
            "expected_sandwich_v2": float(total_sandwich_v2),
            "normalised_sandwich_v2_events (%)": total_normalised_sandwich_v2_percent,
            "expected_sandwich_v3": float(total_sandwich_v3),
            "normalised_sandwich_v3_events (%)": total_normalised_sandwich_v3_percent,
            "expected_sandwich_total": float(total_sandwich),
            "normalised_total_sandwich_events (%)": total_normalised_total_sandwich_percent,
        })

        transform_label = "logicle" if LOGICLE_SOURCE != "asinh-fallback" else "asinh"
        plot_path = plots_dir / f"{safe_filename(stem)}_clusters_{transform_label}.png"

        plot_patient_clusters(
            out_path=plot_path,
            fluor=fluor_roi,
            fsc=fsc_roi,
            cluster_labels=cluster_labels,
            fluor_label=f"{fluor_name} ({transform_label})",
            fsc_label=f"{fsc_name} ({transform_label})",
            title=f"{stem} — ROI clusters (12) | {fsc_name} vs {fluor_name}",
            dg_low_cut=dg_low_cut,
            piecewise_fsc_bin_centres=fsc_bin_centres,
            piecewise_max_fluorescence=max_fluorescence,
            mmb_window=mmb_window,
            b36=b36,
            b68=b68,
        )

        grid_panels.append({
            "name": stem,
            "x": fluor_roi,
            "y": fsc_roi,
            "labels": cluster_labels
        })

        per_summary = {
            "file": f.name,
            "sample": stem,
            "channels": {"fluor": fluor_name, "fsc": fsc_name},
            "transform_source": LOGICLE_SOURCE,
            "windows": {
                    "mmb_window": list(mmb_window),
                    "dg_low_cut": float(dg_low_cut),
                    "piecewise_dg_high_boundary": {
                        "fsc_bin_centres": [float(x) for x in fsc_bin_centres],
                        "max_fluorescence": [float(x) for x in max_fluorescence],
                    }
                },
            "boundaries": {"boundary_3_6": b36, "boundary_6_8": b68},
            "n_total_events": int(df.shape[0]),
            "n_finite": int(finite.sum()),
            "n_roi_events": int(fluor_roi.size),
            "percentiles_init": {
                "LOW_Q": LOW_Q,
                "LOWMID_Q": LOWMID_Q,
                "MIDHIGH_Q": MIDHIGH_Q,
                "HIGH_Q": HIGH_Q,
            },
            "min_events": {
                "MIN_EVENTS_4COMP": MIN_EVENTS_4COMP,
                "MIN_EVENTS_2COMP": MIN_EVENTS_2COMP,
            },
            "hard_counts": hard_counts,
            "expected_counts": expected_counts,
            "per_bead": per_bead_summary,
            "warnings": warnings,
        }

        per_patient_json_dir.mkdir(parents=True, exist_ok=True)
        (per_patient_json_dir / f"{safe_filename(stem)}_summary.json").write_text(
            json.dumps(per_summary, indent=2),
            encoding="utf-8"
        )

        if warnings:
            print("  ⚠️ Warnings:")
            for w in warnings:
                print(f"    - {w}")

        print("  Done.\n")

    if grid_panels:
        transform_label = "logicle" if LOGICLE_SOURCE != "asinh-fallback" else "asinh"
        grid_path = out_root / f"ALL_PATIENTS_GRID_12clusters_{transform_label}.png"

        save_grid_overview(
            out_path=grid_path,
            panels=grid_panels,
            fluor_label=f"{(fluor_pref or 'Fluor')} ({transform_label})",
            fsc_label=f"{(fsc_pref or 'FSC')} ({transform_label})",
            suptitle="All patients — ROI clusters (12)"
        )

    if master_rows:
        df_master = pd.DataFrame(master_rows)

        col_order = [
            "file",
            "sample",
            "bead",
            "n_roi_events",
            "model",
            "mean_only",
            "mean_sandwich_v1",
            "mean_sandwich_v2",
            "mean_sandwich_v3",
            "expected_only",
            "expected_sandwich_v1",
            "normalised_sandwich_v1_events (%)",
            "expected_sandwich_v2",
            "normalised_sandwich_v2_events (%)",
            "expected_sandwich_v3",
            "normalised_sandwich_v3_events (%)",
            "expected_sandwich_total",
            "normalised_total_sandwich_events (%)",
        ]

        df_master = df_master[col_order]

        df_diag = build_diagnostic_rows(df_master, sample_order)

        xlsx_path = out_root / "patient_gmm_summary_12clusters.xlsx"

        save_excel_workbook(
            out_path=xlsx_path,
            df_master=df_master,
            df_diag=df_diag
        )

        print(f"\nSaved Excel workbook -> {xlsx_path}")

    print(f"\nAll outputs written to:\n  {out_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)