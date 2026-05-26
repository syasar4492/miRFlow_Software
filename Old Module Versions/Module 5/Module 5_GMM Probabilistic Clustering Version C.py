# gmm_patient_clustering_4state_per_bead.py
#
# =============================================================================
# WHAT THIS SCRIPT DOES
# =============================================================================
# For each patient .fcs file:
#   0) Load + auto-pick channels (prefers JSON-provided channel names; if not falls back to heuristics)
#   1) Apply Logicle transform (flowutils) with asinh fallback as back-up
#   2) Apply ROI gating (control-derived global windows):
#        - FSC in [MMB_LOW, MMB_HIGH]
#        - FLURO in [DG_LOW, DG_HIGH]
#   3) Assign bead size deterministically using FSC thresholds:
#        - FSC < boundary_3_6 -> MMB-3
#        - boundary_3_6 <= FSC < boundary_6_8 -> MMB-6
#        - FSC >= boundary_6_8 -> MMB-8
#   4) For each sample(patient) bead size subset, fit 1D GMM on fluorescence within ROI:
#        - default: 4 components (only / sandwich_v1 / sandwich_v2 / multi), soft assignment
#        - init means anchored at patient data percentiles within ROI_X: q20, q40, q60, q80
#        - label components by ascending mean:
#             low      = only
#             low-mid  = sandwich_v1
#             mid-high = sandwich_v2
#             high     = multi
#      Exception A:
#        - if too few events for stable 4-comp, fall back to 2-comp
#        - 2-comp fallback labels: only / sandwich_v1
#        - sandwich_v2 and multi are set to 0
#      Exception B:
#        - q20, q40, q60, and q80 chosen heuristically
#        - if percentile values are too close, ε(1e-6) is added to prevent model collapse
#   5) Outputs:
#        - per-patient 2D plot (ROI only): x=fluor, y=FSC, colored by 12 clusters + legend
#          plus MMB gate/boundary lines for interpretability
#        - per-patient JSON summary (counts + posterior-weighted expected counts, efficiencies)
#        - one overview GRID png with all patient plots
#        - a master CSV across all patients (one row per patient x bead size + overall row)
#
# =============================================================================
# PATHS (EDIT IF NEEDED)
# =============================================================================

from __future__ import annotations

JSON_DIR = r"D:\ICL MBE\Year 4\FYP\Software Automation\Visual Flow Data\Json Files"
GATES_BASENAME = "gate_summary_by_type"      # (typo-safe: script searches by prefix/contains)
THRESH_BASENAME = "mmb_fsc_thresholds"       # boundary_3_6 / boundary_6_8
PATIENT_FCS_DIR = r"D:\ICL MBE\Year 4\FYP\Software Automation\Raw Flow Data\06.16.2025_20samples_optimized protocol"

# Output root (will create subfolder inside this)
OUTPUT_ROOT = r"D:\ICL MBE\Year 4\FYP\Software Automation\Visual Flow Data\20 samples optimised protocol"
OUTPUT_SUBFOLDER = "GMM_Clustering_12clusters_new DG Window"

# =============================================================================
# TUNABLE HYPERPARAMETERS
# =============================================================================
LOW_Q = 20       # for init mean 1: only
LOWMID_Q = 40    # for init mean 2: sandwich_v1
MIDHIGH_Q = 60   # for init mean 3: sandwich_v2
HIGH_Q = 80      # for init mean 4: multi

MIN_EVENTS_4COMP = 1000   # below this, use 2-comp fallback
MIN_EVENTS_2COMP = 300    # below this, skip fitting for that bead size

# plotting
MAX_POINTS_PER_PLOT = 30000
MAX_POINTS_PER_PANEL_GRID = 12000

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

# --- Logicle transform  ---
try:
    from flowutils.transforms import logicle as logicle_transform
    LOGICLE_SOURCE = "flowutils.transforms.logicle"
except Exception:
    logicle_transform = None
    LOGICLE_SOURCE = "asinh-fallback"


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


# --- Channel picking ---
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
    fluor = (first_match(cols, ["FL1-H"])
             or first_match(cols, ["B525-H", "BL1-H", "FITC-H", "GFP-H"])
             or first_match(cols, ["B530-H", "B515-H", "530/30", "525/50"])
             or first_match(cols, ["FL1-A", "BL1-A", "B525-A", "FITC-A", "GFP-A"]))

    if fsc is None:
        raise ValueError(f"No FSC channel found in {cols}")

    if fluor is None:
        h_cands = [c for c in cols
                   if c.upper().endswith("-H")
                   and not any(k in c.upper() for k in ["FSC", "SSC", "TIME", "WIDTH"])]
        if h_cands:
            fluor = h_cands[0]
        else:
            a_cands = [c for c in cols
                       if c.upper().endswith("-A")
                       and not any(k in c.upper() for k in ["FSC", "SSC", "TIME", "WIDTH"])]
            if not a_cands:
                raise ValueError(f"No fluorescence channel found in {cols}")
            fluor = a_cands[0]

    return fluor, fsc


# --- JSON loading helpers ---
def find_json_by_basename(folder: Path, basename: str) -> Path:
    """
    Robustly find a JSON file in folder by:
      1) exact match basename.json
      2) startswith(basename) and endswith .json
      3) contains(basename) and endswith .json
    """
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


def load_global_windows(gates_json_path: Path) -> Tuple[Tuple[float, float], Tuple[float, float], Dict[str, str]]:
    """
    Expected structure:
      {
        ".*": {
          "dg_window": [low, high],
          "mmb_window": [low, high],
          "channels": {"fluor": "...", "fsc": "..."}
        }
      }
    """
    d = json.loads(gates_json_path.read_text(encoding="utf-8"))
    if not isinstance(d, dict) or len(d) == 0:
        raise ValueError(f"Unexpected gates JSON structure in {gates_json_path.name}")

    first_key = next(iter(d.keys()))
    block = d[first_key]

    if "mmb_window" not in block or "dg_window" not in block:
        raise KeyError(f"Missing mmb_window/dg_window under top key '{first_key}' in {gates_json_path.name}")

    mmb = block["mmb_window"]
    dg = block["dg_window"]
    if mmb is None or dg is None:
        raise ValueError(f"mmb_window or dg_window is None in {gates_json_path.name}")

    ch = block.get("channels", {})
    return (float(mmb[0]), float(mmb[1])), (float(dg[0]), float(dg[1])), ch


def load_bead_boundaries(thresh_json_path: Path) -> Tuple[float, float, str]:
    """
    Expected from mmb_threshold script:
      {
        ...,
        "fsc_channel": "...",
        "recommended_boundaries": {"boundary_3_6": ..., "boundary_6_8": ...},
        ...
      }
    """
    d = json.loads(thresh_json_path.read_text(encoding="utf-8"))
    rb = d.get("recommended_boundaries", {})
    b36 = rb.get("boundary_3_6", None)
    b68 = rb.get("boundary_6_8", None)
    if b36 is None or b68 is None:
        raise KeyError(f"Could not find boundary_3_6 / boundary_6_8 in {thresh_json_path.name}")

    fsc_channel = d.get("fsc_channel", "FSC-H")
    return float(b36), float(b68), str(fsc_channel)


# --- GMM utilities ---
def ensure_strictly_increasing(means: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    If percentiles collapse, enforce strict increase with tiny epsilon differentiators.
    """
    m = means.astype(float).copy()
    for i in range(1, len(m)):
        if m[i] <= m[i - 1]:
            m[i] = m[i - 1] + eps
    return m


@dataclass
class BeadFitResult:
    bead: int
    n: int
    model_kind: str  # "gmm4" or "gmm2" or "skipped"
    means: List[float]
    p_only: Optional[np.ndarray] = None
    p_sandwich_v1: Optional[np.ndarray] = None
    p_sandwich_v2: Optional[np.ndarray] = None
    p_multi: Optional[np.ndarray] = None
    warn: Optional[str] = None


def fit_gmm_4state_or_fallback(fluor: np.ndarray, bead: int) -> BeadFitResult:
    """
    Fit 4-component 1D GMM if enough events; else fallback to 2-component; else skip.

    4-component state order by ascending mean:
      0 = only
      1 = sandwich_v1
      2 = sandwich_v2
      3 = multi

    2-component fallback:
      0 = only
      1 = sandwich_v1
      sandwich_v2 = 0
      multi = 0
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
        # fallback 2-component: only / sandwich_v1
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
        p_multi = np.zeros_like(p_only)

        return BeadFitResult(
            bead=bead,
            n=n,
            model_kind="gmm2",
            means=means_sorted.tolist(),
            p_only=p_only,
            p_sandwich_v1=p_sandwich_v1,
            p_sandwich_v2=p_sandwich_v2,
            p_multi=p_multi,
            warn=(
                f"Bead {bead}: used 2-comp fallback "
                f"(n={n} < {MIN_EVENTS_4COMP}). sandwich_v2 and multi set to 0."
            )
        )

    # 4-component GMM with percentile init
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
    p_multi = probs[:, order[3]]

    return BeadFitResult(
        bead=bead,
        n=n,
        model_kind="gmm4",
        means=means_sorted.tolist(),
        p_only=p_only,
        p_sandwich_v1=p_sandwich_v1,
        p_sandwich_v2=p_sandwich_v2,
        p_multi=p_multi,
        warn=None
    )


# --- Plotting ---
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
    states = ["only", "sandwich_v1", "sandwich_v2", "multi"]
    return [f"MMB-{b} {s}" for b in (3, 6, 8) for s in states]


def plot_patient_clusters(
    out_path: Path,
    fluor: np.ndarray,
    fsc: np.ndarray,
    cluster_labels: np.ndarray,
    fluor_label: str,
    fsc_label: str,
    title: str,
    dg_window: Tuple[float, float],
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

    dg_low, dg_high = dg_window
    mmb_low, mmb_high = mmb_window
    plt.axvline(dg_low, ls="--", lw=1)
    plt.axvline(dg_high, ls="--", lw=1)
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
    """
    panels: list of dict with keys {name, x, y, labels}
    """
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


# --- Main analysis ---
def main():
    json_dir = Path(JSON_DIR)
    gates_json = find_json_by_basename(json_dir, GATES_BASENAME)
    thresh_json = find_json_by_basename(json_dir, THRESH_BASENAME)

    mmb_window, dg_window, ch = load_global_windows(gates_json)
    b36, b68, fsc_ch_from_thresh = load_bead_boundaries(thresh_json)

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
    print(f"Patient dir: {patient_dir}")
    print(f"Output dir:  {out_root}\n")

    print("Loaded global windows:")
    print(f"  MMB window (FSC): {mmb_window}")
    print(f"  DG window  (FL):  {dg_window}")
    print("Loaded bead boundaries:")
    print(f"  boundary_3_6: {b36}")
    print(f"  boundary_6_8: {b68}\n")

    fluor_pref = (ch.get("fluor") if isinstance(ch, dict) else None)
    fsc_pref = (ch.get("fsc") if isinstance(ch, dict) else None)

    master_rows = []
    grid_panels = []

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
        dg_low, dg_high = dg_window

        roi = (
            (fsc_t >= mmb_low)
            & (fsc_t <= mmb_high)
            & (fluor_t >= dg_low)
            & (fluor_t <= dg_high)
        )

        fluor_roi = fluor_t[roi]
        fsc_roi = fsc_t[roi]

        if fluor_roi.size == 0:
            warn = "ROI empty after applying global gates."
            print(f"  ⚠️ {warn}")

            per_summary = {
                "file": f.name,
                "channels": {"fluor": fluor_name, "fsc": fsc_name},
                "windows": {"mmb_window": list(mmb_window), "dg_window": list(dg_window)},
                "boundaries": {"boundary_3_6": b36, "boundary_6_8": b68},
                "n_total_events": int(df.shape[0]),
                "n_finite": int(finite.sum()),
                "n_roi": 0,
                "warnings": [warn],
                "per_bead": {},
            }

            per_patient_json_dir.mkdir(parents=True, exist_ok=True)
            (per_patient_json_dir / f"{safe_filename(stem)}_summary.json").write_text(
                json.dumps(per_summary, indent=2),
                encoding="utf-8"
            )
            continue

        bead = np.full(fsc_roi.shape, -1, dtype=int)
        bead[fsc_roi < b36] = 3
        bead[(fsc_roi >= b36) & (fsc_roi < b68)] = 6
        bead[fsc_roi >= b68] = 8

        per_bead_results: Dict[int, BeadFitResult] = {}
        warnings = []

        cluster_labels = np.empty(fluor_roi.shape, dtype=object)

        for b in (3, 6, 8):
            idx = (bead == b)
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
                res.p_multi
            ]).T

            hard = np.argmax(P, axis=1)
            states = np.array(["only", "sandwich_v1", "sandwich_v2", "multi"], dtype=object)
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
            idx = (bead == b)
            res = per_bead_results[b]

            if res.model_kind == "skipped" or res.p_only is None:
                per_bead_summary[str(b)] = {
                    "n_roi": int(idx.sum()),
                    "model": res.model_kind,
                    "means": res.means,
                    "expected": {
                        "only": None,
                        "sandwich_v1": None,
                        "sandwich_v2": None,
                        "multi": None,
                    },
                    "eff_sandwich_v1": None,
                    "eff_sandwich_v2": None,
                    "eff_sandwich_total": None,
                    "eff_multi": None,
                    "warning": res.warn,
                }
                continue

            e_only = float(np.sum(res.p_only))
            e_sandwich_v1 = float(np.sum(res.p_sandwich_v1))
            e_sandwich_v2 = float(np.sum(res.p_sandwich_v2))
            e_multi = float(np.sum(res.p_multi))

            denom = e_only + e_sandwich_v1 + e_sandwich_v2 + e_multi

            eff_sandwich_v1 = (e_sandwich_v1 / denom) if denom > 0 else None
            eff_sandwich_v2 = (e_sandwich_v2 / denom) if denom > 0 else None
            eff_sandwich_total = ((e_sandwich_v1 + e_sandwich_v2) / denom) if denom > 0 else None
            eff_multi = (e_multi / denom) if denom > 0 else None

            expected_counts[f"MMB-{b} only"] += e_only
            expected_counts[f"MMB-{b} sandwich_v1"] += e_sandwich_v1
            expected_counts[f"MMB-{b} sandwich_v2"] += e_sandwich_v2
            expected_counts[f"MMB-{b} multi"] += e_multi

            per_bead_summary[str(b)] = {
                "n_roi": int(idx.sum()),
                "model": res.model_kind,
                "means": res.means,
                "expected": {
                    "only": e_only,
                    "sandwich_v1": e_sandwich_v1,
                    "sandwich_v2": e_sandwich_v2,
                    "multi": e_multi,
                },
                "eff_sandwich_v1": eff_sandwich_v1,
                "eff_sandwich_v2": eff_sandwich_v2,
                "eff_sandwich_total": eff_sandwich_total,
                "eff_multi": eff_multi,
                "warning": res.warn,
            }

            master_rows.append({
                "file": f.name,
                "sample": stem,
                "bead": b,
                "n_roi_bead": int(idx.sum()),
                "model": res.model_kind,
                "mean_only": res.means[0] if len(res.means) >= 1 else None,
                "mean_sandwich_v1": res.means[1] if len(res.means) >= 2 else None,
                "mean_sandwich_v2": res.means[2] if len(res.means) >= 3 else None,
                "mean_multi": res.means[3] if len(res.means) >= 4 else None,
                "expected_only": e_only,
                "expected_sandwich_v1": e_sandwich_v1,
                "expected_sandwich_v2": e_sandwich_v2,
                "expected_sandwich_total": e_sandwich_v1 + e_sandwich_v2,
                "expected_multi": e_multi,
                "eff_sandwich_v1": eff_sandwich_v1,
                "eff_sandwich_v2": eff_sandwich_v2,
                "eff_sandwich_total": eff_sandwich_total,
                "eff_multi": eff_multi,
            })

        total_expected = float(sum(expected_counts.values()))
        total_sandwich_v1 = float(
            expected_counts["MMB-3 sandwich_v1"]
            + expected_counts["MMB-6 sandwich_v1"]
            + expected_counts["MMB-8 sandwich_v1"]
        )
        total_sandwich_v2 = float(
            expected_counts["MMB-3 sandwich_v2"]
            + expected_counts["MMB-6 sandwich_v2"]
            + expected_counts["MMB-8 sandwich_v2"]
        )
        total_sandwich = total_sandwich_v1 + total_sandwich_v2
        total_multi = float(
            expected_counts["MMB-3 multi"]
            + expected_counts["MMB-6 multi"]
            + expected_counts["MMB-8 multi"]
        )
        total_only = float(
            expected_counts["MMB-3 only"]
            + expected_counts["MMB-6 only"]
            + expected_counts["MMB-8 only"]
        )

        overall_eff_sandwich_v1 = (total_sandwich_v1 / total_expected) if total_expected > 0 else None
        overall_eff_sandwich_v2 = (total_sandwich_v2 / total_expected) if total_expected > 0 else None
        overall_eff_sandwich_total = (total_sandwich / total_expected) if total_expected > 0 else None
        overall_eff_multi = (total_multi / total_expected) if total_expected > 0 else None

        master_rows.append({
            "file": f.name,
            "sample": stem,
            "bead": "ALL",
            "n_roi_bead": int(fluor_roi.size),
            "model": "mix",
            "mean_only": None,
            "mean_sandwich_v1": None,
            "mean_sandwich_v2": None,
            "mean_multi": None,
            "expected_only": total_only,
            "expected_sandwich_v1": total_sandwich_v1,
            "expected_sandwich_v2": total_sandwich_v2,
            "expected_sandwich_total": total_sandwich,
            "expected_multi": total_multi,
            "eff_sandwich_v1": overall_eff_sandwich_v1,
            "eff_sandwich_v2": overall_eff_sandwich_v2,
            "eff_sandwich_total": overall_eff_sandwich_total,
            "eff_multi": overall_eff_multi,
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
            dg_window=dg_window,
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
            "windows": {"mmb_window": list(mmb_window), "dg_window": list(dg_window)},
            "boundaries": {"boundary_3_6": b36, "boundary_6_8": b68},
            "n_total_events": int(df.shape[0]),
            "n_finite": int(finite.sum()),
            "n_roi": int(fluor_roi.size),
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
            "overall": {
                "eff_sandwich_v1": overall_eff_sandwich_v1,
                "eff_sandwich_v2": overall_eff_sandwich_v2,
                "eff_sandwich_total": overall_eff_sandwich_total,
                "eff_multi": overall_eff_multi,
            },
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
            "n_roi_bead",
            "model",
            "mean_only",
            "mean_sandwich_v1",
            "mean_sandwich_v2",
            "mean_multi",
            "expected_only",
            "expected_sandwich_v1",
            "expected_sandwich_v2",
            "expected_sandwich_total",
            "expected_multi",
            "eff_sandwich_v1",
            "eff_sandwich_v2",
            "eff_sandwich_total",
            "eff_multi",
        ]

        df_master = df_master[col_order]
        csv_path = out_root / "patient_gmm_summary_12clusters.csv"
        df_master.to_csv(csv_path, index=False)
        print(f"\nSaved CSV summary -> {csv_path}")

    print(f"\nAll outputs written to:\n  {out_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)