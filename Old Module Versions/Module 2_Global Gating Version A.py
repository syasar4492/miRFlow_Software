# advanced_fcs_viewer_FSC-H_logicle_named_outputs_subfolders_DG_WINDOW_FULLCOMMENTS.py
#
# =============================================================================
# HIGH-LEVEL OVERVIEW (what this script does and why)
# =============================================================================
# This script automates a “control-derived gating + visualization” workflow for
# your MMB and DG control FCS files.
#
# It:
#   1) Reads all .fcs files from an "unrefined" folder.
#   2) Auto-detects the FSC (size) and the main green fluorescence channel.
#   3) Applies a Logicle transform (preferred) with an asinh fallback.
#   4) Derives per-file gating boundaries from KDE-based 1D density estimates:
#
#      A) MMB FSC window: [left, right]
#         - A classic “dominant bead peak” window around FSC-H
#         - Found by walking left/right from the dominant KDE peak until the density
#           drops to 5% of the peak height.
#
#      B) DG fluorescence window: [low_cut, high_cut]  (IMPORTANT)
#         - Biological goal: keep the intermediate fluorescence band enriched for
#           functionalised complexes, while excluding:
#             (i) low/background events
#             (ii) very-high free DG events (bright, not functionalised)
#
#         - Practical observation: DG fluorescence is NOT always cleanly bimodal.
#           Some DG runs show:
#             * one dominant HIGH peak (free DG) + a long low/background tail
#             * a subtle “bump” on the left that is visible by eye but not a robust
#               full “peak” in the strict statistical sense
#
#         - What worked vs what did not:
#             * The earlier "Option A" (two tallest peaks) worked when the distribution
#               was clearly bimodal, BUT it often failed on unimodal-with-wiggles runs:
#               KDE can contain multiple tiny local maxima within the dominant high peak.
#               Selecting the “top 2 maxima” by height can mistakenly pick 2 maxima
#               inside the SAME high peak -> both cuts collapse around the high peak.
#
#             * The approach below is designed to avoid that failure:
#               - Anchor the HIGH population using the KDE global maximum
#               - Compute high_cut from the LEFT shoulder of that dominant peak (5% rule)
#               - Then search ONLY to the LEFT of high_cut for a genuine background-side
#                 bump using a prominence-based peak detector
#
#         - Key principle: We match the biological model with the statistical model by:
#             * treating the high/free-DG population as the most robust “feature”
#             * treating the background side as potentially subtle
#             * never forcing “two-peak bimodality” when the data does not support it
#
#   5) Produces plots for every file:
#       - 2D scatter:        (fluor vs FSC) saved as PNG
#       - 1D FSC density:    with 2 dashed lines for the FSC window
#       - 1D Fluor density:  for DG files only, with 2 dashed lines for DG window
#                            (MMB fluoro plots are intentionally ungated to keep them simple)
#
#   6) Writes a global summary JSON:
#       - mmb_window aggregated across MMB files only:
#           [min(left_i), max(right_i)]
#       - dg_window aggregated across DG files only:
#           GLOBAL_LOW  = min(low_cut_i)
#           GLOBAL_HIGH = min(high_cut_i)
#
# =============================================================================
# IMPORTANT NOTE: WHAT WAS FIXED (YOUR REQUEST)
# =============================================================================
# You correctly spotted an issue:
#
#   - DG global fluorescence window was already computed using DG files only ✅
#   - BUT the global MMB size window was being computed using *all* files ❌
#     which accidentally included DG files' FSC values in the MMB global window.
#
# That is now fixed:
#
#   ✅ The global MMB FSC window is computed ONLY from MMB files (NOT DG files).
#   ✅ DG fluorescence window remains computed ONLY from DG files.
#
# IMPORTANT: We still compute an FSC window for DG files *for plotting consistency*
# (so DG size distributions still get dashed FSC-window lines), but DG FSC windows
# are NOT used in the global MMB aggregation anymore.
#
# =============================================================================


# =============================================================================
# Imports
# =============================================================================
# - Pathlib for robust Windows paths
# - NumPy/Pandas for data handling
# - Matplotlib (Agg backend) to save PNGs without opening GUI windows
# - SciPy Gaussian KDE for smooth density estimation
# - SciPy find_peaks for robust detection of subtle bumps on DG fluorescence
# - fcsparser to load .fcs into a DataFrame
# =============================================================================

from pathlib import Path
import sys, os, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend: always write PNGs; no GUI required
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import fcsparser


# =============================================================================
# 1) TRANSFORMATION: Logicle (preferred) with asinh fallback
# =============================================================================
# Why transform?
#   Flow cytometry values span orders of magnitude. Logicle is standard because:
#     - linear near zero (keeps low signals interpretable)
#     - log-like at higher values (compresses bright populations)
#     - can accommodate negative values after compensation
#
# We try importing flowutils.transforms.logicle.
# If that fails, we use asinh(arr/150) as a safe fallback.
# =============================================================================

try:
    from flowutils.transforms import logicle as logicle_transform
    LOGICLE_SOURCE = "flowutils.transforms.logicle"
except Exception:
    logicle_transform = None
    LOGICLE_SOURCE = "asinh-fallback"


def apply_logicle_or_asinh(arr: np.ndarray) -> np.ndarray:
    """
    Apply Logicle (via flowutils.transforms.logicle) if available,
    otherwise apply a simple asinh transform.

    Parameters
    ----------
    arr : np.ndarray
        Raw channel values from the FCS file (e.g., FSC-H, B525-H).

    Returns
    -------
    np.ndarray
        Transformed values in a roughly logicle/asinh space.

    Implementation notes
    --------------------
    If logicle is available, we pick conventional parameters:
      - T: top of scale (use at least 262144, typical for cytometry scaling)
      - M: number of decades displayed (4.5 is a common display range)
      - W: width of linear region around 0 (0.5 is a reasonable default)
      - A: allow negatives if the observed min is negative
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

    # asinh fallback: scale factor 150 is arbitrary but typical-ish for cytometry
    return np.arcsinh(arr / 150.0)


# =============================================================================
# 2) Utility: pick a channel by matching patterns in column names
# =============================================================================
# Channel naming differs between instruments and software exports.
# So we match by substring:
#   - case-insensitive
#   - return the first column containing any of the patterns in order
# =============================================================================

def first_match(columns, patterns):
    ups = [c.upper() for c in columns]
    for pat in patterns:
        P = pat.upper()
        for i, cu in enumerate(ups):
            if P in cu:
                return columns[i]
    return None


# =============================================================================
# 3) Channel selection logic (FSC & fluorescence)
# =============================================================================
# We need:
#   - FSC-H (preferred) for size
#   - Green fluorescence (often B525-H for your cytometer)
#
# We:
#   - prefer exact matches (FSC-H, FL1-H, B525-H)
#   - fall back to sensible aliases
#   - finally fall back to "any -H not FSC/SSC/Time/Width"
# =============================================================================

def pick_channels(df: pd.DataFrame):
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


# =============================================================================
# 4) 1D density estimation using Gaussian KDE
# =============================================================================
# We use KDE (Gaussian kernel density estimate) instead of histograms because:
#   - it gives a smooth curve (less sensitive to binning choices)
#   - peaks/minima/shoulders can be detected more reproducibly
#
# Grid range: from 0.1th percentile to 99.9th percentile
# (prevents outliers from exploding the x-axis range)
# =============================================================================

def kde_density(values: np.ndarray, n=1000):
    """
    Compute KDE density for a 1D array.

    Returns
    -------
    xs : np.ndarray
        Grid of x-values across the central bulk of the data.
    ys : np.ndarray
        Estimated density at each x.

    Small-sample behaviour
    ----------------------
    If <10 finite points exist, KDE is unstable. We return:
      - xs over min..max
      - ys as zeros
    """
    v = values[np.isfinite(values)]
    if v.size < 10:
        lo = float(np.min(v)) if v.size else 0.0
        hi = float(np.max(v)) if v.size else 1.0
        xs = np.linspace(lo, hi, n)
        ys = np.zeros_like(xs)
        return xs, ys

    kde = gaussian_kde(v)
    xs = np.linspace(np.percentile(v, 0.1), np.percentile(v, 99.9), n)
    ys = kde(xs)
    return xs, ys


# =============================================================================
# 5) MMB logic: FSC window around dominant bead peak
# =============================================================================
# Steps:
#   - KDE on FSC values
#   - find dominant peak
#   - walk left/right until density drops to 5% of peak height
# Return: [left, right] window.
# =============================================================================

def fsc_window(values: np.ndarray, drop_frac=0.05):
    xs, ys = kde_density(values, n=1024)

    if not np.any(ys):
        v = values[np.isfinite(values)]
        if not v.size:
            return 0.0, 1.0
        med = float(np.median(v))
        iqr = float(np.subtract(*np.percentile(v, [75, 25])))
        return med - iqr, med + iqr

    peak = int(np.argmax(ys))
    ypk = ys[peak]
    thr = drop_frac * ypk

    i = peak
    while i > 0 and ys[i] > thr:
        i -= 1

    j = peak
    while j < len(ys) - 1 and ys[j] > thr:
        j += 1

    left, right = xs[max(i, 0)], xs[min(j, len(xs) - 1)]
    if left >= right:
        width = (xs[-1] - xs[0]) * 0.05
        left, right = xs[peak] - width, xs[peak] + width

    return float(left), float(right)


# =============================================================================
# 6) DG logic: fluorescence window [low_cut, high_cut]
# =============================================================================
# Goal:
#   - keep intermediate fluorescence enriched for functionalised complexes
#   - exclude:
#       (i) low/background
#      (ii) very-high free DG (bright, unfunctionalised)
#
# Robust approach:
#   - anchor high/free-DG as KDE global maximum
#   - high_cut = left shoulder of that peak (5% height rule)
#   - then search left side for a real bump using find_peaks with prominence
#   - low_cut = right shoulder of that bump (5% height rule)
# =============================================================================

def dg_fluoro_window_bio_aligned(values: np.ndarray, shoulder_frac: float = 0.05):
    xs, ys = kde_density(values, n=2048)
    v = values[np.isfinite(values)]

    # If KDE is unreliable, return a conservative central band
    if v.size < 20 or not np.any(ys):
        if v.size:
            lo = float(np.percentile(v, 30))
            hi = float(np.percentile(v, 70))
            if lo < hi:
                return lo, hi
            return float(np.min(v)), float(np.max(v))
        return 0.0, 1.0

    # -------------------------------------------------------------
    # (1) High peak = KDE global max (dominant free-DG population)
    # -------------------------------------------------------------
    high_peak = int(np.argmax(ys))
    high_peak_h = float(ys[high_peak])

    # -------------------------------------------------------------
    # (2) high_cut = left shoulder of high peak (5%)
    # -------------------------------------------------------------
    high_thr = shoulder_frac * high_peak_h
    j = high_peak
    while j > 0 and ys[j] > high_thr:
        j -= 1
    high_cut = float(xs[j])
    high_cut_idx = int(j)

    # Guard for degenerate cases (not enough left-side range)
    if high_cut_idx < 10:
        lo = float(np.percentile(v, 30))
        hi = float(np.percentile(v, 70))
        if lo < hi:
            return lo, hi
        return float(np.min(v)), float(np.max(v))

    # -------------------------------------------------------------
    # (3) Find a meaningful bump on the left side (background-side)
    # -------------------------------------------------------------
    left_region_ys = ys[:high_cut_idx]
    left_region_xs = xs[:high_cut_idx]

    left_max = float(np.max(left_region_ys)) if left_region_ys.size else 0.0
    if left_max <= 0.0:
        low_cut = float(np.percentile(v, 30))
        if low_cut < high_cut:
            return low_cut, high_cut
        lo = float(np.percentile(v, 35))
        hi = float(np.percentile(v, 65))
        return (lo, hi) if lo < hi else (float(np.min(v)), float(np.max(v)))

    # Prominence threshold to ignore tiny KDE wiggles
    prom = 0.02 * float(np.max(ys))  # 2% of global max density
    peaks, props = find_peaks(left_region_ys, prominence=prom)

    if peaks.size > 0:
        # Use rightmost peak on left side (closest to high_cut)
        low_peak = int(peaks[np.argmax(peaks)])
        low_peak_h = float(left_region_ys[low_peak])

        # Sanity check: if bump is extremely small, fallback to left-region max
        if low_peak_h < 0.05 * float(np.max(ys)):
            low_peak = int(np.argmax(left_region_ys))
            low_peak_h = float(left_region_ys[low_peak])
    else:
        low_peak = int(np.argmax(left_region_ys))
        low_peak_h = float(left_region_ys[low_peak])

    # low_cut = right shoulder of chosen bump (5%)
    low_thr = shoulder_frac * low_peak_h
    i = low_peak
    while i < left_region_ys.size - 1 and left_region_ys[i] > low_thr:
        i += 1
    low_cut = float(left_region_xs[i])

    # Enforce proper ordering
    if not (low_cut < high_cut):
        lo = float(np.percentile(v, 35))
        hi = float(np.percentile(v, 65))
        if lo < hi:
            return lo, hi
        return float(np.min(v)), float(np.max(v))

    return low_cut, high_cut


# =============================================================================
# 7) Plotting helpers
# =============================================================================
# - plot_2d_png: saves scatter (fluor vs FSC)
# - plot_1d_png: saves KDE density + optional gate lines
# =============================================================================

def plot_2d_png(out_dir: Path, filename_stem: str, xvals, yvals, xlab, ylab, title: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(xvals, yvals, s=2, alpha=0.5)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    out = out_dir / f"{filename_stem}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def plot_1d_png(out_dir: Path, filename_stem: str, values, label, gate=None, title: str | None = None):
    """
    Gate behaviour:
      - None → no gate lines
      - scalar → draw one vertical line
      - list/tuple length 2 → draw two vertical lines (window)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    xs, ys = kde_density(values, n=500)
    plt.figure(figsize=(6, 2.5))
    plt.plot(xs, ys, lw=1.5)

    if gate is not None:
        if isinstance(gate, (list, tuple)) and len(gate) == 2:
            plt.axvline(gate[0], color="crimson", ls="--", lw=1)
            plt.axvline(gate[1], color="crimson", ls="--", lw=1)
        else:
            plt.axvline(float(gate), color="royalblue", ls="--", lw=1)

    plt.xlabel(label)
    plt.ylabel("Density")
    plt.title(title if title is not None else label)
    plt.tight_layout()
    out = out_dir / f"{filename_stem}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


# =============================================================================
# 8) Core processing: loop over FCS files, gate, and save plots
# =============================================================================
# Key responsibilities:
#   - Find all .fcs matching a regex pattern
#   - For each file:
#       * load with fcsparser
#       * pick channels
#       * transform values
#       * compute:
#           - FSC window (always computed for plotting)
#           - DG fluorescence window (DG only)
#       * save plots to appropriate folders based on DG/MMB naming convention
#
# IMPORTANT FIX YOU REQUESTED:
#   - Global MMB window should be computed ONLY from MMB files
#   - Global DG window should be computed ONLY from DG files
# =============================================================================

def process_pattern(folder: Path, pattern: str, base_out_dir: Path):
    regex = re.compile(pattern, re.IGNORECASE)
    files = [f for f in sorted(folder.glob("*.fcs")) if regex.search(f.name)]
    print(f"\nPattern '{pattern}' -> {len(files)} files")
    if not files:
        print(f"⚠️ Skipping: no .fcs matching /{pattern}/ in {folder}")
        return None

    # Label axes based on which transform we are actually using
    transform_label = "logicle" if LOGICLE_SOURCE != "asinh-fallback" else "asinh"

    # Store per-sample results for auditability/debugging
    per_sample = {}

    # ------------------------------
    # Global aggregation containers
    # ------------------------------
    # ✅ MMB global FSC window should use ONLY MMB files
    mmb_lefts = []
    mmb_rights = []

    # ✅ DG global fluorescence window should use ONLY DG files
    dg_low_cuts = []
    dg_high_cuts = []

    fluor_name, fsc_name = None, None

    for f in files:
        print(f"Loading: {f.name}")
        meta, df = fcsparser.parse(str(f), reformat_meta=True)
        print("Channels:", list(df.columns))

        # Determine channel names from the first file only (assumes consistency)
        if fluor_name is None or fsc_name is None:
            fluor_name, fsc_name = pick_channels(df)
            print(f"Selected channels → Fluor: {fluor_name}  |  FSC: {fsc_name}")

        # Apply transform
        df_t = df.copy()
        df_t[fluor_name] = apply_logicle_or_asinh(df_t[fluor_name].to_numpy())
        df_t[fsc_name] = apply_logicle_or_asinh(df_t[fsc_name].to_numpy())

        sample = f.stem
        is_dg = sample.upper().startswith("DG")
        is_mmb = not is_dg

        # -------------------------------------------------
        # Per-file FSC window (computed for ALL files)
        # -------------------------------------------------
        left, right = fsc_window(df_t[fsc_name].to_numpy(), drop_frac=0.05)

        # ✅ FIX: only aggregate into GLOBAL MMB window if this file is MMB
        if is_mmb:
            mmb_lefts.append(left)
            mmb_rights.append(right)

        # -------------------------------------------------
        # Per-file DG fluorescence window (DG only)
        # -------------------------------------------------
        dg_window = None
        if is_dg:
            low_cut, high_cut = dg_fluoro_window_bio_aligned(
                df_t[fluor_name].to_numpy(),
                shoulder_frac=0.05
            )
            dg_window = [low_cut, high_cut]
            dg_low_cuts.append(low_cut)
            dg_high_cuts.append(high_cut)

        # Decide output subfolders
        if is_dg:
            out_2d_dir = base_out_dir / "DG 2D Plots"
            out_1d_dir = base_out_dir / "DG 1D Plots"
        else:
            out_2d_dir = base_out_dir / "MMB 2D Plots"
            out_1d_dir = base_out_dir / "MMB 1D Plots"

        # ----------------------
        # 2D plot
        # ----------------------
        two_d_name = f"{sample}_{fsc_name} vs {fluor_name}"
        plot_2d_png(
            out_2d_dir,
            two_d_name,
            df_t[fluor_name].to_numpy(),
            df_t[fsc_name].to_numpy(),
            xlab=f"{fluor_name} ({transform_label})",
            ylab=f"{fsc_name} ({transform_label})",
            title=f"{sample}: {fsc_name} vs {fluor_name}",
        )

        # ----------------------
        # 1D size plot (always gated)
        # ----------------------
        size_name = f"{sample}_Size Distribution"
        plot_1d_png(
            out_1d_dir,
            size_name,
            df_t[fsc_name].to_numpy(),
            label=f"{fsc_name} ({transform_label})",
            gate=(left, right),
            title=f"{sample}: Size Distribution",
        )

        # ----------------------
        # 1D fluorescence plot
        #   DG: window shown
        #   MMB: no gate lines
        # ----------------------
        fluoro_name_out = f"{sample}_Fluoro Distribution"
        plot_1d_png(
            out_1d_dir,
            fluoro_name_out,
            df_t[fluor_name].to_numpy(),
            label=f"{fluor_name} ({transform_label})",
            gate=dg_window if is_dg else None,
            title=f"{sample}: Fluoro Distribution",
        )

        # Store per-sample gating values
        per_sample[f.name] = {
            "mmb_window": [left, right],
            "dg_window": dg_window
        }

    # ----------------------------
    # Global summary construction
    # ----------------------------

    # Global MMB window computed ONLY from MMB files
    if mmb_lefts and mmb_rights:
        global_mmb_window = [float(min(mmb_lefts)), float(max(mmb_rights))]
    else:
        global_mmb_window = None

    # Global DG window computed ONLY from DG files (conservative min/min)
    if dg_low_cuts and dg_high_cuts:
        global_dg_window = [float(min(dg_low_cuts)), float(min(dg_high_cuts))]
    else:
        global_dg_window = None

    summary = {
        "pattern": pattern,
        "channels": {"fluor": fluor_name, "fsc": fsc_name},
        "global": {
            "dg_window": global_dg_window,
            "mmb_window": global_mmb_window,
        },
        "per_sample": per_sample,
        "transform_source": LOGICLE_SOURCE,
        "dg_window_definition": {
            "method": "bio_aligned_shoulders_with_bump_detection",
            "shoulder_frac": 0.05,
            "interpretation": "Keep intermediate fluorescence between low/background and high/free-DG",
            "implementation_notes": {
                "high_cut": "left shoulder of dominant (global-max) KDE peak",
                "low_cut": "right shoulder of rightmost prominent background-side bump left of high_cut",
                "bump_prominence": "2% of global max density; with 5% of main-peak height sanity check"
            }
        }
    }
    return summary


# =============================================================================
# 9) main(): configure paths, run processing, write JSON summary
# =============================================================================
# - Processes all files using regex ".*"
# - Writes gate_summary_by_type.json into the output root folder
# =============================================================================

def main():
    folder = Path(
        r"D:\ICL Module Notes and more\Year 4\FYP\Software Automation\Raw Flow Data\MMB & DG Controls Unrefined"
    )

    base_out_dir = Path(
        r"D:\ICL Module Notes and more\Year 4\FYP\Software Automation\Visual Flow Data\MMB & DG Controls Refined"
    )

    patterns = [".*"]

    print(f"Python: {sys.executable}")
    print(f"CWD:    {os.getcwd()}")
    print(f"Input folder:  {folder}")
    print(f"Output root:   {base_out_dir}")
    print(f"Logicle source: {LOGICLE_SOURCE}")

    results = {}
    for pat in patterns:
        res = process_pattern(folder, pat, base_out_dir)
        if res is not None:
            results[pat] = res

    if results:
        base_out_dir.mkdir(parents=True, exist_ok=True)
        out_json = base_out_dir / "gate_summary_by_type.json"

        # Flatten to match your earlier JSON structure:
        # pattern -> { global gates + channels }
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(
                {k: (v["global"] | {"channels": v["channels"]}) for k, v in results.items()},
                fh,
                indent=2,
            )

        print(f"\nSaved combined summary -> {out_json}")
    else:
        print("\nNo matching files for any pattern—nothing to save.")


# =============================================================================
# Python entry point
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
