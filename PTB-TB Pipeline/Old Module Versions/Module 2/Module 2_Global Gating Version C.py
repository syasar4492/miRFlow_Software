# advanced_fcs_viewer_FSC-H_logicle_named_outputs_subfolders_DG_WINDOW_FULLCOMMENTS.py
#
# =============================================================================
# HIGH-LEVEL OVERVIEW (what this script does and why)
# =============================================================================
# This script automates a control-derived 1D gating workflow for your MMB and DG
# control FCS files.
#
# It:
#   1) Reads all .fcs files from an "unrefined" folder.
#   2) Auto-detects the FSC size channel and the main green fluorescence channel.
#   3) Applies a Logicle transform, with an asinh fallback.
#   4) Derives per-file gating boundaries from KDE-based 1D density estimates:
#
#      A) MMB FSC window: [left, right]
#         - A dominant bead peak window around FSC-H.
#         - Found by walking left/right from the dominant KDE peak until the density
#           drops to 5% of the peak height.
#         - Only MMB files contribute to the final global MMB size window.
#
#      B) DG fluorescence window: [low_cut, high_cut]
#         - Biological goal: keep the fluorescence region enriched for potentially
#           useful/functionalised DG-associated events while excluding:
#             (i) low/background events
#             (ii) only the extreme high-fluorescence/free-DG tail
#
#         - Low_cut logic:
#             * Anchor the dominant high/free-DG population using the KDE global maximum.
#             * Compute the LEFT shoulder of that dominant peak.
#             * Search only to the LEFT of that left shoulder for a background-side bump.
#             * low_cut = right shoulder of that background-side bump.
#
#         - High_cut logic:
#             * Anchor the dominant high/free-DG population using the KDE global maximum.
#             * high_cut = RIGHT shoulder of that dominant peak using the same 5% rule.
#
#         - This revised high_cut is less aggressive than the previous version, which
#           used the left shoulder of the dominant peak and excluded too many viable
#           high-fluorescence events.
#
#   5) Produces only the 1D plots needed for gate QC:
#       - MMB files: 1D FSC/size density plots with local size boundaries.
#       - DG files: 1D fluorescence density plots with local DG windows.
#
#   6) Writes a global summary JSON:
#       - mmb_window aggregated across MMB files only:
#           [min(left_i), max(right_i)]
#       - dg_window aggregated across DG files only:
#           GLOBAL_LOW  = min(low_cut_i)
#           GLOBAL_HIGH = min(high_cut_i)
#
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
# 1) TRANSFORMATION: Logicle preferred, with asinh fallback
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
# 2) Utility: pick a channel by matching patterns in column names
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
# 3) Channel selection logic
# =============================================================================

def pick_channels(df: pd.DataFrame):
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
# 4) 1D density estimation using Gaussian KDE
# =============================================================================

def kde_density(values: np.ndarray, n=1000):
    """
    Compute KDE density for a 1D array.
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
# Revised high_cut logic:
#   - Previous method: high_cut = LEFT shoulder of dominant high/free-DG peak.
#   - New method:      high_cut = RIGHT shoulder of dominant high/free-DG peak.
#
# Low_cut logic remains based on the original strategy:
#   - Compute the LEFT shoulder of the dominant high/free-DG peak.
#   - Search only to the LEFT of that left shoulder for a background-side bump.
#   - low_cut = right shoulder of that bump.
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
    # (1) High peak = KDE global maximum
    # -------------------------------------------------------------
    high_peak = int(np.argmax(ys))
    high_peak_h = float(ys[high_peak])
    high_thr = shoulder_frac * high_peak_h

    # -------------------------------------------------------------
    # (2a) LEFT shoulder of high peak
    # Used as anchor point for the low_cut search.
    # -------------------------------------------------------------
    left_shoulder_idx = high_peak

    while left_shoulder_idx > 0 and ys[left_shoulder_idx] > high_thr:
        left_shoulder_idx -= 1

    # -------------------------------------------------------------
    # (2b) high_cut = RIGHT shoulder of high peak
    # -------------------------------------------------------------
    right_shoulder_idx = high_peak

    while right_shoulder_idx < len(ys) - 1 and ys[right_shoulder_idx] > high_thr:
        right_shoulder_idx += 1

    high_cut = float(xs[right_shoulder_idx])

    # Guard for degenerate cases where there is not enough left-side region
    if left_shoulder_idx < 10:
        lo = float(np.percentile(v, 30))
        hi = float(np.percentile(v, 90))

        if lo < hi:
            return lo, hi

        return float(np.min(v)), float(np.max(v))

    # -------------------------------------------------------------
    # (3) Find meaningful bump on the left side for low_cut
    # -------------------------------------------------------------
    left_region_ys = ys[:left_shoulder_idx]
    left_region_xs = xs[:left_shoulder_idx]

    left_max = float(np.max(left_region_ys)) if left_region_ys.size else 0.0

    if left_max <= 0.0:
        low_cut = float(np.percentile(v, 30))

        if low_cut < high_cut:
            return low_cut, high_cut

        lo = float(np.percentile(v, 35))
        hi = float(np.percentile(v, 90))

        return (lo, hi) if lo < hi else (float(np.min(v)), float(np.max(v)))

    # Prominence threshold to ignore tiny KDE wiggles
    prom = 0.02 * float(np.max(ys))
    peaks, props = find_peaks(left_region_ys, prominence=prom)

    if peaks.size > 0:
        # Use rightmost peak on left side, closest to the high/free-DG population
        low_peak = int(peaks[np.argmax(peaks)])
        low_peak_h = float(left_region_ys[low_peak])

        # Sanity check: if bump is extremely small, fallback to left-region max
        if low_peak_h < 0.05 * float(np.max(ys)):
            low_peak = int(np.argmax(left_region_ys))
            low_peak_h = float(left_region_ys[low_peak])
    else:
        low_peak = int(np.argmax(left_region_ys))
        low_peak_h = float(left_region_ys[low_peak])

    # low_cut = right shoulder of chosen bump using 5% rule
    low_thr = shoulder_frac * low_peak_h
    i = low_peak

    while i < left_region_ys.size - 1 and left_region_ys[i] > low_thr:
        i += 1

    low_cut = float(left_region_xs[i])

    # Enforce proper ordering
    if not (low_cut < high_cut):
        lo = float(np.percentile(v, 35))
        hi = float(np.percentile(v, 90))

        if lo < hi:
            return lo, hi

        return float(np.min(v)), float(np.max(v))

    return low_cut, high_cut


# =============================================================================
# 7) Plotting helper: 1D KDE plot only
# =============================================================================

def plot_1d_png(out_dir: Path, filename_stem: str, values, label, gate=None, title: str | None = None):
    """
    Save a 1D KDE density plot.

    Gate behaviour:
      - None → no gate lines
      - scalar → draw one vertical line
      - list/tuple length 2 → draw two vertical lines/window
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
# 8) Core processing: loop over FCS files, gate, save 1D plots, and summarise
# =============================================================================

def process_pattern(folder: Path, pattern: str, base_out_dir: Path):
    regex = re.compile(pattern, re.IGNORECASE)
    files = [f for f in sorted(folder.glob("*.fcs")) if regex.search(f.name)]

    print(f"\nPattern '{pattern}' -> {len(files)} files")

    if not files:
        print(f"⚠️ Skipping: no .fcs matching /{pattern}/ in {folder}")
        return None

    transform_label = "logicle" if LOGICLE_SOURCE != "asinh-fallback" else "asinh"

    per_sample = {}

    # Global MMB window uses MMB files only
    mmb_lefts = []
    mmb_rights = []

    # Global DG fluorescence window uses DG files only
    dg_low_cuts = []
    dg_high_cuts = []

    fluor_name, fsc_name = None, None

    for f in files:
        print(f"Loading: {f.name}")

        meta, df = fcsparser.parse(str(f), reformat_meta=True)
        print("Channels:", list(df.columns))

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
        # Per-file FSC window
        # Only MMB files contribute to the global MMB window.
        # Only MMB files generate 1D size plots.
        # -------------------------------------------------
        left, right = fsc_window(df_t[fsc_name].to_numpy(), drop_frac=0.05)

        if is_mmb:
            mmb_lefts.append(left)
            mmb_rights.append(right)

        # -------------------------------------------------
        # Per-file DG fluorescence window
        # Only DG files contribute to the global DG window.
        # Only DG files generate 1D fluorescence plots.
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

        # -------------------------------------------------
        # Output folders for 1D plots only
        # -------------------------------------------------
        if is_dg:
            out_1d_dir = base_out_dir / "DG 1D Plots"
        else:
            out_1d_dir = base_out_dir / "MMB 1D Plots"

        # ----------------------
        # MMB only: 1D size plot with local size boundaries
        # ----------------------
        if is_mmb:
            size_name = f"{sample}_Size Distribution"

            plot_1d_png(
                out_dir=out_1d_dir,
                filename_stem=size_name,
                values=df_t[fsc_name].to_numpy(),
                label=f"{fsc_name} ({transform_label})",
                gate=(left, right),
                title=f"{sample}: Size Distribution",
            )

        # ----------------------
        # DG only: 1D fluorescence plot with local DG window
        # ----------------------
        if is_dg:
            fluoro_name_out = f"{sample}_Fluoro Distribution"

            plot_1d_png(
                out_dir=out_1d_dir,
                filename_stem=fluoro_name_out,
                values=df_t[fluor_name].to_numpy(),
                label=f"{fluor_name} ({transform_label})",
                gate=dg_window,
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

    if mmb_lefts and mmb_rights:
        global_mmb_window = [float(min(mmb_lefts)), float(max(mmb_rights))]
    else:
        global_mmb_window = None

    if dg_low_cuts and dg_high_cuts:
        global_dg_window = [float(min(dg_low_cuts)), float(min(dg_high_cuts))]
    else:
        global_dg_window = None

    summary = {
        "pattern": pattern,
        "channels": {
            "fluor": fluor_name,
            "fsc": fsc_name
        },
        "global": {
            "dg_window": global_dg_window,
            "mmb_window": global_mmb_window,
        },
        "per_sample": per_sample,
        "transform_source": LOGICLE_SOURCE,
        "dg_window_definition": {
            "method": "bio_aligned_low_cut_with_right_shoulder_high_cut",
            "shoulder_frac": 0.05,
            "interpretation": (
                "Keep fluorescence between the low/background boundary and the "
                "right shoulder of the dominant high/free-DG population."
            ),
            "implementation_notes": {
                "high_cut": "right shoulder of dominant/global-max KDE peak at 5% of peak height",
                "low_cut": (
                    "right shoulder of the rightmost prominent background-side bump, "
                    "searched left of the dominant peak's left shoulder"
                ),
                "bump_prominence": "2% of global max density; with 5% of main-peak height sanity check",
                "plotting_change": (
                    "2D scatter plotting removed from Module 2 because Module 1 now handles "
                    "FSC-H vs B525-H visualisation. Module 2 now only generates MMB 1D size "
                    "plots and DG 1D fluorescence plots."
                )
            }
        }
    }

    return summary


# =============================================================================
# 9) main(): configure paths, run processing, write JSON summary
# =============================================================================

def main():
    folder = Path(
        r"D:\ICL MBE\Year 4\FYP\Software Automation\EC\Raw Flow Data\MMB & DG Controls Unrefined"
    )

    base_out_dir = Path(
        r"D:\ICL MBE\Year 4\FYP\Software Automation\EC\Visual Flow Data\MMB and DG Controls Refined"
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
        print("\nNo matching files for any pattern. Nothing to save.")


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