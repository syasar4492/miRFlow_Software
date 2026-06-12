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
#   4) Derives the MMB FSC window using KDE-based 1D density estimation.
#   5) Uses a fixed user-defined DG fluorescence low cut.
#   6) Produces 1D QC plots:
#       - MMB files: FSC-H density plots with local MMB size window boundaries.
#       - DG files: fluorescence density plots with the fixed DG low cut.
#   7) Writes a global summary JSON:
#       - mmb_window
#       - dg_low_cut
#       - dg_peak_fluorescence
#       - channels
#
# =============================================================================
# IMPORTANT VERSION D UPDATE: FIXED DG LOW CUT
# =============================================================================
# Previous versions derived dg_low_cut statistically from DG-only KDE shapes by:
#   - anchoring the dominant high/free-DG peak,
#   - searching to the left for a background-side bump,
#   - using the right shoulder of that bump as dg_low_cut.
#
# However, because the DG high boundary is now handled downstream by Module 3:
# Piecewise Linear Gating, the DG low cut can be simplified and made more
# transparent.
#
# This version therefore uses:
#
#   dg_low_cut = 0.080000000000
#
# This value is applied globally and exported into gate_summary_by_type.json.
# DG 1D fluorescence QC plots still show the low cut as a red dashed vertical line.
#
# =============================================================================

from pathlib import Path
import sys, os, json, re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
import fcsparser


# =============================================================================
# USER-DEFINED DG SETTINGS
# =============================================================================

# Fixed lower DG fluorescence cut used by Module 5 ROI gating.
# You can easily tune this value later, e.g. 0.07, 0.08, 0.09, 0.10.
DG_LOW_CUT_FIXED = 0.10000000000

# Fixed dominant DG/free-DG fluorescence anchor used downstream by Module 3
# to construct the patient-specific piecewise upper fluorescence boundary.
DG_PEAK_FLUORESCENCE = 0.800000000000


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
    """
    Derive an FSC window around the dominant MMB bead peak.

    The left and right boundaries are found by:
      - computing a KDE over FSC values,
      - finding the dominant peak,
      - walking left and right until density drops to 5% of peak height.
    """
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
# 6) DG logic: fixed fluorescence low cut
# =============================================================================

def dg_fluoro_low_cut_fixed() -> float:
    """
    Return the fixed user-defined DG fluorescence low cut.

    This replaces the previous KDE-derived background-bump algorithm.
    """
    return float(DG_LOW_CUT_FIXED)


# =============================================================================
# 7) Plotting helper: 1D KDE plot only
# =============================================================================

def plot_1d_png(
    out_dir: Path,
    filename_stem: str,
    values,
    label,
    gate=None,
    title: str | None = None,
    scalar_gate_colour: str = "crimson",
):
    """
    Save a 1D KDE density plot.

    Gate behaviour:
      - None → no gate lines
      - scalar → draw one dashed vertical line
      - list/tuple length 2 → draw two dashed vertical lines/window

    For DG fluorescence plots:
      - scalar_gate_colour is set to crimson to show the fixed DG low cut.
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
            plt.axvline(float(gate), color=scalar_gate_colour, ls="--", lw=1)

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

    # Global MMB window uses MMB files only.
    mmb_lefts = []
    mmb_rights = []

    fluor_name, fsc_name = None, None

    # The DG low cut is now fixed globally.
    global_dg_low_cut = dg_fluoro_low_cut_fixed()

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
        # Fixed DG low cut
        # Only DG files generate 1D fluorescence plots, but the same
        # fixed low cut is exported globally and used for all samples.
        # -------------------------------------------------
        dg_low_cut = global_dg_low_cut if is_dg else None

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
        # DG only: 1D fluorescence plot with fixed DG low cut
        # ----------------------
        if is_dg:
            fluoro_name_out = f"{sample}_Fluoro Distribution"

            plot_1d_png(
                out_dir=out_1d_dir,
                filename_stem=fluoro_name_out,
                values=df_t[fluor_name].to_numpy(),
                label=f"{fluor_name} ({transform_label})",
                gate=global_dg_low_cut,
                title=f"{sample}: Fluoro Distribution",
                scalar_gate_colour="crimson",
            )

        # Store per-sample gating values
        per_sample[f.name] = {
            "mmb_window": [left, right],
            "dg_low_cut": float(global_dg_low_cut) if is_dg else None,
            "dg_peak_fluorescence": float(DG_PEAK_FLUORESCENCE) if is_dg else None
        }

    # ----------------------------
    # Global summary construction
    # ----------------------------

    if mmb_lefts and mmb_rights:
        global_mmb_window = [float(min(mmb_lefts)), float(max(mmb_rights))]
    else:
        global_mmb_window = None

    summary = {
        "pattern": pattern,
        "channels": {
            "fluor": fluor_name,
            "fsc": fsc_name
        },
        "global": {
            "dg_low_cut": float(global_dg_low_cut),
            "dg_peak_fluorescence": float(DG_PEAK_FLUORESCENCE),
            "mmb_window": global_mmb_window,
        },
        "per_sample": per_sample,
        "transform_source": LOGICLE_SOURCE,
        "dg_window_definition": {
            "method": "fixed_low_cut_with_fixed_peak_anchor",
            "dg_low_cut_fixed": float(DG_LOW_CUT_FIXED),
            "dg_peak_fluorescence": float(DG_PEAK_FLUORESCENCE),
            "interpretation": (
                "Module 2 now exports a fixed user-defined lower DG fluorescence "
                "boundary rather than deriving dg_low_cut from DG-only KDE shape. "
                "The upper fluorescence boundary is intentionally not exported here "
                "because it is generated per patient in Module 3 using piecewise "
                "linear gating."
            ),
            "implementation_notes": {
                "dg_low_cut": (
                    "Fixed at 0.080000000000. This value is used to retain low-"
                    "fluorescence bead-only events while excluding very low "
                    "background events."
                ),
                "dg_peak_fluorescence": (
                    "Fixed fluorescence anchor currently set to 0.800000000000; "
                    "this is used downstream by Module 3 to build adaptive "
                    "patient-specific high-fluorescence limits."
                ),
                "plotting_change": (
                    "DG 1D fluorescence plots now display the fixed dg_low_cut "
                    "as a red dashed vertical line. Module 2 still only generates "
                    "MMB 1D size plots and DG 1D fluorescence plots."
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
        r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Raw Flow Data\MMB & DG Controls Unrefined"
    )

    base_out_dir = Path(
        r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Visual Flow Data\MMB & DG Controls Refined"
    )

    patterns = [".*"]

    print(f"Python: {sys.executable}")
    print(f"CWD:    {os.getcwd()}")
    print(f"Input folder:  {folder}")
    print(f"Output root:   {base_out_dir}")
    print(f"Logicle source: {LOGICLE_SOURCE}")
    print(f"Fixed DG low cut: {DG_LOW_CUT_FIXED}")
    print(f"Fixed DG peak fluorescence anchor: {DG_PEAK_FLUORESCENCE}")

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