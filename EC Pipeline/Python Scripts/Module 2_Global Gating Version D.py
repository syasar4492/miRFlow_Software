# Module 2_Global Gating Version D(1).py
#
# =============================================================================
# Module 2: Global Gating
# =============================================================================
# This script processes EC assay control .fcs files and generates the global gate
# inputs required by the downstream analysis modules.
#
# It performs four main tasks:
#   1) Reads all .fcs files in the selected control folder.
#   2) Detects the FSC and green fluorescence channels.
#   3) Applies a Logicle transform, with an asinh fallback.
#   4) Exports:
#        - the global MMB FSC window,
#        - a fixed DG fluorescence low cut,
#        - the DG fluorescence anchor used by the piecewise module,
#        - the selected channel names.
#
# The script also saves 1D QC plots:
#   - MMB controls: FSC density plots with the local size window shown.
#   - DG controls: fluorescence density plots with the fixed low cut shown.
#
# No patient-level classification is performed in this module.
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
# User settings
# =============================================================================

# Fixed lower fluorescence cut used later during ROI selection.
DG_LOW_CUT_FIXED = 0.100000000000

# Fixed fluorescence anchor used by the piecewise linear gating module.
DG_PEAK_FLUORESCENCE = 0.800000000000


# =============================================================================
# Signal transformation
# =============================================================================

try:
    from flowutils.transforms import logicle as logicle_transform
    LOGICLE_SOURCE = "flowutils.transforms.logicle"
except Exception:
    logicle_transform = None
    LOGICLE_SOURCE = "asinh-fallback"


def apply_logicle_or_asinh(arr: np.ndarray) -> np.ndarray:
    """
    Apply a Logicle transform when flowutils is available.
    Fall back to an asinh transform if Logicle cannot be imported.
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
# Channel selection
# =============================================================================

def first_match(columns, patterns):
    ups = [c.upper() for c in columns]

    for pat in patterns:
        P = pat.upper()

        for i, cu in enumerate(ups):
            if P in cu:
                return columns[i]

    return None


def pick_channels(df: pd.DataFrame):
    """
    Select the FSC channel and the main green fluorescence channel.
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
# Density estimation
# =============================================================================

def kde_density(values: np.ndarray, n=1000):
    """
    Return a 1D KDE curve for plotting and gate estimation.
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
# MMB FSC window
# =============================================================================

def fsc_window(values: np.ndarray, drop_frac=0.05):
    """
    Estimate the FSC window around the main MMB population.
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
# DG fluorescence low cut
# =============================================================================

def dg_fluoro_low_cut_fixed() -> float:
    """
    Return the fixed fluorescence low cut used for DG-associated signal.
    """
    return float(DG_LOW_CUT_FIXED)


# =============================================================================
# Plotting
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
    Save a 1D density plot with optional gate markers.
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
# Processing
# =============================================================================

def process_pattern(folder: Path, pattern: str, base_out_dir: Path):
    regex = re.compile(pattern, re.IGNORECASE)
    files = [f for f in sorted(folder.glob("*.fcs")) if regex.search(f.name)]

    print(f"\nPattern '{pattern}' -> {len(files)} files")

    if not files:
        print(f"Skipping: no .fcs files matching /{pattern}/ in {folder}")
        return None

    transform_label = "logicle" if LOGICLE_SOURCE != "asinh-fallback" else "asinh"

    per_sample = {}

    mmb_lefts = []
    mmb_rights = []

    fluor_name, fsc_name = None, None
    global_dg_low_cut = dg_fluoro_low_cut_fixed()

    for f in files:
        print(f"Loading: {f.name}")

        meta, df = fcsparser.parse(str(f), reformat_meta=True)
        print("Channels:", list(df.columns))

        if fluor_name is None or fsc_name is None:
            fluor_name, fsc_name = pick_channels(df)
            print(f"Selected channels -> Fluor: {fluor_name}  |  FSC: {fsc_name}")

        df_t = df.copy()
        df_t[fluor_name] = apply_logicle_or_asinh(df_t[fluor_name].to_numpy())
        df_t[fsc_name] = apply_logicle_or_asinh(df_t[fsc_name].to_numpy())

        sample = f.stem
        is_dg = sample.upper().startswith("DG")
        is_mmb = not is_dg

        left, right = fsc_window(df_t[fsc_name].to_numpy(), drop_frac=0.05)

        if is_mmb:
            mmb_lefts.append(left)
            mmb_rights.append(right)

        dg_low_cut = global_dg_low_cut if is_dg else None

        if is_dg:
            out_1d_dir = base_out_dir / "DG 1D Plots"
        else:
            out_1d_dir = base_out_dir / "MMB 1D Plots"

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

        per_sample[f.name] = {
            "mmb_window": [left, right],
            "dg_low_cut": float(global_dg_low_cut) if is_dg else None,
            "dg_peak_fluorescence": float(DG_PEAK_FLUORESCENCE) if is_dg else None,
        }

    if mmb_lefts and mmb_rights:
        global_mmb_window = [float(min(mmb_lefts)), float(max(mmb_rights))]
    else:
        global_mmb_window = None

    summary = {
        "pattern": pattern,
        "channels": {
            "fluor": fluor_name,
            "fsc": fsc_name,
        },
        "global": {
            "dg_low_cut": float(global_dg_low_cut),
            "dg_peak_fluorescence": float(DG_PEAK_FLUORESCENCE),
            "mmb_window": global_mmb_window,
        },
        "per_sample": per_sample,
        "transform_source": LOGICLE_SOURCE,
        "dg_window_definition": {
            "method": "fixed_low_cut_with_piecewise_high_boundary",
            "dg_low_cut_fixed": float(DG_LOW_CUT_FIXED),
            "dg_peak_fluorescence": float(DG_PEAK_FLUORESCENCE),
            "notes": (
                "Module 2 exports a fixed lower fluorescence cut and a fixed "
                "fluorescence anchor. The patient-level upper fluorescence "
                "boundary is generated in the piecewise linear gating module."
            ),
        },
    }

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    folder = Path(
        r" "
    )

    base_out_dir = Path(
        r" "
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

        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(
                {k: (v["global"] | {"channels": v["channels"]}) for k, v in results.items()},
                fh,
                indent=2,
            )

        print(f"\nSaved combined summary -> {out_json}")

    else:
        print("\nNo matching files for any pattern. Nothing to save.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
