# mmb_size_thresholds_from_controls.py
#
# =============================================================================
# PURPOSE
# =============================================================================
# Side-script to derive FSC-H separation thresholds for MMB-3 / MMB-6 / MMB-8
# using the SAME logicle transformation logic as your Advanced FCS Viewer.
#
# It:
#   1) Loads all .fcs files in your specified input folder.
#   2) Filters to MMB-only files labelled like: 3MMB_1.fcs, 6MMB_2.fcs, 8MMB_3.fcs
#   3) Auto-detects FSC channel (prefers FSC-H).
#   4) Applies Logicle transform.
#   5) For each file:
#        - builds a KDE of FSC
#        - finds the dominant peak
#        - computes the LEFT shoulder where KDE drops to a chosen fraction of peak height
#          => "start_fsc" for that bead population
#   6) Aggregates across replicates per bead size and exports JSON:
#        - per-file peak_fsc, start_fsc
#        - per-size median start_fsc
#        - recommended boundaries:
#            boundary_3_6 = median(start_fsc of 6MMB using 5% shoulder)
#            boundary_6_8 = median(start_fsc of 8MMB using 20% shoulder)
#
# ADDITIONAL QC OUTPUTS
# =============================================================================
# In addition to the JSON output, this script now also saves quality-control PNGs
# for the boundary-defining MMB populations:
#
#   - 6MMB files:
#       used to derive boundary_3_6
#       plotted with the file-specific 5% left shoulder start_fsc
#
#   - 8MMB files:
#       used to derive boundary_6_8
#       plotted with the file-specific 20% left shoulder start_fsc
#
# Outputs:
#   - one individual 1D density vs FSC-H plot per relevant 6MMB/8MMB file
#   - one collated overview grid PNG for all 6MMB plots
#   - one collated overview grid PNG for all 8MMB plots
#
# Each plot shows:
#   - KDE density curve
#   - blue dotted vertical line marking the file-specific start_fsc
#
# =============================================================================
# IMPORTANT UPDATE
# =============================================================================
# Previously, the same 5% left-shoulder rule was used for both boundaries.
#
# Because the 6MMB and 8MMB FSC-H populations are close together, the 5% left
# shoulder of the 8MMB population can sit too far left, causing high-FSC 6MMB
# events to be misclassified as 8MMB.
#
# After testing, the best empirical cut value for the MMB 6:8 boundary was found
# to be the 20% left shoulder of the 8MMB KDE peak.
#
# Therefore:
#   - boundary_3_6 uses the 5% left shoulder of 6MMB
#   - boundary_6_8 uses the 20% left shoulder of 8MMB
#
# A 20% shoulder stops closer to the 8MMB peak, raising the 6:8 boundary and
# reducing the chance that 6MMB events are incorrectly assigned as 8MMB.
#
# =============================================================================
# YOUR PATHS
# =============================================================================

from __future__ import annotations

from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fcsparser


# =============================================================================
# USER-EDITABLE BOUNDARY SHOULDER FRACTIONS
# =============================================================================
# These control how far left from the dominant KDE peak the algorithm walks before
# defining the "start" of that bead population.
#
# Smaller value, e.g. 0.05:
#   - walks further left
#   - gives a lower boundary
#
# Larger value, e.g. 0.20:
#   - stops closer to the peak
#   - gives a higher boundary
#
# Current strategy:
#   - 3:6 boundary uses 5%
#   - 6:8 boundary uses 20%
# =============================================================================

BOUNDARY_3_6_DROP_FRAC = 0.05
BOUNDARY_6_8_DROP_FRAC = 0.20


# =============================================================================
# QC PLOT OUTPUT SETTINGS
# =============================================================================

QC_PLOTS_SUBFOLDER = "MMB Threshold QC Plots"
MAX_PANELS_PER_ROW = 3


# ---------------------------------------------------------------------
# 1) Logicle transform
# ---------------------------------------------------------------------
try:
    from flowutils.transforms import logicle as logicle_transform
    LOGICLE_SOURCE = "flowutils.transforms.logicle"
except Exception:
    logicle_transform = None
    LOGICLE_SOURCE = None


def apply_logicle_required(arr: np.ndarray) -> np.ndarray:
    """
    Apply Logicle via flowutils.transforms.logicle. This is REQUIRED.

    Mirrors your Advanced FCS Viewer:
      T = max(262144, vmax)
      M = 4.5
      W = 0.5
      A = 0.5 if vmin < 0 else 0.0
    """
    if logicle_transform is None:
        raise RuntimeError(
            "flowutils logicle is not available but logicle is required.\n"
            "Install with: pip install flowutils"
        )

    arr = arr.astype(float, copy=False)
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return arr

    vmax = float(np.max(v))
    vmin = float(np.min(v))
    T = max(262144.0, vmax)
    M = 4.5
    W = 0.5
    A = 0.5 if vmin < 0 else 0.0

    return logicle_transform(arr, channel_indices=None, t=T, m=M, w=W, a=A)


# ---------------------------------------------------------------------
# 2) Channel matching helpers
# ---------------------------------------------------------------------
def first_match(columns, patterns):
    ups = [c.upper() for c in columns]
    for pat in patterns:
        P = pat.upper()
        for i, cu in enumerate(ups):
            if P in cu:
                return columns[i]
    return None


def pick_fsc_channel(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fsc = first_match(cols, ["FSC-H"]) or first_match(cols, ["FSC-A", "FSC"])
    if fsc is None:
        raise ValueError(f"No FSC channel found in columns: {cols}")
    return fsc


# ---------------------------------------------------------------------
# 3) KDE density and left-shoulder threshold
# ---------------------------------------------------------------------
def kde_density(values: np.ndarray, n: int = 1024):
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


def dominant_peak_left_shoulder(values: np.ndarray, drop_frac: float = 0.05) -> tuple[float, float]:
    """
    Returns:
      peak_fsc  = location of dominant KDE peak
      start_fsc = left shoulder where KDE drops to drop_frac of peak height
    """
    xs, ys = kde_density(values, n=2048)

    if not np.any(ys):
        v = values[np.isfinite(values)]
        if v.size == 0:
            return 0.0, 1.0
        peak_fsc = float(np.median(v))
        start_fsc = float(np.percentile(v, 10))
        return peak_fsc, start_fsc

    peak_idx = int(np.argmax(ys))
    ypk = float(ys[peak_idx])
    thr = drop_frac * ypk

    j = peak_idx
    while j > 0 and ys[j] > thr:
        j -= 1

    peak_fsc = float(xs[peak_idx])
    start_fsc = float(xs[j])
    return peak_fsc, start_fsc


# ---------------------------------------------------------------------
# 4) Filename parsing
# ---------------------------------------------------------------------
def parse_mmb_size_from_filename(p: Path) -> int | None:
    stem = p.stem.upper()
    for s in (3, 6, 8):
        if stem.startswith(f"{s}MMB"):
            return s
    return None


# ---------------------------------------------------------------------
# 5) Plot helpers
# ---------------------------------------------------------------------
def safe_filename(name: str) -> str:
    return "".join(c if c not in r'<>:"/\|?*' else "_" for c in name)


def save_single_density_plot_png(
    out_dir: Path,
    stem: str,
    xs: np.ndarray,
    ys: np.ndarray,
    boundary_x: float,
    x_label: str,
    title: str,
):
    """
    Save one individual 1D density plot for a boundary-defining MMB file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.0, 4.5))
    plt.plot(xs, ys, lw=2.5, color="tab:blue")
    plt.axvline(boundary_x, color="tab:blue", ls=":", lw=2)

    plt.xlabel(x_label)
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()

    out_path = out_dir / f"{safe_filename(stem)}.png"
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"Saved individual QC plot: {out_path}")


def save_density_grid_overview_png(
    out_dir: Path,
    out_name: str,
    samples: list[dict],
    x_label: str,
    suptitle: str,
):
    """
    Save a collated overview grid for a set of MMB threshold QC density plots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    if n == 0:
        return None

    ncols = min(MAX_PANELS_PER_ROW, n)
    nrows = int(math.ceil(n / ncols))

    fig_w = max(10, 4.4 * ncols)
    fig_h = max(4.0, 3.6 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        squeeze=False
    )

    for idx, s in enumerate(samples):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        ax.plot(s["xs"], s["ys"], lw=2.0, color="tab:blue")
        ax.axvline(s["boundary_x"], color="tab:blue", ls=":", lw=1.8)

        ax.set_title(s["name"], fontsize=10)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    out_path = out_dir / f"{safe_filename(out_name)}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved collated QC plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# 6) Defaults
# ---------------------------------------------------------------------
DEFAULT_INPUT_DIR = r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Raw Flow Data\MMB & DG Controls Unrefined"
DEFAULT_OUTPUT_DIR = r"D:\ICL MBE\Year 4\FYP\Software Automation\Pre Term Birth\Visual Flow Data\Json Files"
DEFAULT_OUT_NAME = "mmb_fsc_thresholds.json"


# ---------------------------------------------------------------------
# 7) Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing MMB control .fcs files"
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder to save the output JSON"
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default=DEFAULT_OUT_NAME,
        help="Output JSON filename"
    )
    ap.add_argument(
        "--min_events",
        type=int,
        default=500,
        help="Minimum events required per file"
    )
    args = ap.parse_args()

    if logicle_transform is None:
        raise SystemExit(
            "Logicle transform is required but flowutils is not available.\n"
            "Please install: pip install flowutils"
        )

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_json = out_dir / args.out_name
    qc_plot_dir = out_dir / QC_PLOTS_SUBFOLDER

    if not in_dir.exists():
        raise SystemExit(f"Input directory does not exist:\n  {in_dir}")

    files = sorted(in_dir.glob("*.fcs"))
    if not files:
        raise SystemExit(f"No .fcs files found in:\n  {in_dir}")

    # Find first valid MMB file to pick the FSC channel
    fsc_channel = None
    for fp in files:
        if parse_mmb_size_from_filename(fp) is None:
            continue

        meta, df0 = fcsparser.parse(str(fp), reformat_meta=True)
        fsc_channel = pick_fsc_channel(df0)
        break

    if fsc_channel is None:
        raise SystemExit(
            "No files matched 3MMB/6MMB/8MMB naming.\n"
            "Expected e.g. '3MMB_1.fcs' in the input directory."
        )

    per_file = []
    per_size = {3: [], 6: [], 8: []}

    # QC plots are only generated for boundary-defining populations:
    #   - 6MMB for boundary_3_6
    #   - 8MMB for boundary_6_8
    qc_panels_by_size = {
        6: [],
        8: [],
    }

    for fp in files:
        bead_size = parse_mmb_size_from_filename(fp)
        if bead_size is None:
            continue

        meta, df = fcsparser.parse(str(fp), reformat_meta=True)

        if fsc_channel not in df.columns:
            raise SystemExit(
                f"FSC channel '{fsc_channel}' not found in {fp.name}.\n"
                f"Columns: {list(df.columns)}"
            )

        raw_fsc = df[fsc_channel].to_numpy(dtype=float)
        fsc_t = apply_logicle_required(raw_fsc)
        fsc_t = fsc_t[np.isfinite(fsc_t)]

        if fsc_t.size < args.min_events:
            per_file.append({
                "file": fp.name,
                "mmb_size": bead_size,
                "n_used": int(fsc_t.size),
                "status": "skipped_too_few_events",
            })
            continue

        # -------------------------------------------------------------
        # Use bead-size-specific shoulder fractions.
        #
        # 3MMB:
        #   Not directly used as a recommended boundary, but kept at 5%
        #   for consistency and auditability.
        #
        # 6MMB:
        #   Used to derive boundary_3_6, so use 5%.
        #
        # 8MMB:
        #   Used to derive boundary_6_8, so use 20% to shift the boundary
        #   closer to the 8MMB peak and reduce 6MMB misclassification.
        # -------------------------------------------------------------
        if bead_size == 8:
            drop_frac_used = BOUNDARY_6_8_DROP_FRAC
        else:
            drop_frac_used = BOUNDARY_3_6_DROP_FRAC

        peak_fsc, start_fsc = dominant_peak_left_shoulder(
            fsc_t,
            drop_frac=drop_frac_used
        )

        rec = {
            "file": fp.name,
            "mmb_size": bead_size,
            "n_used": int(fsc_t.size),
            "fsc_channel": fsc_channel,
            "drop_frac_used": float(drop_frac_used),
            "peak_fsc": float(peak_fsc),
            "start_fsc": float(start_fsc),
            "status": "ok",
        }

        per_file.append(rec)
        per_size[bead_size].append(rec)

        # -------------------------------------------------------------
        # Save QC plots for boundary-defining files only:
        #
        #   6MMB → boundary_3_6
        #   8MMB → boundary_6_8
        #
        # Each plot shows the KDE size distribution and the file-specific
        # start_fsc as a blue dotted vertical line.
        # -------------------------------------------------------------
        if bead_size in (6, 8):
            xs_plot, ys_plot = kde_density(fsc_t, n=2048)

            if bead_size == 6:
                boundary_label = "boundary_3_6"
                shoulder_label = "5% left shoulder"
            else:
                boundary_label = "boundary_6_8"
                shoulder_label = "20% left shoulder"

            size_plot_dir = qc_plot_dir / f"MMB-{bead_size} Boundary QC Plots"

            plot_stem = f"{fp.stem}_size_distribution"
            plot_title = f"{fp.stem}: Size Distribution ({shoulder_label})"

            save_single_density_plot_png(
                out_dir=size_plot_dir,
                stem=plot_stem,
                xs=xs_plot,
                ys=ys_plot,
                boundary_x=start_fsc,
                x_label=f"{fsc_channel} (logicle)",
                title=plot_title,
            )

            qc_panels_by_size[bead_size].append({
                "name": f"{fp.stem} ({boundary_label})",
                "xs": xs_plot,
                "ys": ys_plot,
                "boundary_x": start_fsc,
            })

    # -----------------------------------------------------------------
    # Aggregate per size using robust medians
    # -----------------------------------------------------------------
    summary = {}

    for s in (3, 6, 8):
        rows = per_size[s]

        if not rows:
            summary[str(s)] = {"n_files": 0}
            continue

        starts = np.array([r["start_fsc"] for r in rows], dtype=float)
        peaks = np.array([r["peak_fsc"] for r in rows], dtype=float)
        drop_fracs = np.array([r["drop_frac_used"] for r in rows], dtype=float)

        summary[str(s)] = {
            "n_files": int(len(rows)),
            "drop_frac_used": float(np.median(drop_fracs)),
            "start_fsc_median": float(np.median(starts)),
            "start_fsc_iqr": [
                float(np.quantile(starts, 0.25)),
                float(np.quantile(starts, 0.75))
            ],
            "peak_fsc_median": float(np.median(peaks)),
            "peak_fsc_iqr": [
                float(np.quantile(peaks, 0.25)),
                float(np.quantile(peaks, 0.75))
            ],
        }

    boundary_3_6 = summary.get("6", {}).get("start_fsc_median", None)
    boundary_6_8 = summary.get("8", {}).get("start_fsc_median", None)

    collated_plot_paths = {}

    if qc_panels_by_size[6]:
        collated_plot_paths["MMB-6"] = save_density_grid_overview_png(
            out_dir=qc_plot_dir / "MMB-6 Boundary QC Plots",
            out_name="ALL_MMB6_size_distributions",
            samples=qc_panels_by_size[6],
            x_label=f"{fsc_channel} (logicle)",
            suptitle="MMB-6 Size Distributions for boundary_3_6"
        )

    if qc_panels_by_size[8]:
        collated_plot_paths["MMB-8"] = save_density_grid_overview_png(
            out_dir=qc_plot_dir / "MMB-8 Boundary QC Plots",
            out_name="ALL_MMB8_size_distributions",
            samples=qc_panels_by_size[8],
            x_label=f"{fsc_channel} (logicle)",
            suptitle="MMB-8 Size Distributions for boundary_6_8"
        )

    out = {
        "script": "mmb_size_thresholds_from_controls.py",
        "paths": {
            "input_dir": str(in_dir),
            "output_json": str(out_json),
        },
        "transform": {
            "name": "logicle",
            "source": LOGICLE_SOURCE,
            "params_rule": {
                "T": "max(262144, observed vmax)",
                "M": 4.5,
                "W": 0.5,
                "A": "0.5 if vmin < 0 else 0.0",
            },
        },
        "fsc_channel": fsc_channel,
        "shoulder_fraction_strategy": {
            "boundary_3_6": {
                "source_population": "6MMB",
                "drop_frac": float(BOUNDARY_3_6_DROP_FRAC),
                "definition": "median 5% left shoulder of 6MMB controls"
            },
            "boundary_6_8": {
                "source_population": "8MMB",
                "drop_frac": float(BOUNDARY_6_8_DROP_FRAC),
                "definition": "median 20% left shoulder of 8MMB controls"
            },
            "rationale": (
                "The 6MMB and 8MMB FSC-H distributions are close together. "
                "After empirical optimisation, a 20% left shoulder for 8MMB was selected. "
                "This shifts boundary_6_8 closer to the 8MMB peak, reducing the likelihood "
                "that high-FSC 6MMB events are misclassified as 8MMB."
            )
        },
        "qc_plot_outputs": {
            "folder": str(qc_plot_dir),
            "individual_plots": {
                "MMB-6": "One PNG per valid 6MMB file used for boundary_3_6",
                "MMB-8": "One PNG per valid 8MMB file used for boundary_6_8",
            },
            "collated_plots": {
                key: str(value) if value is not None else None
                for key, value in collated_plot_paths.items()
            },
        },
        "min_events_per_file": int(args.min_events),
        "per_size_summary": summary,
        "recommended_boundaries": {
            "boundary_3_6": boundary_3_6,
            "boundary_6_8": boundary_6_8,
            "interpretation": (
                "boundary_3_6 is defined as the median 5% LEFT shoulder of the 6MMB population. "
                "boundary_6_8 is defined as the median 20% LEFT shoulder of the 8MMB population. "
                "The 20% rule for 8MMB was selected after empirical optimisation because the "
                "6MMB and 8MMB FSC-H distributions are close."
            ),
            "assignment_rule": {
                "MMB-3": "FSC < boundary_3_6  (within bead-like region)",
                "MMB-6": "boundary_3_6 <= FSC < boundary_6_8",
                "MMB-8": "FSC >= boundary_6_8",
            },
        },
        "per_file_estimates": per_file,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print("\nRecommended boundaries:")
    print(f"  boundary_3_6 = {boundary_3_6}")
    print(f"  boundary_6_8 = {boundary_6_8}")
    print("\nShoulder strategy:")
    print(f"  3:6 uses {BOUNDARY_3_6_DROP_FRAC * 100:.1f}% left shoulder of 6MMB")
    print(f"  6:8 uses {BOUNDARY_6_8_DROP_FRAC * 100:.1f}% left shoulder of 8MMB")

    if qc_panels_by_size[6] or qc_panels_by_size[8]:
        print("\nQC plots:")
        print(f"  Individual + collated PNGs saved to: {qc_plot_dir}")


if __name__ == "__main__":
    main()