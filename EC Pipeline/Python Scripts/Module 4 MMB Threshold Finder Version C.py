# mmb_size_thresholds_from_controls.py
#
# =============================================================================
# PURPOSE
# =============================================================================
# Module 4 derives the FSC-H threshold used to separate the two magnetic
# microbead populations in the EC assay.
#
# The workflow is:
#   1) Load MMB control .fcs files from the selected input folder.
#   2) Detect the FSC channel, preferring FSC-H where available.
#   3) Apply the same Logicle transform used by the wider miRFlow pipeline.
#   4) Estimate a 1D FSC-H density profile for each control file.
#   5) Locate the dominant bead peak and define the left shoulder of the
#      reference population using a 20% density-drop rule.
#   6) Export the median control-derived threshold to mmb_fsc_thresholds.json.
#
# The EC assay uses two bead populations, referred to here as MMB-A and MMB-B.
# The exported numerical threshold is kept under the existing boundary_6_8 key
# so that downstream modules remain compatible.
#
# QC outputs are also generated:
#   - one 1D FSC-H density plot for each MMB-A control file
#   - one collated overview PNG containing all MMB-A density plots
#
# Each QC plot shows the KDE density curve and the file-specific start_fsc
# boundary as a blue dotted vertical line.
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
# USER-EDITABLE BOUNDARY SHOULDER FRACTION
# =============================================================================
# This controls how far left from the dominant MMB-A KDE peak the algorithm walks
# before defining the start_fsc value. A larger value stops closer to the peak
# and gives a stricter separation threshold.
# =============================================================================

BOUNDARY_6_8_DROP_FRAC = 0.20


# =============================================================================
# QC PLOT OUTPUT SETTINGS
# =============================================================================
QC_PLOTS_SUBFOLDER = "MMB-A Boundary QC Plots"
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
    Apply the Logicle transform used throughout the miRFlow pipeline.

    Parameters follow the same rule used by the other modules:
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
    """
    Parse the internal bead-size code from the control filename.

    The returned numeric code is used only for compatibility with the existing
    downstream JSON structure. In comments and output descriptions, these bead
    populations are referred to as MMB-A and MMB-B.
    """
    stem = p.stem.upper()

    for s in (6, 8):
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
DEFAULT_INPUT_DIR = r" "
DEFAULT_OUTPUT_DIR = r" "
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

    # Find the first valid MMB control file to pick the FSC channel
    fsc_channel = None
    for fp in files:
        if parse_mmb_size_from_filename(fp) is None:
            continue

        meta, df0 = fcsparser.parse(str(fp), reformat_meta=True)
        fsc_channel = pick_fsc_channel(df0)
        break

    if fsc_channel is None:
        raise SystemExit(
            "No recognised MMB control files were found in the input directory."
        )

    per_file = []
    per_size = {6: [], 8: []}
    qc_panels = []

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
        # EC shoulder fraction strategy:
        #
        # MMB-B controls are retained for audit summaries. MMB-A controls are
        # used to derive the exported separation threshold.
        # -------------------------------------------------------------
        if bead_size == 8:
            drop_frac_used = BOUNDARY_6_8_DROP_FRAC
        else:
            # Use the same setting for consistent audit summaries.
            drop_frac_used = BOUNDARY_6_8_DROP_FRAC

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
        # Save QC plots for the MMB-A reference controls. Each plot shows the
        # FSC-H density profile and the file-specific start_fsc threshold.
        # -------------------------------------------------------------
        if bead_size == 8:
            xs_plot, ys_plot = kde_density(fsc_t, n=2048)

            plot_stem = f"{fp.stem}_size_distribution"
            plot_title = f"{fp.stem}: Size Distribution"

            save_single_density_plot_png(
                out_dir=qc_plot_dir,
                stem=plot_stem,
                xs=xs_plot,
                ys=ys_plot,
                boundary_x=start_fsc,
                x_label=f"{fsc_channel} (logicle)",
                title=plot_title,
            )

            qc_panels.append({
                "name": fp.stem,
                "xs": xs_plot,
                "ys": ys_plot,
                "boundary_x": start_fsc,
            })

    # -----------------------------------------------------------------
    # Aggregate per size using robust medians
    # -----------------------------------------------------------------
    summary = {}

    for s in (6, 8):
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

    boundary_6_8 = summary.get("8", {}).get("start_fsc_median", None)

    collated_plot_path = None
    if qc_panels:
        collated_plot_path = save_density_grid_overview_png(
            out_dir=qc_plot_dir,
            out_name="ALL_MMB8_size_distributions",
            samples=qc_panels,
            x_label=f"{fsc_channel} (logicle)",
            suptitle="MMB-A Size Distributions"
        )

    out = {
        "script": "mmb_size_thresholds_from_controls.py",
        "paths": {
            "input_dir": str(in_dir),
            "output_json": str(out_json),
        },
        "assay_context": {
            "system": "EC two-biomarker assay",
            "mmb_populations_used": ["MMB-B", "MMB-A"],
            "note": (
                "The EC assay uses two MMB populations. Only the separation "
                "threshold required by downstream modules is calculated here."
            )
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
            "boundary_6_8": {
                "source_population": "MMB-A",
                "drop_frac": float(BOUNDARY_6_8_DROP_FRAC),
                "definition": "median 20% left shoulder of MMB-A controls"
            },
            "rationale": (
                "The two MMB FSC-H distributions are close together. "
                "A 20% left shoulder for the MMB-A reference population was selected "
                "to place the threshold closer to the MMB-A peak."
            )
        },
        "qc_plot_outputs": {
            "folder": str(qc_plot_dir),
            "individual_plots": "One PNG per valid MMB-A control file",
            "collated_plot": str(collated_plot_path) if collated_plot_path is not None else None,
        },
        "min_events_per_file": int(args.min_events),
        "per_size_summary": summary,
        "recommended_boundaries": {
            "boundary_6_8": boundary_6_8,
            "interpretation": (
                "The exported threshold is defined as the median 20% left shoulder "
                "of the MMB-A reference population."
            ),
            "assignment_rule": {
                "MMB-B": "FSC < boundary_6_8  (within bead-like region)",
                "MMB-A": "FSC >= boundary_6_8",
            },
        },
        "per_file_estimates": per_file,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print("\nRecommended boundary:")
    print(f"  boundary_6_8 = {boundary_6_8}")
    print("\nShoulder strategy:")
    print(f"  threshold uses {BOUNDARY_6_8_DROP_FRAC * 100:.1f}% left shoulder of MMB-A")

    if qc_panels:
        print("\nQC plots:")
        print(f"  Individual + collated PNGs saved to: {qc_plot_dir}")


if __name__ == "__main__":
    main()