# patient_fcs_viewer_logicle_2d_only_with_grid.py
#
# =============================================================================
# PURPOSE
# =============================================================================
# Module 1: FCS Data Visualiser
#
# This script gives a first-pass view of EC assay .fcs files before gating and
# clustering. It reads every .fcs file in the input folder, applies the same
# signal transformation used by the rest of the pipeline, and saves a 2D scatter
# plot for each file.
#
# Plot layout:
#   x-axis: green fluorescence channel
#   y-axis: FSC channel
#
# Output organisation:
#   - files labelled as MMB controls are saved in "MMB 2D Plots"
#   - files labelled as DG controls are saved in "DG 2D Plots"
#   - patient/sample files are saved directly in the output folder
#   - an all-sample overview grid is saved in the output folder
#
# This module is visualisation-only. It does not apply gates, run KDE analysis,
# or create JSON outputs.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import sys, os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # write PNG files without opening a plotting window
import matplotlib.pyplot as plt
import fcsparser


# =============================================================================
# 1) Transformation
# =============================================================================
try:
    from flowutils.transforms import logicle as logicle_transform
    LOGICLE_SOURCE = "flowutils.transforms.logicle"
except Exception:
    logicle_transform = None
    LOGICLE_SOURCE = "asinh-fallback"


def apply_logicle_or_asinh(arr: np.ndarray) -> np.ndarray:
    """
    Apply a Logicle transform when flowutils is available. If not, use an
    inverse hyperbolic sine transform so the script can still run.
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
# 2) Channel selection
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
    cols = list(df.columns)

    # Use the height channel where available, as this is the main size signal
    # used elsewhere in the pipeline.
    fsc = first_match(cols, ["FSC-H"]) or first_match(cols, ["FSC-A", "FSC"])

    # The assay signal is expected in the green fluorescence channel. Several
    # common cytometer labels are checked to keep the script portable.
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
# 3) File grouping
# =============================================================================
def classify_file_type(file_path: Path) -> str:
    """
    Sort files into simple output groups based on the filename.

    The script only uses broad labels:
      - DG controls
      - MMB controls
      - patient/sample files
    """
    stem_upper = file_path.stem.upper()

    if "DG" in stem_upper:
        return "DG"

    if "MMB" in stem_upper:
        return "MMB"

    return "SAMPLE"


# =============================================================================
# 4) Plotting helpers
# =============================================================================
def safe_filename(name: str) -> str:
    return "".join(c if c not in r'<>:"/\\|?*' else "_" for c in name)


def save_single_plot_png(out_dir: Path, stem: str, xvals, yvals, xlab, ylab, title: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(xvals, yvals, s=2, alpha=0.45)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()

    out_path = out_dir / f"{safe_filename(stem)}.png"
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"Saved {out_path}")


def save_grid_overview_png(
    out_dir: Path,
    out_name: str,
    samples: list[dict],
    xlab: str,
    ylab: str,
    suptitle: str,
    max_points_per_panel: int = 15000,
):
    """
    Save a collated grid of all transformed 2D scatter plots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    if n == 0:
        return

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig_w = max(10, 3.2 * ncols)
    fig_h = max(8, 2.8 * nrows)

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

        x = s["x"]
        y = s["y"]

        # Downsample dense files so the overview remains readable and quick to save.
        if x.size > max_points_per_panel:
            rng = np.random.default_rng(0)
            pick = rng.choice(x.size, size=max_points_per_panel, replace=False)
            x_plot = x[pick]
            y_plot = y[pick]
        else:
            x_plot = x
            y_plot = y

        ax.scatter(x_plot, y_plot, s=1.5, alpha=0.45)
        ax.set_title(s["name"], fontsize=9)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path = out_dir / f"{safe_filename(out_name)}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved {out_path}")


# =============================================================================
# 5) Main
# =============================================================================
def main():
    INPUT_FOLDER = Path(
        r" "
    )

    OUTPUT_FOLDER = Path(
        r" "
    )

    MMB_2D_FOLDER = OUTPUT_FOLDER / "MMB 2D Plots"
    DG_2D_FOLDER = OUTPUT_FOLDER / "DG 2D Plots"

    if not INPUT_FOLDER.exists():
        raise SystemExit(f"Input folder does not exist:\n  {INPUT_FOLDER}")

    fcs_files = sorted(INPUT_FOLDER.glob("*.fcs"))

    if not fcs_files:
        raise SystemExit(f"No .fcs files found in:\n  {INPUT_FOLDER}")

    print(f"Python: {sys.executable}")
    print(f"CWD:    {os.getcwd()}")
    print(f"Input:  {INPUT_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"MMB 2D output if filename contains 'MMB': {MMB_2D_FOLDER}")
    print(f"DG 2D output if filename contains 'DG':   {DG_2D_FOLDER}")
    print(f"Other plots are saved directly in:        {OUTPUT_FOLDER}")
    print(f"Transform source: {LOGICLE_SOURCE}")
    print(f"Found {len(fcs_files)} .fcs files\n")

    fluor_name, fsc_name = None, None
    transform_label = "logicle" if LOGICLE_SOURCE != "asinh-fallback" else "asinh"

    grid_samples: list[dict] = []

    for f in fcs_files:
        print(f"Loading: {f.name}")

        meta, df = fcsparser.parse(str(f), reformat_meta=True)

        if fluor_name is None or fsc_name is None:
            fluor_name, fsc_name = pick_channels(df)
            print(f"Selected channels → Fluor: {fluor_name}  |  FSC: {fsc_name}")

        if fsc_name not in df.columns or fluor_name not in df.columns:
            raise ValueError(
                f"Expected channels not found in {f.name}.\n"
                f"Need: {fsc_name}, {fluor_name}\n"
                f"Have: {list(df.columns)}"
            )

        x_raw = df[fluor_name].to_numpy(dtype=float)
        y_raw = df[fsc_name].to_numpy(dtype=float)

        x_t = apply_logicle_or_asinh(x_raw)
        y_t = apply_logicle_or_asinh(y_raw)

        m = np.isfinite(x_t) & np.isfinite(y_t)
        x_t = x_t[m]
        y_t = y_t[m]

        stem = f.stem
        sample_type = classify_file_type(f)

        if sample_type == "DG":
            single_out_dir = DG_2D_FOLDER
        elif sample_type == "MMB":
            single_out_dir = MMB_2D_FOLDER
        else:
            single_out_dir = OUTPUT_FOLDER

        single_name = f"{stem}_{fsc_name}_vs_{fluor_name}_{transform_label}"

        save_single_plot_png(
            out_dir=single_out_dir,
            stem=single_name,
            xvals=x_t,
            yvals=y_t,
            xlab=f"{fluor_name} ({transform_label})",
            ylab=f"{fsc_name} ({transform_label})",
            title=f"{stem}: {fsc_name} vs {fluor_name} ({transform_label})",
        )

        grid_samples.append({
            "name": stem,
            "x": x_t,
            "y": y_t
        })

    save_grid_overview_png(
        out_dir=OUTPUT_FOLDER,
        out_name=f"ALL_SAMPLES_{fsc_name}_vs_{fluor_name}_{transform_label}_GRID",
        samples=grid_samples,
        xlab=f"{fluor_name} ({transform_label})",
        ylab=f"{fsc_name} ({transform_label})",
        suptitle=f"All samples: {fsc_name} vs {fluor_name} ({transform_label})",
        max_points_per_panel=15000,
    )

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
