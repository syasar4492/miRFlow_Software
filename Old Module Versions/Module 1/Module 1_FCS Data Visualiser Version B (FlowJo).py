#!/usr/bin/env python3
# patient_fcs_viewer_flowjo_like_display.py
#
# =============================================================================
# PURPOSE
# =============================================================================
# Viewer for patient/control .fcs files with FlowJo-like visual display:
#   - Read every *.fcs in a folder
#   - Auto-detect FSC (size) and green fluorescence (B525-H / FL1-H)
#   - Apply Logicle transform (flowutils) with asinh fallback (still available)
#   - Save ONE plot per file:
#       x-axis: B525-H (fluorescence)
#       y-axis: FSC-H (size)
#   - Display mode aims to look closer to FlowJo by plotting RAW values on
#     log-style axes whenever possible
#
# PLUS:
#   - Save ONE additional "all samples" grid PNG containing all plots together
#
# NO gating, NO KDE, NO JSON output.
# =============================================================================

from __future__ import annotations

from pathlib import Path
import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for PNG output
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import fcsparser


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
# FLOWJO_LIKE:
#   - Plot RAW channel values
#   - Use logarithmic axes where possible
#   - This gives axis labels much closer to FlowJo
#
# TRANSFORMED:
#   - Plot transformed values directly (your old behaviour)
#
DISPLAY_MODE = "FLOWJO_LIKE"   # options: "FLOWJO_LIKE", "TRANSFORMED"

# Downsample for plotting speed / readability
MAX_POINTS_SINGLE = 40000
MAX_POINTS_GRID = 15000


# =============================================================================
# 1) TRANSFORMATION: Logicle (preferred) with asinh fallback
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

    Mirrors your Advanced FCS Viewer parameter rules:
      T = max(262144, vmax)
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
# 2) Channel picking
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

    # Prefer FSC-H; else FSC-A; else any FSC
    fsc = first_match(cols, ["FSC-H"]) or first_match(cols, ["FSC-A", "FSC"])

    # Prefer FL1-H; then B525-H and other green aliases
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
# 3) Plot helpers
# =============================================================================
def safe_filename(name: str) -> str:
    return "".join(c if c not in r'<>:"/\|?*' else "_" for c in name)


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int, seed: int = 0):
    n = x.size
    if n <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def style_log_axes(ax):
    """
    Make axes look more FlowJo-like using log ticks labelled as powers of 10.
    """
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.tick_params(axis="both", which="major", labelsize=10, width=1.2, length=7)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3.5)


def prepare_plot_arrays(x_raw, y_raw, x_t, y_t, display_mode: str):
    """
    Returns:
      x_plot, y_plot, x_label_suffix, y_label_suffix, using_log_axes
    """
    if display_mode == "FLOWJO_LIKE":
        # Need strictly positive values for pure log axes
        m = np.isfinite(x_raw) & np.isfinite(y_raw) & (x_raw > 0) & (y_raw > 0)
        x_plot = x_raw[m]
        y_plot = y_raw[m]

        # If too many values are non-positive, fall back to transformed plotting
        if x_plot.size < 50 or y_plot.size < 50:
            m = np.isfinite(x_t) & np.isfinite(y_t)
            return x_t[m], y_t[m], "(logicle)", "(logicle)", False

        return x_plot, y_plot, "", "", True

    # Old behaviour
    m = np.isfinite(x_t) & np.isfinite(y_t)
    return x_t[m], y_t[m], "(logicle)", "(logicle)", False


def save_single_plot_png(
    out_dir: Path,
    stem: str,
    x_raw,
    y_raw,
    x_t,
    y_t,
    x_channel,
    y_channel,
    title: str,
    display_mode: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    x_plot, y_plot, x_suffix, y_suffix, use_log_axes = prepare_plot_arrays(
        x_raw, y_raw, x_t, y_t, display_mode
    )
    x_plot, y_plot = downsample_xy(x_plot, y_plot, MAX_POINTS_SINGLE, seed=0)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(x_plot, y_plot, s=4, alpha=0.45)

    if use_log_axes:
        style_log_axes(ax)
        ax.set_xlabel(f"{x_channel}", fontsize=11)
        ax.set_ylabel(f"{y_channel}", fontsize=11)
    else:
        ax.set_xlabel(f"{x_channel} {x_suffix}".strip(), fontsize=11)
        ax.set_ylabel(f"{y_channel} {y_suffix}".strip(), fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

    ax.set_title(title, fontsize=14)
    fig.tight_layout()

    out_path = out_dir / f"{safe_filename(stem)}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_grid_overview_png(
    out_dir: Path,
    out_name: str,
    samples: list[dict],
    x_channel: str,
    y_channel: str,
    suptitle: str,
    display_mode: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    if n == 0:
        return

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig_w = max(10, 3.4 * ncols)
    fig_h = max(8, 3.0 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, s in enumerate(samples):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        x_plot, y_plot, x_suffix, y_suffix, use_log_axes = prepare_plot_arrays(
            s["x_raw"], s["y_raw"], s["x_t"], s["y_t"], display_mode
        )
        x_plot, y_plot = downsample_xy(x_plot, y_plot, MAX_POINTS_GRID, seed=0)

        ax.scatter(x_plot, y_plot, s=2, alpha=0.45)
        ax.set_title(s["name"], fontsize=9)

        if use_log_axes:
            style_log_axes(ax)
            ax.set_xlabel(x_channel, fontsize=8)
            ax.set_ylabel(y_channel, fontsize=8)
        else:
            ax.set_xlabel(f"{x_channel} {x_suffix}".strip(), fontsize=8)
            ax.set_ylabel(f"{y_channel} {y_suffix}".strip(), fontsize=8)
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
# 4) Main
# =============================================================================
def main():
    # -----------------------------------------------------------------
    # Patient input folder
    # -----------------------------------------------------------------
    INPUT_FOLDER = Path(
        r"D:\ICL MBE\Year 4\FYP\Software Automation\Raw Flow Data\06.03.2026 controls in plasma samples"
    )

    # -----------------------------------------------------------------
    # Output folder for PNGs
    # -----------------------------------------------------------------
    OUTPUT_FOLDER = Path(
        r"D:\ICL MBE\Year 4\FYP\Software Automation\Visual Flow Data\Controls in Plasma Samples (FlowJo axes)"
    )

    if not INPUT_FOLDER.exists():
        raise SystemExit(f"Input folder does not exist:\n  {INPUT_FOLDER}")

    fcs_files = sorted(INPUT_FOLDER.glob("*.fcs"))
    if not fcs_files:
        raise SystemExit(f"No .fcs files found in:\n  {INPUT_FOLDER}")

    print(f"Python: {sys.executable}")
    print(f"CWD:    {os.getcwd()}")
    print(f"Input:  {INPUT_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"Transform source: {LOGICLE_SOURCE}")
    print(f"Display mode: {DISPLAY_MODE}")
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

        # RAW values
        x_raw = df[fluor_name].to_numpy(dtype=float)
        y_raw = df[fsc_name].to_numpy(dtype=float)

        # Transformed values (kept available for fallback / consistency)
        x_t = apply_logicle_or_asinh(x_raw)
        y_t = apply_logicle_or_asinh(y_raw)

        stem = f.stem
        single_name = f"{stem}_{fsc_name}_vs_{fluor_name}_{transform_label}_{DISPLAY_MODE}"

        save_single_plot_png(
            out_dir=OUTPUT_FOLDER,
            stem=single_name,
            x_raw=x_raw,
            y_raw=y_raw,
            x_t=x_t,
            y_t=y_t,
            x_channel=fluor_name,
            y_channel=fsc_name,
            title=f"{stem}: {fsc_name} vs {fluor_name}",
            display_mode=DISPLAY_MODE,
        )

        grid_samples.append({
            "name": stem,
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_t": x_t,
            "y_t": y_t,
        })

    save_grid_overview_png(
        out_dir=OUTPUT_FOLDER,
        out_name=f"ALL_SAMPLES_{fsc_name}_vs_{fluor_name}_{DISPLAY_MODE}_GRID",
        samples=grid_samples,
        x_channel=fluor_name,
        y_channel=fsc_name,
        suptitle=f"All samples: {fsc_name} vs {fluor_name}",
        display_mode=DISPLAY_MODE,
    )

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)