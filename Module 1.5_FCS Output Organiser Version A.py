# Make_FCS_Overview_Plots.py

from pathlib import Path
import os
import math
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define function to create overview PNG from a list of PNG files
def make_overview_png_from_files(png_files, out_path: Path, title: str = ""):
    """
    Arrange given PNG files into a grid of subplots and save as a single image.
    """ 
    png_files = sorted(png_files)
    if not png_files:
        print(f"⚠️ No PNG files provided for {out_path.name}, skipping.")
        return

    print(f"Creating overview with {len(png_files)} PNGs -> {out_path}")

    # Layout: up to 3 images per row
    n = len(png_files)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    # Size: roughly 4 inches per column, 3 inches per row
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)

    for idx, png in enumerate(png_files):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        img = mpimg.imread(png)
        ax.imshow(img)
        ax.set_axis_off()
        # Use filename (without extension) as panel title
        ax.set_title(png.stem, fontsize=8)

    # Hide any unused axes
    for idx in range(len(png_files), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # leave a bit of room for suptitle
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"✔ Saved overview: {out_path}")

# Fucntion to divide and organuise PNGs from a folder
def make_overview_for_folder(folder: Path, out_path: Path, title: str = ""):
    """
    Convenience wrapper: make overview from *all* PNGs in a folder.
    """
    png_files = list(folder.glob("*.png"))
    if not png_files:
        print(f"⚠️ No PNG files found in {folder}, skipping.")
        return
    make_overview_png_from_files(png_files, out_path, title)


def main():
    # Base folder where the refined plots + subfolders live
    base = Path(
        r"D:\ICL Module Notes and more\Year 4\FYP\Software Automation\Visual Flow Data\MMB & DG Controls Refined"
    )

    # Subfolders with 2D and 1D plots
    mmb_2d = base / "MMB 2D Plots"
    dg_2d = base / "DG 2D Plots"
    mmb_1d = base / "MMB 1D Plots"
    dg_1d = base / "DG 1D Plots"

    print(f"Base folder: {base}")
    print(f"MMB 2D folder: {mmb_2d}")
    print(f"DG 2D folder:  {dg_2d}")
    print(f"MMB 1D folder: {mmb_1d}")
    print(f"DG 1D folder:  {dg_1d}")

    # ---------- 2D overviews ----------
    mmb_2d_out = base / "MMB_2D_Overview.png"
    dg_2d_out = base / "DG_2D_Overview.png"

    make_overview_for_folder(mmb_2d, mmb_2d_out, title="MMB 2D Plots")
    make_overview_for_folder(dg_2d, dg_2d_out, title="DG 2D Plots")

    # ---------- 1D overviews: DG ----------
    # DG 1D Fluoro Distribution
    dg_fluoro_files = [
        p for p in dg_1d.glob("*.png")
        if p.stem.endswith("Fluoro Distribution")
    ]
    dg_fluoro_out = base / "DG_1D_Fluoro_Overview.png"
    make_overview_png_from_files(dg_fluoro_files, dg_fluoro_out,
                                 title="DG 1D Fluoro Distributions")

    # DG 1D Size Distribution
    dg_size_files = [
        p for p in dg_1d.glob("*.png")
        if p.stem.endswith("Size Distribution")
    ]
    dg_size_out = base / "DG_1D_Size_Overview.png"
    make_overview_png_from_files(dg_size_files, dg_size_out,
                                 title="DG 1D Size Distributions")

    # ---------- 1D overviews: MMB ----------
    # MMB 1D Fluoro Distribution
    mmb_fluoro_files = [
        p for p in mmb_1d.glob("*.png")
        if p.stem.endswith("Fluoro Distribution")
    ]
    mmb_fluoro_out = base / "MMB_1D_Fluoro_Overview.png"
    make_overview_png_from_files(mmb_fluoro_files, mmb_fluoro_out,
                                 title="MMB 1D Fluoro Distributions")

    # MMB 1D Size Distribution
    mmb_size_files = [
        p for p in mmb_1d.glob("*.png")
        if p.stem.endswith("Size Distribution")
    ]
    mmb_size_out = base / "MMB_1D_Size_Overview.png"
    make_overview_png_from_files(mmb_size_files, mmb_size_out,
                                 title="MMB 1D Size Distributions")


if __name__ == "__main__":
    main()

    