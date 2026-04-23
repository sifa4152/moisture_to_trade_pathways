# =============================================================================
# plot_functions.py
# Shared plotting utilities for the figure pipeline.
#
# Only functions used by ≥2 figure scripts live here.
# Figure-specific helpers stay in the figure file itself.
# =============================================================================

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import plot_settings as ps


# =============================================================================
# 1. Data helpers
# =============================================================================

def load_world_geometries(path=None):
    """Read the country shapefile and return a GeoDataFrame indexed by ISO-3.

    Parameters
    ----------
    path : Path, optional  — defaults to plot_settings.WORLD_PATH

    Returns
    -------
    gpd.GeoDataFrame  indexed by iso3
    """
    path = path or ps.WORLD_PATH
    gdf = gpd.read_file(path, columns=["NAME_ENGL", "ISO3_CODE", "geometry"])
    gdf = gdf.rename(columns={"NAME_ENGL": "country", "ISO3_CODE": "iso3"})
    return gdf.set_index("iso3")


def merge_with_geometries(cntr_gdf, metrics_df):
    """Merge a metrics DataFrame onto a world GeoDataFrame by index.

    Works with both DataFrames and Series.  The merge is right-sided on the
    metrics so every country that has metric data is kept, even if it lacks
    geometry (geometry will be NaN — filtered out at plot time).

    Parameters
    ----------
    cntr_gdf   : gpd.GeoDataFrame   — indexed by iso3
    metrics_df : pd.DataFrame or pd.Series — indexed by iso3

    Returns
    -------
    gpd.GeoDataFrame
    """
    if isinstance(metrics_df, pd.Series):
        metrics_df = metrics_df.to_frame()
    return cntr_gdf.merge(
        metrics_df, left_index=True, right_index=True, how="right"
    ).set_geometry("geometry")


# =============================================================================
# 2. Colormap builders
# =============================================================================

def make_discrete_cmap(hex_list, n):
    """Interpolate a list of hex colours into *n* discrete steps.

    Parameters
    ----------
    hex_list : list of str   — e.g. ["#ffffff", "#000000"]
    n : int

    Returns
    -------
    matplotlib.colors.ListedColormap
    """
    gradient = mcolors.LinearSegmentedColormap.from_list("_tmp", hex_list)
    return mcolors.ListedColormap(gradient(np.linspace(0, 1, n)))


def asym_cmap_norm():
    """Return the shared asymmetry-index colormap and BoundaryNorm.

    Returns
    -------
    (ListedColormap, BoundaryNorm)
    """
    cmap  = mcolors.ListedColormap(ps.ASYM_HEX)
    norm  = mcolors.BoundaryNorm(ps.ASYM_BOUNDS, cmap.N)
    return cmap, norm


def discrete_cmap_norm(hex_list, bounds):
    """Generic helper: build a ListedColormap + BoundaryNorm from hex list and bounds.

    The number of colours will be ``len(bounds) - 1``.

    Parameters
    ----------
    hex_list : list of str
    bounds   : list of float

    Returns
    -------
    (ListedColormap, BoundaryNorm)
    """
    n    = len(bounds) - 1
    cmap = make_discrete_cmap(hex_list, n)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# =============================================================================
# 3. Map canvas
# =============================================================================

def make_map_axes(figsize=None, extent=None, facecolor="white"):
    """Create a Robinson-projection figure + axes ready for world-map plotting.

    Parameters
    ----------
    figsize  : tuple, optional  — defaults to plot_settings.FIGSIZE_MAP_SINGLE
    extent   : list, optional   — [xmin, xmax, ymin, ymax] in PlateCarree degrees;
               defaults to plot_settings.MAP_EXTENT
    facecolor: str

    Returns
    -------
    (fig, ax)
    """
    import cartopy.crs as ccrs

    figsize = figsize or ps.FIGSIZE_MAP_SINGLE
    extent  = extent  or ps.MAP_EXTENT

    fig, ax = plt.subplots(
        figsize=figsize, facecolor=facecolor,
        subplot_kw={"projection": ccrs.Robinson()}
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)
    ax.set_frame_on(False)
    return fig, ax


def draw_base_map(ax, cntr_gdf):
    """Draw the grey country outlines (background layer) on *ax*.

    Parameters
    ----------
    ax       : cartopy GeoAxes
    cntr_gdf : gpd.GeoDataFrame — full world geometry
    """
    import cartopy.crs as ccrs

    cntr_gdf.plot(
        ax=ax,
        color=ps.MAP_BASE_COLOR,
        edgecolor=ps.MAP_EDGE_COLOR,
        linewidth=ps.MAP_EDGE_WIDTH,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )


def add_map_colorbar(fig, ax, cmap, norm, label, ticks=None, extend="neither"):
    """Attach a horizontal colorbar below a map axes.

    Parameters
    ----------
    fig, ax  : figure and axes
    cmap     : colormap
    norm     : norm
    label    : str  — colorbar label
    ticks    : list, optional — explicit tick positions
    extend   : str  — 'neither' | 'min' | 'max' | 'both'

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, extend=extend,
        **ps.MAP_CBAR_KWARGS,
    )
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.set_label(label, size=6, weight="bold")
    cb.ax.tick_params(labelsize=6)
    return cb


def apply_map_layout(fig):
    """Apply the standard tight subplot spacing for map figures."""
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.10)


# =============================================================================
# 4. Common annotation helpers
# =============================================================================

def panel_label(ax, letter, x=-0.02, y=1.02, fontsize=10, fontweight="bold", **kwargs):
    """Add a panel label (A, B, C …) in the upper-left of *ax*.

    Parameters
    ----------
    ax     : matplotlib Axes
    letter : str  — e.g. "A"
    x, y   : float — axes-fraction coordinates
    """
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va="bottom", ha="right", **kwargs)


def corner_label_indices(df, xcol, ycol, n=3, x_log=False):
    """Return index labels of the *n* points nearest each of the 4 axis corners.

    Used for auto-labelling scatter plots without overlap.

    Parameters
    ----------
    df    : pd.DataFrame
    xcol  : str   — x-axis column name
    ycol  : str   — y-axis column name
    n     : int   — number of points per corner
    x_log : bool  — normalise x on log10 scale

    Returns
    -------
    list of index values
    """
    x = np.log10(df[xcol]) if x_log else df[xcol]
    y = df[ycol]
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    selected = set()
    for cx, cy in [(1, 1), (0, 1), (1, 0), (0, 0)]:
        dist = np.sqrt((xn - cx) ** 2 + (yn - cy) ** 2)
        selected.update(dist.nsmallest(n).index)
    return list(selected)


# =============================================================================
# 5. Save helper
# =============================================================================

def save_fig(fig, name, folder=None, dpi=None, formats=("svg", "png")):
    """Save *fig* to SVG and/or PNG inside *folder*.

    Parameters
    ----------
    fig     : matplotlib Figure
    name    : str  — filename stem (no extension)
    folder  : Path, optional  — defaults to plot_settings.FIGS_DIR
    dpi     : int, optional   — defaults to plot_settings.SAVE_DPI
    formats : tuple of str    — file formats to write
    """
    folder = folder or ps.FIGS_DIR
    dpi    = dpi    or ps.SAVE_DPI
    folder.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        out = folder / f"{name}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
