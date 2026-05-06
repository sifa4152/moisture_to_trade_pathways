# =============================================================================
# fig_bivariate_maps.py
# Two bivariate choropleths side by side:
#   Left:  Atmospheric moisture — inflow vs outflow per country
#   Right: Virtual water trade  — imports vs exports per country
# Axes are quantile-binned on log scale to handle heavy-tailed distributions.
# Run standalone:  python fig_bivariate_maps.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd

import plot_settings as ps
import plot_functions as pf
from analysis_functions import load_network
import settings as cfg

# =============================================================================
# Settings
# =============================================================================

N_BINS = 4  # tertiles — 4x4 = 16 colour grid, readable without overcrowding

# Teal × burgundy — for atmospheric moisture map
BIVARIATE_COLORS_ATMOS = np.array([
    ["#e8e8e8", "#b8d8d8", "#7db8b8", "#2a9090"],   # export low
    ["#e8c4c4", "#c4b8b8", "#7aaab0", "#2a8898"],   # export mid-low
    ["#c87878", "#a87878", "#7a8898", "#2a6880"],   # export mid-high
    ["#8b1a1a", "#7a3a50", "#3a5870", "#1a4860"],   # export high
]).T  # shape (4 import, 4 export)

# Gold × indigo — for VWT map
BIVARIATE_COLORS_VWT = np.array([
    ["#f0f0f0", "#e8d89a", "#d4b840", "#b89000"],   # export low
    ["#c8c8e8", "#c8c098", "#b8a048", "#907800"],   # export mid-low
    ["#8888c8", "#9090a0", "#908060", "#706030"],   # export mid-high
    ["#2a2a80", "#484868", "#585040", "#403820"],   # export high
]).T  # shape (4 import, 4 export)


threshold = cfg.THRESHOLD_PERCENTILE


# =============================================================================
# Helpers
# =============================================================================
# Thresholding function to set values below the specified percentile to zero
def threshold_func(df, threshold=threshold):
    threshold = float(np.nanpercentile(df, threshold))
    filtered_df = df.where(df >= threshold, other=0.0)
    return filtered_df

def _quantile_bins(series, n=N_BINS):
    """Quantile bin on log scale, zero/NaN → NaN."""
    nonzero_idx = series[series > 0].index
    result = pd.Series(np.nan, index=series.index)
    result.loc[nonzero_idx] = pd.qcut(
        np.log10(series.loc[nonzero_idx]).rank(method="first"),
        q=n, labels=False
    ).values
    return result

def _bivariate_color(import_bin, export_bin, palette):
    if pd.isna(import_bin) or pd.isna(export_bin):
        return np.nan
    return palette[int(import_bin), int(export_bin)]



def _make_legend(ax, n=N_BINS, cmap_array=BIVARIATE_COLORS_ATMOS, label_import="Import", label_export="Export"):
    """Draw the bivariate legend square."""
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(n):        # import axis (x)
        for j in range(n):    # export axis (y)
            ax.add_patch(mpatches.Rectangle(
                (i, j), 1, 1,
                facecolor=cmap_array[i, j],
                edgecolor="white", linewidth=0.5
            ))

    ax.annotate("", xy=(n + 0.15, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.annotate("", xy=(0, n + 0.15), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.text(n / 2, -0.55, label_import, ha="center", va="top",
            fontsize=6.5, color="black")
    ax.text(-0.45, n / 2, label_export, ha="right", va="center",
            fontsize=6.5, color="black", rotation=90)


def _compute_flow_df(network_df, palette):
    """Compute total inflow (imports) and outflow (exports) per country."""
    imports = network_df.sum(axis=0)   # column sums = received
    exports = network_df.sum(axis=1)   # row sums    = sent
    df = pd.DataFrame({"imports": imports, "exports": exports})
    
    print("imports non-zero:", (df["imports"] > 0).sum())
    print("exports non-zero:", (df["exports"] > 0).sum())
    print("imports range:", df["imports"][df["imports"] > 0].min(), 
          "→", df["imports"].max())
    print("exports range:", df["exports"][df["exports"] > 0].min(), 
          "→", df["exports"].max())
    print("import bins:", df["import_bin"].value_counts(dropna=False) if "import_bin" in df else "not yet computed")
    
    df["import_bin"] = _quantile_bins(df["imports"])
    df["export_bin"] = _quantile_bins(df["exports"])
    
    print("import bin counts:", df["import_bin"].value_counts(dropna=False))
    print("export bin counts:", df["export_bin"].value_counts(dropna=False))
    

    df["color"] = df.apply(
        lambda r: _bivariate_color(r["import_bin"], r["export_bin"], palette), axis=1
    )
    return df


# =============================================================================
# Plot
# =============================================================================

def _plot_panel(ax, fig, gdf, cntr_gdf, title, legend_pos,
                label_import, label_export, palette):
    """Draw one bivariate map panel onto an existing cartopy axes."""
    import cartopy.crs as ccrs

    pf.draw_base_map(ax, cntr_gdf)

    plot_gdf = gdf.dropna(subset=["geometry", "color"])
    plot_gdf.plot(
        color=plot_gdf["color"].tolist(),
        ax=ax,
        edgecolor="white",
        linewidth=0.15,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    ax.set_title(title, fontsize=9, pad=6)

    ax_leg = fig.add_axes(legend_pos)
    _make_legend(ax_leg, cmap_array=palette, label_import=label_import, label_export=label_export)


def make_figure(save=ps.ADD_FIGS_DIR):
    """Build both panels and optionally save.

    Returns
    -------
    matplotlib.figure.Figure
    # """
    import cartopy.crs as ccrs

    # --- Load ---
    atmos_df = load_network(ps.ATMOS_PATH)
    vwt_df   = load_network(ps.VWT_PATH)
    
    # atmos_df = threshold_func(atmos_df, threshold=threshold)
    # vwt_df   = threshold_func(vwt_df, threshold=threshold)
    
    atmos_flow = _compute_flow_df(atmos_df, BIVARIATE_COLORS_ATMOS)
    vwt_flow   = _compute_flow_df(vwt_df,   BIVARIATE_COLORS_VWT)

    print(atmos_flow["color"].value_counts())
    print(atmos_flow[["import_bin", "export_bin"]].value_counts(dropna=False))
    
    # --- Geometry ---
    cntr_gdf = gpd.read_file(ps.WORLD_PATH,
                             columns=["NAME_ENGL", "ISO3_CODE", "geometry"])
    cntr_gdf = cntr_gdf.rename(columns={"NAME_ENGL": "country",
                                        "ISO3_CODE": "node"})
    cntr_gdf = cntr_gdf.set_index("node")

    atmos_gdf = cntr_gdf.merge(atmos_flow, left_index=True,
                               right_index=True, how="left").set_geometry("geometry")
    vwt_gdf   = cntr_gdf.merge(vwt_flow,   left_index=True,
                               right_index=True, how="left").set_geometry("geometry")

    # --- Figure: two side-by-side Robinson maps ---
    figsize = (ps.FIGSIZE_MAP_SINGLE[0] * 2, ps.FIGSIZE_MAP_SINGLE[1])
    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        subplot_kw={"projection": ccrs.Robinson()},
        facecolor="white"
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.05, wspace=0.04)

    for ax in axes:
        ax.set_extent(ps.MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.spines["geo"].set_visible(False)
        ax.set_frame_on(False)

    _plot_panel(
        axes[0], fig, atmos_gdf, cntr_gdf,
        title="Atmospheric moisture flows",
        legend_pos=[0.05, 0.18, 0.07, 0.07],
        label_import="Inflow", label_export="Outflow",
        palette=BIVARIATE_COLORS_ATMOS
    )
    _plot_panel(
        axes[1], fig, vwt_gdf, cntr_gdf,
        title="Virtual water trade",
        legend_pos=[0.54, 0.18, 0.07, 0.07],
        label_import="Imports", label_export="Exports",
        palette=BIVARIATE_COLORS_VWT
    )

    if save:
        pf.save_fig(fig, "fig_bivariate_maps", folder=save)

    return fig


if __name__ == "__main__":
    fig = make_figure(save=None)
    plt.show()