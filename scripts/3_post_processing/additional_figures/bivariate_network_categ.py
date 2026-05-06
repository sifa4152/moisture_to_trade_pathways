# =============================================================================
# fig_bivariate_maps.py
# Two bivariate choropleths side by side:
#   Left:  Atmospheric moisture — total flow vs net balance
#   Right: Virtual water trade  — total flow vs net balance
#
# Axes:
#   x: total flow (imports + exports) — how connected is this country
#   y: net balance (imports - exports) / total — net importer (+1) to net exporter (-1)
#
# x is quantile-binned on log scale (handles heavy tail)
# y is binned on linear scale (already normalised -1 to +1)
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


# =============================================================================
# Settings
# =============================================================================
N_BINS = 3
# Teal × burgundy — for atmospheric moisture map
# x-axis (cols): total flow low → high
BIVARIATE_COLORS_ATMOS = np.array([
    ["#f0e0d0", "#d4b898", "#a87848"],   # net exporter (outflow dominant)
    ["#e8e8e8", "#c0c8c8", "#809898"],   # near balanced
    ["#c8d8e8", "#4888a8", "#004878"],   # net importer
]).T  # shape (3 total_bin, 3 balance_bin)

# Gold × indigo — for VWT map
BIVARIATE_COLORS_VWT = np.array([
    ["#f0e8d0", "#d4c070", "#a89020"],   # net exporter
    ["#e8e8f0", "#c0c0d0", "#8888b0"],   # near balanced
    ["#d0d0f0", "#5858b0", "#101070"],   # net importer
]).T  # shape (3 total_bin, 3 balance_bin)


# =============================================================================
# Helpers
# =============================================================================

def _quantile_bins_log(series, n=N_BINS):
    """Quantile bin on log scale for total flow (heavy tailed)."""
    nonzero_idx = series[series > 0].index
    result = pd.Series(np.nan, index=series.index)
    result.loc[nonzero_idx] = pd.qcut(
        np.log10(series.loc[nonzero_idx]).rank(method="first"),
        q=n, labels=False
    ).values
    return result


def _linear_bins(series, n=N_BINS):
    """Bin net balance linearly into n equal-width bins over [-1, 1]."""
    bounds = np.linspace(-1, 1, n + 1)
    result = pd.Series(np.nan, index=series.index)
    valid = series.notna()
    result.loc[valid] = pd.cut(
        series.loc[valid], bins=bounds, labels=False, include_lowest=True
    )
    return result


def _bivariate_color(total_bin, balance_bin, palette):
    if pd.isna(total_bin) or pd.isna(balance_bin):
        return np.nan
    return palette[int(total_bin), int(balance_bin)]


# def _make_legend(ax, n=N_BINS, cmap_array=BIVARIATE_COLORS_ATMOS,
#                  label_x="Total flow", label_y="Net balance"):
#     """Draw the bivariate legend square."""
#     ax.set_xlim(-0.5, n)
#     ax.set_ylim(-0.5, n)
#     ax.set_aspect("equal")
#     ax.axis("off")

#     for i in range(n):        # total flow axis (x)
#         for j in range(n):    # net balance axis (y)
#             ax.add_patch(mpatches.Rectangle(
#                 (i, j), 1, 1,
#                 facecolor=cmap_array[i, j],
#                 edgecolor="white", linewidth=0.5
#             ))

#     ax.annotate("", xy=(n + 0.15, 0), xytext=(0, 0),
#                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
#     ax.annotate("", xy=(0, n + 0.15), xytext=(0, 0),
#                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
#     ax.text(n / 2, -0.7, label_x, ha="center", va="top",
#             fontsize=5, color="black")
#     ax.text(-0.6, n / 2, label_y, ha="right", va="center",
#             fontsize=5, color="black", rotation=90)

#     # Endpoint labels on y-axis
#     # ax.text(-0.15, 0.5, "exp.", ha="right", va="center", fontsize=5, color="#555")
#     # ax.text(-0.15, n - 0.5, "imp.", ha="right", va="center", fontsize=5, color="#555")


def _make_legend(ax, n=N_BINS, cmap_array=BIVARIATE_COLORS_ATMOS,
                 label_x="Total flow", label_y="Net balance",
                 label_export="Net exp", label_import="Net imp"):
    """Draw the bivariate legend square with corner labels."""
    ax.set_xlim(-0.5, n + 1.5)
    ax.set_ylim(-1.5, n + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(n):
        for j in range(n):
            ax.add_patch(mpatches.Rectangle(
                (i, j), 1, 1,
                facecolor=cmap_array[i, j],
                edgecolor="white", linewidth=0.5
            ))

    # Axis arrows
    ax.annotate("", xy=(n - 0.1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=.8))
    ax.annotate("", xy=(0, n - 0.1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=.8))

    # Axis titles
    ax.text(n / 2, -0.9, label_x, ha="center", va="top",
            fontsize=5, color="black", fontstyle="italic")
    ax.text(-0.5, n / 2, label_y, ha="right", va="center",
            fontsize=5, color="black", rotation=90, fontstyle="italic")

    # Corner labels — bottom and top of y-axis
    ax.text(n , -0.15, label_export, ha="center", va="top",
            fontsize=4.5, color="#555")
    ax.text(n / 2, n + 0.15, label_import, ha="center", va="bottom",
            fontsize=4.5, color="#555")

    # Low / high labels on x-axis
    ax.text(0.5, -0.15, "low", ha="center", va="top", fontsize=4.5, color="#555")
    ax.text(n - 0.5, -0.15, "high", ha="center", va="top", fontsize=4.5, color="#555")


def _compute_flow_df(network_df, palette):
    """Compute total flow and net balance per country."""
    imports = network_df.sum(axis=0)
    exports = network_df.sum(axis=1)

    total   = imports + exports
    # Net balance: +1 = pure importer, -1 = pure exporter
    balance = (imports - exports) / total.replace(0, np.nan)

    df = pd.DataFrame({
        "imports": imports,
        "exports": exports,
        "total":   total,
        "balance": balance,
    })

    df["total_bin"]   = _quantile_bins_log(df["total"])
    df["balance_bin"] = _linear_bins(df["balance"])

    df["color"] = df.apply(
        lambda r: _bivariate_color(r["total_bin"], r["balance_bin"], palette), axis=1
    )

    return df


# =============================================================================
# Plot
# =============================================================================

def _plot_panel(ax, fig, gdf, cntr_gdf, title, legend_pos,
                label_x, label_y, palette):
    """Draw one bivariate map panel onto an existing cartopy axes."""
    import cartopy.crs as ccrs

    pf.draw_base_map(ax, cntr_gdf)

    plot_gdf = gdf[gdf["color"].notna() & gdf["geometry"].notna()].copy()
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
    _make_legend(ax_leg, cmap_array=palette, label_x=label_x, label_y=label_y)


# =============================================================================
# Public entry point
# =============================================================================

def make_figure(save=ps.ADD_FIGS_DIR):
    import cartopy.crs as ccrs

    atmos_df = load_network(ps.ATMOS_PATH)
    vwt_df   = load_network(ps.VWT_PATH)

    atmos_flow = _compute_flow_df(atmos_df, BIVARIATE_COLORS_ATMOS)
    vwt_flow   = _compute_flow_df(vwt_df,   BIVARIATE_COLORS_VWT)

    cntr_gdf = gpd.read_file(ps.WORLD_PATH,
                             columns=["NAME_ENGL", "ISO3_CODE", "geometry"])
    cntr_gdf = cntr_gdf.rename(columns={"NAME_ENGL": "country",
                                        "ISO3_CODE": "node"})
    cntr_gdf = cntr_gdf.set_index("node")

    atmos_gdf = cntr_gdf.merge(atmos_flow, left_index=True,
                               right_index=True, how="left").set_geometry("geometry")
    vwt_gdf   = cntr_gdf.merge(vwt_flow,   left_index=True,
                               right_index=True, how="left").set_geometry("geometry")

    figsize = (ps.FIGSIZE_MAP_SINGLE[0] * 2, ps.FIGSIZE_MAP_SINGLE[1])
    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        subplot_kw={"projection": ccrs.Robinson()},
        facecolor="white"
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02, wspace=0.02)

    for ax in axes:
        ax.set_extent(ps.MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.spines["geo"].set_visible(False)
        ax.set_frame_on(False)

    _plot_panel(
        axes[0], fig, atmos_gdf, cntr_gdf,
        title="AMF network",
        legend_pos=[-0.05, 0.15, 0.25, 0.25],
        label_x="Total flow", label_y="Net balance",
        palette=BIVARIATE_COLORS_ATMOS
    )
    _plot_panel(
        axes[1], fig, vwt_gdf, cntr_gdf,
        title="VWT network",
        legend_pos=[0.45, 0.15, 0.25, 0.25], # [left, bottom, width, height]
        label_x="Total flow", label_y="Net balance",
        palette=BIVARIATE_COLORS_VWT
    )

    if save:
        pf.save_fig(fig, "fig_bivariate_maps", folder=save)

    return fig


if __name__ == "__main__":
    fig = make_figure(save=None)
    plt.show()