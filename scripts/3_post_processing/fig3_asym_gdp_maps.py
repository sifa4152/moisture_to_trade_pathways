# =============================================================================
# fig3_asym_gdp_maps.py
# Asymmetry index map, DMR map, and asymmetry vs GDP scatter.
# Run standalone:  python fig3_asym_gdp_maps.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from scipy.stats import pearsonr

import plot_settings as ps
import plot_functions as pf


# =============================================================================
# Figure-specific helpers
# =============================================================================

def _get_label(row, index_val):
    if "name" in row.index and pd.notna(row["name"]):
        return row["name"]
    elif "country" in row.index and pd.notna(row["country"]):
        return row["country"]
    return str(index_val)


# =============================================================================
# Sub-figure functions
# =============================================================================

def _plot_asymmetry_map(data_gdf, cntr_gdf):
    cmap, norm = pf.asym_cmap_norm()
    fig, ax    = pf.make_map_axes(figsize=ps.FIGSIZE_MAP_WIDE)

    pf.draw_base_map(ax, cntr_gdf)
    data_gdf.dropna(subset=["asymmetry_index"]).plot(
        ax=ax, column="asymmetry_index", cmap=cmap, norm=norm,
        edgecolor="#aaaaaa", linewidth=0.1,
        transform=__import__("cartopy.crs", fromlist=["PlateCarree"]).PlateCarree(),
        zorder=2,
    )
    cb = pf.add_map_colorbar(fig, ax, cmap, norm,
                        label="Asymmetry index",
                        ticks=ps.ASYM_BOUNDS) #ticks = [-1, -0.6, -0.2, 0, 0.2, 0.6, 1.0]
    cb.ax.tick_params(labelrotation=25)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15)
    return fig


def _plot_dmr_map(data_gdf, cntr_gdf):
    cmap, norm = pf.discrete_cmap_norm(ps.DMR_HEX, ps.DMR_BOUNDS)
    fig, ax    = pf.make_map_axes(figsize=ps.FIGSIZE_MAP_WIDE)

    pf.draw_base_map(ax, cntr_gdf)
    data_gdf.dropna(subset=["dmr"]).plot(
        ax=ax, column="dmr", cmap=cmap, norm=norm,
        edgecolor="#aaaaaa", linewidth=0.1,
        transform=__import__("cartopy.crs", fromlist=["PlateCarree"]).PlateCarree(),
        zorder=2,
    )
    pf.add_map_colorbar(fig, ax, cmap, norm,
                        label="Domestic moisture recycling ratio",
                        ticks=np.linspace(0, 0.5, 6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15)
    return fig


def _plot_scatter_asymmetry_gdp(data_gdf):
    cmap, norm = pf.asym_cmap_norm()

    xcol   = "gdp_mean_2008_2017_per_cap$"
    ycol   = "asymmetry_index"
    popcol = "pop_2008_2017"

    plot_df = data_gdf.dropna(subset=[ycol, xcol, popcol]).copy()
    pop     = plot_df[popcol]
    sizes   = 30 + 770 * (pop - pop.min()) / (pop.max() - pop.min())

    fig, ax = plt.subplots(figsize=ps.FIGSIZE_SCATTER, facecolor="white")

    # Reference line
    ax.axhline(0, color="#333333", linestyle="-", linewidth=0.3, alpha=0.6)

    ymax = plot_df[ycol].max() * 1.1
    xmin = plot_df[xcol].min() * 0.85
    xmax = plot_df[xcol].max() * 1.15

    # World Bank income bands
    for thresh in ps.WB_THRESHOLDS:
        ax.axvline(thresh, color="#aaaaaa", linestyle="--", linewidth=0.3, alpha=0.6)
    band_edges  = [xmin] + ps.WB_THRESHOLDS + [xmax]
    band_mids   = [np.sqrt(band_edges[i] * band_edges[i+1]) for i in range(4)]
    for mid, lbl in zip(band_mids, ps.WB_LABELS):
        ax.text(mid, ymax * 0.99, lbl, fontsize=6, color="#666666",
                ha="center", va="top", alpha=1)

    ax.scatter(
        plot_df[xcol], plot_df[ycol],
        c=plot_df[ycol], cmap=cmap, norm=norm,
        s=sizes, edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3,
    )

    r, _ = pearsonr(np.log10(plot_df[xcol]), plot_df[ycol])
    print(f"Pearson r = {r:.2f}  (log GDP vs asymmetry index)")

    # Corner labels
    label_idx    = pf.corner_label_indices(plot_df, xcol, ycol, n=10, x_log=True)
    gdp_log_mid  = np.median(np.log10(plot_df[xcol]))
    for idx in label_idx:
        row   = plot_df.loc[idx]
        label = _get_label(row, idx)
        dx    = 8  if np.log10(row[xcol]) > gdp_log_mid else -8
        dy    = 6  if row[ycol] > 0 else -6
        ha    = "left" if np.log10(row[xcol]) > gdp_log_mid else "right"
        ax.annotate(label, xy=(row[xcol], row[ycol]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6, color="#222222", ha=ha)

    # Colorbar
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, orientation="vertical", pad=0.04, shrink=0.4, aspect=30)
    cb.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Asymmetry index", size=8, labelpad=8)

    # Population legend
    legend_pops  = [10e6, 100e6, 500e6, 1.4e9]
    legend_sizes = 30 + 770 * (np.array(legend_pops) - pop.min()) / (pop.max() - pop.min())
    for lp, ls in zip(legend_pops, legend_sizes):
        ax.scatter([], [], s=ls, color="grey", alpha=0.5,
                   edgecolor="black", linewidth=0.3,
                   label=f"{int(lp / 1e6)}M")
    ax.legend(title="Population", title_fontsize=8, fontsize=8,
              loc="lower right", framealpha=0.7, scatterpoints=1)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${int(x/1000)}k" if x >= 1000 else f"${int(x)}")
    )
    ax.set_xlabel("GDP per capita [USD, log scale]", fontsize=13)
    ax.set_ylabel("Asymmetry index", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    return fig


# =============================================================================
# Public entry point
# =============================================================================

def make_figure(save=ps.FIGS_DIR):
    """Load data, draw all three fig3 panels, optionally save.

    Parameters
    ----------
    save : Path or False

    Returns
    -------
    dict  {'asym_map': fig, 'dmr_map': fig, 'scatter': fig}
    """
    import geopandas as gpd

    cntr_gdf = pf.load_world_geometries()
    meta_df  = pd.read_csv(ps.META_PATH)

    cross_sheets   = pd.read_excel(ps.CROSS_NETWORK_XLSX, sheet_name=None, engine="openpyxl")
    asymmetry_df   = cross_sheets["asymmetry_index"].set_index("country")
    dependency_df  = cross_sheets["country_dependency"].set_index("country")

    ko_sheets      = pd.read_excel(ps.KNOCKOUT_XLSX, sheet_name=None, engine="openpyxl")
    med_per_dest   = ko_sheets["mediators_per_destination"].set_index("Destination")

    data_gdf = pf.merge_with_geometries(cntr_gdf, asymmetry_df)
    data_gdf = pf.merge_with_geometries(
        data_gdf,
        meta_df.set_index("iso3")[["name", "gdp_mean_2008_2017_per_cap$", "tmr", "dmr", "pop_2008_2017"]],
    )
    data_gdf = pf.merge_with_geometries(data_gdf, dependency_df)
    data_gdf = pf.merge_with_geometries(data_gdf, med_per_dest)

    fig_asym    = _plot_asymmetry_map(data_gdf, cntr_gdf)
    fig_dmr     = _plot_dmr_map(data_gdf, cntr_gdf)
    fig_scatter = _plot_scatter_asymmetry_gdp(data_gdf)

    if save:
        pf.save_fig(fig_asym,    "fig3_map_asymmetry_index",  folder=save)
        pf.save_fig(fig_dmr,     "fig3_map_dmr",              folder=save)
        pf.save_fig(fig_scatter, "fig3_scatter_asymmetry_gdp", folder=save)

    return {"asym_map": fig_asym, "dmr_map": fig_dmr, "scatter": fig_scatter}


if __name__ == "__main__":
    figs = make_figure()
    plt.show()
