# =============================================================================
# fig_ratio_map.py
# Diverging choropleth of standardised atmospheric vs VWT flow dominance.
# Both networks are z-scored on log scale before differencing, so the map
# shows relative dominance rather than absolute magnitude.
# Run standalone:  python fig_ratio_map.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plot_settings as ps
import plot_functions as pf
from analysis_functions import load_network
import settings as cfg

from cmap import Colormap

# =============================================================================
# Settings
# =============================================================================
cmap = Colormap('colorbrewer:BrBG').to_mpl()
VMIN, VMAX  = -2, 2    # symmetric z-score difference range

threshold = cfg.THRESHOLD_PERCENTILE

# Thresholding function to set values below the specified percentile to zero
def threshold_func(df, threshold=threshold):

    threshold = float(np.nanpercentile(df, threshold))
    filtered_df = df.where(df >= threshold, other=0.0)

    return filtered_df
# =============================================================================
# Plot
# =============================================================================

def _plot_ratio_map(gdf, cntr_gdf):
    import cartopy.crs as ccrs
    
    cmap.set_bad("#6b6b6b")
    norm = mcolors.TwoSlopeNorm(vmin=VMIN, vcenter=0, vmax=VMAX)

    fig, ax = pf.make_map_axes()
    pf.draw_base_map(ax, cntr_gdf)

    plot_gdf = gdf.dropna(subset=["geometry", "ratio"])
    plot_gdf.plot(
        column="ratio",
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.15,
        transform=ccrs.PlateCarree(),
        zorder=2,
        missing_kwds={"color": "#000000"},
    )

    cb = pf.add_map_colorbar(
        fig, ax, cmap, norm,
        label="Relative AMF vs VWT dominance (standardised log difference)",
        ticks=[-2, -1, 0, 1, 2],
        extend="both",
    )
    cb.ax.set_xticklabels(
        ["VWT\ndominant", "", "Equal", "", "AMF\ndominant"],
        fontsize=5
    )

    pf.apply_map_layout(fig)
    return fig


# =============================================================================
# Public entry point
# =============================================================================

def make_figure(save=ps.ADD_FIGS_DIR):
    atmos_df = load_network(ps.ATMOS_PATH)
    vwt_df   = load_network(ps.VWT_PATH)

    atmos_df = threshold_func(atmos_df, threshold=threshold)
    vwt_df   = threshold_func(vwt_df, threshold=threshold)
    
    atmos_total = atmos_df.sum(axis=1) + atmos_df.sum(axis=0)
    vwt_total   = vwt_df.sum(axis=1)   + vwt_df.sum(axis=0)

    flow_df = pd.DataFrame({
        "atmos_total": atmos_total,
        "vwt_total":   vwt_total,
    })
    
    # Only use countries with positive values in both networks
    valid = (flow_df["atmos_total"] > 0) & (flow_df["vwt_total"] > 0)

    log_atmos = np.log10(flow_df.loc[valid, "atmos_total"])
    log_vwt   = np.log10(flow_df.loc[valid, "vwt_total"])

    # Z-score each independently on log scale
    log_atmos_z = (log_atmos - log_atmos.mean()) / log_atmos.std()
    log_vwt_z   = (log_vwt   - log_vwt.mean())   / log_vwt.std()

    # Positive = atmospherically dominant relative to VWT
    flow_df["ratio"] = np.nan
    flow_df.loc[valid, "ratio"] = log_atmos_z - log_vwt_z

    # print(flow_df.loc["GRL", ["atmos_total", "vwt_total", "ratio"]])
    
    cntr_gdf = gpd.read_file(ps.WORLD_PATH,
                             columns=["NAME_ENGL", "ISO3_CODE", "geometry"])
    cntr_gdf = cntr_gdf.rename(columns={"NAME_ENGL": "country",
                                        "ISO3_CODE": "node"})
    cntr_gdf = cntr_gdf.set_index("node")

    gdf = cntr_gdf.merge(flow_df, left_index=True, right_index=True, how="left")
    gdf = gdf.set_geometry("geometry")

    fig = _plot_ratio_map(gdf, cntr_gdf)

    if save:
        pf.save_fig(fig, "fig_ratio_map", folder=save)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
