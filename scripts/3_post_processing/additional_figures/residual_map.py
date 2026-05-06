# =============================================================================
# fig_residual_map.py
# Diverging choropleth of residuals from log(VWT) ~ log(atmos) regression.
# Positive residual = more VWT than expected given atmospheric flow.
# Negative residual = less VWT than expected given atmospheric flow.
# Run standalone:  python fig_residual_map.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from scipy import stats

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
VMIN, VMAX  = -2, 2    # residual range in log10 units

threshold = cfg.THRESHOLD_PERCENTILE



# =============================================================================
# Helpers
# =============================================================================
# Thresholding function to set values below the specified percentile to zero
def threshold_func(df, threshold=threshold):

    threshold = float(np.nanpercentile(df, threshold))
    filtered_df = df.where(df >= threshold, other=0.0)

    return filtered_df

def _compute_residuals(flow_df):
    """Fit log(VWT) ~ log(atmos) OLS and return residuals for all countries."""
    df = flow_df[(flow_df["atmos_total"] > 0) & (flow_df["vwt_total"] > 0)].copy()
    df["log_atmos"] = np.log10(df["atmos_total"])
    df["log_vwt"]   = np.log10(df["vwt_total"])

    slope, intercept, r, p, se = stats.linregress(df["log_atmos"], df["log_vwt"])
    df["residual"] = df["log_vwt"] - (slope * df["log_atmos"] + intercept)

    print(f"Log-log regression: slope={slope:.2f}, intercept={intercept:.2f}, "
          f"r²={r**2:.2f}, p={p:.2e}")
    print(f"Residual range: {df['residual'].min():.2f} to {df['residual'].max():.2f}")

    return df["residual"]


# =============================================================================
# Plot
# =============================================================================

def _plot_residual_map(gdf, cntr_gdf):
    import cartopy.crs as ccrs

    cmap.set_bad("#6b6b6b")
    norm = mcolors.TwoSlopeNorm(vmin=VMIN, vcenter=0, vmax=VMAX)

    fig, ax = pf.make_map_axes()
    pf.draw_base_map(ax, cntr_gdf)

    plot_gdf = gdf.dropna(subset=["geometry", "residual"])
    plot_gdf.plot(
        column="residual",
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.15,
        transform=ccrs.PlateCarree(),
        zorder=2,
        missing_kwds={"color": "#cccccc"},
    )

    cb = pf.add_map_colorbar(
        fig, ax, cmap, norm,
        label="VWT residual from log(VWT) ~ log(atmospheric flow) regression (log₁₀ units)",
        ticks=[VMIN, -1, 0, 1, VMAX],
        extend="both",
    )
    cb.ax.set_xticklabels(
        ["Less VWT\nthan expected", "", "Expected", "", "More VWT\nthan expected"],
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

    residuals = _compute_residuals(flow_df)
    flow_df["residual"] = np.nan
    flow_df.loc[residuals.index, "residual"] = residuals

    cntr_gdf = gpd.read_file(ps.WORLD_PATH,
                             columns=["NAME_ENGL", "ISO3_CODE", "geometry"])
    cntr_gdf = cntr_gdf.rename(columns={"NAME_ENGL": "country",
                                        "ISO3_CODE": "node"})
    cntr_gdf = cntr_gdf.set_index("node")

    gdf = cntr_gdf.merge(flow_df, left_index=True, right_index=True, how="left")
    gdf = gdf.set_geometry("geometry")

    fig = _plot_residual_map(gdf, cntr_gdf)

    if save:
        pf.save_fig(fig, "fig_residual_map", folder=save)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()