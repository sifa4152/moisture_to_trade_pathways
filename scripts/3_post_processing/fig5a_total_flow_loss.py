# =============================================================================
# fig5a_total_flow_loss.py
# Total flow loss map (mediator knockout results).
# Run standalone:  python fig5a_total_flow_loss.py
# =============================================================================

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import plot_settings as ps
import plot_functions as pf


# =============================================================================
# Sub-figure function
# =============================================================================

def _plot_flow_loss_map(mediator_gdf, cntr_gdf):
    import cartopy.crs as ccrs

    cmap = mcolors.LinearSegmentedColormap.from_list("flow_loss", ps.FLOW_HEX)
    cmap.set_under("#eeeeee")

    values    = mediator_gdf["Total_Flow_Loss_m3"].dropna()
    norm      = mpl.colors.LogNorm(vmin=ps.FLOW_VMIN, vmax=values.max())

    fig, ax   = pf.make_map_axes()
    pf.draw_base_map(ax, cntr_gdf)

    mediator_gdf.dropna(subset=["Total_Flow_Loss_m3"]).plot(
        column="Total_Flow_Loss_m3", ax=ax, cmap=cmap, norm=norm,
        edgecolor="white", linewidth=0.1,
        transform=ccrs.PlateCarree(), zorder=2,
    )
    pf.add_map_colorbar(fig, ax, cmap, norm,
                        label="Total flow contribution (m³)",
                        extend="min")
    pf.apply_map_layout(fig)
    return fig


# =============================================================================
# Public entry point
# =============================================================================

def make_figure(save=ps.FIGS_DIR):
    """Load data, draw fig5a, optionally save.

    Parameters
    ----------
    save : Path or False

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Note: world file uses "node" as index name in this figure — kept as-is
    import geopandas as gpd
    cntr_gdf = gpd.read_file(ps.WORLD_PATH, columns=["NAME_ENGL", "ISO3_CODE", "geometry"])
    cntr_gdf = cntr_gdf.rename(columns={"NAME_ENGL": "country", "ISO3_CODE": "node"})
    cntr_gdf = cntr_gdf.set_index("node")

    sheets      = pd.read_excel(ps.KNOCKOUT_XLSX, sheet_name=None, engine="openpyxl")
    mediator_df = sheets["per_mediator"]
    mediator_gdf = pf.merge_with_geometries(cntr_gdf, mediator_df.set_index("Mediator"))

    fig = _plot_flow_loss_map(mediator_gdf, cntr_gdf)

    if save:
        pf.save_fig(fig, "fig5a_total_flow_contribution_map", folder=save)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
