# =============================================================================
# fig4_exposure_n_mediators.py
# Dependency map and number-of-mediators map.
# Run standalone:  python fig4_exposure_n_mediators.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_settings as ps
import plot_functions as pf


# =============================================================================
# Sub-figure functions
# =============================================================================

def _plot_dependency_map(dependency_gdf, cntr_gdf):
    cmap, norm = pf.discrete_cmap_norm(ps.DEP_HEX, ps.DEP_BOUNDS)
    fig, ax    = pf.make_map_axes()

    pf.draw_base_map(ax, cntr_gdf)

    import cartopy.crs as ccrs
    dependency_gdf.dropna(subset=["dependency"]).plot(
        column="dependency", ax=ax, cmap=cmap, norm=norm,
        edgecolor="white", linewidth=0.1,
        transform=ccrs.PlateCarree(), zorder=2,
    )
    pf.add_map_colorbar(fig, ax, cmap, norm,
                        label="Exposure",
                        ticks=ps.DEP_BOUNDS)
    pf.apply_map_layout(fig)
    return fig


def _plot_mediator_count_map(med_count_gdf, cntr_gdf):
    cmap, norm = pf.discrete_cmap_norm(ps.MED_HEX, ps.MED_BOUNDS)
    fig, ax    = pf.make_map_axes()

    pf.draw_base_map(ax, cntr_gdf)

    import cartopy.crs as ccrs
    med_count_gdf.dropna(subset=["Num_Mediators_to_Destination"]).plot(
        column="Num_Mediators_to_Destination", ax=ax, cmap=cmap, norm=norm,
        edgecolor="white", linewidth=0.1,
        transform=ccrs.PlateCarree(), zorder=2,
    )
    pf.add_map_colorbar(fig, ax, cmap, norm,
                        label="Number of virtual water trade intermediaries",
                        ticks=ps.MED_BOUNDS)
    pf.apply_map_layout(fig)
    return fig


# =============================================================================
# Main function to create both panels, save, and optionally export data
# =============================================================================

def make_figure(save=ps.FIGS_DIR):
    """Load data, draw both fig4 panels, optionally save.

    Parameters
    ----------
    save : Path or False

    Returns
    -------
    dict  {'dependency': fig, 'mediator_count': fig}
    """
    cntr_gdf = pf.load_world_geometries()

    results_sheets = pd.read_excel(ps.CROSS_NETWORK_XLSX, sheet_name=None, engine="openpyxl")
    ko_sheets      = pd.read_excel(ps.KNOCKOUT_XLSX,      sheet_name=None, engine="openpyxl")

    dependency_df = results_sheets["country_dependency"].set_index("country")
    med_count_df  = ko_sheets["mediators_per_destination"].set_index("Destination")

    dependency_gdf = pf.merge_with_geometries(cntr_gdf, dependency_df)
    med_count_gdf  = pf.merge_with_geometries(cntr_gdf, med_count_df)

    fig_dep = _plot_dependency_map(dependency_gdf, cntr_gdf)
    fig_med = _plot_mediator_count_map(med_count_gdf, cntr_gdf)

    if save:
        pf.save_fig(fig_dep, "fig4_exposure_map",              folder=save)
        pf.save_fig(fig_med, "fig4_num_mediators_to_destination", folder=save)

        # Also export the underlying data as CSV alongside the figures
        save.mkdir(parents=True, exist_ok=True)
        (dependency_df[["dependency"]].dropna()
         .sort_values("dependency", ascending=False)
         .to_csv(save / "fig4_exposure_map.csv"))
        (med_count_df[["Num_Mediators_to_Destination"]].dropna()
         .sort_values("Num_Mediators_to_Destination", ascending=False)
         .to_csv(save / "fig4_num_mediators_to_destination.csv"))

    return {"dependency": fig_dep, "mediator_count": fig_med}


if __name__ == "__main__":
    figs = make_figure()
    plt.show()
