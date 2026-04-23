# =============================================================================
# fig5b_sankey.py
# Normalised Sankey diagrams for selected mediator countries.
# Run standalone:  python fig5b_sankey.py
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import plot_settings as ps
import plot_functions as pf

from analysis_functions import (
    load_meta_data,load_network, 
    build_thresholded_graph,
    get_feedforward_triples,
)


# =============================================================================
# Figure-specific helpers
# =============================================================================

def _calculate_spaced_y(values, gap_fraction=0.25):
    """Y-positions with white-space gaps distributed evenly between nodes."""
    total_val = sum(values)
    n         = len(values)
    if n == 1:
        return [0.5]
    gap              = gap_fraction / (n - 1)
    available_height = 1.0 - gap_fraction
    y_coords         = []
    cumulative_y     = 0
    for v in values:
        v_height = (v / total_val) * available_height
        y_coords.append(cumulative_y + (v_height / 2))
        cumulative_y += v_height + gap
    # Snap last node to avoid float drift
    y_coords[-1] = 1.0 - ((values[-1] / total_val) * available_height / 2)
    return y_coords


def _get_top_n_plus_others(series, n, iso3_to_name, suffix=""):
    """Return (names_list, values_list) for top-n entries + an 'Others' row."""
    series = series.sort_values(ascending=False)
    if len(series) <= n:
        return [f"{iso3_to_name.get(c, c)}{suffix}" for c in series.index], list(series.values)
    top_part   = series.iloc[:n-1]
    others_val = series.iloc[n-1:].sum()
    names  = [f"{iso3_to_name.get(c, c)}{suffix}" for c in top_part.index] + [f"Others{suffix}"]
    values = list(top_part.values) + [others_val]
    return names, values


def _plot_sankey(triples_df, mediator, iso3_to_name, top_n=ps.SANKEY_TOP_N):
    """Build one Plotly Sankey figure for a single mediator country."""
    df = triples_df[triples_df["Intermediary (B)"] == mediator].copy()
    if df.empty:
        return None

    source_names, a_values = _get_top_n_plus_others(
        df.groupby("Moisture Source (A)")["ET→P Weight"].sum(), top_n, iso3_to_name)
    dest_names, c_values   = _get_top_n_plus_others(
        df.groupby("Trade Destination (C)")["VWT Weight"].sum(), top_n, iso3_to_name, suffix=" ")
    mediator_name          = iso3_to_name.get(mediator, mediator)

    all_nodes     = source_names + [mediator_name] + dest_names
    display_labels = ["" if nd == mediator_name else nd for nd in all_nodes]
    node_idx      = {name: i for i, name in enumerate(all_nodes)}

    # Node colours
    node_colors = []
    for name in all_nodes:
        if "Others" in name:
            node_colors.append("#BDBDBD")
        elif name == mediator_name:
            node_colors.append(ps.SANKEY_MEDIATOR_COLOR)
        elif name in source_names:
            node_colors.append(ps.SANKEY_SOURCE_COLOR)
        else:
            node_colors.append(ps.SANKEY_DEST_COLOR)

    # Coordinates
    a_y      = _calculate_spaced_y(a_values)
    c_y      = _calculate_spaced_y(c_values)
    x_coords = [0.01] * len(source_names) + [0.5] + [0.99] * len(dest_names)
    y_coords = a_y + [0.5] + c_y

    # Links
    sources, targets, values, link_colors = [], [], [], []
    for label, val in zip(source_names, a_values):
        sources.append(node_idx[label])
        targets.append(node_idx[mediator_name])
        values.append((val / sum(a_values)) * 100)
        link_colors.append("rgba(232, 162, 121, 0.4)")

    for label, val in zip(dest_names, c_values):
        sources.append(node_idx[mediator_name])
        targets.append(node_idx[label])
        values.append((val / sum(c_values)) * 100)
        link_colors.append("rgba(98, 131, 152, 0.4)")

    width_px  = 48 / 25.4 * 72
    height_px = 55 / 25.4 * 72

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=10, thickness=10,
            label=display_labels,
            color=node_colors,
            line=dict(width=0),
            x=x_coords, y=y_coords,
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            hovertemplate="%{value:.1f}%<extra></extra>",
        ),
    ))
    fig.update_layout(
        title=dict(text=f"<b>{mediator_name}</b>", x=0.5, y=0.98,
                   font=dict(size=10)),
        font=dict(size=7, family="Arial", color="black"),
        width=width_px, height=height_px,
        margin=dict(l=7, r=7, t=10, b=3),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


def _build_summary_df(triples_df, mediators, iso3_to_name, top_n):
    """Collect source/destination flow shares for all mediators into one DataFrame."""
    rows = []
    for mediator in mediators:
        df = triples_df[triples_df["Intermediary (B)"] == mediator].copy()
        if df.empty:
            continue
        source_totals = df.groupby("Moisture Source (A)")["ET→P Weight"].sum()
        dest_totals   = df.groupby("Trade Destination (C)")["VWT Weight"].sum()
        total_etp     = source_totals.sum()
        total_vwt     = dest_totals.sum()

        for side, series, total in [
            ("Source (Moisture)", source_totals, total_etp),
            ("Destination (Trade)", dest_totals, total_vwt),
        ]:
            top = series.nlargest(top_n)
            for rank, (iso, val) in enumerate(top.items(), 1):
                rows.append(dict(
                    mediator_iso=mediator,
                    mediator_name=iso3_to_name.get(mediator, mediator),
                    side=side, rank=rank,
                    country_iso=iso,
                    country_name=iso3_to_name.get(iso, iso),
                    raw_weight=val,
                    pct_of_side=(val / total * 100) if total > 0 else 0,
                ))
            others_val = series.sum() - top.sum()
            if others_val > 0:
                rows.append(dict(
                    mediator_iso=mediator,
                    mediator_name=iso3_to_name.get(mediator, mediator),
                    side=side, rank=top_n + 1,
                    country_iso="OTHERS", country_name="Other",
                    raw_weight=others_val,
                    pct_of_side=(others_val / total * 100) if total > 0 else 0,
                ))
    return pd.DataFrame(rows)


# =============================================================================
# Public entry point
# =============================================================================

def make_figure(mediators=None, save=ps.SANKEY_DIR):
    """Build Sankey figures for each mediator country and optionally save them.

    Parameters
    ----------
    mediators : list of str, optional  — ISO-3 codes; defaults to plot_settings.SANKEY_MEDIATORS
    save      : Path or False

    Returns
    -------
    dict  {iso3: plotly.graph_objects.Figure}
    """
    mediators  = mediators or ps.SANKEY_MEDIATORS
    top_n      = ps.SANKEY_TOP_N

    meta_df    = load_meta_data(ps.META_PATH)
    iso3_to_name = meta_df.set_index("iso3")["name"].to_dict()

    atmos_df   = load_network(ps.ATMOS_PATH)
    vwt_df     = load_network(ps.VWT_PATH)
    G_atmos    = build_thresholded_graph(atmos_df, percentile=ps.THRESHOLD_PERCENTILE, directed=True)
    G_vwt      = build_thresholded_graph(vwt_df,   percentile=ps.THRESHOLD_PERCENTILE, directed=True)
    triples_df = get_feedforward_triples(G_atmos, G_vwt, include_weights=True)

    figures = {}
    for mediator in mediators:
        fig = _plot_sankey(triples_df, mediator, iso3_to_name, top_n=top_n)
        if fig is None:
            print(f"  Skipped {mediator}: no triples data.")
            continue
        figures[mediator] = fig
        fig.show()
        if save:
            save.mkdir(parents=True, exist_ok=True)
            fig.write_image(save / f"fig5b_sankey_{mediator}.svg")

    if save:
        summary = _build_summary_df(triples_df, mediators, iso3_to_name, top_n)
        summary.to_csv(save / "fig5b_sankey_summary.csv", index=False)
        print(f"Saved summary → {save / 'fig5b_sankey_summary.csv'}")

    return figures


if __name__ == "__main__":
    make_figure()
