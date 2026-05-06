# =============================================================================
# fig2_chord_diagrams.py
# Chord diagrams for the Atmospheric Moisture and VWT networks.
# Countries and ribbons are coloured by continent (source country).
# Directional tips (triangular wedges) are drawn at the target arc.
# Run standalone:  python fig2_chord_diagrams.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import plot_settings as ps
import plot_functions as pf

from analysis_functions import load_network, load_meta_data

from cmap import Colormap
# =============================================================================
# Continent colour palette + ISO3 → continent mapping
# =============================================================================

cm = Colormap('tol:rainbow_discrete_7')

CONTINENT_COLORS = {
    "Africa":        cm(0),
    "Asia":          cm(1),
    "Europe":        cm(5),
    "North America": cm(4),
    "South America": cm(3),
    "Oceania":       cm(2),
    "Antarctica":    "#AAAAAA",  # grey
    "Unknown":       "#888888",  # dark grey
}



# Mapping built from ISO 3166-1 alpha-3 → continent without external libraries.
# Covers all UN-recognised states and common territories.
_ISO3_TO_CONTINENT = {
    # Africa
    "DZA":"Africa","AGO":"Africa","BEN":"Africa","BWA":"Africa","BFA":"Africa",
    "BDI":"Africa","CPV":"Africa","CMR":"Africa","CAF":"Africa","TCD":"Africa",
    "COM":"Africa","COD":"Africa","COG":"Africa","CIV":"Africa","DJI":"Africa",
    "EGY":"Africa","GNQ":"Africa","ERI":"Africa","SWZ":"Africa","ETH":"Africa",
    "GAB":"Africa","GMB":"Africa","GHA":"Africa","GIN":"Africa","GNB":"Africa",
    "KEN":"Africa","LSO":"Africa","LBR":"Africa","LBY":"Africa","MDG":"Africa",
    "MWI":"Africa","MLI":"Africa","MRT":"Africa","MUS":"Africa","MAR":"Africa",
    "MOZ":"Africa","NAM":"Africa","NER":"Africa","NGA":"Africa","RWA":"Africa",
    "STP":"Africa","SEN":"Africa","SLE":"Africa","SOM":"Africa","ZAF":"Africa",
    "SSD":"Africa","SDN":"Africa","TZA":"Africa","TGO":"Africa","TUN":"Africa",
    "UGA":"Africa","ZMB":"Africa","ZWE":"Africa","REU":"Africa","MYT":"Africa",
    "ESH":"Africa","SHN":"Africa","IOT":"Africa",
    # Asia
    "AFG":"Asia","ARM":"Asia","AZE":"Asia","BHR":"Asia","BGD":"Asia",
    "BTN":"Asia","BRN":"Asia","KHM":"Asia","CHN":"Asia","CYP":"Asia",
    "GEO":"Asia","HKG":"Asia","IND":"Asia","IDN":"Asia","IRN":"Asia",
    "IRQ":"Asia","ISR":"Asia","JPN":"Asia","JOR":"Asia","KAZ":"Asia",
    "KWT":"Asia","KGZ":"Asia","LAO":"Asia","LBN":"Asia","MAC":"Asia",
    "MYS":"Asia","MDV":"Asia","MNG":"Asia","MMR":"Asia","NPL":"Asia",
    "PRK":"Asia","OMN":"Asia","PAK":"Asia","PHL":"Asia","QAT":"Asia",
    "SAU":"Asia","SGP":"Asia","KOR":"Asia","LKA":"Asia","SYR":"Asia",
    "TWN":"Asia","TJK":"Asia","THA":"Asia","TLS":"Asia","TUR":"Asia",
    "TKM":"Asia","ARE":"Asia","UZB":"Asia","VNM":"Asia","YEM":"Asia",
    "PSE":"Asia","RUS":"Asia",
    # Europe
    "ALB":"Europe","AND":"Europe","AUT":"Europe","BLR":"Europe","BEL":"Europe",
    "BIH":"Europe","BGR":"Europe","HRV":"Europe","CZE":"Europe","DNK":"Europe",
    "EST":"Europe","FIN":"Europe","FRA":"Europe","DEU":"Europe","GRC":"Europe",
    "HUN":"Europe","ISL":"Europe","IRL":"Europe","ITA":"Europe","XKX":"Europe",
    "LVA":"Europe","LIE":"Europe","LTU":"Europe","LUX":"Europe","MLT":"Europe",
    "MDA":"Europe","MCO":"Europe","MNE":"Europe","NLD":"Europe","MKD":"Europe",
    "NOR":"Europe","POL":"Europe","PRT":"Europe","ROU":"Europe","SMR":"Europe",
    "SRB":"Europe","SVK":"Europe","SVN":"Europe","ESP":"Europe","SWE":"Europe",
    "CHE":"Europe","UKR":"Europe","GBR":"Europe","VAT":"Europe","FRO":"Europe",
    "GIB":"Europe","GGY":"Europe","IMN":"Europe","JEY":"Europe","ALA":"Europe",
    "SJM":"Europe",
    # North America
    "ATG":"North America","BHS":"North America","BRB":"North America",
    "BLZ":"North America","CAN":"North America","CRI":"North America",
    "CUB":"North America","DMA":"North America","DOM":"North America",
    "SLV":"North America","GRD":"North America","GTM":"North America",
    "HTI":"North America","HND":"North America","JAM":"North America",
    "MEX":"North America","NIC":"North America","PAN":"North America",
    "KNA":"North America","LCA":"North America","VCT":"North America",
    "TTO":"North America","USA":"North America","GRL":"North America",
    "PRI":"North America","VIR":"North America","CYM":"North America",
    "BMU":"North America","SPM":"North America","TCA":"North America",
    "ABW":"North America","CUW":"North America","SXM":"North America",
    "AIA":"North America","MSR":"North America","VGB":"North America",
    "BLM":"North America","MAF":"North America",
    # South America
    "ARG":"South America","BOL":"South America","BRA":"South America",
    "CHL":"South America","COL":"South America","ECU":"South America",
    "GUY":"South America","PRY":"South America","PER":"South America",
    "SUR":"South America","URY":"South America","VEN":"South America",
    "GUF":"South America","FLK":"South America",
    # Oceania
    "AUS":"Oceania","FJI":"Oceania","KIR":"Oceania","MHL":"Oceania",
    "FSM":"Oceania","NRU":"Oceania","NZL":"Oceania","PLW":"Oceania",
    "PNG":"Oceania","WSM":"Oceania","SLB":"Oceania","TON":"Oceania",
    "TUV":"Oceania","VUT":"Oceania","NCL":"Oceania","PYF":"Oceania",
    "GUM":"Oceania","MNP":"Oceania","COK":"Oceania","NIU":"Oceania",
    "TKL":"Oceania","WLF":"Oceania","ASM":"Oceania","HMD":"Oceania",
    "NFK":"Oceania","PCN":"Oceania","CXR":"Oceania","CCK":"Oceania",
    # Antarctica
    "ATA":"Antarctica",
}

def _get_continent_color(iso3):
    continent = _ISO3_TO_CONTINENT.get(str(iso3).upper(), "Unknown")
    return CONTINENT_COLORS.get(continent, CONTINENT_COLORS["Unknown"])


# =============================================================================
# Directional tip (triangular wedge at target arc)
# =============================================================================

# def _draw_tip(ax, a_tgt0, a_tgt1, color, alpha,
#               r=None, tip_depth=0.07):
#     """Draw a filled triangular arrowhead pointing inward at the target arc.

#     The tip apex points toward the centre; the base sits on the inner radius arc.
#     """
#     if r is None:
#         r = ps.CHORD_INNER_RADIUS

#     a_mid = (a_tgt0 + a_tgt1) / 2

#     # Base corners on the inner arc
#     base_r = r
#     bx0, by0 = np.cos(a_tgt0) * base_r, np.sin(a_tgt0) * base_r
#     bx1, by1 = np.cos(a_tgt1) * base_r, np.sin(a_tgt1) * base_r

#     # Apex slightly inside the circle
#     apex_r = r - tip_depth
#     ax_pt, ay_pt = np.cos(a_mid) * apex_r, np.sin(a_mid) * apex_r

#     triangle_x = [bx0, bx1, ax_pt]
#     triangle_y = [by0, by1, ay_pt]
#     ax.fill(triangle_x, triangle_y, color=color, alpha=min(alpha + 0.2, 1.0),
#             lw=0, zorder=3)


# =============================================================================
# Ribbon drawing (unchanged geometry, colour now passed from continent palette)
# =============================================================================

def _ribbon_bezier(ax, a_src0, a_src1, a_tgt0, a_tgt1, color, alpha,
                   r=ps.CHORD_INNER_RADIUS, n=120):
    """Draw a filled Bezier ribbon connecting two arcs on the chord circle."""
    t  = np.linspace(0, 1, n)
    p0 = np.array([np.cos(a_src0), np.sin(a_src0)]) * r
    p1 = np.array([np.cos(a_src1), np.sin(a_src1)]) * r
    p2 = np.array([np.cos(a_tgt0), np.sin(a_tgt0)]) * r
    p3 = np.array([np.cos(a_tgt1), np.sin(a_tgt1)]) * r
    ctrl = np.zeros(2)

    c1x = (1-t)**2*p0[0] + 2*(1-t)*t*ctrl[0] + t**2*p2[0]
    c1y = (1-t)**2*p0[1] + 2*(1-t)*t*ctrl[1] + t**2*p2[1]
    c2x = (1-t)**2*p1[0] + 2*(1-t)*t*ctrl[0] + t**2*p3[0]
    c2y = (1-t)**2*p1[1] + 2*(1-t)*t*ctrl[1] + t**2*p3[1]

    theta_src = np.linspace(a_src0, a_src1, 20)
    theta_tgt = np.linspace(a_tgt0, a_tgt1, 20)

    poly_x = np.concatenate([c1x, np.cos(theta_tgt)*r, c2x[::-1], np.cos(theta_src[::-1])*r])
    poly_y = np.concatenate([c1y, np.sin(theta_tgt)*r, c2y[::-1], np.sin(theta_src[::-1])*r])
    ax.fill(poly_x, poly_y, color=color, alpha=alpha, lw=0)


# =============================================================================
# Main chord diagram drawing function
# =============================================================================

def _draw_chord_diagram(ax, df_raw, title,
                        percentile=ps.THRESHOLD_PERCENTILE,
                        top_n=ps.CHORD_TOP_N_NODES,
                        node_labels=None):
    """Draw one chord diagram onto *ax*."""
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    ax.set_facecolor(ps.CHORD_BG_COLOR)
    ax.set_title(title, color="black", fontsize=18, fontweight="bold",
                 fontfamily="monospace", pad=50)

    # Select top-N nodes by total strength
    total_strength = df_raw.sum(axis=1) + df_raw.sum(axis=0)
    top_nodes      = total_strength.nlargest(top_n).index.tolist()
    df             = df_raw.loc[top_nodes, top_nodes].fillna(0.0)

    # Threshold edges
    vals = df.values[df.values > 0]
    if vals.size == 0:
        ax.text(0, 0, "No edges", color="black", ha="center", va="center")
        return
    thresh    = np.nanpercentile(vals, percentile)
    df_thresh = df.where(df >= thresh, other=0.0)

    # Node layout
    node_strength = df_thresh.sum(axis=1) + df_thresh.sum(axis=0)
    nodes         = node_strength.sort_values(ascending=False).index.tolist()
    n             = len(nodes)
    node_idx      = {nd: i for i, nd in enumerate(nodes)}

    gap        = 2 * np.pi * 0.005
    arc_span   = 2 * np.pi - gap * n
    strengths  = np.array([max(node_strength[nd], 1e-9) for nd in nodes])
    arc_sizes  = np.maximum(strengths / strengths.sum() * arc_span, arc_span * 0.004)
    arc_sizes  = arc_sizes / arc_sizes.sum() * arc_span

    start_angles = np.zeros(n)
    angle        = np.pi / 2
    for i in range(n):
        start_angles[i] = angle
        angle          -= arc_sizes[i] + gap
    mid_angles = start_angles - arc_sizes / 2

    # Continent-based node colours
    node_colors = {nd: _get_continent_color(nd) for nd in nodes}

    # Draw ribbons + directional tips
    edge_list = [(s, t, df_thresh.at[s, t]) for s in nodes for t in nodes
                 if df_thresh.at[s, t] > 0 and s != t]
    if edge_list:
        w_vals  = [e[2] for e in edge_list]
        w_min, w_max = min(w_vals), max(w_vals)
        w_range = w_max - w_min if w_max > w_min else 1.0
        edge_list.sort(key=lambda x: x[2])

        out_cursor    = {nd: start_angles[node_idx[nd]] for nd in nodes}
        in_cursor     = {nd: start_angles[node_idx[nd]] for nd in nodes}
        out_strengths = {nd: max(df_thresh.loc[nd].sum(), 1e-9) for nd in nodes}
        in_strengths  = {nd: max(df_thresh[nd].sum(), 1e-9)  for nd in nodes}

        for src, tgt, w in edge_list:
            i_src    = node_idx[src]
            i_tgt    = node_idx[tgt]
            src_span = arc_sizes[i_src] * (w / out_strengths[src])
            tgt_span = arc_sizes[i_tgt] * (w / in_strengths[tgt])

            a_src0, a_src1 = out_cursor[src], out_cursor[src] - src_span
            a_tgt0, a_tgt1 = in_cursor[tgt],  in_cursor[tgt]  - tgt_span
            out_cursor[src] = a_src1
            in_cursor[tgt]  = a_tgt1

            alpha = ps.CHORD_MIN_ALPHA + (ps.CHORD_MAX_ALPHA - ps.CHORD_MIN_ALPHA) * (w - w_min) / w_range
            color = node_colors[src]   # ribbon coloured by source continent

            _ribbon_bezier(ax, a_src0, a_src1, a_tgt0, a_tgt1,
                           color=color, alpha=alpha)
            # _draw_tip(ax, a_tgt0, a_tgt1, color=color, alpha=alpha,
            #           r=ps.CHORD_INNER_RADIUS)

    # Draw node arcs
    outer_r = ps.CHORD_INNER_RADIUS + ps.CHORD_RING_WIDTH
    for i, nd in enumerate(nodes):
        a0    = start_angles[i]
        a1    = a0 - arc_sizes[i]
        theta = np.linspace(a0, a1, 60)
        xs_o  = np.cos(theta) * outer_r
        ys_o  = np.sin(theta) * outer_r
        xs_i  = np.cos(theta[::-1]) * ps.CHORD_INNER_RADIUS
        ys_i  = np.sin(theta[::-1]) * ps.CHORD_INNER_RADIUS
        ax.fill(np.concatenate([xs_o, xs_i]), np.concatenate([ys_o, ys_i]),
                color=node_colors[nd], alpha=0.9, lw=0)
        ax.plot(xs_o, ys_o, color="black", lw=ps.CHORD_ARC_LWIDTH, alpha=0.3)

    # Labels
    label_r = outer_r + ps.CHORD_LABEL_OFFSET * 0.6
    for i, nd in enumerate(nodes):
        angle = mid_angles[i]
        x, y  = np.cos(angle) * label_r, np.sin(angle) * label_r
        deg   = np.degrees(angle)
        ha, rotation = ("left", deg) if -90 < deg <= 90 else ("right", deg + 180)
        label = node_labels.get(nd, str(nd)) if node_labels else str(nd)
        ax.text(x, y, label, ha=ha, va="center",
                rotation=rotation, rotation_mode="anchor",
                fontsize=14, fontfamily="monospace", color="black", alpha=0.85,
                fontweight="bold")


# =============================================================================
# Shared continent legend
# =============================================================================

def _add_continent_legend(fig):
    """Add a single shared continent colour legend at the bottom of the figure."""
    patches = [
        mpatches.Patch(color=color, label=continent)
        for continent, color in CONTINENT_COLORS.items()
        if continent not in ("Unknown","Antarctica")  # Exclude "Unknown" and "Antarctica" from legend
    ]
    legend = fig.legend(
        handles=patches,
        loc="lower center",
        ncol=len(patches),
        fontsize=11,
        frameon=False,
        title="Continent (source)",
        title_fontsize=11,
        bbox_to_anchor=(0.5, -0.01),
    )
    legend.get_title().set_fontweight("bold")


# =============================================================================
# Public entry point
# =============================================================================

def make_figure(save=ps.FIGS_DIR):
    """Build and optionally save fig2 (chord diagrams).

    Parameters
    ----------
    save : Path or False  — folder to save into; pass False to skip saving.

    Returns
    -------
    matplotlib.figure.Figure
    """
    meta_df   = load_meta_data(ps.META_PATH)
    atmos_df  = load_network(ps.ATMOS_PATH)
    vwt_df    = load_network(ps.VWT_PATH)

    iso_to_name = meta_df.set_index("iso3")["name"].to_dict()

    fig, axes = plt.subplots(1, 2, figsize=ps.FIGSIZE_CHORD,
                             facecolor=ps.CHORD_BG_COLOR)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.07, wspace=-0.15)

    _draw_chord_diagram(axes[0], atmos_df, "Atmospheric moisture flow network",
                        node_labels=iso_to_name)
    _draw_chord_diagram(axes[1], vwt_df,   "Virtual water trade network",
                        node_labels=iso_to_name)

    _add_continent_legend(fig)

    if save:
        pf.save_fig(fig, "fig2_chord_diagrams_v2", folder=save, dpi=ps.CHORD_DPI)

    return fig


if __name__ == "__main__":
    make_figure()
    plt.show()
