# =============================================================================
# fig2_chord_diagrams.py
# Chord diagrams for the Atmospheric Moisture and VWT networks.
# Run standalone:  python fig2_chord_diagrams.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot_settings as ps
import plot_functions as pf

# analysis_functions re-used for network loading (avoids duplication)
from analysis_functions import load_network, load_meta_data


# =============================================================================
# Figure-specific helpers
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

    cmap        = plt.get_cmap(ps.CHORD_NODE_CMAP)
    node_colors = {nd: cmap(i / max(n-1, 1)) for i, nd in enumerate(nodes)}

    # Draw ribbons
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
            i_src   = node_idx[src]
            i_tgt   = node_idx[tgt]
            src_span = arc_sizes[i_src] * (w / out_strengths[src])
            tgt_span = arc_sizes[i_tgt] * (w / in_strengths[tgt])

            a_src0, a_src1 = out_cursor[src], out_cursor[src] - src_span
            a_tgt0, a_tgt1 = in_cursor[tgt],  in_cursor[tgt]  - tgt_span
            out_cursor[src] = a_src1
            in_cursor[tgt]  = a_tgt1

            alpha = ps.CHORD_MIN_ALPHA + (ps.CHORD_MAX_ALPHA - ps.CHORD_MIN_ALPHA) * (w - w_min) / w_range
            _ribbon_bezier(ax, a_src0, a_src1, a_tgt0, a_tgt1,
                           color=node_colors[src], alpha=alpha)

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

    _draw_chord_diagram(axes[0], atmos_df, "Atmospheric moisture flow network", node_labels=iso_to_name)
    _draw_chord_diagram(axes[1], vwt_df,   "Virtual water trade network", node_labels=iso_to_name)

    if save:
        pf.save_fig(fig, "fig2_chord_diagrams", folder=save, dpi=ps.CHORD_DPI)

    return fig


if __name__ == "__main__":
    make_figure()
    plt.show()
