# =============================================================================
# plot_all.py
# Optional batch runner — regenerates all figures and stats in one go.
# This is NOT the primary entry point; run individual figure scripts directly
# when iterating on a single figure.
#
# Usage:  python plot_all.py
# =============================================================================

import time
import matplotlib.pyplot as plt

import plot_settings as ps

import fig2_chord_diagrams
import fig3_asym_gdp_maps
import fig4_exposure_n_mediators
import fig5a_total_flow_loss
import fig5b_sankey
import stats_analysis


def main():
    t0 = time.perf_counter()
    print("=" * 55)
    print("  Regenerating all figures")
    print(f"  Output → {ps.FIGS_DIR}")
    print("=" * 55)

    steps = [
        ("Fig 2  — Chord diagrams",          lambda: fig2_chord_diagrams.make_figure()),
        ("Fig 3  — Asymmetry / GDP maps",     lambda: fig3_asym_gdp_maps.make_figure()),
        ("Fig 4  — Exposure / mediators",     lambda: fig4_exposure_n_mediators.make_figure()),
        ("Fig 5a — Total flow loss",          lambda: fig5a_total_flow_loss.make_figure()),
        ("Fig 5b — Sankey diagrams",          lambda: fig5b_sankey.make_figure()),
        ("Stats  — Statistical analysis",     lambda: stats_analysis.run_targeted_stats()),
    ]

    for label, fn in steps:
        print(f"\n  {label} …")
        try:
            fn()
            plt.close("all")   # free memory between figures
        except Exception as exc:
            print(f"  ✗  FAILED: {exc}")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 55}")
    print(f"  Done  ({elapsed:.1f} s)")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
