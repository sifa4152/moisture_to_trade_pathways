#!/usr/bin/env python3
# =============================================================================
# main.py
# Entry point for the atmospheric moisture / virtual water workflow.
#
# Run
# ---
#   python main.py
#
# All parameters are controlled via settings.py.  Set the analysis switches
# RUN_NETWORK_METRICS / RUN_CROSS_NETWORK / RUN_KNOCKOUT to True/False to
# enable or disable individual streams.
# =============================================================================

import sys
import time
from pathlib import Path
import pandas as pd

import settings as cfg
import analysis_functions as af


# =============================================================================
# Helpers
# =============================================================================

def _section(logger, title):
    """Log a clearly visible section header."""
    bar = "─" * 60
    logger.info(bar)
    logger.info("  %s", title)
    logger.info(bar)


# =============================================================================
# Individual analysis streams
# =============================================================================

def run_network_metrics(logger, atmos_df, vwt_df, meta_df, world_gdf):
    """Stream 1 — per-network node-level measures (strength, betweenness …)."""
    _section(logger, "STREAM 1: Network Metrics")

    graphs      = {"atmos": af.build_thresholded_graph(atmos_df,
                                percentile=cfg.THRESHOLD_PERCENTILE,
                                directed=cfg.DIRECTED_GRAPH,
                                inverse_weight=cfg.INVERSE_WEIGHT),
                   "vwt":   af.build_thresholded_graph(vwt_df,
                                percentile=cfg.THRESHOLD_PERCENTILE,
                                directed=cfg.DIRECTED_GRAPH,
                                inverse_weight=cfg.INVERSE_WEIGHT)}

    metrics_dict = af.compute_metrics_for_graphs(graphs,
                                                  meta_df=meta_df,
                                                  world_gdf=world_gdf)
    metrics_df   = af.merge_metrics(metrics_dict)

    logger.info("Network metrics computed for %d nodes.", len(metrics_df))

    if cfg.SAVE_RESULTS:
        out = cfg.OUTPUT_DIR / cfg.NETWORK_METRICS_CSV
        out.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out)
        logger.info("Saved → %s", out)

    return metrics_df


def run_cross_network(logger, atmos_df, vwt_df, meta_df):
    """Stream 2 — feed-forward triples, dependency, asymmetry, mediator betweenness."""
    _section(logger, "STREAM 2: Cross-Network Metrics")

    G_atmos = af.build_thresholded_graph(atmos_df,
                  percentile=cfg.THRESHOLD_PERCENTILE,
                  directed=cfg.DIRECTED_GRAPH,
                  inverse_weight=cfg.INVERSE_WEIGHT)
    G_vwt   = af.build_thresholded_graph(vwt_df,
                  percentile=cfg.THRESHOLD_PERCENTILE,
                  directed=cfg.DIRECTED_GRAPH,
                  inverse_weight=cfg.INVERSE_WEIGHT)

    # --- feed-forward triples pipeline ---
    triples_df   = af.get_feedforward_triples(G_atmos, G_vwt, include_weights=True)
    logger.info("Feed-forward triples found: %d", len(triples_df))

    triples_df   = af.enrich_triples_with_precip(triples_df, meta_df)
    triples_df   = af.calculate_volumetric_attribution(
                        triples_df, weight_cols=("ET→P Weight", "VWT Weight"))
    logger.info("Triples after volumetric attribution: %d", len(triples_df))

    # --- cross-network metrics ---
    dependency_results = af.atmo_to_vwt_dependency(triples_df, return_matrix=True)
    asymmetry_index    = af.atmo_vwt_asymmetry(triples_df, threshold=cfg.ASYM_THRESHOLD)

    # Exclude self-dependency (A == C) for the external-only view
    external_only      = triples_df[
        triples_df["Moisture Source (A)"] != triples_df["Trade Destination (C)"]
    ]
    _                  = af.atmo_to_vwt_dependency(external_only)  # available for extension

    # --- mediator betweenness in tripartite graph ---
    G_tri              = af.build_tripartite_graph(triples_df)
    mediators          = triples_df["Intermediary (B)"].unique()
    mediator_importance= af.calculate_mediator_betweenness(G_tri, mediators)

    logger.info("Cross-network metrics computed for %d countries.",
                len(dependency_results["country_dependency"]))

    if cfg.SAVE_RESULTS:
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        af.write_cross_network_results_to_excel(
            dependency_results,
            asymmetry_index,
            mediator_importance,
            cfg.OUTPUT_DIR / cfg.CROSS_NETWORK_METRICS_XLSX,
        )

        triples_csv = cfg.OUTPUT_DIR / cfg.FEEDFORWARD_TRIPLES_CSV
        triples_df.to_csv(triples_csv, index=False)
        logger.info("Saved triples → %s", triples_csv)

    return {
        "triples":              triples_df,
        "dependency_results":   dependency_results,
        "asymmetry_index":      asymmetry_index,
        "mediator_importance":  mediator_importance,
    }


def run_knockout(logger, atmos_df, vwt_df, meta_df):
    """Stream 3 — mediator knockout simulation."""
    _section(logger, "STREAM 3: Knockout Effects")

    G_atmos = af.build_thresholded_graph(atmos_df,
                  percentile=cfg.THRESHOLD_PERCENTILE,
                  directed=cfg.DIRECTED_GRAPH,
                  inverse_weight=cfg.INVERSE_WEIGHT)
    G_vwt   = af.build_thresholded_graph(vwt_df,
                  percentile=cfg.THRESHOLD_PERCENTILE,
                  directed=cfg.DIRECTED_GRAPH,
                  inverse_weight=cfg.INVERSE_WEIGHT)

    triples_df = af.get_feedforward_triples(G_atmos, G_vwt, include_weights=True)
    triples_df = af.enrich_triples_with_precip(triples_df, meta_df)
    triples_df = af.calculate_volumetric_attribution(
                      triples_df, weight_cols=("ET→P Weight", "VWT Weight"))

    knockout_results = af.calculate_triple_knockout(triples_df)
    knockout_results["mediators_per_destination"] = af.count_mediators_per_destination(triples_df)

    logger.info("Knockout simulation complete — %d mediators evaluated.",
                len(knockout_results["per_mediator"]))

    if cfg.SAVE_RESULTS:
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        af.write_knockout_results_to_excel(
            knockout_results,
            cfg.OUTPUT_DIR / cfg.KNOCKOUT_RESULTS_XLSX,
            metadata={"Threshold_percentile": cfg.THRESHOLD_PERCENTILE},
        )

    return knockout_results


def run_sensitivity(logger, atmos_df, vwt_df, meta_df):
    """Sweep THRESHOLD_PERCENTILE across cfg.SENSITIVITY_THRESHOLDS.
    
    For each threshold: builds graphs, runs cross-network and knockout streams,
    and records scalar summary statistics. Does not write per-threshold Excel
    files — only a single summary CSV at the end.
    """
    _section(logger, "STREAM 4: Sensitivity Analysis")
    
    rows = []
    
    for pct in cfg.SENSITIVITY_THRESHOLDS:
        logger.info("  Threshold: %d %%", pct)
        
        G_atmos = af.build_thresholded_graph(atmos_df, percentile=pct, directed=True)
        G_vwt   = af.build_thresholded_graph(vwt_df,   percentile=pct, directed=True)
        
        # --- structural ---
        n_edges_atmos = G_atmos.number_of_edges()
        n_edges_vwt   = G_vwt.number_of_edges()
        
        # --- triples pipeline (shared by both streams) ---
        triples = af.get_feedforward_triples(G_atmos, G_vwt, include_weights=True)
        triples = af.enrich_triples_with_precip(triples, meta_df)
        triples = af.calculate_volumetric_attribution(triples)
        
        n_triples         = len(triples)
        total_system_flow = triples["contribution"].sum()
        
        # --- cross-network ---
        dep_results    = af.atmo_to_vwt_dependency(triples)
        dep            = dep_results["country_dependency"]
        asym           = af.atmo_vwt_asymmetry(triples)
        
        # --- knockout ---
        ko             = af.calculate_triple_knockout(triples)
        ko_med         = ko["per_mediator"].set_index("Mediator")
        topn_mediators = (ko_med["Global_KO_volume_share"]
                            .nlargest(cfg.TOP_N_MEDIATORS).index.tolist())
        
        rows.append({
            "threshold":            pct,
            "n_edges_atmos":        n_edges_atmos,
            "n_edges_vwt":          n_edges_vwt,
            "n_triples":            n_triples,
            "total_system_flow_m3": total_system_flow,
            "dep_mean":             dep.mean(),
            "dep_median":           dep.median(),
            "dep_std":              dep.std(),
            "asym_mean":            asym.mean(),
            "asym_median":          asym.median(),
            "asym_std":             asym.std(),
            "topn_mediators":       ", ".join(topn_mediators),
        })
        
    summary = pd.DataFrame(rows).set_index("threshold")
    logger.info("\n%s", summary.to_string())
    
    if cfg.SAVE_RESULTS:
        out = cfg.OUTPUT_DIR / "sensitivity_analysis.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out)
        logger.info("Saved → %s", out)
    
    return summary

# =============================================================================
# Shared data loading
# =============================================================================

def load_shared_data(logger):
    """Load meta data, ISO mapping, and network adjacency matrices.

    These are shared across all three streams.  The world shapefile is loaded
    only when geopandas is available and at least one stream needs it.

    Returns
    -------
    meta_df, world_gdf (may be None), atmos_df, vwt_df, id_to_iso
    """
    _section(logger, "Loading shared data")

    meta_df = af.load_meta_data(cfg.META_PATH)
    logger.info("Meta data loaded: %d rows.", len(meta_df))

    # World shapefile — optional; only required for country-name lookups
    world_gdf = None
    try:
        world_gdf = af.load_world_shp(cfg.WORLD_PATH)
        logger.info("World shapefile loaded: %d features.", len(world_gdf))
    except (FileNotFoundError, ImportError) as exc:
        logger.warning("World shapefile not loaded (%s). Country names may be missing.", exc)

    atmos_df = af.load_network(cfg.ATMOS_PATH, rm_self_loops=True)
    vwt_df   = af.load_network(cfg.VWT_PATH, rm_self_loops=True)
    logger.info("Adjacency matrices loaded — atmos: %s, vwt: %s.",
                atmos_df.shape, vwt_df.shape)

    matrix_countries = set(atmos_df.index)
    meta_countries   = set(meta_df["iso3"])
    missing_in_meta  = matrix_countries - meta_countries
    if missing_in_meta:
        logger.warning("Countries in adjacency matrix but not in meta: %s", missing_in_meta)
    
    return meta_df, world_gdf, atmos_df, vwt_df


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.perf_counter()

    # Configure the workflow logger (writes to console + optional log file)
    logger = af.configure_logger(
        name    = "workflow",
        level   = cfg.LOG_LEVEL,
        logfile = cfg.LOG_FILE,
    )

    # Propagate the configured logger to the functions module
    af.logger = logger

    logger.info("=" * 60)
    logger.info("  Atmospheric Moisture / Virtual Water Workflow")
    logger.info("  Threshold percentile : %s", cfg.THRESHOLD_PERCENTILE)
    logger.info("  Save results         : %s", cfg.SAVE_RESULTS)
    logger.info("  Output directory     : %s", cfg.OUTPUT_DIR)
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Check that at least one stream is enabled
    # -------------------------------------------------------------------------
    if not any([cfg.RUN_NETWORK_METRICS, cfg.RUN_CROSS_NETWORK, cfg.RUN_KNOCKOUT]):
        logger.error("All analysis streams are disabled in settings.py — nothing to do.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Load shared data once
    # -------------------------------------------------------------------------
    try:
        meta_df, world_gdf, atmos_df, vwt_df = load_shared_data(logger)
    except FileNotFoundError as exc:
        logger.error("Required data file not found: %s", exc)
        sys.exit(1)

    results = {}

    # -------------------------------------------------------------------------
    # Stream 1: Network metrics
    # -------------------------------------------------------------------------
    if cfg.RUN_NETWORK_METRICS:
        try:
            results["network_metrics"] = run_network_metrics(
                logger, atmos_df, vwt_df, meta_df, world_gdf)
        except Exception:
            logger.exception("Stream 1 (Network Metrics) failed.")

    # -------------------------------------------------------------------------
    # Stream 2: Cross-network metrics
    # -------------------------------------------------------------------------
    if cfg.RUN_CROSS_NETWORK:
        try:
            results["cross_network"] = run_cross_network(
                logger, atmos_df, vwt_df, meta_df)
        except Exception:
            logger.exception("Stream 2 (Cross-Network Metrics) failed.")

    # -------------------------------------------------------------------------
    # Stream 3: Knockout effects
    # -------------------------------------------------------------------------
    if cfg.RUN_KNOCKOUT:
        try:
            results["knockout"] = run_knockout(
                logger, atmos_df, vwt_df, meta_df)
        except Exception:
            logger.exception("Stream 3 (Knockout Effects) failed.")

    # -------------------------------------------------------------------------
    # Stream 4: Sensitivity analysis - test cutoff thresholds
    # -------------------------------------------------------------------------
    if cfg.RUN_SENSITIVITY:
        try:
            results["sensitivity"] = run_sensitivity(logger, atmos_df, vwt_df, meta_df)
        except Exception:
            logger.exception("Sensitivity analysis failed.")
            
    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    _section(logger, f"Workflow complete  ({elapsed:.1f} s)")

    return results


if __name__ == "__main__":
    main()
