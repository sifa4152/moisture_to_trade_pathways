# =============================================================================
# analysis_functions.py
# Function repository for the atmospheric moisture / virtual water workflow.
#
# Sections
# --------
#   1.  Logging
#   2.  Data loading & preparation   (shared across all streams)
#   3.  Network metrics              (stream 1)
#   4.  Cross-network metrics        (stream 2)
#   5.  Knockout effects             (stream 3)
#   6.  Results I/O helpers
# =============================================================================

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# geopandas is only needed for load_world_shp; keep import lazy-ish so the
# rest of the module works even if geopandas is absent.
try:
    import geopandas as gpd
    _HAS_GEOPANDAS = True
except ImportError:
    _HAS_GEOPANDAS = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# =============================================================================
# 1. Logging
# =============================================================================

def configure_logger(name="workflow", level=None, logfile=None, to_console=True):
    """Configure and return a named logger.

    Parameters
    ----------
    name : str
        Logger name (use __name__ when calling from a module).
    level : str or int, optional
        Logging level.  Falls back to the ``ND_LOG_LEVEL`` environment
        variable, then ``INFO``.
    logfile : str or Path, optional
        Rotating log file path.  Omit to log to console only.
    to_console : bool
        Attach a StreamHandler when True.

    Returns
    -------
    logging.Logger
    """
    lvl = level or os.getenv("ND_LOG_LEVEL", "INFO")
    if isinstance(lvl, str):
        lvl = getattr(logging, lvl.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    logger.propagate = False
    logger.handlers = []  # idempotent on re-import / reload

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    if to_console:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if logfile:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(logfile, maxBytes=10_000_000, backupCount=5)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Module-level fallback logger (overridden by main.py)
logger = configure_logger("analysis_functions")


# =============================================================================
# 2. Data loading & preparation
# =============================================================================

def load_world_shp(path, required_cols=("NAME_ENGL", "CNTR_ID", "ISO3_CODE", "geometry")):
    """Read a world shapefile and return a GeoDataFrame indexed by CNTR_ID.

    Parameters
    ----------
    path : str or Path
    required_cols : tuple of str
        Columns to select; missing columns are silently skipped.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    if not _HAS_GEOPANDAS:
        raise ImportError("geopandas is required for load_world_shp.")

    path = Path(path)
    if not path.exists():
        logger.error("World shapefile not found: %s", path)
        raise FileNotFoundError(path)

    try:
        world = gpd.read_file(path, columns=list(required_cols))
    except Exception:
        try:
            world = gpd.read_file(path)
        except Exception:
            logger.exception("Failed to read world shapefile: %s", path)
            raise
        missing = [c for c in required_cols if c not in world.columns]
        if missing:
            logger.info("World file is missing columns (skipped): %s", missing)
        world = world[[c for c in required_cols if c in world.columns]]

    if "CNTR_ID" in world.columns:
        world = world.set_index("CNTR_ID")
    else:
        logger.info("'CNTR_ID' not present in world shapefile; default index kept.")

    return world


def load_meta_data(path, numeric_cols=("area_m2", "pop_2008_2017", "p_total_m3")):
    """Read the meta CSV and coerce numeric columns.

    Parameters
    ----------
    path : str or Path
    numeric_cols : tuple of str
        Columns to coerce to numeric (non-present columns are silently skipped).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        logger.error("Meta CSV not found: %s", path)
        raise FileNotFoundError(path)

    try:
        meta = pd.read_csv(path, keep_default_na=False)
    except Exception:
        logger.exception("Failed to load meta CSV: %s", path)
        raise

    present_numeric = [c for c in numeric_cols if c in meta.columns]
    for c in present_numeric:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")

    return meta


def load_network(data_path, rm_self_loops=True):
    """Load and preprocess a network adjacency matrix from CSV.

    Steps performed
    ---------------
    1. Read CSV (first column → row index).
    2. Coerce all values to numeric (non-numeric → 0).
    3. Optionally zero the diagonal (self-loops).

    Parameters
    ----------
    data_path : str or Path
    rm_self_loops : bool

    Returns
    -------
    pd.DataFrame  — square adjacency matrix
    """
    data_path = Path(data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path, index_col=0)

    try:
        df.columns = df.columns.astype(int)
    except Exception:
        pass  # String labels (e.g. ISO codes) are fine

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    if rm_self_loops:
        for i in df.index:
            if i in df.columns:
                df.at[i, i] = 0.0

    return df


def build_thresholded_graph(df, percentile=99, directed=True, inverse_weight=True):
    """Apply a percentile threshold and return a NetworkX graph.

    Parameters
    ----------
    df : pd.DataFrame
        Square adjacency matrix with numeric values.
    percentile : float
        Edges with weight below this percentile of *positive* weights are
        removed.  ``percentile=90`` keeps the top 10 % of positive weights.
    directed : bool
    inverse_weight : bool
        When True, add a ``'length'`` edge attribute equal to ``1 / weight``
        (used for shortest-path betweenness, where *shorter* = *stronger*).

    Returns
    -------
    nx.DiGraph or nx.Graph
    """
    if not (0 <= percentile <= 100):
        raise ValueError("percentile must be between 0 and 100.")

    values = df.values
    positive_values = values[values > 0]

    graph_type = nx.DiGraph if directed else nx.Graph

    if positive_values.size == 0:
        logger.warning("Adjacency matrix has no positive weights; returning node-only graph.")
        G = graph_type()
        G.add_nodes_from(df.index)
        return G

    threshold = float(np.nanpercentile(positive_values, percentile))
    filtered_df = df.where(df >= threshold, other=0.0)
    G = nx.from_pandas_adjacency(filtered_df, create_using=graph_type)

    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        d["weight"] = w
        if inverse_weight:
            d["length"] = (1.0 / w) if w > 0 else float("inf")

    return G


# =============================================================================
# 3. Network metrics  (stream 1: network_metrics)
# =============================================================================

def _get_iso_to_country_map(meta_df=None, world_gdf=None):
    """Return a dict mapping ISO-3 → country name.

    Prefers *meta_df* over *world_gdf*.  Returns ``{}`` if neither source
    provides a usable name column.
    """
    if meta_df is not None and "iso3" in meta_df.columns:
        name_cols = [c for c in meta_df.columns
                     if c.lower() in ("country", "country_name", "name",
                                      "countryname", "name_english", "name_en",
                                      "name_engl")]
        if name_cols:
            return pd.Series(meta_df[name_cols[0]].values,
                             index=meta_df["iso3"]).to_dict()

    if world_gdf is not None and "ISO3_CODE" in world_gdf.columns:
        return {row.get("ISO3_CODE"): (row.get("NAME_ENGL") or row.get("NAME"))
                for _, row in world_gdf.iterrows() if row.get("ISO3_CODE")}

    return {}


def compute_directed_measures(G, meta_df=None, world_gdf=None,
                               weight_attr="weight", length_attr="length"):
    """Compute directed, weighted node-level measures for graph *G*.

    Measures
    --------
    in_strength, out_strength                 — sum of in/out edge weights
    in_degree_count, out_degree_count         — unweighted in/out degree
    in_degree_centrality_weighted             — in_strength / (n−1)
    out_degree_centrality_weighted            — out_strength / (n−1)
    degree_centrality_weighted                — (in+out) / (n−1)
    degree_centrality                         — unweighted (NetworkX)
    betweenness_weighted                      — using ``length_attr``
    betweenness_unweighted

    Parameters
    ----------
    G : nx.DiGraph or nx.Graph
    meta_df, world_gdf : optional, used for country-name lookup
    weight_attr : str  — edge attribute for strength calculations
    length_attr : str  — edge attribute for betweenness (inverse weight)

    Returns
    -------
    pd.DataFrame  indexed by node label (iso3)
    """
    if G is None:
        raise ValueError("Graph G must be provided.")

    nodes = list(G.nodes)
    n = len(nodes)

    if G.is_directed():
        in_strength  = dict(G.in_degree(weight=weight_attr))
        out_strength = dict(G.out_degree(weight=weight_attr))
        in_count     = dict(G.in_degree(weight=None))
        out_count    = dict(G.out_degree(weight=None))
    else:
        deg          = dict(G.degree(weight=weight_attr))
        in_strength  = out_strength = deg
        in_count     = out_count = dict(G.degree(weight=None))

    deg_cent = nx.degree_centrality(G)
    denom    = float(n - 1) if n > 1 else 1.0

    in_cent_w  = {k: v / denom for k, v in in_strength.items()}
    out_cent_w = {k: v / denom for k, v in out_strength.items()}
    deg_cent_w = {k: (in_strength.get(k, 0.0) + out_strength.get(k, 0.0)) / denom
                  for k in nodes}

    try:
        bet_w = nx.betweenness_centrality(G, weight=length_attr)
    except Exception:
        logger.warning("Weighted betweenness failed; falling back to unweighted.")
        bet_w = nx.betweenness_centrality(G, weight=None)
    bet_uw = nx.betweenness_centrality(G, weight=None)

    df = pd.DataFrame(index=nodes)
    df.index.name = "iso3"
    df["in_strength"]                   = pd.Series(in_strength)
    df["out_strength"]                  = pd.Series(out_strength)
    df["in_degree_count"]               = pd.Series(in_count)
    df["out_degree_count"]              = pd.Series(out_count)
    df["in_degree_centrality_weighted"] = pd.Series(in_cent_w)
    df["out_degree_centrality_weighted"]= pd.Series(out_cent_w)
    df["degree_centrality_weighted"]    = pd.Series(deg_cent_w)
    df["degree_centrality"]             = pd.Series(deg_cent)
    df["betweenness_weighted"]          = pd.Series(bet_w)
    df["betweenness_unweighted"]        = pd.Series(bet_uw)

    iso_to_name = _get_iso_to_country_map(meta_df, world_gdf)
    df["country_name"] = df.index.map(lambda x: iso_to_name.get(x))

    col_order = [
        "country_name", "in_strength", "out_strength",
        "in_degree_count", "out_degree_count",
        "in_degree_centrality_weighted", "out_degree_centrality_weighted",
        "degree_centrality_weighted", "degree_centrality",
        "betweenness_weighted", "betweenness_unweighted",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    return df


def compute_metrics_for_graphs(graphs, meta_df=None, world_gdf=None):
    """Compute directed measures for each graph in *graphs*.

    Parameters
    ----------
    graphs : dict  {name: nx.Graph}
    meta_df, world_gdf : optional, forwarded to compute_directed_measures

    Returns
    -------
    dict  {name: pd.DataFrame}
    """
    return {name: compute_directed_measures(G, meta_df=meta_df, world_gdf=world_gdf)
            for name, G in graphs.items()}


def merge_metrics(results_dict, how="outer"):
    """Merge per-graph metric DataFrames into one wide DataFrame.

    Each metric column is suffixed with ``_<graph_name>``.  ``iso3`` and
    ``country_name`` remain un-suffixed (country_name is filled from the
    first non-null value across graphs).

    Parameters
    ----------
    results_dict : dict  {name: pd.DataFrame}
    how : str  merge strategy (default ``'outer'``)

    Returns
    -------
    pd.DataFrame  indexed by iso3
    """
    merged = None
    for name, df in results_dict.items():
        idx_name = df.index.name or "iso3"
        d = df.reset_index().rename(columns={idx_name: "iso3"})

        keep_cols   = [c for c in ("iso3", "country_name") if c in d.columns]
        metric_cols = [c for c in d.columns if c not in keep_cols]

        d_metrics = d[metric_cols].add_suffix(f"_{name}") if metric_cols else pd.DataFrame(index=d.index)
        d_keep    = d[keep_cols].copy() if keep_cols else pd.DataFrame(index=d.index)
        d2 = pd.concat([d_keep.reset_index(drop=True), d_metrics.reset_index(drop=True)], axis=1)

        if merged is None:
            merged = d2
        else:
            if "country_name" in merged.columns and "country_name" in d2.columns:
                merged["country_name"] = merged["country_name"].where(
                    merged["country_name"].notna(), d2["country_name"])
                d2 = d2.drop(columns=["country_name"])
            merged = merged.merge(d2, on="iso3", how=how)

    return merged.set_index("iso3") if merged is not None else pd.DataFrame()


# =============================================================================
# 4. Cross-network metrics  (stream 2: cross_network_metrics)
# =============================================================================

def get_feedforward_triples(G_atmo, G_vwt, include_weights=False):
    """Enumerate all A → B → C feed-forward triples across the two networks.

    A triple exists when there is an edge A→B in *G_atmo* **and** an edge
    B→C in *G_vwt* for the same intermediate node B.

    Parameters
    ----------
    G_atmo, G_vwt : nx.DiGraph
    include_weights : bool
        When True, append ET→P and VWT edge weights.

    Returns
    -------
    pd.DataFrame
        Columns: ``Moisture Source (A)``, ``Intermediary (B)``,
        ``Trade Destination (C)`` [, ``ET→P Weight``, ``VWT Weight``].
    """
    triples = []
    for A, B in G_atmo.edges():
        if not G_vwt.has_node(B):
            continue
        for C in G_vwt.successors(B):
            if include_weights:
                w_ab = G_atmo[A][B]["weight"]
                w_bc = G_vwt[B][C]["weight"]
                triples.append((A, B, C, w_ab, w_bc))
            else:
                triples.append((A, B, C))

    columns = ["Moisture Source (A)", "Intermediary (B)", "Trade Destination (C)"]
    if include_weights:
        columns += ["ET→P Weight", "VWT Weight"]

    return pd.DataFrame(triples, columns=columns)


def enrich_triples_with_precip(triples, meta_df):
    """Annotate each triple with the mediator B's total precipitation (m³).

    Parameters
    ----------
    triples : pd.DataFrame   — output of get_feedforward_triples
    meta_df : pd.DataFrame   — must contain columns ``iso3`` and ``p_total_m3``

    Returns
    -------
    pd.DataFrame  with additional column ``P_Total_B``
    """
    enriched_df = triples.copy()
    precip_map  = meta_df.set_index("iso3")["p_total_m3"].to_dict()
    enriched_df["P_Total_B"] = enriched_df["Intermediary (B)"].map(precip_map)

    missing = enriched_df[enriched_df["P_Total_B"].isna()]["Intermediary (B)"].unique()
    if len(missing) > 0:
        logger.warning("No precipitation data for %d mediators: %s", len(missing), missing)

    return enriched_df


def calculate_volumetric_attribution(enriched_df, weight_cols=("ET→P Weight", "VWT Weight")):
    """Apply physical attribution: contribution = (ET→P / P_Total_B) × VWT (m³).

    Rows where contribution ≤ 0 (or NaN) are dropped.

    Parameters
    ----------
    enriched_df : pd.DataFrame   — output of enrich_triples_with_precip
    weight_cols : tuple of str   — (atmo_weight_col, trade_weight_col)

    Returns
    -------
    pd.DataFrame
    """
    df = enriched_df.copy()
    df["P_Total_B"] = df["P_Total_B"].replace(0, np.nan)
    df["contribution"] = (df[weight_cols[0]] / df["P_Total_B"]) * df[weight_cols[1]]
    df["contribution"] = df["contribution"].fillna(0)
    return df[df["contribution"] > 0].reset_index(drop=True)


def atmo_to_vwt_dependency(prepared_triples, return_matrix=False):
    """Compute VWT dependency using attributed volumetric flows (m³).

    dependency = Σ moisture-attributed imports / Σ total VWT imports

    Parameters
    ----------
    prepared_triples : pd.DataFrame   — output of calculate_volumetric_attribution
    return_matrix : bool
        When True, also return the A → C influence matrix.

    Returns
    -------
    dict with key ``'country_dependency'`` (pd.Series) and optionally
    ``'influence_matrix'`` (pd.DataFrame).
    """
    total_moisture_tracked = prepared_triples.groupby(
        "Trade Destination (C)")["contribution"].sum()

    total_vwt = (
        prepared_triples
        .groupby(["Intermediary (B)", "Trade Destination (C)"])["VWT Weight"]
        .first()
        .groupby("Trade Destination (C)")
        .sum()
    )

    total_vwt  = total_vwt.reindex(total_moisture_tracked.index).fillna(0)
    dependency = total_moisture_tracked / total_vwt.clip(lower=1e-12)

    result = {"country_dependency": dependency}

    if return_matrix:
        influence = (
            prepared_triples
            .groupby(["Moisture Source (A)", "Trade Destination (C)"])["contribution"]
            .sum()
            .unstack(fill_value=0)
        )
        result["influence_matrix"] = influence

    return result


def atmo_vwt_asymmetry(prepared_triples, threshold=None):
    """Compute the moisture provider / trade receiver asymmetry index.
    asymmetry = (import_vol − export_vol) / (import_vol + export_vol)
    
    Parameters
    ----------
    prepared_triples : pd.DataFrame   — output of calculate_volumetric_attribution
    threshold : float or None, default 0.05
        Fraction of the median total participation below which countries are
        masked as NaN. Pass None to skip thresholding entirely.
    Returns
    -------
    pd.Series  indexed by country iso3
    """
    import_vol = prepared_triples.groupby("Trade Destination (C)")["contribution"].sum()
    export_vol = prepared_triples.groupby("Moisture Source (A)")["contribution"].sum()
    countries  = import_vol.index.union(export_vol.index)
    import_vol = import_vol.reindex(countries)
    export_vol = export_vol.reindex(countries)
    total      = import_vol + export_vol
    asym       = (import_vol - export_vol) / total

    if threshold is not None:
        mask = total > threshold * total.median()
        asym = asym.where(mask)

    return pd.Series(asym, index=countries)


def build_tripartite_graph(prepared_triples):
    """Construct a directed tripartite graph A → B → C from attributed flows.

    Edge weights use volumetric ``contribution`` (m³); ``length`` is added
    as 1/weight for shortest-path metrics.

    Parameters
    ----------
    prepared_triples : pd.DataFrame   — output of calculate_volumetric_attribution

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()

    atmo_edges = (
        prepared_triples
        .groupby(["Moisture Source (A)", "Intermediary (B)"])["contribution"]
        .sum().reset_index()
    )
    for _, row in atmo_edges.iterrows():
        w = row["contribution"]
        if w > 0:
            G.add_edge(row["Moisture Source (A)"], row["Intermediary (B)"],
                       weight=w, length=1.0 / (w + 1e-12))

    trade_edges = (
        prepared_triples
        .groupby(["Intermediary (B)", "Trade Destination (C)"])["contribution"]
        .sum().reset_index()
    )
    for _, row in trade_edges.iterrows():
        w = row["contribution"]
        if w > 0:
            G.add_edge(row["Intermediary (B)"], row["Trade Destination (C)"],
                       weight=w, length=1.0 / (w + 1e-12))

    return G


def calculate_mediator_betweenness(G, mediator_list, normalized=True):
    """Betweenness centrality for mediator nodes in the tripartite graph.

    Uses the ``'length'`` (inverse weight) attribute so high-volume paths
    count as "shortest".

    Parameters
    ----------
    G : nx.DiGraph              — output of build_tripartite_graph
    mediator_list : iterable    — nodes to report scores for
    normalized : bool

    Returns
    -------
    pd.DataFrame  columns: [``mediator_betweenness``], index name: ``iso3``
    """
    bet_all = nx.betweenness_centrality(G, weight="length", normalized=normalized)
    scores  = {node: bet_all.get(node, 0.0) for node in mediator_list}

    df = pd.DataFrame.from_dict(scores, orient="index", columns=["mediator_betweenness"])
    df.index.name = "iso3"
    return df.sort_values("mediator_betweenness", ascending=False)


# =============================================================================
# 5. Knockout effects  (stream 3: knockout_effect)
# =============================================================================

def calculate_triple_knockout(prepared_triples):
    """Simulate the removal of each mediator B and quantify systemic impact.

    Returns two DataFrames:

    ``per_mediator``
        Global metrics per mediator: total flow loss, volume share,
        mean destination impact, triple participation count.

    ``per_destination``
        Per mediator × destination: KO effect ratio and absolute volume loss.

    Parameters
    ----------
    prepared_triples : pd.DataFrame   — output of calculate_volumetric_attribution

    Returns
    -------
    dict  {'per_mediator': pd.DataFrame, 'per_destination': pd.DataFrame}
    """
    baseline_per_dest  = prepared_triples.groupby(
        "Trade Destination (C)")["contribution"].sum()
    total_system_flow  = baseline_per_dest.sum()
    mediators          = prepared_triples["Intermediary (B)"].unique()
    participation_count = (
        prepared_triples.groupby("Intermediary (B)").size()
        .rename("Triple_Participation_Count")
    )

    per_dest_records    = []
    per_mediator_global = []

    iterator = (tqdm(mediators, desc="Simulating Mediator Knockouts")
                if _HAS_TQDM else mediators)

    for B in iterator:
        through_B       = prepared_triples[prepared_triples["Intermediary (B)"] == B]
        total_flow_loss = through_B["contribution"].sum()
        dest_impacts    = through_B.groupby("Trade Destination (C)")["contribution"].sum()

        for dest, vol_loss in dest_impacts.items():
            per_dest_records.append({
                "Mediator":        B,
                "Destination":     dest,
                "KO_effect":       vol_loss / baseline_per_dest[dest],
                "Volume_Loss_m3":  vol_loss,
            })

        global_KO_volume_share = (total_flow_loss / total_system_flow
                                  if total_system_flow > 0 else 0.0)
        ko_series              = (dest_impacts / baseline_per_dest).fillna(0)
        global_KO_mean_dest    = ko_series.mean()

        per_mediator_global.append({
            "Mediator":                   B,
            "Total_Flow_Loss_m3":         total_flow_loss,
            "Global_KO_volume_share":     global_KO_volume_share,
            "Global_KO_mean_dest":        global_KO_mean_dest,
            "Triple_Participation_Count": participation_count.get(B, 0),
        })

    return {
        "per_mediator":    pd.DataFrame(per_mediator_global),
        "per_destination": pd.DataFrame(per_dest_records),
    }


def count_mediators_per_destination(triples, weight_cols=("ET→P Weight", "VWT Weight")):
    """Count how many distinct mediators B supply each destination C.

    Only A→B→C triples where both weights > 0 are counted (when the weight
    columns are present).

    Parameters
    ----------
    triples : pd.DataFrame
    weight_cols : tuple of str

    Returns
    -------
    pd.DataFrame  columns: [``Destination``, ``Num_Mediators_to_Destination``]
    """
    if triples is None or triples.empty:
        return pd.DataFrame(columns=["Destination", "Num_Mediators_to_Destination"])

    key_cols = ["Moisture Source (A)", "Intermediary (B)", "Trade Destination (C)"]
    if not all(c in triples.columns for c in key_cols):
        raise ValueError(f"triples must contain columns: {key_cols}")

    valid = (
        triples[(triples[weight_cols[0]] > 0) & (triples[weight_cols[1]] > 0)]
        if all(c in triples.columns for c in weight_cols) else triples
    )

    receivers = (
        valid[key_cols].drop_duplicates()
        .groupby("Trade Destination (C)")["Intermediary (B)"].nunique()
        .rename("Num_Mediators_to_Destination")
        .reset_index()
        .rename(columns={"Trade Destination (C)": "Destination"})
    )
    return receivers


# =============================================================================
# 6. Results I/O helpers
# =============================================================================

def write_knockout_results_to_excel(results_dict, filename, metadata=None):
    """Write knockout results to an Excel workbook (one sheet per DataFrame).

    Parameters
    ----------
    results_dict : dict  {sheet_name: pd.DataFrame}
    filename : str or Path
    metadata : dict, optional
        Written to a ``'metadata'`` sheet as Parameter / Value pairs.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        for key, df in results_dict.items():
            if df is None:
                continue
            df.to_excel(writer, sheet_name=key, index=False)

        if metadata:
            meta_out = pd.DataFrame(list(metadata.items()), columns=["Parameter", "Value"])
            meta_out.to_excel(writer, sheet_name="metadata", index=False)

    logger.info("Knockout results written to: %s", filename)


def write_cross_network_results_to_excel(
    dependency_results, asymmetry_index, mediator_importance, filename
):
    """Write cross-network results to a multi-sheet Excel workbook.

    Sheets
    ------
    country_dependency, influence_matrix, asymmetry_index, mediator_betweenness

    Parameters
    ----------
    dependency_results : dict   — output of atmo_to_vwt_dependency(return_matrix=True)
    asymmetry_index    : pd.Series
    mediator_importance: pd.DataFrame
    filename           : str or Path
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        dep = dependency_results["country_dependency"].copy()
        dep.name = "dependency"
        dep.index.name = "country"
        dep.to_frame().to_excel(writer, sheet_name="country_dependency",
                                index_label="country")

        infl = dependency_results["influence_matrix"].copy()
        infl.to_excel(writer, sheet_name="influence_matrix", index_label="country")

        asym = asymmetry_index.copy()
        asym.name = "asymmetry_index"
        asym.index.name = "country"
        asym.to_frame().to_excel(writer, sheet_name="asymmetry_index",
                                 index_label="country")

        mediator_importance.to_excel(writer, sheet_name="mediator_betweenness",
                                     index_label="country")

    logger.info("Cross-network results written to: %s", filename)
