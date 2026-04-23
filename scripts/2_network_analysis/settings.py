# =============================================================================
# settings.py
# Central configuration for the atmospheric moisture / virtual water workflow.
# Edit the paths and parameters here; everything else reads from this file.
# =============================================================================

from pathlib import Path

# -----------------------------------------------------------------------------
# Data paths
# -----------------------------------------------------------------------------
DATA_DIR = Path("data")

ATMOS_PATH   = DATA_DIR / "processed/networks/atmos_network.csv"
VWT_PATH     = DATA_DIR / "processed/networks/vwt_network.csv"
META_PATH    = DATA_DIR / "processed/networks/networks_meta_data.csv"
WORLD_PATH   = DATA_DIR / "raw/auxiliary/CNTR_RG_20M_2024_4326.shp/CNTR_RG_20M_2024_4326_edited.shp"


# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------
date = "2026-04-15"  # Update this when you start a new run to avoid overwriting old results
OUTPUT_DIR = Path(f"output/version_{date}/results")

# Per-analysis output filenames (resolved against OUTPUT_DIR in main.py)
NETWORK_METRICS_CSV        = "network_metrics_output.csv"
CROSS_NETWORK_METRICS_XLSX = "cross_network_metrics.xlsx"
FEEDFORWARD_TRIPLES_CSV    = "feed_forward_triples.csv"
KNOCKOUT_RESULTS_XLSX      = "knockout_results.xlsx"

# Log file (set to None to disable file logging)
LOG_FILE = OUTPUT_DIR / "workflow.log"

# -----------------------------------------------------------------------------
# Graph construction parameters
# -----------------------------------------------------------------------------
# Percentile threshold for edge filtering (lower → more edges retained).
# 90 means keep the top 10 % of positive weights.
THRESHOLD_PERCENTILE = 93

# Whether to build directed graphs
DIRECTED_GRAPH = True

# Whether to add inverse-weight 'length' attribute for shortest-path metrics
INVERSE_WEIGHT = True

# -----------------------------------------------------------------------------
# Analysis switches — set any to False to skip that analysis stream
# -----------------------------------------------------------------------------
RUN_NETWORK_METRICS    = True   # per-network node-level metrics (strength, betweenness …)
RUN_CROSS_NETWORK      = True   # feed-forward triples, dependency, asymmetry, mediator betweenness
RUN_KNOCKOUT           = True   # mediator knockout simulation

# Whether to persist results to disk
SAVE_RESULTS = True

# Asymmetry threshold for masking countries in the asymmetry index (threshold * median of total flows)
ASYM_THRESHOLD = None  # set to None to disable masking

# -----------------------------------------------------------------------------
# Sensitivity analysis — set to False to skip that analysis stream
# -----------------------------------------------------------------------------
RUN_SENSITIVITY      = True   # set True to run sweep instead of single run
SENSITIVITY_THRESHOLDS = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 86, 87,
                          88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
TOP_N_MEDIATORS = 10   # Number of top mediators to report in sensitivity analysis (per threshold)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = "INFO"   # DEBUG / INFO / WARNING / ERROR
