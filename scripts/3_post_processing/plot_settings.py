# =============================================================================
# plot_settings.py
# Central configuration for the figure plotting pipeline.
# All figure scripts import from here — edit paths and style constants once.
# =============================================================================
from pathlib import Path
import sys
import matplotlib as mpl

# -----------------------------------------------------------------------------
# Resolve the workflow root and inject network_analysis/ onto the path
# -----------------------------------------------------------------------------
_POST_PROCESSING_DIR = Path(__file__).resolve().parent          # .../workflow/post_processing
_WORKFLOW_ROOT       = _POST_PROCESSING_DIR.parent.parent              # .../workflow
_NETWORK_ANALYSIS    = _WORKFLOW_ROOT / "scripts/2_network_analysis"

if str(_NETWORK_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_NETWORK_ANALYSIS))

# -----------------------------------------------------------------------------
# Import shared paths and parameters from the analysis settings
# -----------------------------------------------------------------------------
from settings import (
    OUTPUT_DIR           as RESULTS_DIR,
    META_PATH,
    WORLD_PATH,
    ATMOS_PATH,
    VWT_PATH,
    THRESHOLD_PERCENTILE,
)

# -----------------------------------------------------------------------------
# Data and output paths
# -----------------------------------------------------------------------------
DATA_DIR    = _WORKFLOW_ROOT / "data"

def _latest_version(root):
    versions = sorted((root / "output").glob("version_*"))
    if not versions:
        raise FileNotFoundError(f"No version_* folders found under {root / 'output'}")
    return versions[-1]

_VERSION_DIR = _latest_version(_WORKFLOW_ROOT)
RESULTS_DIR  = _VERSION_DIR / "results"
FIGS_DIR     = _VERSION_DIR / "figures"
ADD_FIGS_DIR = _VERSION_DIR / "additional_figures"

SANKEY_DIR   = FIGS_DIR / "sankeys"

CROSS_NETWORK_XLSX  = RESULTS_DIR / "cross_network_metrics.xlsx"
KNOCKOUT_XLSX       = RESULTS_DIR / "knockout_results.xlsx"
TRIPLES_CSV         = RESULTS_DIR / "feed_forward_triples.csv"
NETWORK_METRICS_CSV = RESULTS_DIR / "network_metrics_output.csv"

# -----------------------------------------------------------------------------
# Global matplotlib font settings (applied at import time)
# -----------------------------------------------------------------------------
mpl.rcParams["svg.fonttype"]  = "none"   # keep text editable in SVG
mpl.rcParams["pdf.fonttype"]  = 42       # embed TrueType fonts in PDF
mpl.rcParams["ps.fonttype"]   = 42
mpl.rcParams["font.family"]   = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

# -----------------------------------------------------------------------------
# Figure sizes  (width, height) in inches  —  converted from mm where needed
# -----------------------------------------------------------------------------
def mm(w, h):
    """Convert millimetres to inches for figsize."""
    return (w / 25.4, h / 25.4)

FIGSIZE_MAP_SINGLE  = mm(92,  63)   # one-panel world map
FIGSIZE_MAP_WIDE    = mm(91,  57)   # slightly wider single map (fig3 style)
FIGSIZE_CHORD       = (22, 11)      # side-by-side chord diagrams
FIGSIZE_SCATTER     = (11,  7)      # scatter / bubble plot

# -----------------------------------------------------------------------------
# Shared map layout defaults
# -----------------------------------------------------------------------------
MAP_EXTENT      = [-150, 180, -60, 90]   # [xmin, xmax, ymin, ymax]
MAP_EDGE_COLOR  = "#cccccc"
MAP_BASE_COLOR  = "#969696"
MAP_EDGE_WIDTH  = 0.1
MAP_CBAR_KWARGS = dict(orientation="horizontal", pad=0.04, shrink=0.6, aspect=35)

# -----------------------------------------------------------------------------
# Colour palettes (shared across figures)
# -----------------------------------------------------------------------------

# Asymmetry index — diverging brown–teal (used in fig3 and fig5b sankey)
ASYM_HEX = [
    "#003c30", "#01665e", "#35978f", "#80cdc1", "#c7eae5",
    "#f5f5f5",
    "#f6e8c3", "#dfc27d", "#bf812d", "#8c510a", "#543005",
]
ASYM_BOUNDS = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Domestic moisture recycling — warm yellows/greens
DMR_HEX     = ["#E6D29F", "#B69A5C", "#76741A", "#3B5700", "#2D3D0B"]
DMR_BOUNDS  = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Exposure — blue-purple
DEP_HEX     = ["#D0CAE5", "#9B98CC", "#6781C2", "#284F84", "#12194A"]
DEP_BOUNDS  = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# DEP_HEX     = ["#F6F3F7", "#D0CAE5", "#9B98CC", "#6781C2", "#284F84", "#2B3263"]
# DEP_BOUNDS  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# DEP_HEX     = ["#F6F3F7", "#DDD8EE", "#C0BAE0", "#9B98CC", "#7A8CBF", "#6781C2", "#4A6BAF", "#284F84", "#1E3068", "#12194A"]
# DEP_BOUNDS  = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Mediator count — muted purple-pink
# MED_HEX     = ["#F1E7F4", "#E8CEE5", "#E0A7C8", "#BF688F", "#96648E", "#71618B", "#402F5F"]
# MED_BOUNDS  = [0, 10, 20, 30, 40, 50, 60, 70]
MED_HEX     = ["#F1E7F4", "#E8CEE5", "#E0A7C8", "#BF688F", "#96648E", "#71618B", "#2D1245"]
MED_BOUNDS  = [0, 10, 20, 30, 40, 50, 60]

# Flow loss — warm sunset sequence
FLOW_HEX    = [
    "#FEF5DB", "#E7C7A0", "#E8A279", "#E67961",
    "#BC6562", "#91606B", "#6D5F76", "#46597B", "#12385C", "#0D1425",
]
FLOW_VMIN   = 1e7   # LogNorm lower bound

# Chord diagram
CHORD_NODE_CMAP     = "cubehelix" # for continent version (v2), cmap is defined in plotting_script
CHORD_BG_COLOR      = "#ffffff"
CHORD_ARC_LWIDTH    = 0.5
CHORD_RING_WIDTH    = 0.08
CHORD_INNER_RADIUS  = 1.0
CHORD_LABEL_OFFSET  = 0.14
CHORD_MIN_ALPHA     = 0.15
CHORD_MAX_ALPHA     = 0.75
CHORD_TOP_N_NODES   = 50
CHORD_DPI           = 450

# Sankey
SANKEY_MEDIATORS    = ["ARG", "USA", "UKR", "BEL", "CIV", "CAN", "PRY", "KAZ", "RUS", "DEU","GHA","MYS"]
SANKEY_TOP_N        = 7      # top sources / destinations shown (rest → "Others")
SANKEY_SOURCE_COLOR = "#C37572"
SANKEY_DEST_COLOR   = "#6D5F76"
SANKEY_MEDIATOR_COLOR = "#12385C"

# World Bank income thresholds (Atlas method, 2016) --> mean of 2008-2017
WB_THRESHOLDS   = [1015, 4007, 12371]  # USD per capita
WB_LABELS       = ["Low\nincome", "Lower-\nmiddle", "Upper-\nmiddle", "High\nincome"]

# Save DPI for raster exports
SAVE_DPI = 450
