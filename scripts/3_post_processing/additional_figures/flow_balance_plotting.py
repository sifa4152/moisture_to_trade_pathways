#!/usr/bin/env python3
"""
Create flow balance analysis figure with 2x2 layout:
- Top row: Net flow distributions (histograms)
- Bottom row: Top 10 net importers and exporters for each network

Uses distinct colors and original clean style.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── central config ────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_settings import NETWORK_METRICS_CSV, ADD_FIGS_DIR, SAVE_DPI
import plot_functions as pf

ADD_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")

df = pd.read_csv(NETWORK_METRICS_CSV)

# Calculate net flow (in_strength - out_strength)
df['net_flow_atmos'] = df['in_strength_atmos'] - df['out_strength_atmos']
df['net_flow_vwt']   = df['in_strength_vwt']   - df['out_strength_vwt']

print(f"Total countries: {len(df)}")
print(f"Atmospheric network participants: {(df['degree_centrality_atmos'] > 0).sum()}")
print(f"VWT network participants: {(df['degree_centrality_vwt'] > 0).sum()}")

# ============================================================================
# DEFINE DISTINCT COLORS — TEAL vs MAGENTA THEME
# ============================================================================

# Atmospheric (Teal tones)
color_atmos_import = "#013636"   # deep teal   (importer)
color_atmos_export = "#74F1EB"   # light teal  (exporter)

# VWT (Magenta tones)
color_vwt_import = "#350035"     # dark magenta (importer)
color_vwt_export = "#F3A5F3"     # light magenta (exporter)

# ============================================================================
# CREATE FIGURE — 2×2 LAYOUT
# ============================================================================
print("\nCreating flow balance analysis figure...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# ============================================================================
# TOP LEFT: ATMOSPHERIC NET FLOW DISTRIBUTION
# ============================================================================

ax1 = axes[0, 0]
net_atmos = df[df['degree_centrality_atmos'] > 0]['net_flow_atmos']
ax1.hist(net_atmos, bins=50, color="#286D6B", alpha=0.7, edgecolor='black')
ax1.axvline(0, color='k', linestyle='--', linewidth=2, label='Balance point')
ax1.set_xlabel('Net flow (in − out) [m³]', fontsize=11)
ax1.set_ylabel('Number of countries', fontsize=11)
ax1.set_title('a) AMF network – net flow distribution', fontsize=12, weight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# ============================================================================
# TOP RIGHT: VWT NET FLOW DISTRIBUTION
# ============================================================================

ax2 = axes[0, 1]
net_vwt = df[df['degree_centrality_vwt'] > 0]['net_flow_vwt']
ax2.hist(net_vwt, bins=50, color="#742B71", alpha=0.7, edgecolor='black')
ax2.axvline(0, color='k', linestyle='--', linewidth=2, label='Balance point')
ax2.set_xlabel('Net flow (in − out) [m³]', fontsize=11)
ax2.set_ylabel('Number of countries', fontsize=11)
ax2.set_title('b) VWT – net flow distribution', fontsize=12, weight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# ============================================================================
# BOTTOM LEFT: ATMOSPHERIC TOP IMPORTERS & EXPORTERS
# ============================================================================

ax3 = axes[1, 0]

df_atmos_sorted    = df[df['degree_centrality_atmos'] > 0].copy()
top_importers_atmos = df_atmos_sorted.nlargest(10,  'net_flow_atmos')
top_exporters_atmos = df_atmos_sorted.nsmallest(10, 'net_flow_atmos')

y_pos = np.arange(10)
ax3.barh(y_pos,      top_importers_atmos['net_flow_atmos'].values,
         color=color_atmos_import, alpha=0.7, label='Net importers')
ax3.barh(y_pos + 11, top_exporters_atmos['net_flow_atmos'].values,
         color=color_atmos_export, alpha=0.7, label='Net exporters')

ax3.set_yticks(list(y_pos) + list(y_pos + 11))
ax3.set_yticklabels(list(top_importers_atmos['iso3']) + list(top_exporters_atmos['iso3']))
ax3.axvline(0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Net flow [m³]', fontsize=11)
ax3.set_title('c) AMF: Top 10 net importers & exporters', fontsize=12, weight='bold')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# ============================================================================
# BOTTOM RIGHT: VWT TOP IMPORTERS & EXPORTERS
# ============================================================================

ax4 = axes[1, 1]

df_vwt_sorted    = df[df['degree_centrality_vwt'] > 0].copy()
top_importers_vwt = df_vwt_sorted.nlargest(10,  'net_flow_vwt')
top_exporters_vwt = df_vwt_sorted.nsmallest(10, 'net_flow_vwt')

y_pos = np.arange(10)
ax4.barh(y_pos,      top_importers_vwt['net_flow_vwt'].values,
         color=color_vwt_import, alpha=0.7, label='Net importers')
ax4.barh(y_pos + 11, top_exporters_vwt['net_flow_vwt'].values,
         color=color_vwt_export, alpha=0.7, label='Net exporters')

ax4.set_yticks(list(y_pos) + list(y_pos + 11))
ax4.set_yticklabels(list(top_importers_vwt['iso3']) + list(top_exporters_vwt['iso3']))
ax4.axvline(0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Net flow [m³]', fontsize=11)
ax4.set_title('d) VWT: Top 10 net importers & exporters', fontsize=12, weight='bold')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

# ============================================================================
# FINALIZE AND SAVE
# ============================================================================
plt.tight_layout(h_pad=1.5, w_pad=1.5)
plt.show()
pf.save_fig(fig, "flow_balance", folder=ADD_FIGS_DIR, dpi=SAVE_DPI)
plt.close()
