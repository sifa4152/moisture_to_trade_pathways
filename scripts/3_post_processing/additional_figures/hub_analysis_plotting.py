#!/usr/bin/env python3
"""
Create hub analysis comprehensive figure showing:
- Top countries by degree centrality (hubs)
- Top countries by betweenness centrality (intermediaries)
- In-degree vs out-degree comparison
- Network participation distribution


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

from matplotlib.colors import LinearSegmentedColormap

_atmos = LinearSegmentedColormap.from_list('atmos_teal',  ["#013636", "#8FFAF5"])  # deep teal → light teal
_vwt   = LinearSegmentedColormap.from_list('vwt_magenta', ["#300130", "#F8B2F8"])  # dark magenta → light magenta

# Accent colours for in/out grouped bars (dark = in, light = out)
_IN_ATMOS  = "#013636"
_OUT_ATMOS = '#74F1EB'
_IN_VWT    = "#350035"
_OUT_VWT   = '#F3A5F3'

ADD_FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")

df = pd.read_csv(NETWORK_METRICS_CSV)

# Filter active countries
df_atmos = df[df['degree_centrality_atmos'] > 0].copy()
df_vwt   = df[df['degree_centrality_vwt']   > 0].copy()

print(f"Atmospheric network: {len(df_atmos)} countries")
print(f"Virtual water trade network: {len(df_vwt)} countries")

# Define participation categories
def categorize_participation(row):
    atmos_active = row['degree_centrality_atmos'] > 0
    vwt_active   = row['degree_centrality_vwt']   > 0
    if atmos_active and vwt_active:
        return 'Both Networks'
    elif atmos_active:
        return 'Atmospheric Only'
    elif vwt_active:
        return 'Virtual Water Trade Only'
    else:
        return 'Neither'

df['network_participation'] = df.apply(categorize_participation, axis=1)

print("\nNetwork participation:")
print(df['network_participation'].value_counts())

# ============================================================================
# CREATE FIGURE
# ============================================================================
print("\nCreating hub analysis figure...")

fig = plt.figure(figsize=(20, 12))
gs  = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.15)

# ============================================================================
# TOP ROW — ATMOSPHERIC NETWORK
# ============================================================================

# Top hubs by degree centrality – Atmospheric
ax1 = fig.add_subplot(gs[0, :2])
top_atmos_degree = df_atmos.nlargest(15, 'degree_centrality_weighted_atmos')
ax1.barh(range(len(top_atmos_degree)),
         top_atmos_degree['degree_centrality_weighted_atmos'],
         color=_atmos(np.linspace(0.15, 0.85, len(top_atmos_degree))))
ax1.set_yticks(range(len(top_atmos_degree)))
ax1.set_yticklabels(top_atmos_degree['iso3'])
ax1.set_xlabel('Weighted Degree Centrality', fontsize=11)
ax1.set_title('a) Top 15 hubs – AMF network (degree centrality)',
              fontsize=12, weight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Top hubs by betweenness – Atmospheric
ax2 = fig.add_subplot(gs[0, 2])
top_atmos_between = df_atmos.nlargest(15, 'betweenness_weighted_atmos')
ax2.barh(range(len(top_atmos_between)),
         top_atmos_between['betweenness_weighted_atmos'],
         color=_atmos(np.linspace(0.15, 0.85, len(top_atmos_between))))
ax2.set_yticks(range(len(top_atmos_between)))
ax2.set_yticklabels(top_atmos_between['iso3'])
ax2.set_xlabel('Betweenness', fontsize=10)
ax2.set_title('b) Top 15 mediators (betweenness)', fontsize=11, weight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# ============================================================================
# MIDDLE ROW — VIRTUAL WATER TRADE NETWORK
# ============================================================================

# Top hubs by degree centrality – VWT
ax3 = fig.add_subplot(gs[1, :2])
top_vwt_degree = df_vwt.nlargest(15, 'degree_centrality_weighted_vwt')
ax3.barh(range(len(top_vwt_degree)),
         top_vwt_degree['degree_centrality_weighted_vwt'],
         color=_vwt(np.linspace(0.15, 0.85, len(top_vwt_degree))))
ax3.set_yticks(range(len(top_vwt_degree)))
ax3.set_yticklabels(top_vwt_degree['iso3'])
ax3.set_xlabel('Weighted Degree Centrality', fontsize=11)
ax3.set_title('c) Top 15 hubs – VWT network (degree centrality)',
              fontsize=12, weight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# Top hubs by betweenness – VWT
ax4 = fig.add_subplot(gs[1, 2])
top_vwt_between = df_vwt[df_vwt['betweenness_weighted_vwt'] > 0].nlargest(15, 'betweenness_weighted_vwt')
if len(top_vwt_between) > 0:
    ax4.barh(range(len(top_vwt_between)),
             top_vwt_between['betweenness_weighted_vwt'],
             color=_vwt(np.linspace(0.15, 0.85, len(top_vwt_between))))
    ax4.set_yticks(range(len(top_vwt_between)))
    ax4.set_yticklabels(top_vwt_between['iso3'])
ax4.set_xlabel('Betweenness', fontsize=10)
ax4.set_title('d) Top 15 mediators (betweenness)', fontsize=11, weight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# ============================================================================
# BOTTOM ROW — IN/OUT DEGREE COMPARISON AND PARTICIPATION
# ============================================================================

width = 0.35

# In/Out degree comparison – Atmospheric
ax5 = fig.add_subplot(gs[2, 0])
top_combined = df_atmos.nlargest(15, 'degree_centrality_weighted_atmos')
x = np.arange(len(top_combined))
ax5.barh(x - width/2, top_combined['in_degree_count_atmos'],  width, label='In-degree',  color=_IN_ATMOS,  alpha=0.85)
ax5.barh(x + width/2, top_combined['out_degree_count_atmos'], width, label='Out-degree', color=_OUT_ATMOS, alpha=0.85)
ax5.set_yticks(x)
ax5.set_yticklabels(top_combined['iso3'])
ax5.set_xlabel('Degree Count', fontsize=10)
ax5.set_title('e) AMF: In vs out degree', fontsize=11, weight='bold')
ax5.legend(fontsize=9)
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)

# In/Out degree comparison – VWT
ax6 = fig.add_subplot(gs[2, 1])
top_combined_vwt = df_vwt.nlargest(15, 'degree_centrality_weighted_vwt')
x = np.arange(len(top_combined_vwt))
ax6.barh(x - width/2, top_combined_vwt['in_degree_count_vwt'],  width, label='In-degree',  color=_IN_VWT,  alpha=0.85)
ax6.barh(x + width/2, top_combined_vwt['out_degree_count_vwt'], width, label='Out-degree', color=_OUT_VWT, alpha=0.85)
ax6.set_yticks(x)
ax6.set_yticklabels(top_combined_vwt['iso3'])
ax6.set_xlabel('Degree Count', fontsize=10)
ax6.set_title('f) VWT: In vs out degree', fontsize=11, weight='bold')
ax6.legend(fontsize=9)
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3)

# Network participation distribution (pie chart)
ax7 = fig.add_subplot(gs[2, 2])
categories = ['Both\nnetworks', 'AMF\nonly', 'VWT\nonly', 'Neither']
counts = [
    (df['network_participation'] == 'Both Networks').sum(),
    (df['network_participation'] == 'Atmospheric Only').sum(),
    (df['network_participation'] == 'Virtual Water Trade Only').sum(),
    (df['network_participation'] == 'Neither').sum(),
]
# colors_pie = ['#8B4789', '#E74C3C', '#3498DB', '#ECF0F1']
colors_pie = [
    '#6C5CE7',  # Both
    "#013636",  # AMF only
    "#350035",  # VWT only
    "#727272"   # Neither
]
ax7.pie(counts, labels=categories, colors=colors_pie,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10})
ax7.set_title('g) Network participation distribution', fontsize=11, weight='bold')

# ============================================================================
# FINALIZE AND SAVE
# ============================================================================
plt.tight_layout(h_pad=1.5, w_pad=1.5)
plt.show()
pf.save_fig(fig, "hub_analysis", folder=ADD_FIGS_DIR, dpi=SAVE_DPI)
plt.close()