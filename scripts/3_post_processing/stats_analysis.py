# =============================================================================
# stats_analysis.py
# script for simple statistical analyses related to the paper's findings
# simonfa, March 2026
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import plot_settings as ps
import plot_functions as pf

# -------------------------------------------------------------------------
# Wealth vs. Structural Position
# -------------------------------------------------------------------------
def asym_gdp_stats():
    # LOAD DATA
    meta_df = pd.read_csv(ps.META_PATH).set_index("iso3")
    cross_sheets = pd.read_excel(ps.CROSS_NETWORK_XLSX, sheet_name=None, engine="openpyxl")
    ko_sheets = pd.read_excel(ps.KNOCKOUT_XLSX, sheet_name=None, engine="openpyxl")
    
    # Extract relevant dataframes
    asymmetry_df = cross_sheets["asymmetry_index"].set_index("country")

    # Merge Asymmetry and GDP
    df_wealth = asymmetry_df.join(meta_df[["gdp_mean_2008_2017_per_cap$"]], how="inner").dropna()
    df_wealth.rename(columns={"gdp_mean_2008_2017_per_cap$": "gdp"}, inplace=True)

    # Pearson Correlation (using log10 for GDP as it is skewed)
    r, p = stats.pearsonr(np.log10(df_wealth["gdp"]), df_wealth["asymmetry_index"])
    print(f"Correlation (log GDP vs asymmetry): Pearson r = {r:.2f}, p = {p:.4f}")

    # Define High and Low income based on World Bank thresholds from plot_settings
    # thresholds are defined in plot_settings:
    low_threshold = ps.WB_THRESHOLDS[0]
    high_threshold = ps.WB_THRESHOLDS[2]

    low_inc_df = df_wealth[df_wealth["gdp"] <= low_threshold]
    high_inc_df = df_wealth[df_wealth["gdp"] > high_threshold]

    # Calculate percentages
    # Net-importers have asymmetry > 0
    pct_high_importers = (high_inc_df["asymmetry_index"] > 0).mean() * 100
    # Net-exporters have asymmetry < 0
    pct_low_exporters = (low_inc_df["asymmetry_index"] < 0).mean() * 100

    print(f"High-Income (GDP > ${high_threshold}): {pct_high_importers:.1f}% are net-importers (positive side)")
    print(f"Low-Income (GDP <= ${low_threshold}): {pct_low_exporters:.1f}% are net-exporters (negative side)")
    print(f"(Sample sizes: High={len(high_inc_df)}, Low={len(low_inc_df)})")


# -------------------------------------------------------------------------
# Exposure and Redundancy
# -------------------------------------------------------------------------
def exposure_stats():
    # LOAD DATA
    cross_sheets = pd.read_excel(ps.CROSS_NETWORK_XLSX, sheet_name=None, engine="openpyxl")
    ko_sheets = pd.read_excel(ps.KNOCKOUT_XLSX, sheet_name=None, engine="openpyxl")
    
    # Extract relevant dataframes
    dependency_df = cross_sheets["country_dependency"].set_index("country")
    mediators_df = ko_sheets["mediators_per_destination"]  # Column: Num_Mediators_to_Destination

    # Global Exposure (Dependency)
    mean_dep = dependency_df["dependency"].mean()
    median_dep = dependency_df["dependency"].median()
    min_dep = dependency_df["dependency"].min()
    max_dep = dependency_df["dependency"].max()

    print(f"Exposure/Dependency: Mean = {mean_dep:.2f}, Median = {median_dep:.2f}, Range = [{min_dep:.2f}, {max_dep:.2f}]")

    # Path Redundancy (Intermediaries)
    # Column: Num_Mediators_to_Destination
    col_med = "Num_Mediators_to_Destination"
    
    count_1 = (mediators_df[col_med] <= 1).sum()
    count_5 = (mediators_df[col_med] <= 5).sum()

    print(f"Countries reached through a single intermediary (<=1): {count_1}")
    print(f"Countries reached through five or fewer intermediaries (<=5): {count_5}")


# -------------------------------------------------------------------------
# Sensitivity Analysis Plotting
# -------------------------------------------------------------------------
def plot_sensitivity(sensitivity_csv=None, save=ps.FIGS_DIR):
    """Read sensitivity_analysis.csv and plot threshold sensitivity."""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(sensitivity_csv or ps.RESULTS_DIR / "sensitivity_analysis.csv",
                     index_col="threshold")

    # Setting a clean style
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="white", constrained_layout=True)
    
    # Define a consistent color palette
    colors = {"atmos": "#0E3350", "vwt": "#4b0607", "triples": "#444343", "baseline": "#7d15a7"}

    # --- Panel 1: Structural (Dual Axis) ---
    ax1 = axes[0]
    ln1 = ax1.plot(df.index, df["n_edges_atmos"], "o-", color=colors["atmos"], label="Atmospheric edges", markersize=5, linewidth=1.5)
    ln2 = ax1.plot(df.index, df["n_edges_vwt"],   "s-", color=colors["vwt"],   label="VWT edges",   markersize=5, linewidth=1.5)
    
    ax1_twin = ax1.twinx()
    ln3 = ax1_twin.plot(df.index, df["n_triples"], "^--", color=colors["triples"], label="Triples", alpha=0.7)
    
    # Unified Legend for twin axes
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper right", frameon=False, fontsize=9)
    
    # Aesthetic tweaks for P1
    ax1.set_title("(a) Graph connectivity", loc='left', fontweight='bold', pad=10)
    ax1.set_xlabel("Threshold percentile")
    ax1.set_ylabel("Edge count")
    ax1_twin.set_ylabel("Triple count", color=colors["triples"])
    ax1_twin.tick_params(axis='y', labelcolor=colors["triples"])
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    # threshold_values = df.index.tolist()
    # ax1.set_xticks(threshold_values)
    
    # --- Panel 2: Dependency and Asymmetry ---
    ax2 = axes[1]
    ax2.plot(df.index, df["dep_mean"],   "o-",  color="#09491d", label="mean exposure", markersize=3)
    ax2.plot(df.index, df["dep_median"], "o--", color="#74a16f", label="median exposure", markersize=3)
    ax2.plot(df.index, df["asym_mean"],  "D-",  color="#c06e02", label="mean asymmetry", markersize=3)
    ax2.plot(df.index, df["asym_median"], "D--", color="#e9a113", label="median asymmetry", markersize=3)
    
    # Highlight Baseline
    ax2.axvline(90, color=colors["baseline"], linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.text(91, ax2.get_ylim()[0], "Baseline (90)", color=colors["baseline"], verticalalignment='bottom', fontsize=9)

    ax2.set_title("(b) Cross-network metrics", loc='left', fontweight='bold', pad=10)
    ax2.set_xlabel("Threshold percentile")
    ax2.set_ylabel("Metric value")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.spines[['top', 'right']].set_visible(False)
    # ax2.set_xticks(threshold_values)
    
    # Panel 3: top-5 mediator stability (as a text table)
    ax3 = axes[2]
    ax3.axis("off")
    tbl = ax3.table(
        cellText=[[str(t), m] for t, m in df["topn_mediators"].items()],
        colLabels=["Threshold [%]", "Top mediators"],
        loc="center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    ax3.set_title("(c) Mediator rank stability", loc='left', fontweight='bold', pad=10)
    
    # Make header bold and color background and set column widths
    col_widths = [0.22, 0.78]  # adjust to taste, must sum to ~1
    for (row, col), cell in tbl.get_celld().items():
        cell.set_width(col_widths[col])
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#404040')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2') # Zebra striping
            
    if save:
        pf.save_fig(fig, "sensitivity_analysis", folder=save, dpi=450)
    return fig



# runner function
def run_targeted_stats():
    print("\n=== ASYMMETRY AND GDP STATISTICS ===")
    asym_gdp_stats()
    
    print("\n=== EXPOSURE AND REDUNDANCY STATISTICS ===")
    exposure_stats()

    print("\n=== SENSITIVITY ANALYSIS ===")
    plot_sensitivity()
    

if __name__ == "__main__":
    run_targeted_stats()
