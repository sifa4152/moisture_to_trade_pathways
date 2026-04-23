"""
Network Data Preprocessing Pipeline
====================================
Preprocesses VWT and atmospheric moisture networks into a common ISO3-labelled
format for multiplex network analysis.
 
Inputs:
  - VWT network:         .mat file (3D array: countries × countries × years)
  - Atmospheric network: CSV with ISO2 row/column labels (may include ocean regions)
  - Country list:        Excel file mapping VWT row order → country name / FAO code
  - Metadata:            CSV with ISO2, ISO3, and id columns for all countries
 
Outputs (written to output_dir):
  - vwt_network_reformatted.csv       — VWT matrix, ISO3-labelled
  - atmos_network_reformatted.csv     — Atmospheric matrix, ISO3-labelled
  - network_index_mapping.csv         — ISO3 ↔ numeric index lookup table
"""
 
import logging
from pathlib import Path
from typing import Tuple, List
 
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
 
# Manual name → ISO3 mappings for FAO country names that differ from metadata
MANUAL_NAME_TO_ISO3 = {
    "bolivia, plurinational state of": "BOL",
    "brunei darussalam": "BRN",
    "central african republic": "CAF",
    "china, hong kong sar": "HKG",
    "china, macao sar": "MAC",
    "china, mainland": "CHN",
    "china, taiwan province of": "TWN",
    "cocos islands (keeling)": "CCK",
    "congo, democratic republic of the": "COD",
    "cote de ivoire": "CIV",
    "czech republic": "CZE",
    "falkland islands (malvinas)": "FLK",
    "faroe islands": "FRO",
    "french southern and antarctic territories": "ATF",
    "gambia": "GMB",
    "iran, islamic republic of": "IRN",
    "korea, democratic people's republic of": "PRK",
    "korea, republic of": "KOR",
    "lao people's democratic republic": "LAO",
    "libyan arab jamahiriya": "LBY",
    "micronesia, federated states of": "FSM",
    "moldova, republic of": "MDA",
    "netherlands antilles": "ANT",
    "palestinian territory, occupied": "PSE",
    "russian federation": "RUS",
    "saint helena, ascension and tristan da cunha": "SHN",
    "saint kitts and nevis": "KNA",
    "saint lucia": "LCA",
    "saint pierre and miquelon": "SPM",
    "saint vincent and the grenadines": "VCT",
    "sao tome and principe": "STP",
    "serbia and montenegro": "SCG",
    "syrian arab republic": "SYR",
    "tanzania, united republic of": "TZA",
    "the former yugoslav republic of macedonia": "MKD",
    "timor-leste": "TLS",
    "united kingdom": "GBR",
    "united states of america": "USA",
    "venezuela, bolivarian republic of": "VEN",
    "viet nam": "VNM",
    "virgin islands, british": "VGB",
    "virgin islands, u.s.": "VIR",
    "wallis and futuna islands": "WLF",
}
 
 
class NetworkPreprocessor:
 
    def __init__(self, metadata_path: str):
        """Load metadata and build ISO2 → ISO3 lookup."""
        self.metadata = pd.read_csv(metadata_path, keep_default_na=False, na_values=[""])
        self.iso2_to_iso3 = dict(zip(self.metadata["iso"], self.metadata["iso3"]))
        self.name_to_iso3 = dict(zip(self.metadata["name"].str.lower().str.strip(),
                                     self.metadata["iso3"]))
        logger.info(f"Loaded metadata: {len(self.metadata)} countries")
 
    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
 
    def _load_mat_array(self, mat_path: str) -> np.ndarray:
        """Extract the primary data array from a .mat file."""
        mat = sio.loadmat(mat_path)
        if "all_crop" in mat:
            return mat["all_crop"]
        data_keys = [k for k in mat if not k.startswith("__")]
        if len(data_keys) == 1:
            return mat[data_keys[0]]
        raise ValueError(f"Cannot identify data array in .mat file. Keys: {data_keys}")
 
    def load_vwt(self, mat_path: str, country_list_path: str,
                 year_start: int = 2008, year_end: int = 2016) -> pd.DataFrame:
        """
        Load VWT network from a .mat file, average over years, and label with ISO3 codes.
 
        The .mat file is assumed to contain a 3D array (countries × countries × years)
        starting from base_year=1986.
        """
        logger.info(f"\n ======= VWT data processing ==================================")
        vwt_data = self._load_mat_array(mat_path)
 
        base_year = 1986
        start_idx = year_start - base_year
        end_idx   = year_end   - base_year + 1
        vwt_mean  = vwt_data[:, :, start_idx:end_idx].mean(axis=2)
        logger.info(f"VWT: averaged years {year_start}–{year_end}, shape {vwt_mean.shape}")
 
        country_list = pd.read_excel(country_list_path)
        country_list["Country name"] = country_list["Country name"].str.strip("'")
 
        iso3_codes = []
        for name in country_list["Country name"]:
            key = name.lower().strip()
            iso3 = self.name_to_iso3.get(key) or MANUAL_NAME_TO_ISO3.get(key)
            if iso3 is None:
                logger.debug(f"No ISO3 match for VWT country: '{name}'")
            iso3_codes.append(iso3)
 
        df = pd.DataFrame(vwt_mean, index=iso3_codes, columns=iso3_codes)
        # Drop rows/columns with no ISO3 match
        df = df.loc[df.index.notna(), df.columns.notna()]
        logger.info(f"VWT: {len(df)} countries matched to ISO3")
        return df
 
    def load_atmospheric(self, csv_path: str) -> pd.DataFrame:
        """
        Load atmospheric network from CSV, drop ocean regions, and convert to ISO3 labels.
 
        Ocean regions are identified by column names starting with 'OC_'.
        ISO2 codes not found in metadata are kept as-is with a warning.
        """
        logger.info(f"\n ======= Atmospheric data processing ==================================")
        df = pd.read_csv(csv_path, index_col=0, keep_default_na=False, na_values=[""])
        logger.info(f"Atmospheric: raw shape {df.shape}")
 
        # Drop ocean columns/rows
        country_codes = [c for c in df.columns if not c.startswith("OC_")]
        country_codes = [c for c in country_codes if c in df.index]
        df = df.loc[country_codes, country_codes]
 
        # Convert ISO2 → ISO3
        missing = [c for c in df.columns if c not in self.iso2_to_iso3]
        if missing:
            logger.warning(f"No ISO3 mapping for {len(missing)} codes")
        iso3_codes = [self.iso2_to_iso3.get(c, c) for c in df.columns]
        df.index   = iso3_codes
        df.columns = iso3_codes
        logger.info(f"Atmospheric: {len(df)} countries after filtering")
        return df
 
    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
 
    def align_to_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reindex a network to match the full metadata country list (ISO3, sorted).
        Countries present in metadata but absent from the network are padded with zeros.
        Countries present in the network but absent from metadata are dropped.
        """
        logger.info(f"--- Aligning network to metadata country list ------------------------")
        metadata_iso3 = sorted(self.metadata["iso3"].tolist())
 
        extra   = set(df.index) - set(metadata_iso3)
        missing = set(metadata_iso3) - set(df.index)
        if extra:
            logger.info(f"Dropping {len(extra)} countries not in metadata: {extra}")
        if missing:
            logger.info(f"Padding {len(missing)} missing countries with zeros: {missing}")
 
        return df.reindex(index=metadata_iso3, columns=metadata_iso3, fill_value=0.0)
 
    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
 
    def save(self, vwt_df: pd.DataFrame, atmos_df: pd.DataFrame, output_dir: str):
        logger.info(f"\n ======= Saving processed networks ==================================")
        """Save both networks and an ISO3 ↔ numeric index mapping to output_dir."""
        assert vwt_df.shape == atmos_df.shape, \
            f"Network shapes do not match: VWT {vwt_df.shape} vs atmos {atmos_df.shape}"
        assert list(vwt_df.index) == list(atmos_df.index), \
            "Networks have different country orderings — align before saving."
 
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
 
        vwt_df.to_csv(out / "vwt_network.csv")
        atmos_df.to_csv(out / "atmos_network.csv")
 
        # Index mapping: ISO3 ↔ 0-based numeric index
        mapping = pd.DataFrame({
            "network_index": range(len(vwt_df)),
            "iso3":          vwt_df.index,
        })
        mapping.to_csv(out / "network_index_mapping.csv", index=False)
 
        logger.info(f"Saved outputs to {out}/")
        logger.info(f"  VWT:  {vwt_df.shape},  total flow: {vwt_df.sum().sum():.3e},  "
                    f"non-zero entries: {(vwt_df > 0).sum().sum()}")
        logger.info(f"  Atmos: {atmos_df.shape}, total flow: {atmos_df.sum().sum():.3e}, "
                    f"non-zero entries: {(atmos_df > 0).sum().sum()}")
 
    # ------------------------------------------------------------------
    # Comparison statistics
    # ------------------------------------------------------------------
 
    def compute_network_stats(
        self,
        atmos_df: pd.DataFrame,
        vwt_df: pd.DataFrame):
        """
        Compare summary statistics between the atmospheric and VWT networks and
        optionally produce a heatmap of element-wise flow ratios.
 
        Parameters
        ----------
        atmos_df : pd.DataFrame
            Atmospheric moisture network (ISO3-labelled, aligned).
        vwt_df : pd.DataFrame
            Virtual water trade network (ISO3-labelled, aligned).
        plot : bool
            If True, display (and optionally save) a heatmap of individual flow ratios.
        output_dir : str or None
            If provided, the heatmap is saved to this directory instead of displayed.
 
        Returns
        -------
        dict with keys: total_flow_ratio, average_flow_ratio, max_flow_ratio,
                        individual_flow_ratios (np.ndarray)
        """
        logger.info(f"\n ======= Network comparison statistics ================================")
 
        a = atmos_df.values
        v = vwt_df.values
 
        stats = {
            "total_flow_ratio":    a.sum()  / v.sum(),
            "average_flow_ratio":  a.mean() / v.mean(),
            "max_flow_ratio":      a.max()  / v.max(),
            "individual_flow_ratios": np.where(v != 0, a / v, np.nan),
        }
 
        logger.info(f"  Total flow   — atmos: {a.sum():.3e},  vwt: {v.sum():.3e},  "
                    f"ratio: {stats['total_flow_ratio']:.4f}")
        logger.info(f"  Average flow — atmos: {a.mean():.3e}, vwt: {v.mean():.3e}, "
                    f"ratio: {stats['average_flow_ratio']:.4f}")
        logger.info(f"  Max flow     — atmos: {a.max():.3e},  vwt: {v.max():.3e},  "
                    f"ratio: {stats['max_flow_ratio']:.4f}")
 
 
 
# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
 
if __name__ == "__main__":
    # --- File paths ---------------------------------------------------
    VWT_MAT_PATH      = "data/raw/networks/vwt_network_tamea2021/VWT_primarycrops&derived_216items_1986-2016.mat"
    COUNTRY_LIST_PATH = "data/raw/networks/vwt_network_tamea2021/country_list.xlsx"
    ATMOS_PATH        = "data/raw/networks/atmos_network_reconc_cntr_v3_DePetrillo2025/countries_oceans_flux_matrix_eurostat_goas_seavox.csv"
    METADATA_PATH     = "data/processed/networks/archive/networks_meta_data.csv"
    OUTPUT_DIR        = "data/processed/networks"
    # ------------------------------------------------------------------
 
    preprocessor = NetworkPreprocessor(METADATA_PATH)
 
    logger.info("Starting network preprocessing pipeline...")
 
    vwt_df   = preprocessor.load_vwt(VWT_MAT_PATH, COUNTRY_LIST_PATH)
    vwt_df   = preprocessor.align_to_metadata(vwt_df)
 
    atmos_df = preprocessor.load_atmospheric(ATMOS_PATH)
    atmos_df = preprocessor.align_to_metadata(atmos_df)
 
    preprocessor.save(vwt_df, atmos_df, OUTPUT_DIR)
 
    stats = preprocessor.compute_network_stats(atmos_df, vwt_df)
    