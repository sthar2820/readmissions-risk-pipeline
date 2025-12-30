"""
02_clean_join.py - Clean, merge, and engineer features for hospital + SVI data

Steps:
    1. Clean HRRP (hospital general info): normalize Facility ID and ZIP
    2. Clean Hospital General (readmission measures): pivot measures, aggregate by facility
    3. Clean SVI: extract 5-digit ZIP from LOCATION, handle -999 missing codes
    4. Merge all into a single analytics-ready table
    5. Feature engineering:
       - Outcome: err (float), high_err_flag (binary)
       - SVI features: theme scores + interpretable raw percentages
       - Risk buckets: 2x2 matrix of SVI x ERR

Output:
    - data/processed/hospital_svi_analytics.parquet
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_facility_id(df: pd.DataFrame, col: str = "Facility ID") -> pd.DataFrame:
    """Ensure Facility ID is string, zero-padded to 6 digits."""
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().str.zfill(6)
    return df


def clean_zip(df: pd.DataFrame, col: str = "ZIP Code") -> pd.DataFrame:
    """Clean ZIP: strip spaces, keep first 5 digits."""
    df = df.copy()
    df["zip5"] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str[:5]
        .str.zfill(5)
    )
    return df


def clean_hrrp(df: pd.DataFrame) -> pd.DataFrame:
    """Clean HRRP (hospital general info) data."""
    print("\n--- Cleaning HRRP (Hospital General Info) ---")
    print(f"Input shape: {df.shape}")

    df = clean_facility_id(df)
    df = clean_zip(df)

    # Drop duplicates by Facility ID (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["Facility ID"], keep="first")
    print(f"Dropped {before - len(df)} duplicate Facility IDs")

    print(f"Output shape: {df.shape}")
    return df


def clean_hospital_general(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Hospital General (HRRP readmission measures).
    Pivot measures to columns and aggregate by facility.
    """
    print("\n--- Cleaning Hospital General (Readmission Measures) ---")
    print(f"Input shape: {df.shape}")

    df = clean_facility_id(df)

    # Convert numeric columns
    numeric_cols = ["Excess Readmission Ratio", "Predicted Readmission Rate",
                    "Expected Readmission Rate", "Number of Discharges", "Number of Readmissions"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create short measure names for pivoting
    df["measure_short"] = df["Measure Name"].str.extract(r"READM-30-(.+)-HRRP")[0]

    # Pivot Excess Readmission Ratio by measure type
    pivot_err = df.pivot_table(
        index="Facility ID",
        columns="measure_short",
        values="Excess Readmission Ratio",
        aggfunc="first"
    )
    pivot_err.columns = [f"err_{c.lower()}" for c in pivot_err.columns]

    # Calculate aggregate metrics per facility
    agg = df.groupby("Facility ID").agg(
        avg_excess_readmission_ratio=("Excess Readmission Ratio", "mean"),
        total_discharges=("Number of Discharges", "sum"),
        total_readmissions=("Number of Readmissions", "sum"),
        measure_count=("Measure Name", "count")
    ).reset_index()

    # Merge pivot with aggregates
    result = agg.merge(pivot_err.reset_index(), on="Facility ID", how="left")

    print(f"Output shape: {result.shape}")
    print(f"Unique facilities: {result['Facility ID'].nunique()}")
    return result


def clean_svi(df: pd.DataFrame) -> pd.DataFrame:
    """Clean SVI: extract 5-digit ZIP from LOCATION."""
    print("\n--- Cleaning SVI ---")
    print(f"Input shape: {df.shape}")

    df = df.copy()

    # Extract 5-digit ZIP from LOCATION (handles "ZCTA5 12345" or just "12345")
    df["zip5"] = (
        df["LOCATION"]
        .astype(str)
        .str.extract(r"(\d{5})")[0]
        .str.zfill(5)
    )

    # Convert key SVI columns to numeric
    svi_numeric_cols = [
        "RPL_THEMES",  # Overall SVI percentile ranking
        "RPL_THEME1",  # Socioeconomic Status
        "RPL_THEME2",  # Household Characteristics
        "RPL_THEME3",  # Racial/Ethnic Minority Status
        "RPL_THEME4",  # Housing Type/Transportation
        "E_TOTPOP",    # Total population
        "EP_POV150",   # % below 150% poverty
        "EP_UNEMP",    # % unemployed
        "EP_NOHSDP",   # % no high school diploma
        "EP_UNINSUR",  # % uninsured
        "EP_NOVEH",    # % no vehicle
        "EP_AGE65",    # % age 65+
        "EP_DISABL",   # % with disability
    ]

    for col in svi_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Replace CDC's -999 missing code with NaN
            df.loc[df[col] < 0, col] = np.nan

    # Drop duplicates by zip5 (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["zip5"], keep="first")
    print(f"Dropped {before - len(df)} duplicate ZIPs")

    print(f"Output shape: {df.shape}")
    return df


def select_svi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant SVI columns for the final dataset."""
    keep_cols = [
        "zip5",
        # Theme percentile rankings (0-1)
        "RPL_THEMES",   # Overall SVI
        "RPL_THEME1",   # Socioeconomic Status
        "RPL_THEME2",   # Household Characteristics
        "RPL_THEME3",   # Racial/Ethnic Minority Status
        "RPL_THEME4",   # Housing Type/Transportation
        # Population
        "E_TOTPOP",
        # Interpretable raw percentages
        "EP_POV150",    # % below 150% poverty
        "EP_UNEMP",     # % unemployed
        "EP_NOHSDP",    # % no high school diploma
        "EP_UNINSUR",   # % uninsured
        "EP_NOVEH",     # % no vehicle
        "EP_AGE65",     # % age 65+
        "EP_DISABL",    # % with disability
    ]
    return df[[c for c in keep_cols if c in df.columns]]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer outcome variables and risk buckets.

    Features created:
        - err: Excess Readmission Ratio (renamed for clarity)
        - high_err_flag: 1 if err >= 1.0 (hospital has excess readmissions)
        - high_svi_flag: 1 if RPL_THEMES >= 0.75 (high social vulnerability)
        - risk_bucket: 4-category classification based on ERR x SVI
    """
    print("\n--- Feature Engineering ---")
    df = df.copy()

    # Outcome variable: rename for clarity
    df["err"] = df["avg_excess_readmission_ratio"]

    # Binary outcome: high excess readmission ratio
    df["high_err_flag"] = (df["err"] >= 1.0).astype("Int64")  # nullable int

    # Binary SVI flag: high social vulnerability (top 25%)
    df["high_svi_flag"] = (df["RPL_THEMES"] >= 0.75).astype("Int64")

    # 2x2 Risk bucket classification
    # High SVI + High ERR = highest concern
    # High SVI + Low ERR  = resilient hospitals
    # Low SVI + High ERR  = operational issue
    # Low SVI + Low ERR   = strong performers

    conditions = [
        (df["high_svi_flag"] == 1) & (df["high_err_flag"] == 1),
        (df["high_svi_flag"] == 1) & (df["high_err_flag"] == 0),
        (df["high_svi_flag"] == 0) & (df["high_err_flag"] == 1),
        (df["high_svi_flag"] == 0) & (df["high_err_flag"] == 0),
    ]
    choices = [
        "High SVI + High ERR",
        "High SVI + Low ERR",
        "Low SVI + High ERR",
        "Low SVI + Low ERR",
    ]
    df["risk_bucket"] = np.select(conditions, choices, default=None)
    df.loc[df["risk_bucket"] == "0", "risk_bucket"] = None  # np.select quirk

    # Summary stats
    print(f"ERR stats (non-null): n={df['err'].notna().sum()}, "
          f"mean={df['err'].mean():.3f}, median={df['err'].median():.3f}")
    print(f"High ERR flag: {df['high_err_flag'].sum()} / {df['high_err_flag'].notna().sum()} "
          f"({100*df['high_err_flag'].mean():.1f}%)")
    print(f"High SVI flag: {df['high_svi_flag'].sum()} / {df['high_svi_flag'].notna().sum()} "
          f"({100*df['high_svi_flag'].mean():.1f}%)")

    print("\nRisk bucket distribution:")
    bucket_counts = df["risk_bucket"].value_counts(dropna=False)
    for bucket, count in bucket_counts.items():
        label = bucket if bucket else "(missing data)"
        print(f"  {label}: {count:,}")

    return df


def main():
    project_root = Path(__file__).parent.parent.parent
    config = load_config(project_root / "config.yaml")
    processed_dir = project_root / config["paths"]["processed"]

    print("=" * 60)
    print("Step 1: Load parquet files")
    print("=" * 60)

    hrrp = pd.read_parquet(processed_dir / "hrrp.parquet")
    hospital_general = pd.read_parquet(processed_dir / "hospital_general.parquet")
    svi = pd.read_parquet(processed_dir / "svi.parquet")

    print(f"HRRP: {hrrp.shape}")
    print(f"Hospital General: {hospital_general.shape}")
    print(f"SVI: {svi.shape}")

    print("\n" + "=" * 60)
    print("Step 2: Clean datasets")
    print("=" * 60)

    hrrp_clean = clean_hrrp(hrrp)
    hg_clean = clean_hospital_general(hospital_general)
    svi_clean = clean_svi(svi)

    print("\n" + "=" * 60)
    print("Step 3: Merge datasets")
    print("=" * 60)

    # Merge HRRP (hospital info) with Hospital General (readmission measures)
    print("\n--- Merging HRRP + Hospital General on Facility ID ---")
    merged = hrrp_clean.merge(
        hg_clean,
        on="Facility ID",
        how="left"
    )
    print(f"After HRRP + Hospital General: {merged.shape}")

    # Merge with SVI on zip5
    print("\n--- Merging with SVI on zip5 ---")
    svi_subset = select_svi_columns(svi_clean)
    merged = merged.merge(
        svi_subset,
        on="zip5",
        how="left"
    )
    print(f"After SVI merge: {merged.shape}")

    # Report merge coverage
    print("\n--- Merge Coverage ---")
    hg_matched = merged["avg_excess_readmission_ratio"].notna().sum()
    svi_matched = merged["RPL_THEMES"].notna().sum()
    print(f"Hospitals with readmission data: {hg_matched:,} / {len(merged):,} ({100*hg_matched/len(merged):.1f}%)")
    print(f"Hospitals with SVI data: {svi_matched:,} / {len(merged):,} ({100*svi_matched/len(merged):.1f}%)")

    print("\n" + "=" * 60)
    print("Step 4: Feature Engineering")
    print("=" * 60)

    merged = engineer_features(merged)

    print("\n" + "=" * 60)
    print("Step 5: Save analytics table")
    print("=" * 60)

    output_path = processed_dir / "hospital_svi_analytics.parquet"
    merged.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Final shape: {merged.shape[0]:,} rows x {merged.shape[1]} columns")

    # Show sample of key columns
    print("\n--- Sample of key columns ---")
    key_cols = ["Facility ID", "Facility Name", "State", "zip5",
                "err", "high_err_flag", "RPL_THEMES", "high_svi_flag", "risk_bucket"]
    print(merged[key_cols].head(10).to_string())


if __name__ == "__main__":
    main()
