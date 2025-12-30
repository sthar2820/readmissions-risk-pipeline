"""
cleaning.py - Data cleaning utilities for the readmissions pipeline

Provides reusable cleaning functions for:
    - Facility IDs: standardization and zero-padding
    - ZIP codes: normalization to 5-digit format
    - Numeric columns: type conversion and missing value handling
    - Text columns: whitespace and case normalization
    - CDC SVI specific: handling -999 missing codes
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


# ==================== ID Cleaning ====================

def clean_facility_id(
    df: pd.DataFrame,
    col: str = "Facility ID",
    pad_length: int = 6
) -> pd.DataFrame:
    """
    Standardize Facility ID to zero-padded string.

    Args:
        df: DataFrame containing facility IDs
        col: Column name containing facility IDs
        pad_length: Length to zero-pad to (default 6 for CMS CCN)

    Returns:
        DataFrame with cleaned facility ID column
    """
    df = df.copy()
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(r"[^\d]", "", regex=True)  # Remove non-digits
        .str.zfill(pad_length)
    )
    return df


# ==================== ZIP Code Cleaning ====================

def clean_zip(
    df: pd.DataFrame,
    col: str = "ZIP Code",
    output_col: str = "zip5"
) -> pd.DataFrame:
    """
    Normalize ZIP codes to 5-digit format.

    Handles:
        - ZIP+4 format (12345-6789 -> 12345)
        - Whitespace
        - Non-numeric characters
        - Zero-padding for short ZIPs

    Args:
        df: DataFrame containing ZIP codes
        col: Input column name
        output_col: Output column name for cleaned ZIP

    Returns:
        DataFrame with new zip5 column
    """
    df = df.copy()
    df[output_col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)  # Remove whitespace
        .str.split("-").str[0]  # Take first part of ZIP+4
        .str[:5]  # Keep only first 5 digits
        .str.zfill(5)  # Zero-pad if needed
    )
    return df


def extract_zip_from_zcta(
    df: pd.DataFrame,
    col: str = "LOCATION",
    output_col: str = "zip5"
) -> pd.DataFrame:
    """
    Extract 5-digit ZIP from ZCTA location strings.

    Handles formats like:
        - "ZCTA5 12345"
        - "12345"
        - "ZCTA5 01234" (preserves leading zeros)

    Args:
        df: DataFrame containing ZCTA location strings
        col: Input column name
        output_col: Output column name for extracted ZIP

    Returns:
        DataFrame with new zip5 column
    """
    df = df.copy()
    df[output_col] = (
        df[col]
        .astype(str)
        .str.extract(r"(\d{5})")[0]
        .str.zfill(5)
    )
    return df


# ==================== Numeric Cleaning ====================

def to_numeric_safe(
    df: pd.DataFrame,
    columns: Union[str, list[str]],
    errors: str = "coerce"
) -> pd.DataFrame:
    """
    Convert columns to numeric, handling non-numeric values.

    Args:
        df: DataFrame
        columns: Column name or list of column names
        errors: How to handle errors ('coerce' = NaN, 'ignore' = keep original)

    Returns:
        DataFrame with converted columns
    """
    df = df.copy()
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors=errors)

    return df


def replace_missing_codes(
    df: pd.DataFrame,
    columns: Union[str, list[str]],
    missing_codes: list = [-999, -999.0, "-999", "N/A", "NA", "Not Available"],
    replacement: Optional[float] = None
) -> pd.DataFrame:
    """
    Replace missing value codes with NaN or specified value.

    Common in CDC/CMS data where -999 indicates suppressed/missing data.

    Args:
        df: DataFrame
        columns: Column name or list of column names
        missing_codes: Values to treat as missing
        replacement: Value to replace with (None = NaN)

    Returns:
        DataFrame with replaced values
    """
    df = df.copy()
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col in df.columns:
            df.loc[df[col].isin(missing_codes), col] = replacement
            # Also handle numeric -999 after conversion
            if pd.api.types.is_numeric_dtype(df[col]):
                df.loc[df[col] < -900, col] = replacement

    return df


def clean_numeric_column(
    df: pd.DataFrame,
    col: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    replace_out_of_range: bool = True
) -> pd.DataFrame:
    """
    Clean a numeric column: convert type, handle missing codes, validate range.

    Args:
        df: DataFrame
        col: Column name
        min_val: Minimum valid value (values below become NaN if replace_out_of_range)
        max_val: Maximum valid value (values above become NaN if replace_out_of_range)
        replace_out_of_range: Whether to replace out-of-range values with NaN

    Returns:
        DataFrame with cleaned column
    """
    df = df.copy()

    if col not in df.columns:
        return df

    # Convert to numeric
    df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace CDC missing codes
    df.loc[df[col] < -900, col] = np.nan

    # Validate range
    if replace_out_of_range:
        if min_val is not None:
            df.loc[df[col] < min_val, col] = np.nan
        if max_val is not None:
            df.loc[df[col] > max_val, col] = np.nan

    return df


# ==================== Text Cleaning ====================

def clean_text_column(
    df: pd.DataFrame,
    col: str,
    uppercase: bool = False,
    lowercase: bool = False,
    strip: bool = True
) -> pd.DataFrame:
    """
    Clean a text column: strip whitespace, normalize case.

    Args:
        df: DataFrame
        col: Column name
        uppercase: Convert to uppercase
        lowercase: Convert to lowercase
        strip: Strip leading/trailing whitespace

    Returns:
        DataFrame with cleaned column
    """
    df = df.copy()

    if col not in df.columns:
        return df

    series = df[col].astype(str)

    if strip:
        series = series.str.strip()

    if uppercase:
        series = series.str.upper()
    elif lowercase:
        series = series.str.lower()

    df[col] = series
    return df


def standardize_state(df: pd.DataFrame, col: str = "State") -> pd.DataFrame:
    """
    Standardize state codes to 2-letter uppercase abbreviations.

    Args:
        df: DataFrame
        col: Column name containing state codes

    Returns:
        DataFrame with standardized state codes
    """
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper().str[:2]
    return df


# ==================== Deduplication ====================

def deduplicate(
    df: pd.DataFrame,
    subset: Union[str, list[str]],
    keep: str = "first",
    report: bool = True
) -> pd.DataFrame:
    """
    Remove duplicate rows based on subset of columns.

    Args:
        df: DataFrame
        subset: Column(s) to check for duplicates
        keep: Which duplicate to keep ('first', 'last', False)
        report: Whether to print deduplication summary

    Returns:
        Deduplicated DataFrame
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    after = len(df)

    if report and before != after:
        print(f"Deduplicated on {subset}: {before:,} -> {after:,} ({before - after:,} removed)")

    return df


# ==================== SVI-Specific Cleaning ====================

SVI_NUMERIC_COLS = [
    # Theme percentile rankings
    "RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4",
    # Population
    "E_TOTPOP",
    # Percentages
    "EP_POV150", "EP_UNEMP", "EP_NOHSDP", "EP_UNINSUR",
    "EP_NOVEH", "EP_AGE65", "EP_DISABL", "EP_LIMENG",
    # Counts
    "E_POV150", "E_UNEMP", "E_NOHSDP", "E_UNINSUR",
]


def clean_svi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard SVI cleaning: extract ZIP, convert numerics, handle -999.

    Args:
        df: Raw SVI DataFrame

    Returns:
        Cleaned SVI DataFrame
    """
    df = df.copy()

    # Extract ZIP from LOCATION
    df = extract_zip_from_zcta(df, col="LOCATION", output_col="zip5")

    # Convert and clean numeric columns
    for col in SVI_NUMERIC_COLS:
        if col in df.columns:
            df = clean_numeric_column(df, col)

    # Validate RPL columns are 0-1 percentiles
    for col in ["RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]:
        if col in df.columns:
            df.loc[(df[col] < 0) | (df[col] > 1), col] = np.nan

    return df


# ==================== HRRP-Specific Cleaning ====================

def clean_hrrp_measures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean HRRP readmission measures data.

    Args:
        df: Raw HRRP measures DataFrame

    Returns:
        Cleaned HRRP measures DataFrame
    """
    df = df.copy()

    # Clean Facility ID
    df = clean_facility_id(df)

    # Convert numeric measure columns
    numeric_cols = [
        "Excess Readmission Ratio",
        "Predicted Readmission Rate",
        "Expected Readmission Rate",
        "Number of Discharges",
        "Number of Readmissions"
    ]
    df = to_numeric_safe(df, numeric_cols)

    # Replace "N/A" and "Too Few to Report" with NaN
    df = replace_missing_codes(df, numeric_cols)

    return df


def clean_hospital_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean hospital general information data.

    Args:
        df: Raw hospital info DataFrame

    Returns:
        Cleaned hospital info DataFrame
    """
    df = df.copy()

    # Clean IDs and ZIPs
    df = clean_facility_id(df)
    df = clean_zip(df)

    # Standardize state
    df = standardize_state(df)

    # Clean text columns
    for col in ["Facility Name", "City/Town", "County/Parish"]:
        if col in df.columns:
            df = clean_text_column(df, col, strip=True)

    return df
