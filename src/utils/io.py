"""
io.py - I/O utilities for the readmissions pipeline

Provides consistent file reading/writing functions:
    - Config loading
    - CSV/Parquet with standard options
    - Path resolution
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Union


# ==================== Config ====================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses project root config.yaml

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = get_project_root() / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_path(key: str, config: Optional[dict] = None) -> Path:
    """
    Get a path from config.

    Args:
        key: Path key (e.g., 'raw', 'processed', 'reports')
        config: Config dict. If None, loads from file.

    Returns:
        Resolved Path object
    """
    if config is None:
        config = load_config()

    return get_project_root() / config["paths"][key]


# ==================== CSV I/O ====================

def read_csv(
    filepath: Union[str, Path],
    dtype: Optional[dict] = None,
    as_string: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Read CSV with standard options.

    Args:
        filepath: Path to CSV file
        dtype: Column dtypes dict
        as_string: If True, load all columns as strings (preserves leading zeros)
        **kwargs: Additional pandas read_csv arguments

    Returns:
        DataFrame
    """
    if as_string:
        dtype = str

    return pd.read_csv(filepath, dtype=dtype, **kwargs)


def write_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
    **kwargs
) -> None:
    """
    Write CSV with standard options.

    Args:
        df: DataFrame to write
        filepath: Output path
        index: Whether to write index (default False)
        **kwargs: Additional pandas to_csv arguments
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index, **kwargs)


# ==================== Parquet I/O ====================

def read_parquet(
    filepath: Union[str, Path],
    columns: Optional[list[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read Parquet file.

    Args:
        filepath: Path to Parquet file
        columns: Specific columns to read (None = all)
        **kwargs: Additional pandas read_parquet arguments

    Returns:
        DataFrame
    """
    return pd.read_parquet(filepath, columns=columns, **kwargs)


def write_parquet(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
    **kwargs
) -> None:
    """
    Write Parquet file.

    Args:
        df: DataFrame to write
        filepath: Output path
        index: Whether to write index (default False)
        **kwargs: Additional pandas to_parquet arguments
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=index, **kwargs)


# ==================== Pipeline Helpers ====================

def load_raw_data(config: Optional[dict] = None) -> dict[str, pd.DataFrame]:
    """
    Load all raw CSV files as DataFrames.

    Args:
        config: Config dict. If None, loads from file.

    Returns:
        Dict with keys 'hrrp', 'hospital_general', 'svi'
    """
    if config is None:
        config = load_config()

    raw_dir = get_project_root() / config["paths"]["raw"]

    return {
        "hrrp": read_csv(raw_dir / config["files"]["hrrp"], as_string=True),
        "hospital_general": read_csv(raw_dir / config["files"]["hospital_general"], as_string=True),
        "svi": read_csv(raw_dir / config["files"]["svi"], as_string=True),
    }


def load_processed_data(config: Optional[dict] = None) -> dict[str, pd.DataFrame]:
    """
    Load all processed Parquet files as DataFrames.

    Args:
        config: Config dict. If None, loads from file.

    Returns:
        Dict with keys 'hrrp', 'hospital_general', 'svi', 'analytics'
    """
    if config is None:
        config = load_config()

    processed_dir = get_project_root() / config["paths"]["processed"]

    data = {}

    # Individual processed files
    for name in ["hrrp", "hospital_general", "svi"]:
        path = processed_dir / f"{name}.parquet"
        if path.exists():
            data[name] = read_parquet(path)

    # Analytics table
    analytics_path = processed_dir / "hospital_svi_analytics.parquet"
    if analytics_path.exists():
        data["analytics"] = read_parquet(analytics_path)

    return data


def load_analytics() -> pd.DataFrame:
    """
    Load the final analytics table.

    Returns:
        Analytics DataFrame
    """
    config = load_config()
    processed_dir = get_project_root() / config["paths"]["processed"]
    return read_parquet(processed_dir / "hospital_svi_analytics.parquet")
