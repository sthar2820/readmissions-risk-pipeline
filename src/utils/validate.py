"""
validate.py - Data validation utilities for the readmissions pipeline

Provides validation checks at each pipeline stage:
    - Raw data: schema, required columns, basic sanity checks
    - Processed data: merge coverage, value ranges, null rates
    - Export data: final output validation

Usage:
    python -m src.utils.validate          # Run all validations
    python -m src.utils.validate --stage raw
    python -m src.utils.validate --stage processed
    python -m src.utils.validate --stage export
"""

import pandas as pd
import numpy as np
import yaml
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info

    def __str__(self):
        icon = "PASS" if self.passed else ("WARN" if self.severity == "warning" else "FAIL")
        return f"[{icon}] {self.name}: {self.message}"


class DataValidator:
    """Validates data at various pipeline stages."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config = self._load_config()
        self.results: list[ValidationResult] = []

    def _load_config(self) -> dict:
        with open(self.project_root / "config.yaml", "r") as f:
            return yaml.safe_load(f)

    def add_result(self, name: str, passed: bool, message: str, severity: str = "error"):
        self.results.append(ValidationResult(name, passed, message, severity))

    def check_file_exists(self, path: Path, name: str) -> bool:
        exists = path.exists()
        self.add_result(
            f"{name} exists",
            exists,
            f"Found at {path}" if exists else f"Missing: {path}"
        )
        return exists

    def check_row_count(self, df: pd.DataFrame, name: str, min_rows: int, max_rows: Optional[int] = None):
        count = len(df)
        if max_rows:
            passed = min_rows <= count <= max_rows
            expected = f"{min_rows:,}-{max_rows:,}"
        else:
            passed = count >= min_rows
            expected = f">= {min_rows:,}"
        self.add_result(
            f"{name} row count",
            passed,
            f"{count:,} rows (expected {expected})"
        )

    def check_required_columns(self, df: pd.DataFrame, name: str, required: list[str]):
        missing = [c for c in required if c not in df.columns]
        self.add_result(
            f"{name} required columns",
            len(missing) == 0,
            "All present" if not missing else f"Missing: {missing}"
        )

    def check_null_rate(self, df: pd.DataFrame, column: str, max_null_pct: float, severity: str = "warning"):
        if column not in df.columns:
            self.add_result(f"{column} null rate", False, "Column not found")
            return
        null_pct = df[column].isna().mean() * 100
        passed = null_pct <= max_null_pct
        self.add_result(
            f"{column} null rate",
            passed,
            f"{null_pct:.1f}% null (max {max_null_pct}%)",
            severity
        )

    def check_value_range(self, df: pd.DataFrame, column: str, min_val: float, max_val: float):
        if column not in df.columns:
            self.add_result(f"{column} range", False, "Column not found")
            return
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) == 0:
            self.add_result(f"{column} range", False, "No numeric values", "warning")
            return
        actual_min, actual_max = series.min(), series.max()
        passed = actual_min >= min_val and actual_max <= max_val
        self.add_result(
            f"{column} range",
            passed,
            f"[{actual_min:.3f}, {actual_max:.3f}] (expected [{min_val}, {max_val}])"
        )

    def check_unique(self, df: pd.DataFrame, column: str, name: str):
        if column not in df.columns:
            self.add_result(f"{name} unique {column}", False, "Column not found")
            return
        n_total = len(df)
        n_unique = df[column].nunique()
        n_duplicates = n_total - n_unique
        passed = n_duplicates == 0
        self.add_result(
            f"{name} unique {column}",
            passed,
            f"{n_unique:,} unique, {n_duplicates:,} duplicates",
            "warning" if not passed else "error"
        )

    def check_no_negative(self, df: pd.DataFrame, column: str):
        if column not in df.columns:
            self.add_result(f"{column} non-negative", False, "Column not found")
            return
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        n_negative = (series < 0).sum()
        passed = n_negative == 0
        self.add_result(
            f"{column} non-negative",
            passed,
            f"{n_negative:,} negative values found" if not passed else "All values >= 0"
        )

    # ==================== Stage Validators ====================

    def validate_raw(self) -> bool:
        """Validate raw CSV files."""
        print("\n" + "=" * 60)
        print("Validating RAW data")
        print("=" * 60)

        raw_dir = self.project_root / self.config["paths"]["raw"]

        # Check files exist
        hrrp_path = raw_dir / self.config["files"]["hrrp"]
        hosp_path = raw_dir / self.config["files"]["hospital_general"]
        svi_path = raw_dir / self.config["files"]["svi"]

        if not self.check_file_exists(hrrp_path, "HRRP CSV"):
            return False
        if not self.check_file_exists(hosp_path, "Hospital General CSV"):
            return False
        if not self.check_file_exists(svi_path, "SVI CSV"):
            return False

        # Load and validate HRRP (hospital general info)
        hrrp = pd.read_csv(hrrp_path, dtype=str)
        self.check_row_count(hrrp, "HRRP", min_rows=1000, max_rows=10000)
        self.check_required_columns(hrrp, "HRRP", [
            "Facility ID", "Facility Name", "State", "ZIP Code", "Hospital Type"
        ])
        self.check_unique(hrrp, "Facility ID", "HRRP")

        # Load and validate Hospital General (readmission measures)
        hosp = pd.read_csv(hosp_path, dtype=str)
        self.check_row_count(hosp, "Hospital General", min_rows=5000, max_rows=50000)
        self.check_required_columns(hosp, "Hospital General", [
            "Facility ID", "Measure Name", "Excess Readmission Ratio"
        ])

        # Load and validate SVI
        svi = pd.read_csv(svi_path, dtype=str)
        self.check_row_count(svi, "SVI", min_rows=10000, max_rows=100000)
        self.check_required_columns(svi, "SVI", [
            "FIPS", "LOCATION", "RPL_THEMES", "E_TOTPOP"
        ])

        return self._summarize("Raw")

    def validate_processed(self) -> bool:
        """Validate processed parquet files."""
        print("\n" + "=" * 60)
        print("Validating PROCESSED data")
        print("=" * 60)

        processed_dir = self.project_root / self.config["paths"]["processed"]

        # Check parquet files exist
        analytics_path = processed_dir / "hospital_svi_analytics.parquet"
        if not self.check_file_exists(analytics_path, "Analytics parquet"):
            return False

        # Load and validate analytics table
        df = pd.read_parquet(analytics_path)

        # Row count
        self.check_row_count(df, "Analytics", min_rows=1000, max_rows=10000)

        # Required columns
        self.check_required_columns(df, "Analytics", [
            "Facility ID", "Facility Name", "State", "zip5",
            "err", "high_err_flag", "high_svi_flag", "risk_bucket",
            "RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"
        ])

        # Unique facility IDs
        self.check_unique(df, "Facility ID", "Analytics")

        # ERR value range (should be 0-3 typically)
        self.check_value_range(df, "err", min_val=0.0, max_val=3.0)

        # SVI percentile range (should be 0-1)
        self.check_value_range(df, "RPL_THEMES", min_val=0.0, max_val=1.0)

        # Check no negative values in SVI (after -999 cleaning)
        self.check_no_negative(df, "RPL_THEMES")
        self.check_no_negative(df, "EP_POV150")

        # Null rates
        self.check_null_rate(df, "Facility ID", max_null_pct=0)
        self.check_null_rate(df, "State", max_null_pct=0)
        self.check_null_rate(df, "err", max_null_pct=60, severity="warning")  # Not all hospitals in HRRP
        self.check_null_rate(df, "RPL_THEMES", max_null_pct=10, severity="warning")

        # Risk bucket validation
        if "risk_bucket" in df.columns:
            valid_buckets = {
                "High SVI + High ERR", "High SVI + Low ERR",
                "Low SVI + High ERR", "Low SVI + Low ERR", None, "0"
            }
            invalid = set(df["risk_bucket"].dropna().unique()) - valid_buckets
            self.add_result(
                "risk_bucket values",
                len(invalid) == 0,
                "All valid" if not invalid else f"Invalid values: {invalid}"
            )

        # Binary flag validation
        for flag in ["high_err_flag", "high_svi_flag"]:
            if flag in df.columns:
                unique_vals = set(df[flag].dropna().unique())
                valid = unique_vals <= {0, 1}
                self.add_result(
                    f"{flag} is binary",
                    valid,
                    f"Values: {unique_vals}"
                )

        return self._summarize("Processed")

    def validate_export(self) -> bool:
        """Validate exported CSV for Power BI."""
        print("\n" + "=" * 60)
        print("Validating EXPORT data")
        print("=" * 60)

        reports_dir = self.project_root / self.config["paths"]["reports"]
        csv_path = reports_dir / "powerbi_hospital_svi.csv"
        dict_path = reports_dir / "data_dictionary.md"

        if not self.check_file_exists(csv_path, "Power BI CSV"):
            return False
        self.check_file_exists(dict_path, "Data dictionary")

        # Load and validate CSV
        df = pd.read_csv(csv_path)

        # Row count should match processed
        self.check_row_count(df, "Export CSV", min_rows=1000, max_rows=10000)

        # Check key columns for Power BI
        self.check_required_columns(df, "Export", [
            "Facility ID", "Facility Name", "State", "zip5",
            "err", "risk_bucket", "RPL_THEMES"
        ])

        # No fully null columns
        for col in df.columns:
            if df[col].isna().all():
                self.add_result(f"{col} not all null", False, "Column is entirely null")

        # File size check (should be reasonable for Power BI)
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        self.add_result(
            "Export file size",
            size_mb < 50,
            f"{size_mb:.2f} MB",
            "warning"
        )

        return self._summarize("Export")

    def _summarize(self, stage: str) -> bool:
        """Print summary and return overall pass/fail."""
        print(f"\n--- {stage} Validation Results ---")

        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]
        passed = [r for r in self.results if r.passed]

        for r in self.results:
            print(f"  {r}")

        print(f"\nSummary: {len(passed)} passed, {len(warnings)} warnings, {len(errors)} errors")

        # Clear results for next stage
        self.results = []

        return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline data")
    parser.add_argument(
        "--stage",
        choices=["raw", "processed", "export", "all"],
        default="all",
        help="Which stage to validate"
    )
    args = parser.parse_args()

    # Find project root
    project_root = Path(__file__).parent.parent.parent
    validator = DataValidator(project_root)

    stages = {
        "raw": validator.validate_raw,
        "processed": validator.validate_processed,
        "export": validator.validate_export,
    }

    if args.stage == "all":
        results = [func() for func in stages.values()]
        all_passed = all(results)
    else:
        all_passed = stages[args.stage]()

    print("\n" + "=" * 60)
    if all_passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
