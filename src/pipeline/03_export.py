"""
03_export.py - Export analytics data for Power BI

Exports:
    - reports/powerbi_hospital_svi.csv (Power BI ready)
    - reports/data_dictionary.md (column documentation)
"""

import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Column definitions for export and documentation
EXPORT_COLUMNS = {
    # Hospital identifiers
    "Facility ID": "Unique CMS facility identifier (CCN)",
    "Facility Name": "Hospital name",
    "Address": "Street address",
    "City/Town": "City",
    "State": "State abbreviation",
    "zip5": "5-digit ZIP code (normalized)",
    "County/Parish": "County name",
    # Hospital characteristics
    "Hospital Type": "Type of hospital (Acute Care, Critical Access, etc.)",
    "Hospital Ownership": "Ownership type (Government, Non-profit, Proprietary)",
    "Emergency Services": "Has emergency services (Yes/No)",
    "Hospital overall rating": "CMS star rating (1-5)",
    # Readmission outcomes
    "err": "Excess Readmission Ratio - average across all conditions",
    "high_err_flag": "1 if ERR >= 1.0 (excess readmissions), 0 otherwise",
    "total_discharges": "Total discharges across HRRP conditions",
    "total_readmissions": "Total readmissions across HRRP conditions",
    "measure_count": "Number of HRRP measures reported",
    # Condition-specific ERR
    "err_ami": "ERR for Acute Myocardial Infarction",
    "err_cabg": "ERR for Coronary Artery Bypass Graft",
    "err_copd": "ERR for Chronic Obstructive Pulmonary Disease",
    "err_hf": "ERR for Heart Failure",
    "err_hip-knee": "ERR for Hip/Knee Replacement",
    "err_pn": "ERR for Pneumonia",
    # SVI theme scores (percentile 0-1, higher = more vulnerable)
    "RPL_THEMES": "Overall Social Vulnerability Index (0-1 percentile)",
    "RPL_THEME1": "SVI Theme 1: Socioeconomic Status",
    "RPL_THEME2": "SVI Theme 2: Household Characteristics",
    "RPL_THEME3": "SVI Theme 3: Racial/Ethnic Minority Status",
    "RPL_THEME4": "SVI Theme 4: Housing Type/Transportation",
    # SVI raw percentages
    "E_TOTPOP": "Total population in ZIP code",
    "EP_POV150": "% population below 150% poverty level",
    "EP_UNEMP": "% civilian unemployed (age 16+)",
    "EP_NOHSDP": "% with no high school diploma (age 25+)",
    "EP_UNINSUR": "% uninsured",
    "EP_NOVEH": "% households with no vehicle",
    "EP_AGE65": "% population age 65+",
    "EP_DISABL": "% civilian noninstitutionalized with disability",
    # Risk classification
    "high_svi_flag": "1 if RPL_THEMES >= 0.75 (high vulnerability), 0 otherwise",
    "risk_bucket": "Risk category: High/Low SVI + High/Low ERR",
}


def select_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order columns for Power BI export."""
    export_cols = [c for c in EXPORT_COLUMNS.keys() if c in df.columns]
    return df[export_cols]


def generate_data_dictionary(df: pd.DataFrame, output_path: Path) -> None:
    """Generate markdown data dictionary."""
    lines = [
        "# Hospital SVI Analytics - Data Dictionary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Total Records:** {len(df):,} hospitals",
        "",
        "---",
        "",
        "## Column Definitions",
        "",
        "| Column | Description | Type | Non-Null |",
        "|--------|-------------|------|----------|",
    ]

    for col in df.columns:
        desc = EXPORT_COLUMNS.get(col, "—")
        dtype = str(df[col].dtype)
        non_null = f"{df[col].notna().sum():,} ({100*df[col].notna().mean():.0f}%)"
        lines.append(f"| {col} | {desc} | {dtype} | {non_null} |")

    lines.extend([
        "",
        "---",
        "",
        "## Risk Bucket Definitions",
        "",
        "| Bucket | SVI | ERR | Interpretation |",
        "|--------|-----|-----|----------------|",
        "| High SVI + High ERR | >= 0.75 | >= 1.0 | Highest concern - vulnerable population + excess readmissions |",
        "| High SVI + Low ERR | >= 0.75 | < 1.0 | Resilient hospitals - managing well despite challenges |",
        "| Low SVI + High ERR | < 0.75 | >= 1.0 | Operational issue - excess readmissions without social barriers |",
        "| Low SVI + Low ERR | < 0.75 | < 1.0 | Strong performers |",
        "",
        "---",
        "",
        "## Key Metrics Summary",
        "",
    ])

    # Add summary statistics
    if "err" in df.columns:
        err_stats = df["err"].describe()
        lines.extend([
            "### Excess Readmission Ratio (ERR)",
            "",
            f"- Mean: {err_stats['mean']:.3f}",
            f"- Median: {err_stats['50%']:.3f}",
            f"- Std Dev: {err_stats['std']:.3f}",
            f"- Range: {err_stats['min']:.3f} - {err_stats['max']:.3f}",
            f"- Hospitals with ERR >= 1.0: {(df['err'] >= 1.0).sum():,} ({100*(df['err'] >= 1.0).mean():.1f}%)",
            "",
        ])

    if "RPL_THEMES" in df.columns:
        svi_stats = df["RPL_THEMES"].describe()
        lines.extend([
            "### Social Vulnerability Index (RPL_THEMES)",
            "",
            f"- Mean: {svi_stats['mean']:.3f}",
            f"- Median: {svi_stats['50%']:.3f}",
            f"- Hospitals in high-vulnerability areas (>= 0.75): {(df['RPL_THEMES'] >= 0.75).sum():,} ({100*(df['RPL_THEMES'] >= 0.75).mean():.1f}%)",
            "",
        ])

    if "risk_bucket" in df.columns:
        lines.extend([
            "### Risk Bucket Distribution",
            "",
        ])
        for bucket, count in df["risk_bucket"].value_counts().items():
            lines.append(f"- {bucket}: {count:,} ({100*count/len(df):.1f}%)")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Data Sources",
        "",
        "- **HRRP Data**: CMS Hospital Readmissions Reduction Program",
        "- **Hospital Info**: CMS Hospital General Information",
        "- **SVI Data**: CDC/ATSDR Social Vulnerability Index (ZCTA level)",
        "",
    ])

    output_path.write_text("\n".join(lines))


def main():
    project_root = Path(__file__).parent.parent.parent
    config = load_config(project_root / "config.yaml")

    processed_dir = project_root / config["paths"]["processed"]
    reports_dir = project_root / config["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading analytics data")
    print("=" * 60)

    df = pd.read_parquet(processed_dir / "hospital_svi_analytics.parquet")
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n" + "=" * 60)
    print("Exporting for Power BI")
    print("=" * 60)

    # Select columns for export
    export_df = select_export_columns(df)
    print(f"Selected {len(export_df.columns)} columns for export")

    # Export CSV
    csv_path = reports_dir / "powerbi_hospital_svi.csv"
    export_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(f"  Size: {csv_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("Generating data dictionary")
    print("=" * 60)

    dict_path = reports_dir / "data_dictionary.md"
    generate_data_dictionary(export_df, dict_path)
    print(f"\nSaved: {dict_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nFiles created in {reports_dir}/:")
    print(f"  1. powerbi_hospital_svi.csv  ({len(export_df):,} rows, {len(export_df.columns)} cols)")
    print(f"  2. data_dictionary.md")


if __name__ == "__main__":
    main()
