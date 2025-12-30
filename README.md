# Hospital Readmissions Risk Pipeline

A data pipeline that merges CMS Hospital Readmissions data with CDC Social Vulnerability Index (SVI) to analyze readmission risk across socioeconomic factors.

## Overview

This pipeline:
1. Ingests CMS Hospital Readmissions Reduction Program (HRRP) data and CDC SVI data
2. Cleans and normalizes facility IDs, ZIP codes, and numeric fields
3. Merges hospital performance metrics with community vulnerability indicators
4. Engineers risk features including 2×2 risk buckets (High/Low SVI × High/Low ERR)
5. Exports Power BI-ready datasets

## Project Structure

```
├── data/
│   ├── raw/                  # Source CSV files
│   └── processed/            # Cleaned parquet files
├── src/
│   ├── pipeline/
│   │   ├── 01_download.py    # Ingest CSVs → Parquet
│   │   ├── 02_clean_join.py  # Clean, merge, feature engineering
│   │   └── 03_export.py      # Export for Power BI
│   └── utils/
│       ├── cleaning.py       # Data cleaning functions
│       ├── io.py             # File I/O utilities
│       └── validate.py       # Data validation
├── notebooks/
│   └── 01_modeling.ipynb     # ML modeling preparation
├── reports/                  # Generated outputs
└── .github/workflows/        # CI/CD automation
```

## Data Sources

| Dataset | Description | Key Fields |
|---------|-------------|------------|
| HRRP | Hospital Readmissions Reduction Program | Facility ID, Excess Readmission Ratio |
| Hospital General Info | Hospital demographics | Facility ID, ZIP Code, Hospital Type |
| SVI | CDC Social Vulnerability Index | ZCTA, RPL_THEMES (0-1 percentile) |

## Key Features Engineered

- **err**: Excess Readmission Ratio (float)
- **high_err_flag**: Binary flag for ERR >= 1.0
- **high_svi_flag**: Binary flag for SVI >= 75th percentile
- **risk_bucket**: 4-category classification
  - High SVI + High ERR (highest risk)
  - High SVI + Low ERR
  - Low SVI + High ERR
  - Low SVI + Low ERR (lowest risk)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python3 src/pipeline/01_download.py
python3 src/pipeline/02_clean_join.py
python3 src/pipeline/03_export.py
```

## Output

The pipeline generates:
- `data/processed/hospital_svi_analytics.parquet` - Full analytics dataset
- `reports/powerbi_hospital_svi.csv` - Power BI export (37 columns)
- `reports/data_dictionary.md` - Column documentation

## GitHub Actions

The pipeline runs automatically on push via GitHub Actions. Artifacts are uploaded for download after each run.

## Requirements

- Python 3.11+
- pandas >= 2.0.0
- pyarrow >= 14.0.0
- pyyaml >= 6.0
