"""

Loads the 3 raw CSV files (HRRP, Hospital General Info, SVI) and saves them
as Parquet files for faster downstream processing.

Outputs:
    - data/processed/hrrp.parquet
    - data/processed/hospital_general.parquet
    - data/processed/svi.parquet
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ingest_csv(filepath: Path, name: str) -> pd.DataFrame:
    """Load a CSV file and print basic info."""
    print(f"\n{'='*50}")
    print(f"Loading: {name}")
    print(f"{'='*50}")

    df = pd.read_csv(filepath, dtype=str)  # Load as string to preserve data

    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2}. {col}")

    return df


def main():
    # Get project root (assumes script is run from project root)
    project_root = Path(__file__).parent.parent.parent
    config = load_config(project_root / "config.yaml")

    raw_dir = project_root / config["paths"]["raw"]
    processed_dir = project_root / config["paths"]["processed"]

    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Define input/output mappings
    datasets = {
        "hrrp": {
            "input": raw_dir / config["files"]["hrrp"],
            "output": processed_dir / "hrrp.parquet"
        },
        "hospital_general": {
            "input": raw_dir / config["files"]["hospital_general"],
            "output": processed_dir / "hospital_general.parquet"
        },
        "svi": {
            "input": raw_dir / config["files"]["svi"],
            "output": processed_dir / "svi.parquet"
        }
    }

    # Process each dataset
    for name, paths in datasets.items():
        df = ingest_csv(paths["input"], name)

        # Save as parquet
        df.to_parquet(paths["output"], index=False)
        print(f"\nSaved to: {paths['output']}")

    print(f"\n{'='*50}")
    print("Ingestion complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
