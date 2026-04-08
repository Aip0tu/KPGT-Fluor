import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from molvs import standardize_smiles
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


RAW_TO_STANDARD_COLUMNS = {
    "SMILES": "smiles",
    "smiles": "smiles",
    "Solvent": "solvent",
    "solvent": "solvent",
    "Ex (nm)": "absorption",
    "absorption": "absorption",
    "Em (nm)": "emission",
    "emission": "emission",
    "QY": "quantum_yield",
    "quantum_yield": "quantum_yield",
    "Log(ε)": "log_molar_absorptivity",
    "Log(¦Å)": "log_molar_absorptivity",
    "log_molar_absorptivity": "log_molar_absorptivity",
    "ε (cm-1M-1)": "molar_absorptivity",
    "¦Å (cm-1M-1)": "molar_absorptivity",
    "Reference": "reference",
    "reference": "reference",
}

DEFAULT_DATASETS = ["Consolidation", "Cyanine", "Xanthene"]
CSV_ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin1"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Canonicalize raw fluorescence datasets."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("../datasets/raw"),
        help="Directory containing Dataset_*.csv files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset suffixes to process, e.g. Consolidation Cyanine Xanthene.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_canonicalized.csv files.",
    )
    return parser.parse_args()


def canonicalize_smiles(smiles: str) -> Optional[str]:
    try:
        return standardize_smiles(smiles)
    except Exception as exc:
        logger.error(f"Error canonicalizing smiles: {smiles} with error: {exc}")
        return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        column: RAW_TO_STANDARD_COLUMNS[column]
        for column in df.columns
        if column in RAW_TO_STANDARD_COLUMNS
    }
    return df.rename(columns=rename_map)


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            logger.info(f"Trying to read {path} with encoding={encoding}")
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def canonicalize_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    tqdm.pandas(desc=f"canonicalizing {column}")
    logger.info(f"before canonicalizing # {column} in {df.shape[0]} rows")
    df[column] = df[column].progress_apply(canonicalize_smiles)
    df = df.dropna(subset=[column]).reset_index(drop=True)
    logger.info(f"after canonicalizing # {column} in {df.shape[0]} rows")
    return df


def process_dataset(raw_data_dir: Path, dataset_name: str, overwrite: bool = False):
    input_path = raw_data_dir / f"Dataset_{dataset_name}.csv"
    output_path = raw_data_dir / f"Dataset_{dataset_name}_canonicalized.csv"

    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return

    logger.info(f"Reading raw dataset from {input_path}")
    df = read_csv_with_fallback(input_path)
    df = normalize_columns(df)

    required_columns = {"smiles", "solvent"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing_columns)}")

    df = canonicalize_df(df, "smiles")
    df = canonicalize_df(df, "solvent")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved canonicalized dataset to {output_path}")


def main():
    args = parse_args()
    for dataset_name in args.datasets:
        process_dataset(args.raw_data_dir, dataset_name, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
