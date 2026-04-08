import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TASKS = [
    "absorption",
    "emission",
    "quantum_yield",
    "log_molar_absorptivity",
]
DEFAULT_DATASETS = ["consolidation", "cyanine", "xanthene"]
DEFAULT_SPLITS = ["random", "scaffold"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build task-specific training directories from split CSV files."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("../datasets/raw"),
        help="Directory containing random/scaffold split CSV files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../datasets"),
        help="Target directory to create task folders.",
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--split-methods", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--n-folds", type=int, default=5)
    return parser.parse_args()


def build_data(raw_data_path: Path, data_path: Path, task_name: str):
    df = pd.read_csv(raw_data_path)
    logger.info(f"before dropna: {len(df)}")
    df = df.dropna(subset=[task_name]).reset_index(drop=True)

    df_new = pd.DataFrame()
    df_new["smiles"] = df["smiles"]
    df_new["solvent"] = df["solvent"]
    df_new[task_name] = df[task_name]
    df_new["split"] = df["split"]

    data_path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(data_path, index=False)
    logger.info(
        f"Dataset {task_name} with {len(df_new)} samples, "
        f"{len(df_new[df_new['split'] == 'test'])} test, "
        f"{len(df_new[df_new['split'] == 'valid'])} valid, "
        f"{len(df_new[df_new['split'] == 'train'])} train"
    )


def main():
    args = parse_args()
    for split_method in args.split_methods:
        for dataset_name in args.datasets:
            for task_name in args.tasks:
                for fold in range(args.n_folds):
                    logger.info(
                        f"Building {task_name} for {dataset_name} {split_method} fold {fold}"
                    )
                    task_dir = (
                        args.data_dir
                        / split_method
                        / f"{dataset_name}_fold{fold}"
                        / task_name
                    )
                    raw_data_path = (
                        args.raw_data_dir
                        / split_method
                        / f"{dataset_name}_fold{fold}.csv"
                    )
                    data_path = task_dir / f"{task_name}.csv"
                    build_data(raw_data_path, data_path, task_name)


if __name__ == "__main__":
    main()
