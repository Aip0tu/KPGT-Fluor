import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASETS = ["consolidation", "cyanine", "xanthene"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build random/scaffold folds from canonicalized fluorescence datasets."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("../datasets/raw"),
        help="Directory containing Dataset_*_canonicalized.csv files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names in lowercase: consolidation cyanine xanthene.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset(raw_data_dir: Path, dataset_name: str) -> pd.DataFrame:
    suffix = dataset_name.capitalize() if dataset_name != "consolidation" else "Consolidation"
    file_path = raw_data_dir / f"Dataset_{suffix}_canonicalized.csv"
    logger.info(f"Loading {file_path}")
    return pd.read_csv(file_path)


def drop_duplicates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    logger.info(f"before dropping duplicates: {df.shape[0]} rows")
    df = df.drop_duplicates(subset=columns).reset_index(drop=True)
    logger.info(f"after dropping duplicates: {df.shape[0]} rows")
    return df


def random_split(df: pd.DataFrame, save_dir: Path, name: str, n_splits: int, seed: int):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    target_dir = save_dir / "random"
    target_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_index, valid_index) in enumerate(kf.split(df)):
        fold_df = df.copy()
        fold_df.loc[valid_index, "split"] = "valid"
        fold_df.loc[train_index, "split"] = "train"
        test_df = fold_df[fold_df["split"] == "valid"].copy()
        test_df["split"] = "test"
        fold_df = pd.concat([fold_df, test_df], ignore_index=True)
        fold_df.to_csv(target_dir / f"{name}_fold{fold}.csv", index=False)

        logger.info(
            f"length of {name}_fold{fold}: {len(fold_df)}; "
            f"length of train: {len(fold_df[fold_df['split'] == 'train'])}; "
            f"length of valid: {len(fold_df[fold_df['split'] == 'valid'])}; "
            f"length of test: {len(fold_df[fold_df['split'] == 'test'])}"
        )


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )


def scaffold_split(
    smiles_list: List[str],
    k: int = 5,
    balanced: bool = True,
    seed: int = 42,
) -> Tuple[List[Tuple[List[int], List[int]]], List[str]]:
    all_scaffolds = {}
    scaffolds = []
    valid_indices = []

    for idx, smiles in enumerate(tqdm(smiles_list, desc="generating scaffolds")):
        try:
            scaffold = generate_scaffold(smiles, include_chirality=True)
        except Exception:
            logger.warning(f"Error generating scaffold for {smiles}")
            continue
        scaffolds.append(scaffold)
        valid_indices.append(idx)
        all_scaffolds.setdefault(scaffold, []).append(idx)

    if len(valid_indices) != len(smiles_list):
        raise ValueError("Some SMILES failed scaffold generation; please inspect logs.")

    scaffold_sets = list(all_scaffolds.values())
    if balanced:
        random.seed(seed)
        random.shuffle(scaffold_sets)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(scaffold_sets):
        train_fold = []
        val_fold = []
        for idx in train_idx:
            train_fold.extend(scaffold_sets[idx])
        for idx in val_idx:
            val_fold.extend(scaffold_sets[idx])
        folds.append((train_fold, val_fold))

    return folds, scaffolds


def scaffold_split_df(
    df: pd.DataFrame,
    name: str,
    save_dir: Path,
    k: int = 5,
    balanced: bool = True,
    seed: int = 42,
):
    smiles_list = df["smiles"].tolist()
    folds, scaffolds = scaffold_split(smiles_list, k=k, balanced=balanced, seed=seed)
    df = df.copy()
    df["scaffold"] = scaffolds

    target_dir = save_dir / "scaffold"
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"number of scaffolds: {df['scaffold'].nunique()}")
    logger.info(f"number of smiles: {df['smiles'].nunique()}")

    for fold, (train_idx, val_idx) in enumerate(folds):
        fold_df = df.copy()
        fold_df.loc[val_idx, "split"] = "valid"
        fold_df.loc[train_idx, "split"] = "train"
        test_df = fold_df[fold_df["split"] == "valid"].copy()
        test_df["split"] = "test"
        fold_df = pd.concat([fold_df, test_df], ignore_index=True)
        fold_df.to_csv(target_dir / f"{name}_fold{fold}.csv", index=False)

        logger.info(
            f"length of {name}_fold{fold}: {len(fold_df)}; "
            f"length of train: {len(fold_df[fold_df['split'] == 'train'])}; "
            f"length of valid: {len(fold_df[fold_df['split'] == 'valid'])}; "
            f"length of test: {len(fold_df[fold_df['split'] == 'test'])}"
        )


def main():
    args = parse_args()
    for dataset_name in args.datasets:
        df = load_dataset(args.raw_data_dir, dataset_name)
        df = drop_duplicates(df, ["smiles", "solvent"])
        random_split(df, args.raw_data_dir, dataset_name, args.n_splits, args.seed)
        scaffold_split_df(df, dataset_name, args.raw_data_dir, args.n_splits, True, args.seed)


if __name__ == "__main__":
    main()
