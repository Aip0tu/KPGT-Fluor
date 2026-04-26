from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.collator import Collator_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from src.data.featurizer import (
    N_ATOM_TYPES,
    N_BOND_TYPES,
    Vocab,
    smiles_to_graph_tune,
)
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict
from src.utils import get_device, set_random_seed


TASK_CHOICES = [
    "absorption",
    "emission",
    "quantum_yield",
    "log_molar_absorptivity",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict properties for datasets/smile.csv with trained KPGT-Fluor models."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="datasets/smile.csv",
        help="CSV file with at least `smiles` and `solvent` columns.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Prediction result CSV. Default: results/<input_stem>_predictions.csv",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_CHOICES,
        default=None,
        help="Tasks to predict. If omitted, auto-detect available tasks from model files.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Model folds used for ensemble prediction.",
    )
    parser.add_argument("--split-method", type=str, default="random")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="consolidation",
        help="Training dataset name used to locate model weights and label statistics.",
    )
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers. Default 0 to avoid multiprocessing issues.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="Torch CPU thread count.",
    )
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n-predictor-layers", type=int, default=2)
    parser.add_argument("--d-predictor-hidden", type=int, default=256)
    parser.add_argument(
        "--model-template",
        type=str,
        default="models/downstream/{split_method}/{dataset_name}_fold{fold}/{task}.pth",
    )
    parser.add_argument(
        "--stats-template",
        type=str,
        default="datasets/{split_method}/{dataset_name}_fold{fold}/{task}/{task}.csv",
        help="CSV used to recover the train-split mean/std for de-normalizing predictions.",
    )
    parser.add_argument(
        "--save-fold-columns",
        action="store_true",
        help="Also save one prediction column per fold.",
    )
    return parser.parse_args()


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_predictor(
    d_input_feats,
    n_tasks,
    n_layers,
    predictor_drop,
    device,
    d_hidden_feats=None,
):
    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers - 2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)


class SmileInferenceDataset(Dataset):
    def __init__(self, input_csv, path_length):
        self.input_csv = Path(input_csv)
        self.df = pd.read_csv(self.input_csv)
        required_columns = {"smiles", "solvent"}
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(
                "Input CSV is missing required columns: {}".format(
                    ", ".join(sorted(missing_columns))
                )
            )

        self.path_length = path_length
        self.descriptor_generator = RDKit2DNormalized()
        self.records = []
        self.valid_row_ids = []
        self.invalid_reasons = {}

        for row_id, row in tqdm(
            self.df.iterrows(),
            total=len(self.df),
            desc="Building inference features",
        ):
            smiles = str(row["smiles"]).strip()
            solvent = str(row["solvent"]).strip()
            record, reason = self._build_record(row_id, smiles, solvent)
            if record is None:
                self.invalid_reasons[row_id] = reason
                continue
            self.records.append(record)
            self.valid_row_ids.append(row_id)

        if self.records:
            self.d_fps = self.records[0][3].shape[0]
            self.d_mds = self.records[0][4].shape[0]
            self.d_sds = self.records[0][5].shape[0]
        else:
            self.d_fps = 512
            self.d_mds = 200
            self.d_sds = 200

    def _build_record(self, row_id, smiles, solvent):
        if not smiles or smiles.lower() == "nan":
            return None, "empty_smiles"
        if not solvent or solvent.lower() == "nan":
            return None, "empty_solvent"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "invalid_smiles"

        solvent_mol = Chem.MolFromSmiles(solvent)
        if solvent_mol is None:
            return None, "invalid_solvent"

        graph = smiles_to_graph_tune(
            smiles,
            max_length=self.path_length,
            n_virtual_nodes=3,
        )
        if graph is None:
            return None, "graph_build_failed"

        fp = np.asarray(
            list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)),
            dtype=np.float32,
        )
        md = self.descriptor_generator.process(smiles)
        sd = self.descriptor_generator.process(solvent)
        if md is None:
            return None, "molecular_descriptor_failed"
        if sd is None:
            return None, "solvent_descriptor_failed"

        md = np.asarray(md[1:], dtype=np.float32)
        sd = np.asarray(sd[1:], dtype=np.float32)
        md = np.where(np.isnan(md), 0, md)
        sd = np.where(np.isnan(sd), 0, sd)

        label = np.zeros(1, dtype=np.float32)
        return (
            smiles,
            solvent,
            graph,
            torch.from_numpy(fp),
            torch.from_numpy(md),
            torch.from_numpy(sd),
            torch.from_numpy(label),
        ), None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def detect_tasks(args):
    detected_tasks = []
    for task in TASK_CHOICES:
        model_paths = [
            Path(
                args.model_template.format(
                    split_method=args.split_method,
                    dataset_name=args.dataset_name,
                    fold=fold,
                    task=task,
                )
            )
            for fold in args.folds
        ]
        if all(path.exists() for path in model_paths):
            detected_tasks.append(task)
    return detected_tasks


def get_train_stats(stats_csv, task):
    df = pd.read_csv(stats_csv)
    if "split" not in df.columns:
        raise ValueError("Statistics CSV is missing `split` column: {}".format(stats_csv))
    if task not in df.columns:
        raise ValueError(
            "Statistics CSV is missing task column `{}`: {}".format(task, stats_csv)
        )
    train_values = df.loc[df["split"] == "train", task].to_numpy(dtype=np.float32)
    if train_values.size == 0:
        raise ValueError("No train rows found in statistics CSV: {}".format(stats_csv))
    return float(np.nanmean(train_values)), float(np.nanstd(train_values))


def build_model(args, dataset, device, task_model_path):
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    model = LiGhT(
        d_node_feats=config["d_node_feats"],
        d_edge_feats=config["d_edge_feats"],
        d_g_feats=config["d_g_feats"],
        d_fp_feats=dataset.d_fps,
        d_md_feats=dataset.d_mds,
        d_sd_feats=dataset.d_sds,
        d_hpath_ratio=config["d_hpath_ratio"],
        n_mol_layers=config["n_mol_layers"],
        path_length=config["path_length"],
        n_heads=config["n_heads"],
        n_ffn_dense_layers=config["n_ffn_dense_layers"],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size,
    ).to(device)
    model.predictor = get_predictor(
        d_input_feats=config["d_g_feats"] * 4,
        n_tasks=1,
        n_layers=args.n_predictor_layers,
        predictor_drop=0,
        device=device,
        d_hidden_feats=args.d_predictor_hidden,
    )
    state_dict = torch.load(task_model_path, map_location=device)
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state_dict.items()},
        strict=False,
    )
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    model.eval()
    return model


def predict_single_model(model, dataloader, device):
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", total=len(dataloader)):
            _, _, graph, fp, md, sd, _ = batch
            graph = graph.to(device)
            fp = fp.to(device)
            md = md.to(device)
            sd = sd.to(device)
            batch_predictions = model.forward_tune(graph, fp, md, sd)
            predictions.append(batch_predictions.detach().cpu().numpy())
    if not predictions:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(predictions, axis=0).reshape(-1)


def ensure_required_files(args, tasks):
    missing_files = []
    for task in tasks:
        for fold in args.folds:
            model_path = Path(
                args.model_template.format(
                    split_method=args.split_method,
                    dataset_name=args.dataset_name,
                    fold=fold,
                    task=task,
                )
            )
            stats_csv = Path(
                args.stats_template.format(
                    split_method=args.split_method,
                    dataset_name=args.dataset_name,
                    fold=fold,
                    task=task,
                )
            )
            if not model_path.exists():
                missing_files.append(str(model_path))
            if not stats_csv.exists():
                missing_files.append(str(stats_csv))
    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n{}".format("\n".join(missing_files))
        )


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_random_seed(args.seed, args.n_threads)
    device = get_device()
    logging.info("Using device: %s", device)

    if args.tasks is None:
        tasks = detect_tasks(args)
        if not tasks:
            raise FileNotFoundError(
                "No available tasks detected from model files. Use --tasks and check model paths."
            )
        logging.info("Auto-detected tasks: %s", ", ".join(tasks))
    else:
        tasks = args.tasks

    ensure_required_files(args, tasks)

    config = config_dict[args.config]
    dataset = SmileInferenceDataset(args.input_csv, config["path_length"])
    if len(dataset) == 0:
        output_path = (
            Path(args.output_csv)
            if args.output_csv
            else Path("results") / (Path(args.input_csv).stem + "_predictions.csv")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df = dataset.df.copy()
        output_df["prediction_status"] = "invalid_input"
        output_df["invalid_reason"] = output_df.index.map(dataset.invalid_reasons.get)
        output_df.to_csv(output_path, index=False)
        logging.warning("No valid input rows. Wrote invalid-row report to %s", output_path)
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Collator_tune(config["path_length"]),
    )

    output_df = dataset.df.copy()
    output_df["prediction_status"] = "ok"
    output_df["invalid_reason"] = ""
    for row_id, reason in dataset.invalid_reasons.items():
        output_df.at[row_id, "prediction_status"] = "invalid_input"
        output_df.at[row_id, "invalid_reason"] = reason

    valid_row_ids = np.asarray(dataset.valid_row_ids)

    for task in tasks:
        fold_predictions = []
        logging.info("Running task: %s", task)
        for fold in args.folds:
            model_path = Path(
                args.model_template.format(
                    split_method=args.split_method,
                    dataset_name=args.dataset_name,
                    fold=fold,
                    task=task,
                )
            )
            stats_csv = Path(
                args.stats_template.format(
                    split_method=args.split_method,
                    dataset_name=args.dataset_name,
                    fold=fold,
                    task=task,
                )
            )
            mean, std = get_train_stats(stats_csv, task)
            model = build_model(args, dataset, device, model_path)
            normalized_predictions = predict_single_model(model, dataloader, device)
            predictions = normalized_predictions * std + mean
            fold_predictions.append(predictions)
            if args.save_fold_columns:
                column_name = "{}_fold{}_pred".format(task, fold)
                output_df[column_name] = np.nan
                output_df.loc[valid_row_ids, column_name] = predictions

        fold_predictions = np.asarray(fold_predictions)
        output_df["{}_pred".format(task)] = np.nan
        output_df["{}_pred_std".format(task)] = np.nan
        output_df.loc[valid_row_ids, "{}_pred".format(task)] = fold_predictions.mean(
            axis=0
        )
        output_df.loc[valid_row_ids, "{}_pred_std".format(task)] = fold_predictions.std(
            axis=0
        )

    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else Path("results") / (Path(args.input_csv).stem + "_predictions.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logging.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()
