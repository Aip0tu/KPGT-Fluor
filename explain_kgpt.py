import argparse
import copy
import logging
from pathlib import Path
import re

import dgl
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.collator import Collator_tune
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from src.data.featurizer import N_ATOM_TYPES, N_BOND_TYPES, Vocab, smiles_to_graph_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict
from src.utils import get_device, set_random_seed


TASK_CHOICES = [
    "absorption",
    "emission",
    "quantum_yield",
    "log_molar_absorptivity",
]

EXAMPLE_FLUOROPHORES = [
    "Cc1ccc(C(=O)c2cc(C(=O)O)cc3c2CCN3c2c(Cl)cccc2Cl)cc1",
    "Cc1ccc(C(=O)c2cc(C(=O)O)cc3c2CCN3c2c(Cl)cccc2Cl)cc1",
]

EXAMPLE_SOLVENTS = [
    "ClCCl",
    "O",
]


def safe_file_stem(text, max_len=80):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    if not stem:
        stem = "molecule"
    return stem[:max_len]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run atom-level masking explainability for KPGT-Fluor."
    )
    parser.add_argument(
        "--model-root",
        default="models/downstream",
        help="Root directory containing trained downstream checkpoints.",
    )
    parser.add_argument(
        "--data-root",
        default="datasets",
        help="Root directory containing fold CSV files for normalization statistics.",
    )
    parser.add_argument("--split-method", default="random")
    parser.add_argument(
        "--dataset-name",
        default="consolidation",
        help="Training dataset used to fit the checkpoints.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=TASK_CHOICES,
        choices=TASK_CHOICES,
        help="Targets to explain together.",
    )
    parser.add_argument("--fluorophore", help="Single fluorophore SMILES.")
    parser.add_argument("--fluorophores", nargs="+", help="One or more fluorophore SMILES.")
    parser.add_argument("--solvents", nargs="+", help="One solvent SMILES per fluorophore.")
    parser.add_argument("--use_examples", action="store_true")
    parser.add_argument("--solvent", default="O", help="Default solvent SMILES.")
    parser.add_argument(
        "--atom-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of fluorophore atom indices to explain.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Checkpoint folds used for ensemble explanation.",
    )
    parser.add_argument("--config", default="base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--n-predictor-layers", type=int, default=2)
    parser.add_argument("--d-predictor-hidden", type=int, default=256)
    parser.add_argument("--output-dir", default="results/kgpt_explain")
    parser.add_argument("--prefix", default="kgpt_atom_explain")
    parser.add_argument("--no_images", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
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


def model_path_for(args, task, fold):
    return (
        Path(args.model_root)
        / args.split_method
        / f"{args.dataset_name}_fold{fold}"
        / f"{task}.pth"
    )


def stats_csv_for(args, task, fold):
    return (
        Path(args.data_root)
        / args.split_method
        / f"{args.dataset_name}_fold{fold}"
        / task
        / f"{task}.csv"
    )


def get_train_stats(csv_path, task):
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError(f"Missing `split` column in {csv_path}")
    if task not in df.columns:
        raise ValueError(f"Missing task column `{task}` in {csv_path}")
    values = df.loc[df["split"] == "train", task].to_numpy(dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"No train rows found in {csv_path}")
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(mean):
        raise ValueError(f"Non-finite mean for {task} in {csv_path}")
    if not np.isfinite(std) or std == 0:
        std = 1.0
    return mean, std


def get_atom_metadata(mol, atom_index):
    atom = mol.GetAtomWithIdx(atom_index)
    return {
        "atom_index": atom_index,
        "atom_symbol": atom.GetSymbol(),
        "atomic_num": atom.GetAtomicNum(),
        "is_aromatic": bool(atom.GetIsAromatic()),
        "formal_charge": atom.GetFormalCharge(),
        "degree": atom.GetDegree(),
    }


def score_to_color(score, max_abs_score):
    if max_abs_score <= 0:
        return (0.85, 0.85, 0.85)
    intensity = min(abs(score) / max_abs_score, 1.0)
    fade = 0.75 * intensity
    if score >= 0:
        return (1.0, 1.0 - fade, 1.0 - fade)
    return (1.0 - fade, 1.0 - fade, 1.0)


def draw_atom_attribution(smiles, attribution, output_path, score_column, legend=""):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid fluorophore SMILES: {smiles}")

    scores = attribution.set_index("atom_index")[score_column].to_dict()
    max_abs_score = max((abs(float(v)) for v in scores.values()), default=0.0)

    draw_mol = Chem.Mol(mol)
    highlight_atoms = []
    highlight_colors = {}
    highlight_radii = {}

    for atom in draw_mol.GetAtoms():
        atom_index = atom.GetIdx()
        score = float(scores.get(atom_index, 0.0))
        atom.SetProp("atomNote", f"{atom_index}:{score:.2f}")
        highlight_atoms.append(atom_index)
        highlight_colors[atom_index] = score_to_color(score, max_abs_score)
        scale = abs(score) / max_abs_score if max_abs_score else 0.0
        highlight_radii[atom_index] = 0.32 + 0.22 * scale

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 700)
    options = drawer.drawOptions()
    options.addAtomIndices = False
    options.legendFontSize = 24
    drawer.DrawMolecule(
        draw_mol,
        legend=legend,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(drawer.GetDrawingText())
    return str(output_path)


class SampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_feature_tensors(smiles, solvent_smiles, path_length):
    fluorophore = Chem.MolFromSmiles(smiles)
    solvent = Chem.MolFromSmiles(solvent_smiles)
    if fluorophore is None:
        raise ValueError(f"Invalid fluorophore SMILES: {smiles}")
    if solvent is None:
        raise ValueError(f"Invalid solvent SMILES: {solvent_smiles}")

    graph = smiles_to_graph_tune(smiles, max_length=path_length, n_virtual_nodes=3)
    if graph is None:
        raise ValueError(f"Failed to build graph for fluorophore: {smiles}")
    if "atoms" not in graph.ndata:
        raise ValueError("Graph is missing atom mapping metadata.")

    fp = torch.from_numpy(
        np.asarray(
            list(Chem.RDKFingerprint(fluorophore, minPath=1, maxPath=7, fpSize=512)),
            dtype=np.float32,
        )
    )
    generator = RDKit2DNormalized()
    md = generator.process(smiles)
    sd = generator.process(solvent_smiles)
    if md is None:
        raise ValueError(f"Failed to compute molecular descriptors for: {smiles}")
    if sd is None:
        raise ValueError(f"Failed to compute solvent descriptors for: {solvent_smiles}")

    md = torch.from_numpy(np.where(np.isnan(np.asarray(md[1:], dtype=np.float32)), 0, np.asarray(md[1:], dtype=np.float32)))
    sd = torch.from_numpy(np.where(np.isnan(np.asarray(sd[1:], dtype=np.float32)), 0, np.asarray(sd[1:], dtype=np.float32)))
    label = torch.zeros(1, dtype=torch.float32)
    return fluorophore, (smiles, solvent_smiles, graph, fp, md, sd, label)


def mask_graph_for_atom(graph, atom_index):
    masked_graph = copy.deepcopy(graph)
    triplet_atoms = masked_graph.ndata["atoms"]
    begin_end = masked_graph.ndata["begin_end"]
    edge = masked_graph.ndata["edge"]

    match = triplet_atoms == atom_index
    for slot in range(match.shape[1]):
        slot_mask = match[:, slot]
        if torch.any(slot_mask):
            begin_end[slot_mask, slot, :] = 0

    node_mask = torch.any(match, dim=1)
    if torch.any(node_mask):
        edge[node_mask, :] = 0

    return masked_graph


def build_masked_samples(base_sample, atom_indices):
    smiles, solvent_smiles, base_graph, fp, md, sd, label = base_sample
    samples = [
        (
            smiles,
            solvent_smiles,
            copy.deepcopy(base_graph),
            fp.clone(),
            md.clone(),
            sd.clone(),
            label.clone(),
        )
    ]
    for atom_index in atom_indices:
        samples.append(
            (
                smiles,
                solvent_smiles,
                mask_graph_for_atom(base_graph, atom_index),
                fp.clone(),
                md.clone(),
                sd.clone(),
                label.clone(),
            )
        )
    return samples


def build_model(args, feature_dims, device, checkpoint_path):
    config = config_dict[args.config]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    d_fp_feats, d_md_feats, d_sd_feats = feature_dims
    model = LiGhT(
        d_node_feats=config["d_node_feats"],
        d_edge_feats=config["d_edge_feats"],
        d_g_feats=config["d_g_feats"],
        d_fp_feats=d_fp_feats,
        d_md_feats=d_md_feats,
        d_sd_feats=d_sd_feats,
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
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state_dict.items()},
        strict=False,
    )
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    model.eval()
    return model


def predict_samples(model, samples, path_length, batch_size, num_workers, device):
    dataloader = DataLoader(
        SampleDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=Collator_tune(path_length),
    )
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


def explain_task(args, fluorophore_smiles, solvent_smiles, task, device, model_cache):
    config = config_dict[args.config]
    fluorophore, base_sample = build_feature_tensors(
        fluorophore_smiles, solvent_smiles, config["path_length"]
    )
    all_atom_indices = list(range(fluorophore.GetNumAtoms()))
    atom_indices = all_atom_indices if args.atom_indices is None else list(args.atom_indices)
    invalid_indices = sorted(set(atom_indices) - set(all_atom_indices))
    if invalid_indices:
        raise ValueError(f"Atom indices out of range: {invalid_indices}")

    samples = build_masked_samples(base_sample, atom_indices)
    feature_dims = (
        base_sample[3].shape[0],
        base_sample[4].shape[0],
        base_sample[5].shape[0],
    )

    fold_predictions = []
    for fold in args.folds:
        checkpoint_path = model_path_for(args, task, fold)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")
        stats_csv = stats_csv_for(args, task, fold)
        if not stats_csv.exists():
            raise FileNotFoundError(f"Missing statistics CSV: {stats_csv}")

        cache_key = (task, fold)
        if cache_key not in model_cache:
            model_cache[cache_key] = build_model(args, feature_dims, device, checkpoint_path)
        model = model_cache[cache_key]

        mean, std = get_train_stats(stats_csv, task)
        normalized_predictions = predict_samples(
            model,
            samples,
            config["path_length"],
            args.batch_size,
            args.num_workers,
            device,
        )
        fold_predictions.append(normalized_predictions * std + mean)

    fold_predictions = np.asarray(fold_predictions, dtype=np.float32)
    avg_predictions = fold_predictions.mean(axis=0)
    baseline = float(avg_predictions[0])
    masked_predictions = avg_predictions[1:]

    rows = []
    for row_index, atom_index in enumerate(atom_indices):
        masked_prediction = float(masked_predictions[row_index])
        contribution = baseline - masked_prediction
        row = get_atom_metadata(fluorophore, atom_index)
        row[f"baseline_{task}"] = baseline
        row[f"masked_{task}"] = masked_prediction
        row[f"contribution_{task}"] = contribution
        row["abs_contribution"] = abs(contribution)
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["importance_rank"] = (
            result["abs_contribution"].rank(method="dense", ascending=False).astype(int)
        )
    return result, baseline


def explain_all_targets(args, fluorophore_smiles, solvent_smiles, device, model_cache):
    metadata_columns = [
        "atom_index",
        "atom_symbol",
        "atomic_num",
        "is_aromatic",
        "formal_charge",
        "degree",
    ]

    merged = None
    baselines = {}
    per_target_tables = {}
    for target in args.targets:
        target_df, baseline = explain_task(
            args, fluorophore_smiles, solvent_smiles, target, device, model_cache
        )
        per_target_tables[target] = target_df.copy()
        baselines[target] = baseline

        keep_columns = metadata_columns + [
            f"baseline_{target}",
            f"masked_{target}",
            f"contribution_{target}",
            "abs_contribution",
            "importance_rank",
        ]
        target_df = target_df.loc[:, keep_columns].rename(
            columns={
                "abs_contribution": f"abs_contribution_{target}",
                "importance_rank": f"importance_rank_{target}",
            }
        )
        if merged is None:
            merged = target_df
        else:
            merged = merged.merge(target_df, on=metadata_columns, how="inner")

    if merged is None:
        raise ValueError("No targets were provided.")

    merged.insert(0, "solvent_smiles", solvent_smiles)
    merged.insert(0, "fluorophore_smiles", fluorophore_smiles)

    abs_columns = [f"abs_contribution_{target}" for target in args.targets]
    merged["max_abs_contribution"] = merged[abs_columns].max(axis=1)
    merged["overall_importance_rank"] = (
        merged["max_abs_contribution"].rank(method="dense", ascending=False).astype(int)
    )
    return merged, per_target_tables, baselines


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    set_random_seed(args.seed, args.n_threads)
    device = torch.device("cpu") if args.no_cuda else get_device()
    logging.info("Using device: %s", device)

    fluorophores = []
    if args.fluorophore:
        fluorophores.append(args.fluorophore)
    if args.fluorophores:
        fluorophores.extend(args.fluorophores)

    jobs = []
    if fluorophores:
        if args.solvents is not None:
            if len(args.solvents) != len(fluorophores):
                raise ValueError("The number of solvents must match the number of fluorophores.")
            jobs.extend(zip(fluorophores, args.solvents))
        else:
            jobs.extend((fluorophore, args.solvent) for fluorophore in fluorophores)

    if args.use_examples or not fluorophores:
        jobs.extend(zip(EXAMPLE_FLUOROPHORES, EXAMPLE_SOLVENTS))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    image_dirs = []
    model_cache = {}
    for idx, (fluorophore, solvent_smiles) in enumerate(jobs, start=1):
        logging.info(
            "Explaining fluorophore %d/%d", idx, len(jobs)
        )
        merged, per_target_tables, baselines = explain_all_targets(
            args,
            fluorophore,
            solvent_smiles,
            device,
            model_cache,
        )
        tables.append(merged)

        if not args.no_images:
            example_dir = output_dir / f"{idx:02d}_{safe_file_stem(fluorophore)}"
            example_dir.mkdir(parents=True, exist_ok=True)
            image_dirs.append(str(example_dir))
            for target, target_df in per_target_tables.items():
                draw_atom_attribution(
                    smiles=fluorophore,
                    attribution=target_df,
                    output_path=example_dir / f"{target}.png",
                    score_column=f"contribution_{target}",
                    legend=f"{target}: {baselines[target]:.3f}",
                )

    result = pd.concat(tables, ignore_index=True)
    csv_path = output_dir / f"{args.prefix}.csv"
    result.to_csv(csv_path, index=False)

    rank_columns = [f"importance_rank_{target}" for target in args.targets]
    preview_columns = [
        "fluorophore_smiles",
        "atom_index",
        "atom_symbol",
        "overall_importance_rank",
    ] + rank_columns
    print(
        result.loc[:, preview_columns]
        .sort_values(["fluorophore_smiles", "overall_importance_rank"])
        .head(30)
        .to_string(index=False)
    )
    print(f"\nSaved CSV to {csv_path}")
    if not args.no_images:
        print("Saved example images under:")
        for image_dir in image_dirs:
            print(image_dir)


if __name__ == "__main__":
    main()
