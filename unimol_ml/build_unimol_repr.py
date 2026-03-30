import pandas as pd
import pickle
import math
from pathlib import Path
from typing import List, Dict

from rdkit import Chem
from loguru import logger
from tqdm import tqdm

from unimol_tools import UniMolRepr
from unimol_tools.predictor import MolDataset
from unimol_tools.tasks import Trainer
from unimol_tools.data.conformer import UniMolV2Feature


def chunk_list(data: List[str], chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield i // chunk_size, data[i : i + chunk_size]


# ================== 主逻辑 ==================

if __name__ == "__main__":
    # -------- 路径 --------
    data_path = Path("../datasets/raw/scaffold/consolidation_fold0.csv")
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    final_save_path = Path("./smiles2embedding.pkl")

    # -------- 参数 --------
    batch_size = 64  # UniMol 推理批大小
    chunk_size = 2000  # 每个中间文件处理多少个 SMILES

    # -------- 读取数据 --------
    df = pd.read_csv(data_path)
    smiles_list = df["smiles"].astype(str).tolist()
    logger.info(f"Read {len(smiles_list)} SMILES strings")

    # -------- 初始化 UniMol --------
    repr_model = UniMolRepr(
        data_type="molecule",
        remove_hs=True,
        model_name="unimolv2",
        batch_size=batch_size,
    )

    unimol_feature = UniMolV2Feature(**repr_model.params)
    trainer = Trainer(task="repr", **repr_model.params)

    # ================== 分批处理并保存 ==================
    for chunk_idx, chunk_smiles in chunk_list(smiles_list, chunk_size):
        tmp_file = tmp_dir / f"chunk_{chunk_idx:04d}.pkl"
        if tmp_file.exists():
            logger.info(f"Skip existing chunk {tmp_file}")
            continue

        logger.info(f"Processing chunk {chunk_idx}, size={len(chunk_smiles)}")

        valid_smiles: List[str] = []
        features = []

        for smi in tqdm(chunk_smiles, desc=f"chunk {chunk_idx}"):
            try:
                feat, mol = unimol_feature.single_process(smi)
            except Exception as e:
                logger.warning(f"Failed processing SMILES={smi}: {e}")
                continue
            valid_smiles.append(smi)
            features.append(feat)

        if not features:
            logger.warning(f"Chunk {chunk_idx} has no valid molecules, skip")
            continue

        dataset = MolDataset(features)
        embeddings = trainer.inference(
            repr_model.model,
            return_repr=True,
            return_atomic_reprs=False,
            dataset=dataset,
        )

        chunk_result: Dict[str, object] = {
            smi: emb for smi, emb in zip(valid_smiles, embeddings["cls_repr"])
        }

        with open(tmp_file, "wb") as f:
            pickle.dump(chunk_result, f)

        logger.info(f"Saved {len(chunk_result)} embeddings to {tmp_file}")

    # ================== 合并所有 chunk ==================
    logger.info("Merging chunk files...")

    smiles2embedding: Dict[str, object] = {}
    chunk_files = sorted(tmp_dir.glob("chunk_*.pkl"))

    for pkl_file in chunk_files:
        with open(pkl_file, "rb") as f:
            part = pickle.load(f)
        smiles2embedding.update(part)

    with open(final_save_path, "wb") as f:
        pickle.dump(smiles2embedding, f)

    logger.info(f"Final embeddings: {len(smiles2embedding)} saved to {final_save_path}")
