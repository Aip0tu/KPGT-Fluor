import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


def main(
    input_path: str = "smiles2embedding.pkl",
    output_path: str = "scaled_smiles2embedding.pkl",
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info(f"Loading embeddings from: {input_path}")
    smiles2emb = joblib.load(input_path)

    if not isinstance(smiles2emb, dict):
        raise ValueError("Expected a dict: {smiles: embedding}")

    smiles = []
    embeddings = []

    for smi, emb in smiles2emb.items():
        if emb is None:
            continue
        smiles.append(smi)
        embeddings.append(np.asarray(emb))

    embeddings = np.vstack(embeddings)
    logger.info(f"Loaded {len(smiles)} embeddings with shape {embeddings.shape}")

    # ======================
    # 标准化 UniMol 向量
    # ======================
    logger.info("Fitting StandardScaler on embeddings")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # ======================
    # 重新构建 smiles -> embedding 映射
    # ======================
    scaled_smiles2emb = {}
    for smi, emb_scaled in zip(smiles, embeddings_scaled):
        scaled_smiles2emb[smi] = emb_scaled.astype(np.float32)

    logger.info(f"Saving scaled embeddings to: {output_path}")
    joblib.dump(
        {
            "embeddings": scaled_smiles2emb,
            "scaler": scaler,  # 保留 scaler，便于推理阶段对新样本复用同一套变换。
        },
        output_path,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
