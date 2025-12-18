import lightgbm as lgb
import pandas as pd
from pathlib import Path
import numpy as np
import scipy.sparse as sp


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from loguru import logger
import sys
import joblib
from sklearn.metrics import mean_squared_error, r2_score

import sys
sys.path.append("../")
from src.data.finetune_dataset import MoleculeDataset

logger.remove()
logger.add(sys.stdout, level="DEBUG")


class UnimolReprResource:
    def __init__(self, map_path: Path):
        self.map_path = map_path
        self.smiles2embedding = joblib.load(map_path)["embeddings"]
        self.scaler = joblib.load(map_path)["scaler"]

    def get(self, smiles: str):
        return self.smiles2embedding.get(smiles, None)


def get_model(model_name: str, random_state: int = 42, verbose: bool = True):
    """
    Returns the specified regression model based on the model_name.

    :param model_name: Model name as string ('rf', 'svr', 'lightgbm', 'gbrt')
    :param random_state: Random state for reproducibility
    :param verbose: Whether to display detailed logs for the model (for models that support it)

    :return: A model instance
    """
    if model_name == "rf":
        return RandomForestRegressor(
            random_state=random_state, verbose=verbose, n_estimators=500, n_jobs=4
        )
    elif model_name == "svr":
        return SVR()  # SVR doesn't take random_state directly
    elif model_name == "lightgbm":
        return lgb.LGBMRegressor(random_state=random_state, n_estimators=500, n_jobs=4)
    elif model_name == "gbrt":
        return GradientBoostingRegressor(
            random_state=random_state,
            verbose=verbose,
            n_estimators=500,
        )
    else:
        raise ValueError(f"Model {model_name} not found")


def run_single_task(
    data_dir: str,
    split_method: str,
    dataset_name: str,
    fold: int,
    task_name: str,
    save_dir: str,
    model_name: str,
    model_save_dir: str,
):
    logger.info("Loading datasets (train and validation)")

    train_dataset = MoleculeDataset(
        root_path=data_dir,
        dataset=task_name,
        dataset_type="regression",
        split_name="splits",
        split="train",
    )

    val_dataset = MoleculeDataset(
        root_path=data_dir,
        dataset=task_name,
        dataset_type="regression",
        split_name="splits",
        split="val",
    )

    unimol_repr_resource = UnimolReprResource(
        map_path="./scaled_smiles2embedding.pkl"
    )

    # =======================
    # Train set processing
    # =======================
    (
        raw_smiles,
        raw_solvent,
        raw_graphs,
        raw_fps,
        raw_mds,
        raw_sds,
        raw_labels,
    ) = map(list, zip(*train_dataset))

    train_smiles = []
    train_solvent = []
    train_embeddings = []
    train_fps = []
    train_mds = []
    train_sds = []
    train_labels = []

    for smi, solv, fp, md, sd, y in zip(
        raw_smiles, raw_solvent, raw_fps, raw_mds, raw_sds, raw_labels
    ):
        emb = unimol_repr_resource.get(smi)
        if emb is None:
            continue  # 过滤掉没有 Unimol 表征的样本

        train_smiles.append(smi)
        train_solvent.append(solv)
        train_embeddings.append(emb)
        train_fps.append(fp)
        train_mds.append(md)
        train_sds.append(sd)
        train_labels.append(y)

    if len(train_labels) == 0:
        raise RuntimeError("No valid training samples after filtering None embeddings.")

    X_train = np.hstack(
        [
            np.vstack(train_embeddings),
            np.vstack(train_fps),
            np.vstack(train_mds),
            np.vstack(train_sds),
        ]
    )
    y_train = np.hstack(train_labels)

    assert X_train.shape[0] == y_train.shape[0]
    logger.debug(f"Train shape: {X_train.shape}, {y_train.shape}")

    # =======================
    # Model training
    # =======================
    logger.info(f"Training {model_name} model")
    model = get_model(model_name)
    model.fit(X_train, y_train)

    save_path = model_save_dir / f"{task_name}.pkl"
    joblib.dump(model, save_path)
    logger.info(f"Model saved to {save_path}")

    preds_train = model.predict(X_train)

    # =======================
    # Validation set processing
    # =======================
    (
        raw_val_smiles,
        raw_val_solvent,
        raw_val_graphs,
        raw_val_fps,
        raw_val_mds,
        raw_val_sds,
        raw_val_labels,
    ) = map(list, zip(*val_dataset))

    val_smiles = []
    val_solvent = []
    val_embeddings = []
    val_fps = []
    val_mds = []
    val_sds = []
    val_labels = []

    for smi, solv, fp, md, sd, y in zip(
        raw_val_smiles, raw_val_solvent, raw_val_fps, raw_val_mds, raw_val_sds, raw_val_labels
    ):
        emb = unimol_repr_resource.get(smi)
        if emb is None:
            continue

        val_smiles.append(smi)
        val_solvent.append(solv)
        val_embeddings.append(emb)
        val_fps.append(fp)
        val_mds.append(md)
        val_sds.append(sd)
        val_labels.append(y)

    if len(val_labels) == 0:
        raise RuntimeError("No valid validation samples after filtering None embeddings.")

    X_val = np.hstack(
        [
            np.vstack(val_embeddings),
            np.vstack(val_fps),
            np.vstack(val_mds),
            np.vstack(val_sds),
        ]
    )
    y_val = np.hstack(val_labels)

    assert X_val.shape[0] == y_val.shape[0]
    logger.debug(f"Validation shape: {X_val.shape}, {y_val.shape}")

    preds_val = model.predict(X_val)

    # =======================
    # Evaluation
    # =======================
    train_rmse = np.sqrt(mean_squared_error(y_train, preds_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    train_r2 = r2_score(y_train, preds_train)
    val_r2 = r2_score(y_val, preds_val)

    logger.info(f"Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}")
    logger.info(f"Validation RMSE: {val_rmse:.4f}, Validation R²: {val_r2:.4f}")

    # =======================
    # Save predictions
    # =======================
    df_train = pd.DataFrame(
        {
            "smiles": train_smiles,
            "solvent": train_solvent,
            "label": y_train,
            "predictions": preds_train,
        }
    )

    df_val = pd.DataFrame(
        {
            "smiles": val_smiles,
            "solvent": val_solvent,
            "label": y_val,
            "predictions": preds_val,
        }
    )

    df_train.to_csv(save_dir / "train.csv", index=False)
    df_val.to_csv(save_dir / "valid.csv", index=False)

    logger.info("Finished run_single_task successfully")

if __name__ == "__main__":
    for fold in range(5):
        split_method = "scaffold"
        data_dir = "../datasets"
        dataset_name = "consolidation"  # "consolidation", "cyanine", "xanthene"
        task_name = "log_molar_absorptivity"  # "absorption" "emission", "quantum_yield", "log_molar_absorptivity"
        model_name = "lightgbm"  # "rf", "svr", "lightgbm", "gbrt"

        save_dir = Path(
            f"results_{model_name}/{split_method}/{dataset_name}_fold{fold}/{task_name}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        model_save_dir = Path(
            f"models/{model_name}/{split_method}/{dataset_name}_fold{fold}"
        )
        model_save_dir.mkdir(parents=True, exist_ok=True)
        task_dir = Path(data_dir) / split_method / f"{dataset_name}_fold{fold}"
        run_single_task(
            data_dir=task_dir,
            split_method=split_method,
            dataset_name=dataset_name,
            fold=fold,
            task_name=task_name,
            save_dir=save_dir,
            model_name=model_name,
            model_save_dir=model_save_dir,
        )
