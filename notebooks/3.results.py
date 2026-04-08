import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute simple regression metrics from prediction CSV files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Prediction CSV files containing `label` and `predictions` columns.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    df_results = pd.concat(frames, axis=0, ignore_index=True)

    print(f"num_rows={len(df_results)}")
    print(f"r2={r2_score(df_results['label'], df_results['predictions'])}")
    print(f"mae={mean_absolute_error(df_results['label'], df_results['predictions'])}")
    print(
        f"rmse={np.sqrt(mean_squared_error(df_results['label'], df_results['predictions']))}"
    )


if __name__ == "__main__":
    main()
