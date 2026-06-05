"""
RandomForest.py — 风机多输出回归基线（有功功率 + 发电机转速）

用法:
    python prediction/RandomForest.py
    python prediction/RandomForest.py --data 标注的数据-#67_1.xlsx --out-dir rf_output
    python prediction/RandomForest.py --show   # 弹窗显示图表（默认只保存到文件）
"""


from pathlib import Path as _Path
import os as _os
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
_os.chdir(_PROJECT_ROOT)

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core import COL_MAP, load_dataframe

POWER_PF = 0.95
NUM_COLS = ["exog_temp", "exog_wind", "gen_speed", "I_A", "I_B", "I_C", "V_A", "V_B", "V_C"]
FEATURE_COLS = ["exog_temp", "exog_wind", "temp_roll_3", "wind_roll_3"]
TARGET_COLS = ["P_est_kW", "gen_speed"]
TEST_RATIO = 0.20
RF_N_ESTIMATORS = 200
RANDOM_STATE = 42


def load_and_prepare(data_path: str) -> pd.DataFrame:
    # 数据读取/清洗复用 core.load_dataframe；这里只补充本脚本特有的特征
    df = load_dataframe(data_path, required=NUM_COLS, numeric=NUM_COLS, col_map=COL_MAP)

    # 近似有功功率：三相电压×电流之和乘以功率因数
    df["P_est_W"] = POWER_PF * (
        df["V_A"] * df["I_A"] + df["V_B"] * df["I_B"] + df["V_C"] * df["I_C"]
    )
    df["P_est_kW"] = df["P_est_W"] / 1000.0
    df["temp_roll_3"] = df["exog_temp"].rolling(3, min_periods=1).mean()
    df["wind_roll_3"] = df["exog_wind"].rolling(3, min_periods=1).mean()

    return df.dropna(subset=FEATURE_COLS + TARGET_COLS).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_ratio: float = TEST_RATIO):
    n = len(df)
    test_size = int(n * test_ratio)
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    return train_df, test_df


def build_pipeline(n_estimators: int = RF_N_ESTIMATORS) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "reg",
            MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
        ),
    ])


def regression_report(y_true, y_pred, names):
    reports = {}
    for i, name in enumerate(names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        reports[name] = {
            "MAE": mean_absolute_error(y_t, y_p),
            "RMSE": np.sqrt(mean_squared_error(y_t, y_p)),
            "R2": r2_score(y_t, y_p),
        }
    return reports


def plot_predictions(test_df, y_test, y_pred, out_path: Path, show: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(test_df["time"], y_test[:, 0], label="P_true_kW")
    axes[0].plot(test_df["time"], y_pred[:, 0], label="P_pred_kW", alpha=0.8)
    axes[0].set_title("有功功率: 真实 vs 预测")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].plot(test_df["time"], y_test[:, 1], label="gen_speed_true")
    axes[1].plot(test_df["time"], y_pred[:, 1], label="gen_speed_pred", alpha=0.8)
    axes[1].set_title("发电机转速: 真实 vs 预测")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"已保存预测对比图: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="风机多输出随机森林回归基线")
    parser.add_argument(
        "--data",
        default="标注的数据-#67_1.xlsx",
        help="输入 Excel 数据路径",
    )
    parser.add_argument(
        "--out-dir",
        default="rf_output",
        help="输出目录（指标 CSV、预测 CSV、对比图）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存图表后弹窗显示",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_model = load_and_prepare(args.data)
    train_df, test_df = time_split(df_model)

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COLS].values
    y_test = test_df[TARGET_COLS].values

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    reports = regression_report(y_test, y_pred, TARGET_COLS)
    print("回归模型评估（按时间切分，最后 20% 为测试集）:")
    for name, metrics in reports.items():
        print(
            f"  {name}: MAE={metrics['MAE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}"
        )

    metrics_df = pd.DataFrame.from_dict(reports, orient="index")
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"已保存指标: {metrics_path}")

    pred_df = test_df[["time"] + FEATURE_COLS + TARGET_COLS].copy()
    for i, col in enumerate(TARGET_COLS):
        pred_df[f"{col}_pred"] = y_pred[:, i]
    pred_path = out_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"已保存测试集预测: {pred_path}")

    plot_predictions(
        test_df,
        y_test,
        y_pred,
        out_dir / "prediction_comparison.png",
        show=args.show,
    )


if __name__ == "__main__":
    main()
