
from pathlib import Path as _Path
import os as _os
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
_os.chdir(_PROJECT_ROOT)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 读取数据
DIR = "out_cnn_lstm_grid_search_5/run_002/" 
df = pd.read_csv(DIR + "test_predictions.csv", parse_dates=["time"])

# 设置时间为索引（可选）
df.set_index("time", inplace=True)

# 画图
plt.figure(figsize=(14, 6))

plt.plot(df.index, df["OT_true"], label="OT True",linewidth=2)
plt.plot(df.index, df["OT_pred"], label="OT Pred", linestyle="--", linewidth=2)

plt.title("OT True vs OT Pred over Time", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("OT Value", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()


plt.savefig("image/OT_true_vs_pred.png")
plt.close()
