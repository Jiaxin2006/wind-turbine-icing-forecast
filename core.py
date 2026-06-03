"""
core.py — 各实验脚本共用的基础组件。

集中以下在多个脚本里重复的部分，避免逐字复制：
- COL_MAP 与 load_dataframe：读取 Excel、统一列名、排序、数值化、插值
- 指标函数：metric_table（8 项）/ metrics_dict（4 项）
- SeqDataset：序列模型的滑窗数据集
- 序列模型：LSTMReg / CNN1DReg / TransformerReg（支持返回 embedding）
- train_torch_model / predict_torch：通用训练与预测循环

注意：这里的序列模型与训练循环对应主实验 ensemble.py 的实现；
model.py 等使用的是更简单的基线结构，保留在各自脚本内，以保证已有结果可复现。
"""

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Excel 原始列名 -> 脚本内部列名（各脚本所需列的并集）
COL_MAP = {
    "time": "统计时间",
    "OT": "OT",
    "exog_temp": "Exogenous1",
    "exog_wind": "Exogenous2",
    "gen_speed": "平均发电机转速(rpm)",
    "I_A": "平均网侧A相电流(A)",
    "I_B": "平均网侧B相电流(A)",
    "I_C": "平均网侧C相电流(A)",
    "V_A": "平均网侧A相电压(V)",
    "V_B": "平均网侧B相电压(V)",
    "V_C": "平均网侧C相电压(V)",
}


def load_dataframe(data_path, required=("OT", "exog_temp", "exog_wind"),
                   numeric=None, col_map=COL_MAP):
    """读取 Excel，统一列名，按时间排序，对所需列数值化并插值填充。

    required: 必须存在的列；缺失则报错。
    numeric: 需要转为数值的列，默认与 required 相同。
    """
    numeric = list(required) if numeric is None else list(numeric)

    df = pd.read_excel(data_path)
    reverse_map = {v: k for k, v in col_map.items()}
    df = df.rename(columns={orig: new for orig, new in reverse_map.items() if orig in df.columns})

    if "time" not in df.columns:
        raise ValueError("找不到时间列，请检查 col_map 中 time 的映射。")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    for c in required:
        if c not in df.columns:
            raise ValueError(f"缺少必需列: {c}，请检查 col_map")
    for c in numeric:
        if c not in df.columns:
            raise ValueError(f"缺少需要数值化的列: {c}，请检查 col_map")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.interpolate(limit=5).bfill().ffill()


def _mape(y_true, y_pred, eps):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0


def mape(y_true, y_pred, eps=1e-9):
    return _mape(y_true, y_pred, eps)


def metric_table(y_true, y_pred, eps=1e-6):
    """单目标回归的 8 项指标（MAE/MSE/RMSE/NRMSE/R2/MAPE/SMAPE/MASE）。"""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    nrmse = rmse / np.var(y_true) + 1e-9
    r2 = r2_score(y_true, y_pred)
    mape_v = _mape(y_true, y_pred, eps)
    smape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)
    mase = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true - y_true.mean()) + eps)))
    return {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse,
        "R2": r2, "MAPE": mape_v, "SMAPE": smape, "MASE": mase,
    }


def metrics_dict(y_true, y_pred):
    """精简的 4 项指标（MAE/MSE/RMSE/MAPE%），用于主实验汇总。"""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE(%)": _mape(y_true, y_pred, 1e-9)}


class SeqDataset(Dataset):
    """滑窗序列：start_idx/end_idx 为闭区间，输入窗口后一步的 target 为标签。"""

    def __init__(self, df_full, start_idx, end_idx, seq_len, feat_cols, target_col="OT"):
        self.df = df_full
        self.start = start_idx
        self.end = end_idx
        self.seq_len = seq_len
        self.feat_cols = feat_cols
        self.target_col = target_col
        self.n = max(0, (self.end - self.start + 1) - self.seq_len)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        idx0 = self.start + idx
        seq = self.df.iloc[idx0: idx0 + self.seq_len][self.feat_cols].values.astype(np.float32)
        y = self.df.iloc[idx0 + self.seq_len][self.target_col].astype(np.float32)
        return seq, y


class LSTMReg(nn.Module):
    def __init__(self, input_dim, hid=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hid, n_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hid, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x, return_embedding=False):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        emb = torch.relu(self.fc1(last))
        y = self.out(emb)
        return (y, emb) if return_embedding else y


class CNN1DReg(nn.Module):
    def __init__(self, input_dim, hid=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hid, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hid, hid, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc1 = nn.Linear(hid, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x, return_embedding=False):
        x = x.permute(0, 2, 1)
        h = self.conv(x).squeeze(-1)
        emb = torch.relu(self.fc1(h))
        y = self.out(emb)
        return (y, emb) if return_embedding else y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divterm = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * divterm)
        pe[:, 1::2] = torch.cos(pos * divterm)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerReg(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(d_model, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x, return_embedding=False):
        x = self.input_fc(x)
        x = self.pos_enc(x)
        out = self.transformer(x)
        last = out[:, -1, :]
        emb = torch.relu(self.fc1(last))
        y = self.out(emb)
        return (y, emb) if return_embedding else y


def train_torch_model(model, train_loader, val_loader=None, epochs=40, lr=5e-4,
                      device=DEVICE, early_stopping=3):
    """通用训练循环：MSE 损失，按验证集 loss 早停并保存最优权重。"""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_state = None
    best_loss = float("inf")
    patience = 0
    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device).unsqueeze(1)
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple): out = out[0]
            loss = criterion(out, yb)
            loss.backward(); opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= max(1, len(train_loader.dataset))
        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device).unsqueeze(1)
                    out = model(xb)
                    if isinstance(out, tuple): out = out[0]
                    vloss += criterion(out, yb).item() * xb.size(0)
            val_loss = vloss / max(1, len(val_loader.dataset))
        if val_loss < best_loss - 1e-9:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_torch(model, loader, device=DEVICE):
    model = model.to(device)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            if isinstance(out, tuple): out = out[0]
            preds.append(out.cpu().numpy().ravel())
            trues.append(yb.numpy().ravel())
    if len(preds) == 0:
        return np.array([]), np.array([])
    return np.concatenate(trues), np.concatenate(preds)
