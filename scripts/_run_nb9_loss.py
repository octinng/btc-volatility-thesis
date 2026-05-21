"""One-off runner for notebooks/9. loss_curves_v2.ipynb (use my venv Python)."""
from __future__ import annotations

import copy
import json
import random
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PROJECT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT / "data" / "04_final_feature_matrix_v2.csv"
SEARCH_PATH = PROJECT / "results" / "search4.0_global_summary_metrics.csv"
HIST_PATH = PROJECT / "results" / "loss_histories_v2.json"
RESULT_DIR = PROJECT / "results"
LINE_COLORS = ["#507AAA", "#66C2A5", "#FC8D62", "#E78AC3"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PROJECT:", PROJECT, "| device:", device)


class CryptoVolatilityDataset(Dataset):
    def __init__(self, feature_matrix, target_array, lookback_window=7):
        self.features = feature_matrix
        self.targets = target_array
        self.window = lookback_window

    def __len__(self):
        return len(self.features) - self.window + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window]
        y = self.targets[idx + self.window - 1]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        scores = self.attention_weights(lstm_output)
        alpha = F.softmax(scores, dim=1)
        return torch.sum(alpha * lstm_output, dim=1), alpha.squeeze(-1)


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        d = hidden_size * 2
        self.attention = AttentionLayer(d)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc_layers = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.bilstm(x)
        c, w = self.attention(o)
        return self.fc_layers(self.dropout(c)), w


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    skip = {
        "timestamp",
        "target_sigma_t_plus_1",
        "finbert_sentiment_score",
        "llm_composite_score",
        "sentiment_class",
        "action_class",
        "action_score",
    }
    base = [c for c in df.columns if c not in skip]
    return {
        "G1": base,
        "G2": base + ["finbert_sentiment_score"],
        "G3": base + ["llm_composite_score"],
        "G3-M": base + ["sentiment_class", "action_class", "action_score"],
    }


def train_with_history(model, train_loader, val_loader, params, is_attention: bool):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6)
    best_val_loss, best_model_wts = float("inf"), None
    patience, epochs_no_improve, max_epochs = 20, 0, 300
    val_hist, train_hist = [], []

    for _epoch in range(max_epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)[0] if is_attention else model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_hist.append(float(np.mean(train_losses)))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                preds = model(x_v)[0] if is_attention else model(x_v)
                val_losses.append(criterion(preds, y_v).item())
        avg_val_loss = float(np.mean(val_losses))
        val_hist.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return {"val_loss": val_hist, "train_loss": train_hist, "best_val_loss": best_val_loss}


def run_all_trainings() -> dict:
    df = pd.read_csv(DATA_PATH)
    fg_map = feature_groups(df)
    search_df = pd.read_csv(SEARCH_PATH)
    train_end, val_end = int(len(df) * 0.7), int(len(df) * 0.8)
    histories = {}

    for _, row in search_df.iterrows():
        model_type = row["Model"]
        fg = row["Feature_Group"]
        key = f"{model_type}_{fg}"
        is_main = model_type == "Main"
        f_cols = fg_map[fg]
        params = {
            "hidden_size": int(row["Best_Hidden"]),
            "dropout": float(row["Best_Dropout"]),
            "batch_size": int(row["Best_BatchSize"]),
            "lr": float(row["Best_LR"]),
            "weight_decay": float(row["Best_WD"]),
        }
        train_ds = CryptoVolatilityDataset(
            df.iloc[:train_end][f_cols].values, df.iloc[:train_end]["target_sigma_t_plus_1"].values
        )
        val_ds = CryptoVolatilityDataset(
            df.iloc[train_end:val_end][f_cols].values,
            df.iloc[train_end:val_end]["target_sigma_t_plus_1"].values,
        )
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ModelCls = BiLSTMAttention if is_main else BaselineLSTM
        model = ModelCls(len(f_cols), params["hidden_size"], params["dropout"]).to(device)
        print(f"训练 {key}: H={params['hidden_size']} ...")
        hist = train_with_history(model, train_loader, val_loader, params, is_attention=is_main)
        histories[key] = {
            "model": model_type,
            "feature_group": fg,
            "params": params,
            "n_epochs": len(hist["val_loss"]),
            "best_val_loss": hist["best_val_loss"],
            "val_loss": hist["val_loss"],
            "train_loss": hist["train_loss"],
        }
        print(f"  => epochs={histories[key]['n_epochs']} best_val={hist['best_val_loss']:.6f}")

    payload = {
        "seed": SEED,
        "data": "data/04_final_feature_matrix_v2.csv",
        "hyperparams_source": "search4.0_global_summary_metrics.csv",
        "split": "70/10/20",
        "histories": histories,
    }
    HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HIST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("saved", HIST_PATH)
    return histories


def plot_stacked(model_type: str, outfile: str, histories: dict):
    plt.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "stix"
    sns.set_theme(
        style="whitegrid",
        font_scale=1.05,
        rc={"font.sans-serif": ["SimHei", "Microsoft YaHei", "SimSun"], "axes.unicode_minus": False},
    )
    labels = {"Baseline": "基线模型 (LSTM)", "Main": "BiLSTM-Attention"}
    prefix = labels[model_type]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    idx = 0
    for fgs_pair, ax in [(["G1", "G2"], ax1), (["G3", "G3-M"], ax2)]:
        for fg in fgs_pair:
            series = histories[f"{model_type}_{fg}"]["val_loss"]
            ax.plot(series, label=f"{prefix} × 特征组（{fg}）", color=LINE_COLORS[idx % 4], linewidth=1.8, alpha=0.95)
            idx += 1
        ax.set_ylabel("验证集损失 (MSE)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.35)
    ax2.set_xlabel("迭代轮次 (Epoch)")
    plt.tight_layout()
    out = RESULT_DIR / outfile
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    h = run_all_trainings()
    plot_stacked("Baseline", "4.1_baseline_loss_curves.png", h)
    plot_stacked("Main", "4.2_main_loss_curves.png", h)
