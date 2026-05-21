#!/usr/bin/env python3
"""Build notebooks_figures/ from visualization notebooks (read-only inputs, PNG to results_figures/)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB7 = ROOT / "notebooks" / "7. visualizations_v2.ipynb"
DST_DIR = ROOT / "notebooks_figures"
DST_CH5 = DST_DIR / "ch5_figures.ipynb"

FIGURES_NB_NAMES = '("notebooks", "notebooks_figures")'

CH5_REPLACEMENTS: list[tuple[str, str]] = [
    (
        'PROJECT = ROOT.parent if ROOT.name == "notebooks" else ROOT\n'
        'R = PROJECT / "results"\n'
        'M = PROJECT / "models" / "final_models_v2"\n'
        'D = PROJECT / "data" / "04_final_feature_matrix_v2.csv"\n',
        'PROJECT = ROOT.parent if ROOT.name in ("notebooks", "notebooks_figures") else ROOT\n'
        'R_IN = PROJECT / "results"          # 只读：定型 CSV / JSON\n'
        'R_OUT = PROJECT / "results_figures"  # 只写：PNG，不覆盖 results/\n'
        'R_OUT.mkdir(parents=True, exist_ok=True)\n'
        'M = PROJECT / "models" / "final_models_v2"  # 只读：.pth 推理\n'
        'D = PROJECT / "data" / "04_final_feature_matrix_v2.csv"',
    ),
    ("在 `notebooks/` 下运行；图片保存至 `results/`。", "在 `notebooks_figures/` 下运行；**PNG 保存至 `results_figures/`**（不写入 `results/`）。"),
    ("`results/final_model_summary_metrics_v2.csv`", "`results/final_model_summary_metrics_v2.csv`（只读）"),
    ('pd.read_csv(R / "', 'pd.read_csv(R_IN / "'),
    ("HIST_JSON = R / ", "HIST_JSON = R_IN / "),
    ("fn = R / f\"tune", 'fn = R_IN / f"tune'),
    ("df_ab = pd.read_csv(R / ", "df_ab = pd.read_csv(R_IN / "),
    ("out = R / ", "out = R_OUT / "),
    ("out1 = R / ", "out1 = R_OUT / "),
    ("out2 = R / ", "out2 = R_OUT / "),
    ("→ `results/", "→ `results_figures/"),
    ("| `results/", "| `results_figures/"),
    ("保存至 `results/", "保存至 `results_figures/"),
    ("`notebooks/9.", "`notebooks/9."),
    ("写出中间结果：`results/loss_histories_v2.json`", "中间 JSON 仍在 `results/loss_histories_v2.json`（只读；需先由 nb9 生成）"),
    ("回写：`results/loss_histories_v2.json`", "可选 patch 脚本仍写 `results/`"),
]


def patch_ch5_cell(text: str) -> str:
    for old, new in CH5_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def build_ch5() -> None:
    nb = json.loads(SRC_NB7.read_text(encoding="utf-8"))
    nb["cells"][0]["source"] = [
        "# 第 5 章图表（只读复现版）\n",
        "\n",
        "- **只读**：`data/`、`models/final_models_v2/*.pth`、`results/*.csv`、`results/loss_histories_v2.json`\n",
        "- **只写 PNG**：`results_figures/`（**不覆盖** `results/` 与权重）\n",
        "- **不训练**：不运行 `4/6/9` 号 notebook；Loss 图需已有 `loss_histories_v2.json`\n",
        "\n",
        "论文 RMSE/MAE 口径仍来自 `results/final_model_summary_metrics_v2.csv`。\n",
    ]
    for cell in nb["cells"]:
        src = cell.get("source", [])
        if isinstance(src, list):
            cell["source"] = [patch_ch5_cell(line) for line in src]
        elif isinstance(src, str):
            cell["source"] = patch_ch5_cell(src)
    DST_CH5.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def write_ch3() -> None:
    ch3_path = DST_DIR / "ch3_figures.ipynb"
    ch3_path.write_text(_ch3_notebook_json(), encoding="utf-8")


def _ch3_notebook_json() -> str:
    """Return full ch3_figures.ipynb JSON."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [],
    }
    md = """# 第 3–4 章图表（只读复现版）

从已有 `data/*.csv` 读取，**PNG 写入 `results_figures/`**。
不写入 `data/`、`models/`、`results/`。

| 图文件 | 依赖数据 |
|:---|:---|
| `3_1`–`3_2`, `3_10` | `data/01_preprocessed_data.csv` |
| `3_3`–`3_5` | `data/02_finbert_features_added.csv` |
| `3_6`–`3_9` | `data/02_finbert_features_added.csv` |
"""
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [md]})

    setup = '''from pathlib import Path
import os

ROOT = Path.cwd()
PROJECT = ROOT.parent if ROOT.name in ("notebooks", "notebooks_figures") else ROOT
DATA = PROJECT / "data"
R_OUT = PROJECT / "results_figures"
R_OUT.mkdir(parents=True, exist_ok=True)

PATH_01 = DATA / "01_preprocessed_data.csv"
PATH_02 = DATA / "02_finbert_features_added.csv"
print("PROJECT:", PROJECT)
print("PNG ->", R_OUT)
'''
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": setup.splitlines(keepends=True), "outputs": [], "execution_count": None})

    # Read plot code from files we'll embed
    plot_a = open(ROOT / "scripts" / "_ch3_plots_section_a.py", encoding="utf-8").read() if (ROOT / "scripts" / "_ch3_plots_section_a.py").exists() else None
    if plot_a is None:
        plot_a = _CH3_SECTION_A
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["## 图 3.1 / 3.2 / 3.10（`01_preprocessed_data.csv`）\n"]})
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": plot_a.splitlines(keepends=True), "outputs": [], "execution_count": None})

    plot_b = _CH3_SECTION_B
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["## 图 3.3 / 3.4 / 3.5（`02_finbert_features_added.csv`）\n"]})
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": plot_b.splitlines(keepends=True), "outputs": [], "execution_count": None})

    plot_c = _CH3_SECTION_C
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": ["## 图 3.6–3.9（`02_finbert_features_added.csv`）\n"]})
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": plot_c.splitlines(keepends=True), "outputs": [], "execution_count": None})

    return json.dumps(nb, ensure_ascii=False, indent=1)


# --- ch3 plot sections (no CSV writes) ---

_CH3_SECTION_A = r'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="SimHei", rc={"axes.unicode_minus": False})

df = pd.read_csv(PATH_01)
df["timestamp"] = pd.to_datetime(df["timestamp"])
target_series = df["target_sigma_t_plus_1"].dropna()

# 图 3-1
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel("日期 (Date)", fontsize=12)
ax1.set_ylabel("比特币收盘价 (USD)", color="tab:blue", fontsize=12)
line1 = ax1.plot(df["timestamp"], df["close"], color="tab:blue", label="收盘价 (Close)", linewidth=1.5)
ax1.tick_params(axis="y", labelcolor="tab:blue")
plt.setp(ax1.get_xticklabels(), rotation=45)
ax2 = ax1.twinx()
ax2.set_ylabel(r"次日对数极值波动率 ($\sigma_{t+1}$)", color="tab:orange", fontsize=12)
line2 = ax2.plot(df["timestamp"], df["target_sigma_t_plus_1"], color="tab:orange", label="对数波动率", alpha=0.5, linewidth=1)
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax1.legend(line1 + line2, [l.get_label() for l in line1 + line2], loc="upper right")
ax1.set_title("")
fig1.tight_layout()
p1 = R_OUT / "3_1_price_volatility.png"
fig1.savefig(p1, dpi=300, format="png", bbox_inches="tight")
plt.close(fig1)
print("saved", p1)

# 图 3-2
fig2, ax = plt.subplots(figsize=(10, 6))
sns.histplot(target_series, bins=50, kde=True, color="#2c7bb6", ax=ax)
ax.set_xlabel(r"次日对数极值波动率 ($\sigma_{t+1}$)", fontsize=12)
ax.set_ylabel("频数 (Frequency)", fontsize=12)
fig2.tight_layout()
p2 = R_OUT / "3_2_volatility_distribution.png"
fig2.savefig(p2, dpi=300, format="png", bbox_inches="tight")
plt.close(fig2)
print("saved", p2)

# 图 3-10 ACF/PACF
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(target_series, lags=40, ax=axes[0])
axes[0].set_title("ACF")
plot_pacf(target_series, lags=40, ax=axes[1], method="ywm")
axes[1].set_title("PACF")
fig3.tight_layout()
p3 = R_OUT / "3_10_acf_pacf.png"
fig3.savefig(p3, dpi=300, format="png", bbox_inches="tight")
plt.close(fig3)
print("saved", p3)
'''

_CH3_SECTION_B = r'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(PATH_02)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

df["finbert_ma7"] = df["finbert_sentiment_score"].rolling(window=7, min_periods=1).mean()

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df.index, df["finbert_ma7"], color="#1f77b4", linewidth=1.5, label="FinBERT 情感得分 (7日均线)")
ax1.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax1.set_xlabel("日期 (Date)", fontsize=12)
ax1.set_ylabel("FinBERT 情感得分", fontsize=12)
ax1.legend(loc="upper right", frameon=True)
plt.tight_layout()
p = R_OUT / "3_3_finbert_ma7.png"
fig1.savefig(p, dpi=300, bbox_inches="tight")
plt.close()
print("saved", p)

fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
sns.histplot(df["finbert_sentiment_score"], bins=50, kde=True, color="#2ca02c", ax=ax_dist)
ax_dist.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
ax_dist.set_xlabel("FinBERT 情感得分 (Sentiment Score)", fontsize=12)
ax_dist.set_ylabel("频数 (Days)", fontsize=12)
plt.tight_layout()
p = R_OUT / "3_4_finbert_distribution.png"
fig_dist.savefig(p, dpi=300, bbox_inches="tight")
plt.close()
print("saved", p)

cross_tab = pd.crosstab(df["sentiment_class"], df["action_class"])
cross_tab.index = ["负面 (-1)", "中性 (0)", "正面 (1)"]
cross_tab.columns = ["卖出 (-1)", "持有 (0)", "买入 (1)"]
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues", ax=ax2, cbar_kws={"label": "频数 (天)"})
ax2.set_ylabel("情感类别 (sentiment_class)", fontsize=12)
ax2.set_xlabel("行动建议 (action_class)", fontsize=12)
plt.tight_layout()
p = R_OUT / "3_5_sentiment_action_heatmap.png"
fig2.savefig(p, dpi=300, bbox_inches="tight")
plt.close()
print("saved", p)
'''

_CH3_SECTION_C = r'''import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

df = pd.read_csv(PATH_02)
df["llm_composite_score"] = df["sentiment_class"] * (df["action_score"] / 10.0)
df["sigma_t"] = df["target_sigma_t_plus_1"].shift(1)
df = df.ffill().bfill()

plt.figure(figsize=(8, 6))
sns.histplot(df["llm_composite_score"], bins=30, kde=True, color="#2c7bb6", edgecolor="black", alpha=0.7)
plt.xlabel(r"LLM复合情感因子 ($S_{LLM}$)", fontsize=14)
plt.ylabel("频数 (Frequency)", fontsize=14)
plt.tight_layout()
p = R_OUT / "3_6_llm_composite_hist.png"
plt.savefig(p, dpi=300, format="png")
plt.close()
print("saved", p)

plt.figure(figsize=(8, 6))
sns.kdeplot(df["finbert_sentiment_score"], label=r"FinBERT ($S_{FB}$)", fill=True, color="#d7191c", alpha=0.4, linewidth=2)
sns.kdeplot(df["llm_composite_score"], label=r"LLM复合因子 ($S_{LLM}$)", fill=True, color="#2c7bb6", alpha=0.4, linewidth=2)
plt.xlabel("情感得分 (Sentiment Score)", fontsize=14)
plt.ylabel("概率密度 (Density)", fontsize=14)
plt.legend(loc="upper right", fontsize=12)
plt.tight_layout()
p = R_OUT / "3_7_finbert_vs_llm_kde.png"
plt.savefig(p, dpi=300, format="png")
plt.close()
print("saved", p)

core_cols = [
    "target_sigma_t_plus_1", "sigma_t", "finbert_sentiment_score", "llm_composite_score",
    "close", "volume", "transaction-fees-usd", "hash-rate", "fng_value",
]
labels = [r"$\sigma_{t+1}$", r"$\sigma_{t}$", r"$S_{FB}$", r"$S_{LLM}$",
          "Close Price", "Volume", "Tx Fees", "Hash Rate", "FNG Index"]
corr_matrix = df[core_cols].corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlBu_r",
            xticklabels=labels, yticklabels=labels, vmin=-1, vmax=1, center=0, square=True, linewidths=0.5)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
p = R_OUT / "3_8_feature_correlation_heatmap.png"
plt.savefig(p, dpi=300, format="png")
plt.close()
print("saved", p)

max_lags = 7
lags = np.arange(1, max_lags + 1)
corr_llm = [df["llm_composite_score"].corr(df["target_sigma_t_plus_1"].shift(-(k - 1))) for k in lags]
corr_fb = [df["finbert_sentiment_score"].corr(df["target_sigma_t_plus_1"].shift(-(k - 1))) for k in lags]
plt.figure(figsize=(10, 6))
bw = 0.35
xi = np.arange(len(lags))
plt.bar(xi - bw / 2, corr_fb, width=bw, label=r"FinBERT ($S_{FB}$)", color="#d7191c", alpha=0.8)
plt.bar(xi + bw / 2, corr_llm, width=bw, label=r"LLM复合因子 ($S_{LLM}$)", color="#2c7bb6", alpha=0.8)
plt.xlabel(r"滞后天数 ($k$ days)", fontsize=14)
plt.ylabel(r"与对数极值波动率 ($\sigma_{t+k}$) 的相关系数", fontsize=14)
plt.xticks(xi, [f"Lag {k}" for k in lags])
plt.axhline(0, color="black", linewidth=1, linestyle="--")
plt.legend(loc="upper right", fontsize=12)
plt.tight_layout()
p = R_OUT / "3_9_lagged_correlation.png"
plt.savefig(p, dpi=300, format="png")
plt.close()
print("saved", p)
'''


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "results_figures").mkdir(parents=True, exist_ok=True)
    build_ch5()
    write_ch3()
    print("Wrote", DST_CH5.relative_to(ROOT))
    print("Wrote", (DST_DIR / "ch3_figures.ipynb").relative_to(ROOT))


if __name__ == "__main__":
    main()
