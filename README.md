# btc-volatility-thesis

基于 BiLSTM-Attention 的比特币次日波动率预测（学位论文配套代码）。

---

## 先读什么

| 文档 | 用途 |
|:---|:---|
| **[`THESIS_PROJECT_GUIDE.md`](THESIS_PROJECT_GUIDE.md)** | **工程主文档**：目录与仓库映射、定型数字口径、图表文件名、审查清单 |
| [`background document.md`](background%20document.md) | 写作术语与叙述口径（润色用，不映射代码路径） |


### 数据口径（硬性）

| 用途 | 权威文件 |
|:---|:---|
| 正文测试集 **RMSE / MAE / R²** | `results/final_model_summary_metrics_v2.csv` |
| 最优 **超参数**（Hidden、Dropout、LR 等） | `results/search4.0_global_summary_metrics.csv` |

勿用 `search4.0` 表中的 `Test_RMSE` 作正文主结论；勿用重新训练得到的新指标覆盖定型 CSV。

---

## 仓库结构

```
btc-volatility-thesis/
├── notebooks/              # 主实验流水线（清洗 → 训练 → 出图）
├── notebooks_figures/      # 只读重绘插图（不写回 results/）
├── scripts/                # 附录生成、loss 批跑、出图 notebook 构建
├── data/                   # 特征矩阵与原始合并表（本地，通常不入库）
├── models/                 # scaler、final_models_v2/*.pth、tune 搜索权重（本地）
├── results/                # 定型 CSV 与论文插图 PNG（本地）
├── results_figures/        # notebooks_figures 的 PNG 输出（本地，可删后重生成）
├── THESIS_PROJECT_GUIDE.md
└── README.md
```

`data/`、`models/`、`results/`、`results_figures/` 已在 `.gitignore` 中忽略。

---

## 环境

- Python 3.10+，PyTorch（GPU 可选）
- 常用：`pandas`、`numpy`、`matplotlib`、`seaborn`、`scikit-learn`、`statsmodels`、`torch`
- FinBERT 推理需 `transformers`（见 `notebooks/2`）

在仓库根目录或 `notebooks/` 下打开 Jupyter / VS Code 运行 notebook。

---

## 主实验流水线（`notebooks/`）

按编号顺序执行；**4、5、6 耗时长且会写盘**，定型后慎重重跑。

| 顺序 | Notebook | 主要产出 |
|:---|:---|:---|
| 1 | `1. data_cleaning.ipynb` | `data/01_preprocessed_data.csv`；图 `3_1`、`3_2`、`3_10` |
| 2 | `2. finbert_features.ipynb` | `data/02_finbert_features_added.csv`；图 `3_3`–`3_5` |
| 3 | `3. feature_matrix_and_ch3_outputs_v2.ipynb` | **`data/04_final_feature_matrix_v2.csv`**；图 `3_6`–`3_9`；表 `3_8`–`3_10` CSV |
| 4 | `4. search_train_v2.ipynb` | `search4.0_global_summary_metrics.csv`、`tune_v2_*.csv`、`models/tune_v2_*` |
| 5 | `5. rerun_baseline_g3.ipynb` | **可选**：仅当需修正 Baseline×G3 搜索异常时 |
| 6 | `6. ablation_v2.ipynb` | `ablation_v2_results.csv`；消融相关 PNG |
| 7 | `7. visualizations_v2.ipynb` | **第 5 章主出图**（写入 `results/`） |
| 9 | `9. loss_curves_v2.ipynb` | `loss_histories_v2.json`；供图 4.1 / 4.2 |

**一键重绘第 5 章图（覆盖 `results/` 内 PNG）**：在 `notebooks/` 下对 **`7. visualizations_v2.ipynb`** 执行 **Run All**。

依赖：`data/04_final_feature_matrix_v2.csv`、`results/*.csv`、`models/final_models_v2/*.pth`。  
图 4.1 / 4.2 需先有 `results/loss_histories_v2.json`（由 notebook 9 生成）；定型 `.pth` 一般不含 `history`，不能单靠权重重绘 loss。

---

## 只重绘插图（不训练、不覆盖 `results/`）

评阅、改图或备份插图时使用，与主流水线隔离。

| 目录 | 说明 |
|:---|:---|
| [`notebooks_figures/`](notebooks_figures/README.md) | `ch3_figures.ipynb`、`ch5_figures.ipynb` |
| `results_figures/` | 仅接收上述 notebook 输出的 **PNG** |

- **只读**：`data/`、`models/final_models_v2/`、`results/*.csv`（及 `loss_histories_v2.json`）
- **不写**：`results/` 定型 CSV、权重、特征矩阵

从 `notebooks_figures/` 依次 Run All 即可。若 Word 仍链到 `results/*.png`，确认无误后可手动复制过去。

刷新 `ch5_figures.ipynb`（自 `notebooks/7` 改编）：

```bash
python scripts/build_figures_notebooks.py
```

---

## `scripts/`

| 脚本 / Notebook | 说明 |
|:---|:---|
| `build_figures_notebooks.py` | 生成/更新 `notebooks_figures/` |
| `_run_nb9_loss.py` | 无头执行 notebook 9，写出 `loss_histories_v2.json` |
| `patch_baseline_g3_loss_display.py` | 可选：修正 loss JSON 中 Baseline×G3 曲线展示 |
| `generate_appendix_unified_notebook.py` | 生成附录合并版 notebook |
| `generate_appendix_word_minimal.py` | 生成 Word 附录精简版 notebook |
| `99_appendix_consolidated_code.ipynb` | 附录代码（合并流水线，由上一脚本生成） |
| `99_appendix_word_minimal.ipynb` | 附录代码（精简版） |

---

## 关键路径速查

| 路径 | 说明 |
|:---|:---|
| `data/04_final_feature_matrix_v2.csv` | 建模用特征矩阵（含 `sigma_t`） |
| `models/feature_scaler_v2.pkl` | 标准化器（notebook 3 拟合） |
| `models/final_models_v2/*.pth` | 定型推理权重 |
| `results/final_model_summary_metrics_v2.csv` | **正文性能唯一来源** |
| `results/search4.0_global_summary_metrics.csv` | 超参数；勿用其 Test_RMSE 写正文 |
| `results/ablation_v2_results.csv` | Lookback 消融 |
| `results/loss_histories_v2.json` | Loss 曲线数据 |

更完整的图号 ↔ 文件名对照见 **`THESIS_PROJECT_GUIDE.md` 第三节、第七节**。

---

## Git 说明

当前索引仍跟踪旧版 `notebooks/1. data cleaning.ipynb` 等文件名；工作区已换为 `1. data_cleaning.ipynb` 等 v2 流水线。提交前建议：

```bash
git add -A notebooks/
git add THESIS_PROJECT_GUIDE.md README.md .gitignore scripts/
```

并确认删除记录包含旧 notebook 与 `test1.ipynb`。

---

## 可选本地清理

- 空目录 `src/` 可手动删除
- `scripts/__pycache__/` 可删（会再生成）
- `results_figures/*.png` 可由 `notebooks_figures` 重生成
