# 实验研究：区块链异常交易数据增强

本项目围绕以下研究主线开展实验：

> 区块链异常交易标注样本过少，难以支撑复杂模型训练，因此通过规则/统计方法生成候选异常交易图，并使用结构、时间和行为验证筛选高质量增强样本，检验其是否能够提升 GraphSAGE/GAT 的时间外推检测能力。

## 项目内容

- 规则/统计候选生成；
- 结构、时间、行为三类验证；
- GraphSAGE 与 GAT 检测器；
- 滚动时间窗口评估；
- 固定误报率阈值校准；
- 标签与时间分布审计；
- Logistic Regression / Random Forest 统计基线；
- 低数据量增强实验；
- 增强倍率与候选质量筛选实验。

## 目录

- `src/`：数据处理、候选增强、检测器实现；
- `configs/`：实验配置；
- `data/`：本地数据目录，不在仓库中提交原始数据；
- `results/`：实验结果文件；
- `实验架构设计.md`：实验总体架构；
- `实验首轮结果.md`：首轮实验报告；
- `滚动消融实验分析.md`：滚动窗口与增强消融分析；
- `时间分布与阈值校准分析.md`：时间分布和阈值校准分析；
- `标签时间审计与统计基线分析.md`：标签审计与统计基线分析；
- `低数据量增强实验分析.md`：低数据量增强分析。

## 数据准备

将本地的 `ptxphish_address_edges.csv` 放入：

```text
data/ptxphish_address_edges.csv
```

数据来源和授权情况请根据实际研究数据补充。原始交易数据、论文 PDF 和文献库不提交到本项目。

## 环境安装

建议使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果使用 NVIDIA GPU，请安装与驱动匹配的 CUDA 版本 PyTorch，并将配置中的 `device` 改为 `cuda`。

## 运行实验

```powershell
python run_experiment.py
python run_rolling_ablation.py
python run_calibration_experiment.py
python run_audit_experiment.py
python run_low_data_augmentation.py
python run_augmentation_ratio.py
```

运行前请确认 `configs/` 中的 `edge_file` 指向本地数据文件。完整实验报告会同时记录结果是否受到数据量、数据质量、时间漂移和硬件运行效率影响。

## 当前研究注意事项

当前结果表明，增强对 GAT 在部分数据规模下有效，但尚未形成稳定、普适的收益。后续重点是比较增强比例、候选质量和多随机种子结果，不能仅凭单次实验宣称增强方法有效。

## 当前研究状态与边界

当前增强对 GAT 只在部分数据规模和时间窗口下有效，尚未形成稳定、普适的收益。统计基线在部分后期窗口中明显强于当前 GNN，因此后续重点是比较增强比例、候选质量、多随机种子和时间外推表现。

当前 V1 已实现规则/统计候选生成、结构/时间/行为验证、GraphSAGE/GAT 基线和低数据量对照；尚未实现 VAE/扩散生成器、分布距离验证、主动学习、不确定性选择和边特征时序 GNN。详细边界见 `V1实现状态与研究边界.md`。

