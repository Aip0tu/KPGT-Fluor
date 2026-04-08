# KPGT-Fluor

荧光染料因其独特的光学性质，在生物、化学和材料科学等多个领域中都具有重要作用。为了实现高性能染料的理性设计，准确预测这些性质至关重要。随着深度学习（DL）的引入，人工智能驱动的药物设计（AIDD）发生了显著变革，也催生了强大的分子性质预测工具。在分子性质预测基础模型的基础上，我们提出了 KPGT-Fluor，这是对 Knowledge-guided Pre-training of Graph Transformer（KPGT）框架的一种新适配，专门面向荧光染料性质预测任务。原始 KPGT 框架主要用于通用分子性质估计，而 KPGT-Fluor 进一步引入了溶剂表示，使模型能够更好地捕捉影响染料行为的环境效应。KPGT-Fluor 取得了较强的预测性能，在吸收波长和发射波长预测上，均方根误差（RMSE）分别达到 18.91 nm 和 18.56 nm。对于消光系数对数和量子产率，RMSE 分别为 0.159 和 0.126，表现出较高精度。此外，该模型在多个下游染料数据集上也展现出稳健的泛化能力。这些能力使 KPGT-Fluor 成为面向下一代荧光染料数据驱动设计的强大而通用的工具。

![KPGT-Fluor](./framework.png)

## 环境配置

**1. 安装 KPGT 框架**

```bash
git clone https://github.com/MolAstra/KPGT-Fluor.git
cd KPGT
mamba env create  # CUDA11.3, torch1.10
mamba activate KPGT

pip install transformers
```

**2. 将数据集和预训练模型下载到对应目录**

更多细节请参考 [KPGT](https://github.com/lihan97/KPGT) 仓库。本文复用了 KPGT 仓库中的预训练模型和数据集。

## 数据集

用于荧光分子预测的完整数据集大小约为 6 GB。你可以通过邮箱 `molastra@hotmail.com` 联系我们获取数据集副本。

目前数据集也已同步到 Zenodo，欢迎直接下载这个 [压缩包](https://zenodo.org/records/17718274)。

## 实验

```bash
bash train.sh  # 在 consolidation 数据集上训练 KPGT-Fluor
bash predict.sh  # 在 consolidation 数据集上用 KPGT-Fluor 进行预测

bash train_external.sh  # 在外部数据集上微调已在 FluorDB 上训练好的 KPGT-Fluor
bash predict_external.sh  # 在外部数据集上预测已在 FluorDB 上训练好的 KPGT-Fluor
bash predict_direct.sh  # 在外部数据集上进行 zero-shot KPGT-Fluor 预测

python train_ml.py  # 在 consolidation、cyanine、xanthene 等数据集上训练传统机器学习模型
python predict_ml.py  # 在 consolidation、cyanine、xanthene 等数据集上进行传统机器学习预测

python predict_ml_direct.py  # 在外部数据集上进行 zero-shot 传统机器学习预测
```

## 可视化相关的 Jupyter Notebook

更多内容请参考 `notebooks` 和 `plots` 目录，论文中的一些 `case_study` 也可以在这里找到。

```bash
bash case_study.sh
```

## 使用 LightGBM 的 Shapley 值进行模型解释

- `shap`
- `shap.csv`
- `train_ml_absorption.ipynb`
- `train_ml_emission.ipynb`
- `train_ml_log_molar_absorptivity.ipynb`
- `train_ml_quantum_yield.ipynb`
