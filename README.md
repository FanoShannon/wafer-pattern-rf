# Wafer Pattern RF

面向晶圆电测分布图（wafer map）的随机森林分类实验：从 `0=晶圆外、1=通过、2=失效` 的二维数组提取空间特征，识别单模式与混合模式。

包含 168 维基础特征、502 维扩展特征、单标签/多标签分类、训练集组合增强，以及先验证选择、再打开新留出集的评估流程。核心计算可在 CPU 上运行。

**这是图像级分类项目，不输出逐 pattern 分割 mask，也没有验证图案分离、像素定位或工厂部署效果。** 仓库不附原始数据、样本图或逐图实验记录。

## 安装与流程检查

建议使用 Python 3.11 或更新版本，在独立虚拟环境中安装依赖：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m waferbench doctor
python examples/smoke_demo.py --out runs/smoke
```

如需安装 `wafer-rf` 命令，可在仓库中执行 `python -m pip install -e .`。无需激活环境时，Windows也可直接使用 `.venv\Scripts\python.exe` 运行上述命令。

`smoke_demo` 使用小型人工构造数据检查流程，不是实际数据集的性能评估。以下命令均在仓库根目录执行。

## 准备数据

自行从 [MixedWM38 作者仓库](https://github.com/Junliangwangdhu/WaferMap)及其指定下载源取得 `Wafer_Map_Datasets.npz`，放入 `data_sources/`。来源、固定版本、SHA256 和许可见 [DATASETS](docs/DATASETS.md)及[机器可读来源记录](docs/data_sources.json)。

```powershell
python -m waferbench prepare --source data_sources/Wafer_Map_Datasets.npz --out data/mixedwm38 --seed 42 --train-per-combo 200 --val-per-combo 20 --test-per-combo 20
```

准备阶段隔离异常 die 编码，按原图、旋转与镜像等价组去重，再按组合划分训练/验证/测试。数量上限不代表每组一定有足够样本。已有输出目录会受到保护，重新划分应使用新目录。

## 训练与预测

单模式任务只保留正常和单缺陷图；混合任务对八个基础模式分别输出是否存在。

```powershell
python -m waferbench baseline --task single --features base --manifest data/mixedwm38/manifest.jsonl --out runs/mixedwm38/rf_single
python -m waferbench baseline --task multi --features base --manifest data/mixedwm38/manifest.jsonl --out runs/mixedwm38/rf_multi
python -m waferbench baseline --task multi --features advanced --manifest data/mixedwm38/manifest.jsonl --out runs/mixedwm38/rf_advanced
```

训练输出包含 `model.joblib`、`model_metadata.json` 和评估文件。预测新样本时，将 `your_wafer_map.npy` 替换为实际的二维 `0/1/2` 数组文件：

```powershell
python -m waferbench predict --model runs/mixedwm38/rf_advanced/model.joblib --map your_wafer_map.npy --out runs/prediction.json
```

## 组合增强与特征消融

准备默认 MixedWM38 分区后，可直接运行完整研究，不要求预先存在历史训练结果：

```powershell
python -m waferbench.augmentation_study --out runs/augmentation_study --n-per-combo 100
```

研究匹配三种增强臂的增加数量与标签组合分布：训练混合图的 D4 重采样、单模式失效点直接并集、单模式软结构组合后补一次共享背景。参数及方案只按验证集选择；新留出集仅评估基础方案和验证集选定方案。

也可单独准备尚未评分的混合留出集：

```powershell
python -m waferbench fresh-holdout --source data_sources/Wafer_Map_Datasets.npz --reference-manifest data/mixedwm38/manifest.jsonl --out data/mixedwm38_fresh_mixed
```

该留出流程严格重放默认 `seed=42`、每组合 `train/val/test=200/20/20` 的参考分区。它从未使用的原测试池抽取 29 种混合组合各 20 张，并检查 ID 和 D4 组交叉；更改原分区后不能直接沿用本留出协议。

## WM-811K 单模式验证

取得并核对 `LSWMD.pkl` 后运行：

```powershell
python -m waferbench prepare-wm811k --source data_sources/LSWMD.pkl --out data/wm811k
python -m waferbench baseline --task single --features base --manifest data/wm811k/manifest.jsonl --out runs/wm811k/rf_single
```

此任务保留原始 lot 信息并按 lot 隔离训练/验证/测试，独立处理 D4 重复，不把无标签记录当作正常样本。

## 已记录结果与适用边界

历史 580 张新混合留出集上，验证集选出的 502 维+D4 方案，三个训练种子的整组正确率均值为 **80.5%**，168 维基础方案为 **74.3%**。它们是同一公开数据源上的对照结果；MixedWM38 包含合成样本，无法据此保证真实产线效果。

完整聚合表、阈值区别和探索性测试说明见 [RESULTS](docs/RESULTS.md)；特征、增强及评估定义见 [METHODS](docs/METHODS.md)。
