# 数据集与划分

本仓库只提供算法和聚合实验说明，不重新分发晶圆数据、渲染图片或逐图结果。下载后将数据保留在本地 `data_sources/`，生成文件保留在 `data/`、`runs/`。

## 数据编码与标签

输入是电测 wafer map，不是显微图像。数组中 `0` 表示没有 die／晶圆外，`1` 表示通过电测，`2` 表示失效 die。正常类别表示没有被标注的系统性缺陷模式，并不要求失效点数为零。

内部八类顺序固定为：

```text
center, donut, edge_loc, edge_ring, loc, near_full, scratch, random
```

多标签任务用八维 multi-hot；全零目标表示正常。单标签任务额外使用 `normal` 类。

## MixedWM38

[原作者仓库](https://github.com/Junliangwangdhu/WaferMap)对应论文 [Deformable Convolutional Networks for Efficient Mixed-type Wafer Defect Pattern Recognition](https://doi.org/10.1109/TSM.2020.3020985)。作者说明数据来自晶圆制造过程，并用 GAN 生成部分样本平衡类别。发行文件没有逐图真实/合成标志、lot ID 或 GAN 父样本 ID。

核对的 Kaggle 第 4 版下载文件为 `Wafer_Map_Datasets.npz`：

| 项目 | 实际内容 |
|---|---|
| `arr_0` | `int32`，`(38015, 52, 52)` |
| `arr_1` | `int32`，`(38015, 8)` |
| 标签组合 | 1 种正常、8 种单模式、29 种混合模式，共 38 种 |
| 按模式数分布 | 正常 1,000；单模式 7,015；双模式 13,000；三模式 13,000；四模式 4,000 |
| 两个较小的单模式类 | Near-full 149；Random 866 |

八列顺序依据[作者标签图表](https://raw.githubusercontent.com/Junliangwangdhu/WaferMap/master/Dataset%20Figure/Single-Type.png)。作者 README 曾把 multi-hot 称作 one-hot，并明确注明 C7/C9 标签修正；旧版或二手描述可能混淆 Near-full 与 Random，不能依图片位置猜标签列。

原始档有 214 个未文档化的值 `3`，涉及 105 张图。本次版本中为 8 张 Near-full、97 张 Random。[作者答复](https://github.com/Junliangwangdhu/WaferMap/issues/4#issuecomment-2438186171)称 `3` 为噪声。本项目隔离这些整图，保留原文件，不自动将 `3` 改成失效或正常。

默认准备流程先过滤 die 编码，再将原图及八种旋转/镜像变换归入同一 D4 组；每组保留一个代表，标签冲突组隔离。在本次数据上，过滤后 37,910 张，得到 36,890 个 D4 组，标签冲突组为 0。按组合分层、组级 70/15/15 划分后再施加数量上限，得到：

| 分区 | 数量 |
|---|---:|
| train | 7,494 |
| val | 760 |
| 旧 test | 760 |

D4 去重不能消除未知合成父样本带来的相关性。均衡取样也不代表产线自然分布。Random 和 Near-full 不出现在发行版的混合组合中，因此这些类别的新组合泛化没有得到验证。

### 下载和校验

[作者指定 Kaggle 数据页](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets)显示许可为 **Data files © Original Authors**。作者邀请研究使用，但不能据此自行标为 CC0 或 CC-BY；使用范围应遵循原作者和下载源条款。

```powershell
New-Item -ItemType Directory -Force data_sources
Invoke-WebRequest -Uri 'https://www.kaggle.com/api/v1/datasets/download/co1d7era/mixedtype-wafer-defect-datasets?datasetVersionNumber=4' -OutFile data_sources/mixedwm38_kaggle.zip
Expand-Archive -LiteralPath data_sources/mixedwm38_kaggle.zip -DestinationPath data_sources
Get-FileHash -Algorithm SHA256 -LiteralPath data_sources/Wafer_Map_Datasets.npz
```

NPZ 的 SHA256 应为：

```text
a19e791c85ccd5051c080169a3d2b17902a42545cf749e9869f6c96312bcdc69
```

## WM-811K

来源论文为 [Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets](https://doi.org/10.1109/TSM.2014.2364237)，原始分发地址为 [MIR Lab](http://mirlab.org/dataSet/public/MIR-WM811K.zip)。本次使用 [Kaggle 镜像](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)的 `LSWMD.pkl`，该下载页标注 **CC0: Public Domain**。

文件包含 811,457 条真实 wafer map 记录；172,950 条有模式标签，638,507 条无标签。原生有标签分区为 Training 54,355、Test 118,595。矩形和不同尺寸的 map 均保留原始网格；图级特征会按有效 die 归一化。

`failureType` 和原生分区字段可能是嵌套数组，导入器展开后区分空标签与明确的 `none`。保留 `lotName`、`waferIndex`、`trianTestLabel` 的含义；使用局部兼容 Unpickler 读取旧版 pandas 档案，不修改进程全局模块别名。

每个包含原生 Test 记录的 lot 整体归 test，其余 lot 才分 train/val。跨分区或标签冲突的 D4 组整组隔离；矩形图的旋转按各变换后的尺寸一起计算等价组。本次原生有标签 Training/Test lot 没有交叉。施加每类 `200/20/20` 上限后为 train 1,637、val 169、test 180，已选分区之间 lot 和 D4 交叉均为 0。

Near-full 原生 Training 仅 54 条；本协议最终 train 37、val 9、test 20。记录真实短缺，不复制少数类填满数量上限。该 180 张平衡测试子集并非完整原生 test，也不是 MixedWM38 训练模型直接跨域迁移的结果：WM-811K 基线在自己的 train 上训练。

```powershell
Invoke-WebRequest -Uri 'https://www.kaggle.com/api/v1/datasets/download/qingyi/wm811k-wafer-map?datasetVersionNumber=1' -OutFile data_sources/wm811k_kaggle.zip
Expand-Archive -LiteralPath data_sources/wm811k_kaggle.zip -DestinationPath data_sources
Get-FileHash -Algorithm SHA256 -LiteralPath data_sources/LSWMD.pkl
```

下载压缩包约 157 MB，解压文件约 2.10 GB。PKL 的 SHA256 应为：

```text
1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65
```

## 新的 mixed-only 留出集

`fresh-holdout` 重放默认 MixedWM38 分区并核对参考 manifest，从未使用的原始 test 池中抽取 29 种混合组合各 20 个 D4 组，共 580 张；其中双模式 260、三模式 240、四模式 80。它不改变原训练/验证分区，排除已准备或已评估记录及其 D4 等价组，并拒绝不足时用重复样本补齐。

先在原验证集冻结特征/增强方案和训练种子，再评估此留出集。评估过一次后，它不再是新的未观察测试集，后续调参应另留数据。它仍来自同一个公开档案，不是独立产线数据。
