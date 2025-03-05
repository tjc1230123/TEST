<div align="center">

# 公众号的GitHub仓库

<p align="center">
  <img src="logo/公众号头像.jpg" alt="项目Logo" width="200"/>
</p>

分享有趣且先进的论文

</div>

# 计算机视觉论文分类整理

## 目录
- [🔨 1：多模态融合技术](#Category1)
  - [1.1：多模态图像融合](#Category1-1)
  - [1.2：XXX](#Category1-2)
  - [1.3：XXX](#Category1-3)
- [🎨 2：大模型](#Category2)
- [🚀 3：图像配准技术](#Category3)
- [🤖 4：XXX](#Category4)
  - [4.1：XXX](#Category4-1)
  - [4.2：XXX](#Category4-2)
- [📷 5：XXX](#Category5)
  - [5.1：XXX](#Category5-1)

<a name="Category1"></a>
## 🔨 1：多模态融合技术
通过整合来自不同传感器或数据源的多种模态信息（如图像、文本、语音等），实现更全面、准确的信息感知和分析的技术。

<a name="Category1-1"></a>
### 1.1：多模态图像融合
>整合来自不同成像模态（如光学、红外、雷达）的图像信息，提取互补特征，增强图像的细节和语义信息。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| CDDFuse | 基于相关性驱动的双分支特征分解的多模态图像融合 | CVPR | 2023 | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_Feature_Decomposition_for_Multi-Modality_Image_Fusion_CVPR_2023_paper.pdf) | [GitHub](https://github.com/Zhaozixiang1228/MMIF-CDDFuse) | 提出双分支Transformer-CNN框架，通过相关性驱动的分解损失函数提取和融合不同模态图像特征。 |
| SAFNet | 选择性对齐融合网络用于高效HDR成像 | ECCV | 2024 | [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03782.pdf) | [GitHub](https://github.com/ltkong218/SAFNet) | 提出新型选择性对齐融合网络，通过联合优化区域掩码和跨曝光运动，实现高效HDR图像融合。 |

#### 期刊论文

| 论文名称 | 中文论文名 | 期刊名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| MURF | 多模态图像配准与融合的相互强化框架 | IEEE TPAMI | 2023 | [Paper](https://ieeexplore.ieee.org/document/10145843) | [GitHub](https://github.com/hanna-xu/MURF) | 首创配准和融合任务结合的框架，通过相互强化机制提升配准精度和融合性能。 |

<a name="Category1-2"></a>
### 1.2：XXX
> XXX相关研究的描述。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 会议名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

#### 期刊论文

| 论文名称 | 中文论文名 | 期刊名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 期刊名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

<a name="Category1-3"></a>
### 1.3：XXX
> XXX相关研究的描述。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 会议名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

#### 期刊论文

| 论文名称 | 中文论文名 | 期刊名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 期刊名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

<a name="Category2"></a>
## 🎨 2：大模型
具有大规模参数和复杂计算结构的机器学习模型，通常由深度神经网络构建而成，拥有数十亿甚至数千亿个参数。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| DeepSeek-R1 | 通过强化学习激励大语言模型的推理能力 | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2501.12948) | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) | 通过大规模强化学习提升LLM推理能力，并通过知识蒸馏传递到小型模型中。 |

#### 期刊论文

| 论文名称 | 中文论文名 | 期刊名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 期刊名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

<a name="Category3"></a>
## 🚀 3：图像配准技术
图像配准技术是一种将不同时间、不同传感器或不同条件下获取的图像通过几何变换和空间对齐，使它们在空间位置和坐标系上一致的技术。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| LoFTR | 基于Transformer的无检测器局部特征匹配 | CVPR | 2021 | [Paper](https://arxiv.org/pdf/2104.00680) | [GitHub](https://github.com/zju3dv/LoFTR) | 提出无检测器匹配方法，通过Transformer直接建立密集的像素级匹配。 |
| Efficient LoFTR | 具有稀疏级速度的半密集局部特征匹配 | ICCV | 2023 | [Paper](https://zju3dv.github.io/efficientloftr/files/EfficientLoFTR.pdf) | [GitHub](https://github.com/zju3dv/efficientloftr) | 通过聚合注意力机制和两阶段相关性细化提升匹配效率。 |
| XoFTR | 跨模态特征匹配Transformer | CVPR | 2024 | [Paper](https://arxiv.org/pdf/2404.09692) | [GitHub](https://github.com/OnderT/XoFTR) | 解决可见光与热红外图像的跨模态匹配问题，提出两阶段训练方法。 |

<a name="Category4"></a>
## 🤖 4：XXX
XXX相关的研究和项目。

<a name="Category4-1"></a>
### 4.1：XXX
> XXX相关研究的描述。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 会议名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

#### 期刊论文

| 论文名称 | 中文论文名 | 期刊名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 期刊名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

<a name="Category4-2"></a>
### 4.2：XXX
> XXX相关研究的描述。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 会议名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

#### 期刊论文

| 论文名称 | 中文论文名 | 期刊名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 期刊名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

<a name="Category5"></a>
## 📷 5：XXX
XXX相关的研究和项目。

<a name="Category5-1"></a>
### 5.1：XXX
> XXX相关研究的描述。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 会议名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

#### 期刊论文

| 论文名称 | 中文论文名 | 期刊名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 期刊名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

<div align="center">

如果觉得项目还不错, 就点个 ⭐ Star 支持一下吧~

</div>
