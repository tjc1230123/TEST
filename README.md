<div align="center">

# 公众号的GitHub仓库

<p align="center">
  <img src="logo/logo.jpg" alt="项目Logo" width="200"/>
</p>
 

分享有趣且先进的论文

</div>

# 计算机视觉论文分类整理

## 目录
- [🔨 1：多模态融合技术](#Category1)
  - [1.1：多模态图像融合](#Category1-1)
  - [1.2：XXX](#Category1-2)
  - [1.3：XXX](#Category1-3)
- [🎨 2：XXX](#Category2)
  - [2.1：XXX](#Category2-1)
  - [2.2：XXX](#Category2-2)
- [🚀 3：XXX](#Category3)
  - [3.1：XXX](#Category3-1)
  - [3.2：XXX](#Category3-2)
- [🤖 4：XXX](#Category4)
  - [4.1：XXX](#Category4-1)
  - [4.2：XXX](#Category4-2)
- [📷 5：XXX](#Category5)
  - [5.1：XXX](#Category5-1)
  - [5.2：XXX](#Category5-2)

<a name="Category1"></a>
## 🔨 1：多模态融合技术
通过整合来自不同传感器或数据源的多种模态信息（如图像、文本、语音等），
实现更全面、准确的信息感知和分析的技术。

<a name="Category1-1"></a>
### 1.1：多模态图像融合
>整合来自不同成像模态（如光学、红外、雷达）的图像信息，提取互补特征，增强图像的细节和语义信息。

**[SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03782.pdf)**
- 描述：该项目实现了高效的多曝光高动态范围（HDR）成像功能，通过一种新型的Selective Alignment Fusion Network（SAFNet），在处理复杂运动和截断纹理的场景中表现出色。SAFNet通过联合优化有价值的区域掩码和选定区域的跨曝光运动，高效地融合高质量的HDR图像，同时显著提升了模型的运行效率和准确性。
- 特点：
  - 创新点1： 提出了一种区域选择与运动估计联合优化的方法。SAFNet通过金字塔特征提取后，联合优化有价值的区域掩码和选定区域的跨曝光运动，避免在不重要的区域进行复杂的运动估计，从而提高效率和准确性。
  - 创新点2：引入了一种轻量级细节增强模块，利用前阶段的光流、选择掩码和初始HDR预测信息，进一步增强高频细节，提升图像质量。
  - 创新点3：提出了一种窗口分割裁剪方法，在训练阶段结合大尺寸（512×512）和小尺寸（128×128）的图像块，同时优化长距离纹理聚合和短距离细节增强。
  - 创新点4：构建了一个更具挑战性的多曝光HDR数据集，包含更大的运动范围和更高的饱和区域比例，用于更公平地评估不同算法的性能差距。
  - 创新点5：在公共和新开发的数据集上验证了SAFNet的性能，结果表明，该方法不仅在定量和定性上超越了现有的SOTA方法，而且运行速度比基于Transformer的解决方案快一个数量级。
- 代码：[GitHub](https://github.com/ltkong218/SAFNet)

  
**[CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition
for Multi-Modality Image Fusion](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_Feature_Decomposition_for_Multi-Modality_Image_Fusion_CVPR_2023_paper.pdf)**
- 描述：通过一种新型的相关性驱动的双分支特征分解网络（CDDFuse），高效地提取和融合不同模态图像的全局和局部特征，显著提升了融合图像的质量和下游任务的性能。
- 特点：
  - 创新点1：提出了一种双分支Transformer-CNN框架，用于提取和融合全局（低频）和局部（高频）特征，更好地反映了不同模态图像的特有特征和共享特征。
  - 创新点2：引入了相关性驱动的分解损失函数，通过增强低频特征的相关性和减少高频特征的相关性，实现模态特有和模态共享特征的高效分解。
  - 创新点3：采用了轻量级Transformer（LT）块和可逆神经网络（INN）块，在保持计算效率的同时，提升了特征提取和融合的性能。
  - 创新点4：通过两阶段训练策略，有效地解决了多模态图像融合任务中缺乏真实数据的问题，提高了训练的鲁棒性和融合效果。
- 代码：[GitHub](https://github.com/Zhaozixiang1228/MMIF-CDDFuse)


**[MURF: Mutually Reinforcing Multi-Modal Image
Registration and Fusion](https://ieeexplore.ieee.org/document/10145843)**
- 描述：该项目实现了多模态图像的配准与融合功能，通过一种新型的相互强化框架（MURF），首次将图像配准和融合任务结合在一起，突破了传统方法中需要预对齐图像的限制，显著提升了配准精度和融合性能。
- 特点：
  - 创新点1：提出了一种相互强化框架（Mutually Reinforcing Framework），通过共享信息提取模块（SIEM）、多尺度粗配准模块（MCRM）和精细配准与融合模块（F2M），实现了从粗到细的配准流程，并利用融合图像的反馈进一步优化配准结果，同时配准的改进也反过来提升了融合效果。
  - 创新点2：在图像融合方面，不仅保留了原始图像的纹理细节，还通过纹理增强机制进一步提升了图像的视觉效果，使得融合后的图像不仅信息丰富，且纹理更加清晰，更适合后续的视觉任务。
  - 创新点3： 采用多尺度粗配准策略，通过逐步纠正全局刚性偏移，显著提高了配准的效率和精度，尤其是在处理具有显著模态差异的图像时表现出色。
  - 创新点4：引入了对比学习（Contrastive Learning）提取多模态图像的共享信息，有效消除了模态差异，使得配准和融合过程更加鲁棒。
  - 创新点5：设计了梯度通道注意力机制（Gradient Channel Attention Mechanism），用于自适应调整特征图的通道权重，进一步增强了融合图像的纹理细节和视觉效果。
- 代码：[GitHub](https://github.com/hanna-xu/MURF)

<a name="Category1-2"></a>
### 1.2：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<a name="Category1-3"></a>
### 1.3：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<a name="Category2"></a>
## 🎨 2：XXX
XXX相关的研究和项目。

<a name="Category2-1"></a>
### 2.1：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<a name="Category2-2"></a>
### 2.2：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<a name="Category3"></a>
## 🚀 3：XXX
XXX相关的研究和项目。

<a name="Category3-1"></a>
### 3.1：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<a name="Category3-2"></a>
### 3.2：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<a name="Category4"></a>
## 🤖 4：XXX
XXX相关的研究和项目。

<a name="Category4-1"></a>
### 4.1：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<a name="Category4-2"></a>
### 4.2：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<a name="Category5"></a>
## 📷 5：XXX
XXX相关的研究和项目。

<a name="Category5-1"></a>
### 5.1：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<a name="Category5-2"></a>
### 5.2：XXX
> XXX相关研究的描述。

**[项目名称1](项目链接)**
- 描述：这个项目实现了xxx功能
- 特点：
  - 创新点1
  - 创新点2
- 代码：[GitHub](github-link)

<div align="center">

如果觉得项目还不错, 就点个 ⭐ Star 支持一下吧~

</div>
