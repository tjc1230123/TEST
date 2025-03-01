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
- [🎨 2：大模型](#Category2)
- [🚀 3：图像配准技术](#Category3)
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
  - 创新点3：采用多尺度粗配准策略，通过逐步纠正全局刚性偏移，显著提高了配准的效率和精度，尤其是在处理具有显著模态差异的图像时表现出色。
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
## 🎨 2：大模型
具有大规模参数和复杂计算结构的机器学习模型，通常由深度神经网络构建而成，拥有数十亿甚至数千亿个参数。这些模型通过训练海量数据来学习复杂的模式和特征，具有强大的泛化能力和涌现能力。

<a name="Category2-1"></a>

**[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)**
- 描述：这个项目实现了通过大规模强化学习（Reinforcement Learning, RL）提升大型语言模型（LLMs）推理能力的功能，并通过知识蒸馏（Distillation）将推理能力传递到小型模型中，使其在多种推理任务中达到与行业领先模型相当的性能。
- 特点：
  - 创新点1：DeepSeek-R1-Zero通过大规模强化学习，无需监督微调（SFT）作为预处理步骤，成功激励了模型的推理能力，证明了模型可以通过自我演化发展出强大的推理行为。
  - 创新点2：DeepSeek-R1引入了冷启动数据和多阶段训练流程，结合推理导向的强化学习和拒绝采样，显著提升了模型的推理性能，并解决了早期模型在可读性和语言混合方面的问题。
  - 创新点3：项目开源了DeepSeek-R1-Zero、DeepSeek-R1以及多个基于Qwen和Llama系列的小型推理模型（1.5B、7B、8B、14B、32B、70B），为研究社区提供了丰富的资源，推动了推理能力研究的发展。
- [GitHub](https://github.com/deepseek-ai/DeepSeek-R1)

<a name="Category2-2"></a>
<a name="Category3"></a>
## 🚀 3：图像配准技术
图像配准技术是一种将不同时间、不同传感器或不同条件下获取的图像通过几何变换和空间对齐，使它们在空间位置和坐标系上一致，从而便于后续分析和处理的技术。

<a name="Category3-1"></a>

**[LoFTR: Detector-Free Local Feature Matching with Transformers](https://arxiv.org/pdf/2104.00680)**
- 描述：实现了基于Transformer的无检测器局部特征匹配功能。
- 特点：
  - 创新点1：提出了一种无检测器的匹配方法，通过Transformer的自注意力和交叉注意力层，直接在图像间建立密集的像素级匹配。
  - 创新点2：采用粗到细的匹配策略，先在低分辨率特征图上提取密集匹配，再通过相关性方法将匹配细化到亚像素级别。
  - 创新点3：利用Transformer的全局感受野和位置编码，使特征匹配能够依赖全局上下文信息，从而在不依赖手工设计的特征检测器的情况下，实现对复杂场景的高效匹配。
- 代码：[GitHub](https://github.com/zju3dv/LoFTR)


**[Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed](https://zju3dv.github.io/efficientloftr/files/EfficientLoFTR.pdf)**
- 描述：这个项目实现了高效且半密集的图像局部特征匹配功能，旨在解决传统无检测器匹配方法（如LoFTR）在处理大规模视点变化和低纹理场景时效率低下的问题。项目通过重新审视LoFTR的设计选择，并引入多项改进，实现了在保持高精度的同时显著提升匹配效率的目标。
- 特点：
  - 创新点1：提出了一种聚合注意力机制，通过自适应选择关键特征（token）进行高效的特征转换，显著减少了计算量，同时保持了与全特征图变换相当的匹配精度。
  - 创新点2：设计了两阶段相关性细化层，通过先进行像素级匹配，再进行亚像素级细化，有效解决了传统方法中因噪声导致的空间偏差问题，进一步提高了匹配精度。
- 代码：[GitHub](https://github.com/zju3dv/efficientloftr)


**[XoFTR: Cross-modal Feature Matching Transformer](https://arxiv.org/pdf/2404.09692)**
- 描述：这个项目实现了跨模态（可见光与热红外图像）局部特征匹配功能，旨在解决可见光与热红外图像之间因纹理、强度差异以及不同成像机制带来的匹配难题。
- 特点：
  - 创新点1：提出了一种两阶段训练方法，通过掩码图像建模（MIM）预训练和伪热红外图像增强的微调策略，解决了可见光与热红外图像之间因模态差异导致的匹配难题，显著提高了模型对不同光照和纹理条件的适应性。
  - 创新点2：引入了一种新的可见光-热红外图像数据集（METU-VisTIR），涵盖了多种视角差异和天气条件（晴天和阴天），为跨模态匹配算法的评估提供了更具挑战性的测试环境。
  - 创新点3：在粗匹配阶段引入了一对多和多对一的匹配策略，解决了因视角和尺度变化导致的特征不一致性问题，增强了模型在复杂场景下的匹配能力。
- 代码：[GitHub](https://github.com/OnderT/XoFTR?tab=readme-ov-file)

<a name="Category3-2"></a>

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
