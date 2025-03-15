<div align="center">

# 公众号的GitHub仓库

<p align="center">
  <img src="logo/公众号头像.jpg" alt="项目Logo" width="200"/>
  <img src="logo/二维码公众号.png" alt="公众号二维码" width="200"/>
</p>

分享有趣且先进的论文 

欢迎关注公众号获取更多资讯！

</div>

# 计算机视觉论文分类整理

## 目录
- [🔨 1：多模态融合技术](#Category1)
  - [1.1：多模态图像融合](#Category1-1)
  - [1.2：XXX](#Category1-2)
  - [1.3：XXX](#Category1-3)
- [🎨 2：大模型](#Category2)
- [🚀 3：图像配准技术](#Category3)
- [🤖 4：三维重建技术](#Category4)
- [📷 5：多曝光图像融合](#Category5)
  - [5.1：基于多帧序列对齐的多曝光图像融合](#Category5-1)

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
## 🤖 4：三维重建技术
三维重建技术是利用多视角图像、深度数据或多模态感知信息，通过计算方法自动生成真实世界场景或物体的高精度三维几何结构、表面属性及语义表达的技术。

#### 会议论文

| 论文名称                                                                      | 中文论文名             | 会议名称 | 时间   | Paper | Code | 简述                                                                                                                                      |
|---------------------------------------------------------------------------|-------------------|------|------|-------|------|-----------------------------------------------------------------------------------------------------------------------------------------|
| NM-Net                                                                    | 基于可靠邻域挖掘的鲁棒特征对应方法 | CVPR | 2019 | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_NM-Net_Mining_Reliable_Neighbors_for_Robust_Feature_Correspondences_CVPR_2019_paper.html) | [Github](https://github.com/sailor-z/NM-Net) | 通过提出兼容性特定挖掘方法与分层图卷积网络，解决了传统k近邻策略因无法保证错误对应的空间一致性而导致的局部信息可靠性问题，实现了对无序特征对应关系中可靠局部特征的精准提取和鲁棒聚合。                                             |
| Unsupervised Learning of 3D Semantic Keypoints with Mutual Reconstruction | 基于互重建的三维语义关键点无监督学习| ECCV | 2022 | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-20086-1_31) | [Github](https://github.com/YYYYYHC/Learning-3D-Keypoints-with-Mutual-Recosntruction) | 通过提出基于相互重建视角的无监督方法，解决了现有3D语义关键点检测中因隐式生成导致难以提取高层信息（如语义标签和拓扑结构）的问题，并首次实现了从无序点云显式挖掘类别级语义一致的关键点，在无需监督信息的情况下实现了语义与结构的高度一致性，为3D视觉任务提供了新的基础框架。 | 
| TexIR                                                                     | 面向大规模真实室内场景的多视角逆向渲染 | CVPR | 2023 | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Multi-View_Inverse_Rendering_for_Large-Scale_Real-World_Indoor_Scenes_CVPR_2023_paper.html) | [GitHub](https://lzleejean.github.io/TexIR) | 通过提出基于纹理光照的紧凑表示与混合光照模型，并结合三阶段物理约束的材质优化策略，解决了大规模室内场景中全局光照建模效率低下及材质-照明耦合歧义问题，实现了高精度SVBRDF重建和低噪声渲染。                                        |
| HPM-MVS                                                                   | 面向非局部多视图立体的分层先验挖掘 | ICCV | 2023 | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Ren_Hierarchical_Prior_Mining_for_Non-local_Multi-View_Stereo_ICCV_2023_paper.html) | [Github](https://github.com/CLinvx/HPM-MVS) | 通过提出分层先验挖掘框架及其三项核心技术，解决了传统MVS在低纹理区域难以平衡细节重建与全局几何一致性的问题，实现了高精度三维模型恢复并显著提升了计算效率。                                                          |
| MixCycle                                                                  | 基于MixUp增强与循环一致性约束的半监督三维单目标跟踪方法 | ICCV | 2023 | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_MixCycle_Mixup_Assisted_Semi-Supervised_3D_Single_Object_Tracking_with_Cycle_ICCV_2023_paper.html) | [Github](https://github.com/Mumuqiao/MixCycle) | 通过提出半监督框架及其两大循环一致性策略及混合增强方法，解决了传统3D单目标跟踪因依赖密集标注数据而成本高昂、鲁棒性不足的问题，实现了仅需少量标注即可在多样化点云场景中保持高精度跟踪，并显著提升了模型对运动变化和模板噪声的适应能力。                    | 
| HVTrack                                                                   | 高时空变化点云环境下的三维单目标跟踪方法| ECCV | 2024 | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72667-5_16) | [Github](https://github.com/Mumuqiao/HVTrack) | 通过提出HVTrack框架及其三个核心模块，解决了传统3D单目标跟踪在高时空变化点云中因形状突变、背景噪声干扰及相似物体混淆导致的追踪失效问题，实现了精准且稳定的三维目标跟踪能力。                                              |



#### 期刊论文

| 论文名称 | 中文论文名        | 期刊名称  | 时间   | Paper | Code | 简述                                                                                                |
|------|--------------|-------|------|-------|------|---------------------------------------------------------------------------------------------------|
|MV | 基于互投票的三维特征匹配方法 | TPAMI | 2023 | [Paper](https://ieeexplore.ieee.org/abstract/document/10105460) | [GitHub](https://github.com/NWPU-YJQ-3DV/2022_Mutual_Voting) | 通过提出基于图的互评投票框架及其三阶段关键技术，解决了传统三维对应关系评估中因异常点分布不规则和局部兼容性不足导致的可靠性差及计算效率低问题，在三维配准与识别等任务中实现了鲁棒且高效的特征匹配。 |
| MAC  | 基于极大团的点云配准方法 | TPAMI | 2024 | [Paper](https://ieeexplore.ieee.org/abstract/document/10636064) | [GitHub](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques) | 通过提出基于最大团（MAC）及其变体MAC-OP的方法，解决了传统三维点云配准中因过度依赖最大一致集而导致配准失败的问题，实现了高精度且鲁棒的三维配准。                      |
| IBI | 一种迭代的多实例配准框架 | IEEE/CAA Journal of Automatica Sinica | 2025 | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10916674) | [GitHub](https://github.com/caoxy01/IBI) | 通过提出迭代式框架及其稀疏到密集对应方法，解决了传统多实例配准方法因采用一次性框架而难以处理复杂或被遮挡实例的难题，显著提升了多实例配准性能。                           |

<a name="Category4-2"></a>

<a name="Category5"></a>
## 📷 5：多曝光图像融合
一种通过融合多张不同曝光度的图像来生成高动态范围（HDR）图像的技术。

<a name="Category5-1"></a>
### 5.1：基于多帧序列对齐的多曝光图像融合
> 基于多帧序列对齐的多曝光图像融合研究旨在融合多帧图像，其中多帧图像前景目标相对背景静止，输入序列是具有不同曝光值的图像，通过深度学习技术或者传统算法融合序列图像，最终获取一张视觉效果良好，动态范围高的图像。现针对该任务整理了相关会议期刊论文，减少后续研究者们的检索时间，以供大家开展更深层次的研究。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| Deepfuse | DeepFuse: 一种基于深度学习的无监督方法，用于极端曝光条件下的图像融合 | ICCV | 2017 | [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Prabhakar_DeepFuse_A_Deep_ICCV_2017_paper.pdf) | [GitHub](https://github.com/KRamPrabhakar/DeepFuse) | 第一篇无监督MEF任务论文，提出MEF-SSIM损失函数 |
| EBSNetMEF | 学习一种强化代理以实现灵活的曝光包围选择 | CVPR | 2020 | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_a_Reinforced_Agent_for_Flexible_Exposure_Bracketing_Selection_CVPR_2020_paper.pdf) | [GitHub](https://github.com/wzhouxiff/EBSNetMEFNet) | 第一篇使用强化学习进行MEF任务 |
| TransMEF | TransMEF：一种基于Transformer的多曝光图像融合框架，采用自监督多任务学习 | AAAI | 2022 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20109) | [GitHub](https://github.com/miccaiif/TransMEF) | 第一篇使用ViT架构进行自监督MEF任务 |
| MEFLUT | MEFLUT：用于多曝光图像融合的无监督一维查找表 | ICCV | 2023 | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_MEFLUT_Unsupervised_1D_Lookup_Tables_for_Multi-exposure_Image_Fusion_ICCV_2023_paper.pdf) | [GitHub](https://github.com/Hedlen/MEFLUT) | 使用一维查找表进行建模MEF任务，同时提出有真值数据集Mobile数据集 |
| BHFMEF | 滴水穿石：通过增强层次特征提升多曝光图像融合效果 | ACM-MM | 2023 | [Paper](https://arxiv.org/pdf/2404.06033) | [GitHub](https://github.com/ZhiyingDu/BHFMEF) | 使用颜色矫正模块，使得MEF任务在不损失指标结果下，颜色问题得到缓解 |
| IFLM | 基于视觉-语言模型的图像融合 | ICML | 2024 | [Paper](https://arxiv.org/pdf/2402.02235) | [GitHub](https://github.com/Zhaozixiang1228/IF-FILM) | 第一篇使用视觉语言大模型进行MEF任务，同时提出文本-图像数据集 |
| TC-MoA | 用于通用图像融合的任务定制化适配器混合方法 | CVPR | 2024 | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Task-Customized_Mixture_of_Adapters_for_General_Image_Fusion_CVPR_2024_paper.pdf) | [GitHub](https://github.com/YangSun22/TC-MoA) | 第一篇使用混合专家网络进行统一图像融合任务，其中包含MEF任务 |
| HSDS_MEF | 混合监督的双重搜索：利用自动学习实现无损多曝光图像融合 | AAAI | 2024 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28413) | [GitHub](https://github.com/RollingPlain/HSDS_MEF) | 使用搜索网络进行MEF任务，思路很新颖 |

#### 期刊论文

| 论文名称 | 中文论文名 | 期刊名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| SICE | 从多曝光图像中学习深度单图像对比度增强器 | TIP | 2018 | [Paper](https://drive.google.com/file/d/1HSWz1FPK3S-XSFs7awwuLHAOygEhJKIj/view) | [GitHub](https://github.com/csjcai/SICE) | 第一篇提出用于MEF任务的大型有真值图像的数据集命名为SICE数据集 |
| MEFB | 多曝光图像融合算法的基准测试与比较 | IF | 2020 | [Paper](https://arxiv.org/pdf/2007.15156) | [GitHub](https://github.com/xingchenzhang/MEFB) | 第一篇提出基准测试数据集，总共100组测试图片 |
| MEF-Net | 深度引导学习用于快速多曝光图像融合 | TIP | 2020 | [Paper](https://kedema.org/paper/19_TIP_MEF-Net.pdf) | [GitHub](https://github.com/makedede/MEFNet) | 使用序列图像权重进行快速MEF |
| SwinFusion | SwinFusion：基于Swin Transformer的跨域长程学习用于通用图像融合 | IEEE/CAA Journal of Automatica Sinica | 2022 | [Paper](https://www.ieee-jas.net/article/doi/10.1109/JAS.2022.105686) | [GitHub](https://github.com/Linfeng-Tang/SwinFusion) | 第一篇使用Swin Transformer架构的统一图像融合，其中包含了MEF任务 |
| AGAL | 基于注意力引导的全局-局部对抗学习用于细节保留的多曝光图像融合 | TCSVT | 2022 | [Paper](https://ieeexplore.ieee.org/abstract/document/9684913) | [GitHub](https://github.com/JinyuanLiu-CV/AGAL) | 使用注意力机制，全局-局部建模实现MEF任务 |
| HoLoCo | HoLoCo：用于多曝光图像融合的全局与局部对比学习网络 | IF | 2023 | [Paper](https://www.sciencedirect.com/science/article/pii/S1566253523000672) | [GitHub](https://github.com/JinyuanLiu-CV/HoLoCo) | 使用对比学习进行MEF任务，提升了视觉效果和指标 |
| CRMEF | 为鲁棒多曝光图像融合搜索紧凑架构 | TCSVT | 2024 | [Paper](https://ieeexplore.ieee.org/abstract/document/10385157) | [GitHub](https://github.com/LiuZhu-CV/CRMEF) | 使用搜索网络进行MEF任务，和HSDS一样的思路 |

<div align="center">

如果觉得项目还不错, 就点个 ⭐ Star 支持一下吧~

</div>
