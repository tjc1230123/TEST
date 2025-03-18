![image](https://github.com/user-attachments/assets/b0f6a17e-8f85-4a6a-b24b-3ef9f93e8c72)<div align="center">

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
  - [2.1：LLM](#Category2-1)
  - [2.2：VLM](#Category2-2)
  - [2.3：Video LM](#Category2-3)
  - [2.4：Agent](#Category2-4)
- [🚀 3：图像配准技术](#Category3)
- [🤖 4：三维重建技术](#Category4)
- [📷 5：多曝光图像融合](#Category5)
  - [5.1：基于多帧序列对齐的多曝光图像融合](#Category5-1)
- [🔍 6：目标检测技术](#Category6)

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

<a name="Category2-1"></a>
### 2.1：LLM
>LLM是具有海量参数的深度学习模型，专门用于处理和生成自然语言文本。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| DeepSeek-R1 | 通过强化学习激励大语言模型的推理能力 | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2501.12948) | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) | 通过大规模强化学习提升LLM推理能力，并通过知识蒸馏传递到小型模型中。 |
| Qwen 2.5 | 千问2.5技术报告 | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2412.15115) | [GitHub](https://github.com/QwenLM) | 通过高质量数据进行后期微调以贴近人类偏好。Qwen具备自然语言理解、文本生成、视觉理解、音频理解、工具使用、角色扮演、作为AI Agent进行互动等多种能力。 |
|Llama 3.2 | Llama 3模型集群 | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2407.21783) | [GitHub](https://github.com/meta-llama/llama3) | 能理解文本的多模态模型。最重要的是，能够媲美闭源模型。同时拥有超轻量1B/3B版本，解锁了更多终端设备可能性。 |
|InternLM2.5 | InternLM2技术报告 | Arxiv | 2024 | [Paper](https://github.com/InternLM/InternLM-techreport/blob/main/InternLM.pdf) | [GitHub](https://github.com/InternLM) | 能处理1M超长上下文、互联网搜索与信息整合等复杂任务。InternLM2.5目前开源了应用场景最广的轻量级7B版本，模型兼顾速度、效率与性能表现。模型全面增强了在复杂场景下的推理能力并支持1M超长上下文，能自主进行互联网搜索并从上百个网页中完成信息整合。 |
| DeepSeek-R1 | 通过强化学习激励大语言模型的推理能力 | Arxiv | 2024 | [Paper](https://arxiv.org/abs/2501.12948) | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) | 通过大规模强化学习提升LLM推理能力，并通过知识蒸馏传递到小型模型中。 |

<a name="Category2-2"></a>
### 2.2：VLM
>VLM是将视觉信息（如图像或视频）与语言信息相结合的模型。它能够处理图像字幕生成、视觉问答、图像检索等任务。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| Ovis | 多模态大型语言模型的结构嵌入对齐 | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2405.20797) | [GitHub](https://github.com/AIDC-AI/Ovis) | 相比于主流VLM使用编码器对图像进行直接编码，其构建一组视觉标记，利用视觉编码器的输出作为权重组合这些标记并作为视觉特征输入。 |
| InternVL2.5 | 用模型、数据和测试时间扩展开源多模态模型的性能边界 | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2412.05271) | [GitHub](https://github.com/OpenGVLab/InternVL) | 深入研究了模型缩放与性能之间的关系，系统地探索了视觉编码器、语言模型、数据集大小和测试时间配置的性能趋势。 |
|InternLM-XComposer | 一种支持长上下文输入和输出的通用大视觉语言模型 | Arxiv | 2024 | [Paper](https://arxiv.org/pdf/2407.03320) | [GitHub](https://github.com/InternLM/InternLM-XComposer/tree/main) | 仅使用 7B LLM 后端就达到了 GPT-4V 级别的能力。 |
|FastVLM |面向视觉语言模型的高效视觉编码 | CVPR | 2025 | [Paper](https://arxiv.org/pdf/2412.13303) | [GitHub]() | 一种在延迟、模型大小和精度之间实现优化权衡的模型。 |

<a name="Category2-3"></a>
### 2.3：Video LM
>Video LM是专门处理视频内容的语言模型，它能够理解和生成与视频相关的文本。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
|LLaVA-OneVision | 轻松实现视觉任务迁移 | CVPR | 2024 | [Paper](https://arxiv.org/pdf/2408.03326) | [GitHub](https://llava-vl.github.io/blog/llava-onevision) | 同时推动开放LMM在三个重要的计算机视觉场景中的性能边界:单图像、多图像和视频场景，允许跨不同模式/场景进行强大的迁移学习，从而产生新的新兴能力。 |
|NVILA|高效的前沿视觉语言模型 | CVPR | 2024 | [Paper](https://arxiv.org/pdf/2412.04468) | [GitHub](https://github.com/NVlabs/VILA) | 基于 VILA，通过扩展空间和时间分辨率来改进模型架构，然后对视觉 token 进行压缩。 |
| VideoLLaMA 3 | 用于图像与视频理解的前沿多模态基础模型 | CVPR | 2025 | [Paper](https://arxiv.org/pdf/2501.13106) | [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA3) | 以视觉为中心的训练范式和视觉为中心的框架设计，利用以图像为中心的数据的鲁棒性来增强视频理解。 |
| InternVideo2.5 | 赋能视频多模态大模型以支持长时序与丰富上下文建模 | CVPR | 2025 | [Paper](https://arxiv.org/pdf/2501.12386) | [GitHub](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5) | 通过长时序与丰富上下文建模提升视频多模态大语言模型的性能。 |

<a name="Category2-4"></a>
### 2.4：Agent
>在大模型领域，Agent可能结合了语言模型和视觉模型的能力，用于执行复杂的任务，如自动驾驶、机器人控制等。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
|MetaGPT | 多代理协作框架的元编程 | arxiv| 2023| [Paper](https://arxiv.org/pdf/2308.00352) | [GitHub](https://github.com/geekan/MetaGPT) | 一个创新的元编程框架，它将高效的工作流集成到基于LLM的多代理协作中。 |
|ComfyBench|在ComfyUI中基准测试基于大型语言模型的智能体，| CVPR | 2025 | [Paper](https://arxiv.org/pdf/2409.01392) | [GitHub](https://github.com/xxyQwQ/ComfyBench) | 一个综合基准测试，用于评估代理在ComfyUI中设计协作式人工智能系统的能力。 |
| VideoLLaMA 3 | 用于图像与视频理解的前沿多模态基础模型 | CVPR | 2025 | [Paper](https://arxiv.org/pdf/2501.13106) | [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA3) | 以视觉为中心的训练范式和视觉为中心的框架设计，利用以图像为中心的数据的鲁棒性来增强视频理解。 |
| ShowUI| 一种面向GUI可视化Agent的视觉语言动作模型 | CVPR | 2025 | [Paper](https://arxiv.org/pdf/2411.17465) | [GitHub](https://github.com/showlab/ShowUI) | 是一个数字世界中的视觉语言行为模型，其可以通过将屏幕截图表示为UI连通图，自适应地识别其冗余关系，并通过仔细的数据整理和重采样策略来解决显著的数据类型不平衡，构建并使用小规模高质量GUI指令。 |





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
|SuperGlue | 使用图神经网络学习特征匹配 | CVPR | 2020 | [Paper](https://arxiv.org/pdf/1911.11763) | [GitHub](https://github.com/magicleap/SuperGluePretrainedNetwork) | 一种基于图神经网络和注意力机制的特征匹配方法，通过端到端学习解决局部特征匹配中的部分分配问题 |
| LoFTR | 基于Transformer的无检测器局部特征匹配 | CVPR | 2021 | [Paper](https://arxiv.org/pdf/2104.00680) | [GitHub](https://github.com/zju3dv/LoFTR) | 提出无检测器匹配方法，通过Transformer直接建立密集的像素级匹配。 |
| TransMEF | 基于 Transformer 的自监督多任务学习多曝光图像融合框架| AAAI| 2022 | [Paper](https://arxiv.org/pdf/2112.01030) | [GitHub](https://github.com/miccaiif/TransMEF) | 通过双流架构独立提取两幅图像的特征，并利用 Transformer 的自注意力机制实现显式的多级特征匹配，结合空间变换函数生成变形场 |
| Efficient LoFTR | 具有稀疏级速度的半密集局部特征匹配 | ICCV | 2023 | [Paper](https://zju3dv.github.io/efficientloftr/files/EfficientLoFTR.pdf) | [GitHub](https://github.com/zju3dv/efficientloftr) | 通过聚合注意力机制和两阶段相关性细化提升匹配效率。 |
| LightGlue | 光速局部特征匹配 | cvpr | 2023 | [Paper](https://arxiv.org/pdf/2306.13643) | [GitHub](https://github.com/cvg/LightGlue) | 通过改进 SuperGlue 的设计，LightGlue 引入了自适应深度和宽度机制，能够在匹配过程中动态调整计算量，同时利用旋转位置编码和轻量级匹配头显著提升效率和精度。 |
| ASTR| 用于一致局部特征匹配的自适应点引导变压器 | cvpr | 2023 | [Paper](https://arxiv.org/pdf/2303.16624) | [GitHub](https://github.com/ASTR2023/ASTR?tab=readme-ov-file#astr-adaptive-spot-guided-transformer-for-consistent-local-feature-matching) | 通过引入点引导聚合模块和自适应缩放模块，在统一的粗到细架构中联合建模局部一致性和尺度变化，从而在多个标准基准测试中取得了优异的性能。 |
| DALF | 跨模态特征匹配Transformer | CVPR | 2023 | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Potje_Enhancing_Deformable_Local_Features_by_Jointly_Learning_To_Detect_and_CVPR_2023_paper.pdf) | [GitHub](https://github.com/verlab/DALF_CVPR_2023) | 通过联合学习关键点检测和描述符提取，并引入非刚性变形模块和特征融合策略，显著提升了对非刚性变形的鲁棒性和匹配性能。|
| XoFTR | 跨模态特征匹配Transformer | CVPR | 2024 | [Paper](https://arxiv.org/pdf/2404.09692) | [GitHub](https://github.com/OnderT/XoFTR) | 解决可见光与热红外图像的跨模态匹配问题，提出两阶段训练方法。 |
| OmniGlue | 基于基础模型指导的通用特征匹配 | CVPR | 2024 | [Paper](https://arxiv.org/pdf/2405.12979) | [GitHub](https://github.com/google-research/omniglue) | 通过引入基础模型 DINOv2 的指导和关键点位置引导注意力机制，实现了对未见图像域的强泛化能力。 |
|VSFormer | 用于对应性修剪的视觉空间融合变换器 | AAAI | 2024 | [Paper](https://arxiv.org/pdf/2312.08774) | [GitHub](https://github.com/sugar-fly/VSFormer) | 通过提取场景视觉线索并将其与空间线索融合，利用上下文感知先验指导对应点筛选，并结合图神经网络和变换器结构显式捕获局部和全局上下文信息，从而在室内外基准测试中显著提升了对应点筛选和相机位姿估计的性能 |

#### 期刊论文

| 论文名称 | 中文论文名        | 期刊名称  | 时间   | Paper | Code | 简述                                                                                                |
|------|--------------|-------|------|-------|------|---------------------------------------------------------------------------------------------------|
|CorMatcher| 用于局部特征匹配的角引导图神经网络 | Expert Systems with Applications |2024 | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417424020578) |  | 通过模拟人类匹配行为（先匹配角点再扩展到全图）和深度监督技术，利用角点的几何结构信息提升局部特征匹配的性能。 |
|  | |  | | [Paper]() | [GitHub]() |       |
|  | | |  | [Paper]() | [GitHub]() | |

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

<a name="Category6"></a>
## 🔍 6：目标检测技术
目标检测任务是找出图像或视频中人们感兴趣的物体，并同时检测出它们的位置和大小。不同于图像分类任务，目标检测不仅要解决分类问题，还要解决定位问题，是属于Multi-Task的问题。

#### 会议论文

| 论文名称 | 中文论文名 | 会议名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| Faster R-CNN | 基于区域建议网络实现实时目标检测 | ICCV | 2015 | [Paper](https://arxiv.org/pdf/1506.01497.pdf) | [GitHub](https://github.com/jwyang/faster-rcnn.pytorch) |  首个端到端最接近于实时性能的深度学习目标检测算法，提出了区域选择网络用于生成候选框，能极大提升检测框的生成速度。|
| SSD | 单次多框检测器：实时目标检测框架 | ECCV | 2016 | [Paper](https://arxiv.org/pdf/1512.02325) | [GitHub](https://github.com/amdegroot/ssd.pytorch) | 提出了Multi-reference和Multi-resolution的检测技术，在多尺度目标检测的精度上有了很大的提高，对小目标检测效果要好很多。 |
| YOLOv1 | 你只看到一次（YOLO）：统一的实时目标检测 | CVPR | 2016 | [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) | [GitHub](https://github.com/abeardear/pytorch-YOLO-v1) | 第一个一阶段的深度学习检测算法，其检测速度非常快。 |
| FPN | 特征金字塔网络 | CVPR | 2017 | [Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) | [GitHub](https://github.com/jwyang/fpn.pytorch) | FPN提出了一种具有横向连接的自上而下的网络架构，用于在所有具有不同尺度的高底层都构筑出高级语义信息。FPN的提出极大促进了检测网络精度的提高。 |
| RetinaNet | 为密集目标检测设计的焦点损失 | ICCV | 2017 | [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) | [GitHub](https://github.com/yhenon/pytorch-retinanet) | 分析了一阶段网络训练存在的类别不平衡问题，提出能根据Loss大小自动调节权重的Focal loss，代替了标准的交叉熵损失函数，使得模型的训练更专注于困难样本。同时，基于FPN设计了RetinaNet，在精度和速度上都有不俗的表现。 |
| CornerNet | 基于成对关键点的目标检测方法 | ECCV | 2018 | [Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.pdf) | [GitHub](https://github.com/princeton-vl/CornerNet) |  CornerNet是Anchor free技术路线的开创之作，该网络提出了一种新的对象检测方法，将网络对目标边界框的检测转化为一对关键点的检测(即左上角和右下角)，通过将对象检测为成对的关键点，而无需设计Anchor box作为先验框。 |
| CenterNet | 中心点网络：基于三元关键点组的目标检测框架 | ICCV | 2019 | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Duan_CenterNet_Keypoint_Triplets_for_Object_Detection_ICCV_2019_paper.pdf) | [GitHub](https://github.com/Duankaiwen/CenterNet) | 与CornerNet检测算法不同，CenterNet的结构十分简单，它摒弃了左上角和右下角两关键点的思路，而是直接检测目标的中心点，其它特征如大小，3D位置，方向，甚至姿态可以使用中心点位置的图像特征进行回归，是真正意义上的Anchor free。 |
| FCOS | 全卷积单阶段目标检测器 | ICCV | 2019 | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.pdf) | [GitHub](https://github.com/tianzhi0549/FCOS) | FCOS是一种基于FCN的逐像素目标检测算法，实现了无锚点(Anchor free)，无提议(Proposal free)的解决方案，并且提出了中心度Center ness的思想。 |
| DETR | 基于Transformers的端到端目标检测 | ECCV | 2020 | [Paper](https://arxiv.org/abs/2005.12872) | [GitHub](https://github.com/facebookresearch/detr) | 第一篇将Transformer引入目标检测的论文，转化为集合预测问题 |
| Deformable DETR | 可变形Transformer检测器 | ICLR | 2021 | [Paper](https://arxiv.org/abs/2010.04159) | [GitHub](https://github.com/fundamentalvision/Deformable-DETR) | 提出可变形注意力模块，解决原始DETR收敛慢的问题，在COCO上达45.4 AP，训练周期缩减10倍。 |
| Sparse R-CNN | 稀疏查询目标检测器 | CVPR | 2021 | [Paper](https://arxiv.org/abs/2011.12450) | [GitHub](https://github.com/PeizeSun/SparseR-CNN) | 使用固定数量可学习提议框（100个），COCO AP达44.5，比DETR少90%计算量 |
| Swin Transformer | 层级视觉Transformer | ICCV | 2021 | [Paper](https://arxiv.org/pdf/2103.14030.pdf) | [GitHub](https://github.com/microsoft/Swin-Transformer) | 提出层级滑动窗口注意力机制，作为检测骨干网络在COCO上达58.7 AP |
| YOLOX | YOLOX: Exceeding YOLO Series in 2021 | CVPR | 2021 | [Paper](https://arxiv.org/abs/2107.08430) | [GitHub](https://github.com/Megvii-BaseDetection/YOLOX?tab=readme-ov-file) | 首个实现Anchor-Free的YOLO变体，集成SimOTA标签分配策略，COCO AP达47.3，Tesla V100推理速度达105 FPS。 |
| DAB-DETR | 动态锚框DETR | ICLR | 2022 | [Paper](https://arxiv.org/abs/2201.12329) | [GitHub](https://github.com/IDEA-Research/DAB-DETR) | 引入动态锚框(query anchors)机制，通过坐标解耦提升检测稳定性，COCO AP达46.9，训练效率提升2.1倍。 |
| GLIP | 语言引导目标检测框架 | CVPR | 2022 | [Paper](https://arxiv.org/abs/2112.03857) | [GitHub](https://github.com/microsoft/GLIP) | 融合CLIP与检测任务，在COCO上达61.5 AP，零样本检测性能达38.2 AP |
| DINO | DINO：基于DETR的对比学习检测器 | ICLR | 2023 | [Paper](https://arxiv.org/abs/2203.03605) | [GitHub](https://github.com/IDEA-Research/DINO?tab=readme-ov-file) | 融合对比学习与DETR框架，在COCO val2017上达63.2 AP，小目标检测性能提升14.7%. |

#### 期刊论文

| 论文名称 | 中文论文名 | 期刊名称 | 时间 | Paper | Code | 简述 |
|---------|------------|----------|------|-------|------|------|
| 论文1 | 中文名1 | 期刊名 | 年份 | [Paper](link) | [GitHub](link) | 简要描述 |

<div align="center">


如果觉得项目还不错, 就点个 ⭐ Star 支持一下吧~

</div>
