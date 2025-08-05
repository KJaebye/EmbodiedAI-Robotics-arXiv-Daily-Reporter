# Visuo-Acoustic Hand Pose and Contact Estimation 

**Title (ZH)**: 视觉-听觉手部姿态和接触估计 

**Authors**: Yuemin Ma, Uksang Yoo, Yunchao Yao, Shahram Najam Syed, Luca Bondi, Jonathan Francis, Jean Oh, Jeffrey Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2508.00852)  

**Abstract**: Accurately estimating hand pose and hand-object contact events is essential for robot data-collection, immersive virtual environments, and biomechanical analysis, yet remains challenging due to visual occlusion, subtle contact cues, limitations in vision-only sensing, and the lack of accessible and flexible tactile sensing. We therefore introduce VibeMesh, a novel wearable system that fuses vision with active acoustic sensing for dense, per-vertex hand contact and pose estimation. VibeMesh integrates a bone-conduction speaker and sparse piezoelectric microphones, distributed on a human hand, emitting structured acoustic signals and capturing their propagation to infer changes induced by contact. To interpret these cross-modal signals, we propose a graph-based attention network that processes synchronized audio spectra and RGB-D-derived hand meshes to predict contact with high spatial resolution. We contribute: (i) a lightweight, non-intrusive visuo-acoustic sensing platform; (ii) a cross-modal graph network for joint pose and contact inference; (iii) a dataset of synchronized RGB-D, acoustic, and ground-truth contact annotations across diverse manipulation scenarios; and (iv) empirical results showing that VibeMesh outperforms vision-only baselines in accuracy and robustness, particularly in occluded or static-contact settings. 

**Abstract (ZH)**: 准确估计手部姿态和手物接触事件对于机器人数据收集、沉浸式虚拟环境和生物力学分析至关重要，但由于视觉遮挡、微妙的接触提示、单一视觉感知的限制以及缺乏可访问和灵活的触觉传感技术，这仍然是一个难题。因此，我们提出了VibeMesh，一种结合视觉与主动声学传感的新型穿戴系统，用于密集的手部顶点接触和姿态估计。VibeMesh集成了骨传导扬声器和分布在人体手部的稀疏压电微电话筒，发射结构化声信号并捕获其传播，以推断接触引起的改变。为了解释这些跨模态信号，我们提出了一种基于图的注意力网络，该网络处理同步的音频频谱和RGB-D转换的手部网格，以在高空间分辨率下预测接触。我们贡献了：(i) 一种轻量级、非侵入性的视听传感平台；(ii) 一种跨模态图网络，用于联合姿态和接触推断；(iii) 一个跨越多种操作场景的同步RGB-D、声学和地面真实接触标注的数据集；(iv) 实验结果表明，VibeMesh在准确性和鲁棒性方面优于单一视觉基线，尤其是在被遮挡或静态接触的场景中。 

---
# DRKF: Decoupled Representations with Knowledge Fusion for Multimodal Emotion Recognition 

**Title (ZH)**: DRKF: 解耦表示与知识融合在多模态情感识别中的应用 

**Authors**: Peiyuan Jiang, Yao Liu, Qiao Liu, Zongshun Zhang, Jiaye Yang, Lu Liu, Daibing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01644)  

**Abstract**: Multimodal emotion recognition (MER) aims to identify emotional states by integrating and analyzing information from multiple modalities. However, inherent modality heterogeneity and inconsistencies in emotional cues remain key challenges that hinder performance. To address these issues, we propose a Decoupled Representations with Knowledge Fusion (DRKF) method for MER. DRKF consists of two main modules: an Optimized Representation Learning (ORL) Module and a Knowledge Fusion (KF) Module. ORL employs a contrastive mutual information estimation method with progressive modality augmentation to decouple task-relevant shared representations and modality-specific features while mitigating modality heterogeneity. KF includes a lightweight self-attention-based Fusion Encoder (FE) that identifies the dominant modality and integrates emotional information from other modalities to enhance the fused representation. To handle potential errors from incorrect dominant modality selection under emotionally inconsistent conditions, we introduce an Emotion Discrimination Submodule (ED), which enforces the fused representation to retain discriminative cues of emotional inconsistency. This ensures that even if the FE selects an inappropriate dominant modality, the Emotion Classification Submodule (EC) can still make accurate predictions by leveraging preserved inconsistency information. Experiments show that DRKF achieves state-of-the-art (SOTA) performance on IEMOCAP, MELD, and M3ED. The source code is publicly available at this https URL. 

**Abstract (ZH)**: 多模态情感识别（MER）旨在通过整合和分析多种模态的信息来识别情感状态。然而，固有的模态异质性和情感线索的一致性问题仍然是阻碍性能的关键挑战。为了解决这些问题，我们提出了一种解耦表示与知识融合（DRKF）方法用于MER。DRKF包括两个主要模块：优化表示学习（ORL）模块和知识融合（KF）模块。ORL使用对比互信息估计方法结合逐步模态增强来解耦与任务相关的共享表示和模态特定特征，同时减轻模态异质性。KF包含一个轻量级的基于自注意力的融合编码器（FE），它可以识别主导模态并整合其他模态的情感信息以增强融合表示。为了处理在情感不一致条件下可能因错误的主导模态选择而导致的潜在错误，我们引入了一个情感鉴别子模块（ED），它强制融合表示保留情感不一致性的鉴别性线索，从而即使FE选择不合适的主导模态，情感分类子模块（EC）仍可以通过利用保留的不一致性信息做出准确预测。实验结果显示，DRKF在IEMOCAP、MELD和M3ED上取得了当前最佳性能（SOTA）。源代码已在此处公开。 

---
# Platonic Representations for Poverty Mapping: Unified Vision-Language Codes or Agent-Induced Novelty? 

**Title (ZH)**: 柏拉图式的贫困地图表示：统一的视觉-语言编码还是代理引发的 novelty？ 

**Authors**: Satiyabooshan Murugaboopathy, Connor T. Jerzak, Adel Daoud  

**Link**: [PDF](https://arxiv.org/pdf/2508.01109)  

**Abstract**: We investigate whether socio-economic indicators like household wealth leave recoverable imprints in satellite imagery (capturing physical features) and Internet-sourced text (reflecting historical/economic narratives). Using Demographic and Health Survey (DHS) data from African neighborhoods, we pair Landsat images with LLM-generated textual descriptions conditioned on location/year and text retrieved by an AI search agent from web sources. We develop a multimodal framework predicting household wealth (International Wealth Index) through five pipelines: (i) vision model on satellite images, (ii) LLM using only location/year, (iii) AI agent searching/synthesizing web text, (iv) joint image-text encoder, (v) ensemble of all signals. Our framework yields three contributions. First, fusing vision and agent/LLM text outperforms vision-only baselines in wealth prediction (e.g., R-squared of 0.77 vs. 0.63 on out-of-sample splits), with LLM-internal knowledge proving more effective than agent-retrieved text, improving robustness to out-of-country and out-of-time generalization. Second, we find partial representational convergence: fused embeddings from vision/language modalities correlate moderately (median cosine similarity of 0.60 after alignment), suggesting a shared latent code of material well-being while retaining complementary details, consistent with the Platonic Representation Hypothesis. Although LLM-only text outperforms agent-retrieved data, challenging our Agent-Induced Novelty Hypothesis, modest gains from combining agent data in some splits weakly support the notion that agent-gathered information introduces unique representational structures not fully captured by static LLM knowledge. Third, we release a large-scale multimodal dataset comprising more than 60,000 DHS clusters linked to satellite images, LLM-generated descriptions, and agent-retrieved texts. 

**Abstract (ZH)**: 我们探究了诸如家庭财富等社会经济指标是否在卫星imagery（反映物理特征）和互联网来源的文字（反映历史/经济叙事）中留下可恢复的痕迹。使用非洲社区的Demographic and Health Survey (DHS)数据，我们将Landsat图像与基于地理位置/年份的LLM生成的文本描述配对，并通过AI搜索代理从网络来源检索文本。我们开发了一种多模态框架，通过五个管道预测国际财富指数：（i）卫星图像的视觉模型，（ii）仅使用地理位置/年份的LLM，（iii）AI代理搜索/合成网络文本，（iv）联合图像-文本编码器，（v）所有信号的集成。该框架有三大贡献。首先，结合视觉和代理/LLM文本在财富预测方面优于仅视觉基线（例如，在离样本外分割上的决定系数R-squared为0.77，而仅视觉基线为0.63），且LLM内部的知识比代理检索的文本更有效，提高了跨国家和地区的一般化鲁棒性。其次，我们发现了部分表征收敛：融合视觉/语言模态的嵌入在对齐后的相关性适中（中位数余弦相似度为0.60），这表明存在共享的潜在代码，描述物质福祉，同时保留互补细节，这与理念相符。尽管仅由LLM生成的文本优于检索的代理数据，这挑战了我们的代理诱导新颖性假设，但在某些分割中将代理数据结合的微小增益部分支持了代理收集的信息引入独特表征结构的假设，这种结构未被静态LLM知识完全捕捉。第三，我们发布了一个大规模多模态数据集，包括超过60,000个与卫星图像、LLM生成描述和代理检索文本相关联的DHS集群。 

---
# Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera Fusion 

**Title (ZH)**: Hydra: 基于毫米波和摄像头融合的多模态叶湿检测方法 

**Authors**: Yimeng Liu, Maolin Gan, Huaili Zeng, Li Liu, Younsuk Dong, Zhichao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02409)  

**Abstract**: Leaf Wetness Duration (LWD), the time that water remains on leaf surfaces, is crucial in the development of plant diseases. Existing LWD detection lacks standardized measurement techniques, and variations across different plant characteristics limit its effectiveness. Prior research proposes diverse approaches, but they fail to measure real natural leaves directly and lack resilience in various environmental conditions. This reduces the precision and robustness, revealing a notable practical application and effectiveness gap in real-world agricultural settings. This paper presents Hydra, an innovative approach that integrates millimeter-wave (mm-Wave) radar with camera technology to detect leaf wetness by determining if there is water on the leaf. We can measure the time to determine the LWD based on this detection. Firstly, we design a Convolutional Neural Network (CNN) to selectively fuse multiple mm-Wave depth images with an RGB image to generate multiple feature images. Then, we develop a transformer-based encoder to capture the inherent connection among the multiple feature images to generate a feature map, which is further fed to a classifier for detection. Moreover, we augment the dataset during training to generalize our model. Implemented using a frequency-modulated continuous-wave (FMCW) radar within the 76 to 81 GHz band, Hydra's performance is meticulously evaluated on plants, demonstrating the potential to classify leaf wetness with up to 96% accuracy across varying scenarios. Deploying Hydra in the farm, including rainy, dawn, or poorly light nights, it still achieves an accuracy rate of around 90%. 

**Abstract (ZH)**: Hydra：毫米波雷达与摄像头集成检测叶片湿ness的时间duration的方法 

---
# VLM4D: Towards Spatiotemporal Awareness in Vision Language Models 

**Title (ZH)**: VLM4D: 向视知觉语言模型的空间 temporal 时间aware性迈进 

**Authors**: Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Xin Eric Wang, Achuta Kadambi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02095)  

**Abstract**: Vision language models (VLMs) have shown remarkable capabilities in integrating linguistic and visual reasoning but remain fundamentally limited in understanding dynamic spatiotemporal interactions. Humans effortlessly track and reason about object movements, rotations, and perspective shifts-abilities essential for robust dynamic real-world understanding yet notably lacking in current VLMs. In this paper, we introduce VLM4D, the first benchmark specifically designed to evaluate the spatiotemporal reasoning capabilities of VLMs. Our benchmark comprises diverse real-world and synthetic videos accompanied by carefully curated question-answer pairs emphasizing translational and rotational motions, perspective awareness, and motion continuity. Through comprehensive evaluations of state-of-the-art open and closed-source VLMs, we identify significant performance gaps compared to human baselines, highlighting fundamental deficiencies in existing models. Extensive analysis reveals that VLMs struggle particularly with integrating multiple visual cues and maintaining temporal coherence. We further explore promising directions, such as leveraging 4D feature field reconstruction and targeted spatiotemporal supervised fine-tuning, demonstrating their effectiveness in enhancing spatiotemporal comprehension. Our work aims to encourage deeper exploration into improving VLMs' spatial and temporal grounding, paving the way towards more capable and reliable visual intelligence for dynamic environments. 

**Abstract (ZH)**: Vision语言模型（VLMs）在整合语言和视觉推理方面表现出色，但在理解动态时空交互方面仍存在根本性限制。人类能够轻松追踪和推理物体的运动、旋转和视角变化——这些能力对于实现稳健的动态现实理解至关重要，而在当前的VLMs中却明显缺乏。在本文中，我们介绍了VLM4D，这是首个专门用于评估VLMs时空推理能力的基准。我们的基准包含多样化的现实世界和合成视频，并附带了精心设计的问题-答案对，强调了平移和旋转运动、视角意识以及运动连续性。通过全面评估最先进的开源和闭源VLMs，我们发现与人类基线相比存在显著性能差距，突显了现有模型的基本缺陷。广泛分析表明，VLMs特别难以整合多种视觉线索并保持时间连贯性。我们进一步探讨了一些有前景的方向，如利用四维特征场重建和针对性的时空监督微调，证明了这些方法在增强时空理解方面的有效性。我们的工作旨在鼓励更深入地探索改进VLMs的空间和时间定位，为动态环境中的更强大和可靠的视觉智能铺平道路。 

---
# GAID: Frame-Level Gated Audio-Visual Integration with Directional Perturbation for Text-Video Retrieval 

**Title (ZH)**: GAID：具有方向扰动的帧级门控音视频集成用于文本视频检索 

**Authors**: Bowen Yang, Yun Cao, Chen He, Xiaosu Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.01711)  

**Abstract**: Text-to-video retrieval requires precise alignment between language and temporally rich video signals. Existing methods predominantly exploit visual cues and often overlook complementary audio semantics or adopt coarse fusion strategies, leading to suboptimal multimodal representations. We present GAID, a framework that jointly address this gap via two key components: (i) a Frame-level Gated Fusion (FGF) that adaptively integrates audio and visual features under textual guidance, enabling fine-grained temporal alignment; and (ii) a Directional Adaptive Semantic Perturbation (DASP) that injects structure-aware perturbations into text embeddings, enhancing robustness and discrimination without incurring multi-pass inference. These modules complement each other -- fusion reduces modality gaps while perturbation regularizes cross-modal matching -- yielding more stable and expressive representations. Extensive experiments on MSR-VTT, DiDeMo, LSMDC, and VATEX show consistent state-of-the-art results across all retrieval metrics with notable efficiency gains. Our code is available at this https URL. 

**Abstract (ZH)**: 文本到视频检索要求语言和时空丰富的视频信号之间精确对齐。现有方法主要利用视觉线索，往往忽视互补的音频语义或采用粗放的融合策略，导致多模态表示不够优化。我们提出了一种GAID框架，通过两个关键组件共同解决这一问题：（i）帧级门控融合（FGF），在文本引导下适应性地整合音频和视觉特征，实现细粒度的时间对齐；（ii）方向自适应语义扰动（DASP），在文本嵌入中注入结构感知的扰动，增强鲁棒性和区分性而无需多遍推断。这些模块相互补充——融合减少了模态差异，而扰动正则化跨模态匹配——从而产生更稳定和更具表达力的表示。在MSR-VTT、DiDeMo、LSMDC和VATEX上的广泛实验显示，这些模块在所有检索指标上取得了一致的最新成果，并且具有显著的效率提升。我们的代码可在以下网址获取。 

---
# Cure or Poison? Embedding Instructions Visually Alters Hallucination in Vision-Language Models 

**Title (ZH)**: 治愈还是毒药？视觉嵌入指令改变视觉语言模型的幻觉 

**Authors**: Zhaochen Wang, Yiwei Wang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01678)  

**Abstract**: Vision-Language Models (VLMs) often suffer from hallucination, partly due to challenges in aligning multimodal information. We propose Prompt-in-Image, a simple method that embeds textual instructions directly into images. This removes the need for separate text inputs and forces the model to process all content through the visual channel. We evaluate this method on three popular open-source VLMs: Qwen2.5-VL, LLaVA-1.5, and InstructBLIP. The results reveal sharp differences. Prompt-in-Image improves Qwen2.5-VL's performance, increasing POPE accuracy by 4.1 percent (from 80.2 percent to 84.3 percent) and also reducing hallucination rates on MS-COCO. In contrast, LLaVA-1.5 and InstructBLIP experience a severe performance drop, with accuracy falling from around 84 percent to near-random levels. Through detailed analysis, we found that CLIP-based encoders in LLaVA and InstructBLIP exhibit excessive attention bias toward embedded text regions, disrupting visual understanding. In contrast, Qwen's vision encoder handles text-embedded images robustly. Crucially, Prompt-in-Image reduces Qwen's modality gap, enhancing cross-modal alignment by unifying information processing through a single modality. 

**Abstract (ZH)**: 基于图像的提示方法：Vision-Language模型中的文本指令直接嵌入图像以改善多模态对齐 

---
# MAP: Mitigating Hallucinations in Large Vision-Language Models with Map-Level Attention Processing 

**Title (ZH)**: MAP: Map级别注意力处理在大型视觉-语言模型中减轻幻觉问题 

**Authors**: Chenxi Li, Yichen Guo, Benfang Qian, Jinhao You, Kai Tang, Yaosong Du, Zonghao Zhang, Xiande Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01653)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved impressive performance in multimodal tasks, but they still suffer from hallucinations, i.e., generating content that is grammatically accurate but inconsistent with visual inputs. In this work, we introduce a novel map-level perspective to mitigate hallucinations in LVLMs, interpreting the hidden states of the model as a 2D semantic map. We observe that factual information is widely distributed across this map, extending beyond the localized inter- or intra-layer regions targeted by most existing methods (e.g., contrastive decoding and layer-wise consistency). Building on this insight, we propose Map-Level Attention Processing (MAP), a training-free decoding method that effectively leverages factual information through attention-based map-level operations to improve factual consistency. Specifically, we employ Layer-Wise Criss-Cross Attention to progressively refine token representations at each decoding layer by aggregating tokens from both inter- and intra-layer dimensions. Additionally, a Global-Local Logit Fusion mechanism combines logits obtained before and after global attention to further refine predictions and improve accuracy. Our method consistently improves the truthfulness and performance of LVLMs across benchmarks, such as POPE, MME, and MMHal-Bench, demonstrating the potential of the map-level decoding strategy. 

**Abstract (ZH)**: Large Vision-Language Models (LVLMs)在多模态任务中取得了令人印象深刻的性能，但仍存在幻觉问题，即生成语法规正确但与视觉输入不一致的内容。在本文中，我们提出了一种新的地图级视角来减轻LVLMs中的幻觉问题，将模型的隐藏状态解释为2D语义地图。我们观察到，事实信息在该地图上分布广泛，超越了大多数现有方法（例如对比解码和逐层一致性）所关注的局部跨层或同一层区域。基于这一洞察，我们提出了一种无需训练的解码方法——地图级注意力处理（MAP），通过基于注意力的地图级操作有效利用事实信息，提高事实一致性。具体来说，我们使用逐层交叉注意力，逐层逐步细化token表示，通过从跨层和同一层维度聚合token来实现。此外，全局-局部logit融合机制结合了在全局注意力前后获得的logits，进一步细化预测并提高准确性。我们的方法在POPE、MME和MMHal-Bench等基准测试中一致地提高了LVLMs的真相度和性能，表明地图级解码策略的潜力。 

---
# DMTrack: Spatio-Temporal Multimodal Tracking via Dual-Adapter 

**Title (ZH)**: DMTrack: 基于双适配器的时空多模态跟踪 

**Authors**: Weihong Li, Shaohua Dong, Haonan Lu, Yanhao Zhang, Heng Fan, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01592)  

**Abstract**: In this paper, we explore adapter tuning and introduce a novel dual-adapter architecture for spatio-temporal multimodal tracking, dubbed DMTrack. The key of our DMTrack lies in two simple yet effective modules, including a spatio-temporal modality adapter (STMA) and a progressive modality complementary adapter (PMCA) module. The former, applied to each modality alone, aims to adjust spatio-temporal features extracted from a frozen backbone by self-prompting, which to some extent can bridge the gap between different modalities and thus allows better cross-modality fusion. The latter seeks to facilitate cross-modality prompting progressively with two specially designed pixel-wise shallow and deep adapters. The shallow adapter employs shared parameters between the two modalities, aiming to bridge the information flow between the two modality branches, thereby laying the foundation for following modality fusion, while the deep adapter modulates the preliminarily fused information flow with pixel-wise inner-modal attention and further generates modality-aware prompts through pixel-wise inter-modal attention. With such designs, DMTrack achieves promising spatio-temporal multimodal tracking performance with merely \textbf{0.93M} trainable parameters. Extensive experiments on five benchmarks show that DMTrack achieves state-of-the-art results. Code will be available. 

**Abstract (ZH)**: 基于时空多模态跟踪的双适配器架构DMTrack研究 

---
# MagicVL-2B: Empowering Vision-Language Models on Mobile Devices with Lightweight Visual Encoders via Curriculum Learning 

**Title (ZH)**: MagicVL-2B: 通过分阶段学习增强轻量级视觉编码器在移动设备上的多模态语言模型能力 

**Authors**: Yi Liu, Xiao Xu, Zeyu Xu, Meng Zhang, Yibo Li, Haoyu Chen, Junkang Zhang, Qiang Wang, Jifa Sun, Siling Lin, Shengxun Cheng, Lingshu Zhang, Kang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01540)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable breakthroughs in recent years, enabling a diverse array of applications in everyday life. However, the substantial computational and storage demands of VLMs pose significant challenges for their efficient deployment on mobile devices, which represent the most ubiquitous and accessible computing platforms today. In this work, we introduce MagicVL-2B, a novel VLM meticulously optimized for flagship smartphones. MagicVL-2B leverages a lightweight visual encoder with fewer than 100M parameters and features a redesigned dynamic resolution scheme that adaptively generates image tokens without excessive modification of image dimensions. To further enhance the performance of this compact encoder within VLMs, we propose a multimodal curriculum learning strategy that incrementally increases task difficulty and data information density throughout training. This approach substantially improves the model's performance across a variety of sub-tasks. Extensive evaluations on standard VLM benchmarks demonstrate that MagicVL-2B matches the accuracy of current state-of-the-art models while reducing on-device power consumption by 41.1%. These results establish MagicVL-2B as a practical and robust solution for real-world mobile vision-language applications, enabling advanced multimodal intelligence to run directly on smartphones. 

**Abstract (ZH)**: MagicVL-2B：面向旗舰智能手机的高效视觉-语言模型 

---
# MiraGe: Multimodal Discriminative Representation Learning for Generalizable AI-Generated Image Detection 

**Title (ZH)**: MiraGe：多模态区分性表示学习在通用AI生成图像检测中的应用 

**Authors**: Kuo Shi, Jie Lu, Shanshan Ye, Guangquan Zhang, Zhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01525)  

**Abstract**: Recent advances in generative models have highlighted the need for robust detectors capable of distinguishing real images from AI-generated images. While existing methods perform well on known generators, their performance often declines when tested with newly emerging or unseen generative models due to overlapping feature embeddings that hinder accurate cross-generator classification. In this paper, we propose Multimodal Discriminative Representation Learning for Generalizable AI-generated Image Detection (MiraGe), a method designed to learn generator-invariant features. Motivated by theoretical insights on intra-class variation minimization and inter-class separation, MiraGe tightly aligns features within the same class while maximizing separation between classes, enhancing feature discriminability. Moreover, we apply multimodal prompt learning to further refine these principles into CLIP, leveraging text embeddings as semantic anchors for effective discriminative representation learning, thereby improving generalizability. Comprehensive experiments across multiple benchmarks show that MiraGe achieves state-of-the-art performance, maintaining robustness even against unseen generators like Sora. 

**Abstract (ZH)**: Recent advances in生成模型的最新进展强调了需要 Robust Detectors 能够区分真实图像与AI生成图像的强健检测器。现有的方法在已知生成器上表现良好，但在测试新兴或未见过的生成器时，由于特征嵌入的重叠而影响准确的跨生成器分类。本文提出了一种名为MiraGe的方法：面向通用AI生成图像检测的多模态判别表示学习，旨在学习生成器不变特征。受最小化类别内变分和最大化类别间分离的理论启发，MiraGe在同一个类别内紧密对齐特征，同时最大化类间分离，提升特征可判别性。此外，我们应用多模态提示学习进一步细化这些原则，并利用CLIP进行有效判别表示学习，通过文本嵌入作为语义锚点来提高泛化能力。多项基准上的综合实验表明，MiraGe达到了最先进的性能，即使在未见过的生成器如Sora的情况下也保持了鲁棒性。 

---
# Video-based Vehicle Surveillance in the Wild: License Plate, Make, and Model Recognition with Self Reflective Vision-Language Models 

**Title (ZH)**: 基于视频的野外车辆监控：自反性视觉-语言模型的车牌、品牌和型号识别 

**Authors**: Pouya Parsa, Keya Li, Kara M. Kockelman, Seongjin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01387)  

**Abstract**: Automatic license plate recognition (ALPR) and vehicle make and model recognition underpin intelligent transportation systems, supporting law enforcement, toll collection, and post-incident investigation. Applying these methods to videos captured by handheld smartphones or non-static vehicle-mounted cameras presents unique challenges compared to fixed installations, including frequent camera motion, varying viewpoints, occlusions, and unknown road geometry. Traditional ALPR solutions, dependent on specialized hardware and handcrafted OCR pipelines, often degrade under these conditions. Recent advances in large vision-language models (VLMs) enable direct recognition of textual and semantic attributes from arbitrary imagery. This study evaluates the potential of VLMs for ALPR and makes and models recognition using monocular videos captured with handheld smartphones and non-static mounted cameras. The proposed license plate recognition pipeline filters to sharp frames, then sends a multimodal prompt to a VLM using several prompt strategies. Make and model recognition pipeline runs the same VLM with a revised prompt and an optional self-reflection module. In the self-reflection module, the model contrasts the query image with a reference from a 134-class dataset, correcting mismatches. Experiments on a smartphone dataset collected on the campus of the University of Texas at Austin, achieve top-1 accuracies of 91.67% for ALPR and 66.67% for make and model recognition. On the public UFPR-ALPR dataset, the approach attains 83.05% and 61.07%, respectively. The self-reflection module further improves results by 5.72% on average for make and model recognition. These findings demonstrate that VLMs provide a cost-effective solution for scalable, in-motion traffic video analysis. 

**Abstract (ZH)**: 基于单目手机视频和非静止安装摄像头的自动车牌识别和车辆品牌型号识别：利用大型视觉语言模型的潜力 

---
# Effective Damage Data Generation by Fusing Imagery with Human Knowledge Using Vision-Language Models 

**Title (ZH)**: 基于视觉-语言模型融合图像与人类知识的有效损伤数据生成 

**Authors**: Jie Wei, Erika Ardiles-Cruz, Aleksey Panasyuk, Erik Blasch  

**Link**: [PDF](https://arxiv.org/pdf/2508.01380)  

**Abstract**: It is of crucial importance to assess damages promptly and accurately in humanitarian assistance and disaster response (HADR). Current deep learning approaches struggle to generalize effectively due to the imbalance of data classes, scarcity of moderate damage examples, and human inaccuracy in pixel labeling during HADR situations. To accommodate for these limitations and exploit state-of-the-art techniques in vision-language models (VLMs) to fuse imagery with human knowledge understanding, there is an opportunity to generate a diversified set of image-based damage data effectively. Our initial experimental results suggest encouraging data generation quality, which demonstrates an improvement in classifying scenes with different levels of structural damage to buildings, roads, and infrastructures. 

**Abstract (ZH)**: 在人道主义援助与灾难响应中及时准确评估损害至关重要。当前的深度学习方法由于数据类别的不平衡、中等损害示例稀缺以及在灾难响应情况下像素标注的人为不准确，难以有效泛化。为克服这些限制并利用视觉-语言模型（VLMs）的最新技术将图像与人类知识理解相结合，有机会有效生成多样化的目标损害数据集。我们的初步实验结果表明，生成的数据质量令人鼓舞，展示了在区分不同结构损害水平的场景方面有所改进，涉及建筑物、道路和基础设施。 

---
# Masked Omics Modeling for Multimodal Representation Learning across Histopathology and Molecular Profiles 

**Title (ZH)**: 掩码Omics建模在组织病理学和分子特征跨模态表示学习中的应用 

**Authors**: Lucas Robinet, Ahmad Berjaoui, Elizabeth Cohen-Jonathan Moyal  

**Link**: [PDF](https://arxiv.org/pdf/2508.00969)  

**Abstract**: Self-supervised learning has driven major advances in computational pathology by enabling models to learn rich representations from hematoxylin and eosin (H&E)-stained cancer tissue. However, histopathology alone often falls short for molecular characterization and understanding clinical outcomes, as important information is contained in high-dimensional omics profiles like transcriptomics, methylomics, or genomics. In this work, we introduce MORPHEUS, a unified transformer-based pre-training framework that encodes both histopathology and multi-omics data into a shared latent space. At its core, MORPHEUS relies on a masked modeling objective applied to randomly selected omics portions, encouraging the model to learn biologically meaningful cross-modal relationships. The same pre-trained network can be applied to histopathology alone or in combination with any subset of omics modalities, seamlessly adapting to the available inputs. Additionally, MORPHEUS enables any-to-any omics generation, enabling one or more omics profiles to be inferred from any subset of modalities, including H&E alone. Pre-trained on a large pan-cancer cohort, MORPHEUS consistently outperforms state-of-the-art methods across diverse modality combinations and tasks, positioning itself as a promising framework for developing multimodal foundation models in oncology. The code is available at: this https URL 

**Abstract (ZH)**: 自我监督学习通过使模型能够从苏木精和曙红（H&E）染色的癌组织中学习丰富的表示，已在计算病理学领域取得了重大进展。然而，仅靠组织病理学往往不足以进行分子表征和理解临床结局，因为重要信息包含在转录组学、甲基组学或基因组学等高维组学档案中。在此工作中，我们介绍了MORPHEUS，一个统一的基于变换器的预训练框架，将组织病理学和多种组学数据编码到共享的潜在空间中。MORPHEUS的核心在于应用于随机选择的组学部分的遮蔽建模目标，促使模型学习生物意义的跨模态关系。预训练后的同一个网络可以仅应用于组织病理学，或与任意一组组学模态结合使用，无缝适应可用的输入。此外，MORPHEUS还实现了任意到任意组学生成，允许从任意一组模态，包括仅H&E染色，推断一个或多个组学档案。MORPHEUS在泛癌种队列上进行预训练，一致优于多种模态组合和任务下的先进方法，使其成为肿瘤学领域开发多模态基础模型的一个有前景的框架。代码可通过以下链接获取：this https URL 

---
# Rethinking Multimodality: Optimizing Multimodal Deep Learning for Biomedical Signal Classification 

**Title (ZH)**: 重新思考多媒体性：优化多模态深度学习在生物医学信号分类中的应用 

**Authors**: Timothy Oladunni, Alex Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00963)  

**Abstract**: This study proposes a novel perspective on multimodal deep learning for biomedical signal classification, systematically analyzing how complementary feature domains impact model performance. While fusing multiple domains often presumes enhanced accuracy, this work demonstrates that adding modalities can yield diminishing returns, as not all fusions are inherently advantageous. To validate this, five deep learning models were designed, developed, and rigorously evaluated: three unimodal (1D-CNN for time, 2D-CNN for time-frequency, and 1D-CNN-Transformer for frequency) and two multimodal (Hybrid 1, which fuses 1D-CNN and 2D-CNN; Hybrid 2, which combines 1D-CNN, 2D-CNN, and a Transformer). For ECG classification, bootstrapping and Bayesian inference revealed that Hybrid 1 consistently outperformed the 2D-CNN baseline across all metrics (p-values < 0.05, Bayesian probabilities > 0.90), confirming the synergistic complementarity of the time and time-frequency domains. Conversely, Hybrid 2's inclusion of the frequency domain offered no further improvement and sometimes a marginal decline, indicating representational redundancy; a phenomenon further substantiated by a targeted ablation study. This research redefines a fundamental principle of multimodal design in biomedical signal analysis. We demonstrate that optimal domain fusion isn't about the number of modalities, but the quality of their inherent complementarity. This paradigm-shifting concept moves beyond purely heuristic feature selection. Our novel theoretical contribution, "Complementary Feature Domains in Multimodal ECG Deep Learning," presents a mathematically quantifiable framework for identifying ideal domain combinations, demonstrating that optimal multimodal performance arises from the intrinsic information-theoretic complementarity among fused domains. 

**Abstract (ZH)**: 本研究提出了一种新的多模态深度学习在生物医学信号分类中的视角，系统分析了互补特征域如何影响模型性能。虽然融合多个领域通常假定会提高准确性，但本工作证明增加模态可能会适得其反，因为并非所有融合都是固有的有利的。为此，设计、开发并严格评估了五种深度学习模型：三种单模态（时间域的1D-CNN、时频域的2D-CNN、频率域的1D-CNN-Transformer）和两种多模态（Hybrid 1，融合1D-CNN和2D-CNN；Hybrid 2，结合1D-CNN、2D-CNN和Transformer）。对于ECG分类，通过自助采样和贝叶斯推断发现，Hybrid 1在所有指标上始终优于2D-CNN基准（p值<0.05，贝叶斯概率>0.90），证实了时间域和时频域的有效互补性。相反，Hybrid 2引入频率域没有带来进一步的改善，有时甚至略有下降，这表明存在表示冗余；这一现象在针对性的消融研究中进一步得到了证实。本研究重新定义了多模态设计在生物医学信号分析中的基本原则。我们证明了最佳领域融合不是关于模态的数量，而是关于其内在互补性的质量。这一范式转变的概念超越了纯粹的经验特征选择。我们的新型理论贡献“多模态ECG深度学习中的互补特征域”提供了一个可量化识别理想领域组合的数学框架，证明了最佳多模态性能源于融合域之间的固有信息论互补性。 

---
