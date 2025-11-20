# Think Visually, Reason Textually: Vision-Language Synergy in ARC 

**Title (ZH)**: 可视化思考，文本推理：ARC中的视觉-语言协同作用 

**Authors**: Beichen Zhang, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15703)  

**Abstract**: Abstract reasoning from minimal examples remains a core unsolved problem for frontier foundation models such as GPT-5 and Grok 4. These models still fail to infer structured transformation rules from a handful of examples, which is a key hallmark of human intelligence. The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) provides a rigorous testbed for this capability, demanding conceptual rule induction and transfer to novel tasks. Most existing methods treat ARC-AGI as a purely textual reasoning task, overlooking the fact that humans rely heavily on visual abstraction when solving such puzzles. However, our pilot experiments reveal a paradox: naively rendering ARC-AGI grids as images degrades performance due to imprecise rule execution. This leads to our central hypothesis that vision and language possess complementary strengths across distinct reasoning stages: vision supports global pattern abstraction and verification, whereas language specializes in symbolic rule formulation and precise execution. Building on this insight, we introduce two synergistic strategies: (1) Vision-Language Synergy Reasoning (VLSR), which decomposes ARC-AGI into modality-aligned subtasks; and (2) Modality-Switch Self-Correction (MSSC), which leverages vision to verify text-based reasoning for intrinsic error correction. Extensive experiments demonstrate that our approach yields up to a 4.33% improvement over text-only baselines across diverse flagship models and multiple ARC-AGI tasks. Our findings suggest that unifying visual abstraction with linguistic reasoning is a crucial step toward achieving generalizable, human-like intelligence in future foundation models. Source code will be released soon. 

**Abstract (ZH)**: 抽象推理从最小样本出发仍然是前沿基础模型如GPT-5和Grok 4的核心未解问题。这些模型仍然无法从少量示例中推断出结构化的转换规则，这是人类智能的一个关键标志。人工通用智能抽象与推理语料库（ARC-AGI）为这种能力提供了一个严格的测试平台，要求概念规则归纳和向新颖任务的转移。现有大多数方法将ARC-AGI视为纯粹的文字推理任务，忽视了人类在解决这类谜题时高度依赖视觉抽象的事实。然而，我们的初步实验揭示了一个悖论：将ARC-AGI网格直接渲染为图像会导致性能下降，因为规则执行不够精确。这让我们形成一个中心假设，即视觉和语言在不同的推理阶段具有互补的优势：视觉支持全局模式的抽象和验证，而语言则专门负责符号规则的制定和精确执行。基于这一见解，我们引入了两种协同策略：（1）视觉-语言协同推理（VLSR），将ARC-AGI分解为模态对齐的子任务；（2）模式切换自校正（MSSC），利用视觉验证基于文本的推理以进行内在错误校正。广泛实验表明，我们的方法在多种旗舰模型和ARC-AGI任务中相较于纯文本基准提高了多达4.33%的表现。我们的研究结果表明，将视觉抽象与语言推理统一起来是未来基础模型实现可泛化的、类人的智能的关键步骤之一。源代码即将发布。 

---
# GEO-Bench-2: From Performance to Capability, Rethinking Evaluation in Geospatial AI 

**Title (ZH)**: GEO-Bench-2: 从性能到能力，重新思考地理空间AI的评估 

**Authors**: Naomi Simumba, Nils Lehmann, Paolo Fraccaro, Hamed Alemohammad, Geeth De Mel, Salman Khan, Manil Maskey, Nicolas Longepe, Xiao Xiang Zhu, Hannah Kerner, Juan Bernabe-Moreno, Alexander Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2511.15658)  

**Abstract**: Geospatial Foundation Models (GeoFMs) are transforming Earth Observation (EO), but evaluation lacks standardized protocols. GEO-Bench-2 addresses this with a comprehensive framework spanning classification, segmentation, regression, object detection, and instance segmentation across 19 permissively-licensed datasets. We introduce ''capability'' groups to rank models on datasets that share common characteristics (e.g., resolution, bands, temporality). This enables users to identify which models excel in each capability and determine which areas need improvement in future work. To support both fair comparison and methodological innovation, we define a prescriptive yet flexible evaluation protocol. This not only ensures consistency in benchmarking but also facilitates research into model adaptation strategies, a key and open challenge in advancing GeoFMs for downstream tasks.
Our experiments show that no single model dominates across all tasks, confirming the specificity of the choices made during architecture design and pretraining. While models pretrained on natural images (ConvNext ImageNet, DINO V3) excel on high-resolution tasks, EO-specific models (TerraMind, Prithvi, and Clay) outperform them on multispectral applications such as agriculture and disaster response. These findings demonstrate that optimal model choice depends on task requirements, data modalities, and constraints. This shows that the goal of a single GeoFM model that performs well across all tasks remains open for future research. GEO-Bench-2 enables informed, reproducible GeoFM evaluation tailored to specific use cases. Code, data, and leaderboard for GEO-Bench-2 are publicly released under a permissive license. 

**Abstract (ZH)**: GeoFMs的地理空间基础模型正在_transforming地球观测(EO)，但评估缺乏标准化协议。GEO-Bench-2通过涵盖分类、分割、回归、对象检测和实例分割的全面框架，跨越了19个许可使用的数据集，解决了这一问题。我们引入“能力”组，按共享共同特征（例如，分辨率、波段、时间性）的数据集对模型进行排名。这使用户能够确定哪些模型在每个能力方面表现最佳，并确定未来工作需要改进的领域。为了支持公平比较和方法创新，我们定义了一种规范但灵活的评估协议。这不仅确保了基准测试的一致性，还促进了对模型适应策略的研究，这是推进GeoFMs用于下游任务的一个重要且开放的挑战。 

---
# The SA-FARI Dataset: Segment Anything in Footage of Animals for Recognition and Identification 

**Title (ZH)**: SA-FARI数据集：动物影像中的目标分割以实现识别和鉴定 

**Authors**: Dante Francisco Wasmuht, Otto Brookes, Maximillian Schall, Pablo Palencia, Chris Beirne, Tilo Burghardt, Majid Mirmehdi, Hjalmar Kühl, Mimi Arandjelovic, Sam Pottie, Peter Bermant, Brandon Asheim, Yi Jin Toh, Adam Elzinga, Jason Holmberg, Andrew Whitworth, Eleanor Flatt, Laura Gustafson, Chaitanya Ryali, Yuan-Ting Hu, Baishan Guo, Andrew Westbury, Kate Saenko, Didac Suris  

**Link**: [PDF](https://arxiv.org/pdf/2511.15622)  

**Abstract**: Automated video analysis is critical for wildlife conservation. A foundational task in this domain is multi-animal tracking (MAT), which underpins applications such as individual re-identification and behavior recognition. However, existing datasets are limited in scale, constrained to a few species, or lack sufficient temporal and geographical diversity - leaving no suitable benchmark for training general-purpose MAT models applicable across wild animal populations. To address this, we introduce SA-FARI, the largest open-source MAT dataset for wild animals. It comprises 11,609 camera trap videos collected over approximately 10 years (2014-2024) from 741 locations across 4 continents, spanning 99 species categories. Each video is exhaustively annotated culminating in ~46 hours of densely annotated footage containing 16,224 masklet identities and 942,702 individual bounding boxes, segmentation masks, and species labels. Alongside the task-specific annotations, we publish anonymized camera trap locations for each video. Finally, we present comprehensive benchmarks on SA-FARI using state-of-the-art vision-language models for detection and tracking, including SAM 3, evaluated with both species-specific and generic animal prompts. We also compare against vision-only methods developed specifically for wildlife analysis. SA-FARI is the first large-scale dataset to combine high species diversity, multi-region coverage, and high-quality spatio-temporal annotations, offering a new foundation for advancing generalizable multianimal tracking in the wild. The dataset is available at $\href{this https URL}{\text{this http URL}}$. 

**Abstract (ZH)**: 自动视频分析对于野生动物保护至关重要。该领域的一个基础任务是多动物追踪（MAT），其支撑着个体再识别和行为识别等应用。然而，现有的数据集在规模、物种限制或时空多样性方面存在局限性，缺乏适用于跨野生动物种群的一般性MAT模型的基准。为解决这一问题，我们引入了SA-FARI，这是最大的开放源多动物追踪数据集，用于野生動物。该数据集包含从四大洲741个地点收集的约10年（2014-2024）时间跨度的11,609个相机陷阱视频，涵盖99种物种类别。每个视频都被详尽标注，总计约46小时密集标注的视频片段，包含16,224个掩码身份和942,702个个体边界框、分割掩码和物种标签。除了特定任务的标注外，我们还发布了每个视频的匿名相机陷阱位置。最后，我们使用最新的视觉-语言模型对SA-FARI进行全面基准测试，包括SAM 3，该模型用特定物种和通用动物提示进行评估。我们还与专门为野生动物分析开发的仅视觉方法进行了比较。SA-FARI是第一个结合高物种多样性、多区域覆盖和高质量时空标注的大规模数据集，为推进通用多动物追踪提供了新的基础。数据集可以访问：this https URL 

---
# CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking 

**Title (ZH)**: CompTrack: 信息瓶颈引导的低秩动态令牌压缩用于点云跟踪 

**Authors**: Sifan Zhou, Yichao Cao, Jiahao Nie, Yuqian Fu, Ziyu Zhao, Xiaobo Lu, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15580)  

**Abstract**: 3D single object tracking (SOT) in LiDAR point clouds is a critical task in computer vision and autonomous driving. Despite great success having been achieved, the inherent sparsity of point clouds introduces a dual-redundancy challenge that limits existing trackers: (1) vast spatial redundancy from background noise impairs accuracy, and (2) informational redundancy within the foreground hinders efficiency. To tackle these issues, we propose CompTrack, a novel end-to-end framework that systematically eliminates both forms of redundancy in point clouds. First, CompTrack incorporates a Spatial Foreground Predictor (SFP) module to filter out irrelevant background noise based on information entropy, addressing spatial redundancy. Subsequently, its core is an Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module that eliminates the informational redundancy within the foreground. Theoretically grounded in low-rank approximation, this module leverages an online SVD analysis to adaptively compress the redundant foreground into a compact and highly informative set of proxy tokens. Extensive experiments on KITTI, nuScenes and Waymo datasets demonstrate that CompTrack achieves top-performing tracking performance with superior efficiency, running at a real-time 90 FPS on a single RTX 3090 GPU. 

**Abstract (ZH)**: 基于LiDAR点云的3D单目标跟踪（SOT）是计算机视觉和自动驾驶领域的一个关键任务。尽管已经取得了显著的成功，点云的固有稀疏性引入了双重冗余挑战，限制了现有跟踪器：（1）背景噪声带来的巨大空间冗余影响了准确性，（2）前景内的信息冗余阻碍了效率。为了应对这些问题，我们提出了CompTrack，这是一种新颖的端到端框架，系统地消除了点云中的两种冗余。首先，CompTrack引入了一个基于信息熵的 Spatial Foreground Predictor (SFP) 模块，用于过滤掉无关的背景噪声，解决了空间冗余问题。其次，其核心是一个基于Information Bottleneck的动态token压缩（IB-DTC）模块，用于消除前景内的信息冗余。该模块理论上基于低秩逼近，通过在线SVD分析自适应地将冗余前景压缩成一个紧凑且高度信息化的代理token集合。在KITTI、nuScenes和Waymo数据集上的 extensive 实验表明，CompTrack在保持卓越效率的同时实现了顶级的跟踪性能，可在单个RTX 3090 GPU上以90 FPS 实时运行。 

---
# Evaluating Low-Light Image Enhancement Across Multiple Intensity Levels 

**Title (ZH)**: 多强度等级下低光照图像增强的评估 

**Authors**: Maria Pilligua, David Serrano-Lozano, Pai Peng, Ramon Baldrich, Michael S. Brown, Javier Vazquez-Corral  

**Link**: [PDF](https://arxiv.org/pdf/2511.15496)  

**Abstract**: Imaging in low-light environments is challenging due to reduced scene radiance, which leads to elevated sensor noise and reduced color saturation. Most learning-based low-light enhancement methods rely on paired training data captured under a single low-light condition and a well-lit reference. The lack of radiance diversity limits our understanding of how enhancement techniques perform across varying illumination intensities. We introduce the Multi-Illumination Low-Light (MILL) dataset, containing images captured at diverse light intensities under controlled conditions with fixed camera settings and precise illuminance measurements. MILL enables comprehensive evaluation of enhancement algorithms across variable lighting conditions. We benchmark several state-of-the-art methods and reveal significant performance variations across intensity levels. Leveraging the unique multi-illumination structure of our dataset, we propose improvements that enhance robustness across diverse illumination scenarios. Our modifications achieve up to 10 dB PSNR improvement for DSLR and 2 dB for the smartphone on Full HD images. 

**Abstract (ZH)**: 多光照条件低光环境成像数据集（MILL）及其在低光增强中的应用 

---
# RS-CA-HSICT: A Residual and Spatial Channel Augmented CNN Transformer Framework for Monkeypox Detection 

**Title (ZH)**: RS-CA-HSICT: 一种残差和空间通道增强的CNN变换器框架用于猴痘检测 

**Authors**: Rashid Iqbal, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2511.15476)  

**Abstract**: This work proposes a hybrid deep learning approach, namely Residual and Spatial Learning based Channel Augmented Integrated CNN-Transformer architecture, that leverages the strengths of CNN and Transformer towards enhanced MPox detection. The proposed RS-CA-HSICT framework is composed of an HSICT block, a residual CNN module, a spatial CNN block, and a CA, which enhances the diverse feature space, detailed lesion information, and long-range dependencies. The new HSICT module first integrates an abstract representation of the stem CNN and customized ICT blocks for efficient multihead attention and structured CNN layers with homogeneous (H) and structural (S) operations. The customized ICT blocks learn global contextual interactions and local texture extraction. Additionally, H and S layers learn spatial homogeneity and fine structural details by reducing noise and modeling complex morphological variations. Moreover, inverse residual learning enhances vanishing gradient, and stage-wise resolution reduction ensures scale invariance. Furthermore, the RS-CA-HSICT framework augments the learned HSICT channels with the TL-driven Residual and Spatial CNN maps for enhanced multiscale feature space capturing global and localized structural cues, subtle texture, and contrast variations. These channels, preceding augmentation, are refined through the Channel-Fusion-and-Attention block, which preserves discriminative channels while suppressing redundant ones, thereby enabling efficient computation. Finally, the spatial attention mechanism refines pixel selection to detect subtle patterns and intra-class contrast variations in Mpox. Experimental results on both the Kaggle benchmark and a diverse MPox dataset reported classification accuracy as high as 98.30% and an F1-score of 98.13%, which outperforms the existing CNNs and ViTs. 

**Abstract (ZH)**: 基于残差和空间学习的通道增强集成CNN-Transformer架构：Residual and Spatial Learning Based Channel Augmented Integrated CNN-Transformer Architecture for Enhanced MPox检测 

---
# IPTQ-ViT: Post-Training Quantization of Non-linear Functions for Integer-only Vision Transformers 

**Title (ZH)**: IPTQ-ViT：仅整数视觉变换器的后训练量化非线性函数 

**Authors**: Gihwan Kim, Jemin Lee, Hyungshin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.15369)  

**Abstract**: Previous Quantization-Aware Training (QAT) methods for vision transformers rely on expensive retraining to recover accuracy loss in non-linear layer quantization, limiting their use in resource-constrained environments. In contrast, existing Post-Training Quantization (PTQ) methods either partially quantize non-linear functions or adjust activation distributions to maintain accuracy but fail to achieve fully integer-only inference. In this paper, we introduce IPTQ-ViT, a novel PTQ framework for fully integer-only vision transformers without retraining. We present approximation functions: a polynomial-based GELU optimized for vision data and a bit-shifting-based Softmax designed to improve approximation accuracy in PTQ. In addition, we propose a unified metric integrating quantization sensitivity, perturbation, and computational cost to select the optimal approximation function per activation layer. IPTQ-ViT outperforms previous PTQ methods, achieving up to 6.44\%p (avg. 1.78\%p) top-1 accuracy improvement for image classification, 1.0 mAP for object detection. IPTQ-ViT outperforms partial floating-point PTQ methods under W8A8 and W4A8, and achieves accuracy and latency comparable to integer-only QAT methods. We plan to release our code this https URL. 

**Abstract (ZH)**: IPTQ-ViT：无需重新训练的全整数后训练量化视觉变压器 

---
# Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks 

**Title (ZH)**: 基于视频的推理：首次通过迷宫求解任务评估视频模型的推理能力 

**Authors**: Cheng Yang, Haiyuan Wan, Yiran Peng, Xin Cheng, Zhaoyang Yu, Jiayi Zhang, Junchi Yu, Xinlei Yu, Xiawu Zheng, Dongzhan Zhou, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15065)  

**Abstract**: Video Models have achieved remarkable success in high-fidelity video generation with coherent motion dynamics. Analogous to the development from text generation to text-based reasoning in language modeling, the development of video models motivates us to ask: Can video models reason via video generation? Compared with the discrete text corpus, video grounds reasoning in explicit spatial layouts and temporal continuity, which serves as an ideal substrate for spatial reasoning. In this work, we explore the reasoning via video paradigm and introduce VR-Bench -- a comprehensive benchmark designed to systematically evaluate video models' reasoning capabilities. Grounded in maze-solving tasks that inherently require spatial planning and multi-step reasoning, VR-Bench contains 7,920 procedurally generated videos across five maze types and diverse visual styles. Our empirical analysis demonstrates that SFT can efficiently elicit the reasoning ability of video model. Video models exhibit stronger spatial perception during reasoning, outperforming leading VLMs and generalizing well across diverse scenarios, tasks, and levels of complexity. We further discover a test-time scaling effect, where diverse sampling during inference improves reasoning reliability by 10--20%. These findings highlight the unique potential and scalability of reasoning via video for spatial reasoning tasks. 

**Abstract (ZH)**: 视频模型在一致运动动态下实现了高保真视频生成的显著成功。类似于语言模型从文本生成发展到基于文本的推理，视频模型的发展促使我们提出一个问题：视频模型能否通过视频生成来进行推理？与离散的文本语料库相比，视频立足于明确的空间布局和时间连续性，这为空间推理提供了一个理想的基底。在本工作中，我们探索基于视频的推理范式，并介绍VR-Bench——一个全面的基准，旨在系统性地评估视频模型的推理能力。基于固有要求空间规划和多步推理的迷宫求解任务，VR-Bench包含7,920个 procedurally生成的视频，涵盖了五种迷宫类型和多样的视觉风格。我们的实证分析表明，SFT能够有效地激发视频模型的推理能力。视频模型在推理过程中表现出更强的空间感知能力，优于最先进的多模态视觉语言模型，并且能够很好地泛化到各种场景、任务和复杂度级别。我们还发现了一种推理时的扩展效应，即推断过程中多样性的采样可以提高推理可靠性10-20%。这些发现突显了基于视频进行空间推理的独特潜力和扩展性。 

---
# Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation 

**Title (ZH)**: 康定斯基5.0：图像和视频生成的foundation模型家族 

**Authors**: Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko, Viacheslav Vasilev, Alexey Letunovskiy, Maria Kovaleva, Nikolai Vaulin, Ivan Kirillov, Lev Novitskiy, Denis Koposov, Nikita Kiselev, Alexander Varlamov, Dmitrii Mikhailov, Vladimir Polovnikov, Andrey Shutkin, Ilya Vasiliev, Julia Agafonova, Anastasiia Kargapoltseva, Anna Dmitrienko, Anastasia Maltseva, Anna Averchenkova, Olga Kim, Tatiana Nikulina, Denis Dimitrov  

**Link**: [PDF](https://arxiv.org/pdf/2511.14993)  

**Abstract**: This report introduces Kandinsky 5.0, a family of state-of-the-art foundation models for high-resolution image and 10-second video synthesis. The framework comprises three core line-up of models: Kandinsky 5.0 Image Lite - a line-up of 6B parameter image generation models, Kandinsky 5.0 Video Lite - a fast and lightweight 2B parameter text-to-video and image-to-video models, and Kandinsky 5.0 Video Pro - 19B parameter models that achieves superior video generation quality. We provide a comprehensive review of the data curation lifecycle - including collection, processing, filtering and clustering - for the multi-stage training pipeline that involves extensive pre-training and incorporates quality-enhancement techniques such as self-supervised fine-tuning (SFT) and reinforcement learning (RL)-based post-training. We also present novel architectural, training, and inference optimizations that enable Kandinsky 5.0 to achieve high generation speeds and state-of-the-art performance across various tasks, as demonstrated by human evaluation. As a large-scale, publicly available generative framework, Kandinsky 5.0 leverages the full potential of its pre-training and subsequent stages to be adapted for a wide range of generative applications. We hope that this report, together with the release of our open-source code and training checkpoints, will substantially advance the development and accessibility of high-quality generative models for the research community. 

**Abstract (ZH)**: Kandinsky 5.0：一种高分辨率图像和10秒视频合成的先进基础模型家族 

---
# SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification 

**Title (ZH)**: SVBRD-LLM: 自验证行为规则发现的自主车辆识别 

**Authors**: Xiangyu Li, Zhaomiao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.14977)  

**Abstract**: As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence. 

**Abstract (ZH)**: 随着更多的自动驾驶车辆在公共道路上行驶，了解自动驾驶车辆的实际行为对于分析交通安全、制定政策和公众接受度至关重要。本文提出了一种SVBRD-LLM框架，该框架通过零样本提示工程自动发现、验证和应用来自真实交通视频的可解释行为规则。该框架使用YOLOv8和ByteTrack提取车辆轨迹，计算运动特征，并利用GPT-5零样本提示将自动驾驶车辆与人类驾驶车辆进行比较，生成35个结构化的行为规则假设。这些规则在验证集上进行测试，并根据失败案例迭代 refinement 进行筛选，最终编译成高置信度规则库。该框架在独立测试集上对速度变化预测、车道变更预测和自动驾驶车辆识别任务进行了评估。在超过1500小时的真实交通视频实验中，框架在自动驾驶车辆识别任务上的准确率达到90.0%，F1分数达到93.3%。发现的规则清晰揭示了自动驾驶车辆在速度控制平滑性、车道变更保守性和加速度稳定性方面的独特特征，每条规则均附有语义描述、适用场景和验证置信度。 

---
# EGSA-PT:Edge-Guided Spatial Attention with Progressive Training for Monocular Depth Estimation and Segmentation of Transparent Objects 

**Title (ZH)**: 基于边缘指导的空间注意力与渐进训练的单目透明物体深度估算与分割方法：EGSA-PT 

**Authors**: Gbenga Omotara, Ramy Farag, Seyed Mohamad Ali Tousi, G.N. DeSouza  

**Link**: [PDF](https://arxiv.org/pdf/2511.14970)  

**Abstract**: Transparent object perception remains a major challenge in computer vision research, as transparency confounds both depth estimation and semantic segmentation. Recent work has explored multi-task learning frameworks to improve robustness, yet negative cross-task interactions often hinder performance. In this work, we introduce Edge-Guided Spatial Attention (EGSA), a fusion mechanism designed to mitigate destructive interactions by incorporating boundary information into the fusion between semantic and geometric features. On both Syn-TODD and ClearPose benchmarks, EGSA consistently improved depth accuracy over the current state of the art method (MODEST), while preserving competitive segmentation performance, with the largest improvements appearing in transparent regions. Besides our fusion design, our second contribution is a multi-modal progressive training strategy, where learning transitions from edges derived from RGB images to edges derived from predicted depth images. This approach allows the system to bootstrap learning from the rich textures contained in RGB images, and then switch to more relevant geometric content in depth maps, while it eliminates the need for ground-truth depth at training time. Together, these contributions highlight edge-guided fusion as a robust approach capable of improving transparent object perception. 

**Abstract (ZH)**: 透明对象感知仍然是计算机视觉研究中的一个主要挑战，因为透明性会混淆深度估计和语义分割。最近的工作探索了多任务学习框架以提高鲁棒性，但跨任务的负交互作用往往阻碍性能。在本工作中，我们引入了边缘引导空间注意力（EGSA）机制，通过将边界信息融入语义和几何特征的融合中以减轻破坏性交互作用。在Syn-TODD和ClearPose基准上，EGSA在不牺牲竞争力的分割性能的情况下，一致地提高了深度精度，尤其是在透明区域，表现最为显著。除了我们的融合设计，我们的第二个贡献是一种多模态渐进训练策略，其中学习从RGB图像派生的边缘过渡到从预测深度图像派生的边缘。这种做法允许系统从RGB图像中丰富的纹理启动学习，然后切换到深度图中的更有 relevancy 的几何内容，从而在训练时消除了真正的深度数据的需求。这些贡献共同凸显了边缘引导融合作为一种稳健的方法，能够提升透明对象感知。 

---
# When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation 

**Title (ZH)**: 当CNN超越Transformer和眼镜蛇：重新审视深度架构在牙釉质龋分割中的应用 

**Authors**: Aashish Ghimire, Jun Zeng, Roshan Paudel, Nikhil Kumar Tomar, Deepak Ranjan Nayak, Harshith Reddy Nalla, Vivek Jha, Glenda Reynolds, Debesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2511.14860)  

**Abstract**: Accurate identification and segmentation of dental caries in panoramic radiographs are critical for early diagnosis and effective treatment planning. Automated segmentation remains challenging due to low lesion contrast, morphological variability, and limited annotated data. In this study, we present the first comprehensive benchmarking of convolutional neural networks, vision transformers and state-space mamba architectures for automated dental caries segmentation on panoramic radiographs through a DC1000 dataset. Twelve state-of-the-art architectures, including VMUnet, MambaUNet, VMUNetv2, RMAMamba-S, TransNetR, PVTFormer, DoubleU-Net, and ResUNet++, were trained under identical configurations. Results reveal that, contrary to the growing trend toward complex attention based architectures, the CNN-based DoubleU-Net achieved the highest dice coefficient of 0.7345, mIoU of 0.5978, and precision of 0.8145, outperforming all transformer and Mamba variants. In the study, the top 3 results across all performance metrics were achieved by CNN-based architectures. Here, Mamba and transformer-based methods, despite their theoretical advantage in global context modeling, underperformed due to limited data and weaker spatial priors. These findings underscore the importance of architecture-task alignment in domain-specific medical image segmentation more than model complexity. Our code is available at: this https URL. 

**Abstract (ZH)**: 准确识别和分割全景放射图像中的龋齿对于早期诊断和有效治疗规划至关重要。由于病变更量低、形态多变以及标注数据有限，自动化分割仍然具有挑战性。在本研究中，我们通过DC1000数据集首次全面比较了卷积神经网络、视觉变换器和状态空间Mamba架构在全景放射图像中自动化龋齿分割中的性能。十二种最先进的架构，包括VMUnet、MambaUNet、VMUNetv2、RMAMamba-S、TransNetR、PVTFormer、DoubleU-Net和ResUNet++，在相同配置下进行了训练。结果显示，与日益复杂的注意力机制架构趋势相反，基于CNN的DoubleU-Net实现了最高的Dice系数0.7345、mIoU 0.5978和精度0.8145，优于所有变压器和Mamba变体。在本研究中，所有性能指标中前3名结果均来自基于CNN的架构。Mamba和基于变压器的方法尽管在全局上下文建模方面存在理论优势，但由于数据有限和空间先验较弱，表现不佳。这些发现强调了在特定领域医学图像分割中架构与任务匹配的重要性，而不仅仅是模型复杂度。我们的代码可在以下链接获取：this https URL。 

---
# Fully Differentiable dMRI Streamline Propagation in PyTorch 

**Title (ZH)**: 在PyTorch中实现完全可微的dMRI纤维束传播 

**Authors**: Jongyeon Yoon, Elyssa M. McMaster, Michael E. Kim, Gaurav Rudravaram, Kurt G. Schilling, Bennett A. Landman, Daniel Moyer  

**Link**: [PDF](https://arxiv.org/pdf/2511.14807)  

**Abstract**: Diffusion MRI (dMRI) provides a distinctive means to probe the microstructural architecture of living tissue, facilitating applications such as brain connectivity analysis, modeling across multiple conditions, and the estimation of macrostructural features. Tractography, which emerged in the final years of the 20th century and accelerated in the early 21st century, is a technique for visualizing white matter pathways in the brain using dMRI. Most diffusion tractography methods rely on procedural streamline propagators or global energy minimization methods. Although recent advancements in deep learning have enabled tasks that were previously challenging, existing tractography approaches are often non-differentiable, limiting their integration in end-to-end learning frameworks. While progress has been made in representing streamlines in differentiable frameworks, no existing method offers fully differentiable propagation. In this work, we propose a fully differentiable solution that retains numerical fidelity with a leading streamline algorithm. The key is that our PyTorch-engineered streamline propagator has no components that block gradient flow, making it fully differentiable. We show that our method matches standard propagators while remaining differentiable. By translating streamline propagation into a differentiable PyTorch framework, we enable deeper integration of tractography into deep learning workflows, laying the foundation for a new category of macrostructural reasoning that is not only computationally robust but also scientifically rigorous. 

**Abstract (ZH)**: 弥散磁共振成像（dMRI）提供了一种独特的手段来探查活体组织的微观结构架构，促进脑连接分析、多条件建模以及宏观结构特征估计的应用。轨迹追踪技术起源于20世纪末，并在21世纪初加速发展，是一种利用dMRI可视化大脑白质路径的技术。大多数弥散轨迹追踪方法依赖于过程化的流线传播器或全局能量最小化方法。尽管最近深度学习的进展使得一些以前难以完成的任务变得可行，但现有轨迹追踪方法通常是非可微的，限制了其在端到端学习框架中的集成。虽然有进展表示流线可以在不同的可微框架中表示，但目前没有方法能提供完全可微的传播。在这项工作中，我们提出了一种完全可微的解决方案，该方案在保留领先流线算法数值精度的同时，确保了完全可微性。关键在于我们利用PyTorch设计的流线传播器没有阻碍梯度流动的组件，从而实现了完全可微性。我们证明了我们的方法在保持标准传播器性能的同时，仍然可以进行微分。通过将流线传播转化为可微PyTorch框架，我们使轨迹追踪更深地集成到深度学习工作流中，为不仅计算上稳健而且在科学上严格的全新宏观结构推理类别奠定了基础。 

---
# Application of Graph Based Vision Transformers Architectures for Accurate Temperature Prediction in Fiber Specklegram Sensors 

**Title (ZH)**: 基于图卷积视觉变换器架构在光纤斑纹传感器中实现准确的温度预测 

**Authors**: Abhishek Sebastian  

**Link**: [PDF](https://arxiv.org/pdf/2511.14792)  

**Abstract**: Fiber Specklegram Sensors (FSS) are highly effective for environmental monitoring, particularly for detecting temperature variations. However, the nonlinear nature of specklegram data presents significant challenges for accurate temperature prediction. This study investigates the use of transformer-based architectures, including Vision Transformers (ViTs), Swin Transformers, and emerging models such as Learnable Importance Non-Symmetric Attention Vision Transformers (LINA-ViT) and Multi-Adaptive Proximity Vision Graph Attention Transformers (MAP-ViGAT), to predict temperature from specklegram data over a range of 0 to 120 Celsius. The results show that ViTs achieved a Mean Absolute Error (MAE) of 1.15, outperforming traditional models such as CNNs. GAT-ViT and MAP-ViGAT variants also demonstrated competitive accuracy, highlighting the importance of adaptive attention mechanisms and graph-based structures in capturing complex modal interactions and phase shifts in specklegram data. Additionally, this study incorporates Explainable AI (XAI) techniques, including attention maps and saliency maps, to provide insights into the decision-making processes of the transformer models, improving interpretability and transparency. These findings establish transformer architectures as strong benchmarks for optical fiber-based temperature sensing and offer promising directions for industrial monitoring and structural health assessment applications. 

**Abstract (ZH)**: 基于光纤斑纹图传感器的变压器架构在温度监测中的应用：解释性AI技术的集成研究 

---
# ESA: Energy-Based Shot Assembly Optimization for Automatic Video Editing 

**Title (ZH)**: ESA：基于能量的镜头组装优化自动视频编辑 

**Authors**: Yaosen Chen, Wei Wang, Tianheng Zheng, Xuming Wen, Han Yang, Yanru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02505)  

**Abstract**: Shot assembly is a crucial step in film production and video editing, involving the sequencing and arrangement of shots to construct a narrative, convey information, or evoke emotions. Traditionally, this process has been manually executed by experienced editors. While current intelligent video editing technologies can handle some automated video editing tasks, they often fail to capture the creator's unique artistic expression in shot assembly. To address this challenge, we propose an energy-based optimization method for video shot assembly. Specifically, we first perform visual-semantic matching between the script generated by a large language model and a video library to obtain subsets of candidate shots aligned with the script semantics. Next, we segment and label the shots from reference videos, extracting attributes such as shot size, camera motion, and semantics. We then employ energy-based models to learn from these attributes, scoring candidate shot sequences based on their alignment with reference styles. Finally, we achieve shot assembly optimization by combining multiple syntax rules, producing videos that align with the assembly style of the reference videos. Our method not only automates the arrangement and combination of independent shots according to specific logic, narrative requirements, or artistic styles but also learns the assembly style of reference videos, creating a coherent visual sequence or holistic visual expression. With our system, even users with no prior video editing experience can create visually compelling videos. Project page: this https URL 

**Abstract (ZH)**: 帧组装是电影制作和视频编辑中的一个关键步骤，涉及将帧按顺序排列以构建叙述、传达信息或唤起情感。传统上，这一过程由经验丰富的编辑手动执行。尽管当前的智能视频编辑技术可以处理一些自动视频编辑任务，但它们往往无法捕捉创作者在帧组装中的独特艺术表达。为了解决这一挑战，我们提出了一种基于能量的优化方法用于视频帧组装。具体来说，我们首先通过大型语言模型生成的剧本与视频库进行视觉语义匹配，以获取与剧本语义相匹配的候选镜头集。然后，我们对参考视频进行分段和标注，提取诸如镜头大小、摄像机运动和语义等属性。接着，我们利用基于能量的模型从这些属性中学习，并根据参考样式的匹配度对候选镜头序列进行评分。最后，我们通过结合多种语法规则实现镜头组装优化，生成与参考视频组装风格一致的视频。我们的方法不仅能够根据特定逻辑、叙述要求或艺术风格自动化独立镜头的排列与组合，还能学习参考视频的组装风格，从而创建一个连贯的视觉序列或整体视觉表现。借助我们的系统，即使是没有任何视频编辑经验的用户也能制作出视觉上引人注目的视频。项目页面：this https URL。 

---
