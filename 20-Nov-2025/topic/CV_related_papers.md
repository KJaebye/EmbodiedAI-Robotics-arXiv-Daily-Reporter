# Walrus: A Cross-Domain Foundation Model for Continuum Dynamics 

**Title (ZH)**: Walrus: 一种用于连续动力学的跨域基础模型 

**Authors**: Michael McCabe, Payel Mukhopadhyay, Tanya Marwah, Bruno Regaldo-Saint Blancard, Francois Rozet, Cristiana Diaconu, Lucas Meyer, Kaze W. K. Wong, Hadi Sotoudeh, Alberto Bietti, Irina Espejo, Rio Fear, Siavash Golkar, Tom Hehir, Keiya Hirashima, Geraud Krawezik, Francois Lanusse, Rudy Morel, Ruben Ohana, Liam Parker, Mariel Pettee, Jeff Shen, Kyunghyun Cho, Miles Cranmer, Shirley Ho  

**Link**: [PDF](https://arxiv.org/pdf/2511.15684)  

**Abstract**: Foundation models have transformed machine learning for language and vision, but achieving comparable impact in physical simulation remains a challenge. Data heterogeneity and unstable long-term dynamics inhibit learning from sufficiently diverse dynamics, while varying resolutions and dimensionalities challenge efficient training on modern hardware. Through empirical and theoretical analysis, we incorporate new approaches to mitigate these obstacles, including a harmonic-analysis-based stabilization method, load-balanced distributed 2D and 3D training strategies, and compute-adaptive tokenization. Using these tools, we develop Walrus, a transformer-based foundation model developed primarily for fluid-like continuum dynamics. Walrus is pretrained on nineteen diverse scenarios spanning astrophysics, geoscience, rheology, plasma physics, acoustics, and classical fluids. Experiments show that Walrus outperforms prior foundation models on both short and long term prediction horizons on downstream tasks and across the breadth of pretraining data, while ablation studies confirm the value of our contributions to forecast stability, training throughput, and transfer performance over conventional approaches. Code and weights are released for community use. 

**Abstract (ZH)**: 基础模型已转型语言和视觉领域的机器学习，但在物理模拟中的应用仍面临挑战。数据异质性和不稳定的长期动力学阻碍了对足够多样动力学的学习，而不同的分辨率和维度性给现代硬件上的高效训练带来了挑战。通过实证和理论分析，我们引入了新的方法来缓解这些障碍，包括基于谐波分析的稳定化方法、负载均衡的分布式2D和3D训练策略以及计算自适应的分词方法。利用这些工具，我们开发了Walrus，一种主要用于流体-like连续动力学的变压器基础模型。Walrus在天体物理学、地球科学、流变学、等离子体物理、声学和经典流体等十九个不同场景下进行预训练。实验表明，Walrus在短时间和长时间预测窗口以及预训练数据跨度上均优于先前的基础模型，且消融研究表明我们的贡献对预测稳定性、训练吞吐量和迁移性能具有重要价值。社区可以获取代码和权重。 

---
# GEO-Bench-2: From Performance to Capability, Rethinking Evaluation in Geospatial AI 

**Title (ZH)**: GEO-Bench-2: 从性能到能力，重新思考地理空间AI的评估 

**Authors**: Naomi Simumba, Nils Lehmann, Paolo Fraccaro, Hamed Alemohammad, Geeth De Mel, Salman Khan, Manil Maskey, Nicolas Longepe, Xiao Xiang Zhu, Hannah Kerner, Juan Bernabe-Moreno, Alexander Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2511.15658)  

**Abstract**: Geospatial Foundation Models (GeoFMs) are transforming Earth Observation (EO), but evaluation lacks standardized protocols. GEO-Bench-2 addresses this with a comprehensive framework spanning classification, segmentation, regression, object detection, and instance segmentation across 19 permissively-licensed datasets. We introduce ''capability'' groups to rank models on datasets that share common characteristics (e.g., resolution, bands, temporality). This enables users to identify which models excel in each capability and determine which areas need improvement in future work. To support both fair comparison and methodological innovation, we define a prescriptive yet flexible evaluation protocol. This not only ensures consistency in benchmarking but also facilitates research into model adaptation strategies, a key and open challenge in advancing GeoFMs for downstream tasks.
Our experiments show that no single model dominates across all tasks, confirming the specificity of the choices made during architecture design and pretraining. While models pretrained on natural images (ConvNext ImageNet, DINO V3) excel on high-resolution tasks, EO-specific models (TerraMind, Prithvi, and Clay) outperform them on multispectral applications such as agriculture and disaster response. These findings demonstrate that optimal model choice depends on task requirements, data modalities, and constraints. This shows that the goal of a single GeoFM model that performs well across all tasks remains open for future research. GEO-Bench-2 enables informed, reproducible GeoFM evaluation tailored to specific use cases. Code, data, and leaderboard for GEO-Bench-2 are publicly released under a permissive license. 

**Abstract (ZH)**: GeoFMs的地理空间基础模型正在_transforming地球观测(EO)，但评估缺乏标准化协议。GEO-Bench-2通过涵盖分类、分割、回归、对象检测和实例分割的全面框架，跨越了19个许可使用的数据集，解决了这一问题。我们引入“能力”组，按共享共同特征（例如，分辨率、波段、时间性）的数据集对模型进行排名。这使用户能够确定哪些模型在每个能力方面表现最佳，并确定未来工作需要改进的领域。为了支持公平比较和方法创新，我们定义了一种规范但灵活的评估协议。这不仅确保了基准测试的一致性，还促进了对模型适应策略的研究，这是推进GeoFMs用于下游任务的一个重要且开放的挑战。 

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
# Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks 

**Title (ZH)**: 基于视频的推理：首次通过迷宫求解任务评估视频模型的推理能力 

**Authors**: Cheng Yang, Haiyuan Wan, Yiran Peng, Xin Cheng, Zhaoyang Yu, Jiayi Zhang, Junchi Yu, Xinlei Yu, Xiawu Zheng, Dongzhan Zhou, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15065)  

**Abstract**: Video Models have achieved remarkable success in high-fidelity video generation with coherent motion dynamics. Analogous to the development from text generation to text-based reasoning in language modeling, the development of video models motivates us to ask: Can video models reason via video generation? Compared with the discrete text corpus, video grounds reasoning in explicit spatial layouts and temporal continuity, which serves as an ideal substrate for spatial reasoning. In this work, we explore the reasoning via video paradigm and introduce VR-Bench -- a comprehensive benchmark designed to systematically evaluate video models' reasoning capabilities. Grounded in maze-solving tasks that inherently require spatial planning and multi-step reasoning, VR-Bench contains 7,920 procedurally generated videos across five maze types and diverse visual styles. Our empirical analysis demonstrates that SFT can efficiently elicit the reasoning ability of video model. Video models exhibit stronger spatial perception during reasoning, outperforming leading VLMs and generalizing well across diverse scenarios, tasks, and levels of complexity. We further discover a test-time scaling effect, where diverse sampling during inference improves reasoning reliability by 10--20%. These findings highlight the unique potential and scalability of reasoning via video for spatial reasoning tasks. 

**Abstract (ZH)**: 视频模型在一致运动动态下实现了高保真视频生成的显著成功。类似于语言模型从文本生成发展到基于文本的推理，视频模型的发展促使我们提出一个问题：视频模型能否通过视频生成来进行推理？与离散的文本语料库相比，视频立足于明确的空间布局和时间连续性，这为空间推理提供了一个理想的基底。在本工作中，我们探索基于视频的推理范式，并介绍VR-Bench——一个全面的基准，旨在系统性地评估视频模型的推理能力。基于固有要求空间规划和多步推理的迷宫求解任务，VR-Bench包含7,920个 procedurally生成的视频，涵盖了五种迷宫类型和多样的视觉风格。我们的实证分析表明，SFT能够有效地激发视频模型的推理能力。视频模型在推理过程中表现出更强的空间感知能力，优于最先进的多模态视觉语言模型，并且能够很好地泛化到各种场景、任务和复杂度级别。我们还发现了一种推理时的扩展效应，即推断过程中多样性的采样可以提高推理可靠性10-20%。这些发现突显了基于视频进行空间推理的独特潜力和扩展性。 

---
# UniHOI: Unified Human-Object Interaction Understanding via Unified Token Space 

**Title (ZH)**: UniHOI: 统一人类-对象交互理解 via 统一令牌空间 

**Authors**: Panqi Yang, Haodong Jing, Nanning Zheng, Yongqiang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.15046)  

**Abstract**: In the field of human-object interaction (HOI), detection and generation are two dual tasks that have traditionally been addressed separately, hindering the development of comprehensive interaction understanding. To address this, we propose UniHOI, which jointly models HOI detection and generation via a unified token space, thereby effectively promoting knowledge sharing and enhancing generalization. Specifically, we introduce a symmetric interaction-aware attention module and a unified semi-supervised learning paradigm, enabling effective bidirectional mapping between images and interaction semantics even under limited annotations. Extensive experiments demonstrate that UniHOI achieves state-of-the-art performance in both HOI detection and generation. Specifically, UniHOI improves accuracy by 4.9% on long-tailed HOI detection and boosts interaction metrics by 42.0% on open-vocabulary generation tasks. 

**Abstract (ZH)**: 人类对象交互领域的检测与生成：UniHOI及其在知识共享和泛化中的联合建模 

---
# Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation 

**Title (ZH)**: 康定斯基5.0：图像和视频生成的foundation模型家族 

**Authors**: Vladimir Arkhipkin, Vladimir Korviakov, Nikolai Gerasimenko, Denis Parkhomenko, Viacheslav Vasilev, Alexey Letunovskiy, Maria Kovaleva, Nikolai Vaulin, Ivan Kirillov, Lev Novitskiy, Denis Koposov, Nikita Kiselev, Alexander Varlamov, Dmitrii Mikhailov, Vladimir Polovnikov, Andrey Shutkin, Ilya Vasiliev, Julia Agafonova, Anastasiia Kargapoltseva, Anna Dmitrienko, Anastasia Maltseva, Anna Averchenkova, Olga Kim, Tatiana Nikulina, Denis Dimitrov  

**Link**: [PDF](https://arxiv.org/pdf/2511.14993)  

**Abstract**: This report introduces Kandinsky 5.0, a family of state-of-the-art foundation models for high-resolution image and 10-second video synthesis. The framework comprises three core line-up of models: Kandinsky 5.0 Image Lite - a line-up of 6B parameter image generation models, Kandinsky 5.0 Video Lite - a fast and lightweight 2B parameter text-to-video and image-to-video models, and Kandinsky 5.0 Video Pro - 19B parameter models that achieves superior video generation quality. We provide a comprehensive review of the data curation lifecycle - including collection, processing, filtering and clustering - for the multi-stage training pipeline that involves extensive pre-training and incorporates quality-enhancement techniques such as self-supervised fine-tuning (SFT) and reinforcement learning (RL)-based post-training. We also present novel architectural, training, and inference optimizations that enable Kandinsky 5.0 to achieve high generation speeds and state-of-the-art performance across various tasks, as demonstrated by human evaluation. As a large-scale, publicly available generative framework, Kandinsky 5.0 leverages the full potential of its pre-training and subsequent stages to be adapted for a wide range of generative applications. We hope that this report, together with the release of our open-source code and training checkpoints, will substantially advance the development and accessibility of high-quality generative models for the research community. 

**Abstract (ZH)**: Kandinsky 5.0：一种高分辨率图像和10秒视频合成的先进基础模型家族 

---
# EGSA-PT:Edge-Guided Spatial Attention with Progressive Training for Monocular Depth Estimation and Segmentation of Transparent Objects 

**Title (ZH)**: 基于边缘指导的空间注意力与渐进训练的单目透明物体深度估算与分割方法：EGSA-PT 

**Authors**: Gbenga Omotara, Ramy Farag, Seyed Mohamad Ali Tousi, G.N. DeSouza  

**Link**: [PDF](https://arxiv.org/pdf/2511.14970)  

**Abstract**: Transparent object perception remains a major challenge in computer vision research, as transparency confounds both depth estimation and semantic segmentation. Recent work has explored multi-task learning frameworks to improve robustness, yet negative cross-task interactions often hinder performance. In this work, we introduce Edge-Guided Spatial Attention (EGSA), a fusion mechanism designed to mitigate destructive interactions by incorporating boundary information into the fusion between semantic and geometric features. On both Syn-TODD and ClearPose benchmarks, EGSA consistently improved depth accuracy over the current state of the art method (MODEST), while preserving competitive segmentation performance, with the largest improvements appearing in transparent regions. Besides our fusion design, our second contribution is a multi-modal progressive training strategy, where learning transitions from edges derived from RGB images to edges derived from predicted depth images. This approach allows the system to bootstrap learning from the rich textures contained in RGB images, and then switch to more relevant geometric content in depth maps, while it eliminates the need for ground-truth depth at training time. Together, these contributions highlight edge-guided fusion as a robust approach capable of improving transparent object perception. 

**Abstract (ZH)**: 透明对象感知仍然是计算机视觉研究中的一个主要挑战，因为透明性会混淆深度估计和语义分割。最近的工作探索了多任务学习框架以提高鲁棒性，但跨任务的负交互作用往往阻碍性能。在本工作中，我们引入了边缘引导空间注意力（EGSA）机制，通过将边界信息融入语义和几何特征的融合中以减轻破坏性交互作用。在Syn-TODD和ClearPose基准上，EGSA在不牺牲竞争力的分割性能的情况下，一致地提高了深度精度，尤其是在透明区域，表现最为显著。除了我们的融合设计，我们的第二个贡献是一种多模态渐进训练策略，其中学习从RGB图像派生的边缘过渡到从预测深度图像派生的边缘。这种做法允许系统从RGB图像中丰富的纹理启动学习，然后切换到深度图中的更有 relevancy 的几何内容，从而在训练时消除了真正的深度数据的需求。这些贡献共同凸显了边缘引导融合作为一种稳健的方法，能够提升透明对象感知。 

---
# Skin-R1: Toward Trustworthy Clinical Reasoning for Dermatological Diagnosis 

**Title (ZH)**: Skin-R1: 朝着皮肤科诊断可信任临床推理的探索 

**Authors**: Zehao Liu, Wejieying Ren, Jipeng Zhang, Tianxiang Zhao, Jingxi Zhu, Xiaoting Li, Vasant G. Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2511.14900)  

**Abstract**: The emergence of vision-language models (VLMs) has opened new possibilities for clinical reasoning and has shown promising performance in dermatological diagnosis. However, their trustworthiness and clinical utility are often limited by three major factors: (1) Data heterogeneity, where diverse datasets lack consistent diagnostic labels and clinical concept annotations; (2) Absence of grounded diagnostic rationales, leading to a scarcity of reliable reasoning supervision; and (3) Limited scalability and generalization, as models trained on small, densely annotated datasets struggle to transfer nuanced reasoning to large, sparsely-annotated ones.
To address these limitations, we propose SkinR1, a novel dermatological VLM that combines deep, textbook-based reasoning with the broad generalization capabilities of reinforcement learning (RL). SkinR1 systematically resolves the key challenges through a unified, end-to-end framework. First, we design a textbook-based reasoning generator that synthesizes high-fidelity, hierarchy-aware, and differential-diagnosis (DDx)-informed trajectories, providing reliable expert-level supervision. Second, we leverage the constructed trajectories for supervised fine-tuning (SFT) empowering the model with grounded reasoning ability. Third, we develop a novel RL paradigm that, by incorporating the hierarchical structure of diseases, effectively transfers these grounded reasoning patterns to large-scale, sparse data. Extensive experiments on multiple dermatology datasets demonstrate that SkinR1 achieves superior diagnostic accuracy. The ablation study demonstrates the importance of the reasoning foundation instilled by SFT. 

**Abstract (ZH)**: 视知觉语言模型（VLMs）的出现为临床推理带来了新的可能性，并在皮肤科诊断中表现出令人鼓舞的性能。然而，它们的信任度和临床实用性往往受限于三大因素：（1）数据异质性，不同数据集缺乏一致的诊断标签和临床概念标注；（2）缺乏基于事实的诊断推理，导致可靠的推理监督稀缺；（3）有限的可扩展性和泛化能力，训练于小规模、密集标注数据集的模型难以将细微的推理能力转移到大规模、稀疏标注的数据集中。

为了解决这些局限性，我们提出了SkinR1，一种新型的皮肤科VLM，它结合了基于深部教科书的推理与强化学习（RL）的强大泛化能力。SkinR1通过统一的端到端框架系统地解决了关键挑战。首先，我们设计了一种基于教科书的推理生成器，生成高保真、有层次意识且基于鉴别诊断的轨迹，提供可靠的专家级监督。其次，我们利用构建的轨迹进行监督微调（SFT），赋予模型基于事实的推理能力。第三，我们开发了一种新的RL范式，通过引入疾病的层次结构，有效地将这些基于事实的推理模式转移到大规模、稀疏数据中。在多个皮肤科数据集上的广泛实验表明，SkinR1实现了卓越的诊断准确性。消融研究表明，SFT灌输的推理基础的重要性。 

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
# TacEleven: generative tactic discovery for football open play 

**Title (ZH)**: TacEleven: 生成性战术发现用于足球开放进攻 

**Authors**: Siyao Zhao, Hao Ma, Zhiqiang Pu, Jingjing Huang, Yi Pan, Shijie Wang, Zhi Ming  

**Link**: [PDF](https://arxiv.org/pdf/2511.13326)  

**Abstract**: Creating offensive advantages during open play is fundamental to football success. However, due to the highly dynamic and long-sequence nature of open play, the potential tactic space grows exponentially as the sequence progresses, making automated tactic discovery extremely challenging. To address this, we propose TacEleven, a generative framework for football open-play tactic discovery developed in close collaboration with domain experts from AJ Auxerre, designed to assist coaches and analysts in tactical decision-making. TacEleven consists of two core components: a language-controlled tactical generator that produces diverse tactical proposals, and a multimodal large language model-based tactical critic that selects the optimal proposal aligned with a high-level stylistic tactical instruction. The two components enables rapid exploration of tactical proposals and discovery of alternative open-play offensive tactics. We evaluate TacEleven across three tasks with progressive tactical complexity: counterfactual exploration, single-step discovery, and multi-step discovery, through both quantitative metrics and a questionnaire-based qualitative assessment. The results show that the TacEleven-discovered tactics exhibit strong realism and tactical creativity, with 52.50% of the multi-step tactical alternatives rated adoptable in real-world elite football scenarios, highlighting the framework's ability to rapidly generate numerous high-quality tactics for complex long-sequence open-play situations. TacEleven demonstrates the potential of creatively leveraging domain data and generative models to advance tactical analysis in sports. 

**Abstract (ZH)**: 基于生成模型的足球开放play战术发现：TacEleven框架 

---
# ESA: Energy-Based Shot Assembly Optimization for Automatic Video Editing 

**Title (ZH)**: ESA：基于能量的镜头组装优化自动视频编辑 

**Authors**: Yaosen Chen, Wei Wang, Tianheng Zheng, Xuming Wen, Han Yang, Yanru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02505)  

**Abstract**: Shot assembly is a crucial step in film production and video editing, involving the sequencing and arrangement of shots to construct a narrative, convey information, or evoke emotions. Traditionally, this process has been manually executed by experienced editors. While current intelligent video editing technologies can handle some automated video editing tasks, they often fail to capture the creator's unique artistic expression in shot assembly. To address this challenge, we propose an energy-based optimization method for video shot assembly. Specifically, we first perform visual-semantic matching between the script generated by a large language model and a video library to obtain subsets of candidate shots aligned with the script semantics. Next, we segment and label the shots from reference videos, extracting attributes such as shot size, camera motion, and semantics. We then employ energy-based models to learn from these attributes, scoring candidate shot sequences based on their alignment with reference styles. Finally, we achieve shot assembly optimization by combining multiple syntax rules, producing videos that align with the assembly style of the reference videos. Our method not only automates the arrangement and combination of independent shots according to specific logic, narrative requirements, or artistic styles but also learns the assembly style of reference videos, creating a coherent visual sequence or holistic visual expression. With our system, even users with no prior video editing experience can create visually compelling videos. Project page: this https URL 

**Abstract (ZH)**: 镜头装配是电影制作和视频编辑中的关键步骤，涉及镜头的排序和排列以构建叙事、传达信息或引起情感反应。传统上，这一过程由经验丰富的编辑手动执行。尽管当前的智能视频编辑技术可以处理一些自动视频编辑任务，但它们往往无法捕捉创作者在镜头装配中的独特艺术表达。为了解决这一挑战，我们提出了一种基于能量的优化方法，用于视频镜头装配。具体而言，我们首先在剧本生成器（大型语言模型）生成的剧本与视频库之间进行视觉语义匹配，以获取与剧本语义相匹配的候选镜头子集。接着，我们对参考视频中的镜头进行分割和标记，提取诸如镜头大小、摄像机运动和语义等属性。然后，我们使用基于能量的模型从这些属性中学习，根据参考风格对候选镜头序列进行评分。最后，通过结合多个语法规则实现镜头装配优化，生成与参考视频装配风格一致的视频。我们的方法不仅可以根据特定逻辑、叙事需求或艺术风格自动排列和组合独立镜头，还能学习参考视频的装配风格，创造一个连贯的视觉序列或整体的视觉表达。借助我们的系统，即使是没有任何视频编辑经验的用户也可以创作出令人视觉印象深刻的视频。项目页面：this https URL 

---
