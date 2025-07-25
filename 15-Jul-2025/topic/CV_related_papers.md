# Raci-Net: Ego-vehicle Odometry Estimation in Adverse Weather Conditions 

**Title (ZH)**: Raci-Net: 逆向车辆在恶劣天气条件下的里程估计 

**Authors**: Mohammadhossein Talebi, Pragyan Dahal, Davide Possenti, Stefano Arrigoni, Francesco Braghin  

**Link**: [PDF](https://arxiv.org/pdf/2507.10376)  

**Abstract**: Autonomous driving systems are highly dependent on sensors like cameras, LiDAR, and inertial measurement units (IMU) to perceive the environment and estimate their motion. Among these sensors, perception-based sensors are not protected from harsh weather and technical failures. Although existing methods show robustness against common technical issues like rotational misalignment and disconnection, they often degrade when faced with dynamic environmental factors like weather conditions. To address these problems, this research introduces a novel deep learning-based motion estimator that integrates visual, inertial, and millimeter-wave radar data, utilizing each sensor strengths to improve odometry estimation accuracy and reliability under adverse environmental conditions such as snow, rain, and varying light. The proposed model uses advanced sensor fusion techniques that dynamically adjust the contributions of each sensor based on the current environmental condition, with radar compensating for visual sensor limitations in poor visibility. This work explores recent advancements in radar-based odometry and highlights that radar robustness in different weather conditions makes it a valuable component for pose estimation systems, specifically when visual sensors are degraded. Experimental results, conducted on the Boreas dataset, showcase the robustness and effectiveness of the model in both clear and degraded environments. 

**Abstract (ZH)**: 自主驾驶系统高度依赖于摄像头、LiDAR和惯性测量单元(IMU)等传感器来感知环境和估计运动状态。这些传感器中的感知型传感器容易受到恶劣天气和技术故障的影响。尽管现有方法在应对常见的技术问题如旋转对齐错误和断连时展示了鲁棒性，但在面对动态环境因素如天气条件时往往表现不佳。为了解决这些问题，本研究引入了一种基于深度学习的新型运动估计器，该估计器整合了视觉、惯性和毫米波雷达数据，充分利用每种传感器的优势，在雪、雨和不同光照条件等恶劣环境下提高里程计估计的准确性和可靠性。所提出的模型采用先进的传感器融合技术，根据当前环境条件动态调整每种传感器的贡献，毫米波雷达补偿了在能见度低时视觉传感器的局限性。本研究探讨了基于雷达的里程计的最新进展，并指出在视觉传感器退化时，雷达在不同天气条件下的鲁棒性使其成为姿态估计系统中的重要组成部分。实验结果，在Boreas数据集上进行，展示了该模型在清晰和退化环境中的鲁棒性和有效性。 

---
# Online 3D Bin Packing with Fast Stability Validation and Stable Rearrangement Planning 

**Title (ZH)**: 在线3D货箱打包：快速稳定性验证和稳定重新排列规划 

**Authors**: Ziyan Gao, Lijun Wang, Yuntao Kong, Nak Young Chong  

**Link**: [PDF](https://arxiv.org/pdf/2507.09123)  

**Abstract**: The Online Bin Packing Problem (OBPP) is a sequential decision-making task in which each item must be placed immediately upon arrival, with no knowledge of future arrivals. Although recent deep-reinforcement-learning methods achieve superior volume utilization compared with classical heuristics, the learned policies cannot ensure the structural stability of the bin and lack mechanisms for safely reconfiguring the bin when a new item cannot be placed directly. In this work, we propose a novel framework that integrates packing policy with structural stability validation and heuristic planning to overcome these limitations. Specifically, we introduce the concept of Load Bearable Convex Polygon (LBCP), which provides a computationally efficient way to identify stable loading positions that guarantee no bin collapse. Additionally, we present Stable Rearrangement Planning (SRP), a module that rearranges existing items to accommodate new ones while maintaining overall stability. Extensive experiments on standard OBPP benchmarks demonstrate the efficiency and generalizability of our LBCP-based stability validation, as well as the superiority of SRP in finding the effort-saving rearrangement plans. Our method offers a robust and practical solution for automated packing in real-world industrial and logistics applications. 

**Abstract (ZH)**: 在线 bin 包装问题 (OBPP) 是一种顺序决策任务，每个项目到达时必须立即放置，不预先知道未来到达的项目。尽管最近的深度强化学习方法在体积利用率方面优于经典启发式方法，但学习到的策略无法保证 bin 的结构稳定性，缺少当新项目无法直接放置时安全重新配置 bin 的机制。在这项工作中，我们提出了一种新框架，将包装策略与结构稳定性验证和启发式规划相结合，以克服这些限制。具体而言，我们引入了负载可承载凸多边形 (LBCP) 的概念，这是一种计算高效的稳定装载位置识别方法，能确保没有 bin 塌陷。此外，我们提出了稳定重新排列规划 (SRP) 模块，在保持总体稳定性的同时重新排列现有项目以容纳新项目。在标准 OBPP 基准上的广泛实验表明，基于 LBCP 的稳定性验证的高效性和普适性，以及 SRP 在寻找节约努力的重新排列计划方面的优越性。我们的方法为实际工业和物流应用场景中的自动化包装提供了一种稳健且实用的解决方案。 

---
# End-to-End Generation of City-Scale Vectorized Maps by Crowdsourced Vehicles 

**Title (ZH)**: 基于众源车辆的端到端城市规模向量地图生成 

**Authors**: Zebang Feng, Miao Fan, Bao Liu, Shengtong Xu, Haoyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.08901)  

**Abstract**: High-precision vectorized maps are indispensable for autonomous driving, yet traditional LiDAR-based creation is costly and slow, while single-vehicle perception methods lack accuracy and robustness, particularly in adverse conditions. This paper introduces EGC-VMAP, an end-to-end framework that overcomes these limitations by generating accurate, city-scale vectorized maps through the aggregation of data from crowdsourced vehicles. Unlike prior approaches, EGC-VMAP directly fuses multi-vehicle, multi-temporal map elements perceived onboard vehicles using a novel Trip-Aware Transformer architecture within a unified learning process. Combined with hierarchical matching for efficient training and a multi-objective loss, our method significantly enhances map accuracy and structural robustness compared to single-vehicle baselines. Validated on a large-scale, multi-city real-world dataset, EGC-VMAP demonstrates superior performance, enabling a scalable, cost-effective solution for city-wide mapping with a reported 90\% reduction in manual annotation costs. 

**Abstract (ZH)**: 基于 crowdsourced 车辆数据的城市规模高精度向量地图生成框架 EGC-VMAP 

---
# OTAS: Open-vocabulary Token Alignment for Outdoor Segmentation 

**Title (ZH)**: OTAS: 开ocabulary Token 对齐用于室外语段划分 

**Authors**: Simon Schwaiger, Stefan Thalhammer, Wilfried Wöber, Gerald Steinbauer-Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2507.08851)  

**Abstract**: Understanding open-world semantics is critical for robotic planning and control, particularly in unstructured outdoor environments. Current vision-language mapping approaches rely on object-centric segmentation priors, which often fail outdoors due to semantic ambiguities and indistinct semantic class boundaries. We propose OTAS - an Open-vocabulary Token Alignment method for Outdoor Segmentation. OTAS overcomes the limitations of open-vocabulary segmentation models by extracting semantic structure directly from the output tokens of pretrained vision models. By clustering semantically similar structures across single and multiple views and grounding them in language, OTAS reconstructs a geometrically consistent feature field that supports open-vocabulary segmentation queries. Our method operates zero-shot, without scene-specific fine-tuning, and runs at up to ~17 fps. OTAS provides a minor IoU improvement over fine-tuned and open-vocabulary 2D segmentation methods on the Off-Road Freespace Detection dataset. Our model achieves up to a 151% IoU improvement over open-vocabulary mapping methods in 3D segmentation on TartanAir. Real-world reconstructions demonstrate OTAS' applicability to robotic applications. The code and ROS node will be made publicly available upon paper acceptance. 

**Abstract (ZH)**: 开放世界语义理解对于机器人规划与控制至关重要，特别是在非结构化户外环境中。当前的视觉-语言映射方法依赖于以物体为中心的分割先验，由于语义模糊和不明确的语义类别边界，在户外环境中往往失效。我们提出OTAS——一种户外分割的开放词汇项对齐方法。OTAS通过直接从预训练视觉模型的输出标记中提取语义结构，克服了开放词汇项分割模型的限制。通过在单视图和多视图中聚类语义相似结构并在语言中定位，OTAS重建了一个几何上一致的特征场，支持开放词汇项分割查询。该方法实现无监督且无需特定场景的微调，在每秒帧数（fps）达到约17帧。OTAS在Off-Road Freespace Detection数据集上的交错重叠度（IoU）上提供了微小的改进，优于微调和开放词汇项的2D分割方法。在TartanAir上的3D分割中，我们的模型实现了高达151%的IoU改进。实世界重构展示了OTAS在机器人应用中的适用性。论文接受后，代码和ROS节点将公开发布。 

---
# LifelongPR: Lifelong knowledge fusion for point cloud place recognition based on replay and prompt learning 

**Title (ZH)**: lifelongPR：基于重播和提示学习的点云场所识别终身知识融合 

**Authors**: Xianghong Zou, Jianping Li, Zhe Chen, Zhen Cao, Zhen Dong, Qiegen Liu, Bisheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10034)  

**Abstract**: Point cloud place recognition (PCPR) plays a crucial role in photogrammetry and robotics applications such as autonomous driving, intelligent transportation, and augmented reality. In real-world large-scale deployments of a positioning system, PCPR models must continuously acquire, update, and accumulate knowledge to adapt to diverse and dynamic environments, i.e., the ability known as continual learning (CL). However, existing PCPR models often suffer from catastrophic forgetting, leading to significant performance degradation in previously learned scenes when adapting to new environments or sensor types. This results in poor model scalability, increased maintenance costs, and system deployment difficulties, undermining the practicality of PCPR. To address these issues, we propose LifelongPR, a novel continual learning framework for PCPR, which effectively extracts and fuses knowledge from sequential point cloud data. First, to alleviate the knowledge loss, we propose a replay sample selection method that dynamically allocates sample sizes according to each dataset's information quantity and selects spatially diverse samples for maximal representativeness. Second, to handle domain shifts, we design a prompt learning-based CL framework with a lightweight prompt module and a two-stage training strategy, enabling domain-specific feature adaptation while minimizing forgetting. Comprehensive experiments on large-scale public and self-collected datasets are conducted to validate the effectiveness of the proposed method. Compared with state-of-the-art (SOTA) methods, our method achieves 6.50% improvement in mIR@1, 7.96% improvement in mR@1, and an 8.95% reduction in F. The code and pre-trained models are publicly available at this https URL. 

**Abstract (ZH)**: 点云场所识别的终身学习框架（LifelongPR）：一种有效的连续学习方法 

---
# SegVec3D: A Method for Vector Embedding of 3D Objects Oriented Towards Robot manipulation 

**Title (ZH)**: SegVec3D: 一种面向机器人操作的3D对象向量嵌入方法 

**Authors**: Zhihan Kang, Boyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09459)  

**Abstract**: We propose SegVec3D, a novel framework for 3D point cloud instance segmentation that integrates attention mechanisms, embedding learning, and cross-modal alignment. The approach builds a hierarchical feature extractor to enhance geometric structure modeling and enables unsupervised instance segmentation via contrastive clustering. It further aligns 3D data with natural language queries in a shared semantic space, supporting zero-shot retrieval. Compared to recent methods like Mask3D and ULIP, our method uniquely unifies instance segmentation and multimodal understanding with minimal supervision and practical deployability. 

**Abstract (ZH)**: SegVec3D：一种结合注意力机制、嵌入学习和跨模态对齐的3D点云实例分割新框架 

---
# Domain Adaptation and Multi-view Attention for Learnable Landmark Tracking with Sparse Data 

**Title (ZH)**: 基于稀疏数据的领域适应与多视图注意力可学习地标跟踪 

**Authors**: Timothy Chase Jr, Karthik Dantu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09420)  

**Abstract**: The detection and tracking of celestial surface terrain features are crucial for autonomous spaceflight applications, including Terrain Relative Navigation (TRN), Entry, Descent, and Landing (EDL), hazard analysis, and scientific data collection. Traditional photoclinometry-based pipelines often rely on extensive a priori imaging and offline processing, constrained by the computational limitations of radiation-hardened systems. While historically effective, these approaches typically increase mission costs and duration, operate at low processing rates, and have limited generalization. Recently, learning-based computer vision has gained popularity to enhance spacecraft autonomy and overcome these limitations. While promising, emerging techniques frequently impose computational demands exceeding the capabilities of typical spacecraft hardware for real-time operation and are further challenged by the scarcity of labeled training data for diverse extraterrestrial environments. In this work, we present novel formulations for in-situ landmark tracking via detection and description. We utilize lightweight, computationally efficient neural network architectures designed for real-time execution on current-generation spacecraft flight processors. For landmark detection, we propose improved domain adaptation methods that enable the identification of celestial terrain features with distinct, cheaply acquired training data. Concurrently, for landmark description, we introduce a novel attention alignment formulation that learns robust feature representations that maintain correspondence despite significant landmark viewpoint variations. Together, these contributions form a unified system for landmark tracking that demonstrates superior performance compared to existing state-of-the-art techniques. 

**Abstract (ZH)**: 天体表面地形特征的检测与跟踪对于自主太空飞行应用，包括地形相对导航（TRN）、进入、下降和着陆（EDL）、危险分析和科学数据采集至关重要。传统基于光度几何的流水线往往依赖于大量的先验成像和离线处理，受限于辐射加固系统的计算限制。尽管历史上效果显著，这些方法通常会增加任务成本和时间，处理速率较低，并且泛化能力有限。近年来，基于学习的计算机视觉技术得到了广泛应用，以增强航天器的自主性和克服这些限制。虽然前景广阔，但新兴技术往往对典型航天器硬件的实时操作提出了超出其计算能力的要求，并且由于缺乏用于多样外太空环境的标记训练数据而面临挑战。在本文中，我们提出了新颖的原位地标跟踪形式化方法，通过检测和描述实现。我们利用针对当前代际航天器飞行处理器实时执行设计的轻量级、计算高效的神经网络架构。在地标检测方面，我们提出了改进的域适应方法，能够在经济获取的训练数据下识别具有独特特征的天体地理特征。同时，在地标描述方面，我们引入了一种新颖的注意力对齐形式化方法，学习具有鲁棒性且在地标视角变化显著的情况下仍能保持对应关系的特征表示。这些贡献共同形成了一个统一的地标跟踪系统，其性能优于现有的最先进的技术。 

---
# Geo-RepNet: Geometry-Aware Representation Learning for Surgical Phase Recognition in Endoscopic Submucosal Dissection 

**Title (ZH)**: Geo-RepNet：几何aware表示学习在内镜黏膜下剥离手术分期识别中的应用 

**Authors**: Rui Tang, Haochen Yin, Guankun Wang, Long Bai, An Wang, Huxin Gao, Jiazheng Wang, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.09294)  

**Abstract**: Surgical phase recognition plays a critical role in developing intelligent assistance systems for minimally invasive procedures such as Endoscopic Submucosal Dissection (ESD). However, the high visual similarity across different phases and the lack of structural cues in RGB images pose significant challenges. Depth information offers valuable geometric cues that can complement appearance features by providing insights into spatial relationships and anatomical structures. In this paper, we pioneer the use of depth information for surgical phase recognition and propose Geo-RepNet, a geometry-aware convolutional framework that integrates RGB image and depth information to enhance recognition performance in complex surgical scenes. Built upon a re-parameterizable RepVGG backbone, Geo-RepNet incorporates the Depth-Guided Geometric Prior Generation (DGPG) module that extracts geometry priors from raw depth maps, and the Geometry-Enhanced Multi-scale Attention (GEMA) to inject spatial guidance through geometry-aware cross-attention and efficient multi-scale aggregation. To evaluate the effectiveness of our approach, we construct a nine-phase ESD dataset with dense frame-level annotations from real-world ESD videos. Extensive experiments on the proposed dataset demonstrate that Geo-RepNet achieves state-of-the-art performance while maintaining robustness and high computational efficiency under complex and low-texture surgical environments. 

**Abstract (ZH)**: 手术阶段识别在内镜黏膜下剥离等微创手术智能辅助系统开发中发挥着关键作用。然而，不同阶段间的高视觉相似性和RGB图像中缺乏结构线索构成了重大挑战。深度信息提供了有价值的几何线索，可以补充外观特征，提供有关空间关系和解剖结构的见解。在本文中，我们首次探索利用深度信息进行手术阶段识别，并提出Geo-RepNet，一种几何感知卷积框架，通过整合RGB图像和深度信息来增强复杂手术场景下的识别性能。Geo-RepNet基于可重新参数化的RepVGG骨干网络，结合了深度引导几何先验生成（DGPG）模块和几何增强多尺度注意（GEMA），通过几何感知交叉注意和高效多尺度聚合注入空间指导。为了评估我们的方法的有效性，我们构建了一个包含来自真实内镜黏膜下剥离视频的密集帧级注释的九阶段ESD数据集。在所提出数据集上的广泛实验表明，Geo-RepNet在复杂和低纹理手术环境中实现了最先进的性能，同时保持了鲁棒性和高计算效率。 

---
# Learning and Transferring Better with Depth Information in Visual Reinforcement Learning 

**Title (ZH)**: 基于深度信息的学习与在视觉强化学习中的迁移学习改进 

**Authors**: Zichun Xu, Yuntao Li, Zhaomin Wang, Lei Zhuang, Guocai Yang, Jingdong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.09180)  

**Abstract**: Depth information is robust to scene appearance variations and inherently carries 3D spatial details. In this paper, a visual backbone based on the vision transformer is proposed to fuse RGB and depth modalities for enhancing generalization. Different modalities are first processed by separate CNN stems, and the combined convolutional features are delivered to the scalable vision transformer to obtain visual representations. Moreover, a contrastive unsupervised learning scheme is designed with masked and unmasked tokens to accelerate the sample efficiency during the reinforcement learning progress. For sim2real transfer, a flexible curriculum learning schedule is developed to deploy domain randomization over training processes. 

**Abstract (ZH)**: 基于视觉变换器的RGB和深度模态融合视觉骨干网络及其在强化学习中的对比无监督学习方案与仿真实验到现实世界的过渡方法 

---
# Instance space analysis of the capacitated vehicle routing problem 

**Title (ZH)**: 带容量约束车辆路径问题的实例空间分析 

**Authors**: Alessandra M. M. M. Gouvêa, Nuno Paulos, Eduardo Uchoa e Mariá C. V. Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2507.10397)  

**Abstract**: This paper seeks to advance CVRP research by addressing the challenge of understanding the nuanced relationships between instance characteristics and metaheuristic (MH) performance. We present Instance Space Analysis (ISA) as a valuable tool that allows for a new perspective on the field. By combining the ISA methodology with a dataset from the DIMACS 12th Implementation Challenge on Vehicle Routing, our research enabled the identification of 23 relevant instance characteristics. Our use of the PRELIM, SIFTED, and PILOT stages, which employ dimensionality reduction and machine learning methods, allowed us to create a two-dimensional projection of the instance space to understand how the structure of instances affect the behavior of MHs. A key contribution of our work is that we provide a projection matrix, which makes it straightforward to incorporate new instances into this analysis and allows for a new method for instance analysis in the CVRP field. 

**Abstract (ZH)**: 本文旨在通过探讨实例特征与元启发式算法性能之间的复杂关系，推进车辆路线问题（CVRP）研究。我们提出了实例空间分析（ISA）作为一项有价值的工具，提供了该领域的全新视角。结合ISA方法与DIMACS第12届实现挑战赛中的车辆路由数据集，我们的研究识别了23个相关实例特征。通过PRELIM、SIFTED和PILOT阶段，我们采用了降维和机器学习方法，创建了实例空间的二维投影，以理解实例结构如何影响元启发式算法的行为。本文的一个重要贡献是，我们提供了投影矩阵，便于将新实例纳入这种分析，并为CVRP领域提供了实例分析的新方法。 

---
# Self-supervised Learning on Camera Trap Footage Yields a Strong Universal Face Embedder 

**Title (ZH)**: 自我监督学习在相机陷阱视频上的应用yield一个强大的通用面部嵌入器 

**Authors**: Vladimir Iashin, Horace Lee, Dan Schofield, Andrew Zisserman  

**Link**: [PDF](https://arxiv.org/pdf/2507.10552)  

**Abstract**: Camera traps are revolutionising wildlife monitoring by capturing vast amounts of visual data; however, the manual identification of individual animals remains a significant bottleneck. This study introduces a fully self-supervised approach to learning robust chimpanzee face embeddings from unlabeled camera-trap footage. Leveraging the DINOv2 framework, we train Vision Transformers on automatically mined face crops, eliminating the need for identity labels. Our method demonstrates strong open-set re-identification performance, surpassing supervised baselines on challenging benchmarks such as Bossou, despite utilising no labelled data during training. This work underscores the potential of self-supervised learning in biodiversity monitoring and paves the way for scalable, non-invasive population studies. 

**Abstract (ZH)**: 相机trap正在通过捕捉大量视觉数据革新野生动物监测；然而，个体动物的手动识别仍然是一个显著的瓶颈。本研究引入了一种完全自监督的方法，从未标记的相机trap录像中学习 robust 猩猩面部嵌入。利用DINOv2框架，我们对自动提取的面部裁剪进行Vision Transformers训练，消除了身份标签的需求。我们的方法在Bossou等具有挑战性的基准测试上展示了强大的开放集重新识别性能，尽管在训练过程中未使用任何标记数据。本研究强调了自监督学习在生物多样性监测中的潜力，并为可扩展的非侵入性种群研究铺平了道路。 

---
# ScaffoldAvatar: High-Fidelity Gaussian Avatars with Patch Expressions 

**Title (ZH)**: ScaffoldAvatar: 高保真高斯 avatar 与补丁表情 

**Authors**: Shivangi Aneja, Sebastian Weiss, Irene Baeza, Prashanth Chandran, Gaspard Zoss, Matthias Nießner, Derek Bradley  

**Link**: [PDF](https://arxiv.org/pdf/2507.10542)  

**Abstract**: Generating high-fidelity real-time animated sequences of photorealistic 3D head avatars is important for many graphics applications, including immersive telepresence and movies. This is a challenging problem particularly when rendering digital avatar close-ups for showing character's facial microfeatures and expressions. To capture the expressive, detailed nature of human heads, including skin furrowing and finer-scale facial movements, we propose to couple locally-defined facial expressions with 3D Gaussian splatting to enable creating ultra-high fidelity, expressive and photorealistic 3D head avatars. In contrast to previous works that operate on a global expression space, we condition our avatar's dynamics on patch-based local expression features and synthesize 3D Gaussians at a patch level. In particular, we leverage a patch-based geometric 3D face model to extract patch expressions and learn how to translate these into local dynamic skin appearance and motion by coupling the patches with anchor points of Scaffold-GS, a recent hierarchical scene representation. These anchors are then used to synthesize 3D Gaussians on-the-fly, conditioned by patch-expressions and viewing direction. We employ color-based densification and progressive training to obtain high-quality results and faster convergence for high resolution 3K training images. By leveraging patch-level expressions, ScaffoldAvatar consistently achieves state-of-the-art performance with visually natural motion, while encompassing diverse facial expressions and styles in real time. 

**Abstract (ZH)**: 基于局部定义的表情结合3D高斯点绘制生成超高保真实时光ướ实景头部avatar 

---
# Cameras as Relative Positional Encoding 

**Title (ZH)**: 摄像头作为相对位置编码 

**Authors**: Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, Angjoo Kanazawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.10496)  

**Abstract**: Transformers are increasingly prevalent for multi-view computer vision tasks, where geometric relationships between viewpoints are critical for 3D perception. To leverage these relationships, multi-view transformers must use camera geometry to ground visual tokens in 3D space. In this work, we compare techniques for conditioning transformers on cameras: token-level raymap encodings, attention-level relative pose encodings, and a new relative encoding we propose -- Projective Positional Encoding (PRoPE) -- that captures complete camera frustums, both intrinsics and extrinsics, as a relative positional encoding. Our experiments begin by showing how relative camera conditioning improves performance in feedforward novel view synthesis, with further gains from PRoPE. This holds across settings: scenes with both shared and varying intrinsics, when combining token- and attention-level conditioning, and for generalization to inputs with out-of-distribution sequence lengths and camera intrinsics. We then verify that these benefits persist for different tasks, stereo depth estimation and discriminative spatial cognition, as well as larger model sizes. 

**Abstract (ZH)**: 多视图变换器在利用摄像头几何关系进行三维感知方面越来越普遍。我们比较了变换器基于摄像头的条件训练技术：基于代理解码的标记级射线图编码、基于相对位姿的注意级别编码以及我们提出的新相对编码——投影位置编码（PRoPE），它捕捉包括内参和外参在内的完整摄像机视锥作为相对位置编码。我们的实验首先展示了基于相对摄像头条件可以提高前向新颖视图合成的性能，并且通过使用PRoPE还能进一步提升。这种改进在不同场景中保持有效，包括共享和变化内参的场景，当结合标记级和注意级别条件时，以及对于输入的分布外序列长度和内参的一般泛化。此外，我们验证了这些优势可以在不同的任务，如立体深度估计和区分数学空间认知，以及更大模型规模中保持有效。 

---
# AudioMAE++: learning better masked audio representations with SwiGLU FFNs 

**Title (ZH)**: AudioMAE++: 使用SwiGLU FFNs学习更好的掩蔽音频表示 

**Authors**: Sarthak Yadav, Sergios Theodoridis, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10464)  

**Abstract**: Masked Autoencoders (MAEs) trained on audio spectrogram patches have emerged as a prominent approach for learning self-supervised audio representations. While several recent papers have evaluated key aspects of training MAEs on audio data, the majority of these approaches still leverage vanilla transformer building blocks, whereas the transformer community has seen steady integration of newer architectural advancements. In this work, we propose AudioMAE++, a revamped audio masked autoencoder with two such enhancements, namely macaron-style transformer blocks with gated linear units. When pretrained on the AudioSet dataset, the proposed AudioMAE++ models outperform existing MAE based approaches on 10 diverse downstream tasks, demonstrating excellent performance on audio classification and speech-based benchmarks. The proposed AudioMAE++ models also demonstrate excellent scaling characteristics, outperforming directly comparable standard MAE baselines with up to 4x more parameters. 

**Abstract (ZH)**: 基于音频光谱图片段训练的掩码自动编码器（MAEs）已经成为学习自监督音频表示的一种 prominant 方法。尽管近期有多篇论文评估了在音频数据上训练 MAEs 的关键方面，大多数这些方法仍然使用基本的变压器构建块，而变压器社区已经稳定地将新的架构 advancements 融合进来。在此项工作中，我们提出了 AudioMAE++，这是一种带有两种改进的重新设计的音频掩码自动编码器，具体来说是带门线性单元的 macaron 风格变压器块。通过在 AudioSet 数据集上预训练，提出的方法在 10 个不同的下游任务中优于现有的基于 MAE 的方法，展示了在音频分类和语音基准测试中的出色性能。提出的 AudioMAE++ 模型还展示了优异的扩展特性，其参数量最多比直接可比的标准 MAE 基线多 4 倍但仍表现出色。 

---
# RAPNet: A Receptive-Field Adaptive Convolutional Neural Network for Pansharpening 

**Title (ZH)**: RAPNet：一种适应性卷积神经网络用于多光谱与高分辨率影像融合 

**Authors**: Tao Tang, Chengxu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10461)  

**Abstract**: Pansharpening refers to the process of integrating a high resolution panchromatic (PAN) image with a lower resolution multispectral (MS) image to generate a fused product, which is pivotal in remote sensing. Despite the effectiveness of CNNs in addressing this challenge, they are inherently constrained by the uniform application of convolutional kernels across all spatial positions, overlooking local content variations. To overcome this issue, we introduce RAPNet, a new architecture that leverages content-adaptive convolution. At its core, RAPNet employs the Receptive-field Adaptive Pansharpening Convolution (RAPConv), designed to produce spatially adaptive kernels responsive to local feature context, thereby enhancing the precision of spatial detail extraction. Additionally, the network integrates the Pansharpening Dynamic Feature Fusion (PAN-DFF) module, which incorporates an attention mechanism to achieve an optimal balance between spatial detail enhancement and spectral fidelity. Comprehensive evaluations on publicly available datasets confirm that RAPNet delivers superior performance compared to existing approaches, as demonstrated by both quantitative metrics and qualitative assessments. Ablation analyses further substantiate the effectiveness of the proposed adaptive components. 

**Abstract (ZH)**: pansharpening是指将高分辨率Panchromatic（PAN）图像与低分辨率多光谱（MS）图像相结合以生成融合产品的过程，在遥感中至关重要。尽管CNNs在解决这一挑战方面效果显著，但它们固有的缺点是卷积核在所有空间位置上均匀应用，忽视了局部内容的变化。为克服这一问题，我们提出了RAPNet，一种利用内容自适应卷积的新架构。其核心在于使用接收域自适应卷积（RAPConv），旨在生成响应局部特征上下文的自适应卷积核，从而增强空间细节提取的精度。此外，网络整合了Pansharpening动态特征融合（PAN-DFF）模块，该模块包含注意力机制，以实现空间细节增强与光谱保真的最佳平衡。在公开数据集上的全面评估表明，RAPNet在定量指标和定性评估方面均优于现有方法。消融分析进一步证实了所提自适应组件的有效性。 

---
# CoralVQA: A Large-Scale Visual Question Answering Dataset for Coral Reef Image Understanding 

**Title (ZH)**: CoralVQA：一种用于珊瑚礁图像理解的大规模视觉问答数据集 

**Authors**: Hongyong Han, Wei Wang, Gaowei Zhang, Mingjie Li, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10449)  

**Abstract**: Coral reefs are vital yet vulnerable ecosystems that require continuous monitoring to support conservation. While coral reef images provide essential information in coral monitoring, interpreting such images remains challenging due to the need for domain expertise. Visual Question Answering (VQA), powered by Large Vision-Language Models (LVLMs), has great potential in user-friendly interaction with coral reef images. However, applying VQA to coral imagery demands a dedicated dataset that addresses two key challenges: domain-specific annotations and multidimensional questions. In this work, we introduce CoralVQA, the first large-scale VQA dataset for coral reef analysis. It contains 12,805 real-world coral images from 67 coral genera collected from 3 oceans, along with 277,653 question-answer pairs that comprehensively assess ecological and health-related conditions. To construct this dataset, we develop a semi-automatic data construction pipeline in collaboration with marine biologists to ensure both scalability and professional-grade data quality. CoralVQA presents novel challenges and provides a comprehensive benchmark for studying vision-language reasoning in the context of coral reef images. By evaluating several state-of-the-art LVLMs, we reveal key limitations and opportunities. These insights form a foundation for future LVLM development, with a particular emphasis on supporting coral conservation efforts. 

**Abstract (ZH)**: 珊瑚礁是至关重要的但又脆弱的生态系统，需要持续监测以支持保护工作。虽然珊瑚礁图像提供了珊瑚监测所需的重要信息，但由于需要领域专业知识，解读这些图像依然具有挑战性。基于大型视觉-语言模型的视觉问答（VQA）技术在与珊瑚礁图像的友好交互方面具有巨大潜力。然而，将VQA应用于珊瑚图像需要一个专门的数据集来应对两个关键挑战：领域特定的注释和多维问题。本文介绍了CoralVQA，这是首个用于珊瑚礁分析的大规模VQA数据集，包含来自67种珊瑚属、3大海域的12,805张真实珊瑚图像以及277,653个问题回答对，全面评估生态和健康状况。为了构建该数据集，我们与海洋生物学家合作，开发了一种半自动的数据构建 pipeline，确保了可扩展性和专业级数据质量。CoralVQA 提出了新的挑战，并为研究珊瑚礁图像中的视觉-语言推理提供了全面基准。通过评估多种最先进视觉语言模型，我们揭示了关键的局限性和机会。这些见解为未来视觉语言模型的发展奠定了基础，特别是支持珊瑚保护工作。 

---
# Response Wide Shut? Surprising Observations in Basic Vision Language Model Capabilities 

**Title (ZH)**: Wide Shut？基本视觉语言模型能力的惊讶观察 

**Authors**: Shivam Chandhok, Wan-Cyuan Fan, Vered Shwartz, Vineeth N Balasubramanian, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2507.10442)  

**Abstract**: Vision-language Models (VLMs) have emerged as general-purpose tools for addressing a variety of complex computer vision problems. Such models have been shown to be highly capable, but, at the same time, lacking some basic visual understanding skills. In this paper, we set out to understand the limitations of SoTA VLMs on fundamental visual tasks by constructing a series of tests that probe which components of design, specifically, may be lacking. Importantly, we go significantly beyond the current benchmarks, which simply measure the final performance of VLM response, by also comparing and contrasting it to the performance of probes trained directly on features obtained from the visual encoder, intermediate vision-language projection and LLM-decoder output. In doing so, we uncover shortcomings in VLMs and make a number of important observations about their capabilities, robustness and how they process visual information. We hope our insights will guide progress in further improving VLMs. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）已成为解决各种复杂计算机视觉问题的一般工具。这类模型已被证明具有高度的能力，但同时在一些基本的视觉理解技能方面却显得不足。在本文中，我们通过构建一系列测试来理解最先进的VLM在基本视觉任务上的局限性，具体探查哪些设计方案可能缺失。重要的是，我们超越了现有的基准测试，不仅衡量VLM响应的最终性能，还将其与直接在视觉编码器特征、中间的视觉-语言投影以及大语言模型解码器输出上训练的探针的性能进行比较和对照。通过这种方式，我们揭示了VLM的不足，并对其能力、稳健性以及处理视觉信息的方式提出了若干重要观察。我们希望我们的见解能指导进一步改进VLM的工作。 

---
# Devanagari Handwritten Character Recognition using Convolutional Neural Network 

**Title (ZH)**: 基于卷积神经网络的Devanagari手写字符识别 

**Authors**: Diksha Mehta, Prateek Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2507.10398)  

**Abstract**: Handwritten character recognition is getting popular among researchers because of its possible applications in facilitating technological search engines, social media, recommender systems, etc. The Devanagari script is one of the oldest language scripts in India that does not have proper digitization tools. With the advancement of computing and technology, the task of this research is to extract handwritten Hindi characters from an image of Devanagari script with an automated approach to save time and obsolete data. In this paper, we present a technique to recognize handwritten Devanagari characters using two deep convolutional neural network layers. This work employs a methodology that is useful to enhance the recognition rate and configures a convolutional neural network for effective Devanagari handwritten text recognition (DHTR). This approach uses the Devanagari handwritten character dataset (DHCD), an open dataset with 36 classes of Devanagari characters. Each of these classes has 1700 images for training and testing purposes. This approach obtains promising results in terms of accuracy by achieving 96.36% accuracy in testing and 99.55% in training time. 

**Abstract (ZH)**: 手写字符识别由于其在促进技术搜索引擎、社交媒体、推荐系统等方面的应用而日益受到研究者的关注。印度的德文加班字符是其中一种古老的文字体系，缺乏相应的数字化工具。随着计算和科技的进步，本研究的任务是采用自动化方法从德文加班字体图像中提取手写印地文字符，节省时间和避免过时数据。本文提出了一种使用两层深度卷积神经网络层识别手写德文加班字符的技术。该工作采用了一种有助于提高识别率的方法，并配置了一个用于有效识别手写德文文本（DHTR）的卷积神经网络。该方法使用了德文加班手写字符数据集（DHCD），这是一个包含36类德文加班字符的开放数据集，每类有1700张用于训练和测试的图像。该方法在测试和训练时间分别获得了96.36%和99.55%的准确率，取得了令人鼓舞的结果。 

---
# DepViT-CAD: Deployable Vision Transformer-Based Cancer Diagnosis in Histopathology 

**Title (ZH)**: DepViT-CAD: 可部署的基于视觉变换器的病理癌症诊断方法 

**Authors**: Ashkan Shakarami, Lorenzo Nicole, Rocco Cappellesso, Angelo Paolo Dei Tos, Stefano Ghidoni  

**Link**: [PDF](https://arxiv.org/pdf/2507.10250)  

**Abstract**: Accurate and timely cancer diagnosis from histopathological slides is vital for effective clinical decision-making. This paper introduces DepViT-CAD, a deployable AI system for multi-class cancer diagnosis in histopathology. At its core is MAViT, a novel Multi-Attention Vision Transformer designed to capture fine-grained morphological patterns across diverse tumor types. MAViT was trained on expert-annotated patches from 1008 whole-slide images, covering 11 diagnostic categories, including 10 major cancers and non-tumor tissue. DepViT-CAD was validated on two independent cohorts: 275 WSIs from The Cancer Genome Atlas and 50 routine clinical cases from pathology labs, achieving diagnostic sensitivities of 94.11% and 92%, respectively. By combining state-of-the-art transformer architecture with large-scale real-world validation, DepViT-CAD offers a robust and scalable approach for AI-assisted cancer diagnostics. To support transparency and reproducibility, software and code will be made publicly available at GitHub. 

**Abstract (ZH)**: 准确及时地从组织病理切片中进行癌症诊断对于有效的临床决策至关重要。本文介绍了一种可部署的AI系统DepViT-CAD，用于组织病理学中的多类别癌症诊断。其核心是MAViT，这是一种新的多注意力视觉变换器，旨在捕捉不同肿瘤类型中的细微形态模式。MAViT基于1008张全切片图像上的专家注释片段训练，涵盖11个诊断类别，包括10种主要癌症和非肿瘤组织。DepViT-CAD分别在The Cancer Genome Atlas的两个独立队列（275张WSI）和病理实验室的50例常规临床病例中进行了验证，诊断灵敏度分别为94.11%和92%。通过结合最先进的变换器架构和大规模现实世界验证，DepViT-CAD提供了一种稳健且可扩展的人工智能辅助癌症诊断方法。为支持透明性和可重复性，软件和代码将公开发布在GitHub上。 

---
# ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users 

**Title (ZH)**: ProGait: 一种适用于股膝置换假肢使用者的多功能视频数据集和基准 

**Authors**: Xiangyu Yin, Boyuan Yang, Weichen Liu, Qiyao Xue, Abrar Alamri, Goeran Fiedler, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.10223)  

**Abstract**: Prosthetic legs play a pivotal role in clinical rehabilitation, allowing individuals with lower-limb amputations the ability to regain mobility and improve their quality of life. Gait analysis is fundamental for optimizing prosthesis design and alignment, directly impacting the mobility and life quality of individuals with lower-limb amputations. Vision-based machine learning (ML) methods offer a scalable and non-invasive solution to gait analysis, but face challenges in correctly detecting and analyzing prosthesis, due to their unique appearances and new movement patterns. In this paper, we aim to bridge this gap by introducing a multi-purpose dataset, namely ProGait, to support multiple vision tasks including Video Object Segmentation, 2D Human Pose Estimation, and Gait Analysis (GA). ProGait provides 412 video clips from four above-knee amputees when testing multiple newly-fitted prosthetic legs through walking trials, and depicts the presence, contours, poses, and gait patterns of human subjects with transfemoral prosthetic legs. Alongside the dataset itself, we also present benchmark tasks and fine-tuned baseline models to illustrate the practical application and performance of the ProGait dataset. We compared our baseline models against pre-trained vision models, demonstrating improved generalizability when applying the ProGait dataset for prosthesis-specific tasks. Our code is available at this https URL and dataset at this https URL. 

**Abstract (ZH)**: 假肢在临床康复中扮演着关键角色，使下肢缺失者能够恢复 mobility 并提高生活质量。步态分析对于优化假肢设计和对齐至关重要，直接影响下肢缺失者的生活质量和移动能力。基于视觉的机器学习方法为步态分析提供了可扩展且非侵入性的解决方案，但在正确检测和分析假肢方面面临挑战，因为假肢具有独特的外观和新的运动模式。在本文中，我们通过引入名为 ProGait 的多功能数据集来弥补这一差距，该数据集支持包括视频对象分割、2D 人体姿态估计和步态分析在内的多种视觉任务。ProGait 提供了 412 个视频片段，记录了四位膝上截肢者在多次佩戴新装假肢行走试验中的人类主体的假肢存在、轮廓、姿态和步态模式。除了数据集本身，我们还提供基准任务和微调基线模型来说明 ProGait 数据集的实际应用和性能。我们将基准模型与预训练视觉模型进行了比较，展示了使用 ProGait 数据集进行假肢特定任务时的一般泛化能力提高。我们的代码可在以下网址访问：this https URL，数据集可在以下网址访问：this https URL。 

---
# Taming Modern Point Tracking for Speckle Tracking Echocardiography via Impartial Motion 

**Title (ZH)**: 基于公允运动的现代点跟踪驯化技术在Speckle跟踪心脏超声中的应用 

**Authors**: Md Abulkalam Azad, John Nyberg, Håvard Dalen, Bjørnar Grenne, Lasse Lovstakken, Andreas Østvik  

**Link**: [PDF](https://arxiv.org/pdf/2507.10127)  

**Abstract**: Accurate motion estimation for tracking deformable tissues in echocardiography is essential for precise cardiac function measurements. While traditional methods like block matching or optical flow struggle with intricate cardiac motion, modern point tracking approaches remain largely underexplored in this domain. This work investigates the potential of state-of-the-art (SOTA) point tracking methods for ultrasound, with a focus on echocardiography. Although these novel approaches demonstrate strong performance in general videos, their effectiveness and generalizability in echocardiography remain limited. By analyzing cardiac motion throughout the heart cycle in real B-mode ultrasound videos, we identify that a directional motion bias across different views is affecting the existing training strategies. To mitigate this, we refine the training procedure and incorporate a set of tailored augmentations to reduce the bias and enhance tracking robustness and generalization through impartial cardiac motion. We also propose a lightweight network leveraging multi-scale cost volumes from spatial context alone to challenge the advanced spatiotemporal point tracking models. Experiments demonstrate that fine-tuning with our strategies significantly improves models' performances over their baselines, even for out-of-distribution (OOD) cases. For instance, EchoTracker boosts overall position accuracy by 60.7% and reduces median trajectory error by 61.5% across heart cycle phases. Interestingly, several point tracking models fail to outperform our proposed simple model in terms of tracking accuracy and generalization, reflecting their limitations when applied to echocardiography. Nevertheless, clinical evaluation reveals that these methods improve GLS measurements, aligning more closely with expert-validated, semi-automated tools and thus demonstrating better reproducibility in real-world applications. 

**Abstract (ZH)**: 基于最新点跟踪方法的心脏超声变形组织准确运动估计研究 

---
# Lightweight Model for Poultry Disease Detection from Fecal Images Using Multi-Color Space Feature Optimization and Machine Learning 

**Title (ZH)**: 基于多色彩空间特征优化与机器学习的轻量级家禽疾病检测模型研宄 

**Authors**: A. K. M. Shoriful Islam, Md. Rakib Hassan, Macbah Uddin, Md. Shahidur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2507.10056)  

**Abstract**: Poultry farming is a vital component of the global food supply chain, yet it remains highly vulnerable to infectious diseases such as coccidiosis, salmonellosis, and Newcastle disease. This study proposes a lightweight machine learning-based approach to detect these diseases by analyzing poultry fecal images. We utilize multi-color space feature extraction (RGB, HSV, LAB) and explore a wide range of color, texture, and shape-based descriptors, including color histograms, local binary patterns (LBP), wavelet transforms, and edge detectors. Through a systematic ablation study and dimensionality reduction using PCA and XGBoost feature selection, we identify a compact global feature set that balances accuracy and computational efficiency. An artificial neural network (ANN) classifier trained on these features achieved 95.85% accuracy while requiring no GPU and only 638 seconds of execution time in Google Colab. Compared to deep learning models such as Xception and MobileNetV3, our proposed model offers comparable accuracy with drastically lower resource usage. This work demonstrates a cost-effective, interpretable, and scalable alternative to deep learning for real-time poultry disease detection in low-resource agricultural settings. 

**Abstract (ZH)**: 基于轻量级机器学习的 poultry 粪便图像分析方法在检测传染性疾病中的应用：一种在低资源农业环境中实现实时家禽疾病检测的经济、可解释且可扩展的替代方案 

---
# Advanced U-Net Architectures with CNN Backbones for Automated Lung Cancer Detection and Segmentation in Chest CT Images 

**Title (ZH)**: 基于CNN骨干网络的高级U-Net架构在胸部CT图像中自动肺癌检测与分割 

**Authors**: Alireza Golkarieha, Kiana Kiashemshakib, Sajjad Rezvani Boroujenic, Nasibeh Asadi Isakand  

**Link**: [PDF](https://arxiv.org/pdf/2507.09898)  

**Abstract**: This study investigates the effectiveness of U-Net architectures integrated with various convolutional neural network (CNN) backbones for automated lung cancer detection and segmentation in chest CT images, addressing the critical need for accurate diagnostic tools in clinical settings. A balanced dataset of 832 chest CT images (416 cancerous and 416 non-cancerous) was preprocessed using Contrast Limited Adaptive Histogram Equalization (CLAHE) and resized to 128x128 pixels. U-Net models were developed with three CNN backbones: ResNet50, VGG16, and Xception, to segment lung regions. After segmentation, CNN-based classifiers and hybrid models combining CNN feature extraction with traditional machine learning classifiers (Support Vector Machine, Random Forest, and Gradient Boosting) were evaluated using 5-fold cross-validation. Metrics included accuracy, precision, recall, F1-score, Dice coefficient, and ROC-AUC. U-Net with ResNet50 achieved the best performance for cancerous lungs (Dice: 0.9495, Accuracy: 0.9735), while U-Net with VGG16 performed best for non-cancerous segmentation (Dice: 0.9532, Accuracy: 0.9513). For classification, the CNN model using U-Net with Xception achieved 99.1 percent accuracy, 99.74 percent recall, and 99.42 percent F1-score. The hybrid CNN-SVM-Xception model achieved 96.7 percent accuracy and 97.88 percent F1-score. Compared to prior methods, our framework consistently outperformed existing models. In conclusion, combining U-Net with advanced CNN backbones provides a powerful method for both segmentation and classification of lung cancer in CT scans, supporting early diagnosis and clinical decision-making. 

**Abstract (ZH)**: 本研究探讨了结合各种卷积神经网络（CNN）骨干网络的U-Net架构在胸部CT图像中自动化肺癌检测和分割的有效性，满足了临床环境中对准确诊断工具的迫切需求。使用对比受限自适应直方图均衡化（CLAHE）对一个平衡的数据集（832张胸部CT图像，包括416张癌性图像和416张非癌性图像）进行预处理，并将其调整为128x128像素。开发了三种CNN骨干网络（ResNet50、VGG16和Xception）的U-Net模型，用于分割肺部区域。分割后，基于CNN的分类器以及结合CNN特征提取与传统机器学习分类器（支持向量机、随机森林和梯度提升）的混合模型，采用5折交叉验证进行评估。评估指标包括准确率、精确率、召回率、F1分数、Dice系数和ROC-AUC。使用ResNet50的U-Net在癌性肺部分割中表现最佳（Dice：0.9495，准确率：0.9735），而使用VGG16的U-Net在非癌性分割中表现最佳（Dice：0.9532，准确率：0.9513）。在分类方面，使用Xception的U-Net CNN模型实现了99.1%的准确率、99.74%的召回率和99.42%的F1分数。混合CNN-SVM-Xception模型的准确率为96.7%，F1分数为97.88%。与先前的方法相比，我们的框架在各指标上持续优于现有模型。综上所述，结合U-Net与高级CNN骨干网络为CT扫描中的肺癌分割和分类提供了一种强大的方法，支持早期诊断和临床决策。 

---
# Secure and Efficient UAV-Based Face Detection via Homomorphic Encryption and Edge Computing 

**Title (ZH)**: 基于同态加密和边缘计算的无人机Face检测安全与效率提升 

**Authors**: Nguyen Van Duc, Bui Duc Manh, Quang-Trung Luu, Dinh Thai Hoang, Van-Linh Nguyen, Diep N. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09860)  

**Abstract**: This paper aims to propose a novel machine learning (ML) approach incorporating Homomorphic Encryption (HE) to address privacy limitations in Unmanned Aerial Vehicles (UAV)-based face detection. Due to challenges related to distance, altitude, and face orientation, high-resolution imagery and sophisticated neural networks enable accurate face recognition in dynamic environments. However, privacy concerns arise from the extensive surveillance capabilities of UAVs. To resolve this issue, we propose a novel framework that integrates HE with advanced neural networks to secure facial data throughout the inference phase. This method ensures that facial data remains secure with minimal impact on detection accuracy. Specifically, the proposed system leverages the Cheon-Kim-Kim-Song (CKKS) scheme to perform computations directly on encrypted data, optimizing computational efficiency and security. Furthermore, we develop an effective data encoding method specifically designed to preprocess the raw facial data into CKKS form in a Single-Instruction-Multiple-Data (SIMD) manner. Building on this, we design a secure inference algorithm to compute on ciphertext without needing decryption. This approach not only protects data privacy during the processing of facial data but also enhances the efficiency of UAV-based face detection systems. Experimental results demonstrate that our method effectively balances privacy protection and detection performance, making it a viable solution for UAV-based secure face detection. Significantly, our approach (while maintaining data confidentially with HE encryption) can still achieve an accuracy of less than 1% compared to the benchmark without using encryption. 

**Abstract (ZH)**: 基于同态加密的无人机面部检测新型机器学习方法 

---
# AI-Enhanced Pediatric Pneumonia Detection: A CNN-Based Approach Using Data Augmentation and Generative Adversarial Networks (GANs) 

**Title (ZH)**: 基于数据增强和生成对抗网络（GANs）的AI增强儿童肺炎检测：一种CNN方法 

**Authors**: Abdul Manaf, Nimra Mughal  

**Link**: [PDF](https://arxiv.org/pdf/2507.09759)  

**Abstract**: Pneumonia is a leading cause of mortality in children under five, requiring accurate chest X-ray diagnosis. This study presents a machine learning-based Pediatric Chest Pneumonia Classification System to assist healthcare professionals in diagnosing pneumonia from chest X-ray images. The CNN-based model was trained on 5,863 labeled chest X-ray images from children aged 0-5 years from the Guangzhou Women and Children's Medical Center. To address limited data, we applied augmentation techniques (rotation, zooming, shear, horizontal flipping) and employed GANs to generate synthetic images, addressing class imbalance. The system achieved optimal performance using combined original, augmented, and GAN-generated data, evaluated through accuracy and F1 score metrics. The final model was deployed via a Flask web application, enabling real-time classification with probability estimates. Results demonstrate the potential of deep learning and GANs in improving diagnostic accuracy and efficiency for pediatric pneumonia classification, particularly valuable in resource-limited clinical settings this https URL 

**Abstract (ZH)**: 儿童肺炎的胸部X光诊断：基于机器学习的儿科胸部肺炎分类系统 

---
# Prompt Engineering in Segment Anything Model: Methodologies, Applications, and Emerging Challenges 

**Title (ZH)**: Segment Anything模型中的提示工程：方法、应用及新兴挑战 

**Authors**: Yidong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09562)  

**Abstract**: The Segment Anything Model (SAM) has revolutionized image segmentation through its innovative prompt-based approach, yet the critical role of prompt engineering in its success remains underexplored. This paper presents the first comprehensive survey focusing specifically on prompt engineering techniques for SAM and its variants. We systematically organize and analyze the rapidly growing body of work in this emerging field, covering fundamental methodologies, practical applications, and key challenges. Our review reveals how prompt engineering has evolved from simple geometric inputs to sophisticated multimodal approaches, enabling SAM's adaptation across diverse domains including medical imaging and remote sensing. We identify unique challenges in prompt optimization and discuss promising research directions. This survey fills an important gap in the literature by providing a structured framework for understanding and advancing prompt engineering in foundation models for segmentation. 

**Abstract (ZH)**: Segment Anything模型（SAM）通过其创新的提示基础方法在图像分割领域取得了革命性的进展，但提示工程在其成功中的关键作用仍鲜有探索。本文提供了首个专注于SAM及其变种的提示工程技术的全面综述。我们系统地组织和分析了这一新兴领域中迅速增长的研究成果，涵盖基本方法、实用应用以及关键挑战。我们的综述揭示了提示工程从简单的几何输入发展到复杂的多模态方法的过程，使SAM能够跨医学成像和遥感等不同领域进行适应。我们指出了提示优化的独特挑战，并讨论了有希望的研究方向。本文通过提供一个结构化的框架来理解和推动基础模型中的提示工程的发展，填补了文献中的重要空白。 

---
# SDTN and TRN: Adaptive Spectral-Spatial Feature Extraction for Hyperspectral Image Classification 

**Title (ZH)**: SDTN和TRN：适应性谱-空特征提取方法在高光谱图像分类中的应用 

**Authors**: Fuyin Ye, Erwen Yao, Jianyong Chen, Fengmei He, Junxiang Zhang, Lihao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2507.09492)  

**Abstract**: Hyperspectral image classification plays a pivotal role in precision agriculture, providing accurate insights into crop health monitoring, disease detection, and soil analysis. However, traditional methods struggle with high-dimensional data, spectral-spatial redundancy, and the scarcity of labeled samples, often leading to suboptimal performance. To address these challenges, we propose the Self-Adaptive Tensor- Regularized Network (SDTN), which combines tensor decomposition with regularization mechanisms to dynamically adjust tensor ranks, ensuring optimal feature representation tailored to the complexity of the data. Building upon SDTN, we propose the Tensor-Regularized Network (TRN), which integrates the features extracted by SDTN into a lightweight network capable of capturing spectral-spatial features at multiple scales. This approach not only maintains high classification accuracy but also significantly reduces computational complexity, making the framework highly suitable for real-time deployment in resource-constrained environments. Experiments on PaviaU datasets demonstrate significant improvements in accuracy and reduced model parameters compared to state-of-the-art methods. 

**Abstract (ZH)**: 高维光谱图像分类在精准农业中发挥着关键作用，为作物健康监测、病害检测和土壤分析提供准确洞察。然而，传统方法难以处理高维数据、光谱-空间冗余以及标签样本稀缺的问题，常常导致性能不佳。为应对这些挑战，我们提出了一种自适应张量正则化网络（SDTN），它结合了张量分解和正则化机制，动态调整张量秩，确保针对数据复杂性的最优特征表示。在此基础上，我们提出了张量正则化网络（TRN），它将SDTN提取的特征整合到一个轻量级网络中，能够多尺度捕获光谱-空间特征。该方法不仅保持了高分类精度，还显著降低了计算复杂度，使框架在资源受限环境中具有高度实时部署的适用性。实验结果表明，TRN在PaviaU数据集上的准确性和模型参数方面显著优于现有方法。 

---
# AlphaVAE: Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning 

**Title (ZH)**: AlphaVAE：统一的端到端RGBA图像重建与生成及awarealpha表示学习 

**Authors**: Zile Wang, Hao Yu, Jiabo Zhan, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09308)  

**Abstract**: Recent advances in latent diffusion models have achieved remarkable results in high-fidelity RGB image synthesis by leveraging pretrained VAEs to compress and reconstruct pixel data at low computational cost. However, the generation of transparent or layered content (RGBA image) remains largely unexplored, due to the lack of large-scale benchmarks. In this work, we propose ALPHA, the first comprehensive RGBA benchmark that adapts standard RGB metrics to four-channel images via alpha blending over canonical backgrounds. We further introduce ALPHAVAE, a unified end-to-end RGBA VAE that extends a pretrained RGB VAE by incorporating a dedicated alpha channel. The model is trained with a composite objective that combines alpha-blended pixel reconstruction, patch-level fidelity, perceptual consistency, and dual KL divergence constraints to ensure latent fidelity across both RGB and alpha representations. Our RGBA VAE, trained on only 8K images in contrast to 1M used by prior methods, achieves a +4.9 dB improvement in PSNR and a +3.2% increase in SSIM over LayerDiffuse in reconstruction. It also enables superior transparent image generation when fine-tuned within a latent diffusion framework. Our code, data, and models are released on this https URL for reproducibility. 

**Abstract (ZH)**: 近期在潜藏扩散模型方面的进展通过利用预训练的VAE在低computational成本下压缩和重构像素数据，已在高保真RGB图像生成中取得了显著成果。然而，透明或分层内容（RGBA图像）的生成仍缺乏深入探索，主要原因是没有大规模基准数据集。在本工作中，我们提出了ALPHA，这是首个通过alpha融合标准背景来适应四通道图像的全面RGBA基准，同时将标准RGB指标应用于四通道图像。我们还提出了ALPHAVAE，这是一种统一的端到端RGBA VAE，它通过引入专门的alpha通道扩展了预训练的RGB VAE。该模型通过结合alpha融合像素重构、块级别保真度、感知一致性以及双KL散度约束进行训练，以确保在RGB和alpha表示中保持潜空间保真度。与之前方法相比，我们的RGBA VAE仅在8K图像上进行训练而非100万张，实现了比LayerDiffuse更高的4.9 dB的PSNR和3.2%的SSIM的重建性能，并且在潜在扩散框架中微调后能够生成更优的透明图像。我们已将代码、数据和模型发布于此链接以保证可重复性。 

---
# ViT-ProtoNet for Few-Shot Image Classification: A Multi-Benchmark Evaluation 

**Title (ZH)**: ViT-ProtoNet在Few-Shot图像分类中的多基准评估 

**Authors**: Abdulvahap Mutlu, Şengül Doğan, Türker Tuncer  

**Link**: [PDF](https://arxiv.org/pdf/2507.09299)  

**Abstract**: The remarkable representational power of Vision Transformers (ViTs) remains underutilized in few-shot image classification. In this work, we introduce ViT-ProtoNet, which integrates a ViT-Small backbone into the Prototypical Network framework. By averaging class conditional token embeddings from a handful of support examples, ViT-ProtoNet constructs robust prototypes that generalize to novel categories under 5-shot settings. We conduct an extensive empirical evaluation on four standard benchmarks: Mini-ImageNet, FC100, CUB-200, and CIFAR-FS, including overlapped support variants to assess robustness. Across all splits, ViT-ProtoNet consistently outperforms CNN-based prototypical counterparts, achieving up to a 3.2\% improvement in 5-shot accuracy and demonstrating superior feature separability in latent space. Furthermore, it outperforms or is competitive with transformer-based competitors using a more lightweight backbone. Comprehensive ablations examine the impact of transformer depth, patch size, and fine-tuning strategy. To foster reproducibility, we release code and pretrained weights. Our results establish ViT-ProtoNet as a powerful, flexible approach for few-shot classification and set a new baseline for transformer-based meta-learners. 

**Abstract (ZH)**: Vision Transformers (ViTs)在少量样本图像分类中的卓越表征能力尚未充分利用。本文介绍了ViT-ProtoNet，将ViT-Small骨干网络集成到原型网络框架中。通过平均少量支持样本的类条件token嵌入，ViT-ProtoNet在5-shot设置下构建了鲁棒的原型，并能够泛化到新类别。我们在四个标准基准数据集Mini-ImageNet、FC100、CUB-200和CIFAR-FS上进行了广泛的经验评估，包括重叠支持集变体以评估鲁棒性。在所有分割中，ViT-ProtoNet一致优于基于CNN的原型模型，5-shot准确率最高提高3.2%，并在潜在空间中显示出更好的特征可分性。此外，它在使用更轻量级骨干网络时，优于或与基于变换器的竞争者相当。全面的消融实验检查了Transformer深度、 patch大小和微调策略的影响。为了促进可重复性，我们发布了代码和预训练权重。我们的结果确立了ViT-ProtoNet作为一种强大的、灵活的少量样本分类方法，并为基于变换器的元学习者设立了新的基准。 

---
# PanoDiff-SR: Synthesizing Dental Panoramic Radiographs using Diffusion and Super-resolution 

**Title (ZH)**: PanoDiff-SR: 使用扩散和超分辨合成功牙全景放射图像 

**Authors**: Sanyam Jain, Bruna Neves de Freitas, Andreas Basse-OConnor, Alexandros Iosifidis, Ruben Pauwels  

**Link**: [PDF](https://arxiv.org/pdf/2507.09227)  

**Abstract**: There has been increasing interest in the generation of high-quality, realistic synthetic medical images in recent years. Such synthetic datasets can mitigate the scarcity of public datasets for artificial intelligence research, and can also be used for educational purposes. In this paper, we propose a combination of diffusion-based generation (PanoDiff) and Super-Resolution (SR) for generating synthetic dental panoramic radiographs (PRs). The former generates a low-resolution (LR) seed of a PR (256 X 128) which is then processed by the SR model to yield a high-resolution (HR) PR of size 1024 X 512. For SR, we propose a state-of-the-art transformer that learns local-global relationships, resulting in sharper edges and textures. Experimental results demonstrate a Frechet inception distance score of 40.69 between 7243 real and synthetic images (in HR). Inception scores were 2.55, 2.30, 2.90 and 2.98 for real HR, synthetic HR, real LR and synthetic LR images, respectively. Among a diverse group of six clinical experts, all evaluating a mixture of 100 synthetic and 100 real PRs in a time-limited observation, the average accuracy in distinguishing real from synthetic images was 68.5% (with 50% corresponding to random guessing). 

**Abstract (ZH)**: 近年来，对生成高质量、逼真的合成医学图像越来越感兴趣。此类合成数据集可以缓解人工智能研究中公开数据集的稀缺问题，也可以用于教育目的。在本文中，我们提出了一种基于扩散生成（PanoDiff）和超分辨率（SR）结合的方法，用于生成合成牙科全景 radiographs（PRs）。前者生成一个低分辨率（LR）的 PR 种子（256 X 128），然后通过 SR 模型处理以产生高分辨率（HR）的 PR（1024 X 512）。对于超分辨率，我们提出了一种最先进的变换器，它可以学习局部-全局关系，从而产生更清晰的边缘和纹理。实验结果表明，在高分辨率下，7243 张真实和合成图像之间的弗雷切尔入inski 距离评分为 40.69。inception 分数分别为 2.55、2.30、2.90 和 2.98 对应真实高分辨率、合成高分辨率、真实低分辨率和合成低分辨率图像。在一个时间有限的观察中，六位临床专家之一对混合了 100 张合成和 100 张真实 PRs 进行评价，平均区分真实图像和合成图像的准确性为 68.5%（其中 50% 对应随机猜测）。 

---
# Automatic Contouring of Spinal Vertebrae on X-Ray using a Novel Sandwich U-Net Architecture 

**Title (ZH)**: 使用新颖的三明治U-Net架构在X光上自动轮廓化脊椎 vertebrae 

**Authors**: Sunil Munthumoduku Krishna Murthy, Kumar Rajamani, Srividya Tirunellai Rajamani, Yupei Li, Qiyang Sun, Bjoern W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2507.09158)  

**Abstract**: In spinal vertebral mobility disease, accurately extracting and contouring vertebrae is essential for assessing mobility impairments and monitoring variations during flexion-extension movements. Precise vertebral contouring plays a crucial role in surgical planning; however, this process is traditionally performed manually by radiologists or surgeons, making it labour-intensive, time-consuming, and prone to human error. In particular, mobility disease analysis requires the individual contouring of each vertebra, which is both tedious and susceptible to inconsistencies. Automated methods provide a more efficient alternative, enabling vertebra identification, segmentation, and contouring with greater accuracy and reduced time consumption. In this study, we propose a novel U-Net variation designed to accurately segment thoracic vertebrae from anteroposterior view on X-Ray images. Our proposed approach, incorporating a ``sandwich" U-Net structure with dual activation functions, achieves a 4.1\% improvement in Dice score compared to the baseline U-Net model, enhancing segmentation accuracy while ensuring reliable vertebral contour extraction. 

**Abstract (ZH)**: 脊椎移动性疾病中，准确提取和勾勒椎体对于评估移动功能障碍和监测屈伸运动中的变化至关重要。精确的椎体勾勒在手术规划中起着关键作用；然而，这一过程通常由放射科医生或外科医生手工完成，导致劳动密集型、耗时且易出错。特别是移动性疾病分析需要对每个椎体进行单独勾勒，既繁琐又容易出现不一致性。自动化方法提供了更高效的替代方案，能够以更高的精度和更少的时间消耗进行椎体识别、分割和勾勒。在本研究中，我们提出了一种新型U-Net变体，旨在从X射线正位图像中准确分割胸椎。我们提出的方法结合了“三明治”U-Net结构和双重激活函数，与基线U-Net模型相比，在Dice分数上实现了4.1%的改进，同时提高了分割精度并确保可靠的椎体轮廓提取。 

---
# Infinite Video Understanding 

**Title (ZH)**: 无限视频理解 

**Authors**: Dell Zhang, Xiangyu Chen, Jixiang Luo, Mengxi Jia, Changzhi Sun, Ruilong Ren, Jingren Liu, Hao Sun, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09068)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have ushered in remarkable progress in video understanding. However, a fundamental challenge persists: effectively processing and comprehending video content that extends beyond minutes or hours. While recent efforts like Video-XL-2 have demonstrated novel architectural solutions for extreme efficiency, and advancements in positional encoding such as HoPE and VideoRoPE++ aim to improve spatio-temporal understanding over extensive contexts, current state-of-the-art models still encounter significant computational and memory constraints when faced with the sheer volume of visual tokens from lengthy sequences. Furthermore, maintaining temporal coherence, tracking complex events, and preserving fine-grained details over extended periods remain formidable hurdles, despite progress in agentic reasoning systems like Deep Video Discovery. This position paper posits that a logical, albeit ambitious, next frontier for multimedia research is Infinite Video Understanding -- the capability for models to continuously process, understand, and reason about video data of arbitrary, potentially never-ending duration. We argue that framing Infinite Video Understanding as a blue-sky research objective provides a vital north star for the multimedia, and the wider AI, research communities, driving innovation in areas such as streaming architectures, persistent memory mechanisms, hierarchical and adaptive representations, event-centric reasoning, and novel evaluation paradigms. Drawing inspiration from recent work on long/ultra-long video understanding and several closely related fields, we outline the core challenges and key research directions towards achieving this transformative capability. 

**Abstract (ZH)**: 大规模语言模型及其多模态扩展的迅速发展推动了视频理解的显著进步：无限视频理解 

---
# Learning Diffusion Models with Flexible Representation Guidance 

**Title (ZH)**: 学习具有灵活表示指导的扩散模型 

**Authors**: Chenyu Wang, Cai Zhou, Sharut Gupta, Zongyu Lin, Stefanie Jegelka, Stephen Bates, Tommi Jaakkola  

**Link**: [PDF](https://arxiv.org/pdf/2507.08980)  

**Abstract**: Diffusion models can be improved with additional guidance towards more effective representations of input. Indeed, prior empirical work has already shown that aligning internal representations of the diffusion model with those of pre-trained models improves generation quality. In this paper, we present a systematic framework for incorporating representation guidance into diffusion models. We provide alternative decompositions of denoising models along with their associated training criteria, where the decompositions determine when and how the auxiliary representations are incorporated. Guided by our theoretical insights, we introduce two new strategies for enhancing representation alignment in diffusion models. First, we pair examples with target representations either derived from themselves or arisen from different synthetic modalities, and subsequently learn a joint model over the multimodal pairs. Second, we design an optimal training curriculum that balances representation learning and data generation. Our experiments across image, protein sequence, and molecule generation tasks demonstrate superior performance as well as accelerated training. In particular, on the class-conditional ImageNet $256\times 256$ benchmark, our guidance results in $23.3$ times faster training than the original SiT-XL as well as four times speedup over the state-of-the-art method REPA. The code is available at this https URL. 

**Abstract (ZH)**: 使用额外的指导提高扩散模型的表示能力：一个系统框架及其应用 

---
# Theory-Informed Improvements to Classifier-Free Guidance for Discrete Diffusion Models 

**Title (ZH)**: 基于理论指导的离散扩散模型无分类器引导改进 

**Authors**: Kevin Rojas, Ye He, Chieh-Hsin Lai, Yuta Takida, Yuki Mitsufuji, Molei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2507.08965)  

**Abstract**: Classifier-Free Guidance (CFG) is a widely used technique for conditional generation and improving sample quality in continuous diffusion models, and recent works have extended it to discrete diffusion. This paper theoretically analyzes CFG in the context of masked discrete diffusion, focusing on the role of guidance schedules. Our analysis shows that high guidance early in sampling (when inputs are heavily masked) harms generation quality, while late-stage guidance has a larger effect. These findings provide a theoretical explanation for empirical observations in recent studies on guidance schedules. The analysis also reveals an imperfection of the current CFG implementations. These implementations can unintentionally cause imbalanced transitions, such as unmasking too rapidly during the early stages of generation, which degrades the quality of the resulting samples. To address this, we draw insight from the analysis and propose a novel classifier-free guidance mechanism empirically applicable to any discrete diffusion. Intuitively, our method smoothens the transport between the data distribution and the initial (masked/uniform) distribution, which results in improved sample quality. Remarkably, our method is achievable via a simple one-line code change. The efficacy of our method is empirically demonstrated with experiments on ImageNet (masked discrete diffusion) and QM9 (uniform discrete diffusion). 

**Abstract (ZH)**: Classifier-Free Guidance (CFG)在掩码离散扩散中的理论分析：指导计划的角色 

---
