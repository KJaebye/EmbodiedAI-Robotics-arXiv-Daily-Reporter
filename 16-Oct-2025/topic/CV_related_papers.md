# PlanarMesh: Building Compact 3D Meshes from LiDAR using Incremental Adaptive Resolution Reconstruction 

**Title (ZH)**: PlanarMesh：使用增量自适应分辨率重建从LiDAR构建紧凑3D网格 

**Authors**: Jiahao Wang, Nived Chebrolu, Yifu Tao, Lintong Zhang, Ayoung Kim, Maurice Fallon  

**Link**: [PDF](https://arxiv.org/pdf/2510.13599)  

**Abstract**: Building an online 3D LiDAR mapping system that produces a detailed surface reconstruction while remaining computationally efficient is a challenging task. In this paper, we present PlanarMesh, a novel incremental, mesh-based LiDAR reconstruction system that adaptively adjusts mesh resolution to achieve compact, detailed reconstructions in real-time. It introduces a new representation, planar-mesh, which combines plane modeling and meshing to capture both large surfaces and detailed geometry. The planar-mesh can be incrementally updated considering both local surface curvature and free-space information from sensor measurements. We employ a multi-threaded architecture with a Bounding Volume Hierarchy (BVH) for efficient data storage and fast search operations, enabling real-time performance. Experimental results show that our method achieves reconstruction accuracy on par with, or exceeding, state-of-the-art techniques-including truncated signed distance functions, occupancy mapping, and voxel-based meshing-while producing smaller output file sizes (10 times smaller than raw input and more than 5 times smaller than mesh-based methods) and maintaining real-time performance (around 2 Hz for a 64-beam sensor). 

**Abstract (ZH)**: 构建一个在线3D LiDAR建图系统，该系统能够在保持高效计算的同时生成详细表面重建是一项具有挑战性的工作。在本文中，我们提出了PlanarMesh，这是一种新颖的增量式网格基LiDAR重建系统，能够适应性调整网格分辨率，以实现实时的紧凑且详细的重建。PlanarMesh引入了一种新的表示方法，即平面网格，该方法结合了平面建模和网格化技术，以捕捉大面积和详细的几何形状。平面网格可以在考虑局部表面曲率和传感器测量中的自由空间信息的同时进行增量式更新。我们采用多线程架构并结合Bounding Volume Hierarchy (BVH) 来高效存储数据和快速执行搜索操作，从而实现实时性能。实验结果表明，我们的方法在重建准确性方面与最先进的技术（包括截断的符号距离函数、占据映射和体素网格化）相当或优于这些技术，同时生成的输出文件大小更小（原始输入的1/10，比基于网格的方法小5倍以上），并保持实时性能（64束传感器大约为2 Hz）。 

---
# Accelerated Feature Detectors for Visual SLAM: A Comparative Study of FPGA vs GPU 

**Title (ZH)**: 基于FPGA与GPU的加速特征检测器对比研究：视觉SLAM领域的探讨 

**Authors**: Ruiqi Ye, Mikel Luján  

**Link**: [PDF](https://arxiv.org/pdf/2510.13546)  

**Abstract**: Feature detection is a common yet time-consuming module in Simultaneous Localization and Mapping (SLAM) implementations, which are increasingly deployed on power-constrained platforms, such as drones. Graphics Processing Units (GPUs) have been a popular accelerator for computer vision in general, and feature detection and SLAM in particular.
On the other hand, System-on-Chips (SoCs) with integrated Field Programmable Gate Array (FPGA) are also widely available. This paper presents the first study of hardware-accelerated feature detectors considering a Visual SLAM (V-SLAM) pipeline. We offer new insights by comparing the best GPU-accelerated FAST, Harris, and SuperPoint implementations against the FPGA-accelerated counterparts on modern SoCs (Nvidia Jetson Orin and AMD Versal).
The evaluation shows that when using a non-learning-based feature detector such as FAST and Harris, their GPU implementations, and the GPU-accelerated V-SLAM can achieve better run-time performance and energy efficiency than the FAST and Harris FPGA implementations as well as the FPGA-accelerated V-SLAM. However, when considering a learning-based detector such as SuperPoint, its FPGA implementation can achieve better run-time performance and energy efficiency (up to 3.1$\times$ and 1.4$\times$ improvements, respectively) than the GPU implementation. The FPGA-accelerated V-SLAM can also achieve comparable run-time performance compared to the GPU-accelerated V-SLAM, with better FPS in 2 out of 5 dataset sequences. When considering the accuracy, the results show that the GPU-accelerated V-SLAM is more accurate than the FPGA-accelerated V-SLAM in general. Last but not least, the use of hardware acceleration for feature detection could further improve the performance of the V-SLAM pipeline by having the global bundle adjustment module invoked less frequently without sacrificing accuracy. 

**Abstract (ZH)**: 硬件加速在视觉SLAM管道中的特征检测研究：基于GPU与FPGA的比较 

---
# Through the Lens of Doubt: Robust and Efficient Uncertainty Estimation for Visual Place Recognition 

**Title (ZH)**: 怀疑之眼：视觉场所识别中的稳健高效不确定性估计 

**Authors**: Emily Miller, Michael Milford, Muhammad Burhan Hafez, SD Ramchurn, Shoaib Ehsan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13464)  

**Abstract**: Visual Place Recognition (VPR) enables robots and autonomous vehicles to identify previously visited locations by matching current observations against a database of known places. However, VPR systems face significant challenges when deployed across varying visual environments, lighting conditions, seasonal changes, and viewpoints changes. Failure-critical VPR applications, such as loop closure detection in simultaneous localization and mapping (SLAM) pipelines, require robust estimation of place matching uncertainty. We propose three training-free uncertainty metrics that estimate prediction confidence by analyzing inherent statistical patterns in similarity scores from any existing VPR method. Similarity Distribution (SD) quantifies match distinctiveness by measuring score separation between candidates; Ratio Spread (RS) evaluates competitive ambiguity among top-scoring locations; and Statistical Uncertainty (SU) is a combination of SD and RS that provides a unified metric that generalizes across datasets and VPR methods without requiring validation data to select the optimal metric. All three metrics operate without additional model training, architectural modifications, or computationally expensive geometric verification. Comprehensive evaluation across nine state-of-the-art VPR methods and six benchmark datasets confirms that our metrics excel at discriminating between correct and incorrect VPR matches, and consistently outperform existing approaches while maintaining negligible computational overhead, making it deployable for real-time robotic applications across varied environmental conditions with improved precision-recall performance. 

**Abstract (ZH)**: 视觉位置识别(VPR)使机器人和自动驾驶车辆能够通过将当前观察与已知地点的数据库进行匹配来识别先前访问的位置。然而，当部署在变化的视觉环境、光照条件、季节变化和视角变化中时，VPR系统面临着显著挑战。对于关键性的VPR应用，如同时定位与 mapping（SLAM）管道中的环路闭合检测，需要 robust 的位置匹配不确定性估计。我们提出了一种无需训练的不确定性度量方法，通过分析任何现有VPR方法的相似性分数中的固有统计模式来估计预测置信度。相似性分布(SD)通过测量候选者的分数分离来量化匹配的独特性；竞争性模糊度比(RS)评估最高分位置之间的竞争模糊度；统计不确定性(SU)是SD和RS的组合，提供了一个统一的度量标准，可以在无需验证数据选择最优度量的情况下适用于不同数据集和VPR方法。所有三个度量标准无需额外的模型训练、架构修改或昂贵的几何验证。在九种最先进的VPR方法和六个基准数据集上的综合评估表明，我们的度量方法在区分正确和错误的VPR匹配方面表现出色，并且在一致性上优于现有方法，同时保持了微乎其微的计算开销，使其适用于具有改进精度召回性能的多样化环境条件下的实时机器人应用。 

---
# SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms 

**Title (ZH)**: SimULi: 基于无中心变换的实时LiDAR和相机仿真 

**Authors**: Haithem Turki, Qi Wu, Xin Kang, Janick Martinez Esturo, Shengyu Huang, Ruilong Li, Zan Gojcic, Riccardo de Lutio  

**Link**: [PDF](https://arxiv.org/pdf/2510.12901)  

**Abstract**: Rigorous testing of autonomous robots, such as self-driving vehicles, is essential to ensure their safety in real-world deployments. This requires building high-fidelity simulators to test scenarios beyond those that can be safely or exhaustively collected in the real-world. Existing neural rendering methods based on NeRF and 3DGS hold promise but suffer from low rendering speeds or can only render pinhole camera models, hindering their suitability to applications that commonly require high-distortion lenses and LiDAR data. Multi-sensor simulation poses additional challenges as existing methods handle cross-sensor inconsistencies by favoring the quality of one modality at the expense of others. To overcome these limitations, we propose SimULi, the first method capable of rendering arbitrary camera models and LiDAR data in real-time. Our method extends 3DGUT, which natively supports complex camera models, with LiDAR support, via an automated tiling strategy for arbitrary spinning LiDAR models and ray-based culling. To address cross-sensor inconsistencies, we design a factorized 3D Gaussian representation and anchoring strategy that reduces mean camera and depth error by up to 40% compared to existing methods. SimULi renders 10-20x faster than ray tracing approaches and 1.5-10x faster than prior rasterization-based work (and handles a wider range of camera models). When evaluated on two widely benchmarked autonomous driving datasets, SimULi matches or exceeds the fidelity of existing state-of-the-art methods across numerous camera and LiDAR metrics. 

**Abstract (ZH)**: 严格测试自主机器人，如自动驾驶车辆，对于确保其在实际部署中的安全性至关重要。这需要构建高保真模拟器来测试超出现实世界中可以安全或耗尽性收集的场景。现有基于NeRF和3DGS的神经渲染方法前景广阔，但存在渲染速度低或只能渲染针孔相机模型的问题，阻碍了它们在常需高失真镜头和LiDAR数据的应用中的适用性。多传感器模拟还提出了额外挑战，现有方法通过优先考虑某种模态的质量，而牺牲其他模态来处理跨传感器不一致性。为克服这些局限性，我们提出了SimULi，这是首款能够实时渲染任意相机模型和LiDAR数据的方法。我们的方法通过自动切片策略扩展了本来就支持复杂相机模型的3DGUT，并通过基于光线裁剪策略为任意旋转的LiDAR模型提供了LiDAR支持。为解决跨传感器不一致性问题，我们设计了一种因子化的3D高斯表示和锚定策略，相比现有方法将平均相机误差和深度误差降低了最多40%。SimULi的渲染速度比光线跟踪方法快10-20倍，比之前的基于光栅化的工作快1.5-10倍（并且能够处理更广泛的相机模型）。当在两个广泛基准测试的自动驾驶数据集上进行评估时，SimULi在多个相机和LiDAR指标上与现有最先进的方法相当或更优。 

---
# An Analytical Framework to Enhance Autonomous Vehicle Perception for Smart Cities 

**Title (ZH)**: 自主车辆感知增强的分析框架以促进智能城市的发展 

**Authors**: Jalal Khan, Manzoor Khan, Sherzod Turaev, Sumbal Malik, Hesham El-Sayed, Farman Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2510.13230)  

**Abstract**: The driving environment perception has a vital role for autonomous driving and nowadays has been actively explored for its realization. The research community and relevant stakeholders necessitate the development of Deep Learning (DL) models and AI-enabled solutions to enhance autonomous vehicles (AVs) for smart mobility. There is a need to develop a model that accurately perceives multiple objects on the road and predicts the driver's perception to control the car's movements. This article proposes a novel utility-based analytical model that enables perception systems of AVs to understand the driving environment. The article consists of modules: acquiring a custom dataset having distinctive objects, i.e., motorcyclists, rickshaws, etc; a DL-based model (YOLOv8s) for object detection; and a module to measure the utility of perception service from the performance values of trained model instances. The perception model is validated based on the object detection task, and its process is benchmarked by state-of-the-art deep learning models' performance metrics from the nuScense dataset. The experimental results show three best-performing YOLOv8s instances based on mAP@0.5 values, i.e., SGD-based (0.832), Adam-based (0.810), and AdamW-based (0.822). However, the AdamW-based model (i.e., car: 0.921, motorcyclist: 0.899, truck: 0.793, etc.) still outperforms the SGD-based model (i.e., car: 0.915, motorcyclist: 0.892, truck: 0.781, etc.) because it has better class-level performance values, confirmed by the proposed perception model. We validate that the proposed function is capable of finding the right perception for AVs. The results above encourage using the proposed perception model to evaluate the utility of learning models and determine the appropriate perception for AVs. 

**Abstract (ZH)**: 自主驾驶环境感知在智能出行中的作用及其基于Deep Learning的新型分析模型研究 

---
# Scaling Vision Transformers for Functional MRI with Flat Maps 

**Title (ZH)**: 用于功能性磁共振成像的平铺图扩展视觉变换器 

**Authors**: Connor Lane, Daniel Z. Kaplan, Tanishq Mathew Abraham, Paul S. Scotti  

**Link**: [PDF](https://arxiv.org/pdf/2510.13768)  

**Abstract**: A key question for adapting modern deep learning architectures to functional MRI (fMRI) is how to represent the data for model input. To bridge the modality gap between fMRI and natural images, we transform the 4D volumetric fMRI data into videos of 2D fMRI activity flat maps. We train Vision Transformers on 2.3K hours of fMRI flat map videos from the Human Connectome Project using the spatiotemporal masked autoencoder (MAE) framework. We observe that masked fMRI modeling performance improves with dataset size according to a strict power scaling law. Downstream classification benchmarks show that our model learns rich representations supporting both fine-grained state decoding across subjects, as well as subject-specific trait decoding across changes in brain state. This work is part of an ongoing open science project to build foundation models for fMRI data. Our code and datasets are available at this https URL. 

**Abstract (ZH)**: 将现代深度学习架构适应功能性磁共振成像（fMRI）的关键问题是如何表示数据以供模型输入。为了弥合fMRI与自然图像之间的模态差距，我们将4D体积fMRI数据转换为2D fMRI活动平面图的视频。我们使用时空掩蔽自编码器（MAE）框架在人类连接组计划的2.3万小时fMRI平面图视频上训练视觉变换器。我们观察到，掩蔽fMRI建模性能随着数据集大小严格遵循幂律扩展。下游分类基准表明，我们的模型学会了支持跨被试的精细状态解码以及跨脑状态变化的被试特定特质解码的丰富表示。这项工作是构建fMRI数据基础模型的持续开源项目的一部分。我们的代码和数据集可在以下链接获得。 

---
# Multi-Scale High-Resolution Logarithmic Grapher Module for Efficient Vision GNNs 

**Title (ZH)**: 多尺度高分辨率对数图模块：用于高效视觉图神经网络的设计 

**Authors**: Mustafa Munir, Alex Zhang, Radu Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13740)  

**Abstract**: Vision graph neural networks (ViG) have demonstrated promise in vision tasks as a competitive alternative to conventional convolutional neural nets (CNN) and transformers (ViTs); however, common graph construction methods, such as k-nearest neighbor (KNN), can be expensive on larger images. While methods such as Sparse Vision Graph Attention (SVGA) have shown promise, SVGA's fixed step scale can lead to over-squashing and missing multiple connections to gain the same information that could be gained from a long-range link. Through this observation, we propose a new graph construction method, Logarithmic Scalable Graph Construction (LSGC) to enhance performance by limiting the number of long-range links. To this end, we propose LogViG, a novel hybrid CNN-GNN model that utilizes LSGC. Furthermore, inspired by the successes of multi-scale and high-resolution architectures, we introduce and apply a high-resolution branch and fuse features between our high-resolution and low-resolution branches for a multi-scale high-resolution Vision GNN network. Extensive experiments show that LogViG beats existing ViG, CNN, and ViT architectures in terms of accuracy, GMACs, and parameters on image classification and semantic segmentation tasks. Our smallest model, Ti-LogViG, achieves an average top-1 accuracy on ImageNet-1K of 79.9% with a standard deviation of 0.2%, 1.7% higher average accuracy than Vision GNN with a 24.3% reduction in parameters and 35.3% reduction in GMACs. Our work shows that leveraging long-range links in graph construction for ViGs through our proposed LSGC can exceed the performance of current state-of-the-art ViGs. Code is available at this https URL. 

**Abstract (ZH)**: 基于对数可扩展图构建的Vision图神经网络（LogViG） 

---
# CanvasMAR: Improving Masked Autoregressive Video Generation With Canvas 

**Title (ZH)**: CanvasMAR: 在Canvas上改进掩蔽自回归视频生成 

**Authors**: Zian Li, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13669)  

**Abstract**: Masked autoregressive models (MAR) have recently emerged as a powerful paradigm for image and video generation, combining the flexibility of masked modeling with the potential of continuous tokenizer. However, video MAR models suffer from two major limitations: the slow-start problem, caused by the lack of a structured global prior at early sampling stages, and error accumulation across the autoregression in both spatial and temporal dimensions. In this work, we propose CanvasMAR, a novel video MAR model that mitigates these issues by introducing a canvas mechanism--a blurred, global prediction of the next frame, used as the starting point for masked generation. The canvas provides global structure early in sampling, enabling faster and more coherent frame synthesis. Furthermore, we introduce compositional classifier-free guidance that jointly enlarges spatial (canvas) and temporal conditioning, and employ noise-based canvas augmentation to enhance robustness. Experiments on the BAIR and Kinetics-600 benchmarks demonstrate that CanvasMAR produces high-quality videos with fewer autoregressive steps. Our approach achieves remarkable performance among autoregressive models on Kinetics-600 dataset and rivals diffusion-based methods. 

**Abstract (ZH)**: 基于遮蔽机制的视频生成模型CanvasMAR：通过画布机制缓解缓慢启动问题和错误累积 

---
# Modeling Cultural Bias in Facial Expression Recognition with Adaptive Agents 

**Title (ZH)**: 基于适应性代理建模面部表情识别中的文化偏见 

**Authors**: David Freire-Obregón, José Salas-Cáceres, Javier Lorenzo-Navarro, Oliverio J. Santana, Daniel Hernández-Sosa, Modesto Castrillón-Santana  

**Link**: [PDF](https://arxiv.org/pdf/2510.13557)  

**Abstract**: Facial expression recognition (FER) must remain robust under both cultural variation and perceptually degraded visual conditions, yet most existing evaluations assume homogeneous data and high-quality imagery. We introduce an agent-based, streaming benchmark that reveals how cross-cultural composition and progressive blurring interact to shape face recognition robustness. Each agent operates in a frozen CLIP feature space with a lightweight residual adapter trained online at sigma=0 and fixed during testing. Agents move and interact on a 5x5 lattice, while the environment provides inputs with sigma-scheduled Gaussian blur. We examine monocultural populations (Western-only, Asian-only) and mixed environments with balanced (5/5) and imbalanced (8/2, 2/8) compositions, as well as different spatial contact structures. Results show clear asymmetric degradation curves between cultural groups: JAFFE (Asian) populations maintain higher performance at low blur but exhibit sharper drops at intermediate stages, whereas KDEF (Western) populations degrade more uniformly. Mixed populations exhibit intermediate patterns, with balanced mixtures mitigating early degradation, but imbalanced settings amplify majority-group weaknesses under high blur. These findings quantify how cultural composition and interaction structure influence the robustness of FER as perceptual conditions deteriorate. 

**Abstract (ZH)**: 基于代理的流媒体基准揭示了跨文化构成和渐进模糊交互如何塑造面部识别鲁棒性 

---
# Semantic Communication Enabled Holographic Video Processing and Transmission 

**Title (ZH)**: 基于语义通信的全息视频处理与传输 

**Authors**: Jingkai Ying, Zhiyuan Qi, Yulong Feng, Zhijin Qin, Zhu Han, Rahim Tafazolli, Yonina C. Eldar  

**Link**: [PDF](https://arxiv.org/pdf/2510.13408)  

**Abstract**: Holographic video communication is considered a paradigm shift in visual communications, becoming increasingly popular for its ability to offer immersive experiences. This article provides an overview of holographic video communication and outlines the requirements of a holographic video communication system. Particularly, following a brief review of semantic com- munication, an architecture for a semantic-enabled holographic video communication system is presented. Key technologies, including semantic sampling, joint semantic-channel coding, and semantic-aware transmission, are designed based on the proposed architecture. Two related use cases are presented to demonstrate the performance gain of the proposed methods. Finally, potential research topics are discussed to pave the way for the realization of semantic-enabled holographic video communications. 

**Abstract (ZH)**: 全息视频通信被视为视觉通信领域的 paradigm shift，因其能提供沉浸式体验而日益受欢迎。本文对全息视频通信进行了综述，并概述了全息视频通信系统的需求。特别是，在简要回顾语义通信后，提出了一个支持语义的全息视频通信系统的架构。基于该架构，设计了关键技术，包括语义采样、联合语义-信道编码和语义感知传输，并展示了两种相关应用场景以证明所提方法的性能增益。最后，讨论了潜在的研究方向以推动语义支持的全息视频通信的实现。 

---
# MimicParts: Part-aware Style Injection for Speech-Driven 3D Motion Generation 

**Title (ZH)**: MimicParts: 部件aware的风格注入方法用于语音驱动的3D运动生成 

**Authors**: Lianlian Liu, YongKang He, Zhaojie Chu, Xiaofen Xing, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13208)  

**Abstract**: Generating stylized 3D human motion from speech signals presents substantial challenges, primarily due to the intricate and fine-grained relationships among speech signals, individual styles, and the corresponding body movements. Current style encoding approaches either oversimplify stylistic diversity or ignore regional motion style differences (e.g., upper vs. lower body), limiting motion realism. Additionally, motion style should dynamically adapt to changes in speech rhythm and emotion, but existing methods often overlook this. To address these issues, we propose MimicParts, a novel framework designed to enhance stylized motion generation based on part-aware style injection and part-aware denoising network. It divides the body into different regions to encode localized motion styles, enabling the model to capture fine-grained regional differences. Furthermore, our part-aware attention block allows rhythm and emotion cues to guide each body region precisely, ensuring that the generated motion aligns with variations in speech rhythm and emotional state. Experimental results show that our method outperforming existing methods showcasing naturalness and expressive 3D human motion sequences. 

**Abstract (ZH)**: 基于部分感知风格注入和部分感知去噪网络的3D人体运动生成方法：MimicParts 

---
# DriveCritic: Towards Context-Aware, Human-Aligned Evaluation for Autonomous Driving with Vision-Language Models 

**Title (ZH)**: DriveCritic: 面向上下文感知和人类价值观对齐的自动驾驶评价方法研究 

**Authors**: Jingyu Song, Zhenxin Li, Shiyi Lan, Xinglong Sun, Nadine Chang, Maying Shen, Joshua Chen, Katherine A. Skinner, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13108)  

**Abstract**: Benchmarking autonomous driving planners to align with human judgment remains a critical challenge, as state-of-the-art metrics like the Extended Predictive Driver Model Score (EPDMS) lack context awareness in nuanced scenarios. To address this, we introduce DriveCritic, a novel framework featuring two key contributions: the DriveCritic dataset, a curated collection of challenging scenarios where context is critical for correct judgment and annotated with pairwise human preferences, and the DriveCritic model, a Vision-Language Model (VLM) based evaluator. Fine-tuned using a two-stage supervised and reinforcement learning pipeline, the DriveCritic model learns to adjudicate between trajectory pairs by integrating visual and symbolic context. Experiments show DriveCritic significantly outperforms existing metrics and baselines in matching human preferences and demonstrates strong context awareness. Overall, our work provides a more reliable, human-aligned foundation to evaluating autonomous driving systems. 

**Abstract (ZH)**: 基于DriveCritic框架的自主驾驶规划器评估基准尚存在关键挑战，现有的度量标准如扩展预测驾驶员模型评分（EPDMS）在细微情境中缺乏背景意识。为此，我们引入了DriveCritic，一种新型框架，包含两项关键贡献：DriveCritic数据集，一个精心挑选的包含关键背景信息的挑战性场景集合，并标注了两两的人类偏好；以及DriveCritic模型，一种基于视觉语言模型（VLM）的评估器。通过两阶段监督学习和强化学习微调，DriveCritic模型学习通过整合视觉和符号背景来裁定轨迹对。实验表明，DriveCritic显著优于现有度量标准和基线，在匹配人类偏好方面表现更佳，并显示出强大的背景意识。总体而言，我们的工作为评价自主驾驶系统提供了一个更可靠、更符合人类认知的基准。 

---
# True Self-Supervised Novel View Synthesis is Transferable 

**Title (ZH)**: 真正的自我监督新颖视图合成是可迁移的 

**Authors**: Thomas W. Mitchel, Hyunwoo Ryu, Vincent Sitzmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.13063)  

**Abstract**: In this paper, we identify that the key criterion for determining whether a model is truly capable of novel view synthesis (NVS) is transferability: Whether any pose representation extracted from one video sequence can be used to re-render the same camera trajectory in another. We analyze prior work on self-supervised NVS and find that their predicted poses do not transfer: The same set of poses lead to different camera trajectories in different 3D scenes. Here, we present XFactor, the first geometry-free self-supervised model capable of true NVS. XFactor combines pair-wise pose estimation with a simple augmentation scheme of the inputs and outputs that jointly enables disentangling camera pose from scene content and facilitates geometric reasoning. Remarkably, we show that XFactor achieves transferability with unconstrained latent pose variables, without any 3D inductive biases or concepts from multi-view geometry -- such as an explicit parameterization of poses as elements of SE(3). We introduce a new metric to quantify transferability, and through large-scale experiments, we demonstrate that XFactor significantly outperforms prior pose-free NVS transformers, and show that latent poses are highly correlated with real-world poses through probing experiments. 

**Abstract (ZH)**: 在这项工作中，我们确定模型是否真正具备新颖视图合成（NVS）能力的关键标准是可迁移性：任何从一个视频序列提取的姿势表示是否可以在另一个序列中用于重新渲染相同的摄像机轨迹。我们分析了先有的自监督NVS工作，发现它们预测的姿势不具备可迁移性：相同的姿势集在不同的3D场景中生成不同的摄像机轨迹。在此，我们提出了XFactor，这是首个无需几何信息就能实现真正确模型NVS的自监督模型。XFactor结合了一对一姿势估计和简单的输入输出增强方案，同时实现了从摄像机姿态中解耦场景内容，促进了几何推理。令人惊讶的是，我们展示XFactor仅通过无约束的潜在姿态变量即可实现可迁移性，无需任何3D归纳偏置或来自多视图几何的概念，如明确将姿态表示为SE(3)的元素。我们引入了一个新的度量标准来量化可迁移性，并通过大规模实验展示XFactor显著优于先前的无姿态NVS变压器，通过探针实验展示了潜在姿态与真实世界姿态的高度相关性。 

---
# SceneAdapt: Scene-aware Adaptation of Human Motion Diffusion 

**Title (ZH)**: SceneAdapt: 基于场景的 humanoid 运动扩散适配 

**Authors**: Jungbin Cho, Minsu Kim, Jisoo Kim, Ce Zheng, Laszlo A. Jeni, Ming-Hsuan Yang, Youngjae Yu, Seonjoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13044)  

**Abstract**: Human motion is inherently diverse and semantically rich, while also shaped by the surrounding scene. However, existing motion generation approaches address either motion semantics or scene-awareness in isolation, since constructing large-scale datasets with both rich text--motion coverage and precise scene interactions is extremely challenging. In this work, we introduce SceneAdapt, a framework that injects scene awareness into text-conditioned motion models by leveraging disjoint scene--motion and text--motion datasets through two adaptation stages: inbetweening and scene-aware inbetweening. The key idea is to use motion inbetweening, learnable without text, as a proxy task to bridge two distinct datasets and thereby inject scene-awareness to text-to-motion models. In the first stage, we introduce keyframing layers that modulate motion latents for inbetweening while preserving the latent manifold. In the second stage, we add a scene-conditioning layer that injects scene geometry by adaptively querying local context through cross-attention. Experimental results show that SceneAdapt effectively injects scene awareness into text-to-motion models, and we further analyze the mechanisms through which this awareness emerges. Code and models will be released. 

**Abstract (ZH)**: SceneAdapt：通过场景适配注入语境意识的文本条件运动生成框架 

---
# SeqBench: Benchmarking Sequential Narrative Generation in Text-to-Video Models 

**Title (ZH)**: SeqBench: Text-to-Video模型中序列叙事生成的基准测试 

**Authors**: Zhengxu Tang, Zizheng Wang, Luning Wang, Zitao Shuai, Chenhao Zhang, Siyu Qian, Yirui Wu, Bohao Wang, Haosong Rao, Zhenyu Yang, Chenwei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13042)  

**Abstract**: Text-to-video (T2V) generation models have made significant progress in creating visually appealing videos. However, they struggle with generating coherent sequential narratives that require logical progression through multiple events. Existing T2V benchmarks primarily focus on visual quality metrics but fail to evaluate narrative coherence over extended sequences. To bridge this gap, we present SeqBench, a comprehensive benchmark for evaluating sequential narrative coherence in T2V generation. SeqBench includes a carefully designed dataset of 320 prompts spanning various narrative complexities, with 2,560 human-annotated videos generated from 8 state-of-the-art T2V models. Additionally, we design a Dynamic Temporal Graphs (DTG)-based automatic evaluation metric, which can efficiently capture long-range dependencies and temporal ordering while maintaining computational efficiency. Our DTG-based metric demonstrates a strong correlation with human annotations. Through systematic evaluation using SeqBench, we reveal critical limitations in current T2V models: failure to maintain consistent object states across multi-action sequences, physically implausible results in multi-object scenarios, and difficulties in preserving realistic timing and ordering relationships between sequential actions. SeqBench provides the first systematic framework for evaluating narrative coherence in T2V generation and offers concrete insights for improving sequential reasoning capabilities in future models. Please refer to this https URL for more details. 

**Abstract (ZH)**: 基于文本生成视频（Text-to-video，T2V）模型在生成视觉吸引力的视频方面取得了显著进步。然而，它们在生成需要通过多个事件进行逻辑进展的连贯序列叙事方面存在问题。现有的T2V基准主要关注视觉质量指标，但未能评估扩展序列中的叙述连贯性。为弥补这一不足，我们提出了SeqBench，一个用于评估T2V生成中序列叙述连贯性的综合性基准。SeqBench包含一个精心设计的数据集，涵盖各种叙事复杂性，共计320个提示，生成了2,560个人工标注的视频，来自8个最先进的T2V模型。此外，我们设计了一种基于动态时间图（Dynamic Temporal Graphs，DTG）的自动评估指标，该指标可以高效地捕捉长程依赖性和时间顺序关系，同时保持计算效率。我们的基于DTG的指标与人工标注具有很强的相关性。通过使用SeqBench进行系统的评估，我们揭示了当前T2V模型的关键局限性：在多动作序列中保持对象状态一致性的失败、在多对象场景中的物理不合理结果以及在保持顺序动作之间现实的时间和顺序关系方面的困难。SeqBench提供了第一个系统框架来评估T2V生成中的叙述连贯性，并为提高未来模型的序列推理能力提供了具体的见解。请参阅此链接以获取更多详细信息。 

---
