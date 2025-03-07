# ViT-VS: On the Applicability of Pretrained Vision Transformer Features for Generalizable Visual Servoing 

**Title (ZH)**: ViT-VS：预训练视觉变换器特征在通用视觉 servoing 中的应用探索 

**Authors**: Alessandro Scherl, Stefan Thalhammer, Bernhard Neuberger, Wilfried Wöber, José Gracía-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2503.04545)  

**Abstract**: Visual servoing enables robots to precisely position their end-effector relative to a target object. While classical methods rely on hand-crafted features and thus are universally applicable without task-specific training, they often struggle with occlusions and environmental variations, whereas learning-based approaches improve robustness but typically require extensive training. We present a visual servoing approach that leverages pretrained vision transformers for semantic feature extraction, combining the advantages of both paradigms while also being able to generalize beyond the provided sample. Our approach achieves full convergence in unperturbed scenarios and surpasses classical image-based visual servoing by up to 31.2\% relative improvement in perturbed scenarios. Even the convergence rates of learning-based methods are matched despite requiring no task- or object-specific training. Real-world evaluations confirm robust performance in end-effector positioning, industrial box manipulation, and grasping of unseen objects using only a reference from the same category. Our code and simulation environment are available at: this https URL 

**Abstract (ZH)**: 视觉伺服使机器人能够精确定位其末端执行器相对于目标物体的位置。虽然经典的视觉伺服方法依赖于手动设计的特征，因此不需要针对具体任务进行训练即可广泛适用，但它们往往难以应对遮挡和环境变化，而基于学习的方法则提高了鲁棒性，但通常需要大量的训练。我们提出了一种视觉伺服方法，利用预训练的视觉变换器进行语义特征提取，结合了两种范式的优点，并且能够在提供的样本之外泛化。在这种方法中，可以在未受干扰的场景下实现完全收敛，并在受干扰场景下相对于基于图像的经典视觉伺服方法实现了高达31.2%的相对改进。即使在不需要针对特定任务或物体进行训练的情况下，基于学习的方法的收敛速率也得到了匹配。实验证明，该方法在末端执行器定位、工业箱体操作以及从未见过的物体抓取方面表现出稳健的性能，并仅需同一类别的参考图像。我们的代码和模拟环境可在以下链接获取：this https URL。 

---
# EvidMTL: Evidential Multi-Task Learning for Uncertainty-Aware Semantic Surface Mapping from Monocular RGB Images 

**Title (ZH)**: EvidMTL: 证据多任务学习在单目RGB图像中带不确定性意识语义表面映射中的应用 

**Authors**: Rohit Menon, Nils Dengler, Sicong Pan, Gokul Krishna Chenchani, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04441)  

**Abstract**: For scene understanding in unstructured environments, an accurate and uncertainty-aware metric-semantic mapping is required to enable informed action selection by autonomous this http URL mapping methods often suffer from overconfident semantic predictions, and sparse and noisy depth sensing, leading to inconsistent map representations. In this paper, we therefore introduce EvidMTL, a multi-task learning framework that uses evidential heads for depth estimation and semantic segmentation, enabling uncertainty-aware inference from monocular RGB images. To enable uncertainty-calibrated evidential multi-task learning, we propose a novel evidential depth loss function that jointly optimizes the belief strength of the depth prediction in conjunction with evidential segmentation loss. Building on this, we present EvidKimera, an uncertainty-aware semantic surface mapping framework, which uses evidential depth and semantics prediction for improved 3D metric-semantic consistency. We train and evaluate EvidMTL on the NYUDepthV2 and assess its zero-shot performance on ScanNetV2, demonstrating superior uncertainty estimation compared to conventional approaches while maintaining comparable depth estimation and semantic segmentation. In zero-shot mapping tests on ScanNetV2, EvidKimera outperforms Kimera in semantic surface mapping accuracy and consistency, highlighting the benefits of uncertainty-aware mapping and underscoring its potential for real-world robotic applications. 

**Abstract (ZH)**: 面向非结构化环境的场景理解需要一种准确且aware不确定性的度量语义映射，以支持自主系统进行有根据的动作选择。现有的映射方法往往受到自信心过强的语义预测和稀疏的噪声深度传感的影响，导致地图表示不一致。因此，本文提出了一种使用证据头部进行深度估计和语义分割的多任务学习框架EvidMTL，使从单目RGB图像中进行aware不确定性的推理成为可能。为实现校准不确定性的证据多任务学习，我们提出了一种新颖的证据深度损失函数，该函数可以联合优化深度预测的信念强度以及语义分割损失。在此基础上，我们介绍了EvidKimera，这是一种aware不确定性的语义表面映射框架，使用证据深度和语义预测以提高三维度量语义一致性。我们在NYUDepthV2上训练并评估了EvidMTL，并在ScanNetV2上评估其零样本性能，其不确定性的估计优于传统方法，同时保持了类似深度估计和语义分割的性能。在ScanNetV2上的零样本映射测试中，EvidKimera在语义表面映射的准确性和一致性方面优于Kimera，强调了aware不确定性映射的优势，并突显了其在实际机器人应用中的潜力。 

---
# Image-Based Relocalization and Alignment for Long-Term Monitoring of Dynamic Underwater Environments 

**Title (ZH)**: 基于图像的再定位与对准在动态水下环境长期监测中的应用 

**Authors**: Beverley Gorry, Tobias Fischer, Michael Milford, Alejandro Fontan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04096)  

**Abstract**: Effective monitoring of underwater ecosystems is crucial for tracking environmental changes, guiding conservation efforts, and ensuring long-term ecosystem health. However, automating underwater ecosystem management with robotic platforms remains challenging due to the complexities of underwater imagery, which pose significant difficulties for traditional visual localization methods. We propose an integrated pipeline that combines Visual Place Recognition (VPR), feature matching, and image segmentation on video-derived images. This method enables robust identification of revisited areas, estimation of rigid transformations, and downstream analysis of ecosystem changes. Furthermore, we introduce the SQUIDLE+ VPR Benchmark-the first large-scale underwater VPR benchmark designed to leverage an extensive collection of unstructured data from multiple robotic platforms, spanning time intervals from days to years. The dataset encompasses diverse trajectories, arbitrary overlap and diverse seafloor types captured under varying environmental conditions, including differences in depth, lighting, and turbidity. Our code is available at: this https URL 

**Abstract (ZH)**: 有效的水下生态系统监测对于追踪环境变化、指导保护努力并确保长期生态系统健康至关重要。然而，由于水下图像的复杂性给传统视觉定位方法带来了巨大挑战，使用机器人平台自动管理水下生态系统仍然具有挑战性。我们提出了一种结合视觉地方识别（VPR）、特征匹配和图像分割的集成管道。该方法能够稳健地识别 revisit 区域、估计刚性变换，并进行生态系统变化的下游分析。此外，我们引入了 SQUIDLE+ VPR 基准——首个利用多种机器人平台多年数据的大型水下 VPR 基准，涵盖从几天到几年的不同时段。数据集包括在不同环境条件下（包括深度、光照和浑浊度差异）捕捉到的多样轨迹和任意重叠的海床类型。我们的代码可在以下链接获取：this https URL。 

---
# LensDFF: Language-enhanced Sparse Feature Distillation for Efficient Few-Shot Dexterous Manipulation 

**Title (ZH)**: LensDFF: 语言增强的稀疏特征蒸馏用于高效的少样本灵巧 manipulation 

**Authors**: Qian Feng, David S. Martinez Lema, Jianxiang Feng, Zhaopeng Chen, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.03890)  

**Abstract**: Learning dexterous manipulation from few-shot demonstrations is a significant yet challenging problem for advanced, human-like robotic systems. Dense distilled feature fields have addressed this challenge by distilling rich semantic features from 2D visual foundation models into the 3D domain. However, their reliance on neural rendering models such as Neural Radiance Fields (NeRF) or Gaussian Splatting results in high computational costs. In contrast, previous approaches based on sparse feature fields either suffer from inefficiencies due to multi-view dependencies and extensive training or lack sufficient grasp dexterity. To overcome these limitations, we propose Language-ENhanced Sparse Distilled Feature Field (LensDFF), which efficiently distills view-consistent 2D features onto 3D points using our novel language-enhanced feature fusion strategy, thereby enabling single-view few-shot generalization. Based on LensDFF, we further introduce a few-shot dexterous manipulation framework that integrates grasp primitives into the demonstrations to generate stable and highly dexterous grasps. Moreover, we present a real2sim grasp evaluation pipeline for efficient grasp assessment and hyperparameter tuning. Through extensive simulation experiments based on the real2sim pipeline and real-world experiments, our approach achieves competitive grasping performance, outperforming state-of-the-art approaches. 

**Abstract (ZH)**: 从少量示范中学习灵巧操作是先进、类人机器人系统面临的一个重要而具有挑战性的问题。密集提炼特征场通过将丰富的语义特征从2D视觉基础模型提炼到3D领域，解决了这一挑战。然而，它们依赖于神经渲染模型如神经辐射场（NeRF）或高斯点积带来的高计算成本。相比之下，基于稀疏特征场的先前方法要么由于多视图依赖性和广泛的训练而导致效率低下，要么缺乏足够的抓取灵巧性。为克服这些限制，我们提出了语言增强稀疏提炼特征场（LensDFF），该方法使用我们新颖的语言增强特征融合策略，高效地将一致的2D特征投射到3D点上，从而实现单视图少量示范泛化。基于LensDFF，我们进一步提出了一种整合抓取原始操作的少量示范灵巧操作框架，以生成稳定且高度灵巧的抓取。此外，我们提出了一个高效的抓取评估和超参数调整的实2仿抓取评估管道。通过基于实2仿管道和真实世界实验的广泛模拟实验，我们的方法实现了竞争性的抓取性能，并且在某些方面优于现有最佳方法。 

---
# Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation 

**Title (ZH)**: Floxels: 快速无监督体素基场景流估计 

**Authors**: David T. Hoffmann, Syed Haseeb Raza, Hanqiu Jiang, Denis Tananaev, Steffen Klingenhoefer, Martin Meinke  

**Link**: [PDF](https://arxiv.org/pdf/2503.04718)  

**Abstract**: Scene flow estimation is a foundational task for many robotic applications, including robust dynamic object detection, automatic labeling, and sensor synchronization. Two types of approaches to the problem have evolved: 1) Supervised and 2) optimization-based methods. Supervised methods are fast during inference and achieve high-quality results, however, they are limited by the need for large amounts of labeled training data and are susceptible to domain gaps. In contrast, unsupervised test-time optimization methods do not face the problem of domain gaps but usually suffer from substantial runtime, exhibit artifacts, or fail to converge to the right solution. In this work, we mitigate several limitations of existing optimization-based methods. To this end, we 1) introduce a simple voxel grid-based model that improves over the standard MLP-based formulation in multiple dimensions and 2) introduce a new multiframe loss formulation. 3) We combine both contributions in our new method, termed Floxels. On the Argoverse 2 benchmark, Floxels is surpassed only by EulerFlow among unsupervised methods while achieving comparable performance at a fraction of the computational cost. Floxels achieves a massive speedup of more than ~60 - 140x over EulerFlow, reducing the runtime from a day to 10 minutes per sequence. Over the faster but low-quality baseline, NSFP, Floxels achieves a speedup of ~14x. 

**Abstract (ZH)**: 基于场景流估计的 rob 领域应用基础任务，包括稳健的动力学对象检测、自动标注和传感器同步。该问题演化出两种方法：1）监督学习方法和2）优化基方法。监督学习方法在推理时速度快，能够达到高质量的结果，但需要大量标注训练数据，并且容易受到领域差异的影响。相比之下，无监督的测试时优化方法不受领域差异问题的困扰，但通常运行时耗时较长、产生伪影或未能收敛到正确的解。在此项工作中，我们减轻了现有优化基方法的若干局限性。为此，我们1）引入一种基于简单体素格网的模型，该模型在多个维度上改进了标准的基于MLP的表述形式；2）引入一种新的多帧损失表述形式；3）将上述两个贡献结合在一起，提出了一种称为Floxeles的新方法。在Argoverse 2基准测试中，Floxeles在无监督方法中仅次于EulerFlow，而在计算成本仅为后者的一小部分的情况下，实现了可比的性能。Floxeles相比EulerFlow实现了超过60-140倍的加速，将运行时间从一天缩短到每个序列10分钟。与更快但质量较低的基线NSFP相比，Floxeles实现了约14倍的加速。 

---
# Omnidirectional Multi-Object Tracking 

**Title (ZH)**: 全方位多目标跟踪 

**Authors**: Kai Luo, Hao Shi, Sheng Wu, Fei Teng, Mengfei Duan, Chang Huang, Yuhang Wang, Kaiwei Wang, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04565)  

**Abstract**: Panoramic imagery, with its 360° field of view, offers comprehensive information to support Multi-Object Tracking (MOT) in capturing spatial and temporal relationships of surrounding objects. However, most MOT algorithms are tailored for pinhole images with limited views, impairing their effectiveness in panoramic settings. Additionally, panoramic image distortions, such as resolution loss, geometric deformation, and uneven lighting, hinder direct adaptation of existing MOT methods, leading to significant performance degradation. To address these challenges, we propose OmniTrack, an omnidirectional MOT framework that incorporates Tracklet Management to introduce temporal cues, FlexiTrack Instances for object localization and association, and the CircularStatE Module to alleviate image and geometric distortions. This integration enables tracking in large field-of-view scenarios, even under rapid sensor motion. To mitigate the lack of panoramic MOT datasets, we introduce the QuadTrack dataset--a comprehensive panoramic dataset collected by a quadruped robot, featuring diverse challenges such as wide fields of view, intense motion, and complex environments. Extensive experiments on the public JRDB dataset and the newly introduced QuadTrack benchmark demonstrate the state-of-the-art performance of the proposed framework. OmniTrack achieves a HOTA score of 26.92% on JRDB, representing an improvement of 3.43%, and further achieves 23.45% on QuadTrack, surpassing the baseline by 6.81%. The dataset and code will be made publicly available at this https URL. 

**Abstract (ZH)**: 全景图像因其360°的视野，提供了支持多目标跟踪（MOT）所需的全面信息，用于捕捉周围对象的空间和时间关系。然而，大多数MOT算法都是针对具有有限视角的针孔图像定制的，在全景设置中的效果不佳。此外，全景图像失真，如分辨率损失、几何变形和不均匀光照，阻碍了现有MOT方法的直接应用，导致性能显著下降。为了解决这些挑战，我们提出了 OmniTrack，这是一种结合了 Tracklet 管理引入时间线索、FlexiTrack 实例进行目标定位和关联以及 CircularStatE 模块以缓解图像和几何失真的全方位MOT框架。这种集成使得即使在传感器快速移动的情况下也能在大视野场景中进行跟踪。为了解决全景MOT数据集缺乏的问题，我们引入了QuadTrack数据集——由四足机器人收集的全面全景数据集，包含广泛的挑战，如宽广的视野、剧烈的运动和复杂的环境。在公共JRDB数据集和新引入的QuadTrack基准上的广泛实验展示了所提出框架的最先进的性能。OmniTrack在JRDB上的HOTA得分为26.92%，相比基线提高了3.43%，进一步在QuadTrack上达到23.45%，超过了基线6.81%。数据集和代码将在以下网址公开：this https URL。 

---
# ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images 

**Title (ZH)**: ForestLPR：关注多BEV密度图像的森林LiDAR场所识别 

**Authors**: Yanqing Shen, Turcan Tuna, Marco Hutter, Cesar Cadena, Nanning Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.04475)  

**Abstract**: Place recognition is essential to maintain global consistency in large-scale localization systems. While research in urban environments has progressed significantly using LiDARs or cameras, applications in natural forest-like environments remain largely under-explored. Furthermore, forests present particular challenges due to high self-similarity and substantial variations in vegetation growth over time. In this work, we propose a robust LiDAR-based place recognition method for natural forests, ForestLPR. We hypothesize that a set of cross-sectional images of the forest's geometry at different heights contains the information needed to recognize revisiting a place. The cross-sectional images are represented by \ac{bev} density images of horizontal slices of the point cloud at different heights. Our approach utilizes a visual transformer as the shared backbone to produce sets of local descriptors and introduces a multi-BEV interaction module to attend to information at different heights adaptively. It is followed by an aggregation layer that produces a rotation-invariant place descriptor. We evaluated the efficacy of our method extensively on real-world data from public benchmarks as well as robotic datasets and compared it against the state-of-the-art (SOTA) methods. The results indicate that ForestLPR has consistently good performance on all evaluations and achieves an average increase of 7.38\% and 9.11\% on Recall@1 over the closest competitor on intra-sequence loop closure detection and inter-sequence re-localization, respectively, validating our hypothesis 

**Abstract (ZH)**: 基于LiDAR的自然森林场所识别方法ForestLPR 

---
# Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting 

**Title (ZH)**: 手术器械渲染：基于高斯点云的可控逼真重建 

**Authors**: Shuojue Yang, Zijian Wu, Mingxuan Hong, Qian Li, Daiyun Shen, Septimiu E. Salcudean, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.04082)  

**Abstract**: Real2Sim is becoming increasingly important with the rapid development of surgical artificial intelligence (AI) and autonomy. In this work, we propose a novel Real2Sim methodology, \textit{Instrument-Splatting}, that leverages 3D Gaussian Splatting to provide fully controllable 3D reconstruction of surgical instruments from monocular surgical videos. To maintain both high visual fidelity and manipulability, we introduce a geometry pre-training to bind Gaussian point clouds on part mesh with accurate geometric priors and define a forward kinematics to control the Gaussians as flexible as real instruments. Afterward, to handle unposed videos, we design a novel instrument pose tracking method leveraging semantics-embedded Gaussians to robustly refine per-frame instrument poses and joint states in a render-and-compare manner, which allows our instrument Gaussian to accurately learn textures and reach photorealistic rendering. We validated our method on 2 publicly released surgical videos and 4 videos collected on ex vivo tissues and green screens. Quantitative and qualitative evaluations demonstrate the effectiveness and superiority of the proposed method. 

**Abstract (ZH)**: Real2Sim 方法在单目手术视频中的手术器械三维重建：Instrument-Splatting 

---
# COARSE: Collaborative Pseudo-Labeling with Coarse Real Labels for Off-Road Semantic Segmentation 

**Title (ZH)**: COARSE: 基于粗略真实标签的协作伪标签生成方法在离路语义分割中的应用 

**Authors**: Aurelio Noca, Xianmei Lei, Jonathan Becktor, Jeffrey Edlund, Anna Sabel, Patrick Spieler, Curtis Padgett, Alexandre Alahi, Deegan Atha  

**Link**: [PDF](https://arxiv.org/pdf/2503.03947)  

**Abstract**: Autonomous off-road navigation faces challenges due to diverse, unstructured environments, requiring robust perception with both geometric and semantic understanding. However, scarce densely labeled semantic data limits generalization across domains. Simulated data helps, but introduces domain adaptation issues. We propose COARSE, a semi-supervised domain adaptation framework for off-road semantic segmentation, leveraging sparse, coarse in-domain labels and densely labeled out-of-domain data. Using pretrained vision transformers, we bridge domain gaps with complementary pixel-level and patch-level decoders, enhanced by a collaborative pseudo-labeling strategy on unlabeled data. Evaluations on RUGD and Rellis-3D datasets show significant improvements of 9.7\% and 8.4\% respectively, versus only using coarse data. Tests on real-world off-road vehicle data in a multi-biome setting further demonstrate COARSE's applicability. 

**Abstract (ZH)**: 自主离路面导航由于面对多样且结构不规则的环境而面临挑战，需要兼具几何理解和语义理解的鲁棒感知。然而，稀缺的密集标注语义数据限制了其跨领域的泛化能力。模拟数据有所帮助，但引入了领域适应问题。我们提出COARSE，一种结合领域内稀疏粗略标注和领域外密集标注数据的半监督领域适应框架，利用预训练的视觉变换器，通过互补的像素级和块级解码器，并结合协作的伪标签策略，跨越领域鸿沟。在RUGD和Rellis-3D数据集上的评估显示，相比于仅使用粗略标注数据，准确率分别提高了9.7%和8.4%。在多种生物群落的真实世界离路面车辆数据上进一步测试证明了COARSE的适用性。 

---
# The Best of Both Worlds: Integrating Language Models and Diffusion Models for Video Generation 

**Title (ZH)**: 兼收并蓄：将语言模型与扩散模型结合用于视频生成 

**Authors**: Aoxiong Yin, Kai Shen, Yichong Leng, Xu Tan, Xinyu Zhou, Juncheng Li, Siliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04606)  

**Abstract**: Recent advancements in text-to-video (T2V) generation have been driven by two competing paradigms: autoregressive language models and diffusion models. However, each paradigm has intrinsic limitations: language models struggle with visual quality and error accumulation, while diffusion models lack semantic understanding and causal modeling. In this work, we propose LanDiff, a hybrid framework that synergizes the strengths of both paradigms through coarse-to-fine generation. Our architecture introduces three key innovations: (1) a semantic tokenizer that compresses 3D visual features into compact 1D discrete representations through efficient semantic compression, achieving a $\sim$14,000$\times$ compression ratio; (2) a language model that generates semantic tokens with high-level semantic relationships; (3) a streaming diffusion model that refines coarse semantics into high-fidelity videos. Experiments show that LanDiff, a 5B model, achieves a score of 85.43 on the VBench T2V benchmark, surpassing the state-of-the-art open-source models Hunyuan Video (13B) and other commercial models such as Sora, Keling, and Hailuo. Furthermore, our model also achieves state-of-the-art performance in long video generation, surpassing other open-source models in this field. Our demo can be viewed at this https URL. 

**Abstract (ZH)**: Recent advancements in text-to-video (T2V) generation have been driven by two competing paradigms: autoregressive language models and diffusion models. However, each paradigm has intrinsic limitations: language models struggle with visual quality and error accumulation, while diffusion models lack semantic understanding and causal modeling. In this work, we propose LanDiff, a hybrid framework that synergizes the strengths of both paradigms through coarse-to-fine generation. 

---
# ReynoldsFlow: Exquisite Flow Estimation via Reynolds Transport Theorem 

**Title (ZH)**: ReynoldsFlow: 通过雷诺输运定理实现精细流场估计 

**Authors**: Yu-Hsi Chen, Chin-Tien Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04500)  

**Abstract**: Optical flow is a fundamental technique for motion estimation, widely applied in video stabilization, interpolation, and object tracking. Recent advancements in artificial intelligence (AI) have enabled deep learning models to leverage optical flow as an important feature for motion analysis. However, traditional optical flow methods rely on restrictive assumptions, such as brightness constancy and slow motion constraints, limiting their effectiveness in complex scenes. Deep learning-based approaches require extensive training on large domain-specific datasets, making them computationally demanding. Furthermore, optical flow is typically visualized in the HSV color space, which introduces nonlinear distortions when converted to RGB and is highly sensitive to noise, degrading motion representation accuracy. These limitations inherently constrain the performance of downstream models, potentially hindering object tracking and motion analysis tasks. To address these challenges, we propose Reynolds flow, a novel training-free flow estimation inspired by the Reynolds transport theorem, offering a principled approach to modeling complex motion dynamics. Beyond the conventional HSV-based visualization, denoted ReynoldsFlow, we introduce an alternative representation, ReynoldsFlow+, designed to improve flow visualization. We evaluate ReynoldsFlow and ReynoldsFlow+ across three video-based benchmarks: tiny object detection on UAVDB, infrared object detection on Anti-UAV, and pose estimation on GolfDB. Experimental results demonstrate that networks trained with ReynoldsFlow+ achieve state-of-the-art (SOTA) performance, exhibiting improved robustness and efficiency across all tasks. 

**Abstract (ZH)**: 基于瑞利传输定理的无训练流场估计：Reynolds流 

---
# How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects 

**Title (ZH)**: 如何移动你的龙：大词汇量对象的文本到运动合成 

**Authors**: Wonkwang Lee, Jongwon Jeong, Taehong Moon, Hyeon-Jong Kim, Jaehyeon Kim, Gunhee Kim, Byeong-Uk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.04257)  

**Abstract**: Motion synthesis for diverse object categories holds great potential for 3D content creation but remains underexplored due to two key challenges: (1) the lack of comprehensive motion datasets that include a wide range of high-quality motions and annotations, and (2) the absence of methods capable of handling heterogeneous skeletal templates from diverse objects. To address these challenges, we contribute the following: First, we augment the Truebones Zoo dataset, a high-quality animal motion dataset covering over 70 species, by annotating it with detailed text descriptions, making it suitable for text-based motion synthesis. Second, we introduce rig augmentation techniques that generate diverse motion data while preserving consistent dynamics, enabling models to adapt to various skeletal configurations. Finally, we redesign existing motion diffusion models to dynamically adapt to arbitrary skeletal templates, enabling motion synthesis for a diverse range of objects with varying structures. Experiments show that our method learns to generate high-fidelity motions from textual descriptions for diverse and even unseen objects, setting a strong foundation for motion synthesis across diverse object categories and skeletal templates. Qualitative results are available on this link: this http URL 

**Abstract (ZH)**: 多样物体类别的运动合成在3D内容创作中具有巨大潜力，但由于两个关键挑战而鲜有探索：（1）缺乏包含广泛高质量运动和注释的综合运动数据集，（2）缺乏能够处理来自多样化物体的异构骨骼模板的方法。为应对这些挑战，我们做出了以下贡献：首先，我们通过添加详细文本描述来扩展Truebones Zoo数据集，这是一个涵盖超过70种物种的高质量动物运动数据集，使其适合基于文本的运动合成。其次，我们引入了 rig 增强技术，生成多样化运动数据同时保持一致的动力学，使模型能够适应各种骨骼配置。最后，我们重新设计现有的运动扩散模型，使其能够动态适应任意骨骼模板，从而实现具有不同结构的多样化物体的运动合成。实验表明，我们的方法能够从文本描述中生成高保真度的运动，适用于多样甚至未见过的物体，为多样物体类别和骨骼模板的运动合成奠定了坚实的基础。定性结果可在以下链接查看：this http URL 

---
# CA-W3D: Leveraging Context-Aware Knowledge for Weakly Supervised Monocular 3D Detection 

**Title (ZH)**: CA-W3D: 利用上下文感知知识进行弱监督单目3D检测 

**Authors**: Chupeng Liu, Runkai Zhao, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.04154)  

**Abstract**: Weakly supervised monocular 3D detection, while less annotation-intensive, often struggles to capture the global context required for reliable 3D reasoning. Conventional label-efficient methods focus on object-centric features, neglecting contextual semantic relationships that are critical in complex scenes. In this work, we propose a Context-Aware Weak Supervision for Monocular 3D object detection, namely CA-W3D, to address this limitation in a two-stage training paradigm. Specifically, we first introduce a pre-training stage employing Region-wise Object Contrastive Matching (ROCM), which aligns regional object embeddings derived from a trainable monocular 3D encoder and a frozen open-vocabulary 2D visual grounding model. This alignment encourages the monocular encoder to discriminate scene-specific attributes and acquire richer contextual knowledge. In the second stage, we incorporate a pseudo-label training process with a Dual-to-One Distillation (D2OD) mechanism, which effectively transfers contextual priors into the monocular encoder while preserving spatial fidelity and maintaining computational efficiency during inference. Extensive experiments conducted on the public KITTI benchmark demonstrate the effectiveness of our approach, surpassing the SoTA method over all metrics, highlighting the importance of contextual-aware knowledge in weakly-supervised monocular 3D detection. 

**Abstract (ZH)**: 基于上下文感知的单目3D检测弱监督方法：CA-W3D 

---
