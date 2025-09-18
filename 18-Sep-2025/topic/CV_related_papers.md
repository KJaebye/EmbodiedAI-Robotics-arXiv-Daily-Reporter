# MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping 

**Title (ZH)**: MCGS-SLAM：基于高斯点云的多相机SLAM框架，实现高保真映射 

**Authors**: Zhihao Cao, Hanyu Wu, Li Wa Tang, Zizhou Luo, Zihan Zhu, Wei Zhang, Marc Pollefeys, Martin R. Oswald  

**Link**: [PDF](https://arxiv.org/pdf/2509.14191)  

**Abstract**: Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving. 

**Abstract (ZH)**: Recent进展中的密集SLAM主要集中在单目设置上，通常会牺牲鲁棒性和几何覆盖率。我们提出了MCGS-SLAM，这是第一个基于RGB的多相机SLAM系统，构建于3D高斯点绘（3DGS）之上。不同于依赖稀疏地图或惯性数据的先前方法，MCGS-SLAM 将多视角的密集RGB输入融合为一个统一的、连续优化的高斯地图。多相机束调整（MCBA）通过密集的光度和几何残差联合优化姿态和深度，同时，尺度一致性模块使用低秩先验在不同视图之间强制实现度量对齐。该系统支持RGB输入，并在大规模应用中保持实时性能。实验结果表明，MCGS-SLAM 通常能够提供准确的轨迹和逼真的重建，通常优于单目基准。值得注意的是，多相机输入的宽视场能够重建单目设置遗漏的侧视区域，这对于自主操作的安全至关重要。这些结果突显了多相机高斯点绘SLAM在机器人和自动驾驶领域进行高保真地图构建的潜力。 

---
# \textsc{Gen2Real}: Towards Demo-Free Dexterous Manipulation by Harnessing Generated Video 

**Title (ZH)**: Gen2Real: 通过利用生成视频迈向无需演示的灵巧 manipulation 

**Authors**: Kai Ye, Yuhang Wu, Shuyuan Hu, Junliang Li, Meng Liu, Yongquan Chen, Rui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14178)  

**Abstract**: Dexterous manipulation remains a challenging robotics problem, largely due to the difficulty of collecting extensive human demonstrations for learning. In this paper, we introduce \textsc{Gen2Real}, which replaces costly human demos with one generated video and drives robot skill from it: it combines demonstration generation that leverages video generation with pose and depth estimation to yield hand-object trajectories, trajectory optimization that uses Physics-aware Interaction Optimization Model (PIOM) to impose physics consistency, and demonstration learning that retargets human motions to a robot hand and stabilizes control with an anchor-based residual Proximal Policy Optimization (PPO) policy. Using only generated videos, the learned policy achieves a 77.3\% success rate on grasping tasks in simulation and demonstrates coherent executions on a real robot. We also conduct ablation studies to validate the contribution of each component and demonstrate the ability to directly specify tasks using natural language, highlighting the flexibility and robustness of \textsc{Gen2Real} in generalizing grasping skills from imagined videos to real-world execution. 

**Abstract (ZH)**: 灵巧操作仍然是一个具有挑战性的机器人问题，主要是因为难以收集大量的手工演示用于学习。在本文中，我们介绍了\textsc{Gen2Real}，它用一个生成的视频取代了昂贵的手工演示，并从中驱动机器人的技能：它结合了利用视频生成、姿态和深度估计生成示范，以产生手-物体轨迹；使用物理感知交互优化模型（PIOM）进行路径优化，以确保物理一致性；以及通过基于锚点的残差Proximal Policy Optimization（PPO）策略将人类动作重新定向到机器人手中并稳定控制以实现示范学习。仅使用生成的视频，所学习的策略在仿真中的夹取任务中达到了77.3%的成功率，并在实际机器人上展示了连贯的执行。我们还进行了消融研究以验证每个组件的贡献，并展示了可以直接使用自然语言指定任务的能力，突显了\textsc{Gen2Real}在将想象中的视频技能泛化到实际执行中的灵活性和鲁棒性。 

---
# MetricNet: Recovering Metric Scale in Generative Navigation Policies 

**Title (ZH)**: MetricNet: 恢复生成导航策略中的度量尺度 

**Authors**: Abhijeet Nayak, Débora N.P. Oliveira, Samiran Gode, Cordelia Schmid, Wolfram Burgard  

**Link**: [PDF](https://arxiv.org/pdf/2509.13965)  

**Abstract**: Generative navigation policies have made rapid progress in improving end-to-end learned navigation. Despite their promising results, this paradigm has two structural problems. First, the sampled trajectories exist in an abstract, unscaled space without metric grounding. Second, the control strategy discards the full path, instead moving directly towards a single waypoint. This leads to short-sighted and unsafe actions, moving the robot towards obstacles that a complete and correctly scaled path would circumvent. To address these issues, we propose MetricNet, an effective add-on for generative navigation that predicts the metric distance between waypoints, grounding policy outputs in real-world coordinates. We evaluate our method in simulation with a new benchmarking framework and show that executing MetricNet-scaled waypoints significantly improves both navigation and exploration performance. Beyond simulation, we further validate our approach in real-world experiments. Finally, we propose MetricNav, which integrates MetricNet into a navigation policy to guide the robot away from obstacles while still moving towards the goal. 

**Abstract (ZH)**: 基于度量的生成导航政策已在端到端学习导航方面取得了迅速进展。尽管成果显著，该范式存在两个结构问题。首先，采样的轨迹存在于抽象且无尺度的空間中，缺乏度量基准。其次，控制策略只关注单个航点，丢弃了完整的路径信息。这会导致目光短浅且不安全的行为，使机器人朝前方障碍移动，而完整的正确尺度路径会避开这些障碍。为解决这些问题，我们提出MetricNet，这是一种有效的生成导航补充工具，能够预测航点间的度量距离，将策略输出锚定在现实世界的坐标系中。我们在一个新的基准测试框架下对我们的方法进行了模拟评估，并展示了执行MetricNet缩放后的航点显著提高了导航和探索性能。此外，我们在真实世界实验中进一步验证了我们的方法。最后，我们提出MetricNav，将MetricNet整合到导航策略中，引导机器人远离障碍物同时仍朝目标移动。 

---
# HGACNet: Hierarchical Graph Attention Network for Cross-Modal Point Cloud Completion 

**Title (ZH)**: HGACNet：分层图注意网络在跨模态点云完成中的应用 

**Authors**: Yadan Zeng, Jiadong Zhou, Xiaohan Li, I-Ming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.13692)  

**Abstract**: Point cloud completion is essential for robotic perception, object reconstruction and supporting downstream tasks like grasp planning, obstacle avoidance, and manipulation. However, incomplete geometry caused by self-occlusion and sensor limitations can significantly degrade downstream reasoning and interaction. To address these challenges, we propose HGACNet, a novel framework that reconstructs complete point clouds of individual objects by hierarchically encoding 3D geometric features and fusing them with image-guided priors from a single-view RGB image. At the core of our approach, the Hierarchical Graph Attention (HGA) encoder adaptively selects critical local points through graph attention-based downsampling and progressively refines hierarchical geometric features to better capture structural continuity and spatial relationships. To strengthen cross-modal interaction, we further design a Multi-Scale Cross-Modal Fusion (MSCF) module that performs attention-based feature alignment between hierarchical geometric features and structured visual representations, enabling fine-grained semantic guidance for completion. In addition, we proposed the contrastive loss (C-Loss) to explicitly align the feature distributions across modalities, improving completion fidelity under modality discrepancy. Finally, extensive experiments conducted on both the ShapeNet-ViPC benchmark and the YCB-Complete dataset confirm the effectiveness of HGACNet, demonstrating state-of-the-art performance as well as strong applicability in real-world robotic manipulation tasks. 

**Abstract (ZH)**: 点云完成对于机器人感知、对象重建以及抓取规划、障碍物避让和操作等下游任务至关重要。然而，由自我遮挡和传感器限制引起的不完整几何结构会严重降低下游推理和交互的性能。为应对这些挑战，我们提出了一种新的框架HGACNet，通过分层编码3D几何特征并结合单视角RGB图像的图像引导先验信息来重建个体对象的完整点云。我们的方法的核心是分层图注意力（HGA）编码器，它通过图注意力机制下的下采样自适应地选择关键局部点，并逐步细化分层几何特征以更好地捕捉结构连续性和空间关系。为了增强跨模态交互，我们设计了一个多尺度跨模态融合（MSCF）模块，在分层几何特征和结构化视觉表示之间进行注意力辅助特征对齐，从而为完成任务提供精细的语义指导。此外，我们提出了对比损失（C-Loss）以明确对齐不同模态下的特征分布，提高模态差异下的完成精度。最后，通过ShapeNet-ViPC基准和YCB-Complete数据集上的广泛实验验证了HGACNet的有效性，展示了其在机器人操作任务中的先进性能和强大适用性。 

---
# TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning 

**Title (ZH)**: TreeIRL: 基于树搜索和逆强化学习的安全城市驾驶 

**Authors**: Momchil S. Tomov, Sang Uk Lee, Hansford Hendrago, Jinwook Huh, Teawon Han, Forbes Howington, Rafael da Silva, Gianmarco Bernasconi, Marc Heim, Samuel Findler, Xiaonan Ji, Alexander Boule, Michael Napoli, Kuo Chen, Jesse Miller, Boaz Floor, Yunqing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13579)  

**Abstract**: We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving. 

**Abstract (ZH)**: TreeIRL：一种结合蒙特卡洛树搜索和逆强化学习的新型自主驾驶规划器 

---
# Semantic 3D Reconstructions with SLAM for Central Airway Obstruction 

**Title (ZH)**: 基于SLAM的中央气道阻塞的语义3D重建 

**Authors**: Ayberk Acar, Fangjie Li, Hao Li, Lidia Al-Zogbi, Kanyifeechukwu Jane Oguine, Susheela Sharma Stern, Jesse F. d'Almeida, Robert J. Webster III, Ipek Oguz, Jie Ying Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13541)  

**Abstract**: Central airway obstruction (CAO) is a life-threatening condition with increasing incidence, caused by tumors in and outside of the airway. Traditional treatment methods such as bronchoscopy and electrocautery can be used to remove the tumor completely; however, these methods carry a high risk of complications. Recent advances allow robotic interventions with lesser risk. The combination of robot interventions with scene understanding and mapping also opens up the possibilities for automation. We present a novel pipeline that enables real-time, semantically informed 3D reconstructions of the central airway using monocular endoscopic video.
Our approach combines DROID-SLAM with a segmentation model trained to identify obstructive tissues. The SLAM module reconstructs the 3D geometry of the airway in real time, while the segmentation masks guide the annotation of obstruction regions within the reconstructed point cloud. To validate our pipeline, we evaluate the reconstruction quality using ex vivo models.
Qualitative and quantitative results show high similarity between ground truth CT scans and the 3D reconstructions (0.62 mm Chamfer distance). By integrating segmentation directly into the SLAM workflow, our system produces annotated 3D maps that highlight clinically relevant regions in real time. High-speed capabilities of the pipeline allows quicker reconstructions compared to previous work, reflecting the surgical scene more accurately.
To the best of our knowledge, this is the first work to integrate semantic segmentation with real-time monocular SLAM for endoscopic CAO scenarios. Our framework is modular and can generalize to other anatomies or procedures with minimal changes, offering a promising step toward autonomous robotic interventions. 

**Abstract (ZH)**: 基于单目内窥镜视频的实时语义驱动中央气道三维重建方法 

---
# A Generalization of CLAP from 3D Localization to Image Processing, A Connection With RANSAC & Hough Transforms 

**Title (ZH)**: CLAP从三维定位的一般化到图像处理与RANSAC及霍夫变换的联系 

**Authors**: Ruochen Hou, Gabriel I. Fernandez, Alex Xu, Dennis W. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.13605)  

**Abstract**: In previous work, we introduced a 2D localization algorithm called CLAP, Clustering to Localize Across $n$ Possibilities, which was used during our championship win in RoboCup 2024, an international autonomous humanoid soccer competition. CLAP is particularly recognized for its robustness against outliers, where clustering is employed to suppress noise and mitigate against erroneous feature matches. This clustering-based strategy provides an alternative to traditional outlier rejection schemes such as RANSAC, in which candidates are validated by reprojection error across all data points. In this paper, CLAP is extended to a more general framework beyond 2D localization, specifically to 3D localization and image stitching. We also show how CLAP, RANSAC, and Hough transforms are related. The generalization of CLAP is widely applicable to many different fields and can be a useful tool to deal with noise and uncertainty. 

**Abstract (ZH)**: 先前工作中，我们提出了一种名为CLAP的2D定位算法，即Clustering to Localize Across $n$ Possibilities，在2024年国际自主 humanoid 足球锦标赛RoboCup中获得冠军。CLAP特别因其对离群值的鲁棒性而受到认可，其中聚类方法被用于抑制噪声并减轻错误特征匹配的影响。基于聚类的策略为传统的离群值剔除方案（如RANSAC）提供了一种替代方案，RANSAC方案通过重现投影误差来验证候选者。在本文中，我们将CLAP扩展到了更通用的框架中，具体应用到了3D定位和图像拼接。我们还展示了CLAP、RANSAC和霍夫变换之间的关系。CLAP的一般化方法在许多不同的领域都有广泛应用，并可作为应对噪声和不确定性的一种有用工具。 

---
# Dynamic Aware: Adaptive Multi-Mode Out-of-Distribution Detection for Trajectory Prediction in Autonomous Vehicles 

**Title (ZH)**: 动态感知：面向自主车辆轨迹预测的自适应多模式离分布检测 

**Authors**: Tongfei Guo, Lili Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.13577)  

**Abstract**: Trajectory prediction is central to the safe and seamless operation of autonomous vehicles (AVs). In deployment, however, prediction models inevitably face distribution shifts between training data and real-world conditions, where rare or underrepresented traffic scenarios induce out-of-distribution (OOD) cases. While most prior OOD detection research in AVs has concentrated on computer vision tasks such as object detection and segmentation, trajectory-level OOD detection remains largely underexplored. A recent study formulated this problem as a quickest change detection (QCD) task, providing formal guarantees on the trade-off between detection delay and false alarms [1]. Building on this foundation, we propose a new framework that introduces adaptive mechanisms to achieve robust detection in complex driving environments. Empirical analysis across multiple real-world datasets reveals that prediction errors--even on in-distribution samples--exhibit mode-dependent distributions that evolve over time with dataset-specific dynamics. By explicitly modeling these error modes, our method achieves substantial improvements in both detection delay and false alarm rates. Comprehensive experiments on established trajectory prediction benchmarks show that our framework significantly outperforms prior UQ- and vision-based OOD approaches in both accuracy and computational efficiency, offering a practical path toward reliable, driving-aware autonomy. 

**Abstract (ZH)**: 自主车辆中轨迹预测的分布偏移检测对于安全无缝的操作至关重要。然而，在部署过程中，预测模型不可避免地会面临训练数据与实际条件之间的分布偏移，其中罕见或未充分代表的交通场景会导致分布外(OOD)情况。虽然以往关于自主车辆中分布外检测的研究主要集中在目标检测和分割等计算机视觉任务上，但轨迹级别的分布外检测仍待深入探索。近期的研究将该问题表述为快速变化检测(QCD)任务，并对检测延迟和误报之间的权衡提供了形式化的保证[1]。在此基础上，我们提出了一种新的框架，引入适应性机制以在复杂的驾驶环境中实现稳健的检测。跨多个实际数据集的经验分析表明，即使是在分布内样本上，预测误差也表现出受模式依赖的分布，并随数据集特定的动力学而演变。通过明确建模这些误差模式，我们的方法在检测延迟和误报率方面取得了显著改进。在已建立的轨迹预测基准上的综合实验表明，与先前的不确定性量化(UQ)和视觉基线的分布外方法相比，我们的框架在准确性和计算效率上均表现优异，提供了实现可靠、驾驶感知自主性的现实之路。 

---
# MapAnything: Universal Feed-Forward Metric 3D Reconstruction 

**Title (ZH)**: MapAnything: 通用前馈度量3D重建 

**Authors**: Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulò, Christian Richardt, Deva Ramanan, Sebastian Scherer, Peter Kontschieder  

**Link**: [PDF](https://arxiv.org/pdf/2509.13414)  

**Abstract**: We introduce MapAnything, a unified transformer-based feed-forward model that ingests one or more images along with optional geometric inputs such as camera intrinsics, poses, depth, or partial reconstructions, and then directly regresses the metric 3D scene geometry and cameras. MapAnything leverages a factored representation of multi-view scene geometry, i.e., a collection of depth maps, local ray maps, camera poses, and a metric scale factor that effectively upgrades local reconstructions into a globally consistent metric frame. Standardizing the supervision and training across diverse datasets, along with flexible input augmentation, enables MapAnything to address a broad range of 3D vision tasks in a single feed-forward pass, including uncalibrated structure-from-motion, calibrated multi-view stereo, monocular depth estimation, camera localization, depth completion, and more. We provide extensive experimental analyses and model ablations demonstrating that MapAnything outperforms or matches specialist feed-forward models while offering more efficient joint training behavior, thus paving the way toward a universal 3D reconstruction backbone. 

**Abstract (ZH)**: MapAnything：一种统一的Transformer基馈前模型及其在3D场景几何与摄像机直接回归中的应用 

---
# Dense Video Understanding with Gated Residual Tokenization 

**Title (ZH)**: 基于门控残差化令牌化的大密度视频理解 

**Authors**: Haichao Zhang, Wenhao Chai, Shwai He, Ang Li, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14199)  

**Abstract**: High temporal resolution is essential for capturing fine-grained details in video understanding. However, current video large language models (VLLMs) and benchmarks mostly rely on low-frame-rate sampling, such as uniform sampling or keyframe selection, discarding dense temporal information. This compromise avoids the high cost of tokenizing every frame, which otherwise leads to redundant computation and linear token growth as video length increases. While this trade-off works for slowly changing content, it fails for tasks like lecture comprehension, where information appears in nearly every frame and requires precise temporal alignment. To address this gap, we introduce Dense Video Understanding (DVU), which enables high-FPS video comprehension by reducing both tokenization time and token overhead. Existing benchmarks are also limited, as their QA pairs focus on coarse content changes. We therefore propose DIVE (Dense Information Video Evaluation), the first benchmark designed for dense temporal reasoning. To make DVU practical, we present Gated Residual Tokenization (GRT), a two-stage framework: (1) Motion-Compensated Inter-Gated Tokenization uses pixel-level motion estimation to skip static regions during tokenization, achieving sub-linear growth in token count and compute. (2) Semantic-Scene Intra-Tokenization Merging fuses tokens across static regions within a scene, further reducing redundancy while preserving dynamic semantics. Experiments on DIVE show that GRT outperforms larger VLLM baselines and scales positively with FPS. These results highlight the importance of dense temporal information and demonstrate that GRT enables efficient, scalable high-FPS video understanding. 

**Abstract (ZH)**: 高时间分辨率对于视频理解中捕捉细粒度细节至关重要。然而，当前的视频大规模语言模型（VLLMs）和基准主要依赖于低帧率采样，如均匀采样或关键帧选择，从而丢弃了密集的时间信息。这种权衡避免了对每一帧进行分词所导致的高成本，否则会导致冗余计算和随视频长度增加呈线性增长的分词数量。虽然这种权衡对于缓慢变化的内容有效，但对于讲义理解等任务来说却不合适，这类任务中信息几乎出现在每一帧中，并需要精确的时间对齐。为解决这一差距，我们引入了密集视频理解（DVU），它通过减少分词时间和分词开销，使高帧率视频理解成为可能。现有的基准也有局限性，因为它们的问答对主要关注粗粒度的内容变化。因此，我们提出了DIVE（密集信息视频评估）作为第一个专为密集时间推理设计的基准。为了使DVU实用化，我们提出了门控残差分词（GRT），这是一种两阶段框架：（1）运动补偿跨门分词利用像素级别的运动估计，在分词过程中跳过静态区域，实现分词数量和计算量的次线性增长；（2）语义场景内分词合并将场景内静态区域的分词合并，进一步减少冗余，同时保留动态语义。在DIVE上的实验结果显示，GRT比更大规模的VLLM基线表现出色，并且随着帧率增加呈现出正向扩展。这些结果突显了密集时间信息的重要性，并证明了GRT能够实现高效的、可扩展的高帧率视频理解。 

---
# BWCache: Accelerating Video Diffusion Transformers through Block-Wise Caching 

**Title (ZH)**: BWCache：通过块级缓存加速视频扩散变换器 

**Authors**: Hanshuai Cui, Zhiqing Tang, Zhifei Xu, Zhi Yao, Wenyi Zeng, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.13789)  

**Abstract**: Recent advancements in Diffusion Transformers (DiTs) have established them as the state-of-the-art method for video generation. However, their inherently sequential denoising process results in inevitable latency, limiting real-world applicability. Existing acceleration methods either compromise visual quality due to architectural modifications or fail to reuse intermediate features at proper granularity. Our analysis reveals that DiT blocks are the primary contributors to inference latency. Across diffusion timesteps, the feature variations of DiT blocks exhibit a U-shaped pattern with high similarity during intermediate timesteps, which suggests substantial computational redundancy. In this paper, we propose Block-Wise Caching (BWCache), a training-free method to accelerate DiT-based video generation. BWCache dynamically caches and reuses features from DiT blocks across diffusion timesteps. Furthermore, we introduce a similarity indicator that triggers feature reuse only when the differences between block features at adjacent timesteps fall below a threshold, thereby minimizing redundant computations while maintaining visual fidelity. Extensive experiments on several video diffusion models demonstrate that BWCache achieves up to 2.24$\times$ speedup with comparable visual quality. 

**Abstract (ZH)**: Recent advancements in Diffusion Transformers (DiTs) have established them as the state-of-the-art method for video generation. However, their inherently sequential denoising process results in inevitable latency, limiting real-world applicability.现有的扩散变压器（DiTs）发展已经使它们成为视频生成的最先进的方法。然而，其固有的顺序去噪过程会导致不可避免的延迟，限制了其实用性。 

---
# Mitigating Query Selection Bias in Referring Video Object Segmentation 

**Title (ZH)**: 在引用视频对象分割中减轻查询选择偏见 

**Authors**: Dingwei Zhang, Dong Zhang, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13722)  

**Abstract**: Recently, query-based methods have achieved remarkable performance in Referring Video Object Segmentation (RVOS) by using textual static object queries to drive cross-modal alignment. However, these static queries are easily misled by distractors with similar appearance or motion, resulting in \emph{query selection bias}. To address this issue, we propose Triple Query Former (TQF), which factorizes the referring query into three specialized components: an appearance query for static attributes, an intra-frame interaction query for spatial relations, and an inter-frame motion query for temporal association. Instead of relying solely on textual embeddings, our queries are dynamically constructed by integrating both linguistic cues and visual guidance. Furthermore, we introduce two motion-aware aggregation modules that enhance object token representations: Intra-frame Interaction Aggregation incorporates position-aware interactions among objects within a single frame, while Inter-frame Motion Aggregation leverages trajectory-guided alignment across frames to ensure temporal coherence. Extensive experiments on multiple RVOS benchmarks demonstrate the advantages of TQF and the effectiveness of our structured query design and motion-aware aggregation modules. 

**Abstract (ZH)**: 基于三元查询的视频对象分割方法： Triple Query Former (TQF) 用于解决查询选择偏见并增强时空一致性 

---
# CraftMesh: High-Fidelity Generative Mesh Manipulation via Poisson Seamless Fusion 

**Title (ZH)**: CraftMesh：通过泊松无缝融合实现的高保真生成网格 manipulation 

**Authors**: James Jincheng, Youcheng Cai, Ligang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.13688)  

**Abstract**: Controllable, high-fidelity mesh editing remains a significant challenge in 3D content creation. Existing generative methods often struggle with complex geometries and fail to produce detailed results. We propose CraftMesh, a novel framework for high-fidelity generative mesh manipulation via Poisson Seamless Fusion. Our key insight is to decompose mesh editing into a pipeline that leverages the strengths of 2D and 3D generative models: we edit a 2D reference image, then generate a region-specific 3D mesh, and seamlessly fuse it into the original model. We introduce two core techniques: Poisson Geometric Fusion, which utilizes a hybrid SDF/Mesh representation with normal blending to achieve harmonious geometric integration, and Poisson Texture Harmonization for visually consistent texture blending. Experimental results demonstrate that CraftMesh outperforms state-of-the-art methods, delivering superior global consistency and local detail in complex editing tasks. 

**Abstract (ZH)**: 可控的高保真网格编辑仍然是3D内容创建中的一个重大挑战。现有生成方法往往难以处理复杂几何形状，并且无法生成详细的成果。我们提出了CraftMesh，一种通过泊松无缝融合进行高保真生成网格操作的新框架。我们的关键洞察是将网格编辑分解为一个利用2D和3D生成模型优势的流水线：我们编辑一个2D参考图像，然后生成区域特定的3D网格，并将其无缝融合到原始模型中。我们介绍了两种核心技术：泊松几何融合，利用混合SDF/网格表示与法线混合以实现和谐的几何集成，以及泊松纹理谐振，实现视觉一致的纹理混合。实验结果表明，CraftMesh 在复杂编辑任务中优于现有最先进的方法，提供了更好的全局一致性和局部细节。 

---
# ColonCrafter: A Depth Estimation Model for Colonoscopy Videos Using Diffusion Priors 

**Title (ZH)**: ColonCrafter：一种使用扩散先验的结肠镜视频深度估计模型 

**Authors**: Romain Hardy, Tyler Berzin, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2509.13525)  

**Abstract**: Three-dimensional (3D) scene understanding in colonoscopy presents significant challenges that necessitate automated methods for accurate depth estimation. However, existing depth estimation models for endoscopy struggle with temporal consistency across video sequences, limiting their applicability for 3D reconstruction. We present ColonCrafter, a diffusion-based depth estimation model that generates temporally consistent depth maps from monocular colonoscopy videos. Our approach learns robust geometric priors from synthetic colonoscopy sequences to generate temporally consistent depth maps. We also introduce a style transfer technique that preserves geometric structure while adapting real clinical videos to match our synthetic training domain. ColonCrafter achieves state-of-the-art zero-shot performance on the C3VD dataset, outperforming both general-purpose and endoscopy-specific approaches. Although full trajectory 3D reconstruction remains a challenge, we demonstrate clinically relevant applications of ColonCrafter, including 3D point cloud generation and surface coverage assessment. 

**Abstract (ZH)**: 结肠镜检查中的三维场景理解面临显著挑战，需要自动化方法以实现准确的深度估计。现有的内窥镜深度估计模型在视频序列之间难以保持时间一致性，限制了其在三维重建中的应用。我们提出了ColonCrafter，一种基于扩散的深度估计模型，该模型能从单目结肠镜视频中生成时间一致性深度图。我们的方法通过学习合成结肠镜序列中的鲁棒几何先验知识，生成时间一致性深度图。我们还引入了一种风格迁移技术，该技术在保留几何结构的同时，将真实临床视频适应我们的合成训练域。ColonCrafter在C3VD数据集上实现了最先进的零样本性能，优于通用和内窥镜专用方法。尽管全轨迹三维重建仍然是一个挑战，但我们展示了ColonCrafter在临床相关的应用，包括三维点云生成和表面覆盖评估。 

---
# Generative AI Pipeline for Interactive Prompt-driven 2D-to-3D Vascular Reconstruction for Fontan Geometries from Contrast-Enhanced X-Ray Fluoroscopy Imaging 

**Title (ZH)**: 基于生成式AI的工作流：交互式提示驱动的从对比增强X射线荧光成像重建Fontan几何的2D至3D血管重建 

**Authors**: Prahlad G Menon  

**Link**: [PDF](https://arxiv.org/pdf/2509.13372)  

**Abstract**: Fontan palliation for univentricular congenital heart disease progresses to hemodynamic failure with complex flow patterns poorly characterized by conventional 2D imaging. Current assessment relies on fluoroscopic angiography, providing limited 3D geometric information essential for computational fluid dynamics (CFD) analysis and surgical planning.
A multi-step AI pipeline was developed utilizing Google's Gemini 2.5 Flash (2.5B parameters) for systematic, iterative processing of fluoroscopic angiograms through transformer-based neural architecture. The pipeline encompasses medical image preprocessing, vascular segmentation, contrast enhancement, artifact removal, and virtual hemodynamic flow visualization within 2D projections. Final views were processed through Tencent's Hunyuan3D-2mini (384M parameters) for stereolithography file generation.
The pipeline successfully generated geometrically optimized 2D projections from single-view angiograms after 16 processing steps using a custom web interface. Initial iterations contained hallucinated vascular features requiring iterative refinement to achieve anatomically faithful representations. Final projections demonstrated accurate preservation of complex Fontan geometry with enhanced contrast suitable for 3D conversion. AI-generated virtual flow visualization identified stagnation zones in central connections and flow patterns in branch arteries. Complete processing required under 15 minutes with second-level API response times.
This approach demonstrates clinical feasibility of generating CFD-suitable geometries from routine angiographic data, enabling 3D generation and rapid virtual flow visualization for cursory insights prior to full CFD simulation. While requiring refinement cycles for accuracy, this establishes foundation for democratizing advanced geometric and hemodynamic analysis using readily available imaging data. 

**Abstract (ZH)**: Fontan矫治在单心室先天性心脏病中的进展与复杂血流模式的常规二维成像 characterization不足：一种多步AI管道的应用 

---
# Hybrid Quantum-Classical Model for Image Classification 

**Title (ZH)**: 量子经典混合模型在图像分类中的应用 

**Authors**: Muhammad Adnan Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2509.13353)  

**Abstract**: This study presents a systematic comparison between hybrid quantum-classical neural networks and purely classical models across three benchmark datasets (MNIST, CIFAR100, and STL10) to evaluate their performance, efficiency, and robustness. The hybrid models integrate parameterized quantum circuits with classical deep learning architectures, while the classical counterparts use conventional convolutional neural networks (CNNs). Experiments were conducted over 50 training epochs for each dataset, with evaluations on validation accuracy, test accuracy, training time, computational resource usage, and adversarial robustness (tested with $\epsilon=0.1$ perturbations).Key findings demonstrate that hybrid models consistently outperform classical models in final accuracy, achieving {99.38\% (MNIST), 41.69\% (CIFAR100), and 74.05\% (STL10) validation accuracy, compared to classical benchmarks of 98.21\%, 32.25\%, and 63.76\%, respectively. Notably, the hybrid advantage scales with dataset complexity, showing the most significant gains on CIFAR100 (+9.44\%) and STL10 (+10.29\%). Hybrid models also train 5--12$\times$ faster (e.g., 21.23s vs. 108.44s per epoch on MNIST) and use 6--32\% fewer parameters} while maintaining superior generalization to unseen test this http URL robustness tests reveal that hybrid models are significantly more resilient on simpler datasets (e.g., 45.27\% robust accuracy on MNIST vs. 10.80\% for classical) but show comparable fragility on complex datasets like CIFAR100 ($\sim$1\% robustness for both). Resource efficiency analyses indicate that hybrid models consume less memory (4--5GB vs. 5--6GB for classical) and lower CPU utilization (9.5\% vs. 23.2\% on average).These results suggest that hybrid quantum-classical architectures offer compelling advantages in accuracy, training efficiency, and parameter scalability, particularly for complex vision tasks. 

**Abstract (ZH)**: 这项研究在MNIST、CIFAR100和STL10三个基准数据集上系统比较了混合量子-古典神经网络和纯古典模型的性能、效率和鲁棒性。混合模型将参数化量子电路与经典的深度学习架构结合，而纯古典模型则使用传统的卷积神经网络（CNNs）。每个数据集在50个训练周期进行实验，评估验证准确率、测试准确率、训练时间、计算资源使用情况以及对抗鲁棒性（以$\epsilon=0.1$的扰动进行测试）。关键发现表明，混合模型在最终准确率上始终优于经典模型，在MNIST、CIFAR100和STL10上的验证准确率分别为99.38%、41.69%和74.05%，而经典模型的基准值分别为98.21%、32.25%和63.76%。值得注意的是，混合模型的优势随着数据集复杂度的增加而放大，在CIFAR100上提高了9.44%，在STL10上提高了10.29%。混合模型还比经典模型快5-12倍（例如，在MNIST上，每epoch训练时间分别为21.23秒和108.44秒），参数量少6-32%，同时保持更好的对未见测试集的一般化能力。鲁棒性测试结果显示，混合模型在简单数据集上的鲁棒准确率显著更高（例如，在MNIST上的鲁棒准确率为45.27%，而经典模型为10.80%），但在复杂数据集如CIFAR100上表现出相当的脆弱性（两者都约为1%的鲁棒性）。资源效率分析表明，混合模型消耗更少的内存（4-5GB对经典模型的5-6GB），并且更低的CPU利用率（平均9.5%对经典模型的23.2%）。这些结果表明，混合量子-古典架构在准确性、训练效率和参数可扩展性方面，尤其是在复杂的视觉任务中，提供了显著的优势。 

---
# Label-Efficient Grasp Joint Prediction with Point-JEPA 

**Title (ZH)**: 标签高效的抓取关节预测ewith点-JEPA 

**Authors**: Jed Guzelkabaagac, Boris Petrović  

**Link**: [PDF](https://arxiv.org/pdf/2509.13349)  

**Abstract**: We investigate whether 3D self-supervised pretraining with a Joint-Embedding Predictive Architecture (Point-JEPA) enables label-efficient grasp joint-angle prediction. Using point clouds tokenized from meshes and a ShapeNet-pretrained Point-JEPA encoder, we train a lightweight multi-hypothesis head with winner-takes-all and evaluate by top-logit selection. On DLR-Hand II with object-level splits, Point-JEPA reduces RMSE by up to 26% in low-label regimes and reaches parity with full supervision. These results suggest JEPA-style pretraining is a practical approach for data-efficient grasp learning. 

**Abstract (ZH)**: 基于Joint-Embedding Predictive Architecture的3D自监督预训练是否能实现标签高效的抓取关节角预测：点云表示下的实证研究 

---
