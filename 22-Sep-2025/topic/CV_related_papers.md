# Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion 

**Title (ZH)**: 全方位LiDAR-Omni: 基于光度迁移和ESIKF融合的鲁棒多摄像机RGB彩色视觉-惯性-LiDAR里程计 

**Authors**: Yinong Cao, Xin He, Yuwei Chen, Chenyang Zhang, Chengyu Pu, Bingtao Wang, Kaile Wu, Shouzheng Zhu, Fei Han, Shijie Liu, Chunlai Li, Jianyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15673)  

**Abstract**: Wide field-of-view (FoV) LiDAR sensors provide dense geometry across large environments, but most existing LiDAR-inertial-visual odometry (LIVO) systems rely on a single camera, leading to limited spatial coverage and degraded robustness. We present Omni-LIVO, the first tightly coupled multi-camera LIVO system that bridges the FoV mismatch between wide-angle LiDAR and conventional cameras. Omni-LIVO introduces a Cross-View direct tracking strategy that maintains photometric consistency across non-overlapping views, and extends the Error-State Iterated Kalman Filter (ESIKF) with multi-view updates and adaptive covariance weighting. The system is evaluated on public benchmarks and our custom dataset, showing improved accuracy and robustness over state-of-the-art LIVO, LIO, and visual-inertial baselines. Code and dataset will be released upon publication. 

**Abstract (ZH)**: 广视野LiDAR传感器提供了大面积环境下的密集几何信息，但现有的LiDAR-惯性-视觉里程计（LIVO）系统大多仅依赖单一摄像头，导致空间覆盖率有限和鲁棒性下降。本文提出了Omni-LIVO，这是一种首次将宽角LiDAR与常规摄像头视野不匹配问题紧密结合的多摄像头LIVO系统。Omni-LIVO引入了一种跨视图直接跟踪策略，以在非重叠视图中保持光电一致性，并通过多视图更新和自适应协方差加权扩展了错误状态迭代卡尔曼滤波器（ESIKF）。系统在公共基准数据集和我们自建的数据集上进行了评估，结果显示其在姿态估计准确性与鲁棒性方面优于当前最先进的LIVO、LIO和视觉惯性基准方法。代码和数据集将在发表后公开。 

---
# Bench-RNR: Dataset for Benchmarking Repetitive and Non-repetitive Scanning LiDAR for Infrastructure-based Vehicle Localization 

**Title (ZH)**: Bench-RNR：基于基础设施车辆定位的重复扫描与非重复扫描LiDAR基准测试数据集 

**Authors**: Runxin Zhao, Chunxiang Wang, Hanyang Zhuang, Ming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15583)  

**Abstract**: Vehicle localization using roadside LiDARs can provide centimeter-level accuracy for cloud-controlled vehicles while simultaneously serving multiple vehicles, enhanc-ing safety and efficiency. While most existing studies rely on repetitive scanning LiDARs, non-repetitive scanning LiDAR offers advantages such as eliminating blind zones and being more cost-effective. However, its application in roadside perception and localization remains limited. To address this, we present a dataset for infrastructure-based vehicle localization, with data collected from both repetitive and non-repetitive scanning LiDARs, in order to benchmark the performance of different LiDAR scanning patterns. The dataset contains 5,445 frames of point clouds across eight vehicle trajectory sequences, with diverse trajectory types. Our experiments establish base-lines for infrastructure-based vehicle localization and compare the performance of these methods using both non-repetitive and repetitive scanning LiDARs. This work offers valuable insights for selecting the most suitable LiDAR scanning pattern for infrastruc-ture-based vehicle localization. Our dataset is a signifi-cant contribution to the scientific community, supporting advancements in infrastructure-based perception and vehicle localization. The dataset and source code are publicly available at: this https URL. 

**Abstract (ZH)**: 基于路边LiDAR的车辆定位可以为云端控制车辆提供厘米级精度，同时服务于多辆车辆，增强安全性和效率。尽管大多数现有研究依赖重复扫描LiDAR，非重复扫描LiDAR具有消除盲区和更经济的优点，但其在路边感知与定位中的应用仍然有限。为了解决这个问题，我们提出了一个基于基础设施的车辆定位数据集，收集了重复扫描和非重复扫描LiDAR的数据，以评估不同LiDAR扫描模式的性能。该数据集包含八个车辆轨迹序列中的5,445帧点云数据，涵盖了多种轨迹类型。我们的实验建立了基于基础设施的车辆定位基准，并使用非重复扫描和重复扫描LiDAR比较了不同方法的性能。该研究为选择最适合基于基础设施的车辆定位的LiDAR扫描模式提供了有价值的见解。我们的数据集对科学界是一个重要的贡献，支持基础设施感知和车辆定位技术的发展。该数据集和源代码可在以下链接公开获取：this https URL。 

---
# Towards Sharper Object Boundaries in Self-Supervised Depth Estimation 

**Title (ZH)**: 自监督深度估计中更清晰对象边界的研究 

**Authors**: Aurélien Cecille, Stefan Duffner, Franck Davoine, Rémi Agier, Thibault Neveu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15987)  

**Abstract**: Accurate monocular depth estimation is crucial for 3D scene understanding, but existing methods often blur depth at object boundaries, introducing spurious intermediate 3D points. While achieving sharp edges usually requires very fine-grained supervision, our method produces crisp depth discontinuities using only self-supervision. Specifically, we model per-pixel depth as a mixture distribution, capturing multiple plausible depths and shifting uncertainty from direct regression to the mixture weights. This formulation integrates seamlessly into existing pipelines via variance-aware loss functions and uncertainty propagation. Extensive evaluations on KITTI and VKITTIv2 show that our method achieves up to 35% higher boundary sharpness and improves point cloud quality compared to state-of-the-art baselines. 

**Abstract (ZH)**: 单目深度估计对于三维场景理解至关重要，但现有方法往往在物体边界处模糊深度，引入虚假的中间3D点。虽然获得清晰边缘通常需要非常精细的监督，我们的方法仅使用自我监督即可产生清晰的深度不连续性。具体而言，我们将每个像素的深度建模为混合分布，捕获多个可能的深度并从直接回归转移到混合权重的不确定性。该公式通过方差感知的损失函数和不确定性传播无缝集成到现有管道中。在KITTI和VKITTIv2上的 extensive 评估表明，与最先进的基线方法相比，我们的方法在边界锐度上提高了35%以上，并改善了点云质量。 

---
# KoopCast: Trajectory Forecasting via Koopman Operators 

**Title (ZH)**: KoopCast：基于柯普曼算子的轨迹预测 

**Authors**: Jungjin Lee, Jaeuk Shin, Gihwan Kim, Joonho Han, Insoon Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15513)  

**Abstract**: We present KoopCast, a lightweight yet efficient model for trajectory forecasting in general dynamic environments. Our approach leverages Koopman operator theory, which enables a linear representation of nonlinear dynamics by lifting trajectories into a higher-dimensional space. The framework follows a two-stage design: first, a probabilistic neural goal estimator predicts plausible long-term targets, specifying where to go; second, a Koopman operator-based refinement module incorporates intention and history into a nonlinear feature space, enabling linear prediction that dictates how to go. This dual structure not only ensures strong predictive accuracy but also inherits the favorable properties of linear operators while faithfully capturing nonlinear dynamics. As a result, our model offers three key advantages: (i) competitive accuracy, (ii) interpretability grounded in Koopman spectral theory, and (iii) low-latency deployment. We validate these benefits on ETH/UCY, the Waymo Open Motion Dataset, and nuScenes, which feature rich multi-agent interactions and map-constrained nonlinear motion. Across benchmarks, KoopCast consistently delivers high predictive accuracy together with mode-level interpretability and practical efficiency. 

**Abstract (ZH)**: KoopCast:一种轻量高效的动态环境轨迹预测模型 

---
# Structured Information for Improving Spatial Relationships in Text-to-Image Generation 

**Title (ZH)**: 结构化信息以提高文本到图像生成中的空间关系 

**Authors**: Sander Schildermans, Chang Tian, Ying Jiao, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2509.15962)  

**Abstract**: Text-to-image (T2I) generation has advanced rapidly, yet faithfully capturing spatial relationships described in natural language prompts remains a major challenge. Prior efforts have addressed this issue through prompt optimization, spatially grounded generation, and semantic refinement. This work introduces a lightweight approach that augments prompts with tuple-based structured information, using a fine-tuned language model for automatic conversion and seamless integration into T2I pipelines. Experimental results demonstrate substantial improvements in spatial accuracy, without compromising overall image quality as measured by Inception Score. Furthermore, the automatically generated tuples exhibit quality comparable to human-crafted tuples. This structured information provides a practical and portable solution to enhance spatial relationships in T2I generation, addressing a key limitation of current large-scale generative systems. 

**Abstract (ZH)**: 基于文本到图像生成中空间关系的轻量化结构化增强方法 

---
# Fast OTSU Thresholding Using Bisection Method 

**Title (ZH)**: 快速二分法OTSU阈值分割 

**Authors**: Sai Varun Kodathala  

**Link**: [PDF](https://arxiv.org/pdf/2509.16179)  

**Abstract**: The Otsu thresholding algorithm represents a fundamental technique in image segmentation, yet its computational efficiency is severely limited by exhaustive search requirements across all possible threshold values. This work presents an optimized implementation that leverages the bisection method to exploit the unimodal characteristics of the between-class variance function. Our approach reduces the computational complexity from O(L) to O(log L) evaluations while preserving segmentation accuracy. Experimental validation on 48 standard test images demonstrates a 91.63% reduction in variance computations and 97.21% reduction in algorithmic iterations compared to conventional exhaustive search. The bisection method achieves exact threshold matches in 66.67% of test cases, with 95.83% exhibiting deviations within 5 gray levels. The algorithm maintains universal convergence within theoretical logarithmic bounds while providing deterministic performance guarantees suitable for real-time applications. This optimization addresses critical computational bottlenecks in large-scale image processing systems without compromising the theoretical foundations or segmentation quality of the original Otsu method. 

**Abstract (ZH)**: 基于二分法优化的Otsu阈值算法在图像分割中的高效实现 

---
# Shedding Light on Depth: Explainability Assessment in Monocular Depth Estimation 

**Title (ZH)**: 解析深度：单目深度估计的可解释性评估 

**Authors**: Lorenzo Cirillo, Claudio Schiavella, Lorenzo Papa, Paolo Russo, Irene Amerini  

**Link**: [PDF](https://arxiv.org/pdf/2509.15980)  

**Abstract**: Explainable artificial intelligence is increasingly employed to understand the decision-making process of deep learning models and create trustworthiness in their adoption. However, the explainability of Monocular Depth Estimation (MDE) remains largely unexplored despite its wide deployment in real-world applications. In this work, we study how to analyze MDE networks to map the input image to the predicted depth map. More in detail, we investigate well-established feature attribution methods, Saliency Maps, Integrated Gradients, and Attention Rollout on different computationally complex models for MDE: METER, a lightweight network, and PixelFormer, a deep network. We assess the quality of the generated visual explanations by selectively perturbing the most relevant and irrelevant pixels, as identified by the explainability methods, and analyzing the impact of these perturbations on the model's output. Moreover, since existing evaluation metrics can have some limitations in measuring the validity of visual explanations for MDE, we additionally introduce the Attribution Fidelity. This metric evaluates the reliability of the feature attribution by assessing their consistency with the predicted depth map. Experimental results demonstrate that Saliency Maps and Integrated Gradients have good performance in highlighting the most important input features for MDE lightweight and deep models, respectively. Furthermore, we show that Attribution Fidelity effectively identifies whether an explainability method fails to produce reliable visual maps, even in scenarios where conventional metrics might suggest satisfactory results. 

**Abstract (ZH)**: 可解释的人工智能越来越多地被用于理解深度学习模型的决策过程，并在其实用化中建立信任。然而，单目深度估计（MDE）的可解释性尚未得到充分探索，尽管它在实际应用中的部署非常广泛。在本文中，我们研究如何分析MDE网络以将输入图像映射到预测的深度图。具体而言，我们 Investigated 基于特征归属方法、显著图、集成梯度和注意力展开的现有技术，这些方法应用于不同计算复杂度的MDE模型：METER（一种轻量级网络）和PixelFormer（一种深度网络）。我们通过有选择地扰动由解释性方法识别的最相关和最不相关信息像素，并分析这些扰动对模型输出的影响来评估生成的视觉解释的质量。此外，由于现有评估指标在测量MDE的视觉解释有效性时存在一些局限性，我们还引入了归属保真度这一度量标准。该度量标准通过评估特征归属的一致性来评估其与预测深度图的可靠性。实验结果表明，显著图和集成梯度在分别突出轻量级和深度MDE模型的最重要输入特征方面表现出色。此外，我们证明了归属性制度在识别解释性方法未能产生可靠视觉图的情况时是有效的，即使在传统度量可能会给出满意结果的情况下也是如此。 

---
# MoAngelo: Motion-Aware Neural Surface Reconstruction for Dynamic Scenes 

**Title (ZH)**: MoAngelo: 动态场景中基于运动的神经表面重建 

**Authors**: Mohamed Ebbed, Zorah Lähner  

**Link**: [PDF](https://arxiv.org/pdf/2509.15892)  

**Abstract**: Dynamic scene reconstruction from multi-view videos remains a fundamental challenge in computer vision. While recent neural surface reconstruction methods have achieved remarkable results in static 3D reconstruction, extending these approaches with comparable quality for dynamic scenes introduces significant computational and representational challenges. Existing dynamic methods focus on novel-view synthesis, therefore, their extracted meshes tend to be noisy. Even approaches aiming for geometric fidelity often result in too smooth meshes due to the ill-posedness of the problem. We present a novel framework for highly detailed dynamic reconstruction that extends the static 3D reconstruction method NeuralAngelo to work in dynamic settings. To that end, we start with a high-quality template scene reconstruction from the initial frame using NeuralAngelo, and then jointly optimize deformation fields that track the template and refine it based on the temporal sequence. This flexible template allows updating the geometry to include changes that cannot be modeled with the deformation field, for instance occluded parts or the changes in the topology. We show superior reconstruction accuracy in comparison to previous state-of-the-art methods on the ActorsHQ dataset. 

**Abstract (ZH)**: 多视图视频中的动态场景重建仍然是计算机视觉中的一个基本挑战。尽管最近的神经表面重建方法在静态3D重建方面取得了显著成果，但在动态场景中应用这些方法并保持相似的质量引入了巨大的计算和表示挑战。现有的动态方法主要集中在新颖视角合成，因此其提取的网格往往噪声较大。即使是为了几何保真的方法，往往也会因为问题的病态性而导致网格过于光滑。我们提出了一种新的框架，将静态3D重建方法NeuralAngelo扩展到动态场景中使用。我们首先使用NeuralAngelo从初始帧重建高质量的模板场景，然后联合优化追踪模板并基于时间序列进行细化的变形场。这种灵活的模板允许更新几何形状以包含无法通过变形场建模的变化，例如被遮挡的部分或拓扑结构的变化。我们展示了在ActorsHQ数据集上与之前最先进的方法相比具有更高的重建精度。 

---
# Self-Supervised Cross-Modal Learning for Image-to-Point Cloud Registration 

**Title (ZH)**: 自我监督跨模态学习在图像到点云注册中的应用 

**Authors**: Xingmei Wang, Xiaoyu Hu, Chengkai Huang, Ziyan Zeng, Guohao Nie, Quan Z. Sheng, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.15882)  

**Abstract**: Bridging 2D and 3D sensor modalities is critical for robust perception in autonomous systems. However, image-to-point cloud (I2P) registration remains challenging due to the semantic-geometric gap between texture-rich but depth-ambiguous images and sparse yet metrically precise point clouds, as well as the tendency of existing methods to converge to local optima. To overcome these limitations, we introduce CrossI2P, a self-supervised framework that unifies cross-modal learning and two-stage registration in a single end-to-end pipeline. First, we learn a geometric-semantic fused embedding space via dual-path contrastive learning, enabling annotation-free, bidirectional alignment of 2D textures and 3D structures. Second, we adopt a coarse-to-fine registration paradigm: a global stage establishes superpoint-superpixel correspondences through joint intra-modal context and cross-modal interaction modeling, followed by a geometry-constrained point-level refinement for precise registration. Third, we employ a dynamic training mechanism with gradient normalization to balance losses for feature alignment, correspondence refinement, and pose estimation. Extensive experiments demonstrate that CrossI2P outperforms state-of-the-art methods by 23.7% on the KITTI Odometry benchmark and by 37.9% on nuScenes, significantly improving both accuracy and robustness. 

**Abstract (ZH)**: 跨模态2D与3D传感器数据融合对于自主系统中的稳健感知至关重要。然而，由于纹理丰富但深度模糊的图像与稀疏但度量精确的点云之间存在语义-几何差距，以及现有方法倾向于收敛到局部最优的问题，图像到点云（I2P）注册仍然是一个挑战。为克服这些限制，我们提出了一种自监督框架CrossI2P，该框架将跨模态学习和两阶段注册统一于一个端到端管道中。首先，通过双重路径对比学习学习几何-语义融合嵌入空间，实现2D纹理和3D结构的注释免费、双向对齐。其次，采用从粗到细的注册 paradigm：全局阶段通过联合同一模态上下文和跨模态交互建模建立超点-超像素对应关系，随后进行几何约束的点级细化以实现精确注册。最后，采用动态训练机制和梯度规范化来平衡特征对齐、对应关系细化和姿态估计的损失。大量实验表明，CrossI2P在KITTIA载波基准和nuScenes上的性能分别比最先进方法提高了23.7%和37.9%，显著提高了准确性和鲁棒性。 

---
# ChronoForge-RL: Chronological Forging through Reinforcement Learning for Enhanced Video Understanding 

**Title (ZH)**: ChronoForge-RL: 通过强化学习实现的时间顺序锻造以增强视频理解 

**Authors**: Kehua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.15800)  

**Abstract**: Current state-of-the-art video understanding methods typically struggle with two critical challenges: (1) the computational infeasibility of processing every frame in dense video content and (2) the difficulty in identifying semantically significant frames through naive uniform sampling strategies. In this paper, we propose a novel video understanding framework, called ChronoForge-RL, which combines Temporal Apex Distillation (TAD) and KeyFrame-aware Group Relative Policy Optimization (KF-GRPO) to tackle these issues. Concretely, we introduce a differentiable keyframe selection mechanism that systematically identifies semantic inflection points through a three-stage process to enhance computational efficiency while preserving temporal information. Then, two particular modules are proposed to enable effective temporal reasoning: Firstly, TAD leverages variation scoring, inflection detection, and prioritized distillation to select the most informative frames. Secondly, we introduce KF-GRPO which implements a contrastive learning paradigm with a saliency-enhanced reward mechanism that explicitly incentivizes models to leverage both frame content and temporal relationships. Finally, our proposed ChronoForge-RL achieves 69.1% on VideoMME and 52.7% on LVBench compared to baseline methods, clearly surpassing previous approaches while enabling our 7B parameter model to achieve performance comparable to 72B parameter alternatives. 

**Abstract (ZH)**: 当前最先进的视频理解方法通常面临两个关键挑战：（1）密集视频内容中每帧的处理计算上不可行，（2）通过简单的均匀采样策略难以识别具有语义意义的帧。本文提出了一种新颖的视频理解框架，名为ChronoForge-RL，该框架结合了时间峰点精炼（TAD）和关键帧意识组相对策略优化（KF-GRPO），以解决这些问题。具体地，我们引入了一种可微的关键帧选择机制，通过三阶段过程系统地识别语义拐点，以提高计算效率并保留时间信息。然后提出了两个模块以实现有效的时序推理：首先，TAD利用变异评分、拐点检测和优先精炼来选择最具信息量的帧。其次，我们引入了KF-GRPO，这是一种增强显著性奖励机制的对比学习框架，明确激励模型利用帧内容和时序关系。最后，我们提出的ChronoForge-RL在VideoMME上取得了69.1%的成绩，在LVBench上取得了52.7%的成绩，显著超越了基线方法，同时使我们的7B参数模型达到与72B参数模型相近的性能。 

---
# Ideal Registration? Segmentation is All You Need 

**Title (ZH)**: 理想的配准？分割即可。 

**Authors**: Xiang Chen, Fengting Zhang, Qinghao Liu, Min Liu, Kun Wu, Yaonan Wang, Hang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15784)  

**Abstract**: Deep learning has revolutionized image registration by its ability to handle diverse tasks while achieving significant speed advantages over conventional approaches. Current approaches, however, often employ globally uniform smoothness constraints that fail to accommodate the complex, regionally varying deformations characteristic of anatomical motion. To address this limitation, we propose SegReg, a Segmentation-driven Registration framework that implements anatomically adaptive regularization by exploiting region-specific deformation patterns. Our SegReg first decomposes input moving and fixed images into anatomically coherent subregions through segmentation. These localized domains are then processed by the same registration backbone to compute optimized partial deformation fields, which are subsequently integrated into a global deformation field. SegReg achieves near-perfect structural alignment (98.23% Dice on critical anatomies) using ground-truth segmentation, and outperforms existing methods by 2-12% across three clinical registration scenarios (cardiac, abdominal, and lung images) even with automatic segmentation. Our SegReg demonstrates a near-linear dependence of registration accuracy on segmentation quality, transforming the registration challenge into a segmentation problem. The source code will be released upon manuscript acceptance. 

**Abstract (ZH)**: 深度学习通过其处理多样任务的能力和相对于传统方法的显著速度优势，已革命性地改变了图像配准。然而，当前的方法往往采用全局均匀的光滑性约束，无法适应解剖运动中复杂的、区域变化的变形特征。为解决这一问题，我们提出了一种基于分割的配准框架SegReg，该框架通过利用区域特异性变形模式实现解剖学适应性正则化。SegReg首先通过分割将输入的移动和固定图像分解为解剖学一致的子区域，然后使用相同的配准骨干计算优化的部分变形场，并将其整合到全局变形场中。使用地面真相分割，SegReg实现了接近完美的结构对齐（关键解剖部位的Dice系数为98.23%），并在心脏、腹部和肺部图像的三个临床配准场景中表现出色，即使使用自动分割也比现有方法高出2-12%。SegReg展示了注册精度与分割质量近乎线性相关的关系，将配准挑战转化为分割问题。文章接受后将开放源代码。 

---
# FloorSAM: SAM-Guided Floorplan Reconstruction with Semantic-Geometric Fusion 

**Title (ZH)**: FloorSAM：带有语义几何融合的SAM引导式楼层平面图重构 

**Authors**: Han Ye, Haofu Wang, Yunchi Zhang, Jiangjian Xiao, Yuqiang Jin, Jinyuan Liu, Wen-An Zhang, Uladzislau Sychou, Alexander Tuzikov, Vladislav Sobolevskii, Valerii Zakharov, Boris Sokolov, Minglei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15750)  

**Abstract**: Reconstructing building floor plans from point cloud data is key for indoor navigation, BIM, and precise measurements. Traditional methods like geometric algorithms and Mask R-CNN-based deep learning often face issues with noise, limited generalization, and loss of geometric details. We propose FloorSAM, a framework that integrates point cloud density maps with the Segment Anything Model (SAM) for accurate floor plan reconstruction from LiDAR data. Using grid-based filtering, adaptive resolution projection, and image enhancement, we create robust top-down density maps. FloorSAM uses SAM's zero-shot learning for precise room segmentation, improving reconstruction across diverse layouts. Room masks are generated via adaptive prompt points and multistage filtering, followed by joint mask and point cloud analysis for contour extraction and regularization. This produces accurate floor plans and recovers room topological relationships. Tests on Giblayout and ISPRS datasets show better accuracy, recall, and robustness than traditional methods, especially in noisy and complex settings. Code and materials: this http URL. 

**Abstract (ZH)**: 从点云数据重建建筑物楼层平面图对于室内导航、BIM和精确测量至关重要。传统的几何算法和基于Mask R-CNN的深度学习方法常面临噪声问题、泛化能力有限以及几何细节损失等挑战。我们提出了FloorSAM框架，该框架将点云密度图与Segment Anything Model (SAM) 结合，用于从LiDAR数据中准确重建楼层平面图。通过基于网格的过滤、自适应分辨率投影和图像增强，我们生成了稳健的顶部密度图。FloorSAM利用SAM的零样本学习进行精确的房间分割，从而在多样化的布局中提高重建精度。通过自适应提示点和多阶段过滤生成房间掩码，随后进行联合掩码和点云分析以提取轮廓并进行正则化。这产出准确的楼层平面图，并恢复了房间的拓扑关系。在Giblayout和ISPRS数据集上的测试结果表明，FloorSAM在噪声和复杂环境中表现优于传统方法，在准确率和鲁棒性方面尤为突出。代码和材料：this http URL。 

---
# GP3: A 3D Geometry-Aware Policy with Multi-View Images for Robotic Manipulation 

**Title (ZH)**: GP3: 一种基于三维几何的多视图图像机器人操纵策略 

**Authors**: Quanhao Qian, Guoyang Zhao, Gongjie Zhang, Jiuniu Wang, Ran Xu, Junlong Gao, Deli Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.15733)  

**Abstract**: Effective robotic manipulation relies on a precise understanding of 3D scene geometry, and one of the most straightforward ways to acquire such geometry is through multi-view observations. Motivated by this, we present GP3 -- a 3D geometry-aware robotic manipulation policy that leverages multi-view input. GP3 employs a spatial encoder to infer dense spatial features from RGB observations, which enable the estimation of depth and camera parameters, leading to a compact yet expressive 3D scene representation tailored for manipulation. This representation is fused with language instructions and translated into continuous actions via a lightweight policy head. Comprehensive experiments demonstrate that GP3 consistently outperforms state-of-the-art methods on simulated benchmarks. Furthermore, GP3 transfers effectively to real-world robots without depth sensors or pre-mapped environments, requiring only minimal fine-tuning. These results highlight GP3 as a practical, sensor-agnostic solution for geometry-aware robotic manipulation. 

**Abstract (ZH)**: 一种通过多视图输入实现三维几何感知的机器人 manipulation 策略：GP3 

---
# SGMAGNet: A Baseline Model for 3D Cloud Phase Structure Reconstruction on a New Passive Active Satellite Benchmark 

**Title (ZH)**: SGMAGNet：新被动主动卫星基准上三维云相结构重构的基线模型 

**Authors**: Chi Yang, Fu Wang, Xiaofei Yang, Hao Huang, Weijia Cao, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15706)  

**Abstract**: Cloud phase profiles are critical for numerical weather prediction (NWP), as they directly affect radiative transfer and precipitation processes. In this study, we present a benchmark dataset and a baseline framework for transforming multimodal satellite observations into detailed 3D cloud phase structures, aiming toward operational cloud phase profile retrieval and future integration with NWP systems to improve cloud microphysics parameterization. The multimodal observations consist of (1) high--spatiotemporal--resolution, multi-band visible (VIS) and thermal infrared (TIR) imagery from geostationary satellites, and (2) accurate vertical cloud phase profiles from spaceborne lidar (CALIOP\slash CALIPSO) and radar (CPR\slash CloudSat). The dataset consists of synchronized image--profile pairs across diverse cloud regimes, defining a supervised learning task: given VIS/TIR patches, predict the corresponding 3D cloud phase structure. We adopt SGMAGNet as the main model and compare it with several baseline architectures, including UNet variants and SegNet, all designed to capture multi-scale spatial patterns. Model performance is evaluated using standard classification metrics, including Precision, Recall, F1-score, and IoU. The results demonstrate that SGMAGNet achieves superior performance in cloud phase reconstruction, particularly in complex multi-layer and boundary transition regions. Quantitatively, SGMAGNet attains a Precision of 0.922, Recall of 0.858, F1-score of 0.763, and an IoU of 0.617, significantly outperforming all baselines across these key metrics. 

**Abstract (ZH)**: 云相态剖面对于数值天气预测（NWP）至关重要，因为它们直接影响辐射传输和降水过程。在本研究中，我们提供了一个基准数据集和一个基础框架，用于将多模态卫星观测转换为详细的三维云相态结构，旨在实现操作性的云相态剖面提取，并为进一步与NWP系统集成以改善云微物理参数化打下基础。 

---
# Saccadic Vision for Fine-Grained Visual Classification 

**Title (ZH)**: 凝视视知觉细粒度视觉分类 

**Authors**: Johann Schmidt, Sebastian Stober, Joachim Denzler, Paul Bodesheim  

**Link**: [PDF](https://arxiv.org/pdf/2509.15688)  

**Abstract**: Fine-grained visual classification (FGVC) requires distinguishing between visually similar categories through subtle, localized features - a task that remains challenging due to high intra-class variability and limited inter-class differences. Existing part-based methods often rely on complex localization networks that learn mappings from pixel to sample space, requiring a deep understanding of image content while limiting feature utility for downstream tasks. In addition, sampled points frequently suffer from high spatial redundancy, making it difficult to quantify the optimal number of required parts. Inspired by human saccadic vision, we propose a two-stage process that first extracts peripheral features (coarse view) and generates a sample map, from which fixation patches are sampled and encoded in parallel using a weight-shared encoder. We employ contextualized selective attention to weigh the impact of each fixation patch before fusing peripheral and focus representations. To prevent spatial collapse - a common issue in part-based methods - we utilize non-maximum suppression during fixation sampling to eliminate redundancy. Comprehensive evaluation on standard FGVC benchmarks (CUB-200-2011, NABirds, Food-101 and Stanford-Dogs) and challenging insect datasets (EU-Moths, Ecuador-Moths and AMI-Moths) demonstrates that our method achieves comparable performance to state-of-the-art approaches while consistently outperforming our baseline encoder. 

**Abstract (ZH)**: 细粒度视觉分类中的外围特征提取与固定点编码方法：基于人类扫视视觉的两阶段过程 

---
# Towards Size-invariant Salient Object Detection: A Generic Evaluation and Optimization Approach 

**Title (ZH)**: 面向大小不变性的显著目标检测：一种通用的评估与优化方法 

**Authors**: Shilong Bao, Qianqian Xu, Feiran Li, Boyu Han, Zhiyong Yang, Xiaochun Cao, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15573)  

**Abstract**: This paper investigates a fundamental yet underexplored issue in Salient Object Detection (SOD): the size-invariant property for evaluation protocols, particularly in scenarios when multiple salient objects of significantly different sizes appear within a single image. We first present a novel perspective to expose the inherent size sensitivity of existing widely used SOD metrics. Through careful theoretical derivations, we show that the evaluation outcome of an image under current SOD metrics can be essentially decomposed into a sum of several separable terms, with the contribution of each term being directly proportional to its corresponding region size. Consequently, the prediction errors would be dominated by the larger regions, while smaller yet potentially more semantically important objects are often overlooked, leading to biased performance assessments and practical degradation. To address this challenge, a generic Size-Invariant Evaluation (SIEva) framework is proposed. The core idea is to evaluate each separable component individually and then aggregate the results, thereby effectively mitigating the impact of size imbalance across objects. Building upon this, we further develop a dedicated optimization framework (SIOpt), which adheres to the size-invariant principle and significantly enhances the detection of salient objects across a broad range of sizes. Notably, SIOpt is model-agnostic and can be seamlessly integrated with a wide range of SOD backbones. Theoretically, we also present generalization analysis of SOD methods and provide evidence supporting the validity of our new evaluation protocols. Finally, comprehensive experiments speak to the efficacy of our proposed approach. The code is available at this https URL. 

**Abstract (ZH)**: 此论文探讨了显著目标检测（SOD）中一个基础但未充分研究的问题：评估协议的大小不变性属性，特别是在单张图像中出现多个显著大小差异的目标场景下。我们首先提出了一种新型视角来揭示现有广泛使用的SOD度量的固有大小敏感性。通过细致的理论推导，我们表明，在当前SOD度量下的图像评估结果可以从根本上分解为多个可分离的项之和，每一项的贡献与其相应区域大小成正比。因此，预测误差将主要由较大的区域主导，而较小但可能更具有语义重要性的对象往往被忽视，导致性能评估偏差和实际性能下降。为了应对这一挑战，我们提出了一种通用的大小不变性评估（SIEva）框架。核心思想是对每个可分离的组成部分分别进行评估，然后聚合结果，从而有效缓解不同对象间大小不平衡的影响。在此基础上，我们进一步开发了一种专门的优化框架（SIOpt），该框架遵循大小不变性原则，并显著提高了在广泛大小范围内的显著目标检测性能。值得注意的是，SIOpt 框架具有模型无感知性，可以无缝集成到各种SOD主干网络中。从理论上讲，我们还对SOD方法的泛化性进行了分析，并提供了支持我们新评估协议有效性的证据。最后，全面的实验证明了我们提出方法的有效性。相关代码可通过以下链接获取：this https URL。 

---
# GUI-ARP: Enhancing Grounding with Adaptive Region Perception for GUI Agents 

**Title (ZH)**: GUI-ARP：增强 grounding 与自适应区域感知的 GUI 代理 

**Authors**: Xianhang Ye, Yiqing Li, Wei Dai, Miancan Liu, Ziyuan Chen, Zhangye Han, Hongbo Min, Jinkui Ren, Xiantao Zhang, Wen Yang, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.15532)  

**Abstract**: Existing GUI grounding methods often struggle with fine-grained localization in high-resolution screenshots. To address this, we propose GUI-ARP, a novel framework that enables adaptive multi-stage inference. Equipped with the proposed Adaptive Region Perception (ARP) and Adaptive Stage Controlling (ASC), GUI-ARP dynamically exploits visual attention for cropping task-relevant regions and adapts its inference strategy, performing a single-stage inference for simple cases and a multi-stage analysis for more complex scenarios. This is achieved through a two-phase training pipeline that integrates supervised fine-tuning with reinforcement fine-tuning based on Group Relative Policy Optimization (GRPO). Extensive experiments demonstrate that the proposed GUI-ARP achieves state-of-the-art performance on challenging GUI grounding benchmarks, with a 7B model reaching 60.8% accuracy on ScreenSpot-Pro and 30.9% on UI-Vision benchmark. Notably, GUI-ARP-7B demonstrates strong competitiveness against open-source 72B models (UI-TARS-72B at 38.1%) and proprietary models. 

**Abstract (ZH)**: 现有的GUI定位方法在高分辨率屏幕截图中的精细定位常常存在困难。为了解决这一问题，我们提出了一种名为GUI-ARP的新颖框架，能够实现自适应多阶段推理。通过所提出的自适应区域感知（ARP）和自适应阶段控制（ASC），GUI-ARP动态利用视觉注意力对任务相关区域进行裁剪，并自适应其推理策略，在简单情况下进行单阶段推理，在复杂情况下进行多阶段分析。这通过一个两阶段训练管道实现，该管道将监督微调与基于组相对策略优化（GRPO）的强化微调相结合。广泛的实验结果表明，提出的GUI-ARP在具有挑战性的GUI定位基准上的性能达到最新水平，7B模型在ScreenSpot-Pro基准上的准确率为60.8%，在UI-Vision基准上的准确率为30.9%。值得注意的是，GUI-ARP-7B在开放源代码72B模型（UI-TARS-72B在38.1%）和专有模型中表现出 strong 竞争力。 

---
# Incorporating Visual Cortical Lateral Connection Properties into CNN: Recurrent Activation and Excitatory-Inhibitory Separation 

**Title (ZH)**: 将视觉皮层横向连接特性融入CNN：递归激活与兴奋性-抑制性分离 

**Authors**: Jin Hyun Park, Cheng Zhang, Yoonsuck Choe  

**Link**: [PDF](https://arxiv.org/pdf/2509.15460)  

**Abstract**: The original Convolutional Neural Networks (CNNs) and their modern updates such as the ResNet are heavily inspired by the mammalian visual system. These models include afferent connections (retina and LGN to the visual cortex) and long-range projections (connections across different visual cortical areas). However, in the mammalian visual system, there are connections within each visual cortical area, known as lateral (or horizontal) connections. These would roughly correspond to connections within CNN feature maps, and this important architectural feature is missing in current CNN models. In this paper, we present how such lateral connections can be modeled within the standard CNN framework, and test its benefits and analyze its emergent properties in relation to the biological visual system. We will focus on two main architectural features of lateral connections: (1) recurrent activation and (2) separation of excitatory and inhibitory connections. We show that recurrent CNN using weight sharing is equivalent to lateral connections, and propose a custom loss function to separate excitatory and inhibitory weights. The addition of these two leads to increased classification accuracy, and importantly, the activation properties and connection properties of the resulting model show properties similar to those observed in the biological visual system. We expect our approach to help align CNN closer to its biological counterpart and better understand the principles of visual cortical computation. 

**Abstract (ZH)**: 基于标准卷积神经网络框架内的横向连接建模及其生物学启发特点研究 

---
# Region-Aware Deformable Convolutions 

**Title (ZH)**: 区域感知可变形卷积 

**Authors**: Abolfazl Saheban Maleki, Maryam Imani  

**Link**: [PDF](https://arxiv.org/pdf/2509.15436)  

**Abstract**: We introduce Region-Aware Deformable Convolution (RAD-Conv), a new convolutional operator that enhances neural networks' ability to adapt to complex image structures. Unlike traditional deformable convolutions, which are limited to fixed quadrilateral sampling areas, RAD-Conv uses four boundary offsets per kernel element to create flexible, rectangular regions that dynamically adjust their size and shape to match image content. This approach allows precise control over the receptive field's width and height, enabling the capture of both local details and long-range dependencies, even with small 1x1 kernels. By decoupling the receptive field's shape from the kernel's structure, RAD-Conv combines the adaptability of attention mechanisms with the efficiency of standard convolutions. This innovative design offers a practical solution for building more expressive and efficient vision models, bridging the gap between rigid convolutional architectures and computationally costly attention-based methods. 

**Abstract (ZH)**: 区域感知可变形卷积 (RAD-Conv): 一种新的卷积算子，增强神经网络适应复杂图像结构的能力 

---
# Recent Advancements in Microscopy Image Enhancement using Deep Learning: A Survey 

**Title (ZH)**: Recent Advancements in Microscopy Image Enhancement Using Deep Learning: A Survey 

**Authors**: Debasish Dutta, Neeharika Sonowal, Risheraj Barauh, Deepjyoti Chetia, Sanjib Kr Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2509.15363)  

**Abstract**: Microscopy image enhancement plays a pivotal role in understanding the details of biological cells and materials at microscopic scales. In recent years, there has been a significant rise in the advancement of microscopy image enhancement, specifically with the help of deep learning methods. This survey paper aims to provide a snapshot of this rapidly growing state-of-the-art method, focusing on its evolution, applications, challenges, and future directions. The core discussions take place around the key domains of microscopy image enhancement of super-resolution, reconstruction, and denoising, with each domain explored in terms of its current trends and their practical utility of deep learning. 

**Abstract (ZH)**: 显微图像增强在生物细胞和材料微观细节理解中发挥着关键作用。近年来，特别是在深度学习方法的帮助下，显微图像增强取得了显著进展。本文综述了这一快速发展的前沿技术，聚焦其演化、应用、挑战和未来方向。核心讨论围绕显微图像增强的超分辨、重建和去噪等关键领域展开，每领域均从当前趋势及其深度学习的实用价值方面进行探讨。 

---
# Emulating Human-like Adaptive Vision for Efficient and Flexible Machine Visual Perception 

**Title (ZH)**: 模拟人类适应性视觉以实现高效灵活的机器视觉感知 

**Authors**: Yulin Wang, Yang Yue, Yang Yue, Huanqian Wang, Haojun Jiang, Yizeng Han, Zanlin Ni, Yifan Pu, Minglei Shi, Rui Lu, Qisen Yang, Andrew Zhao, Zhuofan Xia, Shiji Song, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15333)  

**Abstract**: Human vision is highly adaptive, efficiently sampling intricate environments by sequentially fixating on task-relevant regions. In contrast, prevailing machine vision models passively process entire scenes at once, resulting in excessive resource demands scaling with spatial-temporal input resolution and model size, yielding critical limitations impeding both future advancements and real-world application. Here we introduce AdaptiveNN, a general framework aiming to drive a paradigm shift from 'passive' to 'active, adaptive' vision models. AdaptiveNN formulates visual perception as a coarse-to-fine sequential decision-making process, progressively identifying and attending to regions pertinent to the task, incrementally combining information across fixations, and actively concluding observation when sufficient. We establish a theory integrating representation learning with self-rewarding reinforcement learning, enabling end-to-end training of the non-differentiable AdaptiveNN without additional supervision on fixation locations. We assess AdaptiveNN on 17 benchmarks spanning 9 tasks, including large-scale visual recognition, fine-grained discrimination, visual search, processing images from real driving and medical scenarios, language-driven embodied AI, and side-by-side comparisons with humans. AdaptiveNN achieves up to 28x inference cost reduction without sacrificing accuracy, flexibly adapts to varying task demands and resource budgets without retraining, and provides enhanced interpretability via its fixation patterns, demonstrating a promising avenue toward efficient, flexible, and interpretable computer vision. Furthermore, AdaptiveNN exhibits closely human-like perceptual behaviors in many cases, revealing its potential as a valuable tool for investigating visual cognition. Code is available at this https URL. 

**Abstract (ZH)**: 人类视觉高度适应性强，能够通过依次聚焦于与任务相关的区域来高效地采样复杂的环境。相比之下，当前的机器视觉模型一次性被动处理整个场景，导致资源需求随着输入的空间-时间分辨率和模型规模的增加而过度增加，从而限制了未来的发展和实际应用。我们引入了AdaptiveNN，这是一种旨在从“被动”转变为“主动、自适应”视觉模型的一般框架。AdaptiveNN将视觉感知形式化为从粗到细的逐步决策过程，逐步识别和关注与任务相关的关键区域，逐步在整个注视点之间结合信息，并在收集足够信息后主动结束观察。我们结合了表示学习与自我奖励强化学习的理论，使得非可微的AdaptiveNN可以在不需要额外注视点监督的情况下实现端到端训练。我们在17个基准测试中评估了AdaptiveNN，涵盖9项任务，包括大规模视觉识别、细粒度鉴别、视觉搜索、处理来自真实驾驶和医疗场景的图像、语言驱动的 embodied AI 以及与人类的侧向比较。AdaptiveNN在不牺牲准确性的前提下将推理成本最多减少28倍，灵活适应不同的任务需求和资源预算而无需重新训练，并通过注视模式提供增强的可解释性，展示了高效、灵活和可解释计算机视觉的前景。此外，在许多情况下，AdaptiveNN表现出接近人类的感知行为，表明其作为研究视觉认知有价值的工具的潜力。代码可在以下网址获取。 

---
# Large Vision Models Can Solve Mental Rotation Problems 

**Title (ZH)**: 大型视觉模型可以解决心理旋转问题 

**Authors**: Sebastian Ray Mason, Anders Gjølbye, Phillip Chavarria Højbjerg, Lenka Tětková, Lars Kai Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2509.15271)  

**Abstract**: Mental rotation is a key test of spatial reasoning in humans and has been central to understanding how perception supports cognition. Despite the success of modern vision transformers, it is still unclear how well these models develop similar abilities. In this work, we present a systematic evaluation of ViT, CLIP, DINOv2, and DINOv3 across a range of mental-rotation tasks, from simple block structures similar to those used by Shepard and Metzler to study human cognition, to more complex block figures, three types of text, and photo-realistic objects. By probing model representations layer by layer, we examine where and how these networks succeed. We find that i) self-supervised ViTs capture geometric structure better than supervised ViTs; ii) intermediate layers perform better than final layers; iii) task difficulty increases with rotation complexity and occlusion, mirroring human reaction times and suggesting similar constraints in embedding space representations. 

**Abstract (ZH)**: 视觉变换器在心理旋转任务中的系统性评估：从Shepard和Metzler的人类认知研究到更复杂的结构和场景 

---
# Causal Reasoning Elicits Controllable 3D Scene Generation 

**Title (ZH)**: 因果推理引导可控的3D场景生成 

**Authors**: Shen Chen, Ruiyu Zhao, Jiale Zhou, Zongkai Wu, Jenq-Neng Hwang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.15249)  

**Abstract**: Existing 3D scene generation methods often struggle to model the complex logical dependencies and physical constraints between objects, limiting their ability to adapt to dynamic and realistic environments. We propose CausalStruct, a novel framework that embeds causal reasoning into 3D scene generation. Utilizing large language models (LLMs), We construct causal graphs where nodes represent objects and attributes, while edges encode causal dependencies and physical constraints. CausalStruct iteratively refines the scene layout by enforcing causal order to determine the placement order of objects and applies causal intervention to adjust the spatial configuration according to physics-driven constraints, ensuring consistency with textual descriptions and real-world dynamics. The refined scene causal graph informs subsequent optimization steps, employing a Proportional-Integral-Derivative(PID) controller to iteratively tune object scales and positions. Our method uses text or images to guide object placement and layout in 3D scenes, with 3D Gaussian Splatting and Score Distillation Sampling improving shape accuracy and rendering stability. Extensive experiments show that CausalStruct generates 3D scenes with enhanced logical coherence, realistic spatial interactions, and robust adaptability. 

**Abstract (ZH)**: 现有的3D场景生成方法往往难以建模物体之间的复杂逻辑依赖和物理约束，限制了其对动态和真实环境的适应能力。我们提出了一种名为CausalStruct的新框架，将因果推理融入3D场景生成。利用大型语言模型（LLMs），构建因果图，节点代表物体和属性，边编码因果依赖和物理约束。CausalStruct通过强制执行因果顺序迭代优化场景布局，确定物体的放置顺序，并应用因果干预根据物理驱动的约束调整空间配置，确保与文本描述和现实世界动力学的一致性。经过细化的场景因果图指导后续优化步骤，使用比例-积分-微分（PID）控制器迭代调整物体的尺寸和位置。我们的方法利用文本或图像指导3D场景中物体的放置和布局，通过3D高斯溅射和评分精炼采样提高形状准确性和渲染稳定性。大量实验表明，CausalStruct生成的3D场景在逻辑连贯性、现实空间交互和鲁棒适应性方面得到了增强。 

---
