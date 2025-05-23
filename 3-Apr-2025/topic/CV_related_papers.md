# Virtual Target Trajectory Prediction for Stochastic Targets 

**Title (ZH)**: 随机目标的虚拟目标轨迹预测 

**Authors**: Marc Schneider, Renato Loureiro, Torbjørn Cunis, Walter Fichter  

**Link**: [PDF](https://arxiv.org/pdf/2504.01851)  

**Abstract**: Trajectory prediction of other vehicles is crucial for autonomous vehicles, with applications from missile guidance to UAV collision avoidance. Typically, target trajectories are assumed deterministic, but real-world aerial vehicles exhibit stochastic behavior, such as evasive maneuvers or gliders circling in thermals. This paper uses Conditional Normalizing Flows, an unsupervised Machine Learning technique, to learn and predict the stochastic behavior of targets of guided missiles using trajectory data. The trained model predicts the distribution of future target positions based on initial conditions and parameters of the dynamics. Samples from this distribution are clustered using a time series k-means algorithm to generate representative trajectories, termed virtual targets. The method is fast and target-agnostic, requiring only training data in the form of target trajectories. Thus, it serves as a drop-in replacement for deterministic trajectory predictions in guidance laws and path planning. Simulated scenarios demonstrate the approach's effectiveness for aerial vehicles with random maneuvers, bridging the gap between deterministic predictions and stochastic reality, advancing guidance and control algorithms for autonomous vehicles. 

**Abstract (ZH)**: 其他车辆轨迹预测对于自主车辆至关重要，其应用范围从导弹引导到无人机避碰。通常，目标轨迹被视为确定性的，但实际的空中车辆表现出随机行为，如规避机动或滑翔机在热气流中盘旋。本文使用条件归一化流，这是一种无监督机器学习技术，通过轨迹数据学习并预测引导导弹目标的随机行为。训练后的模型根据初始条件和动力学参数预测未来目标位置的分布。从该分布中采样的数据使用时间序列k-means算法聚类以生成代表性的轨迹，称为虚拟目标。该方法快速且目标无关，仅需目标轨迹形式的训练数据。因此，它可以用作制导律和路径规划中确定性轨迹预测的即插即用替代方案。仿真场景表明，该方法对于具有随机机动的空中车辆有效，填补了确定性预测与随机现实之间的差距，推动了自主车辆的制导和控制算法的发展。 

---
# DF-Calib: Targetless LiDAR-Camera Calibration via Depth Flow 

**Title (ZH)**: DF-Calib: 无需目标的LiDAR-相机标定 via 深度流 

**Authors**: Shu Han, Xubo Zhu, Ji Wu, Ximeng Cai, Wen Yang, Huai Yu, Gui-Song Xia  

**Link**: [PDF](https://arxiv.org/pdf/2504.01416)  

**Abstract**: Precise LiDAR-camera calibration is crucial for integrating these two sensors into robotic systems to achieve robust perception. In applications like autonomous driving, online targetless calibration enables a prompt sensor misalignment correction from mechanical vibrations without extra targets. However, existing methods exhibit limitations in effectively extracting consistent features from LiDAR and camera data and fail to prioritize salient regions, compromising cross-modal alignment robustness. To address these issues, we propose DF-Calib, a LiDAR-camera calibration method that reformulates calibration as an intra-modality depth flow estimation problem. DF-Calib estimates a dense depth map from the camera image and completes the sparse LiDAR projected depth map, using a shared feature encoder to extract consistent depth-to-depth features, effectively bridging the 2D-3D cross-modal gap. Additionally, we introduce a reliability map to prioritize valid pixels and propose a perceptually weighted sparse flow loss to enhance depth flow estimation. Experimental results across multiple datasets validate its accuracy and generalization,with DF-Calib achieving a mean translation error of 0.635cm and rotation error of 0.045 degrees on the KITTI dataset. 

**Abstract (ZH)**: 精确的激光雷达-相机标定对于将这两种传感器集成到机器人系统中以实现稳健的感知至关重要。在自动驾驶等应用中，在线无靶标标定可以及时校正由机械振动引起的传感器错位，而不需额外的目标。然而，现有方法在从激光雷达和相机数据中提取一致特征方面存在局限性，无法优先处理显著区域，从而削弱了跨模态对齐的稳健性。为解决这些问题，我们提出DF-Calib，一种将标定问题重新表述为跨模态深度流估计问题的激光雷达-相机标定方法。DF-Calib从相机图像估计密集深度图，并完成稀疏的激光雷达投影深度图，通过共享特征编码器提取一致的深度到深度特征，有效地弥合了2D-3D跨模态差距。此外，我们引入可靠性图来优先处理有效像素，并提出感知加权稀疏流损失以提高深度流估计。跨多个数据集的实验结果验证了其准确性和泛化能力，DF-Calib在KITTI数据集上的平均平移误差为0.635cm，旋转误差为0.045度。 

---
# Pedestrian-Aware Motion Planning for Autonomous Driving in Complex Urban Scenarios 

**Title (ZH)**: 行人aware的自主驾驶在复杂城市场景中的运动规划 

**Authors**: Korbinian Moller, Truls Nyberg, Jana Tumova, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2504.01409)  

**Abstract**: Motion planning in uncertain environments like complex urban areas is a key challenge for autonomous vehicles (AVs). The aim of our research is to investigate how AVs can navigate crowded, unpredictable scenarios with multiple pedestrians while maintaining a safe and efficient vehicle behavior. So far, most research has concentrated on static or deterministic traffic participant behavior. This paper introduces a novel algorithm for motion planning in crowded spaces by combining social force principles for simulating realistic pedestrian behavior with a risk-aware motion planner. We evaluate this new algorithm in a 2D simulation environment to rigorously assess AV-pedestrian interactions, demonstrating that our algorithm enables safe, efficient, and adaptive motion planning, particularly in highly crowded urban environments - a first in achieving this level of performance. This study has not taken into consideration real-time constraints and has been shown only in simulation so far. Further studies are needed to investigate the novel algorithm in a complete software stack for AVs on real cars to investigate the entire perception, planning and control pipeline in crowded scenarios. We release the code developed in this research as an open-source resource for further studies and development. It can be accessed at the following link: this https URL 

**Abstract (ZH)**: 在复杂城市环境中不确定条件下进行运动规划是自动驾驶车辆（AVs）面临的关键挑战。本研究旨在探讨如何使AVs在包含多个行人的拥挤、不可预测场景中安全高效地导航。迄今为止，大多数研究都集中在静态或确定性的交通参与者行为上。本文介绍了一种新的算法，通过结合社会力原则模拟真实的人行行为并与一种风险意识的运动规划算法相结合，以实现拥挤空间中的运动规划。本文在2D仿真环境中评估了该新算法，以严格评估AV与行人之间的交互，证明了我们的算法能够实现安全、高效且适应性强的运动规划，特别是在高度拥挤的城市环境中——这是首次在该水平上达到这一性能。本研究未考虑实时约束，仅在仿真环境中展示。需要进一步的研究将该新型算法应用于完整的AV软件栈中的真实车辆，以研究拥挤场景下的整体验测、规划和控制管道。我们已将在此研究中开发的代码作为开源资源发布，以便进一步研究和开发，访问链接为：this https URL 

---
# From Shadows to Safety: Occlusion Tracking and Risk Mitigation for Urban Autonomous Driving 

**Title (ZH)**: 从阴影到安全：城市自主驾驶中的遮挡跟踪与风险缓解 

**Authors**: Korbinian Moller, Luis Schwarzmeier, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2504.01408)  

**Abstract**: Autonomous vehicles (AVs) must navigate dynamic urban environments where occlusions and perception limitations introduce significant uncertainties. This research builds upon and extends existing approaches in risk-aware motion planning and occlusion tracking to address these challenges. While prior studies have developed individual methods for occlusion tracking and risk assessment, a comprehensive method integrating these techniques has not been fully explored. We, therefore, enhance a phantom agent-centric model by incorporating sequential reasoning to track occluded areas and predict potential hazards. Our model enables realistic scenario representation and context-aware risk evaluation by modeling diverse phantom agents, each with distinct behavior profiles. Simulations demonstrate that the proposed approach improves situational awareness and balances proactive safety with efficient traffic flow. While these results underline the potential of our method, validation in real-world scenarios is necessary to confirm its feasibility and generalizability. By utilizing and advancing established methodologies, this work contributes to safer and more reliable AV planning in complex urban environments. To support further research, our method is available as open-source software at: this https URL 

**Abstract (ZH)**: 自主驾驶车辆（AVs）必须导航动态的城市环境，其中遮挡和感知限制引入了显著的不确定性。本研究在风险意识运动规划和遮挡跟踪的现有方法基础上进行扩展，以应对这些挑战。尽管先前的研究开发了单独的遮挡跟踪和风险评估方法，但将这些技术进行全面整合的方法尚未得到充分探索。因此，我们通过引入序列推理来增强基于幽灵代理的模型，以跟踪遮挡区域并预测潜在的危险。我们的模型通过建模具有不同行为特征的多样幽灵代理，实现了现实场景的准确建模和情境感知风险评估。模拟结果表明，所提出的方法可以提高情景意识并平衡积极的安全性和高效的交通流量。尽管这些结果突显了我们方法的潜力，但在实际场景中的验证仍是必要的，以确认其可行性和普适性。通过利用和推进现有方法，本研究为复杂城市环境中的自主驾驶车辆规划的安全性和可靠性做出了贡献。为了支持进一步的研究，我们的方法已作为开源软件提供：this https URL。 

---
# ForestVO: Enhancing Visual Odometry in Forest Environments through ForestGlue 

**Title (ZH)**: ForestVO: 通过ForestGlue增强森林环境下的视觉里程计 

**Authors**: Thomas Pritchard, Saifullah Ijaz, Ronald Clark, Basaran Bahadir Kocer  

**Link**: [PDF](https://arxiv.org/pdf/2504.01261)  

**Abstract**: Recent advancements in visual odometry systems have improved autonomous navigation; however, challenges persist in complex environments like forests, where dense foliage, variable lighting, and repetitive textures compromise feature correspondence accuracy. To address these challenges, we introduce ForestGlue, enhancing the SuperPoint feature detector through four configurations - grayscale, RGB, RGB-D, and stereo-vision - optimised for various sensing modalities. For feature matching, we employ LightGlue or SuperGlue, retrained with synthetic forest data. ForestGlue achieves comparable pose estimation accuracy to baseline models but requires only 512 keypoints - just 25% of the baseline's 2048 - to reach an LO-RANSAC AUC score of 0.745 at a 10° threshold. With only a quarter of keypoints needed, ForestGlue significantly reduces computational overhead, demonstrating effectiveness in dynamic forest environments, and making it suitable for real-time deployment on resource-constrained platforms. By combining ForestGlue with a transformer-based pose estimation model, we propose ForestVO, which estimates relative camera poses using matched 2D pixel coordinates between frames. On challenging TartanAir forest sequences, ForestVO achieves an average relative pose error (RPE) of 1.09 m and a kitti_score of 2.33%, outperforming direct-based methods like DSO by 40% in dynamic scenes. Despite using only 10% of the dataset for training, ForestVO maintains competitive performance with TartanVO while being a significantly lighter model. This work establishes an end-to-end deep learning pipeline specifically tailored for visual odometry in forested environments, leveraging forest-specific training data to optimise feature correspondence and pose estimation, thereby enhancing the accuracy and robustness of autonomous navigation systems. 

**Abstract (ZH)**: Recent advancements in视觉里程计系统最近在视觉里程计系统方面的进展已经提高了自主导航的能力；然而，在森林等复杂环境中仍然存在挑战，其中茂密的植被、多变的光照和重复的纹理损害了特征对应准确性。为了应对这些挑战，我们引入了ForestGlue，通过针对各种传感模态优化的灰度、RGB、RGB-D和立体视觉四种配置增强SuperPoint特征检测器。在特征匹配中，我们使用LightGlue或SuperGlue，并重新训练以适应森林合成数据。ForestGlue在姿态估计准确性上与基线模型相当，但在10°阈值下达到LO-RANSAC AUC分数0.745时，仅需512个关键点，这仅为基线模型所需关键点数的一半（2048个的关键点的25%）。由于只需要四分之一的关键点，ForestGlue显著减少了计算开销，证明了其在动态森林环境中的有效性和适用性，使其适合在资源受限的平台上进行实时部署。通过将ForestGlue与基于变压器的姿态估计模型结合，我们提出了ForestVO，该模型使用帧间匹配的2D像素坐标来估计相对相机姿态。在TartanAir森林序列中，ForestVO实现了平均相对姿态误差（RPE）为1.09米和kitti_score为2.33%，在动态场景中优于直接方法如DSO 40%。尽管仅使用了数据集的10%进行训练，ForestVO仍然保持了与TartanVO相当的竞争性能，但模型更加轻量级。本工作建立了一个针对森林环境视觉里程计的端到端深度学习管道，利用森林特定的训练数据来优化特征对应和姿态估计，从而提高了自主导航系统的准确性和鲁棒性。 

---
# Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness 

**Title (ZH)**: Ross3D: 带有3D意识的生成性视觉指令调优 

**Authors**: Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01901)  

**Abstract**: The rapid development of Large Multimodal Models (LMMs) for 2D images and videos has spurred efforts to adapt these models for interpreting 3D scenes. However, the absence of large-scale 3D vision-language datasets has posed a significant obstacle. To address this issue, typical approaches focus on injecting 3D awareness into 2D LMMs by designing 3D input-level scene representations. This work provides a new perspective. We introduce reconstructive visual instruction tuning with 3D-awareness (Ross3D), which integrates 3D-aware visual supervision into the training procedure. Specifically, it incorporates cross-view and global-view reconstruction. The former requires reconstructing masked views by aggregating overlapping information from other views. The latter aims to aggregate information from all available views to recover Bird's-Eye-View images, contributing to a comprehensive overview of the entire scene. Empirically, Ross3D achieves state-of-the-art performance across various 3D scene understanding benchmarks. More importantly, our semi-supervised experiments demonstrate significant potential in leveraging large amounts of unlabeled 3D vision-only data. 

**Abstract (ZH)**: 大型多模态模型（LMMs）在2D图像和视频上的快速发展促进了对3D场景解释的努力。然而，缺乏大规模的3D视觉-语言数据集构成了一个重大障碍。为了解决这一问题，典型的方法着重于通过设计3D输入级场景表示将3D意识注入2D LMMs。本文提供了新的视角。我们引入了具有3D意识的重构视觉指令调优（Ross3D），该方法将3D意识的视觉监督整合到训练过程中。具体而言，它结合了跨视图和全局视图的重构。前者需要通过聚合其他视图的重叠信息来重建被遮掩的视图。后者旨在从所有可用视图中聚合信息以恢复鸟瞰图图像，从而提供整个场景的全面概述。实验结果表明，Ross3D在各种3D场景理解基准测试中取得了最先进的性能。更为重要的是，我们的半监督实验展示了利用大量未标记的3D视觉数据的巨大潜力。 

---
# Overlap-Aware Feature Learning for Robust Unsupervised Domain Adaptation for 3D Semantic Segmentation 

**Title (ZH)**: 重叠感知特征学习：for robust unsupervised domain adaptation in 3D semantic segmentation 

**Authors**: Junjie Chen, Yuecong Xu, Haosheng Li, Kemi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.01668)  

**Abstract**: 3D point cloud semantic segmentation (PCSS) is a cornerstone for environmental perception in robotic systems and autonomous driving, enabling precise scene understanding through point-wise classification. While unsupervised domain adaptation (UDA) mitigates label scarcity in PCSS, existing methods critically overlook the inherent vulnerability to real-world perturbations (e.g., snow, fog, rain) and adversarial distortions. This work first identifies two intrinsic limitations that undermine current PCSS-UDA robustness: (a) unsupervised features overlap from unaligned boundaries in shared-class regions and (b) feature structure erosion caused by domain-invariant learning that suppresses target-specific patterns. To address the proposed problems, we propose a tripartite framework consisting of: 1) a robustness evaluation model quantifying resilience against adversarial attack/corruption types through robustness metrics; 2) an invertible attention alignment module (IAAM) enabling bidirectional domain mapping while preserving discriminative structure via attention-guided overlap suppression; and 3) a contrastive memory bank with quality-aware contrastive learning that progressively refines pseudo-labels with feature quality for more discriminative representations. Extensive experiments on SynLiDAR-to-SemanticPOSS adaptation demonstrate a maximum mIoU improvement of 14.3\% under adversarial attack. 

**Abstract (ZH)**: 3D点云语义分割的鲁棒无监督领域自适应方法 

---
# FUSION: Frequency-guided Underwater Spatial Image recOnstructioN 

**Title (ZH)**: 频谱引导的水下空间图像重构 

**Authors**: Jaskaran Singh Walia, Shravan Venkatraman, Pavithra LK  

**Link**: [PDF](https://arxiv.org/pdf/2504.01243)  

**Abstract**: Underwater images suffer from severe degradations, including color distortions, reduced visibility, and loss of structural details due to wavelength-dependent attenuation and scattering. Existing enhancement methods primarily focus on spatial-domain processing, neglecting the frequency domain's potential to capture global color distributions and long-range dependencies. To address these limitations, we propose FUSION, a dual-domain deep learning framework that jointly leverages spatial and frequency domain information. FUSION independently processes each RGB channel through multi-scale convolutional kernels and adaptive attention mechanisms in the spatial domain, while simultaneously extracting global structural information via FFT-based frequency attention. A Frequency Guided Fusion module integrates complementary features from both domains, followed by inter-channel fusion and adaptive channel recalibration to ensure balanced color distributions. Extensive experiments on benchmark datasets (UIEB, EUVP, SUIM-E) demonstrate that FUSION achieves state-of-the-art performance, consistently outperforming existing methods in reconstruction fidelity (highest PSNR of 23.717 dB and SSIM of 0.883 on UIEB), perceptual quality (lowest LPIPS of 0.112 on UIEB), and visual enhancement metrics (best UIQM of 3.414 on UIEB), while requiring significantly fewer parameters (0.28M) and lower computational complexity, demonstrating its suitability for real-time underwater imaging applications. 

**Abstract (ZH)**: 基于频域的 underwater 图像融合增强框架 FUSION 

---
# Coarse-to-Fine Learning for Multi-Pipette Localisation in Robot-Assisted In Vivo Patch-Clamp 

**Title (ZH)**: 从粗到细学习在机器人辅助在体膜片钳实验中多通道定位 

**Authors**: Lan Wei, Gema Vera Gonzalez, Phatsimo Kgwarae, Alexander Timms, Denis Zahorovsky, Simon Schultz, Dandan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01044)  

**Abstract**: In vivo image-guided multi-pipette patch-clamp is essential for studying cellular interactions and network dynamics in neuroscience. However, current procedures mainly rely on manual expertise, which limits accessibility and scalability. Robotic automation presents a promising solution, but achieving precise real-time detection of multiple pipettes remains a challenge. Existing methods focus on ex vivo experiments or single pipette use, making them inadequate for in vivo multi-pipette scenarios. To address these challenges, we propose a heatmap-augmented coarse-to-fine learning technique to facilitate multi-pipette real-time localisation for robot-assisted in vivo patch-clamp. More specifically, we introduce a Generative Adversarial Network (GAN)-based module to remove background noise and enhance pipette visibility. We then introduce a two-stage Transformer model that starts with predicting the coarse heatmap of the pipette tips, followed by the fine-grained coordination regression module for precise tip localisation. To ensure robust training, we use the Hungarian algorithm for optimal matching between the predicted and actual locations of tips. Experimental results demonstrate that our method achieved > 98% accuracy within 10 {\mu}m, and > 89% accuracy within 5 {\mu}m for the localisation of multi-pipette tips. The average MSE is 2.52 {\mu}m. 

**Abstract (ZH)**: 基于热图增强的粗细粒度学习方法在机器人辅助在体多电极存留电记录中的实时多电极定位 

---
# Gaze-Guided 3D Hand Motion Prediction for Detecting Intent in Egocentric Grasping Tasks 

**Title (ZH)**: 基于凝视引导的3D手部运动预测以检测第一人称抓取任务中的意图 

**Authors**: Yufei He, Xucong Zhang, Arno H. A. Stienen  

**Link**: [PDF](https://arxiv.org/pdf/2504.01024)  

**Abstract**: Human intention detection with hand motion prediction is critical to drive the upper-extremity assistive robots in neurorehabilitation applications. However, the traditional methods relying on physiological signal measurement are restrictive and often lack environmental context. We propose a novel approach that predicts future sequences of both hand poses and joint positions. This method integrates gaze information, historical hand motion sequences, and environmental object data, adapting dynamically to the assistive needs of the patient without prior knowledge of the intended object for grasping. Specifically, we use a vector-quantized variational autoencoder for robust hand pose encoding with an autoregressive generative transformer for effective hand motion sequence prediction. We demonstrate the usability of these novel techniques in a pilot study with healthy subjects. To train and evaluate the proposed method, we collect a dataset consisting of various types of grasp actions on different objects from multiple subjects. Through extensive experiments, we demonstrate that the proposed method can successfully predict sequential hand movement. Especially, the gaze information shows significant enhancements in prediction capabilities, particularly with fewer input frames, highlighting the potential of the proposed method for real-world applications. 

**Abstract (ZH)**: 基于手部运动预测的人类意图检测对神经康复应用中的上肢辅助机器人至关重要。然而，依赖生理信号测量的传统方法限制性较大，往往缺乏环境上下文。我们提出了一种新颖的方法，预测未来的手部姿态和关节位置序列。该方法整合了视点信息、历史手部运动序列以及环境物体数据，能够动态适应患者的辅助需求，无需事先知道抓取物体的意图。具体而言，我们使用向量量化变分自编码器进行稳健的手部姿态编码，并使用自回归生成变压器进行有效的手部运动序列预测。我们在健康受试者的初步研究中展示了这些新技术的实用性。为训练和评估所提出的方法，我们收集了一个数据集，该数据集包含来自多个受试者的不同物体的多种类型抓取动作。通过广泛的实验，我们证明了所提出的方法可以成功预测序列手部运动。特别是，视点信息在较少输入帧的情况下显著提升了预测能力，突显了所提出方法在实际应用中的潜力。 

---
# Omnidirectional Depth-Aided Occupancy Prediction based on Cylindrical Voxel for Autonomous Driving 

**Title (ZH)**: 基于圆柱体体素的全方位深度辅助占用预测 Autonomous Driving 

**Authors**: Chaofan Wu, Jiaheng Li, Jinghao Cao, Ming Li, Yongkang Feng, Jiayu Wu Shuwen Xu, Zihang Gao, Sidan Du, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.01023)  

**Abstract**: Accurate 3D perception is essential for autonomous driving. Traditional methods often struggle with geometric ambiguity due to a lack of geometric prior. To address these challenges, we use omnidirectional depth estimation to introduce geometric prior. Based on the depth information, we propose a Sketch-Coloring framework OmniDepth-Occ. Additionally, our approach introduces a cylindrical voxel representation based on polar coordinate to better align with the radial nature of panoramic camera views. To address the lack of fisheye camera dataset in autonomous driving tasks, we also build a virtual scene dataset with six fisheye cameras, and the data volume has reached twice that of SemanticKITTI. Experimental results demonstrate that our Sketch-Coloring network significantly enhances 3D perception performance. 

**Abstract (ZH)**: 准确的三维感知对于自动驾驶至关重要。传统方法由于缺乏几何先验常常难以应对几何歧义。为应对这些挑战，我们采用全向深度估计引入几何先验。基于深度信息，我们提出了一种素描着色框架 OmniDepth-Occ。此外，我们的方法引入了一种基于极坐标的空间体素表示，以更好地与全景相机视图的径向特性相匹配。为了解决自动驾驶任务中鱼眼相机数据集的缺乏，我们还构建了一个包含六个鱼眼相机的虚拟场景数据集，数据量达到了SemanticKITTI的两倍。实验结果表明，我们的素描着色网络显著提升了三维感知性能。 

---
# Equivariant Spherical CNNs for Accurate Fiber Orientation Distribution Estimation in Neonatal Diffusion MRI with Reduced Acquisition Time 

**Title (ZH)**: 可用于减少采集时间的胎儿弥散MRI中纤维 orientations分布准确估计的共变ariant球CNN方法 

**Authors**: Haykel Snoussi, Davood Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01925)  

**Abstract**: Early and accurate assessment of brain microstructure using diffusion Magnetic Resonance Imaging (dMRI) is crucial for identifying neurodevelopmental disorders in neonates, but remains challenging due to low signal-to-noise ratio (SNR), motion artifacts, and ongoing myelination. In this study, we propose a rotationally equivariant Spherical Convolutional Neural Network (sCNN) framework tailored for neonatal dMRI. We predict the Fiber Orientation Distribution (FOD) from multi-shell dMRI signals acquired with a reduced set of gradient directions (30% of the full protocol), enabling faster and more cost-effective acquisitions. We train and evaluate the performance of our sCNN using real data from 43 neonatal dMRI datasets provided by the Developing Human Connectome Project (dHCP). Our results demonstrate that the sCNN achieves significantly lower mean squared error (MSE) and higher angular correlation coefficient (ACC) compared to a Multi-Layer Perceptron (MLP) baseline, indicating improved accuracy in FOD estimation. Furthermore, tractography results based on the sCNN-predicted FODs show improved anatomical plausibility, coverage, and coherence compared to those from the MLP. These findings highlight that sCNNs, with their inherent rotational equivariance, offer a promising approach for accurate and clinically efficient dMRI analysis, paving the way for improved diagnostic capabilities and characterization of early brain development. 

**Abstract (ZH)**: 使用扩散磁共振成像（dMRI）早期准确评估新生儿脑微结构对于识别神经发育障碍至关重要，但由于信噪比低、运动伪影和持续髓鞘形成，这一任务仍具有挑战性。在本研究中，我们提出了一种旋转不变的球面卷积神经网络（sCNN）框架，专门用于新生儿dMRI。我们从减少了梯度方向数量的多壳层dMRI信号（仅为完整协议的30%）中预测纤维 Orientation Distribution（FOD），从而实现更快、更经济的数据采集。我们利用开发人类连接组项目（dHCP）提供的43个新生儿dMRI数据集，训练并评估了sCNN的性能。结果表明，与多层感知机（MLP）基线相比，sCNN在FOD估计中实现了显著更低的均方误差（MSE）和更高的角度相关系数（ACC），表明FOD估计准确性更高。此外，基于sCNN预测的FOD进行的追踪图结果在解剖学合理性、覆盖范围和连贯性方面均优于MLP结果。这些发现表明，由于其固有的旋转不变性，sCNN为准确且临床有效的dMRI分析提供了一种有前途的方法，为提高诊断能力和早期脑发育表征铺平了道路。 

---
# Implicit Bias Injection Attacks against Text-to-Image Diffusion Models 

**Title (ZH)**: 面向文本到图像扩散模型的隐式偏见注入攻击 

**Authors**: Huayang Huang, Xiangye Jin, Jiaxu Miao, Yu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01819)  

**Abstract**: The proliferation of text-to-image diffusion models (T2I DMs) has led to an increased presence of AI-generated images in daily life. However, biased T2I models can generate content with specific tendencies, potentially influencing people's perceptions. Intentional exploitation of these biases risks conveying misleading information to the public. Current research on bias primarily addresses explicit biases with recognizable visual patterns, such as skin color and gender. This paper introduces a novel form of implicit bias that lacks explicit visual features but can manifest in diverse ways across various semantic contexts. This subtle and versatile nature makes this bias challenging to detect, easy to propagate, and adaptable to a wide range of scenarios. We further propose an implicit bias injection attack framework (IBI-Attacks) against T2I diffusion models by precomputing a general bias direction in the prompt embedding space and adaptively adjusting it based on different inputs. Our attack module can be seamlessly integrated into pre-trained diffusion models in a plug-and-play manner without direct manipulation of user input or model retraining. Extensive experiments validate the effectiveness of our scheme in introducing bias through subtle and diverse modifications while preserving the original semantics. The strong concealment and transferability of our attack across various scenarios further underscore the significance of our approach. Code is available at this https URL. 

**Abstract (ZH)**: 文本到图像扩散模型（T2I DMs）的发展增加了人工智能生成图像在日常生活中的出现。然而，有偏见的T2I模型可以生成具有特定倾向的内容，可能影响人们的感知。故意利用这些偏见可能会向公众传达误导性信息。当前关于偏见的研究主要关注有明确视觉特征的显式偏见，如肤色和性别。本文介绍了一种新型的隐性偏见，这种偏见缺乏明确的视觉特征，可以在各种语义上下文中以多种方式表现。这种微妙且多变的性质使得这种偏见难以检测，容易传播，并且能够适应广泛的场景。我们进一步提出了一种针对T2I扩散模型的隐性偏见注入攻击框架（IBI-Attacks），通过在提示嵌入空间中预先计算一个通用的偏见方向，并根据不同输入适应性地调整它。我们的攻击模块可以以插件方式无缝集成到预训练的扩散模型中，而无需直接操纵用户输入或重新训练模型。广泛的经验实验证明，我们的方案能够在细微且多样化的修改中引入偏见，同时保留原始语义。我们的攻击跨越各种场景的强大隐蔽性和可传输性进一步突显了我们方法的重要性。代码可在此处访问。 

---
# Dual-stream Transformer-GCN Model with Contextualized Representations Learning for Monocular 3D Human Pose Estimation 

**Title (ZH)**: 带有上下文表示学习的双流 Transformer-GCN 模型在单目三维人体姿态估计中的应用 

**Authors**: Mingrui Ye, Lianping Yang, Hegui Zhu, Zenghao Zheng, Xin Wang, Yantao Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.01764)  

**Abstract**: This paper introduces a novel approach to monocular 3D human pose estimation using contextualized representation learning with the Transformer-GCN dual-stream model. Monocular 3D human pose estimation is challenged by depth ambiguity, limited 3D-labeled training data, imbalanced modeling, and restricted model generalization. To address these limitations, our work introduces a groundbreaking motion pre-training method based on contextualized representation learning. Specifically, our method involves masking 2D pose features and utilizing a Transformer-GCN dual-stream model to learn high-dimensional representations through a self-distillation setup. By focusing on contextualized representation learning and spatial-temporal modeling, our approach enhances the model's ability to understand spatial-temporal relationships between postures, resulting in superior generalization. Furthermore, leveraging the Transformer-GCN dual-stream model, our approach effectively balances global and local interactions in video pose estimation. The model adaptively integrates information from both the Transformer and GCN streams, where the GCN stream effectively learns local relationships between adjacent key points and frames, while the Transformer stream captures comprehensive global spatial and temporal features. Our model achieves state-of-the-art performance on two benchmark datasets, with an MPJPE of 38.0mm and P-MPJPE of 31.9mm on Human3.6M, and an MPJPE of 15.9mm on MPI-INF-3DHP. Furthermore, visual experiments on public datasets and in-the-wild videos demonstrate the robustness and generalization capabilities of our approach. 

**Abstract (ZH)**: 利用Transformer-GCN双重流模型的上下文表示学习的单目3D人体姿态估计新方法 

---
# DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance 

**Title (ZH)**: DreamActor-M1: 全局、表达丰富且稳健的人类图像动画生成方法 

**Authors**: Yuxuan Luo, Zhengkun Rong, Lizhen Wang, Longhao Zhang, Tianshu Hu, Yongming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01724)  

**Abstract**: While recent image-based human animation methods achieve realistic body and facial motion synthesis, critical gaps remain in fine-grained holistic controllability, multi-scale adaptability, and long-term temporal coherence, which leads to their lower expressiveness and robustness. We propose a diffusion transformer (DiT) based framework, DreamActor-M1, with hybrid guidance to overcome these limitations. For motion guidance, our hybrid control signals that integrate implicit facial representations, 3D head spheres, and 3D body skeletons achieve robust control of facial expressions and body movements, while producing expressive and identity-preserving animations. For scale adaptation, to handle various body poses and image scales ranging from portraits to full-body views, we employ a progressive training strategy using data with varying resolutions and scales. For appearance guidance, we integrate motion patterns from sequential frames with complementary visual references, ensuring long-term temporal coherence for unseen regions during complex movements. Experiments demonstrate that our method outperforms the state-of-the-art works, delivering expressive results for portraits, upper-body, and full-body generation with robust long-term consistency. Project Page: this https URL. 

**Abstract (ZH)**: 基于图像的人体动画方法虽然实现了逼真的身体和面部运动合成，但在细粒度的整体可控性、多尺度适应性和长时间时序一致性方面仍存在关键差距，这导致其表现力和鲁棒性较低。我们提出了一种基于扩散变换器的框架DreamActor-M1，结合混合指导信号以克服这些限制。在运动指导方面，我们的混合控制信号结合了隐含的面部表示、3D头部球体和3D身体骨架，实现了面部表情和身体运动的稳健控制，并生成了表现性强且保持身份的动画。在尺度适应方面，为了处理从肖像到全身视图的各种身体姿态和图像尺度，我们采用了使用不同分辨率和尺度数据的逐步训练策略。在外观指导方面，我们结合了序列帧中的运动模式和互补的视觉参考，确保在复杂运动过程中未见区域的时间序列一致性。实验结果显示，我们的方法在肖像、上半身和全身生成方面超越了现有最佳方法，提供了表现力强且长时间一致性稳健的结果。项目页面：这个 https URL。 

---
# Token Pruning in Audio Transformers: Optimizing Performance and Decoding Patch Importance 

**Title (ZH)**: 音频变换器中的Token修剪：优化性能与解码块重要性 

**Authors**: Taehan Lee, Hyukjun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.01690)  

**Abstract**: Vision Transformers (ViTs) have achieved state-of-the-art performance across various computer vision tasks, but their high computational cost remains a challenge. Token pruning has been proposed to reduce this cost by selectively removing less important tokens. While effective in vision tasks by discarding non-object regions, applying this technique to audio tasks presents unique challenges, as distinguishing relevant from irrelevant regions in time-frequency representations is less straightforward. In this study, for the first time, we applied token pruning to ViT-based audio classification models using Mel-spectrograms and analyzed the trade-offs between model performance and computational cost: TopK token pruning can reduce MAC operations of AudioMAE and AST by 30-40%, with less than a 1% drop in classification accuracy. Our analysis reveals that while high-intensity tokens contribute significantly to model accuracy, low-intensity tokens remain important. In particular, they play a more critical role in general audio classification tasks than in speech-specific tasks. 

**Abstract (ZH)**: Vision Transformers (ViTs)在各种计算机视觉任务中达到了最先进的性能，但其高昂的计算成本仍然是一个挑战。通过选择性地移除不重要的标记，标记剪枝已被提出以降低这一成本。虽然在视觉任务中通过丢弃非对象区域有效，但将其应用于音频任务带来了独特的挑战，因为在时频表示中区分相关和不相关区域并不那么简单。在本研究中，我们首次将标记剪枝应用于基于Mel频谱图的ViT音频分类模型，并分析了模型性能与计算成本之间的权衡：TopK标记剪枝可以将AudioMAE和AST的MAC操作减少30-40%，且分类准确率下降不到1%。我们的分析表明，虽然高强度标记对模型精度贡献显著，但低强度标记仍然很重要。特别是，在一般音频分类任务中，它们的作用比在言语特定任务中更重要。 

---
# Bridge 2D-3D: Uncertainty-aware Hierarchical Registration Network with Domain Alignment 

**Title (ZH)**: 桥接2D-3D：具有领域对齐的不确定性感知分层注册网络 

**Authors**: Zhixin Cheng, Jiacheng Deng, Xinjun Li, Baoqun Yin, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01641)  

**Abstract**: The method for image-to-point cloud registration typically determines the rigid transformation using a coarse-to-fine pipeline. However, directly and uniformly matching image patches with point cloud patches may lead to focusing on incorrect noise patches during matching while ignoring key ones. Moreover, due to the significant differences between image and point cloud modalities, it may be challenging to bridge the domain gap without specific improvements in design. To address the above issues, we innovatively propose the Uncertainty-aware Hierarchical Matching Module (UHMM) and the Adversarial Modal Alignment Module (AMAM). Within the UHMM, we model the uncertainty of critical information in image patches and facilitate multi-level fusion interactions between image and point cloud features. In the AMAM, we design an adversarial approach to reduce the domain gap between image and point cloud. Extensive experiments and ablation studies on RGB-D Scene V2 and 7-Scenes benchmarks demonstrate the superiority of our method, making it a state-of-the-art approach for image-to-point cloud registration tasks. 

**Abstract (ZH)**: 基于图像到点云配准的不确定性和层次匹配模块及对抗模态对齐模块研究 

---
# Training-free Dense-Aligned Diffusion Guidance for Modular Conditional Image Synthesis 

**Title (ZH)**: 无需训练密集对齐扩散引导的模块化条件图像合成 

**Authors**: Zixuan Wang, Duo Peng, Feng Chen, Yuwei Yang, Yinjie Lei  

**Link**: [PDF](https://arxiv.org/pdf/2504.01515)  

**Abstract**: Conditional image synthesis is a crucial task with broad applications, such as artistic creation and virtual reality. However, current generative methods are often task-oriented with a narrow scope, handling a restricted condition with constrained applicability. In this paper, we propose a novel approach that treats conditional image synthesis as the modular combination of diverse fundamental condition units. Specifically, we divide conditions into three primary units: text, layout, and drag. To enable effective control over these conditions, we design a dedicated alignment module for each. For the text condition, we introduce a Dense Concept Alignment (DCA) module, which achieves dense visual-text alignment by drawing on diverse textual concepts. For the layout condition, we propose a Dense Geometry Alignment (DGA) module to enforce comprehensive geometric constraints that preserve the spatial configuration. For the drag condition, we introduce a Dense Motion Alignment (DMA) module to apply multi-level motion regularization, ensuring that each pixel follows its desired trajectory without visual artifacts. By flexibly inserting and combining these alignment modules, our framework enhances the model's adaptability to diverse conditional generation tasks and greatly expands its application range. Extensive experiments demonstrate the superior performance of our framework across a variety of conditions, including textual description, segmentation mask (bounding box), drag manipulation, and their combinations. Code is available at this https URL. 

**Abstract (ZH)**: 基于模块化条件单位的图像合成方法 

---
# BiSeg-SAM: Weakly-Supervised Post-Processing Framework for Boosting Binary Segmentation in Segment Anything Models 

**Title (ZH)**: BiSeg-SAM：提升段 Anything 模型二值分割性能的弱监督后处理框架 

**Authors**: Encheng Su, Hu Cao, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2504.01452)  

**Abstract**: Accurate segmentation of polyps and skin lesions is essential for diagnosing colorectal and skin cancers. While various segmentation methods for polyps and skin lesions using fully supervised deep learning techniques have been developed, the pixel-level annotation of medical images by doctors is both time-consuming and costly. Foundational vision models like the Segment Anything Model (SAM) have demonstrated superior performance; however, directly applying SAM to medical segmentation may not yield satisfactory results due to the lack of domain-specific medical knowledge. In this paper, we propose BiSeg-SAM, a SAM-guided weakly supervised prompting and boundary refinement network for the segmentation of polyps and skin lesions. Specifically, we fine-tune SAM combined with a CNN module to learn local features. We introduce a WeakBox with two functions: automatically generating box prompts for the SAM model and using our proposed Multi-choice Mask-to-Box (MM2B) transformation for rough mask-to-box conversion, addressing the mismatch between coarse labels and precise predictions. Additionally, we apply scale consistency (SC) loss for prediction scale alignment. Our DetailRefine module enhances boundary precision and segmentation accuracy by refining coarse predictions using a limited amount of ground truth labels. This comprehensive approach enables BiSeg-SAM to achieve excellent multi-task segmentation performance. Our method demonstrates significant superiority over state-of-the-art (SOTA) methods when tested on five polyp datasets and one skin cancer dataset. 

**Abstract (ZH)**: 准确的结肠息肉和皮肤病变分割对于诊断结直肠癌和皮肤癌至关重要。尽管已经开发出了多种基于全监督深度学习技术的结肠息肉和皮肤病变分割方法，但医生对医学图像进行像素级标注既耗时又昂贵。基础视觉模型如Segment Anything Model (SAM) 展示了优越性能；然而，直接将 SAM 应用于医学分割可能无法获得令人满意的结果，因为缺乏特定领域的医学知识。本文提出了一种名为 BiSeg-SAM 的方法，这是一种由 SAM 引导的弱监督提示和边界精炼网络，用于结肠息肉和皮肤病变分割。具体而言，我们将 SAM 与 CNN 模块微调以学习局部特征。我们引入了 WeakBox，它有两种功能：自动为 SAM 模型生成框提示和使用我们提出的 Multi-choice Mask-to-Box (MM2B) 变换进行粗略的掩膜到框的转换，解决了粗略标签与精确预测之间的不匹配问题。此外，我们应用了尺度一致性 (SC) 损失以实现预测尺度对齐。我们的 DetailRefine 模块通过使用有限的真实标注来细化粗略预测以提高边界精度和分割准确性。该综合方法使 BiSeg-SAM 能够实现优异的多任务分割性能。在五个结肠息肉数据集和一个皮肤癌数据集上测试的结果表明，我们的方法明显优于现有最先进的 (SOTA) 方法。 

---
# MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation 

**Title (ZH)**: MuTri: 多视图三对齐 OCT到OCTA 3D 图像转换 

**Authors**: Zhuangzhuang Chen, Hualiang Wang, Chubin Ou, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.01428)  

**Abstract**: Optical coherence tomography angiography (OCTA) shows its great importance in imaging microvascular networks by providing accurate 3D imaging of blood vessels, but it relies upon specialized sensors and expensive devices. For this reason, previous works show the potential to translate the readily available 3D Optical Coherence Tomography (OCT) images into 3D OCTA images. However, existing OCTA translation methods directly learn the mapping from the OCT domain to the OCTA domain in continuous and infinite space with guidance from only a single view, i.e., the OCTA project map, resulting in suboptimal results. To this end, we propose the multi-view Tri-alignment framework for OCT to OCTA 3D image translation in discrete and finite space, named MuTri. In the first stage, we pre-train two vector-quantized variational auto-encoder (VQ- VAE) by reconstructing 3D OCT and 3D OCTA data, providing semantic prior for subsequent multi-view guidances. In the second stage, our multi-view tri-alignment facilitates another VQVAE model to learn the mapping from the OCT domain to the OCTA domain in discrete and finite space. Specifically, a contrastive-inspired semantic alignment is proposed to maximize the mutual information with the pre-trained models from OCT and OCTA views, to facilitate codebook learning. Meanwhile, a vessel structure alignment is proposed to minimize the structure discrepancy with the pre-trained models from the OCTA project map view, benefiting from learning the detailed vessel structure information. We also collect the first large-scale dataset, namely, OCTA2024, which contains a pair of OCT and OCTA volumes from 846 subjects. 

**Abstract (ZH)**: 多视角三对齐框架：从离散和有限空间实现OCT到OCTA的3D图像转换 

---
# TimeSearch: Hierarchical Video Search with Spotlight and Reflection for Human-like Long Video Understanding 

**Title (ZH)**: TimeSearch: 基于聚光灯和反射的多层次视频搜索以实现类人类长视频理解 

**Authors**: Junwen Pan, Rui Zhang, Xin Wan, Yuan Zhang, Ming Lu, Qi She  

**Link**: [PDF](https://arxiv.org/pdf/2504.01407)  

**Abstract**: Large video-language models (LVLMs) have shown remarkable performance across various video-language tasks. However, they encounter significant challenges when processing long videos because of the large number of video frames involved. Downsampling long videos in either space or time can lead to visual hallucinations, making it difficult to accurately interpret long videos. Motivated by human hierarchical temporal search strategies, we propose \textbf{TimeSearch}, a novel framework enabling LVLMs to understand long videos in a human-like manner. TimeSearch integrates two human-like primitives into a unified autoregressive LVLM: 1) \textbf{Spotlight} efficiently identifies relevant temporal events through a Temporal-Augmented Frame Representation (TAFR), explicitly binding visual features with timestamps; 2) \textbf{Reflection} evaluates the correctness of the identified events, leveraging the inherent temporal self-reflection capabilities of LVLMs. TimeSearch progressively explores key events and prioritizes temporal search based on reflection confidence. Extensive experiments on challenging long-video benchmarks confirm that TimeSearch substantially surpasses previous state-of-the-art, improving the accuracy from 41.8\% to 51.5\% on the LVBench. Additionally, experiments on temporal grounding demonstrate that appropriate TAFR is adequate to effectively stimulate the surprising temporal grounding ability of LVLMs in a simpler yet versatile manner, which improves mIoU on Charades-STA by 11.8\%. The code will be released. 

**Abstract (ZH)**: 大型视频语言模型（LVLMs）在各种视频语言任务中展现了卓越的性能。然而，在处理长视频时，由于涉及大量视频帧，它们面临着显著的挑战。通过时间和空间降采样来处理长视频可能导致视觉幻觉，使得准确地解释长视频变得困难。受人类层级时间搜索策略的启发，我们提出了TimeSearch，一种新颖的框架，使LVLMs能够以类似人类的方式理解长视频。TimeSearch将两种类人的机制集成到统一的自回归LVLM中：1）Spotlight通过时间增强帧表示（TAFR）高效地识别相关的时间事件，明确将视觉特征与时间戳绑定；2）Reflection利用LVLM固有的时间自我反思能力来评估识别的事件的正确性。TimeSearch逐步探索关键事件，并根据反思信心优先进行时间搜索。在具有挑战性的长视频基准上的广泛实验证实，TimeSearch显著超越了之前的最新方法，在LVBench上的准确率从41.8%提高到51.5%。此外，在时间对齐实验中，适当的TAFR足以以简单而通用的方式有效地激发LVLM惊人的时间对齐能力，从而在Charades-STA上的mIoU上提高了11.8%。代码将开源。 

---
# From Easy to Hard: Building a Shortcut for Differentially Private Image Synthesis 

**Title (ZH)**: 从简单到复杂：构建差分隐私图像合成的捷径 

**Authors**: Kecen Li, Chen Gong, Xiaochen Li, Yuzhong Zhao, Xinwen Hou, Tianhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01395)  

**Abstract**: Differentially private (DP) image synthesis aims to generate synthetic images from a sensitive dataset, alleviating the privacy leakage concerns of organizations sharing and utilizing synthetic images. Although previous methods have significantly progressed, especially in training diffusion models on sensitive images with DP Stochastic Gradient Descent (DP-SGD), they still suffer from unsatisfactory performance. In this work, inspired by curriculum learning, we propose a two-stage DP image synthesis framework, where diffusion models learn to generate DP synthetic images from easy to hard. Unlike existing methods that directly use DP-SGD to train diffusion models, we propose an easy stage in the beginning, where diffusion models learn simple features of the sensitive images. To facilitate this easy stage, we propose to use `central images', simply aggregations of random samples of the sensitive dataset. Intuitively, although those central images do not show details, they demonstrate useful characteristics of all images and only incur minimal privacy costs, thus helping early-phase model training. We conduct experiments to present that on the average of four investigated image datasets, the fidelity and utility metrics of our synthetic images are 33.1% and 2.1% better than the state-of-the-art method. 

**Abstract (ZH)**: 差分隐私（DP）图像合成旨在从敏感数据集生成合成图像，缓解组织在共享和使用合成图像时的隐私泄露顾虑。尽管先前的方法在训练敏感图像的扩散模型方面取得了显著进展，尤其是使用差分隐私随机梯度下降（DP-SGD），但它们仍然表现不佳。受curriculum learning启发，我们提出了一种两阶段的DP图像合成框架，其中扩散模型从易到难学习生成DP合成图像。与现有直接使用DP-SGD训练扩散模型的方法不同，我们在开始阶段提出了一种易于学习的阶段，扩散模型学习敏感图像的简单特征。为了促进这一易于学习的阶段，我们提出使用“中心图像”，即敏感数据集的随机样本的简单聚合。直观上，尽管这些中心图像不显示细节，但它们展示了所有图像的有用特征，并仅产生最小的隐私成本，从而帮助早期模型训练。我们在四个研究数据集上的实验表明，我们的合成图像的保真度和效用指标分别比当前最佳方法高33.1%和2.1%。 

---
# CFMD: Dynamic Cross-layer Feature Fusion for Salient Object Detection 

**Title (ZH)**: CFMD：跨层特征动态融合在显著目标检测中的应用 

**Authors**: Jin Lian, Zhongyu Wan, Ming Gao, JunFeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.01326)  

**Abstract**: Cross-layer feature pyramid networks (CFPNs) have achieved notable progress in multi-scale feature fusion and boundary detail preservation for salient object detection. However, traditional CFPNs still suffer from two core limitations: (1) a computational bottleneck caused by complex feature weighting operations, and (2) degraded boundary accuracy due to feature blurring in the upsampling process. To address these challenges, we propose CFMD, a novel cross-layer feature pyramid network that introduces two key innovations. First, we design a context-aware feature aggregation module (CFLMA), which incorporates the state-of-the-art Mamba architecture to construct a dynamic weight distribution mechanism. This module adaptively adjusts feature importance based on image context, significantly improving both representation efficiency and generalization. Second, we introduce an adaptive dynamic upsampling unit (CFLMD) that preserves spatial details during resolution recovery. By adjusting the upsampling range dynamically and initializing with a bilinear strategy, the module effectively reduces feature overlap and maintains fine-grained boundary structures. Extensive experiments on three standard benchmarks using three mainstream backbone networks demonstrate that CFMD achieves substantial improvements in pixel-level accuracy and boundary segmentation quality, especially in complex scenes. The results validate the effectiveness of CFMD in jointly enhancing computational efficiency and segmentation performance, highlighting its strong potential in salient object detection tasks. 

**Abstract (ZH)**: 跨层特征金字塔网络（CFMD）在多尺度特征融合和显著目标检测边界细节保留方面取得了显著进展。 

---
# TenAd: A Tensor-based Low-rank Black Box Adversarial Attack for Video Classification 

**Title (ZH)**: 基于张量低秩的黑盒 adversarial 攻击方法 TenAd 用于视频分类 

**Authors**: Kimia haghjooei, Mansoor Rezghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01228)  

**Abstract**: Deep learning models have achieved remarkable success in computer vision but remain vulnerable to adversarial attacks, particularly in black-box settings where model details are unknown. Existing adversarial attack methods(even those works with key frames) often treat video data as simple vectors, ignoring their inherent multi-dimensional structure, and require a large number of queries, making them inefficient and detectable. In this paper, we propose \textbf{TenAd}, a novel tensor-based low-rank adversarial attack that leverages the multi-dimensional properties of video data by representing videos as fourth-order tensors. By exploiting low-rank attack, our method significantly reduces the search space and the number of queries needed to generate adversarial examples in black-box settings. Experimental results on standard video classification datasets demonstrate that \textbf{TenAd} effectively generates imperceptible adversarial perturbations while achieving higher attack success rates and query efficiency compared to state-of-the-art methods. Our approach outperforms existing black-box adversarial attacks in terms of success rate, query efficiency, and perturbation imperceptibility, highlighting the potential of tensor-based methods for adversarial attacks on video models. 

**Abstract (ZH)**: 基于张量的低秩对抗攻击TenAd在黑盒设置下利用视频数据的多维性质显著减少了搜索空间和查询次数，从而生成不可感知的对抗扰动，同时实现了更高的攻击成功率和查询效率。 

---
# PolygoNet: Leveraging Simplified Polygonal Representation for Effective Image Classification 

**Title (ZH)**: PolygoNet: 利用简化多边形表示实现有效的图像分类 

**Authors**: Salim Khazem, Jeremy Fix, Cédric Pradalier  

**Link**: [PDF](https://arxiv.org/pdf/2504.01214)  

**Abstract**: Deep learning models have achieved significant success in various image related tasks. However, they often encounter challenges related to computational complexity and overfitting. In this paper, we propose an efficient approach that leverages polygonal representations of images using dominant points or contour coordinates. By transforming input images into these compact forms, our method significantly reduces computational requirements, accelerates training, and conserves resources making it suitable for real time and resource constrained applications. These representations inherently capture essential image features while filtering noise, providing a natural regularization effect that mitigates overfitting. The resulting lightweight models achieve performance comparable to state of the art methods using full resolution images while enabling deployment on edge devices. Extensive experiments on benchmark datasets validate the effectiveness of our approach in reducing complexity, improving generalization, and facilitating edge computing applications. This work demonstrates the potential of polygonal representations in advancing efficient and scalable deep learning solutions for real world scenarios. The code for the experiments of the paper is provided in this https URL. 

**Abstract (ZH)**: 深度学习模型在各类图像相关任务中取得了显著成功，但也常遇到计算复杂度高和过拟合的挑战。本文提出了一种有效的方法，利用图像的多边形表示，采用主导点或轮廓坐标。通过将输入图像转换为这些紧凑形式，我们的方法显著降低了计算需求，加速了训练，并节省了资源，使其适用于实时和资源受限的应用。这些表示自然捕捉了图像的基本特征，同时过滤了噪声，提供了一种自然的正则化效果，减轻了过拟合。生成的轻量级模型在使用全分辨率图像达到与最新方法相当的性能的同时，能够部署在边缘设备上。在基准数据集上的大量实验验证了该方法在降低复杂度、提高泛化能力和促进边缘计算应用方面的有效性。该工作展示了多边形表示在推进实际场景中高效可扩展的深度学习解决方案方面的潜力。论文实验代码可在以下链接获取：[此链接]。 

---
# Lightweight Deep Models for Dermatological Disease Detection: A Study on Instance Selection and Channel Optimization 

**Title (ZH)**: 轻量级深度模型在皮肤疾病检测中的研究：基于实例选择与通道优化 

**Authors**: Ian Mateos Gonzalez, Estefani Jaramilla Nava, Abraham Sánchez Morales, Jesús García-Ramírez, Ricardo Ramos-Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2504.01208)  

**Abstract**: The identification of dermatological disease is an important problem in Mexico according with different studies. Several works in literature use the datasets of different repositories without applying a study of the data behavior, especially in medical images domain. In this work, we propose a methodology to preprocess dermaMNIST dataset in order to improve its quality for the classification stage, where we use lightweight convolutional neural networks. In our results, we reduce the number of instances for the neural network training obtaining a similar performance of models as ResNet. 

**Abstract (ZH)**: 皮肤病识别是墨西哥不同研究中一个重要的问题。文献中使用不同仓库的数据集进行研究，但没有特别针对数据行为进行研究，尤其是在医学图像领域。在本文中，我们提出了一种预处理 dermaMNIST 数据集的方法，以提高其质量，从而在分类阶段获得与 ResNet 相似的性能，同时使用轻量级卷积神经网络减少神经网络训练的实例数量。 

---
# RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety 

**Title (ZH)**: RipVIS：海滩监测与安全的冲浪 Undertow 视频实例分割基准 

**Authors**: Andrei Dumitriu, Florin Tatui, Florin Miron, Aakash Ralhan, Radu Tudor Ionescu, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2504.01128)  

**Abstract**: Rip currents are strong, localized and narrow currents of water that flow outwards into the sea, causing numerous beach-related injuries and fatalities worldwide. Accurate identification of rip currents remains challenging due to their amorphous nature and the lack of annotated data, which often requires expert knowledge. To address these issues, we present RipVIS, a large-scale video instance segmentation benchmark explicitly designed for rip current segmentation. RipVIS is an order of magnitude larger than previous datasets, featuring $184$ videos ($212,328$ frames), of which $150$ videos ($163,528$ frames) are with rip currents, collected from various sources, including drones, mobile phones, and fixed beach cameras. Our dataset encompasses diverse visual contexts, such as wave-breaking patterns, sediment flows, and water color variations, across multiple global locations, including USA, Mexico, Costa Rica, Portugal, Italy, Greece, Romania, Sri Lanka, Australia and New Zealand. Most videos are annotated at $5$ FPS to ensure accuracy in dynamic scenarios, supplemented by an additional $34$ videos ($48,800$ frames) without rip currents. We conduct comprehensive experiments with Mask R-CNN, Cascade Mask R-CNN, SparseInst and YOLO11, fine-tuning these models for the task of rip current segmentation. Results are reported in terms of multiple metrics, with a particular focus on the $F_2$ score to prioritize recall and reduce false negatives. To enhance segmentation performance, we introduce a novel post-processing step based on Temporal Confidence Aggregation (TCA). RipVIS aims to set a new standard for rip current segmentation, contributing towards safer beach environments. We offer a benchmark website to share data, models, and results with the research community, encouraging ongoing collaboration and future contributions, at this https URL. 

**Abstract (ZH)**: rip currents:大规模视频实例分割基准，用于rip电流分割 

---
# Knowledge-Base based Semantic Image Transmission Using CLIP 

**Title (ZH)**: 基于知识库的语义图像传输——使用CLIP 

**Authors**: Chongyang Li, Yanmei He, Tianqian Zhang, Mingjian He, Shouyin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01053)  

**Abstract**: This paper proposes a novel knowledge-Base (KB) assisted semantic communication framework for image transmission. At the receiver, a Facebook AI Similarity Search (FAISS) based vector database is constructed by extracting semantic embeddings from images using the Contrastive Language-Image Pre-Training (CLIP) model. During transmission, the transmitter first extracts a 512-dimensional semantic feature using the CLIP model, then compresses it with a lightweight neural network for transmission. After receiving the signal, the receiver reconstructs the feature back to 512 dimensions and performs similarity matching from the KB to retrieve the most semantically similar image. Semantic transmission success is determined by category consistency between the transmitted and retrieved images, rather than traditional metrics like Peak Signal-to-Noise Ratio (PSNR). The proposed system prioritizes semantic accuracy, offering a new evaluation paradigm for semantic-aware communication systems. Experimental validation on CIFAR100 demonstrates the effectiveness of the framework in achieving semantic image transmission. 

**Abstract (ZH)**: 本文提出了一种基于知识库（KB）的语义通信框架，用于图像传输。在接收端，通过使用对比学习语言-图像预训练（CLIP）模型提取图像的语义嵌入，构建了一个基于Facebook AI相似搜索（FAISS）的向量数据库。在传输过程中，发送端首先使用CLIP模型提取一个512维的语义特征，然后通过一个轻量级神经网络对其进行压缩以便传输。接收端接收到信号后，将其重构回512维，并从知识库进行语义匹配以检索最具语义相似性的图像。语义传输的成功由传输图像与检索图像的类别一致性决定，而非传统的峰值信噪比（PSNR）等指标。该系统优先考虑语义准确性，为语义感知通信系统提供了新的评估范式。实验结果在CIFAR100数据集上的验证证明了该框架在实现语义图像传输方面的有效性。 

---
