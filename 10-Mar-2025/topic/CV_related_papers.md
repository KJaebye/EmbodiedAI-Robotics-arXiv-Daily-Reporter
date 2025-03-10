# Joint 3D Point Cloud Segmentation using Real-Sim Loop: From Panels to Trees and Branches 

**Title (ZH)**: 使用实-模环路的联合3D点云分割：从面板到树和枝条 

**Authors**: Tian Qiu, Ruiming Du, Nikolai Spine, Lailiang Cheng, Yu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05630)  

**Abstract**: Modern orchards are planted in structured rows with distinct panel divisions to improve management. Accurate and efficient joint segmentation of point cloud from Panel to Tree and Branch (P2TB) is essential for robotic operations. However, most current segmentation methods focus on single instance segmentation and depend on a sequence of deep networks to perform joint tasks. This strategy hinders the use of hierarchical information embedded in the data, leading to both error accumulation and increased costs for annotation and computation, which limits its scalability for real-world applications. In this study, we proposed a novel approach that incorporated a Real2Sim L-TreeGen for training data generation and a joint model (J-P2TB) designed for the P2TB task. The J-P2TB model, trained on the generated simulation dataset, was used for joint segmentation of real-world panel point clouds via zero-shot learning. Compared to representative methods, our model outperformed them in most segmentation metrics while using 40% fewer learnable parameters. This Sim2Real result highlighted the efficacy of L-TreeGen in model training and the performance of J-P2TB for joint segmentation, demonstrating its strong accuracy, efficiency, and generalizability for real-world applications. These improvements would not only greatly benefit the development of robots for automated orchard operations but also advance digital twin technology. 

**Abstract (ZH)**: 现代果园中基于结构化行植株分割的点云联合分割方法研究：从面板到树木和枝条（P2TB）的任务 

---
# LiDAR-enhanced 3D Gaussian Splatting Mapping 

**Title (ZH)**: LiDAR增强的3D高斯绘制映射 

**Authors**: Jian Shen, Huai Yu, Ji Wu, Wen Yang, Gui-Song Xia  

**Link**: [PDF](https://arxiv.org/pdf/2503.05425)  

**Abstract**: This paper introduces LiGSM, a novel LiDAR-enhanced 3D Gaussian Splatting (3DGS) mapping framework that improves the accuracy and robustness of 3D scene mapping by integrating LiDAR data. LiGSM constructs joint loss from images and LiDAR point clouds to estimate the poses and optimize their extrinsic parameters, enabling dynamic adaptation to variations in sensor alignment. Furthermore, it leverages LiDAR point clouds to initialize 3DGS, providing a denser and more reliable starting points compared to sparse SfM points. In scene rendering, the framework augments standard image-based supervision with depth maps generated from LiDAR projections, ensuring an accurate scene representation in both geometry and photometry. Experiments on public and self-collected datasets demonstrate that LiGSM outperforms comparative methods in pose tracking and scene rendering. 

**Abstract (ZH)**: LiGSM：一种基于LiDAR的新型3D高斯点云重建（3DGS）建图框架 

---
# Evidential Uncertainty Estimation for Multi-Modal Trajectory Prediction 

**Title (ZH)**: 多模态轨迹预测中的证据不确定性估计 

**Authors**: Sajad Marvi, Christoph Rist, Julian Schmidt, Julian Jordan, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2503.05274)  

**Abstract**: Accurate trajectory prediction is crucial for autonomous driving, yet uncertainty in agent behavior and perception noise makes it inherently challenging. While multi-modal trajectory prediction models generate multiple plausible future paths with associated probabilities, effectively quantifying uncertainty remains an open problem. In this work, we propose a novel multi-modal trajectory prediction approach based on evidential deep learning that estimates both positional and mode probability uncertainty in real time. Our approach leverages a Normal Inverse Gamma distribution for positional uncertainty and a Dirichlet distribution for mode uncertainty. Unlike sampling-based methods, it infers both types of uncertainty in a single forward pass, significantly improving efficiency. Additionally, we experimented with uncertainty-driven importance sampling to improve training efficiency by prioritizing underrepresented high-uncertainty samples over redundant ones. We perform extensive evaluations of our method on the Argoverse 1 and Argoverse 2 datasets, demonstrating that it provides reliable uncertainty estimates while maintaining high trajectory prediction accuracy. 

**Abstract (ZH)**: 基于证据深度学习的实时位置和模式概率不确定性估计的多模态轨迹预测方法 

---
# Discrete Contrastive Learning for Diffusion Policies in Autonomous Driving 

**Title (ZH)**: 离散对比学习在自主驾驶中的扩散策略 

**Authors**: Kalle Kujanpää, Daulet Baimukashev, Farzeen Munir, Shoaib Azam, Tomasz Piotr Kucner, Joni Pajarinen, Ville Kyrki  

**Link**: [PDF](https://arxiv.org/pdf/2503.05229)  

**Abstract**: Learning to perform accurate and rich simulations of human driving behaviors from data for autonomous vehicle testing remains challenging due to human driving styles' high diversity and variance. We address this challenge by proposing a novel approach that leverages contrastive learning to extract a dictionary of driving styles from pre-existing human driving data. We discretize these styles with quantization, and the styles are used to learn a conditional diffusion policy for simulating human drivers. Our empirical evaluation confirms that the behaviors generated by our approach are both safer and more human-like than those of the machine-learning-based baseline methods. We believe this has the potential to enable higher realism and more effective techniques for evaluating and improving the performance of autonomous vehicles. 

**Abstract (ZH)**: 基于对比学习从数据中提取驾驶风格字典以实现自动驾驶汽车测试中的准确丰富的人类驾驶行为模拟仍然具有挑战性，因为人类驾驶风格具有高度的多样性和变异性。我们通过提出一种新颖的方法来应对这一挑战，该方法利用对比学习从现有的人类驾驶数据中提取驾驶风格字典。我们通过量化对这些风格进行离散化，并使用这些风格来学习一个条件扩散策略以模拟人类驾驶员。我们的实证评估表明，通过我们的方法生成的行为不仅更安全，而且更接近人类。我们相信这有可能提高自动驾驶汽车评估和性能提升的真实性和有效性。 

---
# GSplatVNM: Point-of-View Synthesis for Visual Navigation Models Using Gaussian Splatting 

**Title (ZH)**: GSplatVNM: 基于高斯插值的视角合成用于视觉导航模型 

**Authors**: Kohei Honda, Takeshi Ishita, Yasuhiro Yoshimura, Ryo Yonitani  

**Link**: [PDF](https://arxiv.org/pdf/2503.05152)  

**Abstract**: This paper presents a novel approach to image-goal navigation by integrating 3D Gaussian Splatting (3DGS) with Visual Navigation Models (VNMs), a method we refer to as GSplatVNM. VNMs offer a promising paradigm for image-goal navigation by guiding a robot through a sequence of point-of-view images without requiring metrical localization or environment-specific training. However, constructing a dense and traversable sequence of target viewpoints from start to goal remains a central challenge, particularly when the available image database is sparse. To address these challenges, we propose a 3DGS-based viewpoint synthesis framework for VNMs that synthesizes intermediate viewpoints to seamlessly bridge gaps in sparse data while significantly reducing storage overhead. Experimental results in a photorealistic simulator demonstrate that our approach not only enhances navigation efficiency but also exhibits robustness under varying levels of image database sparsity. 

**Abstract (ZH)**: 基于3D Gaussian Splatting的视觉导航模型的新型图像目标导航方法：GSplatVNM 

---
# THE-SEAN: A Heart Rate Variation-Inspired Temporally High-Order Event-Based Visual Odometry with Self-Supervised Spiking Event Accumulation Networks 

**Title (ZH)**: THE-SEAN：一种受心率变化启发的时域高阶事件基于视觉里程计及其自监督尖峰事件累积网络 

**Authors**: Chaoran Xiong, Litao Wei, Kehui Ma, Zhen Sun, Yan Xiang, Zihan Nan, Trieu-Kien Truong, Ling Pei  

**Link**: [PDF](https://arxiv.org/pdf/2503.05112)  

**Abstract**: Event-based visual odometry has recently gained attention for its high accuracy and real-time performance in fast-motion systems. Unlike traditional synchronous estimators that rely on constant-frequency (zero-order) triggers, event-based visual odometry can actively accumulate information to generate temporally high-order estimation triggers. However, existing methods primarily focus on adaptive event representation after estimation triggers, neglecting the decision-making process for efficient temporal triggering itself. This oversight leads to the computational redundancy and noise accumulation. In this paper, we introduce a temporally high-order event-based visual odometry with spiking event accumulation networks (THE-SEAN). To the best of our knowledge, it is the first event-based visual odometry capable of dynamically adjusting its estimation trigger decision in response to motion and environmental changes. Inspired by biological systems that regulate hormone secretion to modulate heart rate, a self-supervised spiking neural network is designed to generate estimation triggers. This spiking network extracts temporal features to produce triggers, with rewards based on block matching points and Fisher information matrix (FIM) trace acquired from the estimator itself. Finally, THE-SEAN is evaluated across several open datasets, thereby demonstrating average improvements of 13\% in estimation accuracy, 9\% in smoothness, and 38\% in triggering efficiency compared to the state-of-the-art methods. 

**Abstract (ZH)**: 基于事件的视觉里程计：时空高阶事件累积网络（THE-SEAN）及其在动态调整估计触发决策方面的应用 

---
# Quantifying and Modeling Driving Styles in Trajectory Forecasting 

**Title (ZH)**: 量化与建模轨迹预测中的驾驶风格 

**Authors**: Laura Zheng, Hamidreza Yaghoubi Araghi, Tony Wu, Sandeep Thalapanane, Tianyi Zhou, Ming C. Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.04994)  

**Abstract**: Trajectory forecasting has become a popular deep learning task due to its relevance for scenario simulation for autonomous driving. Specifically, trajectory forecasting predicts the trajectory of a short-horizon future for specific human drivers in a particular traffic scenario. Robust and accurate future predictions can enable autonomous driving planners to optimize for low-risk and predictable outcomes for human drivers around them. Although some work has been done to model driving style in planning and personalized autonomous polices, a gap exists in explicitly modeling human driving styles for trajectory forecasting of human behavior. Human driving style is most certainly a correlating factor to decision making, especially in edge-case scenarios where risk is nontrivial, as justified by the large amount of traffic psychology literature on risky driving. So far, the current real-world datasets for trajectory forecasting lack insight on the variety of represented driving styles. While the datasets may represent real-world distributions of driving styles, we posit that fringe driving style types may also be correlated with edge-case safety scenarios. In this work, we conduct analyses on existing real-world trajectory datasets for driving and dissect these works from the lens of driving styles, which is often intangible and non-standardized. 

**Abstract (ZH)**: 轨迹预测已成为自动驾驶场景模拟相关的流行深度学习任务，尽管在规划和个性化自主政策建模中已经开展了一些工作，但在轨迹预测中显式建模人类驾驶风格仍存在差距。迄今为止，现有的真实世界轨迹预测数据集缺乏对驾驶风格多样性的见解。尽管数据集可能代表了真实世界的驾驶风格分布，我们认为边缘驾驶风格类型也可能与边缘情况下的安全场景相关。本文从驾驶风格的角度分析现有的真实世界轨迹数据集，并从常难以量化且非标准化的驾驶风格视角剖析这些工作。 

---
# Novel Object 6D Pose Estimation with a Single Reference View 

**Title (ZH)**: 基于单参考视角的新型对象6D姿态估计 

**Authors**: Jian Liu, Wei Sun, Kai Zeng, Jin Zheng, Hui Yang, Lin Wang, Hossein Rahmani, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2503.05578)  

**Abstract**: Existing novel object 6D pose estimation methods typically rely on CAD models or dense reference views, which are both difficult to acquire. Using only a single reference view is more scalable, but challenging due to large pose discrepancies and limited geometric and spatial information. To address these issues, we propose a Single-Reference-based novel object 6D (SinRef-6D) pose estimation method. Our key idea is to iteratively establish point-wise alignment in the camera coordinate system based on state space models (SSMs). Specifically, iterative camera-space point-wise alignment can effectively handle large pose discrepancies, while our proposed RGB and Points SSMs can capture long-range dependencies and spatial information from a single view, offering linear complexity and superior spatial modeling capability. Once pre-trained on synthetic data, SinRef-6D can estimate the 6D pose of a novel object using only a single reference view, without requiring retraining or a CAD model. Extensive experiments on six popular datasets and real-world robotic scenes demonstrate that we achieve on-par performance with CAD-based and dense reference view-based methods, despite operating in the more challenging single reference setting. Code will be released at this https URL. 

**Abstract (ZH)**: 基于单参考视图的新型对象6D姿态估计方法（SinRef-6D） 

---
# SplatPose: Geometry-Aware 6-DoF Pose Estimation from Single RGB Image via 3D Gaussian Splatting 

**Title (ZH)**: SplatPose：基于几何 aware 的单张 RGB 图像六自由度姿态估计方法通过 3D 高斯绘制 

**Authors**: Linqi Yang, Xiongwei Zhao, Qihao Sun, Ke Wang, Ao Chen, Peng Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05174)  

**Abstract**: 6-DoF pose estimation is a fundamental task in computer vision with wide-ranging applications in augmented reality and robotics. Existing single RGB-based methods often compromise accuracy due to their reliance on initial pose estimates and susceptibility to rotational ambiguity, while approaches requiring depth sensors or multi-view setups incur significant deployment costs. To address these limitations, we introduce SplatPose, a novel framework that synergizes 3D Gaussian Splatting (3DGS) with a dual-branch neural architecture to achieve high-precision pose estimation using only a single RGB image. Central to our approach is the Dual-Attention Ray Scoring Network (DARS-Net), which innovatively decouples positional and angular alignment through geometry-domain attention mechanisms, explicitly modeling directional dependencies to mitigate rotational ambiguity. Additionally, a coarse-to-fine optimization pipeline progressively refines pose estimates by aligning dense 2D features between query images and 3DGS-synthesized views, effectively correcting feature misalignment and depth errors from sparse ray sampling. Experiments on three benchmark datasets demonstrate that SplatPose achieves state-of-the-art 6-DoF pose estimation accuracy in single RGB settings, rivaling approaches that depend on depth or multi-view images. 

**Abstract (ZH)**: 6-DoF 姿态估计是计算机视觉中的一个基本任务，广泛应用于增强现实和机器人领域。现有的单RGB方法常常由于依赖初始姿态估计和易受旋转歧义性的影响而牺牲精度，而需要深度传感器或多视图设置的方法则会带来显著的部署成本。为了解决这些问题，我们引入了 SplatPose，这是一种新颖的框架，将3D 高斯点积（3DGS）与双分支神经架构相结合，仅使用单张RGB图像即可实现高精度的姿态估计。我们方法的核心是双注意射线评分网络（DARS-Net），这是一种创新的方法，通过几何域注意机制解耦位置和角度对齐，明确建模方向依赖性以减轻旋转歧义性。此外，从粗到细的优化pipeline逐步通过查询图像和3DGS合成视图之间的密集2D特征对齐，有效地纠正了稀疏射线采样引起的特征错位和深度误差。在三个基准数据集上的实验表明，SplatPose 在单RGB设置下的6-DoF姿态估计精度达到了最先进的水平，与依赖深度图像或多视图图像的方法不相上下。 

---
# INTENT: Trajectory Prediction Framework with Intention-Guided Contrastive Clustering 

**Title (ZH)**: 意图引导对比聚类的轨迹预测框架 

**Authors**: Yihong Tang, Wei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04952)  

**Abstract**: Accurate trajectory prediction of road agents (e.g., pedestrians, vehicles) is an essential prerequisite for various intelligent systems applications, such as autonomous driving and robotic navigation. Recent research highlights the importance of environmental contexts (e.g., maps) and the "multi-modality" of trajectories, leading to increasingly complex model structures. However, real-world deployments require lightweight models that can quickly migrate and adapt to new environments. Additionally, the core motivations of road agents, referred to as their intentions, deserves further exploration. In this study, we advocate that understanding and reasoning road agents' intention plays a key role in trajectory prediction tasks, and the main challenge is that the concept of intention is fuzzy and abstract. To this end, we present INTENT, an efficient intention-guided trajectory prediction model that relies solely on information contained in the road agent's trajectory. Our model distinguishes itself from existing models in several key aspects: (i) We explicitly model road agents' intentions through contrastive clustering, accommodating the fuzziness and abstraction of human intention in their trajectories. (ii) The proposed INTENT is based solely on multi-layer perceptrons (MLPs), resulting in reduced training and inference time, making it very efficient and more suitable for real-world deployment. (iii) By leveraging estimated intentions and an innovative algorithm for transforming trajectory observations, we obtain more robust trajectory representations that lead to superior prediction accuracy. Extensive experiments on real-world trajectory datasets for pedestrians and autonomous vehicles demonstrate the effectiveness and efficiency of INTENT. 

**Abstract (ZH)**: 基于路径意图引导的道路上交通参与者的准确轨迹预测 

---
# R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model 

**Title (ZH)**: R1-Zero在视觉推理中的“恍然大悟”时刻：一个2B非SFT模型的研究 

**Authors**: Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2503.05132)  

**Abstract**: Recently DeepSeek R1 demonstrated how reinforcement learning with simple rule-based incentives can enable autonomous development of complex reasoning in large language models, characterized by the "aha moment", in which the model manifest self-reflection and increased response length during training. However, attempts to extend this success to multimodal reasoning often failed to reproduce these key characteristics. In this report, we present the first successful replication of these emergent characteristics for multimodal reasoning on only a non-SFT 2B model. Starting with Qwen2-VL-2B and applying reinforcement learning directly on the SAT dataset, our model achieves 59.47% accuracy on CVBench, outperforming the base model by approximately ~30% and exceeding both SFT setting by ~2%. In addition, we share our failed attempts and insights in attempting to achieve R1-like reasoning using RL with instruct models. aiming to shed light on the challenges involved. Our key observations include: (1) applying RL on instruct model often results in trivial reasoning trajectories, and (2) naive length reward are ineffective in eliciting reasoning capabilities. The project code is available at this https URL 

**Abstract (ZH)**: 最近，DeepSeek R1证明了使用基于规则的激励与强化学习相结合可以促使大型语言模型在训练中自主发展出具有“啊哈时刻”的复杂推理能力。然而，将这一成功扩展到多模态推理时，往往无法再现这些关键特征。在本报告中，我们首次成功实现了仅在非SFT 2B模型上复制这些新兴特征的多模态推理。从Qwen2-VL-2B出发，直接在SAT数据集上应用强化学习，我们的模型在CVBench上的准确率达到59.47%，比基线模型高出约30%，同时超越SFT设置约2%。此外，我们分享了尝试使用指令模型通过RL实现类似R1的推理的失败尝试与见解，以揭示其中的挑战。我们的主要观察结果包括：(1) 在指令模型上应用RL通常会导致简单的推理轨迹，(2) 粗糙的长度奖励在激发推理能力方面无效。项目代码可在此处访问：这个 https URL。 

---
# VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control 

**Title (ZH)**: VideoPainter：基于即用型上下文控制的任意长度视频修复与编辑 

**Authors**: Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05639)  

**Abstract**: Video inpainting, which aims to restore corrupted video content, has experienced substantial progress. Despite these advances, existing methods, whether propagating unmasked region pixels through optical flow and receptive field priors, or extending image-inpainting models temporally, face challenges in generating fully masked objects or balancing the competing objectives of background context preservation and foreground generation in one model, respectively. To address these limitations, we propose a novel dual-stream paradigm VideoPainter that incorporates an efficient context encoder (comprising only 6% of the backbone parameters) to process masked videos and inject backbone-aware background contextual cues to any pre-trained video DiT, producing semantically consistent content in a plug-and-play manner. This architectural separation significantly reduces the model's learning complexity while enabling nuanced integration of crucial background context. We also introduce a novel target region ID resampling technique that enables any-length video inpainting, greatly enhancing our practical applicability. Additionally, we establish a scalable dataset pipeline leveraging current vision understanding models, contributing VPData and VPBench to facilitate segmentation-based inpainting training and assessment, the largest video inpainting dataset and benchmark to date with over 390K diverse clips. Using inpainting as a pipeline basis, we also explore downstream applications including video editing and video editing pair data generation, demonstrating competitive performance and significant practical potential. Extensive experiments demonstrate VideoPainter's superior performance in both any-length video inpainting and editing, across eight key metrics, including video quality, mask region preservation, and textual coherence. 

**Abstract (ZH)**: 视频修复：一种新颖的双流 paradigm VideoPainter 及其实用应用 

---
# TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models 

**Title (ZH)**: TrajectoryCrafter: 通过扩散模型重定向单目视频摄像机轨迹 

**Authors**: Mark YU, Wenbo Hu, Jinbo Xing, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05638)  

**Abstract**: We present TrajectoryCrafter, a novel approach to redirect camera trajectories for monocular videos. By disentangling deterministic view transformations from stochastic content generation, our method achieves precise control over user-specified camera trajectories. We propose a novel dual-stream conditional video diffusion model that concurrently integrates point cloud renders and source videos as conditions, ensuring accurate view transformations and coherent 4D content generation. Instead of leveraging scarce multi-view videos, we curate a hybrid training dataset combining web-scale monocular videos with static multi-view datasets, by our innovative double-reprojection strategy, significantly fostering robust generalization across diverse scenes. Extensive evaluations on multi-view and large-scale monocular videos demonstrate the superior performance of our method. 

**Abstract (ZH)**: 我们提出了TrajectoryCrafter，一种用于单目视频的新型相机轨迹重定向方法。通过解耦确定性的视角转换和随机的内容生成，我们的方法实现了对用户指定相机轨迹的精确控制。我们提出了一种新颖的双流条件视频扩散模型，该模型可以同时整合点云渲染和源视频作为条件，确保准确的视角转换和连贯的4D内容生成。我们不依赖稀少的多视角视频，而是通过我们创新的双重投影策略，将网络规模的单目视频与静态多视角数据集结合，构建混合训练数据集，显著增强了方法在多样场景下的泛化能力。在多视角和大规模单目视频上的广泛评估证明了该方法的优越性能。 

---
# CACTUS: An Open Dataset and Framework for Automated Cardiac Assessment and Classification of Ultrasound Images Using Deep Transfer Learning 

**Title (ZH)**: CACTUS：一种基于深度传输学习的心超图像自动心脏评估与分类的开放数据集及框架 

**Authors**: Hanae Elmekki, Ahmed Alagha, Hani Sami, Amanda Spilkin, Antonela Mariel Zanuttini, Ehsan Zakeri, Jamal Bentahar, Lyes Kadem, Wen-Fang Xie, Philippe Pibarot, Rabeb Mizouni, Hadi Otrok, Shakti Singh, Azzam Mourad  

**Link**: [PDF](https://arxiv.org/pdf/2503.05604)  

**Abstract**: Cardiac ultrasound (US) scanning is a commonly used techniques in cardiology to diagnose the health of the heart and its proper functioning. Therefore, it is necessary to consider ways to automate these tasks and assist medical professionals in classifying and assessing cardiac US images. Machine learning (ML) techniques are regarded as a prominent solution due to their success in numerous applications aimed at enhancing the medical field, including addressing the shortage of echography technicians. However, the limited availability of medical data presents a significant barrier to applying ML in cardiology, particularly regarding US images of the heart. This paper addresses this challenge by introducing the first open graded dataset for Cardiac Assessment and ClassificaTion of UltraSound (CACTUS), which is available online. This dataset contains images obtained from scanning a CAE Blue Phantom and representing various heart views and different quality levels, exceeding the conventional cardiac views typically found in the literature. Additionally, the paper introduces a Deep Learning (DL) framework consisting of two main components. The first component classifies cardiac US images based on the heart view using a Convolutional Neural Network (CNN). The second component uses Transfer Learning (TL) to fine-tune the knowledge from the first component and create a model for grading and assessing cardiac images. The framework demonstrates high performance in both classification and grading, achieving up to 99.43% accuracy and as low as 0.3067 error, respectively. To showcase its robustness, the framework is further fine-tuned using new images representing additional cardiac views and compared to several other state-of-the-art architectures. The framework's outcomes and performance in handling real-time scans were also assessed using a questionnaire answered by cardiac experts. 

**Abstract (ZH)**: 心脏超声（US）扫描是心脏病学中常用的技术，用于诊断心脏健康及其正常功能。因此，考虑自动化这些任务并协助医学专业人员分类和评估心脏US图像的方法是必要的。机器学习（ML）技术因其在增强医疗领域中的广泛应用而被视为一种突出的解决方案，包括解决超声技术人员短缺问题。然而，医疗数据的有限可用性是将ML应用于心脏病学，特别是心脏US图像的一个重要障碍。本文通过引入第一个开放分级数据集Cardiac Assessment and ClassificaTion of UltraSound (CACTUS)，解决了这一挑战，该数据集已在互联网上发布。此数据集包含从CAE Blue Phantom扫描获取的图像，并代表了各种心脏视图和不同的质量水平，超过文献中常规的心脏视图。此外，本文还介绍了一个深度学习（DL）框架，该框架由两个主要组件组成。第一个组件使用卷积神经网络（CNN）根据心脏视图对心脏US图像进行分类。第二个组件利用迁移学习（TL）进一步微调第一个组件的知识，以创建用于评分和评估心脏图像的模型。该框架在分类和评分方面均表现出高性能，分类准确率达到99.43%，评分误差低至0.3067。为展示其鲁棒性，该框架进一步利用额外心脏视图的图像进行微调，并与几种其他最先进的架构进行了比较。该框架在处理实时扫描方面的结果和性能也得到了心脏专家问卷调查的评估。 

---
# Impoola: The Power of Average Pooling for Image-Based Deep Reinforcement Learning 

**Title (ZH)**: Impoola: 平均池化在基于图像的深度强化学习中的作用 

**Authors**: Raphael Trumpp, Ansgar Schäfftlein, Mirco Theile, Marco Caccamo  

**Link**: [PDF](https://arxiv.org/pdf/2503.05546)  

**Abstract**: As image-based deep reinforcement learning tackles more challenging tasks, increasing model size has become an important factor in improving performance. Recent studies achieved this by focusing on the parameter efficiency of scaled networks, typically using Impala-CNN, a 15-layer ResNet-inspired network, as the image encoder. However, while Impala-CNN evidently outperforms older CNN architectures, potential advancements in network design for deep reinforcement learning-specific image encoders remain largely unexplored. We find that replacing the flattening of output feature maps in Impala-CNN with global average pooling leads to a notable performance improvement. This approach outperforms larger and more complex models in the Procgen Benchmark, particularly in terms of generalization. We call our proposed encoder model Impoola-CNN. A decrease in the network's translation sensitivity may be central to this improvement, as we observe the most significant gains in games without agent-centered observations. Our results demonstrate that network scaling is not just about increasing model size - efficient network design is also an essential factor. 

**Abstract (ZH)**: 基于图像的深度强化学习随着处理任务的挑战性增加，模型规模的增大已成为提升性能的重要因素。近期研究通过关注缩放网络的参数效率，通常使用Impala-CNN（一个借鉴ResNet设计的15层网络）作为图像编码器来实现这一点。然而，尽管Impala-CNN显然优于较早的CNN架构，针对深度强化学习的特定图像编码器的网络设计潜在改进仍 largely unexplored。我们发现，将Impala-CNN中的输出特征图展平操作替换为全局平均池化可以显著提高性能。这种方法在Procgen基准测试中优于更大、更复杂的模型，尤其是在泛化能力方面。我们将我们提出的设计命名为Impoola-CNN。网络的平移敏感性降低可能是这一改进的关键，因为我们观察到在没有代理中心观测的游戏场景中获得了最大的收益。我们的研究结果表明，网络规模的增加不仅仅是增加模型大小的问题，高效网络设计也是关键因素。 

---
# FastMap: Fast Queries Initialization Based Vectorized HD Map Reconstruction Framework 

**Title (ZH)**: FastMap: 基于向量化快速建图和查询初始化框架 

**Authors**: Haotian Hu, Jingwei Xu, Fanyi Wang, Toyota Li, Yaonong Wang, Laifeng Hu, Zhiwang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05492)  

**Abstract**: Reconstruction of high-definition maps is a crucial task in perceiving the autonomous driving environment, as its accuracy directly impacts the reliability of prediction and planning capabilities in downstream modules. Current vectorized map reconstruction methods based on the DETR framework encounter limitations due to the redundancy in the decoder structure, necessitating the stacking of six decoder layers to maintain performance, which significantly hampers computational efficiency. To tackle this issue, we introduce FastMap, an innovative framework designed to reduce decoder redundancy in existing approaches. FastMap optimizes the decoder architecture by employing a single-layer, two-stage transformer that achieves multilevel representation capabilities. Our framework eliminates the conventional practice of randomly initializing queries and instead incorporates a heatmap-guided query generation module during the decoding phase, which effectively maps image features into structured query vectors using learnable positional encoding. Additionally, we propose a geometry-constrained point-to-line loss mechanism for FastMap, which adeptly addresses the challenge of distinguishing highly homogeneous features that often arise in traditional point-to-point loss computations. Extensive experiments demonstrate that FastMap achieves state-of-the-art performance in both nuScenes and Argoverse2 datasets, with its decoder operating 3.2 faster than the baseline. Code and more demos are available at this https URL. 

**Abstract (ZH)**: 高-definition地图的重建是自主驾驶环境中环境感知的关键任务，其准确性直接影响下游模块预测和规划能力的可靠性。基于DETR框架的向量地图重建方法因解码器结构的冗余限制，需要堆叠六层解码层以维持性能，这显著降低了计算效率。为此，我们提出FastMap，一种旨在减少现有方法中解码器冗余的创新框架。FastMap通过采用单层两阶段变换器优化解码器架构，实现多层次表示能力。该框架摒弃了随机初始化查询的传统做法，在解码阶段引入了由热图引导的查询生成模块，利用可学习的位置编码将图像特征有效地映射到结构化的查询向量中。此外，我们还提出了适用于FastMap的几何约束点到线损失机制，有效地解决了传统点到点损失计算中经常出现的高同质特征区分难题。广泛实验表明，FastMap在nuScenes和Argoverse2数据集中均达到最新技术水平，其解码器比基线快3.2倍。代码和更多演示可在以下链接获得。 

---
# Attenuation artifact detection and severity classification in intracoronary OCT using mixed image representations 

**Title (ZH)**: 基于混合图像表示的冠状动脉OCT中衰减伪影检测及严重程度分类 

**Authors**: Pierandrea Cancian, Simone Saitta, Xiaojin Gu, Rudolf L.M. van Herten, Thijs J. Luttikholt, Jos Thannhauser, Rick H.J.A. Volleberg, Ruben G.A. van der Waerden, Joske L. van der Zande, Clarisa I. Sánchez, Bram van Ginneken, Niels van Royen, Ivana Išgum  

**Link**: [PDF](https://arxiv.org/pdf/2503.05322)  

**Abstract**: In intracoronary optical coherence tomography (OCT), blood residues and gas bubbles cause attenuation artifacts that can obscure critical vessel structures. The presence and severity of these artifacts may warrant re-acquisition, prolonging procedure time and increasing use of contrast agent. Accurate detection of these artifacts can guide targeted re-acquisition, reducing the amount of repeated scans needed to achieve diagnostically viable images. However, the highly heterogeneous appearance of these artifacts poses a challenge for the automated detection of the affected image regions. To enable automatic detection of the attenuation artifacts caused by blood residues and gas bubbles based on their severity, we propose a convolutional neural network that performs classification of the attenuation lines (A-lines) into three classes: no artifact, mild artifact and severe artifact. Our model extracts and merges features from OCT images in both Cartesian and polar coordinates, where each column of the image represents an A-line. Our method detects the presence of attenuation artifacts in OCT frames reaching F-scores of 0.77 and 0.94 for mild and severe artifacts, respectively. The inference time over a full OCT scan is approximately 6 seconds. Our experiments show that analysis of images represented in both Cartesian and polar coordinate systems outperforms the analysis in polar coordinates only, suggesting that these representations contain complementary features. This work lays the foundation for automated artifact assessment and image acquisition guidance in intracoronary OCT imaging. 

**Abstract (ZH)**: 基于衰减程度的血残留和气泡衰减伪影的光学相干断层扫描自动检测 

---
# Frequency Autoregressive Image Generation with Continuous Tokens 

**Title (ZH)**: 连续Token驱动的频率自回归图像生成 

**Authors**: Hu Yu, Hao Luo, Hangjie Yuan, Yu Rong, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05305)  

**Abstract**: Autoregressive (AR) models for image generation typically adopt a two-stage paradigm of vector quantization and raster-scan ``next-token prediction", inspired by its great success in language modeling. However, due to the huge modality gap, image autoregressive models may require a systematic reevaluation from two perspectives: tokenizer format and regression direction. In this paper, we introduce the frequency progressive autoregressive (\textbf{FAR}) paradigm and instantiate FAR with the continuous tokenizer. Specifically, we identify spectral dependency as the desirable regression direction for FAR, wherein higher-frequency components build upon the lower one to progressively construct a complete image. This design seamlessly fits the causality requirement for autoregressive models and preserves the unique spatial locality of image data. Besides, we delve into the integration of FAR and the continuous tokenizer, introducing a series of techniques to address optimization challenges and improve the efficiency of training and inference processes. We demonstrate the efficacy of FAR through comprehensive experiments on the ImageNet dataset and verify its potential on text-to-image generation. 

**Abstract (ZH)**: 基于频率渐进的自回归（FAR）模型 paradigm及其在图像生成中的应用 

---
# Development and Enhancement of Text-to-Image Diffusion Models 

**Title (ZH)**: 文本到图像扩散模型的开发与优化 

**Authors**: Rajdeep Roshan Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05149)  

**Abstract**: This research focuses on the development and enhancement of text-to-image denoising diffusion models, addressing key challenges such as limited sample diversity and training instability. By incorporating Classifier-Free Guidance (CFG) and Exponential Moving Average (EMA) techniques, this study significantly improves image quality, diversity, and stability. Utilizing Hugging Face's state-of-the-art text-to-image generation model, the proposed enhancements establish new benchmarks in generative AI. This work explores the underlying principles of diffusion models, implements advanced strategies to overcome existing limitations, and presents a comprehensive evaluation of the improvements achieved. Results demonstrate substantial progress in generating stable, diverse, and high-quality images from textual descriptions, advancing the field of generative artificial intelligence and providing new foundations for future applications.
Keywords: Text-to-image, Diffusion model, Classifier-free guidance, Exponential moving average, Image generation. 

**Abstract (ZH)**: 本研究集中于文本到图像去噪扩散模型的开发与增强，针对样本多样性有限和训练不稳定等关键挑战。通过结合Classifier-Free Guidance (CFG)和Exponential Moving Average (EMA)技术，本研究显著提高了图像的质量、多样性和稳定性。利用Hugging Face的前沿文本到图像生成模型，所提出的研究增强建立了生成AI的新基准。本工作探索了扩散模型的基本原理，实施了先进的策略以克服现有限制，并全面评估了所取得的改进。结果表明，在从文本描述生成稳定、多样和高质量图像方面取得了实质性进步，推动了生成人工智能领域的发展，并为未来应用提供了新的基础。关键词：文本到图像、扩散模型、无分类器引导、指数移动平均、图像生成。 

---
# HexPlane Representation for 3D Semantic Scene Understanding 

**Title (ZH)**: 适用于3D语义场景理解的HexPlane表示方法 

**Authors**: Zeren Chen, Yuenan Hou, Yulin Chen, Li Liu, Xiao Sun, Lu Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.05127)  

**Abstract**: In this paper, we introduce the HexPlane representation for 3D semantic scene understanding. Specifically, we first design the View Projection Module (VPM) to project the 3D point cloud into six planes to maximally retain the original spatial information. Features of six planes are extracted by the 2D encoder and sent to the HexPlane Association Module (HAM) to adaptively fuse the most informative information for each point. The fused point features are further fed to the task head to yield the ultimate predictions. Compared to the popular point and voxel representation, the HexPlane representation is efficient and can utilize highly optimized 2D operations to process sparse and unordered 3D point clouds. It can also leverage off-the-shelf 2D models, network weights, and training recipes to achieve accurate scene understanding in 3D space. On ScanNet and SemanticKITTI benchmarks, our algorithm, dubbed HexNet3D, achieves competitive performance with previous algorithms. In particular, on the ScanNet 3D segmentation task, our method obtains 77.0 mIoU on the validation set, surpassing Point Transformer V2 by 1.6 mIoU. We also observe encouraging results in indoor 3D detection tasks. Note that our method can be seamlessly integrated into existing voxel-based, point-based, and range-based approaches and brings considerable gains without bells and whistles. The codes will be available upon publication. 

**Abstract (ZH)**: 本文引入了HexPlane表示方法用于3D语义场景理解。具体而言，我们首先设计了视图投影模块（VPM）将3D点云投影到六个平面上，以最大程度保留原始的空间信息。六个平面上的特征由2D编码器提取，然后送入六边形平面关联模块（HAM）中，以自适应地融合每个点的最具信息量的信息。融合后的点特征进一步传递给任务头以产生最终预测。与流行的点和体素表示方法相比，HexPlane表示方法更高效，并能利用高度优化的2D操作来处理稀疏且无序的3D点云。此外，它还可以利用现成的2D模型、网络权重和训练方案在3D空间中实现准确的场景理解。在ScanNet和SemanticKITTI基准测试中，我们的算法HexNet3D在性能上与先前算法具有竞争力。特别是在ScanNet 3D分割任务中，我们的方法在验证集上获得了77.0的mIoU，超越了Point Transformer V2的1.6个mIoU。我们还在室内3D检测任务中观察到了令人鼓舞的结果。值得注意的是，我们的方法能无缝集成到现有的体素基、点基和距离基方法中，并且在不添加复杂功能的情况下带来显著改进。代码将在发表后公开。 

---
# Adapt3R: Adaptive 3D Scene Representation for Domain Transfer in Imitation Learning 

**Title (ZH)**: Adapt3R: 适应性三维场景表示在imitation learning中的领域迁移 

**Authors**: Albert Wilcox, Mohamed Ghanem, Masoud Moghani, Pierre Barroso, Benjamin Joffe, Animesh Garg  

**Link**: [PDF](https://arxiv.org/pdf/2503.04877)  

**Abstract**: Imitation Learning (IL) has been very effective in training robots to perform complex and diverse manipulation tasks. However, its performance declines precipitously when the observations are out of the training distribution. 3D scene representations that incorporate observations from calibrated RGBD cameras have been proposed as a way to improve generalizability of IL policies, but our evaluations in cross-embodiment and novel camera pose settings found that they show only modest improvement. To address those challenges, we propose Adaptive 3D Scene Representation (Adapt3R), a general-purpose 3D observation encoder which uses a novel architecture to synthesize data from one or more RGBD cameras into a single vector that can then be used as conditioning for arbitrary IL algorithms. The key idea is to use a pretrained 2D backbone to extract semantic information about the scene, using 3D only as a medium for localizing this semantic information with respect to the end-effector. We show that when trained end-to-end with several SOTA multi-task IL algorithms, Adapt3R maintains these algorithms' multi-task learning capacity while enabling zero-shot transfer to novel embodiments and camera poses. Furthermore, we provide a detailed suite of ablation and sensitivity experiments to elucidate the design space for point cloud observation encoders. 

**Abstract (ZH)**: 自适应三维场景表示（Adapt3R）：一种通用的三维观测编码器 

---
# Manboformer: Learning Gaussian Representations via Spatial-temporal Attention Mechanism 

**Title (ZH)**: Manboformer：通过空间-时间注意力机制学习高斯表示 

**Authors**: Ziyue Zhao, Qining Qi, Jianfa Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04863)  

**Abstract**: Compared with voxel-based grid prediction, in the field of 3D semantic occupation prediction for autonomous driving, GaussianFormer proposed using 3D Gaussian to describe scenes with sparse 3D semantic Gaussian based on objects is another scheme with lower memory requirements. Each 3D Gaussian function represents a flexible region of interest and its semantic features, which are iteratively refined by the attention mechanism. In the experiment, it is found that the Gaussian function required by this method is larger than the query resolution of the original dense grid network, resulting in impaired performance. Therefore, we consider optimizing GaussianFormer by using unused temporal information. We learn the Spatial-Temporal Self-attention Mechanism from the previous grid-given occupation network and improve it to GaussianFormer. The experiment was conducted with the NuScenes dataset, and the experiment is currently underway. 

**Abstract (ZH)**: 基于高斯函数的3D语义占用预测：GaussianFormer在自动驾驶领域的另一种低内存要求方案及其时空自注意力机制优化 

---
# ZAugNet for Z-Slice Augmentation in Bio-Imaging 

**Title (ZH)**: ZAugNet在生物成像中的Z切片增强 

**Authors**: Alessandro Pasqui, Sajjad Mahdavi, Benoit Vianay, Alexandra Colin, Alex McDougall, Rémi Dumollard, Yekaterina A. Miroshnikova, Elsa Labrune, Hervé Turlier  

**Link**: [PDF](https://arxiv.org/pdf/2503.04843)  

**Abstract**: Three-dimensional biological microscopy has significantly advanced our understanding of complex biological structures. However, limitations due to microscopy techniques, sample properties or phototoxicity often result in poor z-resolution, hindering accurate cellular measurements. Here, we introduce ZAugNet, a fast, accurate, and self-supervised deep learning method for enhancing z-resolution in biological images. By performing nonlinear interpolation between consecutive slices, ZAugNet effectively doubles resolution with each iteration. Compared on several microscopy modalities and biological objects, it outperforms competing methods on most metrics. Our method leverages a generative adversarial network (GAN) architecture combined with knowledge distillation to maximize prediction speed without compromising accuracy. We also developed ZAugNet+, an extended version enabling continuous interpolation at arbitrary distances, making it particularly useful for datasets with nonuniform slice spacing. Both ZAugNet and ZAugNet+ provide high-performance, scalable z-slice augmentation solutions for large-scale 3D imaging. They are available as open-source frameworks in PyTorch, with an intuitive Colab notebook interface for easy access by the scientific community. 

**Abstract (ZH)**: 三维生物显微镜显著提高了我们对复杂生物结构的理解。然而，由于显微镜技术的限制、样本特性或光毒性，往往会导致较差的z分辨率，阻碍了准确的细胞测量。在这里，我们介绍了ZAugNet，这是一种快速、准确且自我监督的深度学习方法，用于增强生物图像的z分辨率。通过在连续切片间进行非线性插值，ZAugNet在每次迭代中有效提高了分辨率。在多种显微镜模式和生物对象的对比中，其在大多数指标上优于竞争方法。我们的方法结合生成对抗网络（GAN）架构与知识蒸馏，最大化预测速度同时不牺牲准确性。我们还开发了ZAugNet+，一个增强版本，能够在任意距离实现连续插值，特别适用于非均匀切片间距的数据集。ZAugNet和ZAugNet+为大规模3D成像提供了高性能、可扩展的z切片增强解决方案。它们以PyTorch开源框架形式提供，并附有直观的Colab笔记本界面，便于科学界访问。 

---
# DA-STGCN: 4D Trajectory Prediction Based on Spatiotemporal Feature Extraction 

**Title (ZH)**: DA-STGCN：基于时空特征提取的4D轨迹预测 

**Authors**: Yuheng Kuang, Zhengning Wang, Jianping Zhang, Zhenyu Shi, Yuding Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04823)  

**Abstract**: The importance of four-dimensional (4D) trajectory prediction within air traffic management systems is on the rise. Key operations such as conflict detection and resolution, aircraft anomaly monitoring, and the management of congested flight paths are increasingly reliant on this foundational technology, underscoring the urgent demand for intelligent solutions. The dynamics in airport terminal zones and crowded airspaces are intricate and ever-changing; however, current methodologies do not sufficiently account for the interactions among aircraft. To tackle these challenges, we propose DA-STGCN, an innovative spatiotemporal graph convolutional network that integrates a dual attention mechanism. Our model reconstructs the adjacency matrix through a self-attention approach, enhancing the capture of node correlations, and employs graph attention to distill spatiotemporal characteristics, thereby generating a probabilistic distribution of predicted trajectories. This novel adjacency matrix, reconstructed with the self-attention mechanism, is dynamically optimized throughout the network's training process, offering a more nuanced reflection of the inter-node relationships compared to traditional algorithms. The performance of the model is validated on two ADS-B datasets, one near the airport terminal area and the other in dense airspace. Experimental results demonstrate a notable improvement over current 4D trajectory prediction methods, achieving a 20% and 30% reduction in the Average Displacement Error (ADE) and Final Displacement Error (FDE), respectively. The incorporation of a Dual-Attention module has been shown to significantly enhance the extraction of node correlations, as verified by ablation experiments. 

**Abstract (ZH)**: 四维（4D）飞行轨迹预测在空中交通管理系统中的重要性日益凸显：基于双注意机制的时空图卷积网络（DA-STGCN）及其应用 

---
# Normalization through Fine-tuning: Understanding Wav2vec 2.0 Embeddings for Phonetic Analysis 

**Title (ZH)**: 通过微调实现规范化：理解Wav2vec 2.0嵌入在音素分析中的作用 

**Authors**: Yiming Wang, Yi Yang, Jiahong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04814)  

**Abstract**: Phonetic normalization plays a crucial role in speech recognition and analysis, ensuring the comparability of features derived from raw audio data. However, in the current paradigm of fine-tuning pre-trained large transformer models, phonetic normalization is not deemed a necessary step; instead, it is implicitly executed within the models. This study investigates the normalization process within transformer models, especially wav2vec 2.0. Through a comprehensive analysis of embeddings from models fine-tuned for various tasks, our results demonstrate that fine-tuning wav2vec 2.0 effectively achieves phonetic normalization by selectively suppressing task-irrelevant information. We found that models fine-tuned for multiple tasks retain information for both tasks without compromising performance, and that suppressing task-irrelevant information is not necessary for effective classification. These findings provide new insights into how phonetic normalization can be flexibly achieved in speech models and how it is realized in human speech perception. 

**Abstract (ZH)**: 语音归一化在语音识别和分析中起着关键作用，确保从原始音频数据中提取的特征具有可比性。然而，在当前预训练大型变换器模型的微调 paradigm 中，语音归一化并不被视为必要步骤，而是隐式地在模型内部执行。本研究探讨了变换器模型中的归一化过程，特别是 wav2vec 2.0。通过对各种任务微调后的模型的嵌入进行全面分析，我们的结果表明，微调 wav2vec 2.0 通过选择性抑制与任务无关的信息有效地实现了语音归一化。我们发现，针对多个任务微调的模型保留了两个任务的信息而不牺牲性能，而且抑制与任务无关的信息对于有效的分类并不是必要的。这些发现为语音模型中语音归一化如何灵活实现以及其在人类语音感知中的实现方式提供了新的见解。 

---
