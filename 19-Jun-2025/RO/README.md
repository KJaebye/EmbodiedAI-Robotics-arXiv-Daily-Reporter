# Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos 

**Title (ZH)**: 基于粒子-网格神经动力学的可变形物体模型学习方法（从RGB-D视频中学习） 

**Authors**: Kaifeng Zhang, Baoyu Li, Kris Hauser, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.15680)  

**Abstract**: Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at this https URL . 

**Abstract (ZH)**: 基于粒子-网格的动态建模框架：从稀疏视图RGB-D记录中学习变形物体的动力学并生成3D条件动作视频 

---
# Vision in Action: Learning Active Perception from Human Demonstrations 

**Title (ZH)**: 行动中的视觉：从人类示范中学习主动感知 

**Authors**: Haoyu Xiong, Xiaomeng Xu, Jimmy Wu, Yifan Hou, Jeannette Bohg, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.15666)  

**Abstract**: We present Vision in Action (ViA), an active perception system for bimanual robot manipulation. ViA learns task-relevant active perceptual strategies (e.g., searching, tracking, and focusing) directly from human demonstrations. On the hardware side, ViA employs a simple yet effective 6-DoF robotic neck to enable flexible, human-like head movements. To capture human active perception strategies, we design a VR-based teleoperation interface that creates a shared observation space between the robot and the human operator. To mitigate VR motion sickness caused by latency in the robot's physical movements, the interface uses an intermediate 3D scene representation, enabling real-time view rendering on the operator side while asynchronously updating the scene with the robot's latest observations. Together, these design elements enable the learning of robust visuomotor policies for three complex, multi-stage bimanual manipulation tasks involving visual occlusions, significantly outperforming baseline systems. 

**Abstract (ZH)**: Vision在行动（ViA）：一种双臂机器人操作的主动感知系统 

---
# GRIM: Task-Oriented Grasping with Conditioning on Generative Examples 

**Title (ZH)**: GRIM: 以生成示例为条件的面向任务的抓取 

**Authors**: Shailesh, Alok Raj, Nayan Kumar, Priya Shukla, Andrew Melnik, Micheal Beetz, Gora Chand Nandi  

**Link**: [PDF](https://arxiv.org/pdf/2506.15607)  

**Abstract**: Task-Oriented Grasping (TOG) presents a significant challenge, requiring a nuanced understanding of task semantics, object affordances, and the functional constraints dictating how an object should be grasped for a specific task. To address these challenges, we introduce GRIM (Grasp Re-alignment via Iterative Matching), a novel training-free framework for task-oriented grasping. Initially, a coarse alignment strategy is developed using a combination of geometric cues and principal component analysis (PCA)-reduced DINO features for similarity scoring. Subsequently, the full grasp pose associated with the retrieved memory instance is transferred to the aligned scene object and further refined against a set of task-agnostic, geometrically stable grasps generated for the scene object, prioritizing task compatibility. In contrast to existing learning-based methods, GRIM demonstrates strong generalization capabilities, achieving robust performance with only a small number of conditioning examples. 

**Abstract (ZH)**: 面向任务的抓取（TOG） presents a significant challenge, requiring a nuanced understanding of task semantics, object affordances, and the functional constraints dictating how an object should be grasped for a specific task. To address these challenges, we introduce GRIM (Grasp Re-alignment via Iterative Matching), a novel training-free framework for task-oriented grasping. Initially, a coarse alignment strategy is developed using a combination of geometric cues and principal component analysis (PCA)-reduced DINO features for similarity scoring. Subsequently, the full grasp pose associated with the retrieved memory instance is transferred to the aligned scene object and further refined against a set of task-agnostic, geometrically stable grasps generated for the scene object, prioritizing task compatibility. In contrast to existing learning-based methods, GRIM demonstrates strong generalization capabilities, achieving robust performance with only a small number of conditioning examples. 

---
# Aerial Grasping via Maximizing Delta-Arm Workspace Utilization 

**Title (ZH)**: 基于最大化Delta臂工作空间利用率的空中抓取 

**Authors**: Haoran Chen, Weiliang Deng, Biyu Ye, Yifan Xiong, Ximin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15539)  

**Abstract**: The workspace limits the operational capabilities and range of motion for the systems with robotic arms. Maximizing workspace utilization has the potential to provide more optimal solutions for aerial manipulation tasks, increasing the system's flexibility and operational efficiency. In this paper, we introduce a novel planning framework for aerial grasping that maximizes workspace utilization. We formulate an optimization problem to optimize the aerial manipulator's trajectory, incorporating task constraints to achieve efficient manipulation. To address the challenge of incorporating the delta arm's non-convex workspace into optimization constraints, we leverage a Multilayer Perceptron (MLP) to map position points to feasibility this http URL, we employ Reversible Residual Networks (RevNet) to approximate the complex forward kinematics of the delta arm, utilizing efficient model gradients to eliminate workspace constraints. We validate our methods in simulations and real-world experiments to demonstrate their effectiveness. 

**Abstract (ZH)**: 基于机器臂的工作空间限制了系统操作能力和运动范围。最大化工作空间利用率有可能为航空抓取任务提供更优的解决方案，提高系统的灵活性和操作效率。本文介绍了一种新的规划框架，用于最大化航空抓取的工作空间利用率。我们提出了一个优化问题，以优化航空操作器的轨迹，并结合任务约束以实现高效的操纵。为了解决将delta臂的非凸工作空间纳入优化约束的挑战，我们利用多层感知机（MLP）将位置点映射到可行性区域。我们采用可逆残差网络（RevNet）来近似delta臂的复杂正向kinematics，并利用高效的模型梯度来消除工作空间约束。我们通过模拟和实际实验验证了方法的有效性。 

---
# Real-Time Initialization of Unknown Anchors for UWB-aided Navigation 

**Title (ZH)**: 基于UWB辅助导航的未知锚点的实时初始化 

**Authors**: Giulio Delama, Igor Borowski, Roland Jung, Stephan Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2506.15518)  

**Abstract**: This paper presents a framework for the real-time initialization of unknown Ultra-Wideband (UWB) anchors in UWB-aided navigation systems. The method is designed for localization solutions where UWB modules act as supplementary sensors. Our approach enables the automatic detection and calibration of previously unknown anchors during operation, removing the need for manual setup. By combining an online Positional Dilution of Precision (PDOP) estimation, a lightweight outlier detection method, and an adaptive robust kernel for non-linear optimization, our approach significantly improves robustness and suitability for real-world applications compared to state-of-the-art. In particular, we show that our metric which triggers an initialization decision is more conservative than current ones commonly based on initial linear or non-linear initialization guesses. This allows for better initialization geometry and subsequently lower initialization errors. We demonstrate the proposed approach on two different mobile robots: an autonomous forklift and a quadcopter equipped with a UWB-aided Visual-Inertial Odometry (VIO) framework. The results highlight the effectiveness of the proposed method with robust initialization and low positioning error. We open-source our code in a C++ library including a ROS wrapper. 

**Abstract (ZH)**: 本文提出了一种用于UWB辅助导航系统中实时初始化未知超宽带(UWB)锚点的框架。该方法适用于UWB模块作为辅助传感器的定位解决方案。我们的方法能够在运行时自动检测和校准未知的锚点，从而消除手动设置的需要。通过结合在线位置模糊因子(PDOP)估计、轻量级离群值检测方法以及非线性优化中的自适应稳健核，我们的方法在鲁棒性和适合实际应用方面显著优于现有技术。特别是，我们展示的初始化决策阈值指标比当前基于初始线性或非线性初始化猜测的标准更加保守，这能提供更好的初始化几何结构并降低初始化误差。我们在这两种不同的移动机器人上展示了所提出的方法：一台自主叉车和一架配备UWB辅助视觉惯性里程计(VIO)框架的四旋翼无人机。结果突显了所提出方法的有效性，具有稳健的初始化和较低的位置误差。我们以C++库的形式开源了我们的代码，其中包括一个ROS封装。 

---
# SurfAAV: Design and Implementation of a Novel Multimodal Surfing Aquatic-Aerial Vehicle 

**Title (ZH)**: SurfAAV：新型多模式水面-空中机动车辆的设计与实现 

**Authors**: Kun Liu, Junhao Xiao, Hao Lin, Yue Cao, Hui Peng, Kaihong Huang, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15450)  

**Abstract**: Despite significant advancements in the research of aquatic-aerial robots, existing configurations struggle to efficiently perform underwater, surface, and aerial movement simultaneously. In this paper, we propose a novel multimodal surfing aquatic-aerial vehicle, SurfAAV, which efficiently integrates underwater navigation, surface gliding, and aerial flying capabilities. Thanks to the design of the novel differential thrust vectoring hydrofoil, SurfAAV can achieve efficient surface gliding and underwater navigation without the need for a buoyancy adjustment system. This design provides flexible operational capabilities for both surface and underwater tasks, enabling the robot to quickly carry out underwater monitoring activities. Additionally, when it is necessary to reach another water body, SurfAAV can switch to aerial mode through a gliding takeoff, flying to the target water area to perform corresponding tasks. The main contribution of this letter lies in proposing a new solution for underwater, surface, and aerial movement, designing a novel hybrid prototype concept, developing the required control laws, and validating the robot's ability to successfully perform surface gliding and gliding takeoff. SurfAAV achieves a maximum surface gliding speed of 7.96 m/s and a maximum underwater speed of 3.1 m/s. The prototype's surface gliding maneuverability and underwater cruising maneuverability both exceed those of existing aquatic-aerial vehicles. 

**Abstract (ZH)**: 尽管在水空机器人研究方面取得了显著进展，现有的配置仍然难以高效地同时进行水面、水下和空中运动。本文提出了一种新型多模式冲浪水空机器人SurfAAV，有效地集成了水下导航、水面滑行和空中飞行能力。通过新型差动推力矢量水翼的设计，SurfAAV能够在无需调整浮力系统的情况下实现高效水面滑行和水下导航。该设计为水面和水下任务提供了灵活的操作能力，使机器人能够迅速执行水下监测活动。此外，当需要到达另一水体时，SurfAAV可以通过滑行起飞切换至空中模式，飞行至目标水域执行相应任务。本文的主要贡献在于提出了一种新的水下、水面和空中运动解决方案，设计了一种新型混合原型概念，开发了所需的控制律，并验证了机器人能够在水面滑行和滑行起飞方面成功执行的能力。SurfAAV的最大水面滑行速度为7.96 m/s，最大水下速度为3.1 m/s。该原型的水面滑行机动性和水下巡航机动性均优于现有的水空机器人。 

---
# MCOO-SLAM: A Multi-Camera Omnidirectional Object SLAM System 

**Title (ZH)**: 多目全景对象SLAM系统 

**Authors**: Miaoxin Pan, Jinnan Li, Yaowen Zhang, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2506.15402)  

**Abstract**: Object-level SLAM offers structured and semantically meaningful environment representations, making it more interpretable and suitable for high-level robotic tasks. However, most existing approaches rely on RGB-D sensors or monocular views, which suffer from narrow fields of view, occlusion sensitivity, and limited depth perception-especially in large-scale or outdoor environments. These limitations often restrict the system to observing only partial views of objects from limited perspectives, leading to inaccurate object modeling and unreliable data association. In this work, we propose MCOO-SLAM, a novel Multi-Camera Omnidirectional Object SLAM system that fully leverages surround-view camera configurations to achieve robust, consistent, and semantically enriched mapping in complex outdoor scenarios. Our approach integrates point features and object-level landmarks enhanced with open-vocabulary semantics. A semantic-geometric-temporal fusion strategy is introduced for robust object association across multiple views, leading to improved consistency and accurate object modeling, and an omnidirectional loop closure module is designed to enable viewpoint-invariant place recognition using scene-level descriptors. Furthermore, the constructed map is abstracted into a hierarchical 3D scene graph to support downstream reasoning tasks. Extensive experiments in real-world demonstrate that MCOO-SLAM achieves accurate localization and scalable object-level mapping with improved robustness to occlusion, pose variation, and environmental complexity. 

**Abstract (ZH)**: 多视角全景对象SLAM系统：一种在复杂户外场景中实现鲁棒、一致且语义丰富的映射的方法 

---
# Efficient Navigation Among Movable Obstacles using a Mobile Manipulator via Hierarchical Policy Learning 

**Title (ZH)**: 基于分层策略学习的移动 manipulator 在移动障碍物间高效导航 

**Authors**: Taegeun Yang, Jiwoo Hwang, Jeil Jeong, Minsung Yoon, Sung-Eui Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.15380)  

**Abstract**: We propose a hierarchical reinforcement learning (HRL) framework for efficient Navigation Among Movable Obstacles (NAMO) using a mobile manipulator. Our approach combines interaction-based obstacle property estimation with structured pushing strategies, facilitating the dynamic manipulation of unforeseen obstacles while adhering to a pre-planned global path. The high-level policy generates pushing commands that consider environmental constraints and path-tracking objectives, while the low-level policy precisely and stably executes these commands through coordinated whole-body movements. Comprehensive simulation-based experiments demonstrate improvements in performing NAMO tasks, including higher success rates, shortened traversed path length, and reduced goal-reaching times, compared to baselines. Additionally, ablation studies assess the efficacy of each component, while a qualitative analysis further validates the accuracy and reliability of the real-time obstacle property estimation. 

**Abstract (ZH)**: 基于层次强化学习的移动 manipulator 在移动障碍物环境下高效导航框架 

---
# Comparison of Innovative Strategies for the Coverage Problem: Path Planning, Search Optimization, and Applications in Underwater Robotics 

**Title (ZH)**: 创新策略比较研究：覆盖问题中的路径规划、搜索优化及其在水下机器人中的应用 

**Authors**: Ahmed Ibrahim, Francisco F. C. Rego, Éric Busvelle  

**Link**: [PDF](https://arxiv.org/pdf/2506.15376)  

**Abstract**: In many applications, including underwater robotics, the coverage problem requires an autonomous vehicle to systematically explore a defined area while minimizing redundancy and avoiding obstacles. This paper investigates coverage path planning strategies to enhance the efficiency of underwater gliders, particularly in maximizing the probability of detecting a radioactive source while ensuring safe navigation.
We evaluate three path-planning approaches: the Traveling Salesman Problem (TSP), Minimum Spanning Tree (MST), and Optimal Control Problem (OCP). Simulations were conducted in MATLAB, comparing processing time, uncovered areas, path length, and traversal time. Results indicate that OCP is preferable when traversal time is constrained, although it incurs significantly higher computational costs. Conversely, MST-based approaches provide faster but less optimal solutions. These findings offer insights into selecting appropriate algorithms based on mission priorities, balancing efficiency and computational feasibility. 

**Abstract (ZH)**: 水下机器人应用中，覆盖问题需要自主车辆系统地探索定义区域，同时减少冗余并避开障碍物。本文研究覆盖路径规划策略以提高水下滑翔器的效率，特别是在确保安全导航的前提下最大化检测放射性源的概率。 

---
# Offensive Robot Cybersecurity 

**Title (ZH)**: 进攻性机器人网络安全 

**Authors**: Víctor Mayoral-Vilches  

**Link**: [PDF](https://arxiv.org/pdf/2506.15343)  

**Abstract**: Offensive Robot Cybersecurity introduces a groundbreaking approach by advocating for offensive security methods empowered by means of automation. It emphasizes the necessity of understanding attackers' tactics and identifying vulnerabilities in advance to develop effective defenses, thereby improving robots' security posture. This thesis leverages a decade of robotics experience, employing Machine Learning and Game Theory to streamline the vulnerability identification and exploitation process. Intrinsically, the thesis uncovers a profound connection between robotic architecture and cybersecurity, highlighting that the design and creation aspect of robotics deeply intertwines with its protection against attacks. This duality -- whereby the architecture that shapes robot behavior and capabilities also necessitates a defense mechanism through offensive and defensive cybersecurity strategies -- creates a unique equilibrium. Approaching cybersecurity with a dual perspective of defense and attack, rooted in an understanding of systems architecture, has been pivotal. Through comprehensive analysis, including ethical considerations, the development of security tools, and executing cyber attacks on robot software, hardware, and industry deployments, this thesis proposes a novel architecture for cybersecurity cognitive engines. These engines, powered by advanced game theory and machine learning, pave the way for autonomous offensive cybersecurity strategies for robots, marking a significant shift towards self-defending robotic systems. This research not only underscores the importance of offensive measures in enhancing robot cybersecurity but also sets the stage for future advancements where robots are not just resilient to cyber threats but are equipped to autonomously safeguard themselves. 

**Abstract (ZH)**: 进攻性机器人网络安全 

---
# Context-Aware Deep Lagrangian Networks for Model Predictive Control 

**Title (ZH)**: 基于上下文的深拉格朗日网络模型预测控制 

**Authors**: Lucas Schulze, Jan Peters, Oleg Arenz  

**Link**: [PDF](https://arxiv.org/pdf/2506.15249)  

**Abstract**: Controlling a robot based on physics-informed dynamic models, such as deep Lagrangian networks (DeLaN), can improve the generalizability and interpretability of the resulting behavior. However, in complex environments, the number of objects to potentially interact with is vast, and their physical properties are often uncertain. This complexity makes it infeasible to employ a single global model. Therefore, we need to resort to online system identification of context-aware models that capture only the currently relevant aspects of the environment. While physical principles such as the conservation of energy may not hold across varying contexts, ensuring physical plausibility for any individual context-aware model can still be highly desirable, particularly when using it for receding horizon control methods such as Model Predictive Control (MPC). Hence, in this work, we extend DeLaN to make it context-aware, combine it with a recurrent network for online system identification, and integrate it with a MPC for adaptive, physics-informed control. We also combine DeLaN with a residual dynamics model to leverage the fact that a nominal model of the robot is typically available. We evaluate our method on a 7-DOF robot arm for trajectory tracking under varying loads. Our method reduces the end-effector tracking error by 39%, compared to a 21% improvement achieved by a baseline that uses an extended Kalman filter. 

**Abstract (ZH)**: 基于物理知情动力学模型的机器人控制可以提高结果行为的一般化能力和可解释性。然而，在复杂环境中，潜在可互动的对象众多，且其物理属性往往具有不确定性。这种复杂性使得使用单一全局模型变得不切实际。因此，我们需要采用在线系统识别方法，构建仅捕捉当前相关环境方面的情境感知模型。尽管在不同情境下可能会违反物理原理，如能量守恒定律，但对于用于预测控制方法如模型预测控制（MPC）的情境感知模型，确保其物理合理性仍然是非常重要的。因此，在本文中，我们扩展了DeLaN，使其具有情境感知能力，将其与循环网络结合用于在线系统识别，并与MPC集成，以实现自适应的物理知情控制。我们还将DeLaN与残差动力学模型结合，利用通常可用的机器人名义模型的优势。我们在不同负载条件下的7-DOF机器人臂轨迹跟踪中评估了该方法，该方法将末端执行器跟踪误差减少了39%，而基线方法（使用扩展卡尔曼滤波器）仅实现了21%的改进。 

---
# SHeRLoc: Synchronized Heterogeneous Radar Place Recognition for Cross-Modal Localization 

**Title (ZH)**: SHeRLoc: 同步异构雷达场所识别在跨模态定位中的应用 

**Authors**: Hanjun Kim, Minwoo Jung, Wooseong Yang, Ayoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.15175)  

**Abstract**: Despite the growing adoption of radar in robotics, the majority of research has been confined to homogeneous sensor types, overlooking the integration and cross-modality challenges inherent in heterogeneous radar technologies. This leads to significant difficulties in generalizing across diverse radar data types, with modality-aware approaches that could leverage the complementary strengths of heterogeneous radar remaining unexplored. To bridge these gaps, we propose SHeRLoc, the first deep network tailored for heterogeneous radar, which utilizes RCS polar matching to align multimodal radar data. Our hierarchical optimal transport-based feature aggregation method generates rotationally robust multi-scale descriptors. By employing FFT-similarity-based data mining and adaptive margin-based triplet loss, SHeRLoc enables FOV-aware metric learning. SHeRLoc achieves an order of magnitude improvement in heterogeneous radar place recognition, increasing recall@1 from below 0.1 to 0.9 on a public dataset and outperforming state of-the-art methods. Also applicable to LiDAR, SHeRLoc paves the way for cross-modal place recognition and heterogeneous sensor SLAM. The source code will be available upon acceptance. 

**Abstract (ZH)**: 尽管雷达在机器人领域的应用日益增长，大多数研究仍局限于同种传感器类型，忽视了异构雷达技术固有的跨模态整合挑战。这导致在处理多种多样的雷达数据时存在显著困难，利用异构雷达互补优势的方法尚未得到探索。为解决这些差距，我们提出SHeRLoc——首个针对异构雷达的深度网络，利用RCS极化匹配对多模态雷达数据进行对齐。我们的基于分层 optimal transport 的特征聚合方法生成旋转鲁棒的多尺度描述子。通过采用FFT相似性基于的数据挖掘和自适应边界三元组损失，SHeRLoc 实现了视场意识的度量学习。SHeRLoc 在异构雷达位置识别上取得了数量级的改进，在公共数据集上将 recall@1 从低于0.1 提升至0.9，并优于现有最佳方法。SHeRLoc 还适用于 LiDAR，并为跨模态位置识别和异构传感器 SLAM 开辟了道路。接受后将提供源代码。 

---
# Robust Instant Policy: Leveraging Student's t-Regression Model for Robust In-context Imitation Learning of Robot Manipulation 

**Title (ZH)**: 鲁棒即时策略：利用t-回归模型实现机器人 manipulation 的鲁棒上下文模仿学习 

**Authors**: Hanbit Oh, Andrea M. Salcedo-Vázquez, Ixchel G. Ramirez-Alpizar, Yukiyasu Domae  

**Link**: [PDF](https://arxiv.org/pdf/2506.15157)  

**Abstract**: Imitation learning (IL) aims to enable robots to perform tasks autonomously by observing a few human demonstrations. Recently, a variant of IL, called In-Context IL, utilized off-the-shelf large language models (LLMs) as instant policies that understand the context from a few given demonstrations to perform a new task, rather than explicitly updating network models with large-scale demonstrations. However, its reliability in the robotics domain is undermined by hallucination issues such as LLM-based instant policy, which occasionally generates poor trajectories that deviate from the given demonstrations. To alleviate this problem, we propose a new robust in-context imitation learning algorithm called the robust instant policy (RIP), which utilizes a Student's t-regression model to be robust against the hallucinated trajectories of instant policies to allow reliable trajectory generation. Specifically, RIP generates several candidate robot trajectories to complete a given task from an LLM and aggregates them using the Student's t-distribution, which is beneficial for ignoring outliers (i.e., hallucinations); thereby, a robust trajectory against hallucinations is generated. Our experiments, conducted in both simulated and real-world environments, show that RIP significantly outperforms state-of-the-art IL methods, with at least $26\%$ improvement in task success rates, particularly in low-data scenarios for everyday tasks. Video results available at this https URL. 

**Abstract (ZH)**: 基于上下文的鲁棒 imitation 学习算法：利用 t 回归模型对抗幻觉轨迹 

---
# Human Locomotion Implicit Modeling Based Real-Time Gait Phase Estimation 

**Title (ZH)**: 基于人体运动隐式建模的实时步态相位估计 

**Authors**: Yuanlong Ji, Xingbang Yang, Ruoqi Zhao, Qihan Ye, Quan Zheng, Yubo Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15150)  

**Abstract**: Gait phase estimation based on inertial measurement unit (IMU) signals facilitates precise adaptation of exoskeletons to individual gait variations. However, challenges remain in achieving high accuracy and robustness, particularly during periods of terrain changes. To address this, we develop a gait phase estimation neural network based on implicit modeling of human locomotion, which combines temporal convolution for feature extraction with transformer layers for multi-channel information fusion. A channel-wise masked reconstruction pre-training strategy is proposed, which first treats gait phase state vectors and IMU signals as joint observations of human locomotion, thus enhancing model generalization. Experimental results demonstrate that the proposed method outperforms existing baseline approaches, achieving a gait phase RMSE of $2.729 \pm 1.071%$ and phase rate MAE of $0.037 \pm 0.016%$ under stable terrain conditions with a look-back window of 2 seconds, and a phase RMSE of $3.215 \pm 1.303%$ and rate MAE of $0.050 \pm 0.023%$ under terrain transitions. Hardware validation on a hip exoskeleton further confirms that the algorithm can reliably identify gait cycles and key events, adapting to various continuous motion scenarios. This research paves the way for more intelligent and adaptive exoskeleton systems, enabling safer and more efficient human-robot interaction across diverse real-world environments. 

**Abstract (ZH)**: 基于惯性测量单元信号的人体步行相位估计神经网络能够促进外骨骼对个体步行变化的精确适应。然而，在地形变化期间实现高精度和鲁棒性仍然面临挑战。为了解决这个问题，我们开发了一种基于人类运动隐式建模的步行相位估计神经网络，该网络结合了时间卷积用于特征提取，并使用 Transformer 层进行多通道信息融合。提出了一种通道级遮罩重建预训练策略，首先将步行相位状态向量和惯性测量单元信号视为人类运动的联合观测，从而增强模型的泛化能力。实验结果表明，在稳定的地形条件下，使用2秒的视窗，所提出的方法在步行相位RMSE方面优于现有基线方法，达到$2.729 \pm 1.071\%$，相位率MAE为$0.037 \pm 0.016\%$；在地形过渡期间，相位RMSE达到$3.215 \pm 1.303\%$，相位率MAE为$0.050 \pm 0.023\%$。基于髋关节外骨骼的硬件验证进一步证实了该算法可以可靠地识别步行周期和关键事件，适应各种连续运动场景。本研究为进一步智能化和适应性强的外骨骼系统铺平了道路，使人类与机器人在多种实际环境中实现更安全、更高效的交互成为可能。 

---
# TACT: Humanoid Whole-body Contact Manipulation through Deep Imitation Learning with Tactile Modality 

**Title (ZH)**: TACT：通过触觉模态深度模仿学习实现的人形全身接触操作 

**Authors**: Masaki Murooka, Takahiro Hoshi, Kensuke Fukumitsu, Shimpei Masuda, Marwan Hamze, Tomoya Sasaki, Mitsuharu Morisawa, Eiichi Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2506.15146)  

**Abstract**: Manipulation with whole-body contact by humanoid robots offers distinct advantages, including enhanced stability and reduced load. On the other hand, we need to address challenges such as the increased computational cost of motion generation and the difficulty of measuring broad-area contact. We therefore have developed a humanoid control system that allows a humanoid robot equipped with tactile sensors on its upper body to learn a policy for whole-body manipulation through imitation learning based on human teleoperation data. This policy, named tactile-modality extended ACT (TACT), has a feature to take multiple sensor modalities as input, including joint position, vision, and tactile measurements. Furthermore, by integrating this policy with retargeting and locomotion control based on a biped model, we demonstrate that the life-size humanoid robot RHP7 Kaleido is capable of achieving whole-body contact manipulation while maintaining balance and walking. Through detailed experimental verification, we show that inputting both vision and tactile modalities into the policy contributes to improving the robustness of manipulation involving broad and delicate contact. 

**Abstract (ZH)**: 人形机器人通过全身接触进行操作具有独特的优势，包括增强的稳定性及减少的负载。然而，我们也面临着运动生成计算成本增加以及难以测量大面积接触的挑战。因此，我们开发了一种人形机器人控制系统，该系统允许装备了上身触觉传感器的人形机器人通过模仿基于人类远程操作数据的学习来学习全身操作策略。该策略名为触觉模态扩展的ACT（TACT），能够接受关节位置、视觉和触觉等多种传感模态的输入。此外，通过将该策略与基于双足模型的重定位和运动控制相结合，我们展示出全尺寸人形机器人RHP7 Kaleido能够在保持平衡和行走的同时实现全身接触操作。通过详细的实验验证，我们证明将视觉和触觉模态同时输入策略能够提高涉及广泛和精细接触的操作鲁棒性。 

---
# Booster Gym: An End-to-End Reinforcement Learning Framework for Humanoid Robot Locomotion 

**Title (ZH)**: Booster Gym: 人体形机器人运动控制的端到端强化学习框架 

**Authors**: Yushi Wang, Penghui Chen, Xinyu Han, Feng Wu, Mingguo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15132)  

**Abstract**: Recent advancements in reinforcement learning (RL) have led to significant progress in humanoid robot locomotion, simplifying the design and training of motion policies in simulation. However, the numerous implementation details make transferring these policies to real-world robots a challenging task. To address this, we have developed a comprehensive code framework that covers the entire process from training to deployment, incorporating common RL training methods, domain randomization, reward function design, and solutions for handling parallel structures. This library is made available as a community resource, with detailed descriptions of its design and experimental results. We validate the framework on the Booster T1 robot, demonstrating that the trained policies seamlessly transfer to the physical platform, enabling capabilities such as omnidirectional walking, disturbance resistance, and terrain adaptability. We hope this work provides a convenient tool for the robotics community, accelerating the development of humanoid robots. The code can be found in this https URL. 

**Abstract (ZH)**: Recent Advancements in Reinforcement Learning for Humanoid Robot Locomotion: A Comprehensive Code Framework for Deployment 

---
# VIMS: A Visual-Inertial-Magnetic-Sonar SLAM System in Underwater Environments 

**Title (ZH)**: VIMS： underwater环境下的一种视觉-惯性-磁敏-声呐 SLAM 系统 

**Authors**: Bingbing Zhang, Huan Yin, Shuo Liu, Fumin Zhang, Wen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15126)  

**Abstract**: In this study, we present a novel simultaneous localization and mapping (SLAM) system, VIMS, designed for underwater navigation. Conventional visual-inertial state estimators encounter significant practical challenges in perceptually degraded underwater environments, particularly in scale estimation and loop closing. To address these issues, we first propose leveraging a low-cost single-beam sonar to improve scale estimation. Then, VIMS integrates a high-sampling-rate magnetometer for place recognition by utilizing magnetic signatures generated by an economical magnetic field coil. Building on this, a hierarchical scheme is developed for visual-magnetic place recognition, enabling robust loop closure. Furthermore, VIMS achieves a balance between local feature tracking and descriptor-based loop closing, avoiding additional computational burden on the front end. Experimental results highlight the efficacy of the proposed VIMS, demonstrating significant improvements in both the robustness and accuracy of state estimation within underwater environments. 

**Abstract (ZH)**: 海底导航用的新型同时定位与建图系统VIMS 

---
# I Know You're Listening: Adaptive Voice for HRI 

**Title (ZH)**: 我知道你在听：适应性语音在人机交互中的应用 

**Authors**: Paige Tuttösí  

**Link**: [PDF](https://arxiv.org/pdf/2506.15107)  

**Abstract**: While the use of social robots for language teaching has been explored, there remains limited work on a task-specific synthesized voices for language teaching robots. Given that language is a verbal task, this gap may have severe consequences for the effectiveness of robots for language teaching tasks. We address this lack of L2 teaching robot voices through three contributions: 1. We address the need for a lightweight and expressive robot voice. Using a fine-tuned version of Matcha-TTS, we use emoji prompting to create an expressive voice that shows a range of expressivity over time. The voice can run in real time with limited compute resources. Through case studies, we found this voice more expressive, socially appropriate, and suitable for long periods of expressive speech, such as storytelling. 2. We explore how to adapt a robot's voice to physical and social ambient environments to deploy our voices in various locations. We found that increasing pitch and pitch rate in noisy and high-energy environments makes the robot's voice appear more appropriate and makes it seem more aware of its current environment. 3. We create an English TTS system with improved clarity for L2 listeners using known linguistic properties of vowels that are difficult for these listeners. We used a data-driven, perception-based approach to understand how L2 speakers use duration cues to interpret challenging words with minimal tense (long) and lax (short) vowels in English. We found that the duration of vowels strongly influences the perception for L2 listeners and created an "L2 clarity mode" for Matcha-TTS that applies a lengthening to tense vowels while leaving lax vowels unchanged. Our clarity mode was found to be more respectful, intelligible, and encouraging than base Matcha-TTS while reducing transcription errors in these challenging tense/lax minimal pairs. 

**Abstract (ZH)**: 社会机器人用于语言教学的特定任务合成语音研究 

---
# DyNaVLM: Zero-Shot Vision-Language Navigation System with Dynamic Viewpoints and Self-Refining Graph Memory 

**Title (ZH)**: DyNaVLM：具有动态视角和自我优化图记忆的零样本视觉-语言导航系统 

**Authors**: Zihe Ji, Huangxuan Lin, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15096)  

**Abstract**: We present DyNaVLM, an end-to-end vision-language navigation framework using Vision-Language Models (VLM). In contrast to prior methods constrained by fixed angular or distance intervals, our system empowers agents to freely select navigation targets via visual-language reasoning. At its core lies a self-refining graph memory that 1) stores object locations as executable topological relations, 2) enables cross-robot memory sharing through distributed graph updates, and 3) enhances VLM's decision-making via retrieval augmentation. Operating without task-specific training or fine-tuning, DyNaVLM demonstrates high performance on GOAT and ObjectNav benchmarks. Real-world tests further validate its robustness and generalization. The system's three innovations: dynamic action space formulation, collaborative graph memory, and training-free deployment, establish a new paradigm for scalable embodied robot, bridging the gap between discrete VLN tasks and continuous real-world navigation. 

**Abstract (ZH)**: DyNaVLM：一种基于视觉语言模型的端到端视觉语言导航框架 

---
# 3D Vision-tactile Reconstruction from Infrared and Visible Images for Robotic Fine-grained Tactile Perception 

**Title (ZH)**: 基于红外和可见光图像的3D视觉-触觉重建及其在机器人精细触觉感知中的应用 

**Authors**: Yuankai Lin, Xiaofan Lu, Jiahui Chen, Hua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15087)  

**Abstract**: To achieve human-like haptic perception in anthropomorphic grippers, the compliant sensing surfaces of vision tactile sensor (VTS) must evolve from conventional planar configurations to biomimetically curved topographies with continuous surface gradients. However, planar VTSs have challenges when extended to curved surfaces, including insufficient lighting of surfaces, blurring in reconstruction, and complex spatial boundary conditions for surface structures. With an end goal of constructing a human-like fingertip, our research (i) develops GelSplitter3D by expanding imaging channels with a prism and a near-infrared (NIR) camera, (ii) proposes a photometric stereo neural network with a CAD-based normal ground truth generation method to calibrate tactile geometry, and (iii) devises a normal integration method with boundary constraints of depth prior information to correcting the cumulative error of surface integrals. We demonstrate better tactile sensing performance, a 40$\%$ improvement in normal estimation accuracy, and the benefits of sensor shapes in grasping and manipulation tasks. 

**Abstract (ZH)**: 实现类人触觉感知的仿生触觉传感器柔软感知表面从传统平面配置进化到生物学启发的曲面拓扑结构并具有连续表面梯度是关键。然而，平面触觉传感器扩展到曲面时面临的挑战包括表面照明不足、重构模糊以及复杂的空间边界条件。为构建类人指尖，我们的研究开发了GelSplitter3D，通过棱镜和近红外相机扩展成像通道，提出了一种基于CAD的法线地面真实值生成方法的光度立体神经网络来校准触觉几何，并设计了一种结合深度先验信息的空间边界约束法线积分修正方法，以提高表面积分累计误差的准确性。 we demonstrate更好的触觉感知性能，法线估算精度提高40%，并展示了传感器形状在抓取和操作任务中的优势。 

---
# EmojiVoice: Towards long-term controllable expressivity in robot speech 

**Title (ZH)**: EmojiVoice：朝向长期可控表达性的机器人语音研究 

**Authors**: Paige Tuttösí, Shivam Mehta, Zachary Syvenky, Bermet Burkanova, Gustav Eje Henter, Angelica Lim  

**Link**: [PDF](https://arxiv.org/pdf/2506.15085)  

**Abstract**: Humans vary their expressivity when speaking for extended periods to maintain engagement with their listener. Although social robots tend to be deployed with ``expressive'' joyful voices, they lack this long-term variation found in human speech. Foundation model text-to-speech systems are beginning to mimic the expressivity in human speech, but they are difficult to deploy offline on robots. We present EmojiVoice, a free, customizable text-to-speech (TTS) toolkit that allows social roboticists to build temporally variable, expressive speech on social robots. We introduce emoji-prompting to allow fine-grained control of expressivity on a phase level and use the lightweight Matcha-TTS backbone to generate speech in real-time. We explore three case studies: (1) a scripted conversation with a robot assistant, (2) a storytelling robot, and (3) an autonomous speech-to-speech interactive agent. We found that using varied emoji prompting improved the perception and expressivity of speech over a long period in a storytelling task, but expressive voice was not preferred in the assistant use case. 

**Abstract (ZH)**: 人类在长时间交谈中会通过变化表达性来维持与听众的互动。尽管社会机器人通常配备“表达性”的喜悦声音，但缺乏人类语言中长期存在的表达性变化。基础模型文本到语音系统开始模仿人类语言的表达性，但在机器人上离线部署起来较为困难。我们提出了一种名为EmojiVoice的免费可定制文本到语音(TTS)工具包，使社会机器人专家能够在社会机器人上构建时间变化的、富有表现力的语音。我们介绍了emoji提示，以实现语音表达性在相位层面的精细控制，并使用轻量级的Matcha-TTS骨干网络实现实时语音生成。我们探讨了三个案例研究：(1) 与机器人助手进行剧本对话、(2) 故事讲述机器人，以及(3) 自主的语音到语音交互代理。我们发现，在故事讲述任务中使用多样的emoji提示可以提高长时间内语音的感知和表现性，但在助手使用场景中，富有表现力的声音并不受欢迎。 

---
# Assigning Multi-Robot Tasks to Multitasking Robots 

**Title (ZH)**: 多任务机器人分配多机器人任务 

**Authors**: Winston Smith, Andrew Boateng, Taha Shaheen, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15032)  

**Abstract**: One simplifying assumption in existing and well-performing task allocation methods is that the robots are single-tasking: each robot operates on a single task at any given time. While this assumption is harmless to make in some situations, it can be inefficient or even infeasible in others. In this paper, we consider assigning multi-robot tasks to multitasking robots. The key contribution is a novel task allocation framework that incorporates the consideration of physical constraints introduced by multitasking. This is in contrast to the existing work where such constraints are largely ignored. After formulating the problem, we propose a compilation to weighted MAX-SAT, which allows us to leverage existing solvers for a solution. A more efficient greedy heuristic is then introduced. For evaluation, we first compare our methods with a modern baseline that is efficient for single-tasking robots to validate the benefits of multitasking in synthetic domains. Then, using a site-clearing scenario in simulation, we further illustrate the complex task interaction considered by the multitasking robots in our approach to demonstrate its performance. Finally, we demonstrate a physical experiment to show how multitasking enabled by our approach can benefit task efficiency in a realistic setting. 

**Abstract (ZH)**: 多任务机器人下的任务分配：考虑物理约束的新框架及其应用 

---
# Context Matters: Learning Generalizable Rewards via Calibrated Features 

**Title (ZH)**: 背景重要：通过校准特征学习可泛化的奖励 

**Authors**: Alexandra Forsey-Smerek, Julie Shah, Andreea Bobu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15012)  

**Abstract**: A key challenge in reward learning from human input is that desired agent behavior often changes based on context. Traditional methods typically treat each new context as a separate task with its own reward function. For example, if a previously ignored stove becomes too hot to be around, the robot must learn a new reward from scratch, even though the underlying preference for prioritizing safety over efficiency remains unchanged. We observe that context influences not the underlying preference itself, but rather the $\textit{saliency}$--or importance--of reward features. For instance, stove heat affects the importance of the robot's proximity, yet the human's safety preference stays the same. Existing multi-task and meta IRL methods learn context-dependent representations $\textit{implicitly}$--without distinguishing between preferences and feature importance--resulting in substantial data requirements. Instead, we propose $\textit{explicitly}$ modeling context-invariant preferences separately from context-dependent feature saliency, creating modular reward representations that adapt to new contexts. To achieve this, we introduce $\textit{calibrated features}$--representations that capture contextual effects on feature saliency--and present specialized paired comparison queries that isolate saliency from preference for efficient learning. Experiments with simulated users show our method significantly improves sample efficiency, requiring 10x fewer preference queries than baselines to achieve equivalent reward accuracy, with up to 15% better performance in low-data regimes (5-10 queries). An in-person user study (N=12) demonstrates that participants can effectively teach their unique personal contextual preferences using our method, enabling more adaptable and personalized reward learning. 

**Abstract (ZH)**: 从人类输入中学习奖励的关键挑战在于期望的代理行为往往基于上下文而变化。传统方法通常将每个新的上下文视为具有自己奖励函数的独立任务。例如，如果之前未被注意的烤炉变得过热而无法靠近时，机器人必须从头开始学习新的奖励，尽管其优先安全性而非效率的底层偏好并未改变。我们观察到，上下文影响的不是底层偏好本身，而是奖励特征的$\textit{显著性}$——即重要性。例如，烤炉的温度影响机器人靠近的显著性，但人类的安全偏好保持不变。现有的多任务和元IRL方法在不了解偏好的同时学习上下文依赖的表征，导致需要大量数据。相反，我们提出分别明确建模上下文不变的偏好和上下文依赖的特征显著性，创建模块化的奖励表示以适应新上下文。为此，我们引入了$\textit{校准特征}$——能够捕捉特征显著性受上下文影响的表示，并且提出了专门的配对比较查询以有效隔离显著性与偏好，从而使学习更加高效。仿真人实验显示，我们的方法显著提高了样本效率，只需基线方法的十分之一的偏好查询即可达到同等的奖励准确性，特别是在数据稀缺的条件下（5-10个查询时）可以提高多达15%的性能。面对面用户研究（N=12）表明，参与者能够有效使用我们的方法教授其独特的个人上下文偏好，从而实现更具适应性和个性化的奖励学习。 

---
# Six-DoF Hand-Based Teleoperation for Omnidirectional Aerial Robots 

**Title (ZH)**: 基于手控的六自由度全向 aerial 机器人远程操作 

**Authors**: Jinjie Li, Jiaxuan Li, Kotaro Kaneko, Liming Shu, Moju Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15009)  

**Abstract**: Omnidirectional aerial robots offer full 6-DoF independent control over position and orientation, making them popular for aerial manipulation. Although advancements in robotic autonomy, operating by human remains essential in complex aerial environments. Existing teleoperation approaches for multirotors fail to fully leverage the additional DoFs provided by omnidirectional rotation. Additionally, the dexterity of human fingers should be exploited for more engaged interaction. In this work, we propose an aerial teleoperation system that brings the omnidirectionality of human hands into the unbounded aerial workspace. Our system includes two motion-tracking marker sets -- one on the shoulder and one on the hand -- along with a data glove to capture hand gestures. Using these inputs, we design four interaction modes for different tasks, including Spherical Mode and Cartesian Mode for long-range moving as well as Operation Mode and Locking Mode for precise manipulation, where the hand gestures are utilized for seamless mode switching. We evaluate our system on a valve-turning task in real world, demonstrating how each mode contributes to effective aerial manipulation. This interaction framework bridges human dexterity with aerial robotics, paving the way for enhanced teleoperated aerial manipulation in unstructured environments. 

**Abstract (ZH)**: 全方位飞行机器人提供了在位置和姿态上独立控制的6自由度运动，使得它们在空中操作中非常受欢迎。虽然机器人自主性的发展前景广阔，但在复杂空域中仍需人类操作员的干预。现有的多旋翼飞行器遥操作方法未能充分利用全方位旋转提供的额外自由度。此外，人类手指的灵巧性应被利用以实现更深入的交互。在此工作中，我们提出了一种遥操作系统，将人类手部的全方位特性引入无界限的空中工作空间。该系统包括肩部和手部的两个运动追踪标记集以及数据手套以捕捉手部手势。利用这些输入，我们设计了四种不同的交互模式，包括用于长距离移动的球形模式和笛卡尔模式，以及用于精确操作的操作模式和锁定模式，在这些模式中，手部手势被用于无缝模式切换。我们在一个阀门旋转变任务中评估了该系统，展示了每种模式如何有助于有效的空中操作。该交互框架将人类灵巧性与空中机器人结合起来，为在非结构化环境中实现增强的遥控空中操作铺平了道路。 

---
# Time-Optimized Safe Navigation in Unstructured Environments through Learning Based Depth Completion 

**Title (ZH)**: 基于学习的深度完成在非结构化环境中的时间优化安全导航 

**Authors**: Jeffrey Mao, Raghuram Cauligi Srinivas, Steven Nogar, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2506.14975)  

**Abstract**: Quadrotors hold significant promise for several applications such as agriculture, search and rescue, and infrastructure inspection. Achieving autonomous operation requires systems to navigate safely through complex and unfamiliar environments. This level of autonomy is particularly challenging due to the complexity of such environments and the need for real-time decision making especially for platforms constrained by size, weight, and power (SWaP), which limits flight time and precludes the use of bulky sensors like Light Detection and Ranging (LiDAR) for mapping. Furthermore, computing globally optimal, collision-free paths and translating them into time-optimized, safe trajectories in real time adds significant computational complexity. To address these challenges, we present a fully onboard, real-time navigation system that relies solely on lightweight onboard sensors. Our system constructs a dense 3D map of the environment using a novel visual depth estimation approach that fuses stereo and monocular learning-based depth, yielding longer-range, denser, and less noisy depth maps than conventional stereo methods. Building on this map, we introduce a novel planning and trajectory generation framework capable of rapidly computing time-optimal global trajectories. As the map is incrementally updated with new depth information, our system continuously refines the trajectory to maintain safety and optimality. Both our planner and trajectory generator outperforms state-of-the-art methods in terms of computational efficiency and guarantee obstacle-free trajectories. We validate our system through robust autonomous flight experiments in diverse indoor and outdoor environments, demonstrating its effectiveness for safe navigation in previously unknown settings. 

**Abstract (ZH)**: 四旋翼飞行器在农业、搜索与救援以及基础设施检查等领域具有重要的应用潜力。实现自主操作需要系统能够在复杂和不熟悉的环境中安全导航。由于环境的复杂性和对小型化、轻量化和低功耗（SWaP）平台而言实时决策的需要，这种级别的自主操作极具挑战性。此外，计算全局最优且无碰撞的路径并实时转化为时间优化的安全轨迹增加了巨大的计算复杂性。为应对这些挑战，我们提出了一种完全基于机载、实时导航系统，仅依赖轻量级机载传感器。该系统通过一种新颖的视觉深度估计方法融合立体视觉和单目学习深度信息，构建出范围更广、密度更高、噪声更低的三维地图，相比传统立体视觉方法具有明显优势。在此基础上，我们引入了一种新颖的规划与轨迹生成框架，能够快速计算时间最优的全球轨迹。随着地图通过新深度信息逐步更新，我们的系统不断优化轨迹以保持安全性和最优性。我们的规划器和轨迹生成器在计算效率和确保无碰撞轨迹方面均优于现有方法。通过在多种室内和室外环境中进行稳健的自主飞行实验，我们验证了该系统的有效性，证明其在未知环境中的导航能力。 

---
# FEAST: A Flexible Mealtime-Assistance System Towards In-the-Wild Personalization 

**Title (ZH)**: FEAST: 一种灵活的餐饮辅助系统，实现真实环境下的个性化服务 

**Authors**: Rajat Kumar Jenamani, Tom Silver, Ben Dodson, Shiqin Tong, Anthony Song, Yuting Yang, Ziang Liu, Benjamin Howe, Aimee Whitneck, Tapomayukh Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2506.14968)  

**Abstract**: Physical caregiving robots hold promise for improving the quality of life of millions worldwide who require assistance with feeding. However, in-home meal assistance remains challenging due to the diversity of activities (e.g., eating, drinking, mouth wiping), contexts (e.g., socializing, watching TV), food items, and user preferences that arise during deployment. In this work, we propose FEAST, a flexible mealtime-assistance system that can be personalized in-the-wild to meet the unique needs of individual care recipients. Developed in collaboration with two community researchers and informed by a formative study with a diverse group of care recipients, our system is guided by three key tenets for in-the-wild personalization: adaptability, transparency, and safety. FEAST embodies these principles through: (i) modular hardware that enables switching between assisted feeding, drinking, and mouth-wiping, (ii) diverse interaction methods, including a web interface, head gestures, and physical buttons, to accommodate diverse functional abilities and preferences, and (iii) parameterized behavior trees that can be safely and transparently adapted using a large language model. We evaluate our system based on the personalization requirements identified in our formative study, demonstrating that FEAST offers a wide range of transparent and safe adaptations and outperforms a state-of-the-art baseline limited to fixed customizations. To demonstrate real-world applicability, we conduct an in-home user study with two care recipients (who are community researchers), feeding them three meals each across three diverse scenarios. We further assess FEAST's ecological validity by evaluating with an Occupational Therapist previously unfamiliar with the system. In all cases, users successfully personalize FEAST to meet their individual needs and preferences. Website: this https URL 

**Abstract (ZH)**: 家用餐饮辅助机器人在提高全球数百万需要喂食帮助的人的生活质量方面前景广阔。然而，由于部署过程中存在多种活动（如进食、饮水、清洁口腔）、情境（如社交、看电视）、食品项目以及用户偏好，家庭餐饮辅助仍然具有挑战性。在这项工作中，我们提出了FEAST，这是一种灵活的餐时辅助系统，可以在实际环境中个性化以满足个体护理接收者的独特需求。该系统在与两位社区研究人员合作并根据来自多样护理接收者的形成性研究中得到启发下开发，旨在遵循三项关键原则进行实际环境个性化：适应性、透明性和安全性。FEAST通过以下方式体现这些原则：(i) 模块化硬件，允许在喂食、饮水和清洁口腔之间切换，(ii) 多样的交互方式，包括网页界面、头部手势和物理按钮，以适应不同的功能能力和偏好，以及(iii) 可根据大型语言模型安全透明地进行参数化行为树的适应。我们根据形成性研究中确定的个性化需求评估系统，表明FEAST提供了广泛的透明和安全的适应性，并优于仅限于固定定制的最先进的基线系统。为了展示实际应用的可行性，我们在两名护理接收者（他们也是社区研究人员）之间进行了家庭用户研究，各自喂食三次，跨越三种不同的场景。我们还通过一位对系统不熟悉的职业治疗师的评估来进一步检验FEAST的生态效度。在所有情况下，用户都能成功地将FEAST个性化以满足其个别需求和偏好。网站：这个 https URL。 

---
# Efficient and Real-Time Motion Planning for Robotics Using Projection-Based Optimization 

**Title (ZH)**: 基于投影优化的机器人高效实时运动规划 

**Authors**: Xuemin Chi, Hakan Girgin, Tobias Löw, Yangyang Xie, Teng Xue, Jihao Huang, Cheng Hu, Zhitao Liu, Sylvain Calinon  

**Link**: [PDF](https://arxiv.org/pdf/2506.14865)  

**Abstract**: Generating motions for robots interacting with objects of various shapes is a complex challenge, further complicated by the robot geometry and multiple desired behaviors. While current robot programming tools (such as inverse kinematics, collision avoidance, and manipulation planning) often treat these problems as constrained optimization, many existing solvers focus on specific problem domains or do not exploit geometric constraints effectively. We propose an efficient first-order method, Augmented Lagrangian Spectral Projected Gradient Descent (ALSPG), which leverages geometric projections via Euclidean projections, Minkowski sums, and basis functions. We show that by using geometric constraints rather than full constraints and gradients, ALSPG significantly improves real-time performance. Compared to second-order methods like iLQR, ALSPG remains competitive in the unconstrained case. We validate our method through toy examples and extensive simulations, and demonstrate its effectiveness on a 7-axis Franka robot, a 6-axis P-Rob robot and a 1:10 scale car in real-world experiments. Source codes, experimental data and videos are available on the project webpage: this https URL 

**Abstract (ZH)**: 生成与不同形状物体交互的机器人运动是一项复杂的挑战，进一步复杂化的因素包括机器人几何结构和多种期望行为。当前的机器人编程工具（如逆运动学、碰撞避免和操作规划）通常将这些问题视为约束优化问题，但现有的许多求解器往往专注于特定的问题领域，或者未能有效地利用几何约束。我们提出了一种高效的首阶方法——增广拉格朗日谱投影梯度下降（ALSPG），该方法利用欧几里得投影、闵可夫斯基和及基函数进行几何投影。我们通过利用几何约束而非完全的约束和梯度，显著提高了实时性能。与二次方法（如iLQR）相比，在无约束的情况下，ALSPG仍然具有竞争力。我们通过玩具示例和广泛仿真验证了该方法，并在真实世界实验中展示了其在7轴Franka机器人、6轴P-Rob机器人及1:10比例汽车上的有效性。相关源代码、实验数据和视频可在项目网页上获取：this https URL。 

---
# Towards Perception-based Collision Avoidance for UAVs when Guiding the Visually Impaired 

**Title (ZH)**: 基于感知的无人机导盲避碰方法 

**Authors**: Suman Raj, Swapnil Padhi, Ruchi Bhoot, Prince Modi, Yogesh Simmhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.14857)  

**Abstract**: Autonomous navigation by drones using onboard sensors combined with machine learning and computer vision algorithms is impacting a number of domains, including agriculture, logistics, and disaster management. In this paper, we examine the use of drones for assisting visually impaired people (VIPs) in navigating through outdoor urban environments. Specifically, we present a perception-based path planning system for local planning around the neighborhood of the VIP, integrated with a global planner based on GPS and maps for coarse planning. We represent the problem using a geometric formulation and propose a multi DNN based framework for obstacle avoidance of the UAV as well as the VIP. Our evaluations conducted on a drone human system in a university campus environment verifies the feasibility of our algorithms in three scenarios; when the VIP walks on a footpath, near parked vehicles, and in a crowded street. 

**Abstract (ZH)**: 利用机载传感器结合机器学习和计算机视觉算法的无人机自主导航技术正影响着农业、物流和灾难管理等多个领域。本文探讨了无人机在辅助视觉 impaired 人士（VIPs）导航通过户外城市环境中的应用。具体来说，我们提出了一种基于感知的路径规划系统，用于在VIP周围区域进行局部规划，并与基于GPS和地图的全局规划器结合进行粗略规划。我们使用几何形式表示该问题，并提出了一种基于多DNN的框架，用于避免无人机和VIP的障碍物。在大学校园环境下对无人机人类系统进行的评估验证了我们在三种场景下的算法可行性：当VIP走在人行道上、靠近停放的车辆附近以及在拥挤的街道上时。 

---
# Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation -- Adios low-level controllers 

**Title (ZH)**: Feedback-MPPI: 快速基于采样的MPC通过回推微分—— farewell低级控制器 

**Authors**: Tommaso Belvedere, Michael Ziegltrum, Giulio Turrisi, Valerio Modugno  

**Link**: [PDF](https://arxiv.org/pdf/2506.14855)  

**Abstract**: Model Predictive Path Integral control is a powerful sampling-based approach suitable for complex robotic tasks due to its flexibility in handling nonlinear dynamics and non-convex costs. However, its applicability in real-time, highfrequency robotic control scenarios is limited by computational demands. This paper introduces Feedback-MPPI (F-MPPI), a novel framework that augments standard MPPI by computing local linear feedback gains derived from sensitivity analysis inspired by Riccati-based feedback used in gradient-based MPC. These gains allow for rapid closed-loop corrections around the current state without requiring full re-optimization at each timestep. We demonstrate the effectiveness of F-MPPI through simulations and real-world experiments on two robotic platforms: a quadrupedal robot performing dynamic locomotion on uneven terrain and a quadrotor executing aggressive maneuvers with onboard computation. Results illustrate that incorporating local feedback significantly improves control performance and stability, enabling robust, high-frequency operation suitable for complex robotic systems. 

**Abstract (ZH)**: 基于路径积分的反馈模型预测控制：一种适用于复杂机器人任务的新型快速闭环校正框架 

---
# Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence 

**Title (ZH)**: 具身网络代理：连接物理与数字领域以实现集成代理智能 

**Authors**: Yining Hong, Rui Sun, Bingxuan Li, Xingcheng Yao, Maxine Wu, Alexander Chien, Da Yin, Ying Nian Wu, Zhecan James Wang, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15677)  

**Abstract**: AI agents today are mostly siloed - they either retrieve and reason over vast amount of digital information and knowledge obtained online; or interact with the physical world through embodied perception, planning and action - but rarely both. This separation limits their ability to solve tasks that require integrated physical and digital intelligence, such as cooking from online recipes, navigating with dynamic map data, or interpreting real-world landmarks using web knowledge. We introduce Embodied Web Agents, a novel paradigm for AI agents that fluidly bridge embodiment and web-scale reasoning. To operationalize this concept, we first develop the Embodied Web Agents task environments, a unified simulation platform that tightly integrates realistic 3D indoor and outdoor environments with functional web interfaces. Building upon this platform, we construct and release the Embodied Web Agents Benchmark, which encompasses a diverse suite of tasks including cooking, navigation, shopping, tourism, and geolocation - all requiring coordinated reasoning across physical and digital realms for systematic assessment of cross-domain intelligence. Experimental results reveal significant performance gaps between state-of-the-art AI systems and human capabilities, establishing both challenges and opportunities at the intersection of embodied cognition and web-scale knowledge access. All datasets, codes and websites are publicly available at our project page this https URL. 

**Abstract (ZH)**: 当前的AI代理大多是孤立的——它们要么在线检索和推理大量数字信息和知识；要么通过体素感知、规划和行动与物理世界互动——但很少两者兼而有之。这种分离限制了它们解决需要结合物理和数字智能的任务的能力，例如根据在线食谱烹饪、使用动态地图数据导航或使用网络知识解释现实世界地标。我们引入了体素网络代理这一新的AI代理范式，能够流畅地将体素感知与大规模网络推理相结合。为了实现这一概念，我们首先开发了体素网络代理任务环境，这是一个将逼真的3D室内外环境与功能性网络界面紧密结合的统一仿真平台。在此基础上，我们构建并发布了体素网络代理基准，该基准涵盖了烹饪、导航、购物、旅游和地理定位等多种任务，所有任务都需要在物理和数字领域进行协调推理，以便系统评估跨域智能。实验结果揭示了最先进的AI系统与人类能力之间的显著性能差距，明确了体化认知与大规模网络知识访问交叉领域中的挑战和机遇。所有数据集、代码和网站均可在我们的项目页面上公开访问：this https URL。 

---
# FindingDory: A Benchmark to Evaluate Memory in Embodied Agents 

**Title (ZH)**: FindingDory：评估具身智能体记忆能力的标准 benchmark 

**Authors**: Karmesh Yadav, Yusuf Ali, Gunshi Gupta, Yarin Gal, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2506.15635)  

**Abstract**: Large vision-language models have recently demonstrated impressive performance in planning and control tasks, driving interest in their application to real-world robotics. However, deploying these models for reasoning in embodied contexts is limited by their ability to incorporate long-term experience collected across multiple days and represented by vast collections of images. Current VLMs typically struggle to process more than a few hundred images concurrently, highlighting the need for more efficient mechanisms to handle long-term memory in embodied settings. To effectively evaluate these models for long-horizon control, a benchmark must specifically target scenarios where memory is crucial for success. Existing long-video QA benchmarks overlook embodied challenges like object manipulation and navigation, which demand low-level skills and fine-grained reasoning over past interactions. Moreover, effective memory integration in embodied agents involves both recalling relevant historical information and executing actions based on that information, making it essential to study these aspects together rather than in isolation. In this work, we introduce a new benchmark for long-range embodied tasks in the Habitat simulator. This benchmark evaluates memory-based capabilities across 60 tasks requiring sustained engagement and contextual awareness in an environment. The tasks can also be procedurally extended to longer and more challenging versions, enabling scalable evaluation of memory and reasoning. We also present baselines that integrate state-of-the-art VLMs with low level navigation policies, assessing their performance on these memory-intensive tasks and highlight areas for improvement. 

**Abstract (ZH)**: 大型多模态视觉语言模型在规划和控制任务中 recently demonstrated impressive performance, driving interest in their application to real-world robotics. However, deploying these models for reasoning in embodied contexts is limited by their ability to incorporate long-term experience collected across multiple days and represented by vast collections of images. Current VLMs typically struggle to process more than a few hundred images concurrently, highlighting the need for more efficient mechanisms to handle long-term memory in embodied settings. To effectively evaluate these models for long-horizon control, a benchmark must specifically target scenarios where memory is crucial for success. Existing long-video QA benchmarks overlook embodied challenges like object manipulation and navigation, which demand low-level skills and fine-grained reasoning over past interactions. Moreover, effective memory integration in embodied agents involves both recalling relevant historical information and executing actions based on that information, making it essential to study these aspects together rather than in isolation. In this work, we introduce a new benchmark for long-range embodied tasks in the Habitat simulator. This benchmark evaluates memory-based capabilities across 60 tasks requiring sustained engagement and contextual awareness in an environment. The tasks can also be procedurally extended to longer and more challenging versions, enabling scalable evaluation of memory and reasoning. We also present baselines that integrate state-of-the-art VLMs with low-level navigation policies, assessing their performance on these memory-intensive tasks and highlight areas for improvement。 

---
# RaCalNet: Radar Calibration Network for Sparse-Supervised Metric Depth Estimation 

**Title (ZH)**: RaCalNet：稀疏监督度量深度估计的雷达校准网络 

**Authors**: Xingrui Qin, Wentao Zhao, Chuan Cao, Yihe Niu, Houcheng Jiang, Jingchuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15560)  

**Abstract**: Dense metric depth estimation using millimeter-wave radar typically requires dense LiDAR supervision, generated via multi-frame projection and interpolation, to guide the learning of accurate depth from sparse radar measurements and RGB images. However, this paradigm is both costly and data-intensive. To address this, we propose RaCalNet, a novel framework that eliminates the need for dense supervision by using sparse LiDAR to supervise the learning of refined radar measurements, resulting in a supervision density of merely around 1% compared to dense-supervised methods. Unlike previous approaches that associate radar points with broad image regions and rely heavily on dense labels, RaCalNet first recalibrates and refines sparse radar points to construct accurate depth priors. These priors then serve as reliable anchors to guide monocular depth prediction, enabling metric-scale estimation without resorting to dense supervision. This design improves structural consistency and preserves fine details. Despite relying solely on sparse supervision, RaCalNet surpasses state-of-the-art dense-supervised methods, producing depth maps with clear object contours and fine-grained textures. Extensive experiments on the ZJU-4DRadarCam dataset and real-world deployment scenarios demonstrate its effectiveness, reducing RMSE by 35.30% and 34.89%, respectively. 

**Abstract (ZH)**: 基于毫米波雷达的密集度量深度估计通常需要通过多帧投影和插值生成的密集激光雷达监督，以指导从稀疏雷达测量和RGB图像中学习准确深度。为了解决这一问题，我们提出了一种新的框架RaCalNet，通过使用稀疏激光雷达监督稀疏雷达测量的学习，监督密度仅为密集监督方法的约1%。不同于以往方法将雷达点与广泛的图像区域关联并依赖密集标签，RaCalNet首先校准和细化稀疏雷达点以构建准确的深度先验，这些先验作为可靠的锚点引导单目深度预测，从而在无需密集监督的情况下实现度量级估计。这种设计增强了结构一致性并保留了精细细节。尽管仅依赖稀疏监督，RaCalNet仍超越了最新的密集监督方法，生成清晰的物体轮廓和细腻的纹理图。在ZJU-4DRadarCam数据集和实际部署场景中的广泛实验表明其有效性，分别将RMSE降低了35.30%和34.89%。 

---
# Model Predictive Path-Following Control for a Quadrotor 

**Title (ZH)**: 四旋翼飞行器的模型预测路径跟踪控制 

**Authors**: David Leprich, Mario Rosenfelder, Mario Hermle, Jingshan Chen, Peter Eberhard  

**Link**: [PDF](https://arxiv.org/pdf/2506.15447)  

**Abstract**: Automating drone-assisted processes is a complex task. Many solutions rely on trajectory generation and tracking, whereas in contrast, path-following control is a particularly promising approach, offering an intuitive and natural approach to automate tasks for drones and other vehicles. While different solutions to the path-following problem have been proposed, most of them lack the capability to explicitly handle state and input constraints, are formulated in a conservative two-stage approach, or are only applicable to linear systems. To address these challenges, the paper is built upon a Model Predictive Control-based path-following framework and extends its application to the Crazyflie quadrotor, which is investigated in hardware experiments. A cascaded control structure including an underlying attitude controller is included in the Model Predictive Path-Following Control formulation to meet the challenging real-time demands of quadrotor control. The effectiveness of the proposed method is demonstrated through real-world experiments, representing, to the best of the authors' knowledge, a novel application of this MPC-based path-following approach to the quadrotor. Additionally, as an extension to the original method, to allow for deviations of the path in cases where the precise following of the path might be overly restrictive, a corridor path-following approach is presented. 

**Abstract (ZH)**: 基于模型预测控制的路径跟随方法在 Crazyflie 四旋翼无人机上的应用及扩展研究 

---
# Designing Intent: A Multimodal Framework for Human-Robot Cooperation in Industrial Workspaces 

**Title (ZH)**: 设计意图：工业工作空间中人机合作的多模态框架 

**Authors**: Francesco Chiossi, Julian Rasch, Robin Welsch, Albrecht Schmidt, Florian Michahelles  

**Link**: [PDF](https://arxiv.org/pdf/2506.15293)  

**Abstract**: As robots enter collaborative workspaces, ensuring mutual understanding between human workers and robotic systems becomes a prerequisite for trust, safety, and efficiency. In this position paper, we draw on the cooperation scenario of the AIMotive project in which a human and a cobot jointly perform assembly tasks to argue for a structured approach to intent communication. Building on the Situation Awareness-based Agent Transparency (SAT) framework and the notion of task abstraction levels, we propose a multidimensional design space that maps intent content (SAT1, SAT3), planning horizon (operational to strategic), and modality (visual, auditory, haptic). We illustrate how this space can guide the design of multimodal communication strategies tailored to dynamic collaborative work contexts. With this paper, we lay the conceptual foundation for a future design toolkit aimed at supporting transparent human-robot interaction in the workplace. We highlight key open questions and design challenges, and propose a shared agenda for multimodal, adaptive, and trustworthy robotic collaboration in hybrid work environments. 

**Abstract (ZH)**: 随着机器人进入协作 workspace，确保人类工人与机器人系统之间的相互理解成为信任、安全和效率的前提。在本文中，我们借助 AIMotive 项目中的合作场景，即人类和协作机器人共同执行装配任务，论述了一种结构化的意图沟通方法。基于情境意识为基础的代理透明度（SAT）框架和任务抽象层次的概念，我们提出一个多维设计空间，该空间映射意图内容（SAT1, SAT3）、规划 horizon（从操作性到战略性的差异）以及模态（视觉、听觉、触觉）。我们展示了如何利用这一空间来指导适应动态协作工作环境的多模态沟通策略的设计。通过本文，我们为支持工作场所透明人机交互的未来设计工具包奠定了概念基础。我们强调了关键的开放问题和设计挑战，并提出了一项关于多模态、自适应和可信机器人协作在混合工作环境中的共同议程。 

---
# Minimizing Structural Vibrations via Guided Flow Matching Design Optimization 

**Title (ZH)**: 通过引导流匹配设计优化来最小化结构振动 

**Authors**: Jan van Delden, Julius Schultz, Sebastian Rothe, Christian Libner, Sabine C. Langer, Timo Lüddecke  

**Link**: [PDF](https://arxiv.org/pdf/2506.15263)  

**Abstract**: Structural vibrations are a source of unwanted noise in engineering systems like cars, trains or airplanes. Minimizing these vibrations is crucial for improving passenger comfort. This work presents a novel design optimization approach based on guided flow matching for reducing vibrations by placing beadings (indentations) in plate-like structures. Our method integrates a generative flow matching model and a surrogate model trained to predict structural vibrations. During the generation process, the flow matching model pushes towards manufacturability while the surrogate model pushes to low-vibration solutions. The flow matching model and its training data implicitly define the design space, enabling a broader exploration of potential solutions as no optimization of manually-defined design parameters is required. We apply our method to a range of differentiable optimization objectives, including direct optimization of specific eigenfrequencies through careful construction of the objective function. Results demonstrate that our method generates diverse and manufacturable plate designs with reduced structural vibrations compared to designs from random search, a criterion-based design heuristic and genetic optimization. The code and data are available from this https URL. 

**Abstract (ZH)**: 结构振动是汽车、列车或飞机等工程系统中的一个不必要的噪声来源。减少这些振动对于提高乘客舒适度至关重要。本文提出了一种基于引导流匹配的新型设计优化方法，通过在板状结构中放置凸台（凹陷）来减少振动。该方法结合了生成流匹配模型和一个用于预测结构振动的代理模型。在生成过程中，流匹配模型推动模型更具可制造性，而代理模型则推动低振动解决方案。流匹配模型及其训练数据隐式定义了设计空间，使得无需优化手动定义的设计参数即可进行更广泛的设计探索。我们将该方法应用于各种可微优化目标，包括通过精心构建目标函数直接优化特定的固有频率。结果表明，与随机搜索、基于准则的设计启发式方法和遗传优化生成的设计相比，我们的方法能够生成具有减少结构振动的多样化且可制造的板状设计。代码和数据可从该网址获得。 

---
# Multi-Agent Reinforcement Learning for Autonomous Multi-Satellite Earth Observation: A Realistic Case Study 

**Title (ZH)**: 多智能体强化学习在自主多卫星地球观测中的应用：一个现实的案例研究 

**Authors**: Mohamad A. Hady, Siyi Hu, Mahardhika Pratama, Jimmy Cao, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2506.15207)  

**Abstract**: The exponential growth of Low Earth Orbit (LEO) satellites has revolutionised Earth Observation (EO) missions, addressing challenges in climate monitoring, disaster management, and more. However, autonomous coordination in multi-satellite systems remains a fundamental challenge. Traditional optimisation approaches struggle to handle the real-time decision-making demands of dynamic EO missions, necessitating the use of Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL). In this paper, we investigate RL-based autonomous EO mission planning by modelling single-satellite operations and extending to multi-satellite constellations using MARL frameworks. We address key challenges, including energy and data storage limitations, uncertainties in satellite observations, and the complexities of decentralised coordination under partial observability. By leveraging a near-realistic satellite simulation environment, we evaluate the training stability and performance of state-of-the-art MARL algorithms, including PPO, IPPO, MAPPO, and HAPPO. Our results demonstrate that MARL can effectively balance imaging and resource management while addressing non-stationarity and reward interdependency in multi-satellite coordination. The insights gained from this study provide a foundation for autonomous satellite operations, offering practical guidelines for improving policy learning in decentralised EO missions. 

**Abstract (ZH)**: 低地球轨道卫星的指数增长已 revolutionised 地球观测任务，解决了气候监测、灾害管理等方面的问题。然而，多颗卫星系统的自主协调仍然是一个基本挑战。传统优化方法难以应对动态地球观测任务的实时决策需求，因此需要使用强化学习（RL）和多智能体强化学习（MARL）。在本文中，我们通过建模单颗卫星操作并扩展到多颗卫星星座，利用MARL框架研究基于RL的自主地球观测任务规划。我们解决了包括能源和数据存储限制、卫星观测不确定性以及部分可观测下去中心化协调的复杂性在内的关键挑战。通过利用接近真实的卫星仿真环境，我们评估了包括PPO、IPPO、MAPPO和HAPPO在内的先进MARL算法的训练稳定性和性能。研究结果表明，MARL能够在多颗卫星协调中有效平衡成像和资源管理，同时处理非平稳性和奖励的相互依赖性。本研究获得的见解为自主卫星操作提供了基础，并为改进分布式地球观测任务中的策略学习提供了实用指南。 

---
# Probabilistic Trajectory GOSPA: A Metric for Uncertainty-Aware Multi-Object Tracking Performance Evaluation 

**Title (ZH)**: 概率轨迹GOSPA：一种面向不确定性多目标跟踪性能评估的度量标准 

**Authors**: Yuxuan Xia, Ángel F. García-Fernández, Johan Karlsson, Yu Ge, Lennart Svensson, Ting Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15148)  

**Abstract**: This paper presents a generalization of the trajectory general optimal sub-pattern assignment (GOSPA) metric for evaluating multi-object tracking algorithms that provide trajectory estimates with track-level uncertainties. This metric builds on the recently introduced probabilistic GOSPA metric to account for both the existence and state estimation uncertainties of individual object states. Similar to trajectory GOSPA (TGOSPA), it can be formulated as a multidimensional assignment problem, and its linear programming relaxation--also a valid metric--is computable in polynomial time. Additionally, this metric retains the interpretability of TGOSPA, and we show that its decomposition yields intuitive costs terms associated to expected localization error and existence probability mismatch error for properly detected objects, expected missed and false detection error, and track switch error. The effectiveness of the proposed metric is demonstrated through a simulation study. 

**Abstract (ZH)**: 本文提出了一种轨迹广义最优子模式分配（GOSPA）度量的一般化方法，用于评估提供具有轨迹级不确定性轨迹估计的多目标跟踪算法。该度量基于最近引入的概率GOSPA度量，以计算个体对象状态的存在性和状态估计不确定性。类似于轨迹GOSPA（TGOSPA），它可以形式化为一个多维分配问题，其线性规划 relaxation 也是一个有效的度量，并且可以在多项式时间内计算。此外，该度量保留了TGOSPA的可解释性，我们展示了其分解产生了与正确检测对象的期望定位误差和存在概率不匹配误差、预期的遗漏和假检测误差以及轨迹切换误差相关的直观成本项。通过仿真研究证明了所提出度量的有效性。 

---
# HEAL: An Empirical Study on Hallucinations in Embodied Agents Driven by Large Language Models 

**Title (ZH)**: HEAL：大型语言模型驱动的具身代理幻觉现象的实证研究 

**Authors**: Trishna Chakraborty, Udita Ghosh, Xiaopan Zhang, Fahim Faisal Niloy, Yue Dong, Jiachen Li, Amit K. Roy-Chowdhury, Chengyu Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.15065)  

**Abstract**: Large language models (LLMs) are increasingly being adopted as the cognitive core of embodied agents. However, inherited hallucinations, which stem from failures to ground user instructions in the observed physical environment, can lead to navigation errors, such as searching for a refrigerator that does not exist. In this paper, we present the first systematic study of hallucinations in LLM-based embodied agents performing long-horizon tasks under scene-task inconsistencies. Our goal is to understand to what extent hallucinations occur, what types of inconsistencies trigger them, and how current models respond. To achieve these goals, we construct a hallucination probing set by building on an existing benchmark, capable of inducing hallucination rates up to 40x higher than base prompts. Evaluating 12 models across two simulation environments, we find that while models exhibit reasoning, they fail to resolve scene-task inconsistencies-highlighting fundamental limitations in handling infeasible tasks. We also provide actionable insights on ideal model behavior for each scenario, offering guidance for developing more robust and reliable planning strategies. 

**Abstract (ZH)**: 基于大语言模型的体素代理在场景任务不一致情况下长时 horizon 任务中幻觉现象的系统研究 

---
# Advances in Compliance Detection: Novel Models Using Vision-Based Tactile Sensors 

**Title (ZH)**: 基于视觉触觉传感器的合规性检测进展 

**Authors**: Ziteng Li, Malte Kuhlmann, Ilana Nisky, Nicolás Navarro-Guerrero  

**Link**: [PDF](https://arxiv.org/pdf/2506.14980)  

**Abstract**: Compliance is a critical parameter for describing objects in engineering, agriculture, and biomedical applications. Traditional compliance detection methods are limited by their lack of portability and scalability, rely on specialized, often expensive equipment, and are unsuitable for robotic applications. Moreover, existing neural network-based approaches using vision-based tactile sensors still suffer from insufficient prediction accuracy. In this paper, we propose two models based on Long-term Recurrent Convolutional Networks (LRCNs) and Transformer architectures that leverage RGB tactile images and other information captured by the vision-based sensor GelSight to predict compliance metrics accurately. We validate the performance of these models using multiple metrics and demonstrate their effectiveness in accurately estimating compliance. The proposed models exhibit significant performance improvement over the baseline. Additionally, we investigated the correlation between sensor compliance and object compliance estimation, which revealed that objects that are harder than the sensor are more challenging to estimate. 

**Abstract (ZH)**: 合规性是描述工程、农业和生物医学应用中物体的重要参数。传统的合规性检测方法受限于其缺乏便携性和可扩展性，依赖于专门且通常昂贵的设备，并不适合用于机器人应用。此外，现有基于视觉触觉传感器的神经网络方法仍然存在预测精度不足的问题。本文提出两种基于长期递归卷积网络（LRCNs）和 Transformer 架构的模型，利用 GelSight 视觉触觉传感器捕获的 RGB 触觉图像和其他信息，准确预测合规性指标。我们使用多种评估指标验证了这些模型的性能，并展示了它们在准确估计合规性方面的有效性。所提出的模型在基线模型上表现出显著的性能提升。此外，我们研究了传感器合规性与物体合规性估计之间的相关性，结果显示比传感器更硬的物体更难估计。 

---
# Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review 

**Title (ZH)**: 近期多Agent人类轨迹预测的研究进展：一篇综合Review 

**Authors**: Céline Finet, Stephane Da Silva Martins, Jean-Bernard Hayet, Ioannis Karamouzas, Javad Amirian, Sylvie Le Hégarat-Mascle, Julien Pettré, Emanuel Aldea  

**Link**: [PDF](https://arxiv.org/pdf/2506.14831)  

**Abstract**: With the emergence of powerful data-driven methods in human trajectory prediction (HTP), gaining a finer understanding of multi-agent interactions lies within hand's reach, with important implications in areas such as autonomous navigation and crowd modeling. This survey reviews some of the most recent advancements in deep learning-based multi-agent trajectory prediction, focusing on studies published between 2020 and 2024. We categorize the existing methods based on their architectural design, their input representations, and their overall prediction strategies, placing a particular emphasis on models evaluated using the ETH/UCY benchmark. Furthermore, we highlight key challenges and future research directions in the field of multi-agent HTP. 

**Abstract (ZH)**: 基于深度学习的多agent轨迹预测 Recent advancements and future research directions in human trajectory prediction 

---
