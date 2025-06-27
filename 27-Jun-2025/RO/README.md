# WorldVLA: Towards Autoregressive Action World Model 

**Title (ZH)**: WorldVLA：迈向自回归动作世界模型 

**Authors**: Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21539)  

**Abstract**: We present WorldVLA, an autoregressive action world model that unifies action and image understanding and generation. Our WorldVLA intergrates Vision-Language-Action (VLA) model and world model in one single framework. The world model predicts future images by leveraging both action and image understanding, with the purpose of learning the underlying physics of the environment to improve action generation. Meanwhile, the action model generates the subsequent actions based on image observations, aiding in visual understanding and in turn helps visual generation of the world model. We demonstrate that WorldVLA outperforms standalone action and world models, highlighting the mutual enhancement between the world model and the action model. In addition, we find that the performance of the action model deteriorates when generating sequences of actions in an autoregressive manner. This phenomenon can be attributed to the model's limited generalization capability for action prediction, leading to the propagation of errors from earlier actions to subsequent ones. To address this issue, we propose an attention mask strategy that selectively masks prior actions during the generation of the current action, which shows significant performance improvement in the action chunk generation task. 

**Abstract (ZH)**: WorldVLA：一个统一动作与图像理解与生成的自回归世界模型 

---
# Real-time Terrain Analysis for Off-road Autonomous Vehicles 

**Title (ZH)**: 离线自主车辆的实时地形分析 

**Authors**: Edwina Lewis, Aditya Parameshwaran, Laura Redmond, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21347)  

**Abstract**: This research addresses critical autonomous vehicle control challenges arising from road roughness variation, which induces course deviations and potential loss of road contact during steering operations. We present a novel real-time road roughness estimation system employing Bayesian calibration methodology that processes axle accelerations to predict terrain roughness with quantifiable confidence measures. The technical framework integrates a Gaussian process surrogate model with a simulated half-vehicle model, systematically processing vehicle velocity and road surface roughness parameters to generate corresponding axle acceleration responses. The Bayesian calibration routine performs inverse estimation of road roughness from observed accelerations and velocities, yielding posterior distributions that quantify prediction uncertainty for adaptive risk management. Training data generation utilizes Latin Hypercube sampling across comprehensive velocity and roughness parameter spaces, while the calibrated model integrates seamlessly with a Simplex controller architecture to dynamically adjust velocity limits based on real-time roughness predictions. Experimental validation on stochastically generated surfaces featuring varying roughness regions demonstrates robust real-time characterization capabilities, with the integrated Simplex control strategy effectively enhancing autonomous vehicle operational safety through proactive surface condition response. This innovative Bayesian framework establishes a comprehensive foundation for mitigating roughness-related operational risks while simultaneously improving efficiency and safety margins in autonomous vehicle systems. 

**Abstract (ZH)**: 这篇研究着眼于路面粗糙度变化引发的关键自动驾驶车辆控制挑战，这些挑战会在转向操作过程中导致路径偏差和路面接触丢失。我们提出了一种新型的实时路面粗糙度估计系统，该系统采用贝叶斯校准方法处理轴加速度，以定量置信度预测地形粗糙度。该技术框架结合了高斯过程代理模型和模拟半车辆模型，系统地处理车辆速度和路面粗糙度参数，生成相应的轴加速度响应。贝叶斯校准流程从观测的加速度和速度反向估计路面粗糙度，生成后验分布以量化预测不确定性，从而实现自适应风险管理。训练数据生成利用拉丁超立方抽样覆盖全面的速度和粗糙度参数空间，而校准模型无缝集成到简单形控制器架构中，基于实时粗糙度预测动态调整速度限制。在包含不同粗糙度区域的随机生成路面上的实验验证展示了 robust 的实时表征能力，而集成的简单形控制策略通过主动应对路面条件有效提升了自动驾驶车辆的操作安全性。这一创新的贝叶斯框架为减轻与粗糙度相关的操作风险奠定了全面的基础，同时提高了自动驾驶车辆系统的工作效率和安全裕度。 

---
# Active Disturbance Rejection Control for Trajectory Tracking of a Seagoing USV: Design, Simulation, and Field Experiments 

**Title (ZH)**: 基于主动干扰 rejection 控制的海上USV轨迹跟踪控制设计、仿真与场试验 

**Authors**: Jelmer van der Saag, Elia Trevisan, Wouter Falkena, Javier Alonso-Mora  

**Link**: [PDF](https://arxiv.org/pdf/2506.21265)  

**Abstract**: Unmanned Surface Vessels (USVs) face significant control challenges due to uncertain environmental disturbances like waves and currents. This paper proposes a trajectory tracking controller based on Active Disturbance Rejection Control (ADRC) implemented on the DUS V2500. A custom simulation incorporating realistic waves and current disturbances is developed to validate the controller's performance, supported by further validation through field tests in the harbour of Scheveningen, the Netherlands, and at sea. Simulation results demonstrate that ADRC significantly reduces cross-track error across all tested conditions compared to a baseline PID controller but increases control effort and energy consumption. Field trials confirm these findings while revealing a further increase in energy consumption during sea trials compared to the baseline. 

**Abstract (ZH)**: 无人水面船舶（USVs）面对来自波浪和洋流等不确定性环境干扰的显著控制挑战。本文提出了一种基于活性干扰拒绝控制（ADRC）的轨迹跟踪控制器，并在DUS V2500上进行了实现。一个包含真实波浪和水流干扰的自定义仿真被开发出来以验证控制器的性能，并通过在荷兰斯赫维宁恩港和海上进行的进一步现场试验来验证。仿真结果表明，与基线PID控制器相比，ADRC在所有测试条件下都能显著降低横向偏差，但会增加控制努力和能耗。现场试验证实了这些发现，并揭示了与基线相比，在海上试验中能耗进一步增加。 

---
# ACTLLM: Action Consistency Tuned Large Language Model 

**Title (ZH)**: ACTLLM: 行动一致性调优的大语言模型 

**Authors**: Jing Bi, Lianggong Bruce Wen, Zhang Liu, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21250)  

**Abstract**: This paper introduces ACTLLM (Action Consistency Tuned Large Language Model), a novel approach for robot manipulation in dynamic environments. Traditional vision-based systems often struggle to learn visual representations that excel in both task execution and spatial reasoning, thereby limiting their adaptability in dynamic environments. ACTLLM addresses these challenges by harnessing language to craft structured scene descriptors, providing a uniform interface for both spatial understanding and task performance through flexible language instructions. Moreover, we introduce a novel action consistency constraint that aligns visual perception with corresponding actions, thereby enhancing the learning of actionable visual representations. Additionally, we have reformulated the Markov decision process for manipulation tasks into a multi-turn visual dialogue framework. This approach enables the modeling of long-term task execution with enhanced contextual relevance derived from the history of task execution. During our evaluation, ACTLLM excels in diverse scenarios, proving its effectiveness on challenging vision-based robot manipulation tasks. 

**Abstract (ZH)**: ACTLLM：动作一致性调优的大语言模型在动态环境中的机器人操作方法 

---
# Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations 

**Title (ZH)**: 基于高效蒙特卡洛近似的动态风险意识MPPI在人群中的移动机器人路径规划 

**Authors**: Elia Trevisan, Khaled A. Mustafa, Godert Notten, Xinwei Wang, Javier Alonso-Mora  

**Link**: [PDF](https://arxiv.org/pdf/2506.21205)  

**Abstract**: Deploying mobile robots safely among humans requires the motion planner to account for the uncertainty in the other agents' predicted trajectories. This remains challenging in traditional approaches, especially with arbitrarily shaped predictions and real-time constraints. To address these challenges, we propose a Dynamic Risk-Aware Model Predictive Path Integral control (DRA-MPPI), a motion planner that incorporates uncertain future motions modelled with potentially non-Gaussian stochastic predictions. By leveraging MPPI's gradient-free nature, we propose a method that efficiently approximates the joint Collision Probability (CP) among multiple dynamic obstacles for several hundred sampled trajectories in real-time via a Monte Carlo (MC) approach. This enables the rejection of samples exceeding a predefined CP threshold or the integration of CP as a weighted objective within the navigation cost function. Consequently, DRA-MPPI mitigates the freezing robot problem while enhancing safety. Real-world and simulated experiments with multiple dynamic obstacles demonstrate DRA-MPPI's superior performance compared to state-of-the-art approaches, including Scenario-based Model Predictive Control (S-MPC), Frenet planner, and vanilla MPPI. 

**Abstract (ZH)**: 部署移动机器人在人类环境中安全移动需要运动规划器考虑其他代理预测轨迹中的不确定性。传统方法在处理任意形状的预测和实时约束时仍面临挑战。为解决这些挑战，我们提出了一种动态风险意识模型预测路径积分控制（DRA-MPPI），这是一种将潜在非高斯随机预测建模的不确定未来运动纳入考量的运动规划器。通过利用MPPI的无导数性质，我们提出了一种方法，能够在几秒钟内通过蒙特卡洛方法高效地近似多个动态障碍物之间的联合碰撞概率（CP），从而在多个采样轨迹中实时估算碰撞概率。这使得DRA-MPPI能够拒绝超出预定义CP阈值的样本，或将CP作为加权目标集成到导航成本函数中。因此，DRA-MPPI减轻了机器人冻结问题，增强了安全性。多动态障碍物的现实世界和模拟实验表明，DRA-MPPI在性能上优于现有最先进的方法，包括基于场景的模型预测控制（S-MPC）、Frenet规划器和传统的MPPI。 

---
# UAIbot: Beginner-friendly web-based simulator for interactive robotics learning and research 

**Title (ZH)**: UAIbot：面向初学者的基于Web的交互式机器人学习与研究模拟器 

**Authors**: Johnata Brayan, Armando Alves Neto, Pavel Petrovič, Gustavo M Freitas, Vinicius Mariano Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2506.21178)  

**Abstract**: This paper presents UAIbot, a free and open-source web-based robotics simulator designed to address the educational and research challenges conventional simulation platforms generally face. The Python and JavaScript interfaces of UAIbot enable accessible hands-on learning experiences without cumbersome installations. By allowing users to explore fundamental mathematical and physical principles interactively, ranging from manipulator kinematics to pedestrian flow dynamics, UAIbot provides an effective tool for deepening student understanding, facilitating rapid experimentation, and enhancing research dissemination. 

**Abstract (ZH)**: UAIbot：一种免费开源的基于Web的机器人模拟器，用于解决传统模拟平台普遍面临的教育教学和研究挑战 

---
# CURL-SLAM: Continuous and Compact LiDAR Mapping 

**Title (ZH)**: CURL-SLAM: 连续且紧凑的激光雷达建图 

**Authors**: Kaicheng Zhang, Shida Xu, Yining Ding, Xianwen Kong, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21077)  

**Abstract**: This paper studies 3D LiDAR mapping with a focus on developing an updatable and localizable map representation that enables continuity, compactness and consistency in 3D maps. Traditional LiDAR Simultaneous Localization and Mapping (SLAM) systems often rely on 3D point cloud maps, which typically require extensive storage to preserve structural details in large-scale environments. In this paper, we propose a novel paradigm for LiDAR SLAM by leveraging the Continuous and Ultra-compact Representation of LiDAR (CURL) introduced in [1]. Our proposed LiDAR mapping approach, CURL-SLAM, produces compact 3D maps capable of continuous reconstruction at variable densities using CURL's spherical harmonics implicit encoding, and achieves global map consistency after loop closure. Unlike popular Iterative Closest Point (ICP)-based LiDAR odometry techniques, CURL-SLAM formulates LiDAR pose estimation as a unique optimization problem tailored for CURL and extends it to local Bundle Adjustment (BA), enabling simultaneous pose refinement and map correction. Experimental results demonstrate that CURL-SLAM achieves state-of-the-art 3D mapping quality and competitive LiDAR trajectory accuracy, delivering sensor-rate real-time performance (10 Hz) on a CPU. We will release the CURL-SLAM implementation to the community. 

**Abstract (ZH)**: 基于CURL的可更新和定位的3D LiDAR映射研究 

---
# Control of Marine Robots in the Era of Data-Driven Intelligence 

**Title (ZH)**: 数据驱动智能时代的海洋机器人控制 

**Authors**: Lin Hong, Lu Liu, Zhouhua Peng, Fumin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21063)  

**Abstract**: The control of marine robots has long relied on model-based methods grounded in classical and modern control theory. However, the nonlinearity and uncertainties inherent in robot dynamics, coupled with the complexity of marine environments, have revealed the limitations of conventional control methods. The rapid evolution of machine learning has opened new avenues for incorporating data-driven intelligence into control strategies, prompting a paradigm shift in the control of marine robots. This paper provides a review of recent progress in marine robot control through the lens of this emerging paradigm. The review covers both individual and cooperative marine robotic systems, highlighting notable achievements in data-driven control of marine robots and summarizing open-source resources that support the development and validation of advanced control methods. Finally, several future perspectives are outlined to guide research toward achieving high-level autonomy for marine robots in real-world applications. This paper aims to serve as a roadmap toward the next-generation control framework of marine robots in the era of data-driven intelligence. 

**Abstract (ZH)**: 基于数据驱动智能的海洋机器人控制进展reviews the recent progress in marine robot control through the emerging paradigm of data-driven intelligence. 

---
# Knowledge-Driven Imitation Learning: Enabling Generalization Across Diverse Conditions 

**Title (ZH)**: 知识驱动的模仿学习：在多样条件下实现泛化 

**Authors**: Zhuochen Miao, Jun Lv, Hongjie Fang, Yang Jin, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21057)  

**Abstract**: Imitation learning has emerged as a powerful paradigm in robot manipulation, yet its generalization capability remains constrained by object-specific dependencies in limited expert demonstrations. To address this challenge, we propose knowledge-driven imitation learning, a framework that leverages external structural semantic knowledge to abstract object representations within the same category. We introduce a novel semantic keypoint graph as a knowledge template and develop a coarse-to-fine template-matching algorithm that optimizes both structural consistency and semantic similarity. Evaluated on three real-world robotic manipulation tasks, our method achieves superior performance, surpassing image-based diffusion policies with only one-quarter of the expert demonstrations. Extensive experiments further demonstrate its robustness across novel objects, backgrounds, and lighting conditions. This work pioneers a knowledge-driven approach to data-efficient robotic learning in real-world settings. Code and more materials are available on this https URL. 

**Abstract (ZH)**: 基于知识驱动的模仿学习：一种利用外部结构语义知识的框架 

---
# V2X-REALM: Vision-Language Model-Based Robust End-to-End Cooperative Autonomous Driving with Adaptive Long-Tail Modeling 

**Title (ZH)**: V2X-REALM：基于视觉-语言模型的鲁棒端到端协作自动驾驶及自适应长尾建模 

**Authors**: Junwei You, Pei Li, Zhuoyu Jiang, Zilin Huang, Rui Gan, Haotian Shi, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2506.21041)  

**Abstract**: Ensuring robust planning and decision-making under rare, diverse, and visually degraded long-tail scenarios remains a fundamental challenge for autonomous driving in urban environments. This issue becomes more critical in cooperative settings, where vehicles and infrastructure jointly perceive and reason across complex environments. To address this challenge, we propose V2X-REALM, a vision-language model (VLM)-based framework with adaptive multimodal learning for robust cooperative autonomous driving under long-tail scenarios. V2X-REALM introduces three core innovations: (i) a prompt-driven long-tail scenario generation and evaluation pipeline that leverages foundation models to synthesize realistic long-tail conditions such as snow and fog across vehicle- and infrastructure-side views, enriching training diversity efficiently; (ii) a gated multi-scenario adaptive attention module that modulates the visual stream using scenario priors to recalibrate ambiguous or corrupted features; and (iii) a multi-task scenario-aware contrastive learning objective that improves multimodal alignment and promotes cross-scenario feature separability. Extensive experiments demonstrate that V2X-REALM significantly outperforms existing baselines in robustness, semantic reasoning, safety, and planning accuracy under complex, challenging driving conditions, advancing the scalability of end-to-end cooperative autonomous driving. 

**Abstract (ZH)**: 确保在罕见、多样且视觉降级的长尾场景下实现鲁棒的规划与决策仍然是城市环境下自主驾驶中的一个基本挑战。在合作场景中，这一问题尤为关键，因为车辆和基础设施共同感知和推理复杂环境。为应对这一挑战，我们提出了一种基于视觉-语言模型（VLM）并具备自适应多模态学习的V2X-REALM框架，以实现鲁棒的合作自主驾驶。V2X-REALM引入了三项核心创新：（i）一种基于提示的长尾场景生成和评估管道，利用基础模型合成车辆和基础设施视角下的现实长尾条件，如雪和雾，高效丰富训练多样性；（ii）一种门控多场景自适应注意力模块，利用场景先验调节视觉流以校准含糊或损坏的特征；（iii）一种多任务场景感知对比学习目标，改进多模态对齐并促进跨场景特征可分性。广泛实验证明，V2X-REALM在鲁棒性、语义推理、安全性和规划准确性方面显著优于现有基准，推动了端到端合作自主驾驶的可扩展性。 

---
# STEP Planner: Constructing cross-hierarchical subgoal tree as an embodied long-horizon task planner 

**Title (ZH)**: STEP规划器：构建跨层次子目标树作为具身长时序任务规划器 

**Authors**: Zhou Tianxing, Wang Zhirui, Ao Haojia, Chen Guangyan, Xing Boyang, Cheng Jingwen, Yang Yi, Yue Yufeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21030)  

**Abstract**: The ability to perform reliable long-horizon task planning is crucial for deploying robots in real-world environments. However, directly employing Large Language Models (LLMs) as action sequence generators often results in low success rates due to their limited reasoning ability for long-horizon embodied tasks. In the STEP framework, we construct a subgoal tree through a pair of closed-loop models: a subgoal decomposition model and a leaf node termination model. Within this framework, we develop a hierarchical tree structure that spans from coarse to fine resolutions. The subgoal decomposition model leverages a foundation LLM to break down complex goals into manageable subgoals, thereby spanning the subgoal tree. The leaf node termination model provides real-time feedback based on environmental states, determining when to terminate the tree spanning and ensuring each leaf node can be directly converted into a primitive action. Experiments conducted in both the VirtualHome WAH-NL benchmark and on real robots demonstrate that STEP achieves long-horizon embodied task completion with success rates up to 34% (WAH-NL) and 25% (real robot) outperforming SOTA methods. 

**Abstract (ZH)**: 具备可靠长时间任务规划能力对于将机器人部署到实际环境至关重要。在STEP框架中，我们通过一对闭环模型——子目标分解模型和叶节点终止模型，构建子目标树，并开发了一种从粗到细的分层树结构。子目标分解模型利用基础大模型将复杂目标分解为可管理的子目标，从而构建子目标树。叶节点终止模型根据环境状态提供实时反馈，确定停止扩展树的时间，并确保每个叶节点可以直接转换为基本动作。在VirtualHome WAH-NL基准和真实机器人上的实验表明，STEP在WAH-NL的完成成功率高达34%，在真实机器人上的完成成功率高达25%，超过当前最先进的方法。 

---
# Fault-Tolerant Spacecraft Attitude Determination using State Estimation Techniques 

**Title (ZH)**: 基于状态估计技术的容错航天器姿态确定方法 

**Authors**: B. Chidambaram, A. Hilbert, M. Silva  

**Link**: [PDF](https://arxiv.org/pdf/2506.21016)  

**Abstract**: The extended and unscented Kalman filter, and the particle filter provide a robust framework for fault-tolerant attitude estimation on spacecraft. This paper explores how each filter performs for a large satellite in a low earth orbit. Additionally, various techniques, built on these filters, for fault detection, isolation and recovery from erroneous sensor measurements, are analyzed. Key results from this analysis include filter performance for various fault modes. 

**Abstract (ZH)**: 扩展的和无迹Kal曼滤波器以及粒子滤波器为太空卫星在低地球轨道上的容错姿态估计提供了稳健的框架。本文探讨了这些滤波器在大型卫星上的性能，并分析了基于这些滤波器的各种故障检测、隔离和恢复技术。关键结果包括不同故障模式下的滤波器性能。 

---
# ThermalDiffusion: Visual-to-Thermal Image-to-Image Translation for Autonomous Navigation 

**Title (ZH)**: 热扩散：视觉到红外的图像到图像翻译在自主导航中的应用 

**Authors**: Shruti Bansal, Wenshan Wang, Yifei Liu, Parv Maheshwari  

**Link**: [PDF](https://arxiv.org/pdf/2506.20969)  

**Abstract**: Autonomous systems rely on sensors to estimate the environment around them. However, cameras, LiDARs, and RADARs have their own limitations. In nighttime or degraded environments such as fog, mist, or dust, thermal cameras can provide valuable information regarding the presence of objects of interest due to their heat signature. They make it easy to identify humans and vehicles that are usually at higher temperatures compared to their surroundings. In this paper, we focus on the adaptation of thermal cameras for robotics and automation, where the biggest hurdle is the lack of data. Several multi-modal datasets are available for driving robotics research in tasks such as scene segmentation, object detection, and depth estimation, which are the cornerstone of autonomous systems. However, they are found to be lacking in thermal imagery. Our paper proposes a solution to augment these datasets with synthetic thermal data to enable widespread and rapid adaptation of thermal cameras. We explore the use of conditional diffusion models to convert existing RGB images to thermal images using self-attention to learn the thermal properties of real-world objects. 

**Abstract (ZH)**: 自主系统依赖传感器来估计其周围的环境。然而，摄像头、LiDAR和RADAR各自都有局限性。在夜间或能见度降低的环境中，如雾、霭或尘埃，热成像相机由于其热信号可以提供有关目标物体存在的有价值的信息。它们使得识别通常比周围环境温度更高的行人和车辆变得容易。本文专注于将热成像相机应用于机器人技术和自动化领域，最大的障碍是缺乏数据。有多模态数据集可供驾驶机器人研究使用，这些数据集在场景分割、物体检测和深度估计等任务中至关重要，是自主系统的基础。然而，这些数据集缺乏热图像。本文提出了一种解决方案，通过加入合成热数据来增强这些数据集，以便广泛快速地适应热成像相机。我们探讨了使用条件扩散模型将现有的RGB图像转换为热图像的方法，并利用自注意力机制学习真实世界物体的热特性。 

---
# Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends 

**Title (ZH)**: VLA模型后训练与人类运动学习之间的parallel进展、挑战与趋势 

**Authors**: Tian-Yu Xiang, Ao-Qun Jin, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, Shuang-Yi Wang, Sheng-Bin Duan, Fu-Chao Xie, Wen-Kai Wang, Si-Cheng Wang, Ling-Yun Li, Tian Tu, Zeng-Guang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20966)  

**Abstract**: Vision-language-action (VLA) models extend vision-language models (VLM) by integrating action generation modules for robotic manipulation. Leveraging strengths of VLM in vision perception and instruction understanding, VLA models exhibit promising generalization across diverse manipulation tasks. However, applications demanding high precision and accuracy reveal performance gaps without further adaptation. Evidence from multiple domains highlights the critical role of post-training to align foundational models with downstream applications, spurring extensive research on post-training VLA models. VLA model post-training aims to address the challenge of improving an embodiment's ability to interact with the environment for the given tasks, analogous to the process of humans motor skills acquisition. Accordingly, this paper reviews post-training strategies for VLA models through the lens of human motor learning, focusing on three dimensions: environments, embodiments, and tasks. A structured taxonomy is introduced aligned with human learning mechanisms: (1) enhancing environmental perception, (2) improving embodiment awareness, (3) deepening task comprehension, and (4) multi-component integration. Finally, key challenges and trends in post-training VLA models are identified, establishing a conceptual framework to guide future research. This work delivers both a comprehensive overview of current VLA model post-training methods from a human motor learning perspective and practical insights for VLA model development. (Project website: this https URL) 

**Abstract (ZH)**: Vision-语言-动作（VLA）模型通过集成动作生成模块，扩展了视觉-语言模型（VLM）的功能，以实现机器人操作。借助VLM在视觉感知和指令理解方面的优势，VLA模型在多种操作任务中展现了良好的泛化能力。然而，对高精度和高精度要求的应用显示出性能差距，需要进一步适应。来自多个领域的证据强调了后训练在调整基础模型以适应下游应用中的关键作用，推动了对后训练VLA模型的大量研究。VLA模型后训练旨在通过类比人类运动技能获取过程，解决提升实体与环境交互能力的挑战。据此，本文从人类运动学习的角度回顾了VLA模型的后训练策略，重点关注三个维度：环境、实体和任务。引入了一种与人类学习机制对齐的结构化分类体系：（1）增强环境感知，（2）提高实体认知，（3）深化任务理解，（4）多组件集成。最后，本文指出了后训练VLA模型的关键挑战和发展趋势，为未来研究提供了一个概念框架。本文不仅从人类运动学习的角度提供了当前VLA模型后训练方法的全面综述，也为VLA模型开发提供了实用见解。（项目网站：this https URL） 

---
# Cooperative Circumnavigation for Multi-Quadrotor Systems via Onboard Sensing 

**Title (ZH)**: 多旋翼系统基于机载传感的协同巡飞 

**Authors**: Xueming Liu, Lin Li, Xiang Zhou, Qingrui Zhang, Tianjiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20954)  

**Abstract**: A cooperative circumnavigation framework is proposed for multi-quadrotor systems to enclose and track a moving target without reliance on external localization systems. The distinct relationships between quadrotor-quadrotor and quadrotor-target interactions are evaluated using a heterogeneous perception strategy and corresponding state estimation algorithms. A modified Kalman filter is developed to fuse visual-inertial odometry with range measurements to enhance the accuracy of inter-quadrotor relative localization. An event-triggered distributed Kalman filter is designed to achieve robust target state estimation under visual occlusion by incorporating neighbor measurements and estimated inter-quadrotor relative positions. Using the estimation results, a cooperative circumnavigation controller is constructed, leveraging an oscillator-based autonomous formation flight strategy. We conduct extensive indoor and outdoor experiments to validate the efficiency of the proposed circumnavigation framework in occluded environments. Furthermore, a quadrotor failure experiment highlights the inherent fault tolerance property of the proposed framework, underscoring its potential for deployment in search-and-rescue operations. 

**Abstract (ZH)**: 一种多旋翼系统协作环航框架：在无需外部定位系统的情况下包围和跟踪移动目标，并评估旋翼机-旋翼机和旋翼机-目标之间的异构感知关系和相应状态估计算法，开发改进的卡尔曼滤波器融合视觉-惯性里程计与距离测量以提高旋翼机间相对定位的精度，设计事件触发的分布式卡尔曼滤波器以在视觉遮挡下实现鲁棒的目标状态估计，利用振荡器为基础的自主编队飞行策略构建协作环航控制器。通过广泛的室内和室外实验验证所提出环航框架在遮挡环境中的效率，并通过旋翼机故障实验突出所提出框架的固有容错特性，强调其在搜索与救援操作中的潜力。 

---
# Model-Based Real-Time Pose and Sag Estimation of Overhead Power Lines Using LiDAR for Drone Inspection 

**Title (ZH)**: 基于模型的无人机巡检中基于LiDAR的架空输电线路实时姿态和弛度估计 

**Authors**: Alexandre Girard, Steven A. Parkison, Philippe Hamelin  

**Link**: [PDF](https://arxiv.org/pdf/2506.20812)  

**Abstract**: Drones can inspect overhead power lines while they remain energized, significantly simplifying the inspection process. However, localizing a drone relative to all conductors using an onboard LiDAR sensor presents several challenges: (1) conductors provide minimal surface for LiDAR beams limiting the number of conductor points in a scan, (2) not all conductors are consistently detected, and (3) distinguishing LiDAR points corresponding to conductors from other objects, such as trees and pylons, is difficult. This paper proposes an estimation approach that minimizes the error between LiDAR measurements and a single geometric model representing the entire conductor array, rather than tracking individual conductors separately. Experimental results, using data from a power line drone inspection, demonstrate that this method achieves accurate tracking, with a solver converging under 50 ms per frame, even in the presence of partial observations, noise, and outliers. A sensitivity analysis shows that the estimation approach can tolerate up to twice as many outlier points as valid conductors measurements. 

**Abstract (ZH)**: 无人机可以带电检测架空输电线路，显著简化检测过程。然而，使用机载LiDAR传感器相对于所有导线进行定位存在多项挑战：导线对LiDAR光束提供的表面极少，限制了扫描中的导线点数量；并非所有导线都能一致检测；区分与导线对应的LiDAR点与其他物体，如树木和电塔，是困难的。本文提出了一种估计方法，力求最小化LiDAR测量值与表示整个导线阵列的单一几何模型之间的误差，而不是单独追踪每根导线。实验结果，使用输电线路无人机检测数据，显示该方法即使在部分观测、噪声和离群点存在的条件下也能实现准确跟踪，求解器在每帧50毫秒内收敛。敏感性分析表明，该估计方法可以容忍的离群点数量是有效导线测量数量的两倍。 

---
# Online Planning for Cooperative Air-Ground Robot Systems with Unknown Fuel Requirements 

**Title (ZH)**: 具有未知燃料需求的协同空地机器人系统在线规划 

**Authors**: Ritvik Agarwal, Behnoushsadat Hatami, Alvika Gautam, Parikshit Maini  

**Link**: [PDF](https://arxiv.org/pdf/2506.20804)  

**Abstract**: We consider an online variant of the fuel-constrained UAV routing problem with a ground-based mobile refueling station (FCURP-MRS), where targets incur unknown fuel costs. We develop a two-phase solution: an offline heuristic-based planner computes initial UAV and UGV paths, and a novel online planning algorithm that dynamically adjusts rendezvous points based on real-time fuel consumption during target processing. Preliminary Gazebo simulations demonstrate the feasibility of our approach in maintaining UAV-UGV path validity, ensuring mission completion. Link to video: this https URL 

**Abstract (ZH)**: 基于地面移动加油站的在线燃料约束无人机路由问题（FCURP-MRS）：一种两阶段解决方案 

---
# IMA-Catcher: An IMpact-Aware Nonprehensile Catching Framework based on Combined Optimization and Learning 

**Title (ZH)**: IMA-Catcher：一种基于联合优化与学习的感知作用非抓取捕捉框架 

**Authors**: Francesco Tassi, Jianzhuang Zhao, Gustavo J. G. Lahr, Luna Gava, Marco Monforte, Arren Glover, Chiara Bartolozzi, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2506.20801)  

**Abstract**: Robotic catching of flying objects typically generates high impact forces that might lead to task failure and potential hardware damages. This is accentuated when the object mass to robot payload ratio increases, given the strong inertial components characterizing this task. This paper aims to address this problem by proposing an implicitly impact-aware framework that accomplishes the catching task in both pre- and post-catching phases. In the first phase, a motion planner generates optimal trajectories that minimize catching forces, while in the second, the object's energy is dissipated smoothly, minimizing bouncing. In particular, in the pre-catching phase, a real-time optimal planner is responsible for generating trajectories of the end-effector that minimize the velocity difference between the robot and the object to reduce impact forces during catching. In the post-catching phase, the robot's position, velocity, and stiffness trajectories are generated based on human demonstrations when catching a series of free-falling objects with unknown masses. A hierarchical quadratic programming-based controller is used to enforce the robot's constraints (i.e., joint and torque limits) and create a stack of tasks that minimizes the reflected mass at the end-effector as a secondary objective. The initial experiments isolate the problem along one dimension to accurately study the effects of each contribution on the metrics proposed. We show how the same task, without velocity matching, would be infeasible due to excessive joint torques resulting from the impact. The addition of reflected mass minimization is then investigated, and the catching height is increased to evaluate the method's robustness. Finally, the setup is extended to catching along multiple Cartesian axes, to prove its generalization in space. 

**Abstract (ZH)**: 机器人捕获飞行物体通常会产生高冲击力，可能导致任务失败和潜在的硬件损坏。当物体质量与机器人负载比例增加时，这一问题更加明显，因为该任务具有强烈的惯性特征。本文提出了一种隐式冲击感知框架，该框架在捕获前、后阶段均能够完成捕获任务。在捕获前阶段，运动规划器生成可最小化捕获力的最优轨迹；在捕获后阶段，物体的能量被平滑消散，以最小化反弹。具体而言，在捕获前阶段，实时最优规划器负责生成末端执行器的轨迹，以最小化机器人与物体之间的速度差异，从而减少捕获过程中的冲击力。在捕获后阶段，基于人类示范，机器人的位置、速度和刚度轨迹根据一系列未知质量的自由落体物体的捕获生成。基于分层二次规划的控制器被用于强制执行机器人的约束（即关节和扭矩限制），并创建一个任务堆栈，将末端执行器上反射质量最小化作为次要目标。初始实验将问题沿着一个维度隔离，以准确研究每种贡献对所提指标的影响。我们展示了在没有速度匹配的情况下，同样的任务由于冲击导致的关节扭矩过度而变得不可行。随后研究了引入反射质量最小化的有效性，并提高捕获高度以评估该方法的鲁棒性。最后，该设置扩展到多个笛卡尔轴上的捕获，以证明其在空间上的通用性。 

---
# Whole-Body Conditioned Egocentric Video Prediction 

**Title (ZH)**: 全身条件下的自我中心视频预测 

**Authors**: Yutong Bai, Danny Tran, Amir Bar, Yann LeCun, Trevor Darrell, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2506.21552)  

**Abstract**: We train models to Predict Ego-centric Video from human Actions (PEVA), given the past video and an action represented by the relative 3D body pose. By conditioning on kinematic pose trajectories, structured by the joint hierarchy of the body, our model learns to simulate how physical human actions shape the environment from a first-person point of view. We train an auto-regressive conditional diffusion transformer on Nymeria, a large-scale dataset of real-world egocentric video and body pose capture. We further design a hierarchical evaluation protocol with increasingly challenging tasks, enabling a comprehensive analysis of the model's embodied prediction and control abilities. Our work represents an initial attempt to tackle the challenges of modeling complex real-world environments and embodied agent behaviors with video prediction from the perspective of a human. 

**Abstract (ZH)**: 基于人体动作预测视角中心视频（PEVA）：给定过去视频和由相对3D身体姿态表示的动作进行预测 

---
# SAM4D: Segment Anything in Camera and LiDAR Streams 

**Title (ZH)**: SAM4D: 在相机和LiDAR流中进行分割 

**Authors**: Jianyun Xu, Song Wang, Ziqian Ni, Chunyong Hu, Sheng Yang, Jianke Zhu, Qiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21547)  

**Abstract**: We present SAM4D, a multi-modal and temporal foundation model designed for promptable segmentation across camera and LiDAR streams. Unified Multi-modal Positional Encoding (UMPE) is introduced to align camera and LiDAR features in a shared 3D space, enabling seamless cross-modal prompting and interaction. Additionally, we propose Motion-aware Cross-modal Memory Attention (MCMA), which leverages ego-motion compensation to enhance temporal consistency and long-horizon feature retrieval, ensuring robust segmentation across dynamically changing autonomous driving scenes. To avoid annotation bottlenecks, we develop a multi-modal automated data engine that synergizes VFM-driven video masklets, spatiotemporal 4D reconstruction, and cross-modal masklet fusion. This framework generates camera-LiDAR aligned pseudo-labels at a speed orders of magnitude faster than human annotation while preserving VFM-derived semantic fidelity in point cloud representations. We conduct extensive experiments on the constructed Waymo-4DSeg, which demonstrate the powerful cross-modal segmentation ability and great potential in data annotation of proposed SAM4D. 

**Abstract (ZH)**: SAM4D：一种面向相机和LiDAR流的可提示语义分割多模态和时间基础模型 

---
# Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning 

**Title (ZH)**: 基于流的单步完成方法以实现高效且表达能力强的策略学习 

**Authors**: Prajwal Koirala, Cody Fleming  

**Link**: [PDF](https://arxiv.org/pdf/2506.21427)  

**Abstract**: Generative models such as diffusion and flow-matching offer expressive policies for offline reinforcement learning (RL) by capturing rich, multimodal action distributions, but their iterative sampling introduces high inference costs and training instability due to gradient propagation across sampling steps. We propose the \textit{Single-Step Completion Policy} (SSCP), a generative policy trained with an augmented flow-matching objective to predict direct completion vectors from intermediate flow samples, enabling accurate, one-shot action generation. In an off-policy actor-critic framework, SSCP combines the expressiveness of generative models with the training and inference efficiency of unimodal policies, without requiring long backpropagation chains. Our method scales effectively to offline, offline-to-online, and online RL settings, offering substantial gains in speed and adaptability over diffusion-based baselines. We further extend SSCP to goal-conditioned RL, enabling flat policies to exploit subgoal structures without explicit hierarchical inference. SSCP achieves strong results across standard offline RL and behavior cloning benchmarks, positioning it as a versatile, expressive, and efficient framework for deep RL and sequential decision-making. 

**Abstract (ZH)**: 单步完成策略（SSCP）：一种用于离线强化学习的生成政策 

---
# EndoFlow-SLAM: Real-Time Endoscopic SLAM with Flow-Constrained Gaussian Splatting 

**Title (ZH)**: EndoFlow-SLAM：基于流约束高斯点云的实时内窥镜SLAM 

**Authors**: Taoyu Wu, Yiyi Miao, Zhuoxiao Li, Haocheng Zhao, Kang Dang, Jionglong Su, Limin Yu, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.21420)  

**Abstract**: Efficient three-dimensional reconstruction and real-time visualization are critical in surgical scenarios such as endoscopy. In recent years, 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in efficient 3D reconstruction and rendering. Most 3DGS-based Simultaneous Localization and Mapping (SLAM) methods only rely on the appearance constraints for optimizing both 3DGS and camera poses. However, in endoscopic scenarios, the challenges include photometric inconsistencies caused by non-Lambertian surfaces and dynamic motion from breathing affects the performance of SLAM systems. To address these issues, we additionally introduce optical flow loss as a geometric constraint, which effectively constrains both the 3D structure of the scene and the camera motion. Furthermore, we propose a depth regularisation strategy to mitigate the problem of photometric inconsistencies and ensure the validity of 3DGS depth rendering in endoscopic scenes. In addition, to improve scene representation in the SLAM system, we improve the 3DGS refinement strategy by focusing on viewpoints corresponding to Keyframes with suboptimal rendering quality frames, achieving better rendering results. Extensive experiments on the C3VD static dataset and the StereoMIS dynamic dataset demonstrate that our method outperforms existing state-of-the-art methods in novel view synthesis and pose estimation, exhibiting high performance in both static and dynamic surgical scenes. The source code will be publicly available upon paper acceptance. 

**Abstract (ZH)**: 高效三维重建和实时可视化在内窥镜等外科场景中至关重要。近年来，3D高斯点积（3DGS）在高效三维重建与渲染方面展现了显著性能。大多数基于3DGS的SLAM方法仅依赖外观约束来优化3DGS和相机姿态。然而，在内窥镜场景中，非朗伯表面导致的光度不一致性及呼吸引起的动态运动影响了SLAM系统的性能。为解决这些问题，我们还引入了光流损失作为几何约束，有效限制了场景的3D结构和相机运动。此外，我们提出了深度正则化策略，以减轻光度不一致性问题，并确保3DGS在内窥镜场景中的深度渲染有效性。为了改进SLAM系统中的场景表示，我们通过关注渲染质量不佳的关键帧对应的视角来改进3DGS细化策略，从而获得更好的渲染结果。在C3VD静态数据集和StereoMIS动态数据集上的广泛实验显示，我们的方法在新颖视图合成和姿态估计方面优于现有最先进的方法，在静态和动态外科场景中表现出高性能。论文被接受后，源代码将公开。 

---
# ToosiCubix: Monocular 3D Cuboid Labeling via Vehicle Part Annotations 

**Title (ZH)**: ToosiCubix: 通过车辆部件标注的单目3D立方体标定 

**Authors**: Behrooz Nasihatkon, Hossein Resani, Amirreza Mehrzadian  

**Link**: [PDF](https://arxiv.org/pdf/2506.21358)  

**Abstract**: Many existing methods for 3D cuboid annotation of vehicles rely on expensive and carefully calibrated camera-LiDAR or stereo setups, limiting their accessibility for large-scale data collection. We introduce ToosiCubix, a simple yet powerful approach for annotating ground-truth cuboids using only monocular images and intrinsic camera parameters. Our method requires only about 10 user clicks per vehicle, making it highly practical for adding 3D annotations to existing datasets originally collected without specialized equipment. By annotating specific features (e.g., wheels, car badge, symmetries) across different vehicle parts, we accurately estimate each vehicle's position, orientation, and dimensions up to a scale ambiguity (8 DoF). The geometric constraints are formulated as an optimization problem, which we solve using a coordinate descent strategy, alternating between Perspective-n-Points (PnP) and least-squares subproblems. To handle common ambiguities such as scale and unobserved dimensions, we incorporate probabilistic size priors, enabling 9 DoF cuboid placements. We validate our annotations against the KITTI and Cityscapes3D datasets, demonstrating that our method offers a cost-effective and scalable solution for high-quality 3D cuboid annotation. 

**Abstract (ZH)**: 一种仅使用单目图像和内窥镜相机参数进行车辆3D立方体标注的简单有效方法：ToosiCubix 

---
# "Who Should I Believe?": User Interpretation and Decision-Making When a Family Healthcare Robot Contradicts Human Memory 

**Title (ZH)**: “我该相信谁？”：当家庭医疗服务机器人与人类记忆冲突时，用户如何解读和决策 

**Authors**: Hong Wang, Natalia Calvo-Barajas, Katie Winkle, Ginevra Castellano  

**Link**: [PDF](https://arxiv.org/pdf/2506.21322)  

**Abstract**: Advancements in robotic capabilities for providing physical assistance, psychological support, and daily health management are making the deployment of intelligent healthcare robots in home environments increasingly feasible in the near future. However, challenges arise when the information provided by these robots contradicts users' memory, raising concerns about user trust and decision-making. This paper presents a study that examines how varying a robot's level of transparency and sociability influences user interpretation, decision-making and perceived trust when faced with conflicting information from a robot. In a 2 x 2 between-subjects online study, 176 participants watched videos of a Furhat robot acting as a family healthcare assistant and suggesting a fictional user to take medication at a different time from that remembered by the user. Results indicate that robot transparency influenced users' interpretation of information discrepancies: with a low transparency robot, the most frequent assumption was that the user had not correctly remembered the time, while with the high transparency robot, participants were more likely to attribute the discrepancy to external factors, such as a partner or another household member modifying the robot's information. Additionally, participants exhibited a tendency toward overtrust, often prioritizing the robot's recommendations over the user's memory, even when suspecting system malfunctions or third-party interference. These findings highlight the impact of transparency mechanisms in robotic systems, the complexity and importance associated with system access control for multi-user robots deployed in home environments, and the potential risks of users' over reliance on robots in sensitive domains such as healthcare. 

**Abstract (ZH)**: 家用环境中智能医疗机器人的发展使其日益可行，但当机器人提供的信息与用户记忆相矛盾时，用户信任和决策制定方面的问题也随之出现。本文研究了机器人透明度和社交性水平变化对用户在面对机器人矛盾信息时解释、决策和感知信任的影响。在一项包含两个变量的双向在线研究中，共有176名参与者观看了Furhat机器人作为家庭健康助手，建议虚构用户在不同于用户记忆的时间服用药物的视频。研究结果表明，机器人的透明度影响了用户对信息差异的理解：在透明度较低的机器人中，用户最常假设自己没有正确记住时间；而在透明度较高的机器人中，参与者更倾向于将差异归因于外部因素，如伴侣或其他家庭成员修改了机器人的信息。此外，参与者倾向于过度信任机器人，经常优先考虑机器人的建议而非自己的记忆，即使怀疑系统故障或第三方干扰。这些发现强调了机器人系统中透明机制的影响，多用户家庭环境中部署的系统访问控制的复杂性和重要性，以及用户在医疗等敏感领域过度依赖机器人可能带来的风险。 

---
# Real-Time ESFP: Estimating, Smoothing, Filtering, and Pose-Mapping 

**Title (ZH)**: 实时ESFP：估计、平滑、滤波与姿态映射 

**Authors**: Qifei Cui, Yuang Zhou, Ruichen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21234)  

**Abstract**: This paper presents ESFP, an end-to-end pipeline that converts monocular RGB video into executable joint trajectories for a low-cost 4-DoF desktop arm. ESFP comprises four sequential modules. (1) Estimating: ROMP lifts each frame to a 24-joint 3-D skeleton. (2) Smoothing: the proposed HPSTM-a sequence-to-sequence Transformer with self-attention-combines long-range temporal context with a differentiable forward-kinematics decoder, enforcing constant bone lengths and anatomical plausibility while jointly predicting joint means and full covariances. (3) Filtering: root-normalized trajectories are variance-weighted according to HPSTM's uncertainty estimates, suppressing residual noise. (4) Pose-Mapping: a geometric retargeting layer transforms shoulder-elbow-wrist triples into the uArm's polar workspace, preserving wrist orientation. 

**Abstract (ZH)**: ESFP：一种从单目RGB视频端到端生成低成本4-DOF桌面臂可执行关节轨迹的流水线 

---
# World-aware Planning Narratives Enhance Large Vision-Language Model Planner 

**Title (ZH)**: 具有世界意识的规划叙事增强大型视觉语言模型规划者 

**Authors**: Junhao Shi, Zhaoye Fei, Siyin Wang, Qipeng Guo, Jingjing Gong, Xipeng QIu  

**Link**: [PDF](https://arxiv.org/pdf/2506.21230)  

**Abstract**: Large Vision-Language Models (LVLMs) show promise for embodied planning tasks but struggle with complex scenarios involving unfamiliar environments and multi-step goals. Current approaches rely on environment-agnostic imitation learning that disconnects instructions from environmental contexts, causing models to struggle with context-sensitive instructions and rely on supplementary cues rather than visual reasoning during long-horizon interactions. In this work, we propose World-Aware Planning Narrative Enhancement (WAP), a framework that infuses LVLMs with comprehensive environmental understanding through four cognitive capabilities (visual appearance modeling, spatial reasoning, functional abstraction, and syntactic grounding) while developing and evaluating models using only raw visual observations through curriculum learning. Evaluations on the EB-ALFRED benchmark demonstrate substantial improvements, with Qwen2.5-VL achieving a 60.7 absolute improvement in task success rates, particularly in commonsense reasoning (+60.0) and long-horizon planning (+70.0). Notably, our enhanced open-source models outperform proprietary systems like GPT-4o and Claude-3.5-Sonnet by a large margin. 

**Abstract (ZH)**: 具有环境意识的规划叙事增强（WAP）框架通过全面的环境理解提升大型多模态模型在复杂场景和多步目标下的体模规划能力 

---
# Unlocking Constraints: Source-Free Occlusion-Aware Seamless Segmentation 

**Title (ZH)**: 解锁约束：无源域aware遮挡自洽分割 

**Authors**: Yihong Cao, Jiaming Zhang, Xu Zheng, Hao Shi, Kunyu Peng, Hang Liu, Kailun Yang, Hui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21198)  

**Abstract**: Panoramic image processing is essential for omni-context perception, yet faces constraints like distortions, perspective occlusions, and limited annotations. Previous unsupervised domain adaptation methods transfer knowledge from labeled pinhole data to unlabeled panoramic images, but they require access to source pinhole data. To address these, we introduce a more practical task, i.e., Source-Free Occlusion-Aware Seamless Segmentation (SFOASS), and propose its first solution, called UNconstrained Learning Omni-Context Knowledge (UNLOCK). Specifically, UNLOCK includes two key modules: Omni Pseudo-Labeling Learning and Amodal-Driven Context Learning. While adapting without relying on source data or target labels, this framework enhances models to achieve segmentation with 360° viewpoint coverage and occlusion-aware reasoning. Furthermore, we benchmark the proposed SFOASS task through both real-to-real and synthetic-to-real adaptation settings. Experimental results show that our source-free method achieves performance comparable to source-dependent methods, yielding state-of-the-art scores of 10.9 in mAAP and 11.6 in mAP, along with an absolute improvement of +4.3 in mAPQ over the source-only method. All data and code will be made publicly available at this https URL. 

**Abstract (ZH)**: 无源域适应全景图像无遮挡感知无缝分割 

---
# Out-of-Distribution Semantic Occupancy Prediction 

**Title (ZH)**: 分布外语义占用预测 

**Authors**: Yuheng Zhang, Mengfei Duan, Kunyu Peng, Yuhang Wang, Ruiping Liu, Fei Teng, Kai Luo, Zhiyong Li, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21185)  

**Abstract**: 3D Semantic Occupancy Prediction is crucial for autonomous driving, providing a dense, semantically rich environmental representation. However, existing methods focus on in-distribution scenes, making them susceptible to Out-of-Distribution (OoD) objects and long-tail distributions, which increases the risk of undetected anomalies and misinterpretations, posing safety hazards. To address these challenges, we introduce Out-of-Distribution Semantic Occupancy Prediction, targeting OoD detection in 3D voxel space. To fill the gaps in the dataset, we propose a Synthetic Anomaly Integration Pipeline that injects synthetic anomalies while preserving realistic spatial and occlusion patterns, enabling the creation of two datasets: VAA-KITTI and VAA-KITTI-360. We introduce OccOoD, a novel framework integrating OoD detection into 3D semantic occupancy prediction, with Voxel-BEV Progressive Fusion (VBPF) leveraging an RWKV-based branch to enhance OoD detection via geometry-semantic fusion. Experimental results demonstrate that OccOoD achieves state-of-the-art OoD detection with an AuROC of 67.34% and an AuPRCr of 29.21% within a 1.2m region, while maintaining competitive occupancy prediction performance. The established datasets and source code will be made publicly available at this https URL. 

**Abstract (ZH)**: 3D语义占用度外分布检测对于自主驾驶至关重要，它提供了密集且语义丰富的环境表示。然而，现有方法主要关注于分布内的场景，这使得它们容易受到外分布（OoD）对象和长尾分布的影响，增加了未检测异常和误解释的风险，从而带来安全隐患。为解决这些问题，我们提出了外分布语义占用度预测，针对3D体素空间中的OoD检测。为了填补数据集的空白，我们提出了一种合成异常集成管道，该管道在保留现实空间和遮挡模式的同时注入合成异常，从而创建了两个数据集：VAA-KITTI和VAA-KITTI-360。我们提出了OccOoD，这是一种新的框架，将OoD检测集成到3D语义占用度预测中，其中体素-鸟瞰图渐进融合（VBPF）通过几何-语义融合利用RWKV基分支来增强OoD检测。实验结果表明，OccOoD在1.2米区域内的AuROC为67.34%，AuPRCr为29.21%，实现了最先进的OoD检测性能，同时保持了竞争力的占用度预测性能。已建立的数据集和源代码将在此处公开。 

---
# GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction 

**Title (ZH)**: GoIRL: 面向图的逆强化学习 multimodal 轨迹预测 

**Authors**: Muleilan Pei, Shaoshuai Shi, Lu Zhang, Peiliang Li, Shaojie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.21121)  

**Abstract**: Trajectory prediction for surrounding agents is a challenging task in autonomous driving due to its inherent uncertainty and underlying multimodality. Unlike prevailing data-driven methods that primarily rely on supervised learning, in this paper, we introduce a novel Graph-oriented Inverse Reinforcement Learning (GoIRL) framework, which is an IRL-based predictor equipped with vectorized context representations. We develop a feature adaptor to effectively aggregate lane-graph features into grid space, enabling seamless integration with the maximum entropy IRL paradigm to infer the reward distribution and obtain the policy that can be sampled to induce multiple plausible plans. Furthermore, conditioned on the sampled plans, we implement a hierarchical parameterized trajectory generator with a refinement module to enhance prediction accuracy and a probability fusion strategy to boost prediction confidence. Extensive experimental results showcase our approach not only achieves state-of-the-art performance on the large-scale Argoverse & nuScenes motion forecasting benchmarks but also exhibits superior generalization abilities compared to existing supervised models. 

**Abstract (ZH)**: 基于图的逆强化学习（GoIRL）框架在自动驾驶中的路径预测 

---
# Our Coding Adventure: Using LLMs to Personalise the Narrative of a Tangible Programming Robot for Preschoolers 

**Title (ZH)**: 我们的编码冒险：使用大规模语言模型为学龄前儿童个性化物理编程机器人的叙事 

**Authors**: Martin Ruskov  

**Link**: [PDF](https://arxiv.org/pdf/2506.20982)  

**Abstract**: Finding balanced ways to employ Large Language Models (LLMs) in education is a challenge due to inherent risks of poor understanding of the technology and of a susceptible audience. This is particularly so with younger children, who are known to have difficulties with pervasive screen time. Working with a tangible programming robot called Cubetto, we propose an approach to benefit from the capabilities of LLMs by employing such models in the preparation of personalised storytelling, necessary for preschool children to get accustomed to the practice of commanding the robot. We engage in action research to develop an early version of a formalised process to rapidly prototype game stories for Cubetto. Our approach has both reproducible results, because it employs open weight models, and is model-agnostic, because we test it with 5 different LLMs. We document on one hand the process, the used materials and prompts, and on the other the learning experience and outcomes. We deem the generation successful for the intended purposes of using the results as a teacher aid. Testing the models on 4 different task scenarios, we encounter issues of consistency and hallucinations and document the corresponding evaluation process and attempts (some successful and some not) to overcome these issues. Importantly, the process does not expose children to LLMs directly. Rather, the technology is used to help teachers easily develop personalised narratives on children's preferred topics. We believe our method is adequate for preschool classes and we are planning to further experiment in real-world educational settings. 

**Abstract (ZH)**: 在教育中寻� Reeves 大型语言模型 (LLMs) 的平衡应用：鉴于技术理解不足和易受影响的受众固有风险，特别是在年轻儿童中，他们在广泛使用屏幕时间方面存在问题。通过使用名为 Cubetto 的实体编程机器人，我们提出了一种方法，即利用 LLMs 的能力来准备个性化的故事情节，这是幼儿熟悉控制机器人实践所必需的。我们开展行动研究以开发 Cubetto 游戏故事快速原型设计的早期正式化流程。我们的方法既具有可重复性，因为使用的是开放权重模型，又具有模型无关性，因为使用了 5 种不同 LLM 进行测试。我们一方面记录流程、使用的材料和提示，另一方面记录学习体验和成果。我们认为生成的内容适合作为教师辅助材料。在四个不同任务场景中测试模型时，我们遇到了一致性问题和幻觉问题，并记录了相应的评估过程和克服这些问题的尝试（部分成功，部分不成功）。重要的是，该过程不会直接将儿童暴露于 LLMs，而是利用技术帮助教师轻松创建符合儿童偏好主题的个性化叙事。我们认为我们的方法适合幼儿园班级，并计划在实际教育环境中进一步试验。 

---
# Effect of Haptic Feedback on Avoidance Behavior and Visual Exploration in Dynamic VR Pedestrian Environment 

**Title (ZH)**: 动态VR行人环境中触觉反馈对回避行为和视觉探索的影响 

**Authors**: Kyosuke Ishibashi, Atsushi Saito, Zin Y. Tun, Lucas Ray, Megan C. Coram, Akihiro Sakurai, Allison M. Okamura, Ko Yamamoto  

**Link**: [PDF](https://arxiv.org/pdf/2506.20952)  

**Abstract**: Human crowd simulation in virtual reality (VR) is a powerful tool with potential applications including emergency evacuation training and assessment of building layout. While haptic feedback in VR enhances immersive experience, its effect on walking behavior in dense and dynamic pedestrian flows is unknown. Through a user study, we investigated how haptic feedback changes user walking motion in crowded pedestrian flows in VR. The results indicate that haptic feedback changed users' collision avoidance movements, as measured by increased walking trajectory length and change in pelvis angle. The displacements of users' lateral position and pelvis angle were also increased in the instantaneous response to a collision with a non-player character (NPC), even when the NPC was inside the field of view. Haptic feedback also enhanced users' awareness and visual exploration when an NPC approached from the side and back. Furthermore, variation in walking speed was increased by the haptic feedback. These results suggested that the haptic feedback enhanced users' sensitivity to a collision in VR environment. 

**Abstract (ZH)**: 虚拟现实（VR）中的人群仿真：触觉反馈对密集动态行人流中行走行为的影响 

---
# How do Foundation Models Compare to Skeleton-Based Approaches for Gesture Recognition in Human-Robot Interaction? 

**Title (ZH)**: 基于骨骼的动作识别：基础模型与骨架基方法在人机交互中的对比研究 

**Authors**: Stephanie Käs, Anton Burenko, Louis Markert, Onur Alp Culha, Dennis Mack, Timm Linder, Bastian Leibe  

**Link**: [PDF](https://arxiv.org/pdf/2506.20795)  

**Abstract**: Gestures enable non-verbal human-robot communication, especially in noisy environments like agile production. Traditional deep learning-based gesture recognition relies on task-specific architectures using images, videos, or skeletal pose estimates as input. Meanwhile, Vision Foundation Models (VFMs) and Vision Language Models (VLMs) with their strong generalization abilities offer potential to reduce system complexity by replacing dedicated task-specific modules. This study investigates adapting such models for dynamic, full-body gesture recognition, comparing V-JEPA (a state-of-the-art VFM), Gemini Flash 2.0 (a multimodal VLM), and HD-GCN (a top-performing skeleton-based approach). We introduce NUGGET, a dataset tailored for human-robot communication in intralogistics environments, to evaluate the different gesture recognition approaches. In our experiments, HD-GCN achieves best performance, but V-JEPA comes close with a simple, task-specific classification head - thus paving a possible way towards reducing system complexity, by using it as a shared multi-task model. In contrast, Gemini struggles to differentiate gestures based solely on textual descriptions in the zero-shot setting, highlighting the need of further research on suitable input representations for gestures. 

**Abstract (ZH)**: 手势使非言语的人机通信成为可能，特别是在像 agile 生产这样噪声环境中的应用。传统的基于深度学习的手势识别依赖于特定任务的架构，使用图像、视频或骨骼姿态估计作为输入。同时，具有强大泛化能力的视觉基础模型（VFMs）和视觉语言模型（VLMs）有可能通过替代专用的特定任务模块来简化系统复杂性。本研究调查了将此类模型适应于动态的全身心势识别，并对比了 V-JEPA（最先进的 VFMs）、Gemini Flash 2.0（一种多模态 VLM）和 HD-GCN（一种表现最佳的基于骨骼的方法）。我们引入了 NUGGET 数据集，该数据集专门针对仓储物流环境中的手语识别，以评估不同的手语识别方法。在我们的实验中，HD-GCN 达到了最佳性能，而 V-JEPA 通过一个简单的特定任务分类头达到了相近的性能，从而为使用其作为共享多任务模型以减少系统复杂性铺平了一条可能的道路。相比之下，Gemini 在零样本设置下仅凭文字描述难以区分手势，这突显了进一步研究适合手势识别的输入表示形式所需。 

---
# ConViTac: Aligning Visual-Tactile Fusion with Contrastive Representations 

**Title (ZH)**: ConViTac: 对比表示下的视觉-触觉融合对齐 

**Authors**: Zhiyuan Wu, Yongqiang Zhao, Shan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.20757)  

**Abstract**: Vision and touch are two fundamental sensory modalities for robots, offering complementary information that enhances perception and manipulation tasks. Previous research has attempted to jointly learn visual-tactile representations to extract more meaningful information. However, these approaches often rely on direct combination, such as feature addition and concatenation, for modality fusion, which tend to result in poor feature integration. In this paper, we propose ConViTac, a visual-tactile representation learning network designed to enhance the alignment of features during fusion using contrastive representations. Our key contribution is a Contrastive Embedding Conditioning (CEC) mechanism that leverages a contrastive encoder pretrained through self-supervised contrastive learning to project visual and tactile inputs into unified latent embeddings. These embeddings are used to couple visual-tactile feature fusion through cross-modal attention, aiming at aligning the unified representations and enhancing performance on downstream tasks. We conduct extensive experiments to demonstrate the superiority of ConViTac in real world over current state-of-the-art methods and the effectiveness of our proposed CEC mechanism, which improves accuracy by up to 12.0% in material classification and grasping prediction tasks. 

**Abstract (ZH)**: 视觉和触觉是机器人感知和执行任务的基本传感模态，能够提供互补的信息以增强感知和操作任务。以往的研究试图联合学习视觉-触觉表示，以提取更有意义的信息。然而，这些方法通常依赖于特征直接组合，如特征相加和连接，这种方式往往导致特征整合效果较差。本文提出了一种名为ConViTac的视觉-触觉表示学习网络，旨在通过对比表示增强融合过程中的特征对齐。我们的核心贡献是一种对比嵌入条件（CEC）机制，该机制利用通过自我监督对比学习预训练的对比编码器，将视觉和触觉输入投影到统一的潜在嵌入中。这些嵌入通过跨模态注意力机制耦合视觉-触觉特征融合，旨在对齐统一表示并提升下游任务的表现。我们进行了广泛的实验来证明，在真实世界的材料分类和抓取预测任务中，ConViTac在当前最先进的方法中表现出色，并且我们提出的CEC机制的有效性，该机制在材料分类和抓取预测任务中的准确率最多可以提高12.0%。 

---
# SEPT: Standard-Definition Map Enhanced Scene Perception and Topology Reasoning for Autonomous Driving 

**Title (ZH)**: SEPT: 标准清晰度地图增强的场景感知与拓扑推理方法及其在自动驾驶中的应用 

**Authors**: Muleilan Pei, Jiayao Shan, Peiliang Li, Jieqi Shi, Jing Huo, Yang Gao, Shaojie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12246)  

**Abstract**: Online scene perception and topology reasoning are critical for autonomous vehicles to understand their driving environments, particularly for mapless driving systems that endeavor to reduce reliance on costly High-Definition (HD) maps. However, recent advances in online scene understanding still face limitations, especially in long-range or occluded scenarios, due to the inherent constraints of onboard sensors. To address this challenge, we propose a Standard-Definition (SD) Map Enhanced scene Perception and Topology reasoning (SEPT) framework, which explores how to effectively incorporate the SD map as prior knowledge into existing perception and reasoning pipelines. Specifically, we introduce a novel hybrid feature fusion strategy that combines SD maps with Bird's-Eye-View (BEV) features, considering both rasterized and vectorized representations, while mitigating potential misalignment between SD maps and BEV feature spaces. Additionally, we leverage the SD map characteristics to design an auxiliary intersection-aware keypoint detection task, which further enhances the overall scene understanding performance. Experimental results on the large-scale OpenLane-V2 dataset demonstrate that by effectively integrating SD map priors, our framework significantly improves both scene perception and topology reasoning, outperforming existing methods by a substantial margin. 

**Abstract (ZH)**: 标准定义地图增强的场景感知和拓扑推理框架：SEPT 

---
