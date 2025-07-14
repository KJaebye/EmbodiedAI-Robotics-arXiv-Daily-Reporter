# Learning human-to-robot handovers through 3D scene reconstruction 

**Title (ZH)**: 基于3D场景重建的人机交接学习 

**Authors**: Yuekun Wu, Yik Lung Pang, Andrea Cavallaro, Changjae Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.08726)  

**Abstract**: Learning robot manipulation policies from raw, real-world image data requires a large number of robot-action trials in the physical environment. Although training using simulations offers a cost-effective alternative, the visual domain gap between simulation and robot workspace remains a major limitation. Gaussian Splatting visual reconstruction methods have recently provided new directions for robot manipulation by generating realistic environments. In this paper, we propose the first method for learning supervised-based robot handovers solely from RGB images without the need of real-robot training or real-robot data collection. The proposed policy learner, Human-to-Robot Handover using Sparse-View Gaussian Splatting (H2RH-SGS), leverages sparse-view Gaussian Splatting reconstruction of human-to-robot handover scenes to generate robot demonstrations containing image-action pairs captured with a camera mounted on the robot gripper. As a result, the simulated camera pose changes in the reconstructed scene can be directly translated into gripper pose changes. We train a robot policy on demonstrations collected with 16 household objects and {\em directly} deploy this policy in the real environment. Experiments in both Gaussian Splatting reconstructed scene and real-world human-to-robot handover experiments demonstrate that H2RH-SGS serves as a new and effective representation for the human-to-robot handover task. 

**Abstract (ZH)**: 从RGB图像学习基于监督的机器人手递策略：基于稀疏视图高斯散点图重建的方法 

---
# Multi-critic Learning for Whole-body End-effector Twist Tracking 

**Title (ZH)**: 全身末端执行器螺旋追踪的多评价者学习方法 

**Authors**: Aravind Elanjimattathil Vijayan, Andrei Cramariuc, Mattia Risiglione, Christian Gehring, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2507.08656)  

**Abstract**: Learning whole-body control for locomotion and arm motions in a single policy has challenges, as the two tasks have conflicting goals. For instance, efficient locomotion typically favors a horizontal base orientation, while end-effector tracking may benefit from base tilting to extend reachability. Additionally, current Reinforcement Learning (RL) approaches using a pose-based task specification lack the ability to directly control the end-effector velocity, making smoothly executing trajectories very challenging. To address these limitations, we propose an RL-based framework that allows for dynamic, velocity-aware whole-body end-effector control. Our method introduces a multi-critic actor architecture that decouples the reward signals for locomotion and manipulation, simplifying reward tuning and allowing the policy to resolve task conflicts more effectively. Furthermore, we design a twist-based end-effector task formulation that can track both discrete poses and motion trajectories. We validate our approach through a set of simulation and hardware experiments using a quadruped robot equipped with a robotic arm. The resulting controller can simultaneously walk and move its end-effector and shows emergent whole-body behaviors, where the base assists the arm in extending the workspace, despite a lack of explicit formulations. 

**Abstract (ZH)**: 基于强化学习的动态端效应器全身体态控制框架：解决移动和臂动任务间的冲突 

---
# Robotic Calibration Based on Haptic Feedback Improves Sim-to-Real Transfer 

**Title (ZH)**: 基于触觉反馈的机器人标定改进从模拟到现实的迁移 

**Authors**: Juraj Gavura, Michal Vavrecka, Igor Farkas, Connor Gade  

**Link**: [PDF](https://arxiv.org/pdf/2507.08572)  

**Abstract**: When inverse kinematics (IK) is adopted to control robotic arms in manipulation tasks, there is often a discrepancy between the end effector (EE) position of the robot model in the simulator and the physical EE in reality. In most robotic scenarios with sim-to-real transfer, we have information about joint positions in both simulation and reality, but the EE position is only available in simulation. We developed a novel method to overcome this difficulty based on haptic feedback calibration, using a touchscreen in front of the robot that provides information on the EE position in the real environment. During the calibration procedure, the robot touches specific points on the screen, and the information is stored. In the next stage, we build a transformation function from the data based on linear transformation and neural networks that is capable of outputting all missing variables from any partial input (simulated/real joint/EE position). Our results demonstrate that a fully nonlinear neural network model performs best, significantly reducing positioning errors. 

**Abstract (ZH)**: 基于触控屏的触觉反馈校准的逆运动学仿真到现实转移方法 

---
# LiDAR, GNSS and IMU Sensor Alignment through Dynamic Time Warping to Construct 3D City Maps 

**Title (ZH)**: 通过动态时间战争整形准LiDAR、GNSS和IMU传感器以构建3D城市地图 

**Authors**: Haitian Wang, Hezam Albaqami, Xinyu Wang, Muhammad Ibrahim, Zainy M. Malakan, Abdullah M. Algamdi, Mohammed H. Alghamdi, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2507.08420)  

**Abstract**: LiDAR-based 3D mapping suffers from cumulative drift causing global misalignment, particularly in GNSS-constrained environments. To address this, we propose a unified framework that fuses LiDAR, GNSS, and IMU data for high-resolution city-scale mapping. The method performs velocity-based temporal alignment using Dynamic Time Warping and refines GNSS and IMU signals via extended Kalman filtering. Local maps are built using Normal Distributions Transform-based registration and pose graph optimization with loop closure detection, while global consistency is enforced using GNSS-constrained anchors followed by fine registration of overlapping segments. We also introduce a large-scale multimodal dataset captured in Perth, Western Australia to facilitate future research in this direction. Our dataset comprises 144{,}000 frames acquired with a 128-channel Ouster LiDAR, synchronized RTK-GNSS trajectories, and MEMS-IMU measurements across 21 urban loops. To assess geometric consistency, we evaluated our method using alignment metrics based on road centerlines and intersections to capture both global and local accuracy. Our method reduces the average global alignment error from 3.32\,m to 1.24\,m, achieving a 61.4\% improvement. The constructed high-fidelity map supports a wide range of applications, including smart city planning, geospatial data integration, infrastructure monitoring, and GPS-free navigation. Our method, and dataset together establish a new benchmark for evaluating 3D city mapping in GNSS-constrained environments. The dataset and code will be released publicly. 

**Abstract (ZH)**: 基于LiDAR的3D建图受累积漂移影响，特别是在GNSS约束环境中会出现全局错位。为此，我们提出了一种统一框架，融合LiDAR、GNSS和IMU数据以实现高分辨率城市规模建图。该方法使用动态时间弯曲进行基于速度的时间对齐，并通过扩展卡尔曼滤波精化GNSS和IMU信号。局部地图使用基于正态分布变换的注册和位姿图优化构建，并通过环闭检测进行细化注册，而全局一致性则通过GNSS约束锚点实现，并随后对重叠段进行精细注册。我们还在西澳大利亚珀斯引入了一个大规模多模态数据集，以促进未来在此方向上的研究。该数据集包括使用128通道Ouster LiDAR获取的144,000帧数据，同步RTK-GNSS轨迹，以及21个城市环路中的MEMS-IMU测量值。为了评估几何一致性，我们使用基于道路中心线和交叉口的对齐度量来评估我们的方法，以捕捉全局和局部精度。我们的方法将平均全局对齐错误从3.32 m降低到1.24 m，实现了61.4%的改进。构建的高保真地图支持智慧城市规划、地理空间数据整合、基础设施监测和GPS-free导航等广泛的应用。我们的方法和数据集一起为评估GNSS约束环境中3D城市建图设立了新基准。数据集和代码将公开发布。 

---
# Intelligent Control of Spacecraft Reaction Wheel Attitude Using Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的空间飞行器反应轮姿态智能控制 

**Authors**: Ghaith El-Dalahmeh, Mohammad Reza Jabbarpour, Bao Quoc Vo, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.08366)  

**Abstract**: Reliable satellite attitude control is essential for the success of space missions, particularly as satellites increasingly operate autonomously in dynamic and uncertain environments. Reaction wheels (RWs) play a pivotal role in attitude control, and maintaining control resilience during RW faults is critical to preserving mission objectives and system stability. However, traditional Proportional Derivative (PD) controllers and existing deep reinforcement learning (DRL) algorithms such as TD3, PPO, and A2C often fall short in providing the real time adaptability and fault tolerance required for autonomous satellite operations. This study introduces a DRL-based control strategy designed to improve satellite resilience and adaptability under fault conditions. Specifically, the proposed method integrates Twin Delayed Deep Deterministic Policy Gradient (TD3) with Hindsight Experience Replay (HER) and Dimension Wise Clipping (DWC) referred to as TD3-HD to enhance learning in sparse reward environments and maintain satellite stability during RW failures. The proposed approach is benchmarked against PD control and leading DRL algorithms. Experimental results show that TD3-HD achieves significantly lower attitude error, improved angular velocity regulation, and enhanced stability under fault conditions. These findings underscore the proposed method potential as a powerful, fault tolerant, onboard AI solution for autonomous satellite attitude control. 

**Abstract (ZH)**: 可靠的人造卫星姿态控制对于太空任务的成功至关重要，特别是在卫星在动态和不确定环境中自主运行的情况下。反应轮（RWs）在姿态控制中发挥着关键作用，维持RW故障期间的姿态控制韧性对于保持任务目标和系统稳定性至关重要。然而，传统的比例微分（PD）控制器和现有的深度强化学习（DRL）算法如TD3、PPO和A2C往往在提供自主卫星操作所需的实时适应性和容错性方面存在不足。本文提出了一种基于DRL的控制策略，以提高故障条件下的人造卫星韧性与适应性。具体来说，所提出的方法将Twin Delayed Deep Deterministic Policy Gradient（TD3）与Hindsight Experience Replay（HER）和Dimension Wise Clipping（DWC）结合，称为TD3-HD，以增强稀疏奖励环境下的学习能力并在反应轮故障期间保持卫星稳定性。所提出的方法与PD控制和领先DRL算法进行了基准测试。实验结果表明，TD3-HD在故障条件下实现了显著更低的姿态误差、改进的角速度调节和增强的稳定性。这些发现突显了所提出方法作为自主卫星姿态控制的强健且容错的机载人工智能解决方案的巨大潜力。 

---
# Towards Robust Sensor-Fusion Ground SLAM: A Comprehensive Benchmark and A Resilient Framework 

**Title (ZH)**: 面向鲁棒传感器融合地面SLAM：一项全面基准与鲁棒框架研究 

**Authors**: Deteng Zhang, Junjie Zhang, Yan Sun, Tao Li, Hao Yin, Hongzhao Xie, Jie Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.08364)  

**Abstract**: Considerable advancements have been achieved in SLAM methods tailored for structured environments, yet their robustness under challenging corner cases remains a critical limitation. Although multi-sensor fusion approaches integrating diverse sensors have shown promising performance improvements, the research community faces two key barriers: On one hand, the lack of standardized and configurable benchmarks that systematically evaluate SLAM algorithms under diverse degradation scenarios hinders comprehensive performance assessment. While on the other hand, existing SLAM frameworks primarily focus on fusing a limited set of sensor types, without effectively addressing adaptive sensor selection strategies for varying environmental conditions.
To bridge these gaps, we make three key contributions: First, we introduce M3DGR dataset: a sensor-rich benchmark with systematically induced degradation patterns including visual challenge, LiDAR degeneracy, wheel slippage and GNSS denial. Second, we conduct a comprehensive evaluation of forty SLAM systems on M3DGR, providing critical insights into their robustness and limitations under challenging real-world conditions. Third, we develop a resilient modular multi-sensor fusion framework named Ground-Fusion++, which demonstrates robust performance by coupling GNSS, RGB-D, LiDAR, IMU (Inertial Measurement Unit) and wheel odometry. Codes and datasets are publicly available. 

**Abstract (ZH)**: 针对结构化环境定制的SLAM方法取得了显著进展，但在挑战性corner cases中的鲁棒性仍存在重大局限。尽管将各种传感器融合的方法显示出了有前景的性能提升，但研究界面临两大关键障碍：一方面，缺乏标准化和可配置的基准来系统评估多种退化场景下SLAM算法的性能，阻碍了综合性能评估。另一方面，现有的SLAM框架主要集中在融合有限类型的传感器上，未能有效解决适应不同环境条件的传感器选择策略。为弥合这些差距，我们做出了三项关键贡献：首先，我们引入了M3DGR数据集：一个包含系统诱导退化模式的传感器丰富基准，包括视觉挑战、LiDAR退化、车轮打滑和GNSS拒识。其次，我们在M3DGR上对四十种SLAM系统进行了全面评估，提供了在挑战性现实环境条件下其鲁棒性和局限性的关键洞见。第三，我们开发了一种鲁棒的模块化多传感器融合框架，名为Ground-Fusion++，通过结合GNSS、RGB-D、LiDAR、IMU（惯性测量单元）和车轮里程计，展示了鲁棒性能。代码和数据集已公开。 

---
# Joint Optimization-based Targetless Extrinsic Calibration for Multiple LiDARs and GNSS-Aided INS of Ground Vehicles 

**Title (ZH)**: 基于联合优化的目标导向外标定方法及其在GNSS辅助INS与多LiDAR装备地面车辆中的应用 

**Authors**: Junhui Wang, Yan Qiao, Chao Gao, Naiqi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08349)  

**Abstract**: Accurate extrinsic calibration between multiple LiDAR sensors and a GNSS-aided inertial navigation system (GINS) is essential for achieving reliable sensor fusion in intelligent mining environments. Such calibration enables vehicle-road collaboration by aligning perception data from vehicle-mounted sensors to a unified global reference frame. However, existing methods often depend on artificial targets, overlapping fields of view, or precise trajectory estimation, which are assumptions that may not hold in practice. Moreover, the planar motion of mining vehicles leads to observability issues that degrade calibration performance. This paper presents a targetless extrinsic calibration method that aligns multiple onboard LiDAR sensors to the GINS coordinate system without requiring overlapping sensor views or external targets. The proposed approach introduces an observation model based on the known installation height of the GINS unit to constrain unobservable calibration parameters under planar motion. A joint optimization framework is developed to refine both the extrinsic parameters and GINS trajectory by integrating multiple constraints derived from geometric correspondences and motion consistency. The proposed method is applicable to heterogeneous LiDAR configurations, including both mechanical and solid-state sensors. Extensive experiments on simulated and real-world datasets demonstrate the accuracy, robustness, and practical applicability of the approach under diverse sensor setups. 

**Abstract (ZH)**: 多LiDAR传感器与GNSS辅助惯性导航系统(GINS)之间的无目标外部标定对于实现智能采矿环境中可靠的传感器融合至关重要。提出的方法无需重叠传感器视场或外部目标即可将多个车载LiDAR传感器对准GINS坐标系。该方法基于GINS单元的已知安装高度引入观测模型，以在平动运动下约束不可观测的标定参数。开发了一种联合优化框架，通过集成来自几何对应和运动一致性推导出的多个约束来细化外部参数和GINS轨迹。该方法适用于包括机械和固态传感器在内的异构LiDAR配置。在模拟和实际数据集上的广泛实验表明，该方法在不同传感器配置下具有高精度、鲁棒性和实用适用性。 

---
# Learning Robust Motion Skills via Critical Adversarial Attacks for Humanoid Robots 

**Title (ZH)**: 基于关键对抗攻击学习 robust 运动技能的人形机器人 

**Authors**: Yang Zhang, Zhanxiang Cao, Buqing Nie, Haoyang Li, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.08303)  

**Abstract**: Humanoid robots show significant potential in daily tasks. However, reinforcement learning-based motion policies often suffer from robustness degradation due to the sim-to-real dynamics gap, thereby affecting the agility of real robots. In this work, we propose a novel robust adversarial training paradigm designed to enhance the robustness of humanoid motion policies in real worlds. The paradigm introduces a learnable adversarial attack network that precisely identifies vulnerabilities in motion policies and applies targeted perturbations, forcing the motion policy to enhance its robustness against perturbations through dynamic adversarial training. We conduct experiments on the Unitree G1 humanoid robot for both perceptive locomotion and whole-body control tasks. The results demonstrate that our proposed method significantly enhances the robot's motion robustness in real world environments, enabling successful traversal of challenging terrains and highly agile whole-body trajectory tracking. 

**Abstract (ZH)**: 类人机器人在日常任务中展现出显著潜力。然而，基于强化学习的运动策略往往会因模拟到现实的动力学差距而导致鲁棒性下降，从而影响实际机器人动作的敏捷性。本文提出了一种新颖的鲁棒对抗训练范式，旨在增强类人运动策略在真实世界中的鲁棒性。该范式引入了一个可学习的对抗攻击网络，能够精确识别运动策略中的漏洞并施加针对性的扰动，迫使运动策略通过动态对抗训练提升其对干扰的鲁棒性。我们在Unitree G1类人机器人上分别进行了感知移动和全身控制任务实验。结果表明，所提出的方法显著增强了机器人的运动鲁棒性，使其能够在复杂地形中成功穿越并实现高度敏捷的全身轨迹跟踪。 

---
# CL3R: 3D Reconstruction and Contrastive Learning for Enhanced Robotic Manipulation Representations 

**Title (ZH)**: CL3R: 三维重建与对比学习增强的机器人Manipulation表示 

**Authors**: Wenbo Cui, Chengyang Zhao, Yuhui Chen, Haoran Li, Zhizheng Zhang, Dongbin Zhao, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08262)  

**Abstract**: Building a robust perception module is crucial for visuomotor policy learning. While recent methods incorporate pre-trained 2D foundation models into robotic perception modules to leverage their strong semantic understanding, they struggle to capture 3D spatial information and generalize across diverse camera viewpoints. These limitations hinder the policy's effectiveness, especially in fine-grained robotic manipulation scenarios. To address these challenges, we propose CL3R, a novel 3D pre-training framework designed to enhance robotic manipulation policies. Our method integrates both spatial awareness and semantic understanding by employing a point cloud Masked Autoencoder to learn rich 3D representations while leveraging pre-trained 2D foundation models through contrastive learning for efficient semantic knowledge transfer. Additionally, we propose a 3D visual representation pre-training framework for robotic tasks. By unifying coordinate systems across datasets and introducing random fusion of multi-view point clouds, we mitigate camera view ambiguity and improve generalization, enabling robust perception from novel viewpoints at test time. Extensive experiments in both simulation and the real world demonstrate the superiority of our method, highlighting its effectiveness in visuomotor policy learning for robotic manipulation. 

**Abstract (ZH)**: 构建稳健的感知模块对于视觉运动策略学习至关重要。虽然近期方法通过引入预训练的2D基础模型以利用其强大的语义理解能力来增强机器人的感知模块，但它们难以捕捉3D空间信息并跨多种相机视角进行泛化。这些限制阻碍了策略的有效性，特别是在精细的机器人操作场景中。为应对这些挑战，我们提出CL3R，一种新颖的3D预训练框架，旨在增强机器人的操作策略。我们的方法通过结合空间意识和语义理解，利用点云Masked Autoencoder学习丰富的3D表示，并通过对比学习利用预训练的2D基础模型进行高效的语义知识迁移。此外，我们还提出了一种适用于机器人任务的3D视觉表示预训练框架。通过统一数据集之间的坐标系，并引入多视图点云的随机融合，我们解决了相机视角的歧义性并提高了泛化能力，从而在测试时实现新型视角下的稳健感知。广泛的仿真实验和真实世界实验表明了我们方法的优势，突显了其在机器人操作中的视觉运动策略学习中的有效性。 

---
# Making VLMs More Robot-Friendly: Self-Critical Distillation of Low-Level Procedural Reasoning 

**Title (ZH)**: 让大模型更友好数字人：自批判性低级程序推理提取 

**Authors**: Chan Young Park, Jillian Fisher, Marius Memmel, Dipika Khullar, Andy Yun, Abhishek Gupta, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.08224)  

**Abstract**: Large language models (LLMs) have shown promise in robotic procedural planning, yet their human-centric reasoning often omits the low-level, grounded details needed for robotic execution. Vision-language models (VLMs) offer a path toward more perceptually grounded plans, but current methods either rely on expensive, large-scale models or are constrained to narrow simulation settings. We introduce SelfReVision, a lightweight and scalable self-improvement framework for vision-language procedural planning. SelfReVision enables small VLMs to iteratively critique, revise, and verify their own plans-without external supervision or teacher models-drawing inspiration from chain-of-thought prompting and self-instruct paradigms. Through this self-distillation loop, models generate higher-quality, execution-ready plans that can be used both at inference and for continued fine-tuning. Using models varying from 3B to 72B, our results show that SelfReVision not only boosts performance over weak base VLMs but also outperforms models 100X the size, yielding improved control in downstream embodied tasks. 

**Abstract (ZH)**: SelfReVision：一种轻量级可扩展的自我改进框架，用于视觉语言程序规划 

---
# Imitation Learning for Obstacle Avoidance Using End-to-End CNN-Based Sensor Fusion 

**Title (ZH)**: 基于端到端CNN融合的模仿学习用于障碍 avoidance 

**Authors**: Lamiaa H. Zain, Hossam H. Ammar, Raafat E. Shalaby  

**Link**: [PDF](https://arxiv.org/pdf/2507.08112)  

**Abstract**: Obstacle avoidance is crucial for mobile robots' navigation in both known and unknown environments. This research designs, trains, and tests two custom Convolutional Neural Networks (CNNs), using color and depth images from a depth camera as inputs. Both networks adopt sensor fusion to produce an output: the mobile robot's angular velocity, which serves as the robot's steering command. A newly obtained visual dataset for navigation was collected in diverse environments with varying lighting conditions and dynamic obstacles. During data collection, a communication link was established over Wi-Fi between a remote server and the robot, using Robot Operating System (ROS) topics. Velocity commands were transmitted from the server to the robot, enabling synchronized recording of visual data and the corresponding steering commands. Various evaluation metrics, such as Mean Squared Error, Variance Score, and Feed-Forward time, provided a clear comparison between the two networks and clarified which one to use for the application. 

**Abstract (ZH)**: 移动机器人在已知和未知环境中的避障导航至关重要。本研究设计、训练并测试了两个自定义卷积神经网络（CNN），使用深度摄像头的彩色和深度图像作为输入。两个网络均采用传感器融合生成输出：移动机器人的角速度，作为机器人的转向指令。在一个包含多种照明条件和动态障碍物的环境中，收集了新的导航视觉数据集。在数据收集过程中，通过Wi-Fi在远程服务器和机器人之间建立了通信连接，并使用ROS话题传输速度指令，实现在服务器和机器人之间同步记录视觉数据及其对应的转向指令。通过多种评估指标，如均方误差、方差分数和前向传递时间，对两个网络进行了清晰比较，明确了适用于该应用的网络。 

---
# Noise-Enabled Goal Attainment in Crowded Collectives 

**Title (ZH)**: 噪声驱动的目标达成在拥挤集体中 

**Authors**: Lucy Liu, Justin Werfel, Federico Toschi, L. Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2507.08100)  

**Abstract**: In crowded environments, individuals must navigate around other occupants to reach their destinations. Understanding and controlling traffic flows in these spaces is relevant to coordinating robot swarms and designing infrastructure for dense populations. Here, we combine simulations, theory, and robotic experiments to study how noisy motion can disrupt traffic jams and enable flow as agents travel to individual goals. Above a critical noise level, large jams do not persist. From this observation, we analytically approximate the goal attainment rate as a function of the noise level, then solve for the optimal agent density and noise level that maximize the swarm's goal attainment rate. We perform robotic experiments to corroborate our simulated and theoretical results. Finally, we compare simple, local navigation approaches with a sophisticated but computationally costly central planner. A simple reactive scheme performs well up to moderate densities and is far more computationally efficient than a planner, suggesting lessons for real-world problems. 

**Abstract (ZH)**: 在拥挤环境中，个体必须在其他占有人周围导航以到达目的地。理解并控制这些空间中的交通流对于协调机器人集群和设计密集人口的基础设施是相关的。在这里，我们结合模拟、理论和机器人实验来研究噪声运动如何破坏交通拥堵并促进个体目标导向过程中的流量。当噪声水平超过临界值时，大型拥堵不会持续。基于这一观察，我们分析地将目标达成率近似为噪声水平的函数，然后求解使集群目标达成率最大化的最优代理密度和噪声水平。我们进行机器人实验以验证我们的模拟和理论结果。最后，我们将简单的局部导航方法与复杂的但计算成本高的中央规划者进行比较。一个简单的反应方案在中等密度下表现良好，并且比规划者更加计算高效，这为现实世界问题提供了启示。 

---
# SPLASH! Sample-efficient Preference-based inverse reinforcement learning for Long-horizon Adversarial tasks from Suboptimal Hierarchical demonstrations 

**Title (ZH)**: SPLASH! 基于偏好样本高效逆强化学习方法，用于从次优分层示范学习长期对抗任务 

**Authors**: Peter Crowley, Zachary Serlin, Tyler Paine, Makai Mann, Michael Benjamin, Calin Belta  

**Link**: [PDF](https://arxiv.org/pdf/2507.08707)  

**Abstract**: Inverse Reinforcement Learning (IRL) presents a powerful paradigm for learning complex robotic tasks from human demonstrations. However, most approaches make the assumption that expert demonstrations are available, which is often not the case. Those that allow for suboptimality in the demonstrations are not designed for long-horizon goals or adversarial tasks. Many desirable robot capabilities fall into one or both of these categories, thus highlighting a critical shortcoming in the ability of IRL to produce field-ready robotic agents. We introduce Sample-efficient Preference-based inverse reinforcement learning for Long-horizon Adversarial tasks from Suboptimal Hierarchical demonstrations (SPLASH), which advances the state-of-the-art in learning from suboptimal demonstrations to long-horizon and adversarial settings. We empirically validate SPLASH on a maritime capture-the-flag task in simulation, and demonstrate real-world applicability with sim-to-real translation experiments on autonomous unmanned surface vehicles. We show that our proposed methods allow SPLASH to significantly outperform the state-of-the-art in reward learning from suboptimal demonstrations. 

**Abstract (ZH)**: 基于偏好且样本高效的逆强化学习：从次优化层次示学习长时域对抗任务（SPLASH） 

---
# An Embedded Real-time Object Alert System for Visually Impaired: A Monocular Depth Estimation based Approach through Computer Vision 

**Title (ZH)**: 基于单目深度估计的面向视障者的嵌入式实时对象警报系统：计算机视觉方法 

**Authors**: Jareen Anjom, Rashik Iram Chowdhury, Tarbia Hasan, Md. Ishan Arefin Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2507.08165)  

**Abstract**: Visually impaired people face significant challenges in their day-to-day commutes in the urban cities of Bangladesh due to the vast number of obstructions on every path. With many injuries taking place through road accidents on a daily basis, it is paramount for a system to be developed that can alert the visually impaired of objects at close distance beforehand. To overcome this issue, a novel alert system is proposed in this research to assist the visually impaired in commuting through these busy streets without colliding with any objects. The proposed system can alert the individual to objects that are present at a close distance. It utilizes transfer learning to train models for depth estimation and object detection, and combines both models to introduce a novel system. The models are optimized through the utilization of quantization techniques to make them lightweight and efficient, allowing them to be easily deployed on embedded systems. The proposed solution achieved a lightweight real-time depth estimation and object detection model with an mAP50 of 0.801. 

**Abstract (ZH)**: 盲人在 Bangladesh 的城市日常出行中面临严峻挑战，由于道路上遍布各种障碍物。为了应对道路事故频发的问题，亟需开发一种系统，在障碍物靠近时提前警示盲人。为此，本研究提出了一种新型警示系统，旨在帮助盲人在繁忙的街道上安全通行，避免与障碍物相撞。该系统能够警示个体靠近的障碍物。通过迁移学习训练深度估算和目标检测模型，并将两者结合，引入了新的系统。通过量化技术对模型进行优化，使其轻量化且高效，便于部署于嵌入式系统中。所提出的方法实现了轻量级实时深度估算和目标检测模型，mAP50达到0.801。 

---
