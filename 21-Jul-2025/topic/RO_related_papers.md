# Context-Aware Behavior Learning with Heuristic Motion Memory for Underwater Manipulation 

**Title (ZH)**: 基于上下文感知的启发式运动记忆 underwater 操作行为学习 

**Authors**: Markus Buchholz, Ignacio Carlucho, Michele Grimaldi, Maria Koskinopoulou, Yvan R. Petillot  

**Link**: [PDF](https://arxiv.org/pdf/2507.14099)  

**Abstract**: Autonomous motion planning is critical for efficient and safe underwater manipulation in dynamic marine environments. Current motion planning methods often fail to effectively utilize prior motion experiences and adapt to real-time uncertainties inherent in underwater settings. In this paper, we introduce an Adaptive Heuristic Motion Planner framework that integrates a Heuristic Motion Space (HMS) with Bayesian Networks to enhance motion planning for autonomous underwater manipulation. Our approach employs the Probabilistic Roadmap (PRM) algorithm within HMS to optimize paths by minimizing a composite cost function that accounts for distance, uncertainty, energy consumption, and execution time. By leveraging HMS, our framework significantly reduces the search space, thereby boosting computational performance and enabling real-time planning capabilities. Bayesian Networks are utilized to dynamically update uncertainty estimates based on real-time sensor data and environmental conditions, thereby refining the joint probability of path success. Through extensive simulations and real-world test scenarios, we showcase the advantages of our method in terms of enhanced performance and robustness. This probabilistic approach significantly advances the capability of autonomous underwater robots, ensuring optimized motion planning in the face of dynamic marine challenges. 

**Abstract (ZH)**: 自主运动规划对于动态海洋环境下高效安全的水下操作至关重要。当前的运动规划方法往往未能有效利用先前的运动经验并适应水下环境中固有的实时不确定性。本文提出了一种自适应启发式运动规划框架，该框架结合了启发式运动空间（HMS）与贝叶斯网络以增强自主水下操作的运动规划。我们的方法在HMS中使用概率 roadmap（PRM）算法通过最小化综合成本函数（考虑了距离、不确定性、能耗和执行时间）来优化路径。通过利用HMS，我们的框架显著减少了搜索空间，从而提升计算性能并实现实时规划能力。贝叶斯网络用于根据实时传感器数据和环境条件动态更新不确定性估计，从而 refinethe joint probability of path success。通过广泛的仿真和实际测试场景，我们展示了该方法在性能和鲁棒性方面的优势。这种概率方法显著提升了自主水下机器人的能力，在面对动态海洋挑战时实现优化的运动规划。 

---
# Design of a Modular Mobile Inspection and Maintenance Robot for an Orbital Servicing Hub 

**Title (ZH)**: orbits维修枢纽的模块化移动检查与维护机器人设计 

**Authors**: Tianyuan Wang, Mark A Post, Mathieu Deremetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.14059)  

**Abstract**: The use of autonomous robots in space is an essential part of the "New Space" commercial ecosystem of assembly and re-use of space hardware components in Earth orbit and beyond. The STARFAB project aims to create a ground demonstration of an orbital automated warehouse as a hub for sustainable commercial operations and servicing. A critical part of this fully-autonomous robotic facility will be the capability to monitor, inspect, and assess the condition of both the components stored in the warehouse, and the STARFAB facility itself. This paper introduces ongoing work on the STARFAB Mobile Inspection Module (MIM). The MIM uses Standard Interconnects (SI) so that it can be carried by Walking Manipulators (WM) as an independently-mobile robot, and multiple MIMs can be stored and retrieved as needed for operations on STARFAB. The MIM carries high-resolution cameras, a 3D profilometer, and a thermal imaging sensor, with the capability to add other modular sensors. A grasping tool and torque wrench are stored within the modular body for use by an attached WM for maintenance operations. Implementation and testing is still ongoing at the time of writing. This paper details the concept of operations for the MIM as an on-orbit autonomous inspection and maintenance system, the mechanical and electronic design of the MIM, and the sensors package used for non-destructive testing. 

**Abstract (ZH)**: 自主机器人在太空中的应用是“新太空”商业生态系统中地球轨道及 beyond 进行太空硬件组件组装和再利用的核心组成部分。STARFAB 项目旨在创建一个地面展示的轨道自动化仓库，作为可持续商业运营和服务的枢纽。该项目中全自主机器人设施的一个关键组成部分是对存储在仓库中的组件以及 STARFAB 设施本身的监测、检查和评估能力。本文介绍了 STARFAB 移动检查模块（MIM）的研发进展。MIM 使用标准接口（SI），可由步行操作臂（WM）携带作为独立移动的机器人，并可根据操作需求存储和检索多个 MIM。MIM 配备了高分辨率相机、3D 测形仪和热成像传感器，并具备安装其他模块化传感器的能力。一个夹持工具和扭矩扳手储存在模块化机体中，供连接的 WM 在维护操作中使用。撰写本文时，实施和测试仍在进行中。本文详细介绍了 MIM 作为轨道自主检查和维护系统的概念、MIM 的机械和电子设计以及用于非破坏性测试的传感器套件。 

---
# A multi-strategy improved snake optimizer for three-dimensional UAV path planning and engineering problems 

**Title (ZH)**: 基于多策略改进的蛇优化算法在三维无人飞行器路径规划及工程问题中的应用 

**Authors**: Genliang Li, Yaxin Cui, Jinyu Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.14043)  

**Abstract**: Metaheuristic algorithms have gained widespread application across various fields owing to their ability to generate diverse solutions. One such algorithm is the Snake Optimizer (SO), a progressive optimization approach. However, SO suffers from the issues of slow convergence speed and susceptibility to local optima. In light of these shortcomings, we propose a novel Multi-strategy Improved Snake Optimizer (MISO). Firstly, we propose a new adaptive random disturbance strategy based on sine function to alleviate the risk of getting trapped in a local optimum. Secondly, we introduce adaptive Levy flight strategy based on scale factor and leader and endow the male snake leader with flight capability, which makes it easier for the algorithm to leap out of the local optimum and find the global optimum. More importantly, we put forward a position update strategy combining elite leadership and Brownian motion, effectively accelerating the convergence speed while ensuring precision. Finally, to demonstrate the performance of MISO, we utilize 30 CEC2017 test functions and the CEC2022 test suite, comparing it with 11 popular algorithms across different dimensions to validate its effectiveness. Moreover, Unmanned Aerial Vehicle (UAV) has been widely used in various fields due to its advantages of low cost, high mobility and easy operation. However, the UAV path planning problem is crucial for flight safety and efficiency, and there are still challenges in establishing and optimizing the path model. Therefore, we apply MISO to the UAV 3D path planning problem as well as 6 engineering design problems to assess its feasibility in practical applications. The experimental results demonstrate that MISO exceeds other competitive algorithms in terms of solution quality and stability, establishing its strong potential for application. 

**Abstract (ZH)**: Metaheuristic 算法由于能够生成多样化的解决方案，在各个领域得到了广泛应用。其中一种算法是蛇优化器（Snake Optimizer，SO），它是一种渐进优化方法。然而，SO 存在收敛速度慢和易陷入局部最优的问题。鉴于这些不足，我们提出了一种新型的多策略改进蛇优化器（Multi-strategy Improved Snake Optimizer，MISO）。首先，我们提出了一种基于正弦函数的新自适应随机干扰策略，以缓解陷入局部最优的风险。其次，我们引入了基于尺度因子和领导者的自适应莱维飞行策略，并赋予雄蛇领导飞行能力，这使得算法更容易跳出局部最优找到全局最优。更重要的是，我们提出了结合精英领导和布朗运动的位置更新策略，有效地加速了收敛速度并保证了精度。此外，我们使用 30 个 CEC2017 测试函数和 CEC2022 测试集，与 11 种流行的算法在不同维度上进行比较，验证其有效性。同时，由于无人驾驶飞行器（UAV）因其低成本、高移动性和易操作性而在各个领域得到了广泛应用，而 UAV 航线规划对于飞行安全和效率至关重要，并且在建立和优化航线模型方面仍存在挑战。因此，我们将 MISO 应用于 UAV 的 3D 航线规划问题以及 6 个工程设计问题，评估其在实际应用中的可行性。实验结果表明，MISO 在解决方案质量和稳定性方面优于其他竞争算法，展示了其在实际应用中的强大潜力。 

---
# A Minimalist Controller for Autonomously Self-Aggregating Robotic Swarms: Enabling Compact Formations in Multitasking Scenarios 

**Title (ZH)**: 自主自聚集机器人 swarm 的 minimalist 控制器：在多任务场景中实现紧凑 formations 

**Authors**: Maria Eduarda Silva de Macedo, Ana Paula Chiarelli de Souza, Roberto Silvio Ubertino Rosso Jr., Yuri Kaszubowski Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2507.13969)  

**Abstract**: The deployment of simple emergent behaviors in swarm robotics has been well-rehearsed in the literature. A recent study has shown how self-aggregation is possible in a multitask approach -- where multiple self-aggregation task instances occur concurrently in the same environment. The multitask approach poses new challenges, in special, how the dynamic of each group impacts the performance of others. So far, the multitask self-aggregation of groups of robots suffers from generating a circular formation -- that is not fully compact -- or is not fully autonomous. In this paper, we present a multitask self-aggregation where groups of homogeneous robots sort themselves into different compact clusters, relying solely on a line-of-sight sensor. Our multitask self-aggregation behavior was able to scale well and achieve a compact formation. We report scalability results from a series of simulation trials with different configurations in the number of groups and the number of robots per group. We were able to improve the multitask self-aggregation behavior performance in terms of the compactness of the clusters, keeping the proportion of clustered robots found in other studies. 

**Abstract (ZH)**: 基于视线传感器的同构机器人多任务自聚集行为研究 

---
# AeroThrow: An Autonomous Aerial Throwing System for Precise Payload Delivery 

**Title (ZH)**: AeroThrow：一种精确载荷投送的自主飞行投掷系统 

**Authors**: Ziliang Li, Hongming Chen, Yiyang Lin, Biyu Ye, Ximin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13903)  

**Abstract**: Autonomous aerial systems play an increasingly vital role in a wide range of applications, particularly for transport and delivery tasks in complex environments. In airdrop missions, these platforms face the dual challenges of abrupt control mode switching and inherent system delays along with control errors. To address these issues, this paper presents an autonomous airdrop system based on an aerial manipulator (AM). The introduction of additional actuated degrees of freedom enables active compensation for UAV tracking errors. By imposing smooth and continuous constraints on the parabolic landing point, the proposed approach generates aerial throwing trajectories that are less sensitive to the timing of payload release. A hierarchical disturbance compensation strategy is incorporated into the Nonlinear Model Predictive Control (NMPC) framework to mitigate the effects of sudden changes in system parameters, while the predictive capabilities of NMPC are further exploited to improve the precision of aerial throwing. Both simulation and real-world experimental results demonstrate that the proposed system achieves greater agility and precision in airdrop missions. 

**Abstract (ZH)**: 自主飞行系统在多种应用中扮演着越来越重要的角色，特别是在复杂环境下的运输和交付任务中。在空投任务中，这些平台面临着突然控制模式切换和固有系统延迟以及控制误差的双重挑战。为了解决这些问题，本文提出了一种基于航空 manipulator (AM) 的自主空投系统。通过引入额外的可控自由度，该系统能够主动补偿无人机跟踪误差。通过对抛物线着陆点施加平滑且连续的约束，提出的方案生成了对载荷释放时机不那么敏感的空中投掷轨迹。在非线性模型预测控制（NMPC）框架中整合了分层扰动补偿策略，以减轻系统参数突然变化的影响，同时进一步利用NMPC的预测能力以提高空中投掷的精度。仿真和实际实验结果均证明，所提出系统在空投任务中实现了更大的灵活性和精度。 

---
# Design Analysis of an Innovative Parallel Robot for Minimally Invasive Pancreatic Surgery 

**Title (ZH)**: 创新型并联机器人在微创胰腺手术中的设计分析 

**Authors**: Doina Pisla, Alexandru Pusca, Andrei Caprariu, Adrian Pisla, Bogdan Gherman, Calin Vaida, Damien Chablat  

**Link**: [PDF](https://arxiv.org/pdf/2507.13787)  

**Abstract**: This paper focuses on the design of a parallel robot designed for robotic assisted minimally invasive pancreatic surgery. Two alternative architectures, called ATHENA-1 and ATHENA-2, each with 4 degrees of freedom (DOF) are proposed. Their kinematic schemes are presented, and the conceptual 3D CAD models are illustrated. Based on these, two Finite Element Method (FEM) simulations were performed to determine which architecture has the higher stiffness. A workspace quantitative analysis is performed to further assess the usability of the two proposed parallel architectures related to the medical tasks. The obtained results are used to select the architecture which fit the required design criteria and will be used to develop the experimental model of the surgical robot. 

**Abstract (ZH)**: 一种用于机器人辅助微创胰腺手术的并联机器人的设计：ATHENA-1和ATHENA-2架构的提出与评估 

---
# SaWa-ML: Structure-Aware Pose Correction and Weight Adaptation-Based Robust Multi-Robot Localization 

**Title (ZH)**: SaWa-ML: 结构感知姿态纠正与重量自适应基于鲁棒多机器人定位 

**Authors**: Junho Choi, Kihwan Ryoo, Jeewon Kim, Taeyun Kim, Eungchang Lee, Myeongwoo Jeong, Kevin Christiansen Marsim, Hyungtae Lim, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2507.13702)  

**Abstract**: Multi-robot localization is a crucial task for implementing multi-robot systems. Numerous researchers have proposed optimization-based multi-robot localization methods that use camera, IMU, and UWB sensors. Nevertheless, characteristics of individual robot odometry estimates and distance measurements between robots used in the optimization are not sufficiently considered. In addition, previous researches were heavily influenced by the odometry accuracy that is estimated from individual robots. Consequently, long-term drift error caused by error accumulation is potentially inevitable. In this paper, we propose a novel visual-inertial-range-based multi-robot localization method, named SaWa-ML, which enables geometric structure-aware pose correction and weight adaptation-based robust multi-robot localization. Our contributions are twofold: (i) we leverage UWB sensor data, whose range error does not accumulate over time, to first estimate the relative positions between robots and then correct the positions of each robot, thus reducing long-term drift errors, (ii) we design adaptive weights for robot pose correction by considering the characteristics of the sensor data and visual-inertial odometry estimates. The proposed method has been validated in real-world experiments, showing a substantial performance increase compared with state-of-the-art algorithms. 

**Abstract (ZH)**: 多机器人定位是实施多机器人系统的关键任务。众多研究者提出了基于摄像机、IMU和UWB传感器的优化多机器人定位方法。然而，个别机器人的里程计估计特征和机器人之间距离测量在优化中的使用尚不够充分。此外，先前的研究受到从个别机器人估计的里程计精度的严重影响。因此，由误差累积引起的长期漂移误差可能是不可避免的。本文提出了一种新颖的基于视觉-惯性-范围的多机器人定位方法，名为SaWa-ML，该方法实现几何结构感知的位姿校正和基于权重自适应的鲁棒多机器人定位。我们的贡献主要有两点：（i）利用范围误差不随时间累积的UWB传感器数据，首先估计机器人之间的相对位置，然后纠正每个机器人的位置，从而减少长期漂移误差；（ii）通过考虑传感器数据特征和视觉-惯性里程计估计特征设计自适应权重。所提出的算法在真实世界实验中得到了验证，并展示出与先进算法相比显著的性能提升。 

---
# Safe Robotic Capsule Cleaning with Integrated Transpupillary and Intraocular Optical Coherence Tomography 

**Title (ZH)**: 安全的机器人capsule清洁，结合经瞳孔和眼内光学相干断层扫描 

**Authors**: Yu-Ting Lai, Yasamin Foroutani, Aya Barzelay, Tsu-Chin Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2507.13650)  

**Abstract**: Secondary cataract is one of the most common complications of vision loss due to the proliferation of residual lens materials that naturally grow on the lens capsule after cataract surgery. A potential treatment is capsule cleaning, a surgical procedure that requires enhanced visualization of the entire capsule and tool manipulation on the thin membrane. This article presents a robotic system capable of performing the capsule cleaning procedure by integrating a standard transpupillary and an intraocular optical coherence tomography probe on a surgical instrument for equatorial capsule visualization and real-time tool-to-tissue distance feedback. Using robot precision, the developed system enables complete capsule mapping in the pupillary and equatorial regions with in-situ calibration of refractive index and fiber offset, which are still current challenges in obtaining an accurate capsule model. To demonstrate effectiveness, the capsule mapping strategy was validated through five experimental trials on an eye phantom that showed reduced root-mean-square errors in the constructed capsule model, while the cleaning strategy was performed in three ex-vivo pig eyes without tissue damage. 

**Abstract (ZH)**: 二次囊膜混浊的机器人清洗系统研究：一种结合标准透过瞳孔和眼前段光学相干断层扫描探头的手术工具实现囊膜可视化和实时工具-组织距离反馈的方法 

---
# Improved particle swarm optimization algorithm: multi-target trajectory optimization for swarm drones 

**Title (ZH)**: 改进的粒子群优化算法：群无人机多目标轨迹优化 

**Authors**: Minze Li, Wei Zhao, Ran Chen, Mingqiang Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.13647)  

**Abstract**: Real-time trajectory planning for unmanned aerial vehicles (UAVs) in dynamic environments remains a key challenge due to high computational demands and the need for fast, adaptive responses. Traditional Particle Swarm Optimization (PSO) methods, while effective for offline planning, often struggle with premature convergence and latency in real-time scenarios. To overcome these limitations, we propose PE-PSO, an enhanced PSO-based online trajectory planner. The method introduces a persistent exploration mechanism to preserve swarm diversity and an entropy-based parameter adjustment strategy to dynamically adapt optimization behavior. UAV trajectories are modeled using B-spline curves, which ensure path smoothness while reducing optimization complexity. To extend this capability to UAV swarms, we develop a multi-agent framework that combines genetic algorithm (GA)-based task allocation with distributed PE-PSO, supporting scalable and coordinated trajectory generation. The distributed architecture allows for parallel computation and decentralized control, enabling effective cooperation among agents while maintaining real-time performance. Comprehensive simulations demonstrate that the proposed framework outperforms conventional PSO and other swarm-based planners across several metrics, including trajectory quality, energy efficiency, obstacle avoidance, and computation time. These results confirm the effectiveness and applicability of PE-PSO in real-time multi-UAV operations under complex environmental conditions. 

**Abstract (ZH)**: 实时无人机在动态环境下的轨迹规划仍然是一个关键挑战，由于高计算要求和需要快速适应的响应。传统的粒子群优化（PSO）方法虽然适用于离线规划，但在实时场景中常会出现过早收敛和延迟问题。为克服这些局限，我们提出了一种增强的基于PSO的在线轨迹规划方法PE-PSO。该方法引入了一种持久的探索机制以保持种群多样性，并采用基于熵的参数调整策略以动态适应优化行为。无人机轨迹使用B-样条曲线建模，确保路径平滑并降低优化复杂度。为了将此能力扩展到无人机群体，我们开发了一个基于多agent框架，结合了基于遗传算法（GA）的任务分配和分布式PE-PSO，支持可扩展和协调的轨迹生成。分布式架构允许并行计算和去中心化控制，使agent之间能够有效协作，同时保持实时性能。全面的仿真结果显示，所提出的框架在轨迹质量、能量效率、障碍物规避和计算时间等多个指标上优于传统PSO和其他基于群体的规划方法。这些结果证实了PE-PSO在复杂环境条件下的实时多无人机操作中的有效性和适用性。 

---
# Improving Low-Cost Teleoperation: Augmenting GELLO with Force 

**Title (ZH)**: 改进低成本远程操作：增强GELLO以加入力反馈 

**Authors**: Shivakanth Sujit, Luca Nunziante, Dan Ogawa Lillrank, Rousslan Fernand Julien Dossa, Kai Arulkumaran  

**Link**: [PDF](https://arxiv.org/pdf/2507.13602)  

**Abstract**: In this work we extend the low-cost GELLO teleoperation system, initially designed for joint position control, with additional force information. Our first extension is to implement force feedback, allowing users to feel resistance when interacting with the environment. Our second extension is to add force information into the data collection process and training of imitation learning models. We validate our additions by implementing these on a GELLO system with a Franka Panda arm as the follower robot, performing a user study, and comparing the performance of policies trained with and without force information on a range of simulated and real dexterous manipulation tasks. Qualitatively, users with robotics experience preferred our controller, and the addition of force inputs improved task success on the majority of tasks. 

**Abstract (ZH)**: 本文将低成本GELLO远程操作系统扩展为不仅支持关节位置控制，还提供额外的力信息。首先，我们实现了力反馈，使用户在与环境交互时能够感受到阻力。其次，我们将力信息纳入数据收集过程和模仿学习模型的训练中。我们通过将这些扩展应用于配备Franka Panda手臂的GELLO系统，并进行用户研究，比较了带有力信息和不带力信息训练的策略在一系列模拟和真实灵巧操作任务中的性能。定性研究结果显示，有机器人经验的用户更偏好我们的控制器，增加力输入在大多数任务中提高了任务成功率。 

---
# Hard-Stop Synthesis for Multi-DOF Compliant Mechanisms 

**Title (ZH)**: 多自由度 compliant 机制的硬停止合成 

**Authors**: Dean Chen, Armin Pomeroy, Brandon T. Peterson, Will Flanagan, He Kai Lim, Alexandra Stavrakis, Nelson F. SooHoo, Jonathan B. Hopkins, Tyler R. Clites  

**Link**: [PDF](https://arxiv.org/pdf/2507.13455)  

**Abstract**: Compliant mechanisms have significant potential in precision applications due to their ability to guide motion without contact. However, an inherent vulnerability to fatigue and mechanical failure has hindered the translation of compliant mechanisms to real-world applications. This is particularly challenging in service environments where loading is complex and uncertain, and the cost of failure is high. In such cases, mechanical hard stops are critical to prevent yielding and buckling. Conventional hard-stop designs, which rely on stacking single-DOF limits, must be overly restrictive in multi-DOF space to guarantee safety in the presence of unknown loads. In this study, we present a systematic design synthesis method to guarantee overload protection in compliant mechanisms by integrating coupled multi-DOF motion limits within a single pair of compact hard-stop surfaces. Specifically, we introduce a theoretical and practical framework for optimizing the contact surface geometry to maximize the mechanisms multi-DOF working space while still ensuring that the mechanism remains within its elastic regime. We apply this synthesis method to a case study of a caged-hinge mechanism for orthopaedic implants, and provide numerical and experimental validation that the derived design offers reliable protection against fatigue, yielding, and buckling. This work establishes a foundation for precision hard-stop design in compliant systems operating under uncertain loads, which is a crucial step toward enabling the application of compliant mechanisms in real-world systems. 

**Abstract (ZH)**: 具有多自由度耦合约束的紧凑型硬限位设计在柔顺机构中的负载保护研究 

---
# Fixed time convergence guarantees for Higher Order Control Barrier Functions 

**Title (ZH)**: 固定时间收敛保证的高阶控制屏障函数 

**Authors**: Janani S K, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2507.13888)  

**Abstract**: We present a novel method for designing higher-order Control Barrier Functions (CBFs) that guarantee convergence to a safe set within a user-specified finite. Traditional Higher Order CBFs (HOCBFs) ensure asymptotic safety but lack mechanisms for fixed-time convergence, which is critical in time-sensitive and safety-critical applications such as autonomous navigation. In contrast, our approach imposes a structured differential constraint using repeated roots in the characteristic polynomial, enabling closed-form polynomial solutions with exact convergence at a prescribed time. We derive conditions on the barrier function and its derivatives that ensure forward invariance and fixed-time reachability, and we provide an explicit formulation for second-order systems. Our method is evaluated on three robotic systems - a point-mass model, a unicycle, and a bicycle model and benchmarked against existing HOCBF approaches. Results demonstrate that our formulation reliably enforces convergence within the desired time, even when traditional methods fail. This work provides a tractable and robust framework for real-time control with provable finite-time safety guarantees. 

**Abstract (ZH)**: 我们提出了一种新的方法，用于设计高阶控制障碍函数（CBFs），以确保在用户指定的有限时间内收敛至安全集。传统高阶CBFs确保渐近安全，但在自主导航等时间敏感和安全关键应用中缺乏固定时间收敛机制。相比之下，我们的方法通过特征多项式的重复根施加结构化微分约束，使闭式多项式解在指定时间精确收敛。我们推导了确保前向不变性和固定时间可达性的障碍函数及其导数的条件，并为二阶系统提供了显式形式。我们的方法在三种机器人系统——质点模型、单轨车和自行车模型上进行了评估，并与现有高阶CBFs方法进行了比较。结果表明，在传统方法失败时，我们的公式可靠地在期望时间内实现收敛。本工作提供了一种具有可证明固定时间安全保证的实时控制实用且稳健的框架。 

---
# Safe and Performant Controller Synthesis using Gradient-based Model Predictive Control and Control Barrier Functions 

**Title (ZH)**: 基于梯度模型预测控制和控制障碍函数的安全高效控制器综合 

**Authors**: Aditya Singh, Aastha Mishra, Manan Tayal, Shishir Kolathaya, Pushpak Jagtap  

**Link**: [PDF](https://arxiv.org/pdf/2507.13872)  

**Abstract**: Ensuring both performance and safety is critical for autonomous systems operating in real-world environments. While safety filters such as Control Barrier Functions (CBFs) enforce constraints by modifying nominal controllers in real time, they can become overly conservative when the nominal policy lacks safety awareness. Conversely, solving State-Constrained Optimal Control Problems (SC-OCPs) via dynamic programming offers formal guarantees but is intractable in high-dimensional systems. In this work, we propose a novel two-stage framework that combines gradient-based Model Predictive Control (MPC) with CBF-based safety filtering for co-optimizing safety and performance. In the first stage, we relax safety constraints as penalties in the cost function, enabling fast optimization via gradient-based methods. This step improves scalability and avoids feasibility issues associated with hard constraints. In the second stage, we modify the resulting controller using a CBF-based Quadratic Program (CBF-QP), which enforces hard safety constraints with minimal deviation from the reference. Our approach yields controllers that are both performant and provably safe. We validate the proposed framework on two case studies, showcasing its ability to synthesize scalable, safe, and high-performance controllers for complex, high-dimensional autonomous systems. 

**Abstract (ZH)**: 确保自主系统在现实环境中的性能和安全至关重要。虽然通过实时修改常规控制器来施加约束的安全过滤器，如控制屏障函数（CBFs），可以在必要时增强安全性，但如果常规策略缺乏安全意识，可能会变得过于保守。相反，通过动态规划求解状态约束最优控制问题（SC-OCP）虽然可以提供正式保证，但在高维系统中却是不可计算的。在本工作中，我们提出了一种新颖的两阶段框架，结合梯度法模型预测控制（MPC）和基于CBF的安全过滤，以协同优化安全性和性能。在第一阶段，我们将安全约束作为惩罚项加入成本函数中，通过梯度法实现快速优化。这一步骤提高了可扩展性并避免了与硬约束相关的问题。在第二阶段，我们使用基于CBF的二次规划（CBF-QP）修改得出的控制器，以最小偏离参考的方式施加硬安全约束。该方法得到的控制器既具有高性能又可以验证其安全性。我们通过两个案例研究验证了所提出框架的有效性，展示了其为复杂高维自主系统合成可扩展、安全且高性能控制器的能力。 

---
