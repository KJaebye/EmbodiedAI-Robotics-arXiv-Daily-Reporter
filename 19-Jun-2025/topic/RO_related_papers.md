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
# Comparison of Innovative Strategies for the Coverage Problem: Path Planning, Search Optimization, and Applications in Underwater Robotics 

**Title (ZH)**: 创新策略比较研究：覆盖问题中的路径规划、搜索优化及其在水下机器人中的应用 

**Authors**: Ahmed Ibrahim, Francisco F. C. Rego, Éric Busvelle  

**Link**: [PDF](https://arxiv.org/pdf/2506.15376)  

**Abstract**: In many applications, including underwater robotics, the coverage problem requires an autonomous vehicle to systematically explore a defined area while minimizing redundancy and avoiding obstacles. This paper investigates coverage path planning strategies to enhance the efficiency of underwater gliders, particularly in maximizing the probability of detecting a radioactive source while ensuring safe navigation.
We evaluate three path-planning approaches: the Traveling Salesman Problem (TSP), Minimum Spanning Tree (MST), and Optimal Control Problem (OCP). Simulations were conducted in MATLAB, comparing processing time, uncovered areas, path length, and traversal time. Results indicate that OCP is preferable when traversal time is constrained, although it incurs significantly higher computational costs. Conversely, MST-based approaches provide faster but less optimal solutions. These findings offer insights into selecting appropriate algorithms based on mission priorities, balancing efficiency and computational feasibility. 

**Abstract (ZH)**: 水下机器人应用中，覆盖问题需要自主车辆系统地探索定义区域，同时减少冗余并避开障碍物。本文研究覆盖路径规划策略以提高水下滑翔器的效率，特别是在确保安全导航的前提下最大化检测放射性源的概率。 

---
# 3D Vision-tactile Reconstruction from Infrared and Visible Images for Robotic Fine-grained Tactile Perception 

**Title (ZH)**: 基于红外和可见光图像的3D视觉-触觉重建及其在机器人精细触觉感知中的应用 

**Authors**: Yuankai Lin, Xiaofan Lu, Jiahui Chen, Hua Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15087)  

**Abstract**: To achieve human-like haptic perception in anthropomorphic grippers, the compliant sensing surfaces of vision tactile sensor (VTS) must evolve from conventional planar configurations to biomimetically curved topographies with continuous surface gradients. However, planar VTSs have challenges when extended to curved surfaces, including insufficient lighting of surfaces, blurring in reconstruction, and complex spatial boundary conditions for surface structures. With an end goal of constructing a human-like fingertip, our research (i) develops GelSplitter3D by expanding imaging channels with a prism and a near-infrared (NIR) camera, (ii) proposes a photometric stereo neural network with a CAD-based normal ground truth generation method to calibrate tactile geometry, and (iii) devises a normal integration method with boundary constraints of depth prior information to correcting the cumulative error of surface integrals. We demonstrate better tactile sensing performance, a 40$\%$ improvement in normal estimation accuracy, and the benefits of sensor shapes in grasping and manipulation tasks. 

**Abstract (ZH)**: 实现类人触觉感知的仿生触觉传感器柔软感知表面从传统平面配置进化到生物学启发的曲面拓扑结构并具有连续表面梯度是关键。然而，平面触觉传感器扩展到曲面时面临的挑战包括表面照明不足、重构模糊以及复杂的空间边界条件。为构建类人指尖，我们的研究开发了GelSplitter3D，通过棱镜和近红外相机扩展成像通道，提出了一种基于CAD的法线地面真实值生成方法的光度立体神经网络来校准触觉几何，并设计了一种结合深度先验信息的空间边界约束法线积分修正方法，以提高表面积分累计误差的准确性。 we demonstrate更好的触觉感知性能，法线估算精度提高40%，并展示了传感器形状在抓取和操作任务中的优势。 

---
# Assigning Multi-Robot Tasks to Multitasking Robots 

**Title (ZH)**: 多任务机器人分配多机器人任务 

**Authors**: Winston Smith, Andrew Boateng, Taha Shaheen, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15032)  

**Abstract**: One simplifying assumption in existing and well-performing task allocation methods is that the robots are single-tasking: each robot operates on a single task at any given time. While this assumption is harmless to make in some situations, it can be inefficient or even infeasible in others. In this paper, we consider assigning multi-robot tasks to multitasking robots. The key contribution is a novel task allocation framework that incorporates the consideration of physical constraints introduced by multitasking. This is in contrast to the existing work where such constraints are largely ignored. After formulating the problem, we propose a compilation to weighted MAX-SAT, which allows us to leverage existing solvers for a solution. A more efficient greedy heuristic is then introduced. For evaluation, we first compare our methods with a modern baseline that is efficient for single-tasking robots to validate the benefits of multitasking in synthetic domains. Then, using a site-clearing scenario in simulation, we further illustrate the complex task interaction considered by the multitasking robots in our approach to demonstrate its performance. Finally, we demonstrate a physical experiment to show how multitasking enabled by our approach can benefit task efficiency in a realistic setting. 

**Abstract (ZH)**: 多任务机器人下的任务分配：考虑物理约束的新框架及其应用 

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
# Model Predictive Path-Following Control for a Quadrotor 

**Title (ZH)**: 四旋翼飞行器的模型预测路径跟踪控制 

**Authors**: David Leprich, Mario Rosenfelder, Mario Hermle, Jingshan Chen, Peter Eberhard  

**Link**: [PDF](https://arxiv.org/pdf/2506.15447)  

**Abstract**: Automating drone-assisted processes is a complex task. Many solutions rely on trajectory generation and tracking, whereas in contrast, path-following control is a particularly promising approach, offering an intuitive and natural approach to automate tasks for drones and other vehicles. While different solutions to the path-following problem have been proposed, most of them lack the capability to explicitly handle state and input constraints, are formulated in a conservative two-stage approach, or are only applicable to linear systems. To address these challenges, the paper is built upon a Model Predictive Control-based path-following framework and extends its application to the Crazyflie quadrotor, which is investigated in hardware experiments. A cascaded control structure including an underlying attitude controller is included in the Model Predictive Path-Following Control formulation to meet the challenging real-time demands of quadrotor control. The effectiveness of the proposed method is demonstrated through real-world experiments, representing, to the best of the authors' knowledge, a novel application of this MPC-based path-following approach to the quadrotor. Additionally, as an extension to the original method, to allow for deviations of the path in cases where the precise following of the path might be overly restrictive, a corridor path-following approach is presented. 

**Abstract (ZH)**: 基于模型预测控制的路径跟随方法在 Crazyflie 四旋翼无人机上的应用及扩展研究 

---
# Advances in Compliance Detection: Novel Models Using Vision-Based Tactile Sensors 

**Title (ZH)**: 基于视觉触觉传感器的合规性检测进展 

**Authors**: Ziteng Li, Malte Kuhlmann, Ilana Nisky, Nicolás Navarro-Guerrero  

**Link**: [PDF](https://arxiv.org/pdf/2506.14980)  

**Abstract**: Compliance is a critical parameter for describing objects in engineering, agriculture, and biomedical applications. Traditional compliance detection methods are limited by their lack of portability and scalability, rely on specialized, often expensive equipment, and are unsuitable for robotic applications. Moreover, existing neural network-based approaches using vision-based tactile sensors still suffer from insufficient prediction accuracy. In this paper, we propose two models based on Long-term Recurrent Convolutional Networks (LRCNs) and Transformer architectures that leverage RGB tactile images and other information captured by the vision-based sensor GelSight to predict compliance metrics accurately. We validate the performance of these models using multiple metrics and demonstrate their effectiveness in accurately estimating compliance. The proposed models exhibit significant performance improvement over the baseline. Additionally, we investigated the correlation between sensor compliance and object compliance estimation, which revealed that objects that are harder than the sensor are more challenging to estimate. 

**Abstract (ZH)**: 合规性是描述工程、农业和生物医学应用中物体的重要参数。传统的合规性检测方法受限于其缺乏便携性和可扩展性，依赖于专门且通常昂贵的设备，并不适合用于机器人应用。此外，现有基于视觉触觉传感器的神经网络方法仍然存在预测精度不足的问题。本文提出两种基于长期递归卷积网络（LRCNs）和 Transformer 架构的模型，利用 GelSight 视觉触觉传感器捕获的 RGB 触觉图像和其他信息，准确预测合规性指标。我们使用多种评估指标验证了这些模型的性能，并展示了它们在准确估计合规性方面的有效性。所提出的模型在基线模型上表现出显著的性能提升。此外，我们研究了传感器合规性与物体合规性估计之间的相关性，结果显示比传感器更硬的物体更难估计。 

---
