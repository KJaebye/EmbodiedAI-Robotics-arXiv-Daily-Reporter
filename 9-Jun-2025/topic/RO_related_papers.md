# From NLVO to NAO: Reactive Robot Navigation using Velocity and Acceleration Obstacles 

**Title (ZH)**: 从NLVO到NAO：基于速度和加速度障碍物的反应式机器人导航 

**Authors**: Asher Stern, Zvi Shiller  

**Link**: [PDF](https://arxiv.org/pdf/2506.06255)  

**Abstract**: This paper introduces a novel approach for robot navigation in challenging dynamic environments. The proposed method builds upon the concept of Velocity Obstacles (VO) that was later extended to Nonlinear Velocity Obstacles (NLVO) to account for obstacles moving along nonlinear trajectories. The NLVO is extended in this paper to Acceleration Obstacles (AO) and Nonlinear Acceleration Obstacles (NAO) that account for velocity and acceleration constraints. Multi-robot navigation is achieved by using the same avoidance algorithm by all robots. At each time step, the trajectories of all robots are predicted based on their current velocity and acceleration to allow the computation of their respective NLVO, AO and NAO.
The introduction of AO and NAO allows the generation of safe avoidance maneuvers that account for the robot dynamic constraints better than could be done with the NLVO alone. This paper demonstrates the use of AO and NAO for robot navigation in challenging environments. It is shown that using AO and NAO enables simultaneous real-time collision avoidance while accounting for robot kinematics and a direct consideration of its dynamic constraints. The presented approach enables reactive and efficient navigation, with potential application for autonomous vehicles operating in complex dynamic environments. 

**Abstract (ZH)**: 一种用于应对挑战性动态环境的机器人导航新方法：加速障碍与非线性加速障碍在多机器人导航中的应用 

---
# UAV-UGV Cooperative Trajectory Optimization and Task Allocation for Medical Rescue Tasks in Post-Disaster Environments 

**Title (ZH)**: 灾害后环境中无人机-地面机器人协同轨迹优化与任务分配 

**Authors**: Kaiyuan Chen, Wanpeng Zhao, Yongxi Liu, Yuanqing Xia, Wannian Liang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06136)  

**Abstract**: In post-disaster scenarios, rapid and efficient delivery of medical resources is critical and challenging due to severe damage to infrastructure. To provide an optimized solution, we propose a cooperative trajectory optimization and task allocation framework leveraging unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs). This study integrates a Genetic Algorithm (GA) for efficient task allocation among multiple UAVs and UGVs, and employs an informed-RRT* (Rapidly-exploring Random Tree Star) algorithm for collision-free trajectory generation. Further optimization of task sequencing and path efficiency is conducted using Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Simulation experiments conducted in a realistic post-disaster environment demonstrate that our proposed approach significantly improves the overall efficiency of medical rescue operations compared to traditional strategies, showing substantial reductions in total mission completion time and traveled distance. Additionally, the cooperative utilization of UAVs and UGVs effectively balances their complementary advantages, highlighting the system' s scalability and practicality for real-world deployment. 

**Abstract (ZH)**: 在灾后场景中，利用无人机和地面无人车进行高效协同路径优化与任务分配，以优化医疗资源快速精准配送的研究 

---
# Self driving algorithm for an active four wheel drive racecar 

**Title (ZH)**: 主动四轮驱动赛车的自动驾驶算法 

**Authors**: Gergely Bari, Laszlo Palkovics  

**Link**: [PDF](https://arxiv.org/pdf/2506.06077)  

**Abstract**: Controlling autonomous vehicles at their handling limits is a significant challenge, particularly for electric vehicles with active four wheel drive (A4WD) systems offering independent wheel torque control. While traditional Vehicle Dynamics Control (VDC) methods use complex physics-based models, this study explores Deep Reinforcement Learning (DRL) to develop a unified, high-performance controller. We employ the Proximal Policy Optimization (PPO) algorithm to train an agent for optimal lap times in a simulated racecar (TORCS) at the tire grip limit. Critically, the agent learns an end-to-end policy that directly maps vehicle states, like velocities, accelerations, and yaw rate, to a steering angle command and independent torque commands for each of the four wheels. This formulation bypasses conventional pedal inputs and explicit torque vectoring algorithms, allowing the agent to implicitly learn the A4WD control logic needed for maximizing performance and stability. Simulation results demonstrate the RL agent learns sophisticated strategies, dynamically optimizing wheel torque distribution corner-by-corner to enhance handling and mitigate the vehicle's inherent understeer. The learned behaviors mimic and, in aspects of grip utilization, potentially surpass traditional physics-based A4WD controllers while achieving competitive lap times. This research underscores DRL's potential to create adaptive control systems for complex vehicle dynamics, suggesting RL is a potent alternative for advancing autonomous driving in demanding, grip-limited scenarios for racing and road safety. 

**Abstract (ZH)**: 基于深度强化学习的四轮独立驱动自主车辆handling极限控制研究 

---
# Enhanced Trust Region Sequential Convex Optimization for Multi-Drone Thermal Screening Trajectory Planning in Urban Environments 

**Title (ZH)**: 增强信任域序列凸优化在城市环境中的多无人机热筛查轨迹规划 

**Authors**: Kaiyuan Chen, Zhengjie Hu, Shaolin Zhang, Yuanqing Xia, Wannian Liang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06012)  

**Abstract**: The rapid detection of abnormal body temperatures in urban populations is essential for managing public health risks, especially during outbreaks of infectious diseases. Multi-drone thermal screening systems offer promising solutions for fast, large-scale, and non-intrusive human temperature monitoring. However, trajectory planning for multiple drones in complex urban environments poses significant challenges, including collision avoidance, coverage efficiency, and constrained flight environments. In this study, we propose an enhanced trust region sequential convex optimization (TR-SCO) algorithm for optimal trajectory planning of multiple drones performing thermal screening tasks. Our improved algorithm integrates a refined convex optimization formulation within a trust region framework, effectively balancing trajectory smoothness, obstacle avoidance, altitude constraints, and maximum screening coverage. Simulation results demonstrate that our approach significantly improves trajectory optimality and computational efficiency compared to conventional convex optimization methods. This research provides critical insights and practical contributions toward deploying efficient multi-drone systems for real-time thermal screening in urban areas. For reader who are interested in our research, we release our source code at this https URL. 

**Abstract (ZH)**: 城市人群中快速检测异常体温对于管理公共卫生风险至关重要，尤其是在传染病暴发期间。多无人机热筛查系统为快速、大规模和非侵入性的人体温度监控提供了有前景的解决方案。然而，在复杂的城市环境中进行多无人机轨迹规划面临着巨大的挑战，包括避碰、覆盖率和飞行限制。在本研究中，我们提出了一种增强的信任区域序列凸优化（TR-SCO）算法，用于多无人机热筛查任务的最优轨迹规划。我们改进的算法结合了细化的凸优化公式于信任区域框架内，有效地平衡了轨迹平滑性、障碍物回避、高度限制和最大筛查覆盖率。仿真结果表明，与传统的凸优化方法相比，我们的方法显著提高了轨迹优化效果和计算效率。本研究为在城市区域能够进行实时热筛查的高效多无人机系统部署提供了关键见解和实用贡献。对我们的研究感兴趣的读者可以通过此链接获取我们开源的代码：此链接。 

---
# Optimal Robotic Velcro Peeling with Force Feedback 

**Title (ZH)**: 带有力反馈的最优机器人钩快剥离技术 

**Authors**: Jiacheng Yuan, Changhyun Choi, Volkan Isler  

**Link**: [PDF](https://arxiv.org/pdf/2506.05812)  

**Abstract**: We study the problem of peeling a Velcro strap from a surface using a robotic manipulator. The surface geometry is arbitrary and unknown. The robot has access to only the force feedback and its end-effector position. This problem is challenging due to the partial observability of the environment and the incompleteness of the sensor feedback. To solve it, we first model the system with simple analytic state and action models based on quasi-static dynamics assumptions. We then study the fully-observable case where the state of both the Velcro and the robot are given. For this case, we obtain the optimal solution in closed-form which minimizes the total energy cost. Next, for the partially-observable case, we design a state estimator which estimates the underlying state using only force and position feedback. Then, we present a heuristics-based controller that balances exploratory and exploitative behaviors in order to peel the velcro efficiently. Finally, we evaluate our proposed method in environments with complex geometric uncertainties and sensor noises, achieving 100% success rate with less than 80% increase in energy cost compared to the optimal solution when the environment is fully-observable, outperforming the baselines by a large margin. 

**Abstract (ZH)**: 使用机器人 manipulator 剥离 Velcro 条带的问题：从几何未知的表面剥离，基于部分可观测性和传感器反馈不完全性分析与解决 

---
# A Soft Robotic Module with Pneumatic Actuation and Enhanced Controllability Using a Shape Memory Alloy Wire 

**Title (ZH)**: 一种使用形状记忆合金丝增强可控性的气动软机器人模块 

**Authors**: Mohammadnavid Golchin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05741)  

**Abstract**: In this paper, a compressed air-actuated soft robotic module was developed by incorporating a shape memory alloy (SMA) wire into its structure to achieve the desired bending angle with greater precision. First, a fiber-reinforced bending module with a strain-limiting layer made of polypropylene was fabricated. The SMA wire was then placed in a silicon matrix, which was used as a new strain-limiting layer. A simple closed-loop control algorithm was used to regulate the bending angle of the soft robot within its workspace. A camera was utilized to measure the angular changes in the vertical plane. Different angles, ranging from 0 to 65 degrees, were covered to evaluate the performance of the module and the bending angle control algorithm. The experimental tests demonstrate that using the SMA wire results in more precise control of bending in the vertical plane. In addition, it is possible to bend more with less working pressure. The error range was reduced from an average of 5 degrees to 2 degrees, and the rise time was reduced from an average of 19 seconds to 3 seconds. 

**Abstract (ZH)**: 一种集成形状记忆合金丝的压缩空气驱动柔性机器人模块及其精确弯曲角度控制研究 

---
# Advancement and Field Evaluation of a Dual-arm Apple Harvesting Robot 

**Title (ZH)**: 双臂苹果采摘机器人技术进展与田间评估 

**Authors**: Keyi Zhu, Kyle Lammers, Kaixiang Zhang, Chaaran Arunachalam, Siddhartha Bhattacharya, Jiajia Li, Renfu Lu, Zhaojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05714)  

**Abstract**: Apples are among the most widely consumed fruits worldwide. Currently, apple harvesting fully relies on manual labor, which is costly, drudging, and hazardous to workers. Hence, robotic harvesting has attracted increasing attention in recent years. However, existing systems still fall short in terms of performance, effectiveness, and reliability for complex orchard environments. In this work, we present the development and evaluation of a dual-arm harvesting robot. The system integrates a ToF camera, two 4DOF robotic arms, a centralized vacuum system, and a post-harvest handling module. During harvesting, suction force is dynamically assigned to either arm via the vacuum system, enabling efficient apple detachment while reducing power consumption and noise. Compared to our previous design, we incorporated a platform movement mechanism that enables both in-out and up-down adjustments, enhancing the robot's dexterity and adaptability to varying canopy structures. On the algorithmic side, we developed a robust apple localization pipeline that combines a foundation-model-based detector, segmentation, and clustering-based depth estimation, which improves performance in orchards. Additionally, pressure sensors were integrated into the system, and a novel dual-arm coordination strategy was introduced to respond to harvest failures based on sensor feedback, further improving picking efficiency. Field demos were conducted in two commercial orchards in MI, USA, with different canopy structures. The system achieved success rates of 0.807 and 0.797, with an average picking cycle time of 5.97s. The proposed strategy reduced harvest time by 28% compared to a single-arm baseline. The dual-arm harvesting robot enhances the reliability and efficiency of apple picking. With further advancements, the system holds strong potential for autonomous operation and commercialization for the apple industry. 

**Abstract (ZH)**: 双臂苹果采摘机器人开发与评价 

---
# Towards Autonomous In-situ Soil Sampling and Mapping in Large-Scale Agricultural Environments 

**Title (ZH)**: 面向大规模农业环境中的自主就位土壤采样与制图 

**Authors**: Thien Hoang Nguyen, Erik Muller, Michael Rubin, Xiaofei Wang, Fiorella Sibona, Salah Sukkarieh  

**Link**: [PDF](https://arxiv.org/pdf/2506.05653)  

**Abstract**: Traditional soil sampling and analysis methods are labor-intensive, time-consuming, and limited in spatial resolution, making them unsuitable for large-scale precision agriculture. To address these limitations, we present a robotic solution for real-time sampling, analysis and mapping of key soil properties. Our system consists of two main sub-systems: a Sample Acquisition System (SAS) for precise, automated in-field soil sampling; and a Sample Analysis Lab (Lab) for real-time soil property analysis. The system's performance was validated through extensive field trials at a large-scale Australian farm. Experimental results show that the SAS can consistently acquire soil samples with a mass of 50g at a depth of 200mm, while the Lab can process each sample within 10 minutes to accurately measure pH and macronutrients. These results demonstrate the potential of the system to provide farmers with timely, data-driven insights for more efficient and sustainable soil management and fertilizer application. 

**Abstract (ZH)**: 传统的土壤采样与分析方法劳动密集、耗时且空间分辨率有限，不适合大规模精准农业的应用。为解决这些局限性，我们提出了一种机器人解决方案，用于实时采集、分析和制图关键土壤属性。该系统包括两个主要子系统：一个精确的自动化现场土壤采样系统（SAS）；和一个实时土壤属性分析实验室（Lab）。该系统的性能通过在澳大利亚大型农场进行的广泛田间试验得到了验证。实验结果表明，SAS可以一致地在200mm深度处采集50g的土壤样本，而Lab可以在10分钟内处理每个样本以准确测量pH值和宏量元素。这些结果展示了该系统为农民提供及时的数据驱动见解以实现更高效和可持续的土壤管理和肥料施用的潜力。 

---
# Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots 

**Title (ZH)**: 学习恢复：基于轮腿协调的跌倒机器人动态奖励塑形 

**Authors**: Boyuan Deng, Luca Rossini, Jin Wang, Weijie Wang, Nikolaos Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.05516)  

**Abstract**: Adaptive recovery from fall incidents are essential skills for the practical deployment of wheeled-legged robots, which uniquely combine the agility of legs with the speed of wheels for rapid recovery. However, traditional methods relying on preplanned recovery motions, simplified dynamics or sparse rewards often fail to produce robust recovery policies. This paper presents a learning-based framework integrating Episode-based Dynamic Reward Shaping and curriculum learning, which dynamically balances exploration of diverse recovery maneuvers with precise posture refinement. An asymmetric actor-critic architecture accelerates training by leveraging privileged information in simulation, while noise-injected observations enhance robustness against uncertainties. We further demonstrate that synergistic wheel-leg coordination reduces joint torque consumption by 15.8% and 26.2% and improves stabilization through energy transfer mechanisms. Extensive evaluations on two distinct quadruped platforms achieve recovery success rates up to 99.1% and 97.8% without platform-specific tuning. The supplementary material is available at this https URL 

**Abstract (ZH)**: 基于episode动态奖励塑形和课程学习的适应性跌倒恢复对于轮腿机器人实用部署是 essential 技能，这种机器人独特地结合了腿的灵活性和轮的速度以实现快速恢复。然而，依赖于预先规划的恢复动作、简化的动力学模型或稀疏奖励的传统方法往往无法产生 robust 的恢复策略。本文介绍了一种基于学习的框架，结合了基于episode的动态奖励塑形和课程学习，动态平衡多样恢复动作的探索与精确姿态优化。不对称的Actor-Critic架构通过利用仿真中的优先信息加速训练，同时注入噪声的观测增强其对不确定性的稳健性。进一步的实验表明，协同的轮-腿协调减少了关节扭矩消耗15.8%和26.2%，并通过能量转移机制提高了稳定性能。在两个不同的四足平台上的广泛评估表明，恢复成功率分别达到了99.1%和97.8%，无需针对特定平台进行调整。更多补充材料请参见此链接：this https URL。 

---
# Trajectory Optimization for UAV-Based Medical Delivery with Temporal Logic Constraints and Convex Feasible Set Collision Avoidance 

**Title (ZH)**: 基于时间逻辑约束和凸可行集碰撞 avoidance 的无人机医疗配送轨迹优化 

**Authors**: Kaiyuan Chen, Yuhan Suo, Shaowei Cui, Yuanqing Xia, Wannian Liang, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06038)  

**Abstract**: This paper addresses the problem of trajectory optimization for unmanned aerial vehicles (UAVs) performing time-sensitive medical deliveries in urban environments. Specifically, we consider a single UAV with 3 degree-of-freedom dynamics tasked with delivering blood packages to multiple hospitals, each with a predefined time window and priority. Mission objectives are encoded using Signal Temporal Logic (STL), enabling the formal specification of spatial-temporal constraints. To ensure safety, city buildings are modeled as 3D convex obstacles, and obstacle avoidance is handled through a Convex Feasible Set (CFS) method. The entire planning problem-combining UAV dynamics, STL satisfaction, and collision avoidance-is formulated as a convex optimization problem that ensures tractability and can be solved efficiently using standard convex programming techniques. Simulation results demonstrate that the proposed method generates dynamically feasible, collision-free trajectories that satisfy temporal mission goals, providing a scalable and reliable approach for autonomous UAV-based medical logistics. 

**Abstract (ZH)**: 本文探讨了在城市环境中执行时间敏感医疗配送任务的无人驾驶飞行器（UAV）轨迹优化问题。具体而言，我们考虑了一架具有3自由度动力学的单个UAV，其任务是向多个具有预定义时间和优先级的医院配送血包。使用信号时态逻辑（STL）编码任务目标，使得能够正式规定空间-时间约束。为了确保安全，将城市建筑物建模为3D凸障碍，并通过凸可行集（CFS）方法处理障碍物避免问题。整个规划问题结合UAV动力学、STL满足性和碰撞避免，被形式化为一个凸优化问题，该问题具有可处理性，并且可以使用标准凸规划技术高效地求解。仿真实验结果表明，所提出的方法能够生成动态可行、无碰撞的轨迹，满足时间任务目标，提供了一种可扩展且可靠的基于自主UAV的医疗物流方法。 

---
# Robust sensor fusion against on-vehicle sensor staleness 

**Title (ZH)**: 针对车载传感器陈旧性的鲁棒传感器融合 

**Authors**: Meng Fan, Yifan Zuo, Patrick Blaes, Harley Montgomery, Subhasis Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.05780)  

**Abstract**: Sensor fusion is crucial for a performant and robust Perception system in autonomous vehicles, but sensor staleness, where data from different sensors arrives with varying delays, poses significant challenges. Temporal misalignment between sensor modalities leads to inconsistent object state estimates, severely degrading the quality of trajectory predictions that are critical for safety. We present a novel and model-agnostic approach to address this problem via (1) a per-point timestamp offset feature (for LiDAR and radar both relative to camera) that enables fine-grained temporal awareness in sensor fusion, and (2) a data augmentation strategy that simulates realistic sensor staleness patterns observed in deployed vehicles. Our method is integrated into a perspective-view detection model that consumes sensor data from multiple LiDARs, radars and cameras. We demonstrate that while a conventional model shows significant regressions when one sensor modality is stale, our approach reaches consistently good performance across both synchronized and stale conditions. 

**Abstract (ZH)**: 传感器融合对于自主车辆高性能和稳健的感知系统至关重要，但传感器数据延迟不一致带来的挑战显著。不同传感器数据的时间错位导致目标状态估计不一致，严重影响了对安全至关重要的轨迹预测质量。我们提出了一种模型无关的新方法，通过（1）一种针对点的时间戳偏移特征（LiDAR和雷达相对于摄像头），实现传感器融合中的精细时间感知，以及（2）一种模拟部署车辆中观察到的传感器延迟模式的数据增强策略。该方法集成到一个多传感器视图检测模型中，该模型消耗来自多个LiDAR、雷达和摄像头的数据。我们证明，尽管传统模型在某一种传感器模式延迟时表现显著下降，但我们的方法在同步和延迟条件下都能保持一致的良好性能。 

---
# You Only Estimate Once: Unified, One-stage, Real-Time Category-level Articulated Object 6D Pose Estimation for Robotic Grasping 

**Title (ZH)**: 只需一次估计：统一的一阶段实时类别级 articulated 对象 6D 姿态估计用于机器人抓取 

**Authors**: Jingshun Huang, Haitao Lin, Tianyu Wang, Yanwei Fu, Yu-Gang Jiang, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2506.05719)  

**Abstract**: This paper addresses the problem of category-level pose estimation for articulated objects in robotic manipulation tasks. Recent works have shown promising results in estimating part pose and size at the category level. However, these approaches primarily follow a complex multi-stage pipeline that first segments part instances in the point cloud and then estimates the Normalized Part Coordinate Space (NPCS) representation for 6D poses. These approaches suffer from high computational costs and low performance in real-time robotic tasks. To address these limitations, we propose YOEO, a single-stage method that simultaneously outputs instance segmentation and NPCS representations in an end-to-end manner. We use a unified network to generate point-wise semantic labels and centroid offsets, allowing points from the same part instance to vote for the same centroid. We further utilize a clustering algorithm to distinguish points based on their estimated centroid distances. Finally, we first separate the NPCS region of each instance. Then, we align the separated regions with the real point cloud to recover the final pose and size. Experimental results on the GAPart dataset demonstrate the pose estimation capabilities of our proposed single-shot method. We also deploy our synthetically-trained model in a real-world setting, providing real-time visual feedback at 200Hz, enabling a physical Kinova robot to interact with unseen articulated objects. This showcases the utility and effectiveness of our proposed method. 

**Abstract (ZH)**: 本文解决了机器人操作任务中articulated对象类别级别姿态估计的问题。最近的研究在估计部件姿态和大小方面取得了令人鼓舞的结果。然而，这些方法主要遵循一个复杂多阶段的流程，首先对点云中的部件实例进行分割，然后估计归一化部件坐标空间(NPCS)表示以获取6D姿态。这些方法在实时机器人任务中面临高计算成本和低性能的限制。为了解决这些问题，我们提出了YOEO，一种单阶段方法，在端到端的方式下同时输出实例分割和NPCS表示。我们使用一个统一网络生成点级语义标签和质心偏移量，使得同一部件实例的点可以投票支持同一个质心。我们进一步利用聚类算法根据点的估计质心距离对点进行区分。最后，我们首先分离每个实例的NPCS区域，然后将分离的区域与真实点云对齐以恢复最终的姿态和尺寸。在GAPart数据集上的实验结果展示了我们提出的单次检测方法的姿态估计能力。我们还将合成训练的模型部署到实际应用场景中，以200Hz的速度提供实时视觉反馈，使物理Kinova机器人能够与未见过的articulated对象进行互动。这展示了我们提出方法的实用性和有效性。 

---
# A Modular Haptic Display with Reconfigurable Signals for Personalized Information Transfer 

**Title (ZH)**: 一种可重构信号的模块化触觉显示装置，实现个性化信息传输 

**Authors**: Antonio Alvarez Valdivia, Benjamin A. Christie, Dylan P. Losey, Laura H. Blumenschein  

**Link**: [PDF](https://arxiv.org/pdf/2506.05648)  

**Abstract**: We present a customizable soft haptic system that integrates modular hardware with an information-theoretic algorithm to personalize feedback for different users and tasks. Our platform features modular, multi-degree-of-freedom pneumatic displays, where different signal types, such as pressure, frequency, and contact area, can be activated or combined using fluidic logic circuits. These circuits simplify control by reducing reliance on specialized electronics and enabling coordinated actuation of multiple haptic elements through a compact set of inputs. Our approach allows rapid reconfiguration of haptic signal rendering through hardware-level logic switching without rewriting code. Personalization of the haptic interface is achieved through the combination of modular hardware and software-driven signal selection. To determine which display configurations will be most effective, we model haptic communication as a signal transmission problem, where an agent must convey latent information to the user. We formulate the optimization problem to identify the haptic hardware setup that maximizes the information transfer between the intended message and the user's interpretation, accounting for individual differences in sensitivity, preferences, and perceptual salience. We evaluate this framework through user studies where participants interact with reconfigurable displays under different signal combinations. Our findings support the role of modularity and personalization in creating multimodal haptic interfaces and advance the development of reconfigurable systems that adapt with users in dynamic human-machine interaction contexts. 

**Abstract (ZH)**: 可定制的软触觉系统：基于模块化硬件和信息论算法的个性化反馈集成平台 

---
# End-to-End Framework for Robot Lawnmower Coverage Path Planning using Cellular Decomposition 

**Title (ZH)**: 基于细胞分解的自主草坪修剪机器人全覆盖路径规划端到端框架 

**Authors**: Nikunj Shah, Utsav Dey, Kenji Nishimiya  

**Link**: [PDF](https://arxiv.org/pdf/2506.06028)  

**Abstract**: Efficient Coverage Path Planning (CPP) is necessary for autonomous robotic lawnmowers to effectively navigate and maintain lawns with diverse and irregular shapes. This paper introduces a comprehensive end-to-end pipeline for CPP, designed to convert user-defined boundaries on an aerial map into optimized coverage paths seamlessly. The pipeline includes user input extraction, coordinate transformation, area decomposition and path generation using our novel AdaptiveDecompositionCPP algorithm, preview and customization through an interactive coverage path visualizer, and conversion to actionable GPS waypoints. The AdaptiveDecompositionCPP algorithm combines cellular decomposition with an adaptive merging strategy to reduce non-mowing travel thereby enhancing operational efficiency. Experimental evaluations, encompassing both simulations and real-world lawnmower tests, demonstrate the effectiveness of the framework in coverage completeness and mowing efficiency. 

**Abstract (ZH)**: 高效覆盖路径规划（CPP）对于自主 robotic 前后驱动割草机有效导航和维护各种不规则形状的草坪是必要的。本文介绍了一套完整的端到端 CPP 管道，旨在将用户在空中地图上定义的边界无缝转换为优化的覆盖路径。该管道包含用户输入提取、坐标转换、区域分解和路径生成（使用我们新颖的 AdaptiveDecompositionCPP 算法）、通过交互式的覆盖路径可视化器进行预览和自定义，以及转换为可操作的GPS航点。AdaptiveDecompositionCPP 算法结合了细胞分解与自适应合并策略，以减少非割草行进距离从而提高操作效率。实验评估，包括模拟和实际草坪割草机测试，展示了该框架在覆盖完整性和割草效率方面的有效性。 

---
