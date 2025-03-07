# ExoNav II: Design of a Robotic Tool with Follow-the-Leader Motion Capability for Lateral and Ventral Spinal Cord Stimulation (SCS) 

**Title (ZH)**: ExoNav II：具有跟随领导者运动能力的外骨骼导航工具设计，用于侧方和腹侧脊髓电刺激（SCS） 

**Authors**: Behnam Moradkhani, Pejman Kheradmand, Harshith Jella, Joseph Klein, Ajmal Zemmar, Yash Chitalia  

**Link**: [PDF](https://arxiv.org/pdf/2503.04603)  

**Abstract**: Spinal cord stimulation (SCS) electrodes are traditionally placed in the dorsal epidural space to stimulate the dorsal column fibers for pain therapy. Recently, SCS has gained attention in restoring gait. However, the motor fibers triggering locomotion are located in the ventral and lateral spinal cord. Currently, SCS electrodes are steered manually, making it difficult to navigate them to the lateral and ventral motor fibers in the spinal cord. In this work, we propose a helically micro-machined continuum robot that can bend in a helical shape when subjected to actuation tendon forces. Using a stiff outer tube and adding translational and rotational degrees of freedom, this helical continuum robot can perform follow-the-leader (FTL) motion. We propose a kinematic model to relate tendon stroke and geometric parameters of the robot's helical shape to its acquired trajectory and end-effector position. We evaluate the proposed kinematic model and the robot's FTL motion capability experimentally. The stroke-based method, which links tendon stroke values to the robot's shape, showed inaccuracies with a 19.84 mm deviation and an RMSE of 14.42 mm for 63.6 mm of robot's length bending. The position-based method, using kinematic equations to map joint space to task space, performed better with a 10.54 mm deviation and an RMSE of 8.04 mm. Follow-the-leader experiments showed deviations of 11.24 mm and 7.32 mm, with RMSE values of 8.67 mm and 5.18 mm for the stroke-based and position-based methods, respectively. Furthermore, end-effector trajectories in two FTL motion trials are compared to confirm the robot's repeatable behavior. Finally, we demonstrate the robot's operation on a 3D-printed spinal cord phantom model. 

**Abstract (ZH)**: 基于螺线微加工连续体机器人的脊髓刺激电极导航方法研究 

---
# DogLegs: Robust Proprioceptive State Estimation for Legged Robots Using Multiple Leg-Mounted IMUs 

**Title (ZH)**: DogLegs：使用多个腿式IMU的腿部机器人 proprioceptive 状态估计的鲁棒方法 

**Authors**: Yibin Wu, Jian Kuang, Shahram Khorshidi, Xiaoji Niu, Lasse Klingbeil, Maren Bennewitz, Heiner Kuhlmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.04580)  

**Abstract**: Robust and accurate proprioceptive state estimation of the main body is crucial for legged robots to execute tasks in extreme environments where exteroceptive sensors, such as LiDARs and cameras may become unreliable. In this paper, we propose DogLegs, a state estimation system for legged robots that fuses the measurements from a body-mounted inertial measurement unit (Body-IMU), joint encoders, and multiple leg-mounted IMUs (Leg-IMU) using an extended Kalman filter (EKF). The filter system contains the error states of all IMU frames. The Leg-IMUs are used to detect foot contact, thereby providing zero velocity measurements to update the state of the Leg-IMU frames. Additionally, we compute the relative position constraints between the Body-IMU and Leg-IMUs by the leg kinematics and use them to update the main body state and reduce the error drift of the individual IMU frames. Field experimental results have shown that our proposed system can achieve better state estimation accuracy compared to the traditional leg odometry method (using only Body-IMU and joint encoders) across different terrains. We make our datasets publicly available to benefit the research community. 

**Abstract (ZH)**: 腿部机器人在极端环境中的稳健且准确的本体状态估计对于执行任务至关重要，其中外部传感器如激光雷达和摄像头可能变得不可靠。本文提出DogLegs，一种使用扩展卡尔曼滤波器融合装在身体上的惯性测量单元（Body-IMU）、关节编码器和多个装在腿部的惯性测量单元（Leg-IMU）的状态估计系统。滤波器系统包含了所有惯性测量单元坐标系的误差状态。腿部惯性测量单元用于检测足部接触，从而提供零速度测量以更新腿部惯性测量单元坐标系的状态。此外，我们通过腿部运动学计算身体惯性测量单元和腿部惯性测量单元之间的相对位置约束，并使用它们来更新主体状态并减少单个惯性测量单元坐标系的误差漂移。实地实验结果表明，与仅使用身体惯性测量单元和关节编码器的传统腿部里程计方法相比，本提出系统在不同地形上可以获得更好的状态估计精度。我们公开了我们的数据集以造福研究社区。 

---
# Occlusion-Aware Consistent Model Predictive Control for Robot Navigation in Occluded Obstacle-Dense Environments 

**Title (ZH)**: 考虑遮挡的一致模型预测控制在稠密障碍物遮挡环境中的机器人导航 

**Authors**: Minzhe Zheng, Lei Zheng, Lei Zhu, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.04563)  

**Abstract**: Ensuring safety and motion consistency for robot navigation in occluded, obstacle-dense environments is a critical challenge. In this context, this study presents an occlusion-aware Consistent Model Predictive Control (CMPC) strategy. To account for the occluded obstacles, it incorporates adjustable risk regions that represent their potential future locations. Subsequently, dynamic risk boundary constraints are developed online to ensure safety. The CMPC then constructs multiple locally optimal trajectory branches (each tailored to different risk regions) to balance between exploitation and exploration. A shared consensus trunk is generated to ensure smooth transitions between branches without significant velocity fluctuations, further preserving motion consistency. To facilitate high computational efficiency and ensure coordination across local trajectories, we use the alternating direction method of multipliers (ADMM) to decompose the CMPC into manageable sub-problems for parallel solving. The proposed strategy is validated through simulation and real-world experiments on an Ackermann-steering robot platform. The results demonstrate the effectiveness of the proposed CMPC strategy through comparisons with baseline approaches in occluded, obstacle-dense environments. 

**Abstract (ZH)**: 确保机器人在遮挡和障碍物密集环境中导航的安全性和运动一致性是一项关键挑战。在此背景下，本文提出了一种 Awareness-Occlusion 的一致模型预测控制（CMPC）策略。为了考虑遮挡的障碍物，它引入了可调节的风险区域来表示它们的潜在未来位置。随后，开发了在线动态风险边界约束以确保安全。CMPC 构建了多个局部最优轨迹分支（每个分支针对不同的风险区域），以在利用和探索之间取得平衡。生成了一个共享共识主干来确保分支之间的平滑过渡，同时避免显著的速度波动，从而进一步保持运动一致性。为了实现高效的计算效率并确保局部轨迹之间的协调，我们使用交替方向乘子法（ADMM）将 CMPC 分解为可并行求解的子问题。所提出的策略通过仿真实验和在 Ackermann 转向机器人平台上的实地试验得到了验证。结果表明，在遮挡和障碍物密集环境中，所提出的 CMPC 策略优于基线方法。 

---
# On the Analysis of Stability, Sensitivity and Transparency in Variable Admittance Control for pHRI Enhanced by Virtual Fixtures 

**Title (ZH)**: 基于虚拟fixtures增强的pHRI可变阻抗控制中的稳定性、灵敏度和透明度分析 

**Authors**: Davide Tebaldi, Dario Onfiani, Luigi Biagiotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.04414)  

**Abstract**: The interest in Physical Human-Robot Interaction (pHRI) has significantly increased over the last two decades thanks to the availability of collaborative robots that guarantee user safety during force exchanges. For this reason, stability concerns have been addressed extensively in the literature while proposing new control schemes for pHRI applications. Because of the nonlinear nature of robots, stability analyses generally leverage passivity concepts. On the other hand, the proposed algorithms generally consider ideal models of robot manipulators. For this reason, the primary objective of this paper is to conduct a detailed analysis of the sources of instability for a class of pHRI control schemes, namely proxy-based constrained admittance controllers, by considering parasitic effects such as transmission elasticity, motor velocity saturation, and actuation delay. Next, a sensitivity analysis supported by experimental results is carried out, in order to identify how the control parameters affect the stability of the overall system. Finally, an adaptation technique for the proxy parameters is proposed with the goal of maximizing transparency in pHRI. The proposed adaptation method is validated through both simulations and experimental tests. 

**Abstract (ZH)**: 基于代理的约束阻抗控制器下物理人机互动控制方案稳定性分析及其参数调整方法 

---
# SeGMan: Sequential and Guided Manipulation Planner for Robust Planning in 2D Constrained Environments 

**Title (ZH)**: SeGMan: 用于2D约束环境稳健规划的序列引导操作规划器 

**Authors**: Cankut Bora Tuncer, Dilruba Sultan Haliloglu, Ozgur S. Oguz  

**Link**: [PDF](https://arxiv.org/pdf/2503.04409)  

**Abstract**: In this paper, we present SeGMan, a hybrid motion planning framework that integrates sampling-based and optimization-based techniques with a guided forward search to address complex, constrained sequential manipulation challenges, such as pick-and-place puzzles. SeGMan incorporates an adaptive subgoal selection method that adjusts the granularity of subgoals, enhancing overall efficiency. Furthermore, proposed generalizable heuristics guide the forward search in a more targeted manner. Extensive evaluations in maze-like tasks populated with numerous objects and obstacles demonstrate that SeGMan is capable of generating not only consistent and computationally efficient manipulation plans but also outperform state-of-the-art approaches. 

**Abstract (ZH)**: 本文提出了一种名为SeGMan的混合运动规划框架，将基于采样的技术和基于优化的方法与引导式前向搜索相结合，以应对复杂的受约束的顺序操作挑战，如拾取和放置 puzzle。SeGMan 汇入了一种自适应子目标选择方法，可通过调整子目标的粒度来提高整体效率。此外，提出的可泛化的启发式方法以更针对性的方式引导前向搜索。在充满大量物体和障碍物的迷宫式任务中进行的广泛评估表明，SeGMan 不仅有能力生成一致且计算效率高的操作计划，还在性能上超越了现有最先进的方法。 

---
# Energy Consumption of Robotic Arm with the Local Reduction Method 

**Title (ZH)**: 基于局部缩减方法的机器人手臂能耗研究 

**Authors**: Halima Ibrahim Kure, Jishna Retnakumari, Lucian Nita, Saeed Sharif, Hamed Balogun, Augustine O. Nwajana  

**Link**: [PDF](https://arxiv.org/pdf/2503.04340)  

**Abstract**: Energy consumption in robotic arms is a significant concern in industrial automation due to rising operational costs and environmental impact. This study investigates the use of a local reduction method to optimize energy efficiency in robotic systems without compromising performance. The approach refines movement parameters, minimizing energy use while maintaining precision and operational reliability. A three-joint robotic arm model was tested using simulation over a 30-second period for various tasks, including pick-and-place and trajectory-following operations. The results revealed that the local reduction method reduced energy consumption by up to 25% compared to traditional techniques such as Model Predictive Control (MPC) and Genetic Algorithms (GA). Unlike MPC, which requires significant computational resources, and GA, which has slow convergence rates, the local reduction method demonstrated superior adaptability and computational efficiency in real-time applications. The study highlights the scalability and simplicity of the local reduction approach, making it an attractive option for industries seeking sustainable and cost-effective solutions. Additionally, this method can integrate seamlessly with emerging technologies like Artificial Intelligence (AI), further enhancing its application in dynamic and complex environments. This research underscores the potential of the local reduction method as a practical tool for optimizing robotic arm operations, reducing energy demands, and contributing to sustainability in industrial automation. Future work will focus on extending the approach to real-world scenarios and incorporating AI-driven adjustments for more dynamic adaptability. 

**Abstract (ZH)**: 机器人手臂的能效优化：局部减少方法的研究 

---
# Manipulation of Elasto-Flexible Cables with Single or Multiple UAVs 

**Title (ZH)**: 使用单个或多个无人机操纵弹性柔性缆线 

**Authors**: Chiara Gabellieri, Lars Teeuwen, Yaolei Shen, Antonio Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04304)  

**Abstract**: This work considers a large class of systems composed of multiple quadrotors manipulating deformable and extensible cables. The cable is described via a discretized representation, which decomposes it into linear springs interconnected through lumped-mass passive spherical joints. Sets of flat outputs are found for the systems. Numerical simulations support the findings by showing cable manipulation relying on flatness-based trajectories. Eventually, we present an experimental validation of the effectiveness of the proposed discretized cable model for a two-robot example. Moreover, a closed-loop controller based on the identified model and using cable-output feedback is experimentally tested. 

**Abstract (ZH)**: 本研究考虑了一类由多个 quadrotor 操纵可变形和可伸展缆线的系统。缆线通过离散化表示进行描述，将其分解为通过集中质量被动球关节连接的线性弹簧。找到了系统的集控输出集。数值模拟支持这些发现，展示了基于平坦性轨迹的缆线操纵。最后，我们介绍了所提出的离散化缆线模型在两机器人示例中的实验验证，并基于识别的模型设计了一个闭环控制器，使用缆线输出反馈进行了实验测试。 

---
# Simulation-based Analysis Of Highway Trajectory Planning Using High-Order Polynomial For Highly Automated Driving Function 

**Title (ZH)**: 基于高次多项式的高速公路轨迹规划simulations分析：面向高层次自动驾驶功能 

**Authors**: Milin Patel, Marzana Khatun, Rolf Jung, Michael Glaß  

**Link**: [PDF](https://arxiv.org/pdf/2503.04159)  

**Abstract**: One of the fundamental tasks of autonomous driving is safe trajectory planning, the task of deciding where the vehicle needs to drive, while avoiding obstacles, obeying safety rules, and respecting the fundamental limits of road. Real-world application of such a method involves consideration of surrounding environment conditions and movements such as Lane Change, collision avoidance, and lane merge. The focus of the paper is to develop and implement safe collision free highway Lane Change trajectory using high order polynomial for Highly Automated Driving Function (HADF). Planning is often considered as a higher-level process than control. Behavior Planning Module (BPM) is designed that plans the high-level driving actions like Lane Change maneuver to safely achieve the functionality of transverse guidance ensuring safety of the vehicle using motion planning in a scenario including environmental situation. Based on the recommendation received from the (BPM), the function will generate a desire corresponding trajectory. The proposed planning system is situation specific with polynomial based algorithm for same direction two lane highway scenario. To support the trajectory system polynomial curve can be used to reduces overall complexity and thereby allows rapid computation. The proposed Lane Change scenario is modeled, and results has been analyzed (verified and validate) through the MATLAB simulation environment. The method proposed in this paper has achieved a significant improvement in safety and stability of Lane Changing maneuver. 

**Abstract (ZH)**: 基于高次多项式的适配行驶车道变道无碰撞安全轨迹规划方法 

---
# Real-time Spatial-temporal Traversability Assessment via Feature-based Sparse Gaussian Process 

**Title (ZH)**: 基于特征的稀疏高斯过程实时时空可通行性评估 

**Authors**: Senming Tan, Zhenyu Hou, Zhihao Zhang, Long Xu, Mengke Zhang, Zhaoqi He, Chao Xu, Fei Gao, Yanjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04134)  

**Abstract**: Terrain analysis is critical for the practical application of ground mobile robots in real-world tasks, especially in outdoor unstructured environments. In this paper, we propose a novel spatial-temporal traversability assessment method, which aims to enable autonomous robots to effectively navigate through complex terrains. Our approach utilizes sparse Gaussian processes (SGP) to extract geometric features (curvature, gradient, elevation, etc.) directly from point cloud scans. These features are then used to construct a high-resolution local traversability map. Then, we design a spatial-temporal Bayesian Gaussian kernel (BGK) inference method to dynamically evaluate traversability scores, integrating historical and real-time data while considering factors such as slope, flatness, gradient, and uncertainty metrics. GPU acceleration is applied in the feature extraction step, and the system achieves real-time performance. Extensive simulation experiments across diverse terrain scenarios demonstrate that our method outperforms SOTA approaches in both accuracy and computational efficiency. Additionally, we develop an autonomous navigation framework integrated with the traversability map and validate it with a differential driven vehicle in complex outdoor environments. Our code will be open-source for further research and development by the community, this https URL. 

**Abstract (ZH)**: 地形分析对于地面移动机器人在实际任务中的应用至关重要，特别是在户外未结构化环境中。本文提出了一种新颖的时空通达性评估方法，旨在使自主机器人能够有效导航通过复杂地形。我们的方法利用稀疏高斯过程（SGP）直接从点云扫描中提取几何特征（曲率、坡度、高程等），然后利用这些特征构建高分辨率局部通达性地图。随后，我们设计了一种时空贝叶斯高斯核（BGK）推断方法，以动态评估通达性评分，综合历史和实时数据，并考虑坡度、平坦度、坡度和不确定性指标等因素。在特征提取步骤中应用了GPU加速，系统实现了实时性能。在各种地形场景的广泛模拟实验中，我们的方法在准确性和计算效率方面均优于当前最佳方法。此外，我们开发了一种集成了通达性地图的自主导航框架，并在复杂户外环境中用差速驱动车辆进行了验证。我们的代码将开源供社区进一步研究和开发使用，this https URL。 

---
# DVM-SLAM: Decentralized Visual Monocular Simultaneous Localization and Mapping for Multi-Agent Systems 

**Title (ZH)**: 多代理系统中去中心化视觉单目同时定位与建图 

**Authors**: Joshua Bird, Jan Blumenkamp, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2503.04126)  

**Abstract**: Cooperative Simultaneous Localization and Mapping (C-SLAM) enables multiple agents to work together in mapping unknown environments while simultaneously estimating their own positions. This approach enhances robustness, scalability, and accuracy by sharing information between agents, reducing drift, and enabling collective exploration of larger areas. In this paper, we present Decentralized Visual Monocular SLAM (DVM-SLAM), the first open-source decentralized monocular C-SLAM system. By only utilizing low-cost and light-weight monocular vision sensors, our system is well suited for small robots and micro aerial vehicles (MAVs). DVM-SLAM's real-world applicability is validated on physical robots with a custom collision avoidance framework, showcasing its potential in real-time multi-agent autonomous navigation scenarios. We also demonstrate comparable accuracy to state-of-the-art centralized monocular C-SLAM systems. We open-source our code and provide supplementary material online. 

**Abstract (ZH)**: 协作 simultaneous localization and mapping (C-SLAM) 允许多个代理在测绘未知环境的同时共同估计各自的定位。该方法通过共享信息、减少漂移和实现更大区域的集体探索，增强了鲁棒性、可扩展性和准确性。在本文中，我们提出了去中心化单目视觉 SLAM (DVM-SLAM)，这是首个开源的去中心化单目 C-SLAM 系统。仅通过使用低成本和轻量级的单目视觉传感器，我们的系统非常适合小型机器人和微空中车辆 (MAVs)。DVM-SLAM 在具有自定义避碰框架的实际机器人上验证了其现实世界的应用性，并展示了其在实时多代理自主导航场景中的潜力。我们还展示了与最先进的集中式单目 C-SLAM 系统相当的准确性。我们在开源代码并在线提供补充材料。 

---
# RA-DP: Rapid Adaptive Diffusion Policy for Training-Free High-frequency Robotics Replanning 

**Title (ZH)**: RA-DP: 快速自适应扩散策略用于无需训练的高频机器人重规划 

**Authors**: Xi Ye, Rui Heng Yang, Jun Jin, Yinchuan Li, Amir Rasouli  

**Link**: [PDF](https://arxiv.org/pdf/2503.04051)  

**Abstract**: Diffusion models exhibit impressive scalability in robotic task learning, yet they struggle to adapt to novel, highly dynamic environments. This limitation primarily stems from their constrained replanning ability: they either operate at a low frequency due to a time-consuming iterative sampling process, or are unable to adapt to unforeseen feedback in case of rapid replanning. To address these challenges, we propose RA-DP, a novel diffusion policy framework with training-free high-frequency replanning ability that solves the above limitations in adapting to unforeseen dynamic environments. Specifically, our method integrates guidance signals which are often easily obtained in the new environment during the diffusion sampling process, and utilizes a novel action queue mechanism to generate replanned actions at every denoising step without retraining, thus forming a complete training-free framework for robot motion adaptation in unseen environments. Extensive evaluations have been conducted in both well-recognized simulation benchmarks and real robot tasks. Results show that RA-DP outperforms the state-of-the-art diffusion-based methods in terms of replanning frequency and success rate. Moreover, we show that our framework is theoretically compatible with any training-free guidance signal. 

**Abstract (ZH)**: Diffusion模型在机器人任务学习中展现了令人 Impressive 的可扩展性，但它们在适应新型高度动态环境方面存在困难。这种限制主要源于它们受限的重新规划能力：它们要么因耗时的迭代采样过程而以低频运行，要么在需要快速重新规划的情况下无法适应未预见的反馈。为解决这些挑战，我们提出了RA-DP，这是一种具有无训练高频率重新规划能力的新型扩散策略框架，能够解决在适应未预见动态环境方面的上述限制。具体而言，我们的方法在扩散采样过程中整合了通常容易在新环境中获得的引导信号，并利用一种新的动作队列机制在去噪的每一步生成重新规划的动作，从而形成一个完整的无训练框架，用于机器人在未见过的环境中的动作适应。在公认的仿真基准和实际机器人任务中进行了广泛的评估。结果表明，RA-DP在重新规划频率和成功率方面优于现有的基于扩散的方法。此外，我们展示了我们的框架在理论上与任何无训练引导信号兼容。 

---
# Object State Estimation Through Robotic Active Interaction for Biological Autonomous Drilling 

**Title (ZH)**: 基于机器人主动交互的生物自主钻探状态估计 

**Authors**: Xiaofeng Lin, Enduo Zhao, Saúl Alexis Heredia Pérez, Kanako Harada  

**Link**: [PDF](https://arxiv.org/pdf/2503.04043)  

**Abstract**: Estimating the state of biological specimens is challenging due to limited observation through microscopic vision. For instance, during mouse skull drilling, the appearance alters little when thinning bone tissue because of its semi-transparent property and the high-magnification microscopic vision. To obtain the object's state, we introduce an object state estimation method for biological specimens through active interaction based on the deflection. The method is integrated to enhance the autonomous drilling system developed in our previous work. The method and integrated system were evaluated through 12 autonomous eggshell drilling experiment trials. The results show that the system achieved a 91.7% successful ratio and 75% detachable ratio, showcasing its potential applicability in more complex surgical procedures such as mouse skull craniotomy. This research paves the way for further development of autonomous robotic systems capable of estimating the object's state through active interaction. 

**Abstract (ZH)**: 通过主动交互基于偏转的生物标本状态估计方法及其在自主钻孔系统中的应用 

---
# Autonomous Robotic Bone Micro-Milling System with Automatic Calibration and 3D Surface Fitting 

**Title (ZH)**: 自主机器人骨微铣系统及其自动校准与三维表面拟合 

**Authors**: Enduo Zhao, Xiaofeng Lin, Yifan Wang, Kanako Harada  

**Link**: [PDF](https://arxiv.org/pdf/2503.04038)  

**Abstract**: Automating bone micro-milling using a robotic system presents challenges due to the uncertainties in both the external and internal features of bone tissue. For example, during a mouse cranial window creation, a circular path with a radius of 2 to 4 mm needs to be milled on the mouse skull using a microdrill. The uneven surface and non-uniform thickness of the mouse skull make it difficult to fully automate this process, requiring the system to possess advanced perceptual and adaptive capabilities. In this study, we propose an automatic calibration and 3D surface fitting method and integrate it into an autonomous robotic bone micro-milling system, enabling it to quickly, in real-time, and accurately perceive and adapt to the uneven surface and non-uniform thickness of the target without human assistance. Validation experiments on euthanized mice demonstrate that the improved system achieves a success rate of 85.7 % and an average milling time of 2.1 minutes, showing not only significant performance improvements over the previous system but also exceptional accuracy, speed, and stability compared to human operators. 

**Abstract (ZH)**: 利用机器人系统自动化骨微 milling 遇到了由于骨组织内外特征的不确定性所带来的挑战。例如，在创建小鼠颅窗时，需要使用微型钻在小鼠颅骨上铣出直径为 2 至 4 毫米的圆路径。小鼠颅骨不均匀的表面和非均匀的厚度使得完全自动化此过程变得困难，因此系统需要具备先进的感知和自适应能力。在本研究中，我们提出了一种自动校准和三维表面拟合方法，并将其集成到自主机器人骨微 milling 系统中，使其能够在没有人工辅助的情况下快速、实时地感知和适应目标的不均匀表面和非均匀厚度。实验表明，改进后的系统在麻醉小鼠上的成功率达到了 85.7%，平均加工时间为 2.1 分钟，不仅在性能上显著优于之前系统，在准确度、速度和稳定性方面也超过了人工操作者。 

---
# Planning and Control for Deformable Linear Object Manipulation 

**Title (ZH)**: 可变形线性物体操作的规划与控制 

**Authors**: Burak Aksoy, John Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.04007)  

**Abstract**: Manipulating a deformable linear object (DLO) such as wire, cable, and rope is a common yet challenging task due to their high degrees of freedom and complex deformation behaviors, especially in an environment with obstacles. Existing local control methods are efficient but prone to failure in complex scenarios, while precise global planners are computationally intensive and difficult to deploy. This paper presents an efficient, easy-to-deploy framework for collision-free DLO manipulation using mobile manipulators. We demonstrate the effectiveness of leveraging standard planning tools for high-dimensional DLO manipulation without requiring custom planners or extensive data-driven models. Our approach combines an off-the-shelf global planner with a real-time local controller. The global planner approximates the DLO as a series of rigid links connected by spherical joints, enabling rapid path planning without the need for problem-specific planners or large datasets. The local controller employs control barrier functions (CBFs) to enforce safety constraints, maintain the DLO integrity, prevent overstress, and handle obstacle avoidance. It compensates for modeling inaccuracies by using a state-of-the-art position-based dynamics technique that approximates physical properties like Young's and shear moduli. We validate our framework through extensive simulations and real-world demonstrations. In complex obstacle scenarios-including tent pole transport, corridor navigation, and tasks requiring varied stiffness-our method achieves a 100% success rate over thousands of trials, with significantly reduced planning times compared to state-of-the-art techniques. Real-world experiments include transportation of a tent pole and a rope using mobile manipulators. We share our ROS-based implementation to facilitate adoption in various applications. 

**Abstract (ZH)**: 基于移动操作器的无障碍变形线性对象操纵高效易部署框架 

---
# Robotic Compliant Object Prying Using Diffusion Policy Guided by Vision and Force Observations 

**Title (ZH)**: 视觉和力感知引导的扩散策略用于柔顺物体撬取的机器人技术 

**Authors**: Jeon Ho Kang, Sagar Joshi, Ruopeng Huang, Satyandra K. Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2503.03998)  

**Abstract**: The growing adoption of batteries in the electric vehicle industry and various consumer products has created an urgent need for effective recycling solutions. These products often contain a mix of compliant and rigid components, making robotic disassembly a critical step toward achieving scalable recycling processes. Diffusion policy has emerged as a promising approach for learning low-level skills in robotics. To effectively apply diffusion policy to contact-rich tasks, incorporating force as feedback is essential. In this paper, we apply diffusion policy with vision and force in a compliant object prying task. However, when combining low-dimensional contact force with high-dimensional image, the force information may be diluted. To address this issue, we propose a method that effectively integrates force with image data for diffusion policy observations. We validate our approach on a battery prying task that demands high precision and multi-step execution. Our model achieves a 96\% success rate in diverse scenarios, marking a 57\% improvement over the vision-only baseline. Our method also demonstrates zero-shot transfer capability to handle unseen objects and battery types. Supplementary videos and implementation codes are available on our project website. this https URL 

**Abstract (ZH)**: 电动汽车行业和各种消费产品的电池应用越来越多，迫切需要有效的回收解决方案。这些产品通常包含柔性和刚性部件的混合，因此机器人拆卸成为实现可扩展回收过程的关键步骤。扩散策略已 emerges 作为一种有潜力的方法来学习机器人的低级技能。为了有效将扩散策略应用于富含接触的任务，引入力反馈是必不可少的。在本文中，我们在柔体物体分拣任务中应用结合视觉和力的扩散策略。然而，当将低维接触力与高维图像结合时，力信息可能会被稀释。为了解决这一问题，我们提出了一种有效整合力和图像数据的方法，以改善扩散策略的观察效果。我们通过一项需要高精度和多步骤执行的电池分拣任务验证了该方法。我们的模型在多种场景下实现了96%的成功率，比仅使用视觉的基线提高了57%。此外，我们的方法还展示了处理未见过的物体和电池类型的零样本迁移能力。有关补充视频和实现代码，可访问我们的项目网站：this https URL。 

---
# GeoFIK: A Fast and Reliable Geometric Solver for the IK of the Franka Arm based on Screw Theory Enabling Multiple Redundancy Parameters 

**Title (ZH)**: GeoFIK: 基于螺杆理论的Franka臂的快速可靠几何求解器及其在处理冗余参数方面的应用 

**Authors**: Pablo C. Lopez-Custodio, Yuhe Gong, Luis F.C. Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2503.03992)  

**Abstract**: Modern robotics applications require an inverse kinematics (IK) solver that is fast, robust and consistent, and that provides all possible solutions. Currently, the Franka robot arm is the most widely used manipulator in robotics research. With 7 DOFs, the IK of this robot is not only complex due to its 1-DOF redundancy, but also due to the link offsets at the wrist and elbow. Due to this complexity, none of the Franka IK solvers available in the literature provide satisfactory results when used in real-world applications. Therefore, in this paper we introduce GeoFIK (Geometric Franka IK), an analytical IK solver that allows the use of different joint variables to resolve the redundancy. The approach uses screw theory to describe the entire geometry of the robot, allowing the computation of the Jacobian matrix prior to computation of joint angles. All singularities are identified and handled. As an example of how the geometric elements obtained by the IK can be exploited, a solver with the swivel angle as the free variable is provided. Several experiments are carried out to validate the speed, robustness and reliability of the GeoFIK against two state-of-the-art solvers. 

**Abstract (ZH)**: 现代机器人应用需要一个快速、稳健且一致的逆运动学（IK）求解器，并能提供所有可能的解。目前， Franka 机器人臂是机器人研究中应用最广泛的 manipulator。由于其 1-DOF 冗余以及手腕和肘部的连杆偏移，即使有 7 个自由度，Franka 机器人的 IK 也是复杂的。由于这种复杂性，在文献中可用的所有 Franka IK 求解器在实际应用中都无法提供满意的结果。因此，本文提出了一种名为 GeoFIK（几何 Franka 逆运动学）的解析 IK 求解器，该求解器允许使用不同的关节变量来解决冗余问题。该方法使用螺丝理论描述整个机器人的几何结构，允许在计算关节角之前计算雅可比矩阵。所有奇异性都被识别并处理。作为通过 IK 所获得的几何元素的应用示例，提供了一个以俯仰角作为自由变量的求解器。进行了多项实验，以验证 GeoFIK 在速度、稳健性和可靠性方面与两个最先进的求解器之间的对比。 

---
# GO-VMP: Global Optimization for View Motion Planning in Fruit Mapping 

**Title (ZH)**: GO-VMP: 全局优化在水果测绘中的视图运动规划 

**Authors**: Allen Isaac Jose, Sicong Pan, Tobias Zaenker, Rohit Menon, Sebastian Houben, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.03912)  

**Abstract**: Automating labor-intensive tasks such as crop monitoring with robots is essential for enhancing production and conserving resources. However, autonomously monitoring horticulture crops remains challenging due to their complex structures, which often result in fruit occlusions. Existing view planning methods attempt to reduce occlusions but either struggle to achieve adequate coverage or incur high robot motion costs. We introduce a global optimization approach for view motion planning that aims to minimize robot motion costs while maximizing fruit coverage. To this end, we leverage coverage constraints derived from the set covering problem (SCP) within a shortest Hamiltonian path problem (SHPP) formulation. While both SCP and SHPP are well-established, their tailored integration enables a unified framework that computes a global view path with minimized motion while ensuring full coverage of selected targets. Given the NP-hard nature of the problem, we employ a region-prior-based selection of coverage targets and a sparse graph structure to achieve effective optimization outcomes within a limited time. Experiments in simulation demonstrate that our method detects more fruits, enhances surface coverage, and achieves higher volume accuracy than the motion-efficient baseline with a moderate increase in motion cost, while significantly reducing motion costs compared to the coverage-focused baseline. Real-world experiments further confirm the practical applicability of our approach. 

**Abstract (ZH)**: 利用机器人自动化劳动密集型任务如作物监测对于提升生产效率和节约资源至关重要，但由于园艺作物结构复杂常导致果实遮挡，自主监测园艺作物仍然具有挑战性。我们提出了一种全局优化视角运动规划方法，旨在最小化机器人运动成本同时最大化果实覆盖率。为此，我们在最短哈密尔顿路径问题（SHPP）建模中利用来自集合覆盖问题（SCP）的覆盖约束。尽管SCP和SHPP都是成熟的数学模型，但它们的结合使我们能够提出一个统一框架，该框架能在确保覆盖选定目标的同时计算出最小化运动成本的全局视角路径。由于问题是NP难的，我们采用了基于区域优先的选择覆盖目标和稀疏图结构，以在有限时间内实现有效的优化效果。模拟实验结果显示，与运动效率基线相比，我们的方法能够在适度增加运动成本的同时检测更多果实、提高表面覆盖度和实现更高的体积精度，而且与以覆盖为目标的基线相比，显著降低了运动成本。实地实验进一步验证了我们方法的实用性。 

---
# Endpoint-Explicit Differential Dynamic Programming via Exact Resolution 

**Title (ZH)**: 端点显式差分动态规划通过精确分辨率 

**Authors**: Maria Parilli, Sergi Martinez, Carlos Mastalli  

**Link**: [PDF](https://arxiv.org/pdf/2503.03897)  

**Abstract**: We introduce a novel method for handling endpoint constraints in constrained differential dynamic programming (DDP). Unlike existing approaches, our method guarantees quadratic convergence and is exact, effectively managing rank deficiencies in both endpoint and stagewise equality constraints. It is applicable to both forward and inverse dynamics formulations, making it particularly well-suited for model predictive control (MPC) applications and for accelerating optimal control (OC) solvers. We demonstrate the efficacy of our approach across a broad range of robotics problems and provide a user-friendly open-source implementation within CROCODDYL. 

**Abstract (ZH)**: 我们提出了一种在约束差分动力规划（DDP）中处理端点约束的新方法。与现有方法不同，该方法保证了二次收敛性和精确性，有效管理了端点和阶段等式约束中的秩缺陷。该方法适用于正向和逆向动力学 formulations，特别适合模型预测控制（MPC）应用，并能加速最优控制（OC）求解器。我们通过广泛领域的机器人学问题展示了该方法的有效性，并在CROCODDYL中提供了用户友好的开源实现。 

---
# Fair Play in the Fast Lane: Integrating Sportsmanship into Autonomous Racing Systems 

**Title (ZH)**: 公平竞赛在快车道上的实现：将运动精神融入自主赛车系统 

**Authors**: Zhenmin Huang, Ce Hao, Wei Zhan, Jun Ma, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2503.03774)  

**Abstract**: Autonomous racing has gained significant attention as a platform for high-speed decision-making and motion control. While existing methods primarily focus on trajectory planning and overtaking strategies, the role of sportsmanship in ensuring fair competition remains largely unexplored. In human racing, rules such as the one-motion rule and the enough-space rule prevent dangerous and unsportsmanlike behavior. However, autonomous racing systems often lack mechanisms to enforce these principles, potentially leading to unsafe maneuvers. This paper introduces a bi-level game-theoretic framework to integrate sportsmanship (SPS) into versus racing. At the high level, we model racing intentions using a Stackelberg game, where Monte Carlo Tree Search (MCTS) is employed to derive optimal strategies. At the low level, vehicle interactions are formulated as a Generalized Nash Equilibrium Problem (GNEP), ensuring that all agents follow sportsmanship constraints while optimizing their trajectories. Simulation results demonstrate the effectiveness of the proposed approach in enforcing sportsmanship rules while maintaining competitive performance. We analyze different scenarios where attackers and defenders adhere to or disregard sportsmanship rules and show how knowledge of these constraints influences strategic decision-making. This work highlights the importance of balancing competition and fairness in autonomous racing and provides a foundation for developing ethical and safe AI-driven racing systems. 

**Abstract (ZH)**: 自主赛车比赛作为一种高速决策和运动控制的平台引起了广泛关注。虽然现有方法主要集中在轨迹规划和超越策略上，但在确保公平竞争中体育精神的作用仍被很大程度上忽视。在人类赛车中，一动规则和足够空间规则等规则可以防止危险和不道德行为。然而，自主赛车系统往往缺乏执行这些原则的机制，可能导致不安全的操作。本文介绍了一种多层次的游戏理论框架，将体育精神（SPS）整合到对抗赛车中。在高层次上，我们使用Stackelberg博弈来建模赛车意图，并利用蒙特卡洛树搜索（MCTS）来推导最优策略。在低层次上，车辆交互被形式化为广义纳什均衡问题（GNEP），以确保所有代理都遵循体育精神约束并优化它们的轨迹。仿真结果显示，所提出的方法在维护竞争力的同时有效执行体育精神规则。我们分析了攻击者和防守者遵守或忽视体育精神规则的不同情境，并展示了这些约束知识如何影响战略决策。本文强调了在自主赛车中平衡竞争与公平的重要性，并为开发具有伦理和安全性的人工智能驱动赛车系统奠定了基础。 

---
