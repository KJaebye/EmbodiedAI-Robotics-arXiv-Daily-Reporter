# Limits of specifiability for sensor-based robotic planning tasks 

**Title (ZH)**: 基于传感器的机器人规划任务的可指谓极限 

**Authors**: Basak Sakcak, Dylan A. Shell, Jason M. O'Kane  

**Link**: [PDF](https://arxiv.org/pdf/2503.05623)  

**Abstract**: There is now a large body of techniques, many based on formal methods, for describing and realizing complex robotics tasks, including those involving a variety of rich goals and time-extended behavior. This paper explores the limits of what sorts of tasks are specifiable, examining how the precise grounding of specifications, that is, whether the specification is given in terms of the robot's states, its actions and observations, its knowledge, or some other information,is crucial to whether a given task can be specified. While prior work included some description of particular choices for this grounding, our contribution treats this aspect as a first-class citizen: we introduce notation to deal with a large class of problems, and examine how the grounding affects what tasks can be posed. The results demonstrate that certain classes of tasks are specifiable under different combinations of groundings. 

**Abstract (ZH)**: 现有的技术，许多基于形式化方法，可用于描述和实现复杂的机器人任务，包括涉及多种丰富目标和长时间行为的任务。本文探讨了可描述任务的极限，研究了规范的确切 grounding，即规范是以机器人状态、行动和观察、知识或其他信息的形式给出的，对是否可以描述给定任务至关重要。虽然早期工作对此 grounding 有一些描述，但我们在此将这一方面视为一等公民：我们引入符号来处理一类问题，并探讨 grounding 如何影响可提出的任务类型。结果表明，在不同的 grounding 组合下，某些类别的任务是可以描述的。 

---
# A-SEE2.0: Active-Sensing End-Effector for Robotic Ultrasound Systems with Dense Contact Surface Perception Enabled Probe Orientation Adjustment 

**Title (ZH)**: A-SEE2.0: 具有密集接触表面感知的主动传感末端执行器，实现探头姿态调整的机器人超声系统 

**Authors**: Yernar Zhetpissov, Xihan Ma, Kehan Yang, Haichong K. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05569)  

**Abstract**: Conventional freehand ultrasound (US) imaging is highly dependent on the skill of the operator, often leading to inconsistent results and increased physical demand on sonographers. Robotic Ultrasound Systems (RUSS) aim to address these limitations by providing standardized and automated imaging solutions, especially in environments with limited access to skilled operators. This paper presents the development of a novel RUSS system that employs dual RGB-D depth cameras to maintain the US probe normal to the skin surface, a critical factor for optimal image quality. Our RUSS integrates RGB-D camera data with robotic control algorithms to maintain orthogonal probe alignment on uneven surfaces without preoperative data. Validation tests using a phantom model demonstrate that the system achieves robust normal positioning accuracy while delivering ultrasound images comparable to those obtained through manual scanning. A-SEE2.0 demonstrates 2.47 ${\pm}$ 1.25 degrees error for flat surface normal-positioning and 12.19 ${\pm}$ 5.81 degrees normal estimation error on mannequin surface. This work highlights the potential of A-SEE2.0 to be used in clinical practice by testing its performance during in-vivo forearm ultrasound examinations. 

**Abstract (ZH)**: 传统自由手超声成像高度依赖操作者的技能，往往会带来不一致的结果并增加超声技师的体力负担。机器人超声系统（RUSS）旨在通过提供标准化和自动化的成像解决方案来解决这些问题，特别是在限制性操作者技能的环境中。本文介绍了一种采用双RGB-D深度摄像头的新颖RUSS系统，该系统用于保持超声探头与皮肤表面垂直，这对于获得最佳图像质量至关重要。我们的RUSS系统将RGB-D摄像头数据与机器人控制算法相结合，在无需术前数据的情况下，在不平表面上保持探头的正交对准。使用仿真模型进行的验证测试显示，该系统在实现稳健的垂直定位精度的同时，提供的超声图像与手动扫描获得的图像相当。A-SEE2.0在平坦表面的垂直定位误差为2.47 ${\pm}$ 1.25 度，在人体模型表面的垂直估计误差为12.19 ${\pm}$ 5.81 度。这项工作通过在活体前臂超声检查中测试其性能，突显了A-SEE2.0在临床实践中的潜在应用价值。 

---
# Design, Dynamic Modeling and Control of a 2-DOF Robotic Wrist Actuated by Twisted and Coiled Actuators 

**Title (ZH)**: 由扭曲和螺旋执行器驱动的2-DOF机器人手腕的设计、动态建模与控制 

**Authors**: Yunsong Zhang, Xinyu Zhou, Feitian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05508)  

**Abstract**: Robotic wrists play a pivotal role in the functionality of industrial manipulators and humanoid robots, facilitating manipulation and grasping tasks. In recent years, there has been a growing interest in integrating artificial muscle-driven actuators for robotic wrists, driven by advancements in technology offering high energy density, lightweight construction, and compact designs. However, in the study of robotic wrists driven by artificial muscles, dynamic model-based controllers are often overlooked, despite their critical importance for motion analysis and dynamic control of robots. This paper presents a novel design of a two-degree-of-freedom (2-DOF) robotic wrist driven by twisted and coiled actuators (TCA) utilizing a parallel mechanism with a 3RRRR configuration. The proposed robotic wrist is expected to feature lightweight structures and superior motion performance while mitigating friction issues. The Lagrangian dynamic model of the wrist is established, along with a nonlinear model predictive controller (NMPC) designed for trajectory tracking tasks. A prototype of the robotic wrist is developed, and extensive experiments are conducted to validate its superior motion performance and the proposed dynamic model. Subsequently, extensive comparative experiments between NMPC and PID controller were conducted under various operating conditions. The experimental results demonstrate the effectiveness and robustness of the dynamic model-based controller in the motion control of TCA-driven robotic wrists. 

**Abstract (ZH)**: 基于人工肌肉驱动的两自由度并联机械手腕的设计与动态控制 

---
# Reward-Centered ReST-MCTS: A Robust Decision-Making Framework for Robotic Manipulation in High Uncertainty Environments 

**Title (ZH)**: 面向奖励中心的ReST-MCTS：高不确定性环境下机器人操控的稳健决策框架 

**Authors**: Xibai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05226)  

**Abstract**: Monte Carlo Tree Search (MCTS) has emerged as a powerful tool for decision-making in robotics, enabling efficient exploration of large search spaces. However, traditional MCTS methods struggle in environments characterized by high uncertainty and noisy data due to their reliance on final-step reward evaluation. The lack of intermediate feedback during search often results in suboptimal decision-making and computational inefficiencies.
This paper introduces Reward-Centered ReST-MCTS, a novel framework that enhances MCTS by incorporating intermediate reward shaping. The core of our approach is the Rewarding Center, which refines search trajectories by dynamically assigning partial rewards using rule-based validation, heuristic guidance, and neural estimation. By integrating these mechanisms, our method enables real-time optimization of search paths, mitigating the effects of error propagation.
We evaluate Reward-Centered ReST-MCTS in robotic manipulation tasks under high uncertainty, demonstrating consistent improvements in decision accuracy. Compared to baseline methods, including Chain-of-Thought (CoT) prompting and Vanilla ReST-MCTS, our framework achieves a 2-4% accuracy improvement while maintaining computational feasibility. Ablation studies confirm the effectiveness of intermediate feedback in search refinement, particularly in pruning incorrect decision paths early. Furthermore, robustness tests show that our method retains high performance across varying levels of uncertainty. 

**Abstract (ZH)**: 基于奖励中心的ReST-MCTS：一种增强的蒙特卡洛树搜索框架 

---
# Generative Trajectory Stitching through Diffusion Composition 

**Title (ZH)**: 通过扩散合成的生成性轨迹拼接 

**Authors**: Yunhao Luo, Utkarsh A. Mishra, Yilun Du, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05153)  

**Abstract**: Effective trajectory stitching for long-horizon planning is a significant challenge in robotic decision-making. While diffusion models have shown promise in planning, they are limited to solving tasks similar to those seen in their training data. We propose CompDiffuser, a novel generative approach that can solve new tasks by learning to compositionally stitch together shorter trajectory chunks from previously seen tasks. Our key insight is modeling the trajectory distribution by subdividing it into overlapping chunks and learning their conditional relationships through a single bidirectional diffusion model. This allows information to propagate between segments during generation, ensuring physically consistent connections. We conduct experiments on benchmark tasks of various difficulties, covering different environment sizes, agent state dimension, trajectory types, training data quality, and show that CompDiffuser significantly outperforms existing methods. 

**Abstract (ZH)**: 长 horizon 规划中有效的轨迹拼接是机器人决策中的一个显著挑战。虽然扩散模型在规划方面展现了前景，但它们局限于解决与其训练数据相似的任务。我们提出 CompDiffuser，这是一种新颖的生成方法，能够通过学习将之前见过的任务中的较短轨迹片段组合起来解决新任务。我们的关键是通过将轨迹分布划分为重叠片段，并通过单一的双向扩散模型学习它们的条件关系，从而在生成过程中使信息在片段之间传播，确保物理上一致的连接。我们在涵盖不同环境规模、代理状态维度、轨迹类型、训练数据质量的各种难度基准任务上进行实验，结果显示 CompDiffuser 显著优于现有方法。 

---
# Unity RL Playground: A Versatile Reinforcement Learning Framework for Mobile Robots 

**Title (ZH)**: Unity RL Playground: 一种适用于移动机器人的多功能 reinforcement learning 框架 

**Authors**: Linqi Ye, Rankun Li, Xiaowen Hu, Jiayi Li, Boyang Xing, Yan Peng, Bin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05146)  

**Abstract**: This paper introduces Unity RL Playground, an open-source reinforcement learning framework built on top of Unity ML-Agents. Unity RL Playground automates the process of training mobile robots to perform various locomotion tasks such as walking, running, and jumping in simulation, with the potential for seamless transfer to real hardware. Key features include one-click training for imported robot models, universal compatibility with diverse robot configurations, multi-mode motion learning capabilities, and extreme performance testing to aid in robot design optimization and morphological evolution. The attached video can be found at this https URL and the code is coming soon. 

**Abstract (ZH)**: 本文介绍了基于Unity ML-Agents构建的开源强化学习框架Unity RL Playground。Unity RL Playground自动训练移动机器人执行各种运动任务（如行走、跑步和跳跃）的模拟过程，并具有无缝转移到实际硬件的潜力。主要特点包括一键训练导入的机器人模型、适用于多种机器人配置的通用兼容性、多模式运动学习能力和极端性能测试，以助于机器人设计优化和形态演化。有关视频链接请参见此处：https://this-url/ 代码即将发布。 

---
# Adaptive-LIO: Enhancing Robustness and Precision through Environmental Adaptation in LiDAR Inertial Odometry 

**Title (ZH)**: 自适应-LIO：通过环境适应提高激光雷达惯性里程计的稳健性和精度 

**Authors**: Chengwei Zhao, Kun Hu, Jie Xu, Lijun Zhao, Baiwen Han, Kaidi Wu, Maoshan Tian, Shenghai Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05077)  

**Abstract**: The emerging Internet of Things (IoT) applications, such as driverless cars, have a growing demand for high-precision positioning and navigation. Nowadays, LiDAR inertial odometry becomes increasingly prevalent in robotics and autonomous driving. However, many current SLAM systems lack sufficient adaptability to various scenarios. Challenges include decreased point cloud accuracy with longer frame intervals under the constant velocity assumption, coupling of erroneous IMU information when IMU saturation occurs, and decreased localization accuracy due to the use of fixed-resolution maps during indoor-outdoor scene transitions. To address these issues, we propose a loosely coupled adaptive LiDAR-Inertial-Odometry named \textbf{Adaptive-LIO}, which incorporates adaptive segmentation to enhance mapping accuracy, adapts motion modality through IMU saturation and fault detection, and adjusts map resolution adaptively using multi-resolution voxel maps based on the distance from the LiDAR center. Our proposed method has been tested in various challenging scenarios, demonstrating the effectiveness of the improvements we introduce. The code is open-source on GitHub: \href{this https URL}{Adaptive-LIO}. 

**Abstract (ZH)**: 新兴的物联网（IoT）应用，如无人驾驶汽车，对高精度定位和导航提出了日益增长的需求。如今，激光雷达惯性里程计（LiDAR inertial odometry）在机器人技术与自动驾驶中越来越普遍。然而，当前许多SLAM系统在应对各种场景时缺乏足够的适应性。这些挑战包括在恒定速度假设下，随着帧间隔延长导致点云精度下降；在IMU饱和时耦合错误的IMU信息；以及在室内外场景过渡期间使用固定分辨率地图导致的定位精度下降。为了解决这些问题，我们提出了一种松耦合自适应激光雷达-惯性-里程计（Loosely coupled adaptive LiDAR-Inertial-Odometry，命名为Adaptive-LIO），该方法结合了自适应分割以增强建图精度，通过IMU饱和和故障检测来适应运动模式，并使用基于LiDAR中心距离的多分辨率体素地图来自适应调整地图分辨率。我们的方法已在各种挑战场景中进行测试，展示了所提出改进措施的有效性。代码已开源在GitHub上：Adaptive-LIO。 

---
# Prismatic-Bending Transformable (PBT) Joint for a Modular, Foldable Manipulator with Enhanced Reachability and Dexterity 

**Title (ZH)**: 可变换棱柱弯曲接口（PBT接口）：一种具有增强可达性和灵活度的模块化折叠 manipulator 

**Authors**: Jianshu Zhou, Junda Huang, Boyuan Liang, Xiang Zhang, Xin Ma, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2503.05057)  

**Abstract**: Robotic manipulators, traditionally designed with classical joint-link articulated structures, excel in industrial applications but face challenges in human-centered and general-purpose tasks requiring greater dexterity and adaptability. Addressing these limitations, we introduce the Prismatic-Bending Transformable (PBT) Joint, a novel design inspired by the scissors mechanism, enabling transformable kinematic chains. Each PBT joint module provides three degrees of freedom-bending, rotation, and elongation/contraction-allowing scalable and reconfigurable assemblies to form diverse kinematic configurations tailored to specific tasks. This innovative design surpasses conventional systems, delivering superior flexibility and performance across various applications. We present the design, modeling, and experimental validation of the PBT joint, demonstrating its integration into modular and foldable robotic arms. The PBT joint functions as a single SKU, enabling manipulators to be constructed entirely from standardized PBT joints without additional customized components. It also serves as a modular extension for existing systems, such as wrist modules, streamlining design, deployment, transportation, and maintenance. Three sizes-large, medium, and small-have been developed and integrated into robotic manipulators, highlighting their enhanced dexterity, reachability, and adaptability for manipulation tasks. This work represents a significant advancement in robotic design, offering scalable and efficient solutions for dynamic and unstructured environments. 

**Abstract (ZH)**: 普ismatic-Bending Transformable (PBT) 关节: 一种受剪刀机制启发的创新设计及其在可重构机器人 manipulator 中的应用 

---
# A Convex Formulation of Material Points and Rigid Bodies with GPU-Accelerated Async-Coupling for Interactive Simulation 

**Title (ZH)**: 基于GPU加速异步耦合的材料点与刚体凸规划表示的交互模拟 

**Authors**: Chang Yu, Wenxin Du, Zeshun Zong, Alejandro Castro, Chenfanfu Jiang, Xuchen Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.05046)  

**Abstract**: We present a novel convex formulation that weakly couples the Material Point Method (MPM) with rigid body dynamics through frictional contact, optimized for efficient GPU parallelization. Our approach features an asynchronous time-splitting scheme to integrate MPM and rigid body dynamics under different time step sizes. We develop a globally convergent quasi-Newton solver tailored for massive parallelization, achieving up to 500x speedup over previous convex formulations without sacrificing stability. Our method enables interactive-rate simulations of robotic manipulation tasks with diverse deformable objects including granular materials and cloth, with strong convergence guarantees. We detail key implementation strategies to maximize performance and validate our approach through rigorous experiments, demonstrating superior speed, accuracy, and stability compared to state-of-the-art MPM simulators for robotics. We make our method available in the open-source robotics toolkit, Drake. 

**Abstract (ZH)**: 我们提出了一种新型凸优化公式，通过摩擦接触弱化地将物质点方法（MPM）与刚体动力学耦合，适用于高效的GPU并行化。该方法采用异步时间分裂方案，在不同的时间步长下集成MPM和刚体动力学。我们开发了一种全局收敛的拟牛顿求解器，适用于大规模并行化，比之前没有牺牲稳定性的凸优化公式快高达500倍。该方法允许以交互速率模拟具有多种变形物体（包括颗粒材料和布料）的机器人操作任务，并具有强大的收敛保证。我们详细介绍了关键实现策略以最大化性能，并通过严格的实验验证了该方法，表明与最先进的MPM模拟器相比，我们的方法在速度、准确性和稳定性方面表现出优越性。我们将在开源机器人工具包Drake中提供该方法。 

---
# Ergodic Exploration over Meshable Surfaces 

**Title (ZH)**: 可遍历网格化曲面的探索 

**Authors**: Dayi Dong, Albert Xu, Geordan Gutow, Howie Choset, Ian Abraham  

**Link**: [PDF](https://arxiv.org/pdf/2503.05026)  

**Abstract**: Robotic search and rescue, exploration, and inspection require trajectory planning across a variety of domains. A popular approach to trajectory planning for these types of missions is ergodic search, which biases a trajectory to spend time in parts of the exploration domain that are believed to contain more information. Most prior work on ergodic search has been limited to searching simple surfaces, like a 2D Euclidean plane or a sphere, as they rely on projecting functions defined on the exploration domain onto analytically obtained Fourier basis functions. In this paper, we extend ergodic search to any surface that can be approximated by a triangle mesh. The basis functions are approximated through finite element methods on a triangle mesh of the domain. We formally prove that this approximation converges to the continuous case as the mesh approximation converges to the true domain. We demonstrate that on domains where analytical basis functions are available (plane, sphere), the proposed method obtains equivalent results, and while on other domains (torus, bunny, wind turbine), the approach is versatile enough to still search effectively. Lastly, we also compare with an existing ergodic search technique that can handle complex domains and show that our method results in a higher quality exploration. 

**Abstract (ZH)**: 机器人搜索与救援、探索和检查任务需要在多种领域进行路径规划。本文将遍历搜索方法扩展到可以由三角网格近似的任何曲面。基函数通过领域三角网格上的有限元方法进行近似。我们正式证明，在网格逼近趋向真领域时，该近似收敛到连续情况。我们在可获取解析基函数的领域（平面、球面）上验证了所提方法获得等效结果，并在其他领域（环面、兔形模型、风力涡轮机）上展示了方法的灵活性和有效性。最后，我们将我们的方法与一种现有的适用于复杂领域的遍历搜索技术进行了比较，结果表明我们的方法在探索质量上更具优越性。 

---
# GRIP: A General Robotic Incremental Potential Contact Simulation Dataset for Unified Deformable-Rigid Coupled Grasping 

**Title (ZH)**: GRIP: 一种通用的机器人增量潜在接触模拟数据集，用于统一的可变形-刚性耦合抓取 

**Authors**: Siyu Ma, Wenxin Du, Chang Yu, Ying Jiang, Zeshun Zong, Tianyi Xie, Yunuo Chen, Yin Yang, Xuchen Han, Chenfanfu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05020)  

**Abstract**: Grasping is fundamental to robotic manipulation, and recent advances in large-scale grasping datasets have provided essential training data and evaluation benchmarks, accelerating the development of learning-based methods for robust object grasping. However, most existing datasets exclude deformable bodies due to the lack of scalable, robust simulation pipelines, limiting the development of generalizable models for compliant grippers and soft manipulands. To address these challenges, we present GRIP, a General Robotic Incremental Potential contact simulation dataset for universal grasping. GRIP leverages an optimized Incremental Potential Contact (IPC)-based simulator for multi-environment data generation, achieving up to 48x speedup while ensuring efficient, intersection- and inversion-free simulations for compliant grippers and deformable objects. Our fully automated pipeline generates and evaluates diverse grasp interactions across 1,200 objects and 100,000 grasp poses, incorporating both soft and rigid grippers. The GRIP dataset enables applications such as neural grasp generation and stress field prediction. 

**Abstract (ZH)**: GRIP：通用机器人增量潜力接触仿真数据集及其在通用抓取中的应用 

---
# MarsLGPR: Mars Rover Localization with Ground Penetrating Radar 

**Title (ZH)**: 火星LGPR：火星车地面穿透雷达定位 

**Authors**: Anja Sheppard, Katherine A. Skinner  

**Link**: [PDF](https://arxiv.org/pdf/2503.04944)  

**Abstract**: In this work, we propose the use of Ground Penetrating Radar (GPR) for rover localization on Mars. Precise pose estimation is an important task for mobile robots exploring planetary surfaces, as they operate in GPS-denied environments. Although visual odometry provides accurate localization, it is computationally expensive and can fail in dim or high-contrast lighting. Wheel encoders can also provide odometry estimation, but are prone to slipping on the sandy terrain encountered on Mars. Although traditionally a scientific surveying sensor, GPR has been used on Earth for terrain classification and localization through subsurface feature matching. The Perseverance rover and the upcoming ExoMars rover have GPR sensors already equipped to aid in the search of water and mineral resources. We propose to leverage GPR to aid in Mars rover localization. Specifically, we develop a novel GPR-based deep learning model that predicts 1D relative pose translation. We fuse our GPR pose prediction method with inertial and wheel encoder data in a filtering framework to output rover localization. We perform experiments in a Mars analog environment and demonstrate that our GPR-based displacement predictions both outperform wheel encoders and improve multi-modal filtering estimates in high-slip environments. Lastly, we present the first dataset aimed at GPR-based localization in Mars analog environments, which will be made publicly available upon publication. 

**Abstract (ZH)**: 基于地面穿透雷达的火星探测车定位方法研究 

---
# SAFE-TAXI: A Hierarchical Multi-UAS Safe Auto-Taxiing Framework with Runtime Safety Assurance and Conflict Resolution 

**Title (ZH)**: SAFE-TAXI：一种具有运行时安全保证和冲突解决的分层多无人飞行器自动 Taxiing 框架 

**Authors**: Kartik A. Pant, Li-Yu Lin, Worawis Sribunma, Sabine Brunswicker, James M. Goppert, Inseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04942)  

**Abstract**: We present a hierarchical safe auto-taxiing framework to enhance the automated ground operations of multiple unmanned aircraft systems (multi-UAS). The auto-taxiing problem becomes particularly challenging due to (i) unknown disturbances, such as crosswind affecting the aircraft dynamics, (ii) taxiway incursions due to unplanned obstacles, and (iii) spatiotemporal conflicts at the intersections between multiple entry points in the taxiway. To address these issues, we propose a hierarchical framework, i.e., SAFE-TAXI, combining centralized spatiotemporal planning with decentralized MPC-CBF-based control to safely navigate the aircraft through the taxiway while avoiding intersection conflicts and unplanned obstacles (e.g., other aircraft or ground vehicles). Our proposed framework decouples the auto-taxiing problem temporally into conflict resolution and motion planning, respectively. Conflict resolution is handled in a centralized manner by computing conflict-aware reference trajectories for each aircraft. In contrast, safety assurance from unplanned obstacles is handled by an MPC-CBF-based controller implemented in a decentralized manner. We demonstrate the effectiveness of our proposed framework through numerical simulations and experimentally validate it using Night Vapor, a small-scale fixed-wing test platform. 

**Abstract (ZH)**: 一种分层安全自主滑行框架：增强多无人机系统（multi-UAS）的自主地面操作 

---
# Neural Configuration-Space Barriers for Manipulation Planning and Control 

**Title (ZH)**: 基于神经网络的配置空间障碍物在操作规划与控制中的应用 

**Authors**: Kehan Long, Ki Myung Brian Lee, Nikola Raicevic, Niyas Attasseri, Melvin Leok, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2503.04929)  

**Abstract**: Planning and control for high-dimensional robot manipulators in cluttered, dynamic environments require both computational efficiency and robust safety guarantees. Inspired by recent advances in learning configuration-space distance functions (CDFs) as robot body representations, we propose a unified framework for motion planning and control that formulates safety constraints as CDF barriers. A CDF barrier approximates the local free configuration space, substantially reducing the number of collision-checking operations during motion planning. However, learning a CDF barrier with a neural network and relying on online sensor observations introduce uncertainties that must be considered during control synthesis. To address this, we develop a distributionally robust CDF barrier formulation for control that explicitly accounts for modeling errors and sensor noise without assuming a known underlying distribution. Simulations and hardware experiments on a 6-DoF xArm manipulator show that our neural CDF barrier formulation enables efficient planning and robust real-time safe control in cluttered and dynamic environments, relying only on onboard point-cloud observations. 

**Abstract (ZH)**: 高维度机器人 manipulator 在复杂动态环境中的规划与控制需要高效性和稳健的安全保证。受学习配置空间距离函数(CDF)作为机器人体表征的recent进展启发，我们提出了一种统一框架，将安全约束形式化为CDF拦阻器。CDF拦阻器近似局部自由配置空间，大幅减少了运动规划中的碰撞检测操作。然而，使用神经网络学习CDF拦阻器并依赖于在线传感器观测引入了不确定性，必须在控制综合时予以考虑。为此，我们开发了一种分布鲁棒的CDF拦阻器形式化方法，在不假设已知底层分布的情况下，显式考虑建模误差和传感器噪声。仿真和基于xArm六自由度 manipulator 的硬件实验表明，我们的神经CDF拦阻器形式化方法能够在复杂动态环境中实现高效的规划和实时稳健的安全控制，仅依赖于板载点云观测。 

---
