# Robot Drummer: Learning Rhythmic Skills for Humanoid Drumming 

**Title (ZH)**: 机器人鼓手：学习类人鼓击节奏技能 

**Authors**: Asad Ali Shahid, Francesco Braghin, Loris Roveda  

**Link**: [PDF](https://arxiv.org/pdf/2507.11498)  

**Abstract**: Humanoid robots have seen remarkable advances in dexterity, balance, and locomotion, yet their role in expressive domains, such as music performance, remains largely unexplored. Musical tasks, like drumming, present unique challenges, including split-second timing, rapid contacts, and multi-limb coordination over pieces lasting minutes. In this paper, we introduce Robot Drummer, a humanoid system capable of expressive, high-precision drumming across a diverse repertoire of songs. We formulate humanoid drumming as sequential fulfillment of timed-contacts and transform drum scores in to a Rhythmic Contact Chain. To handle the long-horizon nature of musical performance, we decompose each piece into fixed-length segments and train a single policy across all segments in parallel using reinforcement learning. Through extensive experiments on over thirty popular rock, metal, and jazz tracks, our results demonstrate that Robot Drummer consistently achieves high F1 scores. The learned behaviors exhibit emergent human-like drumming strategies, such as cross-arm strikes, and adaptive sticks assignments, demonstrating the potential of reinforcement learning to bring humanoid robots into the domain of creative musical performance. Project page: \href{this https URL}{this http URL} 

**Abstract (ZH)**: 人类机器人在灵巧性、平衡性和移动性方面取得了显著进步，但在像音乐表演这样的表现领域中的作用仍然 largely unexplored。音乐任务，例如打鼓，提出了独特的挑战，包括毫秒级的节拍精度、快速接触以及多肢体在分钟级曲子中的协调。在本文中，我们介绍了机器人鼓手（Robot Drummer），这是一个能够在多种歌曲曲目中执行表达性、高精度打鼓的人形系统。我们将人形打鼓表述为按时间顺序完成接触任务，并将鼓谱转换为节律接触链。为了应对音乐表演的长时间跨度特性，我们将每首曲子分解为固定长度的段落，并利用强化学习在并行训练过程中训练一个单一的策略。通过对三十多首流行摇滚、金属和爵士乐曲的广泛实验，我们的结果表明，机器人鼓手能够一致地实现高F1分数。学习到的行为表现出类似于人类的鼓击策略，如交叉臂击打和自适应的棒分配，这展示了强化学习将人形机器人引入创意音乐表演领域的潜力。项目页面：[这个链接](这个链接)。 

---
# LF: Online Multi-Robot Path Planning Meets Optimal Trajectory Control 

**Title (ZH)**: LF: 在线多机器人路径规划与最优轨迹控制 

**Authors**: Ajay Shankar, Keisuke Okumura, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2507.11464)  

**Abstract**: We propose a multi-robot control paradigm to solve point-to-point navigation tasks for a team of holonomic robots with access to the full environment information. The framework invokes two processes asynchronously at high frequency: (i) a centralized, discrete, and full-horizon planner for computing collision- and deadlock-free paths rapidly, leveraging recent advances in multi-agent pathfinding (MAPF), and (ii) dynamics-aware, robot-wise optimal trajectory controllers that ensure all robots independently follow their assigned paths reliably. This hierarchical shift in planning representation from (i) discrete and coupled to (ii) continuous and decoupled domains enables the framework to maintain long-term scalable motion synthesis. As an instantiation of this idea, we present LF, which combines a fast state-of-the-art MAPF solver (LaCAM), and a robust feedback control stack (Freyja) for executing agile robot maneuvers. LF provides a robust and versatile mechanism for lifelong multi-robot navigation even under asynchronous and partial goal updates, and adapts to dynamic workspaces simply by quick replanning. We present various multirotor and ground robot demonstrations, including the deployment of 15 real multirotors with random, consecutive target updates while a person walks through the operational workspace. 

**Abstract (ZH)**: 我们提出了一种多机器人控制范式，用于具有全域环境信息的一组全向机器人的点对点导航任务。该框架以高频率异步调用两个过程：(i) 中心化、离散、全视野的规划器快速计算无碰撞和无死锁路径，充分利用多智能体路径寻找（MAPF）领域的最新进展；(ii) 动力学意识的、针对每个机器人的最优轨迹控制器，确保所有机器人独立可靠地遵循其分配的路径。这种从(i) 离散和耦合到(ii) 连续和解耦规划表示层次的转变，使框架能够保持长期可扩展的运动合成。作为这一思想的具体实例，我们介绍了LF，它结合了快速最先进的MAPF求解器（LaCAM）和鲁棒的反馈控制堆栈（Freyja），用于执行敏捷的机器人机动。LF为在异步和部分目标更新下提供了稳健且多功能的终身多机器人导航机制，并通过快速重新规划简单适应动态工作空间。我们展示了各种多旋翼和地面机器人的演示，包括在人员穿越操作工作空间时，部署15个实时多旋翼并随机更新目标的场景。 

---
# Multi-IMU Sensor Fusion for Legged Robots 

**Title (ZH)**: 多IMU传感器融合在腿式机器人中的应用 

**Authors**: Shuo Yang, John Z. Zhang, Ibrahima Sory Sow, Zachary Manchester  

**Link**: [PDF](https://arxiv.org/pdf/2507.11447)  

**Abstract**: This paper presents a state-estimation solution for legged robots that uses a set of low-cost, compact, and lightweight sensors to achieve low-drift pose and velocity estimation under challenging locomotion conditions. The key idea is to leverage multiple inertial measurement units on different links of the robot to correct a major error source in standard proprioceptive odometry. We fuse the inertial sensor information and joint encoder measurements in an extended Kalman filter, then combine the velocity estimate from this filter with camera data in a factor-graph-based sliding-window estimator to form a visual-inertial-leg odometry method. We validate our state estimator through comprehensive theoretical analysis and hardware experiments performed using real-world robot data collected during a variety of challenging locomotion tasks. Our algorithm consistently achieves minimal position deviation, even in scenarios involving substantial ground impact, foot slippage, and sudden body rotations. A C++ implementation, along with a large-scale dataset, is available at this https URL. 

**Abstract (ZH)**: 本文提出了一种针对腿式机器人状态估计的解决方案，该方案利用一组低成本、紧凑型和轻量级传感器，在挑战性运动条件下实现低漂移姿态和速度估计。核心思想是利用机器人不同链接上的多个惯性测量单元来修正标准本体感觉里程计的主要误差源。我们通过扩展卡尔曼滤波器融合惯性传感器信息和关节编码器测量值，然后将该滤波器的速度估计值与基于因子图的滑动窗口估计算法结合，形成一种视觉-惯性-腿式里程计方法。我们通过全面的理论分析和使用多种挑战性运动任务中收集的真实机器人数据进行的硬件实验，验证了我们状态估计器的有效性。即使在涉及显著地面冲击、脚底打滑和身体突然旋转的场景中，该算法也能够实现最小的位置偏差。相关C++实现及大规模数据集可从此链接获取。 

---
# From Production Logistics to Smart Manufacturing: The Vision for a New RoboCup Industrial League 

**Title (ZH)**: 从生产物流到智能制造：新机器人世界杯工业联赛的愿景 

**Authors**: Supun Dissanayaka, Alexander Ferrein, Till Hofmann, Kosuke Nakajima, Mario Sanz-Lopez, Jesus Savage, Daniel Swoboda, Matteo Tschesche, Wataru Uemura, Tarik Viehmann, Shohei Yasuda  

**Link**: [PDF](https://arxiv.org/pdf/2507.11402)  

**Abstract**: The RoboCup Logistics League is a RoboCup competition in a smart factory scenario that has focused on task planning, job scheduling, and multi-agent coordination. The focus on production logistics allowed teams to develop highly competitive strategies, but also meant that some recent developments in the context of smart manufacturing are not reflected in the competition, weakening its relevance over the years. In this paper, we describe the vision for the RoboCup Smart Manufacturing League, a new competition designed as a larger smart manufacturing scenario, reflecting all the major aspects of a modern factory. It will consist of several tracks that are initially independent but gradually combined into one smart manufacturing scenario. The new tracks will cover industrial robotics challenges such as assembly, human-robot collaboration, and humanoid robotics, but also retain a focus on production logistics. We expect the reenvisioned competition to be more attractive to newcomers and well-tried teams, while also shifting the focus to current and future challenges of industrial robotics. 

**Abstract (ZH)**: RoboCup智能制造联盟：一个新的涵盖现代工厂各方面挑战的竞赛愿景 

---
# Acting and Planning with Hierarchical Operational Models on a Mobile Robot: A Study with RAE+UPOM 

**Title (ZH)**: 基于RAE+UPOM的移动机器人分层操作模型行动与规划研究 

**Authors**: Oscar Lima, Marc Vinci, Sunandita Patra, Sebastian Stock, Joachim Hertzberg, Martin Atzmueller, Malik Ghallab, Dana Nau, Paolo Traverso  

**Link**: [PDF](https://arxiv.org/pdf/2507.11345)  

**Abstract**: Robotic task execution faces challenges due to the inconsistency between symbolic planner models and the rich control structures actually running on the robot. In this paper, we present the first physical deployment of an integrated actor-planner system that shares hierarchical operational models for both acting and planning, interleaving the Reactive Acting Engine (RAE) with an anytime UCT-like Monte Carlo planner (UPOM). We implement RAE+UPOM on a mobile manipulator in a real-world deployment for an object collection task. Our experiments demonstrate robust task execution under action failures and sensor noise, and provide empirical insights into the interleaved acting-and-planning decision making process. 

**Abstract (ZH)**: 机器人任务执行由于符号规划模型与机器人实际运行的丰富控制结构之间的一致性问题而面临挑战。本文提出了第一个将层级操作模型同时用于执行和规划的集成执行-规划系统的真实物理部署，该系统交替使用反应性执行引擎（RAE）和一个类似UCT的可中断蒙特卡洛规划器（UPOM）。我们在一个移动 manipulator 上为一个物体收集任务实现 RAE+UPOM，并进行了实证研究，表明在动作失败和传感器噪声条件下的任务执行具有鲁棒性，并提供了交替执行和规划决策过程的经验见解。 

---
# Development of an Autonomous Mobile Robotic System for Efficient and Precise Disinfection 

**Title (ZH)**: 自主移动机器人系统的发展以实现高效精准的消毒作业 

**Authors**: Ting-Wei Ou, Jia-Hao Jiang, Guan-Lin Huang, Kuu-Young Young  

**Link**: [PDF](https://arxiv.org/pdf/2507.11270)  

**Abstract**: The COVID-19 pandemic has severely affected public health, healthcare systems, and daily life, especially amid resource shortages and limited workers. This crisis has underscored the urgent need for automation in hospital environments, particularly disinfection, which is crucial to controlling virus transmission and improving the safety of healthcare personnel and patients. Ultraviolet (UV) light disinfection, known for its high efficiency, has been widely adopted in hospital settings. However, most existing research focuses on maximizing UV coverage while paying little attention to the impact of human activity on virus distribution. To address this issue, we propose a mobile robotic system for UV disinfection focusing on the virus hotspot. The system prioritizes disinfection in high-risk areas and employs an approach for optimized UV dosage to ensure that all surfaces receive an adequate level of UV exposure while significantly reducing disinfection time. It not only improves disinfection efficiency but also minimizes unnecessary exposure in low-risk areas. In two representative hospital scenarios, our method achieves the same disinfection effectiveness while reducing disinfection time by 30.7% and 31.9%, respectively. The video of the experiment is available at: this https URL. 

**Abstract (ZH)**: COVID-19 pandemic对公共健康、医疗系统和日常生活造成了严重的影响，尤其是在资源短缺和人手有限的情况下。这场危机凸显了医院环境中自动化需求的紧迫性，尤其是消毒工作，这在控制病毒传播和提高医疗人员和患者的安全方面至关重要。紫外（UV）光消毒以其高效的特性在医院环境中被广泛应用。然而，现有大多数研究侧重于最大化UV覆盖范围，而很少关注人类活动对病毒分布的影响。为解决这一问题，我们提出了一种移动机器人系统以针对病毒热点区域进行UV消毒。该系统优先对高风险区域进行消毒，并采用优化的UV剂量方法，以确保所有表面接受到足够的UV照射，同时显著减少消毒时间。它不仅提高了消毒效率，还最大限度地减少了低风险区域不必要的暴露。在两个代表性医院场景中，我们的方法在分别减少30.7%和31.9%的消毒时间的同时实现了相同的消毒效果。实验视频可在此网址查看：this https URL。 

---
# Comparison of Localization Algorithms between Reduced-Scale and Real-Sized Vehicles Using Visual and Inertial Sensors 

**Title (ZH)**: 使用视觉和惯性传感器在缩小比例模型和真实尺寸车辆之间比较定位算法性能 

**Authors**: Tobias Kern, Leon Tolksdorf, Christian Birkner  

**Link**: [PDF](https://arxiv.org/pdf/2507.11241)  

**Abstract**: Physically reduced-scale vehicles are emerging to accelerate the development of advanced automated driving functions. In this paper, we investigate the effects of scaling on self-localization accuracy with visual and visual-inertial algorithms using cameras and an inertial measurement unit (IMU). For this purpose, ROS2-compatible visual and visual-inertial algorithms are selected, and datasets are chosen as a baseline for real-sized vehicles. A test drive is conducted to record data of reduced-scale vehicles. We compare the selected localization algorithms, OpenVINS, VINS-Fusion, and RTAB-Map, in terms of their pose accuracy against the ground-truth and against data from real-sized vehicles. When comparing the implementation of the selected localization algorithms to real-sized vehicles, OpenVINS has the lowest average localization error. Although all selected localization algorithms have overlapping error ranges, OpenVINS also performs best when applied to a reduced-scale vehicle. When reduced-scale vehicles were compared to real-sized vehicles, minor differences were found in translational vehicle motion estimation accuracy. However, no significant differences were found when comparing the estimation accuracy of rotational vehicle motion, allowing RSVRs to be used as testing platforms for self-localization algorithms. 

**Abstract (ZH)**: 缩小比例的车辆正加速高级自动驾驶功能的发展。本文通过使用摄像头和惯性测量单元（IMU）的研究，考察缩放效应对基于视觉和视觉-惯性算法自定位精度的影响。选用ROS2兼容的视觉和视觉-惯性算法，并选用实际大小车辆的数据集作为基准。进行测试驱动以记录缩小比例车辆的数据。从姿态准确性角度比较选定的定位算法OpenVINS、VINS-Fusion和RTAB-Map，以及与真实大小车辆数据和地面真实值的比较。在将选定的定位算法实施到实际大小车辆进行比较时，OpenVINS具有最低的平均定位误差。尽管所有选定的定位算法具有重叠的误差范围，但当应用于缩小比例车辆时，OpenVINS表现最佳。在将缩小比例车辆与实际大小车辆进行比较时，发现平移车辆运动估计准确性存在轻微差异，但在旋转车辆运动估计准确性方面未发现显著差异，允许使用RSVR作为自定位算法的测试平台。 

---
# MPC-based Coarse-to-Fine Motion Planning for Robotic Object Transportation in Cluttered Environments 

**Title (ZH)**: 基于Model Predictive Control的面向杂乱环境的机器人物体运输粗细motion planning 

**Authors**: Chen Cai, Ernesto Dickel Saraiva, Ya-jun Pan, Steven Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11211)  

**Abstract**: This letter presents a novel coarse-to-fine motion planning framework for robotic manipulation in cluttered, unmodeled environments. The system integrates a dual-camera perception setup with a B-spline-based model predictive control (MPC) scheme. Initially, the planner generates feasible global trajectories from partial and uncertain observations. As new visual data are incrementally fused, both the environment model and motion planning are progressively refined. A vision-based cost function promotes target-driven exploration, while a refined kernel-perceptron collision detector enables efficient constraint updates for real-time planning. The framework accommodates closed-chain kinematics and supports dynamic replanning. Experiments on a multi-arm platform validate its robustness and adaptability under uncertainties and clutter. 

**Abstract (ZH)**: 本信提出了一种用于杂乱未建模环境中机器人操作的自上而下精细优化运动规划框架。该系统结合了基于B样条的模型预测控制（MPC）方案和双摄像头感知设置。初始阶段，规划器从部分和不确定的观测中生成可行的全局轨迹。随着逐步融合新视觉数据，环境模型和运动规划也随之逐步 refinement。基于视觉的代价函数促进目标导向的探索，而细化的核感知碰撞检测器则能高效更新约束以支持实时规划。该框架能够处理闭链运动学，并支持动态重规划。实验在多臂平台上的验证了其在不确定性与杂乱条件下的健壮性和适应性。 

---
# A Robust Controller based on Gaussian Processes for Robotic Manipulators with Unknown Uncertainty 

**Title (ZH)**: 基于高斯过程的鲁棒控制器：用于具有未知不确定性的机器人 manipulator 控制 

**Authors**: Giulio Giacomuzzo, Mohamed Abdelwahab, Marco Calì, Alberto Dalla Libera, Ruggero Carli  

**Link**: [PDF](https://arxiv.org/pdf/2507.11170)  

**Abstract**: In this paper, we propose a novel learning-based robust feedback linearization strategy to ensure precise trajectory tracking for an important family of Lagrangian systems. We assume a nominal knowledge of the dynamics is given but no a-priori bounds on the model mismatch are available. In our approach, the key ingredient is the adoption of a regression framework based on Gaussian Processes (GPR) to estimate the model mismatch. This estimate is added to the outer loop of a classical feedback linearization scheme based on the nominal knowledge available. Then, to compensate for the residual uncertainty, we robustify the controller including an additional term whose size is designed based on the variance provided by the GPR framework. We proved that, with high probability, the proposed scheme is able to guarantee asymptotic tracking of a desired trajectory. We tested numerically our strategy on a 2 degrees of freedom planar robot. 

**Abstract (ZH)**: 本文提出了一种基于学习的鲁棒反馈线性化策略，以确保对拉格朗日系统重要一类的精确轨迹跟踪。我们假设已给定系统的名义动力学知识，但没有关于模型不匹配先验界的信息。在我们的方法中，关键成分是采用基于高斯过程（GPR）的回归框架来估计模型不匹配。此估计值被添加到基于可用名义知识的经典反馈线性化方案的外环中。然后，为了抵消剩余的不确定性，我们使控制器具有鲁棒性，并增加了一个额外的项，其大小是基于GPR框架提供的方差进行设计的。我们证明，有高度概率性，所提出的方案能够保证对期望轨迹的渐近跟踪。我们在一个两自由度平面机器人上进行了数值测试。 

---
# Force-Based Viscosity and Elasticity Measurements for Material Biomechanical Characterisation with a Collaborative Robotic Arm 

**Title (ZH)**: 基于力反馈的材料本构特性表征中黏弹性的协作机器人测量方法 

**Authors**: Luca Beber, Edoardo Lamon, Giacomo Moretti, Matteo Saveriano, Luca Fambri, Luigi Palopoli, Daniele Fontanelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.11133)  

**Abstract**: Diagnostic activities, such as ultrasound scans and palpation, are relatively low-cost. They play a crucial role in the early detection of health problems and in assessing their progression. However, they are also error-prone activities, which require highly skilled medical staff. The use of robotic solutions can be key to decreasing the inherent subjectivity of the results and reducing the waiting list. For a robot to perform palpation or ultrasound scans, it must effectively manage physical interactions with the human body, which greatly benefits from precise estimation of the patient's tissue biomechanical properties. This paper assesses the accuracy and precision of a robotic system in estimating the viscoelastic parameters of various materials, including some tests on ex vivo tissues as a preliminary proof-of-concept demonstration of the method's applicability to biological samples. The measurements are compared against a ground truth derived from silicone specimens with different viscoelastic properties, characterised using a high-precision instrument. Experimental results show that the robotic system's accuracy closely matches the ground truth, increasing confidence in the potential use of robots for such clinical applications. 

**Abstract (ZH)**: 机器人系统在估计多种材料包括离体组织的粘弹性参数方面的准确性和精确性评估：初步概念验证表明其在生物样本中的应用可行性 

---
# Closed Form Time Derivatives of the Equations of Motion of Rigid Body Systems 

**Title (ZH)**: 刚体系统运动方程的闭式时间导数 

**Authors**: Andreas Mueller, Shivesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.11076)  

**Abstract**: Derivatives of equations of motion(EOM) describing the dynamics of rigid body systems are becoming increasingly relevant for the robotics community and find many applications in design and control of robotic systems. Controlling robots, and multibody systems comprising elastic components in particular, not only requires smooth trajectories but also the time derivatives of the control forces/torques, hence of the EOM. This paper presents the time derivatives of the EOM in closed form up to second-order as an alternative formulation to the existing recursive algorithms for this purpose, which provides a direct insight into the structure of the derivatives. The Lie group formulation for rigid body systems is used giving rise to very compact and easily parameterized equations. 

**Abstract (ZH)**: 描述刚体系统动力学的运动方程的时间导数在机器人技术领域越来越受到关注，并在机器人系统的设计与控制中找到许多应用。本文以闭式形式给出了运动方程的一阶和二阶时间导数，作为一种替代现有的递归算法的方法，这提供了一种直接洞察导数结构的方式。使用刚体系统的李群形式化描述导致了非常紧凑且易于参数化的方程。 

---
# Enhancing Autonomous Manipulator Control with Human-in-loop for Uncertain Assembly Environments 

**Title (ZH)**: 基于人为回路增强不确定装配环境中的自主 manipulator 控制 

**Authors**: Ashutosh Mishra, Shreya Santra, Hazal Gozbasi, Kentaro Uno, Kazuya Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2507.11006)  

**Abstract**: This study presents an advanced approach to enhance robotic manipulation in uncertain and challenging environments, with a focus on autonomous operations augmented by human-in-the-loop (HITL) control for lunar missions. By integrating human decision-making with autonomous robotic functions, the research improves task reliability and efficiency for space applications. The key task addressed is the autonomous deployment of flexible solar panels using an extendable ladder-like structure and a robotic manipulator with real-time feedback for precision. The manipulator relays position and force-torque data, enabling dynamic error detection and adaptive control during deployment. To mitigate the effects of sinkage, variable payload, and low-lighting conditions, efficient motion planning strategies are employed, supplemented by human control that allows operators to intervene in ambiguous scenarios. Digital twin simulation enhances system robustness by enabling continuous feedback, iterative task refinement, and seamless integration with the deployment pipeline. The system has been tested to validate its performance in simulated lunar conditions and ensure reliability in extreme lighting, variable terrain, changing payloads, and sensor limitations. 

**Abstract (ZH)**: 本研究提出了一种先进的方法，旨在在不确定和具挑战性的环境中提升机器人的操作能力，并重点关注结合有人参与环回控制（HITL）的自主操作在月球任务中的应用。通过将人类的决策与自主机器人功能相结合，该研究提高了空间应用中的任务可靠性和效率。主要任务是利用可伸展的梯形结构和具有实时反馈的机器人操作臂，实现柔性太阳电池板的自主部署。操作臂传输位置和力-扭矩数据，使部署过程中能够动态检测错误并进行自适应控制。为了应对下沉效应、可变载荷和低光照条件，研究采用了高效的运动规划策略，并结合了人类控制，以便在模棱两可的情境中进行干预。通过数字孪生模拟，该系统能够提供持续反馈、迭代的任务精细化并无缝集成到部署流程中。该系统已在模拟月球条件下进行了测试，以验证其在极端光照、可变地形、载荷变化和传感器限制条件下的可靠性。 

---
# Uncertainty Aware Mapping for Vision-Based Underwater Robots 

**Title (ZH)**: 基于视觉的水下机器人不确定性感知映射 

**Authors**: Abhimanyu Bhowmik, Mohit Singh, Madhushree Sannigrahi, Martin Ludvigsen, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2507.10991)  

**Abstract**: Vision-based underwater robots can be useful in inspecting and exploring confined spaces where traditional sensors and preplanned paths cannot be followed. Sensor noise and situational change can cause significant uncertainty in environmental representation. Thus, this paper explores how to represent mapping inconsistency in vision-based sensing and incorporate depth estimation confidence into the mapping framework. The scene depth and the confidence are estimated using the RAFT-Stereo model and are integrated into a voxel-based mapping framework, Voxblox. Improvements in the existing Voxblox weight calculation and update mechanism are also proposed. Finally, a qualitative analysis of the proposed method is performed in a confined pool and in a pier in the Trondheim fjord. Experiments using an underwater robot demonstrated the change in uncertainty in the visualization. 

**Abstract (ZH)**: 基于视觉的水下机器人能够在传统传感器和预规划路径无法适用的受限空间内进行检查和探索。传感器噪声和情况变化会导致环境表示中的显著不确定性。因此，本文探讨了如何在基于视觉的传感中表示制图不一致性，并将深度估计置信度整合到制图框架中。使用RAFT-Stereo模型估计场景深度和置信度，并将它们集成到基于体素的制图框架Voxblox中。还提出了对现有Voxblox权重计算和更新机制的改进。最后，在特隆赫姆峡湾的游泳池和码头进行定性分析，实验结果表明 proposed方法在可视化中不确定性变化。 

---
# SMART-Merge Planner: A Safe Merging and Real-Time Motion Planner for Autonomous Highway On-Ramp Merging 

**Title (ZH)**: SMART-Merge 计划器：一种安全合并及实时运动规划器，应用于自主高速公路入口匝道合并 

**Authors**: Toktam Mohammadnejad, Jovin D'sa, Behdad Chalaki, Hossein Nourkhiz Mahjoub, Ehsan Moradi-Pari  

**Link**: [PDF](https://arxiv.org/pdf/2507.10968)  

**Abstract**: Merging onto a highway is a complex driving task that requires identifying a safe gap, adjusting speed, often interactions to create a merging gap, and completing the merge maneuver within a limited time window while maintaining safety and driving comfort. In this paper, we introduce a Safe Merging and Real-Time Merge (SMART-Merge) planner, a lattice-based motion planner designed to facilitate safe and comfortable forced merging. By deliberately adapting cost terms to the unique challenges of forced merging and introducing a desired speed heuristic, SMART-Merge planner enables the ego vehicle to merge successfully while minimizing the merge time. We verify the efficiency and effectiveness of the proposed merge planner through high-fidelity CarMaker simulations on hundreds of highway merge scenarios. Our proposed planner achieves the success rate of 100% as well as completes the merge maneuver in the shortest amount of time compared with the baselines, demonstrating our planner's capability to handle complex forced merge tasks and provide a reliable and robust solution for autonomous highway merge. The simulation result videos are available at this https URL. 

**Abstract (ZH)**: 基于格子的智能安全快速并线规划器（SMART-Merge）：应对复杂强制并线任务的实时并线规划 

---
# Unified Modeling and Structural Optimization of Multi-magnet Embedded Soft Continuum Robots for Enhanced Kinematic Performances 

**Title (ZH)**: 多磁体嵌入软连续机器人统一建模与结构优化以提升运动性能 

**Authors**: Zhiwei Wu, Jiahao Luo, Siyi Wei, Jinhui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10950)  

**Abstract**: This paper presents a unified modeling and optimization framework to enhance the kinematic performance of multi-magnet embedded soft continuum robots (MeSCRs). To this end, we establish a differentiable system formulation based on an extended pseudo-rigid-body model. This formulation enables analysis of the equilibrium well-posedness and the geometry of the induced configuration under magnetic actuation. In particular, we show that the maximum controllable degrees of freedom of a MeSCR equal twice the number of embedded magnets. We subsequently develop a structural optimization framework based on differential geometry that links classical kinematic measures (e.g., manipulability and dexterity) to the configuration of embedded magnets. The resulting optimization condition reveals that improving local performance requires structurally modulating the spectrum of the configuration space metric to counteract its distortion. Closed-form solutions for optimal magnet configurations are derived under representative conditions, and a gradient-based numerical method is proposed for general design scenarios. Simulation studies validate the effectiveness of the proposed framework. 

**Abstract (ZH)**: 一种统一建模与优化框架以提升多磁体嵌入软连续机器人(MeSCRs)的运动性能 

---
# Fast Non-Episodic Adaptive Tuning of Robot Controllers with Online Policy Optimization 

**Title (ZH)**: 基于在线策略优化的快速非情景自适应机器人控制器调整 

**Authors**: James A. Preiss, Fengze Xie, Yiheng Lin, Adam Wierman, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.10914)  

**Abstract**: We study online algorithms to tune the parameters of a robot controller in a setting where the dynamics, policy class, and optimality objective are all time-varying. The system follows a single trajectory without episodes or state resets, and the time-varying information is not known in advance. Focusing on nonlinear geometric quadrotor controllers as a test case, we propose a practical implementation of a single-trajectory model-based online policy optimization algorithm, M-GAPS,along with reparameterizations of the quadrotor state space and policy class to improve the optimization landscape. In hardware experiments,we compare to model-based and model-free baselines that impose artificial episodes. We show that M-GAPS finds near-optimal parameters more quickly, especially when the episode length is not favorable. We also show that M-GAPS rapidly adapts to heavy unmodeled wind and payload disturbances, and achieves similar strong improvement on a 1:6-scale Ackermann-steered car. Our results demonstrate the hardware practicality of this emerging class of online policy optimization that offers significantly more flexibility than classic adaptive control, while being more stable and data-efficient than model-free reinforcement learning. 

**Abstract (ZH)**: 我们研究在线算法以在动态、策略类和最优性目标均为时变的情况下调整机器人控制器的参数。系统遵循单一轨迹而无需分段或状态重置，且时变信息事先未知。以非线性几何多旋翼控制器为例，我们提出了一种实用的基于单一轨迹的在线策略优化算法M-GAPS的实现，并改进了多旋翼状态空间和策略类的参数化以优化优化场景。在硬件实验中，我们将M-GAPS与引入人工分段的基于模型和无模型基线进行比较。结果显示，M-GAPS能够更快地找到近最优参数，尤其是在分段长度不利时更为明显。我们还展示了M-GAPS能够迅速适应强烈的未建模风力和载荷干扰，并在1:6比例的Ackermann转向汽车上实现了类似的显著改进。我们的研究结果表明，这种新兴的在线策略优化方法在提供比经典自适应控制更多灵活性的同时，比无模型强化学习更具稳定性和数据效率。 

---
# Mixed Discrete and Continuous Planning using Shortest Walks in Graphs of Convex Sets 

**Title (ZH)**: 混合离散与连续规划：基于凸集图上最短路径的方法 

**Authors**: Savva Morozov, Tobia Marcucci, Bernhard Paus Graesdal, Alexandre Amice, Pablo A. Parrilo, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2507.10878)  

**Abstract**: We study the Shortest-Walk Problem (SWP) in a Graph of Convex Sets (GCS). A GCS is a graph where each vertex is paired with a convex program, and each edge couples adjacent programs via additional costs and constraints. A walk in a GCS is a sequence of vertices connected by edges, where vertices may be repeated. The length of a walk is given by the cumulative optimal value of the corresponding convex programs. To solve the SWP in GCS, we first synthesize a piecewise-quadratic lower bound on the problem's cost-to-go function using semidefinite programming. Then we use this lower bound to guide an incremental-search algorithm that yields an approximate shortest walk. We show that the SWP in GCS is a natural language for many mixed discrete-continuous planning problems in robotics, unifying problems that typically require specialized solutions while delivering high performance and computational efficiency. We demonstrate this through experiments in collision-free motion planning, skill chaining, and optimal control of hybrid systems. 

**Abstract (ZH)**: 我们研究凸集图（GCS）中的最短行走问题（SWP）。凸集图（GCS）是一种图形，其中每个顶点都对应一个凸规划，每条边通过附加的成本和约束将相邻的规划联系起来。在凸集图中的行走是一条由边连接的顶点序列，顶点可以重复。行走的长度是由相应凸规划的累积最优值给出的。为了解决GCS中的最短行走问题，我们首先使用半定规划合成功维二次下界问题的成本-剩下函数。然后，我们使用这个下界来引导一种增量搜索算法，以获得近似的最短行走。我们展示了GCS中的SWP是一种自然的语言，适用于许多机器人中的混合离散-连续规划问题，统一了通常需要专门解决方案的问题，同时提供高效性能和计算效率。我们通过在无碰撞运动规划、技能链动和混合系统最优控制方面的实验进行了演示。 

---
# GeoHopNet: Hopfield-Augmented Sparse Spatial Attention for Dynamic UAV Site Location Problem 

**Title (ZH)**: GeoHopNet：Hopfield强化稀疏空间注意力机制在动态无人机站点定位问题中的应用 

**Authors**: Jianing Zhi, Xinghua Li, Zidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.10636)  

**Abstract**: The rapid development of urban low-altitude unmanned aerial vehicle (UAV) economy poses new challenges for dynamic site selection of UAV landing points and supply stations. Traditional deep reinforcement learning methods face computational complexity bottlenecks, particularly with standard attention mechanisms, when handling large-scale urban-level location problems. This paper proposes GeoHopNet, a Hopfield-augmented sparse spatial attention network specifically designed for dynamic UAV site location problems. Our approach introduces four core innovations: (1) distance-biased multi-head attention mechanism that explicitly encodes spatial geometric information; (2) K-nearest neighbor sparse attention that reduces computational complexity from $O(N^2)$ to $O(NK)$; (3) a modern Hopfield external memory module; and (4) a memory regularization strategy. Experimental results demonstrate that GeoHopNet extends the boundary of solvable problem sizes. For large-scale instances with 1,000 nodes, where standard attention models become prohibitively slow (over 3 seconds per instance) and traditional solvers fail, GeoHopNet finds high-quality solutions (0.22\% optimality gap) in under 0.1 seconds. Compared to the state-of-the-art ADNet baseline on 100-node instances, our method improves solution quality by 22.2\% and is 1.8$\times$ faster. 

**Abstract (ZH)**: GeoHopNet：一种用于动态无人机着陆点和供给站选址问题的霍普菲尔德增强稀疏空间注意力网络 

---
