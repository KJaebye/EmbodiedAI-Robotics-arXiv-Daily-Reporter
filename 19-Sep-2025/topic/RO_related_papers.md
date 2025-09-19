# Parallel Simulation of Contact and Actuation for Soft Growing Robots 

**Title (ZH)**: 软生长机器人接触与动作的并行模拟 

**Authors**: Yitian Gao, Lucas Chen, Priyanka Bhovad, Sicheng Wang, Zachary Kingston, Laura H. Blumenschein  

**Link**: [PDF](https://arxiv.org/pdf/2509.15180)  

**Abstract**: Soft growing robots, commonly referred to as vine robots, have demonstrated remarkable ability to interact safely and robustly with unstructured and dynamic environments. It is therefore natural to exploit contact with the environment for planning and design optimization tasks. Previous research has focused on planning under contact for passively deforming robots with pre-formed bends. However, adding active steering to these soft growing robots is necessary for successful navigation in more complex environments. To this end, we develop a unified modeling framework that integrates vine robot growth, bending, actuation, and obstacle contact. We extend the beam moment model to include the effects of actuation on kinematics under growth and then use these models to develop a fast parallel simulation framework. We validate our model and simulator with real robot experiments. To showcase the capabilities of our framework, we apply our model in a design optimization task to find designs for vine robots navigating through cluttered environments, identifying designs that minimize the number of required actuators by exploiting environmental contacts. We show the robustness of the designs to environmental and manufacturing uncertainties. Finally, we fabricate an optimized design and successfully deploy it in an obstacle-rich environment. 

**Abstract (ZH)**: 软生长机器人，通常称为藤蔓机器人，已经展示了与未结构化和动态环境安全而 robust 地互动的能力。因此，利用与环境的接触来进行规划和设计优化是自然的选择。以往的研究集中于具有先构弯曲的被动变形机器人在接触下的规划。然而，要在更复杂的环境中成功导航，这些软生长机器人需要加入主动转向。为此，我们开发了一个统一的建模框架，整合了藤蔓机器人的生长、弯曲、驱动和障碍接触。我们扩展了梁矩模型，以包括在生长过程中驱动对运动学的影响，然后使用这些模型开发了一个快速并行仿真框架。我们通过实际的机器人实验验证了我们的模型和仿真器。为了展示该框架的能力，我们在一个设计优化任务中应用了我们的模型，以找到能够通过杂乱环境的藤蔓机器人设计，通过利用环境接触来最小化所需驱动器的数量。我们展示了这些设计对环境和制造不确定性具有鲁棒性。最后，我们制造了一个优化的设计，并成功地将其部署在障碍丰富的环境中。 

---
# Energy-Constrained Navigation for Planetary Rovers under Hybrid RTG-Solar Power 

**Title (ZH)**: 行星车在混合放射性同位素温差发电机-太阳能电源系统约束下的能量受限导航 

**Authors**: Tianxin Hu, Weixiang Guo, Ruimeng Liu, Xinhang Xu, Rui Qian, Jinyu Chen, Shenghai Yuan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.15062)  

**Abstract**: Future planetary exploration rovers must operate for extended durations on hybrid power inputs that combine steady radioisotope thermoelectric generator (RTG) output with variable solar photovoltaic (PV) availability. While energy-aware planning has been studied for aerial and underwater robots under battery limits, few works for ground rovers explicitly model power flow or enforce instantaneous power constraints. Classical terrain-aware planners emphasize slope or traversability, and trajectory optimization methods typically focus on geometric smoothness and dynamic feasibility, neglecting energy feasibility. We present an energy-constrained trajectory planning framework that explicitly integrates physics-based models of translational, rotational, and resistive power with baseline subsystem loads, under hybrid RTG-solar input. By incorporating both cumulative energy budgets and instantaneous power constraints into SE(2)-based polynomial trajectory optimization, the method ensures trajectories that are simultaneously smooth, dynamically feasible, and power-compliant. Simulation results on lunar-like terrain show that our planner generates trajectories with peak power within 0.55 percent of the prescribed limit, while existing methods exceed limits by over 17 percent. This demonstrates a principled and practical approach to energy-aware autonomy for long-duration planetary missions. 

**Abstract (ZH)**: 未来的行星探测漫游者必须在结合稳态放射性同位素热电动势发生器（RTG）输出与变异性太阳能光伏（PV）供应的混合电源输入下，进行延长时长的操作。虽然能量感知规划已经在受电池限制的空中和水下机器人中进行了研究，但对于地面漫游者而言，鲜有工作明确建模功率流或强制即时功率约束。传统的地形感知规划强调坡度或可通行性，而轨迹优化方法通常集中在几何光滑性和动态可行性上，忽略了能量可行性。我们提出了一种能量约束轨迹规划框架，该框架明确整合了平移、旋转和阻力功率的物理模型，以及基础子系统的负载，适用于混合RTG-太阳能输入。通过将累积能量预算和即时功率约束整合到SE(2)基于多项式的轨迹优化中，该方法保证了同时光滑、动态可行和功率合规的轨迹。在月球地形上的仿真结果表明，我们的规划器生成的轨迹峰值功率在规定限制的0.55%以内，而现有方法则超过限制17%以上。这证明了一种原则性和实用性的能量感知自主方法，适用于长时长行星任务。 

---
# Online Multi-Robot Coordination and Cooperation with Task Precedence Relationships 

**Title (ZH)**: 具有任务优先级关系的在线多机器人协同与合作 

**Authors**: Walker Gosrich, Saurav Agarwal, Kashish Garg, Siddharth Mayya, Matthew Malencia, Mark Yim, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.15052)  

**Abstract**: We propose a new formulation for the multi-robot task allocation problem that incorporates (a) complex precedence relationships between tasks, (b) efficient intra-task coordination, and (c) cooperation through the formation of robot coalitions. A task graph specifies the tasks and their relationships, and a set of reward functions models the effects of coalition size and preceding task performance. Maximizing task rewards is NP-hard; hence, we propose network flow-based algorithms to approximate solutions efficiently. A novel online algorithm performs iterative re-allocation, providing robustness to task failures and model inaccuracies to achieve higher performance than offline approaches. We comprehensively evaluate the algorithms in a testbed with random missions and reward functions and compare them to a mixed-integer solver and a greedy heuristic. Additionally, we validate the overall approach in an advanced simulator, modeling reward functions based on realistic physical phenomena and executing the tasks with realistic robot dynamics. Results establish efficacy in modeling complex missions and efficiency in generating high-fidelity task plans while leveraging task relationships. 

**Abstract (ZH)**: 我们提出了一种新的多机器人任务分配问题的表示方法，该方法包含（a）任务间的复杂优先关系，（b）高效的任务内协调，以及（c）通过机器人联盟进行合作。任务图规定了任务及其关系，一组奖励函数模型了联盟规模和前置任务性能的影响。最大化任务奖励是NP难问题；因此，我们提出基于网络流的算法来高效地近似求解。一种新颖的在线算法执行迭代重新分配，以实现对任务失败和模型不准确的鲁棒性，从而优于离线方法。我们全面评估了算法在具有随机任务和奖励函数的测试平台上的表现，并将其与混合整数求解器和贪婪启发式方法进行了比较。此外，我们在一个先进的模拟器中验证了整体方法，该模拟器基于现实物理现象建模奖励函数，并使用真实机器人动力学执行任务。结果证实了该方法在建模复杂任务和生成高保真的任务规划方面具有有效性与效率，同时利用了任务之间的关系。 

---
# Semantic-LiDAR-Inertial-Wheel Odometry Fusion for Robust Localization in Large-Scale Dynamic Environments 

**Title (ZH)**: 面向大规模动态环境的语义LiDAR-惯性-陀螺仪里程计融合鲁棒定位方法 

**Authors**: Haoxuan Jiang, Peicong Qian, Yusen Xie, Linwei Zheng, Xiaocong Li, Ming Liu, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.14999)  

**Abstract**: Reliable, drift-free global localization presents significant challenges yet remains crucial for autonomous navigation in large-scale dynamic environments. In this paper, we introduce a tightly-coupled Semantic-LiDAR-Inertial-Wheel Odometry fusion framework, which is specifically designed to provide high-precision state estimation and robust localization in large-scale dynamic environments. Our framework leverages an efficient semantic-voxel map representation and employs an improved scan matching algorithm, which utilizes global semantic information to significantly reduce long-term trajectory drift. Furthermore, it seamlessly fuses data from LiDAR, IMU, and wheel odometry using a tightly-coupled multi-sensor fusion Iterative Error-State Kalman Filter (iESKF). This ensures reliable localization without experiencing abnormal drift. Moreover, to tackle the challenges posed by terrain variations and dynamic movements, we introduce a 3D adaptive scaling strategy that allows for flexible adjustments to wheel odometry measurement weights, thereby enhancing localization precision. This study presents extensive real-world experiments conducted in a one-million-square-meter automated port, encompassing 3,575 hours of operational data from 35 Intelligent Guided Vehicles (IGVs). The results consistently demonstrate that our system outperforms state-of-the-art LiDAR-based localization methods in large-scale dynamic environments, highlighting the framework's reliability and practical value. 

**Abstract (ZH)**: 可靠的、无漂移的全局定位在大规模动态环境中的自主导航中面临重大挑战，但依然至关重要。本文介绍了紧耦合的语义-LiDAR-惯性-陀螺仪 odometry 融合框架，该框架专门设计用于提供在大规模动态环境中的高精度状态估计和稳健定位。该框架利用高效的语义体素地图表示，并采用改进的扫描配准算法，该算法利用全局语义信息显著减少长期轨迹漂移。此外，它通过使用紧耦合多传感器融合迭代错误状态卡尔曼滤波器（iESKF）无缝融合来自 LiDAR、IMU 和轮 odometry 的数据，确保在不发生异常漂移的情况下实现可靠定位。此外，为应对地形变化和动态移动的挑战，我们引入了一种三维自适应缩放策略，允许灵活调整轮 odometry 测量权重，从而提高定位精度。本研究在占地一平方公里的自动化港口进行了广泛的实地实验，涵盖来自35辆智能引导车（IGVs）的3,575小时操作数据。结果一致表明，我们的系统在大规模动态环境中优于最先进的基于 LiDAR 的定位方法，突显了该框架的可靠性和实用性。 

---
# The Role of Touch: Towards Optimal Tactile Sensing Distribution in Anthropomorphic Hands for Dexterous In-Hand Manipulation 

**Title (ZH)**: 触觉的作用：Towards Anthropomorphic Hands中最佳触觉传感分布以实现灵巧的手内操作 

**Authors**: João Damião Almeida, Egidio Falotico, Cecilia Laschi, José Santos-Victor  

**Link**: [PDF](https://arxiv.org/pdf/2509.14984)  

**Abstract**: In-hand manipulation tasks, particularly in human-inspired robotic systems, must rely on distributed tactile sensing to achieve precise control across a wide variety of tasks. However, the optimal configuration of this network of sensors is a complex problem, and while the fingertips are a common choice for placing sensors, the contribution of tactile information from other regions of the hand is often overlooked. This work investigates the impact of tactile feedback from various regions of the fingers and palm in performing in-hand object reorientation tasks. We analyze how sensory feedback from different parts of the hand influences the robustness of deep reinforcement learning control policies and investigate the relationship between object characteristics and optimal sensor placement. We identify which tactile sensing configurations contribute to improving the efficiency and accuracy of manipulation. Our results provide valuable insights for the design and use of anthropomorphic end-effectors with enhanced manipulation capabilities. 

**Abstract (ZH)**: 基于手部的触觉反馈在执行手内物体重新定向任务中的影响：触觉传感器配置优化研究 

---
# M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation 

**Title (ZH)**: M4Diffuser: 多视图扩散策略与可控性感知控制的鲁棒移动 manipulability 控制 

**Authors**: Ju Dong, Lei Zhang, Liding Zhang, Yao Ling, Yu Fu, Kaixin Bai, Zoltán-Csaba Márton, Zhenshan Bing, Zhaopeng Chen, Alois Christian Knoll, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14980)  

**Abstract**: Mobile manipulation requires the coordinated control of a mobile base and a robotic arm while simultaneously perceiving both global scene context and fine-grained object details. Existing single-view approaches often fail in unstructured environments due to limited fields of view, exploration, and generalization abilities. Moreover, classical controllers, although stable, struggle with efficiency and manipulability near singularities. To address these challenges, we propose M4Diffuser, a hybrid framework that integrates a Multi-View Diffusion Policy with a novel Reduced and Manipulability-aware QP (ReM-QP) controller for mobile manipulation. The diffusion policy leverages proprioceptive states and complementary camera perspectives with both close-range object details and global scene context to generate task-relevant end-effector goals in the world frame. These high-level goals are then executed by the ReM-QP controller, which eliminates slack variables for computational efficiency and incorporates manipulability-aware preferences for robustness near singularities. Comprehensive experiments in simulation and real-world environments show that M4Diffuser achieves 7 to 56 percent higher success rates and reduces collisions by 3 to 31 percent over baselines. Our approach demonstrates robust performance for smooth whole-body coordination, and strong generalization to unseen tasks, paving the way for reliable mobile manipulation in unstructured environments. Details of the demo and supplemental material are available on our project website this https URL. 

**Abstract (ZH)**: 移动操作需要协调控制移动基座和机器人臂，并同时感知全局场景上下文和细微的物体细节。现有的单视角方法往往在无结构环境中失败，因为其视场有限、探索能力和泛化能力不足。此外，虽然经典控制器稳定，但在接近奇点时效率和操作性较低。为应对这些挑战，我们提出了一种名为M4Diffuser的混合框架，该框架结合了多视角扩散策略和一种新颖的减维和操作性感知QP（ReM-QP）控制器，用于移动操作。扩散策略利用自身的状态和互补的摄像机视角，结合近距离物体细节和全局场景上下文，生成世界坐标系中的任务相关末端执行器目标。随后，这些高层目标由ReM-QP控制器执行，该控制器通过消除松弛变量提高计算效率，并通过操作性感知的偏好提高在接近奇点时的鲁棒性。在仿真和真实环境中的全面实验表明，M4Diffuser相比基准方法的成功率提高了7%到56%，碰撞减少了3%到31%。我们的方法展示了平滑全身协调的强大性能，并且具有良好的一般性，能够处理未见过的任务，为无结构环境中可靠的移动操作铺平了道路。更多演示细节和补充材料可在我们的项目网站获取：this https URL。 

---
# PA-MPPI: Perception-Aware Model Predictive Path Integral Control for Quadrotor Navigation in Unknown Environments 

**Title (ZH)**: PA-MPPI: awareness of Perception in Model Predictive Path Integral Control for Quadrotor Navigation in Unknown Environments 

**Authors**: Yifan Zhai, Rudolf Reiter, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2509.14978)  

**Abstract**: Quadrotor navigation in unknown environments is critical for practical missions such as search-and-rescue. Solving it requires addressing three key challenges: the non-convexity of free space due to obstacles, quadrotor-specific dynamics and objectives, and the need for exploration of unknown regions to find a path to the goal. Recently, the Model Predictive Path Integral (MPPI) method has emerged as a promising solution that solves the first two challenges. By leveraging sampling-based optimization, it can effectively handle non-convex free space while directly optimizing over the full quadrotor dynamics, enabling the inclusion of quadrotor-specific costs such as energy consumption. However, its performance in unknown environments is limited, as it lacks the ability to explore unknown regions when blocked by large obstacles. To solve this issue, we introduce Perception-Aware MPPI (PA-MPPI). Here, perception-awareness is defined as adapting the trajectory online based on perception objectives. Specifically, when the goal is occluded, PA-MPPI's perception cost biases trajectories that can perceive unknown regions. This expands the mapped traversable space and increases the likelihood of finding alternative paths to the goal. Through hardware experiments, we demonstrate that PA-MPPI, running at 50 Hz with our efficient perception and mapping module, performs up to 100% better than the baseline in our challenging settings where the state-of-the-art MPPI fails. In addition, we demonstrate that PA-MPPI can be used as a safe and robust action policy for navigation foundation models, which often provide goal poses that are not directly reachable. 

**Abstract (ZH)**: 无感知环境下四旋翼导航对于搜索救援等实际任务至关重要。为此，需要解决三个关键挑战：由障碍物引起的非凸自由空间、四旋翼特有的动态特性和目标以及探索未知区域以找到通往目标的路径的需要。最近，模型预测路径积分（MPPI）方法被认为是对前两个挑战的有效解决方案。通过利用基于采样的优化，它可以有效地处理非凸自由空间，并直接优化四旋翼的动力学，从而能够包含四旋翼特定的成本，如能耗。然而，其在未知环境中的表现有限，因为它在被大障碍物阻挡时缺乏探索未知区域的能力。为了解决这个问题，我们引入了感知导向的MPPI（PA-MPPI）。在这里，感知导向指的是根据感知目标在线调整轨迹。具体而言，当目标被遮挡时，PA-MPPI 的感知成本会偏向能够感知未知区域的轨迹，从而扩展可通行空间，并增加找到通往目标的替代路径的概率。通过硬件实验，我们证明，在我们的挑战性设置中，与最先进的MPPI相比，使用我们高效的感知和映射模块在50 Hz运行的PA-MPPI表现提高了100%。此外，我们证明了PA-MPPI可以作为导航基础模型的安全且鲁棒的动作策略，这些模型通常提供了直接不可达的目标姿态。 

---
# Exploratory Movement Strategies for Texture Discrimination with a Neuromorphic Tactile Sensor 

**Title (ZH)**: 触觉神经形态传感器中用于纹理鉴别的一般探索运动策略 

**Authors**: Xingchen Xu, Ao Li, Benjamin Ward-Cherrier  

**Link**: [PDF](https://arxiv.org/pdf/2509.14954)  

**Abstract**: We propose a neuromorphic tactile sensing framework for robotic texture classification that is inspired by human exploratory strategies. Our system utilizes the NeuroTac sensor to capture neuromorphic tactile data during a series of exploratory motions. We first tested six distinct motions for texture classification under fixed environment: sliding, rotating, tapping, as well as the combined motions: sliding+rotating, tapping+rotating, and tapping+sliding. We chose sliding and sliding+rotating as the best motions based on final accuracy and the sample timing length needed to reach converged accuracy. In the second experiment designed to simulate complex real-world conditions, these two motions were further evaluated under varying contact depth and speeds. Under these conditions, our framework attained the highest accuracy of 87.33\% with sliding+rotating while maintaining an extremely low power consumption of only 8.04 mW. These results suggest that the sliding+rotating motion is the optimal exploratory strategy for neuromorphic tactile sensing deployment in texture classification tasks and holds significant promise for enhancing robotic environmental interaction. 

**Abstract (ZH)**: 一种受人类探索策略启发的神经形态触觉传感框架及其在纹理分类中的应用 

---
# PERAL: Perception-Aware Motion Control for Passive LiDAR Excitation in Spherical Robots 

**Title (ZH)**: 感知导向的运动控制以实现球形机器人中被动LiDAR的激发 

**Authors**: Shenghai Yuan, Jason Wai Hao Yee, Weixiang Guo, Zhongyuan Liu, Thien-Minh Nguyen, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.14915)  

**Abstract**: Autonomous mobile robots increasingly rely on LiDAR-IMU odometry for navigation and mapping, yet horizontally mounted LiDARs such as the MID360 capture few near-ground returns, limiting terrain awareness and degrading performance in feature-scarce environments. Prior solutions - static tilt, active rotation, or high-density sensors - either sacrifice horizontal perception or incur added actuators, cost, and power. We introduce PERAL, a perception-aware motion control framework for spherical robots that achieves passive LiDAR excitation without dedicated hardware. By modeling the coupling between internal differential-drive actuation and sensor attitude, PERAL superimposes bounded, non-periodic oscillations onto nominal goal- or trajectory-tracking commands, enriching vertical scan diversity while preserving navigation accuracy. Implemented on a compact spherical robot, PERAL is validated across laboratory, corridor, and tactical environments. Experiments demonstrate up to 96 percent map completeness, a 27 percent reduction in trajectory tracking error, and robust near-ground human detection, all at lower weight, power, and cost compared with static tilt, active rotation, and fixed horizontal baselines. The design and code will be open-sourced upon acceptance. 

**Abstract (ZH)**: 基于感知的运动控制框架：无专用地面机器人LiDAR激发方法 

---
# COMPASS: Confined-space Manipulation Planning with Active Sensing Strategy 

**Title (ZH)**: COMPASS: 有限空间操作规划与主动传感策略 

**Authors**: Qixuan Li, Chen Le, Dongyue Huang, Jincheng Yu, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.14787)  

**Abstract**: Manipulation in confined and cluttered environments remains a significant challenge due to partial observability and complex configuration spaces. Effective manipulation in such environments requires an intelligent exploration strategy to safely understand the scene and search the target. In this paper, we propose COMPASS, a multi-stage exploration and manipulation framework featuring a manipulation-aware sampling-based planner. First, we reduce collision risks with a near-field awareness scan to build a local collision map. Additionally, we employ a multi-objective utility function to find viewpoints that are both informative and conducive to subsequent manipulation. Moreover, we perform a constrained manipulation optimization strategy to generate manipulation poses that respect obstacle constraints. To systematically evaluate method's performance under these difficulties, we propose a benchmark of confined-space exploration and manipulation containing four level challenging scenarios. Compared to exploration methods designed for other robots and only considering information gain, our framework increases manipulation success rate by 24.25% in simulations. Real-world experiments demonstrate our method's capability for active sensing and manipulation in confined environments. 

**Abstract (ZH)**: 在受限和杂乱环境中的操作仍因部分可观测性和复杂的空间配置而构成重大挑战。有效的操作需要智能的探索策略以安全地理解场景并搜索目标。本文提出了一种多阶段操作探索框架COMPASS，该框架具有感知操作的采样计划器。首先，我们通过近场感知扫描来减少碰撞风险并构建局部碰撞图。此外，我们采用多目标效用函数来寻找既具有信息价值又有利于随后操作的视点。进一步地，我们执行受约束的操作优化策略以生成遵守障碍物约束的操作姿态。为了系统地评估该方法在这些困难环境下的性能，我们提出了一种受限空间探索与操作的基准，其中包括四个具有挑战性的场景。与仅考虑信息增益的为其他机器人设计的探索方法相比，在仿真实验中，我们的框架使操作成功率提高了24.25%。实际实验展示了该方法在受限环境中的主动感知和操作能力。 

---
# Investigating the Effect of LED Signals and Emotional Displays in Human-Robot Shared Workspaces 

**Title (ZH)**: 探究LED信号和情感显示在人机共融工作空间中的效果 

**Authors**: Maria Ibrahim, Alap Kshirsagar, Dorothea Koert, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2509.14748)  

**Abstract**: Effective communication is essential for safety and efficiency in human-robot collaboration, particularly in shared workspaces. This paper investigates the impact of nonverbal communication on human-robot interaction (HRI) by integrating reactive light signals and emotional displays into a robotic system. We equipped a Franka Emika Panda robot with an LED strip on its end effector and an animated facial display on a tablet to convey movement intent through colour-coded signals and facial expressions. We conducted a human-robot collaboration experiment with 18 participants, evaluating three conditions: LED signals alone, LED signals with reactive emotional displays, and LED signals with pre-emptive emotional displays. We collected data through questionnaires and position tracking to assess anticipation of potential collisions, perceived clarity of communication, and task performance. The results indicate that while emotional displays increased the perceived interactivity of the robot, they did not significantly improve collision anticipation, communication clarity, or task efficiency compared to LED signals alone. These findings suggest that while emotional cues can enhance user engagement, their impact on task performance in shared workspaces is limited. 

**Abstract (ZH)**: 非言语交流对人类与机器人合作中安全与效率的影响研究：通过集成反应性光信号和情绪展示改善人机交互 

---
# Rethinking Reference Trajectories in Agile Drone Racing: A Unified Reference-Free Model-Based Controller via MPPI 

**Title (ZH)**: 重新思考敏捷无人机竞速中的参考轨迹：基于MPPI的统一参考自由模型控制器 

**Authors**: Fangguo Zhao, Xin Guan, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.14726)  

**Abstract**: While model-based controllers have demonstrated remarkable performance in autonomous drone racing, their performance is often constrained by the reliance on pre-computed reference trajectories. Conventional approaches, such as trajectory tracking, demand a dynamically feasible, full-state reference, whereas contouring control relaxes this requirement to a geometric path but still necessitates a reference. Recent advancements in reinforcement learning (RL) have revealed that many model-based controllers optimize surrogate objectives, such as trajectory tracking, rather than the primary racing goal of directly maximizing progress through gates. Inspired by these findings, this work introduces a reference-free method for time-optimal racing by incorporating this gate progress objective, derived from RL reward shaping, directly into the Model Predictive Path Integral (MPPI) formulation. The sampling-based nature of MPPI makes it uniquely capable of optimizing the discontinuous and non-differentiable objective in real-time. We also establish a unified framework that leverages MPPI to systematically and fairly compare three distinct objective functions with a consistent dynamics model and parameter set: classical trajectory tracking, contouring control, and the proposed gate progress objective. We compare the performance of these three objectives when solved via both MPPI and a traditional gradient-based solver. Our results demonstrate that the proposed reference-free approach achieves competitive racing performance, rivaling or exceeding reference-based methods. Videos are available at this https URL 

**Abstract (ZH)**: 基于模型的控制器在自主无人机竞速中展现出卓越的性能，但其性能往往受限于依赖预计算的参考轨迹。传统的轨迹跟踪方法需要动态可行的全状态参考轨迹，而轮廓控制放宽了这一要求，只需几何路径参考，但仍需参考轨迹。最近在强化学习领域的进展表明，许多基于模型的控制器优化的是轨迹跟踪等替代目标，而不是直接最大化通过障碍门的主要竞速目标。受这一发现的启发，本工作提出了一种无需参考轨迹的方法，通过将从强化学习奖励塑造中得出的障碍门进度目标直接纳入模型预测路径积分（MPPI）公式中来实现时间最优竞速。基于采样的性质使MPPI能够实时优化非连续性和非光滑目标。同时，我们建立了一个统一框架，利用MPPI系统地、公平地比较三种不同的目标函数：经典轨迹跟踪、轮廓控制和所提出的障碍门进度目标，这些方法采用相同的动态模型和参数集。我们比较了这些目标函数通过MPPI和传统梯度求解器求解时的性能。结果表明，提出的无需参考轨迹的方法在竞速性能上具有竞争力，能够匹敌或超越基于参考的方法。相关视频可在以下链接获取。 

---
# Wohlhart's Three-Loop Mechanism: An Overconstrained and Shaky Linkage 

**Title (ZH)**: Wohlhart的三环机制：一个过约束且不稳定的连杆机构 

**Authors**: Andreas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2509.14698)  

**Abstract**: This paper revisits a three-loop spatial linkage that was proposed in an ARK 2004 paper by Karl Wohlhart (as extension of a two-loop linkage proposed by Eddie Baker in 1980) and later analyzed in an ARK 2006 paper by Diez-Martinez et. al. A local analysis shows that this linkage has a finite degree of freedom (DOF) 3 (and is thus overconstrained) while in its reference configuration the differential DOF is 5. It is shown that its configuration space is locally a smooth manifold so that the reference configuration is not a c-space singularity. It is shown that the differential DOF is locally constant, which makes this linkage shaky (so that the reference configuration is not a singularity). The higher-order local analysis is facilitated by the computation of the kinematic tangent cone as well as a local approximation of the c-space. 

**Abstract (ZH)**: 本文重新审视了Karl Wohlhart在2004年ARK论文中提出的一个三环空间连杆机构（该机构是Eddie Baker于1980年提出的一个双环连杆机构的扩展），并随后在Diez-Martinez等人2006年ARK论文中进行了分析。局部分析表明，该连杆机构具有有限的自由度3（因此是过约束的），而在参考配置下，其微分自由度为5。证明其配置空间在局部是光滑流形，从而使参考配置不是c空间奇异点。证明其微分自由度在局部是恒定的，从而使该连杆机构具有脆弱性（因此参考配置不是奇异点）。更高阶的局部分析通过计算渐近切锥以及局部c空间的逼近得以实现。 

---
# exUMI: Extensible Robot Teaching System with Action-aware Task-agnostic Tactile Representation 

**Title (ZH)**: 扩展性机器人教学系统：具备动作感知的任务无关触觉表示 

**Authors**: Yue Xu, Litao Wei, Pengyu An, Qingyu Zhang, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.14688)  

**Abstract**: Tactile-aware robot learning faces critical challenges in data collection and representation due to data scarcity and sparsity, and the absence of force feedback in existing systems. To address these limitations, we introduce a tactile robot learning system with both hardware and algorithm innovations. We present exUMI, an extensible data collection device that enhances the vanilla UMI with robust proprioception (via AR MoCap and rotary encoder), modular visuo-tactile sensing, and automated calibration, achieving 100% data usability. Building on an efficient collection of over 1 M tactile frames, we propose Tactile Prediction Pretraining (TPP), a representation learning framework through action-aware temporal tactile prediction, capturing contact dynamics and mitigating tactile sparsity. Real-world experiments show that TPP outperforms traditional tactile imitation learning. Our work bridges the gap between human tactile intuition and robot learning through co-designed hardware and algorithms, offering open-source resources to advance contact-rich manipulation research. Project page: this https URL. 

**Abstract (ZH)**: 触觉感知机器人学习因数据稀缺性和稀疏性以及现有系统缺乏力反馈而面临关键挑战。为应对这些局限性，我们介绍了一种结合硬件和算法创新的触觉机器人学习系统。我们提出了exUMI，这是一种可扩展的数据采集设备，通过AR MoCap和旋转编码器增强了原始UMI的鲁棒本体感受能力，集成了模块化视触觉感知和自动标定，实现100%的数据利用率。基于高效采集的超过100万帧触觉图像，我们提出了触觉预测预训练（TPP），这是一种通过动作感知的时间触觉预测进行的表征学习框架，捕捉接触动力学并缓解触觉稀疏性。实验证明，TPP在触觉模仿学习中表现更优。我们的工作通过协同设计的硬件和算法将人类触觉直觉与机器人学习结合起来，提供开源资源以促进接触丰富的操作研究。项目页面：this https URL。 

---
# Hierarchical Planning and Scheduling for Reconfigurable Multi-Robot Disassembly Systems under Structural Constraints 

**Title (ZH)**: 基于结构约束的可重构多机器人拆解系统分级规划与调度 

**Authors**: Takuya Kiyokawa, Tomoki Ishikura, Shingo Hamada, Genichiro Matsuda, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2509.14564)  

**Abstract**: This study presents a system integration approach for planning schedules, sequences, tasks, and motions for reconfigurable robots to automatically disassemble constrained structures in a non-destructive manner. Such systems must adapt their configuration and coordination to the target structure, but the large and complex search space makes them prone to local optima. To address this, we integrate multiple robot arms equipped with different types of tools, together with a rotary stage, into a reconfigurable setup. This flexible system is based on a hierarchical optimization method that generates plans meeting multiple preferred conditions under mandatory requirements within a realistic timeframe. The approach employs two many-objective genetic algorithms for sequence and task planning with motion evaluations, followed by constraint programming for scheduling. Because sequence planning has a much larger search space, we introduce a chromosome initialization method tailored to constrained structures to mitigate the risk of local optima. Simulation results demonstrate that the proposed method effectively solves complex problems in reconfigurable robotic disassembly. 

**Abstract (ZH)**: 本研究提出了一种系统集成方法，用于规划可重构机器人拆解受限结构的调度、序列、任务和动作，以非破坏性方式自动拆解。此类系统必须根据目标结构调整其配置和协调，但由于庞大的复杂搜索空间，它们容易陷入局部最优解。为此，我们将多种不同类型工具的机器人手臂与旋转平台整合到一个可重构设置中。该灵活系统基于一种层次优化方法，在实际的时间框架内生成满足多种优选条件并符合强制要求的计划。该方法使用两种多目标遗传算法进行序列和任务规划，并结合运动评估后的约束编程进行调度。由于序列规划具有更大的搜索空间，我们引入了一种针对受限结构定制的染色体初始化方法，以减轻陷入局部最优解的风险。模拟结果显示，所提出的方法有效地解决了可重构机器人拆解中的复杂问题。 

---
# Dual-Arm Hierarchical Planning for Laboratory Automation: Vibratory Sieve Shaker Operations 

**Title (ZH)**: 双臂层次化规划在实验室自动化中的应用：振动筛汞操作 

**Authors**: Haoran Xiao, Xue Wang, Huimin Lu, Zhiwen Zeng, Zirui Guo, Ziqi Ni, Yicong Ye, Wei Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.14531)  

**Abstract**: This paper addresses the challenges of automating vibratory sieve shaker operations in a materials laboratory, focusing on three critical tasks: 1) dual-arm lid manipulation in 3 cm clearance spaces, 2) bimanual handover in overlapping workspaces, and 3) obstructed powder sample container delivery with orientation constraints. These tasks present significant challenges, including inefficient sampling in narrow passages, the need for smooth trajectories to prevent spillage, and suboptimal paths generated by conventional methods. To overcome these challenges, we propose a hierarchical planning framework combining Prior-Guided Path Planning and Multi-Step Trajectory Optimization. The former uses a finite Gaussian mixture model to improve sampling efficiency in narrow passages, while the latter refines paths by shortening, simplifying, imposing joint constraints, and B-spline smoothing. Experimental results demonstrate the framework's effectiveness: planning time is reduced by up to 80.4%, and waypoints are decreased by 89.4%. Furthermore, the system completes the full vibratory sieve shaker operation workflow in a physical experiment, validating its practical applicability for complex laboratory automation. 

**Abstract (ZH)**: 本文针对材料实验室中振动筛分器操作的自动化挑战，聚焦于三个关键任务：1) 在3 cm 清晰空间内的双臂盖子操作，2) 重叠工作空间内的双臂交接，3) 受限定向粉末样本容器传输。这些任务带来了显著的挑战，包括在狭窄通道中的低效取样、防止溢出所需的平滑轨迹需求以及由传统方法生成的次优路径。为克服这些挑战，我们提出了一种层次化规划框架，结合了先验引导路径规划和多步轨迹优化。前者利用有限高斯混合模型提高狭窄通道的取样效率，后者则通过缩短路径、简化路径、施加关节约束和B样条平滑来进一步优化路径。实验结果表明，该框架的有效性：规划时间减少了80.4%，路点减少了89.4%。此外，该系统在物理实验中完成了整个振动筛分器操作工作流，验证了其在复杂实验室自动化中的实际适用性。 

---
# Learning to Pick: A Visuomotor Policy for Clustered Strawberry Picking 

**Title (ZH)**: 学习选择：集群草莓采摘的感知运动策略 

**Authors**: Zhenghao Fei, Wenwu Lu, Linsheng Hou, Chen Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.14530)  

**Abstract**: Strawberries naturally grow in clusters, interwoven with leaves, stems, and other fruits, which frequently leads to occlusion. This inherent growth habit presents a significant challenge for robotic picking, as traditional percept-plan-control systems struggle to reach fruits amid the clutter. Effectively picking an occluded strawberry demands dexterous manipulation to carefully bypass or gently move the surrounding soft objects and precisely access the ideal picking point located at the stem just above the calyx. To address this challenge, we introduce a strawberry-picking robotic system that learns from human demonstrations. Our system features a 4-DoF SCARA arm paired with a human teleoperation interface for efficient data collection and leverages an End Pose Assisted Action Chunking Transformer (ACT) to develop a fine-grained visuomotor picking policy. Experiments under various occlusion scenarios demonstrate that our modified approach significantly outperforms the direct implementation of ACT, underscoring its potential for practical application in occluded strawberry picking. 

**Abstract (ZH)**: 草莓自然生长成簇，与叶片、茎和其他果实交织在一起，这常常导致遮挡。这种固有的生长习性为机器人采摘带来了重大挑战，因为传统的感知-计划-控制系统难以在杂乱的环境中达到目标果实。有效地采摘被遮挡的草莓需要灵巧的操作，以细心绕过或轻柔移动周围的软体物体，并准确地定位在花萼上方靠近茎部的理想采摘点。为应对这一挑战，我们提出了一种通过人类示范学习的草莓采摘机器人系统。该系统配备4-DoF SCARA臂，并结合了人类遥控界面以高效收集数据，并利用端位姿辅助动作片段化变换器（ACT）来开发细化的视觉-运动采摘策略。在多种遮挡场景下的实验表明，我们的改进方法显著优于直接实施ACT，突显了其在实际遮挡草莓采摘中的应用潜力。 

---
# Perception-Integrated Safety Critical Control via Analytic Collision Cone Barrier Functions on 3D Gaussian Splatting 

**Title (ZH)**: 基于分析碰撞圆锥障碍函数的三维高斯点云集成感知安全关键控制 

**Authors**: Dario Tscholl, Yashwanth Nakka, Brian Gunter  

**Link**: [PDF](https://arxiv.org/pdf/2509.14421)  

**Abstract**: We present a perception-driven safety filter that converts each 3D Gaussian Splat (3DGS) into a closed-form forward collision cone, which in turn yields a first-order control barrier function (CBF) embedded within a quadratic program (QP). By exploiting the analytic geometry of splats, our formulation provides a continuous, closed-form representation of collision constraints that is both simple and computationally efficient. Unlike distance-based CBFs, which tend to activate reactively only when an obstacle is already close, our collision-cone CBF activates proactively, allowing the robot to adjust earlier and thereby produce smoother and safer avoidance maneuvers at lower computational cost. We validate the method on a large synthetic scene with approximately 170k splats, where our filter reduces planning time by a factor of 3 and significantly decreased trajectory jerk compared to a state-of-the-art 3DGS planner, while maintaining the same level of safety. The approach is entirely analytic, requires no high-order CBF extensions (HOCBFs), and generalizes naturally to robots with physical extent through a principled Minkowski-sum inflation of the splats. These properties make the method broadly applicable to real-time navigation in cluttered, perception-derived extreme environments, including space robotics and satellite systems. 

**Abstract (ZH)**: 基于感知的安全滤波器：从3D高斯点转换到闭式前方碰撞圆锥，并嵌入二次规划的零阶控制屏障函数 

---
