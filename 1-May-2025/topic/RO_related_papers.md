# Self-Supervised Monocular Visual Drone Model Identification through Improved Occlusion Handling 

**Title (ZH)**: 通过改进遮挡处理的自监督单目视觉无人驾驶车辆模型识别 

**Authors**: Stavrow A. Bahnam, Christophe De Wagter, Guido C.H.E. de Croon  

**Link**: [PDF](https://arxiv.org/pdf/2504.21695)  

**Abstract**: Ego-motion estimation is vital for drones when flying in GPS-denied environments. Vision-based methods struggle when flight speed increases and close-by objects lead to difficult visual conditions with considerable motion blur and large occlusions. To tackle this, vision is typically complemented by state estimation filters that combine a drone model with inertial measurements. However, these drone models are currently learned in a supervised manner with ground-truth data from external motion capture systems, limiting scalability to different environments and drones. In this work, we propose a self-supervised learning scheme to train a neural-network-based drone model using only onboard monocular video and flight controller data (IMU and motor feedback). We achieve this by first training a self-supervised relative pose estimation model, which then serves as a teacher for the drone model. To allow this to work at high speed close to obstacles, we propose an improved occlusion handling method for training self-supervised pose estimation models. Due to this method, the root mean squared error of resulting odometry estimates is reduced by an average of 15%. Moreover, the student neural drone model can be successfully obtained from the onboard data. It even becomes more accurate at higher speeds compared to its teacher, the self-supervised vision-based model. We demonstrate the value of the neural drone model by integrating it into a traditional filter-based VIO system (ROVIO), resulting in superior odometry accuracy on aggressive 3D racing trajectories near obstacles. Self-supervised learning of ego-motion estimation represents a significant step toward bridging the gap between flying in controlled, expensive lab environments and real-world drone applications. The fusion of vision and drone models will enable higher-speed flight and improve state estimation, on any drone in any environment. 

**Abstract (ZH)**: 基于自我监督学习的无人机 ego-运动估计 

---
# LRBO2: Improved 3D Vision Based Hand-Eye Calibration for Collaborative Robot Arm 

**Title (ZH)**: LRBO2: 基于改进的3D视觉的手眼标定方法用于协作机器人臂 

**Authors**: Leihui Li, Lixuepiao Wan, Volker Krueger, Xuping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21619)  

**Abstract**: Hand-eye calibration is a common problem in the field of collaborative robotics, involving the determination of the transformation matrix between the visual sensor and the robot flange to enable vision-based robotic tasks. However, this process typically requires multiple movements of the robot arm and an external calibration object, making it both time-consuming and inconvenient, especially in scenarios where frequent recalibration is necessary. In this work, we extend our previous method, Look at Robot Base Once (LRBO), which eliminates the need for external calibration objects such as a chessboard. We propose a generic dataset generation approach for point cloud registration, focusing on aligning the robot base point cloud with the scanned data. Furthermore, a more detailed simulation study is conducted involving several different collaborative robot arms, followed by real-world experiments in an industrial setting. Our improved method is simulated and evaluated using a total of 14 robotic arms from 9 different brands, including KUKA, Universal Robots, UFACTORY, and Franka Emika, all of which are widely used in the field of collaborative robotics. Physical experiments demonstrate that our extended approach achieves performance comparable to existing commercial hand-eye calibration solutions, while completing the entire calibration procedure in just a few seconds. In addition, we provide a user-friendly hand-eye calibration solution, with the code publicly available at this http URL. 

**Abstract (ZH)**: 一次查看机器人基座（LRBO）方法的扩展：用于协作机器人的眼手标定 

---
# One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms 

**Title (ZH)**: 一网统御：四旋翼飞行器在不同平台上跨域随机化训练 

**Authors**: Robin Ferede, Till Blaha, Erin Lucassen, Christophe De Wagter, Guido C.H.E. de Croon  

**Link**: [PDF](https://arxiv.org/pdf/2504.21586)  

**Abstract**: In high-speed quadcopter racing, finding a single controller that works well across different platforms remains challenging. This work presents the first neural network controller for drone racing that generalizes across physically distinct quadcopters. We demonstrate that a single network, trained with domain randomization, can robustly control various types of quadcopters. The network relies solely on the current state to directly compute motor commands. The effectiveness of this generalized controller is validated through real-world tests on two substantially different crafts (3-inch and 5-inch race quadcopters). We further compare the performance of this generalized controller with controllers specifically trained for the 3-inch and 5-inch drone, using their identified model parameters with varying levels of domain randomization (0%, 10%, 20%, 30%). While the generalized controller shows slightly slower speeds compared to the fine-tuned models, it excels in adaptability across different platforms. Our results show that no randomization fails sim-to-real transfer while increasing randomization improves robustness but reduces speed. Despite this trade-off, our findings highlight the potential of domain randomization for generalizing controllers, paving the way for universal AI controllers that can adapt to any platform. 

**Abstract (ZH)**: 在高速四旋翼飞行器竞速中，找到一个适用于不同平台的单一控制器仍旧具有挑战性。本文提出了首个能够跨不同物理特性四旋翼飞行器泛化的神经网络控制器。我们证明了一个经过领域随机化训练的单一网络能够 robust 地控制不同类型的四旋翼飞行器。该网络仅依赖当前状态直接计算电机指令。通过在两种显著不同的飞行器（3英寸和5英寸竞速四旋翼飞行器）上进行实地测试，验证了该泛化控制器的有效性。我们进一步将该泛化控制器的性能与针对3英寸和5英寸无人机分别训练的控制器进行了比较，使用不同水平的领域随机化（0%，10%，20%，30%）确定的模型参数。虽然泛化控制器在速度上略逊于精细调整的模型，但在不同平台上的适应性方面表现出色。我们的结果显示，没有任何随机化失败于仿真到现实的转移，而增加随机化提高了鲁棒性但降低了速度。尽管存在这种权衡，但我们的研究结果强调了领域随机化在控制器泛化中的潜在价值，为能够适应任何平台的通用人工智能控制器奠定了基础。 

---
# UAV Marketplace Simulation Tool for BVLOS Operations 

**Title (ZH)**: UAV市场交易平台模拟工具（适用于视距外操作） 

**Authors**: Kıvanç Şerefoğlu, Önder Gürcan, Reyhan Aydoğan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21428)  

**Abstract**: We present a simulation tool for evaluating team formation in autonomous multi-UAV (Unmanned Aerial Vehicle) missions that operate Beyond Visual Line of Sight (BVLOS). The tool models UAV collaboration and mission execution in dynamic and adversarial conditions, where Byzantine UAVs attempt to disrupt operations. Our tool allows researchers to integrate and compare various team formation strategies in a controlled environment with configurable mission parameters and adversarial behaviors. The log of each simulation run is stored in a structured way along with performance metrics so that statistical analysis could be done straightforwardly. The tool is versatile for testing and improving UAV coordination strategies in real-world applications. 

**Abstract (ZH)**: 一种评估自主多UAV任务中团队形成工具的研究：适用于超视距操作的拜占庭无人机干扰条件下的协同与任务执行模拟 

---
# Task and Joint Space Dual-Arm Compliant Control 

**Title (ZH)**: 双臂顺应控制及任务空间-联合空间协同控制 

**Authors**: Alexander L. Mitchell, Tobit Flatscher, Ingmar Posner  

**Link**: [PDF](https://arxiv.org/pdf/2504.21159)  

**Abstract**: Robots that interact with humans or perform delicate manipulation tasks must exhibit compliance. However, most commercial manipulators are rigid and suffer from significant friction, limiting end-effector tracking accuracy in torque-controlled modes. To address this, we present a real-time, open-source impedance controller that smoothly interpolates between joint-space and task-space compliance. This hybrid approach ensures safe interaction and precise task execution, such as sub-centimetre pin insertions. We deploy our controller on Frank, a dual-arm platform with two Kinova Gen3 arms, and compensate for modelled friction dynamics using a model-free observer. The system is real-time capable and integrates with standard ROS tools like MoveIt!. It also supports high-frequency trajectory streaming, enabling closed-loop execution of trajectories generated by learning-based methods, optimal control, or teleoperation. Our results demonstrate robust tracking and compliant behaviour even under high-friction conditions. The complete system is available open-source at this https URL. 

**Abstract (ZH)**: 与人类交互或执行精细操作任务的机器人必须表现出顺应性。然而，大多数商业 manipulator 是刚性的，且摩擦显著，限制了扭矩控制模式下末端执行器的跟踪精度。为了解决这个问题，我们提出了一种实时开源阻抗控制器，可以在关节空间和任务空间顺应性之间平滑插值。这种混合方法确保了安全的交互和精确的任务执行，例如毫米级针插入。我们在配备两个 Kinova Gen3 手臂的双臂平台 Frank 上部署了该控制器，并使用模型自由观测器补偿了建模的摩擦动态。该系统具有实时能力，并与标准 ROS 工具（如 MoveIt!）集成。它还支持高频率轨迹流式传输，使得基于学习的方法、最优控制或遥操作生成的轨迹能够进行闭环执行。我们的结果表明，即使在高摩擦条件下，该系统也能实现稳健的跟踪和顺应性行为。完整系统已开源，可从以下链接访问：this https URL。 

---
# How to Coordinate UAVs and UGVs for Efficient Mission Planning? Optimizing Energy-Constrained Cooperative Routing with a DRL Framework 

**Title (ZH)**: 如何协调无人机和地面机器人进行高效的任务规划？基于DRL框架的能量约束协同路径优化 

**Authors**: Md Safwan Mondal, Subramanian Ramasamy, Luca Russo, James D. Humann, James M. Dotterweich, Pranav Bhounsule  

**Link**: [PDF](https://arxiv.org/pdf/2504.21111)  

**Abstract**: Efficient mission planning for cooperative systems involving Unmanned Aerial Vehicles (UAVs) and Unmanned Ground Vehicles (UGVs) requires addressing energy constraints, scalability, and coordination challenges between agents. UAVs excel in rapidly covering large areas but are constrained by limited battery life, while UGVs, with their extended operational range and capability to serve as mobile recharging stations, are hindered by slower speeds. This heterogeneity makes coordination between UAVs and UGVs critical for achieving optimal mission outcomes. In this work, we propose a scalable deep reinforcement learning (DRL) framework to address the energy-constrained cooperative routing problem for multi-agent UAV-UGV teams, aiming to visit a set of task points in minimal time with UAVs relying on UGVs for recharging during the mission. The framework incorporates sortie-wise agent switching to efficiently manage multiple agents, by allocating task points and coordinating actions. Using an encoder-decoder transformer architecture, it optimizes routes and recharging rendezvous for the UAV-UGV team in the task scenario. Extensive computational experiments demonstrate the framework's superior performance over heuristic methods and a DRL baseline, delivering significant improvements in solution quality and runtime efficiency across diverse scenarios. Generalization studies validate its robustness, while dynamic scenario highlights its adaptability to real-time changes with a case study. This work advances UAV-UGV cooperative routing by providing a scalable, efficient, and robust solution for multi-agent mission planning. 

**Abstract (ZH)**: 基于无人机（UAV）与地面机器人（UGV）的多智能体协同航路规划：一种考虑能量约束的可扩展深度强化学习框架 

---
# NavEX: A Multi-Agent Coverage in Non-Convex and Uneven Environments via Exemplar-Clustering 

**Title (ZH)**: NavEX: 一种基于范例聚类的非凸不均匀环境多agents覆盖算法 

**Authors**: Donipolo Ghimire, Carlos Nieto-Granda, Solmaz S. Kia  

**Link**: [PDF](https://arxiv.org/pdf/2504.21113)  

**Abstract**: This paper addresses multi-agent deployment in non-convex and uneven environments. To overcome the limitations of traditional approaches, we introduce Navigable Exemplar-Based Dispatch Coverage (NavEX), a novel dispatch coverage framework that combines exemplar-clustering with obstacle-aware and traversability-aware shortest distances, offering a deployment framework based on submodular optimization. NavEX provides a unified approach to solve two critical coverage tasks: (a) fair-access deployment, aiming to provide equitable service by minimizing agent-target distances, and (b) hotspot deployment, prioritizing high-density target regions. A key feature of NavEX is the use of exemplar-clustering for the coverage utility measure, which provides the flexibility to employ non-Euclidean distance metrics that do not necessarily conform to the triangle inequality. This allows NavEX to incorporate visibility graphs for shortest-path computation in environments with planar obstacles, and traversability-aware RRT* for complex, rugged terrains. By leveraging submodular optimization, the NavEX framework enables efficient, near-optimal solutions with provable performance guarantees for multi-agent deployment in realistic and complex settings, as demonstrated by our simulations. 

**Abstract (ZH)**: 基于导航典范的分布式覆盖框架（NavEX）：非凸不规则环境下的多agent部署 

---
# Is Intermediate Fusion All You Need for UAV-based Collaborative Perception? 

**Title (ZH)**: 基于无人机的协作感知中，中间融合是否足矣？ 

**Authors**: Jiuwu Hao, Liguo Sun, Yuting Wan, Yueyang Wu, Ti Xiang, Haolin Song, Pin Lv  

**Link**: [PDF](https://arxiv.org/pdf/2504.21774)  

**Abstract**: Collaborative perception enhances environmental awareness through inter-agent communication and is regarded as a promising solution to intelligent transportation systems. However, existing collaborative methods for Unmanned Aerial Vehicles (UAVs) overlook the unique characteristics of the UAV perspective, resulting in substantial communication overhead. To address this issue, we propose a novel communication-efficient collaborative perception framework based on late-intermediate fusion, dubbed LIF. The core concept is to exchange informative and compact detection results and shift the fusion stage to the feature representation level. In particular, we leverage vision-guided positional embedding (VPE) and box-based virtual augmented feature (BoBEV) to effectively integrate complementary information from various agents. Additionally, we innovatively introduce an uncertainty-driven communication mechanism that uses uncertainty evaluation to select high-quality and reliable shared areas. Experimental results demonstrate that our LIF achieves superior performance with minimal communication bandwidth, proving its effectiveness and practicality. Code and models are available at this https URL. 

**Abstract (ZH)**: 协作感知通过代理间通信增强环境意识，被视为智能交通系统的有前途的解决方案。然而，现有的基于无人机的协作方法忽略了无人机视角的独特特性，导致了显著的通信开销。为解决这一问题，我们提出了一种基于晚期中间融合的新型通信高效协作感知框架，命名为LIF。核心概念是交换信息丰富且紧凑的检测结果，并将融合阶段转移至特征表示级别。特别是在此基础上，我们利用基于视觉的定位嵌入(VPE)和基于框的虚拟 augmented 特征(BoBEV)有效整合了来自各种代理的互补信息。此外，我们创新性地引入了一种基于不确定性驱动的通信机制，使用不确定性评估来选择高质量和可靠的共享区域。实验结果表明，LIF 在最小通信带宽下实现了优越的性能，证明了其有效性和实用性。相关代码和模型可从此链接获取。 

---
# Designing Control Barrier Function via Probabilistic Enumeration for Safe Reinforcement Learning Navigation 

**Title (ZH)**: 基于概率枚举的安全强化学习导航控制屏障函数设计 

**Authors**: Luca Marzari, Francesco Trotti, Enrico Marchesini, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.21643)  

**Abstract**: Achieving safe autonomous navigation systems is critical for deploying robots in dynamic and uncertain real-world environments. In this paper, we propose a hierarchical control framework leveraging neural network verification techniques to design control barrier functions (CBFs) and policy correction mechanisms that ensure safe reinforcement learning navigation policies. Our approach relies on probabilistic enumeration to identify unsafe regions of operation, which are then used to construct a safe CBF-based control layer applicable to arbitrary policies. We validate our framework both in simulation and on a real robot, using a standard mobile robot benchmark and a highly dynamic aquatic environmental monitoring task. These experiments demonstrate the ability of the proposed solution to correct unsafe actions while preserving efficient navigation behavior. Our results show the promise of developing hierarchical verification-based systems to enable safe and robust navigation behaviors in complex scenarios. 

**Abstract (ZH)**: 实现安全自主导航系统对于在动态和不确定性高的实际环境中部署机器人至关重要。本文提出了一种基于神经网络验证技术的层次控制框架，用于设计控制障碍函数（CBFs）和策略纠正机制，以确保安全的强化学习导航策略。我们的方法依赖于概率枚举来识别操作的不安全区域，然后用于构建适用于任意策略的安全CBF控制层。我们在仿真和实际机器人上均对框架进行了验证，使用标准的移动机器人基准测试和高度动态的水下环境监测任务。这些实验展示了所提出解决方案能够纠正不安全行为的同时保持高效的导航行为的能力。我们的结果表明，基于层次验证的系统在复杂场景中实现安全稳健的导航行为具有很大的潜力。 

---
