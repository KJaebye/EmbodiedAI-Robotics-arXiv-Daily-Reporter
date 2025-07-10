# Bounomodes: the grazing ox algorithm for exploration of clustered anomalies 

**Title (ZH)**: Bounomodes：聚类异常探索的 feeding 牛算法 

**Authors**: Samuel Matloob, Ayan Dutta, O. Patrick Kreidl, Swapnonel Roy, Ladislau Bölöni  

**Link**: [PDF](https://arxiv.org/pdf/2507.06960)  

**Abstract**: A common class of algorithms for informative path planning (IPP) follows boustrophedon ("as the ox turns") patterns, which aim to achieve uniform area coverage. However, IPP is often applied in scenarios where anomalies, such as plant diseases, pollution, or hurricane damage, appear in clusters. In such cases, prioritizing the exploration of anomalous regions over uniform coverage is beneficial. This work introduces a class of algorithms referred to as bounomōdes ("as the ox grazes"), which alternates between uniform boustrophedon sampling and targeted exploration of detected anomaly clusters. While uniform sampling can be designed using geometric principles, close exploration of clusters depends on the spatial distribution of anomalies and must be learned. In our implementation, the close exploration behavior is learned using deep reinforcement learning algorithms. Experimental evaluations demonstrate that the proposed approach outperforms several established baselines. 

**Abstract (ZH)**: 一种用于信息路径规划的常见算法类遵循“交替式牛耕”模式，旨在实现均匀区域覆盖。然而，在异常现象，如植物疾病、污染或飓风损害以集群形式出现的场景中，优先探索异常区域而非均匀覆盖是有益的。本工作引入了一类新的算法类，称为“牛觅食”模式，该类算法交替进行均匀牛耕采样和对检测到的异常区域集群的目标性探索。虽然均匀采样的设计可以基于几何原则，但对集群的密切探索依赖于异常现象的空间分布，并需要通过学习获得。在我们的实现中，紧密探索行为是使用深度强化学习算法学习得到的。实验评估表明，所提出的方法优于几个现有的基线方法。 

---
# ULC: A Unified and Fine-Grained Controller for Humanoid Loco-Manipulation 

**Title (ZH)**: ULC：统一细粒度 humanoid 人類抓举控制单元 

**Authors**: Wandong Sun, Luying Feng, Baoshi Cao, Yang Liu, Yaochu Jin, Zongwu Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.06905)  

**Abstract**: Loco-Manipulation for humanoid robots aims to enable robots to integrate mobility with upper-body tracking capabilities. Most existing approaches adopt hierarchical architectures that decompose control into isolated upper-body (manipulation) and lower-body (locomotion) policies. While this decomposition reduces training complexity, it inherently limits coordination between subsystems and contradicts the unified whole-body control exhibited by humans. We demonstrate that a single unified policy can achieve a combination of tracking accuracy, large workspace, and robustness for humanoid loco-manipulation. We propose the Unified Loco-Manipulation Controller (ULC), a single-policy framework that simultaneously tracks root velocity, root height, torso rotation, and dual-arm joint positions in an end-to-end manner, proving the feasibility of unified control without sacrificing performance. We achieve this unified control through key technologies: sequence skill acquisition for progressive learning complexity, residual action modeling for fine-grained control adjustments, command polynomial interpolation for smooth motion transitions, random delay release for robustness to deploy variations, load randomization for generalization to external disturbances, and center-of-gravity tracking for providing explicit policy gradients to maintain stability. We validate our method on the Unitree G1 humanoid robot with 3-DOF (degrees-of-freedom) waist. Compared with strong baselines, ULC shows better tracking performance to disentangled methods and demonstrating larger workspace coverage. The unified dual-arm tracking enables precise manipulation under external loads while maintaining coordinated whole-body control for complex loco-manipulation tasks. 

**Abstract (ZH)**: 人形机器人的结合操作与移动控制研究：统一策略实现精确操作与广泛工作空间 

---
# Toward a Full-Stack Co-Simulation Platform for Testing of Automated Driving Systems 

**Title (ZH)**: 面向自动驾驶系统测试的全栈协同仿真平台研究 

**Authors**: Dong Bi, Yongqi Zhao, Zhengguo Gu, Tomislav Mihalj, Jia Hu, Arno Eichberger  

**Link**: [PDF](https://arxiv.org/pdf/2507.06884)  

**Abstract**: Virtual testing has emerged as an effective approach to accelerate the deployment of automated driving systems. Nevertheless, existing simulation toolchains encounter difficulties in integrating rapid, automated scenario generation with simulation environments supporting advanced automated driving capabilities. To address this limitation, a full-stack toolchain is presented, enabling automatic scenario generation from real-world datasets and efficient validation through a co-simulation platform based on CarMaker, ROS, and Apollo. The simulation results demonstrate the effectiveness of the proposed toolchain. A demonstration video showcasing the toolchain is available at the provided link: this https URL. 

**Abstract (ZH)**: 虚拟测试已成为加速自动驾驶系统部署的有效方法。然而，现有仿真工具链在将快速、自动化场景生成与支持高级自动驾驶能力的仿真环境集成方面遇到困难。为解决这一限制，提出了一套全栈工具链，能够从真实世界数据集自动生成场景，并通过基于CarMaker、ROS和Apollo的协同仿真平台进行高效验证。仿真结果表明所提出工具链的有效性。演示视频可在提供的链接处查看：this https URL。 

---
# Friction Estimation for In-Hand Planar Motion 

**Title (ZH)**: 手持平面运动的摩擦估计 

**Authors**: Gabriel Arslan Waltersson, Yiannis Karayiannidis  

**Link**: [PDF](https://arxiv.org/pdf/2507.06824)  

**Abstract**: This paper presents a method for online estimation of contact properties during in-hand sliding manipulation with a parallel gripper. We estimate the static and Coulomb friction as well as the contact radius from tactile measurements of contact forces and sliding velocities. The method is validated in both simulation and real-world experiments. Furthermore, we propose a heuristic to deal with fast slip-stick dynamics which can adversely affect the estimation. 

**Abstract (ZH)**: 该文提出了一种基于并联夹爪的手内滑动操作过程中接触属性在线估计的方法。我们利用接触力和滑动速度的触觉测量数据估计静摩擦和库仑摩擦以及接触半径。该方法在仿真和实际实验中得到了验证。此外，我们提出了一种启发式方法来处理会对估计造成负面影响的快速滑移-粘连动态。 

---
# Hierarchical Reinforcement Learning for Articulated Tool Manipulation with Multifingered Hand 

**Title (ZH)**: 多指手驱动 articulated 工具 manipulation 的分层强化学习 

**Authors**: Wei Xu, Yanchao Zhao, Weichao Guo, Xinjun Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06822)  

**Abstract**: Manipulating articulated tools, such as tweezers or scissors, has rarely been explored in previous research. Unlike rigid tools, articulated tools change their shape dynamically, creating unique challenges for dexterous robotic hands. In this work, we present a hierarchical, goal-conditioned reinforcement learning (GCRL) framework to improve the manipulation capabilities of anthropomorphic robotic hands using articulated tools. Our framework comprises two policy layers: (1) a low-level policy that enables the dexterous hand to manipulate the tool into various configurations for objects of different sizes, and (2) a high-level policy that defines the tool's goal state and controls the robotic arm for object-picking tasks. We employ an encoder, trained on synthetic pointclouds, to estimate the tool's affordance states--specifically, how different tool configurations (e.g., tweezer opening angles) enable grasping of objects of varying sizes--from input point clouds, thereby enabling precise tool manipulation. We also utilize a privilege-informed heuristic policy to generate replay buffer, improving the training efficiency of the high-level policy. We validate our approach through real-world experiments, showing that the robot can effectively manipulate a tweezer-like tool to grasp objects of diverse shapes and sizes with a 70.8 % success rate. This study highlights the potential of RL to advance dexterous robotic manipulation of articulated tools. 

**Abstract (ZH)**: 操纵articulated工具（如镊子或剪刀）的研究在以往的工作中较少探索。与刚性工具不同，articulated工具会动态改变形状，为灵巧的手部机器人带来了独特的挑战。在这项工作中，我们提出了一种分层的目标条件强化学习（GCRL）框架，以提高类人手部机器人使用articulated工具的操纵能力。该框架包含两个策略层：（1）一个低层策略，使灵巧的手部能够根据不同大小的对象调整工具的配置；（2）一个高层策略，定义工具的目标状态并控制机器臂进行物体拾取任务。我们利用一个在合成点云上训练的编码器，从输入点云中估计工具的功能状态，特别是不同工具配置（例如镊子张开的角度）如何使不同大小的对象抓取成为可能，从而实现精确的工具操纵。我们还利用基于特权的启发式策略生成回放缓冲区，提高了高层策略的训练效率。通过现实世界的实验验证了我们的方法，结果显示机器人能够以70.8%的成功率有效地操纵类似镊子的工具来抓取各种形状和大小的对象。本研究突显了RL在推进articulated工具的灵巧机器人操控方面的潜力。 

---
# Stream Function-Based Navigation for Complex Quadcopter Obstacle Avoidance 

**Title (ZH)**: 基于流函数的复杂环境中四旋翼避障导航 

**Authors**: Sean Smith, Emmanuel Witrant, Ya-Jun Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06787)  

**Abstract**: This article presents a novel stream function-based navigational control system for obstacle avoidance, where obstacles are represented as two-dimensional (2D) rigid surfaces in inviscid, incompressible flows. The approach leverages the vortex panel method (VPM) and incorporates safety margins to control the stream function and flow properties around virtual surfaces, enabling navigation in complex, partially observed environments using real-time sensing. To address the limitations of the VPM in managing relative distance and avoiding rapidly accelerating obstacles at close proximity, the system integrates a model predictive controller (MPC) based on higher-order control barrier functions (HOCBF). This integration incorporates VPM trajectory generation, state estimation, and constraint handling into a receding-horizon optimization problem. The 2D rigid surfaces are enclosed using minimum bounding ellipses (MBEs), while an adaptive Kalman filter (AKF) captures and predicts obstacle dynamics, propagating these estimates into the MPC-HOCBF for rapid avoidance maneuvers. Evaluation is conducted using a PX4-powered Clover drone Gazebo simulator and real-time experiments involving a COEX Clover quadcopter equipped with a 360 degree LiDAR sensor. 

**Abstract (ZH)**: 基于涡旋面板方法的新型流函数导向导航控制系统：用于障碍物 avoidance 的实时 sensing 和复杂环境下的鲁棒避障 

---
# Distributed Fault-Tolerant Multi-Robot Cooperative Localization in Adversarial Environments 

**Title (ZH)**: 分布式鲁棒多机器人协同定位技术在对抗环境中 

**Authors**: Tohid Kargar Tasooji, Ramviyas Parasuraman  

**Link**: [PDF](https://arxiv.org/pdf/2507.06750)  

**Abstract**: In multi-robot systems (MRS), cooperative localization is a crucial task for enhancing system robustness and scalability, especially in GPS-denied or communication-limited environments. However, adversarial attacks, such as sensor manipulation, and communication jamming, pose significant challenges to the performance of traditional localization methods. In this paper, we propose a novel distributed fault-tolerant cooperative localization framework to enhance resilience against sensor and communication disruptions in adversarial environments. We introduce an adaptive event-triggered communication strategy that dynamically adjusts communication thresholds based on real-time sensing and communication quality. This strategy ensures optimal performance even in the presence of sensor degradation or communication failure. Furthermore, we conduct a rigorous analysis of the convergence and stability properties of the proposed algorithm, demonstrating its resilience against bounded adversarial zones and maintaining accurate state estimation. Robotarium-based experiment results show that our proposed algorithm significantly outperforms traditional methods in terms of localization accuracy and communication efficiency, particularly in adversarial settings. Our approach offers improved scalability, reliability, and fault tolerance for MRS, making it suitable for large-scale deployments in real-world, challenging environments. 

**Abstract (ZH)**: 多机器人系统中基于对抗环境的分布式容错协同定位框架 

---
# LOVON: Legged Open-Vocabulary Object Navigator 

**Title (ZH)**: LOVON: 腿足开放式词汇对象导航器 

**Authors**: Daojie Peng, Jiahang Cao, Qiang Zhang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.06747)  

**Abstract**: Object navigation in open-world environments remains a formidable and pervasive challenge for robotic systems, particularly when it comes to executing long-horizon tasks that require both open-world object detection and high-level task planning. Traditional methods often struggle to integrate these components effectively, and this limits their capability to deal with complex, long-range navigation missions. In this paper, we propose LOVON, a novel framework that integrates large language models (LLMs) for hierarchical task planning with open-vocabulary visual detection models, tailored for effective long-range object navigation in dynamic, unstructured environments. To tackle real-world challenges including visual jittering, blind zones, and temporary target loss, we design dedicated solutions such as Laplacian Variance Filtering for visual stabilization. We also develop a functional execution logic for the robot that guarantees LOVON's capabilities in autonomous navigation, task adaptation, and robust task completion. Extensive evaluations demonstrate the successful completion of long-sequence tasks involving real-time detection, search, and navigation toward open-vocabulary dynamic targets. Furthermore, real-world experiments across different legged robots (Unitree Go2, B2, and H1-2) showcase the compatibility and appealing plug-and-play feature of LOVON. 

**Abstract (ZH)**: 开放世界环境中的物体导航仍然是机器人系统面临的一项艰巨且普遍的挑战，尤其是在执行长时任务时，这些任务需要开放世界物体检测和高级任务规划的结合。传统方法往往难以有效集成这些组件，从而限制了其处理复杂、长距离导航任务的能力。本文提出了一种新的框架LOVON，该框架结合了层次任务规划的大语言模型（LLMs）与面向开放词汇视觉检测的模型，专门针对动态、非结构化环境中的有效长距离物体导航。为了应对包括视觉抖动、盲区和目标暂时丢失在内的现实世界挑战，我们设计了专门的解决方案，如拉普拉斯方差滤波用于视觉稳定。我们还为机器人开发了功能执行逻辑，以确保LOVON在自主导航、任务适应和稳健任务完成方面的能力。广泛的评估表明，LOVON能够成功完成涉及实时检测、搜索和导航至开放词汇动态目标的长时间序列任务。此外，跨不同腿足机器人（Unitree Go2、B2和H1-2）的实际实验展示了LOVON的兼容性和方便的即插即用特性。 

---
# Spatial-Temporal Aware Visuomotor Diffusion Policy Learning 

**Title (ZH)**: 空间-时间知觉运动扩散策略学习 

**Authors**: Zhenyang Liu, Yikai Wang, Kuanning Wang, Longfei Liang, Xiangyang Xue, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06710)  

**Abstract**: Visual imitation learning is effective for robots to learn versatile tasks. However, many existing methods rely on behavior cloning with supervised historical trajectories, limiting their 3D spatial and 4D spatiotemporal awareness. Consequently, these methods struggle to capture the 3D structures and 4D spatiotemporal relationships necessary for real-world deployment. In this work, we propose 4D Diffusion Policy (DP4), a novel visual imitation learning method that incorporates spatiotemporal awareness into diffusion-based policies. Unlike traditional approaches that rely on trajectory cloning, DP4 leverages a dynamic Gaussian world model to guide the learning of 3D spatial and 4D spatiotemporal perceptions from interactive environments. Our method constructs the current 3D scene from a single-view RGB-D observation and predicts the future 3D scene, optimizing trajectory generation by explicitly modeling both spatial and temporal dependencies. Extensive experiments across 17 simulation tasks with 173 variants and 3 real-world robotic tasks demonstrate that the 4D Diffusion Policy (DP4) outperforms baseline methods, improving the average simulation task success rate by 16.4% (Adroit), 14% (DexArt), and 6.45% (RLBench), and the average real-world robotic task success rate by 8.6%. 

**Abstract (ZH)**: 4D扩散策略(DP4)：一种融入时空意识的视觉模仿学习方法 

---
# Integrating Perceptions: A Human-Centered Physical Safety Model for Human-Robot Interaction 

**Title (ZH)**: 集成感知：以人为本的物理安全模型在人机交互中的应用 

**Authors**: Pranav Pandey, Ramviyas Parasuraman, Prashant Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2507.06700)  

**Abstract**: Ensuring safety in human-robot interaction (HRI) is essential to foster user trust and enable the broader adoption of robotic systems. Traditional safety models primarily rely on sensor-based measures, such as relative distance and velocity, to assess physical safety. However, these models often fail to capture subjective safety perceptions, which are shaped by individual traits and contextual factors. In this paper, we introduce and analyze a parameterized general safety model that bridges the gap between physical and perceived safety by incorporating a personalization parameter, $\rho$, into the safety measurement framework to account for individual differences in safety perception. Through a series of hypothesis-driven human-subject studies in a simulated rescue scenario, we investigate how emotional state, trust, and robot behavior influence perceived safety. Our results show that $\rho$ effectively captures meaningful individual differences, driven by affective responses, trust in task consistency, and clustering into distinct user types. Specifically, our findings confirm that predictable and consistent robot behavior as well as the elicitation of positive emotional states, significantly enhance perceived safety. Moreover, responses cluster into a small number of user types, supporting adaptive personalization based on shared safety models. Notably, participant role significantly shapes safety perception, and repeated exposure reduces perceived safety for participants in the casualty role, emphasizing the impact of physical interaction and experiential change. These findings highlight the importance of adaptive, human-centered safety models that integrate both psychological and behavioral dimensions, offering a pathway toward more trustworthy and effective HRI in safety-critical domains. 

**Abstract (ZH)**: 确保人机交互中的安全性对于培养用户信任并促进机器人系统的广泛应用至关重要。传统的安全性模型主要依赖于基于传感器的措施，如相对距离和速度，来评估物理安全性。然而，这些模型往往未能捕捉到主观的安全感受，后者是由个体特质和情境因素形成的。在本文中，我们引入并分析了一个参数化的通用安全性模型，通过在安全性测量框架中引入个人化参数 $\rho$ 来弥合物理安全与感知安全之间的差距，以考虑安全感知的个体差异。通过一系列基于假设的人类主体研究，在模拟的救援场景中，我们探讨了情绪状态、信任和机器人行为如何影响感知安全。研究结果表明，$\rho$ 有效地捕捉到了由情感反应、任务一致性的信任以及用户类型分群驱动的有意义的个体差异。具体而言，我们的研究结果证实，可预测和一致的机器人行为以及唤起积极情感状态，显著增强了感知安全。此外，响应呈现出少量用户类型的分群，支持基于共享安全性模型的适应性个性化。值得注意的是，参与者角色显著影响了安全性感知，反复接触降低了伤亡角色参与者感知的安全性，突显了物理交互和体验变化的影响。这些发现强调了整合心理和行为维度的适应性和以人为中心的安全模型的重要性，为在关键安全领域实现更值得信赖和有效的HRI提供了途径。 

---
# Multi-Task Multi-Agent Reinforcement Learning via Skill Graphs 

**Title (ZH)**: 基于技能图的多任务多代理强化学习 

**Authors**: Guobin Zhu, Rui Zhou, Wenkang Ji, Hongyin Zhang, Donglin Wang, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.06690)  

**Abstract**: Multi-task multi-agent reinforcement learning (MT-MARL) has recently gained attention for its potential to enhance MARL's adaptability across multiple tasks. However, it is challenging for existing multi-task learning methods to handle complex problems, as they are unable to handle unrelated tasks and possess limited knowledge transfer capabilities. In this paper, we propose a hierarchical approach that efficiently addresses these challenges. The high-level module utilizes a skill graph, while the low-level module employs a standard MARL algorithm. Our approach offers two contributions. First, we consider the MT-MARL problem in the context of unrelated tasks, expanding the scope of MTRL. Second, the skill graph is used as the upper layer of the standard hierarchical approach, with training independent of the lower layer, effectively handling unrelated tasks and enhancing knowledge transfer capabilities. Extensive experiments are conducted to validate these advantages and demonstrate that the proposed method outperforms the latest hierarchical MAPPO algorithms. Videos and code are available at this https URL 

**Abstract (ZH)**: 多任务多智能体强化学习（MT-MARL）近年来因其在多任务环境下的适应性增强而受到关注。然而，现有的多任务学习方法难以处理复杂问题，因为它们无法处理无关任务并且知识迁移能力有限。本文提出了一种分层方法来有效应对这些挑战。高层次模块利用技能图，低层次模块采用标准的多智能体强化学习算法。本文方法做出了两个贡献。首先，我们在无关任务的背景下考虑MT-MARL问题，拓展了多任务强化学习迁移学习（MTRL）的范围。其次，技能图作为标准分层方法的高层，训练与低层独立，有效地处理无关任务并增强知识迁移能力。大量的实验证明了这些优势，并展示了所提出的方法在与最新分层MAPPO算法相比的优越性。有关视频和代码可在以下链接获取。 

---
# Q-STAC: Q-Guided Stein Variational Model Predictive Actor-Critic 

**Title (ZH)**: Q-STAC: Q-引导的Stein变分模型预测行为critic 

**Authors**: Shizhe Cai, Jayadeep Jacob, Zeya Yin, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2507.06625)  

**Abstract**: Deep reinforcement learning has shown remarkable success in continuous control tasks, yet often requires extensive training data, struggles with complex, long-horizon planning, and fails to maintain safety constraints during operation. Meanwhile, Model Predictive Control (MPC) offers explainability and constraint satisfaction, but typically yields only locally optimal solutions and demands careful cost function design. This paper introduces the Q-guided STein variational model predictive Actor-Critic (Q-STAC), a novel framework that bridges these approaches by integrating Bayesian MPC with actor-critic reinforcement learning through constrained Stein Variational Gradient Descent (SVGD). Our method optimizes control sequences directly using learned Q-values as objectives, eliminating the need for explicit cost function design while leveraging known system dynamics to enhance sample efficiency and ensure control signals remain within safe boundaries. Extensive experiments on 2D navigation and robotic manipulation tasks demonstrate that Q-STAC achieves superior sample efficiency, robustness, and optimality compared to state-of-the-art algorithms, while maintaining the high expressiveness of policy distributions. Experiment videos are available on our website: this https URL 

**Abstract (ZH)**: 基于Q引导的Stein变分模型预测actor-critic (Q-STAC)：结合贝叶斯MPC与束缚Stein变分梯度下降的新型框架 

---
# Growing Trees with an Agent: Accelerating RRTs with Learned, Multi-Step Episodic Exploration 

**Title (ZH)**: 使用代理培育树木：基于学习的多步 episodic 探索加速 RRTs 

**Authors**: Xinyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06605)  

**Abstract**: Classical sampling-based motion planners like the RRTs suffer from inefficiencies, particularly in cluttered or high-dimensional spaces, due to their reliance on undirected, random sampling. This paper introduces the Episodic RRT, a novel hybrid planning framework that replaces the primitive of a random point with a learned, multi-step "exploratory episode" generated by a Deep Reinforcement Learning agent. By making the DRL agent the engine of exploration, ERRT transforms the search process from a diffuse, volumetric expansion into a directed, branch-like growth. This paradigm shift yields key advantages: it counters the curse of dimensionality with focused exploration, minimizes expensive collision checks by proactively proposing locally valid paths, and improves connectivity by generating inherently connected path segments. We demonstrate through extensive empirical evaluation across 2D, 3D, and 6D environments that ERRT and its variants consistently and significantly outperform their classical counterparts. In a challenging 6D robotic arm scenario, ERRT achieves a 98% success rate compared to 19% for RRT, is up to 107x faster, reduces collision checks by over 99.6%, and finds initial paths that are nearly 50% shorter. Furthermore, its asymptotically optimal variant, ERRT*, demonstrates vastly superior anytime performance, refining solutions to near-optimality up to 29x faster than standard RRT* in 3D environments. Code: this https URL. 

**Abstract (ZH)**: 基于样本的经典运动规划器如RRT在复杂或高维空间中由于依赖于无向的随机采样而效率低下。本文介绍了一种新颖的混合规划框架，Episodic RRT，它将随机点这一基本概念替换为由深度强化学习代理生成的多步“探索性episode”。通过使DRL代理成为探索引擎，Episodic RRT将搜索过程从弥漫性的体积扩展转变为有向的分支式增长。这种范式转变带来了关键优势：它通过集中探索来对抗维度灾难，通过主动提议局部有效的路径来最大限度减少昂贵的碰撞检测，通过生成固有的连接路径段来提高连通性。通过对2D、3D和6D环境的广泛实证评估显示，Episodic RRT及其变体始终且显著优于其经典 counterpart。在一项具有挑战性的6D机器人臂场景中，Episodic RRT的成功率为98%，而RRT仅为19%，相比RRT快了107倍，减少了超过99.6%的碰撞检测，找到了几乎短50%的初始路径。此外，其渐近最优变体Episodic RRT*展示了优于任何时间的性能，在3D环境中比标准RRT*快29倍地将解决方案精炼至接近最优。代码：this https URL。 

---
# AI Space Cortex: An Experimental System for Future Era Space Exploration 

**Title (ZH)**: AI太空 cortex：未来时代太空探索的实验系统 

**Authors**: Thomas Touma, Ersin Daş, Erica Tevere, Martin Feather, Ksenia Kolcio, Maurice Prather, Alberto Candela, Ashish Goel, Erik Kramer, Hari Nayar, Lorraine Fesq, Joel W. Burdick  

**Link**: [PDF](https://arxiv.org/pdf/2507.06574)  

**Abstract**: Our Robust, Explainable Autonomy for Scientific Icy Moon Operations (REASIMO) effort contributes to NASA's Concepts for Ocean worlds Life Detection Technology (COLDTech) program, which explores science platform technologies for ocean worlds such as Europa and Enceladus. Ocean world missions pose significant operational challenges. These include long communication lags, limited power, and lifetime limitations caused by radiation damage and hostile conditions. Given these operational limitations, onboard autonomy will be vital for future Ocean world missions. Besides the management of nominal lander operations, onboard autonomy must react appropriately in the event of anomalies. Traditional spacecraft rely on a transition into 'safe-mode' in which non-essential components and subsystems are powered off to preserve safety and maintain communication with Earth. For a severely time-limited Ocean world mission, resolutions to these anomalies that can be executed without Earth-in-the-loop communication and associated delays are paramount for completion of the mission objectives and science goals. To address these challenges, the REASIMO effort aims to demonstrate a robust level of AI-assisted autonomy for such missions, including the ability to detect and recover from anomalies, and to perform missions based on pre-trained behaviors rather than hard-coded, predetermined logic like all prior space missions. We developed an AI-assisted, personality-driven, intelligent framework for control of an Ocean world mission by combining a mix of advanced technologies. To demonstrate the capabilities of the framework, we perform tests of autonomous sampling operations on a lander-manipulator testbed at the NASA Jet Propulsion Laboratory, approximating possible surface conditions such a mission might encounter. 

**Abstract (ZH)**: 我们鲁棒可解释的冰卫星科学探测自主性（REASIMO）项目为NASA的海洋世界生命探测技术（COLDTech）计划贡献力量，该计划探索诸如欧罗巴和恩赛lasses的海洋世界科学平台技术。海洋世界任务面临着重大操作挑战，包括长通信延迟、有限的电力供应以及由辐射损伤和恶劣环境引起的寿命限制。鉴于这些操作限制，未来的海洋世界任务将依赖于机载自主性。除了常规着陆器操作的管理之外，机载自主性还必须在出现异常时适当地作出反应。传统航天器依赖于一种“安全模式”转移，其中非必需的组件和子系统会被断电以保持安全并维持与地球的通信。对于时间极其有限的海洋世界任务，无需地球干预即可执行的异常解决策略对于完成任务目标和科学目标至关重要。为此，REASIMO项目旨在展示一种适用于此类任务的鲁棒人工智能辅助自主性，包括检测和恢复异常的能力，以及基于预先训练的行为而非所有先前太空任务中的硬编码预定逻辑来进行任务。我们通过结合多种先进技术，开发了一种人工智能辅助、个性驱动的智能控制框架来管理海洋世界任务。为了展示该框架的能力，我们在NASA喷气推进实验室的着陆器操作测试台上进行自主采样操作测试，模拟此类任务可能遇到的表面条件。 

---
# SkyVLN: Vision-and-Language Navigation and NMPC Control for UAVs in Urban Environments 

**Title (ZH)**: SkyVLN: 多智能体无人机在城市环境中的视觉-语言导航与NMPC控制 

**Authors**: Tianshun Li, Tianyi Huai, Zhen Li, Yichun Gao, Haoang Li, Xinhu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.06564)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) have emerged as versatile tools across various sectors, driven by their mobility and adaptability. This paper introduces SkyVLN, a novel framework integrating vision-and-language navigation (VLN) with Nonlinear Model Predictive Control (NMPC) to enhance UAV autonomy in complex urban environments. Unlike traditional navigation methods, SkyVLN leverages Large Language Models (LLMs) to interpret natural language instructions and visual observations, enabling UAVs to navigate through dynamic 3D spaces with improved accuracy and robustness. We present a multimodal navigation agent equipped with a fine-grained spatial verbalizer and a history path memory mechanism. These components allow the UAV to disambiguate spatial contexts, handle ambiguous instructions, and backtrack when necessary. The framework also incorporates an NMPC module for dynamic obstacle avoidance, ensuring precise trajectory tracking and collision prevention. To validate our approach, we developed a high-fidelity 3D urban simulation environment using AirSim, featuring realistic imagery and dynamic urban elements. Extensive experiments demonstrate that SkyVLN significantly improves navigation success rates and efficiency, particularly in new and unseen environments. 

**Abstract (ZH)**: 无人 aerial 车（UAVs）已成为各个领域中的多功能工具，得益于其移动性和适应性。本文介绍了 SkyVLN，这是一种将视觉-语言导航（VLN）与非线性模型预测控制（NMPC）集成的新框架，旨在提高 UAV 在复杂城市环境中的自主性。与传统导航方法不同，SkyVLN 利用大型语言模型（LLMs）解释自然语言指令和视觉观察，使 UAV 能够在动态 3D 空间中以更高的准确性和鲁棒性进行导航。文中提出了一种多模态导航代理，配备了细粒度的空间语言化器和历史路径记忆机制。这些组件使 UAV 能够消除空间歧义、处理含糊的指令并在必要时回退。该框架还引入了 NMPC 模块进行动态障碍物回避，以确保精确的轨迹跟踪和防碰撞。为了验证我们的方法，我们使用 AirSim 开发了一个高保真度的 3D 城市仿真环境，具备现实图像和动态城市元素。大量实验表明，SkyVLN 显著提高了导航成功率和效率，特别是在新且未见过的环境中。 

---
# KLEIYN : A Quadruped Robot with an Active Waist for Both Locomotion and Wall Climbing 

**Title (ZH)**: KLEIYN：具备主动腰部用于行进和攀壁的四足机器人 

**Authors**: Keita Yoneda, Kento Kawaharazuka, Temma Suzuki, Takahiro Hattori, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2507.06562)  

**Abstract**: In recent years, advancements in hardware have enabled quadruped robots to operate with high power and speed, while robust locomotion control using reinforcement learning (RL) has also been realized. As a result, expectations are rising for the automation of tasks such as material transport and exploration in unknown environments. However, autonomous locomotion in rough terrains with significant height variations requires vertical movement, and robots capable of performing such movements stably, along with their control methods, have not yet been fully established. In this study, we developed the quadruped robot KLEIYN, which features a waist joint, and aimed to expand quadruped locomotion by enabling chimney climbing through RL. To facilitate the learning of vertical motion, we introduced Contact-Guided Curriculum Learning (CGCL). As a result, KLEIYN successfully climbed walls ranging from 800 mm to 1000 mm in width at an average speed of 150 mm/s, 50 times faster than conventional robots. Furthermore, we demonstrated that the introduction of a waist joint improves climbing performance, particularly enhancing tracking ability on narrow walls. 

**Abstract (ZH)**: 近年来，硬件的进步使四足机器人能够以高功率和速度运行，同时利用强化学习（RL）实现 robust 的运动控制也已成为现实。因此，对诸如材料运输和未知环境探索等任务的自动化期望正在上升。然而，在具有显著高度变化的崎岖地形上实现自主运动仍然需要垂直运动，而能够稳定执行此类运动的机器人及其控制方法尚未完全建立。在这项研究中，我们开发了配备了腰部关节的四足机器人 KLEIYN，并通过 RL 实现烟囱攀爬，旨在通过 RL 扩展四足运动。为了促进垂直运动的学习，我们引入了接触引导 Curriculum Learning（CGCL）。结果显示，KLEIYN 成功以 150 mm/s 的平均速度攀爬了宽度从 800 mm 到 1000 mm 的墙面，比传统机器人快 50 倍。此外，我们展示了腰部关节的引入提高了攀爬性能，尤其是在狭壁跟踪能力方面的提升。 

---
# Failure Forecasting Boosts Robustness of Sim2Real Rhythmic Insertion Policies 

**Title (ZH)**: 失败预测增强Sim2Real律动插入策略的鲁棒性 

**Authors**: Yuhan Liu, Xinyu Zhang, Haonan Chang, Abdeslam Boularias  

**Link**: [PDF](https://arxiv.org/pdf/2507.06519)  

**Abstract**: This paper addresses the challenges of Rhythmic Insertion Tasks (RIT), where a robot must repeatedly perform high-precision insertions, such as screwing a nut into a bolt with a wrench. The inherent difficulty of RIT lies in achieving millimeter-level accuracy and maintaining consistent performance over multiple repetitions, particularly when factors like nut rotation and friction introduce additional complexity. We propose a sim-to-real framework that integrates a reinforcement learning-based insertion policy with a failure forecasting module. By representing the wrench's pose in the nut's coordinate frame rather than the robot's frame, our approach significantly enhances sim-to-real transferability. The insertion policy, trained in simulation, leverages real-time 6D pose tracking to execute precise alignment, insertion, and rotation maneuvers. Simultaneously, a neural network predicts potential execution failures, triggering a simple recovery mechanism that lifts the wrench and retries the insertion. Extensive experiments in both simulated and real-world environments demonstrate that our method not only achieves a high one-time success rate but also robustly maintains performance over long-horizon repetitive tasks. 

**Abstract (ZH)**: 本文探讨了节律插入任务（RIT）的挑战，其中机器人必须重复执行高精度插入操作，例如使用扳手将螺母拧入螺栓。RIT 的固有难度在于实现毫米级精度并保持多次重复操作的一致性能，尤其是在螺母旋转和摩擦等因素增加复杂性的情况下。我们提出了一种结合基于强化学习的插入策略和故障预测模块的仿真到现实框架。通过以螺母坐标系而不是机器人坐标系表示扳手的姿态，我们的方法显著提高了仿真到现实的可转移性。在仿真中训练的插入策略利用实时六自由度姿态跟踪执行精确对齐、插入和旋转操作。同时，神经网络预测潜在执行失败，触发简单的恢复机制以提起扳手并重新尝试插入。在仿真和真实环境中的大量实验表明，我们的方法不仅实现了一次高成功率，还能够在长时间尺度的重复任务中稳健保持性能。 

---
# Evaluating Robots Like Human Infants: A Case Study of Learned Bipedal Locomotion 

**Title (ZH)**: 评价机器人如人类婴儿般：双足运动学习的案例研究 

**Authors**: Devin Crowley, Whitney G. Cole, Christina M. Hospodar, Ruiting Shen, Karen E. Adolph, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2507.06426)  

**Abstract**: Typically, learned robot controllers are trained via relatively unsystematic regimens and evaluated with coarse-grained outcome measures such as average cumulative reward. The typical approach is useful to compare learning algorithms but provides limited insight into the effects of different training regimens and little understanding about the richness and complexity of learned behaviors. Likewise, human infants and other animals are "trained" via unsystematic regimens, but in contrast, developmental psychologists evaluate their performance in highly-controlled experiments with fine-grained measures such as success, speed of walking, and prospective adjustments. However, the study of learned behavior in human infants is limited by the practical constraints of training and testing babies. Here, we present a case study that applies methods from developmental psychology to study the learned behavior of the simulated bipedal robot Cassie. Following research on infant walking, we systematically designed reinforcement learning training regimens and tested the resulting controllers in simulated environments analogous to those used for babies--but without the practical constraints. Results reveal new insights into the behavioral impact of different training regimens and the development of Cassie's learned behaviors relative to infants who are learning to walk. This interdisciplinary baby-robot approach provides inspiration for future research designed to systematically test effects of training on the development of complex learned robot behaviors. 

**Abstract (ZH)**: 通常，机器人控制器是通过相对非系统的训练程序进行训练，并且通过粗糙的结果指标，如平均累积奖励来进行评估。通常的方法有助于比较学习算法，但提供了有限的关于不同训练程序效果的洞察，并且对学到的行为的丰富性和复杂性缺乏理解。同样，人类婴儿和其他动物也是通过非系统的程序进行“训练”，但相比之下，发展心理学家通过精细测量，如成功、行走速度和前瞻性调整，在高度控制的实验中评估他们的表现。然而，对人类婴儿学到的行为的研究受限于培训和测试婴儿的实际约束。在这里，我们呈现了一个案例研究，将发展心理学的方法应用于研究模拟双足机器人Cassie学到的行为。借鉴婴儿行走的研究，我们系统地设计了强化学习训练程序，并在模拟环境中测试了生成的控制器，这些模拟环境类似于婴儿使用的环境——但没有实际约束。结果揭示了不同训练程序对行为影响的新见解，以及Cassie学到的行为与其正在学习行走的婴儿相比的发展情况。这种跨学科的婴儿-机器人方法为未来旨在系统测试训练对复杂机器人行为发展影响的研究提供了 inspiration。 

---
# Learning to Evaluate Autonomous Behaviour in Human-Robot Interaction 

**Title (ZH)**: 学习评估自主行为的人机交互 

**Authors**: Matteo Tiezzi, Tommaso Apicella, Carlos Cardenas-Perez, Giovanni Fregonese, Stefano Dafarra, Pietro Morerio, Daniele Pucci, Alessio Del Bue  

**Link**: [PDF](https://arxiv.org/pdf/2507.06404)  

**Abstract**: Evaluating and comparing the performance of autonomous Humanoid Robots is challenging, as success rate metrics are difficult to reproduce and fail to capture the complexity of robot movement trajectories, critical in Human-Robot Interaction and Collaboration (HRIC). To address these challenges, we propose a general evaluation framework that measures the quality of Imitation Learning (IL) methods by focusing on trajectory performance. We devise the Neural Meta Evaluator (NeME), a deep learning model trained to classify actions from robot joint trajectories. NeME serves as a meta-evaluator to compare the performance of robot control policies, enabling policy evaluation without requiring human involvement in the loop. We validate our framework on ergoCub, a humanoid robot, using teleoperation data and comparing IL methods tailored to the available platform. The experimental results indicate that our method is more aligned with the success rate obtained on the robot than baselines, offering a reproducible, systematic, and insightful means for comparing the performance of multimodal imitation learning approaches in complex HRI tasks. 

**Abstract (ZH)**: 评估和比较自主 humanoid 机器人的性能具有挑战性，因为成功率指标难以重现且无法捕捉人类-机器人交互与协作（HRIC）中机器人运动轨迹的复杂性。为应对这些挑战，我们提出了一种通用评估框架，通过关注轨迹性能来衡量模仿学习（IL）方法的质量。我们设计了神经元元评估器（NeME），这是一种用于分类机器人关节轨迹的动作的深度学习模型。NeME 作为元评估器，用于比较机器人控制策略的性能，使其能够在循环中无需人类参与即可进行策略评估。我们在使用遥控操作数据和针对可用平台定制的 IL 方法进行比较的 ergoCub 人形机器人上验证了该框架。实验结果表明，我们的方法与机器人上获得的成功率更为一致，提供了一种可重现、系统且具有洞察力的方法，用于比较复杂HRIC任务中多模态模仿学习方法的性能。 

---
# Mapping the Catacombs: An Underwater Cave Segment of the Devil's Eye System 

**Title (ZH)**: 绘制Devil's Eye系统中的水下洞穴段落：卡那封墓穴测绘 

**Authors**: Michalis Chatzispyrou, Luke Horgan, Hyunkil Hwang, Harish Sathishchandra, Monika Roznere, Alberto Quattrini Li, Philippos Mordohai, Ioannis Rekleitis  

**Link**: [PDF](https://arxiv.org/pdf/2507.06397)  

**Abstract**: This paper presents a framework for mapping underwater caves. Underwater caves are crucial for fresh water resource management, underwater archaeology, and hydrogeology. Mapping the cave's outline and dimensions, as well as creating photorealistic 3D maps, is critical for enabling a better understanding of this underwater domain. In this paper, we present the mapping of an underwater cave segment (the catacombs) of the Devil's Eye cave system at Ginnie Springs, FL. We utilized a set of inexpensive action cameras in conjunction with a dive computer to estimate the trajectories of the cameras together with a sparse point cloud. The resulting reconstructions are utilized to produce a one-dimensional retract of the cave passages in the form of the average trajectory together with the boundaries (top, bottom, left, and right). The use of the dive computer enables the observability of the z-dimension in addition to the roll and pitch in a visual/inertial framework (SVIn2). In addition, the keyframes generated by SVIn2 together with the estimated camera poses for select areas are used as input to a global optimization (bundle adjustment) framework -- COLMAP -- in order to produce a dense reconstruction of those areas. The same cave segment is manually surveyed using the MNemo V2 instrument, providing an additional set of measurements validating the proposed approach. It is worth noting that with the use of action cameras, the primary components of a cave map can be constructed. Furthermore, with the utilization of a global optimization framework guided by the results of VI-SLAM package SVIn2, photorealistic dense 3D representations of selected areas can be reconstructed. 

**Abstract (ZH)**: 本文提出了一种海底洞穴测绘的框架。海底洞穴对于淡水资源管理、水下考古和水文地质学至关重要。准确测绘洞穴的轮廓和尺寸，以及创建高保真3D地图，对于更好地理解这一水下领域至关重要。本文介绍了对位于美国佛罗里达州冈尼斯泉的恶魔之眼洞穴系统中一段地下洞穴（冥府洞）的测绘。我们利用一套便宜的运动相机结合潜水计算机来估计相机的轨迹，同时生成稀疏点云。由此产生的重建用于生成洞穴通道的一维收缩，形式为平均轨迹及其边界（顶部、底部、左边和右边）。潜水计算机的使用使得除了俯仰和滚转之外还能够观测到垂直维度。此外，SVIn2视觉惯性SLAM包生成的关键帧以及选定区域的估计相机姿态作为输入，被用于全局优化框架——COLMAP——以生成这些区域的密集重建。同一洞穴段还使用MNemo V2仪器进行人工测绘，提供了验证所提方法的有效性的额外测量数据。值得一提的是，使用运动相机可以构成洞穴地图的主要成分，同时结合全局优化框架SVIn2的结果，可以重建选定区域的高保真密集3D表示。 

---
# Solving the Constrained Random Disambiguation Path Problem via Lagrangian Relaxation and Graph Reduction 

**Title (ZH)**: 基于拉格rangian松弛和图减约的约束随机消歧路径问题求解方法 

**Authors**: Li Zhou, Elvan Ceyhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06346)  

**Abstract**: We study a resource-constrained variant of the Random Disambiguation Path (RDP) problem, a generalization of the Stochastic Obstacle Scene (SOS) problem, in which a navigating agent must reach a target in a spatial environment populated with uncertain obstacles. Each ambiguous obstacle may be disambiguated at a (possibly) heterogeneous resource cost, subject to a global disambiguation budget. We formulate this constrained planning problem as a Weight-Constrained Shortest Path Problem (WCSPP) with risk-adjusted edge costs that incorporate probabilistic blockage and traversal penalties. To solve it, we propose a novel algorithmic framework-COLOGR-combining Lagrangian relaxation with a two-phase vertex elimination (TPVE) procedure. The method prunes infeasible and suboptimal paths while provably preserving the optimal solution, and leverages dual bounds to guide efficient search. We establish correctness, feasibility guarantees, and surrogate optimality under mild assumptions. Our analysis also demonstrates that COLOGR frequently achieves zero duality gap and offers improved computational complexity over prior constrained path-planning methods. Extensive simulation experiments validate the algorithm's robustness across varying obstacle densities, sensor accuracies, and risk models, consistently outperforming greedy baselines and approaching offline-optimal benchmarks. The proposed framework is broadly applicable to stochastic network design, mobility planning, and constrained decision-making under uncertainty. 

**Abstract (ZH)**: 我们研究了一种资源约束下的随机去歧义路径（RDP）问题变体，这是一种随机障碍场景（SOS）问题的一般化，在其中导航代理必须在一个充满不确定障碍的三维环境中到达目标。每个模糊障碍物可能需要在（可能）异质成本下进行去歧义处理，且受到全局去歧义预算的限制。我们将该约束规划问题形式化为带风险调整边成本的加权约束最短路径问题（WCSPP）。为解决该问题，我们提出了一种结合拉格朗日松弛与两阶段顶点消除（TPVE）过程的新型算法框架-COLOGR。该方法在证明保留最优解的同时修剪不可行和次优路径，并利用对偶界引导高效搜索。在温和假设下，我们证明了该方法的正确性和可行性保证，以及替代最优性。我们的分析还表明，COLOGR 经常实现零对偶间隙，并在计算复杂度方面优于先前的约束路径规划方法。广泛的仿真实验验证了算法在不同障碍密度、传感器精度和风险模型下的鲁棒性，始终优于贪婪 baseline 并接近离线最优基准。所提出的框架广泛适用于随机网络设计、移动规划和不确定性下的约束决策。 

---
# Graph-Based Complexity Metrics for Multi-Agent Curriculum Learning: A Validated Approach to Task Ordering in Cooperative Coordination Environments 

**Title (ZH)**: 基于图的复杂度度量方法：一种在协同协调环境中的任务排序验证方法 

**Authors**: Farhaan Ebadulla, Dharini Hindlatti, Srinivaasan NS, Apoorva VH, Ayman Aftab  

**Link**: [PDF](https://arxiv.org/pdf/2507.07074)  

**Abstract**: Multi-agent reinforcement learning (MARL) faces significant challenges in task sequencing and curriculum design, particularly for cooperative coordination scenarios. While curriculum learning has demonstrated success in single-agent domains, principled approaches for multi-agent coordination remain limited due to the absence of validated task complexity metrics. This approach presents a graph-based coordination complexity metric that integrates agent dependency entropy, spatial interference patterns, and goal overlap analysis to predict task difficulty in multi-agent environments. The complexity metric achieves strong empirical validation with rho = 0.952 correlation (p < 0.001) between predicted complexity and empirical difficulty determined by random agent performance evaluation. This approach evaluates the curriculum learning framework using MADDPG across two distinct coordination environments: achieving 56x performance improvement in tight coordination tasks (MultiWalker) and demonstrating systematic task progression in cooperative navigation (Simple Spread). Through systematic analysis, coordination tightness emerges as a predictor of curriculum learning effectiveness, where environments requiring strict agent interdependence benefit substantially from structured progression. This approach provides a validated complexity metric for multi-agent curriculum design and establishes empirical guidelines for multi-robot coordination applications. 

**Abstract (ZH)**: 基于图的合作复杂性度量在多agent强化学习中的 Curriculum 设计与任务排序 

---
# When Context Is Not Enough: Modeling Unexplained Variability in Car-Following Behavior 

**Title (ZH)**: 当背景信息不足以解释时：建模车辆跟随行为中的未解释变异性 

**Authors**: Chengyuan Zhang, Zhengbing He, Cathy Wu, Lijun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.07012)  

**Abstract**: Modeling car-following behavior is fundamental to microscopic traffic simulation, yet traditional deterministic models often fail to capture the full extent of variability and unpredictability in human driving. While many modern approaches incorporate context-aware inputs (e.g., spacing, speed, relative speed), they frequently overlook structured stochasticity that arises from latent driver intentions, perception errors, and memory effects -- factors that are not directly observable from context alone. To fill the gap, this study introduces an interpretable stochastic modeling framework that captures not only context-dependent dynamics but also residual variability beyond what context can explain. Leveraging deep neural networks integrated with nonstationary Gaussian processes (GPs), our model employs a scenario-adaptive Gibbs kernel to learn dynamic temporal correlations in acceleration decisions, where the strength and duration of correlations between acceleration decisions evolve with the driving context. This formulation enables a principled, data-driven quantification of uncertainty in acceleration, speed, and spacing, grounded in both observable context and latent behavioral variability. Comprehensive experiments on the naturalistic vehicle trajectory dataset collected from the German highway, i.e., the HighD dataset, demonstrate that the proposed stochastic simulation method within this framework surpasses conventional methods in both predictive performance and interpretable uncertainty quantification. The integration of interpretability and accuracy makes this framework a promising tool for traffic analysis and safety-critical applications. 

**Abstract (ZH)**: 基于可解释的随机建模框架模拟跟车行为：传统确定性模型往往无法捕捉人类驾驶的全部变异性与不可预测性，而现代方法虽考虑上下文信息但常忽视潜在的结构化随机性。为此，本研究提出了一种可解释的随机建模框架，不仅捕捉上下文相关的动态变化，还捕捉上下文无法解释的残余变异性。该模型利用深度神经网络与非平稳高斯过程结合，采用场景自适应吉布斯核学习加速度决策的动力学时变相关性，其中相关性的强度和持续时间随驾驶上下文变化。这种建模框架能够建立在可观察上下文与潜在行为变异性基础上，实现加速度、速度和间距的不确定性原则性、数据驱动量化。在德国高速公路上采集的自然车辆轨迹数据集HighD上进行的全面实验表明，该框架内的随机仿真方法在预测性能和可解释的不确定性量化方面均优于传统方法。该框架结合了可解释性和准确性，在交通分析和安全关键应用方面具有潜力。 

---
# Hallucinating 360°: Panoramic Street-View Generation via Local Scenes Diffusion and Probabilistic Prompting 

**Title (ZH)**: 全景街景生成：基于局部场景扩散和概率性提示的虚拟视角生成 

**Authors**: Fei Teng, Kai Luo, Sheng Wu, Siyu Li, Pujun Guo, Jiale Wei, Kunyu Peng, Jiaming Zhang, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06971)  

**Abstract**: Panoramic perception holds significant potential for autonomous driving, enabling vehicles to acquire a comprehensive 360° surround view in a single shot. However, autonomous driving is a data-driven task. Complete panoramic data acquisition requires complex sampling systems and annotation pipelines, which are time-consuming and labor-intensive. Although existing street view generation models have demonstrated strong data regeneration capabilities, they can only learn from the fixed data distribution of existing datasets and cannot achieve high-quality, controllable panoramic generation. In this paper, we propose the first panoramic generation method Percep360 for autonomous driving. Percep360 enables coherent generation of panoramic data with control signals based on the stitched panoramic data. Percep360 focuses on two key aspects: coherence and controllability. Specifically, to overcome the inherent information loss caused by the pinhole sampling process, we propose the Local Scenes Diffusion Method (LSDM). LSDM reformulates the panorama generation as a spatially continuous diffusion process, bridging the gaps between different data distributions. Additionally, to achieve the controllable generation of panoramic images, we propose a Probabilistic Prompting Method (PPM). PPM dynamically selects the most relevant control cues, enabling controllable panoramic image generation. We evaluate the effectiveness of the generated images from three perspectives: image quality assessment (i.e., no-reference and with reference), controllability, and their utility in real-world Bird's Eye View (BEV) segmentation. Notably, the generated data consistently outperforms the original stitched images in no-reference quality metrics and enhances downstream perception models. The source code will be publicly available at this https URL. 

**Abstract (ZH)**: 全景感知在自动驾驶中具有重要潜力，能够使车辆在单次拍摄中获得全面的360° Surround View。然而，自动驾驶是一项数据驱动的任务。完整的全景数据获取需要复杂的采样系统和注释管道，这既耗时又劳动密集。尽管现有的街景生成模型展示了强大的数据再生能力，但它们只能从现有数据集的固定数据分布中学习，无法实现高质量、可控的全景生成。在本文中，我们提出了第一个适用于自动驾驶的全景生成方法Percep360。Percep360能够基于拼接的全景数据进行具有控制信号的连贯生成。Percep360重点关注两个关键方面：连贯性和可控性。具体地，为了克服针孔采样过程中的固有信息损失，我们提出了局部场景扩散方法（LSDM）。LSDM将全景生成重新定义为一个空间连续的扩散过程，弥合了不同数据分布之间的差距。此外，为了实现全景图的可控生成，我们提出了概率提示方法（PPM）。PPM动态选择最相关的控制线索，实现可控的全景图像生成。我们从三个角度评估生成图像的有效性：图像质量评估（无参考和有参考）、可控性及其在真实世界Bird's Eye View (BEV)分割中的实用性。值得注意的是，生成的数据在无参考质量指标中始终优于原始拼接图像，并能增强下游感知模型。源代码将在此URL公开。 

---
# A Neural Representation Framework with LLM-Driven Spatial Reasoning for Open-Vocabulary 3D Visual Grounding 

**Title (ZH)**: 基于LLM驱动空间推理的神经表示框架在开放式词汇3D视觉接地中的应用 

**Authors**: Zhenyang Liu, Sixiao Zheng, Siyu Chen, Cairong Zhao, Longfei Liang, Xiangyang Xue, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06719)  

**Abstract**: Open-vocabulary 3D visual grounding aims to localize target objects based on free-form language queries, which is crucial for embodied AI applications such as autonomous navigation, robotics, and augmented reality. Learning 3D language fields through neural representations enables accurate understanding of 3D scenes from limited viewpoints and facilitates the localization of target objects in complex environments. However, existing language field methods struggle to accurately localize instances using spatial relations in language queries, such as ``the book on the chair.'' This limitation mainly arises from inadequate reasoning about spatial relations in both language queries and 3D scenes. In this work, we propose SpatialReasoner, a novel neural representation-based framework with large language model (LLM)-driven spatial reasoning that constructs a visual properties-enhanced hierarchical feature field for open-vocabulary 3D visual grounding. To enable spatial reasoning in language queries, SpatialReasoner fine-tunes an LLM to capture spatial relations and explicitly infer instructions for the target, anchor, and spatial relation. To enable spatial reasoning in 3D scenes, SpatialReasoner incorporates visual properties (opacity and color) to construct a hierarchical feature field. This field represents language and instance features using distilled CLIP features and masks extracted via the Segment Anything Model (SAM). The field is then queried using the inferred instructions in a hierarchical manner to localize the target 3D instance based on the spatial relation in the language query. Extensive experiments show that our framework can be seamlessly integrated into different neural representations, outperforming baseline models in 3D visual grounding while empowering their spatial reasoning capability. 

**Abstract (ZH)**: 基于开放词汇的3D视觉定位旨在根据自由形式的语言查询定位目标物体，这对于自主导航、机器人技术和增强现实等嵌入式AI应用至关重要。通过神经表示学习3D语言场能够从有限视角准确理解3D场景，并简化在复杂环境中定位目标物体的过程。然而，现有语言场方法难以使用语言查询中的空间关系（如“书在椅子上”）准确定位实例。这一局限主要源于对语言查询和3D场景中空间关系推理的不足。在本文中，我们提出SpatialReasoner，这是一种基于神经表示的新颖框架，采用大型语言模型（LLM）驱动的空间推理来构建增强视觉属性的分层次特征场以实现开放词汇的3D视觉定位。为了在语言查询中进行空间推理，SpatialReasoner对LLM进行微调以捕捉空间关系并明确推断目标、锚点和空间关系的指令。为了在3D场景中进行空间推理，SpatialReasoner将视觉属性（不透明度和颜色）纳入分层次特征场的构建中。该场使用从中断的CLIP特征和通过Segment Anything Model (SAM)提取的掩码表示语言和实例特征，并以分层次方式查询这些指令以基于语言查询中的空间关系定位目标3D实例。广泛实验表明，我们的框架可以无缝集成到不同的神经表示中，在3D视觉定位方面优于基线模型，同时增强了它们的空间推理能力。 

---
# StixelNExT++: Lightweight Monocular Scene Segmentation and Representation for Collective Perception 

**Title (ZH)**: StixelNExT++：轻量级单目场景分割与表示及其在集体感知中的应用 

**Authors**: Marcel Vosshans, Omar Ait-Aider, Youcef Mezouar, Markus Enzweiler  

**Link**: [PDF](https://arxiv.org/pdf/2507.06687)  

**Abstract**: This paper presents StixelNExT++, a novel approach to scene representation for monocular perception systems. Building on the established Stixel representation, our method infers 3D Stixels and enhances object segmentation by clustering smaller 3D Stixel units. The approach achieves high compression of scene information while remaining adaptable to point cloud and bird's-eye-view representations. Our lightweight neural network, trained on automatically generated LiDAR-based ground truth, achieves real-time performance with computation times as low as 10 ms per frame. Experimental results on the Waymo dataset demonstrate competitive performance within a 30-meter range, highlighting the potential of StixelNExT++ for collective perception in autonomous systems. 

**Abstract (ZH)**: StixelNExT++：一种用于单目感知系统的新型场景表示方法 

---
# MK-Pose: Category-Level Object Pose Estimation via Multimodal-Based Keypoint Learning 

**Title (ZH)**: MK-Pose：基于多模态关键点学习的类别级对象姿态估计 

**Authors**: Yifan Yang, Peili Song, Enfan Lan, Dong Liu, Jingtai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06662)  

**Abstract**: Category-level object pose estimation, which predicts the pose of objects within a known category without prior knowledge of individual instances, is essential in applications like warehouse automation and manufacturing. Existing methods relying on RGB images or point cloud data often struggle with object occlusion and generalization across different instances and categories. This paper proposes a multimodal-based keypoint learning framework (MK-Pose) that integrates RGB images, point clouds, and category-level textual descriptions. The model uses a self-supervised keypoint detection module enhanced with attention-based query generation, soft heatmap matching and graph-based relational modeling. Additionally, a graph-enhanced feature fusion module is designed to integrate local geometric information and global context. MK-Pose is evaluated on CAMERA25 and REAL275 dataset, and is further tested for cross-dataset capability on HouseCat6D dataset. The results demonstrate that MK-Pose outperforms existing state-of-the-art methods in both IoU and average precision without shape priors. Codes will be released at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于多模态的关键点学习框架（MK-Pose）：类别级物体姿态估计 

---
# VisioPath: Vision-Language Enhanced Model Predictive Control for Safe Autonomous Navigation in Mixed Traffic 

**Title (ZH)**: VisioPath: 视觉语言增强的模型预测控制方法以实现混合交通中的安全自主导航 

**Authors**: Shanting Wang, Panagiotis Typaldos, Chenjun Li, Andreas A. Malikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2507.06441)  

**Abstract**: In this paper, we introduce VisioPath, a novel framework combining vision-language models (VLMs) with model predictive control (MPC) to enable safe autonomous driving in dynamic traffic environments. The proposed approach leverages a bird's-eye view video processing pipeline and zero-shot VLM capabilities to obtain structured information about surrounding vehicles, including their positions, dimensions, and velocities. Using this rich perception output, we construct elliptical collision-avoidance potential fields around other traffic participants, which are seamlessly integrated into a finite-horizon optimal control problem for trajectory planning. The resulting trajectory optimization is solved via differential dynamic programming with an adaptive regularization scheme and is embedded in an event-triggered MPC loop. To ensure collision-free motion, a safety verification layer is incorporated in the framework that provides an assessment of potential unsafe trajectories. Extensive simulations in Simulation of Urban Mobility (SUMO) demonstrate that VisioPath outperforms conventional MPC baselines across multiple metrics. By combining modern AI-driven perception with the rigorous foundation of optimal control, VisioPath represents a significant step forward in safe trajectory planning for complex traffic systems. 

**Abstract (ZH)**: 基于视觉-语言模型与模型预测控制的VisioPath框架：动态交通环境中的安全自主驾驶 

---
# Self-supervised learning predicts plant growth trajectories from multi-modal industrial greenhouse data 

**Title (ZH)**: 自我监督学习预测多模态工业温室数据中的植物生长轨迹 

**Authors**: Adam J Riesselman, Evan M Cofer, Therese LaRue, Wim Meeussen  

**Link**: [PDF](https://arxiv.org/pdf/2507.06336)  

**Abstract**: Quantifying organism-level phenotypes, such as growth dynamics and biomass accumulation, is fundamental to understanding agronomic traits and optimizing crop production. However, quality growing data of plants at scale is difficult to generate. Here we use a mobile robotic platform to capture high-resolution environmental sensing and phenotyping measurements of a large-scale hydroponic leafy greens system. We describe a self-supervised modeling approach to build a map from observed growing data to the entire plant growth trajectory. We demonstrate our approach by forecasting future plant height and harvest mass of crops in this system. This approach represents a significant advance in combining robotic automation and machine learning, as well as providing actionable insights for agronomic research and operational efficiency. 

**Abstract (ZH)**: 利用移动机器人平台量化大规模 hydroponic 叶菜系统中植物的生长动态和生物量积累是理解农艺性状和优化作物生产的基础。然而，生成高质量的植物生长数据具有挑战性。我们使用移动机器人平台捕获大规模水培叶菜系统的高分辨率环境感知和表型测量数据。我们描述了一种半监督建模方法，从观测到的生长数据构建整个植物生长轨迹的地图。我们通过预测该系统中作物未来植株高度和收获质量来展示这种方法。该方法代表了将机器人自动化和机器学习相结合的重要进步，同时也为农艺研究和运营效率提供了实用见解。 

---
