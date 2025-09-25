# A Biomimetic Vertebraic Soft Robotic Tail for High-Speed, High-Force Dynamic Maneuvering 

**Title (ZH)**: 一种用于高-speed高-force动态机动的仿生脊椎软体尾部 

**Authors**: Sicong Liu, Jianhui Liu, Fang Chen, Wenjian Yang, Juan Yi, Yu Zheng, Zheng Wang, Wanchao Chi, Chaoyang Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.20219)  

**Abstract**: Robotic tails can enhance the stability and maneuverability of mobile robots, but current designs face a trade-off between the power of rigid systems and the safety of soft ones. Rigid tails generate large inertial effects but pose risks in unstructured environments, while soft tails lack sufficient speed and force. We present a Biomimetic Vertebraic Soft Robotic (BVSR) tail that resolves this challenge through a compliant pneumatic body reinforced by a passively jointed vertebral column inspired by musculoskeletal structures. This hybrid design decouples load-bearing and actuation, enabling high-pressure actuation (up to 6 bar) for superior dynamics while preserving compliance. A dedicated kinematic and dynamic model incorporating vertebral constraints is developed and validated experimentally. The BVSR tail achieves angular velocities above 670°/s and generates inertial forces and torques up to 5.58 N and 1.21 Nm, indicating over 200% improvement compared to non-vertebraic designs. Demonstrations on rapid cart stabilization, obstacle negotiation, high-speed steering, and quadruped integration confirm its versatility and practical utility for agile robotic platforms. 

**Abstract (ZH)**: Biomimetic Vertebraic Soft Robotic (BVSR) 尾部可以增强移动机器人的稳定性和机动性，但当前设计在刚性系统的功率和软性系统的安全性之间存在权衡。刚性尾巴会产生较大的惯性效应，但在未结构化环境中存在风险，而软性尾巴缺乏足够的速度和力量。我们提出了一种受生物结构启发的 Biomimetic Vertebraic Soft Robotic (BVSR) 尾部，通过借鉴肌肉骨骼结构的被动关节椎柱加强可变形气动主体来解决这一挑战。这种混合设计解耦了承载和驱动，能够实现高达 6 巴的高压驱动（从而提供卓越的动力学性能）同时保持可变形性。开发并实验验证了一个包含椎体约束的专用运动学和动力学模型。BVSR 尾部实现了超过 670°/s 的角速度，并产生了高达 5.58 N 的惯性力和 1.21 Nm 的惯性扭矩，显示出与无椎体设计相比超过 200% 的改进。快速手推车稳定、障碍物导航、高速转向和四足动物集成的演示证实了其在灵活动作机器人平台中的多样性和实用性。 

---
# Orbital Stabilization and Time Synchronization of Unstable Periodic Motions in Underactuated Robots 

**Title (ZH)**: 欠驱动机器人中不稳定周期运动的轨道稳定与时间同步 

**Authors**: Surov Maksim  

**Link**: [PDF](https://arxiv.org/pdf/2509.20082)  

**Abstract**: This paper presents a control methodology for achieving orbital stabilization with simultaneous time synchronization of periodic trajectories in underactuated robotic systems. The proposed approach extends the classical transverse linearization framework to explicitly incorporate time-desynchronization dynamics. To stabilize the resulting extended transverse dynamics, we employ a combination of time-varying LQR and sliding-mode control. The theoretical results are validated experimentally through the implementation of both centralized and decentralized control strategies on a group of six Butterfly robots. 

**Abstract (ZH)**: 本文提出一种控制方法，用于实现欠驱动机器人系统中轨道稳定性和周期轨迹时间同步的同时控制。所提出的方案扩展了经典的横向线性化框架，明确纳入了时间脱同步动力学。为稳定扩展的横向动力学，我们采用了时间变增益LQR与滑模控制的组合方法。理论结果通过在六只蝴蝶机器人上实现集中式和分布式控制策略进行实验验证。 

---
# Lidar-based Tracking of Traffic Participants with Sensor Nodes in Existing Urban Infrastructure 

**Title (ZH)**: 基于现有城市基础设施中的传感器节点的 Lidar 轨迹跟踪方法 

**Authors**: Simon Schäfer, Bassam Alrifaee, Ehsan Hashemi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20009)  

**Abstract**: This paper presents a lidar-only state estimation and tracking framework, along with a roadside sensing unit for integration with existing urban infrastructure. Urban deployments demand scalable, real-time tracking solutions, yet traditional remote sensing remains costly and computationally intensive, especially under perceptually degraded conditions. Our sensor node couples a single lidar with an edge computing unit and runs a computationally efficient, GPU-free observer that simultaneously estimates object state, class, dimensions, and existence probability. The pipeline performs: (i) state updates via an extended Kalman filter, (ii) dimension estimation using a 1D grid-map/Bayesian update, (iii) class updates via a lookup table driven by the most probable footprint, and (iv) existence estimation from track age and bounding-box consistency. Experiments in dynamic urban-like scenes with diverse traffic participants demonstrate real-time performance and high precision: The complete end-to-end pipeline finishes within \SI{100}{\milli\second} for \SI{99.88}{\%} of messages, with an excellent detection rate. Robustness is further confirmed under simulated wind and sensor vibration. These results indicate that reliable, real-time roadside tracking is feasible on CPU-only edge hardware, enabling scalable, privacy-friendly deployments within existing city infrastructure. The framework integrates with existing poles, traffic lights, and buildings, reducing deployment costs and simplifying large-scale urban rollouts and maintenance efforts. 

**Abstract (ZH)**: 基于激光雷达的实时路侧状态估计与跟踪框架：与现有城市基础设施的集成 

---
# An effective control of large systems of active particles: An application to evacuation problem 

**Title (ZH)**: 大型活性粒子系统的有效控制：撤离问题的应用 

**Authors**: Albina Klepach, Egor E. Nuzhin, Alexey A. Tsukanov, Nikolay V. Brilliantov  

**Link**: [PDF](https://arxiv.org/pdf/2509.19972)  

**Abstract**: Manipulation of large systems of active particles is a serious challenge across diverse domains, including crowd management, control of robotic swarms, and coordinated material transport. The development of advanced control strategies for complex scenarios is hindered, however, by the lack of scalability and robustness of the existing methods, in particular, due to the need of an individual control for each agent. One possible solution involves controlling a system through a leader or a group of leaders, which other agents tend to follow. Using such an approach we develop an effective control strategy for a leader, combining reinforcement learning (RL) with artificial forces acting on the system. To describe the guidance of active particles by a leader we introduce the generalized Vicsek model. This novel method is then applied to the problem of the effective evacuation by a robot-rescuer (leader) of large groups of people from hazardous places. We demonstrate, that while a straightforward application of RL yields suboptimal results, even for advanced architectures, our approach provides a robust and efficient evacuation strategy. The source code supporting this study is publicly available at: this https URL. 

**Abstract (ZH)**: 大规模活性颗粒系统的操控在 crowd 管理、机器人集群控制以及协同材料运输等领域是一项严重挑战。然而，由于现有方法缺乏可扩展性和鲁棒性，特别是在需要为每个代理个体化控制的情况下，复杂场景下的高级控制策略的发展受到阻碍。一种可能的解决方案是通过领导者或一群领导者控制系统，其他代理倾向于跟随。采用这种方法，我们结合强化学习（RL）和对系统施加的人工力，开发了领导者的有效控制策略。为描述领导者对活性颗粒的引导，我们引入了广义 Vicsek 模型。随后，我们应用此新方法解决由机器人救援者（领导者）有效疏散大量人群的问题。我们证明，即使对于先进的架构，直接应用 RL 也会导致次优结果，而我们的方法则提供了鲁棒且高效的疏散策略。此研究的支持代码可在以下链接获取：this https URL。 

---
# Robot Trajectron V2: A Probabilistic Shared Control Framework for Navigation 

**Title (ZH)**: Robot Trajectron V2: 一种概率共享控制导航框架 

**Authors**: Pinhao Song, Yurui Du, Ophelie Saussus, Sofie De Schrijver, Irene Caprara, Peter Janssen, Renaud Detry  

**Link**: [PDF](https://arxiv.org/pdf/2509.19954)  

**Abstract**: We propose a probabilistic shared-control solution for navigation, called Robot Trajectron V2 (RT-V2), that enables accurate intent prediction and safe, effective assistance in human-robot interaction. RT-V2 jointly models a user's long-term behavioral patterns and their noisy, low-dimensional control signals by combining a prior intent model with a posterior update that accounts for real-time user input and environmental context. The prior captures the multimodal and history-dependent nature of user intent using recurrent neural networks and conditional variational autoencoders, while the posterior integrates this with uncertain user commands to infer desired actions. We conduct extensive experiments to validate RT-V2 across synthetic benchmarks, human-computer interaction studies with keyboard input, and brain-machine interface experiments with non-human primates. Results show that RT-V2 outperforms the state of the art in intent estimation, provides safe and efficient navigation support, and adequately balances user autonomy with assistive intervention. By unifying probabilistic modeling, reinforcement learning, and safe optimization, RT-V2 offers a principled and generalizable approach to shared control for diverse assistive technologies. 

**Abstract (ZH)**: 一种用于导航的概率共享控制解决方案：机器人轨迹tron V2（RT-V2），该方案能够在人机交互中实现准确的意图预测和安全有效的辅助。 

---
# Trajectory Planning Using Safe Ellipsoidal Corridors as Projections of Orthogonal Trust Regions 

**Title (ZH)**: 使用正交信任区域投影的安全椭球走廊轨迹规划 

**Authors**: Akshay Jaitly, Jon Arrizabalaga, Guanrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19734)  

**Abstract**: Planning collision free trajectories in complex environments remains a core challenge in robotics. Existing corridor based planners which rely on decomposition of the free space into collision free subsets scale poorly with environmental complexity and require explicit allocations of time windows to trajectory segments. We introduce a new trajectory parameterization that represents trajectories in a nonconvex collision free corridor as being in a convex cartesian product of balls. This parameterization allows us to decouple problem size from geometric complexity of the solution and naturally avoids explicit time allocation by allowing trajectories to evolve continuously inside ellipsoidal corridors. Building on this representation, we formulate the Orthogonal Trust Region Problem (Orth-TRP), a specialized convex program with separable block constraints, and develop a solver that exploits this parallel structure and the unique structure of each parallel subproblem for efficient optimization. Experiments on a quadrotor trajectory planning benchmark show that our approach produces smoother trajectories and lower runtimes than state-of-the-art corridor based planners, especially in highly complicated environments. 

**Abstract (ZH)**: 在复杂环境中的碰撞自由轨迹规划仍然是机器人领域的核心挑战。现有的基于走廊的规划方法依赖于将自由空间分解为碰撞自由子集，随着环境复杂性的增加，这些方法的可扩展性较差，并且需要为轨迹段明确分配时间窗口。我们提出了一种新的轨迹参数化方法，将碰撞自由走廊表示为凸笛卡尔球体乘积。这种方法允许我们将问题规模与解决方案的几何复杂性解耦，并且自然地避免了为轨迹段明确分配时间窗口，因为轨迹可以在椭球走廊内连续演化。基于这种表示，我们提出了正交信赖域问题（Orth-TRP），这是一种具有可分块约束的特殊凸规划，并开发了一个利用这种并行结构和每个并行子问题的独特结构进行高效优化的求解器。实验研究表明，在四旋翼飞行器轨迹规划基准测试中，我们的方法比最先进的基于走廊的方法产生了更平滑的轨迹并具有更低的运行时间，尤其是在高度复杂的环境中。 

---
# Simultaneous estimation of contact position and tool shape with high-dimensional parameters using force measurements and particle filtering 

**Title (ZH)**: 基于力测量和粒子滤波的高维参数同时估计刀具位置和形状 

**Authors**: Kyo Kutsuzawa, Mitsuhiro Hayashibe  

**Link**: [PDF](https://arxiv.org/pdf/2509.19732)  

**Abstract**: Estimating the contact state between a grasped tool and the environment is essential for performing contact tasks such as assembly and object manipulation. Force signals are valuable for estimating the contact state, as they can be utilized even when the contact location is obscured by the tool. Previous studies proposed methods for estimating contact positions using force/torque signals; however, most methods require the geometry of the tool surface to be known. Although several studies have proposed methods that do not require the tool shape, these methods require considerable time for estimation or are limited to tools with low-dimensional shape parameters. Here, we propose a method for simultaneously estimating the contact position and tool shape, where the tool shape is represented by a grid, which is high-dimensional (more than 1000 dimensional). The proposed method uses a particle filter in which each particle has individual tool shape parameters, thereby to avoid directly handling a high-dimensional parameter space. The proposed method is evaluated through simulations and experiments using tools with curved shapes on a plane. Consequently, the proposed method can estimate the shape of the tool simultaneously with the contact positions, making the contact-position estimation more accurate. 

**Abstract (ZH)**: 估计被握住的工具与环境之间的接触状态对于执行装配和物体操作等接触任务至关重要。力信号对于估计接触状态很有价值，即使接触位置被工具遮挡时也能够利用。前人研究提出了基于力/力矩信号估计接触位置的方法；然而，大多数方法要求知道工具表面的几何形状。虽然有一些研究提出了不需要知道工具形状的方法，但这些方法需要较长的估计时间，或者只能应用于具有低维形状参数的工具。在这里，我们提出了一种同时估计接触位置和工具形状的方法，其中工具形状用网格表示，具有高维（超过1000维）的参数。所提方法使用了粒子滤波器，每个粒子具有独立的工具形状参数，从而避免直接处理高维参数空间。所提方法通过在平面上使用曲线形状的工具进行仿真和实验进行评估。结果表明，所提方法可以同时估计工具的形状和接触位置，从而提高接触位置估计的准确性。 

---
# Towards Autonomous Robotic Electrosurgery via Thermal Imaging 

**Title (ZH)**: 基于热成像的自主机器人电外科研究 

**Authors**: Naveed D. Riaziat, Joseph Chen, Axel Krieger, Jeremy D. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2509.19725)  

**Abstract**: Electrosurgery is a surgical technique that can improve tissue cutting by reducing cutting force and bleeding. However, electrosurgery adds a risk of thermal injury to surrounding tissue. Expert surgeons estimate desirable cutting velocities based on experience but have no quantifiable reference to indicate if a particular velocity is optimal. Furthermore, prior demonstrations of autonomous electrosurgery have primarily used constant tool velocity, which is not robust to changes in electrosurgical tissue characteristics, power settings, or tool type. Thermal imaging feedback provides information that can be used to reduce thermal injury while balancing cutting force by controlling tool velocity. We introduce Thermography for Electrosurgical Rate Modulation via Optimization (ThERMO) to autonomously reduce thermal injury while balancing cutting force by intelligently controlling tool velocity. We demonstrate ThERMO in tissue phantoms and compare its performance to the constant velocity approach. Overall, ThERMO improves cut success rate by a factor of three and can reduce peak cutting force by a factor of two. ThERMO responds to varying environmental disturbances, reduces damage to tissue, and completes cutting tasks that would otherwise result in catastrophic failure for the constant velocity approach. 

**Abstract (ZH)**: 基于热图像的电外科速率优化热保护方法（ThERMO） 

---
# TopoCut: Learning Multi-Step Cutting with Spectral Rewards and Discrete Diffusion Policies 

**Title (ZH)**: TopoCut: 学习多步切割的光谱奖励与离散扩散策略 

**Authors**: Liquan Wang, Jiangjie Bian, Eric Heiden, Animesh Garg  

**Link**: [PDF](https://arxiv.org/pdf/2509.19712)  

**Abstract**: Robotic manipulation tasks involving cutting deformable objects remain challenging due to complex topological behaviors, difficulties in perceiving dense object states, and the lack of efficient evaluation methods for cutting outcomes. In this paper, we introduce TopoCut, a comprehensive benchmark for multi-step robotic cutting tasks that integrates a cutting environment and generalized policy learning. TopoCut is built upon three core components: (1) We introduce a high-fidelity simulation environment based on a particle-based elastoplastic solver with compliant von Mises constitutive models, augmented by a novel damage-driven topology discovery mechanism that enables accurate tracking of multiple cutting pieces. (2) We develop a comprehensive reward design that integrates the topology discovery with a pose-invariant spectral reward model based on Laplace-Beltrami eigenanalysis, facilitating consistent and robust assessment of cutting quality. (3) We propose an integrated policy learning pipeline, where a dynamics-informed perception module predicts topological evolution and produces particle-wise, topology-aware embeddings to support PDDP (Particle-based Score-Entropy Discrete Diffusion Policy) for goal-conditioned policy learning. Extensive experiments demonstrate that TopoCut supports trajectory generation, scalable learning, precise evaluation, and strong generalization across diverse object geometries, scales, poses, and cutting goals. 

**Abstract (ZH)**: 涉及切削变形物体的机器人操作任务由于复杂的拓扑行为、密集物体状态感知的难度以及切割结果评估方法的缺乏而具有挑战性。本文介绍了TopoCut，一个集成切削环境和广义策略学习的多步骤机器人切削任务全面基准。TopoCut基于三个核心组件构建：（1）我们引入了一个基于颗粒基弹塑性求解器的高保真模拟环境，并结合了一种新颖的损伤驱动拓扑发现机制，能够准确跟踪多个切削件。（2）我们开发了全面的奖励设计，将拓扑发现与基于拉普拉斯-贝尔特拉米特征分析的姿势不变的频谱奖励模型结合起来，促进切割质量的一致性和稳健评估。（3）我们提出了一种集成策略学习流水线，其中动态信息感知模块预测拓扑演化并生成颗粒级别的、拓扑意识的嵌入，以支持基于PDDP（基于颗粒的评分-熵离散扩散策略）的目标条件策略学习。广泛实验表明，TopoCut 支持轨迹生成、可扩展学习、精确评估和在不同物体几何形状、尺度、姿态和切割目标下的强大泛化。 

---
# Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks 

**Title (ZH)**: 基于扩散的阻抗学习在接触丰富的操作任务中 

**Authors**: Noah Geiger, Tamim Asfour, Neville Hogan, Johannes Lachner  

**Link**: [PDF](https://arxiv.org/pdf/2509.19696)  

**Abstract**: Learning methods excel at motion generation in the information domain but are not primarily designed for physical interaction in the energy domain. Impedance Control shapes physical interaction but requires task-aware tuning by selecting feasible impedance parameters. We present Diffusion-Based Impedance Learning, a framework that combines both domains. A Transformer-based Diffusion Model with cross-attention to external wrenches reconstructs a simulated Zero-Force Trajectory (sZFT). This captures both translational and rotational task-space behavior. For rotations, we introduce a novel SLERP-based quaternion noise scheduler that ensures geometric consistency. The reconstructed sZFT is then passed to an energy-based estimator that updates stiffness and damping parameters. A directional rule is applied that reduces impedance along non task axes while preserving rigidity along task directions. Training data were collected for a parkour scenario and robotic-assisted therapy tasks using teleoperation with Apple Vision Pro. With only tens of thousands of samples, the model achieved sub-millimeter positional accuracy and sub-degree rotational accuracy. Its compact model size enabled real-time torque control and autonomous stiffness adaptation on a KUKA LBR iiwa robot. The controller achieved smooth parkour traversal within force and velocity limits and 30/30 success rates for cylindrical, square, and star peg insertions without any peg-specific demonstrations in the training data set. All code for the Transformer-based Diffusion Model, the robot controller, and the Apple Vision Pro telemanipulation framework is publicly available. These results mark an important step towards Physical AI, fusing model-based control for physical interaction with learning-based methods for trajectory generation. 

**Abstract (ZH)**: 基于扩散的阻抗学习：融合信息域的学习方法与能量域的物理交互 

---
# Minimalistic Autonomous Stack for High-Speed Time-Trial Racing 

**Title (ZH)**: 高速计时赛的极简自主堆栈 

**Authors**: Mahmoud Ali, Hassan Jardali, Youwei Yu, Durgakant Pushp, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19636)  

**Abstract**: Autonomous racing has seen significant advancements, driven by competitions such as the Indy Autonomous Challenge (IAC) and the Abu Dhabi Autonomous Racing League (A2RL). However, developing an autonomous racing stack for a full-scale car is often constrained by limited access to dedicated test tracks, restricting opportunities for real-world validation. While previous work typically requires extended development cycles and significant track time, this paper introduces a minimalistic autonomous racing stack for high-speed time-trial racing that emphasizes rapid deployment and efficient system integration with minimal on-track testing. The proposed stack was validated on real speedways, achieving a top speed of 206 km/h within just 11 hours' practice run on the track with 325 km in total. Additionally, we present the system performance analysis, including tracking accuracy, vehicle dynamics, and safety considerations, offering insights for teams seeking to rapidly develop and deploy an autonomous racing stack with limited track access. 

**Abstract (ZH)**: 自主赛车技术取得了显著进步，得益于如印第自主挑战赛（IAC）和阿布扎比自主赛车联盟（A2RL）等比赛的推动。然而，开发适用于全尺寸汽车的自主赛车系统通常受限于专用测试赛道的有限访问权限，限制了实地验证的机会。以往的工作通常需要较长的研发周期和大量的赛道时间，本论文介绍了一种 minimalist 自主赛车堆栈，强调快速部署和高效的系统集成，并最大限度减少赛道上的测试。所提出的堆栈已在实际赛车场上进行了验证，仅用11小时的练习运行便达到了206 km/h的最高速度，总计行驶325 km。此外，我们还呈现了系统的性能分析，包括跟踪精度、车辆动力学和安全性考虑，为希望在有限的赛道访问权限下快速开发和部署自主赛车堆栈的团队提供了见解。 

---
# RoMoCo: Robotic Motion Control Toolbox for Reduced-Order Model-Based Locomotion on Bipedal and Humanoid Robots 

**Title (ZH)**: RoMoCo: 用于 bipedal 和 humanoid 机器人基于降阶模型的运动控制工具箱 

**Authors**: Min Dai, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2509.19545)  

**Abstract**: We present RoMoCo, an open-source C++ toolbox for the synthesis and evaluation of reduced-order model-based planners and whole-body controllers for bipedal and humanoid robots. RoMoCo's modular architecture unifies state-of-the-art planners and whole-body locomotion controllers under a consistent API, enabling rapid prototyping and reproducible benchmarking. By leveraging reduced-order models for platform-agnostic gait generation, RoMoCo enables flexible controller design across diverse robots. We demonstrate its versatility and performance through extensive simulations on the Cassie, Unitree H1, and G1 robots, and validate its real-world efficacy with hardware experiments on the Cassie and G1 humanoids. 

**Abstract (ZH)**: RoMoCo：一种开源C++工具箱，用于 bipedal 和 humanoid 机器人基于降阶模型的规划器和全身控制器的设计与评估 

---
# Autonomous Elemental Characterization Enabled by a Low Cost Robotic Platform Built Upon a Generalized Software Architecture 

**Title (ZH)**: 基于通用软件架构的低成本机器人平台实现自主元素表征 

**Authors**: Xuan Cao, Yuxin Wu, Michael L. Whittaker  

**Link**: [PDF](https://arxiv.org/pdf/2509.19541)  

**Abstract**: Despite the rapidly growing applications of robots in industry, the use of robots to automate tasks in scientific laboratories is less prolific due to lack of generalized methodologies and high cost of hardware. This paper focuses on the automation of characterization tasks necessary for reducing cost while maintaining generalization, and proposes a software architecture for building robotic systems in scientific laboratory environment. A dual-layer (this http URL and ROS) action server design is the basic building block, which facilitates the implementation of a web-based front end for user-friendly operations and the use of ROS Behavior Tree for convenient task planning and execution. A robotic platform for automating mineral and material sample characterization is built upon the architecture, with an open source, low-cost three-axis computer numerical control gantry system serving as the main robot. A handheld laser induced breakdown spectroscopy (LIBS) analyzer is integrated with a 3D printed adapter, enabling automated 2D chemical mapping. We demonstrate the utility of automated chemical mapping by scanning of the surface of a spodumene-bearing pegmatite core sample with a 1071-point dense hyperspectral map acquired at a rate of 1520 bits per second. Automated LIBS scanning enables controlled chemical quantification in the laboratory that complements field-based measurements acquired with the same handheld device, linking resource exploration and processing steps in the supply chain for lithium-based battery materials. 

**Abstract (ZH)**: 尽管工业领域中机器人的应用快速增长，但由于缺乏通用方法和高昂的硬件成本，机器人在科学实验室中的自动化应用相对较少。本文专注于降低自动化成本同时保持通用性的表征任务自动化，并提出了一种适用于科学实验室环境的机器人系统软件架构。该架构采用两层（此链接和ROS）动作服务器设计，便于实现基于Web的前端操作界面以及使用ROS行为树进行便捷的任务规划与执行。在此架构基础上构建了一个用于矿物和材料样品表征的机器人平台，其中开源低成本的三轴计算机数控龙门架系统作为主要机器人。手持式激光诱导击穿光谱（LIBS）分析仪通过3D打印适配器集成，实现自动化的二维化学映射。通过以每秒1520位特率获取的包含1071个点的高光谱图对含锂辉石脉岩芯样品的表面进行扫描，展示了自动化化学映射的实用性。自动化LIBS扫描能够在实验室中实现可控的化学定量测量，补充手持设备在野外获得的测量结果，从而连接锂基电池材料资源勘探和处理步骤的供应链。 

---
# Real-Time Reinforcement Learning for Dynamic Tasks with a Parallel Soft Robot 

**Title (ZH)**: 并行软机器人中的实时强化学习动态任务 

**Authors**: James Avtges, Jake Ketchum, Millicent Schlafly, Helena Young, Taekyoung Kim, Allison Pinosky, Ryan L. Truby, Todd D. Murphey  

**Link**: [PDF](https://arxiv.org/pdf/2509.19525)  

**Abstract**: Closed-loop control remains an open challenge in soft robotics. The nonlinear responses of soft actuators under dynamic loading conditions limit the use of analytic models for soft robot control. Traditional methods of controlling soft robots underutilize their configuration spaces to avoid nonlinearity, hysteresis, large deformations, and the risk of actuator damage. Furthermore, episodic data-driven control approaches such as reinforcement learning (RL) are traditionally limited by sample efficiency and inconsistency across initializations. In this work, we demonstrate RL for reliably learning control policies for dynamic balancing tasks in real-time single-shot hardware deployments. We use a deformable Stewart platform constructed using parallel, 3D-printed soft actuators based on motorized handed shearing auxetic (HSA) structures. By introducing a curriculum learning approach based on expanding neighborhoods of a known equilibrium, we achieve reliable single-deployment balancing at arbitrary coordinates. In addition to benchmarking the performance of model-based and model-free methods, we demonstrate that in a single deployment, Maximum Diffusion RL is capable of learning dynamic balancing after half of the actuators are effectively disabled, by inducing buckling and by breaking actuators with bolt cutters. Training occurs with no prior data, in as fast as 15 minutes, with performance nearly identical to the fully-intact platform. Single-shot learning on hardware facilitates soft robotic systems reliably learning in the real world and will enable more diverse and capable soft robots. 

**Abstract (ZH)**: 软体机器人中的闭环控制仍然是一个开放挑战。软执行器在动态加载条件下的非线性响应限制了对软体机器人控制的分析模型的应用。传统方法在控制软体机器人时未充分利用其配置空间，以避免非线性、滞回现象、大变形以及执行器损坏的风险。此外，基于片断数据驱动的方法，如强化学习（RL），传统上由于样本效率低下和初始化一致性差而受到限制。在本文中，我们演示了在实时单片硬件部署中使用RL可靠地学习动力平衡任务的控制策略。我们使用基于电机驱动铰链膨胀（HSA）结构的并行3D打印软执行器构建的可变形Stewart平台。通过引入基于扩展已知平衡区邻域的 curriculum 学习方法，我们实现了在任意坐标处的可靠单次部署平衡。除了基准测试基于模型和非模型方法的性能之外，我们还展示了在单次部署中，最大扩散RL能够在半数执行器有效失效的情况下学习动力平衡，通过引发屈曲和使用扳手剪断执行器。培训无需任何先验数据，最快可在15分钟内完成，并且性能几乎与完整的平台相同。在硬件上的单次学习使得软体机器人系统能够在现实世界中可靠地学习，并将使软体机器人更加多样化和具备更强的能力。 

---
# Bioinspired SLAM Approach for Unmanned Surface Vehicle 

**Title (ZH)**: 生物启发的自主水面车辆SLAM方法 

**Authors**: Fabio Coelho, Joao Victor T. Borges, Paulo Padrao, Jose Fuentes, Ramon R. Costa, Liu Hsu, Leonardo Bobadilla  

**Link**: [PDF](https://arxiv.org/pdf/2509.19522)  

**Abstract**: This paper presents OpenRatSLAM2, a new version of OpenRatSLAM - a bioinspired SLAM framework based on computational models of the rodent hippocampus. OpenRatSLAM2 delivers low-computation-cost visual-inertial based SLAM, suitable for GPS-denied environments. Our contributions include a ROS2-based architecture, experimental results on new waterway datasets, and insights into system parameter tuning. This work represents the first known application of RatSLAM on USVs. The estimated trajectory was compared with ground truth data using the Hausdorff distance. The results show that the algorithm can generate a semimetric map with an error margin acceptable for most robotic applications. 

**Abstract (ZH)**: OpenRatSLAM2：一种基于啮齿动物海马体计算模型的生物启发式SLAM框架的新版本 

---
# A Bimanual Gesture Interface for ROS-Based Mobile Manipulators Using TinyML and Sensor Fusion 

**Title (ZH)**: 基于TinyML和传感器融合的ROSquila双手手势界面-Mobile操纵器 

**Authors**: Najeeb Ahmed Bhuiyan, M. Nasimul Huq, Sakib H. Chowdhury, Rahul Mangharam  

**Link**: [PDF](https://arxiv.org/pdf/2509.19521)  

**Abstract**: Gesture-based control for mobile manipulators faces persistent challenges in reliability, efficiency, and intuitiveness. This paper presents a dual-hand gesture interface that integrates TinyML, spectral analysis, and sensor fusion within a ROS framework to address these limitations. The system uses left-hand tilt and finger flexion, captured using accelerometer and flex sensors, for mobile base navigation, while right-hand IMU signals are processed through spectral analysis and classified by a lightweight neural network. This pipeline enables TinyML-based gesture recognition to control a 7-DOF Kinova Gen3 manipulator. By supporting simultaneous navigation and manipulation, the framework improves efficiency and coordination compared to sequential methods. Key contributions include a bimanual control architecture, real-time low-power gesture recognition, robust multimodal sensor fusion, and a scalable ROS-based implementation. The proposed approach advances Human-Robot Interaction (HRI) for industrial automation, assistive robotics, and hazardous environments, offering a cost-effective, open-source solution with strong potential for real-world deployment and further optimization. 

**Abstract (ZH)**: 基于手势的移动 manipulator 控制面临可靠性和直观性等方面的持续挑战。本文提出了一种集成 TinyML、谱分析和传感器融合的双臂手势界面，以解决这些限制。该系统利用左臂倾斜和手指弯曲（通过加速度计和弯曲传感器捕获）进行移动基座导航，而右臂惯性测量单元（IMU）信号通过谱分析处理并由轻量级神经网络分类，从而实现基于 TinyML 的手势识别控制 7 自由度 Kinova Gen3 手臂。该框架通过同时支持导航和操作，提高了效率和协调性，相比于顺序方法。主要贡献包括双臂控制架构、实时低功耗手势识别、稳健的多模态传感器融合以及基于 ROS 的可扩展实现。所提出的方法推动了工业自动化、辅助机器人和危险环境中的人机交互（HRI），提供了一种成本效益高、开源的解决方案，在实际部署和进一步优化方面具有很强的潜力。 

---
# Supercomputing for High-speed Avoidance and Reactive Planning in Robots 

**Title (ZH)**: 超算在机器人高速避障与反应规划中的应用 

**Authors**: Kieran S. Lachmansingh, José R. González-Estrada, Ryan E. Grant, Matthew K. X. J. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.19486)  

**Abstract**: This paper presents SHARP (Supercomputing for High-speed Avoidance and Reactive Planning), a proof-of-concept study demonstrating how high-performance computing (HPC) can enable millisecond-scale responsiveness in robotic control. While modern robots face increasing demands for reactivity in human--robot shared workspaces, onboard processors are constrained by size, power, and cost. Offloading to HPC offers massive parallelism for trajectory planning, but its feasibility for real-time robotics remains uncertain due to network latency and jitter. We evaluate SHARP in a stress-test scenario where a 7-DOF manipulator must dodge high-speed foam projectiles. Using a parallelized multi-goal A* search implemented with MPI on both local and remote HPC clusters, the system achieves mean planning latencies of 22.9 ms (local) and 30.0 ms (remote, ~300 km away), with avoidance success rates of 84% and 88%, respectively. These results show that when round-trip latency remains within the tens-of-milliseconds regime, HPC-side computation is no longer the bottleneck, enabling avoidance well below human reaction times. The SHARP results motivate hybrid control architectures: low-level reflexes remain onboard for safety, while bursty, high-throughput planning tasks are offloaded to HPC for scalability. By reporting per-stage timing and success rates, this study provides a reproducible template for assessing real-time feasibility of HPC-driven robotics. Collectively, SHARP reframes HPC offloading as a viable pathway toward dependable, reactive robots in dynamic environments. 

**Abstract (ZH)**: 基于超级计算的高速避障与反应规划（SHARP）原理研究 

---
# Crater Observing Bio-inspired Rolling Articulator (COBRA) 

**Title (ZH)**: 基于生物启发滚动关节的撞击观测装置（COBRA） 

**Authors**: Adarsh Salagame, Henry Noyes, Alireza Ramezani, Eric Sihite, Arash Kalantari  

**Link**: [PDF](https://arxiv.org/pdf/2509.19473)  

**Abstract**: NASA aims to establish a sustainable human basecamp on the Moon as a stepping stone for future missions to Mars and beyond. The discovery of water ice on the Moon's craters located in permanently shadowed regions, which can provide drinking water, oxygen, and rocket fuel, is therefore of critical importance. However, current methods to access lunar ice deposits are limited. While rovers have been used to explore the lunar surface for decades, they face significant challenges in navigating harsh terrains, such as permanently shadowed craters, due to the high risk of immobilization. This report introduces COBRA (Crater Observing Bio-inspired Rolling Articulator), a multi-modal snake-style robot designed to overcome mobility challenges in Shackleton Crater's rugged environment. COBRA combines slithering and tumbling locomotion to adapt to various crater terrains. In snake mode, it uses sidewinding to traverse flat or low inclined surfaces, while in tumbling mode, it forms a circular barrel by linking its head and tail, enabling rapid movement with minimal energy on steep slopes. Equipped with an onboard computer, stereo camera, inertial measurement unit, and joint encoders, COBRA facilitates real-time data collection and autonomous operation. This paper highlights COBRAs robustness and efficiency in navigating extreme terrains through both simulations and experimental validation. 

**Abstract (ZH)**: NASA旨在建立一个可持续的人类基地，作为未来火星及其他更远深空任务的跳板。月球永久阴影区域坑洞中发现的水冰对于提供饮用水、氧气和火箭燃料至关重要，因此具有关键意义。然而，当前获取月球冰资源的方法有限。尽管探测车已用于数十年的月表探索，但它们在恶劣地形，如永久阴影坑洞中面临显著的移动挑战，因其被卡住的风险很高。本报告介绍了COBRA（Crater Observing Bio-inspired Rolling Articulator），一种多模式蛇形机器人，旨在克服谢克尔顿坑崎岖环境中的移动难题。COBRA结合了滑行和滚动运动，以适应各种坑洞地形。在蛇形模式下，它使用侧向迂回穿越平坦或低倾斜表面；在滚动模式下，它通过连接其头部和尾部形成一个圆筒状结构，从而在陡峭坡面实现快速、低能耗移动。配备机载计算机、立体摄像头、惯性测量单元和关节编码器，COBRA促进了实时数据收集和自主操作。本文通过模拟和实验验证突显了COBRA在极端地形中高效、稳健的导航能力。 

---
# HUNT: High-Speed UAV Navigation and Tracking in Unstructured Environments via Instantaneous Relative Frames 

**Title (ZH)**: HUNT: 高速无人机在无结构环境中基于瞬时相对坐标帧的导航与跟踪 

**Authors**: Alessandro Saviolo, Jeffrey Mao, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2509.19452)  

**Abstract**: Search and rescue operations require unmanned aerial vehicles to both traverse unknown unstructured environments at high speed and track targets once detected. Achieving both capabilities under degraded sensing and without global localization remains an open challenge. Recent works on relative navigation have shown robust tracking by anchoring planning and control to a visible detected object, but cannot address navigation when no target is in the field of view. We present HUNT (High-speed UAV Navigation and Tracking), a real-time framework that unifies traversal, acquisition, and tracking within a single relative formulation. HUNT defines navigation objectives directly from onboard instantaneous observables such as attitude, altitude, and velocity, enabling reactive high-speed flight during search. Once a target is detected, the same perception-control pipeline transitions seamlessly to tracking. Outdoor experiments in dense forests, container compounds, and search-and-rescue operations with vehicles and mannequins demonstrate robust autonomy where global methods fail. 

**Abstract (ZH)**: 高速无人 aerial 航空器导航与跟踪 

---
# Robust Near-Optimal Nonlinear Target Enclosing Guidance 

**Title (ZH)**: 鲁棒近最优非线性目标包络制导 

**Authors**: Abhinav Sinha, Rohit V. Nanavati  

**Link**: [PDF](https://arxiv.org/pdf/2509.19477)  

**Abstract**: This paper proposes a nonlinear optimal guidance law that enables a pursuer to enclose a target within arbitrary geometric patterns, which extends beyond conventional circular encirclement. The design operates using only relative state measurements and formulates a target enclosing guidance law in which the vehicle's lateral acceleration serves as the steering control, making it well-suited for aerial vehicles with turning constraints. Our approach generalizes and extends existing guidance strategies that are limited to target encirclement and provides a degree of optimality. At the same time, the exact information of the target's maneuver is unnecessary during the design. The guidance law is developed within the framework of a state-dependent Riccati equation (SDRE), thereby providing a systematic way to handle nonlinear dynamics through a pseudo-linear representation to design locally optimal feedback guidance commands through state-dependent weighting matrices. While SDRE ensures near-optimal performance in the absence of strong disturbances, we further augment the design to incorporate an integral sliding mode manifold to compensate when disturbances push the system away from the nominal trajectory, and demonstrate that the design provides flexibility in the sense that the (possibly time-varying) stand-off curvature could also be treated as unknown. Simulations demonstrate the efficacy of the proposed approach. 

**Abstract (ZH)**: 这篇论文提出了一种非线性最优引导律，使追踪器能够将目标包围在任意几何图案内，超越了传统的圆形包围。该设计仅使用相对状态测量值，并将目标包围引导律公式化，其中车辆的横向加速度用作转向控制，使其适用于具有转弯约束的航空器。我们的方法既泛化了现有的仅限于目标包围的引导策略，又提供了一定程度的最优性。同时，在设计过程中无需知道目标机动的精确信息。引导律在状态依赖型Riccati方程（SDRE）框架内开发，通过状态依赖型加权矩阵提供了一种通过伪线性表示处理非线性动力学的系统方法，以设计局部最优反馈引导指令。虽然SDRE在无强干扰的情况下保证了接近最优的性能，我们进一步通过引入积分滑模流形来补充设计，以补偿当干扰将系统推向非理想轨迹时的情况，并证明设计在某种程度上具有灵活性，即可能随时间变化的临界曲率也可以被视为未知数。仿真展示了所提方法的有效性。 

---
# ROPA: Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation 

**Title (ZH)**: ROPA：RGB-D 双手数据增强中的合成机器人姿态生成 

**Authors**: Jason Chen, I-Chun Arthur Liu, Gaurav Sukhatme, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2509.19454)  

**Abstract**: Training robust bimanual manipulation policies via imitation learning requires demonstration data with broad coverage over robot poses, contacts, and scene contexts. However, collecting diverse and precise real-world demonstrations is costly and time-consuming, which hinders scalability. Prior works have addressed this with data augmentation, typically for either eye-in-hand (wrist camera) setups with RGB inputs or for generating novel images without paired actions, leaving augmentation for eye-to-hand (third-person) RGB-D training with new action labels less explored. In this paper, we propose Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation (ROPA), an offline imitation learning data augmentation method that fine-tunes Stable Diffusion to synthesize third-person RGB and RGB-D observations of novel robot poses. Our approach simultaneously generates corresponding joint-space action labels while employing constrained optimization to enforce physical consistency through appropriate gripper-to-object contact constraints in bimanual scenarios. We evaluate our method on 5 simulated and 3 real-world tasks. Our results across 2625 simulation trials and 300 real-world trials demonstrate that ROPA outperforms baselines and ablations, showing its potential for scalable RGB and RGB-D data augmentation in eye-to-hand bimanual manipulation. Our project website is available at: this https URL. 

**Abstract (ZH)**: 通过模仿学习训练鲁棒的双臂 manipulation 策略需要广泛覆盖机器人姿态、接触点和场景上下文的演示数据。然而，收集多样且精确的现实世界演示数据成本高且耗时，这限制了其可扩展性。以往工作通过数据增强解决这一问题，通常针对带有 RGB 输入的眼手（腕部相机）设置或生成没有配对动作的新图像，而针对眼至手（第三人称）RGB-D 训练数据增强结合新动作标签的研究较少。本文提出了一种名为 Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation (ROPA) 的离线模仿学习数据增强方法，通过微调 Stable Diffusion 合成第三人称 RGB 和 RGB-D 观测值的新机器人姿态。我们的方法同时生成相应的关节空间动作标签，并通过适当的双臂场景中的夹具-物体接触约束来实现物理一致性约束，以确保一致性。我们在 5 个仿真任务和 3 个现实世界任务上评估了该方法。我们的结果表明，ROPA 在 2625 次仿真试验和 300 次现实世界试验中优于基线和消融方法，展示了其在眼至手双臂 manipulation 中实现可扩展的 RGB 和 RGB-D 数据增强的潜力。项目网站: https://github.com/yourusername/ROPA。 

---
