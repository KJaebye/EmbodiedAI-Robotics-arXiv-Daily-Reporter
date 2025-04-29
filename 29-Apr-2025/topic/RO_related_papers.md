# Kinodynamic Trajectory Following with STELA: Simultaneous Trajectory Estimation & Local Adaptation 

**Title (ZH)**: 基于STELA的 kino-dynamic 轨迹跟随：轨迹估计与局部适应的同时进行 

**Authors**: Edgar Granados, Sumanth Tangirala, Kostas E. Bekris  

**Link**: [PDF](https://arxiv.org/pdf/2504.20009)  

**Abstract**: State estimation and control are often addressed separately, leading to unsafe execution due to sensing noise, execution errors, and discrepancies between the planning model and reality. Simultaneous control and trajectory estimation using probabilistic graphical models has been proposed as a unified solution to these challenges. Previous work, however, relies heavily on appropriate Gaussian priors and is limited to holonomic robots with linear time-varying models. The current research extends graphical optimization methods to vehicles with arbitrary dynamical models via Simultaneous Trajectory Estimation and Local Adaptation (STELA). The overall approach initializes feasible trajectories using a kinodynamic, sampling-based motion planner. Then, it simultaneously: (i) estimates the past trajectory based on noisy observations, and (ii) adapts the controls to be executed to minimize deviations from the planned, feasible trajectory, while avoiding collisions. The proposed factor graph representation of trajectories in STELA can be applied for any dynamical system given access to first or second-order state update equations, and introduces the duration of execution between two states in the trajectory discretization as an optimization variable. These features provide both generalization and flexibility in trajectory following. In addition to targeting computational efficiency, the proposed strategy performs incremental updates of the factor graph using the iSAM algorithm and introduces a time-window mechanism. This mechanism allows the factor graph to be dynamically updated to operate over a limited history and forward horizon of the planned trajectory. This enables online updates of controls at a minimum of 10Hz. Experiments demonstrate that STELA achieves at least comparable performance to previous frameworks on idealized vehicles with linear dynamics.[...] 

**Abstract (ZH)**: 同时轨迹估计与局部适应的控制与轨迹估计方法 

---
# Socially-Aware Autonomous Driving: Inferring Yielding Intentions for Safer Interactions 

**Title (ZH)**: 社交意识自主驾驶：推断让行意图以实现更安全的交互 

**Authors**: Jing Wang, Yan Jin, Hamid Taghavifar, Fei Ding, Chongfeng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2504.20004)  

**Abstract**: Since the emergence of autonomous driving technology, it has advanced rapidly over the past decade. It is becoming increasingly likely that autonomous vehicles (AVs) would soon coexist with human-driven vehicles (HVs) on the roads. Currently, safety and reliable decision-making remain significant challenges, particularly when AVs are navigating lane changes and interacting with surrounding HVs. Therefore, precise estimation of the intentions of surrounding HVs can assist AVs in making more reliable and safe lane change decision-making. This involves not only understanding their current behaviors but also predicting their future motions without any direct communication. However, distinguishing between the passing and yielding intentions of surrounding HVs still remains ambiguous. To address the challenge, we propose a social intention estimation algorithm rooted in Directed Acyclic Graph (DAG), coupled with a decision-making framework employing Deep Reinforcement Learning (DRL) algorithms. To evaluate the method's performance, the proposed framework can be tested and applied in a lane-changing scenario within a simulated environment. Furthermore, the experiment results demonstrate how our approach enhances the ability of AVs to navigate lane changes safely and efficiently on roads. 

**Abstract (ZH)**: 自从自主驾驶技术的出现，过去十年间该技术已快速发展。随着自主车辆（AVs）在未来不久可能与人工驾驶车辆（HVs）共同行驶在道路上，当前安全性和可靠的决策仍然是重大挑战，尤其是在AVs进行车道变换并与周围HV互动时。因此，精确估计周围HV的意图可以帮助AVs做出更可靠和安全的车道变换决策。这不仅涉及理解其当前行为，还涉及在没有直接通信的情况下预测其未来运动。然而，区分周围HV的通过意图和礼让意图仍然存在模糊性。为应对这一挑战，我们提出了一种基于有向无环图（DAG）的社会意图估计算法，并结合了一种采用深度强化学习（DRL）算法的决策框架。通过在模拟环境中测试和应用该框架，在车道变换场景中评估该方法的性能。实验结果进一步证明了我们方法如何增强AVs在道路上安全高效地进行车道变换的能力。 

---
# Feelbert: A Feedback Linearization-based Embedded Real-Time Quadrupedal Locomotion Framework 

**Title (ZH)**: Feelbert: 一种基于反馈线性化的嵌入式实时四足行走框架 

**Authors**: Aristide Emanuele Casucci, Federico Nesti, Mauro Marinoni, Giorgio Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2504.19965)  

**Abstract**: Quadruped robots have become quite popular for their ability to adapt their locomotion to generic uneven terrains. For this reason, over time, several frameworks for quadrupedal locomotion have been proposed, but with little attention to ensuring a predictable timing behavior of the controller.
To address this issue, this work presents \NAME, a modular control framework for quadrupedal locomotion suitable for execution on an embedded system under hard real-time execution constraints. It leverages the feedback linearization control technique to obtain a closed-form control law for the body, valid for all configurations of the robot. The control law was derived after defining an appropriate rigid body model that uses the accelerations of the feet as control variables, instead of the estimated contact forces. This work also provides a novel algorithm to compute footholds and gait temporal parameters using the concept of imaginary wheels, and a heuristic algorithm to select the best gait schedule for the current velocity commands.
The proposed framework is developed entirely in C++, with no dependencies on third-party libraries and no dynamic memory allocation, to ensure predictability and real-time performance. Its implementation allows \NAME\ to be both compiled and executed on an embedded system for critical applications, as well as integrated into larger systems such as Robot Operating System 2 (ROS 2). For this reason, \NAME\ has been tested in both scenarios, demonstrating satisfactory results both in terms of reference tracking and temporal predictability, whether integrated into ROS 2 or compiled as a standalone application on a Raspberry Pi 5. 

**Abstract (ZH)**: 四足机器人因其能够适应通用不平地形的运动能力而变得非常流行。因此，随着时间的推移，已经提出了多种四足运动框架，但缺乏对控制器可预测的时间行为的保证。
为了解决这个问题，本工作提出了一种名为\NAME的模块化控制框架，适用于具有严格实时执行约束的嵌入式系统。该框架利用反馈线性化控制技术，获得适用于机器人所有配置的有效闭环控制律。控制律在定义了适当的刚体模型之后得出，该模型使用脚的加速度作为控制变量，而不是估计的接触力。此外，本工作还提供了一种使用想象滚轮的概念来计算 footholds 和步态时间参数的新算法，并提供了一种启发式算法来为当前速度命令选择最佳步态调度。
本框架完全用 C++ 编写，不依赖第三方库，且不进行动态内存分配，以确保可预测性和实时性能。其实现允许 \NAME 在关键应用中编译并执行在嵌入式系统上，同时也能集成到更大的系统中，如 Robot Operating System 2（ROS 2）。因此，\NAME 在这两种场景下都进行了测试，无论是在 ROS 2 中集成还是在 Raspberry Pi 5 上作为独立应用程序进行编译，都能在参考跟踪和时间可预测性方面展示出令人满意的结果。 

---
# Tendon-Actuated Concentric Tube Endonasal Robot (TACTER) 

**Title (ZH)**: 经鼻 concentric 管 tendon 驱动机器人 (TACTER) 

**Authors**: Kent K. Yamamoto, Tanner J. Zachem, Pejman Kheradmand, Patrick Zheng, Jihad Abdelgadir, Jared Laurance Bailey, Kaelyn Pieter, Patrick J. Codd, Yash Chitalia  

**Link**: [PDF](https://arxiv.org/pdf/2504.19948)  

**Abstract**: Endoscopic endonasal approaches (EEA) have become more prevalent for minimally invasive skull base and sinus surgeries. However, rigid scopes and tools significantly decrease the surgeon's ability to operate in tight anatomical spaces and avoid critical structures such as the internal carotid artery and cranial nerves. This paper proposes a novel tendon-actuated concentric tube endonasal robot (TACTER) design in which two tendon-actuated robots are concentric to each other, resulting in an outer and inner robot that can bend independently. The outer robot is a unidirectionally asymmetric notch (UAN) nickel-titanium robot, and the inner robot is a 3D-printed bidirectional robot, with a nickel-titanium bending member. In addition, the inner robot can translate axially within the outer robot, allowing the tool to traverse through structures while bending, thereby executing follow-the-leader motion. A Cosserat-rod based mechanical model is proposed that uses tendon tension of both tendon-actuated robots and the relative translation between the robots as inputs and predicts the TACTER tip position for varying input parameters. The model is validated with experiments, and a human cadaver experiment is presented to demonstrate maneuverability from the nostril to the sphenoid sinus. This work presents the first tendon-actuated concentric tube (TACT) dexterous robotic tool capable of performing follow-the-leader motion within natural nasal orifices to cover workspaces typically required for a successful EEA. 

**Abstract (ZH)**: 经鼻内窥镜 approaches (EEA)在颅底和鼻窦手术中的微创手术中逐渐普及。然而，刚性内窥镜和工具显著降低了外科医生在狭窄解剖空间操作的能力，并且难以避开关键结构如颈内动脉和颅神经。本文提出了一种新型的腱驱 concentric 管内窥镜机器人 (TACTER) 设计，该设计中两个腱驱机器人同心布置，形成可独立弯曲的外机器人和内机器人。外机器人是一个单向不对称槽 (UAN) 镍钛合金机器人，内机器人是一个三维打印的双向机器人，装备有镍钛合金弯曲部件。此外，内机器人可以在外机器人中进行轴向移动，使得器械在弯曲过程中能够穿越结构，执行跟随运动。基于 Cosserat 杆的机械模型被提出，该模型使用两个 tendon-actuated 机器人以及两个机器人之间的相对平移作为输入，预测 TACTER 作用点的位置。实验验证了该模型，并通过人类尸体实验展示了从鼻孔到蝶窦的操作灵活性。本文介绍了首款能够在自然鼻孔内执行跟随运动的腱驱 concentric 管内窥镜机器人，以覆盖EEA通常所需的工作空间。 

---
# On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops 

**Title (ZH)**: 关于解决具有 Kinematic 循环的约束刚体多体系统动力学问题的研究 

**Authors**: Vassilios Tsounis, Ruben Grandia, Moritz Bächer  

**Link**: [PDF](https://arxiv.org/pdf/2504.19771)  

**Abstract**: This technical report provides an in-depth evaluation of both established and state-of-the-art methods for simulating constrained rigid multi-body systems with hard-contact dynamics, using formulations of Nonlinear Complementarity Problems (NCPs). We are particularly interest in examining the simulation of highly coupled mechanical systems with multitudes of closed-loop bilateral kinematic joint constraints in the presence of additional unilateral constraints such as joint limits and frictional contacts with restitutive impacts. This work thus presents an up-to-date literature survey of the relevant fields, as well as an in-depth description of the approaches used for the formulation and solving of the numerical time-integration problem in a maximal coordinate setting. More specifically, our focus lies on a version of the overall problem that decomposes it into the forward dynamics problem followed by a time-integration using the states of the bodies and the constraint reactions rendered by the former. We then proceed to elaborate on the formulations used to model frictional contact dynamics and define a set of solvers that are representative of those currently employed in the majority of the established physics engines. A key aspect of this work is the definition of a benchmarking framework that we propose as a means to both qualitatively and quantitatively evaluate the performance envelopes of the set of solvers on a diverse set of challenging simulation scenarios. We thus present an extensive set of experiments that aim at highlighting the absolute and relative performance of all solvers on particular problems of interest as well as aggravatingly over the complete set defined in the suite. 

**Abstract (ZH)**: 本技术报告对使用非线性互补问题（NCPs）表述的约束刚体多体系统硬接触动力学模拟方法进行了深入评估，包括传统方法和当前最先进的方法。我们特别关注含有大量闭环双边运动学关节约束以及额外的单向约束（如关节限位和具有恢复性碰撞的摩擦接触）的强耦合机械系统的模拟。本工作因此提供了一个相关领域的最新文献综述，并对最大坐标设置下数值时间积分问题的建模和求解方法进行了深入描述。特别是，我们关注的是将整体问题分解为前向动力学问题，随后使用先前得到的身体状态和约束反作用力进行时间积分的方法。报告进一步详细描述了用于建模摩擦接触动力学的公式，并定义了一套代表当前广泛使用的物理引擎中所采用的各种求解器的集合。本研究的一个关键方面是定义了一个基准测试框架，作为评估所研究求解器性能边界的手段，无论是定性还是定量评价。因此，报告呈现了一系列全面的实验，旨在突出所有求解器在特定问题上的绝对和相对性能，并在整个测试套件中进行放大。 

---
# UTTG_ A Universal Teleoperation Approach via Online Trajectory Generation 

**Title (ZH)**: UTTG_基于在线轨迹生成的通用远程操作方法 

**Authors**: Shengjian Fang, Yixuan Zhou, Yu Zheng, Pengyu Jiang, Siyuan Liu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19736)  

**Abstract**: Teleoperation is crucial for hazardous environment operations and serves as a key tool for collecting expert demonstrations in robot learning. However, existing methods face robotic hardware dependency and control frequency mismatches between teleoperation devices and robotic platforms. Our approach automatically extracts kinematic parameters from unified robot description format (URDF) files, and enables pluggable deployment across diverse robots through uniform interfaces. The proposed interpolation algorithm bridges the frequency gap between low-rate human inputs and high-frequency robotic control commands through online continuous trajectory generation, \n{while requiring no access to the closed, bottom-level control loop}. To enhance trajectory smoothness, we introduce a minimum-stretch spline that optimizes the motion quality. The system further provides precision and rapid modes to accommodate different task requirements. Experiments across various robotic platforms including dual-arm ones demonstrate generality and smooth operation performance of our methods. The code is developed in C++ with python interface, and available at this https URL. 

**Abstract (ZH)**: 遥操作对于危险环境作业至关重要，并且是机器人学习中收集专家演示的关键工具。然而，现有方法存在对机器人硬件的依赖以及遥操作设备与机器人平台之间控制频率不匹配的问题。我们的方法自动从统一机器人描述格式（URDF）文件中提取运动参数，并通过统一接口实现跨不同机器人的插件化部署。提出的插值算法通过在线连续轨迹生成，解决了低频人类输入与高频机器人控制命令之间的频率差距问题，且不需要访问关闭的底层控制回路。为了提升轨迹平滑性，我们引入了一种最小拉伸样条线来优化运动质量。系统还提供了精确模式和快速模式以适应不同的任务需求。来自各种机器人平台的实验证明了我们方法的普适性和平滑操作性能。代码使用C++编写，并通过Python接口提供，可在以下链接获取。 

---
# Hector UI: A Flexible Human-Robot User Interface for (Semi-)Autonomous Rescue and Inspection Robots 

**Title (ZH)**: Hector UI: 一种灵活的半自主救援与检测机器人用户界面 

**Authors**: Stefan Fabian, Oskar von Stryk  

**Link**: [PDF](https://arxiv.org/pdf/2504.19728)  

**Abstract**: The remote human operator's user interface (UI) is an important link to make the robot an efficient extension of the operator's perception and action. In rescue applications, several studies have investigated the design of operator interfaces based on observations during major robotics competitions or field deployments. Based on this research, guidelines for good interface design were empirically identified. The investigations on the UIs of teams participating in competitions are often based on external observations during UI application, which may miss some relevant requirements for UI flexibility. In this work, we present an open-source and flexibly configurable user interface based on established guidelines and its exemplary use for wheeled, tracked, and walking robots. We explain the design decisions and cover the insights we have gained during its highly successful applications in multiple robotics competitions and evaluations. The presented UI can also be adapted for other robots with little effort and is available as open source. 

**Abstract (ZH)**: 远程人类操作者的用户界面是使机器人成为操作者感知和行动高效延伸的重要环节。在救援应用中，多项研究基于重要机器人竞赛或现场部署期间的观察，调查了操作员界面的设计。基于这些研究，良好的界面设计原则被实证识别出来。对于参与竞赛团队的UI研究通常基于UI应用期间的外部观察，可能会遗漏一些UI灵活性的相关要求。在本工作中，我们提出了一种基于已确立原则的开源且可灵活配置的用户界面，并展示了其在履带式、轨道式和步行机器人上的典型应用。我们解释了设计决策，并涵盖了在其在多个机器人竞赛和评估中的广泛应用中获得的见解。所展示的UI也可以轻松适应其他机器人，并作为开源软件提供。 

---
# QuickGrasp: Lightweight Antipodal Grasp Planning with Point Clouds 

**Title (ZH)**: QuickGrasp: 基于点云的轻量级反平行夹持规划 

**Authors**: Navin Sriram Ravie, Keerthi Vasan M, Asokan Thondiyath, Bijo Sebastian  

**Link**: [PDF](https://arxiv.org/pdf/2504.19716)  

**Abstract**: Grasping has been a long-standing challenge in facilitating the final interface between a robot and the environment. As environments and tasks become complicated, the need to embed higher intelligence to infer from the surroundings and act on them has become necessary. Although most methods utilize techniques to estimate grasp pose by treating the problem via pure sampling-based approaches in the six-degree-of-freedom space or as a learning problem, they usually fail in real-life settings owing to poor generalization across domains. In addition, the time taken to generate the grasp plan and the lack of repeatability, owing to sampling inefficiency and the probabilistic nature of existing grasp planning approaches, severely limits their application in real-world tasks. This paper presents a lightweight analytical approach towards robotic grasp planning, particularly antipodal grasps, with little to no sampling in the six-degree-of-freedom space. The proposed grasp planning algorithm is formulated as an optimization problem towards estimating grasp points on the object surface instead of directly estimating the end-effector pose. To this extent, a soft-region-growing algorithm is presented for effective plane segmentation, even in the case of curved surfaces. An optimization-based quality metric is then used for the evaluation of grasp points to ensure indirect force closure. The proposed grasp framework is compared with the existing state-of-the-art grasp planning approach, Grasp pose detection (GPD), as a baseline over multiple simulated objects. The effectiveness of the proposed approach in comparison to GPD is also evaluated in a real-world setting using image and point-cloud data, with the planned grasps being executed using a ROBOTIQ gripper and UR5 manipulator. 

**Abstract (ZH)**: 基于最少六自由度空间采样的轻量级抗反 grasp 规划方法 

---
# Tensegrity-based Robot Leg Design with Variable Stiffness 

**Title (ZH)**: 基于张拉整体原理的 VARIABLE STIFFNESS 机器人腿设计 

**Authors**: Erik Mortensen, Jan Petrs, Alexander Dittrich, Dario Floreano  

**Link**: [PDF](https://arxiv.org/pdf/2504.19685)  

**Abstract**: Animals can finely modulate their leg stiffness to interact with complex terrains and absorb sudden shocks. In feats like leaping and sprinting, animals demonstrate a sophisticated interplay of opposing muscle pairs that actively modulate joint stiffness, while tendons and ligaments act as biological springs storing and releasing energy. Although legged robots have achieved notable progress in robust locomotion, they still lack the refined adaptability inherent in animal motor control. Integrating mechanisms that allow active control of leg stiffness presents a pathway towards more resilient robotic systems. This paper proposes a novel mechanical design to integrate compliancy into robot legs based on tensegrity - a structural principle that combines flexible cables and rigid elements to balance tension and compression. Tensegrity structures naturally allow for passive compliance, making them well-suited for absorbing impacts and adapting to diverse terrains. Our design features a robot leg with tensegrity joints and a mechanism to control the joint's rotational stiffness by modulating the tension of the cable actuation system. We demonstrate that the robot leg can reduce the impact forces of sudden shocks by at least 34.7 % and achieve a similar leg flexion under a load difference of 10.26 N by adjusting its stiffness configuration. The results indicate that tensegrity-based leg designs harbors potential towards more resilient and adaptable legged robots. 

**Abstract (ZH)**: 动物能够精细调节腿部刚度以应对复杂的地形并吸收突然的冲击。在跳跃和短跑等动作中，动物展示了对立肌肉群的复杂互动，主动调节关节刚度，而肌腱和韧带则作为生物弹簧储存和释放能量。尽管-legged robots在稳健移动方面取得了显著进展，但仍缺乏动物运动控制中固有的精细适应性。通过集成允许主动控制腿部刚度的机制，可以朝着更具韧性的机器人系统发展。本文提出了一种基于张拉整体原理的新型机械设计，将顺应性整合到机器人腿部中——张拉整体原理将柔性缆线和刚性元件结合在一起，以平衡拉力和压力。张拉整体结构天然地允许被动顺应性，使其适用于吸收冲击和适应各种地形。该设计包括一个具有张拉整体关节的机器人腿和一个通过调节缆线驱动系统张力来控制关节旋转刚度的机制。实验结果表明，该机器人腿可以通过调整刚度配置将突然冲击的冲击力降低至少34.7%，并在负载差为10.26 N的情况下实现相似的腿部弯曲。研究结果表明，基于张拉整体原理的腿部设计具有朝着更具韧性和适应性的腿式机器人发展的潜力。 

---
# Transformation & Translation Occupancy Grid Mapping: 2-Dimensional Deep Learning Refined SLAM 

**Title (ZH)**: 基于转换与翻译 occupancy 网格映射的二维深度学习增强SLAM 

**Authors**: Leon Davies, Baihua Li, Mohamad Saada, Simon Sølvsten, Qinggang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.19654)  

**Abstract**: SLAM (Simultaneous Localisation and Mapping) is a crucial component for robotic systems, providing a map of an environment, the current location and previous trajectory of a robot. While 3D LiDAR SLAM has received notable improvements in recent years, 2D SLAM lags behind. Gradual drifts in odometry and pose estimation inaccuracies hinder modern 2D LiDAR-odometry algorithms in large complex environments. Dynamic robotic motion coupled with inherent estimation based SLAM processes introduce noise and errors, degrading map quality. Occupancy Grid Mapping (OGM) produces results that are often noisy and unclear. This is due to the fact that evidence based mapping represents maps according to uncertain observations. This is why OGMs are so popular in exploration or navigation tasks. However, this also limits OGMs' effectiveness for specific mapping based tasks such as floor plan creation in complex scenes. To address this, we propose our novel Transformation and Translation Occupancy Grid Mapping (TT-OGM). We adapt and enable accurate and robust pose estimation techniques from 3D SLAM to the world of 2D and mitigate errors to improve map quality using Generative Adversarial Networks (GANs). We introduce a novel data generation method via deep reinforcement learning (DRL) to build datasets large enough for training a GAN for SLAM error correction. We demonstrate our SLAM in real-time on data collected at Loughborough University. We also prove its generalisability on a variety of large complex environments on a collection of large scale well-known 2D occupancy maps. Our novel approach enables the creation of high quality OGMs in complex scenes, far surpassing the capabilities of current SLAM algorithms in terms of quality, accuracy and reliability. 

**Abstract (ZH)**: 基于转换与平移的 occupancy 栅格地图 (TT-OGM): 结合3D SLAM的精准_pose估计与生成对抗网络的SLAM误差校正 

---
# Robot Motion Planning using One-Step Diffusion with Noise-Optimized Approximate Motions 

**Title (ZH)**: 基于噪声优化近似运动的一步扩散的机器人运动规划 

**Authors**: Tomoharu Aizu, Takeru Oba, Yuki Kondo, Norimichi Ukita  

**Link**: [PDF](https://arxiv.org/pdf/2504.19652)  

**Abstract**: This paper proposes an image-based robot motion planning method using a one-step diffusion model. While the diffusion model allows for high-quality motion generation, its computational cost is too expensive to control a robot in real time. To achieve high quality and efficiency simultaneously, our one-step diffusion model takes an approximately generated motion, which is predicted directly from input images. This approximate motion is optimized by additive noise provided by our novel noise optimizer. Unlike general isotropic noise, our noise optimizer adjusts noise anisotropically depending on the uncertainty of each motion element. Our experimental results demonstrate that our method outperforms state-of-the-art methods while maintaining its efficiency by one-step diffusion. 

**Abstract (ZH)**: 基于图像的一步扩散模型机器人运动规划方法 

---
# Adaptive Locomotion on Mud through Proprioceptive Sensing of Substrate Properties 

**Title (ZH)**: 通过本体感觉 substrate 属性的适配性运动 

**Authors**: Shipeng Liu, Jiaze Tang, Siyuan Meng, Feifei Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.19607)  

**Abstract**: Muddy terrains present significant challenges for terrestrial robots, as subtle changes in composition and water content can lead to large variations in substrate strength and force responses, causing the robot to slip or get stuck. This paper presents a method to estimate mud properties using proprioceptive sensing, enabling a flipper-driven robot to adapt its locomotion through muddy substrates of varying strength. First, we characterize mud reaction forces through actuator current and position signals from a statically mounted robotic flipper. We use the measured force to determine key coefficients that characterize intrinsic mud properties. The proprioceptively estimated coefficients match closely with measurements from a lab-grade load cell, validating the effectiveness of the proposed method. Next, we extend the method to a locomoting robot to estimate mud properties online as it crawls across different mud mixtures. Experimental data reveal that mud reaction forces depend sensitively on robot motion, requiring joint analysis of robot movement with proprioceptive force to determine mud properties correctly. Lastly, we deploy this method in a flipper-driven robot moving across muddy substrates of varying strengths, and demonstrate that the proposed method allows the robot to use the estimated mud properties to adapt its locomotion strategy, and successfully avoid locomotion failures. Our findings highlight the potential of proprioception-based terrain sensing to enhance robot mobility in complex, deformable natural environments, paving the way for more robust field exploration capabilities. 

**Abstract (ZH)**: 软泥地形给地面机器人带来了显著挑战，因为组成成分和水分含量的细微变化会导致基质强度和力响应产生较大变化，从而使机器人出现打滑或卡住的情况。本文提出了一种利用本体感觉估计泥地属性的方法，使鳍驱动机器人能够适应不同强度泥地地形的运动。首先，我们通过静置安装的机器人鳍片的执行器电流和位置信号来表征泥地的反应力，使用测量的力来确定描述内在泥地属性的关键系数。通过本体感觉估计得到的系数与实验室级载荷细胞测量值高度一致，验证了该方法的有效性。接下来，我们将该方法扩展到行进机器人，以在线估计机器人爬行过程中不同混合泥地的属性。实验数据表明，泥地的反应力对机器人运动高度敏感，需要结合机器人运动和本体感觉力的综合分析来正确确定泥地的属性。最后，我们将在不同强度泥地地形上移动的鳍驱动机器人中部署该方法，并证明所提出的方法使机器人能够利用估计的泥地属性来调整其运动策略，从而成功避免运动失败。我们的研究结果突显了基于本体感觉的地形感知在复杂可变形自然环境中的潜力，为更可靠的野外探索能力铺平了道路。 

---
# Smart Placement, Faster Robots -- A Comparison of Algorithms for Robot Base-Pose Optimization 

**Title (ZH)**: 智能定位，更快的机器人——机器人基座姿态优化算法对比 

**Authors**: Matthias Mayer, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2504.19577)  

**Abstract**: Robotic automation is a key technology that increases the efficiency and flexibility of manufacturing processes. However, one of the challenges in deploying robots in novel environments is finding the optimal base pose for the robot, which affects its reachability and deployment cost. Yet, the existing research for automatically optimizing the base pose of robots has not been compared. We address this problem by optimizing the base pose of industrial robots with Bayesian optimization, exhaustive search, genetic algorithms, and stochastic gradient descent and find that all algorithms can reduce the cycle time for various evaluated tasks in synthetic and real-world environments. Stochastic gradient descent shows superior performance with regard to success rate solving over 90% of our real-world tasks, while genetic algorithms show the lowest final costs. All benchmarks and implemented methods are available as baselines against which novel approaches can be compared. 

**Abstract (ZH)**: 机器人自动化是提高制造过程效率和灵活性的关键技术。然而，在部署机器人到新型环境中时，找到最优基座姿态以影响其可达性和部署成本的一个挑战是现有的自动优化机器人基座姿态的研究尚未进行比较。我们通过使用 Bayesian 优化、穷举搜索、遗传算法和随机梯度下降来优化工业机器人的基座姿态，并发现所有算法都能在合成和实际环境中减少各种评估任务的周期时间。随机梯度下降在解决超过90%的实际任务成功率方面表现出色，而遗传算法具有最低的最终成本。所有基准和实现方法均可作为新型方法的比较基准。 

---
# Video-Based Detection and Analysis of Errors in Robotic Surgical Training 

**Title (ZH)**: 基于视频的机器人手术培训中错误的检测与分析 

**Authors**: Hanna Kossowsky Lev, Yarden Sharon, Alex Geftler, Ilana Nisky  

**Link**: [PDF](https://arxiv.org/pdf/2504.19571)  

**Abstract**: Robot-assisted minimally invasive surgeries offer many advantages but require complex motor tasks that take surgeons years to master. There is currently a lack of knowledge on how surgeons acquire these robotic surgical skills. To help bridge this gap, we previously followed surgical residents learning complex surgical training dry-lab tasks on a surgical robot over six months. Errors are an important measure for self-training and for skill evaluation, but unlike in virtual simulations, in dry-lab training, errors are difficult to monitor automatically. Here, we analyzed the errors in the ring tower transfer task, in which surgical residents moved a ring along a curved wire as quickly and accurately as possible. We developed an image-processing algorithm to detect collision errors and achieved detection accuracy of ~95%. Using the detected errors and task completion time, we found that the surgical residents decreased their completion time and number of errors over the six months. This analysis provides a framework for detecting collision errors in similar surgical training tasks and sheds light on the learning process of the surgical residents. 

**Abstract (ZH)**: 机器人辅助微创手术提供了许多优势，但要求外科医生掌握复杂的运动任务，这需要数年时间。目前尚缺乏关于外科医生如何获得这些机器人手术技能的知识。为了填补这一空白，我们之前在六个月内跟踪了外科住院医师在手术机器人上学习复杂手术培训Dry Lab任务的过程。错误是自我训练和技能评估的重要指标，但在Dry Lab培训中，与虚拟模拟不同，错误难以自动监测。在此，我们分析了环塔楼转移任务中的错误，在该任务中，外科住院医师尽可能快速和准确地沿弯曲线移动环。我们开发了一种图像处理算法来检测碰撞错误，并实现了约95%的检测准确性。利用检测到的错误和任务完成时间，我们发现外科住院医师在六个月内减少了完成时间和错误数量。这一分析为检测类似手术培训任务中的碰撞错误提供了框架，并揭示了外科住院医师的学习过程。 

---
# Simultaneous Pick and Place Detection by Combining SE(3) Diffusion Models with Differential Kinematics 

**Title (ZH)**: 结合SE(3)扩散模型与差分运动学的同时拾放检测 

**Authors**: Tianyi Ko, Takuya Ikeda, Koichi Nishiwaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.19502)  

**Abstract**: Grasp detection methods typically target the detection of a set of free-floating hand poses that can grasp the object. However, not all of the detected grasp poses are executable due to physical constraints. Even though it is straightforward to filter invalid grasp poses in the post-process, such a two-staged approach is computationally inefficient, especially when the constraint is hard. In this work, we propose an approach to take the following two constraints into account during the grasp detection stage, namely, (i) the picked object must be able to be placed with a predefined configuration without in-hand manipulation (ii) it must be reachable by the robot under the joint limit and collision-avoidance constraints for both pick and place cases. Our key idea is to train an SE(3) grasp diffusion network to estimate the noise in the form of spatial velocity, and constrain the denoising process by a multi-target differential inverse kinematics with an inequality constraint, so that the states are guaranteed to be reachable and placement can be performed without collision. In addition to an improved success ratio, we experimentally confirmed that our approach is more efficient and consistent in computation time compared to a naive two-stage approach. 

**Abstract (ZH)**: 基于运动学约束的抓取检测方法 

---
# Motion Generation for Food Topping Challenge 2024: Serving Salmon Roe Bowl and Picking Fried Chicken 

**Title (ZH)**: 2024食品配料挑战中的运动生成：提供鲑鱼子碗和捡起炸鸡 

**Authors**: Koki Inami, Masashi Konosu, Koki Yamane, Nozomu Masuya, Yunhan Li, Yu-Han Shu, Hiroshi Sato, Shinnosuke Homma, Sho Sakaino  

**Link**: [PDF](https://arxiv.org/pdf/2504.19498)  

**Abstract**: Although robots have been introduced in many industries, food production robots are yet to be widely employed because the food industry requires not only delicate movements to handle food but also complex movements that adapt to the environment. Force control is important for handling delicate objects such as food. In addition, achieving complex movements is possible by making robot motions based on human teachings. Four-channel bilateral control is proposed, which enables the simultaneous teaching of position and force information. Moreover, methods have been developed to reproduce motions obtained through human teachings and generate adaptive motions using learning. We demonstrated the effectiveness of these methods for food handling tasks in the Food Topping Challenge at the 2024 IEEE International Conference on Robotics and Automation (ICRA 2024). For the task of serving salmon roe on rice, we achieved the best performance because of the high reproducibility and quick motion of the proposed method. Further, for the task of picking fried chicken, we successfully picked the most pieces of fried chicken among all participating teams. This paper describes the implementation and performance of these methods. 

**Abstract (ZH)**: 尽管机器人已在许多行业中应用，但由于食品行业需要精细操作以处理食物并适应复杂环境，食品生产机器人尚未广泛应用。力控制对于处理如食物等精细物体至关重要。此外，通过基于人类教学的机器人运动，可以实现复杂运动。本文提出了四通道双边控制方法，能够同时教授位置和力信息。同时，已经开发出再现通过人类教学获得的运动并利用学习生成适应性运动的方法。我们在2024年IEEE机器人与自动化国际会议（ICRA 2024）的食品配料挑战赛中展示了这些方法的有效性。对于在日本寿司饭上摆放鲑鱼子的任务，由于所提出方法的高再现性和快速运动，我们取得了最佳性能。此外，对于鸡块抓取任务，我们成功地在所有参赛队伍中抓取了最多的鸡块。本文描述了这些方法的实现及其性能。 

---
# Bearing-Only Tracking and Circumnavigation of a Fast Time-Varied Velocity Target Utilising an LSTM 

**Title (ZH)**: 基于LSTM的快速时变速度目标仅 Bearings 跟踪与环绕控制 

**Authors**: Mitchell Torok, Mohammad Deghat, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.19463)  

**Abstract**: Bearing-only tracking, localisation, and circumnavigation is a problem in which a single or a group of agents attempts to track a target while circumnavigating it at a fixed distance using only bearing measurements. While previous studies have addressed scenarios involving stationary targets or those moving with an unknown constant velocity, the challenge of accurately tracking a target moving with a time-varying velocity remains open. This paper presents an approach utilising a Long Short-Term Memory (LSTM) based estimator for predicting the target's position and velocity. We also introduce a corresponding control strategy. When evaluated against previously proposed estimation and circumnavigation approaches, our approach demonstrates significantly lower control and estimation errors across various time-varying velocity scenarios. Additionally, we illustrate the effectiveness of the proposed method in tracking targets with a double integrator nonholonomic system dynamics that mimic real-world systems. 

**Abstract (ZH)**: 仅凭航向角的目标跟踪、定位和环航是一个问题，即单个或多个代理试图使用仅有的航向测量数据，在固定距离上跟踪目标并沿其环航。尽管先前的研究已经处理了静止目标或未知恒定速度移动目标的情形，但准确跟踪具有时间变化速度的目标的挑战仍然存在。本文提出了一种利用基于长短期记忆（LSTM）的估计算法来预测目标的位置和速度的方法，并介绍了相应的控制策略。当与先前提出的估计和环航方法进行对比评估时，本文提出的方法在各种时间变化速度场景中表现出显著较低的控制和估计误差。此外，本文还展示了在模拟真实系统动力学的双积分非完整系统动力学下，提出的方法在跟踪目标方面的有效性。 

---
# Follow Everything: A Leader-Following and Obstacle Avoidance Framework with Goal-Aware Adaptation 

**Title (ZH)**: 跟随一切：一种带有目标意识适应性的领导者跟随与避障框架 

**Authors**: Qianyi Zhang, Shijian Ma, Boyi Liu, Jingtai Liu, Jianhao Jiao, Dimitrios Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2504.19399)  

**Abstract**: Robust and flexible leader-following is a critical capability for robots to integrate into human society. While existing methods struggle to generalize to leaders of arbitrary form and often fail when the leader temporarily leaves the robot's field of view, this work introduces a unified framework addressing both challenges. First, traditional detection models are replaced with a segmentation model, allowing the leader to be anything. To enhance recognition robustness, a distance frame buffer is implemented that stores leader embeddings at multiple distances, accounting for the unique characteristics of leader-following tasks. Second, a goal-aware adaptation mechanism is designed to govern robot planning states based on the leader's visibility and motion, complemented by a graph-based planner that generates candidate trajectories for each state, ensuring efficient following with obstacle avoidance. Simulations and real-world experiments with a legged robot follower and various leaders (human, ground robot, UAV, legged robot, stop sign) in both indoor and outdoor environments show competitive improvements in follow success rate, reduced visual loss duration, lower collision rate, and decreased leader-follower distance. 

**Abstract (ZH)**: 鲁棒且灵活的跟随能力是机器人融入人类社会的关键能力。现有方法难以泛化到任意形式的领导者，并且当领导者暂时离开机器人的视野范围时往往无法有效应对，本研究提出了一种统一框架以应对上述挑战。首先，传统的检测模型被分割模型所取代，使得领导者可以是任意物体。为了增强识别的鲁棒性，实现了一个距离帧缓冲区，该缓冲区存储了不同距离的领导者嵌入，以考虑跟随任务的独特特性。其次，设计了一个目标感知的自适应机制，根据领导者的可见性和运动来管理机器人的规划状态，配合基于图的规划器为每个状态生成候选轨迹，确保高效跟随的同时实现避障。在室内和室外环境中，使用四足机器人跟随者和多种领导者（人类、地面机器人、UAV、四足机器人、停止标志）进行的仿真和实地实验显示，在跟随成功率、视觉丧失时间、碰撞率和领导者跟随者距离方面均取得了显著改进。 

---
# PolyTouch: A Robust Multi-Modal Tactile Sensor for Contact-rich Manipulation Using Tactile-Diffusion Policies 

**Title (ZH)**: PolyTouch: 一种用于接触丰富操作的稳健多模态触觉传感器及触觉扩散策略 

**Authors**: Jialiang Zhao, Naveen Kuppuswamy, Siyuan Feng, Benjamin Burchfiel, Edward Adelson  

**Link**: [PDF](https://arxiv.org/pdf/2504.19341)  

**Abstract**: Achieving robust dexterous manipulation in unstructured domestic environments remains a significant challenge in robotics. Even with state-of-the-art robot learning methods, haptic-oblivious control strategies (i.e. those relying only on external vision and/or proprioception) often fall short due to occlusions, visual complexities, and the need for precise contact interaction control. To address these limitations, we introduce PolyTouch, a novel robot finger that integrates camera-based tactile sensing, acoustic sensing, and peripheral visual sensing into a single design that is compact and durable. PolyTouch provides high-resolution tactile feedback across multiple temporal scales, which is essential for efficiently learning complex manipulation tasks. Experiments demonstrate an at least 20-fold increase in lifespan over commercial tactile sensors, with a design that is both easy to manufacture and scalable. We then use this multi-modal tactile feedback along with visuo-proprioceptive observations to synthesize a tactile-diffusion policy from human demonstrations; the resulting contact-aware control policy significantly outperforms haptic-oblivious policies in multiple contact-aware manipulation policies. This paper highlights how effectively integrating multi-modal contact sensing can hasten the development of effective contact-aware manipulation policies, paving the way for more reliable and versatile domestic robots. More information can be found at this https URL 

**Abstract (ZH)**: 在无结构家庭环境实现稳健的灵巧操作仍然是机器人技术中的一个重大挑战。即使使用最先进的机器人学习方法，仅依赖外部视觉和/或本体感受的触觉无意识控制策略（即那些不依赖触觉反馈的策略）常常由于遮挡、视觉复杂性和精确接触交互控制需求而效果不佳。为了解决这些限制，我们提出了一种名为PolyTouch的新型机器人手指，其将基于摄像头的触觉传感、声学传感和边缘视觉传感集成到一个紧凑且耐用的设计中。PolyTouch提供了多时间尺度的高分辨率触觉反馈，这对于高效学习复杂操作任务至关重要。实验表明，其寿命至少比商用触觉传感器提高了20倍，且设计易于制造并可扩展。然后，我们利用这种多模态触觉反馈以及视触觉观察，从人类示范中合成了一种触觉扩散策略；结果，该接触感知控制策略在多种接触感知操作中显著优于触觉无意识策略。本文强调了有效集成多模态接触传感如何加速实现有效的接触感知操作策略的开发，为更加可靠和多用途的家庭机器人铺平了道路。更多信息请访问<https://>。 

---
# Efficient COLREGs-Compliant Collision Avoidance using Turning Circle-based Control Barrier Function 

**Title (ZH)**: 基于转向圈基控制约束函数的有效遵 Compliance 航行规则避碰控制 

**Authors**: Changyu Lee, Jinwook Park, Jinwhan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.19247)  

**Abstract**: This paper proposes a computationally efficient collision avoidance algorithm using turning circle-based control barrier functions (CBFs) that comply with international regulations for preventing collisions at sea (COLREGs). Conventional CBFs often lack explicit consideration of turning capabilities and avoidance direction, which are key elements in developing a COLREGs-compliant collision avoidance algorithm. To overcome these limitations, we introduce two CBFs derived from left and right turning circles. These functions establish safety conditions based on the proximity between the traffic ships and the centers of the turning circles, effectively determining both avoidance directions and turning capabilities. The proposed method formulates a quadratic programming problem with the CBFs as constraints, ensuring safe navigation without relying on computationally intensive trajectory optimization. This approach significantly reduces computational effort while maintaining performance comparable to model predictive control-based methods. Simulation results validate the effectiveness of the proposed algorithm in enabling COLREGs-compliant, safe navigation, demonstrating its potential for reliable and efficient operation in complex maritime environments. 

**Abstract (ZH)**: 本文提出了一种基于转向圆控制障碍函数（CBFs）的计算高效避碰算法，并符合国际海上避碰规则（COLREGs）。传统CBFs往往未能明确考虑转向能力和避碰方向，这是制定符合COLREGs的避碰算法的关键要素。为克服这些限制，我们引入了源自左转和右转转向圆的两种CBFs。这些函数基于航分区船舶与转向圆中心之间的接近程度，有效地确定了避碰方向和转向能力。所提出的方法将CBFs作为约束形式ulating一个二次规划问题，确保安全航行而不依赖于计算 intensive的轨迹优化。该方法显著减少了计算负担，同时保持与基于模型预测控制的方法相当的性能。仿真结果验证了所提出算法在实现COLREGs合规、安全航行方面的有效性，展示了其在复杂 maritime环境下可靠和高效操作的潜力。 

---
# Robotic Trail Maker Platform for Rehabilitation in Neurological Conditions: Clinical Use Cases 

**Title (ZH)**: 神经科康复中的机器人路径铺设平台：临床案例 

**Authors**: Srikar Annamraju, Harris Nisar, Dayu Xia, Shankar A. Deka, Anne Horowitz, Nadica Miljković, Dušan M. Stipanović  

**Link**: [PDF](https://arxiv.org/pdf/2504.19230)  

**Abstract**: Patients with neurological conditions require rehabilitation to restore their motor, visual, and cognitive abilities. To meet the shortage of therapists and reduce their workload, a robotic rehabilitation platform involving the clinical trail making test is proposed. Therapists can create custom trails for each patient and the patient can trace the trails using a robotic device. The platform can track the performance of the patient and use these data to provide dynamic assistance through the robot to the patient interface. Therefore, the proposed platform not only functions as an evaluation platform, but also trains the patient in recovery. The developed platform has been validated at a rehabilitation center, with therapists and patients operating the device. It was found that patients performed poorly while using the platform compared to healthy subjects and that the assistance provided also improved performance amongst patients. Statistical analysis demonstrated that the speed of the patients was significantly enhanced with the robotic assistance. Further, neural networks are trained to classify between patients and healthy subjects and to forecast their movements using the data collected. 

**Abstract (ZH)**: 神经条件患者需要康复以恢复其运动、视觉和认知能力。为缓解治疗师短缺和减轻其工作负担，提出了一种涉及临床连线绘制测试的机器人康复平台。治疗师可以为每位患者创建定制化 trails，患者使用机器人设备进行跟踪。该平台可以跟踪患者的性能，并利用这些数据通过机器人向患者界面提供动态辅助。因此，所提出的平台不仅作为评估平台使用，还用于训练患者的康复。所开发的平台已在康复中心得到验证，由治疗师和患者操作设备。研究发现，与健康对照组相比，患者在使用平台时表现较差，但提供的辅助也改善了患者的性能。统计分析表明，机器人辅助显著提高了患者的运动速度。此外，使用收集到的数据训练神经网络以区分患者和健康个体，并预测其运动。 

---
# Geometric Gait Optimization for Kinodynamic Systems Using a Lie Group Integrator 

**Title (ZH)**: 几何步态优化在李群积分器下的动力学系统中应用 

**Authors**: Yanhao Yang, Ross L. Hatton  

**Link**: [PDF](https://arxiv.org/pdf/2504.19072)  

**Abstract**: This paper presents a gait optimization and motion planning framework for a class of locomoting systems with mixed kinematic and dynamic properties. Using Lagrangian reduction and differential geometry, we derive a general dynamic model that incorporates second-order dynamics and nonholonomic constraints, applicable to kinodynamic systems such as wheeled robots with nonholonomic constraints as well as swimming robots with nonisotropic fluid-added inertia and hydrodynamic drag. Building on Lie group integrators and group symmetries, we develop a variational gait optimization method for kinodynamic systems. By integrating multiple gaits and their transitions, we construct comprehensive motion plans that enable a wide range of motions for these systems. We evaluate our framework on three representative examples: roller racer, snakeboard, and swimmer. Simulation and hardware experiments demonstrate diverse motions, including acceleration, steady-state maintenance, gait transitions, and turning. The results highlight the effectiveness of the proposed method and its potential for generalization to other biological and robotic locomoting systems. 

**Abstract (ZH)**: 一种混合动力学与动力学属性的运动系统步态优化与运动规划框架 

---
# Efficient Control Allocation and 3D Trajectory Tracking of a Highly Manoeuvrable Under-actuated Bio-inspired AUV 

**Title (ZH)**: 高效控制分配与 Highly Manoeuvrable Under-actuated Bio-inspired AUV 的三维轨迹跟踪 

**Authors**: Walid Remmas, Christian Meurer, Yuya Hamamatsu, Ahmed Chemori, Maarja Kruusmaa  

**Link**: [PDF](https://arxiv.org/pdf/2504.19049)  

**Abstract**: Fin actuators can be used for for both thrust generation and vectoring. Therefore, fin-driven autonomous underwater vehicles (AUVs) can achieve high maneuverability with a smaller number of actuators, but their control is challenging. This study proposes an analytic control allocation method for underactuated Autonomous Underwater Vehicles (AUVs). By integrating an adaptive hybrid feedback controller, we enable an AUV with 4 actuators to move in 6 degrees of freedom (DOF) in simulation and up to 5-DOF in real-world experiments. The proposed method outperformed state-of-the-art control allocation techniques in 6-DOF trajectory tracking simulations, exhibiting centimeter-scale accuracy and higher energy and computational efficiency. Real-world pool experiments confirmed the method's robustness and efficacy in tracking complex 3D trajectories, with significant computational efficiency gains 0.007 (ms) vs. 22.28 (ms). Our method offers a balance between performance, energy efficiency, and computational efficiency, showcasing a potential avenue for more effective tracking of a large number of DOF for under-actuated underwater robots. 

**Abstract (ZH)**: 基于鳍驱动的欠驱动自治水下车辆的解析控制分配方法 

---
# An SE(3) Noise Model for Range-Azimuth-Elevation Sensors 

**Title (ZH)**: SE(3)噪声模型用于距离-方位-仰角传感器 

**Authors**: Thomas Hitchcox, James Richard Forbes  

**Link**: [PDF](https://arxiv.org/pdf/2504.19009)  

**Abstract**: Scan matching is a widely used technique in state estimation. Point-cloud alignment, one of the most popular methods for scan matching, is a weighted least-squares problem in which the weights are determined from the inverse covariance of the measured points. An inaccurate representation of the covariance will affect the weighting of the least-squares problem. For example, if ellipsoidal covariance bounds are used to approximate the curved, "banana-shaped" noise characteristics of many scanning sensors, the weighting in the least-squares problem may be overconfident. Additionally, sensor-to-vehicle extrinsic uncertainty and odometry uncertainty during submap formation are two sources of uncertainty that are often overlooked in scan matching applications, also likely contributing to overconfidence on the scan matching estimate. This paper attempts to address these issues by developing a model for range-azimuth-elevation sensors on matrix Lie groups. The model allows for the seamless incorporation of extrinsic and odometry uncertainty. Illustrative results are shown both for a simulated example and for a real point-cloud submap collected with an underwater laser scanner. 

**Abstract (ZH)**: 扫描匹配是一种广泛应用于状态估计的技术。点云对齐，作为扫描匹配中最流行的方法之一，是一个加权最小二乘问题，其中权重由测量点的逆协方差决定。协方差的不准确表示将影响最小二乘问题中的权重。例如，如果使用椭球协方差边界来近似许多扫描传感器的弯曲的“香蕉形”噪声特性，最小二乘问题中的权重可能会过于自信。另外，传感器到车辆的外参不确定性以及子地图构建期间的里程计不确定性也是扫描匹配应用中常被忽视的两种不确定性来源，也可能导致对扫描匹配估计过于自信。本文通过在矩阵李群上开发范围-方位-仰角传感器模型试图解决这些问题，该模型允许无缝地整合外参和里程计不确定性。文中展示了模拟示例和使用水下激光扫描器收集的实际点云子地图的示例结果。 

---
# A biconvex method for minimum-time motion planning through sequences of convex sets 

**Title (ZH)**: 双凸方法在凸集序列通过下的最小时间运动规划 

**Authors**: Tobia Marcucci, Mathew Halm, Will Yang, Dongchan Lee, Andrew D. Marchese  

**Link**: [PDF](https://arxiv.org/pdf/2504.18978)  

**Abstract**: We consider the problem of designing a smooth trajectory that traverses a sequence of convex sets in minimum time, while satisfying given velocity and acceleration constraints. This problem is naturally formulated as a nonconvex program. To solve it, we propose a biconvex method that quickly produces an initial trajectory and iteratively refines it by solving two convex subproblems in alternation. This method is guaranteed to converge, returns a feasible trajectory even if stopped early, and does not require the selection of any line-search or trust-region parameter. Exhaustive experiments show that our method finds high-quality trajectories in a fraction of the time of state-of-the-art solvers for nonconvex optimization. In addition, it achieves runtimes comparable to industry-standard waypoint-based motion planners, while consistently designing lower-duration trajectories than existing optimization-based planners. 

**Abstract (ZH)**: 我们考虑设计一条平滑轨迹，使其在满足给定的速度和加速度约束条件下，以最小时间穿越一系列凸集。该问题自然地形式化为一个非凸规划问题。为了解决这个问题，我们提出了一种双凸方法，该方法能够快速生成初始轨迹，并通过交替求解两个凸子问题来逐步优化它。该方法能得到收敛保证，即使提前停止也能返回可行轨迹，且无需选择任何线搜索或信任区域参数。详尽的实验表明，我们的方法在非凸优化中最先进的求解器所需时间的一小部分内就能找到高质量的轨迹。此外，它在运行时间上与基于航点的工业标准轨迹规划器相当，但始终能够设计出比现有基于优化的轨迹规划器更短时间的轨迹。 

---
# Hierarchical Temporal Logic Task and Motion Planning for Multi-Robot Systems 

**Title (ZH)**: 多机器人系统分层时序逻辑任务与运动规划 

**Authors**: Zhongqi Wei, Xusheng Luo, Changliu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18899)  

**Abstract**: Task and motion planning (TAMP) for multi-robot systems, which integrates discrete task planning with continuous motion planning, remains a challenging problem in robotics. Existing TAMP approaches often struggle to scale effectively for multi-robot systems with complex specifications, leading to infeasible solutions and prolonged computation times. This work addresses the TAMP problem in multi-robot settings where tasks are specified using expressive hierarchical temporal logic and task assignments are not pre-determined. Our approach leverages the efficiency of hierarchical temporal logic specifications for task-level planning and the optimization-based graph of convex sets method for motion-level planning, integrating them within a product graph framework. At the task level, we convert hierarchical temporal logic specifications into a single graph, embedding task allocation within its edges. At the motion level, we represent the feasible motions of multiple robots through convex sets in the configuration space, guided by a sampling-based motion planner. This formulation allows us to define the TAMP problem as a shortest path search within the product graph, where efficient convex optimization techniques can be applied. We prove that our approach is both sound and complete under mild assumptions. Additionally, we extend our framework to cooperative pick-and-place tasks involving object handovers between robots. We evaluate our method across various high-dimensional multi-robot scenarios, including simulated and real-world environments with quadrupeds, robotic arms, and automated conveyor systems. Our results show that our approach outperforms existing methods in execution time and solution optimality while effectively scaling with task complexity. 

**Abstract (ZH)**: 多机器人系统中基于表达性层次时序逻辑的任务与运动规划（TAMP）及其应用 

---
# Diffeomorphic Obstacle Avoidance for Contractive Dynamical Systems via Implicit Representations 

**Title (ZH)**: 基于隐式表示的收敛动力系统可变形障碍避让 

**Authors**: Ken-Joel Simmoteit, Philipp Schillinger, Leonel Rozo  

**Link**: [PDF](https://arxiv.org/pdf/2504.18860)  

**Abstract**: Ensuring safety and robustness of robot skills is becoming crucial as robots are required to perform increasingly complex and dynamic tasks. The former is essential when performing tasks in cluttered environments, while the latter is relevant to overcome unseen task situations. This paper addresses the challenge of ensuring both safety and robustness in dynamic robot skills learned from demonstrations. Specifically, we build on neural contractive dynamical systems to provide robust extrapolation of the learned skills, while designing a full-body obstacle avoidance strategy that preserves contraction stability via diffeomorphic transforms. This is particularly crucial in complex environments where implicit scene representations, such as Signed Distance Fields (SDFs), are necessary. To this end, our framework called Signed Distance Field Diffeomorphic Transform, leverages SDFs and flow-based diffeomorphisms to achieve contraction-preserving obstacle avoidance. We thoroughly evaluate our framework on synthetic datasets and several real-world robotic tasks in a kitchen environment. Our results show that our approach locally adapts the learned contractive vector field while staying close to the learned dynamics and without introducing highly-curved motion paths, thus outperforming several state-of-the-art methods. 

**Abstract (ZH)**: 确保机器人技能的安全性和鲁棒性正变得越来越重要，尤其是在机器人需要执行日益复杂和动态的任务时。前者在进行 cluttered 环境中的任务时至关重要，而后者则与克服未知任务情境相关。本文探讨了在示例中学习的动态机器人技能中确保同时具备安全性和鲁棒性的挑战。具体来说，我们基于神经收敛动力学系统提供学习技能的稳健外推，并设计了一种全身障碍物规避策略，通过差分同胚变换保持收缩稳定性。这对于复杂环境中尤为重要，尤其是当需要使用隐式场景表示（如符号距离场 SDF）时。为此，我们提出了一种名为符号距离场差分同胚变换的框架，利用 SDF 和基于流的差分同胚来实现保持收缩的障碍物规避。我们通过合成数据集和厨房环境中的多个真实世界机器人任务全面评估了该框架，结果显示，我们的方法在局部适应所学的收敛向量场的同时保持接近所学的动力学，并避免了引入高度弯曲的运动路径，从而优于几种最先进的方法。 

---
# A Microgravity Simulation Experimental Platform For Small Space Robots In Orbit 

**Title (ZH)**: 轨道小型太空机器人微重力模拟实验平台 

**Authors**: Hang Luo, Nanlin Zhou, Haoxiang Zhang, Kai Han, Ning Zhao, Zhiyuan Yang, Jian Qi, Sikai Zhao, Jie Zhao, Yanhe Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.18842)  

**Abstract**: This study describes the development and validation of a novel microgravity experimental platform that is mainly applied to small robots such as modular self-reconfigurable robots. This platform mainly consists of an air supply system, a microporous platform and glass. By supplying air to the microporous platform to form an air film, the influence of the weight of the air foot and the ventilation hose of traditional air-float platforms on microgravity experiments is solved. The contribution of this work is to provide a platform with less external interference for microgravity simulation experiments on small robots. 

**Abstract (ZH)**: 本研究描述了一种新型微重力实验平台的发展与验证，该平台主要用于模块化自重构机器人等小型机器人。该平台主要由气动供应系统、微孔平台和玻璃构成。通过向微孔平台供气形成气膜，解决了传统气浮平台的气脚重量和通风管对微重力实验的影响。本研究的贡献在于为小型机器人微重力模拟实验提供了一个外部干扰更少的平台。 

---
# Swarming in the Wild: A Distributed Communication-less Lloyd-based Algorithm dealing with Uncertainties 

**Title (ZH)**: 野外集群：一种处理不确定性无需通信的分布式Lloyd算法 

**Authors**: Manuel Boldrer, Vit Kratky, Viktor Walter, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2504.18840)  

**Abstract**: In this work, we present a distributed algorithm for swarming in complex environments that operates with no communication, no a priori information about the environment, and using only onboard sensing and computation capabilities. We provide sufficient conditions to guarantee that each robot reaches its goal region in a finite time, avoiding collisions with obstacles and other robots without exceeding a desired maximum distance from a predefined set of neighbors (flocking constraint). In addition, we show how the proposed algorithm can deal with tracking errors and onboard sensing errors without violating safety and proximity constraints, still providing the conditions for having convergence towards the goal region. To validate the approach, we provide experiments in the field. We tested our algorithm in GNSS-denied environments i.e., a dense forest, where fully autonomous aerial robots swarmed safely to the desired destinations, by relying only on onboard sensors, i.e., without a communication network. This work marks the initial deployment of a fully distributed system where there is no communication between the robots, nor reliance on any global localization system, which at the same time it ensures safety and convergence towards the goal within such complex environments. 

**Abstract (ZH)**: 一种在复杂环境中无需通信的分布式群集算法及其应用 

---
# Aerial Robots Persistent Monitoring and Target Detection: Deployment and Assessment in the Field 

**Title (ZH)**: 空中机器人持续监测与目标检测：现场部署与评估 

**Authors**: Manuel Boldrer, Vit Kratky, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2504.18832)  

**Abstract**: In this manuscript, we present a distributed algorithm for multi-robot persistent monitoring and target detection. In particular, we propose a novel solution that effectively integrates the Time-inverted Kuramoto model, three-dimensional Lissajous curves, and Model Predictive Control. We focus on the implementation of this algorithm on aerial robots, addressing the practical challenges involved in deploying our approach under real-world conditions. Our method ensures an effective and robust solution that maintains operational efficiency even in the presence of what we define as type I and type II failures. Type I failures refer to short-time disruptions, such as tracking errors and communication delays, while type II failures account for long-time disruptions, including malicious attacks, severe communication failures, and battery depletion. Our approach guarantees persistent monitoring and target detection despite these challenges. Furthermore, we validate our method with extensive field experiments involving up to eleven aerial robots, demonstrating the effectiveness, resilience, and scalability of our solution. 

**Abstract (ZH)**: 本论文提出了一种分布式多_robot持久监测与目标检测算法。特别地，我们提出了一种新颖的解决方案，有效地整合了时间倒置库拉莫模型、三维利萨茹曲线和模型预测控制。我们重点讨论了在实际条件下部署该方法所面临的实际挑战，并确保我们的方法即使在我们定义的类型I和类型II故障存在的情况下也能提供有效且 robust 的解决方案。类型I故障指的是短时间中断，如跟踪误差和通信延迟，而类型II故障涉及长时间中断，包括恶意攻击、严重通信故障和电池耗尽。我们的方法能够在这些挑战下保证持久监测和目标检测。此外，我们通过涉及多达十一架飞行机器人的大量野外试验验证了该方法，展示了我们解决方案的有效性、韧性和可扩展性。 

---
# Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy 

**Title (ZH)**: dexonomy: 合成抓持分类学中的所有灵巧握持类型 

**Authors**: Jiayi Chen, Yubin Ke, Lin Peng, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18829)  

**Abstract**: Generalizable dexterous grasping with suitable grasp types is a fundamental skill for intelligent robots. Developing such skills requires a large-scale and high-quality dataset that covers numerous grasp types (i.e., at least those categorized by the GRASP taxonomy), but collecting such data is extremely challenging. Existing automatic grasp synthesis methods are often limited to specific grasp types or object categories, hindering scalability. This work proposes an efficient pipeline capable of synthesizing contact-rich, penetration-free, and physically plausible grasps for any grasp type, object, and articulated hand. Starting from a single human-annotated template for each hand and grasp type, our pipeline tackles the complicated synthesis problem with two stages: optimize the object to fit the hand template first, and then locally refine the hand to fit the object in simulation. To validate the synthesized grasps, we introduce a contact-aware control strategy that allows the hand to apply the appropriate force at each contact point to the object. Those validated grasps can also be used as new grasp templates to facilitate future synthesis. Experiments show that our method significantly outperforms previous type-unaware grasp synthesis baselines in simulation. Using our algorithm, we construct a dataset containing 10.7k objects and 9.5M grasps, covering 31 grasp types in the GRASP taxonomy. Finally, we train a type-conditional generative model that successfully performs the desired grasp type from single-view object point clouds, achieving an 82.3% success rate in real-world experiments. Project page: this https URL. 

**Abstract (ZH)**: 通用可转移的 Dexterous 抓取技能对于智能机器人来说是一项基本技能。开发此类技能需要大规模且高质量的数据集，涵盖多种抓取类型（即按照 GRASP 分类法分类的类型），但收集此类数据极具挑战性。现有的自动抓取合成方法通常局限于特定的抓取类型或对象类别，阻碍了其扩展性。本工作提出了一种高效的工作流程，能够为任何抓取类型、对象和 articulated 手合成接触丰富、无穿刺且物理上可实现的抓取。从每个手和抓取类型的单一个人注释模板开始，本工作流程通过两个阶段解决复杂的合成问题：首先优化对象以匹配手模板，然后在模拟中局部优化手以适应对象。为了验证合成的抓取，我们引入了一种接触感知的控制策略，允许手在每个接触点上施加适当的力到对象上。这些验证过的抓取也可以作为新的抓取模板，以促进未来的合成。实验表明，与之前的无抓取类型感知的抓取合成基准相比，本方法在模拟中表现显著优越。使用我们的算法，我们构建了一个包含 10,700 个对象和 9.5 百万抓取的数据集，涵盖了 GRASP 分类法中的 31 种抓取类型。最后，我们训练了一个基于类型的生成模型，可以从单视图对象点云中成功地执行所需的抓取类型，在真实世界实验中成功率达到 82.3%。项目页面：this https URL。 

---
# Design, Contact Modeling, and Collision-inclusive Planning of a Dual-stiffness Aerial RoboT (DART) 

**Title (ZH)**: 设计、接触建模及碰撞包容规划的双刚度空中机器人(DART) 

**Authors**: Yogesh Kumar, Karishma Patnaik, Wenlong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18780)  

**Abstract**: Collision-resilient quadrotors have gained significant attention given their potential for operating in cluttered environments and leveraging impacts to perform agile maneuvers. However, existing designs are typically single-mode: either safeguarded by propeller guards that prevent deformation or deformable but lacking rigidity, which is crucial for stable flight in open environments. This paper introduces DART, a Dual-stiffness Aerial RoboT, that adapts its post-collision response
by either engaging a locking mechanism for a rigid mode or disengaging it for a flexible mode, respectively. Comprehensive characterization tests highlight the significant difference in post collision responses between its rigid and flexible modes, with the rigid mode offering seven times higher stiffness compared to the flexible mode. To understand and harness the collision dynamics, we propose a novel collision response prediction model based on the linear complementarity system theory. We demonstrate the accuracy of predicting collision forces for both the rigid and flexible modes of DART. Experimental results confirm the accuracy of the model and underscore its potential to advance collision-inclusive trajectory planning in aerial robotics. 

**Abstract (ZH)**: 双刚度空中机器人 DART：基于碰撞后响应适应的新型设计 

---
# Certifiably-Correct Mapping for Safe Navigation Despite Odometry Drift 

**Title (ZH)**: 可认证正确的映射以实现即使在里程计漂移情况下的安全导航 

**Authors**: Devansh R. Agrawal, Taekyung Kim, Rajiv Govindjee, Trushant Adeshara, Jiangbo Yu, Anurekha Ravikumar, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2504.18713)  

**Abstract**: Accurate perception, state estimation and mapping are essential for safe robotic navigation as planners and controllers rely on these components for safety-critical decisions. However, existing mapping approaches often assume perfect pose estimates, an unrealistic assumption that can lead to incorrect obstacle maps and therefore collisions. This paper introduces a framework for certifiably-correct mapping that ensures that the obstacle map correctly classifies obstacle-free regions despite the odometry drift in vision-based localization systems (VIO}/SLAM). By deflating the safe region based on the incremental odometry error at each timestep, we ensure that the map remains accurate and reliable locally around the robot, even as the overall odometry error with respect to the inertial frame grows unbounded.
Our contributions include two approaches to modify popular obstacle mapping paradigms, (I) Safe Flight Corridors, and (II) Signed Distance Fields. We formally prove the correctness of both methods, and describe how they integrate with existing planning and control modules. Simulations using the Replica dataset highlight the efficacy of our methods compared to state-of-the-art techniques. Real-world experiments with a robotic rover show that, while baseline methods result in collisions with previously mapped obstacles, the proposed framework enables the rover to safely stop before potential collisions. 

**Abstract (ZH)**: 准确感知、状态估计与建图对于安全的机器人导航至关重要，因为规划器和控制器依赖这些组件来做安全关键决策。然而，现有建图方法往往假设完美的姿态估计，这是一个不现实的假设，可能导致错误的障碍物地图并因此引发碰撞。本文介绍了一种确认证确的建图框架，确保即使在基于视觉定位系统（VIO/SLAM）的姿态漂移情况下，障碍物地图也能正确分类无障碍区域。通过基于每个时间步的增量姿态误差进行区域收缩，我们确保地图在机器人周围保持局部准确可靠，即使全局姿态误差相对于惯性框架无限增长也是如此。我们的贡献包括两类修改流行障碍物建图方法的途径，（I）安全飞行走廊，（II）符号距离场。我们形式化证明了两种方法的正确性，并描述了它们如何与现有的规划和控制模块集成。复制品数据集的仿真结果显示了我们方法的有效性，优于现有最先进的技术。在机器人漫游车的实际实验中表明，尽管基线方法会导致与先前映射的障碍物相撞，但提出的框架使漫游车能够在潜在碰撞前安全停止。 

---
# Learning-Based Modeling of Soft Actuators Using Euler Spiral-Inspired Curvature 

**Title (ZH)**: 基于欧拉螺旋启发的曲率学习建模方法用于软执行器 

**Authors**: Yu Mei, Shangyuan Yuan, Xinda Qi, Preston Fairchild, Xiaobo Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.18692)  

**Abstract**: Soft robots, distinguished by their inherent compliance and continuum structures, present unique modeling challenges, especially when subjected to significant external loads such as gravity and payloads. In this study, we introduce an innovative data-driven modeling framework leveraging an Euler spiral-inspired shape representations to accurately describe the complex shapes of soft continuum actuators. Based on this representation, we develop neural network-based forward and inverse models to effectively capture the nonlinear behavior of a fiber-reinforced pneumatic bending actuator. Our forward model accurately predicts the actuator's deformation given inputs of pressure and payload, while the inverse model reliably estimates payloads from observed actuator shapes and known pressure inputs. Comprehensive experimental validation demonstrates the effectiveness and accuracy of our proposed approach. Notably, the augmented Euler spiral-based forward model achieves low average positional prediction errors of 3.38%, 2.19%, and 1.93% of the actuator length at the one-third, two-thirds, and tip positions, respectively. Furthermore, the inverse model demonstrates precision of estimating payloads with an average error as low as 0.72% across the tested range. These results underscore the potential of our method to significantly enhance the accuracy and predictive capabilities of modeling frameworks for soft robotic systems. 

**Abstract (ZH)**: 软体机器人因其固有的顺应性和连续结构，在承受重力和载荷等显著外部负载时，呈现出独特的建模挑战。本文引入了一种基于欧拉螺旋启发式的数据驱动建模框架，以准确描述软连续执行器的复杂形状。在此基础上，我们开发了基于神经网络的正向和逆向模型，有效捕捉了纤维增强气动弯曲执行器的非线性行为。我们的正向模型能够根据压力和载荷输入准确预测执行器的变形，而逆向模型可以从观测到的执行器形状和已知的压力输入中可靠地估计载荷。全面的实验验证证明了所提方法的有效性和准确性。值得注意的是，增强的欧拉螺旋基于的正向模型分别在执行器长度的三分之一、二分之一和尖端位置实现了平均位置预测误差低至3.38%、2.19%和1.93%。此外，逆向模型在测试范围内估计载荷的精度平均误差低至0.72%。这些结果强调了本文方法在提升软体机器人系统建模框架的准确性和预测能力方面的潜在价值。 

---
# Collaborative Object Transportation in Space via Impact Interactions 

**Title (ZH)**: 空间中基于碰撞交互的协同对象运输 

**Authors**: Joris Verhagen, Jana Tumova  

**Link**: [PDF](https://arxiv.org/pdf/2504.18667)  

**Abstract**: We present a planning and control approach for collaborative transportation of objects in space by a team of robots. Object and robots in microgravity environments are not subject to friction but are instead free floating. This property is key to how we approach the transportation problem: the passive objects are controlled by impact interactions with the controlled robots. In particular, given a high-level Signal Temporal Logic (STL) specification of the transportation task, we synthesize motion plans for the robots to maximize the specification satisfaction in terms of spatial STL robustness. Given that the physical impact interactions are complex and hard to model precisely, we also present an alternative formulation maximizing the permissible uncertainty in a simplified kinematic impact model. We define the full planning and control stack required to solve the object transportation problem; an offline planner, an online replanner, and a low-level model-predictive control scheme for each of the robots. We show the method in a high-fidelity simulator for a variety of scenarios and present experimental validation of 2-robot, 1-object scenarios on a freeflyer platform. 

**Abstract (ZH)**: 一种用于空间机器人团队协作运输物体的规划与控制方法 

---
# Modelling of Underwater Vehicles using Physics-Informed Neural Networks with Control 

**Title (ZH)**: 使用控制信息的物理知情神经网络建模 underwater vehicles 

**Authors**: Abdelhakim Amer, David Felsager, Yury Brodskiy, Andriy Sarabakha  

**Link**: [PDF](https://arxiv.org/pdf/2504.20019)  

**Abstract**: Physics-informed neural networks (PINNs) integrate physical laws with data-driven models to improve generalization and sample efficiency. This work introduces an open-source implementation of the Physics-Informed Neural Network with Control (PINC) framework, designed to model the dynamics of an underwater vehicle. Using initial states, control actions, and time inputs, PINC extends PINNs to enable physically consistent transitions beyond the training domain. Various PINC configurations are tested, including differing loss functions, gradient-weighting schemes, and hyperparameters. Validation on a simulated underwater vehicle demonstrates more accurate long-horizon predictions compared to a non-physics-informed baseline 

**Abstract (ZH)**: 基于物理的神经网络（PINC）框架：一种用于 underwater 机器人动力学建模的开源实现 

---
# Trajectory Planning with Model Predictive Control for Obstacle Avoidance Considering Prediction Uncertainty 

**Title (ZH)**: 考虑预测不确定性的模型预测控制避障轨迹规划 

**Authors**: Eric Schöneberg, Michael Schröder, Daniel Görges, Hans D. Schotten  

**Link**: [PDF](https://arxiv.org/pdf/2504.19193)  

**Abstract**: This paper introduces a novel trajectory planner for autonomous robots, specifically designed to enhance navigation by incorporating dynamic obstacle avoidance within the Robot Operating System 2 (ROS2) and Navigation 2 (Nav2) framework. The proposed method utilizes Model Predictive Control (MPC) with a focus on handling the uncertainties associated with the movement prediction of dynamic obstacles. Unlike existing Nav2 trajectory planners which primarily deal with static obstacles or react to the current position of dynamic obstacles, this planner predicts future obstacle positions using a stochastic Vector Auto-Regressive Model (VAR). The obstacles' future positions are represented by probability distributions, and collision avoidance is achieved through constraints based on the Mahalanobis distance, ensuring the robot avoids regions where obstacles are likely to be. This approach considers the robot's kinodynamic constraints, enabling it to track a reference path while adapting to real-time changes in the environment. The paper details the implementation, including obstacle prediction, tracking, and the construction of feasible sets for MPC. Simulation results in a Gazebo environment demonstrate the effectiveness of this method in scenarios where robots must navigate around each other, showing improved collision avoidance capabilities. 

**Abstract (ZH)**: 基于ROS2和Nav2框架的具有动态障碍物避障功能的新型轨迹规划器研究 

---
# Advanced Longitudinal Control and Collision Avoidance for High-Risk Edge Cases in Autonomous Driving 

**Title (ZH)**: 高级纵向控制与碰撞避免技术在自主驾驶高风险边缘情况中的应用 

**Authors**: Dianwei Chen, Yaobang Gong, Xianfeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.18931)  

**Abstract**: Advanced Driver Assistance Systems (ADAS) and Advanced Driving Systems (ADS) are key to improving road safety, yet most existing implementations focus primarily on the vehicle ahead, neglecting the behavior of following vehicles. This shortfall often leads to chain reaction collisions in high speed, densely spaced traffic particularly when a middle vehicle suddenly brakes and trailing vehicles cannot respond in time. To address this critical gap, we propose a novel longitudinal control and collision avoidance algorithm that integrates adaptive cruising with emergency braking. Leveraging deep reinforcement learning, our method simultaneously accounts for both leading and following vehicles. Through a data preprocessing framework that calibrates real-world sensor data, we enhance the robustness and reliability of the training process, ensuring the learned policy can handle diverse driving conditions. In simulated high risk scenarios (e.g., emergency braking in dense traffic), the algorithm effectively prevents potential pile up collisions, even in situations involving heavy duty vehicles. Furthermore, in typical highway scenarios where three vehicles decelerate, the proposed DRL approach achieves a 99% success rate far surpassing the standard Federal Highway Administration speed concepts guide, which reaches only 36.77% success under the same conditions. 

**Abstract (ZH)**: 先进的驾驶辅助系统（ADAS）和高级驾驶系统（ADS）对于提高道路安全至关重要，但现有的大多数实施主要关注前方车辆，忽视了跟随车辆的行为。这一不足常常导致在高速、密集交通中发生连锁碰撞，尤其是当中间车辆突然制动而跟随车辆无法及时响应的情况下。为弥补这一关键不足，我们提出了一种新颖的纵向控制和碰撞避免算法，结合了自适应巡航控制和紧急制动。利用深度强化学习，该方法同时考虑了前方和跟随车辆。通过一个数据预处理框架对现实世界传感器数据进行校准，我们增强了训练过程的 robustness 和可靠性，确保学到的策略能够应对多种驾驶条件。在模拟的高风险场景（例如密集交通中的紧急制动）中，该算法有效防止了潜在的堆叠碰撞，即便是涉及重型车辆的情况也是如此。此外，在典型的高速公路上，当三辆车减速时，所提出的DRL方法达到了99%的成功率，远超美国联邦高速公路管理局（FHWA）的速度概念指南，在相同条件下仅达到36.77%的成功率。 

---
