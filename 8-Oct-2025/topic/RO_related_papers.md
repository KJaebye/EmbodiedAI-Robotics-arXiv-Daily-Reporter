# Towards Autonomous Tape Handling for Robotic Wound Redressing 

**Title (ZH)**: 面向机器人伤口换药的自主胶带处理技术 

**Authors**: Xiao Liang, Lu Shen, Peihan Zhang, Soofiyan Atar, Florian Richter, Michael Yip  

**Link**: [PDF](https://arxiv.org/pdf/2510.06127)  

**Abstract**: Chronic wounds, such as diabetic, pressure, and venous ulcers, affect over 6.5 million patients in the United States alone and generate an annual cost exceeding \$25 billion. Despite this burden, chronic wound care remains a routine yet manual process performed exclusively by trained clinicians due to its critical safety demands. We envision a future in which robotics and automation support wound care to lower costs and enhance patient outcomes. This paper introduces an autonomous framework for one of the most fundamental yet challenging subtasks in wound redressing: adhesive tape manipulation. Specifically, we address two critical capabilities: tape initial detachment (TID) and secure tape placement. To handle the complex adhesive dynamics of detachment, we propose a force-feedback imitation learning approach trained from human teleoperation demonstrations. For tape placement, we develop a numerical trajectory optimization method based to ensure smooth adhesion and wrinkle-free application across diverse anatomical surfaces. We validate these methods through extensive experiments, demonstrating reliable performance in both quantitative evaluations and integrated wound redressing pipelines. Our results establish tape manipulation as an essential step toward practical robotic wound care automation. 

**Abstract (ZH)**: 慢性伤口（如糖尿病足溃疡、压力性溃疡和静脉溃疡）在美国影响超过650万名患者，每年产生的成本超过250亿美元。尽管如此，慢性伤口护理仍然是由经过培训的临床医生手工完成的一项关键性流程。我们设想一个未来，在这个未来中，机器人和自动化技术将支持伤口护理，以降低成本并提升患者结局。本文介绍了一种自主框架，用于伤口换药中最基本但也最具挑战性的子任务之一：胶带操作。具体而言，我们解决两个关键能力：初始剥离（TID）和安全胶带定位。为处理复杂的剥离力学过程，我们提出了一种基于人类远程操作示范的力反馈模仿学习方法。对于胶带定位，我们开发了一种基于数值轨迹优化的方法，以确保在各种解剖表面中实现平滑粘合和无皱纹应用。我们通过广泛的实验验证了这些方法，在定量评估和集成的伤口换药流程中展现出可靠的表现。我们的结果确立了胶带操作是实现实际的机器人伤口护理自动化的关键步骤。 

---
# Multi-Robot Distributed Optimization for Exploration and Mapping of Unknown Environments using Bioinspired Tactile-Sensor 

**Title (ZH)**: 基于生物启发的触觉传感器的未知环境探索与建图多机器人分布式优化 

**Authors**: Roman Ibrahimov, Jannik Matthias Heinen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06085)  

**Abstract**: This project proposes a bioinspired multi-robot system using Distributed Optimization for efficient exploration and mapping of unknown environments. Each robot explores its environment and creates a map, which is afterwards put together to form a global 2D map of the environment. Inspired by wall-following behaviors, each robot autonomously explores its neighborhood based on a tactile sensor, similar to the antenna of a cockroach, mounted on the surface of the robot. Instead of avoiding obstacles, robots log collision points when they touch obstacles. This decentralized control strategy ensures effective task allocation and efficient exploration of unknown terrains, with applications in search and rescue, industrial inspection, and environmental monitoring. The approach was validated through experiments using e-puck robots in a simulated 1.5 x 1.5 m environment with three obstacles. The results demonstrated the system's effectiveness in achieving high coverage, minimizing collisions, and constructing accurate 2D maps. 

**Abstract (ZH)**: 本项目提出一种基于分布式优化的生物启发多机器人系统，用于未知环境的有效探索与建图。每个机器人探索其环境并创建地图，随后将这些局部地图组合成环境的全局2D地图。受墙跟随行为的启发，每个机器人根据安装在其表面的类似蟑螂触须的触觉传感器自主探索其周围环境。机器人在碰到障碍物时记录碰撞点，而非避开障碍物。这种去中心化控制策略确保了有效任务分配和未知地形的高效探索，适用于搜救、工业检测和环境监测等领域。通过使用e-puck机器人在包含三个障碍物的1.5×1.5 m模拟环境中进行实验，验证了该方法的有效性，结果显示该系统在高覆盖率、最少碰撞和构建准确的地图方面表现出色。 

---
# Coordinate-Consistent Localization via Continuous-Time Calibration and Fusion of UWB and SLAM Observations 

**Title (ZH)**: 基于连续时间校准和UWB与SLAM观测融合的坐标一致定位方法 

**Authors**: Tien-Dat Nguyen, Thien-Minh Nguyen, Vinh-Hao Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.05992)  

**Abstract**: Onboard simultaneous localization and mapping (SLAM) methods are commonly used to provide accurate localization information for autonomous robots. However, the coordinate origin of SLAM estimate often resets for each run. On the other hand, UWB-based localization with fixed anchors can ensure a consistent coordinate reference across sessions; however, it requires an accurate assignment of the anchor nodes' coordinates. To this end, we propose a two-stage approach that calibrates and fuses UWB data and SLAM data to achieve coordinate-wise consistent and accurate localization in the same environment. In the first stage, we solve a continuous-time batch optimization problem by using the range and odometry data from one full run, incorporating height priors and anchor-to-anchor distance factors to recover the anchors' 3D positions. For the subsequent runs in the second stage, a sliding-window optimization scheme fuses the UWB and SLAM data, which facilitates accurate localization in the same coordinate system. Experiments are carried out on the NTU VIRAL dataset with six scenarios of UAV flight, and we show that calibration using data in one run is sufficient to enable accurate localization in the remaining runs. We release our source code to benefit the community at this https URL. 

**Abstract (ZH)**: 基于UWB和SLAM数据的两阶段校准与融合方法以实现环境内的坐标一致和准确定位 

---
# The DISTANT Design for Remote Transmission and Steering Systems for Planetary Robotics 

**Title (ZH)**: 远程传输与 steering 系统的远端设计：行星机器人应用 

**Authors**: Cristina Luna, Alba Guerra, Almudena Moreno, Manuel Esquer, Willy Roa, Mateusz Krawczak, Robert Popela, Piotr Osica, Davide Nicolis  

**Link**: [PDF](https://arxiv.org/pdf/2510.05981)  

**Abstract**: Planetary exploration missions require robust locomotion systems capable of operating in extreme environments over extended periods. This paper presents the DISTANT (Distant Transmission and Steering Systems) design, a novel approach for relocating rover traction and steering actuators from wheel-mounted positions to a thermally protected warm box within the rover body. The design addresses critical challenges in long-distance traversal missions by protecting sensitive components from thermal cycling, dust contamination, and mechanical wear. A double wishbone suspension configuration with cardan joints and capstan drive steering has been selected as the optimal architecture following comprehensive trade-off analysis. The system enables independent wheel traction, steering control, and suspension management whilst maintaining all motorisation within the protected environment. The design meets a 50 km traverse requirement without performance degradation, with integrated dust protection mechanisms and thermal management solutions. Testing and validation activities are planned for Q1 2026 following breadboard manufacturing at 1:3 scale. 

**Abstract (ZH)**: 行星探测任务需要能够在极端环境中长期运行的 robust 运动系统。本文介绍了 DISTANT（远距离传输与转向系统）设计，这是一种将轮毂安装的动力和转向执行器重新定位至 rove 基体内的热保护温箱内的新颖方法。该设计通过保护敏感组件免受热循环、灰尘污染和机械磨损的影响，解决了长距离探测任务中的关键挑战。经过全面的权衡分析后，选择了双臂悬挂配置，配备万向节和卷筒驱动转向机构，作为最佳架构。该系统实现了独立轮毂牵引、转向控制和悬挂管理，同时将所有电动化装置保留在受保护环境中。该设计满足了 50 km 的探测要求，性能无降级，并具备集成的灰尘防护机制和热管理解决方案。计划于 2026 年第一季度进行原型制造并进行测试与验证活动，比例为 1:3。 

---
# A Co-Design Framework for Energy-Aware Monoped Jumping with Detailed Actuator Modeling 

**Title (ZH)**: 一种考虑能量 Awareness 的单足跳跃协同设计框架及详细执行器建模 

**Authors**: Aman Singh, Aastha Mishra, Deepak Kapa, Suryank Joshi, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2510.05923)  

**Abstract**: A monoped's jump height and energy consumption depend on both, its mechanical design and control strategy. Existing co-design frameworks typically optimize for either maximum height or minimum energy, neglecting their trade-off. They also often omit gearbox parameter optimization and use oversimplified actuator mass models, producing designs difficult to replicate in practice. In this work, we introduce a novel three-stage co-design optimization framework that jointly maximizes jump height while minimizing mechanical energy consumption of a monoped. The proposed method explicitly incorporates realistic actuator mass models and optimizes mechanical design (including gearbox) and control parameters within a unified framework. The resulting design outputs are then used to automatically generate a parameterized CAD model suitable for direct fabrication, significantly reducing manual design iterations. Our experimental evaluations demonstrate a 50 percent reduction in mechanical energy consumption compared to the baseline design, while achieving a jump height of 0.8m. Video presentation is available at this http URL 

**Abstract (ZH)**: 单足跳高高度和能量消耗取决于其机械设计和控制策略的设计与控制策略。现有的联合设计框架通常仅优化其中之一（最大高度或最小能量），而忽略了它们之间的权衡。此外，这些框架通常不进行齿轮箱参数优化，并使用过于简化的执行器质量模型，导致难以实际复制的设计。在本工作中，我们引入了一种新颖的三阶段联合设计优化框架，该框架在统一框架中同时最大化跳跃高度并最小化机械能量消耗。所提出的该方法明确地纳入了现实的执行器质量模型，并优化了机械设计（包括齿轮箱）和控制参数。生成的设计输出可用于自动生成适合直接制造的参数化CAD模型，显著减少了手动设计迭代。我们的实验评估表明，与基准设计相比，机械能量消耗减少了50%，同时实现了0.8米的跳跃高度。视频演示可在以下链接获取：this http URL。 

---
# GO-Flock: Goal-Oriented Flocking in 3D Unknown Environments with Depth Maps 

**Title (ZH)**: GO- flock: 目标导向的三维未知环境中的集群导航方法研究（基于深度图） 

**Authors**: Yan Rui Tan, Wenqi Liu, Wai Lun Leong, John Guan Zhong Tan, Wayne Wen Huei Yong, Fan Shi, Rodney Swee Huat Teo  

**Link**: [PDF](https://arxiv.org/pdf/2510.05553)  

**Abstract**: Artificial Potential Field (APF) methods are widely used for reactive flocking control, but they often suffer from challenges such as deadlocks and local minima, especially in the presence of obstacles. Existing solutions to address these issues are typically passive, leading to slow and inefficient collective navigation. As a result, many APF approaches have only been validated in obstacle-free environments or simplified, pseudo 3D simulations. This paper presents GO-Flock, a hybrid flocking framework that integrates planning with reactive APF-based control. GO-Flock consists of an upstream Perception Module, which processes depth maps to extract waypoints and virtual agents for obstacle avoidance, and a downstream Collective Navigation Module, which applies a novel APF strategy to achieve effective flocking behavior in cluttered environments. We evaluate GO-Flock against passive APF-based approaches to demonstrate their respective merits, such as their flocking behavior and the ability to overcome local minima. Finally, we validate GO-Flock through obstacle-filled environment and also hardware-in-the-loop experiments where we successfully flocked a team of nine drones, six physical and three virtual, in a forest environment. 

**Abstract (ZH)**: 基于人工势场的GO-Flock混合群集框架：规划与反应控制的集成 

---
# Correlation-Aware Dual-View Pose and Velocity Estimation for Dynamic Robotic Manipulation 

**Title (ZH)**: 关联感知的双视图姿态与速度估计用于动态机器人操作 

**Authors**: Mahboubeh Zarei, Robin Chhabra, Farrokh Janabi-Sharifi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05536)  

**Abstract**: Accurate pose and velocity estimation is essential for effective spatial task planning in robotic manipulators. While centralized sensor fusion has traditionally been used to improve pose estimation accuracy, this paper presents a novel decentralized fusion approach to estimate both pose and velocity. We use dual-view measurements from an eye-in-hand and an eye-to-hand vision sensor configuration mounted on a manipulator to track a target object whose motion is modeled as random walk (stochastic acceleration model). The robot runs two independent adaptive extended Kalman filters formulated on a matrix Lie group, developed as part of this work. These filters predict poses and velocities on the manifold $\mathbb{SE}(3) \times \mathbb{R}^3 \times \mathbb{R}^3$ and update the state on the manifold $\mathbb{SE}(3)$. The final fused state comprising the fused pose and velocities of the target is obtained using a correlation-aware fusion rule on Lie groups. The proposed method is evaluated on a UFactory xArm 850 equipped with Intel RealSense cameras, tracking a moving target. Experimental results validate the effectiveness and robustness of the proposed decentralized dual-view estimation framework, showing consistent improvements over state-of-the-art methods. 

**Abstract (ZH)**: 精确的姿态和速度估计对于 robotic manipulators 中的有效空间任务规划至关重要。虽然集中式传感器融合传统上用于提高姿态估计准确性，本文提出了一种新颖的分布式融合方法来同时估计姿态和速度。我们使用 manipulator 上安装的 hand-in-hand 和 hand-to-eye 视觉传感器配置的双视图测量来跟踪其运动模型为随机游走（随机加速度模型）的目标物体。机器人运行两个独立的适应扩展卡尔曼滤波器，这些滤波器是在矩阵李群上开发的，用于预测姿态和速度在流形 \(\mathbb{SE}(3) \times \mathbb{R}^3 \times \mathbb{R}^3\) 上，并在流形 \(\mathbb{SE}(3)\) 上更新状态。最终融合状态，包括目标的融合姿态和速度，是通过李群上的相关性感知融合规则获得的。所提出的方法在配备了 Intel RealSense 相机的 UFactory xArm 850 上进行了评估，跟踪移动目标。实验结果验证了所提出的分布式双视图估计框架的有效性和稳健性，显示出相对于最先进的方法的一致改进。 

---
# AD-NODE: Adaptive Dynamics Learning with Neural ODEs for Mobile Robots Control 

**Title (ZH)**: AD-NODE: 基于神经ODE的自适应动力学习在移动机器人控制中的应用 

**Authors**: Shao-Yi Yu, Jen-Wei Wang, Maya Horii, Vikas Garg, Tarek Zohdi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05443)  

**Abstract**: Mobile robots, such as ground vehicles and quadrotors, are becoming increasingly important in various fields, from logistics to agriculture, where they automate processes in environments that are difficult to access for humans. However, to perform effectively in uncertain environments using model-based controllers, these systems require dynamics models capable of responding to environmental variations, especially when direct access to environmental information is limited. To enable such adaptivity and facilitate integration with model predictive control, we propose an adaptive dynamics model which bypasses the need for direct environmental knowledge by inferring operational environments from state-action history. The dynamics model is based on neural ordinary equations, and a two-phase training procedure is used to learn latent environment representations. We demonstrate the effectiveness of our approach through goal-reaching and path-tracking tasks on three robotic platforms of increasing complexity: a 2D differential wheeled robot with changing wheel contact conditions, a 3D quadrotor in variational wind fields, and the Sphero BOLT robot under two contact conditions for real-world deployment. Empirical results corroborate that our method can handle temporally and spatially varying environmental changes in both simulation and real-world systems. 

**Abstract (ZH)**: 基于动态模型的适应性移动机器人在不确定环境中的应用研究：从轮式地面机器人到四旋翼无人机的实证分析 

---
# Towards Online Robot Interaction Adaptation to Human Upper-limb Mobility Impairments in Return-to-Work Scenarios 

**Title (ZH)**: 面向工作回归场景中上肢 mobility 状况受损的人机互动适应性在线调整 

**Authors**: Marta Lagomarsino, Francesco Tassi  

**Link**: [PDF](https://arxiv.org/pdf/2510.05425)  

**Abstract**: Work environments are often inadequate and lack inclusivity for individuals with upper-body disabilities. This paper presents a novel online framework for adaptive human-robot interaction (HRI) that accommodates users' arm mobility impairments, ultimately aiming to promote active work participation. Unlike traditional human-robot collaboration approaches that assume able-bodied users, our method integrates a mobility model for specific joint limitations into a hierarchical optimal controller. This allows the robot to generate reactive, mobility-aware behaviour online and guides the user's impaired limb to exploit residual functional mobility. The framework was tested in handover tasks involving different upper-limb mobility impairments (i.e., emulated elbow and shoulder arthritis, and wrist blockage), under both standing and seated configurations with task constraints using a mobile manipulator, and complemented by quantitative and qualitative comparisons with state-of-the-art ergonomic HRI approaches. Preliminary results indicated that the framework can personalise the interaction to fit within the user's impaired range of motion and encourage joint usage based on the severity of their functional limitations. 

**Abstract (ZH)**: 上肢残疾人士的工作环境往往缺乏包容性和适应性。本文提出了一种新颖的在线框架，用于适应性的人机交互（HRI），以适应用户的上肢活动能力障碍，最终目标是促进积极参与工作。与传统的假设健全用户的机器人协作方法不同，我们的方法将特定关节限制的移动模型整合到分层优化控制器中。这使得机器人能够在线生成移动意识的反应行为，并引导用户的受损肢体利用剩余的功能活动能力。该框架在使用移动操作臂进行不同上肢活动能力障碍的手递任务（例如模拟肘关节和肩关节关节炎以及腕部阻塞）测试中，考虑站立和坐姿配置下的任务约束，并通过与最新的人机工程学HRI方法的量化和定性比较进行补充。初步结果表明，该框架可以个性化交互，以适应用户的受损活动范围，并基于其功能限制的严重程度鼓励关节使用。 

---
# A multi-modal tactile fingertip design for robotic hands to enhance dexterous manipulation 

**Title (ZH)**: 用于增强灵巧操作的多模态触觉指尖设计 

**Authors**: Zhuowei Xu, Zilin Si, Kevin Zhang, Oliver Kroemer, Zeynep Temel  

**Link**: [PDF](https://arxiv.org/pdf/2510.05382)  

**Abstract**: Tactile sensing holds great promise for enhancing manipulation precision and versatility, but its adoption in robotic hands remains limited due to high sensor costs, manufacturing and integration challenges, and difficulties in extracting expressive and reliable information from signals. In this work, we present a low-cost, easy-to-make, adaptable, and compact fingertip design for robotic hands that integrates multi-modal tactile sensors. We use strain gauge sensors to capture static forces and a contact microphone sensor to measure high-frequency vibrations during contact. These tactile sensors are integrated into a compact design with a minimal sensor footprint, and all sensors are internal to the fingertip and therefore not susceptible to direct wear and tear from interactions. From sensor characterization, we show that strain gauge sensors provide repeatable 2D planar force measurements in the 0-5 N range and the contact microphone sensor has the capability to distinguish contact material properties. We apply our design to three dexterous manipulation tasks that range from zero to full visual occlusion. Given the expressiveness and reliability of tactile sensor readings, we show that different tactile sensing modalities can be used flexibly in different stages of manipulation, solely or together with visual observations to achieve improved task performance. For instance, we can precisely count and unstack a desired number of paper cups from a stack with 100\% success rate which is hard to achieve with vision only. 

**Abstract (ZH)**: 触觉传感对于提高操作精度和灵活性具有巨大潜力，但由于传感器成本高、制造和集成挑战以及从信号中提取丰富可靠信息的困难，其在机器人手中的应用仍然有限。在此项工作中，我们提出了一种低成本、易制作、可适应且紧凑的手指尖设计，该设计集成了多模态触觉传感器。我们使用应变片传感器捕获静态力，并使用接触麦克风传感器测量接触过程中的高频振动。这些触觉传感器通过紧凑的设计集成，具有最小的传感器占地面积，且所有传感器都内置在指尖内部，因此不易受到交互过程中直接磨损和损坏的影响。通过传感器标定，我们展示了应变片传感器在0-5 N范围内提供重复的二维平面力测量，接触麦克风传感器具有区分接触材料属性的能力。我们将该设计应用于三种不同视觉遮挡程度的灵巧操作任务。鉴于触觉传感器读数的表达性和可靠性，我们展示了不同的触觉传感模态可以在操作的不同阶段灵活使用，单独或与其他视觉观察结合，以实现更好的任务性能。例如，我们能够在100%的成功率下精确计数并逐个移除一个纸杯堆中的所需数量的纸杯，仅凭视觉控制难以实现这一点。 

---
