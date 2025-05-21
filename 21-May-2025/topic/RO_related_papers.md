# Traversability-aware path planning in dynamic environments 

**Title (ZH)**: 动态环境中的可通行路径规划 

**Authors**: Yaroslav Marchukov, Luis Montano  

**Link**: [PDF](https://arxiv.org/pdf/2505.14580)  

**Abstract**: Planning in environments with moving obstacles remains a significant challenge in robotics. While many works focus on navigation and path planning in obstacle-dense spaces, traversing such congested regions is often avoidable by selecting alternative routes. This paper presents Traversability-aware FMM (Tr-FMM), a path planning method that computes paths in dynamic environments, avoiding crowded regions. The method operates in two steps: first, it discretizes the environment, identifying regions and their distribution; second, it computes the traversability of regions, aiming to minimize both obstacle risks and goal deviation. The path is then computed by propagating the wavefront through regions with higher traversability. Simulated and real-world experiments demonstrate that the approach enhances significant safety by keeping the robot away from regions with obstacles while reducing unnecessary deviations from the goal. 

**Abstract (ZH)**: 在动态环境中的通达性意识FMM路径规划方法 

---
# NavBench: A Unified Robotics Benchmark for Reinforcement Learning-Based Autonomous Navigation 

**Title (ZH)**: NavBench: 基于强化学习的自主导航统一机器人基准测试 

**Authors**: Matteo El-Hariry, Antoine Richard, Ricard M. Castan, Luis F. W. Batista, Matthieu Geist, Cedric Pradalier, Miguel Olivares-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2505.14526)  

**Abstract**: Autonomous robots must navigate and operate in diverse environments, from terrestrial and aquatic settings to aerial and space domains. While Reinforcement Learning (RL) has shown promise in training policies for specific autonomous robots, existing benchmarks are often constrained to unique platforms, limiting generalization and fair comparisons across different mobility systems. In this paper, we present NavBench, a multi-domain benchmark for training and evaluating RL-based navigation policies across diverse robotic platforms and operational environments. Built on IsaacLab, our framework standardizes task definitions, enabling different robots to tackle various navigation challenges without the need for ad-hoc task redesigns or custom evaluation metrics. Our benchmark addresses three key challenges: (1) Unified cross-medium benchmarking, enabling direct evaluation of diverse actuation methods (thrusters, wheels, water-based propulsion) in realistic environments; (2) Scalable and modular design, facilitating seamless robot-task interchangeability and reproducible training pipelines; and (3) Robust sim-to-real validation, demonstrated through successful policy transfer to multiple real-world robots, including a satellite robotic simulator, an unmanned surface vessel, and a wheeled ground vehicle. By ensuring consistency between simulation and real-world deployment, NavBench simplifies the development of adaptable RL-based navigation strategies. Its modular design allows researchers to easily integrate custom robots and tasks by following the framework's predefined templates, making it accessible for a wide range of applications. Our code is publicly available at NavBench. 

**Abstract (ZH)**: 自主机器人必须在多样化的环境中导航和运行，从陆地和水下环境到空中和太空领域。尽管强化学习（RL）在训练特定自主机器人的策略方面显示出潜力，但现有的基准测试通常局限于独特的平台，限制了不同移动系统之间的泛化和公平比较。在本文中，我们提出了NavBench，这是一个多领域的基准测试，用于跨多样化的机器人平台和运行环境训练和评估基于RL的导航策略。基于IsaacLab，我们的框架标准化了任务定义，使得不同的机器人能够应对各种导航挑战，无需进行特定任务的重新设计或自定义评估指标。我们的基准测试解决了三个关键挑战：（1）统一的跨介质基准测试，允许直接评估多种不同的执行方法（喷水推进、车轮、水基推进）在真实环境中的表现；（2）可扩展和模块化的设计，方便机器人和任务的无缝互换及可再现的训练管道；（3）稳健的仿真到现实的验证，通过多个实际机器人的成功策略转移得到验证，包括卫星机器人模拟器、无人表面船舶和轮式地面车辆。通过确保模拟与实际部署之间的一致性，NavBench 简化了适应性强的基于RL的导航策略的开发。其模块化设计允许研究人员通过遵循框架预定义的模板轻松集成自定义机器人和任务，使其适用于广泛的用途。我们的代码可在NavBench公开获取。 

---
# Robust Immersive Bilateral Teleoperation of Dissimilar Systems with Enhanced Transparency and Sense of Embodiment 

**Title (ZH)**: 增强透明度和身临其境感的鲁棒异构系统双工远程操作 

**Authors**: Mahdi Hejrati, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2505.14486)  

**Abstract**: In human-in-the-loop systems such as teleoperation, especially those involving heavy-duty manipulators, achieving high task performance requires both robust control and strong human engagement. This paper presents a bilateral teleoperation framework that enhances the operator's Sense of Embodiment (SoE), specifically, the senses of agency and self-location, through an immersive virtual reality interface and distributed haptic feedback via an exoskeleton. To support this embodiment and stablish high level of motion and force transparency, we develop a force-sensorless, robust control architecture that tackles input nonlinearities, master-slave asymmetries, unknown uncertainties, and arbitrary time delays. A human-robot augmented dynamic model is integrated into the control loop to enhance human-adaptability of the controller. Theoretical analysis confirms semi-global uniform ultimate boundedness of the closed-loop system. Extensive real-world experiments demonstrate high accuracy tracking under up to 1:13 motion scaling and 1:1000 force scaling, showcasing the significance of the results. Additionally, the stability-transparency tradeoff for motion tracking and force reflection-tracking is establish up to 150 ms of one-way fix and time-varying communication delay. The results of user study with 10 participants (9 male and 1 female) demonstrated that the system can imply a good level of SoE (76.4%), at the same time is very user friendly with no gender limitation. These results are significant given the scale and weight of the heavy-duty manipulators. 

**Abstract (ZH)**: 基于人类在环的双工虚拟现实遥操作框架：重载 manipulator 遂行任务的感身性增强与透明控制 

---
# Local Minima Prediction using Dynamic Bayesian Filtering for UGV Navigation in Unstructured Environments 

**Title (ZH)**: 基于动态贝叶斯过滤的UGV在非结构化环境导航中局部极小值预测 

**Authors**: Seung Hun Lee, Wonse Jo, Lionel P. Robert Jr., Dawn M. Tilbury  

**Link**: [PDF](https://arxiv.org/pdf/2505.14337)  

**Abstract**: Path planning is crucial for the navigation of autonomous vehicles, yet these vehicles face challenges in complex and real-world environments. Although a global view may be provided, it is often outdated, necessitating the reliance of Unmanned Ground Vehicles (UGVs) on real-time local information. This reliance on partial information, without considering the global context, can lead to UGVs getting stuck in local minima. This paper develops a method to proactively predict local minima using Dynamic Bayesian filtering, based on the detected obstacles in the local view and the global goal. This approach aims to enhance the autonomous navigation of self-driving vehicles by allowing them to predict potential pitfalls before they get stuck, and either ask for help from a human, or re-plan an alternate trajectory. 

**Abstract (ZH)**: 基于局部视图和全局目标的动态贝叶斯滤波局部极小值主动预测方法 

---
# Sketch Interface for Teleoperation of Mobile Manipulator to Enable Intuitive and Intended Operation: A Proof of Concept 

**Title (ZH)**: 基于素描界面的移动 manipulator 远程操作概念验证：实现直观并意想中的操作 

**Authors**: Yuka Iwanaga, Masayoshi Tsuchinaga, Kosei Tanada, Yuji Nakamura, Takemitsu Mori, Takashi Yamamoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.13931)  

**Abstract**: Recent advancements in robotics have underscored the need for effective collaboration between humans and robots. Traditional interfaces often struggle to balance robot autonomy with human oversight, limiting their practical application in complex tasks like mobile manipulation. This study aims to develop an intuitive interface that enables a mobile manipulator to autonomously interpret user-provided sketches, enhancing user experience while minimizing burden. We implemented a web-based application utilizing machine learning algorithms to process sketches, making the interface accessible on mobile devices for use anytime, anywhere, by anyone. In the first validation, we examined natural sketches drawn by users for 27 selected manipulation and navigation tasks, gaining insights into trends related to sketch instructions. The second validation involved comparative experiments with five grasping tasks, showing that the sketch interface reduces workload and enhances intuitiveness compared to conventional axis control interfaces. These findings suggest that the proposed sketch interface improves the efficiency of mobile manipulators and opens new avenues for integrating intuitive human-robot collaboration in various applications. 

**Abstract (ZH)**: 近期机器人技术的进步强调了人类与机器人有效协作的必要性。传统的接口往往难以平衡机器人自主性和人的监督，限制了它们在诸如移动操作这类复杂任务中的实际应用。本研究旨在开发一种直观的界面，使移动操作器能够自主解释用户提供的草图，从而改善用户体验并减轻负担。我们利用机器学习算法开发了一个基于网页的应用程序，使接口能够在移动设备上使用，方便任何人、任何时间和地点地访问。首次验证中，我们分析了用户为27项选定的操作和导航任务绘制的自然草图，以了解与草图指示相关的发展趋势。第二次验证通过五项抓取任务的比较实验表明，草图界面与传统轴控制界面相比可以减轻工作负担并提高直观性。这些发现表明，所提出的草图界面可以提高移动操作器的效率，并为各种应用中的人机协作开辟新的途径。 

---
# Robotic Monitoring of Colorimetric Leaf Sensors for Precision Agriculture 

**Title (ZH)**: 机器人监测色谱叶传感器进行精准农业 

**Authors**: Malakhi Hopkins, Alice Kate Li, Shobhita Kramadhati, Jackson Arnold, Akhila Mallavarapu, Chavez Lawrence, Varun Murali, Sanjeev J. Koppal, Cherie Kagan, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.13916)  

**Abstract**: Current remote sensing technologies that measure crop health e.g. RGB, multispectral, hyperspectral, and LiDAR, are indirect, and cannot capture plant stress indicators directly. Instead, low-cost leaf sensors that directly interface with the crop surface present an opportunity to advance real-time direct monitoring. To this end, we co-design a sensor-detector system, where the sensor is a novel colorimetric leaf sensor that directly measures crop health in a precision agriculture setting, and the detector autonomously obtains optical signals from these leaf sensors. This system integrates a ground robot platform with an on-board monocular RGB camera and object detector to localize the leaf sensor, and a hyperspectral camera with motorized mirror and an on-board halogen light to acquire a hyperspectral reflectance image of the leaf sensor, from which a spectral response characterizing crop health can be extracted. We show a successful demonstration of our co-designed system operating in outdoor environments, obtaining spectra that are interpretable when compared to controlled laboratory-grade spectrometer measurements. The system is demonstrated in row-crop environments both indoors and outdoors where it is able to autonomously navigate, locate and obtain a hyperspectral image of all leaf sensors present, and retrieve interpretable spectral resonance from leaf sensors. 

**Abstract (ZH)**: 当前用于测量农作物健康的遥感技术，如RGB、多光谱、高光谱和LiDAR，都是间接的，无法直接捕捉植物压力指标。相比之下，低成本叶传感器可以直接与作物表面接口，为实现即时直接监控提供了机会。为此，我们共同设计了一个传感器-检测系统，其中传感器是一种新型的比色叶传感器，可在精确农业环境中直接测量作物健康状况，而检测器则自主获取这些叶传感器的光学信号。该系统集成了地面机器人平台及上装单目RGB相机和目标检测器以定位叶传感器，以及装有电动反光镜的高光谱相机和上装卤素灯，以获取叶传感器的高光谱反射图像，从而提取表征作物健康的光谱响应。我们展示了该共同设计系统在户外环境中的成功演示，所获得的光谱与受控实验室级光谱仪测量结果可比对解析。该系统在室内和室外的行作物环境中展示出自主导航、定位和获取所有叶传感器高光谱图像的能力，并从叶传感器中检索出可解释的光谱共振。 

---
# Certifiably Safe Manipulation of Deformable Linear Objects via Joint Shape and Tension Prediction 

**Title (ZH)**: 基于联合形状和张力预测的可验证安全可变形线性对象操作 

**Authors**: Yiting Zhang, Shichen Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13889)  

**Abstract**: Manipulating deformable linear objects (DLOs) is challenging due to their complex dynamics and the need for safe interaction in contact-rich environments. Most existing models focus on shape prediction alone and fail to account for contact and tension constraints, which can lead to damage to both the DLO and the robot. In this work, we propose a certifiably safe motion planning and control framework for DLO manipulation. At the core of our method is a predictive model that jointly estimates the DLO's future shape and tension. These predictions are integrated into a real-time trajectory optimizer based on polynomial zonotopes, allowing us to enforce safety constraints throughout the execution. We evaluate our framework on a simulated wire harness assembly task using a 7-DOF robotic arm. Compared to state-of-the-art methods, our approach achieves a higher task success rate while avoiding all safety violations. The results demonstrate that our method enables robust and safe DLO manipulation in contact-rich environments. 

**Abstract (ZH)**: 操纵变形线型对象（DLOs）因其实复杂动态和在接触丰富环境中的安全交互需求而具挑战性。现有的大多数模型仅专注于形状预测，未能考虑接触和张力约束，这可能导致DLO和机器人本身受损。在这项工作中，我们提出了一种可验证安全的运动规划与控制框架，用于DLO manipulation。我们方法的核心是一个预测模型，可以联合估计DLO的未来形状和张力。这些预测被集成到基于多项式zonotopes的实时轨迹优化器中，从而使我们在执行过程中能够强制执行安全约束。我们使用7自由度机械臂在模拟的线束组装任务中评估了该框架。与现有最先进的方法相比，我们的方法在避免所有安全违规的情况下实现了更高的任务成功率。结果表明，我们的方法能够在接触丰富环境中实现稳健且安全的DLO manipulation。 

---
# Enhancing Robot Navigation Policies with Task-Specific Uncertainty Managements 

**Title (ZH)**: 基于任务特定不确定性管理的机器人导航策略增强 

**Authors**: Gokul Puthumanaillam, Paulo Padrao, Jose Fuentes, Leonardo Bobadilla, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2505.13837)  

**Abstract**: Robots navigating complex environments must manage uncertainty from sensor noise, environmental changes, and incomplete information, with different tasks requiring varying levels of precision in different areas. For example, precise localization may be crucial near obstacles but less critical in open spaces. We present GUIDE (Generalized Uncertainty Integration for Decision-Making and Execution), a framework that integrates these task-specific requirements into navigation policies via Task-Specific Uncertainty Maps (TSUMs). By assigning acceptable uncertainty levels to different locations, TSUMs enable robots to adapt uncertainty management based on context. When combined with reinforcement learning, GUIDE learns policies that balance task completion and uncertainty management without extensive reward engineering. Real-world tests show significant performance gains over methods lacking task-specific uncertainty awareness. 

**Abstract (ZH)**: 面向复杂环境导航的机器人必须管理来自传感器噪声、环境变化和信息不完全性的不确定性，不同的任务对不同区域的精度要求各不相同。我们提出了GUIDE（Generalized Uncertainty Integration for Decision-Making and Execution）框架，通过任务特定不确定性地图（TSUMs）将这些任务特定要求整合到导航策略中。通过为不同位置分配可接受的不确定性水平，TSUMs使机器人能够根据上下文调整不确定性管理。结合强化学习时，GUIDE能够学习平衡任务完成和不确定性管理的策略，无需大量奖励工程。实验证明，与缺乏任务特定不确定性意识的方法相比，GUIDE显示出显著的性能提升。 

---
# Duawlfin: A Drone with Unified Actuation for Wheeled Locomotion and Flight Operation 

**Title (ZH)**: Duawlfin：一种兼具轮式移动和飞行操作统一驱动的无人机 

**Authors**: Jerry Tang, Ruiqi Zhang, Kaan Beyduz, Yiwei Jiang, Cody Wiebe, Haoyu Zhang, Osaruese Asoro, Mark W. Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2505.13836)  

**Abstract**: This paper presents Duawlfin, a drone with unified actuation for wheeled locomotion and flight operation that achieves efficient, bidirectional ground mobility. Unlike existing hybrid designs, Duawlfin eliminates the need for additional actuators or propeller-driven ground propulsion by leveraging only its standard quadrotor motors and introducing a differential drivetrain with one-way bearings. This innovation simplifies the mechanical system, significantly reduces energy usage, and prevents the disturbance caused by propellers spinning near the ground, such as dust interference with sensors. Besides, the one-way bearings minimize the power transfer from motors to propellers in the ground mode, which enables the vehicle to operate safely near humans. We provide a detailed mechanical design, present control strategies for rapid and smooth mode transitions, and validate the concept through extensive experimental testing. Flight-mode tests confirm stable aerial performance comparable to conventional quadcopters, while ground-mode experiments demonstrate efficient slope climbing (up to 30°) and agile turning maneuvers approaching 1g lateral acceleration. The seamless transitions between aerial and ground modes further underscore the practicality and effectiveness of our approach for applications like urban logistics and indoor navigation. All the materials including 3-D model files, demonstration video and other assets are open-sourced at this https URL. 

**Abstract (ZH)**: 本文介绍了Duawlfin，一种集成了轮式运动和飞行操作一体驱动的无人机，实现了高效、双向地面移动。与现有的混合设计不同，Duawlfin 通过仅利用标准四旋翼电机并引入差动驱动系统结合单向轴承，消除了额外驱动器或螺旋桨驱动地面推进的需要。这一创新简化了机械系统，显著降低了能耗，并防止了螺旋桨接近地面时对传感器造成的干扰。此外，单向轴承最大限度地减少了地面模式下电机到螺旋桨的动力传输，使得车辆能在接近人类时安全运行。本文提供了详细的机械设计，展示了快速平滑模式转换的控制策略，并通过广泛的实验测试验证了该概念。飞行模式测试确认了其空中性能与传统四旋翼机相当，而地面模式实验则展示了其高效的坡度爬升（坡度达30°）和接近1g侧向加速度的敏捷转弯操作。无缝的空地模式转换进一步凸显了该方法在城市物流和室内导航等应用中的实用性和有效性。所有相关材料，包括3D模型文件、演示视频和其他资源均在此链接处开源：https://github.com/alibaba/Duawlfin。 

---
# From Structural Design to Dynamics Modeling: Control-Oriented Development of a 3-RRR Parallel Ankle Rehabilitation Robot 

**Title (ZH)**: 从结构设计到动力学建模：面向控制的3-RRR并联踝 rehabilitation机器人开发 

**Authors**: Siyuan Zhang, Yufei Zhang, Junlin Lyu, Sunil K. Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2505.13762)  

**Abstract**: This paper presents the development of a wearable ankle rehabilitation robot based on a 3-RRR spherical parallel mechanism (SPM) to support multi-DOF recovery through pitch, roll, and yaw motions. The system features a compact, ergonomic structure designed for comfort, safety, and compatibility with ankle biomechanics. A complete design-to-dynamics pipeline has been implemented, including structural design, kinematic modeling for motion planning, and Lagrangian-based dynamic modeling for torque estimation and simulation analysis. Preliminary simulations verify stable joint coordination and smooth motion tracking under representative rehabilitation trajectories. The control framework is currently being developed to enhance responsiveness across the workspace. Future work will focus on integrating personalized modeling and adaptive strategies to address kinematic singularities through model based control. This work establishes a foundational platform for intelligent, personalized ankle rehabilitation, enabling both static training and potential extension to gait-phase-timed assistance. 

**Abstract (ZH)**: 基于3-RRR球面平行机构的可穿戴踝关节康复机器人开发与研究 

---
# Practice Makes Perfect: A Study of Digital Twin Technology for Assembly and Problem-solving using Lunar Surface Telerobotics 

**Title (ZH)**: 熟能生巧：基于月球表面远程机器人装配及问题解决的数字孪生技术研究 

**Authors**: Xavier O'Keefe, Katy McCutchan, Alexis Muniz, Jack Burns, Daniel Szafir  

**Link**: [PDF](https://arxiv.org/pdf/2505.13722)  

**Abstract**: Robotic systems that can traverse planetary or lunar surfaces to collect environmental data and perform physical manipulation tasks, such as assembling equipment or conducting mining operations, are envisioned to form the backbone of future human activities in space. However, the environmental conditions in which these robots, or "rovers," operate present challenges toward achieving fully autonomous solutions, meaning that rover missions will require some degree of human teleoperation or supervision for the foreseeable future. As a result, human operators require training to successfully direct rovers and avoid costly errors or mission failures, as well as the ability to recover from any issues that arise on the fly during mission activities. While analog environments, such as JPL's Mars Yard, can help with such training by simulating surface environments in the real world, access to such resources may be rare and expensive. As an alternative or supplement to such physical analogs, we explore the design and evaluation of a virtual reality digital twin system to train human teleoperation of robotic rovers with mechanical arms for space mission activities. We conducted an experiment with 24 human operators to investigate how our digital twin system can support human teleoperation of rovers in both pre-mission training and in real-time problem solving in a mock lunar mission in which users directed a physical rover in the context of deploying dipole radio antennas. We found that operators who first trained with the digital twin showed a 28% decrease in mission completion time, an 85% decrease in unrecoverable errors, as well as improved mental markers, including decreased cognitive load and increased situation awareness. 

**Abstract (ZH)**: 能在行星或月球表面行驶以采集环境数据并执行物理操作任务（如设备装配或采矿作业）的机器人系统，预计将构成未来太空活动中枢。然而，这些机器人或“漫游者”在运行时所处的环境条件对其实现完全自主解决方案提出挑战，这意味着可预见的未来漫游者任务仍需要一定程度的人类遥控操作或监督。因此，操作人员需要接受培训，以成功指挥漫游者并避免昂贵的错误或任务失败，并具备在任务活动中应对任何突发问题的能力。虽然像喷气推进实验室的火星 yard 这样的模拟环境有助于此类培训，但获取这些资源可能稀缺且昂贵。作为一种替代或补充，我们探讨了设计和评估一种虚拟现实数字孪生系统，以培训人类遥控机械臂漫游者进行太空任务操作。我们在模拟月球任务中进行了实验，让24名操作者在使用数字孪生系统进行预任务培训和实时问题解决时指导一个物理漫游者部署磁偶极天线。我们发现，使用数字孪生系统进行培训的操作者在任务完成时间上减少了28%，不可恢复错误减少了85%，同时还表现出认知负荷降低、情况意识提高等心理指标的改善。 

---
