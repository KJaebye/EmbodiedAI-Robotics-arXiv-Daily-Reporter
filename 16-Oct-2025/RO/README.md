# InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy 

**Title (ZH)**: InternVLA-M1: 一种空间导向的视觉-语言-行动框架，用于通用机器人政策 

**Authors**: Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, Jiangmiao Pang, Yu Qiao, Yang Tian, Bin Wang, Bolun Wang, Fangjing Wang, Hanqing Wang, Tai Wang, Ziqin Wang, Xueyuan Wei, Chao Wu, Shuai Yang, Jinhui Ye, Junqiu Yu, Jia Zeng, Jingjing Zhang, Jinyu Zhang, Shi Zhang, Feng Zheng, Bowen Zhou, Yangkun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13778)  

**Abstract**: We introduce InternVLA-M1, a unified framework for spatial grounding and robot control that advances instruction-following robots toward scalable, general-purpose intelligence. Its core idea is spatially guided vision-language-action training, where spatial grounding serves as the critical link between instructions and robot actions. InternVLA-M1 employs a two-stage pipeline: (i) spatial grounding pre-training on over 2.3M spatial reasoning data to determine ``where to act'' by aligning instructions with visual, embodiment-agnostic positions, and (ii) spatially guided action post-training to decide ``how to act'' by generating embodiment-aware actions through plug-and-play spatial prompting. This spatially guided training recipe yields consistent gains: InternVLA-M1 outperforms its variant without spatial guidance by +14.6% on SimplerEnv Google Robot, +17% on WidowX, and +4.3% on LIBERO Franka, while demonstrating stronger spatial reasoning capability in box, point, and trace prediction. To further scale instruction following, we built a simulation engine to collect 244K generalizable pick-and-place episodes, enabling a 6.2% average improvement across 200 tasks and 3K+ objects. In real-world clustered pick-and-place, InternVLA-M1 improved by 7.3%, and with synthetic co-training, achieved +20.6% on unseen objects and novel configurations. Moreover, in long-horizon reasoning-intensive scenarios, it surpassed existing works by over 10%. These results highlight spatially guided training as a unifying principle for scalable and resilient generalist robots. Code and models are available at this https URL. 

**Abstract (ZH)**: We介绍InternVLA-M1：一种统一的空间语义接地与机器人控制框架，推动指令遵循机器人向可扩展的通用智能发展。其核心理念是空间引导的视觉-语言-行动训练，其中空间语义接地是指令与机器人动作之间的关键链接。InternVLA-M1采用两阶段管道：（i）在超过230万的空间推理数据上进行空间语义预训练，通过将指令与视觉和身体无关的位置对齐来确定“在哪里行动”；（ii）空间引导的动作后训练，通过插拔式空间提示生成体知性的动作来决定“如何行动”。这种空间引导的训练方案带来了一致的性能提升：InternVLA-M1在SimplerEnv Google Robot上比没有空间引导的变体高出14.6%，在WidowX上高出17%，在LIBERO Franka上高出4.3%，同时在箱、点和轨迹预测中展示了更强的空间推理能力。为了进一步扩大指令遵循的应用范围，我们构建了一个模拟引擎收集了244K可泛化的抓取和放置示例，实现了200个任务和3K+个物体的平均6.2%改进。在真实的集群抓取和放置中，InternVLA-M1提高了7.3%，并通过合成协同训练实现了未见过的物体和新配置上的20.6%改进。此外，在长视窗推理密集的场景中，它在很多方面超过了现有工作超过10%。这些结果突显了空间引导训练作为构建可扩展和健壮通用机器人的一种统一原则。代码和模型可在以下链接获取。 

---
# Hierarchical Discrete Lattice Assembly: An Approach for the Digital Fabrication of Scalable Macroscale Structures 

**Title (ZH)**: 分级离散 lattice 组装：一种大规模结构的数字制造方法 

**Authors**: Miana Smith, Paul Arthur Richard, Alexander Htet Kyaw, Neil Gershenfeld  

**Link**: [PDF](https://arxiv.org/pdf/2510.13686)  

**Abstract**: Although digital fabrication processes at the desktop scale have become proficient and prolific, systems aimed at producing larger-scale structures are still typically complex, expensive, and unreliable. In this work, we present an approach for the fabrication of scalable macroscale structures using simple robots and interlocking lattice building blocks. A target structure is first voxelized so that it can be populated with an architected lattice. These voxels are then grouped into larger interconnected blocks, which are produced using standard digital fabrication processes, leveraging their capability to produce highly complex geometries at a small scale. These blocks, on the size scale of tens of centimeters, are then fed to mobile relative robots that are able to traverse over the structure and place new blocks to form structures on the meter scale. To facilitate the assembly of large structures, we introduce a live digital twin simulation tool for controlling and coordinating assembly robots that enables both global planning for a target structure and live user design, interaction, or intervention. To improve assembly throughput, we introduce a new modular assembly robot, designed for hierarchical voxel handling. We validate this system by demonstrating the voxelization, hierarchical blocking, path planning, and robotic fabrication of a set of meter-scale objects. 

**Abstract (ZH)**: 虽然桌面规模的数字化制造进程已经变得熟练且高效，针对更大规模结构的制造系统仍然通常比较复杂、昂贵且不可靠。在本文中，我们提出了一种使用简单机器人和嵌锁 lattice 建筑块来制造可扩展的宏观结构的方法。首先，目标结构被体素化，以便能够填充一种结构化的 lattice。接着，这些体素被分组为更大且相连的块，这些块通过标准化的数字化制造过程生产，利用其在小尺度上生产高度复杂几何形状的能力。这些尺寸为几十厘米的块被提供给能够跨越结构并在其上放置新块以形成数米规模结构的移动相对机器人。为了便于大型结构的组装，我们引入了一种实时数字孪生仿真工具，用于控制和协调装配机器人，既支持全局性的目标结构规划，也支持实时的设计、交互或干预。为了提高装配效率，我们引入了一种新的模块化装配机器人，设计用于分层体素处理。我们通过示范一系列数米规模的对象的体素化、分层块化、路径规划和机器人制造来验证该系统。 

---
# On Your Own: Pro-level Autonomous Drone Racing in Uninstrumented Arenas 

**Title (ZH)**: 自行其道：在未标定赛场上的专业级自主无人机竞速 

**Authors**: Michael Bosello, Flavio Pinzarrone, Sara Kiade, Davide Aguiari, Yvo Keuter, Aaesha AlShehhi, Gyordan Caminati, Kei Long Wong, Ka Seng Chou, Junaid Halepota, Fares Alneyadi, Jacopo Panerati, Giovanni Pau  

**Link**: [PDF](https://arxiv.org/pdf/2510.13644)  

**Abstract**: Drone technology is proliferating in many industries, including agriculture, logistics, defense, infrastructure, and environmental monitoring. Vision-based autonomy is one of its key enablers, particularly for real-world applications. This is essential for operating in novel, unstructured environments where traditional navigation methods may be unavailable. Autonomous drone racing has become the de facto benchmark for such systems. State-of-the-art research has shown that autonomous systems can surpass human-level performance in racing arenas. However, direct applicability to commercial and field operations is still limited as current systems are often trained and evaluated in highly controlled environments. In our contribution, the system's capabilities are analyzed within a controlled environment -- where external tracking is available for ground-truth comparison -- but also demonstrated in a challenging, uninstrumented environment -- where ground-truth measurements were never available. We show that our approach can match the performance of professional human pilots in both scenarios. We also publicly release the data from the flights carried out by our approach and a world-class human pilot. 

**Abstract (ZH)**: 无人机技术在农业、物流、国防、基础设施和环境监测等多个行业中广泛应用。基于视觉的自主导航是其实现的关键 enabler，特别是在实际应用中。这对于在传统导航方法可能不可用的新型非结构化环境中操作至关重要。自主无人机竞速已成为此类系统的事实标准。最先进的研究显示，自主系统在竞速场地上可以超越人类水平的表现。然而，将其直接应用于商业和现场操作仍然有限，因为当前系统通常是在高度受控的环境中进行训练和评估。在我们的贡献中，系统的能力在可控环境中进行了分析——其中外部跟踪可用以进行地truth比较——但在具有挑战性的、未装备测量仪器的环境中也得到了验证——其中从未获得过地truth测量数据。我们展示了我们的方法在这种两种场景下都能达到职业人类飞行员的性能水平。我们还公开发布了我们方法和世界级人类飞行员飞行过程中的数据。 

---
# LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models 

**Title (ZH)**: LIBERO-Plus:  vision-language-action模型的深入鲁棒性分析 

**Authors**: Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13626)  

**Abstract**: Visual-Language-Action (VLA) models report impressive success rates on robotic manipulation benchmarks, yet these results may mask fundamental weaknesses in robustness. We perform a systematic vulnerability analysis by introducing controlled perturbations across seven dimensions: objects layout, camera viewpoints, robot initial states, language instructions, light conditions, background textures and sensor noise. We comprehensively analyzed multiple state-of-the-art models and revealed consistent brittleness beneath apparent competence. Our analysis exposes critical weaknesses: models exhibit extreme sensitivity to perturbation factors, including camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations. Surprisingly, models are largely insensitive to language variations, with further experiments revealing that models tend to ignore language instructions completely. Our findings challenge the assumption that high benchmark scores equate to true competency and highlight the need for evaluation practices that assess reliability under realistic variation. 

**Abstract (ZH)**: 视觉-语言-动作（VLA）模型在机器人操作基准测试中取得了令人印象深刻的成功率，但这些结果可能掩盖了鲁棒性方面的根本弱点。我们通过在七个维度上引入受控的扰动进行了系统的脆弱性分析：物体布局、相机视角、机器人初始状态、语言指令、光照条件、背景纹理和传感器噪声。我们全面分析了多个领先模型，并揭示了表面看似能力较强的实际脆弱性。我们的分析暴露了一些关键弱点：模型对扰动因素表现出极端的敏感性，包括相机视角和机器人初始状态，在适度扰动下性能从95%下降到低于30%。令人惊讶的是，模型对语言变化的敏感性很小，进一步的实验表明，模型倾向于完全忽略语言指令。我们的研究挑战了高基准分数等同于真正能力的观点，凸显了在实际变化条件下评估可靠性的必要性。 

---
# A Modular Object Detection System for Humanoid Robots Using YOLO 

**Title (ZH)**: 基于YOLO的人形机器人模块化目标检测系统 

**Authors**: Nicolas Pottier, Meng Cheng Lau  

**Link**: [PDF](https://arxiv.org/pdf/2510.13625)  

**Abstract**: Within the field of robotics, computer vision remains a significant barrier to progress, with many tasks hindered by inefficient vision systems. This research proposes a generalized vision module leveraging YOLOv9, a state-of-the-art framework optimized for computationally constrained environments like robots. The model is trained on a dataset tailored to the FIRA robotics Hurocup. A new vision module is implemented in ROS1 using a virtual environment to enable YOLO compatibility. Performance is evaluated using metrics such as frames per second (FPS) and Mean Average Precision (mAP). Performance is then compared to the existing geometric framework in static and dynamic contexts. The YOLO model achieved comparable precision at a higher computational cost then the geometric model, while providing improved robustness. 

**Abstract (ZH)**: 机器人领域中，计算机视觉依然是进步的重要障碍，许多任务受限于低效的视觉系统。本研究提出了一种基于YOLOv9的通用视觉模块，YOLOv9是为计算资源受限的环境（如机器人）优化的先进框架。该模型在专门为FIRA机器人Hurocup比赛设计的数据集上进行训练。通过在ROS1中实现一个新的视觉模块并在虚拟环境中启用YOLO兼容性。性能通过每秒帧数（FPS）和平均平均精确度（mAP）等指标进行评估。然后将YOLO模型的表现与现有的几何框架在静态和动态环境中进行比较。结果表明，YOLO模型在计算成本更高的情况下实现了可比较的精度，并提供了更好的鲁棒性。 

---
# Characterizing Lidar Point-Cloud Adversities Using a Vector Field Visualization 

**Title (ZH)**: 使用向量场可视化表征LIDAR点云 adversities 

**Authors**: Daniel Choate, Jason Rife  

**Link**: [PDF](https://arxiv.org/pdf/2510.13619)  

**Abstract**: In this paper we introduce a visualization methodology to aid a human analyst in classifying adversity modes that impact lidar scan matching. Our methodology is intended for offline rather than real-time analysis. The method generates a vector-field plot that characterizes local discrepancies between a pair of registered point clouds. The vector field plot reveals patterns that would be difficult for the analyst to extract from raw point-cloud data. After introducing our methodology, we apply the process to two proof-of-concept examples: one a simulation study and the other a field experiment. For both data sets, a human analyst was able to reason about a series of adversity mechanisms and iteratively remove those mechanisms from the raw data, to help focus attention on progressively smaller discrepancies. 

**Abstract (ZH)**: 本文介绍了一种可视化方法，旨在辅助人类分析师对影响激光雷达扫描匹配的逆境模式进行分类。该方法适用于离线分析而非实时分析。该方法生成一个向量场图，以描述一对配准点云之间的局部差异。向量场图揭示了难以从原始点云数据中提取的模式。在介绍该方法后，我们将其应用于两个概念验证案例：一个是模拟研究，另一个是实地实验。对于两组数据集，人类分析师能够推断一系列逆境机制，并迭代地从原始数据中去除这些机制，以逐步吸引对越来越小差异的关注。 

---
# Efficient Force and Stiffness Prediction in Robotic Produce Handling with a Piezoresistive Pressure Sensor 

**Title (ZH)**: 基于压阻式压力传感器的机器人农产品处理中高效力和刚度预测 

**Authors**: Preston Fairchild, Claudia Chen, Xiaobo Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13616)  

**Abstract**: Properly handling delicate produce with robotic manipulators is a major part of the future role of automation in agricultural harvesting and processing. Grasping with the correct amount of force is crucial in not only ensuring proper grip on the object, but also to avoid damaging or bruising the product. In this work, a flexible pressure sensor that is both low cost and easy to fabricate is integrated with robotic grippers for working with produce of varying shapes, sizes, and stiffnesses. The sensor is successfully integrated with both a rigid robotic gripper, as well as a pneumatically actuated soft finger. Furthermore, an algorithm is proposed for accelerated estimation of the steady-state value of the sensor output based on the transient response data, to enable real-time applications. The sensor is shown to be effective in incorporating feedback to correctly grasp objects of unknown sizes and stiffnesses. At the same time, the sensor provides estimates for these values which can be utilized for identification of qualities such as ripeness levels and bruising. It is also shown to be able to provide force feedback for objects of variable stiffnesses. This enables future use not only for produce identification, but also for tasks such as quality control and selective distribution based on ripeness levels. 

**Abstract (ZH)**: 适当地使用机器人操作器处理 delicate produce 是未来农业采收和加工中自动化 rôle 的一个重要部分。正确施加力的抓取对于确保正确握持物体以及避免损坏或压伤产品至关重要。在此工作中，一种低成本且易于制造的柔性压力传感器被集成到能够处理不同形状、大小和 stiffness 的 produce 的机器人 gripper 中。该传感器成功地与一种刚性机器人 gripper 以及一种气动驱动的软手指进行了集成。此外，提出了一种算法，该算法基于瞬态响应数据加速估计传感器输出的稳态值，以实现实时应用。传感器被证明能够通过反馈正确抓取未知尺寸和 stiffness 的物体，并同时提供这些值的估计，从而用于识别如成熟度和压伤等品质。此外，该传感器还能为具有不同 stiffness 的物体提供力反馈。这不仅使其能够用于 produce 的识别，还能够用于基于成熟度的品质控制和甄选分配等任务。 

---
# PlanarMesh: Building Compact 3D Meshes from LiDAR using Incremental Adaptive Resolution Reconstruction 

**Title (ZH)**: PlanarMesh：使用增量自适应分辨率重建从LiDAR构建紧凑3D网格 

**Authors**: Jiahao Wang, Nived Chebrolu, Yifu Tao, Lintong Zhang, Ayoung Kim, Maurice Fallon  

**Link**: [PDF](https://arxiv.org/pdf/2510.13599)  

**Abstract**: Building an online 3D LiDAR mapping system that produces a detailed surface reconstruction while remaining computationally efficient is a challenging task. In this paper, we present PlanarMesh, a novel incremental, mesh-based LiDAR reconstruction system that adaptively adjusts mesh resolution to achieve compact, detailed reconstructions in real-time. It introduces a new representation, planar-mesh, which combines plane modeling and meshing to capture both large surfaces and detailed geometry. The planar-mesh can be incrementally updated considering both local surface curvature and free-space information from sensor measurements. We employ a multi-threaded architecture with a Bounding Volume Hierarchy (BVH) for efficient data storage and fast search operations, enabling real-time performance. Experimental results show that our method achieves reconstruction accuracy on par with, or exceeding, state-of-the-art techniques-including truncated signed distance functions, occupancy mapping, and voxel-based meshing-while producing smaller output file sizes (10 times smaller than raw input and more than 5 times smaller than mesh-based methods) and maintaining real-time performance (around 2 Hz for a 64-beam sensor). 

**Abstract (ZH)**: 构建一个在线3D LiDAR建图系统，该系统能够在保持高效计算的同时生成详细表面重建是一项具有挑战性的工作。在本文中，我们提出了PlanarMesh，这是一种新颖的增量式网格基LiDAR重建系统，能够适应性调整网格分辨率，以实现实时的紧凑且详细的重建。PlanarMesh引入了一种新的表示方法，即平面网格，该方法结合了平面建模和网格化技术，以捕捉大面积和详细的几何形状。平面网格可以在考虑局部表面曲率和传感器测量中的自由空间信息的同时进行增量式更新。我们采用多线程架构并结合Bounding Volume Hierarchy (BVH) 来高效存储数据和快速执行搜索操作，从而实现实时性能。实验结果表明，我们的方法在重建准确性方面与最先进的技术（包括截断的符号距离函数、占据映射和体素网格化）相当或优于这些技术，同时生成的输出文件大小更小（原始输入的1/10，比基于网格的方法小5倍以上），并保持实时性能（64束传感器大约为2 Hz）。 

---
# Active Tactile Exploration for Rigid Body Pose and Shape Estimation 

**Title (ZH)**: 刚体姿态与形状估计的主动触觉探索 

**Authors**: Ethan K. Gordon, Bruke Baraki, Hien Bui, Michael Posa  

**Link**: [PDF](https://arxiv.org/pdf/2510.13595)  

**Abstract**: General robot manipulation requires the handling of previously unseen objects. Learning a physically accurate model at test time can provide significant benefits in data efficiency, predictability, and reuse between tasks. Tactile sensing can compliment vision with its robustness to occlusion, but its temporal sparsity necessitates careful online exploration to maintain data efficiency. Direct contact can also cause an unrestrained object to move, requiring both shape and location estimation. In this work, we propose a learning and exploration framework that uses only tactile data to simultaneously determine the shape and location of rigid objects with minimal robot motion. We build on recent advances in contact-rich system identification to formulate a loss function that penalizes physical constraint violation without introducing the numerical stiffness inherent in rigid-body contact. Optimizing this loss, we can learn cuboid and convex polyhedral geometries with less than 10s of randomly collected data after first contact. Our exploration scheme seeks to maximize Expected Information Gain and results in significantly faster learning in both simulated and real-robot experiments. More information can be found at this https URL 

**Abstract (ZH)**: 仅使用触觉数据同时在最小机器人运动下确定刚体对象的形状和位置的学习与探索框架 

---
# Development of an Intuitive GUI for Non-Expert Teleoperation of Humanoid Robots 

**Title (ZH)**: 非专家操作类人机器人直观GUI的发展 

**Authors**: Austin Barret, Meng Cheng Lau  

**Link**: [PDF](https://arxiv.org/pdf/2510.13594)  

**Abstract**: The operation of humanoid robotics is an essential field of research with many practical and competitive applications. Many of these systems, however, do not invest heavily in developing a non-expert-centered graphical user interface (GUI) for operation. The focus of this research is to develop a scalable GUI that is tailored to be simple and intuitive so non-expert operators can control the robot through a FIRA-regulated obstacle course. Using common practices from user interface development (UI) and understanding concepts described in human-robot interaction (HRI) and other related concepts, we will develop a new interface with the goal of a non-expert teleoperation system. 

**Abstract (ZH)**: 人形机器人操作的图形用户界面设计研究：面向非专家的可扩展界面开发 

---
# Hoecken-D Hand: A Novel Robotic Hand for Linear Parallel Pinching and Self-Adaptive Grasping 

**Title (ZH)**: Hoeken-D 手：一种新型线性并行夹持和自适应抓取的机器人手 

**Authors**: Wentao Guo, Wenzeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13553)  

**Abstract**: This paper presents the Hoecken-D Hand, an underactuated robotic gripper that combines a modified Hoecken linkage with a differential spring mechanism to achieve both linear parallel pinching and a mid-stroke transition to adaptive envelope. The original Hoecken linkage is reconfigured by replacing one member with differential links, preserving straight-line guidance while enabling contact-triggered reconfiguration without additional actuators. A double-parallelogram arrangement maintains fingertip parallelism during conventional pinching, whereas the differential mechanism allows one finger to wrap inward upon encountering an obstacle, improving stability on irregular or thin objects. The mechanism can be driven by a single linear actuator, minimizing complexity and cost; in our prototype, each finger is driven by its own linear actuator for simplicity. We perform kinematic modeling and force analysis to characterize grasp performance, including simulated grasping forces and spring-opening behavior under varying geometric parameters. The design was prototyped using PLA-based 3D printing, achieving a linear pinching span of approximately 200 mm. Preliminary tests demonstrate reliable grasping in both modes across a wide range of object geometries, highlighting the Hoecken-D Hand as a compact, adaptable, and cost-effective solution for manipulation in unstructured environments. 

**Abstract (ZH)**: Hoecken-D 手，一种结合修改后的 Hoecken 连接架与差动弹簧机制的欠驱动机器人夹爪，实现线性并行夹持及中程过渡到自适应包络功能 

---
# A Novel Robot Hand with Hoeckens Linkages and Soft Phalanges for Scooping and Self-Adaptive Grasping in Environmental Constraints 

**Title (ZH)**: 一种新型hoeckens连杆与软指节机器人手，在环境约束下进行挖取和自适应抓取 

**Authors**: Wentao Guo, Yizhou Wang, Wenzeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13535)  

**Abstract**: This paper presents a novel underactuated adaptive robotic hand, Hockens-A Hand, which integrates the Hoeckens mechanism, a double-parallelogram linkage, and a specialized four-bar linkage to achieve three adaptive grasping modes: parallel pinching, asymmetric scooping, and enveloping grasping. Hockens-A Hand requires only a single linear actuator, leveraging passive mechanical intelligence to ensure adaptability and compliance in unstructured environments. Specifically, the vertical motion of the Hoeckens mechanism introduces compliance, the double-parallelogram linkage ensures line contact at the fingertip, and the four-bar amplification system enables natural transitions between different grasping modes. Additionally, the inclusion of a mesh-textured silicone phalanx further enhances the ability to envelop objects of various shapes and sizes. This study employs detailed kinematic analysis to optimize the push angle and design the linkage lengths for optimal performance. Simulations validated the design by analyzing the fingertip motion and ensuring smooth transitions between grasping modes. Furthermore, the grasping force was analyzed using power equations to enhance the understanding of the system's this http URL validation using a 3D-printed prototype demonstrates the three grasping modes of the hand in various scenarios under environmental constraints, verifying its grasping stability and broad applicability. 

**Abstract (ZH)**: 一种集成霍肯斯机制的新型欠驱动自适应机器人手Hockens-A Hand及其三种自适应抓握模式的研究 

---
# Bridge the Gap: Enhancing Quadruped Locomotion with Vertical Ground Perturbations 

**Title (ZH)**: 桥接差距：通过垂直地面扰动增强四足运动能力 

**Authors**: Maximilian Stasica, Arne Bick, Nico Bohlinger, Omid Mohseni, Max Johannes Alois Fritzsche, Clemens Hübler, Jan Peters, André Seyfarth  

**Link**: [PDF](https://arxiv.org/pdf/2510.13488)  

**Abstract**: Legged robots, particularly quadrupeds, excel at navigating rough terrains, yet their performance under vertical ground perturbations, such as those from oscillating surfaces, remains underexplored. This study introduces a novel approach to enhance quadruped locomotion robustness by training the Unitree Go2 robot on an oscillating bridge - a 13.24-meter steel-and-concrete structure with a 2.0 Hz eigenfrequency designed to perturb locomotion. Using Reinforcement Learning (RL) with the Proximal Policy Optimization (PPO) algorithm in a MuJoCo simulation, we trained 15 distinct locomotion policies, combining five gaits (trot, pace, bound, free, default) with three training conditions: rigid bridge and two oscillating bridge setups with differing height regulation strategies (relative to bridge surface or ground). Domain randomization ensured zero-shot transfer to the real-world bridge. Our results demonstrate that policies trained on the oscillating bridge exhibit superior stability and adaptability compared to those trained on rigid surfaces. Our framework enables robust gait patterns even without prior bridge exposure. These findings highlight the potential of simulation-based RL to improve quadruped locomotion during dynamic ground perturbations, offering insights for designing robots capable of traversing vibrating environments. 

**Abstract (ZH)**: 腿式机器人，尤其是四足机器人，在穿越崎岖地形方面表现出色，但在应对垂直地面振荡等动态地面扰动方面的性能仍需进一步探索。本研究通过在一种2.0 Hz固有频率的振荡桥梁上训练Unitree Go2四足机器人，提出了一种增强四足机器人运动稳定性和适应性的新方法。该振荡桥梁由13.24米长的钢混结构组成，用于模拟运动的扰动。利用MuJoCo模拟中的强化学习（RL）和 proximal policy optimization (PPO) 算法，我们训练了15种不同的运动策略，结合了五种步态（ tönt, pace, bound, free, default）和三种训练条件：刚性桥梁以及两个不同高度调节策略的振荡桥梁设置（相对于桥梁表面或地面）。领域随机化确保了在实际桥梁上的零样本迁移。结果显示，振荡桥梁上训练的策略在稳定性与适应性方面优于刚性表面上训练的策略。本框架使四足机器人即使没有前桥暴露也能展现出稳健的步态模式。这些发现突显了基于模拟的RL在提高四足机器人在动态地面扰动环境下运动性能方面的潜力，为设计能够穿越振动环境的机器人提供了参考。 

---
# Real-Time Knee Angle Prediction Using EMG and Kinematic Data with an Attention-Based CNN-LSTM Network and Transfer Learning Across Multiple Datasets 

**Title (ZH)**: 基于注意力机制的CNN-LSTM网络及跨数据集迁移学习的实时膝关节角度预测 

**Authors**: Mojtaba Mollahossein, Gholamreza Vossoughi, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2510.13443)  

**Abstract**: Electromyography (EMG) signals are widely used for predicting body joint angles through machine learning (ML) and deep learning (DL) methods. However, these approaches often face challenges such as limited real-time applicability, non-representative test conditions, and the need for large datasets to achieve optimal performance. This paper presents a transfer-learning framework for knee joint angle prediction that requires only a few gait cycles from new subjects. Three datasets - Georgia Tech, the University of California Irvine (UCI), and the Sharif Mechatronic Lab Exoskeleton (SMLE) - containing four EMG channels relevant to knee motion were utilized. A lightweight attention-based CNN-LSTM model was developed and pre-trained on the Georgia Tech dataset, then transferred to the UCI and SMLE datasets. The proposed model achieved Normalized Mean Absolute Errors (NMAE) of 6.8 percent and 13.7 percent for one-step and 50-step predictions on abnormal subjects using EMG inputs alone. Incorporating historical knee angles reduced the NMAE to 3.1 percent and 3.5 percent for normal subjects, and to 2.8 percent and 7.5 percent for abnormal subjects. When further adapted to the SMLE exoskeleton with EMG, kinematic, and interaction force inputs, the model achieved 1.09 percent and 3.1 percent NMAE for one- and 50-step predictions, respectively. These results demonstrate robust performance and strong generalization for both short- and long-term rehabilitation scenarios. 

**Abstract (ZH)**: 基于迁移学习的少量步态周期内膝关节角度预测框架 

---
# Adversarial Fine-tuning in Offline-to-Online Reinforcement Learning for Robust Robot Control 

**Title (ZH)**: Offline-to-Online Reinforcement Learning中的对抗性微调以实现稳健的机器人控制 

**Authors**: Shingo Ayabe, Hiroshi Kera, Kazuhiko Kawamoto  

**Link**: [PDF](https://arxiv.org/pdf/2510.13358)  

**Abstract**: Offline reinforcement learning enables sample-efficient policy acquisition without risky online interaction, yet policies trained on static datasets remain brittle under action-space perturbations such as actuator faults. This study introduces an offline-to-online framework that trains policies on clean data and then performs adversarial fine-tuning, where perturbations are injected into executed actions to induce compensatory behavior and improve resilience. A performance-aware curriculum further adjusts the perturbation probability during training via an exponential-moving-average signal, balancing robustness and stability throughout the learning process. Experiments on continuous-control locomotion tasks demonstrate that the proposed method consistently improves robustness over offline-only baselines and converges faster than training from scratch. Matching the fine-tuning and evaluation conditions yields the strongest robustness to action-space perturbations, while the adaptive curriculum strategy mitigates the degradation of nominal performance observed with the linear curriculum strategy. Overall, the results show that adversarial fine-tuning enables adaptive and robust control under uncertain environments, bridging the gap between offline efficiency and online adaptability. 

**Abstract (ZH)**: 离线强化学习使得在无需冒风险进行在线交互的情况下获得样本效率高的策略成为可能，但用静态数据集训练的策略在动作空间扰动（如执行器故障）下仍然脆弱。本研究引入了一种从离线到在线的框架，在清洁数据上训练策略，然后进行对抗性微调，在执行的动作中注入扰动以诱导补偿行为并提高鲁棒性。性能感知的分阶段训练进一步通过指数移动平均信号调整训练中的扰动概率，以在整个学习过程中平衡鲁棒性和稳定性。实验证实在连续控制运动任务中，所提出的方法在鲁棒性方面持续优于仅离线基线，并且收敛速度更快。匹配微调和评估条件可以在动作空间扰动下实现最强的鲁棒性，而自适应分阶段训练策略缓解了线性分阶段训练策略下名义性能下降的问题。总体而言，结果表明对抗性微调在不确定性环境中实现适应性和鲁棒控制，弥合了离线效率与在线适应性之间的差距。 

---
# MODUR: A Modular Dual-reconfigurable Robot 

**Title (ZH)**: MODUR：一种模块化双重构机器人 

**Authors**: Jie Gu, Tin Lun Lam, Chunxu Tian, Zhihao Xia, Yongheng Xing, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13356)  

**Abstract**: Modular Self-Reconfigurable Robot (MSRR) systems are a class of robots capable of forming higher-level robotic systems by altering the topological relationships between modules, offering enhanced adaptability and robustness in various environments. This paper presents a novel MSRR called MODUR, featuring dual-level reconfiguration capabilities designed to integrate reconfigurable mechanisms into MSRR. Specifically, MODUR can perform high-level self-reconfiguration among modules to create different configurations, while each module is also able to change its shape to execute basic motions. The design of MODUR primarily includes a compact connector and scissor linkage groups that provide actuation, forming a parallel mechanism capable of achieving both connector motion decoupling and adjacent position migration capabilities. Furthermore, the workspace, considering the interdependent connectors, is comprehensively analyzed, laying a theoretical foundation for the design of the module's basic motion. Finally, the motion of MODUR is validated through a series of experiments. 

**Abstract (ZH)**: 模块化自重构机器人（MSRR）系统是一类能够通过改变模块之间的拓扑关系来形成更高层次的机器人系统，从而在各种环境中提供增强的适应性和鲁棒性。本文提出了一种名为MODUR的新型MSRR，具有双层重构能力，旨在将重构机制集成到MSRR中。具体而言，MODUR可以在模块之间进行高层次的自重构以形成不同的配置，而每个模块也可以改变形状以执行基本运动。MODUR的设计主要包括紧凑的连接器和剪刀连杆组，提供驱动功能，形成一个并联机构，能够实现连接器运动解耦和相邻位置迁移能力。此外，考虑到互连连接器的工作空间进行了全面分析，为模块基本运动的设计奠定了理论基础。最后，通过一系列实验验证了MODUR的运动。 

---
# Tactile-Conditioned Diffusion Policy for Force-Aware Robotic Manipulation 

**Title (ZH)**: 触觉条件下的扩散策略用于力感知机器人 manipulation 

**Authors**: Erik Helmut, Niklas Funk, Tim Schneider, Cristiana de Farias, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2510.13324)  

**Abstract**: Contact-rich manipulation depends on applying the correct grasp forces throughout the manipulation task, especially when handling fragile or deformable objects. Most existing imitation learning approaches often treat visuotactile feedback only as an additional observation, leaving applied forces as an uncontrolled consequence of gripper commands. In this work, we present Force-Aware Robotic Manipulation (FARM), an imitation learning framework that integrates high-dimensional tactile data to infer tactile-conditioned force signals, which in turn define a matching force-based action space. We collect human demonstrations using a modified version of the handheld Universal Manipulation Interface (UMI) gripper that integrates a GelSight Mini visual tactile sensor. For deploying the learned policies, we developed an actuated variant of the UMI gripper with geometry matching our handheld version. During policy rollouts, the proposed FARM diffusion policy jointly predicts robot pose, grip width, and grip force. FARM outperforms several baselines across three tasks with distinct force requirements -- high-force, low-force, and dynamic force adaptation -- demonstrating the advantages of its two key components: leveraging force-grounded, high-dimensional tactile observations and a force-based control space. The codebase and design files are open-sourced and available at this https URL . 

**Abstract (ZH)**: 接触丰富的操作依赖于在整个操作任务中施加正确的夹持力，特别是在处理脆弱或可变形对象时。现有的大多数 imitation learning 方法通常将触觉视觉反馈仅视为额外的观察信息，而未控制夹爪命令引起的施力结果。在本文中，我们提出了 Force-Aware Robotic Manipulation (FARM)，这是一种集成高维触觉数据的 imitation learning 框架，用于推断触觉条件下的力信号，进而定义一个匹配的基于力的动作空间。我们使用一种集成了 GelSight Mini 触觉视觉传感器的修改版本的手持 Universal Manipulation Interface (UMI) 夹爪收集人类演示。为部署所学策略，我们开发了一种几何匹配手持版本的带有执行器的 UMI 夹爪。在策略展开过程中，提出的 FARM 扩散策略联合预测机器人的姿态、握持宽度和握持力。FARM 在具有不同力要求的三个任务上优于多种基准方法，证明了其两个关键组件的优势：利用基于力的高维触觉观察和力基控制空间。代码库和设计文件已开源，可在以下链接访问：this https URL。 

---
# DAMM-LOAM: Degeneracy Aware Multi-Metric LiDAR Odometry and Mapping 

**Title (ZH)**: DAMM-LOAM: 退化感知多度量激光雷达里程计与 Mapping 

**Authors**: Nishant Chandna, Akshat Kaushal  

**Link**: [PDF](https://arxiv.org/pdf/2510.13287)  

**Abstract**: LiDAR Simultaneous Localization and Mapping (SLAM) systems are essential for enabling precise navigation and environmental reconstruction across various applications. Although current point-to-plane ICP algorithms perform effec- tively in structured, feature-rich environments, they struggle in scenarios with sparse features, repetitive geometric structures, and high-frequency motion. This leads to degeneracy in 6- DOF pose estimation. Most state-of-the-art algorithms address these challenges by incorporating additional sensing modalities, but LiDAR-only solutions continue to face limitations under such conditions. To address these issues, we propose a novel Degeneracy-Aware Multi-Metric LiDAR Odometry and Map- ping (DAMM-LOAM) module. Our system improves mapping accuracy through point cloud classification based on surface normals and neighborhood analysis. Points are classified into ground, walls, roof, edges, and non-planar points, enabling accurate correspondences. A Degeneracy-based weighted least squares-based ICP algorithm is then applied for accurate odom- etry estimation. Additionally, a Scan Context based back-end is implemented to support robust loop closures. DAMM-LOAM demonstrates significant improvements in odometry accuracy, especially in indoor environments such as long corridors 

**Abstract (ZH)**: 基于退化感知的多度量LiDAR simultaneous localization and mapping (DAMM-LOAM)模块 

---
# ALOHA2 Robot Kitchen Application Scenario Reproduction Report 

**Title (ZH)**: ALOHA2 机器人厨房应用场景再现报告 

**Authors**: Haoyang Wu, Siheng Wu, William X. Liu, Fangui Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.13284)  

**Abstract**: ALOHA2 is an enhanced version of the dual-arm teleoperated robot ALOHA, featuring higher performance and robustness compared to the original design, while also being more ergonomic. Like ALOHA, ALOHA2 consists of two grippers and two ViperX 6-DoF arms, as well as two smaller WidowX arms. Users control the follower mechanical arms by operating the leader mechanical arms through back-driving. The device also includes cameras that generate images from multiple viewpoints, allowing for RGB data collection during teleoperation. The robot is mounted on a 48-inch x 30-inch table, equipped with an aluminum frame that provides additional mounting points for cameras and gravity compensation systems. 

**Abstract (ZH)**: ALOHA2是双臂遥操作机器人ALOHA的增强版本，相比原始设计具有更高的性能和更强的鲁棒性，同时还更加符合人体工程学。ALOHA2由两个 gripper 和两个 ViperX 6-DoF 臂以及两个较小的 WidowX 臂组成。用户通过反驱操作领导者机械臂来控制跟随者机械臂。该设备还包括能够从多个视角生成图像的相机，允许在遥操作过程中收集RGB数据。机器人安装在48英寸×30英寸的桌上，配备了一个铝合金框架，提供了额外的相机和重力补偿系统安装点。 

---
# RoboHiMan: A Hierarchical Evaluation Paradigm for Compositional Generalization in Long-Horizon Manipulation 

**Title (ZH)**: RoboHiMan: 一种层次化的评估范式，用于长时效Manipulation中的组合作用泛化 

**Authors**: Yangtao Chen, Zixuan Chen, Nga Teng Chan, Junting Chen, Junhui Yin, Jieqi Shi, Yang Gao, Yong-Lu Li, Jing Huo  

**Link**: [PDF](https://arxiv.org/pdf/2510.13149)  

**Abstract**: Enabling robots to flexibly schedule and compose learned skills for novel long-horizon manipulation under diverse perturbations remains a core challenge. Early explorations with end-to-end VLA models show limited success, as these models struggle to generalize beyond the training distribution. Hierarchical approaches, where high-level planners generate subgoals for low-level policies, bring certain improvements but still suffer under complex perturbations, revealing limited capability in skill composition. However, existing benchmarks primarily emphasize task completion in long-horizon settings, offering little insight into compositional generalization, robustness, and the interplay between planning and execution. To systematically investigate these gaps, we propose RoboHiMan, a hierarchical evaluation paradigm for compositional generalization in long-horizon manipulation. RoboHiMan introduces HiMan-Bench, a benchmark of atomic and compositional tasks under diverse perturbations, supported by a multi-level training dataset for analyzing progressive data scaling, and proposes three evaluation paradigms (vanilla, decoupled, coupled) that probe the necessity of skill composition and reveal bottlenecks in hierarchical architectures. Experiments highlight clear capability gaps across representative models and architectures, pointing to directions for advancing models better suited to real-world long-horizon manipulation tasks. Videos and open-source code can be found on our project website: this https URL. 

**Abstract (ZH)**: 使机器人能够灵活调度和组合学习到的技能以应对多样化的干扰进行新颖的长期操作调用仍然是一个核心挑战。端到端的VLA模型早期探索显示有限的成功，因为这些模型难以在训练分布之外泛化。分层方法，其中高层规划器为低层策略生成子目标，在复杂干扰下表现出一定的提高，但仍然在技能组合方面显示出有限的能力。然而，现有的基准主要强调在长期设置下的任务完成，对组合泛化、鲁棒性和规划与执行之间的相互作用提供很少的见解。为了系统地研究这些差距，我们提出了RoboHiMan，这是一种分层评估范式，用于长期操作中的组合泛化。RoboHiMan引入了HiMan-Bench，这是一个在多样干扰下包含原子和组合任务的基准，并提供了一个多层次训练数据集以分析逐级数据扩展，并提出了三种评估范式（vanilla、解耦、耦合），以探究技能组合的必要性并揭示分层架构中的瓶颈。实验突显了代表性模型和架构在能力上的明显差距，指出了改进更适合真实世界长期操作任务的模型的方向。更多信息和开源代码可在我们的项目网站上找到：<这个链接>。 

---
# VLA-0: Building State-of-the-Art VLAs with Zero Modification 

**Title (ZH)**: VLA-0: 构建零修改的顶级VLAs 

**Authors**: Ankit Goyal, Hugo Hadfield, Xuning Yang, Valts Blukis, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2510.13054)  

**Abstract**: Vision-Language-Action models (VLAs) hold immense promise for enabling generalist robot manipulation. However, the best way to build them remains an open question. Current approaches often add complexity, such as modifying the existing vocabulary of a Vision-Language Model (VLM) with action tokens or introducing special action heads. Curiously, the simplest strategy of representing actions directly as text has remained largely unexplored. This work introduces VLA-0 to investigate this idea. We find that VLA-0 is not only effective; it is surprisingly powerful. With the right design, VLA-0 outperforms more involved models. On LIBERO, a popular benchmark for evaluating VLAs, VLA-0 outperforms all existing methods trained on the same robotic data, including $\pi_0.5$-KI, OpenVLA-OFT and SmolVLA. Furthermore, without large-scale robotics-specific training, it outperforms methods trained on large-scale robotic data, like $\pi_0.5$-KI, $\pi_0$, GR00T-N1 and MolmoAct. These findings also translate to the real world, where VLA-0 outperforms SmolVLA, a VLA model pre-trained on large-scale real data. This paper summarizes our unexpected findings and spells out the specific techniques required to unlock the high performance of this simple yet potent VLA design. Visual results, code, and trained models are provided here: this https URL. 

**Abstract (ZH)**: Vision-Language-Action模型（VLAs）在实现通用机器人操作方面具有巨大的潜力。然而，如何构建它们仍然是一个开放的问题。当前的方法往往增加了复杂性，例如通过在视觉-语言模型（VLM）中添加动作标记或引入特殊动作头来修改现有词汇表。令人好奇的是，直接将动作表示为文本的最简单策略尚未得到充分探索。本文介绍了VLA-0来研究这一想法。我们发现，VLA-0不仅有效，而且出人意料地强大。在适当的架构设计下，VLA-0超越了更复杂的模型。在LIBERO这一流行的VLA评估基准上，VLA-0在使用相同机器人数据训练的所有现有方法中表现最佳，包括$\pi_0.5$-KI、OpenVLA-OFT和SmolVLA。此外，在未进行大规模机器人特定训练的情况下，VLA-0在使用大型机器人数据训练的方法中表现出色，如$\pi_0.5$-KI、$\pi_0$、GR00T-N1和MolmoAct。这些发现也适用于现实世界，在现实世界中，VLA-0在与大规模真实数据预训练的SmolVLA相比时表现出色。本文总结了我们意想不到的发现，并列出了实现这一简单而强大的VLA设计所需的具体技术。提供了视觉结果、代码和训练模型：this https URL。 

---
# Kinematic Kitbashing for Modeling Functional Articulated Objects 

**Title (ZH)**: 机械动力组件拼装法模拟功能性活动对象 

**Authors**: Minghao Guo, Victor Zordan, Sheldon Andrews, Wojciech Matusik, Maneesh Agrawala, Hsueh-Ti Derek Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13048)  

**Abstract**: We introduce Kinematic Kitbashing, an automatic framework that synthesizes functionality-aware articulated objects by reusing parts from existing models. Given a kinematic graph with a small collection of articulated parts, our optimizer jointly solves for the spatial placement of every part so that (i) attachments remain geometrically sound over the entire range of motion and (ii) the assembled object satisfies user-specified functional goals such as collision-free actuation, reachability, or trajectory following. At its core is a kinematics-aware attachment energy that aligns vector distance function features sampled across multiple articulation snapshots. We embed this attachment term within an annealed Riemannian Langevin dynamics sampler that treats functionality objectives as additional energies, enabling robust global exploration while accommodating non-differentiable functionality objectives and constraints. Our framework produces a wide spectrum of assembled articulated shapes, from trash-can wheels grafted onto car bodies to multi-segment lamps, gear-driven paddlers, and reconfigurable furniture, and delivers strong quantitative improvements over state-of-the-art baselines across geometric, kinematic, and functional metrics. By tightly coupling articulation-aware geometry matching with functionality-driven optimization, Kinematic Kitbashing bridges part-based shape modeling and functional assembly design, empowering rapid creation of interactive articulated assets. 

**Abstract (ZH)**: 基于运动的组件拼装自动框架：合成功能感知的articulated对象 

---
# Development of a Linear Guide-Rail Testbed for Physically Emulating ISAM Operations 

**Title (ZH)**: 基于ISAM操作物理仿真的一种线性导轨实验台开发 

**Authors**: Robert Muldrow, Channing Ludden, Christopher Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13005)  

**Abstract**: In-Space Servicing, Assembly, and Manufacturing (ISAM) is a set of emerging operations that provides several benefits to improve the longevity, capacity, mo- bility, and expandability of existing and future space assets. Serial robotic ma- nipulators are particularly vital in accomplishing ISAM operations, however, the complex perturbation forces and motions associated with movement of a robotic arm on a free-flying satellite presents a complex controls problem requiring addi- tional study. While many dynamical models are developed, experimentally test- ing and validating these models is challenging given that the models operate in space, where satellites have six-degrees-of-freedom (6-DOF). This paper attempts to resolve those challenges by presenting the design and development of a new hardware-in-the-loop (HIL) experimental testbed utilized to emulate ISAM. This emulation will be accomplished by means of a 6-DOF UR3e robotic arm attached to a satellite bus. This satellite bus is mounted to a 1-DOF guide-rail system, en- abling the satellite bus and robotic arm to move freely in one linear direction. This experimental ISAM emulation system will explore and validate models for space motion, serial robot manipulation, and contact mechanics. 

**Abstract (ZH)**: 太空服务、装配与制造（ISAM）操作是一系列新兴运营活动，旨在提高现有和未来太空资产的寿命、容量、机动性和扩展性。串行机器人操作器在完成ISAM操作中至关重要，然而，自由飞卫星上机器人臂运动相关的复杂扰动力和运动带来了复杂的控制问题，需要额外的研究。虽然开发了诸多动态模型，但由于卫星在具有六自由度（6-DOF）的空间中操作，实验性地测试和验证这些模型极具挑战性。本文旨在通过展示一种新的硬件在环（HIL）实验测试平台来解决这些挑战，该平台用于模拟ISAM。这种模拟将通过连接到卫星总线的6-DOF UR3e机器人臂来实现。该卫星总线安装在一个单自由度导轨系统上，使卫星总线和机器人臂能够在单一线性方向上自由移动。该实验ISAM模拟系统将探索和验证空间运动、串行机器人操作以及接触力学的模型。 

---
# UNCAP: Uncertainty-Guided Planning Using Natural Language Communication for Cooperative Autonomous Vehicles 

**Title (ZH)**: UNCAP：基于自然语言通信的不确定性指导规划方法及其在协同自动驾驶车辆中的应用 

**Authors**: Neel P. Bhatt, Po-han Li, Kushagra Gupta, Rohan Siva, Daniel Milan, Alexander T. Hogue, Sandeep P. Chinchali, David Fridovich-Keil, Zhangyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12992)  

**Abstract**: Safe large-scale coordination of multiple cooperative connected autonomous vehicles (CAVs) hinges on communication that is both efficient and interpretable. Existing approaches either rely on transmitting high-bandwidth raw sensor data streams or neglect perception and planning uncertainties inherent in shared data, resulting in systems that are neither scalable nor safe. To address these limitations, we propose Uncertainty-Guided Natural Language Cooperative Autonomous Planning (UNCAP), a vision-language model-based planning approach that enables CAVs to communicate via lightweight natural language messages while explicitly accounting for perception uncertainty in decision-making. UNCAP features a two-stage communication protocol: (i) an ego CAV first identifies the subset of vehicles most relevant for information exchange, and (ii) the selected CAVs then transmit messages that quantitatively express their perception uncertainty. By selectively fusing messages that maximize mutual information, this strategy allows the ego vehicle to integrate only the most relevant signals into its decision-making, improving both the scalability and reliability of cooperative planning. Experiments across diverse driving scenarios show a 63% reduction in communication bandwidth with a 31% increase in driving safety score, a 61% reduction in decision uncertainty, and a four-fold increase in collision distance margin during near-miss events. Project website: this https URL 

**Abstract (ZH)**: 安全的大规模协调多辆合作连接自主车辆依赖于高效且可解释的通信。现有的方法要么依赖于传输高带宽的原始传感器数据流，要么忽视共享数据中固有的感知和规划不确定性，导致系统既不具有可扩展性也不安全。为了解决这些限制，我们提出了一种基于视觉语言模型的合作自主规划方法——不确定性指导自然语言合作自主规划（UNCAP），该方法使自主车辆能够通过轻量级自然语言消息进行通信，并在决策过程中明确考虑感知不确定性。UNCAP 具有两阶段的通信协议：（i）一辆ego自主车辆首先识别最相关的车辆子集以进行信息交换，（ii）然后选择的自主车辆传输量化表示其感知不确定性的消息。通过选择性地融合最大化互信息的消息，该策略允许ego车辆仅整合最相关的信号进行决策，从而提高合作规划的可扩展性和可靠性。在多种驾驶场景下的实验表明，通信带宽减少了63%，驾驶安全性分数提高了31%，决策不确定性减少了61%，在接近碰撞事件中碰撞距离裕度提高了四倍。项目网站：这个 https URL。 

---
# Actron3D: Learning Actionable Neural Functions from Videos for Transferable Robotic Manipulation 

**Title (ZH)**: Actron3D：从视频学习可操作的神经函数以实现可转移的机器人 manipulation 

**Authors**: Anran Zhang, Hanzhi Chen, Yannick Burkhardt, Yao Zhong, Johannes Betz, Helen Oleynikova, Stefan Leutenegger  

**Link**: [PDF](https://arxiv.org/pdf/2510.12971)  

**Abstract**: We present Actron3D, a framework that enables robots to acquire transferable 6-DoF manipulation skills from just a few monocular, uncalibrated, RGB-only human videos. At its core lies the Neural Affordance Function, a compact object-centric representation that distills actionable cues from diverse uncalibrated videos-geometry, visual appearance, and affordance-into a lightweight neural network, forming a memory bank of manipulation skills. During deployment, we adopt a pipeline that retrieves relevant affordance functions and transfers precise 6-DoF manipulation policies via coarse-to-fine optimization, enabled by continuous queries to the multimodal features encoded in the neural functions. Experiments in both simulation and the real world demonstrate that Actron3D significantly outperforms prior methods, achieving a 14.9 percentage point improvement in average success rate across 13 tasks while requiring only 2-3 demonstration videos per task. 

**Abstract (ZH)**: Actron3D：一种从少量未校准单目RGB人类视频中获取可 Transfer 的6-DoF 手 manip 操作技能的框架 

---
# The Omega Turn: A General Turning Template for Elongate Robots 

**Title (ZH)**: Ω型转弯：长形机器人通用转弯模板 

**Authors**: Baxi Chong, Tianyu Wang, Kelimar Diaz, Christopher J. Pierce, Eva Erickson, Julian Whitman, Yuelin Deng, Esteban Flores, Ruijie Fu, Juntao He, Jianfeng Lin, Hang Lu, Guillaume Sartoretti, Howie Choset, Daniel I. Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2510.12970)  

**Abstract**: Elongate limbless robots have the potential to locomote through tightly packed spaces for applications such as search-and-rescue and industrial inspections. The capability to effectively and robustly maneuver elongate limbless robots is crucial to realize such potential. However, there has been limited research on turning strategies for such systems. To achieve effective and robust turning performance in cluttered spaces, we take inspiration from a microscopic nematode, C. elegans, which exhibits remarkable maneuverability in rheologically complex environments partially because of its ability to perform omega turns. Despite recent efforts to analyze omega turn kinematics, it remains unknown if there exists a wave equation sufficient to prescribe an omega turn, let alone its reconstruction on robot platforms. Here, using a comparative theory-biology approach, we prescribe the omega turn as a superposition of two traveling waves. With wave equations as a guideline, we design a controller for limbless robots enabling robust and effective turning behaviors in lab and cluttered field environments. Finally, we show that such omega turn controllers can also generalize to elongate multi-legged robots, demonstrating an alternative effective body-driven turning strategy for elongate robots, with and without limbs. 

**Abstract (ZH)**: 无肢延长型机器人有潜力在紧凑空间中进行运动，适用于搜索救援和工业检测等领域。有效地和稳健地操控无肢延长型机器人的能力对于实现这种潜力至关重要。然而，对于此类系统的转向策略研究有限。为了在拥挤空间中实现有效的稳健转向性能，我们从一种显微线虫C. elegans获得灵感，这种线虫在流变学复杂环境中表现出显著的机动性部分归因于它能够进行欧米伽转弯的能力。尽管最近对欧米伽转弯的运动学进行了分析，但仍不清楚是否存在一个波动方程能够充分规定欧米伽转弯，更不用说在机器人平台上实现其重构了。通过比较理论和生物学方法，我们将欧米伽转弯定义为两个行波的叠加。在波动方程的指导下，我们设计了一个控制器，使其能够使无肢机器人在实验室和拥挤环境中实现稳健且有效的转向行为。最后，我们展示了这样的欧米伽转弯控制器也可泛化到多腿延长型机器人，证明了延长型机器人（有腿或无腿）的有效体驱动转向策略。 

---
# Enhancing Sampling-based Planning with a Library of Paths 

**Title (ZH)**: 基于路径库增强采样基于Planning方法 

**Authors**: Michal Minařík, Vojtěch Vonásek, Robert Pěnička  

**Link**: [PDF](https://arxiv.org/pdf/2510.12962)  

**Abstract**: Path planning for 3D solid objects is a challenging problem, requiring a search in a six-dimensional configuration space, which is, nevertheless, essential in many robotic applications such as bin-picking and assembly. The commonly used sampling-based planners, such as Rapidly-exploring Random Trees, struggle with narrow passages where the sampling probability is low, increasing the time needed to find a solution. In scenarios like robotic bin-picking, various objects must be transported through the same environment. However, traditional planners start from scratch each time, losing valuable information gained during the planning process. We address this by using a library of past solutions, allowing the reuse of previous experiences even when planning for a new, previously unseen object. Paths for a set of objects are stored, and when planning for a new object, we find the most similar one in the library and use its paths as approximate solutions, adjusting for possible mutual transformations. The configuration space is then sampled along the approximate paths. Our method is tested in various narrow passage scenarios and compared with state-of-the-art methods from the OMPL library. Results show significant speed improvements (up to 85% decrease in the required time) of our method, often finding a solution in cases where the other planners fail. Our implementation of the proposed method is released as an open-source package. 

**Abstract (ZH)**: 3D实体对象的路径规划是一个具有挑战性的问题，需要在六维配置空间中进行搜索，尽管如此，在诸如料箱拣选和装配等许多机器人应用中仍然至关重要。常用的基于采样的规划器，如快速扩展随机树，难以处理采样概率低的狭窄通道，从而增加了找到解决方案所需的时间。在如机器人类料箱拣选场景中，各种对象必须通过相同的环境进行运输。然而，传统规划器每次从头开始，会丢失在规划过程中获得的重要信息。我们通过使用过去的解决方案库解决了这一问题，即使在为新且未见过的对象进行规划时，也能重用之前的经历。对于一组对象的路径进行存储，在为新对象规划路径时，我们找到库中最相似的路径并使用其路径作为近似解决方案，调整可能的相关变换。然后沿近似路径对配置空间进行采样。我们的方法在各种狭窄通道场景下进行了测试，并与OMPL库中的先进方法进行了比较。结果显示，我们的方法在所需时间上取得了显著的加速（最多可减少85%），通常能够在其他规划器失败的情况下找到解决方案。我们提出的该方法的实现已作为开源包发布。 

---
# Geometric Model Predictive Path Integral for Agile UAV Control with Online Collision Avoidance 

**Title (ZH)**: 几何模型预测路径积分在带在线碰撞 avoidance 的敏捷无人机控制中应用 

**Authors**: Pavel Pochobradský, Ondřej Procházka, Robert Pěnička, Vojtěch Vonásek, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2510.12924)  

**Abstract**: In this letter, we introduce Geometric Model Predictive Path Integral (GMPPI), a sampling-based controller capable of tracking agile trajectories while avoiding obstacles. In each iteration, GMPPI generates a large number of candidate rollout trajectories and then averages them to create a nominal control to be followed by the Unmanned Aerial Vehicle (UAV). We propose using geometric SE(3) control to generate part of the rollout trajectories, significantly increasing precision in agile flight. Furthermore, we introduce varying rollout simulation time step length and dynamic cost and noise parameters, vastly improving tracking performance of smooth and low-speed trajectories over an existing Model Predictive Path Integral (MPPI) implementation. Finally, we propose an integration of GMPPI with a stereo depth camera, enabling online obstacle avoidance at high speeds, a crucial step towards autonomous UAV flights in complex environments. The proposed controller can track simulated agile reference trajectories with position error similar to the geometric SE(3) controller. However, the same configuration of the proposed controller can avoid obstacles in a simulated forest environment at speeds of up to 13m/s, surpassing the performance of a state-of-the-art obstacle-aware planner. In real-world experiments, GMPPI retains the capability to track agile trajectories and avoids obstacles at speeds of up to 10m/s. 

**Abstract (ZH)**: 基于几何模型预测路径积分的采样控制器GMPPI及其应用 

---
# Gaussian Process Implicit Surfaces as Control Barrier Functions for Safe Robot Navigation 

**Title (ZH)**: 基于高斯过程隐表面的控制障碍函数在机器人安全导航中的应用 

**Authors**: Mouhyemen Khan, Tatsuya Ibuki, Abhijit Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.12919)  

**Abstract**: Level set methods underpin modern safety techniques such as control barrier functions (CBFs), while also serving as implicit surface representations for geometric shapes via distance fields. Inspired by these two paradigms, we propose a unified framework where the implicit surface itself acts as a CBF. We leverage Gaussian process (GP) implicit surface (GPIS) to represent the safety boundaries, using safety samples which are derived from sensor measurements to condition the GP. The GP posterior mean defines the implicit safety surface (safety belief), while the posterior variance provides a robust safety margin. Although GPs have favorable properties such as uncertainty estimation and analytical tractability, they scale cubically with data. To alleviate this issue, we develop a sparse solution called sparse Gaussian CBFs. To the best of our knowledge, GPIS have not been explicitly used to synthesize CBFs. We validate the approach on collision avoidance tasks in two settings: a simulated 7-DOF manipulator operating around the Stanford bunny, and a quadrotor navigating in 3D around a physical chair. In both cases, Gaussian CBFs (with and without sparsity) enable safe interaction and collision-free execution of trajectories that would otherwise intersect the objects. 

**Abstract (ZH)**: 基于拉格朗日方法的高斯过程控制障碍函数：统一框架及其应用 

---
# Learning to Grasp Anything by Playing with Random Toys 

**Title (ZH)**: 通过玩随机玩具学习抓取一切 

**Authors**: Dantong Niu, Yuvan Sharma, Baifeng Shi, Rachel Ding, Matteo Gioia, Haoru Xue, Henry Tsai, Konstantinos Kallidromitis, Anirudh Pai, Shankar Shastry, Trevor Darrell, Jitendra Malik, Roei Herzig  

**Link**: [PDF](https://arxiv.org/pdf/2510.12866)  

**Abstract**: Robotic manipulation policies often struggle to generalize to novel objects, limiting their real-world utility. In contrast, cognitive science suggests that children develop generalizable dexterous manipulation skills by mastering a small set of simple toys and then applying that knowledge to more complex items. Inspired by this, we study if similar generalization capabilities can also be achieved by robots. Our results indicate robots can learn generalizable grasping using randomly assembled objects that are composed from just four shape primitives: spheres, cuboids, cylinders, and rings. We show that training on these "toys" enables robust generalization to real-world objects, yielding strong zero-shot performance. Crucially, we find the key to this generalization is an object-centric visual representation induced by our proposed detection pooling mechanism. Evaluated in both simulation and on physical robots, our model achieves a 67% real-world grasping success rate on the YCB dataset, outperforming state-of-the-art approaches that rely on substantially more in-domain data. We further study how zero-shot generalization performance scales by varying the number and diversity of training toys and the demonstrations per toy. We believe this work offers a promising path to scalable and generalizable learning in robotic manipulation. Demonstration videos, code, checkpoints and our dataset are available on our project page: this https URL . 

**Abstract (ZH)**: 机器人抓取策略往往难以应用于新型物体，限制了其在实际环境中的应用价值。相比之下，认知科学表明，儿童通过掌握少量简单的玩具并将其知识应用于更复杂的物品来发展出可泛化的灵巧操作技能。受此启发，我们研究是否可以通过类似的方式令机器人也具备类似的泛化能力。结果表明，机器人可以使用仅由四种形状基元（球体、立方体、圆柱体和环体）组合而成的随机组装物体来学习可泛化的抓取策略。我们展示，通过训练来熟悉这些“玩具”，机器人可以实现对真实世界物体的稳健泛化，展现出强大的零样本性能。关键在于我们提出的检测聚池机制诱导的对象为中心的视觉表示。在模拟和物理机器人上进行评估，我们的模型在YCB数据集上的真实世界抓取成功率达到了67%，超越了依赖大量领域数据的最新方法。我们还探讨了零样本泛化性能随训练玩具数量和多样性以及每个玩具演示数量变化的规律。我们认为这项工作为可扩展和泛化的机器人操作学习提供了一条有前景的道路。更多详情，请参阅我们的项目页面: this https URL。 

---
# MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control 

**Title (ZH)**: MimicKit：运动模仿与控制的强化学习框架 

**Authors**: Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.13794)  

**Abstract**: MimicKit is an open-source framework for training motion controllers using motion imitation and reinforcement learning. The codebase provides implementations of commonly-used motion-imitation techniques and RL algorithms. This framework is intended to support research and applications in computer graphics and robotics by providing a unified training framework, along with standardized environment, agent, and data structures. The codebase is designed to be modular and easily configurable, enabling convenient modification and extension to new characters and tasks. The open-source codebase is available at: this https URL. 

**Abstract (ZH)**: MimicKit是基于运动模仿和强化学习训练运动控制器的开源框架 

---
# Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents 

**Title (ZH)**: 简化胞嵌入提高actor-critic代理的样本效率 

**Authors**: Johan Obando-Ceron, Walter Mayor, Samuel Lavoie, Scott Fujimoto, Aaron Courville, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2510.13704)  

**Abstract**: Recent works have proposed accelerating the wall-clock training time of actor-critic methods via the use of large-scale environment parallelization; unfortunately, these can sometimes still require large number of environment interactions to achieve a desired level of performance. Noting that well-structured representations can improve the generalization and sample efficiency of deep reinforcement learning (RL) agents, we propose the use of simplicial embeddings: lightweight representation layers that constrain embeddings to simplicial structures. This geometric inductive bias results in sparse and discrete features that stabilize critic bootstrapping and strengthen policy gradients. When applied to FastTD3, FastSAC, and PPO, simplicial embeddings consistently improve sample efficiency and final performance across a variety of continuous- and discrete-control environments, without any loss in runtime speed. 

**Abstract (ZH)**: 使用单纯形嵌入提高演员-评论家方法的样本效率和性能 

---
# Accelerated Feature Detectors for Visual SLAM: A Comparative Study of FPGA vs GPU 

**Title (ZH)**: 基于FPGA与GPU的加速特征检测器对比研究：视觉SLAM领域的探讨 

**Authors**: Ruiqi Ye, Mikel Luján  

**Link**: [PDF](https://arxiv.org/pdf/2510.13546)  

**Abstract**: Feature detection is a common yet time-consuming module in Simultaneous Localization and Mapping (SLAM) implementations, which are increasingly deployed on power-constrained platforms, such as drones. Graphics Processing Units (GPUs) have been a popular accelerator for computer vision in general, and feature detection and SLAM in particular.
On the other hand, System-on-Chips (SoCs) with integrated Field Programmable Gate Array (FPGA) are also widely available. This paper presents the first study of hardware-accelerated feature detectors considering a Visual SLAM (V-SLAM) pipeline. We offer new insights by comparing the best GPU-accelerated FAST, Harris, and SuperPoint implementations against the FPGA-accelerated counterparts on modern SoCs (Nvidia Jetson Orin and AMD Versal).
The evaluation shows that when using a non-learning-based feature detector such as FAST and Harris, their GPU implementations, and the GPU-accelerated V-SLAM can achieve better run-time performance and energy efficiency than the FAST and Harris FPGA implementations as well as the FPGA-accelerated V-SLAM. However, when considering a learning-based detector such as SuperPoint, its FPGA implementation can achieve better run-time performance and energy efficiency (up to 3.1$\times$ and 1.4$\times$ improvements, respectively) than the GPU implementation. The FPGA-accelerated V-SLAM can also achieve comparable run-time performance compared to the GPU-accelerated V-SLAM, with better FPS in 2 out of 5 dataset sequences. When considering the accuracy, the results show that the GPU-accelerated V-SLAM is more accurate than the FPGA-accelerated V-SLAM in general. Last but not least, the use of hardware acceleration for feature detection could further improve the performance of the V-SLAM pipeline by having the global bundle adjustment module invoked less frequently without sacrificing accuracy. 

**Abstract (ZH)**: 硬件加速在视觉SLAM管道中的特征检测研究：基于GPU与FPGA的比较 

---
# Through the Lens of Doubt: Robust and Efficient Uncertainty Estimation for Visual Place Recognition 

**Title (ZH)**: 怀疑之眼：视觉场所识别中的稳健高效不确定性估计 

**Authors**: Emily Miller, Michael Milford, Muhammad Burhan Hafez, SD Ramchurn, Shoaib Ehsan  

**Link**: [PDF](https://arxiv.org/pdf/2510.13464)  

**Abstract**: Visual Place Recognition (VPR) enables robots and autonomous vehicles to identify previously visited locations by matching current observations against a database of known places. However, VPR systems face significant challenges when deployed across varying visual environments, lighting conditions, seasonal changes, and viewpoints changes. Failure-critical VPR applications, such as loop closure detection in simultaneous localization and mapping (SLAM) pipelines, require robust estimation of place matching uncertainty. We propose three training-free uncertainty metrics that estimate prediction confidence by analyzing inherent statistical patterns in similarity scores from any existing VPR method. Similarity Distribution (SD) quantifies match distinctiveness by measuring score separation between candidates; Ratio Spread (RS) evaluates competitive ambiguity among top-scoring locations; and Statistical Uncertainty (SU) is a combination of SD and RS that provides a unified metric that generalizes across datasets and VPR methods without requiring validation data to select the optimal metric. All three metrics operate without additional model training, architectural modifications, or computationally expensive geometric verification. Comprehensive evaluation across nine state-of-the-art VPR methods and six benchmark datasets confirms that our metrics excel at discriminating between correct and incorrect VPR matches, and consistently outperform existing approaches while maintaining negligible computational overhead, making it deployable for real-time robotic applications across varied environmental conditions with improved precision-recall performance. 

**Abstract (ZH)**: 视觉位置识别(VPR)使机器人和自动驾驶车辆能够通过将当前观察与已知地点的数据库进行匹配来识别先前访问的位置。然而，当部署在变化的视觉环境、光照条件、季节变化和视角变化中时，VPR系统面临着显著挑战。对于关键性的VPR应用，如同时定位与 mapping（SLAM）管道中的环路闭合检测，需要 robust 的位置匹配不确定性估计。我们提出了一种无需训练的不确定性度量方法，通过分析任何现有VPR方法的相似性分数中的固有统计模式来估计预测置信度。相似性分布(SD)通过测量候选者的分数分离来量化匹配的独特性；竞争性模糊度比(RS)评估最高分位置之间的竞争模糊度；统计不确定性(SU)是SD和RS的组合，提供了一个统一的度量标准，可以在无需验证数据选择最优度量的情况下适用于不同数据集和VPR方法。所有三个度量标准无需额外的模型训练、架构修改或昂贵的几何验证。在九种最先进的VPR方法和六个基准数据集上的综合评估表明，我们的度量方法在区分正确和错误的VPR匹配方面表现出色，并且在一致性上优于现有方法，同时保持了微乎其微的计算开销，使其适用于具有改进精度召回性能的多样化环境条件下的实时机器人应用。 

---
# Physics-Informed Neural Network Modeling of Vehicle Collision Dynamics in Precision Immobilization Technique Maneuvers 

**Title (ZH)**: 基于物理的神经网络在精确 immobilization 技术操作中车辆碰撞动力学建模 

**Authors**: Yangye Jiang, Jiachen Wang, Daofei Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13461)  

**Abstract**: Accurate prediction of vehicle collision dynamics is crucial for advanced safety systems and post-impact control applications, yet existing methods face inherent trade-offs among computational efficiency, prediction accuracy, and data requirements. This paper proposes a dual Physics-Informed Neural Network framework addressing these challenges through two complementary networks. The first network integrates Gaussian Mixture Models with PINN architecture to learn impact force distributions from finite element analysis data while enforcing momentum conservation and energy consistency constraints. The second network employs an adaptive PINN with dynamic constraint weighting to predict post-collision vehicle dynamics, featuring an adaptive physics guard layer that prevents unrealistic predictions whil e preserving data-driven learning capabilities. The framework incorporates uncertainty quantification through time-varying parameters and enables rapid adaptation via fine-tuning strategies. Validation demonstrates significant improvements: the impact force model achieves relative errors below 15.0% for force prediction on finite element analysis (FEA) datasets, while the vehicle dynamics model reduces average trajectory prediction error by 63.6% compared to traditional four-degree-of-freedom models in scaled vehicle experiments. The integrated system maintains millisecond-level computational efficiency suitable for real-time applications while providing probabilistic confidence bounds essential for safety-critical control. Comprehensive validation through FEA simulation, dynamic modeling, and scaled vehicle experiments confirms the framework's effectiveness for Precision Immobilization Technique scenarios and general collision dynamics prediction. 

**Abstract (ZH)**: 精确预测车辆碰撞动力学对于高级安全系统和碰撞后控制应用至关重要，但现有方法在计算效率、预测准确性和数据需求之间存在固有的权衡。本文提出了一种双物理信息神经网络框架，通过两个互补网络解决这些挑战。第一个网络结合了高斯混合模型和物理信息神经网络架构，利用有限元分析数据学习冲击力分布，并强制遵循动量守恒和能量一致性约束。第二个网络采用自适应物理信息神经网络并动态约束权重预测碰撞后的车辆动力学，该网络具有自适应物理防护层，可以防止不现实的预测同时保留基于数据的學習能力。该框架通过时间变化参数进行不确定性量化，并通过微调策略实现快速适应。验证结果显示，冲击力模型在有限元分析数据集上的力预测相对误差低于15.0%，而车辆动力学模型在比例车辆实验中将平均轨迹预测误差降低了63.6%，相比传统的四自由度模型。综合验证通过有限元模拟、动力学建模和比例车辆实验证实了该框架在精确实施技术和一般碰撞动力学预测方面的有效性。 

---
# A New Perspective on Transformers in Online Reinforcement Learning for Continuous Control 

**Title (ZH)**: 在线连续控制中变换器的新视角 

**Authors**: Nikita Kachaev, Daniil Zelezetsky, Egor Cherepanov, Alexey K. Kovelev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2510.13367)  

**Abstract**: Despite their effectiveness and popularity in offline or model-based reinforcement learning (RL), transformers remain underexplored in online model-free RL due to their sensitivity to training setups and model design decisions such as how to structure the policy and value networks, share components, or handle temporal information. In this paper, we show that transformers can be strong baselines for continuous control in online model-free RL. We investigate key design questions: how to condition inputs, share components between actor and critic, and slice sequential data for training. Our experiments reveal stable architectural and training strategies enabling competitive performance across fully and partially observable tasks, and in both vector- and image-based settings. These findings offer practical guidance for applying transformers in online RL. 

**Abstract (ZH)**: 尽管在离线或基于模型的强化学习（RL）中， Transformers 已显示出其有效性并广受青睐，但它们在在线模型自由 RL 中的应用仍然相对不足，这部分归因于 Transformer 对训练设置和模型设计决策的高度敏感性，如如何结构化策略网络和价值网络、共享组件或处理时间信息。在本文中，我们展示了 Transformers 可以作为在线模型自由 RL 中连续控制任务的有效基线。我们探讨了关键设计问题：如何条件化输入、在决策者和批评者之间共享组件，以及如何按顺序分割数据进行训练。实验结果揭示了稳定的架构和训练策略，使 Transformers 在完全可观测和部分可观测任务，以及向量表示和图像表示的不同设置中都能获得具有竞争力的性能。这些发现为在在线 RL 中应用 Transformers 提供了实用指导。 

---
# Safe Driving in Occluded Environments 

**Title (ZH)**: 遮挡环境下安全驾驶 

**Authors**: Zhuoyuan Wang, Tongyao Jia, Pharuj Rajborirug, Neeraj Ramesh, Hiroyuki Okuda, Tatsuya Suzuki, Soummya Kar, Yorie Nakahira  

**Link**: [PDF](https://arxiv.org/pdf/2510.13114)  

**Abstract**: Ensuring safe autonomous driving in the presence of occlusions poses a significant challenge in its policy design. While existing model-driven control techniques based on set invariance can handle visible risks, occlusions create latent risks in which safety-critical states are not observable. Data-driven techniques also struggle to handle latent risks because direct mappings from risk-critical objects in sensor inputs to safe actions cannot be learned without visible risk-critical objects. Motivated by these challenges, in this paper, we propose a probabilistic safety certificate for latent risk. Our key technical enabler is the application of probabilistic invariance: It relaxes the strict observability requirements imposed by set-invariance methods that demand the knowledge of risk-critical states. The proposed techniques provide linear action constraints that confine the latent risk probability within tolerance. Such constraints can be integrated into model predictive controllers or embedded in data-driven policies to mitigate latent risks. The proposed method is tested using the CARLA simulator and compared with a few existing techniques. The theoretical and empirical analysis jointly demonstrate that the proposed methods assure long-term safety in real-time control in occluded environments without being overly conservative and with transparency to exposed risks. 

**Abstract (ZH)**: 确保遮挡环境下安全自主驾驶的政策设计面临显著挑战。现有基于集合不变性的模型驱动控制技术可以处理可见风险，但遮挡创造了潜在风险，在这种情况下，安全关键状态无法观测。基于数据的技术也难以处理潜在风险，因为无法直接从传感器输入中的风险关键对象到安全行动学习映射关系，除非存在可见的风险关键对象。受此挑战的启发，本文提出了一种针对潜在风险的概率安全证书。我们的关键技术使能器是概率不变性的应用：它放宽了集合不变性方法对严格可观测性的要求，后者要求必须知道风险关键状态。所提出的技术提供了线性的动作约束，将潜在风险概率限制在可接受的范围内。这些约束可以集成到模型预测控制器中或嵌入到数据驱动策略中，以减轻潜在风险。所提出的方法在CARLA仿真环境下进行了测试，并与几种现有技术进行了比较。理论和实证分析共同证明，所提出的方法能够在遮挡环境中实时控制中确保长期安全，同时不显得过于保守，并且对暴露的风险具有透明度。 

---
# DriveCritic: Towards Context-Aware, Human-Aligned Evaluation for Autonomous Driving with Vision-Language Models 

**Title (ZH)**: DriveCritic: 面向情境感知与人类价值观对齐的自动驾驶评估方法研究 

**Authors**: Jingyu Song, Zhenxin Li, Shiyi Lan, Xinglong Sun, Nadine Chang, Maying Shen, Joshua Chen, Katherine A. Skinner, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13108)  

**Abstract**: Benchmarking autonomous driving planners to align with human judgment remains a critical challenge, as state-of-the-art metrics like the Extended Predictive Driver Model Score (EPDMS) lack context awareness in nuanced scenarios. To address this, we introduce DriveCritic, a novel framework featuring two key contributions: the DriveCritic dataset, a curated collection of challenging scenarios where context is critical for correct judgment and annotated with pairwise human preferences, and the DriveCritic model, a Vision-Language Model (VLM) based evaluator. Fine-tuned using a two-stage supervised and reinforcement learning pipeline, the DriveCritic model learns to adjudicate between trajectory pairs by integrating visual and symbolic context. Experiments show DriveCritic significantly outperforms existing metrics and baselines in matching human preferences and demonstrates strong context awareness. Overall, our work provides a more reliable, human-aligned foundation to evaluating autonomous driving systems. 

**Abstract (ZH)**: 基于DriveCritic框架评估自主驾驶规划者以与人类判断一致仍是一项关键挑战，现有先进的指标如扩展预测驾驶模型评分（EPDMS）在细腻场景中缺乏上下文意识。为解决这一问题，我们引入了DriveCritic，这是一个包含两项关键贡献的新框架：DriveCritic数据集，这是一个经过精心筛选的包含关键上下文场景的集合，并标注了两两的人类偏好；以及DriveCritic模型，这是一个基于视觉-语言模型（VLM）的评估器。通过两阶段监督学习和强化学习管道进行微调后，DriveCritic模型学习通过整合视觉和符号上下文来判断轨迹对。实验表明，DriveCritic在匹配人类偏好和展现强大的上下文意识方面显著优于现有指标和基线。总体而言，我们的工作为评估自主驾驶系统提供了一个更加可靠、与人类判断一致的基础。 

---
# Comparison of Forced and Unforced Rendezvous, Proximity Operations, and Docking Under Model Mismatch 

**Title (ZH)**: 模型不符情况下强制和非强制会合、接近操作及对接的比较研究 

**Authors**: Robert Muldrow, Channing Ludden, Christopher Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13004)  

**Abstract**: This paper compares the required fuel usage for forced and unforced motion of a chaser satellite engaged in Rendezvous, Proximity Operations, and Docking (RPOD) maneuvers. Improved RPOD models are vital, particularly as the space industry expands and demands for improved fuel efficiency, cost effectiveness, and mission life span increase. This paper specifically examines the Clohessy- Wiltshire (CW) Equations and the extent of model mismatch by comparing pre- dicted trajectories from this model with a more computationally complex, higher fidelity RPOD model. This paper assesses several test cases of similar mission parameters, in each case comparing natural motion circumnavigation (NMC) with comparable forced motion circumnavigation. The Guidance, Navigation, and Con- trol (GNC) impulse maneuvers required to maintain the supposedly zero fuel CW trajectories is representative of the extent of CW model mismatch. This paper demonstrates that unforced motions are not inherently more fuel efficient than forced motions, thus permitting extended orbital operations given the higher fuel efficiency. 

**Abstract (ZH)**: 本文比较了强制运动和非强制运动下追逐卫星在接近、交会和对接（RPOD）操作中所需的燃料使用情况。随着太空行业的扩展，对改进燃料效率、成本效益和任务寿命的需求不断增加，改进的RPOD模型显得尤为重要。本文特别研究了Clohessy-Wiltshire (CW) 方程及其模型不匹配程度，通过将该模型预测轨迹与更高保真度的RPOD模型进行比较。本文评估了多个具有相似任务参数的测试案例，在每个案例中比较了自然运动环航与类似条件下的强制运动环航。用于维持假定零燃料CW轨迹所需的制导、导航与控制（GNC）冲量机动代表了CW模型不匹配的程度。本文表明，非强制运动并非天然比强制运动更省燃料，因此在提供更高燃料效率的情况下，允许延长轨道操作时间。 

---
# SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms 

**Title (ZH)**: SimULi: 基于无中心变换的实时LiDAR和相机仿真 

**Authors**: Haithem Turki, Qi Wu, Xin Kang, Janick Martinez Esturo, Shengyu Huang, Ruilong Li, Zan Gojcic, Riccardo de Lutio  

**Link**: [PDF](https://arxiv.org/pdf/2510.12901)  

**Abstract**: Rigorous testing of autonomous robots, such as self-driving vehicles, is essential to ensure their safety in real-world deployments. This requires building high-fidelity simulators to test scenarios beyond those that can be safely or exhaustively collected in the real-world. Existing neural rendering methods based on NeRF and 3DGS hold promise but suffer from low rendering speeds or can only render pinhole camera models, hindering their suitability to applications that commonly require high-distortion lenses and LiDAR data. Multi-sensor simulation poses additional challenges as existing methods handle cross-sensor inconsistencies by favoring the quality of one modality at the expense of others. To overcome these limitations, we propose SimULi, the first method capable of rendering arbitrary camera models and LiDAR data in real-time. Our method extends 3DGUT, which natively supports complex camera models, with LiDAR support, via an automated tiling strategy for arbitrary spinning LiDAR models and ray-based culling. To address cross-sensor inconsistencies, we design a factorized 3D Gaussian representation and anchoring strategy that reduces mean camera and depth error by up to 40% compared to existing methods. SimULi renders 10-20x faster than ray tracing approaches and 1.5-10x faster than prior rasterization-based work (and handles a wider range of camera models). When evaluated on two widely benchmarked autonomous driving datasets, SimULi matches or exceeds the fidelity of existing state-of-the-art methods across numerous camera and LiDAR metrics. 

**Abstract (ZH)**: 严格测试自主机器人，如自动驾驶车辆，对于确保其在实际部署中的安全性至关重要。这需要构建高保真模拟器来测试超出现实世界中可以安全或耗尽性收集的场景。现有基于NeRF和3DGS的神经渲染方法前景广阔，但存在渲染速度低或只能渲染针孔相机模型的问题，阻碍了它们在常需高失真镜头和LiDAR数据的应用中的适用性。多传感器模拟还提出了额外挑战，现有方法通过优先考虑某种模态的质量，而牺牲其他模态来处理跨传感器不一致性。为克服这些局限性，我们提出了SimULi，这是首款能够实时渲染任意相机模型和LiDAR数据的方法。我们的方法通过自动切片策略扩展了本来就支持复杂相机模型的3DGUT，并通过基于光线裁剪策略为任意旋转的LiDAR模型提供了LiDAR支持。为解决跨传感器不一致性问题，我们设计了一种因子化的3D高斯表示和锚定策略，相比现有方法将平均相机误差和深度误差降低了最多40%。SimULi的渲染速度比光线跟踪方法快10-20倍，比之前的基于光栅化的工作快1.5-10倍（并且能够处理更广泛的相机模型）。当在两个广泛基准测试的自动驾驶数据集上进行评估时，SimULi在多个相机和LiDAR指标上与现有最先进的方法相当或更优。 

---
# VLURes: Benchmarking VLM Visual and Linguistic Understanding in Low-Resource Languages 

**Title (ZH)**: VLURes: 低资源语言视觉与语言理解基准测试 

**Authors**: Jesse Atuhurra, Iqra Ali, Tomoya Iwakura, Hidetaka Kamigaito, Tatsuya Hiraoka  

**Link**: [PDF](https://arxiv.org/pdf/2510.12845)  

**Abstract**: Vision Language Models (VLMs) are pivotal for advancing perception in intelligent agents. Yet, evaluation of VLMs remains limited to predominantly English-centric benchmarks in which the image-text pairs comprise short texts. To evaluate VLM fine-grained abilities, in four languages under long-text settings, we introduce a novel multilingual benchmark VLURes featuring eight vision-and-language tasks, and a pioneering unrelatedness task, to probe the fine-grained Visual and Linguistic Understanding capabilities of VLMs across English, Japanese, and low-resource languages, Swahili, and Urdu. Our datasets, curated from web resources in the target language, encompass ten diverse image categories and rich textual context, introducing valuable vision-language resources for Swahili and Urdu. By prompting VLMs to generate responses and rationales, evaluated automatically and by native speakers, we uncover performance disparities across languages and tasks critical to intelligent agents, such as object recognition, scene understanding, and relationship understanding. We conducted evaluations of ten VLMs with VLURes. The best performing model, GPT-4o, achieves an overall accuracy of 90.8% and lags human performance by 6.7%, though the gap is larger for open-source models. The gap highlights VLURes' critical role in developing intelligent agents to tackle multi-modal visual reasoning. 

**Abstract (ZH)**: Vision Language Models (VLMs)在智能代理感知能力提升中的关键作用依然受限于主要以英语为中心的基准测试，这些基准测试中的图像-文本对大多由简短的文本组成。为了在长文本设置下评估VLM的细粒度能力，我们引入了包括八项视觉和语言任务及一项开创性的无关性任务的新多语言基准VLURes，以探究VLM在英语、日语及低资源语言斯瓦希里语和乌尔都语中的视觉和语言理解能力。我们从目标语言的网络资源中精心策划的数据集涵盖了十个不同的图像类别和丰富的文本背景，为斯瓦希里语和乌尔都语引入了宝贵的视觉语言资源。通过促使VLM生成响应和解释，并由自动评估和母语者评估，我们揭示了不同语言和任务的性能差异，这些差异对于智能代理如物体识别、场景理解和关系理解至关重要。我们对十种VLM进行了VLURes评估。表现最佳的模型GPT-4o的整体准确率为90.8%，落后人类性能6.7%，尽管开源模型之间的差距更大。这一差距突显了VLURes在开发能够应对多模态视觉推理的智能代理方面的重要性。 

---
