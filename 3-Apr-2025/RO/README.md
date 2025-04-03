# Slot-Level Robotic Placement via Visual Imitation from Single Human Video 

**Title (ZH)**: 基于单人视频的视觉模仿-slot级机器人放置 

**Authors**: Dandan Shan, Kaichun Mo, Wei Yang, Yu-Wei Chao, David Fouhey, Dieter Fox, Arsalan Mousavian  

**Link**: [PDF](https://arxiv.org/pdf/2504.01959)  

**Abstract**: The majority of modern robot learning methods focus on learning a set of pre-defined tasks with limited or no generalization to new tasks. Extending the robot skillset to novel tasks involves gathering an extensive amount of training data for additional tasks. In this paper, we address the problem of teaching new tasks to robots using human demonstration videos for repetitive tasks (e.g., packing). This task requires understanding the human video to identify which object is being manipulated (the pick object) and where it is being placed (the placement slot). In addition, it needs to re-identify the pick object and the placement slots during inference along with the relative poses to enable robot execution of the task. To tackle this, we propose SLeRP, a modular system that leverages several advanced visual foundation models and a novel slot-level placement detector Slot-Net, eliminating the need for expensive video demonstrations for training. We evaluate our system using a new benchmark of real-world videos. The evaluation results show that SLeRP outperforms several baselines and can be deployed on a real robot. 

**Abstract (ZH)**: 利用人类示范视频教学机器人执行新任务：SLeRP模块化系统及其应用 

---
# Strengthening Multi-Robot Systems for SAR: Co-Designing Robotics and Communication Towards 6G 

**Title (ZH)**: 增强搜救多机器人系统：面向6G的机器人与通信协同设计 

**Authors**: Juan Bravo-Arrabal, Ricardo Vázquez-Martín, J.J. Fernández-Lozano, Alfonso García-Cerezo  

**Link**: [PDF](https://arxiv.org/pdf/2504.01940)  

**Abstract**: This paper presents field-tested use cases from Search and Rescue (SAR) missions, highlighting the co-design of mobile robots and communication systems to support Edge-Cloud architectures based on 5G Standalone (SA). The main goal is to contribute to the effective cooperation of multiple robots and first responders. Our field experience includes the development of Hybrid Wireless Sensor Networks (H-WSNs) for risk and victim detection, smartphones integrated into the Robot Operating System (ROS) as Edge devices for mission requests and path planning, real-time Simultaneous Localization and Mapping (SLAM) via Multi-Access Edge Computing (MEC), and implementation of Uncrewed Ground Vehicles (UGVs) for victim evacuation in different navigation modes. These experiments, conducted in collaboration with actual first responders, underscore the need for intelligent network resource management, balancing low-latency and high-bandwidth demands. Network slicing is key to ensuring critical emergency services are performed despite challenging communication conditions. The paper identifies architectural needs, lessons learned, and challenges to be addressed by 6G technologies to enhance emergency response capabilities. 

**Abstract (ZH)**: 基于5G独立组网（SA）的边缘-云架构下搜救任务的现场测试案例与移动机器人及通信系统协同设计研究：面向多机器人与应急响应者的有效协作 

---
# A novel gesture interaction control method for rehabilitation lower extremity exoskeleton 

**Title (ZH)**: 一种用于康复下肢外骨骼的新颖手势交互控制方法 

**Authors**: Shuang Qiu, Zhongcai Pei, Chen Wang, Jing Zhang, Zhiyong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01888)  

**Abstract**: With the rapid development of Rehabilitation Lower Extremity Robotic Exoskeletons (RLEEX) technology, significant advancements have been made in Human-Robot Interaction (HRI) methods. These include traditional physical HRI methods that are easily recognizable and various bio-electrical signal-based HRI methods that can visualize and predict actions. However, most of these HRI methods are contact-based, facing challenges such as operational complexity, sensitivity to interference, risks associated with implantable devices, and, most importantly, limitations in comfort. These challenges render the interaction less intuitive and natural, which can negatively impact patient motivation for rehabilitation. To address these issues, this paper proposes a novel non-contact gesture interaction control method for RLEEX, based on RGB monocular camera depth estimation. This method integrates three key steps: detecting keypoints, recognizing gestures, and assessing distance, thereby applying gesture information and augmented reality triggering technology to control gait movements of RLEEX. Results indicate that this approach provides a feasible solution to the problems of poor comfort, low reliability, and high latency in HRI for RLEEX platforms. Specifically, it achieves a gesture-controlled exoskeleton motion accuracy of 94.11\% and an average system response time of 0.615 seconds through non-contact HRI. The proposed non-contact HRI method represents a pioneering advancement in control interactions for RLEEX, paving the way for further exploration and development in this field. 

**Abstract (ZH)**: 基于RGB单目摄像头深度估计的非接触手势交互控制方法在康复下肢外骨科技术中的应用 

---
# Corner-Grasp: Multi-Action Grasp Detection and Active Gripper Adaptation for Grasping in Cluttered Environments 

**Title (ZH)**: 角落抓取：在拥挤环境中多动作抓取检测与主动 gripper 调适 

**Authors**: Yeong Gwang Son, Seunghwan Um, Juyong Hong, Tat Hieu Bui, Hyouk Ryeol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01861)  

**Abstract**: Robotic grasping is an essential capability, playing a critical role in enabling robots to physically interact with their surroundings. Despite extensive research, challenges remain due to the diverse shapes and properties of target objects, inaccuracies in sensing, and potential collisions with the environment. In this work, we propose a method for effectively grasping in cluttered bin-picking environments where these challenges intersect. We utilize a multi-functional gripper that combines both suction and finger grasping to handle a wide range of objects. We also present an active gripper adaptation strategy to minimize collisions between the gripper hardware and the surrounding environment by actively leveraging the reciprocating suction cup and reconfigurable finger motion. To fully utilize the gripper's capabilities, we built a neural network that detects suction and finger grasp points from a single input RGB-D image. This network is trained using a larger-scale synthetic dataset generated from simulation. In addition to this, we propose an efficient approach to constructing a real-world dataset that facilitates grasp point detection on various objects with diverse characteristics. Experiment results show that the proposed method can grasp objects in cluttered bin-picking scenarios and prevent collisions with environmental constraints such as a corner of the bin. Our proposed method demonstrated its effectiveness in the 9th Robotic Grasping and Manipulation Competition (RGMC) held at ICRA 2024. 

**Abstract (ZH)**: 一种用于杂乱环境bin-picking的有效抓取方法 

---
# Virtual Target Trajectory Prediction for Stochastic Targets 

**Title (ZH)**: 随机目标的虚拟目标轨迹预测 

**Authors**: Marc Schneider, Renato Loureiro, Torbjørn Cunis, Walter Fichter  

**Link**: [PDF](https://arxiv.org/pdf/2504.01851)  

**Abstract**: Trajectory prediction of other vehicles is crucial for autonomous vehicles, with applications from missile guidance to UAV collision avoidance. Typically, target trajectories are assumed deterministic, but real-world aerial vehicles exhibit stochastic behavior, such as evasive maneuvers or gliders circling in thermals. This paper uses Conditional Normalizing Flows, an unsupervised Machine Learning technique, to learn and predict the stochastic behavior of targets of guided missiles using trajectory data. The trained model predicts the distribution of future target positions based on initial conditions and parameters of the dynamics. Samples from this distribution are clustered using a time series k-means algorithm to generate representative trajectories, termed virtual targets. The method is fast and target-agnostic, requiring only training data in the form of target trajectories. Thus, it serves as a drop-in replacement for deterministic trajectory predictions in guidance laws and path planning. Simulated scenarios demonstrate the approach's effectiveness for aerial vehicles with random maneuvers, bridging the gap between deterministic predictions and stochastic reality, advancing guidance and control algorithms for autonomous vehicles. 

**Abstract (ZH)**: 其他车辆轨迹预测对于自主车辆至关重要，其应用范围从导弹引导到无人机避碰。通常，目标轨迹被视为确定性的，但实际的空中车辆表现出随机行为，如规避机动或滑翔机在热气流中盘旋。本文使用条件归一化流，这是一种无监督机器学习技术，通过轨迹数据学习并预测引导导弹目标的随机行为。训练后的模型根据初始条件和动力学参数预测未来目标位置的分布。从该分布中采样的数据使用时间序列k-means算法聚类以生成代表性的轨迹，称为虚拟目标。该方法快速且目标无关，仅需目标轨迹形式的训练数据。因此，它可以用作制导律和路径规划中确定性轨迹预测的即插即用替代方案。仿真场景表明，该方法对于具有随机机动的空中车辆有效，填补了确定性预测与随机现实之间的差距，推动了自主车辆的制导和控制算法的发展。 

---
# SOLAQUA: SINTEF Ocean Large Aquaculture Robotics Dataset 

**Title (ZH)**: SOLAQUA: SINTEF海洋大型水产机器人数据集 

**Authors**: Sveinung Johan Ohrem, Bent Haugaløkken, Eleni Kelasidi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01790)  

**Abstract**: This paper presents a dataset gathered with an underwater robot in a sea-based aquaculture setting. Data was gathered from an operational fish farm and includes data from sensors such as the Waterlinked A50 DVL, the Nortek Nucleus 1000 DVL, Sonardyne Micro Ranger 2 USBL, Sonoptix Mulitbeam Sonar, mono and stereo cameras, and vehicle sensor data such as power usage, IMU, pressure, temperature, and more. Data acquisition is performed during both manual and autonomous traversal of the net pen structure. The collected vision data is of undamaged nets with some fish and marine growth presence, and it is expected that both the research community and the aquaculture industry will benefit greatly from the utilization of the proposed SOLAQUA dataset. 

**Abstract (ZH)**: 本文介绍了在基于海洋的鱼场环境中使用水下机器人收集的一组数据集。数据采集自一个运营中的鱼场，包括Waterlinked A50 DVL、Nortek Nucleus 1000 DVL、Sonardyne Micro Ranger 2 USBL、Sonoptix 多波束声纳、单目和立体摄像机以及车辆传感器数据，如电力使用、IMU、压力、温度等。数据采集在手动和自主穿越网箱结构期间进行。收集的视觉数据包含了无损的网具、一些鱼类和海洋生物生长的迹象，预计SOLAQUA数据集的提议将极大地造福研究界和水产养殖业。 

---
# TransforMerger: Transformer-based Voice-Gesture Fusion for Robust Human-Robot Communication 

**Title (ZH)**: TransforMerger: 基于Transformer的语音-手势融合技术以增强人机通信 

**Authors**: Petr Vanc, Karla Stepanova  

**Link**: [PDF](https://arxiv.org/pdf/2504.01708)  

**Abstract**: As human-robot collaboration advances, natural and flexible communication methods are essential for effective robot control. Traditional methods relying on a single modality or rigid rules struggle with noisy or misaligned data as well as with object descriptions that do not perfectly fit the predefined object names (e.g. 'Pick that red object'). We introduce TransforMerger, a transformer-based reasoning model that infers a structured action command for robotic manipulation based on fused voice and gesture inputs. Our approach merges multimodal data into a single unified sentence, which is then processed by the language model. We employ probabilistic embeddings to handle uncertainty and we integrate contextual scene understanding to resolve ambiguous references (e.g., gestures pointing to multiple objects or vague verbal cues like "this"). We evaluate TransforMerger in simulated and real-world experiments, demonstrating its robustness to noise, misalignment, and missing information. Our results show that TransforMerger outperforms deterministic baselines, especially in scenarios requiring more contextual knowledge, enabling more robust and flexible human-robot communication. Code and datasets are available at: this http URL. 

**Abstract (ZH)**: 随着人机协作的发展，自然灵活的通信方法对于有效的机器人控制至关重要。传统的单一模态或刚性规则的方法难以处理噪声或错位的数据，以及不符合预定义对象名称的对象描述（例如，“捡起那个红色的对象”）。我们提出了基于变换器的TransforMerger模型，该模型基于融合的语音和手势输入推断出结构化的动作指令。我们的方法将多模态数据合并为一个统一的句子，然后由语言模型处理。我们采用概率嵌入来处理不确定性，并结合上下文场景理解来解决模棱两可的引用（例如，手势指向多个对象或模糊的口头提示“这个”）。我们在模拟和真实世界实验中评估了TransforMerger，证明了其对噪声、错位和缺失信息的鲁棒性。实验结果表明，TransforMerger在需要更多上下文知识的场景中优于确定性基线，从而实现了更 robust 和灵活的人机通信。代码和数据集可在以下链接获取：this http URL。 

---
# Anticipating Degradation: A Predictive Approach to Fault Tolerance in Robot Swarms 

**Title (ZH)**: 预见退化：机器人蜂群容错的预测方法 

**Authors**: James O'Keeffe  

**Link**: [PDF](https://arxiv.org/pdf/2504.01594)  

**Abstract**: An active approach to fault tolerance is essential for robot swarms to achieve long-term autonomy. Previous efforts have focused on responding to spontaneous electro-mechanical faults and failures. However, many faults occur gradually over time. Waiting until such faults have manifested as failures before addressing them is both inefficient and unsustainable in a variety of scenarios. This work argues that the principles of predictive maintenance, in which potential faults are resolved before they hinder the operation of the swarm, offer a promising means of achieving long-term fault tolerance. This is a novel approach to swarm fault tolerance, which is shown to give a comparable or improved performance when tested against a reactive approach in almost all cases tested. 

**Abstract (ZH)**: 一种积极的方法对于实现机器人 swarm 的长期自主性来说是实现容错所不可或缺的。预测性维护原理在 swarm 故障容忍中具有潜在的前景，该方法在几乎所有测试案例中表现出与反应性方法相当或更好的性能。 

---
# Building Knowledge from Interactions: An LLM-Based Architecture for Adaptive Tutoring and Social Reasoning 

**Title (ZH)**: 基于交互构建知识：一种用于自适应辅导和社会推理的LLM基础架构 

**Authors**: Luca Garello, Giulia Belgiovine, Gabriele Russo, Francesco Rea, Alessandra Sciutti  

**Link**: [PDF](https://arxiv.org/pdf/2504.01588)  

**Abstract**: Integrating robotics into everyday scenarios like tutoring or physical training requires robots capable of adaptive, socially engaging, and goal-oriented interactions. While Large Language Models show promise in human-like communication, their standalone use is hindered by memory constraints and contextual incoherence. This work presents a multimodal, cognitively inspired framework that enhances LLM-based autonomous decision-making in social and task-oriented Human-Robot Interaction. Specifically, we develop an LLM-based agent for a robot trainer, balancing social conversation with task guidance and goal-driven motivation. To further enhance autonomy and personalization, we introduce a memory system for selecting, storing and retrieving experiences, facilitating generalized reasoning based on knowledge built across different interactions. A preliminary HRI user study and offline experiments with a synthetic dataset validate our approach, demonstrating the system's ability to manage complex interactions, autonomously drive training tasks, and build and retrieve contextual memories, advancing socially intelligent robotics. 

**Abstract (ZH)**: 将机器人融入像辅导或体能训练等日常生活场景需要具备适应性、社会互动性和目标导向性交互能力的机器人。虽然大型语言模型在拟人化交流方面显示出潜力，但它们独立使用时受限于记忆容量和上下文一致性问题。本文提出了一种多模态、受认知启发的框架，以增强基于大型语言模型的自主决策能力，应用于社会性和任务导向性的人机交互。具体来说，我们开发了一个基于大型语言模型的机器人训练员代理，平衡社交对话与任务指导及目标驱动的激励。为进一步增强自主性和个性化，我们引入了记忆系统，用于选择、存储和检索经验，从而基于不同交互中积累的知识进行泛化的推理。初步的人机交互用户研究和使用合成数据集的离线实验证明了该方法的有效性，展示了系统处理复杂交互、自主驱动训练任务以及构建和检索上下文记忆的能力，推动了社会智能机器人技术的发展。 

---
# LL-Localizer: A Life-Long Localization System based on Dynamic i-Octree 

**Title (ZH)**: LL-Localizer: 一种基于动态i-Octree的终身定位系统 

**Authors**: Xinyi Li, Shenghai Yuan, Haoxin Cai, Shunan Lu, Wenhua Wang, Jianqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01583)  

**Abstract**: This paper proposes an incremental voxel-based life-long localization method, LL-Localizer, which enables robots to localize robustly and accurately in multi-session mode using prior maps. Meanwhile, considering that it is difficult to be aware of changes in the environment in the prior map and robots may traverse between mapped and unmapped areas during actual operation, we will update the map when needed according to the established strategies through incremental voxel map. Besides, to ensure high performance in real-time and facilitate our map management, we utilize Dynamic i-Octree, an efficient organization of 3D points based on Dynamic Octree to load local map and update the map during the robot's operation. The experiments show that our system can perform stable and accurate localization comparable to state-of-the-art LIO systems. And even if the environment in the prior map changes or the robots traverse between mapped and unmapped areas, our system can still maintain robust and accurate localization without any distinction. Our demo can be found on Blibili (this https URL) and youtube (this https URL) and the program will be available at this https URL. 

**Abstract (ZH)**: 一种基于增量体素的 lifelong 定位方法 LL-Localizer：在多会话模式下使用先验地图进行鲁棒且准确的定位 

---
# 8-DoFs Cable Driven Parallel Robots for Bimanual Teleportation 

**Title (ZH)**: 8-自由度缆索驱动并联机器人实现双臂远程传送 

**Authors**: Hung Hon Cheng, Josie Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2504.01554)  

**Abstract**: Teleoperation plays a critical role in intuitive robot control and imitation learning, particularly for complex tasks involving mobile manipulators with redundant degrees of freedom (DoFs). However, most existing master controllers are limited to 6-DoF spatial control and basic gripper control, making them insufficient for controlling high-DoF robots and restricting the operator to a small workspace. In this work, we present a novel, low-cost, high-DoF master controller based on Cable-Driven Parallel Robots (CDPRs), designed to overcome these limitations. The system decouples translation and orientation control, following a scalable 3 + 3 + n DoF structure: 3 DoFs for large-range translation using a CDPR, 3 DoFs for orientation using a gimbal mechanism, and n additional DoFs for gripper and redundant joint control. Its lightweight cable-driven design enables a large and adaptable workspace while minimizing actuator load. The end-effector remains stable without requiring continuous high-torque input, unlike most serial robot arms. We developed the first dual-arm CDPR-based master controller using cost-effective actuators and a simple mechanical structure. In demonstrations, the system successfully controlled an 8-DoF robotic arm with a 2-DoF pan-tilt camera, performing tasks such as pick-and-place, knot tying, object sorting, and tape application. The results show precise, versatile, and practical high-DoF teleoperation. 

**Abstract (ZH)**: 基于缆索驱动并联机器人（CDPR）的新型低成本高自由度远程操控器 

---
# Grasping by Spiraling: Reproducing Elephant Movements with Rigid-Soft Robot Synergy 

**Title (ZH)**: 螺旋抓取： rigid-soft 机器人协同实现象的动作复现 

**Authors**: Huishi Huang, Haozhe Wang, Chongyu Fang, Mingge Yan, Ruochen Xu, Yiyuan Zhang, Zhanchi Wang, Fengkang Ying, Jun Liu, Cecilia Laschi, Marcelo H. Ang Jr  

**Link**: [PDF](https://arxiv.org/pdf/2504.01507)  

**Abstract**: The logarithmic spiral is observed as a common pattern in several living beings across kingdoms and species. Some examples include fern shoots, prehensile tails, and soft limbs like octopus arms and elephant trunks. In the latter cases, spiraling is also used for grasping. Motivated by how this strategy simplifies behavior into kinematic primitives and combines them to develop smart grasping movements, this work focuses on the elephant trunk, which is more deeply investigated in the literature. We present a soft arm combined with a rigid robotic system to replicate elephant grasping capabilities based on the combination of a soft trunk with a solid body. In our system, the rigid arm ensures positioning and orientation, mimicking the role of the elephant's head, while the soft manipulator reproduces trunk motion primitives of bending and twisting under proper actuation patterns. This synergy replicates 9 distinct elephant grasping strategies reported in the literature, accommodating objects of varying shapes and sizes. The synergistic interaction between the rigid and soft components of the system minimizes the control complexity while maintaining a high degree of adaptability. 

**Abstract (ZH)**: 对数 Spiral在多个王国和物种的生物体中作为常见模式被观察到，例如蕨类植物的茎、灵长尾巴以及八足和象鼻等柔软肢体。在后者中，螺旋形结构同样用于抓握。受这种策略简化行为成运动学基本模式并将其组合发展出智能抓握动作的启发，本工作专注于大象象鼻，因其在文献中得到了更深入的研究。我们提出了一个结合了刚性机器人系统的软臂，以复制基于柔软象鼻与固体身体结合的大象抓握能力。在我们的系统中，刚性臂确保定位和姿态控制，模拟大象头部的角色，而软操作器则在适当的操作模式下再现弯曲和扭转等象鼻运动基本模式。这种协同作用复制了文献中报告的9种不同的大象抓握策略，适用于各种形状和大小的物体。系统中刚性与软性部件之间的协同作用，在减少控制复杂性的同时保持了高度的适应性。 

---
# Dynamic Initialization for LiDAR-inertial SLAM 

**Title (ZH)**: LiDAR-惯性SLAM的动态初始化方法 

**Authors**: Jie Xu, Yongxin Ma, Yixuan Li, Xuanxuan Zhang, Jun Zhou, Shenghai Yuan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.01451)  

**Abstract**: The accuracy of the initial state, including initial velocity, gravity direction, and IMU biases, is critical for the initialization of LiDAR-inertial SLAM systems. Inaccurate initial values can reduce initialization speed or lead to failure. When the system faces urgent tasks, robust and fast initialization is required while the robot is moving, such as during the swift assessment of rescue environments after natural disasters, bomb disposal, and restarting LiDAR-inertial SLAM in rescue missions. However, existing initialization methods usually require the platform to remain stationary, which is ineffective when the robot is in motion. To address this issue, this paper introduces a robust and fast dynamic initialization method for LiDAR-inertial systems (D-LI-Init). This method iteratively aligns LiDAR-based odometry with IMU measurements to achieve system initialization. To enhance the reliability of the LiDAR odometry module, the LiDAR and gyroscope are tightly integrated within the ESIKF framework. The gyroscope compensates for rotational distortion in the point cloud. Translational distortion compensation occurs during the iterative update phase, resulting in the output of LiDAR-gyroscope odometry. The proposed method can initialize the system no matter the robot is moving or stationary. Experiments on public datasets and real-world environments demonstrate that the D-LI-Init algorithm can effectively serve various platforms, including vehicles, handheld devices, and UAVs. D-LI-Init completes dynamic initialization regardless of specific motion patterns. To benefit the research community, we have open-sourced our code and test datasets on GitHub. 

**Abstract (ZH)**: 基于激光雷达-惯性SLAM系统的初始状态准确性对于系统初始化至关重要，包括初始速度、重力方向和IMU偏差。不准确的初始值会降低初始化速度或导致初始化失败。当系统面临紧急任务时，特别是在机器人移动过程中，需要快速且鲁棒的初始化方法，如自然灾难后的救援环境快速评估、排爆任务以及救援行动中激光雷达-惯性SLAM的重启。然而，现有的初始化方法通常要求平台保持静止，这在机器人移动时是无效的。为解决这一问题，本文提出了一种基于激光雷达-惯性系统的鲁棒快速动态初始化方法（D-LI-Init）。该方法通过迭代对准激光雷达里程计与IMU测量值来实现系统初始化。为了增强激光雷达里程计模块的可靠性，激光雷达与陀螺仪在ESIKF框架下紧密集成，陀螺仪补偿点云中的旋转失真，平移失真补偿发生在迭代更新阶段，从而得到激光雷达-陀螺仪里程计输出。所提出的方法可以在机器人移动或静止时实现系统初始化。在公共数据集和实际环境上的实验表明，D-LI-Init算法可以有效服务于包括车辆、手持设备和无人机在内的各种平台，并且可以根据特定运动模式的差异完成动态初始化。为促进科研社区的发展，我们已在GitHub上开源了我们的代码和测试数据集。 

---
# DF-Calib: Targetless LiDAR-Camera Calibration via Depth Flow 

**Title (ZH)**: DF-Calib: 无需目标的LiDAR-相机标定 via 深度流 

**Authors**: Shu Han, Xubo Zhu, Ji Wu, Ximeng Cai, Wen Yang, Huai Yu, Gui-Song Xia  

**Link**: [PDF](https://arxiv.org/pdf/2504.01416)  

**Abstract**: Precise LiDAR-camera calibration is crucial for integrating these two sensors into robotic systems to achieve robust perception. In applications like autonomous driving, online targetless calibration enables a prompt sensor misalignment correction from mechanical vibrations without extra targets. However, existing methods exhibit limitations in effectively extracting consistent features from LiDAR and camera data and fail to prioritize salient regions, compromising cross-modal alignment robustness. To address these issues, we propose DF-Calib, a LiDAR-camera calibration method that reformulates calibration as an intra-modality depth flow estimation problem. DF-Calib estimates a dense depth map from the camera image and completes the sparse LiDAR projected depth map, using a shared feature encoder to extract consistent depth-to-depth features, effectively bridging the 2D-3D cross-modal gap. Additionally, we introduce a reliability map to prioritize valid pixels and propose a perceptually weighted sparse flow loss to enhance depth flow estimation. Experimental results across multiple datasets validate its accuracy and generalization,with DF-Calib achieving a mean translation error of 0.635cm and rotation error of 0.045 degrees on the KITTI dataset. 

**Abstract (ZH)**: 精确的激光雷达-相机标定对于将这两种传感器集成到机器人系统中以实现稳健的感知至关重要。在自动驾驶等应用中，在线无靶标标定可以及时校正由机械振动引起的传感器错位，而不需额外的目标。然而，现有方法在从激光雷达和相机数据中提取一致特征方面存在局限性，无法优先处理显著区域，从而削弱了跨模态对齐的稳健性。为解决这些问题，我们提出DF-Calib，一种将标定问题重新表述为跨模态深度流估计问题的激光雷达-相机标定方法。DF-Calib从相机图像估计密集深度图，并完成稀疏的激光雷达投影深度图，通过共享特征编码器提取一致的深度到深度特征，有效地弥合了2D-3D跨模态差距。此外，我们引入可靠性图来优先处理有效像素，并提出感知加权稀疏流损失以提高深度流估计。跨多个数据集的实验结果验证了其准确性和泛化能力，DF-Calib在KITTI数据集上的平均平移误差为0.635cm，旋转误差为0.045度。 

---
# Pedestrian-Aware Motion Planning for Autonomous Driving in Complex Urban Scenarios 

**Title (ZH)**: 行人aware的自主驾驶在复杂城市场景中的运动规划 

**Authors**: Korbinian Moller, Truls Nyberg, Jana Tumova, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2504.01409)  

**Abstract**: Motion planning in uncertain environments like complex urban areas is a key challenge for autonomous vehicles (AVs). The aim of our research is to investigate how AVs can navigate crowded, unpredictable scenarios with multiple pedestrians while maintaining a safe and efficient vehicle behavior. So far, most research has concentrated on static or deterministic traffic participant behavior. This paper introduces a novel algorithm for motion planning in crowded spaces by combining social force principles for simulating realistic pedestrian behavior with a risk-aware motion planner. We evaluate this new algorithm in a 2D simulation environment to rigorously assess AV-pedestrian interactions, demonstrating that our algorithm enables safe, efficient, and adaptive motion planning, particularly in highly crowded urban environments - a first in achieving this level of performance. This study has not taken into consideration real-time constraints and has been shown only in simulation so far. Further studies are needed to investigate the novel algorithm in a complete software stack for AVs on real cars to investigate the entire perception, planning and control pipeline in crowded scenarios. We release the code developed in this research as an open-source resource for further studies and development. It can be accessed at the following link: this https URL 

**Abstract (ZH)**: 在复杂城市环境中不确定条件下进行运动规划是自动驾驶车辆（AVs）面临的关键挑战。本研究旨在探讨如何使AVs在包含多个行人的拥挤、不可预测场景中安全高效地导航。迄今为止，大多数研究都集中在静态或确定性的交通参与者行为上。本文介绍了一种新的算法，通过结合社会力原则模拟真实的人行行为并与一种风险意识的运动规划算法相结合，以实现拥挤空间中的运动规划。本文在2D仿真环境中评估了该新算法，以严格评估AV与行人之间的交互，证明了我们的算法能够实现安全、高效且适应性强的运动规划，特别是在高度拥挤的城市环境中——这是首次在该水平上达到这一性能。本研究未考虑实时约束，仅在仿真环境中展示。需要进一步的研究将该新型算法应用于完整的AV软件栈中的真实车辆，以研究拥挤场景下的整体验测、规划和控制管道。我们已将在此研究中开发的代码作为开源资源发布，以便进一步研究和开发，访问链接为：this https URL 

---
# From Shadows to Safety: Occlusion Tracking and Risk Mitigation for Urban Autonomous Driving 

**Title (ZH)**: 从阴影到安全：城市自主驾驶中的遮挡跟踪与风险缓解 

**Authors**: Korbinian Moller, Luis Schwarzmeier, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2504.01408)  

**Abstract**: Autonomous vehicles (AVs) must navigate dynamic urban environments where occlusions and perception limitations introduce significant uncertainties. This research builds upon and extends existing approaches in risk-aware motion planning and occlusion tracking to address these challenges. While prior studies have developed individual methods for occlusion tracking and risk assessment, a comprehensive method integrating these techniques has not been fully explored. We, therefore, enhance a phantom agent-centric model by incorporating sequential reasoning to track occluded areas and predict potential hazards. Our model enables realistic scenario representation and context-aware risk evaluation by modeling diverse phantom agents, each with distinct behavior profiles. Simulations demonstrate that the proposed approach improves situational awareness and balances proactive safety with efficient traffic flow. While these results underline the potential of our method, validation in real-world scenarios is necessary to confirm its feasibility and generalizability. By utilizing and advancing established methodologies, this work contributes to safer and more reliable AV planning in complex urban environments. To support further research, our method is available as open-source software at: this https URL 

**Abstract (ZH)**: 自主驾驶车辆（AVs）必须导航动态的城市环境，其中遮挡和感知限制引入了显著的不确定性。本研究在风险意识运动规划和遮挡跟踪的现有方法基础上进行扩展，以应对这些挑战。尽管先前的研究开发了单独的遮挡跟踪和风险评估方法，但将这些技术进行全面整合的方法尚未得到充分探索。因此，我们通过引入序列推理来增强基于幽灵代理的模型，以跟踪遮挡区域并预测潜在的危险。我们的模型通过建模具有不同行为特征的多样幽灵代理，实现了现实场景的准确建模和情境感知风险评估。模拟结果表明，所提出的方法可以提高情景意识并平衡积极的安全性和高效的交通流量。尽管这些结果突显了我们方法的潜力，但在实际场景中的验证仍是必要的，以确认其可行性和普适性。通过利用和推进现有方法，本研究为复杂城市环境中的自主驾驶车辆规划的安全性和可靠性做出了贡献。为了支持进一步的研究，我们的方法已作为开源软件提供：this https URL。 

---
# Teaching Robots to Handle Nuclear Waste: A Teleoperation-Based Learning Approach< 

**Title (ZH)**: 基于远程操作的机器人处理核废料学习方法 

**Authors**: Joong-Ku Lee, Hyeonseok Choi, Young Soo Park, Jee-Hwan Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01405)  

**Abstract**: This paper presents a Learning from Teleoperation (LfT) framework that integrates human expertise with robotic precision to enable robots to autonomously perform skills learned from human operators. The proposed framework addresses challenges in nuclear waste handling tasks, which often involve repetitive and meticulous manipulation operations. By capturing operator movements and manipulation forces during teleoperation, the framework utilizes this data to train machine learning models capable of replicating and generalizing human skills. We validate the effectiveness of the LfT framework through its application to a power plug insertion task, selected as a representative scenario that is repetitive yet requires precise trajectory and force control. Experimental results highlight significant improvements in task efficiency, while reducing reliance on continuous operator involvement. 

**Abstract (ZH)**: 基于遥操作的学习（LfT）框架：将人类 expertise 与机器人精确性集成以使机器人自主执行从人类操作者学到的技能 

---
# Intuitive Human-Drone Collaborative Navigation in Unknown Environments through Mixed Reality 

**Title (ZH)**: 基于混合现实的直观人类-无人机未知环境协作导航 

**Authors**: Sanket A. Salunkhe, Pranav Nedunghat, Luca Morando, Nishanth Bobbili, Guanrui Li, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2504.01350)  

**Abstract**: Considering the widespread integration of aerial robots in inspection, search and rescue, and monitoring tasks, there is a growing demand to design intuitive human-drone interfaces. These aim to streamline and enhance the user interaction and collaboration process during drone navigation, ultimately expediting mission success and accommodating users' inputs. In this paper, we present a novel human-drone mixed reality interface that aims to (a) increase human-drone spatial awareness by sharing relevant spatial information and representations between the human equipped with a Head Mounted Display (HMD) and the robot and (b) enable safer and intuitive human-drone interactive and collaborative navigation in unknown environments beyond the simple command and control or teleoperation paradigm. We validate our framework through extensive user studies and experiments in a simulated post-disaster scenarios, comparing its performance against a traditional First-Person View (FPV) control systems. Furthermore, multiple tests on several users underscore the advantages of the proposed solution, which offers intuitive and natural interaction with the system. This demonstrates the solution's ability to assist humans during a drone navigation mission, ensuring its safe and effective execution. 

**Abstract (ZH)**: 基于无人机在 inspection、search and rescue 和 monitoring 任务中广泛集成，设计直观的人-无人机界面的需求日益增多。本论文提出了一种新颖的人-无人机混合现实界面，旨在（a）通过在配备头戴式显示器（HMD）的人类与机器人之间共享相关空间信息和表示来提高人-无人机的空间意识，并（b）在未知环境中的安全且直观的人-无人机交互与协作导航，超越简单的命令与控制或遥操作范式。我们通过在模拟灾难场景中的大量用户研究和实验验证了该框架，并将其性能与传统的第一人称视图（FPV）控制系统的性能进行了比较。此外，对多名用户的多项测试表明，所提出的解决方案具有直观且自然的系统交互优势，证明了该解决方案在无人机导航任务中协助人类的能力，确保任务的安全和有效执行。 

---
# Inverse RL Scene Dynamics Learning for Nonlinear Predictive Control in Autonomous Vehicles 

**Title (ZH)**: 基于逆强化学习的场景动力学学习在自动驾驶车辆的非线性预测控制中 

**Authors**: Sorin Grigorescu, Mihai Zaha  

**Link**: [PDF](https://arxiv.org/pdf/2504.01336)  

**Abstract**: This paper introduces the Deep Learning-based Nonlinear Model Predictive Controller with Scene Dynamics (DL-NMPC-SD) method for autonomous navigation. DL-NMPC-SD uses an a-priori nominal vehicle model in combination with a scene dynamics model learned from temporal range sensing information. The scene dynamics model is responsible for estimating the desired vehicle trajectory, as well as to adjust the true system model used by the underlying model predictive controller. We propose to encode the scene dynamics model within the layers of a deep neural network, which acts as a nonlinear approximator for the high order state-space of the operating conditions. The model is learned based on temporal sequences of range sensing observations and system states, both integrated by an Augmented Memory component. We use Inverse Reinforcement Learning and the Bellman optimality principle to train our learning controller with a modified version of the Deep Q-Learning algorithm, enabling us to estimate the desired state trajectory as an optimal action-value function. We have evaluated DL-NMPC-SD against the baseline Dynamic Window Approach (DWA), as well as against two state-of-the-art End2End and reinforcement learning methods, respectively. The performance has been measured in three experiments: i) in our GridSim virtual environment, ii) on indoor and outdoor navigation tasks using our RovisLab AMTU (Autonomous Mobile Test Unit) platform and iii) on a full scale autonomous test vehicle driving on public roads. 

**Abstract (ZH)**: 基于深度学习的场景动力学非线性模型预测控制方法（DL-NMPC-SD）在自主导航中的应用 

---
# Bi-LAT: Bilateral Control-Based Imitation Learning via Natural Language and Action Chunking with Transformers 

**Title (ZH)**: 基于双边控制的自然语言和动作片段变换器辅助的模仿学习：Bi-LAT 

**Authors**: Takumi Kobayashi, Masato Kobayashi, Thanpimon Buamanee, Yuki Uranishi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01301)  

**Abstract**: We present Bi-LAT, a novel imitation learning framework that unifies bilateral control with natural language processing to achieve precise force modulation in robotic manipulation. Bi-LAT leverages joint position, velocity, and torque data from leader-follower teleoperation while also integrating visual and linguistic cues to dynamically adjust applied force. By encoding human instructions such as "softly grasp the cup" or "strongly twist the sponge" through a multimodal Transformer-based model, Bi-LAT learns to distinguish nuanced force requirements in real-world tasks. We demonstrate Bi-LAT's performance in (1) unimanual cup-stacking scenario where the robot accurately modulates grasp force based on language commands, and (2) bimanual sponge-twisting task that requires coordinated force control. Experimental results show that Bi-LAT effectively reproduces the instructed force levels, particularly when incorporating SigLIP among tested language encoders. Our findings demonstrate the potential of integrating natural language cues into imitation learning, paving the way for more intuitive and adaptive human-robot interaction. For additional material, please visit: this https URL 

**Abstract (ZH)**: Bi-LAT：一种联合双臂控制与自然语言处理的新型 imitation learning 框架，实现精确的机器人操作力调控 

---
# AIM: Acoustic Inertial Measurement for Indoor Drone Localization and Tracking 

**Title (ZH)**: AIM: 室内无人机定位与跟踪的声学惯性测量 

**Authors**: Yimiao Sun, Weiguo Wang, Luca Mottola, Ruijin Wang, Yuan He  

**Link**: [PDF](https://arxiv.org/pdf/2504.01297)  

**Abstract**: We present Acoustic Inertial Measurement (AIM), a one-of-a-kind technique for indoor drone localization and tracking. Indoor drone localization and tracking are arguably a crucial, yet unsolved challenge: in GPS-denied environments, existing approaches enjoy limited applicability, especially in Non-Line of Sight (NLoS), require extensive environment instrumentation, or demand considerable hardware/software changes on drones. In contrast, AIM exploits the acoustic characteristics of the drones to estimate their location and derive their motion, even in NLoS settings. We tame location estimation errors using a dedicated Kalman filter and the Interquartile Range rule (IQR). We implement AIM using an off-the-shelf microphone array and evaluate its performance with a commercial drone under varied settings. Results indicate that the mean localization error of AIM is 46% lower than commercial UWB-based systems in complex indoor scenarios, where state-of-the-art infrared systems would not even work because of NLoS settings. We further demonstrate that AIM can be extended to support indoor spaces with arbitrary ranges and layouts without loss of accuracy by deploying distributed microphone arrays. 

**Abstract (ZH)**: 声学惯性测量（AIM）：一种独特的室内无人机定位与跟踪技术 

---
# ForestVO: Enhancing Visual Odometry in Forest Environments through ForestGlue 

**Title (ZH)**: ForestVO: 通过ForestGlue增强森林环境下的视觉里程计 

**Authors**: Thomas Pritchard, Saifullah Ijaz, Ronald Clark, Basaran Bahadir Kocer  

**Link**: [PDF](https://arxiv.org/pdf/2504.01261)  

**Abstract**: Recent advancements in visual odometry systems have improved autonomous navigation; however, challenges persist in complex environments like forests, where dense foliage, variable lighting, and repetitive textures compromise feature correspondence accuracy. To address these challenges, we introduce ForestGlue, enhancing the SuperPoint feature detector through four configurations - grayscale, RGB, RGB-D, and stereo-vision - optimised for various sensing modalities. For feature matching, we employ LightGlue or SuperGlue, retrained with synthetic forest data. ForestGlue achieves comparable pose estimation accuracy to baseline models but requires only 512 keypoints - just 25% of the baseline's 2048 - to reach an LO-RANSAC AUC score of 0.745 at a 10° threshold. With only a quarter of keypoints needed, ForestGlue significantly reduces computational overhead, demonstrating effectiveness in dynamic forest environments, and making it suitable for real-time deployment on resource-constrained platforms. By combining ForestGlue with a transformer-based pose estimation model, we propose ForestVO, which estimates relative camera poses using matched 2D pixel coordinates between frames. On challenging TartanAir forest sequences, ForestVO achieves an average relative pose error (RPE) of 1.09 m and a kitti_score of 2.33%, outperforming direct-based methods like DSO by 40% in dynamic scenes. Despite using only 10% of the dataset for training, ForestVO maintains competitive performance with TartanVO while being a significantly lighter model. This work establishes an end-to-end deep learning pipeline specifically tailored for visual odometry in forested environments, leveraging forest-specific training data to optimise feature correspondence and pose estimation, thereby enhancing the accuracy and robustness of autonomous navigation systems. 

**Abstract (ZH)**: Recent advancements in视觉里程计系统最近在视觉里程计系统方面的进展已经提高了自主导航的能力；然而，在森林等复杂环境中仍然存在挑战，其中茂密的植被、多变的光照和重复的纹理损害了特征对应准确性。为了应对这些挑战，我们引入了ForestGlue，通过针对各种传感模态优化的灰度、RGB、RGB-D和立体视觉四种配置增强SuperPoint特征检测器。在特征匹配中，我们使用LightGlue或SuperGlue，并重新训练以适应森林合成数据。ForestGlue在姿态估计准确性上与基线模型相当，但在10°阈值下达到LO-RANSAC AUC分数0.745时，仅需512个关键点，这仅为基线模型所需关键点数的一半（2048个的关键点的25%）。由于只需要四分之一的关键点，ForestGlue显著减少了计算开销，证明了其在动态森林环境中的有效性和适用性，使其适合在资源受限的平台上进行实时部署。通过将ForestGlue与基于变压器的姿态估计模型结合，我们提出了ForestVO，该模型使用帧间匹配的2D像素坐标来估计相对相机姿态。在TartanAir森林序列中，ForestVO实现了平均相对姿态误差（RPE）为1.09米和kitti_score为2.33%，在动态场景中优于直接方法如DSO 40%。尽管仅使用了数据集的10%进行训练，ForestVO仍然保持了与TartanVO相当的竞争性能，但模型更加轻量级。本工作建立了一个针对森林环境视觉里程计的端到端深度学习管道，利用森林特定的训练数据来优化特征对应和姿态估计，从而提高了自主导航系统的准确性和鲁棒性。 

---
# The Social Life of Industrial Arms: How Arousal and Attention Shape Human-Robot Interaction 

**Title (ZH)**: 工业机器人的社会生活：唤醒程度与注意力如何塑造人机交互 

**Authors**: Roy El-Helou, Matthew K.X.J Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01260)  

**Abstract**: This study explores how human perceptions of a non-anthropomorphic robotic manipulator are shaped by two key dimensions of behaviour: arousal, defined as the robot's movement energy and expressiveness, and attention, defined as the robot's capacity to selectively orient toward and engage with a user. We introduce a novel control architecture that integrates a gaze-like attention engine with an arousal-modulated motion system to generate socially meaningful behaviours. In a user study, we find that robots exhibiting high attention -- actively directing their focus toward users -- are perceived as warmer and more competent, intentional, and lifelike. In contrast, high arousal -- characterized by fast, expansive, and energetic motions -- increases perceptions of discomfort and disturbance. Importantly, a combination of focused attention and moderate arousal yields the highest ratings of trust and sociability, while excessive arousal diminishes social engagement. These findings offer design insights for endowing non-humanoid robots with expressive, intuitive behaviours that support more natural human-robot interaction. 

**Abstract (ZH)**: 本研究探讨了人类对非人格化机器 manipulator 的感知如何受到两类行为维度的影响：唤醒（定义为机器人的运动能量和表达性）和注意力（定义为机器人选择性地朝向和与用户互动的能力）。我们提出了一种新颖的控制架构，结合了类似凝视的注意力引擎和受唤醒程度调节的运动系统，以生成具有社会意义的行为。在用户研究中，我们发现主动将注意力集中在用户身上的高注意力水平的机器人被感知为更加温暖、有能力、有意向并且更加拟人。相反，具有快速、扩展性和能量充沛运动的高唤醒水平增加了不适和干扰的感知。重要的是，结合集中注意力和适度唤醒的组合获得了最高的信任和社会互动评分，而过度的唤醒降低了社会互动。这些发现为赋予非人形机器人表达性和直观的行为提供了设计见解，以支持更自然的人机互动。 

---
# Plan-and-Act using Large Language Models for Interactive Agreement 

**Title (ZH)**: 使用大型语言模型进行计划与行动以达成互动共识 

**Authors**: Kazuhiro Sasabuchi, Naoki Wake, Atsushi Kanehira, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01252)  

**Abstract**: Recent large language models (LLMs) are capable of planning robot actions. In this paper, we explore how LLMs can be used for planning actions with tasks involving situational human-robot interaction (HRI). A key problem of applying LLMs in situational HRI is balancing between "respecting the current human's activity" and "prioritizing the robot's task," as well as understanding the timing of when to use the LLM to generate an action plan. In this paper, we propose a necessary plan-and-act skill design to solve the above problems. We show that a critical factor for enabling a robot to switch between passive / active interaction behavior is to provide the LLM with an action text about the current robot's action. We also show that a second-stage question to the LLM (about the next timing to call the LLM) is necessary for planning actions at an appropriate timing. The skill design is applied to an Engage skill and is tested on four distinct interaction scenarios. We show that by using the skill design, LLMs can be leveraged to easily scale to different HRI scenarios with a reasonable success rate reaching 90% on the test scenarios. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）能够规划机器人动作。本文探讨了如何在涉及情境人类-机器人交互（HRI）的任务中利用LLMs进行动作规划。将LLMs应用于情境HRI的关键问题是平衡“尊重当前人类的活动”和“优先考虑机器人的任务”的关系，以及理解何时使用LLMs生成动作计划。本文提出了一种必要的计划与执行技能设计以解决上述问题。我们表明，使机器人能够切换到被动/主动交互行为的关键因素是向LLMs提供当前机器人动作的文字描述。我们还表明，在适当的时间规划动作需要LLMs的第二个阶段问题（关于何时再次调用LLMs的时间）。该技能设计应用于一种Engage技能，并在四种不同的交互场景中进行了测试。我们表明，通过使用该技能设计，可以在合理成功率达到90%的情况下，轻松地将LLMs扩展到不同的HRI场景中。 

---
# Value Iteration for Learning Concurrently Executable Robotic Control Tasks 

**Title (ZH)**: 并发可执行机器人控制任务的学习价值迭代 

**Authors**: Sheikh A. Tahmid, Gennaro Notomista  

**Link**: [PDF](https://arxiv.org/pdf/2504.01174)  

**Abstract**: Many modern robotic systems such as multi-robot systems and manipulators exhibit redundancy, a property owing to which they are capable of executing multiple tasks. This work proposes a novel method, based on the Reinforcement Learning (RL) paradigm, to train redundant robots to be able to execute multiple tasks concurrently. Our approach differs from typical multi-objective RL methods insofar as the learned tasks can be combined and executed in possibly time-varying prioritized stacks. We do so by first defining a notion of task independence between learned value functions. We then use our definition of task independence to propose a cost functional that encourages a policy, based on an approximated value function, to accomplish its control objective while minimally interfering with the execution of higher priority tasks. This allows us to train a set of control policies that can be executed simultaneously. We also introduce a version of fitted value iteration to learn to approximate our proposed cost functional efficiently. We demonstrate our approach on several scenarios and robotic systems. 

**Abstract (ZH)**: 基于强化学习的冗余机器人多任务并发执行新方法 

---
# Extended Hybrid Zero Dynamics for Bipedal Walking of the Knee-less Robot SLIDER 

**Title (ZH)**: 膝关节-less 机器人SLIDER的扩展混合零动力学双足步行控制 

**Authors**: Rui Zong, Martin Liang, Yuntian Fang, Ke Wang, Xiaoshuai Chen, Wei Chen, Petar Kormushev  

**Link**: [PDF](https://arxiv.org/pdf/2504.01165)  

**Abstract**: Knee-less bipedal robots like SLIDER have the advantage of ultra-lightweight legs and improved walking energy efficiency compared to traditional humanoid robots. In this paper, we firstly introduce an improved hardware design of the bipedal robot SLIDER with new line-feet and more optimized mass distribution which enables higher locomotion speeds. Secondly, we propose an extended Hybrid Zero Dynamics (eHZD) method, which can be applied to prismatic joint robots like SLIDER. The eHZD method is then used to generate a library of gaits with varying reference velocities in an offline way. Thirdly, a Guided Deep Reinforcement Learning (DRL) algorithm is proposed to use the pre-generated library to create walking control policies in real-time. This approach allows us to combine the advantages of both HZD (for generating stable gaits with a full-dynamics model) and DRL (for real-time adaptive gait generation). The experimental results show that this approach achieves 150% higher walking velocity than the previous MPC-based approach. 

**Abstract (ZH)**: 膝关节less双足机器人如SLIDER具有超轻 legs 和更好的步行能效比传统人形机器人。本文首先介绍了具有新型线脚和更优化质量分布的SLIDER双足机器人改进硬件设计，使其能够达到更高的移动速度。其次，提出了扩展的混合零动态(eHZD)方法，该方法适用于如SLIDER这样的普朗特尔关节机器人。然后，使用eHZD方法以离线方式生成具有不同参考速度的步态库。第三，提出了一种引导式深度强化学习(GDRL)算法，利用预先生成的库实现实时步行控制策略生成。该方法结合了混合零动态(HZD)和深度强化学习(DRL)的优势。实验结果表明，该方法比基于MPC的方法实现了150%更高的步行速度。 

---
# Active Learning Design: Modeling Force Output for Axisymmetric Soft Pneumatic Actuators 

**Title (ZH)**: 主动学习设计：建模轴对称软气动执行器的力输出 

**Authors**: Gregory M. Campbell, Gentian Muhaxheri, Leonardo Ferreira Guilhoto, Christian D. Santangelo, Paris Perdikaris, James Pikul, Mark Yim  

**Link**: [PDF](https://arxiv.org/pdf/2504.01156)  

**Abstract**: Soft pneumatic actuators (SPA) made from elastomeric materials can provide large strain and large force. The behavior of locally strain-restricted hyperelastic materials under inflation has been investigated thoroughly for shape reconfiguration, but requires further investigation for trajectories involving external force. In this work we model force-pressure-height relationships for a concentrically strain-limited class of soft pneumatic actuators and demonstrate the use of this model to design SPA response for object lifting. We predict relationships under different loadings by solving energy minimization equations and verify this theory by using an automated test rig to collect rich data for n=22 Ecoflex 00-30 membranes. We collect this data using an active learning pipeline to efficiently model the design space. We show that this learned material model outperforms the theory-based model and naive curve-fitting approaches. We use our model to optimize membrane design for different lift tasks and compare this performance to other designs. These contributions represent a step towards understanding the natural response for this class of actuator and embodying intelligent lifts in a single-pressure input actuator system. 

**Abstract (ZH)**: 基于弹性材料的软气动执行器（SPA）可以提供大的应变和大的力量。虽然已经详细研究了局部应变限制的超弹性材料在膨胀时的行为以实现形状重构，但对于涉及外部力的轨迹仍需进一步研究。在本文中，我们针对中心应变限制类软气动执行器建立了力-压力-高度关系模型，并展示了该模型在设计SPA响应进行物体搬运方面的应用。通过求解能量最小化方程预测不同载荷下的关系，并使用自动测试平台收集n=22个Ecoflex 00-30膜片的丰富数据进行验证。我们使用主动学习管道高效地建模设计空间。结果显示，所学习的材料模型优于基于理论的模型和简单的曲线拟合方法。我们使用该模型优化不同搬运任务下的膜片设计，并与其它设计进行比较。这些贡献代表了理解此类执行器自然响应的一个步骤，并将智能搬运融入单压力输入执行器系统中。 

---
# Making Sense of Robots in Public Spaces: A Study of Trash Barrel Robots 

**Title (ZH)**: 理解公共空间中的机器人：垃圾桶机器人研究 

**Authors**: Fanjun Bu, Kerstin Fischer, Wendy Ju  

**Link**: [PDF](https://arxiv.org/pdf/2504.01121)  

**Abstract**: In this work, we analyze video data and interviews from a public deployment of two trash barrel robots in a large public space to better understand the sensemaking activities people perform when they encounter robots in public spaces. Based on an analysis of 274 human-robot interactions and interviews with N=65 individuals or groups, we discovered that people were responding not only to the robots or their behavior, but also to the general idea of deploying robots as trashcans, and the larger social implications of that idea. They wanted to understand details about the deployment because having that knowledge would change how they interact with the robot. Based on our data and analysis, we have provided implications for design that may be topics for future human-robot design researchers who are exploring robots for public space deployment. Furthermore, our work offers a practical example of analyzing field data to make sense of robots in public spaces. 

**Abstract (ZH)**: 本研究分析了两个垃圾桶机器人在大型公共空间公共部署中的视频数据和访谈，以更好地理解人们在遇到公共空间中的机器人时所进行的意义建构活动。基于对274次人机互动和与N=65名个体或小组进行的访谈的分析，我们发现人们不仅对机器人及其行为作出反应，还对部署机器人作为垃圾桶这一行为本身以及这一行为背后更广泛的社会意义作出了反应。他们希望了解到关于部署的详细信息，因为这些知识会影响他们与机器人互动的方式。基于我们的数据和分析，我们提出了对未来探索公共空间中机器人部署的人机设计研究人员具有指导意义的设计建议。此外，我们的研究为通过分析现场数据来理解公共空间中的机器人提供了实际案例。 

---
# HomeEmergency -- Using Audio to Find and Respond to Emergencies in the Home 

**Title (ZH)**: HomeEmergency —— 使用音频发现并响应家庭紧急情况 

**Authors**: James F. Mullen Jr, Dhruva Kumar, Xuewei Qi, Rajasimman Madhivanan, Arnie Sen, Dinesh Manocha, Richard Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.01089)  

**Abstract**: In the United States alone accidental home deaths exceed 128,000 per year. Our work aims to enable home robots who respond to emergency scenarios in the home, preventing injuries and deaths. We introduce a new dataset of household emergencies based in the ThreeDWorld simulator. Each scenario in our dataset begins with an instantaneous or periodic sound which may or may not be an emergency. The agent must navigate the multi-room home scene using prior observations, alongside audio signals and images from the simulator, to determine if there is an emergency or not.
In addition to our new dataset, we present a modular approach for localizing and identifying potential home emergencies. Underpinning our approach is a novel probabilistic dynamic scene graph (P-DSG), where our key insight is that graph nodes corresponding to agents can be represented with a probabilistic edge. This edge, when refined using Bayesian inference, enables efficient and effective localization of agents in the scene. We also utilize multi-modal vision-language models (VLMs) as a component in our approach, determining object traits (e.g. flammability) and identifying emergencies. We present a demonstration of our method completing a real-world version of our task on a consumer robot, showing the transferability of both our task and our method. Our dataset will be released to the public upon this papers publication. 

**Abstract (ZH)**: 在美国，住宅中的意外死亡人数每年超过128,000人。我们的工作旨在使家居机器人能够在住宅中应对紧急情况，防止受伤和死亡。我们基于ThreeDWorld模拟器引入了一个新的家庭紧急情况数据集。数据集中的每个场景以瞬时或周期性声音开始，这可能是或可能不是紧急情况。代理必须利用先前的观察、模拟器中的音频信号和图像来确定是否存在紧急情况。
除我们新的数据集外，我们还提出了一种模块化方法来定位和识别潜在的家庭紧急情况。该方法的基础是一种新颖的概率动态场景图（P-DSG），我们的关键洞察是，与代理对应的图节点可以用概率边表示。这种边在使用贝叶斯推理进行细化后，能够高效有效地定位场景中的代理。我们还利用多模态视觉语言模型（VLMs）作为方法的一部分，确定物体特征（例如易燃性）并识别紧急情况。我们展示了我们的方法在消费级机器人上完成我们任务的一个实际示例，证明了我们的任务和方法的可迁移性。我们的数据集将在本文发表后向公众发布。 

---
# Overcoming Deceptiveness in Fitness Optimization with Unsupervised Quality-Diversity 

**Title (ZH)**: 克服适应度优化中的误导性问题以实现无监督的质量多样性 

**Authors**: Lisa Coiffard, Paul Templier, Antoine Cully  

**Link**: [PDF](https://arxiv.org/pdf/2504.01915)  

**Abstract**: Policy optimization seeks the best solution to a control problem according to an objective or fitness function, serving as a fundamental field of engineering and research with applications in robotics. Traditional optimization methods like reinforcement learning and evolutionary algorithms struggle with deceptive fitness landscapes, where following immediate improvements leads to suboptimal solutions. Quality-diversity (QD) algorithms offer a promising approach by maintaining diverse intermediate solutions as stepping stones for escaping local optima. However, QD algorithms require domain expertise to define hand-crafted features, limiting their applicability where characterizing solution diversity remains unclear. In this paper, we show that unsupervised QD algorithms - specifically the AURORA framework, which learns features from sensory data - efficiently solve deceptive optimization problems without domain expertise. By enhancing AURORA with contrastive learning and periodic extinction events, we propose AURORA-XCon, which outperforms all traditional optimization baselines and matches, in some cases even improving by up to 34%, the best QD baseline with domain-specific hand-crafted features. This work establishes a novel application of unsupervised QD algorithms, shifting their focus from discovering novel solutions toward traditional optimization and expanding their potential to domains where defining feature spaces poses challenges. 

**Abstract (ZH)**: 无监督质量多样性算法AURORA-XCon在欺骗性优化问题中的应用 

---
# Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness 

**Title (ZH)**: Ross3D: 带有3D意识的重建视觉指令调优 

**Authors**: Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01901)  

**Abstract**: The rapid development of Large Multimodal Models (LMMs) for 2D images and videos has spurred efforts to adapt these models for interpreting 3D scenes. However, the absence of large-scale 3D vision-language datasets has posed a significant obstacle. To address this issue, typical approaches focus on injecting 3D awareness into 2D LMMs by designing 3D input-level scene representations. This work provides a new perspective. We introduce reconstructive visual instruction tuning with 3D-awareness (Ross3D), which integrates 3D-aware visual supervision into the training procedure. Specifically, it incorporates cross-view and global-view reconstruction. The former requires reconstructing masked views by aggregating overlapping information from other views. The latter aims to aggregate information from all available views to recover Bird's-Eye-View images, contributing to a comprehensive overview of the entire scene. Empirically, Ross3D achieves state-of-the-art performance across various 3D scene understanding benchmarks. More importantly, our semi-supervised experiments demonstrate significant potential in leveraging large amounts of unlabeled 3D vision-only data. 

**Abstract (ZH)**: 大規模多模态模型（LMMs）在2D图像和视频上的快速发展中，推动了这些模型适应解释3D场景的努力。然而，缺乏大规模的3D视觉-语言数据集构成了重大障碍。为解决这一问题， typical approaches通常关注通过设计3D输入级场景表示来注入3D意识。本文提供了一个新的视角。我们引入了具有3D意识的重构视觉指令调优（Ross3D），该方法将3D意识的视觉监督整合到训练过程中。具体而言，它结合了跨视图和全局视图的重建。前者要求通过聚合其他视图中的重叠信息来重建遮掩视图。后者旨在从所有可用视图中聚合信息以恢复鸟瞰图图像，从而为整个场景提供全面的概览。实证研究表明，Ross3D在各种3D场景理解基准测试中达到了最先进的性能。更重要的是，我们的半监督实验展示了大量未标注的3D视觉数据的巨大潜力。 

---
# Quattro: Transformer-Accelerated Iterative Linear Quadratic Regulator Framework for Fast Trajectory Optimization 

**Title (ZH)**: Quattro：基于 Transformer 加速的迭代二次调节框架，用于快速轨迹优化 

**Authors**: Yue Wang, Hoayu Wang, Zhaoxing Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.01806)  

**Abstract**: Real-time optimal control remains a fundamental challenge in robotics, especially for nonlinear systems with stringent performance requirements. As one of the representative trajectory optimization algorithms, the iterative Linear Quadratic Regulator (iLQR) faces limitations due to their inherently sequential computational nature, which restricts the efficiency and applicability of real-time control for robotic systems. While existing parallel implementations aim to overcome the above limitations, they typically demand additional computational iterations and high-performance hardware, leading to only modest practical improvements. In this paper, we introduce Quattro, a transformer-accelerated iLQR framework employing an algorithm-hardware co-design strategy to predict intermediate feedback and feedforward matrices. It facilitates effective parallel computations on resource-constrained devices without sacrificing accuracy. Experiments on cart-pole and quadrotor systems show an algorithm-level acceleration of up to 5.3$\times$ and 27$\times$ per iteration, respectively. When integrated into a Model Predictive Control (MPC) framework, Quattro achieves overall speedups of 2.8$\times$ for the cart-pole and 17.8$\times$ for the quadrotor compared to the one that applies traditional iLQR. Transformer inference is deployed on FPGA to maximize performance, achieving up to 27.3$\times$ speedup over commonly used computing devices, with around 2 to 4$\times$ power reduction and acceptable hardware overhead. 

**Abstract (ZH)**: 基于变压器加速的Quattro：一种算法-硬件协同设计的iLQR框架 

---
# Beyond Non-Expert Demonstrations: Outcome-Driven Action Constraint for Offline Reinforcement Learning 

**Title (ZH)**: 超越非专家演示： Offline 强化学习的基于结果的动作约束 

**Authors**: Ke Jiang, Wen Jiang, Yao Li, Xiaoyang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01719)  

**Abstract**: We address the challenge of offline reinforcement learning using realistic data, specifically non-expert data collected through sub-optimal behavior policies. Under such circumstance, the learned policy must be safe enough to manage \textit{distribution shift} while maintaining sufficient flexibility to deal with non-expert (bad) demonstrations from offline this http URL tackle this issue, we introduce a novel method called Outcome-Driven Action Flexibility (ODAF), which seeks to reduce reliance on the empirical action distribution of the behavior policy, hence reducing the negative impact of those bad this http URL be specific, a new conservative reward mechanism is developed to deal with {\it distribution shift} by evaluating actions according to whether their outcomes meet safety requirements - remaining within the state support area, rather than solely depending on the actions' likelihood based on offline this http URL theoretical justification, we provide empirical evidence on widely used MuJoCo and various maze benchmarks, demonstrating that our ODAF method, implemented using uncertainty quantification techniques, effectively tolerates unseen transitions for improved "trajectory stitching," while enhancing the agent's ability to learn from realistic non-expert data. 

**Abstract (ZH)**: 基于现实数据的离线强化学习挑战：一种基于结果驱动的动作灵活性方法 

---
# Reasoning LLMs for User-Aware Multimodal Conversational Agents 

**Title (ZH)**: 面向用户的多模态对话代理的推理大语言模型 

**Authors**: Hamed Rahimi, Jeanne Cattoni, Meriem Beghili, Mouad Abrini, Mahdi Khoramshahi, Maribel Pino, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2504.01700)  

**Abstract**: Personalization in social robotics is critical for fostering effective human-robot interactions, yet systems often face the cold start problem, where initial user preferences or characteristics are unavailable. This paper proposes a novel framework called USER-LLM R1 for a user-aware conversational agent that addresses this challenge through dynamic user profiling and model initiation. Our approach integrates chain-of-thought (CoT) reasoning models to iteratively infer user preferences and vision-language models (VLMs) to initialize user profiles from multimodal inputs, enabling personalized interactions from the first encounter. Leveraging a Retrieval-Augmented Generation (RAG) architecture, the system dynamically refines user representations within an inherent CoT process, ensuring contextually relevant and adaptive responses. Evaluations on the ElderlyTech-VQA Bench demonstrate significant improvements in ROUGE-1 (+23.2%), ROUGE-2 (+0.6%), and ROUGE-L (+8%) F1 scores over state-of-the-art baselines, with ablation studies underscoring the impact of reasoning model size on performance. Human evaluations further validate the framework's efficacy, particularly for elderly users, where tailored responses enhance engagement and trust. Ethical considerations, including privacy preservation and bias mitigation, are rigorously discussed and addressed to ensure responsible deployment. 

**Abstract (ZH)**: 个人化在社会机器人中的关键性对于培养有效的机器人-人类交互至关重要，但系统往往面临初始用户偏好或特征信息不足的冷启动问题。本文提出了一种名为USER-LLM R1的新框架，通过动态用户画像和模型初始化来解决这一挑战。我们的方法结合了链式思考（CoT）推理模型，通过迭代推断用户偏好，并利用多模态输入下的视觉-语言模型（VLMs）初始化用户画像，从而实现初次交互后的个性化互动。利用检索增强生成（RAG）架构，系统在固有的链式思考过程中动态优化用户表示，以确保相关和适应性强的响应。在ElderlyTech-VQA基准上的评估显示，与最先进的基线相比，ROUGE-1提高了23.2%，ROUGE-2提高了0.6%，ROUGE-L提高了8%的F1分数，消融研究强调了推理模型大小对性能的影响。人类评估进一步验证了框架的有效性，特别是对于老年人用户，个性化的响应提高了参与度和信任度。伦理考量，包括隐私保护和偏见缓解，得到了严格的讨论和解决，以确保负责任的部署。 

---
# Overlap-Aware Feature Learning for Robust Unsupervised Domain Adaptation for 3D Semantic Segmentation 

**Title (ZH)**: 重叠感知特征学习：for robust unsupervised domain adaptation in 3D semantic segmentation 

**Authors**: Junjie Chen, Yuecong Xu, Haosheng Li, Kemi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.01668)  

**Abstract**: 3D point cloud semantic segmentation (PCSS) is a cornerstone for environmental perception in robotic systems and autonomous driving, enabling precise scene understanding through point-wise classification. While unsupervised domain adaptation (UDA) mitigates label scarcity in PCSS, existing methods critically overlook the inherent vulnerability to real-world perturbations (e.g., snow, fog, rain) and adversarial distortions. This work first identifies two intrinsic limitations that undermine current PCSS-UDA robustness: (a) unsupervised features overlap from unaligned boundaries in shared-class regions and (b) feature structure erosion caused by domain-invariant learning that suppresses target-specific patterns. To address the proposed problems, we propose a tripartite framework consisting of: 1) a robustness evaluation model quantifying resilience against adversarial attack/corruption types through robustness metrics; 2) an invertible attention alignment module (IAAM) enabling bidirectional domain mapping while preserving discriminative structure via attention-guided overlap suppression; and 3) a contrastive memory bank with quality-aware contrastive learning that progressively refines pseudo-labels with feature quality for more discriminative representations. Extensive experiments on SynLiDAR-to-SemanticPOSS adaptation demonstrate a maximum mIoU improvement of 14.3\% under adversarial attack. 

**Abstract (ZH)**: 3D点云语义分割的鲁棒无监督领域自适应方法 

---
# Proposition of Affordance-Driven Environment Recognition Framework Using Symbol Networks in Large Language Models 

**Title (ZH)**: 基于符号网络的大语言模型驱动的功能导向环境识别框架 

**Authors**: Kazuma Arii, Satoshi Kurihara  

**Link**: [PDF](https://arxiv.org/pdf/2504.01644)  

**Abstract**: In the quest to enable robots to coexist with humans, understanding dynamic situations and selecting appropriate actions based on common sense and affordances are essential. Conventional AI systems face challenges in applying affordance, as it represents implicit knowledge derived from common sense. However, large language models (LLMs) offer new opportunities due to their ability to process extensive human knowledge. This study proposes a method for automatic affordance acquisition by leveraging LLM outputs. The process involves generating text using LLMs, reconstructing the output into a symbol network using morphological and dependency analysis, and calculating affordances based on network distances. Experiments using ``apple'' as an example demonstrated the method's ability to extract context-dependent affordances with high explainability. The results suggest that the proposed symbol network, reconstructed from LLM outputs, enables robots to interpret affordances effectively, bridging the gap between symbolized data and human-like situational understanding. 

**Abstract (ZH)**: 借助大型语言模型自动获取功能的研究：实现机器人与人类共存的动态情况理解与行动选择 

---
# LLM-mediated Dynamic Plan Generation with a Multi-Agent Approach 

**Title (ZH)**: 基于多Agent方法的LLM辅助动态计划生成 

**Authors**: Reo Abe, Akifumi Ito, Kanata Takayasu, Satoshi Kurihara  

**Link**: [PDF](https://arxiv.org/pdf/2504.01637)  

**Abstract**: Planning methods with high adaptability to dynamic environments are crucial for the development of autonomous and versatile robots. We propose a method for leveraging a large language model (GPT-4o) to automatically generate networks capable of adapting to dynamic environments. The proposed method collects environmental "status," representing conditions and goals, and uses them to generate agents. These agents are interconnected on the basis of specific conditions, resulting in networks that combine flexibility and generality. We conducted evaluation experiments to compare the networks automatically generated with the proposed method with manually constructed ones, confirming the comprehensiveness of the proposed method's networks and their higher generality. This research marks a significant advancement toward the development of versatile planning methods applicable to robotics, autonomous vehicles, smart systems, and other complex environments. 

**Abstract (ZH)**: 基于大型语言模型的自动生成适应动态环境网络方法对于自主多功能机器人发展至关重要。我们提出了一种利用GPT-4o大型语言模型自动生成能够适应动态环境的网络的方法。该方法收集环境“状态”，表示条件和目标，并利用这些信息生成代理。这些代理基于特定条件相互连接，形成兼具灵活性和普适性的网络结构。我们通过评估实验将通过所提出的方法自动生成的网络与手工构建的网络进行比较，验证了所提出方法生成网络的全面性和更高的普适性。这项研究标志着向适应机器人、自主车辆、智能系统和其他复杂环境的多功能规划方法发展的重大进步。 

---
# Cuddle-Fish: Exploring a Soft Floating Robot with Flapping Wings for Physical Interactions 

**Title (ZH)**: cuddle-鱼：探索具有拍打翅膀的软体漂浮机器人用于物理交互 

**Authors**: Mingyang Xu, Jiayi Shao, Yulan Ju, Ximing Shen, Qingyuan Gao, Weijen Chen, Qing Zhang, Yun Suen Pai, Giulia Barbareschi, Matthias Hoppe, Kouta Minamizawa, Kai Kunze  

**Link**: [PDF](https://arxiv.org/pdf/2504.01293)  

**Abstract**: Flying robots, such as quadrotor drones, offer new possibilities for human-robot interaction but often pose safety risks due to fast-spinning propellers, rigid structures, and noise. In contrast, lighter-than-air flapping-wing robots, inspired by animal movement, offer a soft, quiet, and touch-safe alternative. Building on these advantages, we present \textit{Cuddle-Fish}, a soft, flapping-wing floating robot designed for safe, close-proximity interactions in indoor spaces. Through a user study with 24 participants, we explored their perceptions of the robot and experiences during a series of co-located demonstrations in which the robot moved near them. Results showed that participants felt safe, willingly engaged in touch-based interactions with the robot, and exhibited spontaneous affective behaviours, such as patting, stroking, hugging, and cheek-touching, without external prompting. They also reported positive emotional responses towards the robot. These findings suggest that the soft floating robot with flapping wings can serve as a novel and socially acceptable alternative to traditional rigid flying robots, opening new possibilities for companionship, play, and interactive experiences in everyday indoor environments. 

**Abstract (ZH)**: 软扑翼浮空机器人Cuddle-Fish：一种安全近距离互动的新颖替代方案 

---
# FUSION: Frequency-guided Underwater Spatial Image recOnstructioN 

**Title (ZH)**: 频率指导的水下空间图像重建 

**Authors**: Jaskaran Singh Walia, Shravan Venkatraman, Pavithra LK  

**Link**: [PDF](https://arxiv.org/pdf/2504.01243)  

**Abstract**: Underwater images suffer from severe degradations, including color distortions, reduced visibility, and loss of structural details due to wavelength-dependent attenuation and scattering. Existing enhancement methods primarily focus on spatial-domain processing, neglecting the frequency domain's potential to capture global color distributions and long-range dependencies. To address these limitations, we propose FUSION, a dual-domain deep learning framework that jointly leverages spatial and frequency domain information. FUSION independently processes each RGB channel through multi-scale convolutional kernels and adaptive attention mechanisms in the spatial domain, while simultaneously extracting global structural information via FFT-based frequency attention. A Frequency Guided Fusion module integrates complementary features from both domains, followed by inter-channel fusion and adaptive channel recalibration to ensure balanced color distributions. Extensive experiments on benchmark datasets (UIEB, EUVP, SUIM-E) demonstrate that FUSION achieves state-of-the-art performance, consistently outperforming existing methods in reconstruction fidelity (highest PSNR of 23.717 dB and SSIM of 0.883 on UIEB), perceptual quality (lowest LPIPS of 0.112 on UIEB), and visual enhancement metrics (best UIQM of 3.414 on UIEB), while requiring significantly fewer parameters (0.28M) and lower computational complexity, demonstrating its suitability for real-time underwater imaging applications. 

**Abstract (ZH)**: 水下图像遭受严重的退化，包括颜色失真、可见度降低和结构细节丧失，这主要由于光谱依赖性的衰减和散射。现有的增强方法主要关注空域处理，忽略了频域在捕捉全局颜色分布和长距离依赖性方面的潜力。为了弥补这些局限性，我们提出了一种双域深度学习框架FUSION，该框架联合利用空域和频域信息。FUSION在空域通过多重尺度卷积核和自适应注意力机制独立处理每个RGB通道，同时通过基于FFT的频域注意力提取全局结构信息。一个频域指导融合模块将两个域中的互补特征整合起来，并通过跨通道融合和自适应通道校准，确保颜色分布平衡。在基准数据集（UIEB、EUVP、SUIM-E）上的广泛实验表明，FUSION在重建保真度（UIEB上的最高PSNR为23.717 dB和SSIM为0.883）、感知质量（UIEB上的最低LPIPS为0.112）和视觉增强指标（UIEB上的最佳UIQM为3.414）方面达到了最先进的性能，同时需要更少的参数（0.28M）和更低的计算复杂度，这表明它适用于实时水下成像应用。 

---
# Coarse-to-Fine Learning for Multi-Pipette Localisation in Robot-Assisted In Vivo Patch-Clamp 

**Title (ZH)**: 从粗到细学习在机器人辅助在体膜片钳实验中多通道定位 

**Authors**: Lan Wei, Gema Vera Gonzalez, Phatsimo Kgwarae, Alexander Timms, Denis Zahorovsky, Simon Schultz, Dandan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01044)  

**Abstract**: In vivo image-guided multi-pipette patch-clamp is essential for studying cellular interactions and network dynamics in neuroscience. However, current procedures mainly rely on manual expertise, which limits accessibility and scalability. Robotic automation presents a promising solution, but achieving precise real-time detection of multiple pipettes remains a challenge. Existing methods focus on ex vivo experiments or single pipette use, making them inadequate for in vivo multi-pipette scenarios. To address these challenges, we propose a heatmap-augmented coarse-to-fine learning technique to facilitate multi-pipette real-time localisation for robot-assisted in vivo patch-clamp. More specifically, we introduce a Generative Adversarial Network (GAN)-based module to remove background noise and enhance pipette visibility. We then introduce a two-stage Transformer model that starts with predicting the coarse heatmap of the pipette tips, followed by the fine-grained coordination regression module for precise tip localisation. To ensure robust training, we use the Hungarian algorithm for optimal matching between the predicted and actual locations of tips. Experimental results demonstrate that our method achieved > 98% accuracy within 10 {\mu}m, and > 89% accuracy within 5 {\mu}m for the localisation of multi-pipette tips. The average MSE is 2.52 {\mu}m. 

**Abstract (ZH)**: 基于热图增强的粗细粒度学习方法在机器人辅助在体多电极存留电记录中的实时多电极定位 

---
# Cal or No Cal? -- Real-Time Miscalibration Detection of LiDAR and Camera Sensors 

**Title (ZH)**: 有校准或不校准？-- 激光雷达和摄像头传感器的实时失校准检测 

**Authors**: Ilir Tahiraj, Jeremialie Swadiryus, Felix Fent, Markus Lienkamp  

**Link**: [PDF](https://arxiv.org/pdf/2504.01040)  

**Abstract**: The goal of extrinsic calibration is the alignment of sensor data to ensure an accurate representation of the surroundings and enable sensor fusion applications. From a safety perspective, sensor calibration is a key enabler of autonomous driving. In the current state of the art, a trend from target-based offline calibration towards targetless online calibration can be observed. However, online calibration is subject to strict real-time and resource constraints which are not met by state-of-the-art methods. This is mainly due to the high number of parameters to estimate, the reliance on geometric features, or the dependence on specific vehicle maneuvers. To meet these requirements and ensure the vehicle's safety at any time, we propose a miscalibration detection framework that shifts the focus from the direct regression of calibration parameters to a binary classification of the calibration state, i.e., calibrated or miscalibrated. Therefore, we propose a contrastive learning approach that compares embedded features in a latent space to classify the calibration state of two different sensor modalities. Moreover, we provide a comprehensive analysis of the feature embeddings and challenging calibration errors that highlight the performance of our approach. As a result, our method outperforms the current state-of-the-art in terms of detection performance, inference time, and resource demand. The code is open source and available on this https URL. 

**Abstract (ZH)**: 目标检测中的传感器偏差检测框架：从直接回归校准参数转向校准状态的二元分类 

---
# Gaze-Guided 3D Hand Motion Prediction for Detecting Intent in Egocentric Grasping Tasks 

**Title (ZH)**: 基于眼动引导的3D手部运动预测以检测第一人称抓取任务中的意图 

**Authors**: Yufei He, Xucong Zhang, Arno H. A. Stienen  

**Link**: [PDF](https://arxiv.org/pdf/2504.01024)  

**Abstract**: Human intention detection with hand motion prediction is critical to drive the upper-extremity assistive robots in neurorehabilitation applications. However, the traditional methods relying on physiological signal measurement are restrictive and often lack environmental context. We propose a novel approach that predicts future sequences of both hand poses and joint positions. This method integrates gaze information, historical hand motion sequences, and environmental object data, adapting dynamically to the assistive needs of the patient without prior knowledge of the intended object for grasping. Specifically, we use a vector-quantized variational autoencoder for robust hand pose encoding with an autoregressive generative transformer for effective hand motion sequence prediction. We demonstrate the usability of these novel techniques in a pilot study with healthy subjects. To train and evaluate the proposed method, we collect a dataset consisting of various types of grasp actions on different objects from multiple subjects. Through extensive experiments, we demonstrate that the proposed method can successfully predict sequential hand movement. Especially, the gaze information shows significant enhancements in prediction capabilities, particularly with fewer input frames, highlighting the potential of the proposed method for real-world applications. 

**Abstract (ZH)**: 基于手部运动预测的人类意图检测对于神经康复应用中的上肢辅助机器人至关重要。然而，依赖生理信号测量的传统方法往往受限且缺乏环境上下文。我们提出了一种新型方法，预测未来的手部姿态和关节位置序列。该方法整合了凝视信息、历史手部运动序列和环境物体数据，能够动态适应患者的辅助需求，而无需事先知道抓取目标物体的意图。具体而言，我们使用向量量化变分自编码器进行稳健的手部姿态编码，并使用自回归生成变换器进行有效的手部运动序列预测。我们在健康受试者的初步研究中展示了这些新技术的实用性。为了训练和评估所提出的方法，我们收集了一个包含多种类型抓取动作的多对象数据集。通过大量实验，我们证明所提出的方法可以成功预测序列手部运动。特别地，凝视信息在较少输入帧的情况下显示出显著的预测能力提升，突显了所提出方法在实际应用中的潜力。 

---
# Omnidirectional Depth-Aided Occupancy Prediction based on Cylindrical Voxel for Autonomous Driving 

**Title (ZH)**: 基于圆柱体体素的全方位深度辅助占用预测 Autonomous Driving 

**Authors**: Chaofan Wu, Jiaheng Li, Jinghao Cao, Ming Li, Yongkang Feng, Jiayu Wu Shuwen Xu, Zihang Gao, Sidan Du, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.01023)  

**Abstract**: Accurate 3D perception is essential for autonomous driving. Traditional methods often struggle with geometric ambiguity due to a lack of geometric prior. To address these challenges, we use omnidirectional depth estimation to introduce geometric prior. Based on the depth information, we propose a Sketch-Coloring framework OmniDepth-Occ. Additionally, our approach introduces a cylindrical voxel representation based on polar coordinate to better align with the radial nature of panoramic camera views. To address the lack of fisheye camera dataset in autonomous driving tasks, we also build a virtual scene dataset with six fisheye cameras, and the data volume has reached twice that of SemanticKITTI. Experimental results demonstrate that our Sketch-Coloring network significantly enhances 3D perception performance. 

**Abstract (ZH)**: 准确的三维感知对于自动驾驶至关重要。传统方法由于缺乏几何先验常常难以应对几何歧义。为应对这些挑战，我们采用全向深度估计引入几何先验。基于深度信息，我们提出了一种素描着色框架 OmniDepth-Occ。此外，我们的方法引入了一种基于极坐标的空间体素表示，以更好地与全景相机视图的径向特性相匹配。为了解决自动驾驶任务中鱼眼相机数据集的缺乏，我们还构建了一个包含六个鱼眼相机的虚拟场景数据集，数据量达到了SemanticKITTI的两倍。实验结果表明，我们的素描着色网络显著提升了三维感知性能。 

---
