# \LARGE GMP$^{3}$: Learning-Driven, Bellman-Guided Trajectory Planning for UAVs in Real-Time on SE(3) 

**Title (ZH)**: GMP$^{3}$: 由学习驱动、贝尔曼导向的实时SE(3)中无人机轨迹规划 

**Authors**: Babak Salamat, Dominik Mattern, Sebastian-Sven Olzem, Gerhard Elsbacher, Christian Seidel, Andrea M. Tonello  

**Link**: [PDF](https://arxiv.org/pdf/2509.21264)  

**Abstract**: We propose $\text{GMP}^{3}$, a multiphase global path planning framework that generates dynamically feasible three-dimensional trajectories for unmanned aerial vehicles (UAVs) operating in cluttered environments. The framework extends traditional path planning from Euclidean position spaces to the Lie group $\mathrm{SE}(3)$, allowing joint learning of translational motion and rotational dynamics. A modified Bellman-based operator is introduced to support reinforcement learning (RL) policy updates while leveraging prior trajectory information for improved convergence. $\text{GMP}^{3}$ is designed as a distributed framework in which agents influence each other and share policy information along the trajectory: each agent refines its assigned segment and shares with its neighbors via a consensus-based scheme, enabling cooperative policy updates and convergence toward a path shaped globally even under kinematic constraints. We also propose DroneManager, a modular ground control software that interfaces the planner with real UAV platforms via the MAVLink protocol, supporting real-time deployment and feedback. Simulation studies and indoor flight experiments validate the effectiveness of the proposed method in constrained 3D environments, demonstrating reliable obstacle avoidance and smooth, feasible trajectories across both position and orientation. The open-source implementation is available at this https URL 

**Abstract (ZH)**: 我们提出了一种多阶段全局路径规划框架GMP³，该框架为在杂乱环境中操作的无人飞行器(UAV)生成动态可行的三维轨迹。该框架将传统的路径规划从欧几里得位置空间扩展到Lie群SE(3)，允许同时学习平移运动和旋转动力学。引入了一个修改后的基于贝尔曼的操作符，以支持强化学习(RL)策略更新，并利用先验轨迹信息以提高收敛性。GMP³被设计为一个分布式框架，其中智能体相互影响并在轨迹上共享策略信息：每个智能体改进其分配的段，并通过基于共识的方案与邻居共享，从而在满足动力学约束的情况下实现合作策略更新，并收敛于一条全局形状的路径。我们还提出了DroneManager，这是一个模块化的地面控制软件，通过MAVLINK协议将规划器与实际的UAV平台接口连接，支持实时部署和反馈。仿真实验和室内飞行实验验证了在受限的3D环境中所提出方法的有效性，展示了可靠的目标躲避和平滑、可行的轨迹，横跨位置和姿态。源代码可在以下链接获得：this https URL 

---
# BiNoMaP: Learning Category-Level Bimanual Non-Prehensile Manipulation Primitives 

**Title (ZH)**: BiNoMaP: 学习类别级双手非抓握 manipulation 原语 

**Authors**: Huayi Zhou, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.21256)  

**Abstract**: Non-prehensile manipulation, encompassing ungraspable actions such as pushing, poking, and pivoting, represents a critical yet underexplored domain in robotics due to its contact-rich and analytically intractable nature. In this work, we revisit this problem from two novel perspectives. First, we move beyond the usual single-arm setup and the strong assumption of favorable external dexterity such as walls, ramps, or edges. Instead, we advocate a generalizable dual-arm configuration and establish a suite of Bimanual Non-prehensile Manipulation Primitives (BiNoMaP). Second, we depart from the prevailing RL-based paradigm and propose a three-stage, RL-free framework to learn non-prehensile skills. Specifically, we begin by extracting bimanual hand motion trajectories from video demonstrations. Due to visual inaccuracies and morphological gaps, these coarse trajectories are difficult to transfer directly to robotic end-effectors. To address this, we propose a geometry-aware post-optimization algorithm that refines raw motions into executable manipulation primitives that conform to specific motion patterns. Beyond instance-level reproduction, we further enable category-level generalization by parameterizing the learned primitives with object-relevant geometric attributes, particularly size, resulting in adaptable and general parameterized manipulation primitives. We validate BiNoMaP across a range of representative bimanual tasks and diverse object categories, demonstrating its effectiveness, efficiency, versatility, and superior generalization capability. 

**Abstract (ZH)**: 非抓握 manipulate 操作，包括推、戳、旋转等不可抓握动作，由于其丰富的接触特性和难以解析的性质，构成了机器人研究中一个关键但未充分探索的领域。本研究从两个新颖的角度重新审视该问题。首先，我们超越了单一手臂设置和对外部灵巧性如墙壁、坡道或边缘的强烈假设，而是提倡一种可推广的双臂配置，并建立了一套双臂非抓握 manipulate 原始动作（BiNoMaP）。其次，我们离开占主导地位的基于 RL 的范式，提出了一种三阶段、无 RL 的框架来学习非抓握技能。具体而言，我们首先从视频示范中提取双臂手部运动轨迹。由于视觉误差和形态差异，这些粗略的运动轨迹难以直接传输给机器人末端执行器。为了解决这个问题，我们提出了一种几何感知后优化算法，将原始运动轨迹优化为符合特定运动模式的可执行 manipulate 原始动作。通过不仅实现示例级别的再现，还通过基于对象相关的几何属性（尤其是尺寸）参数化所学原始动作，进一步实现了分类级别的泛化，从而得到可适应的、通用的参数化 manipulate 原始动作。我们在一系列代表性双臂任务和多种对象类别中验证了 BiNoMaP，展示了其有效性、效率、多功能性和优越的泛化能力。 

---
# FSGlove: An Inertial-Based Hand Tracking System with Shape-Aware Calibration 

**Title (ZH)**: 基于惯性的人手跟踪系统：具有形状感知校准的方法 

**Authors**: Yutong Li, Jieyi Zhang, Wenqiang Xu, Tutian Tang, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21242)  

**Abstract**: Accurate hand motion capture (MoCap) is vital for applications in robotics, virtual reality, and biomechanics, yet existing systems face limitations in capturing high-degree-of-freedom (DoF) joint kinematics and personalized hand shape. Commercial gloves offer up to 21 DoFs, which are insufficient for complex manipulations while neglecting shape variations that are critical for contact-rich tasks. We present FSGlove, an inertial-based system that simultaneously tracks up to 48 DoFs and reconstructs personalized hand shapes via DiffHCal, a novel calibration method. Each finger joint and the dorsum are equipped with IMUs, enabling high-resolution motion sensing. DiffHCal integrates with the parametric MANO model through differentiable optimization, resolving joint kinematics, shape parameters, and sensor misalignment during a single streamlined calibration. The system achieves state-of-the-art accuracy, with joint angle errors of less than 2.7 degree, and outperforms commercial alternatives in shape reconstruction and contact fidelity. FSGlove's open-source hardware and software design ensures compatibility with current VR and robotics ecosystems, while its ability to capture subtle motions (e.g., fingertip rubbing) bridges the gap between human dexterity and robotic imitation. Evaluated against Nokov optical MoCap, FSGlove advances hand tracking by unifying the kinematic and contact fidelity. Hardware design, software, and more results are available at: this https URL. 

**Abstract (ZH)**: 基于惯性的FSGlove手套：同时追踪48个自由度并重建个性化手型 

---
# Next-Generation Aerial Robots -- Omniorientational Strategies: Dynamic Modeling, Control, and Comparative Analysis 

**Title (ZH)**: 下一代空中机器人——全方位姿态策略：动态建模、控制与比较分析 

**Authors**: Ali Kafili Gavgani, Amin Talaeizadeh, Aria Alasty, Hossein Nejat Pishkenari, Esmaeil Najafi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21210)  

**Abstract**: Conventional multi-rotors are under-actuated systems, hindering them from independently controlling attitude from position. In this study, we present several distinct configurations that incorporate additional control inputs for manipulating the angles of the propeller axes. This addresses the mentioned limitations, making the systems "omniorientational". We comprehensively derived detailed dynamic models for all introduced configurations and validated by a methodology using Simscape Multibody simulations. Two controllers are designed: a sliding mode controller for robust handling of disturbances and a novel PID-based controller with gravity compensation integrating linear and non-linear allocators, designed for computational efficiency. A custom control allocation strategy is implemented to manage the input-non-affine nature of these systems, seeking to maximize battery life by minimizing the "Power Consumption Factor" defined in this study. Moreover, the controllers effectively managed harsh disturbances and uncertainties. Simulations compare and analyze the proposed configurations and controllers, majorly considering their power consumption. Furthermore, we conduct a qualitative comparison to evaluate the impact of different types of uncertainties on the control system, highlighting areas for potential model or hardware improvements. The analysis in this study provides a roadmap for future researchers to design omniorientational drones based on their design objectives, offering practical insights into configuration selection and controller design. This research aligns with the project SAC-1, one of the objectives of Sharif AgRoLab. 

**Abstract (ZH)**: 传统的多旋翼无人机是欠驱动系统，限制了其独立控制姿态与位置的能力。本研究提出了几种不同的配置，通过增加额外的控制输入来操控螺旋桨轴的角度，从而解决了上述限制，使系统具备全方位姿态控制能力。本文详细推导了所有引入配置的动态模型，并通过基于Simscape Multibody的仿真方法进行了验证。设计了两种控制器：滑模控制器用于稳健地处理干扰，以及一种新型基于PID的控制器，结合了线性和非线性分配器，用于提高计算效率。实现了一种自定义控制分配策略来管理这些系统的输入非线性特性，旨在通过最小化“功耗因子”来最大化电池寿命。此外，控制器有效处理了恶劣的干扰和不确定性。仿真比较和分析了所提出的配置和控制器，主要考虑它们的功耗。此外，我们进行了定性比较以评估不同类型的不确定性对控制系统的影响，指出了模型或硬件改进的潜在领域。本文的分析为未来研究人员设计全方位姿态控制无人机提供了方向，提供了关于配置选择和控制器设计的实用见解。该研究与Sharif AgRoLab项目SAC-1的目标相一致。 

---
# DAGDiff: Guiding Dual-Arm Grasp Diffusion to Stable and Collision-Free Grasps 

**Title (ZH)**: DAGDiff: 引导双臂抓取扩散以实现稳定且无碰撞的抓取 

**Authors**: Md Faizal Karim, Vignesh Vembar, Keshab Patra, Gaurav Singh, K Madhava Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2509.21145)  

**Abstract**: Reliable dual-arm grasping is essential for manipulating large and complex objects but remains a challenging problem due to stability, collision, and generalization requirements. Prior methods typically decompose the task into two independent grasp proposals, relying on region priors or heuristics that limit generalization and provide no principled guarantee of stability. We propose DAGDiff, an end-to-end framework that directly denoises to grasp pairs in the SE(3) x SE(3) space. Our key insight is that stability and collision can be enforced more effectively by guiding the diffusion process with classifier signals, rather than relying on explicit region detection or object priors. To this end, DAGDiff integrates geometry-, stability-, and collision-aware guidance terms that steer the generative process toward grasps that are physically valid and force-closure compliant. We comprehensively evaluate DAGDiff through analytical force-closure checks, collision analysis, and large-scale physics-based simulations, showing consistent improvements over previous work on these metrics. Finally, we demonstrate that our framework generates dual-arm grasps directly on real-world point clouds of previously unseen objects, which are executed on a heterogeneous dual-arm setup where two manipulators reliably grasp and lift them. 

**Abstract (ZH)**: 可靠的双臂抓取对于操作大型和复杂对象至关重要，但由于稳定性和碰撞要求，仍然是一个具有挑战性的问题。先前的方法通常将任务分解为两个独立的抓取提案，依赖于区域先验或启发式方法，这些方法限制了泛化能力，并没有提供稳定的实质性保证。我们提出了DAGDiff，这是一个端到端框架，可以直接在SE(3) x SE(3)空间中去噪以生成抓取对。我们的关键洞察是，通过使用分类器信号引导扩散过程可以更有效地确保稳定性和避免碰撞，而不是依赖显式的区域检测或对象先验。为此，DAGDiff 结合了几何、稳定性和碰撞感知的引导项，使生成过程偏向于物理上有效的且满足力封闭的抓取。我们通过分析力封闭检查、碰撞分析和大规模物理仿真全面评估了DAGDiff，在这些指标上展示了相对于之前工作的改进。最后，我们证明了该框架可以直接在未见过的实际点云对象上生成双臂抓取，并在异构双臂操作设置中可靠地抓取和提升它们。 

---
# Flight Dynamics to Sensing Modalities: Exploiting Drone Ground Effect for Accurate Edge Detection 

**Title (ZH)**: 从飞行动力学到传感模态：利用无人机地面效应进行精确边缘检测 

**Authors**: Chenyu Zhao, Jingao Xu, Ciyu Ruan, Haoyang Wang, Shengbo Wang, Jiaqi Li, Jirong Zha, Weijie Hong, Zheng Yang, Yunhao Liu, Xiao-Ping Zhang, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21085)  

**Abstract**: Drone-based rapid and accurate environmental edge detection is highly advantageous for tasks such as disaster relief and autonomous navigation. Current methods, using radars or cameras, raise deployment costs and burden lightweight drones with high computational demands. In this paper, we propose AirTouch, a system that transforms the ground effect from a stability "foe" in traditional flight control views, into a "friend" for accurate and efficient edge detection. Our key insight is that analyzing drone basic attitude sensor readings and flight commands allows us to detect ground effect changes. Such changes typically indicate the drone flying over a boundary of two materials, making this information valuable for edge detection. We approach this insight through theoretical analysis, algorithm design, and implementation, fully leveraging the ground effect as a new sensing modality without compromising drone flight stability, thereby achieving accurate and efficient scene edge detection. We also compare this new sensing modality with vision-based methods to clarify its exclusive advantages in resource efficiency and detection capability. Extensive evaluations demonstrate that our system achieves a high detection accuracy with mean detection distance errors of 0.051m, outperforming the baseline method performance by 86%. With such detection performance, our system requires only 43 mW power consumption, contributing to this new sensing modality for low-cost and highly efficient edge detection. 

**Abstract (ZH)**: 基于无人机的快速准确环境边缘检测对灾害救援和自主导航等任务具有高度优势。当前方法使用雷达或摄像头会增加部署成本，并给轻型无人机带来较高的计算负担。本文提出AirTouch系统，将其地面效应从传统飞行控制中的稳定性“敌对因素”转变为准确高效边缘检测的“盟友”。我们的关键洞察是，通过分析无人机的基本姿态传感器读数和飞行指令，可以检测地面效应的变化。这种变化通常表明无人机飞越了两种材料的边界，从而使得这些信息在边缘检测中具有价值。我们通过理论分析、算法设计和实现，充分利用地面效应作为一种新的传感模态，同时不牺牲无人机飞行稳定性，从而实现准确高效场景边缘检测。此外，我们将这种新的传感模态与基于视觉的方法进行比较，以阐明其在资源效率和检测能力方面的独特优势。广泛评估表明，我们的系统在平均检测距离误差为0.051米的情况下实现了高检测精度，相比基线方法性能提升86%。凭借如此高的检测性能，我们的系统仅需43毫瓦的功率消耗，从而推动了低成本和高效边缘检测的新型传感模态的发展。 

---
# Multi-Robot Vision-Based Task and Motion Planning for EV Battery Disassembly and Sorting 

**Title (ZH)**: 基于视觉的多机器人任务与运动规划在电动汽车电池拆解与分类中的应用 

**Authors**: Abdelaziz Shaarawy, Cansu Erdogan, Rustam Stolkin, Alireza Rastegarpanah  

**Link**: [PDF](https://arxiv.org/pdf/2509.21020)  

**Abstract**: Electric-vehicle (EV) battery disassembly requires precise multi-robot coordination, short and reliable motions, and robust collision safety in cluttered, dynamic scenes. We propose a four-layer task-and-motion planning (TAMP) framework that couples symbolic task planning and cost- and accessibility-aware allocation with a TP-GMM-guided motion planner learned from demonstrations. Stereo vision with YOLOv8 provides real-time component localization, while OctoMap-based 3D mapping and FCL(Flexible Collision Library) checks in MoveIt unify predictive digital-twin collision checking with reactive, vision-based avoidance. Validated on two UR10e robots across cable, busbar, service plug, and three leaf-cell removals, the approach yields substantially more compact and safer motions than a default RRTConnect baseline under identical perception and task assignments: average end-effector path length drops by $-63.3\%$ and makespan by $-8.1\%$; per-arm swept volumes shrink (R1: $0.583\rightarrow0.139\,\mathrm{m}^3$; R2: $0.696\rightarrow0.252\,\mathrm{m}^3$), and mutual overlap decreases by $47\%$ ($0.064\rightarrow0.034\,\mathrm{m}^3$). These results highlight improved autonomy, precision, and safety for multi-robot EV battery disassembly in unstructured, dynamic environments. 

**Abstract (ZH)**: 电驱动车辆电池拆卸需要精确的多机器人协调、短且可靠的运动以及在拥挤和动态场景中的 robust 碰撞安全。我们提出了一种四层任务与运动规划（TAMP）框架，该框架将符号任务规划与基于 TP-GMM 的运动规划器引导的成本和可达性分配相结合，该运动规划器从演示中学习。立体视觉与 YOLOv8 提供实时部件定位，而基于 OctoMap 的三维建图和 MoveIt 中的 FCL 碰撞检查统一了预测的数字孪生碰撞检查与反应式、视觉避免。在两个 UR10e 机器人上针对电缆、汇流排、服务插头和三个电芯拆卸进行了验证，该方法在相同的感知和任务分配下相比默认的 RRTConnect 基线产生了显著更加紧凑和安全的运动：末端执行器路径长度平均减少 63.3%，周转时间减少 8.1%；每臂扫过的体积减小（R1：0.583→0.139 立方米；R2：0.696→0.252 立方米），以及相互重叠减少了 47%（0.064→0.034 立方米）。这些结果突显了在未结构化和动态环境中进行多机器人电驱动车辆电池拆卸时改进的自主性、精确性和安全性。 

---
# BactoBot: A Low-Cost, Bacteria-Inspired Soft Underwater Robot for Marine Exploration 

**Title (ZH)**: BactoBot：一种低成本、细菌启发的软体水下机器人用于海洋探索 

**Authors**: Rubaiyat Tasnim Chowdhury, Nayan Bala, Ronojoy Roy, Tarek Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2509.20964)  

**Abstract**: Traditional rigid underwater vehicles pose risks to delicate marine ecosystems. This paper presents BactoBot, a low-cost, soft underwater robot designed for safe and gentle marine exploration. Inspired by bacterial flagellar propulsion, BactoBot features 12 flexible, silicone-based arms arranged on a 3D-printed dodecahedral frame. The design provides inherent compliance, redundancy, and the potential for omnidirectional movement. The prototype was fabricated using accessible DIY methods, including food-grade silicone molding, 3D printing, and off-the-shelf microcontrollers. Waterproofing and buoyancy calibration protocols were developed, and the robot was successfully tested in a controlled water tank, demonstrating forward motion and turning. The results validate the feasibility of replicating complex biological locomotion at low cost. The project lays a foundation for environmentally conscious robotic tools, particularly for marine science in resource-constrained settings, and identifies pathways toward autonomous operation and field deployment. 

**Abstract (ZH)**: BactoBot：一种低成本的仿生软体水下机器人及其在海洋探索中的应用 

---
# Incorporating Human-Inspired Ankle Characteristics in a Forced-Oscillation-Based Reduced-Order Model for Walking 

**Title (ZH)**: 基于强迫振荡的降阶模型中融入启发自人类踝关节的特性 

**Authors**: Chathura Semasinghe, Siavash Rezazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.20689)  

**Abstract**: This paper extends the forced-oscillation-based reduced-order model of walking to a model with ankles and feet. A human-inspired paradigm was designed for the ankle dynamics, which results in improved gait characteristics compared to the point-foot model. In addition, it was shown that while the proposed model can stabilize against large errors in initial conditions through combination of foot placement and ankle strategies, the model is able to stabilize against small perturbations without relying on the foot placement control and solely through the designed proprioceptive ankle scheme. This novel property, which is also observed in humans, can help in better understanding of anthropomorphic walking and its stabilization mechanisms. 

**Abstract (ZH)**: 基于踝关节和足部的强迫振荡降阶行走模型研究 

---
# RAM-NAS: Resource-aware Multiobjective Neural Architecture Search Method for Robot Vision Tasks 

**Title (ZH)**: 资源感知多目标神经架构搜索方法：面向机器人视觉任务的RAM-NAS 

**Authors**: Shouren Mao, Minghao Qin, Wei Dong, Huajian Liu, Yongzhuo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20688)  

**Abstract**: Neural architecture search (NAS) has shown great promise in automatically designing lightweight models. However, conventional approaches are insufficient in training the supernet and pay little attention to actual robot hardware resources. To meet such challenges, we propose RAM-NAS, a resource-aware multi-objective NAS method that focuses on improving the supernet pretrain and resource-awareness on robot hardware devices. We introduce the concept of subnets mutual distillation, which refers to mutually distilling all subnets sampled by the sandwich rule. Additionally, we utilize the Decoupled Knowledge Distillation (DKD) loss to enhance logits distillation performance. To expedite the search process with consideration for hardware resources, we used data from three types of robotic edge hardware to train Latency Surrogate predictors. These predictors facilitated the estimation of hardware inference latency during the search phase, enabling a unified multi-objective evolutionary search to balance model accuracy and latency trade-offs. Our discovered model family, RAM-NAS models, can achieve top-1 accuracy ranging from 76.7% to 81.4% on ImageNet. In addition, the resource-aware multi-objective NAS we employ significantly reduces the model's inference latency on edge hardware for robots. We conducted experiments on downstream tasks to verify the scalability of our methods. The inference time for detection and segmentation is reduced on all three hardware types compared to MobileNetv3-based methods. Our work fills the gap in NAS for robot hardware resource-aware. 

**Abstract (ZH)**: 资源感知多目标NAS方法RAM-NAS及其在机器人硬件资源优化中的应用 

---
# Equi-RO: A 4D mmWave Radar Odometry via Equivariant Networks 

**Title (ZH)**: Equi-RO: 通过等变网络的4D毫米波雷达里程表测量 

**Authors**: Zeyu Han, Shuocheng Yang, Minghan Zhu, Fang Zhang, Shaobing Xu, Maani Ghaffari, Jianqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20674)  

**Abstract**: Autonomous vehicles and robots rely on accurate odometry estimation in GPS-denied environments. While LiDARs and cameras struggle under extreme weather, 4D mmWave radar emerges as a robust alternative with all-weather operability and velocity measurement. In this paper, we introduce Equi-RO, an equivariant network-based framework for 4D radar odometry. Our algorithm pre-processes Doppler velocity into invariant node and edge features in the graph, and employs separate networks for equivariant and invariant feature processing. A graph-based architecture enhances feature aggregation in sparse radar data, improving inter-frame correspondence. Experiments on the open-source dataset and self-collected dataset show Equi-RO outperforms state-of-the-art algorithms in accuracy and robustness. Overall, our method achieves 10.7% and 20.0% relative improvements in translation and rotation accuracy, respectively, compared to the best baseline on the open-source dataset. 

**Abstract (ZH)**: 自主驾驶车辆和机器人在GPS受限环境中依赖于准确的速度估计。尽管LiDAR和摄像头在极端天气下表现不佳，4D毫米波雷达由于其全天候操作能力和速度测量能力成为可靠的选择。本文 introduces Equi-RO，一种基于等变网络的4D雷达速度估计框架。该算法将雷达速度预处理为图中的不变节点和边特征，并使用独立的网络对等变和不变特征进行处理。基于图的架构在稀疏雷达数据中增强了特征聚合，提高了帧间对应关系。在开源数据集和自采集数据集上的实验表明，Equi-RO 在准确性和鲁棒性方面优于现有最佳算法。总体而言，我们的方法在开源数据集的最佳基准上分别实现了10.7%和20.0%的平移和旋转精度改进。 

---
# Cyber Racing Coach: A Haptic Shared Control Framework for Teaching Advanced Driving Skills 

**Title (ZH)**: 网络赛车教练：一种用于教学高级驾驶技巧的触觉协同控制框架 

**Authors**: Congkai Shen, Siyuan Yu, Yifan Weng, Haoran Ma, Chen Li, Hiroshi Yasuda, James Dallas, Michael Thompson, John Subosits, Tulga Ersal  

**Link**: [PDF](https://arxiv.org/pdf/2509.20653)  

**Abstract**: This study introduces a haptic shared control framework designed to teach human drivers advanced driving skills. In this context, shared control refers to a driving mode where the human driver collaborates with an autonomous driving system to control the steering of a vehicle simultaneously. Advanced driving skills are those necessary to safely push the vehicle to its handling limits in high-performance driving such as racing and emergency obstacle avoidance. Previous research has demonstrated the performance and safety benefits of shared control schemes using both subjective and objective evaluations. However, these schemes have not been assessed for their impact on skill acquisition on complex and demanding tasks. Prior research on long-term skill acquisition either applies haptic shared control to simple tasks or employs other feedback methods like visual and auditory aids. To bridge this gap, this study creates a cyber racing coach framework based on the haptic shared control paradigm and evaluates its performance in helping human drivers acquire high-performance driving skills. The framework introduces (1) an autonomous driving system that is capable of cooperating with humans in a highly performant driving scenario; and (2) a haptic shared control mechanism along with a fading scheme to gradually reduce the steering assistance from autonomy based on the human driver's performance during training. Two benchmarks are considered: self-learning (no assistance) and full assistance during training. Results from a human subject study indicate that the proposed framework helps human drivers develop superior racing skills compared to the benchmarks, resulting in better performance and consistency. 

**Abstract (ZH)**: 基于触觉共享控制的高级驾驶技能教学框架研究 

---
# Uncertainty-Aware Active Source Tracking of Marine Pollution using Unmanned Surface Vehicles 

**Title (ZH)**: 使用无人表面车辆的不确定性感知海洋污染主动源追踪 

**Authors**: Song Ma, Richard Bucknall, Yuanchang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20593)  

**Abstract**: This paper proposes an uncertainty-aware marine pollution source tracking framework for unmanned surface vehicles (USVs). By integrating high-fidelity marine pollution dispersion simulation with informative path planning techniques, we demonstrate effective identification of pollution sources in marine environments. The proposed approach is implemented based on Robot Operating System (ROS), processing real-time sensor data to update probabilistic source location estimates. The system progressively refines the estimation of source location while quantifying uncertainty levels in its predictions. Experiments conducted in simulated environments with varying source locations, flow conditions, and starting positions demonstrate the framework's ability to localise pollution sources with high accuracy. Results show that the proposed approach achieves reliable source localisation efficiently. This work contributes to the development of full autonomous environmental monitoring capabilities essential for rapid response to marine pollution incidents. 

**Abstract (ZH)**: 本文提出了一种不确定性感知的无人水面船舶海洋污染源跟踪框架。通过集成高保真海洋污染分散模拟与信息性路径规划技术，我们展示了在海洋环境中有效识别污染源的能力。所提出的方法基于Robot Operating System (ROS) 实现，处理实时传感器数据以更新概率性源位置估计。系统逐步细化源位置估计的同时，量化预测中的不确定性水平。在不同源位置、流条件和起始位置的模拟环境中进行的实验展示了该框架以高精度定位污染源的能力。结果表明，所提出的方法能够高效地实现可靠的源定位。本文为快速应对海洋污染事件所需的整体自主环境监测能力的发展做出了贡献。 

---
# GraspFactory: A Large Object-Centric Grasping Dataset 

**Title (ZH)**: GraspFactory: 一个大型对象中心抓取数据集 

**Authors**: Srinidhi Kalgundi Srinivas, Yash Shukla, Adam Arnold, Sachin Chitta  

**Link**: [PDF](https://arxiv.org/pdf/2509.20550)  

**Abstract**: Robotic grasping is a crucial task in industrial automation, where robots are increasingly expected to handle a wide range of objects. However, a significant challenge arises when robot grasping models trained on limited datasets encounter novel objects. In real-world environments such as warehouses or manufacturing plants, the diversity of objects can be vast, and grasping models need to generalize to this diversity. Training large, generalizable robot-grasping models requires geometrically diverse datasets. In this paper, we introduce GraspFactory, a dataset containing over 109 million 6-DoF grasps collectively for the Franka Panda (with 14,690 objects) and Robotiq 2F-85 grippers (with 33,710 objects). GraspFactory is designed for training data-intensive models, and we demonstrate the generalization capabilities of one such model trained on a subset of GraspFactory in both simulated and real-world settings. The dataset and tools are made available for download at this https URL. 

**Abstract (ZH)**: 机器人抓取是工业自动化中的关键任务，其中机器人被期望处理种类繁多的物体。然而，当基于有限数据集训练的机器人抓取模型遇到新型物体时，会面临重大挑战。在仓库或制造工厂等实际环境中，物体的多样性很高，抓取模型需要能够泛化到这种多样性。训练大规模且泛化能力强的机器人抓取模型需要几何上多样的数据集。本文介绍了GraspFactory数据集，包含超过109百万个6-DoF抓取姿态，涵盖了Franka Panda机械手（14,690个物体）和Robotiq 2F-85夹爪（33,710个物体）。GraspFactory旨在用于训练数据密集型模型，并在模拟和实际环境中展示了基于GraspFactory子集训练的模型的泛化能力。数据集及工具可从此网址下载：this https URL。 

---
# MELEGROS: Monolithic Elephant-inspired Gripper with Optical Sensors 

**Title (ZH)**: MELEGROS: 光学传感器驱动的独体象形夹取器 

**Authors**: Petr Trunin, Diana Cafiso, Anderson Brazil Nardin, Trevor Exley, Lucia Beccai  

**Link**: [PDF](https://arxiv.org/pdf/2509.20510)  

**Abstract**: The elephant trunk exemplifies a natural gripper where structure, actuation, and sensing are seamlessly integrated. Inspired by the distal morphology of the African elephant trunk, we present MELEGROS, a Monolithic ELEphant-inspired GRipper with Optical Sensors, emphasizing sensing as an intrinsic, co-fabricated capability. Unlike multi-material or tendon-based approaches, MELEGROS directly integrates six optical waveguide sensors and five pneumatic chambers into a pneumatically actuated lattice structure (12.5 mm cell size) using a single soft resin and one continuous 3D print. This eliminates mechanical mismatches between sensors, actuators, and body, reducing model uncertainty and enabling simulation-guided sensor design and placement. Only four iterations were required to achieve the final prototype, which features a continuous structure capable of elongation, compression, and bending while decoupling tactile and proprioceptive signals. MELEGROS (132 g) lifts more than twice its weight, performs bioinspired actions such as pinching, scooping, and reaching, and delicately grasps fragile items like grapes. The integrated optical sensors provide distinct responses to touch, bending, and chamber deformation, enabling multifunctional perception. MELEGROS demonstrates a new paradigm for soft robotics where fully embedded sensing and continuous structures inherently support versatile, bioinspired manipulation. 

**Abstract (ZH)**: MELEGROS：一种以光学传感器为代表的单一树脂连续结构的象鼻启发式柔体夹爪 

---
# Revisiting Formal Methods for Autonomous Robots: A Structured Survey 

**Title (ZH)**: 重新审视自主机器人形式化方法：一种结构化综述 

**Authors**: Atef Azaiez, David A. Anisi, Marie Farrell, Matt Luckcuck  

**Link**: [PDF](https://arxiv.org/pdf/2509.20488)  

**Abstract**: This paper presents the initial results from our structured literature review on applications of Formal Methods (FM) to Robotic Autonomous Systems (RAS). We describe our structured survey methodology; including database selection and associated search strings, search filters and collaborative review of identified papers. We categorise and enumerate the FM approaches and formalisms that have been used for specification and verification of RAS. We investigate FM in the context of sub-symbolic AI-enabled RAS and examine the evolution of how FM is used over time in this field. This work complements a pre-existing survey in this area and we examine how this research area has matured over time. Specifically, our survey demonstrates that some trends have persisted as observed in a previous survey. Additionally, it recognized new trends that were not considered previously including a noticeable increase in adopting Formal Synthesis approaches as well as Probabilistic Verification Techniques. 

**Abstract (ZH)**: 本文呈现了我们对形式方法在机器人自主系统中应用的结构化文献综述的初步结果。我们描述了结构化调查方法；包括数据库选择和相关搜索字符串、搜索过滤器以及已识别论文的协作审查。我们将形式化方法和形式语言分类并列举，这些方法和形式语言被用于定义和验证机器人自主系统。我们探讨了形式方法在具有亚符号AI的机器人自主系统中的应用，并考察了这种方法在该领域随时间的发展演变。本文补充了该领域的现有调查，并考察了该研究领域随时间的成熟。具体而言，我们的调查表明，一些趋势在之前的调查中已经持续存在。此外，还识别了一些新的趋势，这些趋势在之前的调查中未被考虑，包括显著增加采用形式综合方法以及概率验证技术。 

---
