# HiP-AD: Hierarchical and Multi-Granularity Planning with Deformable Attention for Autonomous Driving in a Single Decoder 

**Title (ZH)**: HiP-AD：具有可变形注意力的分层和多粒度规划在单一解码器中的自动驾驶 

**Authors**: Yingqi Tang, Zhuoran Xu, Zhaotie Meng, Erkang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.08612)  

**Abstract**: Although end-to-end autonomous driving (E2E-AD) technologies have made significant progress in recent years, there remains an unsatisfactory performance on closed-loop evaluation. The potential of leveraging planning in query design and interaction has not yet been fully explored. In this paper, we introduce a multi-granularity planning query representation that integrates heterogeneous waypoints, including spatial, temporal, and driving-style waypoints across various sampling patterns. It provides additional supervision for trajectory prediction, enhancing precise closed-loop control for the ego vehicle. Additionally, we explicitly utilize the geometric properties of planning trajectories to effectively retrieve relevant image features based on physical locations using deformable attention. By combining these strategies, we propose a novel end-to-end autonomous driving framework, termed HiP-AD, which simultaneously performs perception, prediction, and planning within a unified decoder. HiP-AD enables comprehensive interaction by allowing planning queries to iteratively interact with perception queries in the BEV space while dynamically extracting image features from perspective views. Experiments demonstrate that HiP-AD outperforms all existing end-to-end autonomous driving methods on the closed-loop benchmark Bench2Drive and achieves competitive performance on the real-world dataset nuScenes. 

**Abstract (ZH)**: 尽管近年来端到端自主驾驶（E2E-AD）技术取得了显著进展，但在闭环评估中仍存在不满意的表现。查询设计和交互中规划的潜力尚未充分开发。本文介绍了一种多粒度规划查询表示，整合了包括空间、时间、驾驶风格等多种采样模式的异构航点。它为轨迹预测提供了额外监督，增强了对ego车辆的精确闭环控制。此外，我们显式利用规划轨迹的几何特性，基于物理位置使用可变形注意力有效检索相关图像特征。通过结合这些策略，我们提出了一个新颖的端到端自主驾驶框架HiP-AD，它在一个统一的解码器中同时执行感知、预测和规划。HiP-AD通过允许规划查询与感知查询在BEV空间中迭代交互，并动态从视角中提取图像特征，实现了全面的交互。实验结果表明，HiP-AD在Bench2Drive闭环基准上优于所有现有的端到端自主驾驶方法，并在真实世界数据集nuScenes上表现竞争力。 

---
# Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies 

**Title (ZH)**: 无需故障数据的故障检测是否可行？基于模仿学习策略的不确定性感知运行时故障检测 

**Authors**: Chen Xu, Tony Khuong Nguyen, Emma Dixon, Christopher Rodriguez, Patrick Miller, Robert Lee, Paarth Shah, Rares Ambrus, Haruki Nishimura, Masha Itkina  

**Link**: [PDF](https://arxiv.org/pdf/2503.08558)  

**Abstract**: Recent years have witnessed impressive robotic manipulation systems driven by advances in imitation learning and generative modeling, such as diffusion- and flow-based approaches. As robot policy performance increases, so does the complexity and time horizon of achievable tasks, inducing unexpected and diverse failure modes that are difficult to predict a priori. To enable trustworthy policy deployment in safety-critical human environments, reliable runtime failure detection becomes important during policy inference. However, most existing failure detection approaches rely on prior knowledge of failure modes and require failure data during training, which imposes a significant challenge in practicality and scalability. In response to these limitations, we present FAIL-Detect, a modular two-stage approach for failure detection in imitation learning-based robotic manipulation. To accurately identify failures from successful training data alone, we frame the problem as sequential out-of-distribution (OOD) detection. We first distill policy inputs and outputs into scalar signals that correlate with policy failures and capture epistemic uncertainty. FAIL-Detect then employs conformal prediction (CP) as a versatile framework for uncertainty quantification with statistical guarantees. Empirically, we thoroughly investigate both learned and post-hoc scalar signal candidates on diverse robotic manipulation tasks. Our experiments show learned signals to be mostly consistently effective, particularly when using our novel flow-based density estimator. Furthermore, our method detects failures more accurately and faster than state-of-the-art (SOTA) failure detection baselines. These results highlight the potential of FAIL-Detect to enhance the safety and reliability of imitation learning-based robotic systems as they progress toward real-world deployment. 

**Abstract (ZH)**: Recent年份见证了由模仿学习和生成建模进步驱动的机器人操作系统的 impressive 进展，如扩散-基于流动的方法。随着机器人策略性能的提升，可实现的任务复杂性和时间范围也随之增加，导致难以事先预测的多种意外失败模式。为在安全关键的人类环境中实现可信的策略部署，在策略推断过程中可靠的运行时故障检测变得至关重要。然而，现有的大多数故障检测方法依赖于故障模式的先验知识，并且在训练过程中需要故障数据，这在实际应用中带来了显著的挑战和可扩展性问题。针对这些局限性，我们提出了一种模块化的两阶段方法 FAIL-Detect，用于基于模仿学习的机器人操作故障检测。通过仅从成功的训练数据中准确识别故障，我们将问题归结为序贯离分布（OOD）检测。我们首先将策略输入和输出提炼成与策略失败相关联的标量信号，并捕捉认识论不确定性。然后，FAIL-Detect 使用一致性预测（CP）作为不确定性量化的一个通用框架，带有统计保证。实验中，我们在多种机器人操作任务上全面研究了所学习和后验标量信号候选者。我们的实验结果显示，所学习的信号在大多数情况下都有效地工作，特别是在使用我们新颖的基于流动的概率分布估计器时。此外，我们的方法比现有最先进的（SOTA）故障检测基线更准确且更快地检测故障。这些结果突显了 FAIL-Detect 有潜力增强基于模仿学习的机器人系统在推向实际部署过程中的安全性和可靠性。 

---
# Collaborative Dynamic 3D Scene Graphs for Open-Vocabulary Urban Scene Understanding 

**Title (ZH)**: 协作动态3D场景图在开放词汇城市场景理解中的应用 

**Authors**: Tim Steinke, Martin Büchner, Niclas Vödisch, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2503.08474)  

**Abstract**: Mapping and scene representation are fundamental to reliable planning and navigation in mobile robots. While purely geometric maps using voxel grids allow for general navigation, obtaining up-to-date spatial and semantically rich representations that scale to dynamic large-scale environments remains challenging. In this work, we present CURB-OSG, an open-vocabulary dynamic 3D scene graph engine that generates hierarchical decompositions of urban driving scenes via multi-agent collaboration. By fusing the camera and LiDAR observations from multiple perceiving agents with unknown initial poses, our approach generates more accurate maps compared to a single agent while constructing a unified open-vocabulary semantic hierarchy of the scene. Unlike previous methods that rely on ground truth agent poses or are evaluated purely in simulation, CURB-OSG alleviates these constraints. We evaluate the capabilities of CURB-OSG on real-world multi-agent sensor data obtained from multiple sessions of the Oxford Radar RobotCar dataset. We demonstrate improved mapping and object prediction accuracy through multi-agent collaboration as well as evaluate the environment partitioning capabilities of the proposed approach. To foster further research, we release our code and supplementary material at this https URL. 

**Abstract (ZH)**: 基于多Agent合作的开放词汇动态3D场景图引擎CURB-OSG：城市驾驶场景的层次分解与语义表示 

---
# Optimizing Ride-Pooling Operations with Extended Pickup and Drop-Off Flexibility 

**Title (ZH)**: 优化拼车运营以扩大接乘客和_drop_off_灵活性 

**Authors**: Hao Jiang, Yixing Xu, Pradeep Varakantham  

**Link**: [PDF](https://arxiv.org/pdf/2503.08472)  

**Abstract**: The Ride-Pool Matching Problem (RMP) is central to on-demand ride-pooling services, where vehicles must be matched with multiple requests while adhering to service constraints such as pickup delays, detour limits, and vehicle capacity. Most existing RMP solutions assume passengers are picked up and dropped off at their original locations, neglecting the potential for passengers to walk to nearby spots to meet vehicles. This assumption restricts the optimization potential in ride-pooling operations. In this paper, we propose a novel matching method that incorporates extended pickup and drop-off areas for passengers. We first design a tree-based approach to efficiently generate feasible matches between passengers and vehicles. Next, we optimize vehicle routes to cover all designated pickup and drop-off locations while minimizing total travel distance. Finally, we employ dynamic assignment strategies to achieve optimal matching outcomes. Experiments on city-scale taxi datasets demonstrate that our method improves the number of served requests by up to 13\% and average travel distance by up to 21\% compared to leading existing solutions, underscoring the potential of leveraging passenger mobility to significantly enhance ride-pooling service efficiency. 

**Abstract (ZH)**: 基于乘车共享的服务匹配问题（RMP）对于按需乘车共享服务至关重要，其中车辆需要与多个请求匹配，同时遵守服务约束，如接客延迟、绕行限制和车辆载客量。现有的大多数RMP解决方案假设乘客在原始地点上下车，忽略了乘客步行至附近地点以遇见到达车辆的可能性。这一假设限制了乘车共享操作的优化潜力。本文提出了一种新的匹配方法，将乘客的延长上车和下车区域纳入考虑。首先，设计了一种基于树的方法来高效生成乘客与车辆之间的可行匹配。其次，优化车辆路线以覆盖所有指定的上车和下车地点，同时尽量减少总行驶距离。最后，采用动态分配策略以达到最优的匹配结果。基于城市规模的出租车数据集的实验表明，与现有的领先解决方案相比，我们的方法可将服务的请求数量最多提高13%，平均行驶距离最多减少21%，这突显了利用乘客机动性以显著提高乘车共享服务效率的潜力。 

---
# Dynamic Risk Assessment for Human-Robot Collaboration Using a Heuristics-based Approach 

**Title (ZH)**: 基于启发式方法的人机协作动态风险评估 

**Authors**: Georgios Katranis, Frederik Plahl, Joachim Grimstadt, Ilshat Mamaev, Silvia Vock, Andrey Morozov  

**Link**: [PDF](https://arxiv.org/pdf/2503.08316)  

**Abstract**: Human-robot collaboration (HRC) introduces significant safety challenges, particularly in protecting human operators working alongside collaborative robots (cobots). While current ISO standards emphasize risk assessment and hazard identification, these procedures are often insufficient for addressing the complexity of HRC environments, which involve numerous design factors and dynamic interactions. This publication presents a method for objective hazard analysis to support Dynamic Risk Assessment, extending beyond reliance on expert knowledge. The approach monitors scene parameters, such as the distance between human body parts and the cobot, as well as the cobot`s Cartesian velocity. Additionally, an anthropocentric parameter focusing on the orientation of the human head within the collaborative workspace is introduced. These parameters are transformed into hazard indicators using non-linear heuristic functions. The hazard indicators are then aggregated to estimate the total hazard level of a given scenario. The proposed method is evaluated using an industrial dataset that depicts various interactions between a human operator and a cobot. 

**Abstract (ZH)**: 人类与机器人协作（HRC）引入了显著的安全挑战，特别是在保护与协作机器人（cobot）并肩工作的操作员方面。当前的ISO标准侧重于风险评估和危害识别，但这些程序往往不足以应对HRC环境的复杂性，这种环境涉及众多设计因素和动态交互。本研究提出了一种客观危害分析方法以支持动态风险评估，超越了对专家知识的依赖。该方法监控场景参数，如人类身体部位与协作机器人之间的距离以及协作机器人的笛卡尔速度。此外，引入了一个以人为本的参数，重点关注人类头部在协作工作空间内的方位。这些参数通过非线性启发式函数转换为危害指标。最后，将这些危害指标聚合以估算给定场景的总体危害水平。该提议的方法使用描述人类操作员与协作机器人多种交互的工业数据集进行评估。 

---
# ForceGrip: Data-Free Curriculum Learning for Realistic Grip Force Control in VR Hand Manipulation 

**Title (ZH)**: ForceGrip：无需数据的层次学习方法在VR手部操作中实现现实的握力控制 

**Authors**: DongHeun Han, Byungmin Kim, RoUn Lee, KyeongMin Kim, Hyoseok Hwang, HyeongYeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.08061)  

**Abstract**: Realistic hand manipulation is a key component of immersive virtual reality (VR), yet existing methods often rely on a kinematic approach or motion-capture datasets that omit crucial physical attributes such as contact forces and finger torques. Consequently, these approaches prioritize tight, one-size-fits-all grips rather than reflecting users' intended force levels. We present ForceGrip, a deep learning agent that synthesizes realistic hand manipulation motions, faithfully reflecting the user's grip force intention. Instead of mimicking predefined motion datasets, ForceGrip uses generated training scenarios-randomizing object shapes, wrist movements, and trigger input flows-to challenge the agent with a broad spectrum of physical interactions. To effectively learn from these complex tasks, we employ a three-phase curriculum learning framework comprising Finger Positioning, Intention Adaptation, and Dynamic Stabilization. This progressive strategy ensures stable hand-object contact, adaptive force control based on user inputs, and robust handling under dynamic conditions. Additionally, a proximity reward function enhances natural finger motions and accelerates training convergence. Quantitative and qualitative evaluations reveal ForceGrip's superior force controllability and plausibility compared to state-of-the-art methods. 

**Abstract (ZH)**: 基于深度学习的ForceGrip：真实的手部操作运动合成 

---
# PLK-Calib: Single-shot and Target-less LiDAR-Camera Extrinsic Calibration using Plücker Lines 

**Title (ZH)**: PLK-Calib: 单步目标无关LiDAR-相机外参标定方法基于Plücker线。 

**Authors**: Yanyu Zhang, Jie Xu, Wei Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.07955)  

**Abstract**: Accurate LiDAR-Camera (LC) calibration is challenging but crucial for autonomous systems and robotics. In this paper, we propose two single-shot and target-less algorithms to estimate the calibration parameters between LiDAR and camera using line features. The first algorithm constructs line-to-line constraints by defining points-to-line projection errors and minimizes the projection error. The second algorithm (PLK-Calib) utilizes the co-perpendicular and co-parallel geometric properties of lines in Plücker (PLK) coordinate, and decouples the rotation and translation into two constraints, enabling more accurate estimates. Our degenerate analysis and Monte Carlo simulation indicate that three nonparallel line pairs are the minimal requirements to estimate the extrinsic parameters. Furthermore, we collect an LC calibration dataset with varying extrinsic under three different scenarios and use it to evaluate the performance of our proposed algorithms. 

**Abstract (ZH)**: 精确的激光雷达-相机（LC）标定对于自主系统和机器人技术至关重要但极具挑战性。本文提出两种单步无靶标算法，利用直线特征估算激光雷达和相机之间的标定参数。第一种算法通过定义点到直线投影误差并最小化投影误差构造直线对直线的约束。第二种算法（PLK-Calib）利用Plücker（PLK）坐标系中直线共垂和共平行的几何特性，将旋转和平移分离开来，以获得更准确的估计。我们的退化分析和蒙特卡洛模拟表明，估计外参需要三条非平行直线对。此外，我们在三种不同场景下收集了具有变化外参的LC标定数据集，并用于评估我们算法的性能。 

---
# A Task and Motion Planning Framework Using Iteratively Deepened AND/OR Graph Networks 

**Title (ZH)**: 使用迭代加深的AND/OR图网络的任务与运动规划框架 

**Authors**: Hossein Karami, Antony Thomas, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2503.07700)  

**Abstract**: In this paper, we present an approach for integrated task and motion planning based on an AND/OR graph network, which is used to represent task-level states and actions, and we leverage it to implement different classes of task and motion planning problems (TAMP). Several problems that fall under task and motion planning do not have a predetermined number of sub-tasks to achieve a goal. For example, while retrieving a target object from a cluttered workspace, in principle the number of object re-arrangements required to finally grasp it cannot be known ahead of time. To address this challenge, and in contrast to traditional planners, also those based on AND/OR graphs, we grow the AND/OR graph at run-time by progressively adding sub-graphs until grasping the target object becomes feasible, which yields a network of AND/OR graphs. The approach is extended to enable multi-robot task and motion planning, and (i) it allows us to perform task allocation while coordinating the activity of a given number of robots, and (ii) can handle multi-robot tasks involving an a priori unknown number of sub-tasks. The approach is evaluated and validated both in simulation and with a real dual-arm robot manipulator, that is, Baxter from Rethink Robotics. In particular, for the single-robot task and motion planning, we validated our approach in three different TAMP domains. Furthermore, we also use three different robots for simulation, namely, Baxter, Franka Emika Panda manipulators, and a PR2 robot. Experiments show that our approach can be readily scaled to scenarios with many objects and robots, and is capable of handling different classes of TAMP problems. 

**Abstract (ZH)**: 基于AND/OR图网络的任务与运动集成规划方法及其在多机器人系统中的扩展与验证 

---
# ICPR 2024 Competition on Rider Intention Prediction 

**Title (ZH)**: ICPR 2024 摩托车手意图预测竞赛 

**Authors**: Shankar Gangisetty, Abdul Wasi, Shyam Nandan Rai, C. V. Jawahar, Sajay Raj, Manish Prajapati, Ayesha Choudhary, Aaryadev Chandra, Dev Chandan, Shireen Chand, Suvaditya Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.08437)  

**Abstract**: The recent surge in the vehicle market has led to an alarming increase in road accidents. This underscores the critical importance of enhancing road safety measures, particularly for vulnerable road users like motorcyclists. Hence, we introduce the rider intention prediction (RIP) competition that aims to address challenges in rider safety by proactively predicting maneuvers before they occur, thereby strengthening rider safety. This capability enables the riders to react to the potential incorrect maneuvers flagged by advanced driver assistance systems (ADAS). We collect a new dataset, namely, rider action anticipation dataset (RAAD) for the competition consisting of two tasks: single-view RIP and multi-view RIP. The dataset incorporates a spectrum of traffic conditions and challenging navigational maneuvers on roads with varying lighting conditions. For the competition, we received seventy-five registrations and five team submissions for inference of which we compared the methods of the top three performing teams on both the RIP tasks: one state-space model (Mamba2) and two learning-based approaches (SVM and CNN-LSTM). The results indicate that the state-space model outperformed the other methods across the entire dataset, providing a balanced performance across maneuver classes. The SVM-based RIP method showed the second-best performance when using random sampling and SMOTE. However, the CNN-LSTM method underperformed, primarily due to class imbalance issues, particularly struggling with minority classes. This paper details the proposed RAAD dataset and provides a summary of the submissions for the RIP 2024 competition. 

**Abstract (ZH)**: 车辆市场recent的激增导致了道路事故的急剧增加。这突显了增强道路安全措施的重要性，特别是对摩托车骑行者等脆弱道路使用者。因此，我们引入了骑行者意图预测(RIP)竞赛，旨在通过前瞻性地预测潜在的骑行动作来解决骑行安全挑战，从而加强骑行者安全。这一能力使骑行者能够根据高级驾驶辅助系统(ADAS)检测到的潜在不当动作作出反应。我们收集了一个新的数据集，即骑行者动作预测数据集(RAAD)，其中包括单视图RIP和多视图RIP两个任务。该数据集涵盖了各种交通条件和道路上具有挑战性的导航动作，且光照条件各异。在竞赛中，我们收到了75个注册和5个团队提交进行推断，我们比较了前三个最佳团队在两个RIP任务中的方法：一个状态空间模型(Mamba2)和两个基于学习的方法(SVM和CNN-LSTM)。结果表明，状态空间模型在整个数据集中表现最佳，提供了在各种动作类别中的均衡性能。基于SVM的RIP方法在使用随机抽样和SMOTE时表现出第二好的性能。然而，基于CNN-LSTM的方法表现不佳，主要原因是类别不平衡问题，尤其是在处理少数类时。本文详细介绍了提出的RAAD数据集，并提供了RIP 2024竞赛提交的总结。 

---
# FunGraph: Functionality Aware 3D Scene Graphs for Language-Prompted Scene Interaction 

**Title (ZH)**: FunGraph: awareness of 功能的3D场景图用于语言提示的场景交互 

**Authors**: Dennis Rotondi, Fabio Scaparro, Hermann Blum, Kai O. Arras  

**Link**: [PDF](https://arxiv.org/pdf/2503.07909)  

**Abstract**: The concept of 3D scene graphs is increasingly recognized as a powerful semantic and hierarchical representation of the environment. Current approaches often address this at a coarse, object-level resolution. In contrast, our goal is to develop a representation that enables robots to directly interact with their environment by identifying both the location of functional interactive elements and how these can be used. To achieve this, we focus on detecting and storing objects at a finer resolution, focusing on affordance-relevant parts. The primary challenge lies in the scarcity of data that extends beyond instance-level detection and the inherent difficulty of capturing detailed object features using robotic sensors. We leverage currently available 3D resources to generate 2D data and train a detector, which is then used to augment the standard 3D scene graph generation pipeline. Through our experiments, we demonstrate that our approach achieves functional element segmentation comparable to state-of-the-art 3D models and that our augmentation enables task-driven affordance grounding with higher accuracy than the current solutions. 

**Abstract (ZH)**: 3D 场景图的概念日益被视为环境的有力语义和层次表示。当前的方法往往在粗粒度的物体级别上进行处理。相比之下，我们的目标是开发一种表示方法，使机器人能够直接与其环境交互，识别功能性交互元素的位置及其使用方式。为了实现这一目标，我们专注于在更细粒度级别上检测和存储物体，并重点关注功能相关的部分。主要挑战在于扩展到实例级别检测之外的数据稀缺性，以及机器人传感器捕捉详细物体特征的固有难度。我们利用现有的 3D 资源生成 2D 数据，并训练一个检测器，然后用于增强标准的 3D 场景图生成管道。通过我们的实验，我们证明了我们的方法与最先进的 3D 模型相比能够实现功能元素分割，并且我们的增强方法能够以高于当前解决方案的准确性进行任务驱动的功能定位。 

---
# HIPPO-MAT: Decentralized Task Allocation Using GraphSAGE and Multi-Agent Deep Reinforcement Learning 

**Title (ZH)**: HIPPO-MAT：基于GraphSAGE和多智能体深度强化学习的去中心化任务分配 

**Authors**: Lavanya Ratnabala, Robinroy Peter, Aleksey Fedoseev, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2503.07662)  

**Abstract**: This paper tackles decentralized continuous task allocation in heterogeneous multi-agent systems. We present a novel framework HIPPO-MAT that integrates graph neural networks (GNN) employing a GraphSAGE architecture to compute independent embeddings on each agent with an Independent Proximal Policy Optimization (IPPO) approach for multi-agent deep reinforcement learning. In our system, unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs) share aggregated observation data via communication channels while independently processing these inputs to generate enriched state embeddings. This design enables dynamic, cost-optimal, conflict-aware task allocation in a 3D grid environment without the need for centralized coordination. A modified A* path planner is incorporated for efficient routing and collision avoidance. Simulation experiments demonstrate scalability with up to 30 agents and preliminary real-world validation on JetBot ROS AI Robots, each running its model on a Jetson Nano and communicating through an ESP-NOW protocol using ESP32-S3, which confirms the practical viability of the approach that incorporates simultaneous localization and mapping (SLAM). Experimental results revealed that our method achieves a high 92.5% conflict-free success rate, with only a 16.49% performance gap compared to the centralized Hungarian method, while outperforming the heuristic decentralized baseline based on greedy approach. Additionally, the framework exhibits scalability with up to 30 agents with allocation processing of 0.32 simulation step time and robustness in responding to dynamically generated tasks. 

**Abstract (ZH)**: 基于图神经网络的异构多智能体系统的去中心化连续任务分配框架 

---
# Impact of Level 2/3 Automated Driving Technology on Road Work Zone Safety 

**Title (ZH)**: Level 2/3自动驾驶技术对道路施工区安全的影响 

**Authors**: Zhepu Xu, Ziyi Song, Yupu Dong, Peiyan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07634)  

**Abstract**: As China's road network enters the maintenance era, work zones will become a common sight on the roads. With the development of automated driving, vehicles equipped with Level 2/3 automated driving capabilities will also become a common presence on the roads. When these vehicles pass through work zones, automated driving may disengage, which can have complex effects on traffic safety. This paper explores the impact of Level 2/3 automated driving technology on road safety in high-speed highway work zone environments. Through microscopic traffic simulation method and using full-type traffic conflict technique, factors such as market penetration rate (MPR), traffic volume level, disengagement threshold, and driver takeover style are studied to understand their impact on work zone safety. The study found that the impact of automated driving technology on work zone safety is complex. Disengagement of automated vehicles in work zones reduces the proportion of vehicles that can maintain automated driving status. If takeover is not timely or adequate, it can easily lead to new traffic conflicts. Different factors have varying degrees of impact on work zone safety. Increasing MPR helps reduce the occurrence of single-vehicle conflicts, but it also increases the possibility of multi-vehicle conflicts. Therefore, future research and improvement directions should focus on optimizing the disengagement detection and takeover mechanisms of automated driving systems. 

**Abstract (ZH)**: 中国道路交通维护时代的自动驾驶技术对高速公路工作区交通安全的影响研究 

---
# Chain-of-Thought Reasoning In The Wild Is Not Always Faithful 

**Title (ZH)**: 野外的链式思维推理并不总是可靠的。 

**Authors**: Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2503.08679)  

**Abstract**: Chain-of-Thought (CoT) reasoning has significantly advanced state-of-the-art AI capabilities. However, recent studies have shown that CoT reasoning is not always faithful, i.e. CoT reasoning does not always reflect how models arrive at conclusions. So far, most of these studies have focused on unfaithfulness in unnatural contexts where an explicit bias has been introduced. In contrast, we show that unfaithful CoT can occur on realistic prompts with no artificial bias. Our results reveal concerning rates of several forms of unfaithful reasoning in frontier models: Sonnet 3.7 (30.6%), DeepSeek R1 (15.8%) and ChatGPT-4o (12.6%) all answer a high proportion of question pairs unfaithfully. Specifically, we find that models rationalize their implicit biases in answers to binary questions ("implicit post-hoc rationalization"). For example, when separately presented with the questions "Is X bigger than Y?" and "Is Y bigger than X?", models sometimes produce superficially coherent arguments to justify answering Yes to both questions or No to both questions, despite such responses being logically contradictory. We also investigate restoration errors (Dziri et al., 2023), where models make and then silently correct errors in their reasoning, and unfaithful shortcuts, where models use clearly illogical reasoning to simplify solving problems in Putnam questions (a hard benchmark). Our findings raise challenges for AI safety work that relies on monitoring CoT to detect undesired behavior. 

**Abstract (ZH)**: Chain-of-Thought 理论在先进人工智能能力中的应用已经取得了显著进展，然而最近的研究表明，Chain-of-Thought 理论并非总是忠实的，即它并不总是反映模型如何得出结论。迄今为止，大多数相关研究主要集中在那些故意引入了非自然偏见的不忠实上下文中。相比之下，我们展示了在没有人为偏见的现实提示下也会出现不忠实的 Chain-of-Thought。我们的研究结果揭示了前沿模型中几种形式的不忠实推理的较高频率：Sonnet 3.7（30.6%）、DeepSeek R1（15.8%）和 ChatGPT-4o（12.6%）都以较高的比例不忠实回答了问题对。具体来说，我们发现模型在回答二元问题时通过隐性后验理性化来为自己辩解（“隐性后验理性化”）。例如，当分别展示“X 是否比 Y 大？”和“Y 是否比 X 大？”这两个问题时，模型有时会生成表面上连贯的论据来证明对两个问题都回答“是”或“否”，尽管这样的回答在逻辑上是矛盾的。我们还研究了修复错误（Dziri et al., 2023），即模型在推理过程中犯错误并默默纠正，以及不忠实的捷径，即模型使用明显不合逻辑的推理简化解决培土姆难题中的问题。我们的发现对依赖监控 Chain-of-Thought 以检测不 desired 行为的人工智能安全工作提出了挑战。 

---
# STGDPM:Vessel Trajectory Prediction with Spatio-Temporal Graph Diffusion Probabilistic Model 

**Title (ZH)**: STGDPM：基于时空图扩散概率模型的血管轨迹预测 

**Authors**: Jin Wenzhe, Tang Haina, Zhang Xudong  

**Link**: [PDF](https://arxiv.org/pdf/2503.08065)  

**Abstract**: Vessel trajectory prediction is a critical component for ensuring maritime traffic safety and avoiding collisions. Due to the inherent uncertainty in vessel behavior, trajectory prediction systems must adopt a multimodal approach to accurately model potential future motion states. However, existing vessel trajectory prediction methods lack the ability to comprehensively model behavioral multi-modality. To better capture multimodal behavior in interactive scenarios, we propose modeling interactions as dynamic graphs, replacing traditional aggregation-based techniques that rely on vessel states. By leveraging the natural multimodal capabilities of diffusion models, we frame the trajectory prediction task as an inverse process of motion uncertainty diffusion, wherein uncertainties across potential navigational areas are progressively eliminated until the desired trajectories is produced. In summary, we pioneer the integration of Spatio-Temporal Graph (STG) with diffusion models in ship trajectory prediction. Extensive experiments on real Automatic Identification System (AIS) data validate the superiority of our approach. 

**Abstract (ZH)**: 船舶轨迹预测是确保海上交通安全和避免碰撞的关键组件。由于船舶行为固有的不确定性，轨迹预测系统必须采用多模式方法来准确建模潜在的未来运动状态。然而，现有的船舶轨迹预测方法无法全面建模行为的多模态性。为更好地捕捉交互场景中的多模态行为，我们提出将交互建模为动态图，替代依赖于船舶状态的传统聚合方法。通过利用扩散模型的自然多模态能力，我们将轨迹预测任务框架化为运动不确定性扩散的逆过程，在此过程中，潜在导航区域的不确定性逐渐消除，直至产生所需的轨迹。总之，我们首次将空间时间图（STG）与扩散模型集成应用于船舶轨迹预测。广泛实验验证了我们方法的优越性。 

---
# SQLCritic: Correcting Text-to-SQL Generation via Clause-wise Critic 

**Title (ZH)**: SQLCritic: 通过子句级别批评纠正文本到SQL生成 

**Authors**: Jikai Chen, Leilei Gan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07996)  

**Abstract**: Recent advancements in Text-to-SQL systems have improved the conversion of natural language queries into SQL, but challenges remain in ensuring accuracy and reliability. While self-correction techniques refine outputs, they often introduce new errors. Existing methods focused on execution feedback mainly address syntax issues, leaving semantic errors -- where the query's logic fails to align with the user's intent -- largely unaddressed.
We propose a novel approach combining structured execution feedback with a trained critic agent that provides detailed, interpretable critiques. This method effectively identifies and corrects both syntactic and semantic errors, enhancing accuracy and interpretability. Experimental results show significant improvements on two major Text-to-SQL benchmarks, Spider and BIRD, demonstrating the effectiveness of our approach. 

**Abstract (ZH)**: Recent advancements in Text-to-SQL系统提高了自然语言查询到SQL的转换效率，但确保准确性和可靠性仍面临挑战。尽管自我修正技术可以改进输出，但往往会引入新的错误。现有方法主要依赖执行反馈来解决语法问题，而未能充分处理语义错误，即查询逻辑与用户意图不一致的问题。我们提出了一种结合结构化执行反馈和训练过的批评代理的新方法，该代理能够提供详细的、可解释的批评。此方法有效识别和修正了语法和语义错误，提高了准确性和可解释性。实验结果在Spider和BIRD两大Text-to-SQL基准上显示出显著改进，证明了该方法的有效性。 

---
# Boundary Prompting: Elastic Urban Region Representation via Graph-based Spatial Tokenization 

**Title (ZH)**: 边界提示：基于图的空间词元化弹性城市区域表示 

**Authors**: Haojia Zhu, Jiahui Jin, Dong Kan, Rouxi Shen, Ruize Wang, Xiangguo Sun, Jinghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07991)  

**Abstract**: Urban region representation is essential for various applications such as urban planning, resource allocation, and policy development. Traditional methods rely on fixed, predefined region boundaries, which fail to capture the dynamic and complex nature of real-world urban areas. In this paper, we propose the Boundary Prompting Urban Region Representation Framework (BPURF), a novel approach that allows for elastic urban region definitions. BPURF comprises two key components: (1) A spatial token dictionary, where urban entities are treated as tokens and integrated into a unified token graph, and (2) a region token set representation model which utilize token aggregation and a multi-channel model to embed token sets corresponding to region boundaries. Additionally, we propose fast token set extraction strategy to enable online token set extraction during training and prompting. This framework enables the definition of urban regions through boundary prompting, supporting varying region boundaries and adapting to different tasks. Extensive experiments demonstrate the effectiveness of BPURF in capturing the complex characteristics of urban regions. 

**Abstract (ZH)**: 基于边界提示的城市区域表示框架（BPURF）：弹性城市区域定义方法 

---
# BEARCUBS: A benchmark for computer-using web agents 

**Title (ZH)**: BEARCUBS: 一种用于计算机使用网页代理的基准测试 

**Authors**: Yixiao Song, Katherine Thai, Chau Minh Pham, Yapei Chang, Mazin Nadaf, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2503.07919)  

**Abstract**: Modern web agents possess computer use abilities that allow them to interact with webpages by sending commands to a virtual keyboard and mouse. While such agents have considerable potential to assist human users with complex tasks, evaluating their capabilities in real-world settings poses a major challenge. To this end, we introduce BEARCUBS, a "small but mighty" benchmark of 111 information-seeking questions designed to evaluate a web agent's ability to search, browse, and identify factual information from the web. Unlike prior web agent benchmarks, solving BEARCUBS requires (1) accessing live web content rather than synthetic or simulated pages, which captures the unpredictability of real-world web interactions; and (2) performing a broad range of multimodal interactions (e.g., video understanding, 3D navigation) that cannot be bypassed via text-based workarounds. Each question in BEARCUBS has a corresponding short, unambiguous answer and a human-validated browsing trajectory, allowing for transparent evaluation of agent performance and strategies. A human study confirms that BEARCUBS questions are solvable but non-trivial (84.7% human accuracy), revealing search inefficiencies and domain knowledge gaps as common failure points. By contrast, state-of-the-art computer-using agents underperform, with the best-scoring system (OpenAI's Operator) reaching only 24.3% accuracy. These results highlight critical areas for improvement, including reliable source selection and more powerful multimodal capabilities. To facilitate future research, BEARCUBS will be updated periodically to replace invalid or contaminated questions, keeping the benchmark fresh for future generations of web agents. 

**Abstract (ZH)**: 现代网络代理具有通过发送命令控制虚拟键盘和鼠标来与网页交互的能力。尽管这些代理在协助用户完成复杂任务方面具有巨大的潜力，但在实际环境中评估其能力是一项重大挑战。为了解决这一问题，我们引入了BEARCUBS基准测试，这是一个包含111个信息检索问题的“小巧但强大”的评估体系，旨在评估网络代理在搜索、浏览和从网络中识别事实信息方面的能力。与之前的网络代理基准测试不同，解决BEARCUBS需要访问实际网页内容而非合成或模拟页面，以捕捉实际世界网络交互的不可预测性；并且需要进行多种模态的交互（例如，视频理解、3D导航），这些交互无法通过基于文本的变通办法绕过。每个BEARCUBS问题都有一个对应的具体和明确的答案以及经过人工验证的浏览路径，这使得代理性能和策略的透明评估成为可能。人类研究证实，BEARCUBS问题是可解但非简单的（人类准确率为84.7%），揭示了搜索效率低下和领域知识不足是常见的失败点。相比之下，最先进的计算机使用代理的表现不尽如人意，得分最高的系统（OpenAI的Operator）的准确率仅为24.3%。这些结果强调了可靠的信息源选择和更强大的多模态能力等关键改进领域。为了促进未来的研究，BEARCUBS将定期更新，替换无效或污染的问题，使基准测试保持新鲜，以适应未来一代的网络代理。 

---
# Demystifying the Accuracy-Interpretability Trade-Off: A Case Study of Inferring Ratings from Reviews 

**Title (ZH)**: 揭开准确性和可解释性权衡的神秘面纱：从评价推断评分的案例研究 

**Authors**: Pranjal Atrey, Michael P. Brundage, Min Wu, Sanghamitra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2503.07914)  

**Abstract**: Interpretable machine learning models offer understandable reasoning behind their decision-making process, though they may not always match the performance of their black-box counterparts. This trade-off between interpretability and model performance has sparked discussions around the deployment of AI, particularly in critical applications where knowing the rationale of decision-making is essential for trust and accountability. In this study, we conduct a comparative analysis of several black-box and interpretable models, focusing on a specific NLP use case that has received limited attention: inferring ratings from reviews. Through this use case, we explore the intricate relationship between the performance and interpretability of different models. We introduce a quantitative score called Composite Interpretability (CI) to help visualize the trade-off between interpretability and performance, particularly in the case of composite models. Our results indicate that, in general, the learning performance improves as interpretability decreases, but this relationship is not strictly monotonic, and there are instances where interpretable models are more advantageous. 

**Abstract (ZH)**: 可解释的机器学习模型提供可理解的决策推理过程，尽管它们的性能可能不如其黑盒 counterparts。在这权衡可解释性和模型性能的讨论中，特别是在需要了解决策理由以建立信任和问责制的关键应用中，AI 的部署受到了广泛关注。在本研究中，我们对几种黑盒和可解释模型进行了比较分析，重点关注一个受到较少关注的自然语言处理应用案例：从评论推断评分。通过这一应用案例，我们探索了不同模型的性能与可解释性之间的复杂关系。我们引入了一个名为综合可解释性（CI）的量化评分，以帮助可视化可解释性和性能之间的权衡，特别是在复合模型的情况下。我们的结果显示，总体而言，随着可解释性的降低，学习性能会提高，但这种关系并非严格单调，存在可解释模型更为有利的情况。 

---
# Actual Causation and Nondeterministic Causal Models 

**Title (ZH)**: 实际因果关系与非确定性因果模型 

**Authors**: Sander Beckers  

**Link**: [PDF](https://arxiv.org/pdf/2503.07849)  

**Abstract**: In (Beckers, 2025) I introduced nondeterministic causal models as a generalization of Pearl's standard deterministic causal models. I here take advantage of the increased expressivity offered by these models to offer a novel definition of actual causation (that also applies to deterministic models). Instead of motivating the definition by way of (often subjective) intuitions about examples, I proceed by developing it based entirely on the unique function that it can fulfil in communicating and learning a causal model. First I generalize the more basic notion of counterfactual dependence, second I show how this notion has a vital role to play in the logic of causal discovery, third I introduce the notion of a structural simplification of a causal model, and lastly I bring both notions together in my definition of actual causation. Although novel, the resulting definition arrives at verdicts that are almost identical to those of my previous definition (Beckers, 2021, 2022). 

**Abstract (ZH)**: 在贝克尔斯（2025）中，我介绍了 nondeterministic 负因果模型作为佩尔标准确定性因果模型的推广。本文利用这些模型提供的增强表达能力，提出了一种新的实际因果定义（该定义也适用于确定性模型）。我并非通过（往往是主观的）关于示例的直觉来动机该定义，而是通过完全基于它在因果模型通信和学习中所扮演的独特角色来发展这一定义。首先，我将更基本的反事实依赖性概念一般化，其次，我展示了这一概念在因果发现逻辑中起着至关重要的作用，第三，我引入了因果模型结构简化的概念，最后，我将这两种概念结合起来，提出我的实际因果定义。尽管是新颖的，但由此得出的定义与我之前的定义（贝克尔斯，2021，2022）得出的结论几乎相同。 

---
# Safe Explicable Policy Search 

**Title (ZH)**: 安全可解释的策略搜索 

**Authors**: Akkamahadevi Hanni, Jonathan Montaño, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07848)  

**Abstract**: When users work with AI agents, they form conscious or subconscious expectations of them. Meeting user expectations is crucial for such agents to engage in successful interactions and teaming. However, users may form expectations of an agent that differ from the agent's planned behaviors. These differences lead to the consideration of two separate decision models in the planning process to generate explicable behaviors. However, little has been done to incorporate safety considerations, especially in a learning setting. We present Safe Explicable Policy Search (SEPS), which aims to provide a learning approach to explicable behavior generation while minimizing the safety risk, both during and after learning. We formulate SEPS as a constrained optimization problem where the agent aims to maximize an explicability score subject to constraints on safety and a suboptimality criterion based on the agent's model. SEPS innovatively combines the capabilities of Constrained Policy Optimization and Explicable Policy Search. We evaluate SEPS in safety-gym environments and with a physical robot experiment to show that it can learn explicable behaviors that adhere to the agent's safety requirements and are efficient. Results show that SEPS can generate safe and explicable behaviors while ensuring a desired level of performance w.r.t. the agent's objective, and has real-world relevance in human-AI teaming. 

**Abstract (ZH)**: Safe Explicable Policy Search：在确保安全性的前提下生成可解释行为的方法 

---
# AgentOrca: A Dual-System Framework to Evaluate Language Agents on Operational Routine and Constraint Adherence 

**Title (ZH)**: AgentOrca: 一种评估语言代理操作常规和约束遵守的双重系统框架 

**Authors**: Zekun Li, Shinda Huang, Jiangtian Wang, Nathan Zhang, Antonis Antoniades, Wenyue Hua, Kaijie Zhu, Sirui Zeng, William Yang Wang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08669)  

**Abstract**: As language agents progressively automate critical tasks across domains, their ability to operate within operational constraints and safety protocols becomes essential. While extensive research has demonstrated these agents' effectiveness in downstream task completion, their reliability in following operational procedures and constraints remains largely unexplored. To this end, we present AgentOrca, a dual-system framework for evaluating language agents' compliance with operational constraints and routines. Our framework encodes action constraints and routines through both natural language prompts for agents and corresponding executable code serving as ground truth for automated verification. Through an automated pipeline of test case generation and evaluation across five real-world domains, we quantitatively assess current language agents' adherence to operational constraints. Our findings reveal notable performance gaps among state-of-the-art models, with large reasoning models like o1 demonstrating superior compliance while others show significantly lower performance, particularly when encountering complex constraints or user persuasion attempts. 

**Abstract (ZH)**: 随着语言代理在各个领域逐步自动化关键任务，它们在操作约束和安全协议内的运行能力变得至关重要。虽然大量的研究已经证明了这些代理在下游任务完成上的有效性，但它们遵循操作程序和约束的可靠性仍很大程度上未被探索。为此，我们提出了AgentOrca，一种双系统框架，用于评估语言代理对操作约束和常规程序的遵守情况。该框架通过自然语言提示和相应的可执行代码来编码行动约束和常规程序，后者作为自动验证的标准。通过跨越五个实际领域的自动化测试用例生成和评估管道，我们定量评估当前语言代理遵守操作约束的情况。我们的研究发现，最先进的模型之间存在明显的性能差距，大型推理模型如o1显示出更好的合规性，而其他模型则表现显著较差，尤其是在面对复杂的约束或用户说服尝试时。 

---
# Rethinking Diffusion Model in High Dimension 

**Title (ZH)**: 重新思考高维空间中的扩散模型 

**Authors**: Zhenxin Zheng, Zhenjie Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.08643)  

**Abstract**: Curse of Dimensionality is an unavoidable challenge in statistical probability models, yet diffusion models seem to overcome this limitation, achieving impressive results in high-dimensional data generation. Diffusion models assume that they can learn the statistical properties of the underlying probability distribution, enabling sampling from this distribution to generate realistic samples. But is this really how they work? To address this question, this paper conducts a detailed analysis of the objective function and inference methods of diffusion models, leading to several important conclusions that help answer the above question: 1) In high-dimensional sparse scenarios, the target of the objective function fitting degrades from a weighted sum of multiple samples to a single sample. 2) The mainstream inference methods can all be represented within a simple unified framework, without requiring statistical concepts such as Markov chains and SDEs. 3) Guided by this simple framework, more efficient inference methods can be discovered. 

**Abstract (ZH)**: 高维稀疏场景下的维度灾难以统计概率模型为例是一个不可避免的挑战，而扩散模型似乎克服了这一限制，在高维数据生成中取得了令人印象深刻的成果。扩散模型假设它们能够学习潜在概率分布的统计性质，从而使从该分布中采样以生成现实样本成为可能。但这真的就是这样工作的吗？为了解决这个问题，本文对该类模型的目标函数和推理方法进行了详细分析，得出了一些重要结论，有助于回答上述问题：1）在高维稀疏场景下，目标函数拟合的目标从多个样本的加权和退化为单个样本。2）主流的推理方法都可以在一個简单的统一框架内表示，无需使用马尔可夫链和SDE等统计概念。3）在这一简单框架的指导下，可以发现更高效的推理方法。 

---
# When Discourse Stalls: Moving Past Five Semantic Stopsigns about Generative AI in Design Research 

**Title (ZH)**: 设计研究中对话受阻：超越生成式AI的五个语义路标 

**Authors**: Willem van der Maden, Vera van der Burg, Brett A. Halperin, Petra Jääskeläinen, Joseph Lindley, Derek Lomas, Timothy Merritt  

**Link**: [PDF](https://arxiv.org/pdf/2503.08565)  

**Abstract**: This essay examines how Generative AI (GenAI) is rapidly transforming design practices and how discourse often falls into over-simplified narratives that impede meaningful research and practical progress. We identify and deconstruct five prevalent "semantic stopsigns" -- reductive framings about GenAI in design that halt deeper inquiry and limit productive engagement. Reflecting upon two expert workshops at ACM conferences and semi-structured interviews with design practitioners, we analyze how these stopsigns manifest in research and practice. Our analysis develops mid-level knowledge that bridges theoretical discourse and practical implementation, helping designers and researchers interrogate common assumptions about GenAI in their own contexts. By recasting these stopsigns into more nuanced frameworks, we provide the design research community with practical approaches for thinking about and working with these emerging technologies. 

**Abstract (ZH)**: 这篇论文探讨了生成人工智能（GenAI）如何迅速变革设计实践，并分析了围绕GenAI的讨论往往陷入简化叙事，阻碍了有意义的研究和实际进展。我们识别并拆解了五种常见的“语义路障”——设计中关于GenAI的简化的框架化理解，这些理解阻碍了深入研究和有益的互动。通过反思ACM会议中两位专家的工作坊以及对设计从业者进行半结构化访谈，我们分析了这些路障在研究和实践中的表现形式。我们的分析构建了中等层次的知识，连接了理论讨论和实践实施，帮助设计师和研究者在其特定背景下质疑关于GenAI的常见假设。通过将这些路障重新构建成更微妙的框架，我们为设计研究社区提供了思考和运用这些新兴技术的实际方法。 

---
# A Triple-Inertial Accelerated Alternating Optimization Method for Deep Learning Training 

**Title (ZH)**: 三重惯性加速交替优化方法用于深度学习训练 

**Authors**: Chengcheng Yan, Jiawei Xu, Qingsong Wang, Zheng Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.08489)  

**Abstract**: The stochastic gradient descent (SGD) algorithm has achieved remarkable success in training deep learning models. However, it has several limitations, including susceptibility to vanishing gradients, sensitivity to input data, and a lack of robust theoretical guarantees. In recent years, alternating minimization (AM) methods have emerged as a promising alternative for model training by employing gradient-free approaches to iteratively update model parameters. Despite their potential, these methods often exhibit slow convergence rates. To address this challenge, we propose a novel Triple-Inertial Accelerated Alternating Minimization (TIAM) framework for neural network training. The TIAM approach incorporates a triple-inertial acceleration strategy with a specialized approximation method, facilitating targeted acceleration of different terms in each sub-problem optimization. This integration improves the efficiency of convergence, achieving superior performance with fewer iterations. Additionally, we provide a convergence analysis of the TIAM algorithm, including its global convergence properties and convergence rate. Extensive experiments validate the effectiveness of the TIAM method, showing significant improvements in generalization capability and computational efficiency compared to existing approaches, particularly when applied to the rectified linear unit (ReLU) and its variants. 

**Abstract (ZH)**: 三惯性加速交替极小化（TIAM）框架在神经网络训练中的应用及其收敛性分析 

---
# Accelerating MoE Model Inference with Expert Sharding 

**Title (ZH)**: 加速MoE模型推理 dengan 专家分割 

**Authors**: Oana Balmau, Anne-Marie Kermarrec, Rafael Pires, André Loureiro Espírito Santo, Martijn de Vos, Milos Vujasinovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.08467)  

**Abstract**: Mixture of experts (MoE) models achieve state-of-the-art results in language modeling but suffer from inefficient hardware utilization due to imbalanced token routing and communication overhead. While prior work has focused on optimizing MoE training and decoder architectures, inference for encoder-based MoE models in a multi-GPU with expert parallelism setting remains underexplored. We introduce MoEShard, an inference system that achieves perfect load balancing through tensor sharding of MoE experts. Unlike existing approaches that rely on heuristic capacity factors or drop tokens, MoEShard evenly distributes computation across GPUs and ensures full token retention, maximizing utilization regardless of routing skewness. We achieve this through a strategic row- and column-wise decomposition of expert matrices. This reduces idle time and avoids bottlenecks caused by imbalanced expert assignments. Furthermore, MoEShard minimizes kernel launches by fusing decomposed expert computations, significantly improving throughput. We evaluate MoEShard against DeepSpeed on encoder-based architectures, demonstrating speedups of up to 6.4$\times$ in time to first token (TTFT). Our results show that tensor sharding, when properly applied to experts, is a viable and effective strategy for efficient MoE inference. 

**Abstract (ZH)**: MoE专家模型通过张量分片实现高效的推理负载均衡 

---
# Status and Future Prospects of the Standardization Framework Industry 4.0: A European Perspective 

**Title (ZH)**: 基于欧洲视角的Industry 4.0标准框架现状及未来前景研究 

**Authors**: Olga Meyer, Marvin Boell, Christoph Legat  

**Link**: [PDF](https://arxiv.org/pdf/2503.08460)  

**Abstract**: The rapid development of Industry 4.0 technologies requires robust and comprehensive standardization to ensure interoperability, safety and efficiency in the Industry of the Future. This paper examines the fundamental role and functionality of standardization, with a particular focus on its importance in Europe's regulatory framework. Based on this, selected topics in context of standardization activities in context intelligent manufacturing and digital twins are highlighted and, by that, an overview of the Industry 4.0 standards framework is provided. This paper serves both as an informative guide to the existing standards in Industry 4.0 with respect to Artificial Intelligence and Digital Twins, and as a call to action for increased cooperation between standardization bodies and the research community. By fostering such collaboration, we aim to facilitate the continued development and implementation of standards that will drive innovation and progress in the manufacturing sector. 

**Abstract (ZH)**: 工业4.0技术的快速发展需要 robust 和全面的标准制定以确保未来工业中的互操作性、安全性和效率。本文探讨了标准制定的基础作用和功能，特别是在欧洲监管框架中的重要性，并强调了智能制造和数字孪生背景下标准活动的相关主题，从而提供了工业4.0标准框架的概览。本文不仅是关于人工智能和数字孪生的工业4.0现有标准的信息指南，也是呼吁标准制定机构与研究社区加强合作的呼吁。通过促进这种合作，我们旨在推动标准的持续发展与实施，从而在制造领域推动创新与进步。 

---
# InfluenceNet: AI Models for Banzhaf and Shapley Value Prediction 

**Title (ZH)**: InfluenceNet：AI模型在Banzhaf值和Shapley值预测中的应用 

**Authors**: Benjamin Kempinski, Tal Kachman  

**Link**: [PDF](https://arxiv.org/pdf/2503.08381)  

**Abstract**: Power indices are essential in assessing the contribution and influence of individual agents in multi-agent systems, providing crucial insights into collaborative dynamics and decision-making processes. While invaluable, traditional computational methods for exact or estimated power indices values require significant time and computational constraints, especially for large $(n\ge10)$ coalitions. These constraints have historically limited researchers' ability to analyse complex multi-agent interactions comprehensively. To address this limitation, we introduce a novel Neural Networks-based approach that efficiently estimates power indices for voting games, demonstrating comparable and often superiour performance to existing tools in terms of both speed and accuracy. This method not only addresses existing computational bottlenecks, but also enables rapid analysis of large coalitions, opening new avenues for multi-agent system research by overcoming previous computational limitations and providing researchers with a more accessible, scalable analytical this http URL increased efficiency will allow for the analysis of more complex and realistic multi-agent scenarios. 

**Abstract (ZH)**: 权力指数对于评估多agent系统中个体agent的贡献和影响力至关重要，提供了解协作动态和决策过程的关键见解。尽管如此，传统计算方法在准确或近似计算权力指数值方面需要大量时间和计算资源，尤其是对于较大的$(n\ge10)$联盟。这种限制使研究人员难以全面分析复杂的多agent交互。为了克服这一限制，我们引入了一种基于神经网络的新方法，该方法能够高效地估计投票游戏中的权力指数，其速度和准确度均与现有工具相当，甚至更优。该方法不仅解决了现有的计算瓶颈，还允许快速分析大型联盟，为多agent系统研究开辟了新的途径，克服了先前的计算限制，使研究人员能够获得更便捷、可扩展的分析工具。这将提高效率，使研究人员能够分析更加复杂和现实的多agent情景。 

---
# MINT-Demo: Membership Inference Test Demonstrator 

**Title (ZH)**: MINT-Demo: 成员推理测试演示器 

**Authors**: Daniel DeAlcala, Aythami Morales, Julian Fierrez, Gonzalo Mancera, Ruben Tolosana, Ruben Vera-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2503.08332)  

**Abstract**: We present the Membership Inference Test Demonstrator, to emphasize the need for more transparent machine learning training processes. MINT is a technique for experimentally determining whether certain data has been used during the training of machine learning models. We conduct experiments with popular face recognition models and 5 public databases containing over 22M images. Promising results, up to 89% accuracy are achieved, suggesting that it is possible to recognize if an AI model has been trained with specific data. Finally, we present a MINT platform as demonstrator of this technology aimed to promote transparency in AI training. 

**Abstract (ZH)**: 我们介绍了会员推理测试演示器，以强调需要更加透明的机器学习训练过程。MINT是一种实验性技术，用于确定某些数据是否在机器学习模型的训练过程中被使用。我们使用流行的面部识别模型和包含超过2200万张图像的5个公开数据库进行了实验。实验结果显示，准确率最高可达89%，表明可以通过此技术识别AI模型是否使用了特定数据进行训练。最后，我们展示了MINT平台作为此项技术的演示器，旨在促进AI训练过程的透明度。 

---
# Adding Chocolate to Mint: Mitigating Metric Interference in Machine Translation 

**Title (ZH)**: 将薄荷与巧克力相结合：缓解机器翻译中的度量干扰 

**Authors**: José Pombal, Nuno M. Guerreiro, Ricardo Rei, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2503.08327)  

**Abstract**: As automatic metrics become increasingly stronger and widely adopted, the risk of unintentionally "gaming the metric" during model development rises. This issue is caused by metric interference (Mint), i.e., the use of the same or related metrics for both model tuning and evaluation. Mint can misguide practitioners into being overoptimistic about the performance of their systems: as system outputs become a function of the interfering metric, their estimated quality loses correlation with human judgments. In this work, we analyze two common cases of Mint in machine translation-related tasks: filtering of training data, and decoding with quality signals. Importantly, we find that Mint strongly distorts instance-level metric scores, even when metrics are not directly optimized for -- questioning the common strategy of leveraging a different, yet related metric for evaluation that is not used for tuning. To address this problem, we propose MintAdjust, a method for more reliable evaluation under Mint. On the WMT24 MT shared task test set, MintAdjust ranks translations and systems more accurately than state-of-the-art-metrics across a majority of language pairs, especially for high-quality systems. Furthermore, MintAdjust outperforms AutoRank, the ensembling method used by the organizers. 

**Abstract (ZH)**: 随着自动评估指标越来越强大并且被广泛采用，模型开发过程中无意中“操控指标”的风险也在上升。这一问题源于指标干扰（Metric Interference, Mint），即在同一或相关任务中使用相同的评估指标进行模型调整和评估。指标干扰可能误导从业者对系统性能过于乐观：当系统输出成为干扰指标的函数时，其估计的质量与人类判断之间的相关性会丧失。在本研究中，我们分析了机器翻译相关任务中常见的两种指标干扰实例：训练数据过滤和使用质量信号进行解码。重要的是，我们发现即使在未直接优化这些指标的情况下，指标干扰也强烈地扭曲了实例级别的指标评分，这质疑了使用不用于调优的另一相关指标进行评估的常用策略。为解决这一问题，我们提出了一种名为MintAdjust的方法，以在指标干扰下提供更可靠的评估。在WMT24机器翻译共享任务测试集上，对于大多数语言对，MintAdjust在多数情况下比当前最优指标更准确地排名翻译和系统，特别是在高质量系统方面。此外，MintAdjust还优于组织者使用的AutoRank集成方法。 

---
# Prototype-based Heterogeneous Federated Learning for Blade Icing Detection in Wind Turbines with Class Imbalanced Data 

**Title (ZH)**: 基于原型的异构联邦学习在风电叶片结冰检测中的应用，处理类别不平衡数据 

**Authors**: Lele Qi, Mengna Liu, Xu Cheng, Fan Shi, Xiufeng Liu, Shengyong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08325)  

**Abstract**: Wind farms, typically in high-latitude regions, face a high risk of blade icing. Traditional centralized training methods raise serious privacy concerns. To enhance data privacy in detecting wind turbine blade icing, traditional federated learning (FL) is employed. However, data heterogeneity, resulting from collections across wind farms in varying environmental conditions, impacts the model's optimization capabilities. Moreover, imbalances in wind turbine data lead to models that tend to favor recognizing majority classes, thus neglecting critical icing anomalies. To tackle these challenges, we propose a federated prototype learning model for class-imbalanced data in heterogeneous environments to detect wind turbine blade icing. We also propose a contrastive supervised loss function to address the class imbalance problem. Experiments on real data from 20 turbines across two wind farms show our method outperforms five FL models and five class imbalance methods, with an average improvement of 19.64\% in \( mF_{\beta} \) and 5.73\% in \( m \)BA compared to the second-best method, BiFL. 

**Abstract (ZH)**: 基于异构环境的联邦原型学习模型：解决风电叶片冰冻检测中的类别不平衡问题 

---
# D3PO: Preference-Based Alignment of Discrete Diffusion Models 

**Title (ZH)**: D3PO：基于偏好离散扩散模型的对齐 

**Authors**: Umberto Borso, Davide Paglieri, Jude Wells, Tim Rocktäschel  

**Link**: [PDF](https://arxiv.org/pdf/2503.08295)  

**Abstract**: Diffusion models have achieved state-of-the-art performance across multiple domains, with recent advancements extending their applicability to discrete data. However, aligning discrete diffusion models with task-specific preferences remains challenging, particularly in scenarios where explicit reward functions are unavailable. In this work, we introduce Discrete Diffusion DPO (D3PO), the first adaptation of Direct Preference Optimization (DPO) to discrete diffusion models formulated as continuous-time Markov chains. Our approach derives a novel loss function that directly fine-tunes the generative process using preference data while preserving fidelity to a reference distribution. We validate D3PO on a structured binary sequence generation task, demonstrating that the method effectively aligns model outputs with preferences while maintaining structural validity. Our results highlight that D3PO enables controlled fine-tuning without requiring explicit reward models, making it a practical alternative to reinforcement learning-based approaches. Future research will explore extending D3PO to more complex generative tasks, including language modeling and protein sequence generation, as well as investigating alternative noise schedules, such as uniform noising, to enhance flexibility across different applications. 

**Abstract (ZH)**: 离散扩散模型D3PO：直接偏好优化在离散扩散模型中的应用 

---
# Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks 

**Title (ZH)**: Adv-CPG：一种基于面部 adversarial 攻击的定制化portrait生成框架 

**Authors**: Junying Wang, Hongyuan Zhang, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.08269)  

**Abstract**: Recent Customized Portrait Generation (CPG) methods, taking a facial image and a textual prompt as inputs, have attracted substantial attention. Although these methods generate high-fidelity portraits, they fail to prevent the generated portraits from being tracked and misused by malicious face recognition systems. To address this, this paper proposes a Customized Portrait Generation framework with facial Adversarial attacks (Adv-CPG). Specifically, to achieve facial privacy protection, we devise a lightweight local ID encryptor and an encryption enhancer. They implement progressive double-layer encryption protection by directly injecting the target identity and adding additional identity guidance, respectively. Furthermore, to accomplish fine-grained and personalized portrait generation, we develop a multi-modal image customizer capable of generating controlled fine-grained facial features. To the best of our knowledge, Adv-CPG is the first study that introduces facial adversarial attacks into CPG. Extensive experiments demonstrate the superiority of Adv-CPG, e.g., the average attack success rate of the proposed Adv-CPG is 28.1% and 2.86% higher compared to the SOTA noise-based attack methods and unconstrained attack methods, respectively. 

**Abstract (ZH)**: Recent定制化Portrait生成（CPG）方法 

---
# MT-NAM: An Efficient and Adaptive Model for Epileptic Seizure Detection 

**Title (ZH)**: MT-NAM: 一种高效自适应的癫痫发作检测模型 

**Authors**: Arshia Afzal, Volkan Cevher, Mahsa Shoaran  

**Link**: [PDF](https://arxiv.org/pdf/2503.08251)  

**Abstract**: Enhancing the accuracy and efficiency of machine learning algorithms employed in neural interface systems is crucial for advancing next-generation intelligent therapeutic devices. However, current systems often utilize basic machine learning models that do not fully exploit the natural structure of brain signals. Additionally, existing learning models used for neural signal processing often demonstrate low speed and efficiency during inference. To address these challenges, this study introduces Micro Tree-based NAM (MT-NAM), a distilled model based on the recently proposed Neural Additive Models (NAM). The MT-NAM achieves a remarkable 100$\times$ improvement in inference speed compared to standard NAM, without compromising accuracy. We evaluate our approach on the CHB-MIT scalp EEG dataset, which includes recordings from 24 patients with varying numbers of sessions and seizures. NAM achieves an 85.3\% window-based sensitivity and 95\% specificity. Interestingly, our proposed MT-NAM shows only a 2\% reduction in sensitivity compared to the original NAM. To regain this sensitivity, we utilize a test-time template adjuster (T3A) as an update mechanism, enabling our model to achieve higher sensitivity during test time by accommodating transient shifts in neural signals. With this online update approach, MT-NAM achieves the same sensitivity as the standard NAM while achieving approximately 50$\times$ acceleration in inference speed. 

**Abstract (ZH)**: 增强神经接口系统中采用的机器学习算法的准确性和效率对于推进下一代智能治疗设备至关重要。然而，当前系统往往使用基本的机器学习模型，未能充分利用脑信号的自然结构。此外，现有的用于神经信号处理的机器学习模型在推理过程中通常速度和效率较低。为解决这些挑战，本研究引入了基于最近提出的神经加法模型（NAM）的微树基于NAM（MT-NAM）的精简模型。MT-NAM在不牺牲准确性的前提下，相比标准NAM实现了高达100倍的推理速度提升。我们使用CHB-MIT头皮EEG数据集评估了我们的方法，该数据集包括来自24名患者的不同次数的记录和癫痫发作。NAM实现了85.3%的窗口敏感度和95%的特异度。有趣的是，我们提出的MT-NAM在敏感度上只比原始NAM降低了2%。为了恢复这一敏感度，我们利用测试时间模板调整器（T3A）作为更新机制，在神经信号的短暂变化时调整模型，以提高测试时的敏感度。采用这种在线更新方法，MT-NAM在实现约50倍的推理速度加速的同时，达到了与标准NAM相同的敏感度。 

---
# Investigating Execution-Aware Language Models for Code Optimization 

**Title (ZH)**: 执行感知的编程语言模型用于代码优化 

**Authors**: Federico Di Menna, Luca Traini, Gabriele Bavota, Vittorio Cortellessa  

**Link**: [PDF](https://arxiv.org/pdf/2503.08228)  

**Abstract**: Code optimization is the process of enhancing code efficiency, while preserving its intended functionality. This process often requires a deep understanding of the code execution behavior at run-time to identify and address inefficiencies effectively. Recent studies have shown that language models can play a significant role in automating code optimization. However, these models may have insufficient knowledge of how code execute at run-time. To address this limitation, researchers have developed strategies that integrate code execution information into language models. These strategies have shown promise, enhancing the effectiveness of language models in various software engineering tasks. However, despite the close relationship between code execution behavior and efficiency, the specific impact of these strategies on code optimization remains largely unexplored. This study investigates how incorporating code execution information into language models affects their ability to optimize code. Specifically, we apply three different training strategies to incorporate four code execution aspects -- line executions, line coverage, branch coverage, and variable states -- into CodeT5+, a well-known language model for code. Our results indicate that execution-aware models provide limited benefits compared to the standard CodeT5+ model in optimizing code. 

**Abstract (ZH)**: 代码执行信息嵌入对语言模型代码优化能力的影响研究 

---
# A Grey-box Text Attack Framework using Explainable AI 

**Title (ZH)**: 基于解释性人工智能的灰盒文本攻击框架 

**Authors**: Esther Chiramal, Kelvin Soh Boon Kai  

**Link**: [PDF](https://arxiv.org/pdf/2503.08226)  

**Abstract**: Explainable AI is a strong strategy implemented to understand complex black-box model predictions in a human interpretable language. It provides the evidence required to execute the use of trustworthy and reliable AI systems. On the other hand, however, it also opens the door to locating possible vulnerabilities in an AI model. Traditional adversarial text attack uses word substitution, data augmentation techniques and gradient-based attacks on powerful pre-trained Bidirectional Encoder Representations from Transformers (BERT) variants to generate adversarial sentences. These attacks are generally whitebox in nature and not practical as they can be easily detected by humans E.g. Changing the word from "Poor" to "Rich". We proposed a simple yet effective Grey-box cum Black-box approach that does not require the knowledge of the model while using a set of surrogate Transformer/BERT models to perform the attack using Explainable AI techniques. As Transformers are the current state-of-the-art models for almost all Natural Language Processing (NLP) tasks, an attack generated from BERT1 is transferable to BERT2. This transferability is made possible due to the attention mechanism in the transformer that allows the model to capture long-range dependencies in a sequence. Using the power of BERT generalisation via attention, we attempt to exploit how transformers learn by attacking a few surrogate transformer variants which are all based on a different architecture. We demonstrate that this approach is highly effective to generate semantically good sentences by changing as little as one word that is not detectable by humans while still fooling other BERT models. 

**Abstract (ZH)**: 可解释AI是一种強大策略，用于通过人类可理解的语言理解复杂黑盒模型的预测。它提供了使用可信赖和可靠AI系统的必要证据。另一方面，它也可能揭示AI模型中存在的潜在漏洞。传统文本对抗攻击通过使用词替换、数据增强技术和基于梯度的攻击来生成对抗句子，针对强大的预训练双向编码器表示变换器（BERT）变体。这些攻击通常是白盒性质的，不实用，因为它们很容易被人类检测到，例如将“Poor”变为“Rich”。我们提出了一种简单而有效的灰盒兼黑盒方法，这种方法在使用一组代理变换器/BERT模型进行攻击时不需要了解模型的内部结构，并利用可解释AI技术来执行攻击。由于变换器是几乎所有自然语言处理（NLP）任务的当前最先进的模型，一种基于BERT1生成的攻击可以转移到BERT2。这一可迁移性得益于变换器中的注意力机制，该机制使模型能够捕捉序列中的长距离依赖关系。利用BERT的一般化能力及其注意力机制，我们尝试通过攻击基于不同架构的几种代理变换器变体来利用变换器的学习过程。我们证明，这种方法仅通过改变一个不可被人检测到的词即可生成语义良好的句子，但仍能欺骗其他BERT模型。 

---
# EgoBlind: Towards Egocentric Visual Assistance for the Blind People 

**Title (ZH)**: 自视角盲助视系统：面向盲人的第一人称视觉辅助 

**Authors**: Junbin Xiao, Nanxin Huang, Hao Qiu, Zhulin Tao, Xun Yang, Richang Hong, Meng Wang, Angela Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.08221)  

**Abstract**: We present EgoBlind, the first egocentric VideoQA dataset collected from blind individuals to evaluate the assistive capabilities of contemporary multimodal large language models (MLLMs). EgoBlind comprises 1,210 videos that record the daily lives of real blind users from a first-person perspective. It also features 4,927 questions directly posed or generated and verified by blind individuals to reflect their needs for visual assistance under various scenarios. We provide each question with an average of 3 reference answers to alleviate subjective evaluation. Using EgoBlind, we comprehensively evaluate 15 leading MLLMs and find that all models struggle, with the best performers achieving accuracy around 56\%, far behind human performance of 87.4\%. To guide future advancements, we identify and summarize major limitations of existing MLLMs in egocentric visual assistance for the blind and provide heuristic suggestions for improvement. With these efforts, we hope EgoBlind can serve as a valuable foundation for developing more effective AI assistants to enhance the independence of the blind individuals' lives. 

**Abstract (ZH)**: EgoBlind: 一个来自盲人用户的第一人称视频问答数据集，用于评估当下多模态大语言模型的辅助能力 

---
# XAI4Extremes: An interpretable machine learning framework for understanding extreme-weather precursors under climate change 

**Title (ZH)**: XAI4Extremes: 一种在气候变化下理解极端天气前兆的可解释机器学习框架 

**Authors**: Jiawen Wei, Aniruddha Bora, Vivek Oommen, Chenyu Dong, Juntao Yang, Jeff Adie, Chen Chen, Simon See, George Karniadakis, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2503.08163)  

**Abstract**: Extreme weather events are increasing in frequency and intensity due to climate change. This, in turn, is exacting a significant toll in communities worldwide. While prediction skills are increasing with advances in numerical weather prediction and artificial intelligence tools, extreme weather still present challenges. More specifically, identifying the precursors of such extreme weather events and how these precursors may evolve under climate change remain unclear. In this paper, we propose to use post-hoc interpretability methods to construct relevance weather maps that show the key extreme-weather precursors identified by deep learning models. We then compare this machine view with existing domain knowledge to understand whether deep learning models identified patterns in data that may enrich our understanding of extreme-weather precursors. We finally bin these relevant maps into different multi-year time periods to understand the role that climate change is having on these precursors. The experiments are carried out on Indochina heatwaves, but the methodology can be readily extended to other extreme weather events worldwide. 

**Abstract (ZH)**: 由于气候变迁，极端气候事件的频率和强度在不断增加，这在全球范围内给社区带来了显著的冲击。尽管随着数值天气预报和人工智能工具的进步，预测技巧在不断提高，但极端气候事件仍然存在挑战。更为具体地说，识别此类极端气候事件的前兆以及这些前兆在气候变化下的演变依然是不清楚的。本文提出使用后验可解释性方法构建由深度学习模型识别的关键极端气候事件前兆的相关天气地图。然后我们将这种机器视角与现有的领域知识进行比对，以了解深度学习模型是否在数据中识别出了可能丰富我们对极端气候事件前兆理解的模式。最后，我们将这些相关地图按不同的多年时间周期进行分类，以理解气候变化在这些前兆中的作用。实验在印度支那热浪上进行，但该方法可以方便地扩展到世界各地的其他极端气候事件。 

---
# FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems 

**Title (ZH)**: 流驱动后验采样方法：解决逆问题 

**Authors**: Jeongsol Kim, Bryan Sangwoo Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.08136)  

**Abstract**: Flow matching is a recent state-of-the-art framework for generative modeling based on ordinary differential equations (ODEs). While closely related to diffusion models, it provides a more general perspective on generative modeling. Although inverse problem solving has been extensively explored using diffusion models, it has not been rigorously examined within the broader context of flow models. Therefore, here we extend the diffusion inverse solvers (DIS) - which perform posterior sampling by combining a denoising diffusion prior with an likelihood gradient - into the flow framework. Specifically, by driving the flow-version of Tweedie's formula, we decompose the flow ODE into two components: one for clean image estimation and the other for noise estimation. By integrating the likelihood gradient and stochastic noise into each component, respectively, we demonstrate that posterior sampling for inverse problem solving can be effectively achieved using flows. Our proposed solver, Flow-Driven Posterior Sampling (FlowDPS), can also be seamlessly integrated into a latent flow model with a transformer architecture. Across four linear inverse problems, we confirm that FlowDPS outperforms state-of-the-art alternatives, all without requiring additional training. 

**Abstract (ZH)**: 基于普通微分方程的流匹配是一种_recent_state-of-the-art_生成建模框架。虽然它与扩散模型密切相关，但提供了生成建模更为普适的观点。尽管扩散模型在逆问题求解方面得到了广泛探索，但在流模型更广泛的背景下，其逆问题求解能力尚未得到严格检验。因此，我们在此将扩散逆求解器(DIS)扩展到流框架中，该求解器通过结合去噪扩散先验和似然梯度进行后验采样。通过推动流版本的泰迪法则，我们将流ODE分解为两部分：一部分用于估计清晰图像，另一部分用于估计噪声。通过在每一部分分别整合似然梯度和随机噪声，展示了逆问题求解的后验采样可以有效利用流进行。我们提出的解算器Flow-Driven Posterior Sampling (FlowDPS)还可以无缝集成到具有transformer架构的潜在流模型中。在四种线性逆问题中，我们确认FlowDPS优于现有最佳方案，且无需额外训练。 

---
# Convergence Dynamics and Stabilization Strategies of Co-Evolving Generative Models 

**Title (ZH)**: 共演化的生成模型的收敛动力学与稳定化策略 

**Authors**: Weiguo Gao, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.08117)  

**Abstract**: The increasing prevalence of synthetic data in training loops has raised concerns about model collapse, where generative models degrade when trained on their own outputs. While prior work focuses on this self-consuming process, we study an underexplored yet prevalent phenomenon: co-evolving generative models that shape each other's training through iterative feedback. This is common in multimodal AI ecosystems, such as social media platforms, where text models generate captions that guide image models, and the resulting images influence the future adaptation of the text model. We take a first step by analyzing such a system, modeling the text model as a multinomial distribution and the image model as a conditional multi-dimensional Gaussian distribution. Our analysis uncovers three key results. First, when one model remains fixed, the other collapses: a frozen image model causes the text model to lose diversity, while a frozen text model leads to an exponential contraction of image diversity, though fidelity remains bounded. Second, in fully interactive systems, mutual reinforcement accelerates collapse, with image contraction amplifying text homogenization and vice versa, leading to a Matthew effect where dominant texts sustain higher image diversity while rarer texts collapse faster. Third, we analyze stabilization strategies implicitly introduced by real-world external influences. Random corpus injections for text models and user-content injections for image models prevent collapse while preserving both diversity and fidelity. Our theoretical findings are further validated through experiments. 

**Abstract (ZH)**: 合成数据在训练环中的增多引发了对模型崩溃的担忧，即生成模型在使用自身输出进行训练时性能下降。尽管先前的工作主要关注这一自我消费的过程，我们研究了一个尚未充分探索但普遍存在的现象：通过迭代反馈相互演变的生成模型，它们通过训练相互塑造。这一现象在多模态AI生态系统中尤为常见，例如社交媒体平台，其中文本模型生成描述图像的字幕，这些字幕反过来影响未来文本模型的适应性。我们首次分析了该系统，将文本模型建模为多项式分布，图像模型建模为条件多维正态分布。我们的分析揭示了三个关键结果。首先，当一个模型固定不变时，另一个模型会出现崩溃：冻结的图像模型会导致文本模型失去多样性，而冻结的文本模型会导致图像多样性的指数级收缩，尽管保真度保持在一定范围内。其次，在完全相互作用的系统中，相互强化加速了崩溃，图像多样性收缩放大了文本同质化，反之亦然，导致马太效应，即主导文本保持更高的图像多样性而少见文本则崩溃得更快。第三，我们分析了由真实世界外部影响隐式引入的稳定化策略。对于文本模型，随机文本语料库的注入；对于图像模型，用户内容的注入，可以防止崩溃同时保持多样性和保真度。我们的理论发现通过实验得到了进一步验证。 

---
# Revolution of Wireless Signal Recognition for 6G: Recent Advances, Challenges and Future Directions 

**Title (ZH)**: 6G无线信号识别革命：近期进展、挑战与未来方向 

**Authors**: Hao Zhang, Fuhui Zhou, Hongyang Du, Qihui Wu, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08091)  

**Abstract**: Wireless signal recognition (WSR) is a crucial technique for intelligent communications and spectrum sharing in the next six-generation (6G) wireless communication networks. It can be utilized to enhance network performance and efficiency, improve quality of service (QoS), and improve network security and reliability. Additionally, WSR can be applied for military applications such as signal interception, signal race, and signal abduction. In the past decades, great efforts have been made for the research of WSR. Earlier works mainly focus on model-based methods, including likelihood-based (LB) and feature-based (FB) methods, which have taken the leading position for many years. With the emergence of artificial intelligence (AI), intelligent methods including machine learning-based (ML-based) and deep learning-based (DL-based) methods have been developed to extract the features of the received signals and perform the classification. In this work, we provide a comprehensive review of WSR from the view of applications, main tasks, recent advances, datasets and evaluation metrics, challenges, and future directions. Specifically, intelligent WSR methods are introduced from the perspective of model, data, learning and implementation. Moreover, we analyze the challenges for WSR from the view of complex, dynamic, and open 6G wireless environments and discuss the future directions for WSR. This survey is expected to provide a comprehensive overview of the state-of-the-art WSR techniques and inspire new research directions for WSR in 6G networks. 

**Abstract (ZH)**: 无线信号识别（WSR）在六代（6G）无线通信网络的智能通信和频谱共享中的关键技术及其应用 

---
# Degradation Self-Supervised Learning for Lithium-ion Battery Health Diagnostics 

**Title (ZH)**: 锂离子电池健康诊断的降级自监督学习 

**Authors**: J. C. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.08083)  

**Abstract**: Health evaluation for lithium-ion batteries (LIBs) typically relies on constant charging/discharging protocols, often neglecting scenarios involving dynamic current profiles prevalent in electric vehicles. Conventional health indicators for LIBs also depend on the uniformity of measured data, restricting their adaptability to non-uniform conditions. In this study, a novel training strategy for estimating LIB health based on the paradigm of self-supervised learning is proposed. A multiresolution analysis technique, empirical wavelet transform, is utilized to decompose non-stationary voltage signals in the frequency domain. This allows the removal of ineffective components for the health evaluation model. The transformer neural network serves as the model backbone, and a loss function is designed to describe the capacity degradation behavior with the assumption that the degradation in LIBs across most operating conditions is inevitable and irreversible. The results show that the model can learn the aging characteristics by analyzing sequences of voltage and current profiles obtained at various time intervals from the same LIB cell. The proposed method is successfully applied to the Stanford University LIB aging dataset, derived from electric vehicle real driving profiles. Notably, this approach achieves an average correlation coefficient of 0.9 between the evaluated health index and the degradation of actual capacity, demonstrating its efficacy in capturing LIB health degradation. This research highlights the feasibility of training deep neural networks using unlabeled LIB data, offering cost-efficient means and unleashing the potential of the measured information. 

**Abstract (ZH)**: 基于自我监督学习 paradigm 估计锂离子电池 (LIBs) 健康状态的新培训策略：多分辨率分析在非平稳电压信号中的应用及其在变压器神经网络中的实现 

---
# Generalized Kullback-Leibler Divergence Loss 

**Title (ZH)**: 广义KL散度假损失 

**Authors**: Jiequan Cui, Beier Zhu, Qingshan Xu, Zhuotao Tian, Xiaojuan Qi, Bei Yu, Hanwang Zhang, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.08038)  

**Abstract**: In this paper, we delve deeper into the Kullback-Leibler (KL) Divergence loss and mathematically prove that it is equivalent to the Decoupled Kullback-Leibler (DKL) Divergence loss that consists of (1) a weighted Mean Square Error (wMSE) loss and (2) a Cross-Entropy loss incorporating soft labels. Thanks to the decoupled structure of DKL loss, we have identified two areas for improvement. Firstly, we address the limitation of KL loss in scenarios like knowledge distillation by breaking its asymmetric optimization property along with a smoother weight function. This modification effectively alleviates convergence challenges in optimization, particularly for classes with high predicted scores in soft labels. Secondly, we introduce class-wise global information into KL/DKL to reduce bias arising from individual samples. With these two enhancements, we derive the Generalized Kullback-Leibler (GKL) Divergence loss and evaluate its effectiveness by conducting experiments on CIFAR-10/100, ImageNet, and vision-language datasets, focusing on adversarial training, and knowledge distillation tasks. Specifically, we achieve new state-of-the-art adversarial robustness on the public leaderboard -- RobustBench and competitive knowledge distillation performance across CIFAR/ImageNet models and CLIP models, demonstrating the substantial practical merits. Our code is available at this https URL. 

**Abstract (ZH)**: 本文深入探讨了Kullback-Leibler (KL) 散度损失，并通过数学证明将其等同于由加权均方误差（wMSE）损失和结合软标签的交叉熵损失组成的解耦Kullback-Leibler (DKL) 散度损失。得益于DKL损失的解耦结构，我们识别了两个改进领域。首先，我们通过打破其不对称优化特性并引入更平滑的权重函数来解决KL损失在知识蒸馏等场景中的局限性。这一修改有效地缓解了优化过程中的收敛挑战，特别是在软标签中高预测分值类别的优化问题。其次，我们将类别级别的全局信息引入KL/DKL中，以减少由单个样本引起的偏差。通过这两个增强，我们推导出广义Kullback-Leibler (GKL) 散度损失，并通过在CIFAR-10/100、ImageNet和视觉语言数据集上的实验，以及在对抗训练和知识蒸馏任务中的评估来验证其有效性。具体而言，我们在公共排行榜RobustBench上实现了新的对抗鲁棒性状态，并在CIFAR/ImageNet模型和CLIP模型中的知识蒸馏性能方面取得了竞争力的表现，表明了其实用价值的显著性。我们的代码可在以下链接获取。 

---
# Exploring Bias in over 100 Text-to-Image Generative Models 

**Title (ZH)**: 探索超过100个文本到图像生成模型中的偏差 

**Authors**: Jordan Vice, Naveed Akhtar, Richard Hartley, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2503.08012)  

**Abstract**: We investigate bias trends in text-to-image generative models over time, focusing on the increasing availability of models through open platforms like Hugging Face. While these platforms democratize AI, they also facilitate the spread of inherently biased models, often shaped by task-specific fine-tuning. Ensuring ethical and transparent AI deployment requires robust evaluation frameworks and quantifiable bias metrics. To this end, we assess bias across three key dimensions: (i) distribution bias, (ii) generative hallucination, and (iii) generative miss-rate. Analyzing over 100 models, we reveal how bias patterns evolve over time and across generative tasks. Our findings indicate that artistic and style-transferred models exhibit significant bias, whereas foundation models, benefiting from broader training distributions, are becoming progressively less biased. By identifying these systemic trends, we contribute a large-scale evaluation corpus to inform bias research and mitigation strategies, fostering more responsible AI development.
Keywords: Bias, Ethical AI, Text-to-Image, Generative Models, Open-Source Models 

**Abstract (ZH)**: 我们探讨了文本到图像生成模型随时间的偏差趋势，重点关注通过Hugging Face等开放平台增加的模型可用性。虽然这些平台民主化了AI，但也促进了固有偏差模型的传播，这些模型往往是由特定任务的微调所形成的。确保伦理和透明的AI部署需要 robust 评价框架和可量化的偏差指标。为此，我们从三个关键维度评估偏差：(i) 分布偏差，(ii) 生成幻觉，(iii) 生成失率。通过对超过100个模型的分析，我们揭示了偏差模式随时间和生成任务的变化情况。我们的研究结果表明，艺术性和风格迁移模型具有显著的偏差，而得益于更广泛训练分布的基础模型，其偏差正在逐渐减少。通过识别这些系统性趋势，我们贡献了一个大规模的评价语料库，以指导偏差研究和缓解策略，促进更加负责任的AI开发。

关键词：偏差，伦理AI，文本到图像，生成模型，开源模型。 

---
# Injecting Imbalance Sensitivity for Multi-Task Learning 

**Title (ZH)**: 在多任务学习中注入不平衡敏感性 

**Authors**: Zhipeng Zhou, Liu Liu, Peilin Zhao, Wei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.08006)  

**Abstract**: Multi-task learning (MTL) has emerged as a promising approach for deploying deep learning models in real-life applications. Recent studies have proposed optimization-based learning paradigms to establish task-shared representations in MTL. However, our paper empirically argues that these studies, specifically gradient-based ones, primarily emphasize the conflict issue while neglecting the potentially more significant impact of imbalance/dominance in MTL. In line with this perspective, we enhance the existing baseline method by injecting imbalance-sensitivity through the imposition of constraints on the projected norms. To demonstrate the effectiveness of our proposed IMbalance-sensitive Gradient (IMGrad) descent method, we evaluate it on multiple mainstream MTL benchmarks, encompassing supervised learning tasks as well as reinforcement learning. The experimental results consistently demonstrate competitive performance. 

**Abstract (ZH)**: 多任务学习中基于优化的具有不平衡敏感性的梯度下降方法 

---
# A Neural Symbolic Model for Space Physics 

**Title (ZH)**: 空间物理中的神经符号模型 

**Authors**: Jie Ying, Haowei Lin, Chao Yue, Yajie Chen, Chao Xiao, Quanqi Shi, Yitao Liang, Shing-Tung Yau, Yuan Zhou, Jianzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.07994)  

**Abstract**: In this study, we unveil a new AI model, termed PhyE2E, to discover physical formulas through symbolic regression. PhyE2E simplifies symbolic regression by decomposing it into sub-problems using the second-order derivatives of an oracle neural network, and employs a transformer model to translate data into symbolic formulas in an end-to-end manner. The resulting formulas are refined through Monte-Carlo Tree Search and Genetic Programming. We leverage a large language model to synthesize extensive symbolic expressions resembling real physics, and train the model to recover these formulas directly from data. A comprehensive evaluation reveals that PhyE2E outperforms existing state-of-the-art approaches, delivering superior symbolic accuracy, precision in data fitting, and consistency in physical units. We deployed PhyE2E to five applications in space physics, including the prediction of sunspot numbers, solar rotational angular velocity, emission line contribution functions, near-Earth plasma pressure, and lunar-tide plasma signals. The physical formulas generated by AI demonstrate a high degree of accuracy in fitting the experimental data from satellites and astronomical telescopes. We have successfully upgraded the formula proposed by NASA in 1993 regarding solar activity, and for the first time, provided the explanations for the long cycle of solar activity in an explicit form. We also found that the decay of near-Earth plasma pressure is proportional to r^2 to Earth, where subsequent mathematical derivations are consistent with satellite data from another independent study. Moreover, we found physical formulas that can describe the relationships between emission lines in the extreme ultraviolet spectrum of the Sun, temperatures, electron densities, and magnetic fields. The formula obtained is consistent with the properties that physicists had previously hypothesized it should possess. 

**Abstract (ZH)**: 本研究提出了一个新的AI模型PhyE2E，用于通过符号回归发现物理公式。PhyE2E通过使用占优神经网络的二阶导数将符号回归分解为子问题，并采用变压器模型以端到端的方式将数据转化为符号公式。生成的公式通过蒙特卡洛树搜索和遗传编程进行优化。我们利用大规模语言模型合成大量类似真实物理的符号表达式，并训练模型直接从数据中恢复这些公式。全面评估表明，PhyE2E的表现优于现有最先进的方法，具备更高的符号准确性、数据拟合精度和物理单位一致性。我们将在空间物理学中的五个应用中部署PhyE2E，包括太阳黑子数预测、太阳自转角速度、发射线贡献函数、地球附近等离子体压力和月潮等离子信号。AI生成的物理公式在拟合卫星和天文望远镜的实验数据方面具有很高的准确性。我们成功升级了NASA于1993年关于太阳活动的公式，并首次以明确形式提供了太阳活动长周期的解释。我们还发现，地球附近的等离子体压力衰减与地球距离的平方成正比，后续的数学推导与另一个独立研究的卫星数据一致。此外，我们发现了描述极端紫外太阳谱发射线之间的关系、温度、电子密度和磁场的物理公式，所获得的公式与物理学家之前推测的性质一致。 

---
# Efficient and Accurate Estimation of Lipschitz Constants for Hybrid Quantum-Classical Decision Models 

**Title (ZH)**: 混合量子-经典决策模型的高效准确Lipschitz常数估计 

**Authors**: Sajjad Hashemian, Mohammad Saeed Arvenaghi  

**Link**: [PDF](https://arxiv.org/pdf/2503.07992)  

**Abstract**: In this paper, we propose a novel framework for efficiently and accurately estimating Lipschitz constants in hybrid quantum-classical decision models. Our approach integrates classical neural network with quantum variational circuits to address critical issues in learning theory such as fairness verification, robust training, and generalization.
By a unified convex optimization formulation, we extend existing classical methods to capture the interplay between classical and quantum layers. This integrated strategy not only provide a tight bound on the Lipschitz constant but also improves computational efficiency with respect to the previous methods. 

**Abstract (ZH)**: 本文提出了一种新型框架，用于高效准确地估计混合量子-经典决策模型中的Lipschitz常数。该方法将经典神经网络与量子变分电路集成，以解决学习理论中公平性验证、鲁棒训练与泛化等关键问题。通过统一的凸优化形式，我们将现有的经典方法扩展以捕获经典层与量子层之间的交互作用。这种集成策略不仅提供了Lipschitz常数的紧致界，还相对于之前的方法提高了计算效率。 

---
# A Theory of Learning with Autoregressive Chain of Thought 

**Title (ZH)**: 自动回归链式思考的学习理论 

**Authors**: Nirmit Joshi, Gal Vardi, Adam Block, Surbhi Goel, Zhiyuan Li, Theodor Misiakiewicz, Nathan Srebro  

**Link**: [PDF](https://arxiv.org/pdf/2503.07932)  

**Abstract**: For a given base class of sequence-to-next-token generators, we consider learning prompt-to-answer mappings obtained by iterating a fixed, time-invariant generator for multiple steps, thus generating a chain-of-thought, and then taking the final token as the answer. We formalize the learning problems both when the chain-of-thought is observed and when training only on prompt-answer pairs, with the chain-of-thought latent. We analyze the sample and computational complexity both in terms of general properties of the base class (e.g. its VC dimension) and for specific base classes such as linear thresholds. We present a simple base class that allows for universal representability and computationally tractable chain-of-thought learning. Central to our development is that time invariance allows for sample complexity that is independent of the length of the chain-of-thought. Attention arises naturally in our construction. 

**Abstract (ZH)**: 对于给定的序列到下一个token生成基类，我们考虑通过迭代一个固定的时间不变生成器多次来学习生成链式思考模式，并最终通过取最后一个生成的token作为答案的提示到答案映射。我们形式化了在链式思考可见和仅通过提示-答案对进行训练且链式思考作为潜在变量的情况下的学习问题。我们从基类的一般属性（如它的VC维度）和特定基类（如线性阈值）的角度分析了样本复杂性和计算复杂性。我们提出一个简单的基类，允许通用表示和可计算的链式思考学习。我们的发展核心在于时间不变性使得链式思考长度与样本复杂性无关。注意力在我们的构造中自然出现。 

---
# Measuring directional bias amplification in image captions using predictability 

**Title (ZH)**: 使用可预测性衡量图像字幕中的方向偏见放大 

**Authors**: Rahul Nair, Bhanu Tokas, Hannah Kerner  

**Link**: [PDF](https://arxiv.org/pdf/2503.07878)  

**Abstract**: When we train models on biased ML datasets, they not only learn these biases but can inflate them at test time - a phenomenon called bias amplification. To measure bias amplification in ML datasets, many co-occurrence-based metrics have been proposed. Co-occurrence-based metrics are effective in measuring bias amplification in simple problems like image classification. However, these metrics are ineffective for complex problems like image captioning as they cannot capture the semantics of a caption. To measure bias amplification in captions, prior work introduced a predictability-based metric called Leakage in Captioning (LIC). While LIC captures the semantics and context of captions, it has limitations. LIC cannot identify the direction in which bias is amplified, poorly estimates dataset bias due to a weak vocabulary substitution strategy, and is highly sensitive to attacker models (a hyperparameter in predictability-based metrics). To overcome these issues, we propose Directional Predictability Amplification in Captioning (DPAC). DPAC measures directional bias amplification in captions, provides a better estimate of dataset bias using an improved substitution strategy, and is less sensitive to attacker models. Our experiments on the COCO captioning dataset show how DPAC is the most reliable metric to measure bias amplification in captions. 

**Abstract (ZH)**: 当我们在带有偏见的ML数据集上训练模型时，它们不仅会学习这些偏见，还会在测试时放大这些偏见——这一现象称为偏见放大。为了衡量ML数据集中的偏见放大，已经提出了许多共现基于的指标。共现基于的指标在简单的图像分类等问题中有效，用于衡量偏见放大。然而，对于复杂的图像描述等问题，这些指标无法捕获描述的语义。为了衡量描述中的偏见放大，以往的工作引入了一种可预测性基于的指标，称为描述中的泄漏（LIC）。虽然LIC能够捕获描述的语义和背景，但它存在局限性。LIC无法识别偏见放大的方向，由于弱词汇替换策略导致其对数据集偏见的估计较差，并且非常敏感于攻击者模型（预测性指标中的一个超参数）。为了克服这些问题，我们提出了定向可预测性放大在描述中的方法（DPAC）。DPAC衡量描述中的定向偏见放大，通过改进的替换策略提供对数据集偏见的更好估计，并且对攻击者模型的敏感度更低。我们在COCO描述数据集上的实验表明，DPAC是最可靠的衡量描述中偏见放大的指标。 

---
# Topology-Preserving Loss for Accurate and Anatomically Consistent Cardiac Mesh Reconstruction 

**Title (ZH)**: 拓扑保持损失用于准确且解剖一致的心脏网格重建 

**Authors**: Chenyu Zhang, Yihao Luo, Yinzhe Wu, Choon Hwai Yap, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07874)  

**Abstract**: Accurate cardiac mesh reconstruction from volumetric data is essential for personalized cardiac modeling and clinical analysis. However, existing deformation-based approaches are prone to topological inconsistencies, particularly membrane penetration, which undermines the anatomical plausibility of the reconstructed mesh. To address this issue, we introduce Topology-Preserving Mesh Loss (TPM Loss), a novel loss function that explicitly enforces topological constraints during mesh deformation. By identifying topology-violating points, TPM Loss ensures spatially consistent reconstructions. Extensive experiments on CT and MRI datasets show that TPM Loss reduces topology violations by up to 93.1% while maintaining high segmentation accuracy (DSC: 89.1%-92.9%) and improving mesh fidelity (Chamfer Distance reduction up to 0.26 mm). These results demonstrate that TPM Loss effectively prevents membrane penetration and significantly improves cardiac mesh quality, enabling more accurate and anatomically consistent cardiac reconstructions. 

**Abstract (ZH)**: 从体数据中进行准确的心肌网格重建对于个性化心脏建模和临床分析至关重要。然而，现有的基于变形的方法容易出现拓扑不连续性，特别是膜穿透，这损害了重建网格的解剖合理性。为解决这一问题，我们引入了拓扑保持网格损失（Topology-Preserving Mesh Loss，TPM Loss），这是一种新颖的损失函数，在网格变形过程中显式地施加拓扑约束。通过识别拓扑违例点，TPM Loss 确保了空间上的一致性重建。在CT和MRI数据集上的 extensive 实验表明，TPM Loss 可将拓扑违例降低高达 93.1%，同时保持高分割准确率（DSC: 89.1%-92.9%），并提高网格保真度（切线距离减少至 0.26 mm）。这些结果表明，TPM Loss 有效地防止了膜穿透，显著提高了心脏网格质量，从而实现更准确和解剖上一致的心脏重建。 

---
# MapQA: Open-domain Geospatial Question Answering on Map Data 

**Title (ZH)**: MapQA：基于地图数据的开放域地理空间问答 

**Authors**: Zekun Li, Malcolm Grossman, Eric, Qasemi, Mihir Kulkarni, Muhao Chen, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07871)  

**Abstract**: Geospatial question answering (QA) is a fundamental task in navigation and point of interest (POI) searches. While existing geospatial QA datasets exist, they are limited in both scale and diversity, often relying solely on textual descriptions of geo-entities without considering their geometries. A major challenge in scaling geospatial QA datasets for reasoning lies in the complexity of geospatial relationships, which require integrating spatial structures, topological dependencies, and multi-hop reasoning capabilities that most text-based QA datasets lack. To address these limitations, we introduce MapQA, a novel dataset that not only provides question-answer pairs but also includes the geometries of geo-entities referenced in the questions. MapQA is constructed using SQL query templates to extract question-answer pairs from OpenStreetMap (OSM) for two study regions: Southern California and Illinois. It consists of 3,154 QA pairs spanning nine question types that require geospatial reasoning, such as neighborhood inference and geo-entity type identification. Compared to existing datasets, MapQA expands both the number and diversity of geospatial question types. We explore two approaches to tackle this challenge: (1) a retrieval-based language model that ranks candidate geo-entities by embedding similarity, and (2) a large language model (LLM) that generates SQL queries from natural language questions and geo-entity attributes, which are then executed against an OSM database. Our findings indicate that retrieval-based methods effectively capture concepts like closeness and direction but struggle with questions that require explicit computations (e.g., distance calculations). LLMs (e.g., GPT and Gemini) excel at generating SQL queries for one-hop reasoning but face challenges with multi-hop reasoning, highlighting a key bottleneck in advancing geospatial QA systems. 

**Abstract (ZH)**: 地理空间问答（QA）是导航和兴趣点（POI）搜索中的基本任务。现有地理空间QA数据集存在规模和多样性有限的问题，通常仅依赖于地理实体的文本描述，而忽视了它们的几何形状。在扩展地理空间QA数据集以进行推理时，面临的主要挑战在于地理空间关系的复杂性，这需要结合空间结构、拓扑依赖关系和多跳推理能力，而大多数基于文本的QA数据集缺乏这些功能。为了解决这些限制，我们引入了MapQA这一新型数据集，除了提供问题-答案对之外，还包含了问题中引用的地理实体的几何形状。MapQA利用SQL查询模板从OpenStreetMap（OSM）中提取问题-答案对，涉及两个研究区域：加利福尼亚南部和伊利诺伊州。它包含了3,154个问题-答案对，涉及九种需要地理空间推理的问题类型，如邻里推理和地理实体类型识别。与现有数据集相比，MapQA在地理空间问题的数量和多样性上都有扩展。我们探索了两种应对这一挑战的方法：（1）基于检索的语言模型，通过嵌入相似性对候选地理实体进行排名，以及（2）大型语言模型（LLM），生成从自然语言问题和地理实体属性到SQL查询的转换，然后将这些查询执行在OSM数据库上。我们的研究结果表明，基于检索的方法有效地捕捉了接近性和方向等概念，但对于需要显式计算的问题（如距离计算）则表现不佳。大语言模型（如GPT和Gemini）擅长生成用于单跳推理的SQL查询，但在多跳推理方面面临挑战，这突显了推进地理空间QA系统的一个关键瓶颈。 

---
# Right Reward Right Time for Federated Learning 

**Title (ZH)**: Right Reward at Right Time for Federated Learning 

**Authors**: Thanh Linh Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2503.07869)  

**Abstract**: Critical learning periods (CLPs) in federated learning (FL) refer to early stages during which low-quality contributions (e.g., sparse training data availability) can permanently impair the learning performance of the global model owned by the model owner (i.e., the cloud server). However, strategies to motivate clients with high-quality contributions to join the FL training process and share trained model updates during CLPs remain underexplored. Additionally, existing incentive mechanisms in FL treat all training periods equally, which consequently fails to motivate clients to participate early. Compounding this challenge is the cloud's limited knowledge of client training capabilities due to privacy regulations, leading to information asymmetry. Therefore, in this article, we propose a time-aware incentive mechanism, called Right Reward Right Time (R3T), to encourage client involvement, especially during CLPs, to maximize the utility of the cloud in FL. Specifically, the cloud utility function captures the trade-off between the achieved model performance and payments allocated for clients' contributions, while accounting for clients' time and system capabilities, efforts, joining time, and rewards. Then, we analytically derive the optimal contract for the cloud and devise a CLP-aware mechanism to incentivize early participation and efforts while maximizing cloud utility, even under information asymmetry. By providing the right reward at the right time, our approach can attract the highest-quality contributions during CLPs. Simulation and proof-of-concept studies show that R3T increases cloud utility and is more economically effective than benchmarks. Notably, our proof-of-concept results show up to a 47.6% reduction in the total number of clients and up to a 300% improvement in convergence time while reaching competitive test accuracies compared with incentive mechanism benchmarks. 

**Abstract (ZH)**: 时间敏感的激励机制：Critical Learning Periods (CLPs) 在联邦学习中的 Right Reward Right Time (R3T) 激励策略 

---
# Group Fairness in Multi-Task Reinforcement Learning 

**Title (ZH)**: 多任务强化学习中的团体公平性 

**Authors**: Kefan Song, Runnan Jiang, Rohan Chandra, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07817)  

**Abstract**: This paper addresses a critical societal consideration in the application of Reinforcement Learning (RL): ensuring equitable outcomes across different demographic groups in multi-task settings. While previous work has explored fairness in single-task RL, many real-world applications are multi-task in nature and require policies to maintain fairness across all tasks. We introduce a novel formulation of multi-task group fairness in RL and propose a constrained optimization algorithm that explicitly enforces fairness constraints across multiple tasks simultaneously. We have shown that our proposed algorithm does not violate fairness constraints with high probability and with sublinear regret in the finite-horizon episodic setting. Through experiments in RiverSwim and MuJoCo environments, we demonstrate that our approach better ensures group fairness across multiple tasks compared to previous methods that lack explicit multi-task fairness constraints in both the finite-horizon setting and the infinite-horizon setting. Our results show that the proposed algorithm achieves smaller fairness gaps while maintaining comparable returns across different demographic groups and tasks, suggesting its potential for addressing fairness concerns in real-world multi-task RL applications. 

**Abstract (ZH)**: 本文探讨了强化学习（RL）应用中的一个关键社会考虑因素：在多任务设置中确保不同人口群体的公平结果。尽管以往的工作已探索了单任务RL中的公平性问题，但许多实际应用都是多任务性质，要求策略在所有任务中均维持公平性。我们提出了强化学习中多任务群体公平性的新形式化定义，并提出了一种约束优化算法，该算法同时显式地在多个任务中施加公平性约束。我们证明了所提出的算法以高概率不会违反公平性约束，并在有限时间段的周期性设置中具有亚线性后悔。通过在RiverSwim和MuJoCo环境中进行实验，我们表明，与以往方法相比，我们的方法在有限时间段和无限时间段设置中都能更好地确保多任务场景下不同群体之间的公平性，其中以往方法缺乏明确的多任务公平性约束。我们的结果表明，所提出的算法在不同人口群体和任务中实现了更小的公平性差距，同时保持了相当的回报，这表明其在实际多任务RL应用中解决公平性问题的潜力。 

---
# AgriField3D: A Curated 3D Point Cloud and Procedural Model Dataset of Field-Grown Maize from a Diversity Panel 

**Title (ZH)**: AgriField3D: 田间种植玉米多样面板的精选3D点云和过程建模数据集 

**Authors**: Elvis Kimara, Mozhgan Hadadi, Jackson Godbersen, Aditya Balu, Talukder Jubery, Yawei Li, Adarsh Krishnamurthy, Patrick S. Schnable, Baskar Ganapathysubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2503.07813)  

**Abstract**: The application of artificial intelligence (AI) in three-dimensional (3D) agricultural research, particularly for maize, has been limited by the scarcity of large-scale, diverse datasets. While 2D image datasets are abundant, they fail to capture essential structural details such as leaf architecture, plant volume, and spatial arrangements that 3D data provide. To address this limitation, we present AgriField3D (this https URL), a curated dataset of 3D point clouds of field-grown maize plants from a diverse genetic panel, designed to be AI-ready for advancing agricultural research. Our dataset comprises over 1,000 high-quality point clouds collected using a Terrestrial Laser Scanner, complemented by procedural models that provide structured, parametric representations of maize plants. These procedural models, generated using Non-Uniform Rational B-Splines (NURBS) and optimized via a two-step process combining Particle Swarm Optimization (PSO) and differentiable programming, enable precise, scalable reconstructions of leaf surfaces and plant architectures. To enhance usability, we performed graph-based segmentation to isolate individual leaves and stalks, ensuring consistent labeling across all samples. We also conducted rigorous manual quality control on all datasets, correcting errors in segmentation, ensuring accurate leaf ordering, and validating metadata annotations. The dataset further includes metadata detailing plant morphology and quality, alongside multi-resolution subsampled versions (100k, 50k, 10k points) optimized for various computational needs. By integrating point cloud data of field grown plants with high-fidelity procedural models and ensuring meticulous manual validation, AgriField3D provides a comprehensive foundation for AI-driven phenotyping, plant structural analysis, and 3D applications in agricultural research. 

**Abstract (ZH)**: 农业领域三维（3D）玉米研究中的人工智能应用受限于大规模多样化数据集的稀缺性。虽然二维（2D）图像数据集丰富，但无法捕捉到三维（3D）数据提供的关键结构细节，如叶序结构、植物体积和空间布局。为解决这一限制，我们 presents AgriField3D（详见<a href="this https URL" target="_blank">此处</a>），这是一个由多样基因组系的田间种植玉米植物构建的三维点云数据集，旨在为推进农业研究做好人工智能准备。该数据集包含超过1000个高质量的点云，采用 terrestrial激光扫描仪采集，并补充了过程模型，提供了结构化的参数化玉米植物表示。这些过程模型使用非均匀有理B样条（NURBS）生成，并通过结合粒子群优化（PSO）和可微编程的两步优化过程进行优化，实现了叶面和植物结构的精确且可扩展的重建。为提升易用性，我们进行了图块基的分割以分离单个叶片和茎杆，确保所有样本的标签一致。我们还对所有数据集进行了严格的手动质量控制，纠正分割错误，确保叶片排序准确，并验证元数据注释。该数据集还包括详细的植物形态和质量元数据，以及用于各种计算需求的多分辨率子采样版本（100k、50k、10k点）。通过将田间种植植物的点云数据与高保真过程模型整合，并确保仔细的手动验证，AgriField3D 为基于人工智能的表型分析、植物结构分析和农业研究中的三维应用提供了全面的基础。 

---
# A primer on optimal transport for causal inference with observational data 

**Title (ZH)**: 最优 transport在观察性数据分析中的因果推理入门 

**Authors**: Florian F Gunsilius  

**Link**: [PDF](https://arxiv.org/pdf/2503.07811)  

**Abstract**: The theory of optimal transportation has developed into a powerful and elegant framework for comparing probability distributions, with wide-ranging applications in all areas of science. The fundamental idea of analyzing probabilities by comparing their underlying state space naturally aligns with the core idea of causal inference, where understanding and quantifying counterfactual states is paramount. Despite this intuitive connection, explicit research at the intersection of optimal transport and causal inference is only beginning to develop. Yet, many foundational models in causal inference have implicitly relied on optimal transport principles for decades, without recognizing the underlying connection. Therefore, the goal of this review is to offer an introduction to the surprisingly deep existing connections between optimal transport and the identification of causal effects with observational data -- where optimal transport is not just a set of potential tools, but actually builds the foundation of model assumptions. As a result, this review is intended to unify the language and notation between different areas of statistics, mathematics, and econometrics, by pointing out these existing connections, and to explore novel problems and directions for future work in both areas derived from this realization. 

**Abstract (ZH)**: 最优传输理论已成为一种强大而优雅的概率分布比较框架，在科学的所有领域都有广泛的应用。尽管最优传输与因果推断之间的直观联系显而易见，但二者交集的研究尚处于起步阶段。事实上，许多因果推断中的基础模型已经隐含地依赖于最优传输原理数十年之久，而未意识到这种潜在联系。因此，本文综述旨在介绍最优传输与基于观察数据识别因果效应之间出人意料的深入联系——在此过程中，最优传输不仅是潜在的工具，更是模型假设的基础。本文综述旨在通过指出这些存在的联系，统一统计学、数学和计量经济学领域的语言和符号，并探索这些认识在两个领域产生的新颖问题和未来研究方向。 

---
# Self-supervised Normality Learning and Divergence Vector-guided Model Merging for Zero-shot Congenital Heart Disease Detection in Fetal Ultrasound Videos 

**Title (ZH)**: 基于自监督正常性学习和发散向量引导模型融合的胎儿超声视频先天性心脏病零样本检测 

**Authors**: Pramit Saha, Divyanshu Mishra, Netzahualcoyotl Hernandez-Cruz, Olga Patey, Aris Papageorghiou, Yuki M. Asano, J. Alison Noble  

**Link**: [PDF](https://arxiv.org/pdf/2503.07799)  

**Abstract**: Congenital Heart Disease (CHD) is one of the leading causes of fetal mortality, yet the scarcity of labeled CHD data and strict privacy regulations surrounding fetal ultrasound (US) imaging present significant challenges for the development of deep learning-based models for CHD detection. Centralised collection of large real-world datasets for rare conditions, such as CHD, from large populations requires significant co-ordination and resource. In addition, data governance rules increasingly prevent data sharing between sites. To address these challenges, we introduce, for the first time, a novel privacy-preserving, zero-shot CHD detection framework that formulates CHD detection as a normality modeling problem integrated with model merging. In our framework dubbed Sparse Tube Ultrasound Distillation (STUD), each hospital site first trains a sparse video tube-based self-supervised video anomaly detection (VAD) model on normal fetal heart US clips with self-distillation loss. This enables site-specific models to independently learn the distribution of healthy cases. To aggregate knowledge across the decentralized models while maintaining privacy, we propose a Divergence Vector-Guided Model Merging approach, DivMerge, that combines site-specific models into a single VAD model without data exchange. Our approach preserves domain-agnostic rich spatio-temporal representations, ensuring generalization to unseen CHD cases. We evaluated our approach on real-world fetal US data collected from 5 hospital sites. Our merged model outperformed site-specific models by 23.77% and 30.13% in accuracy and F1-score respectively on external test sets. 

**Abstract (ZH)**: 先天性心脏病检测的一种新颖隐私保护零样本框架：Sparse Tube Ultrasound Distillation（STUD） 

---
# Joint Explainability-Performance Optimization With Surrogate Models for AI-Driven Edge Services 

**Title (ZH)**: 基于代理模型的AI驱动边缘服务联合解释性-性能优化 

**Authors**: Foivos Charalampakos, Thomas Tsouparopoulos, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.07784)  

**Abstract**: Explainable AI is a crucial component for edge services, as it ensures reliable decision making based on complex AI models. Surrogate models are a prominent approach of XAI where human-interpretable models, such as a linear regression model, are trained to approximate a complex (black-box) model's predictions. This paper delves into the balance between the predictive accuracy of complex AI models and their approximation by surrogate ones, advocating that both these models benefit from being learned simultaneously. We derive a joint (bi-level) training scheme for both models and we introduce a new algorithm based on multi-objective optimization (MOO) to simultaneously minimize both the complex model's prediction error and the error between its outputs and those of the surrogate. Our approach leads to improvements that exceed 99% in the approximation of the black-box model through the surrogate one, as measured by the metric of Fidelity, for a compromise of less than 3% absolute reduction in the black-box model's predictive accuracy, compared to single-task and multi-task learning baselines. By improving Fidelity, we can derive more trustworthy explanations of the complex model's outcomes from the surrogate, enabling reliable AI applications for intelligent services at the network edge. 

**Abstract (ZH)**: 可解释AI是边缘服务的关键组件，因为它确保基于复杂AI模型的可靠决策。代理模型是可解释AI的一个主要方法，其中人类可解释的模型，如线性回归模型，被训练以近似复杂（黑盒）模型的预测。本文探讨了复杂AI模型的预测准确性和其由代理模型近似的平衡，提倡同时学习这两种模型。我们推导了一种联合（多层次）训练方案，并引入了一种基于多目标优化的新算法，同时最小化复杂模型的预测误差及其输出与代理模型输出之间的误差。通过Fidelity度量，我们的方法在代理模型中实现了接近100%的黑盒模型近似改善，同时相对减少不到3%的黑盒模型预测准确率，优于单任务和多任务学习基准。通过提高Fidelity，可以从代理模型中获得更可信的复杂模型结果解释，从而为网络边缘的智能服务提供可靠的AI应用。 

---
# Automated Benchmark Generation for Repository-Level Coding Tasks 

**Title (ZH)**: 仓库级编码任务的自动化基准生成 

**Authors**: Konstantinos Vergopoulos, Mark Niklas Müller, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2503.07701)  

**Abstract**: Code Agent development is an extremely active research area, where a reliable performance metric is critical for tracking progress and guiding new developments. This demand is underscored by the meteoric rise in popularity of SWE-Bench. This benchmark challenges code agents to generate patches addressing GitHub issues given the full repository as context. The correctness of generated patches is then evaluated by executing a human-written test suite extracted from the repository after the issue's resolution. However, constructing benchmarks like SWE-Bench requires substantial manual effort to set up historically accurate execution environments for testing. Crucially, this severely limits the number of considered repositories, e.g., just 12 for SWE-Bench. Considering so few repositories, selected for their popularity runs the risk of leading to a distributional mismatch, i.e., the measured performance may not be representative of real-world scenarios potentially misguiding development efforts. In this work, we address this challenge and introduce SetUpAgent, a fully automated system capable of historically accurate dependency setup, test execution, and result parsing. Using SetUpAgent, we generate two new datasets: (i) SWEE-Bench an extended version of SWE-Bench encompassing hundreds of repositories, and (ii) SWA-Bench a benchmark focusing on applications rather than libraries. Comparing these datasets to SWE-Bench with respect to their characteristics and code agent performance, we find significant distributional differences, including lower issue description quality and detail level, higher fix complexity, and most importantly up to 40% lower agent success rates. 

**Abstract (ZH)**: Code Agent 开发是一个极其活跃的研究领域，可靠的性能度量对于追踪进展和指导新技术的发展至关重要。这一需求在 SWE-Bench 的迅猛流行中得到了强化。SWE-Bench 挑战代码代理生成解决问题的补丁，并通过执行提取自仓库的人工编写的测试套件来评估生成补丁的正确性。然而，构建类似 SWE-Bench 的基准需要大量的手动工作来设置历史准确的测试环境。这极大地限制了可以考虑的仓库数量，例如，SWE-Bench 只包括 12 个仓库。在如此有限的仓库中进行选择，可能会导致数据分布不匹配，即测量的性能可能不具有现实世界场景的代表性，从而误导开发努力。在此工作中，我们解决了这一挑战并引入了 SetUpAgent，这是一个能够自动完成历史准确的依赖设置、测试执行和结果解析的系统。使用 SetUpAgent，我们生成了两个新的数据集：(i) SWEE-Bench，它是 SWE-Bench 的扩展版本，涵盖了数百个仓库，(ii) SWA-Bench，一个专注于应用程序而非库的基准。比较这些数据集与 SWE-Bench 在特性和代码代理性能方面的差异，我们发现显著的数据分布差异，包括较低的问题描述质量和细节水平、更高的修复复杂度，最重要的是，代理成功率最多降低了 40%。 

---
# Artificial Intelligence in Deliberation: The AI Penalty and the Emergence of a New Deliberative Divide 

**Title (ZH)**: 人工智能在审议中的应用：AI罚分与审议分裂的新兴分歧 

**Authors**: Andreas Jungherr, Adrian Rauchfleisch  

**Link**: [PDF](https://arxiv.org/pdf/2503.07690)  

**Abstract**: Digital deliberation has expanded democratic participation, yet challenges remain. This includes processing information at scale, moderating discussions, fact-checking, or attracting people to participate. Recent advances in artificial intelligence (AI) offer potential solutions, but public perceptions of AI's role in deliberation remain underexplored. Beyond efficiency, democratic deliberation is about voice and recognition. If AI is integrated into deliberation, public trust, acceptance, and willingness to participate may be affected. We conducted a preregistered survey experiment with a representative sample in Germany (n=1850) to examine how information about AI-enabled deliberation influences willingness to participate and perceptions of deliberative quality. Respondents were randomly assigned to treatments that provided them information about deliberative tasks facilitated by either AI or humans. Our findings reveal a significant AI-penalty. Participants were less willing to engage in AI-facilitated deliberation and rated its quality lower than human-led formats. These effects were moderated by individual predispositions. Perceptions of AI's societal benefits and anthropomorphization of AI showed positive interaction effects on people's interest to participate in AI-enabled deliberative formats and positive quality assessments, while AI risk assessments showed negative interactions with information about AI-enabled deliberation. These results suggest AI-enabled deliberation faces substantial public skepticism, potentially even introducing a new deliberative divide. Unlike traditional participation gaps based on education or demographics, this divide is shaped by attitudes toward AI. As democratic engagement increasingly moves online, ensuring AI's role in deliberation does not discourage participation or deepen inequalities will be a key challenge for future research and policy. 

**Abstract (ZH)**: 数字辩论扩展了民主参与，但仍面临挑战。这包括大规模处理信息、 moderating 讨论、事实核查或吸引人们参与。近年来人工智能（AI）的进步提供了潜在的解决方案，但公众对 AI 在辩论中的角色看法仍需进一步探索。超出效率，数字辩论关乎声音和认同。若 AI 融入辩论，公众信任、接受度和参与意愿可能受到影响。我们在德国进行了一项预先注册的调查实验（n=1850），以考察关于 AI 助力的辩论信息如何影响参与意愿和对辩论质量的看法。受访者被随机分配到使用 AI 或人类助力的辩论任务信息治疗组。我们的研究发现显示了显着的 AI 折扣。参与者更不愿意参与由 AI 助力的辩论，并且认为其质量低于由人类主导的格式。这些效应受到个人倾向的调节。AI 社会利益的认知和将 AI 人性化对个人参与 AI 助力的辩论格式的兴趣和正面质量评估显示出正向交互效应，而对 AI 风险的评估与关于 AI 助力的辩论信息显示出负向交互效应。这些结果表明，AI 助力的辩论面临着显著的公众怀疑，甚至可能引入新的辩论鸿沟。不同于基于教育或人口统计学的传统参与鸿沟，这种鸿沟由对 AI 的态度塑造。随着民主参与越来越多地转移到线上，确保 AI 在辩论中的作用不会阻碍参与或加深不平等将成为未来研究和政策的关键挑战。 

---
# Adaptive routing protocols for determining optimal paths in AI multi-agent systems: a priority- and learning-enhanced approach 

**Title (ZH)**: 基于优先级和学习增强的自适应路由协议：在AI多代理系统中确定最优路径的方法 

**Authors**: Theodor Panayotov, Ivo Emanuilov  

**Link**: [PDF](https://arxiv.org/pdf/2503.07686)  

**Abstract**: As distributed artificial intelligence (AI) and multi-agent architectures grow increasingly complex, the need for adaptive, context-aware routing becomes paramount. This paper introduces an enhanced, adaptive routing algorithm tailored for AI multi-agent networks, integrating priority-based cost functions and dynamic learning mechanisms. Building on an extended Dijkstra-based framework, we incorporate multi-faceted parameters such as task complexity, user request priority, agent capabilities, bandwidth, latency, load, model sophistication, and reliability. We further propose dynamically adaptive weighting factors, tuned via reinforcement learning (RL), to continuously evolve routing policies based on observed network performance. Additionally, heuristic filtering and hierarchical routing structures improve scalability and responsiveness. Our approach yields context-sensitive, load-aware, and priority-focused routing decisions that not only reduce latency for critical tasks but also optimize overall resource utilization, ultimately enhancing the robustness, flexibility, and efficiency of multi-agent systems. 

**Abstract (ZH)**: 随着分布式人工智能（AI）和多智能体架构日益复杂，适应性和上下文感知路由的需求变得至关重要。本文介绍了一种专为AI多智能体网络设计的增强型自适应路由算法，集成基于优先级的成本函数和动态学习机制。在扩展的Dijkstra框架基础上，我们整合了任务复杂度、用户请求优先级、智能体能力、带宽、延迟、负载、模型复杂度和可靠性等多方面参数。此外，我们提出了一种通过强化学习（RL）动态调整的权重因子，以根据观测到的网络性能连续进化路由策略。同时，采用启发式过滤和分层路由结构提高可扩展性和响应性。该方法提供了上下文敏感、负载感知和优先级导向的路由决策，不仅减少了关键任务的延迟，还优化了整体资源利用，最终增强了多智能体系统的 robustness、灵活性和效率。 

---
# Ways of Seeing, and Selling, AI Art 

**Title (ZH)**: 观看方式，与销售方式：AI艺术 

**Authors**: Imke van Heerden  

**Link**: [PDF](https://arxiv.org/pdf/2503.07685)  

**Abstract**: In early 2025, Augmented Intelligence - Christie's first AI art auction - drew criticism for showcasing a controversial genre. Amid wider legal uncertainty, artists voiced concerns over data mining practices, notably with respect to copyright. The backlash could be viewed as a microcosm of AI's contested position in the creative economy. Touching on the auction's presentation, reception, and results, this paper explores how, among social dissonance, machine learning finds its place in the artworld. Foregrounding responsible innovation, the paper provides a balanced perspective that champions creators' rights and brings nuance to this polarised debate. With a focus on exhibition design, it centres framing, which refers to the way a piece is presented to influence consumer perception. Context plays a central role in shaping our understanding of how good, valuable, and even ethical an artwork is. In this regard, Augmented Intelligence situates AI art within a surprisingly traditional framework, leveraging hallmarks of "high art" to establish the genre's cultural credibility. Generative AI has a clear economic dimension, converging questions of artistic merit with those of monetary worth. Scholarship on ways of seeing, or framing, could substantively inform the interpretation and evaluation of creative outputs, including assessments of their aesthetic and commercial value. 

**Abstract (ZH)**: 增强智能艺术——克里斯蒂首次AI艺术拍卖在争议中登场 

---
# A Time Series Multitask Framework Integrating a Large Language Model, Pre-Trained Time Series Model, and Knowledge Graph 

**Title (ZH)**: 一个集成大规模语言模型、预训练时间序列模型和知识图谱的时间序列多任务框架 

**Authors**: Shule Hao, Junpeng Bao, Chuncheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07682)  

**Abstract**: Time series analysis is crucial in fields like finance, transportation, and industry. However, traditional models often focus solely on temporal features, limiting their ability to capture underlying information. This paper proposes a novel time series multitask framework, called LTM, which integrates temporal features with textual descriptions to enhance analytical and predictive capabilities. LTM combines pre-trained time series model, large language model (LLM), and knowledge graph to tackle time series tasks, including forecasting, imputation, and anomaly detection. LTM achieves improved performance with a few trainable parameters. It is very efficient and practical. LTM encodes time series data into patches and enriches user-provided prompts using knowledge graphs to generate enhanced prompts. A novel feature fusion method embeds prompts into each patch encoding, which is processed by a frozen LLM, followed by a feature enhancement module and a time decoder module. During fine-tuning stage, cosine similarity between prompts and temporal patches is integrated into the loss function to boost performance. Experiments on benchmark datasets show that LTM significantly outperforms existing methods. It provides a robust and versatile solution for time series tasks. 

**Abstract (ZH)**: LTM：一种集成时间和文本的多任务时间序列框架 

---
# Using a single actor to output personalized policy for different intersections 

**Title (ZH)**: 使用单个actor输出适用于不同交叉口的个性化策略 

**Authors**: Kailing Zhou, Chengwei Zhang, Furui Zhan, Wanting Liu, Yihong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07678)  

**Abstract**: Recently, with the development of Multi-agent reinforcement learning (MARL), adaptive traffic signal control (ATSC) has achieved satisfactory results. In traffic scenarios with multiple intersections, MARL treats each intersection as an agent and optimizes traffic signal control strategies through learning and real-time decision-making. Considering that observation distributions of intersections might be different in real-world scenarios, shared parameter methods might lack diversity and thus lead to high generalization requirements in the shared-policy network. A typical solution is to increase the size of network parameters. However, simply increasing the scale of the network does not necessarily improve policy generalization, which is validated in our experiments. Accordingly, an approach that considers both the personalization of intersections and the efficiency of parameter sharing is required. To this end, we propose Hyper-Action Multi-Head Proximal Policy Optimization (HAMH-PPO), a Centralized Training with Decentralized Execution (CTDE) MARL method that utilizes a shared PPO policy network to deliver personalized policies for intersections with non-iid observation distributions. The centralized critic in HAMH-PPO uses graph attention units to calculate the graph representations of all intersections and outputs a set of value estimates with multiple output heads for each intersection. The decentralized execution actor takes the local observation history as input and output distributions of action as well as a so-called hyper-action to balance the multiple values estimated from the centralized critic to further guide the updating of TSC policies. The combination of hyper-action and multi-head values enables multiple agents to share a single actor-critic while achieving personalized policies. 

**Abstract (ZH)**: 最近，随着多代理 reinforcement 学习（MARL）的发展，自适应交通信号控制（ATSC）取得了满意的结果。在包含多个交叉口的交通场景中，MARL 将每个交叉口视为一个代理，并通过学习和实时决策优化交通信号控制策略。考虑到实际场景中交叉口的观测分布可能不同，共享参数方法可能会缺乏多样性，从而导致在共享策略网络中对泛化能力提出了高要求。一个典型的解决方案是增加网络参数的规模。然而，仅仅增加网络规模并不一定能提高策略的泛化能力，我们的实验中得到了验证。因此，同时考虑交叉口的个性化和参数共享效率的方法是必要的。为此，我们提出了超动作多头近端策略优化（HAMH-PPO），这是一种集中训练分散执行（CTDE）的 MARL 方法，利用共享的 PPO 策略网络为具有非 i.i.d 观测分布的交叉口提供个性化的策略。HAMH-PPO 的集中式评论家使用图注意单元来计算所有交叉口的图表示，并输出每个交叉口的一组价值估计值，带有多个输出头。分散执行的行动者将局部观察历史作为输入，并输出行动分布以及所谓的超动作，以平衡集中式评论家估计的多个价值，进一步指导 TSC 策略的更新。超动作和多头价值的结合使多个代理能够共享一个单一的行动者-评论家网络，同时实现个性化的策略。 

---
# The Janus Face of Innovation: Global Disparities and Divergent Options 

**Title (ZH)**: 创新的两面性：全球不平等与分歧选择 

**Authors**: Nihat Mugurtay  

**Link**: [PDF](https://arxiv.org/pdf/2503.07676)  

**Abstract**: This article examines how unequal access to AI innovation creates systemic challenges for developing countries. Differential access to AI innovation results from the acute competition between domestic and global actors. While developing nations contribute significantly to AI development through data annotation labor, they face limited access to advanced AI technologies and are increasingly caught between divergent regulatory approaches from democratic and authoritarian tendencies. This brief paper analyzes how more affordable AI engagement and Western countries' development cooperation present developing nations with a complex choice between accessibility and governance standards. I argue this challenge entails new institutional mechanisms for technology transfer and regulatory cooperation, while carefully balancing universal standards with local needs. In turn, good practices could help developing countries close the deepening gap of global technological divides, while ensuring responsible AI development in developing countries. 

**Abstract (ZH)**: 本文探讨了不平等的AI创新访问权如何为发展中国家创造系统性挑战。差异化的AI创新访问是由国内和全球行为体之间的尖锐竞争造成的。尽管发展中国家通过数据标注劳动力在AI发展中做出了重大贡献，但它们仍面临先进AI技术访问受限的问题，并且越来越被民主和威权倾向的监管方法所分歧。本文简要分析了更实惠的AI参与和西方国家的发展合作为发展中国家提供了在可访问性和治理标准之间复杂选择的背景，我认为这一挑战需要新的技术转移和监管合作制度机制，在普及标准与地方需求之间谨慎平衡。良好的实践可以有助于发展中国家缩小日益扩大的全球技术鸿沟，同时确保发展中国家负责任的AI发展。 

---
# TVNet: A Novel Time Series Analysis Method Based on Dynamic Convolution and 3D-Variation 

**Title (ZH)**: TVNet：一种基于动态卷积和3D变异的时间序列分析方法 

**Authors**: Chenghan Li, Mingchen Li, Ruisheng Diao  

**Link**: [PDF](https://arxiv.org/pdf/2503.07674)  

**Abstract**: With the recent development and advancement of Transformer and MLP architectures, significant strides have been made in time series analysis. Conversely, the performance of Convolutional Neural Networks (CNNs) in time series analysis has fallen short of expectations, diminishing their potential for future applications. Our research aims to enhance the representational capacity of Convolutional Neural Networks (CNNs) in time series analysis by introducing novel perspectives and design innovations. To be specific, We introduce a novel time series reshaping technique that considers the inter-patch, intra-patch, and cross-variable dimensions. Consequently, we propose TVNet, a dynamic convolutional network leveraging a 3D perspective to employ time series analysis. TVNet retains the computational efficiency of CNNs and achieves state-of-the-art results in five key time series analysis tasks, offering a superior balance of efficiency and performance over the state-of-the-art Transformer-based and MLP-based models. Additionally, our findings suggest that TVNet exhibits enhanced transferability and robustness. Therefore, it provides a new perspective for applying CNN in advanced time series analysis tasks. 

**Abstract (ZH)**: 基于新颖视角和设计创新提高Convolutional Neural Networks在时间序列分析中的表示能力：提出TVNet及其卓越性能和鲁棒性分析 

---
# The potential role of AI agents in transforming nuclear medicine research and cancer management in India 

**Title (ZH)**: AI代理在转变印度核医学研究和癌症管理中的潜在作用 

**Authors**: Rajat Vashistha, Arif Gulzar, Parveen Kundu, Punit Sharma, Mark Brunstein, Viktor Vegh  

**Link**: [PDF](https://arxiv.org/pdf/2503.07673)  

**Abstract**: India faces a significant cancer burden, with an incidence-to-mortality ratio indicating that nearly three out of five individuals diagnosed with cancer succumb to the disease. While the limitations of physical healthcare infrastructure are widely acknowledged as a primary challenge, concerted efforts by government and healthcare agencies are underway to mitigate these constraints. However, given the country's vast geography and high population density, it is imperative to explore alternative soft infrastructure solutions to complement existing frameworks. Artificial Intelligence agents are increasingly transforming problem-solving approaches across various domains, with their application in medicine proving particularly transformative. In this perspective, we examine the potential role of AI agents in advancing nuclear medicine for cancer research, diagnosis, and management in India. We begin with a brief overview of AI agents and their capabilities, followed by a proposed agent-based ecosystem that can address prevailing sustainability challenges in India nuclear medicine. 

**Abstract (ZH)**: 印度面临严重的癌症负担，癌癥发病率与死亡率比值表明，近三分之二被诊断出癌症的个体最终死于该病。尽管物理 healthcare 基础设施的局限性被广泛承认是主要挑战之一，政府和医疗健康机构正着手解决这些限制。然而，鉴于印度广阔的土地面积和高人口密度，迫切需要探索替代的软基础设施解决方案以补充现有框架。人工智能代理正在各个领域逐渐改变问题解决方式，其在医学中的应用尤其具有变革性。在此视角下，我们探讨人工智能代理在促进印度核医学在癌症研究、诊断和管理中的潜在作用。我们首先简要概述人工智能代理及其能力，然后提出一个基于代理的生态系统，旨在解决印度核医学中现有的可持续性挑战。 

---
# Probabilistic Shielding for Safe Reinforcement Learning 

**Title (ZH)**: 概率性屏蔽以实现安全强化学习 

**Authors**: Edwin Hamel-De le Court, Francesco Belardinelli, Alex W. Goodall  

**Link**: [PDF](https://arxiv.org/pdf/2503.07671)  

**Abstract**: In real-life scenarios, a Reinforcement Learning (RL) agent aiming to maximise their reward, must often also behave in a safe manner, including at training time. Thus, much attention in recent years has been given to Safe RL, where an agent aims to learn an optimal policy among all policies that satisfy a given safety constraint. However, strict safety guarantees are often provided through approaches based on linear programming, and thus have limited scaling. In this paper we present a new, scalable method, which enjoys strict formal guarantees for Safe RL, in the case where the safety dynamics of the Markov Decision Process (MDP) are known, and safety is defined as an undiscounted probabilistic avoidance property. Our approach is based on state-augmentation of the MDP, and on the design of a shield that restricts the actions available to the agent. We show that our approach provides a strict formal safety guarantee that the agent stays safe at training and test time. Furthermore, we demonstrate that our approach is viable in practice through experimental evaluation. 

**Abstract (ZH)**: 在现实场景中，一个旨在最大化奖励的强化学习（RL）代理，通常也需要在训练和测试过程中表现出安全的行为。因此，近年来对安全强化学习（Safe RL）的关注不断增加，其中代理的目标是在所有满足给定安全约束的策略中学习最优策略。然而，严格的安全保证通常通过基于线性规划的方法提供，这在扩展性方面有限。本文提出了一种新的可扩展方法，该方法在马尔可夫决策过程（MDP）的安全动力学已知且安全定义为无折现概率避免属性的情况下，能够为安全强化学习提供严格的正式保证。我们的方法基于MDP的状态拓展，并设计了一个屏蔽机制，限制代理可采取的动作。我们展示了该方法能够在训练和测试时严格保证代理的安全行为。此外，通过实验评估证明了该方法在实践中的可行性。 

---
# Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy 

**Title (ZH)**: 打破模型合并：无需牺牲精度的参数级防御 

**Authors**: Wei Junhao, Yu Zhe, Sakuma Jun  

**Link**: [PDF](https://arxiv.org/pdf/2503.07661)  

**Abstract**: Model merging is a technique that combines multiple finetuned models into a single model without additional training, allowing a free-rider to cheaply inherit specialized capabilities. This study investigates methodologies to suppress unwanted model merging by free-riders. Existing methods such as model watermarking or fingerprinting can only detect merging in hindsight. In contrast, we propose a first proactive defense against model merging. Specifically, our defense method modifies the model parameters so that the model is disrupted if the model is merged with any other model, while its functionality is kept unchanged if not merged with others. Our approach consists of two modules, rearranging MLP parameters and scaling attention heads, which push the model out of the shared basin in parameter space, causing the merging performance with other models to degrade significantly. We conduct extensive experiments on image classification, image generation, and text classification to demonstrate that our defense severely disrupts merging while retaining the functionality of the post-protect model. Moreover, we analyze potential adaptive attacks and further propose a dropout-based pruning to improve our proposal's robustness. 

**Abstract (ZH)**: 一种主动防御方法以抑制免费 Riding者进行模型合并：一种修改模型参数的方法使得模型在与其他模型合并时性能显著下降，而在未合并时功能保持不变。 

---
# Insights into Schizophrenia: Leveraging Machine Learning for Early Identification via EEG, ERP, and Demographic Attributes 

**Title (ZH)**: 基于EEG、ERP和人口统计学特征的机器学习在精神分裂症早期识别中的洞察 

**Authors**: Sara Alkhalifa  

**Link**: [PDF](https://arxiv.org/pdf/2503.07650)  

**Abstract**: The research presents a machine learning (ML) classifier designed to differentiate between schizophrenia patients and healthy controls by utilising features extracted from electroencephalogram (EEG) data, specifically focusing on event-related potentials (ERPs) and certain demographic variables. The dataset comprises data from 81 participants, encompassing 32 healthy controls and 49 schizophrenia patients, all sourced from an online dataset. After preprocessing the dataset, our ML model achieved an accuracy of 99.980%. This performance outperforms earlier research, including those that used deep learning methods. Additionally, an analysis was conducted to assess individual features' contribution to improving classification accuracy. This involved systematically excluding specific features from the original dataset one at a time, and another technique involved an iterative process of removing features based on their entropy scores incrementally. The impact of these removals on model performance was evaluated to identify the most informative features. 

**Abstract (ZH)**: 机器学习分类器用于通过 Electroencephalogram (EEG) 数据中的事件相关电位 (ERPs) 和某些人口统计学变量来区分精神分裂症患者与健康对照组 

---
# TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster 

**Title (ZH)**: TS-RAG: 基于检索增强生成的时间序列基础模型在零样本预测中更加强大 

**Authors**: Kanghui Ning, Zijie Pan, Yu Liu, Yushan Jiang, James Y. Zhang, Kashif Rasul, Anderson Schneider, Lintao Ma, Yuriy Nevmyvaka, Dongjin Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.07649)  

**Abstract**: Recently, Large Language Models (LLMs) and Foundation Models (FMs) have become prevalent for time series forecasting tasks. However, fine-tuning large language models (LLMs) for forecasting enables the adaptation to specific domains but may not generalize well across diverse, unseen datasets. Meanwhile, existing time series foundation models (TSFMs) lack inherent mechanisms for domain adaptation and suffer from limited interpretability, making them suboptimal for zero-shot forecasting. To this end, we present TS-RAG, a retrieval-augmented generation based time series forecasting framework that enhances the generalization capability and interpretability of TSFMs. Specifically, TS-RAG leverages pre-trained time series encoders to retrieve semantically relevant time series segments from a dedicated knowledge database, incorporating contextual patterns for the given time series query. Next, we develop a learnable Mixture-of-Experts (MoE)-based augmentation module, which dynamically fuses retrieved time series patterns with the TSFM's representation of the input query, improving forecasting accuracy without requiring task-specific fine-tuning. Thorough empirical studies on seven public benchmark datasets demonstrate that TS-RAG achieves state-of-the-art zero-shot forecasting performance, outperforming TSFMs by up to 6.51% across diverse domains and showcasing desired interpretability. 

**Abstract (ZH)**: TS-RAG：基于检索增强生成的时间序列 forecasting 框架 

---
# ConstellationNet: Reinventing Spatial Clustering through GNNs 

**Title (ZH)**: 星座网：通过GNN重构空间聚类 

**Authors**: Aidan Gao, Junhong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.07643)  

**Abstract**: Spatial clustering is a crucial field, finding universal use across criminology, pathology, and urban planning. However, most spatial clustering algorithms cannot pull information from nearby nodes and suffer performance drops when dealing with higher dimensionality and large datasets, making them suboptimal for large-scale and high-dimensional clustering. Due to modern data growing in size and dimension, clustering algorithms become weaker when addressing multifaceted issues. To improve upon this, we develop ConstellationNet, a convolution neural network(CNN)-graph neural network(GNN) framework that leverages the embedding power of a CNN, the neighbor aggregation of a GNN, and a neural network's ability to deal with batched data to improve spatial clustering and classification with graph augmented predictions. ConstellationNet achieves state-of-the-art performance on both supervised classification and unsupervised clustering across several datasets, outperforming state-of-the-art classification and clustering while reducing model size and training time by up to tenfold and improving baselines by 10 times. Because of its fast training and powerful nature, ConstellationNet holds promise in fields like epidemiology and medical imaging, able to quickly train on new data to develop robust responses. 

**Abstract (ZH)**: 空间聚类是一种关键领域，广泛应用于犯罪学、病理学和城市规划。然而，大多数空间聚类算法无法从附近节点提取信息，在处理高维度和大规模数据集时性能下降，使其在大规模和高维聚类中效果欠佳。由于现代数据的规模和维度日益增长，聚类算法在处理多方面问题时变得无力。为改善这一现状，我们开发了星座网络（ConstellationNet），这是一种卷积神经网络（CNN）-图神经网络（GNN）框架，利用CNN的嵌入能力、GNN的邻域聚合能力和神经网络处理批量数据的能力，以图增强预测提高空间聚类和分类性能。星座网络在多个数据集上的监督分类和无监督聚类上均达到最佳性能，优于最先进的分类和聚类模型，同时模型大小和训练时间最多可降低十倍，并将基准提高了十倍。由于其快速训练和强大的特性，星座网络在流行病学和医学影像等领域展现出乐观前景，能够快速适应新数据以开发稳健的响应。 

---
# Deep ARTMAP: Generalized Hierarchical Learning with Adaptive Resonance Theory 

**Title (ZH)**: 深度ARTMAP：自适应共振理论下的generalized分层学习 

**Authors**: Niklas M. Melton, Leonardo Enzo Brito da Silva, Sasha Petrenko, Donald. C. Wunsch II  

**Link**: [PDF](https://arxiv.org/pdf/2503.07641)  

**Abstract**: This paper presents Deep ARTMAP, a novel extension of the ARTMAP architecture that generalizes the self-consistent modular ART (SMART) architecture to enable hierarchical learning (supervised and unsupervised) across arbitrary transformations of data. The Deep ARTMAP framework operates as a divisive clustering mechanism, supporting an arbitrary number of modules with customizable granularity within each module. Inter-ART modules regulate the clustering at each layer, permitting unsupervised learning while enforcing a one-to-many mapping from clusters in one layer to the next. While Deep ARTMAP reduces to both ARTMAP and SMART in particular configurations, it offers significantly enhanced flexibility, accommodating a broader range of data transformations and learning modalities. 

**Abstract (ZH)**: Deep ARTMAP：一种将ARTMAP体系结构扩展到支持任意数据变换的分层监督与无监督学习的新型框架 

---
# BrainNet-MoE: Brain-Inspired Mixture-of-Experts Learning for Neurological Disease Identification 

**Title (ZH)**: 脑神经启发的混合专家学习：脑网-MoE方法在神经系统疾病识别中的应用 

**Authors**: Jing Zhang, Xiaowei Yu, Tong Chen, Chao Cao, Mingheng Chen, Yan Zhuang, Yanjun Lyu, Lu Zhang, Li Su, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07640)  

**Abstract**: The Lewy body dementia (LBD) is the second most common neurodegenerative dementia after Alzheimer's disease (AD). Early differentiation between AD and LBD is crucial because they require different treatment approaches, but this is challenging due to significant clinical overlap, heterogeneity, complex pathogenesis, and the rarity of LBD. While recent advances in artificial intelligence (AI) demonstrate powerful learning capabilities and offer new hope for accurate diagnosis, existing methods primarily focus on designing "neural-level networks". Our work represents a pioneering effort in modeling system-level artificial neural network called BrainNet-MoE for brain modeling and diagnosing. Inspired by the brain's hierarchical organization of bottom-up sensory integration and top-down control, we design a set of disease-specific expert groups to process brain sub-network under different condition, A disease gate mechanism guides the specializa-tion of expert groups, while a transformer layer enables communication be-tween all sub-networks, generating a comprehensive whole-brain represen-tation for downstream disease classification. Experimental results show superior classification accuracy with interpretable insights into how brain sub-networks contribute to different neurodegenerative conditions. 

**Abstract (ZH)**: Lewy 体痴呆（LBD）是仅次于阿尔茨海默病（AD）的第二大常见的神经退行性痴呆。早期准确区分AD和LBD至关重要，因为两者需要不同的治疗方法，但这一区分面临挑战，因它们在临床表现、异质性、复杂的病理机制以及LBD的罕见性上有显著重叠。尽管最近的人工智能（AI）技术展示了强大的学习能力，并为准确诊断带来了新的希望，但现有方法主要集中在设计“神经层级网络”方面。我们的一项工作代表了在构建用于脑建模和诊断的系统级人工神经网络方面的一项开创性努力，称为BrainNet-MoE。受大脑自底向上感觉整合和自顶向下控制的层次结构组织启发，我们设计了一系列疾病特异性的专家组，以在不同条件下处理脑子网络。疾病门控机制引导专家组的专业化，而变压器层使所有子网络之间能够通信，生成用于下游疾病分类的全面的大脑整体表示。实验结果显示，在可解释方面具有优越的分类准确性，并揭示了不同神经退行性疾病条件下脑子网络的贡献机制。 

---
# Leveraging Taxonomy Similarity for Next Activity Prediction in Patient Treatment 

**Title (ZH)**: 利用分类相似性进行患者治疗中下一个活动预测 

**Authors**: Martin Kuhn, Joscha Grüger, Tobias Geyer, Ralph Bergmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.07638)  

**Abstract**: The rapid progress in modern medicine presents physicians with complex challenges when planning patient treatment. Techniques from the field of Predictive Business Process Monitoring, like Next-activity-prediction (NAP) can be used as a promising technique to support physicians in treatment planning, by proposing a possible next treatment step. Existing patient data, often in the form of electronic health records, can be analyzed to recommend the next suitable step in the treatment process. However, the use of patient data poses many challenges due to its knowledge-intensive character, high variability and scarcity of medical data. To overcome these challenges, this article examines the use of the knowledge encoded in taxonomies to improve and explain the prediction of the next activity in the treatment process. This study proposes the TS4NAP approach, which uses medical taxonomies (ICD-10-CM and ICD-10-PCS) in combination with graph matching to assess the similarities of medical codes to predict the next treatment step. The effectiveness of the proposed approach will be evaluated using event logs that are derived from the MIMIC-IV dataset. The results highlight the potential of using domain-specific knowledge held in taxonomies to improve the prediction of the next activity, and thus can improve treatment planning and decision-making by making the predictions more explainable. 

**Abstract (ZH)**: 现代医学快速进展给医生在规划患者治疗时带来了复杂挑战。通过利用预测业务过程监控领域的技术，如下一步活动预测(NAP)，可以为医生在治疗规划中提供支持，通过建议可能的下一步治疗措施。现有的患者数据，通常以电子健康记录的形式存在，可以通过分析来推荐治疗过程中的下一个合适步骤。然而，由于其知识密集性、高变异性以及医学数据的稀缺性，使用患者数据存在许多挑战。为了克服这些挑战，本文研究了利用分类法中编码的知识来提高和解释治疗过程下一步活动预测的问题。本文提出了TS4NAP方法，该方法结合图匹配使用医学分类法（ICD-10-CM和ICD-10-PCS）来评估医学代码的相似性以预测下一步治疗措施。通过MIMIC-IV数据集派生的事件日志评估所提出方法的有效性。结果突显了利用分类法中持有的专业知识改善下一步活动预测的潜力，并且可以提高治疗规划和决策的透明度。 

---
# Addressing Selection Bias in Computerized Adaptive Testing: A User-Wise Aggregate Influence Function Approach 

**Title (ZH)**: 计算机化自适应测验中选择偏差的校正：基于用户汇总影响函数的方法 

**Authors**: Soonwoo Kwon, Sojung Kim, Seunghyun Lee, Jin-Young Kim, Suyeong An, Kyuseok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2308.11912)  

**Abstract**: Computerized Adaptive Testing (CAT) is a widely used, efficient test mode that adapts to the examinee's proficiency level in the test domain. CAT requires pre-trained item profiles, for CAT iteratively assesses the student real-time based on the registered items' profiles, and selects the next item to administer using candidate items' profiles. However, obtaining such item profiles is a costly process that involves gathering a large, dense item-response data, then training a diagnostic model on the collected data. In this paper, we explore the possibility of leveraging response data collected in the CAT service. We first show that this poses a unique challenge due to the inherent selection bias introduced by CAT, i.e., more proficient students will receive harder questions. Indeed, when naively training the diagnostic model using CAT response data, we observe that item profiles deviate significantly from the ground-truth. To tackle the selection bias issue, we propose the user-wise aggregate influence function method. Our intuition is to filter out users whose response data is heavily biased in an aggregate manner, as judged by how much perturbation the added data will introduce during parameter estimation. This way, we may enhance the performance of CAT while introducing minimal bias to the item profiles. We provide extensive experiments to demonstrate the superiority of our proposed method based on the three public datasets and one dataset that contains real-world CAT response data. 

**Abstract (ZH)**: 计算机化自适应测试中响应数据的应用探索 

---
# Principal deuterium Hugoniot via Quantum Monte Carlo and $Δ$-learning 

**Title (ZH)**: 使用量子蒙特卡洛和Δ学习的主氘化胡oneot 李维数研究 

**Authors**: Giacomo Tenti, Kousuke Nakano, Andrea Tirelli, Sandro Sorella, Michele Casula  

**Link**: [PDF](https://arxiv.org/pdf/2301.03570)  

**Abstract**: We present a study of the principal deuterium Hugoniot for pressures up to $150$ GPa, using Machine Learning potentials (MLPs) trained with Quantum Monte Carlo (QMC) energies, forces and pressures. In particular, we adopted a recently proposed workflow based on the combination of Gaussian kernel regression and $\Delta$-learning. By fully taking advantage of this method, we explicitly considered finite-temperature electrons in the dynamics, whose effects are highly relevant for temperatures above $10$ kK. The Hugoniot curve obtained by our MLPs shows a good agreement with the most recent experiments, particularly in the region below 60 GPa. At larger pressures, our Hugoniot curve is slightly more compressible than the one yielded by experiments, whose uncertainties generally increase, however, with pressure. Our work demonstrates that QMC can be successfully combined with $\Delta$-learning to deploy reliable MLPs for complex extended systems across different thermodynamic conditions, by keeping the QMC precision at the computational cost of a mean-field calculation. 

**Abstract (ZH)**: 我们使用经量子蒙特卡罗（QMC）能量、力和压力训练的机器学习势（MLPs），研究了直至150 GPa的主要氘化氢霍斯迪昂曲线。特别地，我们采用了最近提出的工作流，该工作流基于高斯核回归和Δ学习的结合。充分利用这种方法，我们在动力学中明确考虑了有限温度电子的影响，这些影响对于超过10 kK的温度尤其重要。我们MLPs获得的霍斯迪昂曲线在60 GPa以下区域与最新实验结果有很好的一致性。在较大压力下，我们获得的霍斯迪昂曲线略比实验结果更可压缩，而实验不确定性的增加程度通常随压力的增大而增大。我们的研究证明了QMC可以成功与Δ学习结合，部署适用于不同热力学条件下复杂扩展系统的可靠MLPs，同时保持QMC计算的精度成本在均场计算的范围内。 

---
