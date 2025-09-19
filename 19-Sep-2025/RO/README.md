# Parallel Simulation of Contact and Actuation for Soft Growing Robots 

**Title (ZH)**: 软生长机器人接触与动作的并行模拟 

**Authors**: Yitian Gao, Lucas Chen, Priyanka Bhovad, Sicheng Wang, Zachary Kingston, Laura H. Blumenschein  

**Link**: [PDF](https://arxiv.org/pdf/2509.15180)  

**Abstract**: Soft growing robots, commonly referred to as vine robots, have demonstrated remarkable ability to interact safely and robustly with unstructured and dynamic environments. It is therefore natural to exploit contact with the environment for planning and design optimization tasks. Previous research has focused on planning under contact for passively deforming robots with pre-formed bends. However, adding active steering to these soft growing robots is necessary for successful navigation in more complex environments. To this end, we develop a unified modeling framework that integrates vine robot growth, bending, actuation, and obstacle contact. We extend the beam moment model to include the effects of actuation on kinematics under growth and then use these models to develop a fast parallel simulation framework. We validate our model and simulator with real robot experiments. To showcase the capabilities of our framework, we apply our model in a design optimization task to find designs for vine robots navigating through cluttered environments, identifying designs that minimize the number of required actuators by exploiting environmental contacts. We show the robustness of the designs to environmental and manufacturing uncertainties. Finally, we fabricate an optimized design and successfully deploy it in an obstacle-rich environment. 

**Abstract (ZH)**: 软生长机器人，通常称为藤蔓机器人，已经展示了与未结构化和动态环境安全而 robust 地互动的能力。因此，利用与环境的接触来进行规划和设计优化是自然的选择。以往的研究集中于具有先构弯曲的被动变形机器人在接触下的规划。然而，要在更复杂的环境中成功导航，这些软生长机器人需要加入主动转向。为此，我们开发了一个统一的建模框架，整合了藤蔓机器人的生长、弯曲、驱动和障碍接触。我们扩展了梁矩模型，以包括在生长过程中驱动对运动学的影响，然后使用这些模型开发了一个快速并行仿真框架。我们通过实际的机器人实验验证了我们的模型和仿真器。为了展示该框架的能力，我们在一个设计优化任务中应用了我们的模型，以找到能够通过杂乱环境的藤蔓机器人设计，通过利用环境接触来最小化所需驱动器的数量。我们展示了这些设计对环境和制造不确定性具有鲁棒性。最后，我们制造了一个优化的设计，并成功地将其部署在障碍丰富的环境中。 

---
# AnoF-Diff: One-Step Diffusion-Based Anomaly Detection for Forceful Tool Use 

**Title (ZH)**: 基于一步扩散的异常检测方法：强力工具使用异常检测 

**Authors**: Yating Lin, Zixuan Huang, Fan Yang, Dmitry Berenson  

**Link**: [PDF](https://arxiv.org/pdf/2509.15153)  

**Abstract**: Multivariate time-series anomaly detection, which is critical for identifying unexpected events, has been explored in the field of machine learning for several decades. However, directly applying these methods to data from forceful tool use tasks is challenging because streaming sensor data in the real world tends to be inherently noisy, exhibits non-stationary behavior, and varies across different tasks and tools. To address these challenges, we propose a method, AnoF-Diff, based on the diffusion model to extract force-torque features from time-series data and use force-torque features to detect anomalies. We compare our method with other state-of-the-art methods in terms of F1-score and Area Under the Receiver Operating Characteristic curve (AUROC) on four forceful tool-use tasks, demonstrating that our method has better performance and is more robust to a noisy dataset. We also propose the method of parallel anomaly score evaluation based on one-step diffusion and demonstrate how our method can be used for online anomaly detection in several forceful tool use experiments. 

**Abstract (ZH)**: 基于扩散模型的多变量时间序列异常检测方法AnoF-Diff及其在强制工具使用任务中的应用 

---
# Energy-Constrained Navigation for Planetary Rovers under Hybrid RTG-Solar Power 

**Title (ZH)**: 行星车在混合放射性同位素温差发电机-太阳能电源系统约束下的能量受限导航 

**Authors**: Tianxin Hu, Weixiang Guo, Ruimeng Liu, Xinhang Xu, Rui Qian, Jinyu Chen, Shenghai Yuan, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.15062)  

**Abstract**: Future planetary exploration rovers must operate for extended durations on hybrid power inputs that combine steady radioisotope thermoelectric generator (RTG) output with variable solar photovoltaic (PV) availability. While energy-aware planning has been studied for aerial and underwater robots under battery limits, few works for ground rovers explicitly model power flow or enforce instantaneous power constraints. Classical terrain-aware planners emphasize slope or traversability, and trajectory optimization methods typically focus on geometric smoothness and dynamic feasibility, neglecting energy feasibility. We present an energy-constrained trajectory planning framework that explicitly integrates physics-based models of translational, rotational, and resistive power with baseline subsystem loads, under hybrid RTG-solar input. By incorporating both cumulative energy budgets and instantaneous power constraints into SE(2)-based polynomial trajectory optimization, the method ensures trajectories that are simultaneously smooth, dynamically feasible, and power-compliant. Simulation results on lunar-like terrain show that our planner generates trajectories with peak power within 0.55 percent of the prescribed limit, while existing methods exceed limits by over 17 percent. This demonstrates a principled and practical approach to energy-aware autonomy for long-duration planetary missions. 

**Abstract (ZH)**: 未来的行星探测漫游者必须在结合稳态放射性同位素热电动势发生器（RTG）输出与变异性太阳能光伏（PV）供应的混合电源输入下，进行延长时长的操作。虽然能量感知规划已经在受电池限制的空中和水下机器人中进行了研究，但对于地面漫游者而言，鲜有工作明确建模功率流或强制即时功率约束。传统的地形感知规划强调坡度或可通行性，而轨迹优化方法通常集中在几何光滑性和动态可行性上，忽略了能量可行性。我们提出了一种能量约束轨迹规划框架，该框架明确整合了平移、旋转和阻力功率的物理模型，以及基础子系统的负载，适用于混合RTG-太阳能输入。通过将累积能量预算和即时功率约束整合到SE(2)基于多项式的轨迹优化中，该方法保证了同时光滑、动态可行和功率合规的轨迹。在月球地形上的仿真结果表明，我们的规划器生成的轨迹峰值功率在规定限制的0.55%以内，而现有方法则超过限制17%以上。这证明了一种原则性和实用性的能量感知自主方法，适用于长时长行星任务。 

---
# Ask-to-Clarify: Resolving Instruction Ambiguity through Multi-turn Dialogue 

**Title (ZH)**: 求证以澄清：通过多轮对话解决指令歧义 

**Authors**: Xingyao Lin, Xinghao Zhu, Tianyi Lu, Sicheng Xie, Hui Zhang, Xipeng Qiu, Zuxuan Wu, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.15061)  

**Abstract**: The ultimate goal of embodied agents is to create collaborators that can interact with humans, not mere executors that passively follow instructions. This requires agents to communicate, coordinate, and adapt their actions based on human feedback. Recently, advances in VLAs have offered a path toward this goal. However, most current VLA-based embodied agents operate in a one-way mode: they receive an instruction and execute it without feedback. This approach fails in real-world scenarios where instructions are often ambiguous. In this paper, we address this problem with the Ask-to-Clarify framework. Our framework first resolves ambiguous instructions by asking questions in a multi-turn dialogue. Then it generates low-level actions end-to-end. Specifically, the Ask-to-Clarify framework consists of two components, one VLM for collaboration and one diffusion for action. We also introduce a connection module that generates conditions for the diffusion based on the output of the VLM. This module adjusts the observation by instructions to create reliable conditions. We train our framework with a two-stage knowledge-insulation strategy. First, we fine-tune the collaboration component using ambiguity-solving dialogue data to handle ambiguity. Then, we integrate the action component while freezing the collaboration one. This preserves the interaction abilities while fine-tuning the diffusion to generate actions. The training strategy guarantees our framework can first ask questions, then generate actions. During inference, a signal detector functions as a router that helps our framework switch between asking questions and taking actions. We evaluate the Ask-to-Clarify framework in 8 real-world tasks, where it outperforms existing state-of-the-art VLAs. The results suggest that our proposed framework, along with the training strategy, provides a path toward collaborative embodied agents. 

**Abstract (ZH)**: 基于VLA的具身代理的Ask-to- Clarify框架 

---
# Online Multi-Robot Coordination and Cooperation with Task Precedence Relationships 

**Title (ZH)**: 具有任务优先级关系的在线多机器人协同与合作 

**Authors**: Walker Gosrich, Saurav Agarwal, Kashish Garg, Siddharth Mayya, Matthew Malencia, Mark Yim, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.15052)  

**Abstract**: We propose a new formulation for the multi-robot task allocation problem that incorporates (a) complex precedence relationships between tasks, (b) efficient intra-task coordination, and (c) cooperation through the formation of robot coalitions. A task graph specifies the tasks and their relationships, and a set of reward functions models the effects of coalition size and preceding task performance. Maximizing task rewards is NP-hard; hence, we propose network flow-based algorithms to approximate solutions efficiently. A novel online algorithm performs iterative re-allocation, providing robustness to task failures and model inaccuracies to achieve higher performance than offline approaches. We comprehensively evaluate the algorithms in a testbed with random missions and reward functions and compare them to a mixed-integer solver and a greedy heuristic. Additionally, we validate the overall approach in an advanced simulator, modeling reward functions based on realistic physical phenomena and executing the tasks with realistic robot dynamics. Results establish efficacy in modeling complex missions and efficiency in generating high-fidelity task plans while leveraging task relationships. 

**Abstract (ZH)**: 我们提出了一种新的多机器人任务分配问题的表示方法，该方法包含（a）任务间的复杂优先关系，（b）高效的任务内协调，以及（c）通过机器人联盟进行合作。任务图规定了任务及其关系，一组奖励函数模型了联盟规模和前置任务性能的影响。最大化任务奖励是NP难问题；因此，我们提出基于网络流的算法来高效地近似求解。一种新颖的在线算法执行迭代重新分配，以实现对任务失败和模型不准确的鲁棒性，从而优于离线方法。我们全面评估了算法在具有随机任务和奖励函数的测试平台上的表现，并将其与混合整数求解器和贪婪启发式方法进行了比较。此外，我们在一个先进的模拟器中验证了整体方法，该模拟器基于现实物理现象建模奖励函数，并使用真实机器人动力学执行任务。结果证实了该方法在建模复杂任务和生成高保真的任务规划方面具有有效性与效率，同时利用了任务之间的关系。 

---
# Semantic-LiDAR-Inertial-Wheel Odometry Fusion for Robust Localization in Large-Scale Dynamic Environments 

**Title (ZH)**: 面向大规模动态环境的语义LiDAR-惯性-陀螺仪里程计融合鲁棒定位方法 

**Authors**: Haoxuan Jiang, Peicong Qian, Yusen Xie, Linwei Zheng, Xiaocong Li, Ming Liu, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.14999)  

**Abstract**: Reliable, drift-free global localization presents significant challenges yet remains crucial for autonomous navigation in large-scale dynamic environments. In this paper, we introduce a tightly-coupled Semantic-LiDAR-Inertial-Wheel Odometry fusion framework, which is specifically designed to provide high-precision state estimation and robust localization in large-scale dynamic environments. Our framework leverages an efficient semantic-voxel map representation and employs an improved scan matching algorithm, which utilizes global semantic information to significantly reduce long-term trajectory drift. Furthermore, it seamlessly fuses data from LiDAR, IMU, and wheel odometry using a tightly-coupled multi-sensor fusion Iterative Error-State Kalman Filter (iESKF). This ensures reliable localization without experiencing abnormal drift. Moreover, to tackle the challenges posed by terrain variations and dynamic movements, we introduce a 3D adaptive scaling strategy that allows for flexible adjustments to wheel odometry measurement weights, thereby enhancing localization precision. This study presents extensive real-world experiments conducted in a one-million-square-meter automated port, encompassing 3,575 hours of operational data from 35 Intelligent Guided Vehicles (IGVs). The results consistently demonstrate that our system outperforms state-of-the-art LiDAR-based localization methods in large-scale dynamic environments, highlighting the framework's reliability and practical value. 

**Abstract (ZH)**: 可靠的、无漂移的全局定位在大规模动态环境中的自主导航中面临重大挑战，但依然至关重要。本文介绍了紧耦合的语义-LiDAR-惯性-陀螺仪 odometry 融合框架，该框架专门设计用于提供在大规模动态环境中的高精度状态估计和稳健定位。该框架利用高效的语义体素地图表示，并采用改进的扫描配准算法，该算法利用全局语义信息显著减少长期轨迹漂移。此外，它通过使用紧耦合多传感器融合迭代错误状态卡尔曼滤波器（iESKF）无缝融合来自 LiDAR、IMU 和轮 odometry 的数据，确保在不发生异常漂移的情况下实现可靠定位。此外，为应对地形变化和动态移动的挑战，我们引入了一种三维自适应缩放策略，允许灵活调整轮 odometry 测量权重，从而提高定位精度。本研究在占地一平方公里的自动化港口进行了广泛的实地实验，涵盖来自35辆智能引导车（IGVs）的3,575小时操作数据。结果一致表明，我们的系统在大规模动态环境中优于最先进的基于 LiDAR 的定位方法，突显了该框架的可靠性和实用性。 

---
# ExT: Towards Scalable Autonomous Excavation via Large-Scale Multi-Task Pretraining and Fine-Tuning 

**Title (ZH)**: ExT: 通过大规模多任务预训练和微调实现可扩展的自主挖掘 

**Authors**: Yifan Zhai, Lorenzo Terenzi, Patrick Frey, Diego Garcia Soto, Pascal Egli, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2509.14992)  

**Abstract**: Scaling up the deployment of autonomous excavators is of great economic and societal importance. Yet it remains a challenging problem, as effective systems must robustly handle unseen worksite conditions and new hardware configurations. Current state-of-the-art approaches rely on highly engineered, task-specific controllers, which require extensive manual tuning for each new scenario. In contrast, recent advances in large-scale pretrained models have shown remarkable adaptability across tasks and embodiments in domains such as manipulation and navigation, but their applicability to heavy construction machinery remains largely unexplored. In this work, we introduce ExT, a unified open-source framework for large-scale demonstration collection, pretraining, and fine-tuning of multitask excavation policies. ExT policies are first trained on large-scale demonstrations collected from a mix of experts, then fine-tuned either with supervised fine-tuning (SFT) or reinforcement learning fine-tuning (RLFT) to specialize to new tasks or operating conditions. Through both simulation and real-world experiments, we show that pretrained ExT policies can execute complete excavation cycles with centimeter-level accuracy, successfully transferring from simulation to real machine with performance comparable to specialized single-task controllers. Furthermore, in simulation, we demonstrate that ExT's fine-tuning pipelines allow rapid adaptation to new tasks, out-of-distribution conditions, and machine configurations, while maintaining strong performance on previously learned tasks. These results highlight the potential of ExT to serve as a foundation for scalable and generalizable autonomous excavation. 

**Abstract (ZH)**: 扩大自主挖掘机的部署对于经济和社会具有重要意义。然而，这仍然是一个挑战性的问题，因为有效的系统必须在面对未见过的施工条件和新的硬件配置时表现出高度的鲁棒性。当前最先进的方法依赖于高度工程化、任务特定的控制器，这些控制器需要对每个新场景进行大量的手动调优。相比之下，最近在大规模预训练模型方面取得的进展在操作和导航等领域展示了出色的跨任务和体态适应性，但这些模型在重型建筑机械中的应用尚未得到充分探索。在本文中，我们介绍了一体化开源框架ExT，用于大规模演示收集、预训练和多任务挖掘策略的微调。ExT策略首先在混合专家收集的大规模演示数据上进行训练，然后通过监督微调（SFT）或强化学习微调（RLFT）进行进一步微调，以专门化于新任务或操作条件。通过仿真和实地试验，我们展示了预训练的ExT策略可以以厘米级的精度执行完整的挖掘循环，并成功地将仿真中的性能转移到实际机器上，性能与专门化单任务控制器相当。此外，在仿真中，我们展示了ExT的微调管道能够快速适应新任务、离分布条件和机器配置，同时在之前学习的任务上保持强劲的性能。这些结果突显了ExT作为可扩展和通用自主挖掘基础的潜力。 

---
# The Role of Touch: Towards Optimal Tactile Sensing Distribution in Anthropomorphic Hands for Dexterous In-Hand Manipulation 

**Title (ZH)**: 触觉的作用：Towards Anthropomorphic Hands中最佳触觉传感分布以实现灵巧的手内操作 

**Authors**: João Damião Almeida, Egidio Falotico, Cecilia Laschi, José Santos-Victor  

**Link**: [PDF](https://arxiv.org/pdf/2509.14984)  

**Abstract**: In-hand manipulation tasks, particularly in human-inspired robotic systems, must rely on distributed tactile sensing to achieve precise control across a wide variety of tasks. However, the optimal configuration of this network of sensors is a complex problem, and while the fingertips are a common choice for placing sensors, the contribution of tactile information from other regions of the hand is often overlooked. This work investigates the impact of tactile feedback from various regions of the fingers and palm in performing in-hand object reorientation tasks. We analyze how sensory feedback from different parts of the hand influences the robustness of deep reinforcement learning control policies and investigate the relationship between object characteristics and optimal sensor placement. We identify which tactile sensing configurations contribute to improving the efficiency and accuracy of manipulation. Our results provide valuable insights for the design and use of anthropomorphic end-effectors with enhanced manipulation capabilities. 

**Abstract (ZH)**: 基于手部的触觉反馈在执行手内物体重新定向任务中的影响：触觉传感器配置优化研究 

---
# M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation 

**Title (ZH)**: M4Diffuser: 多视图扩散策略与适应性控制以实现 robust 移动操作Manipulability-Aware Control for Robust Mobile Manipulation 

**Authors**: Ju Dong, Lei Zhang, Liding Zhang, Yao Ling, Yu Fu, Kaixin Bai, Zoltán-Csaba Márton, Zhenshan Bing, Zhaopeng Chen, Alois Christian Knoll, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14980)  

**Abstract**: Mobile manipulation requires the coordinated control of a mobile base and a robotic arm while simultaneously perceiving both global scene context and fine-grained object details. Existing single-view approaches often fail in unstructured environments due to limited fields of view, exploration, and generalization abilities. Moreover, classical controllers, although stable, struggle with efficiency and manipulability near singularities. To address these challenges, we propose M4Diffuser, a hybrid framework that integrates a Multi-View Diffusion Policy with a novel Reduced and Manipulability-aware QP (ReM-QP) controller for mobile manipulation. The diffusion policy leverages proprioceptive states and complementary camera perspectives with both close-range object details and global scene context to generate task-relevant end-effector goals in the world frame. These high-level goals are then executed by the ReM-QP controller, which eliminates slack variables for computational efficiency and incorporates manipulability-aware preferences for robustness near singularities. Comprehensive experiments in simulation and real-world environments show that M4Diffuser achieves 7 to 56 percent higher success rates and reduces collisions by 3 to 31 percent over baselines. Our approach demonstrates robust performance for smooth whole-body coordination, and strong generalization to unseen tasks, paving the way for reliable mobile manipulation in unstructured environments. Details of the demo and supplemental material are available on our project website this https URL. 

**Abstract (ZH)**: 移动 manipulation 需要协调控制移动基座和机器人臂，同时感知全局场景上下文和细粒度物体细节。现有的单视角方法在未结构化环境中常常因为视野有限、探索能力和泛化能力不足而失败。此外，虽然经典的控制器稳定，但在接近奇异点时效率和可操作性较差。为了解决这些问题，我们提出了 M4Diffuser，这是一种结合了多视角扩散策略和一种新颖的减少并操作性感知的二次规划控制器 (ReM-QP) 的混合框架，用于移动 manipulation。扩散策略利用本体感知状态和互补摄像头视角，结合近距离物体细节和全局场景上下文，生成在世界坐标系中的任务相关末端执行器目标。这些高层次的目标随后由 ReM-QP 控制器执行，该控制器通过消除松弛变量提高计算效率，并结合操作性感知的偏好以在接近奇异点时增强鲁棒性。在模拟和现实环境中的全面实验表明，M4Diffuser 的成功率提高了 7% 至 56%，碰撞减少了 3% 至 31%。我们的方法展示了平滑的全身协调的稳健性能，并在未见过的任务中表现出强大的泛化能力，为在未结构化环境中实现可靠的移动 manipulation 奠定了基础。更多演示细节和补充材料请参见我们的项目网站：这个 https URL。 

---
# PA-MPPI: Perception-Aware Model Predictive Path Integral Control for Quadrotor Navigation in Unknown Environments 

**Title (ZH)**: PA-MPPI: awareness of Perception in Model Predictive Path Integral Control for Quadrotor Navigation in Unknown Environments 

**Authors**: Yifan Zhai, Rudolf Reiter, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2509.14978)  

**Abstract**: Quadrotor navigation in unknown environments is critical for practical missions such as search-and-rescue. Solving it requires addressing three key challenges: the non-convexity of free space due to obstacles, quadrotor-specific dynamics and objectives, and the need for exploration of unknown regions to find a path to the goal. Recently, the Model Predictive Path Integral (MPPI) method has emerged as a promising solution that solves the first two challenges. By leveraging sampling-based optimization, it can effectively handle non-convex free space while directly optimizing over the full quadrotor dynamics, enabling the inclusion of quadrotor-specific costs such as energy consumption. However, its performance in unknown environments is limited, as it lacks the ability to explore unknown regions when blocked by large obstacles. To solve this issue, we introduce Perception-Aware MPPI (PA-MPPI). Here, perception-awareness is defined as adapting the trajectory online based on perception objectives. Specifically, when the goal is occluded, PA-MPPI's perception cost biases trajectories that can perceive unknown regions. This expands the mapped traversable space and increases the likelihood of finding alternative paths to the goal. Through hardware experiments, we demonstrate that PA-MPPI, running at 50 Hz with our efficient perception and mapping module, performs up to 100% better than the baseline in our challenging settings where the state-of-the-art MPPI fails. In addition, we demonstrate that PA-MPPI can be used as a safe and robust action policy for navigation foundation models, which often provide goal poses that are not directly reachable. 

**Abstract (ZH)**: 无感知环境下四旋翼导航对于搜索救援等实际任务至关重要。为此，需要解决三个关键挑战：由障碍物引起的非凸自由空间、四旋翼特有的动态特性和目标以及探索未知区域以找到通往目标的路径的需要。最近，模型预测路径积分（MPPI）方法被认为是对前两个挑战的有效解决方案。通过利用基于采样的优化，它可以有效地处理非凸自由空间，并直接优化四旋翼的动力学，从而能够包含四旋翼特定的成本，如能耗。然而，其在未知环境中的表现有限，因为它在被大障碍物阻挡时缺乏探索未知区域的能力。为了解决这个问题，我们引入了感知导向的MPPI（PA-MPPI）。在这里，感知导向指的是根据感知目标在线调整轨迹。具体而言，当目标被遮挡时，PA-MPPI 的感知成本会偏向能够感知未知区域的轨迹，从而扩展可通行空间，并增加找到通往目标的替代路径的概率。通过硬件实验，我们证明，在我们的挑战性设置中，与最先进的MPPI相比，使用我们高效的感知和映射模块在50 Hz运行的PA-MPPI表现提高了100%。此外，我们证明了PA-MPPI可以作为导航基础模型的安全且鲁棒的动作策略，这些模型通常提供了直接不可达的目标姿态。 

---
# Affordance-Based Disambiguation of Surgical Instructions for Collaborative Robot-Assisted Surgery 

**Title (ZH)**: 基于功能的手术指令歧义消解for协作机器人辅助手术 

**Authors**: Ana Davila, Jacinto Colan, Yasuhisa Hasegawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.14967)  

**Abstract**: Effective human-robot collaboration in surgery is affected by the inherent ambiguity of verbal communication. This paper presents a framework for a robotic surgical assistant that interprets and disambiguates verbal instructions from a surgeon by grounding them in the visual context of the operating field. The system employs a two-level affordance-based reasoning process that first analyzes the surgical scene using a multimodal vision-language model and then reasons about the instruction using a knowledge base of tool capabilities. To ensure patient safety, a dual-set conformal prediction method is used to provide a statistically rigorous confidence measure for robot decisions, allowing it to identify and flag ambiguous commands. We evaluated our framework on a curated dataset of ambiguous surgical requests from cholecystectomy videos, demonstrating a general disambiguation rate of 60% and presenting a method for safer human-robot interaction in the operating room. 

**Abstract (ZH)**: 有效的手术中人机协作受到口头通讯固有模糊性的影响。本文提出了一种基于视觉场景解释和消歧手术指令的外科机器人助手框架。该系统采用基于能力的两层推理过程，首先利用多模态视觉-语言模型分析手术场景，然后利用工具能力知识库推理指令。为确保患者安全，该系统采用双重契合预测方法提供统计上严格的置信度度量，使机器人能够识别和标注模糊命令。我们通过对胆囊切除手术视频中人工整理的模糊手术请求数据集进行评估，展示了60%的消歧率，并提出了一种在手术室中实现更安全人机交互的方法。 

---
# Exploratory Movement Strategies for Texture Discrimination with a Neuromorphic Tactile Sensor 

**Title (ZH)**: 触觉神经形态传感器中用于纹理鉴别的一般探索运动策略 

**Authors**: Xingchen Xu, Ao Li, Benjamin Ward-Cherrier  

**Link**: [PDF](https://arxiv.org/pdf/2509.14954)  

**Abstract**: We propose a neuromorphic tactile sensing framework for robotic texture classification that is inspired by human exploratory strategies. Our system utilizes the NeuroTac sensor to capture neuromorphic tactile data during a series of exploratory motions. We first tested six distinct motions for texture classification under fixed environment: sliding, rotating, tapping, as well as the combined motions: sliding+rotating, tapping+rotating, and tapping+sliding. We chose sliding and sliding+rotating as the best motions based on final accuracy and the sample timing length needed to reach converged accuracy. In the second experiment designed to simulate complex real-world conditions, these two motions were further evaluated under varying contact depth and speeds. Under these conditions, our framework attained the highest accuracy of 87.33\% with sliding+rotating while maintaining an extremely low power consumption of only 8.04 mW. These results suggest that the sliding+rotating motion is the optimal exploratory strategy for neuromorphic tactile sensing deployment in texture classification tasks and holds significant promise for enhancing robotic environmental interaction. 

**Abstract (ZH)**: 一种受人类探索策略启发的神经形态触觉传感框架及其在纹理分类中的应用 

---
# Human Interaction for Collaborative Semantic SLAM using Extended Reality 

**Title (ZH)**: 扩展现实环境下的人机协作语义SLAM 

**Authors**: Laura Ribeiro, Muhammad Shaheer, Miguel Fernandez-Cortizas, Ali Tourani, Holger Voos, Jose Luis Sanchez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2509.14949)  

**Abstract**: Semantic SLAM (Simultaneous Localization and Mapping) systems enrich robot maps with structural and semantic information, enabling robots to operate more effectively in complex environments. However, these systems struggle in real-world scenarios with occlusions, incomplete data, or ambiguous geometries, as they cannot fully leverage the higher-level spatial and semantic knowledge humans naturally apply. We introduce HICS-SLAM, a Human-in-the-Loop semantic SLAM framework that uses a shared extended reality environment for real-time collaboration. The system allows human operators to directly interact with and visualize the robot's 3D scene graph, and add high-level semantic concepts (e.g., rooms or structural entities) into the mapping process. We propose a graph-based semantic fusion methodology that integrates these human interventions with robot perception, enabling scalable collaboration for enhanced situational awareness. Experimental evaluations on real-world construction site datasets demonstrate improvements in room detection accuracy, map precision, and semantic completeness compared to automated baselines, demonstrating both the effectiveness of the approach and its potential for future extensions. 

**Abstract (ZH)**: 基于人类在环的语义SLAM框架 

---
# Multi-CAP: A Multi-Robot Connectivity-Aware Hierarchical Coverage Path Planning Algorithm for Unknown Environments 

**Title (ZH)**: 多CAP：面向未知环境的多机器人连通感知层次覆盖路径规划算法 

**Authors**: Zongyuan Shen, Burhanuddin Shirose, Prasanna Sriganesh, Bhaskar Vundurthy, Howie Choset, Matthew Travers  

**Link**: [PDF](https://arxiv.org/pdf/2509.14941)  

**Abstract**: Efficient coordination of multiple robots for coverage of large, unknown environments is a significant challenge that involves minimizing the total coverage path length while reducing inter-robot conflicts. In this paper, we introduce a Multi-robot Connectivity-Aware Planner (Multi-CAP), a hierarchical coverage path planning algorithm that facilitates multi-robot coordination through a novel connectivity-aware approach. The algorithm constructs and dynamically maintains an adjacency graph that represents the environment as a set of connected subareas. Critically, we make the assumption that the environment, while unknown, is bounded. This allows for incremental refinement of the adjacency graph online to ensure its structure represents the physical layout of the space, both in observed and unobserved areas of the map as robots explore the environment. We frame the task of assigning subareas to robots as a Vehicle Routing Problem (VRP), a well-studied problem for finding optimal routes for a fleet of vehicles. This is used to compute disjoint tours that minimize redundant travel, assigning each robot a unique, non-conflicting set of subareas. Each robot then executes its assigned tour, independently adapting its coverage strategy within each subarea to minimize path length based on real-time sensor observations of the subarea. We demonstrate through simulations and multi-robot hardware experiments that Multi-CAP significantly outperforms state-of-the-art methods in key metrics, including coverage time, total path length, and path overlap ratio. Ablation studies further validate the critical role of our connectivity-aware graph and the global tour planner in achieving these performance gains. 

**Abstract (ZH)**: 多机器人连通性意识规划算法（Multi-CAP）：一种新型的多机器人协调覆盖路径规划方法 

---
# A Novel Task-Driven Diffusion-Based Policy with Affordance Learning for Generalizable Manipulation of Articulated Objects 

**Title (ZH)**: 一种基于任务驱动的扩散策略及其在可迁移操纵关节物体中的 affordance 学习 

**Authors**: Hao Zhang, Zhen Kan, Weiwei Shang, Yongduan Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.14939)  

**Abstract**: Despite recent advances in dexterous manipulations, the manipulation of articulated objects and generalization across different categories remain significant challenges. To address these issues, we introduce DART, a novel framework that enhances a diffusion-based policy with affordance learning and linear temporal logic (LTL) representations to improve the learning efficiency and generalizability of articulated dexterous manipulation. Specifically, DART leverages LTL to understand task semantics and affordance learning to identify optimal interaction points. The {diffusion-based policy} then generalizes these interactions across various categories. Additionally, we exploit an optimization method based on interaction data to refine actions, overcoming the limitations of traditional diffusion policies that typically rely on offline reinforcement learning or learning from demonstrations. Experimental results demonstrate that DART outperforms most existing methods in manipulation ability, generalization performance, transfer reasoning, and robustness. For more information, visit our project website at: this https URL. 

**Abstract (ZH)**: 尽管在灵巧操作方面取得了近期进展，但 articulated 对象的操作以及不同类别间的泛化仍然是显著挑战。为此，我们提出了 DART，一种通过引入作用域学习和线性时序逻辑（LTL）表示来增强基于扩散的策略的新型框架，以提高 articulated 灵巧操作的学习效率和泛化能力。具体而言，DART 利用 LTL 理解任务语义，利用作用域学习识别最优交互点，然后基于这些交互在各类别间进行泛化。此外，我们利用基于交互数据的优化方法来细化动作，克服了传统基于扩散的策略通常依赖脱机强化学习或模仿学习的局限性。实验结果表明，DART 在操作能力、泛化性能、迁移推理能力和鲁棒性方面均优于大多数现有方法。 

---
# CAD-Driven Co-Design for Flight-Ready Jet-Powered Humanoids 

**Title (ZH)**: CAD驱动的人形飞行机器人协同设计 

**Authors**: Punith Reddy Vanteddu, Davide Gorbani, Giuseppe L'Erario, Hosameldin Awadalla Omer Mohamed, Fabio Bergonti, Daniele Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.14935)  

**Abstract**: This paper presents a CAD-driven co-design framework for optimizing jet-powered aerial humanoid robots to execute dynamically constrained trajectories. Starting from the iRonCub-Mk3 model, a Design of Experiments (DoE) approach is used to generate 5,000 geometrically varied and mechanically feasible designs by modifying limb dimensions, jet interface geometry (e.g., angle and offset), and overall mass distribution. Each model is constructed through CAD assemblies to ensure structural validity and compatibility with simulation tools. To reduce computational cost and enable parameter sensitivity analysis, the models are clustered using K-means, with representative centroids selected for evaluation. A minimum-jerk trajectory is used to assess flight performance, providing position and velocity references for a momentum-based linearized Model Predictive Control (MPC) strategy. A multi-objective optimization is then conducted using the NSGA-II algorithm, jointly exploring the space of design centroids and MPC gain parameters. The objectives are to minimize trajectory tracking error and mechanical energy expenditure. The framework outputs a set of flight-ready humanoid configurations with validated control parameters, offering a structured method for selecting and implementing feasible aerial humanoid designs. 

**Abstract (ZH)**: 基于CAD驱动的协同设计框架：优化喷气动力空中人形机器人以执行动态约束轨迹 

---
# Robot Control Stack: A Lean Ecosystem for Robot Learning at Scale 

**Title (ZH)**: 机器人控制栈：大规模机器人学习的精简生态系统 

**Authors**: Tobias Jülg, Pierre Krack, Seongjin Bien, Yannik Blei, Khaled Gamal, Ken Nakahara, Johannes Hechtl, Roberto Calandra, Wolfram Burgard, Florian Walter  

**Link**: [PDF](https://arxiv.org/pdf/2509.14932)  

**Abstract**: Vision-Language-Action models (VLAs) mark a major shift in robot learning. They replace specialized architectures and task-tailored components of expert policies with large-scale data collection and setup-specific fine-tuning. In this machine learning-focused workflow that is centered around models and scalable training, traditional robotics software frameworks become a bottleneck, while robot simulations offer only limited support for transitioning from and to real-world experiments. In this work, we close this gap by introducing Robot Control Stack (RCS), a lean ecosystem designed from the ground up to support research in robot learning with large-scale generalist policies. At its core, RCS features a modular and easily extensible layered architecture with a unified interface for simulated and physical robots, facilitating sim-to-real transfer. Despite its minimal footprint and dependencies, it offers a complete feature set, enabling both real-world experiments and large-scale training in simulation. Our contribution is twofold: First, we introduce the architecture of RCS and explain its design principles. Second, we evaluate its usability and performance along the development cycle of VLA and RL policies. Our experiments also provide an extensive evaluation of Octo, OpenVLA, and Pi Zero on multiple robots and shed light on how simulation data can improve real-world policy performance. Our code, datasets, weights, and videos are available at: this https URL 

**Abstract (ZH)**: Vision-Language-Action模型（VLAs）标志着机器人学习的一个重大转变。它们用大规模数据收集和任务特定的微调取代了专家策略中的专门架构和定制组件。在这一以模型为中心、重视大规模训练的机器学习工作流程中，传统的机器人软件框架成为瓶颈，而机器人模拟仅在从实世界实验到模拟实验的过渡中提供有限的支持。在本项工作中，我们通过引入机器人控制栈（RCS），一个从头开始设计以支持大规模通用策略的机器人学习研究的精简生态系统，填补了这一空白。RCS的核心是一个模块化且易于扩展的分层架构，具有统一的模拟和物理机器人接口，促进从模拟到现实的过渡。尽管RCS占用空间小且依赖项少，但它提供了完整的功能集，支持实世界实验和大规模模拟训练。我们的贡献包括：首先，我们介绍了RCS的架构并解释了其设计原则。其次，我们评估了它在整个VLAs和强化学习（RL）策略开发周期中的可用性和性能。我们的实验还对Octo、OpenVLA和Pi Zero在多个机器人上的性能进行了广泛评估，并揭示了模拟数据如何提高实世界策略的表现。我们的代码、数据集、权重和视频可在以下链接获取：this https URL。 

---
# PERAL: Perception-Aware Motion Control for Passive LiDAR Excitation in Spherical Robots 

**Title (ZH)**: 感知导向的运动控制以实现球形机器人中被动LiDAR的激发 

**Authors**: Shenghai Yuan, Jason Wai Hao Yee, Weixiang Guo, Zhongyuan Liu, Thien-Minh Nguyen, Lihua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.14915)  

**Abstract**: Autonomous mobile robots increasingly rely on LiDAR-IMU odometry for navigation and mapping, yet horizontally mounted LiDARs such as the MID360 capture few near-ground returns, limiting terrain awareness and degrading performance in feature-scarce environments. Prior solutions - static tilt, active rotation, or high-density sensors - either sacrifice horizontal perception or incur added actuators, cost, and power. We introduce PERAL, a perception-aware motion control framework for spherical robots that achieves passive LiDAR excitation without dedicated hardware. By modeling the coupling between internal differential-drive actuation and sensor attitude, PERAL superimposes bounded, non-periodic oscillations onto nominal goal- or trajectory-tracking commands, enriching vertical scan diversity while preserving navigation accuracy. Implemented on a compact spherical robot, PERAL is validated across laboratory, corridor, and tactical environments. Experiments demonstrate up to 96 percent map completeness, a 27 percent reduction in trajectory tracking error, and robust near-ground human detection, all at lower weight, power, and cost compared with static tilt, active rotation, and fixed horizontal baselines. The design and code will be open-sourced upon acceptance. 

**Abstract (ZH)**: 基于感知的运动控制框架：无专用地面机器人LiDAR激发方法 

---
# CollabVLA: Self-Reflective Vision-Language-Action Model Dreaming Together with Human 

**Title (ZH)**: CollabVLA: 自我反思的视觉-语言-行动模型与人类共同梦想 

**Authors**: Nan Sun, Yongchang Li, Chenxu Wang, Huiying Li, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14889)  

**Abstract**: In this work, we present CollabVLA, a self-reflective vision-language-action framework that transforms a standard visuomotor policy into a collaborative assistant. CollabVLA tackles key limitations of prior VLAs, including domain overfitting, non-interpretable reasoning, and the high latency of auxiliary generative models, by integrating VLM-based reflective reasoning with diffusion-based action generation under a mixture-of-experts design. Through a two-stage training recipe of action grounding and reflection tuning, it supports explicit self-reflection and proactively solicits human guidance when confronted with uncertainty or repeated failure. It cuts normalized Time by ~2x and Dream counts by ~4x vs. generative agents, achieving higher success rates, improved interpretability, and balanced low latency compared with existing methods. This work takes a pioneering step toward shifting VLAs from opaque controllers to genuinely assistive agents capable of reasoning, acting, and collaborating with humans. 

**Abstract (ZH)**: CollabVLA: 一种自省的视觉-语言-行动框架 

---
# Scalable Multi-Objective Robot Reinforcement Learning through Gradient Conflict Resolution 

**Title (ZH)**: 通过梯度冲突解决的可扩展多目标机器人强化学习 

**Authors**: Humphrey Munn, Brendan Tidd, Peter Böhm, Marcus Gallagher, David Howard  

**Link**: [PDF](https://arxiv.org/pdf/2509.14816)  

**Abstract**: Reinforcement Learning (RL) robot controllers usually aggregate many task objectives into one scalar reward. While large-scale proximal policy optimisation (PPO) has enabled impressive results such as robust robot locomotion in the real world, many tasks still require careful reward tuning and are brittle to local optima. Tuning cost and sub-optimality grow with the number of objectives, limiting scalability. Modelling reward vectors and their trade-offs can address these issues; however, multi-objective methods remain underused in RL for robotics because of computational cost and optimisation difficulty. In this work, we investigate the conflict between gradient contributions for each objective that emerge from scalarising the task objectives. In particular, we explicitly address the conflict between task-based rewards and terms that regularise the policy towards realistic behaviour. We propose GCR-PPO, a modification to actor-critic optimisation that decomposes the actor update into objective-wise gradients using a multi-headed critic and resolves conflicts based on the objective priority. Our methodology, GCR-PPO, is evaluated on the well-known IsaacLab manipulation and locomotion benchmarks and additional multi-objective modifications on two related tasks. We show superior scalability compared to parallel PPO (p = 0.04), without significant computational overhead. We also show higher performance with more conflicting tasks. GCR-PPO improves on large-scale PPO with an average improvement of 9.5%, with high-conflict tasks observing a greater improvement. The code is available at this https URL. 

**Abstract (ZH)**: 基于梯度贡献的多目标PPO方法：解决机器人控制中的冲突问题 

---
# COMPASS: Confined-space Manipulation Planning with Active Sensing Strategy 

**Title (ZH)**: COMPASS: 有限空间操作规划与主动传感策略 

**Authors**: Qixuan Li, Chen Le, Dongyue Huang, Jincheng Yu, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.14787)  

**Abstract**: Manipulation in confined and cluttered environments remains a significant challenge due to partial observability and complex configuration spaces. Effective manipulation in such environments requires an intelligent exploration strategy to safely understand the scene and search the target. In this paper, we propose COMPASS, a multi-stage exploration and manipulation framework featuring a manipulation-aware sampling-based planner. First, we reduce collision risks with a near-field awareness scan to build a local collision map. Additionally, we employ a multi-objective utility function to find viewpoints that are both informative and conducive to subsequent manipulation. Moreover, we perform a constrained manipulation optimization strategy to generate manipulation poses that respect obstacle constraints. To systematically evaluate method's performance under these difficulties, we propose a benchmark of confined-space exploration and manipulation containing four level challenging scenarios. Compared to exploration methods designed for other robots and only considering information gain, our framework increases manipulation success rate by 24.25% in simulations. Real-world experiments demonstrate our method's capability for active sensing and manipulation in confined environments. 

**Abstract (ZH)**: 在受限和杂乱环境中的操作仍因部分可观测性和复杂的空间配置而构成重大挑战。有效的操作需要智能的探索策略以安全地理解场景并搜索目标。本文提出了一种多阶段操作探索框架COMPASS，该框架具有感知操作的采样计划器。首先，我们通过近场感知扫描来减少碰撞风险并构建局部碰撞图。此外，我们采用多目标效用函数来寻找既具有信息价值又有利于随后操作的视点。进一步地，我们执行受约束的操作优化策略以生成遵守障碍物约束的操作姿态。为了系统地评估该方法在这些困难环境下的性能，我们提出了一种受限空间探索与操作的基准，其中包括四个具有挑战性的场景。与仅考虑信息增益的为其他机器人设计的探索方法相比，在仿真实验中，我们的框架使操作成功率提高了24.25%。实际实验展示了该方法在受限环境中的主动感知和操作能力。 

---
# Designing Latent Safety Filters using Pre-Trained Vision Models 

**Title (ZH)**: 使用预训练视觉模型设计潜在安全性滤波器 

**Authors**: Ihab Tabbara, Yuxuan Yang, Ahmad Hamzeh, Maxwell Astafyev, Hussein Sibai  

**Link**: [PDF](https://arxiv.org/pdf/2509.14758)  

**Abstract**: Ensuring safety of vision-based control systems remains a major challenge hindering their deployment in critical settings. Safety filters have gained increased interest as effective tools for ensuring the safety of classical control systems, but their applications in vision-based control settings have so far been limited. Pre-trained vision models (PVRs) have been shown to be effective perception backbones for control in various robotics domains. In this paper, we are interested in examining their effectiveness when used for designing vision-based safety filters. We use them as backbones for classifiers defining failure sets, for Hamilton-Jacobi (HJ) reachability-based safety filters, and for latent world models. We discuss the trade-offs between training from scratch, fine-tuning, and freezing the PVRs when training the models they are backbones for. We also evaluate whether one of the PVRs is superior across all tasks, evaluate whether learned world models or Q-functions are better for switching decisions to safe policies, and discuss practical considerations for deploying these PVRs on resource-constrained devices. 

**Abstract (ZH)**: 确保基于视觉的控制系统安全仍然是一个主要挑战，阻碍了其在关键环境中的部署。预训练视觉模型（PVRs）在各种机器人领域中已被证明是有效的感知骨干网络，用于控制。本文旨在探讨它们在设计基于视觉的安全过滤器时的有效性。我们将它们用作分类器的骨干网络，定义故障集，用于Hamilton-Jacobi（HJ）可达到性基于的安全过滤器，以及用于潜在世界模型。我们讨论了从零开始训练、微调和冻结PVRs在训练它们作为骨干的模型时的权衡。我们还评估了在各种任务中哪一个PVR更优，评估了学习到的世界模型或Q函数是否更适合切换决策至安全策略，并讨论了在资源受限设备上部署这些PVRs的实用考虑。 

---
# Investigating the Effect of LED Signals and Emotional Displays in Human-Robot Shared Workspaces 

**Title (ZH)**: 探究LED信号和情感显示在人机共融工作空间中的效果 

**Authors**: Maria Ibrahim, Alap Kshirsagar, Dorothea Koert, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2509.14748)  

**Abstract**: Effective communication is essential for safety and efficiency in human-robot collaboration, particularly in shared workspaces. This paper investigates the impact of nonverbal communication on human-robot interaction (HRI) by integrating reactive light signals and emotional displays into a robotic system. We equipped a Franka Emika Panda robot with an LED strip on its end effector and an animated facial display on a tablet to convey movement intent through colour-coded signals and facial expressions. We conducted a human-robot collaboration experiment with 18 participants, evaluating three conditions: LED signals alone, LED signals with reactive emotional displays, and LED signals with pre-emptive emotional displays. We collected data through questionnaires and position tracking to assess anticipation of potential collisions, perceived clarity of communication, and task performance. The results indicate that while emotional displays increased the perceived interactivity of the robot, they did not significantly improve collision anticipation, communication clarity, or task efficiency compared to LED signals alone. These findings suggest that while emotional cues can enhance user engagement, their impact on task performance in shared workspaces is limited. 

**Abstract (ZH)**: 非言语交流对人类与机器人合作中安全与效率的影响研究：通过集成反应性光信号和情绪展示改善人机交互 

---
# Rethinking Reference Trajectories in Agile Drone Racing: A Unified Reference-Free Model-Based Controller via MPPI 

**Title (ZH)**: 重新思考敏捷无人机竞速中的参考轨迹：基于MPPI的统一参考自由模型控制器 

**Authors**: Fangguo Zhao, Xin Guan, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.14726)  

**Abstract**: While model-based controllers have demonstrated remarkable performance in autonomous drone racing, their performance is often constrained by the reliance on pre-computed reference trajectories. Conventional approaches, such as trajectory tracking, demand a dynamically feasible, full-state reference, whereas contouring control relaxes this requirement to a geometric path but still necessitates a reference. Recent advancements in reinforcement learning (RL) have revealed that many model-based controllers optimize surrogate objectives, such as trajectory tracking, rather than the primary racing goal of directly maximizing progress through gates. Inspired by these findings, this work introduces a reference-free method for time-optimal racing by incorporating this gate progress objective, derived from RL reward shaping, directly into the Model Predictive Path Integral (MPPI) formulation. The sampling-based nature of MPPI makes it uniquely capable of optimizing the discontinuous and non-differentiable objective in real-time. We also establish a unified framework that leverages MPPI to systematically and fairly compare three distinct objective functions with a consistent dynamics model and parameter set: classical trajectory tracking, contouring control, and the proposed gate progress objective. We compare the performance of these three objectives when solved via both MPPI and a traditional gradient-based solver. Our results demonstrate that the proposed reference-free approach achieves competitive racing performance, rivaling or exceeding reference-based methods. Videos are available at this https URL 

**Abstract (ZH)**: 基于模型的控制器在自主无人机竞速中展现出卓越的性能，但其性能往往受限于依赖预计算的参考轨迹。传统的轨迹跟踪方法需要动态可行的全状态参考轨迹，而轮廓控制放宽了这一要求，只需几何路径参考，但仍需参考轨迹。最近在强化学习领域的进展表明，许多基于模型的控制器优化的是轨迹跟踪等替代目标，而不是直接最大化通过障碍门的主要竞速目标。受这一发现的启发，本工作提出了一种无需参考轨迹的方法，通过将从强化学习奖励塑造中得出的障碍门进度目标直接纳入模型预测路径积分（MPPI）公式中来实现时间最优竞速。基于采样的性质使MPPI能够实时优化非连续性和非光滑目标。同时，我们建立了一个统一框架，利用MPPI系统地、公平地比较三种不同的目标函数：经典轨迹跟踪、轮廓控制和所提出的障碍门进度目标，这些方法采用相同的动态模型和参数集。我们比较了这些目标函数通过MPPI和传统梯度求解器求解时的性能。结果表明，提出的无需参考轨迹的方法在竞速性能上具有竞争力，能够匹敌或超越基于参考的方法。相关视频可在以下链接获取。 

---
# Wohlhart's Three-Loop Mechanism: An Overconstrained and Shaky Linkage 

**Title (ZH)**: Wohlhart的三环机制：一个过约束且不稳定的连杆机构 

**Authors**: Andreas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2509.14698)  

**Abstract**: This paper revisits a three-loop spatial linkage that was proposed in an ARK 2004 paper by Karl Wohlhart (as extension of a two-loop linkage proposed by Eddie Baker in 1980) and later analyzed in an ARK 2006 paper by Diez-Martinez et. al. A local analysis shows that this linkage has a finite degree of freedom (DOF) 3 (and is thus overconstrained) while in its reference configuration the differential DOF is 5. It is shown that its configuration space is locally a smooth manifold so that the reference configuration is not a c-space singularity. It is shown that the differential DOF is locally constant, which makes this linkage shaky (so that the reference configuration is not a singularity). The higher-order local analysis is facilitated by the computation of the kinematic tangent cone as well as a local approximation of the c-space. 

**Abstract (ZH)**: 本文重新审视了Karl Wohlhart在2004年ARK论文中提出的一个三环空间连杆机构（该机构是Eddie Baker于1980年提出的一个双环连杆机构的扩展），并随后在Diez-Martinez等人2006年ARK论文中进行了分析。局部分析表明，该连杆机构具有有限的自由度3（因此是过约束的），而在参考配置下，其微分自由度为5。证明其配置空间在局部是光滑流形，从而使参考配置不是c空间奇异点。证明其微分自由度在局部是恒定的，从而使该连杆机构具有脆弱性（因此参考配置不是奇异点）。更高阶的局部分析通过计算渐近切锥以及局部c空间的逼近得以实现。 

---
# exUMI: Extensible Robot Teaching System with Action-aware Task-agnostic Tactile Representation 

**Title (ZH)**: 扩展性机器人教学系统：具备动作感知的任务无关触觉表示 

**Authors**: Yue Xu, Litao Wei, Pengyu An, Qingyu Zhang, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.14688)  

**Abstract**: Tactile-aware robot learning faces critical challenges in data collection and representation due to data scarcity and sparsity, and the absence of force feedback in existing systems. To address these limitations, we introduce a tactile robot learning system with both hardware and algorithm innovations. We present exUMI, an extensible data collection device that enhances the vanilla UMI with robust proprioception (via AR MoCap and rotary encoder), modular visuo-tactile sensing, and automated calibration, achieving 100% data usability. Building on an efficient collection of over 1 M tactile frames, we propose Tactile Prediction Pretraining (TPP), a representation learning framework through action-aware temporal tactile prediction, capturing contact dynamics and mitigating tactile sparsity. Real-world experiments show that TPP outperforms traditional tactile imitation learning. Our work bridges the gap between human tactile intuition and robot learning through co-designed hardware and algorithms, offering open-source resources to advance contact-rich manipulation research. Project page: this https URL. 

**Abstract (ZH)**: 触觉感知机器人学习因数据稀缺性和稀疏性以及现有系统缺乏力反馈而面临关键挑战。为应对这些局限性，我们介绍了一种结合硬件和算法创新的触觉机器人学习系统。我们提出了exUMI，这是一种可扩展的数据采集设备，通过AR MoCap和旋转编码器增强了原始UMI的鲁棒本体感受能力，集成了模块化视触觉感知和自动标定，实现100%的数据利用率。基于高效采集的超过100万帧触觉图像，我们提出了触觉预测预训练（TPP），这是一种通过动作感知的时间触觉预测进行的表征学习框架，捕捉接触动力学并缓解触觉稀疏性。实验证明，TPP在触觉模仿学习中表现更优。我们的工作通过协同设计的硬件和算法将人类触觉直觉与机器人学习结合起来，提供开源资源以促进接触丰富的操作研究。项目页面：this https URL。 

---
# RealMirror: A Comprehensive, Open-Source Vision-Language-Action Platform for Embodied AI 

**Title (ZH)**: RealMirror: 一个全面的开源视觉-语言-行动平台，用于具身AI 

**Authors**: Cong Tai, Zhaoyu Zheng, Haixu Long, Hansheng Wu, Haodong Xiang, Zhengbin Long, Jun Xiong, Rong Shi, Shizhuang Zhang, Gang Qiu, He Wang, Ruifeng Li, Jun Huang, Bin Chang, Shuai Feng, Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.14687)  

**Abstract**: The emerging field of Vision-Language-Action (VLA) for humanoid robots faces several fundamental challenges, including the high cost of data acquisition, the lack of a standardized benchmark, and the significant gap between simulation and the real world. To overcome these obstacles, we propose RealMirror, a comprehensive, open-source embodied AI VLA platform. RealMirror builds an efficient, low-cost data collection, model training, and inference system that enables end-to-end VLA research without requiring a real robot. To facilitate model evolution and fair comparison, we also introduce a dedicated VLA benchmark for humanoid robots, featuring multiple scenarios, extensive trajectories, and various VLA models. Furthermore, by integrating generative models and 3D Gaussian Splatting to reconstruct realistic environments and robot models, we successfully demonstrate zero-shot Sim2Real transfer, where models trained exclusively on simulation data can perform tasks on a real robot seamlessly, without any fine-tuning. In conclusion, with the unification of these critical components, RealMirror provides a robust framework that significantly accelerates the development of VLA models for humanoid robots. Project page: this https URL 

**Abstract (ZH)**: Vision-Language-Action (VLA) for类人机器人 emerging领域面临Several Fundamental Challenges，包括数据采集成本高、缺乏标准化基准以及模拟与真实世界之间的显著差距。为克服这些障碍，我们提出了RealMirror，一个全面的开源嵌入式AI VLA平台。RealMirror构建了一个高效的低成本数据采集、模型训练和推理系统，使研究人员无需使用真实机器人即可进行端到端的VLA研究。为了促进模型进化和公平比较，我们还引入了一个针对类人机器人的专用VLA基准，包含多种场景、广泛轨迹和多种VLA模型。此外，通过结合生成模型和3D高斯散点图来重建现实环境和机器人模型，我们成功展示了零-shot的Sim2Real转移，即仅在仿真数据上训练的模型可以在真实机器人上无缝执行任务，无需任何微调。总之，通过统一这些关键组件，RealMirror提供了一个稳健的框架，显著加速了类人机器人VLA模型的发展。项目页面：this https URL 

---
# Efficient 3D Perception on Embedded Systems via Interpolation-Free Tri-Plane Lifting and Volume Fusion 

**Title (ZH)**: 无需插值的三平面提升与体素融合在嵌入式系统中的高效3D感知 

**Authors**: Sibaek Lee, Jiung Yeon, Hyeonwoo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14641)  

**Abstract**: Dense 3D convolutions provide high accuracy for perception but are too computationally expensive for real-time robotic systems. Existing tri-plane methods rely on 2D image features with interpolation, point-wise queries, and implicit MLPs, which makes them computationally heavy and unsuitable for embedded 3D inference. As an alternative, we propose a novel interpolation-free tri-plane lifting and volumetric fusion framework, that directly projects 3D voxels into plane features and reconstructs a feature volume through broadcast and summation. This shifts nonlinearity to 2D convolutions, reducing complexity while remaining fully parallelizable. To capture global context, we add a low-resolution volumetric branch fused with the lifted features through a lightweight integration layer, yielding a design that is both efficient and end-to-end GPU-accelerated. To validate the effectiveness of the proposed method, we conduct experiments on classification, completion, segmentation, and detection, and we map the trade-off between efficiency and accuracy across tasks. Results show that classification and completion retain or improve accuracy, while segmentation and detection trade modest drops in accuracy for significant computational savings. On-device benchmarks on an NVIDIA Jetson Orin nano confirm robust real-time throughput, demonstrating the suitability of the approach for embedded robotic perception. 

**Abstract (ZH)**: 密集三维卷积在感知方面提供了高精度，但对实时机器人系统来说计算成本过高。现有基于三平面的方法依赖于插值、点查询和隐式MLP的二维图像特征，这使得它们计算量大不适用于嵌入式三维推理。为此，我们提出了一种新的无插值三平面提升和体素融合框架，直接将三维体素投影到平面特征，并通过广播和求和重构特征体素。这将非线性转移到二维卷积中，减少了复杂性同时保持完全并行化。为了捕获全局上下文，我们通过一个轻量级整合层将低分辨率体素分支与提升特征融合，从而获得一种既高效又端到端GPU加速的设计。为了验证所提出方法的有效性，我们在分类、完成、分割和检测任务上进行了实验，并映射了效率与准确性的权衡。结果显示，分类和完成保留或提高了准确性，而分割和检测则以适度降低准确性为代价获得显著的计算节省。基于NVIDIA Jetson Orin nano的设备基准测试证实了稳健的实时吞吐量，展示了该方法适用于嵌入式机器人感知的适用性。 

---
# BEV-ODOM2: Enhanced BEV-based Monocular Visual Odometry with PV-BEV Fusion and Dense Flow Supervision for Ground Robots 

**Title (ZH)**: BEV-ODOM2：基于BEV的单目视觉里程计增强方法，结合PV-BEV融合和密集流监督，应用于地面机器人 

**Authors**: Yufei Wei, Wangtao Lu, Sha Lu, Chenxiao Hu, Fuzhang Han, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14636)  

**Abstract**: Bird's-Eye-View (BEV) representation offers a metric-scaled planar workspace, facilitating the simplification of 6-DoF ego-motion to a more robust 3-DoF model for monocular visual odometry (MVO) in intelligent transportation systems. However, existing BEV methods suffer from sparse supervision signals and information loss during perspective-to-BEV projection. We present BEV-ODOM2, an enhanced framework addressing both limitations without additional annotations. Our approach introduces: (1) dense BEV optical flow supervision constructed from 3-DoF pose ground truth for pixel-level guidance; (2) PV-BEV fusion that computes correlation volumes before projection to preserve 6-DoF motion cues while maintaining scale consistency. The framework employs three supervision levels derived solely from pose data: dense BEV flow, 5-DoF for the PV branch, and final 3-DoF output. Enhanced rotation sampling further balances diverse motion patterns in training. Extensive evaluation on KITTI, NCLT, Oxford, and our newly collected ZJH-VO multi-scale dataset demonstrates state-of-the-art performance, achieving 40 improvement in RTE compared to previous BEV methods. The ZJH-VO dataset, covering diverse ground vehicle scenarios from underground parking to outdoor plazas, is publicly available to facilitate future research. 

**Abstract (ZH)**: BEV-ODOM2：一种增强的 bird's-eye-view Odometry 框架 

---
# Toward Embodiment Equivariant Vision-Language-Action Policy 

**Title (ZH)**: 向体态不变的视觉-语言-行动策略迈进 

**Authors**: Anzhe Chen, Yifei Yang, Zhenjie Zhu, Kechun Xu, Zhongxiang Zhou, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14630)  

**Abstract**: Vision-language-action policies learn manipulation skills across tasks, environments and embodiments through large-scale pre-training. However, their ability to generalize to novel robot configurations remains limited. Most approaches emphasize model size, dataset scale and diversity while paying less attention to the design of action spaces. This leads to the configuration generalization problem, which requires costly adaptation. We address this challenge by formulating cross-embodiment pre-training as designing policies equivariant to embodiment configuration transformations. Building on this principle, we propose a framework that (i) establishes a embodiment equivariance theory for action space and policy design, (ii) introduces an action decoder that enforces configuration equivariance, and (iii) incorporates a geometry-aware network architecture to enhance embodiment-agnostic spatial reasoning. Extensive experiments in both simulation and real-world settings demonstrate that our approach improves pre-training effectiveness and enables efficient fine-tuning on novel robot embodiments. Our code is available at this https URL 

**Abstract (ZH)**: 视觉-语言-动作策略通过大规模预训练学习跨任务、跨环境和跨体配置的操纵技能，但其对新型机器人配置的泛化能力仍然有限。大多数方法强调模型规模、数据集规模和多样性，而较少关注动作空间的设计。这导致了体配置泛化问题，需要成本较高的适应。我们通过将跨体配置预训练形式化为体配置变换下策略不变性的设计来应对这一挑战。基于这一原则，我们提出了一种框架，该框架包括：(i) 建立动作空间和策略设计的体配置不变性理论，(ii) 引入一种动作解码器以确保配置不变性，以及(iii) 结合一种几何感知的网络架构以增强体配置无关的时空推理。在仿真和真实世界的广泛实验中，我们的方法证明了预训练效果的改进，并使对新型机器人体的微调更加高效。我们的代码可在以下网址获取：this https URL 

---
# Hierarchical Planning and Scheduling for Reconfigurable Multi-Robot Disassembly Systems under Structural Constraints 

**Title (ZH)**: 基于结构约束的可重构多机器人拆解系统分级规划与调度 

**Authors**: Takuya Kiyokawa, Tomoki Ishikura, Shingo Hamada, Genichiro Matsuda, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2509.14564)  

**Abstract**: This study presents a system integration approach for planning schedules, sequences, tasks, and motions for reconfigurable robots to automatically disassemble constrained structures in a non-destructive manner. Such systems must adapt their configuration and coordination to the target structure, but the large and complex search space makes them prone to local optima. To address this, we integrate multiple robot arms equipped with different types of tools, together with a rotary stage, into a reconfigurable setup. This flexible system is based on a hierarchical optimization method that generates plans meeting multiple preferred conditions under mandatory requirements within a realistic timeframe. The approach employs two many-objective genetic algorithms for sequence and task planning with motion evaluations, followed by constraint programming for scheduling. Because sequence planning has a much larger search space, we introduce a chromosome initialization method tailored to constrained structures to mitigate the risk of local optima. Simulation results demonstrate that the proposed method effectively solves complex problems in reconfigurable robotic disassembly. 

**Abstract (ZH)**: 本研究提出了一种系统集成方法，用于规划可重构机器人拆解受限结构的调度、序列、任务和动作，以非破坏性方式自动拆解。此类系统必须根据目标结构调整其配置和协调，但由于庞大的复杂搜索空间，它们容易陷入局部最优解。为此，我们将多种不同类型工具的机器人手臂与旋转平台整合到一个可重构设置中。该灵活系统基于一种层次优化方法，在实际的时间框架内生成满足多种优选条件并符合强制要求的计划。该方法使用两种多目标遗传算法进行序列和任务规划，并结合运动评估后的约束编程进行调度。由于序列规划具有更大的搜索空间，我们引入了一种针对受限结构定制的染色体初始化方法，以减轻陷入局部最优解的风险。模拟结果显示，所提出的方法有效地解决了可重构机器人拆解中的复杂问题。 

---
# SimCoachCorpus: A naturalistic dataset with language and trajectories for embodied teaching 

**Title (ZH)**: SimCoachCorpus: 一个包含语言和轨迹的自然主义数据集，用于 embodied teaching 

**Authors**: Emily Sumner, Deepak E. Gopinath, Laporsha Dees, Patricio Reyes Gomez, Xiongyi Cui, Andrew Silva, Jean Costa, Allison Morgan, Mariah Schrum, Tiffany L. Chen, Avinash Balachandran, Guy Rosman  

**Link**: [PDF](https://arxiv.org/pdf/2509.14548)  

**Abstract**: Curated datasets are essential for training and evaluating AI approaches, but are often lacking in domains where language and physical action are deeply intertwined. In particular, few datasets capture how people acquire embodied skills through verbal instruction over time. To address this gap, we introduce SimCoachCorpus: a unique dataset of race car simulator driving that allows for the investigation of rich interactive phenomena during guided and unguided motor skill acquisition. In this dataset, 29 humans were asked to drive in a simulator around a race track for approximately ninety minutes. Fifteen participants were given personalized one-on-one instruction from a professional performance driving coach, and 14 participants drove without coaching. \name\ includes embodied features such as vehicle state and inputs, map (track boundaries and raceline), and cone landmarks. These are synchronized with concurrent verbal coaching from a professional coach and additional feedback at the end of each lap. We further provide annotations of coaching categories for each concurrent feedback utterance, ratings on students' compliance with coaching advice, and self-reported cognitive load and emotional state of participants (gathered from surveys during the study). The dataset includes over 20,000 concurrent feedback utterances, over 400 terminal feedback utterances, and over 40 hours of vehicle driving data. Our naturalistic dataset can be used for investigating motor learning dynamics, exploring linguistic phenomena, and training computational models of teaching. We demonstrate applications of this dataset for in-context learning, imitation learning, and topic modeling. The dataset introduced in this work will be released publicly upon publication of the peer-reviewed version of this paper. Researchers interested in early access may register at this https URL. 

**Abstract (ZH)**: 精心策划的数据集对于训练和评估AI方法至关重要，但在语言和物理动作紧密结合的领域往往缺乏。特别是，极少有数据集捕捉到人们如何通过口头指导获得 embodiment 技能的过程。为填补这一空白，我们引入了 SimCoachCorpus：一个独特的赛车模拟器驾驶数据集，允许研究引导和非引导运动技能获得期间丰富的互动现象。在这个数据集中，29名人类参与者在大约90分钟内驾驶赛车模拟器绕赛车道一圈。15名参与者接受了专业驾驶教练的一对一面授指导，而14名参与者则没有教练指导。该数据集涵盖了车况、输入、地图（赛道边界和跑线）以及圆锥标志物等体现特征，并与专业教练的实时口头指导和每圈末尾的额外反馈同步。我们还为每条实时反馈语句提供指导类别注释，学生遵循教练建议的评分，以及参与者自我报告的认知负荷和情绪状态（来自研究期间的调查）。该数据集包含超过20,000条实时反馈语句、超过400条终端反馈语句和超过40小时的车辆驾驶数据。我们自然化的数据集可用于探究运动学习动力学、探索语言现象和训练教学计算模型。本文介绍的数据集将在经过同行评审的版本发表后公开发布。有兴趣获取早期访问的科研人员可访问此链接注册。 

---
# Dual-Arm Hierarchical Planning for Laboratory Automation: Vibratory Sieve Shaker Operations 

**Title (ZH)**: 双臂层次化规划在实验室自动化中的应用：振动筛汞操作 

**Authors**: Haoran Xiao, Xue Wang, Huimin Lu, Zhiwen Zeng, Zirui Guo, Ziqi Ni, Yicong Ye, Wei Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.14531)  

**Abstract**: This paper addresses the challenges of automating vibratory sieve shaker operations in a materials laboratory, focusing on three critical tasks: 1) dual-arm lid manipulation in 3 cm clearance spaces, 2) bimanual handover in overlapping workspaces, and 3) obstructed powder sample container delivery with orientation constraints. These tasks present significant challenges, including inefficient sampling in narrow passages, the need for smooth trajectories to prevent spillage, and suboptimal paths generated by conventional methods. To overcome these challenges, we propose a hierarchical planning framework combining Prior-Guided Path Planning and Multi-Step Trajectory Optimization. The former uses a finite Gaussian mixture model to improve sampling efficiency in narrow passages, while the latter refines paths by shortening, simplifying, imposing joint constraints, and B-spline smoothing. Experimental results demonstrate the framework's effectiveness: planning time is reduced by up to 80.4%, and waypoints are decreased by 89.4%. Furthermore, the system completes the full vibratory sieve shaker operation workflow in a physical experiment, validating its practical applicability for complex laboratory automation. 

**Abstract (ZH)**: 本文针对材料实验室中振动筛分器操作的自动化挑战，聚焦于三个关键任务：1) 在3 cm 清晰空间内的双臂盖子操作，2) 重叠工作空间内的双臂交接，3) 受限定向粉末样本容器传输。这些任务带来了显著的挑战，包括在狭窄通道中的低效取样、防止溢出所需的平滑轨迹需求以及由传统方法生成的次优路径。为克服这些挑战，我们提出了一种层次化规划框架，结合了先验引导路径规划和多步轨迹优化。前者利用有限高斯混合模型提高狭窄通道的取样效率，后者则通过缩短路径、简化路径、施加关节约束和B样条平滑来进一步优化路径。实验结果表明，该框架的有效性：规划时间减少了80.4%，路点减少了89.4%。此外，该系统在物理实验中完成了整个振动筛分器操作工作流，验证了其在复杂实验室自动化中的实际适用性。 

---
# Learning to Pick: A Visuomotor Policy for Clustered Strawberry Picking 

**Title (ZH)**: 学习选择：集群草莓采摘的感知运动策略 

**Authors**: Zhenghao Fei, Wenwu Lu, Linsheng Hou, Chen Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.14530)  

**Abstract**: Strawberries naturally grow in clusters, interwoven with leaves, stems, and other fruits, which frequently leads to occlusion. This inherent growth habit presents a significant challenge for robotic picking, as traditional percept-plan-control systems struggle to reach fruits amid the clutter. Effectively picking an occluded strawberry demands dexterous manipulation to carefully bypass or gently move the surrounding soft objects and precisely access the ideal picking point located at the stem just above the calyx. To address this challenge, we introduce a strawberry-picking robotic system that learns from human demonstrations. Our system features a 4-DoF SCARA arm paired with a human teleoperation interface for efficient data collection and leverages an End Pose Assisted Action Chunking Transformer (ACT) to develop a fine-grained visuomotor picking policy. Experiments under various occlusion scenarios demonstrate that our modified approach significantly outperforms the direct implementation of ACT, underscoring its potential for practical application in occluded strawberry picking. 

**Abstract (ZH)**: 草莓自然生长成簇，与叶片、茎和其他果实交织在一起，这常常导致遮挡。这种固有的生长习性为机器人采摘带来了重大挑战，因为传统的感知-计划-控制系统难以在杂乱的环境中达到目标果实。有效地采摘被遮挡的草莓需要灵巧的操作，以细心绕过或轻柔移动周围的软体物体，并准确地定位在花萼上方靠近茎部的理想采摘点。为应对这一挑战，我们提出了一种通过人类示范学习的草莓采摘机器人系统。该系统配备4-DoF SCARA臂，并结合了人类遥控界面以高效收集数据，并利用端位姿辅助动作片段化变换器（ACT）来开发细化的视觉-运动采摘策略。在多种遮挡场景下的实验表明，我们的改进方法显著优于直接实施ACT，突显了其在实际遮挡草莓采摘中的应用潜力。 

---
# Event-LAB: Towards Standardized Evaluation of Neuromorphic Localization Methods 

**Title (ZH)**: Event-LAB: 向标准评估神经形态定位方法迈进 

**Authors**: Adam D. Hines, Alejandro Fontan, Michael Milford, Tobias Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.14516)  

**Abstract**: Event-based localization research and datasets are a rapidly growing area of interest, with a tenfold increase in the cumulative total number of published papers on this topic over the past 10 years. Whilst the rapid expansion in the field is exciting, it brings with it an associated challenge: a growth in the variety of required code and package dependencies as well as data formats, making comparisons difficult and cumbersome for researchers to implement reliably. To address this challenge, we present Event-LAB: a new and unified framework for running several event-based localization methodologies across multiple datasets. Event-LAB is implemented using the Pixi package and dependency manager, that enables a single command-line installation and invocation for combinations of localization methods and datasets. To demonstrate the capabilities of the framework, we implement two common event-based localization pipelines: Visual Place Recognition (VPR) and Simultaneous Localization and Mapping (SLAM). We demonstrate the ability of the framework to systematically visualize and analyze the results of multiple methods and datasets, revealing key insights such as the association of parameters that control event collection counts and window sizes for frame generation to large variations in performance. The results and analysis demonstrate the importance of fairly comparing methodologies with consistent event image generation parameters. Our Event-LAB framework provides this ability for the research community, by contributing a streamlined workflow for easily setting up multiple conditions. 

**Abstract (ZH)**: 基于事件的定位研究与数据集：一种新的统一框架 

---
# Object Recognition and Force Estimation with the GelSight Baby Fin Ray 

**Title (ZH)**: 基于GelSight Baby Fin Ray的物体识别与力估计 

**Authors**: Sandra Q. Liu, Yuxiang Ma, Edward H. Adelson  

**Link**: [PDF](https://arxiv.org/pdf/2509.14510)  

**Abstract**: Recent advances in soft robotic hands and tactile sensing have enabled both to perform an increasing number of complex tasks with the aid of machine learning. In particular, we presented the GelSight Baby Fin Ray in our previous work, which integrates a camera with a soft, compliant Fin Ray structure. Camera-based tactile sensing gives the GelSight Baby Fin Ray the ability to capture rich contact information like forces, object geometries, and textures. Moreover, our previous work showed that the GelSight Baby Fin Ray can dig through clutter, and classify in-shell nuts. To further examine the potential of the GelSight Baby Fin Ray, we leverage learning to distinguish nut-in-shell textures and to perform force and position estimation. We implement ablation studies with popular neural network structures, including ResNet50, GoogLeNet, and 3- and 5-layer convolutional neural network (CNN) structures. We conclude that machine learning is a promising technique to extract useful information from high-resolution tactile images and empower soft robotics to better understand and interact with the environments. 

**Abstract (ZH)**: 近期软体手和触觉感知的进展使得两者在机器学习的辅助下能够执行越来越多的复杂任务。我们之前的工作介绍了GelSight Baby Fin Ray，该技术结合了摄像头与柔软顺应性的Fin Ray结构。基于摄像头的触觉感知使GelSight Baby Fin Ray能够捕捉丰富的接触信息，如力、物体几何形状和纹理。此外，我们之前的工作表明GelSight Baby Fin Ray能够穿透杂乱环境，并对带壳坚果进行分类。为进一步检验GelSight Baby Fin Ray的潜力，我们利用学习来区分带壳坚果的纹理，并进行力和位置估计。我们使用流行的神经网络结构，包括ResNet50、GoogLeNet以及3层和5层卷积神经网络（CNN）结构，实施了消融研究。我们得出结论，机器学习是提取高分辨率触觉图像中有用信息的一种有前途的技术，并促使软体机器人更好地理解和互动环境。 

---
# Learning Discrete Abstractions for Visual Rearrangement Tasks Using Vision-Guided Graph Coloring 

**Title (ZH)**: 使用视觉引导图着色学习离散抽象以完成视觉重组任务 

**Authors**: Abhiroop Ajith, Constantinos Chamzas  

**Link**: [PDF](https://arxiv.org/pdf/2509.14460)  

**Abstract**: Learning abstractions directly from data is a core challenge in robotics. Humans naturally operate at an abstract level, reasoning over high-level subgoals while delegating execution to low-level motor skills -- an ability that enables efficient problem solving in complex environments. In robotics, abstractions and hierarchical reasoning have long been central to planning, yet they are typically hand-engineered, demanding significant human effort and limiting scalability. Automating the discovery of useful abstractions directly from visual data would make planning frameworks more scalable and more applicable to real-world robotic domains. In this work, we focus on rearrangement tasks where the state is represented with raw images, and propose a method to induce discrete, graph-structured abstractions by combining structural constraints with an attention-guided visual distance. Our approach leverages the inherent bipartite structure of rearrangement problems, integrating structural constraints and visual embeddings into a unified framework. This enables the autonomous discovery of abstractions from vision alone, which can subsequently support high-level planning. We evaluate our method on two rearrangement tasks in simulation and show that it consistently identifies meaningful abstractions that facilitate effective planning and outperform existing approaches. 

**Abstract (ZH)**: 直接从数据中学习抽象是机器人领域的一个核心挑战。在机器人领域，抽象和层次推理长期以来一直是规划的核心内容，但它们通常需要手工设计，消耗大量的手工努力并限制了可扩展性。直接从视觉数据中自动发现有用的抽象将使规划框架更具可扩展性，并更适用于真实的机器人领域。在本文中，我们关注状态用原始图像表示的重新排列任务，提出了一种方法，通过结合结构约束和注意力引导的视觉距离来诱导离散的、图结构化的抽象。我们的方法利用重新排列问题固有的二分结构，将结构约束和视觉嵌入整合到一个统一的框架中。这使得仅从视觉数据中自主发现抽象成为可能，并可支持高层次的规划。我们在模拟中对两种重新排列任务进行了评估，并表明这种方法能够一致地识别出有意义的抽象，促进有效的规划，并优于现有方法。 

---
# Online Learning of Deceptive Policies under Intermittent Observation 

**Title (ZH)**: 基于间歇性观察的欺骗性策略的在线学习 

**Authors**: Gokul Puthumanaillam, Ram Padmanabhan, Jose Fuentes, Nicole Cruz, Paulo Padrao, Ruben Hernandez, Hao Jiang, William Schafer, Leonardo Bobadilla, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2509.14453)  

**Abstract**: In supervisory control settings, autonomous systems are not monitored continuously. Instead, monitoring often occurs at sporadic intervals within known bounds. We study the problem of deception, where an agent pursues a private objective while remaining plausibly compliant with a supervisor's reference policy when observations occur. Motivated by the behavior of real, human supervisors, we situate the problem within Theory of Mind: the representation of what an observer believes and expects to see. We show that Theory of Mind can be repurposed to steer online reinforcement learning (RL) toward such deceptive behavior. We model the supervisor's expectations and distill from them a single, calibrated scalar -- the expected evidence of deviation if an observation were to happen now. This scalar combines how unlike the reference and current action distributions appear, with the agent's belief that an observation is imminent. Injected as a state-dependent weight into a KL-regularized policy improvement step within an online RL loop, this scalar informs a closed-form update that smoothly trades off self-interest and compliance, thus sidestepping hand-crafted or heuristic policies. In real-world, real-time hardware experiments on marine (ASV) and aerial (UAV) navigation, our ToM-guided RL runs online, achieves high return and success with observed-trace evidence calibrated to the supervisor's expectations. 

**Abstract (ZH)**: 基于监控控制设置中的欺骗行为：理论心智引导的在线强化学习研究 

---
# Local-Canonicalization Equivariant Graph Neural Networks for Sample-Efficient and Generalizable Swarm Robot Control 

**Title (ZH)**: 局部规范化等变图神经网络在样本效率和泛化能力方面的群机器人控制 

**Authors**: Keqin Wang, Tao Zhong, David Chang, Christine Allen-Blanchette  

**Link**: [PDF](https://arxiv.org/pdf/2509.14431)  

**Abstract**: Multi-agent reinforcement learning (MARL) has emerged as a powerful paradigm for coordinating swarms of agents in complex decision-making, yet major challenges remain. In competitive settings such as pursuer-evader tasks, simultaneous adaptation can destabilize training; non-kinetic countermeasures often fail under adverse conditions; and policies trained in one configuration rarely generalize to environments with a different number of agents. To address these issues, we propose the Local-Canonicalization Equivariant Graph Neural Networks (LEGO) framework, which integrates seamlessly with popular MARL algorithms such as MAPPO. LEGO employs graph neural networks to capture permutation equivariance and generalization to different agent numbers, canonicalization to enforce E(n)-equivariance, and heterogeneous representations to encode role-specific inductive biases. Experiments on cooperative and competitive swarm benchmarks show that LEGO outperforms strong baselines and improves generalization. In real-world experiments, LEGO demonstrates robustness to varying team sizes and agent failure. 

**Abstract (ZH)**: 局部规范等变图神经网络（LEGO）框架：用于多智能体强化学习的协同与竞争集群基准研究 

---
# Perception-Integrated Safety Critical Control via Analytic Collision Cone Barrier Functions on 3D Gaussian Splatting 

**Title (ZH)**: 基于分析碰撞圆锥障碍函数的三维高斯点云集成感知安全关键控制 

**Authors**: Dario Tscholl, Yashwanth Nakka, Brian Gunter  

**Link**: [PDF](https://arxiv.org/pdf/2509.14421)  

**Abstract**: We present a perception-driven safety filter that converts each 3D Gaussian Splat (3DGS) into a closed-form forward collision cone, which in turn yields a first-order control barrier function (CBF) embedded within a quadratic program (QP). By exploiting the analytic geometry of splats, our formulation provides a continuous, closed-form representation of collision constraints that is both simple and computationally efficient. Unlike distance-based CBFs, which tend to activate reactively only when an obstacle is already close, our collision-cone CBF activates proactively, allowing the robot to adjust earlier and thereby produce smoother and safer avoidance maneuvers at lower computational cost. We validate the method on a large synthetic scene with approximately 170k splats, where our filter reduces planning time by a factor of 3 and significantly decreased trajectory jerk compared to a state-of-the-art 3DGS planner, while maintaining the same level of safety. The approach is entirely analytic, requires no high-order CBF extensions (HOCBFs), and generalizes naturally to robots with physical extent through a principled Minkowski-sum inflation of the splats. These properties make the method broadly applicable to real-time navigation in cluttered, perception-derived extreme environments, including space robotics and satellite systems. 

**Abstract (ZH)**: 基于感知的安全滤波器：从3D高斯点转换到闭式前方碰撞圆锥，并嵌入二次规划的零阶控制屏障函数 

---
# GestOS: Advanced Hand Gesture Interpretation via Large Language Models to control Any Type of Robot 

**Title (ZH)**: GestOS: 通过大型语言模型实现对任何类型机器人的高级手势解读 

**Authors**: Artem Lykov, Oleg Kobzarev, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2509.14412)  

**Abstract**: We present GestOS, a gesture-based operating system for high-level control of heterogeneous robot teams. Unlike prior systems that map gestures to fixed commands or single-agent actions, GestOS interprets hand gestures semantically and dynamically distributes tasks across multiple robots based on their capabilities, current state, and supported instruction sets. The system combines lightweight visual perception with large language model (LLM) reasoning: hand poses are converted into structured textual descriptions, which the LLM uses to infer intent and generate robot-specific commands. A robot selection module ensures that each gesture-triggered task is matched to the most suitable agent in real time. This architecture enables context-aware, adaptive control without requiring explicit user specification of targets or commands. By advancing gesture interaction from recognition to intelligent orchestration, GestOS supports scalable, flexible, and user-friendly collaboration with robotic systems in dynamic environments. 

**Abstract (ZH)**: 基于手势的操作系统：面向异构机器人团队的高层次控制 

---
# RLBind: Adversarial-Invariant Cross-Modal Alignment for Unified Robust Embeddings 

**Title (ZH)**: RLBind: 抗对抗性扰动的跨模态对齐以获得统一的鲁棒嵌入 

**Authors**: Yuhong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14383)  

**Abstract**: Unified multi-modal encoders that bind vision, audio, and other sensors into a shared embedding space are attractive building blocks for robot perception and decision-making. However, on-robot deployment exposes the vision branch to adversarial and natural corruptions, making robustness a prerequisite for safety. Prior defenses typically align clean and adversarial features within CLIP-style encoders and overlook broader cross-modal correspondence, yielding modest gains and often degrading zero-shot transfer. We introduce RLBind, a two-stage adversarial-invariant cross-modal alignment framework for robust unified embeddings. Stage 1 performs unsupervised fine-tuning on clean-adversarial pairs to harden the visual encoder. Stage 2 leverages cross-modal correspondence by minimizing the discrepancy between clean/adversarial features and a text anchor, while enforcing class-wise distributional alignment across modalities. Extensive experiments on Image, Audio, Thermal, and Video data show that RLBind consistently outperforms the LanguageBind backbone and standard fine-tuning baselines in both clean accuracy and norm-bounded adversarial robustness. By improving resilience without sacrificing generalization, RLBind provides a practical path toward safer multi-sensor perception stacks for embodied robots in navigation, manipulation, and other autonomy settings. 

**Abstract (ZH)**: 统一多模态编码器将视觉、音频及其他传感器整合至共享嵌入空间中，是机器人感知与决策的重要构建块。然而，机器人上的部署使得视觉分支暴露于对抗性和自然性破坏之下，因此鲁棒性成为确保安全的先决条件。先前的防御措施通常在CLIP风格的编码器中对齐干净和对抗性特征，忽略了更广泛的跨模态对应关系，导致进步有限且常降低零样本迁移性能。我们提出了一种两阶段鲁棒不变跨模态对齐框架RLBind，以增强统一嵌入的鲁棒性。第一阶段通过无监督微调干净对抗性配对来强化视觉编码器。第二阶段利用跨模态对应关系，通过最小化干净/对抗性特征与文本锚点之间的差异，同时确保各模态类别间的分布对齐。在图像、音频、热成像和视频数据上的广泛实验表明，RLBind在干净准确性和范数限制下的对抗鲁棒性方面均优于LanguageBind主干及标准微调基准。通过提升鲁棒性而不牺牲泛化能力，RLBind提供了一条实现导航、操作及其他自主性设置中多传感器感知堆栈更安全的实用路径。 

---
# CRAFT: Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks 

**Title (ZH)**: CRAFT：使用基础模型自主指导多机器人协调任务的强化学习教练 

**Authors**: Seoyeon Choi, Kanghyun Ryu, Jonghoon Ock, Negar Mehr  

**Link**: [PDF](https://arxiv.org/pdf/2509.14380)  

**Abstract**: Multi-Agent Reinforcement Learning (MARL) provides a powerful framework for learning coordination in multi-agent systems. However, applying MARL to robotics still remains challenging due to high-dimensional continuous joint action spaces, complex reward design, and non-stationary transitions inherent to decentralized settings. On the other hand, humans learn complex coordination through staged curricula, where long-horizon behaviors are progressively built upon simpler skills. Motivated by this, we propose CRAFT: Coaching Reinforcement learning Autonomously using Foundation models for multi-robot coordination Tasks, a framework that leverages the reasoning capabilities of foundation models to act as a "coach" for multi-robot coordination. CRAFT automatically decomposes long-horizon coordination tasks into sequences of subtasks using the planning capability of Large Language Models (LLMs). In what follows, CRAFT trains each subtask using reward functions generated by LLM, and refines them through a Vision Language Model (VLM)-guided reward-refinement loop. We evaluate CRAFT on multi-quadruped navigation and bimanual manipulation tasks, demonstrating its capability to learn complex coordination behaviors. In addition, we validate the multi-quadruped navigation policy in real hardware experiments. 

**Abstract (ZH)**: 基于基础模型的多机器人协调强化学习教学框架：CRAFT 

---
# DreamControl: Human-Inspired Whole-Body Humanoid Control for Scene Interaction via Guided Diffusion 

**Title (ZH)**: DreamControl: 以人为本的全身类人机器人场景交互控制方法基于引导扩散 

**Authors**: Dvij Kalaria, Sudarshan S Harithas, Pushkal Katara, Sangkyung Kwak, Sarthak Bhagat, Shankar Sastry, Srinath Sridhar, Sai Vemprala, Ashish Kapoor, Jonathan Chung-Kuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.14353)  

**Abstract**: We introduce DreamControl, a novel methodology for learning autonomous whole-body humanoid skills. DreamControl leverages the strengths of diffusion models and Reinforcement Learning (RL): our core innovation is the use of a diffusion prior trained on human motion data, which subsequently guides an RL policy in simulation to complete specific tasks of interest (e.g., opening a drawer or picking up an object). We demonstrate that this human motion-informed prior allows RL to discover solutions unattainable by direct RL, and that diffusion models inherently promote natural looking motions, aiding in sim-to-real transfer. We validate DreamControl's effectiveness on a Unitree G1 robot across a diverse set of challenging tasks involving simultaneous lower and upper body control and object interaction. 

**Abstract (ZH)**: DreamControl: 一种基于扩散模型的自主全身类人技能学习新方法 

---
# LeVR: A Modular VR Teleoperation Framework for Imitation Learning in Dexterous Manipulation 

**Title (ZH)**: LeVR: 一种用于灵巧操作模仿学习的模块化VR远程操作框架 

**Authors**: Zhengyang Kris Weng, Matthew L. Elwin, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14349)  

**Abstract**: We introduce LeVR, a modular software framework designed to bridge two critical gaps in robotic imitation learning. First, it provides robust and intuitive virtual reality (VR) teleoperation for data collection using robot arms paired with dexterous hands, addressing a common limitation in existing systems. Second, it natively integrates with the powerful LeRobot imitation learning (IL) framework, enabling the use of VR-based teleoperation data and streamlining the demonstration collection process. To demonstrate LeVR, we release LeFranX, an open-source implementation for the Franka FER arm and RobotEra XHand, two widely used research platforms. LeFranX delivers a seamless, end-to-end workflow from data collection to real-world policy deployment. We validate our system by collecting a public dataset of 100 expert demonstrations and use it to successfully fine-tune state-of-the-art visuomotor policies. We provide our open-source framework, implementation, and dataset to accelerate IL research for the robotics community. 

**Abstract (ZH)**: 我们引入了LeVR，这是一种模块化的软件框架，旨在弥合机器人模仿学习中两个关键差距。首先，它提供了针对配備灵巧手的机器人臂进行数据收集的稳健且直观的虚拟现实（VR）远程操作，解决了现有系统中常见的局限性。其次，它原生集成了强大的LeRobot模仿学习（IL）框架，使得可以利用基于VR的远程操作数据，并简化演示收集过程。为了展示LeVR，我们发布了LeFranX，这是一种开源实现，适用于Franka FER臂和RobotEra XHand，二者是广泛使用的研究平台。LeFranX实现了从数据收集到实际应用策略部署的无缝端到端工作流。我们通过收集包含100个专家演示的公开数据集来验证我们的系统，并使用此数据集成功地微调了最先进的视觉-运动策略。我们提供开源框架、实现和数据集，以加速机器人社区的模仿学习研究。 

---
# Multi-Quadruped Cooperative Object Transport: Learning Decentralized Pinch-Lift-Move 

**Title (ZH)**: 多 leg 合作物体运输：学习去中心化 pinch-lift-move 

**Authors**: Bikram Pandit, Aayam Kumar Shrestha, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2509.14342)  

**Abstract**: We study decentralized cooperative transport using teams of N-quadruped robots with arm that must pinch, lift, and move ungraspable objects through physical contact alone. Unlike prior work that relies on rigid mechanical coupling between robots and objects, we address the more challenging setting where mechanically independent robots must coordinate through contact forces alone without any communication or centralized control. To this end, we employ a hierarchical policy architecture that separates base locomotion from arm control, and propose a constellation reward formulation that unifies position and orientation tracking to enforce rigid contact behavior. The key insight is encouraging robots to behave as if rigidly connected to the object through careful reward design and training curriculum rather than explicit mechanical constraints. Our approach enables coordination through shared policy parameters and implicit synchronization cues - scaling to arbitrary team sizes without retraining. We show extensive simulation experiments to demonstrate robust transport across 2-10 robots on diverse object geometries and masses, along with sim2real transfer results on lightweight objects. 

**Abstract (ZH)**: 基于物理接触的去中心化协同运输：N足机器人手臂抓取和移动不可抓取物体的研究 

---
# FlowDrive: Energy Flow Field for End-to-End Autonomous Driving 

**Title (ZH)**: FlowDrive：端到端自主驾驶的能量流场 

**Authors**: Hao Jiang, Zhipeng Zhang, Yu Gao, Zhigang Sun, Yiru Wang, Yuwen Heng, Shuo Wang, Jinhao Chai, Zhuo Chen, Hao Zhao, Hao Sun, Xi Zhang, Anqing Jiang, Chuan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.14303)  

**Abstract**: Recent advances in end-to-end autonomous driving leverage multi-view images to construct BEV representations for motion planning. In motion planning, autonomous vehicles need considering both hard constraints imposed by geometrically occupied obstacles (e.g., vehicles, pedestrians) and soft, rule-based semantics with no explicit geometry (e.g., lane boundaries, traffic priors). However, existing end-to-end frameworks typically rely on BEV features learned in an implicit manner, lacking explicit modeling of risk and guidance priors for safe and interpretable planning. To address this, we propose FlowDrive, a novel framework that introduces physically interpretable energy-based flow fields-including risk potential and lane attraction fields-to encode semantic priors and safety cues into the BEV space. These flow-aware features enable adaptive refinement of anchor trajectories and serve as interpretable guidance for trajectory generation. Moreover, FlowDrive decouples motion intent prediction from trajectory denoising via a conditional diffusion planner with feature-level gating, alleviating task interference and enhancing multimodal diversity. Experiments on the NAVSIM v2 benchmark demonstrate that FlowDrive achieves state-of-the-art performance with an EPDMS of 86.3, surpassing prior baselines in both safety and planning quality. The project is available at this https URL. 

**Abstract (ZH)**: 端到端自动驾驶 Recent 进展：利用多视图图像构建BEV表示以进行运动规划 

---
# AEGIS: Automated Error Generation and Identification for Multi-Agent Systems 

**Title (ZH)**: AEGIS：多智能体系统中的自动错误生成与识别 

**Authors**: Fanqi Kong, Ruijie Zhang, Huaxiao Yin, Guibin Zhang, Xiaofei Zhang, Ziang Chen, Zhaowei Zhang, Xiaoyuan Zhang, Song-Chun Zhu, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.14295)  

**Abstract**: As Multi-Agent Systems (MAS) become increasingly autonomous and complex, understanding their error modes is critical for ensuring their reliability and safety. However, research in this area has been severely hampered by the lack of large-scale, diverse datasets with precise, ground-truth error labels. To address this bottleneck, we introduce \textbf{AEGIS}, a novel framework for \textbf{A}utomated \textbf{E}rror \textbf{G}eneration and \textbf{I}dentification for Multi-Agent \textbf{S}ystems. By systematically injecting controllable and traceable errors into initially successful trajectories, we create a rich dataset of realistic failures. This is achieved using a context-aware, LLM-based adaptive manipulator that performs sophisticated attacks like prompt injection and response corruption to induce specific, predefined error modes. We demonstrate the value of our dataset by exploring three distinct learning paradigms for the error identification task: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. Our comprehensive experiments show that models trained on AEGIS data achieve substantial improvements across all three learning paradigms. Notably, several of our fine-tuned models demonstrate performance competitive with or superior to proprietary systems an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at this https URL. 

**Abstract (ZH)**: 自动化多agent系统错误生成与识别框架AEGIS 

---
# Out-of-Sight Trajectories: Tracking, Fusion, and Prediction 

**Title (ZH)**: 隐藏轨迹：跟踪、融合与预测 

**Authors**: Haichao Zhang, Yi Xu, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.15219)  

**Abstract**: Trajectory prediction is a critical task in computer vision and autonomous systems, playing a key role in autonomous driving, robotics, surveillance, and virtual reality. Existing methods often rely on complete and noise-free observational data, overlooking the challenges associated with out-of-sight objects and the inherent noise in sensor data caused by limited camera coverage, obstructions, and the absence of ground truth for denoised trajectories. These limitations pose safety risks and hinder reliable prediction in real-world scenarios. In this extended work, we present advancements in Out-of-Sight Trajectory (OST), a novel task that predicts the noise-free visual trajectories of out-of-sight objects using noisy sensor data. Building on our previous research, we broaden the scope of Out-of-Sight Trajectory Prediction (OOSTraj) to include pedestrians and vehicles, extending its applicability to autonomous driving, robotics, surveillance, and virtual reality. Our enhanced Vision-Positioning Denoising Module leverages camera calibration to establish a vision-positioning mapping, addressing the lack of visual references, while effectively denoising noisy sensor data in an unsupervised manner. Through extensive evaluations on the Vi-Fi and JRDB datasets, our approach achieves state-of-the-art performance in both trajectory denoising and prediction, significantly surpassing previous baselines. Additionally, we introduce comparisons with traditional denoising methods, such as Kalman filtering, and adapt recent trajectory prediction models to our task, providing a comprehensive benchmark. This work represents the first initiative to integrate vision-positioning projection for denoising noisy sensor trajectories of out-of-sight agents, paving the way for future advances. The code and preprocessed datasets are available at this http URL 

**Abstract (ZH)**: 视觉不可见轨迹预测：一种利用噪声音频数据预测视觉不可见对象的噪声自由视觉轨迹的新任务 

---
# RynnVLA-001: Using Human Demonstrations to Improve Robot Manipulation 

**Title (ZH)**: RynnVLA-001: 利用人类演示提高机器人操作能力 

**Authors**: Yuming Jiang, Siteng Huang, Shengke Xue, Yaxi Zhao, Jun Cen, Sicong Leng, Kehan Li, Jiayan Guo, Kexiang Wang, Mingxiu Chen, Fan Wang, Deli Zhao, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.15212)  

**Abstract**: This paper presents RynnVLA-001, a vision-language-action(VLA) model built upon large-scale video generative pretraining from human demonstrations. We propose a novel two-stage pretraining methodology. The first stage, Ego-Centric Video Generative Pretraining, trains an Image-to-Video model on 12M ego-centric manipulation videos to predict future frames conditioned on an initial frame and a language instruction. The second stage, Human-Centric Trajectory-Aware Modeling, extends this by jointly predicting future keypoint trajectories, thereby effectively bridging visual frame prediction with action prediction. Furthermore, to enhance action representation, we propose ActionVAE, a variational autoencoder that compresses sequences of actions into compact latent embeddings, reducing the complexity of the VLA output space. When finetuned on the same downstream robotics datasets, RynnVLA-001 achieves superior performance over state-of-the-art baselines, demonstrating that the proposed pretraining strategy provides a more effective initialization for VLA models. 

**Abstract (ZH)**: RynnVLA-001：一种基于大规模人类演示视频生成预训练的视觉-语言-行动模型 

---
# Self-Improving Embodied Foundation Models 

**Title (ZH)**: 自我提升的体感基础模型 

**Authors**: Seyed Kamyar Seyed Ghasemipour, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, Igor Mordatch  

**Link**: [PDF](https://arxiv.org/pdf/2509.15155)  

**Abstract**: Foundation models trained on web-scale data have revolutionized robotics, but their application to low-level control remains largely limited to behavioral cloning. Drawing inspiration from the success of the reinforcement learning stage in fine-tuning large language models, we propose a two-stage post-training approach for robotics. The first stage, Supervised Fine-Tuning (SFT), fine-tunes pretrained foundation models using both: a) behavioral cloning, and b) steps-to-go prediction objectives. In the second stage, Self-Improvement, steps-to-go prediction enables the extraction of a well-shaped reward function and a robust success detector, enabling a fleet of robots to autonomously practice downstream tasks with minimal human supervision. Through extensive experiments on real-world and simulated robot embodiments, our novel post-training recipe unveils significant results on Embodied Foundation Models. First, we demonstrate that the combination of SFT and Self-Improvement is significantly more sample-efficient than scaling imitation data collection for supervised learning, and that it leads to policies with significantly higher success rates. Further ablations highlight that the combination of web-scale pretraining and Self-Improvement is the key to this sample-efficiency. Next, we demonstrate that our proposed combination uniquely unlocks a capability that current methods cannot achieve: autonomously practicing and acquiring novel skills that generalize far beyond the behaviors observed in the imitation learning datasets used during training. These findings highlight the transformative potential of combining pretrained foundation models with online Self-Improvement to enable autonomous skill acquisition in robotics. Our project website can be found at this https URL . 

**Abstract (ZH)**: 基于Web规模数据训练的基础模型已经革命了机器人技术，但其在低级控制中的应用主要局限于行为克隆。受到强化学习在调优大型语言模型中的成功启发，我们提出了一种两阶段后训练方法用于机器人技术。第一阶段，监督细调（SFT），使用行为克隆和剩余步骤预测目标对预训练基础模型进行调优。第二阶段，自我提升，通过剩余步骤预测提取良好的奖励函数和稳健的成功检测器，使机器人能够最少的人类监督下自主练习下游任务。通过在真实和模拟机器人实例上的广泛实验，我们新的后训练方案揭示了嵌入式基础模型的重大成果。首先，我们展示了SFT和自我提升的结合在样本效率方面显著优于模仿数据收集上的扩展，并且这导致了成功率更高的策略。进一步的消融实验表明，大规模预训练和自我提升的结合是这种样本效率的关键。其次，我们展示了我们提出的方法唯一解锁了一种当前方法无法实现的能力：自主练习和获取在训练期间模仿学习数据集中观察到的行为之外能够泛化的新型技能。这些发现突显了将预训练基础模型与在线自我提升结合以在机器人中实现自主技能获取的变革潜力。我们的项目网站可访问：this https URL。 

---
# Nonlinear Cooperative Salvo Guidance with Seeker-Limited Interceptors 

**Title (ZH)**: 有限制 Seeker 的非线性协同齐射制导 

**Authors**: Lohitvel Gopikannan, Shashi Ranjan Kumar, Abhinav Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2509.15136)  

**Abstract**: This paper presents a cooperative guidance strategy for the simultaneous interception of a constant-velocity, non-maneuvering target, addressing the realistic scenario where only a subset of interceptors are equipped with onboard seekers. To overcome the resulting heterogeneity in target observability, a fixed-time distributed observer is employed, enabling seeker-less interceptors to estimate the target state using information from seeker-equipped agents and local neighbors over a directed communication topology. Departing from conventional strategies that approximate time-to-go via linearization or small-angle assumptions, the proposed approach leverages deviated pursuit guidance where the time-to-go expression is exact for such a target. Moreover, a higher-order sliding mode consensus protocol is utilized to establish time-to-go consensus within a finite time. The effectiveness of the proposed guidance and estimation architecture is demonstrated through simulations. 

**Abstract (ZH)**: 一种针对非机动目标的分布式观测导引策略 

---
# RoboEye: Enhancing 2D Robotic Object Identification with Selective 3D Geometric Keypoint Matching 

**Title (ZH)**: RoboEye: 通过选择性3D几何关键点匹配增强二维机器人物体识别 

**Authors**: Xingwu Zhang, Guanxuan Li, Zhuocheng Zhang, Zijun Long  

**Link**: [PDF](https://arxiv.org/pdf/2509.14966)  

**Abstract**: The rapidly growing number of product categories in large-scale e-commerce makes accurate object identification for automated packing in warehouses substantially more difficult. As the catalog grows, intra-class variability and a long tail of rare or visually similar items increase, and when combined with diverse packaging, cluttered containers, frequent occlusion, and large viewpoint changes-these factors amplify discrepancies between query and reference images, causing sharp performance drops for methods that rely solely on 2D appearance features. Thus, we propose RoboEye, a two-stage identification framework that dynamically augments 2D semantic features with domain-adapted 3D reasoning and lightweight adapters to bridge training deployment gaps. In the first stage, we train a large vision model to extract 2D features for generating candidate rankings. A lightweight 3D-feature-awareness module then estimates 3D feature quality and predicts whether 3D re-ranking is necessary, preventing performance degradation and avoiding unnecessary computation. When invoked, the second stage uses our robot 3D retrieval transformer, comprising a 3D feature extractor that produces geometry-aware dense features and a keypoint-based matcher that computes keypoint-correspondence confidences between query and reference images instead of conventional cosine-similarity scoring. Experiments show that RoboEye improves Recall@1 by 7.1% over the prior state of the art (RoboLLM). Moreover, RoboEye operates using only RGB images, avoiding reliance on explicit 3D inputs and reducing deployment costs. The code used in this paper is publicly available at: this https URL. 

**Abstract (ZH)**: 大规模电子商务中产品类别迅速增长使得仓库自动化打包中的准确对象识别变得更加困难。RoboEye：一种两阶段识别框架，通过动态增强2D语义特征并与领域适应的3D推理相结合，以减轻训练部署差距。 

---
# A Real-Time Multi-Model Parametric Representation of Point Clouds 

**Title (ZH)**: 实时多模型参数化表示点云 

**Authors**: Yuan Gao, Wei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.14773)  

**Abstract**: In recent years, parametric representations of point clouds have been widely applied in tasks such as memory-efficient mapping and multi-robot collaboration. Highly adaptive models, like spline surfaces or quadrics, are computationally expensive in detection or fitting. In contrast, real-time methods, such as Gaussian mixture models or planes, have low degrees of freedom, making high accuracy with few primitives difficult. To tackle this problem, a multi-model parametric representation with real-time surface detection and fitting is proposed. Specifically, the Gaussian mixture model is first employed to segment the point cloud into multiple clusters. Then, flat clusters are selected and merged into planes or curved surfaces. Planes can be easily fitted and delimited by a 2D voxel-based boundary description method. Surfaces with curvature are fitted by B-spline surfaces and the same boundary description method is employed. Through evaluations on multiple public datasets, the proposed surface detection exhibits greater robustness than the state-of-the-art approach, with 3.78 times improvement in efficiency. Meanwhile, this representation achieves a 2-fold gain in accuracy over Gaussian mixture models, operating at 36.4 fps on a low-power onboard computer. 

**Abstract (ZH)**: 近年来，点云的参量表示在高效记忆映射和多机器人协作等任务中得到了广泛应用。高度适应的模型，如样条曲面或二次曲面，在检测或拟合时计算成本较高。相比之下，实时方法，如高斯混合模型或平面，自由度较低，使得在少于基元的情况下获得高精度变得困难。为了解决这一问题，提出了一种具有实时表面检测和拟合的多模型参量表示。具体而言，首先使用高斯混合模型对点云进行分割，生成多个簇。然后选择平坦的簇并将其合并为平面或曲面。平面可以通过基于2D体素的边界描述方法轻松拟合和限定。曲面使用B- spline曲面进行拟合，并采用相同的边界描述方法。在多个公开数据集上的评估表明，所提出的表面检测方法比现有的前沿方法具有更高的鲁棒性，效率提升3.78倍。同时，这种方法在准确率上比高斯混合模型提升了两倍，可在低功耗的车载计算机上以每秒36.4帧的速度运行。 

---
