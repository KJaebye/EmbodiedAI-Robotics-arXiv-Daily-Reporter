# Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation 

**Title (ZH)**: 通用机器人策略中的捷径学习：数据集多样性和碎片化的作用 

**Authors**: Youguang Xing, Xu Luo, Junlin Xie, Lianli Gao, Hengtao Shen, Jingkuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.06426)  

**Abstract**: Generalist robot policies trained on large-scale datasets such as Open X-Embodiment (OXE) demonstrate strong performance across a wide range of tasks. However, they often struggle to generalize beyond the distribution of their training data. In this paper, we investigate the underlying cause of this limited generalization capability. We identify shortcut learning -- the reliance on task-irrelevant features -- as a key impediment to generalization. Through comprehensive theoretical and empirical analysis, we uncover two primary contributors to shortcut learning: (1) limited diversity within individual sub-datasets, and (2) significant distributional disparities across sub-datasets, leading to dataset fragmentation. These issues arise from the inherent structure of large-scale datasets like OXE, which are typically composed of multiple sub-datasets collected independently across varied environments and embodiments. Our findings provide critical insights into dataset collection strategies that can reduce shortcut learning and enhance the generalization ability of generalist robot policies. Moreover, in scenarios where acquiring new large-scale data is impractical, we demonstrate that carefully selected robotic data augmentation strategies can effectively reduce shortcut learning in existing offline datasets, thereby improving generalization capabilities of generalist robot policies, e.g., $\pi_0$, in both simulation and real-world environments. More information at this https URL. 

**Abstract (ZH)**: 大型数据集如Open X-Embodiment (OXE)训练的通用机器人策略在广泛的任务中表现出强大的性能，但往往难以超越训练数据分布进行泛化。本文探讨了这种有限泛化能力的根本原因。我们确定了捷径学习——依赖于与任务无关的特征——是泛化的主要障碍之一。通过全面的理论和实证分析，我们发现捷径学习的两个主要成因是：（1）单个子数据集内的多样性有限，（2）子数据集间显著的分布差异导致数据集碎片化。这些问题源于大型数据集如OXE固有的结构，这些数据集通常是由多个独立收集的子数据集组成，这些子数据集跨越了不同的环境和机器人实体。我们的研究结果为减少捷径学习和提高通用机器人策略泛化能力的数据集收集策略提供了关键见解。此外，在获得新的大规模数据不现实的情况下，我们展示了精心选择的机器人数据增强策略可以有效减少现有离线数据集中的捷径学习，从而提升通用机器人策略的泛化能力，例如$\pi_0$，在仿真和实际环境中的表现。更多详情请见此链接。 

---
# V*: An Efficient Motion Planning Algorithm for Autonomous Vehicles 

**Title (ZH)**: V*: 自动驾驶车辆高效路径规划算法 

**Authors**: Abdullah Zareh Andaryan, Michael G.H. Bell, Mohsen Ramezani, Glenn Geers  

**Link**: [PDF](https://arxiv.org/pdf/2508.06404)  

**Abstract**: Autonomous vehicle navigation in structured environments requires planners capable of generating time-optimal, collision-free trajectories that satisfy dynamic and kinematic constraints. We introduce V*, a graph-based motion planner that represents speed and direction as explicit state variables within a discretised space-time-velocity lattice. Unlike traditional methods that decouple spatial search from dynamic feasibility or rely on post-hoc smoothing, V* integrates both motion dimensions directly into graph construction through dynamic graph generation during search expansion. To manage the complexity of high-dimensional search, we employ a hexagonal discretisation strategy and provide formal mathematical proofs establishing optimal waypoint spacing and minimal node redundancy under constrained heading transitions for velocity-aware motion planning. We develop a mathematical formulation for transient steering dynamics in the kinematic bicycle model, modelling steering angle convergence with exponential behaviour, and deriving the relationship for convergence rate parameters. This theoretical foundation, combined with geometric pruning strategies that eliminate expansions leading to infeasible steering configurations, enables V* to evaluate dynamically admissible manoeuvres, ensuring each trajectory is physically realisable without further refinement. We further demonstrate V*'s performance in simulation studies with cluttered and dynamic environments involving moving obstacles, showing its ability to avoid conflicts, yield proactively, and generate safe, efficient trajectories with temporal reasoning capabilities for waiting behaviours and dynamic coordination. 

**Abstract (ZH)**: 自主车辆在结构化环境中的导航需要能够生成时间最优、无碰撞轨迹的规划器，这些轨迹同时满足动力学和运动学约束。我们引入了一种基于图的运动规划器V*，它在离散化的时空速度晶格中将速度和方向表示为显式的状态变量。与传统方法将空间搜索与动态可行性分离或依赖于事后平滑的方法不同，V*在搜索扩展过程中通过动态图生成直接将运动维度整合到图构建中。为了管理高维搜索的复杂性，我们采用六边形离散化策略，并提供了正式的数学证明，建立了速度感知运动规划中最优航点间距和最小节点冗余性，特别是在方向转换受限制的情况下。我们为刚性自行车模型中的瞬态转向动力学开发了一个数学公式，模型转向角收敛表现出指数行为，并推导出收敛速率参数的关系。理论基础与几何修剪策略相结合，可以消除导致不可行转向配置的扩展，从而使V*能够评估动态可行的操作，确保每条轨迹在无需进一步细化的情况下都能实现。此外，我们在包含移动障碍物的拥挤和动态环境中进行的模拟研究进一步展示了V*的能力，能够避免冲突、主动让行，并生成具有时间推理能力的等待行为和动态协调的高效、安全轨迹。 

---
# L2Calib: $SE(3)$-Manifold Reinforcement Learning for Robust Extrinsic Calibration with Degenerate Motion Resilience 

**Title (ZH)**: L2Calib: $SE(3)$流形强化学习及其在退化运动鲁棒外在标定中的应用 

**Authors**: Baorun Li, Chengrui Zhu, Siyi Du, Bingran Chen, Jie Ren, Wenfei Wang, Yong Liu, Jiajun Lv  

**Link**: [PDF](https://arxiv.org/pdf/2508.06330)  

**Abstract**: Extrinsic calibration is essential for multi-sensor fusion, existing methods rely on structured targets or fully-excited data, limiting real-world applicability. Online calibration further suffers from weak excitation, leading to unreliable estimates. To address these limitations, we propose a reinforcement learning (RL)-based extrinsic calibration framework that formulates extrinsic calibration as a decision-making problem, directly optimizes $SE(3)$ extrinsics to enhance odometry accuracy. Our approach leverages a probabilistic Bingham distribution to model 3D rotations, ensuring stable optimization while inherently retaining quaternion symmetry. A trajectory alignment reward mechanism enables robust calibration without structured targets by quantitatively evaluating estimated tightly-coupled trajectory against a reference trajectory. Additionally, an automated data selection module filters uninformative samples, significantly improving efficiency and scalability for large-scale datasets. Extensive experiments on UAVs, UGVs, and handheld platforms demonstrate that our method outperforms traditional optimization-based approaches, achieving high-precision calibration even under weak excitation conditions. Our framework simplifies deployment on diverse robotic platforms by eliminating the need for high-quality initial extrinsics and enabling calibration from routine operating data. The code is available at this https URL. 

**Abstract (ZH)**: 基于强化学习的外参标定框架：解决多传感器融合中的外参标定问题 

---
# Towards Balanced Behavior Cloning from Imbalanced Datasets 

**Title (ZH)**: 不平衡数据集上的平衡行为克隆研究 

**Authors**: Sagar Parekh, Heramb Nemlekar, Dylan P. Losey  

**Link**: [PDF](https://arxiv.org/pdf/2508.06319)  

**Abstract**: Robots should be able to learn complex behaviors from human demonstrations. In practice, these human-provided datasets are inevitably imbalanced: i.e., the human demonstrates some subtasks more frequently than others. State-of-the-art methods default to treating each element of the human's dataset as equally important. So if -- for instance -- the majority of the human's data focuses on reaching a goal, and only a few state-action pairs move to avoid an obstacle, the learning algorithm will place greater emphasis on goal reaching. More generally, misalignment between the relative amounts of data and the importance of that data causes fundamental problems for imitation learning approaches. In this paper we analyze and develop learning methods that automatically account for mixed datasets. We formally prove that imbalanced data leads to imbalanced policies when each state-action pair is weighted equally; these policies emulate the most represented behaviors, and not the human's complex, multi-task demonstrations. We next explore algorithms that rebalance offline datasets (i.e., reweight the importance of different state-action pairs) without human oversight. Reweighting the dataset can enhance the overall policy performance. However, there is no free lunch: each method for autonomously rebalancing brings its own pros and cons. We formulate these advantages and disadvantages, helping other researchers identify when each type of approach is most appropriate. We conclude by introducing a novel meta-gradient rebalancing algorithm that addresses the primary limitations behind existing approaches. Our experiments show that dataset rebalancing leads to better downstream learning, improving the performance of general imitation learning algorithms without requiring additional data collection. See our project website: this https URL. 

**Abstract (ZH)**: 机器人应该能够从人类示范中学习复杂行为。实际上，这些由人类提供的数据集不可避免地存在不平衡：即人类对某些子任务的演示频率高于其他子任务。当前最先进的方法默认认为人类数据集中的每一个元素都是同样重要的。因此，如果——例如——大多数人类数据集中在目标导向行为上，而只有少数状态-动作对涉及避开障碍物，那么学习算法将更重视目标导向行为。更一般地说，数据量与数据重要性之间的不匹配会导致模仿学习方法出现根本性问题。在本文中，我们分析并开发了能够自动处理混合数据集的学习方法。我们正式证明，当每一个状态-动作对都被同等加权时，不平衡的数据会导致生成不平衡的策略；这些策略模拟的是最常出现的行为，并非人类复杂、多任务的示范。我们接下来探索算法，这些算法能够在无人监管的情况下重新平衡离线数据集（即重新加权不同状态-动作对的重要性）。重新加权数据集可以提升整体策略表现，但没有任何免费午餐：每种自动重新平衡的方法都有其优势和劣势。我们阐明了这些优缺点，帮助其他研究人员确定每种方法最合适的情形。我们最后介绍了一种新颖的元梯度重新平衡算法，以解决现有方法背后的主要局限性。我们的实验表明，数据集重新平衡能够提高下游学习效果，改进一般模仿学习算法的表现而无需额外的数据收集工作。更多信息请访问我们的项目网站：this https URL。 

---
# Surrogate-Enhanced Modeling and Adaptive Modular Control of All-Electric Heavy-Duty Robotic Manipulators 

**Title (ZH)**: 增强代理模型与自适应模块化控制的全电气重型机器人 manipulator 系统建模与控制 

**Authors**: Amir Hossein Barjini, Mohammad Bahari, Mahdi Hejrati, Jouni Mattila  

**Link**: [PDF](https://arxiv.org/pdf/2508.06313)  

**Abstract**: This paper presents a unified system-level modeling and control framework for an all-electric heavy-duty robotic manipulator (HDRM) driven by electromechanical linear actuators (EMLAs). A surrogate-enhanced actuator model, combining integrated electromechanical dynamics with a neural network trained on a dedicated testbed, is integrated into an extended virtual decomposition control (VDC) architecture augmented by a natural adaptation law. The derived analytical HDRM model supports a hierarchical control structure that seamlessly maps high-level force and velocity objectives to real-time actuator commands, accompanied by a Lyapunov-based stability proof. In multi-domain simulations of both cubic and a custom planar triangular trajectory, the proposed adaptive modular controller achieves sub-centimeter Cartesian tracking accuracy. Experimental validation of the same 1-DoF platform under realistic load emulation confirms the efficacy of the proposed control strategy. These findings demonstrate that a surrogate-enhanced EMLA model embedded in the VDC approach can enable modular, real-time control of an all-electric HDRM, supporting its deployment in next-generation mobile working machines. 

**Abstract (ZH)**: 本文提出了一个统一的系统级建模与控制框架，用于由电磁线性执行器驱动的全电动重型机器人 manipulator (HDRM)。该框架结合了集成机电动力学与专用测试床训练的神经网络的代理增强执行器模型，并集成到扩展的虚拟分解控制 (VDC) 架构中，该架构带有自然自适应定律。从中推导出的分析型 HDRM 模型支持分层控制结构，能够无缝地将高层力和速度目标映射到实时执行器命令，并伴有基于李雅普诺夫的稳定性证明。在对三维和自定义平面三角形轨迹的多领域仿真中，所提出的自适应模块化控制器实现了亚毫米级笛卡尔跟踪精度。在现实负载模拟下的同一 1-DoF 平台的实验验证确认了所提控制策略的有效性。这些发现表明，嵌入 VDC 方法中的代理增强型 EMLA 模型能够实现全电动 HDRM 的模块化、实时控制，支持其在下一代移动作业机械中的部署。 

---
# Evaluating Robot Program Performance with Power Consumption Driven Metrics in Lightweight Industrial Robots 

**Title (ZH)**: 基于能耗驱动指标评估轻型工业机器人程序性能 

**Authors**: Juan Heredia, Emil Stubbe Kolvig-Raun, Sune Lundo Sorensen, Mikkel Baun Kjaergaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.06295)  

**Abstract**: The code performance of industrial robots is typically analyzed through CPU metrics, which overlook the physical impact of code on robot behavior. This study introduces a novel framework for assessing robot program performance from an embodiment perspective by analyzing the robot's electrical power profile. Our approach diverges from conventional CPU based evaluations and instead leverages a suite of normalized metrics, namely, the energy utilization coefficient, the energy conversion metric, and the reliability coefficient, to capture how efficiently and reliably energy is used during task execution. Complementing these metrics, the established robot wear metric provides further insight into long term reliability. Our approach is demonstrated through an experimental case study in machine tending, comparing four programs with diverse strategies using a UR5e robot. The proposed metrics directly compare and categorize different robot programs, regardless of the specific task, by linking code performance to its physical manifestation through power consumption patterns. Our results reveal the strengths and weaknesses of each strategy, offering actionable insights for optimizing robot programming practices. Enhancing energy efficiency and reliability through this embodiment centric approach not only improves individual robot performance but also supports broader industrial objectives such as sustainable manufacturing and cost reduction. 

**Abstract (ZH)**: 基于实体视角的工业机器人程序性能评估框架：通过电气功率轮廓分析能量使用效率和可靠性 

---
# Real-Time 3D Vision-Language Embedding Mapping 

**Title (ZH)**: 实时3D视觉-语言嵌入映射 

**Authors**: Christian Rauch, Björn Ellensohn, Linus Nwankwo, Vedant Dave, Elmar Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2508.06291)  

**Abstract**: A metric-accurate semantic 3D representation is essential for many robotic tasks. This work proposes a simple, yet powerful, way to integrate the 2D embeddings of a Vision-Language Model in a metric-accurate 3D representation at real-time. We combine a local embedding masking strategy, for a more distinct embedding distribution, with a confidence-weighted 3D integration for more reliable 3D embeddings. The resulting metric-accurate embedding representation is task-agnostic and can represent semantic concepts on a global multi-room, as well as on a local object-level. This enables a variety of interactive robotic applications that require the localisation of objects-of-interest via natural language. We evaluate our approach on a variety of real-world sequences and demonstrate that these strategies achieve a more accurate object-of-interest localisation while improving the runtime performance in order to meet our real-time constraints. We further demonstrate the versatility of our approach in a variety of interactive handheld, mobile robotics and manipulation tasks, requiring only raw image data. 

**Abstract (ZH)**: 一种度量准确的语义3D表示对于许多机器人任务至关重要。本文提出了一种简单而强大的方法，实现在实时条件下将视觉-语言模型的2D嵌入整合到度量准确的3D表示中。我们结合了局部嵌入掩蔽策略以获得更明显的嵌入分布，并使用信心加权的3D整合以获得更可靠的3D嵌入。所得的度量准确嵌入表示具有任务无关性，能够在全球多房间尺度和局部物体尺度上表示语义概念。这使得可以通过自然语言进行目标物体的局部化的一系列交互式机器人应用成为可能。我们在多种现实世界序列上评估了我们的方法，并展示了这些策略在提高目标物体局部化准确性的同时，提高了运行时性能，以满足我们的实时约束。我们还展示了我们方法在各种交互式手持机器人、移动机器人和操作任务中的通用性，仅需使用原始图像数据即可。 

---
# Situationally-aware Path Planning Exploiting 3D Scene Graphs 

**Title (ZH)**: 基于3D场景图的情境感知路径规划 

**Authors**: Saad Ejaz, Marco Giberna, Muhammad Shaheer, Jose Andres Millan-Romera, Ali Tourani, Paul Kremer, Holger Voos, Jose Luis Sanchez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2508.06283)  

**Abstract**: 3D Scene Graphs integrate both metric and semantic information, yet their structure remains underutilized for improving path planning efficiency and interpretability. In this work, we present S-Path, a situationally-aware path planner that leverages the metric-semantic structure of indoor 3D Scene Graphs to significantly enhance planning efficiency. S-Path follows a two-stage process: it first performs a search over a semantic graph derived from the scene graph to yield a human-understandable high-level path. This also identifies relevant regions for planning, which later allows the decomposition of the problem into smaller, independent subproblems that can be solved in parallel. We also introduce a replanning mechanism that, in the event of an infeasible path, reuses information from previously solved subproblems to update semantic heuristics and prioritize reuse to further improve the efficiency of future planning attempts. Extensive experiments on both real-world and simulated environments show that S-Path achieves average reductions of 5.7x in planning time while maintaining comparable path optimality to classical sampling-based planners and surpassing them in complex scenarios, making it an efficient and interpretable path planner for environments represented by indoor 3D Scene Graphs. 

**Abstract (ZH)**: 基于3D场景图的情况感知路径规划 

---
# Mitigating Undesired Conditions in Flexible Production with Product-Process-Resource Asset Knowledge Graphs 

**Title (ZH)**: 基于产品-工艺-资源资产知识图谱的柔性生产中不良条件缓解方法 

**Authors**: Petr Novak, Stefan Biffl, Marek Obitko, Petr Kadera  

**Link**: [PDF](https://arxiv.org/pdf/2508.06278)  

**Abstract**: Contemporary industrial cyber-physical production systems (CPPS) composed of robotic workcells face significant challenges in the analysis of undesired conditions due to the flexibility of Industry 4.0 that disrupts traditional quality assurance mechanisms. This paper presents a novel industry-oriented semantic model called Product-Process-Resource Asset Knowledge Graph (PPR-AKG), which is designed to analyze and mitigate undesired conditions in flexible CPPS. Built on top of the well-proven Product-Process-Resource (PPR) model originating from ISA-95 and VDI-3682, a comprehensive OWL ontology addresses shortcomings of conventional model-driven engineering for CPPS, particularly inadequate undesired condition and error handling representation. The integration of semantic technologies with large language models (LLMs) provides intuitive interfaces for factory operators, production planners, and engineers to interact with the entire model using natural language. Evaluation with the use case addressing electric vehicle battery remanufacturing demonstrates that the PPR-AKG approach efficiently supports resource allocation based on explicitly represented capabilities as well as identification and mitigation of undesired conditions in production. The key contributions include (1) a holistic PPR-AKG model capturing multi-dimensional production knowledge, and (2) the useful combination of the PPR-AKG with LLM-based chatbots for human interaction. 

**Abstract (ZH)**: 当代工业物理- cyber物理生产系统（CPPS）由机器人工作单元组成，面临着由于工业4.0的灵活性而对非期望条件进行分析的重大挑战，这破坏了传统的质量保证机制。本文提出了一种面向行业的新型语义模型——产品-过程-资源资产知识图谱（PPR-AKG），旨在分析和缓解灵活CPPS中的非期望条件。该模型基于可信的ISA-95和VDI-3682起源的PPR模型，在此基础上构建了一个全面的OWL本体，解决了传统模型驱动工程在CPPS中的不足，特别是在非期望条件和错误处理表示方面的不足。语义技术与大型语言模型（LLMs）的集成为工厂操作员、生产计划人员和工程师提供了直观的接口，使其能够使用自然语言与整个模型进行交互。使用电动汽车电池再制造案例研究的评估表明，PPR-AKG方法能够有效地根据显式表示的能力进行资源分配，并识别和缓解生产中的非期望条件。关键贡献包括（1）一个综合的PPR-AKG模型，捕获多维度的生产知识；（2）PPR-AKG与基于LLM的聊天机器人的有效结合，用于人类交互。 

---
# EcBot: Data-Driven Energy Consumption Open-Source MATLAB Library for Manipulators 

**Title (ZH)**: EcBot: 数据驱动的操纵器开源MATLAB能耗库 

**Authors**: Juan Heredia, Christian Schlette, Mikkel Baun Kjærgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.06276)  

**Abstract**: Existing literature proposes models for estimating the electrical power of manipulators, yet two primary limitations prevail. First, most models are predominantly tested using traditional industrial robots. Second, these models often lack accuracy. To address these issues, we introduce an open source Matlab-based library designed to automatically generate \ac{ec} models for manipulators. The necessary inputs for the library are Denavit-Hartenberg parameters, link masses, and centers of mass. Additionally, our model is data-driven and requires real operational data, including joint positions, velocities, accelerations, electrical power, and corresponding timestamps. We validated our methodology by testing on four lightweight robots sourced from three distinct manufacturers: Universal Robots, Franka Emika, and Kinova. The model underwent testing, and the results demonstrated an RMSE ranging from 1.42 W to 2.80 W for the training dataset and from 1.45 W to 5.25 W for the testing dataset. 

**Abstract (ZH)**: 现有的文献提出了估计 manipulator 电气功率的模型，但主要存在两个局限性。首先，大多数模型主要在传统工业机器人上进行测试。其次，这些模型往往缺乏准确性。为解决这些问题，我们介绍了一个基于 Matlab 的开源库，该库旨在自动为 manipulator 生成 \ac{ec} 模型。该库所需的输入包括 Denavit-Hartenberg 参数、连杆质量以及质心。此外，我们的模型是数据驱动的，需要实际操作数据，包括关节位置、速度、加速度、电气功率及其对应的timestamp。我们通过在三个不同制造商的四款轻量化机器人（Universal Robots、Franka Emika 和 Kinova）上进行测试验证了该方法。模型进行了测试，结果显示训练数据集的 RMSE 范围为 1.42 W 至 2.80 W，测试数据集的 RMSE 范围为 1.45 W 至 5.25 W。 

---
# ADPro: a Test-time Adaptive Diffusion Policy for Robot Manipulation via Manifold and Initial Noise Constraints 

**Title (ZH)**: ADPro：基于流形和起始噪声约束的运行时自适应扩散策略用于机器人操作 

**Authors**: Zezeng Li, Rui Yang, Ruochen Chen, ZhongXuan Luo, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06266)  

**Abstract**: Diffusion policies have recently emerged as a powerful class of visuomotor controllers for robot manipulation, offering stable training and expressive multi-modal action modeling. However, existing approaches typically treat action generation as an unconstrained denoising process, ignoring valuable a priori knowledge about geometry and control structure. In this work, we propose the Adaptive Diffusion Policy (ADP), a test-time adaptation method that introduces two key inductive biases into the diffusion. First, we embed a geometric manifold constraint that aligns denoising updates with task-relevant subspaces, leveraging the fact that the relative pose between the end-effector and target scene provides a natural gradient direction, and guiding denoising along the geodesic path of the manipulation manifold. Then, to reduce unnecessary exploration and accelerate convergence, we propose an analytically guided initialization: rather than sampling from an uninformative prior, we compute a rough registration between the gripper and target scenes to propose a structured initial noisy action. ADP is compatible with pre-trained diffusion policies and requires no retraining, enabling test-time adaptation that tailors the policy to specific tasks, thereby enhancing generalization across novel tasks and environments. Experiments on RLBench, CALVIN, and real-world dataset show that ADPro, an implementation of ADP, improves success rates, generalization, and sampling efficiency, achieving up to 25% faster execution and 9% points over strong diffusion baselines. 

**Abstract (ZH)**: 自适应扩散策略：一种适用于机器人操作的测试时自适应方法 

---
# REBot: Reflexive Evasion Robot for Instantaneous Dynamic Obstacle Avoidance 

**Title (ZH)**: REBot: 具有即时动态障碍避让能力的反射性规避机器人 

**Authors**: Zihao Xu, Ce Hao, Chunzheng Wang, Kuankuan Sima, Fan Shi, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.06229)  

**Abstract**: Dynamic obstacle avoidance (DOA) is critical for quadrupedal robots operating in environments with moving obstacles or humans. Existing approaches typically rely on navigation-based trajectory replanning, which assumes sufficient reaction time and leading to fails when obstacles approach rapidly. In such scenarios, quadrupedal robots require reflexive evasion capabilities to perform instantaneous, low-latency maneuvers. This paper introduces Reflexive Evasion Robot (REBot), a control framework that enables quadrupedal robots to achieve real-time reflexive obstacle avoidance. REBot integrates an avoidance policy and a recovery policy within a finite-state machine. With carefully designed learning curricula and by incorporating regularization and adaptive rewards, REBot achieves robust evasion and rapid stabilization in instantaneous DOA tasks. We validate REBot through extensive simulations and real-world experiments, demonstrating notable improvements in avoidance success rates, energy efficiency, and robustness to fast-moving obstacles. Videos and appendix are available on this https URL. 

**Abstract (ZH)**: 四足机器人动态障碍物回避：Reflexive Evasion Robot (REBot) 实时反应性障碍物回避控制框架 

---
# Computer Vision-based Adaptive Control for Back Exoskeleton Performance Optimization 

**Title (ZH)**: 基于计算机视觉的自适应控制以优化背部外骨骼性能 

**Authors**: Andrea Dal Prete, Seyram Ofori, Chan Yon Sin, Ashwin Narayan, Francesco Braghin, Marta Gandolla, Haoyong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06207)  

**Abstract**: Back exoskeletons can reduce musculoskeletal strain, but their effectiveness depends on support modulation and adaptive control. This study addresses two challenges: defining optimal support strategies and developing adaptive control based on payload estimation. We introduce an optimization space based on muscle activity reduction, perceived discomfort, and user preference, constructing functions to identify optimal strategies. Experiments with 12 subjects revealed optimal operating regions, highlighting the need for dynamic modulation. Based on these insights, we developed a vision-based adaptive control pipeline that estimates payloads in real-time by enhancing exoskeleton contextual understanding, minimising latency and enabling support adaptation within the defined optimisation space. Validation with 12 more subjects showed over 80% accuracy and improvements across all metrics. Compared to static control, adaptive modulation reduced peak back muscle activation by up to 23% while preserving user preference and minimising discomfort. These findings validate the proposed framework and highlight the potential of intelligent, context-aware control in industrial exoskeletons. 

**Abstract (ZH)**: 背负外骨骼可以减轻肌骨应力，但其有效性取决于支持调节和自适应控制。本研究解决两大挑战：定义最优支持策略并基于负载估计开发自适应控制。我们基于肌肉活动减少、感知不适和用户偏好引入优化空间，构建函数以识别最优策略。12名受试者的实验揭示了最优操作区域，强调了动态调节的必要性。基于这些洞察，我们开发了一种基于视觉的自适应控制流水线，通过增强外骨骼的上下文理解实时估计负载，减少延迟并使支持适应于定义的优化空间。12名更多受试者的验证显示超过80%的准确率，并在所有指标上有所改善。与静态控制相比，自适应调节可将背部肌肉峰值激活降低高达23%，同时保持用户偏好并最小化不适。这些发现验证了所提出的框架，并突显了智能、上下文感知控制在工业外骨骼中的潜力。 

---
# Affordance-R1: Reinforcement Learning for Generalizable Affordance Reasoning in Multimodal Large Language Model 

**Title (ZH)**: 支撑性推理-R1：多模态大型语言模型中的可泛化支撑性学习推理 

**Authors**: Hanqing Wang, Shaoyang Wang, Yiming Zhong, Zemin Yang, Jiamin Wang, Zhiqing Cui, Jiahao Yuan, Yifan Han, Mingyu Liu, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.06206)  

**Abstract**: Affordance grounding focuses on predicting the specific regions of objects that are associated with the actions to be performed by robots. It plays a vital role in the fields of human-robot interaction, human-object interaction, embodied manipulation, and embodied perception. Existing models often neglect the affordance shared among different objects because they lack the Chain-of-Thought(CoT) reasoning abilities, limiting their out-of-domain (OOD) generalization and explicit reasoning capabilities. To address these challenges, we propose Affordance-R1, the first unified affordance grounding framework that integrates cognitive CoT guided Group Relative Policy Optimization (GRPO) within a reinforcement learning paradigm. Specifically, we designed a sophisticated affordance function, which contains format, perception, and cognition rewards to effectively guide optimization directions. Furthermore, we constructed a high-quality affordance-centric reasoning dataset, ReasonAff, to support training. Trained exclusively via reinforcement learning with GRPO and without explicit reasoning data, Affordance-R1 achieves robust zero-shot generalization and exhibits emergent test-time reasoning capabilities. Comprehensive experiments demonstrate that our model outperforms well-established methods and exhibits open-world generalization. To the best of our knowledge, Affordance-R1 is the first to integrate GRPO-based RL with reasoning into affordance reasoning. The code of our method and our dataset is released on this https URL. 

**Abstract (ZH)**: 面向物体的具体区域预测，以执行机器人所需动作，这种能力被称为效能接地。效能接地在人机交互、人机物交互、体现式操作和体现式感知等领域发挥着重要作用。现有的模型往往忽略了不同物体之间共享的效能，因为它们缺乏链式思考(CoT)推理能力，限制了它们的域外泛化能力和显式推理能力。为了解决这些问题，我们提出了Affordance-R1，这是首个将认知CoT引导下的组相对策略优化( GRPO)统一集成到强化学习框架中的效能接地框架。具体而言，我们设计了一种复杂的效能函数，包含格式、感知和认知奖励，以有效引导优化方向。此外，我们构建了一个高质量的以效能为中心的推理数据集ReasonAff，以支持训练。仅通过强化学习和GRPO训练，且未使用显式推理数据，Affordance-R1实现了稳健的零样本泛化，并表现出测试时的推理能力。全面的实验表明，我们的模型优于已建立的方法，并展现了开放世界的泛化能力。据我们所知，Affordance-R1是首个将基于GRPO的强化学习与推理结合到效能推理中的方法。我们的方法代码和数据集可在以下链接获取。 

---
# Beyond Constant Parameters: Hyper Prediction Models and HyperMPC 

**Title (ZH)**: 超越恒定参数：超前预测模型与超前MPC 

**Authors**: Jan Węgrzynowski, Piotr Kicki, Grzegorz Czechmanowski, Maciej Krupka, Krzysztof Walas  

**Link**: [PDF](https://arxiv.org/pdf/2508.06181)  

**Abstract**: Model Predictive Control (MPC) is among the most widely adopted and reliable methods for robot control, relying critically on an accurate dynamics model. However, existing dynamics models used in the gradient-based MPC are limited by computational complexity and state representation. To address this limitation, we propose the Hyper Prediction Model (HyperPM) - a novel approach in which we project the unmodeled dynamics onto a time-dependent dynamics model. This time-dependency is captured through time-varying model parameters, whose evolution over the MPC prediction horizon is learned using a neural network. Such formulation preserves the computational efficiency and robustness of the base model while equipping it with the capacity to anticipate previously unmodeled phenomena. We evaluated the proposed approach on several challenging systems, including real-world F1TENTH autonomous racing, and demonstrated that it significantly reduces long-horizon prediction errors. Moreover, when integrated within the MPC framework (HyperMPC), our method consistently outperforms existing state-of-the-art techniques. 

**Abstract (ZH)**: 基于预测模型的控制（HyperPM）在机器人控制中的应用：一种通过时间依赖动态模型投影未建模动态的新方法 

---
# Bounding Distributional Shifts in World Modeling through Novelty Detection 

**Title (ZH)**: 基于新颖性检测限定世界建模中的分布偏移 

**Authors**: Eric Jing, Abdeslam Boularias  

**Link**: [PDF](https://arxiv.org/pdf/2508.06096)  

**Abstract**: Recent work on visual world models shows significant promise in latent state dynamics obtained from pre-trained image backbones. However, most of the current approaches are sensitive to training quality, requiring near-complete coverage of the action and state space during training to prevent divergence during inference. To make a model-based planning algorithm more robust to the quality of the learned world model, we propose in this work to use a variational autoencoder as a novelty detector to ensure that proposed action trajectories during planning do not cause the learned model to deviate from the training data distribution. To evaluate the effectiveness of this approach, a series of experiments in challenging simulated robot environments was carried out, with the proposed method incorporated into a model-predictive control policy loop extending the DINO-WM architecture. The results clearly show that the proposed method improves over state-of-the-art solutions in terms of data efficiency. 

**Abstract (ZH)**: Recent Work on Visual World Models Shows Significant Promise in Latent State Dynamics Obtained from Pre-Trained Image Backbones: Using a Variational Autoencoder as a Novelty Detector to Enhance Robustness of Model-Based Planning Algorithms 

---
# Incremental Language Understanding for Online Motion Planning of Robot Manipulators 

**Title (ZH)**: 基于增量语言理解的机器人 manipulator 在线运动规划 

**Authors**: Mitchell Abrams, Thies Oelerich, Christian Hartl-Nesic, Andreas Kugi, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2508.06095)  

**Abstract**: Human-robot interaction requires robots to process language incrementally, adapting their actions in real-time based on evolving speech input. Existing approaches to language-guided robot motion planning typically assume fully specified instructions, resulting in inefficient stop-and-replan behavior when corrections or clarifications occur. In this paper, we introduce a novel reasoning-based incremental parser which integrates an online motion planning algorithm within the cognitive architecture. Our approach enables continuous adaptation to dynamic linguistic input, allowing robots to update motion plans without restarting execution. The incremental parser maintains multiple candidate parses, leveraging reasoning mechanisms to resolve ambiguities and revise interpretations when needed. By combining symbolic reasoning with online motion planning, our system achieves greater flexibility in handling speech corrections and dynamically changing constraints. We evaluate our framework in real-world human-robot interaction scenarios, demonstrating online adaptions of goal poses, constraints, or task objectives. Our results highlight the advantages of integrating incremental language understanding with real-time motion planning for natural and fluid human-robot collaboration. The experiments are demonstrated in the accompanying video at this http URL. 

**Abstract (ZH)**: 人类-机器人交互需要机器人基于不断演化的语音输入进行增量语言处理，并实时调整其行动。现有的基于语言的机器人运动规划方法通常假设完整的指令，这会导致在出现修正或澄清时产生不高效的暂停和重新规划行为。本文介绍了融合在线运动规划算法的认知架构中的一种新颖的基于推理的增量解析器。我们的方法使机器人能够对动态语言输入进行连续适应，无需重新启动即可更新运动计划。增量解析器维护多个候选解析，并利用推理机制在需要时解决歧义并修订解释。通过结合符号推理与在线运动规划，我们的系统在处理语音修正和动态变化的约束方面表现出更大的灵活性。我们在实际的人机交互场景中评估了我们的框架，展示了对目标姿态、约束或任务目标的在线适应。我们的结果突显了将增量语言理解和实时运动规划集成到自然流畅的人机协作中的优势。实验详细情况请参见附带的视频。 

---
# ReNiL: Relative Neural Inertial Locator with Any-Scale Bayesian Inference 

**Title (ZH)**: 相对神经惯性定位器：任意尺度贝叶斯推断 

**Authors**: Kaixuan Wu, Yuanzhuo Xu, Zejun Zhang, Weiping Zhu, Steve Drew, Xiaoguang Niu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06053)  

**Abstract**: Pedestrian inertial localization is key for mobile and IoT services because it provides infrastructure-free positioning. Yet most learning-based methods depend on fixed sliding-window integration, struggle to adapt to diverse motion scales and cadences, and yield inconsistent uncertainty, limiting real-world use. We present ReNiL, a Bayesian deep-learning framework for accurate, efficient, and uncertainty-aware pedestrian localization. ReNiL introduces Inertial Positioning Demand Points (IPDPs) to estimate motion at contextually meaningful waypoints instead of dense tracking, and supports inference on IMU sequences at any scale so cadence can match application needs. It couples a motion-aware orientation filter with an Any-Scale Laplace Estimator (ASLE), a dual-task network that blends patch-based self-supervision with Bayesian regression. By modeling displacements with a Laplace distribution, ReNiL provides homogeneous Euclidean uncertainty that integrates cleanly with other sensors. A Bayesian inference chain links successive IPDPs into consistent trajectories. On RoNIN-ds and a new WUDataset covering indoor and outdoor motion from 28 participants, ReNiL achieves state-of-the-art displacement accuracy and uncertainty consistency, outperforming TLIO, CTIN, iMoT, and RoNIN variants while reducing computation. Application studies further show robustness and practicality for mobile and IoT localization, making ReNiL a scalable, uncertainty-aware foundation for next-generation positioning. 

**Abstract (ZH)**: 基于惯性的人行定位对于移动和物联网服务至关重要，因为它提供了无基础设施的位置定位。然而，大多数基于学习的方法依赖于固定的时间窗集成，难以适应多样的运动规模和步伐，并且会得出不一致的不确定性，限制了其实用性。我们提出了ReNiL，这是一种用于准确、高效且具备不确定性意识的人行定位的贝叶斯深度学习框架。ReNiL 引入了惯性定位需求点（IPDPs）来在上下文相关的重要点处估计运动，而非密集跟踪，并支持在任何规模的 IMU 序列上进行推理，以便步调能够匹配应用需求。该框架结合了运动感知的方向滤波器与任意尺度拉普拉斯估计器（ASLE），这是一种兼具基于块的自监督与贝叶斯回归的双任务网络。通过使用拉普拉斯分布建模位移，ReNiL 提供了均匀的欧几里得不确定性，可以与其他传感器良好集成。贝叶斯推理链将连续的 IPDPs 连接成一致的轨迹。在RoNIN-ds数据集和一个包含28名参与者室内和室外运动的新WUDataset上，ReNiL 达到了最先进的位移精度和不确定性一致性，表现优于TLIO、CTIN、iMoT 和 RoNIN 变体，并且减少了计算量。进一步的研究表明，ReNiL 在移动和物联网定位中具有稳健性和实用性，使其成为下一代定位的可扩展且具备不确定性意识的基础。 

---
# Dynamical Trajectory Planning of Disturbance Consciousness for Air-Land Bimodal Unmanned Aerial Vehicles 

**Title (ZH)**: 扰动意识导向的空地两用无人航空器动态轨迹规划 

**Authors**: Shaoting Liu, Zhou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05972)  

**Abstract**: Air-land bimodal vehicles provide a promising solution for navigating complex environments by combining the flexibility of aerial locomotion with the energy efficiency of ground mobility. To enhance the robustness of trajectory planning under environmental disturbances, this paper presents a disturbance-aware planning framework that incorporates real-time disturbance estimation into both path searching and trajectory optimization. A key component of the framework is a disturbance-adaptive safety boundary adjustment mechanism, which dynamically modifies the vehicle's feasible dynamic boundaries based on estimated disturbances to ensure trajectory feasibility. Leveraging the dynamics model of the bimodal vehicle, the proposed approach achieves adaptive and reliable motion planning across different terrains and operating conditions. A series of real-world experiments and benchmark comparisons on a custom-built platform validate the effectiveness and robustness of the method, demonstrating improvements in tracking accuracy, task efficiency, and energy performance under both ground and aerial disturbances. 

**Abstract (ZH)**: 空气-地面双模车辆通过结合空中运动的灵活性和地面行驶的能效性，提供了一种应对复杂环境的有前景的解决方案。为了在环境干扰下增强轨迹规划的鲁棒性，本文提出了一种干扰感知规划框架，该框架将实时干扰估计融入路径搜索和轨迹优化中。该框架的关键组件是一个干扰自适应安全边界调整机制，该机制根据估计到的干扰动态修改车辆的可行动态边界，以确保轨迹的可行性。利用双模车辆的动力学模型，所提出的方法实现了在不同地形和操作条件下的自适应和可靠的运动规划。在自建平台上进行的一系列实际实验和基准比较验证了该方法的有效性和鲁棒性，展示了在地面和空中干扰下跟踪准确性、任务效率和能源性能的提升。 

---
# Social and Telepresence Robots for Accessibility and Inclusion in Small Museums 

**Title (ZH)**: 社交机器人和远程存在机器人在小型博物馆中的可达性和包容性应用 

**Authors**: Nello Balossino, Rossana Damiano, Cristina Gena, Alberto Lillo, Anna Maria Marras, Claudio Mattutino, Antonio Pizzo, Alessia Prin, Fabiana Vernero  

**Link**: [PDF](https://arxiv.org/pdf/2508.05946)  

**Abstract**: There are still many museums that present accessibility barriers, particularly regarding perceptual, cultural, and cognitive aspects. This is especially evident in low-density population areas. The aim of the ROBSO-PM project is to improve the accessibility of small museums through the use of social robots and social telepresence robots, focusing on three museums as case studies: the Museum of the Holy Shroud in Turin, a small but globally known institution, and two lesser known mountain museums: the Museum of the Champlas du Col Carnival and the Pragelato Museum of Alpine Peoples' Costumes and Traditions. The project explores two main applications for robots: as guides supporting inclusive visits for foreign or disabled visitors, and as telepresence tools allowing people with limited mobility to access museums remotely. From a research perspective, key topics include storytelling, robot personality, empathy, personalization, and, in the case of telepresence, collaboration between the robot and the person, with clearly defined roles and autonomy. 

**Abstract (ZH)**: ROBSO-PM项目旨在通过使用社会机器人和社会远程存在机器人改善低密度人口区域小型博物馆的可达性，以Turin圣lös垂幕博物馆、Champlas du Col Carnival博物馆和Pragelato山地民族服饰与传统博物馆为案例进行研究。该项目探索了两类主要应用：作为导览机器人支持包容性参观，如为外国或残障游客服务；以及作为远程存在工具，使行动不便的人士能够远程访问博物馆。从研究视角来看，关键主题包括故事叙述、机器人个性、同理心、个性化，以及在远程存在的情况下机器人与人员之间的合作与明确分工。 

---
# Latent Policy Barrier: Learning Robust Visuomotor Policies by Staying In-Distribution 

**Title (ZH)**: 潜在策略障碍：通过保持在分布内学习稳健的视觉运动策略 

**Authors**: Zhanyi Sun, Shuran Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.05941)  

**Abstract**: Visuomotor policies trained via behavior cloning are vulnerable to covariate shift, where small deviations from expert trajectories can compound into failure. Common strategies to mitigate this issue involve expanding the training distribution through human-in-the-loop corrections or synthetic data augmentation. However, these approaches are often labor-intensive, rely on strong task assumptions, or compromise the quality of imitation. We introduce Latent Policy Barrier, a framework for robust visuomotor policy learning. Inspired by Control Barrier Functions, LPB treats the latent embeddings of expert demonstrations as an implicit barrier separating safe, in-distribution states from unsafe, out-of-distribution (OOD) ones. Our approach decouples the role of precise expert imitation and OOD recovery into two separate modules: a base diffusion policy solely on expert data, and a dynamics model trained on both expert and suboptimal policy rollout data. At inference time, the dynamics model predicts future latent states and optimizes them to stay within the expert distribution. Both simulated and real-world experiments show that LPB improves both policy robustness and data efficiency, enabling reliable manipulation from limited expert data and without additional human correction or annotation. 

**Abstract (ZH)**: 基于行为克隆训练的视觉运动策略易受 covariate shift 影响，细微的专家轨迹偏差可能导致失败。我们提出了一种名为潜策略障碍（Latent Policy Barrier, LPB）的框架，用于稳健的视觉运动策略学习。LPB 受控制障碍函数启发，将专家演示的潜在嵌入视为安全、同分布状态与不安全、异分布状态之间的隐式障碍。我们的方法将精确的专家模仿和异分布恢复的任务分解为两个独立模块：基于专家数据的基扩散策略和同时基于专家数据和次优策略滚动数据训练的动力学模型。在推理时，动力学模型预测未来潜在状态并优化它们以保持在专家分布内。实验结果表明，LPB 提高了策略的稳健性和数据效率，能够在有限的专家数据下实现可靠的操纵，无需额外的人工纠正或标注。 

---
# Affordance-Guided Dual-Armed Disassembly Teleoperation for Mating Parts 

**Title (ZH)**: 面向功能的双臂拆装远程操作以实现零件对接 

**Authors**: Gen Sako, Takuya Kiyokawa, Kensuke Harada, Tomoki Ishikura, Naoya Miyaji, Genichiro Matsuda  

**Link**: [PDF](https://arxiv.org/pdf/2508.05937)  

**Abstract**: Robotic non-destructive disassembly of mating parts remains challenging due to the need for flexible manipulation and the limited visibility of internal structures. This study presents an affordance-guided teleoperation system that enables intuitive human demonstrations for dual-arm fix-and-disassemble tasks for mating parts. The system visualizes feasible grasp poses and disassembly directions in a virtual environment, both derived from the object's geometry, to address occlusions and structural complexity. To prevent excessive position tracking under load when following the affordance, we integrate a hybrid controller that combines position and impedance control into the teleoperated disassembly arm. Real-world experiments validate the effectiveness of the proposed system, showing improved task success rates and reduced object pose deviation. 

**Abstract (ZH)**: 基于 affordance 引导的遥操作系统：用于配合部件的固定与拆卸任务的直观人机示范方法 

---
# Modular Vacuum-Based Fixturing System for Adaptive Disassembly Workspace Integration 

**Title (ZH)**: 基于模块化真空fixture的自适应拆解工作空间集成系统 

**Authors**: Haohui Pan, Takuya Kiyokawa, Tomoki Ishikura, Shingo Hamada, Genichiro Matsuda, Kensuke Harada  

**Link**: [PDF](https://arxiv.org/pdf/2508.05936)  

**Abstract**: The disassembly of small household appliances poses significant challenges due to their complex and curved geometries, which render traditional rigid fixtures inadequate. In this paper, we propose a modular vacuum-based fixturing system that leverages commercially available balloon-type soft grippers to conform to arbitrarily shaped surfaces and provide stable support during screw-removal tasks. To enable a reliable deployment of the system, we develop a stability-aware planning framework that samples the bottom surface of the target object, filters candidate contact points based on geometric continuity, and evaluates support configurations using convex hull-based static stability criteria. We compare the quality of object placement under different numbers and configurations of balloon hands. In addition, real-world experiments were conducted to compare the success rates of traditional rigid fixtures with our proposed system. The results demonstrate that our method consistently achieves higher success rates and superior placement stability during screw removal tasks. 

**Abstract (ZH)**: 小家电的拆解由于其复杂的曲面几何结构带来了显著挑战，传统的刚性夹具已不足以应对。本文提出了一种模块化真空基座夹具系统，利用商用气球型柔软夹爪来适应任意形状的表面，并在拆卸螺钉任务中提供稳定的支撑。为了确保系统的有效部署，我们开发了一种基于稳定性的规划框架，通过对目标物体底部表面取样、基于几何连续性筛选候选接触点，并使用基于凸包的静态稳定准则评估支撑配置。我们比较了不同数量和配置的气球手对物体放置质量的影响。此外，我们还在现实世界中进行了实验，将传统刚性夹具的成功率与我们提出的系统进行了比较。结果表明，我们的方法在拆卸螺钉任务中始终能实现更高的成功率和更好的放置稳定性。 

---
# Integrating Vision Foundation Models with Reinforcement Learning for Enhanced Object Interaction 

**Title (ZH)**: 视觉基础模型与强化学习结合以增强物体交互 

**Authors**: Ahmad Farooq, Kamran Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2508.05838)  

**Abstract**: This paper presents a novel approach that integrates vision foundation models with reinforcement learning to enhance object interaction capabilities in simulated environments. By combining the Segment Anything Model (SAM) and YOLOv5 with a Proximal Policy Optimization (PPO) agent operating in the AI2-THOR simulation environment, we enable the agent to perceive and interact with objects more effectively. Our comprehensive experiments, conducted across four diverse indoor kitchen settings, demonstrate significant improvements in object interaction success rates and navigation efficiency compared to a baseline agent without advanced perception. The results show a 68% increase in average cumulative reward, a 52.5% improvement in object interaction success rate, and a 33% increase in navigation efficiency. These findings highlight the potential of integrating foundation models with reinforcement learning for complex robotic tasks, paving the way for more sophisticated and capable autonomous agents. 

**Abstract (ZH)**: 本文提出了一种将视觉基础模型与强化学习相结合的新方法，以增强模拟环境中对象交互的能力。通过将Segment Anything Model (SAM)与YOLOv5结合，并在AI2-THOR仿真环境中使用Proximal Policy Optimization (PPO)智能体，使智能体能够更有效地感知和交互对象。跨四个不同的室内厨房场景进行的全面实验表明，与没有高级感知的基线智能体相比，对象交互成功率和导航效率有了显著提高。结果表明，平均累积奖励提高了68%，对象交互成功率提高了52.5%，导航效率提高了33%。这些发现突显了将基础模型与强化学习结合使用在复杂机器人任务中的潜力，为更 sophisticated 和 capable 的自主智能体铺平了道路。 

---
# GPU-Accelerated Barrier-Rate Guided MPPI Control for Tractor-Trailer Systems 

**Title (ZH)**: 基于GPU加速的基于速率引导的MPPI控制算法在拖拉机- 牵引车系统中的应用 eSports 

**Authors**: Keyvan Majd, Hardik Parwana, Bardh Hoxha, Steven Hong, Hideki Okamoto, Georgios Fainekos  

**Link**: [PDF](https://arxiv.org/pdf/2508.05773)  

**Abstract**: Articulated vehicles such as tractor-trailers, yard trucks, and similar platforms must often reverse and maneuver in cluttered spaces where pedestrians are present. We present how Barrier-Rate guided Model Predictive Path Integral (BR-MPPI) control can solve navigation in such challenging environments. BR-MPPI embeds Control Barrier Function (CBF) constraints directly into the path-integral update. By steering the importance-sampling distribution toward collision-free, dynamically feasible trajectories, BR-MPPI enhances the exploration strength of MPPI and improves robustness of resulting trajectories. The method is evaluated in the high-fidelity CarMaker simulator on a 12 [m] tractor-trailer tasked with reverse and forward parking in a parking lot. BR-MPPI computes control inputs in above 100 [Hz] on a single GPU (for scenarios with eight obstacles) and maintains better parking clearance than a standard MPPI baseline and an MPPI with collision cost baseline. 

**Abstract (ZH)**: articulated车辆（如拖拉机-挂车、场内卡车等）常常需要在行人存在的拥挤空间中倒车和机动。我们展示了Barrier-Rate引导的模型预测路径积分（BR-MPPI）控制如何解决在这种具有挑战性的环境中的导航问题。BR-MPPI将控制障碍函数（CBF）约束直接嵌入到路径积分更新中。通过使重要性抽样分布朝向无碰撞的动态可行轨迹，BR-MPPI增强了MPPI的探索强度并提高了结果轨迹的鲁棒性。该方法在高保真CarMaker模拟器中对一辆12米长的拖拉机-挂车进行了评估，该拖车在停车场中进行倒车和前进停车。在包含八个障碍物的场景中，BR-MPPI在单个GPU上以超过100 Hz的频率计算控制输入，并在停车间隙方面优于标准的MPPI基线和带有碰撞成本的MPPI基线。 

---
# Depth Jitter: Seeing through the Depth 

**Title (ZH)**: 深度抖动：透过深度观察 

**Authors**: Md Sazidur Rahman, David Cabecinhas, Ricard Marxer  

**Link**: [PDF](https://arxiv.org/pdf/2508.06227)  

**Abstract**: Depth information is essential in computer vision, particularly in underwater imaging, robotics, and autonomous navigation. However, conventional augmentation techniques overlook depth aware transformations, limiting model robustness in real world depth variations. In this paper, we introduce Depth-Jitter, a novel depth-based augmentation technique that simulates natural depth variations to improve generalization. Our approach applies adaptive depth offsetting, guided by depth variance thresholds, to generate synthetic depth perturbations while preserving structural integrity. We evaluate Depth-Jitter on two benchmark datasets, FathomNet and UTDAC2020 demonstrating its impact on model stability under diverse depth conditions. Extensive experiments compare Depth-Jitter against traditional augmentation strategies such as ColorJitter, analyzing performance across varying learning rates, encoders, and loss functions. While Depth-Jitter does not always outperform conventional methods in absolute performance, it consistently enhances model stability and generalization in depth-sensitive environments. These findings highlight the potential of depth-aware augmentation for real-world applications and provide a foundation for further research into depth-based learning strategies. The proposed technique is publicly available to support advancements in depth-aware augmentation. The code is publicly available on \href{this https URL}{github}. 

**Abstract (ZH)**: 深度信息在计算机视觉中的应用对于水下成像、机器人技术和自主导航至关重要。然而，传统的增强技术忽视了深度感知变换，限制了模型在真实世界深度变化中的鲁棒性。本文引入了一种新颖的基于深度的增强技术——Depth-Jitter，以模拟自然的深度变化从而提高模型的泛化能力。我们的方法通过基于深度方差阈值的应用自适应深度偏移，生成合成的深度扰动同时保持结构完整性。我们在两个基准数据集FathomNet和UTDAC2020上评估了Depth-Jitter，展示了其在不同深度条件下的模型稳定性。广泛的经验研究表明，Depth-Jitter在传统增强策略如ColorJitter中表现更好，特别是在深度敏感环境中增强了模型的稳定性和泛化能力。这些发现突显了深度意识增强在实际应用中的潜力，并为深度学习策略的进一步研究奠定了基础。提出的技巧已公开发布，以支持深度意识增强技术的发展。源代码可在GitHub上公开获取。 

---
# Graph-based Robot Localization Using a Graph Neural Network with a Floor Camera and a Feature Rich Industrial Floor 

**Title (ZH)**: 基于图神经网络的.floor摄像机和丰富特征工业-floor的图表示机器人定位 

**Authors**: Dominik Brämer, Diana Kleingarn, Oliver Urbann  

**Link**: [PDF](https://arxiv.org/pdf/2508.06177)  

**Abstract**: Accurate localization represents a fundamental challenge in
robotic navigation. Traditional methodologies, such as Lidar or QR-code based systems, suffer from inherent scalability and adaptability con straints, particularly in complex environments. In this work, we propose
an innovative localization framework that harnesses flooring characteris tics by employing graph-based representations and Graph Convolutional
Networks (GCNs). Our method uses graphs to represent floor features,
which helps localize the robot more accurately (0.64cm error) and more
efficiently than comparing individual image features. Additionally, this
approach successfully addresses the kidnapped robot problem in every
frame without requiring complex filtering processes. These advancements
open up new possibilities for robotic navigation in diverse environments. 

**Abstract (ZH)**: 精确定位 localization 代表了机器人导航中的一项基本挑战 argent localization represents a fundamental challenge in robotic navigation. Patio présenté une nouvelle cadre framework 创新型的定位框架stdbool Localization Framework déco � santéитесь 使用图论基 based方法-tags 和图卷积网络 GCNs tô 表示地面特征，使得机器人定位更加准确 kinois
is (精度达到6.64cm) � 更加高效 �并且成功解决了复杂过滤过程中的机器人 kidnapping盗贼问题。is进步为机器人在各种环境中应用is提供了可能性。这些进步为机器人 différentis 在érrent是环境 kinoisisis提供了可能性 editText Localization vidé 新环境 mexico提供了可能性。 

---
# GMF-Drive: Gated Mamba Fusion with Spatial-Aware BEV Representation for End-to-End Autonomous Driving 

**Title (ZH)**: GMF-Drive：具有空间 Awareness 摄影机与激光雷达融合表示的端到端自主驾驶 

**Authors**: Jian Wang, Chaokang Jiang, Haitao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06113)  

**Abstract**: Diffusion-based models are redefining the state-of-the-art in end-to-end autonomous driving, yet their performance is increasingly hampered by a reliance on transformer-based fusion. These architectures face fundamental limitations: quadratic computational complexity restricts the use of high-resolution features, and a lack of spatial priors prevents them from effectively modeling the inherent structure of Bird's Eye View (BEV) representations. This paper introduces GMF-Drive (Gated Mamba Fusion for Driving), an end-to-end framework that overcomes these challenges through two principled innovations. First, we supersede the information-limited histogram-based LiDAR representation with a geometrically-augmented pillar format encoding shape descriptors and statistical features, preserving critical 3D geometric details. Second, we propose a novel hierarchical gated mamba fusion (GM-Fusion) architecture that substitutes an expensive transformer with a highly efficient, spatially-aware state-space model (SSM). Our core BEV-SSM leverages directional sequencing and adaptive fusion mechanisms to capture long-range dependencies with linear complexity, while explicitly respecting the unique spatial properties of the driving scene. Extensive experiments on the challenging NAVSIM benchmark demonstrate that GMF-Drive achieves a new state-of-the-art performance, significantly outperforming DiffusionDrive. Comprehensive ablation studies validate the efficacy of each component, demonstrating that task-specific SSMs can surpass a general-purpose transformer in both performance and efficiency for autonomous driving. 

**Abstract (ZH)**: 基于扩散的模型正在重新定义端到端自动驾驶的技术前沿，但它们的性能越来越受到基于变换器的融合依赖性的阻碍。这些架构面临着根本性的局限性：二次计算复杂性限制了高分辨率特征的使用，缺乏空间先验使它们难以有效地建模鸟瞰图（BEV）表示的固有结构。本文介绍了一种端到端框架GMF-Drive（门控-pillar格式融合用于驾驶），通过两大原则性创新克服了这些挑战。首先，我们用几何增强的pillar格式取代了信息有限的历史柱状LiDAR表示，编码形状描述符和统计特征，保留了关键的三维几何细节。其次，我们提出了一种新颖的分层门控mamba融合（GM-Fusion）架构，用高效的空间感知状态空间模型（SSM）替代了昂贵的变换器。我们的核心BEV-SSM利用方向序列和自适应融合机制以线性复杂度捕捉长程依赖性，同时明确尊重驾驶场景的独特空间属性。在具有挑战性的NAVSIM基准上的广泛实验表明，GMF-Drive实现了新的技术前沿性能，显著优于DiffusionDrive。全面的消融研究验证了每个组件的有效性，证明了任务特定的SSM在性能和效率方面可以超越通用的变换器，适用于自动驾驶。 

---
# ME$^3$-BEV: Mamba-Enhanced Deep Reinforcement Learning for End-to-End Autonomous Driving with BEV-Perception 

**Title (ZH)**: ME$^3$-BEV：增强型Mamba端到端自主驾驶的深度强化学习方法基于BEV感知 

**Authors**: Siyi Lu, Run Liu, Dongsheng Yang, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2508.06074)  

**Abstract**: Autonomous driving systems face significant challenges in perceiving complex environments and making real-time decisions. Traditional modular approaches, while offering interpretability, suffer from error propagation and coordination issues, whereas end-to-end learning systems can simplify the design but face computational bottlenecks. This paper presents a novel approach to autonomous driving using deep reinforcement learning (DRL) that integrates bird's-eye view (BEV) perception for enhanced real-time decision-making. We introduce the \texttt{Mamba-BEV} model, an efficient spatio-temporal feature extraction network that combines BEV-based perception with the Mamba framework for temporal feature modeling. This integration allows the system to encode vehicle surroundings and road features in a unified coordinate system and accurately model long-range dependencies. Building on this, we propose the \texttt{ME$^3$-BEV} framework, which utilizes the \texttt{Mamba-BEV} model as a feature input for end-to-end DRL, achieving superior performance in dynamic urban driving scenarios. We further enhance the interpretability of the model by visualizing high-dimensional features through semantic segmentation, providing insight into the learned representations. Extensive experiments on the CARLA simulator demonstrate that \texttt{ME$^3$-BEV} outperforms existing models across multiple metrics, including collision rate and trajectory accuracy, offering a promising solution for real-time autonomous driving. 

**Abstract (ZH)**: 基于深度强化学习的Mamba-BEV模型在实时决策增强的自主驾驶中应用 

---
# PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation 

**Title (ZH)**: PASG：一种用于机器人操作中自动几何原族提取和语义锚定的闭环框架 

**Authors**: Zhihao Zhu, Yifan Zheng, Siyu Pan, Yaohui Jin, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05976)  

**Abstract**: The fragmentation between high-level task semantics and low-level geometric features remains a persistent challenge in robotic manipulation. While vision-language models (VLMs) have shown promise in generating affordance-aware visual representations, the lack of semantic grounding in canonical spaces and reliance on manual annotations severely limit their ability to capture dynamic semantic-affordance relationships. To address these, we propose Primitive-Aware Semantic Grounding (PASG), a closed-loop framework that introduces: (1) Automatic primitive extraction through geometric feature aggregation, enabling cross-category detection of keypoints and axes; (2) VLM-driven semantic anchoring that dynamically couples geometric primitives with functional affordances and task-relevant description; (3) A spatial-semantic reasoning benchmark and a fine-tuned VLM (Qwen2.5VL-PA). We demonstrate PASG's effectiveness in practical robotic manipulation tasks across diverse scenarios, achieving performance comparable to manual annotations. PASG achieves a finer-grained semantic-affordance understanding of objects, establishing a unified paradigm for bridging geometric primitives with task semantics in robotic manipulation. 

**Abstract (ZH)**: 高阶任务语义与低级几何特征之间的碎片化问题仍然是机器人操作中的一个持久挑战。虽然视觉-语言模型（VLMs）在生成知觉aware视觉表示方面展现了潜力，但语义 anchors 缺乏在标准空间中的定位和对动态语义-知觉关系的依赖于手动注释的限制严重限制了其能力。为了解决这些问题，我们提出了一种名为Primitive-Aware Semantic Grounding (PASG) 的闭环框架，该框架引入了：（1）通过几何特征聚合实现自动基本形态提取，跨类别检测关键点和轴线；（2）由VLM驱动的语义锚定，动态连接几何基本形态与功能知觉和任务相关描述；（3）一种空间-语义推理基准和微调后的VLM（Qwen2.5VL-PA）。我们在多种场景下的实际机器人操作任务中展示了PASG的有效性，其性能与手动注释相当。PASG实现了更精细的物体语义-知觉理解，建立了将几何基本形态与机器人操作任务语义相统一的范式。 

---
# Safety of Embodied Navigation: A Survey 

**Title (ZH)**: 附身导航的安全性：一项综述 

**Authors**: Zixia Wang, Jia Hu, Ronghui Mu  

**Link**: [PDF](https://arxiv.org/pdf/2508.05855)  

**Abstract**: As large language models (LLMs) continue to advance and gain influence, the development of embodied AI has accelerated, drawing significant attention, particularly in navigation scenarios. Embodied navigation requires an agent to perceive, interact with, and adapt to its environment while moving toward a specified target in unfamiliar settings. However, the integration of embodied navigation into critical applications raises substantial safety concerns. Given their deployment in dynamic, real-world environments, ensuring the safety of such systems is critical. This survey provides a comprehensive analysis of safety in embodied navigation from multiple perspectives, encompassing attack strategies, defense mechanisms, and evaluation methodologies. Beyond conducting a comprehensive examination of existing safety challenges, mitigation technologies, and various datasets and metrics that assess effectiveness and robustness, we explore unresolved issues and future research directions in embodied navigation safety. These include potential attack methods, mitigation strategies, more reliable evaluation techniques, and the implementation of verification frameworks. By addressing these critical gaps, this survey aims to provide valuable insights that can guide future research toward the development of safer and more reliable embodied navigation systems. Furthermore, the findings of this study have broader implications for enhancing societal safety and increasing industrial efficiency. 

**Abstract (ZH)**: 大型语言模型(Large Language Models, LLMs)不断发展并产生影响，使嵌入式人工智能的发展加速，特别是在导航场景中引起了广泛关注。嵌入式导航要求智能体在移动向指定目标的过程中，能够感知、交互和适应其环境，尤其是在不熟悉的情境中。然而，将嵌入式导航集成到关键应用中引发了重大的安全问题。鉴于其在动态的实际环境中的部署，确保此类系统的安全性至关重要。本综述从多个角度对嵌入式导航中的安全性进行了全面分析，涵盖攻击策略、防御机制和评估方法。除了对现有安全挑战、缓解技术以及评估效果和鲁棒性的各种数据集和指标进行全面审查之外，我们还探讨了嵌入式导航安全性中的未解决问题和未来研究方向，包括潜在的攻击方法、缓解策略、更可靠的评估技术以及验证框架的实施。通过解决这些关键缺口，本综述旨在为未来研究提供有价值的见解，推动更安全和更可靠的嵌入式导航系统的开发。此外，本研究的发现对增强社会安全和提高工业效率具有更广泛的 implications。 

---
# Towards Transparent Ethical AI: A Roadmap for Trustworthy Robotic Systems 

**Title (ZH)**: 面向透明伦理AI：值得信赖的机器人系统路线图 

**Authors**: Ahmad Farooq, Kamran Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2508.05846)  

**Abstract**: As artificial intelligence (AI) and robotics increasingly permeate society, ensuring the ethical behavior of these systems has become paramount. This paper contends that transparency in AI decision-making processes is fundamental to developing trustworthy and ethically aligned robotic systems. We explore how transparency facilitates accountability, enables informed consent, and supports the debugging of ethical algorithms. The paper outlines technical, ethical, and practical challenges in implementing transparency and proposes novel approaches to enhance it, including standardized metrics, explainable AI techniques, and user-friendly interfaces. This paper introduces a framework that connects technical implementation with ethical considerations in robotic systems, focusing on the specific challenges of achieving transparency in dynamic, real-world contexts. We analyze how prioritizing transparency can impact public trust, regulatory policies, and avenues for future research. By positioning transparency as a fundamental element in ethical AI system design, we aim to add to the ongoing discussion on responsible AI and robotics, providing direction for future advancements in this vital field. 

**Abstract (ZH)**: 随着人工智能（AI）和机器人技术 increasingly 渗透社会，确保这些系统的伦理行为已成为至关重要。本文认为，在机器人系统中实现决策过程的透明度是构建可信赖且伦理对齐的系统的根本。我们探讨了透明度如何促进问责制、使知情同意成为可能，并支持伦理算法的调试。本文概述了实现透明度的技术、伦理和实际挑战，并提出了一些增强透明度的新型方法，包括标准化指标、可解释的AI技术以及用户友好的界面。本文提出了一种框架，将技术实现与机器人系统的伦理考虑联系起来，重点关注在动态的实际情境中实现透明度的具体挑战。我们分析了透明度优先的重要性如何影响公众信任、监管政策以及未来研究的途径。通过将透明度定位为伦理AI系统设计中的基本要素，我们旨在为负责任的AI和机器人技术的持续讨论做出贡献，并为这一关键领域的未来进步提供方向。 

---
# A Humanoid Social Robot as a Teaching Assistant in the Classroom 

**Title (ZH)**: 课堂教学中的人形社会机器人助教 

**Authors**: Thomas Sievers  

**Link**: [PDF](https://arxiv.org/pdf/2508.05646)  

**Abstract**: Although innovation and the support of new technologies are much needed to ease the burden on the education system, social robots in schools to help teachers with educational tasks are rare. Child-Robot Interaction (CRI) could support teachers and add an embodied social component to modern multi-modal and multi-sensory learning environments already in use. The social robot Pepper, connected to the Large Language Model (LLM) ChatGPT, was used in a high school classroom to teach new learning content to groups of students. I tested the technical possibilities with the robot on site and asked the students about their acceptance and perceived usefulness of teaching with the help of a social robot. All participants felt that the robot's presentation of the learning material was appropriate or at least partially appropriate and that its use made sense. 

**Abstract (ZH)**: 尽管创新和支持新技术对于减轻教育系统的负担至关重要，但在学校中使用社会机器人帮助教师完成教学任务的情况却较少见。社会机器人交互（CRI）可以支持教师，并为现有的多模态和多感知学习环境增添实体化的社会要素。通过将社会机器人Pepper与大型语言模型ChatGPT连接，我在一所高中课堂中使用它来为学生群体教授新知识内容。我现场测试了机器人的技术可能性，并要求学生评估使用社会机器人辅助教学的接受度和 perceived 实用性。所有参与者都认为机器人呈现学习材料的方式是适当的或至少部分适当，并认为其使用是有意义的。 

---
