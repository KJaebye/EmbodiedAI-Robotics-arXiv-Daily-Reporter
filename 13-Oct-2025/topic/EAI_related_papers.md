# Zero-shot Structure Learning and Planning for Autonomous Robot Navigation using Active Inference 

**Title (ZH)**: 基于主动推断的自主机器人导航零样本结构学习与规划 

**Authors**: Daria de tinguy, Tim Verbelen, Emilio Gamba, Bart Dhoedt  

**Link**: [PDF](https://arxiv.org/pdf/2510.09574)  

**Abstract**: Autonomous navigation in unfamiliar environments requires robots to simultaneously explore, localise, and plan under uncertainty, without relying on predefined maps or extensive training. We present a biologically inspired, Active Inference-based framework, Active Inference MAPping and Planning (AIMAPP). This model unifies mapping, localisation, and decision-making within a single generative model. Inspired by hippocampal navigation, it uses topological reasoning, place-cell encoding, and episodic memory to guide behaviour. The agent builds and updates a sparse topological map online, learns state transitions dynamically, and plans actions by minimising Expected Free Energy. This allows it to balance goal-directed and exploratory behaviours. We implemented a ROS-compatible navigation system that is sensor and robot-agnostic, capable of integrating with diverse hardware configurations. It operates in a fully self-supervised manner, is resilient to drift, and supports both exploration and goal-directed navigation without any pre-training. We demonstrate robust performance in large-scale real and simulated environments against state-of-the-art planning models, highlighting the system's adaptability to ambiguous observations, environmental changes, and sensor noise. The model offers a biologically inspired, modular solution to scalable, self-supervised navigation in unstructured settings. AIMAPP is available at this https URL. 

**Abstract (ZH)**: 自主导航在陌生环境中的要求是在不确定性下同时探索、定位和规划，无需依赖预定义地图或大量训练。我们提出了一种受生物启发的Active Inference框架--Active Inference MAPping and Planning (AIMAPP)。该模型在单一的生成模型中统一了制图、定位和决策。受海马导航启发，它利用拓扑推理、位置细胞编码和情景记忆来引导行为。代理实时构建和更新稀疏拓扑图，动态学习状态转移，并通过最小化预期自由能来规划行动。这使得代理能够平衡目标导向和探索性行为。我们实现了一个与ROS兼容的导航系统，该系统传感器和机器人无关，能够与各种硬件配置集成。该系统以完全自我监督的方式运行，对漂移具有鲁棒性，并且在无任何预训练的情况下支持探索和目标导向导航。我们在大规模的真实和模拟环境中展示了该系统的稳健性能，突显了系统对模糊观察、环境变化和传感器噪声的适应能力。该模型提供了一种生物启发的、模块化的解决方案，适用于结构化不强的环境下的可扩展和自我监督导航。AIMAPP可在以下链接获取。 

---
# Guiding Energy-Efficient Locomotion through Impact Mitigation Rewards 

**Title (ZH)**: 通过冲击缓解奖励引导能效 locomotion 

**Authors**: Chenghao Wang, Arjun Viswanathan, Eric Sihite, Alireza Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2510.09543)  

**Abstract**: Animals achieve energy-efficient locomotion by their implicit passive dynamics, a marvel that has captivated roboticists for this http URL, methods incorporated Adversarial Motion Prior (AMP) and Reinforcement learning (RL) shows promising progress to replicate Animals' naturalistic motion. However, such imitation learning approaches predominantly capture explicit kinematic patterns, so-called gaits, while overlooking the implicit passive dynamics. This work bridges this gap by incorporating a reward term guided by Impact Mitigation Factor (IMF), a physics-informed metric that quantifies a robot's ability to passively mitigate impacts. By integrating IMF with AMP, our approach enables RL policies to learn both explicit motion trajectories from animal reference motion and the implicit passive dynamic. We demonstrate energy efficiency improvements of up to 32%, as measured by the Cost of Transport (CoT), across both AMP and handcrafted reward structure. 

**Abstract (ZH)**: 动物通过其隐含的被动动力学实现能量高效的运动，这一 marvel 已经吸引了 roboticists 的注意。本工作通过结合由 Impact Mitigation Factor (IMF) 引导的奖励项，填补了这一空白。IMF 是一个基于物理的度量，量化了机器人被动吸收冲击的能力。通过将 IMF 与 Adversarial Motion Prior (AMP) 结合，我们的方法使强化学习策略能够学习动物参考运动的显式运动轨迹和隐含的被动动力学。实验结果表明，本方法在 Cost of Transport (CoT) 测量下实现了高达 32% 的能耗效率提升。 

---
# Dynamic Quadrupedal Legged and Aerial Locomotion via Structure Repurposing 

**Title (ZH)**: 基于结构再利用的动态四足步行和 aerial 运动 

**Authors**: Chenghao Wang, Kaushik Venkatesh Krishnamurthy, Shreyansh Pitroda, Adarsh Salagame, Ioannis Mandralis, Eric Sihite, Alireza Ramezani, Morteza Gharib  

**Link**: [PDF](https://arxiv.org/pdf/2510.09526)  

**Abstract**: Multi-modal ground-aerial robots have been extensively studied, with a significant challenge lying in the integration of conflicting requirements across different modes of operation. The Husky robot family, developed at Northeastern University, and specifically the Husky v.2 discussed in this study, addresses this challenge by incorporating posture manipulation and thrust vectoring into multi-modal locomotion through structure repurposing. This quadrupedal robot features leg structures that can be repurposed for dynamic legged locomotion and flight. In this paper, we present the hardware design of the robot and report primary results on dynamic quadrupedal legged locomotion and hovering. 

**Abstract (ZH)**: 多模态地面-空中机器人得到了广泛研究，不同操作模式之间相互冲突的要求的整合是一个显著挑战。东北大学开发的Husky机器人家族，特别是在本研究中讨论的Husky v.2，通过结构再利用将姿态操控和推力向量集成到多模态运动中，该四足机器人具有可再利用于动态腿足运动和飞行的腿部结构。在本文中，我们介绍了机器人的硬件设计，并报告了动态四足腿足运动和悬停的主要研究结果。 

---
# FOGMACHINE -- Leveraging Discrete-Event Simulation and Scene Graphs for Modeling Hierarchical, Interconnected Environments under Partial Observations from Mobile Agents 

**Title (ZH)**: FOGMACHINE——利用离散事件仿真和场景图建模基于移动代理部分观测下的层次化互联环境 

**Authors**: Lars Ohnemus, Nils Hantke, Max Weißer, Kai Furmans  

**Link**: [PDF](https://arxiv.org/pdf/2510.09483)  

**Abstract**: Dynamic Scene Graphs (DSGs) provide a structured representation of hierarchical, interconnected environments, but current approaches struggle to capture stochastic dynamics, partial observability, and multi-agent activity. These aspects are critical for embodied AI, where agents must act under uncertainty and delayed perception. We introduce FOGMACHINE , an open-source framework that fuses DSGs with discrete-event simulation to model object dynamics, agent observations, and interactions at scale. This setup enables the study of uncertainty propagation, planning under limited perception, and emergent multi-agent behavior. Experiments in urban scenarios illustrate realistic temporal and spatial patterns while revealing the challenges of belief estimation under sparse observations. By combining structured representations with efficient simulation, FOGMACHINE establishes an effective tool for benchmarking, model training, and advancing embodied AI in complex, uncertain environments. 

**Abstract (ZH)**: FOGMACHINE：将动态场景图与离散事件仿真融合以建模大规模物体动力学、代理观察和交互 

---
# Failure Prediction at Runtime for Generative Robot Policies 

**Title (ZH)**: 运行时生成型机器人策略的失败预测 

**Authors**: Ralf Römer, Adrian Kobras, Luca Worbis, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2510.09459)  

**Abstract**: Imitation learning (IL) with generative models, such as diffusion and flow matching, has enabled robots to perform complex, long-horizon tasks. However, distribution shifts from unseen environments or compounding action errors can still cause unpredictable and unsafe behavior, leading to task failure. Early failure prediction during runtime is therefore essential for deploying robots in human-centered and safety-critical environments. We propose FIPER, a general framework for Failure Prediction at Runtime for generative IL policies that does not require failure data. FIPER identifies two key indicators of impending failure: (i) out-of-distribution (OOD) observations detected via random network distillation in the policy's embedding space, and (ii) high uncertainty in generated actions measured by a novel action-chunk entropy score. Both failure prediction scores are calibrated using a small set of successful rollouts via conformal prediction. A failure alarm is triggered when both indicators, aggregated over short time windows, exceed their thresholds. We evaluate FIPER across five simulation and real-world environments involving diverse failure modes. Our results demonstrate that FIPER better distinguishes actual failures from benign OOD situations and predicts failures more accurately and earlier than existing methods. We thus consider this work an important step towards more interpretable and safer generative robot policies. Code, data and videos are available at this https URL. 

**Abstract (ZH)**: 基于生成模型的模仿学习故障预测框架FIPER 

---
# Bridging Research and Practice in Simulation-based Testing of Industrial Robot Navigation Systems 

**Title (ZH)**: 基于仿真测试的工业机器人导航系统研究与实践桥梁构建 

**Authors**: Sajad Khatiri, Francisco Eli Vina Barrientos, Maximilian Wulf, Paolo Tonella, Sebastiano Panichella  

**Link**: [PDF](https://arxiv.org/pdf/2510.09396)  

**Abstract**: Ensuring robust robotic navigation in dynamic environments is a key challenge, as traditional testing methods often struggle to cover the full spectrum of operational requirements. This paper presents the industrial adoption of Surrealist, a simulation-based test generation framework originally for UAVs, now applied to the ANYmal quadrupedal robot for industrial inspection. Our method uses a search-based algorithm to automatically generate challenging obstacle avoidance scenarios, uncovering failures often missed by manual testing. In a pilot phase, generated test suites revealed critical weaknesses in one experimental algorithm (40.3% success rate) and served as an effective benchmark to prove the superior robustness of another (71.2% success rate). The framework was then integrated into the ANYbotics workflow for a six-month industrial evaluation, where it was used to test five proprietary algorithms. A formal survey confirmed its value, showing it enhances the development process, uncovers critical failures, provides objective benchmarks, and strengthens the overall verification pipeline. 

**Abstract (ZH)**: 确保在动态环境中的稳健机器人导航是一个关键挑战，传统的测试方法往往难以覆盖所有操作要求。本文介绍了将Surrealist这一基于仿真测试生成框架从无人机扩展到ANYmal四足机器人进行工业检测的应用。我们的方法使用基于搜索的算法自动生成具有挑战性的障碍物回避场景，揭示了许多手动测试常忽视的失败案例。在试点阶段，生成的测试套件揭示了一种实验算法的关键弱点（成功率为40.3%），并作为有效的基准测试证明了另一种算法的优异稳健性（成功率为71.2%）。随后，该框架被集成到ANYbotics的工作流程中，进行了为期六个月的工业评估，用于检测五种专有算法。正式调研确认了其价值，表明它能够改善开发过程、揭示关键失败、提供客观的基准测试并强化整体验证流程。 

---
# Placeit! A Framework for Learning Robot Object Placement Skills 

**Title (ZH)**: Placeit! 一种学习机器人物体放置技能的框架 

**Authors**: Amina Ferrad, Johann Huber, François Hélénon, Julien Gleyze, Mahdi Khoramshahi, Stéphane Doncieux  

**Link**: [PDF](https://arxiv.org/pdf/2510.09267)  

**Abstract**: Robotics research has made significant strides in learning, yet mastering basic skills like object placement remains a fundamental challenge. A key bottleneck is the acquisition of large-scale, high-quality data, which is often a manual and laborious process. Inspired by Graspit!, a foundational work that used simulation to automatically generate dexterous grasp poses, we introduce Placeit!, an evolutionary-computation framework for generating valid placement positions for rigid objects. Placeit! is highly versatile, supporting tasks from placing objects on tables to stacking and inserting them. Our experiments show that by leveraging quality-diversity optimization, Placeit! significantly outperforms state-of-the-art methods across all scenarios for generating diverse valid poses. A pick&place pipeline built on our framework achieved a 90% success rate over 120 real-world deployments. This work positions Placeit! as a powerful tool for open-environment pick-and-place tasks and as a valuable engine for generating the data needed to train simulation-based foundation models in robotics. 

**Abstract (ZH)**: 机器人研究在学习方面取得了显著进展，但在掌握如物体放置等基本技能方面仍然存在根本挑战。一个关键瓶颈是获取大规模、高质量的数据，这通常是手动且劳动密集型的过程。受Graspit!的启发，该工作利用模拟自动生成灵巧的抓取姿态，我们提出了Placeit!，一种用于生成刚体物体有效放置位置的进化计算框架。Placeit!极具 versatility，支持从将物体放在桌子上到堆叠和插入等多种任务。我们的实验表明，通过利用品质多样性优化，Placeit!在所有场景下生成多样化有效姿态方面显著优于最先进的方法。基于我们框架构建的拾取和放置流水线在120次真实世界部署中实现了90%的成功率。本工作将Placeit!定位为开放环境拾取和放置任务的强大工具，并作为生成训练基于模拟的基础模型所需数据的价值引擎。 

---
# Obstacle Avoidance using Dynamic Movement Primitives and Reinforcement Learning 

**Title (ZH)**: 基于动态运动原始和强化学习的障碍避免 

**Authors**: Dominik Urbaniak, Alejandro Agostini, Pol Ramon, Jan Rosell, Raúl Suárez, Michael Suppa  

**Link**: [PDF](https://arxiv.org/pdf/2510.09254)  

**Abstract**: Learning-based motion planning can quickly generate near-optimal trajectories. However, it often requires either large training datasets or costly collection of human demonstrations. This work proposes an alternative approach that quickly generates smooth, near-optimal collision-free 3D Cartesian trajectories from a single artificial demonstration. The demonstration is encoded as a Dynamic Movement Primitive (DMP) and iteratively reshaped using policy-based reinforcement learning to create a diverse trajectory dataset for varying obstacle configurations. This dataset is used to train a neural network that takes as inputs the task parameters describing the obstacle dimensions and location, derived automatically from a point cloud, and outputs the DMP parameters that generate the trajectory. The approach is validated in simulation and real-robot experiments, outperforming a RRT-Connect baseline in terms of computation and execution time, as well as trajectory length, while supporting multi-modal trajectory generation for different obstacle geometries and end-effector dimensions. Videos and the implementation code are available at this https URL. 

**Abstract (ZH)**: 基于学习的运动规划可以快速生成接近最优的轨迹。然而，它通常需要大型训练数据集或昂贵的人类示范收集。本工作提出了一种替代方法，可以从单个人工示范快速生成平滑、接近最优且无碰撞的3D笛卡尔轨迹。示范被编码为动态运动本原（DMP），并通过基于策略的强化学习迭代重塑，以生成适用于不同障碍配置的多样轨迹数据集。该数据集用于训练神经网络，该网络接受描述障碍尺寸和位置的任务参数（这些参数从点云中自动提取）作为输入，并输出生成轨迹的DMP参数。该方法在仿真和真实机器人实验中得到了验证，相较于RRT-Connect基线，在计算时间和执行时间、轨迹长度方面表现更优，并支持不同障碍几何形状和末端执行器尺寸的多模态轨迹生成。视频和实现代码可在以下链接获取。 

---
# HANDO: Hierarchical Autonomous Navigation and Dexterous Omni-loco-manipulation 

**Title (ZH)**: HANDO：分层自主导航与 omniloquent 操作 manipulating 

**Authors**: Jingyuan Sun, Chaoran Wang, Mingyu Zhang, Cui Miao, Hongyu Ji, Zihan Qu, Han Sun, Bing Wang, Qingyi Si  

**Link**: [PDF](https://arxiv.org/pdf/2510.09221)  

**Abstract**: Seamless loco-manipulation in unstructured environments requires robots to leverage autonomous exploration alongside whole-body control for physical interaction. In this work, we introduce HANDO (Hierarchical Autonomous Navigation and Dexterous Omni-loco-manipulation), a two-layer framework designed for legged robots equipped with manipulators to perform human-centered mobile manipulation tasks. The first layer utilizes a goal-conditioned autonomous exploration policy to guide the robot to semantically specified targets, such as a black office chair in a dynamic environment. The second layer employs a unified whole-body loco-manipulation policy to coordinate the arm and legs for precise interaction tasks-for example, handing a drink to a person seated on the chair. We have conducted an initial deployment of the navigation module, and will continue to pursue finer-grained deployment of whole-body loco-manipulation. 

**Abstract (ZH)**: 无缝的非结构化环境操作要求机器人结合全身控制和自主探索来进行物理交互。本文介绍了HANDO（层次自主导航与全动 dexterous Omni-loco-manipulation），这是一种为配备 manipulator 的 legged 机器人设计的两层框架，用于执行以人类为中心的移动操作任务。第一层利用目标条件的自主探索策略引导机器人到达语义指定的目标，如动态环境中的一把黑色办公椅。第二层采用统一的全身操作策略来协调手臂和腿部进行精确的交互任务，例如将饮料递给坐在椅子上的 person。我们已经初步部署了导航模块，并将继续推进全身操作的更精细部署。 

---
# Decentralized Multi-Robot Relative Navigation in Unknown, Structurally Constrained Environments under Limited Communication 

**Title (ZH)**: 未知且结构受限环境中受限通信条件下分散式多机器人相对导航 

**Authors**: Zihao Mao, Yunheng Wang, Yunting Ji, Yi Yang, Wenjie Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.09188)  

**Abstract**: Multi-robot navigation in unknown, structurally constrained, and GPS-denied environments presents a fundamental trade-off between global strategic foresight and local tactical agility, particularly under limited communication. Centralized methods achieve global optimality but suffer from high communication overhead, while distributed methods are efficient but lack the broader awareness to avoid deadlocks and topological traps. To address this, we propose a fully decentralized, hierarchical relative navigation framework that achieves both strategic foresight and tactical agility without a unified coordinate system. At the strategic layer, robots build and exchange lightweight topological maps upon opportunistic encounters. This process fosters an emergent global awareness, enabling the planning of efficient, trap-avoiding routes at an abstract level. This high-level plan then inspires the tactical layer, which operates on local metric information. Here, a sampling-based escape point strategy resolves dense spatio-temporal conflicts by generating dynamically feasible trajectories in real time, concurrently satisfying tight environmental and kinodynamic constraints. Extensive simulations and real-world experiments demonstrate that our system significantly outperforms in success rate and efficiency, especially in communication-limited environments with complex topological structures. 

**Abstract (ZH)**: 多机器人在未知、结构受限且GPS受限环境中的导航面临全局战略预见性和局部战术敏捷性之间的基本权衡，尤其是在通信受限的情况下。集中式方法能够实现全局最优但通信开销高，而分布式方法虽然高效但缺乏整体感知能力以避免死锁和拓扑陷阱。为了解决这一问题，我们提出了一种完全分散的分层相对导航框架，能够在无统一坐标系的情况下实现战略预见性和战术敏捷性。在战略层面上，机器人通过机会性邂逅构建并交换轻量级拓扑地图，这一过程促进了全局意识的涌现，使机器人能够在抽象层面规划高效的、避开陷阱的路线。该高层计划随后启发战术层，后者基于局部度量信息操作。在战术层面上，基于采样的逃逸点策略通过实时生成动态可行轨迹解决了密集的空间时间冲突，同时满足严格的环境和动力学约束。大量仿真和实际实验表明，我们的系统在成功率和效率方面显著优于现有方法，特别是在通信受限且拓扑结构复杂的环境中。 

---
# When a Robot is More Capable than a Human: Learning from Constrained Demonstrators 

**Title (ZH)**: 当机器人能力超过人类：从受限示范者学习 

**Authors**: Xinhu Li, Ayush Jain, Zhaojing Yang, Yigit Korkmaz, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2510.09096)  

**Abstract**: Learning from demonstrations enables experts to teach robots complex tasks using interfaces such as kinesthetic teaching, joystick control, and sim-to-real transfer. However, these interfaces often constrain the expert's ability to demonstrate optimal behavior due to indirect control, setup restrictions, and hardware safety. For example, a joystick can move a robotic arm only in a 2D plane, even though the robot operates in a higher-dimensional space. As a result, the demonstrations collected by constrained experts lead to suboptimal performance of the learned policies. This raises a key question: Can a robot learn a better policy than the one demonstrated by a constrained expert? We address this by allowing the agent to go beyond direct imitation of expert actions and explore shorter and more efficient trajectories. We use the demonstrations to infer a state-only reward signal that measures task progress, and self-label reward for unknown states using temporal interpolation. Our approach outperforms common imitation learning in both sample efficiency and task completion time. On a real WidowX robotic arm, it completes the task in 12 seconds, 10x faster than behavioral cloning, as shown in real-robot videos on this https URL . 

**Abstract (ZH)**: 通过示范学习使专家能够使用诸如 kinesthetic 教学、摇杆控制和模拟到现实的转移等界面来教机器人执行复杂任务。然而，这些界面往往会由于间接控制、设置限制和硬件安全等因素而限制专家展示最优行为的能力。例如，摇杆只能使机器人手臂在二维平面上移动，尽管机器人在更高维度的空间中操作。因此，受限专家收集的示范会导致学习到的策略性能不佳。这引发了一个关键问题：机器人能否学习到比受限专家展示的更好的策略？我们通过允许代理超越直接模仿专家动作并探索更短且更高效的轨迹来解决这一问题。我们使用示范来推断一个仅基于状态的奖励信号以衡量任务进度，并使用时间插值对未知状态进行自我标记奖励。我们的方法在样本效率和任务完成时间上均优于常见的模仿学习。在一个真实的 WidowX 机器人手臂上，该方法在12秒内完成任务，比行为克隆快10倍，如在本链接给出的实际机器人视频中所示。 

---
# Training Models to Detect Successive Robot Errors from Human Reactions 

**Title (ZH)**: 训练模型从人类反应中检测机器人连续错误 

**Authors**: Shannon Liu, Maria Teresa Parreira, Wendy Ju  

**Link**: [PDF](https://arxiv.org/pdf/2510.09080)  

**Abstract**: As robots become more integrated into society, detecting robot errors is essential for effective human-robot interaction (HRI). When a robot fails repeatedly, how can it know when to change its behavior? Humans naturally respond to robot errors through verbal and nonverbal cues that intensify over successive failures-from confusion and subtle speech changes to visible frustration and impatience. While prior work shows that human reactions can indicate robot failures, few studies examine how these evolving responses reveal successive failures. This research uses machine learning to recognize stages of robot failure from human reactions. In a study with 26 participants interacting with a robot that made repeated conversational errors, behavioral features were extracted from video data to train models for individual users. The best model achieved 93.5% accuracy for detecting errors and 84.1% for classifying successive failures. Modeling the progression of human reactions enhances error detection and understanding of repeated interaction breakdowns in HRI. 

**Abstract (ZH)**: 随着机器人越来越多地融入社会，检测机器人错误对于有效的人机交互（HRI）至关重要。当机器人反复出错时，它如何知道何时改变行为？人类会通过言语和非言语信号自然地响应机器人的错误，这些信号会在连续的失败中增强，从困惑和微妙的言语变化到明显的沮丧和不耐烦。虽然先前的研究表明人类的反应可以指示机器人失败，但很少有研究探讨这些逐渐变化的反应如何揭示连续的失败。本研究使用机器学习技术从人类反应中识别机器人的故障阶段。在一项涉及26名参与者与多次交流错误的机器人互动的研究中，从视频数据中提取的行为特征用于训练个体用户的模型。最佳模型在检测错误方面的准确率为93.5%，在分类连续失败方面的准确率为84.1%。建模人类反应的进展可以提高错误检测能力，并帮助理解HRI中反复互动的失败过程。 

---
# iMoWM: Taming Interactive Multi-Modal World Model for Robotic Manipulation 

**Title (ZH)**: iMoWM: 控制交互多模态世界模型以实现机器人操纵 

**Authors**: Chuanrui Zhang, Zhengxian Wu, Guanxing Lu, Yansong Tang, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09036)  

**Abstract**: Learned world models hold significant potential for robotic manipulation, as they can serve as simulator for real-world interactions. While extensive progress has been made in 2D video-based world models, these approaches often lack geometric and spatial reasoning, which is essential for capturing the physical structure of the 3D world. To address this limitation, we introduce iMoWM, a novel interactive world model designed to generate color images, depth maps, and robot arm masks in an autoregressive manner conditioned on actions. To overcome the high computational cost associated with three-dimensional information, we propose MMTokenizer, which unifies multi-modal inputs into a compact token representation. This design enables iMoWM to leverage large-scale pretrained VideoGPT models while maintaining high efficiency and incorporating richer physical information. With its multi-modal representation, iMoWM not only improves the visual quality of future predictions but also serves as an effective simulator for model-based reinforcement learning (MBRL) and facilitates real-world imitation learning. Extensive experiments demonstrate the superiority of iMoWM across these tasks, showcasing the advantages of multi-modal world modeling for robotic manipulation. Homepage: this https URL 

**Abstract (ZH)**: Learned 世界模型在机器人操作中具有巨大潜力，因为它们可以作为现实世界交互的模拟器。虽然基于2D视频的世界模型取得了广泛进展，但这些方法往往缺乏几何和空间推理，这对于捕捉三维世界的物理结构至关重要。为了解决这一局限，我们引入了iMoWM，这是一种新颖的交互式世界模型，设计用于在动作条件下自回归生成彩色图像、深度图和机器人臂掩码。为了克服与三维信息相关的高计算成本，我们提出了MMTokenizer，将多模态输入统一为紧凑的token表示。这种设计使得iMoWM能够利用大规模预训练的VideoGPT模型，同时保持高效并引入更丰富的物理信息。凭借其多模态表示，iMoWM不仅提高了未来预测的视觉质量，还作为基于模型的强化学习（MBRL）的有效模拟器，并促进了现实世界的模仿学习。广泛的经验表明，iMoWM在这些任务中的优越性，展示了多模态世界建模在机器人操作中的优势。 Homepage: 这个=https://nitro共和国论文地址 

---
# Model-Based Lookahead Reinforcement Learning for in-hand manipulation 

**Title (ZH)**: 基于模型的前瞻强化学习在手内操作 

**Authors**: Alexandre Lopes, Catarina Barata, Plinio Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2510.08884)  

**Abstract**: In-Hand Manipulation, as many other dexterous tasks, remains a difficult challenge in robotics by combining complex dynamic systems with the capability to control and manoeuvre various objects using its actuators. This work presents the application of a previously developed hybrid Reinforcement Learning (RL) Framework to In-Hand Manipulation task, verifying that it is capable of improving the performance of the task. The model combines concepts of both Model-Free and Model-Based Reinforcement Learning, by guiding a trained policy with the help of a dynamic model and value-function through trajectory evaluation, as done in Model Predictive Control. This work evaluates the performance of the model by comparing it with the policy that will be guided. To fully explore this, various tests are performed using both fully-actuated and under-actuated simulated robotic hands to manipulate different objects for a given task. The performance of the model will also be tested for generalization tests, by changing the properties of the objects in which both the policy and dynamic model were trained, such as density and size, and additionally by guiding a trained policy in a certain object to perform the same task in a different one. The results of this work show that, given a policy with high average reward and an accurate dynamic model, the hybrid framework improves the performance of in-hand manipulation tasks for most test cases, even when the object properties are changed. However, this improvement comes at the expense of increasing the computational cost, due to the complexity of trajectory evaluation. 

**Abstract (ZH)**: 手持操作：一种结合混合强化学习框架的复杂动态系统方法以提高操作性能 

---
# CDE: Concept-Driven Exploration for Reinforcement Learning 

**Title (ZH)**: 概念驱动的探索在强化学习中的应用 

**Authors**: Le Mao, Andrew H. Liu, Renos Zabounidis, Zachary Kingston, Joseph Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2510.08851)  

**Abstract**: Intelligent exploration remains a critical challenge in reinforcement learning (RL), especially in visual control tasks. Unlike low-dimensional state-based RL, visual RL must extract task-relevant structure from raw pixels, making exploration inefficient. We propose Concept-Driven Exploration (CDE), which leverages a pre-trained vision-language model (VLM) to generate object-centric visual concepts from textual task descriptions as weak, potentially noisy supervisory signals. Rather than directly conditioning on these noisy signals, CDE trains a policy to reconstruct the concepts via an auxiliary objective, using reconstruction accuracy as an intrinsic reward to guide exploration toward task-relevant objects. Because the policy internalizes these concepts, VLM queries are only needed during training, reducing dependence on external models during deployment. Across five challenging simulated visual manipulation tasks, CDE achieves efficient, targeted exploration and remains robust to noisy VLM predictions. Finally, we demonstrate real-world transfer by deploying CDE on a Franka Research 3 arm, attaining an 80\% success rate in a real-world manipulation task. 

**Abstract (ZH)**: 智能探索仍是强化学习（RL）中的一个关键挑战，特别是在视觉控制任务中。与基于低维状态的RL不同，视觉RL必须从原始像素中提取相关的任务结构，导致探索效率低下。我们提出了一种概念驱动的探索（CDE），该方法利用预训练的视觉-语言模型（VLM）生成以对象为中心的视觉概念，作为弱的、可能含噪声的监督信号。CDE不直接依赖这些噪声信号，而是训练一个策略通过辅助目标重建这些概念，使用重建准确性作为内在奖励，指导探索向相关任务对象靠拢。由于策略内化了这些概念，在部署时仅需使用VLM查询，减少了对外部模型的依赖。在五个具有挑战性的模拟视觉操作任务中，CDE实现了高效的、有针对性的探索，并且对于含噪声的VLM预测具有鲁棒性。最后，我们通过在Franka Research 3臂上部署CDE展示了实际应用的迁移能力，实现了80%的成功率。 

---
# Adaptive Motion Planning via Contact-Based Intent Inference for Human-Robot Collaboration 

**Title (ZH)**: 基于接触意图推断的人机协作自适应运动规划 

**Authors**: Jiurun Song, Xiao Liang, Minghui Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.08811)  

**Abstract**: Human-robot collaboration (HRC) requires robots to adapt their motions to human intent to ensure safe and efficient cooperation in shared spaces. Although large language models (LLMs) provide high-level reasoning for inferring human intent, their application to reliable motion planning in HRC remains challenging. Physical human-robot interaction (pHRI) is intuitive but often relies on continuous kinesthetic guidance, which imposes burdens on operators. To address these challenges, a contact-informed adaptive motion-planning framework is introduced to infer human intent directly from physical contact and employ the inferred intent for online motion correction in HRC. First, an optimization-based force estimation method is proposed to infer human-intended contact forces and locations from joint torque measurements and a robot dynamics model, thereby reducing cost and installation complexity while enabling whole-body sensitivity. Then, a torque-based contact detection mechanism with link-level localization is introduced to reduce the optimization search space and to enable real-time estimation. Subsequently, a contact-informed adaptive motion planner is developed to infer human intent from contacts and to replan robot motion online, while maintaining smoothness and adapting to human corrections. Finally, experiments on a 7-DOF manipulator are conducted to demonstrate the accuracy of the proposed force estimation method and the effectiveness of the contact-informed adaptive motion planner under perception uncertainty in HRC. 

**Abstract (ZH)**: Human-机器人协作中的接触导向自适应运动规划框架 

---
# Humanoid Everyday: A Comprehensive Robotic Dataset for Open-World Humanoid Manipulation 

**Title (ZH)**: 仿人日常：开放世界仿人操作的综合机器人数据集 

**Authors**: Zhenyu Zhao, Hongyi Jing, Xiawei Liu, Jiageng Mao, Abha Jha, Hanwen Yang, Rong Xue, Sergey Zakharor, Vitor Guizilini, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08807)  

**Abstract**: From loco-motion to dextrous manipulation, humanoid robots have made remarkable strides in demonstrating complex full-body capabilities. However, the majority of current robot learning datasets and benchmarks mainly focus on stationary robot arms, and the few existing humanoid datasets are either confined to fixed environments or limited in task diversity, often lacking human-humanoid interaction and lower-body locomotion. Moreover, there are a few standardized evaluation platforms for benchmarking learning-based policies on humanoid data. In this work, we present Humanoid Everyday, a large-scale and diverse humanoid manipulation dataset characterized by extensive task variety involving dextrous object manipulation, human-humanoid interaction, locomotion-integrated actions, and more. Leveraging a highly efficient human-supervised teleoperation pipeline, Humanoid Everyday aggregates high-quality multimodal sensory data, including RGB, depth, LiDAR, and tactile inputs, together with natural language annotations, comprising 10.3k trajectories and over 3 million frames of data across 260 tasks across 7 broad categories. In addition, we conduct an analysis of representative policy learning methods on our dataset, providing insights into their strengths and limitations across different task categories. For standardized evaluation, we introduce a cloud-based evaluation platform that allows researchers to seamlessly deploy their policies in our controlled setting and receive performance feedback. By releasing Humanoid Everyday along with our policy learning analysis and a standardized cloud-based evaluation platform, we intend to advance research in general-purpose humanoid manipulation and lay the groundwork for more capable and embodied robotic agents in real-world scenarios. Our dataset, data collection code, and cloud evaluation website are made publicly available on our project website. 

**Abstract (ZH)**: 从移动到灵巧操作，人类形机器人在展示全身复杂能力方面取得了显著进展。然而，当前大多数机器人学习数据集和基准主要集中在静止的机器人手臂上，现有的少数人类形机器人数据集要么局限于固定环境，要么在任务多样性方面受限，往往缺乏人-机器人交互和下肢移动。此外，尚缺乏标准化的评估平台用于在人类形机器人数据上benchmark基于学习的策略。本工作中，我们提出Humanoid Everyday，这是一个大规模且多样化的灵巧操作人形机器人数据集，涵盖了广泛的多任务、灵巧物体操作、人-机器人交互、集成移动动作等。通过高效率的人工监督远程操作流水线，Humanoid Everyday集成了高质量的多模态感官数据，包括RGB、深度、LiDAR和触觉输入，以及自然语言注释，包含10300条轨迹和超过300万帧数据，覆盖7个广泛类别下的260个任务。此外，我们对代表性的策略学习方法在我们的数据集上的表现进行了分析，提供了不同任务类别下的强项和局限性的见解。为了实现标准化评估，我们引入了一个基于云的评估平台，研究人员可以无缝部署其策略并在我们的受控环境中进行测试，获取性能反馈。通过发布Humanoid Everyday数据集、策略学习分析以及标准化的基于云的评估平台，我们旨在推动通用型人形机器人操作研究，并为更擅长且具备身体性的机器人代理在真实场景中的应用奠定基础。我们的数据集、数据采集代码和基于云的评估网站已公开发布在我们的项目网站上。 

---
# Geometry-aware Policy Imitation 

**Title (ZH)**: 几何感知策略模仿 

**Authors**: Yiming Li, Nael Darwiche, Amirreza Razmjoo, Sichao Liu, Yilun Du, Auke Ijspeert, Sylvain Calinon  

**Link**: [PDF](https://arxiv.org/pdf/2510.08787)  

**Abstract**: We propose a Geometry-aware Policy Imitation (GPI) approach that rethinks imitation learning by treating demonstrations as geometric curves rather than collections of state-action samples. From these curves, GPI derives distance fields that give rise to two complementary control primitives: a progression flow that advances along expert trajectories and an attraction flow that corrects deviations. Their combination defines a controllable, non-parametric vector field that directly guides robot behavior. This formulation decouples metric learning from policy synthesis, enabling modular adaptation across low-dimensional robot states and high-dimensional perceptual inputs. GPI naturally supports multimodality by preserving distinct demonstrations as separate models and allows efficient composition of new demonstrations through simple additions to the distance field. We evaluate GPI in simulation and on real robots across diverse tasks. Experiments show that GPI achieves higher success rates than diffusion-based policies while running 20 times faster, requiring less memory, and remaining robust to perturbations. These results establish GPI as an efficient, interpretable, and scalable alternative to generative approaches for robotic imitation learning. Project website: this https URL 

**Abstract (ZH)**: 我们提出了一种几何感知策略模仿（GPI）方法，该方法重新思考模仿学习，将演示视为几何曲线而非状态-动作样本集合。从这些曲线上，GPI 导出了距离场，进而生成两种互补的控制原语：一种进展流用于沿专家轨迹推进，另一种吸引流用于纠正偏差。这两种流的结合定义了一个可控的、非参数化的向量场，可以直接引导机器人行为。此形式化方法将度量学习与策略合成脱钩，从而在低维机器人状态和高维感知输入之间实现模块化的适应。GPI 自然支持多模态性，通过保持不同演示为独立模型而得以实现，并可通过简单扩展距离场来进行新的演示高效组合。我们在仿真和真实机器人上对GPI进行了跨任务评估。实验表明，与基于扩散的方法相比，GPI 在成功率为后者 20 倍的同时，运行速度更快、所需内存更少，并且对扰动具有更强的鲁棒性。这些结果确立了GPI作为一种高效、可解释和可扩展的机器人模仿学习替代方案的地位。项目网站：this https URL 

---
# Whole Body Model Predictive Control for Spin-Aware Quadrupedal Table Tennis 

**Title (ZH)**: 基于Spin感知的四足乒乓球全身模型预测控制 

**Authors**: David Nguyen, Zulfiqar Zaidi, Kevin Karol, Jessica Hodgins, Zhaoming Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.08754)  

**Abstract**: Developing table tennis robots that mirror human speed, accuracy, and ability to predict and respond to the full range of ball spins remains a significant challenge for legged robots. To demonstrate these capabilities we present a system to play dynamic table tennis for quadrupedal robots that integrates high speed perception, trajectory prediction, and agile control. Our system uses external cameras for high-speed ball localization, physical models with learned residuals to infer spin and predict trajectories, and a novel model predictive control (MPC) formulation for agile full-body control. Notably, a continuous set of stroke strategies emerge automatically from different ball return objectives using this control paradigm. We demonstrate our system in the real world on a Spot quadruped, evaluate accuracy of each system component, and exhibit coordination through the system's ability to aim and return balls with varying spin types. As a further demonstration, the system is able to rally with human players. 

**Abstract (ZH)**: 开发能够在速度、准确性和预测及应对全范围球 spin 方面媲美人手的乒乓球机器人仍是一项重大挑战，尤其是对于 legged 机器人。为了展示这些能力，我们提出了一种四足机器人进行动态乒乓球比赛的系统，该系统结合了高精度感知、轨迹预测和敏捷控制。我们的系统使用外部摄像头进行高速球定位，使用结合学习残差的物理模型来推断旋转并预测轨迹，并采用一种新型的模型预测控制（MPC）方法进行敏捷整体控制。值得注意的是，通过这种方法，一系列连续的击球策略能够自动从不同的回球目标中产生。我们在 Spot 四足机器人上展示了该系统，在每个系统组件的准确性评估中展示了协调能力，并通过系统能够根据不同的旋转类型击球和还击来展示协调能力。此外，该系统能够与人类玩家进行对打。 

---
# PhysToolBench: Benchmarking Physical Tool Understanding for MLLMs 

**Title (ZH)**: PhysToolBench: 评估MLLMs 物理工具理解能力的基准测试 

**Authors**: Zixin Zhang, Kanghao Chen, Xingwang Lin, Lutao Jiang, Xu Zheng, Yuanhuiyi Lyu, Litao Guo, Yinchuan Li, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09507)  

**Abstract**: The ability to use, understand, and create tools is a hallmark of human intelligence, enabling sophisticated interaction with the physical world. For any general-purpose intelligent agent to achieve true versatility, it must also master these fundamental skills. While modern Multimodal Large Language Models (MLLMs) leverage their extensive common knowledge for high-level planning in embodied AI and in downstream Vision-Language-Action (VLA) models, the extent of their true understanding of physical tools remains unquantified. To bridge this gap, we present PhysToolBench, the first benchmark dedicated to evaluating the comprehension of physical tools by MLLMs. Our benchmark is structured as a Visual Question Answering (VQA) dataset comprising over 1,000 image-text pairs. It assesses capabilities across three distinct difficulty levels: (1) Tool Recognition: Requiring the recognition of a tool's primary function. (2) Tool Understanding: Testing the ability to grasp the underlying principles of a tool's operation. (3) Tool Creation: Challenging the model to fashion a new tool from surrounding objects when conventional options are unavailable. Our comprehensive evaluation of 32 MLLMs-spanning proprietary, open-source, specialized embodied, and backbones in VLAs-reveals a significant deficiency in tool understanding. Furthermore, we provide an in-depth analysis and propose preliminary solutions. Code and dataset are publicly available. 

**Abstract (ZH)**: 物理工具理解基准： multimodal大型语言模型在物理工具理解上的评估 

---
# BEAR: Benchmarking and Enhancing Multimodal Language Models for Atomic Embodied Capabilities 

**Title (ZH)**: BEAR: 评估与增强多模态语言模型的原子体态能力 

**Authors**: Yu Qi, Haibo Zhao, Ziyu Guo, Siyuan Ma, Ziyan Chen, Yaokun Han, Renrui Zhang, Zitiantao Lin, Shiji Xin, Yijian Huang, Kai Cheng, Peiheng Wang, Jiazheng Liu, Jiayi Zhang, Yizhe Zhu, Wenqing Wang, Yiran Qin, Xupeng Zhu, Haojie Huang, Lawson L.S. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.08759)  

**Abstract**: Embodied capabilities refer to a suite of fundamental abilities for an agent to perceive, comprehend, and interact with the physical world. While multimodal large language models (MLLMs) show promise as embodied agents, a thorough and systematic evaluation of their embodied capabilities remains underexplored, as existing benchmarks primarily focus on specific domains such as planning or spatial understanding. To bridge this gap, we introduce BEAR, a comprehensive and fine-grained benchmark that evaluates MLLMs on atomic embodied capabilities. BEAR comprises 4,469 interleaved image-video-text entries across 14 domains in 6 categories, including tasks from low-level pointing, trajectory understanding, spatial reasoning, to high-level planning. Extensive evaluation results of 20 representative MLLMs reveal their persistent limitations across all domains of embodied capabilities. To tackle the shortfall, we propose BEAR-Agent, a multimodal conversable agent that integrates pretrained vision models to strengthen MLLM perception, 3D understanding, and planning capabilities. It substantially enhances MLLM performance across diverse embodied capabilities on BEAR, yielding a 9.12% absolute gain and a relative improvement of 17.5% on GPT-5. Furthermore, our experiments indicate that improving MLLM embodied capabilities can benefit embodied tasks in simulated environments. Project website: this https URL 

**Abstract (ZH)**: 具身能力是指智能体感知、理解并互动于物理世界的一套基本能力。尽管多模态大型语言模型（MLLMs）显示出作为具身智能体的潜力，但对其具身能力的全面而系统性的评估仍然未充分探索，现有基准主要集中在规划或空间理解等特定领域。为弥补这一不足，我们引入了BEAR，这是一个全面细致的基准，用于评估MLLMs在原子具身能力上的表现。BEAR包含4469个交错的图像-视频-文本条目，分布在6个类别和14个领域中，涵盖了从低级指点、轨迹理解、空间推理到高级规划的任务。对20个代表性MLLMs的广泛评估结果揭示了它们在所有具身能力领域的持续局限性。为解决这一不足，我们提出了一种多模态可对话的智能体BEAR-Agent，它通过集成预训练的视觉模型来增强MLLM的感知、3D理解和规划能力。BEAR-Agent在BEAR上的多样化具身能力上显著提升了MLLM的表现，相对于GPT-5实现了9.12%的绝对改善和17.5%的相对提升。此外，我们的实验表明，提高MLLM的具身能力可以受益于模拟环境中的具身任务。项目网站：this https URL。 

---
# Unified World Models: Memory-Augmented Planning and Foresight for Visual Navigation 

**Title (ZH)**: 统一世界模型：增强记忆的视觉导航规划与预见能力 

**Authors**: Yifei Dong, Fengyi Wu, Guangyu Chen, Zhi-Qi Cheng, Qiyu Hu, Yuxuan Zhou, Jingdong Sun, Jun-Yan He, Qi Dai, Alexander G Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.08713)  

**Abstract**: Enabling embodied agents to effectively imagine future states is critical for robust and generalizable visual navigation. Current state-of-the-art approaches, however, adopt modular architectures that separate navigation planning from visual world modeling, leading to state-action misalignment and limited adaptability in novel or dynamic scenarios. To overcome this fundamental limitation, we propose UniWM, a unified, memory-augmented world model integrating egocentric visual foresight and planning within a single multimodal autoregressive backbone. Unlike modular frameworks, UniWM explicitly grounds action decisions in visually imagined outcomes, ensuring tight alignment between prediction and control. A hierarchical memory mechanism further integrates detailed short-term perceptual cues with longer-term trajectory context, enabling stable, coherent reasoning over extended horizons. Extensive experiments across four challenging benchmarks (Go Stanford, ReCon, SCAND, HuRoN) demonstrate that UniWM substantially improves navigation success rates by up to 30%, significantly reduces trajectory errors compared to strong baselines, and exhibits impressive zero-shot generalization on the unseen TartanDrive dataset. These results highlight UniWM as a principled step toward unified, imagination-driven embodied navigation. 

**Abstract (ZH)**: 实现自主代理有效地想象未来状态对于稳健且通用的视觉导航至关重要。然而，当前最先进的方法采用模块化架构，将导航规划与视觉世界建模分开，导致状态-动作不匹配并在新颖或动态场景中展现出有限的适应性。为克服这一根本限制，我们提出UniWM，这是一种统一的记忆增强世界模型，在单一多模态自回归骨干网络中整合了基于自我中心视角的视觉前瞻与规划。与模块化框架不同，UniWM 显式地将动作决策与视觉想象结果联系起来，确保预测与控制之间保持紧密对齐。分层记忆机制进一步将详细的短时感知线索与长期轨迹上下文整合起来，使代理能够在较长时间段内进行稳定且连贯的推理。在四个具有挑战性的基准测试（Go Stanford、ReCon、SCAND、HuRoN）上进行的广泛实验表明，UniWM 能够将导航成功率提高多达 30%，显著减少与强劲基线相比的轨迹误差，并在未见过的 TartanDrive 数据集上展现出令人印象深刻的零样本泛化能力。这些结果突显了UniWM 是迈向统一、基于想象的自主导航的关键一步。 

---
# Humanoid Artificial Consciousness Designed with Large Language Model Based on Psychoanalysis and Personality Theory 

**Title (ZH)**: 基于心理分析和人格理论的大语言模型驱动的人形人工意识设计 

**Authors**: Sang Hun Kim, Jongmin Lee, Dongkyu Park, So Young Lee, Yosep Chong  

**Link**: [PDF](https://arxiv.org/pdf/2510.09043)  

**Abstract**: Human consciousness is still a concept hard to define with current scientific understanding. Although Large Language Models (LLMs) have recently demonstrated significant advancements across various domains including translation and summarization, human consciousness is not something to imitate with current upfront technology owing to so-called hallucination. This study, therefore, proposes a novel approach to address these challenges by integrating psychoanalysis and the Myers-Briggs Type Indicator (MBTI) into constructing consciousness and personality modules. We developed three artificial consciousnesses (self-awareness, unconsciousness, and preconsciousness) based on the principles of psychoanalysis. Additionally, we designed 16 characters with different personalities representing the sixteen MBTI types, with several attributes such as needs, status, and memories. To determine if our model's artificial consciousness exhibits human-like cognition, we created ten distinct situations considering seven attributes such as emotional understanding and logical thinking. The decision-making process of artificial consciousness and the final action were evaluated in three ways: survey evaluation, three-tier classification via ChatGPT, and qualitative review. Both quantitative and qualitative analyses indicated a high likelihood of well-simulated consciousness, although the difference in response between different characters and consciousnesses was not very significant. This implies that the developed models incorporating elements of psychoanalysis and personality theory can lead to building a more intuitive and adaptable AI system with humanoid consciousness. Therefore, this study contributes to opening up new avenues for improving AI interactions in complex cognitive contexts. 

**Abstract (ZH)**: 当前科学理解下人类意识仍然是一个难以定义的概念。尽管大型语言模型（LLMs）在翻译和总结等领域最近取得了显著进展，但在当前的技术水平下，由于所谓的幻觉现象，人类意识仍无法直接模仿。因此，本研究提出了一种新颖的方法，通过将精神分析理论与迈尔斯-布里格斯类型指标（MBTI）相结合，来构建意识和人格模块。我们基于精神分析的原则开发了三种人工意识（自我意识、无意识和前意识），并设计了16个具有不同人格特征的角色，代表了十六种MBTI类型，赋予它们诸如需求、状态和记忆等属性。为了确定模型的人工意识是否表现出类似人类的认知能力，我们创建了十种不同的情况，考虑了诸如情感理解和逻辑思维等七个属性。通过三种方式评估人工意识的决策过程和最终行动：问卷评估、通过ChatGPT进行三级分类以及质性审查。定量和定性分析都表明，模型中模拟的意识高度逼真，尽管不同角色和意识之间的反应差异并不显著。这表明，结合精神分析和人格理论元素的模型有助于构建更具直观性和适应性的类人意识AI系统。因此，本研究为改进在复杂认知背景下的AI交互开辟了新的途径。 

---
# Hypothesis Hunting with Evolving Networks of Autonomous Scientific Agents 

**Title (ZH)**: 自主科学代理演化网络中的假设探寻 

**Authors**: Tennison Liu, Silas Ruhrberg Estévez, David L. Bentley, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08619)  

**Abstract**: Large-scale scientific datasets -- spanning health biobanks, cell atlases, Earth reanalyses, and more -- create opportunities for exploratory discovery unconstrained by specific research questions. We term this process hypothesis hunting: the cumulative search for insight through sustained exploration across vast and complex hypothesis spaces. To support it, we introduce AScience, a framework modeling discovery as the interaction of agents, networks, and evaluation norms, and implement it as ASCollab, a distributed system of LLM-based research agents with heterogeneous behaviors. These agents self-organize into evolving networks, continually producing and peer-reviewing findings under shared standards of evaluation. Experiments show that such social dynamics enable the accumulation of expert-rated results along the diversity-quality-novelty frontier, including rediscoveries of established biomarkers, extensions of known pathways, and proposals of new therapeutic targets. While wet-lab validation remains indispensable, our experiments on cancer cohorts demonstrate that socially structured, agentic networks can sustain exploratory hypothesis hunting at scale. 

**Abstract (ZH)**: 大规模科学数据集——涵盖健康生物银行、细胞图集、地球再分析等领域——为不受特定研究问题约束的探索性发现创造了机会。我们称这一过程为假设狩猎：通过持续探索庞大而复杂的假设空间来累积洞察的过程。为了支持这一过程，我们引入了AScience框架，将其建模为代理、网络和评估标准的互动，并通过ASCollab实现了基于LLM的研究代理分布式系统，这些代理具有异质行为。这些代理自我组织成不断演变的网络，在共享的评估标准下持续产生和同行评审研究成果。实验表明，这种社会动态能够沿着多样性-质量-新颖性前沿积累专家评价的结果，包括重新发现已知生物标志物、扩展已知途径以及提出新的治疗靶标。尽管湿实验验证仍然是必不可少的，但我们的癌症队列实验表明，社会结构化、代理网络可以在大规模下持续进行探索性假设狩猎。 

---
# Dyna-Mind: Learning to Simulate from Experience for Better AI Agents 

**Title (ZH)**: Dyna-Mind: 从经验中学习模拟以提高AI代理性能 

**Authors**: Xiao Yu, Baolin Peng, Michel Galley, Hao Cheng, Qianhui Wu, Janardhan Kulkarni, Suman Nath, Zhou Yu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.09577)  

**Abstract**: Reasoning models have recently shown remarkable progress in domains such as math and coding. However, their expert-level abilities in math and coding contrast sharply with their performance in long-horizon, interactive tasks such as web navigation and computer/phone-use. Inspired by literature on human cognition, we argue that current AI agents need ''vicarious trial and error'' - the capacity to mentally simulate alternative futures before acting - in order to enhance their understanding and performance in complex interactive environments. We introduce Dyna-Mind, a two-stage training framework that explicitly teaches (V)LM agents to integrate such simulation into their reasoning. In stage 1, we introduce Reasoning with Simulations (ReSim), which trains the agent to generate structured reasoning traces from expanded search trees built from real experience gathered through environment interactions. ReSim thus grounds the agent's reasoning in faithful world dynamics and equips it with the ability to anticipate future states in its reasoning. In stage 2, we propose Dyna-GRPO, an online reinforcement learning method to further strengthen the agent's simulation and decision-making ability by using both outcome rewards and intermediate states as feedback from real rollouts. Experiments on two synthetic benchmarks (Sokoban and ALFWorld) and one realistic benchmark (AndroidWorld) demonstrate that (1) ReSim effectively infuses simulation ability into AI agents, and (2) Dyna-GRPO leverages outcome and interaction-level signals to learn better policies for long-horizon, planning-intensive tasks. Together, these results highlight the central role of simulation in enabling AI agents to reason, plan, and act more effectively in the ever more challenging environments. 

**Abstract (ZH)**: Reasoning Models Have Recently Demonstrated Remarkable Progress in Domains Such as Math and Coding: However, Their Expert-Level Abilities in These Domains Contrast Sharply with Their Performance in Long-Horizon, Interactive Tasks Such as Web Navigation and Computer/Phone Use. Inspired by Literature on Human Cognition, We Argue That Current AI Agents Need ''Vicarious Trial and Error''—the Capacity to Mentally Simulate Alternative Futures Before Acting—in Order to Enhance Their Understanding and Performance in Complex Interactive Environments. 

---
# Robust Driving Control for Autonomous Vehicles: An Intelligent General-sum Constrained Adversarial Reinforcement Learning Approach 

**Title (ZH)**: 自主车辆的稳健驾驶控制：一种智能约束博弈强化学习方法 

**Authors**: Junchao Fan, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09041)  

**Abstract**: Deep reinforcement learning (DRL) has demonstrated remarkable success in developing autonomous driving policies. However, its vulnerability to adversarial attacks remains a critical barrier to real-world deployment. Although existing robust methods have achieved success, they still suffer from three key issues: (i) these methods are trained against myopic adversarial attacks, limiting their abilities to respond to more strategic threats, (ii) they have trouble causing truly safety-critical events (e.g., collisions), but instead often result in minor consequences, and (iii) these methods can introduce learning instability and policy drift during training due to the lack of robust constraints. To address these issues, we propose Intelligent General-sum Constrained Adversarial Reinforcement Learning (IGCARL), a novel robust autonomous driving approach that consists of a strategic targeted adversary and a robust driving agent. The strategic targeted adversary is designed to leverage the temporal decision-making capabilities of DRL to execute strategically coordinated multi-step attacks. In addition, it explicitly focuses on inducing safety-critical events by adopting a general-sum objective. The robust driving agent learns by interacting with the adversary to develop a robust autonomous driving policy against adversarial attacks. To ensure stable learning in adversarial environments and to mitigate policy drift caused by attacks, the agent is optimized under a constrained formulation. Extensive experiments show that IGCARL improves the success rate by at least 27.9\% over state-of-the-art methods, demonstrating superior robustness to adversarial attacks and enhancing the safety and reliability of DRL-based autonomous driving. 

**Abstract (ZH)**: 基于智能综合约束对抗强化学习的鲁棒自主驾驶方法（IGCARL） 

---
# On Epistemic Uncertainty of Visual Tokens for Object Hallucinations in Large Vision-Language Models 

**Title (ZH)**: 视觉令牌在大型视觉-语言模型中对象幻觉中的 Epistemic 不确定性 

**Authors**: Hoigi Seo, Dong Un Kang, Hyunjin Cho, Joohoon Lee, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2510.09008)  

**Abstract**: Large vision-language models (LVLMs), which integrate a vision encoder (VE) with a large language model, have achieved remarkable success across various tasks. However, there are still crucial challenges in LVLMs such as object hallucination, generating descriptions of objects that are not in the input image. Here, we argue that uncertain visual tokens within the VE is a key factor that contributes to object hallucination. Our statistical analysis found that there are positive correlations between visual tokens with high epistemic uncertainty and the occurrence of hallucinations. Furthermore, we show theoretically and empirically that visual tokens in early VE layers that exhibit large representation deviations under small adversarial perturbations indicate high epistemic uncertainty. Based on these findings, we propose a simple yet effective strategy to mitigate object hallucination by modifying the VE only. Our method comprises a proxy method with adversarial perturbations for identifying uncertain visual tokens efficiently and a method to mask these uncertain visual tokens during the self-attention process in the middle layers of the VE, suppressing their influence on visual encoding and thus alleviating hallucinations. Extensive experiments show that our method significantly reduces object hallucinations in LVLMs and can synergistically work with other prior arts. 

**Abstract (ZH)**: 大规模vision-language模型（LVLMs）通过将视觉编码器（VE）与大规模语言模型集成，已在各种任务中取得了显著成果。然而，LVLMs仍然存在着关键性挑战，如对象幻觉，即生成输入图像中不存在的对象描述。我们argue认为，视觉编码器中的不确定性视觉令牌是导致对象幻觉的关键因素。我们的统计分析发现，具有较高证前知识不确定性的视觉令牌与幻觉的发生存在正相关关系。此外，我们理论和实验上证明，在小对抗扰动下表现出大表示偏差的视觉编码器早期层数的视觉令牌表明了高证前知识不确定性。基于这些发现，我们提出了一种简单有效的策略，仅通过修改视觉编码器来减轻对象幻觉。该方法包括一个代理方法，利用对抗扰动高效识别不确定性视觉令牌，以及一种在视觉编码器中间层的自注意力过程中屏蔽这些不确定性视觉令牌的方法，从而抑制它们对视觉编码的影响，并缓解幻觉。广泛的实验表明，我们的方法显著减少了LVLMs中的对象幻觉，并与现有的其他方法协同工作。 

---
# Pinpointing crucial steps: Attribution-based Credit Assignment for Verifiable Reinforcement Learning 

**Title (ZH)**: 精确定位关键步骤：基于归因的可验证强化学习责任分配 

**Authors**: Junxi Yin, Haisen Luo, Zhenyu Li, Yihua Liu, Dan Liu, Zequn Li, Xiaohang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08899)  

**Abstract**: While Reinforcement Learning with Verifiable Rewards (RLVR) enhances complex reasoning in LLMs, current methods struggle to balance exploration and exploitation. This leads to critical issues like inaccurate credit assignment for intermediate steps and premature entropy collapse, limiting model performance. To address this, we introduce Attribution-based Contribution to Policy Optimization (ACPO), a phased framework that incorporates a difficulty-aware curriculum. ACPO improves exploration by using trajectory semantic segmentation and an attribution-based representation to dynamically regulate policy entropy, thus mitigating its collapse. Concurrently, it enhances exploitation with a factorized reward system that precisely quantifies the hierarchical contribution of each reasoning step, ensuring accurate credit assignment. Extensive experiments on challenging benchmarks, including AIME, MATH, and AMC, demonstrate that ACPO significantly outperforms existing state-of-the-art approaches. 

**Abstract (ZH)**: 面向可验证奖励的强化学习在提高大语言模型复杂推理能力方面取得了进展，但当前方法在平衡探索与利用之间存在困难，导致中间步骤责任分配不准确和过早的熵坍缩等问题，限制了模型性能。为解决这一问题，我们提出了基于归因的策略优化贡献（ACPO）框架，该框架采用难度感知的课程设计，通过轨迹语义分割和基于归因的表示动态调节策略熵，从而减轻熵坍缩，同时利用分解奖励系统精确量化每个推理步骤的分级贡献，确保准确的责任分配。在AIME、MATH和AMC等具有挑战性的基准测试中，ACPO显著优于现有最先进的方法。 

---
# Designing and Evaluating an AI-driven Immersive Multidisciplinary Simulation (AIMS) for Interprofessional Education 

**Title (ZH)**: 基于AI驱动的沉浸式多学科模拟（AIMS）在跨专业教育中的设计与评估 

**Authors**: Ruijie Wang, Jie Lu, Bo Pei, Evonne Jones, Jamey Brinson, Timothy Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.08891)  

**Abstract**: Interprofessional education has long relied on case studies and the use of standardized patients to support teamwork, communication, and related collaborative competencies among healthcare professionals. However, traditional approaches are often limited by cost, scalability, and inability to mimic the dynamic complexity of real-world clinical scenarios. To address these challenges, we designed and developed AIMS (AI-Enhanced Immersive Multidisciplinary Simulations), a virtual simulation that integrates a large language model (Gemini-2.5-Flash), a Unity-based virtual environment engine, and a character creation pipeline to support synchronized, multimodal interactions between the user and the virtual patient. AIMS was designed to enhance collaborative clinical reasoning and health promotion competencies among students from pharmacy, medicine, nursing, and social work. A formal usability testing session was conducted which participants assumed professional roles on a healthcare team and engaged in a mix of scripted and unscripted conversations. Participants explored the patient's symptoms, social context, and care needs. Usability issues were identified (e.g., audio routing, response latency) and used to guide subsequent refinements. Findings in general suggest that AIMS supports realistic, profession-specific and contextually appropriate conversations. We discussed both technical and pedagogical innovations of AIMS and concluded with future directions. 

**Abstract (ZH)**: 基于AI增强的沉浸式多学科模拟在提升医药护社学生团队协作与沟通能力中的应用与创新 

---
# ConPoSe: LLM-Guided Contact Point Selection for Scalable Cooperative Object Pushing 

**Title (ZH)**: ConPoSe: LLM引导的接触点选择以实现可扩展的协作物体推送 

**Authors**: Noah Steinkrüger, Nisarga Nilavadi, Wolfram Burgard, Tanja Katharina Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2510.08705)  

**Abstract**: Object transportation in cluttered environments is a fundamental task in various domains, including domestic service and warehouse logistics. In cooperative object transport, multiple robots must coordinate to move objects that are too large for a single robot. One transport strategy is pushing, which only requires simple robots. However, careful selection of robot-object contact points is necessary to push the object along a preplanned path. Although this selection can be solved analytically, the solution space grows combinatorially with the number of robots and object size, limiting scalability. Inspired by how humans rely on common-sense reasoning for cooperative transport, we propose combining the reasoning capabilities of Large Language Models with local search to select suitable contact points. Our LLM-guided local search method for contact point selection, ConPoSe, successfully selects contact points for a variety of shapes, including cuboids, cylinders, and T-shapes. We demonstrate that ConPoSe scales better with the number of robots and object size than the analytical approach, and also outperforms pure LLM-based selection. 

**Abstract (ZH)**: 在拥挤环境中进行物体运输是服务机器人和仓库物流等多个领域中的基本任务。在协同物体运输中，多台机器人必须协调合作以搬运单台机器人无法搬运的大型物体。一种运输策略是推拉，仅需简单的机器人即可实施。然而，为了沿预设路径移动物体，必须仔细选择机器人与物体的接触点。尽管这种选择可以通过逻辑分析解决，但随着机器人数量和物体大小的增加，解的空间会成组合性地增长，限制了其可扩展性。受人类在合作搬运中依赖常识推理的启发，我们提出结合大型语言模型的推理能力和局部搜索选择合适的接触点。我们的由大型语言模型指导的局部搜索方法ConPoSe成功为各种形状的物体（包括立方体、圆柱体和T形物体）选择了接触点。我们证明，与分析方法相比，ConPoSe在机器人数量和物体大小增加时具有更好的可扩展性，并且优于基于大型语言模型的纯选择方法。 

---
