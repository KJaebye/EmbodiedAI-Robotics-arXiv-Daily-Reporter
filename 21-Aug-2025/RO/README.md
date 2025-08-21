# Safe and Transparent Robots for Human-in-the-Loop Meat Processing 

**Title (ZH)**: 循环 human 在环肉类加工中安全透明的机器人系统 

**Authors**: Sagar Parekh, Casey Grothoff, Ryan Wright, Robin White, Dylan P. Losey  

**Link**: [PDF](https://arxiv.org/pdf/2508.14763)  

**Abstract**: Labor shortages have severely affected the meat processing sector. Automated technology has the potential to support the meat industry, assist workers, and enhance job quality. However, existing automation in meat processing is highly specialized, inflexible, and cost intensive. Instead of forcing manufacturers to buy a separate device for each step of the process, our objective is to develop general-purpose robotic systems that work alongside humans to perform multiple meat processing tasks. Through a recently conducted survey of industry experts, we identified two main challenges associated with integrating these collaborative robots alongside human workers. First, there must be measures to ensure the safety of human coworkers; second, the coworkers need to understand what the robot is doing. This paper addresses both challenges by introducing a safety and transparency framework for general-purpose meat processing robots. For safety, we implement a hand-detection system that continuously monitors nearby humans. This system can halt the robot in situations where the human comes into close proximity of the operating robot. We also develop an instrumented knife equipped with a force sensor that can differentiate contact between objects such as meat, bone, or fixtures. For transparency, we introduce a method that detects the robot's uncertainty about its performance and uses an LED interface to communicate that uncertainty to the human. Additionally, we design a graphical interface that displays the robot's plans and allows the human to provide feedback on the planned cut. Overall, our framework can ensure safe operation while keeping human workers in-the-loop about the robot's actions which we validate through a user study. 

**Abstract (ZH)**: 劳动力短缺严重冲击了肉类加工业。自动化技术有望支持肉类行业、辅助工人并提高工作质量。然而，现有的肉类加工自动化技术高度专门化、缺乏灵活性且成本高昂。我们旨在开发通用型机器人系统，与人类工人协同工作，同时执行多种肉类加工任务，而不是要求制造商为每一步骤购买独立设备。本文通过近期对行业专家的调查，识别出将这些协作机器人与人类工人结合时面临的两大挑战：首先，必须采取措施确保人类同事的安全；其次，同事需要理解机器人的行动。本文通过引入通用型肉类加工机器人安全与透明性框架来应对这两个挑战。在安全性方面，我们实现了一个手部检测系统，持续监测附近的人员，在人类接近操作机器人时使机器人暂停。我们还开发了一种带有力传感器的仪器刀，能够区分与肉类、骨骼或其他固定装置的接触。在透明性方面，我们引入了一种方法，用于检测机器人对其自身表现的不确定性，并使用LED界面向人类传达这种不确定性。此外，我们设计了一个图形界面，显示机器人的计划，并允许人类对计划切割提供反馈。总体而言，我们的框架可以在确保安全操作的同时，让人类工人了解机器人的行动，并通过用户研究进行了验证。 

---
# Consistent Pose Estimation of Unmanned Ground Vehicles through Terrain-Aided Multi-Sensor Fusion on Geometric Manifolds 

**Title (ZH)**: 基于地形辅助多传感器融合几何流形上的一致轨迹估计 

**Authors**: Alexander Raab, Stephan Weiss, Alessandro Fornasier, Christian Brommer, Abdalrahman Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2508.14661)  

**Abstract**: Aiming to enhance the consistency and thus long-term accuracy of Extended Kalman Filters for terrestrial vehicle localization, this paper introduces the Manifold Error State Extended Kalman Filter (M-ESEKF). By representing the robot's pose in a space with reduced dimensionality, the approach ensures feasible estimates on generic smooth surfaces, without introducing artificial constraints or simplifications that may degrade a filter's performance. The accompanying measurement models are compatible with common loosely- and tightly-coupled sensor modalities and also implicitly account for the ground geometry. We extend the formulation by introducing a novel correction scheme that embeds additional domain knowledge into the sensor data, giving more accurate uncertainty approximations and further enhancing filter consistency. The proposed estimator is seamlessly integrated into a validated modular state estimation framework, demonstrating compatibility with existing implementations. Extensive Monte Carlo simulations across diverse scenarios and dynamic sensor configurations show that the M-ESEKF outperforms classical filter formulations in terms of consistency and stability. Moreover, it eliminates the need for scenario-specific parameter tuning, enabling its application in a variety of real-world settings. 

**Abstract (ZH)**: 面向陆地车辆定位的增强一致性与长期准确性扩展卡尔曼滤波方法：基于流形误差状态扩展卡尔曼滤波器（M-ESEKF）的研究 

---
# An Informative Planning Framework for Target Tracking and Active Mapping in Dynamic Environments with ASVs 

**Title (ZH)**: 基于ASV的动态环境中目标跟踪与主动建图的信息性规划框架 

**Authors**: Sanjeev Ramkumar Sudha, Marija Popović, Erlend M. Coates  

**Link**: [PDF](https://arxiv.org/pdf/2508.14636)  

**Abstract**: Mobile robot platforms are increasingly being used to automate information gathering tasks such as environmental monitoring. Efficient target tracking in dynamic environments is critical for applications such as search and rescue and pollutant cleanups. In this letter, we study active mapping of floating targets that drift due to environmental disturbances such as wind and currents. This is a challenging problem as it involves predicting both spatial and temporal variations in the map due to changing conditions. We propose an informative path planning framework to map an arbitrary number of moving targets with initially unknown positions in dynamic environments. A key component of our approach is a spatiotemporal prediction network that predicts target position distributions over time. We propose an adaptive planning objective for target tracking that leverages these predictions. Simulation experiments show that our proposed planning objective improves target tracking performance compared to existing methods that consider only entropy reduction as the planning objective. Finally, we validate our approach in field tests using an autonomous surface vehicle, showcasing its ability to track targets in real-world monitoring scenarios. 

**Abstract (ZH)**: 移动机器人平台被广泛用于自动化信息采集任务，如环境监测。在动态环境中高效的目标跟踪对于搜索救援和污染物清理等应用至关重要。本文研究了由于风、洋流等环境干扰导致漂移的浮动目标的主动映射问题。这是一个具有挑战性的问题，因为它涉及到因条件变化而产生的时空地图变化的预测。我们提出了一种信息路径规划框架，用于在动态环境中对具有未知初始位置的任意数量移动目标进行映射。我们方法的关键环节是一个时空预测网络，该网络预测目标位置随时间的变化分布。我们提出了一种适应性的规划目标，利用这些预测来进行目标跟踪。仿真实验表明，相比于仅考虑熵减少作为规划目标的现有方法，我们提出的方法改善了目标跟踪性能。最后，我们在实地测试中使用自主水面车辆验证了该方法，展示了其在实际监测场景中跟踪目标的能力。 

---
# Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination 

**Title (ZH)**: 基于紧迫性意识规划与协调的LLM代理能否解决协作任务？ 

**Authors**: João Vitor de Carvalho Silva, Douglas G. Macharet  

**Link**: [PDF](https://arxiv.org/pdf/2508.14635)  

**Abstract**: The ability to coordinate actions across multiple agents is critical for solving complex, real-world problems. Large Language Models (LLMs) have shown strong capabilities in communication, planning, and reasoning, raising the question of whether they can also support effective collaboration in multi-agent settings. In this work, we investigate the use of LLM agents to solve a structured victim rescue task that requires division of labor, prioritization, and cooperative planning. Agents operate in a fully known graph-based environment and must allocate resources to victims with varying needs and urgency levels. We systematically evaluate their performance using a suite of coordination-sensitive metrics, including task success rate, redundant actions, room conflicts, and urgency-weighted efficiency. This study offers new insights into the strengths and failure modes of LLMs in physically grounded multi-agent collaboration tasks, contributing to future benchmarks and architectural improvements. 

**Abstract (ZH)**: 跨多个代理协调行动的能力对于解决复杂的真实世界问题至关重要。大规模语言模型（LLMs）在通信、规划和推理方面显示出了强大的能力，引起了它们是否也能在多代理环境中支持有效协作的疑问。在本项工作中，我们研究使用LLM代理解决一个需要分工、优先级设定和合作规划的结构化救援任务。代理在完全已知的图环境中共操作，必须将资源分配给具有不同需求和紧迫性的受害者。我们使用一系列敏感于协作的评估指标系统地评估其性能，包括任务成功率、重复动作、房间冲突和紧迫性加权效率。这项研究为LLMs在物理 grounded 多代理协作任务中的优势和失败模式提供了新的见解，有助于未来基准测试和架构改进。 

---
# TRUST-Planner: Topology-guided Robust Trajectory Planner for AAVs with Uncertain Obstacle Spatial-temporal Avoidance 

**Title (ZH)**: TRUST-Planner：基于拓扑学的鲁棒轨迹规划器，用于具有不确定障碍物空间时间避免的AAVs 

**Authors**: Junzhi Li, Teng Long, Jingliang Sun, Jianxin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.14610)  

**Abstract**: Despite extensive developments in motion planning of autonomous aerial vehicles (AAVs), existing frameworks faces the challenges of local minima and deadlock in complex dynamic environments, leading to increased collision risks. To address these challenges, we present TRUST-Planner, a topology-guided hierarchical planning framework for robust spatial-temporal obstacle avoidance. In the frontend, a dynamic enhanced visible probabilistic roadmap (DEV-PRM) is proposed to rapidly explore topological paths for global guidance. The backend utilizes a uniform terminal-free minimum control polynomial (UTF-MINCO) and dynamic distance field (DDF) to enable efficient predictive obstacle avoidance and fast parallel computation. Furthermore, an incremental multi-branch trajectory management framework is introduced to enable spatio-temporal topological decision-making, while efficiently leveraging historical information to reduce replanning time. Simulation results show that TRUST-Planner outperforms baseline competitors, achieving a 96\% success rate and millisecond-level computation efficiency in tested complex environments. Real-world experiments further validate the feasibility and practicality of the proposed method. 

**Abstract (ZH)**: 基于拓扑引导的分层规划框架TRUST-Planner：稳健的空间-时间障碍规避 

---
# EAROL: Environmental Augmented Perception-Aware Planning and Robust Odometry via Downward-Mounted Tilted LiDAR 

**Title (ZH)**: EAROL：环境增强感知aware规划与向下倾斜安装LiDAR的稳健 odometry 

**Authors**: Xinkai Liang, Yigu Ge, Yangxi Shi, Haoyu Yang, Xu Cao, Hao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14554)  

**Abstract**: To address the challenges of localization drift and perception-planning coupling in unmanned aerial vehicles (UAVs) operating in open-top scenarios (e.g., collapsed buildings, roofless mazes), this paper proposes EAROL, a novel framework with a downward-mounted tilted LiDAR configuration (20° inclination), integrating a LiDAR-Inertial Odometry (LIO) system and a hierarchical trajectory-yaw optimization algorithm. The hardware innovation enables constraint enhancement via dense ground point cloud acquisition and forward environmental awareness for dynamic obstacle detection. A tightly-coupled LIO system, empowered by an Iterative Error-State Kalman Filter (IESKF) with dynamic motion compensation, achieves high level 6-DoF localization accuracy in feature-sparse environments. The planner, augmented by environment, balancing environmental exploration, target tracking precision, and energy efficiency. Physical experiments demonstrate 81% tracking error reduction, 22% improvement in perceptual coverage, and near-zero vertical drift across indoor maze and 60-meter-scale outdoor scenarios. This work proposes a hardware-algorithm co-design paradigm, offering a robust solution for UAV autonomy in post-disaster search and rescue missions. We will release our software and hardware as an open-source package for the community. Video: this https URL. 

**Abstract (ZH)**: 面向开放顶场景下无人机定位漂移与感知规划耦合挑战的EAROL新型框架：硬件-算法协同设计 

---
# Taming VR Teleoperation and Learning from Demonstration for Multi-Task Bimanual Table Service Manipulation 

**Title (ZH)**: 驯化VR远程操作并从演示学习以实现多任务双臂桌上传感器操作 

**Authors**: Weize Li, Zhengxiao Han, Lixin Xu, Xiangyu Chen, Harrison Bounds, Chenrui Zhang, Yifan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14542)  

**Abstract**: This technical report presents the champion solution of the Table Service Track in the ICRA 2025 What Bimanuals Can Do (WBCD) competition. We tackled a series of demanding tasks under strict requirements for speed, precision, and reliability: unfolding a tablecloth (deformable-object manipulation), placing a pizza onto the table (pick-and-place), and opening and closing a food container with the lid. Our solution combines VR-based teleoperation and Learning from Demonstrations (LfD) to balance robustness and autonomy. Most subtasks were executed through high-fidelity remote teleoperation, while the pizza placement was handled by an ACT-based policy trained from 100 in-person teleoperated demonstrations with randomized initial configurations. By carefully integrating scoring rules, task characteristics, and current technical capabilities, our approach achieved both high efficiency and reliability, ultimately securing the first place in the competition. 

**Abstract (ZH)**: 本技术报告展示了在ICRA 2025 What Bimanuals Can Do (WBCD) 比赛中Table Service Track的冠军解决方案。我们针对严格的速度、精度和可靠性要求，应对了一系列挑战任务：铺设桌布（变形物体操作）、将披萨放在桌子上（抓取和放置）以及开启和关闭带盖的食物容器。我们的解决方案结合了基于VR的远程操作和学习从示范（LfD）技术，以平衡稳健性和自主性。大多数子任务通过高保真远程操作执行，而披萨放置任务则由一个基于ACT的策略处理，该策略是从100个随机初始配置的人工示范中训练得到的。通过仔细整合评分规则、任务特性及当前的技术能力，我们的方法在效率和可靠性方面都取得了优异成绩，最终赢得了比赛的第一名。 

---
# FBI: Learning Dexterous In-hand Manipulation with Dynamic Visuotactile Shortcut Policy 

**Title (ZH)**: FBI：学习动态视触捷径策略的在手灵巧操作 

**Authors**: Yijin Chen, Wenqiang Xu, Zhenjun Yu, Tutian Tang, Yutong Li, Siqiong Yao, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14441)  

**Abstract**: Dexterous in-hand manipulation is a long-standing challenge in robotics due to complex contact dynamics and partial observability. While humans synergize vision and touch for such tasks, robotic approaches often prioritize one modality, therefore limiting adaptability. This paper introduces Flow Before Imitation (FBI), a visuotactile imitation learning framework that dynamically fuses tactile interactions with visual observations through motion dynamics. Unlike prior static fusion methods, FBI establishes a causal link between tactile signals and object motion via a dynamics-aware latent model. FBI employs a transformer-based interaction module to fuse flow-derived tactile features with visual inputs, training a one-step diffusion policy for real-time execution. Extensive experiments demonstrate that the proposed method outperforms the baseline methods in both simulation and the real world on two customized in-hand manipulation tasks and three standard dexterous manipulation tasks. Code, models, and more results are available in the website this https URL. 

**Abstract (ZH)**: 手部灵巧 manipulation 是由于复杂接触动力学和部分可观测性而在机器人领域长期存在的挑战。虽然人类通过视觉和触觉协同完成这些任务，但机器人方法往往优先考虑一种模态，从而限制了适应性。本文介绍了一种名为 Flow Before Imitation (FBI) 的联合触觉模仿学习框架，该框架通过运动动力学动态融合触觉交互和视觉观测。与之前静态融合方法不同，FBI 通过动力学感知的潜在模型建立了触觉信号与物体运动之间的因果关系。FBI 使用基于变换器的交互模块将基于流动的触觉特征与视觉输入融合，并训练了一步扩散策略以实现实时执行。广泛的实验结果表明，在两个定制的手部灵巧 manipulation 任务和三个标准灵巧 manipulation 任务上，所提出的方法在仿真和现实世界中均优于基线方法。更多代码、模型和实验结果可在以下网址获取 this https URL。 

---
# DEXTER-LLM: Dynamic and Explainable Coordination of Multi-Robot Systems in Unknown Environments via Large Language Models 

**Title (ZH)**: DEXTER-LLM：通过大型语言模型在未知环境中的多机器人系统动态可解释协调 

**Authors**: Yuxiao Zhu, Junfeng Chen, Xintong Zhang, Meng Guo, Zhongkui Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14387)  

**Abstract**: Online coordination of multi-robot systems in open and unknown environments faces significant challenges, particularly when semantic features detected during operation dynamically trigger new tasks. Recent large language model (LLMs)-based approaches for scene reasoning and planning primarily focus on one-shot, end-to-end solutions in known environments, lacking both dynamic adaptation capabilities for online operation and explainability in the processes of planning. To address these issues, a novel framework (DEXTER-LLM) for dynamic task planning in unknown environments, integrates four modules: (i) a mission comprehension module that resolves partial ordering of tasks specified by natural languages or linear temporal logic formulas (LTL); (ii) an online subtask generator based on LLMs that improves the accuracy and explainability of task decomposition via multi-stage reasoning; (iii) an optimal subtask assigner and scheduler that allocates subtasks to robots via search-based optimization; and (iv) a dynamic adaptation and human-in-the-loop verification module that implements multi-rate, event-based updates for both subtasks and their assignments, to cope with new features and tasks detected online. The framework effectively combines LLMs' open-world reasoning capabilities with the optimality of model-based assignment methods, simultaneously addressing the critical issue of online adaptability and explainability. Experimental evaluations demonstrate exceptional performances, with 100% success rates across all scenarios, 160 tasks and 480 subtasks completed on average (3 times the baselines), 62% less queries to LLMs during adaptation, and superior plan quality (2 times higher) for compound tasks. Project page at this https URL 

**Abstract (ZH)**: 在未知环境中的多机器人系统在线协调面临显著挑战，特别是在操作过程中检测到的语义特征动态触发新任务时。基于大型语言模型（LLMs）的场景推理和规划方法主要集中在已知环境的一次性端到端解决方案上，缺乏在线操作的动态适应能力和规划过程中的可解释性。为了解决这些问题，提出了一种新的框架（DEXTER-LLM），用于未知环境中的动态任务规划，该框架整合了四个模块：（i）任务理解模块，解决由自然语言或线性时序逻辑公式（LTL）指定的任务的部分顺序；（ii）基于LLMs的在线子任务生成器，通过多阶段推理提高任务分解的准确性和可解释性；（iii）最优子任务分配和调度模块，通过基于搜索的优化将子任务分配给机器人；（iv）动态适应和人类在环验证模块，通过多速率、事件驱动更新同时处理在线检测到的新特征和任务。该框架有效结合了LLMs的开放世界推理能力和基于模型的分配方法的最优化，同时解决了在线适应性和可解释性这两个关键问题。实验评估显示，该框架在所有场景下均取得100%的成功率，平均完成160个任务和480个子任务，适应过程中LLMs查询量减少62%，多任务规划质量提高2倍。项目页面详见此链接。 

---
# Offline Imitation Learning upon Arbitrary Demonstrations by Pre-Training Dynamics Representations 

**Title (ZH)**: 基于任意演示的离线模仿学习通过先训练动力学表示 

**Authors**: Haitong Ma, Bo Dai, Zhaolin Ren, Yebin Wang, Na Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14383)  

**Abstract**: Limited data has become a major bottleneck in scaling up offline imitation learning (IL). In this paper, we propose enhancing IL performance under limited expert data by introducing a pre-training stage that learns dynamics representations, derived from factorizations of the transition dynamics. We first theoretically justify that the optimal decision variable of offline IL lies in the representation space, significantly reducing the parameters to learn in the downstream IL. Moreover, the dynamics representations can be learned from arbitrary data collected with the same dynamics, allowing the reuse of massive non-expert data and mitigating the limited data issues. We present a tractable loss function inspired by noise contrastive estimation to learn the dynamics representations at the pre-training stage. Experiments on MuJoCo demonstrate that our proposed algorithm can mimic expert policies with as few as a single trajectory. Experiments on real quadrupeds show that we can leverage pre-trained dynamics representations from simulator data to learn to walk from a few real-world demonstrations. 

**Abstract (ZH)**: 在有限专家数据下通过预训练增强离线 imitation learning 性能的研究 

---
# FiReFly: Fair Distributed Receding Horizon Planning for Multiple UAVs 

**Title (ZH)**: FiReFly：公平分布式的回溯_horizon 计划算法在多无人机系统中的应用 

**Authors**: Nicole Fronda, Bardh Hoxha, Houssam Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2508.14381)  

**Abstract**: We propose injecting notions of fairness into multi-robot motion planning. When robots have competing interests, it is important to optimize for some kind of fairness in their usage of resources. In this work, we explore how the robots' energy expenditures might be fairly distributed among them, while maintaining mission success. We formulate a distributed fair motion planner and integrate it with safe controllers in a algorithm called FiReFly. For simulated reach-avoid missions, FiReFly produces fairer trajectories and improves mission success rates over a non-fair planner. We find that real-time performance is achievable up to 15 UAVs, and that scaling up to 50 UAVs is possible with trade-offs between runtime and fairness improvements. 

**Abstract (ZH)**: 我们将公平性概念注入到多机器人运动规划中。当机器人之间存在竞争利益时，优化它们对资源的使用公平性变得尤为重要。在本工作中，我们探讨了如何在保持任务成功的同时，公平地分配机器人的能量消耗。我们提出了一个分布式公平运动规划算法，并将其与安全控制器集成于名为FiReFly的算法中。对于模拟的接近-避免任务，FiReFly生成了更加公平的轨迹，并提高了任务成功率，超过了非公平规划器的表现。我们发现实时性能在15架无人机时是可行的，而在扩展到50架无人机时，可以通过牺牲运行时间和公平性改进来实现。 

---
# Fair-CoPlan: Negotiated Flight Planning with Fair Deconfliction for Urban Air Mobility 

**Title (ZH)**: 公平-CoPlan：具公平解冲突的飞行规划方法研究（适用于城市空中 mobility） 

**Authors**: Nicole Fronda, Phil Smith, Bardh Hoxha, Yash Pant, Houssam Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2508.14380)  

**Abstract**: Urban Air Mobility (UAM) is an emerging transportation paradigm in which Uncrewed Aerial Systems (UAS) autonomously transport passengers and goods in cities. The UAS have different operators with different, sometimes competing goals, yet must share the airspace. We propose a negotiated, semi-distributed flight planner that optimizes UAS' flight lengths {\em in a fair manner}. Current flight planners might result in some UAS being given disproportionately shorter flight paths at the expense of others. We introduce Fair-CoPlan, a planner in which operators and a Provider of Service to the UAM (PSU) together compute \emph{fair} flight paths. Fair-CoPlan has three steps: First, the PSU constrains take-off and landing choices for flights based on capacity at and around vertiports. Then, operators plan independently under these constraints. Finally, the PSU resolves any conflicting paths, optimizing for path length fairness. By fairly spreading the cost of deconfliction Fair-CoPlan encourages wider participation in UAM, ensures safety of the airspace and the areas below it, and promotes greater operator flexibility. We demonstrate Fair-CoPlan through simulation experiments and find fairer outcomes than a non-fair planner with minor delays as a trade-off. 

**Abstract (ZH)**: 城市空中交通(UAM)中的公平协同规划：Uncrewed Aerial Systems (UAS)在城市中自主运输乘客和货物的新兴交通模式中，不同的运营商有着不同的目标，必须共享 airspace。我们提出了一种协商式的半分布式飞行规划器，以公平的方式优化UAS的飞行路径长度。当前的飞行规划器可能会导致一些UAS的飞行路径不公正地被缩短，以牺牲其他UAS为代价。我们引入了Fair-CoPlan规划器，在该规划器中，运营商和服务提供商共同计算公平的飞行路径。Fair-CoPlan包含三个步骤：首先，服务提供商根据 vertiport 的容量及其周边条件限制起飞和降落的选择；其次，运营商在这些限制下独立规划飞行路径；最后，服务提供商解决任何冲突路径，优化路径长度的公平性。通过公平地分摊解冲突的成本，Fair-CoPlan鼓励更广泛地参与UAM、确保空域及其下方区域的安全，并促进运营商更大的灵活性。我们通过仿真实验展示了Fair-CoPlan，并发现其结果比非公平规划器更公平，同时存在微小的延误作为权衡。 

---
# Action-Constrained Imitation Learning 

**Title (ZH)**: 动作约束 imitation 学习 

**Authors**: Chia-Han Yeh, Tse-Sheng Nan, Risto Vuorio, Wei Hung, Hung-Yen Wu, Shao-Hua Sun, Ping-Chun Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2508.14379)  

**Abstract**: Policy learning under action constraints plays a central role in ensuring safe behaviors in various robot control and resource allocation applications. In this paper, we study a new problem setting termed Action-Constrained Imitation Learning (ACIL), where an action-constrained imitator aims to learn from a demonstrative expert with larger action space. The fundamental challenge of ACIL lies in the unavoidable mismatch of occupancy measure between the expert and the imitator caused by the action constraints. We tackle this mismatch through \textit{trajectory alignment} and propose DTWIL, which replaces the original expert demonstrations with a surrogate dataset that follows similar state trajectories while adhering to the action constraints. Specifically, we recast trajectory alignment as a planning problem and solve it via Model Predictive Control, which aligns the surrogate trajectories with the expert trajectories based on the Dynamic Time Warping (DTW) distance. Through extensive experiments, we demonstrate that learning from the dataset generated by DTWIL significantly enhances performance across multiple robot control tasks and outperforms various benchmark imitation learning algorithms in terms of sample efficiency. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 带动作约束的模仿学习在确保各类机器人控制和资源分配应用中的安全行为中起着关键作用。在本文中，我们研究了一种新的问题设定，称为带动作约束的模仿学习（ACIL），其中带动作约束的模仿者旨在从具有更大动作空间的示范专家那里学习。ACIL的基本挑战在于由动作约束引起的专家和模仿者之间的不可避免的态势分布不匹配。我们通过轨迹对齐来应对这种不匹配，并提出DTWIL，它用一个遵循相似状态轨迹同时遵守动作约束的替代数据集来替换原始专家示范。具体而言，我们将轨迹对齐重新定义为一个规划问题，并通过模型预测控制来解决，基于动态时间规整（DTW）距离对替代轨迹进行对齐。通过广泛的实验，我们证明了通过DTWIL生成的数据集在多种机器人控制任务中显著提升了性能，并在样本效率方面优于各种基准模仿学习算法。我们的代码可在如下链接公开获取：this https URL。 

---
# D$^2$-LIO: Enhanced Optimization for LiDAR-IMU Odometry Considering Directional Degeneracy 

**Title (ZH)**: D$^2$-LIO：考虑方向退化时的LiDAR-IMU里程计增强优化 

**Authors**: Guodong Yao, Hao Wang, Qing Chang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14355)  

**Abstract**: LiDAR-inertial odometry (LIO) plays a vital role in achieving accurate localization and mapping, especially in complex environments. However, the presence of LiDAR feature degeneracy poses a major challenge to reliable state estimation. To overcome this issue, we propose an enhanced LIO framework that integrates adaptive outlier-tolerant correspondence with a scan-to-submap registration strategy. The core contribution lies in an adaptive outlier removal threshold, which dynamically adjusts based on point-to-sensor distance and the motion amplitude of platform. This mechanism improves the robustness of feature matching in varying conditions. Moreover, we introduce a flexible scan-to-submap registration method that leverages IMU data to refine pose estimation, particularly in degenerate geometric configurations. To further enhance localization accuracy, we design a novel weighting matrix that fuses IMU preintegration covariance with a degeneration metric derived from the scan-to-submap process. Extensive experiments conducted in both indoor and outdoor environments-characterized by sparse or degenerate features-demonstrate that our method consistently outperforms state-of-the-art approaches in terms of both robustness and accuracy. 

**Abstract (ZH)**: LiDAR-惯性里程计（LIO）在实现精确定位和建图中起着关键作用，尤其是在复杂环境中。然而，LiDAR特征退化现象对可靠状态估计构成了重大挑战。为克服这一问题，我们提出了一种增强的LIO框架，该框架结合了自适应鲁棒对应关系和扫描到子地图注册策略。核心贡献在于一种自适应离群值移除阈值，该阈值根据点到传感器距离和平台的运动幅度动态调整，从而在不同条件下提高特征匹配的鲁棒性。此外，我们引入了一种灵活的扫描到子地图注册方法，该方法利用IMU数据来细化姿态估计，特别是在退化几何构型中。为了进一步提高定位精度，我们设计了一种新的加权矩阵，该矩阵将IMU前积分协方差与从扫描到子地图过程提取的退化度度量融合起来。在包括稀疏或退化特征在内的室内外环境中进行的广泛实验表明，我们的方法在鲁棒性和精度方面均优于现有先进方法。 

---
# Adapting Biological Reflexes for Dynamic Reorientation in Space Manipulator Systems 

**Title (ZH)**: 适应生物学反射的空间 manipulator 系统动态重定向 

**Authors**: Daegyun Choi, Alhim Vera, Donghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.14258)  

**Abstract**: Robotic arms mounted on spacecraft, known as space manipulator systems (SMSs), are critical for enabling on-orbit assembly, satellite servicing, and debris removal. However, controlling these systems in microgravity remains a significant challenge due to the dynamic coupling between the manipulator and the spacecraft base. This study explores the potential of using biological inspiration to address this issue, focusing on animals, particularly lizards, that exhibit mid-air righting reflexes. Based on similarities between SMSs and these animals in terms of behavior, morphology, and environment, their air-righting motion trajectories are extracted from high-speed video recordings using computer vision techniques. These trajectories are analyzed within a multi-objective optimization framework to identify the key behavioral goals and assess their relative importance. The resulting motion profiles are then applied as reference trajectories for SMS control, with baseline controllers used to track them. The findings provide a step toward translating evolved animal behaviors into interpretable, adaptive control strategies for space robotics, with implications for improving maneuverability and robustness in future missions. 

**Abstract (ZH)**: 基于生物启发的空间操纵系统在微重力环境下空气翻转运动轨迹的研究及其在空间机器人控制中的应用 

---
# SLAM-based Safe Indoor Exploration Strategy 

**Title (ZH)**: 基于SLAM的安全室内探索策略 

**Authors**: Omar Mostafa, Nikolaos Evangeliou, Anthony Tzes  

**Link**: [PDF](https://arxiv.org/pdf/2508.14235)  

**Abstract**: This paper suggests a 2D exploration strategy for a planar space cluttered with obstacles. Rather than using point robots capable of adjusting their position and altitude instantly, this research is tailored to classical agents with circular footprints that cannot control instantly their pose. Inhere, a self-balanced dual-wheeled differential drive system is used to explore the place. The system is equipped with linear accelerometers and angular gyroscopes, a 3D-LiDAR, and a forward-facing RGB-D camera. The system performs RTAB-SLAM using the IMU and the LiDAR, while the camera is used for loop closures. The mobile agent explores the planar space using a safe skeleton approach that places the agent as far as possible from the static obstacles. During the exploration strategy, the heading is towards any offered openings of the space. This space exploration strategy has as its highest priority the agent's safety in avoiding the obstacles followed by the exploration of undetected space. Experimental studies with a ROS-enabled mobile agent are presented indicating the path planning strategy while exploring the space. 

**Abstract (ZH)**: 本文建议了一种用于充满障碍的平面空间的2D探索策略。该研究适用于无法即时控制姿态的经典圆形足式机器人，使用自平衡双轮差动驱动系统进行探索。该系统配备线性加速度计和角速率陀螺仪、3D-LiDAR以及前置RGB-D相机。系统使用IMU和LiDAR进行RTAB-SLAM，而相机用于环视闭合。移动代理使用安全骨架方法探索平面空间，尽可能远离静态障碍物。在探索策略中，方向指向空间中的任何开口。该空间探索策略的最高优先级是代理的安全性，以避免障碍物，其次是探索未发现的空间。呈现了使用ROS启用的移动代理的实验研究，表明在探索空间时的路径规划策略。 

---
# Lightweight Tracking Control for Computationally Constrained Aerial Systems with the Newton-Raphson Method 

**Title (ZH)**: 基于牛顿-拉夫森方法的计算受限空中系统轻量跟踪控制 

**Authors**: Evanns Morales-Cuadrado, Luke Baird, Yorai Wardi, Samuel Coogan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14185)  

**Abstract**: We investigate the performance of a lightweight tracking controller, based on a flow version of the Newton-Raphson method, applied to a miniature blimp and a mid-size quadrotor. This tracking technique has been shown to enjoy theoretical guarantees of performance and has been applied with success in simulation studies and on mobile robots with simple motion models. This paper investigates the technique through real-world flight experiments on aerial hardware platforms subject to realistic deployment and onboard computational constraints. The technique's performance is assessed in comparison with the established control frameworks of feedback linearization for the blimp, and nonlinear model predictive control for both quadrotor and blimp. The performance metrics under consideration are (i) root mean square error of flight trajectories with respect to target trajectories, (ii) algorithms' computation times, and (iii) CPU energy consumption associated with the control algorithms. The experimental findings show that the Newton-Raphson flow-based tracking controller achieves comparable or superior tracking performance to the baseline methods with substantially reduced computation time and energy expenditure. 

**Abstract (ZH)**: 我们调查了一种基于流版本的Newton-Raphson方法的轻量级跟踪控制器在微型气球和中型四旋翼无人机上的应用性能。 

---
# SimGenHOI: Physically Realistic Whole-Body Humanoid-Object Interaction via Generative Modeling and Reinforcement Learning 

**Title (ZH)**: SimGenHOI: 通过生成建模和强化学习实现物理上真实的全身类人偶与物体交互 

**Authors**: Yuhang Lin, Yijia Xie, Jiahong Xie, Yuehao Huang, Ruoyu Wang, Jiajun Lv, Yukai Ma, Xingxing Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14120)  

**Abstract**: Generating physically realistic humanoid-object interactions (HOI) is a fundamental challenge in robotics. Existing HOI generation approaches, such as diffusion-based models, often suffer from artifacts such as implausible contacts, penetrations, and unrealistic whole-body actions, which hinder successful execution in physical environments. To address these challenges, we introduce SimGenHOI, a unified framework that combines the strengths of generative modeling and reinforcement learning to produce controllable and physically plausible HOI. Our HOI generative model, based on Diffusion Transformers (DiT), predicts a set of key actions conditioned on text prompts, object geometry, sparse object waypoints, and the initial humanoid pose. These key actions capture essential interaction dynamics and are interpolated into smooth motion trajectories, naturally supporting long-horizon generation. To ensure physical realism, we design a contact-aware whole-body control policy trained with reinforcement learning, which tracks the generated motions while correcting artifacts such as penetration and foot sliding. Furthermore, we introduce a mutual fine-tuning strategy, where the generative model and the control policy iteratively refine each other, improving both motion realism and tracking robustness. Extensive experiments demonstrate that SimGenHOI generates realistic, diverse, and physically plausible humanoid-object interactions, achieving significantly higher tracking success rates in simulation and enabling long-horizon manipulation tasks. Code will be released upon acceptance on our project page: this https URL. 

**Abstract (ZH)**: 生成物理上真实的类人物体交互（HOI）是机器人技术中的一个基本挑战。现有的HOI生成方法，如基于扩散的模型，往往会出现不合理的接触、穿透和不现实的整体动作等问题，这阻碍了在物理环境中的成功执行。为了解决这些问题，我们提出了SimGenHOI，一种结合生成建模和强化学习优势的统一框架，以生成可控和物理上合理的HOI。基于扩散变换器（DiT）的HOI生成模型，在文本提示、物体几何、稀疏物体航点和初始类人姿态的条件下，预测一组关键动作。这些关键动作捕捉到主要的交互动力学，并通过插值生成平滑的运动轨迹，自然支持长时间生成。为确保物理现实性，我们设计了一种基于接触感知的整体控制策略，并用强化学习进行训练，该策略跟踪生成的运动，同时修正穿透和脚滑等缺陷。此外，我们引入了一种互适应微调策略，生成模型和控制策略交替优化，提高运动的真实性和追踪鲁棒性。广泛的经验表明，SimGenHOI生成了真实、多样且物理上合理的类人物体交互，在模拟中实现了显著更高的追踪成功率，并能支持长时间操作任务。代码将在接受后发布在我们的项目页面：this https URL。 

---
# Efficient Environment Design for Multi-Robot Navigation via Continuous Control 

**Title (ZH)**: 多机器人导航的连续控制高效环境设计 

**Authors**: Jahid Chowdhury Choton, John Woods, William Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14105)  

**Abstract**: Multi-robot navigation and path planning in continuous state and action spaces with uncertain environments remains an open challenge. Deep Reinforcement Learning (RL) is one of the most popular paradigms for solving this task, but its real-world application has been limited due to sample inefficiency and long training periods. Moreover, the existing works using RL for multi-robot navigation lack formal guarantees while designing the environment. In this paper, we introduce an efficient and highly customizable environment for continuous-control multi-robot navigation, where the robots must visit a set of regions of interest (ROIs) by following the shortest paths. The task is formally modeled as a Markov Decision Process (MDP). We describe the multi-robot navigation task as an optimization problem and relate it to finding an optimal policy for the MDP. We crafted several variations of the environment and measured the performance using both gradient and non-gradient based RL methods: A2C, PPO, TRPO, TQC, CrossQ and ARS. To show real-world applicability, we deployed our environment to a 3-D agricultural field with uncertainties using the CoppeliaSim robot simulator and measured the robustness by running inference on the learned models. We believe our work will guide the researchers on how to develop MDP-based environments that are applicable to real-world systems and solve them using the existing state-of-the-art RL methods with limited resources and within reasonable time periods. 

**Abstract (ZH)**: 连续状态和动作空间以及不确定环境中多机器人导航和路径规划的研究仍然是一个开放的挑战。基于深度强化学习的连续控制多机器人导航高效自定义环境及其应用 

---
# Domain Translation of a Soft Robotic Arm using Conditional Cycle Generative Adversarial Network 

**Title (ZH)**: 使用条件循环生成对抗网络的软机器人臂领域转换 

**Authors**: Nilay Kushawaha, Carlo Alessi, Lorenzo Fruzzetti, Egidio Falotico  

**Link**: [PDF](https://arxiv.org/pdf/2508.14100)  

**Abstract**: Deep learning provides a powerful method for modeling the dynamics of soft robots, offering advantages over traditional analytical approaches that require precise knowledge of the robot's structure, material properties, and other physical characteristics. Given the inherent complexity and non-linearity of these systems, extracting such details can be challenging. The mappings learned in one domain cannot be directly transferred to another domain with different physical properties. This challenge is particularly relevant for soft robots, as their materials gradually degrade over time. In this paper, we introduce a domain translation framework based on a conditional cycle generative adversarial network (CCGAN) to enable knowledge transfer from a source domain to a target domain. Specifically, we employ a dynamic learning approach to adapt a pose controller trained in a standard simulation environment to a domain with tenfold increased viscosity. Our model learns from input pressure signals conditioned on corresponding end-effector positions and orientations in both domains. We evaluate our approach through trajectory-tracking experiments across five distinct shapes and further assess its robustness under noise perturbations and periodicity tests. The results demonstrate that CCGAN-GP effectively facilitates cross-domain skill transfer, paving the way for more adaptable and generalizable soft robotic controllers. 

**Abstract (ZH)**: 基于条件周期生成对抗网络的领域翻译框架：实现软机器人控制的知识迁移 

---
# Task and Motion Planning for Humanoid Loco-manipulation 

**Title (ZH)**: humanoid 任务与运动规划 

**Authors**: Michal Ciebielski, Victor Dhédin, Majid Khadiv  

**Link**: [PDF](https://arxiv.org/pdf/2508.14099)  

**Abstract**: This work presents an optimization-based task and motion planning (TAMP) framework that unifies planning for locomotion and manipulation through a shared representation of contact modes. We define symbolic actions as contact mode changes, grounding high-level planning in low-level motion. This enables a unified search that spans task, contact, and motion planning while incorporating whole-body dynamics, as well as all constraints between the robot, the manipulated object, and the environment. Results on a humanoid platform show that our method can generate a broad range of physically consistent loco-manipulation behaviors over long action sequences requiring complex reasoning. To the best of our knowledge, this is the first work that enables the resolution of an integrated TAMP formulation with fully acyclic planning and whole body dynamics with actuation constraints for the humanoid loco-manipulation problem. 

**Abstract (ZH)**: 基于优化的任务与运动规划框架：通过共享的接触模式表示实现 locomotion 和 manipulation 的统一规划 

---
# No More Marching: Learning Humanoid Locomotion for Short-Range SE(2) Targets 

**Title (ZH)**: 不再徒步：学习 humanoid 短程 SE(2) 位姿目标的行走动作 

**Authors**: Pranay Dugar, Mohitvishnu S. Gadde, Jonah Siekmann, Yesh Godse, Aayam Shrestha, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2508.14098)  

**Abstract**: Humanoids operating in real-world workspaces must frequently execute task-driven, short-range movements to SE(2) target poses. To be practical, these transitions must be fast, robust, and energy efficient. While learning-based locomotion has made significant progress, most existing methods optimize for velocity-tracking rather than direct pose reaching, resulting in inefficient, marching-style behavior when applied to short-range tasks. In this work, we develop a reinforcement learning approach that directly optimizes humanoid locomotion for SE(2) targets. Central to this approach is a new constellation-based reward function that encourages natural and efficient target-oriented movement. To evaluate performance, we introduce a benchmarking framework that measures energy consumption, time-to-target, and footstep count on a distribution of SE(2) goals. Our results show that the proposed approach consistently outperforms standard methods and enables successful transfer from simulation to hardware, highlighting the importance of targeted reward design for practical short-range humanoid locomotion. 

**Abstract (ZH)**: 实时工作空间中的人形机器人必须频繁执行任务驱动的短距离SE(2)目标位姿移动。为了实用，这些过渡必须快速、Robust且能效高。尽管基于学习的行进已取得显著进展，但现有大多数方法优化速度跟踪而非直接姿态到达，导致在短距离任务中表现出低效的行进步态。在本工作中，我们开发了一种强化学习方法，直接优化人形机器人对SE(2)目标的姿态移动。这种方法的核心是一种新的星座基于的奖励函数，鼓励自然且高效的目标导向移动。为了评估性能，我们引入了一个基准框架，该框架在SE(2)目标的分布上衡量能耗、到达时间和步数。实验结果表明，所提出的方法始终优于标准方法，并能够在仿真与硬件之间实现成功的转移，突显了针对具体任务奖励设计对于实用的短距离人形机器人移动的重要性。 

---
# Research on UAV Applications in Public Administration: Based on an Improved RRT Algorithm 

**Title (ZH)**: 基于改进RRT算法的无人机在公共管理中的应用研究 

**Authors**: Zhanxi Xie, Baili Lu, Yanzhao Gu, Zikun Li, Junhao Wei, Ngai Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2508.14096)  

**Abstract**: This study investigates the application of unmanned aerial vehicles (UAVs) in public management, focusing on optimizing path planning to address challenges such as energy consumption, obstacle avoidance, and airspace constraints. As UAVs transition from 'technical tools' to 'governance infrastructure', driven by advancements in low-altitude economy policies and smart city demands, efficient path planning becomes critical. The research proposes an enhanced Rapidly-exploring Random Tree algorithm (dRRT), incorporating four strategies: Target Bias (to accelerate convergence), Dynamic Step Size (to balance exploration and obstacle navigation), Detour Priority (to prioritize horizontal detours over vertical ascents), and B-spline smoothing (to enhance path smoothness). Simulations in a 500 m3 urban environment with randomized buildings demonstrate dRRT's superiority over traditional RRT, A*, and Ant Colony Optimization (ACO). Results show dRRT achieves a 100\% success rate with an average runtime of 0.01468s, shorter path lengths, fewer waypoints, and smoother trajectories (maximum yaw angles <45°). Despite improvements, limitations include increased computational overhead from added mechanisms and potential local optima due to goal biasing. The study highlights dRRT's potential for efficient UAV deployment in public management scenarios like emergency response and traffic monitoring, while underscoring the need for integration with real-time obstacle avoidance frameworks. This work contributes to interdisciplinary advancements in urban governance, robotics, and computational optimization. 

**Abstract (ZH)**: 本研究探讨了无人机在公共管理中的应用，重点优化路径规划以应对能耗、障碍物规避和 airspace限制等挑战。随着无人机从“技术工具”转变为“治理基础设施”，受低空经济政策和智能城市需求推动，高效路径规划变得至关重要。研究提出了增强型快速扩展随机树算法（dRRT），结合了四种策略：目标偏置（加速收敛）、动态步长（平衡探索与障碍物导航）、绕行优先（优先水平绕行而非垂直上升）和B样条平滑（提高路径平滑度）。在具有随机建筑的500 m³城市环境中模拟结果显示，dRRT在成功率、运行时间、路径长度、航点数量和轨迹平滑度方面优于传统RRT、A*和蚂蚁 colony优化算法（ACO）。结果表明，dRRT 的成功率达到了100%，平均运行时间为0.01468秒，路径较短、航点较少且轨迹更加平滑（最大偏航角<45°）。尽管有所改进，但仍存在因增加机制而导致的计算开销增加以及目标偏置可能陷入局部最优的问题。本研究强调了dRRT在应急响应和交通监控等公共管理场景中高效无人机部署的潜力，并指出了与实时障碍物规避框架集成的必要性。本研究为城市治理、机器人技术和计算优化的跨学科进步做出了贡献。 

---
# Virtual Community: An Open World for Humans, Robots, and Society 

**Title (ZH)**: 虚拟社区：人类、机器人与社会的开放世界 

**Authors**: Qinhong Zhou, Hongxin Zhang, Xiangye Lin, Zheyuan Zhang, Yutian Chen, Wenjun Liu, Zunzhe Zhang, Sunli Chen, Lixing Fang, Qiushi Lyu, Xinyu Sun, Jincheng Yang, Zeyuan Wang, Bao Chi Dang, Zhehuan Chen, Daksha Ladia, Jiageng Liu, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14893)  

**Abstract**: The rapid progress in AI and Robotics may lead to a profound societal transformation, as humans and robots begin to coexist within shared communities, introducing both opportunities and challenges. To explore this future, we present Virtual Community-an open-world platform for humans, robots, and society-built on a universal physics engine and grounded in real-world 3D scenes. With Virtual Community, we aim to study embodied social intelligence at scale: 1) How robots can intelligently cooperate or compete; 2) How humans develop social relations and build community; 3) More importantly, how intelligent robots and humans can co-exist in an open world. To support these, Virtual Community features: 1) An open-source multi-agent physics simulator that supports robots, humans, and their interactions within a society; 2) A large-scale, real-world aligned community generation pipeline, including vast outdoor space, diverse indoor scenes, and a community of grounded agents with rich characters and appearances. Leveraging Virtual Community, we propose two novel challenges. The Community Planning Challenge evaluates multi-agent reasoning and planning ability in open-world settings, such as cooperating to help agents with daily activities and efficiently connecting other agents. The Community Robot Challenge requires multiple heterogeneous robots to collaborate in solving complex open-world tasks. We evaluate various baselines on these tasks and demonstrate the challenges in both high-level open-world task planning and low-level cooperation controls. We hope that Virtual Community will unlock further study of human-robot coexistence within open-world environments. 

**Abstract (ZH)**: AI和机器人技术的迅速发展可能导致社会深刻的转型，人类和机器人开始在共享社区中共存，带来机遇和挑战。为探索这一未来，我们提出虚拟社区——一个基于通用物理引擎且与现实世界3D场景相匹配的开放世界平台，供人类、机器人和社会使用。借助虚拟社区，我们旨在大规模研究具身社交智能：1）机器人如何智能地合作或竞争；2）人类如何发展社交关系并构建社区；3）更重要的是，智能机器人和人类如何在开放世界中共存。为此，虚拟社区具备以下功能：1）开源多智能体物理模拟器，支持机器人、人类及其在社会中的互动；2）大规模、与真实世界对齐的社区生成流水线，包括广阔的户外空间、多样的室内场景以及一个包含丰富人物和外观的真实代理群体。利用虚拟社区，我们提出了两项新颖挑战。社区规划挑战评估多智能体在开放世界中的推理和规划能力，如协作帮助代理完成日常活动和有效连接其他代理。社区机器人挑战要求多个异构机器人协作解决复杂的开放世界任务。我们在这些任务上评估了各种baseline，并展示了高层开放世界任务规划和低层协作控制的挑战。我们希望虚拟社区能够促进对开放世界环境中的人机共存研究。 

---
# Fusing Monocular RGB Images with AIS Data to Create a 6D Pose Estimation Dataset for Marine Vessels 

**Title (ZH)**: 将单目RGB图像与AIS数据融合以创建用于海洋船只的6D姿态估计数据集 

**Authors**: Fabian Holst, Emre Gülsoylu, Simone Frintrop  

**Link**: [PDF](https://arxiv.org/pdf/2508.14767)  

**Abstract**: The paper presents a novel technique for creating a 6D pose estimation dataset for marine vessels by fusing monocular RGB images with Automatic Identification System (AIS) data. The proposed technique addresses the limitations of relying purely on AIS for location information, caused by issues like equipment reliability, data manipulation, and transmission delays. By combining vessel detections from monocular RGB images, obtained using an object detection network (YOLOX-X), with AIS messages, the technique generates 3D bounding boxes that represent the vessels' 6D poses, i.e. spatial and rotational dimensions. The paper evaluates different object detection models to locate vessels in image space. We also compare two transformation methods (homography and Perspective-n-Point) for aligning AIS data with image coordinates. The results of our work demonstrate that the Perspective-n-Point (PnP) method achieves a significantly lower projection error compared to homography-based approaches used before, and the YOLOX-X model achieves a mean Average Precision (mAP) of 0.80 at an Intersection over Union (IoU) threshold of 0.5 for relevant vessel classes. We show indication that our approach allows the creation of a 6D pose estimation dataset without needing manual annotation. Additionally, we introduce the Boats on Nordelbe Kehrwieder (BONK-pose), a publicly available dataset comprising 3753 images with 3D bounding box annotations for pose estimation, created by our data fusion approach. This dataset can be used for training and evaluating 6D pose estimation networks. In addition we introduce a set of 1000 images with 2D bounding box annotations for ship detection from the same scene. 

**Abstract (ZH)**: 一种将单目RGB图像与自动识别系统(AIS)数据融合以创建船舶6D姿态估计数据集的新技术及其应用 

---
# Making Pose Representations More Expressive and Disentangled via Residual Vector Quantization 

**Title (ZH)**: 通过残差向量量化使姿态表示更加富有表现力和解耦 

**Authors**: Sukhyun Jeong, Hong-Gi Shin, Yong-Hoon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14561)  

**Abstract**: Recent progress in text-to-motion has advanced both 3D human motion generation and text-based motion control. Controllable motion generation (CoMo), which enables intuitive control, typically relies on pose code representations, but discrete pose codes alone cannot capture fine-grained motion details, limiting expressiveness. To overcome this, we propose a method that augments pose code-based latent representations with continuous motion features using residual vector quantization (RVQ). This design preserves the interpretability and manipulability of pose codes while effectively capturing subtle motion characteristics such as high-frequency details. Experiments on the HumanML3D dataset show that our model reduces Frechet inception distance (FID) from 0.041 to 0.015 and improves Top-1 R-Precision from 0.508 to 0.510. Qualitative analysis of pairwise direction similarity between pose codes further confirms the model's controllability for motion editing. 

**Abstract (ZH)**: Recent进展在文本到运动转换中推动了3D人类运动生成和基于文本的运动控制。我们提出的方法通过残差向量量化(RVQ)增强基于姿态码的潜在表示以包含连续运动特征，从而克服了仅使用离散姿态码无法捕捉到精细运动细节的限制，提高了表达能力。实验结果表明，我们的模型将Frechet inception距离(FID)从0.041降低到0.015，Top-1 R-精确度从0.508提高到0.510。对姿态码的成对方向相似性分析进一步证实了模型在运动编辑中的可控性。 

---
# Learning Point Cloud Representations with Pose Continuity for Depth-Based Category-Level 6D Object Pose Estimation 

**Title (ZH)**: 基于姿态连续性的点云表示学习用于深度导向的类别级6D物体姿态估计 

**Authors**: Zhujun Li, Shuo Zhang, Ioannis Stamos  

**Link**: [PDF](https://arxiv.org/pdf/2508.14358)  

**Abstract**: Category-level object pose estimation aims to predict the 6D pose and 3D size of objects within given categories. Existing approaches for this task rely solely on 6D poses as supervisory signals without explicitly capturing the intrinsic continuity of poses, leading to inconsistencies in predictions and reduced generalization to unseen poses. To address this limitation, we propose HRC-Pose, a novel depth-only framework for category-level object pose estimation, which leverages contrastive learning to learn point cloud representations that preserve the continuity of 6D poses. HRC-Pose decouples object pose into rotation and translation components, which are separately encoded and leveraged throughout the network. Specifically, we introduce a contrastive learning strategy for multi-task, multi-category scenarios based on our 6D pose-aware hierarchical ranking scheme, which contrasts point clouds from multiple categories by considering rotational and translational differences as well as categorical information. We further design pose estimation modules that separately process the learned rotation-aware and translation-aware embeddings. Our experiments demonstrate that HRC-Pose successfully learns continuous feature spaces. Results on REAL275 and CAMERA25 benchmarks show that our method consistently outperforms existing depth-only state-of-the-art methods and runs in real-time, demonstrating its effectiveness and potential for real-world applications. Our code is at this https URL. 

**Abstract (ZH)**: 类别级对象姿态估计旨在预测给定类别内物体的6D姿态和3D尺寸。现有方法仅依赖于6D姿态作为监督信号，而未明确捕捉姿态的内在连续性，导致预测不一致并降低对未见姿态的泛化能力。为解决这一局限，我们提出了一种基于对比学习的新颖单目深度框架HRC-Pose，该框架利用对比学习学习保留6D姿态连续性的点云表示。HRC-Pose将物体姿态分解为旋转和平移组件，分别进行编码并在网络中利用。具体地，我们基于6D姿态感知的分层排名方案引入了一种多任务、多类别场景下的对比学习策略，通过考虑旋转和平移差异以及类别信息对比来自多个类别的点云。我们进一步设计了姿态估计模块，分别处理旋转感知和平移感知的嵌入。实验结果表明，HRC-Pose成功学习了连续特征空间。在REAL275和CAMERA25基准上的结果表明，我们的方法在实时运行的同时，始终优于现有单目深度最先进的方法，展示了其有效性及在实际应用中的潜力。相关代码链接为：https://github.com/HRCPose/HRC-Pose。 

---
# Towards Unified Probabilistic Verification and Validation of Vision-Based Autonomy 

**Title (ZH)**: 基于视觉的自主性统一概率验证与验证方法研究 

**Authors**: Jordan Peper, Yan Miao, Sayan Mitra, Ivan Ruchkin  

**Link**: [PDF](https://arxiv.org/pdf/2508.14181)  

**Abstract**: Precise and comprehensive situational awareness is a critical capability of modern autonomous systems. Deep neural networks that perceive task-critical details from rich sensory signals have become ubiquitous; however, their black-box behavior and sensitivity to environmental uncertainty and distribution shifts make them challenging to verify formally. Abstraction-based verification techniques for vision-based autonomy produce safety guarantees contingent on rigid assumptions, such as bounded errors or known unique distributions. Such overly restrictive and inflexible assumptions limit the validity of the guarantees, especially in diverse and uncertain test-time environments. We propose a methodology that unifies the verification models of perception with their offline validation. Our methodology leverages interval MDPs and provides a flexible end-to-end guarantee that adapts directly to the out-of-distribution test-time conditions. We evaluate our methodology on a synthetic perception Markov chain with well-defined state estimation distributions and a mountain car benchmark. Our findings reveal that we can guarantee tight yet rigorous bounds on overall system safety. 

**Abstract (ZH)**: 现代自主系统中精确而全面的情境意识是一项关键能力。基于深度神经网络的感知技术能够从丰富的感官信号中识别出任务关键细节，但其黑箱行为及对环境不确定性和分布偏移的敏感性使其难以进行正式验证。基于抽象的视觉自主性验证技术依赖于严格的假设，如有限的误差或已知的独特分布，这些过度限制且不灵活的假设限制了保证的有效性，尤其是在多样性和不确定性的测试环境。我们提出了一种统一感知验证模型与其离线验证的方法论。该方法论利用区间MDP，并提供了灵活的端到端保证，能够直接适应超出分布的测试条件。我们通过一个定义明确状态估计分布的合成感知马尔可夫链和一个山车基准进行了评估。我们的研究发现，我们能够确保整个系统的安全性的紧致而严格的界限。 

---
# RynnEC: Bringing MLLMs into Embodied World 

**Title (ZH)**: RynnEC: 将MLLMs引入具身世界 

**Authors**: Ronghao Dang, Yuqian Yuan, Yunxuan Mao, Kehan Li, Jiangpin Liu, Zhikai Wang, Xin Li, Fan Wang, Deli Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14160)  

**Abstract**: We introduce RynnEC, a video multimodal large language model designed for embodied cognition. Built upon a general-purpose vision-language foundation model, RynnEC incorporates a region encoder and a mask decoder, enabling flexible region-level video interaction. Despite its compact architecture, RynnEC achieves state-of-the-art performance in object property understanding, object segmentation, and spatial reasoning. Conceptually, it offers a region-centric video paradigm for the brain of embodied agents, providing fine-grained perception of the physical world and enabling more precise interactions. To mitigate the scarcity of annotated 3D datasets, we propose an egocentric video based pipeline for generating embodied cognition data. Furthermore, we introduce RynnEC-Bench, a region-centered benchmark for evaluating embodied cognitive capabilities. We anticipate that RynnEC will advance the development of general-purpose cognitive cores for embodied agents and facilitate generalization across diverse embodied tasks. The code, model checkpoints, and benchmark are available at: this https URL 

**Abstract (ZH)**: 我们介绍RynnEC，这是一种用于 embodied cognition的视频多模态大型语言模型。基于通用的视觉-语言基础模型，RynnEC整合了区域编码器和掩码解码器，使其实现灵活的区域级视频交互。尽管其架构紧凑，但在对象属性理解、对象分割和空间推理方面仍实现了最先进的性能。从概念上讲，RynnEC提供了一种以区域为中心的视频范式，为embodied智能体的大脑提供了精细的物理世界感知能力，并促进了更精确的交互。为了缓解标注的3D数据集稀缺问题，我们提出了一种基于第一人称视频的生成embodied cognition数据的流程。此外，我们引入了RynnEC-Bench，这是一种以区域为中心的评估embodied认知能力的基准。我们期望RynnEC能推动通用认知核心的发展，并促进跨多种embodied任务的泛化。代码、模型检查点和基准可在以下链接获取：this https URL。 

---
# Beyond Fixed Morphologies: Learning Graph Policies with Trust Region Compensation in Variable Action Spaces 

**Title (ZH)**: 超越固定形态：在可变动作空间中通过信任区域补偿学习图策略 

**Authors**: Thomas Gallien  

**Link**: [PDF](https://arxiv.org/pdf/2508.14102)  

**Abstract**: Trust region-based optimization methods have become foundational reinforcement learning algorithms that offer stability and strong empirical performance in continuous control tasks. Growing interest in scalable and reusable control policies translate also in a demand for morphological generalization, the ability of control policies to cope with different kinematic structures. Graph-based policy architectures provide a natural and effective mechanism to encode such structural differences. However, while these architectures accommodate variable morphologies, the behavior of trust region methods under varying action space dimensionality remains poorly understood. To this end, we conduct a theoretical analysis of trust region-based policy optimization methods, focusing on both Trust Region Policy Optimization (TRPO) and its widely used first-order approximation, Proximal Policy Optimization (PPO). The goal is to demonstrate how varying action space dimensionality influence the optimization landscape, particularly under the constraints imposed by KL-divergence or policy clipping penalties. Complementing the theoretical insights, an empirical evaluation under morphological variation is carried out using the Gymnasium Swimmer environment. This benchmark offers a systematically controlled setting for varying the kinematic structure without altering the underlying task, making it particularly well-suited to study morphological generalization. 

**Abstract (ZH)**: 基于信任域的优化方法已成为连续控制任务中提供稳定性和强大实证性能的强化学习基础算法。随着对可扩展且可重用控制策略兴趣的增长，对形态通用性的需求也随之增加，即控制策略能够应对不同的运动学结构的能力。基于图的策略架构自然且有效地编码了这些结构差异。然而，尽管这些架构能够适应变化的形态，但信任域方法在变化的动作空间维度下的行为仍知之甚少。为此，我们对基于信任域的策略优化方法进行了理论分析，重点关注信任域策略优化（TRPO）及其广泛使用的近似方法近端策略优化（PPO）。我们的目标是展示变化的动作空间维度如何影响优化景观，特别是在由KL散度或策略剪切惩罚所施加的约束下。通过Gymnasium Swimmer环境进行的经验评估补充了这些理论洞察。该基准提供了一种系统控制的变化运动学结构设置，而不改变底层任务，使其特别适用于研究形态通用性。 

---
