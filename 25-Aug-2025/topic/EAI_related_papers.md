# Terrain Classification for the Spot Quadrupedal Mobile Robot Using Only Proprioceptive Sensing 

**Title (ZH)**: 基于 proprioceptive 感知的 Spot 八足移动机器人地形分类 

**Authors**: Sophie Villemure, Jefferson Silveira, Joshua A. Marshall  

**Link**: [PDF](https://arxiv.org/pdf/2508.16504)  

**Abstract**: Quadrupedal mobile robots can traverse a wider range of terrain types than their wheeled counterparts but do not perform the same on all terrain types. These robots are prone to undesirable behaviours like sinking and slipping on challenging terrains. To combat this issue, we propose a terrain classifier that provides information on terrain type that can be used in robotic systems to create a traversability map to plan safer paths for the robot to navigate. The work presented here is a terrain classifier developed for a Boston Dynamics Spot robot. Spot provides over 100 measured proprioceptive signals describing the motions of the robot and its four legs (e.g., foot penetration, forces, joint angles, etc.). The developed terrain classifier combines dimensionality reduction techniques to extract relevant information from the signals and then applies a classification technique to differentiate terrain based on traversability. In representative field testing, the resulting terrain classifier was able to identify three different terrain types with an accuracy of approximately 97% 

**Abstract (ZH)**: 四足移动机器人能够跨越比轮式机器人更广泛的地形类型，但在所有地形类型上的表现不尽相同。这些机器人在挑战性地形上可能出现下沉和打滑等不良行为。为应对这一问题，我们提出了一种地形分类器，该分类器可以提供地形类型信息，用于机器人系统创建可通行性地图，以规划机器人导航的安全路径。本文展示的工作是专门为Boston Dynamics Spot机器人开发的一种地形分类器。Spot提供了超过100个测得的 proprioceptive 信号，描述了机器人的运动及其四条腿（如脚的穿透力、力、关节角度等）。所开发的地形分类器结合了降维技术以提取信号中的相关信息，然后应用分类技术根据可通行性对地形进行区分。在代表性实地测试中，该地形分类器能够识别三种不同地形类型，准确率约为97%。 

---
# GPL-SLAM: A Laser SLAM Framework with Gaussian Process Based Extended Landmarks 

**Title (ZH)**: GPL-SLAM: 基于高斯过程扩展地标的一种激光SLAM框架 

**Authors**: Ali Emre Balcı, Erhan Ege Keyvan, Emre Özkan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16459)  

**Abstract**: We present a novel Simultaneous Localization and Mapping (SLAM) method that employs Gaussian Process (GP) based landmark (object) representations. Instead of conventional grid maps or point cloud registration, we model the environment on a per object basis using GP based contour representations. These contours are updated online through a recursive scheme, enabling efficient memory usage. The SLAM problem is formulated within a fully Bayesian framework, allowing joint inference over the robot pose and object based map. This representation provides semantic information such as the number of objects and their areas, while also supporting probabilistic measurement to object associations. Furthermore, the GP based contours yield confidence bounds on object shapes, offering valuable information for downstream tasks like safe navigation and exploration. We validate our method on synthetic and real world experiments, and show that it delivers accurate localization and mapping performance across diverse structured environments. 

**Abstract (ZH)**: 基于高斯过程的物体表示的同时定位与mapping方法 

---
# Spatial Policy: Guiding Visuomotor Robotic Manipulation with Spatial-Aware Modeling and Reasoning 

**Title (ZH)**: 空间政策：基于空间感知建模与推理的视觉运动机器人操作指导 

**Authors**: Yijun Liu, Yuwei Liu, Yuan Meng, Jieheng Zhang, Yuwei Zhou, Ye Li, Jiacheng Jiang, Kangye Ji, Shijia Ge, Zhi Wang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15874)  

**Abstract**: Vision-centric hierarchical embodied models have demonstrated strong potential for long-horizon robotic control. However, existing methods lack spatial awareness capabilities, limiting their effectiveness in bridging visual plans to actionable control in complex environments. To address this problem, we propose Spatial Policy (SP), a unified spatial-aware visuomotor robotic manipulation framework via explicit spatial modeling and reasoning. Specifically, we first design a spatial-conditioned embodied video generation module to model spatially guided predictions through a spatial plan table. Then, we propose a spatial-based action prediction module to infer executable actions with coordination. Finally, we propose a spatial reasoning feedback policy to refine the spatial plan table via dual-stage replanning. Extensive experiments show that SP significantly outperforms state-of-the-art baselines, achieving a 33.0% average improvement over the best baseline. With an 86.7% average success rate across 11 diverse tasks, SP substantially enhances the practicality of embodied models for robotic control applications. Code and checkpoints are maintained at this https URL. 

**Abstract (ZH)**: 基于视觉的分层类体学模型在长时 horizon 软件控制中展现了强大的潜力。然而，现有方法缺乏空间感知能力，限制了它们在复杂环境中将视觉计划转化为可执行控制的效果。为应对这一问题，我们提出了一种统一的空间感知视觉-运动机器人操作框架——Spatial Policy (SP)，通过明确的空间建模和推理实现。具体而言，我们首先设计了一种基于空间条件的体学视频生成模块，通过空间计划表来建模空间导向的预测。然后，我们提出了一种基于空间的动作预测模块，以实现协调执行动作的推断。最后，我们提出了一种基于空间推理的反馈策略，通过双阶段重规划来精炼空间计划表。大量实验表明，SP 显著优于最先进的基线方法，在平均改进幅度上达到了33.0%，并在11项不同的任务中实现了86.7%的平均成功率，大幅提升了类体学模型在机器人控制应用中的实用价值。代码和检查点请访问此链接。 

---
# Do What? Teaching Vision-Language-Action Models to Reject the Impossible 

**Title (ZH)**: 教什么？训练视觉-语言-动作模型拒绝不可能的任务 

**Authors**: Wen-Han Hsieh, Elvis Hsieh, Dantong Niu, Trevor Darrell, Roei Herzig, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2508.16292)  

**Abstract**: Recently, Vision-Language-Action (VLA) models have demonstrated strong performance on a range of robotic tasks. These models rely on multimodal inputs, with language instructions playing a crucial role -- not only in predicting actions, but also in robustly interpreting user intent, even when the requests are impossible to fulfill. In this work, we investigate how VLAs can recognize, interpret, and respond to false-premise instructions: natural language commands that reference objects or conditions absent from the environment. We propose Instruct-Verify-and-Act (IVA), a unified framework that (i) detects when an instruction cannot be executed due to a false premise, (ii) engages in language-based clarification or correction, and (iii) grounds plausible alternatives in perception and action. Towards this end, we construct a large-scale instruction tuning setup with structured language prompts and train a VLA model capable of handling both accurate and erroneous requests. Our approach leverages a contextually augmented, semi-synthetic dataset containing paired positive and false-premise instructions, enabling robust detection and natural language correction. Our experiments show that IVA improves false premise detection accuracy by 97.56% over baselines, while increasing successful responses in false-premise scenarios by 50.78%. 

**Abstract (ZH)**: 近年来，视觉-语言-行动（VLA）模型在一系列机器人任务中展现出了强大的性能。这些模型依赖于多模态输入，其中语言指令起着关键作用——不仅能够预测行动，还能够稳健地解释用户意图，即使这些请求无法实现也是如此。在本研究中，我们探讨了VLA如何识别、解释和应对虚假前提指令：那些提及环境不存在的对象或条件的自然语言命令。我们提出了一种统一的框架Instruct-Verify-and-Act（IVA），该框架包括：（i）检测由于虚假前提指令无法执行的情况；（ii）通过语言澄清或修正；（iii）基于感知和行动确立合理的替代方案。为此，我们构建了一个包含结构化语言提示的大规模指令调优设置，并训练了一个能够处理准确和错误请求的VLA模型。我们的方法利用了一个上下文增强的半合成数据集，该数据集包含配对的真实指令和虚假前提指令，从而实现了稳健的检测和自然语言修正。实验结果显示，与基线相比，IVA在虚假前提检测准确性上提高了97.56%，同时在虚假前提情境下的成功响应率提高了50.78%。 

---
# Validating Terrain Models in Digital Twins for Trustworthy sUAS Operations 

**Title (ZH)**: 在数字孪生中验证地形模型以实现可靠的无人驾驶航空系统运营 

**Authors**: Arturo Miguel Russell Bernal, Maureen Petterson, Pedro Antonio Alarcon Granadeno, Michael Murphy, James Mason, Jane Cleland-Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16104)  

**Abstract**: With the increasing deployment of small Unmanned Aircraft Systems (sUAS) in unfamiliar and complex environments, Environmental Digital Twins (EDT) that comprise weather, airspace, and terrain data are critical for safe flight planning and for maintaining appropriate altitudes during search and surveillance operations. With the expansion of sUAS capabilities through edge and cloud computing, accurate EDT are also vital for advanced sUAS capabilities, like geolocation. However, real-world sUAS deployment introduces significant sources of uncertainty, necessitating a robust validation process for EDT components. This paper focuses on the validation of terrain models, one of the key components of an EDT, for real-world sUAS tasks. These models are constructed by fusing U.S. Geological Survey (USGS) datasets and satellite imagery, incorporating high-resolution environmental data to support mission tasks. Validating both the terrain models and their operational use by sUAS under real-world conditions presents significant challenges, including limited data granularity, terrain discontinuities, GPS and sensor inaccuracies, visual detection uncertainties, as well as onboard resources and timing constraints. We propose a 3-Dimensions validation process grounded in software engineering principles, following a workflow across granularity of tests, simulation to real world, and the analysis of simple to edge conditions. We demonstrate our approach using a multi-sUAS platform equipped with a Terrain-Aware Digital Shadow. 

**Abstract (ZH)**: 随着小型无人驾驶航空系统（sUAS）在陌生和复杂环境中的部署增加，集成了气象、空域和地形数据的环境数字 twinning（EDT）对于安全飞行规划以及搜索和监视操作中维持适当高度至关重要。通过边缘和云计算扩展sUAS能力后，准确的EDT也是实现地理定位等高级sUAS功能的关键。然而，实际部署sUAS引入了显著的不确定性来源，需要针对EDT组件制定稳健的验证流程。本文重点关注在实际sUAS任务中验证地形模型，这是EDT的关键组成部分之一。这些模型通过融合美国地质调查局（USGS）数据集和卫星图像构建，加入高分辨率环境数据以支持任务需求。在实际条件下验证地形模型及其在sUAS操作中的应用面临诸多挑战，包括数据粒度有限、地形不连续性、GPS和传感器不准确、视觉检测不确定性，以及机载资源和时间限制。我们提出了一种基于软件工程原则的三维验证过程，该过程遵循从粒度测试、模拟到现实世界的流程，并分析从简单到边缘条件。我们通过装备有地形感知数字阴影的多sUAS平台展示了该方法。 

---
# Hierarchical Decision-Making for Autonomous Navigation: Integrating Deep Reinforcement Learning and Fuzzy Logic in Four-Wheel Independent Steering and Driving Systems 

**Title (ZH)**: 基于层次决策的自主导航：四轮独立转向与驱动系统中深度强化学习与模糊逻辑集成的研究 

**Authors**: Yizhi Wang, Degang Xu, Yongfang Xie, Shuzhong Tan, Xianan Zhou, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.16574)  

**Abstract**: This paper presents a hierarchical decision-making framework for autonomous navigation in four-wheel independent steering and driving (4WISD) systems. The proposed approach integrates deep reinforcement learning (DRL) for high-level navigation with fuzzy logic for low-level control to ensure both task performance and physical feasibility. The DRL agent generates global motion commands, while the fuzzy logic controller enforces kinematic constraints to prevent mechanical strain and wheel slippage. Simulation experiments demonstrate that the proposed framework outperforms traditional navigation methods, offering enhanced training efficiency and stability and mitigating erratic behaviors compared to purely DRL-based solutions. Real-world validations further confirm the framework's ability to navigate safely and effectively in dynamic industrial settings. Overall, this work provides a scalable and reliable solution for deploying 4WISD mobile robots in complex, real-world scenarios. 

**Abstract (ZH)**: 本文提出了一种层次化的决策框架，用于四轮独立转向与驱动（4WISD）系统的自主导航。该提出的approach将深度强化学习（DRL）用于高层次导航，将模糊逻辑用于低层次控制，以确保任务性能和物理可行性。DRL代理生成全局运动命令，而模糊逻辑控制器则施加动力学约束，防止机械应力和车轮打滑。仿真实验表明，提出的框架优于传统的导航方法，提供了增强的训练效率和稳定性，并且与基于纯DRL的解决方案相比，减少了行为的不确定性。进一步的实地验证确认了该框架能够在动态工业环境中安全有效地导航的能力。总体而言，本文为部署4WISD移动机器人在复杂的真实世界场景中提供了一种可扩展且可靠的方法。 

---
# OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval 

**Title (ZH)**: OPERA：一种增强学习驱动的规划执行架构，用于基于推理的多跳检索 

**Authors**: Yu Liu, Yanbing Liu, Fangfang Yuan, Cong Cao, Youbang Sun, Kun Peng, WeiZhuo Chen, Jianjun Li, Zhiyuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.16438)  

**Abstract**: Recent advances in large language models (LLMs) and dense retrievers have driven significant progress in retrieval-augmented generation (RAG). However, existing approaches face significant challenges in complex reasoning-oriented multi-hop retrieval tasks: 1) Ineffective reasoning-oriented planning: Prior methods struggle to generate robust multi-step plans for complex queries, as rule-based decomposers perform poorly on out-of-template questions. 2) Suboptimal reasoning-driven retrieval: Related methods employ limited query reformulation, leading to iterative retrieval loops that often fail to locate golden documents. 3) Insufficient reasoning-guided filtering: Prevailing methods lack the fine-grained reasoning to effectively filter salient information from noisy results, hindering utilization of retrieved knowledge. Fundamentally, these limitations all stem from the weak coupling between retrieval and reasoning in current RAG architectures. We introduce the Orchestrated Planner-Executor Reasoning Architecture (OPERA), a novel reasoning-driven retrieval framework. OPERA's Goal Planning Module (GPM) decomposes questions into sub-goals, which are executed by a Reason-Execute Module (REM) with specialized components for precise reasoning and effective retrieval. To train OPERA, we propose Multi-Agents Progressive Group Relative Policy Optimization (MAPGRPO), a novel variant of GRPO. Experiments on complex multi-hop benchmarks show OPERA's superior performance, validating both the MAPGRPO method and OPERA's design. Code is available at this https URL. 

**Abstract (ZH)**: 最近大语言模型(LLMs)和密集检索技术的进展推动了检索增强生成(RAG)的重大进展。然而，现有方法在复杂的多跳推理检索任务中面临显著挑战：1）无效的推理导向规划：先前的方法在生成复杂查询的健壮多步计划方面表现不佳，因为基于规则的分解器在超出模板的问题上表现较差。2）次优的推理驱动检索：相关方法采用有限的查询重写，导致迭代检索循环，往往无法找到黄金文档。3）不足的推理指导过滤：现有方法缺乏细粒度的推理能力，无法有效过滤噪声结果中的重要信息，阻碍了检索知识的利用。从根本上说，这些限制都源于当前RAG架构中检索与推理之间的弱耦合。我们提出了协调规划执行推理架构(Orchestrated Planner-Executor Reasoning Architecture, OPERA)，这是一种新颖的推理驱动检索框架。OPERA的目标规划模块(GPM)将问题分解为子目标，这些子目标由专门组件支持的Reason-Execute模块(REM)执行。为了训练OPERA，我们提出了多代理渐进组相对策略优化(Multi-Agents Progressive Group Relative Policy Optimization, MAPGRPO)，这是一种GRPO的新变体。复杂多跳基准实验表明，OPERA在性能上优越，验证了MAPGRPO方法和OPERA的设计。代码在以下链接处提供。 

---
