# Whole-Body Proprioceptive Morphing: A Modular Soft Gripper for Robust Cross-Scale Grasping 

**Title (ZH)**: 全身本体感受重构：一种模块化软夹持器，用于稳健的跨尺度抓取 

**Authors**: Dong Heon Han, Xiaohao Xu, Yuxi Chen, Yusheng Zhou, Xinqi Zhang, Jiaqi Wang, Daniel Bruder, Xiaonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27666)  

**Abstract**: Biological systems, such as the octopus, exhibit masterful cross-scale manipulation by adaptively reconfiguring their entire form, a capability that remains elusive in robotics. Conventional soft grippers, while compliant, are mostly constrained by a fixed global morphology, and prior shape-morphing efforts have been largely confined to localized deformations, failing to replicate this biological dexterity. Inspired by this natural exemplar, we introduce the paradigm of collaborative, whole-body proprioceptive morphing, realized in a modular soft gripper architecture. Our design is a distributed network of modular self-sensing pneumatic actuators that enables the gripper to intelligently reconfigure its entire topology, achieving multiple morphing states that are controllable to form diverse polygonal shapes. By integrating rich proprioceptive feedback from embedded sensors, our system can seamlessly transition from a precise pinch to a large envelope grasp. We experimentally demonstrate that this approach expands the grasping envelope and enhances generalization across diverse object geometries (standard and irregular) and scales (up to 10$\times$), while also unlocking novel manipulation modalities such as multi-object and internal hook grasping. This work presents a low-cost, easy-to-fabricate, and scalable framework that fuses distributed actuation with integrated sensing, offering a new pathway toward achieving biological levels of dexterity in robotic manipulation. 

**Abstract (ZH)**: 生物系统，如八足动物，通过适应性重构其整个形态展现卓越的跨尺度操控能力，这一能力在机器人领域仍然难以实现。常规的柔性夹爪虽然具有顺应性，但大多受限于固定的全局形态，此前的形态变化努力主要局限于局部变形，未能复制生物的灵活性。受这一自然范例的启发，我们提出了协作的全身本体感受性形态变化范式，这一范式在模块化柔性夹爪架构中得以实现。我们的设计是一种分布式网络的模块化自感知气动执行器，使夹爪能够智能地重新配置其整个拓扑结构，实现多种可控制的形态变换状态，形成多样化的多边形形状。通过整合嵌入式传感器提供的丰富本体感受反馈，我们的系统能够无缝切换从精确的夹持到大范围抓取。实验结果显示，这种方法扩大了抓取范围，并在多样化的物体几何形状（标准和非标准）和尺度（高达10倍）上实现了更好的泛化能力，同时解锁了诸如多物体抓取和内部钩爪抓取等新的操作模式。这项工作提供了一种低成本、易于制造且可扩展的框架，结合分布式驱动与集成传感技术，为实现类似生物水平的灵活性提供了一条新的途径。 

---
# Toward Accurate Long-Horizon Robotic Manipulation: Language-to-Action with Foundation Models via Scene Graphs 

**Title (ZH)**: 基于场景图的语言到动作转换：通过基础模型实现长期 horizon 机器人操作的准确执行 

**Authors**: Sushil Samuel Dinesh, Shinkyu Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.27558)  

**Abstract**: This paper presents a framework that leverages pre-trained foundation models for robotic manipulation without domain-specific training. The framework integrates off-the-shelf models, combining multimodal perception from foundation models with a general-purpose reasoning model capable of robust task sequencing. Scene graphs, dynamically maintained within the framework, provide spatial awareness and enable consistent reasoning about the environment. The framework is evaluated through a series of tabletop robotic manipulation experiments, and the results highlight its potential for building robotic manipulation systems directly on top of off-the-shelf foundation models. 

**Abstract (ZH)**: 本文提出了一种框架，该框架利用预训练基础模型实现机器人操作，无需领域特定训练。该框架结合了来自基础模型的多模态感知和一个通用推理模型，该模型能够实现稳健的任务序列。场景图在框架中动态维护，提供空间意识并使对环境的一致性推理成为可能。该框架通过一系列桌面机器人操作实验进行了评估，结果突显了其直接基于现成基础模型构建机器人操作系统的能力。 

---
# EBT-Policy: Energy Unlocks Emergent Physical Reasoning Capabilities 

**Title (ZH)**: EBT-Policy：能量解锁 emergent 物理推理能力 

**Authors**: Travis Davies, Yiqi Huang, Alexi Gladstone, Yunxin Liu, Xiang Chen, Heng Ji, Huxian Liu, Luhui Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27545)  

**Abstract**: Implicit policies parameterized by generative models, such as Diffusion Policy, have become the standard for policy learning and Vision-Language-Action (VLA) models in robotics. However, these approaches often suffer from high computational cost, exposure bias, and unstable inference dynamics, which lead to divergence under distribution shifts. Energy-Based Models (EBMs) address these issues by learning energy landscapes end-to-end and modeling equilibrium dynamics, offering improved robustness and reduced exposure bias. Yet, policies parameterized by EBMs have historically struggled to scale effectively. Recent work on Energy-Based Transformers (EBTs) demonstrates the scalability of EBMs to high-dimensional spaces, but their potential for solving core challenges in physically embodied models remains underexplored. We introduce a new energy-based architecture, EBT-Policy, that solves core issues in robotic and real-world settings. Across simulated and real-world tasks, EBT-Policy consistently outperforms diffusion-based policies, while requiring less training and inference computation. Remarkably, on some tasks it converges within just two inference steps, a 50x reduction compared to Diffusion Policy's 100. Moreover, EBT-Policy exhibits emergent capabilities not seen in prior models, such as zero-shot recovery from failed action sequences using only behavior cloning and without explicit retry training. By leveraging its scalar energy for uncertainty-aware inference and dynamic compute allocation, EBT-Policy offers a promising path toward robust, generalizable robot behavior under distribution shifts. 

**Abstract (ZH)**: 基于能量模型的Implicit策略：解决机器人和现实世界核心问题的新架构 

---
# Preliminary Prototyping of Avoidance Behaviors Triggered by a User's Physical Approach to a Robot 

**Title (ZH)**: 基于用户物理接近机器人触发的回避行为初步原型设计 

**Authors**: Tomoko Yonezawa, Hirotake Yamazoe, Atsuo Fujino, Daigo Suhara, Takaya Tamamoto, Yuto Nishiguchi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27436)  

**Abstract**: Human-robot interaction frequently involves physical proximity or contact. In human-human settings, people flexibly accept, reject, or tolerate such approaches depending on the relationship and context. We explore the design of a robot's rejective internal state and corresponding avoidance behaviors, such as withdrawing or pushing away, when a person approaches. We model the accumulation and decay of discomfort as a function of interpersonal distance, and implement tolerance (endurance) and limit-exceeding avoidance driven by the Dominance axis of the PAD affect model. The behaviors and their intensities are realized on an arm robot. Results illustrate a coherent pipeline from internal state parameters to graded endurance motions and, once a limit is crossed, to avoidance actions. 

**Abstract (ZH)**: 人类与机器人交互常涉及身体 proximity 或接触。在人类交互环境中，人们根据关系和情境灵活接受、拒绝或容忍这种接近。我们探索当一个人接近时，机器人拒斥内部状态及其相应的回避行为，如撤回或推开的设计。我们建模人际距离与不适累积及衰减的关系，并基于 PAD 影响模型的支配轴实现容忍（耐受）和极限超越回避。这些行为及其强度在机械臂机器人上得到实现。结果展示了从内部状态参数到分等级的耐受动作，以及一旦达到极限即转变为回避动作的连贯管线。 

---
# Learning Soft Robotic Dynamics with Active Exploration 

**Title (ZH)**: 基于主动探索学习软体机器人力学模型 

**Authors**: Hehui Zheng, Bhavya Sukhija, Chenhao Li, Klemens Iten, Andreas Krause, Robert K. Katzschmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.27428)  

**Abstract**: Soft robots offer unmatched adaptability and safety in unstructured environments, yet their compliant, high-dimensional, and nonlinear dynamics make modeling for control notoriously difficult. Existing data-driven approaches often fail to generalize, constrained by narrowly focused task demonstrations or inefficient random exploration. We introduce SoftAE, an uncertainty-aware active exploration framework that autonomously learns task-agnostic and generalizable dynamics models of soft robotic systems. SoftAE employs probabilistic ensemble models to estimate epistemic uncertainty and actively guides exploration toward underrepresented regions of the state-action space, achieving efficient coverage of diverse behaviors without task-specific supervision. We evaluate SoftAE on three simulated soft robotic platforms -- a continuum arm, an articulated fish in fluid, and a musculoskeletal leg with hybrid actuation -- and on a pneumatically actuated continuum soft arm in the real world. Compared with random exploration and task-specific model-based reinforcement learning, SoftAE produces more accurate dynamics models, enables superior zero-shot control on unseen tasks, and maintains robustness under sensing noise, actuation delays, and nonlinear material effects. These results demonstrate that uncertainty-driven active exploration can yield scalable, reusable dynamics models across diverse soft robotic morphologies, representing a step toward more autonomous, adaptable, and data-efficient control in compliant robots. 

**Abstract (ZH)**: 软体机器人在非结构化环境中的不可替代的适应性和安全性使其动力学模型难以建模，但现有的数据驱动方法往往由于窄范围的任务演示或低效的随机探索而无法泛化。我们提出了一种不确定性感知的主动探索框架SoftAE，该框架能够自主学习软体机器人系统的任务无关且可泛化的动力学模型。SoftAE 使用概率集成模型来估计认知不确定性，并主动引导探索未充分代表的状态-动作空间区域，从而实现对多种行为的有效覆盖，无需特定任务监督。我们在三种模拟软体机器人平台——连续臂、流体中的铰接鱼和具有混合驱动的肌腱骨骼腿——以及一个气动驱动的连续软臂的现实世界环境中评估了SoftAE。相较于随机探索和特定任务的数据驱动强化学习，SoftAE 生成了更为准确的动力学模型，能够实现更好的零样本控制，并在感知噪声、驱动延迟和非线性材料效应下保持鲁棒性。这些结果表明，以不确定性驱动的主动探索可产生适用于多种软体机器人形态的可扩展且可重用的动力学模型，代表着朝向更自主、更适应、更数据高效的软体柔顺机器人控制迈出的一步。 

---
# Towards a Multi-Embodied Grasping Agent 

**Title (ZH)**: 朝向多体态抓取代理的研究 

**Authors**: Roman Freiberg, Alexander Qualmann, Ngo Anh Vien, Gerhard Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2510.27420)  

**Abstract**: Multi-embodiment grasping focuses on developing approaches that exhibit generalist behavior across diverse gripper designs. Existing methods often learn the kinematic structure of the robot implicitly and face challenges due to the difficulty of sourcing the required large-scale data. In this work, we present a data-efficient, flow-based, equivariant grasp synthesis architecture that can handle different gripper types with variable degrees of freedom and successfully exploit the underlying kinematic model, deducing all necessary information solely from the gripper and scene geometry. Unlike previous equivariant grasping methods, we translated all modules from the ground up to JAX and provide a model with batching capabilities over scenes, grippers, and grasps, resulting in smoother learning, improved performance and faster inference time. Our dataset encompasses grippers ranging from humanoid hands to parallel yaw grippers and includes 25,000 scenes and 20 million grasps. 

**Abstract (ZH)**: 多体感抓取专注于开发在多样化的抓取器设计中表现出通用行为的方法。现有方法往往隐式学习机器人的运动结构，并且由于获取足够规模数据的难度而面临挑战。在本工作中，我们提出了一种数据高效、基于流、等变抓取合成架构，能够处理不同自由度的抓取器类型，并成功利用潜在的运动学模型，仅从抓取器和场景几何中推导出所有必要的信息。与之前的等变抓取方法不同，我们将所有模块从基础重新翻译至JAX，并提供场景、抓取器和抓取批量处理能力，从而实现更平滑的学习、更好的性能和更快的推理时间。我们的数据集涵盖了从类人手到并行摆动抓取器的各种抓取器类型，包括25,000个场景和2000万个抓取。 

---
# Vectorized Online POMDP Planning 

**Title (ZH)**: 向量化的在线POMDP规划 

**Authors**: Marcus Hoerger, Muhammad Sudrajat, Hanna Kurniawati  

**Link**: [PDF](https://arxiv.org/pdf/2510.27191)  

**Abstract**: Planning under partial observability is an essential capability of autonomous robots. The Partially Observable Markov Decision Process (POMDP) provides a powerful framework for planning under partial observability problems, capturing the stochastic effects of actions and the limited information available through noisy observations. POMDP solving could benefit tremendously from massive parallelization of today's hardware, but parallelizing POMDP solvers has been challenging. They rely on interleaving numerical optimization over actions with the estimation of their values, which creates dependencies and synchronization bottlenecks between parallel processes that can quickly offset the benefits of parallelization. In this paper, we propose Vectorized Online POMDP Planner (VOPP), a novel parallel online solver that leverages a recent POMDP formulation that analytically solves part of the optimization component, leaving only the estimation of expectations for numerical computation. VOPP represents all data structures related to planning as a collection of tensors and implements all planning steps as fully vectorized computations over this representation. The result is a massively parallel solver with no dependencies and synchronization bottlenecks between parallel computations. Experimental results indicate that VOPP is at least 20X more efficient in computing near-optimal solutions compared to an existing state-of-the-art parallel online solver. 

**Abstract (ZH)**: 基于部分可观测性的规划是自主机器人的一项基本能力。部分可观测马尔可夫决策过程（POMDP）提供了一个强大的框架，用于解决部分可观测性问题，能够捕获动作的随机效应并通过嘈杂观测获取的有限信息。POMDP求解可以从当今硬件的巨大并行化中受益匪浅，但并行化POMDP求解器一直具有挑战性。现有的方法依赖于动作上的数值优化与价值估计的交替进行，这在并行过程中创建了依赖性和同步瓶颈，可能会迅速抵消并行化的益处。本文提出了向量在线POMDP规划器（VOPP），这是一种新颖的并行在线求解器，利用了最近提出的一种POMDP形式化方法，该方法部分地解析了优化组件，仅将期望的估计留给数值计算。VOPP将所有与规划相关的数据结构表示为张量集合，并将所有规划步骤实现为对这一表示的完全向量化计算。结果是一种无依赖性和同步瓶颈的并行求解器。实验结果表明，VOPP在计算近最优解方面至少比现有最先进的并行在线求解器快20倍。 

---
# Learning Generalizable Visuomotor Policy through Dynamics-Alignment 

**Title (ZH)**: 通过动力学对齐学习可泛化的视听运动策略 

**Authors**: Dohyeok Lee, Jung Min Lee, Munkyung Kim, Seokhun Ju, Jin Woo Koo, Kyungjae Lee, Dohyeong Kim, TaeHyun Cho, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27114)  

**Abstract**: Behavior cloning methods for robot learning suffer from poor generalization due to limited data support beyond expert demonstrations. Recent approaches leveraging video prediction models have shown promising results by learning rich spatiotemporal representations from large-scale datasets. However, these models learn action-agnostic dynamics that cannot distinguish between different control inputs, limiting their utility for precise manipulation tasks and requiring large pretraining datasets. We propose a Dynamics-Aligned Flow Matching Policy (DAP) that integrates dynamics prediction into policy learning. Our method introduces a novel architecture where policy and dynamics models provide mutual corrective feedback during action generation, enabling self-correction and improved generalization. Empirical validation demonstrates generalization performance superior to baseline methods on real-world robotic manipulation tasks, showing particular robustness in OOD scenarios including visual distractions and lighting variations. 

**Abstract (ZH)**: 基于动力模型对齐的流动匹配策略（Dynamics-Aligned Flow Matching Policy）用于提升机器人学习的任务通用性 

---
# A Multi-Modal Neuro-Symbolic Approach for Spatial Reasoning-Based Visual Grounding in Robotics 

**Title (ZH)**: 基于空间推理的多模态神经符号方法在机器人视觉定位中的应用 

**Authors**: Simindokht Jahangard, Mehrzad Mohammadi, Abhinav Dhall, Hamid Rezatofighi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27033)  

**Abstract**: Visual reasoning, particularly spatial reasoning, is a challenging cognitive task that requires understanding object relationships and their interactions within complex environments, especially in robotics domain. Existing vision_language models (VLMs) excel at perception tasks but struggle with fine-grained spatial reasoning due to their implicit, correlation-driven reasoning and reliance solely on images. We propose a novel neuro_symbolic framework that integrates both panoramic-image and 3D point cloud information, combining neural perception with symbolic reasoning to explicitly model spatial and logical relationships. Our framework consists of a perception module for detecting entities and extracting attributes, and a reasoning module that constructs a structured scene graph to support precise, interpretable queries. Evaluated on the JRDB-Reasoning dataset, our approach demonstrates superior performance and reliability in crowded, human_built environments while maintaining a lightweight design suitable for robotics and embodied AI applications. 

**Abstract (ZH)**: 视觉推理，特别是空间推理，是一项具有挑战性的认知任务，要求理解对象关系及其在复杂环境中的相互作用，尤其是在机器人领域。现有的视觉语言模型（VLMs）在感知任务上表现出色，但在细粒度的空间推理方面存在困难，这主要是由于它们依赖于图像的隐式、关联驱动的推理。我们提出了一种新颖的神经-符号框架，该框架结合全景图像和3D点云信息，将神经感知与符号推理结合，明确建模空间和逻辑关系。该框架包括一个感知模块用于检测实体并提取属性，以及一个推理模块构建结构化的场景图以支持精确、可解释的查询。在JRDB-Reasoning数据集上的评估表明，我们的方法在拥挤的人造环境中展现了优越的性能和可靠性，并保持了适合机器人和具身AI应用的轻量级设计。 

---
# Heterogeneous Robot Collaboration in Unstructured Environments with Grounded Generative Intelligence 

**Title (ZH)**: 具有地基生成智能的异构机器人在非结构化环境下的协同作业 

**Authors**: Zachary Ravichandran, Fernando Cladera, Ankit Prabhu, Jason Hughes, Varun Murali, Camillo Taylor, George J. Pappas, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.26915)  

**Abstract**: Heterogeneous robot teams operating in realistic settings often must accomplish complex missions requiring collaboration and adaptation to information acquired online. Because robot teams frequently operate in unstructured environments -- uncertain, open-world settings without prior maps -- subtasks must be grounded in robot capabilities and the physical world. While heterogeneous teams have typically been designed for fixed specifications, generative intelligence opens the possibility of teams that can accomplish a wide range of missions described in natural language. However, current large language model (LLM)-enabled teaming methods typically assume well-structured and known environments, limiting deployment in unstructured environments. We present SPINE-HT, a framework that addresses these limitations by grounding the reasoning abilities of LLMs in the context of a heterogeneous robot team through a three-stage process. Given language specifications describing mission goals and team capabilities, an LLM generates grounded subtasks which are validated for feasibility. Subtasks are then assigned to robots based on capabilities such as traversability or perception and refined given feedback collected during online operation. In simulation experiments with closed-loop perception and control, our framework achieves nearly twice the success rate compared to prior LLM-enabled heterogeneous teaming approaches. In real-world experiments with a Clearpath Jackal, a Clearpath Husky, a Boston Dynamics Spot, and a high-altitude UAV, our method achieves an 87\% success rate in missions requiring reasoning about robot capabilities and refining subtasks with online feedback. More information is provided at this https URL. 

**Abstract (ZH)**: 异构机器人团队在现实场景中 often 必须执行需要协作和适应在线获取信息的复杂任务。由于机器人团队通常在未结构化的环境中运作——即不确定的、开放的世界环境，没有先验的地图——子任务必须基于机器人的能力和物理世界。虽然异构团队通常被设计为固定规格，但生成性智能开启了能够通过自然语言描述执行广泛任务的团队的可能性。然而，当前的大规模语言模型（LLM）启用的团队方法通常假设结构化的和已知的环境，限制了其在未结构化环境中的部署。我们提出了 SPINE-HT 框架，该框架通过一个三阶段过程将 LLM 的推理能力锚定在一个异构机器人团队的上下文中，从而解决这些限制。给定描述任务目标和团队能力的语言规范，LLM 生成可验证可行性的地面化子任务。然后根据诸如可通行性或感知之类的机器人能力将子任务分配给机器人，并在收到在线操作期间收集的反馈后进一步优化。在具有闭环感知和控制的仿真实验中，我们框架的成功率几乎是之前 LLM 启用的异构团队方法的两倍。在涉及 Clearpath Jackal、Clearpath Husky、Boston Dynamics Spot 和高空无人机的实际实验中，我们的方法在需要推理机器人能力和在线优化子任务的任务中取得了 87% 的成功率。更多详细信息请访问 <https://www.example.com>。 

---
# NaviTrace: Evaluating Embodied Navigation of Vision-Language Models 

**Title (ZH)**: NaviTrace：评估视觉语言模型的 embodied 导航能力 

**Authors**: Tim Windecker, Manthan Patel, Moritz Reuss, Richard Schwarzkopf, Cesar Cadena, Rudolf Lioutikov, Marco Hutter, Jonas Frey  

**Link**: [PDF](https://arxiv.org/pdf/2510.26909)  

**Abstract**: Vision-language models demonstrate unprecedented performance and generalization across a wide range of tasks and scenarios. Integrating these foundation models into robotic navigation systems opens pathways toward building general-purpose robots. Yet, evaluating these models' navigation capabilities remains constrained by costly real-world trials, overly simplified simulations, and limited benchmarks. We introduce NaviTrace, a high-quality Visual Question Answering benchmark where a model receives an instruction and embodiment type (human, legged robot, wheeled robot, bicycle) and must output a 2D navigation trace in image space. Across 1000 scenarios and more than 3000 expert traces, we systematically evaluate eight state-of-the-art VLMs using a newly introduced semantic-aware trace score. This metric combines Dynamic Time Warping distance, goal endpoint error, and embodiment-conditioned penalties derived from per-pixel semantics and correlates with human preferences. Our evaluation reveals consistent gap to human performance caused by poor spatial grounding and goal localization. NaviTrace establishes a scalable and reproducible benchmark for real-world robotic navigation. The benchmark and leaderboard can be found at this https URL. 

**Abstract (ZH)**: 视觉-语言模型在广泛的任务和场景中展示了前所未有的性能和泛化能力。将这些基础模型集成到机器人导航系统中为构建通用机器人开辟了途径。然而，评估这些模型的导航能力仍然受到昂贵的实地试验、过于简化的模拟以及有限基准的限制。我们引入了NaviTrace，这是一个高质量的视觉问答基准，在该基准中，模型接受一个指令和表现形式（人类、腿足式机器人、轮式机器人、自行车），并必须输出一个二维导航轨迹。在1000个场景和超过3000个专家轨迹上，我们系统地使用新引入的语义意识轨迹评分对该基准中的八种最先进的视觉-语言模型进行了评估。该指标结合了动态时间规整距离、目标端点误差以及来自像素级语义的条件惩罚，并与人类偏好相关。我们的评估揭示了由于空间语义理解和目标定位不佳而产生的与人类性能的一致差距。NaviTrace建立了可扩展且可重现的现实世界机器人导航基准。基准和排行榜可在以下链接找到：this https URL。 

---
# Leveraging Foundation Models for Enhancing Robot Perception and Action 

**Title (ZH)**: 利用基础模型增强机器人感知与行动能力 

**Authors**: Reihaneh Mirjalili  

**Link**: [PDF](https://arxiv.org/pdf/2510.26855)  

**Abstract**: This thesis investigates how foundation models can be systematically leveraged to enhance robotic capabilities, enabling more effective localization, interaction, and manipulation in unstructured environments. The work is structured around four core lines of inquiry, each addressing a fundamental challenge in robotics while collectively contributing to a cohesive framework for semantics-aware robotic intelligence. 

**Abstract (ZH)**: 本论文探究基础模型如何系统性地被利用以增强机器人能力，从而在非结构化环境中实现更加有效的定位、互动和操作。工作围绕四个核心研究方向展开，每个方向都解决机器人领域的一项基本挑战，共同构建起一种语义意识机器人智能的综合框架。 

---
# Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model 

**Title (ZH)**: 双流扩散的世界模型增强视觉-语言-动作模型 

**Authors**: John Won, Kyungmin Lee, Huiwon Jang, Dongyoung Kim, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.27607)  

**Abstract**: Recently, augmenting Vision-Language-Action models (VLAs) with world modeling has shown promise in improving robotic policy learning. However, it remains challenging to jointly predict next-state observations and action sequences because of the inherent difference between the two modalities. To address this, we propose DUal-STream diffusion (DUST), a world-model augmented VLA framework that handles the modality conflict and enhances the performance of VLAs across diverse tasks. Specifically, we propose a multimodal diffusion transformer architecture that explicitly maintains separate modality streams while still enabling cross-modal knowledge sharing. In addition, we introduce independent noise perturbations for each modality and a decoupled flow-matching loss. This design enables the model to learn the joint distribution in a bidirectional manner while avoiding the need for a unified latent space. Based on the decoupling of modalities during training, we also introduce a joint sampling method that supports test-time scaling, where action and vision tokens evolve asynchronously at different rates. Through experiments on simulated benchmarks such as RoboCasa and GR-1, DUST achieves up to 6% gains over baseline methods, while our test-time scaling approach provides an additional 2-5% boost. On real-world tasks with the Franka Research 3, DUST improves success rates by 13%, confirming its effectiveness beyond simulation. Furthermore, pre-training on action-free videos from BridgeV2 yields significant transfer gains on RoboCasa, underscoring DUST's potential for large-scale VLA pretraining. 

**Abstract (ZH)**: 最近，通过世界建模增强视觉-语言-行动模型（VLAs）在提升机器人策略学习方面展示了潜力。然而，由于两种模态之间的固有差异，同时预测下一个状态观测和动作序列仍然是一个挑战。为了解决这个问题，我们提出了DUal-STream扩散（DUST）框架，这是一种增强型VLAs框架，能够处理模态冲突并提升VLAs在多种任务中的性能。具体来说，我们提出了一种多模态扩散变换器架构，明确维护独立的模态流，同时仍允许跨模态知识共享。此外，我们引入了独立的噪声扰动以及解耦的流匹配损失。这种设计使模型能够在双向方式下学习联合分布，避免了统一隐空间的需要。基于训练过程中模态的解耦，我们还提出了一种联合采样方法，支持测试时的扩展，在此方法中，动作和视觉标记以不同的速率异步演化。通过在RoboCasa和GR-1等仿真基准上的实验，DUST实现了基线方法的6%改进，而我们的测试时扩展方法提供了额外的2-5%的提升。在使用Franka Research 3进行的真实世界任务中，DUST将成功率提高了13%，证明了其实用性超越了仿真。此外，基于BridgeV2中的无动作视频进行预训练，在RoboCasa上实现了显著的迁移增益，突显了DUST在大规模VLAs预训练中的潜力。 

---
# Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning 

**Title (ZH)**: 视觉后门攻击：基于对比触发学习的MLLM结构化决策making中的 embodied 决策攻击 

**Authors**: Qiusi Zhan, Hyeonjeong Ha, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27623)  

**Abstract**: Multimodal large language models (MLLMs) have advanced embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into MLLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and MLLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in MLLM-based embodied agents, underscoring the need for robust defenses before real-world deployment. 

**Abstract (ZH)**: 多模态大语言模型中的视觉后门攻击：基于环境物体的BEAT框架 

---
# Realistic pedestrian-driver interaction modelling using multi-agent RL with human perceptual-motor constraints 

**Title (ZH)**: 基于人类感知-运动约束的多智能体RL的现实行人-驾驶人交互建模 

**Authors**: Yueyang Wang, Mehmet Dogar, Gustav Markkula  

**Link**: [PDF](https://arxiv.org/pdf/2510.27383)  

**Abstract**: Modelling pedestrian-driver interactions is critical for understanding human road user behaviour and developing safe autonomous vehicle systems. Existing approaches often rely on rule-based logic, game-theoretic models, or 'black-box' machine learning methods. However, these models typically lack flexibility or overlook the underlying mechanisms, such as sensory and motor constraints, which shape how pedestrians and drivers perceive and act in interactive scenarios. In this study, we propose a multi-agent reinforcement learning (RL) framework that integrates both visual and motor constraints of pedestrian and driver agents. Using a real-world dataset from an unsignalised pedestrian crossing, we evaluate four model variants, one without constraints, two with either motor or visual constraints, and one with both, across behavioural metrics of interaction realism. Results show that the combined model with both visual and motor constraints performs best. Motor constraints lead to smoother movements that resemble human speed adjustments during crossing interactions. The addition of visual constraints introduces perceptual uncertainty and field-of-view limitations, leading the agents to exhibit more cautious and variable behaviour, such as less abrupt deceleration. In this data-limited setting, our model outperforms a supervised behavioural cloning model, demonstrating that our approach can be effective without large training datasets. Finally, our framework accounts for individual differences by modelling parameters controlling the human constraints as population-level distributions, a perspective that has not been explored in previous work on pedestrian-vehicle interaction modelling. Overall, our work demonstrates that multi-agent RL with human constraints is a promising modelling approach for simulating realistic road user interactions. 

**Abstract (ZH)**: 基于人类约束的多智能体强化学习建模对于理解行人与驾驶员行为及开发安全的自动驾驶车辆系统至关重要。 

---
# Reinforcement Learning for Long-Horizon Unordered Tasks: From Boolean to Coupled Reward Machines 

**Title (ZH)**: 长时限序任务的强化学习：从布尔型到耦合奖励机器 

**Authors**: Kristina Levina, Nikolaos Pappas, Athanasios Karapantelakis, Aneta Vulgarakis Feljan, Jendrik Seipp  

**Link**: [PDF](https://arxiv.org/pdf/2510.27329)  

**Abstract**: Reward machines (RMs) inform reinforcement learning agents about the reward structure of the environment. This is particularly advantageous for complex non-Markovian tasks because agents with access to RMs can learn more efficiently from fewer samples. However, learning with RMs is ill-suited for long-horizon problems in which a set of subtasks can be executed in any order. In such cases, the amount of information to learn increases exponentially with the number of unordered subtasks. In this work, we address this limitation by introducing three generalisations of RMs: (1) Numeric RMs allow users to express complex tasks in a compact form. (2) In Agenda RMs, states are associated with an agenda that tracks the remaining subtasks to complete. (3) Coupled RMs have coupled states associated with each subtask in the agenda. Furthermore, we introduce a new compositional learning algorithm that leverages coupled RMs: Q-learning with coupled RMs (CoRM). Our experiments show that CoRM scales better than state-of-the-art RM algorithms for long-horizon problems with unordered subtasks. 

**Abstract (ZH)**: 奖励机器（RMs）为强化学习代理提供了环境奖励结构的信息。这对于复杂的非马尔可夫任务尤其有利，因为具有RMs访问权限的代理可以从更少的数据中更高效地学习。然而，使用RMs学习不适合处理子任务可以以任意顺序执行的长期问题。在这种情况下，需要学习的信息量随无序子任务数量的增加而指数级增长。在本文中，我们通过引入三种RMs的通用形式来解决这一限制：（1）数字RMs允许用户以紧凑的形式表示复杂任务。（2）议程RMs中，状态与一个跟踪剩余需完成子任务的议程相关联。（3）耦合RMs中，每个议程中的子任务关联有耦合状态。此外，我们还引入了一种新的组合学习算法，该算法利用耦合RMs：带有耦合RMs的Q学习（CoRM）。我们的实验表明，CoRM在处理具有无序子任务的长期问题时比最先进的RM算法更具可扩展性。 

---
# Cognition Envelopes for Bounded AI Reasoning in Autonomous UAS Operations 

**Title (ZH)**: 认知边界下的受限人工智能推理在自主UAS操作中 

**Authors**: Pedro Antonio Alarcón Granadeno, Arturo Miguel Bernal Russell, Sofia Nelson, Demetrius Hernandez, Maureen Petterson, Michael Murphy, Walter J. Scheirer, Jane Cleland-Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26905)  

**Abstract**: Cyber-physical systems increasingly rely on Foundational Models such as Large Language Models (LLMs) and Vision-Language Models (VLMs) to increase autonomy through enhanced perception, inference, and planning. However, these models also introduce new types of errors, such as hallucinations, overgeneralizations, and context misalignments, resulting in incorrect and flawed decisions. To address this, we introduce the concept of Cognition Envelopes, designed to establish reasoning boundaries that constrain AI-generated decisions while complementing the use of meta-cognition and traditional safety envelopes. As with safety envelopes, Cognition Envelopes require practical guidelines and systematic processes for their definition, validation, and assurance. 

**Abstract (ZH)**: 基于物理系统的认知包络：大型语言模型和视觉-语言模型引入的新错误及其应对方法 

---
# The Denario project: Deep knowledge AI agents for scientific discovery 

**Title (ZH)**: Denary项目：面向科学发现的深度知识AI代理 

**Authors**: Francisco Villaescusa-Navarro, Boris Bolliet, Pablo Villanueva-Domingo, Adrian E. Bayer, Aidan Acquah, Chetana Amancharla, Almog Barzilay-Siegal, Pablo Bermejo, Camille Bilodeau, Pablo Cárdenas Ramírez, Miles Cranmer, Urbano L. França, ChangHoon Hahn, Yan-Fei Jiang, Raul Jimenez, Jun-Young Lee, Antonio Lerario, Osman Mamun, Thomas Meier, Anupam A. Ojha, Pavlos Protopapas, Shimanto Roy, David N. Spergel, Pedro Tarancón-Álvarez, Ujjwal Tiwari, Matteo Viel, Digvijay Wadekar, Chi Wang, Bonny Y. Wang, Licong Xu, Yossi Yovel, Shuwen Yue, Wen-Han Zhou, Qiyao Zhu, Jiajun Zou, Íñigo Zubeldia  

**Link**: [PDF](https://arxiv.org/pdf/2510.26887)  

**Abstract**: We present Denario, an AI multi-agent system designed to serve as a scientific research assistant. Denario can perform many different tasks, such as generating ideas, checking the literature, developing research plans, writing and executing code, making plots, and drafting and reviewing a scientific paper. The system has a modular architecture, allowing it to handle specific tasks, such as generating an idea, or carrying out end-to-end scientific analysis using Cmbagent as a deep-research backend. In this work, we describe in detail Denario and its modules, and illustrate its capabilities by presenting multiple AI-generated papers generated by it in many different scientific disciplines such as astrophysics, biology, biophysics, biomedical informatics, chemistry, material science, mathematical physics, medicine, neuroscience and planetary science. Denario also excels at combining ideas from different disciplines, and we illustrate this by showing a paper that applies methods from quantum physics and machine learning to astrophysical data. We report the evaluations performed on these papers by domain experts, who provided both numerical scores and review-like feedback. We then highlight the strengths, weaknesses, and limitations of the current system. Finally, we discuss the ethical implications of AI-driven research and reflect on how such technology relates to the philosophy of science. We publicly release the code at this https URL. A Denario demo can also be run directly on the web at this https URL, and the full app will be deployed on the cloud. 

**Abstract (ZH)**: 我们介绍Denario，一个设计用于担任科学研究助理的人工智能多agents系统。Denario能够执行多种任务，如生成想法、查阅文献、制定研究计划、编写和执行代码、制作图表以及起草和审查科研论文。该系统具有模块化架构，允许它处理特定任务，如生成想法或使用Cmbagent作为深度研究后端进行端到端的科学研究分析。在本文中，我们详细描述了Denario及其模块，并通过展示它在多个科学学科（如天体物理、生物学、生物物理学、生物医学信息学、化学、材料科学、数学物理、医学、神经科学和行星科学）中生成的多篇AI论文，来阐述其功能。Denario还擅长将不同学科的想法结合起来，我们通过展示一篇应用量子物理和机器学习方法处理天体物理学数据的论文来说明这一点。我们报告了这些论文领域专家的评估结果，他们提供了数值评分和类似审稿的反馈。然后，我们强调当前系统的优势、弱点和限制。最后，我们讨论了由AI驱动的科研的伦理影响，并反思这种技术与科学哲学的关系。我们已将代码公开发布于此httpsURL。同时，您也可以直接在网页上运行Denario的演示版，完整应用程序将部署在云上。 

---
# Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning 

**Title (ZH)**: 基于空间的SSRL：通过自我监督强化学习增强空间理解 

**Authors**: Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27606)  

**Abstract**: Spatial understanding remains a weakness of Large Vision-Language Models (LVLMs). Existing supervised fine-tuning (SFT) and recent reinforcement learning with verifiable rewards (RLVR) pipelines depend on costly supervision, specialized tools, or constrained environments that limit scale. We introduce Spatial-SSRL, a self-supervised RL paradigm that derives verifiable signals directly from ordinary RGB or RGB-D images. Spatial-SSRL automatically formulates five pretext tasks that capture 2D and 3D spatial structure: shuffled patch reordering, flipped patch recognition, cropped patch inpainting, regional depth ordering, and relative 3D position prediction. These tasks provide ground-truth answers that are easy to verify and require no human or LVLM annotation. Training on our tasks substantially improves spatial reasoning while preserving general visual capabilities. On seven spatial understanding benchmarks in both image and video settings, Spatial-SSRL delivers average accuracy gains of 4.63% (3B) and 3.89% (7B) over the Qwen2.5-VL baselines. Our results show that simple, intrinsic supervision enables RLVR at scale and provides a practical route to stronger spatial intelligence in LVLMs. 

**Abstract (ZH)**: Spatial理解仍是一大难题，宏观视觉-语言模型的短板。现有的监督细调（SFT）和最近的可验证奖励强化学习（RLVR）管道依赖于昂贵的监督、专门的工具或受限的环境，从而限制了模型的规模。我们介绍了一种自我监督的RL范式Spatial-SSRL，该范式直接从普通的RGB或RGB-D图像中提取可验证的信号。Spatial-SSRL自动制定了五个预训练任务，捕捉二维和三维空间结构：打乱贴图的重排序、翻转贴图识别、裁剪贴图修复、区域深度排序以及相对三维位置预测。这些任务提供了易于验证的真实答案，无需人类或LVLM注释。通过我们的任务进行训练，显著改善了空间推理能力，同时保持了通用视觉能力。在七个空间理解基准测试中的图像和视频设置下，Spatial-SSRL分别在Qwen2.5-VL基线上取得了4.63%（3B）和3.89%（7B）的平均准确率提升。我们的结果表明，简单的、内在的监督能够实现大规模的RLVR，并提供了一条通往更强空间智能的实用途径。 

---
# Atlas-Alignment: Making Interpretability Transferable Across Language Models 

**Title (ZH)**: Atlas-Alignment: 让可解释性在语言模型之间可传递 

**Authors**: Bruno Puri, Jim Berend, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2510.27413)  

**Abstract**: Interpretability is crucial for building safe, reliable, and controllable language models, yet existing interpretability pipelines remain costly and difficult to scale. Interpreting a new model typically requires costly training of model-specific sparse autoencoders, manual or semi-automated labeling of SAE components, and their subsequent validation. We introduce Atlas-Alignment, a framework for transferring interpretability across language models by aligning unknown latent spaces to a Concept Atlas - a labeled, human-interpretable latent space - using only shared inputs and lightweight representational alignment techniques. Once aligned, this enables two key capabilities in previously opaque models: (1) semantic feature search and retrieval, and (2) steering generation along human-interpretable atlas concepts. Through quantitative and qualitative evaluations, we show that simple representational alignment methods enable robust semantic retrieval and steerable generation without the need for labeled concept data. Atlas-Alignment thus amortizes the cost of explainable AI and mechanistic interpretability: by investing in one high-quality Concept Atlas, we can make many new models transparent and controllable at minimal marginal cost. 

**Abstract (ZH)**: Atlas-Alignment: 跨语言模型的知识迁移以实现解释性和可操控性 

---
# QiNN-QJ: A Quantum-inspired Neural Network with Quantum Jump for Multimodal Sentiment Analysis 

**Title (ZH)**: QiNN-QJ：一种基于量子跃迁的多模态情感分析量子启发神经网络 

**Authors**: Yiwei Chen, Kehuan Yan, Yu Pan, Daoyi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.27091)  

**Abstract**: Quantum theory provides non-classical principles, such as superposition and entanglement, that inspires promising paradigms in machine learning. However, most existing quantum-inspired fusion models rely solely on unitary or unitary-like transformations to generate quantum entanglement. While theoretically expressive, such approaches often suffer from training instability and limited generalizability. In this work, we propose a Quantum-inspired Neural Network with Quantum Jump (QiNN-QJ) for multimodal entanglement modelling. Each modality is firstly encoded as a quantum pure state, after which a differentiable module simulating the QJ operator transforms the separable product state into the entangled representation. By jointly learning Hamiltonian and Lindblad operators, QiNN-QJ generates controllable cross-modal entanglement among modalities with dissipative dynamics, where structured stochasticity and steady-state attractor properties serve to stabilize training and constrain entanglement shaping. The resulting entangled states are projected onto trainable measurement vectors to produce predictions. In addition to achieving superior performance over the state-of-the-art models on benchmark datasets, including CMU-MOSI, CMU-MOSEI, and CH-SIMS, QiNN-QJ facilitates enhanced post-hoc interpretability through von-Neumann entanglement entropy. This work establishes a principled framework for entangled multimodal fusion and paves the way for quantum-inspired approaches in modelling complex cross-modal correlations. 

**Abstract (ZH)**: 基于量子跃迁的量子启发式神经网络多模态纠缠建模（QiNN-QJ） 

---
# Jasmine: A Simple, Performant and Scalable JAX-based World Modeling Codebase 

**Title (ZH)**: Jasmine: 一个基于JAX的简单、高性能且可扩展的世界建模代码库 

**Authors**: Mihir Mahajan, Alfred Nguyen, Franz Srambical, Stefan Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2510.27002)  

**Abstract**: While world models are increasingly positioned as a pathway to overcoming data scarcity in domains such as robotics, open training infrastructure for world modeling remains nascent. We introduce Jasmine, a performant JAX-based world modeling codebase that scales from single hosts to hundreds of accelerators with minimal code changes. Jasmine achieves an order-of-magnitude faster reproduction of the CoinRun case study compared to prior open implementations, enabled by performance optimizations across data loading, training and checkpointing. The codebase guarantees fully reproducible training and supports diverse sharding configurations. By pairing Jasmine with curated large-scale datasets, we establish infrastructure for rigorous benchmarking pipelines across model families and architectural ablations. 

**Abstract (ZH)**: 尽管世界模型在克服机器人等领域数据稀缺性方面越来越受到重视，但开放训练基础设施的世界建模仍处于初级阶段。我们引入了Jasmine，这是一个基于JAX的世界模型代码库，可以从单个主机扩展到数百个加速器，并且只需进行最少的代码更改即可实现规模扩展。Jasmine通过在数据加载、训练和检查点方面进行性能优化，相比于之前的开放实现实现了数量级的速度提升。该代码库保证了训练的完全可重复性，并支持多种切分配置。通过将Jasmine与精心挑选的大规模数据集相结合，我们建立了跨模型家族和架构删减的严格基准测试管道基础设施。 

---
