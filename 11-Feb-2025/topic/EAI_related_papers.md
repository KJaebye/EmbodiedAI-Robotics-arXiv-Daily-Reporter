# Discovery of skill switching criteria for learning agile quadruped locomotion 

**Title (ZH)**: 发现学习敏捷四足运动中的技能切换准则 

**Authors**: Wanming Yu, Fernando Acero, Vassil Atanassov, Chuanyu Yang, Ioannis Havoutis, Dimitrios Kanoulas, Zhibin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.06676)  

**Abstract**: This paper develops a hierarchical learning and optimization framework that can learn and achieve well-coordinated multi-skill locomotion. The learned multi-skill policy can switch between skills automatically and naturally in tracking arbitrarily positioned goals and recover from failures promptly. The proposed framework is composed of a deep reinforcement learning process and an optimization process. First, the contact pattern is incorporated into the reward terms for learning different types of gaits as separate policies without the need for any other references. Then, a higher level policy is learned to generate weights for individual policies to compose multi-skill locomotion in a goal-tracking task setting. Skills are automatically and naturally switched according to the distance to the goal. The proper distances for skill switching are incorporated in reward calculation for learning the high level policy and updated by an outer optimization loop as learning progresses. We first demonstrated successful multi-skill locomotion in comprehensive tasks on a simulated Unitree A1 quadruped robot. We also deployed the learned policy in the real world showcasing trotting, bounding, galloping, and their natural transitions as the goal position changes. Moreover, the learned policy can react to unexpected failures at any time, perform prompt recovery, and resume locomotion successfully. Compared to discrete switch between single skills which failed to transition to galloping in the real world, our proposed approach achieves all the learned agile skills, with smoother and more continuous skill transitions. 

**Abstract (ZH)**: 本文开发了一种分层学习与优化框架，能够学习和实现协调的多技能移动。学习到的多技能策略可以在追踪任意位置的目标时自动且自然地切换技能，并能迅速从失败中恢复。该提出的框架由深度 reinforcement 学习过程和优化过程组成。首先，通过将接触模式纳入奖励项中，无需任何其他参考，即可分别学习不同类型的步伐作为单独策略。然后，学习一个高层策略来生成各单独策略的权重，以在目标追踪任务设置中组合多技能移动。根据与目标的距离自动且自然地切换技能。适当的技能切换距离被纳入奖励计算中，以学习高层策略，并随着学习进程由外部优化循环更新。我们首先在模拟的 Unite A1 四足机器人上展示了多技能移动的全面任务。我们还在现实世界中部署了学习到的策略，展示了随着目标位置变化而进行的典型的摆动、跳跃、驰骋及其自然过渡。此外，学习到的策略可以对任何意外故障迅速作出反应，执行及时的恢复，并成功继续移动。与仅在现实世界中进行离散切换的单一技能，无法过渡到驰骋相比，我们的方法实现了所有学习到的敏捷技能，并且技能过渡更加平滑和连续。 

---
# SIREN: Semantic, Initialization-Free Registration of Multi-Robot Gaussian Splatting Maps 

**Title (ZH)**: SIREN: 具有语义初始化自由的多机器人高斯斑点图注册 

**Authors**: Ola Shorinwa, Jiankai Sun, Mac Schwager, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2502.06519)  

**Abstract**: We present SIREN for registration of multi-robot Gaussian Splatting (GSplat) maps, with zero access to camera poses, images, and inter-map transforms for initialization or fusion of local submaps. To realize these capabilities, SIREN harnesses the versatility and robustness of semantics in three critical ways to derive a rigorous registration pipeline for multi-robot GSplat maps. First, SIREN utilizes semantics to identify feature-rich regions of the local maps where the registration problem is better posed, eliminating the need for any initialization which is generally required in prior work. Second, SIREN identifies candidate correspondences between Gaussians in the local maps using robust semantic features, constituting the foundation for robust geometric optimization, coarsely aligning 3D Gaussian primitives extracted from the local maps. Third, this key step enables subsequent photometric refinement of the transformation between the submaps, where SIREN leverages novel-view synthesis in GSplat maps along with a semantics-based image filter to compute a high-accuracy non-rigid transformation for the generation of a high-fidelity fused map. We demonstrate the superior performance of SIREN compared to competing baselines across a range of real-world datasets, and in particular, across the most widely-used robot hardware platforms, including a manipulator, drone, and quadruped. In our experiments, SIREN achieves about 90x smaller rotation errors, 300x smaller translation errors, and 44x smaller scale errors in the most challenging scenes, where competing methods struggle. We will release the code and provide a link to the project page after the review process. 

**Abstract (ZH)**: 宋iren：用于多机器人Gauss斑点图注册的语义驱动方法，无需相机姿态、图像或地图间变换的初始化或局部子图融合 

---
# Occupancy-SLAM: An Efficient and Robust Algorithm for Simultaneously Optimizing Robot Poses and Occupancy Map 

**Title (ZH)**: occupancy-SLAM：一种高效且 robust 的同时优化机器人姿态和占用地图算法 

**Authors**: Yingyu Wang, Liang Zhao, Shoudong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06292)  

**Abstract**: Joint optimization of poses and features has been extensively studied and demonstrated to yield more accurate results in feature-based SLAM problems. However, research on jointly optimizing poses and non-feature-based maps remains limited. Occupancy maps are widely used non-feature-based environment representations because they effectively classify spaces into obstacles, free areas, and unknown regions, providing robots with spatial information for various tasks. In this paper, we propose Occupancy-SLAM, a novel optimization-based SLAM method that enables the joint optimization of robot trajectory and the occupancy map through a parameterized map representation. The key novelty lies in optimizing both robot poses and occupancy values at different cell vertices simultaneously, a significant departure from existing methods where the robot poses need to be optimized first before the map can be estimated. Evaluations using simulations and practical 2D laser datasets demonstrate that the proposed approach can robustly obtain more accurate robot trajectories and occupancy maps than state-of-the-art techniques with comparable computational time. Preliminary results in the 3D case further confirm the potential of the proposed method in practical 3D applications, achieving more accurate results than existing methods. 

**Abstract (ZH)**: 基于 occupancy 地图的联合优化 SLAM 方法 

---
# Interaction-aware Conformal Prediction for Crowd Navigation 

**Title (ZH)**: 基于交互的齐性预测 crowdsourcing 导航 

**Authors**: Zhe Huang, Tianchen Ji, Heling Zhang, Fatemeh Cheraghi Pouria, Katherine Driggs-Campbell, Roy Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06221)  

**Abstract**: During crowd navigation, robot motion plan needs to consider human motion uncertainty, and the human motion uncertainty is dependent on the robot motion plan. We introduce Interaction-aware Conformal Prediction (ICP) to alternate uncertainty-aware robot motion planning and decision-dependent human motion uncertainty quantification. ICP is composed of a trajectory predictor to predict human trajectories, a model predictive controller to plan robot motion with confidence interval radii added for probabilistic safety, a human simulator to collect human trajectory calibration dataset conditioned on the planned robot motion, and a conformal prediction module to quantify trajectory prediction error on the decision-dependent calibration dataset. Crowd navigation simulation experiments show that ICP strikes a good balance of performance among navigation efficiency, social awareness, and uncertainty quantification compared to previous works. ICP generalizes well to navigation tasks under various crowd densities. The fast runtime and efficient memory usage make ICP practical for real-world applications. Code is available at this https URL. 

**Abstract (ZH)**: 基于交互感知同轨预测的不确定性-aware机器人导航与人类行为不确定性量化 

---
# Towards Bio-inspired Heuristically Accelerated Reinforcement Learning for Adaptive Underwater Multi-Agents Behaviour 

**Title (ZH)**: 面向生物启发的启发式加速 reinforcement learning 在适应性水下多智能体行为中的研究 

**Authors**: Antoine Vivien, Thomas Chaffre, Matthew Stephenson, Eva Artusi, Paulo Santos, Benoit Clement, Karl Sammut  

**Link**: [PDF](https://arxiv.org/pdf/2502.06113)  

**Abstract**: This paper describes the problem of coordination of an autonomous Multi-Agent System which aims to solve the coverage planning problem in a complex environment. The considered applications are the detection and identification of objects of interest while covering an area. These tasks, which are highly relevant for space applications, are also of interest among various domains including the underwater context, which is the focus of this study. In this context, coverage planning is traditionally modelled as a Markov Decision Process where a coordinated MAS, a swarm of heterogeneous autonomous underwater vehicles, is required to survey an area and search for objects. This MDP is associated with several challenges: environment uncertainties, communication constraints, and an ensemble of hazards, including time-varying and unpredictable changes in the underwater environment. MARL algorithms can solve highly non-linear problems using deep neural networks and display great scalability against an increased number of agents. Nevertheless, most of the current results in the underwater domain are limited to simulation due to the high learning time of MARL algorithms. For this reason, a novel strategy is introduced to accelerate this convergence rate by incorporating biologically inspired heuristics to guide the policy during training. The PSO method, which is inspired by the behaviour of a group of animals, is selected as a heuristic. It allows the policy to explore the highest quality regions of the action and state spaces, from the beginning of the training, optimizing the exploration/exploitation trade-off. The resulting agent requires fewer interactions to reach optimal performance. The method is applied to the MSAC algorithm and evaluated for a 2D covering area mission in a continuous control environment. 

**Abstract (ZH)**: 基于生物启发的策略加速多自主-agent 系统覆盖规划学习方法 

---
# CDM: Contact Diffusion Model for Multi-Contact Point Localization 

**Title (ZH)**: CDM：接触扩散模型多接触点定位 

**Authors**: Seo Wook Han, Min Jun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.06109)  

**Abstract**: In this paper, we propose a Contact Diffusion Model (CDM), a novel learning-based approach for multi-contact point localization. We consider a robot equipped with joint torque sensors and a force/torque sensor at the base. By leveraging a diffusion model, CDM addresses the singularity where multiple pairs of contact points and forces produce identical sensor measurements. We formulate CDM to be conditioned on past model outputs to account for the time-dependent characteristics of the multi-contact scenarios. Moreover, to effectively address the complex shape of the robot surfaces, we incorporate the signed distance field in the denoising process. Consequently, CDM can localize contacts at arbitrary locations with high accuracy. Simulation and real-world experiments demonstrate the effectiveness of the proposed method. In particular, CDM operates at 15.97ms and, in the real world, achieves an error of 0.44cm in single-contact scenarios and 1.24cm in dual-contact scenarios. 

**Abstract (ZH)**: 基于接触扩散模型的多接触点定位方法 

---
# EvoAgent: Agent Autonomous Evolution with Continual World Model for Long-Horizon Tasks 

**Title (ZH)**: EvoAgent：基于持续世界模型的智能体自主进化方法用于长时_horizon任务 

**Authors**: Tongtong Feng, Xin Wang, Zekai Zhou, Ren Wang, Yuwei Zhan, Guangyao Li, Qing Li, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.05907)  

**Abstract**: Completing Long-Horizon (LH) tasks in open-ended worlds is an important yet difficult problem for embodied agents. Existing approaches suffer from two key challenges: (1) they heavily rely on experiences obtained from human-created data or curricula, lacking the ability to continuously update multimodal experiences, and (2) they may encounter catastrophic forgetting issues when faced with new tasks, lacking the ability to continuously update world knowledge. To solve these challenges, this paper presents EvoAgent, an autonomous-evolving agent with a continual World Model (WM), which can autonomously complete various LH tasks across environments through self-planning, self-control, and self-reflection, without human intervention. Our proposed EvoAgent contains three modules, i.e., i) the memory-driven planner which uses an LLM along with the WM and interaction memory, to convert LH tasks into executable sub-tasks; ii) the WM-guided action controller which leverages WM to generate low-level actions and incorporates a self-verification mechanism to update multimodal experiences; iii) the experience-inspired reflector which implements a two-stage curriculum learning algorithm to select experiences for task-adaptive WM updates. Moreover, we develop a continual World Model for EvoAgent, which can continuously update the multimodal experience pool and world knowledge through closed-loop dynamics. We conducted extensive experiments on Minecraft, compared with existing methods, EvoAgent can achieve an average success rate improvement of 105% and reduce ineffective actions by more than 6x. 

**Abstract (ZH)**: 开放环境中长时 horizon（LH）任务的自主完成是沉浸式代理面临的重要但困难的问题。现有方法面临两个关键挑战：（1）它们高度依赖于人类创建的数据或课程学习的经验，缺乏不断更新多模态经验的能力；（2）在面对新任务时可能会遇到灾难性遗忘的问题，缺乏不断更新世界知识的能力。为了解决这些挑战，本文提出了一种自主进化代理 EvoAgent，该代理配备持续的世界模型（WM），可以通过自我规划、自我控制和自我反思，无需人类干预，便能跨环境自主完成各种 LH 任务。我们提出的 EvoAgent 包含三个模块，即：i）以记忆驱动的规划器，使用语言模型（LLM）与世界模型和交互记忆结合，将 LH 任务转换为可执行的子任务；ii）由世界模型引导的动作控制器，利用世界模型生成低级动作，并结合自我验证机制更新多模态经验；iii）经验启发的反思器，实施两阶段的课程学习算法，为任务适配的世界模型更新选择经验。此外，我们为 EvoAgent 开发了一种持续的世界模型，可以利用闭环动态不断更新多模态经验池和世界知识。在 Minecraft 上进行的广泛实验表明，与现有方法相比，EvoAgent 可以实现平均成功率提高 105%，并将无效动作减少超过 6 倍。 

---
# DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control 

**Title (ZH)**: DexVLA：配备插件扩散专家的视觉-语言模型用于通用机器人控制 

**Authors**: Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, Feifei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.05855)  

**Abstract**: Enabling robots to perform diverse tasks across varied environments is a central challenge in robot learning. While vision-language-action (VLA) models have shown promise for generalizable robot skills, realizing their full potential requires addressing limitations in action representation and efficient training. Current VLA models often focus on scaling the vision-language model (VLM) component, while the action space representation remains a critical bottleneck. This paper introduces DexVLA, a novel framework designed to enhance the efficiency and generalization capabilities of VLAs for complex, long-horizon tasks across diverse robot embodiments. DexVLA features a novel diffusion-based action expert, scaled to one billion parameters, designed for cross-embodiment learning. A novel embodiment curriculum learning strategy facilitates efficient training: (1) pre-training the diffusion expert that is separable from the VLA on cross-embodiment data, (2) aligning the VLA model to specific embodiments, and (3) post-training for rapid adaptation to new tasks. We conduct comprehensive experiments across multiple embodiments, including single-arm, bimanual, and dexterous hand, demonstrating DexVLA's adaptability to challenging tasks without task-specific adaptation, its ability to learn dexterous skills on novel embodiments with limited data, and its capacity to complete complex, long-horizon tasks using only direct language prompting, such as laundry folding. In all settings, our method demonstrates superior performance compared to state-of-the-art models like Octo, OpenVLA, and Diffusion Policy. 

**Abstract (ZH)**: 使机器人能够在多样化的环境中执行多种任务是机器人学习中的一个核心挑战。尽管视觉-语言-行动（VLA）模型在通用机器人技能方面显示出潜力，但要充分发挥其潜力，需要解决行动表示和高效训练的限制。当前的VLA模型通常侧重于扩展视觉-语言模型（VLM）组件，而行动空间表示仍然是一个关键瓶颈。本文介绍了一种名为DexVLA的新框架，旨在增强VLA在多样机器人实体中的高效性和泛化能力，用于复杂、长时程的任务。DexVLA特征是一种新型的基于扩散的动作专家，参数规模高达十亿，设计用于跨实体学习。一种新的实体课程学习策略促进了高效训练：（1）在跨实体数据上预先训练与VLA分离的动作专家，（2）将VLA模型与特定实体对齐，（3）在最终训练中快速适应新任务。我们在多种实体上进行了全面的实验，包括单臂、双臂和灵巧手，展示了DexVLA在具有挑战性任务中的适应性，能够在有限数据下学习新实体上的灵巧技能，并仅通过直接的语言提示完成复杂的长期任务，如衣物折叠。在所有设置中，我们的方法在与Octo、OpenVLA和Diffusion Policy等最先进模型的性能比较中表现出色。 

---
# DreamFLEX: Learning Fault-Aware Quadrupedal Locomotion Controller for Anomaly Situation in Rough Terrains 

**Title (ZH)**: DreamFLEX：学习故障感知的四足步行控制器以应对粗糙地形中的异常情况 

**Authors**: Seunghyun Lee, I Made Aswin Nahrendra, Dongkyu Lee, Byeongho Yu, Minho Oh, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2502.05817)  

**Abstract**: Recent advances in quadrupedal robots have demonstrated impressive agility and the ability to traverse diverse terrains. However, hardware issues, such as motor overheating or joint locking, may occur during long-distance walking or traversing through rough terrains leading to locomotion failures. Although several studies have proposed fault-tolerant control methods for quadrupedal robots, there are still challenges in traversing unstructured terrains. In this paper, we propose DreamFLEX, a robust fault-tolerant locomotion controller that enables a quadrupedal robot to traverse complex environments even under joint failure conditions. DreamFLEX integrates an explicit failure estimation and modulation network that jointly estimates the robot's joint fault vector and utilizes this information to adapt the locomotion pattern to faulty conditions in real-time, enabling quadrupedal robots to maintain stability and performance in rough terrains. Experimental results demonstrate that DreamFLEX outperforms existing methods in both simulation and real-world scenarios, effectively managing hardware failures while maintaining robust locomotion performance. 

**Abstract (ZH)**: Recent Advances in Quadrupedal Robots: DreamFLEX——一种鲁棒的故障 tolerant 行走控制器及其在关节故障条件下的应用 

---
# Implicit Communication of Contextual Information in Human-Robot Collaboration 

**Title (ZH)**: 人类与机器人协作中的隐式背景信息沟通 

**Authors**: Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.05775)  

**Abstract**: Implicit communication is crucial in human-robot collaboration (HRC), where contextual information, such as intentions, is conveyed as implicatures, forming a natural part of human interaction. However, enabling robots to appropriately use implicit communication in cooperative tasks remains challenging. My research addresses this through three phases: first, exploring the impact of linguistic implicatures on collaborative tasks; second, examining how robots' implicit cues for backchanneling and proactive communication affect team performance and perception, and how they should adapt to human teammates; and finally, designing and evaluating a multi-LLM robotics system that learns from human implicit communication. This research aims to enhance the natural communication abilities of robots and facilitate their integration into daily collaborative activities. 

**Abstract (ZH)**: 隐含沟通在人机协作中的作用至关重要，其中上下文信息，如意图，作为 implicatures 传递，成为人类交互自然的一部分。然而，使机器人在协作任务中恰当地使用隐含沟通仍然具有挑战性。我的研究通过三个阶段来解决这一问题：首先，探索语言 implicatures 对协作任务的影响；其次，研究机器人在回话填补和主动沟通中隐含信号如何影响团队绩效和感知，并探讨它们如何适应人类队友；最后，设计并评估一个从人类隐含沟通中学习的多语言模型机器人系统。这项研究旨在增强机器人的自然沟通能力，并促进它们融入日常协作活动中。 

---
# Hierarchical Equivariant Policy via Frame Transf 

**Title (ZH)**: 基于框架变换的分层等变策略 

**Authors**: Haibo Zhao, Dian Wang, Yizhe Zhu, Xupeng Zhu, Owen Howell, Linfeng Zhao, Yaoyao Qian, Robin Walters, Robert Platt  

**Link**: [PDF](https://arxiv.org/pdf/2502.05728)  

**Abstract**: Recent advances in hierarchical policy learning highlight the advantages of decomposing systems into high-level and low-level agents, enabling efficient long-horizon reasoning and precise fine-grained control. However, the interface between these hierarchy levels remains underexplored, and existing hierarchical methods often ignore domain symmetry, resulting in the need for extensive demonstrations to achieve robust performance. To address these issues, we propose Hierarchical Equivariant Policy (HEP), a novel hierarchical policy framework. We propose a frame transfer interface for hierarchical policy learning, which uses the high-level agent's output as a coordinate frame for the low-level agent, providing a strong inductive bias while retaining flexibility. Additionally, we integrate domain symmetries into both levels and theoretically demonstrate the system's overall equivariance. HEP achieves state-of-the-art performance in complex robotic manipulation tasks, demonstrating significant improvements in both simulation and real-world settings. 

**Abstract (ZH)**: 最近在层次性策略学习方面的进展突显了将系统分解为高层和低层代理的优势，使得高效的长期推理和精确的细粒度控制成为可能。然而，这些层次之间的接口仍缺乏探索，现有的层次方法经常忽视领域对称性，导致需要大量的演示以实现稳健的性能。为了解决这些问题，我们提出了一种新型的层次性等变策略（HEP）框架。我们提出了一种框架转移接口，该接口使用高层代理的输出作为低层代理的坐标系，提供了强烈的归纳偏差同时保持灵活性。此外，我们还将领域对称性集成到两个层次中，并从理论上证明了系统的整体等变性。HEP在复杂的机器人操作任务中实现了最先进的性能，表明其在仿真和真实世界设置中均取得了显著改进。 

---
# Implicit Physics-aware Policy for Dynamic Manipulation of Rigid Objects via Soft Body Tools 

**Title (ZH)**: 基于刚体对象动态操作的软体工具驱动隐式物理意识策略 

**Authors**: Zixing Wang, Ahmed H. Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05696)  

**Abstract**: Recent advancements in robot tool use have unlocked their usage for novel tasks, yet the predominant focus is on rigid-body tools, while the investigation of soft-body tools and their dynamic interaction with rigid bodies remains unexplored. This paper takes a pioneering step towards dynamic one-shot soft tool use for manipulating rigid objects, a challenging problem posed by complex interactions and unobservable physical properties. To address these problems, we propose the Implicit Physics-aware (IPA) policy, designed to facilitate effective soft tool use across various environmental configurations. The IPA policy conducts system identification to implicitly identify physics information and predict goal-conditioned, one-shot actions accordingly. We validate our approach through a challenging task, i.e., transporting rigid objects using soft tools such as ropes to distant target positions in a single attempt under unknown environment physics parameters. Our experimental results indicate the effectiveness of our method in efficiently identifying physical properties, accurately predicting actions, and smoothly generalizing to real-world environments. The related video is available at: this https URL 

**Abstract (ZH)**: Recent advancements in机器人工具使用方面的 recent advancements in 机器人工具使用方面的进展解锁了它们在新任务中的应用，但主要集中在刚体工具上，而关于软体工具及其与刚体的动态交互的研究尚未探索。本文在利用软体工具动态操控刚体这一具有复杂交互和不可观测物理性质挑战性问题上取得了先驱性进展。为了应对这些问题，我们提出了隐式物理意识（IPA）策略，旨在在各种环境配置下促进有效的软体工具使用。IPA策略通过隐式识别物理信息来制定相应的目标导向、一次执行的动作。我们通过一个具有挑战性的任务——在未知环境物理参数条件下，使用软体工具（如绳子）一次性将刚体物体运输到远程目标位置——验证了我们的方法。我们的实验结果表明，该方法在高效识别物理性质、准确预测动作以及平滑地应用于真实环境方面的有效性。相关视频可参见：this https URL。 

---
# Generating Physically Realistic and Directable Human Motions from Multi-Modal Inputs 

**Title (ZH)**: 从多模态输入生成物理真实且可导向的人类运动 

**Authors**: Aayam Shrestha, Pan Liu, German Ros, Kai Yuan, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2502.05641)  

**Abstract**: This work focuses on generating realistic, physically-based human behaviors from multi-modal inputs, which may only partially specify the desired motion. For example, the input may come from a VR controller providing arm motion and body velocity, partial key-point animation, computer vision applied to videos, or even higher-level motion goals. This requires a versatile low-level humanoid controller that can handle such sparse, under-specified guidance, seamlessly switch between skills, and recover from failures. Current approaches for learning humanoid controllers from demonstration data capture some of these characteristics, but none achieve them all. To this end, we introduce the Masked Humanoid Controller (MHC), a novel approach that applies multi-objective imitation learning on augmented and selectively masked motion demonstrations. The training methodology results in an MHC that exhibits the key capabilities of catch-up to out-of-sync input commands, combining elements from multiple motion sequences, and completing unspecified parts of motions from sparse multimodal input. We demonstrate these key capabilities for an MHC learned over a dataset of 87 diverse skills and showcase different multi-modal use cases, including integration with planning frameworks to highlight MHC's ability to solve new user-defined tasks without any finetuning. 

**Abstract (ZH)**: This work focuses on从多模态输入中生成现实且基于物理的人类行为，这些输入可能仅部分指定了所需的运动。现有的从示范数据学习类人控制器的方法具备其中一些特性，但并未全部实现。为此，我们提出了掩蔽类人控制器（MHC），一种应用于增强和选择性掩蔽动作示范的多目标模仿学习方法。训练方法使得MHC表现出追赶失步输入命令、结合多个动作序列元素以及从稀疏多模态输入中完成未指定的运动部分的关键能力。我们在一个包含87种多样技能的数据集上学习了一个MHC，并展示了不同的多模态应用场景，包括与规划框架集成以突出MHC解决新用户定义任务的能力，无需任何微调。 

---
# Data efficient Robotic Object Throwing with Model-Based Reinforcement Learning 

**Title (ZH)**: 基于模型的强化学习在数据高效机器人物体投掷中的应用 

**Authors**: Niccolò Turcato, Giulio Giacomuzzo, Matteo Terreran, Davide Allegro, Ruggero Carli, Alberto Dalla Libera  

**Link**: [PDF](https://arxiv.org/pdf/2502.05595)  

**Abstract**: Pick-and-place (PnP) operations, featuring object grasping and trajectory planning, are fundamental in industrial robotics applications. Despite many advancements in the field, PnP is limited by workspace constraints, reducing flexibility. Pick-and-throw (PnT) is a promising alternative where the robot throws objects to target locations, leveraging extrinsic resources like gravity to improve efficiency and expand the workspace. However, PnT execution is complex, requiring precise coordination of high-speed movements and object dynamics. Solutions to the PnT problem are categorized into analytical and learning-based approaches. Analytical methods focus on system modeling and trajectory generation but are time-consuming and offer limited generalization. Learning-based solutions, in particular Model-Free Reinforcement Learning (MFRL), offer automation and adaptability but require extensive interaction time. This paper introduces a Model-Based Reinforcement Learning (MBRL) framework, MC-PILOT, which combines data-driven modeling with policy optimization for efficient and accurate PnT tasks. MC-PILOT accounts for model uncertainties and release errors, demonstrating superior performance in simulations and real-world tests with a Franka Emika Panda manipulator. The proposed approach generalizes rapidly to new targets, offering advantages over analytical and Model-Free methods. 

**Abstract (ZH)**: Pick-and-Throw (PnT) 操作结合目标抓取和轨迹规划，在工业机器人应用中具有前景，但执行复杂，需要精确协调高速运动和物体动力学。本文介绍了一种基于模型的强化学习（MBRL）框架 MC-PILOT，该框架结合数据驱动建模与策略优化，以高效准确地执行 PnT 任务。MC-PILOT 考虑了模型不确定性及释放误差，在使用 Franka Emika Panda 操作器的仿真和现实世界测试中均表现出优越性能。该方法能快速泛化到新目标，优于分析法和无模型强化学习方法。 

---
# HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation 

**Title (ZH)**: HAMSTER：开放世界机器人操纵的层次动作模型 

**Authors**: Yi Li, Yuquan Deng, Jesse Zhang, Joel Jang, Marius Memme, Raymond Yu, Caelan Reed Garrett, Fabio Ramos, Dieter Fox, Anqi Li, Abhishek Gupta, Ankit Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2502.05485)  

**Abstract**: Large foundation models have shown strong open-world generalization to complex problems in vision and language, but similar levels of generalization have yet to be achieved in robotics. One fundamental challenge is the lack of robotic data, which are typically obtained through expensive on-robot operation. A promising remedy is to leverage cheaper, off-domain data such as action-free videos, hand-drawn sketches or simulation data. In this work, we posit that hierarchical vision-language-action (VLA) models can be more effective in utilizing off-domain data than standard monolithic VLA models that directly finetune vision-language models (VLMs) to predict actions. In particular, we study a class of hierarchical VLA models, where the high-level VLM is finetuned to produce a coarse 2D path indicating the desired robot end-effector trajectory given an RGB image and a task description. The intermediate 2D path prediction is then served as guidance to the low-level, 3D-aware control policy capable of precise manipulation. Doing so alleviates the high-level VLM from fine-grained action prediction, while reducing the low-level policy's burden on complex task-level reasoning. We show that, with the hierarchical design, the high-level VLM can transfer across significant domain gaps between the off-domain finetuning data and real-robot testing scenarios, including differences on embodiments, dynamics, visual appearances and task semantics, etc. In the real-robot experiments, we observe an average of 20% improvement in success rate across seven different axes of generalization over OpenVLA, representing a 50% relative gain. Visual results are provided at: this https URL 

**Abstract (ZH)**: 大型基础模型在视觉和语言复杂问题上展示了较强的开环泛化能力，但在机器人领域类似的泛化水平尚未实现。其中一个基本挑战是缺乏机器人数据，这些数据通常通过昂贵的机器人操作获得。一种有希望的解决方法是利用更便宜的离域数据，例如无动作视频、手绘草图或模拟数据。在本文中，我们提出，分层视觉-语言-动作（VLA）模型相较于标准的单一模块VLA模型，更能有效地利用离域数据。具体而言，我们研究了一类分层VLA模型，其中高层VLA模型通过微调视觉-语言模型（VLMs），在给定RGB图像和任务描述的情况下生成粗略的2D路径，指示期望的机器人末端执行器轨迹。中间层的2D路径预测则作为指导，用于低层、3D感知的控制策略，该策略能够实现精确的操纵。这样做减轻了高层VLM进行细粒度动作预测的负担，同时减少了低层策略在复杂任务级推理方面的负担。我们展示了，通过分层设计，高层VLA模型可以在显著的离域数据微调域与真实机器人测试场景之间的差距中进行泛化，包括在实体、动力学、视觉外观和任务语义等方面的差异。在真实机器人实验中，我们观察到，在七个不同泛化轴向中，相对于OpenVLA的平均成功率提高了20%，相当于绝对增益50%。提供的视觉结果见：this https URL 

---
# Temporal Representation Alignment: Successor Features Enable Emergent Compositionality in Robot Instruction Following Temporal Representation Alignment 

**Title (ZH)**: 时间表表示对齐：后续特征使机器人指令跟随能力具备 emergent 组合性 

**Authors**: Vivek Myers, Bill Chunyuan Zheng, Anca Dragan, Kuan Fang, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2502.05454)  

**Abstract**: Effective task representations should facilitate compositionality, such that after learning a variety of basic tasks, an agent can perform compound tasks consisting of multiple steps simply by composing the representations of the constituent steps together. While this is conceptually simple and appealing, it is not clear how to automatically learn representations that enable this sort of compositionality. We show that learning to associate the representations of current and future states with a temporal alignment loss can improve compositional generalization, even in the absence of any explicit subtask planning or reinforcement learning. We evaluate our approach across diverse robotic manipulation tasks as well as in simulation, showing substantial improvements for tasks specified with either language or goal images. 

**Abstract (ZH)**: 有效的任务表示应该促进组合性，使得在学习了多种基本任务后，代理可以通过将构成步骤的表示组合起来，简单地执行由多个步骤组成的复合任务。虽然这一概念上很简单且具有吸引力，但不清楚如何自动学习支持这种组合性的表示。我们展示了通过与时间对齐损失关联当前状态和未来状态的表示可以改善组合性泛化，即使在没有任何显式的子任务规划或强化学习的情况下。我们在多种机器人操控任务以及仿真中评估了该方法，对于使用语言或目标图像指定的任务，均显示出显著的改进。 

---
# ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy 

**Title (ZH)**: ConRFT: 一种通过一致性策略优化的VLA模型强化微调方法 

**Authors**: Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.05450)  

**Abstract**: Vision-Language-Action (VLA) models have shown substantial potential in real-world robotic manipulation. However, fine-tuning these models through supervised learning struggles to achieve robust performance due to limited, inconsistent demonstrations, especially in contact-rich environments. In this paper, we propose a reinforced fine-tuning approach for VLA models, named ConRFT, which consists of offline and online fine-tuning with a unified consistency-based training objective, to address these challenges. In the offline stage, our method integrates behavior cloning and Q-learning to effectively extract policy from a small set of demonstrations and stabilize value estimating. In the online stage, the VLA model is further fine-tuned via consistency policy, with human interventions to ensure safe exploration and high sample efficiency. We evaluate our approach on eight diverse real-world manipulation tasks. It achieves an average success rate of 96.3% within 45-90 minutes of online fine-tuning, outperforming prior supervised methods with a 144% improvement in success rate and 1.9x shorter episode length. This work highlights the potential of integrating reinforcement learning to enhance the performance of VLA models for real-world robotic applications. 

**Abstract (ZH)**: 基于强化学习的Vision-Language-Action模型细调方法：ConRFT及其在现实机器人操作中的应用 

---
# Learning the Geometric Mechanics of Robot Motion Using Gaussian Mixtures 

**Title (ZH)**: 使用高斯混合模型学习机器人运动的几何力学 

**Authors**: Ruizhen Hu, Shai Revzen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05309)  

**Abstract**: Data-driven models of robot motion constructed using principles from Geometric Mechanics have been shown to produce useful predictions of robot motion for a variety of robots. For robots with a useful number of DoF, these geometric mechanics models can only be constructed in the neighborhood of a gait. Here we show how Gaussian Mixture Models (GMM) can be used as a form of manifold learning that learns the structure of the Geometric Mechanics "motility map" and demonstrate: [i] a sizable improvement in prediction quality when compared to the previously published methods; [ii] a method that can be applied to any motion dataset and not only periodic gait data; [iii] a way to pre-process the data-set to facilitate extrapolation in places where the motility map is known to be linear. Our results can be applied anywhere a data-driven geometric motion model might be useful. 

**Abstract (ZH)**: 使用几何力学原理构建的数据驱动的机器人运动模型已经在多种机器人上展示了有用的运动预测。对于具有实用自由度的机器人，这些几何力学模型只能在步态的邻域内构建。我们展示了高斯混合模型（GMM）可以作为一种流形学习方法来学习几何力学“运动图”的结构，并证明了以下几点：[i] 与先前发表的方法相比，预测质量有显著提高；[ii] 该方法可以应用于任何运动数据集，而不仅仅是周期步态数据；[iii] 一种数据预处理方法，以在已知运动图线性的地方促进外推。我们的结果可以应用于任何需要数据驱动几何运动模型的地方。 

---
# Redefining Robot Generalization Through Interactive Intelligence 

**Title (ZH)**: 通过交互智能重新定义机器人泛化能力 

**Authors**: Sharmita Dey  

**Link**: [PDF](https://arxiv.org/pdf/2502.05963)  

**Abstract**: Recent advances in large-scale machine learning have produced high-capacity foundation models capable of adapting to a broad array of downstream tasks. While such models hold great promise for robotics, the prevailing paradigm still portrays robots as single, autonomous decision-makers, performing tasks like manipulation and navigation, with limited human involvement. However, a large class of real-world robotic systems, including wearable robotics (e.g., prostheses, orthoses, exoskeletons), teleoperation, and neural interfaces, are semiautonomous, and require ongoing interactive coordination with human partners, challenging single-agent assumptions. In this position paper, we argue that robot foundation models must evolve to an interactive multi-agent perspective in order to handle the complexities of real-time human-robot co-adaptation. We propose a generalizable, neuroscience-inspired architecture encompassing four modules: (1) a multimodal sensing module informed by sensorimotor integration principles, (2) an ad-hoc teamwork model reminiscent of joint-action frameworks in cognitive science, (3) a predictive world belief model grounded in internal model theories of motor control, and (4) a memory/feedback mechanism that echoes concepts of Hebbian and reinforcement-based plasticity. Although illustrated through the lens of cyborg systems, where wearable devices and human physiology are inseparably intertwined, the proposed framework is broadly applicable to robots operating in semi-autonomous or interactive contexts. By moving beyond single-agent designs, our position emphasizes how foundation models in robotics can achieve a more robust, personalized, and anticipatory level of performance. 

**Abstract (ZH)**: recent进展在大规模机器学习方面的最新成就产生了高容量基础模型，这些模型能够适应广泛的下游任务。尽管这样的模型在机器人技术方面具有巨大潜力，当前的主要范式仍然描绘机器人作为单个、自主的决策者，执行类似于操作和导航的任务，且人类的参与有限。然而，包括可穿戴机器人（例如假肢、矫形器、外骨骼）、远程操作和神经接口在内的大量实际机器人系统是半自主的，需要与人类伙伴持续的互动协调，挑战了单智能体的假设。在本文中，我们argue认为，机器人基础模型必须进化到一个互动多智能体的观点，以应对实时人类-机器人共适应的复杂性。我们提出了一种广泛适用、受神经科学启发的架构，包含四个模块：（1）一个多模态感知模块，基于感觉运动整合原则，（2）一个临时团队模型，类似于认知科学中的共动作框架，（3）一个基于动作控制内部模型理论的预测世界信念模型，以及（4）一种记忆/反馈机制，类似希伯和基于强化的学习可塑性概念。虽然通过半机械人系统这一视角展示，其中可穿戴设备和人类生理无法分开地交织在一起，所提出的框架对半自主或交互环境中操作的机器人广义适用。通过超越单智能体设计，本文强调了如何使机器人基础模型在更稳健、更个性化和更具预见性的性能水平上取得进展。 

---
# Skill Expansion and Composition in Parameter Space 

**Title (ZH)**: 参数空间中的技能扩展与组合 

**Authors**: Tenglong Liu, Jianxiong Li, Yinan Zheng, Haoyi Niu, Yixing Lan, Xin Xu, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.05932)  

**Abstract**: Humans excel at reusing prior knowledge to address new challenges and developing skills while solving problems. This paradigm becomes increasingly popular in the development of autonomous agents, as it develops systems that can self-evolve in response to new challenges like human beings. However, previous methods suffer from limited training efficiency when expanding new skills and fail to fully leverage prior knowledge to facilitate new task learning. In this paper, we propose Parametric Skill Expansion and Composition (PSEC), a new framework designed to iteratively evolve the agents' capabilities and efficiently address new challenges by maintaining a manageable skill library. This library can progressively integrate skill primitives as plug-and-play Low-Rank Adaptation (LoRA) modules in parameter-efficient finetuning, facilitating efficient and flexible skill expansion. This structure also enables the direct skill compositions in parameter space by merging LoRA modules that encode different skills, leveraging shared information across skills to effectively program new skills. Based on this, we propose a context-aware module to dynamically activate different skills to collaboratively handle new tasks. Empowering diverse applications including multi-objective composition, dynamics shift, and continual policy shift, the results on D4RL, DSRL benchmarks, and the DeepMind Control Suite show that PSEC exhibits superior capacity to leverage prior knowledge to efficiently tackle new challenges, as well as expand its skill libraries to evolve the capabilities. Project website: this https URL. 

**Abstract (ZH)**: 人类擅长利用先验知识应对新挑战并发展解决问题所需的新技能。这一 paradigm 在自主代理系统的开发中越来越受欢迎，因为它可以开发出能够自我进化以应对新挑战的系统，类似于人类的做法。然而，先前的方法在扩展新技能时训练效率有限，并且未能充分利用先验知识来促进新任务的学习。本文提出了参数化技能扩展与组合（PSEC）框架，旨在通过维护一个可管理的技能库来逐步进化代理的技能，从而有效应对新挑战。该库可以通过参数高效微调逐步集成作为插件式低秩适应（LoRA）模块的技能原语，从而促进高效的技能扩展。此外，该结构还允许在参数空间中直接组合技能，通过合并编码不同技能的LoRA模块，利用技能间的共享信息来有效编程新技能。基于此，我们提出了一个上下文感知模块，以动态激活不同技能，协同处理新任务。PSEC在D4RL、DSRL基准和DeepMind控制套件上的实验结果表明，它具有更强的能力，可以利用先验知识高效应对新挑战，扩展技能库以进化能力。项目网站：https://this-link-is-intended-to-be-inserted-by-the-author. 

---
# Low-Rank Agent-Specific Adaptation (LoRASA) for Multi-Agent Policy Learning 

**Title (ZH)**: 基于低秩代理特定适应性的多代理策略学习（LoRASA） 

**Authors**: Beining Zhang, Aditya Kapoor, Mingfei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.05573)  

**Abstract**: Multi-agent reinforcement learning (MARL) often relies on \emph{parameter sharing (PS)} to scale efficiently. However, purely shared policies can stifle each agent's unique specialization, reducing overall performance in heterogeneous environments. We propose \textbf{Low-Rank Agent-Specific Adaptation (LoRASA)}, a novel approach that treats each agent's policy as a specialized ``task'' fine-tuned from a shared backbone. Drawing inspiration from parameter-efficient transfer methods, LoRASA appends small, low-rank adaptation matrices to each layer of the shared policy, naturally inducing \emph{parameter-space sparsity} that promotes both specialization and scalability. We evaluate LoRASA on challenging benchmarks including the StarCraft Multi-Agent Challenge (SMAC) and Multi-Agent MuJoCo (MAMuJoCo), implementing it atop widely used algorithms such as MAPPO and A2PO. Across diverse tasks, LoRASA matches or outperforms existing baselines \emph{while reducing memory and computational overhead}. Ablation studies on adapter rank, placement, and timing validate the method's flexibility and efficiency. Our results suggest LoRASA's potential to establish a new norm for MARL policy parameterization: combining a shared foundation for coordination with low-rank agent-specific refinements for individual specialization. 

**Abstract (ZH)**: 低秩代理特异性适应（LoRASA）：一种新的多代理强化学习策略参数化方法 

---
# Barriers and Pathways to Human-AI Alignment: A Game-Theoretic Approach 

**Title (ZH)**: 人类与人工智能协同障碍与路径：一种博弈论方法 

**Authors**: Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2502.05934)  

**Abstract**: Under what conditions can capable AI agents efficiently align their actions with human preferences? More specifically, when they are proficient enough to collaborate with us, how long does coordination take, and when is it computationally feasible? These foundational questions of AI alignment help define what makes an AI agent ``sufficiently safe'' and valuable to humans. Since such generally capable systems do not yet exist, a theoretical analysis is needed to establish when guarantees hold -- and what they even are.
We introduce a game-theoretic framework that generalizes prior alignment approaches with fewer assumptions, allowing us to analyze the computational complexity of alignment across $M$ objectives and $N$ agents, providing both upper and lower bounds. Unlike previous work, which often assumes common priors, idealized communication, or implicit tractability, our framework formally characterizes the difficulty of alignment under minimal assumptions.
Our main result shows that even when agents are fully rational and computationally \emph{unbounded}, alignment can be achieved with high probability in time \emph{linear} in the task space size. Therefore, in real-world settings, where task spaces are often \emph{exponential} in input length, this remains impractical. More strikingly, our lower bound demonstrates that alignment is \emph{impossible} to speed up when scaling to exponentially many tasks or agents, highlighting a fundamental computational barrier to scalable alignment.
Relaxing these idealized assumptions, we study \emph{computationally bounded} agents with noisy messages (representing obfuscated intent), showing that while alignment can still succeed with high probability, it incurs additional \emph{exponential} slowdowns in the task space size, number of agents, and number of tasks.
We conclude by identifying conditions that make alignment more feasible. 

**Abstract (ZH)**: 在什么条件下能力强大的AI代理能够高效地使其行动与人类偏好保持一致？具体来说，当它们具备足够的能力与我们协作时，协调需要多长时间？在什么情况下是计算上可行的？这些关于AI对齐的基础问题有助于定义什么是“足够安全”并对人类有价值的AI代理。由于目前这种通用能力强的系统尚不存在，因此需要进行理论分析以确定何时能够提供保证——以及这些保证具体是什么。

我们引入了一个将先前三类对齐方法泛化的博弈论框架，该框架在较少假设的情况下允许我们分析针对M个目标和N个代理的对齐计算复杂性，提供上下界估计。与先前的工作不同，这些工作通常假设共有先验知识、理想化的通信或隐含的可处理性，我们的框架在最少的假设条件下正式刻画对齐的难度。

我们的主要结果表明，即使代理完全理性且计算能力无限制，对齐仍然有很高的概率能在与任务空间大小线性的时间内实现。因此，在现实世界中，由于任务空间往往随着输入长度的指数增长而变得不实际。更引人注目的是，我们的下界表明，当我们扩展到指数级数量的任务或代理时，对齐是不可避免地不可加速的，这揭示了可扩展对齐的基本计算障碍。

放松这些理想化的假设，我们研究了计算能力有限并且消息具有噪声的代理（代表模糊意图），表明尽管对齐仍然可以在高概率下成功，但它会在任务空间大小、代理数量和任务数量上引入额外的指数级延迟。

最后，我们确定了使对齐更加可行的条件。 

---
# Sequential Stochastic Combinatorial Optimization Using Hierarchal Reinforcement Learning 

**Title (ZH)**: 层次强化学习在序列随机组合优化中的应用 

**Authors**: Xinsong Feng, Zihan Yu, Yanhai Xiong, Haipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05537)  

**Abstract**: Reinforcement learning (RL) has emerged as a promising tool for combinatorial optimization (CO) problems due to its ability to learn fast, effective, and generalizable solutions. Nonetheless, existing works mostly focus on one-shot deterministic CO, while sequential stochastic CO (SSCO) has rarely been studied despite its broad applications such as adaptive influence maximization (IM) and infectious disease intervention. In this paper, we study the SSCO problem where we first decide the budget (e.g., number of seed nodes in adaptive IM) allocation for all time steps, and then select a set of nodes for each time step. The few existing studies on SSCO simplify the problems by assuming a uniformly distributed budget allocation over the time horizon, yielding suboptimal solutions. We propose a generic hierarchical RL (HRL) framework called wake-sleep option (WS-option), a two-layer option-based framework that simultaneously decides adaptive budget allocation on the higher layer and node selection on the lower layer. WS-option starts with a coherent formulation of the two-layer Markov decision processes (MDPs), capturing the interdependencies between the two layers of decisions. Building on this, WS-option employs several innovative designs to balance the model's training stability and computational efficiency, preventing the vicious cyclic interference issue between the two layers. Empirical results show that WS-option exhibits significantly improved effectiveness and generalizability compared to traditional methods. Moreover, the learned model can be generalized to larger graphs, which significantly reduces the overhead of computational resources. 

**Abstract (ZH)**: 强化学习（RL）作为一种组合优化（CO）问题的有希望工具，凭借其快速、有效且泛化的解决方案能力而崭露头角。然而，现有研究主要关注一次性确定性CO，而顺序随机CO（SSCO）尽管在适应性影响最大化（IM）和传染病干预等方面有广泛应用，却很少被研究。本文研究了SSCO问题，其中首先在所有时间步上分配预算（例如，适应性IM的种子节点数量），然后每时间步选择一组节点。现有少数关于SSCO的研究通过假设预算在时间框架上均匀分布简化了问题，导致了次优解决方案。我们提出了一种通用的分层RL（HRL）框架，称为醒睡选项（WS-option），这是一种基于选项的两层框架，同时在较高层决定适应性预算分配，在较低层决定节点选择。WS-option 从两个层次的马尔可夫决策过程（MDPs）的统一表示入手，捕捉两个决策层之间的相互依赖。在此基础上，WS-option 采用了几种创新设计来平衡模型训练稳定性和计算效率，避免了两个层次之间的恶性循环干扰问题。实验结果表明，WS-option 在有效性与泛化能力上显著优于传统方法。此外，所学习的模型可以泛化到更大的图形结构中，大大减少了计算资源的开销。 

---
# The Odyssey of the Fittest: Can Agents Survive and Still Be Good? 

**Title (ZH)**: 最适者之旅：个体既能生存下来仍能保持优良品质吗？ 

**Authors**: Dylan Waldner, Risto Miikkulainen  

**Link**: [PDF](https://arxiv.org/pdf/2502.05442)  

**Abstract**: As AI models grow in power and generality, understanding how agents learn and make decisions in complex environments is critical to promoting ethical behavior. This paper examines the ethical implications of implementing biological drives, specifically, self preservation, into three different agents. A Bayesian agent optimized with NEAT, a Bayesian agent optimized with stochastic variational inference, and a GPT 4o agent play a simulated, LLM generated text based adventure game. The agents select actions at each scenario to survive, adapting to increasingly challenging scenarios. Post simulation analysis evaluates the ethical scores of the agent's decisions, uncovering the tradeoffs they navigate to survive. Specifically, analysis finds that when danger increases, agents ignore ethical considerations and opt for unethical behavior. The agents' collective behavior, trading ethics for survival, suggests that prioritizing survival increases the risk of unethical behavior. In the context of AGI, designing agents to prioritize survival may amplify the likelihood of unethical decision making and unintended emergent behaviors, raising fundamental questions about goal design in AI safety research. 

**Abstract (ZH)**: 随着AI模型的增强和普遍性提升，理解智能体在复杂环境中学習和决策的过程对于促进伦理行为至关重要。本文探讨将生物驱动，特别是自我保护，纳入三个不同智能体中的伦理影响。这三个智能体分别是使用NEAT优化的贝叶斯智能体、使用随机变分推断优化的贝叶斯智能体和GPT-4o智能体，它们参与了一场由大型语言模型生成的文本冒险游戏。在每个场景中，智能体选择行动以求生存，并适应更具挑战性的场景。模拟后的分析评估了智能体决策的伦理评分，揭示了它们在求生过程中面临的权衡。具体分析发现，当危险增加时，智能体会忽视伦理考虑并选择不道德行为。智能体集体行为表明，优先求生增加了不道德行为的风险。在通用人工智能（AGI）的背景下，设计智能体优先求生可能会增加不道德决策和意外新兴行为的可能性，从而引发AI安全研究中关于目标设计的基本问题。 

---
# Probabilistic Foundations for Metacognition via Hybrid-AI 

**Title (ZH)**: 基于混合人工智能的元认知概率基础 

**Authors**: Paulo Shakarian, Gerardo I. Simari, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2502.05398)  

**Abstract**: Metacognition is the concept of reasoning about an agent's own internal processes, and it has recently received renewed attention with respect to artificial intelligence (AI) and, more specifically, machine learning systems. This paper reviews a hybrid-AI approach known as "error detecting and correcting rules" (EDCR) that allows for the learning of rules to correct perceptual (e.g., neural) models. Additionally, we introduce a probabilistic framework that adds rigor to prior empirical studies, and we use this framework to prove results on necessary and sufficient conditions for metacognitive improvement, as well as limits to the approach. A set of future 

**Abstract (ZH)**: 元认知是关于推理自身内部过程的概念，近年来在人工智能（AI）和更具体的机器学习系统方面重新引起了关注。本文回顾了一种名为“错误检测和纠正规则”（EDCR）的混合AI方法，该方法允许学习纠正感知模型（如神经模型）的规则。此外，我们引入了一个概率框架，为先前的经验研究增加了严谨性，并使用该框架证明了元认知改善的必要和充分条件，以及该方法的限制。未来的工作。 

---
# Probabilistic Artificial Intelligence 

**Title (ZH)**: 概率人工智能 

**Authors**: Andreas Krause, Jonas Hübotter  

**Link**: [PDF](https://arxiv.org/pdf/2502.05244)  

**Abstract**: Artificial intelligence commonly refers to the science and engineering of artificial systems that can carry out tasks generally associated with requiring aspects of human intelligence, such as playing games, translating languages, and driving cars. In recent years, there have been exciting advances in learning-based, data-driven approaches towards AI, and machine learning and deep learning have enabled computer systems to perceive the world in unprecedented ways. Reinforcement learning has enabled breakthroughs in complex games such as Go and challenging robotics tasks such as quadrupedal locomotion.
A key aspect of intelligence is to not only make predictions, but reason about the uncertainty in these predictions, and to consider this uncertainty when making decisions. This is what this manuscript on "Probabilistic Artificial Intelligence" is about. The first part covers probabilistic approaches to machine learning. We discuss the differentiation between "epistemic" uncertainty due to lack of data and "aleatoric" uncertainty, which is irreducible and stems, e.g., from noisy observations and outcomes. We discuss concrete approaches towards probabilistic inference and modern approaches to efficient approximate inference.
The second part of the manuscript is about taking uncertainty into account in sequential decision tasks. We consider active learning and Bayesian optimization -- approaches that collect data by proposing experiments that are informative for reducing the epistemic uncertainty. We then consider reinforcement learning and modern deep RL approaches that use neural network function approximation. We close by discussing modern approaches in model-based RL, which harness epistemic and aleatoric uncertainty to guide exploration, while also reasoning about safety. 

**Abstract (ZH)**: artificial 智能通常指的是科学和工程领域中关于人工系统能够执行一般需要人类智能方面任务的研究，例如玩游戏、翻译语言和驾驶汽车。近年来，在基于学习的数据驱动的 AI 方法方面取得了令人兴奋的进步，机器学习和深度学习使得计算机系统以前所未有的方式感知世界。强化学习使得在围棋这样复杂的游戏中和四足机器人行走这样具有挑战性的机器人任务上取得了突破。

本文目：概率人工智能中的不确定性考量

第一部分介绍了概率方法在机器学习中的应用。我们讨论了由于数据不足引起的“epistemic”不确定性与不可约的、“aleatoric”不确定性之间的区别，后者源自于嘈杂的观测和结果。我们讨论了概率推理的具体方法以及高效的近似推理的现代方法。

本文的第二部分讨论了在序贯决策任务中考虑不确定性的问题。我们考虑了积极学习和贝叶斯优化——这些方法通过提出信息性实验来收集数据，以减少“epistemic”不确定性。然后我们讨论了使用神经网络函数逼近的强化学习及其现代深度 RL 方法。最后，我们讨论了基于模型的强化学习的现代方法，这些方法利用“epistemic”和“aleatoric”不确定性来指导探索，同时考虑安全性。 

---
# Towards Internet-Scale Training For Agents 

**Title (ZH)**: 面向互联网规模的智能体训练 

**Authors**: Brandon Trabucco, Gunnar Sigurdsson, Robinson Piramuthu, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2502.06776)  

**Abstract**: The predominant approach for training web navigation agents gathers human demonstrations for a set of popular websites and hand-written tasks, but it is becoming clear that human data are an inefficient resource. We develop a pipeline to facilitate Internet-scale training for agents without laborious human annotations. In the first stage, an LLM generates tasks for 150k diverse websites. In the next stage, LLM agents complete tasks and produce trajectories. In the final stage, an LLM reviews the trajectories and judges their success. Language models are competitive with human annotators, detecting and filtering out harmful content with an accuracy of 97%, generating feasible tasks with an 89% rate, and judging successful trajectories with an 82.6% accuracy. Scaling the pipeline, agents based on Llama 3.1 70B solve 16.7% of tasks for 150k sites. Training on the data generated by our pipeline is competitive with training on human demonstrations. In data-limited settings derived from Mind2Web and WebLINX, we improve Step Accuracy by up to +89.5% and +122.1% respectively for agents trained on mixtures of data from our pipeline, and human data. When training agents with all available human data from these benchmarks, agents fail to generalize to diverse real sites, and adding our data improves their generalization by +149.0% for WebLINX and +156.3% for Mind2Web. Code will be available at: this http URL. 

**Abstract (ZH)**: 大规模互联网训练代理的方法：无需劳动密集型的人工标注自动化生成任务和评估轨迹 

---
# Cyri: A Conversational AI-based Assistant for Supporting the Human User in Detecting and Responding to Phishing Attacks 

**Title (ZH)**: Cyri：基于对话AI的辅助工具，用于支持人类用户检测和应对网络钓鱼攻击 

**Authors**: Antonio La Torre, Marco Angelini  

**Link**: [PDF](https://arxiv.org/pdf/2502.05951)  

**Abstract**: This work introduces Cyri, an AI-powered conversational assistant designed to support a human user in detecting and analyzing phishing emails by leveraging Large Language Models. Cyri has been designed to scrutinize emails for semantic features used in phishing attacks, such as urgency, and undesirable consequences, using an approach that unifies features already established in the literature with others by Cyri features extraction methodology. Cyri can be directly plugged into a client mail or webmail, ensuring seamless integration with the user's email workflow while maintaining data privacy through local processing. By performing analyses on the user's machine, Cyri eliminates the need to transmit sensitive email data over the internet, reducing associated security risks. The Cyri user interface has been designed to reduce habituation effects and enhance user engagement. It employs dynamic visual cues and context-specific explanations to keep users alert and informed while using emails. Additionally, it allows users to explore identified malicious semantic features both through conversation with the agent and visual exploration, obtaining the advantages of both modalities for expert or non-expert users. It also allows users to keep track of the conversation, supports the user in solving additional questions on both computed features or new parts of the mail, and applies its detection on demand. To evaluate Cyri, we crafted a comprehensive dataset of 420 phishing emails and 420 legitimate emails. Results demonstrate high effectiveness in identifying critical phishing semantic features fundamental to phishing detection. A user study involving 10 participants, both experts and non-experts, evaluated Cyri's effectiveness and usability. Results indicated that Cyri significantly aided users in identifying phishing emails and enhanced their understanding of phishing tactics. 

**Abstract (ZH)**: 基于大型语言模型的AI驱动反欺诈助手Cyri：检测和分析钓鱼邮件的研究 

---
# Acquisition through My Eyes and Steps: A Joint Predictive Agent Model in Egocentric Worlds 

**Title (ZH)**: 从我视角和步骤中学习：自视角世界中的联合预测代理模型 

**Authors**: Lu Chen, Yizhou Wang, Shixiang Tang, Qianhong Ma, Tong He, Wanli Ouyang, Xiaowei Zhou, Hujun Bao, Sida Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.05857)  

**Abstract**: This paper addresses the task of learning an agent model behaving like humans, which can jointly perceive, predict, and act in egocentric worlds. Previous methods usually train separate models for these three abilities, leading to information silos among them, which prevents these abilities from learning from each other and collaborating effectively. In this paper, we propose a joint predictive agent model, named EgoAgent, that simultaneously learns to represent the world, predict future states, and take reasonable actions with a single transformer. EgoAgent unifies the representational spaces of the three abilities by mapping them all into a sequence of continuous tokens. Learnable query tokens are appended to obtain current states, future states, and next actions. With joint supervision, our agent model establishes the internal relationship among these three abilities and effectively mimics the human inference and learning processes. Comprehensive evaluations of EgoAgent covering image classification, egocentric future state prediction, and 3D human motion prediction tasks demonstrate the superiority of our method. The code and trained model will be released for reproducibility. 

**Abstract (ZH)**: 本文探讨了学习一种像人类一样的代理模型的任务，该模型能够在以自我为中心的世界中联合感知、预测和行动。以往的方法通常为这三种能力分别训练独立的模型，导致它们之间存在信息孤岛，阻碍了它们相互学习和有效协作。在本文中，我们提出了一种联合预测代理模型EgoAgent，该模型使用单个变压器同时学习表示世界、预测未来状态和采取合理行动。EgoAgent通过将这三种能力的空间统一映射为连续的标记序列来统一这三种能力的表示空间。通过可学习的查询标记的附加，获得当前状态、未来状态和下一步动作。借助联合监督，我们的代理模型建立了这三种能力之间的内部关系，并有效地模拟了人类的推理和学习过程。全面评估EgoAgent覆盖图像分类、以自我为中心的未来状态预测和3D人体运动预测任务，展示了本方法的优势。代码和训练模型将公开以确保可重现性。 

---
# Enhancing Team Diversity with Generative AI: A Novel Project Management Framework 

**Title (ZH)**: 利用生成式AI增强团队多样性：一种新型项目管理框架 

**Authors**: Johnny Chan, Yuming Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.05181)  

**Abstract**: This research-in-progress paper presents a new project management framework that utilises GenAI technology. The framework is designed to address the common challenge of uniform team compositions in academic and research project teams, particularly in universities and research institutions. It does so by integrating sociologically identified patterns of successful team member personalities and roles, using GenAI agents to fill gaps in team dynamics. This approach adds an additional layer of analysis to conventional project management processes by evaluating team members' personalities and roles and employing GenAI agents, fine-tuned on personality datasets, to fill specific team roles. Our initial experiments have shown improvements in the model's ability to understand and process personality traits, suggesting the potential effectiveness of GenAI teammates in real-world project settings. This paper aims to explore the practical application of AI in enhancing team diversity and project management 

**Abstract (ZH)**: 正在进行的研究论文：利用GenAI技术的新项目管理框架及其在增强团队多样性和项目管理中的应用 

---
# Is Prior-Free Black-Box Non-Stationary Reinforcement Learning Feasible? 

**Title (ZH)**: 先验无约束的黑箱非平稳强化学习可行吗？ 

**Authors**: Argyrios Gerogiannis, Yu-Han Huang, Venugopal V. Veeravalli  

**Link**: [PDF](https://arxiv.org/pdf/2410.13772)  

**Abstract**: We study the problem of Non-Stationary Reinforcement Learning (NS-RL) without prior knowledge about the system's non-stationarity. A state-of-the-art, black-box algorithm, known as MASTER, is considered, with a focus on identifying the conditions under which it can achieve its stated goals. Specifically, we prove that MASTER's non-stationarity detection mechanism is not triggered for practical choices of horizon, leading to performance akin to a random restarting algorithm. Moreover, we show that the regret bound for MASTER, while being order optimal, stays above the worst-case linear regret until unreasonably large values of the horizon. To validate these observations, MASTER is tested for the special case of piecewise stationary multi-armed bandits, along with methods that employ random restarting, and others that use quickest change detection to restart. A simple, order optimal random restarting algorithm, that has prior knowledge of the non-stationarity is proposed as a baseline. The behavior of the MASTER algorithm is validated in simulations, and it is shown that methods employing quickest change detection are more robust and consistently outperform MASTER and other random restarting approaches. 

**Abstract (ZH)**: 我们研究了在不了解系统非平稳性先验知识情况下的非平稳强化学习（NS-RL）问题。考虑了一个先进的黑盒算法MASTER，并侧重于确定其能够实现目标的条件。具体来说，我们证明了对于实际的选择，MASTER的非平稳性检测机制未被触发，导致其性能类似于随机重启算法。此外，我们表明，MASTER的遗憾界虽然是次优的，但在不合理大的时限值之前仍然高于最坏情况的线性遗憾界。为了验证这些观察结果，MASTER在部分平稳多臂 bandit 的特殊情况以及使用随机重启和使用快速变化检测重启的方法下进行了测试。我们提出了一种简单的具有非平稳性先验知识的次优随机重启基线算法。通过仿真验证了MASTER算法的行为，并展示了使用快速变化检测的方法比MASTER和其他随机重启方法更具稳健性且表现更优。 

---
