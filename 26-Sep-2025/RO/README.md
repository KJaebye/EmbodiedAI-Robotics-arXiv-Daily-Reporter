# Taxonomy-aware Dynamic Motion Generation on Hyperbolic Manifolds 

**Title (ZH)**: 基于-taxonomy的双曲流形上动态运动生成 

**Authors**: Luis Augenstein, Noémie Jaquier, Tamim Asfour, Leonel Rozo  

**Link**: [PDF](https://arxiv.org/pdf/2509.21281)  

**Abstract**: Human-like motion generation for robots often draws inspiration from biomechanical studies, which often categorize complex human motions into hierarchical taxonomies. While these taxonomies provide rich structural information about how movements relate to one another, this information is frequently overlooked in motion generation models, leading to a disconnect between the generated motions and their underlying hierarchical structure. This paper introduces the \ac{gphdm}, a novel approach that learns latent representations preserving both the hierarchical structure of motions and their temporal dynamics to ensure physical consistency. Our model achieves this by extending the dynamics prior of the Gaussian Process Dynamical Model (GPDM) to the hyperbolic manifold and integrating it with taxonomy-aware inductive biases. Building on this geometry- and taxonomy-aware frameworks, we propose three novel mechanisms for generating motions that are both taxonomically-structured and physically-consistent: two probabilistic recursive approaches and a method based on pullback-metric geodesics. Experiments on generating realistic motion sequences on the hand grasping taxonomy show that the proposed GPHDM faithfully encodes the underlying taxonomy and temporal dynamics, and generates novel physically-consistent trajectories. 

**Abstract (ZH)**: 基于几何与分类知识的人形化机器人运动生成 

---
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
# RetoVLA: Reusing Register Tokens for Spatial Reasoning in Vision-Language-Action Models 

**Title (ZH)**: RetoVLA: 重用寄存器令牌进行视觉-语言-动作模型的空间推理 

**Authors**: Jiyeon Koo, Taewan Cho, Hyunjoon Kang, Eunseom Pyo, Tae Gyun Oh, Taeryang Kim, Andrew Jaeyong Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21243)  

**Abstract**: Recent Vision-Language-Action (VLA) models demonstrate remarkable generalization in robotics but are restricted by their substantial size and computational cost, limiting real-world deployment. However, conventional lightweighting methods often sacrifice critical capabilities, particularly spatial reasoning. This creates a trade-off between efficiency and performance. To address this challenge, our work reuses Register Tokens, which were introduced for artifact removal in Vision Transformers but subsequently discarded. We suppose that these tokens contain essential spatial information and propose RetoVLA, a novel architecture that reuses them directly by injecting them into the Action Expert.
RetoVLA maintains a lightweight structure while leveraging this repurposed spatial context to enhance reasoning. We demonstrate RetoVLA's effectiveness through a series of comprehensive experiments. On our custom-built 7-DOF robot arm, the model achieves a 17.1%p absolute improvement in success rates for complex manipulation tasks. Our results confirm that reusing Register Tokens directly enhances spatial reasoning, demonstrating that what was previously discarded as an artifact is in fact a valuable, unexplored resource for robotic intelligence. A video demonstration is available at: this https URL 

**Abstract (ZH)**: 近期的视觉-语言-动作（VLA）模型在机器人技术中展现了出色的泛化能力，但因其庞大的规模和高昂的计算成本而受到限制，阻碍了其实现真正的部署。然而，传统的轻量化方法经常会牺牲关键的能力，特别是在空间推理方面。这就造成了效率与性能之间的权衡。为了解决这一挑战，我们的工作重新利用了注册token（Register Tokens），这种token最初是在视觉变换器中引入以去除干扰物，之后被遗弃。我们假设这些token包含重要的空间信息，并提出了一种新的架构RetoVLA，该架构通过直接注入动作专家中来重新利用这些token，从而保持轻量级结构的同时提升推理能力。我们通过一系列全面的实验展示了RetoVLA的有效性。在我们自建的7-DOF机器人手臂上，该模型在复杂操作任务中的成功率绝对提高了17.1%。我们的结果证实，直接复用注册token能够增强空间推理能力，表明之前被视为干扰物的资源其实是提升机器人智能的宝贵且未被充分探索的资源。视频演示可访问：this https URL。 

---
# FSGlove: An Inertial-Based Hand Tracking System with Shape-Aware Calibration 

**Title (ZH)**: 基于惯性的人手跟踪系统：具有形状感知校准的方法 

**Authors**: Yutong Li, Jieyi Zhang, Wenqiang Xu, Tutian Tang, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.21242)  

**Abstract**: Accurate hand motion capture (MoCap) is vital for applications in robotics, virtual reality, and biomechanics, yet existing systems face limitations in capturing high-degree-of-freedom (DoF) joint kinematics and personalized hand shape. Commercial gloves offer up to 21 DoFs, which are insufficient for complex manipulations while neglecting shape variations that are critical for contact-rich tasks. We present FSGlove, an inertial-based system that simultaneously tracks up to 48 DoFs and reconstructs personalized hand shapes via DiffHCal, a novel calibration method. Each finger joint and the dorsum are equipped with IMUs, enabling high-resolution motion sensing. DiffHCal integrates with the parametric MANO model through differentiable optimization, resolving joint kinematics, shape parameters, and sensor misalignment during a single streamlined calibration. The system achieves state-of-the-art accuracy, with joint angle errors of less than 2.7 degree, and outperforms commercial alternatives in shape reconstruction and contact fidelity. FSGlove's open-source hardware and software design ensures compatibility with current VR and robotics ecosystems, while its ability to capture subtle motions (e.g., fingertip rubbing) bridges the gap between human dexterity and robotic imitation. Evaluated against Nokov optical MoCap, FSGlove advances hand tracking by unifying the kinematic and contact fidelity. Hardware design, software, and more results are available at: this https URL. 

**Abstract (ZH)**: 基于惯性的FSGlove手套：同时追踪48个自由度并重建个性化手型 

---
# SEEC: Stable End-Effector Control with Model-Enhanced Residual Learning for Humanoid Loco-Manipulation 

**Title (ZH)**: SEEC：基于模型增强残差学习的稳定末端执行器控制用于类人操作与 Manipulation 

**Authors**: Jaehwi Jang, Zhuoheng Wang, Ziyi Zhou, Feiyang Wu, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.21231)  

**Abstract**: Arm end-effector stabilization is essential for humanoid loco-manipulation tasks, yet it remains challenging due to the high degrees of freedom and inherent dynamic instability of bipedal robot structures. Previous model-based controllers achieve precise end-effector control but rely on precise dynamics modeling and estimation, which often struggle to capture real-world factors (e.g., friction and backlash) and thus degrade in practice. On the other hand, learning-based methods can better mitigate these factors via exploration and domain randomization, and have shown potential in real-world use. However, they often overfit to training conditions, requiring retraining with the entire body, and still struggle to adapt to unseen scenarios. To address these challenges, we propose a novel stable end-effector control (SEEC) framework with model-enhanced residual learning that learns to achieve precise and robust end-effector compensation for lower-body induced disturbances through model-guided reinforcement learning (RL) with a perturbation generator. This design allows the upper-body policy to achieve accurate end-effector stabilization as well as adapt to unseen locomotion controllers with no additional training. We validate our framework in different simulators and transfer trained policies to the Booster T1 humanoid robot. Experiments demonstrate that our method consistently outperforms baselines and robustly handles diverse and demanding loco-manipulation tasks. 

**Abstract (ZH)**: 基于模型增强残差学习的稳定末端执行器控制框架在仿人移动操作任务中的应用 

---
# Next-Generation Aerial Robots -- Omniorientational Strategies: Dynamic Modeling, Control, and Comparative Analysis 

**Title (ZH)**: 下一代空中机器人——全方位姿态策略：动态建模、控制与比较分析 

**Authors**: Ali Kafili Gavgani, Amin Talaeizadeh, Aria Alasty, Hossein Nejat Pishkenari, Esmaeil Najafi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21210)  

**Abstract**: Conventional multi-rotors are under-actuated systems, hindering them from independently controlling attitude from position. In this study, we present several distinct configurations that incorporate additional control inputs for manipulating the angles of the propeller axes. This addresses the mentioned limitations, making the systems "omniorientational". We comprehensively derived detailed dynamic models for all introduced configurations and validated by a methodology using Simscape Multibody simulations. Two controllers are designed: a sliding mode controller for robust handling of disturbances and a novel PID-based controller with gravity compensation integrating linear and non-linear allocators, designed for computational efficiency. A custom control allocation strategy is implemented to manage the input-non-affine nature of these systems, seeking to maximize battery life by minimizing the "Power Consumption Factor" defined in this study. Moreover, the controllers effectively managed harsh disturbances and uncertainties. Simulations compare and analyze the proposed configurations and controllers, majorly considering their power consumption. Furthermore, we conduct a qualitative comparison to evaluate the impact of different types of uncertainties on the control system, highlighting areas for potential model or hardware improvements. The analysis in this study provides a roadmap for future researchers to design omniorientational drones based on their design objectives, offering practical insights into configuration selection and controller design. This research aligns with the project SAC-1, one of the objectives of Sharif AgRoLab. 

**Abstract (ZH)**: 传统的多旋翼无人机是欠驱动系统，限制了其独立控制姿态与位置的能力。本研究提出了几种不同的配置，通过增加额外的控制输入来操控螺旋桨轴的角度，从而解决了上述限制，使系统具备全方位姿态控制能力。本文详细推导了所有引入配置的动态模型，并通过基于Simscape Multibody的仿真方法进行了验证。设计了两种控制器：滑模控制器用于稳健地处理干扰，以及一种新型基于PID的控制器，结合了线性和非线性分配器，用于提高计算效率。实现了一种自定义控制分配策略来管理这些系统的输入非线性特性，旨在通过最小化“功耗因子”来最大化电池寿命。此外，控制器有效处理了恶劣的干扰和不确定性。仿真比较和分析了所提出的配置和控制器，主要考虑它们的功耗。此外，我们进行了定性比较以评估不同类型的不确定性对控制系统的影响，指出了模型或硬件改进的潜在领域。本文的分析为未来研究人员设计全方位姿态控制无人机提供了方向，提供了关于配置选择和控制器设计的实用见解。该研究与Sharif AgRoLab项目SAC-1的目标相一致。 

---
# Human-like Navigation in a World Built for Humans 

**Title (ZH)**: 基于人类设计的世界中的类人导航 

**Authors**: Bhargav Chandaka, Gloria X. Wang, Haozhe Chen, Henry Che, Albert J. Zhai, Shenlong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21189)  

**Abstract**: When navigating in a man-made environment they haven't visited before--like an office building--humans employ behaviors such as reading signs and asking others for directions. These behaviors help humans reach their destinations efficiently by reducing the need to search through large areas. Existing robot navigation systems lack the ability to execute such behaviors and are thus highly inefficient at navigating within large environments. We present ReasonNav, a modular navigation system which integrates these human-like navigation skills by leveraging the reasoning capabilities of a vision-language model (VLM). We design compact input and output abstractions based on navigation landmarks, allowing the VLM to focus on language understanding and reasoning. We evaluate ReasonNav on real and simulated navigation tasks and show that the agent successfully employs higher-order reasoning to navigate efficiently in large, complex buildings. 

**Abstract (ZH)**: 基于推理的导航系统：利用视觉语言模型融合人类导航技能 

---
# DAGDiff: Guiding Dual-Arm Grasp Diffusion to Stable and Collision-Free Grasps 

**Title (ZH)**: DAGDiff: 引导双臂抓取扩散以实现稳定且无碰撞的抓取 

**Authors**: Md Faizal Karim, Vignesh Vembar, Keshab Patra, Gaurav Singh, K Madhava Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2509.21145)  

**Abstract**: Reliable dual-arm grasping is essential for manipulating large and complex objects but remains a challenging problem due to stability, collision, and generalization requirements. Prior methods typically decompose the task into two independent grasp proposals, relying on region priors or heuristics that limit generalization and provide no principled guarantee of stability. We propose DAGDiff, an end-to-end framework that directly denoises to grasp pairs in the SE(3) x SE(3) space. Our key insight is that stability and collision can be enforced more effectively by guiding the diffusion process with classifier signals, rather than relying on explicit region detection or object priors. To this end, DAGDiff integrates geometry-, stability-, and collision-aware guidance terms that steer the generative process toward grasps that are physically valid and force-closure compliant. We comprehensively evaluate DAGDiff through analytical force-closure checks, collision analysis, and large-scale physics-based simulations, showing consistent improvements over previous work on these metrics. Finally, we demonstrate that our framework generates dual-arm grasps directly on real-world point clouds of previously unseen objects, which are executed on a heterogeneous dual-arm setup where two manipulators reliably grasp and lift them. 

**Abstract (ZH)**: 可靠的双臂抓取对于操作大型和复杂对象至关重要，但由于稳定性和碰撞要求，仍然是一个具有挑战性的问题。先前的方法通常将任务分解为两个独立的抓取提案，依赖于区域先验或启发式方法，这些方法限制了泛化能力，并没有提供稳定的实质性保证。我们提出了DAGDiff，这是一个端到端框架，可以直接在SE(3) x SE(3)空间中去噪以生成抓取对。我们的关键洞察是，通过使用分类器信号引导扩散过程可以更有效地确保稳定性和避免碰撞，而不是依赖显式的区域检测或对象先验。为此，DAGDiff 结合了几何、稳定性和碰撞感知的引导项，使生成过程偏向于物理上有效的且满足力封闭的抓取。我们通过分析力封闭检查、碰撞分析和大规模物理仿真全面评估了DAGDiff，在这些指标上展示了相对于之前工作的改进。最后，我们证明了该框架可以直接在未见过的实际点云对象上生成双臂抓取，并在异构双臂操作设置中可靠地抓取和提升它们。 

---
# Automotive-ENV: Benchmarking Multimodal Agents in Vehicle Interface Systems 

**Title (ZH)**: Automotive-ENV：车辆界面系统中多模态代理的基准测试 

**Authors**: Junfeng Yan, Biao Wu, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21143)  

**Abstract**: Multimodal agents have demonstrated strong performance in general GUI interactions, but their application in automotive systems has been largely unexplored. In-vehicle GUIs present distinct challenges: drivers' limited attention, strict safety requirements, and complex location-based interaction patterns. To address these challenges, we introduce Automotive-ENV, the first high-fidelity benchmark and interaction environment tailored for vehicle GUIs. This platform defines 185 parameterized tasks spanning explicit control, implicit intent understanding, and safety-aware tasks, and provides structured multimodal observations with precise programmatic checks for reproducible evaluation. Building on this benchmark, we propose ASURADA, a geo-aware multimodal agent that integrates GPS-informed context to dynamically adjust actions based on location, environmental conditions, and regional driving norms. Experiments show that geo-aware information significantly improves success on safety-aware tasks, highlighting the importance of location-based context in automotive environments. We will release Automotive-ENV, complete with all tasks and benchmarking tools, to further the development of safe and adaptive in-vehicle agents. 

**Abstract (ZH)**: 多模态代理在通用GUI交互中展现了强大的性能，但在汽车系统中的应用尚未被广泛探索。车载GUI提出了独特的挑战：驾驶员注意力有限、严格的安全要求以及复杂的基于位置的交互模式。为应对这些挑战，我们引入了Automotive-ENV，这是首款针对车辆GUI定制的高度逼真基准和交互环境。该平台定义了185个参数化任务，涵盖显式控制、隐式意图理解以及安全感知任务，并提供了结构化的多模态观察数据和精确的程序检查，以实现可复现的评估。在此基准之上，我们提出了ASURADA，一种地理感知的多模态代理，该代理结合GPS信息的上下文，基于位置、环境条件和地区驾驶规范动态调整行动。实验结果显示，地理感知信息显著提高了安全感知任务的成功率，强调了车载环境中地理位置上下文的重要性。我们将发布Automotive-ENV，包含所有任务和基准测试工具，以促进安全且适应性强的车载代理的发展。 

---
# Rich State Observations Empower Reinforcement Learning to Surpass PID: A Drone Ball Balancing Study 

**Title (ZH)**: 丰富状态观测增强强化学习超越PID：无人机球平衡研究 

**Authors**: Mingjiang Liu, Hailong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.21122)  

**Abstract**: This paper addresses a drone ball-balancing task, in which a drone stabilizes a ball atop a movable beam through cable-based interaction. We propose a hierarchical control framework that decouples high-level balancing policy from low-level drone control, and train a reinforcement learning (RL) policy to handle the high-level decision-making. Simulation results show that the RL policy achieves superior performance compared to carefully tuned PID controllers within the same hierarchical structure. Through systematic comparative analysis, we demonstrate that RL's advantage stems not from improved parameter tuning or inherent nonlinear mapping capabilities, but from its ability to effectively utilize richer state observations. These findings underscore the critical role of comprehensive state representation in learning-based systems and suggest that enhanced sensing could be instrumental in improving controller performance. 

**Abstract (ZH)**: 基于绳索交互的无人机球平衡任务的研究：一种层次控制框架及强化学习政策的训练与评估 

---
# Cross-Modal Instructions for Robot Motion Generation 

**Title (ZH)**: 跨模态指令生成机器人运动 

**Authors**: William Barron, Xiaoxiang Dong, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.21107)  

**Abstract**: Teaching robots novel behaviors typically requires motion demonstrations via teleoperation or kinaesthetic teaching, that is, physically guiding the robot. While recent work has explored using human sketches to specify desired behaviors, data collection remains cumbersome, and demonstration datasets are difficult to scale. In this paper, we introduce an alternative paradigm, Learning from Cross-Modal Instructions, where robots are shaped by demonstrations in the form of rough annotations, which can contain free-form text labels, and are used in lieu of physical motion. We introduce the CrossInstruct framework, which integrates cross-modal instructions as examples into the context input to a foundational vision-language model (VLM). The VLM then iteratively queries a smaller, fine-tuned model, and synthesizes the desired motion over multiple 2D views. These are then subsequently fused into a coherent distribution over 3D motion trajectories in the robot's workspace. By incorporating the reasoning of the large VLM with a fine-grained pointing model, CrossInstruct produces executable robot behaviors that generalize beyond the environment of in the limited set of instruction examples. We then introduce a downstream reinforcement learning pipeline that leverages CrossInstruct outputs to efficiently learn policies to complete fine-grained tasks. We rigorously evaluate CrossInstruct on benchmark simulation tasks and real hardware, demonstrating effectiveness without additional fine-tuning and providing a strong initialization for policies subsequently refined via reinforcement learning. 

**Abstract (ZH)**: 从跨模态指令学习 

---
# Flight Dynamics to Sensing Modalities: Exploiting Drone Ground Effect for Accurate Edge Detection 

**Title (ZH)**: 从飞行动力学到传感模态：利用无人机地面效应进行精确边缘检测 

**Authors**: Chenyu Zhao, Jingao Xu, Ciyu Ruan, Haoyang Wang, Shengbo Wang, Jiaqi Li, Jirong Zha, Weijie Hong, Zheng Yang, Yunhao Liu, Xiao-Ping Zhang, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.21085)  

**Abstract**: Drone-based rapid and accurate environmental edge detection is highly advantageous for tasks such as disaster relief and autonomous navigation. Current methods, using radars or cameras, raise deployment costs and burden lightweight drones with high computational demands. In this paper, we propose AirTouch, a system that transforms the ground effect from a stability "foe" in traditional flight control views, into a "friend" for accurate and efficient edge detection. Our key insight is that analyzing drone basic attitude sensor readings and flight commands allows us to detect ground effect changes. Such changes typically indicate the drone flying over a boundary of two materials, making this information valuable for edge detection. We approach this insight through theoretical analysis, algorithm design, and implementation, fully leveraging the ground effect as a new sensing modality without compromising drone flight stability, thereby achieving accurate and efficient scene edge detection. We also compare this new sensing modality with vision-based methods to clarify its exclusive advantages in resource efficiency and detection capability. Extensive evaluations demonstrate that our system achieves a high detection accuracy with mean detection distance errors of 0.051m, outperforming the baseline method performance by 86%. With such detection performance, our system requires only 43 mW power consumption, contributing to this new sensing modality for low-cost and highly efficient edge detection. 

**Abstract (ZH)**: 基于无人机的快速准确环境边缘检测对灾害救援和自主导航等任务具有高度优势。当前方法使用雷达或摄像头会增加部署成本，并给轻型无人机带来较高的计算负担。本文提出AirTouch系统，将其地面效应从传统飞行控制中的稳定性“敌对因素”转变为准确高效边缘检测的“盟友”。我们的关键洞察是，通过分析无人机的基本姿态传感器读数和飞行指令，可以检测地面效应的变化。这种变化通常表明无人机飞越了两种材料的边界，从而使得这些信息在边缘检测中具有价值。我们通过理论分析、算法设计和实现，充分利用地面效应作为一种新的传感模态，同时不牺牲无人机飞行稳定性，从而实现准确高效场景边缘检测。此外，我们将这种新的传感模态与基于视觉的方法进行比较，以阐明其在资源效率和检测能力方面的独特优势。广泛评估表明，我们的系统在平均检测距离误差为0.051米的情况下实现了高检测精度，相比基线方法性能提升86%。凭借如此高的检测性能，我们的系统仅需43毫瓦的功率消耗，从而推动了低成本和高效边缘检测的新型传感模态的发展。 

---
# Normalizing Flows are Capable Visuomotor Policy Learning Models 

**Title (ZH)**: 正态流是有能力的visuomotor策略学习模型 

**Authors**: Simon Kristoffersson Lind, Jialong Li, Maj Stenmark, Volker Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2509.21073)  

**Abstract**: The field of general purpose robotics has recently embraced powerful probabilistic models, such as diffusion models, to model and learn complex behaviors. However, these models often come with significant trade-offs, namely high computational costs for inference and a fundamental inability to quantify output uncertainty. We argue that a model's trustworthiness, a critical factor for reliable, general-purpose robotics, is inherently linked to its ability to provide confidence measures.
In this work, we introduce Normalizing Flows Policy, a novel visuomotor policy learning model based on Normalizing Flows. We show that Normalizing Flows are a natural and powerful alternative to diffusion models, providing both a statistically sound measure of confidence and a highly efficient inference process. Through comprehensive experiments across four distinct simulated robotic tasks, we demonstrate that Normalizing Flows Policy achieves performance comparable to, and often surpassing, Diffusion Policy, and it does so not only with improved sample efficiency but also with up to 30 times faster inference. Additionally, our ablation study validates several key architectural and training techniques that enable Normalizing Flows to perform well in this domain. 

**Abstract (ZH)**: 通用机器人领域的研究最近采纳了强大的概率模型，如扩散模型，以建模和学习复杂行为。然而，这些模型通常伴随着显著的权衡，即推断的高计算成本和根本无法量化输出不确定性。我们认为，模型的可信度——这是可靠且通用的机器人技术的关键因素——与其提供信心度量的能力息息相关。

在本文中，我们提出了一种基于归一化流的视觉运动策略学习模型——归一化流策略。我们展示归一化流是一种自然且强大的扩散模型替代方案，提供了统计上合理的信心度量和高度高效的推理过程。通过在四个不同的模拟机器人任务上进行全面实验，我们证明归一化流策略不仅达到了与扩散策略相当的性能，而且在样本效率上有所改进，并且推断速度最快可提高30倍。此外，我们的消融研究验证了几种关键的架构和训练技术，这些技术使归一化流在该领域中表现良好。 

---
# MPC-based Deep Reinforcement Learning Method for Space Robotic Control with Fuel Sloshing Mitigation 

**Title (ZH)**: 基于MPC的深度强化学习空间机器人控制方法及燃料晃动缓解 

**Authors**: Mahya Ramezani, M. Amin Alandihallaj, Barış Can Yalçın, Miguel Angel Olivares Mendez, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2509.21045)  

**Abstract**: This paper presents an integrated Reinforcement Learning (RL) and Model Predictive Control (MPC) framework for autonomous satellite docking with a partially filled fuel tank. Traditional docking control faces challenges due to fuel sloshing in microgravity, which induces unpredictable forces affecting stability. To address this, we integrate Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) RL algorithms with MPC, leveraging MPC's predictive capabilities to accelerate RL training and improve control robustness. The proposed approach is validated through Zero-G Lab of SnT experiments for planar stabilization and high-fidelity numerical simulations for 6-DOF docking with fuel sloshing dynamics. Simulation results demonstrate that SAC-MPC achieves superior docking accuracy, higher success rates, and lower control effort, outperforming standalone RL and PPO-MPC methods. This study advances fuel-efficient and disturbance-resilient satellite docking, enhancing the feasibility of on-orbit refueling and servicing missions. 

**Abstract (ZH)**: 基于部分燃料箱的自主卫星对接，结合RL和MPC的方法研究 

---
# KeyWorld: Key Frame Reasoning Enables Effective and Efficient World Models 

**Title (ZH)**: KeyWorld: 关键帧推理使世界模型高效且有效 

**Authors**: Sibo Li, Qianyue Hao, Yu Shang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.21027)  

**Abstract**: Robotic world models are a promising paradigm for forecasting future environment states, yet their inference speed and the physical plausibility of generated trajectories remain critical bottlenecks, limiting their real-world applications. This stems from the redundancy of the prevailing frame-to-frame generation approach, where the model conducts costly computation on similar frames, as well as neglecting the semantic importance of key transitions. To address this inefficiency, we propose KeyWorld, a framework that improves text-conditioned robotic world models by concentrating transformers computation on a few semantic key frames while employing a lightweight convolutional model to fill the intermediate frames. Specifically, KeyWorld first identifies significant transitions by iteratively simplifying the robot's motion trajectories, obtaining the ground truth key frames. Then, a DiT model is trained to reason and generate these physically meaningful key frames from textual task descriptions. Finally, a lightweight interpolator efficiently reconstructs the full video by inpainting all intermediate frames. Evaluations on the LIBERO benchmark demonstrate that KeyWorld achieves a 5.68$\times$ acceleration compared to the frame-to-frame generation baseline, and focusing on the motion-aware key frames further contributes to the physical validity of the generated videos, especially on complex tasks. Our approach highlights a practical path toward deploying world models in real-time robotic control and other domains requiring both efficient and effective world models. Code is released at this https URL. 

**Abstract (ZH)**: 基于关键帧的机器人世界模型：一种提高推理速度和生成物理合理轨迹的方法 

---
# Multi-Robot Vision-Based Task and Motion Planning for EV Battery Disassembly and Sorting 

**Title (ZH)**: 基于视觉的多机器人任务与运动规划在电动汽车电池拆解与分类中的应用 

**Authors**: Abdelaziz Shaarawy, Cansu Erdogan, Rustam Stolkin, Alireza Rastegarpanah  

**Link**: [PDF](https://arxiv.org/pdf/2509.21020)  

**Abstract**: Electric-vehicle (EV) battery disassembly requires precise multi-robot coordination, short and reliable motions, and robust collision safety in cluttered, dynamic scenes. We propose a four-layer task-and-motion planning (TAMP) framework that couples symbolic task planning and cost- and accessibility-aware allocation with a TP-GMM-guided motion planner learned from demonstrations. Stereo vision with YOLOv8 provides real-time component localization, while OctoMap-based 3D mapping and FCL(Flexible Collision Library) checks in MoveIt unify predictive digital-twin collision checking with reactive, vision-based avoidance. Validated on two UR10e robots across cable, busbar, service plug, and three leaf-cell removals, the approach yields substantially more compact and safer motions than a default RRTConnect baseline under identical perception and task assignments: average end-effector path length drops by $-63.3\%$ and makespan by $-8.1\%$; per-arm swept volumes shrink (R1: $0.583\rightarrow0.139\,\mathrm{m}^3$; R2: $0.696\rightarrow0.252\,\mathrm{m}^3$), and mutual overlap decreases by $47\%$ ($0.064\rightarrow0.034\,\mathrm{m}^3$). These results highlight improved autonomy, precision, and safety for multi-robot EV battery disassembly in unstructured, dynamic environments. 

**Abstract (ZH)**: 电驱动车辆电池拆卸需要精确的多机器人协调、短且可靠的运动以及在拥挤和动态场景中的 robust 碰撞安全。我们提出了一种四层任务与运动规划（TAMP）框架，该框架将符号任务规划与基于 TP-GMM 的运动规划器引导的成本和可达性分配相结合，该运动规划器从演示中学习。立体视觉与 YOLOv8 提供实时部件定位，而基于 OctoMap 的三维建图和 MoveIt 中的 FCL 碰撞检查统一了预测的数字孪生碰撞检查与反应式、视觉避免。在两个 UR10e 机器人上针对电缆、汇流排、服务插头和三个电芯拆卸进行了验证，该方法在相同的感知和任务分配下相比默认的 RRTConnect 基线产生了显著更加紧凑和安全的运动：末端执行器路径长度平均减少 63.3%，周转时间减少 8.1%；每臂扫过的体积减小（R1：0.583→0.139 立方米；R2：0.696→0.252 立方米），以及相互重叠减少了 47%（0.064→0.034 立方米）。这些结果突显了在未结构化和动态环境中进行多机器人电驱动车辆电池拆卸时改进的自主性、精确性和安全性。 

---
# AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation 

**Title (ZH)**: AnywhereVLA：语言条件化探索与移动操作 

**Authors**: Konstantin Gubernatorov, Artem Voronov, Roman Voronov, Sergei Pasynkov, Stepan Perminov, Ziang Guo, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2509.21006)  

**Abstract**: We address natural language pick-and-place in unseen, unpredictable indoor environments with AnywhereVLA, a modular framework for mobile manipulation. A user text prompt serves as an entry point and is parsed into a structured task graph that conditions classical SLAM with LiDAR and cameras, metric semantic mapping, and a task-aware frontier exploration policy. An approach planner then selects visibility and reachability aware pre grasp base poses. For interaction, a compact SmolVLA manipulation head is fine tuned on platform pick and place trajectories for the SO-101 by TheRobotStudio, grounding local visual context and sub-goals into grasp and place proposals. The full system runs fully onboard on consumer-level hardware, with Jetson Orin NX for perception and VLA and an Intel NUC for SLAM, exploration, and control, sustaining real-time operation. We evaluated AnywhereVLA in a multi-room lab under static scenes and normal human motion. In this setting, the system achieves a $46\%$ overall task success rate while maintaining throughput on embedded compute. By combining a classical stack with a fine-tuned VLA manipulation, the system inherits the reliability of geometry-based navigation with the agility and task generalization of language-conditioned manipulation. 

**Abstract (ZH)**: AnywhereVLA：适用于未见和不可预测室内环境的模块化移动操作框架 

---
# BactoBot: A Low-Cost, Bacteria-Inspired Soft Underwater Robot for Marine Exploration 

**Title (ZH)**: BactoBot：一种低成本、细菌启发的软体水下机器人用于海洋探索 

**Authors**: Rubaiyat Tasnim Chowdhury, Nayan Bala, Ronojoy Roy, Tarek Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2509.20964)  

**Abstract**: Traditional rigid underwater vehicles pose risks to delicate marine ecosystems. This paper presents BactoBot, a low-cost, soft underwater robot designed for safe and gentle marine exploration. Inspired by bacterial flagellar propulsion, BactoBot features 12 flexible, silicone-based arms arranged on a 3D-printed dodecahedral frame. The design provides inherent compliance, redundancy, and the potential for omnidirectional movement. The prototype was fabricated using accessible DIY methods, including food-grade silicone molding, 3D printing, and off-the-shelf microcontrollers. Waterproofing and buoyancy calibration protocols were developed, and the robot was successfully tested in a controlled water tank, demonstrating forward motion and turning. The results validate the feasibility of replicating complex biological locomotion at low cost. The project lays a foundation for environmentally conscious robotic tools, particularly for marine science in resource-constrained settings, and identifies pathways toward autonomous operation and field deployment. 

**Abstract (ZH)**: BactoBot：一种低成本的仿生软体水下机器人及其在海洋探索中的应用 

---
# Autoregressive End-to-End Planning with Time-Invariant Spatial Alignment and Multi-Objective Policy Refinement 

**Title (ZH)**: 具有时间不变空间对齐的自回归端到端规划和多目标策略精炼 

**Authors**: Jianbo Zhao, Taiyu Ban, Xiangjie Li, Xingtai Gui, Hangning Zhou, Lei Liu, Hongwei Zhao, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20938)  

**Abstract**: The inherent sequential modeling capabilities of autoregressive models make them a formidable baseline for end-to-end planning in autonomous driving. Nevertheless, their performance is constrained by a spatio-temporal misalignment, as the planner must condition future actions on past sensory data. This creates an inconsistent worldview, limiting the upper bound of performance for an otherwise powerful approach. To address this, we propose a Time-Invariant Spatial Alignment (TISA) module that learns to project initial environmental features into a consistent ego-centric frame for each future time step, effectively correcting the agent's worldview without explicit future scene prediction. In addition, we employ a kinematic action prediction head (i.e., acceleration and yaw rate) to ensure physically feasible trajectories. Finally, we introduce a multi-objective post-training stage using Direct Preference Optimization (DPO) to move beyond pure imitation. Our approach provides targeted feedback on specific driving behaviors, offering a more fine-grained learning signal than the single, overall objective used in standard DPO. Our model achieves a state-of-the-art 89.8 PDMS on the NAVSIM dataset among autoregressive models. The video document is available at this https URL. 

**Abstract (ZH)**: 自回归模型固有的序列建模能力使它们成为自主驾驶端到端规划的有力基准。然而，它们的表现受限于时空错位，因为规划器必须基于过去的传感器数据来预测未来的动作。这导致了一种不一致的世界观，限制了这一原本强大方法的上界性能。为此，我们提出了一种时间不变空间对齐（TISA）模块，该模块学习将初始环境特征投影到每个未来的时光步的以自我为中心的框架中，从而有效地纠正代理的世界观，而无需显式预测未来的场景。此外，我们采用了动力学动作预测头部（即加速度和偏航角）以确保物理上可行的轨迹。最后，我们引入了使用直接偏好优化（DPO）的多目标后训练阶段，以超越单纯的模仿。我们的方法对特定驾驶行为提供有针对性的反馈，提供了比标准DPO中使用的单一总体目标更为精细的学习信号。我们的模型在NAVSIM数据集中实现了自回归模型中的最新89.8 PDMS性能。视频文档可在以下链接获取。 

---
# Efficient Differentiable Contact Model with Long-range Influence 

**Title (ZH)**: 长程影响的高效可微接触模型 

**Authors**: Xiaohan Ye, Kui Wu, Zherong Pan, Taku Komura  

**Link**: [PDF](https://arxiv.org/pdf/2509.20917)  

**Abstract**: With the maturation of differentiable physics, its role in various downstream applications: such as model predictive control, robotic design optimization, and neural PDE solvers, has become increasingly important. However, the derivative information provided by differentiable simulators can exhibit abrupt changes or vanish altogether, impeding the convergence of gradient-based optimizers. In this work, we demonstrate that such erratic gradient behavior is closely tied to the design of contact models. We further introduce a set of properties that a contact model must satisfy to ensure well-behaved gradient information. Lastly, we present a practical contact model for differentiable rigid-body simulators that satisfies all of these properties while maintaining computational efficiency. Our experiments show that, even from simple initializations, our contact model can discover complex, contact-rich control signals, enabling the successful execution of a range of downstream locomotion and manipulation tasks. 

**Abstract (ZH)**: 随着可微物理的发展，在不同下游应用中的作用：例如模型预测控制、机器人设计优化和神经偏微分方程求解器等方面的作用日益重要。然而，可微模拟器提供的导数信息可能会出现突然变化甚至消失，阻碍梯度基优化器的收敛。本文表明，这种不规则的梯度行为与接触模型的设计密切相关。我们进一步引入了一组确保良好行为梯度信息的接触模型性质。最后，我们提出了一种适用于可微刚体模拟器的实用接触模型，该模型满足所有这些性质同时保持计算效率。我们的实验结果表明，即使从简单的初始化开始，我们的接触模型也能发现复杂的、接触丰富的控制信号，从而使多种下游运动和操作任务的成功执行成为可能。 

---
# MTRDrive: Memory-Tool Synergistic Reasoning for Robust Autonomous Driving in Corner Cases 

**Title (ZH)**: MTRDrive: 记忆-工具协同推理在corner case中实现稳健自主驾驶 

**Authors**: Ziang Luo, Kangan Qian, Jiahua Wang, Yuechen Luo, Jinyu Miao, Zheng Fu, Yunlong Wang, Sicong Jiang, Zilin Huang, Yifei Hu, Yuhao Yang, Hao Ye, Mengmeng Yang, Xiaojian Dong, Kun Jiang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20843)  

**Abstract**: Vision-Language Models(VLMs) have demonstrated significant potential for end-to-end autonomous driving, yet a substantial gap remains between their current capabilities and the reliability necessary for real-world deployment. A critical challenge is their fragility, characterized by hallucinations and poor generalization in out-of-distribution (OOD) scenarios. To bridge this gap, we introduce MTRDrive, a novel framework that integrates procedural driving experiences with a dynamic toolkit to enhance generalization and proactive decision-making.
MTRDrive addresses these limitations through a closed-loop system that combines a memory-based experience retrieval mechanism with dynamic toolkits. This synergy enables the model to interact more effectively with its environment, improving both reasoning and decision-making capabilities with the help of our memory-tool synergistic reasoning. Additionally, we introduce a new benchmark based on complex Roadwork construction scenarios to rigorously evaluate zero-shot generalization.
Extensive experiments demonstrate the superior effectiveness of our approach. On the public NAVSIM benchmark, our 3B-parameter MTRDrive model achieves an exceptional PDMS of 88.3 without chain-of-thought and sets a state-of-the-art performance bar on high-level planning, with a driving metric score of 79.8\% and a planning accuracy of 82.6\%. Rigorous zero-shot evaluation on the new Roadwork-VLM benchmark shows a strong ability to reason robustly in unseen scenarios, achieving a driving metric score of 80.2\%. These results highlight MTRDrive's potential to advance autonomous driving toward safer and more reliable systems. 

**Abstract (ZH)**: Vision-Language模型在端到端自动驾驶中的潜在应用及其与实际部署所需的可靠性的差距：MTRDrive框架的构建与评估 

---
# ImaginationPolicy: Towards Generalizable, Precise and Reliable End-to-End Policy for Robotic Manipulation 

**Title (ZH)**: 想象策略：迈向通用、精确且可靠的端到端机器人 manipulation 策略 

**Authors**: Dekun Lu, Wei Gao, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.20841)  

**Abstract**: End-to-end robot manipulation policies offer significant potential for enabling embodied agents to understand and interact with the world. Unlike traditional modular pipelines, end-to-end learning mitigates key limitations such as information loss between modules and feature misalignment caused by isolated optimization targets. Despite these advantages, existing end-to-end neural networks for robotic manipulation--including those based on large VLM/VLA models--remain insufficiently performant for large-scale practical deployment. In this paper, we take a step towards an end-to-end manipulation policy that is generalizable, accurate and reliable. To achieve this goal, we propose a novel Chain of Moving Oriented Keypoints (CoMOK) formulation for robotic manipulation. Our formulation is used as the action representation of a neural policy, which can be trained in an end-to-end fashion. Such an action representation is general, as it extends the standard end-effector pose action representation and supports a diverse set of manipulation tasks in a unified manner. The oriented keypoint in our method enables natural generalization to objects with different shapes and sizes, while achieving sub-centimeter accuracy. Moreover, our formulation can easily handle multi-stage tasks, multi-modal robot behaviors, and deformable objects. Extensive simulated and hardware experiments demonstrate the effectiveness of our method. 

**Abstract (ZH)**: 端到端机器人操作策略在使具身代理理解和与世界交互方面具有巨大潜力。与传统的模块化流水线不同，端到端学习减轻了模块间信息丢失和由于孤立优化目标引起的特征对齐不良等关键限制。尽管具有这些优势，目前用于机器人操作的端到端神经网络——包括基于大VLM/VLA模型的网络——在大规模实际部署中仍然表现不足。本文朝着一个通用、准确且可靠的端到端操作策略迈进。为实现这一目标，我们提出了一种新的移动定向关键点链（CoMOK）方法论，用于机器人操作。该方法论作为神经策略的动作表示，可以端到端训练。这种动作表示是通用的，因为它扩展了标准末端执行器姿态动作表示，并以统一的方式支持各种操作任务。我们的方法中的定向关键点能够自然地泛化到不同形状和大小的对象，同时实现亚厘米级的准确性。此外，我们的方法可以轻松处理多阶段任务、多模态机器人行为以及可变形物体。大量的模拟和硬件实验证明了该方法的有效性。 

---
# SemSight: Probabilistic Bird's-Eye-View Prediction of Multi-Level Scene Semantics for Navigation 

**Title (ZH)**: SemSight：多层次场景语义的概率 bird's-eye-view 预测导航 

**Authors**: Jiaxuan He, Jiamei Ren, Chongshang Yan, Wenjie Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.20839)  

**Abstract**: In target-driven navigation and autonomous exploration, reasonable prediction of unknown regions is crucial for efficient navigation and environment understanding. Existing methods mostly focus on single objects or geometric occupancy maps, lacking the ability to model room-level semantic structures. We propose SemSight, a probabilistic bird's-eye-view prediction model for multi-level scene semantics. The model jointly infers structural layouts, global scene context, and target area distributions, completing semantic maps of unexplored areas while estimating probability maps for target categories. To train SemSight, we simulate frontier-driven exploration on 2,000 indoor layout graphs, constructing a diverse dataset of 40,000 sequential egocentric observations paired with complete semantic maps. We adopt an encoder-decoder network as the core architecture and introduce a mask-constrained supervision strategy. This strategy applies a binary mask of unexplored areas so that supervision focuses only on unknown regions, forcing the model to infer semantic structures from the observed context. Experimental results show that SemSight improves prediction performance for key functional categories in unexplored regions and outperforms non-mask-supervised approaches on metrics such as Structural Consistency (SC) and Region Recognition Accuracy (PA). It also enhances navigation efficiency in closed-loop simulations, reducing the number of search steps when guiding robots toward target areas. 

**Abstract (ZH)**: 基于目标导向的导航和自主探索中，合理预测未知区域对于高效导航和环境理解至关重要。现有方法主要关注单个物体或几何占用地图，缺乏建模房间级语义结构的能力。我们提出SemSight，一种多级场景语义的概率bird's-eye-view预测模型。该模型联合推断结构布局、全局场景上下文和目标区域分布，构建未探索区域的语义地图并估计目标类别概率图。为了训练SemSight，我们在2,000个室内布局图上模拟目标驱动的探索，构建了一个包含40,000个序列性第一人称观察及其完整语义地图的多样化数据集。我们采用编码-解码网络作为核心架构，并引入一种基于掩码的监督策略。该策略使用未探索区域的二进制掩码，使得监督仅专注于未知区域，迫使模型从观察上下文中推断语义结构。实验结果表明，SemSight在未探索区域内关键功能类别的预测性能上有所改进，并在结构一致性（SC）和区域识别准确率（PA）等指标上优于未采用掩码监督的方法。此外，SemSight在闭环模拟中提高了导航效率，减少了引导机器人前往目标区域所需的搜索步骤。 

---
# Leveraging Temporally Extended Behavior Sharing for Multi-task Reinforcement Learning 

**Title (ZH)**: 利用扩展时间行为共享进行多任务强化学习 

**Authors**: Gawon Lee, Daesol Cho, H. Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.20766)  

**Abstract**: Multi-task reinforcement learning (MTRL) offers a promising approach to improve sample efficiency and generalization by training agents across multiple tasks, enabling knowledge sharing between them. However, applying MTRL to robotics remains challenging due to the high cost of collecting diverse task data. To address this, we propose MT-Lévy, a novel exploration strategy that enhances sample efficiency in MTRL environments by combining behavior sharing across tasks with temporally extended exploration inspired by Lévy flight. MT-Lévy leverages policies trained on related tasks to guide exploration towards key states, while dynamically adjusting exploration levels based on task success ratios. This approach enables more efficient state-space coverage, even in complex robotics environments. Empirical results demonstrate that MT-Lévy significantly improves exploration and sample efficiency, supported by quantitative and qualitative analyses. Ablation studies further highlight the contribution of each component, showing that combining behavior sharing with adaptive exploration strategies can significantly improve the practicality of MTRL in robotics applications. 

**Abstract (ZH)**: 多任务强化学习（MTRL）提供了一种通过跨多个任务训练代理来提高样本效率和泛化能力的方法，从而在它们之间共享知识。然而，将MTRL应用到机器人领域仍然具有挑战性，因为收集多样化任务数据的成本很高。为了解决这个问题，我们提出了一种新颖的探索策略MT-Lévy，它通过在任务间共享行为并受Lévy飞行启发进行时序扩展探索来增强MTRL环境中的样本效率。MT-Lévy 利用相关任务训练的策略引导探索指向关键状态，并根据任务成功率动态调整探索水平。这种方法能够在复杂机器人环境中更有效地覆盖状态空间。实验结果表明，MT-Lévy 显著提高了探索和样本效率，并通过定量和定性分析予以验证。消融研究进一步强调了各组成部分的贡献，显示将行为共享与自适应探索策略相结合可以显著提高MTRL在机器人应用中的实用性。 

---
# MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM 

**Title (ZH)**: MASt3R-Fusion：结合前馈视觉模型与IMU、GNSS的高功能SLAMcomings 

**Authors**: Yuxuan Zhou, Xingxing Li, Shengyu Li, Zhuohao Yan, Chunxi Xia, Shaoquan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20757)  

**Abstract**: Visual SLAM is a cornerstone technique in robotics, autonomous driving and extended reality (XR), yet classical systems often struggle with low-texture environments, scale ambiguity, and degraded performance under challenging visual conditions. Recent advancements in feed-forward neural network-based pointmap regression have demonstrated the potential to recover high-fidelity 3D scene geometry directly from images, leveraging learned spatial priors to overcome limitations of traditional multi-view geometry methods. However, the widely validated advantages of probabilistic multi-sensor information fusion are often discarded in these pipelines. In this work, we propose MASt3R-Fusion,a multi-sensor-assisted visual SLAM framework that tightly integrates feed-forward pointmap regression with complementary sensor information, including inertial measurements and GNSS data. The system introduces Sim(3)-based visualalignment constraints (in the Hessian form) into a universal metric-scale SE(3) factor graph for effective information fusion. A hierarchical factor graph design is developed, which allows both real-time sliding-window optimization and global optimization with aggressive loop closures, enabling real-time pose tracking, metric-scale structure perception and globally consistent mapping. We evaluate our approach on both public benchmarks and self-collected datasets, demonstrating substantial improvements in accuracy and robustness over existing visual-centered multi-sensor SLAM systems. The code will be released open-source to support reproducibility and further research (this https URL). 

**Abstract (ZH)**: 多传感器辅助视觉SLAM框架MASt3R-Fusion：结合前向点云回归与互补传感器信息进行有效的信息融合 

---
# SLAM-Free Visual Navigation with Hierarchical Vision-Language Perception and Coarse-to-Fine Semantic Topological Planning 

**Title (ZH)**: 基于分层视觉-语言感知和从粗到细语义拓扑规划的无SLAM视觉导航 

**Authors**: Guoyang Zhao, Yudong Li, Weiqing Qi, Kai Zhang, Bonan Liu, Kai Chen, Haoang Li, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.20739)  

**Abstract**: Conventional SLAM pipelines for legged robot navigation are fragile under rapid motion, calibration demands, and sensor drift, while offering limited semantic reasoning for task-driven exploration. To deal with these issues, we propose a vision-only, SLAM-free navigation framework that replaces dense geometry with semantic reasoning and lightweight topological representations. A hierarchical vision-language perception module fuses scene-level context with object-level cues for robust semantic inference. And a semantic-probabilistic topological map supports coarse-to-fine planning: LLM-based global reasoning for subgoal selection and vision-based local planning for obstacle avoidance. Integrated with reinforcement-learning locomotion controllers, the framework is deployable across diverse legged robot platforms. Experiments in simulation and real-world settings demonstrate consistent improvements in semantic accuracy, planning quality, and navigation success, while ablation studies further showcase the necessity of both hierarchical perception and fine local planning. This work introduces a new paradigm for SLAM-free, vision-language-driven navigation, shifting robotic exploration from geometry-centric mapping to semantics-driven decision making. 

**Abstract (ZH)**: 基于视觉的无需SLAM的任务驱动探索导航框架：从几何中心映射到语义驱动决策的新范式 

---
# RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking 

**Title (ZH)**: RobotDancing: 剩余动作强化学习 enables 稳健的长期人体运动跟踪 

**Authors**: Zhenguo Sun, Yibo Peng, Yuan Meng, Xukun Li, Bo-Sheng Huang, Zhenshan Bing, Xinlong Wang, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.20717)  

**Abstract**: Long-horizon, high-dynamic motion tracking on humanoids remains brittle because absolute joint commands cannot compensate model-plant mismatch, leading to error accumulation. We propose RobotDancing, a simple, scalable framework that predicts residual joint targets to explicitly correct dynamics discrepancies. The pipeline is end-to-end--training, sim-to-sim validation, and zero-shot sim-to-real--and uses a single-stage reinforcement learning (RL) setup with a unified observation, reward, and hyperparameter configuration. We evaluate primarily on Unitree G1 with retargeted LAFAN1 dance sequences and validate transfer on H1/H1-2. RobotDancing can track multi-minute, high-energy behaviors (jumps, spins, cartwheels) and deploys zero-shot to hardware with high motion tracking quality. 

**Abstract (ZH)**: 长周期、高动态人类机器人运动追踪仍因模型-系统不匹配导致绝对关节命令失效，引起错误累积。我们提出RobotDancing，一个简单可扩展的框架，预测剩余关节目标以明确修正动力学差异。该框架为端到端设计，包括训练、模拟验证和零样本模拟到现实世界的应用，并采用单一阶段的强化学习设置，具有统一的观察、奖励和超参数配置。我们主要在Unitree G1上使用重定向的LAFAN1舞蹈序列进行评估，并在H1/H1-2上验证了迁移性能。RobotDancing可以追踪多分钟、高能量行为（跳跃、旋转、侧手翻），并以高质量的运动追踪直接部署到硬件上。 

---
# Digital Twin-Guided Robot Path Planning: A Beta-Bernoulli Fusion with Large Language Model as a Sensor 

**Title (ZH)**: 基于数字孪生引导的机器人路径规划：Beta-Bernoulli融合与大型语言模型作为传感器 

**Authors**: Mani Amani, Reza Akhavian  

**Link**: [PDF](https://arxiv.org/pdf/2509.20709)  

**Abstract**: Integrating natural language (NL) prompts into robotic mission planning has attracted significant interest in recent years. In the construction domain, Building Information Models (BIM) encapsulate rich NL descriptions of the environment. We present a novel framework that fuses NL directives with BIM-derived semantic maps via a Beta-Bernoulli Bayesian fusion by interpreting the LLM as a sensor: each obstacle's design-time repulsive coefficient is treated as a Beta(alpha, beta) random variable and LLM-returned danger scores are incorporated as pseudo-counts to update alpha and beta. The resulting posterior mean yields a continuous, context-aware repulsive gain that augments a Euclidean-distance-based potential field for cost heuristics. By adjusting gains based on sentiment and context inferred from user prompts, our method guides robots along safer, more context-aware paths. This provides a numerically stable method that can chain multiple natural commands and prompts from construction workers and foreman to enable planning while giving flexibility to be integrated in any learned or classical AI framework. Simulation results demonstrate that this Beta-Bernoulli fusion yields both qualitative and quantitative improvements in path robustness and validity. 

**Abstract (ZH)**: 将自然语言（NL）提示融入到机器人任务规划中近年来引起了广泛关注。在建筑领域，建筑信息模型（BIM）封装了环境的丰富自然语言描述。我们提出了一种新颖的框架，通过Beta-Bernoulli贝叶斯融合将自然语言指令与由BIM衍生的语义地图相结合：将每个障碍物的设计时排斥系数视为一个Beta(alpha, beta)随机变量，将从LLM返回的危险分数作为伪计数纳入其中以更新alpha和beta。所得到的后验均值产生了一个连续的、基于上下文的排斥增益，以增强基于欧几里得距离的势场的成本启发式。根据用户提示推断出的情绪和上下文调整增益，我们的方法引导机器人沿着更加安全和上下文感知的路径。这种方法提供了一种数值稳定的手段，可以串联来自建筑工人和主管的多个自然命令和提示，以实现规划的灵活性，并能够与任何学习或经典AI框架集成。仿真结果表明，这种Beta-Bernoulli融合在路径鲁棒性和有效性方面均取得了质和量的改进。 

---
# Building Information Models to Robot-Ready Site Digital Twins (BIM2RDT): An Agentic AI Safety-First Framework 

**Title (ZH)**: Building Information Models for Robot-Ready Site Digital Twins (BIM2RDT): 一种以代理为中心的AI安全优先框架 

**Authors**: Reza Akhavian, Mani Amani, Johannes Mootz, Robert Ashe, Behrad Beheshti  

**Link**: [PDF](https://arxiv.org/pdf/2509.20705)  

**Abstract**: The adoption of cyber-physical systems and jobsite intelligence that connects design models, real-time site sensing, and autonomous field operations can dramatically enhance digital management in the construction industry. This paper introduces BIM2RDT (Building Information Models to Robot-Ready Site Digital Twins), an agentic artificial intelligence (AI) framework designed to transform static Building Information Modeling (BIM) into dynamic, robot-ready digital twins (DTs) that prioritize safety during execution. The framework bridges the gap between pre-existing BIM data and real-time site conditions by integrating three key data streams: geometric and semantic information from BIM models, activity data from IoT sensor networks, and visual-spatial data collected by robots during site traversal. The methodology introduces Semantic-Gravity ICP (SG-ICP), a point cloud registration algorithm that leverages large language model (LLM) reasoning. Unlike traditional methods, SG-ICP utilizes an LLM to infer object-specific, plausible orientation priors based on BIM semantics, improving alignment accuracy by avoiding convergence on local minima. This creates a feedback loop where robot-collected data updates the DT, which in turn optimizes paths for missions. The framework employs YOLOE object detection and Shi-Tomasi corner detection to identify and track construction elements while using BIM geometry as a priori maps. The framework also integrates real-time Hand-Arm Vibration (HAV) monitoring, mapping sensor-detected safety events to the digital twin using IFC standards for intervention. Experiments demonstrate SG-ICP's superiority over standard ICP, achieving RMSE reductions of 64.3%--88.3% in alignment across scenarios with occluded features, ensuring plausible orientations. HAV integration triggers warnings upon exceeding exposure limits, enhancing compliance with ISO 5349-1. 

**Abstract (ZH)**: 基于BIM到机器人就绪工地数字孪生的代理人工智能框架：BIM2RDT 

---
# Joint Flow Trajectory Optimization For Feasible Robot Motion Generation from Video Demonstrations 

**Title (ZH)**: 基于视频示范的可行机器人运动生成的联合流轨迹优化 

**Authors**: Xiaoxiang Dong, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20703)  

**Abstract**: Learning from human video demonstrations offers a scalable alternative to teleoperation or kinesthetic teaching, but poses challenges for robot manipulators due to embodiment differences and joint feasibility constraints. We address this problem by proposing the Joint Flow Trajectory Optimization (JFTO) framework for grasp pose generation and object trajectory imitation under the video-based Learning-from-Demonstration (LfD) paradigm. Rather than directly imitating human hand motions, our method treats demonstrations as object-centric guides, balancing three objectives: (i) selecting a feasible grasp pose, (ii) generating object trajectories consistent with demonstrated motions, and (iii) ensuring collision-free execution within robot kinematics. To capture the multimodal nature of demonstrations, we extend flow matching to $\SE(3)$ for probabilistic modeling of object trajectories, enabling density-aware imitation that avoids mode collapse. The resulting optimization integrates grasp similarity, trajectory likelihood, and collision penalties into a unified differentiable objective. We validate our approach in both simulation and real-world experiments across diverse real-world manipulation tasks. 

**Abstract (ZH)**: 基于视频演示的学习方法为操纵器提供了规模化替代遥操作或力觉教学的方案，但由于实体差异和关节可行性约束，给机器人 manipulator 带来了挑战。我们通过提出基于视频演示学习（LfD） paradigm 下的 Joint Flow Trajectory Optimization (JFTO) 框架来解决这一问题，用于抓取姿态生成和物体轨迹模仿。我们的方法将演示视为以物体为中心的指南，平衡以下三个目标：(i) 选择可行的抓取姿态，(ii) 生成与演示动作一致的物体轨迹，(iii) 确保在机器人运动学约束下无碰撞执行。为了捕捉演示的多模态特性，我们将流匹配扩展到 $\SE(3)$，进行物体轨迹的概率建模，从而实现密度感知的模仿，避免模式崩塌。最终优化整合了抓取相似性、轨迹可能性及碰撞惩罚，形成统一可微的目标函数。我们在多种现实世界的操纵任务的仿真和真实实验中验证了该方法。 

---
# RuN: Residual Policy for Natural Humanoid Locomotion 

**Title (ZH)**: RuN: 用于自然人形运动的残差策略 

**Authors**: Qingpeng Li, Chengrui Zhu, Yanming Wu, Xin Yuan, Zhen Zhang, Jian Yang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20696)  

**Abstract**: Enabling humanoid robots to achieve natural and dynamic locomotion across a wide range of speeds, including smooth transitions from walking to running, presents a significant challenge. Existing deep reinforcement learning methods typically require the policy to directly track a reference motion, forcing a single policy to simultaneously learn motion imitation, velocity tracking, and stability maintenance. To address this, we introduce RuN, a novel decoupled residual learning framework. RuN decomposes the control task by pairing a pre-trained Conditional Motion Generator, which provides a kinematically natural motion prior, with a reinforcement learning policy that learns a lightweight residual correction to handle dynamical interactions. Experiments in simulation and reality on the Unitree G1 humanoid robot demonstrate that RuN achieves stable, natural gaits and smooth walk-run transitions across a broad velocity range (0-2.5 m/s), outperforming state-of-the-art methods in both training efficiency and final performance. 

**Abstract (ZH)**: 使仿人机器人在广泛的速度范围内实现自然且动态的运动，包括从行走平滑过渡到奔跑，是一项重大挑战。现有深度强化学习方法通常要求策略直接跟踪参考运动，迫使单一策略同时学习运动模仿、速度跟踪和稳定性维护。为了解决这一问题，我们引入了 RuN，这是一种新型的解耦残差学习框架。RuN 通过将一个预训练的条件运动生成器与一个学习轻量级动态修正的强化学习策略配对，来分解控制任务，其中条件运动生成器提供一种动力学自然的运动先验。实验在 Unitree G1 仿人机器人上的模拟和现实环境中表明，RuN 在广泛的速度假区间（0-2.5 m/s）实现了稳定且自然的步伐，并且平滑的走跑过渡，其在训练效率和最终性能方面均优于现有最先进的方法。 

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
# Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation 

**Title (ZH)**: 从单张图像高效构建隐式表面模型以生成运动 

**Authors**: Wei-Teng Chu, Tianyi Zhang, Matthew Johnson-Roberson, Weiming Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20681)  

**Abstract**: Implicit representations have been widely applied in robotics for obstacle avoidance and path planning. In this paper, we explore the problem of constructing an implicit distance representation from a single image. Past methods for implicit surface reconstruction, such as \emph{NeuS} and its variants generally require a large set of multi-view images as input, and require long training times. In this work, we propose Fast Image-to-Neural Surface (FINS), a lightweight framework that can reconstruct high-fidelity surfaces and SDF fields based on a single or a small set of images. FINS integrates a multi-resolution hash grid encoder with lightweight geometry and color heads, making the training via an approximate second-order optimizer highly efficient and capable of converging within a few seconds. Additionally, we achieve the construction of a neural surface requiring only a single RGB image, by leveraging pre-trained foundation models to estimate the geometry inherent in the image. Our experiments demonstrate that under the same conditions, our method outperforms state-of-the-art baselines in both convergence speed and accuracy on surface reconstruction and SDF field estimation. Moreover, we demonstrate the applicability of FINS for robot surface following tasks and show its scalability to a variety of benchmark datasets. 

**Abstract (ZH)**: 基于单张图像构建隐式距离表示的快速图像到神经表面框架 

---
# Equi-RO: A 4D mmWave Radar Odometry via Equivariant Networks 

**Title (ZH)**: Equi-RO: 通过等变网络的4D毫米波雷达里程表测量 

**Authors**: Zeyu Han, Shuocheng Yang, Minghan Zhu, Fang Zhang, Shaobing Xu, Maani Ghaffari, Jianqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20674)  

**Abstract**: Autonomous vehicles and robots rely on accurate odometry estimation in GPS-denied environments. While LiDARs and cameras struggle under extreme weather, 4D mmWave radar emerges as a robust alternative with all-weather operability and velocity measurement. In this paper, we introduce Equi-RO, an equivariant network-based framework for 4D radar odometry. Our algorithm pre-processes Doppler velocity into invariant node and edge features in the graph, and employs separate networks for equivariant and invariant feature processing. A graph-based architecture enhances feature aggregation in sparse radar data, improving inter-frame correspondence. Experiments on the open-source dataset and self-collected dataset show Equi-RO outperforms state-of-the-art algorithms in accuracy and robustness. Overall, our method achieves 10.7% and 20.0% relative improvements in translation and rotation accuracy, respectively, compared to the best baseline on the open-source dataset. 

**Abstract (ZH)**: 自主驾驶车辆和机器人在GPS受限环境中依赖于准确的速度估计。尽管LiDAR和摄像头在极端天气下表现不佳，4D毫米波雷达由于其全天候操作能力和速度测量能力成为可靠的选择。本文 introduces Equi-RO，一种基于等变网络的4D雷达速度估计框架。该算法将雷达速度预处理为图中的不变节点和边特征，并使用独立的网络对等变和不变特征进行处理。基于图的架构在稀疏雷达数据中增强了特征聚合，提高了帧间对应关系。在开源数据集和自采集数据集上的实验表明，Equi-RO 在准确性和鲁棒性方面优于现有最佳算法。总体而言，我们的方法在开源数据集的最佳基准上分别实现了10.7%和20.0%的平移和旋转精度改进。 

---
# EEG-Driven AR-Robot System for Zero-Touch Grasping Manipulation 

**Title (ZH)**: 基于EEG的AR机器人系统实现零触觉抓取操作 

**Authors**: Junzhe Wang, Jiarui Xie, Pengfei Hao, Zheng Li, Yi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.20656)  

**Abstract**: Reliable brain-computer interface (BCI) control of robots provides an intuitive and accessible means of human-robot interaction, particularly valuable for individuals with motor impairments. However, existing BCI-Robot systems face major limitations: electroencephalography (EEG) signals are noisy and unstable, target selection is often predefined and inflexible, and most studies remain restricted to simulation without closed-loop validation. These issues hinder real-world deployment in assistive scenarios. To address them, we propose a closed-loop BCI-AR-Robot system that integrates motor imagery (MI)-based EEG decoding, augmented reality (AR) neurofeedback, and robotic grasping for zero-touch operation. A 14-channel EEG headset enabled individualized MI calibration, a smartphone-based AR interface supported multi-target navigation with direction-congruent feedback to enhance stability, and the robotic arm combined decision outputs with vision-based pose estimation for autonomous grasping. Experiments are conducted to validate the framework: MI training achieved 93.1 percent accuracy with an average information transfer rate (ITR) of 14.8 bit/min; AR neurofeedback significantly improved sustained control (SCI = 0.210) and achieved the highest ITR (21.3 bit/min) compared with static, sham, and no-AR baselines; and closed-loop grasping achieved a 97.2 percent success rate with good efficiency and strong user-reported control. These results show that AR feedback substantially stabilizes EEG-based control and that the proposed framework enables robust zero-touch grasping, advancing assistive robotic applications and future modes of human-robot interaction. 

**Abstract (ZH)**: 可靠的大脑-计算机接口（BCI）控制的机器人提供了一种直观且易于使用的交互方式，尤其对于运动障碍患者非常有价值。然而，现有的BCI-机器人系统面临重大限制：脑电图（EEG）信号噪声大且不稳定，目标选择通常预定义且灵活性有限，多数研究仍局限于模拟而未能进行闭环验证。这些问题阻碍了在辅助场景中的实际部署。为解决这些问题，我们提出了一种闭环BCI-AR-机器人系统，该系统结合了基于 motor imagery（MI）的 EEG 解码、增强现实（AR）神经反馈以及机械臂抓取操作，实现了零接触操作。14通道EEG头盔实现了个性化的MI校准，基于智能手机的AR界面支持多目标导航并提供了方向一致的反馈以增强稳定性，机械臂结合决策输出与基于视觉的姿态估计实现自主抓取。进行了实验验证该框架：MI训练的准确率达到93.1%，平均信息传输速率为14.8 bit/min；AR神经反馈显著提高了持续控制能力（SCI = 0.210），并实现了最高的信息传输速率（21.3 bit/min），优于静态、假对照和无AR基线；闭环抓取成功率达到97.2%，具有良好的效率和强大的用户报告的控制能力。这些结果表明，AR反馈显著稳定了基于EEG的控制，并且所提出框架实现了稳健的零接触抓取，推动了辅助机器人应用和未来的人机交互模式的发展。 

---
# Cyber Racing Coach: A Haptic Shared Control Framework for Teaching Advanced Driving Skills 

**Title (ZH)**: 网络赛车教练：一种用于教学高级驾驶技巧的触觉协同控制框架 

**Authors**: Congkai Shen, Siyuan Yu, Yifan Weng, Haoran Ma, Chen Li, Hiroshi Yasuda, James Dallas, Michael Thompson, John Subosits, Tulga Ersal  

**Link**: [PDF](https://arxiv.org/pdf/2509.20653)  

**Abstract**: This study introduces a haptic shared control framework designed to teach human drivers advanced driving skills. In this context, shared control refers to a driving mode where the human driver collaborates with an autonomous driving system to control the steering of a vehicle simultaneously. Advanced driving skills are those necessary to safely push the vehicle to its handling limits in high-performance driving such as racing and emergency obstacle avoidance. Previous research has demonstrated the performance and safety benefits of shared control schemes using both subjective and objective evaluations. However, these schemes have not been assessed for their impact on skill acquisition on complex and demanding tasks. Prior research on long-term skill acquisition either applies haptic shared control to simple tasks or employs other feedback methods like visual and auditory aids. To bridge this gap, this study creates a cyber racing coach framework based on the haptic shared control paradigm and evaluates its performance in helping human drivers acquire high-performance driving skills. The framework introduces (1) an autonomous driving system that is capable of cooperating with humans in a highly performant driving scenario; and (2) a haptic shared control mechanism along with a fading scheme to gradually reduce the steering assistance from autonomy based on the human driver's performance during training. Two benchmarks are considered: self-learning (no assistance) and full assistance during training. Results from a human subject study indicate that the proposed framework helps human drivers develop superior racing skills compared to the benchmarks, resulting in better performance and consistency. 

**Abstract (ZH)**: 基于触觉共享控制的高级驾驶技能教学框架研究 

---
# Suction Leap-Hand: Suction Cups on a Multi-fingered Hand Enable Embodied Dexterity and In-Hand Teleoperation 

**Title (ZH)**: 吸力Leap-Hand：多指手上的吸盘使机器人具备实体灵活性和手持远程操作能力 

**Authors**: Sun Zhaole, Xiaofeng Mao, Jihong Zhu, Yuanlong Zhang, Robert B. Fisher  

**Link**: [PDF](https://arxiv.org/pdf/2509.20646)  

**Abstract**: Dexterous in-hand manipulation remains a foundational challenge in robotics, with progress often constrained by the prevailing paradigm of imitating the human hand. This anthropomorphic approach creates two critical barriers: 1) it limits robotic capabilities to tasks humans can already perform, and 2) it makes data collection for learning-based methods exceedingly difficult. Both challenges are caused by traditional force-closure which requires coordinating complex, multi-point contacts based on friction, normal force, and gravity to grasp an object. This makes teleoperated demonstrations unstable and amplifies the sim-to-real gap for reinforcement learning. In this work, we propose a paradigm shift: moving away from replicating human mechanics toward the design of novel robotic embodiments. We introduce the \textbf{S}uction \textbf{Leap}-Hand (SLeap Hand), a multi-fingered hand featuring integrated fingertip suction cups that realize a new form of suction-enabled dexterity. By replacing complex force-closure grasps with stable, single-point adhesion, our design fundamentally simplifies in-hand teleoperation and facilitates the collection of high-quality demonstration data. More importantly, this suction-based embodiment unlocks a new class of dexterous skills that are difficult or even impossible for the human hand, such as one-handed paper cutting and in-hand writing. Our work demonstrates that by moving beyond anthropomorphic constraints, novel embodiments can not only lower the barrier for collecting robust manipulation data but also enable the stable, single-handed completion of tasks that would typically require two human hands. Our webpage is this https URL. 

**Abstract (ZH)**: 在手灵巧操作仍然是机器人技术中的一个基础挑战，进展往往受限于模仿人类手部的主流范式。这一类比人类的手部方法创造了两个关键障碍：1）它限制了机器人的能力，仅限于人类已经能完成的任务；2）它使基于学习的方法的数据采集极其困难。这两个挑战都是由传统的力闭合引起的，传统力闭合要求基于摩擦、法向力和重力协调复杂的多点接触才能抓取物体。这使得遥控演示变得不稳定，并放大了模拟到现实的差距。在本文中，我们提出了范式的转变：从复制人类力学转向设计新颖的机器人实体。我们介绍了Suction Leap-Hand（SLeap Hand），这是一种多功能的手部，配备了内置的指尖吸盘，实现了新的吸盘助力灵巧操作形式。通过用稳定的一点粘附替代复杂的力闭合抓取，我们设计从根本上简化了在手遥控操纵，并促进了高质量演示数据的收集。更重要的是，基于吸盘的设计解锁了一类新的灵巧技能，这些技能对于人类手部来说是难以实现甚至是不可能实现的，例如单手剪纸和在手写字。我们的研究显示，通过超越类人学限制，新颖的实体不仅可以降低收集稳健操作数据的门槛，还能使单手完成通常需要两只手来完成的任务变得稳定。我们的网页地址是：this https URL。 

---
# Learning Terrain-Specialized Policies for Adaptive Locomotion in Challenging Environments 

**Title (ZH)**: 学习适应挑战环境的地形专业化移动策略 

**Authors**: Matheus P. Angarola, Francisco Affonso, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2509.20635)  

**Abstract**: Legged robots must exhibit robust and agile locomotion across diverse, unstructured terrains, a challenge exacerbated under blind locomotion settings where terrain information is unavailable. This work introduces a hierarchical reinforcement learning framework that leverages terrain-specialized policies and curriculum learning to enhance agility and tracking performance in complex environments. We validated our method on simulation, where our approach outperforms a generalist policy by up to 16% in success rate and achieves lower tracking errors as the velocity target increases, particularly on low-friction and discontinuous terrains, demonstrating superior adaptability and robustness across mixed-terrain scenarios. 

**Abstract (ZH)**: 腿足机器人必须在多样且未结构化的地形上表现出稳健和敏捷的运动能力，而在地形信息不可用的盲运动设置下，这一挑战更为严峻。本文提出了一种层次化强化学习框架，该框架利用地形专业化策略和课程学习来增强复杂环境中敏捷性和跟踪性能。我们在模拟中验证了该方法，结果显示，与通用策略相比，我们的方法在成功率上高出多达16%，并且随着目标速度的增加，跟踪误差更低，特别是在低摩擦和不连续地形上，表明该方法在混合地形场景中具有更优的适应性和鲁棒性。 

---
# Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation 

**Title (ZH)**: 潜在激活编辑：学习政策的推理时 refinement 以实现更安全的多机器人导航 

**Authors**: Satyajeet Das, Darren Chiu, Zhehui Huang, Lars Lindemann, Gaurav S. Sukhatme  

**Link**: [PDF](https://arxiv.org/pdf/2509.20623)  

**Abstract**: Reinforcement learning has enabled significant progress in complex domains such as coordinating and navigating multiple quadrotors. However, even well-trained policies remain vulnerable to collisions in obstacle-rich environments. Addressing these infrequent but critical safety failures through retraining or fine-tuning is costly and risks degrading previously learned skills. Inspired by activation steering in large language models and latent editing in computer vision, we introduce a framework for inference-time Latent Activation Editing (LAE) that refines the behavior of pre-trained policies without modifying their weights or architecture. The framework operates in two stages: (i) an online classifier monitors intermediate activations to detect states associated with undesired behaviors, and (ii) an activation editing module that selectively modifies flagged activations to shift the policy towards safer regimes. In this work, we focus on improving safety in multi-quadrotor navigation. We hypothesize that amplifying a policy's internal perception of risk can induce safer behaviors. We instantiate this idea through a latent collision world model trained to predict future pre-collision activations, thereby prompting earlier and more cautious avoidance responses. Extensive simulations and real-world Crazyflie experiments demonstrate that LAE achieves statistically significant reduction in collisions (nearly 90% fewer cumulative collisions compared to the unedited baseline) and substantially increases the fraction of collision-free trajectories, while preserving task completion. More broadly, our results establish LAE as a lightweight paradigm, feasible on resource-constrained hardware, for post-deployment refinement of learned robot policies. 

**Abstract (ZH)**: 强化学习在协调和导航多个四旋翼飞行器等复杂领域取得了显著进展。然而，即使训练良好的策略在障碍密集环境中仍易发生碰撞。通过重新训练或微调来解决这些罕见但关键的安全故障成本高昂，并可能削弱之前学到的技能。受大规模语言模型的激活转向和计算机视觉中的潜在编辑启发，我们提出了一个推理时潜在激活编辑（LAE）框架，该框架在不修改预训练策略的权重或架构的情况下，对其行为进行细化。该框架分两个阶段进行：（i）在线分类器监测中间激活以检测与不良行为相关的状态，（ii）一个激活编辑模块选择性地修改标记的激活，引导策略向更安全的区域转变。在本工作中，我们专注于改进多四旋翼飞行器导航中的安全性。我们假设增强策略对风险的内部感知可以诱发更安全的行为。我们通过训练一个潜在碰撞世界模型来预测预碰撞激活，从而促使更早和更加谨慎的规避响应来实现这一想法。大量模拟和现实世界的Crazyflie实验表明，LAE能显著减少碰撞（与未编辑基线相比，累积碰撞次数减少近90%）并大幅增加无碰撞轨迹的比例，同时保持任务完成。更广泛而言，我们的结果显示LAE是一种轻量级的范式，可以在资源受限的硬件上实现已部署后强化学习的机器人策略。 

---
# Uncertainty-Aware Active Source Tracking of Marine Pollution using Unmanned Surface Vehicles 

**Title (ZH)**: 使用无人表面车辆的不确定性感知海洋污染主动源追踪 

**Authors**: Song Ma, Richard Bucknall, Yuanchang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20593)  

**Abstract**: This paper proposes an uncertainty-aware marine pollution source tracking framework for unmanned surface vehicles (USVs). By integrating high-fidelity marine pollution dispersion simulation with informative path planning techniques, we demonstrate effective identification of pollution sources in marine environments. The proposed approach is implemented based on Robot Operating System (ROS), processing real-time sensor data to update probabilistic source location estimates. The system progressively refines the estimation of source location while quantifying uncertainty levels in its predictions. Experiments conducted in simulated environments with varying source locations, flow conditions, and starting positions demonstrate the framework's ability to localise pollution sources with high accuracy. Results show that the proposed approach achieves reliable source localisation efficiently. This work contributes to the development of full autonomous environmental monitoring capabilities essential for rapid response to marine pollution incidents. 

**Abstract (ZH)**: 本文提出了一种不确定性感知的无人水面船舶海洋污染源跟踪框架。通过集成高保真海洋污染分散模拟与信息性路径规划技术，我们展示了在海洋环境中有效识别污染源的能力。所提出的方法基于Robot Operating System (ROS) 实现，处理实时传感器数据以更新概率性源位置估计。系统逐步细化源位置估计的同时，量化预测中的不确定性水平。在不同源位置、流条件和起始位置的模拟环境中进行的实验展示了该框架以高精度定位污染源的能力。结果表明，所提出的方法能够高效地实现可靠的源定位。本文为快速应对海洋污染事件所需的整体自主环境监测能力的发展做出了贡献。 

---
# GraspFactory: A Large Object-Centric Grasping Dataset 

**Title (ZH)**: GraspFactory: 一个大型以对象为中心的抓取数据集 

**Authors**: Srinidhi Kalgundi Srinivas, Yash Shukla, Adam Arnold, Sachin Chitta  

**Link**: [PDF](https://arxiv.org/pdf/2509.20550)  

**Abstract**: Robotic grasping is a crucial task in industrial automation, where robots are increasingly expected to handle a wide range of objects. However, a significant challenge arises when robot grasping models trained on limited datasets encounter novel objects. In real-world environments such as warehouses or manufacturing plants, the diversity of objects can be vast, and grasping models need to generalize to this diversity. Training large, generalizable robot-grasping models requires geometrically diverse datasets. In this paper, we introduce GraspFactory, a dataset containing over 109 million 6-DoF grasps collectively for the Franka Panda (with 14,690 objects) and Robotiq 2F-85 grippers (with 33,710 objects). GraspFactory is designed for training data-intensive models, and we demonstrate the generalization capabilities of one such model trained on a subset of GraspFactory in both simulated and real-world settings. The dataset and tools are made available for download at this https URL. 

**Abstract (ZH)**: 机器人抓取是工业自动化中的关键任务，其中机器人被期望处理各种各样的物体。然而，当机器人抓取模型在有限的数据集上训练时，面对新型物体时会遇到重大挑战。在如仓库或制造工厂等实际环境中，物体的多样性很大，抓取模型需要泛化到这种多样性。训练大规模且泛化能力强的机器人抓取模型需要几何上多样的数据集。在本文中，我们介绍了GraspFactory数据集，该数据集包含了超109百万个Franka Panda（14,690个物体）和Robotiq 2F-85夹爪（33,710个物体）的6-DoF抓取。GraspFactory旨在用于数据密集型模型的训练，并证明了在其子集上训练的一个模型在仿真和实际环境中的泛化能力。该数据集和工具可在以下链接下载。 

---
# Selective Progress-Aware Querying for Human-in-the-Loop Reinforcement Learning 

**Title (ZH)**: 面向人类在环强化学习的选择性进步感知查询 

**Authors**: Anujith Muraleedharan, Anamika J H  

**Link**: [PDF](https://arxiv.org/pdf/2509.20541)  

**Abstract**: Human feedback can greatly accelerate robot learning, but in real-world settings, such feedback is costly and limited. Existing human-in-the-loop reinforcement learning (HiL-RL) methods often assume abundant feedback, limiting their practicality for physical robot deployment. In this work, we introduce SPARQ, a progress-aware query policy that requests feedback only when learning stagnates or worsens, thereby reducing unnecessary oracle calls. We evaluate SPARQ on a simulated UR5 cube-picking task in PyBullet, comparing against three baselines: no feedback, random querying, and always querying. Our experiments show that SPARQ achieves near-perfect task success, matching the performance of always querying while consuming about half the feedback budget. It also provides more stable and efficient learning than random querying, and significantly improves over training without feedback. These findings suggest that selective, progress-based query strategies can make HiL-RL more efficient and scalable for robots operating under realistic human effort constraints. 

**Abstract (ZH)**: 基于进展的查询策略SPARQ可加速受人性约束的强化学习 

---
# Action-Informed Estimation and Planning: Clearing Clutter on Staircases via Quadrupedal Pedipulation 

**Title (ZH)**: 基于行动指导的估计与规划：通过四足步行清除楼梯上的障碍物 

**Authors**: Prasanna Sriganesh, Barath Satheeshkumar, Anushree Sabnis, Matthew Travers  

**Link**: [PDF](https://arxiv.org/pdf/2509.20516)  

**Abstract**: For robots to operate autonomously in densely cluttered environments, they must reason about and potentially physically interact with obstacles to clear a path. Safely clearing a path on challenging terrain, such as a cluttered staircase, requires controlled interaction. For example, a quadrupedal robot that pushes objects out of the way with one leg while maintaining a stable stance with its three other legs. However, tightly coupled physical actions, such as one-legged pushing, create new constraints on the system that can be difficult to predict at design time. In this work, we present a new method that addresses one such constraint, wherein the object being pushed by a quadrupedal robot with one of its legs becomes occluded from the robot's sensors during manipulation. To address this challenge, we present a tightly coupled perception-action framework that enables the robot to perceive clutter, reason about feasible push paths, and execute the clearing maneuver. Our core contribution is an interaction-aware state estimation loop that uses proprioceptive feedback regarding foot contact and leg position to predict an object's displacement during the occlusion. This prediction guides the perception system to robustly re-detect the object after the interaction, closing the loop between action and sensing to enable accurate tracking even after partial pushes. Using this feedback allows the robot to learn from physical outcomes, reclassifying an object as immovable if a push fails due to it being too heavy. We present results of implementing our approach on a Boston Dynamics Spot robot that show our interaction-aware approach achieves higher task success rates and tracking accuracy in pushing objects on stairs compared to open-loop baselines. 

**Abstract (ZH)**: 自主导航于密集障碍环境中的机器人必须推理并可能物理互动以清除路径。在挑战性地形（如杂乱的楼梯）上安全地清除路径需要受控互动。例如，四足机器人利用一只腿推动物体，同时用其他三只腿保持稳定的姿态。然而，紧密耦合的物理动作（如单腿推动）会给系统带来新的约束，这些约束在设计时难以预测。在本工作中，我们提出了一种新方法，以解决其中一种约束：四足机器人用一只腿推动物体时，该物体会在操作过程中被遮挡，从而无法被机器人的传感器检测到。为了应对这一挑战，我们提出了一种紧密耦合的感知-动作框架，使机器人能够感知障碍物、推理可行的推动路径，并执行清除操作。我们的核心贡献是一种感知-互动感知估计回路，它利用有关足部接触和腿部位置的本体感受反馈来预测物体在遮挡期间的位移。这一预测指导感知系统在互动后 robust 地重新检测物体，从而在动作与感知之间形成闭环，即使在部分推动后也能实现精确跟踪。利用这种反馈，机器人可以从物理结果中学习，如果推动失败是因为物体太重则重新分类物体为无法移动。我们展示了在Boston Dynamics Spot机器人的实现结果，表明我们的感知-互动方法在楼梯上推动物体时比开环基线方法具有更高的任务成功率和跟踪准确性。 

---
# MELEGROS: Monolithic Elephant-inspired Gripper with Optical Sensors 

**Title (ZH)**: MELEGROS: 光学传感器驱动的独体象形夹取器 

**Authors**: Petr Trunin, Diana Cafiso, Anderson Brazil Nardin, Trevor Exley, Lucia Beccai  

**Link**: [PDF](https://arxiv.org/pdf/2509.20510)  

**Abstract**: The elephant trunk exemplifies a natural gripper where structure, actuation, and sensing are seamlessly integrated. Inspired by the distal morphology of the African elephant trunk, we present MELEGROS, a Monolithic ELEphant-inspired GRipper with Optical Sensors, emphasizing sensing as an intrinsic, co-fabricated capability. Unlike multi-material or tendon-based approaches, MELEGROS directly integrates six optical waveguide sensors and five pneumatic chambers into a pneumatically actuated lattice structure (12.5 mm cell size) using a single soft resin and one continuous 3D print. This eliminates mechanical mismatches between sensors, actuators, and body, reducing model uncertainty and enabling simulation-guided sensor design and placement. Only four iterations were required to achieve the final prototype, which features a continuous structure capable of elongation, compression, and bending while decoupling tactile and proprioceptive signals. MELEGROS (132 g) lifts more than twice its weight, performs bioinspired actions such as pinching, scooping, and reaching, and delicately grasps fragile items like grapes. The integrated optical sensors provide distinct responses to touch, bending, and chamber deformation, enabling multifunctional perception. MELEGROS demonstrates a new paradigm for soft robotics where fully embedded sensing and continuous structures inherently support versatile, bioinspired manipulation. 

**Abstract (ZH)**: MELEGROS：一种以光学传感器为代表的单一树脂连续结构的象鼻启发式柔体夹爪 

---
# Boosting Zero-Shot VLN via Abstract Obstacle Map-Based Waypoint Prediction with TopoGraph-and-VisitInfo-Aware Prompting 

**Title (ZH)**: 基于抽象障碍地图的零样本视觉语言导航中的航点预测以及拓扑图和到访信息感知提示增强 

**Authors**: Boqi Li, Siyuan Li, Weiyi Wang, Anran Li, Zhong Cao, Henry X. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20499)  

**Abstract**: With the rapid progress of foundation models and robotics, vision-language navigation (VLN) has emerged as a key task for embodied agents with broad practical applications. We address VLN in continuous environments, a particularly challenging setting where an agent must jointly interpret natural language instructions, perceive its surroundings, and plan low-level actions. We propose a zero-shot framework that integrates a simplified yet effective waypoint predictor with a multimodal large language model (MLLM). The predictor operates on an abstract obstacle map, producing linearly reachable waypoints, which are incorporated into a dynamically updated topological graph with explicit visitation records. The graph and visitation information are encoded into the prompt, enabling reasoning over both spatial structure and exploration history to encourage exploration and equip MLLM with local path planning for error correction. Extensive experiments on R2R-CE and RxR-CE show that our method achieves state-of-the-art zero-shot performance, with success rates of 41% and 36%, respectively, outperforming prior state-of-the-art methods. 

**Abstract (ZH)**: 随着基础模型和机器人技术的迅猛发展，视觉-语言导航（VLN）已成为具有广泛实际应用的体现代理的关键任务。我们关注连续环境中的VLN，这是一个特别具有挑战性的设置，代理必须联合解释自然语言指令、感知周围环境并规划低级动作。我们提出了一种零样本框架，该框架将一个简化但有效的航点预测器与多模态大规模语言模型（MLLM）结合起来。预测器在抽象障碍图上运行，生成线性可达的航点，这些航点被纳入一个动态更新的拓扑图中，其中包含明确的访问记录。图和访问信息被编码到提示中，从而能够在空间结构和探索历史之间进行推理，鼓励探索并为MLLM提供局部路径规划以进行错误纠正。在R2R-CE和RxR-CE上的 extensive 实验表明，我们的方法实现了最先进的零样本性能，成功率分别为 41% 和 36%，超过了先前的最先进的方法。 

---
# Revisiting Formal Methods for Autonomous Robots: A Structured Survey 

**Title (ZH)**: 重新审视自主机器人形式化方法：一种结构化综述 

**Authors**: Atef Azaiez, David A. Anisi, Marie Farrell, Matt Luckcuck  

**Link**: [PDF](https://arxiv.org/pdf/2509.20488)  

**Abstract**: This paper presents the initial results from our structured literature review on applications of Formal Methods (FM) to Robotic Autonomous Systems (RAS). We describe our structured survey methodology; including database selection and associated search strings, search filters and collaborative review of identified papers. We categorise and enumerate the FM approaches and formalisms that have been used for specification and verification of RAS. We investigate FM in the context of sub-symbolic AI-enabled RAS and examine the evolution of how FM is used over time in this field. This work complements a pre-existing survey in this area and we examine how this research area has matured over time. Specifically, our survey demonstrates that some trends have persisted as observed in a previous survey. Additionally, it recognized new trends that were not considered previously including a noticeable increase in adopting Formal Synthesis approaches as well as Probabilistic Verification Techniques. 

**Abstract (ZH)**: 本文呈现了我们对形式方法在机器人自主系统中应用的结构化文献综述的初步结果。我们描述了结构化调查方法；包括数据库选择和相关搜索字符串、搜索过滤器以及已识别论文的协作审查。我们将形式化方法和形式语言分类并列举，这些方法和形式语言被用于定义和验证机器人自主系统。我们探讨了形式方法在具有亚符号AI的机器人自主系统中的应用，并考察了这种方法在该领域随时间的发展演变。本文补充了该领域的现有调查，并考察了该研究领域随时间的成熟。具体而言，我们的调查表明，一些趋势在之前的调查中已经持续存在。此外，还识别了一些新的趋势，这些趋势在之前的调查中未被考虑，包括显著增加采用形式综合方法以及概率验证技术。 

---
# Boosting LiDAR-Based Localization with Semantic Insight: Camera Projection versus Direct LiDAR Segmentation 

**Title (ZH)**: 基于语义洞察提升LiDARベース的定位： Camera投影 versus 直接LiDAR分割 

**Authors**: Sven Ochs, Philip Schörner, Marc René Zofka, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2509.20486)  

**Abstract**: Semantic segmentation of LiDAR data presents considerable challenges, particularly when dealing with diverse sensor types and configurations. However, incorporating semantic information can significantly enhance the accuracy and robustness of LiDAR-based localization techniques for autonomous mobile systems. We propose an approach that integrates semantic camera data with LiDAR segmentation to address this challenge. By projecting LiDAR points into the semantic segmentation space of the camera, our method enhances the precision and reliability of the LiDAR-based localization pipeline.
For validation, we utilize the CoCar NextGen platform from the FZI Research Center for Information Technology, which offers diverse sensor modalities and configurations. The sensor setup of CoCar NextGen enables a thorough analysis of different sensor types. Our evaluation leverages the state-of-the-art Depth-Anything network for camera image segmentation and an adaptive segmentation network for LiDAR segmentation. To establish a reliable ground truth for LiDAR-based localization, we make us of a Global Navigation Satellite System (GNSS) solution with Real-Time Kinematic corrections (RTK). Additionally, we conduct an extensive 55 km drive through the city of Karlsruhe, Germany, covering a variety of environments, including urban areas, multi-lane roads, and rural highways. This multimodal approach paves the way for more reliable and precise autonomous navigation systems, particularly in complex real-world environments. 

**Abstract (ZH)**: 基于语义的LiDAR数据分割在处理多种传感器类型和配置时面临重大挑战，但结合语义信息可以显著提高自主移动系统中基于LiDAR的局部化技术的准确性和鲁棒性。我们提出了一种将语义相机数据与LiDAR分割集成的方法来应对这一挑战。通过将LiDAR点投影到相机的语义分割空间，我们的方法增强了基于LiDAR的局部化管道的精密度和可靠性。 

---
# Finding 3D Positions of Distant Objects from Noisy Camera Movement and Semantic Segmentation Sequences 

**Title (ZH)**: 从嘈杂的相机运动和语义分割序列中寻找远处物体的3D位置 

**Authors**: Julius Pesonen, Arno Solin, Eija Honkavaara  

**Link**: [PDF](https://arxiv.org/pdf/2509.20906)  

**Abstract**: 3D object localisation based on a sequence of camera measurements is essential for safety-critical surveillance tasks, such as drone-based wildfire monitoring. Localisation of objects detected with a camera can typically be solved with dense depth estimation or 3D scene reconstruction. However, in the context of distant objects or tasks limited by the amount of available computational resources, neither solution is feasible. In this paper, we show that the task can be solved using particle filters for both single and multiple target scenarios. The method was studied using a 3D simulation and a drone-based image segmentation sequence with global navigation satellite system (GNSS)-based camera pose estimates. The results showed that a particle filter can be used to solve practical localisation tasks based on camera poses and image segments in these situations where other solutions fail. The particle filter is independent of the detection method, making it flexible for new tasks. The study also demonstrates that drone-based wildfire monitoring can be conducted using the proposed method paired with a pre-existing image segmentation model. 

**Abstract (ZH)**: 基于相机测量序列的3D物体定位对于无人机森林火灾监测等安全关键监控任务至关重要。物体的定位通常可以通过密集深度估计或3D场景重建来解决。但在远处物体或计算资源有限的任务中，这两种解决方案都不实用。本文展示了一种使用粒子滤波器来解决单目标和多目标场景定位任务的方法。该方法在3D仿真和基于全球导航卫星系统（GNSS）的相机姿态估计的无人机图像分割序列中进行了研究。结果表明，粒子滤波器可以在其他解决方案失效的情况下，用于基于相机姿态和图像分割的实际定位任务。粒子滤波器独立于检测方法，具有良好的灵活性，适用于新任务。此外，研究表明，提出的配以预存图像分割模型的方法可用于无人机森林火灾监测。 

---
# Meta-Memory: Retrieving and Integrating Semantic-Spatial Memories for Robot Spatial Reasoning 

**Title (ZH)**: 元记忆：检索和整合语义空间记忆以进行机器人空间 reasoning 

**Authors**: Yufan Mao, Hanjing Ye, Wenlong Dong, Chengjie Zhang, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20754)  

**Abstract**: Navigating complex environments requires robots to effectively store observations as memories and leverage them to answer human queries about spatial locations, which is a critical yet underexplored research challenge. While prior work has made progress in constructing robotic memory, few have addressed the principled mechanisms needed for efficient memory retrieval and integration. To bridge this gap, we propose Meta-Memory, a large language model (LLM)-driven agent that constructs a high-density memory representation of the environment. The key innovation of Meta-Memory lies in its capacity to retrieve and integrate relevant memories through joint reasoning over semantic and spatial modalities in response to natural language location queries, thereby empowering robots with robust and accurate spatial reasoning capabilities. To evaluate its performance, we introduce SpaceLocQA, a large-scale dataset encompassing diverse real-world spatial question-answering scenarios. Experimental results show that Meta-Memory significantly outperforms state-of-the-art methods on both the SpaceLocQA and the public NaVQA benchmarks. Furthermore, we successfully deployed Meta-Memory on real-world robotic platforms, demonstrating its practical utility in complex environments. Project page: this https URL . 

**Abstract (ZH)**: 导航复杂环境要求机器人有效地存储观察作为记忆，并利用这些记忆回答关于空间位置的人类查询，这是一项关键但尚未充分探索的研究挑战。尽管以往的工作在构建机器人记忆方面取得了进展，但很少有研究解决高效记忆检索和整合的原理机制。为弥补这一差距，我们提出了一种由大型语言模型（LLM）驱动的代理Meta-Memory，能够构建环境的高密度记忆表示。Meta-Memory的关键创新在于通过综合推理语义和空间模态来检索和整合相关记忆，以响应自然语言位置查询，从而赋予机器人稳健而准确的空间推理能力。为了评估其性能，我们引入了SpaceLocQA，这是一个包含多样化真实世界空间问答场景的大规模数据集。实验结果表明，Meta-Memory在SpaceLocQA和公开的NaVQA基准上显著优于现有方法。此外，我们成功将Meta-Memory部署在实际机器人平台上，展示了其在复杂环境中的实用价值。项目页面：this https URL。 

---
# Wonder Wins Ways: Curiosity-Driven Exploration through Multi-Agent Contextual Calibration 

**Title (ZH)**: 好奇心驱动的多agent情境校准探索：奇伟之路 

**Authors**: Yiyuan Pan, Zhe Liu, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20648)  

**Abstract**: Autonomous exploration in complex multi-agent reinforcement learning (MARL) with sparse rewards critically depends on providing agents with effective intrinsic motivation. While artificial curiosity offers a powerful self-supervised signal, it often confuses environmental stochasticity with meaningful novelty. Moreover, existing curiosity mechanisms exhibit a uniform novelty bias, treating all unexpected observations equally. However, peer behavior novelty, which encode latent task dynamics, are often overlooked, resulting in suboptimal exploration in decentralized, communication-free MARL settings. To this end, inspired by how human children adaptively calibrate their own exploratory behaviors via observing peers, we propose a novel approach to enhance multi-agent exploration. We introduce CERMIC, a principled framework that empowers agents to robustly filter noisy surprise signals and guide exploration by dynamically calibrating their intrinsic curiosity with inferred multi-agent context. Additionally, CERMIC generates theoretically-grounded intrinsic rewards, encouraging agents to explore state transitions with high information gain. We evaluate CERMIC on benchmark suites including VMAS, Meltingpot, and SMACv2. Empirical results demonstrate that exploration with CERMIC significantly outperforms SoTA algorithms in sparse-reward environments. 

**Abstract (ZH)**: 自主探索在复杂多智能体强化学习（MARL）中的应用：稀疏奖励下的有效内在动机对于自主探索至关重要。基于观察同伴行为，我们提出一种新颖方法以增强多智能体探索。我们引入CERMIC，一种原理性的框架，使智能体能够 robust 地过滤嘈杂的惊讶信号，并通过动态校准内在好奇心与推断出的多智能体上下文来引导探索。此外，CERMIC 生成理论依据内在奖励，鼓励智能体探索具有高信息增益的状态转换。我们将在 VMAS、Meltingpot 和 SMACv2 等基准套件上评估 CERMIC。实验证明，在稀疏奖励环境中，使用 CERMIC 进行探索显著优于现有最佳算法。 

---
# Large Pre-Trained Models for Bimanual Manipulation in 3D 

**Title (ZH)**: 大型预训练模型在3D双手操作中的应用 

**Authors**: Hanna Yurchyk, Wei-Di Chang, Gregory Dudek, David Meger  

**Link**: [PDF](https://arxiv.org/pdf/2509.20579)  

**Abstract**: We investigate the integration of attention maps from a pre-trained Vision Transformer into voxel representations to enhance bimanual robotic manipulation. Specifically, we extract attention maps from DINOv2, a self-supervised ViT model, and interpret them as pixel-level saliency scores over RGB images. These maps are lifted into a 3D voxel grid, resulting in voxel-level semantic cues that are incorporated into a behavior cloning policy. When integrated into a state-of-the-art voxel-based policy, our attention-guided featurization yields an average absolute improvement of 8.2% and a relative gain of 21.9% across all tasks in the RLBench bimanual benchmark. 

**Abstract (ZH)**: 我们将注意力图集成到预训练的视觉变换器中的体素表示中，以增强双手机器人操作。具体来说，我们从自监督ViT模型DINOv2中提取注意力图，并将其解释为RGB图像上的像素级显著性评分。这些图被提升到3D体素网格中，产生体素级语义线索，这些线索被纳入行为克隆策略中。当集成到最先进的基于体素的策略中时，我们的注意力引导特征化在RLBench双手基准中的所有任务上平均绝对改善了8.2%，相对增益为21.9%。 

---
# SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent 

**Title (ZH)**: SceneWeaver：一劳永逸的3D场景合成扩展自反代理 

**Authors**: Yandan Yang, Baoxiong Jia, Shujie Zhang, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20414)  

**Abstract**: Indoor scene synthesis has become increasingly important with the rise of Embodied AI, which requires 3D environments that are not only visually realistic but also physically plausible and functionally diverse. While recent approaches have advanced visual fidelity, they often remain constrained to fixed scene categories, lack sufficient object-level detail and physical consistency, and struggle to align with complex user instructions. In this work, we present SceneWeaver, a reflective agentic framework that unifies diverse scene synthesis paradigms through tool-based iterative refinement. At its core, SceneWeaver employs a language model-based planner to select from a suite of extensible scene generation tools, ranging from data-driven generative models to visual- and LLM-based methods, guided by self-evaluation of physical plausibility, visual realism, and semantic alignment with user input. This closed-loop reason-act-reflect design enables the agent to identify semantic inconsistencies, invoke targeted tools, and update the environment over successive iterations. Extensive experiments on both common and open-vocabulary room types demonstrate that SceneWeaver not only outperforms prior methods on physical, visual, and semantic metrics, but also generalizes effectively to complex scenes with diverse instructions, marking a step toward general-purpose 3D environment generation. Project website: this https URL. 

**Abstract (ZH)**: 室内场景合成随着沉浸式AI的兴起变得越来越重要，这需要既具备视觉逼真度又物理合理且功能多样的3D环境。虽然近期的方法在视觉保真度方面取得了进展，但它们往往局限于固定的场景类别，缺乏足够的物体级细节和物理一致性，并且难以与复杂的用户指令对齐。在本文中，我们提出了SceneWeaver，这是一种反思性自主框架，通过基于工具的迭代优化统一了多种场景合成范式。其核心是SceneWeaver采用基于语言模型的规划器，选择从数据驱动生成模型到基于视觉和LLM的方法等多种可扩展的场景生成工具，这些选择受到对物理合理性、视觉逼真度和语义与用户输入的一致性的自我评估指导。这种闭环的思考-行动-反思设计使代理能够识别语义不一致，激活特定的工具，并在连续迭代中更新环境。在对常见和开放式词汇房间类型的广泛实验中，SceneWeaver不仅在物理、视觉和语义指标上优于先前的方法，而且能够有效地应用于具有多种指令的复杂场景，朝着通用3D环境生成迈进。项目网站：this https URL。 

---
# SGAligner++: Cross-Modal Language-Aided 3D Scene Graph Alignment 

**Title (ZH)**: SGAligner++: 跨模态语言辅助的3D场景图对齐 

**Authors**: Binod Singh, Sayan Deb Sarkar, Iro Armeni  

**Link**: [PDF](https://arxiv.org/pdf/2509.20401)  

**Abstract**: Aligning 3D scene graphs is a crucial initial step for several applications in robot navigation and embodied perception. Current methods in 3D scene graph alignment often rely on single-modality point cloud data and struggle with incomplete or noisy input. We introduce SGAligner++, a cross-modal, language-aided framework for 3D scene graph alignment. Our method addresses the challenge of aligning partially overlapping scene observations across heterogeneous modalities by learning a unified joint embedding space, enabling accurate alignment even under low-overlap conditions and sensor noise. By employing lightweight unimodal encoders and attention-based fusion, SGAligner++ enhances scene understanding for tasks such as visual localization, 3D reconstruction, and navigation, while ensuring scalability and minimal computational overhead. Extensive evaluations on real-world datasets demonstrate that SGAligner++ outperforms state-of-the-art methods by up to 40% on noisy real-world reconstructions, while enabling cross-modal generalization. 

**Abstract (ZH)**: 跨模态语言辅助的3D场景图对齐 

---
