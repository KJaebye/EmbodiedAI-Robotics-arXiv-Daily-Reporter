# TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System 

**Title (ZH)**: TWIST2: 可扩展、便携且全面的人形数据采集系统 

**Authors**: Yanjie Ze, Siheng Zhao, Weizhuo Wang, Angjoo Kanazawa, Rocky Duan, Pieter Abbeel, Guanya Shi, Jiajun Wu, C. Karen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02832)  

**Abstract**: Large-scale data has driven breakthroughs in robotics, from language models to vision-language-action models in bimanual manipulation. However, humanoid robotics lacks equally effective data collection frameworks. Existing humanoid teleoperation systems either use decoupled control or depend on expensive motion capture setups. We introduce TWIST2, a portable, mocap-free humanoid teleoperation and data collection system that preserves full whole-body control while advancing scalability. Our system leverages PICO4U VR for obtaining real-time whole-body human motions, with a custom 2-DoF robot neck (cost around $250) for egocentric vision, enabling holistic human-to-humanoid control. We demonstrate long-horizon dexterous and mobile humanoid skills and we can collect 100 demonstrations in 15 minutes with an almost 100% success rate. Building on this pipeline, we propose a hierarchical visuomotor policy framework that autonomously controls the full humanoid body based on egocentric vision. Our visuomotor policy successfully demonstrates whole-body dexterous manipulation and dynamic kicking tasks. The entire system is fully reproducible and open-sourced at this https URL . Our collected dataset is also open-sourced at this https URL . 

**Abstract (ZH)**: 大规模数据推动了机器人学的进步，从语言模型到双臂操控的视觉-语言-动作模型。然而，类人机器人缺乏同样有效的数据采集框架。现有的类人机器人远程操控系统要么采用解耦控制，要么依赖昂贵的运动捕捉设备。我们提出TWIST2，这是一种便携且无需运动捕捉的类人机器人远程操控和数据采集系统，保持全身控制的同时提高了可扩展性。该系统利用PICO4U VR获取实时全身人类动作，并配备一个自定义的2-DoF机器人颈部（成本约为250美元），以实现以第一人称视角的全方位人类到类人机器人操控。我们展示了长时程灵巧且移动的类人机器人技能，并在15分钟内收集了100个演示，成功率接近100%。在此基础上，我们提出了一种基于以第一人称视角的层次视觉-运动策略框架，能够自主控制整体现人机器人身体。我们的视觉-运动策略成功展示了全身灵巧操控和动态踢球任务。整个系统完全可再现并开源于https://this.url/。我们的收集数据集也开源于https://this.url/。 

---
# XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations 

**Title (ZH)**: XR-1: 向通用视觉-语言-行动模型方向探索统一视觉-运动表示的学习方法 

**Authors**: Shichao Fan, Kun Wu, Zhengping Che, Xinhua Wang, Di Wu, Fei Liao, Ning Liu, Yixue Zhang, Zhen Zhao, Zhiyuan Xu, Meng Li, Qingjie Liu, Shanghang Zhang, Min Wan, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.02776)  

**Abstract**: Recent progress in large-scale robotic datasets and vision-language models (VLMs) has advanced research on vision-language-action (VLA) models. However, existing VLA models still face two fundamental challenges: (i) producing precise low-level actions from high-dimensional observations, (ii) bridging domain gaps across heterogeneous data sources, including diverse robot embodiments and human demonstrations. Existing methods often encode latent variables from either visual dynamics or robotic actions to guide policy learning, but they fail to fully exploit the complementary multi-modal knowledge present in large-scale, heterogeneous datasets. In this work, we present X Robotic Model 1 (XR-1), a novel framework for versatile and scalable VLA learning across diverse robots, tasks, and environments. XR-1 introduces the \emph{Unified Vision-Motion Codes (UVMC)}, a discrete latent representation learned via a dual-branch VQ-VAE that jointly encodes visual dynamics and robotic motion. UVMC addresses these challenges by (i) serving as an intermediate representation between the observations and actions, and (ii) aligning multimodal dynamic information from heterogeneous data sources to capture complementary knowledge. To effectively exploit UVMC, we propose a three-stage training paradigm: (i) self-supervised UVMC learning, (ii) UVMC-guided pretraining on large-scale cross-embodiment robotic datasets, and (iii) task-specific post-training. We validate XR-1 through extensive real-world experiments with more than 14,000 rollouts on six different robot embodiments, spanning over 120 diverse manipulation tasks. XR-1 consistently outperforms state-of-the-art baselines such as $\pi_{0.5}$, $\pi_0$, RDT, UniVLA, and GR00T-N1.5 while demonstrating strong generalization to novel objects, background variations, distractors, and illumination changes. Our project is at this https URL. 

**Abstract (ZH)**: 近年来，大规模机器人数据集和视觉-语言模型的进展推动了视觉-语言-动作（VLA）模型的研究。然而，现有的VLA模型仍然面临两大基本挑战：（i）从高维观察中生成精确的低级动作，（ii）弥合异构数据源之间的领域差距，包括多种机器人形态和人类示范之间的差距。现有方法通常通过编码视觉动态或机器人动作的潜在变量来引导策略学习，但这些方法未能充分利用大规模异构数据集中存在的互补多模态知识。在这项工作中，我们提出了XR-1机器人模型，一种用于跨多种机器人、任务和环境的多样性和可扩展性视觉-语言-动作学习的新框架。XR-1引入了统一的视觉-运动编码（UVMC），这是一种通过双分支VQ-VAE联合编码视觉动态和机器人运动的学习离散潜在表示。UVMC通过（i）作为观察和动作之间的中间表示，以及（ii）将来自异构数据源的多模态动态信息对齐以捕获互补知识来解决这些挑战。为了有效利用UVMC，我们提出了一种三阶段训练范式：（i）自我监督的UVMC学习，（ii）在大规模跨形态机器人数据集上的UVMC引导预训练，以及（iii）任务特定的后训练。我们通过在六个不同机器人形态上的超过14,000个卷积实验进行了广泛的实际验证，涵盖了超过120种多样化的操作任务。XR-1在$\pi_{0.5}$、$\pi_0$、RDT、UniVLA和GR00T-N1.5等最先进的基线模型上表现出色，并且在新型物体、背景变化、干扰和光照变化的泛化能力方面表现出色。我们的项目网页链接为：this https URL。 

---
# Non-Contact Manipulation of Induced Magnetic Dipoles 

**Title (ZH)**: 非接触操纵诱导磁偶极子 

**Authors**: Seth Stewart, Joseph Pawelski, Steve Ward, Andrew J. Petruska  

**Link**: [PDF](https://arxiv.org/pdf/2511.02761)  

**Abstract**: Extending the field of magnetic manipulation to conductive, non-magnetic objects opens the door for a wide array of applications previously limited to hard or soft magnetic materials. Of particular interest is the recycling of space debris through the use of oscillating magnetic fields, which represent a cache of raw materials in an environment particularly suited to the low forces generated from inductive magnetic manipulation. Building upon previous work that demonstrated 3D open-loop position control by leveraging the opposing dipole moment created from induced eddy currents, this work demonstrates closed-loop position control of a semi-buoyant aluminum sphere in lab tests, and the efficacy of varying methods for force inversion is explored. The closed-loop methods represent a critical first step towards wider applications for 3-DOF position control of induced magnetic dipoles. 

**Abstract (ZH)**: 扩大磁操控领域以作用于导电非磁性物体为以前仅限于硬磁或软磁材料的应用打开了新门径。特别感兴趣的是，通过使用振荡磁场回收空间碎片，振荡磁场在由感生环流产生的反向磁偶矩作用下提供了适合低磁力操控的原始材料库。在前人通过利用感生环流产生的反向磁偶矩实现3D开环位置控制的基础上，本研究在实验室测试中展示了诱导磁偶矩半浮铝球的闭环位置控制，并探讨了不同方法在力反转方面的有效性。闭环方法代表了诱导磁偶矩在三维自由度位置控制方面更广泛应用的关键第一步。 

---
# Dexterous Robotic Piano Playing at Scale 

**Title (ZH)**: 大规模灵巧机器人钢琴演奏 

**Authors**: Le Chen, Yi Zhao, Jan Schneider, Quankai Gao, Simon Guist, Cheng Qian, Juho Kannala, Bernhard Schölkopf, Joni Pajarinen, Dieter Büchler  

**Link**: [PDF](https://arxiv.org/pdf/2511.02504)  

**Abstract**: Endowing robot hands with human-level dexterity has been a long-standing goal in robotics. Bimanual robotic piano playing represents a particularly challenging task: it is high-dimensional, contact-rich, and requires fast, precise control. We present OmniPianist, the first agent capable of performing nearly one thousand music pieces via scalable, human-demonstration-free learning. Our approach is built on three core components. First, we introduce an automatic fingering strategy based on Optimal Transport (OT), allowing the agent to autonomously discover efficient piano-playing strategies from scratch without demonstrations. Second, we conduct large-scale Reinforcement Learning (RL) by training more than 2,000 agents, each specialized in distinct music pieces, and aggregate their experience into a dataset named RP1M++, consisting of over one million trajectories for robotic piano playing. Finally, we employ a Flow Matching Transformer to leverage RP1M++ through large-scale imitation learning, resulting in the OmniPianist agent capable of performing a wide range of musical pieces. Extensive experiments and ablation studies highlight the effectiveness and scalability of our approach, advancing dexterous robotic piano playing at scale. 

**Abstract (ZH)**: 赋予机器人手部人类级别的灵巧性一直是机器人学中的长期目标。双臂机器人钢琴演奏是一项特别具有挑战性的任务：它具有高维度、丰富的接触交互，并要求快速精确的控制。我们介绍了OmniPianist，这是首个无需人类示范即可通过可扩展学习执行近一千首乐曲的智能体。我们的方法基于三个核心组件。首先，我们引入了一种基于最优传输（OT）的自动指法策略，使智能体能够从零开始自主发现高效的钢琴演奏策略。第二，我们通过训练超过2000个专精于不同乐曲的智能体进行了大规模强化学习，并将它们的经验聚合成一个名为RP1M++的数据集，包含超过一百万个机器人钢琴演奏的轨迹。最后，我们使用流动匹配变换器利用RP1M++通过大规模模仿学习，从而生成能够演奏多种音乐作品的OmniPianist智能体。广泛的经验和消融实验突显了我们方法的有效性和可扩展性，推动了大规模灵巧机器人钢琴演奏的发展。 

---
# Whole-body motion planning and safety-critical control for aerial manipulation 

**Title (ZH)**: 全身体动规划与安全关键控制在空中操作中 

**Authors**: Lin Yang, Jinwoo Lee, Domenico Campolo, H. Jin Kim, Jeonghyun Byun  

**Link**: [PDF](https://arxiv.org/pdf/2511.02342)  

**Abstract**: Aerial manipulation combines the maneuverability of multirotors with the dexterity of robotic arms to perform complex tasks in cluttered spaces. Yet planning safe, dynamically feasible trajectories remains difficult due to whole-body collision avoidance and the conservativeness of common geometric abstractions such as bounding boxes or ellipsoids. We present a whole-body motion planning and safety-critical control framework for aerial manipulators built on superquadrics (SQs). Using an SQ-plus-proxy representation, we model both the vehicle and obstacles with differentiable, geometry-accurate surfaces. Leveraging this representation, we introduce a maximum-clearance planner that fuses Voronoi diagrams with an equilibrium-manifold formulation to generate smooth, collision-aware trajectories. We further design a safety-critical controller that jointly enforces thrust limits and collision avoidance via high-order control barrier functions. In simulation, our approach outperforms sampling-based planners in cluttered environments, producing faster, safer, and smoother trajectories and exceeding ellipsoid-based baselines in geometric fidelity. Actual experiments on a physical aerial-manipulation platform confirm feasibility and robustness, demonstrating consistent performance across simulation and hardware settings. The video can be found at this https URL. 

**Abstract (ZH)**: 基于超几何体的空中操作器全身运动规划与安全控制框架 

---
# ZJUNlict Extended Team Description Paper 2025 

**Title (ZH)**: 浙江农林大学扩展团队描述论文2025 

**Authors**: Zifei Wu, Lijie Wang, Zhe Yang, Shijie Yang, Liang Wang, Haoran Fu, Yinliang Cai, Rong Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.02315)  

**Abstract**: This paper presents the ZJUNlict team's work over the past year, covering both hardware and software advancements. In the hardware domain, the integration of an IMU into the v2023 robot was completed to enhance posture accuracy and angular velocity planning. On the software side, key modules were optimized, including the strategy and CUDA modules, with significant improvements in decision making efficiency, ball pursuit prediction, and ball possession prediction to adapt to high-tempo game dynamics. 

**Abstract (ZH)**: 本论文介绍了ZJUNlict团队在过去一年中的工作，涵盖了硬件和软件的进展。在硬件方面，完成了将IMU集成到v2023机器人中以提高姿态准确性和角速度规划。在软件方面，优化了关键模块，包括策略和CUDA模块，显著提高了决策效率、球追逐预测和球权预测能力，以适应高速比赛 dynamics。 

---
# SuckTac: Camera-based Tactile Sucker for Unstructured Surface Perception and Interaction 

**Title (ZH)**: SuckTac: 基于相机的触觉吸盘，用于无结构表面感知与交互 

**Authors**: Ruiyong Yuan, Jieji Ren, Zhanxuan Peng, Feifei Chen, Guoying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2511.02294)  

**Abstract**: Suckers are significant for robots in picking, transferring, manipulation and locomotion on diverse surfaces. However, most of the existing suckers lack high-fidelity perceptual and tactile sensing, which impedes them from resolving the fine-grained geometric features and interaction status of the target surface. This limits their robust performance with irregular objects and in complex, unstructured environments. Inspired by the adaptive structure and high-performance sensory capabilities of cephalopod suckers, in this paper, we propose a novel, intelligent sucker, named SuckTac, that integrates a camera-based tactile sensor directly within its optimized structure to provide high-density perception and robust suction. Specifically, through joint structure design and optimization and based on a multi-material integrated casting technique, a camera and light source are embedded into the sucker, which enables in-situ, high-density perception of fine details like surface shape, texture and roughness. To further enhance robustness and adaptability, the sucker's mechanical design is also optimized by refining its profile, adding a compliant lip, and incorporating surface microstructure. Extensive experiments, including challenging tasks such as robotic cloth manipulation and soft mobile robot inspection, demonstrate the superior performance and broad applicability of the proposed system. 

**Abstract (ZH)**: 吸杯在多样化表面上抓取、转移、操作和运动中至关重要，然而现有的吸杯大多缺乏高保真感知和触觉传感能力，这限制了它们对目标表面的精细几何特征和交互状态的解析能力，影响了它们在不规则物体和复杂非结构化环境中的稳健性能。受墨鱼吸盘自适应结构和高性能感知能力的启发，本文提出了一种新型智能吸杯SuckTac，将基于摄像头的触觉传感器直接集成在其优化结构中，提供高密度感知和稳健的吸附力。具体而言，通过联合结构设计与优化以及多材料集成铸造技术，将摄像头和光源嵌入吸杯中，实现对表面形状、纹理和粗糙度等细节的原位高密度感知。为增强其稳健性和适应性，还对其机械设计进行了优化，包括改进外形、增加顺应性唇缘和表面微结构。广泛实验，包括机器人布料操作和软体移动机器人检测等挑战性任务，证明了所提系统的优越性能和广泛应用前景。 

---
# LACY: A Vision-Language Model-based Language-Action Cycle for Self-Improving Robotic Manipulation 

**Title (ZH)**: LACY：一种基于视听语言模型的语言-动作循环 Cycle 用于自我提升的机器人操作 

**Authors**: Youngjin Hong, Houjian Yu, Mingen Li, Changhyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.02239)  

**Abstract**: Learning generalizable policies for robotic manipulation increasingly relies on large-scale models that map language instructions to actions (L2A). However, this one-way paradigm often produces policies that execute tasks without deeper contextual understanding, limiting their ability to generalize or explain their behavior. We argue that the complementary skill of mapping actions back to language (A2L) is essential for developing more holistic grounding. An agent capable of both acting and explaining its actions can form richer internal representations and unlock new paradigms for self-supervised learning. We introduce LACY (Language-Action Cycle), a unified framework that learns such bidirectional mappings within a single vision-language model. LACY is jointly trained on three synergistic tasks: generating parameterized actions from language (L2A), explaining observed actions in language (A2L), and verifying semantic consistency between two language descriptions (L2C). This enables a self-improving cycle that autonomously generates and filters new training data through an active augmentation strategy targeting low-confidence cases, thereby improving the model without additional human labels. Experiments on pick-and-place tasks in both simulation and the real world show that LACY improves task success rates by 56.46% on average and yields more robust language-action grounding for robotic manipulation. Project page: this https URL 

**Abstract (ZH)**: 基于语言-行动循环的学习可迁移策略在机器人操作中的应用逐渐依赖于映射语言指令到操作（L2A）的大规模模型。然而，这种单向范式往往会产生执行任务而不进行深入上下文理解的策略，限制了它们的泛化能力和解释行为的能力。我们认为，将操作映射回语言（A2L）的补充技能对于开发更具综合性的场景是必不可少的。能够执行动作并解释其动作的智能体可以形成更丰富的内部表示，并解锁新的自监督学习范式。我们介绍了LACY（语言-行动循环），这是一种统一框架，能够在单个视觉-语言模型中学习这种双向映射。LACY通过一种综合的训练策略联合训练于三个任务：从语言生成参数化动作（L2A）、用语言解释观察到的动作（A2L）和验证两种语言描述之间的语义一致性（L2C）。这实现了自主生成和过滤新训练数据的循环，通过针对低置信度案例的主动增强策略提高模型性能，而无需额外的人工标签。在模拟和真实世界中的取放任务实验中，LACY平均提高了任务成功率56.46%，并为机器人操作提供了更 robust 的语言-动作 grounding。项目页面: this https URL 

---
# A Quantitative Comparison of Centralised and Distributed Reinforcement Learning-Based Control for Soft Robotic Arms 

**Title (ZH)**: 集中式与分布式基于强化学习的软 robotic � manipulator 控制方法的定量比较 

**Authors**: Linxin Hou, Qirui Wu, Zhihang Qin, Neil Banerjee, Yongxin Guo, Cecilia Laschi  

**Link**: [PDF](https://arxiv.org/pdf/2511.02192)  

**Abstract**: This paper presents a quantitative comparison between centralised and distributed multi-agent reinforcement learning (MARL) architectures for controlling a soft robotic arm modelled as a Cosserat rod in simulation. Using PyElastica and the OpenAI Gym interface, we train both a global Proximal Policy Optimisation (PPO) controller and a Multi-Agent PPO (MAPPO) under identical budgets. Both approaches are based on the arm having $n$ number of controlled sections. The study systematically varies $n$ and evaluates the performance of the arm to reach a fixed target in three scenarios: default baseline condition, recovery from external disturbance, and adaptation to actuator failure. Quantitative metrics used for the evaluation are mean action magnitude, mean final distance, mean episode length, and success rate. The results show that there are no significant benefits of the distributed policy when the number of controlled sections $n\le4$. In very simple systems, when $n\le2$, the centralised policy outperforms the distributed one. When $n$ increases to $4< n\le 12$, the distributed policy shows a high sample efficiency. In these systems, distributed policy promotes a stronger success rate, resilience, and robustness under local observability and yields faster convergence given the same sample size. However, centralised policies achieve much higher time efficiency during training as it takes much less time to train the same size of samples. These findings highlight the trade-offs between centralised and distributed policy in reinforcement learning-based control for soft robotic systems and provide actionable design guidance for future sim-to-real transfer in soft rod-like manipulators. 

**Abstract (ZH)**: 集中式与分布式多智能体强化学习架构在软体机器人臂模拟控制中的定量比较：基于科西西模型的分析 

---
# Kinematic and Ergonomic Design of a Robotic Arm for Precision Laparoscopic Surgery 

**Title (ZH)**: 精密腹腔镜手术中机器人臂的运动学与人机工程学设计 

**Authors**: Tian Hao, Tong Lu, Che Chan  

**Link**: [PDF](https://arxiv.org/pdf/2511.02167)  

**Abstract**: Robotic assistance in minimally invasive surgery can greatly enhance surgical precision and reduce surgeon fatigue. This paper presents a focused investigation on the kinematic and ergonomic design principles for a laparoscopic surgical robotic arm aimed at high-precision tasks. We propose a 7-degree-of-freedom (7-DOF) robotic arm system that incorporates a remote center of motion (RCM) at the instrument insertion point and ergonomic considerations to improve surgeon interaction. The design is implemented on a general-purpose robotic platform, and a series of simulated surgical tasks were performed to evaluate targeting accuracy, task efficiency, and surgeon comfort compared to conventional manual laparoscopy. Experimental results demonstrate that the optimized robotic design achieves significantly improved targeting accuracy (error reduced by over 50%) and shorter task completion times, while substantially lowering operator muscle strain and discomfort. These findings validate the importance of kinematic optimization (such as added articulations and tremor filtering) and human-centered ergonomic design in enhancing the performance of robot-assisted surgery. The insights from this work can guide the development of next-generation surgical robots that improve surgical outcomes and ergonomics for the operating team. 

**Abstract (ZH)**: 微创手术中机器人辅助可以大大提高手术精确度并减轻 Surgeon 的疲劳。本文针对高精度任务，对腹腔镜手术机器人臂的运动学及人机工程学设计原则进行了深入研究。我们提出了一种7自由度（7-DOF）机器人臂系统，该系统在器械插入点处设置了远端中心运动（RCM），并考虑了人机工程学，以提高外科医生的操作体验。该设计在通用机器人平台上实现，并通过一系列模拟手术任务评估了与常规手动腹腔镜手术相比的目标精度、任务效率和外科医生舒适度。实验结果表明，优化的机器人设计显著提高了目标精度（误差减少超过50%）、缩短了任务完成时间，并大幅降低了操作者的肌肉疲劳和不适。这些发现证明了运动学优化（如增加关节和震颤过滤）和以人机工程学为中心的设计对于提高手术辅助手术表现的重要性。本研究的见解可以指导下一代手术机器人的开发，从而提高手术结果和手术团队的人体工程学。 

---
# Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models 

**Title (ZH)**: 使用3D生成AI和视觉语言模型进行文本驱动的多组件对象机器人装配 

**Authors**: Alexander Htet Kyaw, Richa Gupta, Dhruv Shah, Anoop Sinha, Kory Mathewson, Stefanie Pender, Sachin Chitta, Yotto Koga, Faez Ahmed, Lawrence Sass, Randall Davis  

**Link**: [PDF](https://arxiv.org/pdf/2511.02162)  

**Abstract**: Advances in 3D generative AI have enabled the creation of physical objects from text prompts, but challenges remain in creating objects involving multiple component types. We present a pipeline that integrates 3D generative AI with vision-language models (VLMs) to enable the robotic assembly of multi-component objects from natural language. Our method leverages VLMs for zero-shot, multi-modal reasoning about geometry and functionality to decompose AI-generated meshes into multi-component 3D models using predefined structural and panel components. We demonstrate that a VLM is capable of determining which mesh regions need panel components in addition to structural components, based on object functionality. Evaluation across test objects shows that users preferred the VLM-generated assignments 90.6% of the time, compared to 59.4% for rule-based and 2.5% for random assignment. Lastly, the system allows users to refine component assignments through conversational feedback, enabling greater human control and agency in making physical objects with generative AI and robotics. 

**Abstract (ZH)**: 基于视觉语言模型的3D生成AI在自然语言驱动的多组件物体机器人组装中的应用 

---
# Census-Based Population Autonomy For Distributed Robotic Teaming 

**Title (ZH)**: 基于人口普查的分布式机器人团队自主性 

**Authors**: Tyler M. Paine, Anastasia Bizyaeva, Michael R. Benjamin  

**Link**: [PDF](https://arxiv.org/pdf/2511.02147)  

**Abstract**: Collaborating teams of robots show promise due in their ability to complete missions more efficiently and with improved robustness, attributes that are particularly useful for systems operating in marine environments. A key issue is how to model, analyze, and design these multi-robot systems to realize the full benefits of collaboration, a challenging task since the domain of multi-robot autonomy encompasses both collective and individual behaviors. This paper introduces a layered model of multi-robot autonomy that uses the principle of census, or a weighted count of the inputs from neighbors, for collective decision-making about teaming, coupled with multi-objective behavior optimization for individual decision-making about actions. The census component is expressed as a nonlinear opinion dynamics model and the multi-objective behavior optimization is accomplished using interval programming. This model can be reduced to recover foundational algorithms in distributed optimization and control, while the full model enables new types of collective behaviors that are useful in real-world scenarios. To illustrate these points, a new method for distributed optimization of subgroup allocation is introduced where robots use a gradient descent algorithm to minimize portions of the cost functions that are locally known, while being influenced by the opinion states from neighbors to account for the unobserved costs. With this method the group can collectively use the information contained in the Hessian matrix of the total global cost. The utility of this model is experimentally validated in three categorically different experiments with fleets of autonomous surface vehicles: an adaptive sampling scenario, a high value unit protection scenario, and a competitive game of capture the flag. 

**Abstract (ZH)**: 协作机器人团队显示出潜力，因为它们能够更高效地完成任务并具有更好的鲁棒性，这些特性特别适用于海洋环境中的系统。一个关键问题是如何建模、分析和设计这些多机器人系统以实现协作的全部益处，这是一个具有挑战性的工作，因为多机器人自主性的领域涵盖了集体行为和个体行为。本文介绍了一种分层的多机器人自主性模型，该模型使用邻近输入加权计数（即普查）的原则进行集体决策，确定团队合作方式，并结合多目标行为优化进行个体行动决策。普查部分表达为非线性意见动力学模型，而多目标行为优化则通过区间规划实现。该模型可以简化以恢复分布式优化和控制的基石算法，而在完整模型中则能够实现新的集体行为，这在实际场景中很有用。为了说明这一点，提出了一种新的分布式优化子组分配方法，其中机器人使用梯度下降算法最小化局部已知的成本函数部分，同时受到邻居意见状态的影响来考虑未观察到的成本。通过这种方法，群体可以共同利用全局总成本海森矩阵中包含的信息。该模型在三种不同类别的一系列自主水面航行器舰队实验中得到了实验验证：自适应采样场景、高价值单元保护场景以及夺旗竞技游戏。 

---
# A Step Toward World Models: A Survey on Robotic Manipulation 

**Title (ZH)**: 迈向世界模型的一步：机器人操控综述 

**Authors**: Peng-Fei Zhang, Ying Cheng, Xiaofan Sun, Shijie Wang, Lei Zhu, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.02097)  

**Abstract**: Autonomous agents are increasingly expected to operate in complex, dynamic, and uncertain environments, performing tasks such as manipulation, navigation, and decision-making. Achieving these capabilities requires agents to understand the underlying mechanisms and dynamics of the world, moving beyond purely reactive control or simple replication of observed states. This motivates the development of world models as internal representations that encode environmental states, capture dynamics, and enable prediction, planning, and reasoning. Despite growing interest, the definition, scope, architectures, and essential capabilities of world models remain ambiguous. In this survey, rather than directly imposing a fixed definition and limiting our scope to methods explicitly labeled as world models, we examine approaches that exhibit the core capabilities of world models through a review of methods in robotic manipulation. We analyze their roles across perception, prediction, and control, identify key challenges and solutions, and distill the core components, capabilities, and functions that a real world model should possess. Building on this analysis, we aim to outline a roadmap for developing generalizable and practical world models for robotics. 

**Abstract (ZH)**: 自主代理日益被期望在复杂、动态和不确定性环境中执行任务，如操作、导航和决策。实现这些能力要求代理理解世界的内在机制和动态，超越纯粹的反应控制或简单的状态复制。这激发了世界模型的开发，作为一种内部表示，能够编码环境状态、捕捉动态并支持预测、规划和推理。尽管兴趣日益增长，但世界模型的定义、范围、架构和基本能力仍不清楚。在本文综述中，我们不直接施加固定定义，也不将范围局限于明确标记为世界模型的方法，而是通过机器人操作方法的回顾来探讨展示世界模型核心能力的途径。我们分析这些方法在感知、预测和控制中的作用，识别关键挑战和解决方案，并提炼一个真实世界模型应具备的核心组件、能力和功能。基于这一分析，我们旨在为机器人开发可扩展和实用的世界模型制定路线图。 

---
# TACO: Trajectory-Aware Controller Optimization for Quadrotors 

**Title (ZH)**: TACO: 轨迹感知旋翼无人机控制器优化 

**Authors**: Hersh Sanghvi, Spencer Folk, Vijay Kumar, Camillo Jose Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2511.02060)  

**Abstract**: Controller performance in quadrotor trajectory tracking depends heavily on parameter tuning, yet standard approaches often rely on fixed, manually tuned parameters that sacrifice task-specific performance. We present Trajectory-Aware Controller Optimization (TACO), a framework that adapts controller parameters online based on the upcoming reference trajectory and current quadrotor state. TACO employs a learned predictive model and a lightweight optimization scheme to optimize controller gains in real time with respect to a broad class of trajectories, and can also be used to adapt trajectories to improve dynamic feasibility while respecting smoothness constraints. To enable large-scale training, we also introduce a parallelized quadrotor simulator supporting fast data collection on diverse trajectories. Experiments on a variety of trajectory types show that TACO outperforms conventional, static parameter tuning while operating orders of magnitude faster than black-box optimization baselines, enabling practical real-time deployment on a physical quadrotor. Furthermore, we show that adapting trajectories using TACO significantly reduces the tracking error obtained by the quadrotor. 

**Abstract (ZH)**: 四旋翼轨迹跟踪中控制器性能高度依赖于参数调整，而传统方法往往依赖于固定的手动调整参数，牺牲了任务特定的性能。我们提出了轨迹感知控制器优化（TACO）框架，该框架能够根据即将到来的参考轨迹和当前四旋翼状态在线调整控制器参数。TACO 使用了学习预测模型和轻量级优化方案，能够实时针对广泛类别的轨迹优化控制器增益，并且还可以用于适应轨迹以改进动态可行性同时遵守平滑性约束。为了实现大规模训练，我们还引入了一个并行四旋翼模拟器，支持在多样化的轨迹上快速收集数据。实验结果显示，TACO 在各种轨迹类型上表现出色，性能远超黑盒优化基线，能够在物理四旋翼上实现实用的实时部署。此外，我们展示了使用 TACO 调整轨迹可以显著减少四旋翼的跟踪误差。 

---
# TurboMap: GPU-Accelerated Local Mapping for Visual SLAM 

**Title (ZH)**: TurboMap: GPU 加速的局部_mapping 用于视觉 SLAM 

**Authors**: Parsa Hosseininejad, Kimia Khabiri, Shishir Gopinath, Soudabeh Mohammadhashemi, Karthik Dantu, Steven Y. Ko  

**Link**: [PDF](https://arxiv.org/pdf/2511.02036)  

**Abstract**: This paper presents TurboMap, a GPU-accelerated and CPU-optimized local mapping module for visual SLAM systems. We identify key performance bottlenecks in the local mapping process for visual SLAM and address them through targeted GPU and CPU optimizations. Specifically, we offload map point triangulation and fusion to the GPU, accelerate redundant keyframe culling on the CPU, and integrate a GPU-accelerated solver to speed up local bundle adjustment. Our implementation is built on top of ORB-SLAM3 and leverages CUDA for GPU programming. The experimental results show that TurboMap achieves an average speedup of 1.3x in the EuRoC dataset and 1.6x in the TUM-VI dataset in the local mapping module, on both desktop and embedded platforms, while maintaining the accuracy of the original system. 

**Abstract (ZH)**: TurboMap：一种加速视觉SLAM系统局部映射的GPU加速和CPU优化模块 

---
# Stein-based Optimization of Sampling Distributions in Model Predictive Path Integral Control 

**Title (ZH)**: 基于Stein优化的模型预测路径积分控制采样分布优化 

**Authors**: Jace Aldrich, Odest Chadwicke Jenkins  

**Link**: [PDF](https://arxiv.org/pdf/2511.02015)  

**Abstract**: This paper presents a novel method for Model Predictive Path Integral (MPPI) control that optimizes sample generation towards an optimal trajectory through Stein Variational Gradient Descent (SVGD). MPPI is traditionally reliant on randomly sampled trajectories, often by a Gaussian distribution. The result can lead to sample deprivation, under-representing the space of possible trajectories, and yield suboptimal results. Through introducing SVGD updates in between MPPI environment steps, we present Stein-Optimized Path-Integral Inference (SOPPI), an MPPI/SVGD algorithm that can dynamically update noise distributions at runtime to shape a more optimal representation without an excessive increase in computational requirements. We demonstrate the efficacy of our method systems ranging from a Cart-Pole to a two-dimensional bipedal walking task, indicating improved performance above standard MPPI across a range of hyper-parameters and demonstrate feasibility at lower particle counts. We discuss the applicability of this MPPI/SVGD method to higher degree-of-freedom systems, as well as its potential to new developments in state-of-the-art differentiable simulators. 

**Abstract (ZH)**: 基于Stein优化路径积分推理的新型MPPI控制方法：通过SVGD优化样本生成 

---
# TRACE: Textual Reasoning for Affordance Coordinate Extraction 

**Title (ZH)**: TRACE: 文本推理驱动的利用方式坐标抽取 

**Authors**: Sangyun Park, Jin Kim, Yuchen Cui, Matthew S. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2511.01999)  

**Abstract**: Vision-Language Models (VLMs) struggle to translate high-level instructions into the precise spatial affordances required for robotic manipulation. While visual Chain-of-Thought (CoT) methods exist, they are often computationally intensive. In this work, we introduce TRACE (Textual Reasoning for Affordance Coordinate Extraction), a novel methodology that integrates a textual Chain of Reasoning (CoR) into the affordance prediction process. We use this methodology to create the TRACE dataset, a large-scale collection created via an autonomous pipeline that pairs instructions with explicit textual rationales. By fine-tuning a VLM on this data, our model learns to externalize its spatial reasoning before acting. Our experiments show that our TRACE-tuned model achieves state-of-the-art performance, reaching 48.1% accuracy on the primary Where2Place (W2P) benchmark (a 9.6% relative improvement) and 55.0% on the more challenging W2P(h) subset. Crucially, an ablation study demonstrates that performance scales directly with the amount of reasoning data used, confirming the CoR's effectiveness. Furthermore, analysis of the model's attention maps reveals an interpretable reasoning process where focus shifts dynamically across reasoning steps. This work shows that training VLMs to generate a textual CoR is an effective and robust strategy for enhancing the precision, reliability, and interpretability of VLM-based robot control. Our dataset and code are available at this https URL 

**Abstract (ZH)**: 基于文本的推理以提升视觉语言模型在机器人操控中的精确性、可靠性和可解释性 

---
# Many-vs-Many Missile Guidance via Virtual Targets 

**Title (ZH)**: 多对多导弹制导通过虚拟目标 

**Authors**: Marc Schneider, Walter Fichter  

**Link**: [PDF](https://arxiv.org/pdf/2511.02526)  

**Abstract**: This paper presents a novel approach to many-vs-many missile guidance using virtual targets (VTs) generated by a Normalizing Flows-based trajectory predictor. Rather than assigning n interceptors directly to m physical targets through conventional weapon target assignment algorithms, we propose a centralized strategy that constructs n VT trajectories representing probabilistic predictions of maneuvering target behavior. Each interceptor is guided toward its assigned VT using Zero-Effort-Miss guidance during midcourse flight, transitioning to Proportional Navigation guidance for terminal interception. This approach treats many-vs-many engagements as many-vs-distribution scenarios, exploiting numerical superiority (n > m) by distributing interceptors across diverse trajectory hypotheses rather than pursuing identical deterministic predictions. Monte Carlo simulations across various target-interceptor configurations (1-6 targets, 1-8 interceptors) demonstrate that the VT method matches or exceeds baseline straight-line prediction performance by 0-4.1% when n = m, with improvements increasing to 5.8-14.4% when n > m. The results confirm that probabilistic VTs enable effective exploitation of numerical superiority, significantly increasing interception probability in many-vs-many scenarios. 

**Abstract (ZH)**: 基于归一化流的虚拟目标导向的多对多导弹制导新方法 

---
# Keeping it Local, Tiny and Real: Automated Report Generation on Edge Computing Devices for Mechatronic-Based Cognitive Systems 

**Title (ZH)**: 保持局部性、小型化和真实性：基于边缘计算设备的机电认知系统自动化报告生成 

**Authors**: Nicolas Schuler, Lea Dewald, Jürgen Graf  

**Link**: [PDF](https://arxiv.org/pdf/2511.02507)  

**Abstract**: Recent advancements in Deep Learning enable hardware-based cognitive systems, that is, mechatronic systems in general and robotics in particular with integrated Artificial Intelligence, to interact with dynamic and unstructured environments. While the results are impressive, the application of such systems to critical tasks like autonomous driving as well as service and care robotics necessitate the evaluation of large amount of heterogeneous data. Automated report generation for Mobile Robotics can play a crucial role in facilitating the evaluation and acceptance of such systems in various domains. In this paper, we propose a pipeline for generating automated reports in natural language utilizing various multi-modal sensors that solely relies on local models capable of being deployed on edge computing devices, thus preserving the privacy of all actors involved and eliminating the need for external services. In particular, we evaluate our implementation on a diverse dataset spanning multiple domains including indoor, outdoor and urban environments, providing quantitative as well as qualitative evaluation results. Various generated example reports and other supplementary materials are available via a public repository. 

**Abstract (ZH)**: Recent advancements in深度学习使基于硬件的认知系统，即一般而言的机电系统和特别而言的机器人技术，能够集成人工智能与动态且非结构化的环境进行交互。尽管成果令人印象深刻，但在自动驾驶等关键任务以及服务和护理机器人领域的应用要求对大量异构数据进行评估。移动机器人自动生成报告在促进此类系统在各个领域的评估与接受中扮演着关键角色。在本文中，我们提出了一种管道，利用多种多模态传感器生成自然语言报告，该管道仅依赖于可在边缘计算设备上部署的本地模型，从而保护所有参与者的隐私并消除对外部服务的依赖。特别地，我们在涵盖多个领域的多样性数据集上评估了我们的实现，包括室内、室外和城市环境，提供了定量和定性评估结果。生成的各种示例报告及其他补充材料通过公共库提供。 

---
# From the Laboratory to Real-World Application: Evaluating Zero-Shot Scene Interpretation on Edge Devices for Mobile Robotics 

**Title (ZH)**: 从实验室到实际应用：在移动机器人边缘设备上评估零样本场景解释 

**Authors**: Nicolas Schuler, Lea Dewald, Nick Baldig, Jürgen Graf  

**Link**: [PDF](https://arxiv.org/pdf/2511.02427)  

**Abstract**: Video Understanding, Scene Interpretation and Commonsense Reasoning are highly challenging tasks enabling the interpretation of visual information, allowing agents to perceive, interact with and make rational decisions in its environment. Large Language Models (LLMs) and Visual Language Models (VLMs) have shown remarkable advancements in these areas in recent years, enabling domain-specific applications as well as zero-shot open vocabulary tasks, combining multiple domains. However, the required computational complexity poses challenges for their application on edge devices and in the context of Mobile Robotics, especially considering the trade-off between accuracy and inference time. In this paper, we investigate the capabilities of state-of-the-art VLMs for the task of Scene Interpretation and Action Recognition, with special regard to small VLMs capable of being deployed to edge devices in the context of Mobile Robotics. The proposed pipeline is evaluated on a diverse dataset consisting of various real-world cityscape, on-campus and indoor scenarios. The experimental evaluation discusses the potential of these small models on edge devices, with particular emphasis on challenges, weaknesses, inherent model biases and the application of the gained information. Supplementary material is provided via the following repository: this https URL 

**Abstract (ZH)**: 视频理解、场景解释和常识推理是高度具有挑战性的任务，使视觉信息的解释成为可能，使智能体能够在其环境中感知、交互并做出理性的决策。大型语言模型（LLMs）和视觉语言模型（VLMs）近年来在这些领域取得了显著的进步，不仅实现了领域特定的应用，还能完成零样本的开放词汇任务，结合了多个领域。然而，所需的计算复杂性对这些模型在边缘设备上的应用以及在移动机器人领域的应用构成了挑战，特别是在准确性和推理时间之间的权衡方面。在本文中，我们研究了最先进的VLMs在场景解释和动作识别任务中的能力，特别关注能够在移动机器人领域部署到边缘设备的较小VLMs。提出的管道在包含各种真实世界城市景观、校园和室内场景的多样化数据集上进行了评估。实验评估讨论了这些小型模型在边缘设备上的潜力，特别是针对挑战、弱点、模型固有的偏差以及获得的信息的应用进行了重点讨论。补充材料可通过以下repository获取：this https URL。 

---
# Synthetic Crop-Weed Image Generation and its Impact on Model Generalization 

**Title (ZH)**: 合成作物-杂草图像生成及其对模型泛化能力的影响 

**Authors**: Garen Boyadjian, Cyrille Pierre, Johann Laconte, Riccardo Bertoglio  

**Link**: [PDF](https://arxiv.org/pdf/2511.02417)  

**Abstract**: Precise semantic segmentation of crops and weeds is necessary for agricultural weeding robots. However, training deep learning models requires large annotated datasets, which are costly to obtain in real fields. Synthetic data can reduce this burden, but the gap between simulated and real images remains a challenge. In this paper, we present a pipeline for procedural generation of synthetic crop-weed images using Blender, producing annotated datasets under diverse conditions of plant growth, weed density, lighting, and camera angle. We benchmark several state-of-the-art segmentation models on synthetic and real datasets and analyze their cross-domain generalization. Our results show that training on synthetic images leads to a sim-to-real gap of 10%, surpassing previous state-of-the-art methods. Moreover, synthetic data demonstrates good generalization properties, outperforming real datasets in cross-domain scenarios. These findings highlight the potential of synthetic agricultural datasets and support hybrid strategies for more efficient model training. 

**Abstract (ZH)**: 作物和杂草的精准语义分割对于农业除草机器人至关重要。然而，训练深度学习模型需要大量的标注数据集，而在实际田地中获得这些数据的成本很高。合成数据可以减轻这一负担，但模拟和真实图像之间的差距仍然是一个挑战。在本文中，我们提出了一种使用Blender进行程序化生成合成作物-杂草图像的管道，生成在不同植物生长条件、杂草密度、光照和相机角度下的标注数据集。我们在合成数据集和真实数据集上benchmark了几种最先进的分割模型，并分析了它们在跨领域泛化的性能。我们的结果表明，使用合成图像进行训练导致的模拟到现实世界的差距为10%，超过了之前的最先进的方法。此外，合成数据展示了良好的跨领域泛化特性，在跨领域场景中优于真实数据集。这些发现突显了合成农业数据集的潜力，并支持混合策略以实现更高效的模型训练。 

---
# Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization 

**Title (ZH)**: Cycle-Sync：通过增强的循环一致同步实现稳健的全局相机姿态估计 

**Authors**: Shaohan Li, Yunpeng Shi, Gilad Lerman  

**Link**: [PDF](https://arxiv.org/pdf/2511.02329)  

**Abstract**: We introduce Cycle-Sync, a robust and global framework for estimating camera poses (both rotations and locations). Our core innovation is a location solver that adapts message-passing least squares (MPLS) -- originally developed for group synchronization -- to camera location estimation. We modify MPLS to emphasize cycle-consistent information, redefine cycle consistencies using estimated distances from previous iterations, and incorporate a Welsch-type robust loss. We establish the strongest known deterministic exact-recovery guarantee for camera location estimation, showing that cycle consistency alone -- without access to inter-camera distances -- suffices to achieve the lowest sample complexity currently known. To further enhance robustness, we introduce a plug-and-play outlier rejection module inspired by robust subspace recovery, and we fully integrate cycle consistency into MPLS for rotation synchronization. Our global approach avoids the need for bundle adjustment. Experiments on synthetic and real datasets show that Cycle-Sync consistently outperforms leading pose estimators, including full structure-from-motion pipelines with bundle adjustment. 

**Abstract (ZH)**: Cycle-Sync：一种稳健且全局的相机姿态估计框架 

---
# Path-Coordinated Continual Learning with Neural Tangent Kernel-Justified Plasticity: A Theoretical Framework with Near State-of-the-Art Performance 

**Title (ZH)**: 路径协调连续学习：基于神经瞬时核证明的可塑性理论框架及其接近于顶级性能的表现 

**Authors**: Rathin Chandra Shit  

**Link**: [PDF](https://arxiv.org/pdf/2511.02025)  

**Abstract**: Catastrophic forgetting is one of the fundamental issues of continual learning because neural networks forget the tasks learned previously when trained on new tasks. The proposed framework is a new path-coordinated framework of continual learning that unites the Neural Tangent Kernel (NTK) theory of principled plasticity bounds, statistical validation by Wilson confidence intervals, and evaluation of path quality by the use of multiple metrics. Experimental evaluation shows an average accuracy of 66.7% at the cost of 23.4% catastrophic forgetting on Split-CIFAR10, a huge improvement over the baseline and competitive performance achieved, which is very close to state-of-the-art results. Further, it is found out that NTK condition numbers are predictive indicators of learning capacity limits, showing the existence of a critical threshold at condition number $>10^{11}$. It is interesting to note that the proposed strategy shows a tendency of lowering forgetting as the sequence of tasks progresses (27% to 18%), which is a system stabilization. The framework validates 80% of discovered paths with a rigorous statistical guarantee and maintains 90-97% retention on intermediate tasks. The core capacity limits of the continual learning environment are determined in the analysis, and actionable insights to enhance the adaptive regularization are offered. 

**Abstract (ZH)**: 连续学习中灾难性遗忘问题是神经网络在学习新任务时忘记先前任务的一个根本问题。提出的框架是一种新的路径协调连续学习框架，结合了原理性可塑性边界理论中的神经切线核（NTK）理论、Wilson置信区间统计验证以及通过多种指标评估路径质量。实验评估显示，在Split-CIFAR10上的平均准确率为66.7%，灾难性遗忘成本为23.4%，显著优于基线并与达到的性能具有竞争力，接近当前最先进的结果。此外，研究发现NTK条件数是学习能力限制的预测指标，显示出条件数>10^11时存在一个临界阈值。有趣的是，所提出的策略显示出随着任务序列的推进降低遗忘的趋势（从27%降至18%），这是一种系统稳定。框架以严格的统计保证验证了所发现路径的80%，在中间任务上保持90-97%的保留率。分析确定了连续学习环境的核心能力限制，并提供了增强自适应正则化的可操作建议。 

---
# iFlyBot-VLA Technical Report 

**Title (ZH)**: iFlyBot-VLA 技术报告 

**Authors**: Yuan Zhang, Chenyu Xue, Wenjie Xu, Chao Ji, Jiajia wu, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2511.01914)  

**Abstract**: We introduce iFlyBot-VLA, a large-scale Vision-Language-Action (VLA) model trained under a novel framework. The main contributions are listed as follows: (1) a latent action model thoroughly trained on large-scale human and robotic manipulation videos; (2) a dual-level action representation framework that jointly supervises both the Vision-Language Model (VLM) and the action expert during training; (3) a mixed training strategy that combines robot trajectory data with general QA and spatial QA datasets, effectively enhancing the 3D perceptual and reasoning capabilities of the VLM backbone. Specifically, the VLM is trained to predict two complementary forms of actions: latent actions, derived from our latent action model pretrained on cross-embodiment manipulation data, which capture implicit high-level intentions; and structured discrete action tokens, obtained through frequency-domain transformations of continuous control signals, which encode explicit low-level dynamics. This dual supervision aligns the representation spaces of language, vision, and action, enabling the VLM to directly contribute to action generation. Experimental results on the LIBERO Franka benchmark demonstrate the superiority of our frame-work, while real-world evaluations further show that iFlyBot-VLA achieves competitive success rates across diverse and challenging manipulation tasks. Furthermore, we plan to open-source a portion of our self-constructed dataset to support future research in the community 

**Abstract (ZH)**: 我们介绍了一种新型框架下大规模训练的iFlyBot-VLA视觉-语言-行动模型。主要贡献如下：(1) 一个全面训练于大规模人类和机器人操作视频的潜在行动模型；(2) 一个两级行动表示框架，在训练过程中联合监督视觉-语言模型(VLM)和行动专家；(3) 一种混合训练策略，结合机器人轨迹数据与通用QA和空间QA数据集，有效提升VLM主干的三维感知和推理能力。具体而言，VLM 被训练预测两种互补形式的行动：潜在行动，从预训练于跨体态操作数据的潜在行动模型中提取，捕捉隐含的高层意图；以及通过连续控制信号的频域变换获得的结构化离散行动标记，编码显式的低层动态。这种联合监督使语言、视觉和行动的表示空间得以对齐，从而使VLM 直接参与到行动生成中。在LIBERO Franka基准测试上的实验结果表明了我们框架的优越性，而实际应用评估进一步证明了iFlyBot-VLA 在多种挑战性操作任务中取得竞争力的成功率。此外，我们计划开源部分自构建的数据集以支持社区未来的研究。 

---
