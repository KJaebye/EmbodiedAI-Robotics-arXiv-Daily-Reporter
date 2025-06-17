# Critical Insights about Robots for Mental Wellbeing 

**Title (ZH)**: 关于用于心理健康的人工智能机器人的重要见解 

**Authors**: Guy Laban, Micol Spitale, Minja Axelsson, Nida Itrat Abbasi, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2506.13739)  

**Abstract**: Social robots are increasingly being explored as tools to support emotional wellbeing, particularly in non-clinical settings. Drawing on a range of empirical studies and practical deployments, this paper outlines six key insights that highlight both the opportunities and challenges in using robots to promote mental wellbeing. These include (1) the lack of a single, objective measure of wellbeing, (2) the fact that robots don't need to act as companions to be effective, (3) the growing potential of virtual interactions, (4) the importance of involving clinicians in the design process, (5) the difference between one-off and long-term interactions, and (6) the idea that adaptation and personalization are not always necessary for positive outcomes. Rather than positioning robots as replacements for human therapists, we argue that they are best understood as supportive tools that must be designed with care, grounded in evidence, and shaped by ethical and psychological considerations. Our aim is to inform future research and guide responsible, effective use of robots in mental health and wellbeing contexts. 

**Abstract (ZH)**: 社会机器人在促进心理健康和福祉方面的机会与挑战：基于实证研究与实际部署的六点洞见 

---
# What Matters in Learning from Large-Scale Datasets for Robot Manipulation 

**Title (ZH)**: 大规模数据集用于机器人操作学习中值得关注的问题 

**Authors**: Vaibhav Saxena, Matthew Bronars, Nadun Ranawaka Arachchige, Kuancheng Wang, Woo Chul Shin, Soroush Nasiriany, Ajay Mandlekar, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13536)  

**Abstract**: Imitation learning from large multi-task demonstration datasets has emerged as a promising path for building generally-capable robots. As a result, 1000s of hours have been spent on building such large-scale datasets around the globe. Despite the continuous growth of such efforts, we still lack a systematic understanding of what data should be collected to improve the utility of a robotics dataset and facilitate downstream policy learning. In this work, we conduct a large-scale dataset composition study to answer this question. We develop a data generation framework to procedurally emulate common sources of diversity in existing datasets (such as sensor placements and object types and arrangements), and use it to generate large-scale robot datasets with controlled compositions, enabling a suite of dataset composition studies that would be prohibitively expensive in the real world. We focus on two practical settings: (1) what types of diversity should be emphasized when future researchers collect large-scale datasets for robotics, and (2) how should current practitioners retrieve relevant demonstrations from existing datasets to maximize downstream policy performance on tasks of interest. Our study yields several critical insights -- for example, we find that camera poses and spatial arrangements are crucial dimensions for both diversity in collection and alignment in retrieval. In real-world robot learning settings, we find that not only do our insights from simulation carry over, but our retrieval strategies on existing datasets such as DROID allow us to consistently outperform existing training strategies by up to 70%. More results at this https URL 

**Abstract (ZH)**: 从大规模多任务演示数据集中学习模仿：构建通用机器人的一条有前途的道路 

---
# Role of Uncertainty in Model Development and Control Design for a Manufacturing Process 

**Title (ZH)**: 制造过程建模与控制设计中的不确定性作用 

**Authors**: Rongfei Li, Francis Assadian  

**Link**: [PDF](https://arxiv.org/pdf/2506.12273)  

**Abstract**: The use of robotic technology has drastically increased in manufacturing in the 21st century. But by utilizing their sensory cues, humans still outperform machines, especially in the micro scale manufacturing, which requires high-precision robot manipulators. These sensory cues naturally compensate for high level of uncertainties that exist in the manufacturing environment. Uncertainties in performing manufacturing tasks may come from measurement noise, model inaccuracy, joint compliance (e.g., elasticity) etc. Although advanced metrology sensors and high-precision microprocessors, which are utilized in nowadays robots, have compensated for many structural and dynamic errors in robot positioning, but a well-designed control algorithm still works as a comparable and cheaper alternative to reduce uncertainties in automated manufacturing. Our work illustrates that a multi-robot control system can reduce various uncertainties to a great amount. 

**Abstract (ZH)**: 21世纪机器人技术在制造领域中的应用大幅增加。但在微观规模制造中，利用人类的感觉 cues，人类仍然超越机器，特别是在需要高精度机器人操作的场合。这些感觉 cues 自然地补偿了制造环境中存在的高不确定性。执行制造任务时的不确定性可能来自测量噪声、模型不准确、关节顺应性（如弹性）等。虽然现代机器人利用了先进的计量传感器和高精度微处理器来补偿许多结构和动态误差，但精心设计的控制算法仍然是减少自动化制造中不确定性的经济且有效的替代方案。我们的研究展示了一个多机器人控制系统可以大幅减少各种不确定性。 

---
# Using Behavior Trees in Risk Assessment 

**Title (ZH)**: 使用行为树进行风险评估 

**Authors**: Razan Ghzouli, Atieh Hanna, Endre Erös, Rebekka Wohlrab  

**Link**: [PDF](https://arxiv.org/pdf/2506.12089)  

**Abstract**: Cyber-physical production systems increasingly involve collaborative robotic missions, requiring more demand for robust and safe missions. Industries rely on risk assessments to identify potential failures and implement measures to mitigate their risks. Although it is recommended to conduct risk assessments early in the design of robotic missions, the state of practice in the industry is different. Safety experts often struggle to completely understand robotics missions at the early design stages of projects and to ensure that the output of risk assessments is adequately considered during implementation.
This paper presents a design science study that conceived a model-based approach for early risk assessment in a development-centric way. Our approach supports risk assessment activities by using the behavior-tree model. We evaluated the approach together with five practitioners from four companies. Our findings highlight the potential of the behavior-tree model in supporting early identification, visualisation, and bridging the gap between code implementation and risk assessments' outputs. This approach is the first attempt to use the behavior-tree model to support risk assessment; thus, the findings highlight the need for further development. 

**Abstract (ZH)**: 基于模型的早期风险评估设计科学研究：行为树模型在机器人任务设计中的应用 

---
# Parallel Branch Model Predictive Control on GPUs 

**Title (ZH)**: GPU上并行分支模型预测控制 

**Authors**: Luyao Zhang, Chenghuai Lin, Sergio Grammatico  

**Link**: [PDF](https://arxiv.org/pdf/2506.13624)  

**Abstract**: We present a parallel GPU-accelerated solver for branch Model Predictive Control problems. Based on iterative LQR methods, our solver exploits the tree-sparse structure and implements temporal parallelism using the parallel scan algorithm. Consequently, the proposed solver enables parallelism across both the prediction horizon and the scenarios. In addition, we utilize an augmented Lagrangian method to handle general inequality constraints. We compare our solver with state-of-the-art numerical solvers in two automated driving applications. The numerical results demonstrate that, compared to CPU-based solvers, our solver achieves competitive performance for problems with short horizons and small-scale trees, while outperforming other solvers on large-scale problems. 

**Abstract (ZH)**: 我们提出了一种并行GPU加速的分支模型预测控制求解器。基于迭代LQR方法，该求解器利用树稀疏结构并采用并行扫描算法实现时间上的并行ism。因此，所提出的求解器能够在预测 horizon 和场景之间实现并行ism。此外，我们使用增广拉格朗日方法处理一般不等式约束。我们在两个自动驾驶应用中将该求解器与最先进的数值求解器进行比较。数值结果表明，对于短 horizon 和小规模树的问题，与基于CPU的求解器相比，该求解器具有竞争力的性能，而在大规模问题上则优于其他求解器。 

---
# Can you see how I learn? Human observers' inferences about Reinforcement Learning agents' learning processes 

**Title (ZH)**: 你能看出我是如何学习的？人类观察者对强化学习代理学习过程的推断。 

**Authors**: Bernhard Hilpert, Muhan Hou, Kim Baraka, Joost Broekens  

**Link**: [PDF](https://arxiv.org/pdf/2506.13583)  

**Abstract**: Reinforcement Learning (RL) agents often exhibit learning behaviors that are not intuitively interpretable by human observers, which can result in suboptimal feedback in collaborative teaching settings. Yet, how humans perceive and interpret RL agent's learning behavior is largely unknown. In a bottom-up approach with two experiments, this work provides a data-driven understanding of the factors of human observers' understanding of the agent's learning process. A novel, observation-based paradigm to directly assess human inferences about agent learning was developed. In an exploratory interview study (\textit{N}=9), we identify four core themes in human interpretations: Agent Goals, Knowledge, Decision Making, and Learning Mechanisms. A second confirmatory study (\textit{N}=34) applied an expanded version of the paradigm across two tasks (navigation/manipulation) and two RL algorithms (tabular/function approximation). Analyses of 816 responses confirmed the reliability of the paradigm and refined the thematic framework, revealing how these themes evolve over time and interrelate. Our findings provide a human-centered understanding of how people make sense of agent learning, offering actionable insights for designing interpretable RL systems and improving transparency in Human-Robot Interaction. 

**Abstract (ZH)**: 强化学习（RL）代理的学习行为往往难以被人类观察者直观理解，这在协作教学场景中可能导致反馈不足。然而，人类如何感知和解释RL代理的学习行为尚不清楚。通过自下而上的两种实验，本研究提供了关于人类观察者理解代理学习过程的影响因素的数据驱动理解。我们开发了一种基于观察的新型范式，直接评估人类对代理学习的推断。在探索性访谈研究（N=9）中，我们识别了四种核心主题：代理目标、知识、决策制定和学习机制。在确认性研究（N=34）中，我们应用扩展后的范式，在两个任务（导航/操作）和两种RL算法（表Lookup/函数逼近）上进行。对816个响应的分析证实了该范式的可靠性，并细化了主题框架，揭示了这些主题如何随时间发展及其相互关系。我们的研究结果提供了一种以人类为中心的理解方式，说明了人们如何理解代理学习，为设计可解释的RL系统和提高人机交互的透明度提供了实用见解。 

---
# Block-wise Adaptive Caching for Accelerating Diffusion Policy 

**Title (ZH)**: 块级自适应缓存加速扩散策略 

**Authors**: Kangye Ji, Yuan Meng, Hanyun Cui, Ye Li, Shengjia Hua, Lei Chen, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13456)  

**Abstract**: Diffusion Policy has demonstrated strong visuomotor modeling capabilities, but its high computational cost renders it impractical for real-time robotic control. Despite huge redundancy across repetitive denoising steps, existing diffusion acceleration techniques fail to generalize to Diffusion Policy due to fundamental architectural and data divergences. In this paper, we propose Block-wise Adaptive Caching(BAC), a method to accelerate Diffusion Policy by caching intermediate action features. BAC achieves lossless action generation acceleration by adaptively updating and reusing cached features at the block level, based on a key observation that feature similarities vary non-uniformly across timesteps and locks. To operationalize this insight, we first propose the Adaptive Caching Scheduler, designed to identify optimal update timesteps by maximizing the global feature similarities between cached and skipped features. However, applying this scheduler for each block leads to signiffcant error surges due to the inter-block propagation of caching errors, particularly within Feed-Forward Network (FFN) blocks. To mitigate this issue, we develop the Bubbling Union Algorithm, which truncates these errors by updating the upstream blocks with signiffcant caching errors before downstream FFNs. As a training-free plugin, BAC is readily integrable with existing transformer-based Diffusion Policy and vision-language-action models. Extensive experiments on multiple robotic benchmarks demonstrate that BAC achieves up to 3x inference speedup for free. 

**Abstract (ZH)**: 块级自适应缓存(BAC):一种加速扩散政策的方法 

---
# Bridging Data-Driven and Physics-Based Models: A Consensus Multi-Model Kalman Filter for Robust Vehicle State Estimation 

**Title (ZH)**: 数据驱动与物理模型融合的共识多模型卡尔曼滤波器：稳健的车辆状态估计 

**Authors**: Farid Mafi, Ladan Khoshnevisan, Mohammad Pirani, Amir Khajepour  

**Link**: [PDF](https://arxiv.org/pdf/2506.12862)  

**Abstract**: Vehicle state estimation presents a fundamental challenge for autonomous driving systems, requiring both physical interpretability and the ability to capture complex nonlinear behaviors across diverse operating conditions. Traditional methodologies often rely exclusively on either physics-based or data-driven models, each with complementary strengths and limitations that become most noticeable during critical scenarios. This paper presents a novel consensus multi-model Kalman filter framework that integrates heterogeneous model types to leverage their complementary strengths while minimizing individual weaknesses. We introduce two distinct methodologies for handling covariance propagation in data-driven models: a Koopman operator-based linearization approach enabling analytical covariance propagation, and an ensemble-based method providing unified uncertainty quantification across model types without requiring pretraining. Our approach implements an iterative consensus fusion procedure that dynamically weighs different models based on their demonstrated reliability in current operating conditions. The experimental results conducted on an electric all-wheel-drive Equinox vehicle demonstrate performance improvements over single-model techniques, with particularly significant advantages during challenging maneuvers and varying road conditions, confirming the effectiveness and robustness of the proposed methodology for safety-critical autonomous driving applications. 

**Abstract (ZH)**: 车辆状态估计是自主驾驶系统中的一个根本挑战，要求同时具备物理可解释性和捕捉多变操作条件下的复杂非线性行为能力。传统方法通常仅依赖于物理模型或数据驱动模型，每种方法都有其互补的优势和限制，在关键场景中尤为明显。本文提出了一种新颖的共识多模型卡尔曼滤波框架，该框架集成了异构模型类型，充分利用其互补优势并最小化各自的弱点。我们介绍了两种不同的方法来处理数据驱动模型中的协方差传播：基于科廷曼算子的线性化方法以实现分析性的协方差传播，以及基于ensemble的方法以在不同模型类型中提供统一的不确定性量化，无需预先训练。我们的方法实现了一种迭代共识融合过程，该过程根据模型在当前操作条件下的可靠性动态加权。在一辆全轮驱动的电动雪佛兰Equinox车辆上的实验结果表明，与单模型技术相比，该方法在复杂的操作和变化的道路条件下表现出显著的优势，验证了所提出方法在安全关键的自主驾驶应用中的有效性和稳健性。 

---
# Trust-MARL: Trust-Based Multi-Agent Reinforcement Learning Framework for Cooperative On-Ramp Merging Control in Heterogeneous Traffic Flow 

**Title (ZH)**: 基于信任的多智能体强化学习框架：异质交通流中合作式入口匝道并线控制的信任机制 

**Authors**: Jie Pan, Tianyi Wang, Christian Claudel, Jing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12600)  

**Abstract**: Intelligent transportation systems require connected and automated vehicles (CAVs) to conduct safe and efficient cooperation with human-driven vehicles (HVs) in complex real-world traffic environments. However, the inherent unpredictability of human behaviour, especially at bottlenecks such as highway on-ramp merging areas, often disrupts traffic flow and compromises system performance. To address the challenge of cooperative on-ramp merging in heterogeneous traffic environments, this study proposes a trust-based multi-agent reinforcement learning (Trust-MARL) framework. At the macro level, Trust-MARL enhances global traffic efficiency by leveraging inter-agent trust to improve bottleneck throughput and mitigate traffic shockwave through emergent group-level coordination. At the micro level, a dynamic trust mechanism is designed to enable CAVs to adjust their cooperative strategies in response to real-time behaviors and historical interactions with both HVs and other CAVs. Furthermore, a trust-triggered game-theoretic decision-making module is integrated to guide each CAV in adapting its cooperation factor and executing context-aware lane-changing decisions under safety, comfort, and efficiency constraints. An extensive set of ablation studies and comparative experiments validates the effectiveness of the proposed Trust-MARL approach, demonstrating significant improvements in safety, efficiency, comfort, and adaptability across varying CAV penetration rates and traffic densities. 

**Abstract (ZH)**: 智能交通系统需要连接和自动化车辆（CAVs）在复杂的真实世界交通环境中与人类驾驶车辆（HVs）进行安全和高效的协同合作。然而，人类行为的固有不可预测性，特别是在高速公路入口匝道合流区等瓶颈区域，常常扰乱交通流并损害系统性能。为应对异构交通环境中合流协作的挑战，本研究提出了一种基于信任的多智能体 reinforcement 学习（Trust-MARL）框架。在宏观层面，Trust-MARL 通过利用智能体间的信任增强全局交通效率，改善瓶颈处的通行能力和缓解交通波，实现群体级别的协调。在微观层面，设计了一种动态信任机制，使CAVs能够根据与HV和其他CAVs的实时行为和历史交互调整其协同策略。此外，集成了一个基于信任的游戏理论决策模块，指导每个CAV在安全、舒适和效率约束条件下调整其协作因子并执行情境感知的变道决策。通过大量的消融研究和对比实验，验证了所提出的Trust-MARL方法的有效性，展示了在不同CAV渗透率和交通密度下显著提高的安全性、效率、舒适性和适应性。 

---
# Constrained Diffusers for Safe Planning and Control 

**Title (ZH)**: 受约束的扩散器用于安全规划与控制 

**Authors**: Jichen Zhang, Liqun Zhao, Antonis Papachristodoulou, Jack Umenberger  

**Link**: [PDF](https://arxiv.org/pdf/2506.12544)  

**Abstract**: Diffusion models have shown remarkable potential in planning and control tasks due to their ability to represent multimodal distributions over actions and trajectories. However, ensuring safety under constraints remains a critical challenge for diffusion models. This paper proposes Constrained Diffusers, a novel framework that incorporates constraints into pre-trained diffusion models without retraining or architectural modifications. Inspired by constrained optimization, we apply a constrained Langevin sampling mechanism for the reverse diffusion process that jointly optimizes the trajectory and realizes constraint satisfaction through three iterative algorithms: projected method, primal-dual method and augmented Lagrangian approaches. In addition, we incorporate discrete control barrier functions as constraints for constrained diffusers to guarantee safety in online implementation. Experiments in Maze2D, locomotion, and pybullet ball running tasks demonstrate that our proposed methods achieve constraint satisfaction with less computation time, and are competitive to existing methods in environments with static and time-varying constraints. 

**Abstract (ZH)**: 约束扩散器：一种无需重新训练或修改架构将约束整合到预训练扩散模型中的新型框架 

---
# Avoiding Obfuscation with Prover-Estimator Debate 

**Title (ZH)**: 避免混淆：证明者-估计算法辩论 

**Authors**: Jonah Brown-Cohen, Geoffrey Irving, Georgios Piliouras  

**Link**: [PDF](https://arxiv.org/pdf/2506.13609)  

**Abstract**: Training powerful AI systems to exhibit desired behaviors hinges on the ability to provide accurate human supervision on increasingly complex tasks. A promising approach to this problem is to amplify human judgement by leveraging the power of two competing AIs in a debate about the correct solution to a given problem. Prior theoretical work has provided a complexity-theoretic formalization of AI debate, and posed the problem of designing protocols for AI debate that guarantee the correctness of human judgements for as complex a class of problems as possible. Recursive debates, in which debaters decompose a complex problem into simpler subproblems, hold promise for growing the class of problems that can be accurately judged in a debate. However, existing protocols for recursive debate run into the obfuscated arguments problem: a dishonest debater can use a computationally efficient strategy that forces an honest opponent to solve a computationally intractable problem to win. We mitigate this problem with a new recursive debate protocol that, under certain stability assumptions, ensures that an honest debater can win with a strategy requiring computational efficiency comparable to their opponent. 

**Abstract (ZH)**: 训练强大的AI系统展现 desired behaviors 在很大程度上取决于能够对日益复杂的任务提供准确的人类监督。一种有前景的方法是通过让两个竞争的AI在关于给定问题正确解决方案的辩论中发挥其优势来放大人类的判断力。先前的理论工作为AI辩论提供了一个复杂性理论的形式化，并提出了设计协议以保证人类判断的正确性的问题，适用于尽可能复杂的类问题。递归辩论，其中辩论者将复杂问题分解成更简单的问题子集，有潜力扩大可以准确判断的辩论中问题的类别。然而，现有的递归辩论协议遇到了混淆性论证问题：一个不诚实的辩论者可以使用一个计算上高效的策略，迫使诚实的对手解决一个计算上不可解的问题来获胜。我们通过一个新的递归辩论协议来解决这个问题，在某些稳定假设下，该协议确保诚实的辩论者可以用一个与对手计算上效率相当的策略获胜。 

---
# The ASP-based Nurse Scheduling System at the University of Yamanashi Hospital 

**Title (ZH)**: 基于ASP的山形大学医院护士排班系统 

**Authors**: Hidetomo Nabeshima, Mutsunori Banbara, Torsten Schaub, Takehide Soh  

**Link**: [PDF](https://arxiv.org/pdf/2506.13600)  

**Abstract**: We present the design principles of a nurse scheduling system built using Answer Set Programming (ASP) and successfully deployed at the University of Yamanashi Hospital. Nurse scheduling is a complex optimization problem requiring the reconciliation of individual nurse preferences with hospital staffing needs across various wards. This involves balancing hard and soft constraints and the flexibility of interactive adjustments. While extensively studied in academia, real-world nurse scheduling presents unique challenges that go beyond typical benchmark problems and competitions. This paper details the practical application of ASP to address these challenges at the University of Yamanashi Hospital, focusing on the insights gained and the advancements in ASP technology necessary to effectively manage the complexities of real-world deployment. 

**Abstract (ZH)**: 我们基于Answer Set Programming (ASP) 设计并成功实现在山梨大学医院的护士排班系统：原则与实践 

---
# Agent Capability Negotiation and Binding Protocol (ACNBP) 

**Title (ZH)**: 代理能力协商与绑定协议（ACNBP） 

**Authors**: Ken Huang, Akram Sheriff, Vineeth Sai Narajala, Idan Habler  

**Link**: [PDF](https://arxiv.org/pdf/2506.13590)  

**Abstract**: As multi-agent systems evolve to encompass increasingly diverse and specialized agents, the challenge of enabling effective collaboration between heterogeneous agents has become paramount, with traditional agent communication protocols often assuming homogeneous environments or predefined interaction patterns that limit their applicability in dynamic, open-world scenarios. This paper presents the Agent Capability Negotiation and Binding Protocol (ACNBP), a novel framework designed to facilitate secure, efficient, and verifiable interactions between agents in heterogeneous multi-agent systems through integration with an Agent Name Service (ANS) infrastructure that provides comprehensive discovery, negotiation, and binding mechanisms. The protocol introduces a structured 10-step process encompassing capability discovery, candidate pre-screening and selection, secure negotiation phases, and binding commitment with built-in security measures including digital signatures, capability attestation, and comprehensive threat mitigation strategies, while a key innovation of ACNBP is its protocolExtension mechanism that enables backward-compatible protocol evolution and supports diverse agent architectures while maintaining security and interoperability. We demonstrate ACNBP's effectiveness through a comprehensive security analysis using the MAESTRO threat modeling framework, practical implementation considerations, and a detailed example showcasing the protocol's application in a document translation scenario, with the protocol addressing critical challenges in agent autonomy, capability verification, secure communication, and scalable agent ecosystem management. 

**Abstract (ZH)**: 面向异构多智能体系统的智能协商与绑定协议（ACNBP）：一种安全、高效且可验证的智能体交互框架 

---
# From Data-Driven to Purpose-Driven Artificial Intelligence: Systems Thinking for Data-Analytic Automation of Patient Care 

**Title (ZH)**: 从数据驱动到目标驱动的人工智能：面向患者的分析自动化系统的思维模式 

**Authors**: Daniel Anadria, Roel Dobbe, Anastasia Giachanou, Ruurd Kuiper, Richard Bartels, Íñigo Martínez de Rituerto de Troya, Carmen Zürcher, Daniel Oberski  

**Link**: [PDF](https://arxiv.org/pdf/2506.13584)  

**Abstract**: In this work, we reflect on the data-driven modeling paradigm that is gaining ground in AI-driven automation of patient care. We argue that the repurposing of existing real-world patient datasets for machine learning may not always represent an optimal approach to model development as it could lead to undesirable outcomes in patient care. We reflect on the history of data analysis to explain how the data-driven paradigm rose to popularity, and we envision ways in which systems thinking and clinical domain theory could complement the existing model development approaches in reaching human-centric outcomes. We call for a purpose-driven machine learning paradigm that is grounded in clinical theory and the sociotechnical realities of real-world operational contexts. We argue that understanding the utility of existing patient datasets requires looking in two directions: upstream towards the data generation, and downstream towards the automation objectives. This purpose-driven perspective to AI system development opens up new methodological opportunities and holds promise for AI automation of patient care. 

**Abstract (ZH)**: 在基于AI的患者护理自动化中，数据驱动建模范式的反思：一种以临床理论和实际操作背景为基础的目标导向的机器学习范式 

---
# Probabilistic Modeling of Spiking Neural Networks with Contract-Based Verification 

**Title (ZH)**: 基于合同验证的_SPIKING神经网络概率建模 

**Authors**: Zhen Yao, Elisabetta De Maria, Robert De Simone  

**Link**: [PDF](https://arxiv.org/pdf/2506.13340)  

**Abstract**: Spiking Neural Networks (SNN) are models for "realistic" neuronal computation, which makes them somehow different in scope from "ordinary" deep-learning models widely used in AI platforms nowadays. SNNs focus on timed latency (and possibly probability) of neuronal reactive activation/response, more than numerical computation of filters. So, an SNN model must provide modeling constructs for elementary neural bundles and then for synaptic connections to assemble them into compound data flow network patterns. These elements are to be parametric patterns, with latency and probability values instantiated on particular instances (while supposedly constant "at runtime"). Designers could also use different values to represent "tired" neurons, or ones impaired by external drugs, for instance. One important challenge in such modeling is to study how compound models could meet global reaction requirements (in stochastic timing challenges), provided similar provisions on individual neural bundles. A temporal language of logic to express such assume/guarantee contracts is thus needed. This may lead to formal verification on medium-sized models and testing observations on large ones. In the current article, we make preliminary progress at providing a simple model framework to express both elementary SNN neural bundles and their connecting constructs, which translates readily into both a model-checker and a simulator (both already existing and robust) to conduct experiments. 

**Abstract (ZH)**: 基于脉冲神经网络的建模与验证：从基本神经束到复合动力流网络模式的研究 

---
# Generalized Proof-Number Monte-Carlo Tree Search 

**Title (ZH)**: 广义证明数蒙特卡洛树搜索 

**Authors**: Jakub Kowalski, Dennis J. N. J. Soemers, Szymon Kosakowski, Mark H. M. Winands  

**Link**: [PDF](https://arxiv.org/pdf/2506.13249)  

**Abstract**: This paper presents Generalized Proof-Number Monte-Carlo Tree Search: a generalization of recently proposed combinations of Proof-Number Search (PNS) with Monte-Carlo Tree Search (MCTS), which use (dis)proof numbers to bias UCB1-based Selection strategies towards parts of the search that are expected to be easily (dis)proven. We propose three core modifications of prior combinations of PNS with MCTS. First, we track proof numbers per player. This reduces code complexity in the sense that we no longer need disproof numbers, and generalizes the technique to be applicable to games with more than two players. Second, we propose and extensively evaluate different methods of using proof numbers to bias the selection strategy, achieving strong performance with strategies that are simpler to implement and compute. Third, we merge our technique with Score Bounded MCTS, enabling the algorithm to prove and leverage upper and lower bounds on scores - as opposed to only proving wins or not-wins. Experiments demonstrate substantial performance increases, reaching the range of 80% for 8 out of the 11 tested board games. 

**Abstract (ZH)**: 广义证明数蒙特卡洛树搜索：一种证明数搜索与蒙特卡洛树搜索结合的扩展方法 

---
# Towards Explaining Monte-Carlo Tree Search by Using Its Enhancements 

**Title (ZH)**: 利用增强方法解释蒙特卡罗树搜索 

**Authors**: Jakub Kowalski, Mark H. M. Winands, Maksymilian Wiśniewski, Stanisław Reda, Anna Wilbik  

**Link**: [PDF](https://arxiv.org/pdf/2506.13223)  

**Abstract**: Typically, research on Explainable Artificial Intelligence (XAI) focuses on black-box models within the context of a general policy in a known, specific domain. This paper advocates for the need for knowledge-agnostic explainability applied to the subfield of XAI called Explainable Search, which focuses on explaining the choices made by intelligent search techniques. It proposes Monte-Carlo Tree Search (MCTS) enhancements as a solution to obtaining additional data and providing higher-quality explanations while remaining knowledge-free, and analyzes the most popular enhancements in terms of the specific types of explainability they introduce. So far, no other research has considered the explainability of MCTS enhancements. We present a proof-of-concept that demonstrates the advantages of utilizing enhancements. 

**Abstract (ZH)**: 通常，可解释人工智能（XAI）的研究集中在已知特定领域的一般策略下的黑盒模型。本文倡导在可解释搜索子领域中应用知识无关的可解释性，该子领域关注解释智能搜索技术所做的选择。本文提议使用蒙特卡洛树搜索（MCTS）增强技术作为获得额外数据并提供更高质量解释的解决方案，同时保持知识无关性，并分析最受欢迎的增强技术在引入特定类型可解释性方面的差异。迄今为止，尚未有其他研究考虑MCTS增强技术的可解释性。我们提出了一种概念验证方法，以展示利用增强技术的优势。 

---
# NeuroPhysNet: A FitzHugh-Nagumo-Based Physics-Informed Neural Network Framework for Electroencephalograph (EEG) Analysis and Motor Imagery Classification 

**Title (ZH)**: 基于FitzHugh-Nagumo模型的物理信息神经网络框架：EEG分析与 Motor Imagery分类 

**Authors**: Zhenyu Xia, Xinlei Huang, Suvash C. Saha  

**Link**: [PDF](https://arxiv.org/pdf/2506.13222)  

**Abstract**: Electroencephalography (EEG) is extensively employed in medical diagnostics and brain-computer interface (BCI) applications due to its non-invasive nature and high temporal resolution. However, EEG analysis faces significant challenges, including noise, nonstationarity, and inter-subject variability, which hinder its clinical utility. Traditional neural networks often lack integration with biophysical knowledge, limiting their interpretability, robustness, and potential for medical translation. To address these limitations, this study introduces NeuroPhysNet, a novel Physics-Informed Neural Network (PINN) framework tailored for EEG signal analysis and motor imagery classification in medical contexts. NeuroPhysNet incorporates the FitzHugh-Nagumo model, embedding neurodynamical principles to constrain predictions and enhance model robustness. Evaluated on the BCIC-IV-2a dataset, the framework achieved superior accuracy and generalization compared to conventional methods, especially in data-limited and cross-subject scenarios, which are common in clinical settings. By effectively integrating biophysical insights with data-driven techniques, NeuroPhysNet not only advances BCI applications but also holds significant promise for enhancing the precision and reliability of clinical diagnostics, such as motor disorder assessments and neurorehabilitation planning. 

**Abstract (ZH)**: 基于生理约束的神经网络（NeuroPhysNet）：用于医学情境下的脑电图信号分析与运动想象分类 

---
# Machine Learning as Iterated Belief Change a la Darwiche and Pearl 

**Title (ZH)**: 机器学习作为拉里奇和皮尔莱格风格的迭代信念变化 

**Authors**: Theofanis Aravanis  

**Link**: [PDF](https://arxiv.org/pdf/2506.13157)  

**Abstract**: Artificial Neural Networks (ANNs) are powerful machine-learning models capable of capturing intricate non-linear relationships. They are widely used nowadays across numerous scientific and engineering domains, driving advancements in both research and real-world applications. In our recent work, we focused on the statics and dynamics of a particular subclass of ANNs, which we refer to as binary ANNs. A binary ANN is a feed-forward network in which both inputs and outputs are restricted to binary values, making it particularly suitable for a variety of practical use cases. Our previous study approached binary ANNs through the lens of belief-change theory, specifically the Alchourron, Gardenfors and Makinson (AGM) framework, yielding several key insights. Most notably, we demonstrated that the knowledge embodied in a binary ANN (expressed through its input-output behaviour) can be symbolically represented using a propositional logic language. Moreover, the process of modifying a belief set (through revision or contraction) was mapped onto a gradual transition through a series of intermediate belief sets. Analogously, the training of binary ANNs was conceptualized as a sequence of such belief-set transitions, which we showed can be formalized using full-meet AGM-style belief change. In the present article, we extend this line of investigation by addressing some critical limitations of our previous study. Specifically, we show that Dalal's method for belief change naturally induces a structured, gradual evolution of states of belief. More importantly, given the known shortcomings of full-meet belief change, we demonstrate that the training dynamics of binary ANNs can be more effectively modelled using robust AGM-style change operations -- namely, lexicographic revision and moderate contraction -- that align with the Darwiche-Pearl framework for iterated belief change. 

**Abstract (ZH)**: 人工神经网络（ANNs）是强大的机器学习模型，能够捕获复杂的非线性关系。它们在众多科学和工程领域中广泛使用，推动了研究和实际应用的进步。在我们的近期工作中，我们专注于一类特定子类的ANNs，称之为二元ANNs。二元ANNs是输入和输出都限制为二元值的前馈网络，特别适合于多种实际应用场景。我们之前的研究通过信念变化理论的视角，特别是Alchourron、Gardenfors和Makinson（AGM）框架，获得了若干重要见解。我们表明，二元ANNs所包含的知识（通过输入输出行为表达）可以使用命题逻辑语言进行符号表示。同时，信念集的修改过程（通过修订或收缩）被映射为一系列中间信念集的渐进过渡。类似地，二元ANNs的训练被概念化为这种信念集过渡的序列，我们证明了这可以使用全交集AGM样式信念变化的形式化方法进行表述。在本文中，我们通过解决之前研究的一些关键限制，进一步扩展了这一研究方向。具体来说，我们展示了Dalal的信念变化方法自然地诱导了一种结构化、渐进的信念状态演变。更重要的是，鉴于全交集信念变化已知的不足，我们证明了二元ANNs的训练动力学可以更有效地通过鲁棒的AGM样式变化操作进行建模，即词汇修订和适度收缩，这些操作与Darwiche-Pearl框架下的迭代信念变化框架一致。 

---
# Dynamic Reinsurance Treaty Bidding via Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于多代理强化学习的动态再保险合约招标 

**Authors**: Stella C. Dong, James R. Finlay  

**Link**: [PDF](https://arxiv.org/pdf/2506.13113)  

**Abstract**: This paper develops a novel multi-agent reinforcement learning (MARL) framework for reinsurance treaty bidding, addressing long-standing inefficiencies in traditional broker-mediated placement processes. We pose the core research question: Can autonomous, learning-based bidding systems improve risk transfer efficiency and outperform conventional pricing approaches in reinsurance markets?
In our model, each reinsurer is represented by an adaptive agent that iteratively refines its bidding strategy within a competitive, partially observable environment. The simulation explicitly incorporates institutional frictions including broker intermediation, incumbent advantages, last-look privileges, and asymmetric access to underwriting information.
Empirical analysis demonstrates that MARL agents achieve up to 15% higher underwriting profit, 20% lower tail risk (CVaR), and over 25% improvement in Sharpe ratios relative to actuarial and heuristic baselines. Sensitivity tests confirm robustness across hyperparameter settings, and stress testing reveals strong resilience under simulated catastrophe shocks and capital constraints.
These findings suggest that MARL offers a viable path toward more transparent, adaptive, and risk-sensitive reinsurance markets. The proposed framework contributes to emerging literature at the intersection of algorithmic market design, strategic bidding, and AI-enabled financial decision-making. 

**Abstract (ZH)**: 一种基于多Agent强化学习的再保险条约招标新框架：自主学习招标系统能否改善风险转移效率并超越传统定价方法？ 

---
# A Memetic Walrus Algorithm with Expert-guided Strategy for Adaptive Curriculum Sequencing 

**Title (ZH)**: 基于专家引导策略的适应性课程序列化迷因海象算法 

**Authors**: Qionghao Huang, Lingnuo Lu, Xuemei Wu, Fan Jiang, Xizhe Wang, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13092)  

**Abstract**: Adaptive Curriculum Sequencing (ACS) is essential for personalized online learning, yet current approaches struggle to balance complex educational constraints and maintain optimization stability. This paper proposes a Memetic Walrus Optimizer (MWO) that enhances optimization performance through three key innovations: (1) an expert-guided strategy with aging mechanism that improves escape from local optima; (2) an adaptive control signal framework that dynamically balances exploration and exploitation; and (3) a three-tier priority mechanism for generating educationally meaningful sequences. We formulate ACS as a multi-objective optimization problem considering concept coverage, time constraints, and learning style compatibility. Experiments on the OULAD dataset demonstrate MWO's superior performance, achieving 95.3% difficulty progression rate (compared to 87.2% in baseline methods) and significantly better convergence stability (standard deviation of 18.02 versus 28.29-696.97 in competing algorithms). Additional validation on benchmark functions confirms MWO's robust optimization capability across diverse scenarios. The results demonstrate MWO's effectiveness in generating personalized learning sequences while maintaining computational efficiency and solution quality. 

**Abstract (ZH)**: 自适应课程序列化（ACS）是个性化在线学习的关键，但当前方法在平衡复杂教育约束和保持优化稳定性方面存在困难。本文提出了一种遗传 walrus 优化器（MWO），通过三种关键创新增强优化性能：（1）具有老化机制的专家引导策略，提高从局部最优解中跳出的能力；（2）自适应控制信号框架，动态平衡探索与利用；（3）生成教育意义序列的三级优先级机制。我们将 ACS 形式化为一个多目标优化问题，考虑概念覆盖、时间限制和学习风格兼容性。在 OULAD 数据集上的实验表明，MWO 的性能优越，实现 95.3% 的难度进展率（基线方法为 87.2%），并且收敛稳定性显著更好（标准差为 18.02，而竞争对手算法为 28.29-696.97）。基准函数上的额外验证证实了 MWO 在各种场景下稳健的优化能力。结果表明，MWO 在保持计算效率和解质量的同时，有效地生成了个性化学习序列。 

---
# Sectoral Coupling in Linguistic State Space 

**Title (ZH)**: 语言状态空间中的部门耦合 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2506.12927)  

**Abstract**: This work presents a formal framework for quantifying the internal dependencies between functional subsystems within artificial agents whose belief states are composed of structured linguistic fragments. Building on the Semantic Manifold framework, which organizes belief content into functional sectors and stratifies them across hierarchical levels of abstraction, we introduce a system of sectoral coupling constants that characterize how one cognitive sector influences another within a fixed level of abstraction. The complete set of these constants forms an agent-specific coupling profile that governs internal information flow, shaping the agent's overall processing tendencies and cognitive style. We provide a detailed taxonomy of these intra-level coupling roles, covering domains such as perceptual integration, memory access and formation, planning, meta-cognition, execution control, and affective modulation. We also explore how these coupling profiles generate feedback loops, systemic dynamics, and emergent signatures of cognitive behavior. Methodologies for inferring these profiles from behavioral or internal agent data are outlined, along with a discussion of how these couplings evolve across abstraction levels. This framework contributes a mechanistic and interpretable approach to modeling complex cognition, with applications in AI system design, alignment diagnostics, and the analysis of emergent agent behavior. 

**Abstract (ZH)**: 本研究提出了一种形式化框架，用于量化人工代理内部由结构化语言片段组成信念状态的功能子系统之间的内部依赖性。该框架基于语义流形框架，后者将信念内容组织成功能部门，并按抽象层次分层。在此基础上，我们引入了一种部门耦合常数系统，以表征固定抽象层次内一个认知部门如何影响另一个部门。这一整套耦合常数构成了代理特有的耦合特征，决定了内部信息流，塑造了代理的整体处理倾向和认知风格。我们详细探讨了这些同级耦合角色的分类，涵盖感知整合、记忆获取和形成、计划、元认知、执行控制和情感调节等领域。我们还探讨了这些耦合特征如何产生反馈回路、系统动态和认知行为的涌现特征。文中概述了从行为或内部代理数据推断这些特征的方法，并讨论了这些耦合如何随着抽象层次的变化而演变。该框架提供了一种机械的、可解释的方法来建模复杂认知，适用于AI系统设计、对齐诊断以及对涌现代理行为的分析。 

---
# Constraint-Guided Prediction Refinement via Deterministic Diffusion Trajectories 

**Title (ZH)**: 基于确定性扩散轨迹的约束引导预测细化 

**Authors**: Pantelis Dogoulis, Fabien Bernier, Félix Fourreau, Karim Tit, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2506.12911)  

**Abstract**: Many real-world machine learning tasks require outputs that satisfy hard constraints, such as physical conservation laws, structured dependencies in graphs, or column-level relationships in tabular data. Existing approaches rely either on domain-specific architectures and losses or on strong assumptions on the constraint space, restricting their applicability to linear or convex constraints. We propose a general-purpose framework for constraint-aware refinement that leverages denoising diffusion implicit models (DDIMs). Starting from a coarse prediction, our method iteratively refines it through a deterministic diffusion trajectory guided by a learned prior and augmented by constraint gradient corrections. The approach accommodates a wide class of non-convex and nonlinear equality constraints and can be applied post hoc to any base model. We demonstrate the method in two representative domains: constrained adversarial attack generation on tabular data with column-level dependencies and in AC power flow prediction under Kirchhoff's laws. Across both settings, our diffusion-guided refinement improves both constraint satisfaction and performance while remaining lightweight and model-agnostic. 

**Abstract (ZH)**: 约束aware细化的一般框架：基于去噪扩散隐模型的方法 

---
# KCLNet: Physics-Informed Power Flow Prediction via Constraints Projections 

**Title (ZH)**: KCLNet: 基于约束投影的物理guided功率流预测 

**Authors**: Pantelis Dogoulis, Karim Tit, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2506.12902)  

**Abstract**: In the modern context of power systems, rapid, scalable, and physically plausible power flow predictions are essential for ensuring the grid's safe and efficient operation. While traditional numerical methods have proven robust, they require extensive computation to maintain physical fidelity under dynamic or contingency conditions. In contrast, recent advancements in artificial intelligence (AI) have significantly improved computational speed; however, they often fail to enforce fundamental physical laws during real-world contingencies, resulting in physically implausible predictions. In this work, we introduce KCLNet, a physics-informed graph neural network that incorporates Kirchhoff's Current Law as a hard constraint via hyperplane projections. KCLNet attains competitive prediction accuracy while ensuring zero KCL violations, thereby delivering reliable and physically consistent power flow predictions critical to secure the operation of modern smart grids. 

**Abstract (ZH)**: 在现代电力系统的背景下，快速、可扩展且物理上合理的潮流预测对于确保电网的安全和高效运行至关重要。虽然传统的数值方法在保持物理一致性方面表现出 robust 性，但在动态或故障条件下需要大量的计算。相比之下，最近在人工智能（AI）方面的进展显著提高了计算速度，但在实际故障条件下经常无法强制执行基本的物理定律，从而导致物理上不合理的结果。在本工作中，我们引入了 KCLNet，这是一种物理信息图神经网络，通过超平面投影将基尔霍夫电流定律作为硬约束纳入其中。KCLNet 在保持零基尔霍夫电流定律违反的情况下达到竞争力的预测精度，从而实现对现代智能电网安全运行至关重要的可靠且物理上一致的潮流预测。 

---
# Homeostatic Coupling for Prosocial Behavior 

**Title (ZH)**: 恒定耦合促进利他行为 

**Authors**: Naoto Yoshida, Kingson Man  

**Link**: [PDF](https://arxiv.org/pdf/2506.12894)  

**Abstract**: When regarding the suffering of others, we often experience personal distress and feel compelled to help\footnote{Preprint. Under review.}. Inspired by living systems, we investigate the emergence of prosocial behavior among autonomous agents that are motivated by homeostatic self-regulation. We perform multi-agent reinforcement learning, treating each agent as a vulnerable homeostat charged with maintaining its own well-being. We introduce an empathy-like mechanism to share homeostatic states between agents: an agent can either \emph{observe} their partner's internal state ({\bf cognitive empathy}) or the agent's internal state can be \emph{directly coupled} to that of their partner ({\bf affective empathy}). In three simple multi-agent environments, we show that prosocial behavior arises only under homeostatic coupling - when the distress of a partner can affect one's own well-being. Additionally, we show that empathy can be learned: agents can ``decode" their partner's external emotive states to infer the partner's internal homeostatic states. Assuming some level of physiological similarity, agents reference their own emotion-generation functions to invert the mapping from outward display to internal state. Overall, we demonstrate the emergence of prosocial behavior when homeostatic agents learn to ``read" the emotions of others and then to empathize, or feel as they feel. 

**Abstract (ZH)**: 当面对他人的苦难时，我们往往会经历个人的痛苦并感到有必要提供帮助。受生物系统启发，我们研究自主代理在基于稳态自我调节动机的情况下，亲社会行为的涌现。我们进行了多代理强化学习，将每个代理视为需要维护自身福祉的脆弱稳态系统。我们引入了一种类似共情的机制来在代理之间共享稳态状态：代理可以“观察”其伙伴的内部状态（认知共情），或者其内部状态可以“直接耦合”到其伙伴的内部状态（情感共情）。在三个简单的多代理环境中，我们证明了仅在稳态耦合下才会出现亲社会行为——当伙伴的痛苦会影响自身的福祉时。此外，我们还展示了共情可以通过学习获得：代理可以“解码”其伙伴的外部情绪状态以推断其伙伴的内部稳态状态。假设一定程度的生理相似性，代理会参照自身的 emotion 生成函数来反转从外部表现到内部状态的映射。总体而言，我们展示了当稳态代理学会“读取”他人的感情并产生共情时，亲社会行为的涌现。 

---
# Evolutionary Developmental Biology Can Serve as the Conceptual Foundation for a New Design Paradigm in Artificial Intelligence 

**Title (ZH)**: 演化发育生物学可以作为人工智新建模范式的概念基础 

**Authors**: Zeki Doruk Erden, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2506.12891)  

**Abstract**: Artificial intelligence (AI), propelled by advancements in machine learning, has made significant strides in solving complex tasks. However, the current neural network-based paradigm, while effective, is heavily constrained by inherent limitations, primarily a lack of structural organization and a progression of learning that displays undesirable properties. As AI research progresses without a unifying framework, it either tries to patch weaknesses heuristically or draws loosely from biological mechanisms without strong theoretical foundations. Meanwhile, the recent paradigm shift in evolutionary understanding -- driven primarily by evolutionary developmental biology (EDB) -- has been largely overlooked in AI literature, despite a striking analogy between the Modern Synthesis and contemporary machine learning, evident in their shared assumptions, approaches, and limitations upon careful analysis. Consequently, the principles of adaptation from EDB that reshaped our understanding of the evolutionary process can also form the foundation of a unifying conceptual framework for the next design philosophy in AI, going beyond mere inspiration and grounded firmly in biology's first principles. This article provides a detailed overview of the analogy between the Modern Synthesis and modern machine learning, and outlines the core principles of a new AI design paradigm based on insights from EDB. To exemplify our analysis, we also present two learning system designs grounded in specific developmental principles -- regulatory connections, somatic variation and selection, and weak linkage -- that resolve multiple major limitations of contemporary machine learning in an organic manner, while also providing deeper insights into the role of these mechanisms in biological evolution. 

**Abstract (ZH)**: 基于进化发育生物学的类人智能设计新范式 

---
# Rethinking Optimization: A Systems-Based Approach to Social Externalities 

**Title (ZH)**: 重新思考优化：基于系统的方法研究社会外部性 

**Authors**: Pegah Nokhiz, Aravinda Kanchana Ruwanpathirana, Helen Nissenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2506.12825)  

**Abstract**: Optimization is widely used for decision making across various domains, valued for its ability to improve efficiency. However, poor implementation practices can lead to unintended consequences, particularly in socioeconomic contexts where externalities (costs or benefits to third parties outside the optimization process) are significant. To propose solutions, it is crucial to first characterize involved stakeholders, their goals, and the types of subpar practices causing unforeseen outcomes. This task is complex because affected stakeholders often fall outside the direct focus of optimization processes. Also, incorporating these externalities into optimization requires going beyond traditional economic frameworks, which often focus on describing externalities but fail to address their normative implications or interconnected nature, and feedback loops. This paper suggests a framework that combines systems thinking with the economic concept of externalities to tackle these challenges. This approach aims to characterize what went wrong, who was affected, and how (or where) to include them in the optimization process. Economic externalities, along with their established quantification methods, assist in identifying "who was affected and how" through stakeholder characterization. Meanwhile, systems thinking (an analytical approach to comprehending relationships in complex systems) provides a holistic, normative perspective. Systems thinking contributes to an understanding of interconnections among externalities, feedback loops, and determining "when" to incorporate them in the optimization. Together, these approaches create a comprehensive framework for addressing optimization's unintended consequences, balancing descriptive accuracy with normative objectives. Using this, we examine three common types of subpar practices: ignorance, error, and prioritization of short-term goals. 

**Abstract (ZH)**: 优化在各个领域广泛用于决策制定，因其能够提高效率而受到重视。然而，不良实施实践可能导致意外后果，尤其是在外部性（对优化过程之外的第三方的成本或收益）显著的社会经济背景下。为了提出解决方案，首先要明确相关利益相关者、他们的目标以及导致意外结果的不良实践类型。这一任务之所以复杂，是因为受影响的利益相关者往往不在优化过程的核心关注范围内。此外，将这些外部性纳入优化还需要超越传统的经济框架，这些框架通常专注于描述外部性，但未能解决它们的规范含义或相互关系，以及反馈循环问题。本文提出了一种结合系统思维与经济外部性概念的框架来应对这些挑战。该方法旨在界定“问题出在哪里”，“谁受到了影响”，以及如何（或在哪里）将这些因素纳入优化过程。经济外部性及其已建立的量化方法帮助通过利益相关者分析来确定“谁受到了影响以及程度如何”。同时，系统思维（一种理解和分析复杂系统中关系的分析方法）提供了全面且规范的视角。系统思维有助于理解外部性之间的相互关系、反馈循环及其“何时”纳入优化的重要性。结合这些方法，可以建立一个全面的框架，以解决优化的意外后果，兼顾描述准确性和规范目标。我们使用这种方法分析了三种常见的不良实践类型：无知、错误和短期目标优先。 

---
# Fuzzy Propositional Formulas under the Stable Model Semantics 

**Title (ZH)**: 稳定模型语义下的模糊命题公式 

**Authors**: Joohyung Lee, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12804)  

**Abstract**: We define a stable model semantics for fuzzy propositional formulas, which generalizes both fuzzy propositional logic and the stable model semantics of classical propositional formulas. The syntax of the language is the same as the syntax of fuzzy propositional logic, but its semantics distinguishes stable models from non-stable models. The generality of the language allows for highly configurable nonmonotonic reasoning for dynamic domains involving graded truth degrees. We show that several properties of Boolean stable models are naturally extended to this many-valued setting, and discuss how it is related to other approaches to combining fuzzy logic and the stable model semantics. 

**Abstract (ZH)**: 我们定义了一种模糊命题公式的第一稳定模型语义，该语义既泛化了模糊命题逻辑，也泛化了经典命题公式稳定模型语义。语言的语法与模糊命题逻辑相同，但其语义将稳定模型与非稳定模型区分开来。该语言的普适性使得在涉及等级真度的动态领域中能够进行高度配置的非单调推理。我们展示了布尔稳定模型的若干性质可以自然地扩展到这个多值设置，并讨论了它与其他结合模糊逻辑和稳定模型语义的方法的关系。 

---
# LPMLN, Weak Constraints, and P-log 

**Title (ZH)**: LPMLN, 软约束和P-log 

**Authors**: Joohyung Lee, Zhun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12784)  

**Abstract**: LPMLN is a recently introduced formalism that extends answer set programs by adopting the log-linear weight scheme of Markov Logic. This paper investigates the relationships between LPMLN and two other extensions of answer set programs: weak constraints to express a quantitative preference among answer sets, and P-log to incorporate probabilistic uncertainty. We present a translation of LPMLN into programs with weak constraints and a translation of P-log into LPMLN, which complement the existing translations in the opposite directions. The first translation allows us to compute the most probable stable models (i.e., MAP estimates) of LPMLN programs using standard ASP solvers. This result can be extended to other formalisms, such as Markov Logic, ProbLog, and Pearl's Causal Models, that are shown to be translatable into LPMLN. The second translation tells us how probabilistic nonmonotonicity (the ability of the reasoner to change his probabilistic model as a result of new information) of P-log can be represented in LPMLN, which yields a way to compute P-log using standard ASP solvers and MLN solvers. 

**Abstract (ZH)**: LPMLN是最近提出的一种形式化方法，通过采用Markov Logic的对数线性权重方案扩展了回答集程序。本文探讨了LPMLN与其他两种回答集程序扩展之间的关系：弱约束用于表达回答集之间的定量偏好，以及P-log用于整合概率不确定性。我们提出了从LPMLN到弱约束程序的翻译，以及从P-log到LPMLN的翻译，这些翻译补充了现有从相反方向的翻译。第一个翻译使我们能够使用标准ASP求解器计算LPMLN程序的最可能稳定模型（即，MAP估计）。这一结果可以扩展到其他形式化方法，如可以转换为LPMLN的Markov Logic、ProbLog和Pearl因果模型。第二个翻译说明了如何在LPMLN中表示P-log的概率非单调性（推理器根据新信息改变其概率模型的能力），从而提供了一种使用标准ASP求解器和MLN求解器计算P-log的方法。 

---
# Decentralized Decision Making in Two Sided Manufacturing-as-a-Service Marketplaces 

**Title (ZH)**: 两侧制造即服务市场中的去中心化决策Making 

**Authors**: Deepak Pahwa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12730)  

**Abstract**: Advancements in digitization have enabled two sided manufacturing-as-a-service (MaaS) marketplaces which has significantly reduced product development time for designers. These platforms provide designers with access to manufacturing resources through a network of suppliers and have instant order placement capabilities. Two key decision making levers are typically used to optimize the operations of these marketplaces: pricing and matching. The existing marketplaces operate in a centralized structure where they have complete control over decision making. However, a decentralized organization of the platform enables transparency of information across clients and suppliers. This dissertation focuses on developing tools for decision making enabling decentralization in MaaS marketplaces. In pricing mechanisms, a data driven method is introduced which enables small service providers to price services based on specific attributes of the services offered. A data mining method recommends a network based price to a supplier based on its attributes and the attributes of other suppliers on the platform. Three different approaches are considered for matching mechanisms. First, a reverse auction mechanism is introduced where designers bid for manufacturing services and the mechanism chooses a supplier which can match the bid requirements and stated price. The second approach uses mechanism design and mathematical programming to develop a stable matching mechanism for matching orders to suppliers based on their preferences. Empirical simulations are used to test the mechanisms in a simulated 3D printing marketplace and to evaluate the impact of stability on its performance. The third approach considers the matching problem in a dynamic and stochastic environment where demand (orders) and supply (supplier capacities) arrive over time and matching is performed online. 

**Abstract (ZH)**: 数字化进展使得两面市场制造即服务（MaaS）平台得以发展，显著缩短了设计师的产品开发时间。这些平台通过供应商网络为设计师提供制造资源，并具备即时下单能力。通常使用两类决策杠杆来优化这些市场的运营：定价和匹配。现有的市场平台采用集中式架构，对决策具有完全控制。然而，平台的分散组织能促进客户和供应商之间的信息透明。本论文旨在开发支持MaaS平台分散决策的工具。在定价机制中，提出了一种基于数据的方法，使小型服务提供商能够根据所提供的服务特性定价。数据挖掘方法基于供应商及其平台上其他供应商的特性，推荐网络定价。匹配机制分为三种不同方法：首先，引入逆向拍卖机制，设计师为制造服务出价，机制选择能满足出价要求并报价的供应商。其次，采用机制设计和数学规划开发基于供方偏好的稳定匹配机制。使用实证模拟在模拟的3D打印市场中测试这些机制，并评估稳定性的性能影响。最后，第三种方法考虑在动态和随机环境中进行匹配问题，即需求（订单）和供应（供应商能力）随时间到达，并在线进行匹配。 

---
# Rethinking DPO: The Role of Rejected Responses in Preference Misalignment 

**Title (ZH)**: 重新思考DPO：被拒绝响应在偏好不对齐中的作用 

**Authors**: Jay Hyeon Cho, JunHyeok Oh, Myunsoo Kim, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.12725)  

**Abstract**: Direct Preference Optimization (DPO) is a simple and efficient framework that has attracted substantial attention. However, it often struggles to meet its primary objectives -- increasing the generation probability of chosen responses while reducing that of rejected responses -- due to the dominant influence of rejected responses on the loss function. This imbalance leads to suboptimal performance in promoting preferred responses. In this work, we systematically analyze the limitations of DPO and existing algorithms designed to achieve the objectives stated above. To address these limitations, we propose Bounded-DPO (BDPO), a novel method that bounds the influence of rejected responses while maintaining the original optimization structure of DPO. Through theoretical analysis and empirical evaluations, we demonstrate that BDPO achieves a balanced optimization of the chosen and rejected responses, outperforming existing algorithms. 

**Abstract (ZH)**: 直接偏好优化（DPO）是一种简单且高效的框架，引起了广泛关注。然而，它经常难以实现其主要目标——增加选定响应的生成概率同时减少被拒绝响应的概率——这主要是由于被拒绝响应在损失函数中占主导地位的影响。这种不平衡导致在促进偏好响应方面表现不佳。在本文中，我们系统分析了DPO及其现有算法的局限性，旨在实现上述目标。为了解决这些局限性，我们提出了一种新型的方法——有界直接偏好优化（BDPO），该方法限制了被拒绝响应的影响，同时保持了DPO的原始优化结构。通过理论分析和实证评估，我们证明了BDPO实现了选定和被拒绝响应之间的平衡优化，优于现有算法。 

---
# SciSage: A Multi-Agent Framework for High-Quality Scientific Survey Generation 

**Title (ZH)**: SciSage: 一种高质量科学综述生成的多智能体框架 

**Authors**: Xiaofeng Shi, Qian Kou, Yuduo Li, Ning Tang, Jinxin Xie, Longbin Yu, Songjing Wang, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12689)  

**Abstract**: The rapid growth of scientific literature demands robust tools for automated survey-generation. However, current large language model (LLM)-based methods often lack in-depth analysis, structural coherence, and reliable citations. To address these limitations, we introduce SciSage, a multi-agent framework employing a reflect-when-you-write paradigm. SciSage features a hierarchical Reflector agent that critically evaluates drafts at outline, section, and document levels, collaborating with specialized agents for query interpretation, content retrieval, and refinement. We also release SurveyScope, a rigorously curated benchmark of 46 high-impact papers (2020-2025) across 11 computer science domains, with strict recency and citation-based quality controls. Evaluations demonstrate that SciSage outperforms state-of-the-art baselines (LLM x MapReduce-V2, AutoSurvey), achieving +1.73 points in document coherence and +32% in citation F1 scores. Human evaluations reveal mixed outcomes (3 wins vs. 7 losses against human-written surveys), but highlight SciSage's strengths in topical breadth and retrieval efficiency. Overall, SciSage offers a promising foundation for research-assistive writing tools. 

**Abstract (ZH)**: 科学文献的快速增长需要 robust 的自动调查生成工具。然而，当前基于大规模语言模型（LLM）的方法往往缺乏深入分析、结构性连贯性和可靠的引用。为了解决这些限制，我们引入了 SciSage，这是一种采用写时反思的多代理框架。SciSage 特设了一个分层的反思代理，该代理在大纲、段落和文档层面批判性地评估草稿，并与专门的代理合作，用于查询解释、内容检索和改进。我们还发布了 SurveyScope，这是一个严格编目的基准，包含从 2020-2025 年 11 个计算机科学领域的 46 篇高影响力论文，严格的质量控制基于时效性和引用。评估结果显示，SciSage 在文档连贯性上优于最先进的基线（LLM x MapReduce-V2、AutoSurvey），分别提高了 1.73 分和 32% 的引文 F1 分数。人类评估显示结果参差不齐（3 胜 7 负，比人类撰写的调查报告），但强调了 SciSage 在主题广度和检索效率方面的优势。总的来说，SciSage 为研究辅助写作工具提供了有前景的基础。 

---
# Optimizing Blood Transfusions and Predicting Shortages in Resource-Constrained Areas 

**Title (ZH)**: 优化血液输注并在资源受限区域预测短缺 

**Authors**: El Arbi Belfarsi, Sophie Brubaker, Maria Valero  

**Link**: [PDF](https://arxiv.org/pdf/2506.12647)  

**Abstract**: Our research addresses the critical challenge of managing blood transfusions and optimizing allocation in resource-constrained regions. We present heuristic matching algorithms for donor-patient and blood bank selection, alongside machine learning methods to analyze blood transfusion acceptance data and predict potential shortages. We developed simulations to optimize blood bank operations, progressing from random allocation to a system incorporating proximity-based selection, blood type compatibility, expiration prioritization, and rarity scores. Moving from blind matching to a heuristic-based approach yielded a 28.6% marginal improvement in blood request acceptance, while a multi-level heuristic matching resulted in a 47.6% improvement. For shortage prediction, we compared Long Short-Term Memory (LSTM) networks, Linear Regression, and AutoRegressive Integrated Moving Average (ARIMA) models, trained on 170 days of historical data. Linear Regression slightly outperformed others with a 1.40% average absolute percentage difference in predictions. Our solution leverages a Cassandra NoSQL database, integrating heuristic optimization and shortage prediction to proactively manage blood resources. This scalable approach, designed for resource-constrained environments, considers factors such as proximity, blood type compatibility, inventory expiration, and rarity. Future developments will incorporate real-world data and additional variables to improve prediction accuracy and optimization performance. 

**Abstract (ZH)**: 我们的研究解决了资源受限地区血液输注管理和优化分配的关键挑战。我们提出了供者-患者的启发式匹配算法以及血液银行选择方法，并结合机器学习方法分析血液输注接受数据，预测潜在短缺。我们开发了模拟优化血液银行运营，从随机分配发展到基于proximity、血型兼容性、到期优先级和稀有度评分的系统。从盲匹配到启发式方法匹配，血液请求接受率提高了28.6%，而多层次启发式匹配则提高了47.6%。在短缺预测方面，我们比较了Long Short-Term Memory（LSTM）网络、线性回归和自回归整合移动平均（ARIMA）模型，使用170天的历史数据训练模型。线性回归在预测中表现略微优于其他方法，平均绝对百分比误差为1.40%。我们的解决方案利用Cassandra NoSQL数据库，结合启发式优化和短缺预测，以主动管理血液资源。这种可扩展的方法旨在资源受限环境中，考虑了距离、血型兼容性、库存到期和稀有性等因素。未来的研究将整合现实世界数据和额外变量，以提高预测准确性和优化性能。 

---
# AI Flow: Perspectives, Scenarios, and Approaches 

**Title (ZH)**: AI_flow: 视角、场景与方法 

**Authors**: Hongjun An, Sida Huang, Siqi Huang, Ruanjun Li, Yuanzhi Liang, Jiawei Shao, Zihan Wang, Cheng Yuan, Chi Zhang, Hongyuan Zhang, Wenhao Zhuang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12479)  

**Abstract**: Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems. 

**Abstract (ZH)**: 由克劳德·香农的基础信息理论和艾伦·图灵的机器智能 visionary 框架引领，信息和通信技术（IT/CT）的趋同进化创造了不间断的连接和计算波浪。这种协同作用引发了一场技术革命，现在正处于由大型人工智能（AI）模型重塑行业并重新定义人机协作的巅峰期。然而，由于大型模型的资源消耗巨大和高通信带宽需求，实现无所不在的智能面临着重大挑战。为了应对这些挑战，AI Flow作为一种多学科框架被引入，融合了最新的IT和CT进步，并特别强调以下三个关键点。首先，设备-边缘-云框架作为基础，将终端设备、边缘服务器和云集群集成起来，以优化低延迟模型推断的可扩展性和效率。其次，我们引入了家族模型的概念，即一系列具有对齐隐藏特性的不同大小模型，这使得有效的协作和适应不同资源约束和动态场景变得灵活。第三，基于连接和交互的智能涌现是AI Flow的一个新型范式。通过利用通信网络增强连接性，异构节点间的AI模型协作实现了一种超越单个模型能力的涌现智能。AI Flow的创新提供了增强智能、及时响应和无所不在的AI服务访问，为人工智能技术与通信系统的更紧密融合铺平了道路。 

---
# Topology-Assisted Spatio-Temporal Pattern Disentangling for Scalable MARL in Large-scale Autonomous Traffic Control 

**Title (ZH)**: 拓扑辅助时空模式解耦for大规模自主交通控制中的可扩展多智能体 reinforcement 学习 

**Authors**: Rongpeng Li, Jianhang Zhu, Jiahao Huang, Zhifeng Zhao, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12453)  

**Abstract**: Intelligent Transportation Systems (ITSs) have emerged as a promising solution towards ameliorating urban traffic congestion, with Traffic Signal Control (TSC) identified as a critical component. Although Multi-Agent Reinforcement Learning (MARL) algorithms have shown potential in optimizing TSC through real-time decision-making, their scalability and effectiveness often suffer from large-scale and complex environments. Typically, these limitations primarily stem from a fundamental mismatch between the exponential growth of the state space driven by the environmental heterogeneities and the limited modeling capacity of current solutions. To address these issues, this paper introduces a novel MARL framework that integrates Dynamic Graph Neural Networks (DGNNs) and Topological Data Analysis (TDA), aiming to enhance the expressiveness of environmental representations and improve agent coordination. Furthermore, inspired by the Mixture of Experts (MoE) architecture in Large Language Models (LLMs), a topology-assisted spatial pattern disentangling (TSD)-enhanced MoE is proposed, which leverages topological signatures to decouple graph features for specialized processing, thus improving the model's ability to characterize dynamic and heterogeneous local observations. The TSD module is also integrated into the policy and value networks of the Multi-agent Proximal Policy Optimization (MAPPO) algorithm, further improving decision-making efficiency and robustness. Extensive experiments conducted on real-world traffic scenarios, together with comprehensive theoretical analysis, validate the superior performance of the proposed framework, highlighting the model's scalability and effectiveness in addressing the complexities of large-scale TSC tasks. 

**Abstract (ZH)**: 智能交通系统（ITSs）已成为缓解城市交通拥堵的有前途的解决方案，其中交通信号控制（TSC）被认定为关键组成部分。虽然多代理强化学习（MARL）算法在通过实时决策优化TSC方面展现出了潜力，但在大规模和复杂环境中，其可扩展性和有效性往往受到影响。这些问题主要源于环境异质性驱动的状态空间指数增长与当前解决方案有限的建模能力之间的根本性不匹配。为了解决这些问题，本文提出了一种新的MARL框架，该框架结合了动态图神经网络（DGNNs）和拓扑数据分析（TDA），旨在增强环境表示的表达能力并提高代理协调能力。此外，借鉴大型语言模型（LLMs）中的混合专家（MoE）架构，提出了基于拓扑辅助的空间模式解耦（TSD）增强的MoE，利用拓扑特征解耦图特征以进行专门处理，从而提高模型表征动态和异构局部观察的能力。TSD模块也被整合到多代理近端策略优化算法（MAPPO）的策略和价值网络中，进一步提高了决策效率和鲁棒性。在实际交通场景中进行的广泛实验以及全面的理论分析验证了所提出框架的优越性能，突显了该模型在大规模TSC任务复杂性方面的可扩展性和有效性。 

---
# Ghost Policies: A New Paradigm for Understanding and Learning from Failure in Deep Reinforcement Learning 

**Title (ZH)**: 鬼策略：深度强化学习中失败理解与学习的新范式 

**Authors**: Xabier Olaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12366)  

**Abstract**: Deep Reinforcement Learning (DRL) agents often exhibit intricate failure modes that are difficult to understand, debug, and learn from. This opacity hinders their reliable deployment in real-world applications. To address this critical gap, we introduce ``Ghost Policies,'' a concept materialized through Arvolution, a novel Augmented Reality (AR) framework. Arvolution renders an agent's historical failed policy trajectories as semi-transparent ``ghosts'' that coexist spatially and temporally with the active agent, enabling an intuitive visualization of policy divergence. Arvolution uniquely integrates: (1) AR visualization of ghost policies, (2) a behavioural taxonomy of DRL maladaptation, (3) a protocol for systematic human disruption to scientifically study failure, and (4) a dual-learning loop where both humans and agents learn from these visualized failures. We propose a paradigm shift, transforming DRL agent failures from opaque, costly errors into invaluable, actionable learning resources, laying the groundwork for a new research field: ``Failure Visualization Learning.'' 

**Abstract (ZH)**: 深层强化学习（DRL）智能体往往表现出复杂的失败模式，这些模式难以理解、调试和从中学习。这种透明度阻碍了它们在实际应用中的可靠部署。为填补这一关键空白，我们引入了“幽灵策略”这一概念，并通过一种名为Arvolution的新型增强现实（AR）框架予以实现。Arvolution将智能体的历史失败策略轨迹以半透明的“幽灵”形式渲染出来，使其在时空上与活跃的智能体共存，从而直观地展示策略发散。Arvolution独特地整合了以下四个方面：（1）AR中的幽灵策略可视化；（2）DRL适应不良的行为分类学；（3）系统的人类干预协议，用于科学研究中的失败；（4）双重学习循环，其中人类和智能体从这些可视化失败中学习。我们提出了一种范式转变，将DRL智能体的失败从不透明的高昂错误转变为宝贵且可操作的学习资源，为一个新的研究领域——“失败可视化学习”奠定了基础。 

---
# Efficient Network Automatic Relevance Determination 

**Title (ZH)**: 高效网络自动相关性确定 

**Authors**: Hongwei Zhang, Ziqi Ye, Xinyuan Wang, Xin Guo, Zenglin Xu, Yuan Cheng, Zixin Hu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12352)  

**Abstract**: We propose Network Automatic Relevance Determination (NARD), an extension of ARD for linearly probabilistic models, to simultaneously model sparse relationships between inputs $X \in \mathbb R^{d \times N}$ and outputs $Y \in \mathbb R^{m \times N}$, while capturing the correlation structure among the $Y$. NARD employs a matrix normal prior which contains a sparsity-inducing parameter to identify and discard irrelevant features, thereby promoting sparsity in the model. Algorithmically, it iteratively updates both the precision matrix and the relationship between $Y$ and the refined inputs. To mitigate the computational inefficiencies of the $\mathcal O(m^3 + d^3)$ cost per iteration, we introduce Sequential NARD, which evaluates features sequentially, and a Surrogate Function Method, leveraging an efficient approximation of the marginal likelihood and simplifying the calculation of determinant and inverse of an intermediate matrix. Combining the Sequential update with the Surrogate Function method further reduces computational costs. The computational complexity per iteration for these three methods is reduced to $\mathcal O(m^3+p^3)$, $\mathcal O(m^3 + d^2)$, $\mathcal O(m^3+p^2)$, respectively, where $p \ll d$ is the final number of features in the model. Our methods demonstrate significant improvements in computational efficiency with comparable performance on both synthetic and real-world datasets. 

**Abstract (ZH)**: 网络自动相关性确定（NARD）：一种同时建模输入和输出稀疏关系并捕获输出间相关结构的方法 

---
# Ontology Enabled Hybrid Modeling and Simulation 

**Title (ZH)**: 本体驱动的混合建模与仿真 

**Authors**: John Beverley, Andreas Tolk  

**Link**: [PDF](https://arxiv.org/pdf/2506.12290)  

**Abstract**: We explore the role of ontologies in enhancing hybrid modeling and simulation through improved semantic rigor, model reusability, and interoperability across systems, disciplines, and tools. By distinguishing between methodological and referential ontologies, we demonstrate how these complementary approaches address interoperability challenges along three axes: Human-Human, Human-Machine, and Machine-Machine. Techniques such as competency questions, ontology design patterns, and layered strategies are highlighted for promoting shared understanding and formal precision. Integrating ontologies with Semantic Web Technologies, we showcase their dual role as descriptive domain representations and prescriptive guides for simulation construction. Four application cases - sea-level rise analysis, Industry 4.0 modeling, artificial societies for policy support, and cyber threat evaluation - illustrate the practical benefits of ontology-driven hybrid simulation workflows. We conclude by discussing challenges and opportunities in ontology-based hybrid M&S, including tool integration, semantic alignment, and support for explainable AI. 

**Abstract (ZH)**: 我们探索本体在通过改进语义严谨性、模型重用性和跨系统、学科和工具的互操作性来增强混合建模与仿真的作用。通过区分方法论本体和参考本体，我们展示了这些互补方法如何在三条轴上解决互操作性挑战：人与人、人与机器以及机器与机器。我们强调了诸如能力问题、本体设计模式和分层策略等技术，以促进共享理解并提高形式精确度。我们将本体与语义网技术集成，展示了它们作为描述性领域表示和仿真实体制定的规范性指南的双重角色。四个应用案例——海平面上升分析、工业4.0建模、用于政策支持的人工社会以及网络威胁评估——阐述了本体驱动的混合仿真实践流程的实际益处。最后，我们讨论了基于本体的混合建模与仿真中的挑战与机遇，包括工具集成、语义对齐以及支持可解释AI的支持。 

---
# Deep Fictitious Play-Based Potential Differential Games for Learning Human-Like Interaction at Unsignalized Intersections 

**Title (ZH)**: 基于深层虚构博弈的潜在差分博弈方法学习无信号交叉口的人类交互行为 

**Authors**: Kehua Chen, Shucheng Zhang, Yinhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12283)  

**Abstract**: Modeling vehicle interactions at unsignalized intersections is a challenging task due to the complexity of the underlying game-theoretic processes. Although prior studies have attempted to capture interactive driving behaviors, most approaches relied solely on game-theoretic formulations and did not leverage naturalistic driving datasets. In this study, we learn human-like interactive driving policies at unsignalized intersections using Deep Fictitious Play. Specifically, we first model vehicle interactions as a Differential Game, which is then reformulated as a Potential Differential Game. The weights in the cost function are learned from the dataset and capture diverse driving styles. We also demonstrate that our framework provides a theoretical guarantee of convergence to a Nash equilibrium. To the best of our knowledge, this is the first study to train interactive driving policies using Deep Fictitious Play. We validate the effectiveness of our Deep Fictitious Play-Based Potential Differential Game (DFP-PDG) framework using the INTERACTION dataset. The results demonstrate that the proposed framework achieves satisfactory performance in learning human-like driving policies. The learned individual weights effectively capture variations in driver aggressiveness and preferences. Furthermore, the ablation study highlights the importance of each component within our model. 

**Abstract (ZH)**: 无信号交叉口车辆互动的建模是一项具有挑战性的任务，受到潜在博弈理论过程复杂性的制约。尽管先前的研究试图捕捉互动驾驶行为，但大多数方法仅依赖博弈 theoretic 表述，而未利用自然istic驾驶数据集。在本研究中，我们使用深度虚构博弈从无信号交叉口学习类人的互动驾驶策略。具体地，我们首先将车辆互动建模为微分博弈，然后将其重新表述为潜在微分博弈。成本函数中的权重是从数据集中学习得到的，能够捕捉多样化的驾驶风格。我们还证明了我们的框架提供了向纳什均衡收敛的理论保证。据我们所知，这是首次使用深度虚构博弈训练互动驾驶策略的研究。我们使用INTERACTION数据集验证了基于深度虚构博弈的潜在微分博弈（DFP-PDG）框架的有效性。结果表明，所提出的方法能够学习到类人的驾驶策略，学习到的个体权重有效地捕捉了驾驶者侵略性和偏好的变化。此外，消融实验强调了模型中每个组件的重要性。 

---
# Lower Bound on Howard Policy Iteration for Deterministic Markov Decision Processes 

**Title (ZH)**: 确定性马尔可夫决策过程中的何德政策迭代的下界 

**Authors**: Ali Asadi, Krishnendu Chatterjee, Jakob de Raaij  

**Link**: [PDF](https://arxiv.org/pdf/2506.12254)  

**Abstract**: Deterministic Markov Decision Processes (DMDPs) are a mathematical framework for decision-making where the outcomes and future possible actions are deterministically determined by the current action taken. DMDPs can be viewed as a finite directed weighted graph, where in each step, the controller chooses an outgoing edge. An objective is a measurable function on runs (or infinite trajectories) of the DMDP, and the value for an objective is the maximal cumulative reward (or weight) that the controller can guarantee. We consider the classical mean-payoff (aka limit-average) objective, which is a basic and fundamental objective.
Howard's policy iteration algorithm is a popular method for solving DMDPs with mean-payoff objectives. Although Howard's algorithm performs well in practice, as experimental studies suggested, the best known upper bound is exponential and the current known lower bound is as follows: For the input size $I$, the algorithm requires $\tilde{\Omega}(\sqrt{I})$ iterations, where $\tilde{\Omega}$ hides the poly-logarithmic factors, i.e., the current lower bound on iterations is sub-linear with respect to the input size. Our main result is an improved lower bound for this fundamental algorithm where we show that for the input size $I$, the algorithm requires $\tilde{\Omega}(I)$ iterations. 

**Abstract (ZH)**: 确定性马尔可夫决策过程中的平均收益目标的改进下界分析 

---
# Reversing the Paradigm: Building AI-First Systems with Human Guidance 

**Title (ZH)**: 反转范式：在人类指导下的AI优先系统构建 

**Authors**: Cosimo Spera, Garima Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.12245)  

**Abstract**: The relationship between humans and artificial intelligence is no longer science fiction -- it's a growing reality reshaping how we live and work. AI has moved beyond research labs into everyday life, powering customer service chats, personalizing travel, aiding doctors in diagnosis, and supporting educators. What makes this moment particularly compelling is AI's increasing collaborative nature. Rather than replacing humans, AI augments our capabilities -- automating routine tasks, enhancing decisions with data, and enabling creativity in fields like design, music, and writing. The future of work is shifting toward AI agents handling tasks autonomously, with humans as supervisors, strategists, and ethical stewards. This flips the traditional model: instead of humans using AI as a tool, intelligent agents will operate independently within constraints, managing everything from scheduling and customer service to complex workflows. Humans will guide and fine-tune these agents to ensure alignment with goals, values, and context.
This shift offers major benefits -- greater efficiency, faster decisions, cost savings, and scalability. But it also brings risks: diminished human oversight, algorithmic bias, security flaws, and a widening skills gap. To navigate this transition, organizations must rethink roles, invest in upskilling, embed ethical principles, and promote transparency. This paper examines the technological and organizational changes needed to enable responsible adoption of AI-first systems -- where autonomy is balanced with human intent, oversight, and values. 

**Abstract (ZH)**: 人类与人工智能的关系不再局限于科幻——它是一个 growing 现实，正在重塑我们的生活和工作方式。人工智能已从研究实验室走进日常生活，推动客服聊天、个性化旅行、辅助医生诊断和教育工作者的支持。这一时刻尤其引人注目的是人工智能日益增强的协作性。人工智能不是取代人类，而是增强我们的能力——自动化常规任务，利用数据增强决策，并在设计、音乐和写作等领域促进创造力。工作未来将转向人工智能代理自主处理任务，人类作为监管者、策略制定者和道德监护人。这一转变颠覆了传统模型：不再是人类利用人工智能作为工具，而是智能代理在约束条件下独立操作，管理从调度和客户服务到复杂工作流程的一切。人类将指导和微调这些代理，以确保与目标、价值观和情境相一致。这一转变带来了重大益处——更高的效率、更快的决策、成本节约和可扩展性。但也带来了风险——人类监督的减少、算法偏差、安全漏洞以及技能差距的扩大。为应对这一转型，组织必须重新思考角色、投资技能提升、嵌入道德原则并促进透明度。本文探讨了为负责任地采用以人工智能为主导的系统所需的技术和组织变革——平衡自主性与人类意图、监督和价值观。 

---
# Privacy Reasoning in Ambiguous Contexts 

**Title (ZH)**: 在含糊情境中的隐私推理 

**Authors**: Ren Yi, Octavian Suciu, Adria Gascon, Sarah Meiklejohn, Eugene Bagdasarian, Marco Gruteser  

**Link**: [PDF](https://arxiv.org/pdf/2506.12241)  

**Abstract**: We study the ability of language models to reason about appropriate information disclosure - a central aspect of the evolving field of agentic privacy. Whereas previous works have focused on evaluating a model's ability to align with human decisions, we examine the role of ambiguity and missing context on model performance when making information-sharing decisions. We identify context ambiguity as a crucial barrier for high performance in privacy assessments. By designing Camber, a framework for context disambiguation, we show that model-generated decision rationales can reveal ambiguities and that systematically disambiguating context based on these rationales leads to significant accuracy improvements (up to 13.3\% in precision and up to 22.3\% in recall) as well as reductions in prompt sensitivity. Overall, our results indicate that approaches for context disambiguation are a promising way forward to enhance agentic privacy reasoning. 

**Abstract (ZH)**: 我们研究了语言模型在信息披露推理方面的能力——这是不断发展的代理隐私领域的一个核心方面。与以往研究专注于评估模型与人类决策的一致性不同，我们探讨了在进行信息共享决策时，模糊性和缺失背景对模型性能的影响。我们识别出背景模糊性是高性能隐私评估的一个关键障碍。通过设计Camber框架，一种背景消歧框架，我们证明了模型生成的决策理由可以揭示模糊性，并且根据这些理由系统地消歧背景可以显著提高准确率（精确度提高至多13.3%、召回率提高至多22.3%），以及降低提示敏感性。总体而言，我们的结果表明，背景消歧方法是增强代理隐私推理的一个有前景的方向。 

---
# PRO-V: An Efficient Program Generation Multi-Agent System for Automatic RTL Verification 

**Title (ZH)**: PRO-V: 一种高效的程序生成多代理系统，用于自动RTL验证 

**Authors**: Yujie Zhao, Zhijing Wu, Hejia Zhang, Zhongming Yu, Wentao Ni, Chia-Tung Ho, Haoxing Ren, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12200)  

**Abstract**: LLM-assisted hardware verification is gaining substantial attention due to its potential to significantly reduce the cost and effort of crafting effective testbenches. It also serves as a critical enabler for LLM-aided end-to-end hardware language design. However, existing current LLMs often struggle with Register Transfer Level (RTL) code generation, resulting in testbenches that exhibit functional errors in Hardware Description Languages (HDL) logic. Motivated by the strong performance of LLMs in Python code generation under inference-time sampling strategies, and their promising capabilities as judge agents, we propose PRO-V a fully program generation multi-agent system for robust RTL verification. Pro-V incorporates an efficient best-of-n iterative sampling strategy to enhance the correctness of generated testbenches. Moreover, it introduces an LLM-as-a-judge aid validation framework featuring an automated prompt generation pipeline. By converting rule-based static analysis from the compiler into natural language through in-context learning, this pipeline enables LLMs to assist the compiler in determining whether verification failures stem from errors in the RTL design or the testbench. PRO-V attains a verification accuracy of 87.17% on golden RTL implementations and 76.28% on RTL mutants. Our code is open-sourced at this https URL. 

**Abstract (ZH)**: LLM辅助硬件验证在降低有效测试平台成本和努力方面正在获得广泛关注，并作为端到端硬件语言设计的關鍵使能器。PRO-V：一种稳健的RTL验证全方位程序生成多智能体系统 

---
# Artificial Intelligence and Machine Learning in the Development of Vaccines and Immunotherapeutics Yesterday, Today, and Tomorrow 

**Title (ZH)**: 人工智能与机器学习在疫苗和免疫治疗的发展中：昨天、今天和明天 

**Authors**: Elhoucine Elfatimi, Yassir Lekbach, Swayam Prakash, Lbachir BenMohamed  

**Link**: [PDF](https://arxiv.org/pdf/2506.12185)  

**Abstract**: In the past, the development of vaccines and immunotherapeutics relied heavily on trial-and-error experimentation and extensive in vivo testing, often requiring years of pre-clinical and clinical trials. Today, artificial intelligence (AI) and deep learning (DL) are actively transforming vaccine and immunotherapeutic design, by (i) offering predictive frameworks that support rapid, data-driven decision-making; (ii) increasingly being implemented as time- and resource-efficient strategies that integrate computational models, systems vaccinology, and multi-omics data to better phenotype, differentiate, and classify patient diseases and cancers; predict patients' immune responses; and identify the factors contributing to optimal vaccine and immunotherapeutic protective efficacy; (iii) refining the selection of B- and T-cell antigen/epitope targets to enhance efficacy and durability of immune protection; and (iv) enabling a deeper understanding of immune regulation, immune evasion, immune checkpoints, and regulatory pathways. The future of AI and DL points toward (i) replacing animal preclinical testing of drugs, vaccines, and immunotherapeutics with computational-based models, as recently proposed by the United States FDA; and (ii) enabling real-time in vivo modeling for immunobridging and prediction of protection in clinical trials. This may result in a fast and transformative shift for the development of personal vaccines and immunotherapeutics against infectious pathogens and cancers. 

**Abstract (ZH)**: 人工智能和深度学习在疫苗和免疫治疗设计中的应用正在经历革命性的变革：从试验性方法到计算驱动的快速决策 

---
# Diagnosing and Improving Diffusion Models by Estimating the Optimal Loss Value 

**Title (ZH)**: 通过估计最优损失值来诊断和提升扩散模型 

**Authors**: Yixian Xu, Shengjie Luo, Liwei Wang, Di He, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13763)  

**Abstract**: Diffusion models have achieved remarkable success in generative modeling. Despite more stable training, the loss of diffusion models is not indicative of absolute data-fitting quality, since its optimal value is typically not zero but unknown, leading to confusion between large optimal loss and insufficient model capacity. In this work, we advocate the need to estimate the optimal loss value for diagnosing and improving diffusion models. We first derive the optimal loss in closed form under a unified formulation of diffusion models, and develop effective estimators for it, including a stochastic variant scalable to large datasets with proper control of variance and bias. With this tool, we unlock the inherent metric for diagnosing the training quality of mainstream diffusion model variants, and develop a more performant training schedule based on the optimal loss. Moreover, using models with 120M to 1.5B parameters, we find that the power law is better demonstrated after subtracting the optimal loss from the actual training loss, suggesting a more principled setting for investigating the scaling law for diffusion models. 

**Abstract (ZH)**: 扩散模型已在生成建模中取得了显著成功。尽管训练更为稳定，但扩散模型的损失值并不一定意味着绝对的数据拟合质量，因为其最优值通常不为零且未知，导致大最优损失与模型容量不足之间的混淆。在本文中，我们强调估计最优损失值以诊断和提升扩散模型的重要性。我们首先在统一的扩散模型框架下推导出最优损失的闭式解，并开发出有效的估计器，包括一种可扩展到大规模数据集的有效随机变体，以适当控制方差和偏差。借助此工具，我们为流行的扩散模型变体解锁了固有的诊断训练质量的度量，并基于最优损失开发出更高效的训练计划。此外，使用包含120M到1.5B参数的模型，我们发现，在实际训练损失中减去最优损失后，功率律得到了更好的呈现，这表明了在研究扩散模型的扩展律时采用更为原则性的设置。 

---
# BanditWare: A Contextual Bandit-based Framework for Hardware Prediction 

**Title (ZH)**: BanditWare：基于上下文-bandit的硬件预测框架 

**Authors**: Tainã Coleman, Hena Ahmed, Ravi Shende, Ismael Perez, Ïlkay Altintaş  

**Link**: [PDF](https://arxiv.org/pdf/2506.13730)  

**Abstract**: Distributed computing systems are essential for meeting the demands of modern applications, yet transitioning from single-system to distributed environments presents significant challenges. Misallocating resources in shared systems can lead to resource contention, system instability, degraded performance, priority inversion, inefficient utilization, increased latency, and environmental impact.
We present BanditWare, an online recommendation system that dynamically selects the most suitable hardware for applications using a contextual multi-armed bandit algorithm. BanditWare balances exploration and exploitation, gradually refining its hardware recommendations based on observed application performance while continuing to explore potentially better options. Unlike traditional statistical and machine learning approaches that rely heavily on large historical datasets, BanditWare operates online, learning and adapting in real-time as new workloads arrive.
We evaluated BanditWare on three workflow applications: Cycles (an agricultural science scientific workflow) BurnPro3D (a web-based platform for fire science) and a matrix multiplication application. Designed for seamless integration with the National Data Platform (NDP), BanditWare enables users of all experience levels to optimize resource allocation efficiently. 

**Abstract (ZH)**: 分布式计算系统对于满足现代应用的需求是必不可少的，但从单系统环境向分布式环境的转变面临着巨大的挑战。在共享系统中错误分配资源可能导致资源争用、系统不稳定、性能下降、优先级反转、资源利用效率低下、延迟增加以及环境影响。

我们提出了BanditWare，这是一种在线推荐系统，使用上下文多臂赌博机算法动态选择最适合的应用程序的硬件。BanditWare在不断探索和利用之间寻求平衡，根据观察到的应用程序性能逐步优化其硬件建议，同时继续探索可能更好的选项。与依赖于大量历史数据的传统统计和机器学习方法不同，BanditWare在线操作，能够实时学习和适应新的工作负载。

我们分别在三个工作流应用程序上评估了BanditWare：Cycles（一个农业科学科学工作流）、BurnPro3D（一个基于Web的防火科学平台）以及一个矩阵乘法应用程序。BanditWare设计用于无缝集成到国家数据平台（NDP）中，使得所有经验级别的用户都能够有效优化资源分配。 

---
# Value-Free Policy Optimization via Reward Partitioning 

**Title (ZH)**: 无需价值偏好的政策优化通过奖励分割 

**Authors**: Bilal Faye, Hanane Azzag, Mustapha Lebbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.13702)  

**Abstract**: Single-trajectory reinforcement learning (RL) methods aim to optimize policies from datasets consisting of (prompt, response, reward) triplets, where scalar rewards are directly available. This supervision format is highly practical, as it mirrors real-world human feedback, such as thumbs-up/down signals, and avoids the need for structured preference annotations. In contrast, pairwise preference-based methods like Direct Preference Optimization (DPO) rely on datasets with both preferred and dispreferred responses, which are harder to construct and less natural to collect. Among single-trajectory approaches, Direct Reward Optimization (DRO) has shown strong empirical performance due to its simplicity and stability. However, DRO requires approximating a value function, which introduces several limitations: high off-policy variance, coupling between policy and value learning, and a lack of absolute supervision on the policy itself. We introduce Reward Partitioning Optimization (RPO), a new method that resolves these limitations by removing the need to model the value function. Instead, RPO normalizes observed rewards using a partitioning approach estimated directly from data. This leads to a straightforward supervised learning objective on the policy, with no auxiliary models and no joint optimization. RPO provides direct and stable supervision on the policy, making it robust and easy to implement in practice. We validate RPO on scalar-feedback language modeling tasks using Flan-T5 encoder-decoder models. Our results demonstrate that RPO outperforms existing single-trajectory baselines such as DRO and Kahneman-Tversky Optimization (KTO). These findings confirm that RPO is a simple, effective, and theoretically grounded method for single-trajectory policy optimization. 

**Abstract (ZH)**: 单轨迹强化学习（RL）方法旨在通过由(prompt, response, reward)三元组组成的数据集优化策略，其中标量奖励可以直接获得。这种监督格式非常实用，因为它模仿了现实生活中的人类反馈，如拇指点赞/反对信号，并避免了需要结构化偏好的标注。相比之下，基于成对偏好的方法，如直接偏好优化（DPO），依赖于包含偏好和非偏好响应的数据集，这些数据集更难构建且收集起来不太自然。在单轨迹方法中，直接奖励优化（DRO）由于其简单性和稳定性表现出强大的 empirical 性能。然而，DRO 需要近似一个价值函数，这引入了几种限制：高离策方差、策略和价值学习之间的耦合以及策略本身的绝对监督缺失。我们提出了一种新的 Reward Partitioning Optimization（RPO）方法，通过消除建模价值函数的需要来解决这些限制。相反，RPO 通过直接从数据估计的分区方法对观察到的奖励进行归一化。这导致了一个直接且稳定的策略监督学习目标，不需要辅助模型且不需要联合优化。RPO 为策略提供了直接且稳定的监督，使其在实践中更 robust 和容易实现。我们在使用 Flan-T5 编解码器模型的标量反馈语言建模任务上验证了 RPO。我们的结果表明，RPO 在现有的单轨迹基线方法，如 DRO 和 Kahneman-Tversky 方法（KTO）上表现更优。这些发现证实了 RPO 是一种简单、有效且理论基础坚实的方法，用于单轨迹策略优化。 

---
# Meta-learning how to Share Credit among Macro-Actions 

**Title (ZH)**: 宏动作中的功劳共享元学习 

**Authors**: Ionel-Alexandru Hosu, Traian Rebedea, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13690)  

**Abstract**: One proposed mechanism to improve exploration in reinforcement learning is through the use of macro-actions. Paradoxically though, in many scenarios the naive addition of macro-actions does not lead to better exploration, but rather the opposite. It has been argued that this was caused by adding non-useful macros and multiple works have focused on mechanisms to discover effectively environment-specific useful macros. In this work, we take a slightly different perspective. We argue that the difficulty stems from the trade-offs between reducing the average number of decisions per episode versus increasing the size of the action space. Namely, one typically treats each potential macro-action as independent and atomic, hence strictly increasing the search space and making typical exploration strategies inefficient. To address this problem we propose a novel regularization term that exploits the relationship between actions and macro-actions to improve the credit assignment mechanism by reducing the effective dimension of the action space and, therefore, improving exploration. The term relies on a similarity matrix that is meta-learned jointly with learning the desired policy. We empirically validate our strategy looking at macro-actions in Atari games, and the StreetFighter II environment. Our results show significant improvements over the Rainbow-DQN baseline in all environments. Additionally, we show that the macro-action similarity is transferable to related environments. We believe this work is a small but important step towards understanding how the similarity-imposed geometry on the action space can be exploited to improve credit assignment and exploration, therefore making learning more effective. 

**Abstract (ZH)**: 一种通过宏动作提高强化学习探索机制的研究：从减少每episode的平均决策次数与增加动作空间大小的权衡视角出发 

---
# Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning 

**Title (ZH)**: 自我中心视频超长推理的工具链思考（Ego-R1） 

**Authors**: Shulin Tian, Ruiqi Wang, Hongming Guo, Penghao Wu, Yuhao Dong, Xiuying Wang, Jingkang Yang, Hao Zhang, Hongyuan Zhu, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13654)  

**Abstract**: We introduce Ego-R1, a novel framework for reasoning over ultra-long (i.e., in days and weeks) egocentric videos, which leverages a structured Chain-of-Tool-Thought (CoTT) process, orchestrated by an Ego-R1 Agent trained via reinforcement learning (RL). Inspired by human problem-solving strategies, CoTT decomposes complex reasoning into modular steps, with the RL agent invoking specific tools, one per step, to iteratively and collaboratively answer sub-questions tackling such tasks as temporal retrieval and multi-modal understanding. We design a two-stage training paradigm involving supervised finetuning (SFT) of a pretrained language model using CoTT data and RL to enable our agent to dynamically propose step-by-step tools for long-range reasoning. To facilitate training, we construct a dataset called Ego-R1 Data, which consists of Ego-CoTT-25K for SFT and Ego-QA-4.4K for RL. Furthermore, our Ego-R1 agent is evaluated on a newly curated week-long video QA benchmark, Ego-R1 Bench, which contains human-verified QA pairs from hybrid sources. Extensive results demonstrate that the dynamic, tool-augmented chain-of-thought reasoning by our Ego-R1 Agent can effectively tackle the unique challenges of understanding ultra-long egocentric videos, significantly extending the time coverage from few hours to a week. 

**Abstract (ZH)**: 基于结构化工具-思考链（CoTT）的Ego-R1：一种用于超长周期自我中心视频推理的新框架 

---
# Graph-Convolution-Beta-VAE for Synthetic Abdominal Aorta Aneurysm Generation 

**Title (ZH)**: 基于图卷积-贝塔VAE的合成腹主动脉瘤生成 

**Authors**: Francesco Fabbri, Martino Andrea Scarpolini, Angelo Iollo, Francesco Viola, Francesco Tudisco  

**Link**: [PDF](https://arxiv.org/pdf/2506.13628)  

**Abstract**: Synthetic data generation plays a crucial role in medical research by mitigating privacy concerns and enabling large-scale patient data analysis. This study presents a beta-Variational Autoencoder Graph Convolutional Neural Network framework for generating synthetic Abdominal Aorta Aneurysms (AAA). Using a small real-world dataset, our approach extracts key anatomical features and captures complex statistical relationships within a compact disentangled latent space. To address data limitations, low-impact data augmentation based on Procrustes analysis was employed, preserving anatomical integrity. The generation strategies, both deterministic and stochastic, manage to enhance data diversity while ensuring realism. Compared to PCA-based approaches, our model performs more robustly on unseen data by capturing complex, nonlinear anatomical variations. This enables more comprehensive clinical and statistical analyses than the original dataset alone. The resulting synthetic AAA dataset preserves patient privacy while providing a scalable foundation for medical research, device testing, and computational modeling. 

**Abstract (ZH)**: 合成数据生成在减轻隐私担忧和促进大规模患者数据分析方面对医学研究发挥着关键作用。本研究提出了一种beta-变分自编码器图卷积神经网络框架，用于生成腹主动脉瘤（AAA）的合成数据。通过小规模真实数据集，我们的方法提取关键解剖特征并捕捉紧凑分离潜空间内的复杂统计关系。为解决数据限制问题，我们采用了基于Procrustes分析的低影响数据增强方法，保持了解剖完整性。生成策略，无论是确定性的还是随机性的，都能够增强数据多样性并确保真实性。与基于PCA的方法相比，我们的模型在未见数据上表现更稳健，因为它能够捕捉复杂的非线性解剖变异。这使得可以比原数据集更全面地进行临床和统计分析。生成的数据集在保护患者隐私的同时，为医学研究、设备测试和计算建模提供了可扩展的基础。 

---
# EBS-CFL: Efficient and Byzantine-robust Secure Clustered Federated Learning 

**Title (ZH)**: EBS-CFL: 高效且抗拜占庭容错的安全集群联邦学习 

**Authors**: Zhiqiang Li, Haiyong Bao, Menghong Guan, Hao Pan, Cheng Huang, Hong-Ning Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13612)  

**Abstract**: Despite federated learning (FL)'s potential in collaborative learning, its performance has deteriorated due to the data heterogeneity of distributed users. Recently, clustered federated learning (CFL) has emerged to address this challenge by partitioning users into clusters according to their similarity. However, CFL faces difficulties in training when users are unwilling to share their cluster identities due to privacy concerns. To address these issues, we present an innovative Efficient and Robust Secure Aggregation scheme for CFL, dubbed EBS-CFL. The proposed EBS-CFL supports effectively training CFL while maintaining users' cluster identity confidentially. Moreover, it detects potential poisonous attacks without compromising individual client gradients by discarding negatively correlated gradients and aggregating positively correlated ones using a weighted approach. The server also authenticates correct gradient encoding by clients. EBS-CFL has high efficiency with client-side overhead O(ml + m^2) for communication and O(m^2l) for computation, where m is the number of cluster identities, and l is the gradient size. When m = 1, EBS-CFL's computational efficiency of client is at least O(log n) times better than comparison schemes, where n is the number of this http URL addition, we validate the scheme through extensive experiments. Finally, we theoretically prove the scheme's security. 

**Abstract (ZH)**: 一种高效的稳健安全聚合方案EBS-CFL及其在聚类联邦学习中的应用 

---
# A Hybrid Artificial Intelligence Method for Estimating Flicker in Power Systems 

**Title (ZH)**: 电力系统中混合人工智能方法估计闪变 

**Authors**: Javad Enayati, Pedram Asef, Alexandre Benoit  

**Link**: [PDF](https://arxiv.org/pdf/2506.13611)  

**Abstract**: This paper introduces a novel hybrid AI method combining H filtering and an adaptive linear neuron network for flicker component estimation in power distribution this http URL proposed method leverages the robustness of the H filter to extract the voltage envelope under uncertain and noisy conditions followed by the use of ADALINE to accurately identify flicker frequencies embedded in the this http URL synergy enables efficient time domain estimation with rapid convergence and noise resilience addressing key limitations of existing frequency domain this http URL conventional techniques this hybrid AI model handles complex power disturbances without prior knowledge of noise characteristics or extensive this http URL validate the method performance we conduct simulation studies based on IEC Standard 61000 4 15 supported by statistical analysis Monte Carlo simulations and real world this http URL demonstrate superior accuracy robustness and reduced computational load compared to Fast Fourier Transform and Discrete Wavelet Transform based estimators. 

**Abstract (ZH)**: 本文介绍了一种结合H滤波器和自适应线性神经网络的新型混合人工智能方法，用于电力分配中的闪烁成分估计。该方法利用H滤波器在不确定性及噪声条件下提取电压包络的稳健性，继而使用ADALINE准确识别嵌入其中的闪烁频率。这种 synergy 能够实现高效的时域估计，具有快速收敛和抗噪声的能力，解决了现有频域技术的 key limitations。这种混合人工智能模型能够在无需了解噪声特性或进行大量预先知识的情况下处理复杂电力扰动。为了验证该方法的性能，我们基于IEC标准61000-4-15进行了仿真研究，支持以统计分析、蒙特卡洛模拟和实际应用为基础的实验。研究结果表明，与基于快速傅里叶变换和离散小波变换的估计器相比，该方法具有更高的准确性和鲁棒性，并且计算负载更低。 

---
# Flexible-length Text Infilling for Discrete Diffusion Models 

**Title (ZH)**: 长度可变的文本填充用于离散扩散模型 

**Authors**: Andrew Zhang, Anushka Sivakumar, Chiawei Tang, Chris Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2506.13579)  

**Abstract**: Discrete diffusion models are a new class of text generators that offer advantages such as bidirectional context use, parallelizable generation, and flexible prompting compared to autoregressive models. However, a critical limitation of discrete diffusion models is their inability to perform flexible-length or flexible-position text infilling without access to ground-truth positional data. We introduce \textbf{DDOT} (\textbf{D}iscrete \textbf{D}iffusion with \textbf{O}ptimal \textbf{T}ransport Position Coupling), the first discrete diffusion model to overcome this challenge. DDOT jointly denoises token values and token positions, employing a novel sample-level Optimal Transport (OT) coupling. This coupling preserves relative token ordering while dynamically adjusting the positions and length of infilled segments, a capability previously missing in text diffusion. Our method is orthogonal to existing discrete text diffusion methods and is compatible with various pretrained text denoisers. Extensive experiments on text infilling benchmarks such as One-Billion-Word and Yelp demonstrate that DDOT outperforms naive diffusion baselines. Furthermore, DDOT achieves performance on par with state-of-the-art non-autoregressive models and enables significant improvements in training efficiency and flexibility. 

**Abstract (ZH)**: 离散扩散模型是一种新的文本生成器，相较于自回归模型，它具有双向上下文利用、并行生成和灵活提示等优势。然而，离散扩散模型的一个关键局限是它们无法在没有地面真实位置数据的情况下进行灵活长度或灵活位置的文本填充。我们引入了**DDOT（离散扩散与最优传输位置耦合）**，这是第一个克服这一挑战的离散扩散模型。DDOT同时去噪词值和词位置，采用一种新颖的样本级最优传输（OT）耦合。这种耦合保留了词的相对顺序，同时动态调整填充段的位置和长度，这是文本扩散中以前缺失的能力。我们的方法与现有的离散文本扩散方法正交，并且兼容各种预训练的文本去噪器。在One-Billion-Word和Yelp等文本填充基准测试中，DDOT在性能上优于简单的扩散基线模型。此外，DDOT在性能上与最先进的非自回归模型相当，并能显著提高训练效率和灵活性。 

---
# A Production Scheduling Framework for Reinforcement Learning Under Real-World Constraints 

**Title (ZH)**: 基于实际约束条件的强化学习生产调度框架 

**Authors**: Jonathan Hoss, Felix Schelling, Noah Klarmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.13566)  

**Abstract**: The classical Job Shop Scheduling Problem (JSSP) focuses on optimizing makespan under deterministic constraints. Real-world production environments introduce additional complexities that cause traditional scheduling approaches to be less effective. Reinforcement learning (RL) holds potential in addressing these challenges, as it allows agents to learn adaptive scheduling strategies. However, there is a lack of a comprehensive, general-purpose frameworks for effectively training and evaluating RL agents under real-world constraints. To address this gap, we propose a modular framework that extends classical JSSP formulations by incorporating key \mbox{real-world} constraints inherent to the shopfloor, including transport logistics, buffer management, machine breakdowns, setup times, and stochastic processing conditions, while also supporting multi-objective optimization. The framework is a customizable solution that offers flexibility in defining problem instances and configuring simulation parameters, enabling adaptation to diverse production scenarios. A standardized interface ensures compatibility with various RL approaches, providing a robust environment for training RL agents and facilitating the standardized comparison of different scheduling methods under dynamic and uncertain conditions. We release JobShopLab as an open-source tool for both research and industrial applications, accessible at: this https URL 

**Abstract (ZH)**: 经典的作业车间调度问题（JSSP）专注于在确定性约束条件下优化生产周期。现实世界的生产环境引入了额外的复杂性，使得传统的调度方法 effectiveness降低。强化学习（RL）有可能通过允许代理学习适应性调度策略来应对这些挑战。然而，缺乏适用于实际约束条件下有效训练和评估RL代理的综合通用框架。为解决这一问题，我们提出了一种模块化框架，该框架扩展了经典的JSSP形式化模型，整合了车间环境固有的关键现实约束，包括物流运输、缓冲管理、机器故障、设置时间以及随机加工条件，同时支持多目标优化。该框架是一个可定制的解决方案，提供了定义问题实例和配置仿真参数的灵活性，以适应不同的生产情景。标准化的接口确保了与各种RL方法的兼容性，提供了一个稳健的环境来训练RL代理，并促进了在动态和不确定条件下的不同调度方法的标准比较。我们发布JobShopLab作为一款开源工具，适用于研究和工业应用，网址为: this https URL。 

---
# Seismic Acoustic Impedance Inversion Framework Based on Conditional Latent Generative Diffusion Model 

**Title (ZH)**: 基于条件潜在生成扩散模型的地震声学阻抗反演框架 

**Authors**: Jie Chen, Hongling Chen, Jinghuai Gao, Chuangji Meng, Tao Yang, XinXin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13529)  

**Abstract**: Seismic acoustic impedance plays a crucial role in lithological identification and subsurface structure interpretation. However, due to the inherently ill-posed nature of the inversion problem, directly estimating impedance from post-stack seismic data remains highly challenging. Recently, diffusion models have shown great potential in addressing such inverse problems due to their strong prior learning and generative capabilities. Nevertheless, most existing methods operate in the pixel domain and require multiple iterations, limiting their applicability to field data. To alleviate these limitations, we propose a novel seismic acoustic impedance inversion framework based on a conditional latent generative diffusion model, where the inversion process is made in latent space. To avoid introducing additional training overhead when embedding conditional inputs, we design a lightweight wavelet-based module into the framework to project seismic data and reuse an encoder trained on impedance to embed low-frequency impedance into the latent space. Furthermore, we propose a model-driven sampling strategy during the inversion process of this framework to enhance accuracy and reduce the number of required diffusion steps. Numerical experiments on a synthetic model demonstrate that the proposed method achieves high inversion accuracy and strong generalization capability within only a few diffusion steps. Moreover, application to field data reveals enhanced geological detail and higher consistency with well-log measurements, validating the effectiveness and practicality of the proposed approach. 

**Abstract (ZH)**: seismic acoustic阻抗在地质识别和地下结构解释中扮演着关键角色。然而，由于逆问题的固有病态性质，直接从后处理地震数据中估计阻抗仍然极具挑战性。近年来，扩散模型因其强大的先验学习能力和生成能力，在解决此类逆问题方面展示了巨大潜力。尽管如此，现有的大多数方法都在像素域中操作，并需要多次迭代，限制了它们在实地数据中的应用。为克服这些限制，我们提出了一种基于条件潜在生成扩散模型的新型地震声阻抗逆演框架，其中逆演过程在潜在空间中进行。为了在嵌入条件输入时避免引入额外的训练开销，我们在框架中设计了一个轻量级小波模块来投影地震数据，并利用在阻抗上预训练的编码器将低频阻抗嵌入到潜在空间中。此外，在这个框架的逆演过程中，我们提出了一种模型驱动的采样策略，以增强精度并减少所需的扩散步骤数量。数值实验表明，所提出的方法仅在几个扩散步骤内实现了高逆演精度和强大的泛化能力。此外，对实地数据的应用揭示了增强的地质细节，并与井壁测量数据具有更高的一致性，验证了所提方法的有效性和实用性。 

---
# A Two-stage Optimization Method for Wide-range Single-electron Quantum Magnetic Sensing 

**Title (ZH)**: 宽范围单电子量子磁传感的两阶段优化方法 

**Authors**: Shiqian Guo, Jianqing Liu, Thinh Le, Huaiyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13469)  

**Abstract**: Quantum magnetic sensing based on spin systems has emerged as a new paradigm for detecting ultra-weak magnetic fields with unprecedented sensitivity, revitalizing applications in navigation, geo-localization, biology, and beyond. At the heart of quantum magnetic sensing, from the protocol perspective, lies the design of optimal sensing parameters to manifest and then estimate the underlying signals of interest (SoI). Existing studies on this front mainly rely on adaptive algorithms based on black-box AI models or formula-driven principled searches. However, when the SoI spans a wide range and the quantum sensor has physical constraints, these methods may fail to converge efficiently or optimally, resulting in prolonged interrogation times and reduced sensing accuracy. In this work, we report the design of a new protocol using a two-stage optimization method. In the 1st Stage, a Bayesian neural network with a fixed set of sensing parameters is used to narrow the range of SoI. In the 2nd Stage, a federated reinforcement learning agent is designed to fine-tune the sensing parameters within a reduced search space. The proposed protocol is developed and evaluated in a challenging context of single-shot readout of an NV-center electron spin under a constrained total sensing time budget; and yet it achieves significant improvements in both accuracy and resource efficiency for wide-range D.C. magnetic field estimation compared to the state of the art. 

**Abstract (ZH)**: 基于自旋系统的量子磁感应已成为检测超弱磁场的一种新范式，具有前所未有的灵敏度，重新激活了导航、地理定位、生物学等领域中的应用。从协议角度而言，量子磁感应的核心在于设计最优的感应参数以体现并估计感兴趣的信号（SoI）。现有研究主要依赖于基于黑盒AI模型的自适应算法或基于公式的精原则搜索。然而，当SoI的范围广泛且量子传感器受到物理约束时，这些方法可能无法高效或最优地收敛，导致探测时间延长和探测精度降低。在本文中，我们报告了一种新的协议设计，该协议采用两阶段优化方法。在第一阶段，使用固定参数的贝叶斯神经网络来缩小SoI的范围。在第二阶段，设计了一个联邦强化学习代理来在缩减的搜索空间内细调感应参数。所提出协议在受限的总探测时间预算下实现单次读出NV中心电子自旋的挑战性环境中进行开发和评估；与现有技术相比，它在宽范围直流磁场估计的准确性和资源效率方面均取得了显著改进。 

---
# An Interdisciplinary Approach to Human-Centered Machine Translation 

**Title (ZH)**: 以人为本的跨学科机器翻译方法 

**Authors**: Marine Carpuat, Omri Asscher, Kalika Bali, Luisa Bentivogli, Frédéric Blain, Lynne Bowker, Monojit Choudhury, Hal Daumé III, Kevin Duh, Ge Gao, Alvin Grissom II, Marzena Karpinska, Elaine C. Khoong, William D. Lewis, André F. T. Martins, Mary Nurminen, Douglas W. Oard, Maja Popovic, Michel Simard, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2506.13468)  

**Abstract**: Machine Translation (MT) tools are widely used today, often in contexts where professional translators are not present. Despite progress in MT technology, a gap persists between system development and real-world usage, particularly for non-expert users who may struggle to assess translation reliability. This paper advocates for a human-centered approach to MT, emphasizing the alignment of system design with diverse communicative goals and contexts of use. We survey the literature in Translation Studies and Human-Computer Interaction to recontextualize MT evaluation and design to address the diverse real-world scenarios in which MT is used today. 

**Abstract (ZH)**: 机器翻译工具在缺乏专业译者的情况下广泛使用，尽管机器翻译技术取得了进展，但系统开发与实际应用之间仍存在差距，尤其是在非专家用户中，他们可能难以评估翻译的可靠性。本文倡导以人为本的机器翻译方法，强调系统设计应与多元的交流目标和使用情境相一致。我们回顾翻译研究和人机交互领域的文献，重新审视机器翻译的评估与设计，以应对当前机器翻译在各种实际场景中的应用需求。 

---
# A Neural Model for Word Repetition 

**Title (ZH)**: 一种词重复的神经模型 

**Authors**: Daniel Dager, Robin Sobczyk, Emmanuel Chemla, Yair Lakretz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13450)  

**Abstract**: It takes several years for the developing brain of a baby to fully master word repetition-the task of hearing a word and repeating it aloud. Repeating a new word, such as from a new language, can be a challenging task also for adults. Additionally, brain damage, such as from a stroke, may lead to systematic speech errors with specific characteristics dependent on the location of the brain damage. Cognitive sciences suggest a model with various components for the different processing stages involved in word repetition. While some studies have begun to localize the corresponding regions in the brain, the neural mechanisms and how exactly the brain performs word repetition remain largely unknown. We propose to bridge the gap between the cognitive model of word repetition and neural mechanisms in the human brain by modeling the task using deep neural networks. Neural models are fully observable, allowing us to study the detailed mechanisms in their various substructures and make comparisons with human behavior and, ultimately, the brain. Here, we make first steps in this direction by: (1) training a large set of models to simulate the word repetition task; (2) creating a battery of tests to probe the models for known effects from behavioral studies in humans, and (3) simulating brain damage through ablation studies, where we systematically remove neurons from the model, and repeat the behavioral study to examine the resulting speech errors in the "patient" model. Our results show that neural models can mimic several effects known from human research, but might diverge in other aspects, highlighting both the potential and the challenges for future research aimed at developing human-like neural models. 

**Abstract (ZH)**: 发展婴儿的大脑需要几年时间才能完全掌握词重复的任务——即听一个词并将其大声重复。对于成人来说，重复一个新的词，例如来自一种新语言的词，也可能是一项具有挑战性的任务。此外，脑损伤，例如中风，可能导致具有特定特征的系统性言语错误，这些特征取决于脑损伤的位置。认知科学提出了一个涉及词重复的不同处理阶段的各种成分的模型。尽管一些研究已经开始定位脑中的相应区域，但词重复的神经机制及其脑是如何执行这一任务的具体方式仍 largely unknown。我们建议通过使用深度神经网络建模词重复任务，以弥合词重复的认知模型与人类大脑的神经机制之间的差距。神经模型是完全可观察的，这使得我们可以研究其各个子结构中的详细机制，并将其与人类行为和最终的大脑进行比较。在这里，我们朝着这个目标迈出第一步，具体包括：(1) 训练大量模型以模拟词重复任务；(2) 创建一系列测试以探测模型中的已知行为研究效应；(3) 通过移除模型中的神经元进行消融研究，以模拟脑损伤，并重复行为研究以检查“患者”模型中的言语错误。我们的结果显示，神经模型可以模仿人类研究中已知的多种效应，但在其他方面可能会有所不同，这突显了未来旨在开发类似人类的神经模型的研究的潜力和挑战。 

---
# CALM: Consensus-Aware Localized Merging for Multi-Task Learning 

**Title (ZH)**: CALM：共识感知的局部合并多任务学习 

**Authors**: Kunda Yan, Min Zhang, Sen Cui, Zikun Qu, Bo Jiang, Feng Liu, Changshui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13406)  

**Abstract**: Model merging aims to integrate the strengths of multiple fine-tuned models into a unified model while preserving task-specific capabilities. Existing methods, represented by task arithmetic, are typically classified into global- and local-aware methods. However, global-aware methods inevitably cause parameter interference, while local-aware methods struggle to maintain the effectiveness of task-specific details in the merged model. To address these limitations, we propose a Consensus-Aware Localized Merging (CALM) method which incorporates localized information aligned with global task consensus, ensuring its effectiveness post-merging. CALM consists of three key components: (1) class-balanced entropy minimization sampling, providing a more flexible and reliable way to leverage unsupervised data; (2) an efficient-aware framework, selecting a small set of tasks for sequential merging with high scalability; (3) a consensus-aware mask optimization, aligning localized binary masks with global task consensus and merging them conflict-free. Experiments demonstrate the superiority and robustness of our CALM, significantly outperforming existing methods and achieving performance close to traditional MTL. 

**Abstract (ZH)**: 基于共识的局部化模型合并方法（CALM）：一种整合局部信息与全局任务共识的统一模型方法 

---
# Mitigating loss of variance in ensemble data assimilation: machine learning-based and distance-free localizations for better covariance estimation 

**Title (ZH)**: 基于机器学习和无距离度量的局部化方法减轻集成数据同化中协方差估计的方差损失 

**Authors**: Vinicius L. S. Silva, Gabriel S. Seabra, Alexandre A. Emerick  

**Link**: [PDF](https://arxiv.org/pdf/2506.13362)  

**Abstract**: We propose two new methods based/inspired by machine learning for tabular data and distance-free localization to enhance the covariance estimations in an ensemble data assimilation. The main goal is to enhance the data assimilation results by mitigating loss of variance due to sampling errors. We also analyze the suitability of several machine learning models and the balance between accuracy and computational cost of the covariance estimations. We introduce two distance-free localization techniques leveraging machine learning methods specifically tailored for tabular data. The methods are integrated into the Ensemble Smoother with Multiple Data Assimilation (ES-MDA) framework. The results show that the proposed localizations improve covariance accuracy and enhance data assimilation and uncertainty quantification results. We observe reduced variance loss for the input variables using the proposed methods. Furthermore, we compare several machine learning models, assessing their suitability for the problem in terms of computational cost, and quality of the covariance estimation and data match. The influence of ensemble size is also investigated, providing insights into balancing accuracy and computational efficiency. Our findings demonstrate that certain machine learning models are more suitable for this problem. This study introduces two novel methods that mitigate variance loss for model parameters in ensemble-based data assimilation, offering practical solutions that are easy to implement and do not require any additional numerical simulation or hyperparameter tuning. 

**Abstract (ZH)**: 基于机器学习的两种新方法增强表格数据和距离无关局部化的协方差估计在集合数据同化中的应用 

---
# LapDDPM: A Conditional Graph Diffusion Model for scRNA-seq Generation with Spectral Adversarial Perturbations 

**Title (ZH)**: LapDDPM：基于谱对抗扰动的条件图扩散模型ストレスRNA测序生成 

**Authors**: Lorenzo Bini, Stephane Marchand-Maillet  

**Link**: [PDF](https://arxiv.org/pdf/2506.13344)  

**Abstract**: Generating high-fidelity and biologically plausible synthetic single-cell RNA sequencing (scRNA-seq) data, especially with conditional control, is challenging due to its high dimensionality, sparsity, and complex biological variations. Existing generative models often struggle to capture these unique characteristics and ensure robustness to structural noise in cellular networks. We introduce LapDDPM, a novel conditional Graph Diffusion Probabilistic Model for robust and high-fidelity scRNA-seq generation. LapDDPM uniquely integrates graph-based representations with a score-based diffusion model, enhanced by a novel spectral adversarial perturbation mechanism on graph edge weights. Our contributions are threefold: we leverage Laplacian Positional Encodings (LPEs) to enrich the latent space with crucial cellular relationship information; we develop a conditional score-based diffusion model for effective learning and generation from complex scRNA-seq distributions; and we employ a unique spectral adversarial training scheme on graph edge weights, boosting robustness against structural variations. Extensive experiments on diverse scRNA-seq datasets demonstrate LapDDPM's superior performance, achieving high fidelity and generating biologically-plausible, cell-type-specific samples. LapDDPM sets a new benchmark for conditional scRNA-seq data generation, offering a robust tool for various downstream biological applications. 

**Abstract (ZH)**: 基于图扩散的概率模型LapDDPM：用于高保真和生物合理单细胞RNA测序数据生成的新型条件化方法 

---
# Tady: A Neural Disassembler without Structural Constraint Violations 

**Title (ZH)**: Tady：一种无结构约束违规的神经反汇编器 

**Authors**: Siliang Qin, Fengrui Yang, Hao Wang, Bolun Zhang, Zeyu Gao, Chao Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13323)  

**Abstract**: Disassembly is a crucial yet challenging step in binary analysis. While emerging neural disassemblers show promise for efficiency and accuracy, they frequently generate outputs violating fundamental structural constraints, which significantly compromise their practical usability. To address this critical problem, we regularize the disassembly solution space by formalizing and applying key structural constraints based on post-dominance relations. This approach systematically detects widespread errors in existing neural disassemblers' outputs. These errors often originate from models' limited context modeling and instruction-level decoding that neglect global structural integrity. We introduce Tady, a novel neural disassembler featuring an improved model architecture and a dedicated post-processing algorithm, specifically engineered to address these deficiencies. Comprehensive evaluations on diverse binaries demonstrate that Tady effectively eliminates structural constraint violations and functions with high efficiency, while maintaining instruction-level accuracy. 

**Abstract (ZH)**: 二进制分析中的反汇编是一个关键但具挑战性的步骤。虽然新兴的神经网络反汇编器在效率和准确性方面展现了潜力，但它们经常生成违反基本结构约束的输出，这极大地影响了其实用性。为解决这一问题，我们通过形式化并应用基于后支配关系的关键结构约束来规范反汇编解空间。这种方法系统地检测了现有神经网络反汇编器输出中的普遍错误，这些错误通常源于模型有限的上下文建模和忽视全局结构完整性的指令级解码。我们引入了Tady，这是一种新型神经网络反汇编器，配备改进的模型架构和专用后处理算法，特别设计以解决这些问题。在多种二进制代码上的全面评估表明，Tady有效地消除了结构约束违反情况，并以高效率运行，同时保持指令级准确性。 

---
# Vine Copulas as Differentiable Computational Graphs 

**Title (ZH)**: Vine Copulas 作为可微计算图 

**Authors**: Tuoyuan Cheng, Thibault Vatter, Thomas Nagler, Kan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13318)  

**Abstract**: Vine copulas are sophisticated models for multivariate distributions and are increasingly used in machine learning. To facilitate their integration into modern ML pipelines, we introduce the vine computational graph, a DAG that abstracts the multilevel vine structure and associated computations. On this foundation, we devise new algorithms for conditional sampling, efficient sampling-order scheduling, and constructing vine structures for customized conditioning variables. We implement these ideas in torchvinecopulib, a GPU-accelerated Python library built upon PyTorch, delivering improved scalability for fitting, sampling, and density evaluation. Our experiments illustrate how gradient flowing through the vine can improve Vine Copula Autoencoders and that incorporating vines for uncertainty quantification in deep learning can outperform MC-dropout, deep ensembles, and Bayesian Neural Networks in sharpness, calibration, and runtime. By recasting vine copula models as computational graphs, our work connects classical dependence modeling with modern deep-learning toolchains and facilitates the integration of state-of-the-art copula methods in modern machine learning pipelines. 

**Abstract (ZH)**: Vine copulas是多变量分布的复杂模型，在机器学习中应用日益广泛。为便于其集成到现代ML流水线中，我们引入了vine计算图，这是一种抽象多级Vine结构及其相关计算的有向无环图。在此基础上，我们开发了新的条件采样算法、高效的采样顺序调度算法以及用于自定义条件变量的Vine结构构建算法。我们在PyTorch之上构建的GPU加速Python库torchvinecopulib中实现了这些想法，提供了更好的可扩展性，用于模型拟合、采样和密度评估。实验表明，通过Vine传播梯度可以改进Vine Copula自编码器，并且将Vine纳入深度学习中的不确定性量化可以优于MC-dropout、深集成和贝叶斯神经网络，在精确性、校准性和运行时间方面。将vine copula模型重新表述为计算图，我们的工作将经典依赖性建模与现代深度学习工具链连接起来，并促进了先进copula方法在现代机器学习流水线中的集成。 

---
# Quantitative Comparison of Fine-Tuning Techniques for Pretrained Latent Diffusion Models in the Generation of Unseen SAR Image Concepts 

**Title (ZH)**: 预训练潜在扩散模型在生成未见SAR图像概念中的微调技术定量比较 

**Authors**: Solène Debuysère, Nicolas Trouvé, Nathan Letheule, Olivier Lévêque, Elise Colin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13307)  

**Abstract**: This work investigates the adaptation of large pre-trained latent diffusion models to a radically new imaging domain: Synthetic Aperture Radar (SAR). While these generative models, originally trained on natural images, demonstrate impressive capabilities in text-to-image synthesis, they are not natively adapted to represent SAR data, which involves different physics, statistical distributions, and visual characteristics. Using a sizeable SAR dataset (on the order of 100,000 to 1 million images), we address the fundamental question of fine-tuning such models for this unseen modality. We explore and compare multiple fine-tuning strategies, including full model fine-tuning and parameter-efficient approaches like Low-Rank Adaptation (LoRA), focusing separately on the UNet diffusion backbone and the text encoder components. To evaluate generative quality, we combine several metrics: statistical distance from real SAR distributions, textural similarity via GLCM descriptors, and semantic alignment assessed with a CLIP model fine-tuned on SAR data. Our results show that a hybrid tuning strategy yields the best performance: full fine-tuning of the UNet is better at capturing low-level SAR-specific patterns, while LoRA-based partial tuning of the text encoder, combined with embedding learning of the <SAR> token, suffices to preserve prompt alignment. This work provides a methodical strategy for adapting foundation models to unconventional imaging modalities beyond natural image domains. 

**Abstract (ZH)**: 本工作探究了将大规模预训练隐空间扩散模型适应于一种全新的成像领域：合成孔径雷达（SAR）图像。尽管这些生成模型原本在自然图像上进行训练，显示出了在文本到图像合成方面的卓越能力，但它们并不天生适合表示SAR数据，后者涉及不同的物理原理、统计分布和视觉特征。利用数量级在10万到100万张SAR图像的大规模SAR数据集，我们解决了如何将此类模型调整应用于这种未曾见过的成像模态的基本问题。我们探索并比较了多种调整策略，包括完整的模型调整和高效参数调整方法（如LoRA低秩适应），分别对UNet扩散骨干网络和文本编码器组件进行了研究。为了评估生成质量，我们结合了多种指标：与真实SAR分布的统计距离、基于GLCM描述符的纹理相似性，以及使用SAR数据微调的CLIP模型进行语义对齐评估。研究结果表明，混合调整策略表现最佳：完整的UNet调整在捕捉低级SAR特定模式方面效果更好，而基于LoRA的部分文本编码器调整结合<SAR>令牌的嵌入学习足以保持提示对齐。本工作提供了一种方法论策略，用于将基础模型适应于超越自然图像领域的非传统成像模态。 

---
# Seewo's Submission to MLC-SLM: Lessons learned from Speech Reasoning Language Models 

**Title (ZH)**: Seewo向MLC-SLM的提交：来自语音推理语言模型的教训 

**Authors**: Bo Li, Chengben Xu, Wufeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13300)  

**Abstract**: This paper presents Seewo's systems for both tracks of the Multilingual Conversational Speech Language Model Challenge (MLC-SLM), addressing automatic speech recognition (ASR) and speaker diarization with ASR (SD-ASR). We introduce a multi-stage training pipeline that explicitly enhances reasoning and self-correction in speech language models for ASR. Our approach combines curriculum learning for progressive capability acquisition, Chain-of-Thought data augmentation to foster intermediate reflection, and Reinforcement Learning with Verifiable Rewards (RLVR) to further refine self-correction through reward-driven optimization. This approach achieves substantial improvements over the official challenge baselines. On the evaluation set, our best system attains a WER/CER of 11.57% for Track 1 and a tcpWER/tcpCER of 17.67% for Track 2. Comprehensive ablation studies demonstrate the effectiveness of each component under challenge constraints. 

**Abstract (ZH)**: 本文介绍了Seewo在多语言对话语音语言模型挑战（MLC-SLM）两个赛道上的系统，涵盖自动语音识别（ASR）和带有ASR的说话人聚类（SD-ASR）。我们介绍了一种多阶段训练管道，明确增强了语音语言模型在ASR中的推理和自我修正能力。我们的方法结合了课程学习以实现逐步能力获取、使用Chain-of-Thought数据增强以培养中间反思，并采用可验证奖励的强化学习（RLVR）进一步通过奖励驱动优化来细化自我修正，该方法在官方挑战基准上取得了显著改进。在评估集上，我们最佳系统在Track 1的WER/CER达到11.57%，在Track 2的tcpWER/tcpCER达到17.67%。全面的消融研究证明了在挑战约束下每个组件的有效性。 

---
# Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention 

**Title (ZH)**: 公平生成而不引入不公平扭曲：基于拆分注意力的文本到图像生成去偏差化 

**Authors**: Jeonghoon Park, Juyoung Lee, Chaeyeon Chung, Jaeseong Lee, Jaegul Choo, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13298)  

**Abstract**: Recent advancements in diffusion-based text-to-image (T2I) models have enabled the generation of high-quality and photorealistic images from text descriptions. However, they often exhibit societal biases related to gender, race, and socioeconomic status, thereby reinforcing harmful stereotypes and shaping public perception in unintended ways. While existing bias mitigation methods demonstrate effectiveness, they often encounter attribute entanglement, where adjustments to attributes relevant to the bias (i.e., target attributes) unintentionally alter attributes unassociated with the bias (i.e., non-target attributes), causing undesirable distribution shifts. To address this challenge, we introduce Entanglement-Free Attention (EFA), a method that accurately incorporates target attributes (e.g., White, Black, Asian, and Indian) while preserving non-target attributes (e.g., background details) during bias mitigation. At inference time, EFA randomly samples a target attribute with equal probability and adjusts the cross-attention in selected layers to incorporate the sampled attribute, achieving a fair distribution of target attributes. Extensive experiments demonstrate that EFA outperforms existing methods in mitigating bias while preserving non-target attributes, thereby maintaining the output distribution and generation capability of the original model. 

**Abstract (ZH)**: 基于扩散文本到图像模型中纠缠属性的注意力解脱方法：缓解社会偏见的同时保持非目标属性 

---
# Automatic Multi-View X-Ray/CT Registration Using Bone Substructure Contours 

**Title (ZH)**: 使用骨亚结构轮廓的自动多视图X射线/CT配准 

**Authors**: Roman Flepp, Leon Nissen, Bastian Sigrist, Arend Nieuwland, Nicola Cavalcanti, Philipp Fürnstahl, Thomas Dreher, Lilian Calvet  

**Link**: [PDF](https://arxiv.org/pdf/2506.13292)  

**Abstract**: Purpose: Accurate intraoperative X-ray/CT registration is essential for surgical navigation in orthopedic procedures. However, existing methods struggle with consistently achieving sub-millimeter accuracy, robustness under broad initial pose estimates or need manual key-point annotations. This work aims to address these challenges by proposing a novel multi-view X-ray/CT registration method for intraoperative bone registration. Methods: The proposed registration method consists of a multi-view, contour-based iterative closest point (ICP) optimization. Unlike previous methods, which attempt to match bone contours across the entire silhouette in both imaging modalities, we focus on matching specific subcategories of contours corresponding to bone substructures. This leads to reduced ambiguity in the ICP matches, resulting in a more robust and accurate registration solution. This approach requires only two X-ray images and operates fully automatically. Additionally, we contribute a dataset of 5 cadaveric specimens, including real X-ray images, X-ray image poses and the corresponding CT scans. Results: The proposed registration method is evaluated on real X-ray images using mean reprojection error (mRPD). The method consistently achieves sub-millimeter accuracy with a mRPD 0.67mm compared to 5.35mm by a commercial solution requiring manual intervention. Furthermore, the method offers improved practical applicability, being fully automatic. Conclusion: Our method offers a practical, accurate, and efficient solution for multi-view X-ray/CT registration in orthopedic surgeries, which can be easily combined with tracking systems. By improving registration accuracy and minimizing manual intervention, it enhances intraoperative navigation, contributing to more accurate and effective surgical outcomes in computer-assisted surgery (CAS). 

**Abstract (ZH)**: 目的：准确的术中X射线/CT配准对于骨科手术导航至关重要。然而，现有方法在一致实现亚毫米级精度、在广泛初始姿态估计下的鲁棒性或需要手动关键点标注方面存在挑战。本项工作通过提出一种新型的多视角X射线/CT配准方法来解决这些挑战，以实现术中骨骼配准。方法：所提出的方法包括一个多视角的基于轮廓的迭代最近点（ICP）优化。与之前的方法不同，这些方法试图在两种成像模态中将骨骼轮廓整体匹配到整个轮廓 silhouette 上，我们专注于匹配对应于骨骼亚结构的特定轮廓子类。这减少了ICP匹配的歧义性，从而得到更稳健和准确的配准解决方案。该方法仅需要两张X射线图像，并且完全自动运行。此外，我们还贡献了一个包含5具尸体标本的数据库，其中包括真实的X射线图像、X射线图像姿态和对应的CT扫描。结果：所提出的方法在真实X射线图像上使用均方重构误差（mRPD）进行了评估。与需要手动干预的商用解决方案相比，该方法实现了亚毫米级精度，其mRPD为0.67毫米，而商用解决方案的mRPD为5.35毫米。此外，该方法提供了更好的实际适用性，完全自动运行。结论：本方法提供了一种实用、准确且高效的多视角X射线/CT配准解决方案，适用于骨科手术，并可以 easily 与跟踪系统结合使用。通过提高配准精度并减少手动干预，它增强了术中导航，促进了计算机辅助手术（CAS）中更准确和有效的手术结果。 

---
# AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy 

**Title (ZH)**: AceReason-Nemotron 1.1: 通过SFT和RL协同促进数学和代码推理的进步 

**Authors**: Zihan Liu, Zhuolin Yang, Yang Chen, Chankyu Lee, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2506.13284)  

**Abstract**: In this work, we investigate the synergy between supervised fine-tuning (SFT) and reinforcement learning (RL) in developing strong reasoning models. We begin by curating the SFT training data through two scaling strategies: increasing the number of collected prompts and the number of generated responses per prompt. Both approaches yield notable improvements in reasoning performance, with scaling the number of prompts resulting in more substantial gains. We then explore the following questions regarding the synergy between SFT and RL: (i) Does a stronger SFT model consistently lead to better final performance after large-scale RL training? (ii) How can we determine an appropriate sampling temperature during RL training to effectively balance exploration and exploitation for a given SFT initialization? Our findings suggest that (i) holds true, provided effective RL training is conducted, particularly when the sampling temperature is carefully chosen to maintain the temperature-adjusted entropy around 0.3, a setting that strikes a good balance between exploration and exploitation. Notably, the performance gap between initial SFT models narrows significantly throughout the RL process. Leveraging a strong SFT foundation and insights into the synergistic interplay between SFT and RL, our AceReason-Nemotron-1.1 7B model significantly outperforms AceReason-Nemotron-1.0 and achieves new state-of-the-art performance among Qwen2.5-7B-based reasoning models on challenging math and code benchmarks, thereby demonstrating the effectiveness of our post-training recipe. We release the model and data at: this https URL 

**Abstract (ZH)**: 在这种工作中，我们探讨了监督微调（SFT）与强化学习（RL）在开发强大推理模型方面的协同作用。我们通过两种缩放策略来精炼SFT训练数据：增加收集的提示数量和每个提示生成的响应数量。这两种方法都显著提升了推理性能，其中增加提示数量的方法带来了更大的提升。随后，我们探讨了SFT与RL之间协同作用的以下问题：（i）是否更强的SFT模型在大规模RL训练后始终能取得更好的最终性能？（ii）在RL训练过程中，如何确定合适的采样温度以有效地平衡给定SFT初始化条件下的探索与利用？我们的研究结果表明，（i）在进行有效的RL训练时成立，特别是在选择采样温度以保持温度调整后的熵约为0.3的情况下，这种设置可以在探索与利用之间取得良好的平衡。值得注意的是，RL过程中初始SFT模型之间的性能差距显著缩小。利用强大的SFT基础和对SFT与RL之间协同作用的深入了解，我们的AceReason-Nemotron-1.1 7B模型显著优于AceReason-Nemotron-1.0，并在基于Qwen2.5-7B的推理模型中以具有挑战性的数学和代码基准测试实现了新的最佳性能，从而证明了我们后续训练方案的有效性。我们将模型和数据发布在：this https URL。 

---
# SeqPE: Transformer with Sequential Position Encoding 

**Title (ZH)**: SeqPE: 带有序列位置编码的变压器 

**Authors**: Huyang Li, Yahui Liu, Hongyu Sun, Deng Cai, Leyang Cui, Wei Bi, Peilin Zhao, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.13277)  

**Abstract**: Since self-attention layers in Transformers are permutation invariant by design, positional encodings must be explicitly incorporated to enable spatial understanding. However, fixed-size lookup tables used in traditional learnable position embeddings (PEs) limit extrapolation capabilities beyond pre-trained sequence lengths. Expert-designed methods such as ALiBi and RoPE, mitigate this limitation but demand extensive modifications for adapting to new modalities, underscoring fundamental challenges in adaptability and scalability. In this work, we present SeqPE, a unified and fully learnable position encoding framework that represents each $n$-dimensional position index as a symbolic sequence and employs a lightweight sequential position encoder to learn their embeddings in an end-to-end manner. To regularize SeqPE's embedding space, we introduce two complementary objectives: a contrastive objective that aligns embedding distances with a predefined position-distance function, and a knowledge distillation loss that anchors out-of-distribution position embeddings to in-distribution teacher representations, further enhancing extrapolation performance. Experiments across language modeling, long-context question answering, and 2D image classification demonstrate that SeqPE not only surpasses strong baselines in perplexity, exact match (EM), and accuracy--particularly under context length extrapolation--but also enables seamless generalization to multi-dimensional inputs without requiring manual architectural redesign. We release our code, data, and checkpoints at this https URL. 

**Abstract (ZH)**: 自注意力层在Transformer中设计上具有置换不变性，因此需要显式地引入位置编码以实现空间理解。然而，传统可学习位置编码（PE）中固定大小的查找表限制了其超出预训练序列长度的外推能力。专家设计的方法如ALiBi和RoPE减轻了这一限制，但需要对新模态进行大量的修改，突显了适应性和可扩展性的基本挑战。在本文中，我们提出SeqPE，这是一个统一的全可学习位置编码框架，将每个n维位置索引表示为符号序列，并采用一个轻量级的顺序位置编码器以端到端的方式学习其嵌入。为了规整SeqPE的嵌入空间，我们引入了两个互补的目标：对比目标，使嵌入距离与预定义的位置-距离函数对齐；以及知识蒸馏损失，将分布外的位置嵌入锚定到分布内的教师表示，进一步增强外推性能。实验表明，SeqPE不仅在困惑度、精确匹配和准确率等方面超过了强大的基线，特别是在上下文长度外推方面，而且还能够在不需要手动重新设计架构的情况下无缝泛化到多维输入。我们已在以下网址发布了我们的代码、数据和检查点：[此处链接]。 

---
# Energy-Efficient Digital Design: A Comparative Study of Event-Driven and Clock-Driven Spiking Neurons 

**Title (ZH)**: 能效数字设计：事件驱动与时钟驱动脉冲神经元的比较研究 

**Authors**: Filippo Marostica, Alessio Carpegna, Alessandro Savino, Stefano Di Carlo  

**Link**: [PDF](https://arxiv.org/pdf/2506.13268)  

**Abstract**: This paper presents a comprehensive evaluation of Spiking Neural Network (SNN) neuron models for hardware acceleration by comparing event driven and clock-driven implementations. We begin our investigation in software, rapidly prototyping and testing various SNN models based on different variants of the Leaky Integrate and Fire (LIF) neuron across multiple datasets. This phase enables controlled performance assessment and informs design refinement. Our subsequent hardware phase, implemented on FPGA, validates the simulation findings and offers practical insights into design trade offs. In particular, we examine how variations in input stimuli influence key performance metrics such as latency, power consumption, energy efficiency, and resource utilization. These results yield valuable guidelines for constructing energy efficient, real time neuromorphic systems. Overall, our work bridges software simulation and hardware realization, advancing the development of next generation SNN accelerators. 

**Abstract (ZH)**: 本文通过比较事件驱动和时钟驱动实现，对跳变神经网络（SNN）神经元模型进行全面评估。我们在软件阶段快速原型设计并测试了基于不同Leaky Integrate and Fire (LIF) 神经元变体的各种SNN模型，并在多个数据集上进行测试。这一阶段允许我们进行受控性能评估，从而指导设计改进。随后的硬件阶段，在FPGA上实现，验证了仿真结果，并提供了关于设计权衡的实际见解。特别是，我们探讨了输入刺激的变化如何影响关键性能指标，如延迟、功耗、能源效率和资源利用率。这些结果为进一步构建节能的实时类脑系统提供了宝贵指导。总体而言，我们的工作实现了软件仿真和硬件实现之间的桥梁，推动了下一代SNN加速器的发展。 

---
# Distinct Computations Emerge From Compositional Curricula in In-Context Learning 

**Title (ZH)**: 独特的计算能力在上下文学习中的组成课程中 Emerge 

**Authors**: Jin Hwa Lee, Andrew K. Lampinen, Aaditya K. Singh, Andrew M. Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2506.13253)  

**Abstract**: In-context learning (ICL) research often considers learning a function in-context through a uniform sample of input-output pairs. Here, we investigate how presenting a compositional subtask curriculum in context may alter the computations a transformer learns. We design a compositional algorithmic task based on the modular exponential-a double exponential task composed of two single exponential subtasks and train transformer models to learn the task in-context. We compare (a) models trained using an in-context curriculum consisting of single exponential subtasks and, (b) models trained directly on the double exponential task without such a curriculum. We show that models trained with a subtask curriculum can perform zero-shot inference on unseen compositional tasks and are more robust given the same context length. We study how the task and subtasks are represented across the two training regimes. We find that the models employ diverse strategies modulated by the specific curriculum design. 

**Abstract (ZH)**: 基于上下文学习（ICL）研究通常通过均匀的输入-输出样本对来学习一个函数。在此，我们探讨如何在上下文中呈现组合性子任务课程可能如何改变transformer学习的计算方式。我们基于模块化指数任务设计了一个组合性算法任务，该任务由两个单一指数子任务组成，并训练transformer模型在上下文中学习该任务。我们比较了两种情况：(a) 使用仅包含单一指数子任务的上下文课程训练的模型，以及(b) 直接在双指数任务上训练而没有此类课程的模型。我们展示，使用子任务课程训练的模型可以在未见的组合性任务上进行零样本推理，并且在相同的上下文长度下更为 robust。我们研究了两种训练方案下任务和子任务的表示方式。我们发现，模型采用了多种策略，这些策略受到特定课程设计的调节。 

---
# On Immutable Memory Systems for Artificial Agents: A Blockchain-Indexed Automata-Theoretic Framework Using ECDH-Keyed Merkle Chains 

**Title (ZH)**: 面向人工代理的不可变内存系统：基于ECDH键控Merkle链的区块链索引自动机理论框架 

**Authors**: Craig Steven Wright  

**Link**: [PDF](https://arxiv.org/pdf/2506.13246)  

**Abstract**: This paper presents a formalised architecture for synthetic agents designed to retain immutable memory, verifiable reasoning, and constrained epistemic growth. Traditional AI systems rely on mutable, opaque statistical models prone to epistemic drift and historical revisionism. In contrast, we introduce the concept of the Merkle Automaton, a cryptographically anchored, deterministic computational framework that integrates formal automata theory with blockchain-based commitments. Each agent transition, memory fragment, and reasoning step is committed within a Merkle structure rooted on-chain, rendering it non-repudiable and auditably permanent. To ensure selective access and confidentiality, we derive symmetric encryption keys from ECDH exchanges contextualised by hierarchical privilege lattices. This enforces cryptographic access control over append-only DAG-structured knowledge graphs. Reasoning is constrained by formal logic systems and verified through deterministic traversal of policy-encoded structures. Updates are non-destructive and historied, preserving epistemic lineage without catastrophic forgetting. Zero-knowledge proofs facilitate verifiable, privacy-preserving inclusion attestations. Collectively, this architecture reframes memory not as a cache but as a ledger - one whose contents are enforced by protocol, bound by cryptography, and constrained by formal logic. The result is not an intelligent agent that mimics thought, but an epistemic entity whose outputs are provably derived, temporally anchored, and impervious to post hoc revision. This design lays foundational groundwork for legal, economic, and high-assurance computational systems that require provable memory, unforgeable provenance, and structural truth. 

**Abstract (ZH)**: 一种保留不变记忆、可验证推理和受限的知识增长的正式化合成代理架构 

---
# No-Regret Learning Under Adversarial Resource Constraints: A Spending Plan Is All You Need! 

**Title (ZH)**: 在对抗资源约束下的无遗憾学习：只需一个支出计划即可！ 

**Authors**: Francesco Emanuele Stradi, Matteo Castiglioni, Alberto Marchesi, Nicola Gatti, Christian Kroer  

**Link**: [PDF](https://arxiv.org/pdf/2506.13244)  

**Abstract**: We study online decision making problems under resource constraints, where both reward and cost functions are drawn from distributions that may change adversarially over time. We focus on two canonical settings: $(i)$ online resource allocation where rewards and costs are observed before action selection, and $(ii)$ online learning with resource constraints where they are observed after action selection, under full feedback or bandit feedback. It is well known that achieving sublinear regret in these settings is impossible when reward and cost distributions may change arbitrarily over time. To address this challenge, we analyze a framework in which the learner is guided by a spending plan--a sequence prescribing expected resource usage across rounds. We design general (primal-)dual methods that achieve sublinear regret with respect to baselines that follow the spending plan. Crucially, the performance of our algorithms improves when the spending plan ensures a well-balanced distribution of the budget across rounds. We additionally provide a robust variant of our methods to handle worst-case scenarios where the spending plan is highly imbalanced. To conclude, we study the regret of our algorithms when competing against benchmarks that deviate from the prescribed spending plan. 

**Abstract (ZH)**: 我们在资源约束下的在线决策问题中研究，在这种情况下，奖励和成本函数来源于可能随时间敌对地变化的分布。我们关注两类典型的设置：（i）在线资源分配，其中奖励和成本在选择行动之前被观测到，以及（ii）资源约束下的在线学习，其中它们在选择行动之后通过完全反馈或Bandit反馈被观测到。众所周知，在奖励和成本分布可能任意变化的情况下，在这些设置中实现亚线性遗憾是不可能的。为了解决这一挑战，我们分析了一个框架，其中学习者受到一个支出计划的引导——一个规定各轮平均资源使用量的序列。我们设计了一般的（原对偶）方法，这些方法相对于遵循支出计划的基线实现了亚线性遗憾。关键的是，当支出计划确保预算在各轮之间的分配均衡时，我们的算法性能会更好。此外，我们还提供了一种鲁棒变体的方法来处理最坏情况场景，其中支出计划极度不平衡。最后，我们在与偏离规定支出计划的基准竞争时研究了我们算法的遗憾。 

---
# Screen Hijack: Visual Poisoning of VLM Agents in Mobile Environments 

**Title (ZH)**: 屏幕操控：移动环境中VLM代理的视觉污染 

**Authors**: Xuan Wang, Siyuan Liang, Zhe Liu, Yi Yu, Yuliang Lu, Xiaochun Cao, Ee-Chien Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13205)  

**Abstract**: With the growing integration of vision-language models (VLMs), mobile agents are now widely used for tasks like UI automation and camera-based user assistance. These agents are often fine-tuned on limited user-generated datasets, leaving them vulnerable to covert threats during the training process. In this work we present GHOST, the first clean-label backdoor attack specifically designed for mobile agents built upon VLMs. Our method manipulates only the visual inputs of a portion of the training samples - without altering their corresponding labels or instructions - thereby injecting malicious behaviors into the model. Once fine-tuned with this tampered data, the agent will exhibit attacker-controlled responses when a specific visual trigger is introduced at inference time. The core of our approach lies in aligning the gradients of poisoned samples with those of a chosen target instance, embedding backdoor-relevant features into the poisoned training data. To maintain stealth and enhance robustness, we develop three realistic visual triggers: static visual patches, dynamic motion cues, and subtle low-opacity overlays. We evaluate our method across six real-world Android apps and three VLM architectures adapted for mobile use. Results show that our attack achieves high attack success rates (up to 94.67 percent) while maintaining high clean-task performance (FSR up to 95.85 percent). Additionally, ablation studies shed light on how various design choices affect the efficacy and concealment of the attack. Overall, this work is the first to expose critical security flaws in VLM-based mobile agents, highlighting their susceptibility to clean-label backdoor attacks and the urgent need for effective defense mechanisms in their training pipelines. Code and examples are available at: this https URL. 

**Abstract (ZH)**: 基于视觉语言模型的移动代理的首个清洁标签后门攻击：GHOST 

---
# CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction 

**Title (ZH)**: CertDW: 向量化分信心预测的的数据集所有权验证 

**Authors**: Ting Qiao, Yiming Li, Jianbin Li, Yingjia Wang, Leyi Qi, Junfeng Guo, Ruili Feng, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13160)  

**Abstract**: Deep neural networks (DNNs) rely heavily on high-quality open-source datasets (e.g., ImageNet) for their success, making dataset ownership verification (DOV) crucial for protecting public dataset copyrights. In this paper, we find existing DOV methods (implicitly) assume that the verification process is faithful, where the suspicious model will directly verify ownership by using the verification samples as input and returning their results. However, this assumption may not necessarily hold in practice and their performance may degrade sharply when subjected to intentional or unintentional perturbations. To address this limitation, we propose the first certified dataset watermark (i.e., CertDW) and CertDW-based certified dataset ownership verification method that ensures reliable verification even under malicious attacks, under certain conditions (e.g., constrained pixel-level perturbation). Specifically, inspired by conformal prediction, we introduce two statistical measures, including principal probability (PP) and watermark robustness (WR), to assess model prediction stability on benign and watermarked samples under noise perturbations. We prove there exists a provable lower bound between PP and WR, enabling ownership verification when a suspicious model's WR value significantly exceeds the PP values of multiple benign models trained on watermark-free datasets. If the number of PP values smaller than WR exceeds a threshold, the suspicious model is regarded as having been trained on the protected dataset. Extensive experiments on benchmark datasets verify the effectiveness of our CertDW method and its resistance to potential adaptive attacks. Our codes are at \href{this https URL}{GitHub}. 

**Abstract (ZH)**: 深度神经网络（DNNs）的成功高度依赖于高质量的开源数据集（例如ImageNet），因此数据集所有权验证（DOV）对于保护公共数据集版权至关重要。在本文中，我们发现现有的DOV方法（隐含地）假设验证过程是忠实的，可疑模型可以直接通过使用验证样本作为输入并返回其结果来验证所有权。然而，在实际操作中，这一假设不一定成立，其性能在遭受故意或无意的扰动时会急剧下降。为了应对这一局限性，我们提出了第一个经过验证的数据集水印（即CertDW）和基于CertDW的数据集所有权验证方法，该方法在某些条件下（例如受限的像素级扰动）能够确保即使在恶意攻击下也能可靠地进行验证。具体来说，受到容间预测的启发，我们引入了两个统计指标，包括主概率（PP）和水印稳健性（WR），以评估在噪声扰动下模型在良性样本和带有水印的样本上的预测稳定性。我们证明了PP和WR之间存在可证明的下界，当可疑模型的WR值显著超过多个无水印数据集训练的良性模型的PP值时，可以进行所有权验证。如果WR值小于PP值的数量超过阈值时，可疑模型被视为已训练于受保护的数据集上。广泛的基准数据集实验验证了我们CertDW方法的有效性和对潜在适应性攻击的抗性。我们的代码可在GitHub上获取。 

---
# Quantum AGI: Ontological Foundations 

**Title (ZH)**: 量子AGI：本体论基础 

**Authors**: Elija Perrier, Michael Timothy Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2506.13134)  

**Abstract**: We examine the implications of quantum foundations for AGI, focusing on how seminal results such as Bell's theorems (non-locality), the Kochen-Specker theorem (contextuality) and no-cloning theorem problematise practical implementation of AGI in quantum settings. We introduce a novel information-theoretic taxonomy distinguishing between classical AGI and quantum AGI and show how quantum mechanics affects fundamental features of agency. We show how quantum ontology may change AGI capabilities, both via affording computational advantages and via imposing novel constraints. 

**Abstract (ZH)**: 我们探讨量子基础对AGI的影响，重点关注贝尔定理（非局域性）、科欣-斯佩克定理（上下文依赖性）和不可克隆定理如何使在量子环境中实际实施AGI面临挑战。我们引入了一种新的信息理论分类，区分经典AGI和量子AGI，并展示了量子力学如何影响代理的基本特征。我们表明，量子本体论可能通过提供计算优势和施加新型约束来改变AGI的能力。 

---
# PhenoKG: Knowledge Graph-Driven Gene Discovery and Patient Insights from Phenotypes Alone 

**Title (ZH)**: PhenoKG：基于知识图谱的表型驱动基因发现及患者洞察 

**Authors**: Kamilia Zaripova, Ege Özsoy, Nassir Navab, Azade Farshad  

**Link**: [PDF](https://arxiv.org/pdf/2506.13119)  

**Abstract**: Identifying causative genes from patient phenotypes remains a significant challenge in precision medicine, with important implications for the diagnosis and treatment of genetic disorders. We propose a novel graph-based approach for predicting causative genes from patient phenotypes, with or without an available list of candidate genes, by integrating a rare disease knowledge graph (KG). Our model, combining graph neural networks and transformers, achieves substantial improvements over the current state-of-the-art. On the real-world MyGene2 dataset, it attains a mean reciprocal rank (MRR) of 24.64\% and nDCG@100 of 33.64\%, surpassing the best baseline (SHEPHERD) at 19.02\% MRR and 30.54\% nDCG@100. We perform extensive ablation studies to validate the contribution of each model component. Notably, the approach generalizes to cases where only phenotypic data are available, addressing key challenges in clinical decision support when genomic information is incomplete. 

**Abstract (ZH)**: 从患者表型识别致病基因仍然是精准医学中的一个重大挑战，对于遗传性疾病诊断和治疗具有重要意义。我们提出了一种基于图的预测方法，用于从患者表型中预测致病基因，该方法可以有或没有候选基因列表，并通过整合罕见疾病知识图谱（KG）来实现。我们的模型结合了图神经网络和transformer，实现了对当前最先进的方法的显著改进。在实际数据集MyGene2上，我们的模型取得平均倒数排名（MRR）为24.64%和nDCG@100为33.64%，超越了最佳基线（SHEPHERD）的19.02% MRR和30.54% nDCG@100。我们进行了广泛的消融研究以验证每个模型组件的贡献。值得注意的是，该方法能够应用于仅存在表型数据的情况，从而解决了基因组信息不完整时临床决策支持的关键挑战。 

---
# Dynamic Graph Condensation 

**Title (ZH)**: 动态图凝聚 

**Authors**: Dong Chen, Shuai Zheng, Yeyu Yan, Muhao Xu, Zhenfeng Zhu, Yao Zhao, Kunlun He  

**Link**: [PDF](https://arxiv.org/pdf/2506.13099)  

**Abstract**: Recent research on deep graph learning has shifted from static to dynamic graphs, motivated by the evolving behaviors observed in complex real-world systems. However, the temporal extension in dynamic graphs poses significant data efficiency challenges, including increased data volume, high spatiotemporal redundancy, and reliance on costly dynamic graph neural networks (DGNNs). To alleviate the concerns, we pioneer the study of dynamic graph condensation (DGC), which aims to substantially reduce the scale of dynamic graphs for data-efficient DGNN training. Accordingly, we propose DyGC, a novel framework that condenses the real dynamic graph into a compact version while faithfully preserving the inherent spatiotemporal characteristics. Specifically, to endow synthetic graphs with realistic evolving structures, a novel spiking structure generation mechanism is introduced. It draws on the dynamic behavior of spiking neurons to model temporally-aware connectivity in dynamic graphs. Given the tightly coupled spatiotemporal dependencies, DyGC proposes a tailored distribution matching approach that first constructs a semantically rich state evolving field for dynamic graphs, and then performs fine-grained spatiotemporal state alignment to guide the optimization of the condensed graph. Experiments across multiple dynamic graph datasets and representative DGNN architectures demonstrate the effectiveness of DyGC. Notably, our method retains up to 96.2% DGNN performance with only 0.5% of the original graph size, and achieves up to 1846 times training speedup. 

**Abstract (ZH)**: 基于动态图凝缩的深度图学习研究 

---
# Beyond the First Read: AI-Assisted Perceptual Error Detection in Chest Radiography Accounting for Interobserver Variability 

**Title (ZH)**: 超越初次阅读：考虑观察者间变异性的胸部X光辅助感知错误检测 

**Authors**: Adhrith Vutukuri, Akash Awasthi, David Yang, Carol C. Wu, Hien Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13049)  

**Abstract**: Chest radiography is widely used in diagnostic imaging. However, perceptual errors -- especially overlooked but visible abnormalities -- remain common and clinically significant. Current workflows and AI systems provide limited support for detecting such errors after interpretation and often lack meaningful human--AI collaboration. We introduce RADAR (Radiologist--AI Diagnostic Assistance and Review), a post-interpretation companion system. RADAR ingests finalized radiologist annotations and CXR images, then performs regional-level analysis to detect and refer potentially missed abnormal regions. The system supports a "second-look" workflow and offers suggested regions of interest (ROIs) rather than fixed labels to accommodate inter-observer variation. We evaluated RADAR on a simulated perceptual-error dataset derived from de-identified CXR cases, using F1 score and Intersection over Union (IoU) as primary metrics. RADAR achieved a recall of 0.78, precision of 0.44, and an F1 score of 0.56 in detecting missed abnormalities in the simulated perceptual-error dataset. Although precision is moderate, this reduces over-reliance on AI by encouraging radiologist oversight in human--AI collaboration. The median IoU was 0.78, with more than 90% of referrals exceeding 0.5 IoU, indicating accurate regional localization. RADAR effectively complements radiologist judgment, providing valuable post-read support for perceptual-error detection in CXR interpretation. Its flexible ROI suggestions and non-intrusive integration position it as a promising tool in real-world radiology workflows. To facilitate reproducibility and further evaluation, we release a fully open-source web implementation alongside a simulated error dataset. All code, data, demonstration videos, and the application are publicly available at this https URL. 

**Abstract (ZH)**: 胸部X光成像在诊断成像中广泛使用。然而，感知错误——尤其是被忽视但仍可见的异常——仍然常见且具有临床意义。当前的工作流程和AI系统在解释后支持检测这些错误的能力有限，常常缺乏有意义的人工智能合作。我们引入了RADAR（放射科医生—AI诊断辅助和审查）后解释伴侣系统。RADAR接受最终的放射科医生注释和胸部X光图像，然后进行区域级别的分析以检测和指示可能被忽视的异常区域。该系统支持“二次阅片”工作流程，并提供建议感兴趣的区域（ROI）而非固定标签，以适应观察者间的差异。我们在去标识化的胸部X光病例中模拟出了一个感知错误数据集，并使用F1分数和交并比（IoU）作为主要评价指标来评估RADAR。RADAR在模拟的感知错误数据集中检测被忽视异常的召回率为0.78，精确率为0.44，F1分为0.56。尽管精确率中等，但这一结果减少了对AI的过度依赖，鼓励放射科医生在人机协作中进行监督。中位数IoU为0.78，超过90%的建议区域超过0.5 IoU，表明区域定位准确。RADAR有效补充了放射科医生的判断，为胸部X光解释中的感知错误检测提供了有价值的后读支持。其灵活的ROI建议和非侵入性集成使其成为实际放射学工作流程中一种有前景的工具。为了促进可重复性和进一步评估，我们公开发布了一个完全开源的网络实现以及一个模拟错误数据集。所有代码、数据、演示视频和应用均可在以下网址访问：this https URL。 

---
# SpaceTrack-TimeSeries: Time Series Dataset towards Satellite Orbit Analysis 

**Title (ZH)**: SpaceTrack-TimeSeries：面向卫星轨道分析的时间序列数据集 

**Authors**: Zhixin Guo, Qi Shi, Xiaofan Xu, Sixiang Shan, Limin Qin, Linqiang Ge, Rui Zhang, Ya Dai, Hua Zhu, Guowei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13034)  

**Abstract**: With the rapid advancement of aerospace technology and the large-scale deployment of low Earth orbit (LEO) satellite constellations, the challenges facing astronomical observations and deep space exploration have become increasingly pronounced. As a result, the demand for high-precision orbital data on space objects-along with comprehensive analyses of satellite positioning, constellation configurations, and deep space satellite dynamics-has grown more urgent. However, there remains a notable lack of publicly accessible, real-world datasets to support research in areas such as space object maneuver behavior prediction and collision risk assessment. This study seeks to address this gap by collecting and curating a representative dataset of maneuvering behavior from Starlink satellites. The dataset integrates Two-Line Element (TLE) catalog data with corresponding high-precision ephemeris data, thereby enabling a more realistic and multidimensional modeling of space object behavior. It provides valuable insights into practical deployment of maneuver detection methods and the evaluation of collision risks in increasingly congested orbital environments. 

**Abstract (ZH)**: 随着航空航天技术的迅速进展和低地球轨道（LEO）卫星星座的大规模部署，天文学观测和深空探索面临的挑战日益凸显。因此，对空间物体高精度轨道数据的需求以及对卫星定位、星座配置和深空卫星动力学的全面分析变得更为迫切。然而，仍然缺乏支持空间物体机动行为预测和碰撞风险评估研究的公开真实世界数据集。本文旨在通过收集和整理代表性的星链卫星机动行为数据集来填补这一空白。该数据集将Two-Line Element (TLE)目录数据与相应的高精度星历数据相结合，从而实现对空间物体行为更为真实和多维度的建模，为机动检测方法的实际部署和日益拥挤的轨道环境中碰撞风险的评估提供了宝贵的见解。 

---
# AS400-DET: Detection using Deep Learning Model for IBM i (AS/400) 

**Title (ZH)**: AS400-DET: 使用深度学习模型进行IBM i (AS/400) 的检测 

**Authors**: Thanh Tran, Son T. Luu, Quan Bui, Shoshin Nomura  

**Link**: [PDF](https://arxiv.org/pdf/2506.13032)  

**Abstract**: This paper proposes a method for automatic GUI component detection for the IBM i system (formerly and still more commonly known as AS/400). We introduce a human-annotated dataset consisting of 1,050 system screen images, in which 381 images are screenshots of IBM i system screens in Japanese. Each image contains multiple components, including text labels, text boxes, options, tables, instructions, keyboards, and command lines. We then develop a detection system based on state-of-the-art deep learning models and evaluate different approaches using our dataset. The experimental results demonstrate the effectiveness of our dataset in constructing a system for component detection from GUI screens. By automatically detecting GUI components from the screen, AS400-DET has the potential to perform automated testing on systems that operate via GUI screens. 

**Abstract (ZH)**: 本文提出了一种针对IBM i系统（曾被称为AS/400）的自动GUI组件检测方法。我们引入了一个由1,050张系统屏幕图像组成的人工标注数据集，其中381张图像为日语界面的IBM i系统屏幕截图。每张图像包含多个组件，包括文本标签、文本框、选项、表格、说明、键盘和命令行。随后，我们基于最先进的深度学习模型开发了一种检测系统，并使用该数据集评估不同的方法。实验结果表明，我们的数据集在构建从GUI屏幕检测组件的系统方面具有有效性。通过自动检测屏幕上的GUI组件，AS400-DET有可能对通过GUI屏幕操作的系统进行自动化测试。 

---
# Edeflip: Supervised Word Translation between English and Yoruba 

**Title (ZH)**: Edeflip: 英语与约鲁巴语之间的监督词翻译 

**Authors**: Ikeoluwa Abioye, Jiani Ge  

**Link**: [PDF](https://arxiv.org/pdf/2506.13020)  

**Abstract**: In recent years, embedding alignment has become the state-of-the-art machine translation approach, as it can yield high-quality translation without training on parallel corpora. However, existing research and application of embedding alignment mostly focus on high-resource languages with high-quality monolingual embeddings. It is unclear if and how low-resource languages may be similarly benefited. In this study, we implement an established supervised embedding alignment method for word translation from English to Yoruba, the latter a low-resource language. We found that higher embedding quality and normalizing embeddings increase word translation precision, with, additionally, an interaction effect between the two. Our results demonstrate the limitations of the state-of-the-art supervised embedding alignment when it comes to low-resource languages, for which there are additional factors that need to be taken into consideration, such as the importance of curating high-quality monolingual embeddings. We hope our work will be a starting point for further machine translation research that takes into account the challenges that low-resource languages face. 

**Abstract (ZH)**: 近年来，嵌入对齐已成为最先进的机器翻译方法，因为它可以在无需使用平行语料库进行训练的情况下生成高质量的翻译。然而，现有的嵌入对齐研究和应用主要集中在高资源语言和高质量单语嵌入上。低资源语言是否以及如何从中受益尚不清楚。本研究实施了一种成熟的监督嵌入对齐方法，将英语翻译为约鲁巴语，后者是一种低资源语言。我们发现，更高的嵌入质量和归一化嵌入可以提高词语翻译的精度，并且两者之间存在交互效应。我们的结果展示了最先进的监督嵌入对齐方法在低资源语言上的局限性，对于这些语言，还需要考虑其他因素，例如高质量单语嵌入的重要性。我们希望我们的工作能够成为进一步考虑低资源语言挑战的机器翻译研究的起点。 

---
# Symmetry in Neural Network Parameter Spaces 

**Title (ZH)**: 神经网络参数空间中的对称性 

**Authors**: Bo Zhao, Robin Walters, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13018)  

**Abstract**: Modern deep learning models are highly overparameterized, resulting in large sets of parameter configurations that yield the same outputs. A significant portion of this redundancy is explained by symmetries in the parameter space--transformations that leave the network function unchanged. These symmetries shape the loss landscape and constrain learning dynamics, offering a new lens for understanding optimization, generalization, and model complexity that complements existing theory of deep learning. This survey provides an overview of parameter space symmetry. We summarize existing literature, uncover connections between symmetry and learning theory, and identify gaps and opportunities in this emerging field. 

**Abstract (ZH)**: 现代深度学习模型高度过参数化，导致产生大量生成相同输出的参数配置。这一冗余中的大部分可以用参数空间中的对称性来解释——那些使网络函数保持不变的变换。这些对称性塑造了损失景观，并限制了学习动力学，提供了理解优化、泛化和模型复杂性的新视角，补充了现有的深度学习理论。本文综述了参数空间对称性。我们总结了现有文献，揭示了对称性和学习理论之间的联系，并指出了这一新兴领域中的空白和机遇。 

---
# Geometric Embedding Alignment via Curvature Matching in Transfer Learning 

**Title (ZH)**: 曲率匹配下的几何嵌入对齐在迁移学习中的应用 

**Authors**: Sung Moon Ko, Jaewan Lee, Sumin Lee, Soorin Yim, Kyunghoon Bae, Sehui Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.13015)  

**Abstract**: Geometrical interpretations of deep learning models offer insightful perspectives into their underlying mathematical structures. In this work, we introduce a novel approach that leverages differential geometry, particularly concepts from Riemannian geometry, to integrate multiple models into a unified transfer learning framework. By aligning the Ricci curvature of latent space of individual models, we construct an interrelated architecture, namely Geometric Embedding Alignment via cuRvature matching in transfer learning (GEAR), which ensures comprehensive geometric representation across datapoints. This framework enables the effective aggregation of knowledge from diverse sources, thereby improving performance on target tasks. We evaluate our model on 23 molecular task pairs sourced from various domains and demonstrate significant performance gains over existing benchmark model under both random (14.4%) and scaffold (8.3%) data splits. 

**Abstract (ZH)**: 几何学视角下深度学习模型的解释为探索其 underlying 数学结构提供了有益的见解。在本工作中，我们提出了一种新颖的方法，利用微分几何，特别是黎曼几何的概念，将多个模型整合到一个统一的迁移学习框架中。通过对个体模型潜在空间的 Ricci 曲率进行对齐，我们构建了一种相互关联的架构，即 Geometric Embedding Alignment via cuRvature matching in transfer learning (GEAR)，该架构确保了数据点跨域的全面几何表示。该框架能够有效地聚合来自不同来源的知识，从而提高目标任务的性能。我们在来自不同领域的 23 个分子任务对上评估了我们的模型，并在随机 (14.4%) 和骨架 (8.3%) 数据分割下证明了相对于现有基准模型的显著性能提升。 

---
# Distributional Training Data Attribution 

**Title (ZH)**: 分布训练数据归属 

**Authors**: Bruno Mlodozeniec, Isaac Reid, Sam Power, David Krueger, Murat Erdogdu, Richard E. Turner, Roger Grosse  

**Link**: [PDF](https://arxiv.org/pdf/2506.12965)  

**Abstract**: Randomness is an unavoidable part of training deep learning models, yet something that traditional training data attribution algorithms fail to rigorously account for. They ignore the fact that, due to stochasticity in the initialisation and batching, training on the same dataset can yield different models. In this paper, we address this shortcoming through introducing distributional training data attribution (d-TDA), the goal of which is to predict how the distribution of model outputs (over training runs) depends upon the dataset. We demonstrate the practical significance of d-TDA in experiments, e.g. by identifying training examples that drastically change the distribution of some target measurement without necessarily changing the mean. Intriguingly, we also find that influence functions (IFs), a popular but poorly-understood data attribution tool, emerge naturally from our distributional framework as the limit to unrolled differentiation; without requiring restrictive convexity assumptions. This provides a new mathematical motivation for their efficacy in deep learning, and helps to characterise their limitations. 

**Abstract (ZH)**: 分布训练数据归因：预测模型输出分布随数据集变化的情况 

---
# eLog analysis for accelerators: status and future outlook 

**Title (ZH)**: 加速器的eLog分析：现状与未来展望 

**Authors**: Antonin Sulc, Thorsten Hellert, Aaron Reed, Adam Carpenter, Alex Bien, Chris Tennant, Claudio Bisegni, Daniel Lersch, Daniel Ratner, David Lawrence, Diana McSpadden, Hayden Hoschouer, Jason St. John, Thomas Britton  

**Link**: [PDF](https://arxiv.org/pdf/2506.12949)  

**Abstract**: This work demonstrates electronic logbook (eLog) systems leveraging modern AI-driven information retrieval capabilities at the accelerator facilities of Fermilab, Jefferson Lab, Lawrence Berkeley National Laboratory (LBNL), SLAC National Accelerator Laboratory. We evaluate contemporary tools and methodologies for information retrieval with Retrieval Augmented Generation (RAGs), focusing on operational insights and integration with existing accelerator control systems.
The study addresses challenges and proposes solutions for state-of-the-art eLog analysis through practical implementations, demonstrating applications and limitations. We present a framework for enhancing accelerator facility operations through improved information accessibility and knowledge management, which could potentially lead to more efficient operations. 

**Abstract (ZH)**: 本研究展示了在费米实验室、杰斐erson实验室、劳伦斯伯克利国家实验室（LBNL）和SLAC国家加速器实验室的加速器设施中利用现代AI驱动的信息检索功能的电子日志(eLog)系统。我们评估了基于检索增强生成（RAGs）的当前工具和方法在信息检索中的应用，重点关注操作洞察和与现有加速器控制系统集成的情况。

该研究解决了eLog分析的挑战并提出了解决方案，通过实际实施演示了应用和限制。我们提出了一个框架，通过提高信息访问性和知识管理来增强加速器设施的操作，这有可能导致更高效的运行。 

---
# Identifying and Investigating Global News Coverage of Critical Events Such as Disasters and Terrorist Attacks 

**Title (ZH)**: 识别并探究全球新闻对关键事件如灾难和恐怖袭击的报道 

**Authors**: Erica Cai, Xi Chen, Reagan Grey Keeney, Ethan Zuckerman, Brendan O'Connor, Przemyslaw A. Grabowicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12925)  

**Abstract**: Comparative studies of news coverage are challenging to conduct because methods to identify news articles about the same event in different languages require expertise that is difficult to scale. We introduce an AI-powered method for identifying news articles based on an event FINGERPRINT, which is a minimal set of metadata required to identify critical events. Our event coverage identification method, FINGERPRINT TO ARTICLE MATCHING FOR EVENTS (FAME), efficiently identifies news articles about critical world events, specifically terrorist attacks and several types of natural disasters. FAME does not require training data and is able to automatically and efficiently identify news articles that discuss an event given its fingerprint: time, location, and class (such as storm or flood). The method achieves state-of-the-art performance and scales to massive databases of tens of millions of news articles and hundreds of events happening globally. We use FAME to identify 27,441 articles that cover 470 natural disaster and terrorist attack events that happened in 2020. To this end, we use a massive database of news articles in three languages from MediaCloud, and three widely used, expert-curated databases of critical events: EM-DAT, USGS, and GTD. Our case study reveals patterns consistent with prior literature: coverage of disasters and terrorist attacks correlates to death counts, to the GDP of a country where the event occurs, and to trade volume between the reporting country and the country where the event occurred. We share our NLP annotations and cross-country media attention data to support the efforts of researchers and media monitoring organizations. 

**Abstract (ZH)**: 基于事件指纹的新闻文章匹配方法：识别关键世界事件的新闻coverage 

---
# Logit Dynamics in Softmax Policy Gradient Methods 

**Title (ZH)**: softmax策略梯度方法中的Logit动力学 

**Authors**: Yingru Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12912)  

**Abstract**: We analyzes the logit dynamics of softmax policy gradient methods. We derive the exact formula for the L2 norm of the logit update vector: $$ \|\Delta \mathbf{z}\|_2 \propto \sqrt{1-2P_c + C(P)} $$ This equation demonstrates that update magnitudes are determined by the chosen action's probability ($P_c$) and the policy's collision probability ($C(P)$), a measure of concentration inversely related to entropy. Our analysis reveals an inherent self-regulation mechanism where learning vigor is automatically modulated by policy confidence, providing a foundational insight into the stability and convergence of these methods. 

**Abstract (ZH)**: softmax策略梯度方法的logit动力学分析：L2范数的精确公式及其对更新幅度的影响 

---
# Exploring the Potential of Metacognitive Support Agents for Human-AI Co-Creation 

**Title (ZH)**: 探索元认知支持代理在人机共创中的潜力 

**Authors**: Frederic Gmeiner, Kaitao Luo, Ye Wang, Kenneth Holstein, Nikolas Martelaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.12879)  

**Abstract**: Despite the potential of generative AI (GenAI) design tools to enhance design processes, professionals often struggle to integrate AI into their workflows. Fundamental cognitive challenges include the need to specify all design criteria as distinct parameters upfront (intent formulation) and designers' reduced cognitive involvement in the design process due to cognitive offloading, which can lead to insufficient problem exploration, underspecification, and limited ability to evaluate outcomes. Motivated by these challenges, we envision novel metacognitive support agents that assist designers in working more reflectively with GenAI. To explore this vision, we conducted exploratory prototyping through a Wizard of Oz elicitation study with 20 mechanical designers probing multiple metacognitive support strategies. We found that agent-supported users created more feasible designs than non-supported users, with differing impacts between support strategies. Based on these findings, we discuss opportunities and tradeoffs of metacognitive support agents and considerations for future AI-based design tools. 

**Abstract (ZH)**: 尽管生成式人工智能（GenAI）设计工具有可能提升设计过程，专业人士往往难以将AI整合进其工作流程中。基本的认知挑战包括需要提前明确所有设计标准作为独立参数（意图形成），以及由于认知卸载导致设计师在设计过程中的认知参与度降低，这可能导致问题探索不足、标准不明确和对结果评估能力有限。鉴于这些挑战，我们设想了新型元认知支持代理，帮助设计师更反思性地使用GenAI。通过与20名机械设计师进行Wizard of Oz启发式原型设计研究，探索了多种元认知支持策略。我们发现，得到代理支持的用户比未得到支持的用户创造了更可行的设计，但不同支持策略的影响不同。基于这些发现，我们讨论了元认知支持代理的机会与权衡，并对未来基于AI的设计工具进行了考虑。 

---
# Privacy-Preserving Federated Learning against Malicious Clients Based on Verifiable Functional Encryption 

**Title (ZH)**: 基于可验证功能加密的抗恶意客户端隐私保护联邦学习 

**Authors**: Nina Cai, Jinguang Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.12846)  

**Abstract**: Federated learning is a promising distributed learning paradigm that enables collaborative model training without exposing local client data, thereby protect data privacy. However, it also brings new threats and challenges. The advancement of model inversion attacks has rendered the plaintext transmission of local models insecure, while the distributed nature of federated learning makes it particularly vulnerable to attacks raised by malicious clients. To protect data privacy and prevent malicious client attacks, this paper proposes a privacy-preserving federated learning framework based on verifiable functional encryption, without a non-colluding dual-server setup or additional trusted third-party. Specifically, we propose a novel decentralized verifiable functional encryption (DVFE) scheme that enables the verification of specific relationships over multi-dimensional ciphertexts. This scheme is formally treated, in terms of definition, security model and security proof. Furthermore, based on the proposed DVFE scheme, we design a privacy-preserving federated learning framework VFEFL that incorporates a novel robust aggregation rule to detect malicious clients, enabling the effective training of high-accuracy models under adversarial settings. Finally, we provide formal analysis and empirical evaluation of the proposed schemes. The results demonstrate that our approach achieves the desired privacy protection, robustness, verifiability and fidelity, while eliminating the reliance on non-colluding dual-server settings or trusted third parties required by existing methods. 

**Abstract (ZH)**: 联邦学习是一种有前景的分布式学习范式，能够在不暴露本地客户端数据的情况下进行协作模型训练，从而保护数据隐私。然而，它也带来了新的威胁和挑战。模型反转攻击的进步使得本地模型的明文传输变得不安全，而联邦学习的分布式特性使其特别容易受到恶意客户端发起的攻击。为保护数据隐私并防止恶意客户端攻击，本文提出了一种基于验证功能加密的隐私保护联邦学习框架，无需非串通双服务器设置或额外的可信第三方。具体来说，我们提出了一种新颖的去中心化验证功能加密（DVFE）方案，能够验证多维密文上的特定关系。该方案从定义、安全模型和安全证明方面进行正式处理。此外，基于提出的DVFE方案，我们设计了一个隐私保护联邦学习框架VFEFL，该框架包含一种新颖的鲁棒聚合规则以检测恶意客户端，能够在对抗环境中有效训练高精度模型。最后，我们提供了对所提方案的形式分析和实验评估。结果表明，我们的方法实现了所需的隐私保护、鲁棒性、可验证性和保真度，同时消除了现有方法依赖于非串通双服务器设置或可信第三方的需求。 

---
# Fair Bayesian Model-Based Clustering 

**Title (ZH)**: 公平的贝叶斯模型驱动聚类 

**Authors**: Jihu Lee, Kunwoong Kim, Yongdai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.12839)  

**Abstract**: Fair clustering has become a socially significant task with the advancement of machine learning technologies and the growing demand for trustworthy AI. Group fairness ensures that the proportions of each sensitive group are similar in all clusters. Most existing group-fair clustering methods are based on the $K$-means clustering and thus require the distance between instances and the number of clusters to be given in advance. To resolve this limitation, we propose a fair Bayesian model-based clustering called Fair Bayesian Clustering (FBC). We develop a specially designed prior which puts its mass only on fair clusters, and implement an efficient MCMC algorithm. Advantages of FBC are that it can infer the number of clusters and can be applied to any data type as long as the likelihood is defined (e.g., categorical data). Experiments on real-world datasets show that FBC (i) reasonably infers the number of clusters, (ii) achieves a competitive utility-fairness trade-off compared to existing fair clustering methods, and (iii) performs well on categorical data. 

**Abstract (ZH)**: 公平聚类已成为机器学习技术进步和社会对可信AI日益增长的需求背景下一项重要的社会任务。组公平确保了每个敏感群体在所有聚类中的比例相似。现有大多数基于$K$-均值聚类的组公平聚类方法均需提前给定实例间的距离和聚类的数量。为解决这一限制，我们提出了一种基于贝叶斯模型的公平聚类方法，称为公平贝叶斯聚类（Fair Bayesian Clustering, FBC）。我们开发了一种特别设计的先验，该先验仅将质量分配给公平聚类，并实现了一个高效MCMC算法。FBC的优势在于可以推断聚类的数量，并可以应用于只要似然性可以定义的任何数据类型（例如，分类数据）。实证研究结果表明，FBC能够合理地推断聚类的数量，实现与现有公平聚类方法具有竞争力的效用-公平性权衡，并且在分类数据上表现良好。 

---
# Synesthesia of Machines (SoM)-Enhanced Sub-THz ISAC Transmission for Air-Ground Network 

**Title (ZH)**: 机器联觉(SoM)增强的亚太赫兹ISAC空中地面网络传输 

**Authors**: Zonghui Yang, Shijian Gao, Xiang Cheng, Liuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12831)  

**Abstract**: Integrated sensing and communication (ISAC) within sub-THz frequencies is crucial for future air-ground networks, but unique propagation characteristics and hardware limitations present challenges in optimizing ISAC performance while increasing operational latency. This paper introduces a multi-modal sensing fusion framework inspired by synesthesia of machine (SoM) to enhance sub-THz ISAC transmission. By exploiting inherent degrees of freedom in sub-THz hardware and channels, the framework optimizes the radio-frequency environment. Squint-aware beam management is developed to improve air-ground network adaptability, enabling three-dimensional dynamic ISAC links. Leveraging multi-modal information, the framework enhances ISAC performance and reduces latency. Visual data rapidly localizes users and targets, while a customized multi-modal learning algorithm optimizes the hybrid precoder. A new metric provides comprehensive performance evaluation, and extensive experiments demonstrate that the proposed scheme significantly improves ISAC efficiency. 

**Abstract (ZH)**: 亚太赫兹频段集成传感与通信（ISAC）在未來空地网络中的集成至关重要，但独特的传播特性和硬件限制给优化ISAC性能并增加操作延迟带来了挑战。本文提出了一种受机器同感（SoM）启发的多模传感融合框架，以增强亚太赫兹频段ISAC传输。通过利用亚太赫兹硬件和信道固有的自由度，该框架优化了无线频谱环境。发展了射击角感知波束管理以提高空地网络的适应性，实现三维动态ISAC链路。利用多模信息，该框架提升ISAC性能并减少延迟。视觉数据快速定位用户和目标，自定义的多模学习算法优化混合预编码器。一个新的评估指标提供了全面的性能评估， extensive实验表明所提方案显著提升了ISAC效率。 

---
# Taking the GP Out of the Loop 

**Title (ZH)**: 去除GP循环 

**Authors**: David Sweet, Siddhant anand Jadhav  

**Link**: [PDF](https://arxiv.org/pdf/2506.12818)  

**Abstract**: Bayesian optimization (BO) has traditionally solved black box problems where evaluation is expensive and, therefore, design-evaluation pairs (i.e., observations) are few. Recently, there has been growing interest in applying BO to problems where evaluation is cheaper and, thus, observations are more plentiful. An impediment to scaling BO to many observations, $N$, is the $O(N^3)$ scaling of a na{ï}ve query of the Gaussian process (GP) surrogate. Modern implementations reduce this to $O(N^2)$, but the GP remains a bottleneck. We propose Epistemic Nearest Neighbors (ENN), a surrogate that estimates function values and epistemic uncertainty from $K$ nearest-neighbor observations. ENN has $O(N)$ query time and omits hyperparameter fitting, leaving uncertainty uncalibrated. To accommodate the lack of calibration, we employ an acquisition method based on Pareto-optimal tradeoffs between predicted value and uncertainty. Our proposed method, TuRBO-ENN, replaces the GP surrogate in TuRBO with ENN and its Thompson sampling acquisition method with our Pareto-based alternative. We demonstrate numerically that TuRBO-ENN can reduce the time to generate proposals by one to two orders of magnitude compared to TuRBO and scales to thousands of observations. 

**Abstract (ZH)**: 基于知识的最近邻（ENN）在黑箱优化中的应用：减少时间并扩展观测数量 

---
# Flow-Based Policy for Online Reinforcement Learning 

**Title (ZH)**: 基于流的策略在在线强化学习中的应用 

**Authors**: Lei Lv, Yunfei Li, Yu Luo, Fuchun Sun, Tao Kong, Jiafeng Xu, Xiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.12811)  

**Abstract**: We present \textbf{FlowRL}, a novel framework for online reinforcement learning that integrates flow-based policy representation with Wasserstein-2-regularized optimization. We argue that in addition to training signals, enhancing the expressiveness of the policy class is crucial for the performance gains in RL. Flow-based generative models offer such potential, excelling at capturing complex, multimodal action distributions. However, their direct application in online RL is challenging due to a fundamental objective mismatch: standard flow training optimizes for static data imitation, while RL requires value-based policy optimization through a dynamic buffer, leading to difficult optimization landscapes. FlowRL first models policies via a state-dependent velocity field, generating actions through deterministic ODE integration from noise. We derive a constrained policy search objective that jointly maximizes Q through the flow policy while bounding the Wasserstein-2 distance to a behavior-optimal policy implicitly derived from the replay buffer. This formulation effectively aligns the flow optimization with the RL objective, enabling efficient and value-aware policy learning despite the complexity of the policy class. Empirical evaluations on DMControl and Humanoidbench demonstrate that FlowRL achieves competitive performance in online reinforcement learning benchmarks. 

**Abstract (ZH)**: FlowRL：一种基于流的政策表示与Wasserstein-2正则化优化集成的新型在线强化学习框架 

---
# Resilient-native and Intelligent NextG Systems 

**Title (ZH)**: 本源鲁棒性和智能化的NextG系统 

**Authors**: Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2506.12795)  

**Abstract**: Just like power, water and transportation systems, wireless networks are a crucial societal infrastructure. As natural and human-induced disruptions continue to grow, wireless networks must be resilient to unforeseen events, able to withstand and recover from unexpected adverse conditions, shocks, unmodeled disturbances and cascading failures. Despite its critical importance, resilience remains an elusive concept, with its mathematical foundations still underdeveloped. Unlike robustness and reliability, resilience is premised on the fact that disruptions will inevitably happen. Resilience, in terms of elasticity, focuses on the ability to bounce back to favorable states, while resilience as plasticity involves agents (or networks) that can flexibly expand their states, hypotheses and course of actions, by transforming through real-time adaptation and reconfiguration. This constant situational awareness and vigilance of adapting world models and counterfactually reasoning about potential system failures and the corresponding best responses, is a core aspect of resilience. This article seeks to first define resilience and disambiguate it from reliability and robustness, before delving into the mathematics of resilience. Finally, the article concludes by presenting nuanced metrics and discussing trade-offs tailored to the unique characteristics of network resilience. 

**Abstract (ZH)**: 如同电力、水资源和交通系统一样，无线网络是关键的社会基础设施。随着自然和人为干扰的持续增长，无线网络必须具备应对突发事件的能力，能够承受和恢复意外的不利条件、冲击、未建模的干扰以及连锁故障。尽管其至关重要，但韧性仍然是一个难以捉摸的概念，其数学基础仍处于未充分发展状态。与鲁棒性和可靠性不同，韧性基于这样一个事实，即中断不可避免。韧性从弹性角度关注恢复到有利状态的能力，而从可塑性角度则涉及能够灵活扩展其状态、假设和行动方案的实体（或网络），并通过实时适应和重新配置进行转变。这种不断的情境意识以及适应世界模型并基于潜在系统故障进行反事实推理以找到最佳应对措施，是韧性的一个核心方面。本文旨在首先定义韧性并将其与可靠性和鲁棒性区分开来，然后深入探讨韧性数学，最后通过呈现细致的评价指标并讨论适用于网络韧性独特特性的折衷方案来总结。 

---
# Solving tricky quantum optics problems with assistance from (artificial) intelligence 

**Title (ZH)**: 使用人工智能辅助解决棘手的量子光学问题 

**Authors**: Manas Pandey, Bharath Hebbe Madhusudhana, Saikat Ghosh, Dmitry Budker  

**Link**: [PDF](https://arxiv.org/pdf/2506.12770)  

**Abstract**: The capabilities of modern artificial intelligence (AI) as a ``scientific collaborator'' are explored by engaging it with three nuanced problems in quantum optics: state populations in optical pumping, resonant transitions between decaying states (the Burshtein effect), and degenerate mirrorless lasing. Through iterative dialogue, the authors observe that AI models--when prompted and corrected--can reason through complex scenarios, refine their answers, and provide expert-level guidance, closely resembling the interaction with an adept colleague. The findings highlight that AI democratizes access to sophisticated modeling and analysis, shifting the focus in scientific practice from technical mastery to the generation and testing of ideas, and reducing the time for completing research tasks from days to minutes. 

**Abstract (ZH)**: 现代人工智能作为“科学合作者”的能力通过与量子光学三个细腻问题的互动进行探索：光泵中的态分布、衰减态之间的共振跃迁（伯斯廷效应）以及无反射镜杂化激光。通过迭代对话，作者观察到，在被提示和纠正后，AI模型能够理清复杂场景、改进答案，并提供专家级指导，这一过程类似于与一位熟练同事的交互。研究结果表明，AI使高级建模与分析变得更加普及，使科学研究的重心从技术 Mastery 转向思想的产生与验证，并将完成研究任务的时间从几天缩短到几分钟。 

---
# AFBS:Buffer Gradient Selection in Semi-asynchronous Federated Learning 

**Title (ZH)**: AFBS：半异步联邦学习中的缓冲梯度选择 

**Authors**: Chaoyi Lu, Yiding Sun, Jinqian Chen, Zhichuan Yang, Jiangming Pan, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12754)  

**Abstract**: Asynchronous federated learning (AFL) accelerates training by eliminating the need to wait for stragglers, but its asynchronous nature introduces gradient staleness, where outdated gradients degrade performance. Existing solutions address this issue with gradient buffers, forming a semi-asynchronous framework. However, this approach struggles when buffers accumulate numerous stale gradients, as blindly aggregating all gradients can harm training. To address this, we propose AFBS (Asynchronous FL Buffer Selection), the first algorithm to perform gradient selection within buffers while ensuring privacy protection. Specifically, the client sends the random projection encrypted label distribution matrix before training, and the server performs client clustering based on it. During training, server scores and selects gradients within each cluster based on their informational value, discarding low-value gradients to enhance semi-asynchronous federated learning. Extensive experiments in highly heterogeneous system and data environments demonstrate AFBS's superior performance compared to state-of-the-art methods. Notably, on the most challenging task, CIFAR-100, AFBS improves accuracy by up to 4.8% over the previous best algorithm and reduces the time to reach target accuracy by 75%. 

**Abstract (ZH)**: 异步联邦学习中基于缓冲的梯度选择（AFBS）：保护隐私的同时提高性能 

---
# Adaptive Dropout: Unleashing Dropout across Layers for Generalizable Image Super-Resolution 

**Title (ZH)**: 自适应丢弃：跨层释放丢弃以实现泛化图像超分辨率 

**Authors**: Hang Xu, Wei Yu, Jiangtong Tan, Zhen Zou, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12738)  

**Abstract**: Blind Super-Resolution (blind SR) aims to enhance the model's generalization ability with unknown degradation, yet it still encounters severe overfitting issues. Some previous methods inspired by dropout, which enhances generalization by regularizing features, have shown promising results in blind SR. Nevertheless, these methods focus solely on regularizing features before the final layer and overlook the need for generalization in features at intermediate layers. Without explicit regularization of features at intermediate layers, the blind SR network struggles to obtain well-generalized feature representations. However, the key challenge is that directly applying dropout to intermediate layers leads to a significant performance drop, which we attribute to the inconsistency in training-testing and across layers it introduced. Therefore, we propose Adaptive Dropout, a new regularization method for blind SR models, which mitigates the inconsistency and facilitates application across intermediate layers of networks. Specifically, for training-testing inconsistency, we re-design the form of dropout and integrate the features before and after dropout adaptively. For inconsistency in generalization requirements across different layers, we innovatively design an adaptive training strategy to strengthen feature propagation by layer-wise annealing. Experimental results show that our method outperforms all past regularization methods on both synthetic and real-world benchmark datasets, also highly effective in other image restoration tasks. Code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 盲超分辨率（盲SR）旨在通过未知退化来增强模型的泛化能力，但仍面临严重的过拟合问题。一些受dropout启发的方法通过正则化特征来增强泛化能力，在盲SR中显示出有前途的结果。然而，这些方法仅专注于在最后一层之前正则化特征，而忽视了中间层特征也需要泛化的需求。没有中间层特征的显式正则化，盲SR网络难以获得良好的泛化特征表示。然而，关键挑战在于直接在中间层应用dropout会导致显著的性能下降，这归因于训练-测试间以及跨层引入的一致性问题。因此，我们提出了一种新的盲SR模型正则化方法——自适应dropout，该方法减轻了不一致性并促进了在网络中间层的应用。具体而言，对于训练-测试不一致性，我们重新设计了dropout的形式，并在dropout前后适当地整合特征。对于不同层间泛化要求的一致性问题，我们创新设计了一种逐层退火的自适应训练策略以加强特征传播。实验结果表明，我们的方法在合成和真实基准数据集上均优于所有以往的正则化方法，并且在其他图像恢复任务中也很有效。代码可在\href{this https URL}{此链接}获得。 

---
# Unsupervised Contrastive Learning Using Out-Of-Distribution Data for Long-Tailed Dataset 

**Title (ZH)**: 使用域外数据的无监督对比学习用于长尾数据集 

**Authors**: Cuong Manh Hoang, Yeejin Lee, Byeongkeun Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12698)  

**Abstract**: This work addresses the task of self-supervised learning (SSL) on a long-tailed dataset that aims to learn balanced and well-separated representations for downstream tasks such as image classification. This task is crucial because the real world contains numerous object categories, and their distributions are inherently imbalanced. Towards robust SSL on a class-imbalanced dataset, we investigate leveraging a network trained using unlabeled out-of-distribution (OOD) data that are prevalently available online. We first train a network using both in-domain (ID) and sampled OOD data by back-propagating the proposed pseudo semantic discrimination loss alongside a domain discrimination loss. The OOD data sampling and loss functions are designed to learn a balanced and well-separated embedding space. Subsequently, we further optimize the network on ID data by unsupervised contrastive learning while using the previously trained network as a guiding network. The guiding network is utilized to select positive/negative samples and to control the strengths of attractive/repulsive forces in contrastive learning. We also distil and transfer its embedding space to the training network to maintain balancedness and separability. Through experiments on four publicly available long-tailed datasets, we demonstrate that the proposed method outperforms previous state-of-the-art methods. 

**Abstract (ZH)**: 本研究解决了针对长尾数据集的自监督学习任务，旨在为诸如图像分类等下游任务学习平衡且分离良好的表示。为实现此任务，我们研究了利用大量在线获取的未标记异类数据（OOD）进行网络训练的方法。首先，我们通过同时反向传播所提出的伪语义鉴别损失与领域鉴别损失，使用领域内（ID）数据和采样的OOD数据来训练网络。OOD数据的采样和损失函数旨在学习一个平衡且分离良好的嵌入空间。随后，我们进一步通过无监督对比学习在ID数据上优化网络，并使用预先训练的网络作为引导网络。引导网络用于选择正/负样本，并控制对比学习中吸引力/排斥力的强度。我们还将其嵌入空间进行知识蒸馏和迁移，以保持平衡性和可分离性。通过在四个公开的长尾数据集上的实验，我们证明了所提出的方法优于先前的最先进的方法。 

---
# ANIRA: An Architecture for Neural Network Inference in Real-Time Audio Applications 

**Title (ZH)**: ANIRA: 用于实时音频应用的神经网络推理架构 

**Authors**: Valentin Ackva, Fares Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12665)  

**Abstract**: Numerous tools for neural network inference are currently available, yet many do not meet the requirements of real-time audio applications. In response, we introduce anira, an efficient cross-platform library. To ensure compatibility with a broad range of neural network architectures and frameworks, anira supports ONNX Runtime, LibTorch, and TensorFlow Lite as backends. Each inference engine exhibits real-time violations, which anira mitigates by decoupling the inference from the audio callback to a static thread pool. The library incorporates built-in latency management and extensive benchmarking capabilities, both crucial to ensure a continuous signal flow. Three different neural network architectures for audio effect emulation are then subjected to benchmarking across various configurations. Statistical modeling is employed to identify the influence of various factors on performance. The findings indicate that for stateless models, ONNX Runtime exhibits the lowest runtimes. For stateful models, LibTorch demonstrates the fastest performance. Our results also indicate that for certain model-engine combinations, the initial inferences take longer, particularly when these inferences exhibit a higher incidence of real-time violations. 

**Abstract (ZH)**: 面向实时音频应用的高效跨平台神经网络推理库anira及其性能评估 

---
# DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty 

**Title (ZH)**: DR-SAC: 分布鲁棒软Actor-critic方法在不确定性下的强化学习 

**Authors**: Mingxuan Cui, Duo Zhou, Yuxuan Han, Grani A. Hanasusanto, Qiong Wang, Huan Zhang, Zhengyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12622)  

**Abstract**: Deep reinforcement learning (RL) has achieved significant success, yet its application in real-world scenarios is often hindered by a lack of robustness to environmental uncertainties. To solve this challenge, some robust RL algorithms have been proposed, but most are limited to tabular settings. In this work, we propose Distributionally Robust Soft Actor-Critic (DR-SAC), a novel algorithm designed to enhance the robustness of the state-of-the-art Soft Actor-Critic (SAC) algorithm. DR-SAC aims to maximize the expected value with entropy against the worst possible transition model lying in an uncertainty set. A distributionally robust version of the soft policy iteration is derived with a convergence guarantee. For settings where nominal distributions are unknown, such as offline RL, a generative modeling approach is proposed to estimate the required nominal distributions from data. Furthermore, experimental results on a range of continuous control benchmark tasks demonstrate our algorithm achieves up to $9.8$ times the average reward of the SAC baseline under common perturbations. Additionally, compared with existing robust reinforcement learning algorithms, DR-SAC significantly improves computing efficiency and applicability to large-scale problems. 

**Abstract (ZH)**: 分布鲁棒软演员-评论家（DR-SAC）算法：一种增强软演员-评论家算法鲁棒性的新方法 

---
# Konooz: Multi-domain Multi-dialect Corpus for Named Entity Recognition 

**Title (ZH)**: Konooz: 多领域多方言语料库命名实体识别 

**Authors**: Nagham Hamad, Mohammed Khalilia, Mustafa Jarrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.12615)  

**Abstract**: We introduce Konooz, a novel multi-dimensional corpus covering 16 Arabic dialects across 10 domains, resulting in 160 distinct corpora. The corpus comprises about 777k tokens, carefully collected and manually annotated with 21 entity types using both nested and flat annotation schemes - using the Wojood guidelines. While Konooz is useful for various NLP tasks like domain adaptation and transfer learning, this paper primarily focuses on benchmarking existing Arabic Named Entity Recognition (NER) models, especially cross-domain and cross-dialect model performance. Our benchmarking of four Arabic NER models using Konooz reveals a significant drop in performance of up to 38% when compared to the in-distribution data. Furthermore, we present an in-depth analysis of domain and dialect divergence and the impact of resource scarcity. We also measured the overlap between domains and dialects using the Maximum Mean Discrepancy (MMD) metric, and illustrated why certain NER models perform better on specific dialects and domains. Konooz is open-source and publicly available at this https URL 

**Abstract (ZH)**: Konooz：一种涵盖16种阿拉伯方言的多维度语料库及其在阿拉伯命名实体识别模型评估中的应用 

---
# An Exploration of Mamba for Speech Self-Supervised Models 

**Title (ZH)**: Mamba在语音自监督模型中的探索 

**Authors**: Tzu-Quan Lin, Heng-Cheng Kuo, Tzu-Chieh Wei, Hsi-Chun Cheng, Chun-Wei Chen, Hsien-Fu Hsiao, Yu Tsao, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.12606)  

**Abstract**: While Mamba has demonstrated strong performance in language modeling, its potential as a speech self-supervised (SSL) model remains underexplored, with prior studies limited to isolated tasks. To address this, we explore Mamba-based HuBERT models as alternatives to Transformer-based SSL architectures. Leveraging the linear-time Selective State Space, these models enable fine-tuning on long-context ASR with significantly lower compute. Moreover, they show superior performance when fine-tuned for streaming ASR. Beyond fine-tuning, these models show competitive performance on SUPERB probing benchmarks, particularly in causal settings. Our analysis shows that they yield higher-quality quantized representations and capture speaker-related features more distinctly than Transformer-based models. These findings highlight Mamba-based SSL as a promising and complementary direction for long-sequence modeling, real-time speech modeling, and speech unit extraction. 

**Abstract (ZH)**: Mamba在语音自监督学习中的潜力及其在长期序列建模、实时语音建模和语音单元提取中的应用探索 

---
# DoTA-RAG: Dynamic of Thought Aggregation RAG 

**Title (ZH)**: DoTA-RAG: 思维聚合RAG的动态过程 

**Authors**: Saksorn Ruangtanusak, Natthapath Rungseesiripak, Peerawat Rojratchadakorn, Monthol Charattrakool, Natapong Nitarach  

**Link**: [PDF](https://arxiv.org/pdf/2506.12571)  

**Abstract**: In this paper, we introduce DoTA-RAG (Dynamic-of-Thought Aggregation RAG), a retrieval-augmented generation system optimized for high-throughput, large-scale web knowledge indexes. Traditional RAG pipelines often suffer from high latency and limited accuracy over massive, diverse datasets. DoTA-RAG addresses these challenges with a three-stage pipeline: query rewriting, dynamic routing to specialized sub-indexes, and multi-stage retrieval and ranking. We further enhance retrieval by evaluating and selecting a superior embedding model, re-embedding the large FineWeb-10BT corpus. Moreover, we create a diverse Q&A dataset of 500 questions generated via the DataMorgana setup across a broad range of WebOrganizer topics and formats. DoTA-RAG improves the answer correctness score from 0.752 (baseline, using LiveRAG pre-built vector store) to 1.478 while maintaining low latency, and it achieves a 0.929 correctness score on the Live Challenge Day. These results highlight DoTA-RAG's potential for practical deployment in domains requiring fast, reliable access to large and evolving knowledge sources. 

**Abstract (ZH)**: DoTA-RAG：动态思维聚合RAG——一种优化的高通量大规模网络知识索引生成系统 

---
# MVP-CBM:Multi-layer Visual Preference-enhanced Concept Bottleneck Model for Explainable Medical Image Classification 

**Title (ZH)**: MVP-CBM：多层视觉偏好增强概念瓶颈模型可解释的医疗图像分类 

**Authors**: Chunjiang Wang, Kun Zhang, Yandong Liu, Zhiyang He, Xiaodong Tao, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12568)  

**Abstract**: The concept bottleneck model (CBM), as a technique improving interpretability via linking predictions to human-understandable concepts, makes high-risk and life-critical medical image classification credible. Typically, existing CBM methods associate the final layer of visual encoders with concepts to explain the model's predictions. However, we empirically discover the phenomenon of concept preference variation, that is, the concepts are preferably associated with the features at different layers than those only at the final layer; yet a blind last-layer-based association neglects such a preference variation and thus weakens the accurate correspondences between features and concepts, impairing model interpretability. To address this issue, we propose a novel Multi-layer Visual Preference-enhanced Concept Bottleneck Model (MVP-CBM), which comprises two key novel modules: (1) intra-layer concept preference modeling, which captures the preferred association of different concepts with features at various visual layers, and (2) multi-layer concept sparse activation fusion, which sparsely aggregates concept activations from multiple layers to enhance performance. Thus, by explicitly modeling concept preferences, MVP-CBM can comprehensively leverage multi-layer visual information to provide a more nuanced and accurate explanation of model decisions. Extensive experiments on several public medical classification benchmarks demonstrate that MVP-CBM achieves state-of-the-art accuracy and interoperability, verifying its superiority. Code is available at this https URL. 

**Abstract (ZH)**: 多层视觉偏好增强概念瓶颈模型（MVP-CBM）：一种提高医疗图像分类解释性的方法 

---
# Fairness Research For Machine Learning Should Integrate Societal Considerations 

**Title (ZH)**: 机器学习的公平性研究应融入社会考量 

**Authors**: Yijun Bian, Lei You  

**Link**: [PDF](https://arxiv.org/pdf/2506.12556)  

**Abstract**: Enhancing fairness in machine learning (ML) systems is increasingly important nowadays. While current research focuses on assistant tools for ML pipelines to promote fairness within them, we argue that: 1) The significance of properly defined fairness measures remains underestimated; and 2) Fairness research in ML should integrate societal considerations. The reasons include that detecting discrimination is critical due to the widespread deployment of ML systems and that human-AI feedback loops amplify biases, even when only small social and political biases persist. 

**Abstract (ZH)**: 增强机器学习系统中的公平性日益重要：重新审视公平度量的社会考量与偏见放大效应 

---
# Neuromorphic Online Clustering and Its Application to Spike Sorting 

**Title (ZH)**: 神经形态在线聚类及其在尖峰排序中的应用 

**Authors**: James E. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2506.12555)  

**Abstract**: Active dendrites are the basis for biologically plausible neural networks possessing many desirable features of the biological brain including flexibility, dynamic adaptability, and energy efficiency. A formulation for active dendrites using the notational language of conventional machine learning is put forward as an alternative to a spiking neuron formulation. Based on this formulation, neuromorphic dendrites are developed as basic neural building blocks capable of dynamic online clustering. Features and capabilities of neuromorphic dendrites are demonstrated via a benchmark drawn from experimental neuroscience: spike sorting. Spike sorting takes inputs from electrical probes implanted in neural tissue, detects voltage spikes (action potentials) emitted by neurons, and attempts to sort the spikes according to the neuron that emitted them. Many spike sorting methods form clusters based on the shapes of action potential waveforms, under the assumption that spikes emitted by a given neuron have similar shapes and will therefore map to the same cluster. Using a stream of synthetic spike shapes, the accuracy of the proposed dendrite is compared with the more compute-intensive, offline k-means clustering approach. Overall, the dendrite outperforms k-means and has the advantage of requiring only a single pass through the input stream, learning as it goes. The capabilities of the neuromorphic dendrite are demonstrated for a number of scenarios including dynamic changes in the input stream, differing neuron spike rates, and varying neuron counts. 

**Abstract (ZH)**: 活性树突是实现具备生物脑多项 desirable 特征（包括灵活性、动态适应性和能量效率）的生物合现实神经网络的基础。提出了一种使用传统机器学习符号语言的形式化方法来替代脉冲神经元形式化方法，以活性树突为基础开发了神经形态树突，作为基本的神经元构建块，具备动态在线聚类功能。通过实验神经科学中的基准测试——尖峰分类，展示了神经形态树突的特性和能力。尖峰分类从植入神经组织的电极探针接收输入，检测由神经元发出的电压尖峰（动作电位），并尝试按尖峰发出的神经元对其进行分类。许多尖峰分类方法基于动作电位波形的形状形成聚类，假设来自同一神经元的尖峰会具有相似的形状并因此分配到相同的聚类。使用尖峰形状的合成流，比较了提出树突的准确性与计算密集型的离线 k-均值聚类方法。总体而言，树突表现更优，并且具有仅需一次通过输入流即可学习的优点。展示了神经形态树突在包括输入流动态变化、不同的神经元尖峰速率和变化的神经元数量等场景下的能力。 

---
# PLD: A Choice-Theoretic List-Wise Knowledge Distillation 

**Title (ZH)**: PLD：一种基于选择理论的列表型知识蒸馏 

**Authors**: Ejafa Bassam, Dawei Zhu, Kaigui Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.12542)  

**Abstract**: Knowledge distillation is a model compression technique in which a compact "student" network is trained to replicate the predictive behavior of a larger "teacher" network. In logit-based knowledge distillation it has become the de facto approach to augment cross-entropy with a distillation term. Typically this term is either a KL divergence-matching marginal probabilities or a correlation-based loss capturing intra- and inter-class relationships but in every case it sits as an add-on to cross-entropy with its own weight that must be carefully tuned. In this paper we adopt a choice-theoretic perspective and recast knowledge distillation under the Plackett-Luce model by interpreting teacher logits as "worth" scores. We introduce Plackett-Luce Distillation (PLD), a weighted list-wise ranking loss in which the teacher model transfers knowledge of its full ranking of classes, weighting each ranked choice by its own confidence. PLD directly optimizes a single teacher-optimal ranking of the true label first, followed by the remaining classes in descending teacher confidence, yielding a convex, translation-invariant surrogate that subsumes weighted cross-entropy. Empirically on standard image classification benchmarks, PLD improves Top-1 accuracy by an average of +0.42% over DIST (arXiv:2205.10536) and +1.04% over KD (arXiv:1503.02531) in homogeneous settings and by +0.48% and +1.09% over DIST and KD, respectively, in heterogeneous settings. 

**Abstract (ZH)**: 基于Plackett-Luce模型的知识蒸馏：一种带有权重的一致排名损失 

---
# BSA: Ball Sparse Attention for Large-scale Geometries 

**Title (ZH)**: BSA: 球稀疏注意力机制在大规模几何结构中的应用 

**Authors**: Catalin E. Brita, Hieu Nguyen, Lohithsai Yadala Chanchu, Domonkos Nagy, Maksim Zhdanov  

**Link**: [PDF](https://arxiv.org/pdf/2506.12541)  

**Abstract**: Self-attention scales quadratically with input size, limiting its use for large-scale physical systems. Although sparse attention mechanisms provide a viable alternative, they are primarily designed for regular structures such as text or images, making them inapplicable for irregular geometries. In this work, we present Ball Sparse Attention (BSA), which adapts Native Sparse Attention (NSA) (Yuan et al., 2025) to unordered point sets by imposing regularity using the Ball Tree structure from the Erwin Transformer (Zhdanov et al., 2025). We modify NSA's components to work with ball-based neighborhoods, yielding a global receptive field at sub-quadratic cost. On an airflow pressure prediction task, we achieve accuracy comparable to Full Attention while significantly reducing the theoretical computational complexity. Our implementation is available at this https URL. 

**Abstract (ZH)**: 球稀疏注意机制（BSA）：将原生稀疏注意机制（NSA）应用于无序点集（Yuan et al., 2025） 

---
# Similarity as Reward Alignment: Robust and Versatile Preference-based Reinforcement Learning 

**Title (ZH)**: 相似性作为奖励对齐：稳健且用途广泛的基于偏好的强化学习 

**Authors**: Sara Rajaram, R. James Cotton, Fabian H. Sinz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12529)  

**Abstract**: Preference-based Reinforcement Learning (PbRL) entails a variety of approaches for aligning models with human intent to alleviate the burden of reward engineering. However, most previous PbRL work has not investigated the robustness to labeler errors, inevitable with labelers who are non-experts or operate under time constraints. Additionally, PbRL algorithms often target very specific settings (e.g. pairwise ranked preferences or purely offline learning). We introduce Similarity as Reward Alignment (SARA), a simple contrastive framework that is both resilient to noisy labels and adaptable to diverse feedback formats and training paradigms. SARA learns a latent representation of preferred samples and computes rewards as similarities to the learned latent. We demonstrate strong performance compared to baselines on continuous control offline RL benchmarks. We further demonstrate SARA's versatility in applications such as trajectory filtering for downstream tasks, cross-task preference transfer, and reward shaping in online learning. 

**Abstract (ZH)**: 基于偏好强化学习的类似性作为奖励对齐（SARA）：鲁棒性强且适应多种反馈格式和训练范式的简单对比框架 

---
# Delving into Instance-Dependent Label Noise in Graph Data: A Comprehensive Study and Benchmark 

**Title (ZH)**: 图数据中实例依赖的标签噪声：一项全面的研究与基准测试 

**Authors**: Suyeon Kim, SeongKu Kang, Dongwoo Kim, Jungseul Ok, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12468)  

**Abstract**: Graph Neural Networks (GNNs) have achieved state-of-the-art performance in node classification tasks but struggle with label noise in real-world data. Existing studies on graph learning with label noise commonly rely on class-dependent label noise, overlooking the complexities of instance-dependent noise and falling short of capturing real-world corruption patterns. We introduce BeGIN (Benchmarking for Graphs with Instance-dependent Noise), a new benchmark that provides realistic graph datasets with various noise types and comprehensively evaluates noise-handling strategies across GNN architectures, noisy label detection, and noise-robust learning. To simulate instance-dependent corruptions, BeGIN introduces algorithmic methods and LLM-based simulations. Our experiments reveal the challenges of instance-dependent noise, particularly LLM-based corruption, and underscore the importance of node-specific parameterization to enhance GNN robustness. By comprehensively evaluating noise-handling strategies, BeGIN provides insights into their effectiveness, efficiency, and key performance factors. We expect that BeGIN will serve as a valuable resource for advancing research on label noise in graphs and fostering the development of robust GNN training methods. The code is available at this https URL. 

**Abstract (ZH)**: 图神经网络（GNNs）在节点分类任务中取得了最先进的性能，但在应对实际数据中的标签噪声时存在困难。现有的图学习中对抗标签噪声的研究主要依赖于类相关的标签噪声，忽视了实例相关的噪声复杂性，未能捕捉到真实的污染模式。我们引入了BeGIN（Benchmarking for Graphs with Instance-dependent Noise），这是一种新的基准，提供了具有多种噪声类型的现实图数据集，并全面评估了GNN架构、嘈杂标签检测和噪声鲁棒学习的方法。为了模拟实例相关的污染，BeGIN引入了算法方法和基于LLM的模拟。我们的实验揭示了实例相关的噪声，特别是基于LLM的污染所带来的挑战，并强调了节点特定参数化对增强GNN鲁棒性的重要性。通过全面评估噪声处理策略，BeGIN提供了它们有效性的见解、效率和关键性能因素。我们期望BeGIN将成为推动图中标签噪声研究和促进鲁棒GNN训练方法发展的宝贵资源。代码可在以下网址获取。 

---
# Merlin: Multi-View Representation Learning for Robust Multivariate Time Series Forecasting with Unfixed Missing Rates 

**Title (ZH)**: Merlin: 多视图表示学习以实现鲁棒多变量时间序列预测，面对不确定的缺失率 

**Authors**: Chengqing Yu, Fei Wang, Chuanguang Yang, Zezhi Shao, Tao Sun, Tangwen Qian, Wei Wei, Zhulin An, Yongjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12459)  

**Abstract**: Multivariate Time Series Forecasting (MTSF) involves predicting future values of multiple interrelated time series. Recently, deep learning-based MTSF models have gained significant attention for their promising ability to mine semantics (global and local information) within MTS data. However, these models are pervasively susceptible to missing values caused by malfunctioning data collectors. These missing values not only disrupt the semantics of MTS, but their distribution also changes over time. Nevertheless, existing models lack robustness to such issues, leading to suboptimal forecasting performance. To this end, in this paper, we propose Multi-View Representation Learning (Merlin), which can help existing models achieve semantic alignment between incomplete observations with different missing rates and complete observations in MTS. Specifically, Merlin consists of two key modules: offline knowledge distillation and multi-view contrastive learning. The former utilizes a teacher model to guide a student model in mining semantics from incomplete observations, similar to those obtainable from complete observations. The latter improves the student model's robustness by learning from positive/negative data pairs constructed from incomplete observations with different missing rates, ensuring semantic alignment across different missing rates. Therefore, Merlin is capable of effectively enhancing the robustness of existing models against unfixed missing rates while preserving forecasting accuracy. Experiments on four real-world datasets demonstrate the superiority of Merlin. 

**Abstract (ZH)**: 多元时间序列预测中的多视图表示学习（Merlin）：一种应对不固定缺失率的鲁棒方法 

---
# A Pluggable Multi-Task Learning Framework for Sentiment-Aware Financial Relation Extraction 

**Title (ZH)**: 面向情感感知金融关系提取的插件式多任务学习框架 

**Authors**: Jinming Luo, Hailin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12452)  

**Abstract**: Relation Extraction (RE) aims to extract semantic relationships in texts from given entity pairs, and has achieved significant improvements. However, in different domains, the RE task can be influenced by various factors. For example, in the financial domain, sentiment can affect RE results, yet this factor has been overlooked by modern RE models. To address this gap, this paper proposes a Sentiment-aware-SDP-Enhanced-Module (SSDP-SEM), a multi-task learning approach for enhancing financial RE. Specifically, SSDP-SEM integrates the RE models with a pluggable auxiliary sentiment perception (ASP) task, enabling the RE models to concurrently navigate their attention weights with the text's sentiment. We first generate detailed sentiment tokens through a sentiment model and insert these tokens into an instance. Then, the ASP task focuses on capturing nuanced sentiment information through predicting the sentiment token positions, combining both sentiment insights and the Shortest Dependency Path (SDP) of syntactic information. Moreover, this work employs a sentiment attention information bottleneck regularization method to regulate the reasoning process. Our experiment integrates this auxiliary task with several prevalent frameworks, and the results demonstrate that most previous models benefit from the auxiliary task, thereby achieving better results. These findings highlight the importance of effectively leveraging sentiment in the financial RE task. 

**Abstract (ZH)**: 情感意识-最短依赖路径增强模块：面向金融领域的关系提取 

---
# Style-based Composer Identification and Attribution of Symbolic Music Scores: a Systematic Survey 

**Title (ZH)**: 基于风格的乐谱作曲家识别与归属：一项系统综述 

**Authors**: Federico Simonetta  

**Link**: [PDF](https://arxiv.org/pdf/2506.12440)  

**Abstract**: This paper presents the first comprehensive systematic review of literature on style-based composer identification and authorship attribution in symbolic music scores. Addressing the critical need for improved reliability and reproducibility in this field, the review rigorously analyzes 58 peer-reviewed papers published across various historical periods, with the search adapted to evolving terminology. The analysis critically assesses prevailing repertoires, computational approaches, and evaluation methodologies, highlighting significant challenges. It reveals that a substantial portion of existing research suffers from inadequate validation protocols and an over-reliance on simple accuracy metrics for often imbalanced datasets, which can undermine the credibility of attribution claims. The crucial role of robust metrics like Balanced Accuracy and rigorous cross-validation in ensuring trustworthy results is emphasized. The survey also details diverse feature representations and the evolution of machine learning models employed. Notable real-world authorship attribution cases, such as those involving works attributed to Bach, Josquin Desprez, and Lennon-McCartney, are specifically discussed, illustrating the opportunities and pitfalls of applying computational techniques to resolve disputed musical provenance. Based on these insights, a set of actionable guidelines for future research are proposed. These recommendations are designed to significantly enhance the reliability, reproducibility, and musicological validity of composer identification and authorship attribution studies, fostering more robust and interpretable computational stylistic analysis. 

**Abstract (ZH)**: 本文提供了基于风格的作曲家识别和乐谱著作者归属 Literature Review 的首次全面系统性综述。针对该领域可靠性与再现性改进的迫切需求，研究严格分析了跨越不同历史时期的 58 篇同行评审论文，搜索策略适应术语演变。分析批判性地评估了现有的 repertoire、计算方法和评价方法，指出了显著的挑战。研究揭示，现有研究中很大一部分缺乏充分的验证协议，并过度依赖简单的准确度指标，特别是在不平衡数据集的情况下，这可能削弱著作者归属声明的可信度。强调了使用稳健的指标如平衡准确度及严格的交叉验证以确保可信赖结果的重要性。调查还详细介绍了多样化的特征表示及其所使用的机器学习模型的演变。具体讨论了涉及巴赫、若斯坎·迪普雷兹和 Lennon-McCartney 的著名现实世界著作者归属案例，说明了如何利用计算技术解决有争议的音乐来源问题。基于这些见解，提出了未来研究的一系列可操作性指南。这些建议旨在显著提高作曲家识别和著作者归属研究的可靠性、再现性和音乐学有效性，促进更加稳健和可解释的计算风格分析。 

---
# EXGnet: a single-lead explainable-AI guided multiresolution network with train-only quantitative features for trustworthy ECG arrhythmia classification 

**Title (ZH)**: EXGnet：一种基于单导联可解释AI引导的多分辨网络，用于可信的心电图心律失常分类 

**Authors**: Tushar Talukder Showrav, Soyabul Islam Lincoln, Md. Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12404)  

**Abstract**: Background: Deep learning has significantly advanced ECG arrhythmia classification, enabling high accuracy in detecting various cardiac conditions. The use of single-lead ECG systems is crucial for portable devices, as they offer convenience and accessibility for continuous monitoring in diverse settings. However, the interpretability and reliability of deep learning models in clinical applications poses challenges due to their black-box nature. Methods: To address these challenges, we propose EXGnet, a single-lead, trustworthy ECG arrhythmia classification network that integrates multiresolution feature extraction with Explainable Artificial Intelligence (XAI) guidance and train only quantitative features. Results: Trained on two public datasets, including Chapman and Ningbo, EXGnet demonstrates superior performance through key metrics such as Accuracy, F1-score, Sensitivity, and Specificity. The proposed method achieved average five fold accuracy of 98.762%, and 96.932% and average F1-score of 97.910%, and 95.527% on the Chapman and Ningbo datasets, respectively. Conclusions: By employing XAI techniques, specifically Grad-CAM, the model provides visual insights into the relevant ECG segments it analyzes, thereby enhancing clinician trust in its predictions. While quantitative features further improve classification performance, they are not required during testing, making the model suitable for real-world applications. Overall, EXGnet not only achieves better classification accuracy but also addresses the critical need for interpretability in deep learning, facilitating broader adoption in portable ECG monitoring. 

**Abstract (ZH)**: 背景：深度学习显著推进了心电图（ECG）心律失常分类，使其能够在各种心脏条件下实现高精度检测。单导联ECG系统对于便携设备至关重要，因为它们在多样化的环境中提供了便利性和可访问性，用于持续监测。然而，深学习模型在临床应用中的解释性和可靠性因它们的黑匣子性质而受到挑战。方法：为了解决这些挑战，我们提出EXGnet，这是一种结合多分辨率特征提取和可解释人工智能（XAI）指导的单导联、可信赖的心律失常分类网络，仅训练定量特征。结果：EXGnet在两个公开数据集（包括Chapman和宁波）上进行训练，通过关键指标（如准确率、F1分数、灵敏度和特异度）展示了卓越的性能。在Chapman和宁波数据集上，提出的方法分别实现了平均五折准确率为98.762%和96.932%，平均F1分数为97.910%和95.527%。结论：通过采用XAI技术，特别是Grad-CAM，该模型为分析的相关ECG段落提供了可视化见解，从而增强临床医生对其预测的信任。尽管定量特征进一步提高了分类性能，但在测试时无需使用这些特征，使该模型适用于实际应用。总体而言，EXGnet不仅实现了更好的分类准确率，还解决了深度学习解释性方面的重要需求，促进了其在便携式ECG监测中的更广泛采用。 

---
# LARGO: Low-Rank Regulated Gradient Projection for Robust Parameter Efficient Fine-Tuning 

**Title (ZH)**: LARGO: 低秩调节梯度投影的稳健参数高效微调 

**Authors**: Haotian Zhang, Liu Liu, Baosheng Yu, Jiayan Qiu, Yanwei Ren, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12394)  

**Abstract**: The advent of parameter-efficient fine-tuning methods has significantly reduced the computational burden of adapting large-scale pretrained models to diverse downstream tasks. However, existing approaches often struggle to achieve robust performance under domain shifts while maintaining computational efficiency. To address this challenge, we propose Low-rAnk Regulated Gradient Projection (LARGO) algorithm that integrates dynamic constraints into low-rank adaptation methods. Specifically, LARGO incorporates parallel trainable gradient projections to dynamically regulate layer-wise updates, retaining the Out-Of-Distribution robustness of pretrained model while preserving inter-layer independence. Additionally, it ensures computational efficiency by mitigating the influence of gradient dependencies across layers during weight updates. Besides, through leveraging singular value decomposition of pretrained weights for structured initialization, we incorporate an SVD-based initialization strategy that minimizing deviation from pretrained knowledge. Through extensive experiments on diverse benchmarks, LARGO achieves state-of-the-art performance across in-domain and out-of-distribution scenarios, demonstrating improved robustness under domain shifts with significantly lower computational overhead compared to existing PEFT methods. The source code will be released soon. 

**Abstract (ZH)**: 低秩调节梯度投影（LARGO）算法：在保持计算效率的同时提升领域泛化鲁棒性 

---
# Revisiting Clustering of Neural Bandits: Selective Reinitialization for Mitigating Loss of Plasticity 

**Title (ZH)**: 重新审视神经bandits的聚类：选择性重初始化以减轻可塑性丧失的影响 

**Authors**: Zhiyuan Su, Sunhao Dai, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12389)  

**Abstract**: Clustering of Bandits (CB) methods enhance sequential decision-making by grouping bandits into clusters based on similarity and incorporating cluster-level contextual information, demonstrating effectiveness and adaptability in applications like personalized streaming recommendations. However, when extending CB algorithms to their neural version (commonly referred to as Clustering of Neural Bandits, or CNB), they suffer from loss of plasticity, where neural network parameters become rigid and less adaptable over time, limiting their ability to adapt to non-stationary environments (e.g., dynamic user preferences in recommendation). To address this challenge, we propose Selective Reinitialization (SeRe), a novel bandit learning framework that dynamically preserves the adaptability of CNB algorithms in evolving environments. SeRe leverages a contribution utility metric to identify and selectively reset underutilized units, mitigating loss of plasticity while maintaining stable knowledge retention. Furthermore, when combining SeRe with CNB algorithms, the adaptive change detection mechanism adjusts the reinitialization frequency according to the degree of non-stationarity, ensuring effective adaptation without unnecessary resets. Theoretically, we prove that SeRe enables sublinear cumulative regret in piecewise-stationary environments, outperforming traditional CNB approaches in long-term performances. Extensive experiments on six real-world recommendation datasets demonstrate that SeRe-enhanced CNB algorithms can effectively mitigate the loss of plasticity with lower regrets, improving adaptability and robustness in dynamic settings. 

**Abstract (ZH)**: 基于集群的强化学习中选择性重初始化框架（SeRe）：在动态环境中提升神经臂拉普拉斯算法的可适应性和鲁棒性 

---
# Component Based Quantum Machine Learning Explainability 

**Title (ZH)**: 基于组件的量子机器学习可解释性 

**Authors**: Barra White, Krishnendu Guha  

**Link**: [PDF](https://arxiv.org/pdf/2506.12378)  

**Abstract**: Explainable ML algorithms are designed to provide transparency and insight into their decision-making process. Explaining how ML models come to their prediction is critical in fields such as healthcare and finance, as it provides insight into how models can help detect bias in predictions and help comply with GDPR compliance in these fields. QML leverages quantum phenomena such as entanglement and superposition, offering the potential for computational speedup and greater insights compared to classical ML. However, QML models also inherit the black-box nature of their classical counterparts, requiring the development of explainability techniques to be applied to these QML models to help understand why and how a particular output was generated.
This paper will explore the idea of creating a modular, explainable QML framework that splits QML algorithms into their core components, such as feature maps, variational circuits (ansatz), optimizers, kernels, and quantum-classical loops. Each component will be analyzed using explainability techniques, such as ALE and SHAP, which have been adapted to analyse the different components of these QML algorithms. By combining insights from these parts, the paper aims to infer explainability to the overall QML model. 

**Abstract (ZH)**: 可解释的机器学习算法旨在提供其决策过程的透明度和洞察力。在医疗保健和金融等领域，解释机器学习模型的预测过程至关重要，因为它有助于发现预测中的偏差，并有助于这些领域遵守GDPR合规要求。量子机器学习（QML）利用诸如纠缠和叠加等量子现象，有可能比经典机器学习提供更快的计算加速和更多的洞察力。然而，QML模型也继承了其经典 counterparts的黑盒性质，因此需要开发解释性技术，以便理解特定输出是如何生成的。

本文将探讨创建模块化的可解释量子机器学习框架的想法，该框架将QML算法分解为其核心组件，如特征映射、变分电路（Ansatz）、优化器、核函数和量子-经典循环。每种组件都将使用可解释性技术进行分析，如ALE和SHAP，这些技术已被改编以分析这些QML算法的不同组件。通过结合这些部分的见解，本文旨在推断整体QML模型的可解释性。 

---
# Optimized Spectral Fault Receptive Fields for Diagnosis-Informed Prognosis 

**Title (ZH)**: 优化的光谱故障感受野用于诊断驱动的预后 

**Authors**: Stan Muñoz Gutiérrez, Franz Wotawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12375)  

**Abstract**: This paper introduces Spectral Fault Receptive Fields (SFRFs), a biologically inspired technique for degradation state assessment in bearing fault diagnosis and remaining useful life (RUL) estimation. Drawing on the center-surround organization of retinal ganglion cell receptive fields, we propose a frequency-domain feature extraction algorithm that enhances the detection of fault signatures in vibration signals. SFRFs are designed as antagonistic spectral filters centered on characteristic fault frequencies, with inhibitory surrounds that enable robust characterization of incipient faults under variable operating conditions. A multi-objective evolutionary optimization strategy based on NSGA-II algorithm is employed to tune the receptive field parameters by simultaneously minimizing RUL prediction error, maximizing feature monotonicity, and promoting smooth degradation trajectories. The method is demonstrated on the XJTU-SY bearing run-to-failure dataset, confirming its suitability for constructing condition indicators in health monitoring applications. Key contributions include: (i) the introduction of SFRFs, inspired by the biology of vision in the primate retina; (ii) an evolutionary optimization framework guided by condition monitoring and prognosis criteria; and (iii) experimental evidence supporting the detection of early-stage faults and their precursors. Furthermore, we confirm that our diagnosis-informed spectral representation achieves accurate RUL prediction using a bagging regressor. The results highlight the interpretability and principled design of SFRFs, bridging signal processing, biological sensing principles, and data-driven prognostics in rotating machinery. 

**Abstract (ZH)**: 基于生物启发的谱故障感受野在轴承故障诊断和剩余使用寿命评估中的应用：一种降解状态评估和剩余使用寿命（RUL）估计的生物启发技术 

---
# HYPER: A Foundation Model for Inductive Link Prediction with Knowledge Hypergraphs 

**Title (ZH)**: HYPER：一种基于知识超图的归纳链接预测基础模型 

**Authors**: Xingyue Huang, Mikhail Galkin, Michael M. Bronstein, İsmail İlkan Ceylan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12362)  

**Abstract**: Inductive link prediction with knowledge hypergraphs is the task of predicting missing hyperedges involving completely novel entities (i.e., nodes unseen during training). Existing methods for inductive link prediction with knowledge hypergraphs assume a fixed relational vocabulary and, as a result, cannot generalize to knowledge hypergraphs with novel relation types (i.e., relations unseen during training). Inspired by knowledge graph foundation models, we propose HYPER as a foundation model for link prediction, which can generalize to any knowledge hypergraph, including novel entities and novel relations. Importantly, HYPER can learn and transfer across different relation types of varying arities, by encoding the entities of each hyperedge along with their respective positions in the hyperedge. To evaluate HYPER, we construct 16 new inductive datasets from existing knowledge hypergraphs, covering a diverse range of relation types of varying arities. Empirically, HYPER consistently outperforms all existing methods in both node-only and node-and-relation inductive settings, showing strong generalization to unseen, higher-arity relational structures. 

**Abstract (ZH)**: 基于知识超图的归纳链接预测任务是预测涉及完全全新的实体（即训练时未见过的节点）的缺失超边。现有的基于知识超图的归纳链接预测方法假设存在固定的关系词汇表，因此无法泛化到包含新关系类型（即训练时未见过的关系）的知识超图。受知识图谱基础模型的启发，我们提出了HYPER作为链接预测的基础模型，它可以泛化到任何知识超图，包括新的实体和新的关系。重要的是，HYPER可以通过编码每个超边中的实体及其在超边中的相对位置，来学习和迁移不同类型的不同元关系。为了评估HYPER，我们从现有的知识超图构建了16个新的归纳数据集，覆盖了不同类型和不同元数的关系。实验证明，HYPER在节点-only和节点-关系的归纳设置中都显著优于现有方法，显示出对未见过的高元数关系结构的强大泛化能力。 

---
# Theoretical Tensions in RLHF: Reconciling Empirical Success with Inconsistencies in Social Choice Theory 

**Title (ZH)**: RLHF中的理论紧张关系：调和实证成功与社会选择理论中的不一致性 

**Authors**: Jiancong Xiao, Zhekun Shi, Kaizhao Liu, Qi Long, Weijie J. Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.12350)  

**Abstract**: Despite its empirical success, Reinforcement Learning from Human Feedback (RLHF) has been shown to violate almost all the fundamental axioms in social choice theory -- such as majority consistency, pairwise majority consistency, and Condorcet consistency. This raises a foundational question: why does RLHF perform so well in practice if it fails these seemingly essential properties? In this paper, we resolve this paradox by showing that under mild and empirically plausible assumptions on the preference profile, RLHF does satisfy pairwise majority and Condorcet consistency. These assumptions are frequently satisfied in real-world alignment tasks, offering a theoretical explanation for RLHF's strong practical performance. Furthermore, we show that a slight modification to the reward modeling objective can ensure pairwise majority or Condorcet consistency even under general preference profiles, thereby improving the alignment process. Finally, we go beyond classical axioms in economic and social choice theory and introduce new alignment criteria -- preference matching, preference equivalence, and group preference matching -- that better reflect the goal of learning distributions over responses. We show that while RLHF satisfies the first two properties, it fails to satisfy the third. We conclude by discussing how future alignment methods may be designed to satisfy all three. 

**Abstract (ZH)**: 尽管强化学习从人类反馈中学习（RLHF）在实践中表现出色，但已被证明违反了社会选择理论中的几乎全部基本公理——如多数一致性、双边多数一致性及康德尔一致性。这引发了基础性问题：如果RLHF在这些看似至关重要的属性上失败了，那么它为何在实践中表现如此出色？本文通过在偏好配置图下提出温和且符合经验的假设，展示了RLHF实际上满足双边多数一致性和康德尔一致性。这些假设在实际对齐任务中经常被满足，为RLHF的优秀实践性能提供了理论解释。此外，我们证明通过对奖励建模目标进行 slight 修改，即便在一般的偏好配置图下也能确保双边多数一致性和康德尔一致性，从而改进对齐过程。最后，我们超越了经济学和社会选择理论中的经典公理，并引入了新的对齐标准——偏好匹配、偏好等价和群体偏好匹配，这些标准更符合学习响应分布的目标。我们发现尽管RLHF满足前两个属性，但未能满足第三个属性。我们讨论了未来对齐方法如何设计以满足所有三个属性。 

---
# GroupNL: Low-Resource and Robust CNN Design over Cloud and Device 

**Title (ZH)**: GroupNL: 云和设备上的低资源和 robust CNN 设计 

**Authors**: Chuntao Ding, Jianhang Xie, Junna Zhang, Salman Raza, Shangguang Wang, Jiannong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12335)  

**Abstract**: It has become mainstream to deploy Convolutional Neural Network (CNN) models on ubiquitous Internet of Things (IoT) devices with the help of the cloud to provide users with a variety of high-quality services. Most existing methods have two limitations: (i) low robustness in handling corrupted image data collected by IoT devices; and (ii) high consumption of computational and transmission resources. To this end, we propose the Grouped NonLinear transformation generation method (GroupNL), which generates diversified feature maps by utilizing data-agnostic Nonlinear Transformation Functions (NLFs) to improve the robustness of the CNN model. Specifically, partial convolution filters are designated as seed filters in a convolutional layer, and a small set of feature maps, i.e., seed feature maps, are first generated based on vanilla convolution operation. Then, we split seed feature maps into several groups, each with a set of different NLFs, to generate corresponding diverse feature maps with in-place nonlinear processing. Moreover, GroupNL effectively reduces the parameter transmission between multiple nodes during model training by setting the hyperparameters of NLFs to random initialization and not updating them during model training, and reduces the computing resources by using NLFs to generate feature maps instead of most feature maps generated based on sliding windows. Experimental results on CIFAR-10, GTSRB, CIFAR-10-C, Icons50, and ImageNet-1K datasets in NVIDIA RTX GPU platforms show that the proposed GroupNL outperforms other state-of-the-art methods in model robust and training acceleration. Specifically, on the Icons-50 dataset, the accuracy of GroupNL-ResNet-18 achieves approximately 2.86% higher than the vanilla ResNet-18. GroupNL improves training speed by about 53% compared to vanilla CNN when trained on a cluster of 8 NVIDIA RTX 4090 GPUs on the ImageNet-1K dataset. 

**Abstract (ZH)**: 利用非线性变换生成方法提高物联网设备上卷积神经网络模型的鲁棒性和训练加速 

---
# IndoorWorld: Integrating Physical Task Solving and Social Simulation in A Heterogeneous Multi-Agent Environment 

**Title (ZH)**: IndoorWorld: 在异构多agent环境中整合物理任务解决与社会仿真 

**Authors**: Dekun Wu, Frederik Brudy, Bang Liu, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12331)  

**Abstract**: Virtual environments are essential to AI agent research. Existing environments for LLM agent research typically focus on either physical task solving or social simulation, with the former oversimplifying agent individuality and social dynamics, and the latter lacking physical grounding of social behaviors. We introduce IndoorWorld, a heterogeneous multi-agent environment that tightly integrates physical and social dynamics. By introducing novel challenges for LLM-driven agents in orchestrating social dynamics to influence physical environments and anchoring social interactions within world states, IndoorWorld opens up possibilities of LLM-based building occupant simulation for architectural design. We demonstrate the potential with a series of experiments within an office setting to examine the impact of multi-agent collaboration, resource competition, and spatial layout on agent behavior. 

**Abstract (ZH)**: 虚拟环境对于AI代理研究至关重要。现有的LLM代理研究环境通常侧重于物理任务解决或社会模拟，前者过度简化了代理的个体性和社会动态，后者缺乏社会行为的物理基础。我们介绍了IndoorWorld，这是一个将物理和社会动态紧密结合的异构多代理环境。通过在IndoorWorld中引入新型挑战，使LLM驱动的代理能够协调社会动态以影响物理环境，并将社会互动锚定在世界状态中，为建筑设计中的基于LLM的建筑占用者模拟打开了可能性。我们通过一系列实验，在办公室环境中探讨了多代理协作、资源竞争和空间布局对代理行为的影响。 

---
# Three-dimensional Deep Shape Optimization with a Limited Dataset 

**Title (ZH)**: 有限数据集下的三维深度形状优化 

**Authors**: Yongmin Kwon, Namwoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12326)  

**Abstract**: Generative models have attracted considerable attention for their ability to produce novel shapes. However, their application in mechanical design remains constrained due to the limited size and variability of available datasets. This study proposes a deep learning-based optimization framework specifically tailored for shape optimization with limited datasets, leveraging positional encoding and a Lipschitz regularization term to robustly learn geometric characteristics and maintain a meaningful latent space. Through extensive experiments, the proposed approach demonstrates robustness, generalizability and effectiveness in addressing typical limitations of conventional optimization frameworks. The validity of the methodology is confirmed through multi-objective shape optimization experiments conducted on diverse three-dimensional datasets, including wheels and cars, highlighting the model's versatility in producing practical and high-quality design outcomes even under data-constrained conditions. 

**Abstract (ZH)**: 基于深度学习的有限数据集形状优化框架：利用位置编码和Lipschitz正则化项 robustly学习几何特征并保持有意义的隐空间 

---
# Machine Learning Methods for Small Data and Upstream Bioprocessing Applications: A Comprehensive Review 

**Title (ZH)**: 小型数据和上游生物处理应用的机器学习方法：综述 

**Authors**: Johnny Peng, Thanh Tung Khuat, Katarzyna Musial, Bogdan Gabrys  

**Link**: [PDF](https://arxiv.org/pdf/2506.12322)  

**Abstract**: Data is crucial for machine learning (ML) applications, yet acquiring large datasets can be costly and time-consuming, especially in complex, resource-intensive fields like biopharmaceuticals. A key process in this industry is upstream bioprocessing, where living cells are cultivated and optimised to produce therapeutic proteins and biologics. The intricate nature of these processes, combined with high resource demands, often limits data collection, resulting in smaller datasets. This comprehensive review explores ML methods designed to address the challenges posed by small data and classifies them into a taxonomy to guide practical applications. Furthermore, each method in the taxonomy was thoroughly analysed, with a detailed discussion of its core concepts and an evaluation of its effectiveness in tackling small data challenges, as demonstrated by application results in the upstream bioprocessing and other related domains. By analysing how these methods tackle small data challenges from different perspectives, this review provides actionable insights, identifies current research gaps, and offers guidance for leveraging ML in data-constrained environments. 

**Abstract (ZH)**: 基于小数据挑战的机器学习方法在上游生物制药过程中的应用综述 

---
# A Survey of Foundation Models for IoT: Taxonomy and Criteria-Based Analysis 

**Title (ZH)**: 物联网领域基础模型综述：分类与基于标准的分析 

**Authors**: Hui Wei, Dong Yoon Lee, Shubham Rohal, Zhizhang Hu, Shiwei Fang, Shijia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12263)  

**Abstract**: Foundation models have gained growing interest in the IoT domain due to their reduced reliance on labeled data and strong generalizability across tasks, which address key limitations of traditional machine learning approaches. However, most existing foundation model based methods are developed for specific IoT tasks, making it difficult to compare approaches across IoT domains and limiting guidance for applying them to new tasks. This survey aims to bridge this gap by providing a comprehensive overview of current methodologies and organizing them around four shared performance objectives by different domains: efficiency, context-awareness, safety, and security & privacy. For each objective, we review representative works, summarize commonly-used techniques and evaluation metrics. This objective-centric organization enables meaningful cross-domain comparisons and offers practical insights for selecting and designing foundation model based solutions for new IoT tasks. We conclude with key directions for future research to guide both practitioners and researchers in advancing the use of foundation models in IoT applications. 

**Abstract (ZH)**: 基础模型在物联网领域的兴趣逐渐增长，这得益于它们对标注数据的减少依赖及在不同任务上的强泛化能力，这解决了传统机器学习方法的关键局限性。然而，现有大多数基于基础模型的方法都是为特定的物联网任务开发的，这使得跨物联网领域比较方法变得困难，并限制了将这些方法应用于新任务的指导。本文综述旨在通过提供当前方法的全面概述，并围绕四个共享性能目标组织这些方法（由不同领域共享：效率、情境意识、安全性和安全与隐私），来弥补这一差距。对于每个目标，我们回顾代表性工作，总结常用技术和评估指标。这种目标中心化组织方式使得跨领域比较具有意义，并为选择和设计适用于新物联网任务的基础模型解决方案提供了实用洞察。最后，我们提出未来研究的关键方向，以指导实践者和研究人员在物联网应用中更广泛应用基础模型。 

---
# Datrics Text2SQL: A Framework for Natural Language to SQL Query Generation 

**Title (ZH)**: Datrics Text2SQL：一种自然语言到SQL查询生成的框架 

**Authors**: Tetiana Gladkykh, Kyrylo Kirykov  

**Link**: [PDF](https://arxiv.org/pdf/2506.12234)  

**Abstract**: Text-to-SQL systems enable users to query databases using natural language, democratizing access to data analytics. However, they face challenges in understanding ambiguous phrasing, domain-specific vocabulary, and complex schema relationships. This paper introduces Datrics Text2SQL, a Retrieval-Augmented Generation (RAG)-based framework designed to generate accurate SQL queries by leveraging structured documentation, example-based learning, and domain-specific rules. The system builds a rich Knowledge Base from database documentation and question-query examples, which are stored as vector embeddings and retrieved through semantic similarity. It then uses this context to generate syntactically correct and semantically aligned SQL code. The paper details the architecture, training methodology, and retrieval logic, highlighting how the system bridges the gap between user intent and database structure without requiring SQL expertise. 

**Abstract (ZH)**: Text-to-SQL系统使用户能够使用自然语言查询数据库， democratizing数据访问。然而，它们在理解含糊的语言表达、领域特定词汇以及复杂的数据模型关系方面面临挑战。本文介绍了基于检索增强生成（RAG）的Datrics Text2SQL框架，该框架通过利用结构化文档、基于示例的学习和领域特定规则生成准确的SQL查询。该系统从数据库文档和问题-查询示例中构建一个丰富的知识库，并将这些信息存储为向量嵌入并通过语义相似性进行检索。然后，该系统利用此上下文生成语义上正确且语义上对齐的SQL代码。本文描述了该系统的架构、训练方法和检索逻辑，强调了该系统如何在不需SQL专业知识的情况下弥合用户意图与数据库结构之间的差距。 

---
# Mapping Neural Theories of Consciousness onto the Common Model of Cognition 

**Title (ZH)**: 将意识的神经理论映射到共同认知模型上 

**Authors**: Paul S. Rosenbloom, John E. Laird, Christian Lebiere, Andrea Stocco  

**Link**: [PDF](https://arxiv.org/pdf/2506.12224)  

**Abstract**: A beginning is made at mapping four neural theories of consciousness onto the Common Model of Cognition. This highlights how the four jointly depend on recurrent local modules plus a cognitive cycle operating on a global working memory with complex states, and reveals how an existing integrative view of consciousness from a neural perspective aligns with the Com-mon Model. 

**Abstract (ZH)**: 开始将四种神经理论的意识映射到通用认知模型上，这突显了四种理论如何共同依赖于反复循环的局部模块以及在复杂的全球工作记忆上运行的认知循环，并揭示了从神经角度现有的意识整合观点与通用模型的一致性。 

---
# SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes 

**Title (ZH)**: SSLAM：通过音频混合增强自监督模型用于多声部声景 

**Authors**: Tony Alex, Sara Ahmed, Armin Mustafa, Muhammad Awais, Philip JB Jackson  

**Link**: [PDF](https://arxiv.org/pdf/2506.12222)  

**Abstract**: Self-supervised pre-trained audio networks have seen widespread adoption in real-world systems, particularly in multi-modal large language models. These networks are often employed in a frozen state, under the assumption that the SSL pre-training has sufficiently equipped them to handle real-world audio. However, a critical question remains: how well do these models actually perform in real-world conditions, where audio is typically polyphonic and complex, involving multiple overlapping sound sources? Current audio SSL methods are often benchmarked on datasets predominantly featuring monophonic audio, such as environmental sounds, and speech. As a result, the ability of SSL models to generalize to polyphonic audio, a common characteristic in natural scenarios, remains underexplored. This limitation raises concerns about the practical robustness of SSL models in more realistic audio settings. To address this gap, we introduce Self-Supervised Learning from Audio Mixtures (SSLAM), a novel direction in audio SSL research, designed to improve, designed to improve the model's ability to learn from polyphonic data while maintaining strong performance on monophonic data. We thoroughly evaluate SSLAM on standard audio SSL benchmark datasets which are predominantly monophonic and conduct a comprehensive comparative analysis against SOTA methods using a range of high-quality, publicly available polyphonic datasets. SSLAM not only improves model performance on polyphonic audio, but also maintains or exceeds performance on standard audio SSL benchmarks. Notably, it achieves up to a 3.9\% improvement on the AudioSet-2M (AS-2M), reaching a mean average precision (mAP) of 50.2. For polyphonic datasets, SSLAM sets new SOTA in both linear evaluation and fine-tuning regimes with performance improvements of up to 9.1\% (mAP). 

**Abstract (ZH)**: 自监督预训练音频网络在实际系统中得到了广泛应用，特别是在多模态大型语言模型中。这些网络通常以冻结状态使用，在假设它们通过自监督预训练已经获得了处理真实世界音频的能力的前提下。然而，一个关键问题仍然存在：在音频通常为多音性和复杂性的实际条件下，这些模型实际表现如何？当前的音频自监督学习方法通常在以单音性音频为主的环境中进行基准测试，例如环境声和语音。因此，SSL模型在多音性音频上的泛化能力，这是自然场景中的一个常见特性，仍被严重忽视。这一限制引发了对SSL模型在更实际音频设置中的实用鲁棒性的担忧。为解决这一问题，我们提出了自监督学习从音频混合物（SSLAM）这一在音频自监督学习研究中的新方向，旨在提高模型从多音性数据中学习的能力，同时保持在单音性数据上的强大性能。我们彻底评估了SSLAM在以单音性为主的标准音频自监督学习基准数据集上的表现，并使用一系列高质量的公开多音性数据集与当前最先进的方法进行了全面的对比分析。SSLAM不仅在多音性音频上提高了模型性能，而且在标准音频自监督学习基准测试上也保持或超过了性能。值得注意的是，它在AudioSet-2M（AS-2M）上达到了50.2的平均精度（mAP），提高了多达3.9％。在多音性数据集上，SSLAM在线性评估和微调两种模式下均达到新的当前最好水平，性能提高高达9.1％（mAP）。 

---
# BreastDCEDL: Curating a Comprehensive DCE-MRI Dataset and developing a Transformer Implementation for Breast Cancer Treatment Response Prediction 

**Title (ZH)**: BreastDCEDL: 编纂全面的DCE-MRI数据集并开发 Transformer 实现方法以预测乳腺癌治疗反应 

**Authors**: Naomi Fridman, Bubby Solway, Tomer Fridman, Itamar Barnea, Anat Goldshtein  

**Link**: [PDF](https://arxiv.org/pdf/2506.12190)  

**Abstract**: Breast cancer remains a leading cause of cancer-related mortality worldwide, making early detection and accurate treatment response monitoring critical priorities. We present BreastDCEDL, a curated, deep learning-ready dataset comprising pre-treatment 3D Dynamic Contrast-Enhanced MRI (DCE-MRI) scans from 2,070 breast cancer patients drawn from the I-SPY1, I-SPY2, and Duke cohorts, all sourced from The Cancer Imaging Archive. The raw DICOM imaging data were rigorously converted into standardized 3D NIfTI volumes with preserved signal integrity, accompanied by unified tumor annotations and harmonized clinical metadata including pathologic complete response (pCR), hormone receptor (HR), and HER2 status. Although DCE-MRI provides essential diagnostic information and deep learning offers tremendous potential for analyzing such complex data, progress has been limited by lack of accessible, public, multicenter datasets. BreastDCEDL addresses this gap by enabling development of advanced models, including state-of-the-art transformer architectures that require substantial training data. To demonstrate its capacity for robust modeling, we developed the first transformer-based model for breast DCE-MRI, leveraging Vision Transformer (ViT) architecture trained on RGB-fused images from three contrast phases (pre-contrast, early post-contrast, and late post-contrast). Our ViT model achieved state-of-the-art pCR prediction performance in HR+/HER2- patients (AUC 0.94, accuracy 0.93). BreastDCEDL includes predefined benchmark splits, offering a framework for reproducible research and enabling clinically meaningful modeling in breast cancer imaging. 

**Abstract (ZH)**: 乳腺癌仍然是导致癌症相关死亡的主要原因，早期检测和准确的治疗反应监测至关重要。我们介绍了BreastDCEDL，这是一个精心整理、适合深度学习的数据集，包含来自I-SPY1、I-SPY2和Duke队列的2,070名乳腺癌患者的治疗前3D动态对比增强磁共振成像（DCE-MRI）扫描，所有数据来源于《癌症影像档案》。原始DICOM影像数据被严格转化为标准的3D NIfTI体积，保持了信号完整性，并附带统一的肿瘤标注和 harmonized 临床元数据，包括病理性完全缓解（pCR）、雌激素受体（HR）和HER2状态。尽管DCE-MRI提供了重要的诊断信息，且深度学习对于分析这种复杂数据具有巨大的潜力，但进展受限于缺乏可访问的、公开的多中心数据集。BreastDCEDL通过填补这一空白，使得开发高级模型成为可能，包括需要大量训练数据的最先进的变压器架构。为了展示其在稳健建模方面的潜力，我们开发了首个基于变压器的模型用于乳腺DCE-MRI，利用在三个对比阶段（预对比、早期对比后和晚期对比后）融合RGB图像的Vision Transformer（ViT）架构进行训练。我们的ViT模型在HR+/HER2-患者中实现了最新的pCR预测性能（AUC 0.94，准确率 0.93）。BreastDCEDL包括预定义的基准拆分，提供了一种可再现的研究框架，并促进了乳腺癌影像学中的临床相关模型构建。 

---
# MRI-CORE: A Foundation Model for Magnetic Resonance Imaging 

**Title (ZH)**: MRI-CORE: 一个磁共振成像基础模型 

**Authors**: Haoyu Dong, Yuwen Chen, Hanxue Gu, Nicholas Konz, Yaqian Chen, Qihang Li, Maciej A. Mazurowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.12186)  

**Abstract**: The widespread use of Magnetic Resonance Imaging (MRI) and the rise of deep learning have enabled the development of powerful predictive models for a wide range of diagnostic tasks in MRI, such as image classification or object segmentation. However, training models for specific new tasks often requires large amounts of labeled data, which is difficult to obtain due to high annotation costs and data privacy concerns. To circumvent this issue, we introduce MRI-CORE (MRI COmprehensive Representation Encoder), a vision foundation model pre-trained using more than 6 million slices from over 110,000 MRI volumes across 18 main body locations. Experiments on five diverse object segmentation tasks in MRI demonstrate that MRI-CORE can significantly improve segmentation performance in realistic scenarios with limited labeled data availability, achieving an average gain of 6.97% 3D Dice Coefficient using only 10 annotated slices per task. We further demonstrate new model capabilities in MRI such as classification of image properties including body location, sequence type and institution, and zero-shot segmentation. These results highlight the value of MRI-CORE as a generalist vision foundation model for MRI, potentially lowering the data annotation resource barriers for many applications. 

**Abstract (ZH)**: 磁共振成像广泛应用于医学领域，深度学习的兴起促进了基于磁共振成像的 Powerful 预测模型的发展，这些模型可用于图像分类或对象分割等多种诊断任务。然而，为特定新任务训练模型通常需要大量标注数据，这由于标注成本高和数据隐私问题而难以获得。为解决这一问题，我们引入了 MRI-CORE（MRI 综合表示编码器），该模型基于超过 110,000 个来自 18 个主要身体部位的磁共振成像体积中的 600 万多切片进行预训练。在五个不同对象分割任务上的实验表明，MRI-CORE 在有限标注数据的情况下，可以显著提高分割性能，使用每任务仅 10 个标注切片，平均获得 6.97% 的三维 Dice 系数提升。进一步展示了 MRI-CORE 在 MRI 中的新模型能力，包括图像属性（包括身体位置、序列类型和机构）分类和零样本分割。这些结果突显了 MRI-CORE 作为 MRI 通用视觉基础模型的价值，可能降低许多应用的数据标注资源障碍。 

---
# TCN-DPD: Parameter-Efficient Temporal Convolutional Networks for Wideband Digital Predistortion 

**Title (ZH)**: TCN-DPD: 参数高效时序卷积网络在宽带数字预失真中的应用 

**Authors**: Huanqiang Duan, Manno Versluis, Qinyu Chen, Leo C. N. de Vreede, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12165)  

**Abstract**: Digital predistortion (DPD) is essential for mitigating nonlinearity in RF power amplifiers, particularly for wideband applications. This paper presents TCN-DPD, a parameter-efficient architecture based on temporal convolutional networks, integrating noncausal dilated convolutions with optimized activation functions. Evaluated on the OpenDPD framework with the DPA_200MHz dataset, TCN-DPD achieves simulated ACPRs of -51.58/-49.26 dBc (L/R), EVM of -47.52 dB, and NMSE of -44.61 dB with 500 parameters and maintains superior linearization than prior models down to 200 parameters, making it promising for efficient wideband PA linearization. 

**Abstract (ZH)**: 基于时空卷积网络的非因袭扩张卷积与优化激活函数结合的数字预失真（TCN-DPD） 

---
# Scale-Invariance Drives Convergence in AI and Brain Representations 

**Title (ZH)**: 尺度不变性驱动AI和脑部表征的收敛 

**Authors**: Junjie Yu, Wenxiao Ma, Jianyu Zhang, Haotian Deng, Zihan Deng, Yi Guo, Quanying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12117)  

**Abstract**: Despite variations in architecture and pretraining strategies, recent studies indicate that large-scale AI models often converge toward similar internal representations that also align with neural activity. We propose that scale-invariance, a fundamental structural principle in natural systems, is a key driver of this convergence. In this work, we propose a multi-scale analytical framework to quantify two core aspects of scale-invariance in AI representations: dimensional stability and structural similarity across scales. We further investigate whether these properties can predict alignment performance with functional Magnetic Resonance Imaging (fMRI) responses in the visual cortex. Our analysis reveals that embeddings with more consistent dimension and higher structural similarity across scales align better with fMRI data. Furthermore, we find that the manifold structure of fMRI data is more concentrated, with most features dissipating at smaller scales. Embeddings with similar scale patterns align more closely with fMRI data. We also show that larger pretraining datasets and the inclusion of language modalities enhance the scale-invariance properties of embeddings, further improving neural alignment. Our findings indicate that scale-invariance is a fundamental structural principle that bridges artificial and biological representations, providing a new framework for evaluating the structural quality of human-like AI systems. 

**Abstract (ZH)**: 尽管架构和预训练策略存在差异，近期研究表明，大规模AI模型往往朝向相似的内部表示汇聚，这些表示也与神经活动相一致。我们提出，规模不变性，作为自然系统的基本结构性原理，是这一汇聚现象的关键驱动因素。在本文中，我们提出一个多尺度分析框架来量化AI表示中尺度不变性的两个核心方面：维度稳定性和跨尺度的结构相似性。进一步研究这些性质能否预测与功能性磁共振成像(fMRI)视觉皮层反应的对齐性能。分析结果显示，维度更一致且跨尺度结构相似性更高的嵌入更符合fMRI数据。此外，我们发现fMRI数据的流形结构更为集中，大多数特征在较小尺度上消失。具有相似尺度模式的嵌入更接近fMRI数据。我们还表明，使用更大的预训练数据集和包含语言模态能够增强嵌入的尺度不变性特性，进一步改善神经对齐。我们的研究结果表明，尺度不变性作为连接人工与生物表示的基本结构性原理，提供了评估类人AI系统结构质量的新框架。 

---
# Unsupervised Document and Template Clustering using Multimodal Embeddings 

**Title (ZH)**: 基于多模态嵌入的无监督文档和模板聚类 

**Authors**: Phillipe R. Sampaio, Helene Maxcici  

**Link**: [PDF](https://arxiv.org/pdf/2506.12116)  

**Abstract**: This paper investigates a novel approach to unsupervised document clustering by leveraging multimodal embeddings as input to traditional clustering algorithms such as $k$-Means and DBSCAN. Our method aims to achieve a finer-grained document understanding by not only grouping documents at the type level (e.g., invoices, purchase orders), but also distinguishing between different templates within the same document category. This is achieved by using embeddings that capture textual content, layout information, and visual features of documents. We evaluated the effectiveness of this approach using embeddings generated by several state-of-the-art pretrained multimodal models, including SBERT, LayoutLMv1, LayoutLMv3, DiT, Donut, and ColPali. Our findings demonstrate the potential of multimodal embeddings to significantly enhance document clustering, offering benefits for various applications in intelligent document processing, document layout analysis, and unsupervised document classification. This work provides valuable insight into the advantages and limitations of different multimodal models for this task and opens new avenues for future research to understand and organize document collections. 

**Abstract (ZH)**: 本文通过利用多模态嵌入作为传统聚类算法（如$k$-Means和DBSCAN）的输入，探讨了一种新颖的无监督文档聚类方法。该方法旨在通过不仅按类型（如发票、采购订单）分组文档，而且还区分同一类别文档内的不同模板，实现更细粒度的文档理解。通过使用能够捕捉文档文本内容、布局信息和视觉特征的嵌入，实现了这一目标。我们使用几种最新的预训练多模态模型（包括SBERT、LayoutLMv1、LayoutLMv3、DiT、Donut和ColPali）生成的嵌入对这种方法的有效性进行了评估。研究结果显示出多模态嵌入在文档聚类方面的潜在优势，为智能文档处理、文档布局分析和无监督文档分类提供了益处。本文为不同多模态模型在该任务中的优势和局限性提供了宝贵的见解，并为未来研究理解和组织文档集合开辟了新的途径。 

---
# Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs): A Feynman-Based Architecture for Continuous Learning Over Streaming Data 

**Title (ZH)**: 基于费曼原理的量子启发可微积分神经网络（QIDINNs）：一种适用于流式数据连续学习的架构 

**Authors**: Oscar Boullosa Dapena  

**Link**: [PDF](https://arxiv.org/pdf/2506.12111)  

**Abstract**: Real-time continuous learning over streaming data remains a central challenge in deep learning and AI systems. Traditional gradient-based models such as backpropagation through time (BPTT) face computational and stability limitations when dealing with temporally unbounded data. In this paper, we introduce a novel architecture, Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs), which leverages the Feynman technique of differentiation under the integral sign to formulate neural updates as integrals over historical data. This reformulation allows for smoother, more stable learning dynamics that are both physically interpretable and computationally tractable. Inspired by Feynman's path integral formalism and compatible with quantum gradient estimation frameworks, QIDINNs open a path toward hybrid classical-quantum neural computation. We demonstrate our model's effectiveness on synthetic and real-world streaming tasks, and we propose directions for quantum extensions and scalable implementations. 

**Abstract (ZH)**: 实时处理流式数据的连续学习 remains a central challenge in deep learning and AI systems. Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs) leverage the Feynman technique to reformulate neural updates as integrals over historical data, enabling smoother and more stable learning dynamics that are both physically interpretable and computationally tractable. QIDINNs, inspired by Feynman's path integral formalism and compatible with quantum gradient estimation frameworks, pave the way for hybrid classical-quantum neural computation. We demonstrate the model's effectiveness on synthetic and real-world streaming tasks and propose directions for quantum extensions and scalable implementations. 

---
# EconGym: A Scalable AI Testbed with Diverse Economic Tasks 

**Title (ZH)**: EconGym: 一个具备多样化经济任务的可扩展AI试验台 

**Authors**: Qirui Mi, Qipeng Yang, Zijun Fan, Wentian Fan, Heyang Ma, Chengdong Ma, Siyu Xia, Bo An, Jun Wang, Haifeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12110)  

**Abstract**: Artificial intelligence (AI) has become a powerful tool for economic research, enabling large-scale simulation and policy optimization. However, applying AI effectively requires simulation platforms for scalable training and evaluation-yet existing environments remain limited to simplified, narrowly scoped tasks, falling short of capturing complex economic challenges such as demographic shifts, multi-government coordination, and large-scale agent interactions. To address this gap, we introduce EconGym, a scalable and modular testbed that connects diverse economic tasks with AI algorithms. Grounded in rigorous economic modeling, EconGym implements 11 heterogeneous role types (e.g., households, firms, banks, governments), their interaction mechanisms, and agent models with well-defined observations, actions, and rewards. Users can flexibly compose economic roles with diverse agent algorithms to simulate rich multi-agent trajectories across 25+ economic tasks for AI-driven policy learning and analysis. Experiments show that EconGym supports diverse and cross-domain tasks-such as coordinating fiscal, pension, and monetary policies-and enables benchmarking across AI, economic methods, and hybrids. Results indicate that richer task composition and algorithm diversity expand the policy space, while AI agents guided by classical economic methods perform best in complex settings. EconGym also scales to 10k agents with high realism and efficiency. 

**Abstract (ZH)**: 人工智能（AI）已成为经济研究的强大工具，使其能够进行大规模模拟和政策优化。然而，有效应用AI需要可扩展的训练和评估仿真平台——但现有的环境仍然局限于简化和狭隘范围的任务，难以捕捉到人口结构变化、多政府协调和大规模代理互动等复杂的经济挑战。为解决这一问题，我们引入了EconGym，这是一种可扩展且模块化的测试平台，将多样化的经济任务与AI算法相连接。EconGym基于严格的经济建模，实现了11种异质性角色类型（如家庭、企业、银行、政府）、其交互机制以及具有明确观测、行为和奖励的代理模型。用户可以灵活地组合具有各种代理算法的经济角色，以模拟跨越25多项以上经济任务的丰富多代理轨迹，用于AI驱动的政策学习和分析。实验表明，EconGym支持多领域任务，如协调财政、养老金和货币政策，并实现了AI方法、经济方法和混合方法之间的基准测试。结果显示，更丰富的任务组合和算法多样性扩展了政策空间，而受经典经济方法指导的AI代理在复杂环境中表现最佳。EconGym还可扩展至10,000个代理，具有高真实性和高效性。 

---
# A Lightweight IDS for Early APT Detection Using a Novel Feature Selection Method 

**Title (ZH)**: 基于新型特征选择方法的轻量级IDS用于早期APT检测 

**Authors**: Bassam Noori Shaker, Bahaa Al-Musawi, Mohammed Falih Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12108)  

**Abstract**: An Advanced Persistent Threat (APT) is a multistage, highly sophisticated, and covert form of cyber threat that gains unauthorized access to networks to either steal valuable data or disrupt the targeted network. These threats often remain undetected for extended periods, emphasizing the critical need for early detection in networks to mitigate potential APT consequences. In this work, we propose a feature selection method for developing a lightweight intrusion detection system capable of effectively identifying APTs at the initial compromise stage. Our approach leverages the XGBoost algorithm and Explainable Artificial Intelligence (XAI), specifically utilizing the SHAP (SHapley Additive exPlanations) method for identifying the most relevant features of the initial compromise stage. The results of our proposed method showed the ability to reduce the selected features of the SCVIC-APT-2021 dataset from 77 to just four while maintaining consistent evaluation metrics for the suggested system. The estimated metrics values are 97% precision, 100% recall, and a 98% F1 score. The proposed method not only aids in preventing successful APT consequences but also enhances understanding of APT behavior at early stages. 

**Abstract (ZH)**: 一种高级持续性威胁（APT）是一种多阶段、高度复杂且隐蔽的网络威胁，其通过未经授权的方式进入网络，以窃取有价值的数据或破坏目标网络。这些威胁往往会长期未被检测到，这突显了在网络中早期检测以减轻潜在APT后果的重要性。本文提出了一种特征选择方法，用于开发一种轻量级的入侵检测系统，该系统能够在初始 compromise 阶段有效识别 APT。我们的方法利用了 XGBoost 算法和可解释人工智能（XAI），具体使用 SHAP（SHapley Additive exPlanations）方法来识别初始 compromise 阶段的最相关特征。我们提出的方法将 SCVIC-APT-2021 数据集的选定特征从 77 个减少到仅 4 个，同时保持评价指标的一致性。估计的指标值为 97% 的精确率、100% 的召回率和 98% 的 F1 分数。所提出的方法不仅能帮助预防成功的 APT 后果，还能增强对 APT 行为早期阶段的理解。 

---
# "I Hadn't Thought About That": Creators of Human-like AI Weigh in on Ethics And Neurodivergence 

**Title (ZH)**: “我还没有考虑过这一点”：人类似AI的创作者谈伦理与神经多样xing 

**Authors**: Naba Rizvi, Taggert Smith, Tanvi Vidyala, Mya Bolds, Harper Strickland, Andrew Begel, Rua Williams, Imani Munyaka  

**Link**: [PDF](https://arxiv.org/pdf/2506.12098)  

**Abstract**: Human-like AI agents such as robots and chatbots are becoming increasingly popular, but they present a variety of ethical concerns. The first concern is in how we define humanness, and how our definition impacts communities historically dehumanized by scientific research. Autistic people in particular have been dehumanized by being compared to robots, making it even more important to ensure this marginalization is not reproduced by AI that may promote neuronormative social behaviors. Second, the ubiquitous use of these agents raises concerns surrounding model biases and accessibility. In our work, we investigate the experiences of the people who build and design these technologies to gain insights into their understanding and acceptance of neurodivergence, and the challenges in making their work more accessible to users with diverse needs. Even though neurodivergent individuals are often marginalized for their unique communication styles, nearly all participants overlooked the conclusions their end-users and other AI system makers may draw about communication norms from the implementation and interpretation of humanness applied in participants' work. This highlights a major gap in their broader ethical considerations, compounded by some participants' neuronormative assumptions about the behaviors and traits that distinguish "humans" from "bots" and the replication of these assumptions in their work. We examine the impact this may have on autism inclusion in society and provide recommendations for additional systemic changes towards more ethical research directions. 

**Abstract (ZH)**: 具有人类特征的AI代理，如机器人和聊天机器人正变得越来越流行，但它们提出了各种伦理关切。首关切点是我们在如何定义人类以及这种定义如何影响历史上被科学研究剥夺人性的社群方面存在的问题。特别是自闭症人士因其被与机器人相提并论而被剥夺人性，这使得确保AI不会进一步边缘化他们变得尤为重要，一些AI可能会促进以神经正常行为为准则的社会行为。其次，这些代理的普遍使用引发了模型偏见和可访问性方面的担忧。在我们的研究中，我们调查了构建和设计这些技术的人们的体验，以了解他们对神经多样性及其工作向具有不同需求的用户群体的可访问性理解与接受程度。尽管神经多样性个体常因独特的交流方式而被边缘化，但几乎所有参与者都忽视了其最终用户和其他AI系统制作者从参与者的工作中实施和解释的人性化结论中得出的关于交流规范的推论。这突显了他们在更广泛伦理考量中的重大缺口，同时也被一些参与者对辨别“人”与“机器人”的行为和特质的神经正常假设所加剧，并在他们的工作中复刻这些假设。我们探讨了这可能对自闭症在社会中的包容性产生的影响，并提供了朝着更道德研究方向的额外系统性变革建议。 

---
# Military AI Cyber Agents (MAICAs) Constitute a Global Threat to Critical Infrastructure 

**Title (ZH)**: 军用人工智能网络代理构成对关键基础设施的全球性威胁 

**Authors**: Timothy Dubber, Seth Lazar  

**Link**: [PDF](https://arxiv.org/pdf/2506.12094)  

**Abstract**: This paper argues that autonomous AI cyber-weapons - Military-AI Cyber Agents (MAICAs) - create a credible pathway to catastrophic risk. It sets out the technical feasibility of MAICAs, explains why geopolitics and the nature of cyberspace make MAICAs a catastrophic risk, and proposes political, defensive-AI and analogue-resilience measures to blunt the threat. 

**Abstract (ZH)**: 本文 argues that自主人工智能网络武器——军事人工智能网络代理（MAICAs）——构成了灾难性风险的可信途径。它阐述了MAICAs的技术可行性，解释了地缘政治和网络空间的特性为何使MAICAs成为灾难性风险，并提出政治、防御性人工智能以及类比韧性措施以缓解这一威胁。 

---
# Efficient Parallel Training Methods for Spiking Neural Networks with Constant Time Complexity 

**Title (ZH)**: 高效常时间复杂度并行训练方法用于脉冲神经网络 

**Authors**: Wanjin Feng, Xingyu Gao, Wenqian Du, Hailong Shi, Peilin Zhao, Pengcheng Wu, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12087)  

**Abstract**: Spiking Neural Networks (SNNs) often suffer from high time complexity $O(T)$ due to the sequential processing of $T$ spikes, making training computationally expensive.
In this paper, we propose a novel Fixed-point Parallel Training (FPT) method to accelerate SNN training without modifying the network architecture or introducing additional assumptions.
FPT reduces the time complexity to $O(K)$, where $K$ is a small constant (usually $K=3$), by using a fixed-point iteration form of Leaky Integrate-and-Fire (LIF) neurons for all $T$ timesteps.
We provide a theoretical convergence analysis of FPT and demonstrate that existing parallel spiking neurons can be viewed as special cases of our proposed method.
Experimental results show that FPT effectively simulates the dynamics of original LIF neurons, significantly reducing computational time without sacrificing accuracy.
This makes FPT a scalable and efficient solution for real-world applications, particularly for long-term tasks.
Our code will be released at \href{this https URL}{\texttt{this https URL}}. 

**Abstract (ZH)**: Spiking神经网络（SNN）常常由于需要依次处理T个 spikes而导致较高的时间复杂度$O(T)$，从而使训练计算成本高昂。
本文提出了一种新颖的定点并行训练（FPT）方法，可以在不修改网络架构或引入额外假设的情况下加速SNN训练。
FPT通过使用定点迭代形式的Leaky Integrate-and-Fire（LIF）神经元将时间复杂度降低到$O(K)$，其中$K$是一个较小的常数（通常$K=3$）。
我们提供了FPT的理论收敛分析，并展示了现有并行SNN可以被视为我们提出方法的特例。
实验结果表明，FPT能够有效地模拟原始LIF神经元的动力学特性，显著减少了计算时间而不会牺牲准确度。
这使得FPT成为实时应用中的一个可扩展且高效的解决方案，尤其是在长期任务方面。
我们的代码将在\href{this https URL}{https://this https URL}发布。 

---
# Wanting to Be Understood Explains the Meta-Problem of Consciousness 

**Title (ZH)**: 渴望被理解解释了意识的元问题 

**Authors**: Chrisantha Fernando, Dylan Banarse, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2506.12086)  

**Abstract**: Because we are highly motivated to be understood, we created public external representations -- mime, language, art -- to externalise our inner states. We argue that such external representations are a pre-condition for access consciousness, the global availability of information for reasoning. Yet the bandwidth of access consciousness is tiny compared with the richness of `raw experience', so no external representation can reproduce that richness in full. Ordinarily an explanation of experience need only let an audience `grasp' the relevant pattern, not relive the phenomenon. But our drive to be understood, and our low level sensorimotor capacities for `grasping' so rich, that the demand for an explanation of the feel of experience cannot be ``satisfactory''. That inflated epistemic demand (the preeminence of our expectation that we could be perfectly understood by another or ourselves) rather than an irreducible metaphysical gulf -- keeps the hard problem of consciousness alive. But on the plus side, it seems we will simply never give up creating new ways to communicate and think about our experiences. In this view, to be consciously aware is to strive to have one's agency understood by oneself and others. 

**Abstract (ZH)**: 由于我们有强烈的被理解动机，创造了公共外部表征——模仿、语言、艺术——来外部化我们的内心状态。我们认为，这种外部表征是获得性意识的先决条件，即为推理提供全球可用的信息。然而，获得性意识的信息带宽相比于“原始经验”的丰富性要小得多，因此没有外部表征能够完全再现这种丰富性。通常，对体验的解释只需要让观众“抓住”相关模式，而不必重新体验这一现象。但由于我们强烈的被理解动机，加之低层次的感觉运动能力难以“抓住”如此丰富的体验，因此对体验“感觉”的解释需求无法得到“满意”的满足。正是这种膨胀的知识需求（我们对能够被他人或自己完全理解的预期占主导地位）而非不可缩减的本体论鸿沟——使意识的难题保持鲜活。但另一方面，看来我们永远不会停止创造新的沟通和思考体验的方式。在这种观点中，拥有自觉意识就是努力使自己的行动被自己和他人理解。 

---
# The CAISAR Platform: Extending the Reach of Machine Learning Specification and Verification 

**Title (ZH)**: CAISAR平台：扩展机器学习规范与验证的范围 

**Authors**: Michele Alberti, François Bobot, Julien Girard-Satabin, Alban Grastien, Aymeric Varasse, Zakaria Chihani  

**Link**: [PDF](https://arxiv.org/pdf/2506.12084)  

**Abstract**: The formal specification and verification of machine learning programs saw remarkable progress in less than a decade, leading to a profusion of tools. However, diversity may lead to fragmentation, resulting in tools that are difficult to compare, except for very specific benchmarks. Furthermore, this progress is heavily geared towards the specification and verification of a certain class of property, that is, local robustness properties. But while provers are becoming more and more efficient at solving local robustness properties, even slightly more complex properties, involving multiple neural networks for example, cannot be expressed in the input languages of winners of the International Competition of Verification of Neural Networks VNN-Comp. In this tool paper, we present CAISAR, an open-source platform dedicated to machine learning specification and verification. We present its specification language, suitable for modelling complex properties on neural networks, support vector machines and boosted trees. We show on concrete use-cases how specifications written in this language are automatically translated to queries to state-of-the-art provers, notably by using automated graph editing techniques, making it possible to use their off-the-shelf versions. The artifact to reproduce the paper claims is available at the following DOI: this https URL 

**Abstract (ZH)**: 机器学习程序的形式化规范与验证在过去十年取得了显著进展，导致出现众多工具。然而，多样性可能导致碎片化，使得除了特定基准之外，这些工具难以比较。此外，这些进展主要集中在特定类别的属性，即局部鲁棒性属性上。尽管证明器在解决局部鲁棒性问题上正变得越来越高效，但稍微复杂的属性，例如涉及多个神经网络，无法在国际神经网络验证竞赛VNN-Comp获胜工具的输入语言中表达。在本工具论文中，我们介绍了CAISAR，一个开源平台，专注于机器学习的规范与验证。我们展示了其规范语言，适用于模型神经网络、支持向量机和提升树的复杂属性。我们通过使用自动化图编辑技术展示了如何在具体用例中自动将该语言编写的规范转换为最先进的证明器的查询，使其能够使用现成版本。用于复制论文中声明的成果可在以下DOI获取：[this https URL]。 

---
# Latency Optimization for Wireless Federated Learning in Multihop Networks 

**Title (ZH)**: 多跳网络中无线联邦学习的时延优化 

**Authors**: Shaba Shaon, Van-Dinh Nguyen, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12081)  

**Abstract**: In this paper, we study a novel latency minimization problem in wireless federated learning (FL) across multi-hop networks. The system comprises multiple routes, each integrating leaf and relay nodes for FL model training. We explore a personalized learning and adaptive aggregation-aware FL (PAFL) framework that effectively addresses data heterogeneity across participating nodes by harmonizing individual and collective learning objectives. We formulate an optimization problem aimed at minimizing system latency through the joint optimization of leaf and relay nodes, as well as relay routing indicator. We also incorporate an additional energy harvesting scheme for the relay nodes to help with their relay tasks. This formulation presents a computationally demanding challenge, and thus we develop a simple yet efficient algorithm based on block coordinate descent and successive convex approximation (SCA) techniques. Simulation results illustrate the efficacy of our proposed joint optimization approach for leaf and relay nodes with relay routing indicator. We observe significant latency savings in the wireless multi-hop PAFL system, with reductions of up to 69.37% compared to schemes optimizing only one node type, traditional greedy algorithm, and scheme without relay routing indicator. 

**Abstract (ZH)**: 在多跳网络中无线联邦学习中的新型延迟最小化问题研究 

---
# A Synthetic Pseudo-Autoencoder Invites Examination of Tacit Assumptions in Neural Network Design 

**Title (ZH)**: 合成伪自动编码器促使神经网络设计中的隐含假设审视 

**Authors**: Assaf Marron  

**Link**: [PDF](https://arxiv.org/pdf/2506.12076)  

**Abstract**: We present a handcrafted neural network that, without training, solves the seemingly difficult problem of encoding an arbitrary set of integers into a single numerical variable, and then recovering the original elements. While using only standard neural network operations -- weighted sums with biases and identity activation -- we make design choices that challenge common notions in this area around representation, continuity of domains, computation, learnability and more. For example, our construction is designed, not learned; it represents multiple values using a single one by simply concatenating digits without compression, and it relies on hardware-level truncation of rightmost digits as a bit-manipulation mechanism. This neural net is not intended for practical application. Instead, we see its resemblance to -- and deviation from -- standard trained autoencoders as an invitation to examine assumptions that may unnecessarily constrain the development of systems and models based on autoencoding and machine learning. Motivated in part by our research on a theory of biological evolution centered around natural autoencoding of species characteristics, we conclude by refining the discussion with a biological perspective. 

**Abstract (ZH)**: 我们呈现了一个手工构建的神经网络，无需训练即可解决将任意整数集编码为单个数值变量并恢复原始元素这一看似困难的问题。我们仅使用标准神经网络操作（加权求和、偏置和恒等激活函数）进行设计，挑战了这一领域关于表示、域的连续性、计算和可学习性等方面的常见观念。例如，我们的构建方式是设计而非学习的；通过简单地串联数字而不进行压缩来表示多个值，并依赖于硬件层面的最右侧数字截断作为一种位操作机制。该神经网络并非旨在实际应用。相反，我们认为它与标准训练自编码器的相似性和差异可以作为一种邀请，促使我们审视那些可能无必要地限制自编码和机器学习系统与模型发展的假设。受到我们关于以自然自编码为中心的生物演化理论研究的启发，我们从生物学角度进一步细化了讨论。 

---
# T-TExTS (Teaching Text Expansion for Teacher Scaffolding): Enhancing Text Selection in High School Literature through Knowledge Graph-Based Recommendation 

**Title (ZH)**: T-TExTS (教学文本扩展以支持教师支架教学): 基于知识图谱的推荐增强高中文学文本选择 

**Authors**: Nirmal Gelal, Chloe Snow, Ambyr Rios, Hande Küçük McGinty  

**Link**: [PDF](https://arxiv.org/pdf/2506.12075)  

**Abstract**: The implementation of transformational pedagogy in secondary education classrooms requires a broad multiliteracy approach. Due to limited planning time and resources, high school English Literature teachers often struggle to curate diverse, thematically aligned literature text sets. This study addresses the critical need for a tool that provides scaffolds for novice educators in selecting literature texts that are diverse -- in terms of genre, theme, subtheme, and author -- yet similar in context and pedagogical merits. We have developed a recommendation system, Teaching Text Expansion for Teacher Scaffolding (T-TExTS), that suggests high school English Literature books based on pedagogical merits, genre, and thematic relevance using a knowledge graph. We constructed a domain-specific ontology using the KNowledge Acquisition and Representation Methodology (KNARM), transformed into a knowledge graph, which was then embedded using DeepWalk, biased random walk, and a hybrid of both approaches. The system was evaluated using link prediction and recommendation performance metrics, including Area Under the Curve (AUC), Mean Reciprocal Rank (MRR), Hits@K, and normalized Discounted Cumulative Gain (nDCG). DeepWalk outperformed in most ranking metrics, with the highest AUC (0.9431), whereas the hybrid model offered balanced performance. These findings demonstrate the importance of semantic, ontology-driven approaches in recommendation systems and suggest that T-TExTS can significantly ease the burden of English Literature text selection for high school educators, promoting more informed and inclusive curricular decisions. The source code for T-TExTS is available at: this https URL 

**Abstract (ZH)**: 在中学教室实施转变式教学法需要采用广义的多文学素养方法。由于规划时间有限和资源有限，高中英语文学教师在收集多样且主题一致的文学文本集方面常常面临困难。本研究旨在满足新手教师在选择文学文本方面的需求，这些文本在类型、主题、副主题和作者方面多样，但在情境和教学价值方面相似。我们开发了一种推荐系统——教师支架用以扩展教学文本（T-TExTS），该系统基于教学价值、类型和主题相关性建议高中英语文学书籍，并使用知识图谱进行建议。我们使用KNARM知识获取与表示方法构建了一个领域特定的本体，并将其转化为知识图谱，随后使用DeepWalk、带偏向的随机游走及两种方法的混合方法嵌入。系统通过链预测和推荐性能指标（包括AUC、MRR、Hits@K和nDCG）进行了评估。DeepWalk 在大多数排序指标中表现最佳，AUC（0.9431）最高，而混合模型提供了平衡的性能。这些发现展示了语义和本体驱动方法在推荐系统中的重要性，并表明T-TExTS 可显著减轻高中英语文学文本选择的负担，促进更具信息量和包容性的课程决策。T-TExTS 的源代码可通过以下链接获得：this https URL。 

---
# Seamless Dysfluent Speech Text Alignment for Disordered Speech Analysis 

**Title (ZH)**: 无障碍断裂语音文本对齐以分析发音障碍 

**Authors**: Zongli Ye, Jiachen Lian, Xuanru Zhou, Jinming Zhang, Haodong Li, Shuhe Li, Chenxu Guo, Anaisha Das, Peter Park, Zoe Ezzes, Jet Vonk, Brittany Morin, Rian Bogley, Lisa Wauters, Zachary Miller, Maria Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.12073)  

**Abstract**: Accurate alignment of dysfluent speech with intended text is crucial for automating the diagnosis of neurodegenerative speech disorders. Traditional methods often fail to model phoneme similarities effectively, limiting their performance. In this work, we propose Neural LCS, a novel approach for dysfluent text-text and speech-text alignment. Neural LCS addresses key challenges, including partial alignment and context-aware similarity mapping, by leveraging robust phoneme-level modeling. We evaluate our method on a large-scale simulated dataset, generated using advanced data simulation techniques, and real PPA data. Neural LCS significantly outperforms state-of-the-art models in both alignment accuracy and dysfluent speech segmentation. Our results demonstrate the potential of Neural LCS to enhance automated systems for diagnosing and analyzing speech disorders, offering a more accurate and linguistically grounded solution for dysfluent speech alignment. 

**Abstract (ZH)**: 准确对齐非流利语音与意图文本对于自动化神经退行性言语障碍诊断至关重要。传统方法往往无法有效地建模音素相似性，限制了其性能。本文提出了一种新的非流利文本-文本和语音-文本对齐方法——神经最长公共子序列（Neural LCS），通过利用稳健的音素级建模来解决部分对齐和上下文相关相似性映射等关键挑战。我们在使用先进的数据模拟技术生成的大规模模拟数据集和真实PPA数据上评估了该方法，结果显示Neural LCS在对齐准确性和非流利语音分割方面显著优于当前最先进的模型。我们的结果表明，Neural LCS 有潜力增强自动化系统以诊断和分析言语障碍，提供一种更准确且语言学上更可靠的非流利语音对齐方案。 

---
# Evaluating Logit-Based GOP Scores for Mispronunciation Detection 

**Title (ZH)**: 基于逻辑斯蒂回归的GOP评分在错音检测中的评估 

**Authors**: Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2506.12067)  

**Abstract**: Pronunciation assessment relies on goodness of pronunciation (GOP) scores, traditionally derived from softmax-based posterior probabilities. However, posterior probabilities may suffer from overconfidence and poor phoneme separation, limiting their effectiveness. This study compares logit-based GOP scores with probability-based GOP scores for mispronunciation detection. We conducted our experiment on two L2 English speech datasets spoken by Dutch and Mandarin speakers, assessing classification performance and correlation with human ratings. Logit-based methods outperform probability-based GOP in classification, but their effectiveness depends on dataset characteristics. The maximum logit GOP shows the strongest alignment with human perception, while a combination of different GOP scores balances probability and logit features. The findings suggest that hybrid GOP methods incorporating uncertainty modeling and phoneme-specific weighting improve pronunciation assessment. 

**Abstract (ZH)**: 基于逻辑斯谛和概率的发音评分方法在误读检测中的对比研究：综合不确定性建模和音素特异性加权以提高发音评估 

---
# Organizational Adaptation to Generative AI in Cybersecurity: A Systematic Review 

**Title (ZH)**: 组织应对生成式AI在网络安全中的适应：一项系统性回顾 

**Authors**: Christopher Nott  

**Link**: [PDF](https://arxiv.org/pdf/2506.12060)  

**Abstract**: Cybersecurity organizations are adapting to GenAI integration through modified frameworks and hybrid operational processes, with success influenced by existing security maturity, regulatory requirements, and investments in human capital and infrastructure. This qualitative research employs systematic document analysis and comparative case study methodology to examine how cybersecurity organizations adapt their threat modeling frameworks and operational processes to address generative artificial intelligence integration. Through examination of 25 studies from 2022 to 2025, the research documents substantial transformation in organizational approaches to threat modeling, moving from traditional signature-based systems toward frameworks incorporating artificial intelligence capabilities. The research identifies three primary adaptation patterns: Large Language Model integration for security applications, GenAI frameworks for risk detection and response automation, and AI/ML integration for threat hunting. Organizations with mature security infrastructures, particularly in finance and critical infrastructure sectors, demonstrate higher readiness through structured governance approaches, dedicated AI teams, and robust incident response processes. Organizations achieve successful GenAI integration when they maintain appropriate human oversight of automated systems, address data quality concerns and explainability requirements, and establish governance frameworks tailored to their specific sectors. Organizations encounter ongoing difficulties with privacy protection, bias reduction, personnel training, and defending against adversarial attacks. This work advances understanding of how organizations adopt innovative technologies in high-stakes environments and offers actionable insights for cybersecurity professionals implementing GenAI systems. 

**Abstract (ZH)**: 网络空间安全组织通过修改框架和混合运营流程适应生成式人工智能集成，其成功受现有安全成熟度、监管要求以及人力资本和基础设施投资的影响。本质性研究通过系统文件分析和比较案例研究方法，考察网络安全组织如何调整其威胁建模框架和运营流程以应对生成式人工智能集成。通过对2022年至2025年间25项研究的分析，研究记录了组织在威胁建模方面的显著转变，从传统的特征签名系统转向包含人工智能能力的框架。研究确定了三种主要适应模式：大型语言模型在安全应用中的集成、生成式人工智能框架用于风险检测和响应自动化，以及人工智能/机器学习在威胁检测中的集成。拥有成熟安全基础设施的组织，特别是在金融和关键基础设施领域，通过结构化的治理方法、专门的人工智能团队和 robust 的事件响应流程展示了更高的准备度。当组织维持适当的自动化系统的人工监督、解决数据质量和解释性要求，并建立符合其特定领域的治理框架时，它们能够成功实现生成式人工智能的集成。组织在隐私保护、偏见减少、人员培训以及抵御对抗性攻击方面面临持续挑战。本研究加深了对组织如何在高风险环境中采用创新技术的理解，并为实施生成式人工智能系统的网络安全专业人员提供了可操作的见解。 

---
# Towards Unified Neural Decoding with Brain Functional Network Modeling 

**Title (ZH)**: 基于脑功能网络建模的统一神经解码研究 

**Authors**: Di Wu, Linghao Bu, Yifei Jia, Lu Cao, Siyuan Li, Siyu Chen, Yueqian Zhou, Sheng Fan, Wenjie Ren, Dengchang Wu, Kang Wang, Yue Zhang, Yuehui Ma, Jie Yang, Mohamad Sawan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12055)  

**Abstract**: Recent achievements in implantable brain-computer interfaces (iBCIs) have demonstrated the potential to decode cognitive and motor behaviors with intracranial brain recordings; however, individual physiological and electrode implantation heterogeneities have constrained current approaches to neural decoding within single individuals, rendering interindividual neural decoding elusive. Here, we present Multi-individual Brain Region-Aggregated Network (MIBRAIN), a neural decoding framework that constructs a whole functional brain network model by integrating intracranial neurophysiological recordings across multiple individuals. MIBRAIN leverages self-supervised learning to derive generalized neural prototypes and supports group-level analysis of brain-region interactions and inter-subject neural synchrony. To validate our framework, we recorded stereoelectroencephalography (sEEG) signals from a cohort of individuals performing Mandarin syllable articulation. Both real-time online and offline decoding experiments demonstrated significant improvements in both audible and silent articulation decoding, enhanced decoding accuracy with increased multi-subject data integration, and effective generalization to unseen subjects. Furthermore, neural predictions for regions without direct electrode coverage were validated against authentic neural data. Overall, this framework paves the way for robust neural decoding across individuals and offers insights for practical clinical applications. 

**Abstract (ZH)**: 近期可植入脑机接口的进展展示了通过颅内脑记录解码认知和运动行为的潜力；然而，个体生理差异和电极植入异质性限制了当前在单一个体中进行神经解码的方法，使其在个体间神经解码难以实现。为此，我们提出了多个体脑区聚合网络（MIBRAIN），这是一种通过整合多个个体的颅内神经生理记录构建全功能脑网络模型的神经解码框架。MIBRAIN 利用半监督学习提取通用神经原型，并支持脑区间交互和跨个体神经同步的组级分析。为了验证该框架，我们对执行汉语音节发音的一组个体记录了立体脑电图（sEEG）信号。实时在线和离线解码实验均显示了对可闻和无声发音解码的重大改进，并且随着多个体数据集成解码准确性提高，还展示了对未见个体的有效泛化。此外，未直接覆盖电极的区域的神经预测与真实神经数据相符。总体而言，该框架为跨个体的稳健神经解码铺平了道路，并为实际临床应用提供了见解。 

---
# From Proxies to Fields: Spatiotemporal Reconstruction of Global Radiation from Sparse Sensor Sequences 

**Title (ZH)**: 从代理到场域：基于稀疏传感器序列的全球辐射时空重构 

**Authors**: Kazuma Kobayashi, Samrendra Roy, Seid Koric, Diab Abueidda, Syed Bahauddin Alam  

**Link**: [PDF](https://arxiv.org/pdf/2506.12045)  

**Abstract**: Accurate reconstruction of latent environmental fields from sparse and indirect observations is a foundational challenge across scientific domains-from atmospheric science and geophysics to public health and aerospace safety. Traditional approaches rely on physics-based simulators or dense sensor networks, both constrained by high computational cost, latency, or limited spatial coverage. We present the Temporal Radiation Operator Network (TRON), a spatiotemporal neural operator architecture designed to infer continuous global scalar fields from sequences of sparse, non-uniform proxy measurements.
Unlike recent forecasting models that operate on dense, gridded inputs to predict future states, TRON addresses a more ill-posed inverse problem: reconstructing the current global field from sparse, temporally evolving sensor sequences, without access to future observations or dense labels. Demonstrated on global cosmic radiation dose reconstruction, TRON is trained on 22 years of simulation data and generalizes across 65,341 spatial locations, 8,400 days, and sequence lengths from 7 to 90 days. It achieves sub-second inference with relative L2 errors below 0.1%, representing a >58,000X speedup over Monte Carlo-based estimators. Though evaluated in the context of cosmic radiation, TRON offers a domain-agnostic framework for scientific field reconstruction from sparse data, with applications in atmospheric modeling, geophysical hazard monitoring, and real-time environmental risk forecasting. 

**Abstract (ZH)**: 从稀疏间接观测准确重构潜在环境场：时空神经算子网络在跨科学领域中的基础挑战及解决方案 

---
# CRITS: Convolutional Rectifier for Interpretable Time Series Classification 

**Title (ZH)**: CRITS: 卷积修正器用于可解释的时间序列分类 

**Authors**: Alejandro Kuratomi, Zed Lee, Guilherme Dinis Chaliane Junior, Tony Lindgren, Diego García Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2506.12042)  

**Abstract**: Several interpretability methods for convolutional network-based classifiers exist. Most of these methods focus on extracting saliency maps for a given sample, providing a local explanation that highlights the main regions for the classification. However, some of these methods lack detailed explanations in the input space due to upscaling issues or may require random perturbations to extract the explanations. We propose Convolutional Rectifier for Interpretable Time Series Classification, or CRITS, as an interpretable model for time series classification that is designed to intrinsically extract local explanations. The proposed method uses a layer of convolutional kernels, a max-pooling layer and a fully-connected rectifier network (a network with only rectified linear unit activations). The rectified linear unit activation allows the extraction of the feature weights for the given sample, eliminating the need to calculate gradients, use random perturbations and the upscale of the saliency maps to the initial input space. We evaluate CRITS on a set of datasets, and study its classification performance and its explanation alignment, sensitivity and understandability. 

**Abstract (ZH)**: 基于卷积网络的时间序列分类解释方法：Convolutional Rectifier for Interpretable Time Series Classification (CRITS) 

---
# Meta Pruning via Graph Metanetworks : A Meta Learning Framework for Network Pruning 

**Title (ZH)**: 基于图元网络的元剪枝：一种网络剪枝的元学习框架 

**Authors**: Yewei Liu, Xiyuan Wang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12041)  

**Abstract**: Network pruning, aimed at reducing network size while preserving accuracy, has attracted significant research interest. Numerous pruning techniques have been proposed over time. They are becoming increasingly effective, but more complex and harder to interpret as well. Given the inherent complexity of neural networks, we argue that manually designing pruning criteria has reached a bottleneck. To address this, we propose a novel approach in which we "use a neural network to prune neural networks". More specifically, we introduce the newly developed idea of metanetwork from meta-learning into pruning. A metanetwork is a network that takes another network as input and produces a modified network as output. In this paper, we first establish a bijective mapping between neural networks and graphs, and then employ a graph neural network as our metanetwork. We train a metanetwork that learns the pruning strategy automatically which can transform a network that is hard to prune into another network that is much easier to prune. Once the metanetwork is trained, our pruning needs nothing more than a feedforward through the metanetwork and the standard finetuning to prune at state-of-the-art. Our method achieved outstanding results on many popular and representative pruning tasks (including ResNet56 on CIFAR10, VGG19 on CIFAR100, ResNet50 on ImageNet). Our code is available at this https URL 

**Abstract (ZH)**: 基于神经网络修剪神经网络的方法：利用元网络自动学习裁剪策略 

---
# The Maximal Overlap Discrete Wavelet Scattering Transform and Its Application in Classification Tasks 

**Title (ZH)**: 最大重叠离散小波散射变换及其在分类任务中的应用 

**Authors**: Leonardo Fonseca Larrubia, Pedro Alberto Morettin, Chang Chiann  

**Link**: [PDF](https://arxiv.org/pdf/2506.12039)  

**Abstract**: We present the Maximal Overlap Discrete Wavelet Scattering Transform (MODWST), whose construction is inspired by the combination of the Maximal Overlap Discrete Wavelet Transform (MODWT) and the Scattering Wavelet Transform (WST). We also discuss the use of MODWST in classification tasks, evaluating its performance in two applications: stationary signal classification and ECG signal classification. The results demonstrate that MODWST achieved good performance in both applications, positioning itself as a viable alternative to popular methods like Convolutional Neural Networks (CNNs), particularly when the training data set is limited. 

**Abstract (ZH)**: 最大重叠离散小波散射变换（MODWST）及其在分类任务中的应用 

---
# Human-like Forgetting Curves in Deep Neural Networks 

**Title (ZH)**: 类人类的遗忘曲线在深度神经网络中 

**Authors**: Dylan Kline  

**Link**: [PDF](https://arxiv.org/pdf/2506.12034)  

**Abstract**: This study bridges cognitive science and neural network design by examining whether artificial models exhibit human-like forgetting curves. Drawing upon Ebbinghaus' seminal work on memory decay and principles of spaced repetition, we propose a quantitative framework to measure information retention in neural networks. Our approach computes the recall probability by evaluating the similarity between a network's current hidden state and previously stored prototype representations. This retention metric facilitates the scheduling of review sessions, thereby mitigating catastrophic forgetting during deployment and enhancing training efficiency by prompting targeted reviews. Our experiments with Multi-Layer Perceptrons reveal human-like forgetting curves, with knowledge becoming increasingly robust through scheduled reviews. This alignment between neural network forgetting curves and established human memory models identifies neural networks as an architecture that naturally emulates human memory decay and can inform state-of-the-art continual learning algorithms. 

**Abstract (ZH)**: 本研究通过探讨人工模型是否表现出类似人类的记忆衰减曲线，将认知科学与神经网络设计相结合，借鉴艾宾浩斯的记忆衰退及其间隔重复原理，提出了一种量化信息保留度的框架。该方法通过评估网络当前隐藏状态与先前存储的原型表示之间的相似性来计算回忆概率。该保留度指标有助于安排复习时段，从而在部署时减轻灾难性遗忘，并通过促进有针对性的复习来提高训练效率。我们的实验显示，多层感知机表现出类似人类的记忆衰减曲线，通过计划复习，知识逐渐变得更为稳固。神经网络的记忆衰减曲线与已建立的人类记忆模型之间的契合表明，神经网络自然地模仿了人类的记忆衰退，可以指导最新的持续学习算法的发展。 

---
# EMERGENT: Efficient and Manipulation-resistant Matching using GFlowNets 

**Title (ZH)**: EMERGENT: 效率高且抗操控的匹配方法基于GFlowNets 

**Authors**: Mayesha Tasnim, Erman Acar, Sennay Ghebreab  

**Link**: [PDF](https://arxiv.org/pdf/2506.12033)  

**Abstract**: The design of fair and efficient algorithms for allocating public resources, such as school admissions, housing, or medical residency, has a profound social impact. In one-sided matching problems, where individuals are assigned to items based on ranked preferences, a fundamental trade-off exists between efficiency and strategyproofness. Existing algorithms like Random Serial Dictatorship (RSD), Probabilistic Serial (PS), and Rank Minimization (RM) capture only one side of this trade-off: RSD is strategyproof but inefficient, while PS and RM are efficient but incentivize manipulation. We propose EMERGENT, a novel application of Generative Flow Networks (GFlowNets) to one-sided matching, leveraging its ability to sample diverse, high-reward solutions. In our approach, efficient and manipulation-resistant matches emerge naturally: high-reward solutions yield efficient matches, while the stochasticity of GFlowNets-based outputs reduces incentives for manipulation. Experiments show that EMERGENT outperforms RSD in rank efficiency while significantly reducing strategic vulnerability compared to matches produced by RM and PS. Our work highlights the potential of GFlowNets for applications involving social choice mechanisms, where it is crucial to balance efficiency and manipulability. 

**Abstract (ZH)**: 公平且高效的公共资源分配算法设计：生成流网络在单边匹配问题中的应用 

---
# Embedding Trust at Scale: Physics-Aware Neural Watermarking for Secure and Verifiable Data Pipelines 

**Title (ZH)**: 大规模嵌入信任：物理感知神经水印在安全可验证数据管道中的应用 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2506.12032)  

**Abstract**: We present a robust neural watermarking framework for scientific data integrity, targeting high-dimensional fields common in climate modeling and fluid simulations. Using a convolutional autoencoder, binary messages are invisibly embedded into structured data such as temperature, vorticity, and geopotential. Our method ensures watermark persistence under lossy transformations - including noise injection, cropping, and compression - while maintaining near-original fidelity (sub-1\% MSE). Compared to classical singular value decomposition (SVD)-based watermarking, our approach achieves $>$98\% bit accuracy and visually indistinguishable reconstructions across ERA5 and Navier-Stokes datasets. This system offers a scalable, model-compatible tool for data provenance, auditability, and traceability in high-performance scientific workflows, and contributes to the broader goal of securing AI systems through verifiable, physics-aware watermarking. We evaluate on physically grounded scientific datasets as a representative stress-test; the framework extends naturally to other structured domains such as satellite imagery and autonomous-vehicle perception streams. 

**Abstract (ZH)**: 一种用于气候模型和流体模拟中高维数据完整性保护的稳健神经水印框架 

---
# Improving Generalization in Heterogeneous Federated Continual Learning via Spatio-Temporal Gradient Matching with Prototypical Coreset 

**Title (ZH)**: 基于空间-时间梯度匹配和原型coreset的异构联邦连续学习泛化能力提升 

**Authors**: Minh-Duong Nguyen, Le-Tuan Nguyen, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2506.12031)  

**Abstract**: Federated Continual Learning (FCL) has recently emerged as a crucial research area, as data from distributed clients typically arrives as a stream, requiring sequential learning. This paper explores a more practical and challenging FCL setting, where clients may have unrelated or even conflicting data and tasks. In this scenario, statistical heterogeneity and data noise can create spurious correlations, leading to biased feature learning and catastrophic forgetting. Existing FCL approaches often use generative replay to create pseudo-datasets of previous tasks. However, generative replay itself suffers from catastrophic forgetting and task divergence among clients, leading to overfitting in FCL. Existing FCL approaches often use generative replay to create pseudo-datasets of previous tasks. However, generative replay itself suffers from catastrophic forgetting and task divergence among clients, leading to overfitting in FCL. To address these challenges, we propose a novel approach called Spatio-Temporal grAdient Matching with network-free Prototype (STAMP). Our contributions are threefold: 1) We develop a model-agnostic method to determine subset of samples that effectively form prototypes when using a prototypical network, making it resilient to continual learning challenges; 2) We introduce a spatio-temporal gradient matching approach, applied at both the client-side (temporal) and server-side (spatial), to mitigate catastrophic forgetting and data heterogeneity; 3) We leverage prototypes to approximate task-wise gradients, improving gradient matching on the client-side. Extensive experiments demonstrate our method's superiority over existing baselines. 

**Abstract (ZH)**: 联邦连续学习（Federated Continual Learning, FCL）最近已成为一个关键的研究领域，因为来自分布式客户端的数据通常以流的形式到达，需要进行顺序学习。本文探讨了一个更具实践意义和挑战性的FCL设置，其中客户端可能具有无关甚至冲突的数据和任务。在这种场景下，统计异质性和数据噪声可以产生虚假的相关性，导致特征学习偏差和灾难性遗忘。现有FCL方法通常使用生成性重放来创建之前任务的伪数据集。然而，生成性重放本身会受到灾难性遗忘和客户端间任务发散的影响，导致在FCL中出现过拟合。为了解决这些挑战，我们提出了一种名为Spatio-Temporal GrAdient Matching with Network-free Prototype (STAMP)的新方法。我们的贡献包括三个方面：1）我们开发了一种模型无关的方法，用于确定在使用原型网络时形成的样本子集，使其能够应对连续学习的挑战；2）我们引入了一种时空梯度匹配方法，在客户端（时间维度）和服务器端（空间维度）应用，以缓解灾难性遗忘和数据异质性；3）我们利用原型来近似任务梯度，在客户端提高梯度匹配精度。广泛的实验结果表明，我们的方法优于现有的基线方法。 

---
# Impact, Causation and Prediction of Socio-Academic and Economic Factors in Exam-centric Student Evaluation Measures using Machine Learning and Causal Analysis 

**Title (ZH)**: 基于机器学习和因果分析的以考试为中心的学生评价指标中社会学术和经济因素的影响、因果关系及预测研究 

**Authors**: Md. Biplob Hosen, Sabbir Ahmed, Bushra Akter, Mehrin Anannya  

**Link**: [PDF](https://arxiv.org/pdf/2506.12030)  

**Abstract**: Understanding socio-academic and economic factors influencing students' performance is crucial for effective educational interventions. This study employs several machine learning techniques and causal analysis to predict and elucidate the impacts of these factors on academic performance. We constructed a hypothetical causal graph and collected data from 1,050 student profiles. Following meticulous data cleaning and visualization, we analyze linear relationships through correlation and variable plots, and perform causal analysis on the hypothetical graph. Regression and classification models are applied for prediction, and unsupervised causality analysis using PC, GES, ICA-LiNGAM, and GRASP algorithms is conducted. Our regression analysis shows that Ridge Regression achieve a Mean Absolute Error (MAE) of 0.12 and a Mean Squared Error (MSE) of 0.024, indicating robustness, while classification models like Random Forest achieve nearly perfect F1-scores. The causal analysis shows significant direct and indirect effects of factors such as class attendance, study hours, and group study on CGPA. These insights are validated through unsupervised causality analysis. By integrating the best regression model into a web application, we are developing a practical tool for students and educators to enhance academic outcomes based on empirical evidence. 

**Abstract (ZH)**: 理解影响学生学业表现的社会学术和经济因素对于有效的教育干预至关重要。本研究采用了多种机器学习技术和因果分析来预测并阐明这些因素对学业表现的影响。我们构建了一个假设因果图，并收集了1,050名学生的资料。经过细致的数据清洗和可视化处理，我们通过对相关性和变量图分析线性关系，并在假设图上进行因果分析。应用回归和分类模型进行预测，并使用PC、GES、ICA-LiNGAM和GRASP算法进行无监督因果分析。回归分析结果显示，岭回归的平均绝对误差（MAE）为0.12，均方误差（MSE）为0.024，表明其稳健性，而分类模型如随机森林几乎达到完美的F1-score。因果分析显示，班级出勤率、学习时间以及小组学习等因素对GPA有显著的直接影响和间接影响。这些洞察通过无监督因果分析得到验证。通过将最佳回归模型集成到-web应用中，我们正在开发一个基于实证证据的实际工具，帮助学生和教育工作者提升学业成果。 

---
# The Limits of Tractable Marginalization 

**Title (ZH)**: 可处理边际化的局限性 

**Authors**: Oliver Broadrick, Sanyam Agarwal, Guy Van den Broeck, Markus Bläser  

**Link**: [PDF](https://arxiv.org/pdf/2506.12020)  

**Abstract**: Marginalization -- summing a function over all assignments to a subset of its inputs -- is a fundamental computational problem with applications from probabilistic inference to formal verification. Despite its computational hardness in general, there exist many classes of functions (e.g., probabilistic models) for which marginalization remains tractable, and they can be commonly expressed by polynomial size arithmetic circuits computing multilinear polynomials. This raises the question, can all functions with polynomial time marginalization algorithms be succinctly expressed by such circuits? We give a negative answer, exhibiting simple functions with tractable marginalization yet no efficient representation by known models, assuming $\textsf{FP}\neq\#\textsf{P}$ (an assumption implied by $\textsf{P} \neq \textsf{NP}$). To this end, we identify a hierarchy of complexity classes corresponding to stronger forms of marginalization, all of which are efficiently computable on the known circuit models. We conclude with a completeness result, showing that whenever there is an efficient real RAM performing virtual evidence marginalization for a function, then there are small circuits for that function's multilinear representation. 

**Abstract (ZH)**: 隶属运算——对函数的一组输入的所有赋值求和——是概率推理和形式验证等领域的重要计算问题。尽管隶属运算通常计算难度较高，但仍有一些类别的函数（例如概率模型）使其保持可计算性，这些函数可以用计算多项式多项式的算术电路简洁表达。这引发了一个问题：所有具有多项式时间隶属运算算法的函数是否都能用这样的电路简洁表达？我们给出了否定的答案，展示了具有可计算隶属运算但无法用已知模型有效表示的简单函数（假设$\textsf{FP}\neq\#\textsf{P}$，这是一个P$\neq$NP的推论）。为此，我们识别了一个复杂性类层次结构，对应于更强形式的隶属运算，所有这些都在已知的电路模型上高效可计算。最后，我们得出一个完备性结果，表明每当存在一个高效的实数RAM实现某个函数的虚拟证据隶属运算时，都存在该函数的多项式线性表示的小电路。 

---
# Examining the effects of music on cognitive skills of children in early childhood with the Pythagorean fuzzy set approach 

**Title (ZH)**: 使用毕达哥拉斯模糊集方法探究音乐对幼儿认知能力的影响 

**Authors**: Murat Kirisci, Nihat Topac, Musa Bardak  

**Link**: [PDF](https://arxiv.org/pdf/2506.12016)  

**Abstract**: There are many genetic and environmental factors that affect cognitive development. Music education can also be considered as one of the environmental factors. Some researchers emphasize that Music is an action that requires meta-cognitive functions such as mathematics and chess and supports spatial intelligence. The effect of Music on cognitive development in early childhood was examined using the Pythagorean Fuzzy Sets(PFS) method defined by Yager. This study created PFS based on experts' opinions, and an algorithm was given according to PFS. The algorithm's results supported the experts' data on the development of spatial-temporal skills in music education given in early childhood. The algorithm's ranking was done using the Expectation Score Function. The rankings obtained from the algorithm overlap with the experts' rankings. 

**Abstract (ZH)**: 音乐教育对早期儿童认知发展的影响：基于Yager定义的Pythagorean模糊集方法的研究 

---
