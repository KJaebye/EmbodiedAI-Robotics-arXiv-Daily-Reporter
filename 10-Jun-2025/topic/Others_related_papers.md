# A Communication-Latency-Aware Co-Simulation Platform for Safety and Comfort Evaluation of Cloud-Controlled ICVs 

**Title (ZH)**: 一种通信时延感知的联合仿真平台：用于云控制的ICVs安全与舒适性评估 

**Authors**: Yongqi Zhao, Xinrui Zhang, Tomislav Mihalj, Martin Schabauer, Luis Putzer, Erik Reichmann-Blaga, Ádám Boronyák, András Rövid, Gábor Soós, Peizhi Zhang, Lu Xiong, Jia Hu, Arno Eichberger  

**Link**: [PDF](https://arxiv.org/pdf/2506.07696)  

**Abstract**: Testing cloud-controlled intelligent connected vehicles (ICVs) requires simulation environments that faithfully emulate both vehicle behavior and realistic communication latencies. This paper proposes a latency-aware co-simulation platform integrating CarMaker and Vissim to evaluate safety and comfort under real-world vehicle-to-cloud (V2C) latency conditions. Two communication latency models, derived from empirical 5G measurements in China and Hungary, are incorporated and statistically modeled using Gamma distributions. A proactive conflict module (PCM) is proposed to dynamically control background vehicles and generate safety-critical scenarios. The platform is validated through experiments involving an exemplary system under test (SUT) across six testing conditions combining two PCM modes (enabled/disabled) and three latency conditions (none, China, Hungary). Safety and comfort are assessed using metrics including collision rate, distance headway, post-encroachment time, and the spectral characteristics of longitudinal acceleration. Results show that the PCM effectively increases driving environment criticality, while V2C latency primarily affects ride comfort. These findings confirm the platform's effectiveness in systematically evaluating cloud-controlled ICVs under diverse testing conditions. 

**Abstract (ZH)**: 基于延迟感知的集成CarMaker和Vissim协同仿真平台：评估实际车辆到云端（V2C）延迟条件下的安全性和舒适性 

---
# Fractional Collisions: A Framework for Risk Estimation of Counterfactual Conflicts using Autonomous Driving Behavior Simulations 

**Title (ZH)**: 分数碰撞：一种基于自动驾驶行为仿真对抗事实冲突风险估计的框架 

**Authors**: Sreeja Roy-Singh, Sarvesh Kolekar, Daniel P. Bonny, Kyle Foss  

**Link**: [PDF](https://arxiv.org/pdf/2506.07540)  

**Abstract**: We present a methodology for estimating collision risk from counterfactual simulated scenarios built on sensor data from automated driving systems (ADS) or naturalistic driving databases. Two-agent conflicts are assessed by detecting and classifying conflict type, identifying the agents' roles (initiator or responder), identifying the point of reaction of the responder, and modeling their human behavioral expectations as probabilistic counterfactual trajectories. The states are used to compute velocity differentials at collision, which when combined with crash models, estimates severity of loss in terms of probabilistic injury or property damage, henceforth called fractional collisions. The probabilistic models may also be extended to include other uncertainties associated with the simulation, features, and agents. We verify the effectiveness of the methodology in a synthetic simulation environment using reconstructed trajectories from 300+ collision and near-collision scenes sourced from VTTI's SHRP2 database and Nexar dashboard camera data. Our methodology predicted fractional collisions within 1% of ground truth collisions. We then evaluate agent-initiated collision risk of an arbitrary ADS software release by replacing the naturalistic responder in these synthetic reconstructions with an ADS simulator and comparing the outcome to human-response outcomes. Our ADS reduced naturalistic collisions by 4x and fractional collision risk by ~62%. The framework's utility is also demonstrated on 250k miles of proprietary, open-loop sensor data collected on ADS test vehicles, re-simulated with an arbitrary ADS software release. The ADS initiated conflicts that caused 0.4 injury-causing and 1.7 property-damaging fractional collisions, and the ADS improved collision risk in 96% of the agent-initiated conflicts. 

**Abstract (ZH)**: 基于传感器数据的自动驾驶系统中碰撞风险估计方法 

---
# BR-MPPI: Barrier Rate guided MPPI for Enforcing Multiple Inequality Constraints with Learned Signed Distance Field 

**Title (ZH)**: BR-MPPI: 障碍率引导的MPPI方法，用于结合学习到的符号距离场强制执行多个不等式约束 

**Authors**: Hardik Parwana, Taekyung Kim, Kehan Long, Bardh Hoxha, Hideki Okamoto, Georgios Fainekos, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07325)  

**Abstract**: Model Predictive Path Integral (MPPI) controller is used to solve unconstrained optimal control problems and Control Barrier Function (CBF) is a tool to impose strict inequality constraints, a.k.a, barrier constraints. In this work, we propose an integration of these two methods that employ CBF-like conditions to guide the control sampling procedure of MPPI. CBFs provide an inequality constraint restricting the rate of change of barrier functions by a classK function of the barrier itself. We instead impose the CBF condition as an equality constraint by choosing a parametric linear classK function and treating this parameter as a state in an augmented system. The time derivative of this parameter acts as an additional control input that is designed by MPPI. A cost function is further designed to reignite Nagumo's theorem at the boundary of the safe set by promoting specific values of classK parameter to enforce safety. Our problem formulation results in an MPPI subject to multiple state and control-dependent equality constraints which are non-trivial to satisfy with randomly sampled control inputs. We therefore also introduce state transformations and control projection operations, inspired by the literature on path planning for manifolds, to resolve the aforementioned issue. We show empirically through simulations and experiments on quadrotor that our proposed algorithm exhibits better sampled efficiency and enhanced capability to operate closer to the safe set boundary over vanilla MPPI. 

**Abstract (ZH)**: 基于MPPI的MPFI控制器结合控制屏障函数的方法 

---
# Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset 

**Title (ZH)**: 改善Waymo开放运动数据集中的交通信号数据质量 

**Authors**: Xintao Yan, Erdao Liang, Jiawei Wang, Haojie Zhu, Henry X. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07150)  

**Abstract**: Datasets pertaining to autonomous vehicles (AVs) hold significant promise for a range of research fields, including artificial intelligence (AI), autonomous driving, and transportation engineering. Nonetheless, these datasets often encounter challenges related to the states of traffic signals, such as missing or inaccurate data. Such issues can compromise the reliability of the datasets and adversely affect the performance of models developed using them. This research introduces a fully automated approach designed to tackle these issues by utilizing available vehicle trajectory data alongside knowledge from the transportation domain to effectively impute and rectify traffic signal information within the Waymo Open Motion Dataset (WOMD). The proposed method is robust and flexible, capable of handling diverse intersection geometries and traffic signal configurations in real-world scenarios. Comprehensive validations have been conducted on the entire WOMD, focusing on over 360,000 relevant scenarios involving traffic signals, out of a total of 530,000 real-world driving scenarios. In the original dataset, 71.7% of traffic signal states are either missing or unknown, all of which were successfully imputed by our proposed method. Furthermore, in the absence of ground-truth signal states, the accuracy of our approach is evaluated based on the rate of red-light violations among vehicle trajectories. Results show that our method reduces the estimated red-light running rate from 15.7% in the original data to 2.9%, thereby demonstrating its efficacy in rectifying data inaccuracies. This paper significantly enhances the quality of AV datasets, contributing to the wider AI and AV research communities and benefiting various downstream applications. The code and improved traffic signal data are open-sourced at this https URL 

**Abstract (ZH)**: 自主驾驶车辆数据集在人工智能、自动驾驶和交通工程领域的潜在应用及其改进方法 

---
# DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning 

**Title (ZH)**: DriveSuprim: 向着端到端规划中精确轨迹选择的方向 

**Authors**: Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M. Alvarez, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06659)  

**Abstract**: In complex driving environments, autonomous vehicles must navigate safely. Relying on a single predicted path, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each, but face optimization challenges in precisely selecting the best option from thousands of possibilities and distinguishing subtle but safety-critical differences, especially in rare or underrepresented scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, demonstrating superior safetycritical capabilities, including collision avoidance and compliance with rules, while maintaining high trajectory quality in various driving scenarios. 

**Abstract (ZH)**: 在复杂驾驶环境中，自动驾驶车辆必须安全导航。单纯依赖基于回归的方法预测单一路径通常不会明示评估预测轨迹的安全性。选择性方法通过生成和评估多个轨迹候选并为每个候选预测安全评分来解决这一问题，但面临在成千上万种可能性中精确选择最佳选项及区分细微但关键的安全差异的优化挑战，尤其是在罕见或代表性不足的场景中。我们提出了DriveSuprim以克服这些挑战，并通过粗细结合的渐进式候选过滤、基于旋转的增强方法提高在分布外场景中的鲁棒性以及自我蒸馏框架稳定训练来推进选择性方法的范式。DriveSuprim在不使用额外数据的情况下实现了最先进的性能，在NAVSIM v1中达到93.5%的PDMS，在NAVSIM v2中达到87.1%的EPDMS，展示了卓越的安全关键能力，包括碰撞避免和遵守规则，同时在各种驾驶场景中保持高质量的轨迹。 

---
# Semantics-aware Predictive Inspection Path Planning 

**Title (ZH)**: 基于语义的预测性检验路径规划 

**Authors**: Mihir Dharmadhikari, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2506.06560)  

**Abstract**: This paper presents a novel semantics-aware inspection path planning paradigm called "Semantics-aware Predictive Planning" (SPP). Industrial environments that require the inspection of specific objects or structures (called "semantics"), such as ballast water tanks inside ships, often present structured and repetitive spatial arrangements of the semantics of interest. Motivated by this, we first contribute an algorithm that identifies spatially repeating patterns of semantics - exact or inexact - in a semantic scene graph representation and makes predictions about the evolution of the graph in the unseen parts of the environment using these patterns. Furthermore, two inspection path planning strategies, tailored to ballast water tank inspection, that exploit these predictions are proposed. To assess the performance of the novel predictive planning paradigm, both simulation and experimental evaluations are performed. First, we conduct a simulation study comparing the method against relevant state-of-the-art techniques and further present tests showing its ability to handle imperfect patterns. Second, we deploy our method onboard a collision-tolerant aerial robot operating inside the ballast tanks of two real ships. The results, both in simulation and field experiments, demonstrate significant improvement over the state-of-the-art in terms of inspection time while maintaining equal or better semantic surface coverage. A set of videos describing the different parts of the method and the field deployments is available at this https URL. The code for this work is made available at this https URL. 

**Abstract (ZH)**: 基于语义预测规划的新型检测路径规划范式 

---
# CPS-Guard: Framework for Dependability Assurance of AI- and LLM-Based Cyber-Physical Systems 

**Title (ZH)**: CPS-Guard: 身心物理系统中基于AI和大语言模型的可靠性和安全性保障框架 

**Authors**: Trisanth Srinivasan, Santosh Patapati, Himani Musku, Idhant Gode, Aditya Arora, Samvit Bhattacharya, Abubakr Nazriev, Sanika Hirave, Zaryab Kanjiani, Srinjoy Ghose, Srinidhi Shetty  

**Link**: [PDF](https://arxiv.org/pdf/2506.06381)  

**Abstract**: Cyber-Physical Systems (CPS) increasingly depend on advanced AI techniques to operate in critical applications. However, traditional verification and validation methods often struggle to handle the unpredictable and dynamic nature of AI components. In this paper, we introduce CPS-Guard, a novel framework that employs multi-role orchestration to automate the iterative assurance process for AI-powered CPS. By assigning specialized roles (e.g., safety monitoring, security assessment, fault injection, and recovery planning) to dedicated agents within a simulated environment, CPS-Guard continuously evaluates and refines AI behavior against a range of dependability requirements. We demonstrate the framework through a case study involving an autonomous vehicle navigating an intersection with an AI-based planner. Our results show that CPS-Guard effectively detects vulnerabilities, manages performance impacts, and supports adaptive recovery strategies, thereby offering a structured and extensible solution for rigorous V&V in safety- and security-critical systems. 

**Abstract (ZH)**: 基于多角色编排的CPS-Guard框架：AI驱动的CPS的迭代保证方法 

---
# Curriculum Learning With Counterfactual Group Relative Policy Advantage For Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于反事实组相对策略优势的多智能体强化学习 Curriculum 学习 

**Authors**: Weiqiang Jin, Hongyang Du, Guizhong Liu, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07548)  

**Abstract**: Multi-agent reinforcement learning (MARL) has achieved strong performance in cooperative adversarial tasks. However, most existing methods typically train agents against fixed opponent strategies and rely on such meta-static difficulty conditions, which limits their adaptability to changing environments and often leads to suboptimal policies. Inspired by the success of curriculum learning (CL) in supervised tasks, we propose a dynamic CL framework for MARL that employs an self-adaptive difficulty adjustment mechanism. This mechanism continuously modulates opponent strength based on real-time agent training performance, allowing agents to progressively learn from easier to more challenging scenarios. However, the dynamic nature of CL introduces instability due to nonstationary environments and sparse global rewards. To address this challenge, we develop a Counterfactual Group Relative Policy Advantage (CGRPA), which is tightly coupled with the curriculum by providing intrinsic credit signals that reflect each agent's impact under evolving task demands. CGRPA constructs a counterfactual advantage function that isolates individual contributions within group behavior, facilitating more reliable policy updates throughout the curriculum. CGRPA evaluates each agent's contribution through constructing counterfactual action advantage function, providing intrinsic rewards that enhance credit assignment and stabilize learning under non-stationary conditions. Extensive experiments demonstrate that our method improves both training stability and final performance, achieving competitive results against state-of-the-art methods. The code is available at this https URL. 

**Abstract (ZH)**: 多智能体强化学习中的自适应难度动态课程学习 

---
# Active Lubrication of Transluminal Medical Instruments 

**Title (ZH)**: 经腔内医疗器械的主动润滑 

**Authors**: Mostafa A. Atalla, Jelte Nieuwenhuis, Alan Martin, Xuan Wang, Ahranee Canden, Matt J. Carré, Roger Lewis, Aimée Sakes, Michaël Wiertlewski  

**Link**: [PDF](https://arxiv.org/pdf/2506.07225)  

**Abstract**: Transluminal minimally invasive surgery uses natural orifices and small incisions to access internal anatomical structures, promoting quicker recovery and reduced morbidity. However, navigating instruments--catheters and endoscopes--through anatomical pathways creates frictional interactions with luminal walls, risking complications such as perforation, poor haptic feedback, and instrument buckling. In this paper, we present a new approach to actively lubricate transluminal instruments and dynamically reduce friction with surrounding tissues. This approach employs ultrasonic vibrations, at the instrument surface, to generate a pressurized fluid layer at the contact interface, lubricating the interface and thereby reducing friction. We implemented this approach in a prototype catheter, which we validated under dry and liquid-lubricated conditions, across rigid and soft interfaces, and along varied anatomical curvatures. In a cardiac catheter use case, active lubrication reduced friction by up to 42% on ex-vivo porcine aorta tissue and 82% on rigid substrates, denoting its potential performance on healthy and calcified tissue, respectively. Thermal imaging confirmed that temperature at the tissue-catheter interface remained within safe limits. Additionally, the system effectively prevented buckling during catheter insertion experiment, further showcasing its potential. By minimizing injury risk and enhancing procedural stability, active lubrication can drastically enhance the safety and efficacy of transluminal interventions. 

**Abstract (ZH)**: 经自然腔道和小切口的最小侵入手术通过自然开口和小切口访问内部解剖结构，促进更快恢复和减少并发症。然而，通过解剖路径导航器械（导管和内窥镜）会产生与腔壁的摩擦交互，从而增加穿孔、触觉反馈差和器械弯曲等并发症的风险。本文提出了一种新的方法，通过在器械表面产生超声振动来主动润滑器械并动态减少与周围组织的摩擦。该方法通过在接触界面产生压力流体层来润滑界面，从而减少摩擦。我们在此方法上实现了一个原型导管，分别在干式和液体润滑条件下，以及在刚性和软性界面和不同解剖曲率上进行了验证。在心脏导管使用案例中，主动润滑在离体猪主动脉组织和刚性基底上分别减少了高达42%和82%的摩擦，表明其在健康和钙化组织上的潜在性能。热成像证实，组织-导管界面的温度保持在安全范围内。此外，该系统有效防止了导管插入期间的弯曲，进一步展示了其潜力。通过减少损伤风险并增强操作稳定性，主动润滑可以大幅提高经自然腔道干预的安全性和有效性。 

---
# QForce-RL: Quantized FPGA-Optimized Reinforcement Learning Compute Engine 

**Title (ZH)**: QForce-RL: 量化FPGA优化的强化学习计算引擎 

**Authors**: Anushka Jha, Tanushree Dewangan, Mukul Lokhande, Santosh Kumar Vishvakarma  

**Link**: [PDF](https://arxiv.org/pdf/2506.07046)  

**Abstract**: Reinforcement Learning (RL) has outperformed other counterparts in sequential decision-making and dynamic environment control. However, FPGA deployment is significantly resource-expensive, as associated with large number of computations in training agents with high-quality images and possess new challenges. In this work, we propose QForce-RL takes benefits of quantization to enhance throughput and reduce energy footprint with light-weight RL architecture, without significant performance degradation. QForce-RL takes advantages from E2HRL to reduce overall RL actions to learn desired policy and QuaRL for quantization based SIMD for hardware acceleration. We have also provided detailed analysis for different RL environments, with emphasis on model size, parameters, and accelerated compute ops. The architecture is scalable for resource-constrained devices and provide parametrized efficient deployment with flexibility in latency, throughput, power, and energy efficiency. The proposed QForce-RL provides performance enhancement up to 2.3x and better FPS - 2.6x compared to SoTA works. 

**Abstract (ZH)**: QForce-RL：利用量化提高 reinforcement learning  throughput 和降低能量足迹的轻量级架构 

---
# LoopDB: A Loop Closure Dataset for Large Scale Simultaneous Localization and Mapping 

**Title (ZH)**: LoopDB：一种用于大规模同步定位与建图的闭环匹配数据集 

**Authors**: Mohammad-Maher Nakshbandi, Ziad Sharawy, Dorian Cojocaru, Sorin Grigorescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06771)  

**Abstract**: In this study, we introduce LoopDB, which is a challenging loop closure dataset comprising over 1000 images captured across diverse environments, including parks, indoor scenes, parking spaces, as well as centered around individual objects. Each scene is represented by a sequence of five consecutive images. The dataset was collected using a high resolution camera, providing suitable imagery for benchmarking the accuracy of loop closure algorithms, typically used in simultaneous localization and mapping. As ground truth information, we provide computed rotations and translations between each consecutive images. Additional to its benchmarking goal, the dataset can be used to train and fine-tune loop closure methods based on deep neural networks. LoopDB is publicly available at this https URL. 

**Abstract (ZH)**: LoopDB：一种包含超过1000张图像的具有挑战性的环回闭合数据集，这些图像捕捉了包括公园、室内场景、停车位以及个体物体在内的多种环境。每个场景由五个连续图像组成。该数据集使用高分辨率相机收集，适合用于环回闭合算法（常用于即时定位与建图）的准确性benchmark。此外，我们提供了每对连续图像之间的旋转和平移的真实地面truth信息。除了benchmark目标外，该数据集还可用于训练和微调基于深度神经网络的环回闭合方法。LoopDB已公开，可通过以下链接获取：this https URL。 

---
# Towards Infant Sleep-Optimized Driving: Synergizing Wearable and Vehicle Sensing in Intelligent Cruise Control 

**Title (ZH)**: 面向婴儿睡眠优化的驾驶：融合可穿戴设备与车辆传感的智能巡航控制 

**Authors**: Ruitao Chen, Mozhang Guo, Jinge Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06459)  

**Abstract**: Automated driving (AD) has substantially improved vehicle safety and driving comfort, but their impact on passenger well-being, particularly infant sleep, is not sufficiently studied. Sudden acceleration, abrupt braking, and sharp maneuvers can disrupt infant sleep, compromising both passenger comfort and parental convenience. To solve this problem, this paper explores the integration of reinforcement learning (RL) within AD to personalize driving behavior and optimally balance occupant comfort and travel efficiency. In particular, we propose an intelligent cruise control framework that adapts to varying driving conditions to enhance infant sleep quality by effectively synergizing wearable sensing and vehicle data. Long short-term memory (LSTM) and transformer-based neural networks are integrated with RL to model the relationship between driving behavior and infant sleep quality under diverse traffic and road conditions. Based on the sleep quality indicators from the wearable sensors, driving action data from vehicle controllers, and map data from map applications, the model dynamically computes the optimal driving aggressiveness level, which is subsequently translated into specific AD control strategies, e.g., the magnitude and frequency of acceleration, lane change, and overtaking. Simulation results demonstrate that the proposed solution significantly improves infant sleep quality compared to baseline methods, while preserving desirable travel efficiency. 

**Abstract (ZH)**: 自动化驾驶对乘客福祉特别是婴儿睡眠的影响研究：结合强化学习的个性化驾驶行为优化 

---
# $τ^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment 

**Title (ZH)**: $τ^2$-Bench：在双控制环境中评估对话代理 

**Authors**: Victor Barres, Honghua Dong, Soham Ray, Xujie Si, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07982)  

**Abstract**: Existing benchmarks for conversational AI agents simulate single-control environments, where only the AI agent can use tools to interact with the world, while the user remains a passive information provider. This differs from real-world scenarios like technical support, where users need to actively participate in modifying the state of the (shared) world. In order to address this gap, we introduce $\tau^2$-bench, with four key contributions:
1) A novel Telecom dual-control domain modeled as a Dec-POMDP, where both agent and user make use of tools to act in a shared, dynamic environment that tests both agent coordination and communication,
2) A compositional task generator that programmatically creates diverse, verifiable tasks from atomic components, ensuring domain coverage and controlled complexity,
3) A reliable user simulator tightly coupled with the environment, whose behavior is constrained by tools and observable states, improving simulation fidelity,
4) Fine-grained analysis of agent performance through multiple ablations including separating errors arising from reasoning vs communication/coordination.
In particular, our experiments show significant performance drops when agents shift from no-user to dual-control, highlighting the challenges of guiding users. Overall, $\tau^2$-bench provides a controlled testbed for agents that must both reason effectively and guide user actions. 

**Abstract (ZH)**: $\tau^2$-基准：电信双控制域及其实验平台 

---
# Gradients: When Markets Meet Fine-tuning -- A Distributed Approach to Model Optimisation 

**Title (ZH)**: 梯度：当市场遇见微调——一种模型优化的分布式方法 

**Authors**: Christopher Subia-Waud  

**Link**: [PDF](https://arxiv.org/pdf/2506.07940)  

**Abstract**: Foundation model fine-tuning faces a fundamental challenge: existing AutoML platforms rely on single optimisation strategies that explore only a fraction of viable hyperparameter configurations. In this white paper, We introduce Gradients, a decentralised AutoML platform that transforms hyperparameter optimisation into a competitive marketplace where independent miners compete to discover optimal configurations. Economic incentives align individual exploration with collective optimisation goals, driving systematic investigation of hyperparameter regions that centralised methods miss. We evaluate our approach across 180 controlled experiments spanning diverse model architectures (70M to 70B parameters) and task types. Gradients achieves an 82.8\% win rate against HuggingFace AutoTrain and 100\% against TogetherAI, Databricks, and Google Cloud, with mean improvements of 11.8\% and 42.1\% respectively. Complex reasoning and retrieval tasks show particularly strong gains of 30-40\%, whilst diffusion models achieve 23.4\% improvements for person-specific generation. These results demonstrate that competitive, economically-driven approaches can systematically discover superior configurations that centralised AutoML consistently miss. 

**Abstract (ZH)**: 基于模型的微调面临一个根本性的挑战：现有的AutoML平台依赖于单一优化策略，只探索了可行的超参数配置的一小部分。在本白皮书中，我们介绍了一个去中心化的AutoML平台Gradients，它将超参数优化转变为一个竞争性市场，独立的挖掘者在这竞争市场中竞争以发现最优配置。经济激励将个体探索与集体优化目标对齐，驱动对中央化方法忽略的超参数区域进行系统性调查。我们通过涵盖多样模型架构（从70M到70B参数）和任务类型的180个受控实验评估了该方法。Gradients在对抗HuggingFace AutoTrain时取得82.8%的胜率，并在对抗TogetherAI、Databricks和Google Cloud时取得100%的胜率，分别平均提高了11.8%和42.1%。复杂的推理和检索任务显示出特别显著的20-40%的增益，而扩散模型在人物特定生成方面实现了23.4%的改进。这些结果表明，竞争性和经济驱动的方法能够系统性地发现中央化AutoML经常忽略的更优配置。 

---
# A Temporal FRBR/FRBRoo-Based Model for Component-Level Versioning of Legal Norms 

**Title (ZH)**: 基于FRBR/FRBRoo的时间维度组件级法规版本化模型 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07853)  

**Abstract**: Effectively representing legal norms for automated processing is a critical challenge, particularly in tracking the diachronic evolution of their hierarchical components (e.g., articles, paragraphs). While foundational frameworks like FRBR/FRBRoo and standards like Akoma Ntoso model legal documents at a macro level, they lack native mechanisms for granular, component-level versioning. This limitation hinders the deterministic point-in-time reconstruction of legal texts, a fundamental capability for reliable Legal Tech and AI applications. This paper proposes a structured, temporal model that extends the FRBRoo framework to address this gap. It introduces specialized subclasses of Expressio - Temporal Version (TV) and Language Version (LV - to represent the state of a legal norm and its linguistic variations at specific points in time. The model applies this same paradigm hierarchically, introducing Component Work (CW), Component Temporal Version (CTV), and Component Language Version (CLV) to track the lifecycle of individual articles, paragraphs, and clauses. Using the Brazilian Federal Constitution as a case study, the paper demonstrates how each amendment creates new Component Temporal Versions for affected provisions, while unaffected components retain their existing versions. This fine-grained, time-aware architecture enables the precise, deterministic retrieval and reconstruction of any part of a legal text as it existed on a specific date. The model provides a robust foundation for developing advanced legal information systems, knowledge graphs, and AI tools capable of accurate historical analysis and impact assessment, overcoming the limitations of current generative models. 

**Abstract (ZH)**: 有效表示法律规范以实现自动化处理是关键挑战，特别是在追踪其层级组件（例如，条款、段落）的历时演变方面。虽然FRBR/FRBRoo等基础框架和Akoma Ntoso等标准在宏观层面建模法律文件，但缺乏细粒度的组件级版本控制机制。这一限制阻碍了对法律文本在特定时间点的确定性重构，这对于可靠的Legal Tech和AI应用是基本能力。本文提出了一种结构化的时间模型，扩展了FRBRoo框架以解决这一缺口。该模型引入了专门的Expressio - 时间版本（TV）和语言版本（LV）子类，用于表示特定时间点的法律规范及其语言变化状态。模型采用分层范式，引入了组件工作（CW）、组件时间版本（CTV）和组件语言版本（CLV），以跟踪单个条款、段落和条款的生命周期。以巴西联邦宪法为例，本文展示了每次修正案如何为受影响的条款创建新的组件时间版本，而未受影响的组件则保留其现有版本。这种细粒度、时间感知的架构允许精确、确定地检索和重构特定日期存在的任何法律文本部分。该模型为开发先进的法律信息系统、知识图谱和能够进行准确历史分析和影响评估的AI工具提供了坚实基础，克服了当前生成模型的限制。 

---
# A Proposal to Extend the Common Model of Cognition with Metacognition 

**Title (ZH)**: 拟提出将元认知扩展到常用认知模型中 

**Authors**: John Laird, Christian Lebiere, Paul Rosenbloom, Andrea Stocco, Robert Wray  

**Link**: [PDF](https://arxiv.org/pdf/2506.07807)  

**Abstract**: The Common Model of Cognition (CMC) provides an abstract characterization of the structure and processing required by a cognitive architecture for human-like minds. We propose a unified approach to integrating metacognition within the CMC. We propose that metacognition involves reasoning over explicit representations of an agent's cognitive capabilities and processes in working memory. Our proposal exploits the existing cognitive capabilities of the CMC, making minimal extensions in the structure and information available within working memory. We provide examples of metacognition within our proposal. 

**Abstract (ZH)**: 基于认知的通用模型（CMC）提供了一种抽象化的人类思维所需的认知架构的结构和处理特征。我们提出了一种在CMC中集成元认知的统一方法。我们建议元认知涉及在工作记忆中对代理认知能力及过程的显式表示进行推理。我们的建议利用了CMC现有的认知能力，仅在工作记忆的结构和可用信息上做出最小扩展。我们提供了元认知在我们提案中的示例。 

---
# Agent Semantics, Semantic Spacetime, and Graphical Reasoning 

**Title (ZH)**: 代理语义、语义时空与图形推理 

**Authors**: Mark Burgess  

**Link**: [PDF](https://arxiv.org/pdf/2506.07756)  

**Abstract**: Some formal aspects of the Semantic Spacetime graph model are presented, with reference to its use for directed knowledge representations and process modelling. A finite $\gamma(3,4)$ representation is defined to form a closed set of operations that can scale to any degree of semantic complexity. The Semantic Spacetime postulates bring predictability with minimal constraints to pathways in graphs. The ubiquitous appearance of absorbing states in any partial graph means that a graph process leaks information. The issue is closely associated with the issue of division by zero, which signals a loss of closure and the need for manual injection of remedial information. The Semantic Spacetime model (and its Promise Theory) origins help to clarify how such absorbing states are associated with boundary information where intentionality can enter. 

**Abstract (ZH)**: 语义时空图模型的一些形式化方面及其在定向知识表示和过程建模中的应用 

---
# Automating Exploratory Multiomics Research via Language Models 

**Title (ZH)**: 通过语言模型自动化探索性多组学研究 

**Authors**: Shang Qu, Ning Ding, Linhai Xie, Yifei Li, Zaoqu Liu, Kaiyan Zhang, Yibai Xiong, Yuxin Zuo, Zhangren Chen, Ermo Hua, Xingtai Lv, Youbang Sun, Yang Li, Dong Li, Fuchu He, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07591)  

**Abstract**: This paper introduces PROTEUS, a fully automated system that produces data-driven hypotheses from raw data files. We apply PROTEUS to clinical proteogenomics, a field where effective downstream data analysis and hypothesis proposal is crucial for producing novel discoveries. PROTEUS uses separate modules to simulate different stages of the scientific process, from open-ended data exploration to specific statistical analysis and hypothesis proposal. It formulates research directions, tools, and results in terms of relationships between biological entities, using unified graph structures to manage complex research processes. We applied PROTEUS to 10 clinical multiomics datasets from published research, arriving at 360 total hypotheses. Results were evaluated through external data validation and automatic open-ended scoring. Through exploratory and iterative research, the system can navigate high-throughput and heterogeneous multiomics data to arrive at hypotheses that balance reliability and novelty. In addition to accelerating multiomic analysis, PROTEUS represents a path towards tailoring general autonomous systems to specialized scientific domains to achieve open-ended hypothesis generation from data. 

**Abstract (ZH)**: PROTEUS：一种用于从原始数据文件生成数据驱动假设的全自动系统 

---
# Coordinating Search-Informed Reasoning and Reasoning-Guided Search in Claim Verification 

**Title (ZH)**: 基于断言验证中的搜索指导推理和推理引导搜索协调 

**Authors**: Qisheng Hu, Quanyu Long, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07528)  

**Abstract**: Multi-hop claim verification is inherently challenging, requiring multi-step reasoning to construct verification chains while iteratively searching for information to uncover hidden bridging facts. This process is fundamentally interleaved, as effective reasoning relies on dynamically retrieved evidence, while effective search demands reasoning to refine queries based on partial information. To achieve this, we propose Hierarchical Agent Reasoning and Information Search (HARIS), explicitly modeling the coordinated process of reasoning-driven searching and search-informed reasoning. HARIS consists of a high-level reasoning agent that focuses on constructing the main verification chain, generating factual questions when more information is needed, and a low-level search agent that iteratively retrieves more information, refining its search based on intermediate findings. This design allows each agent to specialize in its respective task, enhancing verification accuracy and interpretability. HARIS is trained using reinforcement learning with outcome-based rewards. Experimental results on the EX-FEVER and HOVER benchmarks demonstrate that HARIS achieves strong performance, greatly advancing multi-hop claim verification. 

**Abstract (ZH)**: 多跳声明验证本质上具有挑战性，需要多步推理构建验证链，并在迭代搜索中逐步发现隐藏的中介事实。这一过程本质上是交织的，因为有效的推理依赖于动态检索的证据，而有效的搜索则需要根据部分信息对查询进行细化。为此，我们提出了层次化代理推理和信息搜索（HARIS），明确模型了由推理驱动的搜索和基于搜索的推理的协调过程。HARIS 包含一个高层推理代理，专注于构建主要验证链，在需要更多信息时生成事实性问题，以及一个低层搜索代理，迭代检索更多信息，并根据中间发现来细化其搜索。这种设计允许每个代理在其专门的任务中进行专业化，从而提高验证的准确性和可解释性。HARIS 使用基于结果的奖励进行强化学习训练。在 EX-FEVER 和 HOVER 基准上的实验结果表明，HARIS 在多跳声明验证方面取得了出色的表现，极大地推动了多跳声明验证的发展。 

---
# LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning 

**Title (ZH)**: LegalReasoner: 逐步验证-修正的法律判决推理方法 

**Authors**: Weijie Shi, Han Zhu, Jiaming Ji, Mengze Li, Jipeng Zhang, Ruiyuan Zhang, Jia Zhu, Jiajie Xu, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07443)  

**Abstract**: Legal judgment prediction (LJP) aims to function as a judge by making final rulings based on case claims and facts, which plays a vital role in the judicial domain for supporting court decision-making and improving judicial efficiency. However, existing methods often struggle with logical errors when conducting complex legal reasoning. We propose LegalReasoner, which enhances LJP reliability through step-wise verification and correction of the reasoning process. Specifically, it first identifies dispute points to decompose complex cases, and then conducts step-wise reasoning while employing a process verifier to validate each step's logic from correctness, progressiveness, and potential perspectives. When errors are detected, expert-designed attribution and resolution strategies are applied for correction. To fine-tune LegalReasoner, we release the LegalHK dataset, containing 58,130 Hong Kong court cases with detailed annotations of dispute points, step-by-step reasoning chains, and process verification labels. Experiments demonstrate that LegalReasoner significantly improves concordance with court decisions from 72.37 to 80.27 on LLAMA-3.1-70B. The data is available at this https URL. 

**Abstract (ZH)**: 法律判决预测（LJP）旨在基于案件主张和事实作出最终裁决，对司法领域支持法院决策和提高司法效率发挥重要作用。然而，现有方法在进行复杂法律推理时往往容易出现逻辑错误。我们提出了法律推理器（LegalReasoner），通过逐步验证和修正推理过程来提高LJP的可靠性。具体而言，它首先识别争议点以分解复杂案件，然后进行逐步推理，并使用过程验证器从正确性、进步性和潜在视角验证每一步的逻辑。当检测到错误时，应用专家设计的归因和解决策略进行修正。为了微调法律推理器（LegalReasoner），我们发布了包含58,130个香港法院案件及其详细标注的争议点、逐步推理链和过程验证标签的LegalHK数据集。实验表明，法律推理器在LLAMA-3.1-70B上使一致性与法院决策大幅提升，从72.37提高到80.27。数据可在如下链接获取：this https URL。 

---
# HeTa: Relation-wise Heterogeneous Graph Foundation Attack Model 

**Title (ZH)**: HeTa：关系层面的异构图基础攻击模型 

**Authors**: Yuling Wang, Zihui Chen, Pengfei Jiao, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07428)  

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) are vulnerable, highlighting the need for tailored attacks to assess their robustness and ensure security. However, existing HGNN attacks often require complex retraining of parameters to generate specific perturbations for new scenarios. Recently, foundation models have opened new horizons for the generalization of graph neural networks by capturing shared semantics across various graph distributions. This leads us to ask:Can we design a foundation attack model for HGNNs that enables generalizable perturbations across different HGNNs, and quickly adapts to new heterogeneous graphs (HGs)? Empirical findings reveal that, despite significant differences in model design and parameter space, different HGNNs surprisingly share common vulnerability patterns from a relation-aware perspective. Therefore, we explore how to design foundation HGNN attack criteria by mining shared attack units. In this paper, we propose a novel relation-wise heterogeneous graph foundation attack model, HeTa. We introduce a foundation surrogate model to align heterogeneity and identify the importance of shared relation-aware attack units. Building on this, we implement a serialized relation-by-relation attack based on the identified relational weights. In this way, the perturbation can be transferred to various target HGNNs and easily fine-tuned for new HGs. Extensive experiments exhibit powerful attack performances and generalizability of our method. 

**Abstract (ZH)**: 异质图神经网络（HGNNs）的脆弱性凸显了需要定制攻击模型以评估其鲁棒性和确保安全性的重要性。然而，现有的HGNN攻击通常需要复杂的参数重训练来生成针对新场景的具体扰动。最近，基础模型为图神经网络的一般化提供了新的视角，通过捕获各种图分布之间的共享语义。这使我们思考：我们能否设计一种基础攻击模型，使HGNNs在不同模型之间实现可泛化的扰动，并快速适应新的异质图（HGs）？实证研究发现，尽管不同HGNN的设计模型和参数空间存在显著差异，但从关系感知的角度来看，它们惊人地共享共同的脆弱性模式。因此，我们探索如何通过挖掘共享的关系感知攻击单元来设计基础HGNN攻击标准。在本文中，我们提出了一种新颖的关系感知异质图基础攻击模型HeTa。我们引入了一个基础代理模型来对齐异质性并识别共享关系感知攻击单元的重要性。在此基础上，我们基于识别出的关系权重实施了按关系逐步的攻击。这样，扰动可以转移到各种目标HGNNs，并且可以轻松微调以适应新的HGs。广泛实验展示了我们方法的强大攻击性能和一般化能力。 

---
# Subgoal-Guided Policy Heuristic Search with Learned Subgoals 

**Title (ZH)**: 基于子目标引导的策略启发式搜索与学习到的子目标 

**Authors**: Jake Tuero, Michael Buro, Levi H. S. Lelis  

**Link**: [PDF](https://arxiv.org/pdf/2506.07255)  

**Abstract**: Policy tree search is a family of tree search algorithms that use a policy to guide the search. These algorithms provide guarantees on the number of expansions required to solve a given problem that are based on the quality of the policy. While these algorithms have shown promising results, the process in which they are trained requires complete solution trajectories to train the policy. Search trajectories are obtained during a trial-and-error search process. When the training problem instances are hard, learning can be prohibitively costly, especially when starting from a randomly initialized policy. As a result, search samples are wasted in failed attempts to solve these hard instances. This paper introduces a novel method for learning subgoal-based policies for policy tree search algorithms. The subgoals and policies conditioned on subgoals are learned from the trees that the search expands while attempting to solve problems, including the search trees of failed attempts. We empirically show that our policy formulation and training method improve the sample efficiency of learning a policy and heuristic function in this online setting. 

**Abstract (ZH)**: 基于子目标的学习方法以提高策略树搜索中的样本效率 

---
# Translating Federated Learning Algorithms in Python into CSP Processes Using ChatGPT 

**Title (ZH)**: 将Python中的联邦学习算法转换为CSP进程using ChatGPT 

**Authors**: Miroslav Popovic, Marko Popovic, Miodrag Djukic, Ilija Basicevic  

**Link**: [PDF](https://arxiv.org/pdf/2506.07173)  

**Abstract**: The Python Testbed for Federated Learning Algorithms is a simple Python FL framework that is easy to use by ML&AI developers who do not need to be professional programmers and is also amenable to LLMs. In the previous research, generic federated learning algorithms provided by this framework were manually translated into the CSP processes and algorithms' safety and liveness properties were automatically verified by the model checker PAT. In this paper, a simple translation process is introduced wherein the ChatGPT is used to automate the translation of the mentioned federated learning algorithms in Python into the corresponding CSP processes. Within the process, the minimality of the used context is estimated based on the feedback from ChatGPT. The proposed translation process was experimentally validated by successful translation (verified by the model checker PAT) of both generic centralized and decentralized federated learning algorithms. 

**Abstract (ZH)**: Python测试床中的联邦学习算法：一种易于使用的Python联邦学习框架，适用于无需专业编程知识的ML&AI开发者，并且易于转换为LLMs。本文介绍了一个简单的转换过程，使用ChatGPT自动化转换Python中的联邦学习算法为相应的CSP过程，并基于ChatGPT的反馈估计所用上下文的最小性。所提出的转换过程通过成功转换（由模型检查器PAT验证）通用中心化和去中心化联邦学习算法得到实验验证。 

---
# Reasoning Paths as Signals: Augmenting Multi-hop Fact Verification through Structural Reasoning Progression 

**Title (ZH)**: 推理路径作为信号：通过结构推理进程增强多跳事实核实 

**Authors**: Liwen Zheng, Chaozhuo Li, Haoran Jia, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07075)  

**Abstract**: The growing complexity of factual claims in real-world scenarios presents significant challenges for automated fact verification systems, particularly in accurately aggregating and reasoning over multi-hop evidence. Existing approaches often rely on static or shallow models that fail to capture the evolving structure of reasoning paths, leading to fragmented retrieval and limited interpretability. To address these issues, we propose a Structural Reasoning framework for Multi-hop Fact Verification that explicitly models reasoning paths as structured graphs throughout both evidence retrieval and claim verification stages. Our method comprises two key modules: a structure-enhanced retrieval mechanism that constructs reasoning graphs to guide evidence collection, and a reasoning-path-guided verification module that incrementally builds subgraphs to represent evolving inference trajectories. We further incorporate a structure-aware reasoning mechanism that captures long-range dependencies across multi-hop evidence chains, enabling more precise verification. Extensive experiments on the FEVER and HoVer datasets demonstrate that our approach consistently outperforms strong baselines, highlighting the effectiveness of reasoning-path modeling in enhancing retrieval precision and verification accuracy. 

**Abstract (ZH)**: 复杂事实断言在现实场景中的不断增加使得自动事实验证系统面临重大挑战，特别是在多跳证据聚类和推理方面。现有方法往往依赖于静态或浅层模型，无法捕捉推理路径的演变结构，导致检索碎片化和解释性有限。为解决这些问题，我们提出了一种结构推理框架，用于多跳事实验证，在证据检索和断言验证阶段明确地将推理路径建模为结构化图。该方法包含两个关键模块：一种结构增强的检索机制，用于构建推理图以指导证据收集，以及一种由推理路径引导的验证模块，通过增量构建子图来表示不断演化的推断轨迹。我们进一步引入了一种结构感知的推理机制，能够在多跳证据链中捕捉长范围依赖性，从而实现更精确的验证。在FEVER和HoVer数据集上的 extensive 实验表明，我们的方法在检索精度和验证准确性方面均优于强基线，强调了推理路径建模在提升检索和验证效果方面的有效性。 

---
# Mathesis: Towards Formal Theorem Proving from Natural Languages 

**Title (ZH)**: 数学原理：从自然语言向形式定理证明的探索 

**Authors**: Yu Xuejun, Jianyuan Zhong, Zijin Feng, Pengyi Zhai, Roozbeh Yousefzadeh, Wei Chong Ng, Haoxiong Liu, Ziyi Shou, Jing Xiong, Yudong Zhou, Claudia Beth Ong, Austen Jeremy Sugiarto, Yaoxi Zhang, Wai Ming Tai, Huan Cao, Dongcai Lu, Jiacheng Sun, Qiang Xu, Shen Xin, Zhenguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.07047)  

**Abstract**: Recent advances in large language models show strong promise for formal reasoning. However, most LLM-based theorem provers have long been constrained by the need for expert-written formal statements as inputs, limiting their applicability to real-world problems expressed in natural language. We tackle this gap with Mathesis, the first end-to-end theorem proving pipeline processing informal problem statements. It contributes Mathesis-Autoformalizer, the first autoformalizer using reinforcement learning to enhance the formalization ability of natural language problems, aided by our novel LeanScorer framework for nuanced formalization quality assessment. It also proposes a Mathesis-Prover, which generates formal proofs from the formalized statements. To evaluate the real-world applicability of end-to-end formal theorem proving, we introduce Gaokao-Formal, a benchmark of 488 complex problems from China's national college entrance exam. Our approach is carefully designed, with a thorough study of each component. Experiments demonstrate Mathesis's effectiveness, with the autoformalizer outperforming the best baseline by 22% in pass-rate on Gaokao-Formal. The full system surpasses other model combinations, achieving 64% accuracy on MiniF2F with pass@32 and a state-of-the-art 18% on Gaokao-Formal. 

**Abstract (ZH)**: Recent Advances in Large Language Models Show Strong Promise for Formal Reasoning: Tackling the Gap with Mathesis for End-to-End Theorem Proving 

---
# Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents in Open-Ended Environments 

**Title (ZH)**: Deep RL 需要深入的行为分析：在开放环境中模型自由代理隐含规划的探索 

**Authors**: Riley Simmons-Edler, Ryan P. Badman, Felix Baastad Berg, Raymond Chua, John J. Vastola, Joshua Lunger, William Qian, Kanaka Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06981)  

**Abstract**: Understanding the behavior of deep reinforcement learning (DRL) agents -- particularly as task and agent sophistication increase -- requires more than simple comparison of reward curves, yet standard methods for behavioral analysis remain underdeveloped in DRL. We apply tools from neuroscience and ethology to study DRL agents in a novel, complex, partially observable environment, ForageWorld, designed to capture key aspects of real-world animal foraging -- including sparse, depleting resource patches, predator threats, and spatially extended arenas. We use this environment as a platform for applying joint behavioral and neural analysis to agents, revealing detailed, quantitatively grounded insights into agent strategies, memory, and planning. Contrary to common assumptions, we find that model-free RNN-based DRL agents can exhibit structured, planning-like behavior purely through emergent dynamics -- without requiring explicit memory modules or world models. Our results show that studying DRL agents like animals -- analyzing them with neuroethology-inspired tools that reveal structure in both behavior and neural dynamics -- uncovers rich structure in their learning dynamics that would otherwise remain invisible. We distill these tools into a general analysis framework linking core behavioral and representational features to diagnostic methods, which can be reused for a wide range of tasks and agents. As agents grow more complex and autonomous, bridging neuroscience, cognitive science, and AI will be essential -- not just for understanding their behavior, but for ensuring safe alignment and maximizing desirable behaviors that are hard to measure via reward. We show how this can be done by drawing on lessons from how biological intelligence is studied. 

**Abstract (ZH)**: 理解深度强化学习（DRL）代理的行为——尤其是随着任务和代理复杂性的增加——需要超出简单的奖励曲线比较，而标准的行为分析方法在DRL领域仍然发展不足。我们应用神经科学和行为学的工具，研究代理在ForageWorld这一新颖、复杂、部分可观测环境中的行为，该环境旨在捕捉现实世界动物觅食的关键方面，包括稀疏、可耗尽的资源斑块、捕食者威胁以及空间扩展的竞技场。我们使用这一环境作为平台，进行联合行为和神经元分析，揭示了代理策略、记忆和规划的详细、量化的见解。与常见的假设相反，我们发现，无模型的基于RNN的DRL代理可以通过涌现动力学表现出结构化的、类似于计划的行为，而无需 Explicit的记忆模块或世界模型。我们的结果表明，将DRL代理类比于动物进行研究——使用神经行为学启发的工具来分析其行为和神经动力学中的结构——揭示了他们学习动态中的丰富结构，这些结构否则将是难以察觉的。我们将这些工具提炼为一种通用的分析框架，将核心行为和表示特征与诊断方法联系起来，该框架可以用于一系列任务和代理。随着代理变得越来越复杂和自主，整合神经科学、认知科学和AI将是必不可少的——不仅是为了理解其行为，也是为了确保安全对齐并最大化那些难以通过奖励衡量的有利行为。我们展示了如何通过借鉴研究生物智能的方法来实现这一点。 

---
# Long-Tailed Learning for Generalized Category Discovery 

**Title (ZH)**: 长尾学习在泛化类别发现中的应用 

**Authors**: Cuong Manh Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06965)  

**Abstract**: Generalized Category Discovery (GCD) utilizes labeled samples of known classes to discover novel classes in unlabeled samples. Existing methods show effective performance on artificial datasets with balanced distributions. However, real-world datasets are always imbalanced, significantly affecting the effectiveness of these methods. To solve this problem, we propose a novel framework that performs generalized category discovery in long-tailed distributions. We first present a self-guided labeling technique that uses a learnable distribution to generate pseudo-labels, resulting in less biased classifiers. We then introduce a representation balancing process to derive discriminative representations. By mining sample neighborhoods, this process encourages the model to focus more on tail classes. We conduct experiments on public datasets to demonstrate the effectiveness of the proposed framework. The results show that our model exceeds previous state-of-the-art methods. 

**Abstract (ZH)**: 通用类别发现（GCD）利用已标记的已知类别样本来发现未标记样本中的新类别。现有的方法在具有均衡分布的人工数据集上表现出有效的性能。然而，现实世界的数据集总是不平衡的，显著影响了这些方法的效果。为了解决这个问题，我们提出了一种新的框架，在长尾分布中执行通用类别发现。我们首先提出了一种自我指导的标注技术，使用可学习的分布生成伪标签，从而生成更少偏差的分类器。然后，我们引入了一种表示平衡过程来提取判别性表示。通过挖掘样本邻域，这个过程鼓励模型更多关注尾部类别。我们在公共数据集上进行了实验，以证明所提出框架的有效性。结果显示，我们的模型超越了之前的最先进方法。 

---
# Deontically Constrained Policy Improvement in Reinforcement Learning Agents 

**Title (ZH)**: 契约约束的强化学习代理策略改进 

**Authors**: Alena Makarova, Houssam Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2506.06959)  

**Abstract**: Markov Decision Processes (MDPs) are the most common model for decision making under uncertainty in the Machine Learning community. An MDP captures non-determinism, probabilistic uncertainty, and an explicit model of action. A Reinforcement Learning (RL) agent learns to act in an MDP by maximizing a utility function. This paper considers the problem of learning a decision policy that maximizes utility subject to satisfying a constraint expressed in deontic logic. In this setup, the utility captures the agent's mission - such as going quickly from A to B. The deontic formula represents (ethical, social, situational) constraints on how the agent might achieve its mission by prohibiting classes of behaviors. We use the logic of Expected Act Utilitarianism, a probabilistic stit logic that can be interpreted over controlled MDPs. We develop a variation on policy improvement, and show that it reaches a constrained local maximum of the mission utility. Given that in stit logic, an agent's duty is derived from value maximization, this can be seen as a way of acting to simultaneously maximize two value functions, one of which is implicit, in a bi-level structure. We illustrate these results with experiments on sample MDPs. 

**Abstract (ZH)**: 马尔可夫决策过程（MDPs）是机器学习社区中处理不确定性决策最常用的模型。本文考虑了在满足形述于规范逻辑的约束条件下，学习最大化目标价值的决策策略的问题。在这种设置中，价值代表代理的任务，如从A快速到达B。规范公式代表代理如何实现其任务的（伦理的、社会的、情境的）约束，通过禁止某些行为类别。我们使用期望行为功利逻辑，这是一种可以在控制的MDPs上进行解释的概率stit逻辑。我们开发了一种策略改进的变体，并证明它可以在任务价值上达到一个约束局部最大值。由于在stit逻辑中，代理的职责源于价值最大化，这可以被视为同时最大化两个价值函数的一种方式，其中一个价值函数是隐含的，其结构具有多层次性。我们通过在样本MDPs上的实验来说明这些结果。 

---
# KnowCoder-V2: Deep Knowledge Analysis 

**Title (ZH)**: KnowCoder-V2: 深度知识分析 

**Authors**: Zixuan Li, Wenxuan Liu, Long Bai, Chunmao Zhang, Wei Li, Fenghui Zhang, Quanxin Jin, Ruoyun He, Zhuo Chen, Zhilei Hu, Fei Wang, Bingbing Xu, Xuhui Jiang, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06881)  

**Abstract**: Deep knowledge analysis tasks always involve the systematic extraction and association of knowledge from large volumes of data, followed by logical reasoning to discover insights. However, to solve such complex tasks, existing deep research frameworks face three major challenges: 1) They lack systematic organization and management of knowledge; 2) They operate purely online, making it inefficient for tasks that rely on shared and large-scale knowledge; 3) They cannot perform complex knowledge computation, limiting their abilities to produce insightful analytical results. Motivated by these, in this paper, we propose a \textbf{K}nowledgeable \textbf{D}eep \textbf{R}esearch (\textbf{KDR}) framework that empowers deep research with deep knowledge analysis capability. Specifically, it introduces an independent knowledge organization phase to preprocess large-scale, domain-relevant data into systematic knowledge offline. Based on this knowledge, it extends deep research with an additional kind of reasoning steps that perform complex knowledge computation in an online manner. To enhance the abilities of LLMs to solve knowledge analysis tasks in the above framework, we further introduce \textbf{\KCII}, an LLM that bridges knowledge organization and reasoning via unified code generation. For knowledge organization, it generates instantiation code for predefined classes, transforming data into knowledge objects. For knowledge computation, it generates analysis code and executes on the above knowledge objects to obtain deep analysis results. Experimental results on more than thirty datasets across six knowledge analysis tasks demonstrate the effectiveness of \KCII. Moreover, when integrated into the KDR framework, \KCII can generate high-quality reports with insightful analytical results compared to the mainstream deep research framework. 

**Abstract (ZH)**: 一种具备深度知识分析能力的KDResearch框架 

---
# Bio-Inspired Classification: Combining Information Theory and Spiking Neural Networks -- Influence of the Learning Rules 

**Title (ZH)**: 生物启发分类：结合信息理论与脉冲神经网络——学习规则的影响 

**Authors**: Zofia Rudnicka, Janusz Szczepanski, Agnieszka Pregowska  

**Link**: [PDF](https://arxiv.org/pdf/2506.06750)  

**Abstract**: Training of Spiking Neural Networks (SNN) is challenging due to their unique properties, including temporal dynamics, non-differentiability of spike events, and sparse event-driven activations. In this paper, we widely consider the influence of the type of chosen learning algorithm, including bioinspired learning rules on the accuracy of classification. We proposed a bioinspired classifier based on the combination of SNN and Lempel-Ziv complexity (LZC). This approach synergizes the strengths of SNNs in temporal precision and biological realism with LZC's structural complexity analysis, facilitating efficient and interpretable classification of spatiotemporal neural data. It turned out that the classic backpropagation algorithm achieves excellent classification accuracy, but at extremely high computational cost, which makes it impractical for real-time applications. Biologically inspired learning algorithms such as tempotron and Spikprop provide increased computational efficiency while maintaining competitive classification performance, making them suitable for time-sensitive tasks. The results obtained indicate that the selection of the most appropriate learning algorithm depends on the trade-off between classification accuracy and computational cost as well as application constraints. 

**Abstract (ZH)**: 基于突触神经网络的训练因其实时性、突触事件的非可微性质以及稀疏的事例驱动激活等特点具有挑战性。本文广泛考虑了所选学习算法类型，包括生物启发式学习规则对分类准确性的影响。我们提出了一种结合突触神经网络和Lempel-Ziv复杂性（LZC）的生物启发式分类器。该方法将SNN在时序精确度和生物现实性方面的优势与LZC在结构复杂性分析方面的优势相结合，有助于高效且可解释地分类时空神经数据。经典反向传播算法显示出出色的分类准确性，但计算成本极高，使其不适用于实时应用。生物启发式学习算法如tempotron和Spikprop在保持竞争力的分类性能的同时提高了计算效率，使其适用于时敏任务。实验结果表明，学习算法的选择取决于分类准确性和计算成本之间的权衡以及应用约束。 

---
# Honey, I shrunk the hypothesis space (through logical preprocessing) 

**Title (ZH)**: 蜂蜜，我缩小了假设空间（通过逻辑预处理） 

**Authors**: Andrew Cropper, Filipe Gouveia, David M. Cerna  

**Link**: [PDF](https://arxiv.org/pdf/2506.06739)  

**Abstract**: Inductive logic programming (ILP) is a form of logical machine learning. The goal is to search a hypothesis space for a hypothesis that generalises training examples and background knowledge. We introduce an approach that 'shrinks' the hypothesis space before an ILP system searches it. Our approach uses background knowledge to find rules that cannot be in an optimal hypothesis regardless of the training examples. For instance, our approach discovers relationships such as "even numbers cannot be odd" and "prime numbers greater than 2 are odd". It then removes violating rules from the hypothesis space. We implement our approach using answer set programming and use it to shrink the hypothesis space of a constraint-based ILP system. Our experiments on multiple domains, including visual reasoning and game playing, show that our approach can substantially reduce learning times whilst maintaining predictive accuracies. For instance, given just 10 seconds of preprocessing time, our approach can reduce learning times from over 10 hours to only 2 seconds. 

**Abstract (ZH)**: 基于逻辑的归纳学习（ILP）是一种形式的逻辑机器学习。目标是在假设空间中搜索能够概括训练例子和背景知识的假设。我们提出了一种在ILP系统搜索假设空间之前缩小假设空间的方法。我们的方法利用背景知识找到一些规则，这些规则无论训练例子如何都不会存在于最优假设之中。例如，我们的方法发现诸如“偶数不能是奇数”和“大于2的质数是奇数”这样的关系，并从中移除违反这些规则的假设。我们使用回答集编程实现该方法，并将其应用于基于约束的ILP系统假设空间的缩小。我们的实验涵盖多个领域，包括视觉推理和游戏玩，表明我们的方法可以在保持预测准确性的同时显著减少学习时间。例如，在只需要10秒的预处理时间后，我们的方法可以将学习时间从超过10小时缩短到仅2秒。 

---
# Integrating AI Planning Semantics into SysML System Models for Automated PDDL File Generation 

**Title (ZH)**: 将AI规划语义集成到SysML系统模型中以实现自动化PDDL文件生成 

**Authors**: Hamied Nabizada, Tom Jeleniewski, Lasse Beers, Maximilian Weigand, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06714)  

**Abstract**: This paper presents a SysML profile that enables the direct integration of planning semantics based on the Planning Domain Definition Language (PDDL) into system models. Reusable stereotypes are defined for key PDDL concepts such as types, predicates, functions and actions, while formal OCL constraints ensure syntactic consistency. The profile was derived from the Backus-Naur Form (BNF) definition of PDDL 3.1 to align with SysML modeling practices. A case study from aircraft manufacturing demonstrates the application of the profile: a robotic system with interchangeable end effectors is modeled and enriched to generate both domain and problem descriptions in PDDL format. These are used as input to a PDDL solver to derive optimized execution plans. The approach supports automated and model-based generation of planning descriptions and provides a reusable bridge between system modeling and AI planning in engineering design. 

**Abstract (ZH)**: 本文提出了一种SysML配置文件，使其能够直接将基于Planning Domain Definition Language (PDDL) 的规划语义集成到系统模型中。定义了类型的可重用模型元，确保语义一致性，并通过形式化的OCL约束确保语法一致性。该配置文件从PDDL 3.1的Backus-Naur Form (BNF) 定义推导得出，与SysML建模实践一致。一个来自航空制造业的案例研究展示了该配置文件的应用：一个带有可更换末端执行器的机器人系统被建模并扩展，以生成PDDL格式的领域描述和问题描述。这些描述被用于PDDL求解器以推导出优化的执行计划。该方法支持自动化的、基于模型的规划描述生成，并在工程设计中提供了系统建模与AI规划之间可重用的桥梁。 

---
# GELD: A Unified Neural Model for Efficiently Solving Traveling Salesman Problems Across Different Scales 

**Title (ZH)**: GELD：一种高效解决不同规模 Travelling Salesman Problem 的统一神经模型 

**Authors**: Yubin Xiao, Di Wang, Rui Cao, Xuan Wu, Boyang Li, You Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06634)  

**Abstract**: The Traveling Salesman Problem (TSP) is a well-known combinatorial optimization problem with broad real-world applications. Recent advancements in neural network-based TSP solvers have shown promising results. Nonetheless, these models often struggle to efficiently solve both small- and large-scale TSPs using the same set of pre-trained model parameters, limiting their practical utility. To address this issue, we introduce a novel neural TSP solver named GELD, built upon our proposed broad global assessment and refined local selection framework. Specifically, GELD integrates a lightweight Global-view Encoder (GE) with a heavyweight Local-view Decoder (LD) to enrich embedding representation while accelerating the decision-making process. Moreover, GE incorporates a novel low-complexity attention mechanism, allowing GELD to achieve low inference latency and scalability to larger-scale TSPs. Additionally, we propose a two-stage training strategy that utilizes training instances of different sizes to bolster GELD's generalization ability. Extensive experiments conducted on both synthetic and real-world datasets demonstrate that GELD outperforms seven state-of-the-art models considering both solution quality and inference speed. Furthermore, GELD can be employed as a post-processing method to significantly elevate the quality of the solutions derived by existing neural TSP solvers via spending affordable additional computing time. Notably, GELD is shown as capable of solving TSPs with up to 744,710 nodes, first-of-its-kind to solve this large size TSP without relying on divide-and-conquer strategies to the best of our knowledge. 

**Abstract (ZH)**: 基于广域评估和精域选择的旅行推销员问题神经求解器GELD 

---
# The Optimization Paradox in Clinical AI Multi-Agent Systems 

**Title (ZH)**: 临床AI多代理系统中的优化悖论 

**Authors**: Suhana Bedi, Iddah Mlauzi, Daniel Shin, Sanmi Koyejo, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2506.06574)  

**Abstract**: Multi-agent artificial intelligence systems are increasingly deployed in clinical settings, yet the relationship between component-level optimization and system-wide performance remains poorly understood. We evaluated this relationship using 2,400 real patient cases from the MIMIC-CDM dataset across four abdominal pathologies (appendicitis, pancreatitis, cholecystitis, diverticulitis), decomposing clinical diagnosis into information gathering, interpretation, and differential diagnosis. We evaluated single agent systems (one model performing all tasks) against multi-agent systems (specialized models for each task) using comprehensive metrics spanning diagnostic outcomes, process adherence, and cost efficiency. Our results reveal a paradox: while multi-agent systems generally outperformed single agents, the component-optimized or Best of Breed system with superior components and excellent process metrics (85.5% information accuracy) significantly underperformed in diagnostic accuracy (67.7% vs. 77.4% for a top multi-agent system). This finding underscores that successful integration of AI in healthcare requires not just component level optimization but also attention to information flow and compatibility between agents. Our findings highlight the need for end to end system validation rather than relying on component metrics alone. 

**Abstract (ZH)**: 多智能体人工智能系统在临床环境中的应用日益增多，但组件级优化与系统级性能之间的关系尚不明确。我们利用MIMIC-CDM数据集中2400个实际患者案例，对四种腹腔疾病（阑尾炎、胰腺炎、胆囊炎、憩室炎）进行了评估，将临床诊断分解为信息收集、解释和鉴别诊断。我们使用涵盖诊断结果、流程遵守和成本效率的综合指标，评估了单智能体系统（一个模型执行所有任务）与多智能体系统（为每个任务专门化模型）的表现。研究结果揭示了一个悖论：尽管多智能体系统通常优于单智能体系统，但具有更优组件和卓越过程指标的优化组件系统（85.5%信息准确性）在诊断准确性方面的表现显著逊于顶级多智能体系统（67.7% vs. 77.4%）。这一发现表明，人工智能在医疗保健中的成功集成不仅需要组件级优化，还需要注意信息流和智能体之间的兼容性。我们的研究结果强调了需要端到端系统验证，而不仅仅是依赖组件指标。 

---
# Towards Foundation Model on Temporal Knowledge Graph Reasoning 

**Title (ZH)**: 面向时间知识图谱推理的础模型研究 

**Authors**: Jiaxin Pan, Mojtaba Nayyeri, Osama Mohammed, Daniel Hernandez, Rongchuan Zhang, Cheng Cheng, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2506.06367)  

**Abstract**: Temporal Knowledge Graphs (TKGs) store temporal facts with quadruple formats (s, p, o, t). Existing Temporal Knowledge Graph Embedding (TKGE) models perform link prediction tasks in transductive or semi-inductive settings, which means the entities, relations, and temporal information in the test graph are fully or partially observed during training. Such reliance on seen elements during inference limits the models' ability to transfer to new domains and generalize to real-world scenarios. A central limitation is the difficulty in learning representations for entities, relations, and timestamps that are transferable and not tied to dataset-specific vocabularies. To overcome these limitations, we introduce the first fully-inductive approach to temporal knowledge graph link prediction. Our model employs sinusoidal positional encodings to capture fine-grained temporal patterns and generates adaptive entity and relation representations using message passing conditioned on both local and global temporal contexts. Our model design is agnostic to temporal granularity and time span, effectively addressing temporal discrepancies across TKGs and facilitating time-aware structural information transfer. As a pretrained, scalable, and transferable model, POSTRA demonstrates strong zero-shot performance on unseen temporal knowledge graphs, effectively generalizing to novel entities, relations, and timestamps. Extensive theoretical analysis and empirical results show that a single pretrained model can improve zero-shot performance on various inductive temporal reasoning scenarios, marking a significant step toward a foundation model for temporal KGs. 

**Abstract (ZH)**: 时间知识图谱（TKGs）以四元组格式（s, p, o, t）存储时间事实。现有的时间知识图嵌入（TKGE）模型在transductive或semi-inductive设置下执行链接预测任务，这意味着在训练过程中测试图中的实体、关系和时间信息是完全或部分可观测的。这种依赖于已见元素的推理限制了模型向新领域转移和泛化到真实世界场景的能力。一个主要限制是难以学习转移性的实体、关系和时间戳表示，这些表示不依赖于特定数据集的词汇表。为了克服这些限制，我们首次引入了完全归纳的时间知识图谱链接预测方法。我们的模型使用正弦位置编码捕捉精细的时间模式，并通过本地和全局时间上下文条件下的消息传递生成自适应的实体和关系表示。我们的模型设计不依赖于时间粒度和时间跨度，有效地解决了时间知识图谱（TKGs）中的时间不一致问题，并促进了时间感知结构信息的转移。作为一种预训练、可扩展且可转移的模型，POSTRA在未见过的时间知识图谱上展示了强大的零样本性能，有效地泛化到新的实体、关系和时间戳。广泛的理论分析和实验证明，单个预训练模型可以改善各种归纳时序推理场景下的零样本性能，标志着朝着时间知识图谱基础模型的一个重要步骤。 

---
# Will artificial agents pursue power by default? 

**Title (ZH)**: 人工代理是否会默认追求权力？ 

**Authors**: Christian Tarsney  

**Link**: [PDF](https://arxiv.org/pdf/2506.06352)  

**Abstract**: Researchers worried about catastrophic risks from advanced AI have argued that we should expect sufficiently capable AI agents to pursue power over humanity because power is a convergent instrumental goal, something that is useful for a wide range of final goals. Others have recently expressed skepticism of these claims. This paper aims to formalize the concepts of instrumental convergence and power-seeking in an abstract, decision-theoretic framework, and to assess the claim that power is a convergent instrumental goal. I conclude that this claim contains at least an element of truth, but might turn out to have limited predictive utility, since an agent's options cannot always be ranked in terms of power in the absence of substantive information about the agent's final goals. However, the fact of instrumental convergence is more predictive for agents who have a good shot at attaining absolute or near-absolute power. 

**Abstract (ZH)**: 研究人员关于先进AI带来灾难性风险的担忧认为，我们应预期足够有能力的AI代理会追求对人类的权力，因为权力是一个趋同的工具性目标，对多种最终目标都有用。近期有观点对此表示怀疑。本文旨在通过抽象的决策理论框架来形式化工具性趋同和权力追求的概念，并评估权力是否是一个趋同的工具性目标的主张。我得出结论，该主张包含至少一个真实元素，但由于缺乏关于代理最终目标的实质信息，代理的选项并不总是可以通过权力来排序，因此可能具有有限的预测效用。然而，趋同性的事实对那些有可能获得绝对或近绝对权力的代理更具有预测性。 

---
# Mapping Human-Agent Co-Learning and Co-Adaptation: A Scoping Review 

**Title (ZH)**: 人类-代理共学习与共适应的映射：一项范围性文献综述 

**Authors**: Shruti Kumar, Xiaoyu Chen, Xiaomei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06324)  

**Abstract**: Several papers have delved into the challenges of human-AI-robot co-learning and co-adaptation. It has been noted that the terminology used to describe this collaborative relationship in existing studies needs to be more consistent. For example, the prefix "co" is used interchangeably to represent both "collaborative" and "mutual," and the terms "co-learning" and "co-adaptation" are sometimes used interchangeably. However, they can reflect subtle differences in the focus of the studies. The current scoping review's primary research question (RQ1) aims to gather existing papers discussing this collaboration pattern and examine the terms researchers use to describe this human-agent relationship. Given the relative newness of this area of study, we are also keen on exploring the specific types of intelligent agents and task domains that have been considered in existing research (RQ2). This exploration is significant as it can shed light on the diversity of human-agent interactions, from one-time to continuous learning/adaptation scenarios. It can also help us understand the dynamics of human-agent interactions in different task domains, guiding our expectations towards research situated in dynamic, complex domains. Our third objective (RQ3) is to investigate the cognitive theories and frameworks that have been utilized in existing studies to measure human-agent co-learning and co-adaptation. This investigation is crucial as it can help us understand the theoretical underpinnings of human-agent collaboration and adaptation, and it can also guide us in identifying any new frameworks proposed specifically for this type of relationship. 

**Abstract (ZH)**: Several篇论文探讨了人类-人工智能-机器人共同学习和适应的挑战。研究中用于描述这一合作关系的术语需要更加一致。例如，“co-”前缀既被用来表示“协作”也表示“相互”，同时，“共同学习”和“共同适应”这两个术语有时也被互换使用。然而，它们可以反映出研究重点的细微差别。本综述研究的主要研究问题（RQ1）旨在收集讨论这种合作模式的现有论文，并检查研究人员用来描述人机关系的术语。鉴于这一研究领域的相对新颖性，我们还希望通过研究现有研究中考虑的具体类型智能代理和任务领域（RQ2），来探索人机交互的多样性，从一次性学习/适应到连续学习/适应场景。这有助于我们理解不同任务领域中人机交互的动力学，指导我们在动态复杂领域中的研究预期。我们的第三个研究目标（RQ3）是调查现有研究中用于衡量人类-代理共同学习和共同适应的认知理论和框架。这一调查对于理解人机协作和适应的理论基础至关重要，也有助于我们识别任何为这种关系类型专门提出的新框架。 

---
# NFISiS: New Perspectives on Fuzzy Inference Systems for Renewable Energy Forecasting 

**Title (ZH)**: NFISiS: 新视角下的模糊 inference 系统在可再生能源预测中的应用 

**Authors**: Kaike Sa Teles Rocha Alves, Eduardo Pestana de Aguiar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06285)  

**Abstract**: Evolving Fuzzy Systems (eFS) have gained significant attention due to their ability to adaptively update their structure in response to data dynamics while maintaining interpretability. However, the lack of publicly available implementations of these models limits their accessibility and widespread adoption. To address this gap, we present evolvingfuzzysystems, a Python library that provides implementations of several well-established eFS models, including ePL-KRLS-DISCO, ePL+, eMG, ePL, exTS, Simpl\_eTS, and eTS. The library facilitates model evaluation and comparison by offering built-in tools for training, visualization, and performance assessment. The models are evaluated using the fetch\_california\_housing dataset, with performance measured in terms of normalized root-mean-square error (NRMSE), non-dimensional error index (NDEI), and mean absolute percentage error (MAPE). Additionally, computational complexity is analyzed by measuring execution times and rule evolution during training and testing phases. The results highlight ePL as a simple yet efficient model that balances accuracy and computational cost, making it particularly suitable for real-world applications. By making these models publicly available, evolvingfuzzysystems aims to foster research and practical applications in adaptive and interpretable machine learning. 

**Abstract (ZH)**: 自适应模糊系统(eFS)由于能够根据数据动态自适应地更新其结构并保持可解释性而引起了广泛关注。然而，缺少这些模型的公开实现限制了它们的可访问性和广泛应用。为了解决这个问题，我们介绍了evolvingfuzzysystemsPython库，该库提供了多个已被广泛认可的eFS模型的实现，包括ePL-KRLS-DISCO、ePL+、eMG、ePL、exTS、Simpl\_eTS和eTS。该库通过提供用于训练、可视化和性能评估的内置工具，促进了模型的评估与比较。模型使用fetch\_california\_housing数据集进行评估，性能用归一化均方根误差(NRMSE)、非量纲误差指数(NDEI)和平均绝对百分比误差(MAFE)来衡量。此外，通过测量训练和测试阶段的执行时间和规则演变，分析了计算复杂性。结果表明，ePL作为一种简单高效的模型，在准确性和计算成本之间取得了平衡，特别适合实际应用。通过使这些模型公开，evolvingfuzzysystems旨在促进自适应和可解释机器学习领域的研究和实际应用。 

---
# Unreal Patterns 

**Title (ZH)**: Unreal Patterns 

**Authors**: John Beverley, Jim Logan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06284)  

**Abstract**: This paper introduces a framework for representing information about entities that do not exist or may never exist, such as those involving fictional entities, blueprints, simulations, and future scenarios. Traditional approaches that introduce "dummy instances" or rely on modal logic are criticized, and a proposal is defended in which such cases are modeled using the intersections of actual types rather than specific non existent tokens. The paper positions itself within the Basic Formal Ontology and its realist commitments, emphasizing the importance of practical, implementable solutions over purely metaphysical or philosophical proposals, arguing that existing approaches to non existent entities either overcommit to metaphysical assumptions or introduce computational inefficiencies that hinder applications. By developing a structured ontology driven approach to unreal patterns, the paper aims to provide a useful and computationally viable means of handling references to hypothetical or non existent entities. 

**Abstract (ZH)**: 本文介绍了用于表示不存在或可能永远不会存在的实体信息的框架，这些实体涉及虚构实体、蓝图、模拟以及未来场景。批判了传统的引入“占位实例”或依赖模态逻辑的方法，并提出了一种将此类情况建模为实际类型交集而非特定不存在标记的方案。本文立足于基本正式本体及其现实承诺，强调实用可实施的解决方案的重要性，而非纯粹的形而上学或哲学提案，认为现有的对不存在实体的处理方法要么过度承诺了形而上学假设，要么引入了计算效率问题，阻碍了应用。通过发展一种结构化的本体驱动方法来处理不现实的模式，本文旨在提供一种实用且计算上可行的方法来处理对假设或不存在实体的引用。 

---
# StableMTL: Repurposing Latent Diffusion Models for Multi-Task Learning from Partially Annotated Synthetic Datasets 

**Title (ZH)**: StableMTL: 将潜在扩散模型重新应用于部分标注合成数据集的多任务学习 

**Authors**: Anh-Quan Cao, Ivan Lopes, Raoul de Charette  

**Link**: [PDF](https://arxiv.org/pdf/2506.08013)  

**Abstract**: Multi-task learning for dense prediction is limited by the need for extensive annotation for every task, though recent works have explored training with partial task labels. Leveraging the generalization power of diffusion models, we extend the partial learning setup to a zero-shot setting, training a multi-task model on multiple synthetic datasets, each labeled for only a subset of tasks. Our method, StableMTL, repurposes image generators for latent regression. Adapting a denoising framework with task encoding, per-task conditioning and a tailored training scheme. Instead of per-task losses requiring careful balancing, a unified latent loss is adopted, enabling seamless scaling to more tasks. To encourage inter-task synergy, we introduce a multi-stream model with a task-attention mechanism that converts N-to-N task interactions into efficient 1-to-N attention, promoting effective cross-task sharing. StableMTL outperforms baselines on 7 tasks across 8 benchmarks. 

**Abstract (ZH)**: 利用扩散模型的泛化能力，我们将部分学习扩展到零样本设置，通过训练一个多任务模型在多个合成数据集上工作，每个数据集仅标记少量任务。我们的方法StableMTL重新利用图像生成器进行隐变量回归，并适应去噪框架，采用任务编码、逐任务条件和定制的训练方案。我们采用统一的隐变量损失代替需要谨慎平衡的逐任务损失，从而可以无缝扩展到更多任务。为了促进任务间的协同作用，我们引入了一个多流模型，采用任务注意力机制将N-to-N任务交互转换为高效的1-to-N注意力，促进有效的跨任务共享。StableMTL在8个基准上的7个任务上优于基线方法。 

---
# ProtocolLLM: RTL Benchmark for SystemVerilog Generation of Communication Protocols 

**Title (ZH)**: ProtocolLLM：用于通信协议SystemVerilog生成的RTL基准测试bench 

**Authors**: Arnav Sheth, Ivaxi Sheth, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2506.07945)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown promising capabilities in generating code for general-purpose programming languages. In contrast, their applicability for hardware description languages, particularly for generating synthesizable and functionally correct designs, remains significantly underexplored. HDLs such as SystemVerilog are logic-oriented and demand strict adherence to timing semantics, concurrency, and synthesizability constraints. Moreover, HDL-based design flows encompass a broad set of tasks beyond structural code generation, including testbench development, assertion-based verification, timing closure, and protocol-level integration for on-chip communication. The objective of our paper is to analyze the capabilities of state-of-the-art LLMs in generating SystemVerilog implementations of standard communication protocols, a core component of embedded and System-on-Chip (SoC) architectures. This paper introduces the first benchmark suite targeting four widely used protocols: SPI, I2C, UART, and AXI. We define code generation tasks that capture varying levels of design abstraction and prompt specificity. The generated designs are assessed for syntactic correctness, synthesizability, and functional fidelity via waveform simulation and test benches. 

**Abstract (ZH)**: Recent Advances in Large Language Models in Generating SystemVerilog Implementations of Standard Communication Protocols 

---
# Diffusion of Responsibility in Collective Decision Making 

**Title (ZH)**: 集体决策中的责任扩散 

**Authors**: Pavel Naumov, Jia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07935)  

**Abstract**: The term "diffusion of responsibility'' refers to situations in which multiple agents share responsibility for an outcome, obscuring individual accountability. This paper examines this frequently undesirable phenomenon in the context of collective decision-making mechanisms.
The work shows that if a decision is made by two agents, then the only way to avoid diffusion of responsibility is for one agent to act as a "dictator'', making the decision unilaterally. In scenarios with more than two agents, any diffusion-free mechanism is an "elected dictatorship'' where the agents elect a single agent to make a unilateral decision.
The technical results are obtained by defining a bisimulation of decision-making mechanisms, proving that bisimulation preserves responsibility-related properties, and establishing the results for a smallest bisimular mechanism. 

**Abstract (ZH)**: 责任扩散现象是指多个代理共同承担某结果的责任，从而模糊了个体责任的情况。本文在集体决策机制的背景下探讨了这种通常令人不悦的现象。研究表明，如果决策由两个代理作出，则唯一的避免责任扩散的方法是一个代理充当“独裁者”，单独作出决策。在多于两个代理的情形下，任何无责任扩散的机制都是“选举独裁”，即代理们选举一位代理人单独作出决策。技术成果通过定义决策机制的拟模拟，并证明拟模拟保有一定的责任制性质，以及在最小拟模拟机制上建立结果而获得。 

---
# Uncovering the Functional Roles of Nonlinearity in Memory 

**Title (ZH)**: 揭示非线性在记忆中的功能作用 

**Authors**: Manuel Brenner, Georgia Koppe  

**Link**: [PDF](https://arxiv.org/pdf/2506.07919)  

**Abstract**: Memory and long-range temporal processing are core requirements for sequence modeling tasks across natural language processing, time-series forecasting, speech recognition, and control. While nonlinear recurrence has long been viewed as essential for enabling such mechanisms, recent work suggests that linear dynamics may often suffice. In this study, we go beyond performance comparisons to systematically dissect the functional role of nonlinearity in recurrent networks--identifying both when it is computationally necessary, and what mechanisms it enables. We use Almost Linear Recurrent Neural Networks (AL-RNNs), which allow fine-grained control over nonlinearity, as both a flexible modeling tool and a probe into the internal mechanisms of memory. Across a range of classic sequence modeling tasks and a real-world stimulus selection task, we find that minimal nonlinearity is not only sufficient but often optimal, yielding models that are simpler, more robust, and more interpretable than their fully nonlinear or linear counterparts. Our results provide a principled framework for selectively introducing nonlinearity, bridging dynamical systems theory with the functional demands of long-range memory and structured computation in recurrent neural networks, with implications for both artificial and biological neural systems. 

**Abstract (ZH)**: 记忆和长程时间处理是跨自然语言处理、时间序列预测、语音识别和控制领域的序列建模任务的核心要求。虽然非线性循环长期以来被认为是实现这些机制的关键，但近期研究表明线性动态往往已足以。在本研究中，我们超越性能比较，系统性地剖析循环网络中非线性功能的作用——确定它何时是计算上必要的，以及它所启用的机制。我们使用几乎线性循环神经网络（AL-RNN），既作为一种灵活的建模工具，也是探究记忆内部机制的探针。在一系列经典的序列建模任务和实际刺激选择任务中，我们发现最少的非线性不仅是足够的，而且往往是最佳的，生成的模型比完全非线性或线性的模型更为简单、稳健且易于解释。我们的结果提供了一个有原则的方法来有选择地引入非线性，将动力系统理论与循环神经网络中长程记忆和结构计算的功能需求联系起来，对人工和生物神经系统的均有启示。 

---
# Lightweight Sequential Transformers for Blood Glucose Level Prediction in Type-1 Diabetes 

**Title (ZH)**: 适用于1型糖尿病的轻量级顺序变换器血糖水平预测 

**Authors**: Mirko Paolo Barbato, Giorgia Rigamonti, Davide Marelli, Paolo Napoletano  

**Link**: [PDF](https://arxiv.org/pdf/2506.07864)  

**Abstract**: Type 1 Diabetes (T1D) affects millions worldwide, requiring continuous monitoring to prevent severe hypo- and hyperglycemic events. While continuous glucose monitoring has improved blood glucose management, deploying predictive models on wearable devices remains challenging due to computational and memory constraints. To address this, we propose a novel Lightweight Sequential Transformer model designed for blood glucose prediction in T1D. By integrating the strengths of Transformers' attention mechanisms and the sequential processing of recurrent neural networks, our architecture captures long-term dependencies while maintaining computational efficiency. The model is optimized for deployment on resource-constrained edge devices and incorporates a balanced loss function to handle the inherent data imbalance in hypo- and hyperglycemic events. Experiments on two benchmark datasets, OhioT1DM and DiaTrend, demonstrate that the proposed model outperforms state-of-the-art methods in predicting glucose levels and detecting adverse events. This work fills the gap between high-performance modeling and practical deployment, providing a reliable and efficient T1D management solution. 

**Abstract (ZH)**: Type 1糖尿病(T1D)影响着全球数百万人，需要持续监测以预防严重的低血糖和高血糖事件。尽管连续葡萄糖监测已经改善了血糖管理，但在可穿戴设备上部署预测模型仍面临计算和内存限制的挑战。为了解决这一问题，我们提出了一种新型轻量级序列变换器模型，适用于T1D的血糖预测。通过结合变换器注意力机制和递归神经网络的序列处理优势，该架构捕捉长时依赖关系同时保持计算效率。该模型针对资源受限的边缘设备进行了优化，并融入了平衡的损失函数以处理低血糖和高血糖事件之间固有的数据不平衡问题。在两个基准数据集OhioT1DM和DiaTrend上的实验表明，所提出的方法在预测血糖水平和检测不良事件方面优于现有最先进的方法。这项工作填补了高性能建模与实际部署之间的差距，提供了可靠和高效的T1D管理解决方案。 

---
# Fairness Overfitting in Machine Learning: An Information-Theoretic Perspective 

**Title (ZH)**: 机器学习中的公平过拟合：一种信息论视角 

**Authors**: Firas Laakom, Haobo Chen, Jürgen Schmidhuber, Yuheng Bu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07861)  

**Abstract**: Despite substantial progress in promoting fairness in high-stake applications using machine learning models, existing methods often modify the training process, such as through regularizers or other interventions, but lack formal guarantees that fairness achieved during training will generalize to unseen data. Although overfitting with respect to prediction performance has been extensively studied, overfitting in terms of fairness loss has received far less attention. This paper proposes a theoretical framework for analyzing fairness generalization error through an information-theoretic lens. Our novel bounding technique is based on Efron-Stein inequality, which allows us to derive tight information-theoretic fairness generalization bounds with both Mutual Information (MI) and Conditional Mutual Information (CMI). Our empirical results validate the tightness and practical relevance of these bounds across diverse fairness-aware learning algorithms. Our framework offers valuable insights to guide the design of algorithms improving fairness generalization. 

**Abstract (ZH)**: 尽管在促进高影响应用中机器学习模型的公平性方面取得了显著进展，现有方法往往通过正则化或其他干预措施修改训练过程，但在训练过程中实现的公平性缺乏正式保证，能够推广到未见过的数据。虽然对预测性能过拟合的研究甚多，但关于公平性损失过拟合的研究则较少。本文通过信息论视角提出了一种分析公平性泛化误差的理论框架。我们新颖的边界技术基于Efron-Stein不等式，使得我们能够使用互信息（MI）和条件互信息（CMI）推导出紧致的信息论公平性泛化边界。我们的实证结果验证了这些边界的紧致性和实际相关性，跨越了多种公平性感知学习算法。该框架为指导改进公平性泛化的算法设计提供了宝贵见解。 

---
# Residual Reweighted Conformal Prediction for Graph Neural Networks 

**Title (ZH)**: 图神经网络中的残差加权置信预测 

**Authors**: Zheng Zhang, Jie Bao, Zhixin Zhou, Nicolo Colombo, Lixin Cheng, Rui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07854)  

**Abstract**: Graph Neural Networks (GNNs) excel at modeling relational data but face significant challenges in high-stakes domains due to unquantified uncertainty. Conformal prediction (CP) offers statistical coverage guarantees, but existing methods often produce overly conservative prediction intervals that fail to account for graph heteroscedasticity and structural biases. While residual reweighting CP variants address some of these limitations, they neglect graph topology, cluster-specific uncertainties, and risk data leakage by reusing training sets. To address these issues, we propose Residual Reweighted GNN (RR-GNN), a framework designed to generate minimal prediction sets with provable marginal coverage guarantees.
RR-GNN introduces three major innovations to enhance prediction performance. First, it employs Graph-Structured Mondrian CP to partition nodes or edges into communities based on topological features, ensuring cluster-conditional coverage that reflects heterogeneity. Second, it uses Residual-Adaptive Nonconformity Scores by training a secondary GNN on a held-out calibration set to estimate task-specific residuals, dynamically adjusting prediction intervals according to node or edge uncertainty. Third, it adopts a Cross-Training Protocol, which alternates the optimization of the primary GNN and the residual predictor to prevent information leakage while maintaining graph dependencies. We validate RR-GNN on 15 real-world graphs across diverse tasks, including node classification, regression, and edge weight prediction. Compared to CP baselines, RR-GNN achieves improved efficiency over state-of-the-art methods, with no loss of coverage. 

**Abstract (ZH)**: 基于残差重权的图神经网络（RR-GNN）：生成具有可证明边缘覆盖保证的最小预测集 

---
# Diffusion models under low-noise regime 

**Title (ZH)**: 低噪声条件下扩散模型 

**Authors**: Elizabeth Pavlova, Xue-Xin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.07841)  

**Abstract**: Recent work on diffusion models proposed that they operate in two regimes: memorization, in which models reproduce their training data, and generalization, in which they generate novel samples. While this has been tested in high-noise settings, the behavior of diffusion models as effective denoisers when the corruption level is small remains unclear. To address this gap, we systematically investigated the behavior of diffusion models under low-noise diffusion dynamics, with implications for model robustness and interpretability. Using (i) CelebA subsets of varying sample sizes and (ii) analytic Gaussian mixture benchmarks, we reveal that models trained on disjoint data diverge near the data manifold even when their high-noise outputs converge. We quantify how training set size, data geometry, and model objective choice shape denoising trajectories and affect score accuracy, providing insights into how these models actually learn representations of data distributions. This work starts to address gaps in our understanding of generative model reliability in practical applications where small perturbations are common. 

**Abstract (ZH)**: 近期关于扩散模型的研究提出，它们在两种模式下运行：记忆模式，模型重现训练数据；泛化模式，生成新颖样本。尽管已经在高噪声条件下进行了测试，但在低噪声水平下扩散模型作为有效去噪器的行为尚不清楚。为了填补这一空白，我们系统地研究了在低噪声扩散动力学下扩散模型的行为，这对模型的鲁棒性和可解释性具有重要意义。通过使用(i) 不同样本大小的CelebA子集和(ii) 分析性的高斯混合基准，我们揭示了在训练数据不交集中训练的模型即使在高噪声输出收敛时，仍然会在数据流形附近发散。我们量化了训练集大小、数据几何形状和模型目标选择如何影响去噪轨迹和得分准确性，提供了这些模型是如何实际上学习数据分布表示的见解。这项工作开始弥补了我们对生成模型在实际应用中鲁棒性理解上的空白，特别是在小扰动常见的情况下。 

---
# Are Trees Really Green? A Detection Approach of IoT Malware Attacks 

**Title (ZH)**: 物联网 malware 攻击检测方法：树木really绿色吗？ 

**Authors**: Silvia Lucia Sanna, Diego Soi, Davide Maiorca, Giorgio Giacinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.07836)  

**Abstract**: Nowadays, the Internet of Things (IoT) is widely employed, and its usage is growing exponentially because it facilitates remote monitoring, predictive maintenance, and data-driven decision making, especially in the healthcare and industrial sectors. However, IoT devices remain vulnerable due to their resource constraints and difficulty in applying security patches. Consequently, various cybersecurity attacks are reported daily, such as Denial of Service, particularly in IoT-driven solutions. Most attack detection methodologies are based on Machine Learning (ML) techniques, which can detect attack patterns. However, the focus is more on identification rather than considering the impact of ML algorithms on computational resources. This paper proposes a green methodology to identify IoT malware networking attacks based on flow privacy-preserving statistical features. In particular, the hyperparameters of three tree-based models -- Decision Trees, Random Forest and Extra-Trees -- are optimized based on energy consumption and test-time performance in terms of Matthew's Correlation Coefficient. Our results show that models maintain high performance and detection accuracy while consistently reducing power usage in terms of watt-hours (Wh). This suggests that on-premise ML-based Intrusion Detection Systems are suitable for IoT and other resource-constrained devices. 

**Abstract (ZH)**: 物联网设备流量隐私保护统计特征下的绿色恶意软件网络攻击检测方法 

---
# Decentralizing Multi-Agent Reinforcement Learning with Temporal Causal Information 

**Title (ZH)**: 基于时间因果信息的多智能体强化学习去中心化方法 

**Authors**: Jan Corazza, Hadi Partovi Aria, Hyohun Kim, Daniel Neider, Zhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07829)  

**Abstract**: Reinforcement learning (RL) algorithms can find an optimal policy for a single agent to accomplish a particular task. However, many real-world problems require multiple agents to collaborate in order to achieve a common goal. For example, a robot executing a task in a warehouse may require the assistance of a drone to retrieve items from high shelves. In Decentralized Multi-Agent RL (DMARL), agents learn independently and then combine their policies at execution time, but often must satisfy constraints on compatibility of local policies to ensure that they can achieve the global task when combined. In this paper, we study how providing high-level symbolic knowledge to agents can help address unique challenges of this setting, such as privacy constraints, communication limitations, and performance concerns. In particular, we extend the formal tools used to check the compatibility of local policies with the team task, making decentralized training with theoretical guarantees usable in more scenarios. Furthermore, we empirically demonstrate that symbolic knowledge about the temporal evolution of events in the environment can significantly expedite the learning process in DMARL. 

**Abstract (ZH)**: 利用高层符号知识解决分布式多智能体强化学习的独特挑战 

---
# Accelerating Diffusion Models in Offline RL via Reward-Aware Consistency Trajectory Distillation 

**Title (ZH)**: 在离线RL中通过奖励意识一致性轨迹蒸馏加速扩散模型 

**Authors**: Xintong Duan, Yutong He, Fahim Tajwar, Ruslan Salakhutdinov, J. Zico Kolter, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2506.07822)  

**Abstract**: Although diffusion models have achieved strong results in decision-making tasks, their slow inference speed remains a key limitation. While the consistency model offers a potential solution, its applications to decision-making often struggle with suboptimal demonstrations or rely on complex concurrent training of multiple networks. In this work, we propose a novel approach to consistency distillation for offline reinforcement learning that directly incorporates reward optimization into the distillation process. Our method enables single-step generation while maintaining higher performance and simpler training. Empirical evaluations on the Gym MuJoCo benchmarks and long horizon planning demonstrate that our approach can achieve an 8.7% improvement over previous state-of-the-art while offering up to 142x speedup over diffusion counterparts in inference time. 

**Abstract (ZH)**: 尽管扩散模型在决策任务中取得了强大的成果，但其缓慢的推理速度仍然是一个关键限制。虽然一致性模型提供了一种潜在的解决方案，但在决策任务中的应用往往面临亚优演示或依赖多个网络的复杂并发训练的问题。在本文中，我们提出了一种新的离线强化学习中一致性的蒸馏方法，该方法直接将奖励优化纳入蒸馏过程。该方法能够在保持更高性能和更简单训练的同时实现单步生成。在Gym MuJoCo基准测试和长时规划上的实证评估表明，我们的方法在推理时间上比扩散模型 counterparts 提供高达142倍的速度提升，同时相比之前最先进的方法可实现8.7%的性能提升。 

---
# Enhancing Adversarial Robustness with Conformal Prediction: A Framework for Guaranteed Model Reliability 

**Title (ZH)**: 基于区间预测的 adversarial  robustness 提升框架：一种保障模型可靠性的方法 

**Authors**: Jie Bao, Chuangyin Dang, Rui Luo, Hanwei Zhang, Zhixin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07804)  

**Abstract**: As deep learning models are increasingly deployed in high-risk applications, robust defenses against adversarial attacks and reliable performance guarantees become paramount. Moreover, accuracy alone does not provide sufficient assurance or reliable uncertainty estimates for these models. This study advances adversarial training by leveraging principles from Conformal Prediction. Specifically, we develop an adversarial attack method, termed OPSA (OPtimal Size Attack), designed to reduce the efficiency of conformal prediction at any significance level by maximizing model uncertainty without requiring coverage guarantees. Correspondingly, we introduce OPSA-AT (Adversarial Training), a defense strategy that integrates OPSA within a novel conformal training paradigm. Experimental evaluations demonstrate that our OPSA attack method induces greater uncertainty compared to baseline approaches for various defenses. Conversely, our OPSA-AT defensive model significantly enhances robustness not only against OPSA but also other adversarial attacks, and maintains reliable prediction. Our findings highlight the effectiveness of this integrated approach for developing trustworthy and resilient deep learning models for safety-critical domains. Our code is available at this https URL. 

**Abstract (ZH)**: 随着深度学习模型在高风险应用中的日益部署，对抗攻击的稳健防御和可靠性能保证变得至关重要。此外，单纯依赖准确性不足以提供这些模型的充分保证或可靠的不确定性估计。本研究通过结合校准预测原则推进了对抗训练。具体而言，我们开发了一种称为OPSA（OPtimal Size Attack）的对抗攻击方法，旨在通过最大化模型不确定性来降低任何显著水平下的校准预测效率，而无需提供覆盖保证。相应地，我们引入了OPSA-AT（对抗训练）防御策略，该策略将OPSA集成到一个新颖的校准训练框架中。实验评估表明，与基线方法相比，我们的OPSA攻击方法在各种防御措施下引发了更高的不确定性。相反，我们的OPSA-AT防御模型不仅显著增强了对抗OPSA以及其他对抗攻击的鲁棒性，还保持了可靠的预测性能。我们的研究结果突显了这种集成方法在开发适用于关键安全领域的可信赖且稳健的深度学习模型方面的有效性。我们的代码可在以下链接获取。 

---
# MultiMatch: Multihead Consistency Regularization Matching for Semi-Supervised Text Classification 

**Title (ZH)**: MultiMatch: 多头一致性正则化匹配在半监督文本分类中的应用 

**Authors**: Iustin Sirbu, Robert-Adrian Popovici, Cornelia Caragea, Stefan Trausan-Matu, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2506.07801)  

**Abstract**: We introduce MultiMatch, a novel semi-supervised learning (SSL) algorithm combining the paradigms of co-training and consistency regularization with pseudo-labeling. At its core, MultiMatch features a three-fold pseudo-label weighting module designed for three key purposes: selecting and filtering pseudo-labels based on head agreement and model confidence, and weighting them according to the perceived classification difficulty. This novel module enhances and unifies three existing techniques -- heads agreement from Multihead Co-training, self-adaptive thresholds from FreeMatch, and Average Pseudo-Margins from MarginMatch -- resulting in a holistic approach that improves robustness and performance in SSL settings. Experimental results on benchmark datasets highlight the superior performance of MultiMatch, achieving state-of-the-art results on 9 out of 10 setups from 5 natural language processing datasets and ranking first according to the Friedman test among 19 methods. Furthermore, MultiMatch demonstrates exceptional robustness in highly imbalanced settings, outperforming the second-best approach by 3.26% -- and data imbalance is a key factor for many text classification tasks. 

**Abstract (ZH)**: MultiMatch: 结合共训练和一致性正则化的新型半监督学习算法 

---
# Comparing Credit Risk Estimates in the Gen-AI Era 

**Title (ZH)**: Gen-AI时代下的信用风险估计比较 

**Authors**: Nicola Lavecchia, Sid Fadanelli, Federico Ricciuti, Gennaro Aloe, Enrico Bagli, Pietro Giuffrida, Daniele Vergari  

**Link**: [PDF](https://arxiv.org/pdf/2506.07754)  

**Abstract**: Generative AI technologies have demonstrated significant potential across diverse applications. This study provides a comparative analysis of credit score modeling techniques, contrasting traditional approaches with those leveraging generative AI. Our findings reveal that current generative AI models fall short of matching the performance of traditional methods, regardless of the integration strategy employed. These results highlight the limitations in the current capabilities of generative AI for credit risk scoring, emphasizing the need for further research and development before the possibility of applying generative AI for this specific task, or equivalent ones. 

**Abstract (ZH)**: 生成式AI技术在多种应用中展示了显著潜力。本研究对比分析了信用评分建模技术，将传统方法与利用生成式AI的方法进行了对比。研究发现，当前的生成式AI模型在性能上仍不及传统方法，无论采用何种集成策略。这些结果突显了当前生成式AI在信用风险评分能力上的局限性，强调在将其应用于此类特定任务之前需要进一步的研究与开发。 

---
# ArchiLense: A Framework for Quantitative Analysis of Architectural Styles Based on Vision Large Language Models 

**Title (ZH)**: Archilense：一种基于视觉大规模语言模型的建筑风格定量分析框架 

**Authors**: Jing Zhong, Jun Yin, Peilin Li, Pengyu Zeng, Miao Zhang, Shuai Lu, Ran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07739)  

**Abstract**: Architectural cultures across regions are characterized by stylistic diversity, shaped by historical, social, and technological contexts in addition to geograph-ical conditions. Understanding architectural styles requires the ability to describe and analyze the stylistic features of different architects from various regions through visual observations of architectural imagery. However, traditional studies of architectural culture have largely relied on subjective expert interpretations and historical literature reviews, often suffering from regional biases and limited ex-planatory scope. To address these challenges, this study proposes three core contributions: (1) We construct a professional architectural style dataset named ArchDiffBench, which comprises 1,765 high-quality architectural images and their corresponding style annotations, collected from different regions and historical periods. (2) We propose ArchiLense, an analytical framework grounded in Vision-Language Models and constructed using the ArchDiffBench dataset. By integrating ad-vanced computer vision techniques, deep learning, and machine learning algo-rithms, ArchiLense enables automatic recognition, comparison, and precise classi-fication of architectural imagery, producing descriptive language outputs that ar-ticulate stylistic differences. (3) Extensive evaluations show that ArchiLense achieves strong performance in architectural style recognition, with a 92.4% con-sistency rate with expert annotations and 84.5% classification accuracy, effec-tively capturing stylistic distinctions across images. The proposed approach transcends the subjectivity inherent in traditional analyses and offers a more objective and accurate perspective for comparative studies of architectural culture. 

**Abstract (ZH)**: 地区之间的建筑文化表现为风格多样性，这些风格由历史、社会和技术背景以及地理条件共同塑造。理解建筑风格需要通过建筑图像的视觉观察来描述和分析不同地区建筑师的不同风格特征。然而，传统的建筑文化研究大多依赖于主观专家解释和历史文献回顾，往往存在地域偏见和解释范围有限的问题。为了应对这些挑战，本研究提出了三个核心贡献：（1）我们构建了一个名为ArchDiffBench的专业建筑风格数据集，包含来自不同地区和历史时期的1,765张高质量的建筑图像及其相应的风格注解。（2）我们提出了一种基于视觉-语言模型的分析框架ArchLense，并基于ArchDiffBench数据集构建。通过集成先进的计算机视觉技术、深度学习和机器学习算法，ArchLense能够自动识别、比较和精确分类建筑图像，并生成描述风格差异的语言输出。（3）广泛的评估表明，ArchLense在建筑风格识别方面表现出色，与专家注解的92.4%一致性和84.5%分类准确率，有效地捕捉了图像之间的风格差异。所提出的方法超越了传统分析中的主观性，为建筑文化的比较研究提供了更为客观和准确的视角。 

---
# FMaMIL: Frequency-Driven Mamba Multi-Instance Learning for Weakly Supervised Lesion Segmentation in Medical Images 

**Title (ZH)**: FMaMIL：基于频率的Mamba多实例学习在医学图像弱监督病灶分割中的应用 

**Authors**: Hangbei Cheng, Xiaorong Dong, Xueyu Liu, Jianan Zhang, Xuetao Ma, Mingqiang Wei, Liansheng Wang, Junxin Chen, Yongfei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07652)  

**Abstract**: Accurate lesion segmentation in histopathology images is essential for diagnostic interpretation and quantitative analysis, yet it remains challenging due to the limited availability of costly pixel-level annotations. To address this, we propose FMaMIL, a novel two-stage framework for weakly supervised lesion segmentation based solely on image-level labels. In the first stage, a lightweight Mamba-based encoder is introduced to capture long-range dependencies across image patches under the MIL paradigm. To enhance spatial sensitivity and structural awareness, we design a learnable frequency-domain encoding module that supplements spatial-domain features with spectrum-based information. CAMs generated in this stage are used to guide segmentation training. In the second stage, we refine the initial pseudo labels via a CAM-guided soft-label supervision and a self-correction mechanism, enabling robust training even under label noise. Extensive experiments on both public and private histopathology datasets demonstrate that FMaMIL outperforms state-of-the-art weakly supervised methods without relying on pixel-level annotations, validating its effectiveness and potential for digital pathology applications. 

**Abstract (ZH)**: 准确的病理图像病变分割对于诊断解释和定量分析至关重要，但由于像素级注解成本高且获取有限，这一任务仍具有挑战性。为解决这一问题，我们提出了一种基于图像级标签的新型两阶段弱监督病变分割框架FMaMIL。在第一阶段，引入基于Mamba的编码器在MIL范式下捕捉图像patches间的长程依赖关系。为了增强空间敏感性和结构意识，设计了一种可学习的频域编码模块，通过频谱信息补充空域特征。该阶段生成的CAM用于指导分割训练。在第二阶段，通过CAM引导的软标签监督和自我纠正机制细化初始伪标签，即使在标签噪声下也能实现稳健训练。在公共和私有病理图像数据集上的 extensive 实验表明，FMaMIL 在无需依赖像素级注解的情况下超过了最先进的弱监督方法，验证了其在数字病理学应用中的有效性和潜力。 

---
# PolitiSky24: U.S. Political Bluesky Dataset with User Stance Labels 

**Title (ZH)**: PolitiSky24: 美国政治 bluesky 数据集及其用户立场标签 

**Authors**: Peyman Rostami, Vahid Rahimzadeh, Ali Adibi, Azadeh Shakery  

**Link**: [PDF](https://arxiv.org/pdf/2506.07606)  

**Abstract**: Stance detection identifies the viewpoint expressed in text toward a specific target, such as a political figure. While previous datasets have focused primarily on tweet-level stances from established platforms, user-level stance resources, especially on emerging platforms like Bluesky remain scarce. User-level stance detection provides a more holistic view by considering a user's complete posting history rather than isolated posts. We present the first stance detection dataset for the 2024 U.S. presidential election, collected from Bluesky and centered on Kamala Harris and Donald Trump. The dataset comprises 16,044 user-target stance pairs enriched with engagement metadata, interaction graphs, and user posting histories. PolitiSky24 was created using a carefully evaluated pipeline combining advanced information retrieval and large language models, which generates stance labels with supporting rationales and text spans for transparency. The labeling approach achieves 81\% accuracy with scalable LLMs. This resource addresses gaps in political stance analysis through its timeliness, open-data nature, and user-level perspective. The dataset is available at this https URL 

**Abstract (ZH)**: 基于 Bluesky 的 2024 美国总统选举立场检测数据集：以卡玛拉·哈里斯和唐纳德·特朗普为中心 

---
# FedCGD: Collective Gradient Divergence Optimized Scheduling for Wireless Federated Learning 

**Title (ZH)**: FedCGD：集体梯度散度优化调度的无线联邦学习 

**Authors**: Tan Chen, Jintao Yan, Yuxuan Sun, Sheng Zhou, Zhisheng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07581)  

**Abstract**: Federated learning (FL) is a promising paradigm for multiple devices to cooperatively train a model. When applied in wireless networks, two issues consistently affect the performance of FL, i.e., data heterogeneity of devices and limited bandwidth. Many papers have investigated device scheduling strategies considering the two issues. However, most of them recognize data heterogeneity as a property of individual devices. In this paper, we prove that the convergence speed of FL is affected by the sum of device-level and sample-level collective gradient divergence (CGD). The device-level CGD refers to the gradient divergence of the scheduled device group, instead of the sum of the individual device divergence. The sample-level CGD is statistically upper bounded by sampling variance, which is inversely proportional to the total number of samples scheduled for local update. To derive a tractable form of the device-level CGD, we further consider a classification problem and transform it into the weighted earth moving distance (WEMD) between the group distribution and the global distribution. Then we propose FedCGD algorithm to minimize the sum of multi-level CGDs by balancing WEMD and sampling variance, within polynomial time. Simulation shows that the proposed strategy increases classification accuracy on the CIFAR-10 dataset by up to 4.2\% while scheduling 41.8\% fewer devices, and flexibly switches between reducing WEMD and reducing sampling variance. 

**Abstract (ZH)**: 联邦学习中的设备调度策略：考虑设备级和样本级集体梯度发散的优化方法 

---
# Denoising the Future: Top-p Distributions for Moving Through Time 

**Title (ZH)**: 未来去噪：Moving Through Time中的top-p分布 

**Authors**: Florian Andreas Marwitz, Ralf Möller, Magnus Bender, Marcel Gehrke  

**Link**: [PDF](https://arxiv.org/pdf/2506.07578)  

**Abstract**: Inference in dynamic probabilistic models is a complex task involving expensive operations. In particular, for Hidden Markov Models, the whole state space has to be enumerated for advancing in time. Even states with negligible probabilities are considered, resulting in computational inefficiency and increased noise due to the propagation of unlikely probability mass. We propose to denoise the future and speed up inference by using only the top-p states, i.e., the most probable states with accumulated probability p. We show that the error introduced by using only the top-p states is bound by p and the so-called minimal mixing rate of the underlying model. Moreover, in our empirical evaluation, we show that we can expect speedups of at least an order of magnitude, while the error in terms of total variation distance is below 0.09. 

**Abstract (ZH)**: 动态概率模型中的推断是一个复杂任务，涉及昂贵的操作。特别是对于隐藏马尔可夫模型，必须遍历整个状态空间以实现时间推进。即使是概率可以忽略不计的状态也予以考虑，导致计算效率低下并增加由于 unlikely 概率质量传播造成的噪声。我们提出通过仅使用最 probable 的 top-p 状态来降噪和加速推断。结果显示，仅使用 top-p 状态引入的误差受 p 和底层模型的最小混合率限制。此外，在我们的实证评估中，我们表明可以期望至少一个数量级的加速，同时在总变差距离意义上的误差低于 0.09。 

---
# MoE-MLoRA for Multi-Domain CTR Prediction: Efficient Adaptation with Expert Specialization 

**Title (ZH)**: MoE-MLoRA在多领域点击率预测中的高效适应：专家专业化精化 

**Authors**: Ken Yagel, Eyal German, Aviel Ben Siman Tov  

**Link**: [PDF](https://arxiv.org/pdf/2506.07563)  

**Abstract**: Personalized recommendation systems must adapt to user interactions across different domains. Traditional approaches like MLoRA apply a single adaptation per domain but lack flexibility in handling diverse user behaviors. To address this, we propose MoE-MLoRA, a mixture-of-experts framework where each expert is first trained independently to specialize in its domain before a gating network is trained to weight their contributions dynamically. We evaluate MoE-MLoRA across eight CTR models on Movielens and Taobao, showing that it improves performance in large-scale, dynamic datasets (+1.45 Weighed-AUC in Taobao-20) but offers limited benefits in structured datasets with low domain diversity and sparsity. Further analysis of the number of experts per domain reveals that larger ensembles do not always improve performance, indicating the need for model-aware tuning. Our findings highlight the potential of expert-based architectures for multi-domain recommendation systems, demonstrating that task-aware specialization and adaptive gating can enhance predictive accuracy in complex environments. The implementation and code are available in our GitHub repository. 

**Abstract (ZH)**: 个性化推荐系统必须适应用户在不同领域的交互。传统的MLoRA等方法在每个领域仅应用单一的适应性，缺乏处理多样化用户行为的灵活性。为了解决这一问题，我们提出了一种专家混合框架MoE-MLoRA，在此框架中，每个专家首先在独立训练中 specialize 于其所在领域，之后训练一个门控网络以动态加权其各部分的贡献。我们在 Movielens 和 Taobao 上的八种点击率模型上评估了 MoE-MLoRA，结果显示，它在大规模动态数据集上提高了性能（Taobao-20 上的 Weighed-AUC 提高了 1.45），但在域多样性低且数据稀疏的结构化数据集上提供了有限的益处。进一步分析每个领域的专家数量表明，更大的集成模型并不总是提高性能，表明需要进行模型感知的调优。我们的研究结果强调了基于专家架构在多域推荐系统中的潜力，表明任务感知的专业化和自适应门控能够增强复杂环境中的预测准确性。相关实现和代码已在我们的 GitHub 仓库中提供。 

---
# Synthesize Privacy-Preserving High-Resolution Images via Private Textual Intermediaries 

**Title (ZH)**: 通过私有文本中介合成隐私保护高分辨率图像 

**Authors**: Haoxiang Wang, Zinan Lin, Da Yu, Huishuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07555)  

**Abstract**: Generating high fidelity, differentially private (DP) synthetic images offers a promising route to share and analyze sensitive visual data without compromising individual privacy. However, existing DP image synthesis methods struggle to produce high resolution outputs that faithfully capture the structure of the original data. In this paper, we introduce a novel method, referred to as Synthesis via Private Textual Intermediaries (SPTI), that can generate high resolution DP images with easy adoption. The key idea is to shift the challenge of DP image synthesis from the image domain to the text domain by leveraging state of the art DP text generation methods. SPTI first summarizes each private image into a concise textual description using image to text models, then applies a modified Private Evolution algorithm to generate DP text, and finally reconstructs images using text to image models. Notably, SPTI requires no model training, only inference with off the shelf models. Given a private dataset, SPTI produces synthetic images of substantially higher quality than prior DP approaches. On the LSUN Bedroom dataset, SPTI attains an FID less than or equal to 26.71 under epsilon equal to 1.0, improving over Private Evolution FID of 40.36. Similarly, on MM CelebA HQ, SPTI achieves an FID less than or equal to 33.27 at epsilon equal to 1.0, compared to 57.01 from DP fine tuning baselines. Overall, our results demonstrate that Synthesis via Private Textual Intermediaries provides a resource efficient and proprietary model compatible framework for generating high resolution DP synthetic images, greatly expanding access to private visual datasets. 

**Abstract (ZH)**: 生成高保真差分隐私(DP)合成图像提供了不泄露个体隐私前提下共享和分析敏感视觉数据的有希望途径。然而，现有的DP图像合成方法难以生成高分辨率且忠实再现原始数据结构的输出。在本文中，我们提出了一种新颖的方法，称为通过私有文本中介合成（SPTI），可以生成高分辨率的DP图像且易于采用。核心思想是通过利用最先进的DP文本生成方法，将DP图像合成的挑战从图像域转移到文本域。SPTI 首先使用图像到文本模型将每张私人图像总结为简洁的文本描述，然后应用修改过的私有进化算法生成DP文本，并最终使用文本到图像模型重建图像。值得注意的是，SPTI 不需要模型训练，仅需使用现成的模型进行推断。给定一个私人数据集，SPTI 生成的合成图像的质量显著高于先前的DP方法。在LSUN Bedroom数据集上，当ε=1.0时，SPTI 的FID小于或等于26.71，优于私有进化方法的FID 40.36。同样，在MM CelebA HQ 上，当 ε=1.0 时，SPTI 达到的 FID 小于或等于33.27，而 DP 微调基线的 FID 为 57.01。总体而言，我们的结果表明，通过私有文本中介合成提供了一种资源高效且与专有模型兼容的框架，用于生成高分辨率的DP合成图像，极大地扩展了对私人视觉数据集的访问。 

---
# CoCoA-Mix: Confusion-and-Confidence-Aware Mixture Model for Context Optimization 

**Title (ZH)**: CoCoA-Mix: 混淆与自信-aware 混合模型用于情境优化 

**Authors**: Dasol Hong, Wooju Lee, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2506.07484)  

**Abstract**: Prompt tuning, which adapts vision-language models by freezing model parameters and optimizing only the prompt, has proven effective for task-specific adaptations. The core challenge in prompt tuning is improving specialization for a specific task and generalization for unseen domains. However, frozen encoders often produce misaligned features, leading to confusion between classes and limiting specialization. To overcome this issue, we propose a confusion-aware loss (CoA-loss) that improves specialization by refining the decision boundaries between confusing classes. Additionally, we mathematically demonstrate that a mixture model can enhance generalization without compromising specialization. This is achieved using confidence-aware weights (CoA-weights), which adjust the weights of each prediction in the mixture model based on its confidence within the class domains. Extensive experiments show that CoCoA-Mix, a mixture model with CoA-loss and CoA-weights, outperforms state-of-the-art methods by enhancing specialization and generalization. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 提示调优通过冻结模型参数并仅优化提示，已证明对任务特定适应有效。提示调优的核心挑战是提高对特定任务的专业化程度并增强对未见过领域的泛化能力。然而，冻结的编码器往往会产生对齐不良的特征，导致类之间的混淆并限制专业化程度。为克服这一问题，我们提出了一种意识混淆损失（CoA-loss），通过细化混淆类之间的决策边界来提高专业化程度。此外，我们从数学上证明，混合模型可以在不牺牲专业化的情况下增强泛化能力。这是通过使用意识置信权重（CoA-weights）实现的，这些权重根据其在类域内的置信度调整混合模型中每个预测的权重。广泛实验表明，使用CoA-loss和CoA-weights的CoCoA-Mix混合模型相比现有最佳方法，在提高专业化和泛化方面表现出更优的效果。我们的代码在此处公开：这个链接。 

---
# Premise Selection for a Lean Hammer 

**Title (ZH)**: 精益锤 premise 选择 

**Authors**: Thomas Zhu, Joshua Clune, Jeremy Avigad, Albert Qiaochu Jiang, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2506.07477)  

**Abstract**: Neural methods are transforming automated reasoning for proof assistants, yet integrating these advances into practical verification workflows remains challenging. Hammers are tools that interface with external automatic theorem provers to automate tedious reasoning steps. They have dramatically improved productivity in proof assistants, but the Lean proof assistant still does not have a hammer despite its growing popularity. We present LeanHammer, the first end-to-end domain-general hammer for Lean, built on a novel neural premise selection system for a hammer in dependent type theory. Unlike existing Lean premise selectors, our approach dynamically adapts to user-specific contexts and combines with symbolic proof search and reconstruction to create a practical hammer. With comprehensive evaluations, we show that our premise selector enables LeanHammer to solve 21\% more goals relative to existing premise selectors, and generalize well to diverse domains. Our work bridges the gap between neural retrieval and symbolic reasoning, making formal verification more accessible to researchers and practitioners. 

**Abstract (ZH)**: 神经方法正在Transforming自动化证明助手中的自动推理，但在将这些进步集成到实际验证工作流中仍然面临挑战。Hammer是与外部自动定理证明器接口的工具，用于自动化繁琐的推理步骤。它们显著提高了证明助手的 productivity，但Lean证明助手仍然缺乏Hammer，尽管其 popularity正在增长。我们提出了LeanHammer，这是第一个基于新型神经前提选择系统的通用Hammer，适用于依赖类型理论。与现有的Lean前提选择器不同，我们的方法动态适应用户特定的上下文，并结合符号证明搜索和重建，创建了一个实用的Hammer。通过全面的评估，我们展示我们的前提选择器使LeanHammer相较现有的前提选择器能够解决多21%的目标，并且在不同领域具有良好的泛化能力。我们的工作在神经检索和符号推理之间架起了桥梁，使形式验证对研究人员和实践者更加 accessible。 

---
# Ambiguity-Restrained Text-Video Representation Learning for Partially Relevant Video Retrieval 

**Title (ZH)**: 限制歧义的文本-视频表示学习以实现部分相关视频检索 

**Authors**: CH Cho, WJ Moon, W Jun, MS Jung, JP Heo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07471)  

**Abstract**: Partially Relevant Video Retrieval~(PRVR) aims to retrieve a video where a specific segment is relevant to a given text query. Typical training processes of PRVR assume a one-to-one relationship where each text query is relevant to only one video. However, we point out the inherent ambiguity between text and video content based on their conceptual scope and propose a framework that incorporates this ambiguity into the model learning process. Specifically, we propose Ambiguity-Restrained representation Learning~(ARL) to address ambiguous text-video pairs. Initially, ARL detects ambiguous pairs based on two criteria: uncertainty and similarity. Uncertainty represents whether instances include commonly shared context across the dataset, while similarity indicates pair-wise semantic overlap. Then, with the detected ambiguous pairs, our ARL hierarchically learns the semantic relationship via multi-positive contrastive learning and dual triplet margin loss. Additionally, we delve into fine-grained relationships within the video instances. Unlike typical training at the text-video level, where pairwise information is provided, we address the inherent ambiguity within frames of the same untrimmed video, which often contains multiple contexts. This allows us to further enhance learning at the text-frame level. Lastly, we propose cross-model ambiguity detection to mitigate the error propagation that occurs when a single model is employed to detect ambiguous pairs for its training. With all components combined, our proposed method demonstrates its effectiveness in PRVR. 

**Abstract (ZH)**: 部分相关视频检索：含模糊性的视频表示学习框架 

---
# Fast Geometric Embedding for Node Influence Maximization 

**Title (ZH)**: 快速几何嵌入节点影响最大化 

**Authors**: Alexander Kolpakov, Igor Rivin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07435)  

**Abstract**: Computing classical centrality measures such as betweenness and closeness is computationally expensive on large-scale graphs. In this work, we introduce an efficient force layout algorithm that embeds a graph into a low-dimensional space, where the radial distance from the origin serves as a proxy for various centrality measures. We evaluate our method on multiple graph families and demonstrate strong correlations with degree, PageRank, and paths-based centralities. As an application, it turns out that the proposed embedding allows to find high-influence nodes in a network, and provides a fast and scalable alternative to the standard greedy algorithm. 

**Abstract (ZH)**: 在大规模图上计算经典的中心性度量（如介数和接近中心性）非常耗费计算资源。本文介绍了一种高效的力量布局算法，将图嵌入到低维空间中，其中从原点的径向距离作为各种中心性度量的代理。我们在多个图家族上评估了该方法，并证明了其与度、PageRank和路径中心性的强相关性。作为一种应用，发现所提出的嵌入允许在网络中找到高影响力节点，并提供了一种比标准贪婪算法更快且更具扩展性的替代方案。 

---
# Evidential Spectrum-Aware Contrastive Learning for OOD Detection in Dynamic Graphs 

**Title (ZH)**: 基于证据谱的动态图异常节点检测对比学习 

**Authors**: Nan Sun, Xixun Lin, Zhiheng Zhou, Yanmin Shang, Zhenlin Cheng, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07417)  

**Abstract**: Recently, Out-of-distribution (OOD) detection in dynamic graphs, which aims to identify whether incoming data deviates from the distribution of the in-distribution (ID) training set, has garnered considerable attention in security-sensitive fields. Current OOD detection paradigms primarily focus on static graphs and confront two critical challenges: i) high bias and high variance caused by single-point estimation, which makes the predictions sensitive to randomness in the data; ii) score homogenization resulting from the lack of OOD training data, where the model only learns ID-specific patterns, resulting in overall low OOD scores and a narrow score gap between ID and OOD data. To tackle these issues, we first investigate OOD detection in dynamic graphs through the lens of Evidential Deep Learning (EDL). Specifically, we propose EviSEC, an innovative and effective OOD detector via Evidential Spectrum-awarE Contrastive Learning. We design an evidential neural network to redefine the output as the posterior Dirichlet distribution, explaining the randomness of inputs through the uncertainty of distribution, which is overlooked by single-point estimation. Moreover, spectrum-aware augmentation module generates OOD approximations to identify patterns with high OOD scores, thereby widening the score gap between ID and OOD data and mitigating score homogenization. Extensive experiments on real-world datasets demonstrate that EviSAC effectively detects OOD samples in dynamic graphs. 

**Abstract (ZH)**: 最近，动态图中的离分布（OOD）检测引起了安全敏感领域的广泛关注，该检测旨在识别输入数据是否偏离训练集内分布（ID）的数据分布。当前的OOD检测主要集中在静态图上，并面临两个关键挑战：一是单点估计导致的高度偏差和高度方差，使得预测结果对数据中的随机性敏感；二是由于缺乏OOD训练数据导致的分数同质化问题，模型只能学习特定于ID的模式，从而导致整体OOD分数较低且ID与OOD数据之间的分数差距狭窄。为解决这些问题，我们首先通过证据深度学习（EDL）的视角研究动态图中的OOD检测。具体而言，我们提出了一种名为EviSEC的创新且有效的OOD检测器，通过证据光谱感知对比学习（Evidential Spectrum-aware Contrastive Learning）实现。设计了证据神经网络，将输出重新定义为后验狄利克雷分布，通过分布的不确定性解释输入的随机性，这是单点估计忽略的部分。此外，光谱感知增强模块生成OOD近似值以识别高OOD分数的模式，从而扩大ID与OOD数据之间的分数差距并缓解分数同质化问题。在真实世界数据集上的广泛实验表明，EviSEC能够有效检测动态图中的OOD样本。 

---
# Fractional-order Jacobian Matrix Differentiation and Its Application in Artificial Neural Networks 

**Title (ZH)**: 分数阶雅可比矩阵微分及其在人工神经网络中的应用 

**Authors**: Xiaojun zhou, Chunna Zhao, Yaqun Huang, Chengli Zhou, Junjie Ye, Kemeng Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07408)  

**Abstract**: Fractional-order differentiation has many characteristics different from integer-order differentiation. These characteristics can be applied to the optimization algorithms of artificial neural networks to obtain better results. However, due to insufficient theoretical research, at present, there is no fractional-order matrix differentiation method that is perfectly compatible with automatic differentiation (Autograd) technology. Therefore, we propose a fractional-order matrix differentiation calculation method. This method is introduced by the definition of the integer-order Jacobian matrix. We denote it as fractional-order Jacobian matrix differentiation (${\bf{J}^\alpha }$). Through ${\bf{J}^\alpha }$, we can carry out the matrix-based fractional-order chain rule. Based on the Linear module and the fractional-order differentiation, we design the fractional-order Autograd technology to enable the use of fractional-order differentiation in hidden layers, thereby enhancing the practicality of fractional-order differentiation in deep learning. In the experiment, according to the PyTorch framework, we design fractional-order Linear (FLinear) and replace this http URL in the multilayer perceptron with FLinear. Through the qualitative analysis of the training set and validation set $Loss$, the quantitative analysis of the test set indicators, and the analysis of time consumption and GPU memory usage during model training, we verify the superior performance of ${\bf{J}^\alpha }$ and prove that it is an excellent fractional-order gradient descent method in the field of deep learning. 

**Abstract (ZH)**: 分数阶微分具有与整数阶微分许多不同的特性。这些特性可以应用于人工神经网络的优化算法，以获得更好的结果。然而，由于理论研究不足，目前尚未有完全兼容自动微分（Autograd）技术的分数阶矩阵微分方法。因此，我们提出了一种分数阶矩阵微分计算方法。该方法基于整数阶雅可比矩阵的定义引入，称为分数阶雅可比矩阵微分（${\bf{J}^\alpha }$）。通过${\bf{J}^\alpha }$，我们可以进行基于矩阵的分数阶链式法则。基于线性模块和分数阶微分，我们设计了分数阶Autograd技术，以使分数阶微分能够在隐藏层中使用，从而增强分数阶微分在深度学习中的实用性。在实验中，我们根据PyTorch框架设计了分数阶线性（FLinear），并将该http链接中的多层感知机层替换为FLinear。通过训练集和验证集$Loss$的定性分析、测试集指标的定量分析以及模型训练过程中时间消耗和GPU内存使用情况的分析，我们验证了${\bf{J}^\alpha }$的优越性能，并证明其在深度学习领域是优秀的分数阶梯度下降方法。 

---
# Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation 

**Title (ZH)**: 适配器自然地作为跨域少样本语义分割的解藕器 

**Authors**: Jintao Tong, Ran Ma, Yixiong Zou, Guangyao Chen, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.07376)  

**Abstract**: Cross-domain few-shot segmentation (CD-FSS) is proposed to pre-train the model on a source-domain dataset with sufficient samples, and then transfer the model to target-domain datasets where only a few samples are available for efficient fine-tuning. There are majorly two challenges in this task: (1) the domain gap and (2) fine-tuning with scarce data. To solve these challenges, we revisit the adapter-based methods, and discover an intriguing insight not explored in previous works: the adapter not only helps the fine-tuning of downstream tasks but also naturally serves as a domain information decoupler. Then, we delve into this finding for an interpretation, and find the model's inherent structure could lead to a natural decoupling of domain information. Building upon this insight, we propose the Domain Feature Navigator (DFN), which is a structure-based decoupler instead of loss-based ones like current works, to capture domain-specific information, thereby directing the model's attention towards domain-agnostic knowledge. Moreover, to prevent the potential excessive overfitting of DFN during the source-domain training, we further design the SAM-SVN method to constrain DFN from learning sample-specific knowledge. On target domains, we freeze the model and fine-tune the DFN to learn target-specific knowledge specific. Extensive experiments demonstrate that our method surpasses the state-of-the-art method in CD-FSS significantly by 2.69% and 4.68% MIoU in 1-shot and 5-shot scenarios, respectively. 

**Abstract (ZH)**: 跨领域少样本分割（CD-FSS）预训练模型在源领域数据集上进行训练，然后将模型迁移到目标领域数据集，这些数据集只有少量样本可供高效微调。在这个任务中存在两大挑战：（1）领域差距和（2）基于稀缺数据的微调。为了解决这些挑战，我们重新审视了基于适配器的方法，并在先前工作中发现了有趣的新见解：适配器不仅有助于下游任务的微调，还自然地充当领域信息解耦器。基于这一发现，我们进一步探究其背后的原因，并发现模型固有的结构能够自然地解耦领域信息。利用这一新见解，我们提出了领域特征导航器（DFN），这是一种基于结构的解耦器，不同于当前基于损失的方法，能够捕获领域特定的信息，从而引导模型的关注点转向领域无关的知识。此外，为了防止在源领域训练过程中DFN可能过度拟合，我们进一步设计了SAM-SVN方法来限制DFN学习样本特定的知识。在目标领域中，我们冻结模型并微调DFN以学习特定于目标的知识。广泛实验表明，与当前最先进的方法相比，我们的方法在1-shot和5-shot场景中分别显著提高了2.69%和4.68%的MIoU。 

---
# HyColor: An Efficient Heuristic Algorithm for Graph Coloring 

**Title (ZH)**: HyColor: 一种高效的图着色启发式算法 

**Authors**: Enqiang Zhu, Yu Zhang, Haopeng Sun, Ziqi Wei, Witold Pedrycz, Chanjuan Liu, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07373)  

**Abstract**: The graph coloring problem (GCP) is a classic combinatorial optimization problem that aims to find the minimum number of colors assigned to vertices of a graph such that no two adjacent vertices receive the same color. GCP has been extensively studied by researchers from various fields, including mathematics, computer science, and biological science. Due to the NP-hard nature, many heuristic algorithms have been proposed to solve GCP. However, existing GCP algorithms focus on either small hard graphs or large-scale sparse graphs (with up to 10^7 vertices). This paper presents an efficient hybrid heuristic algorithm for GCP, named HyColor, which excels in handling large-scale sparse graphs while achieving impressive results on small dense graphs. The efficiency of HyColor comes from the following three aspects: a local decision strategy to improve the lower bound on the chromatic number; a graph-reduction strategy to reduce the working graph; and a k-core and mixed degree-based greedy heuristic for efficiently coloring graphs. HyColor is evaluated against three state-of-the-art GCP algorithms across four benchmarks, comprising three large-scale sparse graph benchmarks and one small dense graph benchmark, totaling 209 instances. The results demonstrate that HyColor consistently outperforms existing heuristic algorithms in both solution accuracy and computational efficiency for the majority of instances. Notably, HyColor achieved the best solutions in 194 instances (over 93%), with 34 of these solutions significantly surpassing those of other algorithms. Furthermore, HyColor successfully determined the chromatic number and achieved optimal coloring in 128 instances. 

**Abstract (ZH)**: 图着色问题（GCP）是经典组合优化问题，旨在找到一种方法，将图中的顶点着色，使得每种颜色只出现在不相邻的顶点上，并且使用的颜色数量最少。GCP 已经得到来自数学、计算机科学和生物科学等多个领域的研究人员的广泛研究。由于其 NP 难性，已经提出了许多启发式算法来解决 GCP。然而，现有的 GCP 算法主要集中在小硬图或大规模稀疏图（最多包含 10^7 个顶点）。本文提出了一种高效的混合启发式算法 HyColor，该算法在处理大规模稀疏图的同时，也能在小密集图上取得令人印象深刻的结果。HyColor 的效率来自于以下三个方面：局部决策策略以提高色数的下界；图约简策略以减少工作图；以及基于 k-核心和混合度量的贪婪启发式方法以高效地对图着色。HyColor 在四种基准测试中的三种大规模稀疏图基准测试和一种小密集图基准测试共计 209 个实例与三种最先进的 GCP 算法进行了评估。结果表明，HyColor 在多数实例中在解的准确性和计算效率方面都优于现有启发式算法。值得注意的是，HyColor 在 194 个实例（超过 93%）中达到了最佳解，其中 34 个实例显著优于其他算法。此外，HyColor 成功确定了色数并在 128 个实例中实现了最优着色。 

---
# Multiple Object Stitching for Unsupervised Representation Learning 

**Title (ZH)**: 无监督表示学习中的多对象拼接 

**Authors**: Chengchao Shen, Dawei Liu, Jianxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07364)  

**Abstract**: Contrastive learning for single object centric images has achieved remarkable progress on unsupervised representation, but suffering inferior performance on the widespread images with multiple objects. In this paper, we propose a simple but effective method, Multiple Object Stitching (MOS), to refine the unsupervised representation for multi-object images. Specifically, we construct the multi-object images by stitching the single object centric ones, where the objects in the synthesized multi-object images are predetermined. Hence, compared to the existing contrastive methods, our method provides additional object correspondences between multi-object images without human annotations. In this manner, our method pays more attention to the representations of each object in multi-object image, thus providing more detailed representations for complicated downstream tasks, such as object detection and semantic segmentation. Experimental results on ImageNet, CIFAR and COCO datasets demonstrate that our proposed method achieves the leading unsupervised representation performance on both single object centric images and multi-object ones. The source code is available at this https URL. 

**Abstract (ZH)**: 单物体中心图的对比学习已在无监督表示上取得了显著进展，但在广泛存在的多物体图像上表现较差。本文提出一种简单有效的多物体拼接（MOS）方法，以 refin无监督表示方法在多物体图像上的性能。具体地，通过将单物体中心图拼接生成多物体图像，其中合成的多物体图像中的物体预先确定。因此，与现有的对比学习方法相比，我们的方法为多物体图像之间提供了额外的对象对应关系，无需人工注释。通过这种方式，我们的方法更关注多物体图像中每个物体的表示，从而为复杂下游任务（如物体检测和语义分割）提供更详细的表示。实验结果表明，我们的方法在ImageNet、CIFAR和COCO数据集上的无监督表示性能均居首位。源代码可在以下链接获得。 

---
# Deepfake Technology Unveiled: The Commoditization of AI and Its Impact on Digital Trust 

**Title (ZH)**: Deepfake 技术揭秘：AI 的商品化及其对数字信任的影响 

**Authors**: Claudiu Popa, Rex Pallath, Liam Cunningham, Hewad Tahiri, Abiram Kesavarajah, Tao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07363)  

**Abstract**: Deepfake Technology Unveiled: The Commoditization of AI and Its Impact on Digital Trust. With the increasing accessibility of generative AI, tools for voice cloning, face-swapping, and synthetic media creation have advanced significantly, lowering both financial and technical barriers for their use. While these technologies present innovative opportunities, their rapid growth raises concerns about trust, privacy, and security. This white paper explores the implications of deepfake technology, analyzing its role in enabling fraud, misinformation, and the erosion of authenticity in multimedia. Using cost-effective, easy to use tools such as Runway, Rope, and ElevenLabs, we explore how realistic deepfakes can be created with limited resources, demonstrating the risks posed to individuals and organizations alike. By analyzing the technical and ethical challenges of deepfake mitigation and detection, we emphasize the urgent need for regulatory frameworks, public awareness, and collaborative efforts to maintain trust in digital media. 

**Abstract (ZH)**: Deepfake技术揭示：AI的商品化及其对数字信任的影响。随着生成式人工智能的日益普及，语音克隆、人脸识别替换和合成媒体创作工具显著进步，降低了这两种障碍。虽然这些技术展示了创新的机会，但它们的快速成长引发了对信任、隐私和安全的担忧。本白皮书探讨了Deepfake技术的影响，分析了其在欺诈、误导信息以及多媒体中真实性侵蚀方面的角色。通过使用Runway、Rope和ElevenLabs等经济实惠且易于使用的工具，我们探讨了如何使用有限资源创建逼真的Deepfake，展示了其对个人和组织带来的风险。通过分析Deepfake缓解和检测的技术和伦理挑战，我们强调 Urgent 需要制定监管框架、提高公众意识以及协作努力，以维护数字媒体的信任。 

---
# SALT: A Lightweight Model Adaptation Method for Closed Split Computing Environments 

**Title (ZH)**: SALT：一种轻量级模型适配方法，用于封闭拆分计算环境 

**Authors**: Yuya Okada, Takayuki Nishio  

**Link**: [PDF](https://arxiv.org/pdf/2506.07355)  

**Abstract**: We propose SALT (Split-Adaptive Lightweight Tuning), a lightweight model adaptation framework for Split Computing under closed constraints, where the head and tail networks are proprietary and inaccessible to users. In such closed environments, conventional adaptation methods are infeasible since they require access to model parameters or architectures. SALT addresses this challenge by introducing a compact, trainable adapter on the client side to refine latent features from the head network, enabling user-specific adaptation without modifying the original models or increasing communication overhead. We evaluate SALT on user-specific classification tasks with CIFAR-10 and CIFAR-100, demonstrating improved accuracy with lower training latency compared to fine-tuning methods. Furthermore, SALT facilitates model adaptation for robust inference over lossy networks, a common challenge in edge-cloud environments. With minimal deployment overhead, SALT offers a practical solution for personalized inference in edge AI systems under strict system constraints. 

**Abstract (ZH)**: SALT: 分裂计算环境下具有产权限制的轻量级模型自适应框架 

---
# Distributed Risk-Sensitive Safety Filters for Uncertain Discrete-Time Systems 

**Title (ZH)**: 不确定离散时间系统中分布式的风险敏感安全性滤波器 

**Authors**: Armin Lederer, Erfaun Noorani, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2506.07347)  

**Abstract**: Ensuring safety in multi-agent systems is a significant challenge, particularly in settings where centralized coordination is impractical. In this work, we propose a novel risk-sensitive safety filter for discrete-time multi-agent systems with uncertain dynamics that leverages control barrier functions (CBFs) defined through value functions. Our approach relies on centralized risk-sensitive safety conditions based on exponential risk operators to ensure robustness against model uncertainties. We introduce a distributed formulation of the safety filter by deriving two alternative strategies: one based on worst-case anticipation and another on proximity to a known safe policy. By allowing agents to switch between strategies, feasibility can be ensured. Through detailed numerical evaluations, we demonstrate the efficacy of our approach in maintaining safety without being overly conservative. 

**Abstract (ZH)**: 确保多自主系统安全是一个重要挑战，尤其是在集中协调不切实际的情况下。本文提出了一种新的风险敏感型安全滤波器，适用于具有不确定性动力学的离散时间多自主系统，并利用通过值函数定义的控制障碍函数（CBFs）。我们的方法基于基于指数风险操作符的集中风险敏感型安全条件，以确保对模型不确定性具有鲁棒性。通过推导两种替代策略来获得分布式安全滤波器的分布形式：一种基于最坏情况预测，另一种基于已知安全策略的接近性。通过允许代理在策略之间切换来确保可行性。通过详细的数值评估，我们展示了该方法在保持安全的同时不过于保守的有效性。 

---
# Speech Recognition on TV Series with Video-guided Post-Correction 

**Title (ZH)**: 电视连续剧中的语音识别与视频引导后修正 

**Authors**: Haoyuan Yang, Yue Zhang, Liqiang Jing  

**Link**: [PDF](https://arxiv.org/pdf/2506.07323)  

**Abstract**: Automatic Speech Recognition (ASR) has achieved remarkable success with deep learning, driving advancements in conversational artificial intelligence, media transcription, and assistive technologies. However, ASR systems still struggle in complex environments such as TV series, where overlapping speech, domain-specific terminology, and long-range contextual dependencies pose significant challenges to transcription accuracy. Existing multimodal approaches fail to correct ASR outputs with the rich temporal and contextual information available in video. To address this limitation, we propose a novel multimodal post-correction framework that refines ASR transcriptions by leveraging contextual cues extracted from video. Our framework consists of two stages: ASR Generation and Video-based Post-Correction, where the first stage produces the initial transcript and the second stage corrects errors using Video-based Contextual Information Extraction and Context-aware ASR Correction. We employ the Video-Large Multimodal Model (VLMM) to extract key contextual information using tailored prompts, which is then integrated with a Large Language Model (LLM) to refine the ASR output. We evaluate our method on a multimodal benchmark for TV series ASR and demonstrate its effectiveness in improving ASR performance by leveraging video-based context to enhance transcription accuracy in complex multimedia environments. 

**Abstract (ZH)**: 自动语音识别（ASR）借助深度学习取得了显著成功，推动了对话式人工智能、媒体转录和辅助技术的发展。然而，ASR系统在电视剧等复杂环境中仍面临挑战，如重叠语音、领域特定术语以及长范围上下文依赖性对转录准确性构成了重大挑战。现有跨模态方法无法利用视频中丰富的时序和上下文信息来纠正ASR输出。为解决这一局限，我们提出了一种新的跨模态后处理框架，通过利用从视频中提取的上下文线索来精炼ASR转录。该框架包括两个阶段：ASR生成和视频驱动的后处理，第一阶段生成初始转录，第二阶段使用基于视频的上下文信息提取和上下文感知ASR纠正来修正错误。我们使用定制提示的视频大型跨模态模型（VLMM）提取关键上下文信息，并将其与大型语言模型集成，以精炼ASR输出。我们在电视剧ASR的跨模态基准上评估了我们的方法，并通过利用视频中的上下文增强复杂多媒体环境中的转录准确性，展示了其有效性。 

---
# Generative Modeling of Networked Time-Series via Transformer Architectures 

**Title (ZH)**: 基于变压器架构的网络时间序列生成建模 

**Authors**: Yusuf Elnady  

**Link**: [PDF](https://arxiv.org/pdf/2506.07312)  

**Abstract**: Many security and network applications require having large datasets to train the machine learning models. Limited data access is a well-known problem in the security domain. Recent studies have shown the potential of Transformer models to enlarge the size of data by synthesizing new samples, but the synthesized samples don't improve the models over the real data. To address this issue, we design an efficient transformer-based model as a generative framework to generate time-series data, that can be used to boost the performance of existing and new ML workflows. Our new transformer model achieves the SOTA results. We style our model to be generalizable and work across different datasets, and produce high-quality samples. 

**Abstract (ZH)**: 许多安全和网络应用需要大量数据集来训练机器学习模型。在安全领域，有限的数据访问是一个已知问题。近期研究表明，变换器模型有扩大数据规模的潜力，可以通过合成新样本实现，但合成样本并未提高模型性能。为解决这一问题，我们设计了一个高效的基于变换器的生成模型框架，用于生成时间序列数据，以增强现有和新机器学习流程的性能。我们的新型变换器模型达到了当前最佳结果。我们设计该模型具有通用性，可在不同数据集上工作，并生成高质量的样本。 

---
# Secondary Stakeholders in AI: Fighting for, Brokering, and Navigating Agency 

**Title (ZH)**: 人工智能领域中的次级利益相关者：争取、斡旋和导航代理权 

**Authors**: Leah Hope Ajmani, Nuredin Ali Abdelkadir, Stevie Chancellor  

**Link**: [PDF](https://arxiv.org/pdf/2506.07281)  

**Abstract**: As AI technologies become more human-facing, there have been numerous calls to adapt participatory approaches to AI development -- spurring the idea of participatory AI. However, these calls often focus only on primary stakeholders, such as end-users, and not secondary stakeholders. This paper seeks to translate the ideals of participatory AI to a broader population of secondary AI stakeholders through semi-structured interviews. We theorize that meaningful participation involves three participatory ideals: (1) informedness, (2) consent, and (3) agency. We also explore how secondary stakeholders realize these ideals by traversing a complicated problem space. Like walking up the rungs of a ladder, these ideals build on one another. We introduce three stakeholder archetypes: the reluctant data contributor, the unsupported activist, and the well-intentioned practitioner, who must navigate systemic barriers to achieving agentic AI relationships. We envision an AI future where secondary stakeholders are able to meaningfully participate with the AI systems they influence and are influenced by. 

**Abstract (ZH)**: 随着AI技术更加面向人类，人们呼吁适应参与式AI开发的方法——催生了参与式AI的理念。然而，这些呼吁往往仅关注主要利益相关者，如最终用户，而忽视了次要利益相关者。本文旨在通过半结构化访谈，将参与式AI的理念扩展到更广泛范围的次要AI利益相关者群体。我们认为有意义的参与涉及三个参与式理念：（1）知情权，（2）同意权，（3）自主权。我们还探讨了次要利益相关者如何通过穿越复杂的问题空间来实现这些理念。这些理念层层递进，犹如攀登梯子的每一级。我们介绍了三种利益相关者原型：不愿意的数据贡献者、得不到支持的活动家以及致力于实现自主型人机关系的有良好意愿的从业者。我们构想一个未来的AI世界，在这个世界中，次要利益相关者能够与其影响和被影响的AI系统进行有意义的互动。 

---
# Regularized Adaptive Graph Learning for Large-Scale Traffic Forecasting 

**Title (ZH)**: 正则化自适应图学习在大规模交通预测中的应用 

**Authors**: Kaiqi Wu, Weiyang Kong, Sen Zhang, Yubao Liu, Zitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07179)  

**Abstract**: Traffic prediction is a critical task in spatial-temporal forecasting with broad applications in travel planning and urban management. Adaptive graph convolution networks have emerged as mainstream solutions due to their ability to learn node embeddings in a data-driven manner and capture complex latent dependencies. However, existing adaptive graph learning methods for traffic forecasting often either ignore the regularization of node embeddings, which account for a significant proportion of model parameters, or face scalability issues from expensive graph convolution operations. To address these challenges, we propose a Regularized Adaptive Graph Learning (RAGL) model. First, we introduce a regularized adaptive graph learning framework that synergizes Stochastic Shared Embedding (SSE) and adaptive graph convolution via a residual difference mechanism, achieving both embedding regularization and noise suppression. Second, to ensure scalability on large road networks, we develop the Efficient Cosine Operator (ECO), which performs graph convolution based on the cosine similarity of regularized embeddings with linear time complexity. Extensive experiments on four large-scale real-world traffic datasets show that RAGL consistently outperforms state-of-the-art methods in terms of prediction accuracy and exhibits competitive computational efficiency. 

**Abstract (ZH)**: 交通预测是时空预测中的关键任务，广泛应用于出行规划和城市管理。基于数据驱动的方法学习节点嵌入并捕获复杂潜在依赖性的自适应图卷积网络已成为主流解决方案。然而，现有的交通预测自适应图学习方法往往要么忽视了节点嵌入的正则化，这些嵌入占了模型参数的重要比例，要么由于昂贵的图卷积操作面临可扩展性问题。为解决这些挑战，我们提出了一种正则化自适应图学习（RAGL）模型。首先，我们引入了一种结合随机共享嵌入（SSE）和自适应图卷积的正则化自适应图学习框架，通过残差差分机制实现嵌入正则化和噪声抑制。其次，为了确保在大规模道路网络上的可扩展性，我们开发了高效余弦操作器（ECO），它基于正则化嵌入的余弦相似性进行图卷积，并具有线性时间复杂度。在四个大规模现实世界交通数据集上的 extensive 实验表明，RAGL 在预测准确性上始终优于现有最先进的方法，并且具有竞争力的计算效率。 

---
# CTDGSI: A comprehensive exploitation of instance selection methods for automatic text classification. VII Concurso de Teses, Dissertações e Trabalhos de Graduação em SI -- XXI Simpósio Brasileiro de Sistemas de Informação 

**Title (ZH)**: CTDGSI：自动文本分类中实例选择方法的综合探讨——第七届全国信息系统硕士、博士论文及本科毕业设计竞赛暨第二十一次巴西信息系统研讨会。 

**Authors**: Washington Cunha, Leonardo Rocha, Marcos André Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2506.07169)  

**Abstract**: Progress in Natural Language Processing (NLP) has been dictated by the rule of more: more data, more computing power and more complexity, best exemplified by the Large Language Models. However, training (or fine-tuning) large dense models for specific applications usually requires significant amounts of computing resources. This \textbf{Ph.D. dissertation} focuses on an under-investi\-gated NLP data engineering technique, whose potential is enormous in the current scenario known as Instance Selection (IS). The IS goal is to reduce the training set size by removing noisy or redundant instances while maintaining the effectiveness of the trained models and reducing the training process cost. We provide a comprehensive and scientifically sound comparison of IS methods applied to an essential NLP task -- Automatic Text Classification (ATC), considering several classification solutions and many datasets. Our findings reveal a significant untapped potential for IS solutions. We also propose two novel IS solutions that are noise-oriented and redundancy-aware, specifically designed for large datasets and transformer architectures. Our final solution achieved an average reduction of 41\% in training sets, while maintaining the same levels of effectiveness in all datasets. Importantly, our solutions demonstrated speedup improvements of 1.67x (up to 2.46x), making them scalable for datasets with hundreds of thousands of documents. 

**Abstract (ZH)**: 自然语言处理（NLP）的进步主要受到更多数据、更多计算资源和更复杂模型的驱动，大型语言模型是最典型的例子。然而，为了特定应用训练（或微调）大型密集模型通常需要大量的计算资源。本博士论文专注于一个尚未充分研究的NLP数据工程技术，在当前被称为实例选择（IS）的场景中其潜力巨大。IS的目标是通过去除噪声或冗余实例来减少训练集规模，同时保持训练模型的效果并降低训练过程成本。我们对IS方法在一项重要NLP任务——自动文本分类（ATC）中的应用进行了全面且科学的比较，考虑了多种分类解决方案和多个数据集。我们的研究发现IS解决方案具有巨大的未开发潜力。我们还提出两种针对大型数据集和变换器架构的新颖IS解决方案， noise导向和冗余 aware。最终解决方案在所有数据集上实现了训练集规模平均41%的减少，同时保持相同的效果水平。重要的是，我们的解决方案在某些情况下显示出1.67倍（最高2.46倍）的加速改进，使其适用于包含数十万文档的数据集，具有可扩展性。 

---
# MAGNet: A Multi-Scale Attention-Guided Graph Fusion Network for DRC Violation Detection 

**Title (ZH)**: MAGNet：一种多尺度注意力引导的图融合网络用于DRC违规检测 

**Authors**: Weihan Lu, Hong Cai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07126)  

**Abstract**: Design rule checking (DRC) is of great significance for cost reduction and design efficiency improvement in integrated circuit (IC) designs. Machine-learning-based DRC has become an important approach in computer-aided design (CAD). In this paper, we propose MAGNet, a hybrid deep learning model that integrates an improved U-Net with a graph neural network for DRC violation prediction. The U-Net backbone is enhanced with a Dynamic Attention Module (DAM) and a Multi-Scale Convolution Module (MSCM) to strengthen its capability in extracting fine-grained and multi-scale spatial features. In parallel, we construct a pixel-aligned graph structure based on chip layout tiles, and apply a specialized GNN to model the topological relationships among pins. During graph construction, a graph-to-grid mapping is generated to align GNN features with the layout image. In addition, a label amplification strategy is adopted during training to enhance the model's sensitivity to sparse violation patterns. Overall, MAGNet effectively combines spatial, semantic, and structural information, achieving improved prediction accuracy and reduced false positive rates in DRC hotspot detection. Subsequently, through incremental training, we achieve a more sensitive discrimination ability for hotspots. The results demonstrate that, in comparison with ibUnet, RouteNet, and J-Net, MAGnet significantly outperforms these models, achieving substantial improvements in overall performance. 

**Abstract (ZH)**: 基于机器学习的布局布线验证（DRC）对于集成电路（IC）设计中的成本降低和设计效率提升具有重要意义。本文提出了一种结合改进的U-Net和图神经网络的混合深度学习模型MAGNet，用于DRC违规预测。U-Net主干通过动态注意力模块（DAM）和多尺度卷积模块（MSCM）得到增强，以提高其从细粒度和多尺度空间特征中提取信息的能力。同时，基于芯片布局拼块构建像素对齐的图结构，并使用专门的图神经网络来建模引脚之间的拓扑关系。在图构建过程中，生成图到网格的映射，以使GNN特征与布局图像对齐。此外，在训练过程中采用标签放大策略，以提高模型对稀疏违规模式的敏感性。总体而言，MAGNet有效地结合了空间、语义和结构信息，在DRC热点检测中实现了更高的预测准确性和更低的假阳性率。通过增量训练，实现对热点更敏感的区分能力。结果表明，与ibUnet、RouteNet和J-Net相比，MAGNet在整体性能上显著优于这些模型。 

---
# RBA-FE: A Robust Brain-Inspired Audio Feature Extractor for Depression Diagnosis 

**Title (ZH)**: RBA-FE: 一种稳健的仿脑音频特征提取器用于抑郁诊断 

**Authors**: Yu-Xuan Wu, Ziyan Huang, Bin Hu, Zhi-Hong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07118)  

**Abstract**: This article proposes a robust brain-inspired audio feature extractor (RBA-FE) model for depression diagnosis, using an improved hierarchical network architecture. Most deep learning models achieve state-of-the-art performance for image-based diagnostic tasks, ignoring the counterpart audio features. In order to tailor the noise challenge, RBA-FE leverages six acoustic features extracted from the raw audio, capturing both spatial characteristics and temporal dependencies. This hybrid attribute helps alleviate the precision limitation in audio feature extraction within other learning models like deep residual shrinkage networks. To deal with the noise issues, our model incorporates an improved spiking neuron model, called adaptive rate smooth leaky integrate-and-fire (ARSLIF). The ARSLIF model emulates the mechanism of ``retuning of cellular signal selectivity" in the brain attention systems, which enhances the model robustness against environmental noises in audio data. Experimental results demonstrate that RBA-FE achieves state-of-the-art accuracy on the MODMA dataset, respectively with 0.8750, 0.8974, 0.8750 and 0.8750 in precision, accuracy, recall and F1 score. Extensive experiments on the AVEC2014 and DAIC-WOZ datasets both show enhancements in noise robustness. It is further indicated by comparison that the ARSLIF neuron model suggest the abnormal firing pattern within the feature extraction on depressive audio data, offering brain-inspired interpretability. 

**Abstract (ZH)**: 基于改进层次网络架构的稳健脑启发音频特征提取模型及其在抑郁诊断中的应用 

---
# Filling the Missings: Spatiotemporal Data Imputation by Conditional Diffusion 

**Title (ZH)**: 填补缺失：基于条件扩散的时空数据插补 

**Authors**: Wenying He, Jieling Huang, Junhua Gu, Ji Zhang, Yude Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.07099)  

**Abstract**: Missing data in spatiotemporal systems presents a significant challenge for modern applications, ranging from environmental monitoring to urban traffic management. The integrity of spatiotemporal data often deteriorates due to hardware malfunctions and software failures in real-world deployments. Current approaches based on machine learning and deep learning struggle to model the intricate interdependencies between spatial and temporal dimensions effectively and, more importantly, suffer from cumulative errors during the data imputation process, which propagate and amplify through iterations. To address these limitations, we propose CoFILL, a novel Conditional Diffusion Model for spatiotemporal data imputation. CoFILL builds on the inherent advantages of diffusion models to generate high-quality imputations without relying on potentially error-prone prior estimates. It incorporates an innovative dual-stream architecture that processes temporal and frequency domain features in parallel. By fusing these complementary features, CoFILL captures both rapid fluctuations and underlying patterns in the data, which enables more robust imputation. The extensive experiments reveal that CoFILL's noise prediction network successfully transforms random noise into meaningful values that align with the true data distribution. The results also show that CoFILL outperforms state-of-the-art methods in imputation accuracy. The source code is publicly available at this https URL. 

**Abstract (ZH)**: 空间时序系统中的缺失数据为现代应用带来了重大挑战，范围从环境监测到城市交通管理。实际部署中，空间时序数据的完整性常因硬件故障和软件故障而恶化。基于机器学习和深度学习的现有方法难以有效地建模空间和时间维度之间的复杂相互依赖，并且更重要的是，在数据填充过程中会产生累积误差，这些误差会在迭代中传播和放大。为了解决这些局限性，我们提出了一种新颖的空间时序数据填充模型CoFILL，该模型是一种条件扩散模型。CoFILL 利用扩散模型的固有优势，生成高质量的填充结果，而不依赖于可能带有错误的先验估计。它采用了一种创新的双流架构，同时处理时间域和频域特征。通过融合这些互补特征，CoFILL 能捕获数据中的快速波动和潜在模式，从而实现更 robust 的填充。广泛的数据实验表明，CoFILL 的噪声预测网络成功地将随机噪声转化为与真实数据分布相一致的有意义的值。实验结果还显示，CoFILL 在填充准确性上优于现有最先进的方法。源代码已在该网址公开。 

---
# Patient Similarity Computation for Clinical Decision Support: An Efficient Use of Data Transformation, Combining Static and Time Series Data 

**Title (ZH)**: 临床决策支持中的患者相似性计算：一种高效的数据转换应用，结合静态和时间序列数据 

**Authors**: Joydeb Kumar Sana, Mohammad M. Masud, M Sohel Rahman, M Saifur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2506.07092)  

**Abstract**: Patient similarity computation (PSC) is a fundamental problem in healthcare informatics. The aim of the patient similarity computation is to measure the similarity among patients according to their historical clinical records, which helps to improve clinical decision support. This paper presents a novel distributed patient similarity computation (DPSC) technique based on data transformation (DT) methods, utilizing an effective combination of time series and static data. Time series data are sensor-collected patients' information, including metrics like heart rate, blood pressure, Oxygen saturation, respiration, etc. The static data are mainly patient background and demographic data, including age, weight, height, gender, etc. Static data has been used for clustering the patients. Before feeding the static data to the machine learning model adaptive Weight-of-Evidence (aWOE) and Z-score data transformation (DT) methods have been performed, which improve the prediction performances. In aWOE-based patient similarity models, sensitive patient information has been processed using aWOE which preserves the data privacy of the trained models. We used the Dynamic Time Warping (DTW) approach, which is robust and very popular, for time series similarity. However, DTW is not suitable for big data due to the significant computational run-time. To overcome this problem, distributed DTW computation is used in this study. For Coronary Artery Disease, our DT based approach boosts prediction performance by as much as 11.4%, 10.20%, and 12.6% in terms of AUC, accuracy, and F-measure, respectively. In the case of Congestive Heart Failure (CHF), our proposed method achieves performance enhancement up to 15.9%, 10.5%, and 21.9% for the same measures, respectively. The proposed method reduces the computation time by as high as 40%. 

**Abstract (ZH)**: 基于数据变换的分布式患者相似性计算方法 

---
# On the Generalization of Data-Assisted Control in port-Hamiltonian Systems (DAC-pH) 

**Title (ZH)**: 基于数据辅助控制的端口哈密尔顿系统泛化研究（DAC-pH） 

**Authors**: Mostafa Eslami, Maryam Babazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.07079)  

**Abstract**: This paper introduces a hypothetical hybrid control framework for port-Hamiltonian (p$\mathcal{H}$) systems, employing a dynamic decomposition based on Data-Assisted Control (DAC). The system's evolution is split into two parts with fixed topology: Right-Hand Side (RHS)- an intrinsic Hamiltonian flow handling worst-case parametric uncertainties, and Left-Hand Side (LHS)- a dissipative/input flow addressing both structural and parametric uncertainties. A virtual port variable $\Pi$ serves as the interface between these two components. A nonlinear controller manages the intrinsic Hamiltonian flow, determining a desired port control value $\Pi_c$. Concurrently, Reinforcement Learning (RL) is applied to the dissipative/input flow to learn an agent for providing optimal policy in mapping $\Pi_c$ to the actual system input. This hybrid approach effectively manages RHS uncertainties while preserving the system's inherent structure. Key advantages include adjustable performance via LHS controller parameters, enhanced AI explainability and interpretability through the port variable $\Pi$, the ability to guarantee safety and state attainability with hard/soft constraints, reduced complexity in learning hypothesis classes compared to end-to-end solutions, and improved state/parameter estimation using LHS prior knowledge and system Hamiltonian to address partial observability. The paper details the p$\mathcal{H}$ formulation, derives the decomposition, and presents the modular controller architecture. Beyond design, crucial aspects of stability and robustness analysis and synthesis are investigated, paving the way for deeper theoretical investigations. An application example, a pendulum with nonlinear dynamics, is simulated to demonstrate the approach's empirical and phenomenological benefits for future research. 

**Abstract (ZH)**: 一种基于数据辅助控制的港哈密尔顿系统混合控制框架 

---
# From Axioms to Algorithms: Mechanized Proofs of the vNM Utility Theorem 

**Title (ZH)**: 从公理到算法：机器化证明的冯·诺伊曼-摩根索效用定理 

**Authors**: Li Jingyuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07066)  

**Abstract**: This paper presents a comprehensive formalization of the von Neumann-Morgenstern (vNM) expected utility theorem using the Lean 4 interactive theorem prover. We implement the classical axioms of preference-completeness, transitivity, continuity, and independence-enabling machine-verified proofs of both the existence and uniqueness of utility representations. Our formalization captures the mathematical structure of preference relations over lotteries, verifying that preferences satisfying the vNM axioms can be represented by expected utility maximization.
Our contributions include a granular implementation of the independence axiom, formally verified proofs of fundamental claims about mixture lotteries, constructive demonstrations of utility existence, and computational experiments validating the results. We prove equivalence to classical presentations while offering greater precision at decision boundaries.
This formalization provides a rigorous foundation for applications in economic modeling, AI alignment, and management decision systems, bridging the gap between theoretical decision theory and computational implementation. 

**Abstract (ZH)**: 本文使用Lean 4交互式定理证明器对von Neumann-Morgenstern (vNM)期望效用定理进行了全面的形式化。我们实现了偏好完备性、传递性、连续性和独立性等经典偏好公理，并通过机器验证证明了这些公理的存在性和唯一性效用表示。我们的形式化捕捉了彩票上偏好关系的数学结构，验证了满足vNM公理的偏好可以由期望效用最大化来表示。

本文的贡献包括对独立性公理的细致实现、关于混合彩票的基本断言的形式验证证明、效用存在的建设性演示以及验证结果的计算实验。我们在决策边界上提供了更高精度的同时证明了与经典表述等价性。 

---
# Less is More: some Computational Principles based on Parcimony, and Limitations of Natural Intelligence 

**Title (ZH)**: 少就是多：基于简约的一些计算原理及自然智能的限制 

**Authors**: Laura Cohen, Xavier Hinaut, Lilyana Petrova, Alexandre Pitti, Syd Reynal, Ichiro Tsuda  

**Link**: [PDF](https://arxiv.org/pdf/2506.07060)  

**Abstract**: Natural intelligence (NI) consistently achieves more with less. Infants learn language, develop abstract concepts, and acquire sensorimotor skills from sparse data, all within tight neural and energy limits. In contrast, today's AI relies on virtually unlimited computational power, energy, and data to reach high performance. This paper argues that constraints in NI are paradoxically catalysts for efficiency, adaptability, and creativity. We first show how limited neural bandwidth promotes concise codes that still capture complex patterns. Spiking neurons, hierarchical structures, and symbolic-like representations emerge naturally from bandwidth constraints, enabling robust generalization. Next, we discuss chaotic itinerancy, illustrating how the brain transits among transient attractors to flexibly retrieve memories and manage uncertainty. We then highlight reservoir computing, where random projections facilitate rapid generalization from small datasets. Drawing on developmental perspectives, we emphasize how intrinsic motivation, along with responsive social environments, drives infant language learning and discovery of meaning. Such active, embodied processes are largely absent in current AI. Finally, we suggest that adopting 'less is more' principles -- energy constraints, parsimonious architectures, and real-world interaction -- can foster the emergence of more efficient, interpretable, and biologically grounded artificial systems. 

**Abstract (ZH)**: 自然智能（NI）始终以更少的资源实现更多。婴儿从少量数据中学习语言、发展抽象概念并获得运动感觉技能，这一切都在严格的神经和能量限制内完成。相比之下，当今的AI依靠几乎无限的计算能力、能源和数据来达到高性能。本文认为，NI的限制反而促进了效率、适应性和创造力。我们首先展示有限的神经带宽如何促成简洁的编码，但仍能捕捉到复杂模式。脉冲神经元、分层结构和类符号表示自然地从带宽限制中涌现，使系统具备稳健的泛化能力。接着，我们讨论混沌游动，说明大脑如何在瞬态吸引子之间过渡，以灵活地检索记忆和管理不确定性。然后，我们强调内在动机以及响应性的社会环境在婴儿语言学习和意义发现中的作用。这些主动的、具身的过程在当前的AI中基本上是不存在的。最后，我们建议采用“少即是多”的原则——能量约束、简约的架构和现实世界交互，以促进更高效、可解释且基于生物学的人工系统的生成。 

---
# Policy Gradient with Tree Search: Avoiding Local Optimas through Lookahead 

**Title (ZH)**: 基于树搜索的策略梯度：通过前瞻避免局部最优 

**Authors**: Uri Koren, Navdeep Kumar, Uri Gadot, Giorgia Ramponi, Kfir Yehuda Levy, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2506.07054)  

**Abstract**: Classical policy gradient (PG) methods in reinforcement learning frequently converge to suboptimal local optima, a challenge exacerbated in large or complex environments. This work investigates Policy Gradient with Tree Search (PGTS), an approach that integrates an $m$-step lookahead mechanism to enhance policy optimization. We provide theoretical analysis demonstrating that increasing the tree search depth $m$-monotonically reduces the set of undesirable stationary points and, consequently, improves the worst-case performance of any resulting stationary policy. Critically, our analysis accommodates practical scenarios where policy updates are restricted to states visited by the current policy, rather than requiring updates across the entire state space. Empirical evaluations on diverse MDP structures, including Ladder, Tightrope, and Gridworld environments, illustrate PGTS's ability to exhibit "farsightedness," navigate challenging reward landscapes, escape local traps where standard PG fails, and achieve superior solutions. 

**Abstract (ZH)**: 经典策略梯度（PG）方法在强化学习中经常收敛到次优的局部最优解，这一问题在大型或复杂环境中尤为突出。本文研究了策略梯度与树搜索结合的方法（PGTS），该方法通过引入$m$步前瞻机制来增强策略优化。我们提供了理论分析，证明增加树搜索深度$m$可以单调减少不可取的稳态点集，从而提高任何由此产生的稳态策略的最坏情况性能。关键的是，我们的分析考虑了策略更新仅限于当前策略访问的状态，而不需要对整个状态空间进行更新。对于包括梯子环境、钢丝走绳环境和网格世界环境在内的多种MDP结构，实证评估展示了PGTS能够表现出“远见卓识”，导航复杂的奖励景观，避开标准PG方法失败的局部陷阱，并取得更优的解。 

---
# Efficient $Q$-Learning and Actor-Critic Methods for Robust Average Reward Reinforcement Learning 

**Title (ZH)**: 高效的 $Q$-学习与演员-评论家方法在鲁棒平均奖励强化学习中的应用 

**Authors**: Yang Xu, Swetha Ganesh, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07040)  

**Abstract**: We present the first $Q$-learning and actor-critic algorithms for robust average reward Markov Decision Processes (MDPs) with non-asymptotic convergence under contamination, TV distance and Wasserstein distance uncertainty sets. We show that the robust $Q$ Bellman operator is a strict contractive mapping with respect to a carefully constructed semi-norm with constant functions being quotiented out. This property supports a stochastic approximation update, that learns the optimal robust $Q$ function in $\tilde{\cO}(\epsilon^{-2})$ samples. We also show that the same idea can be used for robust $Q$ function estimation, which can be further used for critic estimation. Coupling it with theories in robust policy mirror descent update, we present a natural actor-critic algorithm that attains an $\epsilon$-optimal robust policy in $\tilde{\cO}(\epsilon^{-3})$ samples. These results advance the theory of distributionally robust reinforcement learning in the average reward setting. 

**Abstract (ZH)**: 我们提出了首个针对受到污染、基于TV距离和Wasserstein距离不确定性集的鲁棒平均奖励马尔可夫决策过程($Q$-学习和演员-评论家)算法，并证明了鲁棒$Q$贝尔曼算子在特定半范数下为严格收缩映射（恒函数除外），这支持了一种随机逼近更新方法，能够在$\tilde{\cO}(\epsilon^{-2})$样本中学习最优鲁棒$Q$函数。我们还展示了相同的思想可以用于鲁棒$Q$函数估计，进而可用于评论家估计。结合鲁棒政策镜像下降更新理论，我们提出了一种自然的演员-评论家算法，能够在$\tilde{\cO}(\epsilon^{-3})$样本中获得$\epsilon$-最优鲁棒策略。这些结果推进了平均奖励设置下分布鲁棒强化学习的理论。 

---
# AnnoDPO: Protein Functional Annotation Learning with Direct Preference Optimization 

**Title (ZH)**: AnnoDPO: 蛋白质功能注释学习与直接偏好优化 

**Authors**: Zixuan Jiang, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07035)  

**Abstract**: Deciphering protein function remains a fundamental challenge in protein representation learning. The task presents significant difficulties for protein language models (PLMs) due to the sheer volume of functional annotation categories and the highly imbalanced distribution of annotated instances across biological ontologies. Inspired by the remarkable success of reinforcement learning from human feedback (RLHF) in large language model (LLM) alignment, we propose AnnoDPO, a novel multi-modal framework for protein function prediction that leverages Direct Preference Optimization (DPO) to enhance annotation learning. Our methodology addresses the dual challenges of annotation scarcity and category imbalance through preference-aligned training objectives, establishing a new paradigm for biological knowledge integration in protein representation learning. 

**Abstract (ZH)**: 解析蛋白质功能仍然是蛋白质表示学习中的一个根本挑战。受大规模语言模型（LLM）对人类反馈强化学习（RLHF）卓越成功的启发，我们提出了一种名为AnnoDPO的新型多模式框架，利用直接偏好优化（DPO）增强标注学习。我们的方法通过偏好对齐的训练目标来应对标注稀缺性和类别不平衡的双重挑战，建立了蛋白质表示学习中生物知识集成的新范式。 

---
# Deep regularization networks for inverse problems with noisy operators 

**Title (ZH)**: 噪声算子下的深度正则化网络 

**Authors**: Fatemeh Pourahmadian, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07008)  

**Abstract**: A supervised learning approach is proposed for regularization of large inverse problems where the main operator is built from noisy data. This is germane to superresolution imaging via the sampling indicators of the inverse scattering theory. We aim to accelerate the spatiotemporal regularization process for this class of inverse problems to enable real-time imaging. In this approach, a neural operator maps each pattern on the right-hand side of the scattering equation to its affiliated regularization parameter. The network is trained in two steps which entails: (1) training on low-resolution regularization maps furnished by the Morozov discrepancy principle with nonoptimal thresholds, and (2) optimizing network predictions through minimization of the Tikhonov loss function regulated by the validation loss. Step 2 allows for tailoring of the approximate maps of Step 1 toward construction of higher quality images. This approach enables direct learning from test data and dispenses with the need for a-priori knowledge of the optimal regularization maps. The network, trained on low-resolution data, quickly generates dense regularization maps for high-resolution imaging. We highlight the importance of the training loss function on the network's generalizability. In particular, we demonstrate that networks informed by the logic of discrepancy principle lead to images of higher contrast. In this case, the training process involves many-objective optimization. We propose a new method to adaptively select the appropriate loss weights during training without requiring an additional optimization process. The proposed approach is synthetically examined for imaging damage evolution in an elastic plate. The results indicate that the discrepancy-informed regularization networks not only accelerate the imaging process, but also remarkably enhance the image quality in complex environments. 

**Abstract (ZH)**: 一种监督学习方法用于基于 noisy 数据构建的主要算子的大规模逆问题正则化，特别是在逆散射理论的采样指标下实现超分辨率成像。我们旨在通过该方法加速此类逆问题的空间-时间正则化过程，以实现实时成像。在此方法中，神经算子将散射方程右侧的每个模式映射为其相应的正则化参数。该网络分为两步进行训练：（1）使用 Morozov 矛盾原理提供的非最优阈值的低分辨率正则化地图进行训练；（2）通过最小化由验证损失调节的泰特kon夫损失函数优化网络预测。第二步使得第一步的近似地图能够更好地构建高质量图像。此方法能够直接从测试数据中学习，并不需要先验知道最优正则化地图。该网络在低分辨率数据上训练，能够迅速生成密集的正则化地图以实现高分辨率成像。我们强调训练损失函数对网络泛化能力的重要性。特别是，我们证明基于矛盾原理逻辑训练的网络能够产生更高对比度的图像。在这种情况下，训练过程涉及多目标优化。我们提出了一种新方法，在训练过程中自适应选择适当的损失权重，而不需额外的优化过程。所提出的方法已在弹性板中损伤演化成像的合成实验中进行了验证。结果表明，基于矛盾原理的正则化网络不仅能加速成像过程，还在复杂环境中显著提高图像质量。 

---
# End-to-End Probabilistic Framework for Learning with Hard Constraints 

**Title (ZH)**: 面向硬约束的端到端概率框架 

**Authors**: Utkarsh Utkarsh, Danielle C. Maddix, Ruijun Ma, Michael W. Mahoney, Yuyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07003)  

**Abstract**: We present a general purpose probabilistic forecasting framework, ProbHardE2E, to learn systems that can incorporate operational/physical constraints as hard requirements. ProbHardE2E enforces hard constraints by exploiting variance information in a novel way; and thus it is also capable of performing uncertainty quantification (UQ) on the model. Our methodology uses a novel differentiable probabilistic projection layer (DPPL) that can be combined with a wide range of neural network architectures. This DPPL allows the model to learn the system in an end-to-end manner, compared to other approaches where the constraints are satisfied either through a post-processing step or at inference. In addition, ProbHardE2E can optimize a strictly proper scoring rule, without making any distributional assumptions on the target, which enables it to obtain robust distributional estimates (in contrast to existing approaches that generally optimize likelihood-based objectives, which are heavily biased by their distributional assumptions and model choices); and it can incorporate a range of non-linear constraints (increasing the power of modeling and flexibility). We apply ProbHardE2E to problems in learning partial differential equations with uncertainty estimates and to probabilistic time-series forecasting, showcasing it as a broadly applicable general setup that connects these seemingly disparate domains. 

**Abstract (ZH)**: ProbHardE2E：一种整合操作/物理约束的通用概率预测框架 

---
# What makes Reasoning Models Different? Follow the Reasoning Leader for Efficient Decoding 

**Title (ZH)**: 什么是推理模型的不同之处？跟随推理领导者进行高效解码 

**Authors**: Ming Li, Zhengyuan Yang, Xiyao Wang, Dianqi Li, Kevin Lin, Tianyi Zhou, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06998)  

**Abstract**: Large reasoning models (LRMs) achieve strong reasoning performance by emitting long chains of thought. Yet, these verbose traces slow down inference and often drift into unnecessary detail, known as the overthinking phenomenon. To better understand LRMs' behavior, we systematically analyze the token-level misalignment between reasoning and non-reasoning models. While it is expected that their primary difference lies in the stylistic "thinking cues", LRMs uniquely exhibit two pivotal, previously under-explored phenomena: a Global Misalignment Rebound, where their divergence from non-reasoning models persists or even grows as response length increases, and more critically, a Local Misalignment Diminish, where the misalignment concentrates at the "thinking cues" each sentence starts with but rapidly declines in the remaining of the sentence. Motivated by the Local Misalignment Diminish, we propose FoReaL-Decoding, a collaborative fast-slow thinking decoding method for cost-quality trade-off. In FoReaL-Decoding, a Leading model leads the first few tokens for each sentence, and then a weaker draft model completes the following tokens to the end of each sentence. FoReaL-Decoding adopts a stochastic gate to smoothly interpolate between the small and the large model. On four popular math-reasoning benchmarks (AIME24, GPQA-Diamond, MATH500, AMC23), FoReaL-Decoding reduces theoretical FLOPs by 30 to 50% and trims CoT length by up to 40%, while preserving 86 to 100% of model performance. These results establish FoReaL-Decoding as a simple, plug-and-play route to controllable cost-quality trade-offs in reasoning-centric tasks. 

**Abstract (ZH)**: 大型推理模型 (LRMs) 通过生成长链条的思考过程实现强大的推理性能。然而，这些冗长的推理痕迹会减慢推理速度，并且经常陷入不必要的细节，这种现象被称为过度推理。为了更好地理解 LRMs 的行为，我们系统地分析了推理模型和非推理模型在 token 级别上的不对齐。虽然预期它们的主要差异在于风格化的“思考提示”，但 LRMs 唯一地表现出两个以往被忽视的关键现象：全局不对齐反弹，即它们与非推理模型的差异在响应长度增加时持续存在甚至增大；更重要的是局部不对齐消减，即不对齐集中于每个句子开头的“思考提示”，但在句子其余部分迅速下降。受局部不对齐消减的启发，我们提出了一种协作的快慢思考解码方法 FoReaL-Decoding，用于成本-质量权衡。在 FoReaL-Decoding 中，领先模型主导每个句子前几个 token 的生成，然后较弱的草稿模型完成其余 token 的生成。FoReaL-Decoding 采用随机门将小型模型和大型模型平滑地结合起来。在四个流行的数学推理基准测试（AIME24、GPQA-Diamond、MATH500、AMC23）上，FoReaL-Decoding 将理论 FLOPs 减少 30% 至 50%，将 CoT 长度缩短至多 40%，同时保留 86% 至 100% 的模型性能。这些结果确立了 FoReaL-Decoding 作为控制性成本-质量权衡任务的简单、即插即用的方法。 

---
# MoXGATE: Modality-aware cross-attention for multi-omic gastrointestinal cancer sub-type classification 

**Title (ZH)**: MoXGATE: 融合模态aware跨注意力机制的多组学胃肠癌亚型分类 

**Authors**: Sajib Acharjee Dip, Uddip Acharjee Shuvo, Dipanwita Mallick, Abrar Rahman Abir, Liqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06980)  

**Abstract**: Cancer subtype classification is crucial for personalized treatment and prognostic assessment. However, effectively integrating multi-omic data remains challenging due to the heterogeneous nature of genomic, epigenomic, and transcriptomic features. In this work, we propose Modality-Aware Cross-Attention MoXGATE, a novel deep-learning framework that leverages cross-attention and learnable modality weights to enhance feature fusion across multiple omics sources. Our approach effectively captures inter-modality dependencies, ensuring robust and interpretable integration. Through experiments on Gastrointestinal Adenocarcinoma (GIAC) and Breast Cancer (BRCA) datasets from TCGA, we demonstrate that MoXGATE outperforms existing methods, achieving 95\% classification accuracy. Ablation studies validate the effectiveness of cross-attention over simple concatenation and highlight the importance of different omics modalities. Moreover, our model generalizes well to unseen cancer types e.g., breast cancer, underscoring its adaptability. Key contributions include (1) a cross-attention-based multi-omic integration framework, (2) modality-weighted fusion for enhanced interpretability, (3) application of focal loss to mitigate data imbalance, and (4) validation across multiple cancer subtypes. Our results indicate that MoXGATE is a promising approach for multi-omic cancer subtype classification, offering improved performance and biological generalizability. 

**Abstract (ZH)**: 癌症亚型分类对于个性化治疗和预后评估至关重要。然而，由于基因组、表观基因组和转录组特征的异质性，有效地整合多组学数据仍然具有挑战性。在本文中，我们提出了一种新的深度学习框架——模态感知跨注意力MoXGATE，该框架利用跨注意力和可学习的模态权重来增强多组学来源间的特征融合。我们的方法有效地捕捉了跨模态依赖性，确保了稳健的可解释性集成。通过TCGA来源的胃肠道腺癌（GIAC）和乳腺癌（BRCA）数据集的实验，我们展示了MoXGATE在分类准确率方面超过了现有方法，达到了95%的分类准确率。消融研究证实了跨注意力在简单连接之上更为有效，并突显了不同组学模态的重要性。此外，该模型在未见过的癌症类型中表现出良好的泛化能力，强调了其适应性。关键贡献包括（1）基于跨注意力的多组学整合框架，（2）模态加权融合以增强可解释性，（3）使用焦点损失来缓解数据不平衡，以及（4）在多种癌症亚型中的验证。结果显示，MoXGATE是一种有前景的多组学癌症亚型分类方法，提供了改进的性能和生物学普适性。 

---
# UdonCare: Hierarchy Pruning for Unseen Domain Discovery in Predictive Healthcare 

**Title (ZH)**: UdonCare：未知领域发现中的层次结构剪枝在预测型医疗保健中的应用 

**Authors**: Pengfei Hu, Xiaoxue Han, Fei Wang, Yue Ning  

**Link**: [PDF](https://arxiv.org/pdf/2506.06977)  

**Abstract**: Domain generalization has become a critical challenge in clinical prediction, where patient cohorts often exhibit shifting data distributions that degrade model performance. Typical domain generalization approaches struggle in real-world healthcare settings for two main reasons: (1) patient-specific domain labels are typically unavailable, making domain discovery especially difficult; (2) purely data-driven approaches overlook key clinical insights, leading to a gap in medical knowledge integration. To address these problems, we leverage hierarchical medical ontologies like the ICD-9-CM hierarchy to group diseases into higher-level categories and discover more flexible latent domains. In this paper, we introduce UdonCare, a hierarchy-guided framework that iteratively prunes fine-grained domains, encodes these refined domains, and applies a Siamese-type inference mechanism to separate domain-related signals from patient-level features. Experimental results on clinical datasets (MIMIC-III and MIMIC-IV) show that the proposed model achieves higher performance compared to other domain generalization baselines when substantial domain gaps presents, highlighting the untapped potential of medical knowledge for enhancing domain generalization in practical healthcare applications. 

**Abstract (ZH)**: 临床预测中的域泛化已成为一个关键挑战，其中患者群体常表现出数据分布的变化，从而降低模型性能。典型的域泛化方法在实际医疗保健环境中因两大原因难以应对：（1）患者特定的域标签通常不可用，使域发现尤为困难；（2）完全数据驱动的方法忽视了关键的临床洞察，导致医学知识整合的鸿沟。为解决这些问题，我们利用ICD-9-CM层次结构等医疗本体论将疾病分组到较高层次的类别中，以发现更灵活的潜在域。在本文中，我们提出UdonCare，这是一种基于层次结构的框架，该框架通过迭代修剪细粒度的域、对这些精炼的域进行编码，并应用双胞胎类型的推理机制来分离与域相关的信号与患者级特征。临床数据集（MIMIC-III和MIMIC-IV）上的实验结果表明，在存在明显域差距时，所提出模型的表现优于其他域泛化基准，突显了提高实际医疗保健应用中域泛化性能的医学知识的未开发潜力。 

---
# Is Your Training Pipeline Production-Ready? A Case Study in the Healthcare Domain 

**Title (ZH)**: 你的训练管道准备好进入生产环境了吗？以医疗健康领域为例 

**Authors**: Daniel Lawand, Lucas Quaresma, Roberto Bolgheroni, Alfredo Goldman, Renato Cordeiro Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2506.06946)  

**Abstract**: Deploying a Machine Learning (ML) training pipeline into production requires robust software engineering practices. This differs significantly from experimental workflows. This experience report investigates this challenge in SPIRA, a project whose goal is to create an ML-Enabled System (MLES) to pre-diagnose insufficiency respiratory via speech analysis. The first version of SPIRA's training pipeline lacked critical software quality attributes. This paper presents an overview of the MLES, then compares three versions of the architecture of the Continuous Training subsystem, which evolved from a Big Ball of Mud, to a Modular Monolith, towards Microservices. By adopting different design principles and patterns to enhance its maintainability, robustness, and extensibility. In this way, the paper seeks to offer insights for both ML Engineers tasked to productionize ML training pipelines and Data Scientists seeking to adopt MLOps practices. 

**Abstract (ZH)**: 将机器学习训练管道部署到生产环境需要 robust 软件工程实践。这与实验性工作流程大不相同。本文经验报告探讨了 SPIRA 项目中的这一挑战，SPIRA 的目标是创建一个基于机器学习的系统 (MLES)，以通过语音分析提前诊断呼吸不足。SPIRA 的第一版训练管道缺乏关键的软件质量属性。本文首先概述 MLES，然后比较了该持续训练子系统的三个架构版本，从一团混乱的代码进化到了模块化单体，最终成为微服务。通过采用不同的设计原则和模式来提高其可维护性、稳健性和可扩展性。本文旨在为负责将机器学习训练管道生产化的 ML 工程师以及寻求采用 MLOps 实践的数据科学家提供见解。 

---
# Rewriting the Budget: A General Framework for Black-Box Attacks Under Cost Asymmetry 

**Title (ZH)**: 重写预算：在成本不对称条件下的一种通用黑箱攻击框架 

**Authors**: Mahdi Salmani, Alireza Abdollahpoorrostam, Seyed-Mohsen Moosavi-Dezfooli  

**Link**: [PDF](https://arxiv.org/pdf/2506.06933)  

**Abstract**: Traditional decision-based black-box adversarial attacks on image classifiers aim to generate adversarial examples by slightly modifying input images while keeping the number of queries low, where each query involves sending an input to the model and observing its output. Most existing methods assume that all queries have equal cost. However, in practice, queries may incur asymmetric costs; for example, in content moderation systems, certain output classes may trigger additional review, enforcement, or penalties, making them more costly than others. While prior work has considered such asymmetric cost settings, effective algorithms for this scenario remain underdeveloped. In this paper, we propose a general framework for decision-based attacks under asymmetric query costs, which we refer to as asymmetric black-box attacks. We modify two core components of existing attacks: the search strategy and the gradient estimation process. Specifically, we propose Asymmetric Search (AS), a more conservative variant of binary search that reduces reliance on high-cost queries, and Asymmetric Gradient Estimation (AGREST), which shifts the sampling distribution to favor low-cost queries. We design efficient algorithms that minimize total attack cost by balancing different query types, in contrast to earlier methods such as stealthy attacks that focus only on limiting expensive (high-cost) queries. Our method can be integrated into a range of existing black-box attacks with minimal changes. We perform both theoretical analysis and empirical evaluation on standard image classification benchmarks. Across various cost regimes, our method consistently achieves lower total query cost and smaller perturbations than existing approaches, with improvements of up to 40% in some settings. 

**Abstract (ZH)**: 决策导向的异构查询成本黑色盒攻击框架 

---
# Graph-Based Physics-Guided Urban PM2.5 Air Quality Imputation with Constrained Monitoring Data 

**Title (ZH)**: 基于图的物理引导城市PM2.5空气质量插值与受限监测数据约束 

**Authors**: Shangjie Du, Hui Wei, Dong Yoon Lee, Zhizhang Hu, Shijia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06917)  

**Abstract**: This work introduces GraPhy, a graph-based, physics-guided learning framework for high-resolution and accurate air quality modeling in urban areas with limited monitoring data. Fine-grained air quality monitoring information is essential for reducing public exposure to pollutants. However, monitoring networks are often sparse in socioeconomically disadvantaged regions, limiting the accuracy and resolution of air quality modeling. To address this, we propose a physics-guided graph neural network architecture called GraPhy with layers and edge features designed specifically for low-resolution monitoring data. Experiments using data from California's socioeconomically disadvantaged San Joaquin Valley show that GraPhy achieves the overall best performance evaluated by mean squared error (MSE), mean absolute error (MAE), and R-square value (R2), improving the performance by 9%-56% compared to various baseline models. Moreover, GraPhy consistently outperforms baselines across different spatial heterogeneity levels, demonstrating the effectiveness of our model design. 

**Abstract (ZH)**: 基于图的物理引导学习框架：GraPhy在 socioeconomically 不利地区城市区域细粒度空气质量建模中的应用 

---
# Uncertainty Estimation on Graphs with Structure Informed Stochastic Partial Differential Equations 

**Title (ZH)**: 基于结构信息的随机偏微分方程的图上不确定性估计 

**Authors**: Fred Xu, Thomas Markovich  

**Link**: [PDF](https://arxiv.org/pdf/2506.06907)  

**Abstract**: Graph Neural Networks have achieved impressive results across diverse network modeling tasks, but accurately estimating uncertainty on graphs remains difficult, especially under distributional shifts. Unlike traditional uncertainty estimation, graph-based uncertainty must account for randomness arising from both the graph's structure and its label distribution, which adds complexity. In this paper, making an analogy between the evolution of a stochastic partial differential equation (SPDE) driven by Matern Gaussian Process and message passing using GNN layers, we present a principled way to design a novel message passing scheme that incorporates spatial-temporal noises motivated by the Gaussian Process approach to SPDE. Our method simultaneously captures uncertainty across space and time and allows explicit control over the covariance kernel smoothness, thereby enhancing uncertainty estimates on graphs with both low and high label informativeness. Our extensive experiments on Out-of-Distribution (OOD) detection on graph datasets with varying label informativeness demonstrate the soundness and superiority of our model to existing approaches. 

**Abstract (ZH)**: 图神经网络在多样化的网络建模任务中取得了显著成果，但在图形上的不确定性估计依然困难，尤其是在分布转移的情况下。与传统的不确定性估计不同，基于图形的不确定性必须同时考虑图形结构和标签分布带来的随机性，这增加了复杂性。在本文中，我们将马特尔高斯过程驱动的随机偏微分方程（SPDE）演化与图神经网络层的消息传递类比，提出了一种基于高斯过程方法的时空噪声纳入的原理性消息传递方案。该方法同时捕捉空间和时间上的不确定性，并允许对协方差核的光滑度进行显式控制，从而提高具有低和高标签信息量的图形上的不确定性估计。在不同标签信息量的图形数据集的异常检测实验中，我们的方法表现出稳健性和优越性。 

---
# Can Biologically Plausible Temporal Credit Assignment Rules Match BPTT for Neural Similarity? E-prop as an Example 

**Title (ZH)**: 生物可实现的时间信用分配规则能否与BPTR对于神经相似性匹配？E-prop为例 

**Authors**: Yuhan Helena Liu, Guangyu Robert Yang, Christopher J. Cueva  

**Link**: [PDF](https://arxiv.org/pdf/2506.06904)  

**Abstract**: Understanding how the brain learns may be informed by studying biologically plausible learning rules. These rules, often approximating gradient descent learning to respect biological constraints such as locality, must meet two critical criteria to be considered an appropriate brain model: (1) good neuroscience task performance and (2) alignment with neural recordings. While extensive research has assessed the first criterion, the second remains underexamined. Employing methods such as Procrustes analysis on well-known neuroscience datasets, this study demonstrates the existence of a biologically plausible learning rule -- namely e-prop, which is based on gradient truncation and has demonstrated versatility across a wide range of tasks -- that can achieve neural data similarity comparable to Backpropagation Through Time (BPTT) when matched for task accuracy. Our findings also reveal that model architecture and initial conditions can play a more significant role in determining neural similarity than the specific learning rule. Furthermore, we observe that BPTT-trained models and their biologically plausible counterparts exhibit similar dynamical properties at comparable accuracies. These results underscore the substantial progress made in developing biologically plausible learning rules, highlighting their potential to achieve both competitive task performance and neural data similarity. 

**Abstract (ZH)**: 理解大脑学习机制可能通过研究生物可实现的学习规则来获得启示。这些规则通常近似梯度下降学习，以遵守局部性等生物约束，必须满足两个关键标准才能被视为合适的大脑模型：（1）良好的神经科学任务表现和（2）与神经记录的一致性。尽管已有大量研究评估了第一标准，但第二标准仍未得到充分考察。通过在著名神经科学数据集上应用Procrustes分析等方法，本研究证明存在一个生物可实现的学习规则——即e-prop，它基于梯度截断，已在多种任务中显示出灵活性，并且在任务准确度匹配的情况下，其神经数据相似度与时间反向传播（BPTT）相当。我们的研究还发现，模型架构和初始条件在决定神经相似度方面发挥的作用可能比具体的学习规则更为重要。此外，我们观察到，在相似准确度水平下，BPTT训练的模型和其生物可实现的对应模型表现出相似的动力学特性。这些结果强调了开发生物可实现学习规则所取得的重大进展，并突显了其在实现竞争性任务性能和神经数据相似度方面的潜力。 

---
# Recursive Semantic Anchoring in ISO 639:2023: A Structural Extension to ISO/TC 37 Frameworks 

**Title (ZH)**: ISO 639:2023中的递归语义锚定：ISO/TC 37框架的结构扩展 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06870)  

**Abstract**: ISO 639:2023 unifies the ISO language-code family and introduces contextual metadata, but it lacks a machine-native mechanism for handling dialectal drift and creole mixtures. We propose a formalisation of recursive semantic anchoring, attaching to every language entity $\chi$ a family of fixed-point operators $\phi_{n,m}$ that model bounded semantic drift via the relation $\phi_{n,m}(\chi) = \chi \oplus \Delta(\chi)$, where $\Delta(\chi)$ is a drift vector in a latent semantic manifold. The base anchor $\phi_{0,0}$ recovers the canonical ISO 639:2023 identity, whereas $\phi_{99,9}$ marks the maximal drift state that triggers a deterministic fallback. Using category theory, we treat the operators $\phi_{n,m}$ as morphisms and drift vectors as arrows in a category $\mathrm{DriftLang}$. A functor $\Phi: \mathrm{DriftLang} \to \mathrm{AnchorLang}$ maps every drifted object to its unique anchor and proves convergence. We provide an RDF/Turtle schema (\texttt{BaseLanguage}, \texttt{DriftedLanguage}, \texttt{ResolvedAnchor}) and worked examples -- e.g., $\phi_{8,4}$ (Standard Mandarin) versus $\phi_{8,7}$ (a colloquial variant), and $\phi_{1,7}$ for Nigerian Pidgin anchored to English. Experiments with transformer models show higher accuracy in language identification and translation on noisy or code-switched input when the $\phi$-indices are used to guide fallback routing. The framework is compatible with ISO/TC 37 and provides an AI-tractable, drift-aware semantic layer for future standards. 

**Abstract (ZH)**: ISO 639:2023 统一了ISO语言代码家族并引入了上下文元数据，但缺乏处理方言漂移和克里奥尔混合的机器原生机制。我们提出了一种递归语义锚定形式化，为每一个语言实体 $\chi$ 附上一族不动点算子 $\phi_{n,m}$，通过关系 $\phi_{n,m}(\chi) = \chi \oplus \Delta(\chi)$ 模型化有界语义漂移，其中 $\Delta(\chi)$ 是在潜在语义流形中的漂移向量。基本锚点 $\phi_{0,0}$ 恢复了ISO 639:2023 的标准标识，而 $\phi_{99,9}$ 标记了触发确定性回退的最大漂移状态。利用范畴论，我们将运算符 $\phi_{n,m}$ 视为范畴 $\mathrm{DriftLang}$ 中的态射，漂移向量视为箭头。函子 $\Phi: \mathrm{DriftLang} \to \mathrm{AnchorLang}$ 将每个漂移对象映射到其唯一的锚定点并证明其收敛性。我们提供了RDF/Turtle模式（BaseLanguage, DriftedLanguage, ResolvedAnchor）及实例——例如 $\phi_{8,4}$（标准普通话）与 $\phi_{8,7}$（一种口语变体）的对比，以及 $\phi_{1,7}$ 对于锚定于英语的尼日利亚皮钦语。实验表明，在使用 $\phi$-索引引导回退路由时，若输入噪声大或混合代码，语言识别和翻译的准确性会更高。该框架兼容ISO/TC 37，并为未来标准提供了一个可由AI处理的、具有漂移感知的语义层。 

---
# SAFE: Finding Sparse and Flat Minima to Improve Pruning 

**Title (ZH)**: SAFE: 寻找稀疏和平坦的极小值以提高剪枝 

**Authors**: Dongyeop Lee, Kwanhee Lee, Jinseok Chung, Namhoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.06866)  

**Abstract**: Sparsifying neural networks often suffers from seemingly inevitable performance degradation, and it remains challenging to restore the original performance despite much recent progress. Motivated by recent studies in robust optimization, we aim to tackle this problem by finding subnetworks that are both sparse and flat at the same time. Specifically, we formulate pruning as a sparsity-constrained optimization problem where flatness is encouraged as an objective. We solve it explicitly via an augmented Lagrange dual approach and extend it further by proposing a generalized projection operation, resulting in novel pruning methods called SAFE and its extension, SAFE$^+$. Extensive evaluations on standard image classification and language modeling tasks reveal that SAFE consistently yields sparse networks with improved generalization performance, which compares competitively to well-established baselines. In addition, SAFE demonstrates resilience to noisy data, making it well-suited for real-world conditions. 

**Abstract (ZH)**: 剪枝神经网络往往伴随着性能下降的问题，即使有近期的进步，恢复原始性能仍然是一个挑战。受到鲁棒优化研究的启发，我们旨在通过寻找同时稀疏和平坦的子网络来解决这一问题。具体而言，我们将剪枝形式化为一个稀疏约束下的优化问题，其中平坦性被鼓励作为目标。我们通过增广拉格朗日对偶方法显式求解，并通过提出通用投影操作进一步扩展，从而得到新的剪枝方法SAFE及其扩展方法SAFE$^+$。在标准图像分类和语言建模任务上的广泛评估表明，SAFE能够一致地生成具有更好泛化性能的稀疏网络，其性能与现有baseline相当。此外，SAFE展现了对噪声数据的鲁棒性，使其更适合实际应用场景。 

---
# High-Fidelity Scientific Simulation Surrogates via Adaptive Implicit Neural Representations 

**Title (ZH)**: 高保真科学模拟代理通过自适应隐式神经表示 

**Authors**: Ziwei Li, Yuhan Duan, Tianyu Xiong, Yi-Tang Chen, Wei-Lun Chao, Han-Wei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06858)  

**Abstract**: Effective surrogate models are critical for accelerating scientific simulations. Implicit neural representations (INRs) offer a compact and continuous framework for modeling spatially structured data, but they often struggle with complex scientific fields exhibiting localized, high-frequency variations. Recent approaches address this by introducing additional features along rigid geometric structures (e.g., grids), but at the cost of flexibility and increased model size. In this paper, we propose a simple yet effective alternative: Feature-Adaptive INR (FA-INR). FA-INR leverages cross-attention to an augmented memory bank to learn flexible feature representations, enabling adaptive allocation of model capacity based on data characteristics, rather than rigid structural assumptions. To further improve scalability, we introduce a coordinate-guided mixture of experts (MoE) that enhances the specialization and efficiency of feature representations. Experiments on three large-scale ensemble simulation datasets show that FA-INR achieves state-of-the-art fidelity while significantly reducing model size, establishing a new trade-off frontier between accuracy and compactness for INR-based surrogates. 

**Abstract (ZH)**: 有效的代理模型对于加速科学模拟至关重要。隐式神经表示（INRs）提供了一种紧凑且连续的框架来建模空间结构化数据，但它们往往难以应对表现出局部高频率变化的复杂科学场。近期的方法通过沿着刚性几何结构（例如网格）引入额外特征来解决这一问题，但代价是灵活性降低和模型尺寸增加。本文提出了一种简单而有效的替代方案：特征自适应隐式神经表示（FA-INR）。FA-INR 利用交叉注意力机制学习灵活的特征表示，基于数据特性而非刚性几何假设进行模型容量的自适应分配。为进一步提高可扩展性，我们引入了坐标引导的专家混合（MoE）机制，增强了特征表示的专业性和效率。在三个大规模集成模拟数据集上的实验表明，FA-INR 在保持顶级准确性的前提下显著减少了模型尺寸，建立了一种基于 INR 的代理模型的新权衡前沿，即准确性和紧凑性之间的权衡。 

---
# A Statistical Framework for Model Selection in LSTM Networks 

**Title (ZH)**: LSTM网络中模型选择的统计框架 

**Authors**: Fahad Mostafa  

**Link**: [PDF](https://arxiv.org/pdf/2506.06840)  

**Abstract**: Long Short-Term Memory (LSTM) neural network models have become the cornerstone for sequential data modeling in numerous applications, ranging from natural language processing to time series forecasting. Despite their success, the problem of model selection, including hyperparameter tuning, architecture specification, and regularization choice remains largely heuristic and computationally expensive. In this paper, we propose a unified statistical framework for systematic model selection in LSTM networks. Our framework extends classical model selection ideas, such as information criteria and shrinkage estimation, to sequential neural networks. We define penalized likelihoods adapted to temporal structures, propose a generalized threshold approach for hidden state dynamics, and provide efficient estimation strategies using variational Bayes and approximate marginal likelihood methods. Several biomedical data centric examples demonstrate the flexibility and improved performance of the proposed framework. 

**Abstract (ZH)**: 基于长短期记忆神经网络的统一统计模型选择框架 

---
# Harnessing Vision-Language Models for Time Series Anomaly Detection 

**Title (ZH)**: 借助视觉语言模型进行时间序列异常检测 

**Authors**: Zelin He, Sarah Alnegheimish, Matthew Reimherr  

**Link**: [PDF](https://arxiv.org/pdf/2506.06836)  

**Abstract**: Time-series anomaly detection (TSAD) has played a vital role in a variety of fields, including healthcare, finance, and industrial monitoring. Prior methods, which mainly focus on training domain-specific models on numerical data, lack the visual-temporal reasoning capacity that human experts have to identify contextual anomalies. To fill this gap, we explore a solution based on vision language models (VLMs). Recent studies have shown the ability of VLMs for visual reasoning tasks, yet their direct application to time series has fallen short on both accuracy and efficiency. To harness the power of VLMs for TSAD, we propose a two-stage solution, with (1) ViT4TS, a vision-screening stage built on a relatively lightweight pretrained vision encoder, which leverages 2-D time-series representations to accurately localize candidate anomalies; (2) VLM4TS, a VLM-based stage that integrates global temporal context and VLM reasoning capacity to refine the detection upon the candidates provided by ViT4TS. We show that without any time-series training, VLM4TS outperforms time-series pretrained and from-scratch baselines in most cases, yielding a 24.6 percent improvement in F1-max score over the best baseline. Moreover, VLM4TS also consistently outperforms existing language-model-based TSAD methods and is on average 36 times more efficient in token usage. 

**Abstract (ZH)**: 基于视觉语言模型的时间序列异常检测 

---
# IMPA-HGAE:Intra-Meta-Path Augmented Heterogeneous Graph Autoencoder 

**Title (ZH)**: IMPA-HGAE：基于元路径增强的异构图自编码器 

**Authors**: Di Lin, Wanjing Ren, Xuanbin Li, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06809)  

**Abstract**: Self-supervised learning (SSL) methods have been increasingly applied to diverse downstream tasks due to their superior generalization capabilities and low annotation costs. However, most existing heterogeneous graph SSL models convert heterogeneous graphs into homogeneous ones via meta-paths for training, which only leverage information from nodes at both ends of meta-paths while underutilizing the heterogeneous node information along the meta-paths. To address this limitation, this paper proposes a novel framework named IMPA-HGAE to enhance target node embeddings by fully exploiting internal node information along meta-paths. Experimental results validate that IMPA-HGAE achieves superior performance on heterogeneous datasets. Furthermore, this paper introduce innovative masking strategies to strengthen the representational capacity of generative SSL models on heterogeneous graph data. Additionally, this paper discuss the interpretability of the proposed method and potential future directions for generative self-supervised learning in heterogeneous graphs. This work provides insights into leveraging meta-path-guided structural semantics for robust representation learning in complex graph scenarios. 

**Abstract (ZH)**: 自监督学习方法（SSL）由于其出色的泛化能力和较低的标注成本，已被广泛应用于多种下游任务。然而，现有的大多数异构图SSL模型通过元路径将异构图转换为同构图进行训练，这仅利用了元路径两端节点的信息，而未充分利用沿元路径的异构节点信息。为解决这一问题，本文提出了一种新颖的框架IMPA-HGAE，通过充分利用沿元路径的内部节点信息来增强目标节点嵌入。实验结果验证了IMPA-HGAE在异构数据集上取得了优越的性能。此外，本文引入了创新的掩码策略，以增强生成性SSL模型在异构图数据上的表示能力。同时，本文讨论了所提出方法的可解释性以及生成式自监督学习在异构图中潜在的研究方向。本工作为在复杂图场景中利用元路径引导的结构语义进行鲁棒表示学习提供了见解。 

---
# Label-semantics Aware Generative Approach for Domain-Agnostic Multilabel Classification 

**Title (ZH)**: 面向领域无关的多标签分类的标签语义aware生成方法 

**Authors**: Subhendu Khatuya, Shashwat Naidu, Saptarshi Ghosh, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2506.06806)  

**Abstract**: The explosion of textual data has made manual document classification increasingly challenging. To address this, we introduce a robust, efficient domain-agnostic generative model framework for multi-label text classification. Instead of treating labels as mere atomic symbols, our approach utilizes predefined label descriptions and is trained to generate these descriptions based on the input text. During inference, the generated descriptions are matched to the pre-defined labels using a finetuned sentence transformer. We integrate this with a dual-objective loss function, combining cross-entropy loss and cosine similarity of the generated sentences with the predefined target descriptions, ensuring both semantic alignment and accuracy. Our proposed model LAGAMC stands out for its parameter efficiency and versatility across diverse datasets, making it well-suited for practical applications. We demonstrate the effectiveness of our proposed model by achieving new state-of-the-art performances across all evaluated datasets, surpassing several strong baselines. We achieve improvements of 13.94% in Micro-F1 and 24.85% in Macro-F1 compared to the closest baseline across all datasets. 

**Abstract (ZH)**: 文本数据的爆炸性增长使得手工文档分类越来越具有挑战性。为此，我们提出了一种稳健高效的领域无关生成模型框架，用于多标签文本分类。我们的方法不仅将标签视为简单的原子符号，还利用预定义的标签描述，并根据输入文本生成这些描述。在推理过程中，生成的描述使用微调的句子变换器与预定义的标签进行匹配。我们整合了双目标损失函数，结合交叉熵损失和生成句子与预定义目标描述的余弦相似度，确保语义对齐和准确性。我们提出的模型LAGAMC因其参数效率和在多种数据集上的适用性而 standout，使其适用于实际应用。我们通过在所有评估数据集上达到新的最佳性能，证明了所提出模型的有效性，超越了几个强基线。与最接近的基线相比，我们在所有数据集上分别实现了13.94%的Micro-F1和24.85%的Macro-F1的改进。 

---
# Is Optimal Transport Necessary for Inverse Reinforcement Learning? 

**Title (ZH)**: 最优传输对于逆强化学习是必要的吗？ 

**Authors**: Zixuan Dong, Yumi Omori, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2506.06793)  

**Abstract**: Inverse Reinforcement Learning (IRL) aims to recover a reward function from expert demonstrations. Recently, Optimal Transport (OT) methods have been successfully deployed to align trajectories and infer rewards. While OT-based methods have shown strong empirical results, they introduce algorithmic complexity, hyperparameter sensitivity, and require solving the OT optimization problems. In this work, we challenge the necessity of OT in IRL by proposing two simple, heuristic alternatives: (1) Minimum-Distance Reward, which assigns rewards based on the nearest expert state regardless of temporal order; and (2) Segment-Matching Reward, which incorporates lightweight temporal alignment by matching agent states to corresponding segments in the expert trajectory. These methods avoid optimization, exhibit linear-time complexity, and are easy to implement. Through extensive evaluations across 32 online and offline benchmarks with three reinforcement learning algorithms, we show that our simple rewards match or outperform recent OT-based approaches. Our findings suggest that the core benefits of OT may arise from basic proximity alignment rather than its optimal coupling formulation, advocating for reevaluation of complexity in future IRL design. 

**Abstract (ZH)**: 逆向强化学习（IRL）旨在从专家演示中恢复奖励函数。最近，最优传输（OT）方法已被成功应用于对齐轨迹并推断奖励。尽管基于OT的方法展示了强大的实证结果，但它们引入了算法复杂性、超参数敏感性，并需要解决OT优化问题。在本文中，我们通过提出两种简单且启发式的替代方案挑战OT在IRL中的必要性：（1）最小距离奖励，基于最近的专家状态分配奖励，而不考虑时间顺序；（2）段匹配奖励，通过将代理状态与专家轨迹中的相应段匹配来引入轻量级的时间对齐。这些方法避免了优化，具有线性时间复杂性，并易于实现。通过在32个在线和离线基准上的广泛评估，结合三种强化学习算法，我们展示出我们的简单奖励能够匹配或超越最近的OT基方法。我们的发现表明，OT的核心益处可能源于基本的邻近对齐，而非其最佳耦合形式，从而为未来IRL设计中的复杂性重新评估提供依据。 

---
# Feature-Based Instance Neighbor Discovery: Advanced Stable Test-Time Adaptation in Dynamic World 

**Title (ZH)**: 基于特征的实例邻居发现：动态世界中高级稳定的测试时自适应 

**Authors**: Qinting Jiang, Chuyang Ye, Dongyan Wei, Bingli Wang, Yuan Xue, Jingyan Jiang, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06782)  

**Abstract**: Despite progress, deep neural networks still suffer performance declines under distribution shifts between training and test domains, leading to a substantial decrease in Quality of Experience (QoE) for applications. Existing test-time adaptation (TTA) methods are challenged by dynamic, multiple test distributions within batches. We observe that feature distributions across different domains inherently cluster into distinct groups with varying means and variances. This divergence reveals a critical limitation of previous global normalization strategies in TTA, which inevitably distort the original data characteristics. Based on this insight, we propose Feature-based Instance Neighbor Discovery (FIND), which comprises three key components: Layer-wise Feature Disentanglement (LFD), Feature Aware Batch Normalization (FABN) and Selective FABN (S-FABN). LFD stably captures features with similar distributions at each layer by constructing graph structures. While FABN optimally combines source statistics with test-time distribution specific statistics for robust feature representation. Finally, S-FABN determines which layers require feature partitioning and which can remain unified, thereby enhancing inference efficiency. Extensive experiments demonstrate that FIND significantly outperforms existing methods, achieving a 30\% accuracy improvement in dynamic scenarios while maintaining computational efficiency. 

**Abstract (ZH)**: 基于特征的实例邻居发现：应对分布转移的测试时自适应方法 

---
# Depth-Optimal Quantum Layout Synthesis as SAT 

**Title (ZH)**: 深度最优量子布局合成作为SAT问题 

**Authors**: Anna B. Jakobsen, Anders B. Clausen, Jaco van de Pol, Irfansha Shaik  

**Link**: [PDF](https://arxiv.org/pdf/2506.06752)  

**Abstract**: Quantum circuits consist of gates applied to qubits. Current quantum hardware platforms impose connectivity restrictions on binary CX gates. Hence, Layout Synthesis is an important step to transpile quantum circuits before they can be executed. Since CX gates are noisy, it is important to reduce the CX count or CX depth of the mapped circuits.
We provide a new and efficient encoding of Quantum-circuit Layout Synthesis in SAT. Previous SAT encodings focused on gate count and CX-gate count. Our encoding instead guarantees that we find mapped circuits with minimal circuit depth or minimal CX-gate depth. We use incremental SAT solving and parallel plans for an efficient encoding. This results in speedups of more than 10-100x compared to OLSQ2, which guarantees depth-optimality. But minimizing depth still takes more time than minimizing gate count with Q-Synth.
We correlate the noise reduction achieved by simulating circuits after (CX)-count and (CX)-depth reduction. We find that minimizing for CX-count correlates better with reducing noise than minimizing for CX-depth. However, taking into account both CX-count and CX-depth provides the best noise reduction. 

**Abstract (ZH)**: 量子电路布局合成的新型高效SAT编码研究 

---
# Ai-Driven Vulnerability Analysis in Smart Contracts: Trends, Challenges and Future Directions 

**Title (ZH)**: 基于人工智能的智能合约漏洞分析：趋势、挑战与未来方向 

**Authors**: Mesut Ozdag  

**Link**: [PDF](https://arxiv.org/pdf/2506.06735)  

**Abstract**: Smart contracts, integral to blockchain ecosystems, enable decentralized applications to execute predefined operations without intermediaries. Their ability to enforce trustless interactions has made them a core component of platforms such as Ethereum. Vulnerabilities such as numerical overflows, reentrancy attacks, and improper access permissions have led to the loss of millions of dollars throughout the blockchain and smart contract sector. Traditional smart contract auditing techniques such as manual code reviews and formal verification face limitations in scalability, automation, and adaptability to evolving development patterns. As a result, AI-based solutions have emerged as a promising alternative, offering the ability to learn complex patterns, detect subtle flaws, and provide scalable security assurances. This paper examines novel AI-driven techniques for vulnerability detection in smart contracts, focusing on machine learning, deep learning, graph neural networks, and transformer-based models. This paper analyzes how each technique represents code, processes semantic information, and responds to real world vulnerability classes. We also compare their strengths and weaknesses in terms of accuracy, interpretability, computational overhead, and real time applicability. Lastly, it highlights open challenges and future opportunities for advancing this domain. 

**Abstract (ZH)**: 智能合约是区块链生态系统的关键组成部分，能够使去中心化应用在没有中介的情况下执行预定义的操作。它们能够在不信任的交互中强制执行信任，成为以太坊等平台的核心组件。数值溢出、重入攻击和不当访问权限等漏洞已导致区块链和智能合约领域损失数百万美元。传统的智能合约审计技术如人工代码审查和形式化验证面临可扩展性、自动化和适应不断变化的开发模式的局限性。因此，基于AI的解决方案已成为有前途的替代方案，能够学习复杂模式、检测细微缺陷并提供可扩展的安全保证。本文探讨了新型AI驱动的智能合约漏洞检测技术，重点关注机器学习、深度学习、图神经网络和变压器模型。本文分析了每种技术如何表示代码、处理语义信息以及应对真实世界的漏洞类别。我们还比较了它们在准确性、可解释性、计算开销和实时适用性方面的优缺点。最后，本文指出了该领域的开放挑战和未来机遇。 

---
# Neural Spectral Band Generation for Audio Coding 

**Title (ZH)**: 基于神经网络的音素频带生成音频编码 

**Authors**: Woongjib Choi, Byeong Hyeon Kim, Hyungseob Lim, Inseon Jang, Hong-Goo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06732)  

**Abstract**: Audio bandwidth extension is the task of reconstructing missing high frequency components of bandwidth-limited audio signals, where bandwidth limitation is a common issue for audio signals due to several reasons, including channel capacity and data constraints. While conventional spectral band replication is a well-established parametric approach to audio bandwidth extension, the SBR usually entails coarse feature extraction and reconstruction techniques, which leads to limitations when processing various types of audio signals. In parallel, numerous deep neural network-based audio bandwidth extension methods have been proposed. These DNN-based methods are usually referred to as blind BWE, as these methods do not rely on prior information extracted from original signals, and only utilize given low frequency band signals to estimate missing high frequency components. In order to replace conventional SBR with DNNs, simply adopting existing DNN-based methodologies results in suboptimal performance due to the blindness of these methods. My proposed research suggests a new approach to parametric non-blind bandwidth extension, as DNN-based side information extraction and DNN-based bandwidth extension are performed only at the front and end of the audio coding pipeline. 

**Abstract (ZH)**: 宽带扩展是 reconstruction 缺失的高频频带组件的任务，其中频带限制是由于多种原因（包括信道容量和数据约束）对音频信号的常见问题。虽然传统的频谱带复制是音频宽带扩展的成熟参数化方法，但频谱带复制通常涉及粗略的特征提取和重建技术，这在处理不同类型音频信号时会带来限制。与此同时，基于深度神经网络的音频宽带扩展方法也被广泛提出。这些基于 DNN 的方法通常被称为盲宽带扩展，因为这些方法不依赖于从原始信号中提取的先验信息，并且仅利用给定的低频带信号来估计缺失的高频组件。为了用 DNN 替换传统的频谱带复制，直接采用现有的 DNN 基础方法会导致性能不佳，因为这些方法是盲目的。我提出的研究所建议的是一种新的参数化非盲宽带扩展方法，因为在音频编码管道的前后分别仅进行基于 DNN 的次要信息提取和基于 DNN 的宽带扩展。 

---
# Do Protein Transformers Have Biological Intelligence? 

**Title (ZH)**: 蛋白质变换器具有生物智能吗？ 

**Authors**: Fudong Lin, Wanrou Du, Jinchan Liu, Tarikul Milon, Shelby Meche, Wu Xu, Xiaoqi Qin, Xu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06701)  

**Abstract**: Deep neural networks, particularly Transformers, have been widely adopted for predicting the functional properties of proteins. In this work, we focus on exploring whether Protein Transformers can capture biological intelligence among protein sequences. To achieve our goal, we first introduce a protein function dataset, namely Protein-FN, providing over 9000 protein data with meaningful labels. Second, we devise a new Transformer architecture, namely Sequence Protein Transformers (SPT), for computationally efficient protein function predictions. Third, we develop a novel Explainable Artificial Intelligence (XAI) technique called Sequence Score, which can efficiently interpret the decision-making processes of protein models, thereby overcoming the difficulty of deciphering biological intelligence bided in Protein Transformers. Remarkably, even our smallest SPT-Tiny model, which contains only 5.4M parameters, demonstrates impressive predictive accuracy, achieving 94.3% on the Antibiotic Resistance (AR) dataset and 99.6% on the Protein-FN dataset, all accomplished by training from scratch. Besides, our Sequence Score technique helps reveal that our SPT models can discover several meaningful patterns underlying the sequence structures of protein data, with these patterns aligning closely with the domain knowledge in the biology community. We have officially released our Protein-FN dataset on Hugging Face Datasets this https URL. Our code is available at this https URL. 

**Abstract (ZH)**: 深神经网络，尤其是变换器，已被广泛用于预测蛋白质的功能属性。在这项工作中，我们重点关注探究蛋白质变换器是否能够捕捉蛋白质序列中的生物智能。为了实现这一目标，我们首先介绍了蛋白质功能数据集Protein-FN，提供了超过9000个带有有意义标签的蛋白质数据。其次，我们设计了一种新的变换器架构，称为序列蛋白质变换器(SPT)，以实现高效的蛋白质功能预测。第三，我们开发了一种新型的可解释人工智能(XAI)技术，称为序列得分(Sequence Score)，该技术能够高效地解释蛋白质模型的决策过程，从而克服解释蛋白质变换器中蕴含的生物智能的困难。值得注意的是，即使是我们最小的SPT-Tiny模型，仅包含5.4M参数，也展示了令人印象深刻的预测准确性，在抗生素耐药性(AR)数据集中达到了94.3%，在Protein-FN数据集中达到了99.6%，均通过从零开始训练实现。此外，我们的序列得分技术帮助揭示了我们的SPT模型能够发现蛋白质数据序列结构背后的多个有意义模式，这些模式与生物领域中的专业知识高度吻合。我们已正式在Hugging Face Datasets上发布了Protein-FN数据集，详细信息请访问以下链接：https://huggingface.co/datasets/Protein-FN。我们的代码可以在以下链接中获取：https://。 

---
# Design and Implementation of a RISC-V SoC with Custom DSP Accelerators for Edge Computing 

**Title (ZH)**: 基于边缘计算的自定义DSP加速器嵌入的RISC-V SoC设计与实现 

**Authors**: Priyanshu Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2506.06693)  

**Abstract**: This paper presents a comprehensive analysis of the RISC-V instruction set architecture, focusing on its modular design, implementation challenges, and performance characteristics. We examine the RV32I base instruction set with extensions for multiplication (M) and atomic operations (A). Through cycle-accurate simulation of a pipelined implementation, we evaluate performance metrics including CPI (cycles per instruction) and power efficiency. Our results demonstrate RISC-V's advantages in embedded systems and its scalability for custom accelerators. Comparative analysis shows a 17% reduction in power consumption compared to ARM Cortex-M0 implementations in similar process nodes. The open-standard nature of RISC-V provides significant flexibility for domain-specific optimizations. 

**Abstract (ZH)**: 本文对RISC-V指令集架构进行了全面分析，重点探讨其模块化设计、实现挑战及性能特征。我们研究了RV32I基础指令集，并包括乘法扩展(M)和原子操作扩展(A)。通过对流水线实现的时钟周期精确仿真，我们评估了每条指令周期数(CPI)和功耗效率等性能指标。结果表明，RISC-V在嵌入式系统中的优势及其面向自定义加速器的可扩展性。与类似工艺节点的ARM Cortex-M0实现相比，功耗降低了17%，展示了RISC-V开放标准带来的灵活性。 

---
# Non-Intrusive Load Monitoring Based on Image Load Signatures and Continual Learning 

**Title (ZH)**: 基于图像负载签名和持续学习的非侵入式负载监测 

**Authors**: Olimjon Toirov, Wei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06637)  

**Abstract**: Non-Intrusive Load Monitoring (NILM) identifies the operating status and energy consumption of each electrical device in the circuit by analyzing the electrical signals at the bus, which is of great significance for smart power management. However, the complex and changeable load combinations and application environments lead to the challenges of poor feature robustness and insufficient model generalization of traditional NILM methods. To this end, this paper proposes a new non-intrusive load monitoring method that integrates "image load signature" and continual learning. This method converts multi-dimensional power signals such as current, voltage, and power factor into visual image load feature signatures, and combines deep convolutional neural networks to realize the identification and classification of multiple devices; at the same time, self-supervised pre-training is introduced to improve feature generalization, and continual online learning strategies are used to overcome model forgetting to adapt to the emergence of new loads. This paper conducts a large number of experiments on high-sampling rate load datasets, and compares a variety of existing methods and model variants. The results show that the proposed method has achieved significant improvements in recognition accuracy. 

**Abstract (ZH)**: 非侵入式负载监测方法整合“图像负载签名”和持续学习技术 

---
# CAtCh: Cognitive Assessment through Cookie Thief 

**Title (ZH)**: CAtCh: 基于认知评估的饼干小偷任务 

**Authors**: Joseph T Colonel, Carolyn Hagler, Guiselle Wismer, Laura Curtis, Jacqueline Becker, Juan Wisnivesky, Alex Federman, Gaurav Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2506.06603)  

**Abstract**: Several machine learning algorithms have been developed for the prediction of Alzheimer's disease and related dementia (ADRD) from spontaneous speech. However, none of these algorithms have been translated for the prediction of broader cognitive impairment (CI), which in some cases is a precursor and risk factor of ADRD. In this paper, we evaluated several speech-based open-source methods originally proposed for the prediction of ADRD, as well as methods from multimodal sentiment analysis for the task of predicting CI from patient audio recordings. Results demonstrated that multimodal methods outperformed unimodal ones for CI prediction, and that acoustics-based approaches performed better than linguistics-based ones. Specifically, interpretable acoustic features relating to affect and prosody were found to significantly outperform BERT-based linguistic features and interpretable linguistic features, respectively. All the code developed for this study is available at this https URL. 

**Abstract (ZH)**: 几种机器学习算法已被用于自发言语预测阿尔茨海默病及相关痴呆（ADRD），然而这些算法尚未用于更广泛认知损害（CI）的预测，而后者在某些情况下是ADRD的前兆和风险因素。本文评估了几种 originally proposed 用于预测ADRD的基于言语的开源方法，以及来自多模态情感分析的方法，用于从患者录音中预测CI的任务。结果表明，多模态方法在CI预测中优于单模态方法，而基于声学的方法优于基于语言学的方法。具体而言，与情感和语调相关的可解释声学特征显著优于基于BERT的语言学特征和可解释语言学特征。本文开发的所有代码均可在此 URL 获取。 

---
# Future of Work with AI Agents: Auditing Automation and Augmentation Potential across the U.S. Workforce 

**Title (ZH)**: AI代理的未来工作：审计美国劳动力的自动化与增强潜力 

**Authors**: Yijia Shao, Humishka Zope, Yucheng Jiang, Jiaxin Pei, David Nguyen, Erik Brynjolfsson, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06576)  

**Abstract**: The rapid rise of compound AI systems (a.k.a., AI agents) is reshaping the labor market, raising concerns about job displacement, diminished human agency, and overreliance on automation. Yet, we lack a systematic understanding of the evolving landscape. In this paper, we address this gap by introducing a novel auditing framework to assess which occupational tasks workers want AI agents to automate or augment, and how those desires align with the current technological capabilities. Our framework features an audio-enhanced mini-interview to capture nuanced worker desires and introduces the Human Agency Scale (HAS) as a shared language to quantify the preferred level of human involvement. Using this framework, we construct the WORKBank database, building on the U.S. Department of Labor's O*NET database, to capture preferences from 1,500 domain workers and capability assessments from AI experts across over 844 tasks spanning 104 occupations. Jointly considering the desire and technological capability divides tasks in WORKBank into four zones: Automation "Green Light" Zone, Automation "Red Light" Zone, R&D Opportunity Zone, Low Priority Zone. This highlights critical mismatches and opportunities for AI agent development. Moving beyond a simple automate-or-not dichotomy, our results reveal diverse HAS profiles across occupations, reflecting heterogeneous expectations for human involvement. Moreover, our study offers early signals of how AI agent integration may reshape the core human competencies, shifting from information-focused skills to interpersonal ones. These findings underscore the importance of aligning AI agent development with human desires and preparing workers for evolving workplace dynamics. 

**Abstract (ZH)**: 快速崛起的组合AI系统（即AI代理）正重塑劳动市场，引发了关于工作岗位替代、人类自主性减弱以及过度依赖自动化的担忧。然而，我们缺乏对这一演变景观的系统性理解。本文通过引入一种新的审计框架来解决这一缺口，该框架评估工人希望AI代理自动化或增强哪些职业任务，以及这些愿望与当前技术能力的契合程度。我们的框架包括音频增强的简短访谈，以捕捉工人复杂的需求，并引入人类自主性量表（HAS）作为共享语言来量化期望的人类参与水平。通过该框架，我们构建了WORKBank数据库，基于美国劳工部的O*NET数据库，涵盖了来自1,500名领域工人和来自844个任务（涉及104种职业）的AI专家的能力评估。结合意愿和技术能力，将WORKBank的任务划分为四个区域：自动化“绿灯”区、自动化“红灯”区、研发机会区、低优先级区。这突显了关键的不匹配和AI代理开发中的机遇。超越简单地决定自动化与否的二分法，我们的结果揭示了不同职业中具有多样性的HAS配置文件，反映了对人类参与的异质性预期。此外，我们的研究提供了有关AI代理整合可能如何重塑核心人类技能的早期信号，从信息聚焦技能转向人际技能。这些发现强调了将AI代理开发与人类愿望相一致的重要性，并为工人适应不断变化的工作场所动态做好准备。 

---
# Graph Persistence goes Spectral 

**Title (ZH)**: 谱图持久性 

**Authors**: Mattie Ji, Amauri H. Souza, Vikas Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.06571)  

**Abstract**: Including intricate topological information (e.g., cycles) provably enhances the expressivity of message-passing graph neural networks (GNNs) beyond the Weisfeiler-Leman (WL) hierarchy. Consequently, Persistent Homology (PH) methods are increasingly employed for graph representation learning. In this context, recent works have proposed decorating classical PH diagrams with vertex and edge features for improved expressivity. However, due to their dependence on features, these methods still fail to capture basic graph structural information. In this paper, we propose SpectRe -- a new topological descriptor for graphs that integrates spectral information into PH diagrams. Notably, SpectRe is strictly more expressive than existing descriptors on graphs. We also introduce notions of global and local stability to analyze existing descriptors and establish that SpectRe is locally stable. Finally, experiments on synthetic and real-world datasets demonstrate the effectiveness of SpectRe and its potential to enhance the capabilities of graph models in relevant learning tasks. 

**Abstract (ZH)**: 包括复杂的拓扑信息（如环）可证地增强消息传递图神经网络（GNNs）的表征能力，超越魏谢夫勒-列曼（WL）层次结构。因此，持久同调（PH）方法越来越多地被用于图表示学习。在此背景下，最近的工作提出了在经典PH图上添加顶点和边特征来提高表征能力。然而，由于对特征的依赖性，这些方法仍然无法捕捉基本的图结构信息。在本文中，我们提出了一种新的拓扑描述符SpectRe，它将谱信息整合到PH图中。值得注意的是，SpectRe在图上的表征能力严格优于现有的描述符。我们还引入了全局稳定性和局部稳定性概念来分析现有描述符，并证明SpectRe具有局部稳定性。最后，合成和真实世界数据集上的实验表明SpectRe的有效性及其在相关学习任务中增强图模型能力的潜力。 

---
# AS-ASR: A Lightweight Framework for Aphasia-Specific Automatic Speech Recognition 

**Title (ZH)**: AS-ASR：一种轻量级的专用于失语症的自动语音识别框架 

**Authors**: Chen Bao, Chuanbing Huo, Qinyu Chen, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06566)  

**Abstract**: This paper proposes AS-ASR, a lightweight aphasia-specific speech recognition framework based on Whisper-tiny, tailored for low-resource deployment on edge devices. Our approach introduces a hybrid training strategy that systematically combines standard and aphasic speech at varying ratios, enabling robust generalization, and a GPT-4-based reference enhancement method that refines noisy aphasic transcripts, improving supervision quality. We conduct extensive experiments across multiple data mixing configurations and evaluation settings. Results show that our fine-tuned model significantly outperforms the zero-shot baseline, reducing WER on aphasic speech by over 30% while preserving performance on standard speech. The proposed framework offers a scalable, efficient solution for real-world disordered speech recognition. 

**Abstract (ZH)**: 本文提出了一种基于Whisper-tiny的轻量级自闭塞性失语症识别框架AS-ASR，针对边缘设备低资源部署进行了优化。该方法引入了一种混合训练策略，系统地结合标准言语和失语症言语的不同比例数据，实现了稳健的泛化能力，并提出了一种基于GPT-4的参考增强方法，以精化失语症记录中的噪声，提高监督质量。我们在多种数据混搭配置和评估设置下进行了广泛的实验。结果表明，我们的微调模型显著优于零样本基线，失语症言语的wer降低了超过30%，同时保持了对标准言语的性能。所提出框架提供了面向实际失序言语识别的可扩展和高效解决方案。 

---
# KramaBench: A Benchmark for AI Systems on Data-to-Insight Pipelines over Data Lakes 

**Title (ZH)**: KramaBench：用于数据湖上数据到洞察管道的AI系统基准测试 

**Authors**: Eugenie Lai, Gerardo Vitagliano, Ziyu Zhang, Sivaprasad Sudhir, Om Chabra, Anna Zeng, Anton A. Zabreyko, Chenning Li, Ferdi Kossmann, Jialin Ding, Jun Chen, Markos Markakis, Matthew Russo, Weiyang Wang, Ziniu Wu, Michael J. Cafarella, Lei Cao, Samuel Madden, Tim Kraska  

**Link**: [PDF](https://arxiv.org/pdf/2506.06541)  

**Abstract**: Constructing real-world data-to-insight pipelines often involves data extraction from data lakes, data integration across heterogeneous data sources, and diverse operations from data cleaning to analysis. The design and implementation of data science pipelines require domain knowledge, technical expertise, and even project-specific insights. AI systems have shown remarkable reasoning, coding, and understanding capabilities. However, it remains unclear to what extent these capabilities translate into successful design and execution of such complex pipelines. We introduce KRAMABENCH: a benchmark composed of 104 manually-curated real-world data science pipelines spanning 1700 data files from 24 data sources in 6 different domains. We show that these pipelines test the end-to-end capabilities of AI systems on data processing, requiring data discovery, wrangling and cleaning, efficient processing, statistical reasoning, and orchestrating data processing steps given a high-level task. Our evaluation tests 5 general models and 3 code generation models using our reference framework, DS-GURU, which instructs the AI model to decompose a question into a sequence of subtasks, reason through each step, and synthesize Python code that implements the proposed design. Our results on KRAMABENCH show that, although the models are sufficiently capable of solving well-specified data science code generation tasks, when extensive data processing and domain knowledge are required to construct real-world data science pipelines, existing out-of-box models fall short. Progress on KramaBench represents crucial steps towards developing autonomous data science agents for real-world applications. Our code, reference framework, and data are available at this https URL. 

**Abstract (ZH)**: 构建现实世界的数据到见解管道通常涉及从数据湖中抽取数据、跨异构数据源进行数据集成，以及从数据清洗到分析的各种操作。数据科学管道的设计与实现需要领域知识、技术专长，甚至项目特定的洞察。AI系统展示了卓越的推理、编码和理解能力。然而，这些能力在成功设计和执行如此复杂的管道中的程度尚不清楚。我们介绍了KRAMABENCH：一个由104个手工整理的真实世界数据科学管道组成的基准，这些管道涵盖了来自6个不同领域、24个数据源的1700个数据文件。我们展示了这些管道测试了AI系统在整个数据处理流程中的端到端能力，包括数据发现、整理和清洗、高效处理、统计推理以及根据高级任务协调数据处理步骤。我们使用参考框架DS-GURU测试了5个通用模型和3个代码生成模型，该框架指导AI模型将一个问题分解为一系列子任务，逐步推理并通过合成Python代码实现提出的解决方案。KRAMABENCH的结果显示，尽管模型足够强大以解决明确定义的数据科学代码生成任务，但当需要大量数据处理和领域知识来构建真实世界的数据科学管道时，现有的开箱即用模型表现不足。KramaBench的进步代表了开发自主数据科学代理应用于真实世界应用的重要步骤。我们的代码、参考框架和数据可在以下网址获取。 

---
# The Economic Dispatch of Power-to-Gas Systems with Deep Reinforcement Learning:Tackling the Challenge of Delayed Rewards with Long-Term Energy Storage 

**Title (ZH)**: 基于深度强化学习的气体储能系统电力调度：长周期能源存储下的延迟奖励挑战解决方法 

**Authors**: Manuel Sage, Khalil Al Handawi, Yaoyao Fiona Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06484)  

**Abstract**: Power-to-Gas (P2G) technologies gain recognition for enabling the integration of intermittent renewables, such as wind and solar, into electricity grids. However, determining the most cost-effective operation of these systems is complex due to the volatile nature of renewable energy, electricity prices, and loads. Additionally, P2G systems are less efficient in converting and storing energy compared to battery energy storage systems (BESs), and the benefits of converting electricity into gas are not immediately apparent. Deep Reinforcement Learning (DRL) has shown promise in managing the operation of energy systems amidst these uncertainties. Yet, DRL techniques face difficulties with the delayed reward characteristic of P2G system operation. Previous research has mostly focused on short-term studies that look at the energy conversion process, neglecting the long-term storage capabilities of P2G.
This study presents a new method by thoroughly examining how DRL can be applied to the economic operation of P2G systems, in combination with BESs and gas turbines, over extended periods. Through three progressively more complex case studies, we assess the performance of DRL algorithms, specifically Deep Q-Networks and Proximal Policy Optimization, and introduce modifications to enhance their effectiveness. These modifications include integrating forecasts, implementing penalties on the reward function, and applying strategic cost calculations, all aimed at addressing the issue of delayed rewards. Our findings indicate that while DRL initially struggles with the complex decision-making required for P2G system operation, the adjustments we propose significantly improve its capability to devise cost-effective operation strategies, thereby unlocking the potential for long-term energy storage in P2G technologies. 

**Abstract (ZH)**: 基于深度强化学习的Power-to-Gas系统经济运行方法研究：结合电池储能系统和燃气涡轮机的长期优化 

---
# Noise Consistency Regularization for Improved Subject-Driven Image Synthesis 

**Title (ZH)**: 基于噪声一致性正则化以提高主体驱动的图像合成 

**Authors**: Yao Ni, Song Wen, Piotr Koniusz, Anoop Cherian  

**Link**: [PDF](https://arxiv.org/pdf/2506.06483)  

**Abstract**: Fine-tuning Stable Diffusion enables subject-driven image synthesis by adapting the model to generate images containing specific subjects. However, existing fine-tuning methods suffer from two key issues: underfitting, where the model fails to reliably capture subject identity, and overfitting, where it memorizes the subject image and reduces background diversity. To address these challenges, we propose two auxiliary consistency losses for diffusion fine-tuning. First, a prior consistency regularization loss ensures that the predicted diffusion noise for prior (non-subject) images remains consistent with that of the pretrained model, improving fidelity. Second, a subject consistency regularization loss enhances the fine-tuned model's robustness to multiplicative noise modulated latent code, helping to preserve subject identity while improving diversity. Our experimental results demonstrate that incorporating these losses into fine-tuning not only preserves subject identity but also enhances image diversity, outperforming DreamBooth in terms of CLIP scores, background variation, and overall visual quality. 

**Abstract (ZH)**: Fine-tuning Stable Diffusion 通过适应模型生成包含特定主题的图像，实现主题驱动的图像合成。然而，现有的细调方法面临两个关键问题：欠拟合和过拟合。针对这些挑战，我们提出了两种辅助一致性损失以优化扩散模型的细调。首先，先验一致性正则化损失确保非主题图像的预测扩散噪声与预训练模型的噪声保持一致，从而提高保真度。其次，主题一致性正则化损失增强细调模型对动态噪声调制潜在代码的鲁棒性，有助于保持主题身份的同时提高多样性。实验结果表明，在细调过程中引入这些损失不仅保留了主题身份，还提升了图像多样性，在CLIP得分、背景变化和总体视觉质量方面优于DreamBooth。 

---
# Edge-Enabled Collaborative Object Detection for Real-Time Multi-Vehicle Perception 

**Title (ZH)**: 边缘赋能的协同目标检测用于实时多车辆感知 

**Authors**: Everett Richards, Bipul Thapa, Lena Mashayekhy  

**Link**: [PDF](https://arxiv.org/pdf/2506.06474)  

**Abstract**: Accurate and reliable object detection is critical for ensuring the safety and efficiency of Connected Autonomous Vehicles (CAVs). Traditional on-board perception systems have limited accuracy due to occlusions and blind spots, while cloud-based solutions introduce significant latency, making them unsuitable for real-time processing demands required for autonomous driving in dynamic environments. To address these challenges, we introduce an innovative framework, Edge-Enabled Collaborative Object Detection (ECOD) for CAVs, that leverages edge computing and multi-CAV collaboration for real-time, multi-perspective object detection. Our ECOD framework integrates two key algorithms: Perceptive Aggregation and Collaborative Estimation (PACE) and Variable Object Tally and Evaluation (VOTE). PACE aggregates detection data from multiple CAVs on an edge server to enhance perception in scenarios where individual CAVs have limited visibility. VOTE utilizes a consensus-based voting mechanism to improve the accuracy of object classification by integrating data from multiple CAVs. Both algorithms are designed at the edge to operate in real-time, ensuring low-latency and reliable decision-making for CAVs. We develop a hardware-based controlled testbed consisting of camera-equipped robotic CAVs and an edge server to evaluate the efficacy of our framework. Our experimental results demonstrate the significant benefits of ECOD in terms of improved object classification accuracy, outperforming traditional single-perspective onboard approaches by up to 75%, while ensuring low-latency, edge-driven real-time processing. This research highlights the potential of edge computing to enhance collaborative perception for latency-sensitive autonomous systems. 

**Abstract (ZH)**: 基于边缘计算的协作对象检测框架（ECOD）：提升连接自动驾驶车辆的安全性和效率 

---
# WISCA: A Consensus-Based Approach to Harmonizing Interpretability in Tabular Datasets 

**Title (ZH)**: WISCA：一种基于共识的方法以谐调表数据集的可解释性 

**Authors**: Antonio Jesús Banegas-Luna, Horacio Pérez-Sánchez, Carlos Martínez-Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2506.06455)  

**Abstract**: While predictive accuracy is often prioritized in machine learning (ML) models, interpretability remains essential in scientific and high-stakes domains. However, diverse interpretability algorithms frequently yield conflicting explanations, highlighting the need for consensus to harmonize results. In this study, six ML models were trained on six synthetic datasets with known ground truths, utilizing various model-agnostic interpretability techniques. Consensus explanations were generated using established methods and a novel approach: WISCA (Weighted Scaled Consensus Attributions), which integrates class probability and normalized attributions. WISCA consistently aligned with the most reliable individual method, underscoring the value of robust consensus strategies in improving explanation reliability. 

**Abstract (ZH)**: 尽管在机器学习模型中预测准确性常常被优先考虑，但在科学和高风险领域中，解释性仍然至关重要。然而，多种解释性算法经常产生相互矛盾的解释，突显了达成共识以协调结果的必要性。在这项研究中，研究人员使用各种模型无关的解释性技术，在六个含有已知真实值的合成数据集上训练了六种机器学习模型，并通过既定方法和一种新方法——加权规范化共识归因（WISCA，Weighted Scaled Consensus Attributions）生成了共识解释。WISCA 一直与最可靠的个体方法一致，强调了采用稳健的共识策略以提高解释可靠性的重要性。 

---
# Unlocking Chemical Insights: Superior Molecular Representations from Intermediate Encoder Layers 

**Title (ZH)**: 解锁化学洞见：中间编码层的优质分子表示 

**Authors**: Luis Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.06443)  

**Abstract**: Pretrained molecular encoders have become indispensable in computational chemistry for tasks such as property prediction and molecular generation. However, the standard practice of relying solely on final-layer embeddings for downstream tasks may discard valuable information. In this work, we challenge this convention by conducting a comprehensive layer-wise analysis of five diverse molecular encoders across 22 ADMET property prediction tasks. Our results demonstrate that embeddings from intermediate layers consistently outperform final-layer representations. Specifically, using fixed embeddings from the optimal intermediate layers improved downstream performance by an average of 5.4%, reaching gains up to 28.6%. Furthermore, finetuning up to these intermediate layers yielded even greater average improvements of 8.5%, with performance increases as high as 40.8%, achieving new state-of-the-art results on several benchmarks. Additionally, a strong positive correlation between fixed embedding performance and finetuning outcomes supports an efficient evaluate-then-finetune approach, enabling identification of optimal layers with reduced computational cost. These findings highlight the importance of exploring the full representational depth of molecular encoders to achieve substantial performance improvements and computational efficiency. The code is made publicly available at this https URL. 

**Abstract (ZH)**: 预训练分子编码器在计算化学中的应用对于诸如性质预测和分子生成等任务已经变得不可或缺。然而，仅依赖最终层嵌入进行下游任务的标准做法可能会丢弃有价值的信息。在本工作中，我们通过在22项ADMET性质预测任务中对五种不同的分子编码器进行全面的逐层分析，挑战了这一惯例。结果显示，中间层嵌入始终优于最终层表示。具体来说，使用来自最优中间层的固定嵌入，下游性能平均提高了5.4%，最高可达28.6%。此外，微调至这些中间层带来了更显著的平均改善，平均提高了8.5%，最高可达40.8%，在多个基准上达到了新的最佳结果。另外，固定嵌入性能与微调结果之间的强烈正相关支持了一种高效的评估-然后微调方法，可以降低计算成本以识别最优层。这些发现突出了探索分子编码器的全部表示深度以实现显著性能提升和计算效率的重要性。代码已在此处公开：这个链接。 

---
# TimeWak: Temporal Chained-Hashing Watermark for Time Series Data 

**Title (ZH)**: TimeWak: 时间链式哈希水印用于时间序列数据 

**Authors**: Zhi Wen Soi, Chaoyi Zhu, Fouad Abiad, Aditya Shankar, Jeroen M. Galjaard, Huijuan Wang, Lydia Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06407)  

**Abstract**: Synthetic time series generated by diffusion models enable sharing privacy-sensitive datasets, such as patients' functional MRI records. Key criteria for synthetic data include high data utility and traceability to verify the data source. Recent watermarking methods embed in homogeneous latent spaces, but state-of-the-art time series generators operate in real space, making latent-based watermarking incompatible. This creates the challenge of watermarking directly in real space while handling feature heterogeneity and temporal dependencies. We propose TimeWak, the first watermarking algorithm for multivariate time series diffusion models. To handle temporal dependence and spatial heterogeneity, TimeWak embeds a temporal chained-hashing watermark directly within the real temporal-feature space. The other unique feature is the $\epsilon$-exact inversion, which addresses the non-uniform reconstruction error distribution across features from inverting the diffusion process to detect watermarks. We derive the error bound of inverting multivariate time series and further maintain high watermark detectability. We extensively evaluate TimeWak on its impact on synthetic data quality, watermark detectability, and robustness under various post-editing attacks, against 5 datasets and baselines of different temporal lengths. Our results show that TimeWak achieves improvements of 61.96% in context-FID score, and 8.44% in correlational scores against the state-of-the-art baseline, while remaining consistently detectable. 

**Abstract (ZH)**: 基于扩散模型的合成时间序列 enables 分享敏感隐私数据集，如患者的功能MRI记录。TimeWak: 一种用于多元时间序列扩散模型的水印算法 

---
# Theoretical Analysis of Positional Encodings in Transformer Models: Impact on Expressiveness and Generalization 

**Title (ZH)**: Transformer模型中位置编码的理论分析：对其表达能力和泛化能力的影响 

**Authors**: Yin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06398)  

**Abstract**: Positional encodings are a core part of transformer-based models, enabling processing of sequential data without recurrence. This paper presents a theoretical framework to analyze how various positional encoding methods, including sinusoidal, learned, relative, and bias-based methods like Attention with Linear Biases (ALiBi), impact a transformer's expressiveness, generalization ability, and extrapolation to longer sequences. Expressiveness is defined via function approximation, generalization bounds are established using Rademacher complexity, and new encoding methods based on orthogonal functions, such as wavelets and Legendre polynomials, are proposed. The extrapolation capacity of existing and proposed encodings is analyzed, extending ALiBi's biasing approach to a unified theoretical context. Experimental evaluation on synthetic sequence-to-sequence tasks shows that orthogonal transform-based encodings outperform traditional sinusoidal encodings in generalization and extrapolation. This work addresses a critical gap in transformer theory, providing insights for design choices in natural language processing, computer vision, and other transformer applications. 

**Abstract (ZH)**: 基于位置的编码是变压器模型的核心组成部分，使模型能够处理序列数据而不依赖循环结构。本文提出了一个理论框架来分析各种位置编码方法，包括正弦、学习、相对和基于偏置的方法（如线性偏置注意力（ALiBi）），对变压器的表征能力、泛化能力和长序列外推能力的影响。表征能力通过函数逼近定义，泛化界限使用拉德马赫复杂性建立，并提出了基于正交函数的新编码方法，如小波和勒让德多项式。现有和提出编码方法的外推能力得到了分析，将ALiBi的偏置方法统一到一个理论框架中。基于合成序列到序列任务的实验评估表明，基于正交变换的位置编码在泛化和外推方面优于传统的正弦编码。本文填补了变压器理论中的关键空白，为自然语言处理、计算机视觉和其他变压器应用的设计选择提供了见解。 

---
# Model-based Neural Data Augmentation for sub-wavelength Radio Localization 

**Title (ZH)**: 基于模型的神经数据扩充方法在亚波长无线电定位中的应用 

**Authors**: Baptiste Chatelier, Vincent Corlay, Musa Furkan Keskin, Matthieu Crussière, Henk Wymeersch, Luc Le Magoarou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06387)  

**Abstract**: The increasing deployment of large antenna arrays at base stations has significantly improved the spatial resolution and localization accuracy of radio-localization methods. However, traditional signal processing techniques struggle in complex radio environments, particularly in scenarios dominated by non line of sight (NLoS) propagation paths, resulting in degraded localization accuracy. Recent developments in machine learning have facilitated the development of machine learning-assisted localization techniques, enhancing localization accuracy in complex radio environments. However, these methods often involve substantial computational complexity during both the training and inference phases. This work extends the well-established fingerprinting-based localization framework by simultaneously reducing its memory requirements and improving its accuracy. Specifically, a model-based neural network is used to learn the location-to-channel mapping, and then serves as a generative neural channel model. This generative model augments the fingerprinting comparison dictionary while reducing the memory requirements. The proposed method outperforms fingerprinting baselines by achieving sub-wavelength localization accuracy, even in NLoS environments. Remarkably, it offers an improvement by several orders of magnitude in localization accuracy, while simultaneously reducing memory requirements by an order of magnitude compared to classical fingerprinting methods. 

**Abstract (ZH)**: 基于模型的神经网络辅助指纹本地化方法在非视距环境下的亚波长精度定位 

---
# Human and AI collaboration in Fitness Education:A Longitudinal Study with a Pilates Instructor 

**Title (ZH)**: 人类与AI在健身教育中的协作：一项与普拉提教练的合作纵向研究 

**Authors**: Qian Huang, King Wang Poon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06383)  

**Abstract**: Artificial intelligence is poised to transform teaching and coaching practices,yet its optimal role alongside human expertise remains this http URL study investigates human and AI collaboration in fitness education through a one year qualitative case study with a Pilates this http URL researcher participated in the instructor classes and conducted biweekly semi structured interviews to explore how generative AI could be integrated into class planning and instruction. 

**Abstract (ZH)**: 人工智能 impending transformation of teaching and coaching practices: exploring the optimal role of AI alongside human expertise through a qualitative case study of Pilates class planning and instruction with generative AI integration. 

---
# Beyond the Norm: A Survey of Synthetic Data Generation for Rare Events 

**Title (ZH)**: 超越常规：罕见事件合成数据生成综述 

**Authors**: Jingyi Gu, Xuan Zhang, Guiling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06380)  

**Abstract**: Extreme events, such as market crashes, natural disasters, and pandemics, are rare but catastrophic, often triggering cascading failures across interconnected systems. Accurate prediction and early warning can help minimize losses and improve preparedness. While data-driven methods offer powerful capabilities for extreme event modeling, they require abundant training data, yet extreme event data is inherently scarce, creating a fundamental challenge. Synthetic data generation has emerged as a powerful solution. However, existing surveys focus on general data with privacy preservation emphasis, rather than extreme events' unique performance requirements. This survey provides the first overview of synthetic data generation for extreme events. We systematically review generative modeling techniques and large language models, particularly those enhanced by statistical theory as well as specialized training and sampling mechanisms to capture heavy-tailed distributions. We summarize benchmark datasets and introduce a tailored evaluation framework covering statistical, dependence, visual, and task-oriented metrics. A central contribution is our in-depth analysis of each metric's applicability in extremeness and domain-specific adaptations, providing actionable guidance for model evaluation in extreme settings. We categorize key application domains and identify underexplored areas like behavioral finance, wildfires, earthquakes, windstorms, and infectious outbreaks. Finally, we outline open challenges, providing a structured foundation for advancing synthetic rare-event research. 

**Abstract (ZH)**: 极值事件的合成数据生成：面向重尾分布的独特性能要求与应用领域研究 

---
# CR-BLEA: Contrastive Ranking for Adaptive Resource Allocation in Bilevel Evolutionary Algorithms 

**Title (ZH)**: CR-BLEA: 对比排序在双层进化算法中自适应资源分配中的应用 

**Authors**: Dejun Xu, Jijia Chen, Gary G. Yen, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06362)  

**Abstract**: Bilevel optimization poses a significant computational challenge due to its nested structure, where each upper-level candidate solution requires solving a corresponding lower-level problem. While evolutionary algorithms (EAs) are effective at navigating such complex landscapes, their high resource demands remain a key bottleneck -- particularly the redundant evaluation of numerous unpromising lower-level tasks. Despite recent advances in multitasking and transfer learning, resource waste persists. To address this issue, we propose a novel resource allocation framework for bilevel EAs that selectively identifies and focuses on promising lower-level tasks. Central to our approach is a contrastive ranking network that learns relational patterns between paired upper- and lower-level solutions online. This knowledge guides a reference-based ranking strategy that prioritizes tasks for optimization and adaptively controls resampling based on estimated population quality. Comprehensive experiments across five state-of-the-art bilevel algorithms show that our framework significantly reduces computational cost while preserving -- or even enhancing -- solution accuracy. This work offers a generalizable strategy to improve the efficiency of bilevel EAs, paving the way for more scalable bilevel optimization. 

**Abstract (ZH)**: bilevel 优化因其嵌套结构导致了显著的计算挑战，在其结构中，每个高层候选解都需要解决相应的低层问题。虽然进化算法（EAs）在导航此类复杂景观方面非常有效，但它们对资源的需求仍然是一个关键瓶颈，尤其是对大量无前景低层任务的冗余评估。尽管最近在多任务和迁移学习方面取得了进展，但资源浪费仍然存在。为此，我们提出了一个用于 bilevel EAs 的新颖资源分配框架，该框架选择性地识别并专注于有前景的低层任务。我们方法的核心是一个在线学习配对高层和低层解决方案之间关系模式的对比排名网络。这些知识引导基于参考的排名策略来优先优化任务，并根据估计的种群质量自适应控制采样。在五个先进的 bilevel 算法的全面实验中，我们的框架显著降低了计算成本，同时保持甚至提高了解决方案准确性。这项工作提供了一种通用策略以提高 bilevel EAs 的效率，为更可扩展的 bilevel 优化奠定了基础。 

---
# Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation 

**Title (ZH)**: 基于深度学习的传输损耗估计用于实时评估 infrasound 事件检测能力 

**Authors**: Alice Janela Cameijo, Alexis Le Pichon, Youcef Sklab, Souhila Arib, Quentin Brissaud, Sven peter Naesholm, Constantino Listowski, Samir Aknine  

**Link**: [PDF](https://arxiv.org/pdf/2506.06358)  

**Abstract**: Accurate modeling of infrasound transmission loss is essential for evaluating the performance of the International Monitoring System, enabling the effective design and maintenance of infrasound stations to support compliance of the Comprehensive Nuclear-Test-Ban Treaty. State-of-the-art propagation modeling tools enable transmission loss to be finely simulated using atmospheric models. However, the computational cost prohibits the exploration of a large parameter space in operational monitoring applications. To address this, recent studies made use of a deep learning algorithm capable of making transmission loss predictions almost instantaneously. However, the use of nudged atmospheric models leads to an incomplete representation of the medium, and the absence of temperature as an input makes the algorithm incompatible with long range propagation. In this study, we address these limitations by using both wind and temperature fields as inputs to a neural network, simulated up to 130 km altitude and 4,000 km distance. We also optimize several aspects of the neural network architecture. We exploit convolutional and recurrent layers to capture spatially and range-dependent features embedded in realistic atmospheric models, improving the overall performance. The neural network reaches an average error of 4 dB compared to full parabolic equation simulations and provides epistemic and data-related uncertainty estimates. Its evaluation on the 2022 Hunga Tonga-Hunga Ha'apai volcanic eruption demonstrates its prediction capability using atmospheric conditions and frequencies not included in the training. This represents a significant step towards near real-time assessment of International Monitoring System detection thresholds of explosive sources. 

**Abstract (ZH)**: 准确的 infrasound 传输损耗建模对于评估国际监测系统性能至关重要，有助于支持《全面禁核试验条约》的合规性设计和维护工作。最先进的传播建模工具可以通过大气模型精细模拟传输损耗。然而，计算成本限制了在实际监测应用中探索大量参数空间。为了解决这一问题，最近的研究采用了能够几乎瞬时进行传输损耗预测的深度学习算法。然而，使用校正的大气模型会使得对介质的表示不完整，缺少温度输入使得算法不适合远程传播。在本研究中，我们通过将风场和温度场作为神经网络的输入，模拟至130公里高度和4000公里距离，解决了这些限制。我们还优化了神经网络架构的多个方面。利用卷积层和循环层捕获现实大气模型中嵌入的空间和距离依赖特征，提升了整体性能。神经网络与完整的抛物方程模拟相比，平均误差为4 dB，并提供了一致性和数据相关的不确定性估计。其对2022年洪加汤加-洪加哈帕伊火山爆发的评估证明了其在训练数据未包含的大气条件和频率下的预测能力。这代表了朝着近实时评估国际监测系统爆炸源检测阈值的重要一步。 

---
# Deep learning methods for modeling infrasound transmission loss in the middle atmosphere 

**Title (ZH)**: 深学习方法用于中高层大气 infrasound 传输衰减建模 

**Authors**: Alexis Le Pichon, Alice Janela Cameijo, Samir Aknine, Youcef Sklab, Souhila Arib, Quentin Brissaud, Sven Peter Naesholm  

**Link**: [PDF](https://arxiv.org/pdf/2506.06351)  

**Abstract**: Accurate modeling of infrasound transmission losses (TLs) is essential to assess the performance of the global International Monitoring System infrasound network. Among existing propagation modeling tools, parabolic equation (PE) method enables TLs to be finely modeled, but its computational cost does not allow exploration of a large parameter space for operational monitoring applications. To reduce computation times, Brissaud et al. 2023 explored the potential of convolutional neural networks trained on a large set of regionally simulated wavefields (< 1000 km from the source) to predict TLs with negligible computation times compared to PE simulations. However, this method struggles in unfavorable initial wind conditions, especially at high frequencies, and causal issues with winds at large distances from the source affecting ground TLs close to the source. In this study, we have developed an optimized convolutional network designed to minimize prediction errors while predicting TLs from globally simulated combined temperature and wind fields spanning over propagation ranges of 4000 km. Our approach enhances the previously proposed one by implementing key optimizations that improve the overall architecture performance. The implemented model predicts TLs with an average error of 8.6 dB in the whole frequency band (0.1-3.2 Hz) and explored realistic atmospheric scenarios. 

**Abstract (ZH)**: 精确 modeling  infrasound 传输损耗 (TLs) 对评估国际监测系统全球 infrasound 网络性能至关重要。现有的传播建模工具中，抛物线方程 (PE) 方法能够精细建模 TLs，但由于计算成本较高，无法在操作监测应用中探索大量参数空间。为减少计算时间，Brissaud 等人（2023）探索了通过在区域模拟波场（<1000 km 从声源）上训练的卷积神经网络预测 TLs 的潜力，从而与 PE 模拟相比具有可忽略的计算时间。然而，这种方法在不利的初始风条件，尤其是高频率下，以及远处风的影响导致靠近声源地面 TLs 时存在因果问题。在本研究中，我们开发了一种优化的卷积网络，旨在预测全球模拟的温度和风场结合产生的 TLs 时最小化预测误差，传播范围为 4000 km。本方法通过实施关键优化措施，提高了整体架构性能。已实现的模型在整个频率范围（0.1-3.2 Hz）内的平均误差为 8.6 dB，并探索了现实的气象场景。 

---
# Explainable-AI powered stock price prediction using time series transformers: A Case Study on BIST100 

**Title (ZH)**: 基于时间序列变换器的可解释AI股价预测：对BIST100的实际案例研究 

**Authors**: Sukru Selim Calik, Andac Akyuz, Zeynep Hilal Kilimci, Kerem Colak  

**Link**: [PDF](https://arxiv.org/pdf/2506.06345)  

**Abstract**: Financial literacy is increasingly dependent on the ability to interpret complex financial data and utilize advanced forecasting tools. In this context, this study proposes a novel approach that combines transformer-based time series models with explainable artificial intelligence (XAI) to enhance the interpretability and accuracy of stock price predictions. The analysis focuses on the daily stock prices of the five highest-volume banks listed in the BIST100 index, along with XBANK and XU100 indices, covering the period from January 2015 to March 2025. Models including DLinear, LTSNet, Vanilla Transformer, and Time Series Transformer are employed, with input features enriched by technical indicators. SHAP and LIME techniques are used to provide transparency into the influence of individual features on model outputs. The results demonstrate the strong predictive capabilities of transformer models and highlight the potential of interpretable machine learning to empower individuals in making informed investment decisions and actively engaging in financial markets. 

**Abstract (ZH)**: 基于变压器的时间序列模型与可解释人工智能的结合：提升股票价格预测的可解释性和准确性 

---
# NR4DER: Neural Re-ranking for Diversified Exercise Recommendation 

**Title (ZH)**: NR4DER：神经网络重排ranking以实现多样化的运动推荐 

**Authors**: Xinghe Cheng, Xufang Zhou, Liangda Fang, Chaobo He, Yuyu Zhou, Weiqi Luo, Zhiguo Gong, Quanlong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06341)  

**Abstract**: With the widespread adoption of online education platforms, an increasing number of students are gaining new knowledge through Massive Open Online Courses (MOOCs). Exercise recommendation have made strides toward improving student learning outcomes. However, existing methods not only struggle with high dropout rates but also fail to match the diverse learning pace of students. They frequently face difficulties in adjusting to inactive students' learning patterns and in accommodating individualized learning paces, resulting in limited accuracy and diversity in recommendations. To tackle these challenges, we propose Neural Re-ranking for Diversified Exercise Recommendation (in short, NR4DER). NR4DER first leverages the mLSTM model to improve the effectiveness of the exercise filter module. It then employs a sequence enhancement method to enhance the representation of inactive students, accurately matches students with exercises of appropriate difficulty. Finally, it utilizes neural re-ranking to generate diverse recommendation lists based on individual students' learning histories. Extensive experimental results indicate that NR4DER significantly outperforms existing methods across multiple real-world datasets and effectively caters to the diverse learning pace of students. 

**Abstract (ZH)**: 基于神经重排的多样化习题推荐（NR4DER） 

---
# Introduction to Predictive Coding Networks for Machine Learning 

**Title (ZH)**: 预测编码网络在机器学习中的介绍 

**Authors**: Mikko Stenlund  

**Link**: [PDF](https://arxiv.org/pdf/2506.06332)  

**Abstract**: Predictive coding networks (PCNs) constitute a biologically inspired framework for understanding hierarchical computation in the brain, and offer an alternative to traditional feedforward neural networks in ML. This note serves as a quick, onboarding introduction to PCNs for machine learning practitioners. We cover the foundational network architecture, inference and learning update rules, and algorithmic implementation. A concrete image-classification task (CIFAR-10) is provided as a benchmark-smashing application, together with an accompanying Python notebook containing the PyTorch implementation. 

**Abstract (ZH)**: 基于预测编码网络（PCNs）构成的生物学启发式框架用于理解大脑中的分层计算，并为机器学习提供了传统前馈神经网络之外的替代方案。本文为机器学习从业者提供了一个快速入门介绍。我们涵盖了基础网络架构、推理和学习更新规则以及算法实现。提供了一个具体的图像分类任务（CIFAR-10）作为 benchmark 的应用示例，并附带了一个包含 PyTorch 实现的 Python 笔记本。 

---
# Evolutionary model for energy trading in community microgrids using Hawk-Dove strategies 

**Title (ZH)**: 基于hawk-dove策略的社区微电网能量交易演化模型 

**Authors**: Viorica Rozina Chifu, Tudor Cioara, Cristina Bianca Pop, Ionut Anghel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06325)  

**Abstract**: This paper proposes a decentralized model of energy cooperation between microgrids, in which decisions are made locally, at the level of the microgrid community. Each microgrid is modeled as an autonomous agent that adopts a Hawk or Dove strategy, depending on the level of energy stored in the battery and its role in the energy trading process. The interactions between selling and buying microgrids are modeled through an evolutionary algorithm. An individual in the algorithm population is represented as an energy trading matrix that encodes the amounts of energy traded between the selling and buying microgrids. The population evolution is achieved by recombination and mutation operators. Recombination uses a specialized operator for matrix structures, and mutation is applied to the matrix elements according to a Gaussian distribution. The evaluation of an individual is made with a multi-criteria fitness function that considers the seller profit, the degree of energy stability at the community level, penalties for energy imbalance at the community level and for the degradation of microgrids batteries. The method was tested on a simulated scenario with 100 microgrids, each with its own selling and buying thresholds, to reflect a realistic environment with variable storage characteristics of microgrids batteries. By applying the algorithm on this scenario, 95 out of the 100 microgrids reached a stable energy state. This result confirms the effectiveness of the proposed model in achieving energy balance both at the individual level, for each microgrid, and at the level of the entire community. 

**Abstract (ZH)**: 本文提出了一种微电网之间的去中心化能源合作模型，在该模型中，决策是在微电网社区的局部水平上做出的。每个微电网被建模为一个自主代理，采用鹰或鸽策略，这取决于电池中储存的能源水平及其在能源交易过程中的角色。通过进化算法模型化卖电和购电微电网之间的交互。算法群体中的个体表示为一个能源交易矩阵，编码了卖电和购电微电网之间的能源交易量。群体的演化通过重组和变异算子实现。重组使用了专门针对矩阵结构的算子，变异则根据高斯分布应用于矩阵元素。个体的评估使用了多准则适应度函数，该函数考虑了销售利润、社区层面的能量稳定性、社区层面能源不平衡的惩罚以及微电网电池退化等因素。该方法在包含100个具有独立卖电和购电阈值的微电网的模拟场景中进行了测试，以反映具有可变存储特性的实际环境。通过对这一场景的应用，95个微电网达到了稳定能源状态。这一结果证实了所提出模型在实现个体微电网及整个社区层面的能源平衡方面的有效性。 

---
# MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes 

**Title (ZH)**: MoE-Gyro: 自监督超出量程重建与 MEMS 陀螺仪降噪 

**Authors**: Feiyang Pan, Shenghe Zheng, Chunyan Yin, Guangbin Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06318)  

**Abstract**: MEMS gyroscopes play a critical role in inertial navigation and motion control applications but typically suffer from a fundamental trade-off between measurement range and noise performance. Existing hardware-based solutions aimed at mitigating this issue introduce additional complexity, cost, and scalability challenges. Deep-learning methods primarily focus on noise reduction and typically require precisely aligned ground-truth signals, making them difficult to deploy in practical scenarios and leaving the fundamental trade-off unresolved. To address these challenges, we introduce Mixture of Experts for MEMS Gyroscopes (MoE-Gyro), a novel self-supervised framework specifically designed for simultaneous over-range signal reconstruction and noise suppression. MoE-Gyro employs two experts: an Over-Range Reconstruction Expert (ORE), featuring a Gaussian-Decay Attention mechanism for reconstructing saturated segments; and a Denoise Expert (DE), utilizing dual-branch complementary masking combined with FFT-guided augmentation for robust noise reduction. A lightweight gating module dynamically routes input segments to the appropriate expert. Furthermore, existing evaluation lack a comprehensive standard for assessing multi-dimensional signal enhancement. To bridge this gap, we introduce IMU Signal Enhancement Benchmark (ISEBench), an open-source benchmarking platform comprising the GyroPeak-100 dataset and a unified evaluation of IMU signal enhancement methods. We evaluate MoE-Gyro using our proposed ISEBench, demonstrating that our framework significantly extends the measurable range from 450 deg/s to 1500 deg/s, reduces Bias Instability by 98.4%, and achieves state-of-the-art performance, effectively addressing the long-standing trade-off in inertial sensing. 

**Abstract (ZH)**: MEMS陀螺仪在惯性导航和运动控制应用中扮演着关键角色，但通常会遇到量程和噪声性能之间的基本权衡。现有的硬件解决方案旨在缓解这一问题，但会引入额外的复杂性、成本和扩展性挑战。深度学习方法主要关注噪声减少，并且通常需要精确对齐的真实信号，使其难以在实际场景中部署，从而未能解决根本性的权衡问题。为应对这些挑战，我们提出了MEMS陀螺仪混合专家模型（MoE-Gyro），这是一种新颖的自监督框架，专门设计用于同时实现过量程信号重建和噪声抑制。MoE-Gyro采用两个专家：过量程重建专家（ORE），采用高斯衰减注意力机制来重建饱和段；降噪专家（DE），利用双分支互补掩蔽结合FFT引导增强进行稳健的噪声减少。一个轻量级门控模块动态将输入段路由到合适的专家。此外，现有的评估缺乏一个多维信号增强的全面标准。为弥补这一不足，我们引入了IMU信号增强基准（ISEBench），这是一个开源基准平台，包含GyroPeak-100数据集和IMU信号增强方法的统一评估。我们使用我们提出的ISEBench评估MoE-Gyro，结果显示我们的框架将可测量范围从450 deg/s扩展到1500 deg/s，减小了98.4%的偏差不稳定性，并实现了最先进的性能，有效解决了惯性传感领域的长期权衡问题。 

---
# How Malicious AI Swarms Can Threaten Democracy 

**Title (ZH)**: 恶意AI集群如何威胁民主 

**Authors**: Daniel Thilo Schroeder, Meeyoung Cha, Andrea Baronchelli, Nick Bostrom, Nicholas A. Christakis, David Garcia, Amit Goldenberg, Yara Kyrychenko, Kevin Leyton-Brown, Nina Lutz, Gary Marcus, Filippo Menczer, Gordon Pennycook, David G. Rand, Frank Schweitzer, Christopher Summerfield, Audrey Tang, Jay Van Bavel, Sander van der Linden, Dawn Song, Jonas R. Kunst  

**Link**: [PDF](https://arxiv.org/pdf/2506.06299)  

**Abstract**: Advances in AI portend a new era of sophisticated disinformation operations. While individual AI systems already create convincing -- and at times misleading -- information, an imminent development is the emergence of malicious AI swarms. These systems can coordinate covertly, infiltrate communities, evade traditional detectors, and run continuous A/B tests, with round-the-clock persistence. The result can include fabricated grassroots consensus, fragmented shared reality, mass harassment, voter micro-suppression or mobilization, contamination of AI training data, and erosion of institutional trust. With democratic processes worldwide increasingly vulnerable, we urge a three-pronged response: (1) platform-side defenses -- always-on swarm-detection dashboards, pre-election high-fidelity swarm-simulation stress-tests, transparency audits, and optional client-side "AI shields" for users; (2) model-side safeguards -- standardized persuasion-risk tests, provenance-authenticating passkeys, and watermarking; and (3) system-level oversight -- a UN-backed AI Influence Observatory. 

**Abstract (ZH)**: AI的进步预示着一个新的复杂虚假信息操作时代。随着个体AI系统已经生成令人信服的——有时是误导性的——信息，即将到来的发展将是恶意AI集群的涌现。这些系统可以隐蔽协调、渗透社区、逃避传统检测器，并且进行持续的A/B测试，具备全天候的持续性。其结果可能包括伪造的草根共识、碎片化的共享现实、大规模骚扰、选民微抑制或动员、AI训练数据污染以及机构信任的侵蚀。随着全球民主进程日益脆弱，我们敦促采取三管齐下的应对措施：（1）平台侧防御——持续检测集群的仪表盘、高保真集群模拟预选前的压力测试、透明度审计以及可选的客户端“AI防护”；（2）模型侧保护——标准化说服风险测试、源认证通行证以及数字水印；（3）系统级监督——由联合国支持的AI影响力观察站。 

---
# Pairwise Calibrated Rewards for Pluralistic Alignment 

**Title (ZH)**: 多元共融的配对校准奖励 

**Authors**: Daniel Halpern, Evi Micha, Ariel D. Procaccia, Itai Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2506.06298)  

**Abstract**: Current alignment pipelines presume a single, universal notion of desirable behavior. However, human preferences often diverge across users, contexts, and cultures. As a result, disagreement collapses into the majority signal and minority perspectives are discounted. To address this, we propose reflecting diverse human preferences through a distribution over multiple reward functions, each inducing a distinct aligned policy. The distribution is learned directly from pairwise preference without annotator identifiers or predefined groups. Instead, annotator disagreements are treated as informative soft labels. Our central criterion is pairwise calibration: for every pair of candidate responses, the proportion of reward functions preferring one response matches the fraction of annotators with that preference. We prove that even a small outlier-free ensemble can accurately represent diverse preference distributions. Empirically, we introduce and validate a practical training heuristic to learn such ensembles, and demonstrate its effectiveness through improved calibration, implying a more faithful representation of pluralistic values. 

**Abstract (ZH)**: 当前对齐管道假设了一种单一同质性的理想行为观念。然而，人类偏好在用户、情境和文化之间常常存在分歧。因此，分歧意见被归结为多数信号，而少数视角被忽视。为解决这一问题，我们提出通过多个奖励函数的概率分布来反映多样的人类偏好，每个奖励函数诱导一种独特的对齐策略。该分布直接从成对偏好中学习，而不依赖标注者的识别信息或预定义的组别。相反，标注者之间的分歧被视为有信息性的软标签。我们核心的标准是成对校准：对于每一对候选响应，偏好某一响应的奖励函数的比例匹配有同样偏好标注者的比例。我们证明，即使是一个无离群值的小组合也能准确代表多样化的偏好分布。实验上，我们引入并验证了一种实用的训练启发式方法来学习这样的组合，并通过提高校准来证明其效果，意味着更加忠实地代表了多元的价值观。 

---
# Optimal patient allocation for echocardiographic assessments 

**Title (ZH)**: 最优患者分配以进行心脏超声评估 

**Authors**: Bozhi Sun, Seda Tierney, Jeffrey A. Feinstein, Frederick Damen, Alison L. Marsden, Daniele E. Schiavazzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06297)  

**Abstract**: Scheduling echocardiographic exams in a hospital presents significant challenges due to non-deterministic factors (e.g., patient no-shows, patient arrival times, diverse exam durations, etc.) and asymmetric resource constraints between fetal and non-fetal patient streams. To address these challenges, we first conducted extensive pre-processing on one week of operational data from the Echo Laboratory at Stanford University's Lucile Packard Children's Hospital, to estimate patient no-show probabilities and derive empirical distributions of arrival times and exam durations. Based on these inputs, we developed a discrete-event stochastic simulation model using SimPy, and integrate it with the open source Gymnasium Python library. As a baseline for policy optimization, we developed a comparative framework to evaluate on-the-fly versus reservation-based allocation strategies, in which different proportions of resources are reserved in advance. Considering a hospital configuration with a 1:6 ratio of fetal to non-fetal rooms and a 4:2 ratio of fetal to non-fetal sonographers, we show that on-the-fly allocation generally yields better performance, more effectively adapting to patient variability and resource constraints. Building on this foundation, we apply reinforcement learning (RL) to derive an approximated optimal dynamic allocation policy. This RL-based policy is benchmarked against the best-performing rule-based strategies, allowing us to quantify their differences and provide actionable insights for improving echo lab efficiency through intelligent, data-driven resource management. 

**Abstract (ZH)**: 在圣地亚哥儿童医院卢西尔·帕克回声实验室中预约心脏超声检查面临着显著挑战，由于非确定性因素（如患者缺席、患者到达时间、多样化的检查时间等）和胎儿和非胎儿患者流之间的非对称资源约束。为应对这些挑战，我们首先对来自斯坦福大学卢西尔·帕克儿童医院回声实验室一周的运营数据进行了广泛的预处理，以估算患者缺席概率并推导出到达时间和检查时间的经验分布。基于这些输入，我们使用SimPy开发了一个离散事件随机仿真模型，并将其与开源的Gymnasium Python库集成。作为政策优化的基准，我们开发了一个比较框架，评估即时分配策略与预约分配策略，其中不同比例的资源提前预留。考虑到1:6的胎儿与非胎儿房间比例和4:2的胎儿与非胎儿超声技师比例，我们证明了即时分配策略通常表现出更好的性能，更有效地适应患者变异性及资源约束。在此基础上，我们应用强化学习（RL）来推导近似最优的动态分配策略。该基于RL的策略被基准测试与表现最佳的基于规则的策略，以量化它们之间的差异，并提供通过智能、数据驱动的资源管理提高回声实验室效率的可操作性见解。 

---
# GLProtein: Global-and-Local Structure Aware Protein Representation Learning 

**Title (ZH)**: GLProtein: 全局与局部结构意识的蛋白质表示学习 

**Authors**: Yunqing Liu, Wenqi Fan, Xiaoyong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06294)  

**Abstract**: Proteins are central to biological systems, participating as building blocks across all forms of life. Despite advancements in understanding protein functions through protein sequence analysis, there remains potential for further exploration in integrating protein structural information. We argue that the structural information of proteins is not only limited to their 3D information but also encompasses information from amino acid molecules (local information) to protein-protein structure similarity (global information). To address this, we propose \textbf{GLProtein}, the first framework in protein pre-training that incorporates both global structural similarity and local amino acid details to enhance prediction accuracy and functional insights. GLProtein innovatively combines protein-masked modelling with triplet structure similarity scoring, protein 3D distance encoding and substructure-based amino acid molecule encoding. Experimental results demonstrate that GLProtein outperforms previous methods in several bioinformatics tasks, including predicting protein-protein interaction, contact prediction, and so on. 

**Abstract (ZH)**: 蛋白质是生物系统的核心，参与所有生命形式中构建块的作用。尽管通过蛋白质序列分析已经取得了对蛋白质功能的深刻理解，但在整合蛋白质结构信息方面仍有潜在的空间。我们认为，蛋白质的结构信息不仅限于其三维信息，还涵盖了从氨基酸分子的局部信息到蛋白质-蛋白质结构相似性的全局信息。为解决这一问题，我们提出了GLProtein框架，这是首个结合全局结构相似性和局部氨基酸细节的蛋白质预训练框架，以提高预测准确性和功能见解。GLProtein创新性地结合了蛋白质遮蔽建模、三重结构相似性评分、蛋白质三维距离编码和基于子结构的氨基酸分子编码。实验结果表明，GLProtein在预测蛋白质-蛋白质相互作用、接触预测等多个生物信息学任务中优于先前的方法。 

---
# Prediction of Bank Credit Ratings using Heterogeneous Topological Graph Neural Networks 

**Title (ZH)**: 使用异构拓扑图神经网络预测银行信用评级 

**Authors**: Junyi Liu, Stanley Kok  

**Link**: [PDF](https://arxiv.org/pdf/2506.06293)  

**Abstract**: Agencies such as Standard & Poor's and Moody's provide bank credit ratings that influence economic stability and decision-making by stakeholders. Accurate and timely predictions support informed decision-making, regulatory actions, and investor protection. However, a complete interbank connection graph is often unavailable due to privacy concerns, complicating the direct application of Graph Neural Networks (GNNs) for rating prediction. our research utilizes persistent homology to construct a network that captures relationships among banks and combines this with a traditional lending network to create a heterogeneous network that integrates information from both sources, leading to improved predictions. Experiments on a global, real-world dataset validate the effectiveness of HTGNN. This research has implications for investors and regulatory bodies in enhancing proactive risk mitigation and the implementation of effective market this http URL code can be find at this https URL. 

**Abstract (ZH)**: 标准普尔和穆迪等机构提供的银行信用评级影响着经济稳定和利益相关方的决策。准确及时的预测支持知情决策、监管行动和投资者保护。但由于隐私问题，完整的银行间连接图通常不可用，这使得直接应用图神经网络（GNNs）进行评级预测变得复杂。我们的研究利用持久同调构建一个网络，捕捉银行之间的关系，并将此与传统贷款网络结合，创建一个异构网络，综合了两种来源的信息，从而提高了预测效果。全球实际数据集上的实验验证了HTGNN的有效性。这项研究对投资者和监管机构在增强前瞻性风险缓解和有效市场实施方面具有重要意义。相关代码可以在以下链接找到：[这里](这里)[这里](这里)。 

---
# Improvement of Optimization using Learning Based Models in Mixed Integer Linear Programming Tasks 

**Title (ZH)**: 基于学习模型在混合整数线性规划任务中优化的改进 

**Authors**: Xiaoke Wang, Batuhan Altundas, Zhaoxin Li, Aaron Zhao, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06291)  

**Abstract**: Mixed Integer Linear Programs (MILPs) are essential tools for solving planning and scheduling problems across critical industries such as construction, manufacturing, and logistics. However, their widespread adoption is limited by long computational times, especially in large-scale, real-time scenarios. To address this, we present a learning-based framework that leverages Behavior Cloning (BC) and Reinforcement Learning (RL) to train Graph Neural Networks (GNNs), producing high-quality initial solutions for warm-starting MILP solvers in Multi-Agent Task Allocation and Scheduling Problems. Experimental results demonstrate that our method reduces optimization time and variance compared to traditional techniques while maintaining solution quality and feasibility. 

**Abstract (ZH)**: 基于行为克隆和强化学习的图神经网络在多代理任务分配与调度问题中混合整数线性规划初解学习框架 

---
# CellCLIP -- Learning Perturbation Effects in Cell Painting via Text-Guided Contrastive Learning 

**Title (ZH)**: CellCLIP —— 通过文本引导的对比学习学习细胞绘画中的干扰效果 

**Authors**: Mingyu Lu, Ethan Weinberger, Chanwoo Kim, Su-In Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.06290)  

**Abstract**: High-content screening (HCS) assays based on high-throughput microscopy techniques such as Cell Painting have enabled the interrogation of cells' morphological responses to perturbations at an unprecedented scale. The collection of such data promises to facilitate a better understanding of the relationships between different perturbations and their effects on cellular state. Towards achieving this goal, recent advances in cross-modal contrastive learning could, in theory, be leveraged to learn a unified latent space that aligns perturbations with their corresponding morphological effects. However, the application of such methods to HCS data is not straightforward due to substantial differences in the semantics of Cell Painting images compared to natural images, and the difficulty of representing different classes of perturbations (e.g., small molecule vs CRISPR gene knockout) in a single latent space. In response to these challenges, here we introduce CellCLIP, a cross-modal contrastive learning framework for HCS data. CellCLIP leverages pre-trained image encoders coupled with a novel channel encoding scheme to better capture relationships between different microscopy channels in image embeddings, along with natural language encoders for representing perturbations. Our framework outperforms current open-source models, demonstrating the best performance in both cross-modal retrieval and biologically meaningful downstream tasks while also achieving significant reductions in computation time. 

**Abstract (ZH)**: 基于高内涵成像技术如Cell Painting的高内涵筛查（HCS） assay能够在前所未有的规模上探究细胞对扰动的形态学响应。通过收集此类数据，有望促进对不同扰动与其对细胞状态影响之间关系的更好理解。为了实现这一目标，近年来在跨模态对比学习方面的进展理论上可以被利用来学习一个统一的潜在空间，将扰动与相应的形态学效应对齐。然而，将此类方法应用于HCS数据并不直接，因为Cell Painting图像与自然图像在语义上存在显著差异，且难以在单个潜在空间中表示不同类别的扰动（例如，小分子与CRISPR基因敲除）。面对这些挑战，我们引入了CellCLIP，一种适用于HCS数据的跨模态对比学习框架。CellCLIP利用预训练的图像编码器和一种新颖的通道编码方案，在图像嵌入中更好地捕捉不同显微镜通道之间的关系，并结合自然语言编码器来表示扰动。我们的框架在跨模态检索和生物学上有意义的下游任务中均表现出色，同时显著减少了计算时间。 

---
# DELPHYNE: A Pre-Trained Model for General and Financial Time Series 

**Title (ZH)**: DELPHYNE：一个通用和金融时间序列的预训练模型 

**Authors**: Xueying Ding, Aakriti Mittal, Achintya Gopal  

**Link**: [PDF](https://arxiv.org/pdf/2506.06288)  

**Abstract**: Time-series data is a vital modality within data science communities. This is particularly valuable in financial applications, where it helps in detecting patterns, understanding market behavior, and making informed decisions based on historical data. Recent advances in language modeling have led to the rise of time-series pre-trained models that are trained on vast collections of datasets and applied to diverse tasks across financial domains. However, across financial applications, existing time-series pre-trained models have not shown boosts in performance over simple finance benchmarks in both zero-shot and fine-tuning settings. This phenomenon occurs because of a i) lack of financial data within the pre-training stage, and ii) the negative transfer effect due to inherently different time-series patterns across domains. Furthermore, time-series data is continuous, noisy, and can be collected at varying frequencies and with varying lags across different variables, making this data more challenging to model than languages. To address the above problems, we introduce a Pre-trained MoDEL for FINance TimE-series (Delphyne). Delphyne achieves competitive performance to existing foundation and full-shot models with few fine-tuning steps on publicly available datasets, and also shows superior performances on various financial tasks. 

**Abstract (ZH)**: 预训练金融时间序列模型（Delphyne） 

---
# Disentangling AI Alignment: A Structured Taxonomy Beyond Safety and Ethics 

**Title (ZH)**: 解构AI对齐：超越安全与伦理的结构化分类体系 

**Authors**: Kevin Baum  

**Link**: [PDF](https://arxiv.org/pdf/2506.06286)  

**Abstract**: Recent advances in AI research make it increasingly plausible that artificial agents with consequential real-world impact will soon operate beyond tightly controlled environments. Ensuring that these agents are not only safe but that they adhere to broader normative expectations is thus an urgent interdisciplinary challenge. Multiple fields -- notably AI Safety, AI Alignment, and Machine Ethics -- claim to contribute to this task. However, the conceptual boundaries and interrelations among these domains remain vague, leaving researchers without clear guidance in positioning their work.
To address this meta-challenge, we develop a structured conceptual framework for understanding AI alignment. Rather than focusing solely on alignment goals, we introduce a taxonomy distinguishing the alignment aim (safety, ethicality, legality, etc.), scope (outcome vs. execution), and constituency (individual vs. collective). This structural approach reveals multiple legitimate alignment configurations, providing a foundation for practical and philosophical integration across domains, and clarifying what it might mean for an agent to be aligned all-things-considered. 

**Abstract (ZH)**: Recent advances in AI研究使具有重要现实世界影响的 artificial agents 随后在受控环境之外运行的可能性越来越大。因此，确保这些agents 不仅是安全的，还符合更广泛的规范性期望，是一个急迫的跨学科挑战。多个领域——尤其是AI安全性、AI对齐和机器伦理——声称有助于这一任务。然而，这些领域的概念边界及其相互关系仍然含糊不清，使研究人员在定位其工作时缺乏清晰的指导。为了应对这一元挑战，我们制定了一个结构化的概念框架，以理解AI对齐。我们不仅关注对齐目标，还引入了一种分类法，区分对齐目标（如安全性、伦理性、合法性等）、范围（结果导向 vs. 执行导向）以及主体（个体 vs. 集体）。这种结构化方法揭示了多种合法的对齐配置，为跨领域提供了实用和哲学上的整合基础，并明确了整体来看一个agent如何对齐的含义。 

---
# Facial Foundational Model Advances Early Warning of Coronary Artery Disease from Live Videos with DigitalShadow 

**Title (ZH)**: 面部基础模型在live视频中通过DigitalShadow早期预警冠状动脉疾病 

**Authors**: Juexiao Zhou, Zhongyi Han, Mankun Xin, Xingwei He, Guotao Wang, Jiaoyan Song, Gongning Luo, Wenjia He, Xintong Li, Yuetan Chu, Juanwen Chen, Bo Wang, Xia Wu, Wenwen Duan, Zhixia Guo, Liyan Bai, Yilin Pan, Xuefei Bi, Lu Liu, Long Feng, Xiaonan He, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06283)  

**Abstract**: Global population aging presents increasing challenges to healthcare systems, with coronary artery disease (CAD) responsible for approximately 17.8 million deaths annually, making it a leading cause of global mortality. As CAD is largely preventable, early detection and proactive management are essential. In this work, we introduce DigitalShadow, an advanced early warning system for CAD, powered by a fine-tuned facial foundation model. The system is pre-trained on 21 million facial images and subsequently fine-tuned into LiveCAD, a specialized CAD risk assessment model trained on 7,004 facial images from 1,751 subjects across four hospitals in China. DigitalShadow functions passively and contactlessly, extracting facial features from live video streams without requiring active user engagement. Integrated with a personalized database, it generates natural language risk reports and individualized health recommendations. With privacy as a core design principle, DigitalShadow supports local deployment to ensure secure handling of user data. 

**Abstract (ZH)**: 全球人口老龄化对医疗卫生系统提出了不断增加的挑战，冠状动脉疾病（CAD）导致每年约1780万人死亡，使其成为全球主要死亡原因之一。由于CAD主要可以通过预防来避免，因此早期检测和主动管理至关重要。本文介绍了一种名为DigitalShadow的高级预警系统，该系统借助微调后的面部基础模型。该系统在2100万张面部图像上进行预训练，并进一步微调为LiveCAD，这是一种专门针对来自中国四家医院1751名受试者7004张面部图像的心脏病风险评估模型。DigitalShadow被动且非接触地工作，无需用户主动参与即可从实时视频流中提取面部特征。结合个性化数据库，该系统生成自然语言风险报告和个人化健康建议。在以隐私为核心设计原则的基础上，DigitalShadow支持本地部署，以确保用户数据的安全处理。 

---
