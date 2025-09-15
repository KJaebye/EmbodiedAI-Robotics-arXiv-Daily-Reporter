# TASC: Task-Aware Shared Control for Teleoperated Manipulation 

**Title (ZH)**: TASC: 任务感知的共享控制方法用于远程操作 manipulation 

**Authors**: Ze Fu, Pinhao Song, Yutong Hu, Renaud Detry  

**Link**: [PDF](https://arxiv.org/pdf/2509.10416)  

**Abstract**: We present TASC, a Task-Aware Shared Control framework for teleoperated manipulation that infers task-level user intent and provides assistance throughout the task. To support everyday tasks without predefined knowledge, TASC constructs an open-vocabulary interaction graph from visual input to represent functional object relationships, and infers user intent accordingly. A shared control policy then provides rotation assistance during both grasping and object interaction, guided by spatial constraints predicted by a vision-language model. Our method addresses two key challenges in general-purpose, long-horizon shared control: (1) understanding and inferring task-level user intent, and (2) generalizing assistance across diverse objects and tasks. Experiments in both simulation and the real world demonstrate that TASC improves task efficiency and reduces user input effort compared to prior methods. To the best of our knowledge, this is the first shared control framework that supports everyday manipulation tasks with zero-shot generalization. The code that supports our experiments is publicly available at this https URL. 

**Abstract (ZH)**: 我们提出了TASC，一种任务感知的共享控制框架，用于远程操作操作，可以推断任务级用户意图并在整个任务过程中提供辅助。为了支持无需预先定义知识的日常任务，TASC 从视觉输入中构建一个开放词汇的交互图来表示功能性对象关系，并据此推断用户意图。然后，共享控制策略在抓取和对象交互过程中根据由视觉-语言模型预测的空间约束提供旋转辅助。我们的方法解决了通用、长期预测中共享控制的两个关键挑战：（1）理解和推断任务级用户意图，以及（2）在不同对象和任务之间泛化辅助。在仿真和真实世界中的实验表明，与先前的方法相比，TASC 提高了任务效率并减少了用户输入的努力。根据我们所知，这是首个支持零样本泛化的日常操作任务共享控制框架。我们的实验代码可在以下网址公开获取：this https URL。 

---
# Self-supervised Learning Of Visual Pose Estimation Without Pose Labels By Classifying LED States 

**Title (ZH)**: 无姿态标签下通过分类LED状态进行自我监督视觉姿态估计的学习 

**Authors**: Nicholas Carlotti, Mirko Nava, Alessandro Giusti  

**Link**: [PDF](https://arxiv.org/pdf/2509.10405)  

**Abstract**: We introduce a model for monocular RGB relative pose estimation of a ground robot that trains from scratch without pose labels nor prior knowledge about the robot's shape or appearance. At training time, we assume: (i) a robot fitted with multiple LEDs, whose states are independent and known at each frame; (ii) knowledge of the approximate viewing direction of each LED; and (iii) availability of a calibration image with a known target distance, to address the ambiguity of monocular depth estimation. Training data is collected by a pair of robots moving randomly without needing external infrastructure or human supervision. Our model trains on the task of predicting from an image the state of each LED on the robot. In doing so, it learns to predict the position of the robot in the image, its distance, and its relative bearing. At inference time, the state of the LEDs is unknown, can be arbitrary, and does not affect the pose estimation performance. Quantitative experiments indicate that our approach: is competitive with SoA approaches that require supervision from pose labels or a CAD model of the robot; generalizes to different domains; and handles multi-robot pose estimation. 

**Abstract (ZH)**: 一种无需姿态标签和机器人形状或外观先验知识的单目RGB单目_relative_pose_估计模型 

---
# Robot guide with multi-agent control and automatic scenario generation with LLM 

**Title (ZH)**: 具有多agent控制和LLM自动场景生成的机器人向导 

**Authors**: Elizaveta D. Moskovskaya, Anton D. Moscowsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.10317)  

**Abstract**: The work describes the development of a hybrid control architecture for an anthropomorphic tour guide robot, combining a multi-agent resource management system with automatic behavior scenario generation based on large language models. The proposed approach aims to overcome the limitations of traditional systems, which rely on manual tuning of behavior scenarios. These limitations include manual configuration, low flexibility, and lack of naturalness in robot behavior. The process of preparing tour scenarios is implemented through a two-stage generation: first, a stylized narrative is created, then non-verbal action tags are integrated into the text. The multi-agent system ensures coordination and conflict resolution during the execution of parallel actions, as well as maintaining default behavior after the completion of main operations, contributing to more natural robot behavior. The results obtained from the trial demonstrate the potential of the proposed approach for automating and scaling social robot control systems. 

**Abstract (ZH)**: 一种融合多Agent资源管理系统和基于大型语言模型的自动行为场景生成的人形导游机器人混合控制架构的研究 

---
# GundamQ: Multi-Scale Spatio-Temporal Representation Learning for Robust Robot Path Planning 

**Title (ZH)**: GundamQ：多尺度时空表示学习在鲁棒机器人路径规划中的应用 

**Authors**: Yutong Shen, Ruizhe Xia, Bokai Yan, Shunqi zhang, Pengrui Xiang, Sicheng He, Yixin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10305)  

**Abstract**: In dynamic and uncertain environments, robotic path planning demands accurate spatiotemporal environment understanding combined with robust decision-making under partial observability. However, current deep reinforcement learning-based path planning methods face two fundamental limitations: (1) insufficient modeling of multi-scale temporal dependencies, resulting in suboptimal adaptability in dynamic scenarios, and (2) inefficient exploration-exploitation balance, leading to degraded path quality. To address these challenges, we propose GundamQ: A Multi-Scale Spatiotemporal Q-Network for Robotic Path Planning. The framework comprises two key modules: (i) the Spatiotemporal Perception module, which hierarchically extracts multi-granularity spatial features and multi-scale temporal dependencies ranging from instantaneous to extended time horizons, thereby improving perception accuracy in dynamic environments; and (ii) the Adaptive Policy Optimization module, which balances exploration and exploitation during training while optimizing for smoothness and collision probability through constrained policy updates. Experiments in dynamic environments demonstrate that GundamQ achieves a 15.3\% improvement in success rate and a 21.7\% increase in overall path quality, significantly outperforming existing state-of-the-art methods. 

**Abstract (ZH)**: 基于多尺度时空Q网络的机器人路径规划：GundamQ 

---
# Efficient Learning-Based Control of a Legged Robot in Lunar Gravity 

**Title (ZH)**: 基于学习的月球重力环境下腿部机器人高效控制 

**Authors**: Philip Arm, Oliver Fischer, Joseph Church, Adrian Fuhrer, Hendrik Kolvenbach, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2509.10128)  

**Abstract**: Legged robots are promising candidates for exploring challenging areas on low-gravity bodies such as the Moon, Mars, or asteroids, thanks to their advanced mobility on unstructured terrain. However, as planetary robots' power and thermal budgets are highly restricted, these robots need energy-efficient control approaches that easily transfer to multiple gravity environments. In this work, we introduce a reinforcement learning-based control approach for legged robots with gravity-scaled power-optimized reward functions. We use our approach to develop and validate a locomotion controller and a base pose controller in gravity environments from lunar gravity (1.62 m/s2) to a hypothetical super-Earth (19.62 m/s2). Our approach successfully scales across these gravity levels for locomotion and base pose control with the gravity-scaled reward functions. The power-optimized locomotion controller reached a power consumption for locomotion of 23.4 W in Earth gravity on a 15.65 kg robot at 0.4 m/s, a 23 % improvement over the baseline policy. Additionally, we designed a constant-force spring offload system that allowed us to conduct real-world experiments on legged locomotion in lunar gravity. In lunar gravity, the power-optimized control policy reached 12.2 W, 36 % less than a baseline controller which is not optimized for power efficiency. Our method provides a scalable approach to developing power-efficient locomotion controllers for legged robots across multiple gravity levels. 

**Abstract (ZH)**: 基于强化学习的重力调整能量优化四足机器人控制方法 

---
# HHI-Assist: A Dataset and Benchmark of Human-Human Interaction in Physical Assistance Scenario 

**Title (ZH)**: HHI-Assist: 人体辅助场景中人类-人类交互的 datasets 和基准 

**Authors**: Saeed Saadatnejad, Reyhaneh Hosseininejad, Jose Barreiros, Katherine M. Tsui, Alexandre Alahi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10096)  

**Abstract**: The increasing labor shortage and aging population underline the need for assistive robots to support human care recipients. To enable safe and responsive assistance, robots require accurate human motion prediction in physical interaction scenarios. However, this remains a challenging task due to the variability of assistive settings and the complexity of coupled dynamics in physical interactions. In this work, we address these challenges through two key contributions: (1) HHI-Assist, a dataset comprising motion capture clips of human-human interactions in assistive tasks; and (2) a conditional Transformer-based denoising diffusion model for predicting the poses of interacting agents. Our model effectively captures the coupled dynamics between caregivers and care receivers, demonstrating improvements over baselines and strong generalization to unseen scenarios. By advancing interaction-aware motion prediction and introducing a new dataset, our work has the potential to significantly enhance robotic assistance policies. The dataset and code are available at: this https URL 

**Abstract (ZH)**: 劳动短缺和人口老龄化加剧了对辅助机器人支持人类护理Recipient的需求。为了实现安全和及时的辅助，机器人在物理交互场景中需要准确预测人类运动。然而，由于辅助环境的变异性以及物理交互中的耦合动力学复杂性，这一任务仍然极具挑战性。在本研究中，我们通过两项关键贡献应对这些挑战：(1) HHI-Assist数据集，包含人类在辅助任务中互动的运动捕捉片段；(2) 基于条件Transformer的去噪扩散模型，用于预测互动代理的姿态。我们的模型有效地捕捉了护理人员与护理接受者之间的耦合动力学，显示出相对于基线的改进和对未见过的场景的强大泛化能力。通过推进交互感知运动预测并引入新的数据集，本研究有潜力显著增强机器人的辅助策略。数据集和代码可在以下链接获取：this https URL。 

---
# Detection of Anomalous Behavior in Robot Systems Based on Machine Learning 

**Title (ZH)**: 基于机器学习的机器人系统异常行为检测 

**Authors**: Mahfuzul I. Nissan, Sharmin Aktar  

**Link**: [PDF](https://arxiv.org/pdf/2509.09953)  

**Abstract**: Ensuring the safe and reliable operation of robotic systems is paramount to prevent potential disasters and safeguard human well-being. Despite rigorous design and engineering practices, these systems can still experience malfunctions, leading to safety risks. In this study, we present a machine learning-based approach for detecting anomalies in system logs to enhance the safety and reliability of robotic systems. We collected logs from two distinct scenarios using CoppeliaSim and comparatively evaluated several machine learning models, including Logistic Regression (LR), Support Vector Machine (SVM), and an Autoencoder. Our system was evaluated in a quadcopter context (Context 1) and a Pioneer robot context (Context 2). Results showed that while LR demonstrated superior performance in Context 1, the Autoencoder model proved to be the most effective in Context 2. This highlights that the optimal model choice is context-dependent, likely due to the varying complexity of anomalies across different robotic platforms. This research underscores the value of a comparative approach and demonstrates the particular strengths of autoencoders for detecting complex anomalies in robotic systems. 

**Abstract (ZH)**: 基于机器学习的系统日志异常检测方法：确保机器人系统的安全可靠运行 

---
# Using the Pepper Robot to Support Sign Language Communication 

**Title (ZH)**: 使用Pepper机器人支持手语交流 

**Authors**: Giulia Botta, Marco Botta, Cristina Gena, Alessandro Mazzei, Massimo Donini, Alberto Lillo  

**Link**: [PDF](https://arxiv.org/pdf/2509.09889)  

**Abstract**: Social robots are increasingly experimented in public and assistive settings, but their accessibility for Deaf users remains quite underexplored. Italian Sign Language (LIS) is a fully-fledged natural language that relies on complex manual and non-manual components. Enabling robots to communicate using LIS could foster more inclusive human robot interaction, especially in social environments such as hospitals, airports, or educational settings. This study investigates whether a commercial social robot, Pepper, can produce intelligible LIS signs and short signed LIS sentences. With the help of a Deaf student and his interpreter, an expert in LIS, we co-designed and implemented 52 LIS signs on Pepper using either manual animation techniques or a MATLAB based inverse kinematics solver. We conducted a exploratory user study involving 12 participants proficient in LIS, both Deaf and hearing. Participants completed a questionnaire featuring 15 single-choice video-based sign recognition tasks and 2 open-ended questions on short signed sentences. Results shows that the majority of isolated signs were recognized correctly, although full sentence recognition was significantly lower due to Pepper's limited articulation and temporal constraints. Our findings demonstrate that even commercially available social robots like Pepper can perform a subset of LIS signs intelligibly, offering some opportunities for a more inclusive interaction design. Future developments should address multi-modal enhancements (e.g., screen-based support or expressive avatars) and involve Deaf users in participatory design to refine robot expressivity and usability. 

**Abstract (ZH)**: 社交机器人在公共和辅助环境中日益增多，但其在聋人用户中的可访问性仍被严重忽视。意大利手语（LIS）是一种完备的自然语言，依赖于复杂的手动和非手动成分。使机器人能够使用LIS进行交流可以促进更具包容性的机器人-人交互，特别是在医院、机场或教育环境中。本研究调查商用社交机器人Pepper是否能够产生可理解的LIS手势和简短的手语句子。在聋人学生及其手语翻译专家的帮助下，我们使用手动动画技术和基于MATLAB的逆向动力学求解器在Pepper上共同设计并实现了52个LIS手势。我们进行了一项探索性用户研究，共有12名 proficient 的LIS用户（聋人和 Hearing 人士）参与。参与者完成了包含15个单选视频手语识别任务的问卷，并回答了关于简短手语句子的两道开放式问题。研究结果表明，虽然大多数孤立手势被正确识别，但完整的句子识别率较低，原因主要是由于Pepper的语言表达能力和时间约束限制。我们的研究结果显示，即使像Pepper这样的商用社交机器人也可以在一定程度上实现可理解的LIS手势，为更具包容性的交互设计提供了某些机会。未来的研究应解决多模态增强（例如，基于屏幕的支持或具有表现力的虚拟角色）问题，并让聋人用户参与联合设计，以改进机器人表达性和易用性。 

---
# MIMo grows! Simulating body and sensory development in a multimodal infant model 

**Title (ZH)**: MIMo 生长了！多模态婴儿模型中的身体和感觉发展模拟 

**Authors**: Francisco M. López, Miles Lenz, Marco G. Fedozzi, Arthur Aubret, Jochen Triesch  

**Link**: [PDF](https://arxiv.org/pdf/2509.09805)  

**Abstract**: Infancy is characterized by rapid body growth and an explosive change of sensory and motor abilities. However, developmental robots and simulation platforms are typically designed in the image of a specific age, which limits their ability to capture the changing abilities and constraints of developing infants. To address this issue, we present MIMo v2, a new version of the multimodal infant model. It includes a growing body with increasing actuation strength covering the age range from birth to 24 months. It also features foveated vision with developing visual acuity as well as sensorimotor delays modeling finite signal transmission speeds to and from an infant's brain. Further enhancements of this MIMo version include an inverse kinematics module, a random environment generator and updated compatiblity with third-party simulation and learning libraries. Overall, this new MIMo version permits increased realism when modeling various aspects of sensorimotor development. The code is available on the official repository (this https URL). 

**Abstract (ZH)**: 婴儿期的特点是快速的身体生长和感觉运动能力的爆发性变化。然而，现有的发展型机器人和模拟平台通常以特定年龄段为模版设计，这限制了它们捕捉婴儿发育过程中变化的能力和约束。为解决这一问题，我们提出了MIMo v2，这是多模态婴儿模型的新版本。它包含一个随时间增长的身体，并逐步增强肌肉控制力，覆盖从出生到24个月的年龄范围。此外，还具备具有发展视敏度的中心视野以及传感器运动延迟，模拟从婴儿大脑到外部环境的信号传递速度限制。这一MIMo版本的进一步增强包括逆运动学模块、随机环境生成器以及与第三方模拟和学习库的更新兼容性。总体而言，这一新的MIMo版本使在建模感觉运动发展的各个方面时能够增加现实感。相关代码已发布在官方仓库（this https URL）。 

---
# MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos 

**Title (ZH)**: MimicDroid：从人类操作视频中进行上下文学习的人形机器人操作方法 

**Authors**: Rutav Shah, Shuijing Liu, Qi Wang, Zhenyu Jiang, Sateesh Kumar, Mingyo Seo, Roberto Martín-Martín, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09769)  

**Abstract**: We aim to enable humanoid robots to efficiently solve new manipulation tasks from a few video examples. In-context learning (ICL) is a promising framework for achieving this goal due to its test-time data efficiency and rapid adaptability. However, current ICL methods rely on labor-intensive teleoperated data for training, which restricts scalability. We propose using human play videos -- continuous, unlabeled videos of people interacting freely with their environment -- as a scalable and diverse training data source. We introduce MimicDroid, which enables humanoids to perform ICL using human play videos as the only training data. MimicDroid extracts trajectory pairs with similar manipulation behaviors and trains the policy to predict the actions of one trajectory conditioned on the other. Through this process, the model acquired ICL capabilities for adapting to novel objects and environments at test time. To bridge the embodiment gap, MimicDroid first retargets human wrist poses estimated from RGB videos to the humanoid, leveraging kinematic similarity. It also applies random patch masking during training to reduce overfitting to human-specific cues and improve robustness to visual differences. To evaluate few-shot learning for humanoids, we introduce an open-source simulation benchmark with increasing levels of generalization difficulty. MimicDroid outperformed state-of-the-art methods and achieved nearly twofold higher success rates in the real world. Additional materials can be found on: this http URL 

**Abstract (ZH)**: 我们旨在使类人机器人能够从少量视频示例中高效地解决新的操纵任务。上下文学习（ICL）因其测试时的数据高效性和快速适应性是实现这一目标的一个有前景的框架。然而，当前的ICL方法依赖于劳动密集型的遥控训练数据，这限制了其可扩展性。我们提出使用人类游戏视频——自由与环境交互的连续未标记视频——作为可扩展且多样化的训练数据来源。我们介绍了MimicDroid，它使类人机器人能够仅使用人类游戏视频作为训练数据进行上下文学习。MimicDroid提取具有类似操纵行为的轨迹对，并训练策略在给定另一个轨迹的情况下预测动作。通过这一过程，模型在测试时获得了适应新对象和环境的ICL能力。为弥补体现实体差距，MimicDroid首先将从RGB视频估计的人类手腕姿态重新定向到类人机器人，利用运动学相似性。在训练过程中，它还应用随机补丁遮罩以减少对人类特定线索的过度拟合并提高对视觉差异的鲁棒性。为评估类人机器人的少量样本学习，我们引入了一个逐步增加泛化难度的开源仿真基准。MimicDroid在仿真和现实世界中均表现出色，成功率几乎提高了一倍。更多材料可在以下链接获取：this http URL 

---
# Mutual Information Tracks Policy Coherence in Reinforcement Learning 

**Title (ZH)**: 互信息追踪强化学习中政策一致性 

**Authors**: Cameron Reid, Wael Hafez, Amirhossein Nazeri  

**Link**: [PDF](https://arxiv.org/pdf/2509.10423)  

**Abstract**: Reinforcement Learning (RL) agents deployed in real-world environments face degradation from sensor faults, actuator wear, and environmental shifts, yet lack intrinsic mechanisms to detect and diagnose these failures. We present an information-theoretic framework that reveals both the fundamental dynamics of RL and provides practical methods for diagnosing deployment-time anomalies. Through analysis of state-action mutual information patterns in a robotic control task, we first demonstrate that successful learning exhibits characteristic information signatures: mutual information between states and actions steadily increases from 0.84 to 2.83 bits (238% growth) despite growing state entropy, indicating that agents develop increasingly selective attention to task-relevant patterns. Intriguingly, states, actions and next states joint mutual information, MI(S,A;S'), follows an inverted U-curve, peaking during early learning before declining as the agent specializes suggesting a transition from broad exploration to efficient exploitation. More immediately actionable, we show that information metrics can differentially diagnose system failures: observation-space, i.e., states noise (sensor faults) produces broad collapses across all information channels with pronounced drops in state-action coupling, while action-space noise (actuator faults) selectively disrupts action-outcome predictability while preserving state-action relationships. This differential diagnostic capability demonstrated through controlled perturbation experiments enables precise fault localization without architectural modifications or performance degradation. By establishing information patterns as both signatures of learning and diagnostic for system health, we provide the foundation for adaptive RL systems capable of autonomous fault detection and policy adjustment based on information-theoretic principles. 

**Abstract (ZH)**: 基于信息论的强化学习代理故障检测与诊断框架 

---
# Data-fused Model Predictive Control with Guarantees: Application to Flying Humanoid Robots 

**Title (ZH)**: 具有保障的数据融合模型预测控制：应用于飞行类人机器人 

**Authors**: Davide Gorbani, Mohamed Elobaid, Giuseppe L'Erario, Hosameldin Awadalla Omer Mohamed, Daniele Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.10353)  

**Abstract**: This paper introduces a Data-Fused Model Predictive Control (DFMPC) framework that combines physics-based models with data-driven representations of unknown dynamics. Leveraging Willems' Fundamental Lemma and an artificial equilibrium formulation, the method enables tracking of changing, potentially unreachable setpoints while explicitly handling measurement noise through slack variables and regularization. We provide guarantees of recursive feasibility and practical stability under input-output constraints for a specific class of reference signals. The approach is validated on the iRonCub flying humanoid robot, integrating analytical momentum models with data-driven turbine dynamics. Simulations show improved tracking and robustness compared to a purely model-based MPC, while maintaining real-time feasibility. 

**Abstract (ZH)**: 基于数据融合的模型预测控制框架：结合物理模型与未知动态的数据驱动表示 

---
# Towards Fully Automated Molecular Simulations: Multi-Agent Framework for Simulation Setup and Force Field Extraction 

**Title (ZH)**: 面向完全自动化的分子模拟：模拟设置与力场提取的多智能体框架 

**Authors**: Marko Petković, Vlado Menkovski, Sofía Calero  

**Link**: [PDF](https://arxiv.org/pdf/2509.10210)  

**Abstract**: Automated characterization of porous materials has the potential to accelerate materials discovery, but it remains limited by the complexity of simulation setup and force field selection. We propose a multi-agent framework in which LLM-based agents can autonomously understand a characterization task, plan appropriate simulations, assemble relevant force fields, execute them and interpret their results to guide subsequent steps. As a first step toward this vision, we present a multi-agent system for literature-informed force field extraction and automated RASPA simulation setup. Initial evaluations demonstrate high correctness and reproducibility, highlighting this approach's potential to enable fully autonomous, scalable materials characterization. 

**Abstract (ZH)**: 基于大模型的多agent系统在孔材料表征中的应用：面向文献的力场提取与自动RASPA模拟设置 

---
# XAgents: A Unified Framework for Multi-Agent Cooperation via IF-THEN Rules and Multipolar Task Processing Graph 

**Title (ZH)**: XAgents：基于IF-THEN规则和多极任务处理图的统一多agent合作框架 

**Authors**: Hailong Yang, Mingxian Gu, Jianqi Wang, Guanjin Wang, Zhaohong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10054)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has significantly enhanced the capabilities of Multi-Agent Systems (MAS) in supporting humans with complex, real-world tasks. However, MAS still face challenges in effective task planning when handling highly complex tasks with uncertainty, often resulting in misleading or incorrect outputs that hinder task execution. To address this, we propose XAgents, a unified multi-agent cooperative framework built on a multipolar task processing graph and IF-THEN rules. XAgents uses the multipolar task processing graph to enable dynamic task planning and handle task uncertainty. During subtask processing, it integrates domain-specific IF-THEN rules to constrain agent behaviors, while global rules enhance inter-agent collaboration. We evaluate the performance of XAgents across three distinct datasets, demonstrating that it consistently surpasses state-of-the-art single-agent and multi-agent approaches in both knowledge-typed and logic-typed question-answering tasks. The codes for XAgents are available at: this https URL. 

**Abstract (ZH)**: 大型语言模型的迅速发展显著增强了多代理系统在支持复杂现实任务方面的能力。然而，多代理系统在处理高复杂性和不确定性任务时，仍面临有效的任务规划挑战，常常导致误导或错误的输出，阻碍任务执行。为解决这一问题，我们提出XAgents，这是一个基于多极任务处理图和IF-THEN规则的一体化多代理协同框架。XAgents利用多极任务处理图实现动态任务规划并处理任务不确定性。在子任务处理过程中，它通过整合领域特定的IF-THEN规则约束代理行为，同时全局规则增强代理间的协作。我们在三个不同的数据集上评估了XAgents的性能，结果显示它在知识型和逻辑型问答任务中均优于现有的单代理和多代理方法。XAgents的代码可在以下链接获得：this https URL。 

---
# Towards an AI-based knowledge assistant for goat farmers based on Retrieval-Augmented Generation 

**Title (ZH)**: 基于检索增强生成的面向羊农的AI知识助手 

**Authors**: Nana Han, Dong Liu, Tomas Norton  

**Link**: [PDF](https://arxiv.org/pdf/2509.09848)  

**Abstract**: Large language models (LLMs) are increasingly being recognised as valuable knowledge communication tools in many industries. However, their application in livestock farming remains limited, being constrained by several factors not least the availability, diversity and complexity of knowledge sources. This study introduces an intelligent knowledge assistant system designed to support health management in farmed goats. Leveraging the Retrieval-Augmented Generation (RAG), two structured knowledge processing methods, table textualization and decision-tree textualization, were proposed to enhance large language models' (LLMs) understanding of heterogeneous data formats. Based on these methods, a domain-specific goat farming knowledge base was established to improve LLM's capacity for cross-scenario generalization. The knowledge base spans five key domains: Disease Prevention and Treatment, Nutrition Management, Rearing Management, Goat Milk Management, and Basic Farming Knowledge. Additionally, an online search module is integrated to enable real-time retrieval of up-to-date information. To evaluate system performance, six ablation experiments were conducted to examine the contribution of each component. The results demonstrated that heterogeneous knowledge fusion method achieved the best results, with mean accuracies of 87.90% on the validation set and 84.22% on the test set. Across the text-based, table-based, decision-tree based Q&A tasks, accuracy consistently exceeded 85%, validating the effectiveness of structured knowledge fusion within a modular design. Error analysis identified omission as the predominant error category, highlighting opportunities to further improve retrieval coverage and context integration. In conclusion, the results highlight the robustness and reliability of the proposed system for practical applications in goat farming. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个行业被日益认可为有价值的知识交流工具，但在畜牧业中的应用仍然受限，主要受知识来源的可用性、多样性和复杂性等因素的制约。本研究介绍了一种智能知识助手系统，旨在支持养羊健康管理工作。利用检索增强生成（RAG）技术，提出了两种结构化知识处理方法：表格文本化和决策树文本化，以增强大型语言模型（LLMs）对异构数据格式的理解能力。基于这些方法，建立了一个专门针对养羊业的知识库，以提高LLMs在跨场景泛化能力。该知识库涵盖了五个关键领域：疾病预防与治疗、营养管理、饲养管理、山羊奶管理以及基础农学知识。此外，还集成了在线搜索模块，以实现实时检索最新信息。为了评估系统性能，进行了六项消融实验，以检查每个组件的贡献。结果表明，异构知识融合方法表现最佳，验证集的平均准确率为87.90%，测试集为84.22%。在基于文本、基于表格和基于决策树的问答任务中，准确率均超过85%，验证了模块化设计中结构化知识融合的有效性。误差分析表明，遗漏是最主要的错误类别，指出了进一步提高检索覆盖率和上下文整合的机会。总之，结果突显了所提出系统在养羊业实际应用中的稳健性和可靠性。 

---
# Generalizing Beyond Suboptimality: Offline Reinforcement Learning Learns Effective Scheduling through Random Data 

**Title (ZH)**: 超越亚优解的泛化：离线强化学习通过随机数据学习有效的调度 

**Authors**: Jesse van Remmerden, Zaharah Bukhsh, Yingqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10303)  

**Abstract**: The Job-Shop Scheduling Problem (JSP) and Flexible Job-Shop Scheduling Problem (FJSP), are canonical combinatorial optimization problems with wide-ranging applications in industrial operations. In recent years, many online reinforcement learning (RL) approaches have been proposed to learn constructive heuristics for JSP and FJSP. Although effective, these online RL methods require millions of interactions with simulated environments that may not capture real-world complexities, and their random policy initialization leads to poor sample efficiency. To address these limitations, we introduce Conservative Discrete Quantile Actor-Critic (CDQAC), a novel offline RL algorithm that learns effective scheduling policies directly from historical data, eliminating the need for costly online interactions, while maintaining the ability to improve upon suboptimal training data. CDQAC couples a quantile-based critic with a delayed policy update, estimating the return distribution of each machine-operation pair rather than selecting pairs outright. Our extensive experiments demonstrate CDQAC's remarkable ability to learn from diverse data sources. CDQAC consistently outperforms the original data-generating heuristics and surpasses state-of-the-art offline and online RL baselines. In addition, CDQAC is highly sample efficient, requiring only 10-20 training instances to learn high-quality policies. Surprisingly, we find that CDQAC performs better when trained on data generated by a random heuristic than when trained on higher-quality data from genetic algorithms and priority dispatching rules. 

**Abstract (ZH)**: 基于保守离线量纲演员评论家的作业车间调度问题与灵活作业车间调度问题的离线强化学习算法 

---
# Reinforcement learning for spin torque oscillator tasks 

**Title (ZH)**: 自旋扭矩振荡器任务的强化学习方法 

**Authors**: Jakub Mojsiejuk, Sławomir Ziętek, Witold Skowroński  

**Link**: [PDF](https://arxiv.org/pdf/2509.10057)  

**Abstract**: We address the problem of automatic synchronisation of the spintronic oscillator (STO) by means of reinforcement learning (RL). A numerical solution of the macrospin Landau-Lifschitz-Gilbert-Slonczewski equation is used to simulate the STO and we train the two types of RL agents to synchronise with a target frequency within a fixed number of steps. We explore modifications to this base task and show an improvement in both convergence and energy efficiency of the synchronisation that can be easily achieved in the simulated environment. 

**Abstract (ZH)**: 我们通过强化学习解决自旋电子振荡器自动同步问题：通过数值求解宏观自旋兰杜-利夫西茨-吉尔伯特-斯隆切夫斯基方程模拟自旋电子振荡器，并训练两种类型的RL代理在固定步数内与目标频率同步。我们探索了对该基本任务的修改，并展示了在模拟环境中轻松实现的同步收敛性和能效的改进。 

---
# Revisiting Actor-Critic Methods in Discrete Action Off-Policy Reinforcement Learning 

**Title (ZH)**: 离散动作 Offline 政策强化学习中 Actor-Critic 方法的重新审视 

**Authors**: Reza Asad, Reza Babanezhad, Sharan Vaswani  

**Link**: [PDF](https://arxiv.org/pdf/2509.09838)  

**Abstract**: Value-based approaches such as DQN are the default methods for off-policy reinforcement learning with discrete-action environments such as Atari. Common policy-based methods are either on-policy and do not effectively learn from off-policy data (e.g. PPO), or have poor empirical performance in the discrete-action setting (e.g. SAC). Consequently, starting from discrete SAC (DSAC), we revisit the design of actor-critic methods in this setting. First, we determine that the coupling between the actor and critic entropy is the primary reason behind the poor performance of DSAC. We demonstrate that by merely decoupling these components, DSAC can have comparable performance as DQN. Motivated by this insight, we introduce a flexible off-policy actor-critic framework that subsumes DSAC as a special case. Our framework allows using an m-step Bellman operator for the critic update, and enables combining standard policy optimization methods with entropy regularization to instantiate the resulting actor objective. Theoretically, we prove that the proposed methods can guarantee convergence to the optimal regularized value function in the tabular setting. Empirically, we demonstrate that these methods can approach the performance of DQN on standard Atari games, and do so even without entropy regularization or explicit exploration. 

**Abstract (ZH)**: 基于价值的方法如DQN是离策强化学习中，默认的方法，特别是在 Atari 这类离散动作环境中。常见的基于策略的方法要么是在线策的，不能有效从离策数据中学习（例如 PPO），要么在离散动作设置中有较差的实证表现（例如 SAC）。因此，从离散 SAC (DSAC) 开始，我们重新审视了这类设置下的演员-评论家方法的设计。首先，我们确定演员和评论家熵之间的耦合是 DSAC 表现不佳的主要原因。我们通过仅解除这些组件之间的耦合，证明 DSAC 可以达到与 DQN 相媲美的性能。受此启发，我们提出了一种灵活的离策演员-评论家框架，将 DSAC 作为其特殊情形。该框架允许使用 m 步贝尔曼运算符来更新评论家，并能够结合标准策略优化方法和熵正则化来实例化结果演员目标。理论上，我们证明所提出的方法可以在表征设置下保证收敛到最优的正则化价值函数。实证上，我们展示这些方法可以接近标准 Atari 游戏中的 DQN 性能，甚至无需熵正则化或显式探索也能做到这一点。 

---
# World Modeling with Probabilistic Structure Integration 

**Title (ZH)**: 基于概率结构集成的世界建模 

**Authors**: Klemen Kotar, Wanhee Lee, Rahul Venkatesh, Honglin Chen, Daniel Bear, Jared Watrous, Simon Kim, Khai Loong Aw, Lilian Naing Chen, Stefan Stojanov, Kevin Feigelis, Imran Thobani, Alex Durango, Khaled Jedoui, Atlas Kazemian, Dan Yamins  

**Link**: [PDF](https://arxiv.org/pdf/2509.09737)  

**Abstract**: We present Probabilistic Structure Integration (PSI), a system for learning richly controllable and flexibly promptable world models from data. PSI consists of a three-step cycle. The first step, Probabilistic prediction, involves building a probabilistic graphical model Psi of the data, in the form of a random-access autoregressive sequence model. Psi supports a complete set of learned conditional distributions describing the dependence of any variables in the data on any other set of variables. In step 2, Structure extraction, we show how to extract underlying low-dimensional properties in the data, corresponding to a diverse set of meaningful "intermediate structures", in a zero-shot fashion via causal inference on Psi. Step 3, Integration, completes the cycle by converting these structures into new token types that are then continually mixed back into the training diet as conditioning signals and prediction targets. Each such cycle augments the capabilities of Psi, both allowing it to model the underlying data better, and creating new control handles -- akin to an LLM-like universal prompting language. We train an instance of Psi on 1.4 trillion tokens of internet video data; we use it to perform a variety of useful video prediction and understanding inferences; we extract state-of-the-art optical flow, self-supervised depth and object segmentation; and we use these structures to support a full cycle of predictive improvements. 

**Abstract (ZH)**: 概率结构集成（PSI）：从数据中学习丰富可控和灵活可调的世界模型系统 

---
# MultimodalHugs: Enabling Sign Language Processing in Hugging Face 

**Title (ZH)**: 多模态拥抱：在 Hugging Face 平台上实现手语处理 

**Authors**: Gerard Sant, Zifan Jiang, Carlos Escolano, Amit Moryossef, Mathias Müller, Rico Sennrich, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2509.09729)  

**Abstract**: In recent years, sign language processing (SLP) has gained importance in the general field of Natural Language Processing. However, compared to research on spoken languages, SLP research is hindered by complex ad-hoc code, inadvertently leading to low reproducibility and unfair comparisons. Existing tools that are built for fast and reproducible experimentation, such as Hugging Face, are not flexible enough to seamlessly integrate sign language experiments. This view is confirmed by a survey we conducted among SLP researchers.
To address these challenges, we introduce MultimodalHugs, a framework built on top of Hugging Face that enables more diverse data modalities and tasks, while inheriting the well-known advantages of the Hugging Face ecosystem. Even though sign languages are our primary focus, MultimodalHugs adds a layer of abstraction that makes it more widely applicable to other use cases that do not fit one of the standard templates of Hugging Face. We provide quantitative experiments to illustrate how MultimodalHugs can accommodate diverse modalities such as pose estimation data for sign languages, or pixel data for text characters. 

**Abstract (ZH)**: 近年来，手语处理（SLP）在自然语言处理领域获得了重要地位。然而，与口头语言研究相比，SLP研究受限于复杂的定制代码，导致可重复性低和不公平的比较。现有的用于快速和可重复实验的工具，如Hugging Face，不足以无缝集成手语实验。我们对SLP研究人员进行的一项调查显示，这一观点得到了证实。为了应对这些挑战，我们引入了MultimodalHugs框架，该框架基于Hugging Face，并能够支持更广泛的数据模态和任务，同时继承了Hugging Face生态系统的显著优势。尽管手语是我们的主要研究对象，但MultimodalHugs通过增加一层抽象，使其更广泛适用于不符合Hugging Face标准模板的其他应用场景。我们提供了定量实验来说明MultimodalHugs如何容纳多样化的模态，如手语的姿势估计数据或文本字符的像素数据。 

---
# TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation 

**Title (ZH)**: TalkPlayData 2: 一种自主式多模态对话音乐推荐合成数据管道 

**Authors**: Keunwoo Choi, Seungheon Doh, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2509.09685)  

**Abstract**: We present TalkPlayData 2, a synthetic dataset for multimodal conversational music recommendation generated by an agentic data pipeline. In TalkPlayData 2 pipeline, multiple large language model (LLM) agents are created under various roles with specialized prompts and access to different parts of information, and the chat data is acquired by logging the conversation between the Listener LLM and the Recsys LLM. To cover various conversation scenarios, for each conversation, the Listener LLM is conditioned on a finetuned conversation goal. Finally, all the LLMs are multimodal with audio and images, allowing a simulation of multimodal recommendation and conversation. In the LLM-as-a-judge and subjective evaluation experiments, TalkPlayData 2 achieved the proposed goal in various aspects related to training a generative recommendation model for music. TalkPlayData 2 and its generation code are open-sourced at this https URL. 

**Abstract (ZH)**: 我们呈现 TalkPlayData 2，这是一个由自主数据管道生成的多模态对话音乐推荐合成数据集 

---
