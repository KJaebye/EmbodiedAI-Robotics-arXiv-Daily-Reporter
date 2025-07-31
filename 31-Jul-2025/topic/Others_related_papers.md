# Bayesian Optimization applied for accelerated Virtual Validation of the Autonomous Driving Function 

**Title (ZH)**: 应用于自主驾驶功能加速虚拟验证的贝叶斯优化方法 

**Authors**: Satyesh Shanker Awasthi, Mohammed Irshadh Ismaaeel Sathyamangalam Imran, Stefano Arrigoni, Francesco Braghin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22769)  

**Abstract**: Rigorous Verification and Validation (V&V) of Autonomous Driving Functions (ADFs) is paramount for ensuring the safety and public acceptance of Autonomous Vehicles (AVs). Current validation relies heavily on simulation to achieve sufficient test coverage within the Operational Design Domain (ODD) of a vehicle, but exhaustively exploring the vast parameter space of possible scenarios is computationally expensive and time-consuming. This work introduces a framework based on Bayesian Optimization (BO) to accelerate the discovery of critical scenarios. We demonstrate the effectiveness of the framework on an Model Predictive Controller (MPC)-based motion planner, showing that it identifies hazardous situations, such as off-road events, using orders of magnitude fewer simulations than brute-force Design of Experiments (DoE) methods. Furthermore, this study investigates the scalability of the framework in higher-dimensional parameter spaces and its ability to identify multiple, distinct critical regions within the ODD of the motion planner used as the case study . 

**Abstract (ZH)**: 严格验证与验证（V&V）对于确保自动驾驶功能（ADFs）的安全性和公众接受度至关重要。当前的验证主要依赖于模拟来实现对车辆操作设计域（ODD）的充分测试覆盖，但全面探索可能场景的庞大参数空间在计算上非常昂贵且耗时。本研究提出了一种基于贝叶斯优化（BO）的框架，以加速关键场景的发现。我们通过在基于模型预测控制（MPC）的运动规划器上验证该框架的有效性，结果显示它使用比暴力设计实验（DoE）方法小数量级的模拟就能识别出危险情况，如脱道路事件。此外，本研究还探讨了该框架在高维参数空间中的可扩展性及其在所研究的运动规划器ODD中识别多个独立关键区域的能力。 

---
# Explainable Deep Anomaly Detection with Sequential Hypothesis Testing for Robotic Sewer Inspection 

**Title (ZH)**: 基于序列假设检验的可解释深度异常检测在机器人 sewer 检查中的应用 

**Authors**: Alex George, Will Shepherd, Simon Tait, Lyudmila Mihaylova, Sean R. Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2507.22546)  

**Abstract**: Sewer pipe faults, such as leaks and blockages, can lead to severe consequences including groundwater contamination, property damage, and service disruption. Traditional inspection methods rely heavily on the manual review of CCTV footage collected by mobile robots, which is inefficient and susceptible to human error. To automate this process, we propose a novel system incorporating explainable deep learning anomaly detection combined with sequential probability ratio testing (SPRT). The anomaly detector processes single image frames, providing interpretable spatial localisation of anomalies, whilst the SPRT introduces temporal evidence aggregation, enhancing robustness against noise over sequences of image frames. Experimental results demonstrate improved anomaly detection performance, highlighting the benefits of the combined spatiotemporal analysis system for reliable and robust sewer inspection. 

**Abstract (ZH)**: Sewer 管道故障（如泄漏和堵塞）可能导致地下水污染、财产损失和服务中断。传统检查方法主要依赖移动机器人收集的CCTV footage的手动审查，效率低且易出错。为自动化这一过程，我们提出了一种结合可解释的深度学习异常检测和序列概率比率测试（SPRT）的新型系统。异常检测器处理单张图像帧，提供可解释的空间定位，而SPRT引入了时间证据聚合，增强了对图像帧序列中噪声的鲁棒性。实验结果表明，这种结合时空分析系统的异常检测性能改进，突显了其在可靠和 robust 管道检查中的优势。 

---
# Operationalization of Scenario-Based Safety Assessment of Automated Driving Systems 

**Title (ZH)**: 基于场景的安全评估自动化驾驶系统操作化 

**Authors**: Olaf Op den Camp, Erwin de Gelder  

**Link**: [PDF](https://arxiv.org/pdf/2507.22433)  

**Abstract**: Before introducing an Automated Driving System (ADS) on the road at scale, the manufacturer must conduct some sort of safety assurance. To structure and harmonize the safety assurance process, the UNECE WP.29 Working Party on Automated/Autonomous and Connected Vehicles (GRVA) is developing the New Assessment/Test Method (NATM) that indicates what steps need to be taken for safety assessment of an ADS. In this paper, we will show how to practically conduct safety assessment making use of a scenario database, and what additional steps must be taken to fully operationalize the NATM. In addition, we will elaborate on how the use of scenario databases fits with methods developed in the Horizon Europe projects that focus on safety assessment following the NATM approach. 

**Abstract (ZH)**: 在大规模部署自动驾驶系统之前，制造商必须开展某种形式的安全保证工作。为了结构化和协调安全保证过程，联合国经济及社会理事会 WP.29 自动/自主及连接车辆工作组（GRVA）正在开发新的评估/测试方法（NATM），以指示对自动驾驶系统进行安全评估所必需的步骤。在本文中，我们将展示如何利用场景数据库进行实际的安全评估，并说明为了完全实现NATM还需要采取哪些额外步骤。此外，我们将详细阐述场景数据库的使用如何与 Horizon Europe 项目中开发的安全评估方法（遵循NATM方法）相契合。 

---
# Comparing Normalizing Flows with Kernel Density Estimation in Estimating Risk of Automated Driving Systems 

**Title (ZH)**: 比较归一化流与核密度估计在评估自动驾驶系统风险中的表现 

**Authors**: Erwin de Gelder, Maren Buermann, Olaf Op den Camp  

**Link**: [PDF](https://arxiv.org/pdf/2507.22429)  

**Abstract**: The development of safety validation methods is essential for the safe deployment and operation of Automated Driving Systems (ADSs). One of the goals of safety validation is to prospectively evaluate the risk of an ADS dealing with real-world traffic. Scenario-based assessment is a widely-used approach, where test cases are derived from real-world driving data. To allow for a quantitative analysis of the system performance, the exposure of the scenarios must be accurately estimated. The exposure of scenarios at parameter level is expressed using a Probability Density Function (PDF). However, assumptions about the PDF, such as parameter independence, can introduce errors, while avoiding assumptions often leads to oversimplified models with limited parameters to mitigate the curse of dimensionality.
This paper considers the use of Normalizing Flows (NF) for estimating the PDF of the parameters. NF are a class of generative models that transform a simple base distribution into a complex one using a sequence of invertible and differentiable mappings, enabling flexible, high-dimensional density estimation without restrictive assumptions on the PDF's shape. We demonstrate the effectiveness of NF in quantifying risk and risk uncertainty of an ADS, comparing its performance with Kernel Density Estimation (KDE), a traditional method for non-parametric PDF estimation. While NF require more computational resources compared to KDE, NF is less sensitive to the curse of dimensionality. As a result, NF can improve risk uncertainty estimation, offering a more precise assessment of an ADS's safety.
This work illustrates the potential of NF in scenario-based safety. Future work involves experimenting more with using NF for scenario generation and optimizing the NF architecture, transformation types, and training hyperparameters to further enhance their applicability. 

**Abstract (ZH)**: 基于Normalizing Flows的安全评估方法在自动驾驶系统中的应用研究 

---
# Multi-Agent Path Finding Among Dynamic Uncontrollable Agents with Statistical Safety Guarantees 

**Title (ZH)**: 动态不可控代理中的统计安全性保证多代理路径寻找 

**Authors**: Kegan J. Strawn, Thomy Phan, Eric Wang, Nora Ayanian, Sven Koenig, Lars Lindemann  

**Link**: [PDF](https://arxiv.org/pdf/2507.22282)  

**Abstract**: Existing multi-agent path finding (MAPF) solvers do not account for uncertain behavior of uncontrollable agents. We present a novel variant of Enhanced Conflict-Based Search (ECBS), for both one-shot and lifelong MAPF in dynamic environments with uncontrollable agents. Our method consists of (1) training a learned predictor for the movement of uncontrollable agents, (2) quantifying the prediction error using conformal prediction (CP), a tool for statistical uncertainty quantification, and (3) integrating these uncertainty intervals into our modified ECBS solver. Our method can account for uncertain agent behavior, comes with statistical guarantees on collision-free paths for one-shot missions, and scales to lifelong missions with a receding horizon sequence of one-shot instances. We run our algorithm, CP-Solver, across warehouse and game maps, with competitive throughput and reduced collisions. 

**Abstract (ZH)**: 不确定行为的不可控代理下的多代理路径寻找：Enhanced Conflict-Based Search 的新型变体及其在动态环境中的应用 

---
# Modified Smith predictor for unstable linear systems 

**Title (ZH)**: 不稳定线性系统的修改Smith预估器 

**Authors**: Anton Pyrkin, Konstantin Kalinin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22243)  

**Abstract**: The paper presents a new control algorithm for unstable linear systems with input delay. In comparison with known analogues, the control law has been designed, which is a modification of the Smith predictor, and is the simplest one to implement without requiring complex integration methods. At the same time, the problem of stabilization of a closed system is effectively solved, ensuring the boundedness of all state variables and the exponential stability of the equilibrium point. 

**Abstract (ZH)**: 一种用于具有输入延迟的不稳定线性系统的控制算法及其实现 

---
# Toward Trusted Onboard AI: Advancing Small Satellite Operations using Reinforcement Learning 

**Title (ZH)**: 面向可信机载AI：增强学习推动小型卫星运行技术发展 

**Authors**: Cannon Whitney, Joseph Melville  

**Link**: [PDF](https://arxiv.org/pdf/2507.22198)  

**Abstract**: A RL (Reinforcement Learning) algorithm was developed for command automation onboard a 3U CubeSat. This effort focused on the implementation of macro control action RL, a technique in which an onboard agent is provided with compiled information based on live telemetry as its observation. The agent uses this information to produce high-level actions, such as adjusting attitude to solar pointing, which are then translated into control algorithms and executed through lower-level instructions. Once trust in the onboard agent is established, real-time environmental information can be leveraged for faster response times and reduced reliance on ground control. The approach not only focuses on developing an RL algorithm for a specific satellite but also sets a precedent for integrating trusted AI into onboard systems. This research builds on previous work in three areas: (1) RL algorithms for issuing high-level commands that are translated into low-level executable instructions; (2) the deployment of AI inference models interfaced with live operational systems, particularly onboard spacecraft; and (3) strategies for building trust in AI systems, especially for remote and autonomous applications. Existing RL research for satellite control is largely limited to simulation-based experiments; in this work, these techniques are tailored by constructing a digital twin of a specific spacecraft and training the RL agent to issue macro actions in this simulated environment. The policy of the trained agent is copied to an isolated environment, where it is fed compiled information about the satellite to make inference predictions, thereby demonstrating the RL algorithm's validity on orbit without granting it command authority. This process enables safe comparison of the algorithm's predictions against actual satellite behavior and ensures operation within expected parameters. 

**Abstract (ZH)**: 一种强化学习算法被开发用于3U立方星上的命令自动化宏控制动作强化学习技术的研究：建立可信赖的人工智能在星上系统中的先例 

---
# The Incomplete Bridge: How AI Research (Mis)Engages with Psychology 

**Title (ZH)**: 不完整的桥梁：AI研究与心理学的誤解与互动 

**Authors**: Han Jiang, Pengda Wang, Xiaoyuan Yi, Xing Xie, Ziang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22847)  

**Abstract**: Social sciences have accumulated a rich body of theories and methodologies for investigating the human mind and behaviors, while offering valuable insights into the design and understanding of Artificial Intelligence (AI) systems. Focusing on psychology as a prominent case, this study explores the interdisciplinary synergy between AI and the field by analyzing 1,006 LLM-related papers published in premier AI venues between 2023 and 2025, along with the 2,544 psychology publications they cite. Through our analysis, we identify key patterns of interdisciplinary integration, locate the psychology domains most frequently referenced, and highlight areas that remain underexplored. We further examine how psychology theories/frameworks are operationalized and interpreted, identify common types of misapplication, and offer guidance for more effective incorporation. Our work provides a comprehensive map of interdisciplinary engagement between AI and psychology, thereby facilitating deeper collaboration and advancing AI systems. 

**Abstract (ZH)**: 社会科学积累了丰富的理论和方法，用于探究人类思维和行为，并为人工智能（AI）系统的设计和理解提供了宝贵的见解。以心理学为重点案例，本研究通过分析2023年至2025年间发表在顶级AI会议上的1006篇与LLM相关的论文及其引用的2544篇心理学出版物，探讨了AI与心理学领域的跨学科协同效应。我们的分析揭示了跨学科整合的关键模式，指出了引用最频繁的心理学领域，并突出了尚未充分探索的领域。进一步研究了心理学理论/框架的实现和解释，明确了常见误用类型，并提供了更有效整合的建议。我们的工作为AI与心理学的跨学科互动提供了全面的地图，从而促进更深入的合作并推动人工智能系统的进步。 

---
# Enhancing Multi-Agent Collaboration with Attention-Based Actor-Critic Policies 

**Title (ZH)**: 基于注意力的演员-评论员策略增强多代理协作 

**Authors**: Hugo Garrido-Lestache, Jeremy Kedziora  

**Link**: [PDF](https://arxiv.org/pdf/2507.22782)  

**Abstract**: This paper introduces Team-Attention-Actor-Critic (TAAC), a reinforcement learning algorithm designed to enhance multi-agent collaboration in cooperative environments. TAAC employs a Centralized Training/Centralized Execution scheme incorporating multi-headed attention mechanisms in both the actor and critic. This design facilitates dynamic, inter-agent communication, allowing agents to explicitly query teammates, thereby efficiently managing the exponential growth of joint-action spaces while ensuring a high degree of collaboration. We further introduce a penalized loss function which promotes diverse yet complementary roles among agents. We evaluate TAAC in a simulated soccer environment against benchmark algorithms representing other multi-agent paradigms, including Proximal Policy Optimization and Multi-Agent Actor-Attention-Critic. We find that TAAC exhibits superior performance and enhanced collaborative behaviors across a variety of metrics (win rates, goal differentials, Elo ratings, inter-agent connectivity, balanced spatial distributions, and frequent tactical interactions such as ball possession swaps). 

**Abstract (ZH)**: 团队注意力优势 critic 算法（TAAC）：一种增强多智能体协作的强化学习方法 

---
# ASP-FZN: A Translation-based Constraint Answer Set Solver 

**Title (ZH)**: ASP-FZN: 基于翻译的约束满足解答集求解器 

**Authors**: Thomas Eiter, Tobias Geibinger, Tobias Kaminski, Nysret Musliu, Johannes Oetsch  

**Link**: [PDF](https://arxiv.org/pdf/2507.22774)  

**Abstract**: We present the solver asp-fzn for Constraint Answer Set Programming (CASP), which extends ASP with linear constraints. Our approach is based on translating CASP programs into the solver-independent FlatZinc language that supports several Constraint Programming and Integer Programming backend solvers. Our solver supports a rich language of linear constraints, including some common global constraints. As for evaluation, we show that asp-fzn is competitive with state-of-the-art ASP solvers on benchmarks taken from past ASP competitions. Furthermore, we evaluate it on several CASP problems from the literature and compare its performance with clingcon, which is a prominent CASP solver that supports most of the asp-fzn language. The performance of asp-fzn is very promising as it is already competitive on plain ASP and even outperforms clingcon on some CASP benchmarks. 

**Abstract (ZH)**: 我们提出了一个基于约束回答集编程(CASP)的求解器asp-fzn，该求解器扩展了ASP以包含线性约束。我们的方法基于将CASP程序转换为支持多种Constraint Programming和Integer Programming后端求解器的求解器独立的FlatZinc语言。asp-fzn支持丰富的线性约束语言，包括一些常见的全局约束。在评估方面，我们展示了asp-fzn在过去的ASP竞赛基准上的竞争力与最先进的ASP求解器持平。此外，我们还将其应用于文献中的几个CASP问题，并将其性能与clingcon进行比较，clingcon是支持asp-fzn语言大部分特性的 prominant CASP求解器。asp-fzn的性能非常有前景，已经与原ASP相当，并在某些CASP基准上优于clingcon。 

---
# MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines 

**Title (ZH)**: MetaAgent：基于有限状态机自动构建多代理系统 

**Authors**: Yaolun Zhang, Xiaogeng Liu, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22606)  

**Abstract**: Large Language Models (LLMs) have demonstrated the ability to solve a wide range of practical tasks within multi-agent systems. However, existing human-designed multi-agent frameworks are typically limited to a small set of pre-defined scenarios, while current automated design methods suffer from several limitations, such as the lack of tool integration, dependence on external training data, and rigid communication structures. In this paper, we propose MetaAgent, a finite state machine based framework that can automatically generate a multi-agent system. Given a task description, MetaAgent will design a multi-agent system and polish it through an optimization algorithm. When the multi-agent system is deployed, the finite state machine will control the agent's actions and the state transitions. To evaluate our framework, we conduct experiments on both text-based tasks and practical tasks. The results indicate that the generated multi-agent system surpasses other auto-designed methods and can achieve a comparable performance with the human-designed multi-agent system, which is optimized for those specific tasks. 

**Abstract (ZH)**: 基于有限状态机的MetaAgent多智能体系统自动生成框架 

---
# Collaborative Medical Triage under Uncertainty: A Multi-Agent Dynamic Matching Approach 

**Title (ZH)**: 不确定情境下的协作医疗分诊：多代理动态匹配方法 

**Authors**: Hongyan Cheng, Chengzhang Yu, Yanshu Shi, Chiyue Wang, Cong Liu, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22504)  

**Abstract**: The post-pandemic surge in healthcare demand, coupled with critical nursing shortages, has placed unprecedented pressure on emergency department triage systems, necessitating innovative AI-driven solutions. We present a multi-agent interactive intelligent system for medical triage that addresses three fundamental challenges in current AI-based triage systems: insufficient medical specialization leading to hallucination-induced misclassifications, heterogeneous department structures across healthcare institutions, and inefficient detail-oriented questioning that impedes rapid triage decisions. Our system employs three specialized agents - RecipientAgent, InquirerAgent, and DepartmentAgent - that collaborate through structured inquiry mechanisms and department-specific guidance rules to transform unstructured patient symptoms into accurate department recommendations. To ensure robust evaluation, we constructed a comprehensive Chinese medical triage dataset from a medical website, comprising 3,360 real-world cases spanning 9 primary departments and 62 secondary departments. Through systematic data imputation using large language models, we address the prevalent issue of incomplete medical records in real-world data. Experimental results demonstrate that our multi-agent system achieves 89.2% accuracy in primary department classification and 73.9% accuracy in secondary department classification after four rounds of patient interaction. The system's pattern-matching-based guidance mechanisms enable efficient adaptation to diverse hospital configurations while maintaining high triage accuracy. Our work provides a scalable framework for deploying AI-assisted triage systems that can accommodate the organizational heterogeneity of healthcare institutions while ensuring clinically sound decision-making. 

**Abstract (ZH)**: 新冠疫情后医疗需求激增，加之护理人员短缺，给急诊分诊系统带来了前所未有的压力， necessitating 创新的AI驱动解决方案。我们提出了一种多智能体交互智能医疗分诊系统，以应对当前AI分诊系统中的三个基本挑战：医学专长不足导致的幻觉错误分类、医疗机构之间的异质部门结构以及低效的细节询问导致的分诊决策延迟。该系统采用了三种专门智能体——受试者智能体、询问者智能体和部门智能体，通过结构化的询问机制和部门特定的指导规则，将患者的非结构化症状转化为准确的部门推荐。为确保稳健的评估，我们从医疗网站构建了一个全面的中文医疗分诊数据集，包含3,360个真实世界病例，涵盖9个主要部门和62个次要部门。通过使用大规模语言模型进行系统的数据插补，我们解决了真实世界数据中普遍存在的医疗记录不完整问题。实验结果表明，经过四轮患者互动后，该多智能体系统在主要部门分类中的准确率为89.2%，在次要部门分类中的准确率为73.9%。基于模式匹配的指导机制使系统能够高效适应不同的医院配置，并保持高分诊准确性。我们的研究为部署能够在保障临床决策准确性的同时适应医疗机构组织差异的AI辅助分诊系统提供了可扩展的框架。 

---
# Nearest-Better Network for Visualizing and Analyzing Combinatorial Optimization Problems: A Unified Tool 

**Title (ZH)**: 用于可视化和分析组合优化问题的nearest-better网络：一种统一工具 

**Authors**: Yiya Diao, Changhe Li, Sanyou Zeng, Xinye Cai, Wenjian Luo, Shengxiang Yang, Carlos A. Coello Coello  

**Link**: [PDF](https://arxiv.org/pdf/2507.22440)  

**Abstract**: The Nearest-Better Network (NBN) is a powerful method to visualize sampled data for continuous optimization problems while preserving multiple landscape features. However, the calculation of NBN is very time-consuming, and the extension of the method to combinatorial optimization problems is challenging but very important for analyzing the algorithm's behavior. This paper provides a straightforward theoretical derivation showing that the NBN network essentially functions as the maximum probability transition network for algorithms. This paper also presents an efficient NBN computation method with logarithmic linear time complexity to address the time-consuming issue. By applying this efficient NBN algorithm to the OneMax problem and the Traveling Salesman Problem (TSP), we have made several remarkable discoveries for the first time: The fitness landscape of OneMax exhibits neutrality, ruggedness, and modality features. The primary challenges of TSP problems are ruggedness, modality, and deception. Two state-of-the-art TSP algorithms (i.e., EAX and LKH) have limitations when addressing challenges related to modality and deception, respectively. LKH, based on local search operators, fails when there are deceptive solutions near global optima. EAX, which is based on a single population, can efficiently maintain diversity. However, when multiple attraction basins exist, EAX retains individuals within multiple basins simultaneously, reducing inter-basin interaction efficiency and leading to algorithm's stagnation. 

**Abstract (ZH)**: 基于概率转移的最近最佳网络在连续优化问题中可视化样本数据并保留多景致特征的方法及其高效计算 

---
# Cross-Border Legal Adaptation of Autonomous Vehicle Design based on Logic and Non-monotonic Reasoning 

**Title (ZH)**: 基于逻辑与非单调推理的自动驾驶车辆跨境法律适应设计 

**Authors**: Zhe Yu, Yiwei Lu, Burkhard Schafer, Zhe Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22432)  

**Abstract**: This paper focuses on the legal compliance challenges of autonomous vehicles in a transnational context. We choose the perspective of designers and try to provide supporting legal reasoning in the design process. Based on argumentation theory, we introduce a logic to represent the basic properties of argument-based practical (normative) reasoning, combined with partial order sets of natural numbers to express priority. Finally, through case analysis of legal texts, we show how the reasoning system we provide can help designers to adapt their design solutions more flexibly in the cross-border application of autonomous vehicles and to more easily understand the legal implications of their decisions. 

**Abstract (ZH)**: 本文关注跨国背景下自动驾驶车辆的法律合规挑战，并从设计者的视角出发，尝试在设计过程中提供支持性的法律推理。基于论证理论，我们引入了一种逻辑来表示基于论证的实际（规范性）推理的基本属性，并结合自然数的部分序集来表达优先级。最后，通过法律文本案例分析，我们展示了所提供的推理系统如何帮助设计者在跨境应用自动驾驶车辆时更灵活地调整设计解决方案，并更易于理解其决策的法律含义。 

---
# On the Definition of Intelligence 

**Title (ZH)**: 关于智能的定义 

**Authors**: Kei-Sing Ng  

**Link**: [PDF](https://arxiv.org/pdf/2507.22423)  

**Abstract**: To engineer AGI, we should first capture the essence of intelligence in a species-agnostic form that can be evaluated, while being sufficiently general to encompass diverse paradigms of intelligent behavior, including reinforcement learning, generative models, classification, analogical reasoning, and goal-directed decision-making. We propose a general criterion based on sample fidelity: intelligence is the ability, given sample(s) from a category, to generate sample(s) from the same category. We formalise this intuition as {\epsilon}-category intelligence: it is {\epsilon}-intelligent with respect to a category if no chosen admissible distinguisher can separate generated from original samples beyond tolerance {\epsilon}. We present the formal framework, outline empirical protocols, and discuss implications for evaluation, safety, and generalization. 

**Abstract (ZH)**: 为了工程化AGI，我们应当首先以一种物种无关的形式捕获智能的本质，这种形式可以被评估并且足够通用以涵盖多样化的智能行为范式，包括强化学习、生成模型、分类、类比推理和目标导向决策。我们提出了一种基于样本保真度的一般标准：智能是在给定某一类别的样本后，生成同一类别的样本的能力。我们将这种直觉形式化为ε-类别智能：如果不存在可选的可接受区分器能够将生成样本与原始样本区分开超出容差ε，则该实体对某一类别而言是ε-智能的。我们呈现了该形式化框架、概述了实证协议，并讨论了其对评估、安全性和泛化的含义。 

---
# Beyond Accuracy: How AI Metacognitive Sensitivity improves AI-assisted Decision Making 

**Title (ZH)**: 超越准确性：AI元认知敏感性提升AI辅助决策效果 

**Authors**: ZhaoBin Li, Mark Steyvers  

**Link**: [PDF](https://arxiv.org/pdf/2507.22365)  

**Abstract**: In settings where human decision-making relies on AI input, both the predictive accuracy of the AI system and the reliability of its confidence estimates influence decision quality. We highlight the role of AI metacognitive sensitivity -- its ability to assign confidence scores that accurately distinguish correct from incorrect predictions -- and introduce a theoretical framework for assessing the joint impact of AI's predictive accuracy and metacognitive sensitivity in hybrid decision-making settings. Our analysis identifies conditions under which an AI with lower predictive accuracy but higher metacognitive sensitivity can enhance the overall accuracy of human decision making. Finally, a behavioral experiment confirms that greater AI metacognitive sensitivity improves human decision performance. Together, these findings underscore the importance of evaluating AI assistance not only by accuracy but also by metacognitive sensitivity, and of optimizing both to achieve superior decision outcomes. 

**Abstract (ZH)**: 在人类决策依赖于AI输入的环境中，AI系统的预测准确性及其信心估计的可靠性都影响决策质量。我们强调AI元认知敏感性的作用——其能够准确地为正确和错误预测分配信心评分的能力——并引入了一个理论框架来评估AI预测准确性和元认知敏感性在混合决策环境中联合影响。我们的分析识定了在某些条件下，尽管预测准确性较低但元认知敏感性较高的AI能够提升人类决策的整体准确性。最后，一个行为实验确认了更高的AI元认知敏感性能够改善人类的决策表现。这些发现强调了不仅通过准确性，还应通过元认知敏感性来评估AI辅助的重要性，并优化两者以实现更优异的决策结果。 

---
# Explainability Through Systematicity: The Hard Systematicity Challenge for Artificial Intelligence 

**Title (ZH)**: 通过系统性实现可解释性：人工智能面临的硬系统性挑战 

**Authors**: Matthieu Queloz  

**Link**: [PDF](https://arxiv.org/pdf/2507.22197)  

**Abstract**: This paper argues that explainability is only one facet of a broader ideal that shapes our expectations towards artificial intelligence (AI). Fundamentally, the issue is to what extent AI exhibits systematicity--not merely in being sensitive to how thoughts are composed of recombinable constituents, but in striving towards an integrated body of thought that is consistent, coherent, comprehensive, and parsimoniously principled. This richer conception of systematicity has been obscured by the long shadow of the "systematicity challenge" to connectionism, according to which network architectures are fundamentally at odds with what Fodor and colleagues termed "the systematicity of thought." I offer a conceptual framework for thinking about "the systematicity of thought" that distinguishes four senses of the phrase. I use these distinctions to defuse the perceived tension between systematicity and connectionism and show that the conception of systematicity that historically shaped our sense of what makes thought rational, authoritative, and scientific is more demanding than the Fodorian notion. To determine whether we have reason to hold AI models to this ideal of systematicity, I then argue, we must look to the rationales for systematization and explore to what extent they transfer to AI models. I identify five such rationales and apply them to AI. This brings into view the "hard systematicity challenge." However, the demand for systematization itself needs to be regulated by the rationales for systematization. This yields a dynamic understanding of the need to systematize thought, which tells us how systematic we need AI models to be and when. 

**Abstract (ZH)**: 本文argues that 可解释性只是更广泛的理想中的一个方面，这种理想塑造了我们对人工智能（AI）的期望。从根本上说，问题在于人工智能在多大程度上表现出系统性——不仅仅是对思想由可重组成分构成敏感，而是在追求一种连贯、一致、全面且简约的原则指导下的综合思想体系。这种更丰富的系统性概念被连结主义的“系统性挑战”的长时间阴影所掩盖，该挑战认为网络结构本质上与福多和同事所称的“思想的系统性”背道而驰。我提出了一个关于“思想的系统性”的概念框架，将其区分为四种含义。我利用这些区分来缓解系统性与连结主义之间的表面紧张关系，并表明历史上塑造我们对理性、权威性和科学性思维的理解的系统性概念比福多意义上的更为严格。然后，为了确定我们是否有理由要求AI模型达到这种系统性的理想，我论证我们需要考虑系统化的目的，并探讨它们在多大程度上适用于AI模型。我确定了五个这样的目的，并将它们应用于AI。这揭示了“硬系统性挑战”。然而，系统化本身的要求必须由系统化的目的来调节。这产生了一种动态的理解，关于我们需要AI模型达到多大的系统性及其何时实现。 

---
# A Bit of Freedom Goes a Long Way: Classical and Quantum Algorithms for Reinforcement Learning under a Generative Model 

**Title (ZH)**: 一点点自由空间效果显著：生成模型下经典与量子强化学习算法 

**Authors**: Andris Ambainis, Joao F. Doriguello, Debbie Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.22854)  

**Abstract**: We propose novel classical and quantum online algorithms for learning finite-horizon and infinite-horizon average-reward Markov Decision Processes (MDPs). Our algorithms are based on a hybrid exploration-generative reinforcement learning (RL) model wherein the agent can, from time to time, freely interact with the environment in a generative sampling fashion, i.e., by having access to a "simulator". By employing known classical and new quantum algorithms for approximating optimal policies under a generative model within our learning algorithms, we show that it is possible to avoid several paradigms from RL like "optimism in the face of uncertainty" and "posterior sampling" and instead compute and use optimal policies directly, which yields better regret bounds compared to previous works. For finite-horizon MDPs, our quantum algorithms obtain regret bounds which only depend logarithmically on the number of time steps $T$, thus breaking the $O(\sqrt{T})$ classical barrier. This matches the time dependence of the prior quantum works of Ganguly et al. (arXiv'23) and Zhong et al. (ICML'24), but with improved dependence on other parameters like state space size $S$ and action space size $A$. For infinite-horizon MDPs, our classical and quantum bounds still maintain the $O(\sqrt{T})$ dependence but with better $S$ and $A$ factors. Nonetheless, we propose a novel measure of regret for infinite-horizon MDPs with respect to which our quantum algorithms have $\operatorname{poly}\log{T}$ regret, exponentially better compared to classical algorithms. Finally, we generalise all of our results to compact state spaces. 

**Abstract (ZH)**: 我们提出了一种新颖的经典和量子在线算法，用于学习有限时期和无限时期的平均回报马尔可夫决策过程（MDP）。我们的算法基于一种混合探索-生成式强化学习（RL）模型，在这种模型中，智能体可以在不同时刻自由地以生成样本的方式与环境交互，即通过访问一个“模拟器”。通过在我们的学习算法中利用已知的经典和新量子算法来近似生成模型下的最优策略，我们展示了可以避免RL中的“面对不确定性乐观”和“后验采样”等范式，而是直接计算并使用最优策略，这相比于之前的工作提供了更好的遗憾界。对于有限时期MDP，我们的量子算法获得了遗憾界，仅依赖于时间步数$T$的对数，从而突破了经典工作中的$O(\sqrt{T})$障碍。这与Ganguly等人的（arXiv'23）和Zhong等人的（ICML'24）先前的量子工作的时间依赖性相匹配，但其他参数如状态空间大小$S$和动作空间大小$A$的依赖性有所改进。对于无限时期的MDP，我们的经典和量子界仍然保持$O(\sqrt{T})$的时间依赖性，但在$S$和$A$因子上有所改进。尽管如此，我们为无限时期的MDP提出了一种新的遗憾度量，根据这种度量，我们的量子算法具有$\operatorname{poly}\log{T}$遗憾界，相比经典算法具有指数级的改进。最后，我们将所有结果推广到紧致状态空间。 

---
# CapRecover: A Cross-Modality Feature Inversion Attack Framework on Vision Language Models 

**Title (ZH)**: CapRecover: 一种针对视觉语言模型的跨模态特征反转攻击框架 

**Authors**: Kedong Xiu, Saiqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22828)  

**Abstract**: As Vision-Language Models (VLMs) are increasingly deployed in split-DNN configurations--with visual encoders (e.g., ResNet, ViT) operating on user devices and sending intermediate features to the cloud--there is a growing privacy risk from semantic information leakage. Existing approaches to reconstructing images from these intermediate features often result in blurry, semantically ambiguous images. To directly address semantic leakage, we propose CapRecover, a cross-modality inversion framework that recovers high-level semantic content, such as labels or captions, directly from intermediate features without image reconstruction.
We evaluate CapRecover on multiple datasets and victim models, demonstrating strong performance in semantic recovery. Specifically, CapRecover achieves up to 92.71% Top-1 label accuracy on CIFAR-10 and generates fluent captions from ResNet50 features on COCO2017 with ROUGE-L scores up to 0.52. Our analysis further reveals that deeper convolutional layers encode significantly more semantic information compared to shallow layers. To mitigate semantic leakage, we introduce a simple yet effective protection method: adding random noise to intermediate features at each layer and removing the noise in the next layer. Experimental results show that this approach prevents semantic leakage without additional training costs. 

**Abstract (ZH)**: 视觉-语言模型在分布式DNN配置中的语义信息泄露防护：CapRecover跨模态反演框架 

---
# Empirical Evaluation of Concept Drift in ML-Based Android Malware Detection 

**Title (ZH)**: 基于概念偏移的机器学习驱动Android恶意软件检测的实证评估 

**Authors**: Ahmed Sabbah, Radi Jarrar, Samer Zein, David Mohaisen  

**Link**: [PDF](https://arxiv.org/pdf/2507.22772)  

**Abstract**: Despite outstanding results, machine learning-based Android malware detection models struggle with concept drift, where rapidly evolving malware characteristics degrade model effectiveness. This study examines the impact of concept drift on Android malware detection, evaluating two datasets and nine machine learning and deep learning algorithms, as well as Large Language Models (LLMs). Various feature types--static, dynamic, hybrid, semantic, and image-based--were considered. The results showed that concept drift is widespread and significantly affects model performance. Factors influencing the drift include feature types, data environments, and detection methods. Balancing algorithms helped with class imbalance but did not fully address concept drift, which primarily stems from the dynamic nature of the malware landscape. No strong link was found between the type of algorithm used and concept drift, the impact was relatively minor compared to other variables since hyperparameters were not fine-tuned, and the default algorithm configurations were used. While LLMs using few-shot learning demonstrated promising detection performance, they did not fully mitigate concept drift, highlighting the need for further investigation. 

**Abstract (ZH)**: 尽管取得了出色的结果，基于机器学习的Android恶意软件检测模型在概念漂移的问题上仍面临挑战，即迅速演变的恶意软件特征降低了模型的有效性。本研究考察了概念漂移对Android恶意软件检测的影响，评估了两个数据集和九种机器学习及深度学习算法，以及大型语言模型（LLMs）。考虑了多种特征类型，包括静态特征、动态特征、混合特征、语义特征和基于图像的特征。研究结果表明，概念漂移普遍存在且显著影响模型性能。影响漂移的因素包括特征类型、数据环境和检测方法。平衡算法有助于解决类别不平衡问题，但未能完全解决概念漂移，主要原因是恶意软件景观的动态性。未发现所使用算法类型与概念漂移之间存在强烈关联，与其他变量相比，其影响相对较小，因为未调整超参数，使用了默认的算法配置。虽然使用少量学习的大型语言模型在检测性能上表现出潜力，但未能完全缓解概念漂移，强调了进一步研究的必要性。 

---
# Teaching the Teacher: Improving Neural Network Distillability for Symbolic Regression via Jacobian Regularization 

**Title (ZH)**: 教学相长：通过雅可比正则化提高神经网络在符号回归中的可精炼性 

**Authors**: Soumyadeep Dhar, Kei Sen Fong, Mehul Motani  

**Link**: [PDF](https://arxiv.org/pdf/2507.22767)  

**Abstract**: Distilling large neural networks into simple, human-readable symbolic formulas is a promising path toward trustworthy and interpretable AI. However, this process is often brittle, as the complex functions learned by standard networks are poor targets for symbolic discovery, resulting in low-fidelity student models. In this work, we propose a novel training paradigm to address this challenge. Instead of passively distilling a pre-trained network, we introduce a \textbf{Jacobian-based regularizer} that actively encourages the ``teacher'' network to learn functions that are not only accurate but also inherently smoother and more amenable to distillation. We demonstrate through extensive experiments on a suite of real-world regression benchmarks that our method is highly effective. By optimizing the regularization strength for each problem, we improve the $R^2$ score of the final distilled symbolic model by an average of \textbf{120\% (relative)} compared to the standard distillation pipeline, all while maintaining the teacher's predictive accuracy. Our work presents a practical and principled method for significantly improving the fidelity of interpretable models extracted from complex neural networks. 

**Abstract (ZH)**: 将大型神经网络精简为简单可读的符号公式是实现可靠且可解释人工智能的一个有前景的路径。然而，这一过程通常是脆弱的，因为标准网络学到的复杂函数不适合作为符号发现的目标，导致低保真度的学生模型。在本文中，我们提出了一种新的训练范式来解决这一挑战。我们引入了一种基于雅可比矩阵的正则化器，以积极鼓励“老师”网络学习不仅准确而且天然更易于精简且更平滑的函数。通过在一系列实际回归基准上的 extensive 实验，我们证明了该方法非常有效。通过为每个问题优化正则化强度，我们最终得到的可解释模型的 $R^2$ 分数相对于标准精简管道平均提高了 \textbf{120\% (相对)}，同时保持了老师的预测准确性。我们的工作提供了一种实用且有原则的方法，以显著提高从复杂神经网络中提取的可解释模型的保真度。 

---
# Bayesian Optimization of Process Parameters of a Sensor-Based Sorting System using Gaussian Processes as Surrogate Models 

**Title (ZH)**: 基于高斯过程代理模型的传感器导向分拣系统工艺参数的贝叶斯优化 

**Authors**: Felix Kronenwett, Georg Maier, Thomas Laengle  

**Link**: [PDF](https://arxiv.org/pdf/2507.22766)  

**Abstract**: Sensor-based sorting systems enable the physical separation of a material stream into two fractions. The sorting decision is based on the image data evaluation of the sensors used and is carried out using actuators. Various process parameters must be set depending on the properties of the material stream, the dimensioning of the system, and the required sorting accuracy. However, continuous verification and re-adjustment are necessary due to changing requirements and material stream compositions. In this paper, we introduce an approach for optimizing, recurrently monitoring and adjusting the process parameters of a sensor-based sorting system. Based on Bayesian Optimization, Gaussian process regression models are used as surrogate models to achieve specific requirements for system behavior with the uncertainties contained therein. This method minimizes the number of necessary experiments while simultaneously considering two possible optimization targets based on the requirements for both material output streams. In addition, uncertainties are considered during determining sorting accuracies in the model calculation. We evaluated the method with three example process parameters. 

**Abstract (ZH)**: 基于传感器的分选系统过程参数的优化、递归监控与调整方法 

---
# Bifröst: Spatial Networking with Bigraphs 

**Title (ZH)**: Bifröst: 基于大图的空间网络技术 

**Authors**: Josh Millar, Ryan Gibb, Roy Ang, Anil Madhavapeddy, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22687)  

**Abstract**: Modern networked environments increasingly rely on spatial reasoning, but lack a coherent representation for coordinating physical space. Consequently, tasks such as enforcing spatial access policies remain fragile and manual. We first propose a unifying representation based on bigraphs, capturing spatial, social, and communication relationships within a single formalism, with user-facing tools to generate bigraphs from physical environments. Second, we present a hierarchical agent architecture for distributed spatial reasoning, with runtimes for agentic processes to interact the spatial representation, and a context-aware execution model that scopes reasoning to the smallest viable subspace. Together, these enable private, reliable, and low-latency spatial networking that can safely interact with agentic workflows. 

**Abstract (ZH)**: 现代网络环境越来越多地依赖于空间推理，但缺乏一种统一表示来协调物理空间。因此，诸如实施空间访问策略之类的任务仍然脆弱且需要人工干预。我们首先提出了一种基于 bigraphs 的统一表示法，能在单一形式主义中捕获空间、社会和通信关系，并提供面向用户的工具来从物理环境中生成 bigraphs。其次，我们提出了一个分层代理架构，用于分布式空间推理，并提供了代理过程的运行时以交互空间表示，并提供一种情境感知执行模型，将推理范围限定在最小可行子空间。这些共同努力实现了私有、可靠和低延迟的空间网络，可以安全地与代理工作流交互。 

---
# Designing for Self-Regulation in Informal Programming Learning: Insights from a Storytelling-Centric Approach 

**Title (ZH)**: 面向自我调节的非正式编程学习设计：一种以叙事为中心的方法的启示 

**Authors**: Sami Saeed Alghamdi, Christopher Bull, Ahmed Kharrufa  

**Link**: [PDF](https://arxiv.org/pdf/2507.22671)  

**Abstract**: Many people learn programming independently from online resources and often report struggles in achieving their personal learning goals. Learners frequently describe their experiences as isolating and frustrating, challenged by abundant uncertainties, information overload, and distraction, compounded by limited guidance. At the same time, social media serves as a personal space where many engage in diverse self-regulation practices, including help-seeking, using external memory aids (e.g., self-notes), self-reflection, emotion regulation, and self-motivation. For instance, learners often mark achievements and set milestones through their posts. In response, we developed a system consisting of a web platform and browser extensions to support self-regulation online. The design aims to add learner-defined structure to otherwise unstructured experiences and bring meaning to curation and reflection activities by translating them into learning stories with AI-generated feedback. We position storytelling as an integrative approach to design that connects resource curation, reflective and sensemaking practice, and narrative practices learners already use across social platforms. We recruited 15 informal programming learners who are regular social media users to engage with the system in a self-paced manner; participation concluded upon submitting a learning story and survey. We used three quantitative scales and a qualitative survey to examine users' characteristics and perceptions of the system's support for their self-regulation. User feedback suggests the system's viability as a self-regulation aid. Learners particularly valued in-situ reflection, automated story feedback, and video annotation, while other features received mixed views. We highlight perceived benefits, friction points, and design opportunities for future AI-augmented self-regulation tools. 

**Abstract (ZH)**: 许多学习者借助在线资源独立学习编程，往往难以实现个人学习目标。学习者经常描述他们的体验是孤立和令人沮丧的，受到大量不确定性、信息过载和分心的挑战，并且缺乏指导。与此同时，社交媒体作为一个个人空间，许多学习者在此参与多种自我调节实践，包括寻求帮助、使用外部记忆辅助工具（例如自我笔记）、自我反思、情绪调节和自我激励。例如，学习者经常通过他们的帖子标记成就并设定里程碑。为应对这一情况，我们开发了一个由网页平台和浏览器扩展组成的系统，以支持在线自我调节。设计旨在为原本无结构的体验增添学习者自定义的结构，并将整理和反思活动转化为AI生成反馈的学习故事。我们定位叙事作为一种集成设计方法，将资源整理、反思与推理实践以及学习者在社交媒体平台上已使用的故事叙述实践联系起来。我们招募了15名定期使用社交媒体的非正式编程学习者，以自主节奏参与该系统；参与活动在提交学习故事和调查问卷后结束。我们使用了三种定量量表和一次定性调查来考察用户特征以及他们对系统支持自我调节的感知。用户反馈表明该系统作为自我调节辅助工具的可行性。学习者特别重视现场反思、自动化故事反馈和视频标注，而其他功能则引起了不同的看法。我们强调了感知到的优势、摩擦点和未来增强自我调节工具的设计机会。 

---
# H2Tune: Federated Foundation Model Fine-Tuning with Hybrid Heterogeneity 

**Title (ZH)**: H2Tune: 嵌合异质性联邦基础模型微调 

**Authors**: Wei Guo, Siyuan Lu, Yiqi Tong, Zhaojun Hu, Fuzhen Zhuang, Xiao Zhang, Tao Fan, Jin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22633)  

**Abstract**: Different from existing federated fine-tuning (FFT) methods for foundation models, hybrid heterogeneous federated fine-tuning (HHFFT) is an under-explored scenario where clients exhibit double heterogeneity in model architectures and downstream tasks. This hybrid heterogeneity introduces two significant challenges: 1) heterogeneous matrix aggregation, where clients adopt different large-scale foundation models based on their task requirements and resource limitations, leading to dimensional mismatches during LoRA parameter aggregation; and 2) multi-task knowledge interference, where local shared parameters, trained with both task-shared and task-specific knowledge, cannot ensure only task-shared knowledge is transferred between clients. To address these challenges, we propose H2Tune, a federated foundation model fine-tuning with hybrid heterogeneity. Our framework H2Tune consists of three key components: (i) sparsified triple matrix decomposition to align hidden dimensions across clients through constructing rank-consistent middle matrices, with adaptive sparsification based on client resources; (ii) relation-guided matrix layer alignment to handle heterogeneous layer structures and representation capabilities; and (iii) alternating task-knowledge disentanglement mechanism to decouple shared and specific knowledge of local model parameters through alternating optimization. Theoretical analysis proves a convergence rate of O(1/\sqrt{T}). Extensive experiments show our method achieves up to 15.4% accuracy improvement compared to state-of-the-art baselines. Our code is available at this https URL. 

**Abstract (ZH)**: 不同于现有基础模型的联邦微调方法，混合异构联邦微调(HHFFT)是一种尚待探索的场景，其中客户端在模型架构和下游任务上表现出双重异构性。这种混合异构性引入了两个显著挑战：1）异构矩阵聚合，客户端根据任务需求和资源限制采用不同的大规模基础模型，导致LoRA参数聚合时出现维度不匹配；2）多任务知识干扰，本地共享参数在同时接受任务共享和任务特定知识的训练下，不能确保只在客户端之间转移任务共享知识。为了解决这些挑战，我们提出了一种混合异构基础模型联邦微调方法H2Tune。我们的框架H2Tune包括三个关键组件：(i)稀疏化三重矩阵分解，通过构建秩一致的中间矩阵来对齐客户端的隐藏维度，基于客户端资源进行自适应稀疏化；(ii) 关系导向的矩阵层对齐，处理异构层结构和表示能力；(iii) 交替任务知识解耦机制，通过交替优化来解耦本地模型参数中的共享和特定知识。理论分析证明了收敛速率为O(1/\sqrt{T})。广泛实验表明，与最新的基线方法相比，我们的方法可实现多达15.4%的准确性改进。我们的代码可在以下链接获取：this https URL。 

---
# Adaptive Duration Model for Text Speech Alignment 

**Title (ZH)**: 文本语音对齐的自适应时长模型 

**Authors**: Junjie Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22612)  

**Abstract**: Speech-to-text alignment is a critical component of neural text to-speech (TTS) models. Autoregressive TTS models typically use an attention mechanism to learn these alignments on-line. However, these alignments tend to be brittle and often fail to generalize to long utterances and out-of-domain text, leading to missing or repeating words. Most non-autoregressive end to-end TTS models rely on durations extracted from external sources, using additional duration models for alignment. In this paper, we propose a novel duration prediction framework that can give compromising phoneme-level duration distribution with given text. In our experiments, the proposed duration model has more precise prediction and condition adaptation ability compared to previous baseline models. Numerically, it has roughly a 11.3 percents immprovement on alignment accuracy, and makes the performance of zero-shot TTS models more robust to the mismatch between prompt audio and input audio. 

**Abstract (ZH)**: 语音到文本对齐是神经文本到语音（TTS）模型的关键组成部分。自回归TTS模型通常使用注意机制在线学习这些对齐。然而，这些对齐很容易变得脆弱，往往无法泛化到长语音和域外文本，导致缺失或重复单词。大多数非自回归端到端TTS模型依赖于从外部源提取的时长，并使用额外的时间长度模型进行对齐。在本文中，我们提出了一种新颖的时长预测框架，可以在给定文本的情况下提供折中的音素级时长分布。在我们的实验中，所提出的时长模型在预测精度和条件适应能力方面优于以前的基线模型。数值上，它在对齐精度上提升了约11.3%，并使零-shot TTS模型在提示音频与输入音频不匹配时的性能更加 robust。 

---
# A Mean-Field Theory of $Θ$-Expectations 

**Title (ZH)**: Θ-期望的均场理论 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22577)  

**Abstract**: The canonical theory of sublinear expectations, a foundation of stochastic calculus under ambiguity, is insensitive to the non-convex geometry of primitive uncertainty models. This paper develops a new stochastic calculus for a structured class of such non-convex models. We introduce a class of fully coupled Mean-Field Forward-Backward Stochastic Differential Equations where the BSDE driver is defined by a pointwise maximization over a law-dependent, non-convex set. Mathematical tractability is achieved via a uniform strong concavity assumption on the driver with respect to the control variable, which ensures the optimization admits a unique and stable solution. A central contribution is to establish the Lipschitz stability of this optimizer from primitive geometric and regularity conditions, which underpins the entire well-posedness theory. We prove local and global well-posedness theorems for the FBSDE system. The resulting valuation functional, the $\Theta$-Expectation, is shown to be dynamically consistent and, most critically, to violate the axiom of sub-additivity. This, along with its failure to be translation invariant, demonstrates its fundamental departure from the convex paradigm. This work provides a rigorous foundation for stochastic calculus under a class of non-convex, endogenous ambiguity. 

**Abstract (ZH)**: 非凸原始不确定性模型下的典范次线性期望的算子基础与发展：一类耦合向前-向后随机微分方程的随机算子理论 

---
# COOkeD: Ensemble-based OOD detection in the era of zero-shot CLIP 

**Title (ZH)**: COOkeD: 零样本CLIP时代基于集成的OOD检测 

**Authors**: Galadrielle Humblot-Renaux, Gianni Franchi, Sergio Escalera, Thomas B. Moeslund  

**Link**: [PDF](https://arxiv.org/pdf/2507.22576)  

**Abstract**: Out-of-distribution (OOD) detection is an important building block in trustworthy image recognition systems as unknown classes may arise at test-time. OOD detection methods typically revolve around a single classifier, leading to a split in the research field between the classical supervised setting (e.g. ResNet18 classifier trained on CIFAR100) vs. the zero-shot setting (class names fed as prompts to CLIP). In both cases, an overarching challenge is that the OOD detection performance is implicitly constrained by the classifier's capabilities on in-distribution (ID) data. In this work, we show that given a little open-mindedness from both ends, remarkable OOD detection can be achieved by instead creating a heterogeneous ensemble - COOkeD combines the predictions of a closed-world classifier trained end-to-end on a specific dataset, a zero-shot CLIP classifier, and a linear probe classifier trained on CLIP image features. While bulky at first sight, this approach is modular, post-hoc and leverages the availability of pre-trained VLMs, thus introduces little overhead compared to training a single standard classifier. We evaluate COOkeD on popular CIFAR100 and ImageNet benchmarks, but also consider more challenging, realistic settings ranging from training-time label noise, to test-time covariate shift, to zero-shot shift which has been previously overlooked. Despite its simplicity, COOkeD achieves state-of-the-art performance and greater robustness compared to both classical and CLIP-based OOD detection methods. Code is available at this https URL 

**Abstract (ZH)**: 离分布（OOD）检测是可信赖图像识别系统的重要组成部分，因为未知类别可能在测试时出现。OOD检测方法通常围绕单一分类器展开，导致研究领域在经典的监督设置（例如，使用CIFAR100训练的ResNet18分类器）与零样本设置（类名称作为提示输入给CLIP）之间分裂。在上述两种情况下，一个普遍的挑战是，OOD检测性能隐式受限于分类器在分布内（ID）数据上的能力。在本文中，我们展示了一种方法，通过在两端稍显开放的态度，COOkeD通过结合特定数据集上端到端训练的封闭世界分类器、零样本CLIP分类器和基于CLIP图像特征的线性探查分类器的预测，实现了显著的OOD检测效果。尽管乍一看方法较为复杂，但这种方法是模块化的、后处理的，并利用预训练的VLMs的可用性，因此与训练单一标准分类器相比，引入的额外开销很小。我们在流行的CIFAR100和ImageNet基准上评估了COOkeD，同时还考虑了包括训练时标签噪声、测试时协变量偏移以及以往被忽视的零样本偏移在内的更具挑战性和现实性的场景。尽管方法相对简单，COOkeD仍然实现了最先进的性能和更强的鲁棒性，相比传统和基于CLIP的OOD检测方法。代码可在此链接获取。 

---
# Explaining Deep Network Classification of Matrices: A Case Study on Monotonicity 

**Title (ZH)**: 矩阵深度网络分类的解释：关于单调性的一项研究 

**Authors**: Leandro Farina, Sergey Korotov  

**Link**: [PDF](https://arxiv.org/pdf/2507.22570)  

**Abstract**: This work demonstrates a methodology for using deep learning to discover simple, practical criteria for classifying matrices based on abstract algebraic properties. By combining a high-performance neural network with explainable AI (XAI) techniques, we can distill a model's learned strategy into human-interpretable rules. We apply this approach to the challenging case of monotone matrices, defined by the condition that their inverses are entrywise nonnegative. Despite their simple definition, an easy characterization in terms of the matrix elements or the derived parameters is not known. Here, we present, to the best of our knowledge, the first systematic machine-learning approach for deriving a practical criterion that distinguishes monotone from non-monotone matrices. After establishing a labelled dataset by randomly generated monotone and non-monotone matrices uniformly on $(-1,1)$, we employ deep neural network algorithms for classifying the matrices as monotone or non-monotone, using both their entries and a comprehensive set of matrix features. By saliency methods, such as integrated gradients, we identify among all features, two matrix parameters which alone provide sufficient information for the matrix classification, with $95\%$ accuracy, namely the absolute values of the two lowest-order coefficients, $c_0$ and $c_1$ of the matrix's characteristic polynomial. A data-driven study of 18,000 random $7\times7$ matrices shows that the monotone class obeys $\lvert c_{0}/c_{1}\rvert\le0.18$ with probability $>99.98\%$; because $\lvert c_{0}/c_{1}\rvert = 1/\mathrm{tr}(A^{-1})$ for monotone $A$, this is equivalent to the simple bound $\mathrm{tr}(A^{-1})\ge5.7$. 

**Abstract (ZH)**: 本研究展示了一种使用深度学习发现基于抽象代数性质分类矩阵的简单实用标准的方法。通过将高性能神经网络与可解释人工智能(XAI)技术结合，我们可以将模型学习到的策略提炼为人可以理解的规则。我们将这种方法应用于单调矩阵这一具有挑战性的案例，单调矩阵的定义是其逆矩阵的每一个元素均为非负。尽管它们的定义很简单，但以矩阵元素或导出参数的简单形式进行描述是未知的。这里，我们据我们所知，首次系统地提出了一个基于机器学习的方法来推导区分单调矩阵和非单调矩阵的实际判据。通过随机生成均匀分布在(-1,1)区间上的单调和非单调矩阵建立标注数据集，我们运用深度神经网络算法根据矩阵的元素及其一系列综合特征将其分类为单调或非单调矩阵。通过显著性方法，如整合梯度，我们发现，在所有特征中，仅有两个矩阵参数足够提供95%的矩阵分类准确性，即矩阵特征多项式的次低阶系数的绝对值|c_0|和|c_1|。通过对18,000个随机生成的7×7矩阵的数据驱动研究发现，单调类满足|c_0/c_1|≤0.18的概率超过99.98%；由于对于单调矩阵A，|c_0/c_1| = 1/Tr(A^-1)，这等价于简单的界限Tr(A^-1)≥5.7。 

---
# RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning 

**Title (ZH)**: RainbowPrompt: 提高持续学习中提示演化的多样性 

**Authors**: Kiseong Hong, Gyeong-hyeon Kim, Eunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.22553)  

**Abstract**: Prompt-based continual learning provides a rehearsal-free solution by tuning small sets of parameters while keeping pre-trained models frozen. To meet the complex demands of sequential tasks, it is crucial to integrate task-specific knowledge within prompts effectively. However, existing works rely on either fixed learned prompts (i.e., prompts whose representations remain unchanged during new task learning) or on prompts generated from an entangled task-shared space, limiting the representational diversity of the integrated prompt. To address this issue, we propose a novel prompt-evolving mechanism to adaptively aggregate base prompts (i.e., task-specific prompts) into a unified prompt while ensuring diversity. By transforming and aligning base prompts, both previously learned and newly introduced, our approach continuously evolves accumulated knowledge to facilitate learning new tasks. We further introduce a learnable probabilistic gate that adaptively determines which layers to activate during the evolution process. We validate our method on image classification and video action recognition tasks in class-incremental learning, achieving average gains of 9.07% and 7.40% over existing methods across all scenarios. 

**Abstract (ZH)**: 基于提示的持续学习通过微调少量参数并保持预训练模型冻结，提供了一种无重温的解决方案。为了满足序列任务复杂的需求，有效地在提示中融合任务特定知识至关重要。然而，现有工作依赖于固定学习提示（即，在新的任务学习过程中提示表示保持不变）或从纠缠的任务共享空间生成提示，这限制了集成提示的表示多样性。为解决该问题，我们提出了一种新颖的提示演化机制，以自适应地将基础提示（即，任务特定提示）聚合为一个统一提示，同时确保多样性。通过变换和对齐基础提示（包括先前学习的和新引入的），我们的方法在不断演化的累积知识基础上促进新任务的学习。我们还引入了一个可学习的概率门控，以自适应地确定在演化过程中激活哪些层。我们在类别增量学习中的图像分类和视频动作识别任务上验证了该方法，实现了平均分别比现有方法高出9.07%和7.40%的性能提升。 

---
# A surrogate model for topology optimisation of elastic structures via parametric autoencoders 

**Title (ZH)**: 基于参数自编码器的弹性结构拓扑优化代理模型 

**Authors**: Matteo Giacomini, Antonio Huerta  

**Link**: [PDF](https://arxiv.org/pdf/2507.22539)  

**Abstract**: A surrogate-based topology optimisation algorithm for linear elastic structures under parametric loads and boundary conditions is proposed. Instead of learning the parametric solution of the state (and adjoint) problems or the optimisation trajectory as a function of the iterations, the proposed approach devises a surrogate version of the entire optimisation pipeline. First, the method predicts a quasi-optimal topology for a given problem configuration as a surrogate model of high-fidelity topologies optimised with the homogenisation method. This is achieved by means of a feed-forward net learning the mapping between the input parameters characterising the system setup and a latent space determined by encoder/decoder blocks reducing the dimensionality of the parametric topology optimisation problem and reconstructing a high-dimensional representation of the topology. Then, the predicted topology is used as an educated initial guess for a computationally efficient algorithm penalising the intermediate values of the design variable, while enforcing the governing equations of the system. This step allows the method to correct potential errors introduced by the surrogate model, eliminate artifacts, and refine the design in order to produce topologies consistent with the underlying physics. Different architectures are proposed and the approximation and generalisation capabilities of the resulting models are numerically evaluated. The quasi-optimal topologies allow to outperform the high-fidelity optimiser by reducing the average number of optimisation iterations by $53\%$ while achieving discrepancies below $4\%$ in the optimal value of the objective functional, even in the challenging scenario of testing the model to extrapolate beyond the training and validation domain. 

**Abstract (ZH)**: 基于代理模型的参数化载荷和边界条件下线性弹性结构的拓扑优化算法 

---
# Accident-Driven Congestion Prediction and Simulation: An Explainable Framework Using Advanced Clustering and Bayesian Networks 

**Title (ZH)**: 基于事故驱动的拥堵预测与仿真：一种基于高级聚类和贝叶斯网络的可解释框架 

**Authors**: Kranthi Kumar Talluri, Galia Weidl, Vaishnavi Kasuluru  

**Link**: [PDF](https://arxiv.org/pdf/2507.22529)  

**Abstract**: Traffic congestion due to uncertainties, such as accidents, is a significant issue in urban areas, as the ripple effect of accidents causes longer delays, increased emissions, and safety concerns. To address this issue, we propose a robust framework for predicting the impact of accidents on congestion. We implement Automated Machine Learning (AutoML)-enhanced Deep Embedding Clustering (DEC) to assign congestion labels to accident data and predict congestion probability using a Bayesian Network (BN). The Simulation of Urban Mobility (SUMO) simulation is utilized to evaluate the correctness of BN predictions using evidence-based scenarios. Results demonstrate that the AutoML-enhanced DEC has outperformed traditional clustering approaches. The performance of the proposed BN model achieved an overall accuracy of 95.6%, indicating its ability to understand the complex relationship of accidents causing congestion. Validation in SUMO with evidence-based scenarios demonstrated that the BN model's prediction of congestion states closely matches those of SUMO, indicating the high reliability of the proposed BN model in ensuring smooth urban mobility. 

**Abstract (ZH)**: 由于事故发生等不确定性因素导致的交通拥堵是城市区域中的一个重大问题，事故的连锁反应会导致更长时间的延误、更高的排放和安全问题。为应对这一问题，我们提出了一种鲁棒性的框架来预测事故对拥堵的影响。我们采用了增强的自动机器学习（AutoML）嵌入式聚类（DEC）方法对事故数据进行聚类并标记拥堵标签，使用贝叶斯网络（BN）预测拥堵概率。我们利用城市移动性仿真（SUMO）仿真来通过基于证据的场景评估BN预测的准确性。结果表明，增强的DEC在性能上优于传统的聚类方法。所提出的BN模型的整体准确率为95.6%，表明其能够理解事故引起拥堵的复杂关系。SUMO仿真中的验证结果表明，基于证据的场景下，所提出的BN模型对拥堵状态的预测与SUMO仿真结果高度一致，证明了该模型在保障城市交通顺畅方面的高度可靠性。 

---
# LoReUn: Data Itself Implicitly Provides Cues to Improve Machine Unlearning 

**Title (ZH)**: LoReUn: 数据本身隐含地提供改善机器卸载的线索 

**Authors**: Xiang Li, Qianli Shen, Haonan Wang, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22499)  

**Abstract**: Recent generative models face significant risks of producing harmful content, which has underscored the importance of machine unlearning (MU) as a critical technique for eliminating the influence of undesired data. However, existing MU methods typically assign the same weight to all data to be forgotten, which makes it difficult to effectively forget certain data that is harder to unlearn than others. In this paper, we empirically demonstrate that the loss of data itself can implicitly reflect its varying difficulty. Building on this insight, we introduce Loss-based Reweighting Unlearning (LoReUn), a simple yet effective plug-and-play strategy that dynamically reweights data during the unlearning process with minimal additional computational overhead. Our approach significantly reduces the gap between existing MU methods and exact unlearning in both image classification and generation tasks, effectively enhancing the prevention of harmful content generation in text-to-image diffusion models. 

**Abstract (ZH)**: 基于损失的重新加权去学习（LoReUn）：在图像分类和生成任务中减少机器去学习差距以增强有害内容生成预防 

---
# LVM-GP: Uncertainty-Aware PDE Solver via coupling latent variable model and Gaussian process 

**Title (ZH)**: LVM-GP：结合潜在变量模型和高斯过程的不确定性aware偏微分方程求解器 

**Authors**: Xiaodong Feng, Ling Guo, Xiaoliang Wan, Hao Wu, Tao Zhou, Wenwen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22493)  

**Abstract**: We propose a novel probabilistic framework, termed LVM-GP, for uncertainty quantification in solving forward and inverse partial differential equations (PDEs) with noisy data. The core idea is to construct a stochastic mapping from the input to a high-dimensional latent representation, enabling uncertainty-aware prediction of the solution. Specifically, the architecture consists of a confidence-aware encoder and a probabilistic decoder. The encoder implements a high-dimensional latent variable model based on a Gaussian process (LVM-GP), where the latent representation is constructed by interpolating between a learnable deterministic feature and a Gaussian process prior, with the interpolation strength adaptively controlled by a confidence function learned from data. The decoder defines a conditional Gaussian distribution over the solution field, where the mean is predicted by a neural operator applied to the latent representation, allowing the model to learn flexible function-to-function mapping. Moreover, physical laws are enforced as soft constraints in the loss function to ensure consistency with the underlying PDE structure. Compared to existing approaches such as Bayesian physics-informed neural networks (B-PINNs) and deep ensembles, the proposed framework can efficiently capture functional dependencies via merging a latent Gaussian process and neural operator, resulting in competitive predictive accuracy and robust uncertainty quantification. Numerical experiments demonstrate the effectiveness and reliability of the method. 

**Abstract (ZH)**: 一种用于解决含噪数据前向和逆向偏微分方程不确定性量化的新颖概率框架：LVM-GP 

---
# Proto-EVFL: Enhanced Vertical Federated Learning via Dual Prototype with Extremely Unaligned Data 

**Title (ZH)**: Proto-EVFL: 增强型垂直联邦学习via双重原型在极不對齐数据下的应用 

**Authors**: Wei Guo, Yiyang Duan, Zhaojun Hu, Yiqi Tong, Fuzhen Zhuang, Xiao Zhang, Jin Dong, Ruofan Wu, Tengfei Liu, Yifan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.22488)  

**Abstract**: In vertical federated learning (VFL), multiple enterprises address aligned sample scarcity by leveraging massive locally unaligned samples to facilitate collaborative learning. However, unaligned samples across different parties in VFL can be extremely class-imbalanced, leading to insufficient feature representation and limited model prediction space. Specifically, class-imbalanced problems consist of intra-party class imbalance and inter-party class imbalance, which can further cause local model bias and feature contribution inconsistency issues, respectively. To address the above challenges, we propose Proto-EVFL, an enhanced VFL framework via dual prototypes. We first introduce class prototypes for each party to learn relationships between classes in the latent space, allowing the active party to predict unseen classes. We further design a probabilistic dual prototype learning scheme to dynamically select unaligned samples by conditional optimal transport cost with class prior probability. Moreover, a mixed prior guided module guides this selection process by combining local and global class prior probabilities. Finally, we adopt an \textit{adaptive gated feature aggregation strategy} to mitigate feature contribution inconsistency by dynamically weighting and aggregating local features across different parties. We proved that Proto-EVFL, as the first bi-level optimization framework in VFL, has a convergence rate of 1/\sqrt T. Extensive experiments on various datasets validate the superiority of our Proto-EVFL. Even in a zero-shot scenario with one unseen class, it outperforms baselines by at least 6.97% 

**Abstract (ZH)**: 垂直联邦学习中增强的双原型框架Proto-EVFL 

---
# Physics-constrained generative machine learning-based high-resolution downscaling of Greenland's surface mass balance and surface temperature 

**Title (ZH)**: 基于物理约束的生成式机器学习方法构建格陵兰表面质量平衡和表面温度高分辨率缩放模型 

**Authors**: Nils Bochow, Philipp Hess, Alexander Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2507.22485)  

**Abstract**: Accurate, high-resolution projections of the Greenland ice sheet's surface mass balance (SMB) and surface temperature are essential for understanding future sea-level rise, yet current approaches are either computationally demanding or limited to coarse spatial scales. Here, we introduce a novel physics-constrained generative modeling framework based on a consistency model (CM) to downscale low-resolution SMB and surface temperature fields by a factor of up to 32 (from 160 km to 5 km grid spacing) in a few sampling steps. The CM is trained on monthly outputs of the regional climate model MARv3.12 and conditioned on ice-sheet topography and insolation. By enforcing a hard conservation constraint during inference, we ensure approximate preservation of SMB and temperature sums on the coarse spatial scale as well as robust generalization to extreme climate states without retraining. On the test set, our constrained CM achieves a continued ranked probability score of 6.31 mmWE for the SMB and 0.1 K for the surface temperature, outperforming interpolation-based downscaling. Together with spatial power-spectral analysis, we demonstrate that the CM faithfully reproduces variability across spatial scales. We further apply bias-corrected outputs of the NorESM2 Earth System Model as inputs to our CM, to demonstrate the potential of our model to directly downscale ESM fields. Our approach delivers realistic, high-resolution climate forcing for ice-sheet simulations with fast inference and can be readily integrated into Earth-system and ice-sheet model workflows to improve projections of the future contribution to sea-level rise from Greenland and potentially other ice sheets and glaciers too. 

**Abstract (ZH)**: 基于物理约束的生成模型框架：高分辨率格陵兰冰 sheet 表面质量平衡和表面温度的细化 

---
# RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function 

**Title (ZH)**: RCR-AF: 通过减小泛化复杂性增强模型泛化能力的激活函数 

**Authors**: Yunrui Yu, Kafeng Wang, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22446)  

**Abstract**: Despite their widespread success, deep neural networks remain critically vulnerable to adversarial attacks, posing significant risks in safety-sensitive applications. This paper investigates activation functions as a crucial yet underexplored component for enhancing model robustness. We propose a Rademacher Complexity Reduction Activation Function (RCR-AF), a novel activation function designed to improve both generalization and adversarial resilience. RCR-AF uniquely combines the advantages of GELU (including smoothness, gradient stability, and negative information retention) with ReLU's desirable monotonicity, while simultaneously controlling both model sparsity and capacity through built-in clipping mechanisms governed by two hyperparameters, $\alpha$ and $\gamma$. Our theoretical analysis, grounded in Rademacher complexity, demonstrates that these parameters directly modulate the model's Rademacher complexity, offering a principled approach to enhance robustness. Comprehensive empirical evaluations show that RCR-AF consistently outperforms widely-used alternatives (ReLU, GELU, and Swish) in both clean accuracy under standard training and in adversarial robustness within adversarial training paradigms. 

**Abstract (ZH)**: 尽管深度神经网络在广泛应用中取得了巨大成功，但仍严重易受对抗攻击的影响，这在安全性关键应用中构成了显著风险。本文探讨了激活函数作为增强模型稳健性的重要但未充分研究的组件。我们提出了一种减小泛化复杂度的激活函数（RCR-AF），这是一种设计用于提高模型泛化能力和对抗鲁棒性的新型激活函数。RCR-AF 独特地结合了 GELU 的优势（包括平滑性、梯度稳定性以及负信息保留）与 ReLU 的单调性优势，并通过由两个超参数 α 和 γ 控制的内置剪裁机制同时控制模型的稀疏性和容量。基于 Rademacher 复杂性的理论分析表明，这些参数直接调节模型的 Rademacher 复杂度，提供了一种提高鲁棒性的原则性方法。全面的经验评估显示，RCR-AF 在标准训练下的干净准确率和对抗训练 paradigms 下的对抗鲁棒性方面均优于广泛使用的替代方案（ReLU、GELU 和 Swish）。 

---
# Theoretical Analysis of Relative Errors in Gradient Computations for Adversarial Attacks with CE Loss 

**Title (ZH)**: 对抗攻击中使用CE损失的梯度计算相对误差理论分析 

**Authors**: Yunrui Yu, Hang Su, Cheng-zhong Xu, Zhizhong Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22428)  

**Abstract**: Gradient-based adversarial attacks using the Cross-Entropy (CE) loss often suffer from overestimation due to relative errors in gradient computation induced by floating-point arithmetic. This paper provides a rigorous theoretical analysis of these errors, conducting the first comprehensive study of floating-point computation errors in gradient-based attacks across four distinct scenarios: (i) unsuccessful untargeted attacks, (ii) successful untargeted attacks, (iii) unsuccessful targeted attacks, and (iv) successful targeted attacks. We establish theoretical foundations characterizing the behavior of relative numerical errors under different attack conditions, revealing previously unknown patterns in gradient computation instability, and identify floating-point underflow and rounding as key contributors. Building on this insight, we propose the Theoretical MIFPE (T-MIFPE) loss function, which incorporates an optimal scaling factor $T = t^*$ to minimize the impact of floating-point errors, thereby enhancing the accuracy of gradient computation in adversarial attacks. Extensive experiments on the MNIST, CIFAR-10, and CIFAR-100 datasets demonstrate that T-MIFPE outperforms existing loss functions, including CE, C\&W, DLR, and MIFPE, in terms of attack potency and robustness evaluation accuracy. 

**Abstract (ZH)**: 基于梯度的对抗攻击使用交叉熵（CE）损失常常由于浮点算术引起的梯度计算相对误差而导致误估。本文提供了对这些误差的严格理论分析，首次全面研究了梯度基于攻击中四种不同场景下的浮点计算误差：(i) 未成功的无目标攻击，(ii) 成功的无目标攻击，(iii) 未成功的有目标攻击，和(iv) 成功的有目标攻击。我们建立了理论基础，以描述在不同攻击条件下相对数值误差的行为，揭示了梯度计算不稳定性的新模式，并识别出浮点下溢和舍入作为关键因素。基于这些洞察，我们提出了理论MIFPE（T-MIFPE）损失函数，该函数引入最优缩放因子 $T = t^*$ 以最小化浮点误差的影响，从而提高对抗攻击中梯度计算的准确性。在MNIST、CIFAR-10和CIFAR-100数据集上的广泛实验证明，T-MIFPE在攻击效力和鲁棒性评估准确性方面均优于现有损失函数，包括交叉熵（CE）、C&W、DLR和MIFPE。 

---
# Aleatoric Uncertainty Medical Image Segmentation Estimation via Flow Matching 

**Title (ZH)**: 基于流匹配的医学图像分割不确定性估计 

**Authors**: Phi Van Nguyen, Ngoc Huynh Trinh, Duy Minh Lam Nguyen, Phu Loc Nguyen, Quoc Long Tran  

**Link**: [PDF](https://arxiv.org/pdf/2507.22418)  

**Abstract**: Quantifying aleatoric uncertainty in medical image segmentation is critical since it is a reflection of the natural variability observed among expert annotators. A conventional approach is to model the segmentation distribution using the generative model, but current methods limit the expression ability of generative models. While current diffusion-based approaches have demonstrated impressive performance in approximating the data distribution, their inherent stochastic sampling process and inability to model exact densities limit their effectiveness in accurately capturing uncertainty. In contrast, our proposed method leverages conditional flow matching, a simulation-free flow-based generative model that learns an exact density, to produce highly accurate segmentation results. By guiding the flow model on the input image and sampling multiple data points, our approach synthesizes segmentation samples whose pixel-wise variance reliably reflects the underlying data distribution. This sampling strategy captures uncertainties in regions with ambiguous boundaries, offering robust quantification that mirrors inter-annotator differences. Experimental results demonstrate that our method not only achieves competitive segmentation accuracy but also generates uncertainty maps that provide deeper insights into the reliability of the segmentation outcomes. The code for this paper is freely available at this https URL 

**Abstract (ZH)**: 量化医学图像分割中的aleatoric不确定性对于反映专家注释者之间观察到的自然变异性至关重要。传统的做法是使用生成模型来建模分割分布，但当前方法限制了生成模型的表达能力。虽然基于扩散的方法在近似数据分布方面表现出色，但其固有的随机采样过程和无法准确建模密度的限制限制了它们在准确捕捉不确定性方面的有效性。相比之下，我们提出的方法利用条件流匹配，这是一种无需模拟的流式生成模型，学习精确的概率密度，从而生成高精度的分割结果。通过在输入图像上引导流模型并采样多个数据点，我们的方法综合了像素级方差可靠的分割样本，这些样本能够捕捉模糊边界区域的不确定性，提供与注释者之间差异相一致的稳健量化。实验结果表明，我们的方法不仅实现了竞争力的分割精度，还生成了不确定性图，提供了分割结果可靠性的更深入见解。本文的代码可以从此链接获取。 

---
# Question Generation for Assessing Early Literacy Reading Comprehension 

**Title (ZH)**: 早期识字阅读理解的疑问生成评估 

**Authors**: Xiaocheng Yang, Sumuk Shashidhar, Dilek Hakkani-Tur  

**Link**: [PDF](https://arxiv.org/pdf/2507.22410)  

**Abstract**: Assessment of reading comprehension through content-based interactions plays an important role in the reading acquisition process. In this paper, we propose a novel approach for generating comprehension questions geared to K-2 English learners. Our method ensures complete coverage of the underlying material and adaptation to the learner's specific proficiencies, and can generate a large diversity of question types at various difficulty levels to ensure a thorough evaluation. We evaluate the performance of various language models in this framework using the FairytaleQA dataset as the source material. Eventually, the proposed approach has the potential to become an important part of autonomous AI-driven English instructors. 

**Abstract (ZH)**: 基于内容的交互评估阅读 comprehension 对阅读习得过程起着重要作用。本文提出了一个针对K-2英语学习者的新型生成理解性问题的方法。该方法确保了对基础材料的完全覆盖，并适应学习者的特定 proficiency，能够生成多种难度级别和类型的大量问题以确保全面评估。我们利用FairytaleQA数据集评估了各种语言模型在此框架中的性能。最终，所提出的这种方法有望成为自主AI驱动的英语教师的重要组成部分。 

---
# MINR: Implicit Neural Representations with Masked Image Modelling 

**Title (ZH)**: MINR: 带有掩码图像建模的隐式神经表示 

**Authors**: Sua Lee, Joonhun Lee, Myungjoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22404)  

**Abstract**: Self-supervised learning methods like masked autoencoders (MAE) have shown significant promise in learning robust feature representations, particularly in image reconstruction-based pretraining task. However, their performance is often strongly dependent on the masking strategies used during training and can degrade when applied to out-of-distribution data. To address these limitations, we introduce the masked implicit neural representations (MINR) framework that synergizes implicit neural representations with masked image modeling. MINR learns a continuous function to represent images, enabling more robust and generalizable reconstructions irrespective of masking strategies. Our experiments demonstrate that MINR not only outperforms MAE in in-domain scenarios but also in out-of-distribution settings, while reducing model complexity. The versatility of MINR extends to various self-supervised learning applications, confirming its utility as a robust and efficient alternative to existing frameworks. 

**Abstract (ZH)**: 像掩蔽自动编码器（MAE）这样的自我监督学习方法在学习稳健的特征表示方面显示出显著的潜力，特别是在基于图像重建的预训练任务中。然而，它们的表现往往强烈依赖于训练过程中使用的掩蔽策略，并且在应用于离分布数据时可能会退化。为了解决这些限制，我们提出了掩蔽隐式神经表示（MINR）框架，将隐式神经表示与掩蔽图像建模相结合。MINR学习一个连续函数来表示图像，使得恢复结果不受掩蔽策略的影响，更加稳健和通用。我们的实验表明，MINR不仅在领域内场景中优于MAE，在离分布设置中也同样表现出色，同时降低了模型复杂度。MINR的灵活性扩展到了各种自我监督学习应用，证实了其作为现有框架的稳健且高效的替代方案的实用性。 

---
# Exploring the Application of Visual Question Answering (VQA) for Classroom Activity Monitoring 

**Title (ZH)**: 探索视觉问答（VQA）在教室活动监控中的应用 

**Authors**: Sinh Trong Vu, Hieu Trung Pham, Dung Manh Nguyen, Hieu Minh Hoang, Nhu Hoang Le, Thu Ha Pham, Tai Tan Mai  

**Link**: [PDF](https://arxiv.org/pdf/2507.22369)  

**Abstract**: Classroom behavior monitoring is a critical aspect of educational research, with significant implications for student engagement and learning outcomes. Recent advancements in Visual Question Answering (VQA) models offer promising tools for automatically analyzing complex classroom interactions from video recordings. In this paper, we investigate the applicability of several state-of-the-art open-source VQA models, including LLaMA2, LLaMA3, QWEN3, and NVILA, in the context of classroom behavior analysis. To facilitate rigorous evaluation, we introduce our BAV-Classroom-VQA dataset derived from real-world classroom video recordings at the Banking Academy of Vietnam. We present the methodology for data collection, annotation, and benchmark the performance of the selected VQA models on this dataset. Our initial experimental results demonstrate that all four models achieve promising performance levels in answering behavior-related visual questions, showcasing their potential in future classroom analytics and intervention systems. 

**Abstract (ZH)**: 课堂行为监控是教育研究中的一个关键方面，对学生的参与度和学习成果具有重大影响。视觉问答（VQA）模型的最新进步为从视频记录中自动分析复杂的课堂互动提供了有前景的工具。本文探讨了LLaMA2、LLaMA3、QWEN3和NVILA等多种开源VQA模型在课堂行为分析中的适用性。为促进严格评估，我们引入了BAV-Classroom-VQA数据集，该数据集源自越南 Banking Academy 的真实课堂视频录制。我们介绍了数据采集、标注的方法，并在该数据集上对所选的VQA模型进行了基准测试。初步的实验结果表明，所有四个模型在回答行为相关的视觉问题方面表现优异，展示了它们在未来课堂分析和干预系统中的潜力。 

---
# Learning from Heterogeneous Structural MRI via Collaborative Domain Adaptation for Late-Life Depression Assessment 

**Title (ZH)**: 基于协作领域适应的异质结构MRI在评估晚年抑郁中的学习 

**Authors**: Yuzhen Gao, Qianqian Wang, Yongheng Sun, Cui Wang, Yongquan Liang, Mingxia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22321)  

**Abstract**: Accurate identification of late-life depression (LLD) using structural brain MRI is essential for monitoring disease progression and facilitating timely intervention. However, existing learning-based approaches for LLD detection are often constrained by limited sample sizes (e.g., tens), which poses significant challenges for reliable model training and generalization. Although incorporating auxiliary datasets can expand the training set, substantial domain heterogeneity, such as differences in imaging protocols, scanner hardware, and population demographics, often undermines cross-domain transferability. To address this issue, we propose a Collaborative Domain Adaptation (CDA) framework for LLD detection using T1-weighted MRIs. The CDA leverages a Vision Transformer (ViT) to capture global anatomical context and a Convolutional Neural Network (CNN) to extract local structural features, with each branch comprising an encoder and a classifier. The CDA framework consists of three stages: (a) supervised training on labeled source data, (b) self-supervised target feature adaptation and (c) collaborative training on unlabeled target data. We first train ViT and CNN on source data, followed by self-supervised target feature adaptation by minimizing the discrepancy between classifier outputs from two branches to make the categorical boundary clearer. The collaborative training stage employs pseudo-labeled and augmented target-domain MRIs, enforcing prediction consistency under strong and weak augmentation to enhance domain robustness and generalization. Extensive experiments conducted on multi-site T1-weighted MRI data demonstrate that the CDA consistently outperforms state-of-the-art unsupervised domain adaptation methods. 

**Abstract (ZH)**: 使用T1加权MRI进行晚期生活抑郁症准确识别的协作领域适应框架对于监测疾病进展和促进及时干预至关重要。现有基于学习的晚期生活抑郁症检测方法常常受到有限样本量（例如十多个样本）的限制，这给可靠的模型训练和泛化带来了巨大挑战。尽管结合辅助数据集可以扩大训练集，但显著的领域异质性，如成像协议、扫描硬件和人口统计学差异，往往削弱了跨领域的可转移性能。为了解决这一问题，我们提出了一种用于晚期生活抑郁症检测的协作领域适应（CDA）框架，利用Vision Transformer捕捉全局解剖上下文，并使用卷积神经网络提取局部结构特征，每个分支包含一个编码器和一个分类器。CDA框架包括三个阶段：（a）在标记的源数据上进行监督训练，（b）在目标数据上进行自监督的目标特征适应，（c）在未标记的目标数据上进行协作训练。我们首先在源数据上训练Vision Transformer和卷积神经网络，然后通过最小化两个分支分类器输出之间的差异来进行自监督目标特征适应，以使类别边界更加清晰。协作训练阶段使用伪标签和增强的目标域MRI数据，通过在强增强和弱增强下强制预测一致性来增强领域鲁棒性和泛化能力。在多中心T1加权MRI数据上的广泛实验表明，CDA始终优于最先进的无监督领域适应方法。 

---
# AdapSCA-PSO: An Adaptive Localization Algorithm with AI-Based Hybrid SCA-PSO for IoT WSNs 

**Title (ZH)**: AdapSCA-PSO：一种基于AI融合SCA-PSO的自适应定位算法用于物联网WSNs 

**Authors**: Ze Zhang, Qian Dong, Wenhan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22317)  

**Abstract**: The accurate localization of sensor nodes is a fundamental requirement for the practical application of the Internet of Things (IoT). To enable robust localization across diverse environments, this paper proposes a hybrid meta-heuristic localization algorithm. Specifically, the algorithm integrates the Sine Cosine Algorithm (SCA), which is effective in global search, with Particle Swarm Optimization (PSO), which excels at local search. An adaptive switching module is introduced to dynamically select between the two algorithms. Furthermore, the initialization, fitness evaluation, and parameter settings of the algorithm have been specifically redesigned and optimized to address the characteristics of the node localization problem. Simulation results across varying numbers of sensor nodes demonstrate that, compared to standalone PSO and the unoptimized SCAPSO algorithm, the proposed method significantly reduces the number of required iterations and achieves an average localization error reduction of 84.97%. 

**Abstract (ZH)**: 物联网(IoT)中传感器节点的准确定位是其实用应用的基本要求。为了在多样化的环境中实现稳健的定位，本文提出了一种混合元启发式定位算法。具体而言，该算法将适用于全局搜索的Sine Cosine Algorithm (SCA)与适用于局部搜索的Particle Swarm Optimization (PSO)相结合，并引入了自适应切换模块以动态选择两者之间的算法。此外，算法的初始化、适应度评估和参数设置均针对节点定位问题进行了专门的设计和优化。仿真实验结果表明，与单独使用PSO以及未优化的SCAPSO算法相比，所提出的方法显著减少了所需的迭代次数，并实现了平均定位误差降低84.97%。 

---
# Agent-centric learning: from external reward maximization to internal knowledge curation 

**Title (ZH)**: 基于代理的学习：从外部奖励最大化到内部知识整理 

**Authors**: Hanqi Zhou, Fryderyk Mantiuk, David G. Nagy, Charley M. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22255)  

**Abstract**: The pursuit of general intelligence has traditionally centered on external objectives: an agent's control over its environments or mastery of specific tasks. This external focus, however, can produce specialized agents that lack adaptability. We propose representational empowerment, a new perspective towards a truly agent-centric learning paradigm by moving the locus of control inward. This objective measures an agent's ability to controllably maintain and diversify its own knowledge structures. We posit that the capacity -- to shape one's own understanding -- is an element for achieving better ``preparedness'' distinct from direct environmental influence. Focusing on internal representations as the main substrate for computing empowerment offers a new lens through which to design adaptable intelligent systems. 

**Abstract (ZH)**: 追求通用智能 traditionally centered on external objectives：agent's control over its environments or mastery of specific tasks. 内省智能：一种以代理为中心的学习范式的新视角 

---
# Using Scaling Laws for Data Source Utility Estimation in Domain-Specific Pre-Training 

**Title (ZH)**: 基于领域特定预训练的数据源效用估计的标度律方法 

**Authors**: Oleksiy Ostapenko, Charles Guille-Escuret, Luke Kumar, Max Tian, Denis Kocetkov, Gopeshh Subbaraj, Raymond Li, Joel Lamy-Poirier, Sebastien Paquet, Torsten Scholak  

**Link**: [PDF](https://arxiv.org/pdf/2507.22250)  

**Abstract**: We introduce a framework for optimizing domain-specific dataset construction in foundation model training. Specifically, we seek a cost-efficient way to estimate the quality of data sources (e.g. synthetically generated or filtered web data, etc.) in order to make optimal decisions about resource allocation for data sourcing from these sources for the stage two pre-training phase, aka annealing, with the goal of specializing a generalist pre-trained model to specific domains. Our approach extends the usual point estimate approaches, aka micro-annealing, to estimating scaling laws by performing multiple annealing runs of varying compute spent on data curation and training. This addresses a key limitation in prior work, where reliance on point estimates for data scaling decisions can be misleading due to the lack of rank invariance across compute scales -- a phenomenon we confirm in our experiments. By systematically analyzing performance gains relative to acquisition costs, we find that scaling curves can be estimated for different data sources. Such scaling laws can inform cost effective resource allocation across different data acquisition methods (e.g. synthetic data), data sources (e.g. user or web data) and available compute resources. We validate our approach through experiments on a pre-trained model with 7 billion parameters. We adapt it to: a domain well-represented in the pre-training data -- the medical domain, and a domain underrepresented in the pretraining corpora -- the math domain. We show that one can efficiently estimate the scaling behaviors of a data source by running multiple annealing runs, which can lead to different conclusions, had one used point estimates using the usual micro-annealing technique instead. This enables data-driven decision-making for selecting and optimizing data sources. 

**Abstract (ZH)**: 一种优化基础模型训练中领域特定数据集构建的框架 

---
# RL from Teacher-Model Refinement: Gradual Imitation Learning for Machine Translation 

**Title (ZH)**: 基于教师模型精炼的RL：逐步模仿学习在机器翻译中的应用 

**Authors**: Dongyub Jude Lee, Zhenyi Ye, Pengcheng He  

**Link**: [PDF](https://arxiv.org/pdf/2507.22219)  

**Abstract**: Preference-learning methods for machine translation (MT)--such as Direct Preference Optimization (DPO)--have achieved impressive gains but depend heavily on large, carefully curated triplet datasets and often struggle to generalize beyond their tuning domains. We propose Reinforcement Learning from Teacher-Model Refinement (RLfR), a novel framework that removes reliance on static triplets by leveraging continuous, high-quality feedback from an external teacher model (GPT-4o). RLfR frames each translation step as a micro-tutorial: the actor generates a hypothesis, the teacher refines it, and the actor is rewarded based on how closely it aligns with the teacher's refinement. Guided by two complementary signals--(i) negative edit distance, promoting lexical and structural fidelity, and (ii) COMET score, ensuring semantic adequacy--the actor progressively learns to emulate the teacher, mirroring a human learning process through incremental, iterative improvement. On the FLORES-200 benchmark (English to and from German, Spanish, Chinese, Korean, and Japanese), RLfR consistently outperforms both MT-SFT and preference-based baselines, significantly improving COMET (semantic adequacy) and M-ETA (entity preservation) scores. 

**Abstract (ZH)**: 基于教师模型细化的强化学习方法（RLfR）：面向机器翻译的偏好学习方法 

---
# Quantum-Inspired Audio Unlearning: Towards Privacy-Preserving Voice Biometrics 

**Title (ZH)**: 量子启发的音频遗忘：迈向隐私保护的语音生物特征识别 

**Authors**: Shreyansh Pathak, Sonu Shreshtha, Richa Singh, Mayank Vatsa  

**Link**: [PDF](https://arxiv.org/pdf/2507.22208)  

**Abstract**: The widespread adoption of voice-enabled authentication and audio biometric systems have significantly increased privacy vulnerabilities associated with sensitive speech data. Compliance with privacy regulations such as GDPR's right to be forgotten and India's DPDP Act necessitates targeted and efficient erasure of individual-specific voice signatures from already-trained biometric models. Existing unlearning methods designed for visual data inadequately handle the sequential, temporal, and high-dimensional nature of audio signals, leading to ineffective or incomplete speaker and accent erasure. To address this, we introduce QPAudioEraser, a quantum-inspired audio unlearning framework. Our our-phase approach involves: (1) weight initialization using destructive interference to nullify target features, (2) superposition-based label transformations that obscure class identity, (3) an uncertainty-maximizing quantum loss function, and (4) entanglement-inspired mixing of correlated weights to retain model knowledge. Comprehensive evaluations with ResNet18, ViT, and CNN architectures across AudioMNIST, Speech Commands, LibriSpeech, and Speech Accent Archive datasets validate QPAudioEraser's superior performance. The framework achieves complete erasure of target data (0% Forget Accuracy) while incurring minimal impact on model utility, with a performance degradation on retained data as low as 0.05%. QPAudioEraser consistently surpasses conventional baselines across single-class, multi-class, sequential, and accent-level erasure scenarios, establishing the proposed approach as a robust privacy-preserving solution. 

**Abstract (ZH)**: 基于量子启发的音频遗忘框架 QPAudioEraser：针对语音签名的高效个性化擦除 

---
# Measuring Time-Series Dataset Similarity using Wasserstein Distance 

**Title (ZH)**: 使用Wasserstein距离衡量时间序列数据集相似度 

**Authors**: Hongjie Chen, Akshay Mehra, Josh Kimball, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22189)  

**Abstract**: The emergence of time-series foundation model research elevates the growing need to measure the (dis)similarity of time-series datasets. A time-series dataset similarity measure aids research in multiple ways, including model selection, finetuning, and visualization. In this paper, we propose a distribution-based method to measure time-series dataset similarity by leveraging the Wasserstein distance. We consider a time-series dataset an empirical instantiation of an underlying multivariate normal distribution (MVN). The similarity between two time-series datasets is thus computed as the Wasserstein distance between their corresponding MVNs. Comprehensive experiments and visualization show the effectiveness of our approach. Specifically, we show how the Wasserstein distance helps identify similar time-series datasets and facilitates inference performance estimation of foundation models in both out-of-distribution and transfer learning evaluation, with high correlations between our proposed measure and the inference loss (>0.60). 

**Abstract (ZH)**: 时间序列基础模型研究的兴起提升了衡量时间序列数据集相似性的需求。时间序列数据集相似性测度有助于模型选择、微调和可视化等多种研究。本文提出一种基于分布的方法，通过利用Wasserstein距离来衡量时间序列数据集的相似性。我们将时间序列数据集视为潜在多元正态分布（MVN）的经验实例。因此，两个时间序列数据集之间的相似性被计算为它们相应MVN之间的Wasserstein距离。全面的实验和可视化结果表明该方法的有效性，特别是展示了Wasserstein距离如何帮助识别相似的时间序列数据集，并在域外和迁移学习评价中促进基础模型推断性能估计，且与我们提出的方法的相关系数超过0.60。 

---
# SourceSplice: Source Selection for Machine Learning Tasks 

**Title (ZH)**: 源选择：机器学习任务中的源选择 

**Authors**: Ambarish Singh, Romila Pradhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22186)  

**Abstract**: Data quality plays a pivotal role in the predictive performance of machine learning (ML) tasks - a challenge amplified by the deluge of data sources available in modern this http URL work in data discovery largely focus on metadata matching, semantic similarity or identifying tables that should be joined to answer a particular query, but do not consider source quality for high performance of the downstream ML this http URL paper addresses the problem of determining the best subset of data sources that must be combined to construct the underlying training dataset for a given ML this http URL propose SourceGrasp and SourceSplice, frameworks designed to efficiently select a suitable subset of sources that maximizes the utility of the downstream ML this http URL the algorithms rely on the core idea that sources (or their combinations) contribute differently to the task utility, and must be judiciously this http URL SourceGrasp utilizes a metaheuristic based on a greediness criterion and randomization, the SourceSplice framework presents a source selection mechanism inspired from gene splicing - a core concept used in protein this http URL empirically evaluate our algorithms on three real-world datasets and synthetic datasets and show that, with significantly fewer subset explorations, SourceSplice effectively identifies subsets of data sources leading to high task this http URL also conduct studies reporting the sensitivity of SourceSplice to the decision choices under several settings. 

**Abstract (ZH)**: 数据质量在机器学习任务的预测性能中起着关键作用——数据洪流加剧了这一挑战。现有的数据 discovery 工作主要集中在元数据匹配、语义相似性或识别用于回答特定查询的表连接，而不考虑数据源的质量以提高后续机器学习任务的性能。本文解决了确定最佳数据源子集的问题，该子集应被结合以构建给定机器学习任务的底层训练数据集。本文提出了SourceGrasp和SourceSplice框架，旨在高效选择一个合适的数据源子集，以最大化下游机器学习任务的效用。算法依赖的核心思想是，数据源（或它们的组合）对任务效用的贡献程度不同，并且必须谨慎选择。SourceGrasp利用基于贪婪准则和随机化的元启发式方法，而SourceSplice框架则借鉴了基因拼接的概念，这是蛋白质合成中的一个核心概念。本文在三个真实世界数据集和合成数据集上实证评估了我们的算法，并展示了SourceSplice在显著减少子集探索的情况下，有效识别出对任务性能有高贡献的数据源子集。此外，还进行了研究，探讨了在不同设置下SourceSplice决策选择的敏感性。 

---
# Spatial-Temporal Reinforcement Learning for Network Routing with Non-Markovian Traffic 

**Title (ZH)**: 基于非马尔可夫交通的时空强化学习网络路由方法 

**Authors**: Molly Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22174)  

**Abstract**: Reinforcement Learning (RL) has become a well-established approach for optimizing packet routing in communication networks. Standard RL algorithms typically are based on the Markov Decision Process (MDP), which assumes that the current state of the environment provides all the necessary information for system evolution and decision-making. However, this Markovian assumption is invalid in many practical scenarios, making the MDP and RL frameworks inadequate to produce the optimal solutions. Additionally, traditional RL algorithms often employ function approximations (e.g., by neural networks) that do not explicitly capture the spatial relationships inherent in environments with complex network topologies. Communication networks are characterized by dynamic traffic patterns and arbitrary numbers of nodes and links, which further complicate the decision-making process. To address these challenges, we propose a spatial-temporal RL approach that integrates Graph Neural Networks (GNNs) and Recurrent Neural Networks (RNNs) to adequately capture the spatial dynamics regarding network topology and temporal traffic patterns, respectively, to enhance routing decisions. Our evaluation demonstrates that the proposed method outperforms and is more robust to changes in the network topology when compared with traditional RL techniques. 

**Abstract (ZH)**: 基于空间-时间的图神经网络和循环神经网络结合的强化学习方法在通信网络包路由优化中的应用 

---
# Tiny Noise-Robust Voice Activity Detector for Voice Assistants 

**Title (ZH)**: Tiny噪声鲁棒语音活动检测器 

**Authors**: Hamed Jafarzadeh Asl, Mahsa Ghazvini Nejad, Amin Edraki, Masoud Asgharian, Vahid Partovi Nia  

**Link**: [PDF](https://arxiv.org/pdf/2507.22157)  

**Abstract**: Voice Activity Detection (VAD) in the presence of background noise remains a challenging problem in speech processing. Accurate VAD is essential in automatic speech recognition, voice-to-text, conversational agents, etc, where noise can severely degrade the performance. A modern application includes the voice assistant, specially mounted on Artificial Intelligence of Things (AIoT) devices such as cell phones, smart glasses, earbuds, etc, where the voice signal includes background noise. Therefore, VAD modules must remain light-weight due to their practical on-device limitation. The existing models often struggle with low signal-to-noise ratios across diverse acoustic environments. A simple VAD often detects human voice in a clean environment, but struggles to detect the human voice in noisy conditions. We propose a noise-robust VAD that comprises a light-weight VAD, with data pre-processing and post-processing added modules to handle the background noise. This approach significantly enhances the VAD accuracy in noisy environments and requires neither a larger model, nor fine-tuning. Experimental results demonstrate that our approach achieves a notable improvement compared to baselines, particularly in environments with high background noise interference. This modified VAD additionally improving clean speech detection. 

**Abstract (ZH)**: 背景噪声环境下语音活动检测（VAD）仍然是语音处理中的一个挑战性问题。 

---
# Runtime Failure Hunting for Physics Engine Based Software Systems: How Far Can We Go? 

**Title (ZH)**: 基于物理引擎的软件系统运行时故障定位：我们能走多远？ 

**Authors**: Shuqing Li, Qiang Chen, Xiaoxue Ren, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22099)  

**Abstract**: Physics Engines (PEs) are fundamental software frameworks that simulate physical interactions in applications ranging from entertainment to safety-critical systems. Despite their importance, PEs suffer from physics failures, deviations from expected physical behaviors that can compromise software reliability, degrade user experience, and potentially cause critical failures in autonomous vehicles or medical robotics. Current testing approaches for PE-based software are inadequate, typically requiring white-box access and focusing on crash detection rather than semantically complex physics failures. This paper presents the first large-scale empirical study characterizing physics failures in PE-based software. We investigate three research questions addressing the manifestations of physics failures, the effectiveness of detection techniques, and developer perceptions of current detection practices. Our contributions include: (1) a taxonomy of physics failure manifestations; (2) a comprehensive evaluation of detection methods including deep learning, prompt-based techniques, and large multimodal models; and (3) actionable insights from developer experiences for improving detection approaches. To support future research, we release PhysiXFails, code, and other materials at this https URL. 

**Abstract (ZH)**: Physics Engines中的物理失败：大规模实证研究及其检测方法探索 

---
# Scaling and Distilling Transformer Models for sEMG 

**Title (ZH)**: Transformer模型的缩放与蒸馏方法在sEMG中的应用 

**Authors**: Nicholas Mehlman, Jean-Christophe Gagnon-Audet, Michael Shvartsman, Kelvin Niu, Alexander H. Miller, Shagun Sodhani  

**Link**: [PDF](https://arxiv.org/pdf/2507.22094)  

**Abstract**: Surface electromyography (sEMG) signals offer a promising avenue for developing innovative human-computer interfaces by providing insights into muscular activity. However, the limited volume of training data and computational constraints during deployment have restricted the investigation of scaling up the model size for solving sEMG tasks. In this paper, we demonstrate that vanilla transformer models can be effectively scaled up on sEMG data and yield improved cross-user performance up to 110M parameters, surpassing the model size regime investigated in other sEMG research (usually <10M parameters). We show that >100M-parameter models can be effectively distilled into models 50x smaller with minimal loss of performance (<1.5% absolute). This results in efficient and expressive models suitable for complex real-time sEMG tasks in real-world environments. 

**Abstract (ZH)**: 基于表面肌电图的vanilla变压器模型在大规模训练数据上的有效扩展及其在实时复杂任务中的应用 

---
# Pathology Foundation Models are Scanner Sensitive: Benchmark and Mitigation with Contrastive ScanGen Loss 

**Title (ZH)**: Pathology Foundation Models are Scanner-Sensitive: Benchmark and Mitigation with Contrastive ScanGen Loss 

**Authors**: Gianluca Carloni, Biagio Brattoli, Seongho Keum, Jongchan Park, Taebum Lee, Chang Ho Ahn, Sergio Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2507.22092)  

**Abstract**: Computational pathology (CPath) has shown great potential in mining actionable insights from Whole Slide Images (WSIs). Deep Learning (DL) has been at the center of modern CPath, and while it delivers unprecedented performance, it is also known that DL may be affected by irrelevant details, such as those introduced during scanning by different commercially available scanners. This may lead to scanner bias, where the model outputs for the same tissue acquired by different scanners may vary. In turn, it hinders the trust of clinicians in CPath-based tools and their deployment in real-world clinical practices. Recent pathology Foundation Models (FMs) promise to provide better domain generalization capabilities. In this paper, we benchmark FMs using a multi-scanner dataset and show that FMs still suffer from scanner bias. Following this observation, we propose ScanGen, a contrastive loss function applied during task-specific fine-tuning that mitigates scanner bias, thereby enhancing the models' robustness to scanner variations. Our approach is applied to the Multiple Instance Learning task of Epidermal Growth Factor Receptor (EGFR) mutation prediction from H\&E-stained WSIs in lung cancer. We observe that ScanGen notably enhances the ability to generalize across scanners, while retaining or improving the performance of EGFR mutation prediction. 

**Abstract (ZH)**: 计算病理学（CPath）在从全视野图像（WSIs）中挖掘可操作见解方面显示出了巨大的潜力。深度学习（DL）一直是现代CPath的中心，尽管它提供了前所未有的性能，但众所周知，DL可能会受到无关细节的影响，例如不同商用扫描仪在扫描过程中引入的细节。这可能导致扫描仪偏差，即模型针对同一组织在不同扫描仪下获得的输出可能有所不同。这种偏差阻碍了 clinicians 对基于CPath的工具的信任及其在实际临床实践中的部署。最近的病理解析基础模型（FMs）有望提供更好的领域泛化能力。在本文中，我们使用多扫描仪数据集对FMs进行了基准测试，并表明FMs仍然受到扫描仪偏差的影响。基于此观察结果，我们提出了一种名为ScanGen的对比损失函数，在任务特定微调过程中应用该函数以减轻扫描仪偏差，从而增强模型对扫描仪变化的鲁棒性。我们通过肺癌中H&E染色WSIs的皮质生长因子受体（EGFR）突变预测的多项实例学习任务验证了该方法。我们观察到，ScanGen显著增强了模型跨扫描仪泛化的性能，同时保持或提高了EGFR突变预测的性能。 

---
# Hybrid activation functions for deep neural networks: S3 and S4 -- a novel approach to gradient flow optimization 

**Title (ZH)**: 混合激活函数用于深度神经网络：S3和S4——一种新的梯度流动优化方法 

**Authors**: Sergii Kavun  

**Link**: [PDF](https://arxiv.org/pdf/2507.22090)  

**Abstract**: Activation functions are critical components in deep neural networks, directly influencing gradient flow, training stability, and model performance. Traditional functions like ReLU suffer from dead neuron problems, while sigmoid and tanh exhibit vanishing gradient issues. We introduce two novel hybrid activation functions: S3 (Sigmoid-Softsign) and its improved version S4 (smoothed S3). S3 combines sigmoid for negative inputs with softsign for positive inputs, while S4 employs a smooth transition mechanism controlled by a steepness parameter k. We conducted comprehensive experiments across binary classification, multi-class classification, and regression tasks using three different neural network architectures. S4 demonstrated superior performance compared to nine baseline activation functions, achieving 97.4% accuracy on MNIST, 96.0% on Iris classification, and 18.7 MSE on Boston Housing regression. The function exhibited faster convergence (-19 for ReLU) and maintained stable gradient flow across network depths. Comparative analysis revealed S4's gradient range of [0.24, 0.59] compared to ReLU's 18% dead neurons in deep networks. The S4 activation function addresses key limitations of existing functions through its hybrid design and smooth transition mechanism. The tunable parameter k allows adaptation to different tasks and network depths, making S4 a versatile choice for deep learning applications. These findings suggest that hybrid activation functions represent a promising direction for improving neural network training dynamics. 

**Abstract (ZH)**: 激活函数是深度神经网络的关键组件，直接影响梯度流动、训练稳定性和模型性能。传统的激活函数如ReLU存在死亡神经元问题，而sigmoid和tanh则表现出梯度消失的问题。我们介绍了两种新型的混合激活函数：S3（Sigmoid-Softsign）及其改进版本S4（光滑的S3）。S3将sigmoid用于负输入，softsign用于正输入，而S4采用了一个由陡峭度参数k控制的平滑过渡机制。我们在二分类、多分类和回归任务中分别使用了三种不同的神经网络架构进行了全面的实验。S4在多项基准激活函数中表现最佳，在MNIST数据集上达到了97.4%的准确率，在Iris分类上达到了96.0%的准确率，在波士顿住房回归任务上达到了18.7的均方误差。该函数展示了更快的收敛速度（比ReLU快19%）并维持了网络深度下的稳定梯度流动。对比分析显示，S4的梯度范围为[0.24, 0.59]，而ReLU在深层网络中的死亡神经元比例高达18%。S4激活函数通过其混合设计和平滑过渡机制解决了现有激活函数的关键限制。可调参数k允许其适应不同的任务和网络深度，使S4成为深度学习应用中的一个灵活选择。这些发现表明，混合激活函数为改善神经网络训练动态提供了有前景的方向。 

---
# Principled Curriculum Learning using Parameter Continuation Methods 

**Title (ZH)**: 原理性的课程学习方法研究——参数延续方法的应用 

**Authors**: Harsh Nilesh Pathak, Randy Paffenroth  

**Link**: [PDF](https://arxiv.org/pdf/2507.22089)  

**Abstract**: In this work, we propose a parameter continuation method for the optimization of neural networks. There is a close connection between parameter continuation, homotopies, and curriculum learning. The methods we propose here are theoretically justified and practically effective for several problems in deep neural networks. In particular, we demonstrate better generalization performance than state-of-the-art optimization techniques such as ADAM for supervised and unsupervised learning tasks. 

**Abstract (ZH)**: 本工作中，我们提出了一种参数 continuuation 方法用于神经网络优化。参数 continuuation、同伦和 curriculum learning 之间存在密切联系。我们提出的方法在深度神经网络的多个问题上具有理论依据和实际效果，特别是在监督和无监督学习任务中展示了比 ADAM 等最先进的优化技术更好的泛化性能。 

---
# Machine Learning Experiences: A story of learning AI for use in enterprise software testing that can be used by anyone 

**Title (ZH)**: 机器学习经验：一个用于企业软件测试的AI学习故事，任何人都可以使用 

**Authors**: Michael Cohoon, Debbie Furman  

**Link**: [PDF](https://arxiv.org/pdf/2507.22064)  

**Abstract**: This paper details the machine learning (ML) journey of a group of people focused on software testing. It tells the story of how this group progressed through a ML workflow (similar to the CRISP-DM process). This workflow consists of the following steps and can be used by anyone applying ML techniques to a project: gather the data; clean the data; perform feature engineering on the data; splitting the data into two sets, one for training and one for testing; choosing a machine learning model; training the model; testing the model and evaluating the model performance. By following this workflow, anyone can effectively apply ML to any project that they are doing. 

**Abstract (ZH)**: 这篇论文详细讲述了专注于软件测试的一群人进行机器学习（ML）的旅程。它讲述了该团队如何通过类似CRISP-DM流程的ML工作流来进步。该工作流包含以下步骤，并且任何应用ML技术到项目的人都可以使用：收集数据；清理数据；对数据进行特征工程；将数据分为两部分，一部分用于训练，另一部分用于测试；选择机器学习模型；训练模型；测试模型并评估模型性能。通过遵循此工作流，任何人都可以有效地将ML应用于他们正在进行的任何项目中。 

---
# GABRIL: Gaze-Based Regularization for Mitigating Causal Confusion in Imitation Learning 

**Title (ZH)**: GABRIL：基于凝视的正则化方法以减轻模仿学习中的因果混淆 

**Authors**: Amin Banayeeanzade, Fatemeh Bahrani, Yutai Zhou, Erdem Bıyık  

**Link**: [PDF](https://arxiv.org/pdf/2507.19647)  

**Abstract**: Imitation Learning (IL) is a widely adopted approach which enables agents to learn from human expert demonstrations by framing the task as a supervised learning problem. However, IL often suffers from causal confusion, where agents misinterpret spurious correlations as causal relationships, leading to poor performance in testing environments with distribution shift. To address this issue, we introduce GAze-Based Regularization in Imitation Learning (GABRIL), a novel method that leverages the human gaze data gathered during the data collection phase to guide the representation learning in IL. GABRIL utilizes a regularization loss which encourages the model to focus on causally relevant features identified through expert gaze and consequently mitigates the effects of confounding variables. We validate our approach in Atari environments and the Bench2Drive benchmark in CARLA by collecting human gaze datasets and applying our method in both domains. Experimental results show that the improvement of GABRIL over behavior cloning is around 179% more than the same number for other baselines in the Atari and 76% in the CARLA setup. Finally, we show that our method provides extra explainability when compared to regular IL agents. 

**Abstract (ZH)**: 基于注视的强化学习正则化在模仿学习中的应用：GABRIL方法 

---
# RecPS: Privacy Risk Scoring for Recommender Systems 

**Title (ZH)**: RecPS: 推荐系统中的隐私风险评分 

**Authors**: Jiajie He, Yuechun Gu, Keke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18365)  

**Abstract**: Recommender systems (RecSys) have become an essential component of many web applications. The core of the system is a recommendation model trained on highly sensitive user-item interaction data. While privacy-enhancing techniques are actively studied in the research community, the real-world model development still depends on minimal privacy protection, e.g., via controlled access. Users of such systems should have the right to choose \emph{not} to share highly sensitive interactions. However, there is no method allowing the user to know which interactions are more sensitive than others. Thus, quantifying the privacy risk of RecSys training data is a critical step to enabling privacy-aware RecSys model development and deployment. We propose a membership-inference attack (MIA)- based privacy scoring method, RecPS, to measure privacy risks at both the interaction and user levels. The RecPS interaction-level score definition is motivated and derived from differential privacy, which is then extended to the user-level scoring method. A critical component is the interaction-level MIA method RecLiRA, which gives high-quality membership estimation. We have conducted extensive experiments on well-known benchmark datasets and RecSys models to show the unique features and benefits of RecPS scoring in risk assessment and RecSys model unlearning. 

**Abstract (ZH)**: 推荐系统（RecSys）已成为许多网络应用中的重要组成部分。系统的核心是一个在高度敏感的用户-项交互数据上训练的推荐模型。尽管在研究领域积极研究增强隐私的技术，但在现实世界中的模型开发仍然依赖于最小的隐私保护，例如通过受控访问。此类系统中的用户应有权选择不共享高度敏感的交互。然而，目前没有方法让用户知道哪些交互比其他交互更为敏感。因此，量化RecSys训练数据的隐私风险是实现具有隐私意识的RecSys模型开发和部署的关键步骤。我们提出了一种基于成员推断攻击（MIA）的隐私评分方法RecPS，以在交互和用户层面衡量隐私风险。RecPS的交互层面评分定义源自差分隐私，并进一步扩展为用户层面的评分方法。一个关键组成部分是交互层面的MIA方法RecLiRA，它提供了高质量的成员估计。我们在知名的基准数据集和RecSys模型上进行了广泛的实验，以展示RecPS评分在风险评估和RecSys模型遗忘中的独特特性和优势。 

---
# Spatial-Temporal Data Mining for Ocean Science: Data, Methodologies, and Opportunities 

**Title (ZH)**: 海洋科学中的时空数据挖掘：数据、方法与机遇 

**Authors**: Hanchen Yang, Wengen Li, Shuyu Wang, Hui Li, Jihong Guan, Shuigeng Zhou, Jiannong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2307.10803)  

**Abstract**: With the rapid amassing of spatial-temporal (ST) ocean data, many spatial-temporal data mining (STDM) studies have been conducted to address various oceanic issues, including climate forecasting and disaster warning. Compared with typical ST data (e.g., traffic data), ST ocean data is more complicated but with unique characteristics, e.g., diverse regionality and high sparsity. These characteristics make it difficult to design and train STDM models on ST ocean data. To the best of our knowledge, a comprehensive survey of existing studies remains missing in the literature, which hinders not only computer scientists from identifying the research issues in ocean data mining but also ocean scientists to apply advanced STDM techniques. In this paper, we provide a comprehensive survey of existing STDM studies for ocean science. Concretely, we first review the widely-used ST ocean datasets and highlight their unique characteristics. Then, typical ST ocean data quality enhancement techniques are explored. Next, we classify existing STDM studies in ocean science into four types of tasks, i.e., prediction, event detection, pattern mining, and anomaly detection, and elaborate on the techniques for these tasks. Finally, promising research opportunities are discussed. This survey can help scientists from both computer science and ocean science better understand the fundamental concepts, key techniques, and open challenges of STDM for ocean science. 

**Abstract (ZH)**: 随着时空（ST）海洋数据的迅速积累，许多时空数据挖掘（STDM）研究已被开展以解决各种海洋问题，包括气候预测和灾害预警。与典型的时空数据（如交通数据）相比，ST海洋数据更为复杂但也具有独特的特征，如多样的地理特性和高稀疏性。这些特征使得在ST海洋数据上设计和训练STDM模型具有挑战性。据我们所知，文献中缺乏对现有研究的全面综述，这不仅阻碍了计算机科学家识别海洋数据挖掘中的研究问题，也阻碍了海洋科学家应用先进的STDM技术。本文为海洋科学提供了STDM研究的全面综述。具体而言，我们首先回顾了广泛使用的ST海洋数据集并突出了它们的独特的特征。然后，探讨了典型的ST海洋数据质量增强技术。接下来，我们将现有的STDM研究在海洋科学中分类为四类任务，即预测、事件检测、模式挖掘和异常检测，并详细阐述了这些任务的技术。最后，讨论了有前景的研究机会。本综述有助于来自计算机科学和海洋科学的科学家更好地理解STDM在海洋科学中的基本概念、关键技术及开放挑战。 

---
