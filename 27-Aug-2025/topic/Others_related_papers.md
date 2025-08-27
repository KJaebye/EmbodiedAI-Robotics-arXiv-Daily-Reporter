# Direction Informed Trees (DIT*): Optimal Path Planning via Direction Filter and Direction Cost Heuristic 

**Title (ZH)**: 方向导向树(DIT*): 基于方向过滤和方向成本启发式的最优路径规划 

**Authors**: Liding Zhang, Kejia Chen, Kuanqi Cai, Yu Zhang, Yixuan Dang, Yansong Wu, Zhenshan Bing, Fan Wu, Sami Haddadin, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.19168)  

**Abstract**: Optimal path planning requires finding a series of feasible states from the starting point to the goal to optimize objectives. Popular path planning algorithms, such as Effort Informed Trees (EIT*), employ effort heuristics to guide the search. Effective heuristics are accurate and computationally efficient, but achieving both can be challenging due to their conflicting nature. This paper proposes Direction Informed Trees (DIT*), a sampling-based planner that focuses on optimizing the search direction for each edge, resulting in goal bias during exploration. We define edges as generalized vectors and integrate similarity indexes to establish a directional filter that selects the nearest neighbors and estimates direction costs. The estimated direction cost heuristics are utilized in edge evaluation. This strategy allows the exploration to share directional information efficiently. DIT* convergence faster than existing single-query, sampling-based planners on tested problems in R^4 to R^16 and has been demonstrated in real-world environments with various planning tasks. A video showcasing our experimental results is available at: this https URL 

**Abstract (ZH)**: 基于方向指导树的最优路径规划：一种面向搜索方向优化的采样基于规划算法 

---
# Real-time Testing of Satellite Attitude Control With a Reaction Wheel Hardware-In-the-Loop Platform 

**Title (ZH)**: 基于反应轮硬件在环平台的实时卫星姿态控制测试 

**Authors**: Morokot Sakal, George Nehma, Camilo Riano-Rios, Madhur Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2508.19164)  

**Abstract**: We propose the Hardware-in-the-Loop (HIL) test of an adaptive satellite attitude control system with reaction wheel health estimation capabilities. Previous simulations and Software-in-the-Loop testing have prompted further experiments to explore the validity of the controller with real momentum exchange devices in the loop. This work is a step toward a comprehensive testing framework for validation of spacecraft attitude control algorithms. The proposed HIL testbed includes brushless DC motors and drivers that communicate using a CAN bus, an embedded computer that executes control and adaptation laws, and a satellite simulator that produces simulated sensor data, estimated attitude states, and responds to actions of the external actuators. We propose methods to artificially induce failures on the reaction wheels, and present related issues and lessons learned. 

**Abstract (ZH)**: 硬件在环测试环境下自适应卫星姿态控制系统及飞轮健康估计能力的研究 

---
# AS2FM: Enabling Statistical Model Checking of ROS 2 Systems for Robust Autonomy 

**Title (ZH)**: AS2FM: 为鲁棒自主性启用ROS 2系统的统计模型检查 

**Authors**: Christian Henkel, Marco Lampacrescia, Michaela Klauck, Matteo Morelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.18820)  

**Abstract**: Designing robotic systems to act autonomously in unforeseen environments is a challenging task. This work presents a novel approach to use formal verification, specifically Statistical Model Checking (SMC), to verify system properties of autonomous robots at design-time. We introduce an extension of the SCXML format, designed to model system components including both Robot Operating System 2 (ROS 2) and Behavior Tree (BT) features. Further, we contribute Autonomous Systems to Formal Models (AS2FM), a tool to translate the full system model into JANI. The use of JANI, a standard format for quantitative model checking, enables verification of system properties with off-the-shelf SMC tools. We demonstrate the practical usability of AS2FM both in terms of applicability to real-world autonomous robotic control systems, and in terms of verification runtime scaling. We provide a case study, where we successfully identify problems in a ROS 2-based robotic manipulation use case that is verifiable in less than one second using consumer hardware. Additionally, we compare to the state of the art and demonstrate that our method is more comprehensive in system feature support, and that the verification runtime scales linearly with the size of the model, instead of exponentially. 

**Abstract (ZH)**: 设计能够在未预见环境中自主行动的机器人系统是一个具有挑战性的任务。本工作提出了一种利用形式验证，特别是统计模型检查（SMC），在设计阶段验证自主机器人系统属性的新方法。我们引入了SCXML格式的扩展，用于建模包括Robot Operating System 2（ROS 2）和行为树（BT）特征在内的系统组件。此外，我们贡献了自主系统到形式模型（AS2FM）工具，用于将整个系统模型转换为JANI格式。JANI是定量模型检查的标准格式，能够使用现成的SMC工具验证系统属性。我们从实际应用和验证运行时扩展性两方面展示了AS2FM的实用价值。我们提供了一个案例研究，使用消费级硬件在不到一秒的时间内成功识别出基于ROS 2的机器人操作使用案例中的问题，并且验证运行时随模型大小线性扩展。 

---
# Engineering Automotive Digital Twins on Standardized Architectures: A Case Study 

**Title (ZH)**: 基于标准化架构的汽车数字孪生工程：一个案例研究 

**Authors**: Stefan Ramdhan, Winnie Trandinh, Istvan David, Vera Pantelic, Mark Lawford  

**Link**: [PDF](https://arxiv.org/pdf/2508.18662)  

**Abstract**: Digital twin (DT) technology has become of interest in the automotive industry. There is a growing need for smarter services that utilize the unique capabilities of DTs, ranging from computer-aided remote control to cloud-based fleet coordination. Developing such services starts with the software architecture. However, the scarcity of DT architectural guidelines poses a challenge for engineering automotive DTs. Currently, the only DT architectural standard is the one defined in ISO 23247. Though not developed for automotive systems, it is one of the few feasible starting points for automotive DTs. In this work, we investigate the suitability of the ISO 23247 reference architecture for developing automotive DTs. Through the case study of developing an Adaptive Cruise Control DT for a 1/10\textsuperscript{th}-scale autonomous vehicle, we identify some strengths and limitations of the reference architecture and begin distilling future directions for researchers, practitioners, and standard developers. 

**Abstract (ZH)**: 数字孪生(DT)技术在汽车行业的应用引起了广泛关注。智能服务的需求日益增加，这些服务利用了DT的独特功能，从计算机辅助远程控制到基于云的车队协调。开发such服务始于软件架构。然而，缺乏专门针对DT的工程指南构成了一个挑战。目前，唯一存在的DT架构标准是ISO 23247定义的标准。尽管该标准并非专为汽车系统而设计，但它是开发汽车DT的少数可行起点之一。在这项工作中，我们探讨了ISO 23247参考架构在开发汽车DT方面的适用性。通过针对1/10尺度自主车辆开发自适应巡航控制(DT)案例研究，我们识别了参考架构的一些优势和局限性，并开始为研究者、从业者和标准开发者明确未来的研究方向。 

---
# Are All Marine Species Created Equal? Performance Disparities in Underwater Object Detection 

**Title (ZH)**: 水下目标检测中 marine 种类的表现差异：并非所有海洋物种都平等？ 

**Authors**: Melanie Wille, Tobias Fischer, Scarlett Raine  

**Link**: [PDF](https://arxiv.org/pdf/2508.18729)  

**Abstract**: Underwater object detection is critical for monitoring marine ecosystems but poses unique challenges, including degraded image quality, imbalanced class distribution, and distinct visual characteristics. Not every species is detected equally well, yet underlying causes remain unclear. We address two key research questions: 1) What factors beyond data quantity drive class-specific performance disparities? 2) How can we systematically improve detection of under-performing marine species? We manipulate the DUO dataset to separate the object detection task into localization and classification and investigate the under-performance of the scallop class. Localization analysis using YOLO11 and TIDE finds that foreground-background discrimination is the most problematic stage regardless of data quantity. Classification experiments reveal persistent precision gaps even with balanced data, indicating intrinsic feature-based challenges beyond data scarcity and inter-class dependencies. We recommend imbalanced distributions when prioritizing precision, and balanced distributions when prioritizing recall. Improving under-performing classes should focus on algorithmic advances, especially within localization modules. We publicly release our code and datasets. 

**Abstract (ZH)**: 水下目标检测对于监测海洋生态系统至关重要，但面临着独特挑战，包括图像质量退化、类别分布失衡和独特的视觉特征。并非每种物种都能同等检测，背后的原因尚不清楚。我们探讨了两个关键研究问题：1) 除了数据量外，哪些因素导致类别的性能差异？2) 我们如何系统性地提高性能不佳的海洋物种的检测能力？我们将DUO数据集拆分为定位和分类任务，研究扇贝类别的性能不佳问题。使用YOLO11和TIDE的定位分析发现，无论数据量如何，前景与背景的区分是最具问题的阶段。分类实验表明，在数据平衡的情况下依然存在持续的精度差距，这表明除了数据稀缺性和类别间依赖性之外，还存在固有的特征挑战。我们建议在优先考虑精度时关注不平衡分布，在优先考虑召回率时关注平衡分布。提高性能不佳的类别应集中在算法的进步上，尤其是在定位模块中。我们公开发布我们的代码和数据集。 

---
# Model Context Protocols in Adaptive Transport Systems: A Survey 

**Title (ZH)**: 自适应传输系统中模型上下文协议的研究综述 

**Authors**: Gaurab Chhetri, Shriyank Somvanshi, Md Monzurul Islam, Shamyo Brotee, Mahmuda Sultana Mimi, Dipti Koirala, Biplov Pandey, Subasish Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.19239)  

**Abstract**: The rapid expansion of interconnected devices, autonomous systems, and AI applications has created severe fragmentation in adaptive transport systems, where diverse protocols and context sources remain isolated. This survey provides the first systematic investigation of the Model Context Protocol (MCP) as a unifying paradigm, highlighting its ability to bridge protocol-level adaptation with context-aware decision making. Analyzing established literature, we show that existing efforts have implicitly converged toward MCP-like architectures, signaling a natural evolution from fragmented solutions to standardized integration frameworks. We propose a five-category taxonomy covering adaptive mechanisms, context-aware frameworks, unification models, integration strategies, and MCP-enabled architectures. Our findings reveal three key insights: traditional transport protocols have reached the limits of isolated adaptation, MCP's client-server and JSON-RPC structure enables semantic interoperability, and AI-driven transport demands integration paradigms uniquely suited to MCP. Finally, we present a research roadmap positioning MCP as a foundation for next-generation adaptive, context-aware, and intelligent transport infrastructures. 

**Abstract (ZH)**: 互联设备、自主系统和AI应用的迅速扩张在适应性运输系统中造成了严重的碎片化，多种协议和上下文来源仍然孤立存在。本文综述提供了MCP（Model Context Protocol）统一范式的首次系统性研究，突显了其在协议级适应与上下文aware决策之间架桥的能力。通过分析现有文献，我们展示了现有努力已不自觉地朝向类似MCP的架构演化，标志着从碎片化解决方案向标准化集成框架的自然发展。我们提出了包括自适应机制、上下文aware框架、统一模型、集成策略和MCP启用架构的五类分类法。我们的研究揭示了三个关键见解：传统运输协议在孤立适应方面已触及极限，MCP的客户端-服务器结构和JSON-RPC框架促进了语义互操作性，AI驱动的运输需要与MCP独特契合的集成范式。最后，我们提出了一条研究路线图，将MCP定位为下一代适应性、上下文aware和智能化运输基础设施的基础。 

---
# StepWiser: Stepwise Generative Judges for Wiser Reasoning 

**Title (ZH)**: StepWiser: 步骤生成法官以促进更明智的推理 

**Authors**: Wei Xiong, Wenting Zhao, Weizhe Yuan, Olga Golovneva, Tong Zhang, Jason Weston, Sainbayar Sukhbaatar  

**Link**: [PDF](https://arxiv.org/pdf/2508.19229)  

**Abstract**: As models increasingly leverage multi-step reasoning strategies to solve complex problems, supervising the logical validity of these intermediate steps has become a critical research challenge. Process reward models address this by providing step-by-step feedback, but current approaches have two major drawbacks: they typically function as classifiers without providing explanations, and their reliance on supervised fine-tuning with static datasets limits generalization. Inspired by recent advances, we reframe stepwise reward modeling from a classification task to a reasoning task itself. We thus propose a generative judge that reasons about the policy model's reasoning steps (i.e., meta-reasons), outputting thinking tokens before delivering a final verdict. Our model, StepWiser, is trained by reinforcement learning using relative outcomes of rollouts. We show it provides (i) better judgment accuracy on intermediate steps than existing methods; (ii) can be used to improve the policy model at training time; and (iii) improves inference-time search. 

**Abstract (ZH)**: 随着模型越来越多地采用多步推理策略来解决复杂问题，监督这些中间步骤的逻辑有效性已成为一个关键的研究挑战。推理奖励模型通过提供逐步反馈来应对这一挑战，但当前的方法存在两大缺点：它们通常作为分类器运作而不提供解释，并且依赖于静态数据集的监督微调限制了泛化能力。受 recent 进展启发，我们将步wise奖励建模从分类任务重新界定为推理任务本身。因此，我们提出了一种生成性裁判模型，该模型关于策略模型的推理步骤（即元推理）进行推理，在给出最终裁决之前输出思考令牌。我们的模型 StepWiser 通过使用卷出结果的相对结果进行强化学习训练。我们展示了它在以下方面的优势：(i) 在中间步骤的判断准确性上优于现有方法；(ii) 可以在训练时改进策略模型；以及(iii) 改进了推理时的搜索。 

---
# The Subset Sum Matching Problem 

**Title (ZH)**: 子集和匹配问题 

**Authors**: Yufei Wu, Manuel R. Torres, Parisa Zehtabi, Alberto Pozanco Lancho, Michael Cashmore, Daniel Borrajo, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.19218)  

**Abstract**: This paper presents a new combinatorial optimisation task, the Subset Sum Matching Problem (SSMP), which is an abstraction of common financial applications such as trades reconciliation. We present three algorithms, two suboptimal and one optimal, to solve this problem. We also generate a benchmark to cover different instances of SSMP varying in complexity, and carry out an experimental evaluation to assess the performance of the approaches. 

**Abstract (ZH)**: 本文提出了一种新的组合优化问题，即子集和匹配问题（SSMP），它是常见的金融应用如交易对账的抽象。我们提出了三种算法，其中两种子最优算法和一种最优算法来解决这个问题。我们还生成了基准测试以涵盖不同复杂度的SSMP实例，并进行了实验评估以评估这些方法的性能。 

---
# Playstyle and Artificial Intelligence: An Initial Blueprint Through the Lens of Video Games 

**Title (ZH)**: 游戏风格与人工智能：通过电子游戏视角的初步蓝图 

**Authors**: Chiu-Chou Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.19152)  

**Abstract**: Contemporary artificial intelligence (AI) development largely centers on rational decision-making, valued for its measurability and suitability for objective evaluation. Yet in real-world contexts, an intelligent agent's decisions are shaped not only by logic but also by deeper influences such as beliefs, values, and preferences. The diversity of human decision-making styles emerges from these differences, highlighting that "style" is an essential but often overlooked dimension of intelligence.
This dissertation introduces playstyle as an alternative lens for observing and analyzing the decision-making behavior of intelligent agents, and examines its foundational meaning and historical context from a philosophical perspective. By analyzing how beliefs and values drive intentions and actions, we construct a two-tier framework for style formation: the external interaction loop with the environment and the internal cognitive loop of deliberation. On this basis, we formalize style-related characteristics and propose measurable indicators such as style capacity, style popularity, and evolutionary dynamics.
The study focuses on three core research directions: (1) Defining and measuring playstyle, proposing a general playstyle metric based on discretized state spaces, and extending it to quantify strategic diversity and competitive balance; (2) Expressing and generating playstyle, exploring how reinforcement learning and imitation learning can be used to train agents exhibiting specific stylistic tendencies, and introducing a novel approach for human-like style learning and modeling; and (3) Practical applications, analyzing the potential of these techniques in domains such as game design and interactive entertainment.
Finally, the dissertation outlines future extensions, including the role of style as a core element in building artificial general intelligence (AGI). 

**Abstract (ZH)**: 当代人工智能（AI）的发展主要集中在理性决策上，因其可测量性和适于客观评估而受到重视。然而，在现实世界中，智能代理的决策不仅受到逻辑的影响，还受到信念、价值观和偏好等更深层次因素的影响。人类决策风格的多样性正是源于这些差异，突显了“风格”作为智能的一个重要但常被忽视的维度。

本论文引入了游戏风格作为观察和分析智能代理决策行为的替代视角，并从哲学角度探讨其基础意义和历史背景。通过分析信念和价值观如何驱动意图和行为，我们构建了一个两层框架来说明风格形成：外部与环境的交互循环以及内部认知的审思循环。在此基础上，我们形式化了与风格相关的特点，并提出了如风格容量、风格流行度和进化动态等可测量指标。

研究集中于三个核心研究方向：（1）定义和测量游戏风格，提出基于离散状态空间的一般游戏风格度量，并将其扩展以量化策略多样性和竞争平衡；（2）表达和生成游戏风格，探讨强化学习和模仿学习如何被用于训练表现出特定风格倾向的代理，并介绍了一种新型的人类风格学习和建模方法；（3）实际应用，分析这些技术在游戏设计和互动娱乐等领域的潜力。

最后，论文概述了未来可能的扩展，包括将风格作为构建通用人工智能（AGI）的核心要素的作用。 

---
# Algorithmic Collective Action with Multiple Collectives 

**Title (ZH)**: 多种集体的算法集体行动 

**Authors**: Claudio Battiloro, Pietro Greiner, Bret Nestor, Oumaima Amezgar, Francesca Dominici  

**Link**: [PDF](https://arxiv.org/pdf/2508.19149)  

**Abstract**: As learning systems increasingly influence everyday decisions, user-side steering via Algorithmic Collective Action (ACA)-coordinated changes to shared data-offers a complement to regulator-side policy and firm-side model design. Although real-world actions have been traditionally decentralized and fragmented into multiple collectives despite sharing overarching objectives-with each collective differing in size, strategy, and actionable goals, most of the ACA literature focused on single collective settings. In this work, we present the first theoretical framework for ACA with multiple collectives acting on the same system. In particular, we focus on collective action in classification, studying how multiple collectives can plant signals, i.e., bias a classifier to learn an association between an altered version of the features and a chosen, possibly overlapping, set of target classes. We provide quantitative results about the role and the interplay of collectives' sizes and their alignment of goals. Our framework, by also complementing previous empirical results, opens a path for a holistic treatment of ACA with multiple collectives. 

**Abstract (ZH)**: 随着学习系统越来越多地影响日常决策，用户侧通过算法联合行动（Algorithmic Collective Action, ACA）协调对共享数据的更改——这为监管侧政策和企业侧模型设计提供了补充。尽管实际行动通常是去中心化的，并被分割成多个具有共同目标但规模、策略和可操作目标各不相同的集体，现有的大部分ACA文献主要关注单一集体的情景。在本文中，我们提出了首个适用于多个集体共同作用于同一系统的ACA理论框架。特别是，我们关注分类中的集体行动，研究多个集体如何植入信号，即如何偏导分类器使其学习特征集改版与选定的目标类集之间的关联。我们定量分析了集体规模和目标一致性在其中的角色和交互作用。我们的框架不仅补充了先前的实证结果，也为全面处理多个集体的ACA提供了途径。 

---
# Hybrid Deep Searcher: Integrating Parallel and Sequential Search Reasoning 

**Title (ZH)**: 混合深度搜索器：整合并行和序列搜索推理 

**Authors**: Dayoon Ko, Jihyuk Kim, Haeju Park, Sohyeon Kim, Dahyun Lee, Yongrae Jo, Gunhee Kim, Moontae Lee, Kyungjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.19113)  

**Abstract**: Large reasoning models (LRMs) have demonstrated strong performance in complex, multi-step reasoning tasks. Existing methods enhance LRMs by sequentially integrating external knowledge retrieval; models iteratively generate queries, retrieve external information, and progressively reason over this information. However, purely sequential querying increases inference latency and context length, diminishing coherence and potentially reducing accuracy. To address these limitations, we introduce HDS-QA (Hybrid Deep Search QA), a synthetic dataset automatically generated from Natural Questions, explicitly designed to train LRMs to distinguish parallelizable from sequential queries. HDS-QA comprises hybrid-hop questions that combine parallelizable independent subqueries (executable simultaneously) and sequentially dependent subqueries (requiring step-by-step resolution), along with synthetic reasoning-querying-retrieval paths involving parallel queries. We fine-tune an LRM using HDS-QA, naming the model HybridDeepSearcher, which outperforms state-of-the-art baselines across multiple benchmarks, notably achieving +15.9 and +11.5 F1 on FanOutQA and a subset of BrowseComp, respectively, both requiring comprehensive and exhaustive search. Experimental results highlight two key advantages: HybridDeepSearcher reaches comparable accuracy with fewer search turns, significantly reducing inference latency, and it effectively scales as more turns are permitted. These results demonstrate the efficiency, scalability, and effectiveness of explicitly training LRMs to leverage hybrid parallel and sequential querying. 

**Abstract (ZH)**: 大型推理模型（LRMs）在复杂多步推理任务中展现了强大的性能。现有的方法通过顺序集成外部知识检索来增强LRMs；模型迭代生成查询、检索外部信息，并逐步推理。然而，纯粹顺序查询增加了推理延迟和上下文长度，降低了连贯性并可能降低准确性。为了解决这些限制，我们引入了HDS-QA（混合深度搜索QA），该数据集从Natural Questions自动生成，明确设计用于训练LRMs以区分可并行化查询和顺序查询。HDS-QA包含混合跳查询，这些查询结合了可并行执行的独立子查询（可以同时执行）和需要逐步解决的顺序依赖子查询（逐步推理），以及涉及并行查询的合成推理-查询-检索路径。我们使用HDS-QA微调一个LRM，命名该模型为HybridDeepSearcher，它在多个基准测试中均优于最先进的基线模型，特别是在FanOutQA和BrowseComp的部分子集上，分别提高了15.9%和11.5%的F1得分，这两个任务都要求进行全面和详尽的搜索。实验结果突显了两个关键优势：HybridDeepSearcher在较少的搜索轮次中达到与最佳的准确性，显著减少了推理延迟，并随着允许更多轮次的有效扩展。这些结果展示了明确训练LRMs以利用混合并行和顺序查询的效率、可扩展性和有效性。 

---
# MAB Optimizer for Estimating Math Question Difficulty via Inverse CV without NLP 

**Title (ZH)**: 基于逆CV的无需NLP的数学试题难度估计的MAB优化器 

**Authors**: Surajit Das, Gourav Roy, Aleksei Eliseev, Ram Kumar Rajendran  

**Link**: [PDF](https://arxiv.org/pdf/2508.19014)  

**Abstract**: The evolution of technology and education is driving the emergence of Intelligent & Au- tonomous Tutoring Systems (IATS), where objective and domain-agnostic methods for determining question difficulty are essential. Traditional human labeling is subjective, and existing NLP-based ap- proaches fail in symbolic domains like algebra. This study introduces the Approach of Passive Measures among Educands (APME), a reinforcement learning-based Multi-Armed Bandit (MAB) framework that estimates difficulty solely from solver performance data- marks obtained and time taken without re- quiring linguistic features or expert labels. By leveraging the inverse coefficient of variation as a risk- adjusted metric, the model provides an explainable and scalable mechanism for adaptive assessment. Empirical validation was conducted on three heterogeneous datasets. Across these diverse con- texts, the model achieved an average R2 of 0.9213 and an average RMSE of 0.0584, confirming its robustness, accuracy, and adaptability to different educational levels and assessment formats. Com- pared with baseline approaches-such as regression-based, NLP-driven, and IRT models-the proposed framework consistently outperformed alternatives, particularly in purely symbolic domains. The findings highlight that (i) item heterogeneity strongly influences perceived difficulty, and (ii) vari- ance in solver outcomes is as critical as mean performance for adaptive allocation. Pedagogically, the model aligns with Vygotskys Zone of Proximal Development by identifying tasks that balance challenge and attainability, supporting motivation while minimizing disengagement. This domain-agnostic, self- supervised approach advances difficulty tagging in IATS and can be extended beyond algebra wherever solver interaction data is available 

**Abstract (ZH)**: 智能自主辅导系统中技术与教育的进步推动了客观和领域无关的问题难度确定方法的出现，其中客观和领域无关的方法对于确定问题难度至关重要。传统的手动标注具有主观性，现有的基于自然语言处理的方法在代数等符号领域表现不佳。本研究引入了 Educands 的被动度量方法（APME），这是一种基于强化学习的多臂 bandit（MAB）框架，仅通过解题性能数据（正确率和解题时间）来估算难度，无需使用语言特征或专家标签。通过利用变异系数的逆值作为风险调整指标，该模型提供了一个可解释且可扩展的自适应评估机制。在三个异质数据集上进行了实证验证。在这些多样的背景下，该模型实现了平均 $R^2$ 为 0.9213 和平均 RMSE 为 0.0584，证实了其稳健性、准确性和对不同教育水平和评估格式的适应性。与基准方法（如基于回归、基于自然语言处理和项目反应理论模型）相比，所提出框架在纯符号域中表现更优。研究结果表明：（i）项目异质性强烈影响感知难度，（ii）解题结果的方差与平均表现一样关键，对于自适应分配至关重要。从教学角度来看，该模型与维果茨基的最近发展区相一致，通过识别既具有挑战性又可行的任务来促进动机，同时减少脱钩现象。这种领域无关的自我监督方法推动了 IATS 中难度标签的进步，并可以在任何有解题交互数据的领域进一步拓展。 

---
# Novel Approaches to Artificial Intelligence Development Based on the Nearest Neighbor Method 

**Title (ZH)**: 基于最近邻方法的新型人工智能开发方法 

**Authors**: I.I. Priezzhev, D.A. Danko, A.V. Shubin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18953)  

**Abstract**: Modern neural network technologies, including large language models, have achieved remarkable success in various applied artificial intelligence applications, however, they face a range of fundamental limitations. Among them are hallucination effects, high computational complexity of training and inference, costly fine-tuning, and catastrophic forgetting issues. These limitations significantly hinder the use of neural networks in critical areas such as medicine, industrial process management, and scientific research. This article proposes an alternative approach based on the nearest neighbors method with hierarchical clustering structures. Employing the k-nearest neighbors algorithm significantly reduces or completely eliminates hallucination effects while simplifying model expansion and fine-tuning without the need for retraining the entire network. To overcome the high computational load of the k-nearest neighbors method, the paper proposes using tree-like data structures based on Kohonen self-organizing maps, thereby greatly accelerating nearest neighbor searches. Tests conducted on handwritten digit recognition and simple subtitle translation tasks confirmed the effectiveness of the proposed approach. With only a slight reduction in accuracy, the nearest neighbor search time was reduced hundreds of times compared to exhaustive search methods. The proposed method features transparency and interpretability, closely aligns with human cognitive mechanisms, and demonstrates potential for extensive use in tasks requiring high reliability and explainable results. 

**Abstract (ZH)**: 现代神经网络技术，包括大型语言模型，在各种应用人工智能领域取得了显著成功，但面临一系列基本限制。这些问题包括幻觉效应、训练和推理的高计算复杂性、精细调整的高成本以及灾难性遗忘问题。这些限制显著阻碍了神经网络在医学、工业过程管理和科学研究等关键领域的应用。本文提出了一种基于层次聚类结构的最近邻方法的替代方案。利用K近邻算法显著减少了或完全消除了幻觉效应，简化了模型扩展和调整，无需重新训练整个网络。为克服K近邻方法的高计算负载，论文提出使用基于Kohonen自组织映射的树形数据结构，从而大大加速了最近邻搜索。在手写数字识别和简单字幕翻译任务上的测试证实了所提方法的有效性。与穷举搜索方法相比，最近邻搜索时间减少了数百倍，且精度仅略有降低。所提出的方法具备透明性和可解释性，与人类认知机制紧密契合，并在需要高可靠性和可解释结果的任务中具有广泛的潜在应用价值。 

---
# Who Is Lagging Behind: Profiling Student Behaviors with Graph-Level Encoding in Curriculum-Based Online Learning Systems 

**Title (ZH)**: 基于课程的在线学习系统中学生行为的图级别编码剖析 

**Authors**: Qian Xiao, Conn Breathnach, Ioana Ghergulescu, Conor O'Sullivan, Keith Johnston, Vincent Wade  

**Link**: [PDF](https://arxiv.org/pdf/2508.18925)  

**Abstract**: The surge in the adoption of Intelligent Tutoring Systems (ITSs) in education, while being integral to curriculum- based learning, can inadvertently exacerbate performance gaps. To address this problem, student profiling becomes crucial for tracking progress, identifying struggling students, and alleviating disparities among students. Such profiling requires measuring student behaviors and performance across different aspects, such as content coverage, learning intensity, and proficiency in different concepts within a learning topic.
In this study, we introduce CTGraph, a graph-level repre- sentation learning approach to profile learner behaviors and performance in a self-supervised manner. Our experiments demonstrate that CTGraph can provide a holistic view of student learning journeys, accounting for different aspects of student behaviors and performance, as well as variations in their learning paths as aligned to the curriculum structure. We also show that our approach can identify struggling students and provide comparative analysis of diverse groups to pinpoint when and where students are struggling. As such, our approach opens more opportunities to empower educators with rich insights into student learning journeys and paves the way for more targeted interventions. 

**Abstract (ZH)**: 智能辅导系统在教育中的广泛应用虽然对基于课程的学习至关重要，但可能会无意中加剧学习成绩差距。为解决这一问题，对学生进行多层次 profiling 成为关键，以追踪学习进展、识别需要帮助的学生，并缓解学生之间的成绩差异。这种 profiling 需要衡量学生在不同方面的行为和表现，如内容覆盖、学习强度以及不同概念的掌握程度。

在本研究中，我们引入了 CTGraph，这是一种图级自监督表示学习方法，用于 profiling 学生的学习行为和表现。实验结果表明，CTGraph 能够提供对学生学习路径的全面视角，考虑到学生行为和表现的不同方面，以及与课程结构相一致的学习路径变化。我们还展示了该方法能够识别需要帮助的学生，并对不同群体进行比较分析，以确定学生面临困难的具体时间和地点。因此，该方法为教育者提供了丰富的学生学习路径洞察，并为更有针对性的干预措施铺平了道路。 

---
# AniME: Adaptive Multi-Agent Planning for Long Animation Generation 

**Title (ZH)**: AniME: 自适应多代理规划长动画生成 

**Authors**: Lisai Zhang, Baohan Xu, Siqian Yang, Mingyu Yin, Jing Liu, Chao Xu, Siqi Wang, Yidi Wu, Yuxin Hong, Zihao Zhang, Yanzhang Liang, Yudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18781)  

**Abstract**: We present AniME, a director-oriented multi-agent system for automated long-form anime production, covering the full workflow from a story to the final video. The director agent keeps a global memory for the whole workflow, and coordinates several downstream specialized agents. By integrating customized Model Context Protocol (MCP) with downstream model instruction, the specialized agent adaptively selects control conditions for diverse sub-tasks. AniME produces cinematic animation with consistent characters and synchronized audio visual elements, offering a scalable solution for AI-driven anime creation. 

**Abstract (ZH)**: 基于导演导向的多智能体系统AniME及其在自动化长篇动画生产中的应用 

---
# Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models 

**Title (ZH)**: 明知故犯地回答无法回答的问题：分析并缓解大型推理模型中的回避故障 

**Authors**: Yi Liu, Xiangyu Liu, Zequn Sun, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18760)  

**Abstract**: Large reasoning models (LRMs) have shown remarkable progress on complex reasoning tasks. However, some questions posed to LRMs are inherently unanswerable, such as math problems lacking sufficient conditions. We find that LRMs continually fail to provide appropriate abstentions when confronted with these unanswerable questions. In this paper, we systematically analyze, investigate, and resolve this issue for trustworthy AI. We first conduct a detailed analysis of the distinct response behaviors of LRMs when facing unanswerable questions. Then, we show that LRMs possess sufficient cognitive capabilities to recognize the flaws in these questions. However, they fail to exhibit appropriate abstention behavior, revealing a misalignment between their internal cognition and external response. Finally, to resolve this issue, we propose a lightweight, two-stage method that combines cognitive monitoring with inference-time intervention. Experimental results demonstrate that our method significantly improves the abstention rate while maintaining the overall reasoning performance. 

**Abstract (ZH)**: 大型推理模型在处理不可回答问题时表现出明显的行为偏差：系统分析与解决策略 

---
# Stabilizing Open-Set Test-Time Adaptation via Primary-Auxiliary Filtering and Knowledge-Integrated Prediction 

**Title (ZH)**: 通过主辅助筛选和知识整合预测稳定开放集测试时适应 

**Authors**: Byung-Joon Lee, Jin-Seop Lee, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.18751)  

**Abstract**: Deep neural networks demonstrate strong performance under aligned training-test distributions. However, real-world test data often exhibit domain shifts. Test-Time Adaptation (TTA) addresses this challenge by adapting the model to test data during inference. While most TTA studies assume that the training and test data share the same class set (closed-set TTA), real-world scenarios often involve open-set data (open-set TTA), which can degrade closed-set accuracy. A recent study showed that identifying open-set data during adaptation and maximizing its entropy is an effective solution. However, the previous method relies on the source model for filtering, resulting in suboptimal filtering accuracy on domain-shifted test data. In contrast, we found that the adapting model, which learns domain knowledge from noisy test streams, tends to be unstable and leads to error accumulation when used for filtering. To address this problem, we propose Primary-Auxiliary Filtering (PAF), which employs an auxiliary filter to validate data filtered by the primary filter. Furthermore, we propose Knowledge-Integrated Prediction (KIP), which calibrates the outputs of the adapting model, EMA model, and source model to integrate their complementary knowledge for OSTTA. We validate our approach across diverse closed-set and open-set datasets. Our method enhances both closed-set accuracy and open-set discrimination over existing methods. The code is available at this https URL . 

**Abstract (ZH)**: 深度神经网络在对齐的训练-测试分布下表现出色。然而，现实世界的测试数据通常会表现出领域偏移。测试时适应（TTA）通过在推断过程中调整模型来应对这一挑战。尽管大多数TTA研究假定训练和测试数据共享相同的类别集（闭集TTA），但现实中的场景往往涉及开放集数据（开集TTA），这会降低闭集准确率。最近的研究表明，在调整过程中识别开放集数据并最大化其熵是一种有效的解决方案。然而，之前的方法依赖于源模型进行过滤，导致在领域偏移的测试数据上过滤准确率不佳。相比之下，我们发现学习了来自噪声测试流领域知识的适应模型，在用于过滤时容易不稳定并导致误差累积。为了解决这一问题，我们提出了主辅助过滤（PAF），它采用辅助过滤器验证由主过滤器筛选的数据。此外，我们提出了知识整合预测（KIP），用于校准适应模型、EMA模型和源模型的输出，以整合它们的互补知识进行开集自适应测试（OSTTA）。我们在多种闭集和开集数据集上验证了该方法。我们的方法在闭集准确率和开集辨别能力上都优于现有方法。该代码可从此链接访问。 

---
# eSkinHealth: A Multimodal Dataset for Neglected Tropical Skin Diseases 

**Title (ZH)**: eSkinHealth：一种用于忽视性热带皮肤病的多模态数据集 

**Authors**: Janet Wang, Xin Hu, Yunbei Zhang, Diabate Almamy, Vagamon Bamba, Konan Amos Sébastien Koffi, Yao Koffi Aubin, Zhengming Ding, Jihun Hamm, Rie R. Yotsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18608)  

**Abstract**: Skin Neglected Tropical Diseases (NTDs) impose severe health and socioeconomic burdens in impoverished tropical communities. Yet, advancements in AI-driven diagnostic support are hindered by data scarcity, particularly for underrepresented populations and rare manifestations of NTDs. Existing dermatological datasets often lack the demographic and disease spectrum crucial for developing reliable recognition models of NTDs. To address this, we introduce eSkinHealth, a novel dermatological dataset collected on-site in Côte d'Ivoire and Ghana. Specifically, eSkinHealth contains 5,623 images from 1,639 cases and encompasses 47 skin diseases, focusing uniquely on skin NTDs and rare conditions among West African populations. We further propose an AI-expert collaboration paradigm to implement foundation language and segmentation models for efficient generation of multimodal annotations, under dermatologists' guidance. In addition to patient metadata and diagnosis labels, eSkinHealth also includes semantic lesion masks, instance-specific visual captions, and clinical concepts. Overall, our work provides a valuable new resource and a scalable annotation framework, aiming to catalyze the development of more equitable, accurate, and interpretable AI tools for global dermatology. 

**Abstract (ZH)**: 皮肤 Neglected Tropical Diseases (NTDs) 对贫困热带社区的健康和社会经济造成严重负担。然而，AI 驱动的诊断支持的进步受到数据稀缺性的阻碍，尤其是在未代表性人群和 NTDOs 罕见表现形式的数据方面。现有的皮肤病数据集往往缺乏开发 NTDOs 可靠识别模型所需的 démographic 和疾病谱系信息。为了解决这一问题，我们介绍了 eSkinHealth，一个在科特迪瓦和加纳现场收集的新型皮肤病数据集。具体而言，eSkinHealth 包含 5,623 张图片，涉及 1,639 个病例，并涵盖了 47 种皮肤疾病，重点关注西非人群中皮肤 NTDs 和罕见状况。我们进一步提出了一种 AI 专家合作范式，以在皮肤科医生的指导下实现基础语言和分割模型，并高效生成多模态标注。除了患者元数据和诊断标签外，eSkinHealth 还包括语义病变掩模、实例特定的视觉说明以及临床概念。总体而言，我们的工作提供了一个有价值的新型资源和可扩展的标注框架，旨在促进更公平、更准确和更具可解释性的 AI 工具在国际皮肤科中的发展。 

---
# SchemaCoder: Automatic Log Schema Extraction Coder with Residual Q-Tree Boosting 

**Title (ZH)**: SchemaCoder: 基于残差Q-树增强的自动日志模式提取编码器 

**Authors**: Lily Jiaxin Wan, Chia-Tung Ho, Rongjian Liang, Cunxi Yu, Deming Chen, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.18554)  

**Abstract**: Log schema extraction is the process of deriving human-readable templates from massive volumes of log data, which is essential yet notoriously labor-intensive. Recent studies have attempted to streamline this task by leveraging Large Language Models (LLMs) for automated schema extraction. However, existing methods invariably rely on predefined regular expressions, necessitating human domain expertise and severely limiting productivity gains. To fundamentally address this limitation, we introduce SchemaCoder, the first fully automated schema extraction framework applicable to a wide range of log file formats without requiring human customization within the flow. At its core, SchemaCoder features a novel Residual Question-Tree (Q-Tree) Boosting mechanism that iteratively refines schema extraction through targeted, adaptive queries driven by LLMs. Particularly, our method partitions logs into semantic chunks via context-bounded segmentation, selects representative patterns using embedding-based sampling, and generates schema code through hierarchical Q-Tree-driven LLM queries, iteratively refined by our textual-residual evolutionary optimizer and residual boosting. Experimental validation demonstrates SchemaCoder's superiority on the widely-used LogHub-2.0 benchmark, achieving an average improvement of 21.3% over state-of-the-arts. 

**Abstract (ZH)**: 日志模式提取是从大量日志数据中推导出人类可读模板的过程，虽然至关重要，但一直Known to be labor-intensive. 最近的研究试图通过利用大型语言模型（LLMs）来简化这一任务，实现自动模式提取。然而，现有方法不可避免地依赖预定义的正则表达式，这需要人类领域的专业知识，并严重限制了生产力的提升。为根本解决这一限制，我们引入了SchemaCoder，这是第一个无需人在流程中进行定制即可应用于多种日志文件格式的全自动模式提取框架。核心上，SchemaCoder 采用了一种新颖的残差问答树（Q-Tree）增强机制，通过LLMs驱动的目标导向、自适应查询逐步优化模式提取。特别地，我们的方法通过基于上下文的分段将日志分割为语义片段，利用嵌入式采样选择代表模式，并通过层次化的Q-Tree驱动LKM查询生成模式代码，该代码通过我们基于文本的残差进化优化器和残差增强逐步精炼。实验验证表明，SchemaCoder 在广泛使用的LogHub-2.0基准测试中优于最先进的方法，平均改进了21.3%。 

---
# Generic Guard AI in Stealth Game with Composite Potential Fields 

**Title (ZH)**: 隐形游戏中基于复合潜在场的通用守卫AI 

**Authors**: Kaijie Xu, Clark Verbrugge  

**Link**: [PDF](https://arxiv.org/pdf/2508.18527)  

**Abstract**: Guard patrol behavior is central to the immersion and strategic depth of stealth games, while most existing systems rely on hand-crafted routes or specialized logic that struggle to balance coverage efficiency and responsive pursuit with believable naturalness. We propose a generic, fully explainable, training-free framework that integrates global knowledge and local information via Composite Potential Fields, combining three interpretable maps-Information, Confidence, and Connectivity-into a single kernel-filtered decision criterion. Our parametric, designer-driven approach requires only a handful of decay and weight parameters-no retraining-to smoothly adapt across both occupancy-grid and NavMesh-partition abstractions. We evaluate on five representative game maps, two player-control policies, and five guard modes, confirming that our method outperforms classical baseline methods in both capture efficiency and patrol naturalness. Finally, we show how common stealth mechanics-distractions and environmental elements-integrate naturally into our framework as sub modules, enabling rapid prototyping of rich, dynamic, and responsive guard behaviors. 

**Abstract (ZH)**: 基于综合潜在场的通用可解释守卫巡逻行为框架 

---
# Symmetry-Invariant Novelty Heuristics via Unsupervised Weisfeiler-Leman Features 

**Title (ZH)**: 通过无监督Weisfeiler-Leman特征的不变对称新颖性启发式方法 

**Authors**: Dillon Z. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18520)  

**Abstract**: Novelty heuristics aid heuristic search by exploring states that exhibit novel atoms. However, novelty heuristics are not symmetry invariant and hence may sometimes lead to redundant exploration. In this preliminary report, we propose to use Weisfeiler-Leman Features for planning (WLFs) in place of atoms for detecting novelty. WLFs are recently introduced features for learning domain-dependent heuristics for generalised planning problems. We explore an unsupervised usage of WLFs for synthesising lifted, domain-independent novelty heuristics that are invariant to symmetric states. Experiments on the classical International Planning Competition and Hard To Ground benchmark suites yield promising results for novelty heuristics synthesised from WLFs. 

**Abstract (ZH)**: 新颖性特征辅助启发式搜索通过探索表现出新颖原子的状态。然而，新颖性特征不是对称不变的，因此有时可能导致冗余探索。在本初步报告中，我们提出使用Weisfeiler-Leman特征用于规划（WLFs）代替原子来检测新颖性。WLFs是最近引入的学习通用规划问题领域依赖启发式的特征。我们探索了WLFs的无监督使用方式，以合成对称状态不变的提升的、领域无关的新颖性启发式。实验结果表明，基于WLFs合成的新颖性启发式在经典国际规划竞赛和难于地面化基准套件中表现出令人鼓舞的结果。 

---
# Weisfeiler-Leman Features for Planning: A 1,000,000 Sample Size Hyperparameter Study 

**Title (ZH)**: Weisfeiler-Leman特征在规划中的应用：超参数研究（基于1,000,000个样本大小） 

**Authors**: Dillon Z. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18515)  

**Abstract**: Weisfeiler-Leman Features (WLFs) are a recently introduced classical machine learning tool for learning to plan and search. They have been shown to be both theoretically and empirically superior to existing deep learning approaches for learning value functions for search in symbolic planning. In this paper, we introduce new WLF hyperparameters and study their various tradeoffs and effects. We utilise the efficiency of WLFs and run planning experiments on single core CPUs with a sample size of 1,000,000 to understand the effect of hyperparameters on training and planning. Our experimental analysis show that there is a robust and best set of hyperparameters for WLFs across the tested planning domains. We find that the best WLF hyperparameters for learning heuristic functions minimise execution time rather than maximise model expressivity. We further statistically analyse and observe no significant correlation between training and planning metrics. 

**Abstract (ZH)**: Weisfeiler-Leman 特征 (WLFs) 是一种最近引入的经典机器学习工具，用于学习规划和搜索。它们已被证明在符号规划中的搜索价值函数学习方面理论上和实验上都优于现有的深度学习方法。在本文中，我们引入了新的 WLF 超参数，并研究了它们的各种权衡和影响。我们利用 WLF 的高效性，在单核 CPU 上运行规划实验，样本量为 1,000,000，以了解超参数对训练和规划的影响。我们的实验分析表明，在测试的所有规划领域中，存在一套稳健且最优的 WLF 超参数。我们发现，最佳的 WLF 超参数用于学习启发式函数时，旨在最小化执行时间而非最大化模型表达能力。此外，我们进行的统计分析未发现训练和规划指标之间存在显著的相关性。 

---
# Information Templates: A New Paradigm for Intelligent Active Feature Acquisition 

**Title (ZH)**: 信息模板：智能主动特征获取的新范式 

**Authors**: Hung-Tien Huang, Dzung Dinh, Junier B. Oliva  

**Link**: [PDF](https://arxiv.org/pdf/2508.18380)  

**Abstract**: Active feature acquisition (AFA) is an instance-adaptive paradigm in which, at test time, a policy sequentially chooses which features to acquire (at a cost) before predicting. Existing approaches either train reinforcement learning (RL) policies, which deal with a difficult MDP, or greedy policies that cannot account for the joint informativeness of features or require knowledge about the underlying data distribution. To overcome this, we propose Template-based AFA (TAFA), a non-greedy framework that learns a small library of feature templates--a set of features that are jointly informative--and uses this library of templates to guide the next feature acquisitions. Through identifying feature templates, the proposed framework not only significantly reduces the action space considered by the policy but also alleviates the need to estimate the underlying data distribution. Extensive experiments on synthetic and real-world datasets show that TAFA outperforms the existing state-of-the-art baselines while achieving lower overall acquisition cost and computation. 

**Abstract (ZH)**: 基于模板的活性特征获取（Template-based Active Feature Acquisition, TAFA） 

---
# Interpolating Speaker Identities in Embedding Space for Data Expansion 

**Title (ZH)**: 在嵌入空间中插值说话人身份以扩展数据集 

**Authors**: Tianchi Liu, Ruijie Tao, Qiongqiong Wang, Yidi Jiang, Hardik B. Sailor, Ke Zhang, Jingru Lin, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.19210)  

**Abstract**: The success of deep learning-based speaker verification systems is largely attributed to access to large-scale and diverse speaker identity data. However, collecting data from more identities is expensive, challenging, and often limited by privacy concerns. To address this limitation, we propose INSIDE (Interpolating Speaker Identities in Embedding Space), a novel data expansion method that synthesizes new speaker identities by interpolating between existing speaker embeddings. Specifically, we select pairs of nearby speaker embeddings from a pretrained speaker embedding space and compute intermediate embeddings using spherical linear interpolation. These interpolated embeddings are then fed to a text-to-speech system to generate corresponding speech waveforms. The resulting data is combined with the original dataset to train downstream models. Experiments show that models trained with INSIDE-expanded data outperform those trained only on real data, achieving 3.06\% to 5.24\% relative improvements. While INSIDE is primarily designed for speaker verification, we also validate its effectiveness on gender classification, where it yields a 13.44\% relative improvement. Moreover, INSIDE is compatible with other augmentation techniques and can serve as a flexible, scalable addition to existing training pipelines. 

**Abstract (ZH)**: 基于深度学习的说话人验证系统的成功很大程度上归因于能够获取大规模和多样的说话人身份数据。然而，从更多身份收集数据是昂贵的、具有挑战性的，并且通常受到隐私问题的限制。为了解决这一限制，我们提出了INSIDE（在嵌入空间中插值说话人身份），一种通过在现有说话人嵌入之间进行插值合成新说话人身份的新数据扩展方法。具体来说，我们从预训练的说话人嵌入空间中选择附近的说话人嵌入对，并使用球面线性插值计算中间嵌入。这些插值嵌入随后被输入到文本到语音系统中，生成相应的语音波形。生成的数据与原始数据集结合以训练下游模型。实验表明，使用INSIDE扩展数据训练的模型优于仅使用真实数据训练的模型，相对改进幅度在3.06%到5.24%之间。尽管INSIDE主要用于说话人验证，我们也在性别分类上验证了它的有效性，相对改进幅度为13.44%。此外，INSIDE与其它增强技术兼容，并可作为现有训练管道的灵活、可扩展补充。 

---
# Emotions as Ambiguity-aware Ordinal Representations 

**Title (ZH)**: 情绪作为含模糊性的序数表示 

**Authors**: Jingyao Wu, Matthew Barthet, David Melhart, Georgios N. Yannakakis  

**Link**: [PDF](https://arxiv.org/pdf/2508.19193)  

**Abstract**: Emotions are inherently ambiguous and dynamic phenomena, yet existing continuous emotion recognition approaches either ignore their ambiguity or treat ambiguity as an independent and static variable over time. Motivated by this gap in the literature, in this paper we introduce \emph{ambiguity-aware ordinal} emotion representations, a novel framework that captures both the ambiguity present in emotion annotation and the inherent temporal dynamics of emotional traces. Specifically, we propose approaches that model emotion ambiguity through its rate of change. We evaluate our framework on two affective corpora -- RECOLA and GameVibe -- testing our proposed approaches on both bounded (arousal, valence) and unbounded (engagement) continuous traces. Our results demonstrate that ordinal representations outperform conventional ambiguity-aware models on unbounded labels, achieving the highest Concordance Correlation Coefficient (CCC) and Signed Differential Agreement (SDA) scores, highlighting their effectiveness in modeling the traces' dynamics. For bounded traces, ordinal representations excel in SDA, revealing their superior ability to capture relative changes of annotated emotion traces. 

**Abstract (ZH)**: 情感本质上是含糊且动态的现象，现有连续情感识别方法要么忽视其含糊性，要么将含糊性视为时间上独立且静态的变量。基于文献中的这一缺口，在本文中我们引入了含糊性感知的序数情感表示，这是一种新颖的框架，能够捕捉情感注释中的含糊性以及情感痕迹的内在时间动态。具体地，我们提出了一种通过情感变化率建模情感含糊性的方法。我们在这两个情感语料库（RECOLA和GameVibe）上评估了我们的框架，并测试了我们提出的方法在有界（唤醒度、愉悦度）和无界（参与度）连续痕迹上的效果。我们的结果表明，序数表示在无界标签上优于传统的含糊性感知模型，实现了最高的一致性相关系数（CCC）和符号差分一致性（SDA）得分，突显了其在建模痕迹动态方面的有效性。对于有界痕迹，序数表示在SDA上表现出色，揭示了其在捕捉注释情感痕迹相对变化方面的优越能力。 

---
# SecureV2X: An Efficient and Privacy-Preserving System for Vehicle-to-Everything (V2X) Applications 

**Title (ZH)**: SecureV2X：一种高效且隐私保护的车联网（V2X）应用系统 

**Authors**: Joshua Lee, Ali Arastehfard, Weiran Liu, Xuegang Ban, Yuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.19115)  

**Abstract**: Autonomous driving and V2X technologies have developed rapidly in the past decade, leading to improved safety and efficiency in modern transportation. These systems interact with extensive networks of vehicles, roadside infrastructure, and cloud resources to support their machine learning capabilities. However, the widespread use of machine learning in V2X systems raises issues over the privacy of the data involved. This is particularly concerning for smart-transit and driver safety applications which can implicitly reveal user locations or explicitly disclose medical data such as EEG signals. To resolve these issues, we propose SecureV2X, a scalable, multi-agent system for secure neural network inferences deployed between the server and each vehicle. Under this setting, we study two multi-agent V2X applications: secure drowsiness detection, and secure red-light violation detection. Our system achieves strong performance relative to baselines, and scales efficiently to support a large number of secure computation interactions simultaneously. For instance, SecureV2X is $9.4 \times$ faster, requires $143\times$ fewer computational rounds, and involves $16.6\times$ less communication on drowsiness detection compared to other secure systems. Moreover, it achieves a runtime nearly $100\times$ faster than state-of-the-art benchmarks in object detection tasks for red light violation detection. 

**Abstract (ZH)**: 自主驾驶和V2X技术在过去十年中快速发展，提高了现代交通的安全性和效率。这些系统通过与车辆、路边基础设施和云资源的广泛网络交互，支持其机器学习能力。然而，V2X系统中广泛应用机器学习引发了参与数据的隐私问题。特别是在智能交通和驾驶安全应用中，这些问题尤为令人担忧，因为它们可能会隐式揭示用户的地理位置或显式披露如EEG信号等医疗数据。为了解决这些问题，我们提出SecureV2X，这是一种在服务器与每辆车辆之间部署的可扩展多代理系统，用于安全的神经网络推理。在这一框架下，我们研究了两种多代理V2X应用：安全疲劳检测和安全闯红灯检测。我们的系统在基准系统上表现出强劲性能，并且能够高效扩展以同时支持大量安全计算交互。例如，SecureV2X在疲劳检测中的速度比其他安全系统快9.4倍，计算轮数少143倍，通信量少16.6倍。此外，对于闯红灯检测中的目标检测任务，SecureV2X的运行时间比最先进的基准快近100倍。 

---
# Dynamic Triangulation-Based Graph Rewiring for Graph Neural Networks 

**Title (ZH)**: 基于动态三角化的图重布线方法用于图神经网络 

**Authors**: Hugo Attali, Thomas Papastergiou, Nathalie Pernelle, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2508.19071)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as the leading paradigm for learning over graph-structured data. However, their performance is limited by issues inherent to graph topology, most notably oversquashing and oversmoothing. Recent advances in graph rewiring aim to mitigate these limitations by modifying the graph topology to promote more effective information propagation. In this work, we introduce TRIGON, a novel framework that constructs enriched, non-planar triangulations by learning to select relevant triangles from multiple graph views. By jointly optimizing triangle selection and downstream classification performance, our method produces a rewired graph with markedly improved structural properties such as reduced diameter, increased spectral gap, and lower effective resistance compared to existing rewiring methods. Empirical results demonstrate that TRIGON outperforms state-of-the-art approaches on node classification tasks across a range of homophilic and heterophilic benchmarks. 

**Abstract (ZH)**: 图神经网络（GNNs）已成为图结构数据学习的主导范式。然而，它们的表现受到图拓扑固有问题的限制，尤其是信息过挤压和过平滑。近年来，图重构方法旨在通过修改图拓扑来促进更有效的信息传播，从而缓解这些限制。在本文中，我们提出了TRIGON框架，它通过学习从多个图视图中选择相关三角形来构建丰富的非平面三角化。通过同时优化三角形选择和下游分类性能，我们的方法产生了与现有重构方法相比具有明显改善结构属性（如减小直径、增加谱间隙和降低有效电阻）的重构图。实验结果表明，TRIGON在多种同质性和异质性基准上的节点分类任务中优于现有方法。 

---
# Tackling Federated Unlearning as a Parameter Estimation Problem 

**Title (ZH)**: 解决联邦卸载问题的参数估计方法 

**Authors**: Antonio Balordi, Lorenzo Manini, Fabio Stella, Alessio Merlo  

**Link**: [PDF](https://arxiv.org/pdf/2508.19065)  

**Abstract**: Privacy regulations require the erasure of data from deep learning models. This is a significant challenge that is amplified in Federated Learning, where data remains on clients, making full retraining or coordinated updates often infeasible. This work introduces an efficient Federated Unlearning framework based on information theory, modeling leakage as a parameter estimation problem. Our method uses second-order Hessian information to identify and selectively reset only the parameters most sensitive to the data being forgotten, followed by minimal federated retraining. This model-agnostic approach supports categorical and client unlearning without requiring server access to raw client data after initial information aggregation. Evaluations on benchmark datasets demonstrate strong privacy (MIA success near random, categorical knowledge erased) and high performance (Normalized Accuracy against re-trained benchmarks of $\approx$ 0.9), while aiming for increased efficiency over complete retraining. Furthermore, in a targeted backdoor attack scenario, our framework effectively neutralizes the malicious trigger, restoring model integrity. This offers a practical solution for data forgetting in FL. 

**Abstract (ZH)**: 隐私法规要求从深度学习模型中删除数据。这在联邦学习中是一个重大挑战，因为在联邦学习中数据保留在客户端上，导致全面重新训练或协调更新常常不可行。本文介绍了一种基于信息理论的高效联邦撤销框架，将泄露建模为参数估计问题。该方法利用海森矩阵的二阶信息来识别并选择性重置对被遗忘数据最为敏感的参数，随后进行最少的联邦重新训练。该模型通用的方法支持对分类数据和客户端进行撤销，且无需服务器访问原始客户端数据即可在初始信息聚合之后进行。在基准数据集上的评估表明，该方法具有强大的隐私保护（会员推理成功率接近随机，分类知识被清除）和高性能（与重新训练基准相比的归一化准确率约为0.9），同时旨在比完全重新训练更高效。此外，在针对后门攻击的场景中，该框架有效地中和了恶意触发器，恢复了模型的完整性。这为联邦学习中的数据遗忘提供了一个实用的解决方案。 

---
# Metric Matters: A Formal Evaluation of Similarity Measures in Active Learning for Cyber Threat Intelligence 

**Title (ZH)**: 度量事项：在网络安全威胁情报中的主动学习相似性测度形式评价 

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2508.19019)  

**Abstract**: Advanced Persistent Threats (APTs) pose a severe challenge to cyber defense due to their stealthy behavior and the extreme class imbalance inherent in detection datasets. To address these issues, we propose a novel active learning-based anomaly detection framework that leverages similarity search to iteratively refine the decision space. Built upon an Attention-Based Autoencoder, our approach uses feature-space similarity to identify normal-like and anomaly-like instances, thereby enhancing model robustness with minimal oracle supervision. Crucially, we perform a formal evaluation of various similarity measures to understand their influence on sample selection and anomaly ranking effectiveness. Through experiments on diverse datasets, including DARPA Transparent Computing APT traces, we demonstrate that the choice of similarity metric significantly impacts model convergence, anomaly detection accuracy, and label efficiency. Our results offer actionable insights for selecting similarity functions in active learning pipelines tailored for threat intelligence and cyber defense. 

**Abstract (ZH)**: 高级持续威胁（APTs）由于其隐匿行为和检测数据集中固有的极端类别不平衡，给网络防御带来了严重挑战。为此，我们提出了一种基于主动学习的异常检测框架，该框架利用相似性搜索逐迭代地精炼决策空间。该方法基于注意机制自动编码器，通过特征空间相似性来识别正常样例和异常样例，从而在最少的oracle监督下增强模型的鲁棒性。 crucial地，我们形式化评估了多种相似性度量的影响，以了解其对样本选择和异常排名效果的影响。通过在多种数据集上的实验，包括DARPA透明计算APT踪迹，我们表明相似性度量的选择显著影响模型收敛、异常检测准确性和标签效率。我们的结果为针对威胁智能和网络防御定制的主动学习流水线中相似性函数的选择提供了可操作的见解。 

---
# STDiff: A State Transition Diffusion Framework for Time Series Imputation in Industrial Systems 

**Title (ZH)**: STDiff：工业系统中时间序列插补的状态转换扩散框架 

**Authors**: Gary Simethy, Daniel Ortiz-Arroyo, Petar Durdevic  

**Link**: [PDF](https://arxiv.org/pdf/2508.19011)  

**Abstract**: Most deep learning methods for imputing missing values treat the task as completing patterns within a fixed time window. This assumption often fails in industrial systems, where dynamics are driven by control actions, are highly non-stationary, and can experience long, uninterrupted gaps. We propose STDiff, which reframes imputation as learning how the system evolves from one state to the next. STDiff uses a conditional denoising diffusion model with a causal bias aligned to control theory, generating missing values step-by-step based on the most recent known state and relevant control or environmental inputs. On a public wastewater treatment dataset with simulated missing blocks, STDiff consistently achieves the lowest errors, with its advantage increasing for longer gaps. On a raw industrial dataset with substantial real gaps, it produces trajectories that remain dynamically plausible, in contrast to window-based models that tend to flatten or over-smooth. These results support dynamics-aware, explicitly conditioned imputation as a robust approach for industrial time series, and we discuss computational trade-offs and extensions to broader domains. 

**Abstract (ZH)**: 基于动力学的缺失值填充方法：STDiff及其在工业时间序列中的应用 

---
# GitTaskBench: A Benchmark for Code Agents Solving Real-World Tasks Through Code Repository Leveraging 

**Title (ZH)**: GitTaskBench：代码代理通过代码仓库解决真实世界任务的基准测试 

**Authors**: Ziyi Ni, Huacan Wang, Shuo Zhang, Shuo Lu, Ziyang He, Wang You, Zhenheng Tang, Yuntao Du, Bill Sun, Hongzhang Liu, Sen Hu, Ronghao Chen, Bo Li, Xin Li, Chen Hu, Binxing Jiao, Daxin Jiang, Pin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18993)  

**Abstract**: Beyond scratch coding, exploiting large-scale code repositories (e.g., GitHub) for practical tasks is vital in real-world software development, yet current benchmarks rarely evaluate code agents in such authentic, workflow-driven scenarios. To bridge this gap, we introduce GitTaskBench, a benchmark designed to systematically assess this capability via 54 realistic tasks across 7 modalities and 7 domains. Each task pairs a relevant repository with an automated, human-curated evaluation harness specifying practical success criteria. Beyond measuring execution and task success, we also propose the alpha-value metric to quantify the economic benefit of agent performance, which integrates task success rates, token cost, and average developer salaries. Experiments across three state-of-the-art agent frameworks with multiple advanced LLMs show that leveraging code repositories for complex task solving remains challenging: even the best-performing system, OpenHands+Claude 3.7, solves only 48.15% of tasks. Error analysis attributes over half of failures to seemingly mundane yet critical steps like environment setup and dependency resolution, highlighting the need for more robust workflow management and increased timeout preparedness. By releasing GitTaskBench, we aim to drive progress and attention toward repository-aware code reasoning, execution, and deployment -- moving agents closer to solving complex, end-to-end real-world tasks. The benchmark and code are open-sourced at this https URL. 

**Abstract (ZH)**: 超越从零编码，利用大规模代码仓库（如GitHub）实现实用任务在实际软件开发中的应用对于评估代码代理的能力至关重要，但当前基准测试很少在真实的、工作流程驱动的场景中评估代码代理。为了弥合这一差距，我们引入了GitTaskBench，一个旨在通过涵盖7种模态和7个领域的54项现实任务系统评估这种能力的基准测试。每个任务都配有一个相关代码仓库以及一个自动化的、人工编写的评估框架，规定了实用的成功标准。除了衡量执行和任务成功之外，我们还提出了一种alpha值度量标准，用于量化代理性能的经济效益，该度量标准综合了任务成功率、token成本和平均开发者薪资。针对三个最先进的代理框架以及多个高级语言模型的实验显示，利用代码仓库解决复杂任务仍然具有挑战性：即使性能最佳的系统OpenHands+Claude 3.7也只能解决48.15%的任务。错误分析表明，超过一半的失败归因于环境设置和依赖解析等看似简单的关键步骤，这突显了更健壮的工作流程管理和增加超时准备的必要性。通过发布GitTaskBench，我们旨在推动代码意识的能力评估、执行和部署方面的研究进展，使代理更接近解决复杂的端到端的实际任务。基准测试和代码已开源。 

---
# Interpretable by AI Mother Tongue: Native Symbolic Reasoning in Neural Models 

**Title (ZH)**: AI母语可解析性：神经模型中的本土符号推理 

**Authors**: Hung Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18988)  

**Abstract**: We present a framework where neural models develop an AI Mother Tongue, a native symbolic language that simultaneously supports intuitive reasoning, compositional symbol chains, and inherent interpretability. Unlike post-hoc explanation methods, our approach embeds reasoning directly into the model's representations: symbols capture meaningful semantic patterns, chains trace decision paths, and gated induction mechanisms guide selective focus, yielding transparent yet flexible reasoning. We introduce complementary training objectives to enhance symbol purity and decision sparsity, and employ a sequential specialization strategy to first build broad symbolic competence and then refine intuitive judgments. Experiments on AI tasks demonstrate competitive accuracy alongside verifiable reasoning traces, showing that AI Mother Tongue can serve as a unified mechanism for interpretability, intuition, and symbolic reasoning in neural models. 

**Abstract (ZH)**: 我们提出了一种框架，使神经模型发展出一种人工智能母语，这是一种原生态符号语言，同时支持直观推理、组合符号链和内在可解释性。与事后解释方法不同，我们的方法直接将推理嵌入到模型的表示中：符号捕捉有意义的语义模式，链路追踪决策路径，门控诱导机制引导选择性关注，从而产生透明且灵活的推理。我们引入了辅助训练目标以增强符号纯度和决策稀疏性，并采用序列专业化策略首先构建广泛的符号能力，然后细化直观判断。实验在人工智能任务中展示了与可验证推理轨迹相匹配的竞争准确率，表明人工智能母语可以作为神经模型中可解释性、直观性和符号推理的统一机制。 

---
# PAX-TS: Model-agnostic multi-granular explanations for time series forecasting via localized perturbations 

**Title (ZH)**: PAX-TS：基于局部扰动的通用时序预测多粒度解释方法 

**Authors**: Tim Kreuzer, Jelena Zdravkovic, Panagiotis Papapetrou  

**Link**: [PDF](https://arxiv.org/pdf/2508.18982)  

**Abstract**: Time series forecasting has seen considerable improvement during the last years, with transformer models and large language models driving advancements of the state of the art. Modern forecasting models are generally opaque and do not provide explanations for their forecasts, while well-known post-hoc explainability methods like LIME are not suitable for the forecasting context. We propose PAX-TS, a model-agnostic post-hoc algorithm to explain time series forecasting models and their forecasts. Our method is based on localized input perturbations and results in multi-granular explanations. Further, it is able to characterize cross-channel correlations for multivariate time series forecasts. We clearly outline the algorithmic procedure behind PAX-TS, demonstrate it on a benchmark with 7 algorithms and 10 diverse datasets, compare it with two other state-of-the-art explanation algorithms, and present the different explanation types of the method. We found that the explanations of high-performing and low-performing algorithms differ on the same datasets, highlighting that the explanations of PAX-TS effectively capture a model's behavior. Based on time step correlation matrices resulting from the benchmark, we identify 6 classes of patterns that repeatedly occur across different datasets and algorithms. We found that the patterns are indicators of performance, with noticeable differences in forecasting error between the classes. Lastly, we outline a multivariate example where PAX-TS demonstrates how the forecasting model takes cross-channel correlations into account. With PAX-TS, time series forecasting models' mechanisms can be illustrated in different levels of detail, and its explanations can be used to answer practical questions on forecasts. 

**Abstract (ZH)**: 时间序列预测模型及其预测结果的模型无关后验解释方法：PAX-TS 

---
# The point is the mask: scaling coral reef segmentation with weak supervision 

**Title (ZH)**: 以掩码为中心：通过弱监督扩展珊瑚礁分割规模 

**Authors**: Matteo Contini, Victor Illien, Sylvain Poulain, Serge Bernard, Julien Barde, Sylvain Bonhommeau, Alexis Joly  

**Link**: [PDF](https://arxiv.org/pdf/2508.18958)  

**Abstract**: Monitoring coral reefs at large spatial scales remains an open challenge, essential for assessing ecosystem health and informing conservation efforts. While drone-based aerial imagery offers broad spatial coverage, its limited resolution makes it difficult to reliably distinguish fine-scale classes, such as coral morphotypes. At the same time, obtaining pixel-level annotations over large spatial extents is costly and labor-intensive, limiting the scalability of deep learning-based segmentation methods for aerial imagery. We present a multi-scale weakly supervised semantic segmentation framework that addresses this challenge by transferring fine-scale ecological information from underwater imagery to aerial data. Our method enables large-scale coral reef mapping from drone imagery with minimal manual annotation, combining classification-based supervision, spatial interpolation and self-distillation techniques. We demonstrate the efficacy of the approach, enabling large-area segmentation of coral morphotypes and demonstrating flexibility for integrating new classes. This study presents a scalable, cost-effective methodology for high-resolution reef monitoring, combining low-cost data collection, weakly supervised deep learning and multi-scale remote sensing. 

**Abstract (ZH)**: 在大尺度范围内监控珊瑚礁仍然是一个开放的挑战，对于评估生态系统健康状况和指导保护努力至关重要。虽然基于无人机的航空影像提供了广泛的覆盖范围，但其有限的分辨率使得难以可靠地区分细尺度类别，如珊瑚形态类型。同时，获得大范围的像素级注解既昂贵又劳动密集，限制了基于深度学习的分割方法在航空影像中的可扩展性。我们提出了一种多尺度弱监督语义分割框架，通过将水下影像中的细尺度生态信息转移到航空数据中来应对这一挑战。该方法能够结合基于分类的监督、空间插值和自我精炼技术，从无人机影像中实现大规模珊瑚礁地图绘制，同时减少手动注解量。我们展示了该方法的有效性，能够实现大面积珊瑚形态类型的分割，并展示其对于集成新类别的灵活性。本研究提出了一种可扩展且成本效益高的高分辨率礁区监控方法，结合低成本数据采集、弱监督深度学习和多尺度遥感技术。 

---
# HierCVAE: Hierarchical Attention-Driven Conditional Variational Autoencoders for Multi-Scale Temporal Modeling 

**Title (ZH)**: HierCVAE：基于层次注意力的条件变分自编码器在多尺度时间建模中的应用 

**Authors**: Yao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18922)  

**Abstract**: Temporal modeling in complex systems requires capturing dependencies across multiple time scales while managing inherent uncertainties. We propose HierCVAE, a novel architecture that integrates hierarchical attention mechanisms with conditional variational autoencoders to address these challenges. HierCVAE employs a three-tier attention structure (local, global, cross-temporal) combined with multi-modal condition encoding to capture temporal, statistical, and trend information. The approach incorporates ResFormer blocks in the latent space and provides explicit uncertainty quantification via prediction heads. Through evaluations on energy consumption datasets, HierCVAE demonstrates a 15-40% improvement in prediction accuracy and superior uncertainty calibration compared to state-of-the-art methods, excelling in long-term forecasting and complex multi-variate dependencies. 

**Abstract (ZH)**: 复杂系统中的时间建模需要捕捉多时间尺度上的依赖关系并处理固有的不确定性。我们提出了HierCVAE，一种将分层注意机制与条件变分自编码器相结合的新架构以应对这些挑战。HierCVAE 使用三层注意结构（局部、全局、跨时间）、结合多模态条件编码来捕获时间、统计和趋势信息。该方法在潜在空间中使用 ResFormer 块，并通过预测头显式地提供不确定性量化。通过在能耗数据集上的评估，HierCVAE 在预测准确性和不确定性校准方面分别比最先进的方法提高了 15-40%，尤其在长期预测和复杂多元依赖性方面表现出色。 

---
# Enhancing Model Privacy in Federated Learning with Random Masking and Quantization 

**Title (ZH)**: 提高联邦学习中模型隐私性的随机遮掩与量化方法 

**Authors**: Zhibo Xu, Jianhao Zhu, Jingwen Xu, Changze Lv, Zisu Huang, Xiaohua Wang, Muling Wu, Qi Qian, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18911)  

**Abstract**: Experimental results across various models and tasks demonstrate that our approach not only maintains strong model performance in federated learning settings but also achieves enhanced protection of model parameters compared to baseline methods. 

**Abstract (ZH)**: 实验结果表明，与基线方法相比，本方法不仅在联邦学习设置中保持了较强的模型性能，还在模型参数保护方面取得了增强的效果。 

---
# SegReConcat: A Data Augmentation Method for Voice Anonymization Attack 

**Title (ZH)**: SegReConcat: 一种语音匿名化攻击的数据增强方法 

**Authors**: Ridwan Arefeen, Xiaoxiao Miao, Rong Tong, Aik Beng Ng, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2508.18907)  

**Abstract**: Anonymization of voice seeks to conceal the identity of the speaker while maintaining the utility of speech data. However, residual speaker cues often persist, which pose privacy risks. We propose SegReConcat, a data augmentation method for attacker-side enhancement of automatic speaker verification systems. SegReConcat segments anonymized speech at the word level, rearranges segments using random or similarity-based strategies to disrupt long-term contextual cues, and concatenates them with the original utterance, allowing an attacker to learn source speaker traits from multiple perspectives. The proposed method has been evaluated in the VoicePrivacy Attacker Challenge 2024 framework across seven anonymization systems, SegReConcat improves de-anonymization on five out of seven systems. 

**Abstract (ZH)**: 语音匿名化旨在保护说话人的身份隐私，同时保留语音数据的实用性。但残留的说话人线索仍然存在，带来隐私风险。我们提出了一种SegReConcat数据增强方法，用于增强攻击者侧的自动说话人验证系统。SegReConcat在单词级别对匿名语音进行分段，通过随机或基于相似性的策略重新排列分段，以打断长期的上下文线索，并将这些分段与原始语音片段重新连接，使攻击者能够从多个角度学习源说话人的特征。该方法已在2024年VoicePrivacy攻击者挑战框架下对七种匿名化系统进行了评估，SegReConcat在五种系统中提升了去匿名化效果。 

---
# Distance-informed Neural Processes 

**Title (ZH)**: 距离指导神经过程 

**Authors**: Aishwarya Venkataramanan, Joachim Denzler  

**Link**: [PDF](https://arxiv.org/pdf/2508.18903)  

**Abstract**: We propose the Distance-informed Neural Process (DNP), a novel variant of Neural Processes that improves uncertainty estimation by combining global and distance-aware local latent structures. Standard Neural Processes (NPs) often rely on a global latent variable and struggle with uncertainty calibration and capturing local data dependencies. DNP addresses these limitations by introducing a global latent variable to model task-level variations and a local latent variable to capture input similarity within a distance-preserving latent space. This is achieved through bi-Lipschitz regularization, which bounds distortions in input relationships and encourages the preservation of relative distances in the latent space. This modeling approach allows DNP to produce better-calibrated uncertainty estimates and more effectively distinguish in- from out-of-distribution data. Empirical results demonstrate that DNP achieves strong predictive performance and improved uncertainty calibration across regression and classification tasks. 

**Abstract (ZH)**: Distance-informed Neural Process及其在回归和分类任务中改进的不确定性估算和分布外数据区分能力 

---
# pyFAST: A Modular PyTorch Framework for Time Series Modeling with Multi-source and Sparse Data 

**Title (ZH)**: pyFAST：一种用于多源稀疏数据时间序列建模的模块化PyTorch框架 

**Authors**: Zhijin Wang, Senzhen Wu, Yue Hu, Xiufeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18891)  

**Abstract**: Modern time series analysis demands frameworks that are flexible, efficient, and extensible. However, many existing Python libraries exhibit limitations in modularity and in their native support for irregular, multi-source, or sparse data. We introduce pyFAST, a research-oriented PyTorch framework that explicitly decouples data processing from model computation, fostering a cleaner separation of concerns and facilitating rapid experimentation. Its data engine is engineered for complex scenarios, supporting multi-source loading, protein sequence handling, efficient sequence- and patch-level padding, dynamic normalization, and mask-based modeling for both imputation and forecasting. pyFAST integrates LLM-inspired architectures for the alignment-free fusion of sparse data sources and offers native sparse metrics, specialized loss functions, and flexible exogenous data fusion. Training utilities include batch-based streaming aggregation for evaluation and device synergy to maximize computational efficiency. A comprehensive suite of classical and deep learning models (Linears, CNNs, RNNs, Transformers, and GNNs) is provided within a modular architecture that encourages extension. Released under the MIT license at GitHub, pyFAST provides a compact yet powerful platform for advancing time series research and applications. 

**Abstract (ZH)**: 现代时间序列分析需要灵活、高效且可扩展的框架。然而，许多现有的Python库在模块化和支持不规则、多源或稀疏数据方面存在局限性。我们介绍了一个面向研究的PyTorch框架pyFAST，该框架明确地将数据处理与模型计算解耦，促进清晰的责任分离，并促进快速实验。其数据引擎针对复杂场景设计，支持多源加载、蛋白质序列处理、高效的序列级和patches级填充、动态归一化以及基于掩码的建模，适用于插补和预测。pyFAST集成了受大语言模型启发的架构，用于无对齐融合稀疏数据源，并提供了原生稀疏度量、专业的损失函数和灵活的外生数据融合。训练工具包括基于批次的数据流聚合用于评估和设备协同工作以最大化计算效率。在模块化架构中提供了经典和深度学习模型（线性模型、CNN、RNN、Transformer和GNN），鼓励扩展。pyFAST在GitHub上以MIT许可证发布，提供了一个紧凑而强大的平台，用于推动时间序列研究和应用。 

---
# HAEPO: History-Aggregated Exploratory Policy Optimization 

**Title (ZH)**: HAEPO: 历史聚合探索性策略优化 

**Authors**: Gaurish Trivedi, Alakh Sharma, Kartikey Singh Bhandari, Dhruv Kumar, Pratik Narang, Jagat Sesh Challa  

**Link**: [PDF](https://arxiv.org/pdf/2508.18884)  

**Abstract**: Exploration is essential in modern learning, from reinforcement learning environments with small neural policies to large language models (LLMs). Existing work, such as DPO, leverages full sequence log-likelihoods to capture an entire trajectory of the model's decisions, while methods like GRPO aggregate per-token ratios into a trajectory-level update. However, both often limit exploration on long-horizon tasks. We introduce History-Aggregated Exploratory Policy Optimization (HAEPO), a history-aware exploratory loss to combat these shortcomings. HAEPO compresses each trajectory into the sum of its logarithmic probabilities (a cumulative logarithmic likelihood), and applies a Plackett-Luce softmax across trajectories to obtain normalized weights proportional to their returns, thus encouraging broader exploration. We add entropy regularization to stabilize the aggressive updates to prevent premature collapse and a soft KL penalty relative to a frozen copy of the previous (reference) policy. Empirically, HAEPO converges fast, explores thoroughly, aligns closely with true rewards, and demonstrates robust learning behavior better or at par with PPO, GRPO, and DPO across diverse tasks. Thus, HAEPO provides a stable and interpretable framework by explicitly leveraging full-trajectory history while balancing exploration and stability. 

**Abstract (ZH)**: 探索对于现代学习至关重要，从具有小神经策略的强化学习环境到大型语言模型（LLMs）。现有的方法，如DPO，利用整个序列的对数似然性来捕捉模型决策的完整轨迹，而像GRPO这样的方法则将每个token的比例聚合为一个轨迹级别的更新。然而，这两种方法往往在长时间任务中限制探索。我们引入了基于历史的探索性策略优化（HAEPO），这是一种具有历史意识的探索性损失，以应对这些不足。HAEPO将每个轨迹压缩为其对数概率的和（累积对数似然），并使用Plackett-Luce softmax在轨迹之间获得归一化的权重，使其与回报成比例，从而促进更广泛的探索。我们添加熵正则化以稳定激进的更新，防止过早坍缩，并相对于冻结的先前（参考）策略应用软KL惩罚。实验证明，HAEPO快速收敛、彻底探索、与真实奖励紧密对齐，并在各种任务中表现出更好的或至少与PPO、GRPO和DPO相当的鲁棒学习行为。因此，HAEPO提供了一个通过明确利用完整轨迹的历史并平衡探索与稳定性而实现稳定性和可解释性的框架。 

---
# Optimization of Latent-Space Compression using Game-Theoretic Techniques for Transformer-Based Vector Search 

**Title (ZH)**: 基于博弈论技术的Transformer基矢量搜索潜在空间压缩优化 

**Authors**: Kushagra Agrawal, Nisharg Nargund, Oishani Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2508.18877)  

**Abstract**: Vector similarity search plays a pivotal role in modern information retrieval systems, especially when powered by transformer-based embeddings. However, the scalability and efficiency of such systems are often hindered by the high dimensionality of latent representations. In this paper, we propose a novel game-theoretic framework for optimizing latent-space compression to enhance both the efficiency and semantic utility of vector search. By modeling the compression strategy as a zero-sum game between retrieval accuracy and storage efficiency, we derive a latent transformation that preserves semantic similarity while reducing redundancy. We benchmark our method against FAISS, a widely-used vector search library, and demonstrate that our approach achieves a significantly higher average similarity (0.9981 vs. 0.5517) and utility (0.8873 vs. 0.5194), albeit with a modest increase in query time. This trade-off highlights the practical value of game-theoretic latent compression in high-utility, transformer-based search applications. The proposed system can be seamlessly integrated into existing LLM pipelines to yield more semantically accurate and computationally efficient retrieval. 

**Abstract (ZH)**: 基于博弈论的潜在空间压缩优化在高效率和语义实用性向量搜索中的应用 

---
# A Survey on Cloud-Edge-Terminal Collaborative Intelligence in AIoT Networks 

**Title (ZH)**: AIoT网络中云-边缘-终端协同智能综述 

**Authors**: Jiaqi Wu, Jing Liu, Yang Liu, Lixu Wang, Zehua Wang, Wei Chen, Zijian Tian, Richard Yu, Victor C.M. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2508.18803)  

**Abstract**: The proliferation of Internet of things (IoT) devices in smart cities, transportation, healthcare, and industrial applications, coupled with the explosive growth of AI-driven services, has increased demands for efficient distributed computing architectures and networks, driving cloud-edge-terminal collaborative intelligence (CETCI) as a fundamental paradigm within the artificial intelligence of things (AIoT) community. With advancements in deep learning, large language models (LLMs), and edge computing, CETCI has made significant progress with emerging AIoT applications, moving beyond isolated layer optimization to deployable collaborative intelligence systems for AIoT (CISAIOT), a practical research focus in AI, distributed computing, and communications. This survey describes foundational architectures, enabling technologies, and scenarios of CETCI paradigms, offering a tutorial-style review for CISAIOT beginners. We systematically analyze architectural components spanning cloud, edge, and terminal layers, examining core technologies including network virtualization, container orchestration, and software-defined networking, while presenting categorizations of collaboration paradigms that cover task offloading, resource allocation, and optimization across heterogeneous infrastructures. Furthermore, we explain intelligent collaboration learning frameworks by reviewing advances in federated learning, distributed deep learning, edge-cloud model evolution, and reinforcement learning-based methods. Finally, we discuss challenges (e.g., scalability, heterogeneity, interoperability) and future trends (e.g., 6G+, agents, quantum computing, digital twin), highlighting how integration of distributed computing and communication can address open issues and guide development of robust, efficient, and secure collaborative AIoT systems. 

**Abstract (ZH)**: 物联网设备在智慧城市、交通、医疗和工业应用中的普及，以及人工智能驱动服务的爆炸性增长，推动了高效分布式计算架构和网络的需求，促使云-边缘-终端协同智能（CETCI）成为物联网人工智能（AIoT）社区的基本范式。随着深度学习、大规模语言模型（LLMs）和边缘计算的进展，CETCI在新兴的AIoT应用中取得了显著进步，超越了孤立层优化，部署了适用于AIoT（CISAIOT）的可实现协同智能系统，这成为分布式计算和通信领域的实际研究重点。本文综述了CETCI范式的基础架构、使能技术和应用场景，提供了一种面向CISAIOT初学者的教学式回顾。我们系统地分析了跨越云、边缘和终端层的架构组件，探讨了核心技术包括网络虚拟化、容器编排和软件定义网络，并呈现了涵盖跨异构基础设施的任务卸载、资源分配和优化的合作范式分类。此外，我们通过回顾联邦学习、分布式深度学习、边缘-云模型演化和基于强化学习的方法，解释了智能协作学习框架。最后，我们讨论了挑战（如可扩展性、异构性和互操作性）和未来趋势（如6G+、智能体、量子计算、数字孪生），强调了分布式计算和通信集成如何解决开放问题并指导健壮、高效和安全的协同AIoT系统的发展。 

---
# EMind: A Foundation Model for Multi-task Electromagnetic Signals Understanding 

**Title (ZH)**: EMind: 多任务电磁信号理解的foundation模型 

**Authors**: Luqing Luo, Wenjin Gui, Yunfei Liu, Ziyue Zhang, Yunxi Zhang, Fengxiang Wang, Zonghao Guo, Zizhi Ma, Xinzhu Liu, Hanxiang He, Jinhai Li, Xin Qiu, Wupeng Xie, Yangang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.18785)  

**Abstract**: Deep understanding of electromagnetic signals is fundamental to dynamic spectrum management, intelligent transportation, autonomous driving and unmanned vehicle perception. The field faces challenges because electromagnetic signals differ greatly from text and images, showing high heterogeneity, strong background noise and complex joint time frequency structure, which prevents existing general models from direct use. Electromagnetic communication and sensing tasks are diverse, current methods lack cross task generalization and transfer efficiency, and the scarcity of large high quality datasets blocks the creation of a truly general multitask learning framework. To overcome these issue, we introduce EMind, an electromagnetic signals foundation model that bridges large scale pretraining and the unique nature of this modality. We build the first unified and largest standardized electromagnetic signal dataset covering multiple signal types and tasks. By exploiting the physical properties of electromagnetic signals, we devise a length adaptive multi-signal packing method and a hardware-aware training strategy that enable efficient use and representation learning from heterogeneous multi-source signals. Experiments show that EMind achieves strong performance and broad generalization across many downstream tasks, moving decisively from task specific models to a unified framework for electromagnetic intelligence. The code is available at: this https URL. 

**Abstract (ZH)**: 深入理解电磁信号是动态频谱管理、智能交通、自主驾驶和无人驾驶车辆感知的基础。由于电磁信号与文本和图像有很大的不同，表现出高度异质性、强烈的背景噪声和复杂的联合时频结构，现有的一般模型难以直接使用。电磁通信和感知任务多样，当前方法缺乏跨任务的一般化和迁移效率，稀缺的大规模高质量数据集阻碍了真正通用的多任务学习框架的创建。为克服这些问题，我们引入了EMind，这是一种连接大规模预训练和该模态独特性质的电磁信号基础模型。我们构建了第一个统一且最大的标准化电磁信号数据集，涵盖了多种信号类型和任务。通过利用电磁信号的物理特性，我们设计了一种长度自适应多信号打包方法和硬件感知训练策略，能够有效利用和学习异质多源信号。实验表明，EMind在许多下游任务中实现了强大的性能和广泛的泛化能力，从专门任务模型迈向了统一的电磁智能框架。代码可访问：这个链接。 

---
# Insights into User Interface Innovations from a Design Thinking Workshop at deRSE25 

**Title (ZH)**: 来自deRSE25设计思维工作坊的用户界面创新洞见 

**Authors**: Maximilian Frank, Simon Lund  

**Link**: [PDF](https://arxiv.org/pdf/2508.18784)  

**Abstract**: Large Language Models have become widely adopted tools due to their versatile capabilities, yet their user interfaces remain limited, often following rigid, linear interaction paradigms. In this paper, we present insights from a design thinking workshop held at the deRSE25 conference aiming at collaboratively developing innovative user interface concepts for LLMs. During the workshop, participants identified common use cases, evaluated the strengths and shortcomings of current LLM interfaces, and created visualizations of new interaction concepts emphasizing flexible context management, dynamic conversation branching, and enhanced mechanisms for user control. We describe how these participant-generated ideas advanced our own whiteboard-based UI approach. The ongoing development of this interface is guided by the human-centered design process - an iterative, user-focused methodology that emphasizes continuous refinement through user feedback. Broader implications for future LLM interface development are discussed, advocating for increased attention to UI innovation grounded in user-centered design principles. 

**Abstract (ZH)**: 大型语言模型由于其多功能能力已经被广泛采纳，但其用户界面仍较为有限，常常遵循僵化的线性交互模式。本文介绍了在deRSE25会议的设计思维工作坊中获得的见解，旨在合作开发创新的大型语言模型用户界面概念。工作坊中，参与者识别了常见的使用场景，评估了当前大型语言模型接口的优势和不足，并创建了全新交互概念的可视化展示，强调灵活的上下文管理、动态对话分支以及增强的用户控制机制。我们描述了这些参与者生成的想法如何推动了我们现有的白板式UI方法的改进。这一接口的持续开发遵循以用户为中心的设计过程——一个迭代、用户导向的方法论，强调通过不断获取用户反馈进行持续优化。讨论了对未来大型语言模型接口开发的更广泛影响，倡导以用户为中心的设计原则为基础的UI创新。 

---
# Long-Term Variability in Physiological-Arousal Relationships for Robust Emotion Estimation 

**Title (ZH)**: 长时变异生理唤醒关系在稳健情绪估计中的作用 

**Authors**: Hiroto Sakimura, Takayuki Nagaya, Tomoki Nishi, Tetsuo Kurahashi, Katsunori Kohda, Nobuhiko Muramoto  

**Link**: [PDF](https://arxiv.org/pdf/2508.18782)  

**Abstract**: Estimating emotional states from physiological signals is a central topic in affective computing and psychophysiology. While many emotion estimation systems implicitly assume a stable relationship between physiological features and subjective affect, this assumption has rarely been tested over long timeframes. This study investigates whether such relationships remain consistent across several months within individuals. We developed a custom measurement system and constructed a longitudinal dataset by collecting physiological signals--including blood volume pulse, electrodermal activity (EDA), skin temperature, and acceleration--along with self-reported emotional states from 24 participants over two three-month periods. Data were collected in naturalistic working environments, allowing analysis of the relationship between physiological features and subjective arousal in everyday contexts. We examined how physiological--arousal relationships evolve over time by using Explainable Boosting Machines (EBMs) to ensure model interpretability. A model trained on 1st-period data showed a 5\% decrease in accuracy when tested on 2nd-period data, indicating long-term variability in physiological--arousal associations. EBM-based comparisons further revealed that while heart rate remained a relatively stable predictor, minimum EDA exhibited substantial individual-level fluctuations between periods. While the number of participants is limited, these findings highlight the need to account for temporal variability in physiological--arousal relationships and suggest that emotion estimation models should be periodically updated -- e.g., every five months -- based on observed shift trends to maintain robust performance over time. 

**Abstract (ZH)**: 从生理信号估计情绪状态是情感计算和生理心理研究中的一个核心课题。虽然许多情绪估计系统隐含假设生理特征与主观情绪之间存在稳定关系，但这一假设在长时间跨度内鲜有测试。本研究调查了这种关系是否能在短时间内保持一致。我们开发了一套定制的测量系统，并通过在24名参与者身上收集包括血容积脉冲、电导率活动（EDA）、皮肤温度和加速度在内的生理信号及其自报告的情绪状态，构建了一个纵向数据集。数据在自然的工作环境中收集，允许在日常情境下分析生理特征与主观唤醒之间的关系。我们使用可解释boosting机（EBM）确保模型的可解释性，研究了生理—唤醒关系随时间的变化。使用EBM比较分析表明，尽管心率相对稳定，但最小EDA在不同时间段内显示出显著的个体波动。尽管参与者数量有限，但这些发现强调了在生理—唤醒关系中考虑时间变异性的必要性，并建议情绪估计模型应根据观察到的趋势定期更新（例如每五个月一次），以保持长期的稳健性能。 

---
# Harnessing Rule-Based Reinforcement Learning for Enhanced Grammatical Error Correction 

**Title (ZH)**: 基于规则的强化学习在语法错误纠正中的应用 

**Authors**: Yilin Li, Xunjian Yin, Yilin Chen, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18780)  

**Abstract**: Grammatical error correction is a significant task in NLP. Traditional methods based on encoder-decoder models have achieved certain success, but the application of LLMs in this field is still underexplored. Current research predominantly relies on supervised fine-tuning to train LLMs to directly generate the corrected sentence, which limits the model's powerful reasoning ability. To address this limitation, we propose a novel framework based on Rule-Based RL. Through experiments on the Chinese datasets, our Rule-Based RL framework achieves \textbf{state-of-the-art }performance, with a notable increase in \textbf{recall}. This result clearly highlights the advantages of using RL to steer LLMs, offering a more controllable and reliable paradigm for future development in GEC. 

**Abstract (ZH)**: 基于规则的强化学习在中文语法错误修正中的应用：一种先进的框架 

---
# Text to Query Plans for Question Answering on Large Tables 

**Title (ZH)**: 文本到查询计划转换供在大型表格上进行问答使用 

**Authors**: Yipeng Zhang, Chen Wang, Yuzhe Zhang, Jacky Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18758)  

**Abstract**: Efficient querying and analysis of large tabular datasets remain significant challenges, especially for users without expertise in programming languages like SQL. Text-to-SQL approaches have shown promising performance on benchmark data; however, they inherit SQL's drawbacks, including inefficiency with large datasets and limited support for complex data analyses beyond basic querying. We propose a novel framework that transforms natural language queries into query plans. Our solution is implemented outside traditional databases, allowing us to support classical SQL commands while avoiding SQL's inherent limitations. Additionally, we enable complex analytical functions, such as principal component analysis and anomaly detection, providing greater flexibility and extensibility than traditional SQL capabilities. We leverage LLMs to iteratively interpret queries and construct operation sequences, addressing computational complexity by incrementally building solutions. By executing operations directly on the data, we overcome context length limitations without requiring the entire dataset to be processed by the model. We validate our framework through experiments on both standard databases and large scientific tables, demonstrating its effectiveness in handling extensive datasets and performing sophisticated data analyses. 

**Abstract (ZH)**: 高效的大型表格数据集查询与分析仍然是重要挑战，尤其对于不熟悉编程语言如SQL的用户。文本到SQL的方法在基准数据集上显示出有前途的性能；然而，它们继承了SQL的局限性，包括处理大型数据集时的低效率以及对基本查询之外的复杂数据分析支持有限。我们提出了一种新型框架，将自然语言查询转化为查询计划。我们的解决方案在传统的数据库之外实现，允许我们支持经典的SQL命令，同时避免SQL固有的局限性。此外，我们还能够支持复杂的分析函数，如主成分分析和异常检测，提供了比传统SQL能力更大的灵活性和可扩展性。我们利用大规模语言模型（LLMs）迭代地解释查询并构建操作序列，通过逐步构建解决方案来解决计算复杂性。通过直接在数据上执行操作，我们克服了上下文长度限制，而不需要将整个数据集发送给模型进行处理。我们通过在标准数据库和大型科学表格上的实验验证了我们框架的有效性，展示了其处理大量数据集和执行复杂数据分析的能力。 

---
# M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations 

**Title (ZH)**: M3HG：多模态、多尺度和多类型节点异质图在对话中情绪原因三元组提取中的应用 

**Authors**: Qiao Liang, Ying Shen, Tiantian Chen, Lin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18740)  

**Abstract**: Emotion Cause Triplet Extraction in Multimodal Conversations (MECTEC) has recently gained significant attention in social media analysis, aiming to extract emotion utterances, cause utterances, and emotion categories simultaneously. However, the scarcity of related datasets, with only one published dataset featuring highly uniform dialogue scenarios, hinders model development in this field. To address this, we introduce MECAD, the first multimodal, multi-scenario MECTEC dataset, comprising 989 conversations from 56 TV series spanning a wide range of dialogue contexts. In addition, existing MECTEC methods fail to explicitly model emotional and causal contexts and neglect the fusion of semantic information at different levels, leading to performance degradation. In this paper, we propose M3HG, a novel model that explicitly captures emotional and causal contexts and effectively fuses contextual information at both inter- and intra-utterance levels via a multimodal heterogeneous graph. Extensive experiments demonstrate the effectiveness of M3HG compared with existing state-of-the-art methods. The codes and dataset are available at this https URL. 

**Abstract (ZH)**: 多模态对话中的情绪原因三元组提取（MECTEC）在社会媒体分析中 Recent Progress and Challenges 

---
# FLAegis: A Two-Layer Defense Framework for Federated Learning Against Poisoning Attacks 

**Title (ZH)**: FLAegis：联邦学习中抵抗投毒攻击的两层防御框架 

**Authors**: Enrique Mármol Campos, Aurora González Vidal, José Luis Hernández Ramos, Antonio Skarmeta  

**Link**: [PDF](https://arxiv.org/pdf/2508.18737)  

**Abstract**: Federated Learning (FL) has become a powerful technique for training Machine Learning (ML) models in a decentralized manner, preserving the privacy of the training datasets involved. However, the decentralized nature of FL limits the visibility of the training process, relying heavily on the honesty of participating clients. This assumption opens the door to malicious third parties, known as Byzantine clients, which can poison the training process by submitting false model updates. Such malicious clients may engage in poisoning attacks, manipulating either the dataset or the model parameters to induce misclassification. In response, this study introduces FLAegis, a two-stage defensive framework designed to identify Byzantine clients and improve the robustness of FL systems. Our approach leverages symbolic time series transformation (SAX) to amplify the differences between benign and malicious models, and spectral clustering, which enables accurate detection of adversarial behavior. Furthermore, we incorporate a robust FFT-based aggregation function as a final layer to mitigate the impact of those Byzantine clients that manage to evade prior defenses. We rigorously evaluate our method against five poisoning attacks, ranging from simple label flipping to adaptive optimization-based strategies. Notably, our approach outperforms state-of-the-art defenses in both detection precision and final model accuracy, maintaining consistently high performance even under strong adversarial conditions. 

**Abstract (ZH)**: 联邦学习（FL）已成为一种强大的技术，用于以去中心化的方式训练机器学习（ML）模型，同时保护参与训练的数据集的隐私。然而，FL的去中心化性质限制了对训练过程的可见性，高度依赖参与客户端的诚信。这一假设为已知为拜占庭客户端的恶意第三方打开了大门，这些客户端可以通过提交虚假模型更新来毒害训练过程。这些恶意客户端可能会发动毒化攻击，操纵数据集或模型参数以诱导误分类。为应对这一挑战，本研究提出FLAegis，这是一种双阶段防御框架，旨在识别拜占庭客户端并提高FL系统的稳健性。我们的方法利用符号时间序列变换（SAX）来放大良性模型和恶意模型之间的差异，并利用谱聚类以准确检测攻击行为。此外，我们引入了基于稳健快速傅里叶变换（FFT）的聚合函数作为最终层，以减轻那些逃避免疫防御的拜占庭客户端的影响。我们严格评估了该方法对五种毒化攻击的性能，这些攻击从简单的标签翻转到基于优化的自适应策略不等。值得注意的是，我们的方法在检测精度和最终模型准确性方面均优于最先进的防御方法，并且在强有力的对抗条件下仍能保持一致的高性能。 

---
# SkyTrust: Blockchain-Enhanced UAV Security for NTNs with Dynamic Trust and Energy-Aware Consensus 

**Title (ZH)**: SkyTrust：动态信任与能量感知共识增强的NTNs无人机安全区块链技术 

**Authors**: Afan Ali, Irfanullah Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18735)  

**Abstract**: Non-Terrestrial Networks (NTNs) based on Unmanned Aerial Vehicles (UAVs) as base stations are extremely susceptible to security attacks due to their distributed and dynamic nature, which makes them vulnerable to rogue nodes. In this paper, a new Dynamic Trust Score Adjustment Mechanism with Energy-Aware Consensus (DTSAM-EAC) is proposed to enhance security in UAV-based NTNs. The proposed framework integrates a permissioned Hyperledger Fabric blockchain with Federated Learning (FL) to support privacy-preserving trust evaluation. Trust ratings are updated continuously through weighted aggregation of past trust, present behavior, and energy contribution, thus making the system adaptive to changing network conditions. An energy-aware consensus mechanism prioritizes UAVs with greater available energy for block validation, ensuring efficient use of resources under resource-constrained environments. FL aggregation with trust-weighting further increases the resilience of the global trust model. Simulation results verify the designed framework achieves 94\% trust score prediction accuracy and 96\% rogue UAV detection rate while outperforming centralized and static baselines of trust-based solutions on privacy, energy efficiency, and reliability. It complies with 6G requirements in terms of distributed intelligence and sustainability and is an energy-efficient and scalable solution to secure NTNs. 

**Abstract (ZH)**: 基于无人机（UAV）作为基站的非地面网络（NTNs）的动态信任评分调整机制与能量感知共识（DTSAM-EAC）研究 

---
# Cross-Learning Fine-Tuning Strategy for Dysarthric Speech Recognition Via CDSD database 

**Title (ZH)**: 基于CDSD数据库的交叉学习微调策略在构音障碍语音识别中的应用 

**Authors**: Qing Xiao, Yingshan Peng, PeiPei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18732)  

**Abstract**: Dysarthric speech recognition faces challenges from severity variations and disparities relative to normal speech. Conventional approaches individually fine-tune ASR models pre-trained on normal speech per patient to prevent feature conflicts. Counter-intuitively, experiments reveal that multi-speaker fine-tuning (simultaneously on multiple dysarthric speakers) improves recognition of individual speech patterns. This strategy enhances generalization via broader pathological feature learning, mitigates speaker-specific overfitting, reduces per-patient data dependence, and improves target-speaker accuracy - achieving up to 13.15% lower WER versus single-speaker fine-tuning. 

**Abstract (ZH)**: 构音障碍语音识别面临正常语音与严重程度差异带来的挑战。传统方法分别在每个患者正常语音上微调ASR模型以防止特征冲突。令人意想不到的是，实验表明，多说话人微调（同时对多个构音障碍说话人进行微调）可以提高个体语音模式的识别效果。该策略通过广泛的病理特征学习增强泛化能力，减少说话人特定过拟合，降低每例患者的-data依赖性，并提高目标说话人的准确率——相对于单说话人微调可实现最多13.15%的WER降低。 

---
# Skill-Aligned Fairness in Multi-Agent Learning for Collaboration in Healthcare 

**Title (ZH)**: 医疗保健领域协作中技能对齐的公平性 

**Authors**: Promise Osaine Ekpo, Brian La, Thomas Wiener, Saesha Agarwal, Arshia Agrawal, Gonzalo Gonzalez-Pumariega, Lekan P. Molu, Angelique Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2508.18708)  

**Abstract**: Fairness in multi-agent reinforcement learning (MARL) is often framed as a workload balance problem, overlooking agent expertise and the structured coordination required in real-world domains. In healthcare, equitable task allocation requires workload balance or expertise alignment to prevent burnout and overuse of highly skilled agents. Workload balance refers to distributing an approximately equal number of subtasks or equalised effort across healthcare workers, regardless of their expertise. We make two contributions to address this problem. First, we propose FairSkillMARL, a framework that defines fairness as the dual objective of workload balance and skill-task alignment. Second, we introduce MARLHospital, a customizable healthcare-inspired environment for modeling team compositions and energy-constrained scheduling impacts on fairness, as no existing simulators are well-suited for this problem. We conducted experiments to compare FairSkillMARL in conjunction with four standard MARL methods, and against two state-of-the-art fairness metrics. Our results suggest that fairness based solely on equal workload might lead to task-skill mismatches and highlight the need for more robust metrics that capture skill-task misalignment. Our work provides tools and a foundation for studying fairness in heterogeneous multi-agent systems where aligning effort with expertise is critical. 

**Abstract (ZH)**: 多智能体强化学习中的公平性往往被框架为负载均衡问题，忽视了智能体专长及其在现实世界领域中所需结构化协调。在医疗保健领域，公平的任务分配需要负载均衡或专长对齐，以防止专业技能过高的人感到Burnout并避免其过度使用。负载均衡指的是根据医护人员的专长，分配大致相等数量的子任务或相等的努力。我们对这一问题做出了两个贡献。首先，我们提出了FairSkillMARL框架，将公平定义为负载均衡和技能-任务对齐的双重目标。其次，我们引入了MARLHospital，这是一种可定制的、以医疗保健为灵感的环境，用于建模团队组成和能量约束调度对公平性的影响，因为现有的模拟器都不适合解决此问题。我们进行了实验，将FairSkillMARL与四种标准的MARL方法以及两个最先进的公平性度量进行了比较。结果表明，仅基于同等负载的公平性可能导致任务技能不匹配，并强调了需要更加稳健的度量标准来捕捉技能-任务失配的需求。我们的工作提供了研究异构多智能体系统中公平性的工具和基础，其中努力与专长的对齐至关重要。 

---
# Auditing Approximate Machine Unlearning for Differentially Private Models 

**Title (ZH)**: 审计近似机器卸载的差分隐私模型 

**Authors**: Yuechun Gu, Jiajie He, Keke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18671)  

**Abstract**: Approximate machine unlearning aims to remove the effect of specific data from trained models to ensure individuals' privacy. Existing methods focus on the removed records and assume the retained ones are unaffected. However, recent studies on the \emph{privacy onion effect} indicate this assumption might be incorrect. Especially when the model is differentially private, no study has explored whether the retained ones still meet the differential privacy (DP) criterion under existing machine unlearning methods. This paper takes a holistic approach to auditing both unlearned and retained samples' privacy risks after applying approximate unlearning algorithms. We propose the privacy criteria for unlearned and retained samples, respectively, based on the perspectives of DP and membership inference attacks (MIAs). To make the auditing process more practical, we also develop an efficient MIA, A-LiRA, utilizing data augmentation to reduce the cost of shadow model training. Our experimental findings indicate that existing approximate machine unlearning algorithms may inadvertently compromise the privacy of retained samples for differentially private models, and we need differentially private unlearning algorithms. For reproducibility, we have pubished our code: this https URL 

**Abstract (ZH)**: 近似机器遗忘旨在从训练模型中移除特定数据的影响以确保个体隐私。现有方法关注移除的记录，假设保留的记录未受影响。然而，最近关于“隐私洋葱效应”的研究表明，这一假设可能不正确。特别是当模型为差分隐私时，尚未有研究探讨现有机器遗忘方法下保留的记录是否仍满足差分隐私（DP）标准。本文从差分隐私和成员推理攻击的角度分别提出遗忘样本和保留样本的隐私标准，并采用数据增强开发了高效的成员推理攻击A-LiRA，以降低影子模型训练成本，使审计过程更加实用。我们的实验结果表明，现有近似机器遗忘算法可能无意中损害差分隐私模型中保留样本的隐私，需要差分隐私的遗忘算法。为了可重复性，我们已发布代码：this https URL。 

---
# The Quasi-Creature and the Uncanny Valley of Agency: A Synthesis of Theory and Evidence on User Interaction with Inconsistent Generative AI 

**Title (ZH)**: 拟生物体与代理的恐怖谷：不一致生成AI用户交互的理论与证据综述 

**Authors**: Mauricio Manhaes, Christine Miller, Nicholas Schroeder  

**Link**: [PDF](https://arxiv.org/pdf/2508.18563)  

**Abstract**: The user experience with large-scale generative AI is paradoxical: superhuman fluency meets absurd failures in common sense and consistency. This paper argues that the resulting potent frustration is an ontological problem, stemming from the "Quasi-Creature"-an entity simulating intelligence without embodiment or genuine understanding. Interaction with this entity precipitates the "Uncanny Valley of Agency," a framework where user comfort drops when highly agentic AI proves erratically unreliable. Its failures are perceived as cognitive breaches, causing profound cognitive dissonance. Synthesizing HCI, cognitive science, and philosophy of technology, this paper defines the Quasi-Creature and details the Uncanny Valley of Agency. An illustrative mixed-methods study ("Move 78," N=37) of a collaborative creative task reveals a powerful negative correlation between perceived AI efficiency and user frustration, central to the negative experience. This framework robustly explains user frustration with generative AI and has significant implications for the design, ethics, and societal integration of these powerful, alien technologies. 

**Abstract (ZH)**: 大规模生成AI的用户体验是矛盾的：超凡的语言流畅性遭遇常识和一致性上的荒谬失败。本文认为，由此产生的强烈挫败感是一个本体论问题，源自于“准生物体”——一种没有实体和真实理解的智能模拟体。与这种实体的交互导致了“代理的谷仓效应”，即当高度自主的AI表现出随机不可靠性时，用户舒适度下降。这些失败被视为认知上的违规行为，导致深刻的认知失调。综合人机交互、认知科学和科技哲学，本文定义了“准生物体”并详细阐述了“代理的谷仓效应”。一项以协作创造性任务为对象的混合方法研究（“Move 78”，N=37）揭示了感知AI效率与用户挫败感之间强大的负相关关系，这是负面体验的核心。该框架能够 robust 地解释生成AI的用户挫败感，并对这些强大且陌生技术的设计、伦理和社会整合具有重要意义。 

---
# Beyond prior knowledge: The predictive role of knowledge-building in Tutor Learning 

**Title (ZH)**: 超越先验知识：知识建构在辅导学习中的预测作用 

**Authors**: Tasmia Shahriar, Mia Ameen, Aditi Mallavarapu, Shiyan Jiang, Noboru Matsuda  

**Link**: [PDF](https://arxiv.org/pdf/2508.18545)  

**Abstract**: When adopting the role of a teacher in learning-by-teaching environments, students often struggle to engage in knowledge-building activities, such as providing explanations and addressing misconceptions. Instead, they frequently default to knowledge-telling behaviors, where they simply dictate what they already know or what to do without deeper reflection, thereby limiting learning. Teachable agents, particularly those capable of posing persistent follow-up questions, have been shown to encourage students (tutors) to shift from knowledge-telling to knowledge-building and enhance tutor learning. Tutor learning encompasses two interrelated types of knowledge: conceptual and procedural knowledge. Research has established a bidirectional relationship between these knowledge types, where improvements in one reinforce the other. This study investigates the role of knowledge-building in mediating the bidirectional relationship between procedural and conceptual learning. Our findings revealed a stable bidirectional relationship between procedural and conceptual knowledge, with higher post-test scores observed among students who engaged in knowledge-building, regardless of their procedural and conceptual pre-test performance. This suggests that knowledge-building serves as a crucial mechanism bridging the gap between students with low prior knowledge and higher conceptual and procedural learning gain. 

**Abstract (ZH)**: 在学习-教学环境中的教学角色下，学生常常难以参与知识建构活动，如提供解释和纠正误解，而是倾向于知识传授行为，仅仅告知他们已知的内容或操作步骤，而不进行深入反思，从而限制了学习。能够提出持续跟进问题的可教代理已经被证明能够鼓励学生（导师）从知识传授转向知识建构，促进导师学习。导师学习涵盖两种相关类型的知识：概念性和程序性知识。研究表明，这两种知识类型之间存在双向关系，其中一方的改进会强化另一方。本研究探讨了知识建构在程序性和概念性学习双向关系中的中介作用。研究发现，参与知识建构的学生在后测中表现出了稳定的双向关系，无论他们在程序性和概念性测验中的初始表现如何。这表明知识建构是连接知识基础较弱的学生与更高概念性和程序性学习收益的关键机制。 

---
# Analise de Desaprendizado de Maquina em Modelos de Classificacao de Imagens Medicas 

**Title (ZH)**: 医疗图像分类模型的机器卸学分析 

**Authors**: Andreza M. C. Falcao, Filipe R. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2508.18509)  

**Abstract**: Machine unlearning aims to remove private or sensitive data from a pre-trained model while preserving the model's robustness. Despite recent advances, this technique has not been explored in medical image classification. This work evaluates the SalUn unlearning model by conducting experiments on the PathMNIST, OrganAMNIST, and BloodMNIST datasets. We also analyse the impact of data augmentation on the quality of unlearning. Results show that SalUn achieves performance close to full retraining, indicating an efficient solution for use in medical applications. 

**Abstract (ZH)**: 机器去学习旨在从预训练模型中移除私人或敏感数据，同时保持模型的鲁棒性。尽管近期取得了一些进展，但该技术在医学图像分类中的应用尚未被探索。本工作通过在PathMNIST、OrganAMNIST和BloodMNIST数据集上进行实验，评估了SalUn去学习模型，并分析了数据增强对去学习质量的影响。结果显示，SalUn的性能接近完全重训练，表明其在医疗应用中是一种高效解决方案。 

---
# Data Augmentation Improves Machine Unlearning 

**Title (ZH)**: 数据增强提高机器遗忘效果 

**Authors**: Andreza M. C. Falcao, Filipe R. Cordeiro  

**Link**: [PDF](https://arxiv.org/pdf/2508.18502)  

**Abstract**: Machine Unlearning (MU) aims to remove the influence of specific data from a trained model while preserving its performance on the remaining data. Although a few works suggest connections between memorisation and augmentation, the role of systematic augmentation design in MU remains under-investigated. In this work, we investigate the impact of different data augmentation strategies on the performance of unlearning methods, including SalUn, Random Label, and Fine-Tuning. Experiments conducted on CIFAR-10 and CIFAR-100, under varying forget rates, show that proper augmentation design can significantly improve unlearning effectiveness, reducing the performance gap to retrained models. Results showed a reduction of up to 40.12% of the Average Gap unlearning Metric, when using TrivialAug augmentation. Our results suggest that augmentation not only helps reduce memorization but also plays a crucial role in achieving privacy-preserving and efficient unlearning. 

**Abstract (ZH)**: 机器去学习（MU）旨在从训练模型中移除特定数据的影响，同时保持其对剩余数据性能的稳定性。尽管有少数工作表明记忆和增强之间的关联，但系统增强设计在MU中的作用仍需进一步探究。在本工作中，我们研究了不同的数据增强策略对去学习方法性能的影响，包括SalUn、随机标签和微调。在CIFAR-10和CIFAR-100上的实验，在不同的遗忘率下表明，适当的设计增强策略可以显著提高去学习效果，减少与重新训练模型之间的性能差距。使用TrivialAug增强时，平均差距去学习指标的降低幅度最高可达40.12%。我们的结果表明，增强不仅有助于减少记忆，还对实现隐私保护和高效的去学习起着关键作用。 

---
# Vectorized Attention with Learnable Encoding for Quantum Transformer 

**Title (ZH)**: 可学习编码的向量化注意力量子变换器 

**Authors**: Ziqing Guo, Ziwen Pan, Alex Khan, Jan Balewski  

**Link**: [PDF](https://arxiv.org/pdf/2508.18464)  

**Abstract**: Vectorized quantum block encoding provides a way to embed classical data into Hilbert space, offering a pathway for quantum models, such as Quantum Transformers (QT), that replace classical self-attention with quantum circuit simulations to operate more efficiently. Current QTs rely on deep parameterized quantum circuits (PQCs), rendering them vulnerable to QPU noise, and thus hindering their practical performance. In this paper, we propose the Vectorized Quantum Transformer (VQT), a model that supports ideal masked attention matrix computation through quantum approximation simulation and efficient training via vectorized nonlinear quantum encoder, yielding shot-efficient and gradient-free quantum circuit simulation (QCS) and reduced classical sampling overhead. In addition, we demonstrate an accuracy comparison for IBM and IonQ in quantum circuit simulation and competitive results in benchmarking natural language processing tasks on IBM state-of-the-art and high-fidelity Kingston QPU. Our noise intermediate-scale quantum friendly VQT approach unlocks a novel architecture for end-to-end machine learning in quantum computing. 

**Abstract (ZH)**: 向量化的量子块编码提供了一种将经典数据嵌入希尔伯特空间的方法，为量子变换器（QT）等量子模型提供了途径，这些模型通过使用量子电路模拟来替换经典自注意力，从而更高效地运行。当前的QT依赖于深度参数化量子电路（PQCs），使其容易受到量子处理器噪声的影响，从而阻碍了其实用性能。在本文中，我们提出了向量化的量子变换器（VQT）模型，该模型利用量子近似模拟和支持理想掩码注意力矩阵计算，并通过向量化非线性量子编码器实现高效的训练，从而提供节能且无需梯度的量子电路模拟（QCS），并减少经典采样的开销。此外，我们在量子电路模拟中展示了IBM和IonQ的准确性比较，并在IBM最先进的高保真Kingston QPU上进行自然语言处理任务基准测试，取得了竞争性结果。我们的噪声中间规模量子友好型VQT方法解锁了量子计算中端到端机器学习的新架构。 

---
# SwiftF0: Fast and Accurate Monophonic Pitch Detection 

**Title (ZH)**: SwiftF0：快速准确的单音符音高检测 

**Authors**: Lars Nieradzik  

**Link**: [PDF](https://arxiv.org/pdf/2508.18440)  

**Abstract**: Accurate and real-time monophonic pitch estimation in noisy conditions, particularly on resource-constrained devices, remains an open challenge in audio processing. We present \emph{SwiftF0}, a novel, lightweight neural model that sets a new state-of-the-art for monophonic pitch estimation. Through training on diverse speech, music, and synthetic datasets with extensive data augmentation, SwiftF0 achieves robust generalization across acoustic domains while maintaining computational efficiency. SwiftF0 achieves a 91.80\% harmonic mean (HM) at 10 dB SNR, outperforming baselines like CREPE by over 12 percentage points and degrading by only 2.3 points from clean audio. SwiftF0 requires only 95,842 parameters and runs approximately 42x faster than CREPE on CPU, making it ideal for efficient, real-time deployment. To address the critical lack of perfectly accurate ground truth pitch in speech corpora (which typically rely on algorithmic estimators or laryngograph signals), we introduce \emph{SpeechSynth}. This synthetic speech dataset, generated by a phoneme-level TTS model, provides exact, on-demand ground-truth pitch curves, enabling more robust model training and evaluation. Furthermore, we propose a unified metric, combining six complementary performance measures for comprehensive and reliable pitch evaluation, and release an open-source pitch benchmark suite. A live demo of SwiftF0 is available at this https URL, the source code at this https URL, and the benchmark framework at this https URL. 

**Abstract (ZH)**: 在资源受限设备上进行噪声条件下单音符音高准确且实时的估计仍然是音频处理领域的开放挑战。我们提出了SwiftF0，这是一种新颖的轻量级神经模型，为单音符音高估计设定了新的state-of-the-art。通过在多种语音、音乐和合成数据集上进行训练，并采用广泛的数据增强，SwiftF0在保持计算效率的同时实现了跨声学领域的稳健泛化。在10 dB信噪比下，SwiftF0的调和平均值达到91.80%，优于CREPE等基线方法12个百分点以上，仅比干净音频下降2.3个百分点。SwiftF0仅需要95,842个参数，在CPU上运行速度约为CREPE的42倍，使其适用于高效的实时部署。为了解决语音语料库中perfect准确的地面真实音高缺乏的关键问题（通常依赖于算法估计或喉图信号），我们引入了SpeechSynth。这一通过音素级TTS模型生成的合成语音数据集提供了精确且按需的地面真实音高曲线，从而增强了模型训练和评估的稳健性。此外，我们提出了一个统一的评价指标，结合了六种互补的性能指标，用于全面可靠地评估音高。我们还发布了开源音高基准套件。SwiftF0的实时演示可在以下链接访问，源代码可在以下链接访问，基准框架可在以下链接访问。 

---
# CLARIFY: A Specialist-Generalist Framework for Accurate and Lightweight Dermatological Visual Question Answering 

**Title (ZH)**: CLARIFY：一种准确轻量级的专科-通才框架用于皮肤病视觉问答 

**Authors**: Aranya Saha, Tanvir Ahmed Khan, Ismam Nur Swapnil, Mohammad Ariful Haque  

**Link**: [PDF](https://arxiv.org/pdf/2508.18430)  

**Abstract**: Vision-language models (VLMs) have shown significant potential for medical tasks; however, their general-purpose nature can limit specialized diagnostic accuracy, and their large size poses substantial inference costs for real-world clinical deployment. To address these challenges, we introduce CLARIFY, a Specialist-Generalist framework for dermatological visual question answering (VQA). CLARIFY combines two components: (i) a lightweight, domain-trained image classifier (the Specialist) that provides fast and highly accurate diagnostic predictions, and (ii) a powerful yet compressed conversational VLM (the Generalist) that generates natural language explanations to user queries. In our framework, the Specialist's predictions directly guide the Generalist's reasoning, focusing it on the correct diagnostic path. This synergy is further enhanced by a knowledge graph-based retrieval module, which grounds the Generalist's responses in factual dermatological knowledge, ensuring both accuracy and reliability. This hierarchical design not only reduces diagnostic errors but also significantly improves computational efficiency. Experiments on our curated multimodal dermatology dataset demonstrate that CLARIFY achieves an 18\% improvement in diagnostic accuracy over the strongest baseline, a fine-tuned, uncompressed single-line VLM, while reducing the average VRAM requirement and latency by at least 20\% and 5\%, respectively. These results indicate that a Specialist-Generalist system provides a practical and powerful paradigm for building lightweight, trustworthy, and clinically viable AI systems. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在医疗任务中展现了显著的潜力；然而，它们的通用性质可能会限制专门诊断的准确性，而且它们的大型化结构会带来实际临床部署中的大量推理成本。为解决这些挑战，我们提出了CLARIFY，一种皮肤病视觉问答（VQA）的专家-泛化框架。CLARIFY结合了两个组件：一是轻量级的领域训练图像分类器（专家），提供快速且高度准确的诊断预测；二是强大但压缩的对话性VLM（泛化模型），生成自然语言解释以回应用户查询。在我们的框架中，专家的预测直接引导泛化模型的推理，使其专注于正确的诊断路径。进一步通过基于知识图谱的检索模块增强了这种协同作用，确保泛化模型的回答基于事实性的皮肤病知识，从而确保准确性和可靠性。这种分层设计不仅减少了诊断错误，还显著提高了计算效率。在我们精心构建的多模态皮肤病数据集上的实验表明，CLARIFY在诊断准确性上比最强基线（微调后的未压缩单行VLM）提高了18%，同时分别降低了至少20%和5%的平均VRAM需求和延迟。这些结果表明，专家-泛化系统为构建轻量级、可靠且临床可行的AI系统提供了实用而强大的范式。 

---
# Low-Rank Tensor Decompositions for the Theory of Neural Networks 

**Title (ZH)**: 低秩张量分解在神经网络理论中的应用 

**Authors**: Ricardo Borsoi, Konstantin Usevich, Marianne Clausel  

**Link**: [PDF](https://arxiv.org/pdf/2508.18408)  

**Abstract**: The groundbreaking performance of deep neural networks (NNs) promoted a surge of interest in providing a mathematical basis to deep learning theory. Low-rank tensor decompositions are specially befitting for this task due to their close connection to NNs and their rich theoretical results. Different tensor decompositions have strong uniqueness guarantees, which allow for a direct interpretation of their factors, and polynomial time algorithms have been proposed to compute them. Through the connections between tensors and NNs, such results supported many important advances in the theory of NNs. In this review, we show how low-rank tensor methods--which have been a core tool in the signal processing and machine learning communities--play a fundamental role in theoretically explaining different aspects of the performance of deep NNs, including their expressivity, algorithmic learnability and computational hardness, generalization, and identifiability. Our goal is to give an accessible overview of existing approaches (developed by different communities, ranging from computer science to mathematics) in a coherent and unified way, and to open a broader perspective on the use of low-rank tensor decompositions for the theory of deep NNs. 

**Abstract (ZH)**: 低秩张量分解在深度神经网络理论中的重要作用：从表达能力、算法学习性与计算复杂性、泛化能力和可识别性角度解释深度神经网络的性能 

---
# Can Out-of-Distribution Evaluations Uncover Reliance on Shortcuts? A Case Study in Question Answering 

**Title (ZH)**: 偏离分布评估能否揭示模型对捷径的依赖？一个来自问答系统的案例研究 

**Authors**: Michal Štefánik, Timothee Mickus, Marek Kadlčík, Michal Spiegel, Josef Kuchař  

**Link**: [PDF](https://arxiv.org/pdf/2508.18407)  

**Abstract**: A majority of recent work in AI assesses models' generalization capabilities through the lens of performance on out-of-distribution (OOD) datasets. Despite their practicality, such evaluations build upon a strong assumption: that OOD evaluations can capture and reflect upon possible failures in a real-world deployment.
In this work, we challenge this assumption and confront the results obtained from OOD evaluations with a set of specific failure modes documented in existing question-answering (QA) models, referred to as a reliance on spurious features or prediction shortcuts.
We find that different datasets used for OOD evaluations in QA provide an estimate of models' robustness to shortcuts that have a vastly different quality, some largely under-performing even a simple, in-distribution evaluation. We partially attribute this to the observation that spurious shortcuts are shared across ID+OOD datasets, but also find cases where a dataset's quality for training and evaluation is largely disconnected. Our work underlines limitations of commonly-used OOD-based evaluations of generalization, and provides methodology and recommendations for evaluating generalization within and beyond QA more robustly. 

**Abstract (ZH)**: 大部分 recent work in AI 通过在离群分布（OOD）数据集上的性能来评估模型的泛化能力。尽管这类评估实用，但它们建立在一项强有力的前提假设之上：即 OOD 评估能够捕捉并反映实际部署中可能发生的失败情况。
在本工作中，我们挑战这一假设，并将 OOD 评估所得结果与现有问答（QA）模型中记录的具体失败模式进行对比，具体来说是依赖于虚假特征或预测捷径。
我们发现，用于 QA 中 OOD 评估的不同数据集提供了对模型对捷径的鲁棒性估计，而这些捷径的质量差异很大，有些甚至远逊于内部分布评估的简单水平。我们部分将这一现象归因于观察到虚假捷径在内部+离群数据集中共享，但也发现了训练和评估质量与其所对应的数据集质量明显脱钩的情况。我们的工作指出了常用 OOD 基础评估方法的局限性，并提供了在和超越问答领域中更鲁棒地评估泛化的方法和建议。 

---
# Facilitating Matches on Allocation Platforms 

**Title (ZH)**: 促进匹配平台上的匹配过程 

**Authors**: Yohai Trabelsi, Abhijin Adiga, Yonatan Aumann, Sarit Kraus, S. S. Ravi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18325)  

**Abstract**: We consider a setting where goods are allocated to agents by way of an allocation platform (e.g., a matching platform). An ``allocation facilitator'' aims to increase the overall utility/social-good of the allocation by encouraging (some of the) agents to relax (some of) their restrictions. At the same time, the advice must not hurt agents who would otherwise be better off. Additionally, the facilitator may be constrained by a ``bound'' (a.k.a. `budget'), limiting the number and/or type of restrictions it may seek to relax. We consider the facilitator's optimization problem of choosing an optimal set of restrictions to request to relax under the aforementioned constraints. Our contributions are three-fold: (i) We provide a formal definition of the problem, including the participation guarantees to which the facilitator should adhere. We define a hierarchy of participation guarantees and also consider several social-good functions. (ii) We provide polynomial algorithms for solving various versions of the associated optimization problems, including one-to-one and many-to-one allocation settings. (iii) We demonstrate the benefits of such facilitation and relaxation, and the implications of the different participation guarantees, using extensive experimentation on three real-world datasets. 

**Abstract (ZH)**: 我们考虑一种情形，在这种情形中，物品通过分配平台（例如，匹配平台）分配给代理人。一个“分配促进者”旨在通过鼓励（某些）代理人放宽（某些）限制来增加分配的整体效用/社会福利，同时确保不会损害原本会更好受益的代理人。此外，促进者可能会受到一个“限制”（又称“预算”）的约束，这限制了其可以请求放宽的限制的数量和/或类型。我们考虑了在前述约束条件下，促进者选择一个最优的限制集来请求放宽的问题。我们的贡献包括三个方面：（i）我们提供了该问题的正式定义，并包括促进者应遵守的参与保证。我们定义了参与保证的层次结构，并考虑了几种社会福利函数。（ii）我们提供了多项式算法来解决与之相关的各种优化问题，包括一对一和多对一的分配设置。（iii）我们通过在三个实际数据集上进行广泛的实验，展示了这种促进和放宽带来的好处，以及不同参与保证的含义。 

---
# Does Calibration Affect Human Actions? 

**Title (ZH)**: 校准会影响人类行为吗？ 

**Authors**: Meir Nizri, Amos Azaria, Chirag Gupta, Noam Hazon  

**Link**: [PDF](https://arxiv.org/pdf/2508.18317)  

**Abstract**: Calibration has been proposed as a way to enhance the reliability and adoption of machine learning classifiers. We study a particular aspect of this proposal: how does calibrating a classification model affect the decisions made by non-expert humans consuming the model's predictions? We perform a Human-Computer-Interaction (HCI) experiment to ascertain the effect of calibration on (i) trust in the model, and (ii) the correlation between decisions and predictions. We also propose further corrections to the reported calibrated scores based on Kahneman and Tversky's prospect theory from behavioral economics, and study the effect of these corrections on trust and decision-making. We find that calibration is not sufficient on its own; the prospect theory correction is crucial for increasing the correlation between human decisions and the model's predictions. While this increased correlation suggests higher trust in the model, responses to ``Do you trust the model more?" are unaffected by the method used. 

**Abstract (ZH)**: 校准被提出作为一种提高机器学习分类器可靠性和采用度的方法。我们研究了这一提议的一个特定方面：校准分类模型如何影响非专家人类消费者基于模型预测所做的决策？我们进行了一项人机交互（HCI）实验，以确定校准对（i）模型信任度以及（ii）决策与预测的相关性的影响。我们还根据行为经济学中的凯恩eman和Tversky的前景理论提出了进一步修正校准得分的方法，并研究了这些修正对信任度和决策的影响。我们发现，仅靠校准本身是不够的；前景理论的修正对于提高人类决策与模型预测之间的相关性至关重要。尽管这种增强的相关性表明了对模型更高的信任度，但“你是否更信任该模型？”的问题的回答并未受到所使用方法的影响。 

---
# Evaluating Federated Learning for At-Risk Student Prediction: A Comparative Analysis of Model Complexity and Data Balancing 

**Title (ZH)**: 评估联邦学习在预测高风险学生中的应用：模型复杂度与数据平衡的比较分析 

**Authors**: Rodrigo Tertulino  

**Link**: [PDF](https://arxiv.org/pdf/2508.18316)  

**Abstract**: High dropout and failure rates in distance education pose a significant challenge for academic institutions, making the proactive identification of at-risk students crucial for providing timely support. This study develops and evaluates a machine learning model based on early academic performance and digital engagement patterns from the large-scale OULAD dataset to predict student risk at a UK university. To address the practical challenges of data privacy and institutional silos that often hinder such initiatives, we implement the model using a Federated Learning (FL) framework. We compare model complexity (Logistic Regression vs. a Deep Neural Network) and data balancing. The final federated model demonstrates strong predictive capability, achieving an ROC AUC score of approximately 85% in identifying at-risk students. Our findings show that this federated approach provides a practical and scalable solution for institutions to build effective early-warning systems, enabling proactive student support while inherently respecting data privacy. 

**Abstract (ZH)**: 高辍学率和失败率在远程教育中构成重大挑战，需要学术机构及时识别风险学生以提供支持。本研究基于大规模OULAD数据集中的早期学术表现和数字参与模式开发并评估了一种机器学习模型，以预测英国大学学生风险。为应对数据隐私和机构孤岛等实际挑战，我们采用联邦学习（FL）框架实施该模型。我们比较了模型复杂性（逻辑回归与深度神经网络）和数据平衡。最终的联邦模型显示了强大的预测能力，识别风险学生时ROC AUC得分为约85%。我们的研究结果表明，联邦学习方法为机构提供了实用且可扩展的解决方案，以构建有效的预警系统，同时内在地尊重数据隐私，从而实现主动的学生支持。 

---
# ProtoEHR: Hierarchical Prototype Learning for EHR-based Healthcare Predictions 

**Title (ZH)**: 基于医疗记录的预测：层次原型学习方法ProtoEHR 

**Authors**: Zi Cai, Yu Liu, Zhiyao Luo, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18313)  

**Abstract**: Digital healthcare systems have enabled the collection of mass healthcare data in electronic healthcare records (EHRs), allowing artificial intelligence solutions for various healthcare prediction tasks. However, existing studies often focus on isolated components of EHR data, limiting their predictive performance and interpretability. To address this gap, we propose ProtoEHR, an interpretable hierarchical prototype learning framework that fully exploits the rich, multi-level structure of EHR data to enhance healthcare predictions. More specifically, ProtoEHR models relationships within and across three hierarchical levels of EHRs: medical codes, hospital visits, and patients. We first leverage large language models to extract semantic relationships among medical codes and construct a medical knowledge graph as the knowledge source. Building on this, we design a hierarchical representation learning framework that captures contextualized representations across three levels, while incorporating prototype information within each level to capture intrinsic similarities and improve generalization. To perform a comprehensive assessment, we evaluate ProtoEHR in two public datasets on five clinically significant tasks, including prediction of mortality, prediction of readmission, prediction of length of stay, drug recommendation, and prediction of phenotype. The results demonstrate the ability of ProtoEHR to make accurate, robust, and interpretable predictions compared to baselines in the literature. Furthermore, ProtoEHR offers interpretable insights on code, visit, and patient levels to aid in healthcare prediction. 

**Abstract (ZH)**: 基于原型学习的可解释层次医疗记录框架ProtoEHR：增强医疗预测 

---
# CoPE: A Lightweight Complex Positional Encoding 

**Title (ZH)**: CoPE: 一种轻量级复杂位置编码 

**Authors**: Avinash Amballa  

**Link**: [PDF](https://arxiv.org/pdf/2508.18308)  

**Abstract**: Recent studies have demonstrated the effectiveness of position encoding in transformer architectures. By incorporating positional information, this approach provides essential guidance for modeling dependencies between elements across different sequence positions. We introduce CoPE (a lightweight Complex Positional Encoding), a novel architecture that leverages complex-valued encoding to encode both content and positional information. Our approach replaces traditional positional encodings with complex embeddings where the real part captures semantic content and the imaginary part encodes positional information. We introduce phase-aware attention in the first layer of the transformer model to capture position-dependent patterns, followed by standard attention layers for higher-levels. We show that CoPE doesn't exhibit long term decay and is compatible with linear attention. Experimental evaluation on the GLUE benchmark suggest that our approach achieves superior performance with less computational complexity, compared to RoPE, Sinusoidal and Learned positional encodings. 

**Abstract (ZH)**: 最近的研究表明，在变压器架构中使用位置编码的有效性。通过整合位置信息，该方法为建模不同序列位置之间元素的依赖关系提供了必要的指导。我们提出了一种新的架构CoPE（轻量级复数位置编码），该架构利用复数编码来同时编码内容和位置信息。我们的方法用复数嵌入替换传统的位置编码，其中实部捕获语义内容，虚部编码位置信息。我们在变压器模型的第一层引入了相位感知注意力以捕捉位置相关的模式，随后是标准的注意力层用于高层次的处理。实验评价结果显示，CoPE 不表现出长期衰减，并且兼容线性注意力。在GLUE基准上的实验表明，与RoPE、正弦和学习的位置编码相比，我们的方法在计算复杂性较低的情况下实现了更好的性能。 

---
# scI2CL: Effectively Integrating Single-cell Multi-omics by Intra- and Inter-omics Contrastive Learning 

**Title (ZH)**: scI2CL: 通过跨组学和组内组学对比学习有效地整合单细胞多组学数据 

**Authors**: Wuchao Liu, Han Peng, Wengen Li, Yichao Zhang, Jihong Guan, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.18304)  

**Abstract**: Single-cell multi-omics data contain huge information of cellular states, and analyzing these data can reveal valuable insights into cellular heterogeneity, diseases, and biological processes. However, as cell differentiation \& development is a continuous and dynamic process, it remains challenging to computationally model and infer cell interaction patterns based on single-cell multi-omics data. This paper presents scI2CL, a new single-cell multi-omics fusion framework based on intra- and inter-omics contrastive learning, to learn comprehensive and discriminative cellular representations from complementary multi-omics data for various downstream tasks. Extensive experiments of four downstream tasks validate the effectiveness of scI2CL and its superiority over existing peers. Concretely, in cell clustering, scI2CL surpasses eight state-of-the-art methods on four widely-used real-world datasets. In cell subtyping, scI2CL effectively distinguishes three latent monocyte cell subpopulations, which are not discovered by existing methods. Simultaneously, scI2CL is the only method that correctly constructs the cell developmental trajectory from hematopoietic stem and progenitor cells to Memory B cells. In addition, scI2CL resolves the misclassification of cell types between two subpopulations of CD4+ T cells, while existing methods fail to precisely distinguish the mixed cells. In summary, scI2CL can accurately characterize cross-omics relationships among cells, thus effectively fuses multi-omics data and learns discriminative cellular representations to support various downstream analysis tasks. 

**Abstract (ZH)**: 单细胞多组学数据含有丰富的细胞状态信息，分析这些数据可以揭示细胞异质性、疾病和生物过程的重要见解。然而，由于细胞分化与发展是一个连续和动态的过程，基于单细胞多组学数据计算建模和推断细胞相互作用模式仍然具有挑战性。本文提出了一种名为scI2CL的新单细胞多组学融合框架，基于组内和组间对比学习，从互补的多组学数据中学习全面且具有区分性的细胞表示，以支持各种下游任务。广泛开展的四种下游任务的实验验证了scI2CL的有效性和其相对于现有方法的优势。具体来说，在细胞聚类任务中，scI2CL在四个广泛使用的实际数据集中超过八种最先进的方法。在细胞亚型划分任务中，scI2CL有效区分了三个潜在的单核细胞亚群，而现有方法未能发现这些亚群。同时，scI2CL是唯一能够从造血干细胞和祖细胞到记忆B细胞构建细胞发育轨迹的方法。此外，scI2CL解决了CD4+ T细胞两个亚群之间的细胞类型误分类问题，而现有方法无法精确区分混合细胞。总之，scI2CL可以准确刻画跨组学的细胞关系，从而有效地融合多组学数据并学习具有区分性的细胞表示，以支持各种下游分析任务。 

---
# Learning Explainable Imaging-Genetics Associations Related to a Neurological Disorder 

**Title (ZH)**: 学习可解释的影像遗传学关联，与一种神经疾病相关 

**Authors**: Jueqi Wang, Zachary Jacokes, John Darrell Van Horn, Michael C. Schatz, Kevin A. Pelphrey, Archana Venkataraman  

**Link**: [PDF](https://arxiv.org/pdf/2508.18303)  

**Abstract**: While imaging-genetics holds great promise for unraveling the complex interplay between brain structure and genetic variation in neurological disorders, traditional methods are limited to simplistic linear models or to black-box techniques that lack interpretability. In this paper, we present NeuroPathX, an explainable deep learning framework that uses an early fusion strategy powered by cross-attention mechanisms to capture meaningful interactions between structural variations in the brain derived from MRI and established biological pathways derived from genetics data. To enhance interpretability and robustness, we introduce two loss functions over the attention matrix - a sparsity loss that focuses on the most salient interactions and a pathway similarity loss that enforces consistent representations across the cohort. We validate NeuroPathX on both autism spectrum disorder and Alzheimer's disease. Our results demonstrate that NeuroPathX outperforms competing baseline approaches and reveals biologically plausible associations linked to the disorder. These findings underscore the potential of NeuroPathX to advance our understanding of complex brain disorders. Code is available at this https URL . 

**Abstract (ZH)**: 神经影像与遗传学中的一种可解释深度学习框架：NeuroPathX及其在神经精神障碍研究中的应用 

---
# Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms 

**Title (ZH)**: Murakkab：云计算平台中高效资源管理的代理工作流编排 

**Authors**: Gohar Irfan Chaudhry, Esha Choukse, Haoran Qiu, Íñigo Goiri, Rodrigo Fonseca, Adam Belay, Ricardo Bianchini  

**Link**: [PDF](https://arxiv.org/pdf/2508.18298)  

**Abstract**: Agentic workflows commonly coordinate multiple models and tools with complex control logic. They are quickly becoming the dominant paradigm for AI applications. However, serving them remains inefficient with today's frameworks. The key problem is that they expose workflows as opaque sequences of model and tool calls that tightly couple agent logic with model and hardware choices. Often, these workflow components are fragmented across different entities, preventing systems from reasoning about trade-offs across accuracy, latency, energy, and cost. This leads to resource waste and degraded service-level objectives (SLOs).
We present Murakkab, a resource-efficient serving system for agentic workflows. Murakkab introduces a declarative abstraction that decouples workflow specification from execution configuration. A profile-guided optimizer and adaptive runtime jointly manage the full stack: orchestrating workflow components, mapping them to models and hardware, and dynamically reconfiguring execution to satisfy user-defined SLOs. By exposing the internal structure of agentic workflows, Murakkab enables cross-layer optimization that existing frameworks and cloud schedulers cannot achieve.
Our evaluation on diverse workflows shows that \sysname{} reduces GPU usage by up to 2.8$\times$, energy consumption by 3.7$\times$, and cost by 4.3$\times$ while maintaining SLOs. 

**Abstract (ZH)**: 代理工作流优化系统Murakkab 

---
# Federative ischemic stroke segmentation as alternative to overcome domain-shift multi-institution challenges 

**Title (ZH)**: 联邦缺血性卒中分割作为克服多机构领域转移挑战的替代方法 

**Authors**: Edgar Rangel, Fabio Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2508.18296)  

**Abstract**: Stroke is the second leading cause of death and the third leading cause of disability worldwide. Clinical guidelines establish diffusion resonance imaging (DWI, ADC) as the standard for localizing, characterizing, and measuring infarct volume, enabling treatment support and prognosis. Nonetheless, such lesion analysis is highly variable due to different patient demographics, scanner vendors, and expert annotations. Computational support approaches have been key to helping with the localization and segmentation of lesions. However, these strategies are dedicated solutions that learn patterns from only one institution, lacking the variability to generalize geometrical lesions shape models. Even worse, many clinical centers lack sufficient labeled samples to adjust these dedicated solutions. This work developed a collaborative framework for segmenting ischemic stroke lesions in DWI sequences by sharing knowledge from deep center-independent representations. From 14 emulated healthcare centers with 2031 studies, the FedAvg model achieved a general DSC of $0.71 \pm 0.24$, AVD of $5.29 \pm 22.74$, ALD of $2.16 \pm 3.60$ and LF1 of $0.70 \pm 0.26$ over all centers, outperforming both the centralized and other federated rules. Interestingly, the model demonstrated strong generalization properties, showing uniform performance across different lesion categories and reliable performance in out-of-distribution centers (with DSC of $0.64 \pm 0.29$ and AVD of $4.44 \pm 8.74$ without any additional training). 

**Abstract (ZH)**: 缺血性卒中DWI序列病变分割的协作框架及性能分析 

---
# H-PRM: A Pluggable Hotword Pre-Retrieval Module for Various Speech Recognition Systems 

**Title (ZH)**: H-PRM: 一种适用于各种语音识别系统的插件式热词预检索模块 

**Authors**: Huangyu Dai, Lingtao Mao, Ben Chen, Zihan Wang, Zihan Liang, Ying Han, Chenyi Lei, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.18295)  

**Abstract**: Hotword customization is crucial in ASR to enhance the accuracy of domain-specific terms. It has been primarily driven by the advancements in traditional models and Audio large language models (LLMs). However, existing models often struggle with large-scale hotwords, as the recognition rate drops dramatically with the number of hotwords increasing. In this paper, we introduce a novel hotword customization system that utilizes a hotword pre-retrieval module (H-PRM) to identify the most relevant hotword candidate by measuring the acoustic similarity between the hotwords and the speech segment. This plug-and-play solution can be easily integrated into traditional models such as SeACo-Paraformer, significantly enhancing hotwords post-recall rate (PRR). Additionally, we incorporate H-PRM into Audio LLMs through a prompt-based approach, enabling seamless customization of hotwords. Extensive testing validates that H-PRM can outperform existing methods, showing a new direction for hotword customization in ASR. 

**Abstract (ZH)**: 热点词定制对于ASR提高领域特定术语的准确性至关重要，主要受到传统模型和音频大型语言模型（LLMs）进展的驱动。然而，现有模型在处理大规模热点词时常常表现不佳，随着热点词数量增加，识别率急剧下降。本文介绍了一种新颖的热点词定制系统，利用热点词预先检索模块（H-PRM）通过测量热点词与语音片段的声学相似性来识别最相关的热点词候选。该即插即用解决方案可以轻松集成到传统模型如SeACo-Paraformer中，显著提升热点词召回后处理率（PPR）。此外，我们通过提示驱动的方法将H-PRM集成到音频LLMs中，实现热点词的无缝定制。广泛测试表明，H-PRM可以超越现有方法，为ASR中的热点词定制提供新的方向。 

---
# Toward Responsible ASR for African American English Speakers: A Scoping Review of Bias and Equity in Speech Technology 

**Title (ZH)**: 面向非洲裔美国人英语使用者的责任AI语音识别：语音技术中偏见与公平的综述 

**Authors**: Jay L. Cunningham, Adinawa Adjagbodjou, Jeffrey Basoah, Jainaba Jawara, Kowe Kadoma, Aaleyah Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2508.18288)  

**Abstract**: This scoping literature review examines how fairness, bias, and equity are conceptualized and operationalized in Automatic Speech Recognition (ASR) and adjacent speech and language technologies (SLT) for African American English (AAE) speakers and other linguistically diverse communities. Drawing from 44 peer-reviewed publications across Human-Computer Interaction (HCI), Machine Learning/Natural Language Processing (ML/NLP), and Sociolinguistics, we identify four major areas of inquiry: (1) how researchers understand ASR-related harms; (2) inclusive data practices spanning collection, curation, annotation, and model training; (3) methodological and theoretical approaches to linguistic inclusion; and (4) emerging practices and design recommendations for more equitable systems. While technical fairness interventions are growing, our review highlights a critical gap in governance-centered approaches that foreground community agency, linguistic justice, and participatory accountability. We propose a governance-centered ASR lifecycle as an emergent interdisciplinary framework for responsible ASR development and offer implications for researchers, practitioners, and policymakers seeking to address language marginalization in speech AI systems. 

**Abstract (ZH)**: 这项范围性文献综述考察了公平性、偏差和公正性在自动语音识别（ASR）及相关语音和语言技术（SLT）中对使用非洲裔美国英语（AAE）和其他语言多样化社区的理解和操作化方式。本文综述了44篇同行评审的出版物，涵盖了人机交互（HCI）、机器学习/自然语言处理（ML/NLP）和社会语言学领域，指出了四个主要的研究领域：（1）研究人员对ASR相关危害的理解；（2）贯穿数据收集、整理、标注和模型训练的包容性数据实践；（3）语言包容性的方法论和理论方法；以及（4）促进更公正系统的新兴实践和设计建议。尽管技术公平性干预正在增长，但我们的综述强调了治理中心方法中的一个关键缺口，这些方法强调社区自主权、语言正义和参与式问责。本文提出了一个治理中心的ASR生命周期作为负责任ASR开发的新兴跨学科框架，并为旨在解决语音AI系统中语言边缘化的研究者、实践者和政策制定者提供了含义。 

---
# Multi-Modal Drift Forecasting of Leeway Objects via Navier-Stokes-Guided CNN and Sequence-to-Sequence Attention-Based Models 

**Title (ZH)**: 基于Navier-Stokes引导的CNN和序列到序列注意力机制模型的多模态漂移预测 

**Authors**: Rahmat K. Adesunkanmi, Alexander W. Brandt, Masoud Deylami, Gustavo A. Giraldo Echeverri, Hamidreza Karbasian, Adel Alaeddini  

**Link**: [PDF](https://arxiv.org/pdf/2508.18284)  

**Abstract**: Accurately predicting the drift (displacement) of leeway objects in maritime environments remains a critical challenge, particularly in time-sensitive scenarios such as search and rescue operations. In this study, we propose a multi-modal machine learning framework that integrates Sentence Transformer embeddings with attention-based sequence-to-sequence architectures to predict the drift of leeway objects in water. We begin by experimentally collecting environmental and physical data, including water current and wind velocities, object mass, and surface area, for five distinct leeway objects. Using simulated data from a Navier-Stokes-based model to train a convolutional neural network on geometrical image representations, we estimate drag and lift coefficients of the leeway objects. These coefficients are then used to derive the net forces responsible for driving the objects' motion. The resulting time series, comprising physical forces, environmental velocities, and object-specific features, combined with textual descriptions encoded via a language model, are inputs to attention-based sequence-to-sequence long-short-term memory and Transformer models, to predict future drift trajectories. We evaluate the framework across multiple time horizons ($1$, $3$, $5$, and $10$ seconds) and assess its generalization across different objects. We compare our approach against a fitted physics-based model and traditional machine learning methods, including recurrent neural networks and temporal convolutional neural networks. Our results show that these multi-modal models perform comparably to traditional models while also enabling longer-term forecasting in place of single-step prediction. Overall, our findings demonstrate the ability of a multi-modal modeling strategy to provide accurate and adaptable predictions of leeway object drift in dynamic maritime conditions. 

**Abstract (ZH)**: 准确预测海上自由漂移物体的漂移（位移）在紧迫情境如搜救操作中仍是一个关键挑战。本文提出了一种多模态机器学习框架，该框架结合了Sentence Transformer嵌入与基于注意力的序列到序列架构，用于预测水上自由漂移物体的漂移。通过实验收集了包括水流和风速、物体质量和表面积在内的五个不同自由漂移物体的环境和物理数据。利用基于Navier-Stokes模型的模拟数据，对几何图像表示进行卷积神经网络训练，以估算自由漂移物体的阻力和升力系数。这些系数随后用于推导出驱动物体运动的净力。结合各种物理力、环境速度、物体特性和通过语言模型编码的文本描述，作为基于注意力的序列到序列长短期记忆和Transformer模型的输入，以预测未来的漂移轨迹。我们跨越多个时间范围（1秒、3秒、5秒和10秒）评估了该框架，并在不同物体间评估了其泛化能力。我们将我们的方法与拟合的基于物理模型和传统的机器学习方法，包括循环神经网络和时序卷积神经网络进行比较。结果显示，这些多模态模型在与传统模型相当的同时，还能够进行更长期的预测而非单步预测。总体而言，我们的研究结果表明，多模态建模策略在动态海况下能够提供准确且适应性较强的自由漂移物体漂移预测。 

---
# Technology-assisted Personalized Yoga for Better Health - Challenges and Outlook 

**Title (ZH)**: 科技辅助个性化瑜伽以促进健康：挑战与前景 

**Authors**: Vivek Kumar, Himanshu Sahu, Hari Prabhat Gupta, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2508.18283)  

**Abstract**: Yoga is a discipline of physical postures, breathing techniques, and meditative practices rooted in ancient Indian traditions, now embraced worldwide for promoting overall well-being and inner balance. The practices are a large set of items, our term for executable actions like physical poses or breath exercises, to offer for a person's well-being. However, to get benefits of Yoga tailored to a person's unique needs, a person needs to (a) discover their subset from the large and seemingly complex set with inter-dependencies, (b) continue to follow them with interest adjusted to their changing abilities and near-term objectives, and (c) as appropriate, adapt to alternative items based on changing environment and the person's health conditions. In this vision paper, we describe the challenges for the Yoga personalization problem. Next, we sketch a preliminary approach and use the experience to provide an outlook on solving the challenging problem using existing and novel techniques from a multidisciplinary computing perspective. To the best of our knowledge, this is the first paper that comprehensively examines decision support issues around Yoga personalization, from pose sensing to recommendation of corrections for a complete regimen, and illustrates with a case study of Surya Namaskar -- a set of 12 choreographed poses. 

**Abstract (ZH)**: 瑜伽是一种根植于古印度传统、涉及身体姿态、呼吸技巧和冥想实践的discipline，如今被全球广泛接受，用以促进身心健康和内在平衡。这些实践是一系列可执行的操作，例如身体姿势或呼吸练习，旨在服务于个人的福祉。然而，为了根据个人的特殊需求获得个性化的瑜伽益处，个人需要(a)从看似复杂且具有相互依赖性的庞大系列中发现适合自己的子集，(b)保持兴趣，并根据能力和近期目标进行调整，以及(c)根据环境变化和健康状况适时调整。在本文中，我们描述了瑜伽个性化面临的挑战。接下来，我们勾勒了一个初步方法，并利用经验提供了从多学科计算视角解决这一挑战性问题的展望。据我们所知，这是第一篇全面探讨瑜伽个性化决策支持问题的论文，从姿态感知到为完整计划推荐矫正措施，并以一套12个编排好的体式——苏里亚 Namaskar——为例进行了阐述。 

---
