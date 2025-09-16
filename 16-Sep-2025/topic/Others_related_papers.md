# E2-BKI: Evidential Ellipsoidal Bayesian Kernel Inference for Uncertainty-aware Gaussian Semantic Mapping 

**Title (ZH)**: E2-BKI: 证据椭球贝叶斯核推理在不确定性感知高斯语义映射中的应用 

**Authors**: Junyoung Kim, Minsik Jeon, Jihong Min, Kiho Kwak, Junwon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2509.11964)  

**Abstract**: Semantic mapping aims to construct a 3D semantic representation of the environment, providing essential knowledge for robots operating in complex outdoor settings. While Bayesian Kernel Inference (BKI) addresses discontinuities of map inference from sparse sensor data, existing semantic mapping methods suffer from various sources of uncertainties in challenging outdoor environments. To address these issues, we propose an uncertainty-aware semantic mapping framework that handles multiple sources of uncertainties, which significantly degrade mapping performance. Our method estimates uncertainties in semantic predictions using Evidential Deep Learning and incorporates them into BKI for robust semantic inference. It further aggregates noisy observations into coherent Gaussian representations to mitigate the impact of unreliable points, while employing geometry-aligned kernels that adapt to complex scene structures. These Gaussian primitives effectively fuse local geometric and semantic information, enabling robust, uncertainty-aware mapping in complex outdoor scenarios. Comprehensive evaluation across diverse off-road and urban outdoor environments demonstrates consistent improvements in mapping quality, uncertainty calibration, representational flexibility, and robustness, while maintaining real-time efficiency. 

**Abstract (ZH)**: 面向复杂室外环境的不确定性aware语义映射框架 

---
# A Software-Only Post-Processor for Indexed Rotary Machining on GRBL-Based CNCs 

**Title (ZH)**: 基于GRBL控制的 indexed rotary 加工软件后处理器 

**Authors**: Pedro Portugal, Damian D. Venghaus, Diego Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2509.11433)  

**Abstract**: Affordable desktop CNC routers are common in education, prototyping, and makerspaces, but most lack a rotary axis, limiting fabrication of rotationally symmetric or multi-sided parts. Existing solutions often require hardware retrofits, alternative controllers, or commercial CAM software, raising cost and complexity. This work presents a software-only framework for indexed rotary machining on GRBL-based CNCs. A custom post-processor converts planar toolpaths into discrete rotary steps, executed through a browser-based interface. While not equivalent to continuous 4- axis machining, the method enables practical rotary-axis fabrication using only standard, off-the- shelf mechanics, without firmware modification. By reducing technical and financial barriers, the framework expands access to multi-axis machining in classrooms, makerspaces, and small workshops, supporting hands-on learning and rapid prototyping. 

**Abstract (ZH)**: 基于GRBL的 CNC 系统的软件化索引旋转加工框架 

---
# TRUST 2025: SCRITA and RTSS @ RO-MAN 2025 

**Title (ZH)**: TRUST 2025: SCRITA和RTSS @ RO-MAN 2025 

**Authors**: Alessandra Rossi, Patrick Holthaus, Gabriella Lakatos, Sílvia Moros, Ali Fallahi, Murat Kirtay, Marie Postma, Erhan Oztop  

**Link**: [PDF](https://arxiv.org/pdf/2509.11402)  

**Abstract**: The TRUST workshop is the result of a collaboration between two established workshops in the field of Human-Robot Interaction: SCRITA (Trust, Acceptance and Social Cues in Human-Robot Interaction) and RTSS (Robot Trust for Symbiotic Societies). This joint initiative brings together the complementary goals of these workshops to advance research on trust from both the human and robot perspectives.
Website: this https URL 

**Abstract (ZH)**: TRUST研讨会是人机交互领域两个成熟研讨会SCRITA（Trust, Acceptance and Social Cues in Human-Robot Interaction）和RTSS（Robot Trust for Symbiotic Societies）合作的成果。这一联合倡议将这两个研讨会互补的目标汇集起来，以从人类和机器人两个视角推进信任研究。 

---
# Deceptive Risk Minimization: Out-of-Distribution Generalization by Deceiving Distribution Shift Detectors 

**Title (ZH)**: 欺骗风险最小化：通过欺骗分布转移检测器实现分布外泛化 

**Authors**: Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2509.12081)  

**Abstract**: This paper proposes deception as a mechanism for out-of-distribution (OOD) generalization: by learning data representations that make training data appear independent and identically distributed (iid) to an observer, we can identify stable features that eliminate spurious correlations and generalize to unseen domains. We refer to this principle as deceptive risk minimization (DRM) and instantiate it with a practical differentiable objective that simultaneously learns features that eliminate distribution shifts from the perspective of a detector based on conformal martingales while minimizing a task-specific loss. In contrast to domain adaptation or prior invariant representation learning methods, DRM does not require access to test data or a partitioning of training data into a finite number of data-generating domains. We demonstrate the efficacy of DRM on numerical experiments with concept shift and a simulated imitation learning setting with covariate shift in environments that a robot is deployed in. 

**Abstract (ZH)**: 本文提出欺骗作为一种机制实现领域外（OOD）泛化：通过学习数据表示使得训练数据对于观察者看起来是独立且相同分布（iid）的，我们可以识别出稳定的不依赖于虚假相关性的特征，并将这些特征泛化到未见过的领域。我们将这一原则称为欺骗风险最小化（DRM），并提出了一种实用的可微目标函数，该函数同时从基于容许 martingales 的检测器的角度学习消除分布偏移的特征，同时最小化特定任务的损失。与领域适应或先验不变表示学习方法不同，DRM 不需要访问测试数据或对训练数据进行有限的领域划分。我们通过概念偏移的数值实验和机器人部署环境中协变量偏移的模拟 imitative 学习设置来验证 DRM 的有效性。 

---
# Bridging Engineering and AI Planning through Model-Based Knowledge Transformation for the Validation of Automated Production System Variants 

**Title (ZH)**: 通过基于模型的知识转换桥梁工程与AI规划以验证自动化生产系统变体 

**Authors**: Hamied Nabizada, Lasse Beers, Alain Chahine, Felix Gehlhoff, Oliver Niggemann, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2509.12091)  

**Abstract**: Engineering models created in Model-Based Systems Engineering (MBSE) environments contain detailed information about system structure and behavior. However, they typically lack symbolic planning semantics such as preconditions, effects, and constraints related to resource availability and timing. This limits their ability to evaluate whether a given system variant can fulfill specific tasks and how efficiently it performs compared to alternatives.
To address this gap, this paper presents a model-driven method that enables the specification and automated generation of symbolic planning artifacts within SysML-based engineering models. A dedicated SysML profile introduces reusable stereotypes for core planning constructs. These are integrated into existing model structures and processed by an algorithm that generates a valid domain file and a corresponding problem file in Planning Domain Definition Language (PDDL). In contrast to previous approaches that rely on manual transformations or external capability models, the method supports native integration and maintains consistency between engineering and planning artifacts.
The applicability of the method is demonstrated through a case study from aircraft assembly. The example illustrates how existing engineering models are enriched with planning semantics and how the proposed workflow is applied to generate consistent planning artifacts from these models. The generated planning artifacts enable the validation of system variants through AI planning. 

**Abstract (ZH)**: 基于模型的系统工程环境中构建的工程模型包含系统的详细结构和行为信息，但通常缺乏与资源可用性和时间相关的前提、效果和约束等符号规划语义。这限制了它们评估给定系统变体是否能执行特定任务及其相对于替代方案的高效性的能力。

为解决这一问题，本文提出了一种模型驱动的方法，能够在SysML基础的工程模型中指定和自动生成符号规划元素。一种专用的SysML配置文件引入了可重用的核心规划构造体模形。这些模形被集成到现有的模型结构中，并由算法生成有效的领域文件和相应的规划域定义语言(PDDL)问题文件。与依赖于手动转换或外部能力模型的先前方法不同，该方法支持原生集成，并保持工程和规划元素的一致性。

该方法的应用通过一架飞机装配的案例研究进行演示。示例说明了如何利用现有工程模型来增强规划语义，并展示了如何应用提出的工作流从这些模型中生成一致的规划元素。生成的规划元素通过AI规划来验证系统变体。 

---
# Human-AI Use Patterns for Decision-Making in Disaster Scenarios: A Systematic Review 

**Title (ZH)**: 人类与人工智能在灾害情景决策中的使用模式：一项系统性回顾 

**Authors**: Emmanuel Adjei Domfeh, Christopher L. Dancy  

**Link**: [PDF](https://arxiv.org/pdf/2509.12034)  

**Abstract**: In high-stakes disaster scenarios, timely and informed decision-making is critical yet often challenged by uncertainty, dynamic environments, and limited resources. This paper presents a systematic review of Human-AI collaboration patterns that support decision-making across all disaster management phases. Drawing from 51 peer-reviewed studies, we identify four major categories: Human-AI Decision Support Systems, Task and Resource Coordination, Trust and Transparency, and Simulation and Training. Within these, we analyze sub-patterns such as cognitive-augmented intelligence, multi-agent coordination, explainable AI, and virtual training environments. Our review highlights how AI systems may enhance situational awareness, improves response efficiency, and support complex decision-making, while also surfacing critical limitations in scalability, interpretability, and system interoperability. We conclude by outlining key challenges and future research directions, emphasizing the need for adaptive, trustworthy, and context-aware Human-AI systems to improve disaster resilience and equitable recovery outcomes. 

**Abstract (ZH)**: 在高风险灾难场景中，及时且知情的决策至关重要，但往往受到不确定性、动态环境和有限资源的挑战。本文系统回顾了支持灾难管理各个阶段决策的人工智能协作模式。基于51篇同行评审研究，我们识别出四大类别：人机决策支持系统、任务和资源协调、信任与透明度、以及模拟与培训。在这些类别中，我们分析了子模式，如认知增强智能、多智能体协调、可解释人工智能和虚拟培训环境。回顾结果突出了人工智能系统在提高态势感知、提升响应效率、支持复杂决策方面的增强作用，同时也揭示了其在可扩展性、可解释性和系统互操作性方面的关键局限性。最后，我们概述了关键挑战和未来研究方向，强调需要发展适应性强、可信赖且情境意识的人机系统，以提高灾难弹性和公平恢复结果。 

---
# MusicSwarm: Biologically Inspired Intelligence for Music Composition 

**Title (ZH)**: MusicSwarm：受生物启发的音乐创作智能 

**Authors**: Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2509.11973)  

**Abstract**: We show that coherent, long-form musical composition can emerge from a decentralized swarm of identical, frozen foundation models that coordinate via stigmergic, peer-to-peer signals, without any weight updates. We compare a centralized multi-agent system with a global critic to a fully decentralized swarm in which bar-wise agents sense and deposit harmonic, rhythmic, and structural cues, adapt short-term memory, and reach consensus. Across symbolic, audio, and graph-theoretic analyses, the swarm yields superior quality while delivering greater diversity and structural variety and leads across creativity metrics. The dynamics contract toward a stable configuration of complementary roles, and self-similarity networks reveal a small-world architecture with efficient long-range connectivity and specialized bridging motifs, clarifying how local novelties consolidate into global musical form. By shifting specialization from parameter updates to interaction rules, shared memory, and dynamic consensus, MusicSwarm provides a compute- and data-efficient route to long-horizon creative structure that is immediately transferable beyond music to collaborative writing, design, and scientific discovery. 

**Abstract (ZH)**: 我们展示了协调的长形式音乐创作可以从一群通过stigmergic、peer-to-peer信号进行协调的去中心化且相同的“冻结”基础模型中 Emerge 出来，而不需要任何权重更新。我们将一个集中式的多Agent系统与全球批评家进行比较，该系统与一个完全去中心化的群体进行对比，在该群体中，小节级别的Agent感知并沉积和弦、节奏和结构暗示，适配短期记忆，并达成共识。在符号、音频和图论分析方面，群体提供了更高的质量同时提供更多样性和结构性变化，并在创造力指标上表现出色。系统动态向互补角色的稳定配置收缩，并且自相似网络揭示了一种小世界架构，具有高效的长程连接和专门化的桥接模式，阐明了局部新颖性如何在音乐结构中巩固。通过从参数更新转向交互规则、共享记忆和动态共识的专门化，MusicSwarm 提供了一条计算和数据高效的路径，以生成长期的创造性结构，并且这种结构可以立即转移到音乐之外的协作写作、设计和科学发现中。 

---
# How to Evaluate Medical AI 

**Title (ZH)**: 如何评价医疗人工智能 

**Authors**: Ilia Kopanichuk, Petr Anokhin, Vladimir Shaposhnikov, Vladimir Makharev, Ekaterina Tsapieva, Iaroslav Bespalov, Dmitry V. Dylov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2509.11941)  

**Abstract**: The integration of artificial intelligence (AI) into medical diagnostic workflows requires robust and consistent evaluation methods to ensure reliability, clinical relevance, and the inherent variability in expert judgments. Traditional metrics like precision and recall often fail to account for the inherent variability in expert judgments, leading to inconsistent assessments of AI performance. Inter-rater agreement statistics like Cohen's Kappa are more reliable but they lack interpretability. We introduce Relative Precision and Recall of Algorithmic Diagnostics (RPAD and RRAD) - a new evaluation metrics that compare AI outputs against multiple expert opinions rather than a single reference. By normalizing performance against inter-expert disagreement, these metrics provide a more stable and realistic measure of the quality of predicted diagnosis. In addition to the comprehensive analysis of diagnostic quality measures, our study contains a very important side result. Our evaluation methodology allows us to avoid selecting diagnoses from a limited list when evaluating a given case. Instead, both the models being tested and the examiners verifying them arrive at a free-form diagnosis. In this automated methodology for establishing the identity of free-form clinical diagnoses, a remarkable 98% accuracy becomes attainable. We evaluate our approach using 360 medical dialogues, comparing multiple large language models (LLMs) against a panel of physicians. Large-scale study shows that top-performing models, such as DeepSeek-V3, achieve consistency on par with or exceeding expert consensus. Moreover, we demonstrate that expert judgments exhibit significant variability - often greater than that between AI and humans. This finding underscores the limitations of any absolute metrics and supports the need to adopt relative metrics in medical AI. 

**Abstract (ZH)**: 将人工智能集成到医疗诊断工作流程中需要 robust 和一致的评估方法以确保可靠性、临床相关性以及专家判断的固有变异。传统的精确度和召回率等评价指标往往无法考虑专家判断的固有变异，导致对人工智能性能的一致性评估失效。Cohen's Kappa等评分者一致性统计指标更可靠，但缺乏可解释性。我们引入了算法诊断的相对精确度和召回率（RPAD和RRAD）——一种新的评价指标，将人工智能输出与多种专家意见进行比较，而非单一参考。通过正常化性能以消除专家间分歧，这些指标提供了更稳定和现实的预测诊断质量衡量标准。除了全面分析诊断质量指标外，我们的研究还包括一个非常重要的附带结果。我们的评价方法使我们在评估某一病例时避免从有限的诊断列表中选择诊断。相反，测试中的模型和验证者均得出自由格式的诊断。在这种自动建立自由格式临床诊断身份的方法中，98%的准确性变得可实现。我们使用360份医疗对话评估了该方法，将多种大型语言模型（LLMs）与一组医生进行了对比。大规模研究表明，表现最佳的模型，如DeepSeek-V3，在一致性方面达到或超过了专家共识。此外，我们证明了专家判断存在显著差异——通常大于人工智能与人类之间的差异。这一发现凸显了任何绝对指标的局限性，并支持在医疗人工智能中采用相对评价指标的必要性。 

---
# Neuromorphic Intelligence 

**Title (ZH)**: 神经形态智能 

**Authors**: Marcel van Gerven  

**Link**: [PDF](https://arxiv.org/pdf/2509.11940)  

**Abstract**: Neuromorphic computing seeks to replicate the remarkable efficiency, flexibility, and adaptability of the human brain in artificial systems. Unlike conventional digital approaches, which depend on massive computational and energy resources, neuromorphic systems exploit brain-inspired principles of computation to achieve orders of magnitude greater energy efficiency. By drawing on insights from artificial intelligence, neuroscience, physics, chemistry, and materials science, neuromorphic computing promises to deliver intelligent systems that are sustainable, transparent, and widely accessible. A central challenge, however, is to identify a unifying theoretical framework capable of bridging these diverse disciplines. We argue that dynamical systems theory provides such a foundation. Rooted in differential calculus, it offers a principled language for modeling inference, learning, and control in both natural and artificial substrates. Within this framework, noise can be harnessed as a resource for learning, while differential genetic programming enables the discovery of dynamical systems that implement adaptive behaviors. Embracing this perspective paves the way toward emergent neuromorphic intelligence, where intel- ligent behavior arises from the dynamics of physical substrates, advancing both the science and sustainability of AI. 

**Abstract (ZH)**: 神经形态计算旨在复制人类大脑在人工系统中的卓越效率、灵活性和适应性。与依赖庞大计算和能源资源的传统数字方法不同，神经形态系统利用神经启发的计算原则，实现了数量级的能源效率提升。通过借鉴人工智能、神经科学、物理学、化学和材料科学的见解，神经形态计算有望提供可持续、透明且普及的智能系统。然而，一个核心挑战是识别一种能够跨这些学科提供统一理论框架的方法。我们认为，动力系统理论正是这样的基础。根植于微分 calculus，它提供了一种原则性的语言来建模自然和人工子体中的推断、学习和控制。在这种框架内，噪声可以作为一种学习资源被利用，而微分遗传编程则使发现实现适应性行为的动力系统成为可能。采纳这一视角为新兴的神经形态智能铺平了道路，在这种智能中，智能行为来源于物理子体的动力学，从而推动了人工智能的科学与可持续性。 

---
# BuildingGym: An open-source toolbox for AI-based building energy management using reinforcement learning 

**Title (ZH)**: BuildingGym: 基于强化学习的建筑能源管理AI工具箱 

**Authors**: Xilei Dai, Ruotian Chen, Songze Guan, Wen-Tai Li, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11922)  

**Abstract**: Reinforcement learning (RL) has proven effective for AI-based building energy management. However, there is a lack of flexible framework to implement RL across various control problems in building energy management. To address this gap, we propose BuildingGym, an open-source tool designed as a research-friendly and flexible framework for training RL control strategies for common challenges in building energy management. BuildingGym integrates EnergyPlus as its core simulator, making it suitable for both system-level and room-level control. Additionally, BuildingGym is able to accept external signals as control inputs instead of taking the building as a stand-alone entity. This feature makes BuildingGym applicable for more flexible environments, e.g. smart grid and EVs community. The tool provides several built-in RL algorithms for control strategy training, simplifying the process for building managers to obtain optimal control strategies. Users can achieve this by following a few straightforward steps to configure BuildingGym for optimization control for common problems in the building energy management field. Moreover, AI specialists can easily implement and test state-of-the-art control algorithms within the platform. BuildingGym bridges the gap between building managers and AI specialists by allowing for the easy configuration and replacement of RL algorithms, simulators, and control environments or problems. With BuildingGym, we efficiently set up training tasks for cooling load management, targeting both constant and dynamic cooling load management. The built-in algorithms demonstrated strong performance across both tasks, highlighting the effectiveness of BuildingGym in optimizing cooling strategies. 

**Abstract (ZH)**: 基于 reinforcement learning 的楼宇能源管理灵活框架：BuildingGym 

---
# HeLoFusion: An Efficient and Scalable Encoder for Modeling Heterogeneous and Multi-Scale Interactions in Trajectory Prediction 

**Title (ZH)**: HeLoFusion：一种用于建模轨迹预测中异构和多尺度交互的高效可扩展编码器 

**Authors**: Bingqing Wei, Lianmin Chen, Zhongyu Xia, Yongtao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11719)  

**Abstract**: Multi-agent trajectory prediction in autonomous driving requires a comprehensive understanding of complex social dynamics. Existing methods, however, often struggle to capture the full richness of these dynamics, particularly the co-existence of multi-scale interactions and the diverse behaviors of heterogeneous agents. To address these challenges, this paper introduces HeLoFusion, an efficient and scalable encoder for modeling heterogeneous and multi-scale agent interactions. Instead of relying on global context, HeLoFusion constructs local, multi-scale graphs centered on each agent, allowing it to effectively model both direct pairwise dependencies and complex group-wise interactions (\textit{e.g.}, platooning vehicles or pedestrian crowds). Furthermore, HeLoFusion tackles the critical challenge of agent heterogeneity through an aggregation-decomposition message-passing scheme and type-specific feature networks, enabling it to learn nuanced, type-dependent interaction patterns. This locality-focused approach enables a principled representation of multi-level social context, yielding powerful and expressive agent embeddings. On the challenging Waymo Open Motion Dataset, HeLoFusion achieves state-of-the-art performance, setting new benchmarks for key metrics including Soft mAP and minADE. Our work demonstrates that a locality-grounded architecture, which explicitly models multi-scale and heterogeneous interactions, is a highly effective strategy for advancing motion forecasting. 

**Abstract (ZH)**: 多Agent轨迹预测在自主驾驶中需要全面理解复杂的社交动态。为了应对这一挑战，本文提出了HeLoFusion，一种高效且可扩展的编码器，用于建模异构和多尺度Agent交互。HeLoFusion通过围绕每个Agent构建局部的多尺度图，而不是依赖全局上下文，有效地建模了直接的成对依赖关系和复杂的群体交互（例如车队或行人 crowds）。此外，HeLoFusion通过聚合-分解消息传递方案和类型特定特征网络解决了Agent异构性的关键挑战，使其能够学习精细的、类型依赖的交互模式。这种以局部为中心的方法为多层次的社会背景提供了一个原则性的表示，从而生成强大的且具有表达力的Agent嵌入。在具有挑战性的Waymo Open Motion数据集中，HeLoFusion实现了最先进的性能，为Soft mAP和minADE等关键指标设立了新的基准。我们的研究证明，一个基于局部的架构，明确建模多尺度和异构交互，是一种极具效力的运动预测方法。 

---
# AMLNet: A Knowledge-Based Multi-Agent Framework to Generate and Detect Realistic Money Laundering Transactions 

**Title (ZH)**: AMLNet：一种基于知识的多代理框架，用于生成和检测现实中的洗钱交易 

**Authors**: Sabin Huda, Ernest Foo, Zahra Jadidi, MA Hakim Newton, Abdul Sattar  

**Link**: [PDF](https://arxiv.org/pdf/2509.11595)  

**Abstract**: Anti-money laundering (AML) research is constrained by the lack of publicly shareable, regulation-aligned transaction datasets. We present AMLNet, a knowledge-based multi-agent framework with two coordinated units: a regulation-aware transaction generator and an ensemble detection pipeline. The generator produces 1,090,173 synthetic transactions (approximately 0.16\% laundering-positive) spanning core laundering phases (placement, layering, integration) and advanced typologies (e.g., structuring, adaptive threshold behavior). Regulatory alignment reaches 75\% based on AUSTRAC rule coverage (Section 4.2), while a composite technical fidelity score of 0.75 summarizes temporal, structural, and behavioral realism components (Section 4.4). The detection ensemble achieves F1 0.90 (precision 0.84, recall 0.97) on the internal test partitions of AMLNet and adapts to the external SynthAML dataset, indicating architectural generalizability across different synthetic generation paradigms. We provide multi-dimensional evaluation (regulatory, temporal, network, behavioral) and release the dataset (Version 1.0, this https URL), to advance reproducible and regulation-conscious AML experimentation. 

**Abstract (ZH)**: 基于知识的多代理框架AMLNet及其应用：面向反洗钱的研究 

---
# Formal Reasoning for Intelligent QA Systems: A Case Study in the Educational Domain 

**Title (ZH)**: 智能问答系统中的形式推理：教育领域的案例研究 

**Authors**: Tuan Bui, An Nguyen, Phat Thai, Minh Hua, Ngan Pham L.N., Ngan Pham T.B., Dung Le, Long Nguyen, Thanh-Tung Tran, Thang Bui, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2509.11572)  

**Abstract**: Reasoning is essential for closed-domain QA systems in which procedural correctness and policy compliance are critical. While large language models (LLMs) have shown strong performance on many reasoning tasks, recent work reveals that their reasoning traces are often unfaithful - serving more as plausible justifications than as causally grounded derivations. Efforts to combine LLMs with symbolic engines (e.g., Prover9, Z3) have improved reliability but remain limited to static forms of logic, struggling with dynamic, state-based reasoning such as multi-step progressions and conditional transitions.
In this paper, we propose MCFR (Model Checking for Formal Reasoning), a neuro-symbolic framework that integrates LLMs with model checking to support property verification. MCFR translates natural language into formal specifications and verifies them over transition models. To support evaluation, we introduce EduMC-QA, a benchmark dataset grounded in real academic procedures. Our results show that MCFR improves reasoning faithfulness and interpretability, offering a viable path toward verifiable QA in high-stakes closed-domain applications. In addition to evaluating MCFR, we compare its performance with state-of-the-art LLMs such as ChatGPT, DeepSeek, and Claude to contextualize its effectiveness. 

**Abstract (ZH)**: 基于模型检验的神经符号框架：改进形式推理的合理性和可解释性 

---
# Task Decoding based on Eye Movements using Synthetic Data Augmentation 

**Title (ZH)**: 基于合成数据增强的眼动任务解码 

**Authors**: Shanmuka Sadhu, Arca Baran, Preeti Pandey, Ayush Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.11547)  

**Abstract**: Machine learning has been extensively used in various applications related to eye-tracking research. Understanding eye movement is one of the most significant subsets of eye-tracking research that reveals the scanning pattern of an individual. Researchers have thoroughly analyzed eye movement data to understand various eye-tracking applications, such as attention mechanisms, navigational behavior, task understanding, etc. The outcome of traditional machine learning algorithms used for decoding tasks based on eye movement data has received a mixed reaction to Yarbus' claim that it is possible to decode the observer's task from their eye movements. In this paper, to support the hypothesis by Yarbus, we are decoding tasks categories while generating synthetic data samples using well-known Synthetic Data Generators CTGAN and its variations such as CopulaGAN and Gretel AI Synthetic Data generators on available data from an in-person user study. Our results show that augmenting more eye movement data combined with additional synthetically generated improves classification accuracy even with traditional machine learning algorithms. We see a significant improvement in task decoding accuracy from 28.1% using Random Forest to 82% using Inception Time when five times more data is added in addition to the 320 real eye movement dataset sample. Our proposed framework outperforms all the available studies on this dataset because of the use of additional synthetic datasets. We validated our claim with various algorithms and combinations of real and synthetic data to show how decoding accuracy increases with the increase in the augmentation of generated data to real data. 

**Abstract (ZH)**: 机器学习在与眼动追踪研究相关的各种应用中被广泛使用。理解眼动是眼动追踪研究中最重要的一组，揭示了个别个体的追踪模式。研究者已经详细分析了眼动数据，以理解诸如注意力机制、导航行为、任务理解等多种眼动追踪应用。传统机器学习算法用于基于眼动数据的解码任务的结果对Yarbus的主张产生了混合反应，即可以从观察者的眼动中解码其任务。在这篇论文中，为了支持Yarbus的假设，我们利用著名的合成数据生成器CTGAN及其变种CopulaGAN和Gretel AI合成数据生成器，在实际用户的在场用户研究数据上生成合成数据样本来进行任务类别解码。我们的结果显示，结合更多眼动数据与额外生成的合成数据可以提高分类准确性，即使在使用传统机器学习算法时也是如此。当在320个真实眼动数据集样本的基础上增加五倍的数据时，从随机森林的28.1%提高到使用Inception Time的82%，任务解码准确性有了显著提升。由于采用了额外的合成数据集，我们提出的框架在该数据集上超过了所有现有的研究。我们通过使用各种算法和真实数据与合成数据的组合验证了我们的主张，以展示生成数据增加对真实数据的解码准确性增加的影响。 

---
# Knowledge-Guided Adaptive Mixture of Experts for Precipitation Prediction 

**Title (ZH)**: 知识引导自适应专家混合模型 для 降水预测 

**Authors**: Chen Jiang, Kofi Osei, Sai Deepthi Yeddula, Dongji Feng, Wei-Shinn Ku  

**Link**: [PDF](https://arxiv.org/pdf/2509.11459)  

**Abstract**: Accurate precipitation forecasting is indispensable in agriculture, disaster management, and sustainable strategies. However, predicting rainfall has been challenging due to the complexity of climate systems and the heterogeneous nature of multi-source observational data, including radar, satellite imagery, and surface-level measurements. The multi-source data vary in spatial and temporal resolution, and they carry domain-specific features, making it challenging for effective integration in conventional deep learning models. Previous research has explored various machine learning techniques for weather prediction; however, most struggle with the integration of data with heterogeneous modalities. To address these limitations, we propose an Adaptive Mixture of Experts (MoE) model tailored for precipitation rate prediction. Each expert within the model specializes in a specific modality or spatio-temporal pattern. We also incorporated a dynamic router that learns to assign inputs to the most relevant experts. Our results show that this modular design enhances predictive accuracy and interpretability. In addition to the modeling framework, we introduced an interactive web-based visualization tool that enables users to intuitively explore historical weather patterns over time and space. The tool was designed to support decision-making for stakeholders in climate-sensitive sectors. We evaluated our approach using a curated multimodal climate dataset capturing real-world conditions during Hurricane Ian in 2022. The benchmark results show that the Adaptive MoE significantly outperformed all the baselines. 

**Abstract (ZH)**: 准确的降水预报对于农业、灾害管理和可持续发展战略至关重要。然而，由于气候系统的复杂性和多源观测数据在空间和时间分辨率上的异质性，降水预测一直具有挑战性，这些数据包括雷达、卫星图像和地表测量数据。多源数据具有领域特定的特征，使得在传统深度学习模型中的有效集成变得困难。 previous研究探索了多种机器学习技术用于天气预测，但大多难以整合具有异质模态的数据。为解决这些限制，我们提出了一种适应性专家混合（MoE）模型，专门用于降水率预测。模型中的每个专家专注于特定的模态或时空模式。我们还引入了一个动态路由器，通过学习将输入分配给最相关的专家。结果表明，这种模块化设计提高了预测准确性和可解释性。除了建模框架，我们还引入了一个交互式的基于Web的数据可视化工具，使用户能够直观地探索时空上的历史天气模式。该工具旨在支持气候变化敏感领域利益相关者的决策。我们使用一个精心收集的多模态气候数据集评估了我们的方法，该数据集涵盖了2022年飓风伊恩期间的真实世界条件。基准结果表明，适应性MoE显著优于所有基准模型。 

---
# The power of dynamic causality in observer-based design for soft sensor applications 

**Title (ZH)**: 基于观察者设计的软传感器应用中动态因果性的力量 

**Authors**: William Farlessyost, Sebastian Oberst, Shweta Singh  

**Link**: [PDF](https://arxiv.org/pdf/2509.11336)  

**Abstract**: This paper introduces a novel framework for optimizing observer-based soft sensors through dynamic causality analysis. Traditional approaches to sensor selection often rely on linearized observability indices or statistical correlations that fail to capture the temporal evolution of complex systems. We address this gap by leveraging liquid-time constant (LTC) networks, continuous-time neural architectures with input-dependent time constants, to systematically identify and prune sensor inputs with minimal causal influence on state estimation. Our methodology implements an iterative workflow: training an LTC observer on candidate inputs, quantifying each input's causal impact through controlled perturbation analysis, removing inputs with negligible effect, and retraining until performance degradation occurs. We demonstrate this approach on three mechanistic testbeds representing distinct physical domains: a harmonically forced spring-mass-damper system, a nonlinear continuous stirred-tank reactor, and a predator-prey model following the structure of the Lotka-Volterra model, but with seasonal forcing and added complexity. Results show that our causality-guided pruning consistently identifies minimal sensor sets that align with underlying physics while improving prediction accuracy. The framework automatically distinguishes essential physical measurements from noise and determines when derived interaction terms provide complementary versus redundant information. Beyond computational efficiency, this approach enhances interpretability by grounding sensor selection decisions in dynamic causal relationships rather than static correlations, offering significant benefits for soft sensing applications across process engineering, ecological monitoring, and agricultural domains. 

**Abstract (ZH)**: 一种基于动态因果分析的优化观测器型软传感器框架 

---
# Decoding Plastic Toxicity: An Intelligent Framework for Conflict-Aware Relational Metapath Extraction from Scientific Abstracts 

**Title (ZH)**: 解码塑化剂毒性：一种考虑冲突的智能元路径提取框架从科学摘要中 

**Authors**: Sudeshna Jana, Manjira Sinha, Tirthankar Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.11330)  

**Abstract**: The widespread use of plastics and their persistence in the environment have led to the accumulation of micro- and nano-plastics across air, water, and soil, posing serious health risks including respiratory, gastrointestinal, and neurological disorders. We propose a novel framework that leverages large language models to extract relational metapaths, multi-hop semantic chains linking pollutant sources to health impacts, from scientific abstracts. Our system identifies and connects entities across diverse contexts to construct structured relational metapaths, which are aggregated into a Toxicity Trajectory Graph that traces pollutant propagation through exposure routes and biological systems. Moreover, to ensure consistency and reliability, we incorporate a dynamic evidence reconciliation module that resolves semantic conflicts arising from evolving or contradictory research findings. Our approach demonstrates strong performance in extracting reliable, high-utility relational knowledge from noisy scientific text and offers a scalable solution for mining complex cause-effect structures in domain-specific corpora. 

**Abstract (ZH)**: 塑料的广泛使用及其在环境中的持久存在导致了空气、水和土壤中微塑料和纳米塑料的积累，引发了包括呼吸系统、消化系统和神经系统的严重健康风险。我们提出了一种新的框架，利用大规模语言模型从科学摘要中提取关系元路径，将污染物来源与健康影响通过多跳语义链连接起来。该系统跨不同情境识别和连接实体，构建结构化关系元路径，并将其聚合为一条毒性轨迹图，该图追踪污染物通过暴露途径和生物系统的传播路径。此外，为了确保一致性和可靠性，我们引入了一个动态证据 reconciliation 模块，用于解决由研究成果演变或矛盾引起的语义冲突。我们的方法在从嘈杂的科学文本中提取可靠且高价值的关系知识方面表现出强大的性能，并为在特定领域语料库中挖掘复杂因果结构提供了可扩展的解决方案。 

---
# AI-Generated Content in Cross-Domain Applications: Research Trends, Challenges and Propositions 

**Title (ZH)**: AI生成内容在跨域应用中的研究趋势、挑战与建议 

**Authors**: Jianxin Li, Liang Qu, Taotao Cai, Zhixue Zhao, Nur Al Hasan Haldar, Aneesh Krishna, Xiangjie Kong, Flavio Romero Macau, Tanmoy Chakraborty, Aniket Deroy, Binshan Lin, Karen Blackmore, Nasimul Noman, Jingxian Cheng, Ningning Cui, Jianliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11151)  

**Abstract**: Artificial Intelligence Generated Content (AIGC) has rapidly emerged with the capability to generate different forms of content, including text, images, videos, and other modalities, which can achieve a quality similar to content created by humans. As a result, AIGC is now widely applied across various domains such as digital marketing, education, and public health, and has shown promising results by enhancing content creation efficiency and improving information delivery. However, there are few studies that explore the latest progress and emerging challenges of AIGC across different domains. To bridge this gap, this paper brings together 16 scholars from multiple disciplines to provide a cross-domain perspective on the trends and challenges of AIGC. Specifically, the contributions of this paper are threefold: (1) It first provides a broader overview of AIGC, spanning the training techniques of Generative AI, detection methods, and both the spread and use of AI-generated content across digital platforms. (2) It then introduces the societal impacts of AIGC across diverse domains, along with a review of existing methods employed in these contexts. (3) Finally, it discusses the key technical challenges and presents research propositions to guide future work. Through these contributions, this vision paper seeks to offer readers a cross-domain perspective on AIGC, providing insights into its current research trends, ongoing challenges, and future directions. 

**Abstract (ZH)**: 人工智能生成内容（AIGC）随着能生成不同形式内容（包括文本、图像、视频及其他模态）的能力迅速涌现，并能达到与人类创建的内容相似的质量，现已广泛应用于数字营销、教育和公共卫生等领域，通过提高内容创作效率和改善信息传递表现出良好的成果。然而，鲜有关于AIGC在不同领域最新进展和新兴挑战的研究。为填补这一空白，本文汇集了来自多个学科的16位学者，提供了跨领域的AIGC趋势与挑战视角。具体而言，本文有三大贡献：（1）首先，它提供了AIGC的更广泛的概述，涵盖生成AI的训练技术、检测方法，以及AI生成内容在数字平台上的传播和使用情况。（2）其次，它介绍了AIGC在不同领域的社会影响，并回顾了这些领域内已经使用的方法。（3）最后，它讨论了关键的技术挑战，并提出研究命题以指导未来的工作。通过这些贡献，本文旨在为读者提供AIGC的跨领域视角，洞察其当前的研究趋势、持续的挑战及其未来方向。 

---
# AlignKT: Explicitly Modeling Knowledge State for Knowledge Tracing with Ideal State Alignment 

**Title (ZH)**: AlignKT：通过理想状态对齐Explicitly modeling知识状态的知识追踪 

**Authors**: Jing Xiao, Chang You, Zhiyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11135)  

**Abstract**: Knowledge Tracing (KT) serves as a fundamental component of Intelligent Tutoring Systems (ITS), enabling these systems to monitor and understand learners' progress by modeling their knowledge state. However, many existing KT models primarily focus on fitting the sequences of learners' interactions, and often overlook the knowledge state itself. This limitation leads to reduced interpretability and insufficient instructional support from the ITS. To address this challenge, we propose AlignKT, which employs a frontend-to-backend architecture to explicitly model a stable knowledge state. In this approach, the preliminary knowledge state is aligned with an additional criterion. Specifically, we define an ideal knowledge state based on pedagogical theories as the alignment criterion, providing a foundation for interpretability. We utilize five encoders to implement this set-up, and incorporate a contrastive learning module to enhance the robustness of the alignment process. Through extensive experiments, AlignKT demonstrates superior performance, outperforming seven KT baselines on three real-world datasets. It achieves state-of-the-art results on two of these datasets and exhibits competitive performance on the third. The code of this work is available at this https URL. 

**Abstract (ZH)**: 知识追踪（KT）是智能辅导系统（ITS）的基本组成部分，通过建模学习者的知识状态来监控和理解学习者的发展。然而，许多现有的KT模型主要侧重于拟合学习者交互的序列，往往忽视了知识状态本身。这种局限性导致了解释性和教学支持不足。为了解决这一挑战，我们提出了AlignKT，它采用从前端到后端的架构来明确建模稳定的知识状态。在此方法中，初步的知识状态与额外的标准进行了对齐。具体而言，我们根据教育理论定义了一个理想的知识状态作为对齐标准，为解释提供了基础。我们使用五个编码器来实现这一设置，并引入了一种对比学习模块以增强对齐过程的鲁棒性。通过广泛的实验，AlignKT展示了卓越的性能，在三个真实世界数据集上优于七种KT基准模型，在其中两个数据集上达到了最先进的结果，并在第三个数据集上表现出竞争力。该工作的代码可在以下链接获取：this https URL。 

---
# Neural cellular automata: applications to biology and beyond classical AI 

**Title (ZH)**: 神经细胞自动机：超越经典AI的应用于生物学及其他领域 

**Authors**: Benedikt Hartl, Michael Levin, Léo Pio-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2509.11131)  

**Abstract**: Neural Cellular Automata (NCA) represent a powerful framework for modeling biological self-organization, extending classical rule-based systems with trainable, differentiable (or evolvable) update rules that capture the adaptive self-regulatory dynamics of living matter. By embedding Artificial Neural Networks (ANNs) as local decision-making centers and interaction rules between localized agents, NCA can simulate processes across molecular, cellular, tissue, and system-level scales, offering a multiscale competency architecture perspective on evolution, development, regeneration, aging, morphogenesis, and robotic control. These models not only reproduce biologically inspired target patterns but also generalize to novel conditions, demonstrating robustness to perturbations and the capacity for open-ended adaptation and reasoning. Given their immense success in recent developments, we here review current literature of NCAs that are relevant primarily for biological or bioengineering applications. Moreover, we emphasize that beyond biology, NCAs display robust and generalizing goal-directed dynamics without centralized control, e.g., in controlling or regenerating composite robotic morphologies or even on cutting-edge reasoning tasks such as ARC-AGI-1. In addition, the same principles of iterative state-refinement is reminiscent to modern generative Artificial Intelligence (AI), such as probabilistic diffusion models. Their governing self-regulatory behavior is constraint to fully localized interactions, yet their collective behavior scales into coordinated system-level outcomes. We thus argue that NCAs constitute a unifying computationally lean paradigm that not only bridges fundamental insights from multiscale biology with modern generative AI, but have the potential to design truly bio-inspired collective intelligence capable of hierarchical reasoning and control. 

**Abstract (ZH)**: 神经细胞自动机（NCA）代表了一种强大的框架，用于模拟生物自我组织，通过引入可训练、可微分（或可演化）的更新规则，扩展了经典规则系统，以捕捉生命物质的适应性自我调节动力学。通过嵌入人工神经网络（ANNs）作为局部决策中心和局部代理之间的交互规则，NCA 可以模拟从分子水平到系统水平的过程，提供一个多层次能力架构视角，用于解释进化、发育、再生、衰老、形态发生和机器人控制。这些模型不仅能够重现生物学启发的目标模式，还能泛化到新条件，表现出对干扰的鲁棒性和开放性适应和推理的能力。鉴于其在最近发展中的巨大成功，我们在此回顾了主要适用于生物学或生物工程应用的 NCAs 现有文献。此外，我们强调除了生物学之外，NCAs 在没有集中控制的情况下展现出鲁棒性和泛化的目标导向动力学，例如在控制或再生复合机器人形态学或即使在尖端推理任务（如 ARC-AGI-1）中也是如此。此外，迭代状态细化的原则与现代生成人工智能（AI），如概率扩散模型相似。它们的主导自我调节行为约束于完全局部化交互，但其集体行为会扩展为协调的系统级结果。因此，我们认为 NCAs 构成了一个统一的计算经济范式，不仅连接了多尺度生物学的基本见解与现代生成AI，还有潜力设计真正以生物启发为基础的集体智能，具备分层推理和控制的能力。 

---
# Agentic Lybic: Multi-Agent Execution System with Tiered Reasoning and Orchestration 

**Title (ZH)**: 代理(liby): 分层推理与编排的多Agent执行系统 

**Authors**: Liangxuan Guo, Bin Zhu, Qingqian Tao, Kangning Liu, Xun Zhao, Xianzhe Qin, Jin Gao, Guangfu Hao  

**Link**: [PDF](https://arxiv.org/pdf/2509.11067)  

**Abstract**: Autonomous agents for desktop automation struggle with complex multi-step tasks due to poor coordination and inadequate quality control. We introduce \textsc{Agentic Lybic}, a novel multi-agent system where the entire architecture operates as a finite-state machine (FSM). This core innovation enables dynamic orchestration. Our system comprises four components: a Controller, a Manager, three Workers (Technician for code-based operations, Operator for GUI interactions, and Analyst for decision support), and an Evaluator. The critical mechanism is the FSM-based routing between these components, which provides flexibility and generalization by dynamically selecting the optimal execution strategy for each subtask. This principled orchestration, combined with robust quality gating, enables adaptive replanning and error recovery. Evaluated officially on the OSWorld benchmark, \textsc{Agentic Lybic} achieves a state-of-the-art 57.07\% success rate in 50 steps, substantially outperforming existing methods. Results demonstrate that principled multi-agent orchestration with continuous quality control provides superior reliability for generalized desktop automation in complex computing environments. 

**Abstract (ZH)**: 自主代理人在处理复杂多步任务时因协调不佳和质量控制不足而困难。我们介绍了一种新型多代理系统\textsc{Agentic Lybic}，其整个架构作为有限状态机（FSM）运作。这一核心创新实现了动态编排。该系统包含四个组件：一个控制器、一个管理器、三个工人（用于代码操作的技术员、用于GUI交互的操作员以及用于决策支持的分析员）和一个评估器。关键机制是这些组件之间的基于FSM的路由，它通过动态选择适合每个子任务的最优执行策略提供了灵活性和泛化能力。这种有原则的编排，结合 robust 的质量控制，实现了自适应重规划和错误恢复。在OSWorld基准上正式评估，\textsc{Agentic Lybic} 在50步中实现了57.07%的最优成功率，显著优于现有方法。结果表明，带有连续质量控制的有原则的多代理编排为复杂计算环境中的一般桌面自动化提供了更高的可靠性。 

---
# From Grounding to Skolemization: A Logic-Constrained Vector Symbolic Architecture for Complex Query Answering 

**Title (ZH)**: 从嵌接地步到斯科伦化：一种逻辑约束向量符号架构用于复杂查询回答 

**Authors**: Yuyin Lu, Hegang Chen, Yanghui Rao  

**Link**: [PDF](https://arxiv.org/pdf/2509.10837)  

**Abstract**: Complex Query Answering (CQA) over incomplete Knowledge Graphs (KGs), typically formalized as reasoning with Existential First-Order predicate logic with one free variable (EFO$_1$), faces a fundamental trade-off between logical soundness and computational efficiency. This work establishes the Grounding-Skolemization dichotomy for systematically analyzing CQA methods through the lens of formal logic. While Grounding-based methods inherently suffer from combinatorial explosion, most Skolemization-based methods neglect to explicitly model Skolem functions and compromise logical consistency. To address these limitations, we propose the Logic-constrained Vector Symbolic Architecture (LVSA), a neuro-symbolic framework that unifies a differentiable Skolemization module and a neural negator, as well as a logical constraint-driven optimization protocol to harmonize geometric and logical requirements. Theoretically, LVSA guarantees universality for all EFO$_1$ queries. Empirically, it outperforms state-of-the-art Skolemization-based methods and reduces inference costs by orders of magnitude compared to Grounding-based baselines. 

**Abstract (ZH)**: 复杂查询回答（CQA）在不完整知识图谱（KGs）上的处理：逻辑准确性和计算效率之间的根本权衡被形式化为存在一阶谓词逻辑（EFO$_1$）与一个自由变量的推理问题。本文通过形式逻辑的视角建立了基础性分析框架，即地面化-斯科莱默化二分法。尽管基于地面化的方法固有地面临组合爆炸的问题，大多数基于斯科莱默化的处理方法未能明确建模斯科莱默函数并牺牲逻辑一致性。为解决这些问题，我们提出了逻辑约束向量符号架构（LVSA），这是一种神经符号框架，它统一了可微斯科莱默化模块、神经否定器以及由逻辑约束驱动的优化协议，以协调几何和逻辑需求。理论上，LVSA能够保证所有EFO$_1$查询的普遍性。实验上，它在斯科莱默化基线方法上实现了性能提升，并将推理成本降低了数量级。 

---
# AgentArch: A Comprehensive Benchmark to Evaluate Agent Architectures in Enterprise 

**Title (ZH)**: AgentArch: 企业中代理架构评估的综合基准 

**Authors**: Tara Bogavelli, Roshnee Sharma, Hari Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2509.10769)  

**Abstract**: While individual components of agentic architectures have been studied in isolation, there remains limited empirical understanding of how different design dimensions interact within complex multi-agent systems. This study aims to address these gaps by providing a comprehensive enterprise-specific benchmark evaluating 18 distinct agentic configurations across state-of-the-art large language models. We examine four critical agentic system dimensions: orchestration strategy, agent prompt implementation (ReAct versus function calling), memory architecture, and thinking tool integration. Our benchmark reveals significant model-specific architectural preferences that challenge the prevalent one-size-fits-all paradigm in agentic AI systems. It also reveals significant weaknesses in overall agentic performance on enterprise tasks with the highest scoring models achieving a maximum of only 35.3\% success on the more complex task and 70.8\% on the simpler task. We hope these findings inform the design of future agentic systems by enabling more empirically backed decisions regarding architectural components and model selection. 

**Abstract (ZH)**: 尽管个体组件在代理架构中的异构性研究已有所涉及，但不同设计维度在复杂多代理系统中的相互作用仍缺乏实证理解。本研究旨在通过对企业特定的基准评估最先进的大型语言模型中的18种不同代理配置，来填补这一空白。我们考察了四个关键的代理系统维度：协调策略、代理指令实现（ReAct与函数调用）、记忆架构以及思维工具集成。我们的基准测试揭示了显著的模型特定架构偏好，这挑战了代理型AI系统中普遍存在的“一刀切”范式。此外，该基准测试还揭示了代理型系统在企业任务上整体表现的显著局限性，最高得分模型在更复杂任务中仅达到35.3%的成功率，在较简单任务中达到70.8%的成功率。希望这些发现能为未来代理型系统的构建提供更有依据的设计决策，特别是在架构组件和模型选择方面。 

---
# AI Answer Engine Citation Behavior An Empirical Analysis of the GEO16 Framework 

**Title (ZH)**: AI答案引擎引用行为：GEO16框架的实证分析 

**Authors**: Arlen Kumar, Leanid Palkhouski  

**Link**: [PDF](https://arxiv.org/pdf/2509.10762)  

**Abstract**: AI answer engines increasingly mediate access to domain knowledge by generating responses and citing web sources. We introduce GEO-16, a 16 pillar auditing framework that converts on page quality signals into banded pillar scores and a normalized GEO score G that ranges from 0 to 1. Using 70 product intent prompts, we collected 1,702 citations across three engines (Brave Summary, Google AI Overviews, and Perplexity) and audited 1,100 unique URLs. In our corpus, the engines differed in the GEO quality of the pages they cited, and pillars related to Metadata and Freshness, Semantic HTML, and Structured Data showed the strongest associations with citation. Logistic models with domain clustered standard errors indicate that overall page quality is a strong predictor of citation, and simple operating points (for example, G at least 0.70 combined with at least 12 pillar hits) align with substantially higher citation rates in our data. We report per engine contrasts, vertical effects, threshold analysis, and diagnostics, then translate findings into a practical playbook for publishers. The study is observational and focuses on English language B2B SaaS pages; we discuss limitations, threats to validity, and reproducibility considerations. 

**Abstract (ZH)**: 基于AI的答案引擎不断通过生成响应和引用网络来源来中介领域知识的访问。我们引入了GE0-16审核框架，该框架将页面质量信号转换为分段支柱评分和范围从0到1的规范化GE0得分。使用70个产品意图提示，我们收集了来自三个引擎（Brave Summary、Google AI概览和Perplexity）的1,702次引证，并审核了1,100个唯一的URL。在我们的语料库中，引擎在所引页面的GE0质量方面存在差异，与元数据和新鲜度、语义HTML和结构化数据相关的支柱显示出最强的相关性。带有领域聚类标准误差的逻辑模型表明，整体页面质量是引证的强预测因素，简单的操作点（例如，GE0至少为0.70，配合至少12个支柱命中）与我们数据中更高的引证率相一致。我们报告了每个引擎对比、垂直效应、阈值分析和诊断，然后将研究发现转化为出版商的实际指南。该研究为观察性质，重点关注英文语言B2B SaaS页面；我们讨论了局限性、有效性的威胁以及可重复性考虑。 

---
# Situation Model of the Transport, Transport Emissions and Meteorological Conditions 

**Title (ZH)**: 运输情况模型、运输排放和气象条件 

**Authors**: V. Benes, M. Svitek, A. Michalikova, M. Melicherik  

**Link**: [PDF](https://arxiv.org/pdf/2509.10541)  

**Abstract**: Air pollution in cities and the possibilities of reducing this pollution represents one of the most important factors that today's society has to deal with. This paper focuses on a systemic approach to traffic emissions with their relation to meteorological conditions, analyzing the effect of weather on the quantity and dispersion of traffic emissions in a city. Using fuzzy inference systems (FIS) the model for prediction of changes in emissions depending on various conditions is developed. The proposed model is based on traffic, meteorology and emission data measured in Prague, Czech Republic. The main objective of the work is to provide insight into how urban planners and policymakers can plan and manage urban transport more effectively with environmental protection in mind. 

**Abstract (ZH)**: 城市中的空气污染及其降低的可能性是当今社会必须应对的重要因素。本文采用系统方法研究交通排放与气象条件的关系，分析气象条件对城市交通排放量及扩散的影响。利用模糊推理系统（FIS）构建了预测排放变化的模型。所提出的模型基于捷克共和国 Prague 测得的交通、气象和排放数据。本文的主要目标是为城市规划者和政策制定者如何在考虑环境保护的情况下更有效地规划和管理城市交通提供见解。 

---
# Dynamic Relational Priming Improves Transformer in Multivariate Time Series 

**Title (ZH)**: 动态关系 priming 改进 Transformer 在多变量时间序列中的表现 

**Authors**: Hunjae Lee, Corey Clark  

**Link**: [PDF](https://arxiv.org/pdf/2509.12196)  

**Abstract**: Standard attention mechanisms in transformers employ static token representations that remain unchanged across all pair-wise computations in each layer. This limits their representational alignment with the potentially diverse relational dynamics of each token-pair interaction. While they excel in domains with relatively homogeneous relationships, standard attention's static relational learning struggles to capture the diverse, heterogeneous inter-channel dependencies of multivariate time series (MTS) data--where different channel-pair interactions within a single system may be governed by entirely different physical laws or temporal dynamics. To better align the attention mechanism for such domain phenomena, we propose attention with dynamic relational priming (prime attention). Unlike standard attention where each token presents an identical representation across all of its pair-wise interactions, prime attention tailors each token dynamically (or per interaction) through learnable modulations to best capture the unique relational dynamics of each token pair, optimizing each pair-wise interaction for that specific relationship. This representational plasticity of prime attention enables effective extraction of relationship-specific information in MTS while maintaining the same asymptotic computational complexity as standard attention. Our results demonstrate that prime attention consistently outperforms standard attention across benchmarks, achieving up to 6.5\% improvement in forecasting accuracy. In addition, we find that prime attention achieves comparable or superior performance using up to 40\% less sequence length compared to standard attention, further demonstrating its superior relational modeling capabilities. 

**Abstract (ZH)**: 带有动态关系预热的标准注意力机制（Prime Attention） 

---
# Multi Anatomy X-Ray Foundation Model 

**Title (ZH)**: 多 Anatomy X-Ray 基础模型 

**Authors**: Nishank Singla, Krisztian Koos, Farzin Haddadpour, Amin Honarmandi Shandiz, Lovish Chum, Xiaojian Xu, Qing Jin, Erhan Bas  

**Link**: [PDF](https://arxiv.org/pdf/2509.12146)  

**Abstract**: X-ray imaging is a ubiquitous in radiology, yet most existing AI foundation models are limited to chest anatomy and fail to generalize across broader clinical tasks. In this work, we introduce XR-0, the multi-anatomy X-ray foundation model using self-supervised learning on a large, private dataset of 1.15 million images spanning diverse anatomical regions and evaluated across 12 datasets and 20 downstream tasks, including classification, retrieval, segmentation, localization, visual grounding, and report generation. XR-0 achieves state-of-the-art performance on most multi-anatomy tasks and remains competitive on chest-specific benchmarks. Our results demonstrate that anatomical diversity and supervision are critical for building robust, general-purpose medical vision models, paving the way for scalable and adaptable AI systems in radiology. 

**Abstract (ZH)**: X-ray成像在放射学中无处不在，但现有的大多数AI基础模型仅限于胸部解剖结构，无法在更广泛的临床任务中泛化。在此工作中，我们介绍了XR-0，这是一种使用大规模私有数据集（包含115万张图像，覆盖多种解剖区域）自我监督学习训练的多解剖X射线基础模型，并在12个数据集和20个下游任务（包括分类、检索、分割、定位、视觉接地和报告生成）上进行了评估。XR-0在大多数多解剖任务上取得了最先进的性能，并在胸部专用基准测试中保持竞争力。我们的结果显示，解剖学多样性与监督对于构建稳健的通用医疗视觉模型至关重要，为放射学中可扩展和适应性强的AI系统铺平了道路。 

---
# $K$-Level Policy Gradients for Multi-Agent Reinforcement Learning 

**Title (ZH)**: $K$级策略梯度在多智能体强化学习中的应用 

**Authors**: Aryaman Reddi, Gabriele Tiboni, Jan Peters, Carlo D'Eramo  

**Link**: [PDF](https://arxiv.org/pdf/2509.12117)  

**Abstract**: Actor-critic algorithms for deep multi-agent reinforcement learning (MARL) typically employ a policy update that responds to the current strategies of other agents. While being straightforward, this approach does not account for the updates of other agents at the same update step, resulting in miscoordination. In this paper, we introduce the $K$-Level Policy Gradient (KPG), a method that recursively updates each agent against the updated policies of other agents, speeding up the discovery of effective coordinated policies. We theoretically prove that KPG with finite iterates achieves monotonic convergence to a local Nash equilibrium under certain conditions. We provide principled implementations of KPG by applying it to the deep MARL algorithms MAPPO, MADDPG, and FACMAC. Empirically, we demonstrate superior performance over existing deep MARL algorithms in StarCraft II and multi-agent MuJoCo. 

**Abstract (ZH)**: 深度多agents强化学习中基于演员-评论家算法的$K$级策略梯度（KPG） 

---
# In-domain SSL pre-training and streaming ASR 

**Title (ZH)**: 领域内SSL预训练与流式ASR 

**Authors**: Jarod Duret, Salima Mdhaffar, Gaëlle Laperrière, Ryan Whetten, Audrey Galametz, Catherine Kobus, Marion-Cécile Martin, Jo Oleiwan, Yannick Estève  

**Link**: [PDF](https://arxiv.org/pdf/2509.12101)  

**Abstract**: In this study, we investigate the benefits of domain-specific self-supervised pre-training for both offline and streaming ASR in Air Traffic Control (ATC) environments. We train BEST-RQ models on 4.5k hours of unlabeled ATC data, then fine-tune on a smaller supervised ATC set. To enable real-time processing, we propose using chunked attention and dynamic convolutions, ensuring low-latency inference. We compare these in-domain SSL models against state-of-the-art, general-purpose speech encoders such as w2v-BERT 2.0 and HuBERT. Results show that domain-adapted pre-training substantially improves performance on standard ATC benchmarks, significantly reducing word error rates when compared to models trained on broad speech corpora. Furthermore, the proposed streaming approach further improves word error rate under tighter latency constraints, making it particularly suitable for safety-critical aviation applications. These findings highlight that specializing SSL representations for ATC data is a practical path toward more accurate and efficient ASR systems in real-world operational settings. 

**Abstract (ZH)**: 本研究探讨了领域特定自监督预训练对空中交通控制（ATC）环境中离线和实时ASR的好处。 

---
# A Time-Series Foundation Model by Universal Delay Embedding 

**Title (ZH)**: 一种由通用延迟嵌入构建的时间序列基础模型 

**Authors**: Zijian Wang, Peng Tao, Jifan Shi, Rui Bao, Rui Liu, Luonan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12080)  

**Abstract**: This study introduces Universal Delay Embedding (UDE), a pretrained foundation model designed to revolutionize time-series forecasting through principled integration of delay embedding representation and Koopman operator prediction. Leveraging Takens' embedding theorem, UDE as a dynamical representation of observed data constructs two-dimensional subspace patches from Hankel matrices, theoretically preserving dynamical and topological properties of underlying dynamical systems. Such patches are viewed as images, which can be efficiently processed by exploiting advanced deep learning technologies. Computationally, these patches further serve as tokens for learning a self-attention encoder, thus enabling accurate prediction of nonlinear time-series by a finite-dimensional Koopman operator in a linear manner in a latent space. Extensive evaluations across various benchmarks and real-world climate datasets demonstrate over 20% average reduction in mean squared error versus state-of-the-art foundation models, alongside superior generalization in fine-tuning scenarios. In particular, the learned dynamical representations and Koopman operator prediction forms from the patches exhibit exceptional interpretability, with consistent identification of topologically informative subspaces and robust encoding of domain-invariant dynamics, establishing UDE as a scalable, interpretable framework for universal time-series modeling and forecasting with broad scientific and industrial applicability. 

**Abstract (ZH)**: Universal Delay Embedding：一种通过原理性结合延迟嵌入表示和科伯曼算子预测来革新时间序列预测的预训练基础模型 

---
# Early Detection of Branched Broomrape (Phelipanche ramosa) Infestation in Tomato Crops Using Leaf Spectral Analysis and Machine Learning 

**Title (ZH)**: 使用叶片光谱分析和机器学习早期检测番茄田中的分枝雀麦寄生 

**Authors**: Mohammadreza Narimani, Alireza Pourreza, Ali Moghimi, Parastoo Farajpoor, Hamid Jafarbiglu, Mohsen B. Mesgaran  

**Link**: [PDF](https://arxiv.org/pdf/2509.12074)  

**Abstract**: Branched broomrape (Phelipanche ramosa) is a chlorophyll-deficient parasitic weed that threatens tomato production by extracting nutrients from the host. We investigate early detection using leaf-level spectral reflectance (400-2500 nm) and ensemble machine learning. In a field experiment in Woodland, California, we tracked 300 tomato plants across growth stages defined by growing degree days (GDD). Leaf reflectance was acquired with a portable spectrometer and preprocessed (band denoising, 1 nm interpolation, Savitzky-Golay smoothing, correlation-based band reduction). Clear class differences were observed near 1500 nm and 2000 nm water absorption features, consistent with reduced leaf water content in infected plants at early stages. An ensemble combining Random Forest, XGBoost, SVM with RBF kernel, and Naive Bayes achieved 89% accuracy at 585 GDD, with recalls of 0.86 (infected) and 0.93 (noninfected). Accuracy declined at later stages (e.g., 69% at 1568 GDD), likely due to senescence and weed interference. Despite the small number of infected plants and environmental confounders, results show that proximal sensing with ensemble learning enables timely detection of broomrape before canopy symptoms are visible, supporting targeted interventions and reduced yield losses. 

**Abstract (ZH)**: Phelipanche ramosa（分枝.INFO盗/photosynthesis）缺乏叶绿素的寄生杂草，通过从寄主植物吸取养分威胁番茄生产。我们利用叶水平光谱反射率（400-2500 nm）和集成机器学习研究早期检测方法。在加利福尼亚州伍德兰进行的田间实验中，我们跟踪了300株番茄植株，这些植株按生长度日（GDD）定义的生长阶段进行跟踪。通过便携式分光光度计获取叶片反射率，并进行了预处理（带噪声去除、1 nm插值、Savitzky-Golay平滑、基于相关性的波段减少）。观察到了1500 nm和2000 nm水吸收特征附近的明显类别差异，这与早期感染植物叶片中水分含量减少一致。结合随机森林、XGBoost、RBF核支持向量机（SVM）和朴素贝叶斯的集成模型在585 GDD时实现了89%的准确性，召回率为0.86（感染）和0.93（非感染）。在后期阶段，准确性下降（例如，在1568 GDD时为69%），这可能归因于叶片衰老和杂草竞争。尽管感染植株数量有限且存在环境混杂因素，结果表明，集成学习的邻近光谱感应能够使研究人员在冠层症状可见之前及时检测到INFO盗，从而支持目标干预并减少产量损失。 

---
# LEGO: Spatial Accelerator Generation and Optimization for Tensor Applications 

**Title (ZH)**: LEGO：张量应用中空间加速器的生成与优化 

**Authors**: Yujun Lin, Zhekai Zhang, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.12053)  

**Abstract**: Modern tensor applications, especially foundation models and generative AI applications require multiple input modalities (both vision and language), which increases the demand for flexible accelerator architecture. Existing frameworks suffer from the trade-off between design flexibility and productivity of RTL generation: either limited to very few hand-written templates or cannot automatically generate the RTL. To address this challenge, we propose the LEGO framework, which targets tensor applications and automatically generates spatial architecture design and outputs synthesizable RTL code without handwritten RTL design templates. Leveraging the affine-transformation-based architecture representation, LEGO front end finds interconnections between function units, synthesizes the memory system, and fuses different spatial dataflow designs based on data reuse analysis. LEGO back end then translates the hardware in a primitive-level graph to perform lower-level optimizations, and applies a set of linear-programming algorithms to optimally insert pipeline registers and reduce the overhead of unused logic when switching spatial dataflows. Our evaluation demonstrates that LEGO can achieve 3.2x speedup and 2.4x energy efficiency compared to previous work Gemmini, and can generate one architecture for diverse modern foundation models in generative AI applications. 

**Abstract (ZH)**: 现代张量应用，尤其是基础模型和生成AI应用，需要多种输入模态（包括视觉和语言），这增加了对灵活加速器架构的需求。现有框架在设计灵活性与RTL生成生产力之间存在权衡：要么仅限于很少的手写模板，要么无法自动生成RTL。为了解决这一挑战，我们提出了LEGO框架，该框架针对张量应用并自动生成空间架构设计和可综合的RTL代码，无需手写RTL设计模板。利用基于仿射变换的架构表示，LEGO前端找到功能单元之间的连接，合成内存系统，并根据数据重用分析融合不同的空间数据流设计。LEGO后端将硬件在原始级图中进行转换以执行低级别优化，并应用一组线性规划算法，在切换空间数据流时优化插入流水线寄存器并减少未使用逻辑的开销。我们的评估显示，LEGO相对于先前工作Gemmini可以实现3.2倍的加速和2.4倍的能量效率改进，并能够为生成AI应用中的各种现代基础模型生成一个架构。 

---
# Interaction-Driven Browsing: A Human-in-the-Loop Conceptual Framework Informed by Human Web Browsing for Browser-Using Agents 

**Title (ZH)**: 基于交互的浏览：由人类网页浏览行为启发的人机在环概念框架 

**Authors**: Hyeonggeun Yun, Jinkyu Jang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12049)  

**Abstract**: Although browser-using agents (BUAs) show promise for web tasks and automation, most BUAs terminate after executing a single instruction, failing to support users' complex, nonlinear browsing with ambiguous goals, iterative decision-making, and changing contexts. We present a human-in-the-loop (HITL) conceptual framework informed by theories of human web browsing behavior. The framework centers on an iterative loop in which the BUA proactively proposes next actions and the user steers the browsing process through feedback. It also distinguishes between exploration and exploitation actions, enabling users to control the breadth and depth of their browsing. Consequently, the framework aims to reduce users' physical and cognitive effort while preserving users' traditional browsing mental model and supporting users in achieving satisfactory outcomes. We illustrate how the framework operates with hypothetical use cases and discuss the shift from manual browsing to interaction-driven browsing. We contribute a theoretically informed conceptual framework for BUAs. 

**Abstract (ZH)**: 尽管浏览器使用的代理（BUAs）在网页任务和自动化方面展现出潜力，但大多数BUAs在执行单个指令后就会终止，无法支持用户复杂的、非线性的浏览行为，以及具有模糊目标的迭代决策和不断变化的情境。我们提出了一个基于人类网页浏览行为理论的人工智能辅助浏览框架（HITL）。该框架围绕BUA主动提案并由用户通过反馈引导浏览过程的迭代循环，区分探索性和利用性行为，使用户能够控制其浏览的广度和深度。因此，该框架旨在减少用户的体力和认知消耗，同时保持用户传统的浏览心智模型，并支持用户实现满意的结果。我们通过假设的应用场景阐述了该框架的操作机制，并讨论了从手动浏览到互动驱动浏览的转变。我们贡献了一个基于理论的人工智能代理框架。 

---
# Imitation Learning as Return Distribution Matching 

**Title (ZH)**: imitation learning作为回报分布匹配 

**Authors**: Filippo Lazzati, Alberto Maria Metelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.12026)  

**Abstract**: We study the problem of training a risk-sensitive reinforcement learning (RL) agent through imitation learning (IL). Unlike standard IL, our goal is not only to train an agent that matches the expert's expected return (i.e., its average performance) but also its risk attitude (i.e., other features of the return distribution, such as variance). We propose a general formulation of the risk-sensitive IL problem in which the objective is to match the expert's return distribution in Wasserstein distance. We focus on the tabular setting and assume the expert's reward is known. After demonstrating the limited expressivity of Markovian policies for this task, we introduce an efficient and sufficiently expressive subclass of non-Markovian policies tailored to it. Building on this subclass, we develop two provably efficient algorithms, RS-BC and RS-KT, for solving the problem when the transition model is unknown and known, respectively. We show that RS-KT achieves substantially lower sample complexity than RS-BC by exploiting dynamics information. We further demonstrate the sample efficiency of return distribution matching in the setting where the expert's reward is unknown by designing an oracle-based variant of RS-KT. Finally, we complement our theoretical analysis of RS-KT and RS-BC with numerical simulations, highlighting both their sample efficiency and the advantages of non-Markovian policies over standard sample-efficient IL algorithms. 

**Abstract (ZH)**: 我们研究通过模仿学习训练风险敏感的强化学习代理的问题。与标准的模仿学习不同，我们的目标不仅是要训练一个能够匹配专家预期回报（即其平均性能）的代理，还要匹配专家的风险态度（即回报分布的其他特征，如方差）。我们提出了一种风险敏感的模仿学习问题的一般公式，其中目标是通过Wasserstein距离匹配专家的回报分布。我们专注于表型设置，并假设专家的奖励是已知的。在展示了马尔可夫策略在此任务上的表达能力有限之后，我们引入了一类高效且足够表达的非马尔可夫策略子类，专门用于此任务。基于这一子类，我们开发了两种各自在转移模型未知和已知情况下解决问题的可证明高效的算法，RS-BC和RS-KT。我们通过利用动力学信息证明RS-KT的样本复杂度显著低于RS-BC。我们还通过设计基于 oracle 的RS-KT变种在专家奖励未知的情况下展示了回报分布匹配的样本效率。最后，通过数值模拟补充了对RS-KT和RS-BC的理论分析，强调了非马尔可夫策略相对于标准样本高效模仿学习算法的优势。 

---
# Generalizing Behavior via Inverse Reinforcement Learning with Closed-Form Reward Centroids 

**Title (ZH)**: 基于闭形式奖励质心的逆强化学习行为泛化 

**Authors**: Filippo Lazzati, Alberto Maria Metelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.12010)  

**Abstract**: We study the problem of generalizing an expert agent's behavior, provided through demonstrations, to new environments and/or additional constraints. Inverse Reinforcement Learning (IRL) offers a promising solution by seeking to recover the expert's underlying reward function, which, if used for planning in the new settings, would reproduce the desired behavior. However, IRL is inherently ill-posed: multiple reward functions, forming the so-called feasible set, can explain the same observed behavior. Since these rewards may induce different policies in the new setting, in the absence of additional information, a decision criterion is needed to select which policy to deploy. In this paper, we propose a novel, principled criterion that selects the "average" policy among those induced by the rewards in a certain bounded subset of the feasible set. Remarkably, we show that this policy can be obtained by planning with the reward centroid of that subset, for which we derive a closed-form expression. We then present a provably efficient algorithm for estimating this centroid using an offline dataset of expert demonstrations only. Finally, we conduct numerical simulations that illustrate the relationship between the expert's behavior and the behavior produced by our method. 

**Abstract (ZH)**: 我们研究如何通过演示将专家代理的行为泛化到新环境或附加约束中。逆强化学习（IRL）提供了一种有前景的解决方案，通过寻求恢复专家的潜在奖励函数，从而在新环境中使用这些奖励进行规划，重现期望的行为。然而，IRL 本质上是病态的：可以解释相同观察行为的各种奖励函数构成了所谓的可行集。由于这些奖励在新环境中可能会诱导出不同的策略，在缺乏额外信息的情况下，需要一个决策标准来选择要部署的策略。在本文中，我们提出了一种新颖且有原则的方法，选择由某个可行集有界子集中的奖励诱导出的“平均”策略。令人惊讶的是，我们展示了可以通过使用该子集的奖励质心进行规划，来获得此策略，并推导出其闭式表达式。我们随后提出了一种仅使用专家演示的离线数据集来证明高效算法来估计此质心。最后，我们进行了数值模拟，以说明专家行为与我们方法产生的行为之间的关系。 

---
# Poison to Detect: Detection of Targeted Overfitting in Federated Learning 

**Title (ZH)**: 毒药检测： Federated Learning 中目标过拟合的检测 

**Authors**: Soumia Zohra El Mestari, Maciej Krzysztof Zuziak, Gabriele Lenzini  

**Link**: [PDF](https://arxiv.org/pdf/2509.11974)  

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralised clients while keeping local data private, making it a widely adopted privacy-enhancing technology (PET). Despite its privacy benefits, FL remains vulnerable to privacy attacks, including those targeting specific clients. In this paper, we study an underexplored threat where a dishonest orchestrator intentionally manipulates the aggregation process to induce targeted overfitting in the local models of specific clients. Whereas many studies in this area predominantly focus on reducing the amount of information leakage during training, we focus on enabling an early client-side detection of targeted overfitting, thereby allowing clients to disengage before significant harm occurs. In line with this, we propose three detection techniques - (a) label flipping, (b) backdoor trigger injection, and (c) model fingerprinting - that enable clients to verify the integrity of the global aggregation. We evaluated our methods on multiple datasets under different attack scenarios. Our results show that the three methods reliably detect targeted overfitting induced by the orchestrator, but they differ in terms of computational complexity, detection latency, and false-positive rates. 

**Abstract (ZH)**: 联邦学习（FL）能够在保持本地数据隐私的同时，使去中心化的客户端进行协作模型训练，使其成为一种广泛应用的隐私增强技术（PET）。尽管具有隐私优势，FL仍然容易受到包括针对特定客户端的攻击在内的隐私攻击。在本文中，我们研究了一种未充分探索的威胁，即不诚实的协调者故意操纵聚合过程，以诱导特定客户端局部模型的目标过拟合。不同于该领域许多研究侧重于减少训练过程中的信息泄露，我们专注于在早期检测特定过拟合，从而允许客户端在造成重大损害之前退出。为此，我们提出了三种检测技术——(a)标签翻转，(b)后门触发注入，以及(c)模型指纹识别，以使客户端能够验证全局聚合的完整性。我们在不同攻击场景下使用多个数据集评估了这些方法。结果显示，这三种方法能够可靠地检测由协调者诱导的目标过拟合，但它们在计算复杂性、检测延迟和误报率方面存在差异。 

---
# A GPU-Accelerated RAG-Based Telegram Assistant for Supporting Parallel Processing Students 

**Title (ZH)**: 基于RAG的GPU加速电信助手，用于支持并行处理学生 

**Authors**: Guy Tel-Zur  

**Link**: [PDF](https://arxiv.org/pdf/2509.11947)  

**Abstract**: This project addresses a critical pedagogical need: offering students continuous, on-demand academic assistance beyond conventional reception hours. I present a domain-specific Retrieval-Augmented Generation (RAG) system powered by a quantized Mistral-7B Instruct model and deployed as a Telegram bot. The assistant enhances learning by delivering real-time, personalized responses aligned with the "Introduction to Parallel Processing" course materials. GPU acceleration significantly improves inference latency, enabling practical deployment on consumer hardware. This approach demonstrates how consumer GPUs can enable affordable, private, and effective AI tutoring for HPC education. 

**Abstract (ZH)**: 本项目满足了一个关键的教学需求：为学生提供超出常规接待时间的连续、按需学术支持。我介绍了一种基于量化Mistral-7B Instruct模型的领域特定检索增强生成（RAG）系统，并将其部署为Telegram机器人。助教通过提供与“并行处理导论”课程材料相契合的实时个性化响应来增强学习。GPU加速显著改善了推理延迟，使得该系统能够在消费级硬件上实用部署。此方法展示了如何通过消费级GPU实现高效、私有的HPC教育中负担得起的AI辅导。 

---
# Probabilistic Robustness Analysis in High Dimensional Space: Application to Semantic Segmentation Network 

**Title (ZH)**: 高维空间中的概率鲁棒性分析：以语义分割网络为例 

**Authors**: Navid Hashemi, Samuel Sasaki, Diego Manzanas Lopez, Ipek Oguz, Meiyi Ma, Taylor T. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2509.11838)  

**Abstract**: Semantic segmentation networks (SSNs) play a critical role in domains such as medical imaging, autonomous driving, and environmental monitoring, where safety hinges on reliable model behavior under uncertainty. Yet, existing probabilistic verification approaches struggle to scale with the complexity and dimensionality of modern segmentation tasks, often yielding guarantees that are too conservative to be practical. We introduce a probabilistic verification framework that is both architecture-agnostic and scalable to high-dimensional outputs. Our approach combines sampling-based reachability analysis with conformal inference (CI) to deliver provable guarantees while avoiding the excessive conservatism of prior methods. To counteract CI's limitations in high-dimensional settings, we propose novel strategies that reduce conservatism without compromising rigor. Empirical evaluation on large-scale segmentation models across CamVid, OCTA-500, Lung Segmentation, and Cityscapes demonstrates that our method provides reliable safety guarantees while substantially tightening bounds compared to SOTA. We also provide a toolbox implementing this technique, available on Github. 

**Abstract (ZH)**: 基于概率验证的高维语义分割网络安全性保障框架 

---
# Data-Driven Analysis of Text-Conditioned AI-Generated Music: A Case Study with Suno and Udio 

**Title (ZH)**: 基于数据驱动的文本条件化AI生成音乐分析：一个使用Suno和Udio的案例研究 

**Authors**: Luca Casini, Laura Cros Vila, David Dalmazzo, Anna-Kaisa Kaila, Bob L.T. Sturm  

**Link**: [PDF](https://arxiv.org/pdf/2509.11824)  

**Abstract**: Online AI platforms for creating music from text prompts (AI music), such as Suno and Udio, are now being used by hundreds of thousands of users. Some AI music is appearing in advertising, and even charting, in multiple countries. How are these platforms being used? What subjects are inspiring their users? This article answers these questions for Suno and Udio using a large collection of songs generated by users of these platforms from May to October 2024. Using a combination of state-of-the-art text embedding models, dimensionality reduction and clustering methods, we analyze the prompts, tags and lyrics, and automatically annotate and display the processed data in interactive plots. Our results reveal prominent themes in lyrics, language preference, prompting strategies, as well as peculiar attempts at steering models through the use of metatags. To promote the musicological study of the developing cultural practice of AI-generated music we share our code and resources. 

**Abstract (ZH)**: 在线AI平台基于文本提示创作音乐（AI音乐），如Suno和Udio，现已被数百万人使用。这些AI音乐开始出现在广告中，并在多个国家的音乐排行榜上出现。这些平台是如何被使用的？什么主题激发了用户？本文通过分析2024年5月至10月这些平台上用户生成的大量歌曲，回答了这些问题。我们结合最新的文本嵌入模型、降维和聚类方法，对提示、标签和歌词进行分析，并自动标注和展示处理后的数据，在交互式图表中呈现。我们的结果显示了歌词中的主要主题、语言偏好、提示策略，以及通过元标签引导模型的独特尝试。为了促进对AI生成音乐这一发展的文化实践的音乐学研究，我们分享了我们的代码和资源。 

---
# CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model 

**Title (ZH)**: CoachMe：基于参考的运动元素教练指令生成模型 

**Authors**: Wei-Hsin Yeh, Yu-An Su, Chih-Ning Chen, Yi-Hsueh Lin, Calvin Ku, Wen-Hsin Chiu, Min-Chun Hu, Lun-Wei Ku  

**Link**: [PDF](https://arxiv.org/pdf/2509.11698)  

**Abstract**: Motion instruction is a crucial task that helps athletes refine their technique by analyzing movements and providing corrective guidance. Although recent advances in multimodal models have improved motion understanding, generating precise and sport-specific instruction remains challenging due to the highly domain-specific nature of sports and the need for informative guidance. We propose CoachMe, a reference-based model that analyzes the differences between a learner's motion and a reference under temporal and physical aspects. This approach enables both domain-knowledge learning and the acquisition of a coach-like thinking process that identifies movement errors effectively and provides feedback to explain how to improve. In this paper, we illustrate how CoachMe adapts well to specific sports such as skating and boxing by learning from general movements and then leveraging limited data. Experiments show that CoachMe provides high-quality instructions instead of directions merely in the tone of a coach but without critical information. CoachMe outperforms GPT-4o by 31.6% in G-Eval on figure skating and by 58.3% on boxing. Analysis further confirms that it elaborates on errors and their corresponding improvement methods in the generated instructions. You can find CoachMe here: this https URL 

**Abstract (ZH)**: 运动指令是通过分析动作并提供纠正指导来帮助运动员精炼技术的关键任务。尽管近期多模态模型在运动理解方面取得了进步，但由于运动领域的高度专业性和需要有信息量的指导，生成精确且运动专项的指示仍然具有挑战性。我们提出了CoachMe，一种基于参考的模型，从时间维度和物理维度分析学习者运动与参考运动之间的差异。该方法既支持专业知识的学习，又能够获得类似教练的思考过程，有效识别运动错误并提供反馈解释如何改进。在本文中，我们展示了CoachMe如何通过从一般动作中学习，再利用有限数据适应特定运动如滑冰和拳击。实验结果表明，CoachMe在花样滑冰和拳击上的G-Eval评分分别比GPT-4o高31.6%和58.3%。进一步的分析证实，生成的指示解释了错误及相应的改进方法。你可以在这里找到CoachMe：this https URL。 

---
# DTGen: Generative Diffusion-Based Few-Shot Data Augmentation for Fine-Grained Dirty Tableware Recognition 

**Title (ZH)**: DTGen: 基于生成扩散的少量样本数据增强方法用于细粒度污损餐具识别 

**Authors**: Lifei Hao, Yue Cheng, Baoqi Huang, Bing Jia, Xuandong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.11661)  

**Abstract**: Intelligent tableware cleaning is a critical application in food safety and smart homes, but existing methods are limited by coarse-grained classification and scarcity of few-shot data, making it difficult to meet industrialization requirements. We propose DTGen, a few-shot data augmentation scheme based on generative diffusion models, specifically designed for fine-grained dirty tableware recognition. DTGen achieves efficient domain specialization through LoRA, generates diverse dirty images via structured prompts, and ensures data quality through CLIP-based cross-modal filtering. Under extremely limited real few-shot conditions, DTGen can synthesize virtually unlimited high-quality samples, significantly improving classifier performance and supporting fine-grained dirty tableware recognition. We further elaborate on lightweight deployment strategies, promising to transfer DTGen's benefits to embedded dishwashers and integrate with cleaning programs to intelligently regulate energy consumption and detergent usage. Research results demonstrate that DTGen not only validates the value of generative AI in few-shot industrial vision but also provides a feasible deployment path for automated tableware cleaning and food safety monitoring. 

**Abstract (ZH)**: 基于生成扩散模型的少量样本数据增强方案DTGen及其在细粒度脏餐具识别中的应用 

---
# Task-Agnostic Learnable Weighted-Knowledge Base Scheme for Robust Semantic Communications 

**Title (ZH)**: 面向任务的可学习加权知识库方案以实现稳健的语义通信 

**Authors**: Shiyao Jiang, Jian Jiao, Xingjian Zhang, Ye Wang, Dusit Niyato, Qinyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11636)  

**Abstract**: With the emergence of diverse and massive data in the upcoming sixth-generation (6G) networks, the task-agnostic semantic communication system is regarded to provide robust intelligent services. In this paper, we propose a task-agnostic learnable weighted-knowledge base semantic communication (TALSC) framework for robust image transmission to address the real-world heterogeneous data bias in KB, including label flipping noise and class imbalance. The TALSC framework incorporates a sample confidence module (SCM) as meta-learner and the semantic coding networks as learners. The learners are updated based on the empirical knowledge provided by the learnable weighted-KB (LW-KB). Meanwhile, the meta-learner evaluates the significance of samples according to the task loss feedback, and adjusts the update strategy of learners to enhance the robustness in semantic recovery for unknown tasks. To strike a balance between SCM parameters and precision of significance evaluation, we design an SCM-grid extension (SCM-GE) approach by embedding the Kolmogorov-Arnold networks (KAN) within SCM, which leverages the concept of spline refinement in KAN and enables scalable SCM with customizable granularity without retraining. Simulations demonstrate that the TALSC framework effectively mitigates the effects of flipping noise and class imbalance in task-agnostic image semantic communication, achieving at least 12% higher semantic recovery accuracy (SRA) and multi-scale structural similarity (MS-SSIM) compared to state-of-the-art methods. 

**Abstract (ZH)**: 面向未知任务的可学习加权知识库语义通信框架：应对知识库中的标签翻转噪声和类别不平衡问题 

---
# SpeCa: Accelerating Diffusion Transformers with Speculative Feature Caching 

**Title (ZH)**: SpeCa: 采用推测特征缓存加速扩散变换器 

**Authors**: Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Fei Ren, Shaobo Wang, Kaixin Li, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11628)  

**Abstract**: Diffusion models have revolutionized high-fidelity image and video synthesis, yet their computational demands remain prohibitive for real-time applications. These models face two fundamental challenges: strict temporal dependencies preventing parallelization, and computationally intensive forward passes required at each denoising step. Drawing inspiration from speculative decoding in large language models, we present SpeCa, a novel 'Forecast-then-verify' acceleration framework that effectively addresses both limitations. SpeCa's core innovation lies in introducing Speculative Sampling to diffusion models, predicting intermediate features for subsequent timesteps based on fully computed reference timesteps. Our approach implements a parameter-free verification mechanism that efficiently evaluates prediction reliability, enabling real-time decisions to accept or reject each prediction while incurring negligible computational overhead. Furthermore, SpeCa introduces sample-adaptive computation allocation that dynamically modulates resources based on generation complexity, allocating reduced computation for simpler samples while preserving intensive processing for complex instances. Experiments demonstrate 6.34x acceleration on FLUX with minimal quality degradation (5.5% drop), 7.3x speedup on DiT while preserving generation fidelity, and 79.84% VBench score at 6.1x acceleration for HunyuanVideo. The verification mechanism incurs minimal overhead (1.67%-3.5% of full inference costs), establishing a new paradigm for efficient diffusion model inference while maintaining generation quality even at aggressive acceleration ratios. Our codes have been released in Github: \textbf{this https URL} 

**Abstract (ZH)**: 扩散模型已 revolutionized 高保真图像和视频合成，但其计算需求仍然阻碍了实时应用。这些模型面临两大根本挑战：严格的时序依赖性阻碍了并行化，以及在每个去噪步骤中所需的计算密集型前向传递。受到大规模语言模型中 speculative decoding 启发，我们提出了 SpeCa，一种新颖的“预测-验证”加速框架，有效解决了上述限制。SpeCa 的核心创新在于引入 speculative sampling 到扩散模型中，基于完全计算的参考时间步预测后续时间步的中间特征。我们的方法实施了一种无参数验证机制，高效评估预测的可靠性，在几乎不增加计算开销的情况下，在实时决策时接受或拒绝每个预测。此外，SpeCa 引入了样本自适应计算分配，根据生成复杂性动态调整资源，为简单的样本减少计算量，同时为复杂的实例保留密集处理。实验结果显示，在 FLUX 上加速 6.34 倍，质量下降 5.5%，在 DiT 上加速 7.3 倍同时保持生成保真度，在 HunyuanVideo 上以 6.1 倍加速实现 79.84% 的 VBench 分数。验证机制带来的开销最小（1.67%-3.5% 的全推理成本），建立了一种新的高效扩散模型推理范式，即使在极端加速比下也维持生成质量。我们的代码已发布在 Github：\textbf{this https URL}。 

---
# Inducing Uncertainty for Test-Time Privacy 

**Title (ZH)**: 测试时隐私的不确定性诱导 

**Authors**: Muhammad H. Ashiq, Peter Triantafillou, Hung Yun Tseng, Grigoris G. Chrysos  

**Link**: [PDF](https://arxiv.org/pdf/2509.11625)  

**Abstract**: Unlearning is the predominant method for removing the influence of data in machine learning models. However, even after unlearning, models often continue to produce the same predictions on the unlearned data with high confidence. This persistent behavior can be exploited by adversaries using confident model predictions on incorrect or obsolete data to harm users. We call this threat model, which unlearning fails to protect against, *test-time privacy*. In particular, an adversary with full model access can bypass any naive defenses which ensure test-time privacy. To address this threat, we introduce an algorithm which perturbs model weights to induce maximal uncertainty on protected instances while preserving accuracy on the rest of the instances. Our core algorithm is based on finetuning with a Pareto optimal objective that explicitly balances test-time privacy against utility. We also provide a certifiable approximation algorithm which achieves $(\varepsilon, \delta)$ guarantees without convexity assumptions. We then prove a tight, non-vacuous bound that characterizes the privacy-utility tradeoff that our algorithms incur. Empirically, our method obtains $>3\times$ stronger uncertainty than pretraining with $<0.2\%$ drops in accuracy on various image recognition benchmarks. Altogether, this framework provides a tool to guarantee additional protection to end users. 

**Abstract (ZH)**: 测试时隐私：卸学无法防护的隐私威胁及其实现算法 

---
# Dynamic Adaptive Parsing of Temporal and Cross-Variable Patterns for Network State Classification 

**Title (ZH)**: 基于时序和跨变量模式的网络状态分类的动态自适应解析方法 

**Authors**: Yuan Gao, Xuelong Wang, Zhenguo Dong, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11601)  

**Abstract**: Effective network state classification is a primary task for ensuring network security and optimizing performance. Existing deep learning models have shown considerable progress in this area. Some methods excel at analyzing the complex temporal periodicities found in traffic data, while graph-based approaches are adept at modeling the dynamic dependencies between different variables. However, a key trade-off remains, as these methods struggle to capture both characteristics simultaneously. Models focused on temporal patterns often overlook crucial variable dependencies, whereas those centered on dependencies may fail to capture fine-grained temporal details. To address this trade-off, we introduce DAPNet, a framework based on a Mixture-of-Experts architecture. DAPNet integrates three specialized networks for periodic analysis, dynamic cross-variable correlation modeling, and hybrid temporal feature extraction. A learnable gating network dynamically assigns weights to experts based on the input sample and computes a weighted fusion of their outputs. Furthermore, a hybrid regularization loss function ensures stable training and addresses the common issue of class imbalance. Extensive experiments on two large-scale network intrusion detection datasets (CICIDS2017/2018) validate DAPNet's higher accuracy for its target application. The generalizability of the architectural design is evaluated across ten public UEA benchmark datasets, positioning DAPNet as a specialized framework for network state classification. 

**Abstract (ZH)**: 有效的网络状态分类是确保网络安全和优化性能的主要任务。现有的深度学习模型在这领域已显示出显著的进步。一些方法擅长分析交通数据中复杂的时序周期性，而图基方法擅长建模不同变量之间的动态依赖关系。然而，这些方法仍然面临一个关键权衡，即它们难以同时捕捉这两种特性。专注于时序模式的模型往往会忽略关键的变量依赖性，而以依赖为中心的模型可能无法捕捉到详细的时序细节。为了解决这一权衡，我们提出了基于Mixture-of-Experts架构的DAPNet框架。DAPNet整合了三种专门网络，分别用于周期性分析、动态跨变量相关性建模以及混合时序特征提取。一个可学习的门控网络根据输入样本动态分配专家权重，并计算它们输出的加权融合。此外，一种混合正则化损失函数确保了稳定的训练，并解决了类别不平衡的常见问题。在两个大规模网络入侵检测数据集（CICIDS2017/2018）上的广泛实验验证了DAPNet在目标应用中的更高准确性。架构设计的泛化能力在十个公开的UEA基准数据集上进行评估，使DAPNet成为网络状态分类的专业化框架。 

---
# Dstack: A Zero Trust Framework for Confidential Containers 

**Title (ZH)**: Dstack: 一种针对机密容器的零信任框架 

**Authors**: Shunfan Zhou, Kevin Wang, Hang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.11555)  

**Abstract**: Web3 applications require execution platforms that maintain confidentiality and integrity without relying on centralized trust authorities. While Trusted Execution Environments (TEEs) offer promising capabilities for confidential computing, current implementations face significant limitations when applied to Web3 contexts, particularly in security reliability, censorship resistance, and vendor independence.
This paper presents dstack, a comprehensive framework that transforms raw TEE technology into a true Zero Trust platform. We introduce three key innovations: (1) Portable Confidential Containers that enable seamless workload migration across heterogeneous TEE environments while maintaining security guarantees, (2) Decentralized Code Management that leverages smart contracts for transparent governance of TEE applications, and (3) Verifiable Domain Management that ensures secure and verifiable application identity without centralized authorities.
These innovations are implemented through three core components: dstack-OS, dstack-KMS, and dstack-Gateway. Together, they demonstrate how to achieve both the performance advantages of VM-level TEE solutions and the trustless guarantees required by Web3 applications. Our evaluation shows that dstack provides comprehensive security guarantees while maintaining practical usability for real-world applications. 

**Abstract (ZH)**: Web3应用程序需要执行平台，该平台能够维护保密性和完整性，而不依赖于中心化的信任权威机构。虽然受信任执行环境（TEEs）为保密计算提供了潜在的能力，但当前的应用在Web3环境中面临重大限制，特别是在安全可靠性、审查阻力和供应商独立性方面。

本文提出了一种名为dstack的综合框架，将原始的TEE技术转化为真正的零信任平台。我们介绍了三项关键技术创新：（1）可移植的保密容器，可在异构TEE环境中无缝迁移工作负载的同时保持安全保证；（2）去中心化的代码管理，利用智能合约实现TEE应用的透明治理；（3）可验证的域管理，确保应用身份的安全和验证，而无需中央权威机构。

这些创新是通过三个核心组件实现的：dstack-OS、dstack-KMS和dstack-Gateway。它们共同展示了如何同时获得虚拟机级别TEE解决方案的性能优势以及Web3应用程序所需的无信任保证。评估结果显示，dstack提供了全面的安全保证，并且在实际应用中具有实用的可用性。 

---
# Know What You Don't Know: Selective Prediction for Early Exit DNNs 

**Title (ZH)**: 知其所不知：选择性预测用于早期退出DNN 

**Authors**: Divya Jyoti Bajpai, Manjesh Kumar Hanawal  

**Link**: [PDF](https://arxiv.org/pdf/2509.11520)  

**Abstract**: Inference latency and trustworthiness of Deep Neural Networks (DNNs) are the bottlenecks in deploying them in critical applications like sensitive tasks. Early Exit (EE) DNNs overcome the latency issues by allowing samples to exit from intermediary layers if they attain `high' confidence scores on the predicted class. However, the DNNs are known to exhibit overconfidence, which can lead to many samples exiting early and render EE strategies untrustworthy. We use Selective Prediction (SP) to overcome this issue by checking the `hardness' of the samples rather than just relying on the confidence score alone. We propose SPEED, a novel approach that uses Deferral Classifiers (DCs) at each layer to check the hardness of samples before performing EEs. Specifically, the DCs identify if a sample is hard to predict at an intermediary layer, leading to hallucination, and defer it to an expert. Early detection of hard samples for inference prevents the wastage of computational resources and improves trust by deferring the hard samples to the expert. We demonstrate that EE aided with SP improves both accuracy and latency. Our method minimizes the risk of wrong prediction by $50\%$ with a speedup of $2.05\times$ as compared to the final layer. The anonymized source code is available at this https URL 

**Abstract (ZH)**: 深度神经网络的推理延迟和可信度是将其应用于敏感任务等关键应用中的瓶颈。选择性预测辅助的早期退出深度神经网络通过检查样本的“难度”而非仅依赖预测置信度来克服这一问题，提出了一种名为SPEED的新方法，该方法在每层使用退避分类器（DCs）在进行早期退出之前检查样本的难度。我们展示了带有选择性预测的早期退出方法可以同时提高准确性和降低延迟。相比最终层，该方法将错误预测的风险降低了50%，并实现了2.05倍的速度提升。源代码已匿名处理，可在以下链接获取。 

---
# Unsupervised Candidate Ranking for Lexical Substitution via Holistic Sentence Semantics 

**Title (ZH)**: 基于整体句子语义的无监督候选词排序方法 

**Authors**: Zhongyang Hu, Naijie Gu, Xiangzhi Tao, Tianhui Gu, Yibing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.11513)  

**Abstract**: A key subtask in lexical substitution is ranking the given candidate words. A common approach is to replace the target word with a candidate in the original sentence and feed the modified sentence into a model to capture semantic differences before and after substitution. However, effectively modeling the bidirectional influence of candidate substitution on both the target word and its context remains challenging. Existing methods often focus solely on semantic changes at the target position or rely on parameter tuning over multiple evaluation metrics, making it difficult to accurately characterize semantic variation. To address this, we investigate two approaches: one based on attention weights and another leveraging the more interpretable integrated gradients method, both designed to measure the influence of context tokens on the target token and to rank candidates by incorporating semantic similarity between the original and substituted sentences. Experiments on the LS07 and SWORDS datasets demonstrate that both approaches improve ranking performance. 

**Abstract (ZH)**: 词替换中的一个关键子任务是对给定的候选项单词进行排名。一种常见的方法是将目标单词替换为候选词，并将修改后的句子输入模型以捕获替换前后语义差异。然而，有效地建模候选词替换对目标单词及其上下文的双向影响仍然是一个挑战。现有方法通常仅专注于目标位置的语义变化，或者依赖于在多个评估指标上进行参数调整，这使得准确刻画语义变异变得困难。为了解决这一问题，我们研究了两种方法：一种基于注意权重的方法，另一种利用更具可解释性的整合梯度方法，这两种方法都旨在测量上下文词对目标词的影响，并通过结合原句和替换句的语义相似性对候选项进行排名。实验表明，这两种方法都提高了排名性能。 

---
# Machine Learning-Driven Predictive Resource Management in Complex Science Workflows 

**Title (ZH)**: 基于机器学习的复杂科学工作流预测资源管理 

**Authors**: Tasnuva Chowdhury, Tadashi Maeno, Fatih Furkan Akman, Joseph Boudreau, Sankha Dutta, Shengyu Feng, Adolfy Hoisie, Kuan-Chieh Hsu, Raees Khan, Jaehyung Kim, Ozgur O. Kilic, Scott Klasky, Alexei Klimentov, Tatiana Korchuganova, Verena Ingrid Martinez Outschoorn, Paul Nilsson, David K. Park, Norbert Podhorszki, Yihui Ren, John Rembrandt Steele, Frédéric Suter, Sairam Sri Vatsavai, Torre Wenaus, Wei Yang, Yiming Yang, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2509.11512)  

**Abstract**: The collaborative efforts of large communities in science experiments, often comprising thousands of global members, reflect a monumental commitment to exploration and discovery. Recently, advanced and complex data processing has gained increasing importance in science experiments. Data processing workflows typically consist of multiple intricate steps, and the precise specification of resource requirements is crucial for each step to allocate optimal resources for effective processing. Estimating resource requirements in advance is challenging due to a wide range of analysis scenarios, varying skill levels among community members, and the continuously increasing spectrum of computing options. One practical approach to mitigate these challenges involves initially processing a subset of each step to measure precise resource utilization from actual processing profiles before completing the entire step. While this two-staged approach enables processing on optimal resources for most of the workflow, it has drawbacks such as initial inaccuracies leading to potential failures and suboptimal resource usage, along with overhead from waiting for initial processing completion, which is critical for fast-turnaround analyses. In this context, our study introduces a novel pipeline of machine learning models within a comprehensive workflow management system, the Production and Distributed Analysis (PanDA) system. These models employ advanced machine learning techniques to predict key resource requirements, overcoming challenges posed by limited upfront knowledge of characteristics at each step. Accurate forecasts of resource requirements enable informed and proactive decision-making in workflow management, enhancing the efficiency of handling diverse, complex workflows across heterogeneous resources. 

**Abstract (ZH)**: 大规模科研社区在科学实验中的协同努力，通常由成千上万的全球成员组成，反映了对探索和发现的巨大承诺。近年来，先进的复杂数据处理在科学实验中变得越来越重要。数据处理工作流通常包含多个复杂的步骤，精确规定资源需求对于每一步有效地分配最佳资源至关重要。由于分析场景的广泛差异、社区成员技能水平的差异以及计算选项范围的不断扩展，提前估计资源需求具有挑战性。一种实用的方法是初始处理每个步骤的一个子集，以实际处理性能数据来准确测量资源利用率，然后再完成整个步骤。虽然这种方法允许大多数工作流在最佳资源上进行处理，但它也存在一些缺点，如初始准确性不足可能导致潜在失败和资源使用不充分，以及等待初始处理完成的额外开销，这对快速周转分析至关重要。在这一背景下，我们的研究介绍了一个全面的工作流管理系统——Production and Distributed Analysis (PanDA)系统中的新型机器学习模型管道。这些模型采用先进的机器学习技术来预测关键的资源需求，克服了对每一步特性的有限前置知识所带来的挑战。准确的资源需求预测有助于在工作流管理中进行知情和主动决策，从而提高处理异构资源上多种复杂工作流的效率。 

---
# CareerPooler: AI-Powered Metaphorical Pool Simulation Improves Experience and Outcomes in Career Exploration 

**Title (ZH)**: CareerPooler：AI赋能的隐喻性池化模拟提高职业探索的经验和成果 

**Authors**: Ziyi Wang, Ziwen Zeng, Yuan Li, Zijian Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.11461)  

**Abstract**: Career exploration is uncertain, requiring decisions with limited information and unpredictable outcomes. While generative AI offers new opportunities for career guidance, most systems rely on linear chat interfaces that produce overly comprehensive and idealized suggestions, overlooking the non-linear and effortful nature of real-world trajectories. We present CareerPooler, a generative AI-powered system that employs a pool-table metaphor to simulate career development as a spatial and narrative interaction. Users strike balls representing milestones, skills, and random events, where hints, collisions, and rebounds embody decision-making under uncertainty. In a within-subjects study with 24 participants, CareerPooler significantly improved engagement, information gain, satisfaction, and career clarity compared to a chatbot baseline. Qualitative findings show that spatial-narrative interaction fosters experience-based learning, resilience through setbacks, and reduced psychological burden. Our findings contribute to the design of AI-assisted career exploration systems and more broadly suggest that visually grounded analogical interactions can make generative systems engaging and satisfying. 

**Abstract (ZH)**: 职业探索充满不确定性，需要在有限信息和不可预测的结果下做决策。虽然生成式AI为职业指导提供了新机遇，但大多数系统依赖线性聊天界面，产生过于全面和理想化的建议，忽视了现实世界轨迹的非线性和努力性质。我们提出CareerPooler，一个采用台球类比来模拟职业发展为主的空间性和故事情节交互的生成式AI系统。用户可以通过击打代表里程碑、技能和随机事件的球来进行互动，提示、碰撞和反弹体现了在不确定性下的决策过程。在24名参与者的单被试实验中，与聊天机器人基准相比，CareerPooler显著提高了参与度、信息获取、满意度和职业清晰度。定性研究结果显示，空间-故事情节交互促进了基于经验的学习、挫折后的韧性并减轻了心理负担。我们的研究结果为AI辅助职业探索系统的设计做出了贡献，并更广泛地表明，视觉化的类比互动可以使生成系统更具吸引力和满意度。 

---
# Tabular Data with Class Imbalance: Predicting Electric Vehicle Crash Severity with Pretrained Transformers (TabPFN) and Mamba-Based Models 

**Title (ZH)**: 不平衡类别的表格数据：基于Pretrained Transformers（TabPFN）和Mamba-Based模型预测电动车辆碰撞严重程度 

**Authors**: Shriyank Somvanshi, Pavan Hebli, Gaurab Chhetri, Subasish Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.11449)  

**Abstract**: This study presents a deep tabular learning framework for predicting crash severity in electric vehicle (EV) collisions using real-world crash data from Texas (2017-2023). After filtering for electric-only vehicles, 23,301 EV-involved crash records were analyzed. Feature importance techniques using XGBoost and Random Forest identified intersection relation, first harmful event, person age, crash speed limit, and day of week as the top predictors, along with advanced safety features like automatic emergency braking. To address class imbalance, Synthetic Minority Over-sampling Technique and Edited Nearest Neighbors (SMOTEENN) resampling was applied. Three state-of-the-art deep tabular models, TabPFN, MambaNet, and MambaAttention, were benchmarked for severity prediction. While TabPFN demonstrated strong generalization, MambaAttention achieved superior performance in classifying severe injury cases due to its attention-based feature reweighting. The findings highlight the potential of deep tabular architectures for improving crash severity prediction and enabling data-driven safety interventions in EV crash contexts. 

**Abstract (ZH)**: 本研究提出了一种深度表格学习框架，使用2017-2023年德克萨斯州实际碰撞数据预测电动汽车（EV）碰撞 severity，经过筛选后的涉及纯电动汽车的碰撞记录共分析了23,301条。特征重要性技术使用XGBoost和随机森林确定了交叉口关系、首次有害事件、人员年龄、碰撞速度限制和星期几等顶级预测因子，同时还包括先进的安全特征如自动紧急制动。为解决类别不平衡问题，应用了合成少数过采样技术与编辑最近邻（SMOTEENN）重采样方法。三种最先进的深度表格模型，TabPFN、MambaNet和MambaAttention，被用于碰撞 severity 预测基准测试。尽管TabPFN在泛化能力上表现出色，但MambaAttention由于基于注意力机制的特征加权，在分类严重损伤案例方面表现更优。本研究结果强调了深度表格架构在提高电动汽车碰撞 severity 预测中的潜力，并促进了基于数据的安全干预措施。 

---
# FuseCodec: Semantic-Contextual Fusion and Supervision for Neural Codecs 

**Title (ZH)**: FuseCodec: 语义-上下文融合与监督神经编码器 

**Authors**: Md Mubtasim Ahasan, Rafat Hasan Khan, Tasnim Mohiuddin, Aman Chadha, Tariq Iqbal, M Ashraful Amin, Amin Ahsan Ali, Md Mofijul Islam, A K M Mahbubur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2509.11425)  

**Abstract**: Speech tokenization enables discrete representation and facilitates speech language modeling. However, existing neural codecs capture low-level acoustic features, overlooking the semantic and contextual cues inherent to human speech. While recent efforts introduced semantic representations from self-supervised speech models or incorporated contextual representations from pre-trained language models, challenges remain in aligning and unifying the semantic and contextual representations. We introduce FuseCodec, which unifies acoustic, semantic, and contextual representations through strong cross-modal alignment and globally informed supervision. We propose three complementary techniques: (i) Latent Representation Fusion, integrating semantic and contextual features directly into the encoder latent space for robust and unified representation learning; (ii) Global Semantic-Contextual Supervision, supervising discrete tokens with globally pooled and broadcasted representations to enhance temporal consistency and cross-modal alignment; and (iii) Temporally Aligned Contextual Supervision, strengthening alignment by dynamically matching contextual and speech tokens within a local window for fine-grained token-level supervision. We further introduce FuseCodec-TTS, demonstrating our methodology's applicability to zero-shot speech synthesis. Empirically, FuseCodec achieves state-of-the-art performance in LibriSpeech, surpassing EnCodec, SpeechTokenizer, and DAC in transcription accuracy, perceptual quality, intelligibility, and speaker similarity. Results highlight the effectiveness of contextually and semantically guided tokenization for speech tokenization and downstream tasks. Code and pretrained models are available at this https URL. 

**Abstract (ZH)**: 语音分词 enables 离散表示并促进语音语言建模。然而，现有的神经编解码器捕获低级声学特征，忽略了人类语音中固有的语义和上下文线索。虽然最近的努力引入了从自监督语音模型获得的语义表示，或将从预训练语言模型获得的上下文表示纳入其中，但语义和上下文表示的对齐和统一仍然存在挑战。我们引入了 FuseCodec，通过强大的跨模态对齐和全局信息监督统一声学、语义和上下文表示。我们提出了三种互补的技术：（i）潜在表示融合，将语义和上下文特征直接整合到编码器的潜在空间中，以实现稳健且统一的表示学习；（ii）全局语义-上下文监督，使用全局聚合和广播表示监督离散的标记，以增强时间一致性和跨模态对齐；（iii）时间对齐的上下文监督，在局部窗口内动态匹配上下文和语音标记，以实现细粒度的标记级监督。此外，我们还引入了 FuseCodec-TTS，展示了我们的方法在零样本语音合成中的适用性。实验结果表明，FuseCodec 在 LibriSpeech 中达到了最先进的性能，超越了 EnCodec、SpeechTokenizer 和 DAC 在转录准确性、感知质量、可懂度和说话人相似度方面的表现。结果突显了上下文和语义引导的分词在语音分词及后续任务中的有效性。代码和预训练模型可通过以下链接获取。 

---
# Framing AI System Benchmarking as a Learning Task: FlexBench and the Open MLPerf Dataset 

**Title (ZH)**: 将AI系统基准测试 framing 为一个学习任务：FlexBench 及开放的 MLPerf 数据集 

**Authors**: Grigori Fursin, Daniel Altunay  

**Link**: [PDF](https://arxiv.org/pdf/2509.11413)  

**Abstract**: Existing AI system benchmarks such as MLPerf often struggle to keep pace with the rapidly evolving AI landscape, making it difficult to support informed deployment, optimization, and co-design decisions for AI systems. We suggest that benchmarking itself can be framed as an AI task - one in which models are continuously evaluated and optimized across diverse datasets, software, and hardware, using key metrics such as accuracy, latency, throughput, energy consumption, and cost. To support this perspective, we present FlexBench: a modular extension of the MLPerf LLM inference benchmark, integrated with HuggingFace and designed to provide relevant and actionable insights. Benchmarking results and metadata are collected into an Open MLPerf Dataset, which can be collaboratively curated, extended, and leveraged for predictive modeling and feature engineering. We successfully validated the FlexBench concept through MLPerf Inference submissions, including evaluations of DeepSeek R1 and LLaMA 3.3 on commodity servers. The broader objective is to enable practitioners to make cost-effective AI deployment decisions that reflect their available resources, requirements, and constraints. 

**Abstract (ZH)**: 现有的AI系统基准测试如MLPerf往往难以跟上快速发展的AI景观，使得无法为AI系统的部署、优化和协同设计提供有力的支持。我们认为，基准测试本身可以被视为一项AI任务——即模型在多样化的数据集、软件和硬件上持续地被评价和优化，使用诸如准确率、延迟、吞吐量、能耗和成本等关键指标。为支持这一观点，我们提出FlexBench：MLPerf LLM推理基准的一个模块化扩展，集成HuggingFace，并旨在提供相关且可行的见解。基准测试结果和元数据被收集到Open MLPerf数据集中，可以协作维护、扩展并用于预测建模和特征工程。我们通过MLPerf推理提交验证了FlexBench的概念，包括在通用服务器上对DeepSeek R1和LLaMA 3.3的评估。更大的目标是使从业者能够根据其可用资源、需求和约束条件，做出成本效益高的AI部署决策。 

---
# From Firewalls to Frontiers: AI Red-Teaming is a Domain-Specific Evolution of Cyber Red-Teaming 

**Title (ZH)**: 从防火墙到前沿：AI 红队演练是网络红队演练的领域特定演化 

**Authors**: Anusha Sinha, Keltin Grimes, James Lucassen, Michael Feffer, Nathan VanHoudnos, Zhiwei Steven Wu, Hoda Heidari  

**Link**: [PDF](https://arxiv.org/pdf/2509.11398)  

**Abstract**: A red team simulates adversary attacks to help defenders find effective strategies to defend their systems in a real-world operational setting. As more enterprise systems adopt AI, red-teaming will need to evolve to address the unique vulnerabilities and risks posed by AI systems. We take the position that AI systems can be more effectively red-teamed if AI red-teaming is recognized as a domain-specific evolution of cyber red-teaming. Specifically, we argue that existing Cyber Red Teams who adopt this framing will be able to better evaluate systems with AI components by recognizing that AI poses new risks, has new failure modes to exploit, and often contains unpatchable bugs that re-prioritize disclosure and mitigation strategies. Similarly, adopting a cybersecurity framing will allow existing AI Red Teams to leverage a well-tested structure to emulate realistic adversaries, promote mutual accountability with formal rules of engagement, and provide a pattern to mature the tooling necessary for repeatable, scalable engagements. In these ways, the merging of AI and Cyber Red Teams will create a robust security ecosystem and best position the community to adapt to the rapidly changing threat landscape. 

**Abstract (ZH)**: 人工智能红队模拟对手攻击以帮助企业防御者在实际操作环境中找到有效的防御策略。随着企业系统越来越多地采用AI，红队行动需要演进以应对AI系统带来的独特漏洞和风险。我们认为，如果将AI红队行动视为网络红队行动的领域特定演进，AI系统可以更有效地接受红队测试。具体而言，现有的网络红队行动如果采取这种框架，将能够更好地评估包含AI组件的系统，认识到AI带来了新的风险、新的利用漏洞模式，并且经常包含无法修补的漏洞，这意味着需要重新调整披露和缓解策略的优先级。同样，采用网络安全部署框架将使现有的AI红队能够利用经过验证的结构来模拟现实的对手、促进共同问责制并通过正式的规则参与合作，并提供一套模式以成熟必要的工具以实现重复性和可扩展性。通过这种方式，AI和网络红队的结合将创造一个稳健的安全生态系统，并使社区能够更好地适应迅速变化的威胁环境。 

---
# A five-layer framework for AI governance: integrating regulation, standards, and certification 

**Title (ZH)**: 五层框架下的AI治理：整合监管、标准与认证 

**Authors**: Avinash Agarwal, Manisha J. Nene  

**Link**: [PDF](https://arxiv.org/pdf/2509.11332)  

**Abstract**: Purpose: The governance of artificial iintelligence (AI) systems requires a structured approach that connects high-level regulatory principles with practical implementation. Existing frameworks lack clarity on how regulations translate into conformity mechanisms, leading to gaps in compliance and enforcement. This paper addresses this critical gap in AI governance.
Methodology/Approach: A five-layer AI governance framework is proposed, spanning from broad regulatory mandates to specific standards, assessment methodologies, and certification processes. By narrowing its scope through progressively focused layers, the framework provides a structured pathway to meet technical, regulatory, and ethical requirements. Its applicability is validated through two case studies on AI fairness and AI incident reporting.
Findings: The case studies demonstrate the framework's ability to identify gaps in legal mandates, standardization, and implementation. It adapts to both global and region-specific AI governance needs, mapping regulatory mandates with practical applications to improve compliance and risk management.
Practical Implications - By offering a clear and actionable roadmap, this work contributes to global AI governance by equipping policymakers, regulators, and industry stakeholders with a model to enhance compliance and risk management.
Social Implications: The framework supports the development of policies that build public trust and promote the ethical use of AI for the benefit of society.
Originality/Value: This study proposes a five-layer AI governance framework that bridges high-level regulatory mandates and implementation guidelines. Validated through case studies on AI fairness and incident reporting, it identifies gaps such as missing standardized assessment procedures and reporting mechanisms, providing a structured foundation for targeted governance measures. 

**Abstract (ZH)**: 目的：人工智能（AI）系统的治理需要一种有条理的方法，将高层次的监管原则与实际实施联系起来。现有框架在将监管要求转化为合规机制方面缺乏清晰性，导致合规性和执行方面存在缺口。本文解决了AI治理中的这一关键缺口。

方法/步骤：提出了一种五层AI治理框架，涵盖从广泛的监管要求到具体的标准、评估方法和认证流程等多个层面。通过逐步聚焦各层，该框架为满足技术、监管和伦理要求提供了结构化的途径。通过两个关于AI公平性和AI事件报告的案例研究，验证了其适用性。

发现：案例研究展示了该框架在识别法律法规、标准化和实施方面的缺口方面的能力。该框架能够适应全球和区域特定的AI治理需求，通过将监管要求与实际应用结合，改善合规性和风险管理。

实践意义：通过提供清晰可操作的路线图，本研究为全球AI治理提供了贡献，使政策制定者、监管机构和行业利益相关者能够提升合规性和风险管理。

社会影响：该框架支持制定了能够建立公众信任并促进AI公正使用的政策，从而造福社会。

原创性/价值：本研究提出了一种五层AI治理框架，连接了高层次的监管要求和实施指南。通过AI公平性和事件报告的案例研究验证，该框架识别了缺失的标准评估程序和报告机制等方面的缺口，为针对性的治理措施提供了结构化的基础。 

---
# Motion Estimation for Multi-Object Tracking using KalmanNet with Semantic-Independent Encoding 

**Title (ZH)**: 使用语义独立编码的KalmanNet运动估计在多对象跟踪中的应用 

**Authors**: Jian Song, Wei Mei, Yunfeng Xu, Qiang Fu, Renke Kou, Lina Bu, Yucheng Long  

**Link**: [PDF](https://arxiv.org/pdf/2509.11323)  

**Abstract**: Motion estimation is a crucial component in multi-object tracking (MOT).
It predicts the trajectory of objects by analyzing the changes in their positions in consecutive frames of images, reducing tracking failures and identity switches.
The Kalman filter (KF) based on the linear constant-velocity model is one of the most commonly used methods in MOT.
However, it may yield unsatisfactory results when KF's parameters are mismatched and objects move in non-stationary.
In this work, we utilize the learning-aided filter to handle the motion estimation of MOT.
In particular, we propose a novel method named Semantic-Independent KalmanNet (SIKNet), which encodes the state vector (the input feature) using a Semantic-Independent Encoder (SIE) by two steps.
First, the SIE uses a 1D convolution with a kernel size of 1, which convolves along the dimension of homogeneous-semantic elements across different state vectors to encode independent semantic information.
Then it employs a fully-connected layer and a nonlinear activation layer to encode nonlinear and cross-dependency information between heterogeneous-semantic elements.
To independently evaluate the performance of the motion estimation module in MOT, we constructed a large-scale semi-simulated dataset from several open-source MOT datasets.
Experimental results demonstrate that the proposed SIKNet outperforms the traditional KF and achieves superior robustness and accuracy than existing learning-aided filters.
The code is available at (this https URL and this https URL). 

**Abstract (ZH)**: 多对象跟踪（MOT）中的运动估计是其关键组成部分。它通过分析图像连续帧中目标位置的变化来预测目标的轨迹，从而减少跟踪失败和身份切换。基于线性恒速模型的卡尔曼滤波器（KF）是MOT中广泛使用的方法之一。然而，当KF的参数不匹配且目标进行非平稳运动时，可能会导致不满意的估计结果。在本文中，我们利用辅助学习滤波器来处理MOT的运动估计。特别地，我们提出了一种名为语义独立卡尔曼网络（SIKNet）的新方法，该方法通过两步将状态向量（输入特征）编码为语义独立编码器（SIE）。首先，SIE使用内核大小为1的1D卷积，沿不同状态向量中的同质语义元素进行卷积，以编码独立的语义信息。然后，它使用全连接层和非线性激活层来编码异质语义元素之间的非线性和交叉依赖信息。为了独立评估MOT中的运动估计模块的性能，我们从几个开源MOT数据集中构建了一个大规模半模拟数据集。实验结果表明，所提出的SIKNet优于传统的卡尔曼滤波器，并且在鲁棒性和准确性方面优于现有的辅助学习滤波器。代码可从 (this https URL 和 this https URL) 获取。 

---
# Weakly Supervised Vulnerability Localization via Multiple Instance Learning 

**Title (ZH)**: 弱监督漏洞定位：基于实例学习的方法 

**Authors**: Wenchao Gu, Yupan Chen, Yanlin Wang, Hongyu Zhang, Cuiyun Gao, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11312)  

**Abstract**: Software vulnerability detection has emerged as a significant concern in the field of software security recently, capturing the attention of numerous researchers and developers. Most previous approaches focus on coarse-grained vulnerability detection, such as at the function or file level. However, the developers would still encounter the challenge of manually inspecting a large volume of code inside the vulnerable function to identify the specific vulnerable statements for modification, indicating the importance of vulnerability localization. Training the model for vulnerability localization usually requires ground-truth labels at the statement-level, and labeling vulnerable statements demands expert knowledge, which incurs high costs. Hence, the demand for an approach that eliminates the need for additional labeling at the statement-level is on the rise. To tackle this problem, we propose a novel approach called WAVES for WeAkly supervised Vulnerability Localization via multiplE inStance learning, which does not need the additional statement-level labels during the training. WAVES has the capability to determine whether a function is vulnerable (i.e., vulnerability detection) and pinpoint the vulnerable statements (i.e., vulnerability localization). Specifically, inspired by the concept of multiple instance learning, WAVES converts the ground-truth label at the function-level into pseudo labels for individual statements, eliminating the need for additional statement-level labeling. These pseudo labels are utilized to train the classifiers for the function-level representation vectors. Extensive experimentation on three popular benchmark datasets demonstrates that, in comparison to previous baselines, our approach achieves comparable performance in vulnerability detection and state-of-the-art performance in statement-level vulnerability localization. 

**Abstract (ZH)**: 软件漏洞检测已成为软件安全领域的一项重要关注点，吸引了众多研究人员和开发者的注意。大多数先前的方法集中在粗粒度的漏洞检测，如函数或文件级别。然而，开发者仍然面临着手动检查大量脆弱函数内部代码以识别特定脆弱语句进行修改的挑战，这突显了漏洞定位的重要性。训练模型进行漏洞定位通常需要语句级别的地面真实标签，而标注脆弱语句需要专家知识，这会产生高昂的成本。因此，减少或消除语句级别标注的需求的方法需求日益增加。为解决这一问题，我们提出了一种名为WAVES的新方法，WAVES通过多实例学习进行弱监督漏洞定位，不需要在训练过程中使用额外的语句级别标签。WAVES具有确定函数是否脆弱（即漏洞检测）并定位具体脆弱语句（即漏洞定位）的能力。具体而言，WAVES借鉴了多实例学习的概念，将函数级别的地面真实标签转化为个别语句的伪标签，从而消除额外的语句级别标注需求。这些伪标签被用于训练表示向量的分类器。在三个流行基准数据集上的广泛实验表明，与先前的基线方法相比，我们的方法在漏洞检测方面取得了可比的性能，并在语句级别漏洞定位方面达到了最先进的效果。 

---
# Opal: An Operator Algebra View of RLHF 

**Title (ZH)**: Opal: RLHF的一种运算代数视角 

**Authors**: Madhava Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2509.11298)  

**Abstract**: We present Opal, an operator view of reinforcement learning from human feedback (RLHF). Objectives are expressed as ladders of two primitives on a base utility: additive penalties and multiplicative pairwise weights. We describe a simple reduction law with if-and-only-if conditions: such ladders collapse to a normal form on pairwise margins when the reference is fixed, penalties are additive, and weights are independent of intermediate margins. When these assumptions do not hold (reference shift, non-additive gates, score-dependent weights), small examples demonstrate non-reducibility.
Building on this view, we introduce GKPO (Generalized Kernel Preference Object), a canonical schema in which many RLHF methods can be represented and, when reducible, mapped back from. GKPO provides a standard JSON serialization, canonicalization and hashing rules, and explicit flags with finite witnesses when assumptions fail.
We illustrate these ideas with GKPO examples for DPO, RRHF, and ORPO, along with cross-method conversions (where assumptions permit) and minimal stress tests (SHIFT/GATE/SCORE) that highlight non-reducibility. A lightweight Python reference library accompanies the schema, implementing canonical hashing and adapters for DPO and RRHF. 

**Abstract (ZH)**: Opal: 一种基于人类反馈强化学习的操作视角 

---
# Energy-Aware 6G Network Design: A Survey 

**Title (ZH)**: 面向能源aware的6G网络设计：一个综述 

**Authors**: Rashmi Kamran, Mahesh Ganesh Bhat, Pranav Jha, Shana Moothedath, Manjesh Hanawal, Prasanna Chaporkar  

**Link**: [PDF](https://arxiv.org/pdf/2509.11289)  

**Abstract**: 6th Generation (6G) mobile networks are envisioned to support several new capabilities and data-centric applications for unprecedented number of users, potentially raising significant energy efficiency and sustainability concerns. This brings focus on sustainability as one of the key objectives in the their design. To move towards sustainable solution, research and standardization community is focusing on several key issues like energy information monitoring and exposure, use of renewable energy, and use of Artificial Intelligence/Machine Learning (AI/ML) for improving the energy efficiency in 6G networks. The goal is to build energy-aware solutions that takes into account the energy information resulting in energy efficient networks. Design of energy-aware 6G networks brings in new challenges like increased overheads in gathering and exposing of energy related information, and the associated user consent management. The aim of this paper is to provide a comprehensive survey of methods used for design of energy efficient 6G networks, like energy harvesting, energy models and parameters, classification of energy-aware services, and AI/ML-based solutions. The survey also includes few use cases that demonstrate the benefits of incorporating energy awareness into network decisions. Several ongoing standardization efforts in 3GPP, ITU, and IEEE are included to provide insights into the ongoing work and highlight the opportunities for new contributions. We conclude this survey with open research problems and challenges that can be explored to make energy-aware design feasible and ensure optimality regarding performance and energy goals for 6G networks. 

**Abstract (ZH)**: 第六代（6G）移动网络的设计面临显著的能源效率和可持续性挑战，旨在支持前所未有的大量用户的新能力和数据中心应用。为此，可持续性成为其设计中的关键目标之一。为实现可持续解决方案，研究与标准化社区重点关注能源信息监控与暴露、可再生能源的使用以及通过人工智能/机器学习（AI/ML）提高6G网络能源效率等多个关键问题。目标是构建考虑能源信息的能源感知解决方案，从而实现能源效率更高的网络。设计能源感知6G网络带来了新的挑战，如能源相关信息的采集和暴露增加的开销，以及与此相关的用户同意管理。本文旨在提供一种全面的调查，涵盖用于设计节能6G网络的方法，如能量采集、能量模型与参数、能源感知服务分类以及基于AI/ML的解决方案。调查还包含几个案例，展示了将能源感知纳入网络决策中的益处。此外，本文包含了3GPP、ITU和IEEE等组织中的多项正在进行的标准制定工作，以提供对正在进行的工作的见解，并突出新的贡献机会。最后，本文以开放的研究问题和挑战作为结论，这些问题和挑战可以探索以使能源感知设计可行，并确保6G网络在性能和能源目标方面的最优化。 

---
# Efficient Single-Step Framework for Incremental Class Learning in Neural Networks 

**Title (ZH)**: 高效的一步增量类学习框架在神经网络中 

**Authors**: Alejandro Dopico-Castro, Oscar Fontenla-Romero, Bertha Guijarro-Berdiñas, Amparo Alonso-Betanzos  

**Link**: [PDF](https://arxiv.org/pdf/2509.11285)  

**Abstract**: Incremental learning remains a critical challenge in machine learning, as models often struggle with catastrophic forgetting -the tendency to lose previously acquired knowledge when learning new information. These challenges are even more pronounced in resource-limited settings. Many existing Class Incremental Learning (CIL) methods achieve high accuracy by continually adapting their feature representations; however, they often require substantial computational resources and complex, iterative training procedures. This work introduces CIFNet (Class Incremental and Frugal Network), a novel CIL approach that addresses these limitations by offering a highly efficient and sustainable solution. CIFNet's key innovation lies in its novel integration of several existing, yet separately explored, components: a pre-trained and frozen feature extractor, a compressed data buffer, and an efficient non-iterative one-layer neural network for classification. A pre-trained and frozen feature extractor eliminates computationally expensive fine-tuning of the backbone. This, combined with a compressed buffer for efficient memory use, enables CIFNet to perform efficient class-incremental learning through a single-step optimization process on fixed features, minimizing computational overhead and training time without requiring multiple weight updates. Experiments on benchmark datasets confirm that CIFNet effectively mitigates catastrophic forgetting at the classifier level, achieving high accuracy comparable to that of existing state-of-the-art methods, while substantially improving training efficiency and sustainability. CIFNet represents a significant advancement in making class-incremental learning more accessible and pragmatic in environments with limited resources, especially when strong pre-trained feature extractors are available. 

**Abstract (ZH)**: 增量学习仍然是机器学习中的一个关键挑战，模型往往挣扎于灾难性遗忘——在学习新信息时失去之前获取的知识的倾向。这些挑战在资源受限的环境中尤为突出。许多现有的类增量学习（CIL）方法通过不断调整其特征表示来实现高精度，但它们通常需要大量的计算资源和复杂的迭代培训过程。本文引入了CIFNet（类增量和节俭网络），这是一种新颖的CIL方法，通过提供高效且可持续的解决方案来解决这些限制。CIFNet的关键创新在于其新颖地整合了几种现有但独立探索的组件：预训练并冻结的特征提取器、压缩数据缓冲区以及用于分类的高效单层神经网络。预训练并冻结的特征提取器消除了对骨干进行昂贵微调的需求。结合高效的压缩缓冲区以实现内存高效使用，CIFNet能够通过固定特征的一步优化过程实现高效的类增量学习，从而最小化计算开销和培训时间，而无需多轮权重更新。基准数据集上的实验证实，CIFNet在分类器层面有效缓解了灾难性遗忘的问题，实现了与现有先进方法相当的高精度，同时显著提高了培训效率和可持续性。CIFNet代表了在资源受限环境下使类增量学习更具可行性和实际应用的重大进展，尤其是在强预训练特征提取器可用的情况下。 

---
# TransZero: Parallel Tree Expansion in MuZero using Transformer Networks 

**Title (ZH)**: TransZero: 使用Transformer网络的MuZero中的并行树扩展 

**Authors**: Emil Malmsten, Wendelin Böhmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.11233)  

**Abstract**: We present TransZero, a model-based reinforcement learning algorithm that removes the sequential bottleneck in Monte Carlo Tree Search (MCTS). Unlike MuZero, which constructs its search tree step by step using a recurrent dynamics model, TransZero employs a transformer-based network to generate multiple latent future states simultaneously. Combined with the Mean-Variance Constrained (MVC) evaluator that eliminates dependence on inherently sequential visitation counts, our approach enables the parallel expansion of entire subtrees during planning. Experiments in MiniGrid and LunarLander show that TransZero achieves up to an eleven-fold speedup in wall-clock time compared to MuZero while maintaining sample efficiency. These results demonstrate that parallel tree construction can substantially accelerate model-based reinforcement learning, bringing real-time decision-making in complex environments closer to practice. The code is publicly available on GitHub. 

**Abstract (ZH)**: 基于模型的 reinforcement 学习算法 TransZero：去除蒙特卡洛树搜索中的序列瓶颈 

---
# Quantum Architecture Search for Solving Quantum Machine Learning Tasks 

**Title (ZH)**: 量子架构搜索用于解决量子机器学习任务 

**Authors**: Michael Kölle, Simon Salfer, Tobias Rohe, Philipp Altmann, Claudia Linnhoff-Popien  

**Link**: [PDF](https://arxiv.org/pdf/2509.11198)  

**Abstract**: Quantum computing leverages quantum mechanics to address computational problems in ways that differ fundamentally from classical approaches. While current quantum hardware remains error-prone and limited in scale, Variational Quantum Circuits offer a noise-resilient framework suitable for today's devices. The performance of these circuits strongly depends on the underlying architecture of their parameterized quantum components. Identifying efficient, hardware-compatible quantum circuit architectures -- known as Quantum Architecture Search (QAS) -- is therefore essential. Manual QAS is complex and error-prone, motivating efforts to automate it. Among various automated strategies, Reinforcement Learning (RL) remains underexplored, particularly in Quantum Machine Learning contexts. This work introduces RL-QAS, a framework that applies RL to discover effective circuit architectures for classification tasks. We evaluate RL-QAS using the Iris and binary MNIST datasets. The agent autonomously discovers low-complexity circuit designs that achieve high test accuracy. Our results show that RL is a viable approach for automated architecture search in quantum machine learning. However, applying RL-QAS to more complex tasks will require further refinement of the search strategy and performance evaluation mechanisms. 

**Abstract (ZH)**: 量子计算利用量子力学以根本不同的方式解决计算问题，不同于经典方法。尽管当前的量子硬件仍然容易出错且规模有限，变异量子电路提供了适合当今设备的噪声鲁棒框架。这些电路的表现强烈依赖于它们的参数化量子组件的基础架构。因此，识别高效且硬件兼容的量子电路架构——称为量子架构搜索（QAS）——是至关重要的。手工进行QAS复杂且易出错，因此推动了自动化努力。在各种自动化策略中，强化学习（RL）在量子机器学习上下文中的应用尚属未竟之域。本文介绍了RL-QAS框架，该框架将RL应用于发现适用于分类任务的有效电路架构。我们使用鸢尾花和二进制MNIST数据集评估了RL-QAS。代理自主发现了低复杂度的电路设计，并实现了高测试准确性。我们的结果表明，RL是量子机器学习中自动化架构搜索的可行方法，但将RL-QAS应用于更复杂任务将需要进一步细化搜索策略和性能评估机制。 

---
# Federated Recommender System with Data Valuation for E-commerce Platform 

**Title (ZH)**: 基于数据估值的联邦推荐系统在电子商务平台中应用 

**Authors**: Jongwon Park, Minku Kang, Wooseok Sim, Soyoung Lee, Hogun Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.11196)  

**Abstract**: Federated Learning (FL) is gaining prominence in machine learning as privacy concerns grow. This paradigm allows each client (e.g., an individual online store) to train a recommendation model locally while sharing only model updates, without exposing the raw interaction logs to a central server, thereby preserving privacy in a decentralized environment. Nonetheless, most existing FL-based recommender systems still rely solely on each client's private data, despite the abundance of publicly available datasets that could be leveraged to enrich local training; this potential remains largely underexplored. To this end, we consider a realistic scenario wherein a large shopping platform collaborates with multiple small online stores to build a global recommender system. The platform possesses global data, such as shareable user and item lists, while each store holds a portion of interaction data privately (or locally). Although integrating global data can help mitigate the limitations of sparse and biased clients' local data, it also introduces additional challenges: simply combining all global interactions can amplify noise and irrelevant patterns, worsening personalization and increasing computational costs. To address these challenges, we propose FedGDVE, which selectively augments each client's local graph with semantically aligned samples from the global dataset. FedGDVE employs: (i) a pre-trained graph encoder to extract global structural features, (ii) a local valid predictor to assess client-specific relevance, (iii) a reinforcement-learning-based probability estimator to filter and sample only the most pertinent global interactions. FedGDVE improves performance by up to 34.86% on recognized benchmarks in FL environments. 

**Abstract (ZH)**: 联邦学习（FL）在隐私担忧增加的情况下正逐渐成为机器学习领域的热点。这种范式允许每个客户端（例如，一个在线商店）在本地训练推荐模型，仅分享模型更新而不暴露原始交互日志，从而在分散的环境中保护隐私。然而，现有的大多数基于FL的推荐系统仍然仅依赖每个客户端的私有数据，尽管有大量的公开数据集可以利用以丰富本地训练，但这一潜力仍未得到充分开发。为此，我们考虑一种现实场景，大型购物平台与多个小型在线商店合作构建全球推荐系统。平台拥有全局数据，如可共享的用户和商品列表，而每个商店则私下持有部分交互数据。尽管整合全局数据可以帮助缓解客户端本地数据稀疏和偏差的限制，但也引入了新的挑战：简单地合并所有全局交互会放大噪声和不相关模式，恶化个性化并增加计算成本。为解决这些问题，我们提出FedGDVE，它选择性地将全局数据中语义对齐的样本补充到每个客户端的本地图中。FedGDVE采用：（i）预训练的图编码器提取全局结构特征，（ii）局部有效预测器评估特定于客户端的相关性，（iii）基于强化学习的概率估计器筛选和采样仅最相关的全局交互。在FL环境中，FedGDVE在已认可的基准测试中可实现高达34.86%的性能提升。 

---
# Investigating the Lottery Ticket Hypothesis for Variational Quantum Circuits 

**Title (ZH)**: 探索变分量子电路中的彩票票据假设 

**Authors**: Michael Kölle, Leonhard Klingert, Julian Schönberger, Philipp Altmann, Tobias Rohe, Claudia Linnhoff-Popien  

**Link**: [PDF](https://arxiv.org/pdf/2509.11190)  

**Abstract**: Quantum computing is an emerging field in computer science that has seen considerable progress in recent years, especially in machine learning. By harnessing the principles of quantum physics, it can surpass the limitations of classical algorithms. However, variational quantum circuits (VQCs), which rely on adjustable parameters, often face the barren plateau phenomenon, hindering optimization. The Lottery Ticket Hypothesis (LTH) is a recent concept in classical machine learning that has led to notable improvements in parameter efficiency for neural networks. It states that within a large network, a smaller, more efficient subnetwork, or ''winning ticket,'' can achieve comparable performance, potentially circumventing plateau challenges. In this work, we investigate whether this idea can apply to VQCs. We show that the weak LTH holds for VQCs, revealing winning tickets that retain just 26.0\% of the original parameters. For the strong LTH, where a pruning mask is learned without any training, we discovered a winning ticket in a binary VQC, achieving 100\% accuracy with only 45\% of the weights. These findings indicate that LTH may mitigate barren plateaus by reducing parameter counts while preserving performance, thus enhancing the efficiency of VQCs in quantum machine learning tasks. 

**Abstract (ZH)**: 量子计算是一种新兴的计算机科学领域，在近年来取得了显著进展，尤其是在机器学习方面。通过利用量子物理的基本原理，它可以超越古典算法的限制。然而，依赖可调参数的变分量子电路（VQCs）往往面临 barren plateau 现象，阻碍了优化过程。经典机器学习中最近提出的彩票票 Hypothesis （LTH）概念，在神经网络的参数效率方面取得了显著改进。它表明，在一个大网络中，存在一个更小、更高效的子网络，即“获胜彩票”，可以实现 comparable 的性能，可能绕过 plateau 挑战。在这项工作中，我们研究了这一思想能否应用于 VQCs。我们证明弱 LTH 在 VQCs 中成立，揭示了一个保留了原参数 26.0% 的获胜彩票。在强 LTH 情况下，通过学习剪枝掩码而无需任何训练的情况下，我们发现了一个二进制 VQC 的获胜彩票，仅使用 45% 的权重实现了 100% 的准确率。这些发现表明，LTH 可能通过减少参数数量同时保持性能来缓解 barren plateaus，从而增强 VQCs 在量子机器学习任务中的效率。 

---
# StegOT: Trade-offs in Steganography via Optimal Transport 

**Title (ZH)**: StegOT：最优输运视角下的隐写术权衡 

**Authors**: Chengde Lin, Xuezhu Gong, Shuxue Ding, Mingzhe Yang, Xijun Lu, Chengjun Mo  

**Link**: [PDF](https://arxiv.org/pdf/2509.11178)  

**Abstract**: Image hiding is often referred to as steganography, which aims to hide a secret image in a cover image of the same resolution. Many steganography models are based on genera-tive adversarial networks (GANs) and variational autoencoders (VAEs). However, most existing models suffer from mode collapse. Mode collapse will lead to an information imbalance between the cover and secret images in the stego image and further affect the subsequent extraction. To address these challenges, this paper proposes StegOT, an autoencoder-based steganography model incorporating optimal transport theory. We designed the multiple channel optimal transport (MCOT) module to transform the feature distribution, which exhibits multiple peaks, into a single peak to achieve the trade-off of information. Experiments demonstrate that we not only achieve a trade-off between the cover and secret images but also enhance the quality of both the stego and recovery images. The source code will be released on this https URL. 

**Abstract (ZH)**: 基于最优运输理论的自编码器隐写模型StegOT 

---
# Your Compiler is Backdooring Your Model: Understanding and Exploiting Compilation Inconsistency Vulnerabilities in Deep Learning Compilers 

**Title (ZH)**: 你的编译器在后门你的模型：理解并利用深度学习编译器中的编译一致性漏洞 

**Authors**: Simin Chen, Jinjun Peng, Yixin He, Junfeng Yang, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2509.11173)  

**Abstract**: Deep learning (DL) compilers are core infrastructure in modern DL systems, offering flexibility and scalability beyond vendor-specific libraries. This work uncovers a fundamental vulnerability in their design: can an official, unmodified compiler alter a model's semantics during compilation and introduce hidden backdoors? We study both adversarial and natural settings. In the adversarial case, we craft benign models where triggers have no effect pre-compilation but become effective backdoors after compilation. Tested on six models, three commercial compilers, and two hardware platforms, our attack yields 100% success on triggered inputs while preserving normal accuracy and remaining undetected by state-of-the-art detectors. The attack generalizes across compilers, hardware, and floating-point settings. In the natural setting, we analyze the top 100 HuggingFace models (including one with 220M+ downloads) and find natural triggers in 31 models. This shows that compilers can introduce risks even without adversarial manipulation.
Our results reveal an overlooked threat: unmodified DL compilers can silently alter model semantics. To our knowledge, this is the first work to expose inherent security risks in DL compiler design, opening a new direction for secure and trustworthy ML. 

**Abstract (ZH)**: 深度学习编译器设计中的根本性漏洞：无修改编译器是否能在编译过程中悄无声息地改变模型 semantics 并引入隐藏后门？ 

---
# An Entropy-Guided Curriculum Learning Strategy for Data-Efficient Acoustic Scene Classification under Domain Shift 

**Title (ZH)**: 基于熵引导的学习策略：在领域迁移下实现数据高效声场景分类 

**Authors**: Peihong Zhang, Yuxuan Liu, Zhixin Li, Rui Sang, Yiqiang Cai, Yizhou Tan, Shengchen Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11168)  

**Abstract**: Acoustic Scene Classification (ASC) faces challenges in generalizing across recording devices, particularly when labeled data is limited. The DCASE 2024 Challenge Task 1 highlights this issue by requiring models to learn from small labeled subsets recorded on a few devices. These models need to then generalize to recordings from previously unseen devices under strict complexity constraints. While techniques such as data augmentation and the use of pre-trained models are well-established for improving model generalization, optimizing the training strategy represents a complementary yet less-explored path that introduces no additional architectural complexity or inference overhead. Among various training strategies, curriculum learning offers a promising paradigm by structuring the learning process from easier to harder examples. In this work, we propose an entropy-guided curriculum learning strategy to address the domain shift problem in data-efficient ASC. Specifically, we quantify the uncertainty of device domain predictions for each training sample by computing the Shannon entropy of the device posterior probabilities estimated by an auxiliary domain classifier. Using entropy as a proxy for domain invariance, the curriculum begins with high-entropy samples and gradually incorporates low-entropy, domain-specific ones to facilitate the learning of generalizable representations. Experimental results on multiple DCASE 2024 ASC baselines demonstrate that our strategy effectively mitigates domain shift, particularly under limited labeled data conditions. Our strategy is architecture-agnostic and introduces no additional inference cost, making it easily integrable into existing ASC baselines and offering a practical solution to domain shift. 

**Abstract (ZH)**: 声场景分类（ASC）在跨录音设备推广时面临挑战，尤其是当标注数据有限时。DCASE 2024 挑战任务 1 突出了这一问题，要求模型从少数设备记录的小标注子集中学到知识，并在严格复杂性约束下推广到未见过的设备的录音。虽然数据增强技术和预训练模型等方法已被证明可以提高模型的推广能力，但优化训练策略作为补充但仍较少探索的道路，且不增加额外的架构复杂度或推理开销。在各种训练策略中，分层学习提供了一种有前途的范式，通过从简单到复杂的示例逐步构建学习过程。在本文中，我们提出了一种基于熵的分层学习策略来解决数据高效ASC中的领域转移问题。具体而言，我们通过计算辅助领域分类器估计的设备后验概率的香农熵来量化每个训练样本的设备领域预测不确定性。利用熵作为领域不变性的代理，分层学习从高熵样本开始，并逐渐引入低熵、设备特定的样本，以促进学习到一般性表示。在多个DCASE 2024 ASC基线上进行的实验结果表明，我们的策略在标注数据有限的条件下有效缓解了领域转移问题。我们的策略对架构具有通用性，并且不引入额外的推理开销，易于集成到现有的ASC基线中，并提供了一种实用的领域转移解决方案。 

---
# Feature Space Topology Control via Hopkins Loss 

**Title (ZH)**: 基于Hopkins损失的空间特征拓扑控制 

**Authors**: Einari Vaaras, Manu Airaksinen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11154)  

**Abstract**: Feature space topology refers to the organization of samples within the feature space. Modifying this topology can be beneficial in machine learning applications, including dimensionality reduction, generative modeling, transfer learning, and robustness to adversarial attacks. This paper introduces a novel loss function, Hopkins loss, which leverages the Hopkins statistic to enforce a desired feature space topology, which is in contrast to existing topology-related methods that aim to preserve input feature topology. We evaluate the effectiveness of Hopkins loss on speech, text, and image data in two scenarios: classification and dimensionality reduction using nonlinear bottleneck autoencoders. Our experiments show that integrating Hopkins loss into classification or dimensionality reduction has only a small impact on classification performance while providing the benefit of modifying feature topology. 

**Abstract (ZH)**: 特征空间拓扑结构是指特征空间内样本的组织方式。修改这一拓扑结构在机器学习应用中包括降维、生成模型、迁移学习和对抗攻击鲁棒性等方面都是有益的。本文介绍了一种新的损失函数——霍普金斯损失，它利用霍普金斯统计量来强制执行期望的特征空间拓扑结构，而现有的拓扑相关方法则旨在保留输入特征拓扑结构。我们在语音、文本和图像数据中分别在分类和使用非线性瓶颈自动编码器的降维两种场景下评估了霍普金斯损失的有效性。实验结果显示，在分类中集成霍普金斯损失对分类性能的影响很小，而能够提供修改特征拓扑结构的好处。 

---
# Agentic Username Suggestion and Multimodal Gender Detection in Online Platforms: Introducing the PNGT-26K Dataset 

**Title (ZH)**: 代理用户名建议与多模态性别检测：PNGT-26K数据集介绍 

**Authors**: Farbod Bijary, Mohsen Ebadpour, Amirhosein Tajbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2509.11136)  

**Abstract**: Persian names present unique challenges for natural language processing applications, particularly in gender detection and digital identity creation, due to transliteration inconsistencies and cultural- specific naming patterns. Existing tools exhibit significant performance degradation on Persian names, while the scarcity of comprehensive datasets further compounds these limitations. To address these challenges, the present research introduces PNGT-26K, a comprehensive dataset of Persian names, their commonly associated gender, and their English transliteration, consisting of approximately 26,000 tuples. As a demonstration of how this resource can be utilized, we also introduce two frameworks, namely Open Gender Detection and Nominalist. Open Gender Detection is a production- grade, ready-to-use framework for using existing data from a user, such as profile photo and name, to give a probabilistic guess about the person's gender. Nominalist, the second framework introduced by this paper, utilizes agentic AI to help users choose a username for their social media accounts on any platform. It can be easily integrated into any website to provide a better user experience. The PNGT-26K dataset, Nominalist and Open Gender Detection frameworks are publicly available on Github. 

**Abstract (ZH)**: 波斯名字在自然语言处理应用中存在独特挑战，特别是在性别检测和数字身份创建方面，由于转写不一致和文化特定的命名模式。现有工具在处理波斯名字时表现出显著性能下降，而全面数据集的稀缺进一步加剧了这些限制。为应对这些挑战，本研究引入了PNGT-26K数据集，该数据集包含波斯名字及其常见的性别和英语转写，共计约26,000个条目。作为该资源的应用示范，我们还引入了两个框架：Open Gender Detection和Nominalist。Open Gender Detection是一个生产级、即用型框架，可用于根据用户的数据（如头像和名字）给出关于性别的人概率性猜测。Nominalist是本文引入的第二个框架，利用代理AI帮助用户为社交媒体账号选择用户名，可以轻松集成到任何网站中以提供更好的用户体验。PNGT-26K数据集、Nominalist和Open Gender Detection框架现已在Github上公开。 

---
# Application of Machine Learning for Correcting Defect-induced Neuromorphic Circuit Inference Errors 

**Title (ZH)**: 基于机器学习的缺陷诱导神经形态电路推理错误校正应用 

**Authors**: Vedant Sawal, Hiu Yung Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.11113)  

**Abstract**: This paper presents a machine learning-based approach to correct inference errors caused by stuck-at faults in fully analog ReRAM-based neuromorphic circuits. Using a Design-Technology Co-Optimization (DTCO) simulation framework, we model and analyze six spatial defect types-circular, circular-complement, ring, row, column, and checkerboard-across multiple layers of a multi-array neuromorphic architecture. We demonstrate that the proposed correction method, which employs a lightweight neural network trained on the circuit's output voltages, can recover up to 35% (from 55% to 90%) inference accuracy loss in defective scenarios. Our results, based on handwritten digit recognition tasks, show that even small corrective networks can significantly improve circuit robustness. This method offers a scalable and energy-efficient path toward enhanced yield and reliability for neuromorphic systems in edge and internet-of-things (IoTs) applications. In addition to correcting the specific defect types used during training, our method also demonstrates the ability to generalize-achieving reasonable accuracy when tested on different types of defects not seen during training. The framework can be readily extended to support real-time adaptive learning, enabling on-chip correction for dynamic or aging-induced fault profiles. 

**Abstract (ZH)**: 基于机器学习的修正方法用于纠正全模拟ReRAM神经形态电路中由固定节点故障引起的推理错误 

---
# Multi-Modal Sensing Aided mmWave Beamforming for V2V Communications with Transformers 

**Title (ZH)**: 基于变压器的多模态感知辅助毫米波波束成型在车对车通信中 

**Authors**: Muhammad Baqer Mollah, Honggang Wang, Hua Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11112)  

**Abstract**: Beamforming techniques are utilized in millimeter wave (mmWave) communication to address the inherent path loss limitation, thereby establishing and maintaining reliable connections. However, adopting standard defined beamforming approach in highly dynamic vehicular environments often incurs high beam training overheads and reduces the available airtime for communications, which is mainly due to exchanging pilot signals and exhaustive beam measurements. To this end, we present a multi-modal sensing and fusion learning framework as a potential alternative solution to reduce such overheads. In this framework, we first extract the features individually from the visual and GPS coordinates sensing modalities by modality specific encoders, and subsequently fuse the multimodal features to obtain predicted top-k beams so that the best line-of-sight links can be proactively established. To show the generalizability of the proposed framework, we perform a comprehensive experiment in four different vehicle-to-vehicle (V2V) scenarios from real-world multi-modal sensing and communication dataset. From the experiment, we observe that the proposed framework achieves up to 77.58% accuracy on predicting top-15 beams correctly, outperforms single modalities, incurs roughly as low as 2.32 dB average power loss, and considerably reduces the beam searching space overheads by 76.56% for top-15 beams with respect to standard defined approach. 

**Abstract (ZH)**: 毫米波通信中多模态感知与融合学习框架用于减少波束训练开销 

---
# Length-Aware Rotary Position Embedding for Text-Speech Alignment 

**Title (ZH)**: 基于长度感知的旋转位置嵌入文本-语音对齐 

**Authors**: Hyeongju Kim, Juheon Lee, Jinhyeok Yang, Jacob Morton  

**Link**: [PDF](https://arxiv.org/pdf/2509.11084)  

**Abstract**: Many recent text-to-speech (TTS) systems are built on transformer architectures and employ cross-attention mechanisms for text-speech alignment. Within these systems, rotary position embedding (RoPE) is commonly used to encode positional information in text and speech representations. In this work, we introduce length-aware RoPE (LARoPE), a simple yet effective extension of RoPE that improves text-speech alignment. Unlike RoPE, which relies on absolute indices, LARoPE computes relative distances between query and key positions using length-normalized indices. Experimental results show that LARoPE consistently outperforms RoPE, offering faster loss convergence, more accurate text-speech alignment, and higher overall TTS quality. Furthermore, LARoPE demonstrates greater resilience to variations in utterance duration and maintains stable performance in extended speech generation up to 30 seconds, whereas RoPE suffers from notable degradation. Notably, our method achieves a state-of-the-art word error rate on a standard zero-shot TTS benchmark. 

**Abstract (ZH)**: 基于长度感知的旋转位置嵌入在文本到语音合成中的应用 

---
# Membership Inference Attacks on Recommender System: A Survey 

**Title (ZH)**: 推荐系统中成员推理攻击：一篇综述 

**Authors**: Jiajie He, Yuechun Gu, Keke Chen, Xintong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11080)  

**Abstract**: Recommender systems (RecSys) have been widely applied to various applications, including E-commerce, finance, healthcare, social media and have become increasingly influential in shaping user behavior and decision-making, highlighting their growing impact in various domains. However, recent studies have shown that RecSys are vulnerable to membership inference attacks (MIAs), which aim to infer whether user interaction record was used to train a target model or not. MIAs on RecSys models can directly lead to a privacy breach. For example, via identifying the fact that a purchase record that has been used to train a RecSys associated with a specific user, an attacker can infer that user's special quirks. In recent years, MIAs have been shown to be effective on other ML tasks, e.g., classification models and natural language processing. However, traditional MIAs are ill-suited for RecSys due to the unseen posterior probability. Although MIAs on RecSys form a newly emerging and rapidly growing research area, there has been no systematic survey on this topic yet. In this article, we conduct the first comprehensive survey on RecSys MIAs. This survey offers a comprehensive review of the latest advancements in RecSys MIAs, exploring the design principles, challenges, attack and defense associated with this emerging field. We provide a unified taxonomy that categorizes different RecSys MIAs based on their characterizations and discuss their pros and cons. Based on the limitations and gaps identified in this survey, we point out several promising future research directions to inspire the researchers who wish to follow this area. This survey not only serves as a reference for the research community but also provides a clear description for researchers outside this research domain. 

**Abstract (ZH)**: 推荐系统（RecSys）已在电子商务、金融、医疗保健、社交媒体等领域广泛应用，并在塑造用户行为和决策方面越来越具有影响力，凸显了其在各个领域中的日益增长的影响。然而，近期研究显示，推荐系统（RecSys）容易受到会员推理攻击（Member Inference Attacks, MIAs）的影响，这种攻击试图推断用户交互记录是否被用于训练目标模型。MIAs会对RecSys模型直接导致隐私泄露。例如，通过识别某个用于训练与特定用户相关的推荐系统的购买记录，攻击者可以推断出用户的一些特殊偏好。近年来，MIAs已经被证明在其他机器学习任务中，如分类模型和自然语言处理任务中是有效的。然而，传统的MIAs并不适合RecSys，因为它们面临着未见后验概率的问题。尽管MIAs在RecSys领域的研究正在成为一个新兴且迅速增长的研究领域，但迄今为止尚未对此进行系统性的综述。在这篇文章中，我们首次对RecSys MIAs进行了全面的综述。本文综述涵盖了RecSys MIAs的最新进展，探讨了这一新兴领域的设计原则、挑战、攻击与防御。我们提供了一个统一的分类体系，基于特征对不同类型的RecSys MIAs进行分类，并讨论了它们各自的优缺点。基于综述中指出的局限性和空白，指出了几个值得进一步研究的方向，以激发对该领域感兴趣的学者。本文不仅为研究界提供了参考，也为该研究领域的外部研究人员提供了清晰的描述。 

---
# FragmentGPT: A Unified GPT Model for Fragment Growing, Linking, and Merging in Molecular Design 

**Title (ZH)**: FragmentGPT：用于分子设计中的片段生长、连接和合并的统一GPT模型 

**Authors**: Xuefeng Liu, Songhao Jiang, Qinan Huang, Tinson Xu, Ian Foster, Mengdi Wang, Hening Lin, Jinbo Xu, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2509.11044)  

**Abstract**: Fragment-Based Drug Discovery (FBDD) is a popular approach in early drug development, but designing effective linkers to combine disconnected molecular fragments into chemically and pharmacologically viable candidates remains challenging. Further complexity arises when fragments contain structural redundancies, like duplicate rings, which cannot be addressed by simply adding or removing atoms or bonds. To address these challenges in a unified framework, we introduce FragmentGPT, which integrates two core components: (1) a novel chemically-aware, energy-based bond cleavage pre-training strategy that equips the GPT-based model with fragment growing, linking, and merging capabilities, and (2) a novel Reward Ranked Alignment with Expert Exploration (RAE) algorithm that combines expert imitation learning for diversity enhancement, data selection and augmentation for Pareto and composite score optimality, and Supervised Fine-Tuning (SFT) to align the learner policy with multi-objective goals. Conditioned on fragment pairs, FragmentGPT generates linkers that connect diverse molecular subunits while simultaneously optimizing for multiple pharmaceutical goals. It also learns to resolve structural redundancies-such as duplicated fragments-through intelligent merging, enabling the synthesis of optimized molecules. FragmentGPT facilitates controlled, goal-driven molecular assembly. Experiments and ablation studies on real-world cancer datasets demonstrate its ability to generate chemically valid, high-quality molecules tailored for downstream drug discovery tasks. 

**Abstract (ZH)**: 基于片段的药物发现（FBDD）是早期药物开发中的一个流行方法，但设计有效的连接剂以将不相连的分子片段组合成化学上和药理上可行的候选药物仍然是一个挑战。当片段包含结构冗余，如重复环时，仅通过添加或移除原子或键无法解决问题，这进一步增加了复杂性。为了解决这些挑战，我们引入了FragmentGPT，它包含两个核心组件：（1）一种新的化学感知的能量基础键断裂预训练策略，使基于GPT的模型具备片段生长、连接和合并的能力；（2）一种新的基于专家探索的奖励排名对齐（RAE）算法，该算法结合了专家模拟学习以增强多样性，数据选择和扩充以优化帕累托和复合评分，以及监督微调（SFT）以使学习策略与多目标目标相一致。基于片段对，FragmentGPT可以同时生成连接不同分子亚单位的连接剂并优化多个药理目标。它还学会了通过智能合并解决结构冗余，如重复片段，以促进优化分子的合成。FragmentGPT促进了受控的目标驱动分子组装。在实际癌症数据集上的实验和消融研究证明了其生成化学上有效、高质量分子以适应下游药物发现任务的能力。 

---
# Hardness, Structural Knowledge, and Opportunity: An Analytical Framework for Modular Performance Modeling 

**Title (ZH)**: 硬度、结构知识与机遇：模块化性能建模的分析框架 

**Authors**: Omid Gheibi, Christian Kästner, Pooyan Jamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11000)  

**Abstract**: Performance-influence models are beneficial for understanding how configurations affect system performance, but their creation is challenging due to the exponential growth of configuration spaces. While gray-box approaches leverage selective "structural knowledge" (like the module execution graph of the system) to improve modeling, the relationship between this knowledge, a system's characteristics (we call them "structural aspects"), and potential model improvements is not well understood. This paper addresses this gap by formally investigating how variations in structural aspects (e.g., the number of modules and options per module) and the level of structural knowledge impact the creation of "opportunities" for improved "modular performance modeling". We introduce and quantify the concept of modeling "hardness", defined as the inherent difficulty of performance modeling. Through controlled experiments with synthetic system models, we establish an "analytical matrix" to measure these concepts. Our findings show that modeling hardness is primarily driven by the number of modules and configuration options per module. More importantly, we demonstrate that both higher levels of structural knowledge and increased modeling hardness significantly enhance the opportunity for improvement. The impact of these factors varies by performance metric; for ranking accuracy (e.g., in debugging task), structural knowledge is more dominant, while for prediction accuracy (e.g., in resource management task), hardness plays a stronger role. These results provide actionable insights for system designers, guiding them to strategically allocate time and select appropriate modeling approaches based on a system's characteristics and a given task's objectives. 

**Abstract (ZH)**: 性能影响模型有助于理解配置如何影响系统性能，但其创建因配置空间的指数级增长而具有挑战性。虽然灰盒方法利用选择性的“结构知识”（如系统的模块执行图）来提高建模效果，但这种知识与系统特征（我们称之为“结构方面”）以及潜在建模改进之间的关系尚不明确。本文通过正式研究结构方面变化（例如模块数量和每个模块的选项数量）和结构知识水平如何影响“模块性能建模”的“机会”创造，来填补这一空白。我们引入并量化了“建模硬度”这一概念，定义为固有的性能建模难度。通过合成系统模型的受控实验，我们建立了一个“分析矩阵”来衡量这些概念。我们的研究结果表明，建模硬度主要由模块数量和每个模块的配置选项数量驱动。更重要的是，我们展示了更高的结构知识水平和增加的建模硬度能够显著提升改进的机会。这些因素对性能度量的影响有所不同；例如，在调试任务的排序准确度中，结构知识占据主导地位，而在资源管理任务的预测准确度中，硬度发挥更重要作用。这些结果为系统设计师提供了可操作的洞察，指导他们根据系统的特征和给定任务的目标，战略性地分配时间和选择合适的建模方法。 

---
# Factor Graph Optimization for Leak Localization in Water Distribution Networks 

**Title (ZH)**: 水分布网络泄漏定位的因子图优化方法 

**Authors**: Paul Irofti, Luis Romero-Ben, Florin Stoican, Vicenç Puig  

**Link**: [PDF](https://arxiv.org/pdf/2509.10982)  

**Abstract**: Detecting and localizing leaks in water distribution network systems is an important topic with direct environmental, economic, and social impact. Our paper is the first to explore the use of factor graph optimization techniques for leak localization in water distribution networks, enabling us to perform sensor fusion between pressure and demand sensor readings and to estimate the network's temporal and structural state evolution across all network nodes. The methodology introduces specific water network factors and proposes a new architecture composed of two factor graphs: a leak-free state estimation factor graph and a leak localization factor graph. When a new sensor reading is obtained, unlike Kalman and other interpolation-based methods, which estimate only the current network state, factor graphs update both current and past states. Results on Modena, L-TOWN and synthetic networks show that factor graphs are much faster than nonlinear Kalman-based alternatives such as the UKF, while also providing improvements in localization compared to state-of-the-art estimation-localization approaches. Implementation and benchmarks are available at this https URL. 

**Abstract (ZH)**: 基于因子图优化技术的供水网络漏损检测与定位研究 

---
# Decoupling Search and Learning in Neural Net Training 

**Title (ZH)**: 在神经网络训练中解耦搜索和学习 

**Authors**: Akshay Vegesna, Samip Dahal  

**Link**: [PDF](https://arxiv.org/pdf/2509.10973)  

**Abstract**: Gradient descent typically converges to a single minimum of the training loss without mechanisms to explore alternative minima that may generalize better. Searching for diverse minima directly in high-dimensional parameter space is generally intractable. To address this, we propose a framework that performs training in two distinct phases: search in a tractable representation space (the space of intermediate activations) to find diverse representational solutions, and gradient-based learning in parameter space by regressing to those searched representations. Through evolutionary search, we discover representational solutions whose fitness and diversity scale with compute--larger populations and more generations produce better and more varied solutions. These representations prove to be learnable: networks trained by regressing to searched representations approach SGD's performance on MNIST, CIFAR-10, and CIFAR-100. Performance improves with search compute up to saturation. The resulting models differ qualitatively from networks trained with gradient descent, following different representational trajectories during training. This work demonstrates how future training algorithms could overcome gradient descent's exploratory limitations by decoupling search in representation space from efficient gradient-based learning in parameter space. 

**Abstract (ZH)**: 基于演化搜索的多样最优解训练框架：超越梯度下降的探索局限 

---
# PHLoRA: data-free Post-hoc Low-Rank Adapter extraction from full-rank checkpoint 

**Title (ZH)**: PHLoRA：从全秩检查点无数据提取低秩适配器的后处理方法 

**Authors**: Bhoomit Vasani, Jack FitzGerald, Anjie Fang, Sushmit Vaish  

**Link**: [PDF](https://arxiv.org/pdf/2509.10971)  

**Abstract**: We introduce PHLoRA (Pronounced "flora"). (Post-hoc LoRA), a simple yet powerful method to extract low-rank adaptation adapters from full-rank fine-tuned models without requiring access to training data or gradients. By computing the low-rank decomposition of weight differences between a base model and its fine-tuned counterpart, our method reconstructs adapter modules that can be merged or dynamically routed at inference time via S-LoRA, or served in scalable, industry settings using platforms like NVIDIA NIM. This approach amortizes latency overhead across requests and yields substantial cost savings. Unlike prior work that trains each adapter explicitly, our approach decouples fine-tuning from adapter generation, allowing adapter extraction from existing full-rank models or third-party checkpoints. Experiments on text, image, and video benchmarks using the Amazon Nova model family demonstrate that extracted adapters preserve high energy from the full weight delta, can be pruned safely, and yield negligible degradation in downstream task performance when re-merged. Overall, PHLoRA provides a practical path for making all existing full-rank checkpoints adapter-ready, democratizing scalable inference for all models. 

**Abstract (ZH)**: PHLoRA (Pronounced "flora"). (事后LoRA): 一种无需训练数据或梯度即可从全秩微调模型中提取低秩适应适配器的简单而强大的方法 

---
# Clarifying Model Transparency: Interpretability versus Explainability in Deep Learning with MNIST and IMDB Examples 

**Title (ZH)**: 澄清模型透明度：基于MNIST和IMDB示例的深度学习可解释性与可阐释性的区分 

**Authors**: Mitali Raj  

**Link**: [PDF](https://arxiv.org/pdf/2509.10929)  

**Abstract**: The impressive capabilities of deep learning models are often counterbalanced by their inherent opacity, commonly termed the "black box" problem, which impedes their widespread acceptance in high-trust domains. In response, the intersecting disciplines of interpretability and explainability, collectively falling under the Explainable AI (XAI) umbrella, have become focal points of research. Although these terms are frequently used as synonyms, they carry distinct conceptual weights. This document offers a comparative exploration of interpretability and explainability within the deep learning paradigm, carefully outlining their respective definitions, objectives, prevalent methodologies, and inherent difficulties. Through illustrative examinations of the MNIST digit classification task and IMDB sentiment analysis, we substantiate a key argument: interpretability generally pertains to a model's inherent capacity for human comprehension of its operational mechanisms (global understanding), whereas explainability is more commonly associated with post-hoc techniques designed to illuminate the basis for a model's individual predictions or behaviors (local explanations). For example, feature attribution methods can reveal why a specific MNIST image is recognized as a '7', and word-level importance can clarify an IMDB sentiment outcome. However, these local insights do not render the complex underlying model globally transparent. A clear grasp of this differentiation, as demonstrated by these standard datasets, is vital for fostering dependable and sound artificial intelligence. 

**Abstract (ZH)**: 深度学习模型的强大能力往往被其固有的不透明性所抵消，这一问题通常被称为“黑箱”问题，阻碍了它们在高可信度领域的广泛应用。为应对这一挑战，可解释性和可解析性这两个相关学科共同构成了可解释人工智能（XAI）的研究焦点。尽管这两个术语经常被当作同义词使用，但它们具有不同的概念内涵。本文在深度学习范式下对可解释性和可解析性进行比较探讨，详细阐述了它们各自的定义、目标、常用方法以及固有困难。通过MNIST数字分类任务和IMDB情感分析的实例分析，我们证明了一个关键论点：可解释性通常涉及模型本身对人类理解其操作机制的能力（全局理解），而可解析性则更常与旨在阐明模型个体预测或行为基础的后续技术相关联（局部解释）。例如，特征归因方法可以揭示为什么特定的MNIST图像被识别为“7”，而词级重要性可以澄清IMDB情感分析的结果。然而，这些局部洞察并不能使复杂的底层模型变得全局透明。明确区分这些概念，对于促进可靠的和稳健的人工智能而言至关重要。 

---
# Optimal message passing for molecular prediction is simple, attentive and spatial 

**Title (ZH)**: 分子预测的最佳消息传递简单、注意且空间化 

**Authors**: Alma C. Castaneda-Leautaud, Rommie E. Amaro  

**Link**: [PDF](https://arxiv.org/pdf/2509.10871)  

**Abstract**: Strategies to improve the predicting performance of Message-Passing Neural-Networks for molecular property predictions can be achieved by simplifying how the message is passed and by using descriptors that capture multiple aspects of molecular graphs. In this work, we designed model architectures that achieved state-of-the-art performance, surpassing more complex models such as those pre-trained on external databases. We assessed dataset diversity to complement our performance results, finding that structural diversity influences the need for additional components in our MPNNs and feature sets.
In most datasets, our best architecture employs bidirectional message-passing with an attention mechanism, applied to a minimalist message formulation that excludes self-perception, highlighting that relatively simpler models, compared to classical MPNNs, yield higher class separability. In contrast, we found that convolution normalization factors do not benefit the predictive power in all the datasets tested. This was corroborated in both global and node-level outputs. Additionally, we analyzed the influence of both adding spatial features and working with 3D graphs, finding that 2D molecular graphs are sufficient when complemented with appropriately chosen 3D descriptors. This approach not only preserves predictive performance but also reduces computational cost by over 50%, making it particularly advantageous for high-throughput screening campaigns. 

**Abstract (ZH)**: 基于消息传递神经网络的分子性质预测性能提升策略可以通过简化消息传递方式和使用多方面捕捉分子图描述符来实现。在本工作中，我们设计了达到迄今最佳性能的模型架构，超越了预训练于外部数据库的更为复杂的模型。我们评估了数据集多样性以补充性能结果，发现结构多样性影响了我们在MPNN中需要的额外组件和特征集的需求。

在大多数数据集中，我们最佳的架构采用双向消息传递并结合注意力机制，应用于排除自我感知的简约消息形式，突显相对简单的模型相较于经典MPNN具有更高的类别可分辨性。相比之下，我们发现卷积规范化因子对所有测试数据集的预测能力并无普遍益处。这一发现被全局和节点级输出结果所验证。此外，我们分析了增加空间特征和处理三维图形的影响，发现当辅以合适选择的三维描述符时，二维分子图足以满足需求。该方法不仅保持了预测性能，还通过超过50%的计算成本降低，使其特别适用于高通量筛选campaign。 

---
# GTHNA: Local-global Graph Transformer with Memory Reconstruction for Holistic Node Anomaly Evaluation 

**Title (ZH)**: GTHNA：基于记忆重构的局部-全局图变换器整体节点异常评估 

**Authors**: Mingkang Li, Xuexiong Luo, Yue Zhang, Yaoyang Li, Fu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.10869)  

**Abstract**: Anomaly detection in graph-structured data is an inherently challenging problem, as it requires the identification of rare nodes that deviate from the majority in both their structural and behavioral characteristics. Existing methods, such as those based on graph convolutional networks (GCNs), often suffer from over-smoothing, which causes the learned node representations to become indistinguishable. Furthermore, graph reconstruction-based approaches are vulnerable to anomalous node interference during the reconstruction process, leading to inaccurate anomaly detection. In this work, we propose a novel and holistic anomaly evaluation framework that integrates three key components: a local-global Transformer encoder, a memory-guided reconstruction mechanism, and a multi-scale representation matching strategy. These components work synergistically to enhance the model's ability to capture both local and global structural dependencies, suppress the influence of anomalous nodes, and assess anomalies from multiple levels of granularity. Anomaly scores are computed by combining reconstruction errors and memory matching signals, resulting in a more robust evaluation. Extensive experiments on seven benchmark datasets demonstrate that our method outperforms existing state-of-the-art approaches, offering a comprehensive and generalizable solution for anomaly detection across various graph domains. 

**Abstract (ZH)**: 图结构数据中的异常检测是一个固有的挑战问题，因为它要求识别在结构和行为特征上与大多数节点显著不同的稀有节点。现有方法，如基于图卷积网络（GCNs）的方法，往往遭受过度平滑的困扰，导致学习到的节点表示变得难以区分。此外，基于图重建的方法在重建过程中容易受到异常节点干扰的影响，导致异常检测不准确。在本工作中，我们提出了一种新颖的综合性异常评估框架，该框架整合了三个关键组件：局部-全局Transformer编码器、记忆引导的重建机制和多层次表示匹配策略。这些组件协同工作，增强模型捕获局部和全局结构依赖性、抑制异常节点影响以及从多个粒度级别评估异常的能力。异常得分通过结合重建误差和记忆匹配信号来计算，从而实现更稳健的评估。在七个基准数据集上的广泛实验表明，我们的方法优于现有最先进的方法，提供了一种适用于各种图域的全面且可泛化的异常检测解决方案。 

---
# Physics-informed neural network solves minimal surfaces in curved spacetime 

**Title (ZH)**: 基于物理的神经网络求解弯曲时空中的极小曲面 

**Authors**: Koji Hashimoto, Koichi Kyo, Masaki Murata, Gakuto Ogiwara, Norihiro Tanahashi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10866)  

**Abstract**: We develop a flexible framework based on physics-informed neural networks (PINNs) for solving boundary value problems involving minimal surfaces in curved spacetimes, with a particular emphasis on singularities and moving boundaries. By encoding the underlying physical laws into the loss function and designing network architectures that incorporate the singular behavior and dynamic boundaries, our approach enables robust and accurate solutions to both ordinary and partial differential equations with complex boundary conditions. We demonstrate the versatility of this framework through applications to minimal surface problems in anti-de Sitter (AdS) spacetime, including examples relevant to the AdS/CFT correspondence (e.g. Wilson loops and gluon scattering amplitudes) popularly used in the context of string theory in theoretical physics. Our methods efficiently handle singularities at boundaries, and also support both "soft" (loss-based) and "hard" (formulation-based) imposition of boundary conditions, including cases where the position of a boundary is promoted to a trainable parameter. The techniques developed here are not limited to high-energy theoretical physics but are broadly applicable to boundary value problems encountered in mathematics, engineering, and the natural sciences, wherever singularities and moving boundaries play a critical role. 

**Abstract (ZH)**: 基于物理知情神经网络的柔性框架：在弯曲时空中的最小曲面边界值问题，特别重视奇点和移动边界 

---
# Pre-Storage Reasoning for Episodic Memory: Shifting Inference Burden to Memory for Personalized Dialogue 

**Title (ZH)**: 预存储推理对情景记忆：将推理负担转移至记忆以实现个性化对话 

**Authors**: Sangyeop Kim, Yohan Lee, Sanghwa Kim, Hyunjong Kim, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.10852)  

**Abstract**: Effective long-term memory in conversational AI requires synthesizing information across multiple sessions. However, current systems place excessive reasoning burden on response generation, making performance significantly dependent on model sizes. We introduce PREMem (Pre-storage Reasoning for Episodic Memory), a novel approach that shifts complex reasoning processes from inference to memory construction. PREMem extracts fine-grained memory fragments categorized into factual, experiential, and subjective information; it then establishes explicit relationships between memory items across sessions, capturing evolution patterns like extensions, transformations, and implications. By performing this reasoning during pre-storage rather than when generating a response, PREMem creates enriched representations while reducing computational demands during interactions. Experiments show significant performance improvements across all model sizes, with smaller models achieving results comparable to much larger baselines while maintaining effectiveness even with constrained token budgets. Code and dataset are available at this https URL. 

**Abstract (ZH)**: Effective Long-term Memory in Conversational AI Requires Synthesizing Information Across Multiple Sessions: Shifting Complex Reasoning Processes from Inference to Memory Construction with PREMem 

---
# A funny companion: Distinct neural responses to perceived AI- versus humangenerated humor 

**Title (ZH)**: 有趣的伴侣：感知AI生成与人类生成幽默的神经响应差异 

**Authors**: Xiaohui Rao, Hanlin Wu, Zhenguang G. Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.10847)  

**Abstract**: As AI companions become capable of human-like communication, including telling jokes, understanding how people cognitively and emotionally respond to AI humor becomes increasingly important. This study used electroencephalography (EEG) to compare how people process humor from AI versus human sources. Behavioral analysis revealed that participants rated AI and human humor as comparably funny. However, neurophysiological data showed that AI humor elicited a smaller N400 effect, suggesting reduced cognitive effort during the processing of incongruity. This was accompanied by a larger Late Positive Potential (LPP), indicating a greater degree of surprise and emotional response. This enhanced LPP likely stems from the violation of low initial expectations regarding AI's comedic capabilities. Furthermore, a key temporal dynamic emerged: human humor showed habituation effects, marked by an increasing N400 and a decreasing LPP over time. In contrast, AI humor demonstrated increasing processing efficiency and emotional reward, with a decreasing N400 and an increasing LPP. This trajectory reveals how the brain can dynamically update its predictive model of AI capabilities. This process of cumulative reinforcement challenges "algorithm aversion" in humor, as it demonstrates how cognitive adaptation to AI's language patterns can lead to an intensified emotional reward. Additionally, participants' social attitudes toward AI modulated these neural responses, with higher perceived AI trustworthiness correlating with enhanced emotional engagement. These findings indicate that the brain responds to AI humor with surprisingly positive and intense reactions, highlighting humor's potential for fostering genuine engagement in human-AI social interaction. 

**Abstract (ZH)**: 随着AI伴侣具备人类般的交流能力，包括讲笑话，理解人们在认知和情感上对AI幽默的反应变得越来越重要。本研究使用脑电图（EEG）比较了人们处理来自AI和人类来源的幽默的方式。行为分析显示，参与者认为AI和人类的幽默一样有趣。然而，神经生理数据表明，AI幽默引起的N400效应较小，表明在处理不协调性时所需的认知努力减少。这伴随着较大的晚正弦波电位（LPP），表明了更大的惊讶和情感反应。这种增强的LPP很可能源于对AI喜剧能力低预期的违反。此外，一个关键的时间动态出现了：人类幽默表现出习惯化效应，其特征是N400逐渐增加，LPP逐渐减少。相反，AI幽默显示出不断增加的处理效率和情感奖励，N400逐渐减少，LPP逐渐增加。这一轨迹揭示了大脑如何动态更新其对AI能力的预测模型。这一累积强化过程挑战了“算法厌恶”在幽默中的观点，因为它展示了认知适应AI语言模式如何导致更加强烈的情感奖励。此外，参与者对AI的社会态度调节了这些神经反应，更高的AI可信度感知与增强的情感参与相关。这些发现表明，大脑对AI幽默的反应是非比寻常的积极和强烈，突显了幽默在促进人机社会互动中真实参与的潜力。 

---
# FACTORS: Factorial Approximation for Complementary Two-factor Optimization with Risk-aware Scoring 

**Title (ZH)**: 因子分析：互补双因子优化的损益感知评分因子近似方法 

**Authors**: Dongseok Kim, Wonjun Jeong, Gisung Oh  

**Link**: [PDF](https://arxiv.org/pdf/2509.10825)  

**Abstract**: We propose FACTORS, a framework that combines design of experiments with Shapley decomposition to address performance and stability issues that are sensitive to combinations of training factors. Our approach consistently estimates main effects and two-factor interactions, then integrates them into a risk-adjusted objective function that jointly accounts for uncertainty and cost, enabling reliable selection of configurations under a fixed budget. Effect estimation is implemented through two complementary paths: a plug-in path based on conditional means, and a least-squares path that reconstructs Shapley contributions from samples. These paths are designed to work complementarily even when design density and bias levels differ. By incorporating standardization of estimates, bias correction, and uncertainty quantification, our procedure ensures comparability across heterogeneous factor spaces and designs, while a lightweight search routine yields configurations within practical time even for large factor spaces. On the theoretical side, we provide error decompositions, sample complexity analysis, and upper bounds on optimality gaps. On the interpretive side, we summarize main effects and interactions in map form, highlighting adjustment priorities and safe improvement pathways. Across diverse datasets and design conditions, our approach improves rank preservation and optimal configuration identification, reduces decision-making risks, and offers a tuning foundation that delivers interpretable justification alongside stable performance gains even under budget constraints. 

**Abstract (ZH)**: 我们提出FACTORS框架，结合实验设计与Shapley分解，以解决受训练因子组合影响的性能和稳定性问题。该方法一致估计主要效应和两因子交互作用，将它们综合到一个同时考虑不确定性和成本的调整风险目标函数中，从而在固定预算下可靠地选择配置。效应估计通过两条互补路径实现：基于条件均值的插值路径和从样本重建Shapley贡献的最小二乘路径。这些路径即使在设计密度和偏差水平不同的情况下也能互补工作。通过引入估计标准化、偏差校正和不确定性量化，我们的方法确保在异质因子空间和设计之间具有可比性，同时轻量级的搜索程序能够在包含大量因子的空间中在实际时间内获得配置。从理论角度来看，我们提供了误差分解、样本复杂性分析和最优性缺口的上限。从解释角度来看，我们将主要效应和交互作用以地图形式总结，突出调整优先级和安全改进路径。在多样化的数据集和设计条件下，我们的方法改善了排名保真度和最优配置识别，减少了决策风险，并在预算约束下提供了一种可解释的优化基础，伴随稳定的性能提升。 

---
# Rethinking Sparse Autoencoders: Select-and-Project for Fairness and Control from Encoder Features Alone 

**Title (ZH)**: 重新思考稀疏自编码器：仅从编码器特征进行选择和投影以实现公平性和控制 

**Authors**: Antonio Bărbălau, Cristian Daniel Păduraru, Teodor Poncu, Alexandru Tifrea, Elena Burceanu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10809)  

**Abstract**: Sparse Autoencoders (SAEs) have proven valuable due to their ability to provide interpretable and steerable representations. Current debiasing methods based on SAEs manipulate these sparse activations presuming that feature representations are housed within decoder weights. We challenge this fundamental assumption and introduce an encoder-focused alternative for representation debiasing, contributing three key findings: (i) we highlight an unconventional SAE feature selection strategy, (ii) we propose a novel SAE debiasing methodology that orthogonalizes input embeddings against encoder weights, and (iii) we establish a performance-preserving mechanism during debiasing through encoder weight interpolation. Our Selection and Projection framework, termed S\&P TopK, surpasses conventional SAE usage in fairness metrics by a factor of up to 3.2 and advances state-of-the-art test-time VLM debiasing results by a factor of up to 1.8 while maintaining downstream performance. 

**Abstract (ZH)**: Sparse 自编码器 (SAEs) 由于能够提供可解释且可操控的表示而证明了其价值。当前基于 SAEs 的去偏方法通过假定特征表示存储在解码器权重中来操纵这些稀疏激活。我们挑战这一基本假设，并提出了一种以编码器为中心的表示去偏替代方法，贡献了三个关键发现：(i) 强调了一种非传统的 SAE 特征选择策略，(ii) 提出了一种新的 SAE 去偏方法，通过对输入嵌入与编码器权重进行正交化，(iii) 在去偏过程中通过编码器权重插值保持性能。我们的选择和投影框架，称为 S&P TopK，通过公平性指标超越了传统的 SAE 使用，最多提高了 3.2 倍，并在测试时 VLM 去偏方面达到了最先进的结果，最多提高了 1.8 倍，同时保持了下游性能。 

---
# Branched Broomrape Detection in Tomato Farms Using Satellite Imagery and Time-Series Analysis 

**Title (ZH)**: 使用卫星影像和时间序列分析在番茄农场中检测分枝雀麦草 

**Authors**: Mohammadreza Narimani, Alireza Pourreza, Ali Moghimi, Parastoo Farajpoor, Hamid Jafarbiglu, Mohsen Mesgaran  

**Link**: [PDF](https://arxiv.org/pdf/2509.10804)  

**Abstract**: Branched broomrape (Phelipanche ramosa (L.) Pomel) is a chlorophyll-deficient parasitic plant that threatens tomato production by extracting nutrients from the host, with reported yield losses up to 80 percent. Its mostly subterranean life cycle and prolific seed production (more than 200,000 seeds per plant, viable for up to 20 years) make early detection essential. We present an end-to-end pipeline that uses Sentinel-2 imagery and time-series analysis to identify broomrape-infested tomato fields in California. Regions of interest were defined from farmer-reported infestations, and images with less than 10 percent cloud cover were retained. We processed 12 spectral bands and sun-sensor geometry, computed 20 vegetation indices (e.g., NDVI, NDMI), and derived five plant traits (Leaf Area Index, Leaf Chlorophyll Content, Canopy Chlorophyll Content, Fraction of Absorbed Photosynthetically Active Radiation, and Fractional Vegetation Cover) using a neural network calibrated with ground-truth and synthetic data. Trends in Canopy Chlorophyll Content delineated transplanting-to-harvest periods, and phenology was aligned using growing degree days. Vegetation pixels were segmented and used to train a Long Short-Term Memory (LSTM) network on 18,874 pixels across 48 growing-degree-day time points. The model achieved 88 percent training accuracy and 87 percent test accuracy, with precision 0.86, recall 0.92, and F1 0.89. Permutation feature importance ranked NDMI, Canopy Chlorophyll Content, FAPAR, and a chlorophyll red-edge index as most informative, consistent with the physiological effects of infestation. Results show the promise of satellite-driven time-series modeling for scalable detection of parasitic stress in tomato farms. 

**Abstract (ZH)**: 分支雀舌草（Phelipanche ramosa (L.) Pomel）是一种缺乏叶绿素的寄生植物，通过从宿主体内吸取养分威胁番茄生产，据报道可造成高达80%的产量损失。其主要地下生活史和大量的种子生产（每株植物超过200,000颗种子，可存活长达20年）使其早期检测至关重要。我们提出了一种端到端的工作流程，利用Sentinel-2遥感图像和时间序列分析来识别加利福尼亚受雀舌草寄生的番茄田地。通过农民报告的寄生区划定了感兴趣区域，并保留了云覆盖少于10%的图像。我们处理了12个光谱波段和太阳传感器几何形状，计算了20个植被指数（例如NDVI、NDMI），并通过神经网络利用地面实况和合成数据计算了五种植物特征（叶面积指数、叶叶绿素含量、冠层叶绿素含量、光合有效辐射吸收比例和植被覆盖度）特征。冠层叶绿素含量的趋势划定了移栽到收获的时期，并通过生长度日对植物学阶段进行了对齐。植被像素被分割，并用于在48个生长度日时间点上训练长短期记忆（LSTM）网络，共有18,874个像素。该模型实现了88%的训练准确率和87%的测试准确率，精确度为0.86，召回率为0.92，F1分为0.89。重要性排列特征显示，NDMI、冠层叶绿素含量、FAPAR和叶绿素红边指数是最重要的特征，与寄生的生理影响一致。结果表明，基于卫星的数据驱动时间序列建模在番茄农场中 scalable 检测寄生胁迫具有潜力。 

---
# Contextual Budget Bandit for Food Rescue Volunteer Engagement 

**Title (ZH)**: 基于上下文的预算多臂老虎机模型：应用于食品救援志愿者参与 

**Authors**: Ariana Tang, Naveen Raman, Fei Fang, Zheyuan Ryan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10777)  

**Abstract**: Volunteer-based food rescue platforms tackle food waste by matching surplus food to communities in need. These platforms face the dual problem of maintaining volunteer engagement and maximizing the food rescued. Existing algorithms to improve volunteer engagement exacerbate geographical disparities, leaving some communities systematically disadvantaged. We address this issue by proposing Contextual Budget Bandit. Contextual Budget Bandit incorporates context-dependent budget allocation in restless multi-armed bandits, a model of decision-making which allows for stateful arms. By doing so, we can allocate higher budgets to communities with lower match rates, thereby alleviating geographical disparities. To tackle this problem, we develop an empirically fast heuristic algorithm. Because the heuristic algorithm can achieve a poor approximation when active volunteers are scarce, we design the Mitosis algorithm, which is guaranteed to compute the optimal budget allocation. Empirically, we demonstrate that our algorithms outperform baselines on both synthetic and real-world food rescue datasets, and show how our algorithm achieves geographical fairness in food rescue. 

**Abstract (ZH)**: 基于志愿者的食品救援平台通过匹配过剩食品与有需求的社区来解决食品浪费问题。这些平台面临维持志愿者参与和最大化救援食品的双重挑战。现有的提高志愿者参与度的算法加剧了地理层面的不平等，使一些社区处于系统性劣势。我们通过提出上下文预算-bandit（Contextual Budget Bandit）来解决这一问题。上下文预算-bandit 在活跃臂可以保持状态的无奈多臂老虎机模型中融入了情境相关的预算分配，这样可以将更高预算分配给匹配率较低的社区，从而缓解地理不平等。为解决这一问题，我们开发了一种经验上快速的启发式算法。由于在活跃志愿者稀缺时启发式算法可能会产生较差的近似结果，我们设计了保证能够计算最优预算分配的Mitosis算法。实证研究表明，我们的算法在合成和真实食品救援数据集上均优于基线算法，并展示了我们的算法如何在食品救援中实现地理公平性。 

---
# A Content-dependent Watermark for Safeguarding Image Attribution 

**Title (ZH)**: 基于内容的水印用于保护图像 Attribution 

**Authors**: Tong Zhou, Ruyi Ding, Gaowen Liu, Charles Fleming, Ramana Rao Kompella, Yunsi Fei, Xiaolin Xu, Shaolei Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.10766)  

**Abstract**: The rapid growth of digital and AI-generated images has amplified the need for secure and verifiable methods of image attribution. While digital watermarking offers more robust protection than metadata-based approaches--which can be easily stripped--current watermarking techniques remain vulnerable to forgery, creating risks of misattribution that can damage the reputations of AI model developers and the rights of digital artists. These vulnerabilities arise from two key issues: (1) content-agnostic watermarks, which, once learned or leaked, can be transferred across images to fake attribution, and (2) reliance on detector-based verification, which is unreliable since detectors can be tricked. We present MetaSeal, a novel framework for content-dependent watermarking with cryptographic security guarantees to safeguard image attribution. Our design provides (1) forgery resistance, preventing unauthorized replication and enforcing cryptographic verification; (2) robust, self-contained protection, embedding attribution directly into images while maintaining resilience against benign transformations; and (3) evidence of tampering, making malicious alterations visually detectable. Experiments demonstrate that MetaSeal effectively mitigates forgery attempts and applies to both natural and AI-generated images, establishing a new standard for secure image attribution. 

**Abstract (ZH)**: 数字和AI生成图像的迅速增长加大了对安全可验证的图像归属方法的需求。现有的水印技术虽比基于元数据的方法提供了更 robust 的保护，但也存在伪造风险，这可能导致误归属，损害AI模型开发者和数字艺术家的声誉。这些漏洞源于两个核心问题：(1) 内容无关的水印一旦被学习或泄露，可以跨图像转移伪造归属，(2) 依赖检测器验证，但由于检测器可被欺骗，因此不可靠。我们提出了一种名为MetaSeal的新型框架，提供基于内容的水印技术和 cryptographic 安全保证以保护图像归属。该设计包括：(1) 抵抗伪造，防止未经授权复制并强制执行 cryptographic 验证；(2) 强大且自包含的保护，直接将归属嵌入图像中，同时保持对良性变换的鲁棒性；以及(3) 恶意篡改的视觉证据，使恶意修改可被视觉检测到。实验表明，MetaSeal 有效地抵御了伪造尝试，并适用于自然和AI生成的图像，从而确立了安全图像归属的新标准。 

---
# Kalman Bayesian Transformer 

**Title (ZH)**: Kalman-Bayesian Transformer 

**Authors**: Haoming Jing, Oren Wright, José M. F. Moura, Yorie Nakahira  

**Link**: [PDF](https://arxiv.org/pdf/2509.10695)  

**Abstract**: Sequential fine-tuning of transformers is useful when new data arrive sequentially, especially with shifting distributions. Unlike batch learning, sequential learning demands that training be stabilized despite a small amount of data by balancing new information and previously learned knowledge in the pre-trained models. This challenge is further complicated when training is to be completed in latency-critical environments and learning must additionally quantify and be mediated by uncertainty. Motivated by these challenges, we propose a novel method that frames sequential fine-tuning as a posterior inference problem within a Bayesian framework. Our approach integrates closed-form moment propagation of random variables, Kalman Bayesian Neural Networks, and Taylor approximations of the moments of softmax functions. By explicitly accounting for pre-trained models as priors and adaptively balancing them against new information based on quantified uncertainty, our method achieves robust and data-efficient sequential learning. The effectiveness of our method is demonstrated through numerical simulations involving sequential adaptation of a decision transformer to tasks characterized by distribution shifts and limited memory resources. 

**Abstract (ZH)**: 基于贝叶斯框架的闭形式矩传播和卡尔曼贝叶斯神经网络的序贯微调新方法 

---
# Learning Concave Bid Shading Strategies in Online Auctions via Measure-valued Proximal Optimization 

**Title (ZH)**: 通过测度值 proximal 优化学习在线拍卖中的凹投标阴影策略 

**Authors**: Iman Nodozi, Djordje Gligorijevic, Abhishek Halder  

**Link**: [PDF](https://arxiv.org/pdf/2509.10693)  

**Abstract**: This work proposes a bid shading strategy for first-price auctions as a measure-valued optimization problem. We consider a standard parametric form for bid shading and formulate the problem as convex optimization over the joint distribution of shading parameters. After each auction, the shading parameter distribution is adapted via a regularized Wasserstein-proximal update with a data-driven energy functional. This energy functional is conditional on the context, i.e., on publisher/user attributes such as domain, ad slot type, device, or location. The proposed algorithm encourages the bid distribution to place more weight on values with higher expected surplus, i.e., where the win probability and the value gap are both large. We show that the resulting measure-valued convex optimization problem admits a closed form solution. A numerical example illustrates the proposed method. 

**Abstract (ZH)**: 本研究提出了一种针对首价拍卖的出价遮蔽策略，将其形式化为测度值优化问题。我们考虑标准参数形式的出价遮蔽，并将问题形式化为联合分布下遮蔽参数的凸优化问题。每次拍卖后，通过带有数据驱动能量函数的正则化 Wasserstein 近邻更新来调整遮蔽参数分布。该能量函数基于上下文，如发布者/用户属性（如领域、广告槽类型、设备或位置）。所提出的算法鼓励出价分布将更多权重分配给预期剩余价值较高的值，即赢标概率和价值差距都较大的地方。我们证明了由此产生的测度值凸优化问题具有闭式解。数值例子阐述了所提出的方法。 

---
# Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy 

**Title (ZH)**: 基于可解释自适应差分隐私的隐私保护去中心化联邦学习 

**Authors**: Fardin Jalil Piran, Zhiling Chen, Yang Zhang, Qianyu Zhou, Jiong Tang, Farhad Imani  

**Link**: [PDF](https://arxiv.org/pdf/2509.10691)  

**Abstract**: Decentralized federated learning faces privacy risks because model updates can leak data through inference attacks and membership inference, a concern that grows over many client exchanges. Differential privacy offers principled protection by injecting calibrated noise so confidential information remains secure on resource-limited IoT devices. Yet without transparency, black-box training cannot track noise already injected by previous clients and rounds, which forces worst-case additions and harms accuracy. We propose PrivateDFL, an explainable framework that joins hyperdimensional computing with differential privacy and keeps an auditable account of cumulative noise so each client adds only the difference between the required noise and what has already been accumulated. We evaluate on MNIST, ISOLET, and UCI-HAR to span image, signal, and tabular modalities, and we benchmark against transformer-based and deep learning-based baselines trained centrally with Differentially Private Stochastic Gradient Descent (DP-SGD) and Renyi Differential Privacy (RDP). PrivateDFL delivers higher accuracy, lower latency, and lower energy across IID and non-IID partitions while preserving formal (epsilon, delta) guarantees and operating without a central server. For example, under non-IID partitions, PrivateDFL achieves 24.42% higher accuracy than the Vision Transformer on MNIST while using about 10x less training time, 76x lower inference latency, and 11x less energy, and on ISOLET it exceeds Transformer accuracy by more than 80% with roughly 10x less training time, 40x lower inference latency, and 36x less training energy. Future work will extend the explainable accounting to adversarial clients and adaptive topologies with heterogeneous privacy budgets. 

**Abstract (ZH)**: 去中心联邦学习面临隐私风险，因为模型更新可能通过推断攻击和成员推断泄露数据，这种风险随着客户端交换次数增加而增大。差分隐私通过注入校准噪声提供 principled 保护，确保在资源有限的物联网设备上保密信息的安全。然而，缺乏透明性使得黑盒训练无法跟踪之前客户端和轮次已注入的噪声，这导致最坏情况下的添加并损害准确性。我们提出了一个可解释框架 PrivateDFL，该框架将超维度计算与差分隐私相结合，并保持可审计的累积噪声记录，使得每个客户端仅添加所需的噪声与已积累的噪声之间的差异。我们在 MNIST、ISOLET 和 UCI-HAR 上进行评估，涵盖了图像、信号和表格模态，并与使用 Differentially Private Stochastic Gradient Descent (DP-SGD) 和 Rényi 差分隐私 (RDP) 中心训练的 Transformer 基线和深度学习基线进行基准测试。PrivateDFL 在同质划分和非同质划分中均实现了更高的准确率、更低的延迟和更低的能耗，同时保持形式化（ε, δ）保证，并且无需中央服务器。例如，在非同质划分下，PrivateDFL 在 MNIST 上比 Vision Transformer 的准确率高出 24.42%，同时训练时间减少约 10 倍，推断延迟降低约 76 倍，能耗降低约 11 倍；在 ISOLET 上，它比 Transformer 的准确率高出超过 80%，训练时间减少约 10 倍，推断延迟降低约 40 倍，训练能耗降低约 36 倍。未来的工作将扩展可解释的会计记录到 adversarial 客户端和具有异构隐私预算的自适应拓扑结构。 

---
# Self-Supervised Goal-Reaching Results in Multi-Agent Cooperation and Exploration 

**Title (ZH)**: 自我监督的目标导向学习促进多智能体的合作与探索 

**Authors**: Chirayu Nimonkar, Shlok Shah, Catherine Ji, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2509.10656)  

**Abstract**: For groups of autonomous agents to achieve a particular goal, they must engage in coordination and long-horizon reasoning. However, designing reward functions to elicit such behavior is challenging. In this paper, we study how self-supervised goal-reaching techniques can be leveraged to enable agents to cooperate. The key idea is that, rather than have agents maximize some scalar reward, agents aim to maximize the likelihood of visiting a certain goal. This problem setting enables human users to specify tasks via a single goal state rather than implementing a complex reward function. While the feedback signal is quite sparse, we will demonstrate that self-supervised goal-reaching techniques enable agents to learn from such feedback. On MARL benchmarks, our proposed method outperforms alternative approaches that have access to the same sparse reward signal as our method. While our method has no explicit mechanism for exploration, we observe that self-supervised multi-agent goal-reaching leads to emergent cooperation and exploration in settings where alternative approaches never witness a single successful trial. 

**Abstract (ZH)**: 自主代理群体实现特定目标时需要进行协调和长视角推理，设计能够激发这种行为的奖励函数具有挑战性。本文研究如何利用自我监督的目的达成技术使代理能够进行合作。关键思想是，不是让代理最大化某个标量奖励，而是让代理最大化访问特定目标的可能性。这种问题设置使得人类用户可以通过指定单个目标状态来说明任务，而无需实现复杂的奖励函数。尽管反馈信号非常稀疏，我们证明自我监督的目的达成技术仍能使代理从这样的反馈中学习。在多智能体强化学习基准测试中，我们提出的方法在可以访问相同稀疏奖励信号的替代方法中表现出色。虽然我们的方法没有显式探索机制，但我们观察到自我监督的多代理目的达成会导致在替代方法从未见证成功的场景中出现自发的合作和探索。 

---
# SCOR: A Framework for Responsible AI Innovation in Digital Ecosystems 

**Title (ZH)**: SCOR：数字生态系统中负责任的AI创新框架 

**Authors**: Mohammad Saleh Torkestani, Taha Mansouri  

**Link**: [PDF](https://arxiv.org/pdf/2509.10653)  

**Abstract**: AI-driven digital ecosystems span diverse stakeholders including technology firms, regulators, accelerators and civil society, yet often lack cohesive ethical governance. This paper proposes a four-pillar framework (SCOR) to embed accountability, fairness, and inclusivity across such multi-actor networks. Leveraging a design science approach, we develop a Shared Ethical Charter(S), structured Co-Design and Stakeholder Engagement protocols(C), a system of Continuous Oversight and Learning(O), and Adaptive Regulatory Alignment strategies(R). Each component includes practical guidance, from lite modules for resource-constrained start-ups to in-depth auditing systems for larger consortia. Through illustrative vignettes in healthcare, finance, and smart city contexts, we demonstrate how the framework can harmonize organizational culture, leadership incentives, and cross-jurisdictional compliance. Our mixed-method KPI design further ensures that quantitative targets are complemented by qualitative assessments of user trust and cultural change. By uniting ethical principles with scalable operational structures, this paper offers a replicable pathway toward responsible AI innovation in complex digital ecosystems. 

**Abstract (ZH)**: AI驱动的数字生态系统跨越多元利益相关者，包括技术公司、监管机构、加速器和民间社会，但往往缺乏统一的伦理治理。本文提出一种四支柱框架（SCOR），以嵌入问责、公平和包容性于此类多利益相关者网络中。基于设计科学方法，我们开发了一个共同的伦理宪章（S）、结构化的共同设计和利益相关者参与协议（C）、持续监督和学习体系（O），以及适应性监管对齐策略（R）。每个组成部分自资源受限的初创公司的轻量级模块到大型联盟的深度审计系统，均包含实用指导。通过医疗保健、金融和智慧城市背景下的示例情景，展示框架如何协调组织文化、领导激励和跨司法辖区合规性。我们的混合方法KPI设计进一步确保了定量目标与用户信任和文化变革的定性评估相补充。通过将伦理原则与可扩展的操作结构相结合，本文为复杂数字生态系统中的负责任AI创新提供了一种可复制的道路。 

---
# Vibe Coding for UX Design: Understanding UX Professionals' Perceptions of AI-Assisted Design and Development 

**Title (ZH)**: 用户体验设计中的Vibe编码：理解用户体验专业人士对AI辅助设计与开发的看法 

**Authors**: Jie Li, Youyang Hou, Laura Lin, Ruihao Zhu, Hancheng Cao, Abdallah El Ali  

**Link**: [PDF](https://arxiv.org/pdf/2509.10652)  

**Abstract**: Generative AI is reshaping UX design practices through "vibe coding," where UX professionals express intent in natural language and AI translates it into functional prototypes and code. Despite rapid adoption, little research has examined how vibe coding reconfigures UX workflows and collaboration. Drawing on interviews with 20 UX professionals across enterprises, startups, and academia, we show how vibe coding follows a four-stage workflow of ideation, AI generation, debugging, and review. This accelerates iteration, supports creativity, and lowers barriers to participation. However, professionals reported challenges of code unreliability, integration, and AI over-reliance. We find tensions between efficiency-driven prototyping ("intending the right design") and reflection ("designing the right intention"), introducing new asymmetries in trust, responsibility, and social stigma within teams. Through the lens of responsible human-AI collaboration for AI-assisted UX design and development, we contribute a deeper understanding of deskilling, ownership and disclosure, and creativity safeguarding in the age of vibe coding. 

**Abstract (ZH)**: 生成式AI通过“氛围编码”重塑UX设计实践：从创想到审查的四阶段工作流及其实质分析 

---
# Optimal Multimarginal Schrödinger Bridge: Minimum Spanning Tree over Measure-valued Vertices 

**Title (ZH)**: 最优多边际薛定谔桥：测度值顶点的最小生成树 

**Authors**: Georgiy A. Bondar, Abhishek Halder  

**Link**: [PDF](https://arxiv.org/pdf/2509.10626)  

**Abstract**: The Multimarginal Schrödinger Bridge (MSB) finds the optimal coupling among a collection of random vectors with known statistics and a known correlation structure. In the MSB formulation, this correlation structure is specified \emph{a priori} as an undirected connected graph with measure-valued vertices. In this work, we formulate and solve the problem of finding the optimal MSB in the sense we seek the optimal coupling over all possible graph structures. We find that computing the optimal MSB amounts to solving the minimum spanning tree problem over measure-valued vertices. We show that the resulting problem can be solved in two steps. The first step constructs a complete graph with edge weight equal to a sum of the optimal value of the corresponding bimarginal SB and the entropies of the endpoints. The second step solves a standard minimum spanning tree problem over that complete weighted graph. Numerical experiments illustrate the proposed solution. 

**Abstract (ZH)**: 多边际薛定谔桥（MSB）在已知统计数据和相关结构的情况下，寻找一组随机向量的最佳耦合。在MSB形式化中，这种相关结构被先验地指定为一个具有测度顶点的无向连通图。在本文中，我们提出了在所有可能的图结构中寻找最优MSB的问题，并发现计算最优MSB等同于在测度顶点上求解最小生成树问题。我们证明了该问题可以分为两步求解。第一步构造一个完整的图，边权重等于相应双边际SB的最优值和端点熵之和。第二步在该加权完全图上求解标准的最小生成树问题。数值实验验证了提出的方法。 

---
# National Running Club Database: Assessing Collegiate Club Athletes' Cross Country Race Results 

**Title (ZH)**: 国家级跑步俱乐部数据库：评估大学生跨项赛跑成绩 

**Authors**: Jonathan A. Karr Jr, Ben Darden, Nicholas Pell, Ryan M. Fryer, Kayla Ambrose, Evan Hall, Ramzi K. Bualuan, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2509.10600)  

**Abstract**: The National Running Club Database (NRCD) aggregates 15,397 race results of 5,585 athletes from the 2023 and 2024 cross country seasons. This paper introduces the NRCD dataset, which provides insights into individual athlete progressions, enabling data-driven decision-making. Analysis reveals that runners' improvement per calendar day for women, racing 6,000m, and men, racing 8,000m, is more pronounced in athletes with slower initial race times and those who race more frequently. Additionally, we factor in course conditions, including weather and elevation gain, to standardize improvement. While the NRCD shows a gender imbalance, 3,484 men vs. 2,101 women, the racing frequency between genders is comparable. This publication makes the NRCD dataset accessible to the research community, addressing a previous challenge where smaller datasets, often limited to 500 entries, had to be manually scraped from the internet. Focusing on club athletes rather than elite professionals offers a unique lens into the performance of real-world runners who balance competition with academics and other commitments. These results serve as a valuable resource for runners, coaches, and teams, bridging the gap between raw data and applied sports science. 

**Abstract (ZH)**: 国家跑步俱乐部数据库（NRCD）汇集了2023和2024赛季5585名运动员共15,397场比赛的结果。本文介绍了NRCD数据集，提供了关于个别运动员进步的见解，有助于基于数据的决策制定。分析表明，在6000米赛跑中起跑较慢且比赛频率较高的女性运动员以及在8000米赛跑中起跑较慢且比赛频率较高的男性运动员每天历日的进步更为显著。此外，我们还考虑了包括天气和海拔变化在内的赛道条件，以标准化进步程度。尽管NRCD显示出性别不平衡，男性3484人 vs. 女性2101人，但男女运动员的竞赛频率相当。本文使NRCD数据集对研究人员开放，解决了此前小数据集（通常仅包含500条记录）需要手动从互联网抓取的难题。专注于俱乐部运动员而非精英职业选手，为我们提供了一个独特的视角，深入了解了兼顾比赛、学术和其他承诺的实际跑者的表现。这些结果为跑步者、教练员和团队提供了宝贵的资源，填补了原始数据与应用运动科学之间的差距。 

---
# GenAI Voice Mode in Programming Education 

**Title (ZH)**: GenAI语音模式在编程教育中的应用 

**Authors**: Sven Jacobs, Natalie Kiesler  

**Link**: [PDF](https://arxiv.org/pdf/2509.10596)  

**Abstract**: Real-time voice interfaces using multimodal Generative AI (GenAI) can potentially address the accessibility needs of novice programmers with disabilities (e.g., related to vision). Yet, little is known about how novices interact with GenAI tools and their feedback quality in the form of audio output. This paper analyzes audio dialogues from nine 9th-grade students using a voice-enabled tutor (powered by OpenAI's Realtime API) in an authentic classroom setting while learning Python. We examined the students' voice prompts and AI's responses (1210 messages) by using qualitative coding. We also gathered students' perceptions via the Partner Modeling Questionnaire. The GenAI Voice Tutor primarily offered feedback on mistakes and next steps, but its correctness was limited (71.4% correct out of 416 feedback outputs). Quality issues were observed, particularly when the AI attempted to utter programming code elements. Students used the GenAI voice tutor primarily for debugging. They perceived it as competent, only somewhat human-like, and flexible. The present study is the first to explore the interaction dynamics of real-time voice GenAI tutors and novice programmers, informing future educational tool design and potentially addressing accessibility needs of diverse learners. 

**Abstract (ZH)**: 实时语音接口使用多模态生成AI（GenAI）可能解决视觉障碍等残疾初学者程序员的 accessibility 需求。然而，关于初学者如何与生成AI工具互动及其语音输出反馈质量的研究尚少。本文分析了九名九年级学生在使用语音驱动辅导系统（基于OpenAI的Realtime API）学习Python过程中的音频对话，通过定性编码研究了学生的语音提示和AI的回应（共1210条消息），并通过伙伴建模问卷收集了学生的观点。生成AI语音辅导主要提供错误反馈和下一步建议，但其正确性有限（正确率为71.4%）。特别是在生成编程代码元素时，出现了质量缺陷。学生主要使用生成AI语音辅导进行调试。他们认为其表现专业、略带人性化且灵活。本研究是首次探索实时语音生成AI辅导系统与初学者程序员的互动动态，为未来教育工具设计提供参考并可能解决不同学习者的可访问性需求。 

---
# Assisting the Grading of a Handwritten General Chemistry Exam with Artificial Intelligence 

**Title (ZH)**: 利用人工智能辅助手写无机化学考试评分 

**Authors**: Jan Cvengros, Gerd Kortemeyer  

**Link**: [PDF](https://arxiv.org/pdf/2509.10591)  

**Abstract**: We explore the effectiveness and reliability of an artificial intelligence (AI)-based grading system for a handwritten general chemistry exam, comparing AI-assigned scores to human grading across various types of questions. Exam pages and grading rubrics were uploaded as images to account for chemical reaction equations, short and long open-ended answers, numerical and symbolic answer derivations, drawing, and sketching in pencil-and-paper format. Using linear regression analyses and psychometric evaluations, the investigation reveals high agreement between AI and human graders for textual and chemical reaction questions, while highlighting lower reliability for numerical and graphical tasks. The findings emphasize the necessity for human oversight to ensure grading accuracy, based on selective filtering. The results indicate promising applications for AI in routine assessment tasks, though careful consideration must be given to student perceptions of fairness and trust in integrating AI-based grading into educational practice. 

**Abstract (ZH)**: 基于人工智能的笔试试卷评分系统的有效性和可靠性探究：以无机化学考试为例 

---
# Machine Unlearning for Responsible and Adaptive AI in Education 

**Title (ZH)**: 负责任且适应性的教育人工智能中的机器遗忘技术 

**Authors**: Betty Mayeku, Sandra Hummel, Parisa Memarmoshrefi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10590)  

**Abstract**: The concept of Machine Unlearning (MU) has gained popularity in various domains due to its ability to address several issues in Machine Learning (ML) models, particularly those related to privacy, security, bias mitigation, and adaptability. With these abilities, MU is evolving into a promising technology in upholding Responsible AI principles and optimizing ML models' performance. However, despite its promising potential, the concept has not received much attention in the education sector. In an attempt to encourage further uptake of this promising technology in the educational landscape, this paper demonstrates that MU indeed has great potential to serve as a practical mechanism for operationalizing Responsible AI principles as well as an essential tool for Adaptive AI within the educational application domain hence fostering trust in AI-driven educational systems. Through a structured review of 42 peer-reviewed sources, we identify four domains where MU holds particular promise namely privacy protection, resilience against adversarial inputs, mitigation of systemic bias, and adaptability in evolving learning contexts. We systematically explore these potentials and their interventions to core challenges in ML-based education systems. As a conceptual contribution, we present a reference Machine Unlearning application architecture for Responsible and Adaptive AI (MU-RAAI) in education context. 

**Abstract (ZH)**: 机器遗忘（MU）的概念由于其在解决机器学习模型中隐私、安全、偏见缓解和适应性等方面问题的能力而在各个领域中获得了 popularity，并正在演进成为维护负责任人工智能原则和优化机器学习模型性能的有希望的技术。尽管 MU 具有很大的潜力，但这一概念在教育领域尚未受到广泛关注。为了促进这一有希望的技术在教育领域的进一步应用，本文通过结构化的文献回顾（共 42 篇同行评审来源）展示了机器遗忘确实具有作为实现负责任和适应性人工智能原则的实用机制以及教育应用领域中适应性人工智能的重要工具的潜力，从而促进对人工智能驱动教育系统的信任。在此基础上，我们提出了一个适用于教育背景的责任和适应性人工智能的机器遗忘应用架构（MU-RAAI）。 

---
# LearnLens: An AI-Enhanced Dashboard to Support Teachers in Open-Ended Classrooms 

**Title (ZH)**: LearnLens: 一种增强型教学板面，支持开放教室中的教师 

**Authors**: Namrata Srivastava, Shruti Jain, Clayton Cohn, Naveeduddin Mohammed, Umesh Timalsina, Gautam Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2509.10582)  

**Abstract**: Exploratory learning environments (ELEs), such as simulation-based platforms and open-ended science curricula, promote hands-on exploration and problem-solving but make it difficult for teachers to gain timely insights into students' conceptual understanding. This paper presents LearnLens, a generative AI (GenAI)-enhanced teacher-facing dashboard designed to support problem-based instruction in middle school science. LearnLens processes students' open-ended responses from digital assessments to provide various insights, including sample responses, word clouds, bar charts, and AI-generated summaries. These features elucidate students' thinking, enabling teachers to adjust their instruction based on emerging patterns of understanding. The dashboard was informed by teacher input during professional development sessions and implemented within a middle school Earth science curriculum. We report insights from teacher interviews that highlight the dashboard's usability and potential to guide teachers' instruction in the classroom. 

**Abstract (ZH)**: 基于生成式AI的面向教师的学习镜像板：促进中学科学问题导向教学的探索性学习环境 

---
# The Coding Limits of Robust Watermarking for Generative Models 

**Title (ZH)**: 稳健水印技术在生成模型中的编码极限 

**Authors**: Danilo Francati, Yevin Nikhel Goonatilake, Shubham Pawar, Daniele Venturi, Giuseppe Ateniese  

**Link**: [PDF](https://arxiv.org/pdf/2509.10577)  

**Abstract**: We prove a sharp threshold for the robustness of cryptographic watermarking for generative models. This is achieved by introducing a coding abstraction, which we call messageless secret-key codes, that formalizes sufficient and necessary requirements of robust watermarking: soundness, tamper detection, and pseudorandomness. Thus, we establish that robustness has a precise limit: For binary outputs no scheme can survive if more than half of the encoded bits are modified, and for an alphabet of size q the corresponding threshold is $(1-1/q)$ of the symbols.
Complementing this impossibility, we give explicit constructions that meet the bound up to a constant slack. For every ${\delta} > 0$, assuming pseudorandom functions and access to a public counter, we build linear-time codes that tolerate up to $(1/2)(1-{\delta})$ errors in the binary case and $(1-1/q)(1-{\delta})$ errors in the $q$-ary case. Together with the lower bound, these yield the maximum robustness achievable under standard cryptographic assumptions.
We then test experimentally whether this limit appears in practice by looking at the recent watermarking for images of Gunn, Zhao, and Song (ICLR 2025). We show that a simple crop and resize operation reliably flipped about half of the latent signs and consistently prevented belief-propagation decoding from recovering the codeword, erasing the watermark while leaving the image visually intact.
These results provide a complete characterization of robust watermarking, identifying the threshold at which robustness fails, constructions that achieve it, and an experimental confirmation that the threshold is already reached in practice. 

**Abstract (ZH)**: 我们证明了一个严格的阈值，以确定加密水印在生成模型中的鲁棒性。通过引入一种编码抽象——我们称之为无信息密钥消息编码，来正式化鲁棒水印的充分必要条件：正确性、篡改检测和伪随机性。因此，我们建立了鲁棒性具有精确的界限：对于二进制输出，如果超过一半的编码位被修改，任何方案都无法生存；对于大小为q的字母表，相应的阈值为符号的$(1-1/q)$。补充这一不可能性，我们给出了明确的构造，这些构造在常数误差容差范围内达到了上述界限。对于每个$\delta > 0$，假设伪随机函数并访问公共计数器，我们构建了线性时间码，能够在二进制情况下容忍最多$(1/2)(1-\delta)$的错误，在$q$-进制情况下容忍最多$(1-1/q)(1-\delta)$的错误。这些线性时间码与下界结合，得出了在标准密码学假设下的最大鲁棒性。然后，我们通过实验检查这种界限在实践中是否出现，研究了Gunn, Zhao, 和 Song（ICLR 2025）的图像水印。我们表明，简单的裁剪和缩放操作可靠地翻转了大约一半的潜在标志，并且一致地阻止了基于传播的解码恢复码字，消除了水印同时使图像保持视觉上不变。这些结果为鲁棒水印提供了完整的规范，确定了鲁棒性失效的阈值，给出了实现此阈值的构造，并通过实验验证，在实践中已经达到了该阈值。 

---
# Aesthetic Experience and Educational Value in Co-creating Art with Generative AI: Evidence from a Survey of Young Learners 

**Title (ZH)**: 与生成性人工智能共创艺术的审美体验与教育价值：基于年轻学习者调查的证据 

**Authors**: Chengyuan Zhang, Suzhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10576)  

**Abstract**: This study investigates the aesthetic experience and educational value of collaborative artmaking with generative artificial intelligence (AI) among young learners and art students. Based on a survey of 112 participants, we examine how human creators renegotiate their roles, how conventional notions of originality are challenged, how the creative process is transformed, and how aesthetic judgment is formed in human--AI co-creation. Empirically, participants generally view AI as a partner that stimulates ideation and expands creative boundaries rather than a passive tool, while simultaneously voicing concerns about stylistic homogenization and the erosion of traditional authorship. Theoretically, we synthesize Dewey's aesthetics of experience, Ihde's postphenomenology, and actor--network theory (ANT) into a single analytical framework to unpack the dynamics between human creators and AI as a non-human actant. Findings indicate (i) a fluid subjectivity in which creators shift across multiple stances (director, dialogic partner, discoverer); (ii) an iterative, dialogic workflow (intent--generate--select--refine) that centers critical interpretation; and (iii) an educational value shift from technical skill training toward higher-order competencies such as critical judgment, cross-modal ideation, and reflexivity. We argue that arts education should cultivate a \emph{critical co-creation} stance toward technology, guiding learners to collaborate with AI while preserving human distinctiveness in concept formation, judgment, and meaning-making. 

**Abstract (ZH)**: 本研究调查了年轻学习者和艺术学生在与生成型人工智能（AI）协作艺术创作中的审美体验和教育价值。基于对112名参与者的调查，我们探讨了人类创作者重新谈判其角色的方式、传统原创性观念的挑战、创作过程的转变以及人类与AI共创作中的审美判断形成。实证上，参与者普遍将AI视为激发创意和扩展创作边界的合作伙伴，而不是被动工具，同时表达了对风格同质化和传统作者身份侵蚀的担忧。理论层面，我们将杜威的经验美学、伊赫德的后现象学以及行动者网络理论（ANT）整合为一个分析框架，以分析人类创作者与非人类行动者AI之间的动态关系。研究发现包括：（i）流变的主体性，在其中创作者转换不同的立场（导演、对话伙伴、发现者）；（ii）迭代的、对话的工作流程（意图—生成—选择—提炼），以批判性解释为中心；（iii）从技术技能训练到更高阶能力，如批判性判断、跨模态创意和反思性的转变。我们认为，艺术教育应培养一种对技术的“批判性共创作”态度，引导学习者与AI合作同时保持人类在概念形成、判断和意义建构中的独特性。 

---
# MarkDiffusion: An Open-Source Toolkit for Generative Watermarking of Latent Diffusion Models 

**Title (ZH)**: MarkDiffusion: 开源生成水印工具包——面向潜层扩散模型的生成性水印技术 

**Authors**: Leyi Pan, Sheng Guan, Zheyu Fu, Luyang Si, Zian Wang, Xuming Hu, Irwin King, Philip S. Yu, Aiwei Liu, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.10569)  

**Abstract**: We introduce MarkDiffusion, an open-source Python toolkit for generative watermarking of latent diffusion models. It comprises three key components: a unified implementation framework for streamlined watermarking algorithm integrations and user-friendly interfaces; a mechanism visualization suite that intuitively showcases added and extracted watermark patterns to aid public understanding; and a comprehensive evaluation module offering standard implementations of 24 tools across three essential aspects - detectability, robustness, and output quality - plus 8 automated evaluation pipelines. Through MarkDiffusion, we seek to assist researchers, enhance public awareness and engagement in generative watermarking, and promote consensus while advancing research and applications. 

**Abstract (ZH)**: MarkDiffusion：一个开源Python工具包，用于生成-latent扩散模型中的水印 

---
# Biomarkers of brain diseases 

**Title (ZH)**: 脑疾病标志物 

**Authors**: Pascal Helson, Arvind Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.10547)  

**Abstract**: Despite the diversity of brain data acquired and advanced AI-based algorithms to analyze them, brain features are rarely used in clinics for diagnosis and prognosis. Here we argue that the field continues to rely on cohort comparisons to seek biomarkers, despite the well-established degeneracy of brain features. Using a thought experiment, we show that more data and more powerful algorithms will not be sufficient to identify biomarkers of brain diseases. We argue that instead of comparing patient versus healthy controls using single data type, we should use multimodal (e.g. brain activity, neurotransmitters, neuromodulators, brain imaging) and longitudinal brain data to guide the grouping before defining multidimensional biomarkers for brain diseases. 

**Abstract (ZH)**: 尽管获取的脑数据多样且存在先进的基于AI的分析算法，脑特征在临床诊断和预后中使用仍然很少。我们argue继续依赖队列比较来寻找生物标志物，尽管已建立的脑特征退化现象已被确立。通过思想实验，我们表明，更多的数据和更强大的算法不足以识别脑疾病的生物标志物。我们argue应在使用多模态（如脑活动、神经递质、神经调节物、脑成像）和纵向脑数据来指导分组之前，而不是使用单一数据类型来比较患者与健康对照组，以定义脑疾病的多维生物标志物。 

---
# Robust DDoS-Attack Classification with 3D CNNs Against Adversarial Methods 

**Title (ZH)**: 基于3D CNNs的抗对抗方法的稳健DDoS攻击分类 

**Authors**: Landon Bragg, Nathan Dorsey, Josh Prior, John Ajit, Ben Kim, Nate Willis, Pablo Rivas  

**Link**: [PDF](https://arxiv.org/pdf/2509.10543)  

**Abstract**: Distributed Denial-of-Service (DDoS) attacks remain a serious threat to online infrastructure, often bypassing detection by altering traffic in subtle ways. We present a method using hive-plot sequences of network data and a 3D convolutional neural network (3D CNN) to classify DDoS traffic with high accuracy. Our system relies on three main ideas: (1) using spatio-temporal hive-plot encodings to set a pattern-recognition baseline, (2) applying adversarial training with FGSM and PGD alongside spatial noise and image shifts, and (3) analyzing frame-wise predictions to find early signals. On a benchmark dataset, our method lifts adversarial accuracy from 50-55% to over 93% while maintaining clean-sample performance. Frames 3-4 offer strong predictive signals, showing early-stage classification is possible. 

**Abstract (ZH)**: 分布式拒绝服务（DDoS）攻击仍然是在线基础设施的重大威胁，常通过微妙改变流量的方式规避检测。我们提出了一种方法，利用网络数据的蜂巢图序列和三维卷积神经网络（3D CNN）对DDoS流量进行高精度分类。该系统依赖于三个主要理念：（1）使用时空蜂巢图编码设定模式识别基准，（2）结合FGSM和PGD对抗训练及空间噪声和图像位移，（3）分析帧级预测以寻找早期信号。在基准数据集上，该方法将对抗准确性从50-55%提升至超过93%的同时保持干净样本性能。第3-4帧提供了强大的预测信号，表明早期阶段的分类是可能的。 

---
# On Using Large-Batches in Federated Learning 

**Title (ZH)**: 使用大批次在联邦学习中的应用 

**Authors**: Sahil Tyagi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10537)  

**Abstract**: Efficient Federated learning (FL) is crucial for training deep networks over devices with limited compute resources and bounded networks. With the advent of big data, devices either generate or collect multimodal data to train either generic or local-context aware networks, particularly when data privacy and locality is vital. FL algorithms generally trade-off between parallel and statistical performance, improving model quality at the cost of higher communication frequency, or vice versa. Under frequent synchronization settings, FL over a large cluster of devices may perform more work per-training iteration by processing a larger global batch-size, thus attaining considerable training speedup. However, this may result in poor test performance (i.e., low test loss or accuracy) due to generalization degradation issues associated with large-batch training. To address these challenges with large-batches, this work proposes our vision of exploiting the trade-offs between small and large-batch training, and explore new directions to enjoy both the parallel scaling of large-batches and good generalizability of small-batch training. For the same number of iterations, we observe that our proposed large-batch training technique attains about 32.33% and 3.74% higher test accuracy than small-batch training in ResNet50 and VGG11 models respectively. 

**Abstract (ZH)**: 高效的联邦学习在计算资源有限和网络带宽受限的设备上训练深度网络至关重要。随着大数据的到来，设备要么生成要么收集多模态数据来训练通用或局部上下文感知的网络，特别是在数据隐私和本地化至关重要的情况下。联邦学习算法通常在并行性能和统计性能之间进行权衡，在增加通信频率的同时提高模型质量，反之亦然。在频繁同步的设置下，联邦学习可以在大型设备集群中通过处理更大的全局批量大小，在每次训练迭代中执行更多的工作，从而实现显著的训练加速。然而，这可能导致由于大规模训练引起的泛化性能下降，从而导致较差的测试性能（即较高的测试损失或准确性）。为了解决这些问题，本文提出了我们的视角，即利用小批量和大批量训练之间的权衡，并探索新的方向，以同时享受大批量训练的并行扩展性和小批量训练的良好泛化性。对于相同数量的迭代，我们观察到，我们提出的大批量训练技术在ResNet50和VGG11模型中的测试准确率分别比小批量训练高32.33%和3.74%。 

---
# Semantic-guided LoRA Parameters Generation 

**Title (ZH)**: 语义导向的LoRA参数生成 

**Authors**: Miaoge Li, Yang Chen, Zhijie Rao, Can Jiang, Jingcai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.10535)  

**Abstract**: Low-Rank Adaptation (LoRA) has demonstrated strong generalization capabilities across a variety of tasks for efficiently fine-tuning AI models, especially on resource-constrained edges. However, in real-world applications, edge users often exhibit task-specific preferences that are difficult to handle with a unified model trained under a closed-world assumption, and the challenge may further increase when there are significant domain shifts between training and deployment. Meanwhile, retraining/fine-tuning models for each user is also impractical due to its cost-intensive nature and privacy concerns over raw data utilization from edges. To address these challenges, we propose Semantic-guided LoRA Parameter Generation (SG-LoRA), the first of its kind framework to efficiently produce user-specific LoRA parameters without any additional training on user tasks or access to user-specific data. Concretely, SG-LoRA uses task descriptions as the semantic bridge, measuring their proximity to a set of known expert tasks in a shared embedding space. Based on this semantic guidance, it models the target task's LoRA parameter distribution to generate high-performing parameters for novel tasks. SG-LoRA enables the real-time construction of LoRA models aligned with individual intents by distilling knowledge from prominent LoRA experts and, meanwhile, offering a privacy-preserving solution for personalized model adaptation in a novel zero-shot open-world setting proposed in this work. Extensive experiments on multiple challenging tasks confirm the superior performance and remarkable adaptability of SG-LoRA. Code is available at this https URL. 

**Abstract (ZH)**: 基于语义指导的LoRA参数生成（SG-LoRA） 

---
# FinXplore: An Adaptive Deep Reinforcement Learning Framework for Balancing and Discovering Investment Opportunities 

**Title (ZH)**: FinXplore：一种平衡与发现投资机会的自适应深度强化学习框架 

**Authors**: Himanshu Choudhary, Arishi Orra, Manoj Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2509.10531)  

**Abstract**: Portfolio optimization is essential for balancing risk and return in financial decision-making. Deep Reinforcement Learning (DRL) has stood out as a cutting-edge tool for portfolio optimization that learns dynamic asset allocation using trial-and-error interactions. However, most DRL-based methods are restricted to allocating assets within a pre-defined investment universe and overlook exploring new opportunities. This study introduces an investment landscape that integrates exploiting existing assets with exploring new investment opportunities in an extended universe. The proposed approach leverages two DRL agents and dynamically balances these objectives to adapt to evolving markets while enhancing portfolio performance. One agent allocates assets within the existing universe, while another assists in exploring new opportunities in the extended universe. The effciency of the proposed methodology is determined using two real-world market data sets. The experiments demonstrate the superiority of the suggested approach against the state-of-the-art portfolio strategies and baseline methods. 

**Abstract (ZH)**: 投资组合优化是平衡风险和回报的金融决策中必不可少的。深度强化学习（DRL）作为一种通过试错交互学习动态资产分配的前沿工具，在投资组合优化中脱颖而出。然而，大多数基于DRL的方法仅限于在预定义的投资范围内分配资产，忽略了探索新机会。本研究引入了一个将利用现有资产与探索扩展投资范围内的新机会相结合的投资景觀。提出的这种方法利用两个DRL代理动态平衡这些目标，以适应不断变化的市场环境并提升投资组合表现。一个代理在现有范围内分配资产，另一个则在扩展范围内协助探索新机会。利用两个实际市场数据集评估所提出方法的有效性。实验结果证明，建议的方法在与最新投资组合策略和基准方法的对比中具有优越性。 

---
# Dynamic Adaptive Shared Experts with Grouped Multi-Head Attention Mixture of Experts 

**Title (ZH)**: 动态自适应分组多头注意力专家混合模型 

**Authors**: Cheng Li, Jiexiong Liu, Yixuan Chen, Jie ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.10530)  

**Abstract**: Transformer models based on the Mixture of Experts (MoE) architecture have made significant progress in long-sequence modeling, but existing models still have shortcomings in computational efficiency and the ability to capture long-range dependencies, especially in terms of the dynamic adaptability of expert resource allocation. In this paper, we propose a Dynamic Adaptive Shared Expert and Grouped Multi-Head Attention Hybrid Model (DASG-MoE) to enhance long-sequence modeling capabilities by integrating three modules. First, we employ the Grouped Multi-Head Attention (GMHA) mechanism to effectively reduce the computational complexity of long sequences. By parallel processing through sequence grouping, local sliding window attention, and feature aggregation, we address long-range dependency issues and the model's lack of generalization for local information. Second, we design a Dual-Scale Shared Expert Structure (DSSE), where shallow experts use lightweight computations to quickly respond to low-dimensional features, while deep experts process high-dimensional complex semantics through pre-training transfer and post-training optimization, achieving a dynamic balance between efficiency and accuracy. Third, we propose a hierarchical Adaptive Dynamic Routing (ADR) mechanism that dynamically selects expert levels based on feature complexity and task requirements, and optimizes resource allocation through a local expert activation strategy. Experiments on multiple long-sequence benchmark datasets demonstrate that our DASG-MoE model outperforms state-of-the-art models. 

**Abstract (ZH)**: 基于Mixture of Experts（MoE）架构的Dynamic Adaptive Shared Expert and Grouped Multi-Head Attention Hybrid Model（DASG-MoE）在长序列建模中的动态适配研究 

---
# Mitigating Catastrophic Forgetting and Mode Collapse in Text-to-Image Diffusion via Latent Replay 

**Title (ZH)**: 基于潜在空间重播缓解文本到图像扩散模型中的灾难性遗忘和模式collapse 

**Authors**: Aoi Otani  

**Link**: [PDF](https://arxiv.org/pdf/2509.10529)  

**Abstract**: Continual learning -- the ability to acquire knowledge incrementally without forgetting previous skills -- is fundamental to natural intelligence. While the human brain excels at this, artificial neural networks struggle with "catastrophic forgetting," where learning new tasks erases previously acquired knowledge. This challenge is particularly severe for text-to-image diffusion models, which generate images from textual prompts. Additionally, these models face "mode collapse," where their outputs become increasingly repetitive over time. To address these challenges, we apply Latent Replay, a neuroscience-inspired approach, to diffusion models. Traditional replay methods mitigate forgetting by storing and revisiting past examples, typically requiring large collections of images. Latent Replay instead retains only compact, high-level feature representations extracted from the model's internal architecture. This mirrors the hippocampal process of storing neural activity patterns rather than raw sensory inputs, reducing memory usage while preserving critical information. Through experiments with five sequentially learned visual concepts, we demonstrate that Latent Replay significantly outperforms existing methods in maintaining model versatility. After learning all concepts, our approach retained 77.59% Image Alignment (IA) on the earliest concept, 14% higher than baseline methods, while maintaining diverse outputs. Surprisingly, random selection of stored latent examples outperforms similarity-based strategies. Our findings suggest that Latent Replay enables efficient continual learning for generative AI models, paving the way for personalized text-to-image models that evolve with user needs without excessive computational costs. 

**Abstract (ZH)**: 持续学习——能够在不遗忘先前技能的情况下逐步获取知识的能力——是自然智能的基础。尽管人脑在这方面表现出色，但人工神经网络却难以克服“灾难性遗忘”的挑战，即学习新任务会抹去先前习得的知识。这种挑战尤其严重地影响了从文本生成图像的扩散模型，这些模型会产生基于文本提示的图像。此外，这些模型还面临着“模式塌陷”的问题，随着时间的推移，其输出变得越来越重复。为了解决这些挑战，我们应用了受神经科学启发的潜在重放方法到扩散模型中。传统的重放方法通过存储和回顾过去的示例来减轻遗忘，通常需要大量的图像集合。相比之下，潜在重放只保留模型内部架构提取的紧凑的高层特征表示，这类似于海马体存储神经活动模式而不是原始感官输入，从而减少内存使用同时保留关键信息。通过五种连续学习的视觉概念实验，我们证明了潜在重放在保持模型的通用性方面显著优于现有方法。在学习所有概念之后，我们的方法在最早的概念上保留了77.59%的图像对齐（IA），比基线方法高14%，同时保持了多样的输出。令人惊讶的是，随机选择存储的潜在示例的表现优于基于相似性的策略。我们的研究结果表明，潜在重放能够实现生成型AI模型的有效持续学习，为根据用户需求发展个性化的文本到图像模型铺平了道路，且无需过多的计算成本。 

---
# STM-Graph: A Python Framework for Spatio-Temporal Mapping and Graph Neural Network Predictions 

**Title (ZH)**: STM-Graph: 一种时空映射与图神经网络预测的Python框架 

**Authors**: Amirhossein Ghaffari, Huong Nguyen, Lauri Lovén, Ekaterina Gilman  

**Link**: [PDF](https://arxiv.org/pdf/2509.10528)  

**Abstract**: Urban spatio-temporal data present unique challenges for predictive analytics due to their dynamic and complex nature. We introduce STM-Graph, an open-source Python framework that transforms raw spatio-temporal urban event data into graph representations suitable for Graph Neural Network (GNN) training and prediction. STM-Graph integrates diverse spatial mapping methods, urban features from OpenStreetMap, multiple GNN models, comprehensive visualization tools, and a graphical user interface (GUI) suitable for professional and non-professional users. This modular and extensible framework facilitates rapid experimentation and benchmarking. It allows integration of new mapping methods and custom models, making it a valuable resource for researchers and practitioners in urban computing. The source code of the framework and GUI are available at: this https URL and this https URL. 

**Abstract (ZH)**: 城市时空数据因其动态和复杂性，为预测分析带来了独特的挑战。本文介绍了STM-Graph，一个开源Python框架，将原始的时空城市事件数据转换为适用于图神经网络（GNN）训练和预测的图表示。STM-Graph集成了多种空间映射方法、来自OpenStreetMap的urban特征、多种GNN模型、全面的可视化工具以及适合专业和非专业用户的图形用户界面（GUI）。该模块化和可扩展的框架便于快速实验和基准测试。它允许集成新的映射方法和自定义模型，使其成为城市计算研究者和实践者的重要资源。该框架和GUI的源代码可在以下链接获取：this https URL 和 this https URL。 

---
# Resource-Aware Neural Network Pruning Using Graph-based Reinforcement Learning 

**Title (ZH)**: 基于图强化学习的资源意识神经网络剪枝 

**Authors**: Dieter Balemans, Thomas Huybrechts, Jan Steckel, Siegfried Mercelis  

**Link**: [PDF](https://arxiv.org/pdf/2509.10526)  

**Abstract**: This paper presents a novel approach to neural network pruning by integrating a graph-based observation space into an AutoML framework to address the limitations of existing methods. Traditional pruning approaches often depend on hand-crafted heuristics and local optimization perspectives, which can lead to suboptimal performance and inefficient pruning strategies. Our framework transforms the pruning process by introducing a graph representation of the target neural network that captures complete topological relationships between layers and channels, replacing the limited layer-wise observation space with a global view of network structure. The core innovations include a Graph Attention Network (GAT) encoder that processes the network's graph representation and generates a rich embedding. Additionally, for the action space we transition from continuous pruning ratios to fine-grained binary action spaces which enables the agent to learn optimal channel importance criteria directly from data, moving away from predefined scoring functions. These contributions are modelled within a Constrained Markov Decision Process (CMDP) framework, allowing the agent to make informed pruning decisions while adhering to resource constraints such as target compression rates. For this, we design a self-competition reward system that encourages the agent to outperform its previous best performance while satisfying the defined constraints. We demonstrate the effectiveness of our approach through extensive experiments on benchmark datasets including CIFAR-10, CIFAR-100, and ImageNet. The experiments show that our method consistently outperforms traditional pruning techniques, showing state-of-the-art results while learning task-specific pruning strategies that identify functionally redundant connections beyond simple weight magnitude considerations. 

**Abstract (ZH)**: 基于图观察空间的自动机器学习框架下的神经网络剪枝新方法 

---
# Data-Efficient Psychiatric Disorder Detection via Self-supervised Learning on Frequency-enhanced Brain Networks 

**Title (ZH)**: 基于频率增强脑网络的自我监督学习高效精神障碍检测 

**Authors**: Mujie Liu, Mengchu Zhu, Qichao Dong, Ting Dang, Jiangang Ma, Jing Ren, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.10524)  

**Abstract**: Psychiatric disorders involve complex neural activity changes, with functional magnetic resonance imaging (fMRI) data serving as key diagnostic evidence. However, data scarcity and the diverse nature of fMRI information pose significant challenges. While graph-based self-supervised learning (SSL) methods have shown promise in brain network analysis, they primarily focus on time-domain representations, often overlooking the rich information embedded in the frequency domain. To overcome these limitations, we propose Frequency-Enhanced Network (FENet), a novel SSL framework specially designed for fMRI data that integrates time-domain and frequency-domain information to improve psychiatric disorder detection in small-sample datasets. FENet constructs multi-view brain networks based on the inherent properties of fMRI data, explicitly incorporating frequency information into the learning process of representation. Additionally, it employs domain-specific encoders to capture temporal-spectral characteristics, including an efficient frequency-domain encoder that highlights disease-relevant frequency features. Finally, FENet introduces a domain consistency-guided learning objective, which balances the utilization of diverse information and generates frequency-enhanced brain graph representations. Experiments on two real-world medical datasets demonstrate that FENet outperforms state-of-the-art methods while maintaining strong performance in minimal data conditions. Furthermore, we analyze the correlation between various frequency-domain features and psychiatric disorders, emphasizing the critical role of high-frequency information in disorder detection. 

**Abstract (ZH)**: 基于频率增强的神经网络（FENet）在fMRI数据中的精神病障碍检测 

---
# From Predictions to Explanations: Explainable AI for Autism Diagnosis and Identification of Critical Brain Regions 

**Title (ZH)**: 从预测到解释：自解释人工智能在自闭症诊断及关键脑区识别中的应用 

**Authors**: Kush Gupta, Amir Aly, Emmanuel Ifeachor, Rohit Shankar  

**Link**: [PDF](https://arxiv.org/pdf/2509.10523)  

**Abstract**: Autism spectrum disorder (ASD) is a neurodevelopmental condition characterized by atypical brain maturation. However, the adaptation of transfer learning paradigms in machine learning for ASD research remains notably limited. In this study, we propose a computer-aided diagnostic framework with two modules. This chapter presents a two-module framework combining deep learning and explainable AI for ASD diagnosis. The first module leverages a deep learning model fine-tuned through cross-domain transfer learning for ASD classification. The second module focuses on interpreting the model decisions and identifying critical brain regions. To achieve this, we employed three explainable AI (XAI) techniques: saliency mapping, Gradient-weighted Class Activation Mapping, and SHapley Additive exPlanations (SHAP) analysis. This framework demonstrates that cross-domain transfer learning can effectively address data scarcity in ASD research. In addition, by applying three established explainability techniques, the approach reveals how the model makes diagnostic decisions and identifies brain regions most associated with ASD. These findings were compared against established neurobiological evidence, highlighting strong alignment and reinforcing the clinical relevance of the proposed approach. 

**Abstract (ZH)**: 自闭症谱系 disorder (ASD) 是一种神经发育条件，特征为大脑发育异常。然而，将迁移学习范式应用于机器学习的 ASD 研究仍然明显受限。本研究提出了一种计算机辅助诊断框架，包含两个模块。本章介绍了结合深度学习和可解释人工智能的两模块框架，用于 ASD 诊断。第一个模块利用通过跨域迁移学习调整的深度学习模型进行 ASD 分类。第二个模块专注于解释模型决策并识别关键脑区。为此，我们采用了三种可解释人工智能 (XAI) 技术：梯度调控类激活映射、显著性映射和 SHapley 加权解释分析（SHAP）分析。该框架表明，跨域迁移学习可以有效解决 ASD 研究中的数据稀疏问题。此外，通过应用三种成熟的可解释性技术，该方法揭示了模型如何做出诊断决策，并识别与 ASD 最相关的脑区。这些发现与既定的神经生物学证据进行了比较，突出了强烈的一致性，并强化了所提方法的临床相关性。 

---
# A Comparative Benchmark of Federated Learning Strategies for Mortality Prediction on Heterogeneous and Imbalanced Clinical Data 

**Title (ZH)**: 异质性和不平衡临床数据环境下联邦学习策略的死亡率预测比较基准研究 

**Authors**: Rodrigo Tertulino  

**Link**: [PDF](https://arxiv.org/pdf/2509.10517)  

**Abstract**: Machine learning models hold significant potential for predicting in-hospital mortality, yet data privacy constraints and the statistical heterogeneity of real-world clinical data often hamper their development. Federated Learning (FL) offers a privacy-preserving solution, but its performance under non-Independent and Identically Distributed (non-IID) and imbalanced conditions requires rigorous investigation. The study presents a comparative benchmark of five federated learning strategies: FedAvg, FedProx, FedAdagrad, FedAdam, and FedCluster for mortality prediction. Using the large-scale MIMIC-IV dataset, we simulate a realistic non-IID environment by partitioning data by clinical care unit. To address the inherent class imbalance of the task, the SMOTE-Tomek technique is applied to each client's local training data. Our experiments, conducted over 50 communication rounds, reveal that the regularization-based strategy, FedProx, consistently outperformed other methods, achieving the highest F1-Score of 0.8831 while maintaining stable convergence. While the baseline FedAvg was the most computationally efficient, its predictive performance was substantially lower. Our findings indicate that regularization-based FL algorithms like FedProx offer a more robust and effective solution for heterogeneous and imbalanced clinical prediction tasks than standard or server-side adaptive aggregation methods. The work provides a crucial empirical benchmark for selecting appropriate FL strategies for real-world healthcare applications. 

**Abstract (ZH)**: 联邦学习策略在住院 mortality 预测中的比较研究：基于 FedProx 的正则化方法在非独立同分布和不平衡数据条件下的优越性 

---
# Privacy-Preserving Personalization in Education: A Federated Recommender System for Student Performance Prediction 

**Title (ZH)**: 教育中隐私保护的个性化学习：基于学生的性能预测联邦推荐系统 

**Authors**: Rodrigo Tertulino  

**Link**: [PDF](https://arxiv.org/pdf/2509.10516)  

**Abstract**: The increasing digitalization of education presents unprecedented opportunities for data-driven personalization, yet it introduces significant student data privacy challenges. Conventional recommender systems rely on centralized data, a paradigm often incompatible with modern data protection regulations. A novel privacy-preserving recommender system is proposed and evaluated to address this critical issue using Federated Learning (FL). The approach utilizes a Deep Neural Network (DNN) with rich, engineered features from the large-scale ASSISTments educational dataset. A rigorous comparative analysis of federated aggregation strategies was conducted, identifying FedProx as a significantly more stable and effective method for handling heterogeneous student data than the standard FedAvg baseline. The optimized federated model achieves a high-performance F1-Score of 76.28\%, corresponding to 82.85\% of the performance of a powerful, centralized XGBoost model. These findings validate that a federated approach can provide highly effective content recommendations without centralizing sensitive student data. Consequently, our work presents a viable and robust solution to the personalization-privacy dilemma in modern educational platforms. 

**Abstract (ZH)**: 数字化教育的增加为数据驱动的个性化教学提供了前所未有的机会，但也引入了重大学生数据隐私挑战。传统的推荐系统依赖于集中式数据，这一范式常与现代数据保护法规不兼容。提出并评估了一种基于联邦学习（FL）的新型隐私保护推荐系统，利用深层神经网络（DNN）和大规模ASSISTments教育数据集中的丰富特征来解决这一关键问题。进行了严格的联邦聚合策略对比分析，发现FedProx在处理异构学生数据方面比标准的FedAvg基线更稳定和有效。优化后的联邦模型实现了高性能的F1-Score为76.28%，相当于强大集中式XGBoost模型性能的82.85%。这些发现证明，联邦方法可以在不集中敏感学生数据的情况下提供高效的內容推荐。因此，我们的工作为现代教育平台中的个性化-隐私困境提供了可行且 robust 的解决方案。 

---
# FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification 

**Title (ZH)**: FireGNN：具有可训练模糊规则的神经符号图神经网络及其在可解释医疗图像分类中的应用 

**Authors**: Prajit Sengupta, Islem Rekik  

**Link**: [PDF](https://arxiv.org/pdf/2509.10510)  

**Abstract**: Medical image classification requires not only high predictive performance but also interpretability to ensure clinical trust and adoption. Graph Neural Networks (GNNs) offer a powerful framework for modeling relational structures within datasets; however, standard GNNs often operate as black boxes, limiting transparency and usability, particularly in clinical settings. In this work, we present an interpretable graph-based learning framework named FireGNN that integrates trainable fuzzy rules into GNNs for medical image classification. These rules embed topological descriptors - node degree, clustering coefficient, and label agreement - using learnable thresholds and sharpness parameters to enable intrinsic symbolic reasoning. Additionally, we explore auxiliary self-supervised tasks (e.g., homophily prediction, similarity entropy) as a benchmark to evaluate the contribution of topological learning. Our fuzzy-rule-enhanced model achieves strong performance across five MedMNIST benchmarks and the synthetic dataset MorphoMNIST, while also generating interpretable rule-based explanations. To our knowledge, this is the first integration of trainable fuzzy rules within a GNN. 

**Abstract (ZH)**: 医学图像分类需要不仅具备高性能的预测能力，还需要具备可解释性以确保临床信任与应用。图神经网络（GNNs）提供了一种强大的框架来建模数据集中的关系结构；然而，标准GNNs通常作为黑匣子操作，限制了透明度和实用性，尤其是在临床环境中。在本文中，我们提出了一种名为FireGNN的可解释的图基学习框架，通过将可训练的模糊规则集成到GNNs中，用于医学图像分类。这些规则嵌入拓扑描述符——节点度数、聚类系数和标签一致性——使用可学习的阈值和锐度参数来实现固有的符号推理能力。此外，我们探讨了辅助半监督任务（如同质性预测、相似性熵）作为测试拓扑学习贡献的基准。我们的增强模糊规则模型在五个MedMNIST基准和合成数据集MorphoMNIST上取得了出色的性能，同时生成了可解释的基于规则的解释。据我们所知，这是将可训练的模糊规则集成到GNN中的首次尝试。 

---
# CAR-BRAINet: Sub-6GHz Aided Spatial Adaptive Beam Prediction with Multi Head Attention for Heterogeneous Vehicular Networks 

**Title (ZH)**: CAR-BRAINet: 基于子6GHz辅助空间自适应波束预测的多头注意力异构车辆网络 

**Authors**: Aathira G Menon, Prabu Krishnan, Shyam Lal  

**Link**: [PDF](https://arxiv.org/pdf/2509.10508)  

**Abstract**: Heterogeneous Vehicular Networks (HetVNets) play a key role by stacking different communication technologies such as sub-6GHz, mm-wave and DSRC to meet diverse connectivity needs of 5G/B5G vehicular networks. HetVNet helps address the humongous user demands-but maintaining a steady connection in a highly mobile, real-world conditions remain a challenge. Though there has been ample of studies on beam prediction models a dedicated solution for HetVNets is sparsely explored. Hence, it is the need of the hour to develop a reliable beam prediction solution, specifically for HetVNets. This paper introduces a lightweight deep learning-based solution termed-"CAR-BRAINet" which consists of convolutional neural networks with a powerful multi-head attention (MHA) mechanism. Existing literature on beam prediction is largely studied under a limited, idealised vehicular scenario, often overlooking the real-time complexities and intricacies of vehicular networks. Therefore, this study aims to mimic the complexities of a real-time driving scenario by incorporating key factors such as prominent MAC protocols-3GPP-C-V2X and IEEE 802.11BD, the effect of Doppler shifts under high velocity and varying distance and SNR levels into three high-quality dynamic datasets pertaining to urban, rural and highway vehicular networks. CAR-BRAINet performs effectively across all the vehicular scenarios, demonstrating precise beam prediction with minimal beam overhead and a steady improvement of 17.9422% on the spectral efficiency over the existing methods. Thus, this study justifies the effectiveness of CAR-BRAINet in complex HetVNets, offering promising performance without relying on the location angle and antenna dimensions of the mobile users, and thereby reducing the redundant sensor-latency. 

**Abstract (ZH)**: Heterogeneous 车载网络中的轻量级深度学习beam预测解决方案：CAR-BRAINet 

---
# An Internet of Intelligent Things Framework for Decentralized Heterogeneous Platforms 

**Title (ZH)**: 智能事物互联网框架：去中心化异构平台 

**Authors**: Vadim Allayev, Mahbubur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2509.10507)  

**Abstract**: Internet of Intelligent Things (IoIT), an emerging field, combines the utility of Internet of Things (IoT) devices with the innovation of embedded AI algorithms. However, it does not come without challenges, and struggles regarding available computing resources, energy supply, and storage limitations. In particular, many impediments to IoIT are linked to the energy-efficient deployment of machine learning (ML)/deep learning (DL) models in embedded devices. Research has been conducted to design energy-efficient IoIT platforms, but these papers often focus on centralized systems, in which some central entity processes all the data and coordinates actions. This can be problematic, e.g., serve as bottleneck or lead to security concerns. In a decentralized system, nodes/devices would self-organize and make their own decisions. Therefore, to address such issues, we propose a heterogeneous, decentralized sensing and monitoring IoIT peer-to-peer mesh network system model. Nodes in the network will coordinate towards several optimization goals: reliability, energy efficiency, and latency. The system employs federated learning to train nodes in a distributed manner, metaheuristics to optimize task allocation and routing paths, and multi-objective optimization to balance conflicting performance goals. 

**Abstract (ZH)**: 互联网智能事物(IoIT)：一种新兴领域及其挑战与对策 

---
# Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs 

**Title (ZH)**: 基于树结构MDP中最坏路径策略优化的 retrosynthesis 规划 

**Authors**: Mianchu Wang, Giovanni Montana  

**Link**: [PDF](https://arxiv.org/pdf/2509.10504)  

**Abstract**: Retrosynthesis planning aims to decompose target molecules into available building blocks, forming a synthesis tree where each internal node represents an intermediate compound and each leaf ideally corresponds to a purchasable reactant. However, this tree becomes invalid if any leaf node is not a valid building block, making the planning process vulnerable to the "weakest link" in the synthetic route. Existing methods often optimise for average performance across branches, failing to account for this worst-case sensitivity. In this paper, we reframe retrosynthesis as a worst-path optimisation problem within tree-structured Markov Decision Processes (MDPs). We prove that this formulation admits a unique optimal solution and offers monotonic improvement guarantees. Building on this insight, we introduce Interactive Retrosynthesis Planning (InterRetro), a method that interacts with the tree MDP, learns a value function for worst-path outcomes, and improves its policy through self-imitation, preferentially reinforcing past decisions with high estimated advantage. Empirically, InterRetro achieves state-of-the-art results, solving 100% of targets on the Retro*-190 benchmark, shortening synthetic routes by 4.9%, and achieving promising performance using only 10% of the training data - representing a significant advance in computational retrosynthesis planning. 

**Abstract (ZH)**: 逆合成规划旨在将目标分子分解为可用的构建模块，形成一棵合成树，每个内部节点代表一个中间化合物，每个叶子节点理想地对应于可购买的反应物。然而，如果任何叶子节点不是一个有效的构建模块，这棵树就会失效，使得规划过程容易受到合成路线中最弱环节的影响。现有方法通常优化分支的平均性能，未能考虑最坏情况的敏感性。本文将逆合成规划重新构想为树结构马尔可夫决策过程（MDP）中的最坏路径优化问题。我们证明这种表述存在一个唯一的最优解，并提供单调改进保证。基于这一洞察，我们引入了交互式逆合成规划（InterRetro）方法，该方法与树结构MDP相互作用，学习最坏路径结果的价值函数，并通过自我模仿提升其策略，优先强化高估计优势的以往决策。实验证明，InterRetro达到了最先进的成果，在Retro*-190基准上解决了100%的目标，缩短了合成路线4.9%，并仅使用10%的训练数据就取得了令人鼓舞的性能，这代表了计算逆合成规划的一个显著进展。 

---
# FEDEXCHANGE: Bridging the Domain Gap in Federated Object Detection for Free 

**Title (ZH)**: FEDEXCHANGE：跨域 federated 物体检测中的领域差距桥梁构建 

**Authors**: Haolin Yuan, Jingtao Li, Weiming Zhuang, Chen Chen, Lingjuan Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10503)  

**Abstract**: Federated Object Detection (FOD) enables clients to collaboratively train a global object detection model without accessing their local data from diverse domains. However, significant variations in environment, weather, and other domain specific factors hinder performance, making cross domain generalization a key challenge. Existing FOD methods often overlook the hardware constraints of edge devices and introduce local training regularizations that incur high computational costs, limiting real-world applicability. In this paper, we propose FEDEXCHANGE, a novel FOD framework that bridges domain gaps without introducing additional local computational overhead. FEDEXCHANGE employs a server side dynamic model exchange strategy that enables each client to gain insights from other clients' domain data without direct data sharing. Specifically, FEDEXCHANGE allows the server to alternate between model aggregation and model exchange. During aggregation rounds, the server aggregates all local models as usual. In exchange rounds, FEDEXCHANGE clusters and exchanges local models based on distance measures, allowing local models to learn from a variety of domains. As all operations are performed on the server side, clients can achieve improved cross domain utility without any additional computational overhead. Extensive evaluations demonstrate that FEDEXCHANGE enhances FOD performance, achieving 1.6X better mean average precision in challenging domains, such as rainy conditions, while requiring only 0.8X the computational resources compared to baseline methods. 

**Abstract (ZH)**: 联邦目标检测 (Federated Object Detection, FOD) 允许客户端在不访问本地多域数据的情况下协同训练一个全局目标检测模型。然而，环境、天气以及其他领域特定因素的重大差异阻碍了性能提升，使得跨域泛化成为主要挑战。现有的 FOD 方法往往忽视了边缘设备的硬件限制并引入了高计算成本的局部训练正则化，限制了其实用性。在本文中，我们提出了一种名为 FEDEXCHANGE 的新颖 FOD 框架，该框架在不增加额外局部计算开销的情况下弥合领域差异。FEDEXCHANGE 采用服务器端动态模型交换策略，使每个客户端能够从其他客户端的领域数据中获得见解，而无需直接共享数据。具体而言，FEDEXCHANGE 允许服务器在模型聚合和模型交换之间交替。在聚合轮次中，服务器按常规聚合所有本地模型。在交换轮次中，FEDEXCHANGE 基于距离度量对齐并交换本地模型，使得局部模型能够学习来自多种领域的知识。由于所有操作都在服务器端执行，客户端可以在没有任何额外计算开销的情况下实现跨域性能改进。广泛评估表明，FEDEXCHANGE 提升了 FOD 性能，在恶劣天气等挑战性领域中实现了 1.6 倍的平均精度，同时仅需基线方法 0.8 倍的计算资源。 

---
# From Noise to Precision: A Diffusion-Driven Approach to Zero-Inflated Precipitation Prediction 

**Title (ZH)**: 从噪声到精准：零 inflated 降水预测的扩散驱动方法 

**Authors**: Wentao Gao, Jiuyong Li, Lin Liu, Thuc Duy Le, Xiongren Chen, Xiaojing Du, Jixue Liu, Yanchang Zhao, Yun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.10501)  

**Abstract**: Zero-inflated data pose significant challenges in precipitation forecasting due to the predominance of zeros with sparse non-zero events. To address this, we propose the Zero Inflation Diffusion Framework (ZIDF), which integrates Gaussian perturbation for smoothing zero-inflated distributions, Transformer-based prediction for capturing temporal patterns, and diffusion-based denoising to restore the original data structure. In our experiments, we use observational precipitation data collected from South Australia along with synthetically generated zero-inflated data. Results show that ZIDF demonstrates significant performance improvements over multiple state-of-the-art precipitation forecasting models, achieving up to 56.7\% reduction in MSE and 21.1\% reduction in MAE relative to the baseline Non-stationary Transformer. These findings highlight ZIDF's ability to robustly handle sparse time series data and suggest its potential generalizability to other domains where zero inflation is a key challenge. 

**Abstract (ZH)**: Zero-Inflated 数据在降水量预报中占主导地位且非零事件稀疏，这对预报构成重大挑战。为应对这一挑战，我们提出了零膨胀扩散框架（ZIDF），该框架结合了高斯扰动用于平滑零膨胀分布、基于变压器的预测用于捕捉时间模式，以及基于扩散的去噪以恢复原始数据结构。实验中，我们使用来自南澳大利亚的观测降水量数据以及合成的零膨胀数据。结果表明，ZIDF 在多个最先进的降水量预报模型中表现出显著的性能提升，相对于非stationary变压器 baseline，MSE 减少了高达 56.7%，MAE 减少了 21.1%。这些发现突显了 ZIDF 在处理稀疏时间序列数据方面的鲁棒性，并暗示其在其他领域中的通用性，尤其是在零膨胀是一个关键挑战的领域中。 

---
# Towards Scalable O-RAN Resource Management: Graph-Augmented Proximal Policy Optimization 

**Title (ZH)**: 面向可扩展的O-RAN资源管理：图增强邻近策略优化 

**Authors**: Duc-Thinh Ngo, Kandaraj Piamrat, Ons Aouedi, Thomas Hassan, Philippe Raipin-Parvédy  

**Link**: [PDF](https://arxiv.org/pdf/2509.10499)  

**Abstract**: Open Radio Access Network (O-RAN) architectures enable flexible, scalable, and cost-efficient mobile networks by disaggregating and virtualizing baseband functions. However, this flexibility introduces significant challenges for resource management, requiring joint optimization of functional split selection and virtualized unit placement under dynamic demands and complex topologies. Existing solutions often address these aspects separately or lack scalability in large and real-world scenarios. In this work, we propose a novel Graph-Augmented Proximal Policy Optimization (GPPO) framework that leverages Graph Neural Networks (GNNs) for topology-aware feature extraction and integrates action masking to efficiently navigate the combinatorial decision space. Our approach jointly optimizes functional split and placement decisions, capturing the full complexity of O-RAN resource allocation. Extensive experiments on both small-and large-scale O-RAN scenarios demonstrate that GPPO consistently outperforms state-of-the-art baselines, achieving up to 18% lower deployment cost and 25% higher reward in generalization tests, while maintaining perfect reliability. These results highlight the effectiveness and scalability of GPPO for practical O-RAN deployments. 

**Abstract (ZH)**: 基于图增强近端策略优化的O-RAN资源管理方法 

---
# Online Learning Based Efficient Resource Allocation for LoRaWAN Network 

**Title (ZH)**: 基于在线学习的LoRaWAN网络高效资源分配 

**Authors**: Ruiqi Wang, Jing Ren, Tongyu Song, Wenjun Li, Xiong Wang, Sheng Wang, Shizhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10493)  

**Abstract**: The deployment of large-scale LoRaWAN networks requires jointly optimizing conflicting metrics like Packet Delivery Ratio (PDR) and Energy Efficiency (EE) by dynamically allocating transmission parameters, including Carrier Frequency, Spreading Factor, and Transmission Power. Existing methods often oversimplify this challenge, focusing on a single metric or lacking the adaptability needed for dynamic channel environments, leading to suboptimal performance. To address this, we propose two online learning-based resource allocation frameworks that intelligently navigate the PDR-EE trade-off. Our foundational proposal, D-LoRa, is a fully distributed framework that models the problem as a Combinatorial Multi-Armed Bandit. By decomposing the joint parameter selection and employing specialized, disaggregated reward functions, D-LoRa dramatically reduces learning complexity and enables nodes to autonomously adapt to network dynamics. To further enhance performance in LoRaWAN networks, we introduce CD-LoRa, a hybrid framework that integrates a lightweight, centralized initialization phase to perform a one-time, quasi-optimal channel assignment and action space pruning, thereby accelerating subsequent distributed learning. Extensive simulations and real-world field experiments demonstrate the superiority of our frameworks, showing that D-LoRa excels in non-stationary environments while CD-LoRa achieves the fastest convergence in stationary conditions. In physical deployments, our methods outperform state-of-the-art baselines, improving PDR by up to 10.8% and EE by 26.1%, confirming their practical effectiveness for scalable and efficient LoRaWAN networks. 

**Abstract (ZH)**: 大规模LoRaWAN网络的部署需要通过动态分配传输参数（包括载波频率、扩频因子和传输功率）来同时优化如包送达率（PDR）和能量效率（EE）等冲突的指标。现有的方法往往过于简化这一挑战，要么关注单一指标，要么缺乏适应动态信道环境的能力，导致性能不佳。为此，我们提出了两种基于在线学习的资源分配框架，智能化地平衡PDR-EEtrade-off。我们的基础提案D-LoRa是一个完全分布式框架，将其问题建模为组合多臂 bandit 问题。通过分解联合参数选择，并采用专业化、分而治之的奖励函数，D-LoRa明显降低了学习复杂性，并使节点能够自主适应网络动态。为了进一步提升LoRaWAN网络的性能，我们引入了CD-LoRa，这是一种混合框架，集成了一个轻量级的集中式初始化阶段，进行一次性、近似最优的信道分配和动作空间修剪，从而加速后续的分布式学习。通过广泛的仿真实验和实地测试，我们的框架显示出优越性，D-LoRa在非平稳环境中表现出色，而CD-LoRa在稳定条件下实现最快收敛。在物理部署中，我们的方法优于最先进的基线，PDR 提高最多10.8%，EE 提高26.1%，确认了其在可扩展和高效LoRaWAN网络中的实际效果。 

---
# Distributed Gossip-GAN for Low-overhead CSI Feedback Training in FDD mMIMO-OFDM Systems 

**Title (ZH)**: 分布式 gossip-GAN 用于 FDD mMIMO-OFDM 系统的低开销CSI反馈训练 

**Authors**: Yuwen Cao, Guijun Liu, Tomoaki Ohtsuki, Howard H. Yang, Tony Q. S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2509.10490)  

**Abstract**: The deep autoencoder (DAE) framework has turned out to be efficient in reducing the channel state information (CSI) feedback overhead in massive multiple-input multipleoutput (mMIMO) systems. However, these DAE approaches presented in prior works rely heavily on large-scale data collected through the base station (BS) for model training, thus rendering excessive bandwidth usage and data privacy issues, particularly for mMIMO systems. When considering users' mobility and encountering new channel environments, the existing CSI feedback models may often need to be retrained. Returning back to previous environments, however, will make these models perform poorly and face the risk of catastrophic forgetting. To solve the above challenging problems, we propose a novel gossiping generative adversarial network (Gossip-GAN)-aided CSI feedback training framework. Notably, Gossip-GAN enables the CSI feedback training with low-overhead while preserving users' privacy. Specially, each user collects a small amount of data to train a GAN model. Meanwhile, a fully distributed gossip-learning strategy is exploited to avoid model overfitting, and to accelerate the model training as well. Simulation results demonstrate that Gossip-GAN can i) achieve a similar CSI feedback accuracy as centralized training with real-world datasets, ii) address catastrophic forgetting challenges in mobile scenarios, and iii) greatly reduce the uplink bandwidth usage. Besides, our results show that the proposed approach possesses an inherent robustness. 

**Abstract (ZH)**: 基于Gossip-GAN的低开销 CSI 反馈训练框架 

---
# SABR: A Stable Adaptive Bitrate Framework Using Behavior Cloning Pretraining and Reinforcement Learning Fine-Tuning 

**Title (ZH)**: SABR：一种基于行为克隆预训练和强化学习微调的稳定自适应比特率框架 

**Authors**: Pengcheng Luo, Yunyang Zhao, Bowen Zhang, Genke Yang, Boon-Hee Soong, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2509.10486)  

**Abstract**: With the advent of 5G, the internet has entered a new video-centric era. From short-video platforms like TikTok to long-video platforms like Bilibili, online video services are reshaping user consumption habits. Adaptive Bitrate (ABR) control is widely recognized as a critical factor influencing Quality of Experience (QoE). Recent learning-based ABR methods have attracted increasing attention. However, most of them rely on limited network trace sets during training and overlook the wide-distribution characteristics of real-world network conditions, resulting in poor generalization in out-of-distribution (OOD) scenarios. To address this limitation, we propose SABR, a training framework that combines behavior cloning (BC) pretraining with reinforcement learning (RL) fine-tuning. We also introduce benchmarks, ABRBench-3G and ABRBench-4G+, which provide wide-coverage training traces and dedicated OOD test sets for assessing robustness to unseen network conditions. Experimental results demonstrate that SABR achieves the best average rank compared with Pensieve, Comyco, and NetLLM across the proposed benchmarks. These results indicate that SABR enables more stable learning across wide distributions and improves generalization to unseen network conditions. 

**Abstract (ZH)**: 随着5G的到来，互联网进入了一个以视频为中心的新时代。从TikTok这样的短视频平台到Bilibili这样的长视频平台，在线视频服务正在重塑用户的消费习惯。自适应比特率（ABR）控制被认为是影响用户体验质量（QoE）的关键因素。近期基于学习的ABR方法引起了越来越多的关注。然而，大多数方法在训练过程中依赖于有限的网络追踪集，并且忽略了实际网络条件的广泛分布特性，导致在未见过的分布（OOD）场景下泛化能力较差。为了解决这一局限，我们提出了一种名为SABR的训练框架，该框架结合了行为克隆（BC）预训练与强化学习（RL）微调。我们还引入了ABRBench-3G和ABRBench-4G+基准测试，提供了广泛的训练轨迹和专门的ODD测试集，用于评估对未见过的网络条件的鲁棒性。实验结果表明，在提出的基准测试中，SABR在与Pensieve、Comyco和NetLLM相比时，平均排名最高。这些结果表明，SABR能够在广泛的分布下实现更稳定的训练，并改善对未见过的网络条件的泛化能力。 

---
# AegisShield: Democratizing Cyber Threat Modeling with Generative AI 

**Title (ZH)**: AegisShield: 通过生成式AI普及网络威胁建模 

**Authors**: Matthew Grofsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.10482)  

**Abstract**: The increasing sophistication of technology systems makes traditional threat modeling hard to scale, especially for small organizations with limited resources. This paper develops and evaluates AegisShield, a generative AI enhanced threat modeling tool that implements STRIDE and MITRE ATT&CK to automate threat generation and provide systematic assessments. By integrating real time threat intelligence from the National Vulnerability Database and AlienVault Open Threat Exchange, AegisShield produces streamlined and accessible threat descriptions. Our assessment of 243 threats from 15 case studies and over 8000 AI generated threats shows that AegisShield reduces complexity (p less than 0.001), yields outputs semantically aligned with expert developed threats (p less than 0.05), and achieves an 85.4 percent success rate in mapping threats to MITRE ATT&CK techniques (p less than 0.001). Automating and standardizing threat modeling helps under resourced organizations address risk earlier and supports wider adoption of secure by design practices. 

**Abstract (ZH)**: 技术系统的日益复杂性使得传统的威胁建模难以推广，尤其是对于资源有限的小组织。本文开发并评估了AegisShield，这是一种增强型生成AI威胁建模工具，采用STRIDE和MITRE ATT&CK来自动化威胁生成并提供系统性的评估。通过整合来自National Vulnerability Database和AlienVault Open Threat Exchange的实时威胁情报，AegisShield生成了简化且易于访问的威胁描述。对来自15个案例研究的243种威胁和超过8000种AI生成的威胁的评估显示，AegisShield降低了复杂性（p<0.001）、生成的输出与专家开发的威胁在语义上保持一致（p<0.05），并且在将威胁映射到MITRE ATT&CK技术方面达到了85.4%的成功率（p<0.001）。自动化和标准化的威胁建模有助于资源不足的组织更早地应对风险，并支持“设计即安全”的实践更广泛的采用。 

---
# Real-Time RAG for the Identification of Supply Chain Vulnerabilities 

**Title (ZH)**: 实时RAG在识别供应链漏洞中的应用 

**Authors**: Jesse Ponnock, Grace Kenneally, Michael Robert Briggs, Elinor Yeo, Tyrone Patterson III, Nicholas Kinberg, Matthew Kalinowski, David Hechtman  

**Link**: [PDF](https://arxiv.org/pdf/2509.10469)  

**Abstract**: New technologies in generative AI can enable deeper analysis into our nation's supply chains but truly informative insights require the continual updating and aggregation of massive data in a timely manner. Large Language Models (LLMs) offer unprecedented analytical opportunities however, their knowledge base is constrained to the models' last training date, rendering these capabilities unusable for organizations whose mission impacts rely on emerging and timely information. This research proposes an innovative approach to supply chain analysis by integrating emerging Retrieval-Augmented Generation (RAG) preprocessing and retrieval techniques with advanced web-scraping technologies. Our method aims to reduce latency in incorporating new information into an augmented-LLM, enabling timely analysis of supply chain disruptors. Through experimentation, this study evaluates the combinatorial effects of these techniques towards timeliness and quality trade-offs. Our results suggest that in applying RAG systems to supply chain analysis, fine-tuning the embedding retrieval model consistently provides the most significant performance gains, underscoring the critical importance of retrieval quality. Adaptive iterative retrieval, which dynamically adjusts retrieval depth based on context, further enhances performance, especially on complex sup- ply chain queries. Conversely, fine-tuning the LLM yields limited improvements and higher resource costs, while techniques such as downward query abstraction significantly outperforms upward abstraction in practice. 

**Abstract (ZH)**: 新兴生成AI技术可以深入分析我国的供应链，但真正具有信息价值的见解需要及时更新和聚合大量数据。大型语言模型（LLMs）提供了前所未有的分析机会，然而，它们的知识库仅限于模型最后一次训练的日期，使得依赖于及时信息的组织无法使用这些功能。本研究提出了一种创新的供应链分析方法，即将新兴的检索增强生成（RAG）预处理和检索技术与先进的网络抓取技术结合。该方法旨在减少将新信息纳入增强LLM中的延迟，从而实现对供应链中断因素的及时分析。通过实验，本研究评估了这些技术组合对及时性和质量权衡的影响。结果显示，在将RAG系统应用于供应链分析时，持续优化嵌入式检索模型提供了显著的性能改进，突显了检索质量的至关重要性。动态迭代检索（根据上下文动态调整检索深度）进一步提升了性能，尤其是在复杂的供应链查询中。相反，微调LLM仅提供了有限的改进并增加了更多的资源成本，而实际应用中向下抽象查询的效果显著优于向上抽象。 

---
# Momentum-integrated Multi-task Stock Recommendation with Converge-based Optimization 

**Title (ZH)**: 基于收敛优化的动量集成多任务股票推荐 

**Authors**: Hao Wang, Jingshu Peng, Yanyan Shen, Xujia Li, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.10461)  

**Abstract**: Stock recommendation is critical in Fintech applications, which use price series and alternative information to estimate future stock performance. Although deep learning models are prevalent in stock recommendation systems, traditional time-series forecasting training often fails to capture stock trends and rankings simultaneously, which are essential consideration factors for investors. To tackle this issue, we introduce a Multi-Task Learning (MTL) framework for stock recommendation, \textbf{M}omentum-\textbf{i}ntegrated \textbf{M}ulti-task \textbf{Stoc}k \textbf{R}ecommendation with Converge-based Optimization (\textbf{MiM-StocR}). To improve the model's ability to capture short-term trends, we novelly invoke a momentum line indicator in model training. To prioritize top-performing stocks and optimize investment allocation, we propose a list-wise ranking loss function called Adaptive-k ApproxNDCG. Moreover, due to the volatility and uncertainty of the stock market, existing MTL frameworks face overfitting issues when applied to stock time series. To mitigate this issue, we introduce the Converge-based Quad-Balancing (CQB) method. We conducted extensive experiments on three stock benchmarks: SEE50, CSI 100, and CSI 300. MiM-StocR outperforms state-of-the-art MTL baselines across both ranking and profitable evaluations. 

**Abstract (ZH)**: 基于动量整合的多任务学习股票推荐模型MiM-StocR 

---
# Program Skeletons for Automated Program Translation 

**Title (ZH)**: 自动程序翻译的程序骨架 

**Authors**: Bo Wang, Tianyu Li, Ruishi Li, Umang Mathur, Prateek Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2504.07483)  

**Abstract**: Translating software between programming languages is a challenging task, for which automated techniques have been elusive and hard to scale up to larger programs. A key difficulty in cross-language translation is that one has to re-express the intended behavior of the source program into idiomatic constructs of a different target language. This task needs abstracting away from the source language-specific details, while keeping the overall functionality the same. In this work, we propose a novel and systematic approach for making such translation amenable to automation based on a framework we call program skeletons. A program skeleton retains the high-level structure of the source program by abstracting away and effectively summarizing lower-level concrete code fragments, which can be mechanically translated to the target programming language. A skeleton, by design, permits many different ways of filling in the concrete implementation for fragments, which can work in conjunction with existing data-driven code synthesizers. Most importantly, skeletons can conceptually enable sound decomposition, i.e., if each individual fragment is correctly translated, taken together with the mechanically translated skeleton, the final translated program is deemed to be correct as a whole. We present a prototype system called Skel embodying the idea of skeleton-based translation from Python to JavaScript. Our results show promising scalability compared to prior works. For 9 real-world Python programs, some with more than about 1k lines of code, 95% of their code fragments can be automatically translated, while about 5% require manual effort. All the final translations are correct with respect to whole-program test suites. 

**Abstract (ZH)**: 基于程序框架的跨语言自动化翻译：从Python到JavaScript 

---
