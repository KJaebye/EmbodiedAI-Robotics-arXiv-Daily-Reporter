# Rapid Modeling Architecture for Lightweight Simulator to Accelerate and Improve Decision Making for Industrial Systems 

**Title (ZH)**: 轻量化模拟器的快速建模架构以加速并改善工业系统的决策制定 

**Authors**: Takumi Kato, Zhi Li Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17990)  

**Abstract**: Designing industrial systems, such as building, improving, and automating distribution centers and manufacturing plants, involves critical decision-making with limited information in the early phases. The lack of information leads to less accurate designs of the systems, which are often difficult to resolve later. It is effective to use simulators to model the designed system and find out the issues early. However, the modeling time required by conventional simulators is too long to allow for rapid model creation to meet decision-making demands. In this paper, we propose a Rapid Modeling Architecture (RMA) for a lightweight industrial simulator that mitigates the modeling burden while maintaining the essential details in order to accelerate and improve decision-making. We have prototyped a simulator based on the RMA and applied it to the actual factory layout design problem. We also compared the modeling time of our simulator to that of an existing simulator, and as a result, our simulator achieved a 78.3% reduction in modeling time compared to conventional simulators. 

**Abstract (ZH)**: 基于快速建模架构的轻量级工业仿真器设计与应用 

---
# Automated Brake Onset Detection in Naturalistic Driving Data 

**Title (ZH)**: 自然驾驶数据中自动刹车起始检测 

**Authors**: Shu-Yuan Liu, Johan Engström, Gustav Markkula  

**Link**: [PDF](https://arxiv.org/pdf/2507.17943)  

**Abstract**: Response timing measures play a crucial role in the assessment of automated driving systems (ADS) in collision avoidance scenarios, including but not limited to establishing human benchmarks and comparing ADS to human driver response performance. For example, measuring the response time (of a human driver or ADS) to a conflict requires the determination of a stimulus onset and a response onset. In existing studies, response onset relies on manual annotation or vehicle control signals such as accelerator and brake pedal movements. These methods are not applicable when analyzing large scale data where vehicle control signals are not available. This holds in particular for the rapidly expanding sets of ADS log data where the behavior of surrounding road users is observed via onboard sensors. To advance evaluation techniques for ADS and enable measuring response timing when vehicle control signals are not available, we developed a simple and efficient algorithm, based on a piecewise linear acceleration model, to automatically estimate brake onset that can be applied to any type of driving data that includes vehicle longitudinal time series data. We also proposed a manual annotation method to identify brake onset and used it as ground truth for validation. R2 was used as a confidence metric to measure the accuracy of the algorithm, and its classification performance was analyzed using naturalistic collision avoidance data of both ADS and humans, where our method was validated against human manual annotation. Although our algorithm is subject to certain limitations, it is efficient, generalizable, applicable to any road user and scenario types, and is highly configurable. 

**Abstract (ZH)**: 自动驾驶系统在碰撞避免场景中的响应时间评估在自动驾驶系统的评估中扮演着关键角色，包括但不限于建立人类基准和将自动驾驶系统与人类驾驶员的响应性能进行比较。例如，测量人类驾驶员或自动驾驶系统对冲突的响应时间需要确定刺激的开始和响应的开始。在现有研究中，响应的开始依赖于人工标注或车辆控制信号，如油门和刹车踏板的动作。当分析大规模数据且没有车辆控制信号可用时，这些方法并不适用。特别地，对于通过车载传感器观察周围道路使用者行为的快速扩增的自动驾驶日志数据集，这一点尤为成立。为了推进自动驾驶系统的评估技术并能够在没有车辆控制信号的情况下测量响应时间，我们基于分段线性加速度模型开发了一个简单且高效的算法，以自动估计刹车开始时间，并且该算法可以应用于包含车辆纵向时间序列数据的任何类型驾驶数据。我们还提出了一种人工标注方法来识别刹车开始时间，并将其作为验证的金标准。我们使用决定系数R²作为置信度指标来衡量算法的准确性，并使用自然驾驶环境中的碰撞避免数据（包括人类和自动驾驶系统的数据）来分析其分类性能，我们的方法与人类手动标注进行了验证。尽管我们的算法存在一定的局限性，但它具有高效性、通用性、适用于任何道路用户和场景类型，并且具有高度的可配置性。 

---
# On the Performance of Concept Probing: The Influence of the Data (Extended Version) 

**Title (ZH)**: 概念探查的性能：数据的影响（扩展版） 

**Authors**: Manuel de Sousa Ribeiro, Afonso Leote, João Leite  

**Link**: [PDF](https://arxiv.org/pdf/2507.18550)  

**Abstract**: Concept probing has recently garnered increasing interest as a way to help interpret artificial neural networks, dealing both with their typically large size and their subsymbolic nature, which ultimately renders them unfeasible for direct human interpretation. Concept probing works by training additional classifiers to map the internal representations of a model into human-defined concepts of interest, thus allowing humans to peek inside artificial neural networks. Research on concept probing has mainly focused on the model being probed or the probing model itself, paying limited attention to the data required to train such probing models. In this paper, we address this gap. Focusing on concept probing in the context of image classification tasks, we investigate the effect of the data used to train probing models on their performance. We also make available concept labels for two widely used datasets. 

**Abstract (ZH)**: 概念探针近年来引起了越来越多的关注，作为一种帮助解释人工神经网络的方法，它既处理了神经网络通常较大的规模，也处理了它们的次符号本质，这最终使其直接供人类解释变得不切实际。概念探针通过训练附加分类器将模型的内部表示映射到人类定义的概念中，从而使人类能够窥视人工神经网络的内部。关于概念探针的研究主要集中在被探针的模型或探针模型本身，对用于训练这些探针模型的数据关注较少。在本文中，我们解决了这一差距。聚焦于图像分类任务中概念探针的应用，我们探讨了用于训练探针模型的数据对其性能的影响。我们还为两个广泛使用的数据集提供了概念标签。 

---
# GPU Accelerated Compact-Table Propagation 

**Title (ZH)**: GPU 加速紧凑表传播 

**Authors**: Enrico Santi, Fabio Tardivo, Agostino Dovier, Andrea Formisano  

**Link**: [PDF](https://arxiv.org/pdf/2507.18413)  

**Abstract**: Constraint Programming developed within Logic Programming in the Eighties; nowadays all Prolog systems encompass modules capable of handling constraint programming on finite domains demanding their solution to a constraint solver. This work focuses on a specific form of constraint, the so-called table constraint, used to specify conditions on the values of variables as an enumeration of alternative options. Since every condition on a set of finite domain variables can be ultimately expressed as a finite set of cases, Table can, in principle, simulate any other constraint. These characteristics make Table one of the most studied constraints ever, leading to a series of increasingly efficient propagation algorithms. Despite this, it is not uncommon to encounter real-world problems with hundreds or thousands of valid cases that are simply too many to be handled effectively with standard CPU-based approaches. In this paper, we deal with the Compact-Table (CT) algorithm, the state-of-the-art propagation algorithms for Table. We describe how CT can be enhanced by exploiting the massive computational power offered by modern GPUs to handle large Table constraints. In particular, we report on the design and implementation of GPU-accelerated CT, on its integration into an existing constraint solver, and on an experimental validation performed on a significant set of instances. 

**Abstract (ZH)**: 约束编程在八十年代发展于逻辑编程之中；如今，所有的Prolog系统都包含了能够处理有限域上约束编程的模块，要求将其解决方案交给约束求解器。本文专注于一类特定的约束——所谓的表约束，用于规定变量值的条件，以列举备选方案的形式表示。由于有限域变量集上的任何条件最终都可以表示为有限的几种情况，因此在原则上，表可以模拟任何其他约束。这些特性使得表成为了迄今为止最受研究的约束之一，导致了一系列越来越高效的传播算法的出现。尽管如此，仍然不罕见遇到具有成百上千个有效案例的实际问题，这些案例对于基于标准CPU的方法来说过多且无效。本文探讨了最先进的表传播算法Compact-Table (CT)，并说明了如何通过利用现代GPU提供的巨大计算能力来增强CT，以处理大规模的表约束。特别地，本文描述了GPU加速CT的设计与实现、将其整合到现有约束求解器中以及在大量实例上进行的实验验证。 

---
# Optimising Call Centre Operations using Reinforcement Learning: Value Iteration versus Proximal Policy Optimisation 

**Title (ZH)**: 使用强化学习优化呼叫中心运营：值迭代与接近策略优化的比较 

**Authors**: Kwong Ho Li, Wathsala Karunarathne  

**Link**: [PDF](https://arxiv.org/pdf/2507.18398)  

**Abstract**: This paper investigates the application of Reinforcement Learning (RL) to optimise call routing in call centres to minimise client waiting time and staff idle time. Two methods are compared: a model-based approach using Value Iteration (VI) under known system dynamics, and a model-free approach using Proximal Policy Optimisation (PPO) that learns from experience. For the model-based approach, a theoretical model is used, while a simulation model combining Discrete Event Simulation (DES) with the OpenAI Gym environment is developed for model-free learning. Both models frame the problem as a Markov Decision Process (MDP) within a Skills-Based Routing (SBR) framework, with Poisson client arrivals and exponentially distributed service and abandonment times. For policy evaluation, random, VI, and PPO policies are evaluated using the simulation model. After 1,000 test episodes, PPO consistently achives the highest rewards, along with the lowest client waiting time and staff idle time, despite requiring longer training time. 

**Abstract (ZH)**: 本文研究强化学习在呼叫中心呼叫路由优化中的应用，以最小化客户等待时间和员工空闲时间。两种方法进行比较：基于模型的方法使用值迭代（VI）在已知系统动力学条件下进行优化，以及基于经验的方法使用proximal策略优化（PPO）。对于基于模型的方法，使用理论模型；对于基于经验的方法，开发了一个结合离散事件仿真（DES）和OpenAI Gym环境的仿真模型进行学习。两种方法都将问题框架化为基于技能的路由（SBR）框架中的马尔可夫决策过程（MDP），其中客户到达服从泊松分布，服务时间和放弃时间服从指数分布。在策略评估中，使用仿真模型评估随机策略、VI策略和PPO策略。经过1000个测试时段后，尽管PPO需要更长的训练时间，但PPO始终获得最高奖励，并且客户等待时间和员工空闲时间最低。 

---
# The AlphaPhysics Term Rewriting System for Marking Algebraic Expressions in Physics Exams 

**Title (ZH)**: AlphaPhysics术语重写系统在物理考试中标记代数表达式 

**Authors**: Peter Baumgartner, Lachlan McGinness  

**Link**: [PDF](https://arxiv.org/pdf/2507.18337)  

**Abstract**: We present our method for automatically marking Physics exams. The marking problem consists in assessing typed student answers for correctness with respect to a ground truth solution. This is a challenging problem that we seek to tackle using a combination of a computer algebra system, an SMT solver and a term rewriting system. A Large Language Model is used to interpret and remove errors from student responses and rewrite these in a machine readable format. Once formalized and language-aligned, the next step then consists in applying automated reasoning techniques for assessing student solution correctness. We consider two methods of automated theorem proving: off-the-shelf SMT solving and term rewriting systems tailored for physics problems involving trigonometric expressions. The development of the term rewrite system and establishing termination and confluence properties was not trivial, and we describe it in some detail in the paper. We evaluate our system on a rich pool of over 1500 real-world student exam responses from the 2023 Australian Physics Olympiad. 

**Abstract (ZH)**: 我们提出了一种自动标记物理考试的方法。评分问题在于评估学生回答与标准答案的正确性。这是一个具有挑战性的问题，我们希望通过结合计算机代数系统、SMT求解器和术语重写系统来解决这一问题。大规模语言模型用于解释和纠正学生的回答错误，并将其转换为机器可读格式。一旦形式化和语言对齐，下一步则是应用自动推理技术来评估学生解决方案的正确性。我们考虑了两种自动定理证明方法：现成的SMT求解和为涉及三角表达式的物理问题量身定制的术语重写系统。术语重写系统的开发以及确定其终止性和会聚性性质并不简单，我们在论文中对此进行了详细描述。我们对该系统进行了评估，使用了来自2023年澳大利亚物理奥林匹克竞赛的超过1500份真实学生考试回答。 

---
# Foundations for Risk Assessment of AI in Protecting Fundamental Rights 

**Title (ZH)**: AI在保护基本权利中的风险评估基础 

**Authors**: Antonino Rotolo, Beatrice Ferrigno, Jose Miguel Angel Garcia Godinez, Claudio Novelli, Giovanni Sartor  

**Link**: [PDF](https://arxiv.org/pdf/2507.18290)  

**Abstract**: This chapter introduces a conceptual framework for qualitative risk assessment of AI, particularly in the context of the EU AI Act. The framework addresses the complexities of legal compliance and fundamental rights protection by itegrating definitional balancing and defeasible reasoning. Definitional balancing employs proportionality analysis to resolve conflicts between competing rights, while defeasible reasoning accommodates the dynamic nature of legal decision-making. Our approach stresses the need for an analysis of AI deployment scenarios and for identifying potential legal violations and multi-layered impacts on fundamental rights. On the basis of this analysis, we provide philosophical foundations for a logical account of AI risk analysis. In particular, we consider the basic building blocks for conceptually grasping the interaction between AI deployment scenarios and fundamental rights, incorporating in defeasible reasoning definitional balancing and arguments about the contextual promotion or demotion of rights. This layered approach allows for more operative models of assessment of both high-risk AI systems and General Purpose AI (GPAI) systems, emphasizing the broader applicability of the latter. Future work aims to develop a formal model and effective algorithms to enhance AI risk assessment, bridging theoretical insights with practical applications to support responsible AI governance. 

**Abstract (ZH)**: 本章介绍了一种欧盟AI法案背景下人工智能定性风险评估的概念框架。该框架通过整合定义性平衡和可撤销推理来应对法律合规性和基本权利保护的复杂性。定义性平衡采用比例分析来解决冲突权利之间的冲突，而可撤销推理则承认法律决策的动态性质。我们的方法强调对人工智能部署场景进行分析的必要性，并识别潜在的法律违规行为和多层次的基本权利影响。基于这些分析，我们为一种逻辑的人工智能风险分析提供了哲学基础。特别地，我们考虑了概念上理解人工智能部署场景与基本权利之间相互作用的基本构建模块，并在可撤销推理中整合定义性平衡以及关于权利在特定情境下的促进或贬抑的论述。多层次的方法允许对高风险人工智能系统和通用人工智能系统（GPAI）进行更有操作性的评估模型，突出了后者的更广泛适用性。未来工作旨在发展正式模型和有效算法以增强人工智能风险评估，将理论洞察与实际应用相结合，以支持负责任的人工智能治理。 

---
# Comparing Non-minimal Semantics for Disjunction in Answer Set Programming 

**Title (ZH)**: 比较答案集编程中析取的非最小语义 

**Authors**: Felicidad Aguado, Pedro Cabalar, Brais Muñiz, Gilberto Pérez, Concepción Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2507.18198)  

**Abstract**: In this paper, we compare four different semantics for disjunction in Answer Set Programming that, unlike stable models, do not adhere to the principle of model minimality. Two of these approaches, Cabalar and Muñiz' \emph{Justified Models} and Doherty and Szalas' \emph{Strongly Supported Models}, directly provide an alternative non-minimal semantics for disjunction. The other two, Aguado et al's \emph{Forks} and Shen and Eiter's \emph{Determining Inference} (DI) semantics, actually introduce a new disjunction connective, but are compared here as if they constituted new semantics for the standard disjunction operator. We are able to prove that three of these approaches (Forks, Justified Models and a reasonable relaxation of the DI semantics) actually coincide, constituting a common single approach under different definitions. Moreover, this common semantics always provides a superset of the stable models of a program (in fact, modulo any context) and is strictly stronger than the fourth approach (Strongly Supported Models), that actually treats disjunctions as in classical logic. 

**Abstract (ZH)**: 本文将比较四种不同的析取语义在回答集编程中的应用，这四种语义不同于稳定模型，不遵循模型最小性原则。其中两种方法，Cabalar和Muñiz的J modeled模型和Doherty和Szalas的强支持模型，直接提供了非最小性的替代语义。另外两种方法，Aguado等人提出的Forks和Shen和Eiter的确定推理(DI)语义，实际上引入了新的析取连接词，但在这里被比较为标准析取运算符的新语义。我们能够证明其中三种方法（Forks、J modeled模型及DI语义的合理放松）实际上是一致的，即在不同定义下构成单一的方法。此外，这种共同语义总是提供给定程序（事实上，任何上下文中）稳定模型的超集，并且严格强于第四种方法（强支持模型），该方法实际上将析取处理为经典逻辑中的方式。 

---
# Logical Characterizations of GNNs with Mean Aggregation 

**Title (ZH)**: 带有均值聚合的GNN的逻辑表征 

**Authors**: Moritz Schönherr, Carsten Lutz  

**Link**: [PDF](https://arxiv.org/pdf/2507.18145)  

**Abstract**: We study the expressive power of graph neural networks (GNNs) with mean as the aggregation function. In the non-uniform setting, we show that such GNNs have exactly the same expressive power as ratio modal logic, which has modal operators expressing that at least a certain ratio of the successors of a vertex satisfies a specified property. The non-uniform expressive power of mean GNNs is thus higher than that of GNNs with max aggregation, but lower than for sum aggregation--the latter are characterized by modal logic and graded modal logic, respectively. In the uniform setting, we show that the expressive power relative to MSO is exactly that of alternation-free modal logic, under the natural assumptions that combination functions are continuous and classification functions are thresholds. This implies that, relative to MSO and in the uniform setting, mean GNNs are strictly less expressive than sum GNNs and max GNNs. When any of the assumptions is dropped, the expressive power increases. 

**Abstract (ZH)**: 我们研究了以均值为聚合函数的图神经网络（GNNs）的表示能力。在非均匀设置中，我们证明了这样的GNNs与比率模态逻辑具有相同的表示能力，后者通过模态运算符表达至少有一部分后续节点满足特定属性。以均值为聚合函数的非均匀GNNs的表示能力高于最大值聚合函数的GNNs，但低于求和聚合函数的GNNs——后者分别由模态逻辑和等级模态逻辑表征。在均匀设置中，我们证明了相对于MSO的表示能力恰好等同于无交替模态逻辑，前提是组合函数连续且分类函数为阈值函数。这表明，在相对于MSO且在均匀设置中，以均值为聚合函数的GNNs在表示能力上严格小于以求和或最大值为聚合函数的GNNs。如果放弃任一假设，表示能力会增加。 

---
# Actively evaluating and learning the distinctions that matter: Vaccine safety signal detection from emergency triage notes 

**Title (ZH)**: 积极评估和学习关键区别：从急诊triage笔记中检测疫苗安全性信号 

**Authors**: Sedigh Khademi, Christopher Palmer, Muhammad Javed, Hazel Clothier, Jim Buttery, Gerardo Luis Dimaguila, Jim Black  

**Link**: [PDF](https://arxiv.org/pdf/2507.18123)  

**Abstract**: The rapid development of COVID-19 vaccines has showcased the global communitys ability to combat infectious diseases. However, the need for post-licensure surveillance systems has grown due to the limited window for safety data collection in clinical trials and early widespread implementation. This study aims to employ Natural Language Processing techniques and Active Learning to rapidly develop a classifier that detects potential vaccine safety issues from emergency department notes. ED triage notes, containing expert, succinct vital patient information at the point of entry to health systems, can significantly contribute to timely vaccine safety signal surveillance. While keyword-based classification can be effective, it may yield false positives and demand extensive keyword modifications. This is exacerbated by the infrequency of vaccination-related ED presentations and their similarity to other reasons for ED visits. NLP offers a more accurate and efficient alternative, albeit requiring annotated data, which is often scarce in the medical field. Active learning optimizes the annotation process and the quality of annotated data, which can result in faster model implementation and improved model performance. This work combines active learning, data augmentation, and active learning and evaluation techniques to create a classifier that is used to enhance vaccine safety surveillance from ED triage notes. 

**Abstract (ZH)**: COVID-19疫苗的快速发展彰显了全球社区应对传染性疾病的能力，但由于临床试验中安全数据收集窗口有限以及早期广泛实施的需求，事后的监测系统需求日益增长。本研究旨在利用自然语言处理技术和主动学习快速开发一个分类器，从急诊部门病历中检测潜在的疫苗安全问题。急诊初诊记录包含患者进入卫生系统时的重要信息，可显著促进疫苗安全信号的及时监测。尽管基于关键词的分类可以有效，但它可能会产生假阳性，并需要大量的关键词调整。这进一步受到与疫苗相关的急诊就诊频率低及其与其他急诊就诊原因相似性的影响。自然语言处理提供了更准确和高效的方法，尽管需要标注数据，而在医学领域此类数据通常稀缺。主动学习优化了标注过程和标注数据的质量，可以加速模型的实施并提高模型性能。本研究结合了主动学习、数据扩充以及主动学习与评估技术，以创建一个从急诊部门初诊记录中增强疫苗安全监测的分类器。 

---
# Agentic AI framework for End-to-End Medical Data Inference 

**Title (ZH)**: 代理型AI整体医疗数据推断框架 

**Authors**: Soorya Ram Shimgekar, Shayan Vassef, Abhay Goyal, Navin Kumar, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2507.18115)  

**Abstract**: Building and deploying machine learning solutions in healthcare remains expensive and labor-intensive due to fragmented preprocessing workflows, model compatibility issues, and stringent data privacy constraints. In this work, we introduce an Agentic AI framework that automates the entire clinical data pipeline, from ingestion to inference, through a system of modular, task-specific agents. These agents handle both structured and unstructured data, enabling automatic feature selection, model selection, and preprocessing recommendation without manual intervention. We evaluate the system on publicly available datasets from geriatrics, palliative care, and colonoscopy imaging. For example, in the case of structured data (anxiety data) and unstructured data (colonoscopy polyps data), the pipeline begins with file-type detection by the Ingestion Identifier Agent, followed by the Data Anonymizer Agent ensuring privacy compliance, where we first identify the data type and then anonymize it. The Feature Extraction Agent identifies features using an embedding-based approach for tabular data, extracting all column names, and a multi-stage MedGemma-based approach for image data, which infers modality and disease name. These features guide the Model-Data Feature Matcher Agent in selecting the best-fit model from a curated repository. The Preprocessing Recommender Agent and Preprocessing Implementor Agent then apply tailored preprocessing based on data type and model requirements. Finally, the ``Model Inference Agent" runs the selected model on the uploaded data and generates interpretable outputs using tools like SHAP, LIME, and DETR attention maps. By automating these high-friction stages of the ML lifecycle, the proposed framework reduces the need for repeated expert intervention, offering a scalable, cost-efficient pathway for operationalizing AI in clinical environments. 

**Abstract (ZH)**: 构建和部署医疗领域的机器学习解决方案仍然由于数据预处理工作流碎片化、模型兼容性问题以及严格的data隐私约束而昂贵且劳动密集。在此工作中，我们引入了一个自主智能（Agentic AI）框架，通过模块化、任务特定的代理系统自动化从数据摄入到推理的整个临床数据管道。这些代理处理结构化和非结构化数据，实现自动特征选择、模型选择和预处理建议，无需手动干预。我们在老年医学、姑息治疗和结肠镜影像的公开数据集上评估了该系统。例如，在处理结构化数据（焦虑数据）和非结构化数据（结肠镜息肉数据）时，管道首先由数据摄入标识代理检测文件类型，随后由数据匿名化代理确保隐私合规，在此过程中首先识别数据类型，然后对其进行匿名化。特征提取代理使用基于嵌入的方法识别表格数据的特征，提取所有列名；对于图像数据，使用多阶段MedGemma方法推断模态和疾病名称。这些特征指导模型-数据特征匹配代理从预设库中选择最适合的模型。预处理建议代理和预处理实施代理根据数据类型和模型需求应用定制预处理。最后，“模型推理代理”在上传的数据上运行选定的模型，并使用SHAP、LIME和DETR注意力图等工具生成可解释的输出。通过自动化这些高摩擦阶段的ML生命周期，所提出框架减少了重复专家干预的需要，提供了一种在临床环境中实现AI的可扩展、成本效益高的途径。 

---
# Multi-Agent Guided Policy Optimization 

**Title (ZH)**: 多智能体引导策略优化 

**Authors**: Yueheng Li, Guangming Xie, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18059)  

**Abstract**: Due to practical constraints such as partial observability and limited communication, Centralized Training with Decentralized Execution (CTDE) has become the dominant paradigm in cooperative Multi-Agent Reinforcement Learning (MARL). However, existing CTDE methods often underutilize centralized training or lack theoretical guarantees. We propose Multi-Agent Guided Policy Optimization (MAGPO), a novel framework that better leverages centralized training by integrating centralized guidance with decentralized execution. MAGPO uses an auto-regressive joint policy for scalable, coordinated exploration and explicitly aligns it with decentralized policies to ensure deployability under partial observability. We provide theoretical guarantees of monotonic policy improvement and empirically evaluate MAGPO on 43 tasks across 6 diverse environments. Results show that MAGPO consistently outperforms strong CTDE baselines and matches or surpasses fully centralized approaches, offering a principled and practical solution for decentralized multi-agent learning. Our code and experimental data can be found in this https URL. 

**Abstract (ZH)**: 基于集中训练与分散执行的多智能体引导策略优化 

---
# Does visualization help AI understand data? 

**Title (ZH)**: 可视化有助于AI理解数据吗？ 

**Authors**: Victoria R. Li, Johnathan Sun, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.18022)  

**Abstract**: Charts and graphs help people analyze data, but can they also be useful to AI systems? To investigate this question, we perform a series of experiments with two commercial vision-language models: GPT 4.1 and Claude 3.5. Across three representative analysis tasks, the two systems describe synthetic datasets more precisely and accurately when raw data is accompanied by a scatterplot, especially as datasets grow in complexity. Comparison with two baselines -- providing a blank chart and a chart with mismatched data -- shows that the improved performance is due to the content of the charts. Our results are initial evidence that AI systems, like humans, can benefit from visualization. 

**Abstract (ZH)**: 图表和图形帮助人们分析数据，但它们对AI系统也有用吗？我们的实验使用两种商业视觉语言模型GPT 4.1和Claude 3.5表明，在三项代表性分析任务中，当原始数据伴随散点图时，这两种系统能够更精确和准确地描述合成数据集，特别是在数据集变得更加复杂时。与两个基线（空白图表和数据不匹配的图表）的比较表明，性能提升归因于图表的内容。我们的结果初步证明，AI系统像人类一样可以从可视化中受益。 

---
# Synthesis of timeline-based planning strategies avoiding determinization 

**Title (ZH)**: 基于时间线的规划策略合成避免确定化 

**Authors**: Dario Della Monica, Angelo Montanari, Pietro Sala  

**Link**: [PDF](https://arxiv.org/pdf/2507.17988)  

**Abstract**: Qualitative timeline-based planning models domains as sets of independent, but
interacting, components whose behaviors over time, the timelines, are governed
by sets of qualitative temporal constraints (ordering relations), called
synchronization rules.
Its plan-existence problem has been shown to be PSPACE-complete; in
particular, PSPACE-membership has been proved via reduction to the
nonemptiness problem for nondeterministic finite automata.
However, nondeterministic automata cannot be directly used to synthesize
planning strategies as a costly determinization step is needed.
In this paper, we identify a fragment of qualitative timeline-based planning
whose plan-existence problem can be directly mapped into the nonemptiness
problem of deterministic finite automata, which can then
synthesize strategies.
In addition, we identify a maximal subset of Allen's relations that fits into
such a deterministic fragment. 

**Abstract (ZH)**: 基于定性时间线的规划模型将领域视为一组独立但相互作用的组件，其随时间的行为（时间线）由一组定性的时间约束（序关系）控制，这些约束被称为同步规则。 

---
# I2I-STRADA -- Information to Insights via Structured Reasoning Agent for Data Analysis 

**Title (ZH)**: I2I-STRADA —— 通过结构化推理代理从信息到洞察的数据分析 

**Authors**: SaiBarath Sundar, Pranav Satheesan, Udayaadithya Avadhanam  

**Link**: [PDF](https://arxiv.org/pdf/2507.17874)  

**Abstract**: Recent advances in agentic systems for data analysis have emphasized automation of insight generation through multi-agent frameworks, and orchestration layers. While these systems effectively manage tasks like query translation, data transformation, and visualization, they often overlook the structured reasoning process underlying analytical thinking. Reasoning large language models (LLMs) used for multi-step problem solving are trained as general-purpose problem solvers. As a result, their reasoning or thinking steps do not adhere to fixed processes for specific tasks. Real-world data analysis requires a consistent cognitive workflow: interpreting vague goals, grounding them in contextual knowledge, constructing abstract plans, and adapting execution based on intermediate outcomes. We introduce I2I-STRADA (Information-to-Insight via Structured Reasoning Agent for Data Analysis), an agentic architecture designed to formalize this reasoning process. I2I-STRADA focuses on modeling how analysis unfolds via modular sub-tasks that reflect the cognitive steps of analytical reasoning. Evaluations on the DABstep and DABench benchmarks show that I2I-STRADA outperforms prior systems in planning coherence and insight alignment, highlighting the importance of structured cognitive workflows in agent design for data analysis. 

**Abstract (ZH)**: 近年来，代理系统在数据分析领域的进展强调了通过多代理框架和编排层自动产生洞见。虽然这些系统有效管理查询转换、数据转换和可视化等任务，但往往忽视了分析思考背后的结构化推理过程。用于多步问题解决的大型语言模型（LLMs）作为通用问题解决者进行训练，因此它们的推理或思维步骤不符合特定任务的固定流程。实际数据分析需要一致的认知工作流程：解释模糊的目标、在背景知识中定位它们、构建抽象计划，并根据中间结果调整执行。我们提出了I2I-STRADA（Information-to-Insight via Structured Reasoning Agent for Data Analysis），这是一种代理架构，旨在正式化这一推理过程。I2I-STRADA着重于建模分析如何通过反映分析推理认知步骤的模块化子任务展开。在DABstep和DABench基准测试上的评估结果显示，I2I-STRADA在规划连贯性和洞见对齐方面优于之前系统，突显了在数据分析中设计代理时结构化认知工作流程的重要性。 

---
# ASP-Assisted Symbolic Regression: Uncovering Hidden Physics in Fluid Mechanics 

**Title (ZH)**: ASP辅助符号回归：揭示流体力学中的隐藏物理规律 

**Authors**: Theofanis Aravanis, Grigorios Chrimatopoulos, Mohammad Ferdows, Michalis Xenos, Efstratios Em Tzirtzilakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.17777)  

**Abstract**: Unlike conventional Machine-Learning (ML) approaches, often criticized as "black boxes", Symbolic Regression (SR) stands out as a powerful tool for revealing interpretable mathematical relationships in complex physical systems, requiring no a priori assumptions about models' structures. Motivated by the recognition that, in fluid mechanics, an understanding of the underlying flow physics is as crucial as accurate prediction, this study applies SR to model a fundamental three-dimensional (3D) incompressible flow in a rectangular channel, focusing on the (axial) velocity and pressure fields under laminar conditions. By employing the PySR library, compact symbolic equations were derived directly from numerical simulation data, revealing key characteristics of the flow dynamics. These equations not only approximate the parabolic velocity profile and pressure drop observed in the studied fluid flow, but also perfectly coincide with analytical solutions from the literature. Furthermore, we propose an innovative approach that integrates SR with the knowledge-representation framework of Answer Set Programming (ASP), combining the generative power of SR with the declarative reasoning strengths of ASP. The proposed hybrid SR/ASP framework ensures that the SR-generated symbolic expressions are not only statistically accurate, but also physically plausible, adhering to domain-specific principles. Overall, the study highlights two key contributions: SR's ability to simplify complex flow behaviours into concise, interpretable equations, and the potential of knowledge-representation approaches to improve the reliability and alignment of data-driven SR models with domain principles. Insights from the examined 3D channel flow pave the way for integrating such hybrid approaches into efficient frameworks, [...] where explainable predictions and real-time data analysis are crucial. 

**Abstract (ZH)**: 不同于常被批评为“黑盒”方法的传统机器学习（ML）技术，符号回归（SR）作为一种工具能够揭示复杂物理系统中的可解释数学关系，无需预先假设模型结构。基于流体力学中对流体物理学理解与精确预测同样重要的认识，本研究将SR应用于建模矩形通道中的三维（3D）不可压缩流动，重点关注层流条件下的轴向速度和压力场。通过使用PySR库，直接从数值模拟数据中推导出简洁的符号方程，揭示了流动态的关键特征。这些方程不仅近似了所研究流体流动中观察到的抛物线型速度分布和压力降，还与文献中的解析解完全吻合。此外，本文还提出了一种创新的方法，将SR与解答集编程（ASP）的知识表示框架相结合，利用SR的生成能力与ASP的声明式推理优势。所提出的 hybrid SR/ASP 框架确保SR生成的符号表达式不仅在统计上准确，还在物理上合理，遵循特定领域原理。总体而言，本文突出了两个关键贡献：SR简化复杂流行为为简洁可解释方程的能力，以及知识表示方法提升数据驱动SR模型与领域原理可靠性和一致性的潜力。针对检查的三维通道流动，为将此类混合方法集成到高效框架中奠定了基础，该框架中可解释的预测和实时数据分析至关重要。 

---
# SIDA: Synthetic Image Driven Zero-shot Domain Adaptation 

**Title (ZH)**: SIDA: 合成图像驱动的零样本领域适应 

**Authors**: Ye-Chan Kim, SeungJu Cha, Si-Woo Kim, Taewhan Kim, Dong-Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18632)  

**Abstract**: Zero-shot domain adaptation is a method for adapting a model to a target domain without utilizing target domain image data. To enable adaptation without target images, existing studies utilize CLIP's embedding space and text description to simulate target-like style features. Despite the previous achievements in zero-shot domain adaptation, we observe that these text-driven methods struggle to capture complex real-world variations and significantly increase adaptation time due to their alignment process. Instead of relying on text descriptions, we explore solutions leveraging image data, which provides diverse and more fine-grained style cues. In this work, we propose SIDA, a novel and efficient zero-shot domain adaptation method leveraging synthetic images. To generate synthetic images, we first create detailed, source-like images and apply image translation to reflect the style of the target domain. We then utilize the style features of these synthetic images as a proxy for the target domain. Based on these features, we introduce Domain Mix and Patch Style Transfer modules, which enable effective modeling of real-world variations. In particular, Domain Mix blends multiple styles to expand the intra-domain representations, and Patch Style Transfer assigns different styles to individual patches. We demonstrate the effectiveness of our method by showing state-of-the-art performance in diverse zero-shot adaptation scenarios, particularly in challenging domains. Moreover, our approach achieves high efficiency by significantly reducing the overall adaptation time. 

**Abstract (ZH)**: 基于合成图像的零样本域适应方法 

---
# 3D Software Synthesis Guided by Constraint-Expressive Intermediate Representation 

**Title (ZH)**: 基于约束表达中间表示的3D软件合成 

**Authors**: Shuqing Li, Anson Y. Lam, Yun Peng, Wenxuan Wang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18625)  

**Abstract**: Graphical user interface (UI) software has undergone a fundamental transformation from traditional two-dimensional (2D) desktop/web/mobile interfaces to spatial three-dimensional (3D) environments. While existing work has made remarkable success in automated 2D software generation, such as HTML/CSS and mobile app interface code synthesis, the generation of 3D software still remains under-explored. Current methods for 3D software generation usually generate the 3D environments as a whole and cannot modify or control specific elements in the software. Furthermore, these methods struggle to handle the complex spatial and semantic constraints inherent in the real world. To address the challenges, we present Scenethesis, a novel requirement-sensitive 3D software synthesis approach that maintains formal traceability between user specifications and generated 3D software. Scenethesis is built upon ScenethesisLang, a domain-specific language that serves as a granular constraint-aware intermediate representation (IR) to bridge natural language requirements and executable 3D software. It serves both as a comprehensive scene description language enabling fine-grained modification of 3D software elements and as a formal constraint-expressive specification language capable of expressing complex spatial constraints. By decomposing 3D software synthesis into stages operating on ScenethesisLang, Scenethesis enables independent verification, targeted modification, and systematic constraint satisfaction. Our evaluation demonstrates that Scenethesis accurately captures over 80% of user requirements and satisfies more than 90% of hard constraints while handling over 100 constraints simultaneously. Furthermore, Scenethesis achieves a 42.8% improvement in BLIP-2 visual evaluation scores compared to the state-of-the-art method. 

**Abstract (ZH)**: 图形用户界面（UI）软件从传统的二维（2D）桌面/网页/移动接口经历了根本性的转变，发展到三维（3D）空间环境。尽管现有工作在自动化的2D软件生成方面取得了显著的成功，例如HTML/CSS和移动应用界面代码合成，但3D软件的生成仍然被严重忽视。当前的3D软件生成方法通常一次性生成整个3D环境，无法修改或控制软件中的特定元素，并且难以处理现实世界中固有的复杂空间和语义约束。为应对这些挑战，我们提出了Scenethesis，一种新的、面向要求的3D软件合成方法，它在用户规范和生成的3D软件之间保持了形式化的可追溯性。Scenethesis基于ScenethesisLang，这是一种领域特定语言，作为粒度感知的形式化约束中间表示（IR），连接自然语言需求和可执行的3D软件。它既是一种全面的场景描述语言，允许对3D软件元素进行精细修改，也是一种能够表达复杂空间约束的形式化约束表达规范语言。通过将3D软件合成分解为ScenethesisLang上的阶段操作，Scenethesis实现了独立验证、目标修改和系统约束满足。我们的评估表明，Scenethesis能够准确捕获用户需求的80%以上，并满足超过90%的硬约束，同时处理超过100个约束。此外，与最先进的方法相比，Scenethesis在BLIP-2可视化评估得分上获得了42.8%的提升。 

---
# Approximate SMT Counting Beyond Discrete Domains 

**Title (ZH)**: 超越离散域的近似SMT计数 

**Authors**: Arijit Shaw, Kuldeep S. Meel  

**Link**: [PDF](https://arxiv.org/pdf/2507.18612)  

**Abstract**: Satisfiability Modulo Theory (SMT) solvers have advanced automated reasoning, solving complex formulas across discrete and continuous domains. Recent progress in propositional model counting motivates extending SMT capabilities toward model counting, especially for hybrid SMT formulas. Existing approaches, like bit-blasting, are limited to discrete variables, highlighting the challenge of counting solutions projected onto the discrete domain in hybrid formulas.
We introduce pact, an SMT model counter for hybrid formulas that uses hashing-based approximate model counting to estimate solutions with theoretical guarantees. pact makes a logarithmic number of SMT solver calls relative to the projection variables, leveraging optimized hash functions. pact achieves significant performance improvements over baselines on a large suite of benchmarks. In particular, out of 14,202 instances, pact successfully finished on 603 instances, while Baseline could only finish on 13 instances. 

**Abstract (ZH)**: Satisfiability Modulo Theory (SMT)求解器通过解决离散和连续域的复杂公式，推动了自动化推理的发展。命题模型计数的最近进展促使SMT能力向模型计数扩展，特别是针对混合SMT公式。现有的方法如位爆炸处理，仅限于离散变量，凸显了在混合公式中对投射到离散域的解进行计数的挑战。

我们引入了pact，一种针对混合公式的SMT模型计数器，利用基于哈希的近似模型计数来提供理论上的解估计。pact相对于投影变量仅需要对数级的SMT求解器调用，并利用优化的哈希函数。pact在大量基准测试上实现了显著的性能改进，在14,202个实例中，pact成功完成的有603个，而基线只能完成13个。 

---
# A Foundation Model for Massive MIMO Precoding with an Adaptive per-User Rate-Power Tradeoff 

**Title (ZH)**: 大规模MIMO预编码的自适应用户率-功率tradeoff基础模型 

**Authors**: Jérôme Emery, Ali Hasanzadeh Karkan, Jean-François Frigon, François Leduc-Primeau  

**Link**: [PDF](https://arxiv.org/pdf/2507.18587)  

**Abstract**: Deep learning (DL) has emerged as a solution for precoding in massive multiple-input multiple-output (mMIMO) systems due to its capacity to learn the characteristics of the propagation environment. However, training such a model requires high-quality, local datasets at the deployment site, which are often difficult to collect. We propose a transformer-based foundation model for mMIMO precoding that seeks to minimize the energy consumption of the transmitter while dynamically adapting to per-user rate requirements. At equal energy consumption, zero-shot deployment of the proposed foundation model significantly outperforms zero forcing, and approaches weighted minimum mean squared error performance with 8x less complexity. To address model adaptation in data-scarce settings, we introduce a data augmentation method that finds training samples similar to the target distribution by computing the cosine similarity between the outputs of the pre-trained feature extractor. Our work enables the implementation of DL-based solutions in practice by addressing challenges of data availability and training complexity. Moreover, the ability to dynamically configure per-user rate requirements can be leveraged by higher level resource allocation and scheduling algorithms for greater control over energy efficiency, spectral efficiency and fairness. 

**Abstract (ZH)**: 基于变压器的基础模型在大规模多输入多输出系统中的稀疏数据场景下实现深度学习预编码能耗最小化动态适配 

---
# DR.EHR: Dense Retrieval for Electronic Health Record with Knowledge Injection and Synthetic Data 

**Title (ZH)**: DR.EHR：知识注入与合成数据驱动的电子健康记录密集检索 

**Authors**: Zhengyun Zhao, Huaiyuan Ying, Yue Zhong, Sheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18583)  

**Abstract**: Electronic Health Records (EHRs) are pivotal in clinical practices, yet their retrieval remains a challenge mainly due to semantic gap issues. Recent advancements in dense retrieval offer promising solutions but existing models, both general-domain and biomedical-domain, fall short due to insufficient medical knowledge or mismatched training corpora. This paper introduces \texttt{this http URL}, a series of dense retrieval models specifically tailored for EHR retrieval. We propose a two-stage training pipeline utilizing MIMIC-IV discharge summaries to address the need for extensive medical knowledge and large-scale training data. The first stage involves medical entity extraction and knowledge injection from a biomedical knowledge graph, while the second stage employs large language models to generate diverse training data. We train two variants of \texttt{this http URL}, with 110M and 7B parameters, respectively. Evaluated on the CliniQ benchmark, our models significantly outperforms all existing dense retrievers, achieving state-of-the-art results. Detailed analyses confirm our models' superiority across various match and query types, particularly in challenging semantic matches like implication and abbreviation. Ablation studies validate the effectiveness of each pipeline component, and supplementary experiments on EHR QA datasets demonstrate the models' generalizability on natural language questions, including complex ones with multiple entities. This work significantly advances EHR retrieval, offering a robust solution for clinical applications. 

**Abstract (ZH)**: 电子健康记录（EHRs）在临床实践中至关重要，但其检索仍面临挑战，主要原因是语义差距问题。最近在密集检索方面的进步提供了有希望的解决方案，但现有的通用领域和生物医学领域模型由于缺乏医学知识或训练语料库不匹配的原因仍存在不足。本文介绍了\texttt{this http URL}，这是一种专门针对EHR检索的密集检索模型系列。我们提出了一种两阶段训练管道，利用MIMIC-IV出院总结来解决需要广泛医学知识和大规模训练数据的问题。第一阶段涉及从生物医学知识图谱中提取医学实体并注入知识，第二阶段则使用大规模语言模型生成多样化的训练数据。我们分别训练了带有110M和7B参数的\texttt{this http URL}两种变体。在CliniQ基准测试中，我们的模型显著优于所有现有的密集检索器，达到了最先进的结果。详细的分析证实了我们在各种匹配和查询类型中模型的优越性，特别是在含义和缩写等具有挑战性的语义匹配中。消融研究验证了每个管道组件的有效性，并补充实验表明，模型在自然语言问答数据集上的泛化能力，包括具有多个实体的复杂问题。这项工作极大地推动了EHR检索的发展，为临床应用提供了稳健的解决方案。 

---
# Advancing Financial Engineering with Foundation Models: Progress, Applications, and Challenges 

**Title (ZH)**: 借助基础模型推动金融工程发展：进展、应用与挑战 

**Authors**: Liyuan Chen, Shuoling Liu, Jiangpeng Yan, Xiaoyu Wang, Henglin Liu, Chuang Li, Kecheng Jiao, Jixuan Ying, Yang Veronica Liu, Qiang Yang, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18577)  

**Abstract**: The advent of foundation models (FMs) - large-scale pre-trained models with strong generalization capabilities - has opened new frontiers for financial engineering. While general-purpose FMs such as GPT-4 and Gemini have demonstrated promising performance in tasks ranging from financial report summarization to sentiment-aware forecasting, many financial applications remain constrained by unique domain requirements such as multimodal reasoning, regulatory compliance, and data privacy. These challenges have spurred the emergence of Financial Foundation Models (FFMs) - a new class of models explicitly designed for finance. This survey presents a comprehensive overview of FFMs, with a taxonomy spanning three key modalities: Financial Language Foundation Models (FinLFMs), Financial Time-Series Foundation Models (FinTSFMs), and Financial Visual-Language Foundation Models (FinVLFMs). We review their architectures, training methodologies, datasets, and real-world applications. Furthermore, we identify critical challenges in data availability, algorithmic scalability, and infrastructure constraints, and offer insights into future research opportunities. We hope this survey serves as both a comprehensive reference for understanding FFMs and a practical roadmap for future innovation. An updated collection of FFM-related publications and resources will be maintained on our website this https URL. 

**Abstract (ZH)**: 基础模型(FMs)的兴起——大规模预训练模型具备强大的通用泛化能力——为金融工程开辟了新的领域。尽管通用的基础模型如GPT-4和Gemini在金融报告总结和情感感知预测等任务中已展现出有希望的表现，但许多金融应用仍受到多模态推理、合规性要求和数据隐私等独特领域需求的限制。这些挑战促进了金融基础模型(FFMs)的出现——一类明确针对金融领域设计的新模型。本文综述了FFMs，涵盖三种关键模态的分类体系：金融语言基础模型(FinLFMs)、金融时间序列基础模型(FinTSFMs)和金融视觉语言基础模型(FinVLFMs)，并对其架构、训练方法、数据集和实际应用进行了回顾。此外，我们还识别了数据可用性、算法扩展性和基础设施限制等关键挑战，并提供了未来研究机会的见解。我们希望本文综述能够成为理解FFMs的全面参考，并为未来的创新提供实用的路线图。有关FFM的相关出版物和资源将在我们网站上持续更新。 

---
# PosterMate: Audience-driven Collaborative Persona Agents for Poster Design 

**Title (ZH)**: PosterMate: 以受众为导向的合作型人物代理模型用于海报设计 

**Authors**: Donghoon Shin, Daniel Lee, Gary Hsieh, Gromit Yeuk-Yin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2507.18572)  

**Abstract**: Poster designing can benefit from synchronous feedback from target audiences. However, gathering audiences with diverse perspectives and reconciling them on design edits can be challenging. Recent generative AI models present opportunities to simulate human-like interactions, but it is unclear how they may be used for feedback processes in design. We introduce PosterMate, a poster design assistant that facilitates collaboration by creating audience-driven persona agents constructed from marketing documents. PosterMate gathers feedback from each persona agent regarding poster components, and stimulates discussion with the help of a moderator to reach a conclusion. These agreed-upon edits can then be directly integrated into the poster design. Through our user study (N=12), we identified the potential of PosterMate to capture overlooked viewpoints, while serving as an effective prototyping tool. Additionally, our controlled online evaluation (N=100) revealed that the feedback from an individual persona agent is appropriate given its persona identity, and the discussion effectively synthesizes the different persona agents' perspectives. 

**Abstract (ZH)**: Poster设计可以从目标受众的同步反馈中受益，但 Gather多元视角的受众并就设计修改达成一致具有挑战性。近期的生成式AI模型提供了模拟人类互动的机会，但它们在设计反馈过程中的应用尚不明确。我们介绍了PosterMate，这是一种通过从营销文件中构建受众驱动的人物代理来促进合作的Poster设计助手。PosterMate从每个人物代理处收集关于Poster组件的反馈，并在调解人的帮助下促进讨论，以达成共识。这些达成一致的修改可以直接集成到Poster设计中。通过我们的用户研究（N=12），我们发现PosterMate能够捕捉被忽视的观点，同时作为一种有效的原型制作工具。此外，我们的可控在线评估（N=100）表明，单个人物代理提供的反馈与其人物身份相符，讨论有效地概括了不同人物代理的观点。 

---
# Proceedings 19th International Workshop on the ACL2 Theorem Prover and Its Applications 

**Title (ZH)**: 第19届ACL2定理证明器及其应用国际研讨会论文集 

**Authors**: Ruben Gamboa, Panagiotis Manolios  

**Link**: [PDF](https://arxiv.org/pdf/2507.18567)  

**Abstract**: The ACL2 Workshop series is the major technical forum for users of the ACL2 theorem proving system to present research related to the ACL2 theorem prover and its applications. ACL2 is an industrial-strength automated reasoning system, the latest in the Boyer-Moore family of theorem provers. The 2005 ACM Software System Award was awarded to Boyer, Kaufmann, and Moore for their work on ACL2 and the other theorem provers in the Boyer-Moore family. 

**Abstract (ZH)**: ACL2研讨会系列是ACL2定理证明系统用户讨论与ACL2定理证明器及其应用相关的研究的主要技术论坛。ACL2是Boyer-Moore定理证明家族中的工业强度自动化推理系统。Boyer、Kaufmann和Moore因ACL2及其他Boyer-Moore家族的定理证明器的工作于2005年获得ACM软件系统奖。 

---
# GIIFT: Graph-guided Inductive Image-free Multimodal Machine Translation 

**Title (ZH)**: GIIFT：图引导的归纳图像无介 multimodal 机器翻译 

**Authors**: Jiafeng Xiong, Yuting Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18562)  

**Abstract**: Multimodal Machine Translation (MMT) has demonstrated the significant help of visual information in machine translation. However, existing MMT methods face challenges in leveraging the modality gap by enforcing rigid visual-linguistic alignment whilst being confined to inference within their trained multimodal domains. In this work, we construct novel multimodal scene graphs to preserve and integrate modality-specific information and introduce GIIFT, a two-stage Graph-guided Inductive Image-Free MMT framework that uses a cross-modal Graph Attention Network adapter to learn multimodal knowledge in a unified fused space and inductively generalize it to broader image-free translation domains. Experimental results on the Multi30K dataset of English-to-French and English-to-German tasks demonstrate that our GIIFT surpasses existing approaches and achieves the state-of-the-art, even without images during inference. Results on the WMT benchmark show significant improvements over the image-free translation baselines, demonstrating the strength of GIIFT towards inductive image-free inference. 

**Abstract (ZH)**: 多模态机器翻译（MMT）已经证明了视觉信息在机器翻译中的显著帮助。然而，现有的MMT方法在利用模态差距时面临挑战，它们通过强制执行刚性的视觉-语言对齐而在训练的多模态领域内进行推理时受到限制。在本文中，我们构建了新颖的多模态场景图以保留和整合模态特定信息，并引入了GIIFT，一种两阶段的图引导归纳无图像多模态机器翻译框架，该框架使用跨模态图注意网络适配器在统一融合空间中学习多模态知识，并归纳泛化到更广泛的无图像翻译领域。在Multi30K数据集的英法和英德任务上的实验结果表明，我们的GIIFT超过了现有方法并达到了最新水平，即使在推理时不使用图像也是如此。在WMT基准上的结果表明，GIIFT在无图像翻译基线上的显著改进，证明了GIIFT在归纳无图像推理中的优势。 

---
# Beyond Internal Data: Constructing Complete Datasets for Fairness Testing 

**Title (ZH)**: 超越内部数据：构建全面数据集以进行公平性测试 

**Authors**: Varsha Ramineni, Hossein A. Rahmani, Emine Yilmaz, David Barber  

**Link**: [PDF](https://arxiv.org/pdf/2507.18561)  

**Abstract**: As AI becomes prevalent in high-risk domains and decision-making, it is essential to test for potential harms and biases. This urgency is reflected by the global emergence of AI regulations that emphasise fairness and adequate testing, with some mandating independent bias audits. However, procuring the necessary data for fairness testing remains a significant challenge. Particularly in industry settings, legal and privacy concerns restrict the collection of demographic data required to assess group disparities, and auditors face practical and cultural challenges in gaining access to data. Further, internal historical datasets are often insufficiently representative to identify real-world biases. This work focuses on evaluating classifier fairness when complete datasets including demographics are inaccessible. We propose leveraging separate overlapping datasets to construct complete synthetic data that includes demographic information and accurately reflects the underlying relationships between protected attributes and model features. We validate the fidelity of the synthetic data by comparing it to real data, and empirically demonstrate that fairness metrics derived from testing on such synthetic data are consistent with those obtained from real data. This work, therefore, offers a path to overcome real-world data scarcity for fairness testing, enabling independent, model-agnostic evaluation of fairness, and serving as a viable substitute where real data is limited. 

**Abstract (ZH)**: 随着人工智能在高风险领域和决策中的普及，测试潜在危害和偏差变得至关重要。这反映在世界各国出台的强调公平和充分测试的人工智能法规中，部分法规甚至要求进行独立的偏差审计。然而，获取必要的公平测试数据依然是一项重大挑战。特别是在工业环境中，法律和隐私顾虑限制了用于评估群体差异所需的人口统计数据的收集，而审计人员在获取数据方面也面临实际和文化方面的挑战。此外，内部的历史数据集往往不足以识别现实世界的偏差。本文关注在无法访问完整数据集包括人口统计数据的情况下评估分类器公平性的问题。我们建议利用重叠的分离数据集来构建包含人口统计信息并准确反映受保护属性与模型特征之间关系的合成数据。通过将合成数据与真实数据进行对比评估其忠实度，并实证证明，基于合成数据进行测试所获得的公平性指标与基于真实数据所获得的结果是一致的。因此，本文为解决公平测试中的现实数据稀缺性问题提供了一条路径，有助于实现独立的、模型无关的公平性评估，并在现实数据有限时提供一个可行的替代方案。 

---
# GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface 

**Title (ZH)**: GLiNER2：一种基于模式驱动接口的高效多任务信息提取系统 

**Authors**: Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, Ash Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2507.18546)  

**Abstract**: Information extraction (IE) is fundamental to numerous NLP applications, yet existing solutions often require specialized models for different tasks or rely on computationally expensive large language models. We present GLiNER2, a unified framework that enhances the original GLiNER architecture to support named entity recognition, text classification, and hierarchical structured data extraction within a single efficient model. Built pretrained transformer encoder architecture, GLiNER2 maintains CPU efficiency and compact size while introducing multi-task composition through an intuitive schema-based interface. Our experiments demonstrate competitive performance across extraction and classification tasks with substantial improvements in deployment accessibility compared to LLM-based alternatives. We release GLiNER2 as an open-source pip-installable library with pre-trained models and documentation at this https URL. 

**Abstract (ZH)**: 基于GLiNER2的统一框架：一种高效的多任务信息提取和分类模型 

---
# C2G-KD: PCA-Constrained Generator for Data-Free Knowledge Distillation 

**Title (ZH)**: C2G-KD: PCA约束生成器用于无数据知识精炼 

**Authors**: Magnus Bengtsson, Kenneth Östberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.18533)  

**Abstract**: We introduce C2G-KD, a data-free knowledge distillation framework where a class-conditional generator is trained to produce synthetic samples guided by a frozen teacher model and geometric constraints derived from PCA. The generator never observes real training data but instead learns to activate the teacher's output through a combination of semantic and structural losses. By constraining generated samples to lie within class-specific PCA subspaces estimated from as few as two real examples per class, we preserve topological consistency and diversity. Experiments on MNIST show that even minimal class structure is sufficient to bootstrap useful synthetic training pipelines. 

**Abstract (ZH)**: 面向类别的数据免费知识蒸馏框架：利用PCA约束的生成器引导教师模型和几何约束进行学习 

---
# GLANCE: Graph Logic Attention Network with Cluster Enhancement for Heterophilous Graph Representation Learning 

**Title (ZH)**: GLANCE: 图逻辑注意力网络结合簇增强的异ophilous图表示学习 

**Authors**: Zhongtian Sun, Anoushka Harit, Alexandra Cristea, Christl A. Donnelly, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2507.18521)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated significant success in learning from graph-structured data but often struggle on heterophilous graphs, where connected nodes differ in features or class labels. This limitation arises from indiscriminate neighbor aggregation and insufficient incorporation of higher-order structural patterns. To address these challenges, we propose GLANCE (Graph Logic Attention Network with Cluster Enhancement), a novel framework that integrates logic-guided reasoning, dynamic graph refinement, and adaptive clustering to enhance graph representation learning. GLANCE combines a logic layer for interpretable and structured embeddings, multi-head attention-based edge pruning for denoising graph structures, and clustering mechanisms for capturing global patterns. Experimental results in benchmark datasets, including Cornell, Texas, and Wisconsin, demonstrate that GLANCE achieves competitive performance, offering robust and interpretable solutions for heterophilous graph scenarios. The proposed framework is lightweight, adaptable, and uniquely suited to the challenges of heterophilous graphs. 

**Abstract (ZH)**: Graph逻辑注意力网络与聚类增强（GLANCE）：一种用于异构图的新型图表示学习框架 

---
# Generation of Synthetic Clinical Text: A Systematic Review 

**Title (ZH)**: 合成临床文本的生成：一项系统评价 

**Authors**: Basel Alshaikhdeeb, Ahmed Abdelmonem Hemedan, Soumyabrata Ghosh, Irina Balaur, Venkata Satagopam  

**Link**: [PDF](https://arxiv.org/pdf/2507.18451)  

**Abstract**: Generating clinical synthetic text represents an effective solution for common clinical NLP issues like sparsity and privacy. This paper aims to conduct a systematic review on generating synthetic medical free-text by formulating quantitative analysis to three research questions concerning (i) the purpose of generation, (ii) the techniques, and (iii) the evaluation methods. We searched PubMed, ScienceDirect, Web of Science, Scopus, IEEE, Google Scholar, and arXiv databases for publications associated with generating synthetic medical unstructured free-text. We have identified 94 relevant articles out of 1,398 collected ones. A great deal of attention has been given to the generation of synthetic medical text from 2018 onwards, where the main purpose of such a generation is towards text augmentation, assistive writing, corpus building, privacy-preserving, annotation, and usefulness. Transformer architectures were the main predominant technique used to generate the text, especially the GPTs. On the other hand, there were four main aspects of evaluation, including similarity, privacy, structure, and utility, where utility was the most frequent method used to assess the generated synthetic medical text. Although the generated synthetic medical text demonstrated a moderate possibility to act as real medical documents in different downstream NLP tasks, it has proven to be a great asset as augmented, complementary to the real documents, towards improving the accuracy and overcoming sparsity/undersampling issues. Yet, privacy is still a major issue behind generating synthetic medical text, where more human assessments are needed to check for the existence of any sensitive information. Despite that, advances in generating synthetic medical text will considerably accelerate the adoption of workflows and pipeline development, discarding the time-consuming legalities of data transfer. 

**Abstract (ZH)**: 生成临床合成文本代表了解决临床自然语言处理问题（如稀疏性和隐私问题）的一种有效方法。本文旨在通过定量分析三个研究问题——（i）生成目的，（ii）技术，和（iii）评估方法——对生成合成医学自由文本进行系统综述。我们搜索了PubMed、ScienceDirect、Web of Science、Scopus、IEEE、Google Scholar和arXiv数据库，寻找与生成合成医学非结构化自由文本相关的出版物。我们一共找到了1398篇文献中的94篇相关文章。自2018年以来，合成医学文本的生成得到了广泛关注，其主要目的是文本增强、辅助写作、构建语料库、隐私保护、标注和实用性。Transformer架构是主要用于生成文本的主要技术，特别是GPTs。另一方面，在评估方面主要有四个主要方面，包括相似性、隐私、结构和实用性，其中实用性是最常用于评估生成的合成医学文本的方法。虽然生成的合成医学文本在不同的下游NLP任务中显示出作为真实医学文档的中等可能性，但它作为增强的真实文档的补充资产，已被证明能够提高准确性并克服稀疏性/欠采样问题。然而，隐私仍然是生成合成医学文本的主要问题，需要更多的手工评估来检查是否存在敏感信息。尽管如此，合成医学文本生成的进步将显著加速工作流程和管道开发的采用，摆脱繁琐的数据传输法律程序。 

---
# Digital Twin Technologies in Predictive Maintenance: Enabling Transferability via Sim-to-Real and Real-to-Sim Transfer 

**Title (ZH)**: 数字孪生技术在预测性维护中的应用：通过模拟到现实和现实到模拟的转移实现可转移性 

**Authors**: Sizhe Ma, Katherine A. Flanigan, Mario Bergés  

**Link**: [PDF](https://arxiv.org/pdf/2507.18449)  

**Abstract**: The advancement of the Internet of Things (IoT) and Artificial Intelligence has catalyzed the evolution of Digital Twins (DTs) from conceptual ideas to more implementable realities. Yet, transitioning from academia to industry is complex due to the absence of standardized frameworks. This paper builds upon the authors' previously established functional and informational requirements supporting standardized DT development, focusing on a crucial aspect: transferability. While existing DT research primarily centers on asset transfer, the significance of "sim-to-real transfer" and "real-to-sim transfer"--transferring knowledge between simulations and real-world operations--is vital for comprehensive lifecycle management in DTs. A key challenge in this process is calibrating the "reality gap," the discrepancy between simulated predictions and actual outcomes. Our research investigates the impact of integrating a single Reality Gap Analysis (RGA) module into an existing DT framework to effectively manage both sim-to-real and real-to-sim transfers. This integration is facilitated by data pipelines that connect the RGA module with the existing components of the DT framework, including the historical repository and the simulation model. A case study on a pedestrian bridge at Carnegie Mellon University showcases the performance of different levels of integration of our approach with an existing framework. With full implementation of an RGA module and a complete data pipeline, our approach is capable of bidirectional knowledge transfer between simulations and real-world operations without compromising efficiency. 

**Abstract (ZH)**: 物联网和人工智能的进步推动了数字孪生从概念性想法向更可实施的现实转化。然而，从学术界向工业界的转型由于缺乏标准化框架而复杂化。本文基于作者之前建立的功能性和信息性要求，支持标准化数字孪生的开发，重点关注一个关键方面：可转移性。现有数字孪生研究主要集中在资产转移上，而“仿真到现实转移”和“现实到仿真转移”的重要性——即在仿真与实际操作之间转移知识——对于数字孪生的全面生命周期管理至关重要。这一过程中的一项主要挑战是校准“现实差距”，即仿真预测与实际结果之间的差异。我们的研究探讨了将在现有数字孪生框架中集成一个单一的现实差距分析（RGA）模块以有效管理仿真到现实和现实到仿真的转移的影响。这种集成通过将RGA模块与现有数字孪生框架中的历史仓库和仿真模型等组件连接起来，由数据管道来实现。以卡内基梅隆大学的行人桥为例，研究展示了不同集成水平的方法与现有框架结合后的性能。当完全实施RGA模块和完整的数据管道时，我们的方法能够在不牺牲效率的情况下在仿真与现实操作之间实现双向知识转移。 

---
# A Concept for Efficient Scalability of Automated Driving Allowing for Technical, Legal, Cultural, and Ethical Differences 

**Title (ZH)**: 一种适用于技术、法律、文化及伦理差异的自动驾驶高效扩展概念 

**Authors**: Lars Ullrich, Michael Buchholz, Jonathan Petit, Klaus Dietmayer, Knut Graichen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18326)  

**Abstract**: Efficient scalability of automated driving (AD) is key to reducing costs, enhancing safety, conserving resources, and maximizing impact. However, research focuses on specific vehicles and context, while broad deployment requires scalability across various configurations and environments. Differences in vehicle types, sensors, actuators, but also traffic regulations, legal requirements, cultural dynamics, or even ethical paradigms demand high flexibility of data-driven developed capabilities. In this paper, we address the challenge of scalable adaptation of generic capabilities to desired systems and environments. Our concept follows a two-stage fine-tuning process. In the first stage, fine-tuning to the specific environment takes place through a country-specific reward model that serves as an interface between technological adaptations and socio-political requirements. In the second stage, vehicle-specific transfer learning facilitates system adaptation and governs the validation of design decisions. In sum, our concept offers a data-driven process that integrates both technological and socio-political aspects, enabling effective scalability across technical, legal, cultural, and ethical differences. 

**Abstract (ZH)**: 自动驾驶（AD）的高效扩展是降低成本、提升安全、节约资源和最大化影响的关键。然而，研究主要集中在特定车辆和情境上，而广泛部署则需要在各种配置和环境中实现扩展。车辆类型、传感器、执行器的差异，以及交通法规、法律要求、文化动态，甚至伦理观念都需要高度的数据驱动开发能力的灵活性。在本文中，我们解决了一般能力可扩展适配到目标系统和环境的挑战。我们的概念遵循两阶段微调过程。在第一阶段，通过国家特定的奖励模型进行环境适配微调，作为技术适应与社会政治要求之间的接口。在第二阶段，车辆特定的迁移学习促进系统适配并指导设计决策的验证。总之，我们的概念提供了一个数据驱动的过程，整合了技术和社会政治两个方面，能够在技术、法律、文化和伦理差异中实现有效的扩展。 

---
# A Multi-Dataset Benchmark for Semi-Supervised Semantic Segmentation in ECG Delineation 

**Title (ZH)**: 一个多数据集基准，用于ECG边界标注的半监督语义分割 

**Authors**: Minje Park, Jeonghwa Lim, Taehyung Yu, Sunghoon Joo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18323)  

**Abstract**: Electrocardiogram (ECG) delineation, the segmentation of meaningful waveform features, is critical for clinical diagnosis. Despite recent advances using deep learning, progress has been limited by the scarcity of publicly available annotated datasets. Semi-supervised learning presents a promising solution by leveraging abundant unlabeled ECG data. In this study, we present the first systematic benchmark for semi-supervised semantic segmentation (SemiSeg) in ECG delineation. We curated and unified multiple public datasets, including previously underused sources, to support robust and diverse evaluation. We adopted five representative SemiSeg algorithms from computer vision, implemented them on two different architectures: the convolutional network and the transformer, and evaluated them in two different settings: in-domain and cross-domain. Additionally, we propose ECG-specific training configurations and augmentation strategies and introduce a standardized evaluation framework. Our results show that the transformer outperforms the convolutional network in semi-supervised ECG delineation. We anticipate that our benchmark will serve as a foundation for advancing semi-supervised ECG delineation methods and will facilitate further research in this domain. 

**Abstract (ZH)**: ECG波形特征半监督语义分割基准研究 

---
# TCM-Tongue: A Standardized Tongue Image Dataset with Pathological Annotations for AI-Assisted TCM Diagnosis 

**Title (ZH)**: TCM-舌象：一种包含病理标注的标准舌象图像数据集，用于辅助中医诊断的AI系统 

**Authors**: Xuebo Jin, Longfei Gao, Anshuo Tong, Zhengyang Chen, Jianlei Kong, Ning Sun, Huijun Ma, Qiang Wang, Yuting Bai, Tingli Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.18288)  

**Abstract**: Traditional Chinese medicine (TCM) tongue diagnosis, while clinically valuable, faces standardization challenges due to subjective interpretation and inconsistent imaging protocols, compounded by the lack of large-scale, annotated datasets for AI development. To address this gap, we present the first specialized dataset for AI-driven TCM tongue diagnosis, comprising 6,719 high-quality images captured under standardized conditions and annotated with 20 pathological symptom categories (averaging 2.54 clinically validated labels per image, all verified by licensed TCM practitioners). The dataset supports multiple annotation formats (COCO, TXT, XML) for broad usability and has been benchmarked using nine deep learning models (YOLOv5/v7/v8 variants, SSD, and MobileNetV2) to demonstrate its utility for AI development. This resource provides a critical foundation for advancing reliable computational tools in TCM, bridging the data shortage that has hindered progress in the field, and facilitating the integration of AI into both research and clinical practice through standardized, high-quality diagnostic data. 

**Abstract (ZH)**: 传统中医舌诊图像数据集：一个专为AI驱动舌诊设计的标准化数据资源 

---
# Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models 

**Title (ZH)**: 定位并聚焦：提高语音语言模型中的术语翻译 

**Authors**: Suhang Wu, Jialong Tang, Chengyi Yang, Pei Zhang, Baosong Yang, Junhui Li, Junfeng Yao, Min Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.18263)  

**Abstract**: Direct speech translation (ST) has garnered increasing attention nowadays, yet the accurate translation of terminology within utterances remains a great challenge. In this regard, current studies mainly concentrate on leveraging various translation knowledge into ST models. However, these methods often struggle with interference from irrelevant noise and can not fully utilize the translation knowledge. To address these issues, in this paper, we propose a novel Locate-and-Focus method for terminology translation. It first effectively locates the speech clips containing terminologies within the utterance to construct translation knowledge, minimizing irrelevant information for the ST model. Subsequently, it associates the translation knowledge with the utterance and hypothesis from both audio and textual modalities, allowing the ST model to better focus on translation knowledge during translation. Experimental results across various datasets demonstrate that our method effectively locates terminologies within utterances and enhances the success rate of terminology translation, while maintaining robust general translation performance. 

**Abstract (ZH)**: 直接speech翻译（ST）近年来引起了越来越多的关注，但语句中术语的准确翻译仍然是一项重大挑战。针对这一问题，现有研究主要集中在将各种翻译知识应用于ST模型中。然而，这些方法往往难以应对无关噪声的干扰，无法充分利用翻译知识。为了解决这些问题，本文提出了一种新颖的定位与聚焦方法（Locate-and-Focus method）用于术语翻译。该方法首先有效定位包含术语的语音片段以构建翻译知识，最小化ST模型中的无关信息。随后，该方法将翻译知识与来自音频和文本模态的语句和假设关联起来，使ST模型在翻译过程中能够更好地关注翻译知识。在多个数据集上的实验结果表明，本方法有效定位了语句中的术语，提高了术语翻译的成功率，同时保持了稳健的一般翻译性能。 

---
# From Individual Learning to Market Equilibrium: Correcting Structural and Parametric Biases in RL Simulations of Economic Models 

**Title (ZH)**: 从个体学习到市场均衡：纠正经济学模型中基于RL的模拟中的结构性和参数偏置 

**Authors**: Zeqiang Zhang, Ruxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18229)  

**Abstract**: The application of Reinforcement Learning (RL) to economic modeling reveals a fundamental conflict between the assumptions of equilibrium theory and the emergent behavior of learning agents. While canonical economic models assume atomistic agents act as `takers' of aggregate market conditions, a naive single-agent RL simulation incentivizes the agent to become a `manipulator' of its environment. This paper first demonstrates this discrepancy within a search-and-matching model with concave production, showing that a standard RL agent learns a non-equilibrium, monopsonistic policy. Additionally, we identify a parametric bias arising from the mismatch between economic discounting and RL's treatment of intertemporal costs. To address both issues, we propose a calibrated Mean-Field Reinforcement Learning framework that embeds a representative agent in a fixed macroeconomic field and adjusts the cost function to reflect economic opportunity costs. Our iterative algorithm converges to a self-consistent fixed point where the agent's policy aligns with the competitive equilibrium. This approach provides a tractable and theoretically sound methodology for modeling learning agents in economic systems within the broader domain of computational social science. 

**Abstract (ZH)**: 强化学习在经济建模中的应用揭示了均衡理论假设与学习代理涌现行为之间的基本冲突。这篇论文首先在一个具有凹生产函数的搜寻匹配模型中展示了这种分歧，表明标准的RL代理学习了一个非均衡的垄断工政策。此外，我们还指出了由于经济学折现与RL对跨期成本处理之间的不匹配而产生的参数偏见。为了解决这些问题，我们提出了一种校准的平均场强化学习框架，该框架将一个代表性代理嵌入固定的大宏观经济场中，并调整成本函数以反映经济机会成本。我们的迭代算法在自我一致的固定点收敛，其中代理的政策与竞争性均衡一致。这种方法为在计算社会科学的大背景下对经济系统中学习代理的建模提供了可操作且合乎理论的方法。 

---
# FedSA-GCL: A Semi-Asynchronous Federated Graph Learning Framework with Personalized Aggregation and Cluster-Aware Broadcasting 

**Title (ZH)**: FedSA-GCL：一种带有个性化聚合和聚类意识广播的半异步联邦图学习框架 

**Authors**: Zhongzheng Yuan, Lianshuai Guo, Xunkai Li, Yinlin Zhu, Wenyu Wang, Meixia Qu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18219)  

**Abstract**: Federated Graph Learning (FGL) is a distributed learning paradigm that enables collaborative training over large-scale subgraphs located on multiple local systems. However, most existing FGL approaches rely on synchronous communication, which leads to inefficiencies and is often impractical in real-world deployments. Meanwhile, current asynchronous federated learning (AFL) methods are primarily designed for conventional tasks such as image classification and natural language processing, without accounting for the unique topological properties of graph data. Directly applying these methods to graph learning can possibly result in semantic drift and representational inconsistency in the global model. To address these challenges, we propose FedSA-GCL, a semi-asynchronous federated framework that leverages both inter-client label distribution divergence and graph topological characteristics through a novel ClusterCast mechanism for efficient training. We evaluate FedSA-GCL on multiple real-world graph datasets using the Louvain and Metis split algorithms, and compare it against 9 baselines. Extensive experiments demonstrate that our method achieves strong robustness and outstanding efficiency, outperforming the baselines by an average of 2.92% with the Louvain and by 3.4% with the Metis. 

**Abstract (ZH)**: 联邦图学习（FGL）是一种分布式学习范式，能够在多个本地系统上协作训练大规模子图。然而，现有的大多数FGL方法依赖于同步通信，这会导致效率低下，并且在实际部署中往往不切实际。同时，当前的异步联邦学习（AFL）方法主要针对传统的任务如图像分类和自然语言处理进行设计，而未考虑图数据的独特拓扑特性。直接将这些方法应用于图学习可能会导致全局模型在语义和表示上的一致性问题。为了解决这些问题，我们提出了一种名为FedSA-GCL的半异步联邦框架，通过一种新颖的ClusterCast机制利用客户端标签分布差异和图拓扑特性实现高效的训练。我们在使用Louvain和Metis划分算法的多个真实世界图数据集上评估了FedSA-GCL，并将其与9种基线方法进行比较。实验结果表明，我们的方法在鲁棒性和效率上表现优秀，分别在Louvain和Metis划分算法下平均优于基线方法2.92%和3.4%。 

---
# Differential-UMamba: Rethinking Tumor Segmentation Under Limited Data Scenarios 

**Title (ZH)**: Differential-UMamba: 重新思考有限数据场景下的肿瘤分割 

**Authors**: Dhruv Jain, Romain Modzelewski, Romain Hérault, Clement Chatelain, Eva Torfeh, Sebastien Thureau  

**Link**: [PDF](https://arxiv.org/pdf/2507.18177)  

**Abstract**: In data-scarce scenarios, deep learning models often overfit to noise and irrelevant patterns, which limits their ability to generalize to unseen samples. To address these challenges in medical image segmentation, we introduce Diff-UMamba, a novel architecture that combines the UNet framework with the mamba mechanism for modeling long-range dependencies. At the heart of Diff-UMamba is a Noise Reduction Module (NRM), which employs a signal differencing strategy to suppress noisy or irrelevant activations within the encoder. This encourages the model to filter out spurious features and enhance task-relevant representations, thereby improving its focus on clinically meaningful regions. As a result, the architecture achieves improved segmentation accuracy and robustness, particularly in low-data settings. Diff-UMamba is evaluated on multiple public datasets, including MSD (lung and pancreas) and AIIB23, demonstrating consistent performance gains of 1-3% over baseline methods across diverse segmentation tasks. To further assess performance under limited-data conditions, additional experiments are conducted on the BraTS-21 dataset by varying the proportion of available training samples. The approach is also validated on a small internal non-small cell lung cancer (NSCLC) dataset for gross tumor volume (GTV) segmentation in cone beam CT (CBCT), where it achieves a 4-5% improvement over the baseline. 

**Abstract (ZH)**: 在数据稀缺场景中，深度学习模型常常过拟合到噪声和无关模式，限制了它们对未见样本的泛化能力。为了解决这些挑战在医学图像分割中的问题，我们引入了Diff-UMamba架构，该架构结合了UNet框架和mamba机制以建模长程依赖关系。Diff-UMamba的核心是噪声减少模块（NRM），该模块采用信号差分策略抑制编码器中的噪声或无关激活，鼓励模型过滤掉虚假特征并增强与任务相关的表示，从而提升其对临床有意义区域的关注。因此，该架构在低数据设置中实现了改进的分割准确性和鲁棒性。Diff-UMamba在多个公开数据集（包括MSD（肺和胰腺）和AIIB23）上进行评估，与基线方法相比，在多种分割任务中展示了1-3%的一致性能提升。为了进一步评估在有限数据条件下的表现，还对BraTS-21数据集进行了额外实验，通过改变可用训练样本的比例进行实验。该方法也在一个小型内部的非小细胞肺癌（NSCLC）数据集（锥束CT（CBCT）中的GTV分割）上进行了验证，相比基线方法实现了4-5%的改进。 

---
# Deep Learning for Glioblastoma Morpho-pathological Features Identification: A BraTS-Pathology Challenge Solution 

**Title (ZH)**: 基于深度学习的胶质母细胞瘤形态病理特征识别：BraTS-Pathology挑战赛解决方案 

**Authors**: Juexin Zhang, Ying Weng, Ke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18133)  

**Abstract**: Glioblastoma, a highly aggressive brain tumor with diverse molecular and pathological features, poses a diagnostic challenge due to its heterogeneity. Accurate diagnosis and assessment of this heterogeneity are essential for choosing the right treatment and improving patient outcomes. Traditional methods rely on identifying specific features in tissue samples, but deep learning offers a promising approach for improved glioblastoma diagnosis. In this paper, we present our approach to the BraTS-Path Challenge 2024. We leverage a pre-trained model and fine-tune it on the BraTS-Path training dataset. Our model demonstrates poor performance on the challenging BraTS-Path validation set, as rigorously assessed by the Synapse online platform. The model achieves an accuracy of 0.392229, a recall of 0.392229, and a F1-score of 0.392229, indicating a consistent ability to correctly identify instances under the target condition. Notably, our model exhibits perfect specificity of 0.898704, showing an exceptional capacity to correctly classify negative cases. Moreover, a Matthews Correlation Coefficient (MCC) of 0.255267 is calculated, to signify a limited positive correlation between predicted and actual values and highlight our model's overall predictive power. Our solution also achieves the second place during the testing phase. 

**Abstract (ZH)**: 胶质母细胞瘤：一种具有多样分子和病理特征的高度恶惟能，由于其异质性导致诊断困难。准确诊断和评估这种异质性对于选择合适的治疗方案和改善患者预后至关重要。传统方法依赖于在组织样本中识别特定特征，而深度学习为改进胶质母细胞瘤诊断提供了有希望的方法。本文介绍了我们对2024年BraTS-Path挑战赛的方法。我们利用预训练模型，并在BraTS-Path训练数据集上进行微调。我们的模型在Synapse在线平台严格评估的具有挑战性的BraTS-Path验证集中表现不佳，准确率为0.392229，召回率为0.392229，F1分为0.392229，表明模型在正确识别目标条件下的实例方面具有一致性能力。值得注意的是，我们的模型在负例分类方面表现出完美的特异性0.898704，显示出出色的分类能力。另计算的麦考利相关系数（MCC）为0.255267，表明预测值与实际值之间有有限的正相关，并突显了模型的整体预测能力。此外，在测试阶段，我们的解决方案获得第二名。 

---
# U-Net Based Healthy 3D Brain Tissue Inpainting 

**Title (ZH)**: U-Net 基础健康的3D脑组织修复 

**Authors**: Juexin Zhang, Ying Weng, Ke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18126)  

**Abstract**: This paper introduces a novel approach to synthesize healthy 3D brain tissue from masked input images, specifically focusing on the task of 'ASNR-MICCAI BraTS Local Synthesis of Tissue via Inpainting'. Our proposed method employs a U-Net-based architecture, which is designed to effectively reconstruct the missing or corrupted regions of brain MRI scans. To enhance our model's generalization capabilities and robustness, we implement a comprehensive data augmentation strategy that involves randomly masking healthy images during training. Our model is trained on the BraTS-Local-Inpainting dataset and demonstrates the exceptional performance in recovering healthy brain tissue. The evaluation metrics employed, including Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Mean Squared Error (MSE), consistently yields impressive results. On the BraTS-Local-Inpainting validation set, our model achieved an SSIM score of 0.841, a PSNR score of 23.257, and an MSE score of 0.007. Notably, these evaluation metrics exhibit relatively low standard deviations, i.e., 0.103 for SSIM score, 4.213 for PSNR score and 0.007 for MSE score, which indicates that our model's reliability and consistency across various input scenarios. Our method also secured first place in the challenge. 

**Abstract (ZH)**: 本文介绍了一种从蒙版输入图像合成健康3D脑组织的新型方法，特别关注“ASNR-MICCAI BraTS局部组织修复合成任务”。我们提出的方法采用基于U-Net的架构，旨在有效重建脑MRI扫描中缺失或损坏的区域。为了增强模型的泛化能力和鲁棒性，我们在训练过程中实施了全面的数据增强策略，包括随机蒙版健康图像。我们的模型在BraTS-Local-Inpainting数据集上进行训练，并展示了在恢复健康脑组织方面的出色性能。所采用的评价指标，包括结构相似性指数（SSIM）、峰值信噪比（PSNR）和均方误差（MSE），均取得了令人印象深刻的成果。在BraTS-Local-Inpainting验证集上，我们的模型取得了SSIM评分为0.841、PSNR评分为23.257和MSE评分为0.007的结果。值得注意的是，这些评价指标的标准差相对较低，即SSIM评分为0.103、PSNR评分为4.213和MSE评分为0.007，这表明我们的模型在各种输入情境下的可靠性和一致性。我们的方法也在挑战中获得了第一名。 

---
# Distributional Uncertainty for Out-of-Distribution Detection 

**Title (ZH)**: 分布不确定性进行异分布检测 

**Authors**: JinYoung Kim, DaeUng Jo, Kimin Yun, Jeonghyo Song, Youngjoon Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18106)  

**Abstract**: Estimating uncertainty from deep neural networks is a widely used approach for detecting out-of-distribution (OoD) samples, which typically exhibit high predictive uncertainty. However, conventional methods such as Monte Carlo (MC) Dropout often focus solely on either model or data uncertainty, failing to align with the semantic objective of OoD detection. To address this, we propose the Free-Energy Posterior Network, a novel framework that jointly models distributional uncertainty and identifying OoD and misclassified regions using free energy. Our method introduces two key contributions: (1) a free-energy-based density estimator parameterized by a Beta distribution, which enables fine-grained uncertainty estimation near ambiguous or unseen regions; and (2) a loss integrated within a posterior network, allowing direct uncertainty estimation from learned parameters without requiring stochastic sampling. By integrating our approach with the residual prediction branch (RPL) framework, the proposed method goes beyond post-hoc energy thresholding and enables the network to learn OoD regions by leveraging the variance of the Beta distribution, resulting in a semantically meaningful and computationally efficient solution for uncertainty-aware segmentation. We validate the effectiveness of our method on challenging real-world benchmarks, including Fishyscapes, RoadAnomaly, and Segment-Me-If-You-Can. 

**Abstract (ZH)**: 基于自由能后验网络的分布不确定性与异常区域联合建模 

---
# Fashion-AlterEval: A Dataset for Improved Evaluation of Conversational Recommendation Systems with Alternative Relevant Items 

**Title (ZH)**: Fashion-AlterEval：一种用于评估具有替代相关项的对话推荐系统的新数据集 

**Authors**: Maria Vlachou  

**Link**: [PDF](https://arxiv.org/pdf/2507.18017)  

**Abstract**: In Conversational Recommendation Systems (CRS), a user provides feedback on recommended items at each turn, leading the CRS towards improved recommendations. Due to the need for a large amount of data, a user simulator is employed for both training and evaluation. Such user simulators critique the current retrieved item based on knowledge of a single target item. However, system evaluation in offline settings with simulators is limited by the focus on a single target item and their unlimited patience over a large number of turns. To overcome these limitations of existing simulators, we propose Fashion-AlterEval, a new dataset that contains human judgments for a selection of alternative items by adding new annotations in common fashion CRS datasets. Consequently, we propose two novel meta-user simulators that use the collected judgments and allow simulated users not only to express their preferences about alternative items to their original target, but also to change their mind and level of patience. In our experiments using the Shoes and Fashion IQ as the original datasets and three CRS models, we find that using the knowledge of alternatives by the simulator can have a considerable impact on the evaluation of existing CRS models, specifically that the existing single-target evaluation underestimates their effectiveness, and when simulatedusers are allowed to instead consider alternative relevant items, the system can rapidly respond to more quickly satisfy the user. 

**Abstract (ZH)**: 在对话式推荐系统中的Fashion-AlterEval：一种包含替代物品人类判断的新数据集及其应用 

---
# Machine Unlearning of Traffic State Estimation and Prediction 

**Title (ZH)**: 交通状态估计与预测的机器遗忘技术 

**Authors**: Xin Wang, R. Tyrrell Rockafellar, Xuegang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17984)  

**Abstract**: Data-driven traffic state estimation and prediction (TSEP) relies heavily on data sources that contain sensitive information. While the abundance of data has fueled significant breakthroughs, particularly in machine learning-based methods, it also raises concerns regarding privacy, cybersecurity, and data freshness. These issues can erode public trust in intelligent transportation systems. Recently, regulations have introduced the "right to be forgotten", allowing users to request the removal of their private data from models. As machine learning models can remember old data, simply removing it from back-end databases is insufficient in such systems. To address these challenges, this study introduces a novel learning paradigm for TSEP-Machine Unlearning TSEP-which enables a trained TSEP model to selectively forget privacy-sensitive, poisoned, or outdated data. By empowering models to "unlearn," we aim to enhance the trustworthiness and reliability of data-driven traffic TSEP. 

**Abstract (ZH)**: 基于数据的交通状态估计与预测（TSEP）高度依赖包含敏感信息的数据源。虽然数据的丰富性促进了机器学习方法的重大突破，但也引发了隐私、网络安全和数据新鲜度方面的担忧。这些问题可能侵蚀公众对智能交通系统的信任。最近，法规引入了“被遗忘权”，允许用户要求从模型中删除其私人数据。由于机器学习模型会记住旧数据，仅从后端数据库中删除是不够的。为应对这些挑战，本研究引入了一种新的学习范式——TSEP机器遗忘，使得训练好的TSEP模型能够有选择地遗忘敏感、被污染或过时的数据。通过使模型能够“重新学习”，我们旨在提升基于数据的交通TSEP的信任度和可靠性。 

---
# MeAJOR Corpus: A Multi-Source Dataset for Phishing Email Detection 

**Title (ZH)**: MeAJOR语料库：诈骗邮件检测的多源数据集 

**Authors**: Paulo Mendes, Eva Maia, Isabel Praça  

**Link**: [PDF](https://arxiv.org/pdf/2507.17978)  

**Abstract**: Phishing emails continue to pose a significant threat to cybersecurity by exploiting human vulnerabilities through deceptive content and malicious payloads. While Machine Learning (ML) models are effective at detecting phishing threats, their performance largely relies on the quality and diversity of the training data. This paper presents MeAJOR (Merged email Assets from Joint Open-source Repositories) Corpus, a novel, multi-source phishing email dataset designed to overcome critical limitations in existing resources. It integrates 135894 samples representing a broad number of phishing tactics and legitimate emails, with a wide spectrum of engineered features. We evaluated the dataset's utility for phishing detection research through systematic experiments with four classification models (RF, XGB, MLP, and CNN) across multiple feature configurations. Results highlight the dataset's effectiveness, achieving 98.34% F1 with XGB. By integrating broad features from multiple categories, our dataset provides a reusable and consistent resource, while addressing common challenges like class imbalance, generalisability and reproducibility. 

**Abstract (ZH)**: 钓鱼邮件继续通过诱骗内容和恶意载荷利用人类漏洞对网络安全构成重大威胁。尽管机器学习模型在检测钓鱼威胁方面非常有效，但其性能很大程度上取决于训练数据的质量和多样性。本文介绍了MeAJOR（合并的联合开源仓库电子邮件资产）语料库，这是一个新颖的多源钓鱼邮件数据集，旨在克服现有资源的关键限制。该数据集整合了135,894个样本，涵盖了广泛的钓鱼策略和合法邮件，并具有广泛的工程特征。我们通过在多个特征配置下使用四种分类模型（RF、XGB、MLP和CNN）进行系统实验，评估了数据集在钓鱼检测研究中的实用性。结果显示，该数据集非常有效，XGB达到了98.34%的F1值。通过整合多个类别中的广泛特征，我们的数据集提供了一个可重复使用的规范资源，同时解决了常见的挑战，如类别不平衡、泛化能力和可重复性问题。 

---
# Improving the Computational Efficiency and Explainability of GeoAggregator 

**Title (ZH)**: 提高GeoAggregator的计算效率和可解释性 

**Authors**: Rui Deng, Ziqi Li, Mingshu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17977)  

**Abstract**: Accurate modeling and explaining geospatial tabular data (GTD) are critical for understanding geospatial phenomena and their underlying processes. Recent work has proposed a novel transformer-based deep learning model named GeoAggregator (GA) for this purpose, and has demonstrated that it outperforms other statistical and machine learning approaches. In this short paper, we further improve GA by 1) developing an optimized pipeline that accelerates the dataloading process and streamlines the forward pass of GA to achieve better computational efficiency; and 2) incorporating a model ensembling strategy and a post-hoc model explanation function based on the GeoShapley framework to enhance model explainability. We validate the functionality and efficiency of the proposed strategies by applying the improved GA model to synthetic datasets. Experimental results show that our implementation improves the prediction accuracy and inference speed of GA compared to the original implementation. Moreover, explanation experiments indicate that GA can effectively captures the inherent spatial effects in the designed synthetic dataset. The complete pipeline has been made publicly available for community use (this https URL). 

**Abstract (ZH)**: 准确建模和解释地理空间表型数据（GTD）对于理解地理空间现象及其 underlying 过程至关重要。近期工作提出了一种基于变压器的深度学习模型 GeoAggregator (GA) 用于此目的，并证明其优于其他统计和机器学习方法。在本文中，我们进一步改进了 GA，通过 1) 开发一个优化的工作流程以加速数据加载过程并简化 GA 的前向传播，从而提高计算效率；以及 2) 结合基于 GeoShapley 架构的模型集成策略和后验模型解释功能，以增强模型解释性。我们通过将改进后的 GA 应用于合成数据集来验证所提出策略的功能性和效率。实验结果表明，我们的实现相比原始实现提高了 GA 的预测准确性和推理速度。此外，解释实验表明 GA 能够有效捕捉设计的合成数据集中的固有空间效应。完整的管道已公开供社区使用（请参阅此链接）。 

---
# Natural Language Processing for Tigrinya: Current State and Future Directions 

**Title (ZH)**: Tigre语自然语言处理：当前状态与未来方向 

**Authors**: Fitsum Gaim, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.17974)  

**Abstract**: Despite being spoken by millions of people, Tigrinya remains severely underrepresented in Natural Language Processing (NLP) research. This work presents a comprehensive survey of NLP research for Tigrinya, analyzing over 40 studies spanning more than a decade of work from 2011 to 2025. We systematically review the current state of computational resources, models, and applications across ten distinct downstream tasks, including morphological processing, machine translation, speech recognition, and question-answering. Our analysis reveals a clear trajectory from foundational, rule-based systems to modern neural architectures, with progress consistently unlocked by resource creation milestones. We identify key challenges rooted in Tigrinya's morphological complexity and resource scarcity, while highlighting promising research directions, including morphology-aware modeling, cross-lingual transfer, and community-centered resource development. This work serves as both a comprehensive reference for researchers and a roadmap for advancing Tigrinya NLP. A curated metadata of the surveyed studies and resources is made publicly available.\footnote{Tigrinya NLP Anthology: this https URL. 

**Abstract (ZH)**: 尽管有数百万人使用提格雷尼亚语，但在自然语言处理（NLP）研究中，提格雷尼亚语仍严重缺失。本文提供了一篇关于提格雷尼亚语NLP研究的综合调查，分析了从2011年到2025年超过10年时间跨度内的超过40项研究。我们系统地回顾了在这十年间跨十个不同下游任务（包括形态学处理、机器翻译、语音识别和问答）中计算资源、模型和应用的现状。我们的分析揭示了从基于规则的基础系统向现代神经架构的明确发展轨迹，进展的一贯实现基于资源创建的重要里程碑。我们指出了根源在于提格雷尼亚语复杂形态和资源稀缺性的关键挑战，同时强调了具有前景的研究方向，包括形态意识建模、跨语言迁移和以社区为中心的资源开发。本文既是研究人员的全面参考，又是推进提格雷尼亚语NLP的路线图。被调查的研究和资源的元数据已公开发布。 

---
# VIBE: Video-Input Brain Encoder for fMRI Response Modeling 

**Title (ZH)**: 视频输入脑编码器：针对fMRI响应建模 

**Authors**: Daniel Carlstrom Schad, Shrey Dixit, Janis Keck, Viktor Studenyak, Aleksandr Shpilevoi, Andrej Bicanski  

**Link**: [PDF](https://arxiv.org/pdf/2507.17958)  

**Abstract**: We present VIBE, a two-stage Transformer that fuses multi-modal video, audio, and text features to predict fMRI activity. Representations from open-source models (Qwen2.5, BEATs, Whisper, SlowFast, V-JEPA) are merged by a modality-fusion transformer and temporally decoded by a prediction transformer with rotary embeddings. Trained on 65 hours of movie data from the CNeuroMod dataset and ensembled across 20 seeds, VIBE attains mean parcel-wise Pearson correlations of 32.25 on in-distribution Friends S07 and 21.25 on six out-of-distribution films. An earlier iteration of the same architecture obtained 0.3198 and 0.2096, respectively, winning Phase-1 and placing second overall in the Algonauts 2025 Challenge. 

**Abstract (ZH)**: VIBE：一种融合多模态视频、音频和文本特征的两阶段Transformer，用于预测fMRI活动 

---
# VERIRAG: Healthcare Claim Verification via Statistical Audit in Retrieval-Augmented Generation 

**Title (ZH)**: VERIRAG: 医疗索赔验证中的检索增强生成统计审计 

**Authors**: Shubham Mohole, Hongjun Choi, Shusen Liu, Christine Klymko, Shashank Kushwaha, Derek Shi, Wesam Sakla, Sainyam Galhotra, Ruben Glatt  

**Link**: [PDF](https://arxiv.org/pdf/2507.17948)  

**Abstract**: Retrieval-augmented generation (RAG) systems are increasingly adopted in clinical decision support, yet they remain methodologically blind-they retrieve evidence but cannot vet its scientific quality. A paper claiming "Antioxidant proteins decreased after alloferon treatment" and a rigorous multi-laboratory replication study will be treated as equally credible, even if the former lacked scientific rigor or was even retracted. To address this challenge, we introduce VERIRAG, a framework that makes three notable contributions: (i) the Veritable, an 11-point checklist that evaluates each source for methodological rigor, including data integrity and statistical validity; (ii) a Hard-to-Vary (HV) Score, a quantitative aggregator that weights evidence by its quality and diversity; and (iii) a Dynamic Acceptance Threshold, which calibrates the required evidence based on how extraordinary a claim is. Across four datasets-comprising retracted, conflicting, comprehensive, and settled science corpora-the VERIRAG approach consistently outperforms all baselines, achieving absolute F1 scores ranging from 0.53 to 0.65, representing a 10 to 14 point improvement over the next-best method in each respective dataset. We will release all materials necessary for reproducing our results. 

**Abstract (ZH)**: 增强检索生成（RAG）系统在临床决策支持中的应用：VERIRAG框架的方法学评估与证据综合 

---
# Minimax Data Sanitization with Distortion Constraint and Adversarial Inference 

**Title (ZH)**: 最小最大数据脱敏与失真约束及对抗推断 

**Authors**: Amirarsalan Moatazedian, Yauhen Yakimenka, Rémi A. Chou, Jörg Kliewer  

**Link**: [PDF](https://arxiv.org/pdf/2507.17942)  

**Abstract**: We study a privacy-preserving data-sharing setting where a privatizer transforms private data into a sanitized version observed by an authorized reconstructor and two unauthorized adversaries, each with access to side information correlated with the private data.
The reconstructor is evaluated under a distortion function, while each adversary is evaluated using a separate loss function. The privatizer ensures the reconstructor distortion remains below a fixed threshold while maximizing the minimum loss across the two adversaries. This two-adversary setting models cases where individual users cannot reconstruct the data accurately, but their combined side information enables estimation within the distortion threshold. The privatizer maximizes individual loss while permitting accurate reconstruction only through collaboration. This echoes secret-sharing principles, but with lossy rather than perfect recovery. We frame this as a constrained data-driven minimax optimization problem and propose a data-driven training procedure that alternately updates the privatizer, reconstructor, and adversaries. We also analyze the Gaussian and binary cases as special scenarios where optimal solutions can be obtained. These theoretical optimal results are benchmarks for evaluating the proposed minimax training approach. 

**Abstract (ZH)**: 我们研究一种保护隐私的数据共享设置，其中，私有化器将隐私数据转换为受授权重构者和两个未经授权的对手观察的清洗版本，每个对手都可访问与隐私数据相关联的侧信息。重构者在失真函数下进行评估，而每个对手则使用单独的损失函数进行评估。私有化器确保重构者的失真不超过固定阈值，同时最大化两个对手中最小的损失。这种双对手设置模拟了个体用户无法准确重构数据的情况，但结合的侧信息能够在失真阈值内进行估计。私有化器在允许通过合作进行准确重构的前提下，最大化个体损失。这类似于秘密共享原则，但允许损失而非完美的恢复。我们将此问题表述为受限的数据驱动最小最大优化问题，并提出了一种交替更新私有化器、重构者和对手的数据驱动训练程序。我们还分析了高斯和二进制情况作为特殊场景，其中可以得到最优解。这些理论最优结果是评估所提出的最小最大训练方法的基准。 

---
# Multimodal Fine-grained Reasoning for Post Quality Evaluation 

**Title (ZH)**: 多模态细粒度推理在帖子质量评估中的应用 

**Authors**: Xiaoxu Guo, Siyan Liang, Yachao Cui, Juxiang Zhou, Lei Wang, Han Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17934)  

**Abstract**: Accurately assessing post quality requires complex relational reasoning to capture nuanced topic-post relationships. However, existing studies face three major limitations: (1) treating the task as unimodal categorization, which fails to leverage multimodal cues and fine-grained quality distinctions; (2) introducing noise during deep multimodal fusion, leading to misleading signals; and (3) lacking the ability to capture complex semantic relationships like relevance and comprehensiveness. To address these issues, we propose the Multimodal Fine-grained Topic-post Relational Reasoning (MFTRR) framework, which mimics human cognitive processes. MFTRR reframes post-quality assessment as a ranking task and incorporates multimodal data to better capture quality variations. It consists of two key modules: (1) the Local-Global Semantic Correlation Reasoning Module, which models fine-grained semantic interactions between posts and topics at both local and global levels, enhanced by a maximum information fusion mechanism to suppress noise; and (2) the Multi-Level Evidential Relational Reasoning Module, which explores macro- and micro-level relational cues to strengthen evidence-based reasoning. We evaluate MFTRR on three newly constructed multimodal topic-post datasets and the public Lazada-Home dataset. Experimental results demonstrate that MFTRR significantly outperforms state-of-the-art baselines, achieving up to 9.52% NDCG@3 improvement over the best unimodal method on the Art History dataset. 

**Abstract (ZH)**: 准确评估帖子质量需要复杂的相关推理以捕捉细微的主题-帖子关系。然而，现有研究面临三个主要局限：（1）将任务视为单一模态分类，未能利用多模态线索和精细的质量区分；（2）在深度多模态融合过程中引入噪声，导致误导性信号；（3）缺乏捕捉相关性和全面性等复杂语义关系的能力。为解决这些问题，我们提出了一种多模态细粒度主题-帖子关系推理（MFTRR）框架，该框架模仿人类的认知过程。MFTRR将帖子质量评估重新定义为排序任务，并结合多模态数据以更好地捕捉质量变化。它包含两个关键模块：（1）局部-全局语义关系推理模块，该模块在局部和全局层面建模帖子与主题之间的细粒度语义交互，并通过最大信息融合机制抑制噪声；（2）多层次证据关系推理模块，该模块探索宏观和微观层次的关系线索以加强基于证据的推理。我们使用三个新构建的多模态主题-帖子数据集和公开的Lazada-Home数据集对MFTRR进行了评估。实验结果表明，MFTRR显著优于最先进的基线方法，在Art History数据集上实现了高达9.52%的NDCG@3改进。 

---
# UrbanPulse: A Cross-City Deep Learning Framework for Ultra-Fine-Grained Population Transfer Prediction 

**Title (ZH)**: UrbanPulse: 一种跨城市深度学习框架，用于超精细人口流动预测 

**Authors**: Hongrong Yang, Markus Schlaepfer  

**Link**: [PDF](https://arxiv.org/pdf/2507.17924)  

**Abstract**: Accurate population flow prediction is essential for urban planning, transportation management, and public health. Yet existing methods face key limitations: traditional models rely on static spatial assumptions, deep learning models struggle with cross-city generalization, and Large Language Models (LLMs) incur high computational costs while failing to capture spatial structure. Moreover, many approaches sacrifice resolution by clustering Points of Interest (POIs) or restricting coverage to subregions, limiting their utility for city-wide analytics. We introduce UrbanPulse, a scalable deep learning framework that delivers ultra-fine-grained, city-wide OD flow predictions by treating each POI as an individual node. It combines a temporal graph convolutional encoder with a transformer-based decoder to model multi-scale spatiotemporal dependencies. To ensure robust generalization across urban contexts, UrbanPulse employs a three-stage transfer learning strategy: pretraining on large-scale urban graphs, cold-start adaptation, and reinforcement learning this http URL on over 103 million cleaned GPS records from three metropolitan areas in California, UrbanPulse achieves state-of-the-art accuracy and scalability. Through efficient transfer learning, UrbanPulse takes a key step toward making high-resolution, AI-powered urban forecasting deployable in practice across diverse cities. 

**Abstract (ZH)**: 准确的人员流动预测对于城市规划、交通管理和公共卫生至关重要。然而，现有方法面临关键限制：传统模型依赖静态空间假设，深度学习模型在跨城市泛化方面挣扎，而大型语言模型在计算成本高且难以捕捉空间结构。此外，许多方法通过聚类兴趣点（POIs）或限制覆盖范围以子区域来牺牲分辨率，限制了它们在全市尺度分析中的应用价值。我们提出UrbanPulse，一种可扩展的深度学习框架，通过将每个POI视为单独节点来实现全市尺度上超细粒度的OD流预测。UrbanPulse结合了时变图卷积编码器和基于变换器的解码器，以建模多尺度时空依赖关系。为了确保在城市上下文中的稳健泛化，UrbanPulse采用三阶段迁移学习策略：大规模城市图上的预训练、冷启动适应以及基于此的强化学习。UrbanPulse在超过10300万条来自加利福尼亚三个大都市区的清洁GPS记录上进行训练，实现了最先进的准确性和可扩展性。通过高效的迁移学习，UrbanPulse朝着使高分辨率的AI驱动城市预报在各种城市中可践行迈出了一步。 

---
# From Seed to Harvest: Augmenting Human Creativity with AI for Red-teaming Text-to-Image Models 

**Title (ZH)**: 从种子到收获：利用AI增强人类创造力以测试文本到图像模型 

**Authors**: Jessica Quaye, Charvi Rastogi, Alicia Parrish, Oana Inel, Minsuk Kahng, Lora Aroyo, Vijay Janapa Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2507.17922)  

**Abstract**: Text-to-image (T2I) models have become prevalent across numerous applications, making their robust evaluation against adversarial attacks a critical priority. Continuous access to new and challenging adversarial prompts across diverse domains is essential for stress-testing these models for resilience against novel attacks from multiple vectors. Current techniques for generating such prompts are either entirely authored by humans or synthetically generated. On the one hand, datasets of human-crafted adversarial prompts are often too small in size and imbalanced in their cultural and contextual representation. On the other hand, datasets of synthetically-generated prompts achieve scale, but typically lack the realistic nuances and creative adversarial strategies found in human-crafted prompts. To combine the strengths of both human and machine approaches, we propose Seed2Harvest, a hybrid red-teaming method for guided expansion of culturally diverse, human-crafted adversarial prompt seeds. The resulting prompts preserve the characteristics and attack patterns of human prompts while maintaining comparable average attack success rates (0.31 NudeNet, 0.36 SD NSFW, 0.12 Q16). Our expanded dataset achieves substantially higher diversity with 535 unique geographic locations and a Shannon entropy of 7.48, compared to 58 locations and 5.28 entropy in the original dataset. Our work demonstrates the importance of human-machine collaboration in leveraging human creativity and machine computational capacity to achieve comprehensive, scalable red-teaming for continuous T2I model safety evaluation. 

**Abstract (ZH)**: 基于文本到图像模型的对抗攻击稳健性评估：Seed2Harvest——一种混合红队方法iliaguided扩展文化多样化的手工crafted对抗提示种子 

---
# Deep learning-aided inverse design of porous metamaterials 

**Title (ZH)**: 深度学习辅助的多孔超材料逆设计 

**Authors**: Phu Thien Nguyen, Yousef Heider, Dennis M. Kochmann, Fadi Aldakheel  

**Link**: [PDF](https://arxiv.org/pdf/2507.17907)  

**Abstract**: The ultimate aim of the study is to explore the inverse design of porous metamaterials using a deep learning-based generative framework. Specifically, we develop a property-variational autoencoder (pVAE), a variational autoencoder (VAE) augmented with a regressor, to generate structured metamaterials with tailored hydraulic properties, such as porosity and permeability. While this work uses the lattice Boltzmann method (LBM) to generate intrinsic permeability tensor data for limited porous microstructures, a convolutional neural network (CNN) is trained using a bottom-up approach to predict effective hydraulic properties. This significantly reduces the computational cost compared to direct LBM simulations. The pVAE framework is trained on two datasets: a synthetic dataset of artificial porous microstructures and CT-scan images of volume elements from real open-cell foams. The encoder-decoder architecture of the VAE captures key microstructural features, mapping them into a compact and interpretable latent space for efficient structure-property exploration. The study provides a detailed analysis and interpretation of the latent space, demonstrating its role in structure-property mapping, interpolation, and inverse design. This approach facilitates the generation of new metamaterials with desired properties. The datasets and codes used in this study will be made open-access to support further research. 

**Abstract (ZH)**: 基于深度学习的生成框架探索多孔超材料的逆向设计 

---
# VeriMinder: Mitigating Analytical Vulnerabilities in NL2SQL 

**Title (ZH)**: VeriMinder: 消除NL2SQL中的分析漏洞 

**Authors**: Shubham Mohole, Sainyam Galhotra  

**Link**: [PDF](https://arxiv.org/pdf/2507.17896)  

**Abstract**: Application systems using natural language interfaces to databases (NLIDBs) have democratized data analysis. This positive development has also brought forth an urgent challenge to help users who might use these systems without a background in statistical analysis to formulate bias-free analytical questions. Although significant research has focused on text-to-SQL generation accuracy, addressing cognitive biases in analytical questions remains underexplored. We present VeriMinder, this https URL, an interactive system for detecting and mitigating such analytical vulnerabilities. Our approach introduces three key innovations: (1) a contextual semantic mapping framework for biases relevant to specific analysis contexts (2) an analytical framework that operationalizes the Hard-to-Vary principle and guides users in systematic data analysis (3) an optimized LLM-powered system that generates high-quality, task-specific prompts using a structured process involving multiple candidates, critic feedback, and self-reflection.
User testing confirms the merits of our approach. In direct user experience evaluation, 82.5% participants reported positively impacting the quality of the analysis. In comparative evaluation, VeriMinder scored significantly higher than alternative approaches, at least 20% better when considered for metrics of the analysis's concreteness, comprehensiveness, and accuracy. Our system, implemented as a web application, is set to help users avoid "wrong question" vulnerability during data analysis. VeriMinder code base with prompts, this https URL, is available as an MIT-licensed open-source software to facilitate further research and adoption within the community. 

**Abstract (ZH)**: 基于自然语言界面的数据库应用系统（NLIDBs）已使数据剖析民主化。这一积极的发展也带来了一个紧迫的挑战，即帮助那些可能未接受过统计分析背景教育的用户，提出无偏见的分析问题。尽管对文本到SQL生成的准确性进行了大量研究，但在解决分析问题中的认知偏见方面仍然研究不足。我们提出VeriMinder，这是一个用于检测和缓解此类分析漏洞的交互式系统。我们的方法引入了三项关键技术创新：（1）与特定分析上下文相关的偏见上下文语义映射框架；（2）将难以变化原则操作化的分析框架，并引导用户进行系统数据分析；（3）优化的大规模语言模型驱动系统，使用结构化流程生成高质量、特定任务的提示，该流程涉及多个候选人、批评反馈和自我反思。用户测试证实了我们方法的优势。在直接用户体验评估中，82.5%的参与者报告说提高了分析的质量。在比较评估中，VeriMinder在分析具体性、全面性和准确性等指标上显著优于其他方法，至少提高了20%。我们的系统作为网络应用实现，旨在帮助用户避免数据分析中的“错误问题”漏洞。VeriMinder代码库和提示，已作为MIT许可的开源软件发布，以促进社区内的进一步研究和采用。 

---
# Action-List Reinforcement Learning Syndrome Decoding for Binary Linear Block Codes 

**Title (ZH)**: 基于动作列表强化学习的二进制线性块码译码算法 

**Authors**: Milad Taghipour, Bane Vasic  

**Link**: [PDF](https://arxiv.org/pdf/2507.17893)  

**Abstract**: This paper explores the application of reinforcement learning techniques to enhance the performance of decoding of linear block codes based on flipping bits and finding optimal decisions. We describe the methodology for mapping the iterative decoding process into Markov Decision Processes (MDPs) and propose different methods to reduce the number of states in the MDP. A truncated MDP is proposed to reduce the number of states in the MDP by learning a Hamming ball with a specified radius around codewords. We then propose a general scheme for reinforcement learning based decoders applicable to any class of codes to improve the performance of decoders. We call this scheme an action-list decoding. We design an action-list decoder based on the Deep-Q network values that substantially enhance performance. We also get benefit of automorphism group of code to further improve the code performance. Additionally, we propose a feedback-based method to exploit and enhance the performance of existing high-performing decoders by applying reinforcement learning algorithms after the existing decoders. These approaches effectively reduces the complexity of the reinforcement learning block. Finally, we present experimental results for the Low-Density Parity Check (LDPC) codes over the Binary Symmetric Channel (BSC) to demonstrate the efficiency of the proposed methods. 

**Abstract (ZH)**: 基于强化学习技术提高翻比特线性分组码译码性能的方法研究 

---
# Towards Facilitated Fairness Assessment of AI-based Skin Lesion Classifiers Through GenAI-based Image Synthesis 

**Title (ZH)**: 基于GenAI图像合成促进基于AI的皮肤病变分类器公平性评估 

**Authors**: Ko Watanabe. Stanislav Frolov. Adriano Lucieri. Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2507.17860)  

**Abstract**: Recent advancements in Deep Learning and its application on the edge hold great potential for the revolution of routine screenings for skin cancers like Melanoma. Along with the anticipated benefits of this technology, potential dangers arise from unforseen and inherent biases. Thus, assessing and improving the fairness of such systems is of utmost importance. A key challenge in fairness assessment is to ensure that the evaluation dataset is sufficiently representative of different Personal Identifiable Information (PII) (sex, age, and race) and other minority groups. Against the backdrop of this challenge, this study leverages the state-of-the-art Generative AI (GenAI) LightningDiT model to assess the fairness of publicly available melanoma classifiers. The results suggest that fairness assessment using highly realistic synthetic data is a promising direction. Yet, our findings indicate that verifying fairness becomes difficult when the melanoma-detection model used for evaluation is trained on data that differ from the dataset underpinning the synthetic images. Nonetheless, we propose that our approach offers a valuable new avenue for employing synthetic data to gauge and enhance fairness in medical-imaging GenAI systems. 

**Abstract (ZH)**: Recent advancements in深度学习及其在边缘的应用为皮肤癌如黑色素瘤的常规筛查革命带来了巨大潜力。然而，随着这项技术预期带来的益处，潜在的风险也来自未预见和固有的偏见。因此，评估和改进此类系统的公平性至关重要。公平性评估的关键挑战在于确保评估数据集充分代表不同个人可识别信息（如性别、年龄和种族）以及其他少数群体。面对这一挑战，本研究利用最先进的生成人工智能（GenAI）LightningDiT模型评估公开可用的黑色素瘤分类器的公平性。结果表明，使用高度真实的合成数据进行公平性评估是一个有前景的方向。然而，我们的研究发现，在评估的黑色素瘤检测模型所使用的训练数据与合成图像底层数据集不同时，验证公平性变得困难。尽管如此，我们提出我们的方法为使用合成数据评估和提升医疗成像GenAI系统的公平性提供了一个宝贵的新途径。 

---
# Technical Implementation of Tippy: Multi-Agent Architecture and System Design for Drug Discovery Laboratory Automation 

**Title (ZH)**: Tippy的技術實現：药物发现实验室自动化多agent架构与系统设计 

**Authors**: Yao Fehlis, Charles Crain, Aidan Jensen, Michael Watson, James Juhasz, Paul Mandel, Betty Liu, Shawn Mahon, Daren Wilson, Nick Lynch-Jonely, Ben Leedom, David Fuller  

**Link**: [PDF](https://arxiv.org/pdf/2507.17852)  

**Abstract**: Building on the conceptual framework presented in our previous work on agentic AI for pharmaceutical research, this paper provides a comprehensive technical analysis of Tippy's multi-agent system implementation for drug discovery laboratory automation. We present a distributed microservices architecture featuring five specialized agents (Supervisor, Molecule, Lab, Analysis, and Report) that coordinate through OpenAI Agents SDK orchestration and access laboratory tools via the Model Context Protocol (MCP). The system architecture encompasses agent-specific tool integration, asynchronous communication patterns, and comprehensive configuration management through Git-based tracking. Our production deployment strategy utilizes Kubernetes container orchestration with Helm charts, Docker containerization, and CI/CD pipelines for automated testing and deployment. The implementation integrates vector databases for RAG functionality and employs an Envoy reverse proxy for secure external access. This work demonstrates how specialized AI agents can effectively coordinate complex laboratory workflows while maintaining security, scalability, reliability, and integration with existing laboratory infrastructure through standardized protocols. 

**Abstract (ZH)**: 基于我们在制药研究中提出的代理型AI概念框架，本文提供了Tippy多智能体系统在药物发现实验室自动化方面的全面技术分析。该系统采用分布式微服务架构，包含五个专业化智能体（Supervisor、Molecule、Lab、Analysis和Report），并通过OpenAI Agents SDK编排协调，借助Model Context Protocol（MCP）访问实验室工具。系统架构涵盖了智能体特定工具集成、异步通信模式和基于Git的配置管理。我们的生产部署策略使用Kubernetes容器编排、Helm图表、Docker容器化和CI/CD管道进行自动化测试和部署。实现中整合了向量数据库以支持RAG功能，并使用Envoymesh代理以安全方式访问外部资源。本工作展示了如何通过标准化协议使专业化AI智能体有效协调复杂的实验室工作流程，同时保持安全、可扩展性、可靠性和与现有实验室基础设施的集成。 

---
# Performance Evaluation and Threat Mitigation in Large-scale 5G Core Deployment 

**Title (ZH)**: 大规模5G核心网部署的性能评估与威胁缓解 

**Authors**: Rodrigo Moreira, Larissa F. Rodrigues Moreira, Flávio de Oliveira Silva  

**Link**: [PDF](https://arxiv.org/pdf/2507.17850)  

**Abstract**: The deployment of large-scale software-based 5G core functions presents significant challenges due to their reliance on optimized and intelligent resource provisioning for their services. Many studies have focused on analyzing the impact of resource allocation for complex deployments using mathematical models, queue theories, or even Artificial Intelligence (AI). This paper elucidates the effects of chaotic workloads, generated by Distributed Denial of Service (DDoS) on different Network Functions (NFs) on User Equipment registration performance. Our findings highlight the necessity of diverse resource profiles to ensure Service-Level Agreement (SLA) compliance in large-scale 5G core deployments. Additionally, our analysis of packet capture approaches demonstrates the potential of kernel-based monitoring for scalable security threat defense. Finally, our empirical evaluation provides insights into the effective deployment of 5G NFs in complex scenarios. 

**Abstract (ZH)**: 大规模软件基5G核心功能的部署因其服务依赖于优化和智能化的资源分配而面临显著挑战。许多研究专注于使用数学模型、队列理论或人工智能分析复杂部署中的资源分配影响。本文阐明了分布式拒绝服务（DDoS）生成的混沌工作负载对用户设备注册性能在不同网络功能（NFs）上的影响。我们的研究发现强调了在大规模5G核心部署中确保服务级别协议（SLA）合规性的需求。此外，我们对数据包捕获方法的分析表明基于内核的监控在可扩展的安全威胁防御中的潜力。最后，我们的实证评估提供了在复杂场景中有效部署5G NFs的见解。 

---
# Explainable Graph Neural Networks via Structural Externalities 

**Title (ZH)**: 具有结构性外部性的可解释图神经网络 

**Authors**: Lijun Wu, Dong Hao, Zhiyi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17848)  

**Abstract**: Graph Neural Networks (GNNs) have achieved outstanding performance across a wide range of graph-related tasks. However, their "black-box" nature poses significant challenges to their explainability, and existing methods often fail to effectively capture the intricate interaction patterns among nodes within the network. In this work, we propose a novel explainability framework, GraphEXT, which leverages cooperative game theory and the concept of social externalities. GraphEXT partitions graph nodes into coalitions, decomposing the original graph into independent subgraphs. By integrating graph structure as an externality and incorporating the Shapley value under externalities, GraphEXT quantifies node importance through their marginal contributions to GNN predictions as the nodes transition between coalitions. Unlike traditional Shapley value-based methods that primarily focus on node attributes, our GraphEXT places greater emphasis on the interactions among nodes and the impact of structural changes on GNN predictions. Experimental studies on both synthetic and real-world datasets show that GraphEXT outperforms existing baseline methods in terms of fidelity across diverse GNN architectures , significantly enhancing the explainability of GNN models. 

**Abstract (ZH)**: 基于合作博弈论和社交外部性概念的GraphEXT解释框架已经在多种图神经网络架构上展示了更高的保真度，显著增强了图神经网络模型的解释性。 

---
# Towards Robust Foundation Models for Digital Pathology 

**Title (ZH)**: 面向数字病理学的稳健基础模型研究 

**Authors**: Jonah Kömen, Edwin D. de Jong, Julius Hense, Hannah Marienwald, Jonas Dippel, Philip Naumann, Eric Marcus, Lukas Ruff, Maximilian Alber, Jonas Teuwen, Frederick Klauschen, Klaus-Robert Müller  

**Link**: [PDF](https://arxiv.org/pdf/2507.17845)  

**Abstract**: Biomedical Foundation Models (FMs) are rapidly transforming AI-enabled healthcare research and entering clinical validation. However, their susceptibility to learning non-biological technical features -- including variations in surgical/endoscopic techniques, laboratory procedures, and scanner hardware -- poses risks for clinical deployment. We present the first systematic investigation of pathology FM robustness to non-biological features. Our work (i) introduces measures to quantify FM robustness, (ii) demonstrates the consequences of limited robustness, and (iii) proposes a framework for FM robustification to mitigate these issues. Specifically, we developed PathoROB, a robustness benchmark with three novel metrics, including the robustness index, and four datasets covering 28 biological classes from 34 medical centers. Our experiments reveal robustness deficits across all 20 evaluated FMs, and substantial robustness differences between them. We found that non-robust FM representations can cause major diagnostic downstream errors and clinical blunders that prevent safe clinical adoption. Using more robust FMs and post-hoc robustification considerably reduced (but did not yet eliminate) the risk of such errors. This work establishes that robustness evaluation is essential for validating pathology FMs before clinical adoption and demonstrates that future FM development must integrate robustness as a core design principle. PathoROB provides a blueprint for assessing robustness across biomedical domains, guiding FM improvement efforts towards more robust, representative, and clinically deployable AI systems that prioritize biological information over technical artifacts. 

**Abstract (ZH)**: 生物医学基础模型（FMs）正在迅速改变AI驱动的医疗保健研究，并进入临床验证阶段。然而，它们学习非生物学技术特征的能力——包括手术/内镜技术、实验室程序和扫描硬件的差异——增加了在临床部署中的风险。我们首次系统地调查了病理FM对非生物学特征的鲁棒性。我们的工作（i）引入了衡量FM鲁棒性的方法，（ii）展示了鲁棒性有限的后果，并（iii）提出了一个FM鲁棒化框架以减轻这些问题。具体而言，我们开发了PathoROB，一个包含三个新指标的鲁棒性基准，包括鲁棒性指数，以及涵盖34家医疗机构28个生物类别的四个数据集。我们的实验表明，在评估的20个FM中普遍存在鲁棒性不足，且它们之间的鲁棒性差异显著。我们发现，不鲁棒的FM表示可能导致重大诊断错误和临床失误，阻碍其安全的临床应用。使用更鲁棒的FM和事后鲁棒化显著减少了（但尚未完全消除）此类错误的风险。这项工作确立了在临床采用前评估病理FM鲁棒性的必要性，并证明了未来FM开发必须将鲁棒性作为核心设计原则。PathoROB为评估生物医学领域鲁棒性提供了蓝图，指导了FM改进努力，以实现更鲁棒、更具代表性和临床可部署的AI系统，优先考虑生物学信息而非技术特征。 

---
# Helix 1.0: An Open-Source Framework for Reproducible and Interpretable Machine Learning on Tabular Scientific Data 

**Title (ZH)**: Helix 1.0：一种用于表格科学数据可再现和可解释的机器学习的开源框架 

**Authors**: Eduardo Aguilar-Bejarano, Daniel Lea, Karthikeyan Sivakumar, Jimiama M. Mase, Reza Omidvar, Ruizhe Li, Troy Kettle, James Mitchell-White, Morgan R Alexander, David A Winkler, Grazziela Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2507.17791)  

**Abstract**: Helix is an open-source, extensible, Python-based software framework to facilitate reproducible and interpretable machine learning workflows for tabular data. It addresses the growing need for transparent experimental data analytics provenance, ensuring that the entire analytical process -- including decisions around data transformation and methodological choices -- is documented, accessible, reproducible, and comprehensible to relevant stakeholders. The platform comprises modules for standardised data preprocessing, visualisation, machine learning model training, evaluation, interpretation, results inspection, and model prediction for unseen data. To further empower researchers without formal training in data science to derive meaningful and actionable insights, Helix features a user-friendly interface that enables the design of computational experiments, inspection of outcomes, including a novel interpretation approach to machine learning decisions using linguistic terms all within an integrated environment. Released under the MIT licence, Helix is accessible via GitHub and PyPI, supporting community-driven development and promoting adherence to the FAIR principles. 

**Abstract (ZH)**: Helix是一个开源可扩展的基于Python的软件框架，用于促进表格数据的可重现和可解释机器学习工作流。该框架解决了日益增长的透明实验数据分析源头需求，确保整个分析过程，包括数据转换和方法论选择的决策，都是记录在案、可访问、可重现和易于相关利益相关者理解的。该平台包括标准化数据预处理、可视化、机器学习模型训练、评估、解释、结果检查以及对未见数据的模型预测模块。为了进一步赋能没有正式数据科学训练的研究人员，获取有意义和可操作的洞察，Helix提供了一个用户友好的界面，允许设计计算实验、检查结果，包括一种新的基于语言术语的机器学习决策解释方法，均在一个集成环境中完成。Helix在MIT许可下发布，可通过GitHub和PyPI访问，支持社区驱动开发并促进遵守FAIR原则。 

---
# In Reverie Together: Ten Years of Mathematical Discovery with a Machine Collaborator 

**Title (ZH)**: 与机器合作者共梦十年：数学发现之旅 

**Authors**: Randy Davila, Boris Brimkov, Ryan Pepper  

**Link**: [PDF](https://arxiv.org/pdf/2507.17780)  

**Abstract**: We present four open conjectures in graph theory generated by the automated conjecturing system \texttt{TxGraffiti}. Each conjecture is concise, grounded in natural graph invariants, and empirically validated across hundreds of graphs. Despite extensive effort, these statements remain unresolved--defying both proof and counterexample. They are not only mathematical challenges but creative expressions--born of symbolic pattern recognition and mathematician-defined heuristics, refined through years of human dialogue, and now offered back to the community as collaborative artifacts. These conjectures invite not only formal proof, but also reflection on how machines can evoke wonder, spark curiosity, and contribute to the raw material of discovery. By highlighting these problems, we aim to inspire both human mathematicians and AI systems to engage with them--not only to solve them, but to reflect on what it means when machines participate meaningfully in the creative process of mathematical thought. 

**Abstract (ZH)**: 我们展示了由自动猜想系统\texttt{TxGraffiti}生成的四个开放猜想。每个猜想简洁明了，基于自然图不变量，并在数百个图中得到经验验证。尽管付出大量努力，这些陈述仍未解决——既无法证明也无法找到反例。它们不仅是数学挑战，也是富有创造性的表达——源自符号模式识别和数学家定义的启发式方法，并通过多年的人机对话 refinement，现在作为协作成果回馈给社区。这些猜想不仅邀请正式证明，还引发了如何激发惊奇感、激发求知欲以及为发现的基础材料做贡献的思考。通过突出这些难题，我们旨在激励人类数学家和AI系统与之互动——不仅是为了解决它们，更是为了反思当机器以有意义的方式参与数学思考的创造过程时意味着什么。 

---
# An advanced AI driven database system 

**Title (ZH)**: 一种先进的AI驱动数据库系统 

**Authors**: M. Tedeschi, S. Rizwan, C. Shringi, V. Devram Chandgir, S. Belich  

**Link**: [PDF](https://arxiv.org/pdf/2507.17778)  

**Abstract**: Contemporary database systems, while effective, suffer severe issues related to complexity and usability, especially among individuals who lack technical expertise but are unfamiliar with query languages like Structured Query Language (SQL). This paper presents a new database system supported by Artificial Intelligence (AI), which is intended to improve the management of data using natural language processing (NLP) - based intuitive interfaces, and automatic creation of structured queries and semi-structured data formats like yet another markup language (YAML), java script object notation (JSON), and application program interface (API) documentation. The system is intended to strengthen the potential of databases through the integration of Large Language Models (LLMs) and advanced machine learning algorithms. The integration is purposed to allow the automation of fundamental tasks such as data modeling, schema creation, query comprehension, and performance optimization. We present in this paper a system that aims to alleviate the main problems with current database technologies. It is meant to reduce the need for technical skills, manual tuning for better performance, and the potential for human error. The AI database employs generative schema inference and format selection to build its schema models and execution formats. 

**Abstract (ZH)**: 当代数据库系统虽然有效，但在复杂性和易用性方面存在严重问题，尤其是在缺乏技术背景但不熟悉查询语言（如结构化查询语言SQL）的个体中更为明显。本文提出一种以人工智能（AI）支持的新数据库系统，旨在通过基于自然语言处理（NLP）的直观界面和自动创建结构化查询及半结构化数据格式（如YAML、JSON和API文档），改进数据管理。该系统通过集成大规模语言模型（LLMs）和先进机器学习算法，旨在增强数据库的潜在能力，并允许自动化数据建模、模式创建、查询理解及性能优化等基本任务。本文提出了一种旨在缓解当前数据库技术主要问题的系统，旨在降低对技术技能的需求、手动调优以提高性能的需求以及人为错误的可能性。该AI数据库采用生成性模式推断和格式选择来构建其模式模型和执行格式。 

---
# Axiomatizing Rumsfeld Ignorance 

**Title (ZH)**: 公理化鲁斯福德无知 

**Authors**: Jie Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17776)  

**Abstract**: In a recent paper, Kit Fine presents some striking results concerning the logical properties of (first-order) ignorance, second-order ignorance and Rumsfeld ignorance. However, Rumsfeld ignorance is definable in terms of ignorance, which makes some existing results and the axiomatization problem trivial. A main reason is that the accessibility relations for the implicit knowledge operator contained in the packaged operators of ignorance and Rumsfeld ignorance are the same. In this work, we assume the two accessibility relations to be different so that one of them is an arbitrary subset of the other. This will avoid the definability issue and retain most of the previous validities. The main results are axiomatizations over various proper bi-frame classes. Finally we apply our framework to analyze Fine's results. 

**Abstract (ZH)**: 近期，Kit Fine发表了一篇论文，探讨了（一阶）无知、二阶无知和Rumsfeld无知的逻辑性质。然而，Rumsfeld无知可以通过无知来定义，这使得一些现有结果和公理化问题变得平凡。主要原因在于包含在无知和Rumsfeld无知打包操作符中的隐含知识操作符的可达关系相同。在本文中，我们假设这两种可达关系不同，使得其中之一是另一者的任意子集。这将避免定义问题并保留大多数之前的有效性。主要结果是对各种适当双框架类的公理化。最后，我们将该框架应用于分析Fine的结果。 

---
# Comparison of Optimised Geometric Deep Learning Architectures, over Varying Toxicological Assay Data Environments 

**Title (ZH)**: 优化几何深度学习架构在不同毒理学检测数据环境中的比较 

**Authors**: Alexander D. Kalian, Lennart Otte, Jaewook Lee, Emilio Benfenati, Jean-Lou C.M. Dorne, Claire Potter, Olivia J. Osborne, Miao Guo, Christer Hogstrand  

**Link**: [PDF](https://arxiv.org/pdf/2507.17775)  

**Abstract**: Geometric deep learning is an emerging technique in Artificial Intelligence (AI) driven cheminformatics, however the unique implications of different Graph Neural Network (GNN) architectures are poorly explored, for this space. This study compared performances of Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs) and Graph Isomorphism Networks (GINs), applied to 7 different toxicological assay datasets of varying data abundance and endpoint, to perform binary classification of assay activation. Following pre-processing of molecular graphs, enforcement of class-balance and stratification of all datasets across 5 folds, Bayesian optimisations were carried out, for each GNN applied to each assay dataset (resulting in 21 unique Bayesian optimisations). Optimised GNNs performed at Area Under the Curve (AUC) scores ranging from 0.728-0.849 (averaged across all folds), naturally varying between specific assays and GNNs. GINs were found to consistently outperform GCNs and GATs, for the top 5 of 7 most data-abundant toxicological assays. GATs however significantly outperformed over the remaining 2 most data-scarce assays. This indicates that GINs are a more optimal architecture for data-abundant environments, whereas GATs are a more optimal architecture for data-scarce environments. Subsequent analysis of the explored higher-dimensional hyperparameter spaces, as well as optimised hyperparameter states, found that GCNs and GATs reached measurably closer optimised states with each other, compared to GINs, further indicating the unique nature of GINs as a GNN algorithm. 

**Abstract (ZH)**: 几何深度学习是人工智能驱动计算化学中的新兴技术，然而不同图神经网络（GNN）架构的独特影响在该领域研究不足。本研究对比了图卷积网络（GCNs）、图注意网络（GATs）和图同构网络（GINs）在七个不同毒性 assay 数据集上的性能，这些数据集具有不同的数据量和终点，用于执行 assay 活性二分类。在分子图预处理、类平衡强制执行和所有数据集在5折中的分层后，为每个assay数据集应用的每个GNN进行了贝叶斯优化（总共21次贝叶斯优化）。优化后的GNN在所有折的曲线下面积（AUC）分数范围从0.728到0.849，特定assay和GNN之间自然有所不同。对于七个数据最丰富的毒性assay的前五种，GINs持续优于GCNs和GATs。然而，GATs在两个数据最稀缺的assay中显著优于其他模型。这表明，GINs在数据丰富的环境中更优，而GATs在数据稀缺的环境中更优。进一步分析探索的高维度超参数空间及其优化状态发现，GCNs和GATs达到的优化状态更接近，相较于GINs，这进一步表明GINs作为图神经网络算法的独特性质。 

---
# Caching Techniques for Reducing the Communication Cost of Federated Learning in IoT Environments 

**Title (ZH)**: 物联网环境中降低联邦学习通信成本的缓存技术 

**Authors**: Ahmad Alhonainy, Praveen Rao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17772)  

**Abstract**: Federated Learning (FL) allows multiple distributed devices to jointly train a shared model without centralizing data, but communication cost remains a major bottleneck, especially in resource-constrained environments. This paper introduces caching strategies - FIFO, LRU, and Priority-Based - to reduce unnecessary model update transmissions. By selectively forwarding significant updates, our approach lowers bandwidth usage while maintaining model accuracy. Experiments on CIFAR-10 and medical datasets show reduced communication with minimal accuracy loss. Results confirm that intelligent caching improves scalability, memory efficiency, and supports reliable FL in edge IoT networks, making it practical for deployment in smart cities, healthcare, and other latency-sensitive applications. 

**Abstract (ZH)**: 联邦学习(Federated Learning)允许多个分布式的设备联合训练共享模型而无需集中数据，但通信成本仍然是主要瓶颈，尤其是在资源受限的环境中。本文介绍了使用FIFO、LRU和基于优先级的缓存策略来减少不必要的模型更新传输。通过选择性地转发重要的更新，我们的方法降低了带宽使用量的同时保持了模型准确性。实验结果表明，在CIFAR-10和医疗数据集上减少了通信量，并且几乎不影响准确性。结果证实，智能缓存提高了联邦学习的可扩展性、内存效率，并支持边缘物联网网络中的可靠联邦学习，使其在智能城市、医疗保健等对延迟敏感的应用中具有实际部署价值。 

---
# ASR-Guided Speaker-Role Diarization and Diarization-Guided ASR Decoding 

**Title (ZH)**: ASR引导的说话人角色辨识与辨识引导的ASR解码 

**Authors**: Arindam Ghosh, Mark Fuhs, Bongjun Kim, Anurag Chowdhury, Monika Woszczyna  

**Link**: [PDF](https://arxiv.org/pdf/2507.17765)  

**Abstract**: From an application standpoint, speaker-role diarization (RD), such as doctor vs. patient, host vs. guest, etc. is often more useful than traditional speaker diarization (SD), which assigns generic labels like speaker-1, speaker-2 etc. In the context of joint automatic speech recognition (ASR) + SD (who spoke what?), recent end-to-end models employ an auxiliary SD transducer, synchronized with the ASR transducer, to predict speakers per word. In this paper, we extend this framework to RD with three key contributions: (1) we simplify the training via forced alignment and cross-entropy loss instead of RNNT loss, (2) we show that word prediction and role prediction require different amounts of predictor's context, leading to separate task-specific predictors, unlike existing shared-predictor models, and (3) we propose a way to leverage RD posterior activity to influence ASR decoding and reduce small-word deletion errors. 

**Abstract (ZH)**: 从应用角度来看，说话人角色分割（RD），如医生 vs. 患者、主持人 vs. 客人等，通常比传统说话人分割（SD）更具有实用性，后者使用如speaker-1、speaker-2等通用标签。在联合自动语音识别（ASR）+ SD（谁说了什么？）的上下文中，最近的端到端模型通过同步的ASR转录机和辅助SD转录机来预测每词的说话人。在本文中，我们通过三种关键贡献将此框架扩展到RD：（1）我们通过强制对齐和交叉熵损失简化训练，而不是使用RNNT损失；（2）我们表明词预测和角色预测需要不同类型预测器的不同上下文，导致分离的任务特定预测器，不同于现有共享预测器模型；（3）我们提出了一种利用RD后验活动来影响ASR解码并减少短词删除错误的方法。 

---
# How Instructional Sequence and Personalized Support Impact Diagnostic Strategy Learning 

**Title (ZH)**: 指令序列和个人化支持对诊断策略学习的影响 

**Authors**: Fatma Betül Güreş, Tanya Nazaretsky, Bahar Radmehr, Martina Rau, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2507.17760)  

**Abstract**: Supporting students in developing effective diagnostic reasoning is a key challenge in various educational domains. Novices often struggle with cognitive biases such as premature closure and over-reliance on heuristics. Scenario-based learning (SBL) can address these challenges by offering realistic case experiences and iterative practice, but the optimal sequencing of instruction and problem-solving activities remains unclear. This study examines how personalized support can be incorporated into different instructional sequences and whether providing explicit diagnostic strategy instruction before (I-PS) or after problem-solving (PS-I) improves learning and its transfer. We employ a between-groups design in an online SBL environment called PharmaSim, which simulates real-world client interactions for pharmacy technician apprentices. Results indicate that while both instruction types are beneficial, PS-I leads to significantly higher performance in transfer tasks. 

**Abstract (ZH)**: 支持学生发展有效的诊断推理是各个教育领域的一项关键挑战。新手常受到认知偏差如过早封闭和过度依赖启发式的困扰。基于情景的学习可以通过提供现实案例经验和迭代实践来应对这些挑战，但最优的教学序列和问题解决活动的排列尚不明确。本研究探讨了不同教学序列中个性化支持的融入方式，并分析了在问题解决前（I-PS）还是问题解决后（PS-I）提供明确的诊断策略指导是否能更有效地促进学习及其迁移。我们使用一个名为PharmaSim的在线基于情景的学习环境，该环境模拟了药房技术员学徒与实际客户互动的情景。研究结果表明，虽然两种指导类型都有益处，但PS-I的教学序列在迁移任务中的表现显著更高。 

---
# Insights from Railway Professionals: Rethinking Railway assumptions regarding safety and autonomy 

**Title (ZH)**: 来自铁路专业人员的见解：重新审视铁路关于安全与自主性的假设 

**Authors**: Josh Hunter, John McDermid, Simon Burton  

**Link**: [PDF](https://arxiv.org/pdf/2507.17756)  

**Abstract**: This study investigates how railway professionals perceive safety as a concept within rail, with the intention to help inform future technological developments within the industry. Through a series of interviews with drivers, route planners,and administrative personnel, the research explores the currentstate of safety practices, the potential for automation and the understanding of the railway as a system of systems. Key findings highlight a cautious attitude towards automation, a preference for assistive technologies, and a complex understanding of safety that integrates human, systematic and technological factors. The study also addresses the limitations of transferring automotive automation technologies to railways and the need for a railway-specific causation model to better evaluate and enhance safety in an evolving technological landscape. This study aims to bridge thegap between contemporary research and practical applications, contributing to the development of more effective safety metrics. 

**Abstract (ZH)**: 本研究调查了铁路专业人员对 rail 领域安全这一概念的看法，旨在帮助未来的技术发展提供参考。通过与驾驶员、路线规划员和行政人员的一系列访谈，研究探索了当前的安全实践状况、自动化Potential及其对铁路作为一个系统网络的理解。关键发现表明，对自动化持谨慎态度、更倾向于辅助技术，并对安全的概念有复杂的理解，整合了人类、系统和技术因素。研究还讨论了将汽车自动化技术转移到铁路领域的局限性，并强调了需要针对铁路的因果模型来更好地评估和提升安全水平。本研究旨在弥合当代研究与实际应用之间的差距，为更有效的安全指标的发展作出贡献。 

---
# A Custom-Built Ambient Scribe Reduces Cognitive Load and Documentation Burden for Telehealth Clinicians 

**Title (ZH)**: 自定义构建的环境记录员减轻了远程健康 clinicians 的认知负荷和记录负担。 

**Authors**: Justin Morse, Kurt Gilbert, Kyle Shin, Rick Cooke, Peyton Rose, Jack Sullivan, Angelo Sisante  

**Link**: [PDF](https://arxiv.org/pdf/2507.17754)  

**Abstract**: Clinician burnout has motivated the growing adoption of ambient medical scribes in the clinic. In this work, we introduce a custom-built ambient scribe application integrated into the EHR system at Included Health, a personalized all-in-one healthcare company offering telehealth services. The application uses Whisper for transcription and a modular in-context learning pipeline with GPT-4o to automatically generate SOAP notes and patient instructions. Testing on mock visit data shows that the notes generated by the application exceed the quality of expert-written notes as determined by an LLM-as-a-judge. The application has been widely adopted by the clinical practice, with over 540 clinicians at Included Health using the application at least once. 94% (n = 63) of surveyed clinicians report reduced cognitive load during visits and 97% (n = 66) report less documentation burden when using the application. Additionally, we show that post-processing notes with a fine-tuned BART model improves conciseness. These findings highlight the potential for AI systems to ease administrative burdens and support clinicians in delivering efficient, high-quality care. 

**Abstract (ZH)**: 临床医生的burnout促使了门诊中ambient medical scribes的广泛应用。在此工作中，我们介绍了一个集成于Included Health EHR系统的自定义ambient scribe应用，Included Health是一家提供个性化一站式医疗服务并涵盖远程医疗服务的公司。该应用使用Whisper进行转录，并结合GPT-4o的模块化上下文学习流水线自动生成SOAP笔记和患者指导。测试结果显示，应用生成的笔记在LLM-as-a-judge的评估中质量超过专家手写的笔记。该应用已被广泛采用，在Included Health有超过540名临床医生至少使用一次。在接受调查的临床医生中，94%（n=63）的人报告称，在使用该应用时访视中的认知负荷减少，97%（n=66）的人报告称，使用该应用时记录文档的负担减轻。此外，我们还展示了使用微调的BART模型处理笔记可以提高简洁性。这些发现突显了AI系统在减轻行政负担和支持临床医生提供高效、高质量护理方面的潜力。 

---
