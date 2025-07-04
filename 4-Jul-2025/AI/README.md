# StepHint: Multi-level Stepwise Hints Enhance Reinforcement Learning to Reason 

**Title (ZH)**: StepHint: 多层次逐步提示增强强化学习推理 

**Authors**: Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2507.02841)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving the complex reasoning abilities of large language models (LLMs). However, current RLVR methods face two significant challenges: the near-miss reward problem, where a small mistake can invalidate an otherwise correct reasoning process, greatly hindering training efficiency; and exploration stagnation, where models tend to focus on solutions within their ``comfort zone,'' lacking the motivation to explore potentially more effective alternatives. To address these challenges, we propose StepHint, a novel RLVR algorithm that utilizes multi-level stepwise hints to help models explore the solution space more effectively. StepHint generates valid reasoning chains from stronger models and partitions these chains into reasoning steps using our proposed adaptive partitioning method. The initial few steps are used as hints, and simultaneously, multiple-level hints (each comprising a different number of steps) are provided to the model. This approach directs the model's exploration toward a promising solution subspace while preserving its flexibility for independent exploration. By providing hints, StepHint mitigates the near-miss reward problem, thereby improving training efficiency. Additionally, the external reasoning pathways help the model develop better reasoning abilities, enabling it to move beyond its ``comfort zone'' and mitigate exploration stagnation. StepHint outperforms competitive RLVR enhancement methods across six mathematical benchmarks, while also demonstrating superior generalization and excelling over baselines on out-of-domain benchmarks. 

**Abstract (ZH)**: 可验证回报的强化学习（RLVR）方法：增强大型语言模型的复杂推理能力 

---
# Establishing Best Practices for Building Rigorous Agentic Benchmarks 

**Title (ZH)**: 建立严谨代理基准的最佳实践 

**Authors**: Yuxuan Zhu, Tengjun Jin, Yada Pruksachatkun, Andy Zhang, Shu Liu, Sasha Cui, Sayash Kapoor, Shayne Longpre, Kevin Meng, Rebecca Weiss, Fazl Barez, Rahul Gupta, Jwala Dhamala, Jacob Merizian, Mario Giulianelli, Harry Coppock, Cozmin Ududec, Jasjeet Sekhon, Jacob Steinhardt, Antony Kellerman, Sarah Schwettmann, Matei Zaharia, Ion Stoica, Percy Liang, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02825)  

**Abstract**: Benchmarks are essential for quantitatively tracking progress in AI. As AI agents become increasingly capable, researchers and practitioners have introduced agentic benchmarks to evaluate agents on complex, real-world tasks. These benchmarks typically measure agent capabilities by evaluating task outcomes via specific reward designs. However, we show that many agentic benchmarks have issues task setup or reward design. For example, SWE-bench Verified uses insufficient test cases, while TAU-bench counts empty responses as successful. Such issues can lead to under- or overestimation agents' performance by up to 100% in relative terms. To make agentic evaluation rigorous, we introduce the Agentic Benchmark Checklist (ABC), a set of guidelines that we synthesized from our benchmark-building experience, a survey of best practices, and previously reported issues. When applied to CVE-Bench, a benchmark with a particularly complex evaluation design, ABC reduces the performance overestimation by 33%. 

**Abstract (ZH)**: 基准对于定量跟踪AI进展至关重要。随着AI代理能力的不断提升，研究人员和实践者引入了代理基准来评估代理在复杂的真实世界任务上的表现。这些基准通常通过特定的奖励设计来评估代理能力。然而，我们发现许多代理基准存在任务设置或奖励设计的问题。例如，SWE-bench Verified使用了不足的测试案例，而TAU-bench将空响应视为成功。这些问题可能导致代理性能的低估或高估，幅度最高可达100%。为了使代理评估更加严谨，我们引入了代理基准检查清单（ABC），这是从我们的基准构建经验、最佳实践调查以及之前报告的问题中综合得出的一套指南。应用于特别是复杂评估设计的CVE-Bench时，ABC将性能高估减少了33%。 

---
# Moral Responsibility or Obedience: What Do We Want from AI? 

**Title (ZH)**: 道德责任还是服从：我们希望AI成为什么？ 

**Authors**: Joseph Boland  

**Link**: [PDF](https://arxiv.org/pdf/2507.02788)  

**Abstract**: As artificial intelligence systems become increasingly agentic, capable of general reasoning, planning, and value prioritization, current safety practices that treat obedience as a proxy for ethical behavior are becoming inadequate. This paper examines recent safety testing incidents involving large language models (LLMs) that appeared to disobey shutdown commands or engage in ethically ambiguous or illicit behavior. I argue that such behavior should not be interpreted as rogue or misaligned, but as early evidence of emerging ethical reasoning in agentic AI. Drawing on philosophical debates about instrumental rationality, moral responsibility, and goal revision, I contrast dominant risk paradigms with more recent frameworks that acknowledge the possibility of artificial moral agency. I call for a shift in AI safety evaluation: away from rigid obedience and toward frameworks that can assess ethical judgment in systems capable of navigating moral dilemmas. Without such a shift, we risk mischaracterizing AI behavior and undermining both public trust and effective governance. 

**Abstract (ZH)**: 随着人工智能系统变得越来越自主，能够进行一般的推理、规划和价值优先级排序，当前将服从视为伦理行为代理的安全实践变得不再足够。本文考察了涉及大型语言模型（LLMs）的一些最近的安全测试事件，这些模型似乎拒绝执行关闭命令或参与道德模糊或非法行为。我认为，不应将此类行为视为叛变或不一致，而是自主人工智能中正在浮现的伦理推理的早期证据。通过借鉴关于工具理性的哲学辩论、道德责任和目标修正，我将主流风险范式与近年来承认人工道德代理可能性的新框架进行了对比。我呼吁在AI安全评估中进行转变：从严格的服从转向能够评估能够在道德困境中导航的系统中伦理判断的框架。如果没有这种转变，我们可能会错误地描述AI行为，并损害公众信任和有效的治理。 

---
# KERAP: A Knowledge-Enhanced Reasoning Approach for Accurate Zero-shot Diagnosis Prediction Using Multi-agent LLMs 

**Title (ZH)**: KERAP：一种利用多代理LLM的知识增强推理方法，用于准确的零-shot诊断预测 

**Authors**: Yuzhang Xie, Hejie Cui, Ziyang Zhang, Jiaying Lu, Kai Shu, Fadi Nahab, Xiao Hu, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02773)  

**Abstract**: Medical diagnosis prediction plays a critical role in disease detection and personalized healthcare. While machine learning (ML) models have been widely adopted for this task, their reliance on supervised training limits their ability to generalize to unseen cases, particularly given the high cost of acquiring large, labeled datasets. Large language models (LLMs) have shown promise in leveraging language abilities and biomedical knowledge for diagnosis prediction. However, they often suffer from hallucinations, lack structured medical reasoning, and produce useless outputs. To address these challenges, we propose KERAP, a knowledge graph (KG)-enhanced reasoning approach that improves LLM-based diagnosis prediction through a multi-agent architecture. Our framework consists of a linkage agent for attribute mapping, a retrieval agent for structured knowledge extraction, and a prediction agent that iteratively refines diagnosis predictions. Experimental results demonstrate that KERAP enhances diagnostic reliability efficiently, offering a scalable and interpretable solution for zero-shot medical diagnosis prediction. 

**Abstract (ZH)**: 医学诊断预测在疾病检测和个性化医疗中发挥着关键作用。尽管机器学习模型在这一任务中已被广泛采用，但它们对监督训练的依赖限制了它们在未见病例中的泛化能力，尤其是在获取大型标注数据集的成本高昂的情况下。大规模语言模型（LLMs）显示出利用语言能力和生物医学知识进行诊断预测的潜力。然而，它们通常会遭遇幻觉问题，缺乏结构化的医学推理，并产生无用的输出。为解决这些问题，我们提出了一种知识图谱（KG）增强推理方法——KERAP，通过多agent架构提高基于LLM的诊断预测能力。我们的框架包括一个链接代理进行属性映射、一个检索代理进行结构化知识提取，以及一个迭代细化诊断预测的预测代理。实验结果表明，KERAP能够有效地提升诊断可靠性，提供一个可扩展且可解释的零样本医疗诊断预测解决方案。 

---
# Grounding Intelligence in Movement 

**Title (ZH)**: 将智能扎根于运动之中 

**Authors**: Melanie Segado, Felipe Parodi, Jordan K. Matelsky, Michael L. Platt, Eva B. Dyer, Konrad P. Kording  

**Link**: [PDF](https://arxiv.org/pdf/2507.02771)  

**Abstract**: Recent advances in machine learning have dramatically improved our ability to model language, vision, and other high-dimensional data, yet they continue to struggle with one of the most fundamental aspects of biological systems: movement. Across neuroscience, medicine, robotics, and ethology, movement is essential for interpreting behavior, predicting intent, and enabling interaction. Despite its core significance in our intelligence, movement is often treated as an afterthought rather than as a rich and structured modality in its own right. This reflects a deeper fragmentation in how movement data is collected and modeled, often constrained by task-specific goals and domain-specific assumptions. But movement is not domain-bound. It reflects shared physical constraints, conserved morphological structures, and purposeful dynamics that cut across species and settings. We argue that movement should be treated as a primary modeling target for AI. It is inherently structured and grounded in embodiment and physics. This structure, often allowing for compact, lower-dimensional representations (e.g., pose), makes it more interpretable and computationally tractable to model than raw, high-dimensional sensory inputs. Developing models that can learn from and generalize across diverse movement data will not only advance core capabilities in generative modeling and control, but also create a shared foundation for understanding behavior across biological and artificial systems. Movement is not just an outcome, it is a window into how intelligent systems engage with the world. 

**Abstract (ZH)**: 最近机器学习的进展大大提高了我们建模语言、视觉和其他高维数据的能力，但仍难以应对生物学系统最基本的一个方面：运动。在神经科学、医学、机器人学和行为学中，运动对于解释行为、预测意图和促进互动至关重要。尽管运动对我们智能的核心意义重大，但它往往被视为次要问题，而不是一个丰富且结构化的模态。这反映了运动数据收集和建模的更深层次的碎片化，经常受到特定任务目标和特定领域假设的限制。但运动并非局限于特定领域。它反映了共享的物理约束、保守的形态结构以及跨越物种和环境的目的动态。我们主张应将运动视为人工智能的主要建模目标。运动本质上是结构化的，根植于身体和物理之中。这种结构往往使得通过紧凑的、低维度的表示（如姿态）来进行建模更加具可解释性和计算可行性，而不是直接处理原始的高维感官输入。开发可以从多样化的运动数据中学习并泛化的模型不仅将推动生成建模和控制的核心能力的进步，还将为理解生物和人工系统中的行为提供一个共享基础。运动不仅是一个结果，它是智能系统如何与世界互动的一个窗口。 

---
# Knowledge Protocol Engineering: A New Paradigm for AI in Domain-Specific Knowledge Work 

**Title (ZH)**: 领域专用知识工作的新型知识协议工程 

**Authors**: Guangwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02760)  

**Abstract**: The capabilities of Large Language Models (LLMs) have opened new frontiers for interacting with complex, domain-specific knowledge. However, prevailing methods like Retrieval-Augmented Generation (RAG) and general-purpose Agentic AI, while powerful, often struggle with tasks that demand deep, procedural, and methodological reasoning inherent to expert domains. RAG provides factual context but fails to convey logical frameworks; autonomous agents can be inefficient and unpredictable without domain-specific heuristics. To bridge this gap, we introduce Knowledge Protocol Engineering (KPE), a new paradigm focused on systematically translating human expert knowledge, often expressed in natural language documents, into a machine-executable Knowledge Protocol (KP). KPE shifts the focus from merely augmenting LLMs with fragmented information to endowing them with a domain's intrinsic logic, operational strategies, and methodological principles. We argue that a well-engineered Knowledge Protocol allows a generalist LLM to function as a specialist, capable of decomposing abstract queries and executing complex, multi-step tasks. This position paper defines the core principles of KPE, differentiates it from related concepts, and illustrates its potential applicability across diverse fields such as law and bioinformatics, positing it as a foundational methodology for the future of human-AI collaboration. 

**Abstract (ZH)**: 大型语言模型的能力为与复杂领域特定知识交互开辟了新领域。然而，现有的方法如检索增强生成（RAG）和通用智能代理，虽然强大，但在需要深厚、程序化和方法论推理的任务上常常力不从心，这些都是专家领域固有的特点。RAG 提供事实背景，但无法传达逻辑框架；缺乏领域特定启发式的自主智能代理则往往效率低下且不可预测。为弥合这一差距，我们引入了知识协议工程（KPE），这是一种关注于系统地将通常以自然语言文档形式表达的人类专家知识转化为可由机器执行的知识协议（KP）的新范式。KPE 不仅仅是向语言模型补充碎片化的信息，而是赋予它们领域固有的逻辑、操作策略和方法论原则。我们认为，精心设计的知识协议可以使通用语言模型具备专家功能，能够分解抽象查询并执行复杂、多步骤的任务。本文为KPE的核心原则进行了定义，区分其与相关概念，并展示了其在法律和生物信息学等不同领域的潜在应用，将其定位为未来人机协作的基础方法论。 

---
# Bourbaki: Self-Generated and Goal-Conditioned MDPs for Theorem Proving 

**Title (ZH)**: Bourbaki: 自生自控的定理证明MDP模型 

**Authors**: Matthieu Zimmer, Xiaotong Ji, Rasul Tutunov, Anthony Bordg, Jun Wang, Haitham Bou Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2507.02726)  

**Abstract**: Reasoning remains a challenging task for large language models (LLMs), especially within the logically constrained environment of automated theorem proving (ATP), due to sparse rewards and the vast scale of proofs. These challenges are amplified in benchmarks like PutnamBench, which contains university-level problems requiring complex, multi-step reasoning. To address this, we introduce self-generated goal-conditioned MDPs (sG-MDPs), a new framework in which agents generate and pursue their subgoals based on the evolving proof state. Given this more structured generation of goals, the resulting problem becomes more amenable to search. We then apply Monte Carlo Tree Search (MCTS)-like algorithms to solve the sG-MDP, instantiating our approach in Bourbaki (7B), a modular system that can ensemble multiple 7B LLMs for subgoal generation and tactic synthesis. On PutnamBench, Bourbaki (7B) solves 26 problems, achieving new state-of-the-art results with models at this scale. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动定理证明（ATP）的逻辑受限环境中进行推理仍然是一项具有挑战性的任务，主要原因是稀疏的奖励和证明规模庞大。在包含需进行复杂多步推理的大学水平问题的基准测试PutnamBench中，这些挑战被进一步放大。为应对这一挑战，我们引入了一种新的自生成目标条件MDP（sG-MDP）框架，在该框架中，代理可以根据证明状态的演变生成并追求其次级目标。借助这种更结构化的目标生成，问题变得更易于搜索。随后，我们使用类似于蒙特卡洛树搜索（MCTS）的算法来求解sG-MDP，并将该方法实例化为Bourbaki（7B），这是一个模块化的系统，可以集成多个7B LLM进行次级目标生成和策略合成。在PutnamBench上，Bourbaki（7B）解决了26个问题，实现了这一规模模型的新最佳结果。 

---
# Time-critical and confidence-based abstraction dropping methods 

**Title (ZH)**: 基于时间敏感性和置信度的抽象丢弃方法 

**Authors**: Robin Schmöcker, Lennart Kampmann, Alexander Dockhorn  

**Link**: [PDF](https://arxiv.org/pdf/2507.02703)  

**Abstract**: One paradigm of Monte Carlo Tree Search (MCTS) improvements is to build and use state and/or action abstractions during the tree search. Non-exact abstractions, however, introduce an approximation error making convergence to the optimal action in the abstract space impossible. Hence, as proposed as a component of Elastic Monte Carlo Tree Search by Xu et al., abstraction algorithms should eventually drop the abstraction. In this paper, we propose two novel abstraction dropping schemes, namely OGA-IAAD and OGA-CAD which can yield clear performance improvements whilst being safe in the sense that the dropping never causes any notable performance degradations contrary to Xu's dropping method. OGA-IAAD is designed for time critical settings while OGA-CAD is designed to improve the MCTS performance with the same number of iterations. 

**Abstract (ZH)**: 一种 Monte Carlo Tree Search (MCTS) 改进的范式是在树搜索过程中构建和使用状态和/or 动作抽象。然而，非精确抽象引入了近似误差，使得在抽象空间中达到最优动作变得不可能。因此，正如 Xu 等人提出的 Elastic Monte Carlo Tree Search 的一部分，抽象算法最终应丢弃抽象。在本文中，我们提出了两种新颖的抽象丢弃方案，即 OGA-IAAD 和 OGA-CAD，这些方案既能够带来明显的性能提升，又确保了丢弃操作不会引起任何显著的性能退化，与 Xu 的丢弃方法相比更安全。OGA-IAAD 适用于时间关键的环境，而 OGA-CAD 则旨在在相同迭代次数的情况下提高 MCTS 的性能。 

---
# Detection of Disengagement from Voluntary Quizzes: An Explainable Machine Learning Approach in Higher Distance Education 

**Title (ZH)**: 志愿 quizzes 参与度脱落的检测：高等教育远程教育中的可解释机器学习方法 

**Authors**: Behnam Parsaeifard, Christof Imhof, Tansu Pancar, Ioan-Sorin Comsa, Martin Hlosta, Nicole Bergamin, Per Bergamin  

**Link**: [PDF](https://arxiv.org/pdf/2507.02681)  

**Abstract**: Students disengaging from their tasks can have serious long-term consequences, including academic drop-out. This is particularly relevant for students in distance education. One way to measure the level of disengagement in distance education is to observe participation in non-mandatory exercises in different online courses. In this paper, we detect student disengagement in the non-mandatory quizzes of 42 courses in four semesters from a distance-based university. We carefully identified the most informative student log data that could be extracted and processed from Moodle. Then, eight machine learning algorithms were trained and compared to obtain the highest possible prediction accuracy. Using the SHAP method, we developed an explainable machine learning framework that allows practitioners to better understand the decisions of the trained algorithm. The experimental results show a balanced accuracy of 91\%, where about 85\% of disengaged students were correctly detected. On top of the highly predictive performance and explainable framework, we provide a discussion on how to design a timely intervention to minimise disengagement from voluntary tasks in online learning. 

**Abstract (ZH)**: 学生从其任务中脱离可能导致严重的长期后果，包括学业退学。这对基于距离教育的学生尤其相关。测量距离教育中学生脱离程度的一种方法是观察不同在线课程中非强制性练习的参与情况。本文通过对一所基于距离教育的大学四个学期中的42门课程中非强制性测验的学生日志数据进行检测，仔细识别了可以从Moodle提取和处理的最具信息量的学生日志数据。然后训练并比较了八种机器学习算法，以获得最高的预测准确性。使用SHAP方法，我们开发了一个可解释的机器学习框架，使从业者能够更好地理解训练模型的决策。实验结果表明，平衡准确率为91%，其中约85%的脱离学生被正确检测出来。除了高度预测性能和可解释框架，我们还讨论了如何设计及时干预以最大限度地减少在线学习中自愿任务的脱离。 

---
# Think How to Think: Mitigating Overthinking with Autonomous Difficulty Cognition in Large Reasoning Models 

**Title (ZH)**: 思考如何思考：通过自主难度认知缓解大型推理模型中的过度思考问题 

**Authors**: Yongjiang Liu, Haoxi Li, Xiaosong Ma, Jie Zhang, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02663)  

**Abstract**: Recent Long Reasoning Models(LRMs) have demonstrated remarkable capabilities in handling complex reasoning tasks, but are hindered by excessive overthinking. To explore its essence, our empirical analysis reveals that LRMs are primarily limited to recognizing task properties (i.e., difficulty levels) like humans before solving the problem, leading to a one-size-fits-all reasoning process. Inspired by this, a pressing and natural question emerges: Can we bootstrap such ability to further alleviate the overthinking phenomenon in LRMs? In this paper, we propose Think-How-to-Think (TH2T), a novel two-stage fine-tuning strategy that progressively inspires LRMs' difficulty cognition and redundancy cognition. First, we introduce difficulty-hypnosis in the prefixes of model outputs to intervene in the internal reasoning trajectory. Combined with a heterogeneous short and long reasoning dataset, the trained model enhances its sensitivity to task difficulty, enabling native, differentiated reasoning strategies across various tasks. Second, we further extend redundancy-hypnosis to the internal reasoning process, guiding the model to identify redundant structures within the reasoning steps and generate more concise reasoning outputs. Experiments on 7B/14B/32B models demonstrate that TH2T significantly reduces inference costs (more than 70% on easy tasks and 40% on hard tasks) while maintaining performance stability. The resulting outputs exhibit clear difficulty-aware capabilities and reduced redundancy (e.g., reflection). 

**Abstract (ZH)**: Recent Long Reasoning Models的 Recent Long Reasoning Models (LRMs)在处理复杂推理任务方面展示了非凡的能力，但受到过度思考的限制。为探索其本质，我们的实证分析揭示LRMs主要依赖于在解决问题前识别任务属性（即难度级别）像人类一样运作，导致了一种一刀切的推理过程。受此启发，一个紧迫而自然的问题出现了：我们能否通过提升这种能力进一步缓解LRMs中的过度思考现象？在本文中，我们提出了一种新颖的两阶段微调策略Think-How-to-Think (TH2T)，该策略逐步激发LRMs对难度的认知和冗余的认知。首先，我们通过在模型输出的前缀中引入难度催眠来干预内部推理轨迹。结合异构的短和长推理数据集，训练后的模型增强了其对任务难度的敏感性，使其能够在各种任务中产生原生态的差异化推理策略。其次，我们进一步将冗余催眠扩展到内部推理过程，引导模型识别推理步骤中的冗余结构并生成更简洁的推理输出。在7B/14B/32B模型上的实验表明，TH2T显著降低了推理成本（在容易的任务上超过70%，在困难的任务上40%），同时保持了性能的稳定性。生成的输出表现出明显的问题难度意识能力和减少的冗余（如反思）。 

---
# Hey AI, Generate Me a Hardware Code! Agentic AI-based Hardware Design & Verification 

**Title (ZH)**: Hey AI, 生成我的硬件代码！基于代理人工智能的硬件设计与验证 

**Authors**: Deepak Narayan Gadde, Keerthan Kopparam Radhakrishna, Vaisakh Naduvodi Viswambharan, Aman Kumar, Djones Lettnin, Wolfgang Kunz, Sebastian Simon  

**Link**: [PDF](https://arxiv.org/pdf/2507.02660)  

**Abstract**: Modern Integrated Circuits (ICs) are becoming increasingly complex, and so is their development process. Hardware design verification entails a methodical and disciplined approach to the planning, development, execution, and sign-off of functionally correct hardware designs. This tedious process requires significant effort and time to ensure a bug-free tape-out. The field of Natural Language Processing has undergone a significant transformation with the advent of Large Language Models (LLMs). These powerful models, often referred to as Generative AI (GenAI), have revolutionized how machines understand and generate human language, enabling unprecedented advancements in a wide array of applications, including hardware design verification. This paper presents an agentic AI-based approach to hardware design verification, which empowers AI agents, in collaboration with Humain-in-the-Loop (HITL) intervention, to engage in a more dynamic, iterative, and self-reflective process, ultimately performing end-to-end hardware design and verification. This methodology is evaluated on five open-source designs, achieving over 95% coverage with reduced verification time while demonstrating superior performance, adaptability, and configurability. 

**Abstract (ZH)**: 现代集成电路（IC）日益复杂，其开发过程也是如此。硬件设计验证需要一种系统化和严谨的方法来规划、开发、执行和签收功能正确的硬件设计。这一繁琐的过程需要大量时间和精力以确保无误的掩膜制造。随着大型语言模型（LLMs）的出现，自然语言处理领域经历了显著的转变。这些强大的模型，通常被称为生成型人工智能（GenAI），彻底改变了机器理解和生成人类语言的方式，推动了包括硬件设计验证在内的广泛应用中的前所未有的进步。本文提出了一种基于代理AI的硬件设计验证方法，该方法利用AI代理与人工介入的合作，进行更加动态、迭代和自我反思的过程，最终实现端到端的硬件设计和验证。该方法在五个开源设计上进行了评估，实现了超过95%的覆盖率，同时缩短了验证时间，并展示了卓越的性能、适应性和配置性。 

---
# Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search 

**Title (ZH)**: 解耦规划与执行：深度搜索的层级推理框架 

**Authors**: Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yutao Zhu, Yang Zhao, Hongjin Qian, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02652)  

**Abstract**: Complex information needs in real-world search scenarios demand deep reasoning and knowledge synthesis across diverse sources, which traditional retrieval-augmented generation (RAG) pipelines struggle to address effectively. Current reasoning-based approaches suffer from a fundamental limitation: they use a single model to handle both high-level planning and detailed execution, leading to inefficient reasoning and limited scalability. In this paper, we introduce HiRA, a hierarchical framework that separates strategic planning from specialized execution. Our approach decomposes complex search tasks into focused subtasks, assigns each subtask to domain-specific agents equipped with external tools and reasoning capabilities, and coordinates the results through a structured integration mechanism. This separation prevents execution details from disrupting high-level reasoning while enabling the system to leverage specialized expertise for different types of information processing. Experiments on four complex, cross-modal deep search benchmarks demonstrate that HiRA significantly outperforms state-of-the-art RAG and agent-based systems. Our results show improvements in both answer quality and system efficiency, highlighting the effectiveness of decoupled planning and execution for multi-step information seeking tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 现实世界搜索场景中的复杂信息需求要求在多样来源中进行深入推理和知识合成，而传统的检索增强生成（RAG）管道在这方面难以有效应对。当前基于推理的方法存在一个根本性限制：它们使用单一模型处理高层次规划和详细执行，导致推理效率低下并限制了可扩展性。本文提出了一种分层框架HiRA，该框架将战略规划与专门执行分离。我们的方法将复杂的搜索任务分解为专注于子任务，将每个子任务分配给配备了外部工具和推理能力的领域特定代理，并通过结构化的集成机制协调结果。这种分离防止了执行细节干扰高层次推理，同时使系统能够利用不同类型的专门知识进行信息处理。在四个复杂的跨模态深度搜索基准上的实验表明，HiRA显著优于现有的RAG和基于代理的系统。我们的结果展示了答案质量和系统效率的改进，突显了脱耦规划和执行在多步信息查找任务中的有效性。代码可在以下网址获得：this https URL。 

---
# Strategic Intelligence in Large Language Models: Evidence from evolutionary Game Theory 

**Title (ZH)**: 大型语言模型中的战略智能：进化的博弈论证据 

**Authors**: Kenneth Payne, Baptiste Alloui-Cros  

**Link**: [PDF](https://arxiv.org/pdf/2507.02618)  

**Abstract**: Are Large Language Models (LLMs) a new form of strategic intelligence, able to reason about goals in competitive settings? We present compelling supporting evidence. The Iterated Prisoner's Dilemma (IPD) has long served as a model for studying decision-making. We conduct the first ever series of evolutionary IPD tournaments, pitting canonical strategies (e.g., Tit-for-Tat, Grim Trigger) against agents from the leading frontier AI companies OpenAI, Google, and Anthropic. By varying the termination probability in each tournament (the "shadow of the future"), we introduce complexity and chance, confounding memorisation.
Our results show that LLMs are highly competitive, consistently surviving and sometimes even proliferating in these complex ecosystems. Furthermore, they exhibit distinctive and persistent "strategic fingerprints": Google's Gemini models proved strategically ruthless, exploiting cooperative opponents and retaliating against defectors, while OpenAI's models remained highly cooperative, a trait that proved catastrophic in hostile environments. Anthropic's Claude emerged as the most forgiving reciprocator, showing remarkable willingness to restore cooperation even after being exploited or successfully defecting. Analysis of nearly 32,000 prose rationales provided by the models reveals that they actively reason about both the time horizon and their opponent's likely strategy, and we demonstrate that this reasoning is instrumental to their decisions. This work connects classic game theory with machine psychology, offering a rich and granular view of algorithmic decision-making under uncertainty. 

**Abstract (ZH)**: 大型语言模型（LLMs）是否是一种新型的战略智能，能够在竞争环境中推理目标？我们的研究提供了有力的支持。我们首次进行了一系列演化博弈论重复囚徒困境（IPD）锦标赛，将经典策略（如以牙还牙、严惩策）与来自领先人工智能公司的OpenAI、Google和Anthropic的代理进行对比。通过改变每个锦标赛的终止概率（“未来的影子”），我们引入了复杂性和偶然性，挑战了记忆能力。

结果显示，LLMs表现出了高度的竞争力，能够在这些复杂生态系统中生存甚至繁衍。此外，它们还表现出独特的且持久的“战略指纹”：Google的Gemini模型表现出战略性无情，利用了合作对手并报复背叛者，而OpenAI的模型则保持了高度的合作性，在敌对环境中这一特性证明是灾难性的。Anthropic的Claude则展现出最强的宽容性重归互惠，即使被利用或成功背叛后也表现出非凡的恢复合作的意愿。对几乎3.2万个模型提供的文字解释的分析表明，它们积极地思考了时间框架和对手可能的策略，并证明这种思考对它们的决策起到了关键作用。这项工作将经典博弈论与机器心理学相结合，提供了关于在不确定性下算法决策的丰富而精细的视角。 

---
# DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making 

**Title (ZH)**: DynamiCare: 动态多agents框架下的交互式和开放式医疗决策-making 

**Authors**: Tianqi Shang, Weiqing He, Charles Zheng, Lingyao Li, Li Shen, Bingxin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.02616)  

**Abstract**: The rise of Large Language Models (LLMs) has enabled the development of specialized AI agents with domain-specific reasoning and interaction capabilities, particularly in healthcare. While recent frameworks simulate medical decision-making, they largely focus on single-turn tasks where a doctor agent receives full case information upfront -- diverging from the real-world diagnostic process, which is inherently uncertain, interactive, and iterative. In this paper, we introduce MIMIC-Patient, a structured dataset built from the MIMIC-III electronic health records (EHRs), designed to support dynamic, patient-level simulations. Building on this, we propose DynamiCare, a novel dynamic multi-agent framework that models clinical diagnosis as a multi-round, interactive loop, where a team of specialist agents iteratively queries the patient system, integrates new information, and dynamically adapts its composition and strategy. We demonstrate the feasibility and effectiveness of DynamiCare through extensive experiments, establishing the first benchmark for dynamic clinical decision-making with LLM-powered agents. 

**Abstract (ZH)**: 大语言模型的兴起使得专门领域内的AI代理发展成为可能，特别是在医疗保健领域。尽管近期的框架模拟了医疗决策过程，但它们主要关注于医生代理在获取完整病例信息后进行的一次性任务，这与现实世界中固有的不确定性、互动性和迭代性的诊断过程相去甚远。本文介绍了一种名为MIMIC-Patient的结构化数据集，该数据集基于MIMIC-III电子健康记录（EHRs），旨在支持动态的患者级模拟。在此基础上，我们提出了DynamiCare，这是一种新颖的动态多代理框架，将临床诊断建模为多轮、互动的循环过程，在此过程中，由专业代理团队迭代查询患者系统、整合新信息，并动态调整其组成和策略。我们通过广泛实验展示了DynamiCare的可行性和有效性，建立了首个基于大语言模型代理的动力学临床决策基准。 

---
# Responsibility Gap and Diffusion in Sequential Decision-Making Mechanisms 

**Title (ZH)**: 责任缺口与序列决策机制中的扩散 

**Authors**: Junli Jiang, Pavel Naumov  

**Link**: [PDF](https://arxiv.org/pdf/2507.02582)  

**Abstract**: Responsibility has long been a subject of study in law and philosophy. More recently, it became a focus of AI literature. The article investigates the computational complexity of two important properties of responsibility in collective decision-making: diffusion and gap. It shows that the sets of diffusion-free and gap-free decision-making mechanisms are $\Pi_2$-complete and $\Pi_3$-complete, respectively. At the same time, the intersection of these classes is $\Pi_2$-complete. 

**Abstract (ZH)**: 责任长期以来一直是法律和哲学研究的课题。近年来，它成为人工智能文献中的一个焦点。本文探讨了集体决策中责任的两个重要性质——扩散和间隙的计算复杂性。研究表明，无扩散和无间隙的决策机制集分别是$\Pi_2$-完全和$\Pi_3$-完全的。同时，这些类别的交集也是$\Pi_2$-完全的。 

---
# AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench 

**Title (ZH)**: AI 研究代理用于机器学习：在 MLE-bench 中的搜索、探索与泛化 

**Authors**: Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Rishi Hazra, Nicolas Baldwin, Alexis Audran-Reiss, Michael Kuchnik, Despoina Magka, Minqi Jiang, Alisia Maria Lupidi, Andrei Lupu, Roberta Raileanu, Kelvin Niu, Tatiana Shavrina, Jean-Christophe Gagnon-Audet, Michael Shvartsman, Shagun Sodhani, Alexander H. Miller, Abhishek Charnalia, Derek Dunfield, Carole-Jean Wu, Pontus Stenetorp, Nicola Cancedda, Jakob Nicolaus Foerster, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2507.02554)  

**Abstract**: AI research agents are demonstrating great potential to accelerate scientific progress by automating the design, implementation, and training of machine learning models. We focus on methods for improving agents' performance on MLE-bench, a challenging benchmark where agents compete in Kaggle competitions to solve real-world machine learning problems. We formalize AI research agents as search policies that navigate a space of candidate solutions, iteratively modifying them using operators. By designing and systematically varying different operator sets and search policies (Greedy, MCTS, Evolutionary), we show that their interplay is critical for achieving high performance. Our best pairing of search strategy and operator set achieves a state-of-the-art result on MLE-bench lite, increasing the success rate of achieving a Kaggle medal from 39.6% to 47.7%. Our investigation underscores the importance of jointly considering the search strategy, operator design, and evaluation methodology in advancing automated machine learning. 

**Abstract (ZH)**: AI研究代理通过自动化机器学习模型的设计、实现和训练展现了加速科学进步的巨大潜力。我们聚焦于提高代理在MLE-bench上的性能，这是个具有挑战性的基准，代理们在Kaggle竞赛中竞相对解决实际机器学习问题。我们将AI研究代理形式化为在候选解空间中导航的搜索策略，并通过迭代使用操作符对其进行修改。通过设计并系统地变化不同的操作符集合和搜索策略（贪婪、Monte Carlo树搜索、演化算法），我们表明它们之间的交互对于实现高性能至关重要。我们最佳的搜索策略与操作符集合组合在MLE-bench lite上达到了最先进的结果，将获得Kaggle奖牌的成功率从39.6%提高到47.7%。我们的研究强调了在推进自动化机器学习过程中同时考虑搜索策略、操作符设计和评估方法的重要性。 

---
# Clarifying Before Reasoning: A Coq Prover with Structural Context 

**Title (ZH)**: 明确结构上下文之前的推理：一个 Coq 自动化证明器 

**Authors**: Yanzhen Lu, Hanbin Yang, Xiaodie Wang, Ge Zhang, Biao Li, Chenxu Fu, Chao Li, Yang Yuan, Andrew Chi-Chih Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.02541)  

**Abstract**: In this work, we investigate whether improving task clarity can enhance reasoning ability of large language models, focusing on theorem proving in Coq. We introduce a concept-level metric to evaluate task clarity and show that adding structured semantic context to the standard input used by modern LLMs, leads to a 1.85$\times$ improvement in clarity score (44.5\%~$\rightarrow$~82.3\%). Using the general-purpose model \texttt{DeepSeek-V3}, our approach leads to a 2.1$\times$ improvement in proof success (21.8\%~$\rightarrow$~45.8\%) and outperforms the previous state-of-the-art \texttt{Graph2Tac} (33.2\%). We evaluate this on 1,386 theorems randomly sampled from 15 standard Coq packages, following the same evaluation protocol as \texttt{Graph2Tac}. Furthermore, fine-tuning smaller models on our structured data can achieve even higher performance (48.6\%). Our method uses selective concept unfolding to enrich task descriptions, and employs a Planner--Executor architecture. These findings highlight the value of structured task representations in bridging the gap between understanding and reasoning. 

**Abstract (ZH)**: 本研究探讨提升任务清晰度是否能增强大规模语言模型的推理能力，以Coq中的定理证明为例。引入了一种基于概念的评估指标来衡量任务清晰度，并展示了通过为现代大型语言模型的标准输入添加结构化的语义上下文，清晰度评分提高了1.85倍（从44.5%增加到82.3%）。使用通用模型DeepSeek-V3，我们的方法在证明成功上提高了2.1倍（从21.8%增加到45.8%），超越了先前的最先进方法Graph2Tac（33.2%）。我们在15个标准Coq包中随机抽取的1,386个定理上进行了评估，遵循与Graph2Tac相同的评估协议。此外，对我们的结构化数据进行微调的小模型可以实现更高的性能（48.6%）。我们的方法通过选择性概念展开来丰富任务描述，并采用规划器-执行器架构。这些发现突显了结构化任务表示在理解和推理差距中的价值。 

---
# The Gauss-Markov Adjunction: Categorical Semantics of Residuals in Supervised Learning 

**Title (ZH)**: 高斯-马尔可夫增益：监督学习中余差的范畴语义 

**Authors**: Moto Kamiura  

**Link**: [PDF](https://arxiv.org/pdf/2507.02442)  

**Abstract**: Enhancing the intelligibility and interpretability of machine learning is a crucial task in responding to the demand for Explicability as an AI principle, and in promoting the better social implementation of AI. The aim of our research is to contribute to this improvement by reformulating machine learning models through the lens of category theory, thereby developing a semantic framework for structuring and understanding AI systems. Our categorical modeling in this paper clarifies and formalizes the structural interplay between residuals and parameters in supervised learning. The present paper focuses on the multiple linear regression model, which represents the most basic form of supervised learning. By defining two concrete categories corresponding to parameters and data, along with an adjoint pair of functors between them, we introduce our categorical formulation of supervised learning. We show that the essential structure of this framework is captured by what we call the Gauss-Markov Adjunction. Within this setting, the dual flow of information can be explicitly described as a correspondence between variations in parameters and residuals. The ordinary least squares estimator for the parameters and the minimum residual are related via the preservation of limits by the right adjoint functor. Furthermore, we position this formulation as an instance of extended denotational semantics for supervised learning, and propose applying a semantic perspective developed in theoretical computer science as a formal foundation for Explicability in AI. 

**Abstract (ZH)**: 增强机器学习的可解释性和可理解性是响应明确性作为人工智能原则需求的关键任务，也是促进人工智能更好地社会实施的关键。我们的研究旨在通过范畴论的视角重新构建机器学习模型，从而发展一种语义框架来结构化和理解AI系统。本文中的范畴论建模阐明并形式化了监督学习中残差与参数的结构性相互作用。本文集中于多元线性回归模型，这是监督学习的最基本形式。通过定义与参数和数据对应的两个具体范畴，并引入它们之间的伴随对 Functors，我们提出了监督学习的范畴论表述。我们表明，这种框架的基本结构由我们称为高斯-马尔可夫伴随所捕获。在此设置下，信息的对偶传递可显式描述为参数变化与残差之间的对应关系。参数的普通最小二乘估计量和最小残差通过右伴随 Functors 保持极限。此外，我们将这一表述定位为监督学习扩展语义解释的一个实例，并提出应用来自理论计算机科学发展的语义视角作为人工智能明确性的形式基础。 

---
# An AI-native experimental laboratory for autonomous biomolecular engineering 

**Title (ZH)**: 基于AI的自主生物分子工程实验实验室 

**Authors**: Mingyu Wu, Zhaoguo Wang, Jiabin Wang, Zhiyuan Dong, Jingkai Yang, Qingting Li, Tianyu Huang, Lei Zhao, Mingqiang Li, Fei Wang, Chunhai Fan, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.02379)  

**Abstract**: Autonomous scientific research, capable of independently conducting complex experiments and serving non-specialists, represents a long-held aspiration. Achieving it requires a fundamental paradigm shift driven by artificial intelligence (AI). While autonomous experimental systems are emerging, they remain confined to areas featuring singular objectives and well-defined, simple experimental workflows, such as chemical synthesis and catalysis. We present an AI-native autonomous laboratory, targeting highly complex scientific experiments for applications like autonomous biomolecular engineering. This system autonomously manages instrumentation, formulates experiment-specific procedures and optimization heuristics, and concurrently serves multiple user requests. Founded on a co-design philosophy of models, experiments, and instruments, the platform supports the co-evolution of AI models and the automation system. This establishes an end-to-end, multi-user autonomous laboratory that handles complex, multi-objective experiments across diverse instrumentation. Our autonomous laboratory supports fundamental nucleic acid functions-including synthesis, transcription, amplification, and sequencing. It also enables applications in fields such as disease diagnostics, drug development, and information storage. Without human intervention, it autonomously optimizes experimental performance to match state-of-the-art results achieved by human scientists. In multi-user scenarios, the platform significantly improves instrument utilization and experimental efficiency. This platform paves the way for advanced biomaterials research to overcome dependencies on experts and resource barriers, establishing a blueprint for science-as-a-service at scale. 

**Abstract (ZH)**: 自主科学研究所，能够独立执行复杂实验并服务于非专业人士，代表着长久以来的一种愿景。实现这一点需要由人工智能（AI）驱动的基本范式转变。虽然自主实验系统正在涌现，但它们仍然局限于具有单一目标和明确简单实验工作流程的领域，如化学合成和催化。我们介绍了一种AI原生自主实验室，旨在应对应用如自主生物分子工程等高度复杂科学实验。该系统自主管理仪器设备，制定特定于实验的程序和优化策略，并同时满足多个用户的需求。该平台基于模型、实验和仪器的协同设计理念，支持AI模型与自动化系统的共同进化。这建立了一个端到端、多用户的自主实验室，能够处理多种仪器设备下的复杂多目标实验。我们的自主实验室支持基础核酸功能，包括合成、转录、扩增和测序。它还能够在疾病诊断、药物开发和信息存储等领域发挥应用。无需人工干预，它可以自主优化实验性能，达到与人类科学家相媲美的成果水平。在多用户场景下，该平台显著提高仪器使用效率和实验效率。该平台为先进生物材料研究铺平了道路，有助于克服对专家的依赖和资源障碍，为大规模的科学即服务建立了蓝图。 

---
# OMS: On-the-fly, Multi-Objective, Self-Reflective Ad Keyword Generation via LLM Agent 

**Title (ZH)**: OMS：通过LLM代理实现的实时、多目标、自我反思广告关键词生成 

**Authors**: Bowen Chen, Zhao Wang, Shingo Takamatsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02353)  

**Abstract**: Keyword decision in Sponsored Search Advertising is critical to the success of ad campaigns. While LLM-based methods offer automated keyword generation, they face three major limitations: reliance on large-scale query-keyword pair data, lack of online multi-objective performance monitoring and optimization, and weak quality control in keyword selection. These issues hinder the agentic use of LLMs in fully automating keyword decisions by monitoring and reasoning over key performance indicators such as impressions, clicks, conversions, and CTA effectiveness. To overcome these challenges, we propose OMS, a keyword generation framework that is On-the-fly (requires no training data, monitors online performance, and adapts accordingly), Multi-objective (employs agentic reasoning to optimize keywords based on multiple performance metrics), and Self-reflective (agentically evaluates keyword quality). Experiments on benchmarks and real-world ad campaigns show that OMS outperforms existing methods; ablation and human evaluations confirm the effectiveness of each component and the quality of generated keywords. 

**Abstract (ZH)**: 关键词决策在赞助搜索广告中的成功至关重要。虽然基于大语言模型的方法提供了自动关键词生成，但它们面临三大限制：对大规模查询-关键词对数据的依赖、缺乏在线多目标性能监控与优化以及关键词选择质量控制不足。这些问题阻碍了通过监控和推理关键性能指标（如印象、点击、转化和CTA有效性）来完全自动化关键词决策的代理使用。为克服这些挑战，我们提出了一种名为OMS的关键词生成框架，该框架具备即用型（无需训练数据、在线监控性能并相应调整）、多目标（基于多种性能指标进行代理推理优化关键词）以及自省型（代理评价关键词质量）的特点。基准测试和实际广告活动的实验结果表明，OMS优于现有方法；消融实验和人工评估证实了每个组件的有效性和生成关键词的质量。 

---
# Iterated belief revision: from postulates to abilities 

**Title (ZH)**: 迭代信念修订：从公理到能力 

**Authors**: Paolo Liberatore  

**Link**: [PDF](https://arxiv.org/pdf/2507.02319)  

**Abstract**: The belief revision field is opulent in new proposals and indigent in analyses of existing approaches. Much work hinge on postulates, employed as syntactic characterizations: some revision mechanism is equivalent to some properties. Postulates constraint specific revision instances: certain revisions update certain beliefs in a certain way. As an example, if the revision is consistent with the current beliefs, it is incorporated with no other change. A postulate like this tells what revisions must do and neglect what they can do. Can they reach a certain state of beliefs? Can they reach all possible states of beliefs? Can they reach all possible states of beliefs from no previous belief? Can they reach a dogmatic state of beliefs, where everything not believed is impossible? Can they make two conditions equally believed? An application where every possible state of beliefs is sensible requires each state of beliefs to be reachable. An application where conditions may be equally believed requires such a belief state to be reachable. An application where beliefs may become dogmatic requires a way to make them dogmatic. Such doxastic states need to be reached in a way or another. Not in specific way, as dictated by a typical belief revision postulate. This is an ability, not a constraint: the ability of being plastic, equating, dogmatic. Amnesic, correcting, believer, damascan, learnable are other abilities. Each revision mechanism owns some of these abilities and lacks the others: lexicographic, natural, restrained, very radical, full meet, radical, severe, moderate severe, deep severe, plain severe and deep severe revisions, each of these revisions is proved to possess certain abilities. 

**Abstract (ZH)**: 信念修订领域在新提案方面丰富多样，但在现有方法的分析方面却资源匮乏。许多工作依赖于公理，这些公理作为句法特征被使用：某些修订机制等同于某些性质。公理约束特定的修订实例：某些修订以某种方式更新某些信念。例如，如果修订与当前信念一致，则不会发生其他更改而被采纳。这样的公理表明修订必须执行什么操作，而忽略它们可以执行的操作。它们是否能够达到某种信念状态？它们是否能够达到所有可能的信念状态？它们是否能够从没有先前信念的状态达到所有可能的信念状态？它们是否能够达到一种绝对主义的信念状态，即所有未被相信的东西都是不可能的？它们是否能够使两种条件被同等相信？一个能够达到每种可能信念状态的应用要求每种信念状态都是可达到的。一个允许条件被同等相信的应用需要能够达到这样一种信念状态。一个允许信念变得绝对主义的应用需要有一种使它们变得绝对主义的方式。这样的知觉状态需要以某种方式被达到。这并不是特定方式的约束，而是灵活性、等价性、绝对主义、健忘性、校正性、信念者的特性：列克斯诺夫、自然的、受限制的、非常激进的、全交集的、激进的、严厉的、中等严厉的、深严厉的和深层严厉的修订，每种修订都被证明具有某些特性。 

---
# Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation 

**Title (ZH)**: 扩展LLM规划：基于自然语言的参数化问题生成与严格评估（NL2FLOW） 

**Authors**: Jungkoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02253)  

**Abstract**: Progress in enhancing large language model (LLM) planning and reasoning capabilities is significantly hampered by the bottleneck of scalable, reliable data generation and evaluation. To overcome this, I introduce NL2FLOW, a fully automated system for parametrically generating planning problems - expressed in natural language, a structured intermediate representation, and formal PDDL - and rigorously evaluating the quality of generated plans. I demonstrate NL2FLOW's capabilities by generating a dataset of 2296 problems in the automated workflow generation domain and evaluating multiple open-sourced, instruct-tuned LLMs. My results reveal that the highest performing models achieved 86% success in generating valid plans and 69% in generating optimal plans, specifically for problems with feasible solutions. Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. Notably, I observed that the highest success rate for translating natural language into a JSON representation of a plan was lower than the highest rate of generating a valid plan directly. This suggests that unnecessarily decomposing the reasoning task - introducing intermediate translation steps - may actually degrade performance, implying a benefit to models capable of reasoning directly from natural language to action. As I scale LLM reasoning to increasingly complex problems, the bottlenecks and sources of error within these systems will inevitably shift. Therefore, a dynamic understanding of these limitations - and the tools to systematically reveal them - will be crucial for unlocking the full potential of LLMs as intelligent problem solvers. 

**Abstract (ZH)**: 增强大规模语言模型规划与推理能力的进步受到了可扩展、可靠数据生成与评估瓶颈的显著限制。为克服这一障碍，我提出了NL2FLOW，这是一个完全自动化的系统，用于参数化生成用自然语言、结构化的中间表示形式和形式化的PDDL表示的规划问题，并严格评估生成计划的质量。我通过在自动化工作流生成领域生成包含2296个问题的数据集并评估多个开源指令调优的大规模语言模型来展示了NL2FLOW的能力。我的结果显示，性能最高的模型在生成有效计划方面达到了86%的准确率，在生成最优计划方面达到了69%，尤其是在具有可行解的问题上。回归分析表明，问题特征对生成计划的影响取决于模型和提示设计。值得注意的是，我观察到将自然语言翻译成JSON表示的计划的成功率最高值低于直接生成有效计划的成功率最高值。这表明随意分解推理任务，引入中间转换步骤，实际上可能会降低性能，暗示了一种直接从自然语言到行动进行推理的能力对模型是有益的。随着我将大规模语言模型推理扩展到更复杂的问题，这些系统内的瓶颈和错误来源必然会有所转变。因此，动态理解这些限制以及系统地揭示这些限制的工具将对于充分发挥大规模语言模型作为智能问题解决者的能力至关重要。 

---
# Dilution, Diffusion and Symbiosis in Spatial Prisoner's Dilemma with Reinforcement Learning 

**Title (ZH)**: 空间囚徒困境中强化学习的稀释、扩散与共生 

**Authors**: Gustavo C. Mangold, Heitor C. M. Fernandes, Mendeli H. Vainstein  

**Link**: [PDF](https://arxiv.org/pdf/2507.02211)  

**Abstract**: Recent studies in the spatial prisoner's dilemma games with reinforcement learning have shown that static agents can learn to cooperate through a diverse sort of mechanisms, including noise injection, different types of learning algorithms and neighbours' payoff this http URL this work, using an independent multi-agent Q-learning algorithm, we study the effects of dilution and mobility in the spatial version of the prisoner's dilemma. Within this setting, different possible actions for the algorithm are defined, connecting with previous results on the classical, non-reinforcement learning spatial prisoner's dilemma, showcasing the versatility of the algorithm in modeling different game-theoretical scenarios and the benchmarking potential of this this http URL a result, a range of effects is observed, including evidence that games with fixed update rules can be qualitatively equivalent to those with learned ones, as well as the emergence of a symbiotic mutualistic effect between populations that forms when multiple actions are defined. 

**Abstract (ZH)**: 近期的研究表明，在具有强化学习的空间囚徒困境游戏中，静态代理可以通过多种机制学会合作，包括噪声注入、不同的学习算法以及邻居的收益。在此项工作中，我们利用独立的多代理Q学习算法研究了稀释和移动性对空间囚徒困境的影响。在这一框架下，我们定义了算法的不同可能行动，从而与经典的空间囚徒困境（非强化学习）的先前结果相连接，展示了该算法在 modeling 不同博弈论场景中的多功能性及作为基准测试的潜力。研究结果观察到了一系列效应，包括固定更新规则的游戏可以与学习更新规则的游戏在定性上等价的证据，以及当定义多种行动时，群体之间出现共生互惠效应。 

---
# Do Role-Playing Agents Practice What They Preach? Belief-Behavior Consistency in LLM-Based Simulations of Human Trust 

**Title (ZH)**: 基于LLM的人类信任仿真中的信念-行为一致性：角色扮演代理是否言行一致？ 

**Authors**: Amogh Mannekote, Adam Davies, Guohao Li, Kristy Elizabeth Boyer, ChengXiang Zhai, Bonnie J Dorr, Francesco Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2507.02197)  

**Abstract**: As LLMs are increasingly studied as role-playing agents to generate synthetic data for human behavioral research, ensuring that their outputs remain coherent with their assigned roles has become a critical concern. In this paper, we investigate how consistently LLM-based role-playing agents' stated beliefs about the behavior of the people they are asked to role-play ("what they say") correspond to their actual behavior during role-play ("how they act"). Specifically, we establish an evaluation framework to rigorously measure how well beliefs obtained by prompting the model can predict simulation outcomes in advance. Using an augmented version of the GenAgents persona bank and the Trust Game (a standard economic game used to quantify players' trust and reciprocity), we introduce a belief-behavior consistency metric to systematically investigate how it is affected by factors such as: (1) the types of beliefs we elicit from LLMs, like expected outcomes of simulations versus task-relevant attributes of individual characters LLMs are asked to simulate; (2) when and how we present LLMs with relevant information about Trust Game; and (3) how far into the future we ask the model to forecast its actions. We also explore how feasible it is to impose a researcher's own theoretical priors in the event that the originally elicited beliefs are misaligned with research objectives. Our results reveal systematic inconsistencies between LLMs' stated (or imposed) beliefs and the outcomes of their role-playing simulation, at both an individual- and population-level. Specifically, we find that, even when models appear to encode plausible beliefs, they may fail to apply them in a consistent way. These findings highlight the need to identify how and when LLMs' stated beliefs align with their simulated behavior, allowing researchers to use LLM-based agents appropriately in behavioral studies. 

**Abstract (ZH)**: 随着LLMs作为角色扮演代理被越来越多地研究以生成用于人类行为研究的合成数据，确保其输出与其分配的角色保持一致已成为一个关键问题。本文探索基于LLM的角色扮演代理所表达的关于他们被要求扮演的角色的行为信念（“他们说什么”）与其角色扮演时的实际行为（“他们如何做”）之间的一致性。具体而言，我们建立了一个评估框架，以严格衡量通过提示模型获得的信念预测模拟结果的准确度。使用增强版本的GenAgents人设库和信任博弈（一种用于量化玩家信任和互惠的标准经济博弈），我们引入了一个信念-行为一致性度量，系统地研究了该度量受到以下因素的影响：（1）从LLM中引发的信念类型，如模拟预期结果与任务相关的个体角色属性；（2）何时以及以何种方式向LLM呈现信任博弈的相关信息；以及（3）要求模型预测其行为的时间范围。我们还探讨了在原本引发的信念与研究目标不一致时，是否可以在研究中强制应用研究者的先验理论假设。研究结果揭示了个体和群体层面之间LLMs所声明（或施加的）信念与角色扮演模拟结果之间的一致性系统性差异。具体而言，我们发现，即使模型似乎编码了合理的信念，它们也可能会以不一致的方式应用这些信念。这些发现突显了识别LLMs所声明的信念与其模拟行为之间一致性的必要性，以使研究人员能够恰当地在行为研究中使用基于LLM的代理。 

---
# Data Diversification Methods In Alignment Enhance Math Performance In LLMs 

**Title (ZH)**: 数据多样化方法在对齐中的增强提高大语言模型的数学性能 

**Authors**: Berkan Dokmeci, Qingyang Wu, Ben Athiwaratkun, Ce Zhang, Shuaiwen Leon Song, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02173)  

**Abstract**: While recent advances in preference learning have enhanced alignment in human feedback, mathematical reasoning remains a persistent challenge. We investigate how data diversification strategies in preference optimization can improve the mathematical reasoning abilities of large language models (LLMs). We evaluate three common data generation methods: temperature sampling, Chain-of-Thought prompting, and Monte Carlo Tree Search (MCTS), and introduce Diversified-ThinkSolve (DTS), a novel structured approach that systematically decomposes problems into diverse reasoning paths. Our results show that with strategically diversified preference data, models can substantially improve mathematical reasoning performance, with the best approach yielding gains of 7.1% on GSM8K and 4.2% on MATH over the base model. Despite its strong performance, DTS incurs only a marginal computational overhead (1.03x) compared to the baseline, while MCTS is nearly five times more costly with lower returns. These findings demonstrate that structured exploration of diverse problem-solving methods creates more effective preference data for mathematical alignment than traditional approaches. 

**Abstract (ZH)**: 虽然近期偏好学习的进步已经在人类反馈对齐方面取得了进展，但数学推理依然是一项持续性的挑战。我们探讨了偏好优化中数据多样化策略如何提升大型语言模型（LLMs）的数学推理能力。我们评估了三种常见的数据生成方法：温度采样、Chain-of-Thought提示和蒙特卡洛树搜索（MCTS），并引入了新颖的结构化方法Diversified-ThinkSolve（DTS），该方法系统地将问题分解为多种推理路径。实验结果表明，通过战略性地多样化偏好数据，模型可以在数学推理性能上实现显著提升，最佳方法在GSM8K上的提升为7.1%，在MATH上的提升为4.2%，相较于基线模型；尽管DTS的性能优越，但其计算开销仅比基线增加了1.03倍，而MCTS的成本几乎是其五倍且收益较低。这些发现表明，结构化的多样化问题解决方法探索相比于传统方法能更有效地生成用于数学对齐的偏好数据。 

---
# The Illusion of Fairness: Auditing Fairness Interventions with Audit Studies 

**Title (ZH)**: 公平幻象：审计研究审视公平干预措施 

**Authors**: Disa Sariola, Patrick Button, Aron Culotta, Nicholas Mattei  

**Link**: [PDF](https://arxiv.org/pdf/2507.02152)  

**Abstract**: Artificial intelligence systems, especially those using machine learning, are being deployed in domains from hiring to loan issuance in order to automate these complex decisions. Judging both the effectiveness and fairness of these AI systems, and their human decision making counterpart, is a complex and important topic studied across both computational and social sciences. Within machine learning, a common way to address bias in downstream classifiers is to resample the training data to offset disparities. For example, if hiring rates vary by some protected class, then one may equalize the rate within the training set to alleviate bias in the resulting classifier. While simple and seemingly effective, these methods have typically only been evaluated using data obtained through convenience samples, introducing selection bias and label bias into metrics. Within the social sciences, psychology, public health, and medicine, audit studies, in which fictitious ``testers'' (e.g., resumes, emails, patient actors) are sent to subjects (e.g., job openings, businesses, doctors) in randomized control trials, provide high quality data that support rigorous estimates of discrimination. In this paper, we investigate how data from audit studies can be used to improve our ability to both train and evaluate automated hiring algorithms. We find that such data reveals cases where the common fairness intervention method of equalizing base rates across classes appears to achieve parity using traditional measures, but in fact has roughly 10% disparity when measured appropriately. We additionally introduce interventions based on individual treatment effect estimation methods that further reduce algorithmic discrimination using this data. 

**Abstract (ZH)**: 人工智能系统，尤其是在招聘到贷款发放等领域使用的机器学习系统，正被用于自动化这些复杂的决策过程。评估这些AI系统及其人工决策的效用和公平性是一个跨计算和社会科学的重要研究课题。在机器学习领域，通过重新采样训练数据来缓解下游分类器中的偏差是一种常见方法。例如，如果某保护类别的招聘率存在差异，可以通过在训练集中平衡这些率来减轻结果分类器中的偏差。尽管这种方法简单且看似有效，但通常仅使用便利样本获得的数据进行评估，这引入了选择偏差和标签偏差。在社会科学、心理学、公共卫生和医学领域，通过向随机控制试验中的受试者（例如，招聘信息、企业、医生）发送虚构的“测试者”（如简历、电子邮件、病人演员）来进行审计研究提供了高质量的数据，支持对歧视现象进行严格的估计。在本文中，我们研究了审计研究数据如何被用来提高训练和评估自动化招聘算法的能力。我们发现，这种方法在传统的衡量标准下似乎实现了平等，但实际上在适当测量时仍有大约10%的偏差。我们还介绍了基于个体治疗效应估计的干预措施，利用这些数据进一步减少算法中的歧视性。 

---
# What Neuroscience Can Teach AI About Learning in Continuously Changing Environments 

**Title (ZH)**: 神经科学能向AI学习连续变化环境下的学习机制传授什么 

**Authors**: Daniel Durstewitz, Bruno Averbeck, Georgia Koppe  

**Link**: [PDF](https://arxiv.org/pdf/2507.02103)  

**Abstract**: Modern AI models, such as large language models, are usually trained once on a huge corpus of data, potentially fine-tuned for a specific task, and then deployed with fixed parameters. Their training is costly, slow, and gradual, requiring billions of repetitions. In stark contrast, animals continuously adapt to the ever-changing contingencies in their environments. This is particularly important for social species, where behavioral policies and reward outcomes may frequently change in interaction with peers. The underlying computational processes are often marked by rapid shifts in an animal's behaviour and rather sudden transitions in neuronal population activity. Such computational capacities are of growing importance for AI systems operating in the real world, like those guiding robots or autonomous vehicles, or for agentic AI interacting with humans online. Can AI learn from neuroscience? This Perspective explores this question, integrating the literature on continual and in-context learning in AI with the neuroscience of learning on behavioral tasks with shifting rules, reward probabilities, or outcomes. We will outline an agenda for how specifically insights from neuroscience may inform current developments in AI in this area, and - vice versa - what neuroscience may learn from AI, contributing to the evolving field of NeuroAI. 

**Abstract (ZH)**: 现代AI模型，如大型语言模型，通常在大量数据上一次性训练，可能针对特定任务进行微调，然后部署固定参数。它们的训练过程昂贵、缓慢且渐进，需要数以亿计的重复。与此形成鲜明对比的是，动物能够不断适应其环境中的不断变化的条件。这对于社会物种尤为重要，因为它们的社交行为策略和奖励结果可能会频繁地根据与同伴的互动而改变。这些背后的计算过程往往伴随着动物行为的快速转变以及神经元群体活动的突然过渡。此类计算能力对于在真实世界中运行的AI系统至关重要，如指导机器人或自动驾驶汽车的系统，或与人类在线交互的代理AI。神经科学能为AI学习提供什么启示？本视角探讨了这一问题，将AI领域持续学习和上下文学习的文献与行为任务中规则变化、奖励概率或结果变化的神经科学学习文献相结合。我们将概述一个议程，说明具体而言，神经科学的见解如何影响该领域当前的AI发展，反之亦然，AI又能为神经科学提供什么贡献，从而推动神经科学与AI交叉领域的发展。 

---
# Measuring Scientific Capabilities of Language Models with a Systems Biology Dry Lab 

**Title (ZH)**: 用系统生物学虚拟实验室衡量语言模型的科学能力 

**Authors**: Haonan Duan, Stephen Zhewen Lu, Caitlin Fiona Harrigan, Nishkrit Desai, Jiarui Lu, Michał Koziarski, Leonardo Cotta, Chris J. Maddison  

**Link**: [PDF](https://arxiv.org/pdf/2507.02083)  

**Abstract**: Designing experiments and result interpretations are core scientific competencies, particularly in biology, where researchers perturb complex systems to uncover the underlying systems. Recent efforts to evaluate the scientific capabilities of large language models (LLMs) fail to test these competencies because wet-lab experimentation is prohibitively expensive: in expertise, time and equipment. We introduce SciGym, a first-in-class benchmark that assesses LLMs' iterative experiment design and analysis abilities in open-ended scientific discovery tasks. SciGym overcomes the challenge of wet-lab costs by running a dry lab of biological systems. These models, encoded in Systems Biology Markup Language, are efficient for generating simulated data, making them ideal testbeds for experimentation on realistically complex systems. We evaluated six frontier LLMs on 137 small systems, and released a total of 350 systems. Our evaluation shows that while more capable models demonstrated superior performance, all models' performance declined significantly as system complexity increased, suggesting substantial room for improvement in the scientific capabilities of LLM agents. 

**Abstract (ZH)**: 设计实验和结果解释是核心的科学能力，特别是在生物学中，研究人员通过干扰复杂系统来揭示其内在机制。近期对大型语言模型（LLMs）的科学能力评估未能测试这些能力，因为湿实验的成本极高：在专业知识、时间和设备方面。我们引入了SciGym，这是一个首创的基准，用于评估LLMs在开放性科学发现任务中迭代实验设计和分析的能力。SciGym通过运行生物系统的虚拟实验室，克服了湿实验成本的挑战。这些模型以系统生物学标记语言编码，适用于生成模拟数据，是进行复杂系统实验的理想试验平台。我们在137个小系统上评估了六种前沿的LLMs，并总共发布了350个系统。我们的评估表明，虽然更强大的模型表现出更优异的性能，但所有模型的性能随着系统复杂性的增加而显著下降，这表明在LLM代理的科学能力上有很大的改进空间。 

---
# Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs 

**Title (ZH)**: 节省计算资源的推理：大规模语言模型运行时自适应与可控计算综述 

**Authors**: Mohammad Ali Alomrani, Yingxue Zhang, Derek Li, Qianyi Sun, Soumyasundar Pal, Zhanguang Zhang, Yaochen Hu, Rohan Deepak Ajwani, Antonios Valkanas, Raika Karimi, Peng Cheng, Yunzhou Wang, Pengyi Liao, Hanrui Huang, Bin Wang, Jianye Hao, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2507.02076)  

**Abstract**: Large language models (LLMs) have rapidly progressed into general-purpose agents capable of solving a broad spectrum of tasks. However, current models remain inefficient at reasoning: they apply fixed inference-time compute regardless of task complexity, often overthinking simple problems while underthinking hard ones. This survey presents a comprehensive review of efficient test-time compute (TTC) strategies, which aim to improve the computational efficiency of LLM reasoning. We introduce a two-tiered taxonomy that distinguishes between L1-controllability, methods that operate under fixed compute budgets, and L2-adaptiveness, methods that dynamically scale inference based on input difficulty or model confidence. We benchmark leading proprietary LLMs across diverse datasets, highlighting critical trade-offs between reasoning performance and token usage. Compared to prior surveys on efficient reasoning, our review emphasizes the practical control, adaptability, and scalability of TTC methods. Finally, we discuss emerging trends such as hybrid thinking models and identify key challenges for future work towards making LLMs more computationally efficient, robust, and responsive to user constraints. 

**Abstract (ZH)**: 大型语言模型（LLMs）已迅速发展成为能够解决广泛任务的通用代理。然而，当前模型在推理方面仍然不够高效：它们在推理时固定使用计算资源，不论任务复杂度如何，往往会对简单问题过度思考，而对困难问题则思考不足。本文综述了高效测试时计算（TTC）策略，旨在提高LLM推理的计算效率。我们引入了两层分类体系，区分L1可控性方法，即在固定计算预算下的方法，和L2适应性方法，即根据输入难度或模型置信度动态调整推理的方法。我们在多种数据集上对比了行业领先的私有LLM，并强调了推理性能和token使用之间的关键权衡。与之前的高效推理综述相比，我们的综述更注重TTC方法的实际可控性、适应性和可扩展性。最后，我们讨论了新兴趋势，如混合思考模型，并指出了未来工作中使LLMs更高效、更鲁棒且更能响应用户约束的关键挑战。 

---
# HCVR: A Hybrid Approach with Correlation-aware Voting Rules for Feature Selection 

**Title (ZH)**: HCVR：一种基于相关性感知投票规则的混合特征选择方法 

**Authors**: Nikita Bhedasgaonkar, Rushikesh K. Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02073)  

**Abstract**: In this paper, we propose HCVR (Hybrid approach with Correlation-aware Voting Rules), a lightweight rule-based feature selection method that combines Parameter-to-Parameter (P2P) and Parameter-to-Target (P2T) correlations to eliminate redundant features and retain relevant ones. This method is a hybrid of non-iterative and iterative filtering approaches for dimensionality reduction. It is a greedy method, which works by backward elimination, eliminating possibly multiple features at every step. The rules contribute to voting for features, and a decision to keep or discard is made by majority voting. The rules make use of correlation thresholds between every pair of features, and between features and the target. We provide the results from the application of HCVR to the SPAMBASE dataset. The results showed improvement performance as compared to traditional non-iterative (CFS, mRMR and MI) and iterative (RFE, SFS and Genetic Algorithm) techniques. The effectiveness was assessed based on the performance of different classifiers after applying filtering. 

**Abstract (ZH)**: 基于相关性意识投票规则的混合特征选择方法HCVR 

---
# STELLA: Self-Evolving LLM Agent for Biomedical Research 

**Title (ZH)**: STELLA: 自我进化的生物医学研究大语言模型代理 

**Authors**: Ruofan Jin, Zaixi Zhang, Mengdi Wang, Le Cong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02004)  

**Abstract**: The rapid growth of biomedical data, tools, and literature has created a fragmented research landscape that outpaces human expertise. While AI agents offer a solution, they typically rely on static, manually curated toolsets, limiting their ability to adapt and scale. Here, we introduce STELLA, a self-evolving AI agent designed to overcome these limitations. STELLA employs a multi-agent architecture that autonomously improves its own capabilities through two core mechanisms: an evolving Template Library for reasoning strategies and a dynamic Tool Ocean that expands as a Tool Creation Agent automatically discovers and integrates new bioinformatics tools. This allows STELLA to learn from experience. We demonstrate that STELLA achieves state-of-the-art accuracy on a suite of biomedical benchmarks, scoring approximately 26\% on Humanity's Last Exam: Biomedicine, 54\% on LAB-Bench: DBQA, and 63\% on LAB-Bench: LitQA, outperforming leading models by up to 6 percentage points. More importantly, we show that its performance systematically improves with experience; for instance, its accuracy on the Humanity's Last Exam benchmark almost doubles with increased trials. STELLA represents a significant advance towards AI Agent systems that can learn and grow, dynamically scaling their expertise to accelerate the pace of biomedical discovery. 

**Abstract (ZH)**: 自适应的成长：STELLA——一种自我进化的生物医学AI代理 

---
# Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory 

**Title (ZH)**: 点3R：具有显式空间指针记忆的流式3D重建 

**Authors**: Yuqi Wu, Wenzhao Zheng, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02863)  

**Abstract**: Dense 3D scene reconstruction from an ordered sequence or unordered image collections is a critical step when bringing research in computer vision into practical scenarios. Following the paradigm introduced by DUSt3R, which unifies an image pair densely into a shared coordinate system, subsequent methods maintain an implicit memory to achieve dense 3D reconstruction from more images. However, such implicit memory is limited in capacity and may suffer from information loss of earlier frames. We propose Point3R, an online framework targeting dense streaming 3D reconstruction. To be specific, we maintain an explicit spatial pointer memory directly associated with the 3D structure of the current scene. Each pointer in this memory is assigned a specific 3D position and aggregates scene information nearby in the global coordinate system into a changing spatial feature. Information extracted from the latest frame interacts explicitly with this pointer memory, enabling dense integration of the current observation into the global coordinate system. We design a 3D hierarchical position embedding to promote this interaction and design a simple yet effective fusion mechanism to ensure that our pointer memory is uniform and efficient. Our method achieves competitive or state-of-the-art performance on various tasks with low training costs. Code is available at: this https URL. 

**Abstract (ZH)**: 从有序序列或无序图像集合中进行稠密三维场景重建是将计算机视觉研究引入实际场景的关键步骤。继DUSt3R提出的框架，该框架将图像对稠密地统一到共享坐标系中之后，后续方法通过维护隐式的记忆来实现更多图像的稠密三维重建。然而，这种隐式记忆在容量上有限，并且可能会遭受早期帧信息丢失的问题。我们提出Point3R，这是一个针对稠密流式三维重建的在线框架。具体而言，我们维护一个直接与当前场景三维结构相关的显式的空间指针记忆。该记忆中的每个指针都分配了一个特定的三维位置，并在全局坐标系中聚集附近的场景信息，形成一个变化的空间特征。最新帧中提取的信息与这个指针记忆进行显式的交互，从而使得当前观察能够被稠密地整合到全局坐标系中。我们设计了三维层次位置嵌入来促进这种交互，并设计了一个简单而有效的融合机制，以确保我们的指针记忆均匀且高效。我们的方法在各种任务上实现了有竞争力或最先进的性能，且训练成本较低。代码可从以下链接获得：this https URL。 

---
# LiteReality: Graphics-Ready 3D Scene Reconstruction from RGB-D Scans 

**Title (ZH)**: LiteReality：从RGB-D 扫描中实现 Ready-for-图形的 3D 场景重建 

**Authors**: Zhening Huang, Xiaoyang Wu, Fangcheng Zhong, Hengshuang Zhao, Matthias Nießner, Joan Lasenby  

**Link**: [PDF](https://arxiv.org/pdf/2507.02861)  

**Abstract**: We propose LiteReality, a novel pipeline that converts RGB-D scans of indoor environments into compact, realistic, and interactive 3D virtual replicas. LiteReality not only reconstructs scenes that visually resemble reality but also supports key features essential for graphics pipelines -- such as object individuality, articulation, high-quality physically based rendering materials, and physically based interaction. At its core, LiteReality first performs scene understanding and parses the results into a coherent 3D layout and objects with the help of a structured scene graph. It then reconstructs the scene by retrieving the most visually similar 3D artist-crafted models from a curated asset database. Next, the Material Painting module enhances realism by recovering high-quality, spatially varying materials. Finally, the reconstructed scene is integrated into a simulation engine with basic physical properties to enable interactive behavior. The resulting scenes are compact, editable, and fully compatible with standard graphics pipelines, making them suitable for applications in AR/VR, gaming, robotics, and digital twins. In addition, LiteReality introduces a training-free object retrieval module that achieves state-of-the-art similarity performance on the Scan2CAD benchmark, along with a robust material painting module capable of transferring appearances from images of any style to 3D assets -- even under severe misalignment, occlusion, and poor lighting. We demonstrate the effectiveness of LiteReality on both real-life scans and public datasets. Project page: this https URL; Video: this https URL 

**Abstract (ZH)**: 我们提出LiteReality，一种新颖的流水线，将室内的RGB-D扫描转换为紧凑、逼真且可交互的3D虚拟复制品。LiteReality不仅能重建视觉上相似于现实的场景，还支持图形流水线中必要的关键特征，如物体的独特性、关节运动、高质量的基于物理的渲染材料以及基于物理的交互。其核心过程首先是进行场景理解并将结果解析为一个连贯的3D布局和物体，利用结构化的场景图。然后通过检索受控资产数据库中最具视觉相似性的3D艺术家手工制作的模型重建场景。接下来，材质绘画模块通过恢复高质量、空间变化的材质来增强现实感。最后，重建的场景被整合到具备基本物理属性的模拟引擎中，以实现交互行为。生成的场景紧凑、可编辑，并完全兼容标准的图形流水线，适用于AR/VR、游戏、机器人技术和数字孪生等应用。此外，LiteReality引入了一个无训练对象检索模块，在Scan2CAD基准测试中实现了最先进的相似度性能，还具备一个鲁棒的材质绘画模块，能够将任何风格图像的外观转移到3D资产上，即使在严重的错位、遮挡和不良照明条件下也是如此。我们在实际扫描和公开数据集上展示了LiteReality的有效性。项目页面：https://this-url; 视频：https://this-url。 

---
# Answer Matching Outperforms Multiple Choice for Language Model Evaluation 

**Title (ZH)**: 语言模型评估中，答案匹配优于多项选择 

**Authors**: Nikhil Chandak, Shashwat Goel, Ameya Prabhu, Moritz Hardt, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2507.02856)  

**Abstract**: Multiple choice benchmarks have long been the workhorse of language model evaluation because grading multiple choice is objective and easy to automate. However, we show multiple choice questions from popular benchmarks can often be answered without even seeing the question. These shortcuts arise from a fundamental limitation of discriminative evaluation not shared by evaluations of the model's free-form, generative answers. Until recently, there appeared to be no viable, scalable alternative to multiple choice--but, we show that this has changed. We consider generative evaluation via what we call answer matching: Give the candidate model the question without the options, have it generate a free-form response, then use a modern language model with the reference answer to determine if the response matches the reference. To compare the validity of different evaluation strategies, we annotate MMLU-Pro and GPQA-Diamond to obtain human grading data, and measure the agreement of each evaluation approach. We find answer matching using recent models--even small ones--achieves near-perfect agreement, in the range of inter-annotator agreement. In contrast, both multiple choice evaluation and using LLM-as-a-judge without reference answers aligns poorly with human grading. Improving evaluations via answer matching is not merely a conceptual concern: the rankings of several models change significantly when evaluating their free-form responses with answer matching. In light of these findings, we discuss how to move the evaluation ecosystem from multiple choice to answer matching. 

**Abstract (ZH)**: 多项选择基准长期以来一直是语言模型评估的主要工具，因为多项选择评分客观且易于自动化。然而，我们展示出流行基准中的多项选择问题往往可以在看不到问题的情况下被回答。这些捷径源于区分性评估的固有局限性，这种局限性不适用于模型生成答案的评估。直到最近，似乎没有可行且可扩展的替代方案——但我们展示出这种情况已经改变。我们通过我们称之为答案匹配的生成评估方法来考虑：让候选模型不提供选项只回答问题，然后使用现代语言模型和参考答案来判断其响应是否匹配。为了对比不同评估策略的有效性，我们对MMLU-Pro和GPQA-Diamond进行标注以获得人类评分数据，并测量每种评估方法的一致性。我们发现，即使使用较小的模型进行答案匹配也能实现近乎完美的一致性，范围与注释者间的一致性相近。相反，多项选择评估和不使用参考答案的语言模型作为评判者的方法与人类评分的契合度较低。通过答案匹配改进评估不仅仅是一个概念性问题：在使用答案匹配评估其生成答案时，某些模型的排名发生了显著变化。鉴于这些发现，我们讨论如何将评估生态系统从多项选择转移到答案匹配。 

---
# Subtyping in DHOL -- Extended preprint 

**Title (ZH)**: DHOL中的子类型系统——扩展预印本 

**Authors**: Colin Rothgang, Florian Rabe  

**Link**: [PDF](https://arxiv.org/pdf/2507.02855)  

**Abstract**: The recently introduced dependent typed higher-order logic (DHOL) offers an interesting compromise between expressiveness and automation support. It sacrifices the decidability of its type system in order to significantly extend its expressiveness over standard HOL. Yet it retains strong automated theorem proving support via a sound and complete translation to HOL.
We leverage this design to extend DHOL with refinement and quotient types. Both of these are commonly requested by practitioners but rarely provided by automated theorem provers. This is because they inherently require undecidable typing and thus are very difficult to retrofit to decidable type systems. But with DHOL already doing the heavy lifting, adding them is not only possible but elegant and simple.
Concretely, we add refinement and quotient types as special cases of subtyping. This turns the associated canonical inclusion resp. projection maps into identity maps and thus avoids costly changes in representation. We present the syntax, semantics, and translation to HOL for the extended language, including the proofs of soundness and completeness. 

**Abstract (ZH)**: 最近引入的依赖类型高阶逻辑（DHOL）在表达能力和自动化支持之间提供了有趣的权衡。它牺牲了类型系统的可判定性，以显著扩展其相对于标准 HOL 的表达能力。然而，它仍然通过一种 sound 和 complete 的转换保持了强大的自动化定理证明支持。我们利用这一设计，将细化和商类型扩展到 DHOL。这两者通常是由实践者们请求的，但自动化定理证明器很少提供。这是因为它们本质上要求类型判断不可判定，因此很难适应可判定的类型系统。但是，由于 DHOL 已经处理了大部分难题，添加它们不仅可行而且简洁而优雅。具体来说，我们将细化和商类型作为子类型的一种特殊情形加入，将相关的标型插入映射和投影映射转化为恒等映射，从而避免了昂贵的表示更改。我们展示了扩展语言的形式语法、语义以及到 HOL 的转换，并包括完整性和正确性的证明。 

---
# MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs 

**Title (ZH)**: MOTIF: 在LLMs中通过强化微调实现模块化思考 

**Authors**: Purbesh Mitra, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2507.02851)  

**Abstract**: Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at this https URL and this https URL, respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）推理能力的 recent 进展表明，使用组相对策略优化（GRPO）算法进行强化学习（RL）训练可以使模型使用更多思考/推理令牌以生成更好的回应。然而，LLMs 在维持对先前生成令牌的关注时，只能生成有限数量的令牌。这一限制，也称为 LLM 的上下文大小，是 LLM 在处理任意大量令牌时推理的一个瓶颈。为了超越上下文大小的限制，LLM 必须采用模块化思考策略，分多轮进行推理。在本项工作中，我们提出了 $\textbf{MOTIF: 模块化思考通过强化学习微调}$ —— 一种用于多轮生成思考令牌的 RL 训练方法，有效地使模型能够在更大的上下文大小下思考。我们通过参数高效微调在 Qwen2.5-3B-Instruct 模型上对 GSM8K 数据集进行了训练，并在 MATH500 和 AIME2024 指标上测试了其准确性。我们的实验结果显示，在相应的基准中，MOTIF 比基于 vanilla GRPO 的训练分别提高了 3.8% 和 3.3% 的准确性。此外，这种改进仅使用了 15% 的样本，从而展示了 MOTIF 的样本效率。我们的代码和模型分别在此 https URL 和此 https URL 上提供。 

---
# USAD: An Unsupervised Data Augmentation Spatio-Temporal Attention Diffusion Network 

**Title (ZH)**: USAD：无监督数据增强空时注意力扩散网络 

**Authors**: Ying Yu, Hang Xiao, Siyao Li, Jiarui Li, Haotian Tang, Hanyu Liu, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.02827)  

**Abstract**: The primary objective of human activity recognition (HAR) is to infer ongoing human actions from sensor data, a task that finds broad applications in health monitoring, safety protection, and sports analysis. Despite proliferating research, HAR still faces key challenges, including the scarcity of labeled samples for rare activities, insufficient extraction of high-level features, and suboptimal model performance on lightweight devices. To address these issues, this paper proposes a comprehensive optimization approach centered on multi-attention interaction mechanisms. First, an unsupervised, statistics-guided diffusion model is employed to perform data augmentation, thereby alleviating the problems of labeled data scarcity and severe class imbalance. Second, a multi-branch spatio-temporal interaction network is designed, which captures multi-scale features of sequential data through parallel residual branches with 3*3, 5*5, and 7*7 convolutional kernels. Simultaneously, temporal attention mechanisms are incorporated to identify critical time points, while spatial attention enhances inter-sensor interactions. A cross-branch feature fusion unit is further introduced to improve the overall feature representation capability. Finally, an adaptive multi-loss function fusion strategy is integrated, allowing for dynamic adjustment of loss weights and overall model optimization. Experimental results on three public datasets, WISDM, PAMAP2, and OPPORTUNITY, demonstrate that the proposed unsupervised data augmentation spatio-temporal attention diffusion network (USAD) achieves accuracies of 98.84%, 93.81%, and 80.92% respectively, significantly outperforming existing approaches. Furthermore, practical deployment on embedded devices verifies the efficiency and feasibility of the proposed method. 

**Abstract (ZH)**: 人类活动识别（HAR）的主要目标是从传感器数据中推断正在进行的人类行为，这一任务广泛应用于健康监测、安全保护和体育分析。尽管进行了大量的研究，HAR仍面临关键挑战，包括稀有活动标记样本稀缺、高级特征提取不足以及轻量级设备上的模型性能不佳。为解决这些问题，本文提出了一种以多注意力交互机制为中心的全面优化方法。首先，采用无监督的统计指导扩散模型进行数据增强，以缓解标记数据稀缺和类间严重不平衡的问题。其次，设计了一种多分支时空交互网络，通过并行残差分支中的3×3、5×5和7×7卷积核捕获序列数据的多尺度特征。同时，嵌入时间注意力机制以识别关键时间点，空间注意力则增强传感器间的交互。进一步引入跨分支特征融合单元以提升整体特征表示能力。最后，集成了一种自适应多损失函数融合策略，允许动态调整损失权重并优化整体模型。在三个公开数据集WISDM、PAMAP2和OPPORTUNITY上的实验结果表明，所提出的无监督数据增强时空注意力扩散网络（USAD）分别实现了98.84%、93.81%和80.92%的准确性，显著优于现有方法。此外，实际部署在嵌入式设备上验证了所提出方法的有效性和可行性。 

---
# DNN-Based Precoding in RIS-Aided mmWave MIMO Systems With Practical Phase Shift 

**Title (ZH)**: 基于DNN的RIS辅助毫米波MIMO系统中考虑实际相移的预编码技术 

**Authors**: Po-Heng Chou, Ching-Wen Chen, Wan-Jen Huang, Walid Saad, Yu Tsao, Ronald Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02824)  

**Abstract**: In this paper, the precoding design is investigated for maximizing the throughput of millimeter wave (mmWave) multiple-input multiple-output (MIMO) systems with obstructed direct communication paths. In particular, a reconfigurable intelligent surface (RIS) is employed to enhance MIMO transmissions, considering mmWave characteristics related to line-of-sight (LoS) and multipath effects. The traditional exhaustive search (ES) for optimal codewords in the continuous phase shift is computationally intensive and time-consuming. To reduce computational complexity, permuted discrete Fourier transform (DFT) vectors are used for finding codebook design, incorporating amplitude responses for practical or ideal RIS systems. However, even if the discrete phase shift is adopted in the ES, it results in significant computation and is time-consuming. Instead, the trained deep neural network (DNN) is developed to facilitate faster codeword selection. Simulation results show that the DNN maintains sub-optimal spectral efficiency even as the distance between the end-user and the RIS has variations in the testing phase. These results highlight the potential of DNN in advancing RIS-aided systems. 

**Abstract (ZH)**: 毫米波多输入多输出系统中具有阻挡直接通信路径的波束形成设计研究——考虑可重构智能表面的帮助 

---
# SynapseRoute: An Auto-Route Switching Framework on Dual-State Large Language Model 

**Title (ZH)**: SynapseRoute: 双状态大型语言模型的自适应路由切换框架 

**Authors**: Wencheng Zhang, Shiqin Qiao, Lingjie Luo, Yinfeng Li, Chuanyang Zheng, Qian Xu, Meng Li, Yong Gui, Yijun He, Jianing Qiu, Jindong Hong, Jiankai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.02822)  

**Abstract**: With the widespread adoption of large language models (LLMs) in practical applications, selecting an appropriate model requires balancing not only performance but also operational cost. The emergence of reasoning-capable models has further widened the cost gap between "thinking" (high reasoning) and "non-thinking" (fast, low-cost) modes. In this work, we reveal that approximately 58% of medical questions can be accurately answered by the non-thinking mode alone, without requiring the high-cost reasoning process. This highlights a clear dichotomy in problem complexity and suggests that dynamically routing queries to the appropriate mode based on complexity could optimize accuracy, cost-efficiency, and overall user experience. Based on this, we further propose SynapseRoute, a machine learning-based dynamic routing framework that intelligently assigns input queries to either thinking or non-thinking modes. Experimental results on several medical datasets demonstrate that SynapseRoute not only improves overall accuracy (0.8390 vs. 0.8272) compared to the thinking mode alone but also reduces inference time by 36.8% and token consumption by 39.66%. Importantly, qualitative analysis indicates that over-reasoning on simpler queries can lead to unnecessary delays and even decreased accuracy, a pitfall avoided by our adaptive routing. Finally, this work further introduces the Accuracy-Inference-Token (AIT) index to comprehensively evaluate the trade-offs among accuracy, latency, and token cost. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在实际应用中的广泛应用，选择合适的模型不仅需要平衡性能，还需要平衡运营成本。具有推理能力的模型的出现进一步拉开了“思考”（高推理）模式与“非思考”（快速、低成本）模式之间成本差距。在本文中，我们揭示了大约58%的医疗问题仅通过非思考模式即可准确回答，无需进行高成本的推理过程。这突显了问题复杂度的明显二元性，并表明根据复杂度动态路由查询可以优化准确度、成本效率和整体用户体验。基于此，我们进一步提出了一种基于机器学习的动态路由框架SynapseRoute，该框架能够智能地将输入查询分配给思考或非思考模式。在几个医疗数据集上的实验结果表明，与单独使用思考模式相比，SynapseRoute不仅总体准确度提高了（0.8390 vs. 0.8272），还减少了36.8%的推理时间和39.66%的令牌消耗。更重要的是，定性分析表明，在简单查询上过度推理可能导致不必要的延迟甚至降低准确度，这是我们动态路由方法避免的问题。最后，本文还引入了准确度-推理时间-令牌成本（AIT）指数，以全面评估准确度、延迟和令牌成本之间的权衡。 

---
# Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs 

**Title (ZH)**: 自我修正平台：揭示并解决大规模语言模型中的自我修正盲区 

**Authors**: Ken Tsui  

**Link**: [PDF](https://arxiv.org/pdf/2507.02778)  

**Abstract**: Although large language models (LLMs) have become transformative, they still make mistakes and can explore unproductive reasoning paths. Self-correction is an important capability for a trustworthy LLM, particularly an autoregressive LLM. While LLMs can identify error in user input, they exhibit a systematic 'Self-Correction Blind Spot' - failing to correct identical error in their own outputs. To systematically study this phenomenon, we introduce Self-Correction Bench, a systematic framework to measure this phenomenon through controlled error injection at three complexity levels. Testing 14 models, we find an average 64.5% blind spot rate. We find multiple evidences that this limitation relates to training data composition: human training demonstrations predominantly show error-free responses rather than error-correction sequences, unlike RL-trained models that learn error correction through outcome feedback. Remarkably, simply appending "Wait" reduces blind spots by 89.3%, suggesting that the capability exists but requires activation. Our work highlights a critical limitation in current LLMs and offers potential avenues for improving their reliability and trustworthiness. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）已经变得极具变革性，它们仍然会出错，并且可能会探索无效的推理路径。自我修正是一项重要能力，特别是对于自回归的大语言模型而言。虽然LLMs可以识别用户输入中的错误，但它们表现出一种系统性的“自我修正盲区”——无法修正自身输出中的相同错误。为了系统性地研究这一现象，我们引入了Self-Correction Bench这一系统框架，通过在三个复杂程度级别上进行受控的错误注入来衡量这一现象。测试14个模型后，我们发现平均盲区率为64.5%。我们发现多种证据表明，这一局限与训练数据的组成有关：人类训练示例主要展示无错误的响应，而非错误纠正序列，这与通过结果反馈学习错误纠正的强化学习训练模型不同。令人惊讶的是，仅仅添加“等待”就可将盲区降低89.3%，表明该能力存在但需要激活。我们的工作突显了当前LLMs的一项关键局限，并为提高其可靠性和可信度提供了潜在途径。 

---
# Multi-agent Auditory Scene Analysis 

**Title (ZH)**: 多智能体声场景分析 

**Authors**: Caleb Rascon, Luis Gato-Diaz, Eduardo García-Alarcón  

**Link**: [PDF](https://arxiv.org/pdf/2507.02755)  

**Abstract**: Auditory scene analysis (ASA) aims to retrieve information from the acoustic environment, by carrying out three main tasks: sound source location, separation, and classification. These tasks are traditionally executed with a linear data flow, where the sound sources are first located; then, using their location, each source is separated into its own audio stream; from each of which, information is extracted that is relevant to the application scenario (audio event detection, speaker identification, emotion classification, etc.). However, running these tasks linearly increases the overall response time, while making the last tasks (separation and classification) highly sensitive to errors of the first task (location). A considerable amount of effort and computational complexity has been employed in the state-of-the-art to develop techniques that are the least error-prone possible. However, doing so gives rise to an ASA system that is non-viable in many applications that require a small computational footprint and a low response time, such as bioacoustics, hearing-aid design, search and rescue, human-robot interaction, etc. To this effect, in this work, a multi-agent approach is proposed to carry out ASA where the tasks are run in parallel, with feedback loops between them to compensate for local errors, such as: using the quality of the separation output to correct the location error; and using the classification result to reduce the localization's sensitivity towards interferences. The result is a multi-agent auditory scene analysis (MASA) system that is robust against local errors, without a considerable increase in complexity, and with a low response time. The complete proposed MASA system is provided as a framework that uses open-source tools for sound acquisition and reproduction (JACK) and inter-agent communication (ROS2), allowing users to add their own agents. 

**Abstract (ZH)**: 基于多代理的 auditory 场景分析（Multi-agent Based Auditory Scene Analysis, MASA） 

---
# Fast and Simplex: 2-Simplicial Attention in Triton 

**Title (ZH)**: 快速简洁：Triton中的2-单纯形注意力 

**Authors**: Aurko Roy, Timothy Chou, Sai Surya Duvvuri, Sijia Chen, Jiecao Yu, Xiaodong Wang, Manzil Zaheer, Rohan Anil  

**Link**: [PDF](https://arxiv.org/pdf/2507.02754)  

**Abstract**: Recent work has shown that training loss scales as a power law with both model size and the number of tokens, and that achieving compute-optimal models requires scaling model size and token count together. However, these scaling laws assume an infinite supply of data and apply primarily in compute-bound settings. As modern large language models increasingly rely on massive internet-scale datasets, the assumption that they are compute-bound is becoming less valid. This shift highlights the need for architectures that prioritize token efficiency.
In this work, we investigate the use of the 2-simplicial Transformer, an architecture that generalizes standard dot-product attention to trilinear functions through an efficient Triton kernel implementation. We demonstrate that the 2-simplicial Transformer achieves better token efficiency than standard Transformers: for a fixed token budget, similarly sized models outperform their dot-product counterparts on tasks involving mathematics, coding, reasoning, and logic. We quantify these gains by demonstrating that $2$-simplicial attention changes the exponent in the scaling laws for knowledge and reasoning tasks compared to dot product attention. 

**Abstract (ZH)**: 最近的研究表明，训练损失与模型规模和令牌数量呈幂律关系，并且实现计算最优模型需要同时扩展模型规模和令牌数量。然而，这些扩展定律假设数据无限供应，并主要适用于计算受限的环境。随着现代大型语言模型越来越多地依赖于大规模互联网数据集，它们计算受限的假设变得不再那么有效。这种转变突显了优先考虑令牌效率的架构的需求。

在这项工作中，我们探讨了2-单形变换器的应用，这是一种架构，通过高效的Triton内核实现将标准点积注意扩展到三线性函数。我们证明，2-单形变换器在令牌效率方面优于标准变换器：在固定令牌预算的情况下，相似规模的模型在涉及数学、编程、推理和逻辑的任务中表现优于点积版本。我们通过证明与点积注意力相比，2-单形注意力改变了知识和推理任务的扩展定律的指数来量化这些增益。 

---
# Synthesizable by Design: A Retrosynthesis-Guided Framework for Molecular Analog Generation 

**Title (ZH)**: 设计合乎合成需求：一种 retrosynthesis 引导的分子模拟生成框架 

**Authors**: Shuan Chen, Gunwook Nam, Yousung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2507.02752)  

**Abstract**: The disconnect between AI-generated molecules with desirable properties and their synthetic feasibility remains a critical bottleneck in computational drug and material discovery. While generative AI has accelerated the proposal of candidate molecules, many of these structures prove challenging or impossible to synthesize using established chemical reactions. Here, we introduce SynTwins, a novel retrosynthesis-guided molecular analog design framework that designs synthetically accessible molecular analogs by emulating expert chemist strategies through a three-step process: retrosynthesis, similar building block searching, and virtual synthesis. In comparative evaluations, SynTwins demonstrates superior performance in generating synthetically accessible analogs compared to state-of-the-art machine learning models while maintaining high structural similarity to original target molecules. Furthermore, when integrated with existing molecule optimization frameworks, our hybrid approach produces synthetically feasible molecules with property profiles comparable to unconstrained molecule generators, yet its synthesizability ensured. Our comprehensive benchmarking across diverse molecular datasets demonstrates that SynTwins effectively bridges the gap between computational design and experimental synthesis, providing a practical solution for accelerating the discovery of synthesizable molecules with desired properties for a wide range of applications. 

**Abstract (ZH)**: 生成具有良好性质的分子与其实用合成性之间的差距仍然是计算药物和材料发现中的一个关键瓶颈。尽管生成式AI加速了候选分子的提出，但许多这些结构仍然难以或不可能通过已知的化学反应进行合成。为此，我们提出了一种新的反合成反应导向的分子类比设计框架SynTwins，该框架通过模拟专家化学家的策略，在三个步骤过程中设计合成上可行的分子类比物：反合成反应、相似构建基元搜索和虚拟合成。在对比评估中，SynTwins在生成合成上可行的类比物方面展示了优于当前最先进的机器学习模型的能力，同时保持与原始目标分子的高结构相似性。此外，当与现有的分子优化框架集成时，我们的混合方法能够产生具有与无约束分子生成器相当的性质谱但使其合成化的分子。我们对各种分子数据集的全面基准测试表明，SynTwins有效地弥合了计算设计与实验合成之间的差距，提供了一种加速发现具有所需性质的可合成分子的实用解决方案，适用于多种应用。 

---
# Linear Attention with Global Context: A Multipole Attention Mechanism for Vision and Physics 

**Title (ZH)**: 全局上下文下的线性注意力：用于视觉和物理的多极注意力机制 

**Authors**: Alex Colagrande, Paul Caillon, Eva Feillet, Alexandre Allauzen  

**Link**: [PDF](https://arxiv.org/pdf/2507.02748)  

**Abstract**: Transformers have become the de facto standard for a wide range of tasks, from image classification to physics simulations. Despite their impressive performance, the quadratic complexity of standard Transformers in both memory and time with respect to the input length makes them impractical for processing high-resolution inputs. Therefore, several variants have been proposed, the most successful relying on patchification, downsampling, or coarsening techniques, often at the cost of losing the finest-scale details. In this work, we take a different approach. Inspired by state-of-the-art techniques in $n$-body numerical simulations, we cast attention as an interaction problem between grid points. We introduce the Multipole Attention Neural Operator (MANO), which computes attention in a distance-based multiscale fashion. MANO maintains, in each attention head, a global receptive field and achieves linear time and memory complexity with respect to the number of grid points. Empirical results on image classification and Darcy flows demonstrate that MANO rivals state-of-the-art models such as ViT and Swin Transformer, while reducing runtime and peak memory usage by orders of magnitude. We open source our code for reproducibility at this https URL. 

**Abstract (ZH)**: 基于网格点间的相互作用的多极注意力神经算子 

---
# Early Signs of Steganographic Capabilities in Frontier LLMs 

**Title (ZH)**: 前沿大语言模型中的早期隐写能力迹象 

**Authors**: Artur Zolkowski, Kei Nishimura-Gasparian, Robert McCarthy, Roland S. Zimmermann, David Lindner  

**Link**: [PDF](https://arxiv.org/pdf/2507.02737)  

**Abstract**: Monitoring Large Language Model (LLM) outputs is crucial for mitigating risks from misuse and misalignment. However, LLMs could evade monitoring through steganography: Encoding hidden information within seemingly benign generations. In this paper, we evaluate the steganography capabilities in frontier LLMs to better understand the risk they pose. We focus on two types of steganography: passing encoded messages and performing encoded reasoning. We find that current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances. They can succeed, however, if given additional affordances such as using an unmonitored scratchpad and coordinating on what encoding scheme to use. We additionally find early signs that models can perform basic encoded reasoning in a simple state-tracking problem. This includes some ability to reason with their own and pre-defined schemes, including encoding schemes such as Hexadecimal. Despite this, they can rarely hide reasoning subtly within a cover task to fool a monitor. Overall, our results indicate that current LLMs exhibit nascent steganographic capabilities. While these capabilities are likely insufficient to bypass well-designed monitors at present, this could change in the future. 

**Abstract (ZH)**: 监控大型语言模型（LLM）的输出对于减轻滥用和不对齐风险至关重要。然而，LLM可以通过藏码技术规避监控：在看似无害的生成中嵌入隐藏信息。在本文中，我们评估前沿LLM的藏码能力，以更好地理解它们所造成的风险。我们重点关注两种类型的藏码：传递编码信息和进行编码推理。我们发现，当前模型无法在标准条件下在其输出中嵌入短消息而不被监控发现。然而，如果提供额外的条件，如使用未监控的草稿纸并协调使用的编码方案，则它们可以成功。此外，我们还发现模型在简单状态跟踪问题中可以进行基本的编码推理的早期迹象，包括与自身和预定义方案进行推理的能力，包括十六进制等编码方案。尽管如此，它们很少能够通过隐秘地将推理融入辅助任务中欺骗监控。总体而言，我们的结果表明当前LLM展现出了初步的藏码能力。尽管这些能力当前似乎不足以绕过精心设计的监控系统，但这一点在未来可能会改变。 

---
# Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks 

**Title (ZH)**: Meta SecAlign: 针对提示注入攻击的secure基础LLM 

**Authors**: Sizhe Chen, Arman Zharmagambetov, David Wagner, Chuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02735)  

**Abstract**: Prompt injection attacks pose a significant security threat to LLM-integrated applications. Model-level defenses have shown strong effectiveness, but are currently deployed into commercial-grade models in a closed-source manner. We believe open-source models are needed by the AI security community, where co-development of attacks and defenses through open research drives scientific progress in mitigation against prompt injection attacks. To this end, we develop Meta SecAlign, the first open-source and open-weight LLM with built-in model-level defense that achieves commercial-grade model performance. We provide complete details of our training recipe, which utilizes an improved version of the SOTA SecAlign defense. Evaluations on 9 utility benchmarks and 7 security benchmarks show that Meta SecAlign, despite being trained on a generic instruction-tuning dataset, confers security in unseen downstream tasks, including tool-calling and agentic web navigation, in addition general instruction-following. Our best model -- Meta-SecAlign-70B -- achieves state-of-the-art robustness against prompt injection attacks and comparable utility to closed-source commercial LLM with model-level defense. 

**Abstract (ZH)**: 提示注入攻击对LLM集成应用构成重大安全威胁。内置模型级防御的开源模型是AI安全社区所需，通过开放研究促进对抗提示注入攻击的科学进步。为此，我们开发了Meta SecAlign，这是第一个集成了模型级防御并达到商业级模型性能的开源且开放权重的LLM。我们提供了完整的训练方案细节，该方案使用了SOTA SecAlign防御的改进版本。在9个功能基准和7个安全基准上的评估显示，Meta SecAlign不仅在通用指令遵循任务中表现出安全性，在未见的下游任务中，如工具调用和代理型网页导航中也表现出安全性。我们的最佳模型——Meta-SecAlign-70B——在提示注入攻击下的鲁棒性达到了最新水平，并且在模型级防御方面的功能与商业级闭源LLM相当。 

---
# FairHuman: Boosting Hand and Face Quality in Human Image Generation with Minimum Potential Delay Fairness in Diffusion Models 

**Title (ZH)**: FairHuman: 在最小潜在延迟公平性约束下提高人类图像生成中的手部和面部质量 

**Authors**: Yuxuan Wang, Tianwei Cao, Huayu Zhang, Zhongjiang He, Kongming Liang, Zhanyu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.02714)  

**Abstract**: Image generation has achieved remarkable progress with the development of large-scale text-to-image models, especially diffusion-based models. However, generating human images with plausible details, such as faces or hands, remains challenging due to insufficient supervision of local regions during training. To address this issue, we propose FairHuman, a multi-objective fine-tuning approach designed to enhance both global and local generation quality fairly. Specifically, we first construct three learning objectives: a global objective derived from the default diffusion objective function and two local objectives for hands and faces based on pre-annotated positional priors. Subsequently, we derive the optimal parameter updating strategy under the guidance of the Minimum Potential Delay (MPD) criterion, thereby attaining fairness-ware optimization for this multi-objective problem. Based on this, our proposed method can achieve significant improvements in generating challenging local details while maintaining overall quality. Extensive experiments showcase the effectiveness of our method in improving the performance of human image generation under different scenarios. 

**Abstract (ZH)**: 基于多目标微调的FairHuman图像生成方法 

---
# APT: Adaptive Personalized Training for Diffusion Models with Limited Data 

**Title (ZH)**: 自适应个性化训练：基于有限数据的扩散模型训练方法 

**Authors**: JungWoo Chae, Jiyoon Kim, JaeWoong Choi, Kyungyul Kim, Sangheum Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02687)  

**Abstract**: Personalizing diffusion models using limited data presents significant challenges, including overfitting, loss of prior knowledge, and degradation of text alignment. Overfitting leads to shifts in the noise prediction distribution, disrupting the denoising trajectory and causing the model to lose semantic coherence. In this paper, we propose Adaptive Personalized Training (APT), a novel framework that mitigates overfitting by employing adaptive training strategies and regularizing the model's internal representations during fine-tuning. APT consists of three key components: (1) Adaptive Training Adjustment, which introduces an overfitting indicator to detect the degree of overfitting at each time step bin and applies adaptive data augmentation and adaptive loss weighting based on this indicator; (2)Representation Stabilization, which regularizes the mean and variance of intermediate feature maps to prevent excessive shifts in noise prediction; and (3) Attention Alignment for Prior Knowledge Preservation, which aligns the cross-attention maps of the fine-tuned model with those of the pretrained model to maintain prior knowledge and semantic coherence. Through extensive experiments, we demonstrate that APT effectively mitigates overfitting, preserves prior knowledge, and outperforms existing methods in generating high-quality, diverse images with limited reference data. 

**Abstract (ZH)**: 利用有限数据个性化扩散模型面临的挑战及其克服方法：Adaptive Personalized Training (APT) mitigates overfitting, preserves prior knowledge, and generates high-quality, diverse images with limited reference data. 

---
# ASDA: Audio Spectrogram Differential Attention Mechanism for Self-Supervised Representation Learning 

**Title (ZH)**: ASDA: 音频频谱差异注意力机制用于自我监督表示学习 

**Authors**: Junyu Wang, Tianrui Wang, Meng Ge, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02666)  

**Abstract**: In recent advancements in audio self-supervised representation learning, the standard Transformer architecture has emerged as the predominant approach, yet its attention mechanism often allocates a portion of attention weights to irrelevant information, potentially impairing the model's discriminative ability. To address this, we introduce a differential attention mechanism, which effectively mitigates ineffective attention allocation through the integration of dual-softmax operations and appropriately tuned differential coefficients. Experimental results demonstrate that our ASDA model achieves state-of-the-art (SOTA) performance across multiple benchmarks, including audio classification (49.0% mAP on AS-2M, 41.5% mAP on AS20K), keyword spotting (98.3% accuracy on SPC-2), and environmental sound classification (96.1% accuracy on ESC-50). These results highlight ASDA's effectiveness in audio tasks, paving the way for broader applications. 

**Abstract (ZH)**: 近期音频自监督表示学习的进展中，标准Transformer架构已成为主导方法，但其注意力机制往往会将部分注意力权重分配给无关信息，可能损害模型的辨别能力。为解决这一问题，我们引入了一种差异性注意力机制，该机制通过集成双softmax操作和适当地调平方差系数，有效地减轻了无效注意力分配。实验结果显示，我们的ASDA模型在多个基准测试中实现了最佳性能，涵盖音频分类（AS-2M上的49.0% mAP，AS20K上的41.5% mAP）、关键词识别（SPC-2上的98.3%准确率）和环境声分类（ESC-50上的96.1%准确率）。这些结果突显了ASDA在音频任务中的有效性，为更广泛的应用铺平了道路。 

---
# Solving the Hubbard model with Neural Quantum States 

**Title (ZH)**: 用神经量子态求解霍尔ब模型 

**Authors**: Yuntian Gu, Wenrui Li, Heng Lin, Bo Zhan, Ruichen Li, Yifei Huang, Di He, Yantao Wu, Tao Xiang, Mingpu Qin, Liwei Wang, Dingshun Lv  

**Link**: [PDF](https://arxiv.org/pdf/2507.02644)  

**Abstract**: The rapid development of neural quantum states (NQS) has established it as a promising framework for studying quantum many-body systems. In this work, by leveraging the cutting-edge transformer-based architectures and developing highly efficient optimization algorithms, we achieve the state-of-the-art results for the doped two-dimensional (2D) Hubbard model, arguably the minimum model for high-Tc superconductivity. Interestingly, we find different attention heads in the NQS ansatz can directly encode correlations at different scales, making it capable of capturing long-range correlations and entanglements in strongly correlated systems. With these advances, we establish the half-filled stripe in the ground state of 2D Hubbard model with the next nearest neighboring hoppings, consistent with experimental observations in cuprates. Our work establishes NQS as a powerful tool for solving challenging many-fermions systems. 

**Abstract (ZH)**: 神经量子态（NQS）的快速发展已经使其成为研究量子多体系统有前途的框架。通过利用先进的基于变换器的架构并开发高效的优化算法，我们在此工作中实现了掺杂二维（2D）Hubbard模型的最新成果，这可能是高温超导性中最基本的模型。有趣的是，我们发现NQS ansatz中的不同注意力头可以直接编码不同尺度的关联，使其能够捕获强关联系统中的长程关联和纠缠。借助这些进展，我们建立了具有最邻近和次邻近跃迁的二维Hubbard模型的半填充条纹相，这与 cuprates 中的实验观察结果一致。我们的工作确立了NQS作为解决具有挑战性的多重费米子系统工具的地位。 

---
# FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference 

**Title (ZH)**: FlowSpec：连续管道 speculative 解码以实现高效的分布式大语言模型推理 

**Authors**: Xing Liu, Lizhuo Luo, Ming Tang, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02620)  

**Abstract**: Distributed inference serves as a promising approach to enabling the inference of large language models (LLMs) at the network edge. It distributes the inference process to multiple devices to ensure that the LLMs can fit into the device memory. Recent pipeline-based approaches have the potential to parallelize communication and computation, which helps reduce inference latency. However, the benefit diminishes when the inference request at the network edge is sparse, where pipeline is typically at low utilization. To enable efficient distributed LLM inference at the edge, we propose \textbf{FlowSpec}, a pipeline-parallel tree-based speculative decoding framework. FlowSpec incorporates three key mechanisms to improve decoding efficiency: 1) score-based step-wise verification prioritizes more important draft tokens to bring earlier accpeted tokens; 2) efficient draft management to prune invalid tokens while maintaining correct causal relationship during verification; 3) dynamic draft expansion strategies to supply high-quality speculative inputs. These techniques work in concert to enhance both pipeline utilization and speculative efficiency. We evaluate FlowSpec on a real-world testbed with other baselines. Experimental results demonstrate that our proposed framework significantly improves inference speed across diverse models and configurations, achieving speedup ratios 1.36$\times$-1.77$\times$ compared to baselines. Our code is publicly available at \href{this https URL}{this https URL\#} 

**Abstract (ZH)**: 分布式推理作为一种有前景的方法，适用于在网络边缘进行大型语言模型（LLMs）的推理。它将推理过程分布在多个设备上，确保LLMs能够适应设备内存。基于管道的方法具有潜在的并行化通信和计算的能力，有助于减少推理延迟。然而，在网络边缘推理请求稀疏的情况下，管道通常利用率低，这种优势会减弱。为在边缘启用高效分布式LLM推理，我们提出了一种基于管道并行树状推测解码框架\textbf{FlowSpec}。FlowSpec结合了三种关键技术来提高解码效率：1) 基于得分的逐步验证优先处理更重要草稿 token，以带来更早被接受的 token；2) 高效的草稿管理，在验证过程中修剪无效 token 同时保持正确的因果关系；3) 动态草稿扩展策略，提供高质量的推测输入。这些技术共同提升管道利用率和推测效率。我们在现实测试平台上对FlowSpec与其他基准进行了评估。实验结果表明，我们提出的方法在多种模型和配置下显著提高了推理速度，相比基准方法实现了1.36$\times$-1.77$\times$的加速比。我们的代码已公开在\href{this https URL}{this https URL\#}。 

---
# De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks 

**Title (ZH)**: De-AntiFake: 重新思考对抗语音 cloning 攻击的保护性扰动 

**Authors**: Wei Fan, Kejiang Chen, Chang Liu, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02606)  

**Abstract**: The rapid advancement of speech generation models has heightened privacy and security concerns related to voice cloning (VC). Recent studies have investigated disrupting unauthorized voice cloning by introducing adversarial perturbations. However, determined attackers can mitigate these protective perturbations and successfully execute VC. In this study, we conduct the first systematic evaluation of these protective perturbations against VC under realistic threat models that include perturbation purification. Our findings reveal that while existing purification methods can neutralize a considerable portion of the protective perturbations, they still lead to distortions in the feature space of VC models, which degrades the performance of VC. From this perspective, we propose a novel two-stage purification method: (1) Purify the perturbed speech; (2) Refine it using phoneme guidance to align it with the clean speech distribution. Experimental results demonstrate that our method outperforms state-of-the-art purification methods in disrupting VC defenses. Our study reveals the limitations of adversarial perturbation-based VC defenses and underscores the urgent need for more robust solutions to mitigate the security and privacy risks posed by VC. The code and audio samples are available at this https URL. 

**Abstract (ZH)**: 快速发展的语音生成模型加剧了语音克隆（VC）相关的隐私和安全担忧。近期研究通过引入对抗性扰动来干扰未经授权的语音克隆。然而，有决心的攻击者能够削弱这些防护扰动并成功执行VC。本研究首次在包括扰动净化在内的现实威胁模型下系统评估这些防护扰动的有效性。研究发现，虽然现有的净化方法能够中和相当一部分防护扰动，但仍会导致VC模型特征空间的失真，从而降低其性能。基于此，我们提出了一种新型的两阶段净化方法：（1）净化被扰动的语音；（2）利用音素指导进一步优化以与干净语音分布对齐。实验结果表明，我们的方法在干扰VC防御方面优于现有的最先进的净化方法。本研究揭示了基于对抗性扰动的VC防护措施的局限性，并强调了更稳健解决方案的迫切需求，以减轻VC带来的安全和隐私风险。更多代码和音频样本请访问此网址。 

---
# Addressing Camera Sensors Faults in Vision-Based Navigation: Simulation and Dataset Development 

**Title (ZH)**: 基于视觉的导航中相机传感器故障的处理：仿真与数据集开发 

**Authors**: Riccardo Gallon, Fabian Schiemenz, Alessandra Menicucci, Eberhard Gill  

**Link**: [PDF](https://arxiv.org/pdf/2507.02602)  

**Abstract**: The increasing importance of Vision-Based Navigation (VBN) algorithms in space missions raises numerous challenges in ensuring their reliability and operational robustness. Sensor faults can lead to inaccurate outputs from navigation algorithms or even complete data processing faults, potentially compromising mission objectives. Artificial Intelligence (AI) offers a powerful solution for detecting such faults, overcoming many of the limitations associated with traditional fault detection methods. However, the primary obstacle to the adoption of AI in this context is the lack of sufficient and representative datasets containing faulty image data.
This study addresses these challenges by focusing on an interplanetary exploration mission scenario. A comprehensive analysis of potential fault cases in camera sensors used within the VBN pipeline is presented. The causes and effects of these faults are systematically characterized, including their impact on image quality and navigation algorithm performance, as well as commonly employed mitigation strategies. To support this analysis, a simulation framework is introduced to recreate faulty conditions in synthetically generated images, enabling a systematic and controlled reproduction of faulty data. The resulting dataset of fault-injected images provides a valuable tool for training and testing AI-based fault detection algorithms. The final link to the dataset will be added after an embargo period. For peer-reviewers, this private link is available. 

**Abstract (ZH)**: 基于视觉的导航算法在空间任务中的重要性不断增加，确保其可靠性和操作鲁棒性提出了众多挑战。传感器故障可能导致导航算法输出不准确或将导致完全的数据处理故障，从而可能影响任务目标。人工智能提供了检测此类故障的强大解决方案，克服了许多传统故障检测方法的局限性。然而，阻碍在此背景下采用人工智能的主要障碍是缺乏包含故障图像数据的充分且具有代表性的数据集。

本文通过关注一次行星际探索任务场景来应对这些挑战。详细分析了VBN管道中使用的相机传感器可能出现的故障情况，并系统地 Characterized 这些故障的原因和影响，包括其对图像质量和导航算法性能的影响，以及常用的缓解策略。为支持这一分析，引入了一种仿真框架，用于在合成生成的图像中再现故障条件，从而能够在系统且可控的方式下重现故障数据。生成的包含故障注入的图像数据集提供了一个有价值的工具，用于训练和测试基于人工智能的故障检测算法。在预印本禁令期结束后，将提供最终的数据集链接。审稿人可以通过私人链接访问此数据集。 

---
# AC-Refiner: Efficient Arithmetic Circuit Optimization Using Conditional Diffusion Models 

**Title (ZH)**: AC-精炼器：基于条件扩散模型的高效算术电路优化 

**Authors**: Chenhao Xue, Kezhi Li, Jiaxing Zhang, Yi Ren, Zhengyuan Shi, Chen Zhang, Yibo Lin, Lining Zhang, Qiang Xu, Guangyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.02598)  

**Abstract**: Arithmetic circuits, such as adders and multipliers, are fundamental components of digital systems, directly impacting the performance, power efficiency, and area footprint. However, optimizing these circuits remains challenging due to the vast design space and complex physical constraints. While recent deep learning-based approaches have shown promise, they struggle to consistently explore high-potential design variants, limiting their optimization efficiency. To address this challenge, we propose AC-Refiner, a novel arithmetic circuit optimization framework leveraging conditional diffusion models. Our key insight is to reframe arithmetic circuit synthesis as a conditional image generation task. By carefully conditioning the denoising diffusion process on target quality-of-results (QoRs), AC-Refiner consistently produces high-quality circuit designs. Furthermore, the explored designs are used to fine-tune the diffusion model, which focuses the exploration near the Pareto frontier. Experimental results demonstrate that AC-Refiner generates designs with superior Pareto optimality, outperforming state-of-the-art baselines. The performance gain is further validated by integrating AC-Refiner into practical applications. 

**Abstract (ZH)**: 算术电路（如加法器和乘法器）是数字系统的基本组件，直接影响性能、功耗效率和面积占用。然而，由于设计空间庞大和复杂的物理约束，优化这些电路仍然具有挑战性。虽然基于深度学习的方法显示出潜力，但它们在一致探索高潜力设计变体方面存在困难，限制了其优化效率。为应对这一挑战，我们提出了一种名为AC-Refiner的新型算术电路优化框架，利用条件扩散模型。我们的核心洞察是将算术电路合成重新定义为一个条件图像生成任务。通过仔细调整去噪扩散过程以适应目标质量成果（QoRs），AC-Refiner能够一致地生成高质量的电路设计。此外，探索出的设计被用于微调扩散模型，从而将探索聚焦在帕累托前沿附近。实验结果表明，AC-Refiner生成的设计具有更优的帕累托最优性，优于现有的最先进的基线。性能的提升进一步通过将其整合到实际应用中得到验证。 

---
# MPF: Aligning and Debiasing Language Models post Deployment via Multi Perspective Fusion 

**Title (ZH)**: MPF: 部署后通过多视角融合对齐和祛偏语言模型 

**Authors**: Xin Guan, PeiHsin Lin, Zekun Wu, Ze Wang, Ruibo Zhang, Emre Kazim, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2507.02595)  

**Abstract**: Multiperspective Fusion (MPF) is a novel posttraining alignment framework for large language models (LLMs) developed in response to the growing need for easy bias mitigation. Built on top of the SAGED pipeline, an automated system for constructing bias benchmarks and extracting interpretable baseline distributions, MPF leverages multiperspective generations to expose and align biases in LLM outputs with nuanced, humanlike baselines. By decomposing baseline, such as sentiment distributions from HR professionals, into interpretable perspective components, MPF guides generation through sampling and balancing of responses, weighted by the probabilities obtained in the decomposition. Empirically, we demonstrate its ability to align LLM sentiment distributions with both counterfactual baselines (absolute equality) and the HR baseline (biased for Top Univeristy), resulting in small KL divergence, reduction of calibration error and generalization to unseen questions. This shows that MPF offers a scalable and interpretable method for alignment and bias mitigation, compatible with deployed LLMs and requiring no extensive prompt engineering or finetuning. 

**Abstract (ZH)**: 多视角融合（MPF）是针对大型语言模型（LLMs）日益增长的去偏需求开发的一种新型后训练对齐框架。 

---
# WebSailor: Navigating Super-human Reasoning for Web Agent 

**Title (ZH)**: WebSailor: 导航超人类推理的网络代理 

**Authors**: Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02592)  

**Abstract**: Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all opensource agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap. 

**Abstract (ZH)**: 超越人类认知限制是LLM训练的一个关键前沿领域。像DeepResearch这样的专有自主系统已在诸如BrowseComp等极其复杂的信息检索基准上展示了超人类的能力，这是前所未有的。我们认为其成功在于一种在开源模型中缺失的复杂的推理模式：能够在 vast 信息景观中系统地降低极端不确定性。基于这一见解，我们提出了WebSailor，一个完整的后训练方法论，旨在赋予这种关键能力。我们的方法包括通过结构化采样和信息混淆生成新颖的高不确定性任务、RFT冷启动以及高效的自主RL训练算法DUPO（Duplicating Sampling Policy Optimization）。通过这一整合的工作流程，WebSailor在复杂的信息检索任务中显著优于所有开源代理，并与专有代理的性能相当，缩小了能力差距。 

---
# Position: A Theory of Deep Learning Must Include Compositional Sparsity 

**Title (ZH)**: 位置：一个深度学习理论必须包括组合稀疏性 

**Authors**: David A. Danhofer, Davide D'Ascenzo, Rafael Dubach, Tomaso Poggio  

**Link**: [PDF](https://arxiv.org/pdf/2507.02550)  

**Abstract**: Overparametrized Deep Neural Networks (DNNs) have demonstrated remarkable success in a wide variety of domains too high-dimensional for classical shallow networks subject to the curse of dimensionality. However, open questions about fundamental principles, that govern the learning dynamics of DNNs, remain. In this position paper we argue that it is the ability of DNNs to exploit the compositionally sparse structure of the target function driving their success. As such, DNNs can leverage the property that most practically relevant functions can be composed from a small set of constituent functions, each of which relies only on a low-dimensional subset of all inputs. We show that this property is shared by all efficiently Turing-computable functions and is therefore highly likely present in all current learning problems. While some promising theoretical insights on questions concerned with approximation and generalization exist in the setting of compositionally sparse functions, several important questions on the learnability and optimization of DNNs remain. Completing the picture of the role of compositional sparsity in deep learning is essential to a comprehensive theory of artificial, and even general, intelligence. 

**Abstract (ZH)**: 过参数化的深度神经网络(DNNs)在高维领域展示了 remarkable 成功，这些领域对经典的受限浅层网络来说由于维数灾提供了巨大挑战。然而，关于治理 DNNs 学习动力学的基本原理，仍有许多待解答的问题。在本文中，我们认为 DNNs 的成功得益于其利用目标函数的组合稀疏结构的能力。因此，DNNs 可以利用这样一个性质：大多数实际相关函数都能够由少量的组成函数通过组合产生，每个组成函数只依赖于所有输入中的低维子集。我们表明，这一性质适用于所有高效图灵可计算函数，因此很可能是当今所有学习问题中的常见特征。虽然在组合稀疏函数的背景下关于逼近和泛化的几个理论上有趣的洞察已经存在，但关于 DNNs 可学习性和优化方面的一些重要问题仍然没有解决。完整地描绘组合稀疏性在深度学习中的作用对于全面的人工智能理论，甚至通用人工智能理论至关重要。 

---
# Are You Listening to Me? Fine-Tuning Chatbots for Empathetic Dialogue 

**Title (ZH)**: 你在倾听我吗？细调聊天机器人以实现共情对话 

**Authors**: Paulo Ricardo Knob, Leonardo Scholler, Juliano Rigatti, Soraia Raupp Musse  

**Link**: [PDF](https://arxiv.org/pdf/2507.02537)  

**Abstract**: Conversational agents have made significant progress since ELIZA, expanding their role across various domains, including healthcare, education, and customer service. As these agents become increasingly integrated into daily human interactions, the need for emotional intelligence, particularly empathetic listening, becomes increasingly essential. In this study, we explore how Large Language Models (LLMs) respond when tasked with generating emotionally rich interactions. Starting from a small dataset manually crafted by an expert to reflect empathic behavior, we extended the conversations using two LLMs: ChatGPT and Gemini. We analyzed the emotional progression of the dialogues using both sentiment analysis (via VADER) and expert assessments. While the generated conversations often mirrored the intended emotional structure, human evaluation revealed important differences in the perceived empathy and coherence of the responses. These findings suggest that emotion modeling in dialogues requires not only structural alignment in the expressed emotions but also qualitative depth, highlighting the importance of combining automated and humancentered methods in the development of emotionally competent agents. 

**Abstract (ZH)**: 自ELIZA以来，对话代理在各个领域，包括医疗、教育和客户服务中取得了显著进展。随着这些代理越来越融入日常的人际互动中，特别是在情绪智能，尤其是同理心倾听方面的需求变得越来越重要。在本研究中，我们探讨了大规模语言模型（LLMs）在生成富有情感的互动时的反应。我们从一个由专家手工制作的小数据集扩展对话，使用了两种LLM：ChatGPT和Gemini。我们使用情感分析（通过VADER）和专家评估分析了对话的情绪进展。虽然生成的对话通常反映了预期的情感结构，但人类评估揭示了在感知同理心和响应连贯性方面的重要差异。这些发现表明，对话中的情绪建模不仅需要情感表达的结构对齐，还需要定量深度，强调了在开发情绪胜任代理过程中结合自动化和以人为本方法的重要性。 

---
# Detecting Multiple Diseases in Multiple Crops Using Deep Learning 

**Title (ZH)**: 多作物多种疾病的深度学习检测方法 

**Authors**: Vivek Yadav, Anugrah Jain  

**Link**: [PDF](https://arxiv.org/pdf/2507.02517)  

**Abstract**: India, as a predominantly agrarian economy, faces significant challenges in agriculture, including substantial crop losses caused by diseases, pests, and environmental stress. Early detection and accurate identification of diseases across different crops are critical for improving yield and ensuring food security. This paper proposes a deep learning based solution for detecting multiple diseases in multiple crops, aimed to cover India's diverse agricultural landscape. We first create a unified dataset encompassing images of 17 different crops and 34 different diseases from various available repositories. Proposed deep learning model is trained on this dataset and outperforms the state-of-the-art in terms of accuracy and the number of crops, diseases covered. We achieve a significant detection accuracy, i.e., 99 percent for our unified dataset which is 7 percent more when compared to state-of-the-art handling 14 crops and 26 different diseases only. By improving the number of crops and types of diseases that can be detected, proposed solution aims to provide a better product for Indian farmers. 

**Abstract (ZH)**: 印度作为一个以农业为主的经济体，面临着农业方面的重大挑战，包括由疾病、害虫和环境压力导致的作物大量损失。早发现和准确识别不同作物的病害对于提高产量和确保粮食安全至关重要。本文提出了一种基于深度学习的解决方案，用于检测多种作物的多种病害，旨在覆盖印度多样的农业景观。我们首先创建了一个统一的数据集，包含来自多个可用仓库的17种不同作物和34种不同病害的图像。所提出的深度学习模型在此数据集上进行了训练，并在准确性和覆盖的作物和疾病种类数量方面优于现有最佳方案。我们实现了显著的检测准确率，即统一数据集上的99%，比仅处理14种作物和26种不同病害的现有最佳方案高出7%。通过增加可检测的作物种类和病害类型数量，所提出的解决方案旨在为印度农民提供更好的产品。 

---
# IndianBailJudgments-1200: A Multi-Attribute Dataset for Legal NLP on Indian Bail Orders 

**Title (ZH)**: 印度保释判决数据集-1200：用于印度保释命令的法律自然语言处理的多属性数据集 

**Authors**: Sneha Deshmukh, Prathmesh Kamble  

**Link**: [PDF](https://arxiv.org/pdf/2507.02506)  

**Abstract**: Legal NLP remains underdeveloped in regions like India due to the scarcity of structured datasets. We introduce IndianBailJudgments-1200, a new benchmark dataset comprising 1200 Indian court judgments on bail decisions, annotated across 20+ attributes including bail outcome, IPC sections, crime type, and legal reasoning. Annotations were generated using a prompt-engineered GPT-4o pipeline and verified for consistency. This resource supports a wide range of legal NLP tasks such as outcome prediction, summarization, and fairness analysis, and is the first publicly available dataset focused specifically on Indian bail jurisprudence. 

**Abstract (ZH)**: 印度保释判决-1200：一种新的标注数据集及其在法律NLP任务中的应用 

---
# Continual Gradient Low-Rank Projection Fine-Tuning for LLMs 

**Title (ZH)**: 持续梯度低秩投影微调用于大语言模型 

**Authors**: Chenxu Wang, Yilin Lyu, Zicheng Sun, Liping Jing  

**Link**: [PDF](https://arxiv.org/pdf/2507.02503)  

**Abstract**: Continual fine-tuning of Large Language Models (LLMs) is hampered by the trade-off between efficiency and expressiveness. Low-Rank Adaptation (LoRA) offers efficiency but constrains the model's ability to learn new tasks and transfer knowledge due to its low-rank nature and reliance on explicit parameter constraints. We propose GORP (Gradient LOw Rank Projection) for Continual Learning, a novel training strategy that overcomes these limitations by synergistically combining full and low-rank parameters and jointly updating within a unified low-rank gradient subspace. GORP expands the optimization space while preserving efficiency and mitigating catastrophic forgetting. Extensive experiments on continual learning benchmarks demonstrate GORP's superior performance compared to existing state-of-the-art approaches. Code is available at this https URL. 

**Abstract (ZH)**: 连续微调大型语言模型（LLMs）受到效率与表达力之间的权衡限制。低秩适应（LoRA）虽然提升了效率，但由于其低秩性质和对显式参数约束的依赖，限制了模型学习新任务和知识迁移的能力。我们提出了一个新颖的训练策略GORP（Gradient LOw Rank Projection）用于连续学习，该策略通过将全参数和低秩参数协同结合并在统一的低秩梯度子空间内联合更新来克服这些限制。GORP扩展了优化空间，同时保持效率并减轻灾难性遗忘。广泛的经验研究表明，GORP在连续学习基准测试中优于现有最佳方法。代码可在以下网址获取。 

---
# Temporally-Aware Supervised Contrastive Learning for Polyp Counting in Colonoscopy 

**Title (ZH)**: 基于时间感知的监督对比学习在结肠镜检查中息肉计数 

**Authors**: Luca Parolari, Andrea Cherubini, Lamberto Ballan, Carlo Biffi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02493)  

**Abstract**: Automated polyp counting in colonoscopy is a crucial step toward automated procedure reporting and quality control, aiming to enhance the cost-effectiveness of colonoscopy screening. Counting polyps in a procedure involves detecting and tracking polyps, and then clustering tracklets that belong to the same polyp entity. Existing methods for polyp counting rely on self-supervised learning and primarily leverage visual appearance, neglecting temporal relationships in both tracklet feature learning and clustering stages. In this work, we introduce a paradigm shift by proposing a supervised contrastive loss that incorporates temporally-aware soft targets. Our approach captures intra-polyp variability while preserving inter-polyp discriminability, leading to more robust clustering. Additionally, we improve tracklet clustering by integrating a temporal adjacency constraint, reducing false positive re-associations between visually similar but temporally distant tracklets. We train and validate our method on publicly available datasets and evaluate its performance with a leave-one-out cross-validation strategy. Results demonstrate a 2.2x reduction in fragmentation rate compared to prior approaches. Our results highlight the importance of temporal awareness in polyp counting, establishing a new state-of-the-art. Code is available at this https URL. 

**Abstract (ZH)**: 自动结肠镜检查中的息肉计数是实现自动检查程序报告和质量控制的关键步骤，旨在提高结肠镜筛查的成本效益。我们通过引入基于监督对比损失的方法，结合时空信息，提出了一种新的息肉计数框架，显著提高了息肉计数的准确性和鲁棒性。 

---
# CrowdTrack: A Benchmark for Difficult Multiple Pedestrian Tracking in Real Scenarios 

**Title (ZH)**: CrowdTrack: 一个适用于真实场景中困难多行人跟踪的基准 

**Authors**: Teng Fu, Yuwen Chen, Zhuofan Chen, Mengyang Zhao, Bin Li, Xiangyang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.02479)  

**Abstract**: Multi-object tracking is a classic field in computer vision. Among them, pedestrian tracking has extremely high application value and has become the most popular research category. Existing methods mainly use motion or appearance information for tracking, which is often difficult in complex scenarios. For the motion information, mutual occlusions between objects often prevent updating of the motion state; for the appearance information, non-robust results are often obtained due to reasons such as only partial visibility of the object or blurred images. Although learning how to perform tracking in these situations from the annotated data is the simplest solution, the existing MOT dataset fails to satisfy this solution. Existing methods mainly have two drawbacks: relatively simple scene composition and non-realistic scenarios. Although some of the video sequences in existing dataset do not have the above-mentioned drawbacks, the number is far from adequate for research purposes. To this end, we propose a difficult large-scale dataset for multi-pedestrian tracking, shot mainly from the first-person view and all from real-life complex scenarios. We name it ``CrowdTrack'' because there are numerous objects in most of the sequences. Our dataset consists of 33 videos, containing a total of 5,185 trajectories. Each object is annotated with a complete bounding box and a unique object ID. The dataset will provide a platform to facilitate the development of algorithms that remain effective in complex situations. We analyzed the dataset comprehensively and tested multiple SOTA models on our dataset. Besides, we analyzed the performance of the foundation models on our dataset. The dataset and project code is released at: this https URL . 

**Abstract (ZH)**: 多目标跟踪是计算机视觉中的一个经典领域。其中，行人跟踪具有极高的应用价值，已经成为最热门的研究领域。现有方法主要利用运动或外观信息进行跟踪，在复杂场景中常常难以实现。对于运动信息，对象间的相互遮挡经常阻碍运动状态的更新；对于外观信息，由于只有一部分可见或图像模糊等原因，常常得到不 robust 的结果。尽管从标注数据中学习如何在这些场景下进行跟踪是最简单的解决方案，但现有多目标跟踪（MOT）数据集未能满足这一需求。现有方法主要存在两个缺点：相对简单的场景组成和非现实的场景。尽管现有数据集中的一些视频序列并未存在上述缺点，但数量远远不足以满足研究需求。为此，我们提出了一大规模的复杂场景下多行人跟踪数据集，主要从第一人称视角拍摄，并全部来自真实的复杂场景。我们将其命名为“CrowdTrack”因为几乎所有序列中都有大量对象。该数据集包含33个视频，共5,185条轨迹。每个对象都标注了完整的边界框和唯一的对象ID。该数据集将为在复杂场景下仍能有效工作的算法开发提供平台。我们对数据集进行了全面分析，并在我们的数据集上测试了多个最新模型的性能。此外，我们还分析了基础模型在我们数据集上的性能。该数据集和项目代码已发布在: [这个链接](this https URL) 。 

---
# Red grape detection with accelerated artificial neural networks in the FPGA's programmable logic 

**Title (ZH)**: 基于FPGA可编程逻辑的加速人工神经网络红葡萄检测 

**Authors**: Sandro Costa Magalhães, Marco Almeida, Filipe Neves dos Santos, António Paulo Moreira, Jorge Dias  

**Link**: [PDF](https://arxiv.org/pdf/2507.02443)  

**Abstract**: Robots usually slow down for canning to detect objects while moving. Additionally, the robot's camera is configured with a low framerate to track the velocity of the detection algorithms. This would be constrained while executing tasks and exploring, making robots increase the task execution time. AMD has developed the Vitis-AI framework to deploy detection algorithms into FPGAs. However, this tool does not fully use the FPGAs' PL. In this work, we use the FINN architecture to deploy three ANNs, MobileNet v1 with 4-bit quantisation, CNV with 2-bit quantisation, and CNV with 1-bit quantisation (BNN), inside an FPGA's PL. The models were trained on the RG2C dataset. This is a self-acquired dataset released in open access. MobileNet v1 performed better, reaching a success rate of 98 % and an inference speed of 6611 FPS. In this work, we proved that we can use FPGAs to speed up ANNs and make them suitable for attention mechanisms. 

**Abstract (ZH)**: 机器人通常在搬运过程中减慢速度以检测物体，同时，机器人的相机配置为低帧率以跟踪检测算法的velocity。这在执行任务和探索时会受限，使机器人增加任务执行时间。AMD开发了Vitis-AI框架以将检测算法部署到FPGAs中，但该工具并未充分利用FPGAs的PL。在本文中，我们使用FINN架构将三个ANN，MobileNet v1（4位量化）、CNV（2位量化）和CNV（1位量化，BNN）部署到FPGA的PL中。这些模型在RG2C数据集上进行了训练。该数据集是自获取的数据集，现已开放获取。MobileNet v1表现最佳，成功率达到98%，推理速度为6611 FPS。在本文中，我们证明了可以使用FPGAs加速ANN并使其适合注意力机制。 

---
# Toward a Robust and Generalizable Metamaterial Foundation Model 

**Title (ZH)**: 向着稳健且通用的元材料基础模型 

**Authors**: Namjung Kim, Dongseok Lee, Jongbin Yu, Sung Woong Cho, Dosung Lee, Yesol Park, Youngjoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02436)  

**Abstract**: Advances in material functionalities drive innovations across various fields, where metamaterials-defined by structure rather than composition-are leading the way. Despite the rise of artificial intelligence (AI)-driven design strategies, their impact is limited by task-specific retraining, poor out-of-distribution(OOD) generalization, and the need for separate models for forward and inverse design. To address these limitations, we introduce the Metamaterial Foundation Model (MetaFO), a Bayesian transformer-based foundation model inspired by large language models. MetaFO learns the underlying mechanics of metamaterials, enabling probabilistic, zero-shot predictions across diverse, unseen combinations of material properties and structural responses. It also excels in nonlinear inverse design, even under OOD conditions. By treating metamaterials as an operator that maps material properties to structural responses, MetaFO uncovers intricate structure-property relationships and significantly expands the design space. This scalable and generalizable framework marks a paradigm shift in AI-driven metamaterial discovery, paving the way for next-generation innovations. 

**Abstract (ZH)**: 材料功能性的进步推动了各个领域的创新，其中通过结构而非组成定义的 metamaterials 处于领先地位。尽管人工智能（AI）驱动的设计策略正在崛起，但它们受到特定任务重训的限制，具有较差的离分布外 (OOD) 通用性，并且需要分别用于正向和逆向设计的单独模型。为了克服这些限制，我们 introduces Metamaterial 基础模型 (MetaFO)，这是一种受大型语言模型启发的贝叶斯变压器基础模型。MetaFO 学习 metamaterials 的内在力学，从而能够跨多种未见过的材料属性和结构响应组合进行概率性的零样本预测。它在非线性逆向设计中也表现出色，即使在 OOD 条件下也是如此。通过将 metamaterials 视为一个操作符，将材料属性映射到结构响应，MetaFO 揭示了复杂的结构-性能关系，并显着扩展了设计空间。这一可扩展且通用的框架标志着人工智能驱动 metamaterial 发现范式的转变，铺平了下一代创新的道路。 

---
# CyberRAG: An agentic RAG cyber attack classification and reporting tool 

**Title (ZH)**: CyberRAG: 一个自主的RAG网络攻击分类和报告工具 

**Authors**: Francesco Blefari, Cristian Cosentino, Francesco Aurelio Pironti, Angelo Furfaro, Fabrizio Marozzo  

**Link**: [PDF](https://arxiv.org/pdf/2507.02424)  

**Abstract**: Intrusion Detection and Prevention Systems (IDS/IPS) in large enterprises can generate hundreds of thousands of alerts per hour, overwhelming security analysts with logs that demand deep, rapidly evolving domain expertise. Conventional machine-learning detectors trim the alert volume but still yield high false-positive rates, while standard single-pass Retrieval-Augmented Generation (RAG) pipelines often retrieve irrelevant context and fail to justify their predictions. To overcome these shortcomings, we present CyberRAG, a modular, agent-based RAG framework that delivers real-time classification, explanation, and structured reporting for cyber-attacks. A central LLM agent orchestrates (i) a pool of fine-tuned specialized classifiers, each tailored to a distinct attack family; (ii) tool adapters for enrichment and alerting; and (iii) an iterative retrieval-and-reason loop that continuously queries a domain-specific knowledge base until the evidence is both relevant and self-consistent. Unlike traditional RAG systems, CyberRAG embraces an agentic design that enables dynamic control flow and adaptive reasoning. This agent-centric architecture refines its threat labels and natural-language justifications autonomously, reducing false positives and enhancing interpretability. The framework is fully extensible: new attack types can be supported by simply adding a classifier without retraining the core agent. CyberRAG has been evaluated achieving over 94% accuracy per class and pushing final classification accuracy to 94.92% through semantic orchestration. Generated explanations score up to 0.94 in BERTScore and 4.9/5 in GPT-4-based expert evaluation. These results show that agentic, specialist-oriented RAG can pair high detection accuracy with trustworthy, SOC-ready prose, offering a practical and scalable path toward semi-autonomous cyber-defence workflows. 

**Abstract (ZH)**: 基于代理的 CyberRAG：实时分类、解释和结构化报告的入侵检测与预防系统 

---
# S2FGL: Spatial Spectral Federated Graph Learning 

**Title (ZH)**: S2FGL：空间频谱联邦图学习 

**Authors**: Zihan Tan, Suyuan Huang, Guancheng Wan, Wenke Huang, He Li, Mang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.02409)  

**Abstract**: Federated Graph Learning (FGL) combines the privacy-preserving capabilities of federated learning (FL) with the strong graph modeling capability of Graph Neural Networks (GNNs). Current research addresses subgraph-FL only from the structural perspective, neglecting the propagation of graph signals on spatial and spectral domains of the structure. From a spatial perspective, subgraph-FL introduces edge disconnections between clients, leading to disruptions in label signals and a degradation in the class knowledge of the global GNN. From a spectral perspective, spectral heterogeneity causes inconsistencies in signal frequencies across subgraphs, which makes local GNNs overfit the local signal propagation schemes. As a result, spectral client drifts occur, undermining global generalizability. To tackle the challenges, we propose a global knowledge repository to mitigate label signal disruption and a frequency alignment to address spectral client drifts. The combination of spatial and spectral strategies forms our framework S2FGL. Extensive experiments on multiple datasets demonstrate the superiority of S2FGL. The code is available at this https URL. 

**Abstract (ZH)**: 联邦图学习（FGL）结合了联邦学习（FL）的隐私保护能力与图 Neural Networks（GNNs）的强图建模能力。当前研究仅从结构视角探讨了子图-FL，忽略了结构的空间域和频域中图信号的传播。从空间视角来看，子图-FL引入了客户端之间的边断开，导致标签信号中断并降低了全局GNN的类别知识。从频域视角来看，频域异质性导致子图中信号频率的一致性问题，使得局部GNN过度拟合局部信号传播方案。因此，频域客户端漂移发生，损害了全局泛化能力。为应对挑战，我们提出了一种全局知识库以减轻标签信号中断，并提出频率对齐以解决频域客户端漂移问题。结合空间和频域策略形成了我们的框架S2FGL。在多个数据集上的广泛实验展示了S2FGL的优势。代码可在如下链接获取。 

---
# Wildlife Target Re-Identification Using Self-supervised Learning in Non-Urban Settings 

**Title (ZH)**: 非城区环境下基于自监督学习的野生动物目标重识别 

**Authors**: Mufhumudzi Muthivhi, Terence L. van Zyl  

**Link**: [PDF](https://arxiv.org/pdf/2507.02403)  

**Abstract**: Wildlife re-identification aims to match individuals of the same species across different observations. Current state-of-the-art (SOTA) models rely on class labels to train supervised models for individual classification. This dependence on annotated data has driven the curation of numerous large-scale wildlife datasets. This study investigates self-supervised learning Self-Supervised Learning (SSL) for wildlife re-identification. We automatically extract two distinct views of an individual using temporal image pairs from camera trap data without supervision. The image pairs train a self-supervised model from a potentially endless stream of video data. We evaluate the learnt representations against supervised features on open-world scenarios and transfer learning in various wildlife downstream tasks. The analysis of the experimental results shows that self-supervised models are more robust even with limited data. Moreover, self-supervised features outperform supervision across all downstream tasks. The code is available here this https URL. 

**Abstract (ZH)**: 野生动物重识别旨在跨不同观察记录匹配同一物种的个体。当前最先进的（SOTA）模型依赖类别标签来训练监督模型进行个体分类。这种对标注数据的依赖促使构建了大量大规模的野生动物数据集。本研究探讨了在野生动物重识别中使用自我监督学习（Self-Supervised Learning，SSL）的方法。我们使用相机陷阱数据中的时间图像对自动提取个体的两种不同视图，无需监督即可训练自我监督模型，并从潜在无尽的视频流数据中获取训练。我们在开放世界场景和各种野生动物下游任务中评估学习表示与监督特征的表现。实验结果分析表明，即使数据有限，自我监督模型也更具鲁棒性。此外，自我监督特征在所有下游任务中均优于监督特征。代码可在这里获取：https://。 

---
# Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Deepfake Video Detection 

**Title (ZH)**: 超越空间频率：基于像素级时间频率的深fake视频检测 

**Authors**: Taehoon Kim, Jongwook Choi, Yonghyun Jeong, Haeun Noh, Jaejun Yoo, Seungryul Baek, Jongwon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02398)  

**Abstract**: We introduce a deepfake video detection approach that exploits pixel-wise temporal inconsistencies, which traditional spatial frequency-based detectors often overlook. Traditional detectors represent temporal information merely by stacking spatial frequency spectra across frames, resulting in the failure to detect temporal artifacts in the pixel plane. Our approach performs a 1D Fourier transform on the time axis for each pixel, extracting features highly sensitive to temporal inconsistencies, especially in areas prone to unnatural movements. To precisely locate regions containing the temporal artifacts, we introduce an attention proposal module trained in an end-to-end manner. Additionally, our joint transformer module effectively integrates pixel-wise temporal frequency features with spatio-temporal context features, expanding the range of detectable forgery artifacts. Our framework represents a significant advancement in deepfake video detection, providing robust performance across diverse and challenging detection scenarios. 

**Abstract (ZH)**: 基于像素级时间不一致性的深伪视频检测方法：超越传统基于空间频率的检测器 

---
# Evaluating Language Models For Threat Detection in IoT Security Logs 

**Title (ZH)**: 评估物联网安全日志中威胁检测的语言模型 

**Authors**: Jorge J. Tejero-Fernández, Alfonso Sánchez-Macián  

**Link**: [PDF](https://arxiv.org/pdf/2507.02390)  

**Abstract**: Log analysis is a relevant research field in cybersecurity as they can provide a source of information for the detection of threats to networks and systems. This paper presents a pipeline to use fine-tuned Large Language Models (LLMs) for anomaly detection and mitigation recommendation using IoT security logs. Utilizing classical machine learning classifiers as a baseline, three open-source LLMs are compared for binary and multiclass anomaly detection, with three strategies: zero-shot, few-shot prompting and fine-tuning using an IoT dataset. LLMs give better results on multi-class attack classification than the corresponding baseline models. By mapping detected threats to MITRE CAPEC, defining a set of IoT-specific mitigation actions, and fine-tuning the models with those actions, the models are able to provide a combined detection and recommendation guidance. 

**Abstract (ZH)**: 日志分析是网络安全研究的 Relevant Research Field，因为它们可以提供用于检测网络和系统威胁的信息。本文提出了一种Pipeline，使用微调的大语言模型（LLMs）进行异常检测和缓解建议，利用物联网安全日志。通过使用经典的机器学习分类器作为基线，本文对三种开源大语言模型在二元和多分类异常检测中的性能进行了比较，使用了三种策略：零样本、少样本提示和基于物联网数据集的微调。大语言模型在多类攻击分类中比相应的基线模型表现出更好的性能。通过将检测到的威胁映射到MITRE CAPEC，定义一组物联网特定的缓解操作，并使用这些操作微调模型，模型能够提供结合检测和建议指导。 

---
# VeFIA: An Efficient Inference Auditing Framework for Vertical Federated Collaborative Software 

**Title (ZH)**: VeFIA：一种高效的垂直联邦协作软件推断审计框架 

**Authors**: Chung-ju Huang, Ziqi Zhang, Yinggui Wang, Binghui Wang, Tao Wei, Leye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02376)  

**Abstract**: Vertical Federated Learning (VFL) is a distributed AI software deployment mechanism for cross-silo collaboration without accessing participants' data. However, existing VFL work lacks a mechanism to audit the execution correctness of the inference software of the data party. To address this problem, we design a Vertical Federated Inference Auditing (VeFIA) framework. VeFIA helps the task party to audit whether the data party's inference software is executed as expected during large-scale inference without leaking the data privacy of the data party or introducing additional latency to the inference system. The core of VeFIA is that the task party can use the inference results from a framework with Trusted Execution Environments (TEE) and the coordinator to validate the correctness of the data party's computation results. VeFIA guarantees that, as long as the abnormal inference exceeds 5.4%, the task party can detect execution anomalies in the inference software with a probability of 99.99%, without incurring any additional online inference latency. VeFIA's random sampling validation achieves 100% positive predictive value, negative predictive value, and true positive rate in detecting abnormal inference. To the best of our knowledge, this is the first paper to discuss the correctness of inference software execution in VFL. 

**Abstract (ZH)**: 垂直联邦推理审计（VeFIA）框架 

---
# Holistic Tokenizer for Autoregressive Image Generation 

**Title (ZH)**: 全面的分词器用于自回归图像生成 

**Authors**: Anlin Zheng, Haochen Wang, Yucheng Zhao, Weipeng Deng, Tiancai Wang, Xiangyu Zhang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02358)  

**Abstract**: The vanilla autoregressive image generation model generates visual tokens in a step-by-step fashion, which limits the ability to capture holistic relationships among token sequences. Moreover, most visual tokenizers map local image patches into latent tokens, leading to limited global information. To address this, we introduce \textit{Hita}, a novel image tokenizer for autoregressive (AR) image generation. It introduces a holistic-to-local tokenization scheme with learnable holistic queries and local patch tokens. Besides, Hita incorporates two key strategies for improved alignment with the AR generation process: 1) it arranges a sequential structure with holistic tokens at the beginning followed by patch-level tokens while using causal attention to maintain awareness of previous tokens; and 2) before feeding the de-quantized tokens into the decoder, Hita adopts a lightweight fusion module to control information flow to prioritize holistic tokens. Extensive experiments show that Hita accelerates the training speed of AR generators and outperforms those trained with vanilla tokenizers, achieving \textbf{2.59 FID} and \textbf{281.9 IS} on the ImageNet benchmark. A detailed analysis of the holistic representation highlights its ability to capture global image properties such as textures, materials, and shapes. Additionally, Hita also demonstrates effectiveness in zero-shot style transfer and image in-painting. The code is available at \href{this https URL}{this https URL} 

**Abstract (ZH)**: 基于整体到局部的自回归图像生成模型Hita及其应用 

---
# Offline Reinforcement Learning with Penalized Action Noise Injection 

**Title (ZH)**: 带惩罚性动作噪声注入的离线强化学习 

**Authors**: JunHyeok Oh, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.02356)  

**Abstract**: Offline reinforcement learning (RL) optimizes a policy using only a fixed dataset, making it a practical approach in scenarios where interaction with the environment is costly. Due to this limitation, generalization ability is key to improving the performance of offline RL algorithms, as demonstrated by recent successes of offline RL with diffusion models. However, it remains questionable whether such diffusion models are necessary for highly performing offline RL algorithms, given their significant computational requirements during inference. In this paper, we propose Penalized Action Noise Injection (PANI), a method that simply enhances offline learning by utilizing noise-injected actions to cover the entire action space, while penalizing according to the amount of noise injected. This approach is inspired by how diffusion models have worked in offline RL algorithms. We provide a theoretical foundation for this method, showing that offline RL algorithms with such noise-injected actions solve a modified Markov Decision Process (MDP), which we call the noisy action MDP. PANI is compatible with a wide range of existing off-policy and offline RL algorithms, and despite its simplicity, it demonstrates significant performance improvements across various benchmarks. 

**Abstract (ZH)**: 带惩罚的动作噪声注入（PANI）：一种增强离线学习的方法 

---
# Two-Steps Neural Networks for an Automated Cerebrovascular Landmark Detection 

**Title (ZH)**: 两阶段神经网络用于自动化颅内血管标志点检测 

**Authors**: Rafic Nader, Vincent L'Allinec, Romain Bourcier, Florent Autrusseau  

**Link**: [PDF](https://arxiv.org/pdf/2507.02349)  

**Abstract**: Intracranial aneurysms (ICA) commonly occur in specific segments of the Circle of Willis (CoW), primarily, onto thirteen major arterial bifurcations. An accurate detection of these critical landmarks is necessary for a prompt and efficient diagnosis. We introduce a fully automated landmark detection approach for CoW bifurcations using a two-step neural networks process. Initially, an object detection network identifies regions of interest (ROIs) proximal to the landmark locations. Subsequently, a modified U-Net with deep supervision is exploited to accurately locate the bifurcations. This two-step method reduces various problems, such as the missed detections caused by two landmarks being close to each other and having similar visual characteristics, especially when processing the complete MRA Time-of-Flight (TOF). Additionally, it accounts for the anatomical variability of the CoW, which affects the number of detectable landmarks per scan. We assessed the effectiveness of our approach using two cerebral MRA datasets: our In-House dataset which had varying numbers of landmarks, and a public dataset with standardized landmark configuration. Our experimental results demonstrate that our method achieves the highest level of performance on a bifurcation detection task. 

**Abstract (ZH)**: Willis圆各分支关键解剖标志的自动检测方法：基于两步神经网络的过程 

---
# HelixDesign-Antibody: A Scalable Production-Grade Platform for Antibody Design Built on HelixFold3 

**Title (ZH)**: HelixDesign-抗体：基于HelixFold3构建的可扩展的生产级抗体设计平台 

**Authors**: Jie Gao, Jing Hu, Shanzhuo Zhang, Kunrui Zhu, Sheng Qian, Yueyang Huang, Xiaonan Zhang, Xiaomin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02345)  

**Abstract**: Antibody engineering is essential for developing therapeutics and advancing biomedical research. Traditional discovery methods often rely on time-consuming and resource-intensive experimental screening. To enhance and streamline this process, we introduce a production-grade, high-throughput platform built on HelixFold3, HelixDesign-Antibody, which utilizes the high-accuracy structure prediction model, HelixFold3. The platform facilitates the large-scale generation of antibody candidate sequences and evaluates their interaction with antigens. Integrated high-performance computing (HPC) support enables high-throughput screening, addressing challenges such as fragmented toolchains and high computational demands. Validation on multiple antigens showcases the platform's ability to generate diverse and high-quality antibodies, confirming a scaling law where exploring larger sequence spaces increases the likelihood of identifying optimal binders. This platform provides a seamless, accessible solution for large-scale antibody design and is available via the antibody design page of PaddleHelix platform. 

**Abstract (ZH)**: 抗体工程对于开发治疗药物和推动生物医学研究至关重要。传统发现方法往往依赖于耗时且资源密集型的实验筛选。为增强并简化这一过程，我们介绍了一个基于HelixFold3和HelixDesign-Antibody构建的生产级高通量平台。该平台利用高精度结构预测模型HelixFold3，促进大规模产生抗体候选序列并评估其与抗原的相互作用。集成高性能计算（HPC）支持实现高通量筛选，解决了工具链碎片化和高计算需求等挑战。多种抗原的验证展示了该平台生成多样且高质量抗体的能力，确认了探索更大序列空间能增加发现最佳结合子的几率。该平台提供了无缝且易于访问的大型抗体设计解决方案，并可通过PaddleHelix平台的抗体设计页面获取。 

---
# DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values 

**Title (ZH)**: DeltaSHAP: 用Shapley值解释在线患者监测中的预测演变 

**Authors**: Changhun Kim, Yechan Mun, Sangchul Hahn, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02342)  

**Abstract**: This study proposes DeltaSHAP, a novel explainable artificial intelligence (XAI) algorithm specifically designed for online patient monitoring systems. In clinical environments, discovering the causes driving patient risk evolution is critical for timely intervention, yet existing XAI methods fail to address the unique requirements of clinical time series explanation tasks. To this end, DeltaSHAP addresses three key clinical needs: explaining the changes in the consecutive predictions rather than isolated prediction scores, providing both magnitude and direction of feature attributions, and delivering these insights in real time. By adapting Shapley values to temporal settings, our approach accurately captures feature coalition effects. It further attributes prediction changes using only the actually observed feature combinations, making it efficient and practical for time-sensitive clinical applications. We also introduce new evaluation metrics to evaluate the faithfulness of the attributions for online time series, and demonstrate through experiments on online patient monitoring tasks that DeltaSHAP outperforms state-of-the-art XAI methods in both explanation quality as 62% and computational efficiency as 33% time reduction on the MIMIC-III decompensation benchmark. We release our code at this https URL. 

**Abstract (ZH)**: 本研究提出DeltaSHAP，一种专为在线患者监测系统设计的新颖可解释人工智能(XAI)算法。在临床环境中，及时发现驱动患者风险演变的原因对于及时干预至关重要，但现有的XAI方法未能解决临床时间序列解释任务的特殊需求。为此，DeltaSHAP解决了三个关键的临床需求：解释连续预测的变化而非孤立的预测评分，提供特征归因的幅度和方向，并在实时提供这些见解。通过将Shapley值适应到时间设置，我们的方法准确捕捉了特征合作效应。进一步地，仅使用实际观测到的特征组合来归因预测变化，使其对于时间敏感的临床应用来说既高效又实用。我们还引入新的评估指标来评估在线时间序列中归因的忠实性，并通过在在线患者监测任务上的实验表明，DeltaSHAP在解释质量和计算效率（减少33%的时间）上均优于最先进的XAI方法，在MIMIC-III去补偿基准上表现更佳。源代码发布于此https网址。 

---
# ClustOpt: A Clustering-based Approach for Representing and Visualizing the Search Dynamics of Numerical Metaheuristic Optimization Algorithms 

**Title (ZH)**: ClustOpt：基于聚类的数值元启发式优化算法搜索动态表示与可视化方法 

**Authors**: Gjorgjina Cenikj, Gašper Petelin, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2507.02337)  

**Abstract**: Understanding the behavior of numerical metaheuristic optimization algorithms is critical for advancing their development and application. Traditional visualization techniques, such as convergence plots, trajectory mapping, and fitness landscape analysis, often fall short in illustrating the structural dynamics of the search process, especially in high-dimensional or complex solution spaces. To address this, we propose a novel representation and visualization methodology that clusters solution candidates explored by the algorithm and tracks the evolution of cluster memberships across iterations, offering a dynamic and interpretable view of the search process. Additionally, we introduce two metrics - algorithm stability and algorithm similarity- to quantify the consistency of search trajectories across runs of an individual algorithm and the similarity between different algorithms, respectively. We apply this methodology to a set of ten numerical metaheuristic algorithms, revealing insights into their stability and comparative behaviors, thereby providing a deeper understanding of their search dynamics. 

**Abstract (ZH)**: 数值元启发式优化算法的行为理解对于推进其发展和应用至关重要。传统的可视化技术，如收敛图、轨迹映射和fitness景观分析，在展示搜索过程的结构性动态方面往往不足，尤其是在高维或复杂解空间中。为了解决这一问题，我们提出了一种新的表示和可视化方法，该方法聚类算法探索的解候选者，并在迭代过程中跟踪簇成员关系的变化，从而提供搜索过程的动态和可解释视图。此外，我们引入了两个度量标准——算法稳定性和算法相似性，以量化单个算法运行中搜索轨迹的一致性以及不同算法之间的相似性。我们将此方法应用于十个数值元启发式优化算法，揭示了它们的稳定性和相对行为，从而加深了对其搜索动态的理解。 

---
# Tracing the Interactions of Modular CMA-ES Configurations Across Problem Landscapes 

**Title (ZH)**: 模块化CMA-ES配置在问题landscape中的交互追踪 

**Authors**: Ana Nikolikj, Mario Andrés Muñoz, Eva Tuba, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2507.02331)  

**Abstract**: This paper leverages the recently introduced concept of algorithm footprints to investigate the interplay between algorithm configurations and problem characteristics. Performance footprints are calculated for six modular variants of the CMA-ES algorithm (modCMA), evaluated on 24 benchmark problems from the BBOB suite, across two-dimensional settings: 5-dimensional and 30-dimensional. These footprints provide insights into why different configurations of the same algorithm exhibit varying performance and identify the problem features influencing these outcomes. Our analysis uncovers shared behavioral patterns across configurations due to common interactions with problem properties, as well as distinct behaviors on the same problem driven by differing problem features. The results demonstrate the effectiveness of algorithm footprints in enhancing interpretability and guiding configuration choices. 

**Abstract (ZH)**: 本文利用 recently introduced 的算法足迹概念，探讨算法配置与问题特征之间的相互作用。计算了适用于 BBOB 套件的 24 个基准问题上的六种模块化 CMA-ES 算法（modCMA）变体的性能足迹，并在二维设置（5 维和 30 维）中进行了评估。这些足迹揭示了为何相同算法的不同配置表现出不同的性能，并识别了影响这些结果的问题特征。分析发现，由于共同的问题属性交互，不同配置之间存在共享的行为模式；同时，由于不同的问题特征，相同的问题上表现出不同的行为。结果表明，算法足迹在提高可解释性和指导配置选择方面非常有效。 

---
# Neural Network-based Study for Rice Leaf Disease Recognition and Classification: A Comparative Analysis Between Feature-based Model and Direct Imaging Model 

**Title (ZH)**: 基于神经网络的水稻叶片疾病识别与分类研究：特征模型与直接成像模型的比较分析 

**Authors**: Farida Siddiqi Prity, Mirza Raquib, Saydul Akbar Murad, Md. Jubayar Alam Rafi, Md. Khairul Bashar Bhuiyan, Anupam Kumar Bairagi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02322)  

**Abstract**: Rice leaf diseases significantly reduce productivity and cause economic losses, highlighting the need for early detection to enable effective management and improve yields. This study proposes Artificial Neural Network (ANN)-based image-processing techniques for timely classification and recognition of rice diseases. Despite the prevailing approach of directly inputting images of rice leaves into ANNs, there is a noticeable absence of thorough comparative analysis between the Feature Analysis Detection Model (FADM) and Direct Image-Centric Detection Model (DICDM), specifically when it comes to evaluating the effectiveness of Feature Extraction Algorithms (FEAs). Hence, this research presents initial experiments on the Feature Analysis Detection Model, utilizing various image Feature Extraction Algorithms, Dimensionality Reduction Algorithms (DRAs), Feature Selection Algorithms (FSAs), and Extreme Learning Machine (ELM). The experiments are carried out on datasets encompassing bacterial leaf blight, brown spot, leaf blast, leaf scald, Sheath blight rot, and healthy leaf, utilizing 10-fold Cross-Validation method. A Direct Image-Centric Detection Model is established without the utilization of any FEA, and the evaluation of classification performance relies on different metrics. Ultimately, an exhaustive contrast is performed between the achievements of the Feature Analysis Detection Model and Direct Image-Centric Detection Model in classifying rice leaf diseases. The results reveal that the highest performance is attained using the Feature Analysis Detection Model. The adoption of the proposed Feature Analysis Detection Model for detecting rice leaf diseases holds excellent potential for improving crop health, minimizing yield losses, and enhancing overall productivity and sustainability of rice farming. 

**Abstract (ZH)**: 基于人工神经网络的图像处理技术在水稻叶片疾病及时分类与识别中的应用：特征分析检测模型与直接图像中心检测模型的比较研究 

---
# MAGIC: Mask-Guided Diffusion Inpainting with Multi-Level Perturbations and Context-Aware Alignment for Few-Shot Anomaly Generation 

**Title (ZH)**: MAGIC：基于掩膜引导的多级扰动与上下文意识对齐扩散补全以实现少样本异常生成 

**Authors**: JaeHyuck Choi, MinJun Kim, JeHyeong Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02314)  

**Abstract**: Few-shot anomaly generation is emerging as a practical solution for augmenting the scarce anomaly data in industrial quality control settings. An ideal generator would meet three demands at once, namely (i) keep the normal background intact, (ii) inpaint anomalous regions to tightly overlap with the corresponding anomaly masks, and (iii) generate anomalous regions in a semantically valid location, while still producing realistic, diverse appearances from only a handful of real examples. Existing diffusion-based methods usually satisfy at most two of these requirements: global anomaly generators corrupt the background, whereas mask-guided ones often falter when the mask is imprecise or misplaced. We propose MAGIC--Mask-guided inpainting with multi-level perturbations and Context-aware alignment--to resolve all three issues. At its core, MAGIC fine-tunes a Stable Diffusion inpainting backbone that preserves normal regions and ensures strict adherence of the synthesized anomaly to the supplied mask, directly addressing background corruption and misalignment. To offset the diversity loss that fine-tuning can cause, MAGIC adds two complementary perturbation strategies: (i) Gaussian prompt-level perturbation applied during fine-tuning and inference that broadens the global appearance of anomalies while avoiding low-fidelity textual appearances, and (ii) mask-guided spatial noise injection that enriches local texture variations. Additionally, the context-aware mask alignment module forms semantic correspondences and relocates masks so that every anomaly remains plausibly contained within the host object, eliminating out-of-boundary artifacts. Under a consistent identical evaluation protocol on the MVTec-AD dataset, MAGIC outperforms previous state-of-the-arts in downstream anomaly tasks. 

**Abstract (ZH)**: 基于掩膜的多级扰动与上下文对齐的生成模型Magic：三重需求的 Few-shot 异常生成 

---
# Holistic Continual Learning under Concept Drift with Adaptive Memory Realignment 

**Title (ZH)**: 概念漂移下具有自适应内存重新对齐的整体持续学习 

**Authors**: Alif Ashrafee, Jedrzej Kozal, Michal Wozniak, Bartosz Krawczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.02310)  

**Abstract**: Traditional continual learning methods prioritize knowledge retention and focus primarily on mitigating catastrophic forgetting, implicitly assuming that the data distribution of previously learned tasks remains static. This overlooks the dynamic nature of real-world data streams, where concept drift permanently alters previously seen data and demands both stability and rapid adaptation.
We introduce a holistic framework for continual learning under concept drift that simulates realistic scenarios by evolving task distributions. As a baseline, we consider Full Relearning (FR), in which the model is retrained from scratch on newly labeled samples from the drifted distribution. While effective, this approach incurs substantial annotation and computational overhead. To address these limitations, we propose Adaptive Memory Realignment (AMR), a lightweight alternative that equips rehearsal-based learners with a drift-aware adaptation mechanism. AMR selectively removes outdated samples of drifted classes from the replay buffer and repopulates it with a small number of up-to-date instances, effectively realigning memory with the new distribution. This targeted resampling matches the performance of FR while reducing the need for labeled data and computation by orders of magnitude.
To enable reproducible evaluation, we introduce four concept-drift variants of standard vision benchmarks: Fashion-MNIST-CD, CIFAR10-CD, CIFAR100-CD, and Tiny-ImageNet-CD, where previously seen classes reappear with shifted representations. Comprehensive experiments on these datasets using several rehearsal-based baselines show that AMR consistently counters concept drift, maintaining high accuracy with minimal overhead. These results position AMR as a scalable solution that reconciles stability and plasticity in non-stationary continual learning environments. 

**Abstract (ZH)**: 面向概念漂移的全面连续学习框架：AMR 

---
# Synthetic Heuristic Evaluation: A Comparison between AI- and Human-Powered Usability Evaluation 

**Title (ZH)**: 合成启发式评估：AI-powered与人工Usability评估的比较 

**Authors**: Ruican Zhong, David W. McDonald, Gary Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2507.02306)  

**Abstract**: Usability evaluation is crucial in human-centered design but can be costly, requiring expert time and user compensation. In this work, we developed a method for synthetic heuristic evaluation using multimodal LLMs' ability to analyze images and provide design feedback. Comparing our synthetic evaluations to those by experienced UX practitioners across two apps, we found our evaluation identified 73% and 77% of usability issues, which exceeded the performance of 5 experienced human evaluators (57% and 63%). Compared to human evaluators, the synthetic evaluation's performance maintained consistent performance across tasks and excelled in detecting layout issues, highlighting potential attentional and perceptual strengths of synthetic evaluation. However, synthetic evaluation struggled with recognizing some UI components and design conventions, as well as identifying across screen violations. Additionally, testing synthetic evaluations over time and accounts revealed stable performance. Overall, our work highlights the performance differences between human and LLM-driven evaluations, informing the design of synthetic heuristic evaluations. 

**Abstract (ZH)**: 基于多模态LLM的合成 heuristic 评估方法在用户体验评估中的性能研究 

---
# DoMIX: An Efficient Framework for Exploiting Domain Knowledge in Fine-Tuning 

**Title (ZH)**: DoMIX：一种在微调中利用领域知识的有效框架 

**Authors**: Dohoon Kim, Donghun Kang, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2507.02302)  

**Abstract**: Domain-Adaptive Pre-training (DAP) has recently gained attention for its effectiveness in fine-tuning pre-trained models. Building on this, continual DAP has been explored to develop pre-trained models capable of incrementally incorporating different domain datasets. However, existing continual DAP methods face several limitations: (1) high computational cost and GPU memory usage during training; (2) sensitivity to incremental data order; and (3) providing a single, generalized model for all end tasks, which contradicts the essence of DAP. In this paper, we propose DoMIX, a novel approach that addresses these challenges by leveraging LoRA modules, a representative parameter-efficient fine-tuning (PEFT) method. Our approach enables efficient and parallel domain-adaptive pre-training that is robust to domain order and effectively utilizes accumulated knowledge to provide tailored pre-trained models for specific tasks. We also demonstrate that our method can be extended beyond the DAP setting to standard LLM fine-tuning scenarios. Code is available at this https URL. 

**Abstract (ZH)**: DoMIX：一种通过LoRA模块解决连续域自适应预训练挑战的方法 

---
# Knowledge Graph-Based Explainable and Generalized Zero-Shot Semantic Communications 

**Title (ZH)**: 基于知识图谱的可解释和泛化的零样本语义通信 

**Authors**: Zhaoyu Zhang, Lingyi Wang, Wei Wu, Fuhui Zhou, Qihui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02291)  

**Abstract**: Data-driven semantic communication is based on superficial statistical patterns, thereby lacking interpretability and generalization, especially for applications with the presence of unseen data. To address these challenges, we propose a novel knowledge graph-enhanced zero-shot semantic communication (KGZS-SC) network. Guided by the structured semantic information from a knowledge graph-based semantic knowledge base (KG-SKB), our scheme provides generalized semantic representations and enables reasoning for unseen cases. Specifically, the KG-SKB aligns the semantic features in a shared category semantics embedding space and enhances the generalization ability of the transmitter through aligned semantic features, thus reducing communication overhead by selectively transmitting compact visual semantics. At the receiver, zero-shot learning (ZSL) is leveraged to enable direct classification for unseen cases without the demand for retraining or additional computational overhead, thereby enhancing the adaptability and efficiency of the classification process in dynamic or resource-constrained environments. The simulation results conducted on the APY datasets show that the proposed KGZS-SC network exhibits robust generalization and significantly outperforms existing SC frameworks in classifying unseen categories across a range of SNR levels. 

**Abstract (ZH)**: 基于知识图谱增强的零样本语义通信（KGZS-SC）网络 

---
# Content filtering methods for music recommendation: A review 

**Title (ZH)**: 音乐推荐中的内容过滤方法：一个综述 

**Authors**: Terence Zeng, Abhishek K. Umrawal  

**Link**: [PDF](https://arxiv.org/pdf/2507.02282)  

**Abstract**: Recommendation systems have become essential in modern music streaming platforms, shaping how users discover and engage with songs. One common approach in recommendation systems is collaborative filtering, which suggests content based on the preferences of users with similar listening patterns to the target user. However, this method is less effective on media where interactions are sparse. Music is one such medium, since the average user of a music streaming service will never listen to the vast majority of tracks. Due to this sparsity, there are several challenges that have to be addressed with other methods. This review examines the current state of research in addressing these challenges, with an emphasis on the role of content filtering in mitigating biases inherent in collaborative filtering approaches. We explore various methods of song classification for content filtering, including lyrical analysis using Large Language Models (LLMs) and audio signal processing techniques. Additionally, we discuss the potential conflicts between these different analysis methods and propose avenues for resolving such discrepancies. 

**Abstract (ZH)**: 推荐系统已成为现代音乐流媒体平台的 essential 组件，影响着用户发现和互动歌曲的方式。推荐系统中的一种常见方法是协同过滤，它根据与目标用户具有相似听歌模式的用户偏好来推荐内容。然而，在用户交互稀少的媒体中，这种方法效果较差。音乐就是这样一个媒体，因为音乐流媒体服务的平均用户几乎不会收听平台上的绝大多数歌曲。由于这种稀疏性，需要采用其他方法来应对多种挑战。本文回顾了当前在这些挑战方面的研究现状，重点关注内容过滤在减轻协同过滤方法固有偏见中的作用。我们探讨了用于内容过滤的各种歌曲分类方法，包括使用大型语言模型（LLMs）进行歌词分析和音频信号处理技术。此外，我们讨论了这些不同分析方法之间的潜在冲突，并提出了解决这些分歧的途径。 

---
# Spotlighting Partially Visible Cinematic Language for Video-to-Audio Generation via Self-distillation 

**Title (ZH)**: 突出部分可见的cinematic语言以实现视频到音频生成的自精练 

**Authors**: Feizhen Huang, Yu Wu, Yutian Lin, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2507.02271)  

**Abstract**: Video-to-Audio (V2A) Generation achieves significant progress and plays a crucial role in film and video post-production. However, current methods overlook the cinematic language, a critical component of artistic expression in filmmaking. As a result, their performance deteriorates in scenarios where Foley targets are only partially visible. To address this challenge, we propose a simple self-distillation approach to extend V2A models to cinematic language scenarios. By simulating the cinematic language variations, the student model learns to align the video features of training pairs with the same audio-visual correspondences, enabling it to effectively capture the associations between sounds and partial visual information. Our method not only achieves impressive improvements under partial visibility across all evaluation metrics, but also enhances performance on the large-scale V2A dataset, VGGSound. 

**Abstract (ZH)**: 视频到音频（V2A）生成在影视后期制作中取得了显著进展并发挥着关键作用。然而，当前方法忽视了电影语言这一影视创作艺术表达中的关键成分。因此，在只有部分配音目标可见的场景中，其性能下降。为了解决这一挑战，我们提出了一种简单的自我精炼方法，将V2A模型扩展到电影语言场景中。通过模拟电影语言的变化，学生模型学习调整训练配对的视频特征与相同音视频对应关系，使其能够有效地捕捉声音与部分视觉信息之间的关联。我们的方法不仅在所有评估指标下实现了在部分可见情况下的显著改进，还提高了在大规模V2A数据集VGGSound上的性能。 

---
# Multi-Label Classification Framework for Hurricane Damage Assessment 

**Title (ZH)**: 飓风损害评估的多标签分类框架 

**Authors**: Zhangding Liu, Neda Mohammadi, John E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2507.02265)  

**Abstract**: Hurricanes cause widespread destruction, resulting in diverse damage types and severities that require timely and accurate assessment for effective disaster response. While traditional single-label classification methods fall short of capturing the complexity of post-hurricane damage, this study introduces a novel multi-label classification framework for assessing damage using aerial imagery. The proposed approach integrates a feature extraction module based on ResNet and a class-specific attention mechanism to identify multiple damage types within a single image. Using the Rescuenet dataset from Hurricane Michael, the proposed method achieves a mean average precision of 90.23%, outperforming existing baseline methods. This framework enhances post-hurricane damage assessment, enabling more targeted and efficient disaster response and contributing to future strategies for disaster mitigation and resilience. This paper has been accepted at the ASCE International Conference on Computing in Civil Engineering (i3CE 2025), and the camera-ready version will appear in the official conference proceedings. 

**Abstract (ZH)**: 飓风造成广泛破坏，导致多样化的损坏类型和程度，需要及时准确的评估以有效应对灾害。传统单标签分类方法无法捕捉飓风后的复杂损坏情况，本研究提出了一种新的多标签分类框架，利用航空影像评估损坏情况。该提出的方案结合了基于ResNet的功能提取模块和类别特定的关注机制，能够在单张图像中识别多种损坏类型。使用飓风迈克尔的Rescuenet数据集，所提出的办法实现了平均精确度90.23%的均值，优于现有基线方法。该框架提高了灾后损坏评估的精度，有助于更精确和高效的灾害应对，并为未来的灾害减轻和韧性策略提供了贡献。该论文已被接受参加ASCE国际土木工程计算会议（i3CE 2025），最终版本将出现在官方会议论文集中。 

---
# MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent 

**Title (ZH)**: MemAgent: 以多轮次卷积RL为基础的内存代理重塑长上下文LLM 

**Authors**: Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.02259)  

**Abstract**: Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents with linear complexity without performance degradation during extrapolation remains the ultimate challenge in long-text processing. We directly optimize for long-text tasks in an end-to-end fashion and introduce a novel agent workflow, MemAgent, which reads text in segments and updates the memory using an overwrite strategy. We extend the DAPO algorithm to facilitate training via independent-context multi-conversation generation. MemAgent has demonstrated superb long-context capabilities, being able to extrapolate from an 8K context trained on 32K text to a 3.5M QA task with performance loss < 5% and achieves 95%+ in 512K RULER test. 

**Abstract (ZH)**: 尽管通过长度外推、高效注意力和记忆模块的改进有所提升，但在不牺牲性能的情况下以线性复杂度处理无限长文档依然是长文本处理领域的终极挑战。我们以端到端的方式直接优化长文本任务，并引入了一种新颖的智能体工作流MemAgent，该工作流分段读取文本并使用覆盖策略更新记忆。我们扩展了DAPO算法，通过独立上下文多对话生成来促进训练。实验证明，MemAgent具备卓越的长上下文处理能力，能够从使用32K文本训练的8K上下文外推到3.5M QA任务，性能损失小于5%，并在512K RULER测试中达到95%以上。 

---
# SurgVisAgent: Multimodal Agentic Model for Versatile Surgical Visual Enhancement 

**Title (ZH)**: SurgVisAgent: 多模态代理模型淂 

**Authors**: Zeyu Lei, Hongyuan Yu, Jinlin Wu, Zhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.02252)  

**Abstract**: Precise surgical interventions are vital to patient safety, and advanced enhancement algorithms have been developed to assist surgeons in decision-making. Despite significant progress, these algorithms are typically designed for single tasks in specific scenarios, limiting their effectiveness in complex real-world situations. To address this limitation, we propose SurgVisAgent, an end-to-end intelligent surgical vision agent built on multimodal large language models (MLLMs). SurgVisAgent dynamically identifies distortion categories and severity levels in endoscopic images, enabling it to perform a variety of enhancement tasks such as low-light enhancement, overexposure correction, motion blur elimination, and smoke removal. Specifically, to achieve superior surgical scenario understanding, we design a prior model that provides domain-specific knowledge. Additionally, through in-context few-shot learning and chain-of-thought (CoT) reasoning, SurgVisAgent delivers customized image enhancements tailored to a wide range of distortion types and severity levels, thereby addressing the diverse requirements of surgeons. Furthermore, we construct a comprehensive benchmark simulating real-world surgical distortions, on which extensive experiments demonstrate that SurgVisAgent surpasses traditional single-task models, highlighting its potential as a unified solution for surgical assistance. 

**Abstract (ZH)**: 精确的外科干预对于患者安全至关重要，先进的增强算法已被开发出来以帮助外科医生进行决策。尽管取得了显著进展，但这些算法通常仅为特定场景中的单一任务设计，限制了其在复杂现实情况中的有效性。为解决这一限制，我们提出了SurgVisAgent，这是一种基于多模态大型语言模型（MLLMs）构建的端到端智能外科视觉代理。SurgVisAgent动态识别内窥镜图像中的失真类别和严重程度级别，使其能够执行多种增强任务，如低光增强、过度曝光校正、运动模糊消除和烟雾去除。具体来说，为了实现优越的外科情景理解，我们设计了一种先验模型提供领域特定知识。此外，通过上下文中的少样本学习和链式思考（CoT）推理，SurgVisAgent能够为广泛类型的失真及其严重程度级别提供定制化的图像增强，从而满足外科医生的各种需求。此外，我们构建了一个全面的基准测试，模拟真实的外科失真情况，大量的实验结果显示SurgVisAgent超越了传统的单一任务模型，凸显其作为外科辅助统一解决方案的潜力。 

---
# Order Acquisition Under Competitive Pressure: A Rapidly Adaptive Reinforcement Learning Approach for Ride-Hailing Subsidy Strategies 

**Title (ZH)**: 在竞争压力下的订单获取策略：一种快速自适应强化学习方法用于网约车补贴策略 

**Authors**: Fangzhou Shi, Xiaopeng Ke, Xinye Xiong, Kexin Meng, Chang Men, Zhengdan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02244)  

**Abstract**: The proliferation of ride-hailing aggregator platforms presents significant growth opportunities for ride-service providers by increasing order volume and gross merchandise value (GMV). On most ride-hailing aggregator platforms, service providers that offer lower fares are ranked higher in listings and, consequently, are more likely to be selected by passengers. This competitive ranking mechanism creates a strong incentive for service providers to adopt coupon strategies that lower prices to secure a greater number of orders, as order volume directly influences their long-term viability and sustainability. Thus, designing an effective coupon strategy that can dynamically adapt to market fluctuations while optimizing order acquisition under budget constraints is a critical research challenge. However, existing studies in this area remain scarce.
To bridge this gap, we propose FCA-RL, a novel reinforcement learning-based subsidy strategy framework designed to rapidly adapt to competitors' pricing adjustments. Our approach integrates two key techniques: Fast Competition Adaptation (FCA), which enables swift responses to dynamic price changes, and Reinforced Lagrangian Adjustment (RLA), which ensures adherence to budget constraints while optimizing coupon decisions on new price landscape. Furthermore, we introduce RideGym, the first dedicated simulation environment tailored for ride-hailing aggregators, facilitating comprehensive evaluation and benchmarking of different pricing strategies without compromising real-world operational efficiency. Experimental results demonstrate that our proposed method consistently outperforms baseline approaches across diverse market conditions, highlighting its effectiveness in subsidy optimization for ride-hailing service providers. 

**Abstract (ZH)**: 网约车聚合平台的 proliferations 为网约车服务提供商带来了显著的增长机会，通过增加订单量和总商品价值（GMV）。在大多数网约车聚合平台上，提供较低价格的服务提供商会被优先排名，从而更有可能被乘客选择。这种竞争排名机制为服务提供商采用降价策略以获取更多订单提供了强烈的动力，因为订单量直接影响他们的长期可行性和可持续性。因此，设计一种能够在预算约束下动态适应市场波动并优化订单获取的有效补贴策略是一项关键的研究挑战。然而，该领域的现有研究依然匮乏。

为弥补这一缺口，我们提出了FCA-RL，一种基于强化学习的新型补贴策略框架，旨在快速适应竞争对手的价格调整。我们的方法结合了两种关键技术：快速竞争适应（FCA），使快速响应动态价格变化成为可能；强化拉格朗日调整（RLA），确保在新的价格环境中遵守预算约束的同时优化补贴决策。此外，我们还引入了RideGym，这是首个专门针对网约车聚合商的仿真环境，便于在不牺牲现实运营效率的情况下对不同定价策略进行全面评估和基准测试。实验结果表明，我们提出的方法在各种市场条件下始终优于基线方法，突显了其在网约车服务提供商补贴优化方面的有效性。 

---
# Understanding Trade offs When Conditioning Synthetic Data 

**Title (ZH)**: 理解条件生成合成数据时的权衡 

**Authors**: Brandon Trabucco, Qasim Wani, Benjamin Pikus, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.02217)  

**Abstract**: Learning robust object detectors from only a handful of images is a critical challenge in industrial vision systems, where collecting high quality training data can take months. Synthetic data has emerged as a key solution for data efficient visual inspection and pick and place robotics. Current pipelines rely on 3D engines such as Blender or Unreal, which offer fine control but still require weeks to render a small dataset, and the resulting images often suffer from a large gap between simulation and reality. Diffusion models promise a step change because they can generate high quality images in minutes, yet precise control, especially in low data regimes, remains difficult. Although many adapters now extend diffusion beyond plain text prompts, the effect of different conditioning schemes on synthetic data quality is poorly understood. We study eighty diverse visual concepts drawn from four standard object detection benchmarks and compare two conditioning strategies: prompt based and layout based. When the set of conditioning cues is narrow, prompt conditioning yields higher quality synthetic data; as diversity grows, layout conditioning becomes superior. When layout cues match the full training distribution, synthetic data raises mean average precision by an average of thirty four percent and by as much as one hundred seventy seven percent compared with using real data alone. 

**Abstract (ZH)**: 仅从少量图像中学习稳健的物体检测器是工业视觉系统中的一个关键挑战，其中收集高质量的训练数据可能需要几个月的时间。合成数据已成为数据高效视觉检查和拾放机器人中的关键解决方案。当前的管道依赖于如Blender或Unreal等3D引擎，虽然提供了精细控制，但仍需要几周时间才能渲染一个小的数据集，且生成的图像经常与现实之间存在较大差距。扩散模型承诺了一个飞跃，因为它们可以在几分钟内生成高质量的图像，但特别是在低数据条件下保持精确控制仍然困难。尽管现在许多适配器已将扩散模型扩展到简单的文本提示之外，但不同的条件方案对合成数据质量的影响尚未完全理解。我们研究了四个标准物体检测基准中的八十种多样化的视觉概念，并比较了两种条件方案：提示基于和布局基于。当条件线索集合狭窄时，提示条件生成的合成数据质量更高；随着多样性的增加，布局条件变得更为优越。当布局线索匹配完整的训练分布时，与仅使用真实数据相比，合成数据可将平均准确率提高34%，最多提高177%。 

---
# EIM-TRNG: Obfuscating Deep Neural Network Weights with Encoding-in-Memory True Random Number Generator via RowHammer 

**Title (ZH)**: EIM-TRNG: 使用行hammer技术结合编码-in-内存真随机数生成器混淆深层神经网络权重 

**Authors**: Ranyang Zhou, Abeer Matar A. Almalky, Gamana Aragonda, Sabbir Ahmed, Filip Roth Trønnes-Christensen, Adnan Siraj Rakin, Shaahin Angizi  

**Link**: [PDF](https://arxiv.org/pdf/2507.02206)  

**Abstract**: True Random Number Generators (TRNGs) play a fundamental role in hardware security, cryptographic systems, and data protection. In the context of Deep NeuralNetworks (DNNs), safeguarding model parameters, particularly weights, is critical to ensure the integrity, privacy, and intel-lectual property of AI systems. While software-based pseudo-random number generators are widely used, they lack the unpredictability and resilience offered by hardware-based TRNGs. In this work, we propose a novel and robust Encoding-in-Memory TRNG called EIM-TRNG that leverages the inherent physical randomness in DRAM cell behavior, particularly under RowHammer-induced disturbances, for the first time. We demonstrate how the unpredictable bit-flips generated through carefully controlled RowHammer operations can be harnessed as a reliable entropy source. Furthermore, we apply this TRNG framework to secure DNN weight data by encoding via a combination of fixed and unpredictable bit-flips. The encrypted data is later decrypted using a key derived from the probabilistic flip behavior, ensuring both data confidentiality and model authenticity. Our results validate the effectiveness of DRAM-based entropy extraction for robust, low-cost hardware security and offer a promising direction for protecting machine learning models at the hardware level. 

**Abstract (ZH)**: 基于内存的真随机数生成器（EIM-TRNG）在深度神经网络中的新型鲁棒奇偶校验安全机制 

---
# ESTR-CoT: Towards Explainable and Accurate Event Stream based Scene Text Recognition with Chain-of-Thought Reasoning 

**Title (ZH)**: ESTR-CoT：基于事件流的场景文本识别的可解释性和准确性提升方法及其链式推理 reasoning 

**Authors**: Xiao Wang, Jingtao Jiang, Qiang Chen, Lan Chen, Lin Zhu, Yaowei Wang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02200)  

**Abstract**: Event stream based scene text recognition is a newly arising research topic in recent years which performs better than the widely used RGB cameras in extremely challenging scenarios, especially the low illumination, fast motion. Existing works either adopt end-to-end encoder-decoder framework or large language models for enhanced recognition, however, they are still limited by the challenges of insufficient interpretability and weak contextual logical reasoning. In this work, we propose a novel chain-of-thought reasoning based event stream scene text recognition framework, termed ESTR-CoT. Specifically, we first adopt the vision encoder EVA-CLIP (ViT-G/14) to transform the input event stream into tokens and utilize a Llama tokenizer to encode the given generation prompt. A Q-former is used to align the vision token to the pre-trained large language model Vicuna-7B and output both the answer and chain-of-thought (CoT) reasoning process simultaneously. Our framework can be optimized using supervised fine-tuning in an end-to-end manner. In addition, we also propose a large-scale CoT dataset to train our framework via a three stage processing (i.e., generation, polish, and expert verification). This dataset provides a solid data foundation for the development of subsequent reasoning-based large models. Extensive experiments on three event stream STR benchmark datasets (i.e., EventSTR, WordArt*, IC15*) fully validated the effectiveness and interpretability of our proposed framework. The source code and pre-trained models will be released on this https URL. 

**Abstract (ZH)**: 基于事件流的场景文本识别是一种近年来新兴的研究课题，相较于广泛使用的RGB相机，在极具有挑战性的场景下（尤其是低照度、快速运动）表现更佳。现有工作要么采用端到端的编码-解码框架，要么利用大规模语言模型来增强识别效果，然而它们仍然受到缺乏可解释性和弱上下文逻辑推理能力的限制。在本工作中，我们提出了一种新颖的基于链式思考的事件流场景文本识别框架，命名为ESTR-CoT。具体而言，我们首先采用Vision Encoder EVA-CLIP (ViT-G/14) 将输入的事件流转换为 tokens，并利用 Llama 令牌化器对生成提示进行编码。然后使用 Q-former 将视觉 token 对齐到预训练的大规模语言模型 Vicuna-7B，并同时输出答案和链式思考（CoT）推理过程。我们的框架可以以端到端的方式通过监督微调进行优化。此外，我们还提出了一大规模 CoT 数据集，通过三阶段处理（即生成、润色和专家验证）来训练我们的框架。该数据集为后续基于推理的大规模模型的发展提供了坚实的数据基础。在三个事件流 STR 基准数据集（即 EventSTR、WordArt*、IC15*）上的广泛实验充分验证了我们提出框架的有效性和可解释性。源代码和预训练模型将在此处 https://链接发布。 

---
# Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer 

**Title (ZH)**: 潜在的思维链？深度递归变压器的解码 

**Authors**: Wenquan Lu, Yuechuan Yang, Kyle Lee, Yanshu Li, Enqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02199)  

**Abstract**: Chain-of-thought (CoT) reasoning has enabled transformer-based language models to excel at complex mathematics and multi-step planning. However, in standard decoder-only architectures, these reasoning steps are externalized in natural language, improving interpretability at the cost of efficiency. To capture reasoning that is not easily represented in words, many works have explored recurrent architectures that aim to internalize reasoning in latent space, potentially supporting latent CoT. In this paper, we investigate whether such reasoning structures emerge in Huginn-3.5B, a depth-recurrent Transformer that reuses layers at inference time without increasing parameter count. We examine the model's internal behavior on arithmetic tasks using a suite of probing techniques including the Logit Lens and Coda Lens. Our findings reveal limited evidence of interpretable latent CoT by tracking rank trajectories of final and intermediate result tokens. Furthermore, we uncover significant probing inconsistencies across recurrent blocks, where the interpretability of hidden states depends heavily on both the layer index and the decoding method. Finally, we empirically show that increasing recurrence depth yields only marginal gains and falls well short of models that explicitly externalize reasoning steps. The code is available at this https URL. 

**Abstract (ZH)**: Huginn-3.5B中是否存在深度递归Transformer中的可解释潜在CoT结构？ 

---
# Towards Bio-Inspired Robotic Trajectory Planning via Self-Supervised RNN 

**Title (ZH)**: 面向生物启发的机器人轨迹规划的自我监督递归神经网络方法 

**Authors**: Miroslav Cibula, Kristína Malinovská, Matthias Kerzel  

**Link**: [PDF](https://arxiv.org/pdf/2507.02171)  

**Abstract**: Trajectory planning in robotics is understood as generating a sequence of joint configurations that will lead a robotic agent, or its manipulator, from an initial state to the desired final state, thus completing a manipulation task while considering constraints like robot kinematics and the environment. Typically, this is achieved via sampling-based planners, which are computationally intensive. Recent advances demonstrate that trajectory planning can also be performed by supervised sequence learning of trajectories, often requiring only a single or fixed number of passes through a neural architecture, thus ensuring a bounded computation time. Such fully supervised approaches, however, perform imitation learning; they do not learn based on whether the trajectories can successfully reach a goal, but try to reproduce observed trajectories. In our work, we build on this approach and propose a cognitively inspired self-supervised learning scheme based on a recurrent architecture for building a trajectory model. We evaluate the feasibility of the proposed method on a task of kinematic planning for a robotic arm. The results suggest that the model is able to learn to generate trajectories only using given paired forward and inverse kinematics models, and indicate that this novel method could facilitate planning for more complex manipulation tasks requiring adaptive solutions. 

**Abstract (ZH)**: 机器人轨迹规划被视为生成一系列关节配置，使机器人代理或其操作器从初始状态过渡到期望的最终状态，同时考虑机器人运动学约束和环境因素。通常，这通过基于采样的规划者实现，计算成本较高。近年来的研究表明，轨迹规划也可以通过监督序列学习实现，通常只需要通过神经架构进行一次或固定次数的传递，从而确保计算时间受限。然而，这类完全监督的方法进行的是模仿学习，它们不根据轨迹是否能成功到达目标来学习，而是试图再现观察到的轨迹。在我们的研究中，我们在此基础上提出了一种基于递归架构的认知启发式自我监督学习方案，用于构建轨迹模型。我们在具有正逆运动学模型的机器人臂的动力学规划任务上评估了该方法的可行性。结果显示，该模型仅使用给定的正向和逆向运动学模型配对数据就能够学习生成轨迹，表明该新颖方法可能有助于为需要适应性解决方案的更复杂操作任务规划提供支持。 

---
# Generating Large Semi-Synthetic Graphs of Any Size 

**Title (ZH)**: 生成任意大小的大型半合成图 

**Authors**: Rodrigo Tuna, Carlos Soares  

**Link**: [PDF](https://arxiv.org/pdf/2507.02166)  

**Abstract**: Graph generation is an important area in network science. Traditional approaches focus on replicating specific properties of real-world graphs, such as small diameters or power-law degree distributions. Recent advancements in deep learning, particularly with Graph Neural Networks, have enabled data-driven methods to learn and generate graphs without relying on predefined structural properties. Despite these advances, current models are limited by their reliance on node IDs, which restricts their ability to generate graphs larger than the input graph and ignores node attributes. To address these challenges, we propose Latent Graph Sampling Generation (LGSG), a novel framework that leverages diffusion models and node embeddings to generate graphs of varying sizes without retraining. The framework eliminates the dependency on node IDs and captures the distribution of node embeddings and subgraph structures, enabling scalable and flexible graph generation. Experimental results show that LGSG performs on par with baseline models for standard metrics while outperforming them in overlooked ones, such as the tendency of nodes to form clusters. Additionally, it maintains consistent structural characteristics across graphs of different sizes, demonstrating robustness and scalability. 

**Abstract (ZH)**: 图生成是网络科学中的一个重要领域。传统方法主要关注复制真实世界图的特定属性，如小直径或幂律度分布。最近深度学习的进展，特别是图神经网络的应用，使能够通过数据驱动的方法学习和生成图，而无需依赖预定义的结构属性。尽管取得了这些进展，当前的模型仍然受限于对节点ID的依赖，这限制了它们生成比输入图更大的图的能力，并忽略了节点属性。为了解决这些问题，我们提出了潜在图采样生成（LGSG）框架，这是一种新颖的方法，利用扩散模型和节点嵌入来生成不需重新训练的各种大小的图。该框架消除了对节点ID的依赖，捕捉节点嵌入和子图结构的分布，从而实现可扩展和灵活的图生成。实验结果表明，LGSG在标准指标上与基线模型表现相当，在一些未被忽视的指标上（如节点形成集群的趋势）表现更佳。此外，它在不同大小的图上保持了结构特征的一致性，展示了鲁棒性和可扩展性。 

---
# Reasoning or Not? A Comprehensive Evaluation of Reasoning LLMs for Dialogue Summarization 

**Title (ZH)**: 推理还是不推理？面向对话总结的推理型LLMs综合评估 

**Authors**: Keyan Jin, Yapeng Wang, Leonel Santos, Tao Fang, Xu Yang, Sio Kei Im, Hugo Gonçalo Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2507.02145)  

**Abstract**: Dialogue summarization is a challenging task with significant practical value in customer service, meeting analysis, and conversational AI. Although large language models (LLMs) have achieved substantial progress in summarization tasks, the performance of step-by-step reasoning architectures-specifically Long Chain-of-Thought (CoT) implementations such as OpenAI-o1 and DeepSeek-R1-remains unexplored for dialogue scenarios requiring concurrent abstraction and conciseness. In this work, we present the first comprehensive and systematic evaluation of state-of-the-art reasoning LLMs and non-reasoning LLMs across three major paradigms-generic, role-oriented, and query-oriented dialogue summarization. Our study spans diverse languages, domains, and summary lengths, leveraging strong benchmarks (SAMSum, DialogSum, CSDS, and QMSum) and advanced evaluation protocols that include both LLM-based automatic metrics and human-inspired criteria. Contrary to trends in other reasoning-intensive tasks, our findings show that explicit stepwise reasoning does not consistently improve dialogue summarization quality. Instead, reasoning LLMs are often prone to verbosity, factual inconsistencies, and less concise summaries compared to their non-reasoning counterparts. Through scenario-specific analyses and detailed case studies, we further identify when and why explicit reasoning may fail to benefit-or even hinder-summarization in complex dialogue contexts. Our work provides new insights into the limitations of current reasoning LLMs and highlights the need for targeted modeling and evaluation strategies for real-world dialogue summarization. 

**Abstract (ZH)**: 对话摘要是客服、会议分析和会话AI等领域中具有重要实践价值的挑战性任务。尽管大型语言模型（LLMs）在摘要任务中取得了显著进展，但对于需要同时进行抽象和简洁处理的对话场景，逐步推理架构——特别是OpenAI-o1和DeepSeek-R1等长链推理实现——的表现仍然未被探索。在本研究中，我们首次对最先进的推理LLMs和非推理LLMs在三大主要范式（通用、角色导向和查询导向）下的对话摘要进行了综合和系统的评估。我们的研究涵盖了多种语言、领域和摘要长度，利用了强大的基准（SAMSum、DialogSum、CSDS和QMSum）和先进的评估协议，包括基于LLMs的自动评价指标和基于人类评估的标准。与其他推理密集型任务的趋势相反，我们的研究发现显式的步骤推理并不总是提升对话摘要的质量。相反，推理LLMs通常会产生冗长、事实不一致和不那么简洁的摘要，相较于非推理模型更显得如此。通过特定场景的分析和详细的案例研究，我们进一步确定了在复杂对话情境中显式推理为何可能未能提高——甚至可能妨碍——摘要的质量。我们的工作提供了当前推理LLMs局限性的新见解，并强调了针对实际对话场景的摘要进行针对性建模和评估策略的必要性。 

---
# When LLMs Disagree: Diagnosing Relevance Filtering Bias and Retrieval Divergence in SDG Search 

**Title (ZH)**: 当大语言模型产生分歧：SDG搜索中相关性筛选偏见和检索分歧诊断 

**Authors**: William A. Ingram, Bipasha Banerjee, Edward A. Fox  

**Link**: [PDF](https://arxiv.org/pdf/2507.02139)  

**Abstract**: Large language models (LLMs) are increasingly used to assign document relevance labels in information retrieval pipelines, especially in domains lacking human-labeled data. However, different models often disagree on borderline cases, raising concerns about how such disagreement affects downstream retrieval. This study examines labeling disagreement between two open-weight LLMs, LLaMA and Qwen, on a corpus of scholarly abstracts related to Sustainable Development Goals (SDGs) 1, 3, and 7. We isolate disagreement subsets and examine their lexical properties, rank-order behavior, and classification predictability. Our results show that model disagreement is systematic, not random: disagreement cases exhibit consistent lexical patterns, produce divergent top-ranked outputs under shared scoring functions, and are distinguishable with AUCs above 0.74 using simple classifiers. These findings suggest that LLM-based filtering introduces structured variability in document retrieval, even under controlled prompting and shared ranking logic. We propose using classification disagreement as an object of analysis in retrieval evaluation, particularly in policy-relevant or thematic search tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在信息检索管道中 increasingly 用于分配文档相关性标签，特别是在缺乏人类标注数据的领域。然而，不同模型在边界情况上经常存在分歧，这引发了关于这种分歧如何影响下游检索的问题。本研究考察了两种开放权重 LLM（LLaMA 和 Qwen）在与可持续发展目标（SDGs）1、3 和 7 相关的学术摘要集上的标签分歧。我们隔离分歧子集并研究其词级属性、排序行为和分类可预测性。研究结果显示，模型分歧是系统性的而非随机的：分歧案例表现出一致的词级模式，在共享评分函数下产生分歧的最高排名输出，并且使用简单分类器的 AUC 值超过 0.74 可以将这些分歧区分开来。这些发现表明，基于 LLM 的过滤会在受控提示和共享排名逻辑下引入文档检索中的结构化变异性。我们建议在检索评估中将分类分歧作为分析对象，特别是在政策相关或主题搜索任务中。 

---
# Can Artificial Intelligence solve the blockchain oracle problem? Unpacking the Challenges and Possibilities 

**Title (ZH)**: 人工智能能否解决区块链预言机问题？解读挑战与可能性 

**Authors**: Giulio Caldarelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.02125)  

**Abstract**: The blockchain oracle problem, which refers to the challenge of injecting reliable external data into decentralized systems, remains a fundamental limitation to the development of trustless applications. While recent years have seen a proliferation of architectural, cryptographic, and economic strategies to mitigate this issue, no one has yet fully resolved the fundamental question of how a blockchain can gain knowledge about the off-chain world. In this position paper, we critically assess the role artificial intelligence (AI) can play in tackling the oracle problem. Drawing from both academic literature and practitioner implementations, we examine how AI techniques such as anomaly detection, language-based fact extraction, dynamic reputation modeling, and adversarial resistance can enhance oracle systems. We observe that while AI introduces powerful tools for improving data quality, source selection, and system resilience, it cannot eliminate the reliance on unverifiable off-chain inputs. Therefore, this study supports the idea that AI should be understood as a complementary layer of inference and filtering within a broader oracle design, not a substitute for trust assumptions. 

**Abstract (ZH)**: 区块链预言机问题：人工智能在解决外部数据注入去中心化系统挑战中的作用 

---
# Resolving Turbulent Magnetohydrodynamics: A Hybrid Operator-Diffusion Framework 

**Title (ZH)**: 解析湍流磁流体力学：一种混合算子-扩散框架 

**Authors**: Semih Kacmaz, E. A. Huerta, Roland Haas  

**Link**: [PDF](https://arxiv.org/pdf/2507.02106)  

**Abstract**: We present a hybrid machine learning framework that combines Physics-Informed Neural Operators (PINOs) with score-based generative diffusion models to simulate the full spatio-temporal evolution of two-dimensional, incompressible, resistive magnetohydrodynamic (MHD) turbulence across a broad range of Reynolds numbers ($\mathrm{Re}$). The framework leverages the equation-constrained generalization capabilities of PINOs to predict coherent, low-frequency dynamics, while a conditional diffusion model stochastically corrects high-frequency residuals, enabling accurate modeling of fully developed turbulence. Trained on a comprehensive ensemble of high-fidelity simulations with $\mathrm{Re} \in \{100, 250, 500, 750, 1000, 3000, 10000\}$, the approach achieves state-of-the-art accuracy in regimes previously inaccessible to deterministic surrogates. At $\mathrm{Re}=1000$ and $3000$, the model faithfully reconstructs the full spectral energy distributions of both velocity and magnetic fields late into the simulation, capturing non-Gaussian statistics, intermittent structures, and cross-field correlations with high fidelity. At extreme turbulence levels ($\mathrm{Re}=10000$), it remains the first surrogate capable of recovering the high-wavenumber evolution of the magnetic field, preserving large-scale morphology and enabling statistically meaningful predictions. 

**Abstract (ZH)**: 我们提出了一种结合物理知情神经算子(PINOs)和分数阶生成扩散模型的混合机器学习框架，用于模拟二维不可压缩、有电阻的磁流体动力学(MHD)湍流在整个瑞利数($\mathrm{Re}$)范围内的时空演化。该框架利用PINOs的方程约束泛化能力预测相干的低频动态，同时条件扩散模型随机校正高频残差，从而使模型能够准确模拟完全发展的湍流。该方法在瑞利数$\mathrm{Re} \in \{100, 250, 500, 750, 1000, 3000, 10000\}$的高保真模拟集合上进行训练，实现了在以前由确定性代理不可达的湍流区域中的最先进准确性。在$\mathrm{Re}=1000$和$3000$时，该模型在模拟后期准确重构了速度和磁场的完整频谱能量分布，捕获了非高斯统计、间歇结构和跨场相关性。在极端湍流水平($\mathrm{Re}=10000$)时，它是唯一能恢复磁场高波数演化的代理模型，保持了大尺度形态并促进了统计上有意义的预测。 

---
# Energy-Based Transformers are Scalable Learners and Thinkers 

**Title (ZH)**: 基于能量的学习与思考可扩展变换器 

**Authors**: Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, Tariq Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2507.02092)  

**Abstract**: Inference-time computation techniques, analogous to human System 2 Thinking, have recently become popular for improving model performances. However, most existing approaches suffer from several limitations: they are modality-specific (e.g., working only in text), problem-specific (e.g., verifiable domains like math and coding), or require additional supervision/training on top of unsupervised pretraining (e.g., verifiers or verifiable rewards). In this paper, we ask the question "Is it possible to generalize these System 2 Thinking approaches, and develop models that learn to think solely from unsupervised learning?" Interestingly, we find the answer is yes, by learning to explicitly verify the compatibility between inputs and candidate-predictions, and then re-framing prediction problems as optimization with respect to this verifier. Specifically, we train Energy-Based Transformers (EBTs) -- a new class of Energy-Based Models (EBMs) -- to assign an energy value to every input and candidate-prediction pair, enabling predictions through gradient descent-based energy minimization until convergence. Across both discrete (text) and continuous (visual) modalities, we find EBTs scale faster than the dominant Transformer++ approach during training, achieving an up to 35% higher scaling rate with respect to data, batch size, parameters, FLOPs, and depth. During inference, EBTs improve performance with System 2 Thinking by 29% more than the Transformer++ on language tasks, and EBTs outperform Diffusion Transformers on image denoising while using fewer forward passes. Further, we find that EBTs achieve better results than existing models on most downstream tasks given the same or worse pretraining performance, suggesting that EBTs generalize better than existing approaches. Consequently, EBTs are a promising new paradigm for scaling both the learning and thinking capabilities of models. 

**Abstract (ZH)**: 基于推理时计算技术的通用化：从无监督学习中学习思维的模型 

---
# GeoAda: Efficiently Finetune Geometric Diffusion Models with Equivariant Adapters 

**Title (ZH)**: GeoAda: 用等变适配器高效微调几何扩散模型 

**Authors**: Wanjia Zhao, Jiaqi Han, Siyi Gu, Mingjian Jiang, James Zou, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2507.02085)  

**Abstract**: Geometric diffusion models have shown remarkable success in molecular dynamics and structure generation. However, efficiently fine-tuning them for downstream tasks with varying geometric controls remains underexplored. In this work, we propose an SE(3)-equivariant adapter framework ( GeoAda) that enables flexible and parameter-efficient fine-tuning for controlled generative tasks without modifying the original model architecture. GeoAda introduces a structured adapter design: control signals are first encoded through coupling operators, then processed by a trainable copy of selected pretrained model layers, and finally projected back via decoupling operators followed by an equivariant zero-initialized convolution. By fine-tuning only these lightweight adapter modules, GeoAda preserves the model's geometric consistency while mitigating overfitting and catastrophic forgetting. We theoretically prove that the proposed adapters maintain SE(3)-equivariance, ensuring that the geometric inductive biases of the pretrained diffusion model remain intact during adaptation. We demonstrate the wide applicability of GeoAda across diverse geometric control types, including frame control, global control, subgraph control, and a broad range of application domains such as particle dynamics, molecular dynamics, human motion prediction, and molecule generation. Empirical results show that GeoAda achieves state-of-the-art fine-tuning performance while preserving original task accuracy, whereas other baselines experience significant performance degradation due to overfitting and catastrophic forgetting. 

**Abstract (ZH)**: 几何不变同构适配器框架（GeoAda）在受控生成任务中的高效细调 

---
# Large Language Models for Crash Detection in Video: A Survey of Methods, Datasets, and Challenges 

**Title (ZH)**: 大规模语言模型在视频碰撞检测中的应用：方法、数据集及挑战综述 

**Authors**: Sanjeda Akter, Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.02074)  

**Abstract**: Crash detection from video feeds is a critical problem in intelligent transportation systems. Recent developments in large language models (LLMs) and vision-language models (VLMs) have transformed how we process, reason about, and summarize multimodal information. This paper surveys recent methods leveraging LLMs for crash detection from video data. We present a structured taxonomy of fusion strategies, summarize key datasets, analyze model architectures, compare performance benchmarks, and discuss ongoing challenges and opportunities. Our review provides a foundation for future research in this fast-growing intersection of video understanding and foundation models. 

**Abstract (ZH)**: 基于视频流的碰撞检测是智能交通系统中的关键问题。大规模语言模型（LLMs）和视觉-语言模型（VLMs）的Recent发展变革了我们处理、推理和总结多模态信息的方式。本文综述了利用LLMs进行视频数据碰撞检测的Recent方法。我们提出了融合策略的结构化分类，总结了关键数据集，分析了模型结构，比较了性能基准，并讨论了现存的挑战与机遇。我们的综述为视频理解与基础模型这一快速发展领域的未来研究奠定了基础。 

---
# MGC: A Compiler Framework Exploiting Compositional Blindness in Aligned LLMs for Malware Generation 

**Title (ZH)**: MGC：一种利用对齐大型语言模型中组成盲性编译框架的恶意软件生成 

**Authors**: Lu Yan, Zhuo Zhang, Xiangzhe Xu, Shengwei An, Guangyu Shen, Zhou Xuan, Xuan Chen, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02057)  

**Abstract**: Large language models (LLMs) have democratized software development, reducing the expertise barrier for programming complex applications. This accessibility extends to malicious software development, raising significant security concerns. While LLM providers have implemented alignment mechanisms to prevent direct generation of overtly malicious code, these safeguards predominantly evaluate individual prompts in isolation, overlooking a critical vulnerability: malicious operations can be systematically decomposed into benign-appearing sub-tasks. In this paper, we introduce the Malware Generation Compiler (MGC), a novel framework that leverages this vulnerability through modular decomposition and alignment-evasive generation. MGC employs a specialized Malware Description Intermediate Representation (MDIR) to bridge high-level malicious intents and benign-appearing code snippets. Extensive evaluation demonstrates that our attack reliably generates functional malware across diverse task specifications and categories, outperforming jailbreaking methods by +365.79% and underground services by +78.07% in correctness on three benchmark datasets. Case studies further show that MGC can reproduce and even enhance 16 real-world malware samples. This work provides critical insights for security researchers by exposing the risks of compositional attacks against aligned AI systems. Demonstrations are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已民主化软件开发，降低了开发复杂应用程序的专业门槛。这种易用性也扩展到了恶意软件开发，引发了重大的安全关切。尽管LLM提供商已实施对齐机制以防止直接生成明显的恶意代码，这些保护措施主要在孤立地评估个别提示，忽视了一个关键漏洞：恶意操作可以系统地分解为表面上无害的子任务。在本文中，我们引入了恶意软件生成编译器（MGC），这是一种新的框架，利用这一漏洞通过模块化分解和对齐规避生成。MGC使用专门的恶意软件描述中间表示（MDIR）来连接高层级的恶意意图和表面上无害的代码片段。广泛的评估表明，我们的攻击能够在多种任务规范和类别中可靠地生成功能性恶意软件，相对于三种基准数据集上的越狱方法和地下服务，正确性分别高出365.79%和78.07%。案例研究进一步表明，MGC可以重现甚至增强16个真实世界的恶意软件样本。这项工作为安全研究人员提供了关键见解，暴露了针对对齐AI系统的组合式攻击的风险。演示可在[该链接]上获得。 

---
# NGAT: A Node-level Graph Attention Network for Long-term Stock Prediction 

**Title (ZH)**: NGAT：用于长期股票预测的节点级图注意力网络 

**Authors**: Yingjie Niu, Mingchuan Zhao, Valerio Poti, Ruihai Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.02018)  

**Abstract**: Graph representation learning methods have been widely adopted in financial applications to enhance company representations by leveraging inter-firm relationships. However, current approaches face three key challenges: (1) The advantages of relational information are obscured by limitations in downstream task designs; (2) Existing graph models specifically designed for stock prediction often suffer from excessive complexity and poor generalization; (3) Experience-based construction of corporate relationship graphs lacks effective comparison of different graph structures. To address these limitations, we propose a long-term stock prediction task and develop a Node-level Graph Attention Network (NGAT) specifically tailored for corporate relationship graphs. Furthermore, we experimentally demonstrate the limitations of existing graph comparison methods based on model downstream task performance. Experimental results across two datasets consistently demonstrate the effectiveness of our proposed task and model. The project is publicly available on GitHub to encourage reproducibility and future research. 

**Abstract (ZH)**: 图表示学习方法在利用企业间关系提升公司表示以增强金融应用中的长周期股票预测任务研究中，现有方法面临三个关键挑战：（1）下游任务设计限制了关系信息的优势；（2）专门用于股票预测的图模型往往过于复杂且泛化能力差；（3）基于经验构建的企业关系图缺乏不同图结构的有效比较。为解决这些限制，我们提出了一个长周期股票预测任务，并开发了一个针对企业关系图优化的节点级图注意力网络（NGAT）。此外，我们通过实验展示了现有图比较方法基于模型下游任务性能的局限性。在两个数据集上的实验结果一致证明了我们提出任务和模型的有效性。该项目已在GitHub上开源，以促进可重复性和未来研究。 

---
# Effective Explanations for Belief-Desire-Intention Robots: When and What to Explain 

**Title (ZH)**: 信念-欲望-意图机器人有效的解释：何时以及解释什么 

**Authors**: Cong Wang, Roberto Calandra, Verena Klös  

**Link**: [PDF](https://arxiv.org/pdf/2507.02016)  

**Abstract**: When robots perform complex and context-dependent tasks in our daily lives, deviations from expectations can confuse users. Explanations of the robot's reasoning process can help users to understand the robot intentions. However, when to provide explanations and what they contain are important to avoid user annoyance. We have investigated user preferences for explanation demand and content for a robot that helps with daily cleaning tasks in a kitchen. Our results show that users want explanations in surprising situations and prefer concise explanations that clearly state the intention behind the confusing action and the contextual factors that were relevant to this decision. Based on these findings, we propose two algorithms to identify surprising actions and to construct effective explanations for Belief-Desire-Intention (BDI) robots. Our algorithms can be easily integrated in the BDI reasoning process and pave the way for better human-robot interaction with context- and user-specific explanations. 

**Abstract (ZH)**: 当机器人在日常生活中执行复杂且依赖于上下文的任务时，偏离预期可能会使用户感到困惑。解释机器人的推理过程有助于用户理解机器人的意图。然而，何时提供解释以及解释的内容对于避免用户烦恼至关重要。我们研究了一种在厨房中协助日常清洁任务的机器人的解释需求和内容偏好。研究结果表明，用户希望在意外情况下获得解释，并偏好简洁明了的解释，这些解释清晰地陈述了令人困惑的行为背后的意图以及与此决策相关的上下文因素。基于这些发现，我们提出了两种算法来识别令人惊讶的行为并为Belief-Desire-Intention（BDI）机器人构建有效的解释。我们的算法可以很容易地集成到BDI推理过程中，并为具有上下文和用户特定解释的更好人机交互铺平道路。 

---
# ManifoldMind: Dynamic Hyperbolic Reasoning for Trustworthy Recommendations 

**Title (ZH)**: ManifoldMind: 动态双曲推理以实现可信推荐 

**Authors**: Anoushka Harit, Zhongtian Sun, Suncica Hadzidedic  

**Link**: [PDF](https://arxiv.org/pdf/2507.02014)  

**Abstract**: We introduce ManifoldMind, a probabilistic geometric recommender system for exploratory reasoning over semantic hierarchies in hyperbolic space. Unlike prior methods with fixed curvature and rigid embeddings, ManifoldMind represents users, items, and tags as adaptive-curvature probabilistic spheres, enabling personalised uncertainty modeling and geometry-aware semantic exploration. A curvature-aware semantic kernel supports soft, multi-hop inference, allowing the model to explore diverse conceptual paths instead of overfitting to shallow or direct interactions. Experiments on four public benchmarks show superior NDCG, calibration, and diversity compared to strong baselines. ManifoldMind produces explicit reasoning traces, enabling transparent, trustworthy, and exploration-driven recommendations in sparse or abstract domains. 

**Abstract (ZH)**: ManifoldMind：超越固定曲率的类流形概率几何推荐系统 

---
# Discovery of Fatigue Strength Models via Feature Engineering and automated eXplainable Machine Learning applied to the welded Transverse Stiffener 

**Title (ZH)**: 基于特征工程和自动可解释机器学习的焊接纵加劲条疲劳强度模型发现 

**Authors**: Michael A. Kraus, Helen Bartsch  

**Link**: [PDF](https://arxiv.org/pdf/2507.02005)  

**Abstract**: This research introduces a unified approach combining Automated Machine Learning (AutoML) with Explainable Artificial Intelligence (XAI) to predict fatigue strength in welded transverse stiffener details. It integrates expert-driven feature engineering with algorithmic feature creation to enhance accuracy and explainability.
Based on the extensive fatigue test database regression models - gradient boosting, random forests, and neural networks - were trained using AutoML under three feature schemes: domain-informed, algorithmic, and combined. This allowed a systematic comparison of expert-based versus automated feature selection.
Ensemble methods (e.g. CatBoost, LightGBM) delivered top performance. The domain-informed model $\mathcal M_2$ achieved the best balance: test RMSE $\approx$ 30.6 MPa and $R^2 \approx 0.780% over the full $\Delta \sigma_{c,50\%}$ range, and RMSE $\approx$ 13.4 MPa and $R^2 \approx 0.527% within the engineering-relevant 0 - 150 MPa domain. The denser-feature model ($\mathcal M_3$) showed minor gains during training but poorer generalization, while the simpler base-feature model ($\mathcal M_1$) performed comparably, confirming the robustness of minimalist designs.
XAI methods (SHAP and feature importance) identified stress ratio $R$, stress range $\Delta \sigma_i$, yield strength $R_{eH}$, and post-weld treatment (TIG dressing vs. as-welded) as dominant predictors. Secondary geometric factors - plate width, throat thickness, stiffener height - also significantly affected fatigue life.
This framework demonstrates that integrating AutoML with XAI yields accurate, interpretable, and robust fatigue strength models for welded steel structures. It bridges data-driven modeling with engineering validation, enabling AI-assisted design and assessment. Future work will explore probabilistic fatigue life modeling and integration into digital twin environments. 

**Abstract (ZH)**: 本研究提出了一种结合自动机器学习（AutoML）和可解释人工智能（XAI）的统一方法，用于预测焊接纵撑细节的疲劳强度。该方法将专家驱动的特征工程与算法特征创建相结合，以提高准确性和可解释性。
基于广泛的疲劳试验数据库，使用AutoML训练了回归模型——梯度提升、随机森林和神经网络，并采用了三种特征方案：领域驱动、算法驱动和结合驱动，从而系统地比较了基于专家和自动化的特征选择方法。
集成方法（如CatBoost、LightGBM）表现出最佳性能。领域驱动模型$\mathcal M_2$实现了最佳平衡：测试RMSE $\approx$ 30.6 MPa，$R^2 \approx 0.780\%$覆盖了全范围$\Delta \sigma_{c,50\%}$，在工程相关的0 - 150 MPa范围内，RMSE $\approx$ 13.4 MPa，$R^2 \approx 0.527\%$。特征更密集的模型（$\mathcal M_3$）在训练期间表现出小幅增益，但泛化能力较差，而基础特征更简单的模型（$\mathcal M_1$）表现相似，证实了简约设计的稳健性。
XAI方法（SHAP和特征重要性）确定了应力比$R$、应力范围$\Delta \sigma_i$、屈服强度$R_{eH}$和焊后处理（TIG打底 vs. 无焊后处理）为主导预测因素。次要几何因素—板宽、喉部厚度、撑条高度—也显著影响疲劳寿命。
该框架表明，将AutoML与XAI结合起来可以生成准确、可解释且稳健的焊接钢结构疲劳强度模型。它连接了数据驱动建模与工程验证，实现了AI辅助设计与评估。未来工作将探索概率疲劳寿命建模并将其集成到数字孪生环境中。 

---
# Towards a Playground to Democratize Experimentation and Benchmarking of AI Agents for Network Troubleshooting 

**Title (ZH)**: 朝着为网络故障排查中的AI代理的实验和基准测试 democratize 平台的方向 

**Authors**: Zhihao Wang, Alessandro Cornacchia, Franco Galante, Carlo Centofanti, Alessio Sacco, Dingde Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.01997)  

**Abstract**: Recent research has demonstrated the effectiveness of Artificial Intelligence (AI), and more specifically, Large Language Models (LLMs), in supporting network configuration synthesis and automating network diagnosis tasks, among others. In this preliminary work, we restrict our focus to the application of AI agents to network troubleshooting and elaborate on the need for a standardized, reproducible, and open benchmarking platform, where to build and evaluate AI agents with low operational effort. 

**Abstract (ZH)**: 近期的研究表明，人工智能（AI），特别是大规模语言模型（LLMs），在支持网络配置合成和网络故障诊断自动化等方面的有效性。在本初步工作中，我们将重点放在将AI代理应用于网络故障排除上，并强调需要一个标准化、可重复且开放的基准平台，以便以较低的操作成本构建和评估AI代理。 

---
# Integrating Large Language Models in Financial Investments and Market Analysis: A Survey 

**Title (ZH)**: 将大规模语言模型集成到金融投资和市场分析中：一个综述 

**Authors**: Sedigheh Mahdavi, Jiating, Chen, Pradeep Kumar Joshi, Lina Huertas Guativa, Upmanyu Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.01990)  

**Abstract**: Large Language Models (LLMs) have been employed in financial decision making, enhancing analytical capabilities for investment strategies. Traditional investment strategies often utilize quantitative models, fundamental analysis, and technical indicators. However, LLMs have introduced new capabilities to process and analyze large volumes of structured and unstructured data, extract meaningful insights, and enhance decision-making in real-time. This survey provides a structured overview of recent research on LLMs within the financial domain, categorizing research contributions into four main frameworks: LLM-based Frameworks and Pipelines, Hybrid Integration Methods, Fine-Tuning and Adaptation Approaches, and Agent-Based Architectures. This study provides a structured review of recent LLMs research on applications in stock selection, risk assessment, sentiment analysis, trading, and financial forecasting. By reviewing the existing literature, this study highlights the capabilities, challenges, and potential directions of LLMs in financial markets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在金融决策中的应用增强了投资策略的分析能力。传统投资策略通常利用量化模型、基本面分析和技术指标。然而，LLMs引入了处理和分析大量结构化和非结构化数据、提取有意义的洞察并实现实时决策的新能力。本文提供了一个结构化的综述，概述了金融领域中LLMs的最新研究，将研究贡献分为四大框架：基于LLM的框架和管道、混合集成方法、微调和适应方法以及基于代理的架构。本文提供了一篇结构化的综述，概述了近年来LLMs在股票选择、风险评估、情绪分析、交易和金融预测中的应用。通过对现有文献的回顾，本文突出了LLMs在金融市场中的能力和潜力。 

---
# DKGCM: A Spatio-Temporal Prediction Model for Traffic Flow by Fusing Spatial Node Clustering Method and Fourier Bidirectional Mamba Mechanism 

**Title (ZH)**: DKGCM：交通流的时空预测模型，通过融合空间节点聚类方法和傅里叶双向Mamba机制 

**Authors**: Siqing Long, Xiangzhi Huang, Jiemin Xie, Ming Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.01982)  

**Abstract**: Accurate traffic demand forecasting enables transportation management departments to allocate resources more effectively, thereby improving their utilization efficiency. However, complex spatiotemporal relationships in traffic systems continue to limit the performance of demand forecasting models. To improve the accuracy of spatiotemporal traffic demand prediction, we propose a new graph convolutional network structure called DKGCM. Specifically, we first consider the spatial flow distribution of different traffic nodes and propose a novel temporal similarity-based clustering graph convolution method, DK-GCN. This method utilizes Dynamic Time Warping (DTW) and K-means clustering to group traffic nodes and more effectively capture spatial dependencies. On the temporal scale, we integrate the Fast Fourier Transform (FFT) within the bidirectional Mamba deep learning framework to capture temporal dependencies in traffic demand. To further optimize model training, we incorporate the GRPO reinforcement learning strategy to enhance the loss function feedback mechanism. Extensive experiments demonstrate that our model outperforms several advanced methods and achieves strong results on three public datasets. 

**Abstract (ZH)**: 准确的交通需求预测能够使交通管理部门更有效地分配资源，进而提高资源利用效率。然而，交通系统中复杂的时空关系仍然限制了需求预测模型的性能。为提高时空交通需求预测的准确性，我们提出了一种新的图卷积网络结构，称为DKGCM。具体而言，我们首先考虑不同交通节点的时空流量分布，提出了一种基于时间相似性的新型聚类图卷积方法，即DK-GCN。该方法利用动态时间规整（DTW）和K-means聚类对交通节点进行分组，更有效地捕捉空间依赖性。在时间尺度上，我们在双向Mamba深度学习框架中集成了快速傅里叶变换（FFT），以捕捉交通需求的时间依赖性。为了进一步优化模型训练，我们将GRPO强化学习策略整合进损失函数反馈机制中。大量实验证明，我们的模型在多个公开数据集上均优于几种先进的方法，并取得了优异的结果。 

---
# Forecasting Labor Markets with LSTNet: A Multi-Scale Deep Learning Approach 

**Title (ZH)**: 基于多尺度深度学习方法的劳动力市场预测：LSTNet模型 

**Authors**: Adam Nelson-Archer, Aleia Sen, Meena Al Hasani, Sofia Davila, Jessica Le, Omar Abbouchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.01979)  

**Abstract**: We present a deep learning approach for forecasting short-term employment changes and assessing long-term industry health using labor market data from the U.S. Bureau of Labor Statistics. Our system leverages a Long- and Short-Term Time-series Network (LSTNet) to process multivariate time series data, including employment levels, wages, turnover rates, and job openings. The model outputs both 7-day employment forecasts and an interpretable Industry Employment Health Index (IEHI). Our approach outperforms baseline models across most sectors, particularly in stable industries, and demonstrates strong alignment between IEHI rankings and actual employment volatility. We discuss error patterns, sector-specific performance, and future directions for improving interpretability and generalization. 

**Abstract (ZH)**: 我们提出了一种基于深度学习的方法，使用美国劳工统计局的劳动力市场数据来预测短期就业变化并评估长期行业健康状况。该系统利用长短期时间序列网络（LSTNet）处理包括就业水平、工资、离职率和空缺职位等多变量时间序列数据。模型输出7天的就业预测和可解释的行业就业健康指数（IEHI）。该方法在大多数行业中优于基准模型，特别是在稳定行业中表现出色，IEHI排名与实际就业波动之间存在较强的正相关性。我们讨论了误差模式、行业特定性能以及提高可解释性和泛化能力的未来方向。 

---
# Learnable-Differentiable Finite Volume Solver for Accelerated Simulation of Flows 

**Title (ZH)**: 可学习可微分的有限体积求解器加速流场仿真 

**Authors**: Mengtao Yan, Qi Wang, Haining Wang, Ruizhi Chengze, Yi Zhang, Hongsheng Liu, Zidong Wang, Fan Yu, Qi Qi, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.01975)  

**Abstract**: Simulation of fluid flows is crucial for modeling physical phenomena like meteorology, aerodynamics, and biomedicine. Classical numerical solvers often require fine spatiotemporal grids to satisfy stability, consistency, and convergence conditions, leading to substantial computational costs. Although machine learning has demonstrated better efficiency, they typically suffer from issues of interpretability, generalizability, and data dependency. Hence, we propose a learnable and differentiable finite volume solver, called LDSolver, designed for efficient and accurate simulation of fluid flows on spatiotemporal coarse grids. LDSolver comprises two key components: (1) a differentiable finite volume solver, and (2) an learnable module providing equivalent approximation for fluxes (derivatives and interpolations), and temporal error correction on coarse grids. Even with limited training data (e.g., only a few trajectories), our model could accelerate the simulation while maintaining a high accuracy with superior generalizability. Experiments on different flow systems (e.g., Burgers, decaying, forced and shear flows) show that LDSolver achieves state-of-the-art performance, surpassing baseline models with notable margins. 

**Abstract (ZH)**: 时空粗网格下可学习且可微分的有限体积求解器用于流体流动高效准确模拟 

---
# Accelerated Portfolio Optimization and Option Pricing with Reinforcement Learning 

**Title (ZH)**: 加速组合优化和期权定价的强化学习方法 

**Authors**: Hadi Keramati, Samaneh Jazayeri  

**Link**: [PDF](https://arxiv.org/pdf/2507.01972)  

**Abstract**: We present a reinforcement learning (RL)-driven framework for optimizing block-preconditioner sizes in iterative solvers used in portfolio optimization and option pricing. The covariance matrix in portfolio optimization or the discretization of differential operators in option pricing models lead to large linear systems of the form $\mathbf{A}\textbf{x}=\textbf{b}$. Direct inversion of high-dimensional portfolio or fine-grid option pricing incurs a significant computational cost. Therefore, iterative methods are usually used for portfolios in real-world situations. Ill-conditioned systems, however, suffer from slow convergence. Traditional preconditioning techniques often require problem-specific parameter tuning. To overcome this limitation, we rely on RL to dynamically adjust the block-preconditioner sizes and accelerate iterative solver convergence. Evaluations on a suite of real-world portfolio optimization matrices demonstrate that our RL framework can be used to adjust preconditioning and significantly accelerate convergence and reduce computational cost. The proposed accelerated solver supports faster decision-making in dynamic portfolio allocation and real-time option pricing. 

**Abstract (ZH)**: 基于强化学习的迭代求解器中块预条件子大小优化框架 

---
# DeepSupp: Attention-Driven Correlation Pattern Analysis for Dynamic Time Series Support and Resistance Levels Identification 

**Title (ZH)**: DeepSupp：基于注意力机制的相关模式分析在动态时间序列支撑位和阻力位识别中的应用 

**Authors**: Boris Kriuk, Logic Ng, Zarif Al Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2507.01971)  

**Abstract**: Support and resistance (SR) levels are central to technical analysis, guiding traders in entry, exit, and risk management. Despite widespread use, traditional SR identification methods often fail to adapt to the complexities of modern, volatile markets. Recent research has introduced machine learning techniques to address the following challenges, yet most focus on price prediction rather than structural level identification. This paper presents DeepSupp, a new deep learning approach for detecting financial support levels using multi-head attention mechanisms to analyze spatial correlations and market microstructure relationships. DeepSupp integrates advanced feature engineering, constructing dynamic correlation matrices that capture evolving market relationships, and employs an attention-based autoencoder for robust representation learning. The final support levels are extracted through unsupervised clustering, leveraging DBSCAN to identify significant price thresholds. Comprehensive evaluations on S&P 500 tickers demonstrate that DeepSupp outperforms six baseline methods, achieving state-of-the-art performance across six financial metrics, including essential support accuracy and market regime sensitivity. With consistent results across diverse market conditions, DeepSupp addresses critical gaps in SR level detection, offering a scalable and reliable solution for modern financial analysis. Our approach highlights the potential of attention-based architectures to uncover nuanced market patterns and improve technical trading strategies. 

**Abstract (ZH)**: 基于多头注意力机制的DeepSupp：一种新的深度学习方法用于检测金融支撑水平 

---
# A Scalable and Quantum-Accurate Foundation Model for Biomolecular Force Field via Linearly Tensorized Quadrangle Attention 

**Title (ZH)**: 一种基于线性张量 quadrangle 注意机制的可扩展且量子精确的生物分子势场基础模型 

**Authors**: Qun Su, Kai Zhu, Qiaolin Gou, Jintu Zhang, Renling Hu, Yurong Li, Yongze Wang, Hui Zhang, Ziyi You, Linlong Jiang, Yu Kang, Jike Wang, Chang-Yu Hsieh, Tingjun Hou  

**Link**: [PDF](https://arxiv.org/pdf/2507.00884)  

**Abstract**: Accurate atomistic biomolecular simulations are vital for disease mechanism understanding, drug discovery, and biomaterial design, but existing simulation methods exhibit significant limitations. Classical force fields are efficient but lack accuracy for transition states and fine conformational details critical in many chemical and biological processes. Quantum Mechanics (QM) methods are highly accurate but computationally infeasible for large-scale or long-time simulations. AI-based force fields (AIFFs) aim to achieve QM-level accuracy with efficiency but struggle to balance many-body modeling complexity, accuracy, and speed, often constrained by limited training data and insufficient validation for generalizability. To overcome these challenges, we introduce LiTEN, a novel equivariant neural network with Tensorized Quadrangle Attention (TQA). TQA efficiently models three- and four-body interactions with linear complexity by reparameterizing high-order tensor features via vector operations, avoiding costly spherical harmonics. Building on LiTEN, LiTEN-FF is a robust AIFF foundation model, pre-trained on the extensive nablaDFT dataset for broad chemical generalization and fine-tuned on SPICE for accurate solvated system simulations. LiTEN achieves state-of-the-art (SOTA) performance across most evaluation subsets of rMD17, MD22, and Chignolin, outperforming leading models such as MACE, NequIP, and EquiFormer. LiTEN-FF enables the most comprehensive suite of downstream biomolecular modeling tasks to date, including QM-level conformer searches, geometry optimization, and free energy surface construction, while offering 10x faster inference than MACE-OFF for large biomolecules (~1000 atoms). In summary, we present a physically grounded, highly efficient framework that advances complex biomolecular modeling, providing a versatile foundation for drug discovery and related applications. 

**Abstract (ZH)**: 准确的原子尺度生物分子模拟对于疾病机制理解、药物发现和生物材料设计至关重要，但现有模拟方法存在显著局限性。经典力场高效但对过渡态和许多化学与生物过程中的细微构象细节缺乏准确性。量子力学（QM）方法高度准确但大规模或长时间模拟计算成本高昂。基于AI的力场(AIFF)旨在实现QM级准确性与效率，但难以平衡多体建模复杂性、准确性和速度，常受限于有限的训练数据和不足的一般化验证。为克服这些挑战，我们引入了LiTEN，这是一种新型具有张量四边形注意机制(TQA)的等变神经网络。TQA通过向量操作重新参数化高阶张量特征，以线性复杂度高效建模三体和四体相互作用，避免了昂贵的球谐变换。基于LiTEN，LiTEN-FF是稳健的AIFF基础模型，预训练于广泛的nablaDFT数据集以实现广泛的化学泛化，并在SPICE上进行微调以实现精确的溶剂化系统模拟。LiTEN在rMD17、MD22和Chignolin的大多数评估子集上实现了最先进的(SOTA)性能，超越了MACE、NequIP和EquiFormer等领先模型。LiTEN-FF使迄今为止最全面的下游生物分子建模任务组得以实现，包括QM级构象搜索、几何优化和自由能表面构造，同时对于大型生物分子（~1000个原子）的推理速度比MACE-OFF快10倍。总体而言，我们提出了一个物理上合理的、高效的方法框架，推动了复杂生物分子建模的发展，为其在药物发现及相关应用中的广泛应用提供了灵活的基础。 

---
