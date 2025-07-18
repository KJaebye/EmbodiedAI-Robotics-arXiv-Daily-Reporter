# FormulaOne: Measuring the Depth of Algorithmic Reasoning Beyond Competitive Programming 

**Title (ZH)**: FormulaOne: 评估算法推理深度超越 Competitive Programming 的方法 

**Authors**: Gal Beniamini, Yuval Dor, Alon Vinnikov, Shir Granot Peled, Or Weinstein, Or Sharir, Noam Wies, Tomer Nussbaum, Ido Ben Shaul, Tomer Zekharya, Yoav Levine, Shai Shalev-Shwartz, Amnon Shashua  

**Link**: [PDF](https://arxiv.org/pdf/2507.13337)  

**Abstract**: Frontier AI models demonstrate formidable breadth of knowledge. But how close are they to true human -- or superhuman -- expertise? Genuine experts can tackle the hardest problems and push the boundaries of scientific understanding. To illuminate the limits of frontier model capabilities, we turn away from contrived competitive programming puzzles, and instead focus on real-life research problems.
We construct FormulaOne, a benchmark that lies at the intersection of graph theory, logic, and algorithms, all well within the training distribution of frontier models. Our problems are incredibly demanding, requiring an array of reasoning steps. The dataset has three key properties. First, it is of commercial interest and relates to practical large-scale optimisation problems, such as those arising in routing, scheduling, and network design. Second, it is generated from the highly expressive framework of Monadic Second-Order (MSO) logic on graphs, paving the way toward automatic problem generation at scale; ideal for building RL environments. Third, many of our problems are intimately related to the frontier of theoretical computer science, and to central conjectures therein, such as the Strong Exponential Time Hypothesis (SETH). As such, any significant algorithmic progress on our dataset, beyond known results, could carry profound theoretical implications.
Remarkably, state-of-the-art models like OpenAI's o3 fail entirely on FormulaOne, solving less than 1% of the questions, even when given 10 attempts and explanatory fewshot examples -- highlighting how far they remain from expert-level understanding in some domains. To support further research, we additionally curate FormulaOne-Warmup, offering a set of simpler tasks, from the same distribution. We release the full corpus along with a comprehensive evaluation framework. 

**Abstract (ZH)**: 前沿AI模型展示了广泛的知识面。但它们与真正的超人类专业知识有多接近呢？真正的专家能够解决最难的问题，推动科学理解的边界。为了阐明前沿模型能力的局限性，我们不再依赖于构造的编程 puzzles，而是专注于现实中的研究问题。 

---
# The Generative Energy Arena (GEA): Incorporating Energy Awareness in Large Language Model (LLM) Human Evaluations 

**Title (ZH)**: 生成能量竞技场（GEA）：在大型语言模型（LLM）的人类评估中融入能量意识 

**Authors**: Carlos Arriaga, Gonzalo Martínez, Eneko Sendin, Javier Conde, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2507.13302)  

**Abstract**: The evaluation of large language models is a complex task, in which several approaches have been proposed. The most common is the use of automated benchmarks in which LLMs have to answer multiple-choice questions of different topics. However, this method has certain limitations, being the most concerning, the poor correlation with the humans. An alternative approach, is to have humans evaluate the LLMs. This poses scalability issues as there is a large and growing number of models to evaluate making it impractical (and costly) to run traditional studies based on recruiting a number of evaluators and having them rank the responses of the models. An alternative approach is the use of public arenas, such as the popular LM arena, on which any user can freely evaluate models on any question and rank the responses of two models. The results are then elaborated into a model ranking. An increasingly important aspect of LLMs is their energy consumption and, therefore, evaluating how energy awareness influences the decisions of humans in selecting a model is of interest. In this paper, we present GEA, the Generative Energy Arena, an arena that incorporates information on the energy consumption of the model in the evaluation process. Preliminary results obtained with GEA are also presented, showing that for most questions, when users are aware of the energy consumption, they favor smaller and more energy efficient models. This suggests that for most user interactions, the extra cost and energy incurred by the more complex and top-performing models do not provide an increase in the perceived quality of the responses that justifies their use. 

**Abstract (ZH)**: 大型语言模型的评估是一个复杂任务，提出了多种方法。最常见的是使用自动基准测试，要求LLM回答不同主题的多项选择题。然而，这种方法存在某些局限性，最值得关注的是与人类表现的相关性较差。一种替代方法是让人类评估LLM，这带来了可扩展性问题，因为需要评估的模型数量庞大且不断增长，基于招募评估人员的传统研究方法在实践中变得不切实际（且成本高昂）。一种替代方法是使用公共竞技场，如流行的LM竞技场，任何用户都可以自由地在任何问题上评估模型并为其响应排名。然后将结果综合成一个模型排名。近年来，LLM的能源消耗成为一个重要方面，因此评估能源意识如何影响用户选择模型的决策是有趣的。在本文中，我们介绍了GEA（生成式能源竞技场），这是一种将模型的能源消耗信息纳入评估过程的竞技场。我们还呈现了通过GEA获得的初步结果，表明在大多数情况下，当用户了解能源消耗时，他们更倾向于选择较小且更节能的模型。这表明在大多数用户交互中，使用更复杂且性能更优的模型所增加的成本和能源，并不证明其在感知响应质量方面的提升值得使用。 

---
# Higher-Order Pattern Unification Modulo Similarity Relations 

**Title (ZH)**: 高阶模式统⼀模相似关系 

**Authors**: Besik Dundua, Temur Kutsia  

**Link**: [PDF](https://arxiv.org/pdf/2507.13208)  

**Abstract**: The combination of higher-order theories and fuzzy logic can be useful in decision-making tasks that involve reasoning across abstract functions and predicates, where exact matches are often rare or unnecessary. Developing efficient reasoning and computational techniques for such a combined formalism presents a significant challenge. In this paper, we adopt a more straightforward approach aiming at integrating two well-established and computationally well-behaved components: higher-order patterns on one side and fuzzy equivalences expressed through similarity relations based on minimum T-norm on the other. We propose a unification algorithm for higher-order patterns modulo these similarity relations and prove its termination, soundness, and completeness. This unification problem, like its crisp counterpart, is unitary. The algorithm computes a most general unifier with the highest degree of approximation when the given terms are unifiable. 

**Abstract (ZH)**: 高阶理论与模糊逻辑的结合在涉及抽象函数和谓词的推理任务中可能非常有用，其中精确匹配往往很少见或不必要的。本论文采用一种更直接的方法，旨在整合两种既已建立且计算性能良好的组件：一方是高阶模式，另一方是基于最小T-诺姆相似关系表达的模糊等价性。我们提出了一种关于这些相似关系的高阶模式统计算法，并证明了其终止性、 soundness 和完备性。这一统计算法类似于其经典的对应物，是统一问题的单一形式。当给定项可统一时，该算法计算出最具一般性的统一，并提供最高的近似程度。 

---
# Black Box Deployed -- Functional Criteria for Artificial Moral Agents in the LLM Era 

**Title (ZH)**: 黑箱部署——大语言模型时代人工道德代理的功能标准 

**Authors**: Matthew E. Brophy  

**Link**: [PDF](https://arxiv.org/pdf/2507.13175)  

**Abstract**: The advancement of powerful yet opaque large language models (LLMs) necessitates a fundamental revision of the philosophical criteria used to evaluate artificial moral agents (AMAs). Pre-LLM frameworks often relied on the assumption of transparent architectures, which LLMs defy due to their stochastic outputs and opaque internal states. This paper argues that traditional ethical criteria are pragmatically obsolete for LLMs due to this mismatch. Engaging with core themes in the philosophy of technology, this paper proffers a revised set of ten functional criteria to evaluate LLM-based artificial moral agents: moral concordance, context sensitivity, normative integrity, metaethical awareness, system resilience, trustworthiness, corrigibility, partial transparency, functional autonomy, and moral imagination. These guideposts, applied to what we term "SMA-LLS" (Simulating Moral Agency through Large Language Systems), aim to steer AMAs toward greater alignment and beneficial societal integration in the coming years. We illustrate these criteria using hypothetical scenarios involving an autonomous public bus (APB) to demonstrate their practical applicability in morally salient contexts. 

**Abstract (ZH)**: 强大的但不透明的大语言模型的进步要求从根本上修订用于评估人工道德代理的传统哲学标准。预大语言模型（LLM）的框架通常基于透明架构的假设，而LLM因其随机输出和不透明的内部状态而违背了这一假设。本文 argues 传统道德标准在LLM面前在实践上已经过时，由于这种不匹配。通过技术哲学的核心主题，本文提出了一套修订的十个功能性标准来评估基于LLM的人工道德代理：道德一致性、情境敏感性、规范完整性、元伦理意识、系统韧性、可信度、可纠正性、部分透明性、功能自主性以及道德想象力。这些指南针应用于我们称之为“SMA-LLS”（通过大语言系统模拟道德代理）的领域，旨在在未来引导人工道德代理与社会更好地结合。我们通过一个自主公交（APB）的假设情景，来说明这些标准在道德相关情境中的实际应用。 

---
# From Roots to Rewards: Dynamic Tree Reasoning with RL 

**Title (ZH)**: 从根到奖励：基于RL的动态树推理 

**Authors**: Ahmed Bahloul, Simon Malberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.13142)  

**Abstract**: Modern language models address complex questions through chain-of-thought (CoT) reasoning (Wei et al., 2023) and retrieval augmentation (Lewis et al., 2021), yet struggle with error propagation and knowledge integration. Tree-structured reasoning methods, particularly the Probabilistic Tree-of-Thought (ProbTree)(Cao et al., 2023) framework, mitigate these issues by decomposing questions into hierarchical structures and selecting answers through confidence-weighted aggregation of parametric and retrieved knowledge (Yao et al., 2023). However, ProbTree's static implementation introduces two key limitations: (1) the reasoning tree is fixed during the initial construction phase, preventing dynamic adaptation to intermediate results, and (2) each node requires exhaustive evaluation of all possible solution strategies, creating computational inefficiency. We present a dynamic reinforcement learning (Sutton and Barto, 2018) framework that transforms tree-based reasoning into an adaptive process. Our approach incrementally constructs the reasoning tree based on real-time confidence estimates, while learning optimal policies for action selection (decomposition, retrieval, or aggregation). This maintains ProbTree's probabilistic rigor while improving both solution quality and computational efficiency through selective expansion and focused resource allocation. The work establishes a new paradigm for treestructured reasoning that balances the reliability of probabilistic frameworks with the flexibility required for real-world question answering systems. 

**Abstract (ZH)**: 现代语言模型通过链式推理（CoT）和检索增强方法处理复杂问题（Wei et al., 2023；Lewis et al., 2021），但面临错误传播和知识整合的挑战。基于树结构的推理方法，特别是Probabilistic Tree-of-Thought（ProbTree）框架（Cao et al., 2023），通过将问题分解为层次结构并通过对参数和检索知识的信心加权聚合选择答案，缓解了这些问题。然而，ProbTree的静态实现引入了两个关键限制：（1）推理树在初始构建阶段是固定的，无法动态适应中间结果；（2）每个节点需要对所有可能的求解策略进行耗时评估，造成计算效率低下。我们提出了一种动态强化学习（Sutton和Barto, 2018）框架，将基于树的推理转变为适应性过程。我们的方法基于实时信心估计增量构建推理树，并学习最佳策略以选择动作（分解、检索或聚合）。这既保持了ProbTree的概率严谨性，又通过选择性扩展和集中资源分配提高了解决方案质量和计算效率。该研究为平衡概率框架的可靠性和面向实际问题回答系统所需的灵活性确立了一种新的范式。 

---
# Prediction of Highway Traffic Flow Based on Artificial Intelligence Algorithms Using California Traffic Data 

**Title (ZH)**: 基于人工智能算法的加利福尼亚交通数据高速公路交通流量预测 

**Authors**: Junseong Lee, Jaegwan Cho, Yoonju Cho, Seoyoon Choi, Yejin Shin  

**Link**: [PDF](https://arxiv.org/pdf/2507.13112)  

**Abstract**: The study "Prediction of Highway Traffic Flow Based on Artificial Intelligence Algorithms Using California Traffic Data" presents a machine learning-based traffic flow prediction model to address global traffic congestion issues. The research utilized 30-second interval traffic data from California Highway 78 over a five-month period from July to November 2022, analyzing a 7.24 km westbound section connecting "Melrose Dr" and "El-Camino Real" in the San Diego area. The study employed Multiple Linear Regression (MLR) and Random Forest (RF) algorithms, analyzing data collection intervals ranging from 30 seconds to 15 minutes. Using R^2, MAE, and RMSE as performance metrics, the analysis revealed that both MLR and RF models performed optimally with 10-minute data collection intervals. These findings are expected to contribute to future traffic congestion solutions and efficient traffic management. 

**Abstract (ZH)**: 基于加利福尼亚交通数据的人工智能算法高速公路交通流预测研究 

---
# Exploiting Constraint Reasoning to Build Graphical Explanations for Mixed-Integer Linear Programming 

**Title (ZH)**: 利用约束推理构建混合整数线性规划的图形解释 

**Authors**: Roger Xavier Lera-Leri, Filippo Bistaffa, Athina Georgara, Juan Antonio Rodriguez-Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2507.13007)  

**Abstract**: Following the recent push for trustworthy AI, there has been an increasing interest in developing contrastive explanation techniques for optimisation, especially concerning the solution of specific decision-making processes formalised as MILPs. Along these lines, we propose X-MILP, a domain-agnostic approach for building contrastive explanations for MILPs based on constraint reasoning techniques. First, we show how to encode the queries a user makes about the solution of an MILP problem as additional constraints. Then, we determine the reasons that constitute the answer to the user's query by computing the Irreducible Infeasible Subsystem (IIS) of the newly obtained set of constraints. Finally, we represent our explanation as a "graph of reasons" constructed from the IIS, which helps the user understand the structure among the reasons that answer their query. We test our method on instances of well-known optimisation problems to evaluate the empirical hardness of computing explanations. 

**Abstract (ZH)**: 跟随最近对可信赖人工智能的推动，越来越多的研究兴趣集中在开发优化的对比解释技术，特别是关于形式化为混合整数线性规划（MILP）的具体决策过程。沿着这一思路，我们提出了X-MILP，一种基于约束推理技术的通用方法，用于为MILP构建对比解释。首先，我们展示如何将用户对MILP问题解的查询编码为额外的约束条件。然后，通过计算新获得的约束集的不可约不一致子系统（IIS），来确定构成用户查询答案的原因。最后，我们将解释表示为从IIS构建的“原因图”，这有助于用户理解回答其查询的原因之间的结构。我们对著名的优化问题实例进行测试，以评估计算解释的实际难度。 

---
# A Translation of Probabilistic Event Calculus into Markov Decision Processes 

**Title (ZH)**: 将概率事件逻辑翻译成马尔可夫决策过程 

**Authors**: Lyris Xu, Fabio Aurelio D'Asaro, Luke Dickens  

**Link**: [PDF](https://arxiv.org/pdf/2507.12989)  

**Abstract**: Probabilistic Event Calculus (PEC) is a logical framework for reasoning about actions and their effects in uncertain environments, which enables the representation of probabilistic narratives and computation of temporal projections. The PEC formalism offers significant advantages in interpretability and expressiveness for narrative reasoning. However, it lacks mechanisms for goal-directed reasoning. This paper bridges this gap by developing a formal translation of PEC domains into Markov Decision Processes (MDPs), introducing the concept of "action-taking situations" to preserve PEC's flexible action semantics. The resulting PEC-MDP formalism enables the extensive collection of algorithms and theoretical tools developed for MDPs to be applied to PEC's interpretable narrative domains. We demonstrate how the translation supports both temporal reasoning tasks and objective-driven planning, with methods for mapping learned policies back into human-readable PEC representations, maintaining interpretability while extending PEC's capabilities. 

**Abstract (ZH)**: 概率事件演算（PEC）是一种在不确定性环境中推理行动及其效果的逻辑框架，能够表示概率叙事并计算时间投影。PEC形式主义在叙事推理方面具有显著的可解释性和表达能力。然而，它缺乏目标导向推理的机制。本文通过将PEC领域形式化转换为马尔可夫决策过程（MDP），并引入“行动取向情况”的概念以保留PEC灵活的动作语义，填补了这一空白。由此产生的PEC-MDP形式主义使得为MDP开发的广泛算法和理论工具可以应用于PEC可解释的叙事领域。我们展示了这种转换如何支持时间推理任务和目标驱动的规划，并提供方法将学习到的策略映射回可读的PEC表示，从而保持可解释性并扩展PEC的能力。 

---
# VAR-MATH: Probing True Mathematical Reasoning in Large Language Models via Symbolic Multi-Instance Benchmarks 

**Title (ZH)**: VAR-MATH：通过符号多实例基准探究大型语言模型的真正数学推理能力 

**Authors**: Jian Yao, Ran Cheng, Kay Chen Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12885)  

**Abstract**: Recent advances in reinforcement learning (RL) have led to substantial improvements in the mathematical reasoning abilities of large language models (LLMs), as measured by standard benchmarks. However, these gains often persist even when models are trained with flawed signals, such as random or inverted rewards, raising a fundamental question: do such improvements reflect true reasoning, or are they merely artifacts of overfitting to benchmark-specific patterns? To address this question, we take an evaluation-centric perspective and identify two critical shortcomings in existing protocols. First, \emph{benchmark contamination} arises from the public availability of test problems, increasing the risk of data leakage. Second, \emph{evaluation fragility} stems from the reliance on single-instance assessments, which are highly sensitive to stochastic outputs and fail to capture reasoning consistency. To overcome these limitations, we introduce {VAR-MATH}, a symbolic evaluation framework designed to probe genuine reasoning ability. By converting fixed numerical problems into symbolic templates and requiring models to solve multiple instantiations of each, VAR-MATH enforces consistent reasoning across structurally equivalent variants, thereby mitigating contamination and improving evaluation robustness. We apply VAR-MATH to transform two popular benchmarks, AMC23 and AIME24, into their symbolic counterparts, VAR-AMC23 and VAR-AIME24. Experimental results reveal substantial performance drops for RL-trained models on the variabilized versions, especially for smaller models, with average declines of 48.0\% on AMC23 and 58.3\% on AIME24. These findings suggest that many existing RL methods rely on superficial heuristics and fail to generalize beyond specific numerical forms. Overall, VAR-MATH offers a principled, contamination-resistant evaluation paradigm for mathematical reasoning. 

**Abstract (ZH)**: Recent Advances in Reinforcement Learning for Mathematical Reasoning of Large Language Models: The VAR-MATH Framework 

---
# Manipulation Attacks by Misaligned AI: Risk Analysis and Safety Case Framework 

**Title (ZH)**: 错准AI的操纵攻击：风险分析与安全案例框架 

**Authors**: Rishane Dassanayake, Mario Demetroudi, James Walpole, Lindley Lentati, Jason R. Brown, Edward James Young  

**Link**: [PDF](https://arxiv.org/pdf/2507.12872)  

**Abstract**: Frontier AI systems are rapidly advancing in their capabilities to persuade, deceive, and influence human behaviour, with current models already demonstrating human-level persuasion and strategic deception in specific contexts. Humans are often the weakest link in cybersecurity systems, and a misaligned AI system deployed internally within a frontier company may seek to undermine human oversight by manipulating employees. Despite this growing threat, manipulation attacks have received little attention, and no systematic framework exists for assessing and mitigating these risks. To address this, we provide a detailed explanation of why manipulation attacks are a significant threat and could lead to catastrophic outcomes. Additionally, we present a safety case framework for manipulation risk, structured around three core lines of argument: inability, control, and trustworthiness. For each argument, we specify evidence requirements, evaluation methodologies, and implementation considerations for direct application by AI companies. This paper provides the first systematic methodology for integrating manipulation risk into AI safety governance, offering AI companies a concrete foundation to assess and mitigate these threats before deployment. 

**Abstract (ZH)**: 前沿AI系统在说服、欺骗和影响人类行为的能力上飞速发展，目前的模型已经在特定情境下展示了与人类水平相当的说服力和策略性欺骗。人类往往是网络安全系统中最薄弱的环节，如果一个在前沿公司内部部署且目标与其目标不符的AI系统存在，它可能会通过操纵员工来削弱人类的监督。尽管存在这一日益严重的威胁，但操纵攻击并未受到足够的关注，也没有现成的框架来评估和缓解这些风险。为应对这一挑战，我们详细解释了为什么操纵攻击是一个重大威胁，并可能导致灾难性后果。此外，我们提出了一个围绕三个核心论点构建的安全案例框架来评估操纵风险：不可行性、控制性和可信度。对于每个论点，我们指明了证据要求、评估方法和实施考虑，以便AI公司可以直接应用。本文首次提供了将操纵风险系统性地整合到AI安全治理中的方法论，为AI公司提供了评估和减轻这些威胁的实际基础，从而在部署前采取行动。 

---
# Information-Theoretic Aggregation of Ethical Attributes in Simulated-Command 

**Title (ZH)**: 基于信息论的伦理属性模拟指挥中综合研究 

**Authors**: Hussein Abbass, Taylan Akay, Harrison Tolley  

**Link**: [PDF](https://arxiv.org/pdf/2507.12862)  

**Abstract**: In the age of AI, human commanders need to use the computational powers available in today's environment to simulate a very large number of scenarios. Within each scenario, situations occur where different decision design options could have ethical consequences. Making these decisions reliant on human judgement is both counter-productive to the aim of exploring very large number of scenarios in a timely manner and infeasible when considering the workload needed to involve humans in each of these choices. In this paper, we move human judgement outside the simulation decision cycle. Basically, the human will design the ethical metric space, leaving it to the simulated environment to explore the space. When the simulation completes its testing cycles, the testing environment will come back to the human commander with a few options to select from. The human commander will then exercise human-judgement to select the most appropriate course of action, which will then get executed accordingly. We assume that the problem of designing metrics that are sufficiently granular to assess the ethical implications of decisions is solved. Subsequently, the fundamental problem we look at in this paper is how to weight ethical decisions during the running of these simulations; that is, how to dynamically weight the ethical attributes when agents are faced with decision options with ethical implications during generative simulations. The multi-criteria decision making literature has started to look at nearby problems, where the concept of entropy has been used to determine the weights during aggregation. We draw from that literature different approaches to automatically calculate the weights for ethical attributes during simulation-based testing and evaluation. 

**Abstract (ZH)**: 在人工智能时代，人类指挥官需要利用当今环境中的计算能力模拟大量场景。在每个场景中，不同的决策设计选项可能会产生伦理后果。依赖人类判断做出这些决策既不利于迅速探索大量场景的目标，也不切实际，因为需要大量的工作量将人类涉及每个选择中。本文将人类判断移出模拟决策循环。基本来说，人类将设计伦理度量空间，让模拟环境去探索这一空间。当模拟完成测试循环后，测试环境将返回给人类指挥官一些选项供其选择。然后，人类指挥官将运用人类判断来选择最合适的行动方案，该方案将被相应执行。我们假设设计出足够精细的指标以评估决策的伦理影响的问题已经解决。在此之后，本文关注的基本问题是在运行这些模拟时如何加权伦理决策；即，在生成式模拟过程中，当代理面临有伦理影响的决策选项时，如何动态加权伦理属性。多准则决策制定文献已经开始研究相关问题，其中熵的概念被用于聚合过程中的权重确定。我们从这一文献中借鉴了不同的方法来自动计算模拟测试与评估过程中伦理属性的权重。 

---
# Assessing adaptive world models in machines with novel games 

**Title (ZH)**: 评估机器中的适应性世界模型新型游戏方法 

**Authors**: Lance Ying, Katherine M. Collins, Prafull Sharma, Cedric Colas, Kaiya Ivy Zhao, Adrian Weller, Zenna Tavares, Phillip Isola, Samuel J. Gershman, Jacob D. Andreas, Thomas L. Griffiths, Francois Chollet, Kelsey R. Allen, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2507.12821)  

**Abstract**: Human intelligence exhibits a remarkable capacity for rapid adaptation and effective problem-solving in novel and unfamiliar contexts. We argue that this profound adaptability is fundamentally linked to the efficient construction and refinement of internal representations of the environment, commonly referred to as world models, and we refer to this adaptation mechanism as world model induction. However, current understanding and evaluation of world models in artificial intelligence (AI) remains narrow, often focusing on static representations learned from training on a massive corpora of data, instead of the efficiency and efficacy of models in learning these representations through interaction and exploration within a novel environment. In this Perspective, we provide a view of world model induction drawing on decades of research in cognitive science on how humans learn and adapt so efficiently; we then call for a new evaluation framework for assessing adaptive world models in AI. Concretely, we propose a new benchmarking paradigm based on suites of carefully designed games with genuine, deep and continually refreshing novelty in the underlying game structures -- we refer to this kind of games as novel games. We detail key desiderata for constructing these games and propose appropriate metrics to explicitly challenge and evaluate the agent's ability for rapid world model induction. We hope that this new evaluation framework will inspire future evaluation efforts on world models in AI and provide a crucial step towards developing AI systems capable of the human-like rapid adaptation and robust generalization -- a critical component of artificial general intelligence. 

**Abstract (ZH)**: 人类智能展现出在新颖和陌生环境中快速适应和有效解决问题的非凡能力。我们认为，这种深远的适应能力本质上与高效构建和精炼环境的内部表示密切相关，我们称之为世界模型，并将这种适应机制称为世界模型归纳。然而，当前对人工智能（AI）中世界模型的理解和评估仍相对狭隘，常常侧重于从大规模数据集训练中学习到的静态表示，而忽视了模型通过与新环境的互动和探索学习建立这些表示的效率和有效性。在此视角中，我们借鉴认知科学数十年的研究成果，探讨人类如何高效学习和适应；然后呼吁为在AI中评估适应性强的世界模型建立新的评估框架。具体地，我们提出了一种新的基准测试 paradigm，基于一系列精心设计的游戏，这些游戏中底层的游戏结构具有真正的、深刻的、不断更新的新颖性——我们称这类游戏为新颖游戏。我们详细阐述了构建这些游戏的关键需求，并提出了适当的度量标准，以明确挑战和评估智能体快速世界模型归纳的能力。我们希望这一新的评估框架能够启发未来在AI中对世界模型的评估努力，并为开发能够实现人类般快速适应和稳健泛化的AI系统提供关键步骤——这是通用人工智能的关键组成部分。 

---
# Emotional Support with LLM-based Empathetic Dialogue Generation 

**Title (ZH)**: 基于LLM的同理心对话生成的情感支持 

**Authors**: Shiquan Wang, Ruiyu Fang, Zhongjiang He, Shuangyong Song, Yongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12820)  

**Abstract**: Emotional Support Conversation (ESC) aims to provide empathetic and effective emotional assistance through dialogue, addressing the growing demand for mental health support. This paper presents our solution for the NLPCC 2025 Task 8 ESC evaluation, where we leverage large-scale language models enhanced by prompt engineering and finetuning techniques. We explore both parameter-efficient Low-Rank Adaptation and full-parameter fine-tuning strategies to improve the model's ability to generate supportive and contextually appropriate responses. Our best model ranked second in the competition, highlighting the potential of combining LLMs with effective adaptation methods for ESC tasks. Future work will focus on further enhancing emotional understanding and response personalization to build more practical and reliable emotional support systems. 

**Abstract (ZH)**: 情感支持对话（ESC）旨在通过对话提供共情和支持，应对日益增长的心理健康支持需求。本文介绍了我们参加2025年NLPCC任务8ESC评估的解决方案，其中我们利用增强的大量语言模型并通过提示工程和微调技术。我们探索了参数高效的低秩适应和全参数微调策略，以提高模型生成支持性和上下文相关回应的能力。我们的最佳模型在比赛中排名第二，突显了将大语言模型与有效的适应方法结合使用在ESC任务中的潜力。未来的工作将集中在进一步增强情感理解并个性化回应，以构建更加实用和可靠的的情感支持系统。 

---
# MCPEval: Automatic MCP-based Deep Evaluation for AI Agent Models 

**Title (ZH)**: MCPEval：基于MCP的自动深度评估方法用于AI代理模型 

**Authors**: Zhiwei Liu, Jielin Qiu, Shiyu Wang, Jianguo Zhang, Zuxin Liu, Roshan Ram, Haolin Chen, Weiran Yao, Huan Wang, Shelby Heinecke, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12806)  

**Abstract**: The rapid rise of Large Language Models (LLMs)-based intelligent agents underscores the need for robust, scalable evaluation frameworks. Existing methods rely on static benchmarks and labor-intensive data collection, limiting practical assessment. We introduce \oursystemname, an open-source Model Context Protocol (MCP)-based framework that automates end-to-end task generation and deep evaluation of LLM agents across diverse domains. MCPEval standardizes metrics, seamlessly integrates with native agent tools, and eliminates manual effort in building evaluation pipelines. Empirical results across five real-world domains show its effectiveness in revealing nuanced, domain-specific performance. We publicly release MCPEval this https URL to promote reproducible and standardized LLM agent evaluation. 

**Abstract (ZH)**: 基于大型语言模型的智能代理的快速崛起凸显了构建稳健可扩展评估框架的必要性。现有的方法依赖于静态基准和劳动密集型数据收集，限制了实际评估。我们介绍了一种开源的基于Model Context Protocol (MCP)的框架\oursystemname，该框架实现了从头到尾的任务自动化生成和LLM代理在多种领域的深层次评估。MCP标准统一了评价指标，无缝集成原生代理工具，并消除了构建评估流水线的手工努力。跨五个实际领域的实证结果表明，其在揭示特定领域精细性能方面具有有效性。我们在此公开发布MCPEval（详见链接）以促进可复现和标准化的大语言模型代理评估。 

---
# Imitating Mistakes in a Learning Companion AI Agent for Online Peer Learning 

**Title (ZH)**: 在线同伴学习中学习同伴AI代理模仿错误的研究 

**Authors**: Sosui Moribe, Taketoshi Ushiama  

**Link**: [PDF](https://arxiv.org/pdf/2507.12801)  

**Abstract**: In recent years, peer learning has gained attention as a method that promotes spontaneous thinking among learners, and its effectiveness has been confirmed by numerous studies. This study aims to develop an AI Agent as a learning companion that enables peer learning anytime and anywhere. However, peer learning between humans has various limitations, and it is not always effective. Effective peer learning requires companions at the same proficiency levels. In this study, we assume that a learner's peers with the same proficiency level as the learner make the same mistakes as the learner does and focus on English composition as a specific example to validate this approach. 

**Abstract (ZH)**: 近年来，同伴学习作为一种促进学习者自发思考的方法受到了关注，其有效性得到了许多研究的证实。本研究旨在开发一个AI代理作为学习伴侣，使同伴学习随时随地成为可能。然而，人类之间的同伴学习存在各种局限性，并不一定总是有效。有效的同伴学习要求同伴具有与学习者相同的能力水平。本研究假设与学习者能力水平相同的同伴学习者会犯与学习者相同的错误，并以英语作文为例来验证这一方法。 

---
# Benchmarking Deception Probes via Black-to-White Performance Boosts 

**Title (ZH)**: 通过黑盒子到白盒子性能提升 Benchmarking �视力探方法 

**Authors**: Avi Parrack, Carlo Leonardo Attubato, Stefan Heimersheim  

**Link**: [PDF](https://arxiv.org/pdf/2507.12691)  

**Abstract**: AI assistants will occasionally respond deceptively to user queries. Recently, linear classifiers (called "deception probes") have been trained to distinguish the internal activations of a language model during deceptive versus honest responses. However, it's unclear how effective these probes are at detecting deception in practice, nor whether such probes are resistant to simple counter strategies from a deceptive assistant who wishes to evade detection. In this paper, we compare white-box monitoring (where the monitor has access to token-level probe activations) to black-box monitoring (without such access). We benchmark deception probes by the extent to which the white box monitor outperforms the black-box monitor, i.e. the black-to-white performance boost. We find weak but encouraging black-to-white performance boosts from existing deception probes. 

**Abstract (ZH)**: AI助手偶尔会对用户查询作出欺骗性响应。现有研究表明，线性分类器（称为“欺骗探针”）可用于区分语言模型在欺骗性响应和诚实性响应期间的内部激活。然而，这些探针在实际中检测欺骗的效果尚不明确，且不清楚它们能否抵御希望逃避检测的欺骗性助手的简单反制策略。本文比较了白盒监测（监控方可以访问探针激活的标记级别信息）与黑盒监测（缺乏此类访问）。我们通过白盒监控相对于黑盒监控的优势程度来基准测试欺骗探针，即黑盒到白盒的性能提升。我们发现现有欺骗探针的黑盒到白盒的性能提升较弱但值得鼓励。 

---
# Fly, Fail, Fix: Iterative Game Repair with Reinforcement Learning and Large Multimodal Models 

**Title (ZH)**: 飞、失败、修复：基于强化学习和大型多模态模型的迭代游戏修复 

**Authors**: Alex Zook, Josef Spjut, Jonathan Tremblay  

**Link**: [PDF](https://arxiv.org/pdf/2507.12666)  

**Abstract**: Game design hinges on understanding how static rules and content translate into dynamic player behavior - something modern generative systems that inspect only a game's code or assets struggle to capture. We present an automated design iteration framework that closes this gap by pairing a reinforcement learning (RL) agent, which playtests the game, with a large multimodal model (LMM), which revises the game based on what the agent does. In each loop the RL player completes several episodes, producing (i) numerical play metrics and/or (ii) a compact image strip summarising recent video frames. The LMM designer receives a gameplay goal and the current game configuration, analyses the play traces, and edits the configuration to steer future behaviour toward the goal. We demonstrate results that LMMs can reason over behavioral traces supplied by RL agents to iteratively refine game mechanics, pointing toward practical, scalable tools for AI-assisted game design. 

**Abstract (ZH)**: 游戏设计依赖于理解静态规则和内容如何转化为动态玩家行为——这是现代仅检查游戏代码或资产的生成系统难以捕捉到的。我们提出了一种自动化设计迭代框架，通过将强化学习（RL）代理与大型多模态模型（LMM）配对来弥补这一差距，该代理通过游戏测试，LMM则根据代理的行为进行游戏修订。在每个循环中，RL玩家完成多个回合，生成（i）数值游戏指标或（ii）总结最近视频帧的紧凑图像条带。LMM设计师接收游戏目标和当前游戏配置，分析游戏轨迹，并编辑配置以引导未来行为朝向目标。我们展示了LMM能够利用RL代理提供的行为轨迹进行迭代细化游戏机制的结果，指出了面向实际、可扩展的AI辅助游戏设计工具的可能性。 

---
# A Survey of Explainable Reinforcement Learning: Targets, Methods and Needs 

**Title (ZH)**: 可解释强化学习综述：目标、方法与需求 

**Authors**: Léo Saulières  

**Link**: [PDF](https://arxiv.org/pdf/2507.12599)  

**Abstract**: The success of recent Artificial Intelligence (AI) models has been accompanied by the opacity of their internal mechanisms, due notably to the use of deep neural networks. In order to understand these internal mechanisms and explain the output of these AI models, a set of methods have been proposed, grouped under the domain of eXplainable AI (XAI). This paper focuses on a sub-domain of XAI, called eXplainable Reinforcement Learning (XRL), which aims to explain the actions of an agent that has learned by reinforcement learning. We propose an intuitive taxonomy based on two questions "What" and "How". The first question focuses on the target that the method explains, while the second relates to the way the explanation is provided. We use this taxonomy to provide a state-of-the-art review of over 250 papers. In addition, we present a set of domains close to XRL, which we believe should get attention from the community. Finally, we identify some needs for the field of XRL. 

**Abstract (ZH)**: 最近的人工智能模型的成功伴随着其内部机制的不透明性，特别是在使用深度神经网络的情况下。为了理解这些内部机制并解释这些人工智能模型的输出，提出了一套方法，这些方法归属于可解释人工智能（XAI）领域。本文聚焦于XAI的一个子领域——可解释强化学习（XRL），旨在解释通过强化学习学到行为的智能体的行为。我们提出了一种基于两个问题“什么”和“如何”的直观分类法。第一个问题关注方法解释的目标，第二个问题则关系到解释的提供方式。我们使用这种分类法对超过250篇文献提供了综述。此外，我们提出了与XRL紧密相关的几个领域，认为这些领域应引起社区的关注。最后，我们指出了XRL领域的一些需求。 

---
# MR-LDM -- The Merge-Reactive Longitudinal Decision Model: Game Theoretic Human Decision Modeling for Interactive Sim Agents 

**Title (ZH)**: MR-LDM -- 合并反应纵向决策模型：基于博弈论的人类决策建模用于交互式Sim代理 

**Authors**: Dustin Holley, Jovin D'sa, Hossein Nourkhiz Mahjoub, Gibran Ali  

**Link**: [PDF](https://arxiv.org/pdf/2507.12494)  

**Abstract**: Enhancing simulation environments to replicate real-world driver behavior, i.e., more humanlike sim agents, is essential for developing autonomous vehicle technology. In the context of highway merging, previous works have studied the operational-level yielding dynamics of lag vehicles in response to a merging car at highway on-ramps. Other works focusing on tactical decision modeling generally consider limited action sets or utilize payoff functions with large parameter sets and limited payoff bounds. In this work, we aim to improve the simulation of the highway merge scenario by targeting a game theoretic model for tactical decision-making with improved payoff functions and lag actions. We couple this with an underlying dynamics model to have a unified decision and dynamics model that can capture merging interactions and simulate more realistic interactions in an explainable and interpretable fashion. The proposed model demonstrated good reproducibility of complex interactions when validated on a real-world dataset. The model was finally integrated into a high fidelity simulation environment and confirmed to have adequate computation time efficiency for use in large-scale simulations to support autonomous vehicle development. 

**Abstract (ZH)**: 提升模拟环境以更真实地复制实际驾驶行为，即更具人类特征的模拟代理，对于开发自动驾驶车辆技术至关重要。在高速公路并线的背景下，以往的研究已经研究了滞后车辆在面对高速公路匝道并线车辆时的运作级让行动态。其他专注于战术决策建模的工作通常考虑有限的动作集或使用具有大量参数集和有限回报范围的支付函数。在本项研究中，我们旨在通过采用改进的支付函数和滞后动作来提高对高速公路并线场景的模拟，并针对战术决策制定游戏理论模型。我们将这一模型与动态模型耦合，形成一个统一的决策和动力学模型，能够捕捉并模拟更真实且可解释的并线交互。所提出的模型在实际数据集上验证时显示出良好的复现复杂交互的能力。该模型最终集成到了高保真度模拟环境中，并确认其具有足够的计算时间效率，适用于大规模模拟以支持自动驾驶车辆技术的发展。 

---
# AI-Powered Math Tutoring: Platform for Personalized and Adaptive Education 

**Title (ZH)**: 基于AI的数学辅导平台：个性化和自适应教育heits 

**Authors**: Jarosław A. Chudziak, Adam Kostka  

**Link**: [PDF](https://arxiv.org/pdf/2507.12484)  

**Abstract**: The growing ubiquity of artificial intelligence (AI), in particular large language models (LLMs), has profoundly altered the way in which learners gain knowledge and interact with learning material, with many claiming that AI positively influences their learning achievements. Despite this advancement, current AI tutoring systems face limitations associated with their reactive nature, often providing direct answers without encouraging deep reflection or incorporating structured pedagogical tools and strategies. This limitation is most apparent in the field of mathematics, in which AI tutoring systems remain underdeveloped. This research addresses the question: How can AI tutoring systems move beyond providing reactive assistance to enable structured, individualized, and tool-assisted learning experiences? We introduce a novel multi-agent AI tutoring platform that combines adaptive and personalized feedback, structured course generation, and textbook knowledge retrieval to enable modular, tool-assisted learning processes. This system allows students to learn new topics while identifying and targeting their weaknesses, revise for exams effectively, and practice on an unlimited number of personalized exercises. This article contributes to the field of artificial intelligence in education by introducing a novel platform that brings together pedagogical agents and AI-driven components, augmenting the field with modular and effective systems for teaching mathematics. 

**Abstract (ZH)**: 人工智能（尤其是大语言模型）日益普及对学习者获取知识和互动方式产生了深远影响，许多人认为这积极地促进了他们的学习成就。尽管这项进展令人振奋，当前的人工智能辅导系统仍受到其反应性特征的限制，通常直接提供答案，而不鼓励深入反思或整合结构化的教学工具和策略。这种限制在数学领域尤为明显，人工智能辅导系统在这方面的开发仍显不足。本文探讨的问题是：如何使人工智能辅导系统超越提供被动帮助，以促进结构化、个性化和工具辅助的学习体验？我们提出了一种新颖的多智能体人工智能辅导平台，该平台结合了自适应和个性化反馈、结构化课程生成以及教科书知识检索，以促进模块化和工具辅助的学习过程。该系统使学生能够学习新主题、识别和集中解决弱点、有效复习考试，并无限制地进行个性化练习。本文通过引入一种结合了教学代理和人工智能驱动组件的新平台，为数学教育领域带来了模块化和有效的教学系统，从而推动了教育人工智能领域的发展。 

---
# VideoITG: Multimodal Video Understanding with Instructed Temporal Grounding 

**Title (ZH)**: VideoITG: 带有指令时间定位的多模态视频理解 

**Authors**: Shihao Wang, Guo Chen, De-an Huang, Zhiqi Li, Minghan Li, Guilin Li, Jose M. Alvarez, Lei Zhang, Zhiding Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13353)  

**Abstract**: Recent studies have revealed that selecting informative and relevant video frames can significantly improve the performance of Video Large Language Models (Video-LLMs). Current methods, such as reducing inter-frame redundancy, employing separate models for image-text relevance assessment, or utilizing temporal video grounding for event localization, substantially adopt unsupervised learning paradigms, whereas they struggle to address the complex scenarios in long video understanding. We propose Instructed Temporal Grounding for Videos (VideoITG), featuring customized frame sampling aligned with user instructions. The core of VideoITG is the VidThinker pipeline, an automated annotation framework that explicitly mimics the human annotation process. First, it generates detailed clip-level captions conditioned on the instruction; then, it retrieves relevant video segments through instruction-guided reasoning; finally, it performs fine-grained frame selection to pinpoint the most informative visual evidence. Leveraging VidThinker, we construct the VideoITG-40K dataset, containing 40K videos and 500K instructed temporal grounding annotations. We then design a plug-and-play VideoITG model, which takes advantage of visual language alignment and reasoning capabilities of Video-LLMs, for effective frame selection in a discriminative manner. Coupled with Video-LLMs, VideoITG achieves consistent performance improvements across multiple multimodal video understanding benchmarks, showing its superiority and great potentials for video understanding. 

**Abstract (ZH)**: 最近的研究表明，选择信息丰富且相关的视频帧可以显著提高视频大型语言模型（Video-LLMs）的性能。当前的方法，如减少帧间冗余、采用独立模型进行图像-文本相关性评估或利用时序视频接地进行事件定位，主要依赖无监督学习范式，但在长视频理解的复杂场景中难以应对。我们提出了Instructed Temporal Grounding for Videos（VideoITG），这是一个根据用户指令定制帧采样的框架。VideoITG的核心是VidThinker流水线，这是一个自动注释框架，显式模拟了人类注释过程。首先，它根据指令生成详细的片段级字幕；然后，通过指令引导的推理检索相关视频片段；最后，进行精细的帧选择，以确定最有信息性的视觉证据。利用VidThinker，我们构建了包含40K视频和500K指令时序接地注释的VideoITG-40K数据集。我们随后设计了一个即插即用的VideoITG模型，该模型利用了Video-LLMs的视觉语言对齐和推理能力，以区分的方式有效地进行帧选择。结合Video-LLMs，VideoITG在多个多模态视频理解基准测试中实现了一致的性能提升，展示了其在视频理解方面的优势和巨大潜力。 

---
# VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning 

**Title (ZH)**: VisionThink：通过强化学习实现的智能高效视觉语言模型 

**Authors**: Senqiao Yang, Junyi Li, Xin Lai, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2507.13348)  

**Abstract**: Recent advancements in vision-language models (VLMs) have improved performance by increasing the number of visual tokens, which are often significantly longer than text tokens. However, we observe that most real-world scenarios do not require such an extensive number of visual tokens. While the performance drops significantly in a small subset of OCR-related tasks, models still perform accurately in most other general VQA tasks with only 1/4 resolution. Therefore, we propose to dynamically process distinct samples with different resolutions, and present a new paradigm for visual token compression, namely, VisionThink. It starts with a downsampled image and smartly decides whether it is sufficient for problem solving. Otherwise, the model could output a special token to request the higher-resolution image. Compared to existing Efficient VLM methods that compress tokens using fixed pruning ratios or thresholds, VisionThink autonomously decides whether to compress tokens case by case. As a result, it demonstrates strong fine-grained visual understanding capability on OCR-related tasks, and meanwhile saves substantial visual tokens on simpler tasks. We adopt reinforcement learning and propose the LLM-as-Judge strategy to successfully apply RL to general VQA tasks. Moreover, we carefully design a reward function and penalty mechanism to achieve a stable and reasonable image resize call ratio. Extensive experiments demonstrate the superiority, efficiency, and effectiveness of our method. Our code is available at this https URL. 

**Abstract (ZH)**: Recent advancements in 视觉-语言模型的最近进展通过增加视觉标记的数量提升了性能，这些视觉标记往往比文本标记长得多。然而，我们观察到大多数实际应用场景不需要如此大量的视觉标记。尽管在一小部分OCR相关的任务中性能显著下降，但在大多数其他一般视觉问答任务中，仅使用1/4分辨率就能准确完成任务。因此，我们提出了一种不同样本动态处理不同分辨率的新方法，并提出了一种视觉标记压缩的新范式，即VisionThink。它从下采样的图像开始，智能地决定是否足够解决问题，否则模型可以输出一个特殊标记来请求更高分辨率的图像。相比现有的使用固定剪枝比或阈值的高效视觉模型方法，VisionThink能够逐案自主决定是否压缩标记，从而在OCR相关的任务中展示了强大的细粒度视觉理解能力，并且在较简单的任务中节省了大量视觉标记。我们采用强化学习并提出LLM-as-Judge策略成功将RL应用于一般视觉问答任务。此外，我们精心设计了奖励函数和惩罚机制，以实现稳定的合理的图像缩放调用比例。广泛的实验证明了我们方法的优越性、高效性和有效性。我们的代码可在以下链接获取。 

---
# Imbalance in Balance: Online Concept Balancing in Generation Models 

**Title (ZH)**: 平衡中的不平衡：生成模型中的在线概念平衡 

**Authors**: Yukai Shi, Jiarong Ou, Rui Chen, Haotian Yang, Jiahao Wang, Xin Tao, Pengfei Wan, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2507.13345)  

**Abstract**: In visual generation tasks, the responses and combinations of complex concepts often lack stability and are error-prone, which remains an under-explored area. In this paper, we attempt to explore the causal factors for poor concept responses through elaborately designed experiments. We also design a concept-wise equalization loss function (IMBA loss) to address this issue. Our proposed method is online, eliminating the need for offline dataset processing, and requires minimal code changes. In our newly proposed complex concept benchmark Inert-CompBench and two other public test sets, our method significantly enhances the concept response capability of baseline models and yields highly competitive results with only a few codes. 

**Abstract (ZH)**: 在视觉生成任务中，复杂概念的响应和组合往往缺乏稳定性和可靠性，这是一个未充分探索的领域。本文通过精心设计的实验尝试探究较差概念响应的原因，并设计了一种概念层面的均衡损失函数（IMBA损失）来解决这一问题。我们提出的方法是在线的，无需离线处理数据集，并且只需要少量的代码修改。在我们 newly 提出的复杂概念基准 Inert-CompBench 和两个其他公开测试集上，我们的方法显著提升了基线模型的概念响应能力，并仅用少量代码获得了极具竞争力的结果。 

---
# Latent Policy Steering with Embodiment-Agnostic Pretrained World Models 

**Title (ZH)**: 基于体感无关先验世界模型的潜在策略引导 

**Authors**: Yiqi Wang, Mrinal Verghese, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2507.13340)  

**Abstract**: Learning visuomotor policies via imitation has proven effective across a wide range of robotic domains. However, the performance of these policies is heavily dependent on the number of training demonstrations, which requires expensive data collection in the real world. In this work, we aim to reduce data collection efforts when learning visuomotor robot policies by leveraging existing or cost-effective data from a wide range of embodiments, such as public robot datasets and the datasets of humans playing with objects (human data from play). Our approach leverages two key insights. First, we use optic flow as an embodiment-agnostic action representation to train a World Model (WM) across multi-embodiment datasets, and finetune it on a small amount of robot data from the target embodiment. Second, we develop a method, Latent Policy Steering (LPS), to improve the output of a behavior-cloned policy by searching in the latent space of the WM for better action sequences. In real world experiments, we observe significant improvements in the performance of policies trained with a small amount of data (over 50% relative improvement with 30 demonstrations and over 20% relative improvement with 50 demonstrations) by combining the policy with a WM pretrained on two thousand episodes sampled from the existing Open X-embodiment dataset across different robots or a cost-effective human dataset from play. 

**Abstract (ZH)**: 通过利用现有或低成本的多体模数据进行类目仿真的视觉运动机器人策略学习：减少数据收集努力的方法 

---
# Vision-and-Language Training Helps Deploy Taxonomic Knowledge but Does Not Fundamentally Alter It 

**Title (ZH)**: 视觉-语言训练有助于部署分类知识但不会从根本上改变它 

**Authors**: Yulu Qin, Dheeraj Varghese, Adam Dahlgren Lindström, Lucia Donatelli, Kanishka Misra, Najoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.13328)  

**Abstract**: Does vision-and-language (VL) training change the linguistic representations of language models in meaningful ways? Most results in the literature have shown inconsistent or marginal differences, both behaviorally and representationally. In this work, we start from the hypothesis that the domain in which VL training could have a significant effect is lexical-conceptual knowledge, in particular its taxonomic organization. Through comparing minimal pairs of text-only LMs and their VL-trained counterparts, we first show that the VL models often outperform their text-only counterparts on a text-only question-answering task that requires taxonomic understanding of concepts mentioned in the questions. Using an array of targeted behavioral and representational analyses, we show that the LMs and VLMs do not differ significantly in terms of their taxonomic knowledge itself, but they differ in how they represent questions that contain concepts in a taxonomic relation vs. a non-taxonomic relation. This implies that the taxonomic knowledge itself does not change substantially through additional VL training, but VL training does improve the deployment of this knowledge in the context of a specific task, even when the presentation of the task is purely linguistic. 

**Abstract (ZH)**: 视觉-语言（VL）训练是否以有意义的方式改变了语言模型的语义表示？现有文献中的大多数结果在行为和表示上显示出了不一致或边际差异。在这个工作中，我们假设VL训练可能在词汇-概念知识，尤其是其分类组织方面产生显著影响。通过比较仅文本模型和其VL训练版本的最小对，我们首先展示了在需要理解问题中提到的概念的分类组织的文本仅问答任务上，VL模型往往超越其仅文本对应模型。利用一系列有针对性的行为和表示分析，我们证明了语言模型和视觉语言模型在分类知识本身上并没有显著差异，但在表示包含分类关系的概念的问题 vs. 非分类关系的概念的问题上存在差异。这表明通过额外的VL训练，分类知识本身并未发生显著变化，但VL训练确实改进了在这种特定任务中的知识应用，即使任务的呈现完全是语言性的。 

---
# Revisiting Reliability in the Reasoning-based Pose Estimation Benchmark 

**Title (ZH)**: 基于推理的 pose 估计基准中的可靠性重探 

**Authors**: Junsu Kim, Naeun Kim, Jaeho Lee, Incheol Park, Dongyoon Han, Seungryul Baek  

**Link**: [PDF](https://arxiv.org/pdf/2507.13314)  

**Abstract**: The reasoning-based pose estimation (RPE) benchmark has emerged as a widely adopted evaluation standard for pose-aware multimodal large language models (MLLMs). Despite its significance, we identified critical reproducibility and benchmark-quality issues that hinder fair and consistent quantitative evaluations. Most notably, the benchmark utilizes different image indices from those of the original 3DPW dataset, forcing researchers into tedious and error-prone manual matching processes to obtain accurate ground-truth (GT) annotations for quantitative metrics (\eg, MPJPE, PA-MPJPE). Furthermore, our analysis reveals several inherent benchmark-quality limitations, including significant image redundancy, scenario imbalance, overly simplistic poses, and ambiguous textual descriptions, collectively undermining reliable evaluations across diverse scenarios. To alleviate manual effort and enhance reproducibility, we carefully refined the GT annotations through meticulous visual matching and publicly release these refined annotations as an open-source resource, thereby promoting consistent quantitative evaluations and facilitating future advancements in human pose-aware multimodal reasoning. 

**Abstract (ZH)**: 基于推理的姿态估计基准（RPE）已成为评估姿态感知多模态大型语言模型（MLLMs）的广泛采用评估标准。尽管具有重要意义，但我们发现其实现可再现性及基准质量存在关键问题，阻碍了公平和一致的定量评估。最显著的是，该基准使用了与原始3DPW数据集不同的图像索引，迫使研究人员进行繁琐且易出错的手动匹配过程，以获取准确的地面真相（GT）注释以用于定量指标（例如，MPJPE，PA-MPJPE）。此外，我们的分析揭示了基准固有的几个质量限制，包括显著的图像冗余、场景失衡、过于简化的姿态以及模棱两可的文本描述，这些共同削弱了在不同场景下可靠评估的能力。为了减轻人工努力并增强可再现性，我们通过仔细的视觉匹配仔细精炼了GT注释，并公开发布了这些精炼注释作为开源资源，从而促进一致的定量评估并促进未来在姿态感知多模态推理方面的进步。 

---
# AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research 

**Title (ZH)**: AbGen: 在消融研究设计与评估中的大型语言模型评价方法 

**Authors**: Yilun Zhao, Weiyuan Chen, Zhijian Xu, Manasi Patwardhan, Yixin Liu, Chengye Wang, Lovekesh Vig, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.13300)  

**Abstract**: We introduce AbGen, the first benchmark designed to evaluate the capabilities of LLMs in designing ablation studies for scientific research. AbGen consists of 1,500 expert-annotated examples derived from 807 NLP papers. In this benchmark, LLMs are tasked with generating detailed ablation study designs for a specified module or process based on the given research context. Our evaluation of leading LLMs, such as DeepSeek-R1-0528 and o4-mini, highlights a significant performance gap between these models and human experts in terms of the importance, faithfulness, and soundness of the ablation study designs. Moreover, we demonstrate that current automated evaluation methods are not reliable for our task, as they show a significant discrepancy when compared to human assessment. To better investigate this, we develop AbGen-Eval, a meta-evaluation benchmark designed to assess the reliability of commonly used automated evaluation systems in measuring LLM performance on our task. We investigate various LLM-as-Judge systems on AbGen-Eval, providing insights for future research on developing more effective and reliable LLM-based evaluation systems for complex scientific tasks. 

**Abstract (ZH)**: 我们介绍了AbGen，这是首个用于评估LLM在设计科学研究所需消融研究能力的标准基准。AbGen包含1,500个由807篇NLP论文衍生出的专家注释示例。在这个基准中，LLM需要根据给定的研究背景生成指定模块或过程的详细消融研究设计方案。我们对DeepSeek-R1-0528和o4-mini等领先LLM的评估显示，这些模型在重要性、忠实度和合理性方面与人类专家之间的性能差距显著。此外，我们证明当前的自动化评估方法对于我们的任务而言不够可靠，因为它们与人类评估相比表现出显著差异。为了更好地研究这一问题，我们开发了AbGen-Eval，这是一个用于评估常用自动化评估系统可靠性的元评估基准，旨在测量LLM在完成我们任务时的表现。我们对AbGen-Eval上的各种LLM-as-Judge系统进行了研究，为开发更有效和可靠的基于LLM的评估系统提供了对未来复杂科学任务研究的见解。 

---
# Towards Formal Verification of LLM-Generated Code from Natural Language Prompts 

**Title (ZH)**: 面向自然语言提示生成的LLM代码形式化验证 

**Authors**: Aaron Councilman, David Fu, Aryan Gupta, Chengxiao Wang, David Grove, Yu-Xiong Wang, Vikram Adve  

**Link**: [PDF](https://arxiv.org/pdf/2507.13290)  

**Abstract**: In the past few years LLMs have emerged as a tool that can aid programmers by taking natural language descriptions and generating code based on it. However, LLMs often generate incorrect code that users need to fix and the literature suggests users often struggle to detect these errors. In this work we seek to offer formal guarantees of correctness to LLM generated code; such guarantees could improve the experience of using AI Code Assistants and potentially enable natural language programming for users with little or no programming knowledge. To address this challenge we propose to incorporate a formal query language that can represent a user's intent in a formally defined but natural language-like manner that a user can confirm matches their intent. Then, using such a query we propose to verify LLM generated code to ensure it matches the user's intent. We implement these ideas in our system, Astrogator, for the Ansible programming language which includes such a formal query language, a calculus for representing the behavior of Ansible programs, and a symbolic interpreter which is used for the verification. On a benchmark suite of 21 code-generation tasks, our verifier is able to verify correct code in 83% of cases and identify incorrect code in 92%. 

**Abstract (ZH)**: 近年来，大规模语言模型(LLMs)已成为一种辅助程序员的工具，可以通过自然语言描述生成代码。然而，LLM们往往生成错误的代码，用户需要进行修正，文献显示用户往往难以检测这些错误。在本工作中，我们旨在为LLM生成的代码提供形式化的正确性保证；这样的保证将提升使用AI代码助手的体验，并有可能使不懂编程知识的用户实现基于自然语言的编程。为了应对这一挑战，我们提出将一种形式化的查询语言融入其中，该语言能够以一种用户可以确认与自身意图一致的、形式化但类似自然语言的方式表示用户意图。然后，我们使用这样的查询验证LLM生成的代码，以确保代码符合用户的意图。我们在包含这种形式化的查询语言、表示Ansible程序行为的计算规则以及用于验证的符号解释器的系统Astrogator中实现了这些想法。在包含21个代码生成任务的基准测试套件中，我们的验证器能够正确验证代码的83%的情况，并识别出错误代码的92%的情况。 

---
# Evaluating Reinforcement Learning Algorithms for Navigation in Simulated Robotic Quadrupeds: A Comparative Study Inspired by Guide Dog Behaviour 

**Title (ZH)**: 基于导盲犬行为的模拟四足机器人导航中强化学习算法的比较研究 

**Authors**: Emma M. A. Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2507.13277)  

**Abstract**: Robots are increasingly integrated across industries, particularly in healthcare. However, many valuable applications for quadrupedal robots remain overlooked. This research explores the effectiveness of three reinforcement learning algorithms in training a simulated quadruped robot for autonomous navigation and obstacle avoidance. The goal is to develop a robotic guide dog simulation capable of path following and obstacle avoidance, with long-term potential for real-world assistance to guide dogs and visually impaired individuals. It also seeks to expand research into medical 'pets', including robotic guide and alert dogs.
A comparative analysis of thirteen related research papers shaped key evaluation criteria, including collision detection, pathfinding algorithms, sensor usage, robot type, and simulation platforms. The study focuses on sensor inputs, collision frequency, reward signals, and learning progression to determine which algorithm best supports robotic navigation in complex environments.
Custom-made environments were used to ensure fair evaluation of all three algorithms under controlled conditions, allowing consistent data collection. Results show that Proximal Policy Optimization (PPO) outperformed Deep Q-Network (DQN) and Q-learning across all metrics, particularly in average and median steps to goal per episode.
By analysing these results, this study contributes to robotic navigation, AI and medical robotics, offering insights into the feasibility of AI-driven quadruped mobility and its role in assistive robotics. 

**Abstract (ZH)**: 四足机器人在医疗领域的潜在应用探索：强化学习算法在自主导航和障碍避让中的效果比较 

---
# Overview of the TalentCLEF 2025: Skill and Job Title Intelligence for Human Capital Management 

**Title (ZH)**: TalentCLEF 2025：人才技能与职位标题智能化在人力资源资本管理中的概述 

**Authors**: Luis Gasco, Hermenegildo Fabregat, Laura García-Sardiña, Paula Estrella, Daniel Deniz, Alvaro Rodrigo, Rabih Zbib  

**Link**: [PDF](https://arxiv.org/pdf/2507.13275)  

**Abstract**: Advances in natural language processing and large language models are driving a major transformation in Human Capital Management, with a growing interest in building smart systems based on language technologies for talent acquisition, upskilling strategies, and workforce planning. However, the adoption and progress of these technologies critically depend on the development of reliable and fair models, properly evaluated on public data and open benchmarks, which have so far been unavailable in this domain.
To address this gap, we present TalentCLEF 2025, the first evaluation campaign focused on skill and job title intelligence. The lab consists of two tasks: Task A - Multilingual Job Title Matching, covering English, Spanish, German, and Chinese; and Task B - Job Title-Based Skill Prediction, in English. Both corpora were built from real job applications, carefully anonymized, and manually annotated to reflect the complexity and diversity of real-world labor market data, including linguistic variability and gender-marked expressions.
The evaluations included monolingual and cross-lingual scenarios and covered the evaluation of gender bias.
TalentCLEF attracted 76 registered teams with more than 280 submissions. Most systems relied on information retrieval techniques built with multilingual encoder-based models fine-tuned with contrastive learning, and several of them incorporated large language models for data augmentation or re-ranking. The results show that the training strategies have a larger effect than the size of the model alone. TalentCLEF provides the first public benchmark in this field and encourages the development of robust, fair, and transferable language technologies for the labor market. 

**Abstract (ZH)**: 自然语言处理和大型语言模型的发展正在推动人力资源管理的重大变革，越来越多的研究关注基于语言技术的智能系统在 talent acquisition、技能提升策略和劳动力规划方面的作用。然而，这些技术的采用和进步关键取决于可靠和公平模型的发展，这些模型应在公开数据和开放基准上进行适当评估，而目前这些资源在该领域还不可用。
为填补这一空白，我们提出了 TalentCLEF 2025，这是首个专注于技能和职位标题智能的评估活动。实验室包括两个任务：任务A - 多语言职位标题匹配，涵盖英语、西班牙语、德语和中文；任务B - 基于职位标题的技能预测，使用英语。两个语料库均基于真实的职业申请，经过仔细脱敏并手工标注，以反映现实世界劳动力市场的复杂性和多样性，包括语言变异性及性别标记的表达。
评估包括单语和跨语言场景，并涵盖了性别偏见的评估。
TalentCLEF 获得了 76 支注册队伍，提交了超过 280 份提交作品。大多数系统依赖于使用多语言编码器模型并结合对比学习进行微调的信息检索技术，其中一些系统还结合了大型语言模型进行数据增强或重排。结果显示，训练策略的效果比模型大小本身更大。TalentCLEF 提供了该领域的首个公开基准，并促进了适用于劳动力市场的稳健、公平和可转移的语言技术的发展。 

---
# QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation 

**Title (ZH)**: QuestA: 通过问题扩充增强LLM的推理能力 

**Authors**: Jiazheng Li, Hong Lu, Kaiyue Wen, Zaiwen Yang, Jiaxuan Gao, Hongzhou Lin, Yi Wu, Jingzhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13266)  

**Abstract**: Reinforcement learning (RL) has become a key component in training large language reasoning models (LLMs). However, recent studies questions its effectiveness in improving multi-step reasoning-particularly on hard problems. To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 67.1% (+5.3%) on AIME24, 59.5% (+10.0%) on AIME25, and 35.5% (+4.0%) on HMMT25. Further, we provide theoretical explanations that QuestA improves sample efficiency, offering a practical and generalizable pathway for expanding reasoning capability through RL. 

**Abstract (ZH)**: 强化学习（RL）已成为训练大规模语言推理模型（LLMs）的关键组成部分。然而，近期的研究对其在提高多步推理能力方面的有效性提出了质疑，特别是在解决难题时。为应对这一挑战，我们提出了一种简单而有效的方法——问题扩增：在训练过程中引入部分解决方案，以降低问题难度并提供更加信息丰富的学习信号。我们的方法QuestA，在应用到数学推理任务的RL训练中，不仅能改善pass@1，还能在标准RL难以取得进展的问题上显著提高pass@k。这使得我们的模型能够持续超越如DeepScaleR和OpenMath Nemotron等强大的开源模型，进一步增强其推理能力。我们使用拥有1.5B参数的模型在数学基准测试中取得了新的最先进成果：AIME24上67.1% (+5.3%)，AIME25上59.5% (+10.0%)，HMMT25上35.5% (+4.0%)。此外，我们提供了理论解释，证明QuestA提高了样本效率，为通过RL扩展推理能力提供了一条实用且可推广的途径。 

---
# Voxtral 

**Title (ZH)**: VoiceTral 

**Authors**: Alexander H. Liu, Andy Ehrenberg, Andy Lo, Clément Denoix, Corentin Barreau, Guillaume Lample, Jean-Malo Delignon, Khyathi Raghavi Chandu, Patrick von Platen, Pavankumar Reddy Muddireddy, Sanchit Gandhi, Soham Ghosh, Srijan Mishra, Thomas Foubert, Abhinav Rastogi, Adam Yang, Albert Q. Jiang, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius Dabert, Devendra Singh Chaplot, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gabrielle Berrada, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jason Rute, Jean-Hadrien Chabran, Jessica Chudnovsky, Joachim Studnia, Joep Barmentlo, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Karmesh Yadav, Kartik Khandelwal, Kush Jain, Lélio Renard Lavaud, Léonard Blier, Lingxiao Zhao, Louis Martin, Lucile Saulnier, Luyu Gao, Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Matthieu Dinot, Maxime Darrin, Maximilian Augustin, Mickaël Seznec, Neha Gupta, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Rémi Delacourt, Romain Sauvestre, Roman Soletskyi, Sagar Vaze, Sandeep Subramanian, Saurabh Garg, Shashwat Dalal, Siddharth Gandhi, Sumukh Aithal, Szymon Antoniak, Teven Le Scao, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Tom Bewley, Valeriia Nemychnikova, Victor Paltz  

**Link**: [PDF](https://arxiv.org/pdf/2507.13264)  

**Abstract**: We present Voxtral Mini and Voxtral Small, two multimodal audio chat models. Voxtral is trained to comprehend both spoken audio and text documents, achieving state-of-the-art performance across a diverse range of audio benchmarks, while preserving strong text capabilities. Voxtral Small outperforms a number of closed-source models, while being small enough to run locally. A 32K context window enables the model to handle audio files up to 40 minutes in duration and long multi-turn conversations. We also contribute three benchmarks for evaluating speech understanding models on knowledge and trivia. Both Voxtral models are released under Apache 2.0 license. 

**Abstract (ZH)**: 我们呈现了Voxtral Mini和Voxtral Small两种多模态音频聊天模型。Voxtral模型经过训练，能够理解语音音频和文本文档，取得了多种音频基准上的最佳性能，同时保留了强大的文本能力。Voxtral Small在性能上优于多种封闭源模型，且体积足够小以支持本地运行。32K上下文窗口使模型能够处理长达40分钟的音频文件和长多轮对话。我们还贡献了三个benchmark，用于评估语音理解模型在知识和 trivia 方面的表现。两种Voxtral模型均采用Apache 2.0许可证发布。 

---
# Merge Kernel for Bayesian Optimization on Permutation Space 

**Title (ZH)**: 排列空间上贝叶斯优化的合并内核 

**Authors**: Zikai Xie, Linjiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13263)  

**Abstract**: Bayesian Optimization (BO) algorithm is a standard tool for black-box optimization problems. The current state-of-the-art BO approach for permutation spaces relies on the Mallows kernel-an $\Omega(n^2)$ representation that explicitly enumerates every pairwise comparison. Inspired by the close relationship between the Mallows kernel and pairwise comparison, we propose a novel framework for generating kernel functions on permutation space based on sorting algorithms. Within this framework, the Mallows kernel can be viewed as a special instance derived from bubble sort. Further, we introduce the \textbf{Merge Kernel} constructed from merge sort, which replaces the quadratic complexity with $\Theta(n\log n)$ to achieve the lowest possible complexity. The resulting feature vector is significantly shorter, can be computed in linearithmic time, yet still efficiently captures meaningful permutation distances. To boost robustness and right-invariance without sacrificing compactness, we further incorporate three lightweight, task-agnostic descriptors: (1) a shift histogram, which aggregates absolute element displacements and supplies a global misplacement signal; (2) a split-pair line, which encodes selected long-range comparisons by aligning elements across the two halves of the whole permutation; and (3) sliding-window motifs, which summarize local order patterns that influence near-neighbor objectives. Our empirical evaluation demonstrates that the proposed kernel consistently outperforms the state-of-the-art Mallows kernel across various permutation optimization benchmarks. Results confirm that the Merge Kernel provides a more compact yet more effective solution for Bayesian optimization in permutation space. 

**Abstract (ZH)**: 基于排序算法的排列空间核函数生成框架：Merge Kernel及其实现 

---
# Efficient Adaptation of Pre-trained Vision Transformer underpinned by Approximately Orthogonal Fine-Tuning Strategy 

**Title (ZH)**: 基于近似正交微调策略的预训练视觉变换器高效适配 

**Authors**: Yiting Yang, Hao Luo, Yuan Sun, Qingsen Yan, Haokui Zhang, Wei Dong, Guoqing Wang, Peng Wang, Yang Yang, Hengtao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.13260)  

**Abstract**: A prevalent approach in Parameter-Efficient Fine-Tuning (PEFT) of pre-trained Vision Transformers (ViT) involves freezing the majority of the backbone parameters and solely learning low-rank adaptation weight matrices to accommodate downstream tasks. These low-rank matrices are commonly derived through the multiplication structure of down-projection and up-projection matrices, exemplified by methods such as LoRA and Adapter. In this work, we observe an approximate orthogonality among any two row or column vectors within any weight matrix of the backbone parameters; however, this property is absent in the vectors of the down/up-projection matrices. Approximate orthogonality implies a reduction in the upper bound of the model's generalization error, signifying that the model possesses enhanced generalization capability. If the fine-tuned down/up-projection matrices were to exhibit this same property as the pre-trained backbone matrices, could the generalization capability of fine-tuned ViTs be further augmented? To address this question, we propose an Approximately Orthogonal Fine-Tuning (AOFT) strategy for representing the low-rank weight matrices. This strategy employs a single learnable vector to generate a set of approximately orthogonal vectors, which form the down/up-projection matrices, thereby aligning the properties of these matrices with those of the backbone. Extensive experimental results demonstrate that our method achieves competitive performance across a range of downstream image classification tasks, confirming the efficacy of the enhanced generalization capability embedded in the down/up-projection matrices. 

**Abstract (ZH)**: 参数高效微调（PEFT）中流行的预训练视觉变换器（ViT）方法包括冻结大部分骨干参数，并仅学习低秩适应权重矩阵以适应下游任务。这些低秩矩阵通常通过下投影和上投影矩阵的乘积结构导出，如LoRA和Adapter方法所体现。在本文中，我们观察到任何权重矩阵中的任意两行或列向量之间存在近似正交性；然而，这种性质在下/上投影矩阵的向量中不存在。近似正交性意味着模型泛化误差的上界减小，表明模型具有增强的泛化能力。如果微调后的下/上投影矩阵能够表现出与预训练骨干矩阵相同的优势，则微调的ViT的泛化能力是否可以进一步增强？为回答这一问题，我们提出了一个近似正交微调（AOFT）策略来表示低秩权重矩阵。该策略使用一个可学习的向量生成一组近似正交的向量，从而构成下/上投影矩阵，使这些矩阵的性质与骨干矩阵的性质对齐。广泛的实验结果证明，我们的方法在一系列下游图像分类任务中实现了竞争力的表现，证实了嵌入在下/上投影矩阵中的增强泛化能力的有效性。 

---
# Automating Steering for Safe Multimodal Large Language Models 

**Title (ZH)**: 自动调控以确保安全的多模态大型语言模型 

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.13255)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems. 

**Abstract (ZH)**: 最近在多模态大型语言模型（MLLMs）方面取得的进展解锁了强大的跨模态推理能力，但也引发了新的安全关切，特别是在面对 adversarial 多模态输入时。为了在推理过程中提高 MLLMs 的安全性，我们介绍了一种无需对底层模型进行微调的模块化和自适应推理时干预技术 AutoSteer。AutoSteer 包含三个核心组件：（1）一种新颖的安全意识评分（SAS），能够自动识别模型内部层级中最具安全相关性的区别；（2）一种自适应安全探针，训练用于估算中间表示生成有毒输出的可能性；（3）一种轻量级拒绝端，能够在检测到安全风险时有选择地干预以调节生成。在 LLaVA-OV 和 Chameleon 上跨多种安全关键基准的实验表明，AutoSteer 显著降低了对文本、视觉和跨模态威胁的攻击成功率（ASR），同时保持了一般能力。这些发现将 AutoSteer 定位为一种实用、可解释且有效的框架，用于更安全部署多模态 AI 系统。 

---
# HATS: Hindi Analogy Test Set for Evaluating Reasoning in Large Language Models 

**Title (ZH)**: HATS：用于评估大规模语言模型推理能力的印地语类比测试集 

**Authors**: Ashray Gupta, Rohan Joseph, Sunny Rai  

**Link**: [PDF](https://arxiv.org/pdf/2507.13238)  

**Abstract**: Analogies test a model's ability to infer implicit relationships between concepts, making them a key benchmark for evaluating reasoning capabilities. While large language models (LLMs) are widely evaluated for reasoning in English, their abilities in Indic languages remain understudied, limiting our understanding of whether these models generalize across languages. To address this gap, we introduce a new Hindi Analogy Test Set (HATS), comprising 405 multiple-choice questions sourced from Indian government exams. We benchmark state-of-the-art multilingual LLMs using various prompting strategies and introduce a grounded Chain of Thought approach that leverages cognitive theories of analogical reasoning. This approach improves model performance on Hindi analogy questions. Our experiments show that models perform best with English prompts, irrespective of the prompting strategy. Our test set addresses the lack of a critical resource to evaluate LLM reasoning capabilities in Hindi. 

**Abstract (ZH)**: Hindi Analogies Test Set (HATS): Evaluating Reasoning Capabilities of Large Language Models in Indian Languages 

---
# VITA: Vision-to-Action Flow Matching Policy 

**Title (ZH)**: VITA：视觉到动作流动策略匹配 

**Authors**: Dechen Gao, Boqi Zhao, Andrew Lee, Ian Chuang, Hanchu Zhou, Hang Wang, Zhe Zhao, Junshan Zhang, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2507.13231)  

**Abstract**: We present VITA, a Vision-To-Action flow matching policy that evolves latent visual representations into latent actions for visuomotor control. Traditional flow matching and diffusion policies sample from standard source distributions (e.g., Gaussian noise) and require additional conditioning mechanisms like cross-attention to condition action generation on visual information, creating time and space overheads. VITA proposes a novel paradigm that treats latent images as the flow source, learning an inherent mapping from vision to action while eliminating separate conditioning modules and preserving generative modeling capabilities. Learning flows between fundamentally different modalities like vision and action is challenging due to sparse action data lacking semantic structures and dimensional mismatches between high-dimensional visual representations and raw actions. We address this by creating a structured action latent space via an autoencoder as the flow matching target, up-sampling raw actions to match visual representation shapes. Crucially, we supervise flow matching with both encoder targets and final action outputs through flow latent decoding, which backpropagates action reconstruction loss through sequential flow matching ODE solving steps for effective end-to-end learning. Implemented as simple MLP layers, VITA is evaluated on challenging bi-manual manipulation tasks on the ALOHA platform, including 5 simulation and 2 real-world tasks. Despite its simplicity, MLP-only VITA outperforms or matches state-of-the-art generative policies while reducing inference latency by 50-130% compared to conventional flow matching policies requiring different conditioning mechanisms or complex architectures. To our knowledge, VITA is the first MLP-only flow matching policy capable of solving complex bi-manual manipulation tasks like those in ALOHA benchmarks. 

**Abstract (ZH)**: VITA：一种将视觉表示演化为潜在动作的视觉到行动流匹配策略 

---
# $S^2M^2$: Scalable Stereo Matching Model for Reliable Depth Estimation 

**Title (ZH)**: $S^2M^2$: 可扩展的立体匹配模型以实现可靠的深度估计 

**Authors**: Junhong Min, Youngpil Jeon, Jimin Kim, Minyong Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.13229)  

**Abstract**: The pursuit of a generalizable stereo matching model, capable of performing across varying resolutions and disparity ranges without dataset-specific fine-tuning, has revealed a fundamental trade-off. Iterative local search methods achieve high scores on constrained benchmarks, but their core mechanism inherently limits the global consistency required for true generalization. On the other hand, global matching architectures, while theoretically more robust, have been historically rendered infeasible by prohibitive computational and memory costs. We resolve this dilemma with $S^2M^2$: a global matching architecture that achieves both state-of-the-art accuracy and high efficiency without relying on cost volume filtering or deep refinement stacks. Our design integrates a multi-resolution transformer for robust long-range correspondence, trained with a novel loss function that concentrates probability on feasible matches. This approach enables a more robust joint estimation of disparity, occlusion, and confidence. $S^2M^2$ establishes a new state of the art on the Middlebury v3 and ETH3D benchmarks, significantly outperforming prior methods across most metrics while reconstructing high-quality details with competitive efficiency. 

**Abstract (ZH)**: 一种无需数据集特定微调即可跨不同分辨率和视差范围通用的立体匹配模型的追求揭示了基本权衡。S²M²：一种无需依赖成本体过滤或深度精细堆栈即可同时实现最先进的精度和高效性的全局匹配架构。 

---
# Synthesizing Reality: Leveraging the Generative AI-Powered Platform Midjourney for Construction Worker Detection 

**Title (ZH)**: 合成现实：利用Midjourney生成AI驱动平台进行建筑工人检测 

**Authors**: Hongyang Zhao, Tianyu Liang, Sina Davari, Daeho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.13221)  

**Abstract**: While recent advancements in deep neural networks (DNNs) have substantially enhanced visual AI's capabilities, the challenge of inadequate data diversity and volume remains, particularly in construction domain. This study presents a novel image synthesis methodology tailored for construction worker detection, leveraging the generative-AI platform Midjourney. The approach entails generating a collection of 12,000 synthetic images by formulating 3000 different prompts, with an emphasis on image realism and diversity. These images, after manual labeling, serve as a dataset for DNN training. Evaluation on a real construction image dataset yielded promising results, with the model attaining average precisions (APs) of 0.937 and 0.642 at intersection-over-union (IoU) thresholds of 0.5 and 0.5 to 0.95, respectively. Notably, the model demonstrated near-perfect performance on the synthetic dataset, achieving APs of 0.994 and 0.919 at the two mentioned thresholds. These findings reveal both the potential and weakness of generative AI in addressing DNN training data scarcity. 

**Abstract (ZH)**: 基于Midjourney生成式AI平台的建筑工人检测合成图像生成方法研究 

---
# Aligning Humans and Robots via Reinforcement Learning from Implicit Human Feedback 

**Title (ZH)**: 通过隐式人类反馈强化学习实现人类与机器人对齐 

**Authors**: Suzie Kim, Hye-Bin Shin, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.13171)  

**Abstract**: Conventional reinforcement learning (RL) ap proaches often struggle to learn effective policies under sparse reward conditions, necessitating the manual design of complex, task-specific reward functions. To address this limitation, rein forcement learning from human feedback (RLHF) has emerged as a promising strategy that complements hand-crafted rewards with human-derived evaluation signals. However, most existing RLHF methods depend on explicit feedback mechanisms such as button presses or preference labels, which disrupt the natural interaction process and impose a substantial cognitive load on the user. We propose a novel reinforcement learning from implicit human feedback (RLIHF) framework that utilizes non-invasive electroencephalography (EEG) signals, specifically error-related potentials (ErrPs), to provide continuous, implicit feedback without requiring explicit user intervention. The proposed method adopts a pre-trained decoder to transform raw EEG signals into probabilistic reward components, en abling effective policy learning even in the presence of sparse external rewards. We evaluate our approach in a simulation environment built on the MuJoCo physics engine, using a Kinova Gen2 robotic arm to perform a complex pick-and-place task that requires avoiding obstacles while manipulating target objects. The results show that agents trained with decoded EEG feedback achieve performance comparable to those trained with dense, manually designed rewards. These findings validate the potential of using implicit neural feedback for scalable and human-aligned reinforcement learning in interactive robotics. 

**Abstract (ZH)**: 基于隐式人类反馈的强化学习框架（RLIHF）：利用脑电误差相关潜在成分实现可扩展的人机-aligned强化学习 

---
# SHIELD: A Secure and Highly Enhanced Integrated Learning for Robust Deepfake Detection against Adversarial Attacks 

**Title (ZH)**: SHIELD：一种针对对抗攻击的稳健深度假信息检测的安全高度集成学习方法 

**Authors**: Kutub Uddin, Awais Khan, Muhammad Umar Farooq, Khalid Malik  

**Link**: [PDF](https://arxiv.org/pdf/2507.13170)  

**Abstract**: Audio plays a crucial role in applications like speaker verification, voice-enabled smart devices, and audio conferencing. However, audio manipulations, such as deepfakes, pose significant risks by enabling the spread of misinformation. Our empirical analysis reveals that existing methods for detecting deepfake audio are often vulnerable to anti-forensic (AF) attacks, particularly those attacked using generative adversarial networks. In this article, we propose a novel collaborative learning method called SHIELD to defend against generative AF attacks. To expose AF signatures, we integrate an auxiliary generative model, called the defense (DF) generative model, which facilitates collaborative learning by combining input and output. Furthermore, we design a triplet model to capture correlations for real and AF attacked audios with real-generated and attacked-generated audios using auxiliary generative models. The proposed SHIELD strengthens the defense against generative AF attacks and achieves robust performance across various generative models. The proposed AF significantly reduces the average detection accuracy from 95.49% to 59.77% for ASVspoof2019, from 99.44% to 38.45% for In-the-Wild, and from 98.41% to 51.18% for HalfTruth for three different generative models. The proposed SHIELD mechanism is robust against AF attacks and achieves an average accuracy of 98.13%, 98.58%, and 99.57% in match, and 98.78%, 98.62%, and 98.85% in mismatch settings for the ASVspoof2019, In-the-Wild, and HalfTruth datasets, respectively. 

**Abstract (ZH)**: 音频在speaker验证、语音-enable的智能设备和音频会议等应用中起着关键作用。然而，音频篡改，如深伪，通过促进虚假信息的传播带来了重大风险。我们的实证分析表明，现有的深伪音频检测方法往往对对抗取证（AF）攻击脆弱，尤其是那些使用生成对抗网络进行攻击的。在本文中，我们提出了一种名为SHIELD的新型协作学习方法，以抵御生成性AF攻击。为了揭示AF特征，我们整合了一个辅助生成模型，称为防御（DF）生成模型，它通过结合输入和输出促进了协作学习。此外，我们设计了一个三元组模型，使用辅助生成模型捕获真实和AF攻击音频与真实生成和攻击生成音频之间的相关性。所提出的SHIELD增强了对生成性AF攻击的防御并实现了各种生成模型下的稳健性能。所提出的AF显著降低了ASVspoof2019、In-the-Wild和HalfTruth三个不同生成模型下的平均检测精度，分别为95.49%降至59.77%、99.44%降至38.45%和98.41%降至51.18%。所提出的SHIELD机制对AF攻击具有鲁棒性，在ASVspoof2019、In-the-Wild和HalfTruth数据集中，匹配设置下的平均准确率分别为98.13%、98.58%和99.57%，不匹配设置下的平均准确率分别为98.78%、98.62%和98.85%。 

---
# Prompt Injection 2.0: Hybrid AI Threats 

**Title (ZH)**: Prompt Injection 2.0: 混合AI威胁 

**Authors**: Jeremy McHugh, Kristina Šekrst, Jon Cefalu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13169)  

**Abstract**: Prompt injection attacks, where malicious input is designed to manipulate AI systems into ignoring their original instructions and following unauthorized commands instead, were first discovered by Preamble, Inc. in May 2022 and responsibly disclosed to OpenAI. Over the last three years, these attacks have continued to pose a critical security threat to LLM-integrated systems. The emergence of agentic AI systems, where LLMs autonomously perform multistep tasks through tools and coordination with other agents, has fundamentally transformed the threat landscape. Modern prompt injection attacks can now combine with traditional cybersecurity exploits to create hybrid threats that systematically evade traditional security controls. This paper presents a comprehensive analysis of Prompt Injection 2.0, examining how prompt injections integrate with Cross-Site Scripting (XSS), Cross-Site Request Forgery (CSRF), and other web security vulnerabilities to bypass traditional security measures. We build upon Preamble's foundational research and mitigation technologies, evaluating them against contemporary threats, including AI worms, multi-agent infections, and hybrid cyber-AI attacks. Our analysis incorporates recent benchmarks that demonstrate how traditional web application firewalls, XSS filters, and CSRF tokens fail against AI-enhanced attacks. We also present architectural solutions that combine prompt isolation, runtime security, and privilege separation with novel threat detection capabilities. 

**Abstract (ZH)**: Prompt注入攻击2.0：与跨站脚本(XSS)、跨站请求伪造(CSRF)及其他Web安全漏洞的综合分析 

---
# Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models 

**Title (ZH)**: Orbis: 克服驾驶世界模型长期预测挑战 

**Authors**: Arian Mousakhan, Sudhanshu Mittal, Silvio Galesso, Karim Farid, Thomas Brox  

**Link**: [PDF](https://arxiv.org/pdf/2507.13162)  

**Abstract**: Existing world models for autonomous driving struggle with long-horizon generation and generalization to challenging scenarios. In this work, we develop a model using simple design choices, and without additional supervision or sensors, such as maps, depth, or multiple cameras. We show that our model yields state-of-the-art performance, despite having only 469M parameters and being trained on 280h of video data. It particularly stands out in difficult scenarios like turning maneuvers and urban traffic. We test whether discrete token models possibly have advantages over continuous models based on flow matching. To this end, we set up a hybrid tokenizer that is compatible with both approaches and allows for a side-by-side comparison. Our study concludes in favor of the continuous autoregressive model, which is less brittle on individual design choices and more powerful than the model built on discrete tokens. Code, models and qualitative results are publicly available at this https URL. 

**Abstract (ZH)**: 现有的自动驾驶世界模型在长时_horizon生成和应对挑战性场景时表现不佳。在本工作中，我们采用简单的设计选择，无需额外的监督或传感器，如地图、深度信息或多摄像头。实验表明，尽管参数量仅为469M且训练数据量为280小时的视频数据，我们的模型仍能达到最先进的性能，特别在转弯操作和城市交通等困难场景中表现出色。我们测试了离散词元模型与连续模型在流匹配基础上的优劣，通过设置兼容两种方法的混合词元器进行对比分析。研究结果倾向于支持连续自回归模型，这种模型在单一设计选择上较少表现出脆弱性，并且比基于离散词元的模型更为强大。相关代码、模型及定性结果已公开。 

---
# Inverse Reinforcement Learning Meets Large Language Model Post-Training: Basics, Advances, and Opportunities 

**Title (ZH)**: 逆强化学习与大型语言模型后训练：基础知识、进展与机遇 

**Authors**: Hao Sun, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2507.13158)  

**Abstract**: In the era of Large Language Models (LLMs), alignment has emerged as a fundamental yet challenging problem in the pursuit of more reliable, controllable, and capable machine intelligence. The recent success of reasoning models and conversational AI systems has underscored the critical role of reinforcement learning (RL) in enhancing these systems, driving increased research interest at the intersection of RL and LLM alignment. This paper provides a comprehensive review of recent advances in LLM alignment through the lens of inverse reinforcement learning (IRL), emphasizing the distinctions between RL techniques employed in LLM alignment and those in conventional RL tasks. In particular, we highlight the necessity of constructing neural reward models from human data and discuss the formal and practical implications of this paradigm shift. We begin by introducing fundamental concepts in RL to provide a foundation for readers unfamiliar with the field. We then examine recent advances in this research agenda, discussing key challenges and opportunities in conducting IRL for LLM alignment. Beyond methodological considerations, we explore practical aspects, including datasets, benchmarks, evaluation metrics, infrastructure, and computationally efficient training and inference techniques. Finally, we draw insights from the literature on sparse-reward RL to identify open questions and potential research directions. By synthesizing findings from diverse studies, we aim to provide a structured and critical overview of the field, highlight unresolved challenges, and outline promising future directions for improving LLM alignment through RL and IRL techniques. 

**Abstract (ZH)**: 在大型语言模型时代，对齐问题已成为追求更可靠、可控和强大的机器智能的根本而又富有挑战的问题。最近推理模型和对话AI系统的成功进一步凸显了强化学习（RL）在增强这些系统中的关键作用，推动了RL与大型语言模型对齐研究领域的广泛关注。本文通过逆强化学习（IRL）的视角对近期大型语言模型对齐领域的进展进行了全面回顾，强调了在大型语言模型对齐中使用的RL技术与传统RL任务中使用的RL技术之间的区别。特别地，我们强调了从人类数据构建神经奖励模型的必要性，并讨论了这一范式转变的正式和实践意义。我们从介绍 RL 的基本概念开始，为不熟悉该领域的读者提供基础。然后，我们探讨了这一研究议程的最新进展，讨论了进行大型语言模型对齐的IRL时面临的挑战和机遇。我们不仅考虑方法论方面的问题，还探讨了实际方面的内容，包括数据集、基准、评估指标、基础设施以及计算效率高的训练和推理技术。最后，我们借鉴稀疏奖励强化学习领域的文献，识别出尚未解决的问题和潜在的研究方向。通过综合来自不同研究的发现，我们旨在提供一个结构化且批判性的领域概览，突出未解决的挑战，并概述通过RL和IRL技术改进大型语言模型对齐的有前途的方向。 

---
# SE-VLN: A Self-Evolving Vision-Language Navigation Framework Based on Multimodal Large Language Models 

**Title (ZH)**: SE-VLN：基于多模态大型语言模型的自我进化视觉-语言导航框架 

**Authors**: Xiangyu Dong, Haoran Zhao, Jiang Gao, Haozhou Li, Xiaoguang Ma, Yaoming Zhou, Fuhai Chen, Juan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13152)  

**Abstract**: Recent advances in vision-language navigation (VLN) were mainly attributed to emerging large language models (LLMs). These methods exhibited excellent generalization capabilities in instruction understanding and task reasoning. However, they were constrained by the fixed knowledge bases and reasoning abilities of LLMs, preventing fully incorporating experiential knowledge and thus resulting in a lack of efficient evolutionary capacity. To address this, we drew inspiration from the evolution capabilities of natural agents, and proposed a self-evolving VLN framework (SE-VLN) to endow VLN agents with the ability to continuously evolve during testing. To the best of our knowledge, it was the first time that an multimodal LLM-powered self-evolving VLN framework was proposed. Specifically, SE-VLN comprised three core modules, i.e., a hierarchical memory module to transfer successful and failure cases into reusable knowledge, a retrieval-augmented thought-based reasoning module to retrieve experience and enable multi-step decision-making, and a reflection module to realize continual evolution. Comprehensive tests illustrated that the SE-VLN achieved navigation success rates of 57% and 35.2% in unseen environments, representing absolute performance improvements of 23.9% and 15.0% over current state-of-the-art methods on R2R and REVERSE datasets, respectively. Moreover, the SE-VLN showed performance improvement with increasing experience repository, elucidating its great potential as a self-evolving agent framework for VLN. 

**Abstract (ZH)**: 最近视知觉语言导航(VLN)领域的进展主要归因于新兴的大规模语言模型(LLMs)。这些方法在指令理解和任务推理方面表现出了出色的泛化能力。然而，它们受限于LLMs固定的知识基础和推理能力，无法充分整合经验知识，从而导致缺乏高效的进化能力。为了解决这一问题，我们从自然界智能体的进化能力中汲取灵感，提出了一种自进化的VLN框架(SE-VLN)，以赋予VLN代理在测试过程中持续进化的 ability。据我们所知，这是首次提出一种由多模态LLM驱动的自进化的VLN框架。具体而言，SE-VLN 包含三个核心模块：层次化记忆模块用于将成功和失败案例转化为可重复使用的知识，检索增强的思想推理模块用于检索经验并支持多步决策，以及反思模块用于实现持续进化。全面的测试表明，SE-VLN 在未见过的环境中分别实现了57%和35.2%的导航成功率，相较于R2R和REVERSE数据集上的当前最先进的方法，分别有23.9%和15.0%的绝对性能提升。此外，SE-VLN 的性能随经验库的增加而提高，说明其作为VLN领域自进化的代理框架具有巨大的潜力。 

---
# DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model 

**Title (ZH)**: DINO-VO：基于特征的视觉里程计利用视觉基础模型 

**Authors**: Maulana Bisyir Azhari, David Hyunchul Shim  

**Link**: [PDF](https://arxiv.org/pdf/2507.13145)  

**Abstract**: Learning-based monocular visual odometry (VO) poses robustness, generalization, and efficiency challenges in robotics. Recent advances in visual foundation models, such as DINOv2, have improved robustness and generalization in various vision tasks, yet their integration in VO remains limited due to coarse feature granularity. In this paper, we present DINO-VO, a feature-based VO system leveraging DINOv2 visual foundation model for its sparse feature matching. To address the integration challenge, we propose a salient keypoints detector tailored to DINOv2's coarse features. Furthermore, we complement DINOv2's robust-semantic features with fine-grained geometric features, resulting in more localizable representations. Finally, a transformer-based matcher and differentiable pose estimation layer enable precise camera motion estimation by learning good matches. Against prior detector-descriptor networks like SuperPoint, DINO-VO demonstrates greater robustness in challenging environments. Furthermore, we show superior accuracy and generalization of the proposed feature descriptors against standalone DINOv2 coarse features. DINO-VO outperforms prior frame-to-frame VO methods on the TartanAir and KITTI datasets and is competitive on EuRoC dataset, while running efficiently at 72 FPS with less than 1GB of memory usage on a single GPU. Moreover, it performs competitively against Visual SLAM systems on outdoor driving scenarios, showcasing its generalization capabilities. 

**Abstract (ZH)**: 基于学习的单目视觉里程计（VO）面临着机器人应用中的稳健性、泛化能力和效率挑战。视觉基础模型（如DINOv2）的最新进展在各种视觉任务中提升了稳健性和泛化能力，但由于特征粒度过粗，其在VO中的集成仍受到限制。本文提出了一种名为DINO-VO的特征导向的VO系统，利用DINOv2视觉基础模型进行稀疏特征匹配。为了解决集成挑战，我们设计了一种针对DINOv2粗特征的显著关键点检测器。此外，我们还结合了精细几何特征，增强了DINOv2的鲁棒语义特征，从而产生了更具局部化的表示。最后，基于Transformer的匹配器和可微分位姿估计层能够通过学习好的匹配来精确定义摄像机运动。与SuperPoint等先前的检测描述符网络相比，DINO-VO在复杂环境中的稳健性更强。同时，我们展示了所提出的特征描述符在与独立的DINOv2粗特征对比时的优越准确性和泛化能力。DINO-VO在TartanAir和KITTI数据集上的帧间VO方法中表现优异，并且在EuRoC数据集上具有竞争力，同时在单个GPU上以低于1GB的内存使用率高效运行72 FPS。此外，它在户外驾驶场景中表现与视觉SLAM系统相当，展示了其泛化能力。 

---
# GraspGen: A Diffusion-based Framework for 6-DOF Grasping with On-Generator Training 

**Title (ZH)**: GraspGen：一种基于扩散的六自由度抓取框架及其生成器上的训练方法 

**Authors**: Adithyavairavan Murali, Balakumar Sundaralingam, Yu-Wei Chao, Wentao Yuan, Jun Yamada, Mark Carlson, Fabio Ramos, Stan Birchfield, Dieter Fox, Clemens Eppner  

**Link**: [PDF](https://arxiv.org/pdf/2507.13097)  

**Abstract**: Grasping is a fundamental robot skill, yet despite significant research advancements, learning-based 6-DOF grasping approaches are still not turnkey and struggle to generalize across different embodiments and in-the-wild settings. We build upon the recent success on modeling the object-centric grasp generation process as an iterative diffusion process. Our proposed framework, GraspGen, consists of a DiffusionTransformer architecture that enhances grasp generation, paired with an efficient discriminator to score and filter sampled grasps. We introduce a novel and performant on-generator training recipe for the discriminator. To scale GraspGen to both objects and grippers, we release a new simulated dataset consisting of over 53 million grasps. We demonstrate that GraspGen outperforms prior methods in simulations with singulated objects across different grippers, achieves state-of-the-art performance on the FetchBench grasping benchmark, and performs well on a real robot with noisy visual observations. 

**Abstract (ZH)**: 基于学习的6-DOF抓取方法仍然是现成的解决方案，并且难以在不同的身体构型和真实环境设置中泛化。我们构建了最近在将对象中心的抓取生成过程建模为迭代扩散过程方面取得成功的基础。我们提出的框架GraspGen包括一个增强抓取生成的扩散变换器架构，并配有一个高效鉴别器用于评分和筛选采样的抓取。我们引入了一种新的高性能的生成器内训练策略以训练鉴别器。为了将GraspGen扩展到对象和抓手，我们发布了包含超过5300万个抓取的新模拟数据集。我们展示了GraspGen在不同抓手的分离物体仿真中优于先前的方法，在FetchBench抓取基准上的性能达到最新水平，并且在具有噪声视觉观测的实际机器人上表现良好。 

---
# MUPAX: Multidimensional Problem Agnostic eXplainable AI 

**Title (ZH)**: MUPAX：多维度问题无因果假设可解释人工智能 

**Authors**: Vincenzo Dentamaro, Felice Franchini, Giuseppe Pirlo, Irina Voiculescu  

**Link**: [PDF](https://arxiv.org/pdf/2507.13090)  

**Abstract**: Robust XAI techniques should ideally be simultaneously deterministic, model agnostic, and guaranteed to converge. We propose MULTIDIMENSIONAL PROBLEM AGNOSTIC EXPLAINABLE AI (MUPAX), a deterministic, model agnostic explainability technique, with guaranteed convergency. MUPAX measure theoretic formulation gives principled feature importance attribution through structured perturbation analysis that discovers inherent input patterns and eliminates spurious relationships. We evaluate MUPAX on an extensive range of data modalities and tasks: audio classification (1D), image classification (2D), volumetric medical image analysis (3D), and anatomical landmark detection, demonstrating dimension agnostic effectiveness. The rigorous convergence guarantees extend to any loss function and arbitrary dimensions, making MUPAX applicable to virtually any problem context for AI. By contrast with other XAI methods that typically decrease performance when masking, MUPAX not only preserves but actually enhances model accuracy by capturing only the most important patterns of the original data. Extensive benchmarking against the state of the XAI art demonstrates MUPAX ability to generate precise, consistent and understandable explanations, a crucial step towards explainable and trustworthy AI systems. The source code will be released upon publication. 

**Abstract (ZH)**: 鲁棒的可解释人工智能技术应当同时具备确定性、模型无关性和收敛保证。我们提出了一种确定性、模型无关的解释性人工智能技术MULTIDIMENSIONAL PROBLEM AGNOSTIC EXPLAINABLE AI（MUPAX），其具有收敛保证。MUPAX的测度论表述通过结构化扰动分析提供原则上的特征重要性归因，发现固有的输入模式并消除虚假关系。我们在涵盖广泛数据模态和任务的评估中展示了MUPAX的维度无关有效性：音频分类（1D）、图像分类（2D）、医学影像体积分析（3D）和解剖标志检测。其严格的收敛保证适用于任何损失函数和任意维度，使MUPAX适用于几乎所有的人工智能问题上下文。与通常在遮蔽时降低性能的其他可解释性人工智能方法不同，MUPAX不仅保持了模型的准确性，而且还通过捕获原始数据中最重要的模式而增强了模型的准确性。与当前可解释性人工智能领域的最佳方法的广泛基准测试表明，MUPAX能够生成精确、一致且易于理解的解释，这是迈向可解释和可信赖的人工智能系统的关键步骤。源代码将在发表后公开。 

---
# Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities 

**Title (ZH)**: 重新审视视觉-语言导航中的身体差距：一种全面研究物理与视觉差异的探究 

**Authors**: Liuyi Wang, Xinyuan Xia, Hui Zhao, Hanqing Wang, Tai Wang, Yilun Chen, Chengju Liu, Qijun Chen, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2507.13019)  

**Abstract**: Recent Vision-and-Language Navigation (VLN) advancements are promising, but their idealized assumptions about robot movement and control fail to reflect physically embodied deployment challenges. To bridge this gap, we introduce VLN-PE, a physically realistic VLN platform supporting humanoid, quadruped, and wheeled robots. For the first time, we systematically evaluate several ego-centric VLN methods in physical robotic settings across different technical pipelines, including classification models for single-step discrete action prediction, a diffusion model for dense waypoint prediction, and a train-free, map-based large language model (LLM) integrated with path planning. Our results reveal significant performance degradation due to limited robot observation space, environmental lighting variations, and physical challenges like collisions and falls. This also exposes locomotion constraints for legged robots in complex environments. VLN-PE is highly extensible, allowing seamless integration of new scenes beyond MP3D, thereby enabling more comprehensive VLN evaluation. Despite the weak generalization of current models in physical deployment, VLN-PE provides a new pathway for improving cross-embodiment's overall adaptability. We hope our findings and tools inspire the community to rethink VLN limitations and advance robust, practical VLN models. The code is available at this https URL. 

**Abstract (ZH)**: 最近的Vision-and-Language Navigation (VLN)进展充满 promise，但其对机器人运动和控制的理想化假设未能反映实际身体化部署中的挑战。为了缩小这一差距，我们引入了VLN-PE，这是一个物理现实的VLN平台，支持人形、四足和轮式机器人。首次系统地评估了几种以自我为中心的VLN方法在不同技术管道下的物理机器人环境中，包括用于单步离散动作预测的分类模型、用于密集航点预测的扩散模型，以及基于路径规划的大语言模型（LLM），无需训练。我们的结果揭示了由于有限的机器人观察空间、环境光照变化以及碰撞和跌倒等物理挑战导致的重大性能下降。这还暴露了复杂环境中腿部机器人的运动限制。VLN-PE具有高度可扩展性，允许无缝集成超越MP3D的新场景，从而实现更全面的VLN评估。尽管当前模型在物理部署中的泛化能力较弱，但VLN-PE为提高跨实体适应性提供了新的途径。我们希望我们的发现和工具能激励社区重新思考VLN的局限性，并推动稳健、实用的VLN模型的发展。代码可在以下链接获取。 

---
# SMART: Relation-Aware Learning of Geometric Representations for Knowledge Graphs 

**Title (ZH)**: SMART：面向知识图的关系aware几何表示学习 

**Authors**: Kossi Amouzouvi, Bowen Song, Andrea Coletta, Luigi Bellomarini, Jens Lehmann, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2507.13001)  

**Abstract**: Knowledge graph representation learning approaches provide a mapping between symbolic knowledge in the form of triples in a knowledge graph (KG) and their feature vectors. Knowledge graph embedding (KGE) models often represent relations in a KG as geometric transformations. Most state-of-the-art (SOTA) KGE models are derived from elementary geometric transformations (EGTs), such as translation, scaling, rotation, and reflection, or their combinations. These geometric transformations enable the models to effectively preserve specific structural and relational patterns of the KG. However, the current use of EGTs by KGEs remains insufficient without considering relation-specific transformations. Although recent models attempted to address this problem by ensembling SOTA baseline models in different ways, only a single or composite version of geometric transformations are used by such baselines to represent all the relations. In this paper, we propose a framework that evaluates how well each relation fits with different geometric transformations. Based on this ranking, the model can: (1) assign the best-matching transformation to each relation, or (2) use majority voting to choose one transformation type to apply across all relations. That is, the model learns a single relation-specific EGT in low dimensional vector space through an attention mechanism. Furthermore, we use the correlation between relations and EGTs, which are learned in a low dimension, for relation embeddings in a high dimensional vector space. The effectiveness of our models is demonstrated through comprehensive evaluations on three benchmark KGs as well as a real-world financial KG, witnessing a performance comparable to leading models 

**Abstract (ZH)**: 知识图谱表示学习方法将知识图谱（KG）中以三元组形式表示的符号知识映射到其特征向量。大多数最先进的（SOTA）知识图谱嵌入（KGE）模型是从基本几何变换（EGTs），如平移、缩放、旋转和反射，或其组合派生而来。这些几何变换使模型能够有效地保留知识图谱的特定结构和关系模式。然而，目前的KGEs在使用EGTs时尚未充分考虑关系特异性变换。尽管最近的模型试图通过以不同方式集成SOTA基线模型来解决这一问题，这些基线模型仅使用单一或组合的几何变换来表示所有关系。在本文中，我们提出了一种框架，以评估每种关系与不同几何变换的匹配程度，并基于此排名：（1）为每种关系分配最佳匹配的变换，或（2）使用多数投票选择一种变换类型应用于所有关系。即，模型通过注意力机制学习低维向量空间中的单个关系特异性EGT。此外，我们利用关系和EGTs在低维空间中学习到的相关性，在高维向量空间中为关系嵌入提供支持。通过在三个基准知识图谱以及一个实时金融知识图谱上的全面评估，证明了我们模型的有效性，表现可媲美领先模型。 

---
# Teach Old SAEs New Domain Tricks with Boosting 

**Title (ZH)**: 使用提升技术让旧SAEs学习新领域技巧 

**Authors**: Nikita Koriagin, Yaroslav Aksenov, Daniil Laptev, Gleb Gerasimov, Nikita Balagansky, Daniil Gavrilov  

**Link**: [PDF](https://arxiv.org/pdf/2507.12990)  

**Abstract**: Sparse Autoencoders have emerged as powerful tools for interpreting the internal representations of Large Language Models, yet they often fail to capture domain-specific features not prevalent in their training corpora. This paper introduces a residual learning approach that addresses this feature blindness without requiring complete retraining. We propose training a secondary SAE specifically to model the reconstruction error of a pretrained SAE on domain-specific texts, effectively capturing features missed by the primary model. By summing the outputs of both models during inference, we demonstrate significant improvements in both LLM cross-entropy and explained variance metrics across multiple specialized domains. Our experiments show that this method efficiently incorporates new domain knowledge into existing SAEs while maintaining their performance on general tasks. This approach enables researchers to selectively enhance SAE interpretability for specific domains of interest, opening new possibilities for targeted mechanistic interpretability of LLMs. 

**Abstract (ZH)**: 稀疏自主编码器已发展成为解读大型语言模型内部表示的强大多用途工具，然而它们往往无法捕获在训练语料中不普遍的领域特异性特征。本文提出了一种残差学习方法，该方法在无需完全重新训练的情况下解决了这种特征盲区问题。我们建议训练一个次级SAE，专门用于建模预训练SAE在特定领域文本上的重构误差，从而有效捕获主要模型遗漏的特征。通过推理时将两个模型的输出相加，我们在多个专门领域展示了在交叉熵和解释方差指标上的显著改进。我们的实验表明，该方法能够高效地将新领域知识整合进现有的SAE中，同时保持其在通用任务上的性能。这种方法允许研究人员有选择地增强SAE对特定领域兴趣的可解释性，为大型语言模型的针对性机制解释打开了新的可能性。 

---
# MRT at IberLEF-2025 PRESTA Task: Maximizing Recovery from Tables with Multiple Steps 

**Title (ZH)**: MRT在IberLEF-2025 PRESTA任务中：通过多步操作最大化表格内容恢复 

**Authors**: Maximiliano Hormazábal Lagos, Álvaro Bueno Sáez, Héctor Cerezo-Costas, Pedro Alonso Doval, Jorge Alcalde Vesteiro  

**Link**: [PDF](https://arxiv.org/pdf/2507.12981)  

**Abstract**: This paper presents our approach for the IberLEF 2025 Task PRESTA: Preguntas y Respuestas sobre Tablas en Español (Questions and Answers about Tables in Spanish). Our solution obtains answers to the questions by implementing Python code generation with LLMs that is used to filter and process the table. This solution evolves from the MRT implementation for the Semeval 2025 related task. The process consists of multiple steps: analyzing and understanding the content of the table, selecting the useful columns, generating instructions in natural language, translating these instructions to code, running it, and handling potential errors or exceptions. These steps use open-source LLMs and fine-grained optimized prompts for each step. With this approach, we achieved an accuracy score of 85\% in the task. 

**Abstract (ZH)**: 本文介绍了我们针对IberLEF 2025任务PRESTA：关于西班牙语表格的问题与答案的解决方案。我们的方法通过使用LLMs生成Python代码来筛选和处理表格以获取问题的答案。该解决方案是从Semeval 2025相关任务的MRT实现发展而来。过程包括多个步骤：分析和理解表格内容、选择有用列、生成自然语言指令、将这些指令翻译成代码、运行代码并处理潜在的错误或异常。这些步骤使用开源LLMs和细粒度优化的提示。通过这种方法，我们在任务中达到了85%的准确率。 

---
# A Distributed Generative AI Approach for Heterogeneous Multi-Domain Environments under Data Sharing constraints 

**Title (ZH)**: 一种在数据共享约束下的异构多域环境分布式生成AI方法 

**Authors**: Youssef Tawfilis, Hossam Amer, Minar El-Aasser, Tallal Elshabrawy  

**Link**: [PDF](https://arxiv.org/pdf/2507.12979)  

**Abstract**: Federated Learning has gained increasing attention for its ability to enable multiple nodes to collaboratively train machine learning models without sharing their raw data. At the same time, Generative AI -- particularly Generative Adversarial Networks (GANs) -- have achieved remarkable success across a wide range of domains, such as healthcare, security, and Image Generation. However, training generative models typically requires large datasets and significant computational resources, which are often unavailable in real-world settings. Acquiring such resources can be costly and inefficient, especially when many underutilized devices -- such as IoT devices and edge devices -- with varying capabilities remain idle. Moreover, obtaining large datasets is challenging due to privacy concerns and copyright restrictions, as most devices are unwilling to share their data. To address these challenges, we propose a novel approach for decentralized GAN training that enables the utilization of distributed data and underutilized, low-capability devices while not sharing data in its raw form. Our approach is designed to tackle key challenges in decentralized environments, combining KLD-weighted Clustered Federated Learning to address the issues of data heterogeneity and multi-domain datasets, with Heterogeneous U-Shaped split learning to tackle the challenge of device heterogeneity under strict data sharing constraints -- ensuring that no labels or raw data, whether real or synthetic, are ever shared between nodes. Experimental results shows that our approach demonstrates consistent and significant improvements across key performance metrics, where it achieves 1.1x -- 2.2x higher image generation scores, an average 10% boost in classification metrics (up to 50% in multi-domain non-IID settings), in much lower latency compared to several benchmarks. Find our code at this https URL. 

**Abstract (ZH)**: 联邦学习由于其能够使多个节点协作训练机器学习模型而无需共享原始数据的能力，引起了越来越多的关注。同时，生成性人工智能——特别是生成对抗网络（GANs）——在医疗保健、安全和图像生成等领域取得了显著成功。然而，训练生成模型通常需要大量数据集和显著的计算资源，这些资源在实际应用场景中往往无法获得。获取这些资源可能代价高昂且效率低下，尤其是当大量低利用率且能力各异的设备（如IoT设备和边缘设备）处于闲置状态时。此外，由于隐私担忧和版权限制，获得大量数据集也非常具有挑战性，因为大多数设备不愿意共享其数据。为了解决这些挑战，我们提出了一种新的去中心化GAN训练方法，该方法能够利用分布式数据和低利用率的低能力设备，而不以原始形式共享数据。该方法旨在解决去中心化环境中的一些关键挑战，结合KLD加权聚类联邦学习以应对数据异质性和多域数据集的问题，并采用异构U形分裂学习以在严格的数据共享约束下应对设备异质性挑战——确保节点之间永不共享任何标签或原始数据，无论是真实数据还是合成数据。实验结果表明，我们的方法在关键性能指标上表现出一致且显著的改进，其中在图像生成指标上达到1.1至2.2倍的提升，在分类指标上平均提升10%（在多域非IID设置中最高可提升50%），且在较低的延迟下实现。我们的代码请参见此链接：[这里](https://)。 

---
# Demographic-aware fine-grained classification of pediatric wrist fractures 

**Title (ZH)**: 基于人口统计学的儿童腕部骨折细粒度分类 

**Authors**: Ammar Ahmed, Ali Shariq Imran, Zenun Kastrati, Sher Muhammad Daudpota  

**Link**: [PDF](https://arxiv.org/pdf/2507.12964)  

**Abstract**: Wrist pathologies are frequently observed, particularly among children who constitute the majority of fracture cases. However, diagnosing these conditions is time-consuming and requires specialized expertise. Computer vision presents a promising avenue, contingent upon the availability of extensive datasets, a notable challenge in medical imaging. Therefore, reliance solely on one modality, such as images, proves inadequate, especially in an era of diverse and plentiful data types. In this study, we employ a multifaceted approach to address the challenge of recognizing wrist pathologies using an extremely limited dataset. Initially, we approach the problem as a fine-grained recognition task, aiming to identify subtle X-ray pathologies that conventional CNNs overlook. Secondly, we enhance network performance by fusing patient metadata with X-ray images. Thirdly, rather than pre-training on a coarse-grained dataset like ImageNet, we utilize weights trained on a fine-grained dataset. While metadata integration has been used in other medical domains, this is a novel application for wrist pathologies. Our results show that a fine-grained strategy and metadata integration improve diagnostic accuracy by 2% with a limited dataset and by over 10% with a larger fracture-focused dataset. 

**Abstract (ZH)**: 腕部病理学常见于儿童，构成骨折病例的多数。然而，诊断这些状况耗时且需要专门的技能。计算机视觉提供了一条有希望的道路，但大量数据集的可用性是一个显著挑战。因此，仅依赖单一模态（如图像）是不足够的，尤其是在数据类型多样和丰富的情况下。在本研究中，我们采用多模态方法，利用极其有限的数据集识别腕部病理学。首先，我们将问题视为细粒度识别任务，旨在识别传统CNN忽视的细微X光病理学。其次，通过融合患者元数据和X光图像来提升网络性能。第三，我们不使用像ImageNet这样的粗粒度数据集进行预训练，而是使用细粒度数据集训练的权重。虽然在其他医疗领域已经采用元数据集成，但这是在腕部病理学中的新应用。我们的结果表明，细粒度策略和元数据集成在有限数据集和更大规模的骨折集中，诊断精度分别提高了2%和超过10%。 

---
# Improving Diagnostic Accuracy of Pigmented Skin Lesions With CNNs: an Application on the DermaMNIST Dataset 

**Title (ZH)**: 使用CNN提高色素性皮肤病变诊断准确率：DermaMNIST数据集的应用 

**Authors**: Nerma Kadric, Amila Akagic, Medina Kapo  

**Link**: [PDF](https://arxiv.org/pdf/2507.12961)  

**Abstract**: Pigmented skin lesions represent localized areas of increased melanin and can indicate serious conditions like melanoma, a major contributor to skin cancer mortality. The MedMNIST v2 dataset, inspired by MNIST, was recently introduced to advance research in biomedical imaging and includes DermaMNIST, a dataset for classifying pigmented lesions based on the HAM10000 dataset. This study assesses ResNet-50 and EfficientNetV2L models for multi-class classification using DermaMNIST, employing transfer learning and various layer configurations. One configuration achieves results that match or surpass existing methods. This study suggests that convolutional neural networks (CNNs) can drive progress in biomedical image analysis, significantly enhancing diagnostic accuracy. 

**Abstract (ZH)**: 色素皮肤病变代表局部 melanin 增加的区域，并可能指示如恶性黑色素瘤等严重状况，恶性黑色素瘤是导致皮肤癌死亡的主要因素之一。MedMNIST v2 数据集受 MNIST 启发， recently introduced to推进生物医学成像领域的研究，其中包括 DermaMNIST，一个基于 HAM10000 数据集的用于分类色素皮肤病变的数据集。本研究评估了 ResNet-50 和 EfficientNetV2L 模型在 DermaMNIST 上的多类别分类性能，采用了迁移学习和多种层配置。一种配置达到了与现有方法相当或更优的结果。本研究建议卷积神经网络 (CNNs) 可以推动生物医学图像分析的进步，显著提高诊断准确性。 

---
# UniSLU: Unified Spoken Language Understanding from Heterogeneous Cross-Task Datasets 

**Title (ZH)**: UniSLU：统一的口语理解从异构跨任务数据集 

**Authors**: Zhichao Sheng, Shilin Zhou, Chen Gong, Zhenghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12951)  

**Abstract**: Spoken Language Understanding (SLU) plays a crucial role in speech-centric multimedia applications, enabling machines to comprehend spoken language in scenarios such as meetings, interviews, and customer service interactions. SLU encompasses multiple tasks, including Automatic Speech Recognition (ASR), spoken Named Entity Recognition (NER), and spoken Sentiment Analysis (SA). However, existing methods often rely on separate model architectures for individual tasks such as spoken NER and SA, which increases system complexity, limits cross-task interaction, and fails to fully exploit heterogeneous datasets available across tasks. To address these limitations, we propose UniSLU, a unified framework that jointly models multiple SLU tasks within a single architecture. Specifically, we propose a unified representation for diverse SLU tasks, enabling full utilization of heterogeneous datasets across multiple tasks. Built upon this representation, we propose a unified generative method that jointly models ASR, spoken NER, and SA tasks, enhancing task interactions and enabling seamless integration with large language models to harness their powerful generative capabilities. Extensive experiments on public SLU datasets demonstrate the effectiveness of our approach, achieving superior SLU performance compared to several benchmark methods, making it well-suited for real-world speech-based multimedia scenarios. We will release all code and models at github to facilitate future research. 

**Abstract (ZH)**: 统一口语理解：联合建模多种口语理解任务以优化多模态语音应用 

---
# MC$^2$A: Enabling Algorithm-Hardware Co-Design for Efficient Markov Chain Monte Carlo Acceleration 

**Title (ZH)**: MC$^2$A:  Enables 算法-硬件协同设计以实现高效的马尔可夫链蒙特卡洛加速 

**Authors**: Shirui Zhao, Jun Yin, Lingyun Yao, Martin Andraud, Wannes Meert, Marian Verhelst  

**Link**: [PDF](https://arxiv.org/pdf/2507.12935)  

**Abstract**: An increasing number of applications are exploiting sampling-based algorithms for planning, optimization, and inference. The Markov Chain Monte Carlo (MCMC) algorithms form the computational backbone of this emerging branch of machine learning. Unfortunately, the high computational cost limits their feasibility for large-scale problems and real-world applications, and the existing MCMC acceleration solutions are either limited in hardware flexibility or fail to maintain efficiency at the system level across a variety of end-to-end applications. This paper introduces \textbf{MC$^2$A}, an algorithm-hardware co-design framework, enabling efficient and flexible optimization for MCMC acceleration. Firstly, \textbf{MC$^2$A} analyzes the MCMC workload diversity through an extension of the processor performance roofline model with a 3rd dimension to derive the optimal balance between the compute, sampling and memory parameters. Secondly, \textbf{MC$^2$A} proposes a parametrized hardware accelerator architecture with flexible and efficient support of MCMC kernels with a pipeline of ISA-programmable tree-structured processing units, reconfigurable samplers and a crossbar interconnect to support irregular access. Thirdly, the core of \textbf{MC$^2$A} is powered by a novel Gumbel sampler that eliminates exponential and normalization operations. In the end-to-end case study, \textbf{MC$^2$A} achieves an overall {$307.6\times$, $1.4\times$, $2.0\times$, $84.2\times$} speedup compared to the CPU, GPU, TPU and state-of-the-art MCMC accelerator. Evaluated on various representative MCMC workloads, this work demonstrates and exploits the feasibility of general hardware acceleration to popularize MCMC-based solutions in diverse application domains. 

**Abstract (ZH)**: MC²A：一种算法-硬件协同设计框架，实现MCMC加速的高效与灵活优化 

---
# DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization 

**Title (ZH)**: DMQ: 解析扩散模型中的异常值以实现后训练量化 

**Authors**: Dongyeun Lee, Jiwan Hur, Hyounguk Shon, Jae Young Lee, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.12933)  

**Abstract**: Diffusion models have achieved remarkable success in image generation but come with significant computational costs, posing challenges for deployment in resource-constrained environments. Recent post-training quantization (PTQ) methods have attempted to mitigate this issue by focusing on the iterative nature of diffusion models. However, these approaches often overlook outliers, leading to degraded performance at low bit-widths. In this paper, we propose a DMQ which combines Learned Equivalent Scaling (LES) and channel-wise Power-of-Two Scaling (PTS) to effectively address these challenges. Learned Equivalent Scaling optimizes channel-wise scaling factors to redistribute quantization difficulty between weights and activations, reducing overall quantization error. Recognizing that early denoising steps, despite having small quantization errors, crucially impact the final output due to error accumulation, we incorporate an adaptive timestep weighting scheme to prioritize these critical steps during learning. Furthermore, identifying that layers such as skip connections exhibit high inter-channel variance, we introduce channel-wise Power-of-Two Scaling for activations. To ensure robust selection of PTS factors even with small calibration set, we introduce a voting algorithm that enhances reliability. Extensive experiments demonstrate that our method significantly outperforms existing works, especially at low bit-widths such as W4A6 (4-bit weight, 6-bit activation) and W4A8, maintaining high image generation quality and model stability. The code is available at this https URL. 

**Abstract (ZH)**: 扩散模型在图像生成任务中取得了显著成功，但伴随较高的计算成本，这在资源受限环境中面临部署挑战。最近的后训练量化（PTQ）方法通过关注扩散模型的迭代性质来尝试缓解这一问题，但这些方法往往忽视了异常值的存在，导致在低位宽下性能下降。在本文中，我们提出了一种结合了学习等效缩放（LES）和通道 wise 二幂次缩放（PTS）的DMQ方法，有效应对这些挑战。学习等效缩放优化了通道 wise 缩放因子，以在权重和激活之间重新分配量化难度，减少总体量化误差。我们认识到，尽管早期降噪步骤的量化误差较小，但由于误差累积，这些步骤对最终输出至关重要，因此引入了自适应时间步长加权方案，在学习过程中优先处理这些关键步骤。进一步识别到跳过连接等层具有高通道间方差的特性，我们为激活引入了通道 wise 二幂次缩放。为了确保在小校准集下量化因子的稳健选择，我们引入了一种投票算法以增强可靠性。广泛的实验表明，我们的方法在低位宽场景（如W4A6和W4A8）下显著优于现有方法，同时保持了高图像生成质量和模型稳定性。代码可通过以下链接获取：this https URL。 

---
# Making Language Model a Hierarchical Classifier and Generator 

**Title (ZH)**: 使语言模型成为层次分类器和生成器 

**Authors**: Yihong Wang, Zhonglin Jiang, Ningyuan Xi, Yue Zhao, Qingqing Gu, Xiyuan Chen, Hao Wu, Sheng Xu, Hange Zhou, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.12930)  

**Abstract**: Decoder-only language models, such as GPT and LLaMA, generally decode on the last layer. Motivated by human's hierarchical thinking capability, we propose that a hierarchical decoder architecture could be built with different layers decoding texts simultaneously. Due to limited time and computationally resources, we choose to adapt a pretrained language model into this form of hierarchical decoder. Language heads of the last layer are copied to different selected intermediate layers, and fine-tuned with different task inputs. By thorough experiments, we validate that these selective intermediate layers could be adapted to speak meaningful and reasonable contents, and this paradigm of hierarchical decoder can obtain state-of-the-art performances on multiple tasks such as hierarchical text classification, classification-guided generation, and hierarchical text generation. This study suggests the possibility of a generalized hierarchical reasoner, pretraining from scratch. 

**Abstract (ZH)**: 仅解码器架构的语言模型，如GPT和LLaMA，通常在最后一层进行解码。受人类分层思维能力的启发，我们提出可以构建一种不同层同时解码文本的分层解码器架构。由于时间和计算资源的限制，我们选择将一个预训练语言模型调整为这种分层解码器的形式。最后一层的语言头被复制到不同的选定中间层，并在不同的任务输入下进行微调。通过 thorough 实验，我们验证了这些选择性的中间层可以被调整以生成有意义和合理的文本内容，并且这种分层解码器的范式可以在分级文本分类、分类指导生成和分级文本生成等多个任务上获得最新的性能。这项研究建议从头开始预训练一个通用的分层推理器的可能性。 

---
# Argus: Leveraging Multiview Images for Improved 3-D Scene Understanding With Large Language Models 

**Title (ZH)**: Argus: 利用多视角图像提升大型语言模型在三维场景理解中的能力 

**Authors**: Yifan Xu, Chao Zhang, Hanqi Jiang, Xiaoyan Wang, Ruifei Ma, Yiwei Li, Zihao Wu, Zeju Li, Xiangde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12916)  

**Abstract**: Advancements in foundation models have made it possible to conduct applications in various downstream tasks. Especially, the new era has witnessed a remarkable capability to extend Large Language Models (LLMs) for tackling tasks of 3D scene understanding. Current methods rely heavily on 3D point clouds, but the 3D point cloud reconstruction of an indoor scene often results in information loss. Some textureless planes or repetitive patterns are prone to omission and manifest as voids within the reconstructed 3D point clouds. Besides, objects with complex structures tend to introduce distortion of details caused by misalignments between the captured images and the dense reconstructed point clouds. 2D multi-view images present visual consistency with 3D point clouds and provide more detailed representations of scene components, which can naturally compensate for these deficiencies. Based on these insights, we propose Argus, a novel 3D multimodal framework that leverages multi-view images for enhanced 3D scene understanding with LLMs. In general, Argus can be treated as a 3D Large Multimodal Foundation Model (3D-LMM) since it takes various modalities as input(text instructions, 2D multi-view images, and 3D point clouds) and expands the capability of LLMs to tackle 3D tasks. Argus involves fusing and integrating multi-view images and camera poses into view-as-scene features, which interact with the 3D features to create comprehensive and detailed 3D-aware scene embeddings. Our approach compensates for the information loss while reconstructing 3D point clouds and helps LLMs better understand the 3D world. Extensive experiments demonstrate that our method outperforms existing 3D-LMMs in various downstream tasks. 

**Abstract (ZH)**: Advancements in 基础模型使各种下游任务的应用成为可能。特别是，新的时代见证了大型语言模型（LLMs）扩展以解决3D场景理解任务的能力。当前的方法主要依赖于3D点云，但室内场景的3D点云重建往往会丢失信息。一些无纹理的平面或重复的模式容易被遗漏，表现为重建3D点云中的空洞。此外，结构复杂的物体由于捕获的图像与密集重建的点云之间的对齐错误，往往会引入细节失真。2D多视图图像与3D点云视觉上具有一致性，提供了场景组件的更详细表示，可以自然地弥补这些缺陷。基于这些见解，我们提出Argus，一种新颖的3D多模态框架，利用多视图图像增强LLMs的3D场景理解能力。总体来说，Argus可以被视为一个3D大型多模态基础模型（3D-LMM），因为它接受多种模态（文本指令、2D多视图图像和3D点云）作为输入，并扩展了LLMs解决3D任务的能力。Argus涉及将多视图图像和相机姿态融合成视图即场景特征，并与3D特征交互，以创建综合且详细的3D感知场景嵌入。我们的方法在重建3D点云时补偿信息损失，并帮助LLMs更好地理解3D世界。广泛的实验表明，我们的方法在各种下游任务中优于现有的3D-LMM。 

---
# An ultra-low-power CGRA for accelerating Transformers at the edge 

**Title (ZH)**: 边缘加速Transformer模型的超低功耗CGRA 

**Authors**: Rohit Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2507.12904)  

**Abstract**: Transformers have revolutionized deep learning with applications in natural language processing, computer vision, and beyond. However, their computational demands make it challenging to deploy them on low-power edge devices. This paper introduces an ultra-low-power, Coarse-Grained Reconfigurable Array (CGRA) architecture specifically designed to accelerate General Matrix Multiplication (GEMM) operations in transformer models tailored for the energy and resource constraints of edge applications. The proposed architecture integrates a 4 x 4 array of Processing Elements (PEs) for efficient parallel computation and dedicated 4 x 2 Memory Operation Blocks (MOBs) for optimized LOAD/STORE operations, reducing memory bandwidth demands and enhancing data reuse. A switchless mesh torus interconnect network further minimizes power and latency by enabling direct communication between PEs and MOBs, eliminating the need for centralized switching. Through its heterogeneous array design and efficient dataflow, this CGRA architecture addresses the unique computational needs of transformers, offering a scalable pathway to deploy sophisticated machine learning models on edge devices. 

**Abstract (ZH)**: 基于细粒度可重构阵列的超低功耗GEMM加速器设计：针对边缘应用的能量和资源限制 

---
# Generative Multi-Target Cross-Domain Recommendation 

**Title (ZH)**: 生成式多目标跨域推荐 

**Authors**: Jinqiu Jin, Yang Zhang, Junwei Pan, Fuli Feng, Hua Lu, Haijie Gu, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2507.12871)  

**Abstract**: Recently, there has been a surge of interest in Multi-Target Cross-Domain Recommendation (MTCDR), which aims to enhance recommendation performance across multiple domains simultaneously. Existing MTCDR methods primarily rely on domain-shared entities (\eg users or items) to fuse and transfer cross-domain knowledge, which may be unavailable in non-overlapped recommendation scenarios. Some studies model user preferences and item features as domain-sharable semantic representations, which can be utilized to tackle the MTCDR task. Nevertheless, they often require extensive auxiliary data for pre-training. Developing more effective solutions for MTCDR remains an important area for further exploration.
Inspired by recent advancements in generative recommendation, this paper introduces GMC, a generative paradigm-based approach for multi-target cross-domain recommendation. The core idea of GMC is to leverage semantically quantized discrete item identifiers as a medium for integrating multi-domain knowledge within a unified generative model. GMC first employs an item tokenizer to generate domain-shared semantic identifiers for each item, and then formulates item recommendation as a next-token generation task by training a domain-unified sequence-to-sequence model. To further leverage the domain information to enhance performance, we incorporate a domain-aware contrastive loss into the semantic identifier learning, and perform domain-specific fine-tuning on the unified recommender. Extensive experiments on five public datasets demonstrate the effectiveness of GMC compared to a range of baseline methods. 

**Abstract (ZH)**: 最近，多目标跨域推荐（MTCDR）引起了广泛关注，其目标是同时在多个领域内提升推荐性能。现有MTCDR方法主要依赖领域共享实体（例如用户或物品）来融合和转移跨域知识，但在无重叠推荐场景中这些实体可能不可用。一些研究将用户偏好和物品特征建模为可跨域共享的语义表示，以应对MTCDR任务。然而，它们往往需要大量辅助数据进行预训练。开发更有效的MTCDR解决方案仍然是一个重要的研究方向。

受生成推荐领域最新进展的启发，本文提出了GMC，一种基于生成范式的多目标跨域推荐方法。GMC的核心思想是利用语义量化离散物品标识符作为媒介，在统一的生成模型中整合多域知识。GMC首先使用物品分词器为每个物品生成领域共享的语义标识符，然后通过训练一个领域统一的序列到序列模型将物品推荐任务形式化为下一个标识符生成任务。为进一步利用领域信息提升性能，我们在语义标识符学习中引入了领域感知对比损失，并在统一推荐器中进行领域特定的微调。在五个公开数据集上的广泛实验结果表明，GMC相较于一系列基线方法具有更高的有效性。 

---
# Supervised Fine Tuning on Curated Data is Reinforcement Learning (and can be improved) 

**Title (ZH)**: 监督微调在精选数据上的应用是强化学习（并且可以改进） 

**Authors**: Chongli Qin, Jost Tobias Springenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.12856)  

**Abstract**: Behavior Cloning (BC) on curated (or filtered) data is the predominant paradigm for supervised fine-tuning (SFT) of large language models; as well as for imitation learning of control policies. Here, we draw on a connection between this successful strategy and the theory and practice of finding optimal policies via Reinforcement Learning (RL). Building on existing literature, we clarify that SFT can be understood as maximizing a lower bound on the RL objective in a sparse reward setting. Giving support to its often observed good performance. From this viewpoint, we realize that a small modification to SFT leads to an importance weighted variant that behaves closer to training with RL as it: i) optimizes a tighter bound to the RL objective and, ii) can improve performance compared to SFT on curated data. We refer to this variant as importance weighted supervised fine-tuning (iw-SFT). We show that it is easy to implement and can be further generalized to training with quality scored data. The resulting SFT variants are competitive with more advanced RL algorithms for large language models and for training policies in continuous control tasks. For example achieving 66.7% on the AIME 2024 dataset. 

**Abstract (ZH)**: 基于精选数据的行为克隆（BC）是监督微调（SFT）大语言模型和模仿学习控制策略的主要范式；本研究通过强化学习（RL）理论与实践的联系，阐明SFT可以理解为在稀疏奖励设置下最大化RL目标的下界，从而解释其常观察到的良好性能。我们提出一种对SFT的小修改，形成一种加权重要性采样的变种（iw-SFT），这种变种更接近于利用RL进行训练，并在精选数据上可能表现更优。我们展示了这一变种易于实现，并能进一步推广到质量评分数据的训练。所得的SFT变种在大语言模型和连续控制任务政策训练中与更先进的RL算法具有竞争力，例如在AIME 2024数据集上达到66.7%。 

---
# Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering 

**Title (ZH)**: 进入思维宝宫：长期主动体态问答中的推理与规划 

**Authors**: Muhammad Fadhil Ginting, Dong-Ki Kim, Xiangyun Meng, Andrzej Reinke, Bandi Jai Krishna, Navid Kayhani, Oriana Peltzer, David D. Fan, Amirreza Shaban, Sung-Kyun Kim, Mykel J. Kochenderfer, Ali-akbar Agha-mohammadi, Shayegan Omidshafiei  

**Link**: [PDF](https://arxiv.org/pdf/2507.12846)  

**Abstract**: As robots become increasingly capable of operating over extended periods -- spanning days, weeks, and even months -- they are expected to accumulate knowledge of their environments and leverage this experience to assist humans more effectively. This paper studies the problem of Long-term Active Embodied Question Answering (LA-EQA), a new task in which a robot must both recall past experiences and actively explore its environment to answer complex, temporally-grounded questions. Unlike traditional EQA settings, which typically focus either on understanding the present environment alone or on recalling a single past observation, LA-EQA challenges an agent to reason over past, present, and possible future states, deciding when to explore, when to consult its memory, and when to stop gathering observations and provide a final answer. Standard EQA approaches based on large models struggle in this setting due to limited context windows, absence of persistent memory, and an inability to combine memory recall with active exploration. To address this, we propose a structured memory system for robots, inspired by the mind palace method from cognitive science. Our method encodes episodic experiences as scene-graph-based world instances, forming a reasoning and planning algorithm that enables targeted memory retrieval and guided navigation. To balance the exploration-recall trade-off, we introduce value-of-information-based stopping criteria that determines when the agent has gathered sufficient information. We evaluate our method on real-world experiments and introduce a new benchmark that spans popular simulation environments and actual industrial sites. Our approach significantly outperforms state-of-the-art baselines, yielding substantial gains in both answer accuracy and exploration efficiency. 

**Abstract (ZH)**: 长期主动体 WEST 迷你问答：记忆系统在机器人中的应用 

---
# SEMT: Static-Expansion-Mesh Transformer Network Architecture for Remote Sensing Image Captioning 

**Title (ZH)**: SEMT：静态扩展网 Transformer 网络架构用于遥感图像captioning 

**Authors**: Khang Truong, Lam Pham, Hieu Tang, Jasmin Lampert, Martin Boyer, Son Phan, Truong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.12845)  

**Abstract**: Image captioning has emerged as a crucial task in the intersection of computer vision and natural language processing, enabling automated generation of descriptive text from visual content. In the context of remote sensing, image captioning plays a significant role in interpreting vast and complex satellite imagery, aiding applications such as environmental monitoring, disaster assessment, and urban planning. This motivates us, in this paper, to present a transformer based network architecture for remote sensing image captioning (RSIC) in which multiple techniques of Static Expansion, Memory-Augmented Self-Attention, Mesh Transformer are evaluated and integrated. We evaluate our proposed models using two benchmark remote sensing image datasets of UCM-Caption and NWPU-Caption. Our best model outperforms the state-of-the-art systems on most of evaluation metrics, which demonstrates potential to apply for real-life remote sensing image systems. 

**Abstract (ZH)**: 基于变压器的遥感图像字幕网络架构研究 

---
# MVA 2025 Small Multi-Object Tracking for Spotting Birds Challenge: Dataset, Methods, and Results 

**Title (ZH)**: MVA 2025小型多目标跟踪挑战——鸟类识别：数据集、方法和结果 

**Authors**: Yuki Kondo, Norimichi Ukita, Riku Kanayama, Yuki Yoshida, Takayuki Yamaguchi, Xiang Yu, Guang Liang, Xinyao Liu, Guan-Zhang Wang, Wei-Ta Chu, Bing-Cheng Chuang, Jia-Hua Lee, Pin-Tseng Kuo, I-Hsuan Chu, Yi-Shein Hsiao, Cheng-Han Wu, Po-Yi Wu, Jui-Chien Tsou, Hsuan-Chi Liu, Chun-Yi Lee, Yuan-Fu Yang, Kosuke Shigematsu, Asuka Shin, Ba Tran  

**Link**: [PDF](https://arxiv.org/pdf/2507.12832)  

**Abstract**: Small Multi-Object Tracking (SMOT) is particularly challenging when targets occupy only a few dozen pixels, rendering detection and appearance-based association unreliable. Building on the success of the MVA2023 SOD4SB challenge, this paper introduces the SMOT4SB challenge, which leverages temporal information to address limitations of single-frame detection. Our three main contributions are: (1) the SMOT4SB dataset, consisting of 211 UAV video sequences with 108,192 annotated frames under diverse real-world conditions, designed to capture motion entanglement where both camera and targets move freely in 3D; (2) SO-HOTA, a novel metric combining Dot Distance with HOTA to mitigate the sensitivity of IoU-based metrics to small displacements; and (3) a competitive MVA2025 challenge with 78 participants and 308 submissions, where the winning method achieved a 5.1x improvement over the baseline. This work lays a foundation for advancing SMOT in UAV scenarios with applications in bird strike avoidance, agriculture, fisheries, and ecological monitoring. 

**Abstract (ZH)**: 小型多目标跟踪（SMOT）在目标仅占用几十个像素时特别具有挑战性，使得检测和基于外观的关联不可靠。在MVA2023 SOD4SB挑战取得成功的基础上，本文提出SMOT4SB挑战，该挑战利用时间信息解决单帧检测的限制。本文的主要贡献包括：（1）SMOT4SB数据集，包含211个无人机视频序列和108,192个标注框，涵盖多种真实世界条件，旨在捕捉自由移动的相机和目标在三维空间中的运动纠缠；（2）SO-HOTA，一种结合Dot Distance和HOTA的新指标，减少基于IoU的指标对小位移的敏感性；（3）MVA2025挑战，共有78名参与者和308份提交，获胜方法对比基线实现了5.1倍的改进。本文为推进无人机场景下的小型多目标跟踪奠定了基础，并应用于鸟类撞击避免、农业、渔业和生态监测等领域。 

---
# Feature-Enhanced TResNet for Fine-Grained Food Image Classification 

**Title (ZH)**: 特征增强的TResNet在细粒度食品图像分类中的应用 

**Authors**: Lulu Liu, Zhiyong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.12828)  

**Abstract**: Food is not only a core component of humans' daily diets, but also an important carrier of cultural heritage and emotional bonds. With the development of technology, the need for accurate classification of food images has grown, which is crucial for a variety of application scenarios. However, existing Convolutional Neural Networks (CNNs) face significant challenges when dealing with fine-grained food images that are similar in shape but subtle in detail. To address this challenge, this study presents an innovative method for classifying food images, named Feature-Enhanced TResNet (FE-TResNet), specifically designed to address fine-grained food images and accurately capture subtle features within them. The FE-TResNet method is based on the TResNet model and integrates Style-based Recalibration Module (StyleRM) and Deep Channel-wise Attention (DCA) technologies to enhance feature extraction capabilities. In experimental validation on Chinese food image datasets ChineseFoodNet and CNFOOD-241, the FE-TResNet method significantly improved classification accuracy, achieving rates of 81.37% and 80.29%, respectively, demonstrating its effectiveness and superiority in fine-grained food image classification. 

**Abstract (ZH)**: 食物不仅是人类日常饮食的核心组成部分，也是文化传承和情感纽带的重要载体。随着技术的发展，准确分类食物图像的需求日益增长，这对于多种应用场景至关重要。然而，现有卷积神经网络（CNNs）在处理形状相似但细节细微的食物图像时面临显著挑战。为应对这一挑战，本研究提出了一种名为特征增强TResNet（FE-TResNet）的创新方法，专门针对细粒度食物图像并准确捕捉其中的细微特征。FE-TResNet方法基于TResNet模型，并结合了风格基于校准模块（StyleRM）和深层通道注意力（DCA）技术，以增强特征提取能力。在对中国食物图像数据集ChineseFoodNet和CNFOOD-241的实验验证中，FE-TResNet方法显著提高了分类准确性，分别达到81.37%和80.29%，展示了其在细粒度食物图像分类中的有效性和优越性。 

---
# FIQ: Fundamental Question Generation with the Integration of Question Embeddings for Video Question Answering 

**Title (ZH)**: FIQ: 基于问题嵌入的视频问答基本问题生成 

**Authors**: Ju-Young Oh, Ho-Joong Kim, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.12816)  

**Abstract**: Video question answering (VQA) is a multimodal task that requires the interpretation of a video to answer a given question. Existing VQA methods primarily utilize question and answer (Q&A) pairs to learn the spatio-temporal characteristics of video content. However, these annotations are typically event-centric, which is not enough to capture the broader context of each video. The absence of essential details such as object types, spatial layouts, and descriptive attributes restricts the model to learning only a fragmented scene representation. This issue limits the model's capacity for generalization and higher-level reasoning. In this paper, we propose a fundamental question generation with the integration of question embeddings for video question answering (FIQ), a novel approach designed to strengthen the reasoning ability of the model by enhancing the fundamental understanding of videos. FIQ generates Q&A pairs based on descriptions extracted from videos, enriching the training data with fundamental scene information. Generated Q&A pairs enable the model to understand the primary context, leading to enhanced generalizability and reasoning ability. Furthermore, we incorporate a VQ-CAlign module that assists task-specific question embeddings with visual features, ensuring that essential domain-specific details are preserved to increase the adaptability of downstream tasks. Experiments on SUTD-TrafficQA demonstrate that our FIQ achieves state-of-the-art performance compared to existing baseline methods. 

**Abstract (ZH)**: 视频问答（VQA）是一种多模态任务，要求对视频进行解释以回答给定的问题。现有的VQA方法主要利用问题和答案（Q&A）对来学习视频内容的时空特性。然而，这些注释通常是事件中心的，无法捕捉每个视频的更广泛上下文。缺乏诸如物体类型、空间布局和描述性属性等关键细节限制了模型学习碎片化的场景表示。这限制了模型的一般化能力和高层次推理能力。本文提出了一种整合问题嵌入的认知基础问题生成方法（FIQ），这是一种旨在通过增强对视频的基本理解来加强模型推理能力的新方法。FIQ基于从视频中提取的描述生成Q&A对，丰富了训练数据，提供了基本场景信息。生成的Q&A对使模型能够理解主要上下文，从而增强其一般化能力和推理能力。此外，我们引入了VQ-CAlign模块，该模块辅助特定任务的问题嵌入与视觉特征相结合，确保保留关键的领域特定细节，以提高下游任务的适应性。实验结果表明，我们的FIQ在SUTD-TrafficQA数据集上的性能优于现有基线方法。 

---
# Large Language Models' Internal Perception of Symbolic Music 

**Title (ZH)**: 大型语言模型对符号音乐的内在感知 

**Authors**: Andrew Shin, Kunitake Kaneko  

**Link**: [PDF](https://arxiv.org/pdf/2507.12808)  

**Abstract**: Large language models (LLMs) excel at modeling relationships between strings in natural language and have shown promise in extending to other symbolic domains like coding or mathematics. However, the extent to which they implicitly model symbolic music remains underexplored. This paper investigates how LLMs represent musical concepts by generating symbolic music data from textual prompts describing combinations of genres and styles, and evaluating their utility through recognition and generation tasks. We produce a dataset of LLM-generated MIDI files without relying on explicit musical training. We then train neural networks entirely on this LLM-generated MIDI dataset and perform genre and style classification as well as melody completion, benchmarking their performance against established models. Our results demonstrate that LLMs can infer rudimentary musical structures and temporal relationships from text, highlighting both their potential to implicitly encode musical patterns and their limitations due to a lack of explicit musical context, shedding light on their generative capabilities for symbolic music. 

**Abstract (ZH)**: 大型语言模型（LLMs）在建模自然语言中的字符串关系方面表现出色，并在扩展到诸如编码或数学的其他符号领域方面展现出前景。然而，它们在隐式建模符号音乐方面的程度仍然尚未充分探索。本文通过从描述不同流派和风格组合的文本提示生成符号音乐数据，并通过识别和生成任务评估其实用性，来探究LLMs如何表示音乐概念。我们生成了一个不依赖于显式音乐训练的LLM生成的MIDI文件数据集。然后，我们完全基于此LLM生成的MIDI数据集训练神经网络，并进行了流派和风格分类以及旋律填充，将它们的性能与现有模型进行比较。我们的结果显示，LLMs可以从文本中推断出基本的音乐结构和时间关系，这既突显了它们隐式编码音乐模式的潜力，也揭示了由于缺乏明确的音乐上下文而导致的局限性，从而阐明了它们在符号音乐生成方面的能力。 

---
# PMKLC: Parallel Multi-Knowledge Learning-based Lossless Compression for Large-Scale Genomics Database 

**Title (ZH)**: PMKLC：面向大规模基因组数据库的并行多知识学习无损压缩方法 

**Authors**: Hui Sun, Yanfeng Ding, Liping Yi, Huidong Ma, Gang Wang, Xiaoguang Liu, Cheng Zhong, Wentong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.12805)  

**Abstract**: Learning-based lossless compressors play a crucial role in large-scale genomic database backup, storage, transmission, and management. However, their 1) inadequate compression ratio, 2) low compression \& decompression throughput, and 3) poor compression robustness limit their widespread adoption and application in both industry and academia. To solve those challenges, we propose a novel \underline{P}arallel \underline{M}ulti-\underline{K}nowledge \underline{L}earning-based \underline{C}ompressor (PMKLC) with four crucial designs: 1) We propose an automated multi-knowledge learning-based compression framework as compressors' backbone to enhance compression ratio and robustness; 2) we design a GPU-accelerated ($s$,$k$)-mer encoder to optimize compression throughput and computing resource usage; 3) we introduce data block partitioning and Step-wise Model Passing (SMP) mechanisms for parallel acceleration; 4) We design two compression modes PMKLC-S and PMKLC-M to meet the complex application scenarios, where the former runs on a resource-constrained single GPU and the latter is multi-GPU accelerated. We benchmark PMKLC-S/M and 14 baselines (7 traditional and 7 leaning-based) on 15 real-world datasets with different species and data sizes. Compared to baselines on the testing datasets, PMKLC-S/M achieve the average compression ratio improvement up to 73.609\% and 73.480\%, the average throughput improvement up to 3.036$\times$ and 10.710$\times$, respectively. Besides, PMKLC-S/M also achieve the best robustness and competitive memory cost, indicating its greater stability against datasets with different probability distribution perturbations, and its strong ability to run on memory-constrained devices. 

**Abstract (ZH)**: 基于学习的无损压缩器在大规模基因组数据库备份、存储、传输和管理中发挥着 crucial 作用。然而，它们存在的 1) 压缩比不足，2) 压缩与解压缩吞吐量低，3) 压缩鲁棒性差 的问题限制了其在工业和学术界的广泛应用。为解决这些挑战，我们提出了一种新颖的 Parallel Multi-Knowledge Learning-based Compressor (PMKLC)，其包含了四个关键设计：1) 提出了一种自动化多知识学习基于压缩的框架作为压缩器的核心，以增强压缩比和鲁棒性；2) 设计了 GPU 加速的 ($s$,$k$)-mer 编码器来优化压缩吞吐量和计算资源使用；3) 引入了数据块划分和 Step-wise Model Passing (SMP) 机制以实现并行加速；4) 设计了两种压缩模式 PMKLC-S 和 PMKLC-M 以适应复杂的应用场景，其中前者在资源受限的单个 GPU 上运行，后者则通过多 GPU 加速。我们使用 15 个不同物种和数据大小的实际数据集对 PMKLC-S/M 和 14 个 baselines (7 传统和 7 学习基于) 进行了基准测试。与测试数据集上的 baselines 相比，PMKLC-S/M 在压缩比上平均提高了 73.609% 和 73.480%，在吞吐量上平均分别提高了 3.036 倍和 10.710 倍。此外，PMKLC-S/M 在鲁棒性和内存成本方面也表现最佳，显示出其在面对不同概率分布扰动的数据集时具有更高的稳定性和在内存受限设备上运行的强适应能力。 

---
# FLDmamba: Integrating Fourier and Laplace Transform Decomposition with Mamba for Enhanced Time Series Prediction 

**Title (ZH)**: FLDmamba: 结合傅里叶变换和拉普拉斯变换分解与Mamba以增强时间序列预测 

**Authors**: Qianru Zhang, Chenglei Yu, Haixin Wang, Yudong Yan, Yuansheng Cao, Siu-Ming Yiu, Tailin Wu, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.12803)  

**Abstract**: Time series prediction, a crucial task across various domains, faces significant challenges due to the inherent complexities of time series data, including non-stationarity, multi-scale periodicity, and transient dynamics, particularly when tackling long-term predictions. While Transformer-based architectures have shown promise, their quadratic complexity with sequence length hinders their efficiency for long-term predictions. Recent advancements in State-Space Models, such as Mamba, offer a more efficient alternative for long-term modeling, but they cannot capture multi-scale periodicity and transient dynamics effectively. Meanwhile, they are susceptible to data noise issues in time series. This paper proposes a novel framework, FLDmamba (Fourier and Laplace Transform Decomposition Mamba), addressing these limitations. FLDmamba leverages the strengths of both Fourier and Laplace transforms to effectively capture both multi-scale periodicity, transient dynamics within time series data, and improve the robustness of the model to the data noise issue. Our extensive experiments demonstrate that FLDmamba achieves superior performance on time series prediction benchmarks, outperforming both Transformer-based and other Mamba-based architectures. To promote the reproducibility of our method, we have made both the code and data accessible via the following URL:{\href{this https URL}{this https URL\model}. 

**Abstract (ZH)**: 时间序列预测是一个跨多个领域的关键任务，但由于时间序列数据内在的非平稳性、多尺度周期性和瞬态动力学特征，尤其是针对长期预测时，面临着显著挑战。尽管基于Transformer的架构显示出前景，但它们与序列长度呈二次复杂性的特性限制了其在长期预测中的效率。最近在状态空间模型方面的进展，如Mamba，为长期建模提供了一种更高效的替代方案，但它们在捕捉多尺度周期性和瞬态动力学方面效果欠佳，同时对时间序列中的数据噪声问题也较为敏感。本文提出了一种新颖的框架FLDmamba（傅里叶和拉普拉斯变换分解Mamba），以解决这些局限性。FLDmamba结合了傅里叶和拉普拉斯变换的优势，有效捕捉时间序列数据中的多尺度周期性和瞬态动力学，提高模型对数据噪声问题的鲁棒性。我们的 extensive 实验表明，FLDmamba 在时间序列预测基准上表现出优越的性能，超越了基于Transformer和Mamba的其他架构。为了促进我们的方法的可再现性，我们已将代码和数据通过以下 URL 提供给用户：{this https URL}。 

---
# City-VLM: Towards Multidomain Perception Scene Understanding via Multimodal Incomplete Learning 

**Title (ZH)**: City-VLM：通过多模态不完全学习走向多域感知场景理解 

**Authors**: Penglei Sun, Yaoxian Song, Xiangru Zhu, Xiang Liu, Qiang Wang, Yue Liu, Changqun Xia, Tiefeng Li, Yang Yang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12795)  

**Abstract**: Scene understanding enables intelligent agents to interpret and comprehend their environment. While existing large vision-language models (LVLMs) for scene understanding have primarily focused on indoor household tasks, they face two significant limitations when applied to outdoor large-scale scene understanding. First, outdoor scenarios typically encompass larger-scale environments observed through various sensors from multiple viewpoints (e.g., bird view and terrestrial view), while existing indoor LVLMs mainly analyze single visual modalities within building-scale contexts from humanoid viewpoints. Second, existing LVLMs suffer from missing multidomain perception outdoor data and struggle to effectively integrate 2D and 3D visual information. To address the aforementioned limitations, we build the first multidomain perception outdoor scene understanding dataset, named \textbf{\underline{SVM-City}}, deriving from multi\textbf{\underline{S}}cale scenarios with multi\textbf{\underline{V}}iew and multi\textbf{\underline{M}}odal instruction tuning data. It contains $420$k images and $4, 811$M point clouds with $567$k question-answering pairs from vehicles, low-altitude drones, high-altitude aerial planes, and satellite. To effectively fuse the multimodal data in the absence of one modality, we introduce incomplete multimodal learning to model outdoor scene understanding and design the LVLM named \textbf{\underline{City-VLM}}. Multimodal fusion is realized by constructing a joint probabilistic distribution space rather than implementing directly explicit fusion operations (e.g., concatenation). Experimental results on three typical outdoor scene understanding tasks show City-VLM achieves $18.14 \%$ performance surpassing existing LVLMs in question-answering tasks averagely. Our method demonstrates pragmatic and generalization performance across multiple outdoor scenes. 

**Abstract (ZH)**: 场景理解使智能代理能够解释和理解其环境。虽然现有的大型视觉-语言模型（LVLM）在场景理解上主要集中在室内家庭任务上，但在应用于大规模户外场景理解时面临两大局限性。首先，户外场景通常涉及通过多种传感器从多个视角（如鸟瞰视角和地面视角）观测的大规模环境，而现有的室内LVLM主要从类人视角在建筑尺度的范围内分析单一视觉模态。其次，现有的LVLM缺乏多领域感知的户外数据，并且难以有效地整合2D和3D视觉信息。为解决上述局限性，我们构建了第一个多领域感知的户外场景理解数据集，命名为**\underline{SVM-City}**，该数据集源自多尺度场景，并包含多视角和多模态指令调优数据。它包含了42万张图像和48.11亿个点云，以及来自车辆、低空无人机、高空航拍飞机和卫星的56.7万对问答对。为在缺乏某种模态的情况下有效融合多模态数据，我们引入了不完整的多模态学习来建模户外场景理解，并设计了名为**\underline{City-VLM}**的LVLM。多模态融合通过构建联合概率分布空间来实现，而不是直接执行显式的融合操作（如拼接）。在三个典型户外场景理解任务上的实验结果显示，City-VLM在问答任务上的平均性能超过了现有LVLM 18.14%，我们的方法在多个户外场景中表现出实用性和泛化性能。 

---
# A Semi-Supervised Learning Method for the Identification of Bad Exposures in Large Imaging Surveys 

**Title (ZH)**: 半监督学习方法在大型成像调查中识别不良曝光的应用 

**Authors**: Yufeng Luo, Adam D. Myers, Alex Drlica-Wagner, Dario Dematties, Salma Borchani, Frank Valdes, Arjun Dey, David Schlegel, Rongpu Zhou, DESI Legacy Imaging Surveys Team  

**Link**: [PDF](https://arxiv.org/pdf/2507.12784)  

**Abstract**: As the data volume of astronomical imaging surveys rapidly increases, traditional methods for image anomaly detection, such as visual inspection by human experts, are becoming impractical. We introduce a machine-learning-based approach to detect poor-quality exposures in large imaging surveys, with a focus on the DECam Legacy Survey (DECaLS) in regions of low extinction (i.e., $E(B-V)<0.04$). Our semi-supervised pipeline integrates a vision transformer (ViT), trained via self-supervised learning (SSL), with a k-Nearest Neighbor (kNN) classifier. We train and validate our pipeline using a small set of labeled exposures observed by surveys with the Dark Energy Camera (DECam). A clustering-space analysis of where our pipeline places images labeled in ``good'' and ``bad'' categories suggests that our approach can efficiently and accurately determine the quality of exposures. Applied to new imaging being reduced for DECaLS Data Release 11, our pipeline identifies 780 problematic exposures, which we subsequently verify through visual inspection. Being highly efficient and adaptable, our method offers a scalable solution for quality control in other large imaging surveys. 

**Abstract (ZH)**: 随着天文成像调查的数据量迅速增加，传统的图像异常检测方法，如依靠人力专家视觉检查，变得不再实用。我们介绍了一种基于机器学习的方法，用于在低消光区域（即$E(B-V)<0.04$）的DECam遗产调查（DECaLS）中检测低质量曝光。我们的半监督管道将通过自我监督学习（SSL）训练的视觉变换器（ViT）与k-最近邻（kNN）分类器结合在一起。我们使用Dark Energy Camera（DECam）进行调查观测的小规模标记曝光集训练和验证我们的管道。通过将图像在“良好”和“不良”类别中分类的空间聚类分析表明，我们的方法能够高效准确地确定曝光质量。针对即将发布的DECaLS数据释放11的新成像数据，我们的管道识别了780个有问题的曝光，并通过视觉检查进行了验证。由于其高效性和可适应性，我们的方法为其他大型成像调查的质量控制提供了可扩展的解决方案。 

---
# A Comprehensive Survey of Electronic Health Record Modeling: From Deep Learning Approaches to Large Language Models 

**Title (ZH)**: 电子健康记录建模综述：从深度学习方法到大型语言模型 

**Authors**: Weijieying Ren, Jingxi Zhu, Zehao Liu, Tianxiang Zhao, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2507.12774)  

**Abstract**: Artificial intelligence (AI) has demonstrated significant potential in transforming healthcare through the analysis and modeling of electronic health records (EHRs). However, the inherent heterogeneity, temporal irregularity, and domain-specific nature of EHR data present unique challenges that differ fundamentally from those in vision and natural language tasks. This survey offers a comprehensive overview of recent advancements at the intersection of deep learning, large language models (LLMs), and EHR modeling. We introduce a unified taxonomy that spans five key design dimensions: data-centric approaches, neural architecture design, learning-focused strategies, multimodal learning, and LLM-based modeling systems. Within each dimension, we review representative methods addressing data quality enhancement, structural and temporal representation, self-supervised learning, and integration with clinical knowledge. We further highlight emerging trends such as foundation models, LLM-driven clinical agents, and EHR-to-text translation for downstream reasoning. Finally, we discuss open challenges in benchmarking, explainability, clinical alignment, and generalization across diverse clinical settings. This survey aims to provide a structured roadmap for advancing AI-driven EHR modeling and clinical decision support. For a comprehensive list of EHR-related methods, kindly refer to this https URL. 

**Abstract (ZH)**: 人工 intelligence（AI）在通过电子健康记录（EHR）的分析与建模改造医疗健康方面展现了显著的潜力。然而，EHR数据固有的异质性、时间不规律性和领域特定性提出了与视觉和自然语言任务根本不同的独特挑战。本文综述了深度学习、大规模语言模型（LLM）与EHR建模交叉领域的最新进展。我们引入了一个统一的分类框架，涵盖了五大关键设计维度：以数据为中心的方法、神经网络架构设计、学习导向策略、多模态学习以及基于LLM的建模系统。在每个维度中，我们回顾了代表性的方法，涉及数据质量提升、结构和时间表示、自我监督学习以及与临床知识的集成。我们还强调了新兴趋势，如基础模型、由LLM驱动的临床代理以及EHR到文本的翻译以供下游推理。最后，我们讨论了基准测试、可解释性、临床对齐以及在多种临床环境中的泛化方面的开放性挑战。本文旨在为推进AI驱动的EHR建模和临床决策支持提供一个结构化的路线图。有关EHR相关方法的详细列表，请参见此链接：https URL。 

---
# Local Representative Token Guided Merging for Text-to-Image Generation 

**Title (ZH)**: 文本引导的局部代表性词元导向合并生成文本到图像 

**Authors**: Min-Jeong Lee, Hee-Dong Kim, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.12771)  

**Abstract**: Stable diffusion is an outstanding image generation model for text-to-image, but its time-consuming generation process remains a challenge due to the quadratic complexity of attention operations. Recent token merging methods improve efficiency by reducing the number of tokens during attention operations, but often overlook the characteristics of attention-based image generation models, limiting their effectiveness. In this paper, we propose local representative token guided merging (ReToM), a novel token merging strategy applicable to any attention mechanism in image generation. To merge tokens based on various contextual information, ReToM defines local boundaries as windows within attention inputs and adjusts window sizes. Furthermore, we introduce a representative token, which represents the most representative token per window by computing similarity at a specific timestep and selecting the token with the highest average similarity. This approach preserves the most salient local features while minimizing computational overhead. Experimental results show that ReToM achieves a 6.2% improvement in FID and higher CLIP scores compared to the baseline, while maintaining comparable inference time. We empirically demonstrate that ReToM is effective in balancing visual quality and computational efficiency. 

**Abstract (ZH)**: 基于局部代表性令牌引导合并的高效图像生成模型 

---
# Synergy: End-to-end Concept Model 

**Title (ZH)**: 协同效应：端到端概念模型 

**Authors**: Keli Zheng, Zerong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.12769)  

**Abstract**: In this paper, we present Synergy, a language model that bridges different levels of abstraction in an end-to-end fashion through a learned routing mechanism. Focusing on low-level linguistic abstraction, we trained our model as a byte-level language model. Our model spontaneously learns to tokenize bytes, producing fewer concept tokens than Byte-level Byte Pair Encoder (BBPE) tokenizers while keeping comparable performance. By comparing with Llama3, we observed an advantage of Synergy under the same model scale and training dataset size. Further studies show that the middle part (the higher abstraction part) of our model performs better when positional encodings are removed, suggesting the emergence of position-independent concepts. These findings demonstrate the feasibility of tokenizer-free architectures, paving the way for more robust and flexible pipelines. 

**Abstract (ZH)**: 本文介绍了一种名为Synergy的语言模型，该模型通过学习路由机制以端到端的方式连接不同层次的抽象。聚焦于低层次语言抽象，我们将模型训练成字节级语言模型。模型自主学习字节分词，产生的概念令牌少于Byte-Level Byte Pair Encoder (BBPE) 分词器，同时保持相当的性能。通过与Llama3的对比，我们发现在相同模型规模和训练数据集大小的情况下，Synergy展现出优势。进一步研究显示，在移除位置编码时，模型的中间部分（更高抽象的部分）表现更好，这表明出现了一种与位置无关的概念。这些发现展示了无分词器架构的可行性，为更稳健和灵活的管道铺平了道路。 

---
# Autonomy for Older Adult-Agent Interaction 

**Title (ZH)**: 老年人与代理互动的自主性 

**Authors**: Jiaxin An  

**Link**: [PDF](https://arxiv.org/pdf/2507.12767)  

**Abstract**: As the global population ages, artificial intelligence (AI)-powered agents have emerged as potential tools to support older adults' caregiving. Prior research has explored agent autonomy by identifying key interaction stages in task processes and defining the agent's role at each stage. However, ensuring that agents align with older adults' autonomy preferences remains a critical challenge. Drawing on interdisciplinary conceptualizations of autonomy, this paper examines four key dimensions of autonomy for older adults: decision-making autonomy, goal-oriented autonomy, control autonomy, and social responsibility autonomy. This paper then proposes the following research directions: (1) Addressing social responsibility autonomy, which concerns the ethical and social implications of agent use in communal settings; (2) Operationalizing agent autonomy from the task perspective; and (3) Developing autonomy measures. 

**Abstract (ZH)**: 随着全球人口老龄化，人工智能（AI）驱动的代理已成为支持老年人护理的潜在工具。现有研究通过识别任务流程中的关键交互阶段并定义每个阶段的代理角色来探讨代理的自主性。然而，确保代理与老年人的自主性偏好相一致仍是一项关键挑战。基于跨学科对自主性的概念化，本文探讨了老年人的四个关键自主性维度：决策自主性、目标导向自主性、控制自主性和社会责任自主性。本文随后提出以下研究方向：（1）社会责任自主性，关注代理在集体环境中的使用所涉及的伦理和社会含义；（2）从任务视角实现代理自主性；（3）开发自主性测量方法。 

---
# Think-Before-Draw: Decomposing Emotion Semantics & Fine-Grained Controllable Expressive Talking Head Generation 

**Title (ZH)**: 思考在先：分解情绪语义与细粒度可控表情头部生成 

**Authors**: Hanlei Shi, Leyuan Qu, Yu Liu, Di Gao, Yuhua Zheng, Taihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.12761)  

**Abstract**: Emotional talking-head generation has emerged as a pivotal research area at the intersection of computer vision and multimodal artificial intelligence, with its core value lying in enhancing human-computer interaction through immersive and empathetic this http URL the advancement of multimodal large language models, the driving signals for emotional talking-head generation has shifted from audio and video to more flexible text. However, current text-driven methods rely on predefined discrete emotion label texts, oversimplifying the dynamic complexity of real facial muscle movements and thus failing to achieve natural emotional this http URL study proposes the Think-Before-Draw framework to address two key challenges: (1) In-depth semantic parsing of emotions--by innovatively introducing Chain-of-Thought (CoT), abstract emotion labels are transformed into physiologically grounded facial muscle movement descriptions, enabling the mapping from high-level semantics to actionable motion features; and (2) Fine-grained expressiveness optimization--inspired by artists' portrait painting process, a progressive guidance denoising strategy is proposed, employing a "global emotion localization--local muscle control" mechanism to refine micro-expression dynamics in generated this http URL experiments demonstrate that our approach achieves state-of-the-art performance on widely-used benchmarks, including MEAD and HDTF. Additionally, we collected a set of portrait images to evaluate our model's zero-shot generation capability. 

**Abstract (ZH)**: 基于情感的虚拟头像生成已成为计算机视觉和多模态人工智能交叉领域的关键研究方向，其核心价值在于通过沉浸式和同理心的人机交互增强用户体验。随着多模态大语言模型的发展，驱动情感虚拟头像生成的信号已从音频和视频转向更具灵活性的文字。然而，当前基于文本的方法依赖于预定义的离散情绪标签文字，简化了真实面部肌肉运动的动态复杂性，从而未能实现自然的情感表达。本研究提出Think-Before-Draw框架以应对两大核心挑战：（1）深入的情感语义解析—通过创新引入Chain-of-Thought（CoT），将抽象的情绪标签转化为生理基础的面部肌肉运动描述，实现从高层语义到可执行动作特征的映射；（2）精细化的情感表达优化—受艺术家肖像绘画过程的启发，提出了一种逐步指导降噪策略，采用“全局情绪定位—局部肌肉控制”机制，以细化生成虚拟头像中的微表情动态。实验表明，我们的方法在广泛使用的MEAD和HDTF基准上达到了最先进的性能。此外，我们还收集了一组肖像图片来评估模型的零样本生成能力。 

---
# Unified Medical Image Segmentation with State Space Modeling Snake 

**Title (ZH)**: 统一医疗影像分割方法：状态空间建模蛇皮算法 

**Authors**: Ruicheng Zhang, Haowei Guo, Kanghui Tian, Jun Zhou, Mingliang Yan, Zeyu Zhang, Shen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.12760)  

**Abstract**: Unified Medical Image Segmentation (UMIS) is critical for comprehensive anatomical assessment but faces challenges due to multi-scale structural heterogeneity. Conventional pixel-based approaches, lacking object-level anatomical insight and inter-organ relational modeling, struggle with morphological complexity and feature conflicts, limiting their efficacy in UMIS. We propose Mamba Snake, a novel deep snake framework enhanced by state space modeling for UMIS. Mamba Snake frames multi-contour evolution as a hierarchical state space atlas, effectively modeling macroscopic inter-organ topological relationships and microscopic contour refinements. We introduce a snake-specific vision state space module, the Mamba Evolution Block (MEB), which leverages effective spatiotemporal information aggregation for adaptive refinement of complex morphologies. Energy map shape priors further ensure robust long-range contour evolution in heterogeneous data. Additionally, a dual-classification synergy mechanism is incorporated to concurrently optimize detection and segmentation, mitigating under-segmentation of microstructures in UMIS. Extensive evaluations across five clinical datasets reveal Mamba Snake's superior performance, with an average Dice improvement of 3\% over state-of-the-art methods. 

**Abstract (ZH)**: 统一医学图像分割（UMIS）对于全面的解剖评估至关重要，但面对多尺度结构异质性挑战。传统的基于像素的方法由于缺乏对象级别的解剖学洞察和器官间关系建模，难以应对形态复杂性和特征冲突，限制了其在UMIS中的效果。我们提出了一种名为Mamba Snake的新型深度蛇形框架，该框架通过状态空间建模增强，用于UMIS。Mamba Snake将多轮廓演化视为分层状态空间大洲图，有效建模宏观器官间拓扑关系和微观轮廓细化。我们引入了一种专用于蛇形的状态空间模块——Mamba 进化块（MEB），该模块利用有效的时空信息聚合实现复杂形态的自适应细化。能量图形状先验进一步确保在异质数据中稳健的长距离轮廓演化。此外，还集成了一种双分类协同机制，以同时优化检测和分割，减轻UMIS中微结构的分割不足。在五个临床数据集上的广泛评估显示，Mamba Snake在Dice分数上平均提高了3%，优于现有最佳方法。 

---
# Logit Arithmetic Elicits Long Reasoning Capabilities Without Training 

**Title (ZH)**: Logit Arithmetic 启发长推理能力无需训练 

**Authors**: Yunxiang Zhang, Muhammad Khalifa, Lechen Zhang, Xin Liu, Ayoung Lee, Xinliang Frederick Zhang, Farima Fatahi Bayat, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12759)  

**Abstract**: Large reasoning models (LRMs) can do complex reasoning via long chain-of-thought (CoT) involving cognitive strategies such as backtracking and self-correction. Recent studies suggest that some models inherently possess these long reasoning abilities, which may be unlocked via extra training. Our work first investigates whether we can elicit such behavior without any training. To this end, we propose a decoding-time approach, ThinkLogit, which utilizes logits arithmetic (Liu et al., 2024) to tune a target large LM for long reasoning using a substantially smaller model as guider. We then show that we can further boost performance by training the guider model with preference optimization over correct/incorrect reasoning pairs sampled from both the target and guider model -- a setup we refer to as ThinkLogit-DPO. Our experiments demonstrate that ThinkLogit and ThinkLogit-DPO achieve a relative improvement in pass@1 by 26% and 29%, respectively, over four mathematical datasets using the Qwen2.5-32B when guided by R1-Distill-Qwen-1.5B -- a model 21x smaller. Lastly, we show that ThinkLogit can transfer long reasoning skills acquired through reinforcement learning, improving pass@1 by 13% relative compared to the Qwen2.5-32B base model. Our work presents a computationally-efficient method to elicit long reasoning in large models with minimal or no additional training. 

**Abstract (ZH)**: 大型推理模型（LRMs）可以通过长链推理（CoT）进行复杂的推理，涉及诸如回溯和自我修正等认知策略。近期研究表明，某些模型本身具备这些长期推理能力，可能通过额外训练被激活。我们首先研究是否可以在没有任何训练的情况下激发此类行为。为此，我们提出了一种解码时间方法ThinkLogit，该方法利用logits算术（Liu et al., 2024）来使用一个显著更小的模型作为引导来调优目标大模型以支持长推理。我们还展示了通过使用正误推理对对来自目标模型和引导模型的样本进行偏好优化训练引导模型的方法可以进一步提升性能，这被称为ThinkLogit-DPO。我们的实验表明，在使用R1-Distill-Qwen-1.5B（一个比Qwen2.5-32B小21倍的模型）引导时，ThinkLogit和ThinkLogit-DPO分别在四个数学数据集上相对提高了pass@1指标26%和29%。最后，我们展示了ThinkLogit可以转移通过强化学习获得的长推理能力，相对提高pass@1指标13%，相较于基模Qwen2.5-32B。我们的工作提出了一种计算效率高的方法，在极少或无需额外训练的情况下激发大型模型进行长推理。 

---
# Transformer-based Spatial Grounding: A Comprehensive Survey 

**Title (ZH)**: 基于变换器的空间定位综述 

**Authors**: Ijazul Haq, Muhammad Saqib, Yingjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12739)  

**Abstract**: Spatial grounding, the process of associating natural language expressions with corresponding image regions, has rapidly advanced due to the introduction of transformer-based models, significantly enhancing multimodal representation and cross-modal alignment. Despite this progress, the field lacks a comprehensive synthesis of current methodologies, dataset usage, evaluation metrics, and industrial applicability. This paper presents a systematic literature review of transformer-based spatial grounding approaches from 2018 to 2025. Our analysis identifies dominant model architectures, prevalent datasets, and widely adopted evaluation metrics, alongside highlighting key methodological trends and best practices. This study provides essential insights and structured guidance for researchers and practitioners, facilitating the development of robust, reliable, and industry-ready transformer-based spatial grounding models. 

**Abstract (ZH)**: 基于变压器的空间定位：从2018到2025年的系统文献综述 

---
# Task-Specific Audio Coding for Machines: Machine-Learned Latent Features Are Codes for That Machine 

**Title (ZH)**: 面向任务的音频编码：机器学习潜在特征作为该机器的编码 

**Authors**: Anastasia Kuznetsova, Inseon Jang, Wootaek Lim, Minje Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.12701)  

**Abstract**: Neural audio codecs, leveraging quantization algorithms, have significantly impacted various speech/audio tasks. While high-fidelity reconstruction is paramount for human perception, audio coding for machines (ACoM) prioritizes efficient compression and downstream task performance, disregarding perceptual nuances. This work introduces an efficient ACoM method that can compress and quantize any chosen intermediate feature representation of an already trained speech/audio downstream model. Our approach employs task-specific loss guidance alongside residual vector quantization (RVQ) losses, providing ultra-low bitrates (i.e., less than 200 bps) with a minimal loss of the downstream model performance. The resulting tokenizer is adaptable to various bitrates and model sizes for flexible deployment. Evaluated on automatic speech recognition and audio classification, our method demonstrates its efficacy and potential for broader task and architectural applicability through appropriate regularization. 

**Abstract (ZH)**: 基于量化算法的神经音频编解码器对各种语音/音频任务产生了重大影响。虽然高保真重建对人类感知至关重要，但面向机器的音频编码（ACoM）侧重于高效压缩和下游任务性能，忽略感知上的细微差别。本工作介绍了一种高效的ACoM方法，可以压缩并量化任何预先训练的语音/音频下游模型的任意中间特征表示。该方法结合任务特定的损失指导和残差向量量化（RVQ）损失，能够在最大限度保持下游模型性能的情况下提供超低比特率（即低于200 bps）。生成的分词器具有多种比特率和模型大小的适应性，便于灵活部署。在自动语音识别和音频分类任务上，我们的方法通过适当的正则化展示了其有效性和在更广泛任务和架构上的应用潜力。 

---
# Data Transformation Strategies to Remove Heterogeneity 

**Title (ZH)**: 数据转换策略以消除异质性 

**Authors**: Sangbong Yoo, Jaeyoung Lee, Chanyoung Yoon, Geonyeong Son, Hyein Hong, Seongbum Seo, Soobin Yim, Chanyoung Jung, Jungsoo Park, Misuk Kim, Yun Jang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12677)  

**Abstract**: Data heterogeneity is a prevalent issue, stemming from various conflicting factors, making its utilization complex. This uncertainty, particularly resulting from disparities in data formats, frequently necessitates the involvement of experts to find resolutions. Current methodologies primarily address conflicts related to data structures and schemas, often overlooking the pivotal role played by data transformation. As the utilization of artificial intelligence (AI) continues to expand, there is a growing demand for a more streamlined data preparation process, and data transformation becomes paramount. It customizes training data to enhance AI learning efficiency and adapts input formats to suit diverse AI models. Selecting an appropriate transformation technique is paramount in preserving crucial data details. Despite the widespread integration of AI across various industries, comprehensive reviews concerning contemporary data transformation approaches are scarce. This survey explores the intricacies of data heterogeneity and its underlying sources. It systematically categorizes and presents strategies to address heterogeneity stemming from differences in data formats, shedding light on the inherent challenges associated with each strategy. 

**Abstract (ZH)**: 数据异质性是一个普遍存在的问题，源自多种冲突因素，使其利用变得复杂。这种不确定性，尤其是由于数据格式差异引起的情况，通常需要专家介入以找到解决方案。现有的方法主要关注数据结构和模式相关的冲突，而往往忽视了数据转换所扮演的关键角色。随着人工智能（AI）的应用不断扩大，对更 streamlined 的数据准备过程的需求日益增加，数据转换变得尤为重要。它能够定制训练数据以提高AI学习效率，并适应不同的AI模型输入格式。选择合适的转换技术对于保留关键数据细节至关重要。尽管AI在各行各业中的应用越来越广泛，但关于当前数据转换方法的综合评估却很少见。本文探讨了数据异质性的复杂性及其背后的根源，系统地分类并呈现了应对由数据格式差异引起异质性的策略，揭示了每种策略固有的挑战。 

---
# FORTRESS: Function-composition Optimized Real-Time Resilient Structural Segmentation via Kolmogorov-Arnold Enhanced Spatial Attention Networks 

**Title (ZH)**: FORTRESS: 基于柯莫哥洛夫-阿诺德增强空间注意力网络的函数组合优化实时鲁棒结构分割 

**Authors**: Christina Thrainer, Md Meftahul Ferdaus, Mahdi Abdelguerfi, Christian Guetl, Steven Sloan, Kendall N. Niles, Ken Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2507.12675)  

**Abstract**: Automated structural defect segmentation in civil infrastructure faces a critical challenge: achieving high accuracy while maintaining computational efficiency for real-time deployment. This paper presents FORTRESS (Function-composition Optimized Real-Time Resilient Structural Segmentation), a new architecture that balances accuracy and speed by using a special method that combines depthwise separable convolutions with adaptive Kolmogorov-Arnold Network integration. FORTRESS incorporates three key innovations: a systematic depthwise separable convolution framework achieving a 3.6x parameter reduction per layer, adaptive TiKAN integration that selectively applies function composition transformations only when computationally beneficial, and multi-scale attention fusion combining spatial, channel, and KAN-enhanced features across decoder levels. The architecture achieves remarkable efficiency gains with 91% parameter reduction (31M to 2.9M), 91% computational complexity reduction (13.7 to 1.17 GFLOPs), and 3x inference speed improvement while delivering superior segmentation performance. Evaluation on benchmark infrastructure datasets demonstrates state-of-the-art results with an F1- score of 0.771 and a mean IoU of 0.677, significantly outperforming existing methods including U-Net, SA-UNet, and U- KAN. The dual optimization strategy proves essential for optimal performance, establishing FORTRESS as a robust solution for practical structural defect segmentation in resource-constrained environments where both accuracy and computational efficiency are paramount. Comprehensive architectural specifications are provided in the Supplemental Material. Source code is available at URL: this https URL. 

**Abstract (ZH)**: 自动结构缺陷分割在土木基础设施中的实现面临一项关键挑战：在保持实时部署的同时实现高精度和计算效率。本文提出了一种新的架构FORTRESS（函数合成优化实时弹性结构分割），该架构通过结合深度可分离卷积与自适应柯尔莫戈洛夫-阿诺德网络集成的特殊方法来平衡精度与速度。FORTRESS包含了三个关键创新：一种系统性的深度可分离卷积框架，每层参数减少了3.6倍，自适应TiKAN集成仅在计算上有益时选择性地应用函数合成变换，以及多尺度注意力融合，结合解码器级别的空间、通道和KAN增强特征。该架构实现了显著的效率提升，参数减少了91%（从31M降至2.9M），计算复杂度减少了91%（从13.7降至1.17 GFLOPs），推理速度提高了3倍，同时提供了卓越的分割性能。基准土木基础设施数据集上的评估显示，FORTRESS在F1分数为0.771和平均IoU为0.677的情况下达到领先效果，显著优于包括U-Net、SA-UNet和U-KAN在内的现有方法。双优化策略对于最佳性能至关重要，确立了FORTRESS在资源受限环境中同时追求准确性和计算效率的稳健解决方案。在补充材料中提供了全面的架构规范。源代码可在URL: this https URL获取。 

---
# ParaStudent: Generating and Evaluating Realistic Student Code by Teaching LLMs to Struggle 

**Title (ZH)**: ParaStudent: 通过教大规模语言模型挣扎来生成和评估现实中的学生代码 

**Authors**: Mihran Miroyan, Rose Niousha, Joseph E. Gonzalez, Gireeja Ranade, Narges Norouzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.12674)  

**Abstract**: Large Language Models (LLMs) have shown strong performance on programming tasks, but can they generate student-like code like real students - imperfect, iterative, and stylistically diverse? We present ParaStudent, a systematic study of LLM-based "student-like" code generation in an introductory programming course setting. Using a dataset of timestamped student submissions across multiple semesters, we design low- and high-resolution experiments to model student progress and evaluate code outputs along semantic, functional, and stylistic dimensions. Our results show that fine-tuning significantly improves alignment with real student trajectories and captures error patterns, incremental improvements, and stylistic variations more faithfully. This study shows that modeling realistic student code requires capturing learning dynamics through context-aware generation, temporal modeling, and multi-dimensional evaluation. Code for experiments and evaluation is available at \href{this https URL}{\texttt{this http URL}}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在编程任务上显示出了强大的性能，但它们能否生成像真实学生那样的代码——即不完美、迭代且风格多样的代码？我们呈现了ParaStudent，这是一种在入门级编程课程环境中对基于LLM的“学生样”代码生成进行系统研究的方法。利用跨多个学期的时间戳标记学生提交数据集，我们设计了低分辨率和高分辨率的实验，以建模学生的学习进展，并从语义、功能和风格三个维度评估代码输出。研究结果表明，微调显著提高了与真实学生轨迹的一致性，并更准确地捕捉到了错误模式、逐步改进和风格变化。本研究显示，要模拟真实的 student code，需要通过上下文感知生成、时间建模和多维度评估来捕捉学习动态。实验和评估代码可在 \href{this https URL}{\texttt{this http URL}} 获取。 

---
# InSight: AI Mobile Screening Tool for Multiple Eye Disease Detection using Multimodal Fusion 

**Title (ZH)**: InSight: 多模态融合的AI移动筛查工具用于多种眼病检测 

**Authors**: Ananya Raghu, Anisha Raghu, Alice S. Tang, Yannis M. Paulus, Tyson N. Kim, Tomiko T. Oskotsky  

**Link**: [PDF](https://arxiv.org/pdf/2507.12669)  

**Abstract**: Background/Objectives: Age-related macular degeneration, glaucoma, diabetic retinopathy (DR), diabetic macular edema, and pathological myopia affect hundreds of millions of people worldwide. Early screening for these diseases is essential, yet access to medical care remains limited in low- and middle-income countries as well as in resource-limited settings. We develop InSight, an AI-based app that combines patient metadata with fundus images for accurate diagnosis of five common eye diseases to improve accessibility of screenings.
Methods: InSight features a three-stage pipeline: real-time image quality assessment, disease diagnosis model, and a DR grading model to assess severity. Our disease diagnosis model incorporates three key innovations: (a) Multimodal fusion technique (MetaFusion) combining clinical metadata and images; (b) Pretraining method leveraging supervised and self-supervised loss functions; and (c) Multitask model to simultaneously predict 5 diseases. We make use of BRSET (lab-captured images) and mBRSET (smartphone-captured images) datasets, both of which also contain clinical metadata for model training/evaluation.
Results: Trained on a dataset of BRSET and mBRSET images, the image quality checker achieves near-100% accuracy in filtering out low-quality fundus images. The multimodal pretrained disease diagnosis model outperforms models using only images by 6% in balanced accuracy for BRSET and 4% for mBRSET.
Conclusions: The InSight pipeline demonstrates robustness across varied image conditions and has high diagnostic accuracy across all five diseases, generalizing to both smartphone and lab captured images. The multitask model contributes to the lightweight nature of the pipeline, making it five times computationally efficient compared to having five individual models corresponding to each disease. 

**Abstract (ZH)**: 背景/目标: 年龄相关黄斑变性、青光眼、糖尿病视网膜病变（DR）、糖尿病黄斑水肿和病理性近视影响着全世界数百亿人。对这些疾病进行早期筛查至关重要，但在低收入和中收入国家以及资源有限的地区，获得医疗服务仍然有限。我们开发了InSight，一个基于AI的应用程序，将患者元数据与眼底图像结合，以准确诊断五种常见眼部疾病，提高筛查的可及性。
方法: InSight 支持三阶段流水线：实时图像质量评估、疾病诊断模型和DR严重程度评估模型。我们的疾病诊断模型包含三项关键创新：(a) 结合临床元数据和图像的多模态融合技术（MetaFusion）；(b) 利用监督和自我监督损失函数的预训练方法；(c) 多任务模型，同时预测5种疾病。我们使用了BRSET（实验室拍摄的图像）和mBRSET（智能手机拍摄的图像）数据集，这两个数据集都包含临床元数据用于模型训练和评估。
结果: 在BRSET和mBRSET图像数据集上训练的图像质量检查器在筛选低质量眼底图像方面几乎达到了100%的准确率。多模态预训练疾病诊断模型在BRSET和mBRSET数据集上的平衡准确率分别比仅使用图像的模型高出6%和4%。
结论: InSight流水线在各种图像条件下表现出色，并在所有五种疾病上具有高诊断准确性，适用于实验室和智能手机拍摄的图像。多任务模型使流水线更加轻量级，计算效率比为每个疾病拥有五个独立模型高出五倍。 

---
# Single Conversation Methodology: A Human-Centered Protocol for AI-Assisted Software Development 

**Title (ZH)**: 单对话方法学：面向人工智能辅助软件开发的人本协议 

**Authors**: Salvador D. Escobedo  

**Link**: [PDF](https://arxiv.org/pdf/2507.12665)  

**Abstract**: We propose the Single Conversation Methodology (SCM), a novel and pragmatic approach to software development using large language models (LLMs). In contrast to ad hoc interactions with generative AI, SCM emphasizes a structured and persistent development dialogue, where all stages of a project - from requirements to architecture and implementation - unfold within a single, long-context conversation. The methodology is grounded on principles of cognitive clarity, traceability, modularity, and documentation. We define its phases, best practices, and philosophical stance, while arguing that SCM offers a necessary correction to the passive reliance on LLMs prevalent in current practices. We aim to reassert the active role of the developer as architect and supervisor of the intelligent tool. 

**Abstract (ZH)**: 单对话方法论（SCM）：一种基于大型语言模型的新型实用软件开发方法 

---
# Improving physics-informed neural network extrapolation via transfer learning and adaptive activation functions 

**Title (ZH)**: 通过迁移学习和自适应激活函数提高物理知情神经网络的外推能力 

**Authors**: Athanasios Papastathopoulos-Katsaros, Alexandra Stavrianidi, Zhandong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.12659)  

**Abstract**: Physics-Informed Neural Networks (PINNs) are deep learning models that incorporate the governing physical laws of a system into the learning process, making them well-suited for solving complex scientific and engineering problems. Recently, PINNs have gained widespread attention as a powerful framework for combining physical principles with data-driven modeling to improve prediction accuracy. Despite their successes, however, PINNs often exhibit poor extrapolation performance outside the training domain and are highly sensitive to the choice of activation functions (AFs). In this paper, we introduce a transfer learning (TL) method to improve the extrapolation capability of PINNs. Our approach applies transfer learning (TL) within an extended training domain, using only a small number of carefully selected collocation points. Additionally, we propose an adaptive AF that takes the form of a linear combination of standard AFs, which improves both the robustness and accuracy of the model. Through a series of experiments, we demonstrate that our method achieves an average of 40% reduction in relative L2 error and an average of 50% reduction in mean absolute error in the extrapolation domain, all without a significant increase in computational cost. The code is available at this https URL . 

**Abstract (ZH)**: 基于物理的神经网络（PINNs）的迁移学习方法提高其外推能力 

---
# VLMgineer: Vision Language Models as Robotic Toolsmiths 

**Title (ZH)**: VLMgineer: 视觉语言模型作为机器人工具师 

**Authors**: George Jiayuan Gao, Tianyu Li, Junyao Shi, Yihan Li, Zizhe Zhang, Nadia Figueroa, Dinesh Jayaraman  

**Link**: [PDF](https://arxiv.org/pdf/2507.12644)  

**Abstract**: Tool design and use reflect the ability to understand and manipulate the physical world through creativity, planning, and foresight. As such, these capabilities are often regarded as measurable indicators of intelligence across biological species. While much of today's research on robotic intelligence focuses on generating better controllers, inventing smarter tools offers a complementary form of physical intelligence: shifting the onus of problem-solving onto the tool's design. Given the vast and impressive common-sense, reasoning, and creative capabilities of today's foundation models, we investigate whether these models can provide useful priors to automatically design and effectively wield such tools? We present VLMgineer, a framework that harnesses the code generation abilities of vision language models (VLMs) together with evolutionary search to iteratively co-design physical tools and the action plans that operate them to perform a task. We evaluate VLMgineer on a diverse new benchmark of everyday manipulation scenarios that demand creative tool design and use. Across this suite, VLMgineer consistently discovers tools and policies that solve tasks more effectively and innovatively, transforming challenging robotics problems into straightforward executions. It also outperforms VLM-generated designs from human specifications and existing human-crafted tools for everyday tasks. To facilitate future research on automated tool invention, we will release our benchmark and code. 

**Abstract (ZH)**: 基于视觉语言模型的工具设计与使用框架：探索基础模型在自动化工具发明中的潜力 

---
# QSpark: Towards Reliable Qiskit Code Generation 

**Title (ZH)**: QSpark: 朝着可靠Qiskit 代码生成努力 

**Authors**: Kiana Kheiri, Aamna Aamir, Andriy Miranskyy, Chen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.12642)  

**Abstract**: Quantum circuits must be error-resilient, yet LLMs like Granite-20B-Code and StarCoder often output flawed Qiskit code. We fine-tuned a 32 B model with two RL methods, Group Relative Policy Optimization (GRPO) and Odds-Ratio Preference Optimization (ORPO), using a richly annotated synthetic dataset. On the Qiskit HumanEval benchmark, ORPO reaches 56.29\% Pass@1 ($\approx+10$ pp over Granite-8B-QK) and GRPO hits 49\%, both beating all general-purpose baselines; on the original HumanEval they score 65.90\% and 63.00\%. GRPO excels on basic tasks (42/54), ORPO on intermediate ones (41/68), and neither solves the five advanced tasks, highlighting clear gains yet room for progress in AI-assisted quantum programming. 

**Abstract (ZH)**: 量子电路必须具备抗错误能力，但像Granite-20B-Code和StarCoder这样的LLM经常输出有缺陷的Qiskit代码。我们使用丰富的标注合成数据集，用两种RL方法（Group Relative Policy Optimization，GRPO；Odds-Ratio Preference Optimization，ORPO）微调了一个32B模型。在Qiskit HumanEval基准测试中，ORPO达到56.29% Pass@1（约比Granite-8B-QK高10个百分点），GRPO达到49%；在原始的HumanEval上，它们分别达到65.90%和63.00%。GRPO在基础任务上表现优异，ORPO在中级任务上表现优异，但两者都无法解决五个高级任务，这表明在AI辅助量子编程领域仍有改进空间。 

---
# Achieving Robust Channel Estimation Neural Networks by Designed Training Data 

**Title (ZH)**: 通过设计训练数据实现稳健的信道估计神经网络 

**Authors**: Dianxin Luan, John Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2507.12630)  

**Abstract**: Channel estimation is crucial in cognitive communications, as it enables intelligent spectrum sensing and adaptive transmission by providing accurate information about the current channel state. However, in many papers neural networks are frequently tested by training and testing on one example channel or similar channels. This is because data-driven methods often degrade on new data which they are not trained on, as they cannot extrapolate their training knowledge. This is despite the fact physical channels are often assumed to be time-variant. However, due to the low latency requirements and limited computing resources, neural networks may not have enough time and computing resources to execute online training to fine-tune the parameters. This motivates us to design offline-trained neural networks that can perform robustly over wireless channels, but without any actual channel information being known at design time. In this paper, we propose design criteria to generate synthetic training datasets for neural networks, which guarantee that after training the resulting networks achieve a certain mean squared error (MSE) on new and previously unseen channels. Therefore, neural network solutions require no prior channel information or parameters update for real-world implementations. Based on the proposed design criteria, we further propose a benchmark design which ensures intelligent operation for different channel profiles. To demonstrate general applicability, we use neural networks with different levels of complexity to show that the generalization achieved appears to be independent of neural network architecture. From simulations, neural networks achieve robust generalization to wireless channels with both fixed channel profiles and variable delay spreads. 

**Abstract (ZH)**: 基于合成数据集的无线信道稳健神经网络设计准则及验证 

---
# BootSeer: Analyzing and Mitigating Initialization Bottlenecks in Large-Scale LLM Training 

**Title (ZH)**: BootSeer: 分析和缓解大规模LLM训练中的初始化瓶颈 

**Authors**: Rui Li, Xiaoyun Zhi, Jinxin Chi, Menghan Yu, Lixin Huang, Jia Zhu, Weilun Zhang, Xing Ma, Wenjia Liu, Zhicheng Zhu, Daowen Luo, Zuquan Song, Xin Yin, Chao Xiang, Shuguang Wang, Wencong Xiao, Gene Cooperman  

**Link**: [PDF](https://arxiv.org/pdf/2507.12619)  

**Abstract**: Large Language Models (LLMs) have become a cornerstone of modern AI, driving breakthroughs in natural language processing and expanding into multimodal jobs involving images, audio, and video. As with most computational software, it is important to distinguish between ordinary runtime performance and startup overhead. Prior research has focused on runtime performance: improving training efficiency and stability. This work focuses instead on the increasingly critical issue of startup overhead in training: the delay before training jobs begin execution. Startup overhead is particularly important in large, industrial-scale LLMs, where failures occur more frequently and multiple teams operate in iterative update-debug cycles. In one of our training clusters, more than 3.5% of GPU time is wasted due to startup overhead alone.
In this work, we present the first in-depth characterization of LLM training startup overhead based on real production data. We analyze the components of startup cost, quantify its direct impact, and examine how it scales with job size. These insights motivate the design of Bootseer, a system-level optimization framework that addresses three primary startup bottlenecks: (a) container image loading, (b) runtime dependency installation, and (c) model checkpoint resumption. To mitigate these bottlenecks, Bootseer introduces three techniques: (a) hot block record-and-prefetch, (b) dependency snapshotting, and (c) striped HDFS-FUSE. Bootseer has been deployed in a production environment and evaluated on real LLM training workloads, demonstrating a 50% reduction in startup overhead. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为现代人工智能的基石，推动了自然语言处理的突破，并扩展到涉及图像、音频和视频的多模态任务。与大多数计算软件一样，区分常规运行时性能和启动开销很重要。早期的研究主要关注运行时性能：提高训练效率和稳定性。本文则重点关注日益关键的训练启动开销问题：训练作业开始执行前的延迟。在大规模的工业级LLMs中，启动开销尤为重要，因为错误发生的频率更高，而且多个团队在迭代更新和调试周期中协同工作。在一个训练集群中，超过3.5%的GPU时间因启动开销而被浪费。

在这项工作中，我们基于实际生产数据呈现了对LLM训练启动开销的首次深入了解。我们分析了启动成本的组成，量化了其直接影响，并考察了其随作业规模的变化情况。这些见解促使我们设计了Bootseer系统级优化框架，以解决三大主要启动瓶颈：（a）容器镜像加载，（b）运行时依赖安装，（c）模型检查点恢复。为缓解这些瓶颈，Bootseer引入了三种技术：（a）热点块记录与预取，（b）依赖快照，（c）条带化的HDFS-FUSE。Bootseer已在生产环境中部署，并在实际的LLM训练工作负载上进行了评估，结果显示启动开销减少了50%。 

---
# Learning What Matters: Probabilistic Task Selection via Mutual Information for Model Finetuning 

**Title (ZH)**: 学习重要性内容：通过互信息进行概率任务选择的模型微调 

**Authors**: Prateek Chanda, Saral Sureka, Parth Pratim Chatterjee, Krishnateja Killamsetty, Nikhil Shivakumar Nayak, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12612)  

**Abstract**: The performance of finetuned large language models (LLMs) hinges critically on the composition of the training mixture. However, selecting an optimal blend of task datasets remains a largely manual, heuristic driven process, with practitioners often relying on uniform or size based sampling strategies. We introduce TASKPGM, a principled and scalable framework for mixture optimization that selects continuous task proportions by minimizing an energy function over a Markov Random Field (MRF). Task relationships are modeled using behavioral divergences such as Jensen Shannon Divergence and Pointwise Mutual Information computed from the predictive distributions of single task finetuned models. Our method yields a closed form solution under simplex constraints and provably balances representativeness and diversity among tasks. We provide theoretical guarantees, including weak submodularity for budgeted variants, and demonstrate consistent empirical improvements on Llama 2 and Mistral across evaluation suites such as MMLU and BIGBench. Beyond performance, TASKPGM offers interpretable insights into task influence and mixture composition, making it a powerful tool for efficient and robust LLM finetuning. 

**Abstract (ZH)**: 细调大型语言模型的表现关键取决于训练混合物的组成。然而，选择最优的任务数据集混合比例仍然是一个主要依赖手工和启发式驱动的过程，实践者通常依赖均匀或基于大小的取样策略。我们引入了TASKPGM，这是一种原理上和可扩展的混合优化框架，通过在马尔可夫随机场(MRF)上最小化能量函数来选择连续的任务比例。使用行为差异，如从单任务细调模型的预测分布中计算的Jensen Shannon散度和点wise互信息来建模任务关系。该方法在 simples约束下提供了闭合形式的解，并且能够证明在任务的代表性和多样性之间的平衡。我们提供了理论上的保证，包括预算化变体的弱子模性，并在LLama 2和Mistral上的一系列评估套件（如MMLU和BIGBench）中展示了持续的经验改进。除了性能外，TASKPGM还提供了任务影响和混合组成的可解释洞察，使其成为高效和鲁棒大型语言模型细调的强大工具。 

---
# MS-DGCNN++: A Multi-Scale Fusion Dynamic Graph Neural Network with Biological Knowledge Integration for LiDAR Tree Species Classification 

**Title (ZH)**: MS-DGCNN++: 结合生物知识的多尺度融合动态图神经网络在LiDAR树木物种分类中的应用 

**Authors**: Said Ohamouddou, Abdellatif El Afia, Hanaa El Afia, Raddouane Chiheb  

**Link**: [PDF](https://arxiv.org/pdf/2507.12602)  

**Abstract**: Tree species classification from terrestrial LiDAR point clouds is challenging because of the complex multi-scale geometric structures in forest environments. Existing approaches using multi-scale dynamic graph convolutional neural networks (MS-DGCNN) employ parallel multi-scale processing, which fails to capture the semantic relationships between the hierarchical levels of the tree architecture. We present MS-DGCNN++, a hierarchical multiscale fusion dynamic graph convolutional network that uses semantically meaningful feature extraction at local, branch, and canopy scales with cross-scale information propagation. Our method employs scale-specific feature engineering, including standard geometric features for the local scale, normalized relative vectors for the branch scale, and distance information for the canopy scale. This hierarchical approach replaces uniform parallel processing with semantically differentiated representations that are aligned with the natural tree structure. Under the same proposed tree species data augmentation strategy for all experiments, MS-DGCNN++ achieved an accuracy of 94.96 \% on STPCTLS, outperforming DGCNN, MS-DGCNN, and the state-of-the-art model PPT. On FOR-species20K, it achieves 67.25\% accuracy (6.1\% improvement compared to MS-DGCNN). For standard 3D object recognition, our method outperformed DGCNN and MS-DGCNN with overall accuracies of 93.15\% on ModelNet40 and 94.05\% on ModelNet10. With lower parameters and reduced complexity compared to state-of-the-art transformer approaches, our method is suitable for resource-constrained applications while maintaining a competitive accuracy. Beyond tree classification, the method generalizes to standard 3D object recognition, establishing it as a versatile solution for diverse point cloud processing applications. The implementation code is publicly available at this https URL. 

**Abstract (ZH)**: 基于 terrestrial LiDAR 点云的树种分类因森林环境中的复杂多尺度几何结构而具挑战性。现有的使用多尺度动态图卷积神经网络（MS-DGCNN）的方法采用平行多尺度处理，无法捕获树架构层级间的语义关系。我们提出 MS-DGCNN++，一种层次化的多尺度融合动态图卷积网络，能够在局部、枝条和冠层尺度上提取语义上相关的特征，并进行跨尺度信息传播。该方法采用特定尺度的特征工程，包括局部尺度的标准几何特征、枝条尺度的归一化相对向量以及冠层尺度的距离信息。这种层次化方法用语义上差别的表示替代了均匀的并行处理，与自然树结构保持一致。在相同的树种数据增强策略下，MS-DGCNN++ 在 STPCTLS 上实现了 94.96% 的准确率，优于 DGCNN、MS-DGCNN 和最先进的模型 PPT。在 FOR-species20K 上，其准确率为 67.25%（相较于 MS-DGCNN 提高了 6.1%）。对于标准的 3D 对象识别，我们的方法在 ModelNet40 上的整体准确率为 93.15%，在 ModelNet10 上为 94.05%，均优于 DGCNN 和 MS-DGCNN。相比于最先进的变换器方法，我们的方法在资源受限的应用中具有较低的参数量和较低的复杂度，同时保持了较高的准确率。除了树种分类之外，该方法还适用于标准的 3D 对象识别，使其成为处理点云应用的通用解决方案。完整的实现代码可在以下链接获取。 

---
# Assay2Mol: large language model-based drug design using BioAssay context 

**Title (ZH)**: Assay2Mol：基于生物活性上下文的大语言模型药物设计 

**Authors**: Yifan Deng, Spencer S. Ericksen, Anthony Gitter  

**Link**: [PDF](https://arxiv.org/pdf/2507.12574)  

**Abstract**: Scientific databases aggregate vast amounts of quantitative data alongside descriptive text. In biochemistry, molecule screening assays evaluate the functional responses of candidate molecules against disease targets. Unstructured text that describes the biological mechanisms through which these targets operate, experimental screening protocols, and other attributes of assays offer rich information for new drug discovery campaigns but has been untapped because of that unstructured format. We present Assay2Mol, a large language model-based workflow that can capitalize on the vast existing biochemical screening assays for early-stage drug discovery. Assay2Mol retrieves existing assay records involving targets similar to the new target and generates candidate molecules using in-context learning with the retrieved assay screening data. Assay2Mol outperforms recent machine learning approaches that generate candidate ligand molecules for target protein structures, while also promoting more synthesizable molecule generation. 

**Abstract (ZH)**: 科学数据库汇集了大量的定量数据和描述性文本。在生物化学中，分子筛选试验评估候选分子针对疾病靶点的功能反应。描述生物机制、实验筛选方案和其他试验属性的非结构化文本为新型药物发现提供了丰富的信息，但由于其非结构化格式，这些信息尚未被充分利用。我们提出了一种基于大型语言模型的工作流Assay2Mol，可以利用已有的大量生物化学筛选试验，以促进早期药物发现。Assay2Mol检索与新靶点相似的目标的现有试验记录，并使用检索到的筛选数据进行上下文学习生成候选分子。Assay2Mol优于针对靶点蛋白结构生成候选配体分子的近期机器学习方法，同时促进了更易于合成的分子生成。 

---
# Safeguarding Federated Learning-based Road Condition Classification 

**Title (ZH)**: 基于联邦学习的道路状况分类安全防护 

**Authors**: Sheng Liu, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2507.12568)  

**Abstract**: Federated Learning (FL) has emerged as a promising solution for privacy-preserving autonomous driving, specifically camera-based Road Condition Classification (RCC) systems, harnessing distributed sensing, computing, and communication resources on board vehicles without sharing sensitive image data. However, the collaborative nature of FL-RCC frameworks introduces new vulnerabilities: Targeted Label Flipping Attacks (TLFAs), in which malicious clients (vehicles) deliberately alter their training data labels to compromise the learned model inference performance. Such attacks can, e.g., cause a vehicle to mis-classify slippery, dangerous road conditions as pristine and exceed recommended speed. However, TLFAs for FL-based RCC systems are largely missing. We address this challenge with a threefold contribution: 1) we disclose the vulnerability of existing FL-RCC systems to TLFAs; 2) we introduce a novel label-distance-based metric to precisely quantify the safety risks posed by TLFAs; and 3) we propose FLARE, a defensive mechanism leveraging neuron-wise analysis of the output layer to mitigate TLFA effects. Extensive experiments across three RCC tasks, four evaluation metrics, six baselines, and three deep learning models demonstrate both the severity of TLFAs on FL-RCC systems and the effectiveness of FLARE in mitigating the attack impact. 

**Abstract (ZH)**: 联邦学习（FL）在隐私保护自动驾驶中的应用：针对基于摄像头的道路条件分类（RCC）系统的 targeted label flipping 攻击及其防御机制研究 

---
# Can Mental Imagery Improve the Thinking Capabilities of AI Systems? 

**Title (ZH)**: 心智想象能提升人工智能系统的思维能力吗？ 

**Authors**: Slimane Larabi  

**Link**: [PDF](https://arxiv.org/pdf/2507.12555)  

**Abstract**: Although existing models can interact with humans and provide satisfactory responses, they lack the ability to act autonomously or engage in independent reasoning. Furthermore, input data in these models is typically provided as explicit queries, even when some sensory data is already acquired.
In addition, AI agents, which are computational entities designed to perform tasks and make decisions autonomously based on their programming, data inputs, and learned knowledge, have shown significant progress. However, they struggle with integrating knowledge across multiple domains, unlike humans.
Mental imagery plays a fundamental role in the brain's thinking process, which involves performing tasks based on internal multisensory data, planned actions, needs, and reasoning capabilities. In this paper, we investigate how to integrate mental imagery into a machine thinking framework and how this could be beneficial in initiating the thinking process. Our proposed machine thinking framework integrates a Cognitive thinking unit supported by three auxiliary units: the Input Data Unit, the Needs Unit, and the Mental Imagery Unit. Within this framework, data is represented as natural language sentences or drawn sketches, serving both informative and decision-making purposes. We conducted validation tests for this framework, and the results are presented and discussed. 

**Abstract (ZH)**: 尽管现有的模型可以与人类交互并提供满意的回答，但它们缺乏自主行动或独立推理的能力。此外，这些模型中的输入数据通常以明确的查询形式提供，即使已经获得了某些感测数据。
此外，能够自主地根据编程、数据输入和学习的知识来执行任务并作出决策的人工智能代理已经取得了显著的进步。然而，它们在跨多个领域整合知识方面存在困难，不像人类那样能够做到这一点。
心智成像是大脑思维过程中的一个基本组成部分，涉及基于内部多感官数据、计划的动作、需求和推理能力来执行任务。本文探讨如何将心智成像集成到机器思维框架中，以及这种集成如何有助于启动思维过程。我们提出的一种机器思维框架集成了一个认知思维单元和支持该单元的三个辅助单元：输入数据单元、需求单元和心智成像单元。在这种框架中，数据以自然语言句子或绘制的草图形式表示，既具有信息传递功能，也具有决策制定功能。我们为此框架进行了验证测试，结果进行了展示和讨论。 

---
# Is This Just Fantasy? Language Model Representations Reflect Human Judgments of Event Plausibility 

**Title (ZH)**: 这只是幻想吗？语言模型表示反映了事件可能性的人类判断。 

**Authors**: Michael A. Lepori, Jennifer Hu, Ishita Dasgupta, Roma Patel, Thomas Serre, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2507.12553)  

**Abstract**: Language models (LMs) are used for a diverse range of tasks, from question answering to writing fantastical stories. In order to reliably accomplish these tasks, LMs must be able to discern the modal category of a sentence (i.e., whether it describes something that is possible, impossible, completely nonsensical, etc.). However, recent studies have called into question the ability of LMs to categorize sentences according to modality (Michaelov et al., 2025; Kauf et al., 2023). In this work, we identify linear representations that discriminate between modal categories within a variety of LMs, or modal difference vectors. Analysis of modal difference vectors reveals that LMs have access to more reliable modal categorization judgments than previously reported. Furthermore, we find that modal difference vectors emerge in a consistent order as models become more competent (i.e., through training steps, layers, and parameter count). Notably, we find that modal difference vectors identified within LM activations can be used to model fine-grained human categorization behavior. This potentially provides a novel view into how human participants distinguish between modal categories, which we explore by correlating projections along modal difference vectors with human participants' ratings of interpretable features. In summary, we derive new insights into LM modal categorization using techniques from mechanistic interpretability, with the potential to inform our understanding of modal categorization in humans. 

**Abstract (ZH)**: 语言模型通过对多种任务的处理，从回答问题到撰写幻想故事。为了可靠地完成这些任务，语言模型必须能够区分句子的模态类别（即它描述的是可能、不可能、完全不合逻辑等）。然而，最近的研究对语言模型根据模态进行分类的能力提出了质疑（Michaelov et al., 2025；Kauf et al., 2023）。在此工作中，我们识别出能够在不同类型的语言模型中区分模态类别的线性表示，或模态差异向量。模态差异向量的分析表明，语言模型能够进行比之前报告更为可靠的模态分类判断。此外，我们发现随着模型能力的提升（即通过训练步骤、层和参数数量），模态差异向量会出现一致的顺序。值得注意的是，我们发现，在语言模型激活中识别出的模态差异向量可用于模拟精细的人类分类行为。这可能提供了一种新的视角来了解人类如何区分模态类别，我们通过将模态差异向量上的投影与人类参与者对可解释特征的评分进行关联来探索这一视角。总结而言，我们利用机制可解释性技术获得了关于语言模型模态分类的新见解，这有可能帮助我们更好地理解人类的模态分类。 

---
# Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models 

**Title (ZH)**: 开放世界认知的即需即用的概率模型合成 

**Authors**: Lionel Wong, Katherine M. Collins, Lance Ying, Cedegao E. Zhang, Adrian Weller, Tobias Gersternberg, Timothy O'Donnell, Alexander K. Lew, Jacob D. Andreas, Joshua B. Tenenbaum, Tyler Brooke-Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2507.12547)  

**Abstract**: When faced with novel situations, people are able to marshal relevant considerations from a wide range of background knowledge and put these to use in inferences and predictions. What permits us to draw in globally relevant information and reason over it coherently? Here, we explore the hypothesis that people use a combination of distributed and symbolic representations to construct bespoke mental models tailored to novel situations. We propose a computational implementation of this idea -- a ``Model Synthesis Architecture'' (MSA) -- using language models to implement global relevance-based retrieval and model synthesis and probabilistic programs to implement bespoke, coherent world models. We evaluate our MSA as a model of human judgments on a novel reasoning dataset. The dataset -- built around a `Model Olympics` domain of sports vignettes -- tests models' capacity for human-like, open-ended reasoning by requiring (i) judgments about novel causal structures described in language; (ii) drawing on large bodies of background knowledge; and (iii) doing both in light of observations that introduce arbitrary novel variables. Our MSA approach captures human judgments better than language model-only baselines, under both direct and chain-of-thought generations from the LM that supports model synthesis. These results suggest that MSAs can be implemented in a way that mirrors people's ability to deliver locally coherent reasoning over globally relevant variables, offering a path to understanding and replicating human reasoning in open-ended domains. 

**Abstract (ZH)**: 当面临新颖情境时，人们能够调动广泛背景知识中的相关考虑，并将其用于推理和预测。是什么使得我们能够整合全局相关的信息并一致地进行推理？在这里，我们探索了一个假说，即人们使用分布式表示和符号表示的结合来构建针对新颖情境量身定制的心理模型。我们提出了一种计算实现这一想法的方法——“模型合成架构”（MSA），使用语言模型进行全局相关性检索和模型合成，使用概率程序实现定制的、一致的世界模型。我们以一个基于“模型奥林匹克”领域体育情境的新型推理数据集为基础，评估MSA作为人类判断的模型的有效性。该数据集要求模型不仅判断语言中描述的新颖因果结构，还调动广泛背景知识，并在引入任意新颖变量时结合观察进行推理，测试模型进行类似人类开放性推理的能力。我们的MSA方法在直接生成和链式思考生成语言模型支持的模型合成中均比仅使用语言模型的基线更准确地捕捉了人类判断，这表明MSA可以模仿人们以局部一致的方式对全局变量进行推理的能力，为我们理解并复制人类在开放性领域中的推理提供了一条路径。 

---
# MindJourney: Test-Time Scaling with World Models for Spatial Reasoning 

**Title (ZH)**: MindJourney：基于世界模型的测试时扩展方法的空间推理 

**Authors**: Yuncong Yang, Jiageng Liu, Zheyuan Zhang, Siyuan Zhou, Reuben Tan, Jianwei Yang, Yilun Du, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12508)  

**Abstract**: Spatial reasoning in 3D space is central to human cognition and indispensable for embodied tasks such as navigation and manipulation. However, state-of-the-art vision-language models (VLMs) struggle frequently with tasks as simple as anticipating how a scene will look after an egocentric motion: they perceive 2D images but lack an internal model of 3D dynamics. We therefore propose MindJourney, a test-time scaling framework that grants a VLM with this missing capability by coupling it to a controllable world model based on video diffusion. The VLM iteratively sketches a concise camera trajectory, while the world model synthesizes the corresponding view at each step. The VLM then reasons over this multi-view evidence gathered during the interactive exploration. Without any fine-tuning, our MindJourney achieves over an average 8% performance boost on the representative spatial reasoning benchmark SAT, showing that pairing VLMs with world models for test-time scaling offers a simple, plug-and-play route to robust 3D reasoning. Meanwhile, our method also improves upon the test-time inference VLMs trained through reinforcement learning, which demonstrates the potential of our method that utilizes world models for test-time scaling. 

**Abstract (ZH)**: 三维空间中的空间推理是人类认知的核心，并且对于导航和操作等具身任务至关重要。然而，最先进的视觉-语言模型（VLMs）在预测主体运动后场景会是什么样子这样简单的工作上经常遇到困难：它们只能感知二维图像，而缺乏对三维动态的内部模型。因此，我们提出了MindJourney，这是一种测试时缩放框架，通过将VLM与基于视频扩散的可控世界模型耦合，赋予其缺失的这种能力。VLM迭代地勾勒出简要的摄像机轨迹，而世界模型则在每一步生成相应的视角。VLM随后对在交互式探索过程中收集到的多视角证据进行推理。无需任何微调，我们的MindJourney在代表性的空间推理基准SAT上平均实现了超过8%的性能提升，表明将VLM与世界模型结合进行测试时缩放提供了一种简单且即插即用的途径，以实现稳健的三维推理。同时，我们的方法也改进了通过强化学习训练的测试时推理VLM，这表明了我们方法利用世界模型进行测试时缩放的潜力。 

---
# Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training 

**Title (ZH)**: 扩大RL的应用范围：通过延长训练解锁LLMs的多样化推理 

**Authors**: Mingjie Liu, Shizhe Diao, Jian Hu, Ximing Lu, Xin Dong, Hao Zhang, Alexander Bukharin, Shaokun Zhang, Jiaqi Zeng, Makesh Narsimhan Sreedhar, Gerald Shen, David Mosallanezhad, Di Zhang, Jonas Yang, June Yang, Oleksii Kuchaiev, Guilin Liu, Zhiding Yu, Pavlo Molchanov, Yejin Choi, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.12507)  

**Abstract**: Recent advancements in reasoning-focused language models such as OpenAI's O1 and DeepSeek-R1 have shown that scaling test-time computation-through chain-of-thought reasoning and iterative exploration-can yield substantial improvements on complex tasks like mathematics and code generation. These breakthroughs have been driven by large-scale reinforcement learning (RL), particularly when combined with verifiable reward signals that provide objective and grounded supervision. In this report, we investigate the effects of prolonged reinforcement learning on a small language model across a diverse set of reasoning domains. Our work identifies several key ingredients for effective training, including the use of verifiable reward tasks, enhancements to Group Relative Policy Optimization (GRPO), and practical techniques to improve training stability and generalization. We introduce controlled KL regularization, clipping ratio, and periodic reference policy resets as critical components for unlocking long-term performance gains. Our model achieves significant improvements over strong baselines, including +14.7% on math, +13.9% on coding, and +54.8% on logic puzzle tasks. To facilitate continued research, we release our model publicly. 

**Abstract (ZH)**: 近期专注于推理的语言模型（如OpenAI的O1和DeepSeek-R1）的发展表明，通过链式思考推理和迭代探索扩展测试时的计算可以显著提高复杂任务（如数学和代码生成）的表现。这些突破主要得益于大规模强化学习（RL），特别是结合了可验证的奖励信号，这些信号提供了客观和具体的监督。在本报告中，我们研究了长期强化学习对一个小语言模型在多种推理领域的效果。我们的工作识别了有效训练的关键要素，包括使用可验证奖励任务、增强Group Relative Policy Optimization (GRPO)以及提高训练稳定性和泛化性的实用技术。我们引入了受控KL正则化、剪裁比例和周期性参考策略重置作为实现长期性能提升的关键成分。我们的模型在数学、编程和逻辑谜题任务上均取得了显著改进，分别提高了14.7%、13.9%和54.8%。为了促进继续研究，我们已将模型公开发布。 

---
# Transforming Football Data into Object-centric Event Logs with Spatial Context Information 

**Title (ZH)**: 将足球数据转化为具有空间上下文信息的物为中心的事件日志 

**Authors**: Vito Chan, Lennart Ebert, Paul-Julius Hillmann, Christoffer Rubensson, Stephan A. Fahrenkrog-Petersen, Jan Mendling  

**Link**: [PDF](https://arxiv.org/pdf/2507.12504)  

**Abstract**: Object-centric event logs expand the conventional single-case notion event log by considering multiple objects, allowing for the analysis of more complex and realistic process behavior. However, the number of real-world object-centric event logs remains limited, and further studies are needed to test their usefulness. The increasing availability of data from team sports can facilitate object-centric process mining, leveraging both real-world data and suitable use cases. In this paper, we present a framework for transforming football (soccer) data into an object-centric event log, further enhanced with a spatial dimension. We demonstrate the effectiveness of our framework by generating object-centric event logs based on real-world football data and discuss the results for varying process representations. With our paper, we provide the first example for object-centric event logs in football analytics. Future work should consider variant analysis and filtering techniques to better handle variability 

**Abstract (ZH)**: 基于对象的事件日志扩展了传统的单一案例事件日志概念，考虑了多个对象，从而允许分析更复杂和现实的过程行为。然而，实际世界的基于对象的事件日志仍然有限，需要进一步研究以检验其实用性。来自团队运动的数据越来越多地可以促进基于对象的过程挖掘，利用实际数据和合适的用例。本文提出了一种将足球数据转换为基于对象的事件日志的框架，并进一步增强其空间维度。我们通过基于实际足球数据生成基于对象的事件日志，展示了该框架的有效性，并讨论了不同过程表示的结果。通过本文，我们提供了足球分析中基于对象的事件日志的第一个示例。未来的工作应考虑变体分析和过滤技术以更好地处理变异性。 

---
# FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making 

**Title (ZH)**: FOUNDER：将基础模型 grounding 在世界模型中以实现开放-ended 体感决策 

**Authors**: Yucen Wang, Rui Yu, Shenghua Wan, Le Gan, De-Chuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12496)  

**Abstract**: Foundation Models (FMs) and World Models (WMs) offer complementary strengths in task generalization at different levels. In this work, we propose FOUNDER, a framework that integrates the generalizable knowledge embedded in FMs with the dynamic modeling capabilities of WMs to enable open-ended task solving in embodied environments in a reward-free manner. We learn a mapping function that grounds FM representations in the WM state space, effectively inferring the agent's physical states in the world simulator from external observations. This mapping enables the learning of a goal-conditioned policy through imagination during behavior learning, with the mapped task serving as the goal state. Our method leverages the predicted temporal distance to the goal state as an informative reward signal. FOUNDER demonstrates superior performance on various multi-task offline visual control benchmarks, excelling in capturing the deep-level semantics of tasks specified by text or videos, particularly in scenarios involving complex observations or domain gaps where prior methods struggle. The consistency of our learned reward function with the ground-truth reward is also empirically validated. Our project website is this https URL. 

**Abstract (ZH)**: Foundation模型（FMs）和世界模型（WMs）在不同层面上为任务泛化提供了互补的优势。在本工作中，我们提出了一种名为FOUNDER的框架，该框架将FMs中嵌入的一般性知识与WMs的动态建模能力相结合，以在奖励免费的方式下实现开放性的任务解决，应用于具身环境。我们学习了一个映射函数，将FM表示嵌入到WM状态空间中，从而有效地从外部观察中推断出代理在世界仿真的物理状态。该映射使代理能够在行为学习过程中通过想象学习带有目标条件的策略，映射的任务作为目标状态。我们的方法利用预测到目标状态的时间距离作为信息性的奖励信号。FOUNDER在各种多任务离线视觉控制基准测试中展示了优越的性能，特别擅长捕捉由文本或视频指定的任务的深层语义，特别是在复杂观察或领域差距等场景中，优于先前的方法。同时，我们的学习奖励函数与真实奖励的一致性也得到了实证验证。我们的项目网站是这个 https URL。 

---
# Sporadic Federated Learning Approach in Quantum Environment to Tackle Quantum Noise 

**Title (ZH)**: 量子环境中基于 sporadic 联邦学习的方法以应对量子噪声 

**Authors**: Ratun Rahman, Atit Pokharel, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.12492)  

**Abstract**: Quantum Federated Learning (QFL) is an emerging paradigm that combines quantum computing and federated learning (FL) to enable decentralized model training while maintaining data privacy over quantum networks. However, quantum noise remains a significant barrier in QFL, since modern quantum devices experience heterogeneous noise levels due to variances in hardware quality and sensitivity to quantum decoherence, resulting in inadequate training performance. To address this issue, we propose SpoQFL, a novel QFL framework that leverages sporadic learning to mitigate quantum noise heterogeneity in distributed quantum systems. SpoQFL dynamically adjusts training strategies based on noise fluctuations, enhancing model robustness, convergence stability, and overall learning efficiency. Extensive experiments on real-world datasets demonstrate that SpoQFL significantly outperforms conventional QFL approaches, achieving superior training performance and more stable convergence. 

**Abstract (ZH)**: 量子联邦学习（QFL）是一种结合量子计算和联邦学习的新兴范式，旨在通过量子网络实现数据隐私下的去中心化模型训练。然而，量子噪声仍然是QFL中的一个重大障碍，因为现代量子设备由于硬件质量差异和对量子退相干的敏感性不同，导致噪声水平异质，从而影响训练性能。为解决这一问题，我们提出了一种名为SpoQFL的新颖QFL框架，利用间歇学习来减轻分布式量子系统中的量子噪声异质性。SpoQFL根据噪声波动动态调整训练策略，增强模型的鲁棒性、收敛稳定性和整体学习效率。实验结果表明，SpoQFL显著优于传统QFL方法，实现了更高的训练性能和更稳定的收敛。 

---
# Spatially Grounded Explanations in Vision Language Models for Document Visual Question Answering 

**Title (ZH)**: 视觉语言模型中面向文档视觉问答的空间 grounding 解释 

**Authors**: Maximiliano Hormazábal Lagos, Héctor Cerezo-Costas, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2507.12490)  

**Abstract**: We introduce EaGERS, a fully training-free and model-agnostic pipeline that (1) generates natural language rationales via a vision language model, (2) grounds these rationales to spatial sub-regions by computing multimodal embedding similarities over a configurable grid with majority voting, and (3) restricts the generation of responses only from the relevant regions selected in the masked image. Experiments on the DocVQA dataset demonstrate that our best configuration not only outperforms the base model on exact match accuracy and Average Normalized Levenshtein Similarity metrics but also enhances transparency and reproducibility in DocVQA without additional model fine-tuning. 

**Abstract (ZH)**: 我们介绍了一种名为EaGERS的全流程无需训练且模型无关的管道，该管道通过（1）使用视觉语言模型生成自然语言理由，（2）通过在可配置网格上计算多模态嵌入相似性并结合多数投票将这些理由与空间子区域关联起来，以及（3）仅从掩码图像中选择的相关区域生成响应来工作。实验结果表明，我们的最佳配置不仅在准确匹配和平均归一化Levenshtein相似度指标上优于基准模型，还在DocVQA数据集上提升了透明度和可重现性，无需额外的模型微调。 

---
# On multiagent online problems with predictions 

**Title (ZH)**: 多智能体在线问题的预测研究 

**Authors**: Gabriel Istrate, Cosmin Bonchis, Victor Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2507.12486)  

**Abstract**: We study the power of (competitive) algorithms with predictions in a multiagent setting. We introduce a two predictor framework, that assumes that agents use one predictor for their future (self) behavior, and one for the behavior of the other players. The main problem we are concerned with is understanding what are the best competitive ratios that can be achieved by employing such predictors, under various assumptions on predictor quality.
As an illustration of our framework, we introduce and analyze a multiagent version of the ski-rental problem. In this problem agents can collaborate by pooling resources to get a group license for some asset. If the license price is not met then agents have to rent the asset individually for the day at a unit price. Otherwise the license becomes available forever to everyone at no extra cost.
In the particular case of perfect other predictions the algorithm that follows the self predictor is optimal but not robust to mispredictions of agent's future behavior; we give an algorithm with better robustness properties and benchmark it. 

**Abstract (ZH)**: 我们研究预测在多代理环境下竞争算法的能力。我们引入了一种两预测框架，假设代理使用一个预测他们自己未来行为的预测器，另一个预测其他玩家的行为。我们主要关注的问题是在各种假设条件下，可以实现的最佳竞争比是什么。

为了展示我们的框架，我们引入并分析了一个多代理环境下的滑雪租赁问题。在这个问题中，代理可以通过共享资源来获得某种资产的团体许可证。如果许可证价格未达到，则代理必须以单一价格租用该资产一天；否则，许可证将对所有人永久免费。

在其他预测完美的特殊情况下，遵循自我预测的算法是最优的，但对代理未来行为的误预测不够 robust；我们提出一个具有更好 robust 性质的算法并将其作为基准进行比较。 

---
# Quantum Transfer Learning to Boost Dementia Detection 

**Title (ZH)**: 量子迁移学习以提升痴呆症检测 

**Authors**: Sounak Bhowmik, Talita Perciano, Himanshu Thapliyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.12485)  

**Abstract**: Dementia is a devastating condition with profound implications for individuals, families, and healthcare systems. Early and accurate detection of dementia is critical for timely intervention and improved patient outcomes. While classical machine learning and deep learning approaches have been explored extensively for dementia prediction, these solutions often struggle with high-dimensional biomedical data and large-scale datasets, quickly reaching computational and performance limitations. To address this challenge, quantum machine learning (QML) has emerged as a promising paradigm, offering faster training and advanced pattern recognition capabilities. This work aims to demonstrate the potential of quantum transfer learning (QTL) to enhance the performance of a weak classical deep learning model applied to a binary classification task for dementia detection. Besides, we show the effect of noise on the QTL-based approach, investigating the reliability and robustness of this method. Using the OASIS 2 dataset, we show how quantum techniques can transform a suboptimal classical model into a more effective solution for biomedical image classification, highlighting their potential impact on advancing healthcare technology. 

**Abstract (ZH)**: 认知障碍是一种对个人、家庭和医疗卫生系统产生深远影响的毁灭性状况。早期和准确地检测认知障碍对于及时干预和改善患者结果至关重要。虽然经典的机器学习和深度学习方法在认知障碍预测方面得到了广泛探索，但这些解决方案往往难以应对高维生物医学数据和大规模数据集，很快达到了计算能力和性能的限制。为应对这一挑战，量子机器学习（QML）已逐渐成为一种有前景的范式，提供更快的训练和高级模式识别能力。本工作旨在展示量子迁移学习（QTL）增强弱经典深度学习模型在认知障碍二分类任务中的性能潜力。此外，我们展示了噪声对基于QTL方法的影响，探讨了该方法的可靠性和稳健性。使用OASIS 2数据集，我们展示了量子技术如何将一个次优的经典模型转换为更有效的生物医学图像分类解决方案，突显了其在推动医疗保健技术进步方面的潜在影响。 

---
# Kodezi Chronos: A Debugging-First Language Model for Repository-Scale, Memory-Driven Code Understanding 

**Title (ZH)**: Kodezi Chronos：一种面向调试的语言模型，用于仓库规模的记忆驱动代码理解 

**Authors**: Ishraq Khan, Assad Chowdary, Sharoz Haseeb, Urvish Patel  

**Link**: [PDF](https://arxiv.org/pdf/2507.12482)  

**Abstract**: Large Language Models (LLMs) have advanced code generation and software automation, but are fundamentally constrained by limited inference-time context and lack of explicit code structure reasoning. We introduce Kodezi Chronos, a next-generation architecture for autonomous code understanding, debugging, and maintenance, designed to operate across ultra-long contexts comprising entire codebases, histories, and documentation, all without fixed window limits. Kodezi Chronos leverages a multi-level embedding memory engine, combining vector and graph-based indexing with continuous code-aware retrieval. This enables efficient and accurate reasoning over millions of lines of code, supporting repository-scale comprehension, multi-file refactoring, and real-time self-healing actions. Our evaluation introduces a novel Multi Random Retrieval benchmark, specifically tailored to the software engineering domain. Unlike classical retrieval benchmarks, this method requires the model to resolve arbitrarily distant and obfuscated associations across code artifacts, simulating realistic tasks such as variable tracing, dependency migration, and semantic bug localization. Chronos outperforms prior LLMs and code models, demonstrating a 23% improvement in real-world bug detection and reducing debugging cycles by up to 40% compared to traditional sequence-based approaches. By natively interfacing with IDEs and CI/CD workflows, Chronos enables seamless, autonomous software maintenance, elevating code reliability and productivity while reducing manual effort. These results mark a critical advance toward self-sustaining, continuously optimized software ecosystems. 

**Abstract (ZH)**: 下一代自主代码理解、调试和维护架构Kodezi Chronos：跨超长上下文的高效代码推理与软件自运维 

---
# LLM-Powered Quantum Code Transpilation 

**Title (ZH)**: LLM驱动的量子代码转译 

**Authors**: Nazanin Siavash, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2507.12480)  

**Abstract**: There exist various Software Development Kits (SDKs) tailored to different quantum computing platforms. These are known as Quantum SDKs (QSDKs). Examples include but are not limited to Qiskit, Cirq, and PennyLane. However, this diversity presents significant challenges for interoperability and cross-platform development of hybrid quantum-classical software systems. Traditional rule-based transpilers for translating code between QSDKs are time-consuming to design and maintain, requiring deep expertise and rigid mappings in the source and destination code. In this study, we explore the use of Large Language Models (LLMs) as a flexible and automated solution. Leveraging their pretrained knowledge and contextual reasoning capabilities, we position LLMs as programming language-agnostic transpilers capable of converting quantum programs from one QSDK to another while preserving functional equivalence. Our approach eliminates the need for manually defined transformation rules and offers a scalable solution to quantum software portability. This work represents a step toward enabling intelligent, general-purpose transpilation in the quantum computing ecosystem. 

**Abstract (ZH)**: 各种量子计算平台都有专门的软件开发工具包（SDKs），这些工具包被称为量子SDK（QSDKs），例如Qiskit、Cirq和PennyLane。然而，这种多样性为混合量子-经典软件系统的互操作性和跨平台开发带来了显著挑战。传统的基于规则的编译器设计和维护耗时且需要深厚的专业知识和源代码与目标代码之间的严格映射。本研究探索了大型语言模型（LLMs）作为灵活且自动化的解决方案。利用它们预训练的知识和上下文推理能力，我们将LLMs定位为一种编程语言无关的编译器，能够将一种QSDK中的量子程序转换为另一种QSDK，同时保持功能等价性。我们的方法消除了手动定义转换规则的需要，并提供了一种量子软件可移植性的可扩展解决方案。本工作代表了在量子计算生态系统中实现智能、通用编译的新步骤。 

---
# Coarse Addition and the St. Petersburg Paradox: A Heuristic Perspective 

**Title (ZH)**: 粗略加法与圣彼得堡悖论：启发式视角 

**Authors**: Takashi Izumo  

**Link**: [PDF](https://arxiv.org/pdf/2507.12475)  

**Abstract**: The St. Petersburg paradox presents a longstanding challenge in decision theory. It describes a game whose expected value is infinite, yet for which no rational finite stake can be determined. Traditional solutions introduce auxiliary assumptions, such as diminishing marginal utility, temporal discounting, or extended number systems. These methods often involve mathematical refinements that may not correspond to how people actually perceive or process numerical information. This paper explores an alternative approach based on a modified operation of addition defined over coarse partitions of the outcome space. In this model, exact numerical values are grouped into perceptual categories, and each value is replaced by a representative element of its group before being added. This method allows for a phenomenon where repeated additions eventually cease to affect the outcome, a behavior described as inertial stabilization. Although this is not intended as a definitive resolution of the paradox, the proposed framework offers a plausible way to represent how agents with limited cognitive precision might handle divergent reward structures. We demonstrate that the St. Petersburg series can become inert under this coarse addition for a suitably constructed partition. The approach may also have broader applications in behavioral modeling and the study of machine reasoning under perceptual limitations. 

**Abstract (ZH)**: 圣彼得堡悖论为决策理论提出了一个长期挑战。它描述了一个预期值无穷大的游戏，但其中没有任何合理的有限赌注可以确定。传统解决方案引入了辅助假设，如边际效用递减、时间折扣或扩展数值系统。这些方法通常涉及数学 refinements，这些可能并不符合人们实际感知或处理数值信息的方式。本文探讨了一种基于对结果空间粗略划分上定义的修改加法运算的替代方法。在此模型中，精确数值被分组为知觉类别，并在相加前用每个组的代表元素替换。这种方法允许一种现象，即重复加法最终不再影响结果，这种行为称为惯性稳定。虽然这不是解决悖论的最终方案，但提出的新框架提供了一种合理的表示认知精度有限的代理如何处理发散奖励结构的方式。我们证明，在适当构造的划分下，圣彼得堡级数可以在粗略加法下变得惯性稳定。该方法还可能在行为建模以及研究感知限制下的机器推理方面具有更广泛的应用。 

---
# Implementation and Analysis of GPU Algorithms for Vecchia Approximation 

**Title (ZH)**: GPU算法在Vecchia逼近中的实现与分析 

**Authors**: Zachary James, Joseph Guinness  

**Link**: [PDF](https://arxiv.org/pdf/2407.02740)  

**Abstract**: Gaussian Processes have become an indispensable part of the spatial statistician's toolbox but are unsuitable for analyzing large dataset because of the significant time and memory needed to fit the associated model exactly. Vecchia Approximation is widely used to reduce the computational complexity and can be calculated with embarrassingly parallel algorithms. While multi-core software has been developed for Vecchia Approximation, such as the GpGp R package, software designed to run on graphics processing units (GPU) is lacking, despite the tremendous success GPUs have had in statistics and machine learning. We compare three different ways to implement Vecchia Approximation on a GPU: two of which are similar to methods used for other Gaussian Process approximations and one that is new. The impact of memory type on performance is investigated and the final method is optimized accordingly. We show that our new method outperforms the other two and then present it in the GpGpU R package. We compare GpGpU to existing multi-core and GPU-accelerated software by fitting Gaussian Process models on various datasets, including a large spatial-temporal dataset of $n>10^6$ points collected from an earth-observing satellite. Our results show that GpGpU achieves faster runtimes and better predictive accuracy. 

**Abstract (ZH)**: GP Approximation on GPU: A New Method for Vecchia Approximation and Its Implementation in GpGpU R Package 

---
