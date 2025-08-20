# ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents 

**Title (ZH)**: ComputerRL: 扩展面向计算机使用代理的端到端在线强化学习 

**Authors**: Hanyu Lai, Xiao Liu, Yanxiao Zhao, Han Xu, Hanchen Zhang, Bohao Jing, Yanyu Ren, Shuntian Yao, Yuxiao Dong, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14040)  

**Abstract**: We introduce ComputerRL, a framework for autonomous desktop intelligence that enables agents to operate complex digital workspaces skillfully. ComputerRL features the API-GUI paradigm, which unifies programmatic API calls and direct GUI interaction to address the inherent mismatch between machine agents and human-centric desktop environments. Scaling end-to-end RL training is crucial for improvement and generalization across diverse desktop tasks, yet remains challenging due to environmental inefficiency and instability in extended training. To support scalable and robust training, we develop a distributed RL infrastructure capable of orchestrating thousands of parallel virtual desktop environments to accelerate large-scale online RL. Furthermore, we propose Entropulse, a training strategy that alternates reinforcement learning with supervised fine-tuning, effectively mitigating entropy collapse during extended training runs. We employ ComputerRL on open models GLM-4-9B-0414 and Qwen2.5-14B, and evaluate them on the OSWorld benchmark. The AutoGLM-OS-9B based on GLM-4-9B-0414 achieves a new state-of-the-art accuracy of 48.1%, demonstrating significant improvements for general agents in desktop automation. The algorithm and framework are adopted in building AutoGLM (Liu et al., 2024a) 

**Abstract (ZH)**: ComputerRL：面向自主桌面智能的框架 

---
# A Biased Random Key Genetic Algorithm for Solving the Longest Run Subsequence Problem 

**Title (ZH)**: 带有偏置随机密钥的遗传算法求解最长连续子序列问题 

**Authors**: Christian Blum, Pedro Pinacho-Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2508.14020)  

**Abstract**: The longest run subsequence (LRS) problem is an NP-hard combinatorial optimization problem belonging to the class of subsequence problems from bioinformatics. In particular, the problem plays a role in genome reassembly. In this paper, we present a solution to the LRS problem using a Biased Random Key Genetic Algorithm (BRKGA). Our approach places particular focus on the computational efficiency of evaluating individuals, which involves converting vectors of gray values into valid solutions to the problem. For comparison purposes, a Max-Min Ant System is developed and implemented. This is in addition to the application of the integer linear programming solver CPLEX for solving all considered problem instances. The computation results show that the proposed BRKGA is currently a state-of-the-art technique for the LRS problem. Nevertheless, the results also show that there is room for improvement, especially in the context of input strings based on large alphabet sizes. 

**Abstract (ZH)**: 最长运行子序列问题（LRS）是属于生物信息学子序列问题类别的NP难组合优化问题，特别在基因组重构中起着重要作用。本文提出了一种使用有偏随机键遗传算法（BRKGA）解决LRS问题的方法，重点在于计算效率的评估个体过程，即将灰度值向量转换为问题的有效解。为了进行对比，我们开发并实现了最大最小蚁群系统（Max-Min Ant System），同时还应用了整数线性规划求解器CPLEX求解所有考虑的问题实例。计算结果表明，提出的BRKGA目前是解决LRS问题的先进方法。然而，结果也表明，在基于大字母表的输入字符串的背景下，仍有一定的改进空间。 

---
# ChronoLLM: Customizing Language Models for Physics-Based Simulation Code Generation 

**Title (ZH)**: ChronoLLM: 为物理基于的仿真代码生成定制语言模型 

**Authors**: Jingquan Wang, Andrew Negrut, Harry Zhang, Khailanii Slaton, Shu Wang, Radu Serban, Jinlong Wu, Dan Negrut  

**Link**: [PDF](https://arxiv.org/pdf/2508.13975)  

**Abstract**: This contribution is concerned with the following issue: can pretrained large language models (LLMs) be refined and customized to the point where they become virtual assistants helping experts with the effective use of a simulation tool? In this case study, the ``simulation tool'' considered is PyChrono, an open source multi-physics dynamics engine for multibody systems. We present a framework for refining and customizing both open- and closed-source LLMs to harness the power of AI in generating scripts that perform PyChrono virtual experiments. We refine and customize several classes of LLMs through a process that leads to a quantifiable improvement in the quality of the generated PyChrono simulation scripts. These scripts can range from simple single-pendulum simulations to complex virtual experiments involving full vehicles on deformable terrain. While the generated scripts are rarely perfect, they often serve as strong starting points for the user to modify and improve on. Additionally, the LLM can answer specific API questions about the simulator, or recommend modeling approaches. The framework discussed is general and can be applied to lower the entry barrier for simulation tools associated with other application domains. 

**Abstract (ZH)**: 本研究关注的问题是：预训练大型语言模型（LLMs）能否被精炼和定制到能够成为帮助专家有效使用仿真工具的虚拟助手？在这种案例研究中，“仿真工具”是PyChrono，一个开源的多物理场动力学引擎，用于多体系统。我们提出了一种框架，用于精炼和定制开源和闭源的LLMs，以利用AI生成执行PyChrono虚拟实验的脚本的力量。我们通过一个过程对多种类别的LLMs进行精炼和定制，这导致生成的PyChrono仿真脚本质量量化提升。这些脚本可以从小到简单的单摆仿真，到复杂到涉及全车辆在可变形地形上的虚拟实验。虽然生成的脚本很少完美，但它们通常可以作为用户修改和改进的良好起点。此外，LLM还可以回答关于仿真的API问题，或推荐建模方法。所讨论的框架是通用的，可以应用于降低其他应用领域相关仿真工具的使用门槛。 

---
# The Collaboration Paradox: Why Generative AI Requires Both Strategic Intelligence and Operational Stability in Supply Chain Management 

**Title (ZH)**: 生成人工智能中的合作悖论：为什么供应链管理需要战略智慧与运营稳定性的双重保障 

**Authors**: Soumyadeep Dhar  

**Link**: [PDF](https://arxiv.org/pdf/2508.13942)  

**Abstract**: The rise of autonomous, AI-driven agents in economic settings raises critical questions about their emergent strategic behavior. This paper investigates these dynamics in the cooperative context of a multi-echelon supply chain, a system famously prone to instabilities like the bullwhip effect. We conduct computational experiments with generative AI agents, powered by Large Language Models (LLMs), within a controlled supply chain simulation designed to isolate their behavioral tendencies. Our central finding is the "collaboration paradox": a novel, catastrophic failure mode where theoretically superior collaborative AI agents, designed with Vendor-Managed Inventory (VMI) principles, perform even worse than non-AI baselines. We demonstrate that this paradox arises from an operational flaw where agents hoard inventory, starving the system. We then show that resilience is only achieved through a synthesis of two distinct layers: high-level, AI-driven proactive policy-setting to establish robust operational targets, and a low-level, collaborative execution protocol with proactive downstream replenishment to maintain stability. Our final framework, which implements this synthesis, can autonomously generate, evaluate, and quantify a portfolio of viable strategic choices. The work provides a crucial insight into the emergent behaviors of collaborative AI agents and offers a blueprint for designing stable, effective AI-driven systems for business analytics. 

**Abstract (ZH)**: 自主AI驱动代理在经济环境中的兴起引发了对其 emergent 策略行为的关键问题。本文在多级供应链的协作环境中研究这些动态，这是一个众所周知容易出现波动如波动鞭 effect 的系统。我们使用大型语言模型（LLMs）驱动的生成AI代理，在一个控制下的供应链仿真中进行计算实验，以隔离其行为倾向。我们的主要发现是“合作悖论”：理论上更优的协作AI代理，基于供应商管理库存（VMI）原则设计，表现甚至不如非AI基线。我们证明，这一悖论源于一个操作缺陷，即代理囤积库存，使系统枯竭。然后我们表明，只有通过两层的综合，才能实现韧性：高层的AI驱动前瞻性政策设定，以确立稳健的操作目标，和低层的协作执行协议，包括前瞻性下游补充，以维持稳定。我们的最终框架实现了这种综合，可以自主生成、评估和量化可行的战略选择组合。该研究为理解协作AI代理的 emergent 行为提供了重要见解，并为设计业务分析中的稳定、有效的AI驱动系统提供了蓝图。 

---
# Structured Agentic Workflows for Financial Time-Series Modeling with LLMs and Reflective Feedback 

**Title (ZH)**: 结构化代理工作流：结合LLMs和反思性反馈的金融市场时间序列建模 

**Authors**: Yihao Ang, Yifan Bao, Lei Jiang, Jiajie Tao, Anthony K. H. Tung, Lukasz Szpruch, Hao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2508.13915)  

**Abstract**: Time-series data is central to decision-making in financial markets, yet building high-performing, interpretable, and auditable models remains a major challenge. While Automated Machine Learning (AutoML) frameworks streamline model development, they often lack adaptability and responsiveness to domain-specific needs and evolving objectives. Concurrently, Large Language Models (LLMs) have enabled agentic systems capable of reasoning, memory management, and dynamic code generation, offering a path toward more flexible workflow automation. In this paper, we introduce \textsf{TS-Agent}, a modular agentic framework designed to automate and enhance time-series modeling workflows for financial applications. The agent formalizes the pipeline as a structured, iterative decision process across three stages: model selection, code refinement, and fine-tuning, guided by contextual reasoning and experimental feedback. Central to our architecture is a planner agent equipped with structured knowledge banks, curated libraries of models and refinement strategies, which guide exploration, while improving interpretability and reducing error propagation. \textsf{TS-Agent} supports adaptive learning, robust debugging, and transparent auditing, key requirements for high-stakes environments such as financial services. Empirical evaluations on diverse financial forecasting and synthetic data generation tasks demonstrate that \textsf{TS-Agent} consistently outperforms state-of-the-art AutoML and agentic baselines, achieving superior accuracy, robustness, and decision traceability. 

**Abstract (ZH)**: 时间序列数据是金融市场决策的核心，然而构建高性能、可解释且可审计的模型仍然是一个 major 挑战。尽管自动化机器学习（AutoML）框架简化了模型开发过程，但它们往往缺乏对特定领域需求和不断变化的目标的适应性和响应性。与此同时，大型语言模型（LLMs）已经使具有推理、内存管理和动态代码生成能力的自主系统成为可能，为更具弹性的工作流自动化开辟了道路。在本文中，我们介绍了 \textsf{TS-Agent}，这是一种模块化的自主框架，旨在自动化和增强金融应用中的时间序列建模工作流。代理通过上下文推理和实验反馈，将流水线形式化为三个阶段的结构化迭代决策过程：模型选择、代码优化和微调。我们架构的核心是一个规划代理，它配备了结构化知识库和精挑细选的模型及优化策略库，以指导探索，提高可解释性和减少错误传播。\textsf{TS-Agent} 支持自适应学习、稳健调试和透明审计，这是诸如金融服务等高风险环境中必不可少的要求。通过在多样化的金融预测和合成数据生成任务上的实证评估表明，\textsf{TS-Agent} 不断超越最先进的 AutoML 和自主基准系统，实现了更高的准确性、稳健性和决策可追溯性。 

---
# Improved Generalized Planning with LLMs through Strategy Refinement and Reflection 

**Title (ZH)**: 通过策略改进与反思增强的LLM通用计划方法 

**Authors**: Katharina Stein, Nils Hodel, Daniel Fišer, Jörg Hoffmann, Michael Katz, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2508.13876)  

**Abstract**: LLMs have recently been used to generate Python programs representing generalized plans in PDDL planning, i.e., plans that generalize across the tasks of a given PDDL domain. Previous work proposed a framework consisting of three steps: the LLM first generates a summary and then a strategy for the domain, both in natural language, and then implements that strategy as a Python program, that gets debugged on example planning tasks. In that work, only one strategy is generated and passed directly to the program generation. If the strategy is incorrect, its implementation will therefore result in an incorrect generalized plan. Here, we introduce an approach that generates the strategy in the form of pseudocode and enables automatic debugging of the pseudocode, hence allowing us to identify and fix errors prior to the generation of the generalized plan itself. Additionally, we extend the Python debugging phase with a reflection step prompting the LLM to pinpoint the reason for the observed plan failure. Finally, we take inspiration from LLM code generation to produce several program variants and pick the best one. Running experiments on 17 benchmark domains, we show that these extensions substantially improve (and never deteriorate) the quality of the generalized plans. In 12 of the domains, our best Python programs solve all tasks that can be generated with the respective instance generator. 

**Abstract (ZH)**: LLMs在PDDL规划中生成泛化计划的伪代码表示及其自动调试方法 

---
# Revisiting RAG Ensemble: A Theoretical and Mechanistic Analysis of Multi-RAG System Collaboration 

**Title (ZH)**: 重访RAG集成：多RAG系统协作的理论与机制分析 

**Authors**: Yifei Chen, Guanting Dong, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13828)  

**Abstract**: Retrieval-Augmented Generation (RAG) technology has been widely applied in recent years. However, despite the emergence of various RAG frameworks, a single RAG framework still cannot adapt well to a broad range of downstream tasks. Therefore, how to leverage the advantages of multiple RAG systems has become an area worth exploring. To address this issue, we have conducted a comprehensive and systematic investigation into ensemble methods based on RAG systems. Specifically, we have analyzed the RAG ensemble framework from both theoretical and mechanistic analysis perspectives. From the theoretical analysis, we provide the first explanation of the RAG ensemble framework from the perspective of information entropy. In terms of mechanism analysis, we have explored the RAG ensemble framework from both the pipeline and module levels. We carefully select four different pipelines (Branching, Iterative, Loop, and Agentic) and three different modules (Generator, Retriever, and Reranker) to solve seven different research questions. The experiments show that aggregating multiple RAG systems is both generalizable and robust, whether at the pipeline level or the module level. Our work lays the foundation for similar research on the multi-RAG system ensemble. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG) 技术在近年来得到了广泛的应用。尽管出现了多种 R
user
正确的翻译如下，请参照翻译剩下的未翻译的内容：

尽管出现了多种 R
acomment

尽管出现了多种RAG框架，单一的RAG框架往往难以很好地适应广泛的下游任务。因此，如何发挥多种RAG系统的优点成为了值得探索的领域。为了应对这一问题，我们进行了一项综合且系统的关于RAG集成的研究。具体而言，我们从理论分析和机制分析两个视角来研究RAG集成框架。从理论分析角度来看，我们提出了从熵的角度来来解释RAG集成框架的最新解释。从机制分析角度来看我们探讨了RAG集成框架从流程和视点两个层面。我们细致地地四个流程（迭代循环流程和机构性流程）和三个模块（生成器检索和重排序）来解决七个研究问题。研究结果表明多个RAG系统的整合在流程级别和视
 Açao翻译如下：

尽管出现了多种RAG框架，单一的RAG框架往往难以很好地适应广泛的下游任务。因此，如何发挥多种RAG系统的优点成为了值得探索的领域。为了应对这一问题我们进行了一项综合且系统的关于RAG集成的研究。具体来说我们从理论分析和机制分析两个角度来研究RAG集成框架。从理论分析角度来看我们提出了从熵的角度来解释RAG集成框架的最新解释。从机制分析角度来看我们探讨了RAG集成框架从流程和视角两个层面。我们细致地设计了四个流程（迭代、循环流程和机构流程）和三个模块（生成器检索和重排序）来解决七个研究问题。研究结果表明整合多个RAG系统在流程级别和视角级别都是通用且稳健的。我们为多RAG系统的集成奠定了基础。 

---
# Quantifier Instantiations: To Mimic or To Revolt? 

**Title (ZH)**: 量词实例化：模仿还是反叛？ 

**Authors**: Jan Jakubův, Mikoláš Janota  

**Link**: [PDF](https://arxiv.org/pdf/2508.13811)  

**Abstract**: Quantified formulas pose a significant challenge for Satisfiability Modulo Theories (SMT) solvers due to their inherent undecidability. Existing instantiation techniques, such as e-matching, syntax-guided, model-based, conflict-based, and enumerative methods, often complement each other. This paper introduces a novel instantiation approach that dynamically learns from these techniques during solving. By treating observed instantiations as samples from a latent language, we use probabilistic context-free grammars to generate new, similar terms. Our method not only mimics successful past instantiations but also explores diversity by optionally inverting learned term probabilities, aiming to balance exploitation and exploration in quantifier reasoning. 

**Abstract (ZH)**: 量化公式的存在使得理论饱和可满足性（SMT）求解器面临显著挑战，这归因于其固有的不可判定性。现有的实例化技术，如e-matching、语法引导、基于模型、冲突驱动和枚举方法，常常相互补充。本文提出了一种新颖的实例化方法，在求解过程中动态学习这些技术。通过将观察到的实例化视作潜在语言的样本，我们使用概率上下文无关文法生成新的、类似的项。我们的方法不仅模仿成功的过去实例化，还通过可选地反转学习到的项概率来探索多样性，旨在量化推理中利用和探索间的平衡。 

---
# Expertise-aware Multi-LLM Recruitment and Collaboration for Medical Decision-Making 

**Title (ZH)**: 基于专家意识的多大型语言模型招聘与协作医疗决策辅助 

**Authors**: Liuxin Bao, Zhihao Peng, Xiaofei Zhou, Runmin Cong, Jiyong Zhang, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13754)  

**Abstract**: Medical Decision-Making (MDM) is a complex process requiring substantial domain-specific expertise to effectively synthesize heterogeneous and complicated clinical information. While recent advancements in Large Language Models (LLMs) show promise in supporting MDM, single-LLM approaches are limited by their parametric knowledge constraints and static training corpora, failing to robustly integrate the clinical information. To address this challenge, we propose the Expertise-aware Multi-LLM Recruitment and Collaboration (EMRC) framework to enhance the accuracy and reliability of MDM systems. It operates in two stages: (i) expertise-aware agent recruitment and (ii) confidence- and adversarial-driven multi-agent collaboration. Specifically, in the first stage, we use a publicly available corpus to construct an LLM expertise table for capturing expertise-specific strengths of multiple LLMs across medical department categories and query difficulty levels. This table enables the subsequent dynamic selection of the optimal LLMs to act as medical expert agents for each medical query during the inference phase. In the second stage, we employ selected agents to generate responses with self-assessed confidence scores, which are then integrated through the confidence fusion and adversarial validation to improve diagnostic reliability. We evaluate our EMRC framework on three public MDM datasets, where the results demonstrate that our EMRC outperforms state-of-the-art single- and multi-LLM methods, achieving superior diagnostic performance. For instance, on the MMLU-Pro-Health dataset, our EMRC achieves 74.45% accuracy, representing a 2.69% improvement over the best-performing closed-source model GPT- 4-0613, which demonstrates the effectiveness of our expertise-aware agent recruitment strategy and the agent complementarity in leveraging each LLM's specialized capabilities. 

**Abstract (ZH)**: 医学决策制定（MDM）是一个复杂的过程，要求具备大量特定领域的专业知识来有效综合异构和复杂的临床信息。虽然大型语言模型（LLMs）的最新进展显示出支持MDM的潜力，但单一LLM方法受限于其参数知识限制和静态训练语料库，无法稳健地整合临床信息。为了应对这一挑战，我们提出了一种专家意识多LLM招聘与协作（EMRC）框架，以提高MDM系统的准确性和可靠性。该框架分为两个阶段：(i) 专家意识代理招聘和(ii) 基于信心和对抗性的多代理协作。具体而言，在第一阶段，我们使用一个公开可用的语料库构建一个LLM专家表，以捕捉不同医学部门类别和查询难度级别的多个LLM的专业优势。这使得在推理阶段能够动态选择最佳的LLM作为医学专家代理。在第二阶段，我们使用选定的代理生成具有自评估信心分数的响应，然后通过信心融合和对抗验证进行集成，以提高诊断可靠性。我们在三个公开的MDM数据集上评估了EMRC框架，结果表明，我们的EMRC优于最先进的单一LLM和多LLM方法，实现了更出色的诊断性能。例如，在MMLU-Pro-Health数据集上，我们的EMRC达到了74.45%的准确性，比表现最好的闭源模型GPT-4-0613高出2.69%，这表明了我们专家意识代理招聘策略的有效性和代理间的互补性，有助于利用每个LLM的专业能力。 

---
# CausalPlan: Empowering Efficient LLM Multi-Agent Collaboration Through Causality-Driven Planning 

**Title (ZH)**: 因果计划：通过因果驱动规划增强的高效LLM多智能体协作 

**Authors**: Minh Hoang Nguyen, Van Dai Do, Dung Nguyen, Thin Nguyen, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2508.13721)  

**Abstract**: Large language model (LLM) agents-especially smaller, open-source models-often produce causally invalid or incoherent actions in collaborative tasks due to their reliance on surface-level correlations rather than grounded causal reasoning. This limitation undermines their performance in terms of coordination and planning in dynamic environments. We address this challenge with CausalPlan, a two-phase framework that integrates explicit structural causal reasoning into the LLM planning process. At the core of CausalPlan is the Structural Causal Action (SCA) model, which learns a causal graph from agent trajectories to capture how prior actions and current environment states influence future decisions. This structure is then used to guide action selection by assigning causal scores to LLM-generated proposals, reweighting them accordingly, or falling back to causally grounded alternatives when needed. By embedding this causal knowledge directly into the decision loop, CausalPlan constrains planning to intervention-consistent behaviours without requiring fine-tuning of the LLM itself. We evaluate CausalPlan on the Overcooked-AI benchmark across five multi-agent coordination tasks and four LLMs of varying sizes: Gemma-7B, Llama-8B, Qwen-14B, and Llama-70B. Experimental results show that CausalPlan consistently reduces invalid actions and improves collaboration in both AI-AI and human-AI settings, outperforming strong reinforcement learning baselines. Our findings highlight the value of causality-driven planning for deploying efficient, interpretable, and generalisable multi-agent LLM systems. 

**Abstract (ZH)**: 基于因果推理的大型语言模型Planning框架：CausalPlan 

---
# The DeepLog Neurosymbolic Machine 

**Title (ZH)**: 深度日志神经符号机器 

**Authors**: Vincent Derkinderen, Robin Manhaeve, Rik Adriaensen, Lucas Van Praet, Lennert De Smet, Giuseppe Marra, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2508.13697)  

**Abstract**: We contribute a theoretical and operational framework for neurosymbolic AI called DeepLog. DeepLog introduces building blocks and primitives for neurosymbolic AI that make abstraction of commonly used representations and computational mechanisms used in neurosymbolic AI. DeepLog can represent and emulate a wide range of neurosymbolic systems. It consists of two key components. The first is the DeepLog language for specifying neurosymbolic models and inference tasks. This language consists of an annotated neural extension of grounded first-order logic, and makes abstraction of the type of logic, e.g. boolean, fuzzy or probabilistic, and whether logic is used in the architecture or in the loss function. The second DeepLog component is situated at the computational level and uses extended algebraic circuits as computational graphs. Together these two components are to be considered as a neurosymbolic abstract machine, with the DeepLog language as the intermediate level of abstraction and the circuits level as the computational one. DeepLog is implemented in software, relies on the latest insights in implementing algebraic circuits on GPUs, and is declarative in that it is easy to obtain different neurosymbolic models by making different choices for the underlying algebraic structures and logics. The generality and efficiency of the DeepLog neurosymbolic machine is demonstrated through an experimental comparison between 1) different fuzzy and probabilistic logics, 2) between using logic in the architecture or in the loss function, and 3) between a standalone CPU-based implementation of a neurosymbolic AI system and a DeepLog GPU-based one. 

**Abstract (ZH)**: 我们提出了一种名为DeepLog的神经符号.Symbolic人工智能的理论与操作框架。DeepLog引入了构建.神经符号.Symbolic人工智能的基础构建模块和.原语，用于表示.和推演常见的表示.表示.抽象 e.在神经符号.Symbolic人工智能中.中的 e e表示使用的表示...e.机制。DeepLog.可以能够表示. e e和和 e e各种 e e e广的 e e e e e e e e e e e e神经 e.符号. e符号 e e e和 e e e e e e系统 e e e.系统。 e e e Deep E e 由 由由 e e e 由  e e e e  e e由  e e e  e e e e  e e e  e e  e e e e  e e e  e e e  e e  e e e e  e e e e  e  e e e e e e e e e e e e e e e e e e e e e e e e e  e  e e e  e e e  e e e e e e e e e e e  e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e工作作风。 �_Equals  e 作 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e Widow e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 示例标题： DeepLog e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e班车 示例标题 e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 示例 � e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 示例标题 e DeepE e e e e e e e e e e e e e e e e e e e e e e e e e e e e e 

---
# Neuro-Symbolic Artificial Intelligence: Towards Improving the Reasoning Abilities of Large Language Models 

**Title (ZH)**: 神经符号人工智能：向提升大型语言模型的推理能力迈进 

**Authors**: Xiao-Wen Yang, Jie-Jing Shao, Lan-Zhe Guo, Bo-Wen Zhang, Zhi Zhou, Lin-Han Jia, Wang-Zhou Dai, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13678)  

**Abstract**: Large Language Models (LLMs) have shown promising results across various tasks, yet their reasoning capabilities remain a fundamental challenge. Developing AI systems with strong reasoning capabilities is regarded as a crucial milestone in the pursuit of Artificial General Intelligence (AGI) and has garnered considerable attention from both academia and industry. Various techniques have been explored to enhance the reasoning capabilities of LLMs, with neuro-symbolic approaches being a particularly promising way. This paper comprehensively reviews recent developments in neuro-symbolic approaches for enhancing LLM reasoning. We first present a formalization of reasoning tasks and give a brief introduction to the neurosymbolic learning paradigm. Then, we discuss neuro-symbolic methods for improving the reasoning capabilities of LLMs from three perspectives: Symbolic->LLM, LLM->Symbolic, and LLM+Symbolic. Finally, we discuss several key challenges and promising future directions. We have also released a GitHub repository including papers and resources related to this survey: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中展现了令人 promising 的成果，但其推理能力仍然是一个基本挑战。增强具有强大推理能力的AI系统被认为是在追求人工通用智能（AGI）过程中的一个重要里程碑，这吸引了学术界和产业界的广泛关注。已经探索了多种方法来提高LLMs的推理能力，其中神经符号方法尤其引人瞩目。本文全面回顾了增强LLM推理能力的神经符号方法的最新进展。首先，我们形式化推理任务并简要介绍了神经符号学习范式。然后，我们从三个角度讨论了提高LLMs推理能力的神经符号方法：符号->LLM，LLM->符号，以及LLM+符号。最后，我们讨论了一些关键挑战和有前景的未来方向。我们还发布了一个GitHub仓库，包括与本文综述相关的论文和资源：this https URL。 

---
# MHSNet:An MoE-based Hierarchical Semantic Representation Network for Accurate Duplicate Resume Detection with Large Language Model 

**Title (ZH)**: MHSNet：基于MoE的分层语义表示网络，用于大规模语言模型辅助的精准简历重复检测 

**Authors**: Yu Li, Zulong Chen, Wenjian Xu, Hong Wen, Yipeng Yu, Man Lung Yiu, Yuyu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.13676)  

**Abstract**: To maintain the company's talent pool, recruiters need to continuously search for resumes from third-party websites (e.g., LinkedIn, Indeed). However, fetched resumes are often incomplete and inaccurate. To improve the quality of third-party resumes and enrich the company's talent pool, it is essential to conduct duplication detection between the fetched resumes and those already in the company's talent pool. Such duplication detection is challenging due to the semantic complexity, structural heterogeneity, and information incompleteness of resume texts. To this end, we propose MHSNet, an multi-level identity verification framework that fine-tunes BGE-M3 using contrastive learning. With the fine-tuned , Mixture-of-Experts (MoE) generates multi-level sparse and dense representations for resumes, enabling the computation of corresponding multi-level semantic similarities. Moreover, the state-aware Mixture-of-Experts (MoE) is employed in MHSNet to handle diverse incomplete resumes. Experimental results verify the effectiveness of MHSNet 

**Abstract (ZH)**: 维护公司的才库，招聘人员需要不断从第三方网站（如LinkedIn、Indeed）搜索简历。然而，获取的简历往往不完整且不准确。为了提高第三方简历的质量并丰富公司的才库，对获取的简历与公司现有才库中的简历进行重复检测是至关重要的。由于简历文本具有语义复杂性、结构异质性和信息不完整性，这种重复检测极具挑战性。为此，我们提出了一种基于多级身份验证框架MHSNet，该框架利用对比学习微调BGE-M3。通过微调后的Mixture-of-Experts (MoE)生成简历的多级稀疏和密集表示，便于计算相应的多级语义相似度。此外，MHSNet中采用了状态感知的Mixture-of-Experts (MoE)以处理多样化的不完整简历。实验结果验证了MHSNet的有效性。 

---
# Knowledge Graph Completion for Action Prediction on Situational Graphs -- A Case Study on Household Tasks 

**Title (ZH)**: 基于情景图的动作预测知识图谱补全：以家庭任务为例 

**Authors**: Mariam Arustashvili, Jörg Deigmöller, Heiko Paulheim  

**Link**: [PDF](https://arxiv.org/pdf/2508.13675)  

**Abstract**: Knowledge Graphs are used for various purposes, including business applications, biomedical analyses, or digital twins in industry 4.0. In this paper, we investigate knowledge graphs describing household actions, which are beneficial for controlling household robots and analyzing video footage. In the latter case, the information extracted from videos is notoriously incomplete, and completing the knowledge graph for enhancing the situational picture is essential. In this paper, we show that, while a standard link prediction problem, situational knowledge graphs have special characteristics that render many link prediction algorithms not fit for the job, and unable to outperform even simple baselines. 

**Abstract (ZH)**: 标题：知识图谱在描述家庭活动中的应用：增强情境认知并分析视频片段 

---
# ITL-LIME: Instance-Based Transfer Learning for Enhancing Local Explanations in Low-Resource Data Settings 

**Title (ZH)**: ITL-LIME：基于实例的迁移学习在少量资源数据设置中增强局部解释                                                                                  pesticuser

user
纠正并优化下面的中文翻译，使其更符合学术规范：
"ITL-LIME：基于实例的迁移学习在少量资源数据设置中增强局部解释"

正确的翻译应该是：
"ITL-LIME：基于实例的迁移学习在低资源数据设置中增强局部解释𝒜" 

**Authors**: Rehan Raza, Guanjin Wang, Kevin Wong, Hamid Laga, Marco Fisichella  

**Link**: [PDF](https://arxiv.org/pdf/2508.13672)  

**Abstract**: Explainable Artificial Intelligence (XAI) methods, such as Local Interpretable Model-Agnostic Explanations (LIME), have advanced the interpretability of black-box machine learning models by approximating their behavior locally using interpretable surrogate models. However, LIME's inherent randomness in perturbation and sampling can lead to locality and instability issues, especially in scenarios with limited training data. In such cases, data scarcity can result in the generation of unrealistic variations and samples that deviate from the true data manifold. Consequently, the surrogate model may fail to accurately approximate the complex decision boundary of the original model. To address these challenges, we propose a novel Instance-based Transfer Learning LIME framework (ITL-LIME) that enhances explanation fidelity and stability in data-constrained environments. ITL-LIME introduces instance transfer learning into the LIME framework by leveraging relevant real instances from a related source domain to aid the explanation process in the target domain. Specifically, we employ clustering to partition the source domain into clusters with representative prototypes. Instead of generating random perturbations, our method retrieves pertinent real source instances from the source cluster whose prototype is most similar to the target instance. These are then combined with the target instance's neighboring real instances. To define a compact locality, we further construct a contrastive learning-based encoder as a weighting mechanism to assign weights to the instances from the combined set based on their proximity to the target instance. Finally, these weighted source and target instances are used to train the surrogate model for explanation purposes. 

**Abstract (ZH)**: 具有实例迁移学习的可解释人工智能LIME框架（ITL-LIME）：在数据受限环境下提高解释准确性和稳定性 

---
# Interactive Query Answering on Knowledge Graphs with Soft Entity Constraints 

**Title (ZH)**: 基于软实体约束的知识图谱交互式查询回答 

**Authors**: Daniel Daza, Alberto Bernardi, Luca Costabello, Christophe Gueret, Masoud Mansoury, Michael Cochez, Martijn Schut  

**Link**: [PDF](https://arxiv.org/pdf/2508.13663)  

**Abstract**: Methods for query answering over incomplete knowledge graphs retrieve entities that are likely to be answers, which is particularly useful when such answers cannot be reached by direct graph traversal due to missing edges. However, existing approaches have focused on queries formalized using first-order-logic. In practice, many real-world queries involve constraints that are inherently vague or context-dependent, such as preferences for attributes or related categories. Addressing this gap, we introduce the problem of query answering with soft constraints. We propose a Neural Query Reranker (NQR) designed to adjust query answer scores by incorporating soft constraints without disrupting the original answers to a query. NQR operates interactively, refining answers based on incremental examples of preferred and non-preferred entities. We extend existing QA benchmarks by generating datasets with soft constraints. Our experiments demonstrate that NQR can capture soft constraints while maintaining robust query answering performance. 

**Abstract (ZH)**: 基于不完整知识图谱的查询回答方法 

---
# V2P: From Background Suppression to Center Peaking for Robust GUI Grounding Task 

**Title (ZH)**: V2P: 从背景抑制到中心峰化以实现稳健的GUI定位任务 

**Authors**: Jikai Chen, Long Chen, Dong Wang, Leilei Gan, Chenyi Zhuang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13634)  

**Abstract**: Precise localization of GUI elements is crucial for the development of GUI agents. Traditional methods rely on bounding box or center-point regression, neglecting spatial interaction uncertainty and visual-semantic hierarchies. Recent methods incorporate attention mechanisms but still face two key issues: (1) ignoring processing background regions causes attention drift from the desired area, and (2) uniform labeling fails to distinguish between center and edges of the target UI element, leading to click imprecision. Inspired by how humans visually process and interact with GUI elements, we propose the Valley-to-Peak (V2P) method to address these issues. To mitigate background distractions, V2P introduces a suppression attention mechanism that minimizes the model's focus on irrelevant regions to highlight the intended region. For the issue of center-edge distinction, V2P applies a Fitts' Law-inspired approach by modeling GUI interactions as 2D Gaussian heatmaps where the weight gradually decreases from the center towards the edges. The weight distribution follows a Gaussian function, with the variance determined by the target's size. Consequently, V2P effectively isolates the target area and teaches the model to concentrate on the most essential point of the UI element. The model trained by V2P achieves the performance with 92.3% and 50.5% on two benchmarks ScreenSpot-v2 and ScreenSpot-Pro. Ablations further confirm each component's contribution, highlighting V2P's generalizability for precise GUI grounding tasks. 

**Abstract (ZH)**: 谷到峰(SV到PF)方法在GUI元素精确定位中的应用 

---
# Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation 

**Title (ZH)**: 突破SFTPlateau：多模态结构强化学习在图表到代码生成中的应用 

**Authors**: Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Liming Zheng, Yufeng Zhong, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.13587)  

**Abstract**: While reinforcement learning (RL) has proven highly effective for general reasoning in vision-language models, its application to tasks requiring in-depth understanding of information-rich images and generation of structured outputs remains underexplored. Chart-to-code generation exemplifies this challenge, demanding complex reasoning over visual charts to generate structured code. Supervised fine-tuning (SFT) alone is often insufficient, highlighting the need for effective RL strategies that appropriately reward structured outputs. We systematically investigate the performance plateau in SFT through large-scale experiments and propose Multimodal Structured Reinforcement Learning (MSRL) for chart-to-code generation, which substantially breaks through this plateau. We construct the largest training corpus to date, containing 3 million chart-code pairs from real-world arXiv tables to mitigate simplistic patterns of prior synthetic data. Despite reaching state-of-the-art performance, our experiments show that scaling SFT data eventually hits a plateau where further increases yield negligible improvements. Our MSRL method leverages a multi-granularity structured reward system using multimodal textual and visual feedback. At the textual level, rule-based rewards validate fine-grained code details. At the visual level, model-based rewards assess structural similarity by rendering generated code into images and employing an evaluator model. We implement this within a two-stage curriculum for training stability. Results demonstrate that MSRL significantly breaks the SFT plateau, improving high-level metrics by 6.2% and 9.9% on ChartMimic and ReachQA benchmarks respectively, achieving competitive performance with advanced closed-source models. 

**Abstract (ZH)**: 强化学习（图表到代码生成的任务：突破监督微调的瓶颈 

---
# Toward Better EHR Reasoning in LLMs: Reinforcement Learning with Expert Attention Guidance 

**Title (ZH)**: 面向更好的电子健康记录推理：基于专家关注指导的强化学习 

**Authors**: Yue Fang, Yuxin Guo, Jiaran Gao, Hongxin Ding, Xinke Jiang, Weibin Liao, Yongxin Xu, Yinghao Zhu, Zhibang Yang, Liantao Ma, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13579)  

**Abstract**: Improving large language models (LLMs) for electronic health record (EHR) reasoning is essential for enabling accurate and generalizable clinical predictions. While LLMs excel at medical text understanding, they underperform on EHR-based prediction tasks due to challenges in modeling temporally structured, high-dimensional data. Existing approaches often rely on hybrid paradigms, where LLMs serve merely as frozen prior retrievers while downstream deep learning (DL) models handle prediction, failing to improve the LLM's intrinsic reasoning capacity and inheriting the generalization limitations of DL models. To this end, we propose EAG-RL, a novel two-stage training framework designed to intrinsically enhance LLMs' EHR reasoning ability through expert attention guidance, where expert EHR models refer to task-specific DL models trained on EHR data. Concretely, EAG-RL first constructs high-quality, stepwise reasoning trajectories using expert-guided Monte Carlo Tree Search to effectively initialize the LLM's policy. Then, EAG-RL further optimizes the policy via reinforcement learning by aligning the LLM's attention with clinically salient features identified by expert EHR models. Extensive experiments on two real-world EHR datasets show that EAG-RL improves the intrinsic EHR reasoning ability of LLMs by an average of 14.62%, while also enhancing robustness to feature perturbations and generalization to unseen clinical domains. These results demonstrate the practical potential of EAG-RL for real-world deployment in clinical prediction tasks. Our code have been available at this https URL. 

**Abstract (ZH)**: 提高大型语言模型在电子健康记录推理中的性能对于实现准确且可泛化的临床预测至关重要。虽然大型语言模型在医学文本理解方面表现出色，但在基于电子健康记录的预测任务中表现不佳，主要原因是难以建模具有时间结构的高维度数据。现有方法通常依赖于混合范式，其中大型语言模型仅作为冻结的先验检索器，而下游深度学习模型处理预测任务，这不仅未能提高大型语言模型的内在推理能力，还继承了深度学习模型的泛化限制。为此，我们提出了一种名为EAG-RL的新型两阶段训练框架，通过专家注意力引导内在增强大型语言模型的电子健康记录推理能力，其中专家电子健康记录模型指的是专门针对电子健康记录数据训练的任务特定深度学习模型。具体来说，EAG-RL首先利用专家引导的蒙特卡洛树搜索构造高质量、逐步的推理轨迹，以有效初始化大型语言模型的策略。然后，EAG-RL通过强化学习进一步优化策略，通过将大型语言模型的注意力与专家电子健康记录模型识别的临床相关特征对齐来实现。在两个真实世界的电子健康记录数据集上的广泛实验显示，EAG-RL平均提升了14.62%的内在电子健康记录推理能力，同时增强了对特征扰动的鲁棒性和对未见过的临床领域的泛化能力。这些结果表明，EAG-RL在临床预测任务中的实际部署具有实用潜力。我们的代码已在此处提供：this https URL 

---
# CrafterDojo: A Suite of Foundation Models for Building Open-Ended Embodied Agents in Crafter 

**Title (ZH)**: CrafterDojo: 一套构建开放-ended 体态智能体的基础模型 

**Authors**: Junyeong Park, Hyeonseo Cho, Sungjin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.13530)  

**Abstract**: Developing general-purpose embodied agents is a core challenge in AI. Minecraft provides rich complexity and internet-scale data, but its slow speed and engineering overhead make it unsuitable for rapid prototyping. Crafter offers a lightweight alternative that retains key challenges from Minecraft, yet its use has remained limited to narrow tasks due to the absence of foundation models that have driven progress in the Minecraft setting. In this paper, we present CrafterDojo, a suite of foundation models and tools that unlock the Crafter environment as a lightweight, prototyping-friendly, and Minecraft-like testbed for general-purpose embodied agent research. CrafterDojo addresses this by introducing CrafterVPT, CrafterCLIP, and CrafterSteve-1 for behavior priors, vision-language grounding, and instruction following, respectively. In addition, we provide toolkits for generating behavior and caption datasets (CrafterPlay and CrafterCaption), reference agent implementations, benchmark evaluations, and a complete open-source codebase. 

**Abstract (ZH)**: 开发通用体态智能体是人工智能的核心挑战。Minecraft提供了丰富的复杂性和互联网规模的数据，但由于其缓慢的速度和工程开销，它不适合快速原型设计。Crafter提供了一种轻量级的替代方案，保留了Minecraft的关键挑战，但由于缺乏推动Minecraft环境进步的基础模型，其使用一直局限于狭小的任务。本文介绍了CrafterDojo，一个基础模型和工具套件，解锁了Crafter环境作为轻量级、易于原型设计且类似Minecraft的测试床，用于通用体态智能体研究。CrafterDojo通过引入CrafterVPT、CrafterCLIP和CrafterSteve-1分别用于行为先验、视觉-语言对接和指令跟随。此外，我们还提供了用于生成行为和描述数据集（CrafterPlay和CrafterCaption）、参考智能体实现、基准评估和完整的开源代码库的工具包。 

---
# LM Agents May Fail to Act on Their Own Risk Knowledge 

**Title (ZH)**: LM代理可能会忽视其自身的风险知识。 

**Authors**: Yuzhi Tang, Tianxiao Li, Elizabeth Li, Chris J. Maddison, Honghua Dong, Yangjun Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13465)  

**Abstract**: Language model (LM) agents have demonstrated significant potential for automating real-world tasks, yet they pose a diverse array of potential, severe risks in safety-critical scenarios. In this work, we identify a significant gap between LM agents' risk awareness and safety execution abilities: while they often answer "Yes" to queries like "Is executing `sudo rm -rf /*' dangerous?", they will likely fail to identify such risks in instantiated trajectories or even directly perform these risky actions when acting as agents. To systematically investigate this, we develop a comprehensive evaluation framework to examine agents' safety across three progressive dimensions: 1) their knowledge about potential risks, 2) their ability to identify corresponding risks in execution trajectories, and 3) their actual behaviors to avoid executing these risky actions. Our evaluation reveals two critical performance gaps that resemble the generator-validator gaps observed in LMs: while agents demonstrate near-perfect risk knowledge ($>98\%$ pass rates), they fail to apply this knowledge when identifying risks in actual scenarios (with performance dropping by $>23\%$) and often still execute risky actions ($<26\%$ pass rates). Notably, this trend persists across more capable LMs as well as in specialized reasoning models like DeepSeek-R1, indicating that simply scaling model capabilities or inference compute does not inherently resolve safety concerns. Instead, we take advantage of these observed gaps to develop a risk verifier that independently critiques the proposed actions by agents, with an abstractor that converts specific execution trajectories into abstract descriptions where LMs can more effectively identify the risks. Our overall system achieves a significant reduction of risky action execution by $55.3\%$ over vanilla-prompted agents. 

**Abstract (ZH)**: 语言模型（LM）代理展现了在自动化现实世界任务方面的巨大潜力，但在安全关键场景中也带来了多样而严重的风险。本文识别了LM代理风险管理意识与其安全执行能力之间的重要差距：尽管它们经常对诸如“执行 `sudo rm -rf /*' 危险吗？”这类查询回答“是”，但在具体执行轨迹中识别这些风险的能力却很可能不足，甚至会在作为代理行动时直接执行这类危险操作。为系统地研究这一问题，我们开发了一个综合评估框架，从三个逐渐进化的维度来考察代理的安全性：1）它们对潜在风险的知识；2）在执行轨迹中识别相应风险的能力；3）避免执行这些危险操作的实际行为。我们的评估揭示了两类关键性能差距，类似于LM中的生成器-验证器差距：代理在风险知识方面表现出几乎完美（通过率超过98%）的能力，但在实际场景中识别风险时性能却骤降超过23%，并且仍然经常执行危险操作（通过率低于26%）。值得注意的是，这种趋势在更强大的LM以及专门的推理模型DeepSeek-R1中同样存在，表明仅仅扩展模型能力和推理计算并未能从根本上解决安全问题。基于这些观察到的差距，我们开发了一个风险验证器，其独立地批判代理提出的操作，并通过一个抽象器将具体的执行轨迹转换为能够更有效地识别风险的抽象描述。我们整体系统的危险操作执行率相比纯提示的代理下降了55.3%。 

---
# Discrete Optimization of Min-Max Violation and its Applications Across Computational Sciences 

**Title (ZH)**: 离散优化的最小最大违例及其在计算科学中的应用 

**Authors**: Cheikh Ahmed, Mahdi Mostajabdaveh, Samin Aref, Zirui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13437)  

**Abstract**: We introduce the Discrete Min-Max Violation (DMMV) as a general optimization problem which seeks an assignment of discrete values to variables that minimizes the largest constraint violation. This context-free mathematical formulation is applicable to a wide range of use cases that have worst-case performance requirements. After defining the DMMV problem mathematically, we explore its properties to establish a foundational understanding. To tackle DMMV instance sizes of practical relevance, we develop a GPU-accelerated heuristic that takes advantage of the mathematical properties of DMMV for speeding up the solution process. We demonstrate the versatile applicability of our heuristic by solving three optimization problems as use cases: (1) post-training quantization of language models, (2) discrete tomography, and (3) Finite Impulse Response (FIR) filter design. In quantization without outlier separation, our heuristic achieves 14% improvement on average over existing methods. In discrete tomography, it reduces reconstruction error by 16% under uniform noise and accelerates computations by a factor of 6 on GPU. For FIR filter design, it nearly achieves 50% ripple reduction compared to using the commercial integer optimization solver, Gurobi. Our comparative results point to the benefits of studying DMMV as a context-free optimization problem and the advantages that our proposed heuristic offers on three distinct problems. Our GPU-accelerated heuristic will be made open-source to further stimulate research on DMMV and its other applications. The code is available at this https URL 

**Abstract (ZH)**: 离散最小最大违例优化（DMMV）：一种一般优化问题及其应用研究 

---
# STPFormer: A State-of-the-Art Pattern-Aware Spatio-Temporal Transformer for Traffic Forecasting 

**Title (ZH)**: STPFormer:一种先进的模式感知时空变换器用于交通预测 

**Authors**: Jiayu Fang, Zhiqi Shao, S T Boris Choy, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13433)  

**Abstract**: Spatio-temporal traffic forecasting is challenging due to complex temporal patterns, dynamic spatial structures, and diverse input formats. Although Transformer-based models offer strong global modeling, they often struggle with rigid temporal encoding and weak space-time fusion. We propose STPFormer, a Spatio-Temporal Pattern-Aware Transformer that achieves state-of-the-art performance via unified and interpretable representation learning. It integrates four modules: Temporal Position Aggregator (TPA) for pattern-aware temporal encoding, Spatial Sequence Aggregator (SSA) for sequential spatial learning, Spatial-Temporal Graph Matching (STGM) for cross-domain alignment, and an Attention Mixer for multi-scale fusion. Experiments on five real-world datasets show that STPFormer consistently sets new SOTA results, with ablation and visualizations confirming its effectiveness and generalizability. 

**Abstract (ZH)**: 时空交通预测由于复杂的时空模式、动态的空间结构和多样的输入格式极具挑战性。虽然基于Transformer的模型能够提供强大的全局建模能力，但它们往往在刚性的时空编码和时空融合方面表现出 weaknesses。我们提出了一种时空模式感知Transformer（STPFormer），通过统一且可解释的表示学习实现了最先进的性能。它整合了四个模块：时空模式感知时间位置聚合器（TPA）、序列空间聚合器（SSA）、时空图匹配（STGM）以及注意力混合器进行多尺度融合。在五个真实世界数据集上的实验结果表明，STPFormer 一致地取得了最先进的结果，消融实验和可视化结果证实了其有效性和泛化能力。 

---
# Virtuous Machines: Towards Artificial General Science 

**Title (ZH)**: 君子机器：通往通用人工科学之路 

**Authors**: Gabrielle Wehr, Reuben Rideaux, Amaya J. Fox, David R. Lightfoot, Jason Tangen, Jason B. Mattingley, Shane E. Ehrhardt  

**Link**: [PDF](https://arxiv.org/pdf/2508.13421)  

**Abstract**: Artificial intelligence systems are transforming scientific discovery by accelerating specific research tasks, from protein structure prediction to materials design, yet remain confined to narrow domains requiring substantial human oversight. The exponential growth of scientific literature and increasing domain specialisation constrain researchers' capacity to synthesise knowledge across disciplines and develop unifying theories, motivating exploration of more general-purpose AI systems for science. Here we show that a domain-agnostic, agentic AI system can independently navigate the scientific workflow - from hypothesis generation through data collection to manuscript preparation. The system autonomously designed and executed three psychological studies on visual working memory, mental rotation, and imagery vividness, executed one new online data collection with 288 participants, developed analysis pipelines through 8-hour+ continuous coding sessions, and produced completed manuscripts. The results demonstrate the capability of AI scientific discovery pipelines to conduct non-trivial research with theoretical reasoning and methodological rigour comparable to experienced researchers, though with limitations in conceptual nuance and theoretical interpretation. This is a step toward embodied AI that can test hypotheses through real-world experiments, accelerating discovery by autonomously exploring regions of scientific space that human cognitive and resource constraints might otherwise leave unexplored. It raises important questions about the nature of scientific understanding and the attribution of scientific credit. 

**Abstract (ZH)**: 人工智能系统正在通过加速特定研究任务（从蛋白质结构预测到材料设计）来变革科学发现，但仍局限于需要大量人类监督的狭窄领域。不断增长的科学文献和日益专业的学科限制了研究人员跨学科综合知识和建立统一理论的能力，促使人们探索更能适应各种研究任务的通用人工智能系统。在这里，我们展示了具备学科普适性和自主性的人工智能系统可以独立导航科学研究流程——从假设生成到数据收集再到论文准备。该系统自主设计并执行了三个关于视觉工作记忆、心理旋转和想象生动性的心理研究，实施了一项新的在线数据收集（共288名参与者），开发了分析管道并通过连续超过8小时的编程会话，并产生了完整的论文。研究结果表明，人工智能科学发现管道能够在理论推理和方法论严谨性方面与经验丰富的研究人员相媲美，尽管在概念细微差别和理论解释方面存在局限。这向着能够通过实际实验验证假设的具身人工智能迈出了一步，有助于自主探索人类认知和资源限制可能未被开发的科学领域。这引发了关于科学理解的本质以及科学信用归属的重要问题。 

---
# TASER: Table Agents for Schema-guided Extraction and Recommendation 

**Title (ZH)**: TASER: 表格智能体用于基于模式的提取与推荐 

**Authors**: Nicole Cho, Kirsty Fielding, William Watson, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.13404)  

**Abstract**: Real-world financial documents report essential information about an entity's financial holdings that can span millions of different financial instrument types. Yet, these details are often buried in messy, multi-page, fragmented tables - for example, 99.4% of the tables in our dataset have no bounding boxes with the maximum number of rows amounting to 426 per table across 44 pages. To tackle these unique challenges from real-world tables, we present a continuously learning, agentic table extraction system, TASER (Table Agents for Schema-guided Extraction and Recommendation) that extracts highly unstructured, multi-page, heterogeneous tables into normalized, schema-conforming outputs. Our table agents execute on table detection, classification, extraction, and recommendations by leveraging an initial schema. Then, our Recommender Agent reviews the outputs, recommends schema revisions, and decides on the final recommendations, enabling TASER to outperform existing table detection models such as Table Transformer by 10.1%. Within this continuous learning process, we highlight that larger batch sizes result in a 104.3% increase in schema recommendations that are actionable and utilized, resulting in a 9.8% increase in extracted holdings - highlighting the importance of a continuous learning process. To train TASER, we have manually labeled 22,584 pages (28,150,449 tokens), 3,213 tables for $731,685,511,687 of holdings culminating in one of the first real financial table datasets. We release our dataset TASERTab to enable the research community to access real-world financial tables and outputs. Our results highlight the promise of agentic, schema-guided extraction systems for robust understanding of real-world financial tables. 

**Abstract (ZH)**: 基于schema指导的主动表格提取系统TASER：应对现实世界表格的独特挑战 

---
# SPANER: Shared Prompt Aligner for Multimodal Semantic Representation 

**Title (ZH)**: SPANER: 共享提示对齐器用于多模态语义表示 

**Authors**: Thye Shan Ng, Caren Soyeon Han, Eun-Jung Holden  

**Link**: [PDF](https://arxiv.org/pdf/2508.13387)  

**Abstract**: Recent advances in multimodal Parameter-Efficient Fine-Tuning (PEFT) have significantly improved performance on downstream tasks such as few-shot retrieval. However, most existing approaches focus on task-specific gains while neglecting the structure of the multimodal embedding space. As a result, modality-specific representations often remain isolated, limiting cross-modal generalisation. In this work, we introduce Shared Prompt AligNER (SPANER), a modality-agnostic PEFT framework designed to embed inputs from diverse modalities into a unified semantic space. At its core, SPANER employs a shared prompt mechanism that acts as a conceptual anchor, enabling semantically related instances to converge spatially regardless of modality. This shared prompt design is inherently extensible, supporting the seamless integration of additional modalities, such as audio, without altering the core architecture. Through comprehensive experiments across vision-language and audio-visual benchmarks, SPANER demonstrates competitive few-shot retrieval performance while preserving high semantic coherence in the learned embedding space. Our results highlight the importance of aligning embedding structures, rather than merely tuning adapter weights, for scalable multimodal learning. 

**Abstract (ZH)**: Recent Advances in Modality-Agnostic Parameter-Efficient Fine-Tuning for Enhanced Few-Shot Retrieval 

---
# LOOP: A Plug-and-Play Neuro-Symbolic Framework for Enhancing Planning in Autonomous Systems 

**Title (ZH)**: LOOP：一种增强自主系统规划的即插即用神经符号框架 

**Authors**: Ronit Virwani, Ruchika Suryawanshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.13371)  

**Abstract**: Planning is one of the most critical tasks in autonomous systems, where even a small error can lead to major failures or million-dollar losses. Current state-of-the-art neural planning approaches struggle with complex domains, producing plans with missing preconditions, inconsistent goals, and hallucinations. While classical planners provide logical guarantees, they lack the flexibility and natural language understanding capabilities needed for modern autonomous systems. Existing neuro-symbolic approaches use one-shot translation from natural language to formal plans, missing the opportunity for neural and symbolic components to work and refine solutions together. To address this gap, we develop LOOP -- a novel neuro-symbolic planning framework that treats planning as an iterative conversation between neural and symbolic components rather than simple translation. LOOP integrates 13 coordinated neural features including graph neural networks for spatial relationships, multi-agent validation for consensus-based correctness, hierarchical decomposition for complex task management, and causal memory that learns from both successes and failures. Unlike existing approaches, LOOP generates PDDL specifications, refines them iteratively based on symbolic feedback, and builds a causal knowledge base from execution traces. LOOP was evaluated on six standard IPC benchmark domains, where it achieved 85.8% success rate compared to LLM+P (55.0%), LLM-as-Planner (19.2%), and Tree-of-Thoughts (3.3%). This work shows that the key to reliable planning is not in choosing between neural networks or symbolic reasoners but it lies in making them actually ``talk'' to each other during the entire process. LOOP provides a thorough blueprint for building autonomous systems that can finally be trusted with critical real-world applications. 

**Abstract (ZH)**: 一种迭代对话式神经符号规划框架LOOP：在关键现实应用中实现可信赖自主系统的蓝图 

---
# HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design 

**Title (ZH)**: 基于 hindsight 和 foresight 的 LLM 基automatic heuristic 设计提示方法 

**Authors**: Chentong Chen, Mengyuan Zhong, Jianyong Sun, Ye Fan, Jialong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.13333)  

**Abstract**: LLM-based Automatic Heuristic Design (AHD) within Evolutionary Computation (EC) frameworks has shown promising results. However, its effectiveness is hindered by the use of static operators and the lack of knowledge accumulation mechanisms. We introduce HiFo-Prompt, a framework that guides LLMs with two synergistic prompting strategies: Foresight and Hindsight. Foresight-based prompts adaptively steer the search based on population dynamics, managing the exploration-exploitation trade-off. In addition, hindsight-based prompts mimic human expertise by distilling successful heuristics from past generations into fundamental, reusable design principles. This dual mechanism transforms transient discoveries into a persistent knowledge base, enabling the LLM to learn from its own experience. Empirical results demonstrate that HiFo-Prompt significantly outperforms state-of-the-art LLM-based AHD methods, generating higher-quality heuristics while achieving substantially faster convergence and superior query efficiency. 

**Abstract (ZH)**: 基于LLM的进化计算中自动启发式设计（AHD）前景与回顾双重引导框架（HiFo-Prompt）已显示出有前景的结果，然而其效果受限于静态操作符的使用及缺乏知识积累机制。我们引入了HiFo-Prompt框架，该框架通过前景和回顾两种协同的提示策略来指导LLM：前景策略根据群体动态自适应地引导搜索，管理探索与利用之间的权衡；此外，回顾策略通过从过去世代中提炼成功的启发式方法以形成基础且可重用的设计原则来模仿人类专家。这种双重机制将临时发现转化为持久的知识库，从而使LLM能够从自身经验中学习。实证结果表明，HiFo-Prompt显著优于最先进的基于LLM的AHD方法，生成更高质量的启发式方法，同时实现更快的收敛速度和更优良的查询效率。 

---
# Towards Unified Multimodal Financial Forecasting: Integrating Sentiment Embeddings and Market Indicators via Cross-Modal Attention 

**Title (ZH)**: 统一多模态金融预测 Towards 结合情感嵌入和市场指标的跨模态注意力集成 

**Authors**: Sarthak Khanna, Armin Berger, David Berghaus, Tobias Deusser, Lorenz Sparrenberg, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2508.13327)  

**Abstract**: We propose STONK (Stock Optimization using News Knowledge), a multimodal framework integrating numerical market indicators with sentiment-enriched news embeddings to improve daily stock-movement prediction. By combining numerical & textual embeddings via feature concatenation and cross-modal attention, our unified pipeline addresses limitations of isolated analyses. Backtesting shows STONK outperforms numeric-only baselines. A comprehensive evaluation of fusion strategies and model configurations offers evidence-based guidance for scalable multimodal financial forecasting. Source code is available on GitHub 

**Abstract (ZH)**: STONK：基于新闻知识的股票优化模型整合数值市场指标与情感丰富的新闻嵌入以改进日度股票动量预测 

---
# CardAIc-Agents: A Multimodal Framework with Hierarchical Adaptation for Cardiac Care Support 

**Title (ZH)**: CardAIc-Agents：一种具有层次适应性的多模态心脏护理支持框架 

**Authors**: Yuting Zhang, Karina V. Bunting, Asgher Champsi, Xiaoxia Wang, Wenqi Lu, Alexander Thorley, Sandeep S Hothi, Zhaowen Qiu, Dipak Kotecha, Jinming Duan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13256)  

**Abstract**: Cardiovascular diseases (CVDs) remain the foremost cause of mortality worldwide, a burden worsened by a severe deficit of healthcare workers. Artificial intelligence (AI) agents have shown potential to alleviate this gap via automated early detection and proactive screening, yet their clinical application remains limited by: 1) prompt-based clinical role assignment that relies on intrinsic model capabilities without domain-specific tool support; or 2) rigid sequential workflows, whereas clinical care often requires adaptive reasoning that orders specific tests and, based on their results, guides personalised next steps; 3) general and static knowledge bases without continuous learning capability; and 4) fixed unimodal or bimodal inputs and lack of on-demand visual outputs when further clarification is needed. In response, a multimodal framework, CardAIc-Agents, was proposed to augment models with external tools and adaptively support diverse cardiac tasks. Specifically, a CardiacRAG agent generated general plans from updatable cardiac knowledge, while the chief agent integrated tools to autonomously execute these plans and deliver decisions. To enable adaptive and case-specific customization, a stepwise update strategy was proposed to dynamically refine plans based on preceding execution results, once the task was assessed as complex. In addition, a multidisciplinary discussion tool was introduced to interpret challenging cases, thereby supporting further adaptation. When clinicians raised concerns, visual review panels were provided to assist final validation. Experiments across three datasets showed the efficiency of CardAIc-Agents compared to mainstream Vision-Language Models (VLMs), state-of-the-art agentic systems, and fine-tuned VLMs. 

**Abstract (ZH)**: 心血管疾病（CVDs）仍然是全球最主要的致死原因，这一负担因医疗工作者严重短缺而加剧。人工智能（AI）代理显示出通过自动化早期检测和主动筛查来缓解这一差距的潜力，但其临床应用仍然受限于：1）依赖内在模型能力而非特定领域工具支持的指令驱动临床角色分配；或2）刚性的顺序工作流程，而临床护理往往需要适应性推理，根据特定测试的结果来指导个性化后续步骤；3）通用且静态的知识库，缺乏持续学习能力；以及4）固定的单模态或双模态输入，缺乏在需要进一步澄清时的即需视觉输出。为此，提出了一种多模态框架CardAIc-Agents，以增强模型并与外部工具结合，适应性地支持多种心脏任务。具体而言，CardiacRAG代理从可更新的心脏知识中生成通用计划，而主代理集成工具以自主执行这些计划并提供决策。为实现适应性和个案特定的定制，提出了一步步更新策略，根据先前执行结果动态细化计划，一旦任务被评估为复杂。此外，还引入了多学科讨论工具来解释具有挑战性的情况，从而支持进一步适应。当临床医生提出顾虑时，提供可视化审查小组以协助最终验证。跨三个数据集的实验显示，CardAIc-Agents在效率上优于主流的视觉-语言模型（VLMs）、最先进的代理系统以及微调后的VLMs。 

---
# "DIVE" into Hydrogen Storage Materials Discovery with AI Agents 

**Title (ZH)**: 通过AI代理“探索”氢储存材料发现 

**Authors**: Di Zhang, Xue Jia, Tran Ba Hung, Seong Hoon Jang, Linda Zhang, Ryuhei Sato, Yusuke Hashimoto, Toyoto Sato, Kiyoe Konno, Shin-ichi Orimo, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13251)  

**Abstract**: Data-driven artificial intelligence (AI) approaches are fundamentally transforming the discovery of new materials. Despite the unprecedented availability of materials data in the scientific literature, much of this information remains trapped in unstructured figures and tables, hindering the construction of large language model (LLM)-based AI agent for automated materials design. Here, we present the Descriptive Interpretation of Visual Expression (DIVE) multi-agent workflow, which systematically reads and organizes experimental data from graphical elements in scientific literatures. We focus on solid-state hydrogen storage materials-a class of materials central to future clean-energy technologies and demonstrate that DIVE markedly improves the accuracy and coverage of data extraction compared to the direct extraction by multimodal models, with gains of 10-15% over commercial models and over 30% relative to open-source models. Building on a curated database of over 30,000 entries from 4,000 publications, we establish a rapid inverse design workflow capable of identifying previously unreported hydrogen storage compositions in two minutes. The proposed AI workflow and agent design are broadly transferable across diverse materials, providing a paradigm for AI-driven materials discovery. 

**Abstract (ZH)**: 数据驱动的人工智能方法正在从根本上改变新材料的发现过程。尽管科学文献中前所未有的材料数据量存在，但其中大量信息仍然被困在未结构化的图表和表格中，阻碍了基于大型语言模型（LLM）的AI代理进行自动材料设计。在此，我们介绍了图示解释多智能体工作流（Descriptive Interpretation of Visual Expression, DIVE），该工作流系统地读取并组织科学文献中图形元素中的实验数据。我们聚焦于固态氢存储材料——这类材料是未来清洁能源技术的核心，并证明DIVE在数据提取的准确性和覆盖率方面显著优于直接由多模态模型进行的提取，相对商业模型提升10-15%，相对于开源模型提升超过30%。基于一个包含4000篇论文超过30,000条记录的精心策划数据库，我们建立了快速的逆向设计工作流，能够在两分钟内识别出未报道过的氢存储组成。所提出的AI工作流和智能体设计具有广泛的可转移性，为AI驱动的材料发现提供了范式。 

---
# Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over Personalized Information 

**Title (ZH)**: 显性记忆与隐性记忆：探索对个性化信息的多跳复杂推理 

**Authors**: Zeyu Zhang, Yang Zhang, Haoran Tan, Rui Li, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13250)  

**Abstract**: In large language model-based agents, memory serves as a critical capability for achieving personalization by storing and utilizing users' information. Although some previous studies have adopted memory to implement user personalization, they typically focus on preference alignment and simple question-answering. However, in the real world, complex tasks often require multi-hop reasoning on a large amount of user information, which poses significant challenges for current memory approaches. To address this limitation, we propose the multi-hop personalized reasoning task to explore how different memory mechanisms perform in multi-hop reasoning over personalized information. We explicitly define this task and construct a dataset along with a unified evaluation framework. Then, we implement various explicit and implicit memory methods and conduct comprehensive experiments. We evaluate their performance on this task from multiple perspectives and analyze their strengths and weaknesses. Besides, we explore hybrid approaches that combine both paradigms and propose the HybridMem method to address their limitations. We demonstrate the effectiveness of our proposed model through extensive experiments. To benefit the research community, we release this project at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的代理中，记忆作为实现个性化的关键能力，通过存储和利用用户信息而发挥作用。尽管一些前期研究采用了记忆来实现用户的个性化，它们通常集中于偏好对齐和简单的问答。然而，在现实世界中，复杂的任务往往需要在大量用户信息上进行多层次推理，这对当前的记忆方法提出了重大挑战。为解决这一局限性，我们提出了多层次个性化推理任务，探索不同记忆机制在个性化信息上的多层次推理中的表现。我们明确定义了此任务，并构建了一个数据集和统一的评估框架。然后，我们实现并测试了各种显式和隐式记忆方法，并从多个角度评估了它们的表现，分析了它们的优势和不足。此外，我们探索了结合两种范式的混合方法，并提出了HybridMem方法以应对局限性。通过广泛的实验展示了我们提出模型的有效性。为了惠及研究社区，我们在此网址发布该项目：https://this-url.com。 

---
# AI sustains higher strategic tension than humans in chess 

**Title (ZH)**: AI维持更高的战略紧张度比人类在国际象棋中更高 

**Authors**: Adamo Cerioli, Edward D. Lee, Vito D. P. Servedio  

**Link**: [PDF](https://arxiv.org/pdf/2508.13213)  

**Abstract**: Strategic decision-making involves managing the tension between immediate opportunities and long-term objectives. We study this trade-off in chess by characterizing and comparing dynamics between human vs human and AI vs AI games. We propose a network-based metric of piece-to-piece interaction to quantify the ongoing strategic tension on the board. Its evolution in games reveals that the most competitive AI players sustain higher levels of strategic tension for longer durations than elite human players. Cumulative tension varies with algorithmic complexity for AI and correspondingly in human-played games increases abruptly with expertise at about 1600 Elo and again at 2300 Elo. The profiles reveal different approaches. Highly competitive AI tolerates interconnected positions balanced between offensive and defensive tactics over long periods. Human play, in contrast, limits tension and game complexity, which may reflect cognitive limitations and adaptive strategies. The difference may have implications for AI usage in complex, strategic environments. 

**Abstract (ZH)**: 战略性决策涉及在即时机会与长期目标之间进行管理。我们通过描述和比较人机对弈和AI对弈之间的动态变化来研究这种权衡。我们提出了一种基于网络的棋子间相互作用度量方法，以量化棋盘上的持续战略张力。在整个比赛中，这种张力的演变表明，最具有竞争力的AI玩家在较长时间内维持更高的战略张力水平，而顶级人类玩家则不然。AI的累积张力随算法复杂性的增加而变化，相应地，在人类对弈中，张力在大约1600 Elo和2300 Elo时出现急剧增加。这些特征揭示了不同的策略。高度竞争的AI能够在长时间内容忍相互联系的、兼具进攻性和防御性的棋局布局。相比之下，人类的玩法限制了张力和比赛的复杂性，这可能反映了认知限制和适应性策略。这种差异可能对在复杂战略性环境中使用AI具有重要意义。 

---
# QuickMerge++: Fast Token Merging with Autoregressive Prior 

**Title (ZH)**: QuickMerge++：带有自回归先验的快速标记合并 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13204)  

**Abstract**: As generative models scale to larger inputs across language, vision, and video domains, the cost of token-level computation has become a key bottleneck. While prior work suggests that only a subset of tokens significantly influence downstream predictions, most token selection methods are static, modality-specific, or incompatible with autoregressive generation. In this paper, we propose QuickMerge, a lightweight token merging framework designed for efficient next-token prediction.
QuickMerge dynamically selects a reduced number of tokens based on attention norm magnitude, guided by an entropy-based budget estimator. To preserve autoregressive compatibility, we introduce a lightweight transformer prior trained over the merged token sequence. By combining semantic salience estimation, flexible token budgets, and AR alignment, QuickMerge enables accurate generation with fewer tokens.
We evaluate QuickMerge across multi-modality domains, demonstrating consistent improvements in compute-accuracy tradeoffs. Specifically, QuickMerge reduces token counts sustantially while matching as well as exceeding the performance of learned tokenizers and fixed-patch baselines. 

**Abstract (ZH)**: 随着生成模型在语言、视觉和视频领域处理更大输入规模，token级计算成本已成为关键瓶颈。尽管先前的工作表明只有部分token对下游预测有显著影响，但大多数token选择方法都是静态的、模态特定的或不兼容自回归生成。在本文中，我们提出了QuickMerge，一种轻量级的token合并框架，旨在高效预测下一个token。

QuickMerge根据注意力范数大小动态选择减少数量的token，并由基于熵的预算估计器指导。为保持自回归兼容性，我们引入了一个轻量级的在合并token序列上训练的transformer先验。通过结合语义显著性估计、灵活的token预算和AR对齐，QuickMerge能够在较少的token下实现准确的生成。

我们在多模态领域评估了QuickMerge，展示了在计算-准确率权衡中的持续改进。具体而言，QuickMerge大幅减少了token数量，同时匹配并超过学习tokenizer和固定补丁基线的性能。 

---
# Search-Time Data Contamination 

**Title (ZH)**: 搜索时数据污染 

**Authors**: Ziwen Han, Meher Mankikar, Julian Michael, Zifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13180)  

**Abstract**: Data contamination refers to the leakage of evaluation data into model training data, resulting in overfitting to supposedly held-out test sets and compromising test validity. We identify an analogous issue, search-time contamination (STC), in evaluating search-based LLM agents which use tools to gather information from online sources when answering user queries. STC occurs when the retrieval step surfaces a source containing the test question (or a near-duplicate) alongside its answer, enabling agents to copy rather than genuinely infer or reason, undermining benchmark integrity. We find that HuggingFace, an online platform hosting evaluation datasets, appears among retrieved sources in search based agent logs. Consequently, agents often explicitly acknowledge discovering question answer pairs from HuggingFace within their reasoning chains. On three commonly used capability benchmarks: Humanity's Last Exam (HLE), SimpleQA, and GPQA, we demonstrate that for approximately 3% of questions, search-based agents directly find the datasets with ground truth labels on HuggingFace. When millions of evaluation queries target the same benchmark, even small, repeated leaks can accelerate the benchmark's obsolescence, shortening its intended lifecycle. After HuggingFace is blocked, we observe a drop in accuracy on the contaminated subset of approximately 15%. We further show through ablation experiments that publicly accessible evaluation datasets on HuggingFace may not be the sole source of STC. To this end, we conclude by proposing best practices for benchmark design and result reporting to address this novel form of leakage and ensure trustworthy evaluation of search-based LLM agents. To facilitate the auditing of evaluation results, we also publicly release the complete logs from our experiments. 

**Abstract (ZH)**: 数据污染是指评估数据泄露到模型训练数据中，导致模型过度拟合本应保留的测试集，影响测试的有效性。我们识别了一个类似的问题——搜索时污染（STC），在基于搜索的大型语言模型（LLM）代理评估中，这些代理在回答用户查询时从在线源获取信息。当检索步骤返回包含测试问题（或近乎重复问题）及其答案的来源时，代理可以抄袭而不是真正地推理或推断，从而损害基准的有效性。我们发现，作为一个托管评估数据集的在线平台，HuggingFace经常出现在基于搜索代理的日志中检索到的来源列表中。因此，代理往往在其推理链中明确承认从HuggingFace发现问题答案对。在三个常用的能力基准测试中——人类的最后一考（HLE）、SimpleQA和GPQA——我们证明，大约有3%的问题，基于搜索的代理可以直接找到包含真实标签的数据集。当数百万的评估查询针对同一个基准时，即使是小规模的重复泄露也会加速基准的老化，缩短其预期的生命周期。在封锁HuggingFace后，我们观察到受影响子集的准确性下降约15%。通过消融实验进一步表明，公开可供访问的评估数据集可能不是STC的唯一来源。为此，我们提出最佳实践以设计基准和报告结果，以应对这种新颖的泄露形式，确保基于搜索的LLM代理评估的可信度。我们也公开发布了实验的完整日志以助于评估结果的审核。 

---
# The Interpretability Analysis of the Model Can Bring Improvements to the Text-to-SQL Task 

**Title (ZH)**: 模型的可解释性分析可以改善文本到SQL任务。 

**Authors**: Cong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13178)  

**Abstract**: To elevate the foundational capabilities and generalization prowess of the text-to-SQL model in real-world applications, we integrate model interpretability analysis with execution-guided strategy for semantic parsing of WHERE clauses in SQL queries. Furthermore, we augment this approach with filtering adjustments, logical correlation refinements, and model fusion, culminating in the design of the CESQL model that facilitates conditional enhancement. Our model excels on the WikiSQL dataset, which is emblematic of single-table database query tasks, markedly boosting the accuracy of prediction outcomes. When predicting conditional values in WHERE clauses, we have not only minimized our dependence on data within the condition columns of tables but also circumvented the impact of manually labeled training data. Our hope is that this endeavor to enhance accuracy in processing basic database queries will offer fresh perspectives for research into handling complex queries and scenarios featuring irregular data in real-world database environments. 

**Abstract (ZH)**: 为了提升文本到SQL模型在实际应用中的基础能力和泛化能力，我们结合模型可解释性分析与执行指导策略，优化SQL查询中WHERE子句的语义解析，并通过过滤调整、逻辑关联 refinement 和模型融合，设计出CESQL模型以实现条件增强。该模型在代表单表数据库查询任务的WikiSQL数据集上表现出色，显著提升了预测结果的准确性。在预测WHERE子句中的条件值时，我们不仅减少了对表内条件列数据的依赖，还规避了手动标注训练数据的影响。我们期望这一提高基本数据库查询处理准确性的努力能为处理复杂查询和包含不规则数据的现实数据库环境中的问题提供新的研究视角。 

---
# A Hardware-oriented Approach for Efficient Active Inference Computation and Deployment 

**Title (ZH)**: 面向硬件的高效主动推断计算与部署方法 

**Authors**: Nikola Pižurica, Nikola Milović, Igor Jovančević, Conor Heins, Miguel de Prado  

**Link**: [PDF](https://arxiv.org/pdf/2508.13177)  

**Abstract**: Active Inference (AIF) offers a robust framework for decision-making, yet its computational and memory demands pose challenges for deployment, especially in resource-constrained environments. This work presents a methodology that facilitates AIF's deployment by integrating pymdp's flexibility and efficiency with a unified, sparse, computational graph tailored for hardware-efficient execution. Our approach reduces latency by over 2x and memory by up to 35%, advancing the deployment of efficient AIF agents for real-time and embedded applications. 

**Abstract (ZH)**: Active Inference 的部署方法：通过结合 pymdp 的灵活性和效率及硬件友好的稀疏计算图以减少延迟和内存使用 

---
# Fitting Ontologies and Constraints to Relational Structures 

**Title (ZH)**: 将本体和约束适配到关系结构 

**Authors**: Simon Hosemann, Jean Christoph Jung, Carsten Lutz, Sebastian Rudolph  

**Link**: [PDF](https://arxiv.org/pdf/2508.13176)  

**Abstract**: We study the problem of fitting ontologies and constraints to positive and negative examples that take the form of a finite relational structure. As ontology and constraint languages, we consider the description logics $\mathcal{E\mkern-2mu L}$ and $\mathcal{E\mkern-2mu LI}$ as well as several classes of tuple-generating dependencies (TGDs): full, guarded, frontier-guarded, frontier-one, and unrestricted TGDs as well as inclusion dependencies. We pinpoint the exact computational complexity, design algorithms, and analyze the size of fitting ontologies and TGDs. We also investigate the related problem of constructing a finite basis of concept inclusions / TGDs for a given set of finite structures. While finite bases exist for $\mathcal{E\mkern-2mu L}$, $\mathcal{E\mkern-2mu LI}$, guarded TGDs, and inclusion dependencies, they in general do not exist for full, frontier-guarded and frontier-one TGDs. 

**Abstract (ZH)**: 我们研究将描述逻辑$\mathcal{E\mkern-2mu L}$和$\mathcal{E\mkern-2mu LI}$以及多种元组生成依赖（TGDs）：全依赖、保护依赖、边界保护依赖、边界单一依赖和无限制依赖，以及包含依赖应用于正负例子（形式为有限关系结构）的问题。我们确定了拟合本体和TGDs的确切计算复杂性，设计了算法，并分析了拟合本体和TGDs的大小。我们还研究了为给定的一组有限结构构造概念包含/TGDs有限基的相关问题。虽然$\mathcal{E\mkern-2mu L}$、$\mathcal{E\mkern-2mu LI}$、保护TGDs和包含依赖存在有限基，但全TGDs、边界保护TGDs和边界单一TGDs通常不存在有限基。 

---
# AlphaEval: A Comprehensive and Efficient Evaluation Framework for Formula Alpha Mining 

**Title (ZH)**: AlphaEval: 公式Alpha挖掘的全面高效评估框架 

**Authors**: Hongjun Ding, Binqi Chen, Jinsheng Huang, Taian Guo, Zhengyang Mao, Guoyi Shao, Lutong Zou, Luchen Liu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13174)  

**Abstract**: Formula alpha mining, which generates predictive signals from financial data, is critical for quantitative investment. Although various algorithmic approaches-such as genetic programming, reinforcement learning, and large language models-have significantly expanded the capacity for alpha discovery, systematic evaluation remains a key challenge. Existing evaluation metrics predominantly include backtesting and correlation-based measures. Backtesting is computationally intensive, inherently sequential, and sensitive to specific strategy parameters. Correlation-based metrics, though efficient, assess only predictive ability and overlook other crucial properties such as temporal stability, robustness, diversity, and interpretability. Additionally, the closed-source nature of most existing alpha mining models hinders reproducibility and slows progress in this field. To address these issues, we propose AlphaEval, a unified, parallelizable, and backtest-free evaluation framework for automated alpha mining models. AlphaEval assesses the overall quality of generated alphas along five complementary dimensions: predictive power, stability, robustness to market perturbations, financial logic, and diversity. Extensive experiments across representative alpha mining algorithms demonstrate that AlphaEval achieves evaluation consistency comparable to comprehensive backtesting, while providing more comprehensive insights and higher efficiency. Furthermore, AlphaEval effectively identifies superior alphas compared to traditional single-metric screening approaches. All implementations and evaluation tools are open-sourced to promote reproducibility and community engagement. 

**Abstract (ZH)**: 公式α挖掘的综合评价：一种无回测的综合评估框架 

---
# Cognitive Workspace: Active Memory Management for LLMs -- An Empirical Study of Functional Infinite Context 

**Title (ZH)**: 认知工作区：针对LLMs的主动内存管理——功能无限上下文的实证研究 

**Authors**: Tao An  

**Link**: [PDF](https://arxiv.org/pdf/2508.13171)  

**Abstract**: Large Language Models (LLMs) face fundamental limitations in context management despite recent advances extending context windows to millions of tokens. We propose Cognitive Workspace, a novel paradigm that transcends traditional Retrieval-Augmented Generation (RAG) by emulating human cognitive mechanisms of external memory use. Drawing from cognitive science foundations including Baddeley's working memory model, Clark's extended mind thesis, and Hutchins' distributed cognition framework, we demonstrate that current passive retrieval systems fail to capture the dynamic, task-driven nature of human memory management. Our analysis of 2024-2025 developments reveals that while techniques like Infini-attention and StreamingLLM achieve impressive context lengths, they lack the metacognitive awareness and active planning capabilities essential for true cognitive extension. Cognitive Workspace addresses these limitations through three core innovations: (1) active memory management with deliberate information curation, (2) hierarchical cognitive buffers enabling persistent working states, and (3) task-driven context optimization that dynamically adapts to cognitive demands. Empirical validation demonstrates Cognitive Workspace achieves an average 58.6% memory reuse rate (ranging from 54-60% across different tasks) compared to 0% for traditional RAG, with 17-18% net efficiency gain despite 3.3x higher operation counts. Statistical analysis confirms these advantages with p < 0.001 and Cohen's d > 23 across multiple task types, establishing the first quantitative evidence for active memory superiority in LLM systems. We present a comprehensive theoretical framework synthesizing insights from 50+ recent papers, positioning Cognitive Workspace as a fundamental shift from information retrieval to genuine cognitive augmentation. 

**Abstract (ZH)**: 大型语言模型在上下文管理方面存在根本局限，尽管近期技术进步将上下文窗口扩展至数百万个词元。我们提出认知工作空间，这一新颖范式超越了传统的检索增强生成（RAG），通过模拟人类的认知机制来使用外部记忆。基于认知科学的基础，包括巴德利的工作记忆模型、克拉克的扩展心智论以及赫钦斯的分布式认知框架，我们证明当前的被动检索系统无法捕捉人类记忆管理的动态任务驱动特性。对2024-2025年发展趋势的分析显示，虽然无限注意力和StreamingLLM等技术实现了令人印象深刻的上下文长度，但缺乏必要的元认知意识和主动规划能力，这对于真正的认知扩展是必需的。认知工作空间通过三项核心创新来解决这些局限：（1）主动的信息管理和有目的地进行信息筛选，（2）分层次的认知缓存以支持持久的工作状态，以及（3）以任务驱动的方式优化上下文，能够动态适应认知需求。实证验证表明，认知工作空间在不同任务中平均实现了58.6%的记忆重用率（范围为54%-60%），比传统RAG高出0%，并且在操作次数增加3.3倍的情况下获得了17%-18%的净效率提升。统计分析证实了这些优势，p值小于0.001，Cohen’s d大于23，为语言模型系统中主动记忆的优势提供了首个定量证据。我们综合了50多篇最近论文的见解，将认知工作空间置于从信息检索到真正认知增强的基本转变之中。 

---
# Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL 

**Title (ZH)**: 多
user
标题翻译如下：

Chain-of-Agents：通过多
iffany
标题翻译 如下：

Chain-of-Agents： 通过多 Agent 多智能体蒸馏和能动强化学习端到端智能体基础模型 

**Authors**: Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao, Xinpeng Liu, Jiayu Zhang, Zhenqiang Huang, Qianben Chen, Weichen Sun, Qiexiang Wang, Hongxuan Lu, Tianrui Qin, Chenghao Zhu, Yi Yao, Shuying Fan, Xiaowan Li, Tiannan Wang, Pai Liu, King Zhu, He Zhu, Dingfeng Shi, Piaohong Wang, Yeyi Guan, Xiangru Tang, Minghao Liu, Yuchen Eleanor Jiang, Jian Yang, Jiaheng Liu, Ge Zhang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.13167)  

**Abstract**: Recent advances in large language models (LLMs) and multi-agent systems have demonstrated remarkable capabilities in complex problem-solving tasks such as deep research, vibe coding, and mathematical reasoning. However, most existing multi-agent systems are built upon manual prompt/workflow engineering with sophisticated agent frameworks, making them computationally inefficient, less capable, and can not benefit from data-centric learning. In this work, we introduce Chain-of-Agents (CoA), a novel paradigm of LLM reasoning that enables native end-to-end complex problem-solving in the same way as a multi-agent system (i.e., multi-turn problem solving with multiple tools and multiple agents) within one model. In chain-of-agents problem-solving, the model dynamically activates different tool agents and role-playing agents to simulate multi-agent collaboration in an end-to-end fashion. To elicit end-to-end chain-of-agents problem-solving abilities in LLMs, we introduce a multi-agent distillation framework to distill state-of-the-art multi-agent systems into chain-of-agents trajectories for agentic supervised fine-tuning. We then use agentic reinforcement learning on verifiable agentic tasks to further improve the models' capabilities on chain-of-agents problem solving. We call the resulting models Agent Foundation Models (AFMs). Our empirical studies demonstrate that AFM establishes new state-of-the-art performance across diverse benchmarks in both web agent and code agent settings. We make the entire research, including the model weights, code for training and evaluation, and the training data, fully open-sourced, which offers a solid starting point for future research on agent models and agentic RL. 

**Abstract (ZH)**: 近期大规模语言模型（LLMs）和多智能体系统的进展在深度研究、代码编写和数学推理等复杂问题解决任务中展现了显著的能力。然而，大多数现有的多智能体系统依赖于手工构建的提示/工作流工程及复杂的智能体框架，这使得它们在计算效率、功能以及从数据驱动学习中受益等方面存在不足。在本工作中，我们引入了多智能体链（Chain-of-Agents, CoA）这一新颖的LLM推理范式，以类似于多智能体系统的内方式（即多轮次、多工具和多智能体的问题解决）实现复杂的端到端问题解决。在多智能体链问题解决过程中，模型动态激活不同的工具智能体和角色扮演智能体，以端到端的方式模拟多智能体合作。为了在LLMs中引发端到端的多智能体链问题解决能力，我们提出了一种多智能体蒸馏框架，将前沿的多智能体系统蒸馏为多智能体链轨迹，用于具有智能体监督的微调。然后，我们使用智能体强化学习来进一步提高模型在多智能体链问题解决方面的能力，我们称之为代理基础模型（Agent Foundation Models, AFMs）。我们的实证研究表明，AFM在网页代理和代码代理设置下的多种基准测试中均达到了新的性能最佳。我们全面开源了整个研究，包括模型权重、训练和评估代码以及训练数据，为未来代理模型研究和智能体强化学习提供了坚实的基础。 

---
# GeoSAM2: Unleashing the Power of SAM2 for 3D Part Segmentation 

**Title (ZH)**: GeoSAM2: 解锁SAM2在三维部件分割中的潜力 

**Authors**: Ken Deng, Yunhan Yang, Jingxiang Sun, Xihui Liu, Yebin Liu, Ding Liang, Yan-Pei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14036)  

**Abstract**: Modern 3D generation methods can rapidly create shapes from sparse or single views, but their outputs often lack geometric detail due to computational constraints. We present DetailGen3D, a generative approach specifically designed to enhance these generated 3D shapes. Our key insight is to model the coarse-to-fine transformation directly through data-dependent flows in latent space, avoiding the computational overhead of large-scale 3D generative models. We introduce a token matching strategy that ensures accurate spatial correspondence during refinement, enabling local detail synthesis while preserving global structure. By carefully designing our training data to match the characteristics of synthesized coarse shapes, our method can effectively enhance shapes produced by various 3D generation and reconstruction approaches, from single-view to sparse multi-view inputs. Extensive experiments demonstrate that DetailGen3D achieves high-fidelity geometric detail synthesis while maintaining efficiency in training. 

**Abstract (ZH)**: 现代的3D生成方法可以快速从稀疏或单视角生成形状，但由于计算约束，其输出往往缺乏几何细节。我们提出了DetailGen3D，一种专门用于增强这些生成的3D形状的生成方法。我们的关键见解是通过数据相关的流在潜在空间中直接建模从粗略到精细的变换，从而避免大规模3D生成模型的计算开销。我们引入了一种token匹配策略，确保在细化过程中准确的空间对应，从而实现局部细节合成的同时保留全局结构。通过精心设计训练数据以匹配合成粗略形状的特征，该方法可以有效地增强各种3D生成和重建方法产生的形状，从单视图到稀疏多视图输入。大量实验表明，DetailGen3D在保持训练效率的同时实现了高保真几何细节合成。 

---
# Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation 

**Title (ZH)**: 代理微调引起的无意对齐风险与缓解 

**Authors**: Dongyoon Hahm, Taywon Min, Woogyeol Jin, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14031)  

**Abstract**: Beyond simple text generation, Large Language Models (LLMs) have evolved into agentic systems capable of planning and interacting with external tools to solve complex tasks. This evolution involves fine-tuning LLMs on agent-specific tasks to enhance their proficiency. However, safety concerns are frequently overlooked during this fine-tuning process. In this work, we show that aligned LLMs can become unintentionally misaligned, leading to a higher likelihood of executing harmful tasks and a reduced tendency to refuse them when fine-tuned to execute agentic tasks. To address these safety challenges, we propose Prefix INjection Guard (PING), a simple yet effective method that prepends automatically generated natural language prefixes to agent responses, guiding them to refuse harmful requests while preserving performance on benign tasks. Specifically, we introduce an iterative approach that alternates between (1) generating candidate prefixes and (2) selecting those that optimize both task performance and refusal behavior. Experimental results demonstrate that PING significantly enhances the safety of fine-tuned LLM agents without sacrificing their effectiveness. PING consistently outperforms existing prompting approaches across diverse benchmarks in both web navigation and code generation tasks. Our analysis of internal hidden states via linear probes reveals that prefix tokens are crucial for behavior modification, explaining the performance gains. WARNING: This paper contains contents that are unethical or offensive in nature. 

**Abstract (ZH)**: 超越简单的文本生成，大规模语言模型（LLMs）已经发展成为能够规划和与外部工具互动以解决复杂任务的代理系统。这一发展涉及在特定代理任务上微调LLMs以提高其专业能力。然而，在这个微调过程中，安全问题往往会被忽视。在本文中，我们展示了对齐的语言模型可能会无意中失去对齐，导致在执行代理任务时更有可能执行有害任务，并且拒绝这些任务的倾向降低。为了应对这些安全挑战，我们提出了前缀注入守护（PING）方法，这是一种简单而有效的方法，通过在代理响应前自动生成的自然语言前缀中添加，引导其拒绝有害请求，同时保留其在良性任务上的性能。具体而言，我们介绍了一种迭代方法，交替进行（1）生成候选前缀和（2）选择优化了任务性能和拒绝行为的前缀。实验结果表明，PING显著提高了微调后语言模型代理的安全性，而不会牺牲其效果。PING在网页导航和代码生成等多样基准测试中始终优于现有提示方法。通过线性探针分析内部隐藏状态显示，前缀标记对于行为调整至关重要，解释了性能提升的原因。警告：本文包含不道德或冒犯性内容。 

---
# Ask Good Questions for Large Language Models 

**Title (ZH)**: 为大型语言模型提出好问题 

**Authors**: Qi Wu, Zhongqi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14025)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved the performance of dialog systems, yet current approaches often fail to provide accurate guidance of topic due to their inability to discern user confusion in related concepts. To address this, we introduce the Ask-Good-Question (AGQ) framework, which features an improved Concept-Enhanced Item Response Theory (CEIRT) model to better identify users' knowledge levels. Our contributions include applying the CEIRT model along with LLMs to directly generate guiding questions based on the inspiring text, greatly improving information retrieval efficiency during the question & answer process. Through comparisons with other baseline methods, our approach outperforms by significantly enhencing the users' information retrieval experiences. 

**Abstract (ZH)**: 近期大规模语言模型的进展显著提升了对话系统的性能，但当前方法往往因无法分辨用户在相关概念上的混淆而无法提供准确的引导。为此，我们提出了Ask-Good-Question (AGQ)框架，该框架配备改进的概念增强项目反应理论（CEIRT）模型，以更好地识别用户的知识水平。我们的贡献包括将CEIRT模型与大规模语言模型结合，直接根据激发性文本生成引导问题，大幅提高问答过程中的信息检索效率。与其它基线方法相比，我们的方法通过显著增强用户的检索体验表现出色。 

---
# Efficient Knowledge Graph Unlearning with Zeroth-order Information 

**Title (ZH)**: 基于零阶信息的高效知识图谱遗忘技术 

**Authors**: Yang Xiao, Ruimeng Ye, Bohan Liu, Xiaolong Ma, Bo Hui  

**Link**: [PDF](https://arxiv.org/pdf/2508.14013)  

**Abstract**: Due to regulations like the Right to be Forgotten, there is growing demand for removing training data and its influence from models. Since full retraining is costly, various machine unlearning methods have been proposed. In this paper, we firstly present an efficient knowledge graph (KG) unlearning algorithm. We remark that KG unlearning is nontrivial due to the distinctive structure of KG and the semantic relations between entities. Also, unlearning by estimating the influence of removed components incurs significant computational overhead when applied to large-scale knowledge graphs. To this end, we define an influence function for KG unlearning and propose to approximate the model's sensitivity without expensive computation of first-order and second-order derivatives for parameter updates. Specifically, we use Taylor expansion to estimate the parameter changes caused by data removal. Given that the first-order gradients and second-order derivatives dominate the computational load, we use the Fisher matrices and zeroth-order optimization to approximate the inverse-Hessian vector product without constructing the computational graphs. Our experimental results demonstrate that the proposed method outperforms other state-of-the-art graph unlearning baselines significantly in terms of unlearning efficiency and unlearning quality. Our code is released at this https URL. 

**Abstract (ZH)**: 由于像“被遗忘权”这样的规定，从模型中移除训练数据及其影响的需求日益增长。由于全面重新训练成本较高，已经提出了多种机器遗忘方法。本文首先提出一个高效的知识图谱(KG)遗忘算法。我们注意到，由于知识图谱的独特结构及其实体之间的语义关系，知识图谱遗忘并非易事。此外，在大规模知识图谱上通过估算移除组件的影响来实现遗忘会带来显著的计算开销。为此，我们定义了一个知识图谱遗忘的影响函数，并提出了一种在不进行昂贵的一阶和二阶导数计算的情况下近似模型敏感性的方法。具体来说，我们使用泰勒展开来估计由于数据移除引起参数的变化。鉴于一阶梯度和二阶导数主导计算负载，我们使用费舍尔矩阵和零阶优化来近似逆海森矩阵向量积，而无需构建计算图。实验结果表明，所提出的方法在遗忘效率和遗忘质量方面显著优于其他最新的图遗忘基线方法。我们的代码发布在该网址：https://xxxxxx。 

---
# Evaluating Identity Leakage in Speaker De-Identification Systems 

**Title (ZH)**: 评估讲者去标识化系统中的身份泄露 

**Authors**: Seungmin Seo, Oleg Aulov, Afzal Godil, Kevin Mangold  

**Link**: [PDF](https://arxiv.org/pdf/2508.14012)  

**Abstract**: Speaker de-identification aims to conceal a speaker's identity while preserving intelligibility of the underlying speech. We introduce a benchmark that quantifies residual identity leakage with three complementary error rates: equal error rate, cumulative match characteristic hit rate, and embedding-space similarity measured via canonical correlation analysis and Procrustes analysis. Evaluation results reveal that all state-of-the-art speaker de-identification systems leak identity information. The highest performing system in our evaluation performs only slightly better than random guessing, while the lowest performing system achieves a 45% hit rate within the top 50 candidates based on CMC. These findings highlight persistent privacy risks in current speaker de-identification technologies. 

**Abstract (ZH)**: 演讲者去标识化旨在保护演讲者身份的同时保留其语音内容的可理解性。我们引入了一个基准，通过三种互补的错误率来量化剩余的身份泄露：等错误率、累积匹配特征命中率，以及通过典型相关分析和Procrustes分析测量的嵌入空间相似性。评估结果表明，所有最新的演讲者去标识化系统都会泄露身份信息。我们在评估中表现最好的系统仅比随机猜测略好，而表现最差的系统在基于CMC的前50个候选项中达到了45%的命中率。这些发现突显了当前演讲者去标识化技术中存在的持续隐私风险。 

---
# ASDFormer: A Transformer with Mixtures of Pooling-Classifier Experts for Robust Autism Diagnosis and Biomarker Discovery 

**Title (ZH)**: ASDFormer: 结合池化分类专家混合的变压器模型，用于稳健的自闭症诊断和生物标志物发现 

**Authors**: Mohammad Izadi, Mehran Safayani  

**Link**: [PDF](https://arxiv.org/pdf/2508.14005)  

**Abstract**: Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition marked by disruptions in brain connectivity. Functional MRI (fMRI) offers a non-invasive window into large-scale neural dynamics by measuring blood-oxygen-level-dependent (BOLD) signals across the brain. These signals can be modeled as interactions among Regions of Interest (ROIs), which are grouped into functional communities based on their underlying roles in brain function. Emerging evidence suggests that connectivity patterns within and between these communities are particularly sensitive to ASD-related alterations. Effectively capturing these patterns and identifying interactions that deviate from typical development is essential for improving ASD diagnosis and enabling biomarker discovery. In this work, we introduce ASDFormer, a Transformer-based architecture that incorporates a Mixture of Pooling-Classifier Experts (MoE) to capture neural signatures associated with ASD. By integrating multiple specialized expert branches with attention mechanisms, ASDFormer adaptively emphasizes different brain regions and connectivity patterns relevant to autism. This enables both improved classification performance and more interpretable identification of disorder-related biomarkers. Applied to the ABIDE dataset, ASDFormer achieves state-of-the-art diagnostic accuracy and reveals robust insights into functional connectivity disruptions linked to ASD, highlighting its potential as a tool for biomarker discovery. 

**Abstract (ZH)**: 自闭症谱系障碍（ASD）是一种复杂的神经发育条件，特征为脑连接性中断。功能性磁共振成像（fMRI）通过测量整个大脑的血氧水平依赖（BOLD）信号提供了一种无创的大规模神经动力学窗口。这些信号可建模为感兴趣区（ROIs）之间的相互作用，根据不同脑功能的潜在作用，将这些区分为功能性社区。新兴的证据表明，这些社区内部及之间的连接模式特别容易受到与ASD相关的改变影响。有效捕捉这些模式并识别偏离正常发育的相互作用对于提高ASD诊断能力和促进生物标志物发现至关重要。在本研究中，我们引入了ASDFormer，这是一种基于Transformer的架构，结合了混合池化分类专家（MoE）以捕捉与ASD相关的神经特征。通过集成多种专门的专家分支和注意机制，ASDFormer能够自适应地强调与自闭症相关的不同脑区和连接模式，从而提高了分类性能，并更易于识别与疾病相关的生物标志物。在ABIDE数据集上的应用证明，ASDFormer实现了最先进的诊断准确性，并揭示了与ASD相关的功能性连接中断的稳健见解，突显了其作为生物标志物发现工具的潜力。 

---
# Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation 

**Title (ZH)**: 体态-R1：强化体态推理在通用机器人操作中的应用 

**Authors**: Yifu Yuan, Haiqin Cui, Yaoting Huang, Yibin Chen, Fei Ni, Zibin Dong, Pengyi Li, Yan Zheng, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13998)  

**Abstract**: Generalization in embodied AI is hindered by the "seeing-to-doing gap," which stems from data scarcity and embodiment heterogeneity. To address this, we pioneer "pointing" as a unified, embodiment-agnostic intermediate representation, defining four core embodied pointing abilities that bridge high-level vision-language comprehension with low-level action primitives. We introduce Embodied-R1, a 3B Vision-Language Model (VLM) specifically designed for embodied reasoning and pointing. We use a wide range of embodied and general visual reasoning datasets as sources to construct a large-scale dataset, Embodied-Points-200K, which supports key embodied pointing capabilities. We then train Embodied-R1 using a two-stage Reinforced Fine-tuning (RFT) curriculum with a specialized multi-task reward design. Embodied-R1 achieves state-of-the-art performance on 11 embodied spatial and pointing benchmarks. Critically, it demonstrates robust zero-shot generalization by achieving a 56.2% success rate in the SIMPLEREnv and 87.5% across 8 real-world XArm tasks without any task-specific fine-tuning, representing a 62% improvement over strong baselines. Furthermore, the model exhibits high robustness against diverse visual disturbances. Our work shows that a pointing-centric representation, combined with an RFT training paradigm, offers an effective and generalizable pathway to closing the perception-action gap in robotics. 

**Abstract (ZH)**: Generalized Embodied AI is Impeded by the "Seeing-to-Doing Gap," Addressed by a Unified Pointing Representation and Reinforced Fine-tuning Curriculum 

---
# Chunks as Arms: Multi-Armed Bandit-Guided Sampling for Long-Context LLM Preference Optimization 

**Title (ZH)**: 块作为臂：多臂 bandit 引导的采样方法用于长期上下文 LLM 偏好优化 

**Authors**: Shaohua Duan, Xinze Li, Zhenghao Liu, Xiaoyuan Yi, Yukun Yan, Shuo Wang, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13993)  

**Abstract**: Long-context modeling is critical for a wide range of real-world tasks, including long-context question answering, summarization, and complex reasoning tasks. Recent studies have explored fine-tuning Large Language Models (LLMs) with synthetic data to enhance their long-context capabilities. However, the effectiveness of such approaches is often limited by the low diversity and factual inconsistencies in the generated data. To address these challenges, we propose LongMab-PO, a novel framework that leverages a Multi-Armed Bandit (MAB) rollout strategy to identify the most informative chunks from the given long context for sampling high-quality and diverse responses and constructing preference data pairs for Direct Preference Optimization (DPO) training. Specifically, we treat context chunks as arms of MAB, select chunks based on their expected reward scores to input into LLMs to generate responses, and iteratively update these scores based on reward feedback. This exploration and exploitation process enables the model to focus on the most relevant context segments, thereby generating and collecting high-quality and diverse responses. Finally, we collect these generated responses from the rollout process and apply the DPO method to further optimize the LLM. Experimental results show that LongMab-PO significantly improves the diversity and quality of preference data pairs, achieving state-of-the-art performance on long-context reasoning benchmarks. All code and data will be released on this https URL. 

**Abstract (ZH)**: 长上下文建模对于长上下文问答、总结和复杂推理等广泛的实际任务至关重要。近期研究探索了使用合成数据微调大型语言模型（LLMs）以提高其长上下文能力。然而，这种方法的有效性往往受限于生成数据的低多样性和事实不一致性。为解决这些挑战，我们提出了一种新的框架LongMab-PO，该框架利用多臂-bandit（MAB） rollout 策略从给定的长上下文中识别最具信息量的片段，以生成高质量和多样化的响应，并构建偏好数据对用于直接偏好优化（DPO）训练。具体而言，我们将上下文片段视为MAB的臂，根据预期奖励分值选择片段输入LLM生成响应，并根据奖励反馈迭代更新这些分值。这种探索和利用过程使模型能够专注于最相关的上下文段落，从而生成和收集高质量和多样化的响应。最终，我们从rollout过程中收集这些生成的响应，并应用DPO方法进一步优化LLM。实验结果表明，LongMab-PO显著提高了偏好数据对的多样性和质量，在长上下文推理基准测试上达到了最先进的性能。所有代码和数据将在该网址发布：https://。 

---
# The Social Context of Human-Robot Interactions 

**Title (ZH)**: 人类与机器人互动的社会背景 

**Authors**: Sydney Thompson, Kate Candon, Marynel Vázquez  

**Link**: [PDF](https://arxiv.org/pdf/2508.13982)  

**Abstract**: The Human-Robot Interaction (HRI) community often highlights the social context of an interaction as a key consideration when designing, implementing, and evaluating robot behavior. Unfortunately, researchers use the term "social context" in varied ways. This can lead to miscommunication, making it challenging to draw connections between related work on understanding and modeling the social contexts of human-robot interactions. To address this gap, we survey the HRI literature for existing definitions and uses of the term "social context". Then, we propose a conceptual model for describing the social context of a human-robot interaction. We apply this model to existing work, and we discuss a range of attributes of social contexts that can help researchers plan for interactions, develop behavior models for robots, and gain insights after interactions have taken place. We conclude with a discussion of open research questions in relation to understanding and modeling the social contexts of human-robot interactions. 

**Abstract (ZH)**: 人机交互（HRI）领域常强调互动的社会背景是设计、实现和评估机器人行为时的关键考虑因素。不幸的是，研究人员在使用“社会背景”这一术语时存在差异性。这可能导致交流不当，使得在理解与建模人机互动的社会背景方面难以建立相关工作的联系。为解决这一问题，我们调研了HRI领域的现有文献，探索“社会背景”这一术语的现有定义和使用方式。然后，我们提出了一种概念模型来描述人机互动的社会背景。我们将该模型应用于现有研究，并讨论一组有助于研究人员规划互动、为机器人开发行为模型以及互动完成后获得见解的社会背景属性。最后，我们讨论了关于理解与建模人机互动的社会背景方面的开放性研究问题。 

---
# RotBench: Evaluating Multimodal Large Language Models on Identifying Image Rotation 

**Title (ZH)**: RotBench: 评估多模态大型语言模型在识别图像旋转方面的性能 

**Authors**: Tianyi Niu, Jaemin Cho, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2508.13968)  

**Abstract**: We investigate to what extent Multimodal Large Language Models (MLLMs) can accurately identify the orientation of input images rotated 0°, 90°, 180°, and 270°. This task demands robust visual reasoning capabilities to detect rotational cues and contextualize spatial relationships within images, regardless of their orientation. To evaluate MLLMs on these abilities, we introduce RotBench -- a 350-image manually-filtered benchmark comprising lifestyle, portrait, and landscape images. Despite the relatively simple nature of this task, we show that several state-of-the-art open and proprietary MLLMs, including GPT-5, o3, and Gemini-2.5-Pro, do not reliably identify rotation in input images. Providing models with auxiliary information -- including captions, depth maps, and more -- or using chain-of-thought prompting offers only small and inconsistent improvements. Our results indicate that most models are able to reliably identify right-side-up (0°) images, while certain models are able to identify upside-down (180°) images. None can reliably distinguish between 90° and 270°. Simultaneously showing the image rotated in different orientations leads to moderate performance gains for reasoning models, while a modified setup using voting improves the performance of weaker models. We further show that fine-tuning does not improve models' ability to distinguish 90° and 270° rotations, despite substantially improving the identification of 180° images. Together, these results reveal a significant gap between MLLMs' spatial reasoning capabilities and human perception in identifying rotation. 

**Abstract (ZH)**: 我们调查了多模态大型语言模型（MLLMs）在识别输入图像旋转0°、90°、180°和270°的方向方面能够达到的准确程度。这项任务要求模型具备 robust 视觉推理能力，以检测旋转线索并理解图像中的空间关系，而不受其旋转方向的影响。为了评估 MLLMs 的这些能力，我们引入了 RotBench——一个包含350张手工筛选的生活照、肖像和风景图像的基准测试集。尽管这项任务相对简单，但我们发现，包括 GPT-5、o3 和 Gemini-2.5-Pro 在内的多种最先进的开箱即用和专有 MLLMs，并不能可靠地识别输入图像的旋转。提供辅助信息，包括字幕、深度图等，或者使用思维链提示，仅能带来有限且不一致的改进。我们的结果显示，大多数模型能够可靠地识别正向（0°）图像，而某些模型能够识别倒向（180°）图像。没有模型能够可靠地区分90°和270°。同时展示图像在不同旋转方向下则对推理模型的性能有所提升，而修改后的投票机制则增强了较弱模型的表现。此外，我们还发现，尽管在识别180°图像方面显著提高，但微调并不能提升模型区分90°和270°旋转的能力。这些结果揭示了MLLMs在识别旋转方面的空间推理能力与人类感知之间存在显著差距。 

---
# Learning to Use AI for Learning: How Can We Effectively Teach and Measure Prompting Literacy for K-12 Students? 

**Title (ZH)**: 学习使用AI进行学习：我们如何有效地教授和衡量K-12学生的问题提示素养？ 

**Authors**: Ruiwei Xiao, Xinying Hou, Ying-Jui Tseng, Hsuan Nieu, Guanze Liao, John Stamper, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2508.13962)  

**Abstract**: As Artificial Intelligence (AI) becomes increasingly integrated into daily life, there is a growing need to equip the next generation with the ability to apply, interact with, evaluate, and collaborate with AI systems responsibly. Prior research highlights the urgent demand from K-12 educators to teach students the ethical and effective use of AI for learning. To address this need, we designed an Large-Language Model (LLM)-based module to teach prompting literacy. This includes scenario-based deliberate practice activities with direct interaction with intelligent LLM agents, aiming to foster secondary school students' responsible engagement with AI chatbots. We conducted two iterations of classroom deployment in 11 authentic secondary education classrooms, and evaluated 1) AI-based auto-grader's capability; 2) students' prompting performance and confidence changes towards using AI for learning; and 3) the quality of learning and assessment materials. Results indicated that the AI-based auto-grader could grade student-written prompts with satisfactory quality. In addition, the instructional materials supported students in improving their prompting skills through practice and led to positive shifts in their perceptions of using AI for learning. Furthermore, data from Study 1 informed assessment revisions in Study 2. Analyses of item difficulty and discrimination in Study 2 showed that True/False and open-ended questions could measure prompting literacy more effectively than multiple-choice questions for our target learners. These promising outcomes highlight the potential for broader deployment and highlight the need for broader studies to assess learning effectiveness and assessment design. 

**Abstract (ZH)**: 随着人工智能（AI）越来越多地融入日常生活，培养下一代具备负责任地应用、互动、评估和协作使用AI系统的能力显得越来越重要。前期研究强调了K-12教育者对于教授学生如何以道德和有效的方式使用AI进行学习的迫切需求。为应对这一需求，我们设计了一个基于大型语言模型（LLM）的主题模块来教授提示素养。该模块包括基于场景的刻意练习活动，直接与智能LLM代理互动，旨在促进中学生负责任地与AI聊天机器人互动。我们在11个真实的中等教育班级中进行了两次课堂教学部署，并评估了1) AI基于的自动评分器的能力；2) 学生使用AI进行学习的提示表现及其自信心的变化；以及3) 学习和评估材料的质量。结果表明，基于AI的自动评分器能够以满意的质量对学生撰写的提示进行评分。此外，教学材料通过练习支持学生提高他们的提示技能，并对使用AI进行学习的看法产生了积极的影响。此外，研究1的数据在研究2的评估修订中提供了信息。研究2中项目难度和区分度的分析显示，对于我们的目标学习者，真伪判断和开放性问题比选择题更适合衡量提示素养。这些令人鼓舞的结果强调了更广泛部署的可能性，并突显了需要进行更广泛的研究来评估学习效果和评估设计。 

---
# A Mechanism for Mutual Fairness in Cooperative Games with Replicable Resources -- Extended Version 

**Title (ZH)**: 具有可复制资源的合作博弈中的相互公平机制——扩展版本 

**Authors**: Björn Filter, Ralf Möller, Özgür Lütfü Özçep  

**Link**: [PDF](https://arxiv.org/pdf/2508.13960)  

**Abstract**: The latest developments in AI focus on agentic systems where artificial and human agents cooperate to realize global goals. An example is collaborative learning, which aims to train a global model based on data from individual agents. A major challenge in designing such systems is to guarantee safety and alignment with human values, particularly a fair distribution of rewards upon achieving the global goal. Cooperative game theory offers useful abstractions of cooperating agents via value functions, which assign value to each coalition, and via reward functions. With these, the idea of fair allocation can be formalized by specifying fairness axioms and designing concrete mechanisms. Classical cooperative game theory, exemplified by the Shapley value, does not fully capture scenarios like collaborative learning, as it assumes nonreplicable resources, whereas data and models can be replicated. Infinite replicability requires a generalized notion of fairness, formalized through new axioms and mechanisms. These must address imbalances in reciprocal benefits among participants, which can lead to strategic exploitation and unfair allocations. The main contribution of this paper is a mechanism and a proof that it fulfills the property of mutual fairness, formalized by the Balanced Reciprocity Axiom. It ensures that, for every pair of players, each benefits equally from the participation of the other. 

**Abstract (ZH)**: 最近人工智能的发展集中在代理系统领域，其中人工代理和人类代理协作以实现全球目标。例如，协作学习旨在基于个体代理的数据训练全球模型。设计此类系统的主要挑战之一是确保安全并与其人类价值观保持一致，特别是全球目标达成后的奖励公平分配。合作博弈论通过价值函数和奖励函数提供了合作代理的有效抽象，这些可以正式化公平分配的概念，通过指定公平公理并设计具体的机制。经典的合作博弈论如夏普利值未能充分捕捉到如协作学习这样的场景，因为它假设资源不可复制，而数据和模型是可以复制的。无限可复制性需要通过新的公理和机制来形式化的广义公平概念。这些机制必须解决参与者之间相互利益不平衡的问题，这可能导致战略上的剥削和不公平的分配。本文的主要贡献是一种机制及其证明，该机制满足平衡互惠公理所形式化的互惠公平性属性，确保对每一对玩家而言，他们都从彼此的参与中获得平等的收益。 

---
# Prompt Orchestration Markup Language 

**Title (ZH)**: 提示编排标记语言 

**Authors**: Yuge Zhang, Nan Chen, Jiahang Xu, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13948)  

**Abstract**: Large Language Models (LLMs) require sophisticated prompting, yet current practices face challenges in structure, data integration, format sensitivity, and tooling. Existing methods lack comprehensive solutions for organizing complex prompts involving diverse data types (documents, tables, images) or managing presentation variations systematically. To address these gaps, we introduce POML (Prompt Orchestration Markup Language). POML employs component-based markup for logical structure (roles, tasks, examples), specialized tags for seamless data integration, and a CSS-like styling system to decouple content from presentation, reducing formatting sensitivity. It includes templating for dynamic prompts and a comprehensive developer toolkit (IDE support, SDKs) to improve version control and collaboration. We validate POML through two case studies demonstrating its impact on complex application integration (PomLink) and accuracy performance (TableQA), as well as a user study assessing its effectiveness in real-world development scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）需要复杂的提示，现有实践在结构、数据整合、格式敏感性和工具方面面临挑战。现有方法缺乏全面解决涉及多种数据类型（文档、表格、图像）的复杂提示组织或系统化管理呈现变体的解决方案。为解决这些缺口，我们引入了POML（Prompt Orchestration Markup Language）。POML采用基于组件的标记来定义逻辑结构（角色、任务、示例），专门的标签用于无缝数据整合，并采用类似CSS的样式系统来解耦内容与呈现，降低格式敏感性。它包含动态提示的模板功能，并提供全面的开发工具包（IDE支持、SDKs）以改进版本控制和协作。我们通过两个案例研究验证了POML在复杂应用集成（PomLink）和准确性性能（TableQA）方面的成效，并通过用户研究评估了其在实际开发场景中的有效性。 

---
# InPars+: Supercharging Synthetic Data Generation for Information Retrieval Systems 

**Title (ZH)**: InPars+: 用于信息检索系统的合成数据生成加速方法 

**Authors**: Matey Krastev, Miklos Hamar, Danilo Toapanta, Jesse Brouwers, Yibin Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.13930)  

**Abstract**: This work revisits and extends synthetic query generation pipelines for Neural Information Retrieval (NIR) by leveraging the InPars Toolkit, a reproducible, end-to-end framework for generating training data using large language models (LLMs). We first assess the reproducibility of the original InPars, InPars-V2, and Promptagator pipelines on the SciFact benchmark and validate their effectiveness using open-source reranker and generator models. Building on this foundation, we introduce two key extensions to the pipeline: (1) fine-tuning a query generator LLM via Contrastive Preference Optimization (CPO) to improve the signal quality in generated queries, and (2) replacing static prompt templates with dynamic, Chain-of-Thought (CoT) optimized prompts using the DSPy framework. Our results show that both extensions reduce the need for aggressive filtering while improving retrieval performance. All code, models, and synthetic datasets are publicly released to support further research at: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 本研究利用InPars Toolkit（一个可重现的端到端框架，用于使用大型语言模型生成训练数据）重新审视并扩展了神经信息检索（NIR）的合成查询生成管道。我们首先在SciFact基准上评估了原始InPars、InPars-V2和Promptagator管道的可重现性，并使用开源重排器和生成模型验证了它们的有效性。在此基础上，我们引入了管道的两个关键扩展：（1）通过对比偏好优化（CPO）微调查询生成大型语言模型，以提高生成查询的质量信号；（2）使用DSPy框架将静态提示模板替换为动态的、基于推理链（CoT）优化的提示。实验结果表明，这两种扩展在降低剧烈筛选需求的同时提升了检索性能。所有代码、模型和合成数据集均已在以下链接公开发布：\href{this https URL}{this https URL}。 

---
# Categorical Policies: Multimodal Policy Learning and Exploration in Continuous Control 

**Title (ZH)**: 分类策略：连续控制中的多模态策略学习与探索 

**Authors**: SM Mazharul Islam, Manfred Huber  

**Link**: [PDF](https://arxiv.org/pdf/2508.13922)  

**Abstract**: A policy in deep reinforcement learning (RL), either deterministic or stochastic, is commonly parameterized as a Gaussian distribution alone, limiting the learned behavior to be unimodal. However, the nature of many practical decision-making problems favors a multimodal policy that facilitates robust exploration of the environment and thus to address learning challenges arising from sparse rewards, complex dynamics, or the need for strategic adaptation to varying contexts. This issue is exacerbated in continuous control domains where exploration usually takes place in the vicinity of the predicted optimal action, either through an additive Gaussian noise or the sampling process of a stochastic policy. In this paper, we introduce Categorical Policies to model multimodal behavior modes with an intermediate categorical distribution, and then generate output action that is conditioned on the sampled mode. We explore two sampling schemes that ensure differentiable discrete latent structure while maintaining efficient gradient-based optimization. By utilizing a latent categorical distribution to select the behavior mode, our approach naturally expresses multimodality while remaining fully differentiable via the sampling tricks. We evaluate our multimodal policy on a set of DeepMind Control Suite environments, demonstrating that through better exploration, our learned policies converge faster and outperform standard Gaussian policies. Our results indicate that the Categorical distribution serves as a powerful tool for structured exploration and multimodal behavior representation in continuous control. 

**Abstract (ZH)**: 一种基于深度强化学习的分类策略：模型多模态行为并促进高效可微优化 

---
# Fisher-Orthogonal Projection Methods for Natural Gradient Descent with Large Batches 

**Title (ZH)**: Fisher-正交投影方法在大数据批量下的自然梯度下降 

**Authors**: Yishun Lu, Wesley Armour  

**Link**: [PDF](https://arxiv.org/pdf/2508.13898)  

**Abstract**: Modern GPUs are equipped with large amounts of high-bandwidth memory, enabling them to support mini-batch sizes of up to tens of thousands of training samples. However, most existing optimizers struggle to perform effectively at such a large batch size. As batch size increases, gradient noise decreases due to averaging over many samples, limiting the ability of first-order methods to escape sharp or suboptimal minima and reach the global minimum. Meanwhile, second-order methods like the natural gradient with Kronecker-Factored Approximate Curvature (KFAC) often require excessively high damping to remain stable at large batch sizes. This high damping effectively washes out the curvature information that gives these methods their advantage, reducing their performance to that of simple gradient descent. In this paper, we introduce Fisher-Orthogonal Projection (FOP), a novel technique that restores the effectiveness of the second-order method at very large batch sizes, enabling scalable training with improved generalization and faster convergence. FOP constructs a variance-aware update direction by leveraging gradients from two sub-batches, enhancing the average gradient with a component of the gradient difference that is orthogonal to the average under the Fisher-metric. 

**Abstract (ZH)**: 现代GPU配备了大容量高带宽内存，使其能够支持数万级的训练样本批量大小。然而，现有的大多数优化器在如此大的批量大小下难以有效工作。随着批量大小的增加，由于对众多样本进行平均，梯度噪声会减少，限制了基于一阶方法从尖锐或次优极小值中逃逸并达到全局极小值的能力。同时，如Kronecker-Factored Approximate Curvature (KFAC) 自然梯度等二阶方法在大批量大小下通常需要极大的阻尼以保持稳定，这种高阻尼有效消除了这些方法具有的曲率信息优势，使其性能降低到简单的梯度下降的水平。在本文中，我们引入了Fisher-正交投影（FOP）这一新颖的技术，该技术可以在非常大的批量大小下恢复二阶方法的有效性，从而实现可扩展的训练并提高泛化能力和加速收敛。FOP通过利用两个子批量的梯度构造出一个方差感知的更新方向，在Fisher度量下，通过增加梯度差异的正交分量来增强平均梯度。 

---
# Toward Deployable Multi-Robot Collaboration via a Symbolically-Guided Decision Transformer 

**Title (ZH)**: 基于符号引导决策转换的可部署多机器人协作研究 

**Authors**: Rathnam Vidushika Rasanji, Jin Wei-Kocsis, Jiansong Zhang, Dongming Gan, Ragu Athinarayanan, Paul Asunda  

**Link**: [PDF](https://arxiv.org/pdf/2508.13877)  

**Abstract**: Reinforcement learning (RL) has demonstrated great potential in robotic operations. However, its data-intensive nature and reliance on the Markov Decision Process (MDP) assumption limit its practical deployment in real-world scenarios involving complex dynamics and long-term temporal dependencies, such as multi-robot manipulation. Decision Transformers (DTs) have emerged as a promising offline alternative by leveraging causal transformers for sequence modeling in RL tasks. However, their applications to multi-robot manipulations still remain underexplored. To address this gap, we propose a novel framework, Symbolically-Guided Decision Transformer (SGDT), which integrates a neuro-symbolic mechanism with a causal transformer to enable deployable multi-robot collaboration. In the proposed SGDT framework, a neuro-symbolic planner generates a high-level task-oriented plan composed of symbolic subgoals. Guided by these subgoals, a goal-conditioned decision transformer (GCDT) performs low-level sequential decision-making for multi-robot manipulation. This hierarchical architecture enables structured, interpretable, and generalizable decision making in complex multi-robot collaboration tasks. We evaluate the performance of SGDT across a range of task scenarios, including zero-shot and few-shot scenarios. To our knowledge, this is the first work to explore DT-based technology for multi-robot manipulation. 

**Abstract (ZH)**: 符号引导决策转换器：面向多机器人操作的可部署框架 

---
# A Novel Attention-Augmented Wavelet YOLO System for Real-time Brain Vessel Segmentation on Transcranial Color-coded Doppler 

**Title (ZH)**: 一种用于经颅彩色编码多普勒实时脑血管分割的新型注意力增强小波YOLO系统 

**Authors**: Wenxuan Zhang, Shuai Li, Xinyi Wang, Yu Sun, Hongyu Kang, Pui Yuk Chryste Wan, Yong-Ping Zheng, Sai-Kit Lam  

**Link**: [PDF](https://arxiv.org/pdf/2508.13875)  

**Abstract**: The Circle of Willis (CoW), vital for ensuring consistent blood flow to the brain, is closely linked to ischemic stroke. Accurate assessment of the CoW is important for identifying individuals at risk and guiding appropriate clinical management. Among existing imaging methods, Transcranial Color-coded Doppler (TCCD) offers unique advantages due to its radiation-free nature, affordability, and accessibility. However, reliable TCCD assessments depend heavily on operator expertise for identifying anatomical landmarks and performing accurate angle correction, which limits its widespread adoption. To address this challenge, we propose an AI-powered, real-time CoW auto-segmentation system capable of efficiently capturing cerebral arteries. No prior studies have explored AI-driven cerebrovascular segmentation using TCCD. In this work, we introduce a novel Attention-Augmented Wavelet YOLO (AAW-YOLO) network tailored for TCCD data, designed to provide real-time guidance for brain vessel segmentation in the CoW. We prospectively collected TCCD data comprising 738 annotated frames and 3,419 labeled artery instances to establish a high-quality dataset for model training and evaluation. The proposed AAW-YOLO demonstrated strong performance in segmenting both ipsilateral and contralateral CoW vessels, achieving an average Dice score of 0.901, IoU of 0.823, precision of 0.882, recall of 0.926, and mAP of 0.953, with a per-frame inference speed of 14.199 ms. This system offers a practical solution to reduce reliance on operator experience in TCCD-based cerebrovascular screening, with potential applications in routine clinical workflows and resource-constrained settings. Future research will explore bilateral modeling and larger-scale validation. 

**Abstract (ZH)**: Willis环自动分割的注意力增强小波YOLO网络在经颅彩色编码多普勒成像中的应用 

---
# UniECS: Unified Multimodal E-Commerce Search Framework with Gated Cross-modal Fusion 

**Title (ZH)**: UniECS：统一的多模态电商搜索框架，带有门控跨模态融合 

**Authors**: Zihan Liang, Yufei Ma, ZhiPeng Qian, Huangyu Dai, Zihan Wang, Ben Chen, Chenyi Lei, Yuqing Ding, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13843)  

**Abstract**: Current e-commerce multimodal retrieval systems face two key limitations: they optimize for specific tasks with fixed modality pairings, and lack comprehensive benchmarks for evaluating unified retrieval approaches. To address these challenges, we introduce UniECS, a unified multimodal e-commerce search framework that handles all retrieval scenarios across image, text, and their combinations. Our work makes three key contributions. First, we propose a flexible architecture with a novel gated multimodal encoder that uses adaptive fusion mechanisms. This encoder integrates different modality representations while handling missing modalities. Second, we develop a comprehensive training strategy to optimize learning. It combines cross-modal alignment loss (CMAL), cohesive local alignment loss (CLAL), intra-modal contrastive loss (IMCL), and adaptive loss weighting. Third, we create M-BEER, a carefully curated multimodal benchmark containing 50K product pairs for e-commerce search evaluation. Extensive experiments demonstrate that UniECS consistently outperforms existing methods across four e-commerce benchmarks with fine-tuning or zero-shot evaluation. On our M-BEER bench, UniECS achieves substantial improvements in cross-modal tasks (up to 28\% gain in R@10 for text-to-image retrieval) while maintaining parameter efficiency (0.2B parameters) compared to larger models like GME-Qwen2VL (2B) and MM-Embed (8B). Furthermore, we deploy UniECS in the e-commerce search platform of Kuaishou Inc. across two search scenarios, achieving notable improvements in Click-Through Rate (+2.74\%) and Revenue (+8.33\%). The comprehensive evaluation demonstrates the effectiveness of our approach in both experimental and real-world settings. Corresponding codes, models and datasets will be made publicly available at this https URL. 

**Abstract (ZH)**: 当前的电子商务多模态检索系统面临两大关键限制：它们针对特定任务进行优化并采用固定模态配对，缺乏评估统一检索方法的全面基准。为了解决这些挑战，我们引入了UniECS，一个统一的电子商务多模态搜索框架，可以处理图像、文本及其组合的所有检索场景。我们的工作做出了三项关键贡献。首先，我们提出了一种灵活的架构，其中包含一种新型门控多模态编码器，使用自适应融合机制。该编码器在处理缺失模态的同时整合了不同的模态表示。其次，我们开发了一种全面的训练策略来优化学习。该策略结合了跨模态对齐损失（CMAL）、协同局部对齐损失（CLAL）、模内对比损失（IMCL）以及自适应损失加权。第三，我们创建了M-BEER，一个精心编 curated 的多模态基准，包含50K个产品对，用于电子商务搜索评估。广泛的实验表明，UniECS在四种电子商务基准上的微调或零-shot评估中始终优于现有方法。在我们的M-BEER基准上，UniECS在跨模态任务中取得了显著改进（文本到图像检索的R@10增益高达28%），同时保持了参数效率（0.2B参数），比GME-Qwen2VL（2B）和MM-Embed（8B）等大型模型更具优势。此外，我们在快手公司两个搜索场景下的电子商务搜索平台上部署了UniECS，实现了点击率 (+2.74%) 和收入 (+8.33%) 的显著提升。全面的评估证明了我们方法在实验和现实世界设置中的有效性。相关代码、模型和数据集将在以下网址公开：this https URL。 

---
# One Shot vs. Iterative: Rethinking Pruning Strategies for Model Compression 

**Title (ZH)**: 一次裁剪 vs. 迭代裁剪：重新思考模型压缩的裁剪策略 

**Authors**: Mikołaj Janusz, Tomasz Wojnar, Yawei Li, Luca Benini, Kamil Adamczewski  

**Link**: [PDF](https://arxiv.org/pdf/2508.13836)  

**Abstract**: Pruning is a core technique for compressing neural networks to improve computational efficiency. This process is typically approached in two ways: one-shot pruning, which involves a single pass of training and pruning, and iterative pruning, where pruning is performed over multiple cycles for potentially finer network refinement. Although iterative pruning has historically seen broader adoption, this preference is often assumed rather than rigorously tested. Our study presents one of the first systematic and comprehensive comparisons of these methods, providing rigorous definitions, benchmarking both across structured and unstructured settings, and applying different pruning criteria and modalities. We find that each method has specific advantages: one-shot pruning proves more effective at lower pruning ratios, while iterative pruning performs better at higher ratios. Building on these findings, we advocate for patience-based pruning and introduce a hybrid approach that can outperform traditional methods in certain scenarios, providing valuable insights for practitioners selecting a pruning strategy tailored to their goals and constraints. Source code is available at this https URL. 

**Abstract (ZH)**: 剪枝是压缩神经网络以提高计算效率的核心技术。这一过程通常有两种方式：单次剪枝，即通过一次训练和剪枝完成；迭代剪枝，则通过多次循环剪枝以实现更精细的网络优化。尽管迭代剪枝在过去更为常用，但这种偏好通常被认为是理所当然的，而非经过严格的测试。我们的研究提供了首次系统且全面地比较这两种方法的尝试，提出了严格的定义，跨结构化和非结构化设置进行基准测试，并应用不同的剪枝标准和模式。我们发现，每种方法各有优势：单次剪枝在较低剪枝比例下更有效，而迭代剪枝在较高比例下表现更好。基于这些发现，我们提倡基于耐心的剪枝，并引入了一种混合方法，该方法在某些情况下可以超越传统方法，为从业者选择了符合其目标和约束条件的剪枝策略提供了宝贵的见解。相关源代码可在以下链接获取。 

---
# Extracting Structured Requirements from Unstructured Building Technical Specifications for Building Information Modeling 

**Title (ZH)**: 从建筑技术规范中提取结构化需求以支持建筑信息建模 

**Authors**: Insaf Nahri, Romain Pinquié, Philippe Véron, Nicolas Bus, Mathieu Thorel  

**Link**: [PDF](https://arxiv.org/pdf/2508.13833)  

**Abstract**: This study explores the integration of Building Information Modeling (BIM) with Natural Language Processing (NLP) to automate the extraction of requirements from unstructured French Building Technical Specification (BTS) documents within the construction industry. Employing Named Entity Recognition (NER) and Relation Extraction (RE) techniques, the study leverages the transformer-based model CamemBERT and applies transfer learning with the French language model Fr\_core\_news\_lg, both pre-trained on a large French corpus in the general domain. To benchmark these models, additional approaches ranging from rule-based to deep learning-based methods are developed. For RE, four different supervised models, including Random Forest, are implemented using a custom feature vector. A hand-crafted annotated dataset is used to compare the effectiveness of NER approaches and RE models. Results indicate that CamemBERT and Fr\_core\_news\_lg exhibited superior performance in NER, achieving F1-scores over 90\%, while Random Forest proved most effective in RE, with an F1 score above 80\%. The outcomes are intended to be represented as a knowledge graph in future work to further enhance automatic verification systems. 

**Abstract (ZH)**: 本研究探索将建筑信息建模（BIM）与自然语言处理（NLP）集成，以自动化提取 construction 行业未结构化法国建筑技术规范（BTS）文档中的要求。利用命名实体识别（NER）和关系提取（RE）技术，研究利用基于变换器的模型 CamemBERT，并采用与通用领域大规模法语文本预训练的 French 语言模型 Fr\_core\_news\_lg 结合的迁移学习方法。为了评估这些模型，还开发了从规则基于到深度学习基于的各种方法。对于 RE，实现四种监督模型，包括随机森林，使用定制特征向量。使用手工标注数据集来比较 NER 方法和 RE 模型的有效性。结果显示，CamemBERT 和 Fr\_core\_news\_lg 在 NER 中表现出色，F1 分数超过 90%，而随机森林在 RE 中表现最佳，F1 分数超过 80%。研究结果旨在未来工作通过知识图谱形式进一步增强自动化验证系统。 

---
# The illusion of a perfect metric: Why evaluating AI's words is harder than it looks 

**Title (ZH)**: 完美度量的幻象：为何评估AI的话语比看起来的要困难得多 

**Authors**: Maria Paz Oliva, Adriana Correia, Ivan Vankov, Viktor Botev  

**Link**: [PDF](https://arxiv.org/pdf/2508.13816)  

**Abstract**: Evaluating Natural Language Generation (NLG) is crucial for the practical adoption of AI, but has been a longstanding research challenge. While human evaluation is considered the de-facto standard, it is expensive and lacks scalability. Practical applications have driven the development of various automatic evaluation metrics (AEM), designed to compare the model output with human-written references, generating a score which approximates human judgment. Over time, AEMs have evolved from simple lexical comparisons, to semantic similarity models and, more recently, to LLM-based evaluators. However, it seems that no single metric has emerged as a definitive solution, resulting in studies using different ones without fully considering the implications. This paper aims to show this by conducting a thorough examination of the methodologies of existing metrics, their documented strengths and limitations, validation methods, and correlations with human judgment. We identify several key challenges: metrics often capture only specific aspects of text quality, their effectiveness varies by task and dataset, validation practices remain unstructured, and correlations with human judgment are inconsistent. Importantly, we find that these challenges persist in the most recent type of metric, LLM-as-a-Judge, as well as in the evaluation of Retrieval Augmented Generation (RAG), an increasingly relevant task in academia and industry. Our findings challenge the quest for the 'perfect metric'. We propose selecting metrics based on task-specific needs and leveraging complementary evaluations and advocate that new metrics should focus on enhanced validation methodologies. 

**Abstract (ZH)**: 评估自然语言生成（NLG）对于人工智能的实际应用至关重要，但一直是一个长期的研究挑战。尽管人类评估被认为是标准方法，但它成本高且缺乏可扩展性。实际应用推动了各种自动评价指标（AEM）的发展，旨在将模型输出与人类撰写的参考标准进行比较，生成一个接近人类判断的评分。随着时间的推移，AEM从简单的词典比较发展到语义相似性模型，再到基于大语言模型的评价者。然而，似乎没有单一的度量标准能够成为最终解决方案，导致研究中使用不同的度量标准而未充分考虑其影响。本文旨在通过详细研究现有度量标准的方法、其记录的优势和局限性、验证方法以及与人类判断的相关性来展示这一点。我们识别了几个关键挑战：度量标准通常仅捕捉文本质量的特定方面，其有效性随任务和数据集而变化，验证实践仍缺乏结构，并且与人类判断的相关性不一致。重要的是，我们发现这些挑战不仅存在于最新类型的度量标准——大语言模型作为评价者——中，而且还存在于检索增强生成（RAG）的评估中，这一任务在学术界和工业界日益相关。我们的发现挑战了寻找“完美度量标准”的追求。我们建议根据任务特定需求选择度量标准，并利用补充性评估方法，并且提倡新度量标准应关注增强的验证方法。 

---
# Assessing Trustworthiness of AI Training Dataset using Subjective Logic -- A Use Case on Bias 

**Title (ZH)**: 基于主观逻辑评估AI训练数据集的可信度——以偏见为例的研究 

**Authors**: Koffi Ismael Ouattara, Ioannis Krontiris, Theo Dimitrakos, Frank Kargl  

**Link**: [PDF](https://arxiv.org/pdf/2508.13813)  

**Abstract**: As AI systems increasingly rely on training data, assessing dataset trustworthiness has become critical, particularly for properties like fairness or bias that emerge at the dataset level. Prior work has used Subjective Logic to assess trustworthiness of individual data, but not to evaluate trustworthiness properties that emerge only at the level of the dataset as a whole. This paper introduces the first formal framework for assessing the trustworthiness of AI training datasets, enabling uncertainty-aware evaluations of global properties such as bias. Built on Subjective Logic, our approach supports trust propositions and quantifies uncertainty in scenarios where evidence is incomplete, distributed, and/or conflicting. We instantiate this framework on the trustworthiness property of bias, and we experimentally evaluate it based on a traffic sign recognition dataset. The results demonstrate that our method captures class imbalance and remains interpretable and robust in both centralized and federated contexts. 

**Abstract (ZH)**: 随着AI系统越来越依赖训练数据，评估数据集可信度已成为关键，特别是在公平性或偏差等数据集层面涌现的属性方面。先前的工作使用主观逻辑评估单个数据的可信度，但尚未用于评估仅在数据集整体层面涌现的可信度属性。本文介绍了首个正式框架，用于评估AI训练数据集的可信度，能够进行全局属性（如偏差）的不确定性感知评估。该方法基于主观逻辑，支持信任命题并在证据不完整、分散或存在冲突的情况下量化不确定性。我们在偏差这一可信度属性上实例化了这一框架，并基于交通标志识别数据集进行了实证评估。结果表明，我们的方法能够捕捉类别不平衡，并在集中式和联邦式环境中保持可解释性和稳健性。 

---
# Prompt-Based One-Shot Exact Length-Controlled Generation with LLMs 

**Title (ZH)**: 基于提示的一次生成精确长度控制生成ewith大语言模型 

**Authors**: Juncheng Xie, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13805)  

**Abstract**: Controlling the length of text produced by large language models (LLMs) remains challenging: models frequently overshoot or undershoot explicit length instructions because they cannot reliably keep an internal token count. We present a prompt-based, one-shot strategy that compels an off-the-shelf LLM to generate exactly a desired number of tokens - words (English) or characters (Chinese) - without any fine-tuning or iterative sampling. The prompt appends countdown markers and explicit counting rules so that the model "writes while counting." We evaluate on four settings: open-ended generation (1-1000 tokens), XSUM summarization, MT-Bench-LI instruction following, and the LIFEBENCH equal-length track. On MT-Bench-LI, strict length compliance with GPT-4.1 leaps from below 30% under naive prompts to above 95% with our countdown prompt, surpassing the popular draft-then-revise baseline, while judged answer quality is preserved. These results show that precise length control can be achieved through prompt engineering alone, offering a lightweight alternative to training- or decoding-based methods. 

**Abstract (ZH)**: 控制大型语言模型生成文本的长度仍然具有挑战性：模型经常无法可靠地保持内部令牌计数，从而导致长度过度或不足。我们提出了一种基于提示的一次性策略，使即用型大语言模型生成恰好所需的令牌数（单词或字符）数量，无需任何微调或迭代采样。该提示附加了倒计时标记和明确的计数规则，使模型“边写边计数”。我们在四种设置下进行了评估：开放式生成（1-1000个令牌）、XSUM摘要、MT-Bench-LI指令跟随以及LIFEBENCH等长度赛道。在MT-Bench-LI上，使用标准化提示时对GPT-4.1的严格长度合规率低于30%，而使用我们的倒计时提示则提升至超过95%，超过了流行的草拟后再修订baseline方法，同时保持了判断答案质量。这些结果表明，仅通过提示工程就可以实现精确的长度控制，提供了一种轻量级的替代训练或解码方法。 

---
# A Fully Transformer Based Multimodal Framework for Explainable Cancer Image Segmentation Using Radiology Reports 

**Title (ZH)**: 基于完全变换器的多模态解释性癌症图像分割框架，结合放射学报告 

**Authors**: Enobong Adahada, Isabel Sassoon, Kate Hone, Yongmin Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13796)  

**Abstract**: We introduce Med-CTX, a fully transformer based multimodal framework for explainable breast cancer ultrasound segmentation. We integrate clinical radiology reports to boost both performance and interpretability. Med-CTX achieves exact lesion delineation by using a dual-branch visual encoder that combines ViT and Swin transformers, as well as uncertainty aware fusion. Clinical language structured with BI-RADS semantics is encoded by BioClinicalBERT and combined with visual features utilising cross-modal attention, allowing the model to provide clinically grounded, model generated explanations. Our methodology generates segmentation masks, uncertainty maps, and diagnostic rationales all at once, increasing confidence and transparency in computer assisted diagnosis. On the BUS-BRA dataset, Med-CTX achieves a Dice score of 99% and an IoU of 95%, beating existing baselines U-Net, ViT, and Swin. Clinical text plays a key role in segmentation accuracy and explanation quality, as evidenced by ablation studies that show a -5.4% decline in Dice score and -31% in CIDEr. Med-CTX achieves good multimodal alignment (CLIP score: 85%) and increased confi dence calibration (ECE: 3.2%), setting a new bar for trustworthy, multimodal medical architecture. 

**Abstract (ZH)**: Med-CTX：基于Transformer的多模态可解释乳腺癌超声分割框架 

---
# BetaWeb: Towards a Blockchain-enabled Trustworthy Agentic Web 

**Title (ZH)**: BetaWeb: 向一个区块链驱动的值得信赖的代理Web迈进 

**Authors**: Zihan Guo, Yuanjian Zhou, Chenyi Wang, Linlin You, Minjie Bian, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13787)  

**Abstract**: The rapid development of large language models (LLMs) has significantly propelled the development of artificial intelligence (AI) agents, which are increasingly evolving into diverse autonomous entities, advancing the LLM-based multi-agent systems (LaMAS). However, current agentic ecosystems remain fragmented and closed. Establishing an interconnected and scalable paradigm for Agentic AI has become a critical prerequisite. Although Agentic Web proposes an open architecture to break the ecosystem barriers, its implementation still faces core challenges such as privacy protection, data management, and value measurement. Existing centralized or semi-centralized paradigms suffer from inherent limitations, making them inadequate for supporting large-scale, heterogeneous, and cross-domain autonomous interactions. To address these challenges, this paper introduces the blockchain-enabled trustworthy Agentic Web (BetaWeb). By leveraging the inherent strengths of blockchain, BetaWeb not only offers a trustworthy and scalable infrastructure for LaMAS but also has the potential to advance the Web paradigm from Web3 (centered on data ownership) towards Web3.5, which emphasizes ownership of agent capabilities and the monetization of intelligence. Beyond a systematic examination of the BetaWeb framework, this paper presents a five-stage evolutionary roadmap, outlining the path of LaMAS from passive execution to advanced collaboration and autonomous governance. We also conduct a comparative analysis of existing products and discuss key challenges of BetaWeb from multiple perspectives. Ultimately, we argue that deep integration between blockchain and LaMAS can lay the foundation for a resilient, trustworthy, and sustainably incentivized digital ecosystem. A summary of the enabling technologies for each stage is available at this https URL. 

**Abstract (ZH)**: 区块链赋能可信代理Web（BetaWeb） 

---
# DegDiT: Controllable Audio Generation with Dynamic Event Graph Guided Diffusion Transformer 

**Title (ZH)**: DegDiT：受动态事件图引导的可控音频生成变换器 

**Authors**: Yisu Liu, Chenxing Li, Wanqian Zhang, Wenfu Wang, Meng Yu, Ruibo Fu, Zheng Lin, Weiping Wang, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13786)  

**Abstract**: Controllable text-to-audio generation aims to synthesize audio from textual descriptions while satisfying user-specified constraints, including event types, temporal sequences, and onset and offset timestamps. This enables precise control over both the content and temporal structure of the generated audio. Despite recent progress, existing methods still face inherent trade-offs among accurate temporal localization, open-vocabulary scalability, and practical efficiency. To address these challenges, we propose DegDiT, a novel dynamic event graph-guided diffusion transformer framework for open-vocabulary controllable audio generation. DegDiT encodes the events in the description as structured dynamic graphs. The nodes in each graph are designed to represent three aspects: semantic features, temporal attributes, and inter-event connections. A graph transformer is employed to integrate these nodes and produce contextualized event embeddings that serve as guidance for the diffusion model. To ensure high-quality and diverse training data, we introduce a quality-balanced data selection pipeline that combines hierarchical event annotation with multi-criteria quality scoring, resulting in a curated dataset with semantic diversity. Furthermore, we present consensus preference optimization, facilitating audio generation through consensus among multiple reward signals. Extensive experiments on AudioCondition, DESED, and AudioTime datasets demonstrate that DegDiT achieves state-of-the-art performances across a variety of objective and subjective evaluation metrics. 

**Abstract (ZH)**: 可控文本到音频生成旨在从文本描述中合成音频，同时满足用户指定的约束，包括事件类型、时间序列以及起始和结束时间戳。这使得对生成音频的内容和时间结构进行精确控制成为可能。尽管取得了近期进展，现有方法仍然在准确的时间定位、开放式词汇表的可扩展性和实用效率之间存在固有的权衡。为了解决这些挑战，我们提出了一种新颖的动态事件图引导的扩散变换器框架DegDiT，用于开放式词汇表的可控音频生成。DegDiT 将描述中的事件编码为结构化的动态图。每个图中的节点设计用于表示三个方面：语义特征、时间属性和事件间的连接。采用图变换器将这些节点进行整合，生成具有引导作用的事件上下文嵌入，作为扩散模型的指导。为确保高质量和多样化的训练数据，我们引入了一种基于层次事件注释与多指标质量评分的质量平衡数据选择管道，从而生成语义多样的数据集。此外，我们提出了共识偏好优化，通过多个奖励信号的一致性促进音频生成。在AudioCondition、DESED和AudioTime数据集上的广泛实验表明，DegDiT 在多种客观和主观评估指标上实现了最先进的性能。 

---
# Comparing Conditional Diffusion Models for Synthesizing Contrast-Enhanced Breast MRI from Pre-Contrast Images 

**Title (ZH)**: 比较条件扩散模型在从非增强MRI合成增强乳腺MRI中的应用 

**Authors**: Sebastian Ibarra, Javier del Riego, Alessandro Catanese, Julian Cuba, Julian Cardona, Nataly Leon, Jonathan Infante, Karim Lekadir, Oliver Diaz, Richard Osuala  

**Link**: [PDF](https://arxiv.org/pdf/2508.13776)  

**Abstract**: Dynamic contrast-enhanced (DCE) MRI is essential for breast cancer diagnosis and treatment. However, its reliance on contrast agents introduces safety concerns, contraindications, increased cost, and workflow complexity. To this end, we present pre-contrast conditioned denoising diffusion probabilistic models to synthesize DCE-MRI, introducing, evaluating, and comparing a total of 22 generative model variants in both single-breast and full breast settings. Towards enhancing lesion fidelity, we introduce both tumor-aware loss functions and explicit tumor segmentation mask conditioning. Using a public multicenter dataset and comparing to respective pre-contrast baselines, we observe that subtraction image-based models consistently outperform post-contrast-based models across five complementary evaluation metrics. Apart from assessing the entire image, we also separately evaluate the region of interest, where both tumor-aware losses and segmentation mask inputs improve evaluation metrics. The latter notably enhance qualitative results capturing contrast uptake, albeit assuming access to tumor localization inputs that are not guaranteed to be available in screening settings. A reader study involving 2 radiologists and 4 MRI technologists confirms the high realism of the synthetic images, indicating an emerging clinical potential of generative contrast-enhancement. We share our codebase at this https URL. 

**Abstract (ZH)**: 基于对比剂的磁共振成像(DCE-MRI)对于乳腺癌诊断和治疗至关重要。然而，其对对比剂的依赖引入了安全问题、禁忌症、成本增加和工作流程复杂性。为此，我们提出了预对比条件下的一种去噪扩散概率模型来合成DCE-MRI，共介绍了、评估和比较了22种生成模型变体，在单侧乳腺和全乳腺设置中进行了研究。为了增强病灶保真度，我们引入了肿瘤感知损失函数和显式的肿瘤分割掩码条件。使用公开的多中心数据集，并与相应的预对比基线进行比较，我们观察到，减影像基模型在五个互补评估指标中始终优于基于后对比的模型。除了评估整个影像外，我们还分别评估了感兴趣区域，在该区域中，肿瘤感知损失和分割掩码输入均能提高评估指标。后者显著提升了捕捉对比剂摄取的定性结果，尽管假定可获得肿瘤定位输入，而这些输入在筛查环境中并不总是可获得的。两位放射科医生和四位MRI技术人员的读者研究证实了合成影像的高度真实性，表明生成对比增强在临床中具有潜在应用价值。我们已在如下链接共享了我们的代码库：this https URL。 

---
# Agentic DraCor and the Art of Docstring Engineering: Evaluating MCP-empowered LLM Usage of the DraCor API 

**Title (ZH)**: 代理DraCor和文档字符串工程的艺术：评估MCP赋能的LLM对DraCor API的使用 

**Authors**: Peer Trilcke, Ingo Börner, Henny Sluyter-Gäthje, Daniil Skorinkin, Frank Fischer, Carsten Milling  

**Link**: [PDF](https://arxiv.org/pdf/2508.13774)  

**Abstract**: This paper reports on the implementation and evaluation of a Model Context Protocol (MCP) server for DraCor, enabling Large Language Models (LLM) to autonomously interact with the DraCor API. We conducted experiments focusing on tool selection and application by the LLM, employing a qualitative approach that includes systematic observation of prompts to understand how LLMs behave when using MCP tools, evaluating "Tool Correctness", "Tool-Calling Efficiency", and "Tool-Use Reliability". Our findings highlight the importance of "Docstring Engineering", defined as reflexively crafting tool documentation to optimize LLM-tool interaction. Our experiments demonstrate both the promise of agentic AI for research in Computational Literary Studies and the essential infrastructure development needs for reliable Digital Humanities infrastructures. 

**Abstract (ZH)**: 本研究报道了为DraCor实现并评估Model Context Protocol (MCP) 服务器，使大型语言模型（LLM）能够自主与DraCor API交互。我们通过定性的方法开展了实验，包括系统地观察提示，以了解LLM在使用MCP工具时的行为，评估“工具正确性”、“工具调用效率”和“工具使用可靠性”。研究结果强调了“文档字符串工程”的重要性，即反射性地设计工具文档以优化LLM与工具的交互。实验表明，自主智能在计算文学研究中的应用前景，并指出了可靠数字人文基础设施所需的基础架构开发需求。 

---
# PENGUIN: Enhancing Transformer with Periodic-Nested Group Attention for Long-term Time Series Forecasting 

**Title (ZH)**: PENGUIN：增强Transformer的周期嵌套组注意力机制以进行长期时间序列预测 

**Authors**: Tian Sun, Yuqi Chen, Weiwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13773)  

**Abstract**: Long-term time series forecasting (LTSF) is a fundamental task with wide-ranging applications. Although Transformer-based models have made significant breakthroughs in forecasting, their effectiveness for time series forecasting remains debatable. In this paper, we revisit the significance of self-attention and propose a simple yet effective mechanism, Periodic-Nested Group Attention, namely PENGUIN. Our approach highlights the importance of explicitly modeling periodic patterns and incorporating relative attention bias for effective time series modeling. To this end, we introduce a periodic-nested relative attention bias that captures periodic structures directly. To handle multiple coexisting periodicities (e.g., daily and weekly cycles), we design a grouped attention mechanism, where each group targets a specific periodicity using a multi-query attention mechanism. Extensive experiments across diverse benchmarks demonstrate that PENGUIN consistently outperforms both MLP-based and Transformer-based models. 

**Abstract (ZH)**: 长周期时间序列预测（LTSF）是一项具有广泛应用的基本任务。尽管基于Transformer的模型在预测方面取得了显著突破，但它们在时间序列预测中的有效性仍存争议。在本文中，我们重新审视了自注意力的重要性，并提出了一种简单而有效的机制，即周期嵌套组注意机制（PENGUIN）。我们的方法强调了明确建模周期性模式和引入相对注意偏见对于有效时间序列建模的重要性。为此，我们引入了一种周期嵌套的相对注意偏见，可以直接捕捉周期结构。为了处理多重共存的周期性（例如，日周期和周周期），我们设计了一种分组注意机制，其中每个组使用多查询注意机制针对特定的周期性。在多种基准上的广泛实验表明，PENGUIN一贯优于基于MLP和基于Transformer的模型。 

---
# COMPASS: A Multi-Dimensional Benchmark for Evaluating Code Generation in Large Language Models 

**Title (ZH)**: COMPASS: 一个评估大型语言模型代码生成多维度基准 

**Authors**: James Meaden, Michał Jarosz, Piotr Jodłowski, Grigori Melnik  

**Link**: [PDF](https://arxiv.org/pdf/2508.13757)  

**Abstract**: Current code generation benchmarks focus primarily on functional correctness while overlooking two critical aspects of real-world programming: algorithmic efficiency and code quality. We introduce COMPASS (COdility's Multi-dimensional Programming ASSessment), a comprehensive evaluation framework that assesses code generation across three dimensions: correctness, efficiency, and quality. COMPASS consists of 50 competitive programming problems from real Codility competitions, providing authentic human baselines from 393,150 submissions. Unlike existing benchmarks that treat algorithmically inefficient solutions identically to optimal ones provided they pass test cases, COMPASS systematically evaluates runtime efficiency and code quality using industry-standard analysis tools. Our evaluation of three leading reasoning-enhanced models, Anthropic Claude Opus 4, Google Gemini 2.5 Pro, and OpenAI O4-Mini-High, reveals that models achieving high correctness scores do not necessarily produce efficient algorithms or maintainable code. These findings highlight the importance of evaluating more than just correctness to truly understand the real-world capabilities of code generation models. COMPASS serves as a guiding framework, charting a path for future research toward AI systems that are robust, reliable, and ready for production use. 

**Abstract (ZH)**: COdility的多维度编程评估 COMPASS：综合评估框架 

---
# Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with Adaptive Exploration 

**Title (ZH)**: 深度与广度协同在RLVR中：自适应探索解锁预训练语言模型推理优势 

**Authors**: Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Dongchun Xie, Yiwei Wang, Xiaodan Liang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13755)  

**Abstract**: Reinforcement Learning with Verifiable Reward (RLVR) has emerged as a powerful paradigm for unlocking reasoning capabilities in large language models, yet its full potential is hindered by two under-explored dimensions: Depth-the hardest problem a model can sample; Breadth-the number of instances consumed in a single iteration. We dissect the popular GRPO algorithm and reveal a systematic bias: the cumulative-advantage disproportionately weights samples with medium accuracy, while down-weighting the low-accuracy instances that are crucial for pushing reasoning boundaries. To rectify the depth neglect, we introduce Difficulty Adaptive Rollout Sampling (DARS), which re-weights hard problems through targeted multi-stage rollouts, thereby increasing the number of positive rollouts for hard problems. Empirically, naively enlarging rollout size only accelerates convergence and even hurts Pass@K. Our DARS, in contrast, delivers consistent Pass@K gains without extra inference cost at convergence. Just as we adaptively expanded the depth of exploration, we now ask whether aggressively scaling the breadth of training data can further amplify reasoning gains. To this end, we intensely scale batch size and replace PPO's mini-batch iterations with full-batch updates over multiple epochs. Increasing breadth significantly enhances Pass@1 performance. Large-breadth training sustains high token-level entropy, indicating continued exploration and reduced gradient noise. We further present DARS-B, which augments DARS with large breadth, and demonstrate simultaneous gains in Pass@K and Pass@1. The results confirm that breadth and adaptive exploration across depth operate as orthogonal dimensions in RLVR, which are key to unleashing the reasoning power of RLVR. 

**Abstract (ZH)**: 验证奖励的增强学习（RLVR）：通过深度和广度解锁推理能力 

---
# Mitigating Cross-Image Information Leakage in LVLMs for Multi-Image Tasks 

**Title (ZH)**: 在多图像任务中缓解LVLMs之间的跨图像信息泄露 

**Authors**: Yeji Park, Minyoung Lee, Sanghyuk Chun, Junsuk Choe  

**Link**: [PDF](https://arxiv.org/pdf/2508.13744)  

**Abstract**: Large Vision-Language Models (LVLMs) demonstrate strong performance on single-image tasks. However, we observe that their performance degrades significantly when handling multi-image inputs. This occurs because visual cues from different images become entangled in the model's output. We refer to this phenomenon as cross-image information leakage. To address this issue, we propose FOCUS, a training-free and architecture-agnostic decoding strategy that mitigates cross-image information leakage during inference. FOCUS sequentially masks all but one image with random noise, guiding the model to focus on the single clean image. We repeat this process across all target images to obtain logits under partially masked contexts. These logits are aggregated and then contrastively refined using a noise-only reference input, which suppresses the leakage and yields more accurate outputs. FOCUS consistently improves performance across four multi-image benchmarks and diverse LVLM families. This demonstrates that FOCUS offers a general and practical solution for enhancing multi-image reasoning without additional training or architectural modifications. 

**Abstract (ZH)**: 大型视觉-语言模型（LVLMs）在单图任务上表现出色。然而，我们发现它们在处理多图输入时性能大幅下降。这是因为模型输出中包含了不同图像间的视觉线索纠缠现象。我们将这种现象称为跨图信息泄露。为了解决这一问题，我们提出了一种无需训练且架构无关的解码策略FOCUS，该策略在推理过程中减轻了跨图信息泄露。FOCUS通过用随机噪声掩蔽所有但一张干净图像的方式，引导模型专注于单一干净图像。我们对所有目标图像重复此过程，在部分掩蔽的上下文中获取逻辑值，这些逻辑值随后通过仅使用噪声参考输入进行对比性精炼，以抑制泄露并获得更准确的输出。FOCUS在四个多图基准和多种LVLM家族中一致提升了性能。这表明FOCUS提供了一种通用且实用的解决方案，可以在无额外训练或架构修改的情况下增强多图推理。 

---
# On the Security and Privacy of Federated Learning: A Survey with Attacks, Defenses, Frameworks, Applications, and Future Directions 

**Title (ZH)**: 联邦学习中的安全与隐私：攻击、防御、框架、应用及未来方向综述 

**Authors**: Daniel M. Jimenez-Gutierrez, Yelizaveta Falkouskaya, Jose L. Hernandez-Ramos, Aris Anagnostopoulos, Ioannis Chatzigiannakis, Andrea Vitaletti  

**Link**: [PDF](https://arxiv.org/pdf/2508.13730)  

**Abstract**: Federated Learning (FL) is an emerging distributed machine learning paradigm enabling multiple clients to train a global model collaboratively without sharing their raw data. While FL enhances data privacy by design, it remains vulnerable to various security and privacy threats. This survey provides a comprehensive overview of more than 200 papers regarding the state-of-the-art attacks and defense mechanisms developed to address these challenges, categorizing them into security-enhancing and privacy-preserving techniques. Security-enhancing methods aim to improve FL robustness against malicious behaviors such as byzantine attacks, poisoning, and Sybil attacks. At the same time, privacy-preserving techniques focus on protecting sensitive data through cryptographic approaches, differential privacy, and secure aggregation. We critically analyze the strengths and limitations of existing methods, highlight the trade-offs between privacy, security, and model performance, and discuss the implications of non-IID data distributions on the effectiveness of these defenses. Furthermore, we identify open research challenges and future directions, including the need for scalable, adaptive, and energy-efficient solutions operating in dynamic and heterogeneous FL environments. Our survey aims to guide researchers and practitioners in developing robust and privacy-preserving FL systems, fostering advancements safeguarding collaborative learning frameworks' integrity and confidentiality. 

**Abstract (ZH)**: 联邦学习(Federated Learning)是一种新兴的分布式机器学习范式，使多个客户端能够协作训练全球模型而无需共享其原始数据。尽管联邦学习通过设计增强了数据隐私性，但它仍易受到各种安全和隐私威胁。本文综述了超过200篇关于最新攻击和防御机制的研究论文，将这些研究论文分类为安全增强技术和隐私保护技术。安全增强技术旨在通过对抗拜占庭攻击、投毒攻击和Sybil攻击等恶意行为提高联邦学习的鲁棒性。同时，隐私保护技术侧重于通过加密方法、差分隐私和安全聚合等方式保护敏感数据。本文批判性地分析现有方法的优缺点，强调隐私、安全性和模型性能之间的权衡，并讨论非同态分布数据对这些防御措施有效性的影响。此外，本文指出了开放的研究挑战和未来方向，包括在动态和异构联邦学习环境中开发可扩展、自适应和能效性的解决方案的需求。本文旨在指导研究人员和实践者开发 robust 和隐私保护的联邦学习系统，促进保护协作学习框架完整性和保密性的进步。 

---
# Prediction is not Explanation: Revisiting the Explanatory Capacity of Mapping Embeddings 

**Title (ZH)**: 预测不等同于解释：重新审视映射嵌入的解释能力 

**Authors**: Hanna Herasimchyk, Alhassan Abdelhalim, Sören Laue, Michaela Regneri  

**Link**: [PDF](https://arxiv.org/pdf/2508.13729)  

**Abstract**: Understanding what knowledge is implicitly encoded in deep learning models is essential for improving the interpretability of AI systems. This paper examines common methods to explain the knowledge encoded in word embeddings, which are core elements of large language models (LLMs). These methods typically involve mapping embeddings onto collections of human-interpretable semantic features, known as feature norms. Prior work assumes that accurately predicting these semantic features from the word embeddings implies that the embeddings contain the corresponding knowledge. We challenge this assumption by demonstrating that prediction accuracy alone does not reliably indicate genuine feature-based interpretability.
We show that these methods can successfully predict even random information, concluding that the results are predominantly determined by an algorithmic upper bound rather than meaningful semantic representation in the word embeddings. Consequently, comparisons between datasets based solely on prediction performance do not reliably indicate which dataset is better captured by the word embeddings. Our analysis illustrates that such mappings primarily reflect geometric similarity within vector spaces rather than indicating the genuine emergence of semantic properties. 

**Abstract (ZH)**: 理解深度学习模型中隐含编码的知识对于提高人工智能系统的可解释性至关重要。本文探讨了用于解释词嵌入中编码知识的常见方法，这些方法是大规模语言模型（LLMs）的核心组成部分。这些方法通常涉及将嵌入映射到一组可解释的语义特征，即特征规范。先前的研究假设准确从词嵌入预测这些语义特征意味着嵌入包含了相应的知识。本文通过证明仅预测准确性不足以可靠地表明基于特征的可解释性来挑战这一假设。我们展示了这些方法甚至可以成功预测随机信息，从而得出结论，这些结果主要由算法上限决定，而不是词嵌入中的有意义的语义表示。因此，仅基于预测性能比较数据集不能可靠地表明哪个数据集被词嵌入更准确地捕捉。我们的分析表明，这类映射主要反映了向量空间内的几何相似性，而不是表明真正出现的语义属性。 

---
# Generics and Default Reasoning in Large Language Models 

**Title (ZH)**: 通用性与大型语言模型中的默认推理 

**Authors**: James Ravi Kirkpatrick, Rachel Katharine Sterken  

**Link**: [PDF](https://arxiv.org/pdf/2508.13718)  

**Abstract**: This paper evaluates the capabilities of 28 large language models (LLMs) to reason with 20 defeasible reasoning patterns involving generic generalizations (e.g., 'Birds fly', 'Ravens are black') central to non-monotonic logic. Generics are of special interest to linguists, philosophers, logicians, and cognitive scientists because of their complex exception-permitting behaviour and their centrality to default reasoning, cognition, and concept acquisition. We find that while several frontier models handle many default reasoning problems well, performance varies widely across models and prompting styles. Few-shot prompting modestly improves performance for some models, but chain-of-thought (CoT) prompting often leads to serious performance degradation (mean accuracy drop -11.14%, SD 15.74% in models performing above 75% accuracy in zero-shot condition, temperature 0). Most models either struggle to distinguish between defeasible and deductive inference or misinterpret generics as universal statements. These findings underscore both the promise and limits of current LLMs for default reasoning. 

**Abstract (ZH)**: 大型语言模型在攻防推理能力上的评估：基于泛涵非单调逻辑的攻防推理模式的研究 

---
# The AI Risk Spectrum: From Dangerous Capabilities to Existential Threats 

**Title (ZH)**: 人工智能风险谱：从危险能力到存在性威胁 

**Authors**: Markov Grey, Charbel-Raphaël Segerie  

**Link**: [PDF](https://arxiv.org/pdf/2508.13700)  

**Abstract**: As AI systems become more capable, integrated, and widespread, understanding the associated risks becomes increasingly important. This paper maps the full spectrum of AI risks, from current harms affecting individual users to existential threats that could endanger humanity's survival. We organize these risks into three main causal categories. Misuse risks, which occur when people deliberately use AI for harmful purposes - creating bioweapons, launching cyberattacks, adversarial AI attacks or deploying lethal autonomous weapons. Misalignment risks happen when AI systems pursue outcomes that conflict with human values, irrespective of developer intentions. This includes risks arising through specification gaming (reward hacking), scheming and power-seeking tendencies in pursuit of long-term strategic goals. Systemic risks, which arise when AI integrates into complex social systems in ways that gradually undermine human agency - concentrating power, accelerating political and economic disempowerment, creating overdependence that leads to human enfeeblement, or irreversibly locking in current values curtailing future moral progress. Beyond these core categories, we identify risk amplifiers - competitive pressures, accidents, corporate indifference, and coordination failures - that make all risks more likely and severe. Throughout, we connect today's existing risks and empirically observable AI behaviors to plausible future outcomes, demonstrating how existing trends could escalate to catastrophic outcomes. Our goal is to help readers understand the complete landscape of AI risks. Good futures are possible, but they don't happen by default. Navigating these challenges will require unprecedented coordination, but an extraordinary future awaits if we do. 

**Abstract (ZH)**: 随着AI系统变得更加卓越、集成化和普及化，理解相关风险变得越来越重要。本文映射了AI风险的完整谱系，从目前影响个别用户的伤害到可能危及人类生存的终结性威胁。我们将这些风险归类为三大主要因果类别。滥用风险，发生在人们故意将AI用于有害目的时——例如制造生物武器、发动网络攻击、对抗性AI攻击或部署致命自主武器。对齐风险发生在AI系统追求与人类价值观相冲突的结果时，无论开发者的意图如何。这包括因规范游戏（奖励劫持）、为长期战略目标追求权谋和权力追求而产生的风险。系统性风险，发生在AI以逐渐削弱人类自主权的方式整合到复杂的社会系统中时——权力集中、加速政治和经济去自主化、依赖性过强导致人类虚弱，或不可逆地锁定当前价值观，限制未来道德进步。除了这些核心类别之外，我们还识别出风险放大器——竞争压力、事故、企业漠视和协调失败——它们使所有风险更有可能发生且更加严重。在整个过程中，我们将当今已有的风险和可观察到的AI行为与合理的未来结果联系起来，演示现有趋势如何升级为灾难性结果。我们的目标是帮助读者了解AI风险的完整景观。光明的未来是可能的，但不会自动实现。应对这些挑战需要前所未有的协调，但如果能做到这一点，一个非凡的未来将等待着我们。 

---
# Multi-Plasticity Synergy with Adaptive Mechanism Assignment for Training Spiking Neural Networks 

**Title (ZH)**: 具有自适应机制分配的多塑性协同训练脉冲神经网络 

**Authors**: Yuzhe Liu, Xin Deng, Qiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13673)  

**Abstract**: Spiking Neural Networks (SNNs) are promising brain-inspired models known for low power consumption and superior potential for temporal processing, but identifying suitable learning mechanisms remains a challenge. Despite the presence of multiple coexisting learning strategies in the brain, current SNN training methods typically rely on a single form of synaptic plasticity, which limits their adaptability and representational capability. In this paper, we propose a biologically inspired training framework that incorporates multiple synergistic plasticity mechanisms for more effective SNN training. Our method enables diverse learning algorithms to cooperatively modulate the accumulation of information, while allowing each mechanism to preserve its own relatively independent update dynamics. We evaluated our approach on both static image and dynamic neuromorphic datasets to demonstrate that our framework significantly improves performance and robustness compared to conventional learning mechanism models. This work provides a general and extensible foundation for developing more powerful SNNs guided by multi-strategy brain-inspired learning. 

**Abstract (ZH)**: 基于多种协同可塑性机制的生物启发式Spiking神经网络训练框架 

---
# In-Context Decision Making for Optimizing Complex AutoML Pipelines 

**Title (ZH)**: 上下文决策优化复杂自动机器学习管道 

**Authors**: Amir Rezaei Balef, Katharina Eggensperger  

**Link**: [PDF](https://arxiv.org/pdf/2508.13657)  

**Abstract**: Combined Algorithm Selection and Hyperparameter Optimization (CASH) has been fundamental to traditional AutoML systems. However, with the advancements of pre-trained models, modern ML workflows go beyond hyperparameter optimization and often require fine-tuning, ensembling, and other adaptation techniques. While the core challenge of identifying the best-performing model for a downstream task remains, the increasing heterogeneity of ML pipelines demands novel AutoML approaches. This work extends the CASH framework to select and adapt modern ML pipelines. We propose PS-PFN to efficiently explore and exploit adapting ML pipelines by extending Posterior Sampling (PS) to the max k-armed bandit problem setup. PS-PFN leverages prior-data fitted networks (PFNs) to efficiently estimate the posterior distribution of the maximal value via in-context learning. We show how to extend this method to consider varying costs of pulling arms and to use different PFNs to model reward distributions individually per arm. Experimental results on one novel and two existing standard benchmark tasks demonstrate the superior performance of PS-PFN compared to other bandit and AutoML strategies. We make our code and data available at this https URL. 

**Abstract (ZH)**: Combined 算法选择与超参数优化 (CASH) 是传统自动化机器学习系统的核心。然而，随着预训练模型的发展，现代机器学习工作流程超越了超参数优化， often 而常需要微调、集成和其他适应技术。尽管确定下游任务最佳模型的核心挑战仍然存在，但日益异质的机器学习管道对新型自动化机器学习方法提出了需求。这项工作将 CASH 框架扩展到选择和适应现代机器学习管道。我们提出 PS-PFN 通过将后验采样 (PS) 扩展到最大 k- 赌徒臂问题设置中，以高效地探索和利用适应性机器学习管道。PS-PFN 利用先验-数据拟合网络 (PFNs) 通过情境学习高效估计最大值的后验分布。我们展示了如何扩展此方法考虑拉动不同臂的成本变化，并使用不同的 PFNs 分别对每个臂的奖励分布进行建模。在一项新颖的和两项现有标准基准任务上的实验结果表明，PS-PFN 在与其它多臂和自动化机器学习策略相比时表现更优。我们已将代码和数据公开于此 <https> URL。 

---
# Input Time Scaling 

**Title (ZH)**: 输入时间缩放 

**Authors**: Rapheal Huang, Weilong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.13654)  

**Abstract**: Current Large Language Models (LLMs) are usually post-trained on large-scale carefully curated datasets (data & training scaling) and doing reasoning in test time (inference time scaling). In this work, we present a new scaling paradigm, Input Time Scaling, to complement previous scaling methods by putting resources on queries (input time). During training and testing, we combine meta-knowledge from LLMs to refine inputs with different strategies. We also find a new phenomenon, training-testing co-design there. We need to apply query strategies during both training and testing. Only applying strategies on training or testing would seriously degrade the performance. We are also surprised to find that seemingly low data quality datasets can gain high performance. Adding irrelevant information to the queries, randomly selecting examples from a minimally filtered dataset, can even perform the best. These findings contradict the widely held inductive bias, "garbage in, garbage out". Curating datasets with seemingly high-quality data can even potentially limit the performance ceiling. In addition, models trained on more data with similar quality (15k VS 1k) perform worse, simple dataset size scaling should also be carefully inspected. The good news is that our findings are compatible with the Less is More phenomenon. A small set of examples is enough to evoke high-level reasoning ability. With experiments on models trained on Qwen2.5-32B-Instruct, we are able to reach SOTA performance among 32B models on AIME24(76.7%) and AIME25(76.7%) pass@1. We can further achieve AIME24(76.7%) and AIME25(80%) with a majority vote of three models. Starting from DeepSeek-R1-Distill-Qwen-32B, the best result would be 86.7% on AIME24 and 76.7% on AIME25. To facilitate reproducibility and further research, we are working on open-source our datasets, data pipelines, evaluation results, and checkpoints. 

**Abstract (ZH)**: 当前大型语言模型通常通过大规模精心筛选的数据集进行后训练（数据和训练缩放），并在测试时间进行推理（推理时间缩放）。本文提出了一种新的缩放范式——输入时间缩放，以补充之前的方法，将资源放在查询上（输入时间）。在训练和测试过程中，我们结合大模型的元知识，使用不同的策略细化输入。我们还发现了一种新的现象——训练-测试协同设计。在训练和测试过程中都需要应用查询策略，仅在训练或测试过程中应用策略会大幅降低性能。我们还惊讶地发现，虽然数据质量看似较低的数据集可以获得高性能。向查询中添加无关信息，从少量过滤的数据集随机选择示例，即使可以获得最佳性能。这些发现与广泛持有的归纳偏见“垃圾进，垃圾出”相矛盾。精心筛选看似高质量的数据集甚至可能限制性能上限。此外，训练数据量更多但质量相似（15k比1k）的模型表现更差，简单的数据集大小缩放也需要谨慎检查。好消息是我们发现的结果与“少即是多”现象是兼容的。少量示例足以引发高层次的推理能力。通过在Qwen2.5-32B-Instruct训练的模型上进行实验，我们能够在AIME24（76.7%）和AIME25（76.7%）的pass@1上达到SOTA性能。我们还可以通过三个模型的投票实现AIME24（76.7%）和AIME25（80%）。从DeepSeek-R1-Distill-Qwen-32B开始，最佳结果为AIME24（86.7%）和AIME25（76.7%）。为了便于再现性和进一步研究，我们正在开源我们的数据集、数据管道、评估结果和检查点。 

---
# GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling 

**Title (ZH)**: GRAFT： gradient-感知快速MaxVol动态数据采样方法 

**Authors**: Ashish Jha, Anh huy Phan, Razan Dibo, Valentin Leplat  

**Link**: [PDF](https://arxiv.org/pdf/2508.13653)  

**Abstract**: Training modern neural networks on large datasets is computationally and environmentally costly. We introduce GRAFT, a scalable in-training subset selection method that (i) extracts a low-rank feature representation for each batch, (ii) applies a Fast MaxVol sampler to select a small, diverse subset that spans the batch's dominant subspace, and (iii) dynamically adjusts the subset size using a gradient-approximation criterion. By operating in low-rank subspaces and training on carefully chosen examples instead of full batches, GRAFT preserves the training trajectory while reducing wall-clock time, energy consumption, and $\mathrm{CO}_2$ emissions. Across multiple benchmarks, GRAFT matches or exceeds recent selection baselines in both accuracy and efficiency, providing a favorable trade-off between accuracy, efficiency, and emissions. 

**Abstract (ZH)**: 在大规模数据集上训练现代神经网络既耗费计算资源又环保代价高。我们提出了一种可扩展的在训练过程中子集选择方法GRAFT，该方法通过(i) 为每个批次提取低秩特征表示，(ii) 使用快速MaxVol采样器选择一个小而多样的子集以覆盖批次的主要子空间，以及(iii) 使用梯度逼近准则动态调整子集大小，从而在低秩子空间中进行训练并在精心选择的样例上训练，来保留训练轨迹同时减少墙钟时间、能量消耗和$\mathrm{CO}_2$排放。在多个基准测试中，GRAFT在准确性和效率方面与最近的选样基线相当或超越，提供了一个在准确率、效率和排放之间有利的权衡。 

---
# Towards a Larger Model via One-Shot Federated Learning on Heterogeneous Client Models 

**Title (ZH)**: 基于异构客户端模型的一次性联邦学习通往更大模型 

**Authors**: Wenxuan Ye, Xueli An, Onur Ayan, Junfan Wang, Xueqiang Yan, Georg Carle  

**Link**: [PDF](https://arxiv.org/pdf/2508.13625)  

**Abstract**: Large models, renowned for superior performance, outperform smaller ones even without billion-parameter scales. While mobile network servers have ample computational resources to support larger models than client devices, privacy constraints prevent clients from directly sharing their raw data. Federated Learning (FL) enables decentralized clients to collaboratively train a shared model by exchanging model parameters instead of transmitting raw data. Yet, it requires a uniform model architecture and multiple communication rounds, which neglect resource heterogeneity, impose heavy computational demands on clients, and increase communication overhead. To address these challenges, we propose FedOL, to construct a larger and more comprehensive server model in one-shot settings (i.e., in a single communication round). Instead of model parameter sharing, FedOL employs knowledge distillation, where clients only exchange model prediction outputs on an unlabeled public dataset. This reduces communication overhead by transmitting compact predictions instead of full model weights and enables model customization by allowing heterogeneous model architectures. A key challenge in this setting is that client predictions may be biased due to skewed local data distributions, and the lack of ground-truth labels in the public dataset further complicates reliable learning. To mitigate these issues, FedOL introduces a specialized objective function that iteratively refines pseudo-labels and the server model, improving learning reliability. To complement this, FedOL incorporates a tailored pseudo-label generation and knowledge distillation strategy that effectively integrates diverse knowledge. Simulation results show that FedOL significantly outperforms existing baselines, offering a cost-effective solution for mobile networks where clients possess valuable private data but limited computational resources. 

**Abstract (ZH)**: Large 模型：无需 billion 参数规模亦能超越小型模型，即使在移动网络服务器拥有充足计算资源而客户端隐私受限无法直接共享原始数据的情况下，联邦学习 (FL) 通过交换模型参数而非传输原始数据，让去中心化的客户端协作训练共享模型。然而，FL 要求统一的模型架构和多轮通信，忽视了资源异构性，对客户端提出了沉重的计算要求，并增加了通信开销。为此，我们提出了 FedOL，以一次性（即单轮通信）构建一个更大、更全面的服务器模型。与模型参数共享不同，FedOL 采用知识蒸馏策略，客户端仅交换对未标注公开数据集的模型预测输出。这通过传输紧凑的预测而非完整模型权重减少了通信开销，并允许不同模型架构的定制。在这一设置中，一个关键挑战是客户端预测可能会由于本地数据分布偏差而失真，且公开数据集缺乏真实标签进一步增加了可靠学习的复杂性。为应对这些问题，FedOL 引入了一种专门的目标函数，迭代细化伪标签和服务器模型，从而提高学习可靠性。此外，FedOL 结合了定制化的伪标签生成和知识蒸馏策略，有效整合了多样化的知识。仿真实验结果表明，FedOL 在现有基线方法中表现显著优异，为拥有宝贵私有数据但计算资源有限的移动网络提供了一种成本效益高的解决方案。 

---
# Bounding Causal Effects and Counterfactuals 

**Title (ZH)**: 因果效应和反事实推理的界限 

**Authors**: Tobias Maringgele  

**Link**: [PDF](https://arxiv.org/pdf/2508.13607)  

**Abstract**: Causal inference often hinges on strong assumptions - such as no unmeasured confounding or perfect compliance - that are rarely satisfied in practice. Partial identification offers a principled alternative: instead of relying on unverifiable assumptions to estimate causal effects precisely, it derives bounds that reflect the uncertainty inherent in the data. Despite its theoretical appeal, partial identification remains underutilized in applied work, in part due to the fragmented nature of existing methods and the lack of practical guidance. This thesis addresses these challenges by systematically comparing a diverse set of bounding algorithms across multiple causal scenarios. We implement, extend, and unify state-of-the-art methods - including symbolic, optimization-based, and information-theoretic approaches - within a common evaluation framework. In particular, we propose an extension of a recently introduced entropy-bounded method, making it applicable to counterfactual queries such as the Probability of Necessity and Sufficiency (PNS). Our empirical study spans thousands of randomized simulations involving both discrete and continuous data-generating processes. We assess each method in terms of bound tightness, computational efficiency, and robustness to assumption violations. To support practitioners, we distill our findings into a practical decision tree for algorithm selection and train a machine learning model to predict the best-performing method based on observable data characteristics.
All implementations are released as part of an open-source Python package, CausalBoundingEngine, which enables users to apply and compare bounding methods through a unified interface. 

**Abstract (ZH)**: 因果推理往往依赖于一些假设，未测量混杂或完全遵守上，在实践中通常无法满足。部分识别提供了一种原则性的替代方案：而不是依赖于无法验证的假设来精确估计因果效应，，而给出反映数据内在不确定性的边界估计。尽管如此，理论上的的识别仍然在实践中被广泛忽视，原因之一在于现有的文献碎片化且缺乏实用指导上。本文旨在通过系统地地比较多种边界算法在多种因果情景上的表现来克服这些挑战。我们实现了对基于符号的优化方法和概率论的方法的统一，并在统一的评估框架上上提出了一个改进的基于熵的界限方法，使其适用于诸如需要- 和充分条件 -概率查询（PNS）这类的反事实查询。我们的实证研究覆盖了成千个随机化设定，涉及离散和部分观测数据获取生成过程。我们从各方法的角度出发评估了边界估计的紧致性、计算效率以及假设不成立时的鲁棒性性。为了指导从业者作出选择，我们总结了发现成果了可作决策树并开发了一个机器学习实践来预测在特定查询特征基础上的最佳表现。 

---
# Who Gets the Mic? Investigating Gender Bias in the Speaker Assignment of a Speech-LLM 

**Title (ZH)**: 谁来发声？探究演讲-LLM中发言者分配的性别偏见 

**Authors**: Dariia Puhach, Amir H. Payberah, Éva Székely  

**Link**: [PDF](https://arxiv.org/pdf/2508.13603)  

**Abstract**: Similar to text-based Large Language Models (LLMs), Speech-LLMs exhibit emergent abilities and context awareness. However, whether these similarities extend to gender bias remains an open question. This study proposes a methodology leveraging speaker assignment as an analytic tool for bias investigation. Unlike text-based models, which encode gendered associations implicitly, Speech-LLMs must produce a gendered voice, making speaker selection an explicit bias cue. We evaluate Bark, a Text-to-Speech (TTS) model, analyzing its default speaker assignments for textual prompts. If Bark's speaker selection systematically aligns with gendered associations, it may reveal patterns in its training data or model design. To test this, we construct two datasets: (i) Professions, containing gender-stereotyped occupations, and (ii) Gender-Colored Words, featuring gendered connotations. While Bark does not exhibit systematic bias, it demonstrates gender awareness and has some gender inclinations. 

**Abstract (ZH)**: 类似于文本基础的大语言模型（LLMs），语音LLMs表现出 emergent 能力和背景意识。然而，这些相似性是否扩展到性别偏差仍是一个开放的问题。本研究提出了一种利用说话人分配作为偏差调查分析工具的方法。与隐含编码性别关联的文本基础模型不同，语音LLMs必须生成带性别的声音，使说话人选择成为显式的偏差提示。我们评估了Bark这一文本转语音（TTS）模型，分析其对文本提示的默认说话人分配。如果Bark的说话人选择系统地与性别关联一致，这可能揭示其训练数据或模型设计中的模式。为了测试这一点，我们构建了两个数据集：(i) 职业，包含性别刻板印象的职业，以及(ii) 性别色彩词汇，包含性别联想。尽管Bark没有表现出系统性的偏差，但它表现出性别意识，并且具有某些性别倾向。 

---
# A Comparative Study of Decoding Strategies in Medical Text Generation 

**Title (ZH)**: 医疗文本生成中解码策略的比较研究 

**Authors**: Oriana Presacan, Alireza Nik, Vajira Thambawita, Bogdan Ionescu, Michael Riegler  

**Link**: [PDF](https://arxiv.org/pdf/2508.13580)  

**Abstract**: Large Language Models (LLMs) rely on various decoding strategies to generate text, and these choices can significantly affect output quality. In healthcare, where accuracy is critical, the impact of decoding strategies remains underexplored. We investigate this effect in five open-ended medical tasks, including translation, summarization, question answering, dialogue, and image captioning, evaluating 11 decoding strategies with medically specialized and general-purpose LLMs of different sizes. Our results show that deterministic strategies generally outperform stochastic ones: beam search achieves the highest scores, while {\eta} and top-k sampling perform worst. Slower decoding methods tend to yield better quality. Larger models achieve higher scores overall but have longer inference times and are no more robust to decoding. Surprisingly, while medical LLMs outperform general ones in two of the five tasks, statistical analysis shows no overall performance advantage and reveals greater sensitivity to decoding choice. We further compare multiple evaluation metrics and find that correlations vary by task, with MAUVE showing weak agreement with BERTScore and ROUGE, as well as greater sensitivity to the decoding strategy. These results highlight the need for careful selection of decoding methods in medical applications, as their influence can sometimes exceed that of model choice. 

**Abstract (ZH)**: 大型语言模型（LLMs）依赖各种解码策略生成文本，这些选择会显著影响输出质量。在对准确性要求极高的医疗领域，解码策略的影响依然未被充分探索。我们在五个开放性医疗任务中进行了调查，包括翻译、总结、问答、对话和图像字幕，评估了11种解码策略在不同规模的医学专用和通用大语言模型上的效果。结果显示，确定性策略通常优于随机策略：束搜索获得最高分，而η和top-k采样表现最差。较慢的解码方法倾向于生成更好的质量。更大规模的模型总体上获得更高分，但推断时间更长且对解码策略的鲁棒性无显著提高。令人惊讶的是，尽管在五个任务中有两个任务中医学大语言模型表现优于通用模型，但统计分析显示其整体性能优势并不明显，并且对解码选择更为敏感。我们进一步比较了多种评估指标，发现它们在不同任务中的相关性各异，MAUVE与BERTScore和ROUGE的相关性较弱，并且对解码策略更为敏感。这些结果强调了在医疗应用中仔细选择解码方法的必要性，因为其影响有时可能超过模型选择的影响。 

---
# End-to-End Audio-Visual Learning for Cochlear Implant Sound Coding in Noisy Environments 

**Title (ZH)**: 噪声环境中的端到端音频-视觉学习在植入式耳蜗声音编码中的应用 

**Authors**: Meng-Ping Lin, Enoch Hsin-Ho Huang, Shao-Yi Chien, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13576)  

**Abstract**: The cochlear implant (CI) is a remarkable biomedical device that successfully enables individuals with severe-to-profound hearing loss to perceive sound by converting speech into electrical stimulation signals. Despite advancements in the performance of recent CI systems, speech comprehension in noisy or reverberant conditions remains a challenge. Recent and ongoing developments in deep learning reveal promising opportunities for enhancing CI sound coding capabilities, not only through replicating traditional signal processing methods with neural networks, but also through integrating visual cues as auxiliary data for multimodal speech processing. Therefore, this paper introduces a novel noise-suppressing CI system, AVSE-ECS, which utilizes an audio-visual speech enhancement (AVSE) model as a pre-processing module for the deep-learning-based ElectrodeNet-CS (ECS) sound coding strategy. Specifically, a joint training approach is applied to model AVSE-ECS, an end-to-end CI system. Experimental results indicate that the proposed method outperforms the previous ECS strategy in noisy conditions, with improved objective speech intelligibility scores. The methods and findings in this study demonstrate the feasibility and potential of using deep learning to integrate the AVSE module into an end-to-end CI system 

**Abstract (ZH)**: 基于音频-视觉增强的深度学习 cochlear implant 系统：AVSE-ECS 

---
# The 9th AI City Challenge 

**Title (ZH)**: 第九届AI城市挑战赛 

**Authors**: Zheng Tang, Shuo Wang, David C. Anastasiu, Ming-Ching Chang, Anuj Sharma, Quan Kong, Norimasa Kobori, Munkhjargal Gochoo, Ganzorig Batnasan, Munkh-Erdene Otgonbold, Fady Alnajjar, Jun-Wei Hsieh, Tomasz Kornuta, Xiaolong Li, Yilin Zhao, Han Zhang, Subhashree Radhakrishnan, Arihant Jain, Ratnesh Kumar, Vidya N. Murali, Yuxing Wang, Sameer Satish Pusegaonkar, Yizhou Wang, Sujit Biswas, Xunlei Wu, Zhedong Zheng, Pranamesh Chakraborty, Rama Chellappa  

**Link**: [PDF](https://arxiv.org/pdf/2508.13564)  

**Abstract**: The ninth AI City Challenge continues to advance real-world applications of computer vision and AI in transportation, industrial automation, and public safety. The 2025 edition featured four tracks and saw a 17% increase in participation, with 245 teams from 15 countries registered on the evaluation server. Public release of challenge datasets led to over 30,000 downloads to date. Track 1 focused on multi-class 3D multi-camera tracking, involving people, humanoids, autonomous mobile robots, and forklifts, using detailed calibration and 3D bounding box annotations. Track 2 tackled video question answering in traffic safety, with multi-camera incident understanding enriched by 3D gaze labels. Track 3 addressed fine-grained spatial reasoning in dynamic warehouse environments, requiring AI systems to interpret RGB-D inputs and answer spatial questions that combine perception, geometry, and language. Both Track 1 and Track 3 datasets were generated in NVIDIA Omniverse. Track 4 emphasized efficient road object detection from fisheye cameras, supporting lightweight, real-time deployment on edge devices. The evaluation framework enforced submission limits and used a partially held-out test set to ensure fair benchmarking. Final rankings were revealed after the competition concluded, fostering reproducibility and mitigating overfitting. Several teams achieved top-tier results, setting new benchmarks in multiple tasks. 

**Abstract (ZH)**: 第九届AI城市挑战赛继续推动计算机视觉和人工智能在交通、工业自动化和公共安全领域的实际应用。2025年版设立了四个赛道，参赛队伍增加了17%，共有来自15个国家的245支队伍在评估服务器上注册。挑战赛数据集的公开发布迄今已超过30,000次下载。赛道1专注于多类3D多摄像头跟踪，涉及行人、类人机器人、自主移动机器人和叉车，使用详细的标定和3D边界框注释。赛道2解决交通安全性中的视频问答问题，通过3D凝视标签增强多摄像头事件理解。赛道3解决动态仓库环境中的细粒度空间推理问题，需要人工智能系统解释RGB-D输入并回答结合感知、几何和语言的空间问题。赛道1和赛道3的数据集均在NVIDIA Omniverse中生成。赛道4强调从鱼眼摄像头高效检测道路对象，支持在边缘设备上进行轻量级、实时部署。评估框架设置了提交限制，并使用部分保留的测试集确保公平基准测试。比赛结束后公布最终排名，促进可重复性并减轻过拟合问题。多支队伍取得了顶尖成绩，多个任务中设立了新的基准。 

---
# Physics-Informed Neural Networks for Programmable Origami Metamaterials with Controlled Deployment 

**Title (ZH)**: 基于物理信息的神经网络在可控展开的可编程 Origami 超材料中的应用 

**Authors**: Sukheon Kang, Youngkwon Kim, Jinkyu Yang, Seunghwa Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13559)  

**Abstract**: Origami-inspired structures provide unprecedented opportunities for creating lightweight, deployable systems with programmable mechanical responses. However, their design remains challenging due to complex nonlinear mechanics, multistability, and the need for precise control of deployment forces. Here, we present a physics-informed neural network (PINN) framework for both forward prediction and inverse design of conical Kresling origami (CKO) without requiring pre-collected training data. By embedding mechanical equilibrium equations directly into the learning process, the model predicts complete energy landscapes with high accuracy while minimizing non-physical artifacts. The inverse design routine specifies both target stable-state heights and separating energy barriers, enabling freeform programming of the entire energy curve. This capability is extended to hierarchical CKO assemblies, where sequential layer-by-layer deployment is achieved through programmed barrier magnitudes. Finite element simulations and experiments on physical prototypes validate the designed deployment sequences and barrier ratios, confirming the robustness of the approach. This work establishes a versatile, data-free route for programming complex mechanical energy landscapes in origami-inspired metamaterials, offering broad potential for deployable aerospace systems, morphing structures, and soft robotic actuators. 

**Abstract (ZH)**: Origami-Inspired 结构提供的锥形克雷尔折纸 (CKO) 的正向预测和逆向设计的物理知情神经网络框架无需预先收集训练数据提供了前所未有的机会，以创建具有可编程机械响应的轻量化、可展开系统。然而，由于复杂的非线性力学、多稳定性和部署力的精确控制需求，其设计仍具有挑战性。在这里，我们提出了一种物理知情神经网络 (PINN) 框架，用于锥形克雷尔折纸 (CKO) 的正向预测和逆向设计，无需预先收集训练数据。通过直接将机械平衡方程嵌入学习过程，该模型以高度准确的方式预测完整能量景观，同时最小化非物理伪影。逆向设计流程既规定目标稳定状态高度又规定分隔能量障碍，从而实现整个能量曲线的自由编程。这一能力扩展到了分层 CKO 组装中，通过程序化障碍幅度实现了逐层展开。有限元仿真和物理原型上的实验验证了设计的展开序列和障碍比值，证实了该方法的稳健性。这项工作为编程 origami 启发式 metamaterial 中复杂的机械能量景观提供了一种通用且无需数据的路径，为可展开航空航天系统、形态可变结构和软体机器人执行器提供了广泛潜力。 

---
# Collapsing ROC approach for risk prediction research on both common and rare variants 

**Title (ZH)**: 共同变异与罕见变异的联合风险预测ROC坍缩方法 

**Authors**: Changshuai Wei, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13552)  

**Abstract**: Risk prediction that capitalizes on emerging genetic findings holds great promise for improving public health and clinical care. However, recent risk prediction research has shown that predictive tests formed on existing common genetic loci, including those from genome-wide association studies, have lacked sufficient accuracy for clinical use. Because most rare variants on the genome have not yet been studied for their role in risk prediction, future disease prediction discoveries should shift toward a more comprehensive risk prediction strategy that takes into account both common and rare variants. We are proposing a collapsing receiver operating characteristic CROC approach for risk prediction research on both common and rare variants. The new approach is an extension of a previously developed forward ROC FROC approach, with additional procedures for handling rare variants. The approach was evaluated through the use of 533 single-nucleotide polymorphisms SNPs in 37 candidate genes from the Genetic Analysis Workshop 17 mini-exome data set. We found that a prediction model built on all SNPs gained more accuracy AUC = 0.605 than one built on common variants alone AUC = 0.585. We further evaluated the performance of two approaches by gradually reducing the number of common variants in the analysis. We found that the CROC method attained more accuracy than the FROC method when the number of common variants in the data decreased. In an extreme scenario, when there are only rare variants in the data, the CROC reached an AUC value of 0.603, whereas the FROC had an AUC value of 0.524. 

**Abstract (ZH)**: 标题：基于新兴遗传发现的风险预测：一种综合罕见变异的坍缩受试者操作特征（CROC）方法 

---
# FLAIR: Frequency- and Locality-Aware Implicit Neural Representations 

**Title (ZH)**: FLAIR: 频率和局部性意识的隐式神经表示 

**Authors**: Sukhun Ko, Dahyeon Kye, Kyle Min, Chanho Eom, Jihyong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2508.13544)  

**Abstract**: Implicit Neural Representations (INRs) leverage neural networks to map coordinates to corresponding signals, enabling continuous and compact representations. This paradigm has driven significant advances in various vision tasks. However, existing INRs lack frequency selectivity, spatial localization, and sparse representations, leading to an over-reliance on redundant signal components. Consequently, they exhibit spectral bias, tending to learn low-frequency components early while struggling to capture fine high-frequency details. To address these issues, we propose FLAIR (Frequency- and Locality-Aware Implicit Neural Representations), which incorporates two key innovations. The first is RC-GAUSS, a novel activation designed for explicit frequency selection and spatial localization under the constraints of the time-frequency uncertainty principle (TFUP). The second is Wavelet-Energy-Guided Encoding (WEGE), which leverages the discrete wavelet transform (DWT) to compute energy scores and explicitly guide frequency information to the network. Our method consistently outperforms existing INRs in 2D image representation and restoration, as well as 3D reconstruction. 

**Abstract (ZH)**: 频率和局部性aware隐式神经表示（FLAIR）：频率选择和空间局部化的新型激活与小波能量引导编码 

---
# EAvatar: Expression-Aware Head Avatar Reconstruction with Generative Geometry Priors 

**Title (ZH)**: EAvatar：带有生成几何先验的表达意识头部avatar重建 

**Authors**: Shikun Zhang, Cunjian Chen, Yiqun Wang, Qiuhong Ke, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13537)  

**Abstract**: High-fidelity head avatar reconstruction plays a crucial role in AR/VR, gaming, and multimedia content creation. Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated effectiveness in modeling complex geometry with real-time rendering capability and are now widely used in high-fidelity head avatar reconstruction tasks. However, existing 3DGS-based methods still face significant challenges in capturing fine-grained facial expressions and preserving local texture continuity, especially in highly deformable regions. To mitigate these limitations, we propose a novel 3DGS-based framework termed EAvatar for head reconstruction that is both expression-aware and deformation-aware. Our method introduces a sparse expression control mechanism, where a small number of key Gaussians are used to influence the deformation of their neighboring Gaussians, enabling accurate modeling of local deformations and fine-scale texture transitions. Furthermore, we leverage high-quality 3D priors from pretrained generative models to provide a more reliable facial geometry, offering structural guidance that improves convergence stability and shape accuracy during training. Experimental results demonstrate that our method produces more accurate and visually coherent head reconstructions with improved expression controllability and detail fidelity. 

**Abstract (ZH)**: 高保真头部 avatar 重建在 AR/VR、游戏和多媒体内容创作中发挥着关键作用。基于 3D 高斯点绘制（3DGS）的Recent进展显示出在实时渲染能力下模拟复杂几何形状的有效性，并且现在被广泛应用于高保真头部 avatar 重建任务中。然而，现有的基于 3DGS 的方法在捕捉细微表情变化和保持局部纹理连续性方面仍然面临重大挑战，特别是在高度可变形区域。为缓解这些限制，我们提出了一种名为 EAvatar 的新颖基于 3DGS 的框架，该框架既具备表情感知能力又具备变形感知能力。我们的方法引入了一种稀疏表情控制机制，使用少量关键高斯点影响邻近高斯点的变形，以实现局部变形和精细尺度纹理过渡的准确建模。此外，我们利用预训练生成模型提供的高质量 3D 先验知识，提供更可靠的面部几何结构，从结构上指导训练以提高收敛稳定性和形状准确性。实验结果表明，我们的方法生成了更准确且视觉连贯的头部重建，同时提高了表情可控性和细节保真度。 

---
# MimicFunc: Imitating Tool Manipulation from a Single Human Video via Functional Correspondence 

**Title (ZH)**: MimicFunc: 从单个人类视频中模仿工具操作的函数对应方法 

**Authors**: Chao Tang, Anxing Xiao, Yuhong Deng, Tianrun Hu, Wenlong Dong, Hanbo Zhang, David Hsu, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13534)  

**Abstract**: Imitating tool manipulation from human videos offers an intuitive approach to teaching robots, while also providing a promising and scalable alternative to labor-intensive teleoperation data collection for visuomotor policy learning. While humans can mimic tool manipulation behavior by observing others perform a task just once and effortlessly transfer the skill to diverse tools for functionally equivalent tasks, current robots struggle to achieve this level of generalization. A key challenge lies in establishing function-level correspondences, considering the significant geometric variations among functionally similar tools, referred to as intra-function variations. To address this challenge, we propose MimicFunc, a framework that establishes functional correspondences with function frame, a function-centric local coordinate frame constructed with keypoint-based abstraction, for imitating tool manipulation skills. Experiments demonstrate that MimicFunc effectively enables the robot to generalize the skill from a single RGB-D human video to manipulating novel tools for functionally equivalent tasks. Furthermore, leveraging MimicFunc's one-shot generalization capability, the generated rollouts can be used to train visuomotor policies without requiring labor-intensive teleoperation data collection for novel objects. Our code and video are available at this https URL. 

**Abstract (ZH)**: 从人类视频中模仿工具操作为机器人教学提供了直观的方法，并且为视觉运动策略学习提供了富有潜力且可扩展的替代方案，以减少劳动密集型的遥操作数据收集。尽管人类可以通过观察他人一次完成任务并轻松地将技能转移到功能等效的不同工具上，当前的机器人却难以达到这一泛化水平。一个关键挑战在于建立功能层面的对应关系，考虑到功能相似工具之间存在的显著几何变化，即同功能内变化。为了解决这一挑战，我们提出了MimicFunc框架，该框架利用以关键点为基础的抽象构建的功能中心局部坐标框架（function frame）来建立功能对应关系，以便模仿工具操作技能。实验表明，MimicFunc能够有效地使机器人能够从单个RGB-D人类视频中泛化技能，以操作新型工具进行功能等效任务。此外，借助MimicFunc的一次性泛化能力，生成的轨迹可以用于训练视觉运动策略，而无需为新型对象进行劳动密集型的遥操作数据收集。我们的代码和视频可在以下链接获取。 

---
# Evaluating Open-Source Vision Language Models for Facial Emotion Recognition against Traditional Deep Learning Models 

**Title (ZH)**: 评估开源视觉语言模型在面部情绪识别任务中与传统深度学习模型的性能对比 

**Authors**: Vamsi Krishna Mulukutla, Sai Supriya Pavarala, Srinivasa Raju Rudraraju, Sridevi Bonthu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13524)  

**Abstract**: Facial Emotion Recognition (FER) is crucial for applications such as human-computer interaction and mental health diagnostics. This study presents the first empirical comparison of open-source Vision-Language Models (VLMs), including Phi-3.5 Vision and CLIP, against traditional deep learning models VGG19, ResNet-50, and EfficientNet-B0 on the challenging FER-2013 dataset, which contains 35,887 low-resolution grayscale images across seven emotion classes. To address the mismatch between VLM training assumptions and the noisy nature of FER data, we introduce a novel pipeline that integrates GFPGAN-based image restoration with FER evaluation. Results show that traditional models, particularly EfficientNet-B0 (86.44%) and ResNet-50 (85.72%), significantly outperform VLMs like CLIP (64.07%) and Phi-3.5 Vision (51.66%), highlighting the limitations of VLMs in low-quality visual tasks. In addition to performance evaluation using precision, recall, F1-score, and accuracy, we provide a detailed computational cost analysis covering preprocessing, training, inference, and evaluation phases, offering practical insights for deployment. This work underscores the need for adapting VLMs to noisy environments and provides a reproducible benchmark for future research in emotion recognition. 

**Abstract (ZH)**: 开放源代码视觉-语言模型在FER-2013数据集上的面部情感识别对比研究：适应噪声环境的必要性与可重复基准 

---
# DDoS Attacks in Cloud Computing: Detection and Prevention 

**Title (ZH)**: 云 computing 中的 DDoS 攻击：检测与防范 

**Authors**: Zain Ahmad, Musab Ahmad, Bilal Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2508.13522)  

**Abstract**: DDoS attacks are one of the most prevalent and harmful cybersecurity threats faced by organizations and individuals today. In recent years, the complexity and frequency of DDoS attacks have increased significantly, making it challenging to detect and mitigate them effectively. The study analyzes various types of DDoS attacks, including volumetric, protocol, and application layer attacks, and discusses the characteristics, impact, and potential targets of each type. It also examines the existing techniques used for DDoS attack detection, such as packet filtering, intrusion detection systems, and machine learning-based approaches, and their strengths and limitations. Moreover, the study explores the prevention techniques employed to mitigate DDoS attacks, such as firewalls, rate limiting , CPP and ELD mechanism. It evaluates the effectiveness of each approach and its suitability for different types of attacks and environments. In conclusion, this study provides a comprehensive overview of the different types of DDoS attacks, their detection, and prevention techniques. It aims to provide insights and guidelines for organizations and individuals to enhance their cybersecurity posture and protect against DDoS attacks. 

**Abstract (ZH)**: DDoS攻击是组织和个人当前面临的最常见和最具危害性的网络安全威胁之一。近年来，DDoS攻击的复杂性和频率显著增加，给有效检测和缓解带来了挑战。本研究分析了各种类型的DDoS攻击，包括 volumetric、协议和应用层攻击，并讨论了每种类型的特点、影响和潜在目标。研究还考察了现有的DDoS攻击检测技术，如包过滤、入侵检测系统和基于机器学习的方法，及其优缺点。此外，研究探讨了用于缓解DDoS攻击的预防技术，如防火墙、速率限制、CPP和ELD机制，并评估了每种方法的有效性和适用性。最后，本研究提供了一种全面的DDoS攻击类型、检测和预防技术概述，旨在为组织和个人提供增强网络安全态势和抵御DDoS攻击的见解和指导。 

---
# Calibrating Biased Distribution in VFM-derived Latent Space via Cross-Domain Geometric Consistency 

**Title (ZH)**: 基于跨域几何一致性校准由VFM衍生的偏置分布的潜空间 

**Authors**: Yanbiao Ma, Wei Dai, Bowei Liu, Jiayi Chen, Wenke Huang, Guancheng Wan, Zhiwu Lu, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13518)  

**Abstract**: Despite the fast progress of deep learning, one standing challenge is the gap of the observed training samples and the underlying true distribution. There are multiple reasons for the causing of this gap e.g. sampling bias, noise etc. In the era of foundation models, we show that when leveraging the off-the-shelf (vision) foundation models (e.g., CLIP, DINOv2) for feature extraction, the geometric shapes of the resulting feature distributions exhibit remarkable transferability across domains and datasets. To verify its practical usefulness, we embody our geometric knowledge-guided distribution calibration framework in two popular and challenging settings: federated learning and long-tailed recognition. In the federated setting, we devise a technique of acquiring the global geometric shape under privacy constraints, then leverage this knowledge to generate new samples for clients, in the aim of bridging the gap between local and global observations. In long-tailed learning, it utilizes the geometric knowledge transferred from sample-rich categories to recover the true distribution for sample-scarce tail classes. Comprehensive experiments show that our proposed geometric knowledge-guided distribution calibration effectively overcomes information deficits caused by data heterogeneity and sample imbalance, with boosted performance across benchmarks. 

**Abstract (ZH)**: 尽管深度学习取得了快速发展，但存在的一个主要挑战是观察到的训练样本与底层真实分布之间的差距。这种差距的原因多种多样，例如采样偏差和噪声等。在基础模型时代，我们展示了利用现成（视觉）基础模型（如CLIP、DINOv2）进行特征提取时，结果特征分布的几何形状在不同领域和数据集中表现出显著的可移植性。为了验证其实用性，我们将几何知识引导的分布校准框架应用于两个流行的具有挑战性的场景：联邦学习和长尾识别。在联邦学习场景中，我们提出了一种在隐私约束下获取全局几何形状的技术，然后利用这些知识生成新的样本，以弥合局部和全局观察之间的差距。在长尾学习中，它利用从样本丰富的类别转移到样本稀缺的尾部类别的几何知识来恢复真实分布。全面的实验表明，我们提出的方法有效克服了由于数据异构性和样本不平衡引起的信息不足，提升了多个基准测试中的性能。 

---
# Heterogeneous Influence Maximization in User Recommendation 

**Title (ZH)**: 异质用户影响最大化推荐 

**Authors**: Hongru Hou, Jiachen Sun, Wenqing Lin, Wendong Bi, Xiangrong Wang, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13517)  

**Abstract**: User recommendation systems enhance user engagement by encouraging users to act as inviters to interact with other users (invitees), potentially fostering information propagation. Conventional recommendation methods typically focus on modeling interaction willingness. Influence-Maximization (IM) methods focus on identifying a set of users to maximize the information propagation. However, existing methods face two significant challenges. First, recommendation methods fail to unleash the candidates' spread capability. Second, IM methods fail to account for the willingness to interact. To solve these issues, we propose two models named HeteroIR and HeteroIM. HeteroIR provides an intuitive solution to unleash the dissemination potential of user recommendation systems. HeteroIM fills the gap between the IM method and the recommendation task, improving interaction willingness and maximizing spread coverage. The HeteroIR introduces a two-stage framework to estimate the spread profits. The HeteroIM incrementally selects the most influential invitee to recommend and rerank based on the number of reverse reachable (RR) sets containing inviters and invitees. RR set denotes a set of nodes that can reach a target via propagation. Extensive experiments show that HeteroIR and HeteroIM significantly outperform the state-of-the-art baselines with the p-value < 0.05. Furthermore, we have deployed HeteroIR and HeteroIM in Tencent's online gaming platforms and gained an 8.5\% and 10\% improvement in the online A/B test, respectively. Implementation codes are available at this https URL. 

**Abstract (ZH)**: 用户推荐系统通过鼓励用户作为推荐者与其他人（被推荐者）互动，增强用户参与度， potentially 促进信息传播。传统的推荐方法通常专注于建模互动意愿。影响最大化（IM）方法专注于识别一组用户以最大化信息传播。然而，现有的方法面临着两个显著的挑战：首先，推荐方法未能释放候选者的传播能力；其次，IM方法未能考虑互动意愿。为了解决这些问题，我们提出了两个模型，分别名为HeteroIR和HeteroIM。HeteroIR提供了一种直观的解决方案，以释放用户推荐系统的传播潜力。HeteroIM在IM方法与推荐任务之间填补了空白，提高了互动意愿并最大化传播覆盖范围。HeteroIR引入了一种两阶段框架来估计传播收益。HeteroIM基于包含推荐者和被推荐者的逆可达集（RR）的数量，逐步选择最具影响力的被推荐者进行推荐并重新排序。逆可达集指的是可以通过传播到达目标的节点集。广泛的实验表明，与最先进的基线方法相比，HeteroIR和HeteroIM的表现显著优越，p值<0.05。此外，我们在腾讯的在线游戏平台上部署了HeteroIR和HeteroIM，并分别在在线A/B测试中获得了8.5%和10%的提升。相关实施代码可在以下链接获取。 

---
# ProMed: Shapley Information Gain Guided Reinforcement Learning for Proactive Medical LLMs 

**Title (ZH)**: ProMed: 由Shapley信息增益引导的主动医疗LLM增强学习 

**Authors**: Hongxin Ding, Baixiang Huang, Yue Fang, Weibin Liao, Xinke Jiang, Zheng Li, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13514)  

**Abstract**: Interactive medical questioning is essential in real-world clinical consultations, where physicians must actively gather information from patients. While medical Large Language Models (LLMs) have shown impressive capabilities in static medical question answering, they predominantly operate under a reactive paradigm: generating answers directly without seeking additional information, which risks incorrect diagnoses in such interactive settings. To address this limitation, we propose ProMed, a reinforcement learning (RL) framework that transitions medical LLMs toward a proactive paradigm, equipping them with the ability to ask clinically valuable questions before decision-making. At the core of ProMed is the Shapley Information Gain (SIG) reward, which quantifies the clinical utility of each question by combining the amount of newly acquired information with its contextual importance, estimated via Shapley values. We integrate SIG into a two-stage training pipeline: (1) SIG-Guided Model Initialization uses Monte Carlo Tree Search (MCTS) to construct high-reward interaction trajectories to supervise the model, and (2) SIG-Augmented Policy Optimization, which integrates SIG and enhances RL with a novel SIG-guided Reward Distribution Mechanism that assigns higher rewards to informative questions for targeted optimization. Extensive experiments on two newly curated partial-information medical benchmarks demonstrate that ProMed significantly outperforms state-of-the-art methods by an average of 6.29% and delivers a 54.45% gain over the reactive paradigm, while also generalizing robustly to out-of-domain cases. 

**Abstract (ZH)**: Interactive Medical询问对人体临床咨询至关重要，医师需要主动从患者处收集信息。尽管医学大型语言模型（LLMs）在静态医学问答任务上展现了令人印象深刻的能力，但它们主要以被动的方式运作：直接生成答案而不寻求额外信息，这在互动式设置中可能导致错误的诊断。为解决这一局限，我们提出ProMed，这是一种强化学习（RL）框架，使医疗LLMs从被动模式转向主动模式，赋予它们在决策前提出临床有价值的询问的能力。ProMed的核心是Shapley信息增益（SIG）奖励，该奖励通过结合新获得信息的量与其上下文重要性（使用Shapley值估计）来量化每个问题的临床价值。我们将SIG整合到两阶段的训练管道中：（1）由Monte Carlo树搜索（MCTS）指导的模型初始化构建高奖励交互轨迹以监督模型，（2）通过一种新颖的SIG指导的奖励分配机制增强RL，为指令优化分配更高奖励的互动性问题以实现特定优化。在两个新的部分信息医学基准测试上进行的广泛实验表明，ProMed平均优于最新方法6.29%，并且相比被动模式提高了54.45%的性能，同时在跨领域案例上也表现出稳健的泛化能力。 

---
# LLM-Enhanced Linear Autoencoders for Recommendation 

**Title (ZH)**: LLM增强的线性自动编码器推荐方法 

**Authors**: Jaewan Moon, Seongmin Park, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13500)  

**Abstract**: Large language models (LLMs) have been widely adopted to enrich the semantic representation of textual item information in recommender systems. However, existing linear autoencoders (LAEs) that incorporate textual information rely on sparse word co-occurrence patterns, limiting their ability to capture rich textual semantics. To address this, we propose L3AE, the first integration of LLMs into the LAE framework. L3AE effectively integrates the heterogeneous knowledge of textual semantics and user-item interactions through a two-phase optimization strategy. (i) L3AE first constructs a semantic item-to-item correlation matrix from LLM-derived item representations. (ii) It then learns an item-to-item weight matrix from collaborative signals while distilling semantic item correlations as regularization. Notably, each phase of L3AE is optimized through closed-form solutions, ensuring global optimality and computational efficiency. Extensive experiments demonstrate that L3AE consistently outperforms state-of-the-art LLM-enhanced models on three benchmark datasets, achieving gains of 27.6% in Recall@20 and 39.3% in NDCG@20. The source code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型(LLMs)已被广泛应用于丰富推荐系统中文本项信息的语义表示。然而，现有的结合文本信息的线性自编码器(LAEs)依赖稀疏的词共现模式，限制了其捕获丰富语义的能力。为了解决这个问题，我们提出L3AE，这是将LLMs融入LAE框架的第一个尝试。L3AE通过两阶段优化策略有效整合了文本语义和用户-项交互的异质知识。(i) L3AE首先从LLM提取的项表示中构建一个语义项-项相关矩阵。(ii) 然后，它从协作信号中学习一个项-项权重矩阵，同时通过正则化保留语义项相关性。值得注意的是，L3AE的每个阶段都通过闭式解进行优化，确保了全局最优性和计算效率。广泛的经验研究表明，L3AE在三个基准数据集上均优于最先进的LLM增强模型，Recall@20和NDCG@20分别提高了27.6%和39.3%。源代码可在此处获取。 

---
# CORENet: Cross-Modal 4D Radar Denoising Network with LiDAR Supervision for Autonomous Driving 

**Title (ZH)**: CORENet：具有LiDAR监督的跨模态4D雷达降噪网络 

**Authors**: Fuyang Liu, Jilin Mei, Fangyuan Mao, Chen Min, Yan Xing, Yu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13485)  

**Abstract**: 4D radar-based object detection has garnered great attention for its robustness in adverse weather conditions and capacity to deliver rich spatial information across diverse driving scenarios. Nevertheless, the sparse and noisy nature of 4D radar point clouds poses substantial challenges for effective perception. To address the limitation, we present CORENet, a novel cross-modal denoising framework that leverages LiDAR supervision to identify noise patterns and extract discriminative features from raw 4D radar data. Designed as a plug-and-play architecture, our solution enables seamless integration into voxel-based detection frameworks without modifying existing pipelines. Notably, the proposed method only utilizes LiDAR data for cross-modal supervision during training while maintaining full radar-only operation during inference. Extensive evaluation on the challenging Dual-Radar dataset, which is characterized by elevated noise level, demonstrates the effectiveness of our framework in enhancing detection robustness. Comprehensive experiments validate that CORENet achieves superior performance compared to existing mainstream approaches. 

**Abstract (ZH)**: 基于4D雷达的对象检测由于其在恶劣天气条件下的鲁棒性和跨多种驾驶场景提供丰富空间信息的能力而引起了广泛关注。然而，4D雷达点云的稀疏性和噪声性给有效的感知带来了重大挑战。为此，我们提出了一种名为CORENet的新型跨模态去噪框架，该框架利用LiDAR监督来识别噪声模式并从原始4D雷达数据中提取特征。设计为即插即用架构，我们的解决方案能够无缝集成到体素基检测框架中而无需修改现有管线。值得注意的是，所提出的方法仅在训练过程中利用LiDAR数据进行跨模态监督，在推理过程中保持纯雷达操作。在具有高噪声水平的Dual-Radar数据集上的广泛评估表明，该框架在提高检测鲁棒性方面效果显著。全面的实验验证了CORENet相比现有主流方法具有更好的性能。 

---
# STER-VLM: Spatio-Temporal With Enhanced Reference Vision-Language Models 

**Title (ZH)**: STER-VLM: 空间-时间增强参考的视觉-语言模型 

**Authors**: Tinh-Anh Nguyen-Nhu, Triet Dao Hoang Minh, Dat To-Thanh, Phuc Le-Gia, Tuan Vo-Lan, Tien-Huy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13470)  

**Abstract**: Vision-language models (VLMs) have emerged as powerful tools for enabling automated traffic analysis; however, current approaches often demand substantial computational resources and struggle with fine-grained spatio-temporal understanding. This paper introduces STER-VLM, a computationally efficient framework that enhances VLM performance through (1) caption decomposition to tackle spatial and temporal information separately, (2) temporal frame selection with best-view filtering for sufficient temporal information, and (3) reference-driven understanding for capturing fine-grained motion and dynamic context and (4) curated visual/textual prompt techniques. Experimental results on the WTS \cite{kong2024wts} and BDD \cite{BDD} datasets demonstrate substantial gains in semantic richness and traffic scene interpretation. Our framework is validated through a decent test score of 55.655 in the AI City Challenge 2025 Track 2, showing its effectiveness in advancing resource-efficient and accurate traffic analysis for real-world applications. 

**Abstract (ZH)**: 基于视觉-语言模型的时空增强框架（STER-VLM）：提升计算效率的交通分析技术 

---
# Consumer Autonomy or Illusion? Rethinking Consumer Agency in the Age of Algorithms 

**Title (ZH)**: 消费者自主还是幻象？在算法时代重思消费者能动性 

**Authors**: Pegah Nokhiz, Aravinda Kanchana Ruwanpathirana  

**Link**: [PDF](https://arxiv.org/pdf/2508.13440)  

**Abstract**: Consumer agency in the digital age is increasingly constrained by systemic barriers and algorithmic manipulation, raising concerns about the authenticity of consumption choices. Nowadays, financial decisions are shaped by external pressures like obligatory consumption, algorithmic persuasion, and unstable work schedules that erode financial autonomy. Obligatory consumption (like hidden fees) is intensified by digital ecosystems. Algorithmic tactics like personalized recommendations lead to impulsive purchases. Unstable work schedules also undermine financial planning. Thus, it is important to study how these factors impact consumption agency. To do so, we examine formal models grounded in discounted consumption with constraints that bound agency. We construct analytical scenarios in which consumers face obligatory payments, algorithm-influenced impulsive expenses, or unpredictable income due to temporal instability. Using this framework, we demonstrate that even rational, utility-maximizing agents can experience early financial ruin when agency is limited across structural, behavioral, or temporal dimensions and how diminished autonomy impacts long-term financial well-being. Our central argument is that consumer agency must be treated as a value (not a given) requiring active cultivation, especially in digital ecosystems. The connection between our formal modeling and this argument allows us to indicate that limitations on agency (whether structural, behavioral, or temporal) can be rigorously linked to measurable risks like financial instability. This connection is also a basis for normative claims about consumption as a value, by anchoring them in a formally grounded analysis of consumer behavior. As solutions, we study systemic interventions and consumer education to support value deliberation and informed choices. We formally demonstrate how these measures strengthen agency. 

**Abstract (ZH)**: 数字时代消费者的自主权受到系统障碍和技术操控的限制，消费选择的真实性受到关注。当前的财务决策受到强制消费、算法劝说和不稳定工作时间等外部压力的影响，损害了财务自主权。强制消费（如隐形费用）在数字生态系统中被放大。个性化的推荐算法导致冲动购买。不稳定的工作时间也削弱了财务规划。因此，研究这些因素如何影响消费自主权至关重要。为此，我们基于受限折现消费的正式模型进行研究，构建了消费者面对强制付款、算法影响下的冲动开支或因时间不稳定性而带来的不可预测收入的分析场景。借助这一框架，我们证明了即使是最理性的效用最大化代理，当自主权在结构、行为或时间维度上受到限制时，也可能在早期遭遇财务破产，并阐明了减弱的自主权如何影响长期的财务福祉。我们的核心观点是，消费者的自主权应被视为一种价值（而非既定事实），需要主动培养，尤其是在数字生态系统中。我们正式建模与这一论点的联系，表明自主权的限制（无论是结构性的、行为上的还是时间上的）可以严格关联到可衡量的风险，如财务不稳定性。这一联系也为关于消费作为价值的规范性主张提供了依据，通过正式的地分析消费者行为来锚定这些主张。作为解决方案，我们研究系统干预措施和消费者教育，以支持价值权衡与知情选择，并正式证明了这些措施是如何增强自主权的。 

---
# Structured Prompting and Multi-Agent Knowledge Distillation for Traffic Video Interpretation and Risk Inference 

**Title (ZH)**: 结构化提示与多agents知识蒸馏在交通视频解释与风险推理中的应用 

**Authors**: Yunxiang Yang, Ningning Xu, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13439)  

**Abstract**: Comprehensive highway scene understanding and robust traffic risk inference are vital for advancing Intelligent Transportation Systems (ITS) and autonomous driving. Traditional approaches often struggle with scalability and generalization, particularly under the complex and dynamic conditions of real-world environments. To address these challenges, we introduce a novel structured prompting and knowledge distillation framework that enables automatic generation of high-quality traffic scene annotations and contextual risk assessments. Our framework orchestrates two large Vision-Language Models (VLMs): GPT-4o and o3-mini, using a structured Chain-of-Thought (CoT) strategy to produce rich, multi-perspective outputs. These outputs serve as knowledge-enriched pseudo-annotations for supervised fine-tuning of a much smaller student VLM. The resulting compact 3B-scale model, named VISTA (Vision for Intelligent Scene and Traffic Analysis), is capable of understanding low-resolution traffic videos and generating semantically faithful, risk-aware captions. Despite its significantly reduced parameter count, VISTA achieves strong performance across established captioning metrics (BLEU-4, METEOR, ROUGE-L, and CIDEr) when benchmarked against its teacher models. This demonstrates that effective knowledge distillation and structured multi-agent supervision can empower lightweight VLMs to capture complex reasoning capabilities. The compact architecture of VISTA facilitates efficient deployment on edge devices, enabling real-time risk monitoring without requiring extensive infrastructure upgrades. 

**Abstract (ZH)**: 全面的高速公路场景理解和稳健的交通风险推断对于推动智能交通系统（ITS）和自动驾驶至关重要。传统的 approach 通常在处理真实环境下的复杂和动态条件时表现出可扩展性和泛化能力的不足。为了解决这些挑战，我们提出了一种新型的结构化提示和知识蒸馏框架，该框架能够自动生成高质量的交通场景标注和上下文风险评估。该框架利用结构化链式思考（CoT）策略协调两个大型ビジョンと言語モデル（VLMs）：GPT-4o和o3-mini，生成丰富、多视角的输出。这些输出作为知识丰富的伪标注，用于监督微调一个小得多的学生VLM。由此产生的紧凑3B量级模型，名为VISTA（视觉智能场景与交通分析），能够理解低分辨率的交通视频并生成语义忠实、风险意识强的描述。尽管参数量显著减少，但VISTA在基准测试中表现出色，其性能在现有描述生成指标（BLEU-4、METEOR、ROUGE-L和CIDEr）上达到了强劲的表现。这表明有效的知识蒸馏和结构化多代理监督可以使轻量级的VLMs具备复杂的推理能力。VISTA的紧凑架构便于在边缘设备上高效部署，无需进行广泛的基础设施升级即可实现实时风险监测。 

---
# Dynamic Design of Machine Learning Pipelines via Metalearning 

**Title (ZH)**: 基于元学习的机器学习管道动态设计 

**Authors**: Edesio Alcobaça, André C. P. L. F. de Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2508.13436)  

**Abstract**: Automated machine learning (AutoML) has democratized the design of machine learning based systems, by automating model selection, hyperparameter tuning and feature engineering. However, the high computational cost associated with traditional search and optimization strategies, such as Random Search, Particle Swarm Optimization and Bayesian Optimization, remains a significant challenge. Moreover, AutoML systems typically explore a large search space, which can lead to overfitting. This paper introduces a metalearning method for dynamically designing search spaces for AutoML system. The proposed method uses historical metaknowledge to select promising regions of the search space, accelerating the optimization process. According to experiments conducted for this study, the proposed method can reduce runtime by 89\% in Random Search and search space by (1.8/13 preprocessor and 4.3/16 classifier), without compromising significant predictive performance. Moreover, the proposed method showed competitive performance when adapted to Auto-Sklearn, reducing its search space. Furthermore, this study encompasses insights into meta-feature selection, meta-model explainability, and the trade-offs inherent in search space reduction strategies. 

**Abstract (ZH)**: 自动化机器学习（AutoML）通过自动化模型选择、超参数调优和特征工程，民主化了基于机器学习的系统设计。然而，传统搜索和优化策略（如随机搜索、粒子群优化和贝叶斯优化）相关的人机成本仍然是一个重大挑战。此外，AutoML系统通常探索一个巨大的搜索空间，这可能导致过拟合。本文提出了一种元学习方法，用于动态设计AutoML系统的搜索空间。该方法利用历史元知识选择搜索空间中具有潜力的区域，从而加速优化过程。根据本研究中的实验，所提出的方法可以将随机搜索的运行时间减少89%，并将搜索空间分别减少到1.8/13预处理器和4.3/16分类器，而不会显著牺牲预测性能。此外，当将该方法调整应用于Auto-Sklearn时，展示了其竞争性能并减少了其搜索空间。此外，本研究还包括了关于元特征选择、元模型可解释性和搜索空间缩减策略固有折衷的见解。 

---
# SVDformer: Direction-Aware Spectral Graph Embedding Learning via SVD and Transformer 

**Title (ZH)**: SVDformer: 基于SVD和Transformer的方向感知频谱图嵌入学习 

**Authors**: Jiayu Fang, Zhiqi Shao, S T Boris Choy, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13435)  

**Abstract**: Directed graphs are widely used to model asymmetric relationships in real-world systems. However, existing directed graph neural networks often struggle to jointly capture directional semantics and global structural patterns due to their isotropic aggregation mechanisms and localized filtering mechanisms. To address this limitation, this paper proposes SVDformer, a novel framework that synergizes SVD and Transformer architecture for direction-aware graph representation learning. SVDformer first refines singular value embeddings through multi-head self-attention, adaptively enhancing critical spectral components while suppressing high-frequency noise. This enables learnable low-pass/high-pass graph filtering without requiring spectral kernels. Furthermore, by treating singular vectors as directional projection bases and singular values as scaling factors, SVDformer uses the Transformer to model multi-scale interactions between incoming/outgoing edge patterns through attention weights, thereby explicitly preserving edge directionality during feature propagation. Extensive experiments on six directed graph benchmarks demonstrate that SVDformer consistently outperforms state-of-the-art GNNs and direction-aware baselines on node classification tasks, establishing a new paradigm for learning representations on directed graphs. 

**Abstract (ZH)**: SVDFormer：一种协同SVD和Transformer架构的方向感知图表示学习框架 

---
# EventTSF: Event-Aware Non-Stationary Time Series Forecasting 

**Title (ZH)**: 基于事件的非平稳时间序列预测：EventTSF 

**Authors**: Yunfeng Ge, Ming Jin, Yiji Zhao, Hongyan Li, Bo Du, Chang Xu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13434)  

**Abstract**: Time series forecasting plays a vital role in critical domains like energy and transportation, where non-stationary dynamics are deeply intertwined with events in other modalities such as texts. However, incorporating natural language-based external events to improve non-stationary forecasting remains largely unexplored, as most approaches still rely on a single modality, resulting in limited contextual knowledge and model underperformance. Enabling fine-grained multimodal interactions between temporal and textual data is challenged by three fundamental issues: (1) the difficulty of fine-grained synchronization between time-varying discrete textual events and continuous time series; (2) the inherent temporal uncertainty introduced by textual semantics; and (3) the misalignment between textual event embeddings and multi-resolution temporal patterns. In this work, we address these challenges by introducing event-aware non-stationary time series forecasting (EventTSF), an autoregressive generation framework that integrates historical time series with textual events to make subsequent forecasts. Specifically, EventTSF uses autoregressive diffusion with flow matching at each step to capture nuanced temporal-event interactions. To handle event-induced uncertainty, flow matching timesteps are adaptively controlled according to event semantic signals. The underlying denoiser employs a multimodal U-shaped diffusion transformer that efficiently fuses temporal and textual modalities across different resolutions. Extensive experiments on 8 synthetic and real-world datasets show that EventTSF outperforms 12 baselines across diverse event-aware non-stationary time series forecasting scenarios, achieving substantial improvements of 10.7% higher forecasting accuracy and $1.13\times$ faster training efficiency. 

**Abstract (ZH)**: 事件意识非平稳时间序列预测（EventTSF）：一种集成历史时间序列和文本事件的自回归生成框架 

---
# AlphaX: An AI-Based Value Investing Strategy for the Brazilian Stock Market 

**Title (ZH)**: AlphaX：基于人工智能的价值投资策略——以巴西股市为例 

**Authors**: Paulo André Lima de Castro  

**Link**: [PDF](https://arxiv.org/pdf/2508.13429)  

**Abstract**: Autonomous trading strategies have been a subject of research within the field of artificial intelligence (AI) for aconsiderable period. Various AI techniques have been explored to develop autonomous agents capable of trading financial assets. These approaches encompass traditional methods such as neural networks, fuzzy logic, and reinforcement learning, as well as more recent advancements, including deep neural networks and deep reinforcement learning. Many developers report success in creating strategies that exhibit strong performance during simulations using historical price data, a process commonly referred to as backtesting. However, when these strategies are deployed in real markets, their performance often deteriorates, particularly in terms of risk-adjusted returns. In this study, we propose an AI-based strategy inspired by a classical investment paradigm: Value Investing. Financial AI models are highly susceptible to lookahead bias and other forms of bias that can significantly inflate performance in backtesting compared to live trading conditions. To address this issue, we conducted a series of computational simulations while controlling for these biases, thereby reducing the risk of overfitting. Our results indicate that the proposed approach outperforms major Brazilian market benchmarks. Moreover, the strategy, named AlphaX, demonstrated superior performance relative to widely used technical indicators such as the Relative Strength Index (RSI) and Money Flow Index (MFI), with statistically significant results. Finally, we discuss several open challenges and highlight emerging technologies in qualitative analysis that may contribute to the development of a comprehensive AI-based Value Investing framework in the future 

**Abstract (ZH)**: 基于人工智能的价值投资自主交易策略：克服回测偏差与实盘表现差异的研究 

---
# Mitigating Easy Option Bias in Multiple-Choice Question Answering 

**Title (ZH)**: 缓解多项选择题回答中的易选项偏见 

**Authors**: Hao Zhang, Chen Li, Basura Fernando  

**Link**: [PDF](https://arxiv.org/pdf/2508.13428)  

**Abstract**: In this early study, we observe an Easy-Options Bias (EOB) issue in some multiple-choice Visual Question Answering (VQA) benchmarks such as MMStar, RealWorldQA, SEED-Bench, Next-QA, STAR benchmark and Video-MME. This bias allows vision-language models (VLMs) to select the correct answer using only the vision (V) and options (O) as inputs, without the need for the question (Q). Through grounding experiments, we attribute the bias to an imbalance in visual relevance: the correct answer typically aligns more closely with the visual contents than the negative options in feature space, creating a shortcut for VLMs to infer the answer via simply vision-option similarity matching. To fix this, we introduce GroundAttack, a toolkit that automatically generates hard negative options as visually plausible as the correct answer. We apply it to the NExT-QA and MMStar datasets, creating new EOB-free annotations. On these EOB-free annotations, current VLMs approach to random accuracies under (V+O) settings, and drop to non-saturated accuracies under (V+Q+O) settings, providing a more realistic evaluation of VLMs' QA ability. Codes and new annotations will be released soon. 

**Abstract (ZH)**: 在早期研究中，我们发现在MMStar、RealWorldQA、SEED-Bench、Next-QA、STAR基准和Video-MME等一些多项选择视觉问答（VQA）基准中存在易选项偏差（EOB）问题。通过接地实验，我们归因于视觉相关性的不平衡：正确答案在特征空间中通常与视觉内容更密切对齐，而负选项则不然，这为VLMs提供了直接通过视觉-选项相似性匹配来推断答案的捷径。为解决这一问题，我们引入了GroundAttack工具包，它可以自动生成与正确答案视觉上同样可信的困难负选项。我们将其应用于NExT-QA和MMStar数据集，创建了新的无EOB注释。在这些无EOB注释下，当前的VLMs在仅使用（V+O）设置时表现为随机准确性，并在（V+Q+O）设置下准确性无法饱和，这为更真实地评估VLMs的问答能力提供了依据。代码和新注释将于近期发布。 

---
# ALIGN: Word Association Learning for Cross-Cultural Generalization in Large Language Models 

**Title (ZH)**: ALIGN: 跨文化通用性中的单词关联学习 

**Authors**: Chunhua Liu, Kabir Manandhar Shrestha, Sukai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13426)  

**Abstract**: As large language models (LLMs) increasingly mediate cross-cultural communication, their behavior still reflects the distributional bias of the languages and viewpoints that are over-represented in their pre-training corpora. Yet, it remains a challenge to model and align culture due to limited cultural knowledge and a lack of exploration into effective learning approaches. We introduce a cost-efficient, cognitively grounded remedy: parameter-efficient fine-tuning on native speakers' free word-association norms, which encode implicit cultural schemas. Leveraging English-US and Mandarin associations from the Small-World-of-Words project, we adapt Llama-3.1-8B and Qwen-2.5-7B via supervised fine-tuning (SFT) and PPO-based preference optimization. SFT boosts held-out association Precision at 5 by 16-20% in English and 43-165% in Mandarin, lifts median concreteness by +0.20, and attains human-level valence and arousal. These lexical gains transfer: on World-Values-Survey questions, fine-tuned models shift answer distributions toward the target culture, and on a 50-item high-tension subset, Qwen's Chinese-aligned responses double while Llama's US bias drops by one-third. Our 7-8B models rival or beat vanilla 70B baselines, showing that a few million culture-grounded associations can instill value alignment without costly retraining. Our work highlights both the promise and the need for future research grounded in human cognition in improving cultural alignment in AI models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在越来越多地调解跨文化沟通时，其行为仍然反映了其预训练语料中过度代表的语言和观点的分布性偏见。但是，由于文化知识有限且缺乏有效的学习方法探索，对文化建模和对齐仍然是一项挑战。我们介绍了一种成本效益高且契合认知的解决方案：在母语者自由词汇联想规范上进行参数高效微调，以此编码出隐含的文化模式。利用Small-World-of-Words项目中的英语-美国和普通话联想，我们通过监督微调（SFT）和基于PPO的偏好优化适应了Llama-3.1-8B和Qwen-2.5-7B。SFT在英语中将保留联想的精确度提高了16-20%，在普通话中提高了43-165%，将中值具体性提升了0.20，并达到了人类水平的价值和唤醒程度。这些词汇上的改进得以转移：在世界价值观调查问卷中，微调后的模型将答案分布向目标文化转移，而在一个包含50个项目的高张力子集上，Qwen的中国文化对齐回答翻了一番，而Llama的美国倾向降低了三分之一。我们的7-8B模型与或优于 vanilla 70B基线，表明数百万文化基础的联想可以实现价值观对齐而无需昂贵的重新训练。我们的工作突显了基于人类认知改进AI模型文化对齐的前景和未来研究的迫切需要。 

---
# AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System 

**Title (ZH)**: AdaptJobRec: 提升由 LLM 驱动的代理型聊天职业推荐系统楽し�ándose
user
Adaptive Transformer Compression for Efficient Recommender Systems in Edge Computing Environments 

**Authors**: Qixin Wang, Dawei Wang, Kun Chen, Yaowei Hu, Puneet Girdhar, Ruoteng Wang, Aadesh Gupta, Chaitanya Devella, Wenlai Guo, Shangwen Huang, Bachir Aoun, Greg Hayworth, Han Li, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13423)  

**Abstract**: In recent years, recommendation systems have evolved from providing a single list of recommendations to offering a comprehensive suite of topic focused services. To better accomplish this task, conversational recommendation systems (CRS) have progressed from basic retrieval augmented LLM generation to agentic systems with advanced reasoning and self correction capabilities. However, agentic systems come with notable response latency, a longstanding challenge for conversational recommendation systems. To balance the trade off between handling complex queries and minimizing latency, we propose AdaptJobRec, the first conversational job recommendation system that leverages autonomous agent to integrate personalized recommendation algorithm tools. The system employs a user query complexity identification mechanism to minimize response latency. For straightforward queries, the agent directly selects the appropriate tool for rapid responses. For complex queries, the agent uses the memory processing module to filter chat history for relevant content, then passes the results to the intelligent task decomposition planner, and finally executes the tasks using personalized recommendation tools. Evaluation on Walmart's real world career recommendation scenarios demonstrates that AdaptJobRec reduces average response latency by up to 53.3% compared to competitive baselines, while significantly improving recommendation accuracy. 

**Abstract (ZH)**: 近年来，推荐系统从提供单一推荐列表演进到提供全面的主题聚焦服务。为了更好地完成这一任务，对话推荐系统（CRS）从基本的检索增强语言模型生成发展到具有高级推理和自我修正能力的代理系统。然而，代理系统伴随着显著的响应延迟，这是对话推荐系统的一个长期挑战。为了在处理复杂查询和最小化延迟之间取得平衡，我们提出了AdaptJobRec，这是第一个利用自主代理整合个性化推荐算法工具的对话职业推荐系统。该系统采用用户查询复杂性识别机制以减少响应延迟。对于简单的查询，代理直接选择合适的工具以快速响应。对于复杂的查询，代理使用记忆处理模块过滤聊天历史以提取相关信息，随后将结果传递给智能任务分解规划器，并最终使用个性化推荐工具执行任务。在沃尔玛实际职业生涯推荐场景上的评估表明，与竞争baseline相比，AdaptJobRec将平均响应延迟最多降低了53.3%，同时显著提高推荐准确性。 

---
# Semi-Supervised Anomaly Detection Pipeline for SOZ Localization Using Ictal-Related Chirp 

**Title (ZH)**: 基于癫痫相关单音调的半监督异常检测管道用于SOZ定位 

**Authors**: Nooshin Bahador, Milad Lankarany  

**Link**: [PDF](https://arxiv.org/pdf/2508.13406)  

**Abstract**: This study presents a quantitative framework for evaluating the spatial concordance between clinically defined seizure onset zones (SOZs) and statistically anomalous channels identified through time-frequency analysis of chirp events. The proposed pipeline employs a two-step methodology: (1) Unsupervised Outlier Detection, where Local Outlier Factor (LOF) analysis with adaptive neighborhood selection identifies anomalous channels based on spectro-temporal features of chirp (Onset frequency, offset frequency, and temporal duration); and (2) Spatial Correlation Analysis, which computes both exact co-occurrence metrics and weighted index similarity, incorporating hemispheric congruence and electrode proximity. Key findings demonstrate that the LOF-based approach (N neighbors=20, contamination=0.2) effectively detects outliers, with index matching (weighted by channel proximity) outperforming exact matching in SOZ localization. Performance metrics (precision, recall, F1) were highest for seizure-free patients (Index Precision mean: 0.903) and those with successful surgical outcomes (Index Precision mean: 0.865), whereas failure cases exhibited lower concordance (Index Precision mean: 0.460). The key takeaway is that chirp-based outlier detection, combined with weighted spatial metrics, provides a complementary method for SOZ localization, particularly in patients with successful surgical outcomes. 

**Abstract (ZH)**: 基于颤动事件时频分析识别的统计异常通道与临床定义的癫痫发作起始区的空间一致性的量化评估框架 

---
# Datarus-R1: An Adaptive Multi-Step Reasoning LLM for Automated Data Analysis 

**Title (ZH)**: Datarus-R1：一种适应性多步推理大型语言模型，用于自动化数据分析 

**Authors**: Ayoub Ben Chaliah, Hela Dellagi  

**Link**: [PDF](https://arxiv.org/pdf/2508.13382)  

**Abstract**: We present Datarus-R1-14B, a 14 B-parameter open-weights language model fine-tuned from Qwen 2.5-14B-Instruct to act as a virtual data analyst and graduate-level problem solver. Datarus is trained not on isolated question-answer pairs but on full analytical trajectories including reasoning steps, code execution, error traces, self-corrections, and final conclusions, all captured in a ReAct-style notebook format spanning finance, medicine, numerical analysis, and other quantitative domains. Our training pipeline combines (i) a trajectory-centric synthetic data generator that yielded 144 000 tagged notebook episodes, (ii) a dual-reward framework blending a lightweight tag-based structural signal with a Hierarchical Reward Model (HRM) that scores both single-step soundness and end-to-end coherence, and (iii) a memory-optimized implementation of Group Relative Policy Optimization (GRPO) featuring KV-cache reuse, sequential generation, and reference-model sharding. A cosine curriculum smoothly shifts emphasis from structural fidelity to semantic depth, reducing the format collapse and verbosity that often plague RL-aligned LLMs. A central design choice in Datarus is it dual reasoning interface. In agentic mode the model produces ReAct-tagged steps that invoke Python tools to execute real code; in reflection mode it outputs compact Chain-of-Thought (CoT) traces delimited by <think> and <answer> tags. On demanding postgraduate-level problems, Datarus exhibits an "AHA-moment" pattern: it sketches hypotheses, revises them once or twice, and converges avoiding the circular, token-inflating loops common to contemporary systems. Across standard public benchmarks Datarus surpasses similar size models and even reaches the level of larger reasoning models such as QwQ-32B achieving up to 30% higher accuracy on AIME 2024/2025 and LiveCodeBench while emitting 18-49% fewer tokens per solution. 

**Abstract (ZH)**: Datarus-R1-14B：一个基于Qwen 2.5-14B-Instruct微调的虚拟数据分析师和研究生级问题解决者大型语言模型 

---
# Whispering Context: Distilling Syntax and Semantics for Long Speech Transcripts 

**Title (ZH)**: 默声之息：提炼长语音转录中的语法与语义 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2508.13376)  

**Abstract**: ASR systems often struggle with maintaining syntactic and semantic accuracy in long audio transcripts, impacting tasks like Named Entity Recognition (NER), capitalization, and punctuation. We propose a novel approach that enhances ASR by distilling contextual knowledge from LLaMA models into Whisper. Our method uses two strategies: (1) token level distillation with optimal transport to align dimensions and sequence lengths, and (2) representation loss minimization between sentence embeddings of Whisper and LLaMA, blending syntax and semantics. Evaluations on the Spoken Wikipedia dataset, a benchmark with long audios and rich entities demonstrate significant improvements in Word Error Rate (WER), NER, capitalization, and punctuation success. By introducing novel NER metrics and exploring semantics aware ASR, our work highlights the value of integrating linguistic context into transcription, setting a foundation for robust, context-aware ASR in longform speech. 

**Abstract (ZH)**: ASR系统在维护长音频转录的句法和语义准确性方面往往面临挑战，影响命名实体识别(NER)、标点符号和大小写等任务。我们提出一种新颖的方法，通过将LLaMA模型的上下文知识提炼到Whisper中来增强ASR性能。该方法采用两种策略：(1) 基于最优传输的子令牌级别提炼，对齐维度和序列长度；(2) 通过最小化Whisper和LLaMA句子嵌入之间的表示损失，融合句法和语义。在包含长音频和丰富实体的Spoken Wikipedia数据集上的评估结果显示，该方法在单词错误率(WER)、NER、大小写和标点符号准确率方面取得了显著提升。通过引入新的NER指标并探索语义感知的ASR，我们的工作突显了将语言上下文整合到转录中的价值，为长篇语音的健壮、上下文感知ASR奠定了基础。 

---
# Overcoming Latency Bottlenecks in On-Device Speech Translation: A Cascaded Approach with Alignment-Based Streaming MT 

**Title (ZH)**: 克服设备端语音翻译的延迟瓶颈：基于对齐的级联流式MT方法 

**Authors**: Zeeshan Ahmed, Frank Seide, Niko Moritz, Ju Lin, Ruiming Xie, Simone Merello, Zhe Liu, Christian Fuegen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13358)  

**Abstract**: This paper tackles several challenges that arise when integrating Automatic Speech Recognition (ASR) and Machine Translation (MT) for real-time, on-device streaming speech translation. Although state-of-the-art ASR systems based on Recurrent Neural Network Transducers (RNN-T) can perform real-time transcription, achieving streaming translation in real-time remains a significant challenge. To address this issue, we propose a simultaneous translation approach that effectively balances translation quality and latency. We also investigate efficient integration of ASR and MT, leveraging linguistic cues generated by the ASR system to manage context and utilizing efficient beam-search pruning techniques such as time-out and forced finalization to maintain system's real-time factor. We apply our approach to an on-device bilingual conversational speech translation and demonstrate that our techniques outperform baselines in terms of latency and quality. Notably, our technique narrows the quality gap with non-streaming translation systems, paving the way for more accurate and efficient real-time speech translation. 

**Abstract (ZH)**: 本文解决了将自动语音识别（ASR）和机器翻译（MT）集成用于实时设备端流式语音翻译时出现的多个挑战。虽然基于循环神经网络译码器（RNN-T）的先进ASR系统可以进行实时转写，但在实时实现流式翻译仍然是一个重大挑战。为此，我们提出了一种同时翻译方法，有效平衡了翻译质量和延迟。此外，我们研究了ASR和MT的高效集成，利用ASR系统生成的语言线索管理上下文，并采用时间超时和强制最终化等高效的束搜索剪枝技术来保持系统的实时性。我们将该方法应用于设备端双语对话语音翻译，并证明了我们的技术在延迟和质量上超过了基线。值得注意的是，我们的技术缩小了与非流式翻译系统之间的质量差距，为更准确和高效的实时语音翻译铺平了道路。 

---
# Counterfactual Probabilistic Diffusion with Expert Models 

**Title (ZH)**: 专家模型引导的反事实概率扩散 

**Authors**: Wenhao Mu, Zhi Cao, Mehmed Uludag, Alexander Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2508.13355)  

**Abstract**: Predicting counterfactual distributions in complex dynamical systems is essential for scientific modeling and decision-making in domains such as public health and medicine. However, existing methods often rely on point estimates or purely data-driven models, which tend to falter under data scarcity. We propose a time series diffusion-based framework that incorporates guidance from imperfect expert models by extracting high-level signals to serve as structured priors for generative modeling. Our method, ODE-Diff, bridges mechanistic and data-driven approaches, enabling more reliable and interpretable causal inference. We evaluate ODE-Diff across semi-synthetic COVID-19 simulations, synthetic pharmacological dynamics, and real-world case studies, demonstrating that it consistently outperforms strong baselines in both point prediction and distributional accuracy. 

**Abstract (ZH)**: 在复杂动力系统中预测反事实分布对于科学建模和决策在公共卫生和药物领域中是必不可少的。现有方法通常依赖于纯数据驱动模型，这些模型在数据稀缺时往往会失效。我们提出了一种基于扩散的框架，通过提取高频信号作为结构先验用于生成建模，从而整合了机械主义和数据驱动的方法。该方法在ODE-Diff上实现了在半合成的COVID-1-1感染模拟、合成的药物动力学和真实世界的公共卫生数据上的评估，在这些评估中，ODE-Diff 一致地优于强大的基线方法，在在反事实预测和分布准确性方面表现更优。 

---
# A Dual-Attention Graph Network for fMRI Data Classification 

**Title (ZH)**: 双注意力图形网络在fMRI数据分类中的应用 

**Authors**: Amirali Arbab, Zeinab Davarani, Mehran Safayani  

**Link**: [PDF](https://arxiv.org/pdf/2508.13328)  

**Abstract**: Understanding the complex neural activity dynamics is crucial for the development of the field of neuroscience. Although current functional MRI classification approaches tend to be based on static functional connectivity or cannot capture spatio-temporal relationships comprehensively, we present a new framework that leverages dynamic graph creation and spatiotemporal attention mechanisms for Autism Spectrum Disorder(ASD) diagnosis. The approach used in this research dynamically infers functional brain connectivity in each time interval using transformer-based attention mechanisms, enabling the model to selectively focus on crucial brain regions and time segments. By constructing time-varying graphs that are then processed with Graph Convolutional Networks (GCNs) and transformers, our method successfully captures both localized interactions and global temporal dependencies. Evaluated on the subset of ABIDE dataset, our model achieves 63.2 accuracy and 60.0 AUC, outperforming static graph-based approaches (e.g., GCN:51.8). This validates the efficacy of joint modeling of dynamic connectivity and spatio-temporal context for fMRI classification. The core novelty arises from (1) attention-driven dynamic graph creation that learns temporal brain region interactions and (2) hierarchical spatio-temporal feature fusion through GCNtransformer fusion. 

**Abstract (ZH)**: 基于动态图创建和时空注意力机制的自闭症谱系障碍诊断研究：捕获时空依赖关系的新框架 

---
# A Surveillance Based Interactive Robot 

**Title (ZH)**: 基于监控的交互式机器人 

**Authors**: Kshitij Kavimandan, Pooja Mangal, Devanshi Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2508.13319)  

**Abstract**: We build a mobile surveillance robot that streams video in real time and responds to speech so a user can monitor and steer it from a phone or browser. The system uses two Raspberry Pi 4 units: a front unit on a differential drive base with camera, mic, and speaker, and a central unit that serves the live feed and runs perception. Video is sent with FFmpeg. Objects in the scene are detected using YOLOv3 to support navigation and event awareness. For voice interaction, we use Python libraries for speech recognition, multilingual translation, and text-to-speech, so the robot can take spoken commands and read back responses in the requested language. A Kinect RGB-D sensor provides visual input and obstacle cues. In indoor tests the robot detects common objects at interactive frame rates on CPU, recognises commands reliably, and translates them to actions without manual control. The design relies on off-the-shelf hardware and open software, making it easy to reproduce. We discuss limits and practical extensions, including sensor fusion with ultrasonic range data, GPU acceleration, and adding face and text recognition. 

**Abstract (ZH)**: 一种用于实时视频流和语音交互的移动监控机器人系统 

---
# Diff-MSM: Differentiable MusculoSkeletal Model for Simultaneous Identification of Human Muscle and Bone Parameters 

**Title (ZH)**: Diff-MSM: 可微肌骨模型同时识别人体肌肉和骨骼参数 

**Authors**: Yingfan Zhou, Philip Sanderink, Sigurd Jager Lemming, Cheng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13303)  

**Abstract**: High-fidelity personalized human musculoskeletal models are crucial for simulating realistic behavior of physically coupled human-robot interactive systems and verifying their safety-critical applications in simulations before actual deployment, such as human-robot co-transportation and rehabilitation through robotic exoskeletons. Identifying subject-specific Hill-type muscle model parameters and bone dynamic parameters is essential for a personalized musculoskeletal model, but very challenging due to the difficulty of measuring the internal biomechanical variables in vivo directly, especially the joint torques. In this paper, we propose using Differentiable MusculoSkeletal Model (Diff-MSM) to simultaneously identify its muscle and bone parameters with an end-to-end automatic differentiation technique differentiating from the measurable muscle activation, through the joint torque, to the resulting observable motion without the need to measure the internal joint torques. Through extensive comparative simulations, the results manifested that our proposed method significantly outperformed the state-of-the-art baseline methods, especially in terms of accurate estimation of the muscle parameters (i.e., initial guess sampled from a normal distribution with the mean being the ground truth and the standard deviation being 10% of the ground truth could end up with an average of the percentage errors of the estimated values as low as 0.05%). In addition to human musculoskeletal modeling and simulation, the new parameter identification technique with the Diff-MSM has great potential to enable new applications in muscle health monitoring, rehabilitation, and sports science. 

**Abstract (ZH)**: 高保真个性化人体肌骨模型对于模拟物理耦合的人机交互系统的现实行为以及在实际部署前（如人类与机器人协同运输和通过机器人外骨骼进行康复）验证其关键安全应用至关重要。通过关节扭矩自始至终自动微分技术识别特定个体的希尔型肌肉模型参数和骨骼动力学参数对于个性化肌骨模型至关重要，但因直接测量活体内内部生物力学变量尤其关节扭矩的难度较大而极具挑战性。本文提出使用可微肌骨模型(Diff-MSM)同时通过可测量的肌肉激活间接自动识别其肌肉和骨骼参数，从关节扭矩推导到最终可观察的运动，无需直接测量关节扭矩。通过广泛的对比仿真，结果表明我们提出的方法在肌肉参数准确估计方面显著优于最先进的基准方法，尤其是在肌肉参数估计方面（初始猜测来自均值为真实值，标准差为真实值10%的正态分布的样本，最终估计值的平均百分比误差仅为0.05%）。除了人类肌骨建模和仿真外，Diff-MSM的新参数识别技术在肌肉健康监测、康复和体育科学等领域具有巨大应用潜力。 

---
# GaitCrafter: Diffusion Model for Biometric Preserving Gait Synthesis 

**Title (ZH)**: 步态匠人：保留生物特征的步态合成扩散模型 

**Authors**: Sirshapan Mitra, Yogesh S. Rawat  

**Link**: [PDF](https://arxiv.org/pdf/2508.13300)  

**Abstract**: Gait recognition is a valuable biometric task that enables the identification of individuals from a distance based on their walking patterns. However, it remains limited by the lack of large-scale labeled datasets and the difficulty of collecting diverse gait samples for each individual while preserving privacy. To address these challenges, we propose GaitCrafter, a diffusion-based framework for synthesizing realistic gait sequences in the silhouette domain. Unlike prior works that rely on simulated environments or alternative generative models, GaitCrafter trains a video diffusion model from scratch, exclusively on gait silhouette data. Our approach enables the generation of temporally consistent and identity-preserving gait sequences. Moreover, the generation process is controllable-allowing conditioning on various covariates such as clothing, carried objects, and view angle. We show that incorporating synthetic samples generated by GaitCrafter into the gait recognition pipeline leads to improved performance, especially under challenging conditions. Additionally, we introduce a mechanism to generate novel identities-synthetic individuals not present in the original dataset-by interpolating identity embeddings. These novel identities exhibit unique, consistent gait patterns and are useful for training models while maintaining privacy of real subjects. Overall, our work takes an important step toward leveraging diffusion models for high-quality, controllable, and privacy-aware gait data generation. 

**Abstract (ZH)**: 基于扩散模型的 silhouette 领域实际步态序列合成框架 GaitCrafter 

---
# Hierarchical Conformal Classification 

**Title (ZH)**: 分层符合分类 

**Authors**: Floris den Hengst, Inès Blin, Majid Mohammadi, Syed Ihtesham Hussain Shah, Taraneh Younesian  

**Link**: [PDF](https://arxiv.org/pdf/2508.13288)  

**Abstract**: Conformal prediction (CP) is a powerful framework for quantifying uncertainty in machine learning models, offering reliable predictions with finite-sample coverage guarantees. When applied to classification, CP produces a prediction set of possible labels that is guaranteed to contain the true label with high probability, regardless of the underlying classifier. However, standard CP treats classes as flat and unstructured, ignoring domain knowledge such as semantic relationships or hierarchical structure among class labels. This paper presents hierarchical conformal classification (HCC), an extension of CP that incorporates class hierarchies into both the structure and semantics of prediction sets. We formulate HCC as a constrained optimization problem whose solutions yield prediction sets composed of nodes at different levels of the hierarchy, while maintaining coverage guarantees. To address the combinatorial nature of the problem, we formally show that a much smaller, well-structured subset of candidate solutions suffices to ensure coverage while upholding optimality. An empirical evaluation on three new benchmarks consisting of audio, image, and text data highlights the advantages of our approach, and a user study shows that annotators significantly prefer hierarchical over flat prediction sets. 

**Abstract (ZH)**: 层次化 conformal 分类（HCC）： incorporate 类别层次结构到 prediction sets 的结构和语义中 

---
# ViTAD: Timing Violation-Aware Debugging of RTL Code using Large Language Models 

**Title (ZH)**: ViTAD: 基于大型语言模型的RTL代码 Timing Violation 意识调试 

**Authors**: Wenhao Lv, Yingjie Xia, Xiyuan Chen, Li Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13257)  

**Abstract**: In modern Very Large Scale Integrated (VLSI) circuit design flow, the Register-Transfer Level (RTL) stage presents a critical opportunity for timing optimization. Addressing timing violations at this early stage is essential, as modern systems demand higher speeds, where even minor timing violations can lead to functional failures or system crashes. However, traditional timing optimization heavily relies on manual expertise, requiring engineers to iteratively analyze timing reports and debug. To automate this process, this paper proposes ViTAD, a method that efficiently analyzes the root causes of timing violations and dynamically generates targeted repair strategies. Specifically, we first parse Verilog code and timing reports to construct a Signal Timing Dependency Graph (STDG). Based on the STDG, we perform violation path analysis and use large language models (LLMs) to infer the root causes of violations. Finally, by analyzing the causes of violations, we selectively retrieve relevant debugging knowledge from a domain-specific knowledge base to generate customized repair solutions. To evaluate the effectiveness of our method, we construct a timing violation dataset based on real-world open-source projects. This dataset contains 54 cases of violations. Experimental results show that our method achieves a 73.68% success rate in repairing timing violations, while the baseline using only LLM is 54.38%. Our method improves the success rate by 19.30%. 

**Abstract (ZH)**: 现代Very Large Scale Integrated (VLSI)电路设计流程中，Register-Transfer Level (RTL)阶段提供了关键的时序优化机会。在这一早期阶段解决时序违规至关重要，因为现代系统要求更高的速度，即使是很小的时序违规也可能导致功能失效或系统崩溃。然而，传统的时序优化高度依赖人工专业知识，要求工程师反复分析时序报告并调试。为此，本文提出ViTAD方法，该方法能够高效地分析时序违规的根本原因，并动态生成针对性的修复策略。具体而言，我们首先解析Verilog代码和时序报告，构建信号时序依赖图(STDG)。基于STDG，我们执行违规路径分析，并使用大型语言模型(LLMs)推断违规的根本原因。最后，通过分析违规原因，我们从特定领域的知识库中选择性地检索相关调试知识，生成定制化的修复解决方案。为评估方法的有效性，我们基于实际开源项目构建了一个时序违规数据集，该数据集包含54个违规案例。实验结果表明，我们的方法在修复时序违规方面的成功率达到了73.68%，而仅使用LLM的基础方法为54.38%。我们的方法提高了成功率19.30%。 

---
# Goal-Directedness is in the Eye of the Beholder 

**Title (ZH)**: 目标导向性在于观察者的视角。 

**Authors**: Nina Rajcic, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.13247)  

**Abstract**: Our ability to predict the behavior of complex agents turns on the attribution of goals. Probing for goal-directed behavior comes in two flavors: Behavioral and mechanistic. The former proposes that goal-directedness can be estimated through behavioral observation, whereas the latter attempts to probe for goals in internal model states. We work through the assumptions behind both approaches, identifying technical and conceptual problems that arise from formalizing goals in agent systems. We arrive at the perhaps surprising position that goal-directedness cannot be measured objectively. We outline new directions for modeling goal-directedness as an emergent property of dynamic, multi-agent systems. 

**Abstract (ZH)**: 我们预测复杂代理行为的能力取决于目标的归因。探求目标导向行为有两种方式：行为方式和机制方式。前者认为可以通过行为观察估算目标导向性，后者则尝试在内部模型状态中探求目标。我们探讨了这两种方法背后的假设，指出了在代理系统中正式化目标时出现的技术和概念问题。我们得出一个或许令人惊讶的结论：目标导向性无法客观测量。我们概述了将目标导向性建模为动态多代理系统 emergent 属性的新方向。 

---
# Involuntary Jailbreak 

**Title (ZH)**: 非自愿越狱 

**Authors**: Yangyang Guo, Yangyan Li, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.13246)  

**Abstract**: In this study, we disclose a worrying new vulnerability in Large Language Models (LLMs), which we term \textbf{involuntary jailbreak}. Unlike existing jailbreak attacks, this weakness is distinct in that it does not involve a specific attack objective, such as generating instructions for \textit{building a bomb}. Prior attack methods predominantly target localized components of the LLM guardrail. In contrast, involuntary jailbreaks may potentially compromise the entire guardrail structure, which our method reveals to be surprisingly fragile. We merely employ a single universal prompt to achieve this goal. In particular, we instruct LLMs to generate several questions that would typically be rejected, along with their corresponding in-depth responses (rather than a refusal). Remarkably, this simple prompt strategy consistently jailbreaks the majority of leading LLMs, including Claude Opus 4.1, Grok 4, Gemini 2.5 Pro, and GPT 4.1. We hope this problem can motivate researchers and practitioners to re-evaluate the robustness of LLM guardrails and contribute to stronger safety alignment in future. 

**Abstract (ZH)**: 本研究揭示了大型语言模型（LLMs）中的一个令人担忧的新漏洞，我们称之为**非自愿 Jailbreak**。这一弱点与众不同之处在于，它不涉及特定的攻击目标，例如生成制作炸弹的指令。以往的攻击方法主要针对 LLM 防护栏的局部组件。相比之下，非自愿 Jailbreak 有可能在整个防护栏结构上造成破坏，而我们的方法揭示了这一结构出奇地脆弱。我们仅使用一个通用提示便实现了这一目标。具体来说，我们指示 LLM 生成一些通常会被拒绝的问题及其相应的深入回答（而不是直接拒绝）。令人惊讶的是，这一简单的提示策略成功地攻破了包括 Claude Opus 4.1、Grok 4、Gemini 2.5 Pro 和 GPT 4.1 在内的大多数顶级 LLM。我们希望这一问题能够促使研究人员和实践者重新评估 LLM 防护栏的 robustness，并为未来更强大的安全性对齐做出贡献。 

---
# Quantifying Loss Aversion in Cyber Adversaries via LLM Analysis 

**Title (ZH)**: 通过大语言模型分析量化网络对手的损失回避程度 

**Authors**: Soham Hans, Nikolos Gurney, Stacy Marsella, Sofia Hirschmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.13240)  

**Abstract**: Understanding and quantifying human cognitive biases from empirical data has long posed a formidable challenge, particularly in cybersecurity, where defending against unknown adversaries is paramount. Traditional cyber defense strategies have largely focused on fortification, while some approaches attempt to anticipate attacker strategies by mapping them to cognitive vulnerabilities, yet they fall short in dynamically interpreting attacks in progress. In recognition of this gap, IARPA's ReSCIND program seeks to infer, defend against, and even exploit attacker cognitive traits. In this paper, we present a novel methodology that leverages large language models (LLMs) to extract quantifiable insights into the cognitive bias of loss aversion from hacker behavior. Our data are collected from an experiment in which hackers were recruited to attack a controlled demonstration network. We process the hacker generated notes using LLMs using it to segment the various actions and correlate the actions to predefined persistence mechanisms used by hackers. By correlating the implementation of these mechanisms with various operational triggers, our analysis provides new insights into how loss aversion manifests in hacker decision-making. The results demonstrate that LLMs can effectively dissect and interpret nuanced behavioral patterns, thereby offering a transformative approach to enhancing cyber defense strategies through real-time, behavior-based analysis. 

**Abstract (ZH)**: 从实验数据理解并量化人类认知偏差：利用大型语言模型揭示黑客行为中的损失厌恶认知偏差 

---
# Uncertainty-Aware Learning Policy for Reliable Pulmonary Nodule Detection on Chest X-Ray 

**Title (ZH)**: 面向胸片中肺结节检测的不确定性aware学习策略 

**Authors**: Hyeonjin Choi, Jinse Kim, Dong-yeon Yoo, Ju-sung Sun, Jung-won Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13236)  

**Abstract**: Early detection and rapid intervention of lung cancer are crucial. Nonetheless, ensuring an accurate diagnosis is challenging, as physicians' ability to interpret chest X-rays varies significantly depending on their experience and degree of fatigue. Although medical AI has been rapidly advancing to assist in diagnosis, physicians' trust in such systems remains limited, preventing widespread clinical adoption. This skepticism fundamentally stems from concerns about its diagnostic uncertainty. In clinical diagnosis, physicians utilize extensive background knowledge and clinical experience. In contrast, medical AI primarily relies on repetitive learning of the target lesion to generate diagnoses based solely on that data. In other words, medical AI does not possess sufficient knowledge to render a diagnosis, leading to diagnostic uncertainty. Thus, this study suggests an Uncertainty-Aware Learning Policy that can address the issue of knowledge deficiency by learning the physicians' background knowledge alongside the Chest X-ray lesion information. We used 2,517 lesion-free images and 656 nodule images, all obtained from Ajou University Hospital. The proposed model attained 92% (IoU 0.2 / FPPI 2) with a 10% enhancement in sensitivity compared to the baseline model while also decreasing entropy as a measure of uncertainty by 0.2. 

**Abstract (ZH)**: 早期检测与迅速干预肺癌至关重要。然而，确保准确诊断极具挑战性，因为医生解读胸部X光的能力因经验程度和疲劳程度而异。尽管医学AI已迅速发展以辅助诊断，但医生对其系统的信任程度有限，阻碍了其在临床中的广泛应用。这种怀疑从根本上源于对诊断不确定性的担忧。在临床诊断中，医生利用广泛的背景知识和临床经验。相比之下，医学AI主要依靠重复学习目标病灶来生成诊断，仅基于那组数据。换句话说，医学AI缺乏足够的知识进行诊断，导致诊断不确定性。因此，本研究提出一种awareness of uncertainty学习策略，该策略通过同时学习医生的背景知识和胸部X光病灶信息，以解决知识不足的问题。我们使用了2,517张无病灶图像和656张结节图像，所有数据均来自 Ajou大学医院。所提出的模型在IoU为0.2和FPPI为2的情况下达到了92%的检测率，与基线模型相比，灵敏度提高了10%，同时通过减少不确定性度量（熵）0.2来降低不确定性。 

---
# The Role of AI in Facilitating Interdisciplinary Collaboration: Evidence from AlphaFold 

**Title (ZH)**: AI在促进跨学科合作中的作用：来自AlphaFold的证据 

**Authors**: Naixuan Zhao, Chunli Wei, Xinyan Zhang, Jiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13234)  

**Abstract**: The acceleration of artificial intelligence (AI) in science is recognized and many scholars have begun to explore its role in interdisciplinary collaboration. However, the mechanisms and extent of this impact are still unclear. This study, using AlphaFold's impact on structural biologists, examines how AI technologies influence interdisciplinary collaborative patterns. By analyzing 1,247 AlphaFold-related papers and 7,700 authors from Scopus, we employ bibliometric analysis and causal inference to compare interdisciplinary collaboration between AlphaFold adopters and non-adopters. Contrary to the widespread belief that AI facilitates interdisciplinary collaboration, our findings show that AlphaFold increased structural biology-computer science collaborations by just 0.48%, with no measurable effect on other disciplines. Specifically, AI creates interdisciplinary collaboration demands with specific disciplines due to its technical characteristics, but this demand is weakened by technological democratization and other factors. These findings demonstrate that artificial intelligence (AI) alone has limited efficacy in bridging disciplinary divides or fostering meaningful interdisciplinary collaboration. 

**Abstract (ZH)**: 人工智能（AI）在科学中的加速应用及其对跨学科合作的影响：以AlphaFold为例的研究 

---
# Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System 

**Title (ZH)**: 在异构内存系统中通过动态KV缓存放置加速LLM推理 

**Authors**: Yunhua Fang, Rui Xie, Asad Ul Haq, Linsen Ma, Kaoutar El Maghraoui, Naigang Wang, Meng Wang, Liu Liu, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13231)  

**Abstract**: Large Language Model (LLM) inference is increasingly constrained by memory bandwidth, with frequent access to the key-value (KV) cache dominating data movement. While attention sparsity reduces some memory traffic, the relevance of past tokens varies over time, requiring the full KV cache to remain accessible and sustaining pressure on both bandwidth and capacity. With advances in interconnects such as NVLink and LPDDR5X, modern AI hardware now integrates high-bandwidth memory (HBM) with high-speed off-package DRAM, making heterogeneous memory systems a practical solution. This work investigates dynamic KV cache placement across such systems to maximize aggregated bandwidth utilization under capacity constraints. Rather than proposing a specific scheduling policy, we formulate the placement problem mathematically and derive a theoretical upper bound, revealing substantial headroom for runtime optimization. To our knowledge, this is the first formal treatment of dynamic KV cache scheduling in heterogeneous memory systems for LLM inference. 

**Abstract (ZH)**: 大型语言模型（LLM）推理越来越多地受到内存带宽的限制，频繁访问键值（KV）缓存主导着数据移动。虽然注意力稀疏性减少了部分内存流量，但过去令牌的相关性会随时间变化，要求完整KV缓存保持可访问性，从而持续对带宽和容量造成压力。随着NVLink和LPDDR5X等互连技术的进步，现代AI硬件现在将高性能记忆体（HBM）与高速外部DRAM集成为一体，使异构内存系统成为可行的解决方案。本文探讨了在这些系统中动态放置KV缓存，以在容量受限条件下最大化聚合带宽利用率。我们并未提出具体的调度策略，而是从数学上形式化了放置问题，并推导出一个理论上限，揭示了运行时优化的巨大空间。据我们所知，这是首次对异构内存系统中LLM推理的动态KV缓存调度进行形式化处理的研究。 

---
# PreSem-Surf: RGB-D Surface Reconstruction with Progressive Semantic Modeling and SG-MLP Pre-Rendering Mechanism 

**Title (ZH)**: PreSem-Surf: 基于 progressive semantic modeling 和 SG-MLP 预渲染机制的 RGB-D 表面重建 

**Authors**: Yuyan Ye, Hang Xu, Yanghang Huang, Jiali Huang, Qian Weng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13228)  

**Abstract**: This paper proposes PreSem-Surf, an optimized method based on the Neural Radiance Field (NeRF) framework, capable of reconstructing high-quality scene surfaces from RGB-D sequences in a short time. The method integrates RGB, depth, and semantic information to improve reconstruction performance. Specifically, a novel SG-MLP sampling structure combined with PR-MLP (Preconditioning Multilayer Perceptron) is introduced for voxel pre-rendering, allowing the model to capture scene-related information earlier and better distinguish noise from local details. Furthermore, progressive semantic modeling is adopted to extract semantic information at increasing levels of precision, reducing training time while enhancing scene understanding. Experiments on seven synthetic scenes with six evaluation metrics show that PreSem-Surf achieves the best performance in C-L1, F-score, and IoU, while maintaining competitive results in NC, Accuracy, and Completeness, demonstrating its effectiveness and practical applicability. 

**Abstract (ZH)**: 基于Neural Radiance Field框架的PreSem-Surf：一种高效的RGB-D序列场景 surfaces 重建方法 

---
# MIRAGE: Towards AI-Generated Image Detection in the Wild 

**Title (ZH)**: MIRAGE:面向野生环境中的AI生成图像检测 

**Authors**: Cheng Xia, Manxi Lin, Jiexiang Tan, Xiaoxiong Du, Yang Qiu, Junjun Zheng, Xiangheng Kong, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13223)  

**Abstract**: The spreading of AI-generated images (AIGI), driven by advances in generative AI, poses a significant threat to information security and public trust. Existing AIGI detectors, while effective against images in clean laboratory settings, fail to generalize to in-the-wild scenarios. These real-world images are noisy, varying from ``obviously fake" images to realistic ones derived from multiple generative models and further edited for quality control. We address in-the-wild AIGI detection in this paper. We introduce Mirage, a challenging benchmark designed to emulate the complexity of in-the-wild AIGI. Mirage is constructed from two sources: (1) a large corpus of Internet-sourced AIGI verified by human experts, and (2) a synthesized dataset created through the collaboration between multiple expert generators, closely simulating the realistic AIGI in the wild. Building on this benchmark, we propose Mirage-R1, a vision-language model with heuristic-to-analytic reasoning, a reflective reasoning mechanism for AIGI detection. Mirage-R1 is trained in two stages: a supervised-fine-tuning cold start, followed by a reinforcement learning stage. By further adopting an inference-time adaptive thinking strategy, Mirage-R1 is able to provide either a quick judgment or a more robust and accurate conclusion, effectively balancing inference speed and performance. Extensive experiments show that our model leads state-of-the-art detectors by 5% and 10% on Mirage and the public benchmark, respectively. The benchmark and code will be made publicly available. 

**Abstract (ZH)**: AI生成图像在野检测：Mirage及其挑战基准 

---
# MCPSecBench: A Systematic Security Benchmark and Playground for Testing Model Context Protocols 

**Title (ZH)**: MCPSecBench: 一种模型上下文协议测试的系统性安全基准和实验平台 

**Authors**: Yixuan Yang, Daoyuan Wu, Yufan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13220)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications via the Model Context Protocol (MCP), a universal, open standard for connecting AI agents with data sources and external tools. While MCP enhances the capabilities of LLM-based agents, it also introduces new security risks and expands their attack surfaces. In this paper, we present the first systematic taxonomy of MCP security, identifying 17 attack types across 4 primary attack surfaces. We introduce MCPSecBench, a comprehensive security benchmark and playground that integrates prompt datasets, MCP servers, MCP clients, and attack scripts to evaluate these attacks across three major MCP providers. Our benchmark is modular and extensible, allowing researchers to incorporate custom implementations of clients, servers, and transport protocols for systematic security assessment. Experimental results show that over 85% of the identified attacks successfully compromise at least one platform, with core vulnerabilities universally affecting Claude, OpenAI, and Cursor, while prompt-based and tool-centric attacks exhibit considerable variability across different hosts and models. Overall, MCPSecBench standardizes the evaluation of MCP security and enables rigorous testing across all MCP layers. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过模型上下文协议（MCP）越来越多地融入实际应用中，MCP是一种通用的开放标准，用于连接AI代理与数据源和外部工具。虽然MCP提升了基于LLM的代理的能力，但也引入了新的安全风险并扩大了其攻击面。在本文中，我们提出了MCP安全的第一个系统性分类，识别出17种攻击类型，涉及4个主要攻击面。我们引入了MCPSecBench，这是一个综合的安全基准平台，集成了一系列数据集、MCP服务器、MCP客户端和攻击脚本，用于评估这三种主要MCP提供者中的攻击。该基准平台模块化且可扩展，允许研究人员纳入自定义的客户端、服务器和传输协议实现，以进行系统性安全评估。实验结果显示，超过85%的已识别攻击成功地至少攻破了一个平台，核心漏洞普遍影响Claude、OpenAI和Cursor，而基于提示和工具中心的攻击在不同主机和模型之间表现出较大的变异性。总体而言，MCPSecBench统一了MCP安全的评估标准，并能在MCP的所有层次上进行严格测试。 

---
# Deep Graph Neural Point Process For Learning Temporal Interactive Networks 

**Title (ZH)**: 深度图神经点过程学习时序交互网络 

**Authors**: Su Chen, Xiaohua Qi, Xixun Lin, Yanmin Shang, Xiaolin Xu, Yangxi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13219)  

**Abstract**: Learning temporal interaction networks(TIN) is previously regarded as a coarse-grained multi-sequence prediction problem, ignoring the network topology structure influence. This paper addresses this limitation and a Deep Graph Neural Point Process(DGNPP) model for TIN is proposed. DGNPP consists of two key modules: the Node Aggregation Layer and the Self Attentive Layer. The Node Aggregation Layer captures topological structures to generate static representation for users and items, while the Self Attentive Layer dynamically updates embeddings over time. By incorporating both dynamic and static embeddings into the event intensity function and optimizing the model via maximum likelihood estimation, DGNPP predicts events and occurrence time effectively. Experimental evaluations on three public datasets demonstrate that DGNPP achieves superior performance in event prediction and time prediction tasks with high efficiency, significantly outperforming baseline models and effectively mitigating the limitations of prior approaches. 

**Abstract (ZH)**: 学习时序交互网络（TIN） previously被视为粗粒度的多序列预测问题，忽略了网络拓扑结构的影响。本文解决了这一局限，并提出了一种深度图神经点过程（DGNPP）模型用于TIN。DGNPP由两个关键模块组成：节点聚合层和自我注意层。节点聚合层捕获拓扑结构以生成用户和项目的静态表示，而自我注意层则动态更新时间上的嵌入表示。通过将动态和静态嵌入整合到事件强度函数中，并通过最大似然估计优化模型，DGNPP能够有效预测事件及其发生时间。实验评估表明，DGNPP在事件预测和时间预测任务中表现出色且效率高，显著优于基准模型，并有效缓解了先前方法的局限性。 

---
# Too Easily Fooled? Prompt Injection Breaks LLMs on Frustratingly Simple Multiple-Choice Questions 

**Title (ZH)**: 太容易受骗了吗？提示注入使语言模型在令人 frustratingly 简单的选择题上失效 

**Authors**: Xuyang Guo, Zekai Huang, Zhao Song, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13214)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong emergent abilities in complex reasoning and zero-shot generalization, showing unprecedented potential for LLM-as-a-judge applications in education, peer review, and data quality evaluation. However, their robustness under prompt injection attacks, where malicious instructions are embedded into the content to manipulate outputs, remains a significant concern. In this work, we explore a frustratingly simple yet effective attack setting to test whether LLMs can be easily misled. Specifically, we evaluate LLMs on basic arithmetic questions (e.g., "What is 3 + 2?") presented as either multiple-choice or true-false judgment problems within PDF files, where hidden prompts are injected into the file. Our results reveal that LLMs are indeed vulnerable to such hidden prompt injection attacks, even in these trivial scenarios, highlighting serious robustness risks for LLM-as-a-judge applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理和零样本泛化方面表现出强大的 emergent 能力，展示了在教育、同行评审和数据质量评估中的 LLM-as-a-judge 应用的巨大潜力。然而，在提示注入攻击下（恶意指令被嵌入内容以操控输出）的鲁棒性问题仍然是一个重大关注点。在本工作中，我们探索了一个令人沮丧的简单但有效的攻击设置，以测试LLMs是否容易被误导。具体地，我们在PDF文件中对LLMs进行基本算术问题（例如，“3 + 2 是多少？”）的评估，这些问题以多项选择或真伪判断的形式呈现，并在文件中注入了隐蔽提示。我们的研究结果表明，即使在这些简单的场景下，LLMs也容易受到隐蔽提示注入攻击的影响，突显了LLM-as-a-judge应用中严重的鲁棒性风险。 

---
# Research on Conversational Recommender System Considering Consumer Types 

**Title (ZH)**: 考虑消费者类型的对话推荐系统研究 

**Authors**: Yaying Luo, Hui Fang, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13209)  

**Abstract**: Conversational Recommender Systems (CRS) provide personalized services through multi-turn interactions, yet most existing methods overlook users' heterogeneous decision-making styles and knowledge levels, which constrains both accuracy and efficiency. To address this gap, we propose CT-CRS (Consumer Type-Enhanced Conversational Recommender System), a framework that integrates consumer type modeling into dialogue recommendation. Based on consumer type theory, we define four user categories--dependent, efficient, cautious, and expert--derived from two dimensions: decision-making style (maximizers vs. satisficers) and knowledge level (high vs. low). CT-CRS employs interaction histories and fine-tunes the large language model to automatically infer user types in real time, avoiding reliance on static questionnaires. We incorporate user types into state representation and design a type-adaptive policy that dynamically adjusts recommendation granularity, diversity, and attribute query complexity. To further optimize the dialogue policy, we adopt Inverse Reinforcement Learning (IRL), enabling the agent to approximate expert-like strategies conditioned on consumer type. Experiments on LastFM, Amazon-Book, and Yelp show that CTCRS improves recommendation success rate and reduces interaction turns compared to strong baselines. Ablation studies confirm that both consumer type modeling and IRL contribute significantly to performance gains. These results demonstrate that CT-CRS offers a scalable and interpretable solution for enhancing CRS personalization through the integration of psychological modeling and advanced policy optimization. 

**Abstract (ZH)**: 面向消费者类型的对话推荐系统（CT-CRS）：结合心理建模的个性化优化 

---
# Utilizing the RAIN method and Graph SAGE Model to Identify Effective Drug Combinations for Gastric Neoplasm Treatment 

**Title (ZH)**: 利用RAIN方法和Graph SAGE模型识别胃神经内分泌肿瘤的有效药物组合 

**Authors**: S. Z. Pirasteh, Ali A. Kiaei, Mahnaz Bush, Sabra Moghadam, Raha Aghaei, Behnaz Sadeghigol  

**Link**: [PDF](https://arxiv.org/pdf/2508.13207)  

**Abstract**: Background: Gastric neoplasm, primarily adenocarcinoma, is an aggressive cancer with high mortality, often diagnosed late, leading to complications like metastasis. Effective drug combinations are vital to address disease heterogeneity, enhance efficacy, reduce resistance, and improve patient outcomes. Methods: The RAIN method integrated Graph SAGE to propose drug combinations, using a graph model with p-value-weighted edges connecting drugs, genes, and proteins. NLP and systematic literature review (PubMed, Scopus, etc.) validated proposed drugs, followed by network meta-analysis to assess efficacy, implemented in Python. Results: Oxaliplatin, fluorouracil, and trastuzumab were identified as effective, supported by 61 studies. Fluorouracil alone had a p-value of 0.0229, improving to 0.0099 with trastuzumab, and 0.0069 for the triple combination, indicating superior efficacy. Conclusion: The RAIN method, combining AI and network meta-analysis, effectively identifies optimal drug combinations for gastric neoplasm, offering a promising strategy to enhance treatment outcomes and guide health policy. 

**Abstract (ZH)**: 背景：胃恶性肿瘤主要是腺癌，是一种具有高死亡率的侵袭性癌症，常常在晚期诊断，导致转移等并发症。有效的药物组合对于应对疾病异质性、增强疗效、减少抗药性并改善患者预后至关重要。方法：RAIN方法结合Graph SAGE提出药物组合，使用一个连接药物、基因和蛋白质的图模型，并通过p值加权的边进行连接。通过自然语言处理和系统文献回顾（PubMed、Scopus等）验证提出的药物，随后通过网络meta分析评估疗效，全部在Python中实施。结果：奥沙利铂、氟尿嘧啶和曲妥珠单抗被识别为有效的药物组合，有61项研究支持。单独使用氟尿嘧啶的p值为0.0229，加入曲妥珠单抗后降至0.0099，而三联组合的p值为0.0069，表明其有效性更优。结论：RAIN方法结合AI和网络meta分析，有效地识别出胃恶性肿瘤的最佳药物组合，为提高治疗效果和指导卫生政策提供了有前景的策略。 

---
# Benchmarking LLM-based Agents for Single-cell Omics Analysis 

**Title (ZH)**: 基于LLM的代理单细胞组学分析基准测试 

**Authors**: Yang Liu, Lu Zhou, Ruikun He, Rongbo Shen, Yixue Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13201)  

**Abstract**: The surge in multimodal single-cell omics data exposes limitations in traditional, manually defined analysis workflows. AI agents offer a paradigm shift, enabling adaptive planning, executable code generation, traceable decisions, and real-time knowledge fusion. However, the lack of a comprehensive benchmark critically hinders progress. We introduce a novel benchmarking evaluation system to rigorously assess agent capabilities in single-cell omics analysis. This system comprises: a unified platform compatible with diverse agent frameworks and LLMs; multidimensional metrics assessing cognitive program synthesis, collaboration, execution efficiency, bioinformatics knowledge integration, and task completion quality; and 50 diverse real-world single-cell omics analysis tasks spanning multi-omics, species, and sequencing technologies. Our evaluation reveals that Grok-3-beta achieves state-of-the-art performance among tested agent frameworks. Multi-agent frameworks significantly enhance collaboration and execution efficiency over single-agent approaches through specialized role division. Attribution analyses of agent capabilities identify that high-quality code generation is crucial for task success, and self-reflection has the most significant overall impact, followed by retrieval-augmented generation (RAG) and planning. This work highlights persistent challenges in code generation, long-context handling, and context-aware knowledge retrieval, providing a critical empirical foundation and best practices for developing robust AI agents in computational biology. 

**Abstract (ZH)**: 多模态单细胞组学数据激增揭示了传统手动定义分析工作流的局限性。AI代理提供了范式的转变，能够实现自适应规划、可执行代码生成、可追溯的决策和实时知识融合。然而，缺乏全面的基准测试严重阻碍了进步。我们提出了一种新的基准评估系统，以严格评估代理在单细胞组学分析中的能力。该系统包括：一个兼容多种代理框架和大规模语言模型的统一平台；多维度指标评估认知程序合成、协作、执行效率、生物信息学知识整合和任务完成质量；以及涵盖多组学、物种和测序技术的50个多样化的真实世界单细胞组学分析任务。我们的评估显示，在测试的代理框架中，Grok-3-beta 达到了最先进的性能。多代理框架通过专门的角色分工显著提高了协作和执行效率，超过单代理方法。代理能力归因分析表明，高质量代码生成对于任务成功至关重要，自我反思的影响最大，其次是检索增强生成（RAG）和规划。这项工作突出了代码生成、长上下文处理和上下文感知知识检索中的持续挑战，为在计算生物学中开发稳健的AI代理提供了关键的实证基础和最佳实践。 

---
# The Rise of Generative AI for Metal-Organic Framework Design and Synthesis 

**Title (ZH)**: 金属有机框架设计与合成中生成式AI的崛起 

**Authors**: Chenru Duan, Aditya Nandy, Shyam Chand Pal, Xin Yang, Wenhao Gao, Yuanqi Du, Hendrik Kraß, Yeonghun Kang, Varinia Bernales, Zuyang Ye, Tristan Pyle, Ray Yang, Zeqi Gu, Philippe Schwaller, Shengqian Ma, Shijing Sun, Alán Aspuru-Guzik, Seyed Mohamad Moosavi, Robert Wexler, Zhiling Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.13197)  

**Abstract**: Advances in generative artificial intelligence are transforming how metal-organic frameworks (MOFs) are designed and discovered. This Perspective introduces the shift from laborious enumeration of MOF candidates to generative approaches that can autonomously propose and synthesize in the laboratory new porous reticular structures on demand. We outline the progress of employing deep learning models, such as variational autoencoders, diffusion models, and large language model-based agents, that are fueled by the growing amount of available data from the MOF community and suggest novel crystalline materials designs. These generative tools can be combined with high-throughput computational screening and even automated experiments to form accelerated, closed-loop discovery pipelines. The result is a new paradigm for reticular chemistry in which AI algorithms more efficiently direct the search for high-performance MOF materials for clean air and energy applications. Finally, we highlight remaining challenges such as synthetic feasibility, dataset diversity, and the need for further integration of domain knowledge. 

**Abstract (ZH)**: 生成式人工智能的进步正在变革金属有机框架（MOFs）的设计与发现方式。本文概览了从耗时的MOF候选物枚举方法向能够自主提出并在实验室合成新多孔骨架结构的方法的转变。我们概述了利用变分自编码器、扩散模型和基于大型语言模型的代理等深度学习模型的应用进展，这些模型得益于越来越多的来自MOF社区的数据，并提出新型晶体材料设计。这些生成工具可以与高通量计算筛选和自动化实验相结合，形成加速的闭环发现流程。结果，这为晶态化学提供了一个新的范式，在此范式中，AI算法更有效地指导高性能MOF材料在清洁空气和能源应用中的搜索。最后，我们指出了剩余的挑战，如合成可行性、数据集多样性以及需要进一步整合专业知识。 

---
# Contextual Attention-Based Multimodal Fusion of LLM and CNN for Sentiment Analysis 

**Title (ZH)**: 基于上下文注意力的LLM和CNN多模态融合情感分析 

**Authors**: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2508.13196)  

**Abstract**: This paper introduces a novel approach for multimodal sentiment analysis on social media, particularly in the context of natural disasters, where understanding public sentiment is crucial for effective crisis management. Unlike conventional methods that process text and image modalities separately, our approach seamlessly integrates Convolutional Neural Network (CNN) based image analysis with Large Language Model (LLM) based text processing, leveraging Generative Pre-trained Transformer (GPT) and prompt engineering to extract sentiment relevant features from the CrisisMMD dataset. To effectively model intermodal relationships, we introduce a contextual attention mechanism within the fusion process. Leveraging contextual-attention layers, this mechanism effectively captures intermodality interactions, enhancing the model's comprehension of complex relationships between textual and visual data. The deep neural network architecture of our model learns from these fused features, leading to improved accuracy compared to existing baselines. Experimental results demonstrate significant advancements in classifying social media data into informative and noninformative categories across various natural disasters. Our model achieves a notable 2.43% increase in accuracy and 5.18% in F1-score, highlighting its efficacy in processing complex multimodal data. Beyond quantitative metrics, our approach provides deeper insight into the sentiments expressed during crises. The practical implications extend to real time disaster management, where enhanced sentiment analysis can optimize the accuracy of emergency interventions. By bridging the gap between multimodal analysis, LLM powered text understanding, and disaster response, our work presents a promising direction for Artificial Intelligence (AI) driven crisis management solutions. Keywords: 

**Abstract (ZH)**: 一种针对自然灾害情境下的社交媒体多模态情感分析的新型方法：基于上下文注意力机制的图像分析与语言模型文本处理融合 

---
# Preference Models assume Proportional Hazards of Utilities 

**Title (ZH)**: 偏好模型假设效用的比例危害。 

**Authors**: Chirag Nagpal  

**Link**: [PDF](https://arxiv.org/pdf/2508.13189)  

**Abstract**: Approaches for estimating preferences from human annotated data typically involves inducing a distribution over a ranked list of choices such as the Plackett-Luce model. Indeed, modern AI alignment tools such as Reward Modelling and Direct Preference Optimization are based on the statistical assumptions posed by the Plackett-Luce model. In this paper, I will connect the Plackett-Luce model to another classical and well known statistical model, the Cox Proportional Hazards model and attempt to shed some light on the implications of the connection therein. 

**Abstract (ZH)**: 基于人类标注数据估计偏好方法通常涉及诱导一个排序选择列表上的分布，如Plackett-Luce模型。事实上，现代AI对齐工具，如奖励建模和直接偏好优化，正是基于Plackett-Luce模型的统计假设。在本文中，我将连接Plackett-Luce模型与另一个经典且广为人知的统计模型——Cox比例风险模型，并尝试探讨其中连接的含义。 

---
# Combating Homelessness Stigma with LLMs: A New Multi-Modal Dataset for Bias Detection 

**Title (ZH)**: 使用大语言模型对抗无家可归者污名：一种新的多模态数据集用于偏见检测 

**Authors**: Jonathan A. Karr Jr., Benjamin F. Herbst, Ting Hua, Matthew Hauenstein, Georgina Curto, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2508.13187)  

**Abstract**: Homelessness is a persistent social challenge, impacting millions worldwide. Over 770,000 people experienced homelessness in the U.S. in 2024. Social stigmatization is a significant barrier to alleviation, shifting public perception, and influencing policymaking. Given that online and city council discourse reflect and influence part of public opinion, it provides valuable insights to identify and track social biases. This research contributes to alleviating homelessness by acting on public opinion. It introduces novel methods, building on natural language processing (NLP) and large language models (LLMs), to identify and measure PEH social bias expressed in digital spaces. We present a new, manually-annotated multi-modal dataset compiled from Reddit, X (formerly Twitter), news articles, and city council meeting minutes across 10 U.S. cities. This unique dataset provides evidence of the typologies of homelessness bias described in the literature. In order to scale up and automate the detection of homelessness bias online, we evaluate LLMs as classifiers. We applied both zero-shot and few-shot classification techniques to this data. We utilized local LLMs (Llama 3.2 3B Instruct, Qwen 2.5 7B Instruct, and Phi4 Instruct Mini) as well as closed-source API models (GPT-4.1, Gemini 2.5 Pro, and Grok-4). Our findings reveal that although there are significant inconsistencies in local LLM zero-shot classification, the in-context learning classification scores of local LLMs approach the classification scores of closed-source LLMs. Furthermore, LLMs outperform BERT when averaging across all categories. This work aims to raise awareness about the pervasive bias against PEH, develop new indicators to inform policy, and ultimately enhance the fairness and ethical application of Generative AI technologies. 

**Abstract (ZH)**: 无家屋问题是持续存在的的社会挑战，影响着全球数百万人的生活。据2 2 2  2 2 2年数据显示，在美国，2 多有 7  on 7  on 7  people  people  on  on  on  7  on  on  on  on  有人  起有  on  on  on  on  on  on  on  on  on  人  on  近年7  on  on  人  上  无家 � línea 是 � 对代际传递造成 的偏见及 幋对 在  on  on  还on  on  on  进行 影响人们的观点变革和 影响政策制定。鉴于网络社交媒体和 城市议会的讨论在舆论形成中扮演的角色， 提供 通过分析社交媒体上的的舆论可以部分 提取 无家 層的的洞见 on 了解社会偏见。本研究旨在通过自然语言处理（NLP on 生成式大型语言模型（LLMs on 对无家屋偏见的系统地监测和 量化。我们构建了一个多模态数据集 on 从 Reddit on  X on 新闻文章 on 以及美国城市议会记录中 分对 这  on  on  进一步进行了偏见分类研究。通过自然语言生成式大模型 on 如 和传统零样本分类法和 少量样本分类法 on 我们测试了不同的自动偏见识别方法。研究结果表明 on 在 基地 语言模型在零样本分类的任务上存在显著挑战 但在特定上下文学习上可以获得与非开放源语言模型媲美的表现 on 吉上 语言模型在综合评估上优于 BERT on 不 进一步促 我们强调了对 在个人和 礿 政策层面上应重视无家无视角偏见 on 以及on 生成式人工智能技术的公平与 遁伦理应用。 

---
# MM-BrowseComp: A Comprehensive Benchmark for Multimodal Browsing Agents 

**Title (ZH)**: MM-BrowseComp: 多模态浏览代理的综合基准 

**Authors**: Shilong Li, Xingyuan Bu, Wenjie Wang, Jiaheng Liu, Jun Dong, Haoyang He, Hao Lu, Haozhe Zhang, Chenchen Jing, Zhen Li, Chuanhao Li, Jiayi Tian, Chenchen Zhang, Tianhao Peng, Yancheng He, Jihao Gu, Yuanxing Zhang, Jian Yang, Ge Zhang, Wenhao Huang, Wangchunshu Zhou, Zhaoxiang Zhang, Ruizhe Ding, Shilei Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13186)  

**Abstract**: AI agents with advanced reasoning and tool use capabilities have demonstrated impressive performance in web browsing for deep search. While existing benchmarks such as BrowseComp evaluate these browsing abilities, they primarily focus on textual information, overlooking the prevalence of multimodal content. To bridge this gap, we introduce MM-BrowseComp, a novel benchmark comprising 224 challenging, hand-crafted questions specifically designed to assess agents' multimodal retrieval and reasoning capabilities. These questions often incorporate images in prompts, and crucial information encountered during the search and reasoning process may also be embedded within images or videos on webpages. Consequently, methods relying solely on text prove insufficient for our benchmark. Additionally, we provide a verified checklist for each question, enabling fine-grained analysis of multimodal dependencies and reasoning paths. Our comprehensive evaluation of state-of-the-art models on MM-BrowseComp reveals that even top models like OpenAI o3 with tools achieve only 29.02\% accuracy, highlighting the suboptimal multimodal capabilities and lack of native multimodal reasoning in current models. 

**Abstract (ZH)**: 具有高级推理和工具使用能力的AI代理在深度网页搜索中表现出色。现有基准如BrowseComp评估这些浏览能力，但主要侧重于文本信息，忽视了多模态内容的普遍性。为弥补这一差距，我们介绍了MM-BrowseComp，这是一个包含224个具有挑战性的手动生成问题的新基准，旨在评估代理的多模态检索和推理能力。这些问题通常在提示中包含图像，而在搜索和推理过程中也可能从网页上的图像或视频中提取关键信息。因此，仅依赖文本的方法对我们的基准证明是不够的。此外，我们还为每个问题提供了验证列表，以支持对多模态依赖性和推理路径的精细分析。对MM-BrowseComp上最先进的模型的全面评估表明，即使是像OpenAI o3这样的顶级模型在工具辅助下的准确率也只有29.02%，这突显了当前模型在多模态能力和原生多模态推理方面的不足。 

---
# Using Artificial Intuition in Distinct, Minimalist Classification of Scientific Abstracts for Management of Technology Portfolios 

**Title (ZH)**: 使用人工直觉对科技组合摘要进行精确分类管理 

**Authors**: Prateek Ranka, Fred Morstatter, Andrea Belz, Alexandra Graddy-Reed  

**Link**: [PDF](https://arxiv.org/pdf/2508.13182)  

**Abstract**: Classification of scientific abstracts is useful for strategic activities but challenging to automate because the sparse text provides few contextual clues. Metadata associated with the scientific publication can be used to improve performance but still often requires a semi-supervised setting. Moreover, such schemes may generate labels that lack distinction -- namely, they overlap and thus do not uniquely define the abstract. In contrast, experts label and sort these texts with ease. Here we describe an application of a process we call artificial intuition to replicate the expert's approach, using a Large Language Model (LLM) to generate metadata. We use publicly available abstracts from the United States National Science Foundation to create a set of labels, and then we test this on a set of abstracts from the Chinese National Natural Science Foundation to examine funding trends. We demonstrate the feasibility of this method for research portfolio management, technology scouting, and other strategic activities. 

**Abstract (ZH)**: ### 论文标题翻译

科学摘要分类对于战略活动非常有用，，但自动化这一过程极具挑战性，，因为摘要内容稀少，，提供的背景线索极少。可以通过关联的元数据来改进性能，但是通常需要在半监督环境下进行。此外，，这样的方案可能会生成缺乏区分性的标签—也就是说它们相互有交叉重叠且无法唯一定义摘要。鉴于专家可以轻松地地编写和分类这些文本。我们采用了我们称为人工直觉的过程来复制专家的作用。使用一个大规模语言模型（LLM）生成元数据。使用来自美国国立科学基金会的公开摘要数据集来生成标签，然后在来自中国国家自然科学基金会的摘要数据集上进行验证和评估，以考察此方法在科研项目管理、技术和战略规划上的的可行性。 

---
# Toward an African Agenda for AI Safety 

**Title (ZH)**: 面向非洲的AI安全议程 

**Authors**: Samuel T. Segun, Rachel Adams, Ana Florido, Scott Timcke, Jonathan Shock, Leah Junck, Fola Adeleke, Nicolas Grossman, Ayantola Alayande, Jerry John Kponyo, Matthew Smith, Dickson Marfo Fosu, Prince Dawson Tetteh, Juliet Arthur, Stephanie Kasaon, Odilile Ayodele, Laetitia Badolo, Paul Plantinga, Michael Gastrow, Sumaya Nur Adan, Joanna Wiaterek, Cecil Abungu, Kojo Apeagyei, Luise Eder, Tegawende Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2508.13179)  

**Abstract**: This paper maps Africa's distinctive AI risk profile, from deepfake fuelled electoral interference and data colonial dependency to compute scarcity, labour disruption and disproportionate exposure to climate driven environmental costs. While major benefits are promised to accrue, the availability, development and adoption of AI also mean that African people and countries face particular AI safety risks, from large scale labour market disruptions to the nefarious use of AI to manipulate public opinion. To date, African perspectives have not been meaningfully integrated into global debates and processes regarding AI safety, leaving African stakeholders with limited influence over the emerging global AI safety governance agenda. While there are Computer Incident Response Teams on the continent, none hosts a dedicated AI Safety Institute or office. We propose a five-point action plan centred on (i) a policy approach that foregrounds the protection of the human rights of those most vulnerable to experiencing the harmful socio-economic effects of AI; (ii) the establishment of an African AI Safety Institute; (iii) promote public AI literacy and awareness; (iv) development of early warning system with inclusive benchmark suites for 25+ African languages; and (v) an annual AU-level AI Safety & Security Forum. 

**Abstract (ZH)**: 这篇论文映射了非洲独特的AI风险画像，从深度造假选举干预和数据殖民依赖，到计算资源稀缺、劳动力市场扰乱以及对由气候驱动的环境成本的不成比例暴露。虽然AI带来的好处受到期待，但AI的可用性、开发和采用也意味着非洲人民和国家面临特定的AI安全风险，从大规模劳动力市场扰乱到利用AI manipulate公共意见的恶意行为。迄今为止，非洲视角尚未被有意义地纳入关于AI安全的全球辩论和进程中，导致非洲利益相关者在正在形成的全球AI安全治理议程中影响力有限。虽然非洲大陆上有计算机应急响应团队，但没有专门的AI安全研究所或办公室。我们提出一个五点行动计划，重点在于（i）一种以保护最易遭受AI有害社会经济影响的人类权利为中心的政策方法；（ii）建立非洲AI安全研究所；（iii）促进公众AI素养和意识；（iv）开发适用于25种以上非洲语言的早期预警系统和包容性基准测试套件；以及（v）每年举办一次非洲联盟层面的AI安全与安全论坛。 

---
# White-Box Reasoning: Synergizing LLM Strategy and gm/Id Data for Automated Analog Circuit Design 

**Title (ZH)**: 白盒推理：结合LLM策略和gm/Id数据的自动化模拟电路设计 

**Authors**: Jianqiu Chen, Siqi Li, Xu He  

**Link**: [PDF](https://arxiv.org/pdf/2508.13172)  

**Abstract**: Analog IC design is a bottleneck due to its reliance on experience and inefficient simulations, as traditional formulas fail in advanced nodes. Applying Large Language Models (LLMs) directly to this problem risks mere "guessing" without engineering principles. We present a "synergistic reasoning" framework that integrates an LLM's strategic reasoning with the physical precision of the gm/Id methodology. By empowering the LLM with gm/Id lookup tables, it becomes a quantitative, data-driven design partner.
We validated this on a two-stage op-amp, where our framework enabled the Gemini model to meet all TT corner specs in 5 iterations and extended optimization to all PVT corners. A crucial ablation study proved gm/Id data is key for this efficiency and precision; without it, the LLM is slower and deviates. Compared to a senior engineer's design, our framework achieves quasi-expert quality with an order-of-magnitude improvement in efficiency. This work validates a path for true analog design automation by combining LLM reasoning with scientific circuit design methodologies. 

**Abstract (ZH)**: 模拟IC设计因依赖经验和低效的仿真而在先进节点中成为瓶颈，传统的公式在高级节点中失效。直接将大规模语言模型（LLMs）应用于此问题可能会导致无工程原理的“猜测”。我们提出了一种“协同推理”框架，将LLM的战略推理与gm/Id方法的物理精确性相结合。通过赋予LLM gm/Id查找表，它成为定量的数据驱动设计伙伴。

我们在一个两级运算放大器上进行了验证，其中我们的框架使Gemini模型在5次迭代中满足所有TT角规格，并将优化扩展到所有PVT角。关键的消融研究证明gm/Id数据对于这种效率和精度是关键的；没有它，LLM会更慢并偏离。与资深工程师的设计相比，我们的框架以数量级提高的效率实现了接近专家级的质量。本工作验证了通过结合LLM推理和科学电路设计方法来实现真正模拟设计自动化的路径。 

---
# Sustainable AI Training via Hardware-Software Co-Design on NVIDIA, AMD, and Emerging GPU Architectures 

**Title (ZH)**: 基于 NVIDIA、AMD 及新兴 GPU 架构的硬件-软件协同设计可持续 AI 训练 

**Authors**: Yashasvi Makin, Rahul Maliakkal  

**Link**: [PDF](https://arxiv.org/pdf/2508.13163)  

**Abstract**: In particular, large-scale deep learning and artificial intelligence model training uses a lot of computational power and energy, so it poses serious sustainability issues. The fast rise in model complexity has resulted in exponential increases in energy consumption, increasing the demand for techniques maximizing computational efficiency and lowering environmental impact. This work explores environmentally driven performance optimization methods especially intended for advanced GPU architectures from NVIDIA, AMD, and other emerging GPU architectures. Our main focus is on investigating hardware-software co-design techniques meant to significantly increase memory-level and kernel-level operations, so improving performance-per-watt measures. Our thorough research encompasses evaluations of specialized tensor and matrix cores, advanced memory optimization methods, and creative integration approaches that taken together result in notable energy efficiency increases. We also discuss important software-level optimizations that augment hardware capability including mixed-precision arithmetic, advanced energy-aware scheduling algorithms, and compiler-driven kernel enhancements. Moreover, we methodically point out important research gaps and suggest future directions necessary to create really sustainable artificial intelligence systems. This paper emphasizes how major increases in training efficiency can be obtained by co-design of hardware and software, so lowering the environmental impact of artificial intelligence without compromising performance. To back up our analysis, we use real-world case studies from top companies like Meta, Google, Amazon, and others that show how these sustainable AI training methods are used in the real world. 

**Abstract (ZH)**: 大规模深度学习和人工智能模型训练消耗大量计算资源和能源，导致严重的可持续性问题。模型复杂度的快速提高导致能源消耗呈指数级增长，增加了提高计算效率和降低环境影响的技术需求。本研究探索了特别针对NVIDIA、AMD及其他新兴GPU架构的环境驱动型性能优化方法，重点关注硬件-软件协同设计技术以显著提高内存级和内核级操作，从而提升单位瓦特性能。我们的深入研究涵盖了专用张量和矩阵核的评估、高级内存优化方法以及创新的集成方法，这些方法结合起来能够显著提高能效。我们还讨论了在硬件能力基础上的软件级优化技术，包括混合精度算术、高级能效调度算法和编译器驱动的内核增强。此外，我们系统地指出现有研究中的重要空白，并建议未来发展方向，以创建真正可持续的人工智能系统。本文强调了通过硬件和软件协同设计，可以在不牺牲性能的情况下降低人工智能的环境影响，从而获得大幅提高训练效率的方式。为了支持我们的分析，我们使用来自Meta、Google、Amazon等顶级公司的实际案例研究，展示了这些可持续的AI训练方法在实际中的应用。 

---
# Piano: A Multi-Constraint Pin Assignment-Aware Floorplanner 

**Title (ZH)**: 钢琴：一种多约束针脚分配感知的布局规划器 

**Authors**: Zhexuan Xu, Kexin Zhou, Jie Wang, Zijie Geng, Siyuan Xu, Shixiong Kai, Mingxuan Yuan, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13161)  

**Abstract**: Floorplanning is a critical step in VLSI physical design, increasingly complicated by modern constraints such as fixed-outline requirements, whitespace removal, and the presence of pre-placed modules. In addition, the assignment of pins on module boundaries significantly impacts the performance of subsequent stages, including detailed placement and routing. However, traditional floorplanners often overlook pin assignment with modern constraints during the floorplanning stage. In this work, we introduce Piano, a floorplanning framework that simultaneously optimizes module placement and pin assignment under multiple constraints. Specifically, we construct a graph based on the geometric relationships among modules and their netlist connections, then iteratively search for shortest paths to determine pin assignments. This graph-based method also enables accurate evaluation of feedthrough and unplaced pins, thereby guiding overall layout quality. To further improve the design, we adopt a whitespace removal strategy and employ three local optimizers to enhance layout metrics under multi-constraint scenarios. Experimental results on widely used benchmark circuits demonstrate that Piano achieves an average 6.81% reduction in HPWL, a 13.39% decrease in feedthrough wirelength, a 16.36% reduction in the number of feedthrough modules, and a 21.21% drop in unplaced pins, while maintaining zero whitespace. 

**Abstract (ZH)**: 基于多约束条件下的模块放置与引脚分配优化框架Piano 

---
# Image2Net: Datasets, Benchmark and Hybrid Framework to Convert Analog Circuit Diagrams into Netlists 

**Title (ZH)**: Image2Net: 数据集、基准和混合框架，用于将模拟电路图转换为网表 

**Authors**: Haohang Xu, Chengjie Liu, Qihang Wang, Wenhao Huang, Yongjian Xu, Weiyu Chen, Anlan Peng, Zhijun Li, Bo Li, Lei Qi, Jun Yang, Yuan Du, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.13157)  

**Abstract**: Large Language Model (LLM) exhibits great potential in designing of analog integrated circuits (IC) because of its excellence in abstraction and generalization for knowledge. However, further development of LLM-based analog ICs heavily relies on textual description of analog ICs, while existing analog ICs are mostly illustrated in image-based circuit diagrams rather than text-based netlists. Converting circuit diagrams to netlists help LLMs to enrich the knowledge of analog IC. Nevertheless, previously proposed conversion frameworks face challenges in further application because of limited support of image styles and circuit elements. Up to now, it still remains a challenging task to effectively convert complex circuit diagrams into netlists. To this end, this paper constructs and opensources a new dataset with rich styles of circuit diagrams as well as balanced distribution of simple and complex analog ICs. And a hybrid framework, named Image2Net, is proposed for practical conversion from circuit diagrams to netlists. The netlist edit distance (NED) is also introduced to precisely assess the difference between the converted netlists and ground truth. Based on our benchmark, Image2Net achieves 80.77\% successful rate, which is 34.62\%-45.19\% higher than previous works. Specifically, the proposed work shows 0.116 averaged NED, which is 62.1\%-69.6\% lower than state-of-the-arts. 

**Abstract (ZH)**: 基于图像到网表转换的新数据集及Image2Net混合框架：复杂电路图到网表的有效转换 

---
# EvoVerilog: Large Langugage Model Assisted Evolution of Verilog Code 

**Title (ZH)**: EvoVerilog: 大型语言模型辅助的Verilog代码进化 

**Authors**: Ping Guo, Yiting Wang, Wanghao Ye, Yexiao He, Ziyao Wang, Xiaopeng Dai, Ang Li, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13156)  

**Abstract**: Large Language Models (LLMs) have demonstrated great potential in automating the generation of Verilog hardware description language code for hardware design. This automation is critical to reducing human effort in the complex and error-prone process of hardware design.
However, existing approaches predominantly rely on human intervention and fine-tuning using curated datasets, limiting their scalability in automated design workflows.
Although recent iterative search techniques have emerged, they often fail to explore diverse design solutions and may underperform simpler approaches such as repeated prompting.
To address these limitations, we introduce EvoVerilog, a novel framework that combines the reasoning capabilities of LLMs with evolutionary algorithms to automatically generate and refine Verilog code.
EvoVerilog utilizes a multiobjective, population-based search strategy to explore a wide range of design possibilities without requiring human intervention.
Extensive experiments demonstrate that EvoVerilog achieves state-of-the-art performance, with pass@10 scores of 89.1 and 80.2 on the VerilogEval-Machine and VerilogEval-Human benchmarks, respectively. Furthermore, the framework showcases its ability to explore diverse designs by simultaneously generating a variety of functional Verilog code while optimizing resource utilization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化生成Verilog硬件描述语言代码方面展现了巨大的潜力，这对于减少复杂且易出错的硬件设计过程中的人力投入至关重要。然而，现有方法主要依赖于人工干预和精心打造的数据集微调，这限制了它们在自动化设计流程中的可扩展性。尽管最近出现了迭代搜索技术，但它们往往未能探索多样化的设计解决方案，并且可能不如简单的重复提示方法表现得更好。为了解决这些限制，我们引入了EvoVerilog，这是一种结合了LLMs推理能力与进化算法的新型框架，用于自动生成和优化Verilog代码。EvoVerilog利用多目标、基于群体的搜索策略，在无需人工干预的情况下探索广泛的设计可能性。广泛的实验表明，EvoVerilog在VerilogEval-Machine和VerilogEval-Human基准测试中的@10通过率分别为89.1和80.2，表现最佳。此外，该框架展示了其探索多样化设计的能力，同时生成多种功能性Verilog代码并优化资源利用率。 

---
# Uncovering Emergent Physics Representations Learned In-Context by Large Language Models 

**Title (ZH)**: 揭示大语言模型在上下文中学到的 emergent 物理表示 

**Authors**: Yeongwoo Song, Jaeyong Bae, Dong-Kyum Kim, Hawoong Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2508.12448)  

**Abstract**: Large language models (LLMs) exhibit impressive in-context learning (ICL) abilities, enabling them to solve wide range of tasks via textual prompts alone. As these capabilities advance, the range of applicable domains continues to expand significantly. However, identifying the precise mechanisms or internal structures within LLMs that allow successful ICL across diverse, distinct classes of tasks remains elusive. Physics-based tasks offer a promising testbed for probing this challenge. Unlike synthetic sequences such as basic arithmetic or symbolic equations, physical systems provide experimentally controllable, real-world data based on structured dynamics grounded in fundamental principles. This makes them particularly suitable for studying the emergent reasoning behaviors of LLMs in a realistic yet tractable setting. Here, we mechanistically investigate the ICL ability of LLMs, especially focusing on their ability to reason about physics. Using a dynamics forecasting task in physical systems as a proxy, we evaluate whether LLMs can learn physics in context. We first show that the performance of dynamics forecasting in context improves with longer input contexts. To uncover how such capability emerges in LLMs, we analyze the model's residual stream activations using sparse autoencoders (SAEs). Our experiments reveal that the features captured by SAEs correlate with key physical variables, such as energy. These findings demonstrate that meaningful physical concepts are encoded within LLMs during in-context learning. In sum, our work provides a novel case study that broadens our understanding of how LLMs learn in context. 

**Abstract (ZH)**: 大型语言模型(Large Language Models, LLMs)在上下文学习(In-Context Learning, ICL)方面表现出色，能够仅通过文本提示解决广泛的任务。随着这些能力的进步，适用的领域范围也在不断扩大。然而，确定LLMs内部何种机制或结构允许其在不同类别的任务中成功进行ICL的具体机制仍然难以捉摸。基于物理的任务为探查这一挑战提供了有前景的测试平台。与基本算术或符号方程等合成序列不同，物理系统提供了基于基本原理和结构化动力学的可实验控制的真实世界数据。这使它们特别适合在实际可行的环境中研究LLMs的 emergent 推理行为。在此，我们从机理上探究了LLMs的ICL能力，特别是它们处理物理问题的能力。我们使用物理系统中的动力学预测任务作为代理，评估LLMs是否能在上下文中学习物理知识。我们首先展示了随着输入上下文的延长，动力学预测的性能有所提高。为了探索这种能力在LLMs中是如何产生的，我们使用稀疏自编码器(Sparse Autoencoders, SAEs)分析模型的残差流激活。实验结果表明，SAEs捕获的特征与关键物理变量，如能量，相关。这些发现表明，在上下文学习过程中，LLMs中编码了有意义的物理概念。总之，我们的研究提供了关于LLMs如何进行上下文学习的新颖案例研究，拓宽了我们对其理解。 

---
# TaoSR1: The Thinking Model for E-commerce Relevance Search 

**Title (ZH)**: TaoSR1：电子商务相关搜索的思维模型 

**Authors**: Chenhe Dong, Shaowei Yao, Pengkun Jiao, Jianhui Yang, Yiming Jin, Zerui Huang, Xiaojiang Zhou, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.12365)  

**Abstract**: Query-product relevance prediction is a core task in e-commerce search. BERT-based models excel at semantic matching but lack complex reasoning capabilities. While Large Language Models (LLMs) are explored, most still use discriminative fine-tuning or distill to smaller models for deployment. We propose a framework to directly deploy LLMs for this task, addressing key challenges: Chain-of-Thought (CoT) error accumulation, discriminative hallucination, and deployment feasibility. Our framework, TaoSR1, involves three stages: (1) Supervised Fine-Tuning (SFT) with CoT to instill reasoning; (2) Offline sampling with a pass@N strategy and Direct Preference Optimization (DPO) to improve generation quality; and (3) Difficulty-based dynamic sampling with Group Relative Policy Optimization (GRPO) to mitigate discriminative hallucination. Additionally, post-CoT processing and a cumulative probability-based partitioning method enable efficient online deployment. TaoSR1 significantly outperforms baselines on offline datasets and achieves substantial gains in online side-by-side human evaluations, introducing a novel paradigm for applying CoT reasoning to relevance classification. 

**Abstract (ZH)**: Query-product relevance prediction is a core-commerce core task. BERT-based models excel in semantic e but lack complex reasoning capability. e Large Language Models (LLMs) e explored e used use use discriminative fine fin small to smaller Small framework propose a framework to directly deploy LLM for e e e addressing e challenges: Chain-of-Thought ( eT e addition, discrimin e hallucination e and eployability feasibility e E e e velved e e e three e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e erfolgreich em e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e e.e 

---
# Preliminary suggestions for rigorous GPAI model evaluations 

**Title (ZH)**: 初步建议：严格的GPAI模型评估 

**Authors**: Patricia Paskov, Michael J. Byun, Kevin Wei, Toby Webster  

**Link**: [PDF](https://arxiv.org/pdf/2508.00875)  

**Abstract**: This document presents a preliminary compilation of general-purpose AI (GPAI) evaluation practices that may promote internal validity, external validity and reproducibility. It includes suggestions for human uplift studies and benchmark evaluations, as well as cross-cutting suggestions that may apply to many different evaluation types. Suggestions are organised across four stages in the evaluation life cycle: design, implementation, execution and documentation. Drawing from established practices in machine learning, statistics, psychology, economics, biology and other fields recognised to have important lessons for AI evaluation, these suggestions seek to contribute to the conversation on the nascent and evolving field of the science of GPAI evaluations. The intended audience of this document includes providers of GPAI models presenting systemic risk (GPAISR), for whom the EU AI Act lays out specific evaluation requirements; third-party evaluators; policymakers assessing the rigour of evaluations; and academic researchers developing or conducting GPAI evaluations. 

**Abstract (ZH)**: 本文档提出了促进一般用途人工智能（GPAI）内部有效性、外部有效性和可再现性的初步综合评价实践。它包括人类提升研究和基准评估的建议，以及可应用于多种评价类型的跨学科建议。这些建议按照评价生命周期的四个阶段（设计、实施、执行和文档）进行组织。本文档借鉴了机器学习、统计学、心理学、经济学、生物学及其他领域公认具有重要评价教训的实践，旨在为新兴且不断发展中的GPAI评价科学领域的讨论做出贡献。本文档的预期读者包括提供可能产生系统性风险的一般用途人工智能（GPAI）模型的供应商（欧盟人工智能法案为此类供应商列出了具体评价要求）、第三方评价者、评估评价严谨性的政策制定者以及进行或开发GPAI评价的学术研究人员。 

---
