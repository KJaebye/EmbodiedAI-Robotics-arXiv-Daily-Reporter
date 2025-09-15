# Mutual Information Tracks Policy Coherence in Reinforcement Learning 

**Title (ZH)**: 互信息反映强化学习中政策一致性 

**Authors**: Cameron Reid, Wael Hafez, Amirhossein Nazeri  

**Link**: [PDF](https://arxiv.org/pdf/2509.10423)  

**Abstract**: Reinforcement Learning (RL) agents deployed in real-world environments face degradation from sensor faults, actuator wear, and environmental shifts, yet lack intrinsic mechanisms to detect and diagnose these failures. We present an information-theoretic framework that reveals both the fundamental dynamics of RL and provides practical methods for diagnosing deployment-time anomalies. Through analysis of state-action mutual information patterns in a robotic control task, we first demonstrate that successful learning exhibits characteristic information signatures: mutual information between states and actions steadily increases from 0.84 to 2.83 bits (238% growth) despite growing state entropy, indicating that agents develop increasingly selective attention to task-relevant patterns. Intriguingly, states, actions and next states joint mutual information, MI(S,A;S'), follows an inverted U-curve, peaking during early learning before declining as the agent specializes suggesting a transition from broad exploration to efficient exploitation. More immediately actionable, we show that information metrics can differentially diagnose system failures: observation-space, i.e., states noise (sensor faults) produces broad collapses across all information channels with pronounced drops in state-action coupling, while action-space noise (actuator faults) selectively disrupts action-outcome predictability while preserving state-action relationships. This differential diagnostic capability demonstrated through controlled perturbation experiments enables precise fault localization without architectural modifications or performance degradation. By establishing information patterns as both signatures of learning and diagnostic for system health, we provide the foundation for adaptive RL systems capable of autonomous fault detection and policy adjustment based on information-theoretic principles. 

**Abstract (ZH)**: 基于信息论的强化学习代理故障诊断框架 

---
# Abduct, Act, Predict: Scaffolding Causal Inference for Automated Failure Attribution in Multi-Agent Systems 

**Title (ZH)**: 演绎、行动、预测：多代理系统中自动化故障归因的因果推理支架方法 

**Authors**: Alva West, Yixuan Weng, Minjun Zhu, Zhen Lin, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10401)  

**Abstract**: Failure attribution in multi-agent systems -- pinpointing the exact step where a decisive error occurs -- is a critical yet unsolved challenge. Current methods treat this as a pattern recognition task over long conversation logs, leading to critically low step-level accuracy (below 17\%), which renders them impractical for debugging complex systems. Their core weakness is a fundamental inability to perform robust counterfactual reasoning: to determine if correcting a single action would have actually averted the task failure. To bridge this counterfactual inference gap, we introduce Abduct-Act-Predict (A2P) Scaffolding, a novel agent framework that transforms failure attribution from pattern recognition into a structured causal inference task. A2P explicitly guides a large language model through a formal three-step reasoning process within a single inference pass: (1) Abduction, to infer the hidden root causes behind an agent's actions; (2) Action, to define a minimal corrective intervention; and (3) Prediction, to simulate the subsequent trajectory and verify if the intervention resolves the failure. This structured approach leverages the holistic context of the entire conversation while imposing a rigorous causal logic on the model's analysis. Our extensive experiments on the Who\&When benchmark demonstrate its efficacy. On the Algorithm-Generated dataset, A2P achieves 47.46\% step-level accuracy, a 2.85$\times$ improvement over the 16.67\% of the baseline. On the more complex Hand-Crafted dataset, it achieves 29.31\% step accuracy, a 2.43$\times$ improvement over the baseline's 12.07\%. By reframing the problem through a causal lens, A2P Scaffolding provides a robust, verifiable, and significantly more accurate solution for automated failure attribution. 

**Abstract (ZH)**: 多智能体系统中的失败归因——准确定位关键错误发生的具体步骤——是一项至关重要的但尚未解决的挑战。现有的方法将其视为长时间对话日志中的模式识别任务，导致步骤级准确率极低（低于17%），使其在调试复杂系统时不可行。它们的核心弱点是根本无法进行稳健的反事实推理：无法确定纠正单一行动是否实际上可以避免任务失败。为了弥合这种反事实推理差距，我们提出了Abduct-Act-Predict (A2P) 支架，这是一种新颖的代理框架，将失败归因从模式识别转换为结构化的因果推理任务。A2P 明确指导大语言模型在单次推理过程中通过一个正式的三步推理过程：（1）推理，以推断出智能体行为背后隐藏的根本原因；（2）行动，定义最小的纠正干预措施；（3）预测，模拟后续轨迹并验证干预是否解决了失败。这种结构化的推理方法利用了整个对话的全面背景，并在模型分析中施加了严格的因果逻辑。我们对Who\&When基准进行的广泛实验展示了其有效性。在Algorithm-Generated数据集中，A2P 达到了47.46%的步骤级准确率，是基线16.67%的2.85倍改进。在更复杂的Hand-Crafted数据集中，它达到了29.31%的步长准确率，是基线12.07%的2.43倍改进。通过从因果角度重新定义问题，A2P 支架提供了更稳健、可验证且显著更准确的自动化失败归因解决方案。 

---
# State Algebra for Propositional Logic 

**Title (ZH)**: 命题逻辑的态代数 

**Authors**: Dmitry Lesnik, Tobias Schäfer  

**Link**: [PDF](https://arxiv.org/pdf/2509.10326)  

**Abstract**: This paper presents State Algebra, a novel framework designed to represent and manipulate propositional logic using algebraic methods. The framework is structured as a hierarchy of three representations: Set, Coordinate, and Row Decomposition. These representations anchor the system in well-known semantics while facilitating the computation using a powerful algebraic engine. A key aspect of State Algebra is its flexibility in representation. We show that although the default reduction of a state vector is not canonical, a unique canonical form can be obtained by applying a fixed variable order during the reduction process. This highlights a trade-off: by foregoing guaranteed canonicity, the framework gains increased flexibility, potentially leading to more compact representations of certain classes of problems. We explore how this framework provides tools to articulate both search-based and knowledge compilation algorithms and discuss its natural extension to probabilistic logic and Weighted Model Counting. 

**Abstract (ZH)**: 本文提出了状态代数，一种用于使用代数方法表示和操作命题逻辑的新框架。该框架结构化为三级表示：集合、坐标和行分解。这些表示使系统基于已知语义，同时利用强大的代数引擎进行计算。状态代数的关键在于其表示的灵活性。我们展示了虽然状态向量的默认归约不是规范的，但在归约过程中采用固定变量顺序可获得唯一的规范形式。这突显了一个权衡：通过牺牲规范性保证，框架获得了更高的灵活性，可能使某些类问题的表示更加紧凑。我们探讨了该框架如何提供表述基于搜索和知识编译算法的工具，并讨论了其自然扩展到概率逻辑和加权模型计数的方法。 

---
# The Morality of Probability: How Implicit Moral Biases in LLMs May Shape the Future of Human-AI Symbiosis 

**Title (ZH)**: 概率的道德性：LLM中隐含的道德偏见如何塑造人类与人工智能共生的未来 

**Authors**: Eoin O'Doherty, Nicole Weinrauch, Andrew Talone, Uri Klempner, Xiaoyuan Yi, Xing Xie, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10297)  

**Abstract**: Artificial intelligence (AI) is advancing at a pace that raises urgent questions about how to align machine decision-making with human moral values. This working paper investigates how leading AI systems prioritize moral outcomes and what this reveals about the prospects for human-AI symbiosis. We address two central questions: (1) What moral values do state-of-the-art large language models (LLMs) implicitly favour when confronted with dilemmas? (2) How do differences in model architecture, cultural origin, and explainability affect these moral preferences? To explore these questions, we conduct a quantitative experiment with six LLMs, ranking and scoring outcomes across 18 dilemmas representing five moral frameworks. Our findings uncover strikingly consistent value biases. Across all models, Care and Virtue values outcomes were rated most moral, while libertarian choices were consistently penalized. Reasoning-enabled models exhibited greater sensitivity to context and provided richer explanations, whereas non-reasoning models produced more uniform but opaque judgments. This research makes three contributions: (i) Empirically, it delivers a large-scale comparison of moral reasoning across culturally distinct LLMs; (ii) Theoretically, it links probabilistic model behaviour with underlying value encodings; (iii) Practically, it highlights the need for explainability and cultural awareness as critical design principles to guide AI toward a transparent, aligned, and symbiotic future. 

**Abstract (ZH)**: 人工智能（AI）的进步速度引发了关于如何使机器决策与人类道德价值观相一致的紧迫问题。本文研究了领先AI系统如何优先考虑道德结果，并揭示了这对其与人类的共生前景的影响。我们探讨了两个核心问题：（1）当面对困境时，最先进的大型语言模型（LLMs）隐含地倾向于哪些道德价值观？（2）模型架构、文化起源和解释性差异如何影响这些道德偏好？为了探索这些问题，我们对六种LLMs进行了定量实验，对18个代表五种道德框架的困境进行排名和评分。研究发现揭示了显著一致的价值偏好。所有模型中，关怀和美德价值观的结果被评为最具道德性，而自由意志的选择则持续受到惩罚。具备推理能力的模型对情境更为敏感，并提供了更丰富的解释，而非推理模型则产生了更为一致但不透明的判断。本研究做出了三项贡献：（i）从实证角度看，它提供了跨文化大型语言模型之间道德推理的大规模比较；（ii）从理论角度看，它将概率模型行为与潜在的价值编码联系起来；（iii）从实践角度看，它强调了解释性和文化意识作为关键设计原则的必要性，以指导AI走向透明、一致和共生的未来。 

---
# Investigating Language Model Capabilities to Represent and Process Formal Knowledge: A Preliminary Study to Assist Ontology Engineering 

**Title (ZH)**: 探究语言模型在表示和处理正式知识方面的能力：一项辅助本体工程的初步研究 

**Authors**: Hanna Abi Akl  

**Link**: [PDF](https://arxiv.org/pdf/2509.10249)  

**Abstract**: Recent advances in Language Models (LMs) have failed to mask their shortcomings particularly in the domain of reasoning. This limitation impacts several tasks, most notably those involving ontology engineering. As part of a PhD research, we investigate the consequences of incorporating formal methods on the performance of Small Language Models (SLMs) on reasoning tasks. Specifically, we aim to orient our work toward using SLMs to bootstrap ontology construction and set up a series of preliminary experiments to determine the impact of expressing logical problems with different grammars on the performance of SLMs on a predefined reasoning task. Our findings show that it is possible to substitute Natural Language (NL) with a more compact logical language while maintaining a strong performance on reasoning tasks and hope to use these results to further refine the role of SLMs in ontology engineering. 

**Abstract (ZH)**: Recent Advances in Language Models: Incorporating Formal Methods to Enhance Reasoning Tasks in Small Language Models for Ontology Engineering 

---
# Compartmentalised Agentic Reasoning for Clinical NLI 

**Title (ZH)**: 模块化自主推理在临床NLI中的应用 

**Authors**: Maël Jullien, Lei Xu, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2509.10222)  

**Abstract**: A common assumption holds that scaling data and parameters yields increasingly structured, generalisable internal representations. We interrogate this assumption in clinical natural language inference (NLI) by adopting a benchmark decomposed into four reasoning families, Causal Attribution, Compositional Grounding, Epistemic Verification, and Risk State Abstraction, and introducing CARENLI, a Compartmentalised Agentic Reasoning for Clinical NLI that separates knowledge access from principled inference. CARENLI routes each premise, statement pair to a family specific solver and enforces auditable procedures via a planner, verifier, and refiner.
Across four LLMs, CARENLI improves fidelity by up to 42 points, reaching 98.0% in Causal Attribution and 81.2% in Risk State Abstraction. Verifiers flag violations with near-ceiling reliability, while refiners correct a substantial share of epistemic errors. Remaining failures cluster in routing, identifying family classification as the main bottleneck. These results show that LLMs often retain relevant facts but default to heuristics when inference is underspecified, a dissociation CARENLI makes explicit while offering a framework for safer, auditable reasoning. 

**Abstract (ZH)**: 一种常见假设认为，增大数据和参数会导致越来越有结构、可泛化的内部表征。我们通过采用分解为四种推理家族的基准测试——因果归因、成分 grounding、知识验证和风险状态抽象，并引入 CARENLI（Compartmentalised Agentic Reasoning for Clinical NLI），来探讨这一假设在临床自然语言推理中的有效性。CARENLI 将知识获取与基于原理的推理分离，并通过规划器、验证器和精炼器确保可追溯的程序。在四款语言模型中，CARENLI 将保真度提高了最多 42 个百分点，分别达到因果归因 98.0% 和风险状态抽象 81.2%。验证器以接近天花板的可靠性标记违规情况，而精炼器纠正了大量的知识验证错误。剩余的失败集中在推理路径的选择上，识别出推理类别的分类被认为是主要瓶颈。这些结果表明，语言模型通常保留了相关事实，但在推理不明确时会依赖于启发式方法，CARENLI 使这种区分变得明确，并提供了一种更安全、可追溯的推理框架。 

---
# Towards Fully Automated Molecular Simulations: Multi-Agent Framework for Simulation Setup and Force Field Extraction 

**Title (ZH)**: 面向完全自动化的分子模拟：模拟设置与力场提取的多智能体框架 

**Authors**: Marko Petković, Vlado Menkovski, Sofía Calero  

**Link**: [PDF](https://arxiv.org/pdf/2509.10210)  

**Abstract**: Automated characterization of porous materials has the potential to accelerate materials discovery, but it remains limited by the complexity of simulation setup and force field selection. We propose a multi-agent framework in which LLM-based agents can autonomously understand a characterization task, plan appropriate simulations, assemble relevant force fields, execute them and interpret their results to guide subsequent steps. As a first step toward this vision, we present a multi-agent system for literature-informed force field extraction and automated RASPA simulation setup. Initial evaluations demonstrate high correctness and reproducibility, highlighting this approach's potential to enable fully autonomous, scalable materials characterization. 

**Abstract (ZH)**: 基于大模型的多agent系统在孔材料表征中的应用：面向文献的力场提取与自动RASPA模拟设置 

---
# Online Robust Planning under Model Uncertainty: A Sample-Based Approach 

**Title (ZH)**: 基于样本的方法下的模型不确定性下的在线鲁棒规划 

**Authors**: Tamir Shazman, Idan Lev-Yehudi, Ron Benchetit, Vadim Indelman  

**Link**: [PDF](https://arxiv.org/pdf/2509.10162)  

**Abstract**: Online planning in Markov Decision Processes (MDPs) enables agents to make sequential decisions by simulating future trajectories from the current state, making it well-suited for large-scale or dynamic environments. Sample-based methods such as Sparse Sampling and Monte Carlo Tree Search (MCTS) are widely adopted for their ability to approximate optimal actions using a generative model. However, in practical settings, the generative model is often learned from limited data, introducing approximation errors that can degrade performance or lead to unsafe behaviors. To address these challenges, Robust MDPs (RMDPs) offer a principled framework for planning under model uncertainty, yet existing approaches are typically computationally intensive and not suited for real-time use. In this work, we introduce Robust Sparse Sampling (RSS), the first online planning algorithm for RMDPs with finite-sample theoretical performance guarantees. Unlike Sparse Sampling, which estimates the nominal value function, RSS computes a robust value function by leveraging the efficiency and theoretical properties of Sample Average Approximation (SAA), enabling tractable robust policy computation in online settings. RSS is applicable to infinite or continuous state spaces, and its sample and computational complexities are independent of the state space size. We provide theoretical performance guarantees and empirically show that RSS outperforms standard Sparse Sampling in environments with uncertain dynamics. 

**Abstract (ZH)**: Robust Sparse Sampling in Markov Decision Processes 

---
# Virtual Agent Economies 

**Title (ZH)**: 虚拟代理经济体 

**Authors**: Nenad Tomasev, Matija Franklin, Joel Z. Leibo, Julian Jacobs, William A. Cunningham, Iason Gabriel, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2509.10147)  

**Abstract**: The rapid adoption of autonomous AI agents is giving rise to a new economic layer where agents transact and coordinate at scales and speeds beyond direct human oversight. We propose the "sandbox economy" as a framework for analyzing this emergent system, characterizing it along two key dimensions: its origins (emergent vs. intentional) and its degree of separateness from the established human economy (permeable vs. impermeable). Our current trajectory points toward a spontaneous emergence of a vast and highly permeable AI agent economy, presenting us with opportunities for an unprecedented degree of coordination as well as significant challenges, including systemic economic risk and exacerbated inequality. Here we discuss a number of possible design choices that may lead to safely steerable AI agent markets. In particular, we consider auction mechanisms for fair resource allocation and preference resolution, the design of AI "mission economies" to coordinate around achieving collective goals, and socio-technical infrastructure needed to ensure trust, safety, and accountability. By doing this, we argue for the proactive design of steerable agent markets to ensure the coming technological shift aligns with humanity's long-term collective flourishing. 

**Abstract (ZH)**: 自主AI代理的迅速采用正在催生一个新经济层，在这个层中，代理在超出直接人类监管的规模和速度下进行交易和协调。我们提出“沙盒经济”作为分析这种新兴系统的框架，沿着两个关键维度对其进行刻画：其起源（自发产生 vs. 故意创建）及其与既有的人类经济的分离程度（渗透 vs. 封闭）。目前的轨迹指向一个自发形成的 vast 和高度渗透的AI代理经济的大规模出现，这为我们带来了前所未有的协调机会，同时也带来了包括系统性经济风险和加剧不平等在内的重大挑战。在这里，我们讨论了几种可能的设计选择，旨在实现安全可控的AI代理市场。特别地，我们考虑了用于公平资源分配和偏好多边协调的拍卖机制，设计围绕实现共同目标的AI“使命经济”，以及确保信任、安全和问责所需的社会技术基础设施。通过这样做，我们认为应主动设计可调控的代理市场，以确保即将到来的技术变革与人性的长期共同繁荣相一致。 

---
# AI Harmonics: a human-centric and harms severity-adaptive AI risk assessment framework 

**Title (ZH)**: AI和谐共振：以人为中心且适应危害严重程度的AI风险评估框架 

**Authors**: Sofia Vei, Paolo Giudici, Pavlos Sermpezis, Athena Vakali, Adelaide Emma Bernardelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.10104)  

**Abstract**: The absolute dominance of Artificial Intelligence (AI) introduces unprecedented societal harms and risks. Existing AI risk assessment models focus on internal compliance, often neglecting diverse stakeholder perspectives and real-world consequences. We propose a paradigm shift to a human-centric, harm-severity adaptive approach grounded in empirical incident data. We present AI Harmonics, which includes a novel AI harm assessment metric (AIH) that leverages ordinal severity data to capture relative impact without requiring precise numerical estimates. AI Harmonics combines a robust, generalized methodology with a data-driven, stakeholder-aware framework for exploring and prioritizing AI harms. Experiments on annotated incident data confirm that political and physical harms exhibit the highest concentration and thus warrant urgent mitigation: political harms erode public trust, while physical harms pose serious, even life-threatening risks, underscoring the real-world relevance of our approach. Finally, we demonstrate that AI Harmonics consistently identifies uneven harm distributions, enabling policymakers and organizations to target their mitigation efforts effectively. 

**Abstract (ZH)**: 人工智能的绝对主导引入了前所未有的社会危害和风险。现有的人工智能风险评估模型侧重于内部合规性，往往忽略了多元利益相关者的视角和实际后果。我们提出了一种以人文为中心、基于危害严重程度的适应性方法，该方法基于实证事故数据。我们提出了AI谐波，其中包括一个新的AI危害评估指标（AIH），该指标利用序数严重性数据捕捉相对影响，而无需精确的数值估计。AI谐波结合了稳健的一般方法和数据驱动、考虑利益相关者的框架，以探索和优先处理AI危害。标注的事故数据实验证实，政治和物理危害的集中度最高，因此需要紧急缓解：政治危害侵蚀公众信任，而物理危害则构成严重的甚至性命攸关的风险，强调了我们方法的实际相关性。最后，我们展示了AI谐波一致地识别不均匀的危害分布，使政策制定者和组织能够有效针对其缓解努力。 

---
# XAgents: A Unified Framework for Multi-Agent Cooperation via IF-THEN Rules and Multipolar Task Processing Graph 

**Title (ZH)**: XAgents：基于IF-THEN规则和多极任务处理图的统一多agent合作框架 

**Authors**: Hailong Yang, Mingxian Gu, Jianqi Wang, Guanjin Wang, Zhaohong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10054)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has significantly enhanced the capabilities of Multi-Agent Systems (MAS) in supporting humans with complex, real-world tasks. However, MAS still face challenges in effective task planning when handling highly complex tasks with uncertainty, often resulting in misleading or incorrect outputs that hinder task execution. To address this, we propose XAgents, a unified multi-agent cooperative framework built on a multipolar task processing graph and IF-THEN rules. XAgents uses the multipolar task processing graph to enable dynamic task planning and handle task uncertainty. During subtask processing, it integrates domain-specific IF-THEN rules to constrain agent behaviors, while global rules enhance inter-agent collaboration. We evaluate the performance of XAgents across three distinct datasets, demonstrating that it consistently surpasses state-of-the-art single-agent and multi-agent approaches in both knowledge-typed and logic-typed question-answering tasks. The codes for XAgents are available at: this https URL. 

**Abstract (ZH)**: 大型语言模型的迅速发展显著增强了多代理系统在支持复杂现实任务方面的能力。然而，多代理系统在处理高复杂性和不确定性任务时，仍面临有效的任务规划挑战，常常导致误导或错误的输出，阻碍任务执行。为解决这一问题，我们提出XAgents，这是一个基于多极任务处理图和IF-THEN规则的一体化多代理协同框架。XAgents利用多极任务处理图实现动态任务规划并处理任务不确定性。在子任务处理过程中，它通过整合领域特定的IF-THEN规则约束代理行为，同时全局规则增强代理间的协作。我们在三个不同的数据集上评估了XAgents的性能，结果显示它在知识型和逻辑型问答任务中均优于现有的单代理和多代理方法。XAgents的代码可在以下链接获得：this https URL。 

---
# GAMA: A General Anonymizing Multi-Agent System for Privacy Preservation Enhanced by Domain Rules and Disproof Method 

**Title (ZH)**: GAMA：一个通过领域规则和反驳方法增强隐私保护的通用匿名多agent系统 

**Authors**: Hailong Yang, Renhuo Zhao, Guanjin Wang, Zhaohong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10018)  

**Abstract**: With the rapid advancement of Large Language Model (LLM), LLM-based agents exhibit exceptional abilities in understanding and generating natural language, facilitating human-like collaboration and information transmission in LLM-based Multi-Agent System (MAS). High-performance LLMs are often hosted on remote servers in public spaces. When tasks involve privacy data, MAS cannot securely utilize these LLMs without implementing privacy-preserving mechanisms. To address this challenge, we propose a General Anonymizing Multi-Agent system (GAMA), which divides the agents' workspace into private and public spaces and protects privacy through the anonymizing mechanism. In the private space, agents handle sensitive data, while in the public space, only anonymized data is utilized. GAMA incorporates two key modules to mitigate semantic loss caused by anonymization: Domain-Rule-based Knowledge Enhancement (DRKE) and Disproof-based Logic Enhancement (DLE). We evaluate GAMA on two public question-answering datasets: Trivia Creative Writing and Logic Grid Puzzle. The results demonstrate that GAMA has superior performance compared to the state-of-the-art models. To further assess its privacy-preserving capabilities, we designed two new datasets: Knowledge Privacy Preservation and Logic Privacy Preservation. The final results highlight GAMA's exceptional effectiveness in both task processing and privacy preservation. 

**Abstract (ZH)**: 基于大型语言模型的通用匿名多智能体系统 

---
# Evaluation of Black-Box XAI Approaches for Predictors of Values of Boolean Formulae 

**Title (ZH)**: 黑盒XAI方法在布尔公式值预测器评估中的应用 

**Authors**: Stav Armoni-Friedmann, Hana Chockler, David A. Kelly  

**Link**: [PDF](https://arxiv.org/pdf/2509.09982)  

**Abstract**: Evaluating explainable AI (XAI) approaches is a challenging task in general, due to the subjectivity of explanations. In this paper, we focus on tabular data and the specific use case of AI models predicting the values of Boolean functions. We extend the previous work in this domain by proposing a formal and precise measure of importance of variables based on actual causality, and we evaluate state-of-the-art XAI tools against this measure. We also present a novel XAI tool B-ReX, based on the existing tool ReX, and demonstrate that it is superior to other black-box XAI tools on a large-scale benchmark. Specifically, B-ReX achieves a Jensen-Shannon divergence of 0.072 $\pm$ 0.012 on random 10-valued Boolean formulae 

**Abstract (ZH)**: 评估可解释AI（XAI）方法是一项具有挑战性的工作，由于解释的主观性所致。本文专注于表格数据以及AI模型预测布尔函数值的特定应用场景。我们在此领域扩展了先前的工作，提出了一种基于实际因果性的形式化和精确的变量重要性度量方法，并将最先进的XAI工具与该度量方法进行了评估。此外，我们介绍了一种基于现有工具ReX的新XAI工具B-ReX，并证明其在大规模基准测试中优于其他黑盒XAI工具，具体而言，B-ReX 在随机10值布尔公式上的Jensen-Shannon散度为0.072 $\pm$ 0.012。 

---
# A Markovian Framing of WaveFunctionCollapse for Procedurally Generating Aesthetically Complex Environments 

**Title (ZH)**: 基于马尔可夫框架的波函数塌缩模型用于 procedurally 生成美学复杂环境 

**Authors**: Franklin Yiu, Mohan Lu, Nina Li, Kevin Joseph, Tianxu Zhang, Julian Togelius, Timothy Merino, Sam Earle  

**Link**: [PDF](https://arxiv.org/pdf/2509.09919)  

**Abstract**: Procedural content generation often requires satisfying both designer-specified objectives and adjacency constraints implicitly imposed by the underlying tile set. To address the challenges of jointly optimizing both constraints and objectives, we reformulate WaveFunctionCollapse (WFC) as a Markov Decision Process (MDP), enabling external optimization algorithms to focus exclusively on objective maximization while leveraging WFC's propagation mechanism to enforce constraint satisfaction. We empirically compare optimizing this MDP to traditional evolutionary approaches that jointly optimize global metrics and local tile placement. Across multiple domains with various difficulties, we find that joint optimization not only struggles as task complexity increases, but consistently underperforms relative to optimization over the WFC-MDP, underscoring the advantages of decoupling local constraint satisfaction from global objective optimization. 

**Abstract (ZH)**: 基于过程的内容生成常常需要同时满足设计师指定的目标和底层砖块集隐式施加的相邻约束。为了应对同时优化这两者带来的挑战，我们将WaveFunctionCollapse（WFC）重新表述为马尔可夫决策过程（MDP），这使得外部优化算法能够专注于目标最大化，同时利用WFC的传播机制确保约束满足。我们通过实验比较了优化这个MDP与传统联合优化全局度量和局部砖块放置的进化方法的效果。在不同难度级别的多个领域中，我们发现，联合优化不仅随着任务复杂性的增加而表现困难，而且始终在优化WFC-MDP方面表现较差，突显了将局部约束满足与全局目标优化解耦的优势。 

---
# The (R)evolution of Scientific Workflows in the Agentic AI Era: Towards Autonomous Science 

**Title (ZH)**: 代理人工智能时代的科学工作流(演变):走向自主科学研究 

**Authors**: Woong Shin, Renan Souza, Daniel Rosendo, Frédéric Suter, Feiyi Wang, Prasanna Balaprakash, Rafael Ferreira da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2509.09915)  

**Abstract**: Modern scientific discovery increasingly requires coordinating distributed facilities and heterogeneous resources, forcing researchers to act as manual workflow coordinators rather than scientists. Advances in AI leading to AI agents show exciting new opportunities that can accelerate scientific discovery by providing intelligence as a component in the ecosystem. However, it is unclear how this new capability would materialize and integrate in the real world. To address this, we propose a conceptual framework where workflows evolve along two dimensions which are intelligence (from static to intelligent) and composition (from single to swarm) to chart an evolutionary path from current workflow management systems to fully autonomous, distributed scientific laboratories. With these trajectories in mind, we present an architectural blueprint that can help the community take the next steps towards harnessing the opportunities in autonomous science with the potential for 100x discovery acceleration and transformational scientific workflows. 

**Abstract (ZH)**: 现代科学发现越来越需要协调分布的设施和异构资源，迫使研究人员成为手工工作流协调员而非科学家。人工智能技术的进步使得人工智能代理展现出激动人心的新机遇，通过在生态系统中提供智能来加速科学发现。然而，尚不清楚这种新能力如何在现实中实现和集成。为了解决这一问题，我们提出了一种概念框架，其中工作流沿着智能（从静态到智能）和组成（从单一到群体）两个维度进化，从而勾画出从当前工作流管理系统到完全自主的分布式科学实验室的进化路径。基于这些轨迹，我们提出了一个架构蓝图，旨在协助社区采取下一步行动，利用自主科学的机遇，潜力实现100倍的发现加速和变革性的科学工作流。 

---
# LLMs as Agentic Cooperative Players in Multiplayer UNO 

**Title (ZH)**: LLMs作为具有自主合作能力的玩家在多人UNO游戏中的应用 

**Authors**: Yago Romano Matinez, Jesse Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2509.09867)  

**Abstract**: LLMs promise to assist humans -- not just by answering questions, but by offering useful guidance across a wide range of tasks. But how far does that assistance go? Can a large language model based agent actually help someone accomplish their goal as an active participant? We test this question by engaging an LLM in UNO, a turn-based card game, asking it not to win but instead help another player to do so. We built a tool that allows decoder-only LLMs to participate as agents within the RLCard game environment. These models receive full game-state information and respond using simple text prompts under two distinct prompting strategies. We evaluate models ranging from small (1B parameters) to large (70B parameters) and explore how model scale impacts performance. We find that while all models were able to successfully outperform a random baseline when playing UNO, few were able to significantly aid another player. 

**Abstract (ZH)**: 大规模语言模型承诺辅助人类——不仅通过回答问题，还能在一系列任务中提供有用指导。但这种辅助能走多远？一个基于大语言模型的代理是否能作为积极参与者帮助他人实现目标？我们通过让大语言模型参与Uno牌游来测试这一问题，要求模型不求取胜，而是帮助另一玩家取胜。我们构建了一个工具，使仅解码器的大语言模型能够在RLCard游戏环境中作为代理参与。这些模型接收完整的游戏状态信息，并在两种不同的提示策略下使用简单的文本提示进行回应。我们评估了从小型（10亿参数）到大型（700亿参数）的各种模型，并探索模型规模对性能的影响。我们发现，尽管所有模型在玩Uno时都能成功超越随机 baseline，但很少有模型能显著帮助另一玩家。 

---
# Towards an AI-based knowledge assistant for goat farmers based on Retrieval-Augmented Generation 

**Title (ZH)**: 基于检索增强生成的面向羊农的AI知识助手 

**Authors**: Nana Han, Dong Liu, Tomas Norton  

**Link**: [PDF](https://arxiv.org/pdf/2509.09848)  

**Abstract**: Large language models (LLMs) are increasingly being recognised as valuable knowledge communication tools in many industries. However, their application in livestock farming remains limited, being constrained by several factors not least the availability, diversity and complexity of knowledge sources. This study introduces an intelligent knowledge assistant system designed to support health management in farmed goats. Leveraging the Retrieval-Augmented Generation (RAG), two structured knowledge processing methods, table textualization and decision-tree textualization, were proposed to enhance large language models' (LLMs) understanding of heterogeneous data formats. Based on these methods, a domain-specific goat farming knowledge base was established to improve LLM's capacity for cross-scenario generalization. The knowledge base spans five key domains: Disease Prevention and Treatment, Nutrition Management, Rearing Management, Goat Milk Management, and Basic Farming Knowledge. Additionally, an online search module is integrated to enable real-time retrieval of up-to-date information. To evaluate system performance, six ablation experiments were conducted to examine the contribution of each component. The results demonstrated that heterogeneous knowledge fusion method achieved the best results, with mean accuracies of 87.90% on the validation set and 84.22% on the test set. Across the text-based, table-based, decision-tree based Q&A tasks, accuracy consistently exceeded 85%, validating the effectiveness of structured knowledge fusion within a modular design. Error analysis identified omission as the predominant error category, highlighting opportunities to further improve retrieval coverage and context integration. In conclusion, the results highlight the robustness and reliability of the proposed system for practical applications in goat farming. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个行业被日益认可为有价值的知识交流工具，但在畜牧业中的应用仍然受限，主要受知识来源的可用性、多样性和复杂性等因素的制约。本研究介绍了一种智能知识助手系统，旨在支持养羊健康管理工作。利用检索增强生成（RAG）技术，提出了两种结构化知识处理方法：表格文本化和决策树文本化，以增强大型语言模型（LLMs）对异构数据格式的理解能力。基于这些方法，建立了一个专门针对养羊业的知识库，以提高LLMs在跨场景泛化能力。该知识库涵盖了五个关键领域：疾病预防与治疗、营养管理、饲养管理、山羊奶管理以及基础农学知识。此外，还集成了在线搜索模块，以实现实时检索最新信息。为了评估系统性能，进行了六项消融实验，以检查每个组件的贡献。结果表明，异构知识融合方法表现最佳，验证集的平均准确率为87.90%，测试集为84.22%。在基于文本、基于表格和基于决策树的问答任务中，准确率均超过85%，验证了模块化设计中结构化知识融合的有效性。误差分析表明，遗漏是最主要的错误类别，指出了进一步提高检索覆盖率和上下文整合的机会。总之，结果突显了所提出系统在养羊业实际应用中的稳健性和可靠性。 

---
# Towards a Common Framework for Autoformalization 

**Title (ZH)**: 面向自动形式化统一框架的研究 

**Authors**: Agnieszka Mensfelt, David Tena Cucala, Santiago Franco, Angeliki Koutsoukou-Argyraki, Vince Trencsenyi, Kostas Stathis  

**Link**: [PDF](https://arxiv.org/pdf/2509.09810)  

**Abstract**: Autoformalization has emerged as a term referring to the automation of formalization - specifically, the formalization of mathematics using interactive theorem provers (proof assistants). Its rapid development has been driven by progress in deep learning, especially large language models (LLMs). More recently, the term has expanded beyond mathematics to describe the broader task of translating informal input into formal logical representations. At the same time, a growing body of research explores using LLMs to translate informal language into formal representations for reasoning, planning, and knowledge representation - often without explicitly referring to this process as autoformalization. As a result, despite addressing similar tasks, the largely independent development of these research areas has limited opportunities for shared methodologies, benchmarks, and theoretical frameworks that could accelerate progress. The goal of this paper is to review - explicit or implicit - instances of what can be considered autoformalization and to propose a unified framework, encouraging cross-pollination between different fields to advance the development of next generation AI systems. 

**Abstract (ZH)**: Autoformalization: 从形式化自动化到自然语言到形式化表示的自动转换及其统一框架 

---
# A Modular and Multimodal Generative AI Framework for Urban Building Energy Data: Generating Synthetic Homes 

**Title (ZH)**: 一个模块化多模态生成AI框架：生成合成住宅用于城市建筑能源数据 

**Authors**: Jackson Eshbaugh, Chetan Tiwari, Jorge Silveyra  

**Link**: [PDF](https://arxiv.org/pdf/2509.09794)  

**Abstract**: Computational models have emerged as powerful tools for energy modeling research, touting scalability and quantitative results. However, these models require a plethora of data, some of which is inaccessible, expensive, or raises privacy concerns. We introduce a modular multimodal framework to produce this data from publicly accessible residential information and images using generative artificial intelligence (AI). Additionally, we provide a pipeline demonstrating this framework, and we evaluate its generative AI components. Our experiments show that our framework's use of AI avoids common issues with generative models. Our framework produces realistic, labeled data. By reducing dependence on costly or restricted data sources, we pave a path towards more accessible and reproducible research. 

**Abstract (ZH)**: 计算模型已成为能源建模研究中的强大工具，具备扩展性和量化结果的优点，但这些模型需要大量的数据，其中一些数据是不可访问、昂贵或涉及隐私问题。我们提出了一种模块化多模态框架，利用生成人工智能（AI）从公开的住宅信息和图像中生成所需数据。此外，我们提供了一条流水线来展示该框架，并评估其生成AI组件。我们的实验表明，我们的框架使用AI避免了生成模型中的常见问题。该框架生成的是真实且带有标签的数据。通过减少对昂贵或受限数据源的依赖，我们为更广泛的可访问性和可重复性研究铺平了道路。 

---
# How well can LLMs provide planning feedback in grounded environments? 

**Title (ZH)**: 大型语言模型在 grounded 环境中提供规划反馈的能力如何？ 

**Authors**: Yuxuan Li, Victor Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2509.09790)  

**Abstract**: Learning to plan in grounded environments typically requires carefully designed reward functions or high-quality annotated demonstrations. Recent works show that pretrained foundation models, such as large language models (LLMs) and vision language models (VLMs), capture background knowledge helpful for planning, which reduces the amount of reward design and demonstrations needed for policy learning. We evaluate how well LLMs and VLMs provide feedback across symbolic, language, and continuous control environments. We consider prominent types of feedback for planning including binary feedback, preference feedback, action advising, goal advising, and delta action feedback. We also consider inference methods that impact feedback performance, including in-context learning, chain-of-thought, and access to environment dynamics. We find that foundation models can provide diverse high-quality feedback across domains. Moreover, larger and reasoning models consistently provide more accurate feedback, exhibit less bias, and benefit more from enhanced inference methods. Finally, feedback quality degrades for environments with complex dynamics or continuous state spaces and action spaces. 

**Abstract (ZH)**: 预训练基础模型在接地环境中提供规划反馈的能力研究 

---
# Executable Ontologies: Synthesizing Event Semantics with Dataflow Architecture 

**Title (ZH)**: 可执行本体：基于数据流架构合成事件语义 

**Authors**: Aleksandr Boldachev  

**Link**: [PDF](https://arxiv.org/pdf/2509.09775)  

**Abstract**: This paper presents boldsea, Boldachev's semantic-event approach -- an architecture for modeling complex dynamic systems using executable ontologies -- semantic models that act as dynamic structures, directly controlling process execution. We demonstrate that integrating event semantics with a dataflow architecture addresses the limitations of traditional Business Process Management (BPM) systems and object-oriented semantic technologies. The paper presents the formal BSL (boldsea Semantic Language), including its BNF grammar, and outlines the boldsea-engine's architecture, which directly interprets semantic models as executable algorithms without compilation. It enables the modification of event models at runtime, ensures temporal transparency, and seamlessly merges data and business logic within a unified semantic framework. 

**Abstract (ZH)**: 本文介绍boldsea，一种基于可执行本体的Boldachev语义事件方法——通过使用语义模型作为动态结构直接控制过程执行来建模复杂动态系统的方法。我们演示了将事件语义与数据流架构结合使用以解决传统业务过程管理(BPM)系统和面向对象语义技术的局限性。本文介绍了形式化的BSL（boldsea语义语言），包括其BNF语法，并概述了boldsea-engine的架构，该架构可以直接解释语义模型为可执行算法而无需编译。它允许在运行时修改事件模型，确保时间透明性，并在统一的语义框架内无缝融合数据和业务逻辑。 

---
# Human-AI Collaboration Increases Efficiency in Regulatory Writing 

**Title (ZH)**: 人类-人工智能合作提高监管文件写作效率 

**Authors**: Umut Eser, Yael Gozin, L. Jay Stallons, Ari Caroline, Martin Preusse, Brandon Rice, Scott Wright, Andrew Robertson  

**Link**: [PDF](https://arxiv.org/pdf/2509.09738)  

**Abstract**: Background: Investigational New Drug (IND) application preparation is time-intensive and expertise-dependent, slowing early clinical development. Objective: To evaluate whether a large language model (LLM) platform (AutoIND) can reduce first-draft composition time while maintaining document quality in regulatory submissions. Methods: Drafting times for IND nonclinical written summaries (eCTD modules 2.6.2, 2.6.4, 2.6.6) generated by AutoIND were directly recorded. For comparison, manual drafting times for IND summaries previously cleared by the U.S. FDA were estimated from the experience of regulatory writers ($\geq$6 years) and used as industry-standard benchmarks. Quality was assessed by a blinded regulatory writing assessor using seven pre-specified categories: correctness, completeness, conciseness, consistency, clarity, redundancy, and emphasis. Each sub-criterion was scored 0-3 and normalized to a percentage. A critical regulatory error was defined as any misrepresentation or omission likely to alter regulatory interpretation (e.g., incorrect NOAEL, omission of mandatory GLP dose-formulation analysis). Results: AutoIND reduced initial drafting time by $\sim$97% (from $\sim$100 h to 3.7 h for 18,870 pages/61 reports in IND-1; and to 2.6 h for 11,425 pages/58 reports in IND-2). Quality scores were 69.6\% and 77.9\% for IND-1 and IND-2. No critical regulatory errors were detected, but deficiencies in emphasis, conciseness, and clarity were noted. Conclusions: AutoIND can dramatically accelerate IND drafting, but expert regulatory writers remain essential to mature outputs to submission-ready quality. Systematic deficiencies identified provide a roadmap for targeted model improvements. 

**Abstract (ZH)**: 背景与目的： investigational new drug (IND) 申请准备耗时且依赖专业知识，延缓了早期临床开发。目标：评估大型语言模型 (LLM) 平台 (AutoIND) 是否能在保持文件质量的前提下，减少初步稿件的撰写时间。方法：AutoIND 生成的 IND 非临床书面摘要（eCTD 模块 2.6.2, 2.6.4, 2.6.6）的撰写时间直接记录。为了比较，通过具有 6 年以上经验的注册编写人员的经验估计手动撰写 IND 摘要的时间，并用作行业标准基准。质量通过盲评注册编写评审员使用七类预设标准：正确性、完整性、简洁性、一致性、清晰度、冗余性和重点来评估。每个子标准评分范围为 0-3，并归一化为百分比。关键监管错误被定义为任何可能改变监管解读的误述或遗漏（例如，不正确的 NOAEL，遗漏的必需的 GLP 剂型分析）。结果：AutoIND 将初始撰写时间减少了约 97%（从约 100 小时减少到 18,870 页/61 份报告为 3.7 小时；IND-2 为 11,425 页/58 份报告为 2.6 小时）。质量评分为 IND-1 为 69.6%，IND-2 为 77.9%。未发现关键监管错误，但存在重点、简洁性和清晰度方面的不足。结论：AutoIND 可显著加速 IND 撰写，但专家级监管编写人员对于产出提交级质量仍然是必要的。发现的系统性不足为模型目标改进指明了方向。 

---
# Standards in the Preparation of Biomedical Research Metadata: A Bridge2AI Perspective 

**Title (ZH)**: 生物医学研究元数据准备标准：Bridge2AI 的视角 

**Authors**: Harry Caufield, Satrajit Ghosh, Sek Wong Kong, Jillian Parker, Nathan Sheffield, Bhavesh Patel, Andrew Williams, Timothy Clark, Monica C. Munoz-Torres  

**Link**: [PDF](https://arxiv.org/pdf/2509.10432)  

**Abstract**: AI-readiness describes the degree to which data may be optimally and ethically used for subsequent AI and Machine Learning (AI/ML) methods, where those methods may involve some combination of model training, data classification, and ethical, explainable prediction. The Bridge2AI consortium has defined the particular criteria a biomedical dataset may possess to render it AI-ready: in brief, a dataset's readiness is related to its FAIRness, provenance, degree of characterization, explainability, sustainability, and computability, in addition to its accompaniment with documentation about ethical data practices.
To ensure AI-readiness and to clarify data structure and relationships within Bridge2AI's Grand Challenges (GCs), particular types of metadata are necessary. The GCs within the Bridge2AI initiative include four data-generating projects focusing on generating AI/ML-ready datasets to tackle complex biomedical and behavioral research problems. These projects develop standardized, multimodal data, tools, and training resources to support AI integration, while addressing ethical data practices. Examples include using voice as a biomarker, building interpretable genomic tools, modeling disease trajectories with diverse multimodal data, and mapping cellular and molecular health indicators across the human body.
This report assesses the state of metadata creation and standardization in the Bridge2AI GCs, provides guidelines where required, and identifies gaps and areas for improvement across the program. New projects, including those outside the Bridge2AI consortium, would benefit from what we have learned about creating metadata as part of efforts to promote AI readiness. 

**Abstract (ZH)**: AI就绪性描述了数据用于后续AI和机器学习方法时的优化和伦理使用程度，其中这些方法可能包括模型训练、数据分类和伦理化解释预测。Bridge2AI联盟定义了生物医学数据集可能具备的特定标准，以使其达到AI就绪状态：简而言之，数据集的就绪程度与其FAIR性、来源、特征化程度、可解释性、可持续性和计算性相关，以及与其伴随的伦理数据实践文档相关。

为了确保AI就绪性和明晰数据结构及关系，Bridge2AI的挑战项目（GCs）中需要特定类型的元数据。Bridge2AI计划中的GCs包括四个数据生成项目，旨在生成AI/ML就绪的数据集，以解决复杂的生物医学和行为研究问题。这些项目开发标准化的多模态数据、工具和培训资源，支持AI整合，同时解决伦理数据实践问题。例如，使用声音作为生物标志物、构建可解释的基因组工具、使用多元多模态数据建模疾病轨迹以及跨人体绘制细胞和分子健康指标图谱。

本报告评估了Bridge2AI GCs中的元数据创建和标准化状况，提供必要的指南，并识别项目中的空白点和改进领域。新项目，包括Bridge2AI联盟之外的项目，将从我们关于创建元数据的经验中受益，以促进AI就绪性。 

---
# Is In-Context Learning Learning? 

**Title (ZH)**: 基于上下文学习是一种学习吗？ 

**Authors**: Adrian de Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2509.10414)  

**Abstract**: In-context learning (ICL) allows some autoregressive models to solve tasks via next-token prediction and without needing further training. This has led to claims about these model's ability to solve (learn) unseen tasks with only a few shots (exemplars) in the prompt. However, deduction does not always imply learning, as ICL does not explicitly encode a given observation. Instead, the models rely on their prior knowledge and the exemplars given, if any. We argue that, mathematically, ICL does constitute learning, but its full characterisation requires empirical work. We then carry out a large-scale analysis of ICL ablating out or accounting for memorisation, pretraining, distributional shifts, and prompting style and phrasing. We find that ICL is an effective learning paradigm, but limited in its ability to learn and generalise to unseen tasks. We note that, in the limit where exemplars become more numerous, accuracy is insensitive to exemplar distribution, model, prompt style, and the input's linguistic features. Instead, it deduces patterns from regularities in the prompt, which leads to distributional sensitivity, especially in prompting styles such as chain-of-thought. Given the varied accuracies on formally similar tasks, we conclude that autoregression's ad-hoc encoding is not a robust mechanism, and suggests limited all-purpose generalisability. 

**Abstract (ZH)**: 上下文约束学习（ICL）使一些自回归模型能够通过下一个token预测来解决任务，而无需进一步训练。这导致了这些模型能够仅通过几个示例（shot）在提示中解决（学习）未见过的任务的说法。然而，推理并不总是意味着学习，因为ICL不明确编码给定的观察。相反，模型依赖于它们的先验知识和给定的示例（如果有的话）。我们认为，从数学上讲，ICL构成了一种学习，但其完全表征需要通过实证工作来完成。然后，我们进行了一项大规模分析，消融或考虑记忆、预训练、分布转移以及提示样式和措辞。我们发现ICL是一种有效的学习范式，但在学习和泛化到未见过的任务方面有限。我们注意到，在示例（exemplar）变得更加频繁时，准确性对示例分布、模型、提示样式和输入的语言特征具有鲁棒性。相反，它从提示中的规律性推导出模式，这导致了分布敏感性，尤其是在链式思考提示样式中尤为明显。鉴于在形式上相似任务上表现各异，我们得出结论，自回归的任意编码机制不够稳健，并暗示其泛化能力有限。 

---
# Multimodal SAM-adapter for Semantic Segmentation 

**Title (ZH)**: 多模态SAM适配器用于语义分割 

**Authors**: Iacopo Curti, Pierluigi Zama Ramirez, Alioscia Petrelli, Luigi Di Stefano  

**Link**: [PDF](https://arxiv.org/pdf/2509.10408)  

**Abstract**: Semantic segmentation, a key task in computer vision with broad applications in autonomous driving, medical imaging, and robotics, has advanced substantially with deep learning. Nevertheless, current approaches remain vulnerable to challenging conditions such as poor lighting, occlusions, and adverse weather. To address these limitations, multimodal methods that integrate auxiliary sensor data (e.g., LiDAR, infrared) have recently emerged, providing complementary information that enhances robustness. In this work, we present MM SAM-adapter, a novel framework that extends the capabilities of the Segment Anything Model (SAM) for multimodal semantic segmentation. The proposed method employs an adapter network that injects fused multimodal features into SAM's rich RGB features. This design enables the model to retain the strong generalization ability of RGB features while selectively incorporating auxiliary modalities only when they contribute additional cues. As a result, MM SAM-adapter achieves a balanced and efficient use of multimodal information. We evaluate our approach on three challenging benchmarks, DeLiVER, FMB, and MUSES, where MM SAM-adapter delivers state-of-the-art performance. To further analyze modality contributions, we partition DeLiVER and FMB into RGB-easy and RGB-hard subsets. Results consistently demonstrate that our framework outperforms competing methods in both favorable and adverse conditions, highlighting the effectiveness of multimodal adaptation for robust scene understanding. The code is available at the following link: this https URL. 

**Abstract (ZH)**: 多模态Semantic分割适应器：一种扩展Segment Anything Model (SAM) 的多模态语义分割框架 

---
# Diversified recommendations of cultural activities with personalized determinantal point processes 

**Title (ZH)**: 基于个性化行列式点过程的文体活动多元化推荐 

**Authors**: Carole Ibrahim, Hiba Bederina, Daniel Cuesta, Laurent Montier, Cyrille Delabre, Jill-Jênn Vie  

**Link**: [PDF](https://arxiv.org/pdf/2509.10392)  

**Abstract**: While optimizing recommendation systems for user engagement is a well-established practice, effectively diversifying recommendations without negatively impacting core business metrics remains a significant industry challenge. In line with our initiative to broaden our audience's cultural practices, this study investigates using personalized Determinantal Point Processes (DPPs) to sample diverse and relevant recommendations. We rely on a well-known quality-diversity decomposition of the similarity kernel to give more weight to user preferences. In this paper, we present our implementations of the personalized DPP sampling, evaluate the trade-offs between relevance and diversity through both offline and online metrics, and give insights for practitioners on their use in a production environment. For the sake of reproducibility, we release the full code for our platform and experiments on GitHub. 

**Abstract (ZH)**: 在优化推荐系统以提升用户参与度的同时有效多样化推荐而不负面影响核心业务指标仍是一项重要的行业挑战。为了拓宽受众的文化实践，本研究探讨使用个性化行列式点过程（DPPs）来采集多样化和相关推荐。我们利用相似核的质量-多样性分解赋予用户偏好更多的权重。在本文中，我们展示了个性化DPP采样的实现，通过离线和在线指标评估相关性和多样性的权衡，并为在生产环境中使用这些方法提供了实践见解。为了便于可重复性，我们在GitHub上发布了我们的平台和实验的完整代码。 

---
# Improving Audio Event Recognition with Consistency Regularization 

**Title (ZH)**: 使用一致性正则化改进音频事件识别 

**Authors**: Shanmuka Sadhu, Weiran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10391)  

**Abstract**: Consistency regularization (CR), which enforces agreement between model predictions on augmented views, has found recent benefits in automatic speech recognition [1]. In this paper, we propose the use of consistency regularization for audio event recognition, and demonstrate its effectiveness on AudioSet. With extensive ablation studies for both small ($\sim$20k) and large ($\sim$1.8M) supervised training sets, we show that CR brings consistent improvement over supervised baselines which already heavily utilize data augmentation, and CR using stronger augmentation and multiple augmentations leads to additional gain for the small training set. Furthermore, we extend the use of CR into the semi-supervised setup with 20K labeled samples and 1.8M unlabeled samples, and obtain performance improvement over our best model trained on the small set. 

**Abstract (ZH)**: 一致性正则化（CR）通过在增强视角之间强制模型预测的一致性，在自动语音识别中发现了近期的益处。在本文中，我们提出将一致性正则化用于音频事件识别，并在AudioSet上展示了其有效性。通过针对小规模（约20k）和大规模（约1.8M）监督训练集进行广泛的消融研究，我们展示了即使在已经大量使用数据增强的监督基准上，CR也带来了持续的改进，并且使用更强的增强和多种增强方法会对小训练集带来额外增益。此外，我们将CR扩展应用于20K标注样本和1.8M未标注样本的半监督设置，并相对于仅在小训练集上训练的最佳模型获得了性能提升。 

---
# Data distribution impacts the performance and generalisability of contrastive learning-based foundation models of electrocardiograms 

**Title (ZH)**: 数据分布影响基于对比学习的心电图基础模型的性能和泛化能力 

**Authors**: Gul Rukh Khattak, Konstantinos Patlatzoglou, Joseph Barker, Libor Pastika, Boroumand Zeidaabadi, Ahmed El-Medany, Hesham Aggour, Yixiu Liang, Antonio H. Ribeiro, Jeffrey Annis, Antonio Luiz Pinho Ribeiro, Junbo Ge, Daniel B. Kramer, Jonathan W. Waks, Evan Brittain, Nicholas Peters, Fu Siong Ng, Arunashis Sau  

**Link**: [PDF](https://arxiv.org/pdf/2509.10369)  

**Abstract**: Contrastive learning is a widely adopted self-supervised pretraining strategy, yet its dependence on cohort composition remains underexplored. We present Contrasting by Patient Augmented Electrocardiograms (CAPE) foundation model and pretrain on four cohorts (n = 5,203,352), from diverse populations across three continents (North America, South America, Asia). We systematically assess how cohort demographics, health status, and population diversity influence the downstream performance for prediction tasks also including two additional cohorts from another continent (Europe). We find that downstream performance depends on the distributional properties of the pretraining cohort, including demographics and health status. Moreover, while pretraining with a multi-centre, demographically diverse cohort improves in-distribution accuracy, it reduces out-of-distribution (OOD) generalisation of our contrastive approach by encoding cohort-specific artifacts. To address this, we propose the In-Distribution Batch (IDB) strategy, which preserves intra-cohort consistency during pretraining and enhances OOD robustness. This work provides important insights for developing clinically fair and generalisable foundation models. 

**Abstract (ZH)**: 对比学习是一种广泛采用的自监督预训练策略，但其对群体构成的依赖性尚未充分探索。我们提出了基于患者增强心电图的对比学习基础模型（CAPE），并在来自三大洲（北美洲、南美洲、亚洲）多样人口的四个群体（n=5,203,352）上进行预训练。我们系统地评估了群体人口统计学特征、健康状况和人群多样性如何影响下游预测任务的性能，并包括了另一个洲（欧洲）的两个额外群体。我们发现，下游性能取决于预训练群体的分布特性，包括人口统计学特征和健康状况。此外，在多元中心、人口统计学多样性的群体上进行预训练可以提高分布内准确性，但会通过编码群体特异性伪像降低我们的对比方法的跨分布外泛化能力。为此，我们提出了内在分布批次（IDB）策略，在预训练期间保持群体内一致性并增强跨分布外鲁棒性。该项工作为开发临床公平和普适的基础模型提供了重要见解。 

---
# Towards Understanding Visual Grounding in Visual Language Models 

**Title (ZH)**: 理解视觉语言模型中的视觉接地 

**Authors**: Georgios Pantazopoulos, Eda B. Özyiğit  

**Link**: [PDF](https://arxiv.org/pdf/2509.10345)  

**Abstract**: Visual grounding refers to the ability of a model to identify a region within some visual input that matches a textual description. Consequently, a model equipped with visual grounding capabilities can target a wide range of applications in various domains, including referring expression comprehension, answering questions pertinent to fine-grained details in images or videos, caption visual context by explicitly referring to entities, as well as low and high-level control in simulated and real environments. In this survey paper, we review representative works across the key areas of research on modern general-purpose vision language models (VLMs). We first outline the importance of grounding in VLMs, then delineate the core components of the contemporary paradigm for developing grounded models, and examine their practical applications, including benchmarks and evaluation metrics for grounded multimodal generation. We also discuss the multifaceted interrelations among visual grounding, multimodal chain-of-thought, and reasoning in VLMs. Finally, we analyse the challenges inherent to visual grounding and suggest promising directions for future research. 

**Abstract (ZH)**: 视觉 grounding 指的是模型识别视觉输入中与文本描述匹配的区域的能力。因此，具备视觉 grounding 能力的模型可以应用于各个领域的多种应用场景，包括参照表达理解、回答与图像或视频中精细细节相关的问题、通过明确指代实体caption视觉上下文，以及在模拟和实际环境中实现低级和高级控制。在本文综述中，我们回顾了现代通用视觉语言模型（VLMs）关键研究领域的代表性工作。我们首先概述了在 VLMs 中进行 grounding 的重要性，然后阐述了开发 grounded 模型的现代范式的核心组件及其实际应用，包括grounded 多模态生成的基准和评估指标。我们还讨论了视觉 grounding、多模态链式思考和推理在 VLMs 中的多方面关系。最后，我们分析了视觉 grounding 内在的挑战，并提出了未来研究的潜在方向。 

---
# GLAM: Geometry-Guided Local Alignment for Multi-View VLP in Mammography 

**Title (ZH)**: GLAM: 几何引导的局部对齐方法用于乳腺X线摄影的多视图三维重建 

**Authors**: Yuexi Du, Lihui Chen, Nicha C. Dvornek  

**Link**: [PDF](https://arxiv.org/pdf/2509.10344)  

**Abstract**: Mammography screening is an essential tool for early detection of breast cancer. The speed and accuracy of mammography interpretation have the potential to be improved with deep learning methods. However, the development of a foundation visual language model (VLM) is hindered by limited data and domain differences between natural and medical images. Existing mammography VLMs, adapted from natural images, often ignore domain-specific characteristics, such as multi-view relationships in mammography. Unlike radiologists who analyze both views together to process ipsilateral correspondence, current methods treat them as independent images or do not properly model the multi-view correspondence learning, losing critical geometric context and resulting in suboptimal prediction. We propose GLAM: Global and Local Alignment for Multi-view mammography for VLM pretraining using geometry guidance. By leveraging the prior knowledge about the multi-view imaging process of mammograms, our model learns local cross-view alignments and fine-grained local features through joint global and local, visual-visual, and visual-language contrastive learning. Pretrained on EMBED [14], one of the largest open mammography datasets, our model outperforms baselines across multiple datasets under different settings. 

**Abstract (ZH)**: 多视角乳腺X光筛查的全局和局部对齐预训练视觉语言模型 

---
# I-Segmenter: Integer-Only Vision Transformer for Efficient Semantic Segmentation 

**Title (ZH)**: I-Segmenter：仅整数视觉变换器用于高效语义分割 

**Authors**: Jordan Sassoon, Michal Szczepanski, Martyna Poreba  

**Link**: [PDF](https://arxiv.org/pdf/2509.10334)  

**Abstract**: Vision Transformers (ViTs) have recently achieved strong results in semantic segmentation, yet their deployment on resource-constrained devices remains limited due to their high memory footprint and computational cost. Quantization offers an effective strategy to improve efficiency, but ViT-based segmentation models are notoriously fragile under low precision, as quantization errors accumulate across deep encoder-decoder pipelines. We introduce I-Segmenter, the first fully integer-only ViT segmentation framework. Building on the Segmenter architecture, I-Segmenter systematically replaces floating-point operations with integer-only counterparts. To further stabilize both training and inference, we propose $\lambda$-ShiftGELU, a novel activation function that mitigates the limitations of uniform quantization in handling long-tailed activation distributions. In addition, we remove the L2 normalization layer and replace bilinear interpolation in the decoder with nearest neighbor upsampling, ensuring integer-only execution throughout the computational graph. Extensive experiments show that I-Segmenter achieves accuracy within a reasonable margin of its FP32 baseline (5.1 % on average), while reducing model size by up to 3.8x and enabling up to 1.2x faster inference with optimized runtimes. Notably, even in one-shot PTQ with a single calibration image, I-Segmenter delivers competitive accuracy, underscoring its practicality for real-world deployment. 

**Abstract (ZH)**: 完全整数化ViT分割框架I-Segmenter 

---
# Generalizing Beyond Suboptimality: Offline Reinforcement Learning Learns Effective Scheduling through Random Data 

**Title (ZH)**: 超越亚优解的泛化：离线强化学习通过随机数据学习有效的调度 

**Authors**: Jesse van Remmerden, Zaharah Bukhsh, Yingqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10303)  

**Abstract**: The Job-Shop Scheduling Problem (JSP) and Flexible Job-Shop Scheduling Problem (FJSP), are canonical combinatorial optimization problems with wide-ranging applications in industrial operations. In recent years, many online reinforcement learning (RL) approaches have been proposed to learn constructive heuristics for JSP and FJSP. Although effective, these online RL methods require millions of interactions with simulated environments that may not capture real-world complexities, and their random policy initialization leads to poor sample efficiency. To address these limitations, we introduce Conservative Discrete Quantile Actor-Critic (CDQAC), a novel offline RL algorithm that learns effective scheduling policies directly from historical data, eliminating the need for costly online interactions, while maintaining the ability to improve upon suboptimal training data. CDQAC couples a quantile-based critic with a delayed policy update, estimating the return distribution of each machine-operation pair rather than selecting pairs outright. Our extensive experiments demonstrate CDQAC's remarkable ability to learn from diverse data sources. CDQAC consistently outperforms the original data-generating heuristics and surpasses state-of-the-art offline and online RL baselines. In addition, CDQAC is highly sample efficient, requiring only 10-20 training instances to learn high-quality policies. Surprisingly, we find that CDQAC performs better when trained on data generated by a random heuristic than when trained on higher-quality data from genetic algorithms and priority dispatching rules. 

**Abstract (ZH)**: 基于保守离线量纲演员评论家的作业车间调度问题与灵活作业车间调度问题的离线强化学习算法 

---
# We Need a New Ethics for a World of AI Agents 

**Title (ZH)**: 我们需要一种适用于AI代理的世界的新伦理规范 

**Authors**: Iason Gabriel, Geoff Keeling, Arianna Manzini, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2509.10289)  

**Abstract**: The deployment of capable AI agents raises fresh questions about safety, human-machine relationships and social coordination. We argue for greater engagement by scientists, scholars, engineers and policymakers with the implications of a world increasingly populated by AI agents. We explore key challenges that must be addressed to ensure that interactions between humans and agents, and among agents themselves, remain broadly beneficial. 

**Abstract (ZH)**: 有能力的AI代理的部署引发了关于安全、人机关系和社会协调的新问题。我们主张科学家、学者、工程师和政策制定者更多地关注日益普及的AI代理所带来的影响。我们探索必须解决的关键挑战，以确保人类与代理之间的互动以及代理彼此之间的互动保持总体有益。 

---
# SignClip: Leveraging Mouthing Cues for Sign Language Translation by Multimodal Contrastive Fusion 

**Title (ZH)**: SignClip: 利用口型线索实现多模态对比融合的手语翻译 

**Authors**: Wenfang Wu, Tingting Yuan, Yupeng Li, Daling Wang, Xiaoming Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10266)  

**Abstract**: Sign language translation (SLT) aims to translate natural language from sign language videos, serving as a vital bridge for inclusive communication. While recent advances leverage powerful visual backbones and large language models, most approaches mainly focus on manual signals (hand gestures) and tend to overlook non-manual cues like mouthing. In fact, mouthing conveys essential linguistic information in sign languages and plays a crucial role in disambiguating visually similar signs. In this paper, we propose SignClip, a novel framework to improve the accuracy of sign language translation. It fuses manual and non-manual cues, specifically spatial gesture and lip movement features. Besides, SignClip introduces a hierarchical contrastive learning framework with multi-level alignment objectives, ensuring semantic consistency across sign-lip and visual-text modalities. Extensive experiments on two benchmark datasets, PHOENIX14T and How2Sign, demonstrate the superiority of our approach. For example, on PHOENIX14T, in the Gloss-free setting, SignClip surpasses the previous state-of-the-art model SpaMo, improving BLEU-4 from 24.32 to 24.71, and ROUGE from 46.57 to 48.38. 

**Abstract (ZH)**: 手语翻译（SLT）旨在将自然语言翻译为手语视频，作为包容性沟通的重要桥梁。虽然近期进展利用了强大的视觉骨干和大型语言模型，但大多数方法主要关注手动信号（手势）并倾向于忽略唇部动作等非手动提示。实际上，唇部动作在手语中传达了重要的语言信息，在区分视觉上相似的手语方面起着关键作用。在本文中，我们提出了一种新的框架SignClip，以提高手语翻译的准确性。它融合了手动和非手动提示，特别是空间手势和唇部运动特征。此外，SignClip引入了一种分层对比学习框架，包含多级对齐目标，确保手语-唇部和视觉-文本模态的一致性。在两个基准数据集PHOENIX14T和How2Sign上的广泛实验表明了我们方法的优越性。例如，在PHOENIX14T数据集上，无词形设置下，SignClip超越了之前的最佳模型SpaMo，BLEU-4性能从24.32提高到24.71，ROUGE性能从46.57提高到48.38。 

---
# Openness in AI and downstream governance: A global value chain approach 

**Title (ZH)**: AI及其下游治理的开放性：全球价值链视角 

**Authors**: Christopher Foster  

**Link**: [PDF](https://arxiv.org/pdf/2509.10220)  

**Abstract**: The rise of AI has been rapid, becoming a leading sector for investment and promising disruptive impacts across the economy. Within the critical analysis of the economic impacts, AI has been aligned to the critical literature on data power and platform capitalism - further concentrating power and value capture amongst a small number of "big tech" leaders.
The equally rapid rise of openness in AI (here taken to be claims made by AI firms about openness, "open source" and free provision) signals an interesting development. It highlights an emerging ecosystem of open AI models, datasets and toolchains, involving massive capital investment. It poses questions as to whether open resources can support technological transfer and the ability for catch-up, even in the face of AI industry power.
This work seeks to add conceptual clarity to these debates by conceptualising openness in AI as a unique type of interfirm relation and therefore amenable to value chain analysis. This approach then allows consideration of the capitalist dynamics of "outsourcing" of foundational firms in value chains, and consequently the types of governance and control that might emerge downstream as AI is adopted. This work, therefore, extends previous mapping of AI value chains to build a framework which links foundational AI with downstream value chains.
Overall, this work extends our understanding of AI as a productive sector. While the work remains critical of the power of leading AI firms, openness in AI may lead to potential spillovers stemming from the intense competition for global technological leadership in AI. 

**Abstract (ZH)**: AI的发展及其开放性的兴起：对经济影响的批判性分析与价值链治理框架 

---
# SI-FACT: Mitigating Knowledge Conflict via Self-Improving Faithfulness-Aware Contrastive Tuning 

**Title (ZH)**: SI-FACT：通过自我提升忠实性意识对比调优减轻知识冲突 

**Authors**: Shengqiang Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10208)  

**Abstract**: Large Language Models often generate unfaithful responses in knowledge intensive tasks due to knowledge conflict,that is,a preference for relying on internal parametric knowledge rather than the provided this http URL address this issue,we propose a novel self improving framework,Self Improving Faithfulness Aware Contrastive this http URL framework uses a self instruct mechanism that allows the base LLM to automatically generate high quality,structured contrastive learning data,including anchor samples,semantically equivalent positive samples,and negative samples simulating unfaithful this http URL approach significantly reduces the cost of manual this http URL,contrastive learning is applied to train the model,enabling it to pull faithful responses closer and push unfaithful responses farther apart in the representation this http URL on knowledge conflict evaluation benchmarks ECARE KRE and COSE KRE show that the SI FACT model based on Llama3 8B Instruct improves the Contextual Recall Rate by 6.2% over the best baseline method,while significantly reducing dependence on internal this http URL results indicate that SI FACT provides strong effectiveness and high data efficiency in enhancing the contextual faithfulness of LLMs,offering a practical pathway toward building more proactive and trustworthy language models. 

**Abstract (ZH)**: 大规模语言模型在知识密集型任务中由于知识冲突往往会生成不忠实的响应，即偏好使用内部参数知识而非提供的知识。为解决这一问题，我们提出了一种新颖的自我提升框架，Self Improving Faithfulness Aware Contrastive框架。该框架采用自我指令机制，使基础LLM能够自动生成高质量的结构化对比学习数据，包括锚样本、语义等价正样本和模拟不忠实的负样本。这种方法显著降低了手工制作数据的成本。对比学习被应用于模型训练，使其在表示空间中将忠实响应拉近而不忠实响应推开。在知识冲突评估基准ECARE、KRE和COSE KRE上，基于Llama3 8B Instruct的SI FACT模型相比最佳基线方法，上下文再现率提高了6.2%，同时减少了对内部知识的依赖。结果表明，SI FACT在提升LLM的上下文忠实性方面具有强大的效果和高数据效率，提供了一条构建更具主动性和可信度的语言模型的实际路径。 

---
# Benchmark of stylistic variation in LLM-generated texts 

**Title (ZH)**: LLM生成文本的风格变异基准 

**Authors**: Jiří Milička, Anna Marklová, Václav Cvrček  

**Link**: [PDF](https://arxiv.org/pdf/2509.10179)  

**Abstract**: This study investigates the register variation in texts written by humans and comparable texts produced by large language models (LLMs). Biber's multidimensional analysis (MDA) is applied to a sample of human-written texts and AI-created texts generated to be their counterparts to find the dimensions of variation in which LLMs differ most significantly and most systematically from humans. As textual material, a new LLM-generated corpus AI-Brown is used, which is comparable to BE-21 (a Brown family corpus representing contemporary British English). Since all languages except English are underrepresented in the training data of frontier LLMs, similar analysis is replicated on Czech using AI-Koditex corpus and Czech multidimensional model. Examined were 16 frontier models in various settings and prompts, with emphasis placed on the difference between base models and instruction-tuned models. Based on this, a benchmark is created through which models can be compared with each other and ranked in interpretable dimensions. 

**Abstract (ZH)**: 本研究探讨了人类撰写的文本和大型语言模型（LLMs）生成的可比文本之间的体裁变异。应用Biber的多维分析（MDA）对人类撰写的文本样本和AI生成的相应文本进行分析，以找出LLMs在变异维度上与人类差异最为显著和系统性的方面。作为文本材料，使用了一个新的AI生成的语料库AI-Brown，它与BE-21（一个代表当代英国英语的Brown家族语料库）相当。由于前沿LLMs的训练数据中除了英语外其他语言严重不足，因此类似分析在捷克语上进行了复制，使用了AI-Koditex语料库和捷克语多维模型。研究考察了16种不同的前沿模型在各种情境和提示下，重点关注基础模型与指令调整模型之间的差异。基于此，创建了一个基准，通过该基准可以衡量和排名模型在可解释维度上的表现。 

---
# BenchECG and xECG: a benchmark and baseline for ECG foundation models 

**Title (ZH)**: BenchECG 和 xECG：心电图基础模型的基准和基线 

**Authors**: Riccardo Lunelli, Angus Nicolson, Samuel Martin Pröll, Sebastian Johannes Reinstadler, Axel Bauer, Clemens Dlaska  

**Link**: [PDF](https://arxiv.org/pdf/2509.10151)  

**Abstract**: Electrocardiograms (ECGs) are inexpensive, widely used, and well-suited to deep learning. Recently, interest has grown in developing foundation models for ECGs - models that generalise across diverse downstream tasks. However, consistent evaluation has been lacking: prior work often uses narrow task selections and inconsistent datasets, hindering fair comparison. Here, we introduce BenchECG, a standardised benchmark comprising a comprehensive suite of publicly available ECG datasets and versatile tasks. We also propose xECG, an xLSTM-based recurrent model trained with SimDINOv2 self-supervised learning, which achieves the best BenchECG score compared to publicly available state-of-the-art models. In particular, xECG is the only publicly available model to perform strongly on all datasets and tasks. By standardising evaluation, BenchECG enables rigorous comparison and aims to accelerate progress in ECG representation learning. xECG achieves superior performance over earlier approaches, defining a new baseline for future ECG foundation models. 

**Abstract (ZH)**: 心电图（ECGs）是经济实惠、广泛应用且非常适合深度学习的工具。近年来，人们越来越关注开发适用于ECGs的基础模型——能够在多种下游任务中泛化的模型。然而，一致的评估一直缺乏：之前的工作常常使用狭窄的任务选择和不一致的数据集，阻碍了公平的比较。在此，我们引入了BenchECG，这是一个标准化的基准，包含了一个全面的公开可用的心电图数据集套件和多功能任务。我们还提出了一种基于xLSTM的递归模型xECG，该模型使用SimDINOv2自监督学习进行训练，并在BenchECG上达到了与现有公开的最先进模型相比的最佳得分。特别是，xECG是唯一能在所有数据集和任务上表现强劲的公开可用模型。通过标准化评估，BenchECG实现了严格的比较，并旨在加速心电图表示学习的进步。xECG在早期方法上实现了更优的性能，为其后的心电图基础模型设定了新的基准。 

---
# Efficient Learning-Based Control of a Legged Robot in Lunar Gravity 

**Title (ZH)**: 基于学习的月球重力环境下腿部机器人高效控制 

**Authors**: Philip Arm, Oliver Fischer, Joseph Church, Adrian Fuhrer, Hendrik Kolvenbach, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2509.10128)  

**Abstract**: Legged robots are promising candidates for exploring challenging areas on low-gravity bodies such as the Moon, Mars, or asteroids, thanks to their advanced mobility on unstructured terrain. However, as planetary robots' power and thermal budgets are highly restricted, these robots need energy-efficient control approaches that easily transfer to multiple gravity environments. In this work, we introduce a reinforcement learning-based control approach for legged robots with gravity-scaled power-optimized reward functions. We use our approach to develop and validate a locomotion controller and a base pose controller in gravity environments from lunar gravity (1.62 m/s2) to a hypothetical super-Earth (19.62 m/s2). Our approach successfully scales across these gravity levels for locomotion and base pose control with the gravity-scaled reward functions. The power-optimized locomotion controller reached a power consumption for locomotion of 23.4 W in Earth gravity on a 15.65 kg robot at 0.4 m/s, a 23 % improvement over the baseline policy. Additionally, we designed a constant-force spring offload system that allowed us to conduct real-world experiments on legged locomotion in lunar gravity. In lunar gravity, the power-optimized control policy reached 12.2 W, 36 % less than a baseline controller which is not optimized for power efficiency. Our method provides a scalable approach to developing power-efficient locomotion controllers for legged robots across multiple gravity levels. 

**Abstract (ZH)**: 基于强化学习的重力调整能量优化四足机器人控制方法 

---
# Population-Aligned Persona Generation for LLM-based Social Simulation 

**Title (ZH)**: 基于LLM的社交模拟中的人格生成对齐方法 

**Authors**: Zhengyu Hu, Zheyuan Xiao, Max Xiong, Yuxuan Lei, Tianfu Wang, Jianxun Lian, Kaize Ding, Ziang Xiao, Nicholas Jing Yuan, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.10127)  

**Abstract**: Recent advances in large language models (LLMs) have enabled human-like social simulations at unprecedented scale and fidelity, offering new opportunities for computational social science. A key challenge, however, is the construction of persona sets that authentically represent the diversity and distribution of real-world populations. Most existing LLM-based social simulation studies focus primarily on designing agentic frameworks and simulation environments, often overlooking the complexities of persona generation and the potential biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality traits. To address the needs of specific simulation contexts, we further introduce a task-specific module that adapts the globally aligned persona set to targeted subpopulations. Extensive experiments demonstrate that our method significantly reduces population-level bias and enables accurate, flexible social simulation for a wide range of research and policy applications. 

**Abstract (ZH)**: 近期大型语言模型的发展使得以空前的规模和 fidelity 实现人类般的社会模拟成为可能，开启了一系列计算社会科学研究的新机遇。然而，一个关键挑战是如何构建能够真实反映现实世界人口多样性和分布的人格集合。大多数现有的基于大型语言模型的社会模拟研究主要关注于设计代理框架和模拟环境，往往忽视了人格生成的复杂性和由代表性不足的人格集合引入的偏见问题。本文提出了一种系统框架，用于合成高质量且与人口相匹配的人格集合，以驱动大型语言模型的社会模拟。我们的方法首先利用大型语言模型从长期社交媒体数据中生成叙事性人格，随后进行严格的质量评估，过滤掉低质量的个人资料。接着，我们使用重要性抽样使其与参考的心理统计分布（如五大人格特质）实现全局对齐。为满足特定模拟场景的需求，我们进一步引入了一个任务特定模块，使全局对齐的人格集合适应目标亚人群。广泛的实验表明，我们的方法显著减少了人口层面的偏见，并使得社会模拟能够在广泛的科研和政策应用中实现准确性和灵活性。 

---
# Realism Control One-step Diffusion for Real-World Image Super-Resolution 

**Title (ZH)**: 现实主义控制一步扩散用于真实世界图像超分辨率 

**Authors**: Zongliang Wu, Siming Zheng, Peng-Tao Jiang, Xin Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.10122)  

**Abstract**: Pre-trained diffusion models have shown great potential in real-world image super-resolution (Real-ISR) tasks by enabling high-resolution reconstructions. While one-step diffusion (OSD) methods significantly improve efficiency compared to traditional multi-step approaches, they still have limitations in balancing fidelity and realism across diverse scenarios. Since the OSDs for SR are usually trained or distilled by a single timestep, they lack flexible control mechanisms to adaptively prioritize these competing objectives, which are inherently manageable in multi-step methods through adjusting sampling steps. To address this challenge, we propose a Realism Controlled One-step Diffusion (RCOD) framework for Real-ISR. RCOD provides a latent domain grouping strategy that enables explicit control over fidelity-realism trade-offs during the noise prediction phase with minimal training paradigm modifications and original training data. A degradation-aware sampling strategy is also introduced to align distillation regularization with the grouping strategy and enhance the controlling of trade-offs. Moreover, a visual prompt injection module is used to replace conventional text prompts with degradation-aware visual tokens, enhancing both restoration accuracy and semantic consistency. Our method achieves superior fidelity and perceptual quality while maintaining computational efficiency. Extensive experiments demonstrate that RCOD outperforms state-of-the-art OSD methods in both quantitative metrics and visual qualities, with flexible realism control capabilities in the inference stage. The code will be released. 

**Abstract (ZH)**: 预训练扩散模型在真实场景超分辨率任务中展示了巨大的潜力，通过实现高分辨率重构。虽然一阶扩散（OSD）方法在效率上显著优于传统的多步骤方法，但在不同场景中平衡保真度和真实感方面仍有局限性。由于SR的OSD通常由单一时间步训练或蒸馏，缺乏灵活的控制机制来适应性地优先考虑这些相互竞争的目标，而这些目标在多步骤方法中通过调整采样步骤是可管理的。为解决这一挑战，我们提出了一种用于真实场景超分辨率的现实控制一阶扩散（RCOD）框架。RCOD提供了一种潜在领域分组策略，在噪声预测阶段能显式控制保真度-真实感权衡，同时最小化训练范式修改和使用原始训练数据。引入了一种退化感知采样策略来使蒸馏正则化与分组策略对齐，并增强权衡控制。此外，使用视觉提示注入模块，用退化感知视觉标记替代传统文本提示，提高恢复准确性和语义一致性。我们的方法在保持计算效率的同时实现了更优的保真度和感知质量。大量实验表明，RCOD在定量指标和视觉质量上均优于最先进的OSD方法，在推断阶段具有灵活的现实控制能力。代码将开源。 

---
# Generating Energy-Efficient Code via Large-Language Models -- Where are we now? 

**Title (ZH)**: 基于大型语言模型生成能源高效代码：我们现在在哪里？ 

**Authors**: Radu Apsan, Vincenzo Stoico, Michel Albonico, Rudra Dhar, Karthik Vaidhyanathan, Ivano Malavolta  

**Link**: [PDF](https://arxiv.org/pdf/2509.10099)  

**Abstract**: Context. The rise of Large Language Models (LLMs) has led to their widespread adoption in development pipelines. Goal. We empirically assess the energy efficiency of Python code generated by LLMs against human-written code and code developed by a Green software expert. Method. We test 363 solutions to 9 coding problems from the EvoEval benchmark using 6 widespread LLMs with 4 prompting techniques, and comparing them to human-developed solutions. Energy consumption is measured on three different hardware platforms: a server, a PC, and a Raspberry Pi for a total of ~881h (36.7 days). Results. Human solutions are 16% more energy-efficient on the server and 3% on the Raspberry Pi, while LLMs outperform human developers by 25% on the PC. Prompting does not consistently lead to energy savings, where the most energy-efficient prompts vary by hardware platform. The code developed by a Green software expert is consistently more energy-efficient by at least 17% to 30% against all LLMs on all hardware platforms. Conclusions. Even though LLMs exhibit relatively good code generation capabilities, no LLM-generated code was more energy-efficient than that of an experienced Green software developer, suggesting that as of today there is still a great need of human expertise for developing energy-efficient Python code. 

**Abstract (ZH)**: 大规模语言模型的兴起导致了其在开发管道中的广泛应用。目标. 我们实证评估生成的Python代码（由LLM生成）与人工编写的代码及绿色软件专家开发的代码相比的能源效率。方法. 我们使用6种流行的LLM及其4种提示技术，对EvoEval基准中的9个编码问题的363个解决方案进行了测试，并将其与人工开发的解决方案进行了比较。能源消耗在三种不同的硬件平台上进行了测量：服务器、PC和Raspberry Pi，总计约881小时（36.7天）。结果. 人工解决方案在服务器上比LLM代码节能16%，在Raspberry Pi上节能3%；而LLM代码在PC上的能源效率比人工开发者高25%。不同硬件平台上，最节能的提示并不一致。绿色软件专家开发的代码在所有硬件平台上比所有LLM代码至少节能17%到30%。结论. 尽管LLM展现出相当不错的代码生成能力，但没有任何LLM生成的代码比经验丰富的绿色软件开发者编写的代码更节能，这表明目前仍然需要人类专家来开发节能的Python代码。 

---
# Established Psychometric vs. Ecologically Valid Questionnaires: Rethinking Psychological Assessments in Large Language Models 

**Title (ZH)**: 传统的心理测量问卷 vs. 生态有效的问卷：重新思考大规模语言模型的心理评估 

**Authors**: Dongmin Choi, Woojung Song, Jongwook Han, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.10078)  

**Abstract**: Researchers have applied established psychometric questionnaires (e.g., BFI, PVQ) to measure the personality traits and values reflected in the responses of Large Language Models (LLMs). However, concerns have been raised about applying these human-designed questionnaires to LLMs. One such concern is their lack of ecological validity--the extent to which survey questions adequately reflect and resemble real-world contexts in which LLMs generate texts in response to user queries. However, it remains unclear how established questionnaires and ecologically valid questionnaires differ in their outcomes, and what insights these differences may provide. In this paper, we conduct a comprehensive comparative analysis of the two types of questionnaires. Our analysis reveals that established questionnaires (1) yield substantially different profiles of LLMs from ecologically valid ones, deviating from the psychological characteristics expressed in the context of user queries, (2) suffer from insufficient items for stable measurement, (3) create misleading impressions that LLMs possess stable constructs, and (4) yield exaggerated profiles for persona-prompted LLMs. Overall, our work cautions against the use of established psychological questionnaires for LLMs. Our code will be released upon publication. 

**Abstract (ZH)**: 研究人员已运用成熟的心理测量问卷（如BFI、PVQ）来衡量大型语言模型（LLMs）回应用户查询时体现的人格特质和价值观。然而，有人对将这些由人类设计的问卷应用于LLMs表示关切。其中一项关切在于它们缺乏生态效度——问卷问题在多大程度上能真实反映和模拟LLMs生成文本时所处的实际情境。然而，尚不清楚这两种类型的问卷在结果上存在何种差异，这些差异又能提供哪些洞见。在本文中，我们进行了全面的比较分析。我们的分析显示，成熟的问卷（1）在生成的LLM人格特质画像上与生态效度问卷有显著差异，偏离了用户查询情境中体现的心理特征；（2）存在项目不足，难以稳定测量；（3）营造了错误的持久结构印象，使人们误以为LLMs具备稳定的心理结构；（4）夸大了角色提示下LLM的人格特征画像。总体而言，我们的研究警示不应使用成熟的心理健康问卷来评估LLMs。论文发表后，我们将发布相关代码。 

---
# Predictive Spike Timing Enables Distributed Shortest Path Computation in Spiking Neural Networks 

**Title (ZH)**: 预测尖峰时间实现分布式最短路径计算在尖峰神经网络中 

**Authors**: Simen Storesund, Kristian Valset Aars, Robin Dietrich, Nicolai Waniek  

**Link**: [PDF](https://arxiv.org/pdf/2509.10077)  

**Abstract**: Efficient planning and sequence selection are central to intelligence, yet current approaches remain largely incompatible with biological computation. Classical graph algorithms like Dijkstra's or A* require global state and biologically implausible operations such as backtracing, while reinforcement learning methods rely on slow gradient-based policy updates that appear inconsistent with rapid behavioral adaptation observed in natural systems.
We propose a biologically plausible algorithm for shortest-path computation that operates through local spike-based message-passing with realistic processing delays. The algorithm exploits spike-timing coincidences to identify nodes on optimal paths: Neurons that receive inhibitory-excitatory message pairs earlier than predicted reduce their response delays, creating a temporal compression that propagates backwards from target to source. Through analytical proof and simulations on random spatial networks, we demonstrate that the algorithm converges and discovers all shortest paths using purely timing-based mechanisms. By showing how short-term timing dynamics alone can compute shortest paths, this work provides new insights into how biological networks might solve complex computational problems through purely local computation and relative spike-time prediction. These findings open new directions for understanding distributed computation in biological and artificial systems, with possible implications for computational neuroscience, AI, reinforcement learning, and neuromorphic systems. 

**Abstract (ZH)**: 高效的路径规划和序列选择是智能的关键，但当前的方法与生物计算仍不兼容。经典的图算法如迪杰斯特拉算法或A*算法需要全局状态和不具生物合理性的回溯操作，而基于梯度的强化学习方法依赖于缓慢的策略更新，这似乎与自然系统中观察到的快速行为适应不一致。
我们提出了一种具有生物合理性的最短路径计算算法，通过局部尖峰基消息传递并在具备现实处理延迟的情况下工作。该算法利用尖峰时间 coincidence 来识别最优路径上的节点：较早接收到抑制性-兴奋性消息对的神经元降低其响应延迟，从而在目标到源头方向上产生时间压缩。通过分析证明和在随机空间网络上的模拟，我们展示了该算法仅通过基于时间的机制即可收敛并发现所有最短路径。通过展示仅通过短期时间动态即可计算最短路径，这项工作为理解生物网络如何通过纯粹的局部计算和相对尖峰时间预测解决复杂计算问题提供了新的见解。这些发现为理解生物和人工系统的分布式计算开辟了新的方向，并可能对计算神经科学、AI、强化学习和类脑系统产生影响。 

---
# TwinTac: A Wide-Range, Highly Sensitive Tactile Sensor with Real-to-Sim Digital Twin Sensor Model 

**Title (ZH)**: TwinTac: 一种宽范围高性能触觉传感器及其实时到仿真数字孪生传感器模型 

**Authors**: Xiyan Huang, Zhe Xu, Chenxi Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.10063)  

**Abstract**: Robot skill acquisition processes driven by reinforcement learning often rely on simulations to efficiently generate large-scale interaction data. However, the absence of simulation models for tactile sensors has hindered the use of tactile sensing in such skill learning processes, limiting the development of effective policies driven by tactile perception. To bridge this gap, we present TwinTac, a system that combines the design of a physical tactile sensor with its digital twin model. Our hardware sensor is designed for high sensitivity and a wide measurement range, enabling high quality sensing data essential for object interaction tasks. Building upon the hardware sensor, we develop the digital twin model using a real-to-sim approach. This involves collecting synchronized cross-domain data, including finite element method results and the physical sensor's outputs, and then training neural networks to map simulated data to real sensor responses. Through experimental evaluation, we characterized the sensitivity of the physical sensor and demonstrated the consistency of the digital twin in replicating the physical sensor's output. Furthermore, by conducting an object classification task, we showed that simulation data generated by our digital twin sensor can effectively augment real-world data, leading to improved accuracy. These results highlight TwinTac's potential to bridge the gap in cross-domain learning tasks. 

**Abstract (ZH)**: 基于孪生模型的物理触觉传感器及其数字孪生系统在强化学习驱动的机器人技能获取中的应用 

---
# Multimodal Mathematical Reasoning Embedded in Aerial Vehicle Imagery: Benchmarking, Analysis, and Exploration 

**Title (ZH)**: 基于航拍图像的多模态数学推理：基准测试、分析与探索 

**Authors**: Yue Zhou, Litong Feng, Mengcheng Lan, Xue Yang, Qingyun Li, Yiping Ke, Xue Jiang, Wayne Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10059)  

**Abstract**: Mathematical reasoning is critical for tasks such as precise distance and area computations, trajectory estimations, and spatial analysis in unmanned aerial vehicle (UAV) based remote sensing, yet current vision-language models (VLMs) have not been adequately tested in this domain. To address this gap, we introduce AVI-Math, the first benchmark to rigorously evaluate multimodal mathematical reasoning in aerial vehicle imagery, moving beyond simple counting tasks to include domain-specific knowledge in areas such as geometry, logic, and algebra. The dataset comprises 3,773 high-quality vehicle-related questions captured from UAV views, covering 6 mathematical subjects and 20 topics. The data, collected at varying altitudes and from multiple UAV angles, reflects real-world UAV scenarios, ensuring the diversity and complexity of the constructed mathematical problems. In this paper, we benchmark 14 prominent VLMs through a comprehensive evaluation and demonstrate that, despite their success on previous multimodal benchmarks, these models struggle with the reasoning tasks in AVI-Math. Our detailed analysis highlights significant limitations in the mathematical reasoning capabilities of current VLMs and suggests avenues for future research. Furthermore, we explore the use of Chain-of-Thought prompting and fine-tuning techniques, which show promise in addressing the reasoning challenges in AVI-Math. Our findings not only expose the limitations of VLMs in mathematical reasoning but also offer valuable insights for advancing UAV-based trustworthy VLMs in real-world applications. The code, and datasets will be released at this https URL 

**Abstract (ZH)**: 数学推理对于无人驾驶航空车辆（UAV）远程 sensing 中的精确距离和面积计算、轨迹估算以及空间分析至关重要，然而当前的视觉-语言模型（VLMs）在这个领域中尚未得到充分测试。为解决这一差距，我们引入了 AVI-Math，这是首个严格评估空中车辆图像中多模态数学推理的标准，超越了简单的计数任务，包括几何学、逻辑学和代数等领域的专业知识。数据集包含3,773个高质量的与车辆相关的问题，覆盖6个数学主题和20个领域。数据在不同高度和多个无人机视角下收集，反映出真实的无人机场景，确保了构建的数学问题的多样性和复杂性。在本文中，我们通过全面的评估对比了14个著名的VLMs，并展示了尽管这些模型在之前的多模态基准测试中表现出色，但在AVI-Math中的推理任务中却面临挑战。我们详细的分析突显了当前VLMs在数学推理能力方面的重大局限性，并提出了未来研究的方向。此外，我们探讨了利用思维链提示和微调技术，这些技术有望解决AVI-Math中的推理挑战。我们的发现不仅揭示了VLMs在数学推理方面的局限性，还为在实际应用中推进基于无人机的信任VLMs提供了有价值的见解。代码和数据集将在此链接中发布：this https URL。 

---
# Reinforcement learning for spin torque oscillator tasks 

**Title (ZH)**: 自旋扭矩振荡器任务的强化学习方法 

**Authors**: Jakub Mojsiejuk, Sławomir Ziętek, Witold Skowroński  

**Link**: [PDF](https://arxiv.org/pdf/2509.10057)  

**Abstract**: We address the problem of automatic synchronisation of the spintronic oscillator (STO) by means of reinforcement learning (RL). A numerical solution of the macrospin Landau-Lifschitz-Gilbert-Slonczewski equation is used to simulate the STO and we train the two types of RL agents to synchronise with a target frequency within a fixed number of steps. We explore modifications to this base task and show an improvement in both convergence and energy efficiency of the synchronisation that can be easily achieved in the simulated environment. 

**Abstract (ZH)**: 我们通过强化学习解决自旋电子振荡器自动同步问题：通过数值求解宏观自旋兰杜-利夫西茨-吉尔伯特-斯隆切夫斯基方程模拟自旋电子振荡器，并训练两种类型的RL代理在固定步数内与目标频率同步。我们探索了对该基本任务的修改，并展示了在模拟环境中轻松实现的同步收敛性和能效的改进。 

---
# Exploring Expert Specialization through Unsupervised Training in Sparse Mixture of Experts 

**Title (ZH)**: 通过无监督训练在稀疏专家混合模型中探索专家专业化 

**Authors**: Strahinja Nikolic, Ilker Oguz, Demetri Psaltis  

**Link**: [PDF](https://arxiv.org/pdf/2509.10025)  

**Abstract**: Understanding the internal organization of neural networks remains a fundamental challenge in deep learning interpretability. We address this challenge by exploring a novel Sparse Mixture of Experts Variational Autoencoder (SMoE-VAE) architecture. We test our model on the QuickDraw dataset, comparing unsupervised expert routing against a supervised baseline guided by ground-truth labels. Surprisingly, we find that unsupervised routing consistently achieves superior reconstruction performance. The experts learn to identify meaningful sub-categorical structures that often transcend human-defined class boundaries. Through t-SNE visualizations and reconstruction analysis, we investigate how MoE models uncover fundamental data structures that are more aligned with the model's objective than predefined labels. Furthermore, our study on the impact of dataset size provides insights into the trade-offs between data quantity and expert specialization, offering guidance for designing efficient MoE architectures. 

**Abstract (ZH)**: 理解神经网络的内部组织仍然是深度学习可解释性中的一个基本挑战。我们通过探索一种新颖的稀疏专家混合变分自动编码器（SMoE-VAE）架构来应对这一挑战。我们在QuickDraw数据集上测试了我们的模型，将无监督专家路由与受 ground-truth 标签引导的有监督基线进行比较。令人惊讶的是，我们发现无监督路由在重构性能上持续表现出优越的成果。专家学会了识别有意义的子分类结构，这些结构往往超越了人类定义的类别边界。通过t-SNE可视化和重构分析，我们研究了MoE模型如何揭示与模型目标更加一致的基本数据结构，而不是预定义的标签。此外，我们关于数据集大小影响的研究为专家专业化与数据量之间的权衡提供了见解，为设计高效的MoE架构提供了指导。 

---
# Intrinsic Dimension Estimating Autoencoder (IDEA) Using CancelOut Layer and a Projected Loss 

**Title (ZH)**: 基于CancelOut层和投影损失的内在维度估计自编码器（IDEA） 

**Authors**: Antoine Orioua, Philipp Krah, Julian Koellermeier  

**Link**: [PDF](https://arxiv.org/pdf/2509.10011)  

**Abstract**: This paper introduces the Intrinsic Dimension Estimating Autoencoder (IDEA), which identifies the underlying intrinsic dimension of a wide range of datasets whose samples lie on either linear or nonlinear manifolds. Beyond estimating the intrinsic dimension, IDEA is also able to reconstruct the original dataset after projecting it onto the corresponding latent space, which is structured using re-weighted double CancelOut layers. Our key contribution is the introduction of the projected reconstruction loss term, guiding the training of the model by continuously assessing the reconstruction quality under the removal of an additional latent dimension. We first assess the performance of IDEA on a series of theoretical benchmarks to validate its robustness. These experiments allow us to test its reconstruction ability and compare its performance with state-of-the-art intrinsic dimension estimators. The benchmarks show good accuracy and high versatility of our approach. Subsequently, we apply our model to data generated from the numerical solution of a vertically resolved one-dimensional free-surface flow, following a pointwise discretization of the vertical velocity profile in the horizontal direction, vertical direction, and time. IDEA succeeds in estimating the dataset's intrinsic dimension and then reconstructs the original solution by working directly within the projection space identified by the network. 

**Abstract (ZH)**: 本文介绍了本征维数估计自编码器（IDEA），该方法能够识别广泛范围的数据集的本征维数，这些数据集的样本位于线性或非线性流形上。除了估计本征维数外，IDEA 还能够在将数据集投影到相应的潜在空间后重建原始数据集，该潜在空间使用加权双CancelOut层结构化。我们的主要贡献在于引入了投影重构损失项，通过在移除额外潜在维数的情况下连续评估重构质量来引导模型训练。我们首先在一系列理论基准上评估IDEA的性能，以验证其鲁棒性。这些实验允许我们测试其重构能力，并将其性能与现有的本征维数估计器进行比较。基准测试表明，我们的方法具有良好的准确性和高度的通用性。随后，我们将模型应用于垂直解析的一维自由表面流动的数值解数据，该数据通过水平方向、垂直方向和时间上的点式离散化生成。IDEA 成功地估计了数据集的本征维数，并通过在由网络识别的投影空间中直接工作来重构原始解。 

---
# Unsupervised Hallucination Detection by Inspecting Reasoning Processes 

**Title (ZH)**: 无监督的幻觉检测通过检验推理过程 

**Authors**: Ponhvoan Srey, Xiaobao Wu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10004)  

**Abstract**: Unsupervised hallucination detection aims to identify hallucinated content generated by large language models (LLMs) without relying on labeled data. While unsupervised methods have gained popularity by eliminating labor-intensive human annotations, they frequently rely on proxy signals unrelated to factual correctness. This misalignment biases detection probes toward superficial or non-truth-related aspects, limiting generalizability across datasets and scenarios. To overcome these limitations, we propose IRIS, an unsupervised hallucination detection framework, leveraging internal representations intrinsic to factual correctness. IRIS prompts the LLM to carefully verify the truthfulness of a given statement, and obtain its contextualized embedding as informative features for training. Meanwhile, the uncertainty of each response is considered a soft pseudolabel for truthfulness. Experimental results demonstrate that IRIS consistently outperforms existing unsupervised methods. Our approach is fully unsupervised, computationally low cost, and works well even with few training data, making it suitable for real-time detection. 

**Abstract (ZH)**: 无监督幻觉检测旨在无需依赖标注数据的情况下识别由大规模语言模型（LLMs）生成的幻觉内容。虽然无监督方法通过消除 labor-intensive 的人工标注而受到青睐，但它们经常依赖于与事实正确性无关的代理信号。这种不对齐使检测探针偏向于表面或非事实相关的方面，限制了其跨数据集和场景的一般性。为克服这些限制，我们提出 IRIS，一种利用与事实正确性内在一致的内部表示的无监督幻觉检测框架。IRIS 促使 LLM 仔细验证给定陈述的真实性和获得其上下文化嵌入作为训练的特征信息。同时，每个响应的不确定性被视为真实性的软伪标签。实验结果表明，IRIS 一贯优于现有的无监督方法。我们的方法完全无监督、计算成本低，并即使在少量训练数据的情况下也能很好地工作，使其适用于实时检测。 

---
# Drone-Based Multispectral Imaging and Deep Learning for Timely Detection of Branched Broomrape in Tomato Farms 

**Title (ZH)**: 基于无人机多光谱成像和深度学习的及时检测番茄田间枝状雀麦方法 

**Authors**: Mohammadreza Narimani, Alireza Pourreza, Ali Moghimi, Mohsen Mesgaran, Parastoo Farajpoor, Hamid Jafarbiglu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09972)  

**Abstract**: This study addresses the escalating threat of branched broomrape (Phelipanche ramosa) to California's tomato industry, which supplies over 90 percent of U.S. processing tomatoes. The parasite's largely underground life cycle makes early detection difficult, while conventional chemical controls are costly, environmentally harmful, and often ineffective. To address this, we combined drone-based multispectral imagery with Long Short-Term Memory (LSTM) deep learning networks, using the Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance. Research was conducted on a known broomrape-infested tomato farm in Woodland, Yolo County, CA, across five key growth stages determined by growing degree days (GDD). Multispectral images were processed to isolate tomato canopy reflectance. At 897 GDD, broomrape could be detected with 79.09 percent overall accuracy and 70.36 percent recall without integrating later stages. Incorporating sequential growth stages with LSTM improved detection substantially. The best-performing scenario, which integrated all growth stages with SMOTE augmentation, achieved 88.37 percent overall accuracy and 95.37 percent recall. These results demonstrate the strong potential of temporal multispectral analysis and LSTM networks for early broomrape detection. While further real-world data collection is needed for practical deployment, this study shows that UAV-based multispectral sensing coupled with deep learning could provide a powerful precision agriculture tool to reduce losses and improve sustainability in tomato production. 

**Abstract (ZH)**: 基于无人机多谱段成像和LSTM网络的加州番茄Industry髯草早期检测研究 

---
# Securing LLM-Generated Embedded Firmware through AI Agent-Driven Validation and Patching 

**Title (ZH)**: 通过AI代理驱动的验证和补丁修复确保LLM生成的嵌入式固件安全 

**Authors**: Seyed Moein Abtahi, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09970)  

**Abstract**: Large Language Models (LLMs) show promise in generating firmware for embedded systems, but often introduce security flaws and fail to meet real-time performance constraints. This paper proposes a three-phase methodology that combines LLM-based firmware generation with automated security validation and iterative refinement in a virtualized environment. Using structured prompts, models like GPT-4 generate firmware for networking and control tasks, deployed on FreeRTOS via QEMU. These implementations are tested using fuzzing, static analysis, and runtime monitoring to detect vulnerabilities such as buffer overflows (CWE-120), race conditions (CWE-362), and denial-of-service threats (CWE-400). Specialized AI agents for Threat Detection, Performance Optimization, and Compliance Verification collaborate to improve detection and remediation. Identified issues are categorized using CWE, then used to prompt targeted LLM-generated patches in an iterative loop. Experiments show a 92.4\% Vulnerability Remediation Rate (37.3\% improvement), 95.8\% Threat Model Compliance, and 0.87 Security Coverage Index. Real-time metrics include 8.6ms worst-case execution time and 195{\mu}s jitter. This process enhances firmware security and performance while contributing an open-source dataset for future research. 

**Abstract (ZH)**: 基于大规模语言模型的嵌入式系统固件生成及其自动化安全验证和迭代优化方法 

---
# Large Language Models Meet Legal Artificial Intelligence: A Survey 

**Title (ZH)**: 大型语言模型与法律人工智能的交汇：一个综述 

**Authors**: Zhitian Hou, Zihan Ye, Nanli Zeng, Tianyong Hao, Kun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09969)  

**Abstract**: Large Language Models (LLMs) have significantly advanced the development of Legal Artificial Intelligence (Legal AI) in recent years, enhancing the efficiency and accuracy of legal tasks. To advance research and applications of LLM-based approaches in legal domain, this paper provides a comprehensive review of 16 legal LLMs series and 47 LLM-based frameworks for legal tasks, and also gather 15 benchmarks and 29 datasets to evaluate different legal capabilities. Additionally, we analyse the challenges and discuss future directions for LLM-based approaches in the legal domain. We hope this paper provides a systematic introduction for beginners and encourages future research in this field. Resources are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）近年来显著推动了法律人工智能（Legal AI）的发展，提高了法律任务的效率和准确性。为了推进基于LLM的方法在法律领域的研究和应用，本文对16系列法律LLM和47个基于LLM的法律任务框架进行了全面回顾，并收集了15个基准和29个数据集以评估不同的法律能力。此外，我们分析了基于LLM方法的挑战，并讨论了法律领域的未来发展方向。我们希望本文能为初学者提供系统的介绍，并鼓励未来对该领域的研究。相关资源可在以下链接获取：this https URL。 

---
# Limited Reference, Reliable Generation: A Two-Component Framework for Tabular Data Generation in Low-Data Regimes 

**Title (ZH)**: 有限参考，可靠生成：低数据环境下表数据生成的两组件框架 

**Authors**: Mingxuan Jiang, Yongxin Wang, Ziyue Dai, Yicun Liu, Hongyi Nie, Sen Liu, Hongfeng Chai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09960)  

**Abstract**: Synthetic tabular data generation is increasingly essential in data management, supporting downstream applications when real-world and high-quality tabular data is insufficient. Existing tabular generation approaches, such as generative adversarial networks (GANs), diffusion models, and fine-tuned Large Language Models (LLMs), typically require sufficient reference data, limiting their effectiveness in domain-specific databases with scarce records. While prompt-based LLMs offer flexibility without parameter tuning, they often fail to capture dataset-specific feature-label dependencies and generate redundant data, leading to degradation in downstream task performance. To overcome these issues, we propose ReFine, a framework that (i) derives symbolic "if-then" rules from interpretable models and embeds them into prompts to explicitly guide generation toward domain-specific feature distribution, and (ii) applies a dual-granularity filtering strategy that suppresses over-sampling patterns and selectively refines rare but informative samples to reduce distributional imbalance. Extensive experiments on various regression and classification benchmarks demonstrate that ReFine consistently outperforms state-of-the-art methods, achieving up to 0.44 absolute improvement in R-squared for regression and 10.0 percent relative improvement in F1 score for classification tasks. 

**Abstract (ZH)**: 合成表格数据生成在数据管理中的应用越来越重要，当现实世界的高质量表格数据不足时，它可以支持下游应用。现有表格生成方法，如生成对抗网络（GANs）、扩散模型和微调的大语言模型（LLMs），通常需要足够的参考数据，这限制了它们在稀少记录的专业领域数据库中的效果。虽然基于提示的大语言模型具有灵活性且不需要参数调整，但它们往往无法捕捉到数据集特定的特征-标签依赖关系，并生成冗余数据，从而在下游任务性能上产生退化。为了克服这些问题，我们提出了ReFine框架，该框架通过（i）从可解释模型中推导出符号的“如果-那么”规则，并将这些规则嵌入到提示中，以明确地指导生成朝向特定领域的特征分布，以及（ii）应用一种双粒度的过滤策略，抑制过度采样的模式，并选择性地精炼稀少但具有信息性的样本，以减少分布不平衡。在各种回归和分类基准上的 extensive 实验表明，ReFine 一致地超越了最先进的方法，在回归任务中 R-squared 值上最高提高了 0.44，在分类任务中 F1 分数上相对提高了 10.0%。 

---
# Zero-Shot Referring Expression Comprehension via Visual-Language True/False Verification 

**Title (ZH)**: 零样本引用表达理解通过视觉-语言真伪验证 

**Authors**: Jeffrey Liu, Rongbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09958)  

**Abstract**: Referring Expression Comprehension (REC) is usually addressed with task-trained grounding models. We show that a zero-shot workflow, without any REC-specific training, can achieve competitive or superior performance. Our approach reformulates REC as box-wise visual-language verification: given proposals from a COCO-clean generic detector (YOLO-World), a general-purpose VLM independently answers True/False queries for each region. This simple procedure reduces cross-box interference, supports abstention and multiple matches, and requires no fine-tuning. On RefCOCO, RefCOCO+, and RefCOCOg, our method not only surpasses a zero-shot GroundingDINO baseline but also exceeds reported results for GroundingDINO trained on REC and GroundingDINO+CRG. Controlled studies with identical proposals confirm that verification significantly outperforms selection-based prompting, and results hold with open VLMs. Overall, we show that workflow design, rather than task-specific pretraining, drives strong zero-shot REC performance. 

**Abstract (ZH)**: 参考表达理解（REC）通常通过任务训练的 grounding 模型来解决。我们表明，在没有任何 REC 特异性训练的情况下，一种零样本工作流能够达到竞争性或更优的性能。我们的方法将 REC 重新表述为框级别的视觉-语言验证：给定 COCO-clean 通用检测器（YOLO-World）的建议框，一个通用的 VLM 独立地对每个区域进行 True/False 查询的回答。这一简单的过程减少了跨框干扰，支持弃权和多匹配，并不需要微调。在 RefCOCO、RefCOCO+ 和 RefCOCOg 上，我们的方法不仅超越了零样本 GroundingDINO 基线，还超过了在 REC 上训练的 GroundingDINO 和 GroundingDINO+CRG 的报告结果。通过使用相同的建议框进行的控制实验确认了验证显著优于基于选择的提示，且结果适用于开放的 VLM。总体而言，我们表明，工作流设计而非任务特异性预训练驱动了强大的零样本 REC 表现。 

---
# Adaptive Token Merging for Efficient Transformer Semantic Communication at the Edge 

**Title (ZH)**: 边缘设备上高效的Transformer语义通信自适应 token 合并 

**Authors**: Omar Erak, Omar Alhussein, Hatem Abou-Zeid, Mehdi Bennis, Sami Muhaidat  

**Link**: [PDF](https://arxiv.org/pdf/2509.09955)  

**Abstract**: Large-scale transformers are central to modern semantic communication, yet their high computational and communication costs hinder deployment on resource-constrained edge devices. This paper introduces a training-free framework for adaptive token merging, a novel mechanism that compresses transformer representations at runtime by selectively merging semantically redundant tokens under per-layer similarity thresholds. Unlike prior fixed-ratio reduction, our approach couples merging directly to input redundancy, enabling data-dependent adaptation that balances efficiency and task relevance without retraining. We cast the discovery of merging strategies as a multi-objective optimization problem and leverage Bayesian optimization to obtain Pareto-optimal trade-offs between accuracy, inference cost, and communication cost. On ImageNet classification, we match the accuracy of the unmodified transformer with 30\% fewer floating-point operations per second and under 20\% of the original communication cost, while for visual question answering our method achieves performance competitive with the full LLaVA model at less than one-third of the compute and one-tenth of the bandwidth. Finally, we show that our adaptive merging is robust across varying channel conditions and provides inherent privacy benefits, substantially degrading the efficacy of model inversion attacks. Our framework provides a practical and versatile solution for deploying powerful transformer models in resource-limited edge intelligence scenarios. 

**Abstract (ZH)**: 一种基于训练的自适应 token 合并框架：在受限边缘设备上高效压缩变压器表示 

---
# SmartCoder-R1: Towards Secure and Explainable Smart Contract Generation with Security-Aware Group Relative Policy Optimization 

**Title (ZH)**: SmartCoder-R1：面向安全可解释的智能合约生成的安全意识群体相对策略优化方法 

**Authors**: Lei Yu, Jingyuan Zhang, Xin Wang, Jiajia Ma, Li Yang, Fengjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09942)  

**Abstract**: Smart contracts automate the management of high-value assets, where vulnerabilities can lead to catastrophic financial losses. This challenge is amplified in Large Language Models (LLMs) by two interconnected failures: they operate as unauditable "black boxes" lacking a transparent reasoning process, and consequently, generate code riddled with critical security vulnerabilities. To address both issues, we propose SmartCoder-R1 (based on Qwen2.5-Coder-7B), a novel framework for secure and explainable smart contract generation. It begins with Continual Pre-training (CPT) to specialize the model. We then apply Long Chain-of-Thought Supervised Fine-Tuning (L-CoT SFT) on 7,998 expert-validated reasoning-and-code samples to train the model to emulate human security analysis. Finally, to directly mitigate vulnerabilities, we employ Security-Aware Group Relative Policy Optimization (S-GRPO), a reinforcement learning phase that refines the generation policy by optimizing a weighted reward signal for compilation success, security compliance, and format correctness. Evaluated against 17 baselines on a benchmark of 756 real-world functions, SmartCoder-R1 establishes a new state of the art, achieving top performance across five key metrics: a ComPass of 87.70%, a VulRate of 8.60%, a SafeAval of 80.16%, a FuncRate of 53.84%, and a FullRate of 50.53%. This FullRate marks a 45.79% relative improvement over the strongest baseline, DeepSeek-R1. Crucially, its generated reasoning also excels in human evaluations, achieving high-quality ratings for Functionality (82.7%), Security (85.3%), and Clarity (90.7%). 

**Abstract (ZH)**: 智能合约自动管理高价值资产，漏洞可能导致巨大的财务损失。在大型语言模型（LLMs）中，这一挑战由两个相互关联的失败放大：它们作为不可审计的“黑箱”运行，缺乏透明的推理过程，从而生成充满关键安全漏洞的代码。为解决这两个问题，我们提出了一种新的智能合约生成框架SmartCoder-R1（基于Qwen2.5-Coder-7B）。该框架首先通过持续预训练（CPT）使模型专门化。然后，我们应用长推理链监督微调（L-CoT SFT）针对7,998个专家验证的推理和代码样本对模型进行训练，使其模仿人类的安全分析。最后，为了直接缓解漏洞，我们引入了感知安全的组相对策略优化（S-GRPO），这是一种强化学习阶段，通过优化编译成功率、安全合规性和格式正确性的加权奖励信号来细化生成策略。在涵盖756个真实函数基准测试中，SmartCoder-R1在五个关键指标上实现了顶级性能：ComPass为87.70%，VulRate为8.60%，SafeAval为80.16%，FuncRate为53.84%，FullRate为50.53%。其中，FullRate相较最强基线DeepSeek-R1提高了45.79%。此外，生成的推理在人工评估中也表现出色，功能、安全性和清晰度分别获得了高质量评分82.7%、85.3%和90.7%。 

---
# WALL: A Web Application for Automated Quality Assurance using Large Language Models 

**Title (ZH)**: WALL：一种基于大型语言模型的自动化质量保证Web应用程序 

**Authors**: Seyed Moein Abtahi, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09918)  

**Abstract**: As software projects become increasingly complex, the volume and variety of issues in code files have grown substantially. Addressing this challenge requires efficient issue detection, resolution, and evaluation tools. This paper presents WALL, a web application that integrates SonarQube and large language models (LLMs) such as GPT-3.5 Turbo and GPT-4o to automate these tasks. WALL comprises three modules: an issue extraction tool, code issues reviser, and code comparison tool. Together, they enable a seamless pipeline for detecting software issues, generating automated code revisions, and evaluating the accuracy of revisions. Our experiments, conducted on 563 files with over 7,599 issues, demonstrate WALL's effectiveness in reducing human effort while maintaining high-quality revisions. Results show that employing a hybrid approach of cost-effective and advanced LLMs can significantly lower costs and improve revision rates. Future work aims to enhance WALL's capabilities by integrating open-source LLMs and eliminating human intervention, paving the way for fully automated code quality management. 

**Abstract (ZH)**: 随着软件项目日益复杂，代码文件中的问题数量和种类大幅增加。应对这一挑战需要高效的 ISSUE 检测、解决和评估工具。本文介绍了一种名为 WALL 的网络应用，该应用集成了 SonarQube 和大型语言模型（LLMs），如 GPT-3.5 Turbo 和 GPT-4o，以自动化这些任务。WALL 包含三个模块：问题提取工具、代码问题修改器和代码比较工具。它们共同实现了从检测软件问题、生成自动化代码修改到评估修改准确性的无缝流程。我们在 563 个文件（超过 7,599 个问题）上进行的实验表明，WALL 在减少人力投入的同时，仍能保持高质量的修改。结果表明，采用经济有效和先进技术相结合的 LLM 混合方法，可以显著降低成本并提高修改率。未来的工作将通过集成开源 LLM 并消除人工干预，提升 WALL 的能力，为完全自动化的代码质量管理工作铺平道路。 

---
# An Autoencoder and Vision Transformer-based Interpretability Analysis of the Differences in Automated Staging of Second and Third Molars 

**Title (ZH)**: 基于自动编码器和视觉变换器的第二和第三磨牙自动化分期差异可解释性分析 

**Authors**: Barkin Buyukcakir, Jannick De Tobel, Patrick Thevissen, Dirk Vandermeulen, Peter Claes  

**Link**: [PDF](https://arxiv.org/pdf/2509.09911)  

**Abstract**: The practical adoption of deep learning in high-stakes forensic applications, such as dental age estimation, is often limited by the 'black box' nature of the models. This study introduces a framework designed to enhance both performance and transparency in this context. We use a notable performance disparity in the automated staging of mandibular second (tooth 37) and third (tooth 38) molars as a case study. The proposed framework, which combines a convolutional autoencoder (AE) with a Vision Transformer (ViT), improves classification accuracy for both teeth over a baseline ViT, increasing from 0.712 to 0.815 for tooth 37 and from 0.462 to 0.543 for tooth 38. Beyond improving performance, the framework provides multi-faceted diagnostic insights. Analysis of the AE's latent space metrics and image reconstructions indicates that the remaining performance gap is data-centric, suggesting high intra-class morphological variability in the tooth 38 dataset is a primary limiting factor. This work highlights the insufficiency of relying on a single mode of interpretability, such as attention maps, which can appear anatomically plausible yet fail to identify underlying data issues. By offering a methodology that both enhances accuracy and provides evidence for why a model may be uncertain, this framework serves as a more robust tool to support expert decision-making in forensic age estimation. 

**Abstract (ZH)**: 高风险法医应用（如牙龄估计）中深度学习的实际应用往往受限于模型的“黑盒”性质。本文提出了一种框架以提高性能和透明度。我们以下颔第二前磨牙（牙号37）和第三前磨牙（牙号38）的自动分期为例，展示了该框架的优势。提出的框架结合了卷积自编码器（AE）和视觉变换器（ViT），在牙号37和牙号38的分类准确率上均超过了基线ViT，分别从0.712提高到0.815和从0.462提高到0.543。除了提高性能，该框架还提供了多方面的诊断洞察。分析自编码器的潜在空间度量和图像重构表明，剩余的性能差距主要来源于数据问题，暗示牙号38的数据集中存在较高的类内形态变异是主要限制因素。本研究强调单靠一种解释性模式（如注意图）是不足的，因为这类模式虽然在解剖学上看似合理，却未能识别潜在的数据问题。通过提供既能提高准确性又能解释模型不确定性的方法，该框架为法医年龄估计中的专家决策提供了更为 robust 的支持工具。 

---
# Tackling One Health Risks: How Large Language Models are leveraged for Risk Negotiation and Consensus-building 

**Title (ZH)**: 应对跨学科健康风险：大型语言模型在风险谈判与共识构建中的应用 

**Authors**: Alexandra Fetsch, Iurii Savvateev, Racem Ben Romdhane, Martin Wiedmann, Artemiy Dimov, Maciej Durkalec, Josef Teichmann, Jakob Zinsstag, Konstantinos Koutsoumanis, Andreja Rajkovic, Jason Mann, Mauro Tonolla, Monika Ehling-Schulz, Matthias Filter, Sophia Johler  

**Link**: [PDF](https://arxiv.org/pdf/2509.09906)  

**Abstract**: Key global challenges of our times are characterized by complex interdependencies and can only be effectively addressed through an integrated, participatory effort. Conventional risk analysis frameworks often reduce complexity to ensure manageability, creating silos that hinder comprehensive solutions. A fundamental shift towards holistic strategies is essential to enable effective negotiations between different sectors and to balance the competing interests of stakeholders. However, achieving this balance is often hindered by limited time, vast amounts of information, and the complexity of integrating diverse perspectives. This study presents an AI-assisted negotiation framework that incorporates large language models (LLMs) and AI-based autonomous agents into a negotiation-centered risk analysis workflow. The framework enables stakeholders to simulate negotiations, systematically model dynamics, anticipate compromises, and evaluate solution impacts. By leveraging LLMs' semantic analysis capabilities we could mitigate information overload and augment decision-making process under time constraints. Proof-of-concept implementations were conducted in two real-world scenarios: (i) prudent use of a biopesticide, and (ii) targeted wild animal population control. Our work demonstrates the potential of AI-assisted negotiation to address the current lack of tools for cross-sectoral engagement. Importantly, the solution's open source, web based design, suits for application by a broader audience with limited resources and enables users to tailor and develop it for their own needs. 

**Abstract (ZH)**: 当前全球面临的根本性挑战具有复杂的相互依赖性，只有通过整合和参与的方式才能得到有效解决。传统的风险分析框架常常简化复杂性以确保可管理性，从而形成信息孤岛，阻碍了全面解决方案的形成。向整体策略的根本转变，对于促进不同部门之间的有效谈判并平衡各利益相关方的竞争利益至关重要。然而，实现这种平衡往往受到时间有限、信息量庞大以及整合多样视角的复杂性的影响。本研究提出了一种基于AI的谈判框架，将大型语言模型（LLMs）和基于AI的自主代理融入以谈判为中心的风险分析工作流程中。该框架使利益相关方能够模拟谈判、系统地建模动态、预见妥协并评估解决方案的影响。通过利用LLMs的语义分析能力，我们可以缓解信息过载，并在时间受限的情况下增强决策过程。概念验证的实施在两个实际场景中进行：（i）生物农药的谨慎使用；（ii）野生物种数量的靶向控制。我们的研究表明，基于AI的谈判能够弥合当前跨部门交流中缺乏工具的空白。重要的是，该解决方案的开源、基于Web的设计，有利于资源有限的更广泛用户群体，并允许用户根据自身需求进行定制和开发。 

---
# Self-Augmented Robot Trajectory: Efficient Imitation Learning via Safe Self-augmentation with Demonstrator-annotated Precision 

**Title (ZH)**: 自我增强机器人轨迹：通过安全自我扩充实现的高效模仿学习 

**Authors**: Hanbit Oh, Masaki Murooka, Tomohiro Motoda, Ryoichi Nakajo, Yukiyasu Domae  

**Link**: [PDF](https://arxiv.org/pdf/2509.09893)  

**Abstract**: Imitation learning is a promising paradigm for training robot agents; however, standard approaches typically require substantial data acquisition -- via numerous demonstrations or random exploration -- to ensure reliable performance. Although exploration reduces human effort, it lacks safety guarantees and often results in frequent collisions -- particularly in clearance-limited tasks (e.g., peg-in-hole) -- thereby, necessitating manual environmental resets and imposing additional human burden. This study proposes Self-Augmented Robot Trajectory (SART), a framework that enables policy learning from a single human demonstration, while safely expanding the dataset through autonomous augmentation. SART consists of two stages: (1) human teaching only once, where a single demonstration is provided and precision boundaries -- represented as spheres around key waypoints -- are annotated, followed by one environment reset; (2) robot self-augmentation, where the robot generates diverse, collision-free trajectories within these boundaries and reconnects to the original demonstration. This design improves the data collection efficiency by minimizing human effort while ensuring safety. Extensive evaluations in simulation and real-world manipulation tasks show that SART achieves substantially higher success rates than policies trained solely on human-collected demonstrations. Video results available at this https URL . 

**Abstract (ZH)**: 自我增强机器人轨迹（SART）：一种从单个人机演示中学习的框架 

---
# Automated Tuning for Diffusion Inverse Problem Solvers without Generative Prior Retraining 

**Title (ZH)**: 无需重新训练生成式先验的扩散逆问题求解器自动调优 

**Authors**: Yaşar Utku Alçalar, Junno Yun, Mehmet Akçakaya  

**Link**: [PDF](https://arxiv.org/pdf/2509.09880)  

**Abstract**: Diffusion/score-based models have recently emerged as powerful generative priors for solving inverse problems, including accelerated MRI reconstruction. While their flexibility allows decoupling the measurement model from the learned prior, their performance heavily depends on carefully tuned data fidelity weights, especially under fast sampling schedules with few denoising steps. Existing approaches often rely on heuristics or fixed weights, which fail to generalize across varying measurement conditions and irregular timestep schedules. In this work, we propose Zero-shot Adaptive Diffusion Sampling (ZADS), a test-time optimization method that adaptively tunes fidelity weights across arbitrary noise schedules without requiring retraining of the diffusion prior. ZADS treats the denoising process as a fixed unrolled sampler and optimizes fidelity weights in a self-supervised manner using only undersampled measurements. Experiments on the fastMRI knee dataset demonstrate that ZADS consistently outperforms both traditional compressed sensing and recent diffusion-based methods, showcasing its ability to deliver high-fidelity reconstructions across varying noise schedules and acquisition settings. 

**Abstract (ZH)**: 零-shot自适应扩散采样（ZADS）：一种无需重新训练扩散先验的测试时优化方法 

---
# From Hugging Face to GitHub: Tracing License Drift in the Open-Source AI Ecosystem 

**Title (ZH)**: 从Hugging Face到GitHub：追踪开源AI生态系统中的许可漂移 

**Authors**: James Jewitt, Hao Li, Bram Adams, Gopi Krishnan Rajbahadur, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09873)  

**Abstract**: Hidden license conflicts in the open-source AI ecosystem pose serious legal and ethical risks, exposing organizations to potential litigation and users to undisclosed risk. However, the field lacks a data-driven understanding of how frequently these conflicts occur, where they originate, and which communities are most affected. We present the first end-to-end audit of licenses for datasets and models on Hugging Face, as well as their downstream integration into open-source software applications, covering 364 thousand datasets, 1.6 million models, and 140 thousand GitHub projects. Our empirical analysis reveals systemic non-compliance in which 35.5% of model-to-application transitions eliminate restrictive license clauses by relicensing under permissive terms. In addition, we prototype an extensible rule engine that encodes almost 200 SPDX and model-specific clauses for detecting license conflicts, which can solve 86.4% of license conflicts in software applications. To support future research, we release our dataset and the prototype engine. Our study highlights license compliance as a critical governance challenge in open-source AI and provides both the data and tools necessary to enable automated, AI-aware compliance at scale. 

**Abstract (ZH)**: 开源AI生态系统中的隐藏许可冲突存在严重的法律和伦理风险，使组织面临潜在诉讼，用户面临未披露的风险。然而，该领域缺乏对这些冲突发生频率、起源以及受影响最严重的社区的理解。我们首次对Hugging Face上数据集、模型及其在开源软件应用中的下游集成进行全面审计，覆盖36.4万个项目集、160万模型和14万GitHub项目。我们的实证分析揭示了系统性的不合规情况，其中35.5%的模型到应用过渡通过转用宽松条款来消除限制性许可条款。此外，我们设计了一个可扩展的规则引擎，编码了近200条SPDX和模型特定条款以检测许可冲突，该引擎可以解决软件应用中86.4%的许可冲突。为了支持未来研究，我们发布了我们的数据集和原型引擎。我们的研究突出了开源AI中的许可合规性作为一个关键治理挑战，并提供了必要的数据和工具以实现大规模、智能感知的自动化合规管理。 

---
# Emulating Public Opinion: A Proof-of-Concept of AI-Generated Synthetic Survey Responses for the Chilean Case 

**Title (ZH)**: 模仿公众意见：智利案例的AI生成合成调查回应原理验证 

**Authors**: Bastián González-Bustamante, Nando Verelst, Carla Cisternas  

**Link**: [PDF](https://arxiv.org/pdf/2509.09871)  

**Abstract**: Large Language Models (LLMs) offer promising avenues for methodological and applied innovations in survey research by using synthetic respondents to emulate human answers and behaviour, potentially mitigating measurement and representation errors. However, the extent to which LLMs recover aggregate item distributions remains uncertain and downstream applications risk reproducing social stereotypes and biases inherited from training data. We evaluate the reliability of LLM-generated synthetic survey responses against ground-truth human responses from a Chilean public opinion probabilistic survey. Specifically, we benchmark 128 prompt-model-question triplets, generating 189,696 synthetic profiles, and pool performance metrics (i.e., accuracy, precision, recall, and F1-score) in a meta-analysis across 128 question-subsample pairs to test for biases along key sociodemographic dimensions. The evaluation spans OpenAI's GPT family and o-series reasoning models, as well as Llama and Qwen checkpoints. Three results stand out. First, synthetic responses achieve excellent performance on trust items (F1-score and accuracy > 0.90). Second, GPT-4o, GPT-4o-mini and Llama 4 Maverick perform comparably on this task. Third, synthetic-human alignment is highest among respondents aged 45-59. Overall, LLM-based synthetic samples approximate responses from a probabilistic sample, though with substantial item-level heterogeneity. Capturing the full nuance of public opinion remains challenging and requires careful calibration and additional distributional tests to ensure algorithmic fidelity and reduce errors. 

**Abstract (ZH)**: 大型语言模型在通过合成受访者模拟人类答案和行为以进行调查研究的方法论和应用创新方面的前景令人鼓舞，这可能减轻测量和代表误差。然而，大型语言模型恢复汇总项目分布的程度仍然不确定，下游应用存在复制训练数据中继承的社会刻板印象和偏见的风险。我们评估了大型语言模型生成的合成调查响应与来自智利公共意见概率调查的真实人类响应的一致性。具体而言，我们针对128个提示-模型-问题三元组基准测试，生成了189,696个合成档案，并在128个问题子样本对中汇总准确率、召回率、精确率和F1分数等性能指标进行元分析，以检验关键社会人口维度上的偏差。评估涵盖OpenAI的GPT家族和o系列推理模型、Llama和Qwen等检查点。三个结果突出：首先，合成响应在信任项目上表现优异（F1分数和准确率>0.90）；其次，GPT-4o、GPT-4o-mini和Llama 4 Maverick在这一任务上表现相当；第三，45-59岁受访者的合成-人类对齐度最高。总体而言，基于大型语言模型的合成样本能够逼近概率样本的回答，尽管存在项目层面的显著异质性。全面捕捉公众意见的细微差别仍然具有挑战性，需要仔细校准和进行额外的分布测试，以确保算法忠实度并减少错误。 

---
# Vibe Check: Understanding the Effects of LLM-Based Conversational Agents' Personality and Alignment on User Perceptions in Goal-Oriented Tasks 

**Title (ZH)**: 情绪检验：基于LLM的对话代理个性和对齐对其在目标导向任务中用户感知的影响理解 

**Authors**: Hasibur Rahman, Smit Desai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09870)  

**Abstract**: Large language models (LLMs) enable conversational agents (CAs) to express distinctive personalities, raising new questions about how such designs shape user perceptions. This study investigates how personality expression levels and user-agent personality alignment influence perceptions in goal-oriented tasks. In a between-subjects experiment (N=150), participants completed travel planning with CAs exhibiting low, medium, or high expression across the Big Five traits, controlled via our novel Trait Modulation Keys framework. Results revealed an inverted-U relationship: medium expression produced the most positive evaluations across Intelligence, Enjoyment, Anthropomorphism, Intention to Adopt, Trust, and Likeability, significantly outperforming both extremes. Personality alignment further enhanced outcomes, with Extraversion and Emotional Stability emerging as the most influential traits. Cluster analysis identified three distinct compatibility profiles, with "Well-Aligned" users reporting substantially positive perceptions. These findings demonstrate that personality expression and strategic trait alignment constitute optimal design targets for CA personality, offering design implications as LLM-based CAs become increasingly prevalent. 

**Abstract (ZH)**: 大型语言模型（LLMs）使对话代理（CAs）能够表现出独特的个性，从而引发关于此类设计如何影响用户感知的新问题。本研究探讨了个性表达水平和用户-代理个性一致性如何影响目标导向任务中的感知。在被试间实验中（N=150），参与者使用表现出低、中、高强度五大特质表达的CAs来完成旅行规划任务，通过我们新颖的特质调节键框架控制这一差异。结果表明存在倒U型关系：中等表达水平在智力、享受、拟人性、采用意图、信任和吸引力方面产生了最多的正面评价，显著优于两端极端水平。个性一致性进一步提升了结果，外向性和情绪稳定性是最具影响力的特质。聚类分析识别出三种不同的兼容性模式，“完美兼容”的用户报告了显著的正面感知。这些发现表明，个性表达和战略性特质一致性构成了CA个性的最佳设计目标，随着基于LLM的CA越来越普遍，这为设计提供了重要启示。 

---
# Surrogate Supervision for Robust and Generalizable Deformable Image Registration 

**Title (ZH)**: 代理监督学习以实现鲁棒且可泛化的变形图像配准 

**Authors**: Yihao Liu, Junyu Chen, Lianrui Zuo, Shuwen Wei, Brian D. Boyd, Carmen Andreescu, Olusola Ajilore, Warren D. Taylor, Aaron Carass, Bennett A. Landman  

**Link**: [PDF](https://arxiv.org/pdf/2509.09869)  

**Abstract**: Objective: Deep learning-based deformable image registration has achieved strong accuracy, but remains sensitive to variations in input image characteristics such as artifacts, field-of-view mismatch, or modality difference. We aim to develop a general training paradigm that improves the robustness and generalizability of registration networks. Methods: We introduce surrogate supervision, which decouples the input domain from the supervision domain by applying estimated spatial transformations to surrogate images. This allows training on heterogeneous inputs while ensuring supervision is computed in domains where similarity is well defined. We evaluate the framework through three representative applications: artifact-robust brain MR registration, mask-agnostic lung CT registration, and multi-modal MR registration. Results: Across tasks, surrogate supervision demonstrated strong resilience to input variations including inhomogeneity field, inconsistent field-of-view, and modality differences, while maintaining high performance on well-curated data. Conclusions: Surrogate supervision provides a principled framework for training robust and generalizable deep learning-based registration models without increasing complexity. Significance: Surrogate supervision offers a practical pathway to more robust and generalizable medical image registration, enabling broader applicability in diverse biomedical imaging scenarios. 

**Abstract (ZH)**: 基于深度学习的可变形图像配准稳健性和普适性提高的通用训练范式：通过替代监督实现 

---
# Latency and Token-Aware Test-Time Compute 

**Title (ZH)**: 时延感知的标记aware测试时计算 

**Authors**: Jenny Y. Huang, Mehul Damani, Yousef El-Kurdi, Ramon Astudillo, Wei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.09864)  

**Abstract**: Inference-time scaling has emerged as a powerful way to improve large language model (LLM) performance by generating multiple candidate responses and selecting among them. However, existing work on dynamic allocation for test-time compute typically considers only parallel generation methods such as best-of-N, overlooking incremental decoding methods like beam search, and has largely ignored latency, focusing only on token usage. We formulate inference-time scaling as a problem of dynamic compute allocation and method selection, where the system must decide which strategy to apply and how much compute to allocate on a per-query basis. Our framework explicitly incorporates both token cost and wall-clock latency, the latter being critical for user experience and particularly for agentic workflows where models must issue multiple queries efficiently. Experiments on reasoning benchmarks show that our approach consistently outperforms static strategies, achieving favorable accuracy-cost trade-offs while remaining practical for deployment. 

**Abstract (ZH)**: 推理时动态计算分配与方法选择：一种兼顾 Tokens 成本和 wall-clock 延迟的策略 

---
# SWE-Effi: Re-Evaluating Software AI Agent System Effectiveness Under Resource Constraints 

**Title (ZH)**: SWE-Effi: 在资源约束条件下重新评估软件AI代理系统的效果 

**Authors**: Zhiyu Fan, Kirill Vasilevski, Dayi Lin, Boyuan Chen, Yihao Chen, Zhiqing Zhong, Jie M. Zhang, Pinjia He, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09853)  

**Abstract**: The advancement of large language models (LLMs) and code agents has demonstrated significant potential to assist software engineering (SWE) tasks, such as autonomous issue resolution and feature addition. Existing AI for software engineering leaderboards (e.g., SWE-bench) focus solely on solution accuracy, ignoring the crucial factor of effectiveness in a resource-constrained world. This is a universal problem that also exists beyond software engineering tasks: any AI system should be more than correct - it must also be cost-effective. To address this gap, we introduce SWE-Effi, a set of new metrics to re-evaluate AI systems in terms of holistic effectiveness scores. We define effectiveness as the balance between the accuracy of outcome (e.g., issue resolve rate) and the resources consumed (e.g., token and time). In this paper, we specifically focus on the software engineering scenario by re-ranking popular AI systems for issue resolution on a subset of the SWE-bench benchmark using our new multi-dimensional metrics. We found that AI system's effectiveness depends not just on the scaffold itself, but on how well it integrates with the base model, which is key to achieving strong performance in a resource-efficient manner. We also identified systematic challenges such as the "token snowball" effect and, more significantly, a pattern of "expensive failures". In these cases, agents consume excessive resources while stuck on unsolvable tasks - an issue that not only limits practical deployment but also drives up the cost of failed rollouts during RL training. Lastly, we observed a clear trade-off between effectiveness under the token budget and effectiveness under the time budget, which plays a crucial role in managing project budgets and enabling scalable reinforcement learning, where fast responses are essential. 

**Abstract (ZH)**: 大语言模型（LLMs）和代码代理的进步展示了在软件工程（SWE）任务中，如自主问题解决和功能增加方面的显著潜力。现有的软件工程AI排行榜（如SWE-bench）仅侧重于解决方案的准确性，忽视了资源受限世界中的有效性的关键因素。这是一个普遍的问题，也存在于软件工程任务之外：任何AI系统不仅要正确，还必须成本效益高。为解决这一差距，我们提出了SWE-Effi，一套新的度量标准，用于从整体有效性评分方面重新评估AI系统。我们将有效性定义为结果准确性（例如，问题解决率）与消耗资源（例如，token和时间）之间的平衡。在本文中，我们特别关注通过重新排序SWE-bench基准中的一部分问题解决AI系统来评估我们的新多维度指标。我们发现AI系统的有效性不仅取决于框架本身，还取决于其与基础模型的整合程度，这对于以资源高效的方式实现良好的性能至关重要。我们还指出了系统性的挑战，如“token雪球”效应，并且更加显著的是“昂贵的失败”模式。在这些情况下，代理在无法解决的任务上消耗过度资源，这不仅限制了实际部署，还在强化学习（RL）训练期间导致失败的推出成本增加。最后，我们观察到，在token预算下的有效性与在时间预算下的有效性之间存在明显的权衡关系，这对管理项目预算和实现可扩展的强化学习至关重要，因为快速响应是必要的。 

---
# HGEN: Heterogeneous Graph Ensemble Networks 

**Title (ZH)**: HGEN: 异构图ensemble网络 

**Authors**: Jiajun Shen, Yufei Jin, Yi He, Xingquan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09843)  

**Abstract**: This paper presents HGEN that pioneers ensemble learning for heterogeneous graphs. We argue that the heterogeneity in node types, nodal features, and local neighborhood topology poses significant challenges for ensemble learning, particularly in accommodating diverse graph learners. Our HGEN framework ensembles multiple learners through a meta-path and transformation-based optimization pipeline to uplift classification accuracy. Specifically, HGEN uses meta-path combined with random dropping to create Allele Graph Neural Networks (GNNs), whereby the base graph learners are trained and aligned for later ensembling. To ensure effective ensemble learning, HGEN presents two key components: 1) a residual-attention mechanism to calibrate allele GNNs of different meta-paths, thereby enforcing node embeddings to focus on more informative graphs to improve base learner accuracy, and 2) a correlation-regularization term to enlarge the disparity among embedding matrices generated from different meta-paths, thereby enriching base learner diversity. We analyze the convergence of HGEN and attest its higher regularization magnitude over simple voting. Experiments on five heterogeneous networks validate that HGEN consistently outperforms its state-of-the-art competitors by substantial margin. 

**Abstract (ZH)**: 本论文提出了HGEN，开创了异构图的集成学习方法。我们argue认为节点类型、节点特征以及局部邻域拓扑的异构性对集成学习构成了重大挑战，特别是对于容纳多种图学习器的挑战。HGEN框架通过元路径和基于转化的优化管道组合多个学习器，以提升分类准确性。具体而言，HGEN使用结合随机丢弃的元路径来创建等位基因图神经网络（GNNs），通过这种方式训练和对齐基图学习器以供后续集成使用。为了确保有效的集成学习，HGEN提出了两个关键组件：1）剩余注意机制，以校准不同元路径的等位基因GNNs，从而增强节点表示，集中于更具信息量的图，以提高基学习器的准确性；2）相关性正则化项，旨在扩大由不同元路径生成的嵌入矩阵之间的差异，从而增加基学习器的多样性。我们分析了HGEN的收敛性，并证实其正则化程度高于简单的投票方法。实验结果表明，HGEN在五个异构网络上的表现显著优于当前最先进的竞争对手。 

---
# Revisiting Actor-Critic Methods in Discrete Action Off-Policy Reinforcement Learning 

**Title (ZH)**: 离散动作 Offline 政策强化学习中 Actor-Critic 方法的重新审视 

**Authors**: Reza Asad, Reza Babanezhad, Sharan Vaswani  

**Link**: [PDF](https://arxiv.org/pdf/2509.09838)  

**Abstract**: Value-based approaches such as DQN are the default methods for off-policy reinforcement learning with discrete-action environments such as Atari. Common policy-based methods are either on-policy and do not effectively learn from off-policy data (e.g. PPO), or have poor empirical performance in the discrete-action setting (e.g. SAC). Consequently, starting from discrete SAC (DSAC), we revisit the design of actor-critic methods in this setting. First, we determine that the coupling between the actor and critic entropy is the primary reason behind the poor performance of DSAC. We demonstrate that by merely decoupling these components, DSAC can have comparable performance as DQN. Motivated by this insight, we introduce a flexible off-policy actor-critic framework that subsumes DSAC as a special case. Our framework allows using an m-step Bellman operator for the critic update, and enables combining standard policy optimization methods with entropy regularization to instantiate the resulting actor objective. Theoretically, we prove that the proposed methods can guarantee convergence to the optimal regularized value function in the tabular setting. Empirically, we demonstrate that these methods can approach the performance of DQN on standard Atari games, and do so even without entropy regularization or explicit exploration. 

**Abstract (ZH)**: 基于价值的方法如DQN是离策强化学习中，默认的方法，特别是在 Atari 这类离散动作环境中。常见的基于策略的方法要么是在线策的，不能有效从离策数据中学习（例如 PPO），要么在离散动作设置中有较差的实证表现（例如 SAC）。因此，从离散 SAC (DSAC) 开始，我们重新审视了这类设置下的演员-评论家方法的设计。首先，我们确定演员和评论家熵之间的耦合是 DSAC 表现不佳的主要原因。我们通过仅解除这些组件之间的耦合，证明 DSAC 可以达到与 DQN 相媲美的性能。受此启发，我们提出了一种灵活的离策演员-评论家框架，将 DSAC 作为其特殊情形。该框架允许使用 m 步贝尔曼运算符来更新评论家，并能够结合标准策略优化方法和熵正则化来实例化结果演员目标。理论上，我们证明所提出的方法可以在表征设置下保证收敛到最优的正则化价值函数。实证上，我们展示这些方法可以接近标准 Atari 游戏中的 DQN 性能，甚至无需熵正则化或显式探索也能做到这一点。 

---
# CoDiCodec: Unifying Continuous and Discrete Compressed Representations of Audio 

**Title (ZH)**: CoDiCodec: 统一连续和离散音频压缩表示 

**Authors**: Marco Pasini, Stefan Lattner, George Fazekas  

**Link**: [PDF](https://arxiv.org/pdf/2509.09836)  

**Abstract**: Efficiently representing audio signals in a compressed latent space is critical for latent generative modelling. However, existing autoencoders often force a choice between continuous embeddings and discrete tokens. Furthermore, achieving high compression ratios while maintaining audio fidelity remains a challenge. We introduce CoDiCodec, a novel audio autoencoder that overcomes these limitations by both efficiently encoding global features via summary embeddings, and by producing both compressed continuous embeddings at ~ 11 Hz and discrete tokens at a rate of 2.38 kbps from the same trained model, offering unprecedented flexibility for different downstream generative tasks. This is achieved through Finite Scalar Quantization (FSQ) and a novel FSQ-dropout technique, and does not require additional loss terms beyond the single consistency loss used for end-to-end training. CoDiCodec supports both autoregressive decoding and a novel parallel decoding strategy, with the latter achieving superior audio quality and faster decoding. CoDiCodec outperforms existing continuous and discrete autoencoders at similar bitrates in terms of reconstruction audio quality. Our work enables a unified approach to audio compression, bridging the gap between continuous and discrete generative modelling paradigms. 

**Abstract (ZH)**: CoDiCodec：克服现有限制的新型音频自编码器 

---
# SoilSound: Smartphone-based Soil Moisture Estimation 

**Title (ZH)**: 土壤之声：基于智能手机的土壤含水量估计 

**Authors**: Yixuan Gao, Tanvir Ahmed, Shuang He, Zhongqi Cheng, Rajalakshmi Nandakumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.09823)  

**Abstract**: Soil moisture monitoring is essential for agriculture and environmental management, yet existing methods require either invasive probes disturbing the soil or specialized equipment, limiting access to the public. We present SoilSound, an ubiquitous accessible smartphone-based acoustic sensing system that can measure soil moisture without disturbing the soil. We leverage the built-in speaker and microphone to perform a vertical scan mechanism to accurately measure moisture without any calibration. Unlike existing work that use transmissive properties, we propose an alternate model for acoustic reflections in soil based on the surface roughness effect to enable moisture sensing without disturbing the soil. The system works by sending acoustic chirps towards the soil and recording the reflections during a vertical scan, which are then processed and fed to a convolutional neural network for on-device soil moisture estimation with negligible computational, memory, or power overhead. We evaluated the system by training with curated soils in boxes in the lab and testing in the outdoor fields and show that SoilSound achieves a mean absolute error (MAE) of 2.39% across 10 different locations. Overall, the evaluation shows that SoilSound can accurately track soil moisture levels ranging from 15.9% to 34.0% across multiple soil types, environments, and users; without requiring any calibration or disturbing the soil, enabling widespread moisture monitoring for home gardeners, urban farmers, citizen scientists, and agricultural communities in resource-limited settings. 

**Abstract (ZH)**: 土壤水分监测对于农业和环境管理至关重要，但现有方法要么需要破坏土壤的侵入式探针，要么需要专门的设备，限制了公众的访问。我们提出了SoilSound，这是一种基于智能手机的普遍可访问声学传感系统，能够在不破坏土壤的情况下测量土壤水分。我们利用内置的扬声器和麦克风进行垂直扫描机制，以准确测量水分而无需任何校准。与现有方法依赖透射特性不同，我们提出了基于表面粗糙效应的土壤声学反射模型，以实现不破坏土壤的情况下的水分感知。该系统通过向土壤发送声脉冲并在垂直扫描过程中记录反射，然后对这些数据进行处理并输入卷积神经网络进行设备上的土壤水分估计，几乎没有任何计算、内存或功耗开销。我们通过在实验室中的定制土壤盒子中训练并在户外田地进行测试，展示了SoilSound在不同位置实现了2.39%的平均绝对误差（MAE）。总体来说，评估表明SoilSound能够在多种土壤类型、环境和用户的情况下准确跟踪土壤水分水平（15.9%至34.0%），无需任何校准或破坏土壤，从而使具有限资源的环境下家庭园丁、城市农民、公民科学家和农业社区能够进行广泛的水分监测。 

---
# HEFT: A Coarse-to-Fine Hierarchy for Enhancing the Efficiency and Accuracy of Language Model Reasoning 

**Title (ZH)**: HEFT：一种从粗到细的层级框架，以提高语言模型推理的效率和准确性 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.09801)  

**Abstract**: The adaptation of large language models (LLMs) to specialized reasoning tasks is fundamentally constrained by computational resources. Parameter-Efficient Fine-Tuning (PEFT) methods have emerged as a powerful solution, yet the landscape of these techniques is diverse, with distinct methods operating in either the model's weight space or its representation space. This paper investigates the hypothesis that a synergistic combination of these paradigms can unlock superior performance and efficiency. We introduce HEFT (Hierarchical Efficient Fine-Tuning), a novel hierarchical adaptation strategy that composes two distinct PEFT methods in a coarse-to-fine manner: first, a broad, foundational adaptation in the weight space using Low-Rank Adaptation (LoRA), followed by a precise, surgical refinement of internal activations using Representation Fine-Tuning (ReFT). We evaluate this approach by fine-tuning a Llama-2-7B model on the BoolQ benchmark, a challenging dataset for inferential reasoning. Our results reveal a profound synergistic effect. A model fine-tuned for only three epochs with our HEFT strategy achieves an accuracy of 85.17\%, exceeding the performance of models trained for 20 epochs with either LoRA-only (85.05\%) or ReFT-only (83.36\%) methodologies. This work demonstrates that the thoughtful composition of PEFT methods is a potent algorithmic innovation, offering a more efficient and effective path toward advancing the reasoning capabilities of language models. By achieving superior results with a fraction of the computational budget, our findings present a principled approach to overcoming the obstacles inherent in adapting large-scale models for complex cognitive tasks. 

**Abstract (ZH)**: 高效层次微调：Hierarchical Efficient Fine-Tuning 

---
# ZORRO: Zero-Knowledge Robustness and Privacy for Split Learning (Full Version) 

**Title (ZH)**: ZORRO: 零知识鲁棒性和隐私保护的分裂学习（完整版本） 

**Authors**: Nojan Sheybani, Alessandro Pegoraro, Jonathan Knauer, Phillip Rieger, Elissa Mollakuqe, Farinaz Koushanfar, Ahmad-Reza Sadeghi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09787)  

**Abstract**: Split Learning (SL) is a distributed learning approach that enables resource-constrained clients to collaboratively train deep neural networks (DNNs) by offloading most layers to a central server while keeping in- and output layers on the client-side. This setup enables SL to leverage server computation capacities without sharing data, making it highly effective in resource-constrained environments dealing with sensitive data. However, the distributed nature enables malicious clients to manipulate the training process. By sending poisoned intermediate gradients, they can inject backdoors into the shared DNN. Existing defenses are limited by often focusing on server-side protection and introducing additional overhead for the server. A significant challenge for client-side defenses is enforcing malicious clients to correctly execute the defense algorithm.
We present ZORRO, a private, verifiable, and robust SL defense scheme. Through our novel design and application of interactive zero-knowledge proofs (ZKPs), clients prove their correct execution of a client-located defense algorithm, resulting in proofs of computational integrity attesting to the benign nature of locally trained DNN portions. Leveraging the frequency representation of model partitions enables ZORRO to conduct an in-depth inspection of the locally trained models in an untrusted environment, ensuring that each client forwards a benign checkpoint to its succeeding client. In our extensive evaluation, covering different model architectures as well as various attack strategies and data scenarios, we show ZORRO's effectiveness, as it reduces the attack success rate to less than 6\% while causing even for models storing \numprint{1000000} parameters on the client-side an overhead of less than 10 seconds. 

**Abstract (ZH)**: ZORRO：一种私密、可验证且 robust 的 Split Learning 防护方案 

---
# LAVa: Layer-wise KV Cache Eviction with Dynamic Budget Allocation 

**Title (ZH)**: LAVa：层级键值缓存逐层 eviction 动态预算分配 

**Authors**: Yiqun Shen, Song Yuan, Zhengze Zhang, Xiaoliang Wang, Daxin Jiang, Nguyen Cam-Tu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09754)  

**Abstract**: KV Cache is commonly used to accelerate LLM inference with long contexts, yet its high memory demand drives the need for cache compression. Existing compression methods, however, are largely heuristic and lack dynamic budget allocation. To address this limitation, we introduce a unified framework for cache compression by minimizing information loss in Transformer residual streams. Building on it, we analyze the layer attention output loss and derive a new metric to compare cache entries across heads, enabling layer-wise compression with dynamic head budgets. Additionally, by contrasting cross-layer information, we also achieve dynamic layer budgets. LAVa is the first unified strategy for cache eviction and dynamic budget allocation that, unlike prior methods, does not rely on training or the combination of multiple strategies. Experiments with benchmarks (LongBench, Needle-In-A-Haystack, Ruler, and InfiniteBench) demonstrate its superiority. Moreover, our experiments reveal a new insight: dynamic layer budgets are crucial for generation tasks (e.g., code completion), while dynamic head budgets play a key role in extraction tasks (e.g., extractive QA). As a fully dynamic compression method, LAVa consistently maintains top performance across task types. Our code is available at this https URL. 

**Abstract (ZH)**: 统一框架下的Transformer残差流压缩：具备动态预算分配的LAVa缓存优化方法 

---
# Meta-Learning Reinforcement Learning for Crypto-Return Prediction 

**Title (ZH)**: 元学习强化学习加密货币收益预测 

**Authors**: Junqiao Wang, Zhaoyang Guan, Guanyu Liu, Tianze Xia, Xianzhi Li, Shuo Yin, Xinyuan Song, Chuhan Cheng, Tianyu Shi, Alex Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09751)  

**Abstract**: Predicting cryptocurrency returns is notoriously difficult: price movements are driven by a fast-shifting blend of on-chain activity, news flow, and social sentiment, while labeled training data are scarce and expensive. In this paper, we present Meta-RL-Crypto, a unified transformer-based architecture that unifies meta-learning and reinforcement learning (RL) to create a fully self-improving trading agent. Starting from a vanilla instruction-tuned LLM, the agent iteratively alternates between three roles-actor, judge, and meta-judge-in a closed-loop architecture. This learning process requires no additional human supervision. It can leverage multimodal market inputs and internal preference feedback. The agent in the system continuously refines both the trading policy and evaluation criteria. Experiments across diverse market regimes demonstrate that Meta-RL-Crypto shows good performance on the technical indicators of the real market and outperforming other LLM-based baselines. 

**Abstract (ZH)**: 预测加密货币回报历来具有挑战性：价格变动由快速变化的链上活动、新闻流和社会情绪混合驱动，而标训练数据稀缺且昂贵。本文提出了一种名为Meta-RL-Crypto的统一变压器架构，将元学习和强化学习（RL）结合在一起，创建一个完全自主改进的交易代理。该代理从一个普通的指令调优的大语言模型开始，交替扮演行动者、裁判和元裁判的角色，在一个闭环架构中运行。这一学习过程无需额外的人类监督，可以利用多模态市场输入和内部偏好反馈。系统中的代理持续优化交易策略和评价标准。实验结果表明，Meta-RL-Crypto在多种市场状态下表现出良好的实际市场技术指标，并优于其他基于大语言模型的基线方法。 

---
# A Co-Training Semi-Supervised Framework Using Faster R-CNN and YOLO Networks for Object Detection in Densely Packed Retail Images 

**Title (ZH)**: 基于Faster R-CNN和YOLO网络的co-training半监督框架在密集零售图像目标检测中的应用 

**Authors**: Hossein Yazdanjouei, Arash Mansouri, Mohammad Shokouhifar  

**Link**: [PDF](https://arxiv.org/pdf/2509.09750)  

**Abstract**: This study proposes a semi-supervised co-training framework for object detection in densely packed retail environments, where limited labeled data and complex conditions pose major challenges. The framework combines Faster R-CNN (utilizing a ResNet backbone) for precise localization with YOLO (employing a Darknet backbone) for global context, enabling mutual pseudo-label exchange that improves accuracy in scenes with occlusion and overlapping objects. To strengthen classification, it employs an ensemble of XGBoost, Random Forest, and SVM, utilizing diverse feature representations for higher robustness. Hyperparameters are optimized using a metaheuristic-driven algorithm, enhancing precision and efficiency across models. By minimizing reliance on manual labeling, the approach reduces annotation costs and adapts effectively to frequent product and layout changes common in retail. Experiments on the SKU-110k dataset demonstrate strong performance, highlighting the scalability and practicality of the proposed framework for real-world retail applications such as automated inventory tracking, product monitoring, and checkout systems. 

**Abstract (ZH)**: 本研究提出了一种半监督协同训练框架，用于密集零售环境中物体检测，该框架在有限标注数据和复杂条件下面临重大挑战。该框架结合使用ResNet主干的Faster R-CNN进行精确定位，和使用Darknet主干的YOLO捕捉全局上下文，实现伪标签的相互交换，提高遮挡和重叠物体场景的准确性。为增强分类，采用XGBoost、随机森林和SVM的集成方法，利用多种特征表示提高鲁棒性。通过元启发式算法优化超参数，增强模型的精确性和效率。通过减少对人工标注的依赖，该方法降低了标注成本，并能够有效适应零售环境中常见的产品和布局频繁变化。实验在SKU-110k数据集上展示了强大性能，突显了所提框架在自动库存跟踪、产品监控和结账系统等真实零售应用中的可扩展性和实用性。 

---
# D-CAT: Decoupled Cross-Attention Transfer between Sensor Modalities for Unimodal Inference 

**Title (ZH)**: D-CAT：传感器模态间的解耦跨注意力转移用于单模推断 

**Authors**: Leen Daher, Zhaobo Wang, Malcolm Mielle  

**Link**: [PDF](https://arxiv.org/pdf/2509.09747)  

**Abstract**: Cross-modal transfer learning is used to improve multi-modal classification models (e.g., for human activity recognition in human-robot collaboration). However, existing methods require paired sensor data at both training and inference, limiting deployment in resource-constrained environments where full sensor suites are not economically and technically usable. To address this, we propose Decoupled Cross-Attention Transfer (D-CAT), a framework that aligns modality-specific representations without requiring joint sensor modality during inference. Our approach combines a self-attention module for feature extraction with a novel cross-attention alignment loss, which enforces the alignment of sensors' feature spaces without requiring the coupling of the classification pipelines of both modalities. We evaluate D-CAT on three multi-modal human activity datasets (IMU, video, and audio) under both in-distribution and out-of-distribution scenarios, comparing against uni-modal models. Results show that in in-distribution scenarios, transferring from high-performing modalities (e.g., video to IMU) yields up to 10% F1-score gains over uni-modal training. In out-of-distribution scenarios, even weaker source modalities (e.g., IMU to video) improve target performance, as long as the target model isn't overfitted on the training data. By enabling single-sensor inference with cross-modal knowledge, D-CAT reduces hardware redundancy for perception systems while maintaining accuracy, which is critical for cost-sensitive or adaptive deployments (e.g., assistive robots in homes with variable sensor availability). Code is available at this https URL. 

**Abstract (ZH)**: 跨模态脱耦转移学习用于提高多模态分类模型（例如，在人机协作中的人类活动识别） 

---
# Structure Matters: Brain Graph Augmentation via Learnable Edge Masking for Data-efficient Psychiatric Diagnosis 

**Title (ZH)**: 结构 Matters：基于可学习边掩蔽的脑图增广方法在数据高效的精神疾病诊断中的应用 

**Authors**: Mujie Liu, Chenze Wang, Liping Chen, Nguyen Linh Dan Le, Niharika Tewari, Ting Dang, Jiangang Ma, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.09744)  

**Abstract**: The limited availability of labeled brain network data makes it challenging to achieve accurate and interpretable psychiatric diagnoses. While self-supervised learning (SSL) offers a promising solution, existing methods often rely on augmentation strategies that can disrupt crucial structural semantics in brain graphs. To address this, we propose SAM-BG, a two-stage framework for learning brain graph representations with structural semantic preservation. In the pre-training stage, an edge masker is trained on a small labeled subset to capture key structural semantics. In the SSL stage, the extracted structural priors guide a structure-aware augmentation process, enabling the model to learn more semantically meaningful and robust representations. Experiments on two real-world psychiatric datasets demonstrate that SAM-BG outperforms state-of-the-art methods, particularly in small-labeled data settings, and uncovers clinically relevant connectivity patterns that enhance interpretability. Our code is available at this https URL. 

**Abstract (ZH)**: 有限的标记脑网络数据使得实现准确可解释的心理精神疾病诊断具有挑战性。虽然自监督学习（SSL）提供了一种有 promise 的解决方案，但现有方法往往依赖可能破坏脑图关键结构语义的增强策略。为了解决这一问题，我们提出了一种名为 SAM-BG 的两阶段框架，用于在结构语义保留下学习脑图表示。在预训练阶段，边掩码器在一小部分标记数据上训练以捕获关键的结构语义。在 SSL 阶段，提取的结构先验指导结构感知的增强过程，使模型能够学习更多语义上有意义且健壮的表示。实验结果表明，SAM-BG 在小标记数据集上优于现有方法，并揭示出有助于增强可解释性的临床相关连通性模式。我们的代码可在以下链接获取：this https URL。 

---
# HypoGeneAgent: A Hypothesis Language Agent for Gene-Set Cluster Resolution Selection Using Perturb-seq Datasets 

**Title (ZH)**: HypoGeneAgent：一种基于扰动-seq数据集的基因集簇分辨率选择假设语言代理 

**Authors**: Ying Yuan, Xing-Yue Monica Ge, Aaron Archer Waterman, Tommaso Biancalani, David Richmond, Yogesh Pandit, Avtar Singh, Russell Littman, Jin Liu, Jan-Christian Huetter, Vladimir Ermakov  

**Link**: [PDF](https://arxiv.org/pdf/2509.09740)  

**Abstract**: Large-scale single-cell and Perturb-seq investigations routinely involve clustering cells and subsequently annotating each cluster with Gene-Ontology (GO) terms to elucidate the underlying biological programs. However, both stages, resolution selection and functional annotation, are inherently subjective, relying on heuristics and expert curation. We present HYPOGENEAGENT, a large language model (LLM)-driven framework, transforming cluster annotation into a quantitatively optimizable task. Initially, an LLM functioning as a gene-set analyst analyzes the content of each gene program or perturbation module and generates a ranked list of GO-based hypotheses, accompanied by calibrated confidence scores. Subsequently, we embed every predicted description with a sentence-embedding model, compute pair-wise cosine similarities, and let the agent referee panel score (i) the internal consistency of the predictions, high average similarity within the same cluster, termed intra-cluster agreement (ii) their external distinctiveness, low similarity between clusters, termed inter-cluster separation. These two quantities are combined to produce an agent-derived resolution score, which is maximized when clusters exhibit simultaneous coherence and mutual exclusivity. When applied to a public K562 CRISPRi Perturb-seq dataset as a preliminary test, our Resolution Score selects clustering granularities that exhibit alignment with known pathway compared to classical metrics such silhouette score, modularity score for gene functional enrichment summary. These findings establish LLM agents as objective adjudicators of cluster resolution and functional annotation, thereby paving the way for fully automated, context-aware interpretation pipelines in single-cell multi-omics studies. 

**Abstract (ZH)**: 基于大型语言模型的HYPOGENEAGENT框架：客观化单细胞和Perturb-seq聚类解析与功能注释 

---
# World Modeling with Probabilistic Structure Integration 

**Title (ZH)**: 基于概率结构集成的世界建模 

**Authors**: Klemen Kotar, Wanhee Lee, Rahul Venkatesh, Honglin Chen, Daniel Bear, Jared Watrous, Simon Kim, Khai Loong Aw, Lilian Naing Chen, Stefan Stojanov, Kevin Feigelis, Imran Thobani, Alex Durango, Khaled Jedoui, Atlas Kazemian, Dan Yamins  

**Link**: [PDF](https://arxiv.org/pdf/2509.09737)  

**Abstract**: We present Probabilistic Structure Integration (PSI), a system for learning richly controllable and flexibly promptable world models from data. PSI consists of a three-step cycle. The first step, Probabilistic prediction, involves building a probabilistic graphical model Psi of the data, in the form of a random-access autoregressive sequence model. Psi supports a complete set of learned conditional distributions describing the dependence of any variables in the data on any other set of variables. In step 2, Structure extraction, we show how to extract underlying low-dimensional properties in the data, corresponding to a diverse set of meaningful "intermediate structures", in a zero-shot fashion via causal inference on Psi. Step 3, Integration, completes the cycle by converting these structures into new token types that are then continually mixed back into the training diet as conditioning signals and prediction targets. Each such cycle augments the capabilities of Psi, both allowing it to model the underlying data better, and creating new control handles -- akin to an LLM-like universal prompting language. We train an instance of Psi on 1.4 trillion tokens of internet video data; we use it to perform a variety of useful video prediction and understanding inferences; we extract state-of-the-art optical flow, self-supervised depth and object segmentation; and we use these structures to support a full cycle of predictive improvements. 

**Abstract (ZH)**: 概率结构集成（PSI）：从数据中学习丰富可控和灵活可调的世界模型系统 

---
# MCP-AgentBench: Evaluating Real-World Language Agent Performance with MCP-Mediated Tools 

**Title (ZH)**: MCP-AgentBench: 通过MCP中介工具评估真实世界语言代理性能 

**Authors**: Zikang Guo, Benfeng Xu, Chiwei Zhu, Wentao Hong, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2509.09734)  

**Abstract**: The Model Context Protocol (MCP) is rapidly emerging as a pivotal open standard, designed to enhance agent-tool integration and interoperability, and is positioned to unlock a new era of powerful, interconnected, and genuinely utilitarian agentic AI. However, despite MCP's growing adoption, existing benchmarks often fail to capture real-world agent performance within this new paradigm, leading to a distorted perception of their true operational value and an inability to reliably differentiate proficiencies. To bridge this critical evaluation gap, we introduce MCP-AgentBench -- a comprehensive benchmark specifically engineered to rigorously assess language agent capabilities in MCP-mediated tool interactions. Core contributions of MCP-AgentBench include: the establishment of a robust MCP testbed comprising 33 operational servers with 188 distinct tools; the development of a benchmark featuring 600 systematically designed queries distributed across 6 distinct categories of varying interaction complexity; and the introduction of MCP-Eval, a novel outcome-oriented evaluation methodology prioritizing real-world task success. Through extensive empirical evaluation of leading language agents, we provide foundational insights. MCP-AgentBench aims to equip the research community with a standardized and reliable framework to build, validate, and advance agents capable of fully leveraging MCP's transformative benefits, thereby accelerating progress toward truly capable and interoperable AI systems. 

**Abstract (ZH)**: MCP-AgentBench：一种专门用于评估MCP中介语言代理能力的综合基准 

---
# MITS: A Large-Scale Multimodal Benchmark Dataset for Intelligent Traffic Surveillance 

**Title (ZH)**: MITS：智能交通监控的大规模多模态基准数据集 

**Authors**: Kaikai Zhao, Zhaoxiang Liu, Peng Wang, Xin Wang, Zhicheng Ma, Yajun Xu, Wenjing Zhang, Yibing Nan, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2509.09730)  

**Abstract**: General-domain large multimodal models (LMMs) have achieved significant advances in various image-text tasks. However, their performance in the Intelligent Traffic Surveillance (ITS) domain remains limited due to the absence of dedicated multimodal datasets. To address this gap, we introduce MITS (Multimodal Intelligent Traffic Surveillance), the first large-scale multimodal benchmark dataset specifically designed for ITS. MITS includes 170,400 independently collected real-world ITS images sourced from traffic surveillance cameras, annotated with eight main categories and 24 subcategories of ITS-specific objects and events under diverse environmental conditions. Additionally, through a systematic data generation pipeline, we generate high-quality image captions and 5 million instruction-following visual question-answer pairs, addressing five critical ITS tasks: object and event recognition, object counting, object localization, background analysis, and event reasoning. To demonstrate MITS's effectiveness, we fine-tune mainstream LMMs on this dataset, enabling the development of ITS-specific applications. Experimental results show that MITS significantly improves LMM performance in ITS applications, increasing LLaVA-1.5's performance from 0.494 to 0.905 (+83.2%), LLaVA-1.6's from 0.678 to 0.921 (+35.8%), Qwen2-VL's from 0.584 to 0.926 (+58.6%), and Qwen2.5-VL's from 0.732 to 0.930 (+27.0%). We release the dataset, code, and models as open-source, providing high-value resources to advance both ITS and LMM research. 

**Abstract (ZH)**: 通用领域大型多模态模型在各种图像-文本任务中取得了显著进展。然而，它们在智能交通 surveillance (ITS) 领域的表现仍然受限，因为缺乏专门的多模态数据集。为了填补这一空白，我们引入了 MITS（多模态智能交通 surveillance），首个专门针对 ITS 设计的大规模多模态基准数据集。MITS 包含 170,400 张独立收集的实际 ITS 图像，来源于交通监控摄像头，并在多种环境条件下对八类主要类别和 24 个子类别的 ITS 特定对象和事件进行了标注。此外，通过系统化数据生成管道，我们生成了高质量的图像描述和 500 万条指令跟随的视觉问答对，用于解决五个关键的 ITS 任务：对象和事件识别、对象计数、对象定位、背景分析和事件推理。为了展示 MITS 的有效性，我们在该数据集上微调了主流的多模态模型，促进了面向 ITS 的特定应用开发。实验结果显示，MITS 显著提高了多模态模型在 ITS 应用中的性能，LLaVA-1.5 的性能从 0.494 提高到 0.905 (+83.2%)，LLaVA-1.6 的性能从 0.678 提高到 0.921 (+35.8%)，Qwen2-VL 的性能从 0.584 提高到 0.926 (+58.6%)，Qwen2.5-VL 的性能从 0.732 提高到 0.930 (+27.0%)。我们发布了数据集、代码和模型，作为开源资源，以促进 ITS 和多模态模型研究的进步。 

---
# MultimodalHugs: Enabling Sign Language Processing in Hugging Face 

**Title (ZH)**: 多模态拥抱：在 Hugging Face 平台上实现手语处理 

**Authors**: Gerard Sant, Zifan Jiang, Carlos Escolano, Amit Moryossef, Mathias Müller, Rico Sennrich, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2509.09729)  

**Abstract**: In recent years, sign language processing (SLP) has gained importance in the general field of Natural Language Processing. However, compared to research on spoken languages, SLP research is hindered by complex ad-hoc code, inadvertently leading to low reproducibility and unfair comparisons. Existing tools that are built for fast and reproducible experimentation, such as Hugging Face, are not flexible enough to seamlessly integrate sign language experiments. This view is confirmed by a survey we conducted among SLP researchers.
To address these challenges, we introduce MultimodalHugs, a framework built on top of Hugging Face that enables more diverse data modalities and tasks, while inheriting the well-known advantages of the Hugging Face ecosystem. Even though sign languages are our primary focus, MultimodalHugs adds a layer of abstraction that makes it more widely applicable to other use cases that do not fit one of the standard templates of Hugging Face. We provide quantitative experiments to illustrate how MultimodalHugs can accommodate diverse modalities such as pose estimation data for sign languages, or pixel data for text characters. 

**Abstract (ZH)**: 近年来，手语处理（SLP）在自然语言处理领域获得了重要地位。然而，与口头语言研究相比，SLP研究受限于复杂的定制代码，导致可重复性低和不公平的比较。现有的用于快速和可重复实验的工具，如Hugging Face，不足以无缝集成手语实验。我们对SLP研究人员进行的一项调查显示，这一观点得到了证实。为了应对这些挑战，我们引入了MultimodalHugs框架，该框架基于Hugging Face，并能够支持更广泛的数据模态和任务，同时继承了Hugging Face生态系统的显著优势。尽管手语是我们的主要研究对象，但MultimodalHugs通过增加一层抽象，使其更广泛适用于不符合Hugging Face标准模板的其他应用场景。我们提供了定量实验来说明MultimodalHugs如何容纳多样化的模态，如手语的姿势估计数据或文本字符的像素数据。 

---
# DiTTO-LLM: Framework for Discovering Topic-based Technology Opportunities via Large Language Model 

**Title (ZH)**: DiTTO-LLM：基于话题发现技术机会的框架 via 大型语言模型 

**Authors**: Wonyoung Kim, Sujeong Seo, Juhyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09724)  

**Abstract**: Technology opportunities are critical information that serve as a foundation for advancements in technology, industry, and innovation. This paper proposes a framework based on the temporal relationships between technologies to identify emerging technology opportunities. The proposed framework begins by extracting text from a patent dataset, followed by mapping text-based topics to discover inter-technology relationships. Technology opportunities are then identified by tracking changes in these topics over time. To enhance efficiency, the framework leverages a large language model to extract topics and employs a prompt for a chat-based language model to support the discovery of technology opportunities. The framework was evaluated using an artificial intelligence patent dataset provided by the United States Patent and Trademark Office. The experimental results suggest that artificial intelligence technology is evolving into forms that facilitate everyday accessibility. This approach demonstrates the potential of the proposed framework to identify future technology opportunities. 

**Abstract (ZH)**: 技术机会是促进技术、产业和创新前进的基础性信息。本文提出了一种基于技术创新时间关系的框架以识别新兴技术机会。该框架首先从专利数据集中提取文本，随后将基于文本的主题映射到发现跨技术关系，通过跟踪这些主题随时间的变化来识别技术机会。为了提高效率，该框架利用了大规模语言模型来提取主题，并使用一种面向聊天的语言模型提示以支持技术机会的发现。该框架使用美国专利商标局提供的人工智能专利数据集进行了评估，实验结果表明，人工智能技术正在发展成便于日常使用的形态。该方法展示了所提框架识别未来技术机会的潜力。 

---
# ALIGNS: Unlocking nomological networks in psychological measurement through a large language model 

**Title (ZH)**: ALIGNS: 通过大型语言模型解锁心理测量中的nomological网络 

**Authors**: Kai R. Larsen, Sen Yan, Roland Müller, Lan Sang, Mikko Rönkkö, Ravi Starzl, Donald Edmondson  

**Link**: [PDF](https://arxiv.org/pdf/2509.09723)  

**Abstract**: Psychological measurement is critical to many disciplines. Despite advances in measurement, building nomological networks, theoretical maps of how concepts and measures relate to establish validity, remains a challenge 70 years after Cronbach and Meehl proposed them as fundamental to validation. This limitation has practical consequences: clinical trials may fail to detect treatment effects, and public policy may target the wrong outcomes. We introduce Analysis of Latent Indicators to Generate Nomological Structures (ALIGNS), a large language model-based system trained with validated questionnaire measures. ALIGNS provides three comprehensive nomological networks containing over 550,000 indicators across psychology, medicine, social policy, and other fields. This represents the first application of large language models to solve a foundational problem in measurement validation. We report classification accuracy tests used to develop the model, as well as three evaluations. In the first evaluation, the widely used NIH PROMIS anxiety and depression instruments are shown to converge into a single dimension of emotional distress. The second evaluation examines child temperament measures and identifies four potential dimensions not captured by current frameworks, and questions one existing dimension. The third evaluation, an applicability check, engages expert psychometricians who assess the system's importance, accessibility, and suitability. ALIGNS is freely available at this http URL, complementing traditional validation methods with large-scale nomological analysis. 

**Abstract (ZH)**: 心理测量对于许多领域至关重要。尽管测量技术有了进步，但在Cronbach和Meehl提出构建诺姆ological网络70年后，建立反映概念和测量关系的理论地图以确立有效性的诺姆ological网络依然是一项挑战。这一局限性具有实际后果：临床试验可能无法检测到治疗效果，而公共政策可能针对错误的目标结果。我们引入了基于大型语言模型生成诺姆ological结构的隐性指标分析（ALIGNS）系统，该系统使用验证的问卷测量进行训练。ALIGNS提供了三个综合性的诺姆ological网络，包含超过550,000个指标，涵盖心理学、医学、社会政策及其他领域。这代表了首次将大型语言模型应用于解决测量验证中的基础问题。我们报告了用于开发模型的分类准确度测试及三项评估。在首次评估中，广泛使用的NIH PROMIS焦虑和抑郁量表合并为情感困扰的一个维度。第二次评估检查了儿童气质量表，并识别出四个当前框架未能捕获的潜在维度，质疑了一个现有维度。第三次评估是对适用性的检查，邀请了专家心理测量学家评估系统的重要性、可访问性和适用性。ALIGNS可在以下网址免费获得，它补充了传统的验证方法，提供了大规模的诺姆ological分析。 

---
# A Multimodal RAG Framework for Housing Damage Assessment: Collaborative Optimization of Image Encoding and Policy Vector Retrieval 

**Title (ZH)**: 多模态RAG框架在住房损害评估中的应用：图像编码与策略向量检索的协作优化 

**Authors**: Jiayi Miao, Dingxin Lu, Zhuqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09721)  

**Abstract**: After natural disasters, accurate evaluations of damage to housing are important for insurance claims response and planning of resources. In this work, we introduce a novel multimodal retrieval-augmented generation (MM-RAG) framework. On top of classical RAG architecture, we further the framework to devise a two-branch multimodal encoder structure that the image branch employs a visual encoder composed of ResNet and Transformer to extract the characteristic of building damage after disaster, and the text branch harnesses a BERT retriever for the text vectorization of posts as well as insurance policies and for the construction of a retrievable restoration index. To impose cross-modal semantic alignment, the model integrates a cross-modal interaction module to bridge the semantic representation between image and text via multi-head attention. Meanwhile, in the generation module, the introduced modal attention gating mechanism dynamically controls the role of visual evidence and text prior information during generation. The entire framework takes end-to-end training, and combines the comparison loss, the retrieval loss and the generation loss to form multi-task optimization objectives, and achieves image understanding and policy matching in collaborative learning. The results demonstrate superior performance in retrieval accuracy and classification index on damage severity, where the Top-1 retrieval accuracy has been improved by 9.6%. 

**Abstract (ZH)**: 灾害发生后，房屋损坏评估对于保险理赔响应和资源规划至关重要。本文引入一种新型多模态检索增强生成（MM-RAG）框架。在传统RAG架构基础上，进一步设计了一种两分支多模态编码结构，图像分支采用由ResNet和Transformer组成的视觉编码器提取灾害后建筑损坏特征，文本分支利用BERT检索器对帖子、保险政策进行文本向量化，并构建可检索的修复索引。为了实现跨模态语义对齐，模型整合了一种跨模态交互模块，通过多头注意力机制在图像和文本之间建立语义表示连接。同时，在生成模块中，引入的模态注意力门控机制动态控制生成过程中视觉证据和文本先验信息的角色。整个框架端到端训练，并将比较损失、检索损失和生成损失结合形成多任务优化目标，在协作学习中实现图像理解与政策匹配。实验结果表明，在损坏程度检索准确率和分类指标上表现出优异性能，Top-1检索准确率提高了9.6%。 

---
# VStyle: A Benchmark for Voice Style Adaptation with Spoken Instructions 

**Title (ZH)**: VStyle：基于口头指示的语音风格适应基准 

**Authors**: Jun Zhan, Mingyang Han, Yuxuan Xie, Chen Wang, Dong Zhang, Kexin Huang, Haoxiang Shi, DongXiao Wang, Tengtao Song, Qinyuan Cheng, Shimin Li, Jun Song, Xipeng Qiu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09716)  

**Abstract**: Spoken language models (SLMs) have emerged as a unified paradigm for speech understanding and generation, enabling natural human machine interaction. However, while most progress has focused on semantic accuracy and instruction following, the ability of SLMs to adapt their speaking style based on spoken instructions has received limited attention. We introduce Voice Style Adaptation (VSA), a new task that examines whether SLMs can modify their speaking style, such as timbre, prosody, or persona following natural language spoken commands. To study this task, we present VStyle, a bilingual (Chinese & English) benchmark covering four categories of speech generation: acoustic attributes, natural language instruction, role play, and implicit empathy. We also introduce the Large Audio Language Model as a Judge (LALM as a Judge) framework, which progressively evaluates outputs along textual faithfulness, style adherence, and naturalness, ensuring reproducible and objective assessment. Experiments on commercial systems and open source SLMs demonstrate that current models face clear limitations in controllable style adaptation, highlighting both the novelty and challenge of this task. By releasing VStyle and its evaluation toolkit, we aim to provide the community with a foundation for advancing human centered spoken interaction. The dataset and code are publicly available at \href{this https URL}{project's homepage}. 

**Abstract (ZH)**: 语音风格适应：一种新的说话风格适应任务 

---
# Investigating Symbolic Triggers of Hallucination in Gemma Models Across HaluEval and TruthfulQA 

**Title (ZH)**: 在HaluEval和TruthfulQA中探究Gemma模型幻觉的符号触发因素 

**Authors**: Naveen Lamba, Sanju Tiwari, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2509.09715)  

**Abstract**: Hallucination in Large Language Models (LLMs) is a well studied problem. However, the properties that make LLM intrinsically vulnerable to hallucinations have not been identified and studied. This research identifies and characterizes the key properties, allowing us to pinpoint vulnerabilities within the model's internal mechanisms. To solidify on these properties, we utilized two established datasets, HaluEval and TruthfulQA and convert their existing format of question answering into various other formats to narrow down these properties as the reason for the hallucinations. Our findings reveal that hallucination percentages across symbolic properties are notably high for Gemma-2-2B, averaging 79.0% across tasks and datasets. With increased model scale, hallucination drops to 73.6% for Gemma-2-9B and 63.9% for Gemma-2-27B, reflecting a 15 percentage point reduction overall. Although the hallucination rate decreases as the model size increases, a substantial amount of hallucination caused by symbolic properties still persists. This is especially evident for modifiers (ranging from 84.76% to 94.98%) and named entities (ranging from 83.87% to 93.96%) across all Gemma models and both datasets. These findings indicate that symbolic elements continue to confuse the models, pointing to a fundamental weakness in how these LLMs process such inputs--regardless of their scale. 

**Abstract (ZH)**: 大规模语言模型（LLMs）中的幻觉是一个研究充分的问题。然而，使其固有易受幻觉影响的关键属性尚未被识别和研究。本研究识别并描述了这些关键属性，使我们能够确定模型内部机制中的漏洞。为了进一步强调这些属性，我们使用了两个现有的数据集HaluEval和TruthfulQA，并将它们的问题回答格式转换为多种其他格式，以缩小这些属性作为幻觉原因的可能性。我们的研究结果表明，对于Gemma-2-2B，在符号属性下幻觉比例非常高，平均在各种任务和数据集中达到79.0%。随着模型规模的增加，Gemma-2-9B 的幻觉比例下降到73.6%，而Gemma-2-27B下降到63.9%，总体下降了15个百分点。尽管随着模型规模的增加，幻觉率有所下降，但由符号属性引起的很大一部分幻觉仍然存在。这在修饰词（从84.76%到94.98%）和命名实体（从83.87%到93.96%）方面尤其明显，这些标志贯穿于所有Gemma模型和两个数据集中。这些发现表明，符号元素继续困扰模型，揭示了这些LLMs处理此类输入的基本弱点，无论其规模大小。 

---
# How Small Transformation Expose the Weakness of Semantic Similarity Measures 

**Title (ZH)**: 小变换揭示语义相似度度量的弱点 

**Authors**: Serge Lionel Nikiema, Albérick Euraste Djire, Abdoul Aziz Bonkoungou, Micheline Bénédicte Moumoula, Jordan Samhi, Abdoul Kader Kabore, Jacques Klein, Tegawendé F. Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2509.09714)  

**Abstract**: This research examines how well different methods measure semantic similarity, which is important for various software engineering applications such as code search, API recommendations, automated code reviews, and refactoring tools. While large language models are increasingly used for these similarity assessments, questions remain about whether they truly understand semantic relationships or merely recognize surface patterns.
The study tested 18 different similarity measurement approaches, including word-based methods, embedding techniques, LLM-based systems, and structure-aware algorithms. The researchers created a systematic testing framework that applies controlled changes to text and code to evaluate how well each method handles different types of semantic relationships.
The results revealed significant issues with commonly used metrics. Some embedding-based methods incorrectly identified semantic opposites as similar up to 99.9 percent of the time, while certain transformer-based approaches occasionally rated opposite meanings as more similar than synonymous ones. The study found that embedding methods' poor performance often stemmed from how they calculate distances; switching from Euclidean distance to cosine similarity improved results by 24 to 66 percent. LLM-based approaches performed better at distinguishing semantic differences, producing low similarity scores (0.00 to 0.29) for genuinely different meanings, compared to embedding methods that incorrectly assigned high scores (0.82 to 0.99) to dissimilar content. 

**Abstract (ZH)**: 本研究探讨了不同方法在测量语义相似性方面的有效性，这对于代码搜索、API推荐、自动化代码审查和重构工具等软件工程应用至关重要。尽管大型语言模型在这些相似性评估中越来越被使用，但仍有疑问，即它们是否真正理解了语义关系，还是仅仅识别了表面模式。 

---
# HANRAG: Heuristic Accurate Noise-resistant Retrieval-Augmented Generation for Multi-hop Question Answering 

**Title (ZH)**: HANRAG:启发式准确的抗噪声检索增强生成多跳问答 

**Authors**: Duolin Sun, Dan Yang, Yue Shen, Yihan Jiao, Zhehao Tan, Jie Feng, Lianzhen Zhong, Jian Wang, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09713)  

**Abstract**: The Retrieval-Augmented Generation (RAG) approach enhances question-answering systems and dialogue generation tasks by integrating information retrieval (IR) technologies with large language models (LLMs). This strategy, which retrieves information from external knowledge bases to bolster the response capabilities of generative models, has achieved certain successes. However, current RAG methods still face numerous challenges when dealing with multi-hop queries. For instance, some approaches overly rely on iterative retrieval, wasting too many retrieval steps on compound queries. Additionally, using the original complex query for retrieval may fail to capture content relevant to specific sub-queries, resulting in noisy retrieved content. If the noise is not managed, it can lead to the problem of noise accumulation. To address these issues, we introduce HANRAG, a novel heuristic-based framework designed to efficiently tackle problems of varying complexity. Driven by a powerful revelator, HANRAG routes queries, decomposes them into sub-queries, and filters noise from retrieved documents. This enhances the system's adaptability and noise resistance, making it highly capable of handling diverse queries. We compare the proposed framework against other leading industry methods across various benchmarks. The results demonstrate that our framework obtains superior performance in both single-hop and multi-hop question-answering tasks. 

**Abstract (ZH)**: 基于检索增强生成的多跳查询处理方法：HANRAG框架 

---
# The Thinking Therapist: Training Large Language Models to Deliver Acceptance and Commitment Therapy using Supervised Fine-Tuning and Odds Ratio Policy Optimization 

**Title (ZH)**: 思辨治疗师：通过监督微调和 odds ratio 策略优化训练大型语言模型提供接受与承诺疗法 

**Authors**: Talha Tahir  

**Link**: [PDF](https://arxiv.org/pdf/2509.09712)  

**Abstract**: Acceptance and Commitment Therapy (ACT) is a third-wave cognitive behavioral therapy with emerging evidence of efficacy in several psychiatric conditions. This study investigates the impact of post-training methodology and explicit reasoning on the ability of a small open-weight large language model (LLM) to deliver ACT. Using 50 sets of synthetic ACT transcripts generated by Mistral-Large, we trained Llama-3.2-3b-Instruct with two distinct approaches, supervised fine-tuning (SFT) and odds ratio policy optimization (ORPO), each with and without an explicit chain-of-thought (COT) reasoning step. Performance was evaluated by comparing these four post-trained variants against the base Instruct model. These models were benchmarked in simulated therapy sessions, with performance quantitatively assessed on the ACT Fidelity Measure (ACT-FM) and the Therapist Empathy Scale (TES) by an LLM judge that had been fine-tuned on human evaluations. Our findings demonstrate that the ORPO-trained models significantly outperformed both their SFT and Instruct counterparts on ACT fidelity ($\chi^2(5) = 185.15, p < .001$) and therapeutic empathy ($\chi^2(5) = 140.37, p < .001$). The effect of COT was conditional as it provided a significant benefit to SFT models, improving ACT-FM scores by an average of 2.68 points ($p < .001$), while offering no discernible advantage to the superior ORPO or instruct-tuned variants. We posit that the superiority of ORPO stems from its ability to learn the therapeutic `process' over imitating `content,' a key aspect of ACT, while COT acts as a necessary scaffold for models trained only via imitation. This study establishes that preference-aligned policy optimization can effectively instill ACT competencies in small LLMs, and that the utility of explicit reasoning is highly dependent on the underlying training paradigm. 

**Abstract (ZH)**: 接纳与承诺疗法（ACT）的应用及其在小开放权重大型语言模型中的效果研究：后训练方法与显式推理的影响 

---
# Psychiatry-Bench: A Multi-Task Benchmark for LLMs in Psychiatry 

**Title (ZH)**: 精神医学-床旁：精神医学领域LLM的多任务基准 

**Authors**: Aya E. Fouda, Abdelrahamn A. Hassan, Radwa J. Hanafy, Mohammed E. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2509.09711)  

**Abstract**: Large language models (LLMs) hold great promise in enhancing psychiatric practice, from improving diagnostic accuracy to streamlining clinical documentation and therapeutic support. However, existing evaluation resources heavily rely on small clinical interview corpora, social media posts, or synthetic dialogues, which limits their clinical validity and fails to capture the full complexity of psychiatric reasoning. In this work, we introduce PsychiatryBench, a rigorously curated benchmark grounded exclusively in authoritative, expert-validated psychiatric textbooks and casebooks. PsychiatryBench comprises eleven distinct question-answering tasks ranging from diagnostic reasoning and treatment planning to longitudinal follow-up, management planning, clinical approach, sequential case analysis, and multiple-choice/extended matching formats totaling over 5,300 expert-annotated items. We evaluate a diverse set of frontier LLMs (including Google Gemini, DeepSeek, LLaMA 3, and QWQ-32) alongside leading open-source medical models (e.g., OpenBiloLLM, MedGemma) using both conventional metrics and an "LLM-as-judge" similarity scoring framework. Our results reveal substantial gaps in clinical consistency and safety, particularly in multi-turn follow-up and management tasks, underscoring the need for specialized model tuning and more robust evaluation paradigms. PsychiatryBench offers a modular, extensible platform for benchmarking and improving LLM performance in high-stakes mental health applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在提升精神卫生实践方面展现出巨大潜力，从提高诊断准确性到简化临床记录和提供治疗方法支持。然而，现有的评估资源主要依赖于小型临床访谈语料库、社交媒体帖子或合成对话，这限制了它们的临床有效性，并未能捕捉到精神科推理的全部复杂性。本文介绍了PsychiatryBench，这是一个基于权威且经专家验证的精神疾病教科书和案例集精心整理的标准基准。PsychiatryBench 包含十一项不同的问答任务，涵盖了诊断推理、治疗规划、纵向随访、管理规划、临床方法、序贯病例分析以及超过5,300个专家标注的多项选择/扩展匹配格式题目。我们使用传统的评估指标和“模型作为裁判”的相似度评分框架，评估了一系列前沿的大规模语言模型（包括Google Gemini、DeepSeek、LLaMA 3和QWQ-32），以及领先的开源医疗模型（如OpenBiloLLM、MedGemma）。我们的结果显示，在多轮随访和管理任务中存在显著的临床一致性和安全性差距，强调了需要专门的模型调整和更 robust 的评估范式的必要性。PsychiatryBench 提供了一个模块化、可扩展的平台，用于在高风险的精神健康应用中评估和提升大规模语言模型的表现。 

---
# Generating Individual Travel Diaries Using Large Language Models Informed by Census and Land-Use Data 

**Title (ZH)**: 基于普查和用地数据引导的大语言模型生成个体旅行日志 

**Authors**: Sepehr Golrokh Amin, Devin Rhoads, Fatemeh Fakhrmoosavi, Nicholas E. Lownes, John N. Ivan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09710)  

**Abstract**: This study introduces a Large Language Model (LLM) scheme for generating individual travel diaries in agent-based transportation models. While traditional approaches rely on large quantities of proprietary household travel surveys, the method presented in this study generates personas stochastically from open-source American Community Survey (ACS) and Smart Location Database (SLD) data, then synthesizes diaries through direct prompting. This study features a novel one-to-cohort realism score: a composite of four metrics (Trip Count Score, Interval Score, Purpose Score, and Mode Score) validated against the Connecticut Statewide Transportation Study (CSTS) diaries, matched across demographic variables. The validation utilizes Jensen-Shannon Divergence to measure distributional similarities between generated and real diaries. When compared to diaries generated with classical methods (Negative Binomial for trip generation; Multinomial Logit for mode/purpose) calibrated on the validation set, LLM-generated diaries achieve comparable overall realism (LLM mean: 0.485 vs. 0.455). The LLM excels in determining trip purpose and demonstrates greater consistency (narrower realism score distribution), while classical models lead in numerical estimates of trip count and activity duration. Aggregate validation confirms the LLM's statistical representativeness (LLM mean: 0.612 vs. 0.435), demonstrating LLM's zero-shot viability and establishing a quantifiable metric of diary realism for future synthetic diary evaluation systems. 

**Abstract (ZH)**: 本研究介绍了一种大规模语言模型（LLM）方案，用于生成基于代理的交通模型中的个体旅行日记。传统方法依赖大量私有家庭旅行调查数据，而本研究提出的方法则通过直接提示，从开源的美国社区调查（ACS）和智能位置数据库（SLD）数据中随机生成人物，进而合成旅行日记。本研究特色在于提出了一种新颖的一对群体现实度评分：由四个指标（行程数量评分、时间间隔评分、目的评分和出行方式评分）组成，这些指标针对与康涅狄格州州级交通研究（CSTS）日记匹配的多个人口统计变量进行了验证。验证利用詹森-沙恩 divergence 测量生成日记与真实日记的分布相似性。与使用经典方法（负二项分布用于行程生成；多项逻辑回归用于出行方式/目的）在验证集上校准生成的旅行日记相比，LLM生成的旅行日记在整体现实度方面表现出相当的水平（LLM平均值：0.485 vs. 0.455）。LLM在确定行程目的方面表现尤为出色，并表现出更高的一致性（现实度评分分布较窄），而经典模型在行程数量和活动持续时间的数值估计方面占优。总体验证结果表明，LLM在统计代表性方面表现出色（LLM平均值：0.612 vs. 0.435），证明了LLM的零样本可行性，并为未来合成日记评估系统提供了可量化的现实度指标。 

---
# Assisting Research Proposal Writing with Large Language Models: Evaluation and Refinement 

**Title (ZH)**: 利用大型语言模型辅助研究提案写作：评估与优化 

**Authors**: Jing Ren, Weiqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09709)  

**Abstract**: Large language models (LLMs) like ChatGPT are increasingly used in academic writing, yet issues such as incorrect or fabricated references raise ethical concerns. Moreover, current content quality evaluations often rely on subjective human judgment, which is labor-intensive and lacks objectivity, potentially compromising the consistency and reliability. In this study, to provide a quantitative evaluation and enhance research proposal writing capabilities of LLMs, we propose two key evaluation metrics--content quality and reference validity--and an iterative prompting method based on the scores derived from these two metrics. Our extensive experiments show that the proposed metrics provide an objective, quantitative framework for assessing ChatGPT's writing performance. Additionally, iterative prompting significantly enhances content quality while reducing reference inaccuracies and fabrications, addressing critical ethical challenges in academic contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）如ChatGPT在学术写作中的应用日益增加，但不准确或伪造的参考文献等问题引发了伦理上的关切。目前的内容质量评估往往依赖于主观的人类判断，这既劳动密集又缺乏客观性，可能影响评估的一致性和可靠性。为此，本研究旨在通过提供定量评价并增强LLMs的研究设计写作能力，提出了两个关键评估指标——内容质量和参考有效性，并基于这两个指标的评分提出了迭代提示方法。大量实验表明，提出的指标为评估ChatGPT的写作性能提供了一个客观的定量框架。此外，迭代提示方法显著提高了内容质量，减少了参考文献的不准确性和伪造现象，解决了学术情境下重要的伦理挑战。 

---
# Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal 

**Title (ZH)**: Beyond I'm Sorry, I Can't: 分析大规模语言模型的拒绝行为 

**Authors**: Nirmalendu Prakash, Yeo Wei Jie, Amir Abdullah, Ranjan Satapathy, Erik Cambria, Roy Ka Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09708)  

**Abstract**: Refusal on harmful prompts is a key safety behaviour in instruction-tuned large language models (LLMs), yet the internal causes of this behaviour remain poorly understood. We study two public instruction-tuned models, Gemma-2-2B-IT and LLaMA-3.1-8B-IT, using sparse autoencoders (SAEs) trained on residual-stream activations. Given a harmful prompt, we search the SAE latent space for feature sets whose ablation flips the model from refusal to compliance, demonstrating causal influence and creating a jailbreak. Our search proceeds in three stages: (1) Refusal Direction: find a refusal-mediating direction and collect SAE features near that direction; (2) Greedy Filtering: prune to a minimal set; and (3) Interaction Discovery: fit a factorization machine (FM) that captures nonlinear interactions among the remaining active features and the minimal set. This pipeline yields a broad set of jailbreak-critical features, offering insight into the mechanistic basis of refusal. Moreover, we find evidence of redundant features that remain dormant unless earlier features are suppressed. Our findings highlight the potential for fine-grained auditing and targeted intervention in safety behaviours by manipulating the interpretable latent space. 

**Abstract (ZH)**: 有害提示的拒绝是一种关键的安全行为，存在于指令调优的大语言模型中，但其内部原因仍不甚明了。我们使用基于残差流激活训练的稀疏自编码器（SAEs），研究了两个公开的指令调优模型Gemma-2-2B-IT和LLaMA-3.1-8B-IT。给定一个有害提示，我们搜索SAE的潜在空间，找到能够使模型从拒绝转变为合规的功能集合，以此证明因果影响并创建一个逃逸攻击。我们的搜索分为三个阶段：（1）拒绝方向：找到一个拒绝中介方向并收集该方向附近的SAE特征；（2）贪婪筛选：精简为最小集；（3）交互发现：拟合一个因素分解机（FM），以捕获剩余活跃特征与最小集之间的非线性交互。该流程生成了一组关键的逃逸攻击特征，为理解拒绝机制提供了见解。此外，我们发现了冗余特征，除非较早的功能被抑制，否则这些特征将保持沉默。我们的发现突显了通过操纵解析的潜在空间进行细粒度审核和针对干预以改进安全行为的潜力。 

---
# LLM-Based Instance-Driven Heuristic Bias In the Context of a Biased Random Key Genetic Algorithm 

**Title (ZH)**: 基于LLM的实例驱动启发式偏差在偏倚随机键遗传算法中的应用 

**Authors**: Camilo Chacón Sartori, Martín Isla Pino, Pedro Pinacho-Davidson, Christian Blum  

**Link**: [PDF](https://arxiv.org/pdf/2509.09707)  

**Abstract**: Integrating Large Language Models (LLMs) within metaheuristics opens a novel path for solving complex combinatorial optimization problems. While most existing approaches leverage LLMs for code generation to create or refine specific heuristics, they often overlook the structural properties of individual problem instances. In this work, we introduce a novel framework that integrates LLMs with a Biased Random-Key Genetic Algorithm (BRKGA) to solve the NP-hard Longest Run Subsequence problem. Our approach extends the instance-driven heuristic bias paradigm by introducing a human-LLM collaborative process to co-design and implement a set of computationally efficient metrics. The LLM analyzes these instance-specific metrics to generate a tailored heuristic bias, which steers the BRKGA toward promising areas of the search space. We conduct a comprehensive experimental evaluation, including rigorous statistical tests, convergence and behavioral analyses, and targeted ablation studies, comparing our method against a standard BRKGA baseline across 1,050 generated instances of varying complexity. Results show that our top-performing hybrid, BRKGA+Llama-4-Maverick, achieves statistically significant improvements over the baseline, particularly on the most complex instances. Our findings confirm that leveraging an LLM to produce an a priori, instance-driven heuristic bias is a valuable approach for enhancing metaheuristics in complex optimization domains. 

**Abstract (ZH)**: 将大型语言模型（LLMs）整合到元启发式算法中为解决复杂组合优化问题开辟了一条新路径。尽管大多数现有方法通过代码生成利用LLMs来创建或完善特定启发式算法，但它们往往忽视了单个问题实例的结构特性。在本文中，我们介绍了一种新颖的框架，将LLMs与带偏见的随机键遗传算法（BRKGA）结合起来，用于解决NP难的最长运行子序列问题。我们的方法扩展了实例驱动启发式偏差范式，通过引入人类-LLMs协作过程来共同设计和实施一套计算高效的度量标准。LLMs分析这些实例特定的度量标准以生成定制的启发式偏差，引导BRKGA向搜索空间中的有希望区域移动。我们进行了全面的实验评估，包括严格的统计检验、收敛性和行为分析以及针对复杂度不同的1,050个生成实例的目标消融研究，将我们的方法与标准BRKGA基线进行比较。结果显示，我们的性能最佳的混合算法BRKGA+Llama-4-Maverick在复杂实例上相较于基线实现了统计显著的改进。我们的研究结果表明，利用LLMs生成先验的实例驱动启发式偏差是增强复杂优化域中元启发式算法的有效方法。 

---
# Differential Robustness in Transformer Language Models: Empirical Evaluation Under Adversarial Text Attacks 

**Title (ZH)**: Transformer语言模型的差异鲁棒性：对抗文本攻击下的实证评估 

**Authors**: Taniya Gidatkar, Oluwaseun Ajao, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2509.09706)  

**Abstract**: This study evaluates the resilience of large language models (LLMs) against adversarial attacks, specifically focusing on Flan-T5, BERT, and RoBERTa-Base. Using systematically designed adversarial tests through TextFooler and BERTAttack, we found significant variations in model robustness. RoBERTa-Base and FlanT5 demonstrated remarkable resilience, maintaining accuracy even when subjected to sophisticated attacks, with attack success rates of 0%. In contrast. BERT-Base showed considerable vulnerability, with TextFooler achieving a 93.75% success rate in reducing model accuracy from 48% to just 3%. Our research reveals that while certain LLMs have developed effective defensive mechanisms, these safeguards often require substantial computational resources. This study contributes to the understanding of LLM security by identifying existing strengths and weaknesses in current safeguarding approaches and proposes practical recommendations for developing more efficient and effective defensive strategies. 

**Abstract (ZH)**: 本研究评估了大型语言模型（LLMs）在对抗攻击下的韧性，重点关注Flan-T5、BERT和RoBERTa-Base。通过使用TextFooler和BERTAttack进行系统设计的对抗性测试，我们发现模型的稳健性存在显著差异。RoBERTa-Base和FlanT5表现出显著的韧性，在遭受复杂攻击时仍能保持准确率，攻击成功率均为0%。相比之下，BERT-Base显示出明显的脆弱性，TextFooler将模型准确率从48%降至3%，成功率达到93.75%。本研究揭示了某些LLM已经开发出有效的防御机制，但这些保护措施往往需要大量计算资源。本研究通过识别当前保护方法中已有的强项和弱点，为开发更为高效和有效的防御策略提供实践建议。 

---
# The Non-Determinism of Small LLMs: Evidence of Low Answer Consistency in Repetition Trials of Standard Multiple-Choice Benchmarks 

**Title (ZH)**: 小语言模型的非确定性：标准多项选择基准重复试验中答案一致性低的证据 

**Authors**: Claudio Pinhanez, Paulo Cavalin, Cassia Sanctos, Marcelo Grave, Yago Primerano  

**Link**: [PDF](https://arxiv.org/pdf/2509.09705)  

**Abstract**: This work explores the consistency of small LLMs (2B-8B parameters) in answering multiple times the same question. We present a study on known, open-source LLMs responding to 10 repetitions of questions from the multiple-choice benchmarks MMLU-Redux and MedQA, considering different inference temperatures, small vs. medium models (50B-80B), finetuned vs. base models, and other parameters. We also look into the effects of requiring multi-trial answer consistency on accuracy and the trade-offs involved in deciding which model best provides both of them. To support those studies, we propose some new analytical and graphical tools. Results show that the number of questions which can be answered consistently vary considerably among models but are typically in the 50%-80% range for small models at low inference temperatures. Also, accuracy among consistent answers seems to reasonably correlate with overall accuracy. Results for medium-sized models seem to indicate much higher levels of answer consistency. 

**Abstract (ZH)**: 本研究探讨了小规模LLM（参数量为2B-8B）在多次回答相同问题时的一致性。我们对来自多项选择基准测试MMLU-Redux和MedQA的10次重复问题进行了研究，考虑了不同的推理温度、小型 vs. 中型模型（50B-80B）、微调 vs. 基模型以及其他参数的影响。我们还考察了强制要求多次回答一致性对准确性的效果及其权衡。为了支持这些研究，我们提出了一些新的分析和图形工具。结果显示，不同模型能够一致回答的问题数量差异明显，小模型在低推理温度下的准确率范围通常在50%-80%之间。而且，一致回答的准确率似乎与总体准确率之间存在合理的关联。对于中型模型的研究结果表明其回答一致性水平更高。 

---
# Temporal Preferences in Language Models for Long-Horizon Assistance 

**Title (ZH)**: 语言模型长时间 horizon 辅助的Temporal偏好 

**Authors**: Ali Mazyaki, Mohammad Naghizadeh, Samaneh Ranjkhah Zonouzaghi, Hossein Setareh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09704)  

**Abstract**: We study whether language models (LMs) exhibit future- versus present-oriented preferences in intertemporal choice and whether those preferences can be systematically manipulated. Using adapted human experimental protocols, we evaluate multiple LMs on time-tradeoff tasks and benchmark them against a sample of human decision makers. We introduce an operational metric, the Manipulability of Time Orientation (MTO), defined as the change in an LM's revealed time preference between future- and present-oriented prompts. In our tests, reasoning-focused models (e.g., DeepSeek-Reasoner and grok-3-mini) choose later options under future-oriented prompts but only partially personalize decisions across identities or geographies. Moreover, models that correctly reason about time orientation internalize a future orientation for themselves as AI decision makers. We discuss design implications for AI assistants that should align with heterogeneous, long-horizon goals and outline a research agenda on personalized contextual calibration and socially aware deployment. 

**Abstract (ZH)**: 我们研究语言模型在时间偏好选择中是否表现出未来导向或当下导向的偏好，以及这些偏好是否可以系统地操控。通过改编的人类实验协议，我们评估了多个语言模型在时间取舍任务中的表现，并将其与人类决策者样本进行基准比较。我们引入了一个操作性指标，即时间导向操控性（MTO），定义为在未来导向和当下导向提示下语言模型公开的时间偏好变化。在我们的测试中，注重推理的语言模型（如DeepSeek-Reasoner和grok-3-mini）在未来导向提示下选择较晚的选项，但在不同身份或地理上的个性化决策方面仅部分实现。此外，能够正确推理时间偏好的模型会使其作为AI决策者内化未来导向。我们讨论了与异质性和长期目标相一致的AI助手设计影响，并概述了个性化上下文校准和社会意识部署的研究议程。 

---
# CTCC: A Robust and Stealthy Fingerprinting Framework for Large Language Models via Cross-Turn Contextual Correlation Backdoor 

**Title (ZH)**: CTCC：一种基于跨回合上下文相关后门的鲁棒且隐蔽的大语言模型指纹识别框架 

**Authors**: Zhenhua Xu, Xixiang Zhao, Xubin Yue, Shengwei Tian, Changting Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.09703)  

**Abstract**: The widespread deployment of large language models (LLMs) has intensified concerns around intellectual property (IP) protection, as model theft and unauthorized redistribution become increasingly feasible. To address this, model fingerprinting aims to embed verifiable ownership traces into LLMs. However, existing methods face inherent trade-offs between stealthness, robustness, and generalizability, being either detectable via distributional shifts, vulnerable to adversarial modifications, or easily invalidated once the fingerprint is revealed. In this work, we introduce CTCC, a novel rule-driven fingerprinting framework that encodes contextual correlations across multiple dialogue turns, such as counterfactual, rather than relying on token-level or single-turn triggers. CTCC enables fingerprint verification under black-box access while mitigating false positives and fingerprint leakage, supporting continuous construction under a shared semantic rule even if partial triggers are exposed. Extensive experiments across multiple LLM architectures demonstrate that CTCC consistently achieves stronger stealth and robustness than prior work. Our findings position CTCC as a reliable and practical solution for ownership verification in real-world LLM deployment scenarios. Our code and data are publicly available at <this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的广泛应用加剧了对知识产权（IP）保护的担忧，因为模型盗用和未经授权的重新分发变得越来越可行。为此，模型指纹技术旨在将可验证的所有权痕迹嵌入到LLMs中。然而，现有方法在隐蔽性、鲁棒性和通用性之间存在固有的权衡，要么可以通过分布变化被检测到，要么容易受到 adversarial 修改的影响，要么在指纹被揭示后容易被无效化。在此工作中，我们引入了CTCC，一种新颖的基于规则的指纹框架，它编码了多个对话回合中的上下文关联，例如反事实，而不是依赖于令牌级或单回合触发器。CTCC 允许在黑盒访问情况下进行指纹验证，同时减少误报和指纹泄漏，并支持即使部分触发器被暴露，也可以在共享语义规则下持续构建。在多种LLM架构上的广泛实验表明，CTCC 在隐蔽性和鲁棒性方面均显著优于先前的工作。我们的研究结果将CTCC 定位为在实际场景中验证LLM所有权的一种可靠且实用的解决方案。我们的代码和数据已公开发布在 <this https URL>。 

---
# Creativity Benchmark: A benchmark for marketing creativity for LLM models 

**Title (ZH)**: 创造力基准：面向LLM模型的市场营销创造力基准 

**Authors**: Ninad Bhat, Kieran Browne, Pip Bingemann  

**Link**: [PDF](https://arxiv.org/pdf/2509.09702)  

**Abstract**: We introduce Creativity Benchmark, an evaluation framework for large language models (LLMs) in marketing creativity. The benchmark covers 100 brands (12 categories) and three prompt types (Insights, Ideas, Wild Ideas). Human pairwise preferences from 678 practising creatives over 11,012 anonymised comparisons, analysed with Bradley-Terry models, show tightly clustered performance with no model dominating across brands or prompt types: the top-bottom spread is $\Delta\theta \approx 0.45$, which implies a head-to-head win probability of $0.61$; the highest-rated model beats the lowest only about $61\%$ of the time. We also analyse model diversity using cosine distances to capture intra- and inter-model variation and sensitivity to prompt reframing. Comparing three LLM-as-judge setups with human rankings reveals weak, inconsistent correlations and judge-specific biases, underscoring that automated judges cannot substitute for human evaluation. Conventional creativity tests also transfer only partially to brand-constrained tasks. Overall, the results highlight the need for expert human evaluation and diversity-aware workflows. 

**Abstract (ZH)**: 创意基准：营销创意的大语言模型评估框架 

---
# Cross-Layer Attention Probing for Fine-Grained Hallucination Detection 

**Title (ZH)**: 跨层注意力探查以检测细粒度幻觉 

**Authors**: Malavika Suresh, Rahaf Aljundi, Ikechukwu Nkisi-Orji, Nirmalie Wiratunga  

**Link**: [PDF](https://arxiv.org/pdf/2509.09700)  

**Abstract**: With the large-scale adoption of Large Language Models (LLMs) in various applications, there is a growing reliability concern due to their tendency to generate inaccurate text, i.e. hallucinations. In this work, we propose Cross-Layer Attention Probing (CLAP), a novel activation probing technique for hallucination detection, which processes the LLM activations across the entire residual stream as a joint sequence. Our empirical evaluations using five LLMs and three tasks show that CLAP improves hallucination detection compared to baselines on both greedy decoded responses as well as responses sampled at higher temperatures, thus enabling fine-grained detection, i.e. the ability to disambiguate hallucinations and non-hallucinations among different sampled responses to a given prompt. This allows us to propose a detect-then-mitigate strategy using CLAP to reduce hallucinations and improve LLM reliability compared to direct mitigation approaches. Finally, we show that CLAP maintains high reliability even when applied out-of-distribution. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各种应用中的广泛应用，由于它们生成不准确文本（即幻觉）的趋势，可靠性的关注也越来越大。在这项工作中，我们提出了一种新颖的激活探查技术Cross-Layer Attention Probing（CLAP），该技术将LLM激活跨整个残差流作为联合序列进行处理，以进行幻觉检测。我们的实证评估使用了五种LLM和三种任务表明，与基线相比，CLAP在贪婪解码和较高温度下采样的响应中都能提高幻觉检测效果，从而实现细粒度检测，即在不同采样响应中区分幻觉和非幻觉的能力。这使得我们能够提出一种检测后再缓解的战略，使用CLAP来减少幻觉并提高LLM的可靠性，优于直接缓解方法。最后，我们展示了即便在离分布情况下，CLAP仍能保持高可靠性。 

---
# Structured Information Matters: Explainable ICD Coding with Patient-Level Knowledge Graphs 

**Title (ZH)**: 结构化信息很重要：基于患者级知识图谱的可解释ICD编码 

**Authors**: Mingyang Li, Viktor Schlegel, Tingting Mu, Warren Del-Pinto, Goran Nenadic  

**Link**: [PDF](https://arxiv.org/pdf/2509.09699)  

**Abstract**: Mapping clinical documents to standardised clinical vocabularies is an important task, as it provides structured data for information retrieval and analysis, which is essential to clinical research, hospital administration and improving patient care. However, manual coding is both difficult and time-consuming, making it impractical at scale. Automated coding can potentially alleviate this burden, improving the availability and accuracy of structured clinical data. The task is difficult to automate, as it requires mapping to high-dimensional and long-tailed target spaces, such as the International Classification of Diseases (ICD). While external knowledge sources have been readily utilised to enhance output code representation, the use of external resources for representing the input documents has been underexplored. In this work, we compute a structured representation of the input documents, making use of document-level knowledge graphs (KGs) that provide a comprehensive structured view of a patient's condition. The resulting knowledge graph efficiently represents the patient-centred input documents with 23\% of the original text while retaining 90\% of the information. We assess the effectiveness of this graph for automated ICD-9 coding by integrating it into the state-of-the-art ICD coding architecture PLM-ICD. Our experiments yield improved Macro-F1 scores by up to 3.20\% on popular benchmarks, while improving training efficiency. We attribute this improvement to different types of entities and relationships in the KG, and demonstrate the improved explainability potential of the approach over the text-only baseline. 

**Abstract (ZH)**: 临床文档到标准化临床词汇的映射是一项重要任务，它提供了用于信息检索和分析的结构化数据，对于临床研究、医院管理和改善患者护理至关重要。然而，手工编码既困难又耗时，使其在大规模应用中不可行。自动化编码可以缓解这一负担，提高结构化临床数据的可用性和准确性。该任务难以自动化，因为它需要映射到高维和长尾的目标空间，例如国际疾病分类（ICD）。虽然外部知识源已被广泛用于增强输出代码表示，但利用外部资源表示输入文档仍被忽视。在本文中，我们计算了输入文档的结构化表示，利用提供患者状况全面结构化视图的文档级别知识图（KG）。生成的知识图以原文本的23%的大小高效地表示患者中心的输入文档，同时保留了90%的信息。通过将其集成到最先进的ICD编码架构PLM-ICD中，评估了该图在自动化ICD-9编码中的效果。实验结果在流行基准上的宏F1分数上提高了最多3.20%，同时提高了解析效率。我们将这一改进归因于知识图中的不同类型的实体和关系，并证明了该方法在解释性方面的改进潜力，优于仅基于文本的基线。 

---
# Wave-Based Semantic Memory with Resonance-Based Retrieval: A Phase-Aware Alternative to Vector Embedding Stores 

**Title (ZH)**: 基于波的语义记忆与共振检索：相位感知的向量嵌入存储替代方案 

**Authors**: Aleksandr Listopad  

**Link**: [PDF](https://arxiv.org/pdf/2509.09691)  

**Abstract**: Conventional vector-based memory systems rely on cosine or inner product similarity within real-valued embedding spaces. While computationally efficient, such approaches are inherently phase-insensitive and limited in their ability to capture resonance phenomena crucial for meaning representation. We propose Wave-Based Semantic Memory, a novel framework that models knowledge as wave patterns $\psi(x) = A(x) e^{i\phi(x)}$ and retrieves it through resonance-based interference. This approach preserves both amplitude and phase information, enabling more expressive and robust semantic similarity. We demonstrate that resonance-based retrieval achieves higher discriminative power in cases where vector methods fail, including phase shifts, negations, and compositional queries. Our implementation, ResonanceDB, shows scalability to millions of patterns with millisecond latency, positioning wave-based memory as a viable alternative to vector stores for AGI-oriented reasoning and knowledge representation. 

**Abstract (ZH)**: 基于波的语义记忆：通过共振干扰建模和检索知识 

---
# Personas within Parameters: Fine-Tuning Small Language Models with Low-Rank Adapters to Mimic User Behaviors 

**Title (ZH)**: Personas within Parameters: 使用低秩适配器Fine-tuning小型语言模型以模拟用户行为 

**Authors**: Himanshu Thakur, Eshani Agrawal, Smruthi Mukund  

**Link**: [PDF](https://arxiv.org/pdf/2509.09689)  

**Abstract**: A long-standing challenge in developing accurate recommendation models is simulating user behavior, mainly due to the complex and stochastic nature of user interactions. Towards this, one promising line of work has been the use of Large Language Models (LLMs) for simulating user behavior. However, aligning these general-purpose large pre-trained models with user preferences necessitates: (i) effectively and continously parsing large-scale tabular user-item interaction data, (ii) overcoming pre-training-induced inductive biases to accurately learn user specific knowledge, and (iii) achieving the former two at scale for millions of users. While most previous works have focused on complex methods to prompt an LLM or fine-tune it on tabular interaction datasets, our approach shifts the focus to extracting robust textual user representations using a frozen LLM and simulating cost-effective, resource-efficient user agents powered by fine-tuned Small Language Models (SLMs). Further, we showcase a method for training multiple low-rank adapters for groups of users or \textit{persona}, striking an optimal balance between scalability and performance of user behavior agents. Our experiments provide compelling empirical evidence of the efficacy of our methods, demonstrating that user agents developed using our approach have the potential to bridge the gap between offline metrics and real-world performance of recommender systems. 

**Abstract (ZH)**: 持续构建准确推荐模型的一大长期挑战在于模拟用户行为，主要归因于用户交互的复杂性和随机性。为此，利用大型语言模型（LLMs）模拟用户行为的研究方向展现出了一定的前景。然而，将这些通用的大规模预训练模型与用户偏好相结合需解决：（i）有效且持续地解析大规模的用户-项目交互表格数据，（ii）克服预训练带来的归纳偏见以准确学习用户特定的知识，以及（iii）在数百万用户规模上实现上述两点。尽管大多数先前的研究侧重于使用复杂方法来提示LLM或在其上进行微调，我们的方式则转向使用冻结的LLM提取稳健的文本用户表示，并通过微调的小型语言模型（SLMs）驱动成本效益和资源效率更高的用户代理。此外，我们展示了训练多个低秩适配器以模拟不同用户群或角色的方法，以在用户行为代理的可扩展性和性能之间取得最佳平衡。我们的实验提供了关于方法有效性的有力实证证据，表明使用我们这种方式构建的用户代理有可能弥合推荐系统离线指标与实际性能之间的差距。 

---
# AI-Powered Assistant for Long-Term Access to RHIC Knowledge 

**Title (ZH)**: 基于AI的RHIC知识长期访问辅助系统 

**Authors**: Mohammad Atif, Vincent Garonne, Eric Lancon, Jerome Lauret, Alexandr Prozorov, Michal Vranovsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.09688)  

**Abstract**: As the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory concludes 25 years of operation, preserving not only its vast data holdings ($\sim$1 ExaByte) but also the embedded scientific knowledge becomes a critical priority. The RHIC Data and Analysis Preservation Plan (DAPP) introduces an AI-powered assistant system that provides natural language access to documentation, workflows, and software, with the aim of supporting reproducibility, education, and future discovery. Built upon Large Language Models using Retrieval-Augmented Generation and the Model Context Protocol, this assistant indexes structured and unstructured content from RHIC experiments and enables domain-adapted interaction. We report on the deployment, computational performance, ongoing multi-experiment integration, and architectural features designed for a sustainable and explainable long-term AI access. Our experience illustrates how modern AI/ML tools can transform the usability and discoverability of scientific legacy data. 

**Abstract (ZH)**: 随着布鲁海文国家实验室相对论重离子 Collider (RHIC) 完成25年的运营，在保存其庞大数据 holdings（约1艾字节）的同时，保留嵌入其中的科学知识也变得至关重要。RHIC 数据和分析保存计划 (DAPP) 引入了一个基于 AI 的助手系统，该系统提供自然语言访问文档、工作流和软件的方式，旨在支持可重复性、教育和未来发现。基于大型语言模型并使用检索增强生成和模型上下文协议构建，该助手为 RHIC 实验内容索引结构化和非结构化内容，并实现领域适应交互。我们报告了该助手系统的部署、计算性能、多实验集成的持续进行以及为实现可持续性和可解释性长期 AI 访问而设计的架构特征。我们的经验展示了现代 AI/ML 工具如何改变科学遗产数据的可用性和发现性。 

---
# GeoGPT.RAG Technical Report 

**Title (ZH)**: GeoGPT.RAG 技术报告 

**Authors**: Fei Huang, Fan Wu, Zeqing Zhang, Qihao Wang, Long Zhang, Grant Michael Boquet, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09686)  

**Abstract**: GeoGPT is an open large language model system built to advance research in the geosciences. To enhance its domain-specific capabilities, we integrated Retrieval Augmented Generation(RAG), which augments model outputs with relevant information retrieved from an external knowledge source. GeoGPT uses RAG to draw from the GeoGPT Library, a specialized corpus curated for geoscientific content, enabling it to generate accurate, context-specific answers. Users can also create personalized knowledge bases by uploading their own publication lists, allowing GeoGPT to retrieve and respond using user-provided materials. To further improve retrieval quality and domain alignment, we fine-tuned both the embedding model and a ranking model that scores retrieved passages by relevance to the query. These enhancements optimize RAG for geoscience applications and significantly improve the system's ability to deliver precise and trustworthy outputs. GeoGPT reflects a strong commitment to open science through its emphasis on collaboration, transparency, and community driven development. As part of this commitment, we have open-sourced two core RAG components-GeoEmbedding and GeoReranker-to support geoscientists, researchers, and professionals worldwide with powerful, accessible AI tools. 

**Abstract (ZH)**: GeoGPT是一个旨在推进地球科学领域研究的开放大语言模型系统。为了增强其领域特定能力，我们整合了检索增强生成（RAG）技术，通过从外部知识库中检索相关的信息来补充模型输出。GeoGPT利用RAG从专门为地球科学内容定制的GeoGPT库中获取信息，生成准确且上下文相关的答案。用户还可以通过上传自己的出版列表创建个性化的知识库，使GeoGPT能够利用用户提供的材料获取和响应。为了进一步提高检索质量和领域一致性，我们对嵌入模型和评分模型进行了微调，该评分模型根据查询的相关性对检索到的段落进行评分。这些改进使RAG更适合地球科学应用，并极大地提升了系统的精确性和可信度。GeoGPT体现了对开放科学的强烈承诺，强调合作、透明度和社区驱动的发展。作为这一承诺的一部分，我们已开源了两个核心的RAG组件——GeoEmbedding和GeoReranker，以支持全球范围内的地球科学家、研究人员和专业人士使用强大的、易获取的人工智能工具。 

---
# TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation 

**Title (ZH)**: TalkPlayData 2: 一种自主式多模态对话音乐推荐合成数据管道 

**Authors**: Keunwoo Choi, Seungheon Doh, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2509.09685)  

**Abstract**: We present TalkPlayData 2, a synthetic dataset for multimodal conversational music recommendation generated by an agentic data pipeline. In TalkPlayData 2 pipeline, multiple large language model (LLM) agents are created under various roles with specialized prompts and access to different parts of information, and the chat data is acquired by logging the conversation between the Listener LLM and the Recsys LLM. To cover various conversation scenarios, for each conversation, the Listener LLM is conditioned on a finetuned conversation goal. Finally, all the LLMs are multimodal with audio and images, allowing a simulation of multimodal recommendation and conversation. In the LLM-as-a-judge and subjective evaluation experiments, TalkPlayData 2 achieved the proposed goal in various aspects related to training a generative recommendation model for music. TalkPlayData 2 and its generation code are open-sourced at this https URL. 

**Abstract (ZH)**: 我们呈现 TalkPlayData 2，这是一个由自主数据管道生成的多模态对话音乐推荐合成数据集 

---
# Text-to-SQL Oriented to the Process Mining Domain: A PT-EN Dataset for Query Translation 

**Title (ZH)**: 面向过程挖掘领域的文本到SQL转换：PT-EN数据集用于查询翻译 

**Authors**: Bruno Yui Yamate, Thais Rodrigues Neubauer, Marcelo Fantinato, Sarajane Marques Peres  

**Link**: [PDF](https://arxiv.org/pdf/2509.09684)  

**Abstract**: This paper introduces text-2-SQL-4-PM, a bilingual (Portuguese-English) benchmark dataset designed for the text-to-SQL task in the process mining domain. Text-to-SQL conversion facilitates natural language querying of databases, increasing accessibility for users without SQL expertise and productivity for those that are experts. The text-2-SQL-4-PM dataset is customized to address the unique challenges of process mining, including specialized vocabularies and single-table relational structures derived from event logs. The dataset comprises 1,655 natural language utterances, including human-generated paraphrases, 205 SQL statements, and ten qualifiers. Methods include manual curation by experts, professional translations, and a detailed annotation process to enable nuanced analyses of task complexity. Additionally, a baseline study using GPT-3.5 Turbo demonstrates the feasibility and utility of the dataset for text-to-SQL applications. The results show that text-2-SQL-4-PM supports evaluation of text-to-SQL implementations, offering broader applicability for semantic parsing and other natural language processing tasks. 

**Abstract (ZH)**: 文本到SQL用于过程挖掘的双语（葡萄牙语-英语）基准数据集：text-2-SQL-4-PM 

---
# Forecasting Clicks in Digital Advertising: Multimodal Inputs and Interpretable Outputs 

**Title (ZH)**: 数字广告中的点击预测：多模态输入与可解释输出 

**Authors**: Briti Gangopadhyay, Zhao Wang, Shingo Takamatsu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09683)  

**Abstract**: Forecasting click volume is a key task in digital advertising, influencing both revenue and campaign strategy. Traditional time series models rely solely on numerical data, often overlooking rich contextual information embedded in textual elements, such as keyword updates. We present a multimodal forecasting framework that combines click data with textual logs from real-world ad campaigns and generates human-interpretable explanations alongside numeric predictions. Reinforcement learning is used to improve comprehension of textual information and enhance fusion of modalities. Experiments on a large-scale industry dataset show that our method outperforms baselines in both accuracy and reasoning quality. 

**Abstract (ZH)**: 数字广告中点击量预测是一项关键任务，影响收入和campaign策略。传统时间序列模型仅依赖数值数据，往往忽视了嵌入在文本元素中的丰富上下文信息，如关键词更新。我们提出了一种多模态预测框架，结合点击数据和实际广告campaign的文本日志，并生成与数值预测并行的人类可解释的解释。强化学习用于提高对文本信息的理解并增强模态融合。大规模工业数据集上的实验表明，我们的方法在准确性和推理质量上均优于基线方法。 

---
# DB3 Team's Solution For Meta KDD Cup' 25 

**Title (ZH)**: DB3团队的Meta KDD Cup'25解决方案 

**Authors**: Yikuan Xia, Jiazun Chen, Yirui Zhan, Suifeng Zhao, Weipeng Jiang, Chaorui Zhang, Wei Han, Bo Bai, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.09681)  

**Abstract**: This paper presents the db3 team's winning solution for the Meta CRAG-MM Challenge 2025 at KDD Cup'25. Addressing the challenge's unique multi-modal, multi-turn question answering benchmark (CRAG-MM), we developed a comprehensive framework that integrates tailored retrieval pipelines for different tasks with a unified LLM-tuning approach for hallucination control. Our solution features (1) domain-specific retrieval pipelines handling image-indexed knowledge graphs, web sources, and multi-turn conversations; and (2) advanced refusal training using SFT, DPO, and RL. The system achieved 2nd place in Task 1, 2nd place in Task 2, and 1st place in Task 3, securing the grand prize for excellence in ego-centric queries through superior handling of first-person perspective challenges. 

**Abstract (ZH)**: db3团队在KDD Cup'25 Meta CRAG-MM挑战赛中的获奖解决方案 

---
# AEGIS: An Agent for Extraction and Geographic Identification in Scholarly Proceedings 

**Title (ZH)**: AEGIS: 一种用于学术会议论文中提取和地理识别的智能体 

**Authors**: Om Vishesh, Harshad Khadilkar, Deepak Akkil  

**Link**: [PDF](https://arxiv.org/pdf/2509.09470)  

**Abstract**: Keeping pace with the rapid growth of academia literature presents a significant challenge for researchers, funding bodies, and academic societies. To address the time-consuming manual effort required for scholarly discovery, we present a novel, fully automated system that transitions from data discovery to direct action. Our pipeline demonstrates how a specialized AI agent, 'Agent-E', can be tasked with identifying papers from specific geographic regions within conference proceedings and then executing a Robotic Process Automation (RPA) to complete a predefined action, such as submitting a nomination form. We validated our system on 586 papers from five different conferences, where it successfully identified every target paper with a recall of 100% and a near perfect accuracy of 99.4%. This demonstration highlights the potential of task-oriented AI agents to not only filter information but also to actively participate in and accelerate the workflows of the academic community. 

**Abstract (ZH)**: 随着学术文献的迅速增长，研究人员、资助机构和学术社会面临着重大挑战。为了解决学者发现过程中耗时的手动努力，我们提出了一种全新的完全自动系统，从数据发现过渡到直接行动。我们的流水线展示了如何专门的AI代理“Agent-E”被任务化为识别会议论文中的特定地理区域的论文，并执行机器人过程自动化（RPA）以完成预定义的动作，如提交提名表。我们在五次不同会议的586篇论文上验证了该系统，成功识别了每篇目标论文，召回率为100%，准确率为99.4%。这一演示突显了任务导向型AI代理不仅能够过滤信息，还能积极参与和加速学术社区的工作流程的潜力。 

---
# Clip Your Sequences Fairly: Enforcing Length Fairness for Sequence-Level RL 

**Title (ZH)**: 裁剪你的序列以实现公平性：在序列级强化学习中 enforcing 长度公平性 

**Authors**: Hanyi Mao, Quanjia Xiao, Lei Pang, Haixiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09177)  

**Abstract**: We propose FSPO (Fair Sequence Policy Optimization), a sequence-level reinforcement learning method for LLMs that enforces length-fair clipping directly in the importance-sampling (IS) weight space. We revisit sequence-level RL methods and identify a mismatch when PPO/GRPO-style clipping is transplanted to sequences: a fixed clip range systematically reweights short vs. long responses, distorting the effective objective. Theoretically, we formalize length fairness via a Length Reweighting Error (LRE) and prove that small LRE yields a directional cosine guarantee between the clipped and true updates. FSPO introduces a simple, Gaussian-motivated remedy: we clip the sequence log-IS ratio with a band that applies a KL-corrected drift term and scales as $\sqrt{L}$. Empirically, FSPO flattens clip rates across length bins, stabilizes training, and outperforms all baselines across multiple evaluation datasets. 

**Abstract (ZH)**: 我们提出FSPO（公平序列策略优化），这是一种直接在重要性加权空间中实施长度公平剪裁的LLM序列级强化学习方法。 

---
# Generative Engine Optimization: How to Dominate AI Search 

**Title (ZH)**: 生成引擎优化：如何主导AI搜索 

**Authors**: Mahe Chen, Xiaoxuan Wang, Kaiwen Chen, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2509.08919)  

**Abstract**: The rapid adoption of generative AI-powered search engines like ChatGPT, Perplexity, and Gemini is fundamentally reshaping information retrieval, moving from traditional ranked lists to synthesized, citation-backed answers. This shift challenges established Search Engine Optimization (SEO) practices and necessitates a new paradigm, which we term Generative Engine Optimization (GEO).
This paper presents a comprehensive comparative analysis of AI Search and traditional web search (Google). Through a series of large-scale, controlled experiments across multiple verticals, languages, and query paraphrases, we quantify critical differences in how these systems source information. Our key findings reveal that AI Search exhibit a systematic and overwhelming bias towards Earned media (third-party, authoritative sources) over Brand-owned and Social content, a stark contrast to Google's more balanced mix. We further demonstrate that AI Search services differ significantly from each other in their domain diversity, freshness, cross-language stability, and sensitivity to phrasing.
Based on these empirical results, we formulate a strategic GEO agenda. We provide actionable guidance for practitioners, emphasizing the critical need to: (1) engineer content for machine scannability and justification, (2) dominate earned media to build AI-perceived authority, (3) adopt engine-specific and language-aware strategies, and (4) overcome the inherent "big brand bias" for niche players. Our work provides the foundational empirical analysis and a strategic framework for achieving visibility in the new generative search landscape. 

**Abstract (ZH)**: 生成式AI驱动的搜索引擎（如ChatGPT、Perplexity和Gemini）的快速采纳根本性地重塑了信息检索，从传统的排名列表转变为合成的、带有引文支持的答案。这一转变挑战了现有的搜索引擎优化（SEO）实践，需要一种新的范式，我们称之为生成式引擎优化（GEO）。

本文对AI搜索和传统Web搜索（如Google）进行了全面的比较分析。通过在多个垂直领域、多种语言和查询变体上进行大规模控制实验，我们量化了这些系统获取信息的关键差异。我们的主要发现表明，AI搜索系统系统性和压倒性地偏向于第三方权威来源的内容（Earned媒体），而忽视了品牌自有内容和社会媒体内容，这与Google更加平衡的混合形式形成了鲜明对比。我们进一步证明，不同的AI搜索服务在领域多样性、新鲜度、跨语言稳定性以及对措辞的敏感性方面存在显著差异。

基于这些实证结果，我们制定了生成式引擎优化（GEO）的战略议程。我们为实践者提供了可操作的指导，强调了以下几个关键需求：（1）设计内容以提高机器可扫描性和合理性，（2）主导赚取的媒体以建立AI感知的权威，（3）采用特定于引擎和语言的策略，（4）克服内在的“大品牌偏好”以帮助小玩家。我们的研究提供了生成式搜索景观中实现可见性的基础实证分析和战略框架。 

---
