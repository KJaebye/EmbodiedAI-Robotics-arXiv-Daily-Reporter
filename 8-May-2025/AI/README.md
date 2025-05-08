# Qualitative Analysis of $ω$-Regular Objectives on Robust MDPs 

**Title (ZH)**: ω-正规目标下鲁棒MDP的定性分析 

**Authors**: Ali Asadi, Krishnendu Chatterjee, Ehsan Kafshdar Goharshady, Mehrdad Karrabi, Ali Shafiee  

**Link**: [PDF](https://arxiv.org/pdf/2505.04539)  

**Abstract**: Robust Markov Decision Processes (RMDPs) generalize classical MDPs that consider uncertainties in transition probabilities by defining a set of possible transition functions. An objective is a set of runs (or infinite trajectories) of the RMDP, and the value for an objective is the maximal probability that the agent can guarantee against the adversarial environment. We consider (a) reachability objectives, where given a target set of states, the goal is to eventually arrive at one of them; and (b) parity objectives, which are a canonical representation for $\omega$-regular objectives. The qualitative analysis problem asks whether the objective can be ensured with probability 1.
In this work, we study the qualitative problem for reachability and parity objectives on RMDPs without making any assumption over the structures of the RMDPs, e.g., unichain or aperiodic. Our contributions are twofold. We first present efficient algorithms with oracle access to uncertainty sets that solve qualitative problems of reachability and parity objectives. We then report experimental results demonstrating the effectiveness of our oracle-based approach on classical RMDP examples from the literature scaling up to thousands of states. 

**Abstract (ZH)**: 鲁棒马尔可夫决策过程中的稳健性问题：无结构假设下的可达性和优先级目标的质性分析 

---
# Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving 

**Title (ZH)**: 超越定理证明：形式问题求解的表述、框架与基准 

**Authors**: Qi Liu, Xinhao Zheng, Renqiu Xia, Xingzhi Qi, Qinxiang Cao, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04528)  

**Abstract**: As a seemingly self-explanatory task, problem-solving has been a significant component of science and engineering. However, a general yet concrete formulation of problem-solving itself is missing. With the recent development of AI-based problem-solving agents, the demand for process-level verifiability is rapidly increasing yet underexplored. To fill these gaps, we present a principled formulation of problem-solving as a deterministic Markov decision process; a novel framework, FPS (Formal Problem-Solving), which utilizes existing FTP (formal theorem proving) environments to perform process-verified problem-solving; and D-FPS (Deductive FPS), decoupling solving and answer verification for better human-alignment. The expressiveness, soundness and completeness of the frameworks are proven. We construct three benchmarks on problem-solving: FormalMath500, a formalization of a subset of the MATH500 benchmark; MiniF2F-Solving and PutnamBench-Solving, adaptations of FTP benchmarks MiniF2F and PutnamBench. For faithful, interpretable, and human-aligned evaluation, we propose RPE (Restricted Propositional Equivalence), a symbolic approach to determine the correctness of answers by formal verification. We evaluate four prevalent FTP models and two prompting methods as baselines, solving at most 23.77% of FormalMath500, 27.47% of MiniF2F-Solving, and 0.31% of PutnamBench-Solving. 

**Abstract (ZH)**: 一种形式化的问题求解框架：D-FPS及其应用 

---
# On some improvements to Unbounded Minimax 

**Title (ZH)**: 关于Unbounded Minimax的一些改进 

**Authors**: Quentin Cohen-Solal, Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2505.04525)  

**Abstract**: This paper presents the first experimental evaluation of four previously untested modifications of Unbounded Best-First Minimax algorithm. This algorithm explores the game tree by iteratively expanding the most promising sequences of actions based on the current partial game tree. We first evaluate the use of transposition tables, which convert the game tree into a directed acyclic graph by merging duplicate states. Second, we compare the original algorithm by Korf & Chickering with the variant proposed by Cohen-Solal, which differs in its backpropagation strategy: instead of stopping when a stable value is encountered, it updates values up to the root. This change slightly improves performance when value ties or transposition tables are involved. Third, we assess replacing the exact terminal evaluation function with the learned heuristic function. While beneficial when exact evaluations are costly, this modification reduces performance in inexpensive settings. Finally, we examine the impact of the completion technique that prioritizes resolved winning states and avoids resolved losing states. This technique also improves performance. Overall, our findings highlight how targeted modifications can enhance the efficiency of Unbounded Best-First Minimax. 

**Abstract (ZH)**: 本论文首次对四种未测试的Unbounded Best-First Minimax算法修改进行实验评估。该算法通过迭代扩展当前部分游戏树中最有可能的动作序列来探索游戏树。首先评估了转置表的使用情况，该技术通过合并重复状态将游戏树转换为有向无环图。其次，将Korf & Chickering的原始算法与Cohen-Solal提出的变体进行比较，后者在回传策略上有差异：不再遇到稳定值时停止，而是更新到根的所有值。这种变化在涉及价值平局或转置表时轻微提高了性能。第三，评估用学习启发式函数替换精确终局评估函数的情况，虽然在精确评估成本高的情况下有益，但这种修改在成本低廉的情况下降低了性能。最后，研究了优先处理已解决的获胜状态并避免已解决的失败状态的完成技术对其性能的提升作用。总体而言，本文的研究结果强调了有针对性的修改如何提升Unbounded Best-First Minimax算法的效率。 

---
# TrajEvo: Designing Trajectory Prediction Heuristics via LLM-driven Evolution 

**Title (ZH)**: TrajEvo: 通过大语言模型驱动的进化设计轨迹预测启发式方法 

**Authors**: Zhikai Zhao, Chuanbo Hua, Federico Berto, Kanghoon Lee, Zihan Ma, Jiachen Li, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.04480)  

**Abstract**: Trajectory prediction is a crucial task in modeling human behavior, especially in fields as social robotics and autonomous vehicle navigation. Traditional heuristics based on handcrafted rules often lack accuracy, while recently proposed deep learning approaches suffer from computational cost, lack of explainability, and generalization issues that limit their practical adoption. In this paper, we introduce TrajEvo, a framework that leverages Large Language Models (LLMs) to automatically design trajectory prediction heuristics. TrajEvo employs an evolutionary algorithm to generate and refine prediction heuristics from past trajectory data. We introduce a Cross-Generation Elite Sampling to promote population diversity and a Statistics Feedback Loop allowing the LLM to analyze alternative predictions. Our evaluations show TrajEvo outperforms previous heuristic methods on the ETH-UCY datasets, and remarkably outperforms both heuristics and deep learning methods when generalizing to the unseen SDD dataset. TrajEvo represents a first step toward automated design of fast, explainable, and generalizable trajectory prediction heuristics. We make our source code publicly available to foster future research at this https URL. 

**Abstract (ZH)**: 基于大规模语言模型的轨迹进化预测框架 

---
# Uncertain Machine Ethics Planning 

**Title (ZH)**: 不确定性机器伦理规划 

**Authors**: Simon Kolker, Louise A. Dennis, Ramon Fraga Pereira, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04352)  

**Abstract**: Machine Ethics decisions should consider the implications of uncertainty over decisions. Decisions should be made over sequences of actions to reach preferable outcomes long term. The evaluation of outcomes, however, may invoke one or more moral theories, which might have conflicting judgements. Each theory will require differing representations of the ethical situation. For example, Utilitarianism measures numerical values, Deontology analyses duties, and Virtue Ethics emphasises moral character. While balancing potentially conflicting moral considerations, decisions may need to be made, for example, to achieve morally neutral goals with minimal costs. In this paper, we formalise the problem as a Multi-Moral Markov Decision Process and a Multi-Moral Stochastic Shortest Path Problem. We develop a heuristic algorithm based on Multi-Objective AO*, utilising Sven-Ove Hansson's Hypothetical Retrospection procedure for ethical reasoning under uncertainty. Our approach is validated by a case study from Machine Ethics literature: the problem of whether to steal insulin for someone who needs it. 

**Abstract (ZH)**: 机器伦理决策应考虑决策中的不确定性影响。应基于行动序列以实现长期更佳结果作出决策。然而，结果评估可能涉及一个或多个道德理论，这些理论可能产生相互冲突的判断。每个理论都需要不同的伦理情景表示。例如，功利主义衡量数值，义务论分析义务，美德伦理学强调道德品格。在平衡潜在冲突的道德考量时，可能需要作出决策，例如，以最小成本实现道德中立的目标。在这篇论文中，我们将问题形式化为多道德马尔可夫决策过程和多道德随机最短路径问题。我们开发了一个基于多目标AO*的启发式算法，并利用Sven-Ove Hansson的假设回顾程序进行不确定性下的伦理推理。我们的方法通过机器伦理文献中的一个案例研究得到了验证：是否应为需要的人偷窃胰岛素的问题。 

---
# Mastering Multi-Drone Volleyball through Hierarchical Co-Self-Play Reinforcement Learning 

**Title (ZH)**: 通过分层协同自博弈强化学习掌握多无人机排球技能 

**Authors**: Ruize Zhang, Sirui Xiang, Zelai Xu, Feng Gao, Shilong Ji, Wenhao Tang, Wenbo Ding, Chao Yu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04317)  

**Abstract**: In this paper, we tackle the problem of learning to play 3v3 multi-drone volleyball, a new embodied competitive task that requires both high-level strategic coordination and low-level agile control. The task is turn-based, multi-agent, and physically grounded, posing significant challenges due to its long-horizon dependencies, tight inter-agent coupling, and the underactuated dynamics of quadrotors. To address this, we propose Hierarchical Co-Self-Play (HCSP), a hierarchical reinforcement learning framework that separates centralized high-level strategic decision-making from decentralized low-level motion control. We design a three-stage population-based training pipeline to enable both strategy and skill to emerge from scratch without expert demonstrations: (I) training diverse low-level skills, (II) learning high-level strategy via self-play with fixed low-level controllers, and (III) joint fine-tuning through co-self-play. Experiments show that HCSP achieves superior performance, outperforming non-hierarchical self-play and rule-based hierarchical baselines with an average 82.9\% win rate and a 71.5\% win rate against the two-stage variant. Moreover, co-self-play leads to emergent team behaviors such as role switching and coordinated formations, demonstrating the effectiveness of our hierarchical design and training scheme. 

**Abstract (ZH)**: 基于层次协作自博弈的三对三多旋翼排球学习方法 

---
# KERAIA: An Adaptive and Explainable Framework for Dynamic Knowledge Representation and Reasoning 

**Title (ZH)**: KERAIA：一种适应性和可解释的动态知识表示与推理框架 

**Authors**: Stephen Richard Varey, Alessandro Di Stefano, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.04313)  

**Abstract**: In this paper, we introduce KERAIA, a novel framework and software platform for symbolic knowledge engineering designed to address the persistent challenges of representing, reasoning with, and executing knowledge in dynamic, complex, and context-sensitive environments. The central research question that motivates this work is: How can unstructured, often tacit, human expertise be effectively transformed into computationally tractable algorithms that AI systems can efficiently utilise? KERAIA seeks to bridge this gap by building on foundational concepts such as Minsky's frame-based reasoning and K-lines, while introducing significant innovations. These include Clouds of Knowledge for dynamic aggregation, Dynamic Relations (DRels) for context-sensitive inheritance, explicit Lines of Thought (LoTs) for traceable reasoning, and Cloud Elaboration for adaptive knowledge transformation. This approach moves beyond the limitations of traditional, often static, knowledge representation paradigms. KERAIA is designed with Explainable AI (XAI) as a core principle, ensuring transparency and interpretability, particularly through the use of LoTs. The paper details the framework's architecture, the KSYNTH representation language, and the General Purpose Paradigm Builder (GPPB) to integrate diverse inference methods within a unified structure. We validate KERAIA's versatility, expressiveness, and practical applicability through detailed analysis of multiple case studies spanning naval warfare simulation, industrial diagnostics in water treatment plants, and strategic decision-making in the game of RISK. Furthermore, we provide a comparative analysis against established knowledge representation paradigms (including ontologies, rule-based systems, and knowledge graphs) and discuss the implementation aspects and computational considerations of the KERAIA platform. 

**Abstract (ZH)**: KERAIA：一种用于动态复杂环境中的符号知识工程的新框架与软件平台 

---
# Flow Models for Unbounded and Geometry-Aware Distributional Reinforcement Learning 

**Title (ZH)**: 流模型在无界和几何感知分布强化学习中的应用 

**Authors**: Simo Alami C., Rim Kaddah, Jesse Read, Marie-Paule Cani  

**Link**: [PDF](https://arxiv.org/pdf/2505.04310)  

**Abstract**: We introduce a new architecture for Distributional Reinforcement Learning (DistRL) that models return distributions using normalizing flows. This approach enables flexible, unbounded support for return distributions, in contrast to categorical approaches like C51 that rely on fixed or bounded representations. It also offers richer modeling capacity to capture multi-modality, skewness, and tail behavior than quantile based approaches. Our method is significantly more parameter-efficient than categorical approaches. Standard metrics used to train existing models like KL divergence or Wasserstein distance either are scale insensitive or have biased sample gradients, especially when return supports do not overlap. To address this, we propose a novel surrogate for the Cramèr distance, that is geometry-aware and computable directly from the return distribution's PDF, avoiding the costly CDF computation. We test our model on the ATARI-5 sub-benchmark and show that our approach outperforms PDF based models while remaining competitive with quantile based methods. 

**Abstract (ZH)**: 我们提出了一种新的分布强化学习（DistRL）架构，使用归一化流来建模回报分布。该方法提供了灵活且无界的回报分布支持，相比之下，如C51等基于分类的方法依赖于固定或有界的表现形式。此外，该方法具有更强的建模能力，能够捕捉多模态、偏斜度和尾部行为，优于基于分位数的方法。与分类方法相比，我们的方法参数效率更高。用于训练现有模型的标准度量标准，如KL散度或Wasserstein距离，要么对尺度不敏感，要么在回报支持不重叠的情况下有偏的样本梯度。为了解决这个问题，我们提出了一种新的Cramèr距离的替代方法，该方法具有几何感知性并且可以直接从回报分布的PDF中计算，避免了昂贵的CDF计算。我们在ATARI-5子基准上测试了我们的模型，并证明了我们的方法在保持与基于分位数方法竞争力的同时优于基于PDF的方法。 

---
# Polynomial-Time Relational Probabilistic Inference in Open Universes 

**Title (ZH)**: 开放式宇宙中的多项式时间关系概率推理 

**Authors**: Luise Ge, Brendan Juba, Kris Nilsson  

**Link**: [PDF](https://arxiv.org/pdf/2505.04115)  

**Abstract**: Reasoning under uncertainty is a fundamental challenge in Artificial Intelligence. As with most of these challenges, there is a harsh dilemma between the expressive power of the language used, and the tractability of the computational problem posed by reasoning. Inspired by human reasoning, we introduce a method of first-order relational probabilistic inference that satisfies both criteria, and can handle hybrid (discrete and continuous) variables. Specifically, we extend sum-of-squares logic of expectation to relational settings, demonstrating that lifted reasoning in the bounded-degree fragment for knowledge bases of bounded quantifier rank can be performed in polynomial time, even with an a priori unknown and/or countably infinite set of objects. Crucially, our notion of tractability is framed in proof-theoretic terms, which extends beyond the syntactic properties of the language or queries. We are able to derive the tightest bounds provable by proofs of a given degree and size and establish completeness in our sum-of-squares refutations for fixed degrees. 

**Abstract (ZH)**: 在不确定性下的推理是人工智能中的一个根本性挑战。受到人类推理的启发，我们提出了一种一阶关系概率推理方法，该方法同时满足表达力和计算可处理性的要求，并能处理混合（离散和连续）变量。具体来说，我们将期望的平方和逻辑扩展到关系设置中，证明在有界度片段中，即使面对的是先验未知且可能是可数无穷的对象集，基于有界量词级的知识库也可以在多项式时间内进行提升推理。最关键的是，我们关于计算可处理性的概念是用证明论的方式定义的，这超越了语言或查询的句法属性。我们能够推导出由给定证明度和大小可证明的最紧边界，并为固定度数建立了平方和反驳的完备性。 

---
# Extending Decision Predicate Graphs for Comprehensive Explanation of Isolation Forest 

**Title (ZH)**: 扩展决策谓词图以全面解释孤立森林 

**Authors**: Matteo Ceschin, Leonardo Arrighi, Luca Longo, Sylvio Barbon Junior  

**Link**: [PDF](https://arxiv.org/pdf/2505.04019)  

**Abstract**: The need to explain predictive models is well-established in modern machine learning. However, beyond model interpretability, understanding pre-processing methods is equally essential. Understanding how data modifications impact model performance improvements and potential biases and promoting a reliable pipeline is mandatory for developing robust machine learning solutions. Isolation Forest (iForest) is a widely used technique for outlier detection that performs well. Its effectiveness increases with the number of tree-based learners. However, this also complicates the explanation of outlier selection and the decision boundaries for inliers. This research introduces a novel Explainable AI (XAI) method, tackling the problem of global explainability. In detail, it aims to offer a global explanation for outlier detection to address its opaque nature. Our approach is based on the Decision Predicate Graph (DPG), which clarifies the logic of ensemble methods and provides both insights and a graph-based metric to explain how samples are identified as outliers using the proposed Inlier-Outlier Propagation Score (IOP-Score). Our proposal enhances iForest's explainability and provides a comprehensive view of the decision-making process, detailing which features contribute to outlier identification and how the model utilizes them. This method advances the state-of-the-art by providing insights into decision boundaries and a comprehensive view of holistic feature usage in outlier identification. -- thus promoting a fully explainable machine learning pipeline. 

**Abstract (ZH)**: 现代机器学习中解释预测模型的需求已经得到确立。然而，除了模型可解释性之外，理解预处理方法同样至关重要。理解数据修改如何影响模型性能改进和潜在偏见，并促进可靠的工作流程是开发稳健机器学习解决方案的必要条件。孤立森林（iForest）是一种广泛使用的离群点检测技术，性能良好。其有效性随树基学习器数量的增加而提高。然而，这也使得解释离群点选择及其内点的决策边界变得复杂。本研究引入了一种新的可解释人工智能（XAI）方法，旨在解决全局解释性的问题。具体而言，该方法旨在提供一种全局解释，以应对离群点检测的不透明性。我们的方法基于决策谓词图（DPG），阐明了集成方法的逻辑，并提供了有关样本如何被识别为离群点的图谱解释及其提出的内点-离群点传播得分（IOP-Score）。我们的提议增强了iForest的可解释性，并为决策过程提供了全面视角，详细说明了哪些特征对离群点识别有贡献以及模型是如何利用这些特征的。该方法通过提供决策边界的见解和在离群点识别中整体特征使用的全面视角，促进了最先进的可解释机器学习管道的发展。 

---
# An alignment safety case sketch based on debate 

**Title (ZH)**: 基于辩论的对齐安全案例草图 

**Authors**: Marie Davidsen Buhl, Jacob Pfau, Benjamin Hilton, Geoffrey Irving  

**Link**: [PDF](https://arxiv.org/pdf/2505.03989)  

**Abstract**: If AI systems match or exceed human capabilities on a wide range of tasks, it may become difficult for humans to efficiently judge their actions -- making it hard to use human feedback to steer them towards desirable traits. One proposed solution is to leverage another superhuman system to point out flaws in the system's outputs via a debate. This paper outlines the value of debate for AI safety, as well as the assumptions and further research required to make debate work. It does so by sketching an ``alignment safety case'' -- an argument that an AI system will not autonomously take actions which could lead to egregious harm, despite being able to do so. The sketch focuses on the risk of an AI R\&D agent inside an AI company sabotaging research, for example by producing false results. To prevent this, the agent is trained via debate, subject to exploration guarantees, to teach the system to be honest. Honesty is maintained throughout deployment via online training. The safety case rests on four key claims: (1) the agent has become good at the debate game, (2) good performance in the debate game implies that the system is mostly honest, (3) the system will not become significantly less honest during deployment, and (4) the deployment context is tolerant of some errors. We identify open research problems that, if solved, could render this a compelling argument that an AI system is safe. 

**Abstract (ZH)**: 如果AI系统在广泛的任务上匹配或超越人类能力，人类可能很难高效地判断其行为，从而难以利用人类反馈引导其发展出可取的特质。一种提出的解决方案是利用另一超人类系统通过辩论指出系统输出中的缺陷。本文概述了辩论在AI安全中的价值，以及使其有效工作的假设和进一步研究需求。这通过勾勒一个“对齐安全案例”来实现——一种论证尽管AI系统有能力采取可能导致严重危害的行动，但它不会自主采取这样的行动。该勾勒重点在于防范AI研发代理在AI公司内部搞破坏的风险，例如通过制造虚假结果。为防止这种情况，代理通过辩论训练，并受到探索保证的约束，以教系统保持诚实。在整个部署过程中通过在线训练维持诚实。安全案例基于四个关键主张：（1）代理擅长辩论游戏；（2）在辩论游戏中表现良好意味着系统主要是诚实的；（3）系统在部署过程中不会变得显著不那么诚实；（4）部署环境能容忍某些错误。我们确定了需要解决的开放研究问题，如果这些问题得以解决，这将使得该论证对一个AI系统是安全的具有说服力。 

---
# LogiDebrief: A Signal-Temporal Logic based Automated Debriefing Approach with Large Language Models Integration 

**Title (ZH)**: LogiDebrief: 基于信号时序逻辑并整合大型语言模型的自动化认知辅助方法 

**Authors**: Zirong Chen, Ziyan An, Jennifer Reynolds, Kristin Mullen, Stephen Martini, Meiyi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.03985)  

**Abstract**: Emergency response services are critical to public safety, with 9-1-1 call-takers playing a key role in ensuring timely and effective emergency operations. To ensure call-taking performance consistency, quality assurance is implemented to evaluate and refine call-takers' skillsets. However, traditional human-led evaluations struggle with high call volumes, leading to low coverage and delayed assessments. We introduce LogiDebrief, an AI-driven framework that automates traditional 9-1-1 call debriefing by integrating Signal-Temporal Logic (STL) with Large Language Models (LLMs) for fully-covered rigorous performance evaluation. LogiDebrief formalizes call-taking requirements as logical specifications, enabling systematic assessment of 9-1-1 calls against procedural guidelines. It employs a three-step verification process: (1) contextual understanding to identify responder types, incident classifications, and critical conditions; (2) STL-based runtime checking with LLM integration to ensure compliance; and (3) automated aggregation of results into quality assurance reports. Beyond its technical contributions, LogiDebrief has demonstrated real-world impact. Successfully deployed at Metro Nashville Department of Emergency Communications, it has assisted in debriefing 1,701 real-world calls, saving 311.85 hours of active engagement. Empirical evaluation with real-world data confirms its accuracy, while a case study and extensive user study highlight its effectiveness in enhancing call-taking performance. 

**Abstract (ZH)**: 一种基于信号时序逻辑和大规模语言模型的AI驱动紧急 debriefing 框架：LogiDebrief 

---
# The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete 

**Title (ZH)**: 故事的力量：叙述 priming 影响大语言模型代理的合作与竞争 

**Authors**: Gerrit Großmann, Larisa Ivanova, Sai Leela Poduru, Mohaddeseh Tabrizian, Islam Mesabah, David A. Selby, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03961)  

**Abstract**: According to Yuval Noah Harari, large-scale human cooperation is driven by shared narratives that encode common beliefs and values. This study explores whether such narratives can similarly nudge LLM agents toward collaboration. We use a finitely repeated public goods game in which LLM agents choose either cooperative or egoistic spending strategies. We prime agents with stories highlighting teamwork to different degrees and test how this influences negotiation outcomes. Our experiments explore four questions:(1) How do narratives influence negotiation behavior? (2) What differs when agents share the same story versus different ones? (3) What happens when the agent numbers grow? (4) Are agents resilient against self-serving negotiators? We find that story-based priming significantly affects negotiation strategies and success rates. Common stories improve collaboration, benefiting each agent. By contrast, priming agents with different stories reverses this effect, and those agents primed toward self-interest prevail. We hypothesize that these results carry implications for multi-agent system design and AI alignment. 

**Abstract (ZH)**: 根据尤瓦尔·诺亚·哈拉里观点，大规模人类合作由共享叙事驱动，这些叙事编码了共同的信念和价值观。本研究探讨此类叙事是否能类似地促使LLM代理趋向合作。我们使用有限重复的公共品博弈实验，其中LLM代理选择合作或自私的支出策略。我们用不同程度强调团队合作的故事对代理进行预处理，并测试这如何影响谈判结果。本研究探讨四个问题：(1) 故事如何影响谈判行为？(2) 当代理共享相同的故事还是不同故事时，结果有何不同？(3) 随着代理数量的增长，会发生什么？(4) 代理是否能抵御自私的谈判者？我们发现，基于故事的预处理显著影响了谈判策略和成功率。共同的故事提高了合作水平，使每个代理受益。相反，使用不同故事的预处理逆转了这一效果，那些被引导为自私的故事的代理占上风。我们假设这些结果对多代理系统设计和AI对齐有重要意义。 

---
# Frog Soup: Zero-Shot, In-Context, and Sample-Efficient Frogger Agents 

**Title (ZH)**: 青蛙汤：零样本、上下文相关且样本高效的青蛙 Agents 

**Authors**: Xiang Li, Yiyang Hao, Doug Fulop  

**Link**: [PDF](https://arxiv.org/pdf/2505.03947)  

**Abstract**: One of the primary aspirations in reinforcement learning research is developing general-purpose agents capable of rapidly adapting to and mastering novel tasks. While RL gaming agents have mastered many Atari games, they remain slow and costly to train for each game. In this work, we demonstrate that latest reasoning LLMs with out-of-domain RL post-training can play a challenging Atari game called Frogger under a zero-shot setting. We then investigate the effect of in-context learning and the amount of reasoning effort on LLM performance. Lastly, we demonstrate a way to bootstrap traditional RL method with LLM demonstrations, which significantly improves their performance and sample efficiency. Our implementation is open sourced at this https URL. 

**Abstract (ZH)**: 强化学习研究中的一项主要目标是开发能够快速适应并掌握新型任务的一般用途代理。尽管RL游戏代理已经掌握了许多Atari游戏，但它们在每个游戏中进行训练仍显得缓慢且成本较高。在本文中，我们展示了最新推理大语言模型在域外RL训练后，在零样本设置下能够玩一个名为Frogger的具有挑战性的Atari游戏。然后，我们调查了上下文学习和推理努力对大语言模型性能的影响。最后，我们展示了如何通过大语言模型的示范来提升传统RL方法，这显著提高了其性能和样本效率。我们的实现已在以下链接开源：this https URL。 

---
# GRAML: Dynamic Goal Recognition As Metric Learning 

**Title (ZH)**: GRAML: 动态目标识别作为元度学习 

**Authors**: Matan Shamir, Reuth Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.03941)  

**Abstract**: Goal Recognition (GR) is the problem of recognizing an agent's objectives based on observed actions. Recent data-driven approaches for GR alleviate the need for costly, manually crafted domain models. However, these approaches can only reason about a pre-defined set of goals, and time-consuming training is needed for new emerging goals. To keep this model-learning automated while enabling quick adaptation to new goals, this paper introduces GRAML: Goal Recognition As Metric Learning. GRAML uses a Siamese network to treat GR as a deep metric learning task, employing an RNN that learns a metric over an embedding space, where the embeddings for observation traces leading to different goals are distant, and embeddings of traces leading to the same goals are close. This metric is especially useful when adapting to new goals, even if given just one example observation trace per goal. Evaluated on a versatile set of environments, GRAML shows speed, flexibility, and runtime improvements over the state-of-the-art GR while maintaining accurate recognition. 

**Abstract (ZH)**: Goal Recognition as Metric Learning (GRAML) 

---
# Design description of Wisdom Computing Persperctive 

**Title (ZH)**: 智慧计算视角下的设计描述 

**Authors**: TianYi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03800)  

**Abstract**: This course design aims to develop and research a handwriting matrix recognition and step-by-step visual calculation process display system, addressing the issue of abstract formulas and complex calculation steps that students find difficult to understand when learning mathematics. By integrating artificial intelligence with visualization animation technology, the system enhances precise recognition of handwritten matrix content through the introduction of Mamba backbone networks, completes digital extraction and matrix reconstruction using the YOLO model, and simultaneously combines CoordAttention coordinate attention mechanisms to improve the accurate grasp of character spatial positions. The calculation process is demonstrated frame by frame through the Manim animation engine, vividly showcasing each mathematical calculation step, helping students intuitively understand the intrinsic logic of mathematical operations. Through dynamically generating animation processes for different computational tasks, the system exhibits high modularity and flexibility, capable of generating various mathematical operation examples in real-time according to student needs. By innovating human-computer interaction methods, it brings mathematical calculation processes to life, helping students bridge the gap between knowledge and understanding on a deeper level, ultimately achieving a learning experience where "every step is understood." The system's scalability and interactivity make it an intuitive, user-friendly, and efficient auxiliary tool in education. 

**Abstract (ZH)**: 本课程设计旨在开发和研究一种手写矩阵识别及逐步可视化计算过程展示系统，解决学生在学习数学时遇到的抽象公式和复杂的计算步骤难以理解的问题。通过将人工智能与可视化动画技术相结合，系统利用Mamba骨干网络提高手写矩阵内容的精确识别，借助YOLO模型完成数字提取和矩阵重构，并结合CoordAttention坐标注意力机制以提高字符空间位置的准确把握。计算过程通过Manim动画引擎逐帧展示，生动呈现每一个数学计算步骤，帮助学生直观理解数学运算的内在逻辑。通过动态生成不同的计算任务动画过程，系统展现出高度的模块化和灵活性，能够根据学生需求实时生成各种数学运算示例。通过创新人机交互方式，系统使数学计算过程栩栩如生，帮助学生在更深层次上弥合知识与理解之间的差距，最终实现“每一步都理解”的学习体验。该系统的可扩展性和交互性使其成为教育中直观、用户友好且高效的辅助工具。 

---
# Proceedings of 1st Workshop on Advancing Artificial Intelligence through Theory of Mind 

**Title (ZH)**: 第一届关于通过理论心智促进人工智能研讨会论文集 

**Authors**: Mouad Abrini, Omri Abend, Dina Acklin, Henny Admoni, Gregor Aichinger, Nitay Alon, Zahra Ashktorab, Ashish Atreja, Moises Auron, Alexander Aufreiter, Raghav Awasthi, Soumya Banerjee, Joe M. Barnby, Rhea Basappa, Severin Bergsmann, Djallel Bouneffouf, Patrick Callaghan, Marc Cavazza, Thierry Chaminade, Sonia Chernova, Mohamed Chetouan, Moumita Choudhury, Axel Cleeremans, Jacek B. Cywinski, Fabio Cuzzolin, Hokin Deng, N'yoma Diamond, Camilla Di Pasquasio, Guillaume Dumas, Max van Duijn, Mahapatra Dwarikanath, Qingying Gao, Ashok Goel, Rebecca Goldstein, Matthew Gombolay, Gabriel Enrique Gonzalez, Amar Halilovic, Tobias Halmdienst, Mahimul Islam, Julian Jara-Ettinger, Natalie Kastel, Renana Keydar, Ashish K. Khanna, Mahdi Khoramshahi, JiHyun Kim, MiHyeon Kim, YoungBin Kim, Senka Krivic, Nikita Krasnytskyi, Arun Kumar, JuneHyoung Kwon, Eunju Lee, Shane Lee, Peter R. Lewis, Xue Li, Yijiang Li, Michal Lewandowski, Nathan Lloyd, Matthew B. Luebbers, Dezhi Luo, Haiyun Lyu, Dwarikanath Mahapatra, Kamal Maheshwari, Mallika Mainali, Piyush Mathur, Patrick Mederitsch, Shuwa Miura, Manuel Preston de Miranda, Reuth Mirsky, Shreya Mishra, Nina Moorman, Katelyn Morrison, John Muchovej, Bernhard Nessler, Felix Nessler, Hieu Minh Jord Nguyen, Abby Ortego, Francis A. Papay, Antoine Pasquali, Hamed Rahimi, Charumathi Raghu, Amanda Royka, Stefan Sarkadi, Jaelle Scheuerman, Simon Schmid, Paul Schrater, Anik Sen, Zahra Sheikhbahaee, Ke Shi, Reid Simmons, Nishant Singh, Mason O. Smith, Ramira van der Meulen, Anthia Solaki, Haoran Sun, Viktor Szolga, Matthew E. Taylor, Travis Taylor, Sanne Van Waveren, Juan David Vargas  

**Link**: [PDF](https://arxiv.org/pdf/2505.03770)  

**Abstract**: This volume includes a selection of papers presented at the Workshop on Advancing Artificial Intelligence through Theory of Mind held at AAAI 2025 in Philadelphia US on 3rd March 2025. The purpose of this volume is to provide an open access and curated anthology for the ToM and AI research community. 

**Abstract (ZH)**: 本volume收录了于2025年3月3日在美国费城举行的第30届AAAI会议上的“通过理论思维促进人工智能”研讨会 presentations，旨在为理论思维和人工智能研究社区提供开放访问和精选的合集。 

---
# EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning 

**Title (ZH)**: EchoInk-R1：通过强化学习探索多模态LLM中的音视频推理 

**Authors**: Zhenghao Xing, Xiaowei Hu, Chi-Wing Fu, Wenhai Wang, Jifeng Dai, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04623)  

**Abstract**: Multimodal large language models (MLLMs) have advanced perception across text, vision, and audio, yet they often struggle with structured cross-modal reasoning, particularly when integrating audio and visual signals. We introduce EchoInk-R1, a reinforcement learning framework that enhances such reasoning in MLLMs. Built upon the Qwen2.5-Omni-7B foundation and optimized with Group Relative Policy Optimization (GRPO), EchoInk-R1 tackles multiple-choice question answering over synchronized audio-image pairs. To enable this, we curate AVQA-R1-6K, a dataset pairing such audio-image inputs with multiple-choice questions derived from OmniInstruct-v1. EchoInk-R1-7B achieves 85.77% accuracy on the validation set, outperforming the base model, which scores 80.53%, using only 562 reinforcement learning steps. Beyond accuracy, EchoInk-R1 demonstrates reflective reasoning by revisiting initial interpretations and refining responses when facing ambiguous multimodal inputs. These results suggest that lightweight reinforcement learning fine-tuning enhances cross-modal reasoning in MLLMs. EchoInk-R1 is the first framework to unify audio, visual, and textual modalities for general open-world reasoning via reinforcement learning. Code and data are publicly released to facilitate further research. 

**Abstract (ZH)**: multimodal大型语言模型（MLLMs）已在文本、视觉和音频感知方面取得了进展，但在音频和视觉信号综合的结构化跨模态推理方面往往表现出色不足。我们介绍了EchoInk-R1，一种增强MLLMs此类推理的强化学习框架。基于Qwen2.5-Omni-7B基础模型并使用Group Relative Policy Optimization (GRPO)进行优化，EchoInk-R1解决了同步音频-图像对的多项选择题回答问题。为了实现这一点，我们创建了AVQA-R1-6K数据集，将此类音频-图像输入与来自OmniInstruct-v1的多项选择题配对。EchoInk-R1-7B在验证集上的准确率达到85.77%，比仅使用562步强化学习步骤的基模型80.53%的准确率更高。除了准确性外，EchoInk-R1还展示了反思性推理，能够在面对模态性输入时重新评估初始解释并完善响应。这些结果表明，轻量级的强化学习微调可以增强MLLMs的跨模态推理能力。EchoInk-R1是首个通过强化学习统一音频、视觉和文本模态进行一般开放世界推理的框架。代码和数据已公开发布，以促进进一步研究。 

---
# Score Distillation Sampling for Audio: Source Separation, Synthesis, and Beyond 

**Title (ZH)**: 音频的评分蒸馏采样：源分离、合成与 beyond 

**Authors**: Jessie Richter-Powell, Antonio Torralba, Jonathan Lorraine  

**Link**: [PDF](https://arxiv.org/pdf/2505.04621)  

**Abstract**: We introduce Audio-SDS, a generalization of Score Distillation Sampling (SDS) to text-conditioned audio diffusion models. While SDS was initially designed for text-to-3D generation using image diffusion, its core idea of distilling a powerful generative prior into a separate parametric representation extends to the audio domain. Leveraging a single pretrained model, Audio-SDS enables a broad range of tasks without requiring specialized datasets. In particular, we demonstrate how Audio-SDS can guide physically informed impact sound simulations, calibrate FM-synthesis parameters, and perform prompt-specified source separation. Our findings illustrate the versatility of distillation-based methods across modalities and establish a robust foundation for future work using generative priors in audio tasks. 

**Abstract (ZH)**: 我们介绍Audio-SDS，这是一种将Score Distillation Sampling (SDS)推广至文本条件音頻扩散模型的方法。尽管SDS最初设计用于基于图像的3D生成，但其核心思想——将强大的生成先验知识提炼为独立的参数表示——适用于音頻领域。通过单一预训练模型，Audio-SDS能够支持广泛的任务，无需专门的数据集。特别是在物理信息冲击声模拟、FM合成参数校准和按提示进行源分离方面，我们展示了Audio-SDS的应用。我们的研究成果证明了基于提炼方法在不同模态下的灵活性，并为未来在音頻任务中使用生成先验的工作奠定了坚实的基础。 

---
# WATCH: Weighted Adaptive Testing for Changepoint Hypotheses via Weighted-Conformal Martingales 

**Title (ZH)**: WATCH: 加权自适应测试在加权齐性库恩 martingales 作用下的变化点假设检验 

**Authors**: Drew Prinster, Xing Han, Anqi Liu, Suchi Saria  

**Link**: [PDF](https://arxiv.org/pdf/2505.04608)  

**Abstract**: Responsibly deploying artificial intelligence (AI) / machine learning (ML) systems in high-stakes settings arguably requires not only proof of system reliability, but moreover continual, post-deployment monitoring to quickly detect and address any unsafe behavior. Statistical methods for nonparametric change-point detection -- especially the tools of conformal test martingales (CTMs) and anytime-valid inference -- offer promising approaches to this monitoring task. However, existing methods are restricted to monitoring limited hypothesis classes or ``alarm criteria,'' such as data shifts that violate certain exchangeability assumptions, or do not allow for online adaptation in response to shifts. In this paper, we expand the scope of these monitoring methods by proposing a weighted generalization of conformal test martingales (WCTMs), which lay a theoretical foundation for online monitoring for any unexpected changepoints in the data distribution while controlling false-alarms. For practical applications, we propose specific WCTM algorithms that accommodate online adaptation to mild covariate shifts (in the marginal input distribution) while raising alarms in response to more severe shifts, such as concept shifts (in the conditional label distribution) or extreme (out-of-support) covariate shifts that cannot be easily adapted to. On real-world datasets, we demonstrate improved performance relative to state-of-the-art baselines. 

**Abstract (ZH)**: 负责任地在高风险环境中部署人工智能（AI）/机器学习（ML）系统不仅需要系统的可靠性证明，而且还要求在部署后进行持续监控，以快速检测和解决任何不安全的行为。通过非参数变化点检测的统计方法——尤其是符合性测试鞅（CTMs）和任意时有效的推断工具——为这一监测任务提供了有前景的方法。然而，现有方法限制在监测有限的假设类别或“警报标准”，如违反某些交换性假设的数据偏移，或者不允许在偏移发生时进行在线适应。在本文中，我们通过提出加权的符合性测试鞅（WCTMs）的一般化形式，扩展了这些监测方法的应用范围，为数据分布中任何意外的变化点提供在线监控的基础，同时控制误报。针对实际应用，我们提出了具体的WCTM算法，能够适应轻微边缘输入分布的变化，并在更严重的偏移如条件标签分布的变化或超出支持范围的极端变化发生时发出警报。我们在实际数据集上的实验结果表明，与现有最佳基线相比，性能有所提升。 

---
# AI Governance to Avoid Extinction: The Strategic Landscape and Actionable Research Questions 

**Title (ZH)**: AI治理以避免灭绝：战略格局与可操作的研究问题 

**Authors**: Peter Barnett, Aaron Scher  

**Link**: [PDF](https://arxiv.org/pdf/2505.04592)  

**Abstract**: Humanity appears to be on course to soon develop AI systems that substantially outperform human experts in all cognitive domains and activities. We believe the default trajectory has a high likelihood of catastrophe, including human extinction. Risks come from failure to control powerful AI systems, misuse of AI by malicious rogue actors, war between great powers, and authoritarian lock-in. This research agenda has two aims: to describe the strategic landscape of AI development and to catalog important governance research questions. These questions, if answered, would provide important insight on how to successfully reduce catastrophic risks.
We describe four high-level scenarios for the geopolitical response to advanced AI development, cataloging the research questions most relevant to each. Our favored scenario involves building the technical, legal, and institutional infrastructure required to internationally restrict dangerous AI development and deployment (which we refer to as an Off Switch), which leads into an internationally coordinated Halt on frontier AI activities at some point in the future. The second scenario we describe is a US National Project for AI, in which the US Government races to develop advanced AI systems and establish unilateral control over global AI development. We also describe two additional scenarios: a Light-Touch world similar to that of today and a Threat of Sabotage situation where countries use sabotage and deterrence to slow AI development.
In our view, apart from the Off Switch and Halt scenario, all of these trajectories appear to carry an unacceptable risk of catastrophic harm. Urgent action is needed from the US National Security community and AI governance ecosystem to answer key research questions, build the capability to halt dangerous AI activities, and prepare for international AI agreements. 

**Abstract (ZH)**: 人类似乎即将开发出在所有认知领域和活动中显著超越人类专家的AI系统。我们认为，默认轨迹有很大的可能带来灾难性后果，包括人类灭绝。这些风险源于无法控制强大的AI系统、恶意行为者滥用AI、大国之间的战争以及专制锁定。这项研究议程有两个目标：描述AI发展的战略格局，并列出重要的治理研究问题。如果这些问题得以解答，将会为如何成功降低灾难性风险提供重要见解。

我们描述了四种 geopolitics 对先进AI发展的回应情景，并列出了每个情景中最相关的研究问题。我们偏好的情景是建立国际上限制危险AI开发和部署的技术、法律和制度基础设施（我们称为“关断开关”），并最终在未来某个时候推动国际协调一致的暂停前沿AI活动。我们描述的第二种情景是美国AI国家级项目，其中美国政府将竞相开发先进AI系统并建立对全球AI发展的单方面控制。我们还描述了另外两种情景：一个类似当今世界的轻触监管世界和一个破坏威胁情景，其中国家使用破坏和威慑手段减缓AI的发展。

在我们看来，除了“关断开关”和暂停情景外，所有这些轨迹似乎都带来了不可接受的灾难性危害风险。需要美国国家安全社区和AI治理体系的紧急行动来解答关键研究问题、建立停止危险AI活动的能力，并为国际AI协议的制定做好准备。 

---
# Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization 

**Title (ZH)**: 以火制火：通过奖励中和防御恶意RL微调 

**Authors**: Wenjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04578)  

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models. 

**Abstract (ZH)**: 强化学习（RL）微调虽然能改造大规模语言模型，但也引入了一个我们通过实验验证的漏洞：我们的实验显示，恶意的RL微调以令人惊讶的效率拆解了安全防护措施，仅需50步和少量对抗提示，有害行为从0-2迅速升级到7-9。这种攻击途径特别威胁具有参数级访问权限的开源模型。现有的针对监督微调的防御措施无法抵抗RL的动态反馈机制。我们引入了奖励中和（Reward Neutralization），这是首个专门针对RL微调攻击的防御框架，通过建立简洁的拒绝模式使其上的恶意奖励信号无效。我们的方法训练模型产生最小信息量的拒绝，攻击者无法利用，系统地中和了向有害输出优化的尝试。实验验证显示，我们的方法在200步攻击后保持较低的有害评分（不超过2分），而标准模型迅速恶化。这项工作首次提供了构建性的证明，即针对日益可访问的RL攻击实现鲁棒防御是可行的，填补了开源权重模型的关键安全空白。 

---
# Purity Law for Generalizable Neural TSP Solvers 

**Title (ZH)**: 普遍性原则：通用神经TSP求解器 

**Authors**: Wenzhao Liu, Haoran Li, Congying Han, Zicheng Zhang, Anqi Li, Tiande Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.04558)  

**Abstract**: Achieving generalization in neural approaches across different scales and distributions remains a significant challenge for the Traveling Salesman Problem~(TSP). A key obstacle is that neural networks often fail to learn robust principles for identifying universal patterns and deriving optimal solutions from diverse instances. In this paper, we first uncover Purity Law (PuLa), a fundamental structural principle for optimal TSP solutions, defining that edge prevalence grows exponentially with the sparsity of surrounding vertices. Statistically validated across diverse instances, PuLa reveals a consistent bias toward local sparsity in global optima. Building on this insight, we propose Purity Policy Optimization~(PUPO), a novel training paradigm that explicitly aligns characteristics of neural solutions with PuLa during the solution construction process to enhance generalization. Extensive experiments demonstrate that PUPO can be seamlessly integrated with popular neural solvers, significantly enhancing their generalization performance without incurring additional computational overhead during inference. 

**Abstract (ZH)**: 在不同规模和分布下实现神经方法在旅行销售商问题（TSP）中的泛化仍然是一个重大挑战。一个关键障碍是神经网络往往无法学习到识别普遍模式并从多种实例中推导出最优解的稳健原则。在本文中，我们首先揭示了纯度定律（PuLa），这是一种基本的结构原则，定义了边的频度随着周围顶点稀疏性的增加而呈指数增长。PuLa在多种实例上通过统计验证，揭示了全局最优中的局部稀疏性的一致偏差。基于这一洞见，我们提出了一种新的训练范式——纯度策略优化（PUPO），该范式在解决方案构建过程中明确地将神经解决方案的特征与PuLa对齐，以提高泛化能力。广泛的实验表明，PUPO可以无缝集成到流行的神经求解器中，显著增强其泛化性能，而在推理过程中不会增加额外的计算开销。 

---
# Risk-sensitive Reinforcement Learning Based on Convex Scoring Functions 

**Title (ZH)**: 基于凸评分函数的风险敏感强化学习 

**Authors**: Shanyu Han, Yang Liu, Xiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04553)  

**Abstract**: We propose a reinforcement learning (RL) framework under a broad class of risk objectives, characterized by convex scoring functions. This class covers many common risk measures, such as variance, Expected Shortfall, entropic Value-at-Risk, and mean-risk utility. To resolve the time-inconsistency issue, we consider an augmented state space and an auxiliary variable and recast the problem as a two-state optimization problem. We propose a customized Actor-Critic algorithm and establish some theoretical approximation guarantees. A key theoretical contribution is that our results do not require the Markov decision process to be continuous. Additionally, we propose an auxiliary variable sampling method inspired by the alternating minimization algorithm, which is convergent under certain conditions. We validate our approach in simulation experiments with a financial application in statistical arbitrage trading, demonstrating the effectiveness of the algorithm. 

**Abstract (ZH)**: 我们提出了一种广义风险目标下的强化学习框架，由凸评分函数 characterization。该框架涵盖了许多常见风险度量，如方差、预期短falls、熵值 at-risk、均值-风险效用。为解决时间不一致性问题，我们考虑了扩充状态空间和辅助变量，并将问题重新表述为两状态优化问题。我们提出了一种定制化的Actor-Critic算法，并建立了若干理论近似保证。一个重要的理论贡献是，我们的结果不要求马尔可夫决策过程是连续的。此外，我们提出了一种受交替最小化算法启发的辅助变量采样方法，在某些条件下该方法是收敛的。我们在统计套利交易的仿真实验中验证了该方法，展示了该算法的有效性。 

---
# Overcoming Data Scarcity in Generative Language Modelling for Low-Resource Languages: A Systematic Review 

**Title (ZH)**: 克服生成语言模型中低资源语言数据稀缺问题：一项系统性审查 

**Authors**: Josh McGiff, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2505.04531)  

**Abstract**: Generative language modelling has surged in popularity with the emergence of services such as ChatGPT and Google Gemini. While these models have demonstrated transformative potential in productivity and communication, they overwhelmingly cater to high-resource languages like English. This has amplified concerns over linguistic inequality in natural language processing (NLP). This paper presents the first systematic review focused specifically on strategies to address data scarcity in generative language modelling for low-resource languages (LRL). Drawing from 54 studies, we identify, categorise and evaluate technical approaches, including monolingual data augmentation, back-translation, multilingual training, and prompt engineering, across generative tasks. We also analyse trends in architecture choices, language family representation, and evaluation methods. Our findings highlight a strong reliance on transformer-based models, a concentration on a small subset of LRLs, and a lack of consistent evaluation across studies. We conclude with recommendations for extending these methods to a wider range of LRLs and outline open challenges in building equitable generative language systems. Ultimately, this review aims to support researchers and developers in building inclusive AI tools for underrepresented languages, a necessary step toward empowering LRL speakers and the preservation of linguistic diversity in a world increasingly shaped by large-scale language technologies. 

**Abstract (ZH)**: 生成语言模型因ChatGPT和Google Gemini等服务的出现而日益流行。尽管这些模型在提高生产力和沟通方面展现出变革性的潜力，但它们主要服务于如英语之类的高资源语言。这加剧了自然语言处理（NLP）领域语言不平等的问题。本文首次系统性地回顾了针对低资源语言（LRL）生成语言模型数据稀缺问题的策略。我们从54篇研究中识别、分类并评估了包括单语数据增强、回译、多语言训练和提示工程在内的技术方法，涵盖了各种生成任务。我们还分析了架构选择、语言家族表示和评估方法的趋势。研究结果突出显示了对基于转子器模型的强烈依赖、对少数几种LRL的集中关注以及研究间缺乏一致的评估。我们提出了将这些方法扩展到更广泛的LRL范围的建议，并概述了构建公平的生成语言系统所面临的开放性挑战。最终，本文旨在支持研究者和开发人员为未充分代表的语言构建包容性AI工具，这是使LRL使用者受益并保护语言多样性的重要步骤。 

---
# DFVO: Learning Darkness-free Visible and Infrared Image Disentanglement and Fusion All at Once 

**Title (ZH)**: DFVO: 一次学习无暗场的可见光和红外图像解耦与融合 

**Authors**: Qi Zhou, Yukai Shi, Xiaojun Yang, Xiaoyu Xian, Lunjia Liao, Ruimao Zhang, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04526)  

**Abstract**: Visible and infrared image fusion is one of the most crucial tasks in the field of image fusion, aiming to generate fused images with clear structural information and high-quality texture features for high-level vision tasks. However, when faced with severe illumination degradation in visible images, the fusion results of existing image fusion methods often exhibit blurry and dim visual effects, posing major challenges for autonomous driving. To this end, a Darkness-Free network is proposed to handle Visible and infrared image disentanglement and fusion all at Once (DFVO), which employs a cascaded multi-task approach to replace the traditional two-stage cascaded training (enhancement and fusion), addressing the issue of information entropy loss caused by hierarchical data transmission. Specifically, we construct a latent-common feature extractor (LCFE) to obtain latent features for the cascaded tasks strategy. Firstly, a details-extraction module (DEM) is devised to acquire high-frequency semantic information. Secondly, we design a hyper cross-attention module (HCAM) to extract low-frequency information and preserve texture features from source images. Finally, a relevant loss function is designed to guide the holistic network learning, thereby achieving better image fusion. Extensive experiments demonstrate that our proposed approach outperforms state-of-the-art alternatives in terms of qualitative and quantitative evaluations. Particularly, DFVO can generate clearer, more informative, and more evenly illuminated fusion results in the dark environments, achieving best performance on the LLVIP dataset with 63.258 dB PSNR and 0.724 CC, providing more effective information for high-level vision tasks. Our code is publicly accessible at this https URL. 

**Abstract (ZH)**: 可见光和红外图像融合是图像融合领域中最关键的任务之一，旨在生成具有清晰结构信息和高质量纹理特征的融合图像，以满足高级视觉任务的需求。然而，当面对可见光图像中的严重光照退化时，现有图像融合方法的融合结果通常表现出模糊和暗淡的视觉效果，这给自主驾驶带来了重大挑战。为解决这一问题，提出了一种黑暗无感网络（DFVO）来一次性处理可见光和红外图像的解耦合与融合，该网络采用级联多任务方法替代传统的两阶段级联训练（增强和融合），以解决由分层数据传输引起的信息熵损失问题。具体地，构建了一个潜在通用特征提取器（LCFE）以获取级联任务策略的潜在特征。首先，设计了一个细节提取模块（DEM）以获取高频语义信息。其次，设计了一个超交叉注意模块（HCAM）以提取低频信息并保留源图像的纹理特征。最后，设计了一个相关损失函数以指导整体网络学习，从而实现更好的图像融合。 extensive实验表明，与现有先进方法相比，我们提出的方法在定性和定量评估中均表现出 superiority。特别是，DFVO可以在黑暗环境中生成更清晰、更具信息量且亮度更均匀的融合结果，并在LLVIP数据集上实现了63.258 dB PSNR和0.724 CC的最佳性能，为高级视觉任务提供更有效信息。源代码已公开。 

---
# Defining and Quantifying Creative Behavior in Popular Image Generators 

**Title (ZH)**: 定义并量化流行图像生成器中的创造性行为 

**Authors**: Aditi Ramaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2505.04497)  

**Abstract**: Creativity of generative AI models has been a subject of scientific debate in the last years, without a conclusive answer. In this paper, we study creativity from a practical perspective and introduce quantitative measures that help the user to choose a suitable AI model for a given task. We evaluated our measures on a number of popular image-to-image generation models, and the results of this suggest that our measures conform to human intuition. 

**Abstract (ZH)**: 生成AI模型的创造力在近年来的科学讨论中一直存在争议，缺乏定论。从实用角度研究创造力并引入定量指标以帮助用户选择适合的任务模型——基于这一视角，我们评估了我们的指标在多个流行图像到图像生成模型上的表现，结果表明我们的指标符合人类直觉。 

---
# Model-Based AI planning and Execution Systems for Robotics 

**Title (ZH)**: 基于模型的AI规划与执行系统在机器人领域的应用 

**Authors**: Or Wertheim, Ronen I. Brafman  

**Link**: [PDF](https://arxiv.org/pdf/2505.04493)  

**Abstract**: Model-based planning and execution systems offer a principled approach to building flexible autonomous robots that can perform diverse tasks by automatically combining a host of basic skills. This idea is almost as old as modern robotics. Yet, while diverse general-purpose reasoning architectures have been proposed since, general-purpose systems that are integrated with modern robotic platforms have emerged only recently, starting with the influential ROSPlan system. Since then, a growing number of model-based systems for robot task-level control have emerged. In this paper, we consider the diverse design choices and issues existing systems attempt to address, the different solutions proposed so far, and suggest avenues for future development. 

**Abstract (ZH)**: 基于模型的规划与执行系统提供了一种原理性的方法，用于构建能够通过自动组合多种基本技能来执行多样化任务的灵活自主机器人。这一理念几乎与现代机器人技术一样古老。然而，尽管自那时以来提出了多种通用推理架构，将现代机器人平台与通用系统相结合的系统仅在ROSPlan系统之后才开始出现。自那时以来，越来越多的基于模型的机器人任务级控制系统已经涌现。在本文中，我们考虑了现有系统试图解决的各种设计选择和问题，不同的解决方案及其迄今为止的进展，并建议未来的开发方向。 

---
# "I Can See Forever!": Evaluating Real-time VideoLLMs for Assisting Individuals with Visual Impairments 

**Title (ZH)**: “我可以看到永远！”: 评估实时视频LLM辅助视觉障碍个体的技术 

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Zifan Peng, Yuemeng Zhao, Zichun Wang, Zeren Luo, Ruiting Zuo, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2505.04488)  

**Abstract**: The visually impaired population, especially the severely visually impaired, is currently large in scale, and daily activities pose significant challenges for them. Although many studies use large language and vision-language models to assist the blind, most focus on static content and fail to meet real-time perception needs in dynamic and complex environments, such as daily activities. To provide them with more effective intelligent assistance, it is imperative to incorporate advanced visual understanding technologies. Although real-time vision and speech interaction VideoLLMs demonstrate strong real-time visual understanding, no prior work has systematically evaluated their effectiveness in assisting visually impaired individuals. In this work, we conduct the first such evaluation. First, we construct a benchmark dataset (VisAssistDaily), covering three categories of assistive tasks for visually impaired individuals: Basic Skills, Home Life Tasks, and Social Life Tasks. The results show that GPT-4o achieves the highest task success rate. Next, we conduct a user study to evaluate the models in both closed-world and open-world scenarios, further exploring the practical challenges of applying VideoLLMs in assistive contexts. One key issue we identify is the difficulty current models face in perceiving potential hazards in dynamic environments. To address this, we build an environment-awareness dataset named SafeVid and introduce a polling mechanism that enables the model to proactively detect environmental risks. We hope this work provides valuable insights and inspiration for future research in this field. 

**Abstract (ZH)**: 视觉受损人群，尤其是重度视觉受损人群，目前规模庞大，日常活动对其构成重大挑战。尽管许多研究利用大规模语言模型和多模态模型来协助视障人士，大多数研究重点在于静态内容，未能满足在动态和复杂环境中（如日常活动）的实时感知需求。为了提供更多有效的智能辅助，有必要融入先进的视觉理解技术。虽然实时视觉和语音交互VideoLLMs展现了强大的实时视觉理解能力，但此前没有任何研究系统性地评估其在辅助视障人士中的有效性。在本工作中，我们首次进行了此类评价。首先，我们构建了一个基准数据集（VisAssistDaily），涵盖三种视障辅助任务类别：基本技能、家庭生活任务和社会生活任务。结果显示，GPT-4o 达到最高的任务成功率。随后，我们进行了一项用户研究，评估模型在闭世界和开放世界场景中的表现，进一步探讨将VideoLLMs应用于辅助情境的实际挑战。一个我们识别的关键问题是当前模型在动态环境中感知潜在风险的困难。为此，我们构建了一个环境感知数据集SafeVid，并引入了一种投票机制，使模型能够主动检测环境风险。我们希望本工作为未来该领域的研究提供有价值的见解和灵感。 

---
# Efficient Flow Matching using Latent Variables 

**Title (ZH)**: 高效流匹配使用潜在变量 

**Authors**: Anirban Samaddar, Yixuan Sun, Viktor Nilsson, Sandeep Madireddy  

**Link**: [PDF](https://arxiv.org/pdf/2505.04486)  

**Abstract**: Flow matching models have shown great potential in image generation tasks among probabilistic generative models. Building upon the ideas of continuous normalizing flows, flow matching models generalize the transport path of the diffusion models from a simple prior distribution to the data. Most flow matching models in the literature do not explicitly model the underlying structure/manifold in the target data when learning the flow from a simple source distribution like the standard Gaussian. This leads to inefficient learning, especially for many high-dimensional real-world datasets, which often reside in a low-dimensional manifold. Existing strategies of incorporating manifolds, including data with underlying multi-modal distribution, often require expensive training and hence frequently lead to suboptimal performance. To this end, we present \texttt{Latent-CFM}, which provides simplified training/inference strategies to incorporate multi-modal data structures using pretrained deep latent variable models. Through experiments on multi-modal synthetic data and widely used image benchmark datasets, we show that \texttt{Latent-CFM} exhibits improved generation quality with significantly less training ($\sim 50\%$ less in some cases) and computation than state-of-the-art flow matching models. Using a 2d Darcy flow dataset, we demonstrate that our approach generates more physically accurate samples than competitive approaches. In addition, through latent space analysis, we demonstrate that our approach can be used for conditional image generation conditioned on latent features. 

**Abstract (ZH)**: 基于流动匹配模型在概率生成模型中的图像生成任务中展示了巨大的潜力。基于连续正则化流动的思想，流动匹配模型将扩散模型的传输路径从简单的先验分布推广到数据分布。现有的大多数流动匹配模型在从标准高斯分布这样的简单源分布学习流动时，并不明确建模目标数据的潜在结构/流形。这导致了效率低下，尤其是在许多高维的实际数据集上，这些数据集往往存在于低维流形中。现有集成流形的策略，包括具有潜在多元分布的数据，通常需要昂贵的训练，因此经常导致次优性能。为了解决这一问题，我们提出了\texttt{Latent-CFM}，它提供了一种简化的训练/推理策略，使用预训练的深度潜变量模型来集成多元数据结构。通过在多元合成数据和广泛使用的图像基准数据集上的实验，我们展示了\texttt{Latent-CFM}在显著减少训练时间和计算的同时，生成质量有所提高。通过2D达西流动数据集，我们证明了我们的方法能够生成比竞争方法更符合物理特性的样本。此外，通过潜空间分析，我们展示了我们的方法可以用潜特征条件生成图像。 

---
# Spectral and Temporal Denoising for Differentially Private Optimization 

**Title (ZH)**: 差分隐私优化的谱域和时域去噪方法 

**Authors**: Hyeju Shin, Kyudan Jung, Seongwon Yun, Juyoung Yun  

**Link**: [PDF](https://arxiv.org/pdf/2505.04468)  

**Abstract**: This paper introduces the FFT-Enhanced Kalman Filter (FFTKF), a differentially private optimization method that addresses the challenge of preserving performance in DP-SGD, where added noise typically degrades model utility. FFTKF integrates frequency-domain noise shaping with Kalman filtering to enhance gradient quality while preserving $(\varepsilon, \delta)$-DP guarantees. It employs a high-frequency shaping mask in the Fourier domain to concentrate differential privacy noise in less informative spectral components, preserving low-frequency gradient signals. A scalar-gain Kalman filter with finite-difference Hessian approximation further refines the denoised gradients. With a per-iteration complexity of $\mathcal{O}(d \log d)$, FFTKF demonstrates improved test accuracy over DP-SGD and DiSK across MNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets using CNNs, Wide ResNets, and Vision Transformers. Theoretical analysis confirms that FFTKF maintains equivalent privacy guarantees while achieving a tighter privacy-utility trade-off through reduced noise and controlled bias. 

**Abstract (ZH)**: FFT增强卡尔曼滤波器：一种针对DP-SGD性能保护的差分隐私优化方法 

---
# Discriminative Ordering Through Ensemble Consensus 

**Title (ZH)**: 基于集成共识的鉴别性排序 

**Authors**: Louis Ohl, Fredrik Lindsten  

**Link**: [PDF](https://arxiv.org/pdf/2505.04464)  

**Abstract**: Evaluating the performance of clustering models is a challenging task where the outcome depends on the definition of what constitutes a cluster. Due to this design, current existing metrics rarely handle multiple clustering models with diverse cluster definitions, nor do they comply with the integration of constraints when available. In this work, we take inspiration from consensus clustering and assume that a set of clustering models is able to uncover hidden structures in the data. We propose to construct a discriminative ordering through ensemble clustering based on the distance between the connectivity of a clustering model and the consensus matrix. We first validate the proposed method with synthetic scenarios, highlighting that the proposed score ranks the models that best match the consensus first. We then show that this simple ranking score significantly outperforms other scoring methods when comparing sets of different clustering algorithms that are not restricted to a fixed number of clusters and is compatible with clustering constraints. 

**Abstract (ZH)**: 评估聚类模型的性能是一个具有挑战性的任务，Outcome取决于何为聚类的定义。由于这一设计，现有的度量标准很少能够处理具有多种聚类定义的多个聚类模型，也不符合当有约束条件时的集成。在这项工作中，我们从共识聚类中汲取灵感，假设一个聚类模型集合能够揭示数据中的隐藏结构。我们提出通过基于聚类模型连通性和共识矩阵之间距离的集成聚类来构建一个区分性排序。我们首先使用合成场景验证所提出的方法，表明提出的评分方法首先对与共识匹配最佳的模型进行排名。然后，我们展示这种简单的评分方法在比较不受固定聚类数限制的不同聚类算法集合时显著优于其他评分方法，并且兼容聚类约束。 

---
# A Survey on Temporal Interaction Graph Representation Learning: Progress, Challenges, and Opportunities 

**Title (ZH)**: Temporal Interaction图表示学习综述：进展、挑战与机遇 

**Authors**: Pengfei Jiao, Hongjiang Chen, Xuan Guo, Zhidong Zhao, Dongxiao He, Di Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04461)  

**Abstract**: Temporal interaction graphs (TIGs), defined by sequences of timestamped interaction events, have become ubiquitous in real-world applications due to their capability to model complex dynamic system behaviors. As a result, temporal interaction graph representation learning (TIGRL) has garnered significant attention in recent years. TIGRL aims to embed nodes in TIGs into low-dimensional representations that effectively preserve both structural and temporal information, thereby enhancing the performance of downstream tasks such as classification, prediction, and clustering within constantly evolving data environments. In this paper, we begin by introducing the foundational concepts of TIGs and emphasize the critical role of temporal dependencies. We then propose a comprehensive taxonomy of state-of-the-art TIGRL methods, systematically categorizing them based on the types of information utilized during the learning process to address the unique challenges inherent to TIGs. To facilitate further research and practical applications, we curate the source of datasets and benchmarks, providing valuable resources for empirical investigations. Finally, we examine key open challenges and explore promising research directions in TIGRL, laying the groundwork for future advancements that have the potential to shape the evolution of this field. 

**Abstract (ZH)**: 时间交互图（TIGs）由时间戳标注的交互事件序列定义，因其能够 modeling 复杂动态系统行为而在实际应用中无处不在。因此，时间交互图表示学习（TIGRL）近年来引起了广泛关注。TIGRL旨在将时间交互图中的节点嵌入到低维表示中，有效地保留结构和时间信息，从而在不断变化的数据环境中增强分类、预测和聚类等下游任务的性能。在本文中，我们首先介绍时间交互图的基本概念，并强调时间依赖关系的关键作用。然后，我们提出了一种关于最先进的 TIGRL 方法的全面分类框架，基于学习过程中利用的信息类型系统地对其进行分类，以应对时间交互图固有的独特挑战。为了促进进一步的研究和实际应用，我们汇总了数据集和基准数据的来源，提供了实证研究中宝贵的资源。最后，我们探讨了 TIGRL 中的关键开放挑战，并探索了有希望的研究方向，为该领域的未来进步奠定了基础。 

---
# Automatic Music Transcription using Convolutional Neural Networks and Constant-Q transform 

**Title (ZH)**: 使用卷积神经网络和常数Q变换的自动音乐转录 

**Authors**: Yohannis Telila, Tommaso Cucinotta, Davide Bacciu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04451)  

**Abstract**: Automatic music transcription (AMT) is the problem of analyzing an audio recording of a musical piece and detecting notes that are being played. AMT is a challenging problem, particularly when it comes to polyphonic music. The goal of AMT is to produce a score representation of a music piece, by analyzing a sound signal containing multiple notes played simultaneously. In this work, we design a processing pipeline that can transform classical piano audio files in .wav format into a music score representation. The features from the audio signals are extracted using the constant-Q transform, and the resulting coefficients are used as an input to the convolutional neural network (CNN) model. 

**Abstract (ZH)**: 自动音乐转录（AMT）是分析音乐乐谱录音并检测正在演奏的音符的问题。AMT 是一个具有挑战性的问题，尤其是在处理多声部音乐时。AMT 的目标是通过分析包含多个同时演奏音符的声信号来生成音乐作品的乐谱表示。在本工作中，我们设计了一个处理管道，可以将 Classical Piano 音频文件转换为 .wav 格式并转化为乐谱表示。从声信号中提取的特征使用常数-Q 变换获取，最终系数作为卷积神经网络（CNN）模型的输入。 

---
# FedBWO: Enhancing Communication Efficiency in Federated Learning 

**Title (ZH)**: FedBWO: 提高联邦学习中的通信效率 

**Authors**: Vahideh Hayyolalam, Öznur Özkasap  

**Link**: [PDF](https://arxiv.org/pdf/2505.04435)  

**Abstract**: Federated Learning (FL) is a distributed Machine Learning (ML) setup, where a shared model is collaboratively trained by various clients using their local datasets while keeping the data private. Considering resource-constrained devices, FL clients often suffer from restricted transmission capacity. Aiming to enhance the system performance, the communication between clients and server needs to be diminished. Current FL strategies transmit a tremendous amount of data (model weights) within the FL process, which needs a high communication bandwidth. Considering resource constraints, increasing the number of clients and, consequently, the amount of data (model weights) can lead to a bottleneck. In this paper, we introduce the Federated Black Widow Optimization (FedBWO) technique to decrease the amount of transmitted data by transmitting only a performance score rather than the local model weights from clients. FedBWO employs the BWO algorithm to improve local model updates. The conducted experiments prove that FedBWO remarkably improves the performance of the global model and the communication efficiency of the overall system. According to the experimental outcomes, FedBWO enhances the global model accuracy by an average of 21% over FedAvg, and 12% over FedGWO. Furthermore, FedBWO dramatically decreases the communication cost compared to other methods. 

**Abstract (ZH)**: 联邦黑寡妇优化（FedBWO）算法：减少数据传输量提高联邦学习系统性能 

---
# Recognizing Ornaments in Vocal Indian Art Music with Active Annotation 

**Title (ZH)**: 用主动注释识别印度艺术音乐中的装饰音 

**Authors**: Sumit Kumar, Parampreet Singh, Vipul Arora  

**Link**: [PDF](https://arxiv.org/pdf/2505.04419)  

**Abstract**: Ornamentations, embellishments, or microtonal inflections are essential to melodic expression across many musical traditions, adding depth, nuance, and emotional impact to performances. Recognizing ornamentations in singing voices is key to MIR, with potential applications in music pedagogy, singer identification, genre classification, and controlled singing voice generation. However, the lack of annotated datasets and specialized modeling approaches remains a major obstacle for progress in this research area. In this work, we introduce Rāga Ornamentation Detection (ROD), a novel dataset comprising Indian classical music recordings curated by expert musicians. The dataset is annotated using a custom Human-in-the-Loop tool for six vocal ornaments marked as event-based labels. Using this dataset, we develop an ornamentation detection model based on deep time-series analysis, preserving ornament boundaries during the chunking of long audio recordings. We conduct experiments using different train-test configurations within the ROD dataset and also evaluate our approach on a separate, manually annotated dataset of Indian classical concert recordings. Our experimental results support the superior performance of our proposed approach over the baseline CRNN. 

**Abstract (ZH)**: 装饰音、装饰手法或微分音变 harmonization 在许多音乐传统的旋律表达中是必不可少的，它们为表演增添了深度、细腻和情感冲击力。在唱歌声音中识别装饰音是音乐信息检索中的关键问题，具有音乐教育、歌手识别、音乐体裁分类和受控唱歌声音生成等方面的应用潜力。然而，缺乏标注数据集和专门的建模方法仍然是该研究领域进步的主要障碍。在这项工作中，我们引入了Rāga装饰音检测（ROD），这是一个由专家音乐家精心挑选的印度古典音乐录音组成的新型数据集。该数据集使用自定义的人机交互标注工具为六个音区装饰音标记事件级标签。利用该数据集，我们基于深度时间序列分析开发了一个装饰音检测模型，在长音频录音的分段过程中保留了装饰音边界。我们使用ROD数据集的不同训练-测试配置进行了实验，并且还对该数据集之外的手动标注印度古典音乐会录音数据集进行了评估。我们的实验结果支持我们提出的模型优于基准CRNN的优越性能。 

---
# OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models 

**Title (ZH)**: OBLIVIATE：大型语言模型的稳健且实用的机器遗忘技术 

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04416)  

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose OBLIVIATE, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA), it ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including the Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: forget quality (new document-level memorization score), model utility, and fluency. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在广泛训练语料后存在记忆敏感、受版权保护或有毒内容的风险。为解决这一问题，我们提出OBLIVIATE，一种稳健的遗忘框架，能够在删除指定数据的同时保持模型的效果。该框架遵循结构化流程：提取目标token，构建保留集，并使用包含掩码、蒸馏和世界事实三种组件的定制损失函数进行微调。通过低秩适配器（LoRA），确保高效性而不牺牲遗忘质量。我们在哈利·波特系列、WMDP和TOFU等多个数据集上进行了实验，并使用全面的度量标准评估结果：遗忘质量（新文档级记忆得分）、模型效果和流畅性。实验结果表明，该框架在抵抗成员推断攻击、最小化保留数据影响以及在多种场景下保持鲁棒性方面表现出有效性。 

---
# YABLoCo: Yet Another Benchmark for Long Context Code Generation 

**Title (ZH)**: YABLoCo: 另一个长上下文代码生成基准 

**Authors**: Aidar Valeev, Roman Garaev, Vadim Lomshakov, Irina Piontkovskaya, Vladimir Ivanov, Israel Adewuyi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04406)  

**Abstract**: Large Language Models demonstrate the ability to solve various programming tasks, including code generation. Typically, the performance of LLMs is measured on benchmarks with small or medium-sized context windows of thousands of lines of code. At the same time, in real-world software projects, repositories can span up to millions of LoC. This paper closes this gap by contributing to the long context code generation benchmark (YABLoCo). The benchmark featured a test set of 215 functions selected from four large repositories with thousands of functions. The dataset contained metadata of functions, contexts of the functions with different levels of dependencies, docstrings, functions bodies, and call graphs for each repository. This paper presents three key aspects of the contribution. First, the benchmark aims at function body generation in large repositories in C and C++, two languages not covered by previous benchmarks. Second, the benchmark contains large repositories from 200K to 2,000K LoC. Third, we contribute a scalable evaluation pipeline for efficient computing of the target metrics and a tool for visual analysis of generated code. Overall, these three aspects allow for evaluating code generation in large repositories in C and C++. 

**Abstract (ZH)**: 大型语言模型展示了解决各种编程任务的能力，包括代码生成。通常，大型语言模型的性能是在包含数千行代码的小或中等大小上下文窗口的基准测试中进行衡量的。同时，在实际的软件项目中，代码仓库可能包含多达数百万行代码。本文通过贡献一个长上下文代码生成基准（YABLoCo）来弥补这一差距。该基准包含来自四个大型代码仓库的215个函数测试集，每个仓库包含数千个函数。数据集包含函数的元数据、具有不同依赖程度的函数上下文、文档字符串、函数体和调用图。本文呈现了贡献的三个方面。首先，基准旨在生成大型C和C++代码仓库中的函数体，这是之前基准未涵盖的语言。其次，基准包含从20万行到200万行代码的大仓库。第三，我们贡献了一个可扩展的评估管道，用于高效计算目标指标，并提供一个生成代码的可视化分析工具。总之，这三个方面允许对大型C和C++代码仓库中的代码生成进行评估。 

---
# High-speed multiwavelength photonic temporal integration using silicon photonics 

**Title (ZH)**: 基于硅光子学的高速多波长光电时域积分技术 

**Authors**: Yi Zhang, Nikolaos Farmakidis, Ioannis Roumpos, Miltiadis Moralis-Pegios, Apostolos Tsakyridis, June Sang Lee, Bowei Dong, Yuhan He, Samarth Aggarwal, Nikolaos Pleros, Harish Bhaskaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.04405)  

**Abstract**: Optical systems have been pivotal for energy-efficient computing, performing high-speed, parallel operations in low-loss carriers. While these predominantly analog optical accelerators bypass digitization to perform parallel floating-point computations, scaling optical hardware to map large-vector sizes for AI tasks remains challenging. Here, we overcome this limitation by unfolding scalar operations in time and introducing a photonic-heater-in-lightpath (PHIL) unit for all-optical temporal integration. Counterintuitively, we exploit a slow heat dissipation process to integrate optical signals modulated at 50 GHz bridging the speed gap between the widely applied thermo-optic effects and ultrafast photonics. This architecture supports optical end-to-end signal processing, eliminates inefficient electro-optical conversions, and enables both linear and nonlinear operations within a unified framework. Our results demonstrate a scalable path towards high-speed photonic computing through thermally driven integration. 

**Abstract (ZH)**: 光学系统在高效计算中发挥着关键作用，通过低损耗介质实现高速并行运算。尽管这些主要基于模拟的光学加速器可以通过并行浮点计算规避数字化，但将光学硬件扩展以适应大规模向量尺寸进行AI任务仍具有挑战性。通过在时间上展开标量操作并引入光路光子加热单元（PHIL单元）以实现全光学时间积分，我们克服了这一限制。出人意料的是，我们利用缓慢的热耗散过程将50 GHz频率下的光学信号进行集成，从而弥合了广泛应用的温度调制效应与超快光子学之间的速度差距。该架构支持端到端光学信号处理，消除不必要的电光转换，并在统一框架内实现线性和非线性操作。我们的结果展示了通过热驱动集成实现高速光子计算的可扩展路径。 

---
# In-Context Adaptation to Concept Drift for Learned Database Operations 

**Title (ZH)**: 基于上下文的 Learned 数据库操作概念漂移适应 

**Authors**: Jiaqi Zhu, Shaofeng Cai, Yanyan Shen, Gang Chen, Fang Deng, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04404)  

**Abstract**: Machine learning has demonstrated transformative potential for database operations, such as query optimization and in-database data analytics. However, dynamic database environments, characterized by frequent updates and evolving data distributions, introduce concept drift, which leads to performance degradation for learned models and limits their practical applicability. Addressing this challenge requires efficient frameworks capable of adapting to shifting concepts while minimizing the overhead of retraining or fine-tuning.
In this paper, we propose FLAIR, an online adaptation framework that introduces a new paradigm called \textit{in-context adaptation} for learned database operations. FLAIR leverages the inherent property of data systems, i.e., immediate availability of execution results for predictions, to enable dynamic context construction. By formalizing adaptation as $f:(\mathbf{x} \,| \,\mathcal{C}_t) \to \mathbf{y}$, with $\mathcal{C}_t$ representing a dynamic context memory, FLAIR delivers predictions aligned with the current concept, eliminating the need for runtime parameter optimization. To achieve this, FLAIR integrates two key modules: a Task Featurization Module for encoding task-specific features into standardized representations, and a Dynamic Decision Engine, pre-trained via Bayesian meta-training, to adapt seamlessly using contextual information at runtime. Extensive experiments across key database tasks demonstrate that FLAIR outperforms state-of-the-art baselines, achieving up to 5.2x faster adaptation and reducing error by 22.5% for cardinality estimation. 

**Abstract (ZH)**: 机器学习在数据库操作中的应用，如查询优化和数据库内数据分析，已经显示出变革性的潜力。然而，动态数据库环境因其频繁的更新和数据分布的演变导致的概念漂移，使得学习模型的性能下降，并限制了其实际应用。解决这一挑战需要能够高效适应不断变化的概念的同时，尽量减小重新训练或微调的开销。

本文提出FLAIR，一种在线适应框架，引入了一种新的范式——集成上下文适应，用于学习的数据库操作。FLAIR利用数据系统固有的特性，即能够即时获取执行结果以进行预测，从而实现动态上下文构建。通过将适应形式化为 $f:(\mathbf{x} \,|\, \mathcal{C}_t) \to \mathbf{y}$，其中 $\mathcal{C}_t$ 代表动态上下文记忆，FLAIR能够生成与当前概念对齐的预测，消除运行时参数优化的需要。为了实现这一目标，FLAIR集成两个关键模块：任务特征化模块，用于将任务特定特征编码为标准化表示，和基于贝叶斯元训练预训练的动态决策引擎，能够在运行时利用上下文信息无缝适应。在关键数据库任务的广泛实验中，FLAIR优于现有基线，实现高达5.2倍的适应速度提升，并将基数估计的误差减少了22.5%。 

---
# Deep residual learning with product units 

**Title (ZH)**: 深度残差学习与产品单元 

**Authors**: Ziyuan Li, Uwe Jaekel, Babette Dellen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04397)  

**Abstract**: We propose a deep product-unit residual neural network (PURe) that integrates product units into residual blocks to improve the expressiveness and parameter efficiency of deep convolutional networks. Unlike standard summation neurons, product units enable multiplicative feature interactions, potentially offering a more powerful representation of complex patterns. PURe replaces conventional convolutional layers with 2D product units in the second layer of each residual block, eliminating nonlinear activation functions to preserve structural information. We validate PURe on three benchmark datasets. On Galaxy10 DECaLS, PURe34 achieves the highest test accuracy of 84.89%, surpassing the much deeper ResNet152, while converging nearly five times faster and demonstrating strong robustness to Poisson noise. On ImageNet, PURe architectures outperform standard ResNet models at similar depths, with PURe34 achieving a top-1 accuracy of 80.27% and top-5 accuracy of 95.78%, surpassing deeper ResNet variants (ResNet50, ResNet101) while utilizing significantly fewer parameters and computational resources. On CIFAR-10, PURe consistently outperforms ResNet variants across varying depths, with PURe272 reaching 95.01% test accuracy, comparable to ResNet1001 but at less than half the model size. These results demonstrate that PURe achieves a favorable balance between accuracy, efficiency, and robustness. Compared to traditional residual networks, PURe not only achieves competitive classification performance with faster convergence and fewer parameters, but also demonstrates greater robustness to noise. Its effectiveness across diverse datasets highlights the potential of product-unit-based architectures for scalable and reliable deep learning in computer vision. 

**Abstract (ZH)**: 我们提出了一种深度产品单元残差神经网络（PURe），通过将产品单元整合到残差块中，以提高深卷积网络的表达能力和参数效率。与标准求和神经元不同，产品单元能够实现乘法特征交互，可能提供对复杂模式更具表现力的表示。PURe在每个残差块的第二层用2D产品单元替换传统的卷积层，保留结构信息的同时消除非线性激活函数。我们在三个基准数据集上验证了PURe。在Galaxy10 DECaLS上，PURe34达到最高的测试准确率84.89%，超过更深的ResNet152，同时收敛速度快近五倍，并且表现出对泊松噪声较强的鲁棒性。在ImageNet上，与深度相似的标准ResNet模型相比，PURe结构表现出更好的性能，PURe34的Top-1准确率为80.27%，Top-5准确率为95.78%，使用的参数和计算资源远远少于更深的ResNet变体（ResNet50, ResNet101）。在CIFAR-10上，PURe在不同深度下始终超越ResNet变体，PURe272达到95.01%的测试准确率，性能相当于ResNet1001但模型大小仅为一半。这些结果表明PURe在准确率、效率和鲁棒性之间实现了良好的平衡。与传统残差网络相比，PURe不仅在收敛速度更快和参数更少的情况下实现了竞争力的分类性能，而且还表现出对噪声的更强鲁棒性。其在多样数据集上的有效性突显了基于产品单元架构在计算机视觉中实现可扩展性和可靠性的潜力。 

---
# The Aloe Family Recipe for Open and Specialized Healthcare LLMs 

**Title (ZH)**: Aloe 家族配方：开放与专业化医疗LLM 

**Authors**: Dario Garcia-Gasulla, Jordi Bayarri-Planas, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Adrian Tormos, Daniel Hinjos, Pablo Bernabeu-Perez, Anna Arias-Duart, Pablo Agustin Martin-Torres, Marta Gonzalez-Mallo, Sergio Alvarez-Napagao, Eduard Ayguadé-Parra, Ulises Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2505.04388)  

**Abstract**: Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license.
Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results.
Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models.
Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare. 

**Abstract (ZH)**: 目的：随着大型语言模型（LLMs）在医疗领域的进步，保护公众利益的 competitive 开源模型变得尤为重要。这项工作通过优化数据预处理和训练的关键阶段，并展示如何通过直接偏好优化（DPO）提高模型安全性、通过 Retrieval-Augmented Generation（RAG）提高模型有效性，为开放医疗 LLM 领域做出了贡献。所采用的评价方法包括四种不同类型的测试，为领域内设定了新的标准。这些模型展示出与最佳私有替代产品竞争力，并在宽松的许可下发布。

方法：基于强大的基础模型如 Llama 3.1 和 Qwen 2.5，Aloe Beta 使用一个自定义的数据集，通过合成链式思考示例增强公共数据。模型经过直接偏好优化（DPO）对齐，强调在面对破解攻击时的伦理和政策对齐性能。评价包括封闭式、开放式、安全性和人类评估，以最大化结果的可靠性。

结果：在整个管道中提出了推荐方案，依托 Aloe 家族的强大性能。这些模型在医疗健康基准测试和医疗领域表现出色，并经常得到医疗专业人员的青睐。在偏见和毒性方面，Aloe Beta 模型显著提高了安全性，并显示出了对未见破解攻击的抗性。为了负责任地发布，Aloe 家族模型附带着详细的风险评估，具体针对医疗健康领域。

结论：Aloe Beta 模型及其制作方法是开源医疗 LLM 领域的重要贡献，提供了顶级性能并保持高标准的伦理要求。这项工作为在医疗健康领域开发和报告对齐的 LLM 设定了新标准。 

---
# Consensus-Aware AV Behavior: Trade-offs Between Safety, Interaction, and Performance in Mixed Urban Traffic 

**Title (ZH)**: 共识导向的自动驾驶车辆行为：混合城市交通中安全、交互与性能之间的权衡 

**Authors**: Mohammad Elayan, Wissam Kontar  

**Link**: [PDF](https://arxiv.org/pdf/2505.04379)  

**Abstract**: Transportation systems have long been shaped by complexity and heterogeneity, driven by the interdependency of agent actions and traffic outcomes. The deployment of automated vehicles (AVs) in such systems introduces a new challenge: achieving consensus across safety, interaction quality, and traffic performance. In this work, we position consensus as a fundamental property of the traffic system and aim to quantify it. We use high-resolution trajectory data from the Third Generation Simulation (TGSIM) dataset to empirically analyze AV and human-driven vehicle (HDV) behavior at a signalized urban intersection and around vulnerable road users (VRUs). Key metrics, including Time-to-Collision (TTC), Post-Encroachment Time (PET), deceleration patterns, headways, and string stability, are evaluated across the three performance dimensions. Results show that full consensus across safety, interaction, and performance is rare, with only 1.63% of AV-VRU interaction frames meeting all three conditions. These findings highlight the need for AV models that explicitly balance multi-dimensional performance in mixed-traffic environments. Full reproducibility is supported via our open-source codebase on this https URL. 

**Abstract (ZH)**: 交通运输系统长期以来受到复杂性和异质性的影响，由代理行动的相互依赖性和交通结果驱动。自动驾驶车辆（AVs）的部署为系统带来了新的挑战：在安全性、互动质量和交通性能之间达成共识。在本工作中，我们将共识视为交通系统的基本属性，并致力于定量评估它。我们使用来自TGSIM数据集的高分辨率轨迹数据，在信号控制的城市交叉口和弱势道路使用者（VRUs）周围，实证分析AV和人工驾驶车辆（HDVs）的行为。通过评估安全性、互动质量和性能三个维度的关键指标，包括碰撞时间（TTC）、侵入后时间（PET）、减速模式、车距和车队稳定性，结果显示在安全性、互动性和性能三个维度上完全达成共识极为罕见，仅有1.63%的AV-VRU交互帧满足所有三个条件。这些发现强调了在混合交通环境中需要明确平衡多维度性能的AV模型。通过我们的开源代码库，实现了完全可重复性（详见：this https URL）。 

---
# Balancing Accuracy, Calibration, and Efficiency in Active Learning with Vision Transformers Under Label Noise 

**Title (ZH)**: 在标签噪声条件下，基于视觉变换器的主动学习中平衡准确率、校准性和效率 

**Authors**: Moseli Mots'oehli, Hope Mogale, Kyungim Baek  

**Link**: [PDF](https://arxiv.org/pdf/2505.04375)  

**Abstract**: Fine-tuning pre-trained convolutional neural networks on ImageNet for downstream tasks is well-established. Still, the impact of model size on the performance of vision transformers in similar scenarios, particularly under label noise, remains largely unexplored. Given the utility and versatility of transformer architectures, this study investigates their practicality under low-budget constraints and noisy labels. We explore how classification accuracy and calibration are affected by symmetric label noise in active learning settings, evaluating four vision transformer configurations (Base and Large with 16x16 and 32x32 patch sizes) and three Swin Transformer configurations (Tiny, Small, and Base) on CIFAR10 and CIFAR100 datasets, under varying label noise rates. Our findings show that larger ViT models (ViTl32 in particular) consistently outperform their smaller counterparts in both accuracy and calibration, even under moderate to high label noise, while Swin Transformers exhibit weaker robustness across all noise levels. We find that smaller patch sizes do not always lead to better performance, as ViTl16 performs consistently worse than ViTl32 while incurring a higher computational cost. We also find that information-based Active Learning strategies only provide meaningful accuracy improvements at moderate label noise rates, but they result in poorer calibration compared to models trained on randomly acquired labels, especially at high label noise rates. We hope these insights provide actionable guidance for practitioners looking to deploy vision transformers in resource-constrained environments, where balancing model complexity, label noise, and compute efficiency is critical in model fine-tuning or distillation. 

**Abstract (ZH)**: 预训练卷积神经网络在ImageNet上的微调在下游任务中已有广泛应用，但在类似场景下（特别是存在标签噪声时），视觉变换器模型大小对性能的影响仍 largely unexplored。鉴于变换器架构的实用性和灵活性，本研究探讨了在低预算约束和噪声标签条件下变换器的实际应用。我们研究了噪声环境下活性学习设置中分类准确性和校准度受到对称标签噪声的影响，并在CIFAR10和CIFAR100数据集上评估了四种视觉变换器配置（16x16和32x32嵌patch大小的Base和Large模型）以及三种Swin Transformer配置（Tiny、Small和Base模型）在不同标签噪声率下的表现。研究发现，在中等到高标签噪声水平下，较大的ViT模型（特别是ViTl32）在准确性和校准度上均优于较小的模型，而Swin Transformer在所有噪声水平下的鲁棒性较弱。我们还发现，较小的嵌patch大小并不总是带来更好的性能，因为ViTl16的表现通常劣于ViTl32且计算成本更高。此外，我们发现基于信息的活性学习策略仅在中等标签噪声率下能提供有意义的准确度提升，但在高标签噪声率下会降低校准度，尤其是在随机获取标签训练的模型环境下表现更差。希望这些见解为希望在资源受限环境中部署视觉变换器的实践者提供可操作的指导，特别是在模型微调或知识蒸馏时需要平衡模型复杂性、标签噪声和计算效率。 

---
# Optimization Problem Solving Can Transition to Evolutionary Agentic Workflows 

**Title (ZH)**: 优化问题求解可以过渡到进化代理工作流 

**Authors**: Wenhao Li, Bo Jin, Mingyi Hong, Changhong Lu, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04354)  

**Abstract**: This position paper argues that optimization problem solving can transition from expert-dependent to evolutionary agentic workflows. Traditional optimization practices rely on human specialists for problem formulation, algorithm selection, and hyperparameter tuning, creating bottlenecks that impede industrial adoption of cutting-edge methods. We contend that an evolutionary agentic workflow, powered by foundation models and evolutionary search, can autonomously navigate the optimization space, comprising problem, formulation, algorithm, and hyperparameter spaces. Through case studies in cloud resource scheduling and ADMM parameter adaptation, we demonstrate how this approach can bridge the gap between academic innovation and industrial implementation. Our position challenges the status quo of human-centric optimization workflows and advocates for a more scalable, adaptive approach to solving real-world optimization problems. 

**Abstract (ZH)**: 这篇立场论文 argue 讨论了优化问题求解可以从依赖专家转变为进化代理工作流。传统的优化方法依赖于人类专家进行问题表述、算法选择和超参数调整，从而形成阻碍前沿方法在工业中应用的瓶颈。我们认为，基于基础模型和进化搜索的进化代理工作流能够自主导航优化空间，包括问题、表述、算法和超参数空间。通过云资源调度和ADMM参数适应的案例研究，我们展示了这种方法如何弥合学术创新与工业实施之间的差距。我们的立场挑战了以人为中心的优化工作流现状，并倡导一种更具扩展性、更适应实际优化问题的解决方法。 

---
# Multi-Granular Attention based Heterogeneous Hypergraph Neural Network 

**Title (ZH)**: 基于多粒度注意力的异构超图神经网络 

**Authors**: Hong Jin, Kaicheng Zhou, Jie Yin, Lan You, Zhifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04340)  

**Abstract**: Heterogeneous graph neural networks (HeteGNNs) have demonstrated strong abilities to learn node representations by effectively extracting complex structural and semantic information in heterogeneous graphs. Most of the prevailing HeteGNNs follow the neighborhood aggregation paradigm, leveraging meta-path based message passing to learn latent node representations. However, due to the pairwise nature of meta-paths, these models fail to capture high-order relations among nodes, resulting in suboptimal performance. Additionally, the challenge of ``over-squashing'', where long-range message passing in HeteGNNs leads to severe information distortion, further limits the efficacy of these models. To address these limitations, this paper proposes MGA-HHN, a Multi-Granular Attention based Heterogeneous Hypergraph Neural Network for heterogeneous graph representation learning. MGA-HHN introduces two key innovations: (1) a novel approach for constructing meta-path based heterogeneous hypergraphs that explicitly models higher-order semantic information in heterogeneous graphs through multiple views, and (2) a multi-granular attention mechanism that operates at both the node and hyperedge levels. This mechanism enables the model to capture fine-grained interactions among nodes sharing the same semantic context within a hyperedge type, while preserving the diversity of semantics across different hyperedge types. As such, MGA-HHN effectively mitigates long-range message distortion and generates more expressive node representations. Extensive experiments on real-world benchmark datasets demonstrate that MGA-HHN outperforms state-of-the-art models, showcasing its effectiveness in node classification, node clustering and visualization tasks. 

**Abstract (ZH)**: 基于多粒度注意力的异构超图神经网络（MGA-HHN）：用于异构图表示学习 

---
# Detecting Concept Drift in Neural Networks Using Chi-squared Goodness of Fit Testing 

**Title (ZH)**: 使用卡方拟合优度检验检测神经网络中的概念漂移 

**Authors**: Jacob Glenn Ayers, Buvaneswari A. Ramanan, Manzoor A. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04318)  

**Abstract**: As the adoption of deep learning models has grown beyond human capacity for verification, meta-algorithms are needed to ensure reliable model inference. Concept drift detection is a field dedicated to identifying statistical shifts that is underutilized in monitoring neural networks that may encounter inference data with distributional characteristics diverging from their training data. Given the wide variety of model architectures, applications, and datasets, it is important that concept drift detection algorithms are adaptable to different inference scenarios. In this paper, we introduce an application of the $\chi^2$ Goodness of Fit Hypothesis Test as a drift detection meta-algorithm applied to a multilayer perceptron, a convolutional neural network, and a transformer trained for machine vision as they are exposed to simulated drift during inference. To that end, we demonstrate how unexpected drops in accuracy due to concept drift can be detected without directly examining the inference outputs. Our approach enhances safety by ensuring models are continually evaluated for reliability across varying conditions. 

**Abstract (ZH)**: 随着深度学习模型的应用超出人类验证能力，需要元算法来确保模型推理的可靠性。概念漂移检测专注于识别统计变化，但在监测可能出现与训练数据分布特性不同的推理数据的神经网络时尚未充分利用。鉴于模型架构、应用场景和数据集的多样性，概念漂移检测算法应适应不同的推理场景。在本文中，我们介绍了一种$\chi^2$拟合优度假设检验在多层感知器、卷积神经网络和用于机器视觉的变压器中的应用，这些模型在暴露于模拟的推理漂移期间进行训练。我们展示了如何在不直接检查推理输出的情况下检测由于概念漂移导致的意外准确度下降。我们的方法通过确保模型在各种条件下持续评估可靠性来增强安全性。 

---
# Guardians of the Web: The Evolution and Future of Website Information Security 

**Title (ZH)**: 网络守护者：网站信息安全的演变与未来 

**Authors**: Md Saiful Islam, Li Xiangdong  

**Link**: [PDF](https://arxiv.org/pdf/2505.04308)  

**Abstract**: Website information security has become a critical concern in the digital age. This article explores the evolution of website information security, examining its historical development, current practices, and future directions. The early beginnings from the 1960s to the 1980s laid the groundwork for modern cybersecurity, with the development of ARPANET, TCP/IP, public-key cryptography, and the first antivirus programs. The 1990s marked a transformative era, driven by the commercialization of the Internet and the emergence of web-based services. As the Internet grew, so did the range and sophistication of cyber threats, leading to advancements in security technologies such as the Secure Sockets Layer (SSL) protocol, password protection, and firewalls. Current practices in website information security involve a multi-layered approach, including encryption, secure coding practices, regular security audits, and user education. The future of website information security is expected to be shaped by emerging technologies such as artificial intelligence, blockchain, and quantum computing, as well as the increasing importance of international cooperation and standardization efforts. As cyber threats continue to evolve, ongoing research and innovation in website information security will be essential to protect sensitive information and maintain trust in the digital world. 

**Abstract (ZH)**: 网站信息安全已成为数字时代的关键关切。本文探讨了网站信息安全的发展演变，审视其历史发展、当前实践和未来方向。从20世纪60年代到80年代的早期 beginnings奠定了现代网络安全的基础，ARPANET、TCP/IP、公钥加密和首批防病毒程序的发展都是其中的重要组成部分。20世纪90年代标志着一个变革的时代，这一时期互联网的商业化和基于Web的服务的出现是其主要推动力。随着互联网的发展，网络威胁的种类和复杂性也随之增加，这推动了如安全套接层（SSL）协议、密码保护和防火墙等安全技术的进步。当前的网站信息安全实践采用了多层方法，包括加密、安全编码实践、定期安全审计和用户教育。网站信息安全的未来将受新兴技术（如人工智能、区块链和量子计算）以及国际协作和标准制定努力不断增强重要性的影响。随着网络威胁持续演变，网站信息安全领域的持续研究与创新对于保护敏感信息和维护数字世界的信任将是必不可少的。 

---
# Sparsity is All You Need: Rethinking Biological Pathway-Informed Approaches in Deep Learning 

**Title (ZH)**: 稀疏性即为关键：重思生物途径导向的深度学习方法 

**Authors**: Isabella Caranzano, Corrado Pancotti, Cesare Rollo, Flavio Sartori, Pietro Liò, Piero Fariselli, Tiziana Sanavia  

**Link**: [PDF](https://arxiv.org/pdf/2505.04300)  

**Abstract**: Biologically-informed neural networks typically leverage pathway annotations to enhance performance in biomedical applications. We hypothesized that the benefits of pathway integration does not arise from its biological relevance, but rather from the sparsity it introduces. We conducted a comprehensive analysis of all relevant pathway-based neural network models for predictive tasks, critically evaluating each study's contributions. From this review, we curated a subset of methods for which the source code was publicly available. The comparison of the biologically informed state-of-the-art deep learning models and their randomized counterparts showed that models based on randomized information performed equally well as biologically informed ones across different metrics and datasets. Notably, in 3 out of the 15 analyzed models, the randomized versions even outperformed their biologically informed counterparts. Moreover, pathway-informed models did not show any clear advantage in interpretability, as randomized models were still able to identify relevant disease biomarkers despite lacking explicit pathway information. Our findings suggest that pathway annotations may be too noisy or inadequately explored by current methods. Therefore, we propose a methodology that can be applied to different domains and can serve as a robust benchmark for systematically comparing novel pathway-informed models against their randomized counterparts. This approach enables researchers to rigorously determine whether observed performance improvements can be attributed to biological insights. 

**Abstract (ZH)**: 生物信息导向的神经网络通常通过路径注释来提高生物医学应用中的性能。我们假设路径集成的好处并非来自其生物学相关性，而是来自其引入的稀疏性。我们对所有与预测任务相关的基于路径的神经网络模型进行了全面分析，并对每一项研究的贡献进行了批判性评估。从这项综述中，我们筛选出了一组公开可获取源代码的方法。对比生物信息导向的前沿深度学习模型与其随机化版本的性能，结果显示基于随机信息的模型在不同指标和数据集上表现与生物信息导向的模型相当。值得注意的是，在分析的15个模型中有3个模型的随机化版本甚至优于其生物信息导向的版本。此外，路径导向的模型在可解释性方面并没有显示出明显的优势，尽管缺乏显式路径信息，随机化模型仍然能够识别出相关的疾病生物标志物。我们的研究结果表明，当前方法可能尚未充分探索路径注释的有效性，因此我们提出了一种可在不同领域应用的方法，并可作为系统比较新路径导向模型与其随机化版本的稳健基准。该方法允许研究人员严格确定观察到的性能改进是否源于生物学洞察。 

---
# GASCADE: Grouped Summarization of Adverse Drug Event for Enhanced Cancer Pharmacovigilance 

**Title (ZH)**: GCASCADE：分组药物不良事件总结以增强癌症药品监控 

**Authors**: Sofia Jamil, Aryan Dabad, Bollampalli Areen Reddy, Sriparna Saha, Rajiv Misra, Adil A. Shakur  

**Link**: [PDF](https://arxiv.org/pdf/2505.04284)  

**Abstract**: In the realm of cancer treatment, summarizing adverse drug events (ADEs) reported by patients using prescribed drugs is crucial for enhancing pharmacovigilance practices and improving drug-related decision-making. While the volume and complexity of pharmacovigilance data have increased, existing research in this field has predominantly focused on general diseases rather than specifically addressing cancer. This work introduces the task of grouped summarization of adverse drug events reported by multiple patients using the same drug for cancer treatment. To address the challenge of limited resources in cancer pharmacovigilance, we present the MultiLabeled Cancer Adverse Drug Reaction and Summarization (MCADRS) dataset. This dataset includes pharmacovigilance posts detailing patient concerns regarding drug efficacy and adverse effects, along with extracted labels for drug names, adverse drug events, severity, and adversity of reactions, as well as summaries of ADEs for each drug. Additionally, we propose the Grouping and Abstractive Summarization of Cancer Adverse Drug events (GASCADE) framework, a novel pipeline that combines the information extraction capabilities of Large Language Models (LLMs) with the summarization power of the encoder-decoder T5 model. Our work is the first to apply alignment techniques, including advanced algorithms like Direct Preference Optimization, to encoder-decoder models using synthetic datasets for summarization tasks. Through extensive experiments, we demonstrate the superior performance of GASCADE across various metrics, validated through both automated assessments and human evaluations. This multitasking approach enhances drug-related decision-making and fosters a deeper understanding of patient concerns, paving the way for advancements in personalized and responsive cancer care. The code and dataset used in this work are publicly available. 

**Abstract (ZH)**: 基于患者报告的抗癌药物不良药物事件分组总结在癌症治疗领域药物警戒实践中的应用研究 

---
# Non-stationary Diffusion For Probabilistic Time Series Forecasting 

**Title (ZH)**: 非平稳扩散的概率时间序列预测 

**Authors**: Weiwei Ye, Zhuopeng Xu, Ning Gui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04278)  

**Abstract**: Due to the dynamics of underlying physics and external influences, the uncertainty of time series often varies over time. However, existing Denoising Diffusion Probabilistic Models (DDPMs) often fail to capture this non-stationary nature, constrained by their constant variance assumption from the additive noise model (ANM). In this paper, we innovatively utilize the Location-Scale Noise Model (LSNM) to relax the fixed uncertainty assumption of ANM. A diffusion-based probabilistic forecasting framework, termed Non-stationary Diffusion (NsDiff), is designed based on LSNM that is capable of modeling the changing pattern of uncertainty. Specifically, NsDiff combines a denoising diffusion-based conditional generative model with a pre-trained conditional mean and variance estimator, enabling adaptive endpoint distribution modeling. Furthermore, we propose an uncertainty-aware noise schedule, which dynamically adjusts the noise levels to accurately reflect the data uncertainty at each step and integrates the time-varying variances into the diffusion process. Extensive experiments conducted on nine real-world and synthetic datasets demonstrate the superior performance of NsDiff compared to existing approaches. Code is available at this https URL. 

**Abstract (ZH)**: 基于位置-尺度噪声模型的非稳态扩散概率预测框架 

---
# Object-Shot Enhanced Grounding Network for Egocentric Video 

**Title (ZH)**: 基于对象短语增强的自我中心视频目标 grounding 网络 

**Authors**: Yisen Feng, Haoyu Zhang, Meng Liu, Weili Guan, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.04270)  

**Abstract**: Egocentric video grounding is a crucial task for embodied intelligence applications, distinct from exocentric video moment localization. Existing methods primarily focus on the distributional differences between egocentric and exocentric videos but often neglect key characteristics of egocentric videos and the fine-grained information emphasized by question-type queries. To address these limitations, we propose OSGNet, an Object-Shot enhanced Grounding Network for egocentric video. Specifically, we extract object information from videos to enrich video representation, particularly for objects highlighted in the textual query but not directly captured in the video features. Additionally, we analyze the frequent shot movements inherent to egocentric videos, leveraging these features to extract the wearer's attention information, which enhances the model's ability to perform modality alignment. Experiments conducted on three datasets demonstrate that OSGNet achieves state-of-the-art performance, validating the effectiveness of our approach. Our code can be found at this https URL. 

**Abstract (ZH)**: 自我中心视频锚定是体现智能应用中的一个关键任务，与外视角视频moment定位不同。现有方法主要关注自我中心视频和外视角视频之间的分布差异，但往往忽视了自我中心视频的关键特征以及文本查询强调的细粒度信息。为了解决这些局限性，我们提出了一种基于对象-shot的锚定网络OSGNet，用于自我中心视频。具体而言，我们从视频中提取对象信息以丰富视频表示，特别是对于文本查询中强调但在视频特征中未直接捕捉到的对象。此外，我们分析了自我中心视频中固有的频繁镜头运动，利用这些特征来提取佩戴者注意力信息，从而增强模型的模态对齐能力。在三个数据集上的实验结果显示，OSGNet 达到了最先进的性能，验证了我们方法的有效性。我们的代码可以通过以下链接获取：this https URL。 

---
# Weaponizing Language Models for Cybersecurity Offensive Operations: Automating Vulnerability Assessment Report Validation; A Review Paper 

**Title (ZH)**: 利用语言模型进行网络安全进攻性操作：自动化漏洞评估报告验证——一篇综述论文 

**Authors**: Abdulrahman S Almuhaidib, Azlan Mohd Zain, Zalmiyah Zakaria, Izyan Izzati Kamsani, Abdulaziz S Almuhaidib  

**Link**: [PDF](https://arxiv.org/pdf/2505.04265)  

**Abstract**: This, with the ever-increasing sophistication of cyberwar, calls for novel solutions. In this regard, Large Language Models (LLMs) have emerged as a highly promising tool for defensive and offensive cybersecurity-related strategies. While existing literature has focused much on the defensive use of LLMs, when it comes to their offensive utilization, very little has been reported-namely, concerning Vulnerability Assessment (VA) report validation. Consequentially, this paper tries to fill that gap by investigating the capabilities of LLMs in automating and improving the validation process of the report of the VA. From the critical review of the related literature, this paper hereby proposes a new approach to using the LLMs in the automation of the analysis and within the validation process of the report of the VA that could potentially reduce the number of false positives and generally enhance efficiency. These results are promising for LLM automatization for improving validation on reports coming from VA in order to improve accuracy while reducing human effort and security postures. The contribution of this paper provides further evidence about the offensive and defensive LLM capabilities and therefor helps in devising more appropriate cybersecurity strategies and tools accordingly. 

**Abstract (ZH)**: 随着网络战的日益复杂，需要提出新的解决方案。在这方面，大规模语言模型（LLMs）已成为防御性和进攻性网络安全策略中极具前景的工具。尽管现有文献主要关注LLMs的防御性使用，但在其进攻性利用方面，特别是漏洞评估（VA）报告验证方面，报道相对较少。因此，本文试图通过研究LLMs在自动化和改进VA报告验证过程中的能力来填补这一空白。通过对相关文献的批判性回顾，本文提出了一种新的方法，用于自动化VA报告的分析和验证过程，该方法有望减少假阳性的数量，并提高整体效率。这些结果对使用LLM自动化改进来自VA的报告验证以提高准确性、减少人力投入和安全态势具有积极意义。本文的贡献为进一步证明了LLMs的进攻性和防御性能力，并有助于制定更合适的网络安全策略和工具。 

---
# Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering 

**Title (ZH)**: 可引导聊天机器人：基于偏好激活引导的个性化大语言模型 

**Authors**: Jessica Y. Bo, Tianyu Xu, Ishan Chatterjee, Katrina Passarella-Ward, Achin Kulshrestha, D Shin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04260)  

**Abstract**: As large language models (LLMs) improve in their capacity to serve as personal AI assistants, their ability to output uniquely tailored, personalized responses that align with the soft preferences of their users is essential for enhancing user satisfaction and retention. However, untrained lay users have poor prompt specification abilities and often struggle with conveying their latent preferences to AI assistants. To address this, we leverage activation steering to guide LLMs to align with interpretable preference dimensions during inference. In contrast to memory-based personalization methods that require longer user history, steering is extremely lightweight and can be easily controlled by the user via an linear strength factor. We embed steering into three different interactive chatbot interfaces and conduct a within-subjects user study (n=14) to investigate how end users prefer to personalize their conversations. The results demonstrate the effectiveness of preference-based steering for aligning real-world conversations with hidden user preferences, and highlight further insights on how diverse values around control, usability, and transparency lead users to prefer different interfaces. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的能力不断增强，使其能够作为个人AI助手并输出与用户柔和偏好相匹配的独特个性化响应，这对于提高用户满意度和留存至关重要。然而，未经训练的普通用户在提示指定能力较弱，常常难以向AI助手传达其隐性偏好。为解决这一问题，我们利用激活引导在推理过程中引导LLMs与可解释的偏好维度对齐。与需要更长用户历史的基于记忆的个性化方法相比，激活引导极为轻量级，并且可以通过一个线性强度因子轻松由用户控制。我们将激活引导嵌入到三个不同的交互式聊天机器人界面中，并进行一项针对单个用户的用户研究（n=14），以调查最终用户如何偏好个性化其对话的方式。研究结果展示了基于偏好的激活引导在将实际对话与隐藏用户偏好对齐方面的有效性，并进一步揭示了不同控制、易用性和透明度价值观如何引导用户偏好不同的界面。 

---
# Facilitating Trustworthy Human-Agent Collaboration in LLM-based Multi-Agent System oriented Software Engineering 

**Title (ZH)**: 面向软件工程的基于大语言模型的多智能体系统中可信赖的人机协作促进 

**Authors**: Krishna Ronanki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04251)  

**Abstract**: Multi-agent autonomous systems (MAS) are better at addressing challenges that spans across multiple domains than singular autonomous agents. This holds true within the field of software engineering (SE) as well. The state-of-the-art research on MAS within SE focuses on integrating LLMs at the core of autonomous agents to create LLM-based multi-agent autonomous (LMA) systems. However, the introduction of LMA systems into SE brings a plethora of challenges. One of the major challenges is the strategic allocation of tasks between humans and the LMA system in a trustworthy manner. To address this challenge, a RACI-based framework is proposed in this work in progress article, along with implementation guidelines and an example implementation of the framework. The proposed framework can facilitate efficient collaboration, ensure accountability, and mitigate potential risks associated with LLM-driven automation while aligning with the Trustworthy AI guidelines. The future steps for this work delineating the planned empirical validation method are also presented. 

**Abstract (ZH)**: 多代理自主系统在软件工程领域中比单一自主代理更擅长应对跨多个领域的挑战。本文进展文章提出了一种基于RACI的框架，并结合实施指南和框架的示例实现，以有效地分配任务，确保责任明确，并减少LLM驱动自动化可能带来的风险，同时符合可信人工智能指南。文中还提出了未来的工作计划和拟议的实证验证方法。 

---
# FRAIN to Train: A Fast-and-Reliable Solution for Decentralized Federated Learning 

**Title (ZH)**: FRAIN to Train: 一种快速可靠的分布式联邦学习解决方案 

**Authors**: Sanghyeon Park, Soo-Mook Moon  

**Link**: [PDF](https://arxiv.org/pdf/2505.04223)  

**Abstract**: Federated learning (FL) enables collaborative model training across distributed clients while preserving data locality. Although FedAvg pioneered synchronous rounds for global model averaging, slower devices can delay collective progress. Asynchronous FL (e.g., FedAsync) addresses stragglers by continuously integrating client updates, yet naive implementations risk client drift due to non-IID data and stale contributions. Some Blockchain-based FL approaches (e.g., BRAIN) employ robust weighting or scoring of updates to resist malicious or misaligned proposals. However, performance drops can still persist under severe data heterogeneity or high staleness, and synchronization overhead has emerged as a new concern due to its aggregator-free architectures.
We introduce Fast-and-Reliable AI Network, FRAIN, a new asynchronous FL method that mitigates these limitations by incorporating two key ideas. First, our FastSync strategy eliminates the need to replay past model versions, enabling newcomers and infrequent participants to efficiently approximate the global model. Second, we adopt spherical linear interpolation (SLERP) when merging parameters, preserving models' directions and alleviating destructive interference from divergent local training.
Experiments with a CNN image-classification model and a Transformer-based language model demonstrate that FRAIN achieves more stable and robust convergence than FedAvg, FedAsync, and BRAIN, especially under harsh environments: non-IID data distributions, networks that experience delays and require frequent re-synchronization, and the presence of malicious nodes. 

**Abstract (ZH)**: 快速可靠AI网络：一种新的异步联邦学习方法 

---
# To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay 

**Title (ZH)**: 是否判断：使用大语言模型判断广告关键词的相关性于eBay 

**Authors**: Soumik Dey, Hansi Wu, Binbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.04209)  

**Abstract**: E-commerce sellers are recommended keyphrases based on their inventory on which they advertise to increase buyer engagement (clicks/sales). The relevance of advertiser keyphrases plays an important role in preventing the inundation of search systems with numerous irrelevant items that compete for attention in auctions, in addition to maintaining a healthy seller perception. In this work, we describe the shortcomings of training Advertiser keyphrase relevance filter models on click/sales/search relevance signals and the importance of aligning with human judgment, as sellers have the power to adopt or reject said keyphrase recommendations. In this study, we frame Advertiser keyphrase relevance as a complex interaction between 3 dynamical systems -- seller judgment, which influences seller adoption of our product, Advertising, which provides the keyphrases to bid on, and Search, who holds the auctions for the same keyphrases. This study discusses the practicalities of using human judgment via a case study at eBay Advertising and demonstrate that using LLM-as-a-judge en-masse as a scalable proxy for seller judgment to train our relevance models achieves a better harmony across the three systems -- provided that they are bound by a meticulous evaluation framework grounded in business metrics. 

**Abstract (ZH)**: 电子商务卖家根据其库存推荐的关键短语用于广告以增加买家互动（点击/销售），广告商关键短语的相关性在防止搜索引擎被大量不相关项目淹没以及维护健康的卖家形象方面起着重要作用。本文描述了在使用点击/销售/搜索相关性信号训练广告商关键短语相关性过滤模型方面的不足，并强调了与人工判断对齐的重要性，因为卖家有权接受或拒绝这些关键短语建议。在本研究中，我们将广告商关键短语相关性视为卖家判断、广告提供可竞标的关键短语和搜索引擎进行相同关键短语拍卖之间的复杂动态交互。本文通过eBay广告案例研究探讨了使用人工判断的实践，并证明了将大规模使用的LLM作为卖家判断的可扩展代理来训练相关性模型，在确保其基于企业指标的细致评价框架的前提下，可以在三个系统之间实现更好的和谐。 

---
# An Enhanced YOLOv8 Model for Real-Time and Accurate Pothole Detection and Measurement 

**Title (ZH)**: 增强的YOLOv8模型实现实时精准坑洞检测与测量 

**Authors**: Mustafa Yurdakul, Şakir Tasdemir  

**Link**: [PDF](https://arxiv.org/pdf/2505.04207)  

**Abstract**: Potholes cause vehicle damage and traffic accidents, creating serious safety and economic problems. Therefore, early and accurate detection of potholes is crucial. Existing detection methods are usually only based on 2D RGB images and cannot accurately analyze the physical characteristics of potholes. In this paper, a publicly available dataset of RGB-D images (PothRGBD) is created and an improved YOLOv8-based model is proposed for both pothole detection and pothole physical features analysis. The Intel RealSense D415 depth camera was used to collect RGB and depth data from the road surfaces, resulting in a PothRGBD dataset of 1000 images. The data was labeled in YOLO format suitable for segmentation. A novel YOLO model is proposed based on the YOLOv8n-seg architecture, which is structurally improved with Dynamic Snake Convolution (DSConv), Simple Attention Module (SimAM) and Gaussian Error Linear Unit (GELU). The proposed model segmented potholes with irregular edge structure more accurately, and performed perimeter and depth measurements on depth maps with high accuracy. The standard YOLOv8n-seg model achieved 91.9% precision, 85.2% recall and 91.9% mAP@50. With the proposed model, the values increased to 93.7%, 90.4% and 93.8% respectively. Thus, an improvement of 1.96% in precision, 6.13% in recall and 2.07% in mAP was achieved. The proposed model performs pothole detection as well as perimeter and depth measurement with high accuracy and is suitable for real-time applications due to its low model complexity. In this way, a lightweight and effective model that can be used in deep learning-based intelligent transportation solutions has been acquired. 

**Abstract (ZH)**: 基于RGB-D图像的沥青路面坑洞检测与物理特征分析 

---
# VideoPath-LLaVA: Pathology Diagnostic Reasoning Through Video Instruction Tuning 

**Title (ZH)**: VideoPath-LLaVA: 通过视频指令调优进行病理诊断推理 

**Authors**: Trinh T.L. Vuong, Jin Tae Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2505.04192)  

**Abstract**: We present VideoPath-LLaVA, the first large multimodal model (LMM) in computational pathology that integrates three distinct image scenarios, single patch images, automatically keyframe-extracted clips, and manually segmented video pathology images, to mimic the natural diagnostic process of pathologists. By generating detailed histological descriptions and culminating in a definitive sign-out diagnosis, VideoPath-LLaVA bridges visual narratives with diagnostic reasoning.
Central to our approach is the VideoPath-Instruct dataset, comprising 4278 video and diagnosis-specific chain-of-thought instructional pairs sourced from educational histopathology videos on YouTube. Although high-quality data is critical for enhancing diagnostic reasoning, its creation is time-intensive and limited in volume. To overcome this challenge, we transfer knowledge from existing single-image instruction datasets to train on weakly annotated, keyframe-extracted clips, followed by fine-tuning on manually segmented videos. VideoPath-LLaVA establishes a new benchmark in pathology video analysis and offers a promising foundation for future AI systems that support clinical decision-making through integrated visual and diagnostic reasoning. Our code, data, and model are publicly available at this https URL. 

**Abstract (ZH)**: VideoPath-LLaVA：一种集成单张图像、自动关键帧提取片段和手动分割病理视频的大型多模态模型，用于病理学计算中的自然诊断过程衔接 

---
# S3D: Sketch-Driven 3D Model Generation 

**Title (ZH)**: S3D: 草图驱动的3D模型生成 

**Authors**: Hail Song, Wonsik Shin, Naeun Lee, Soomin Chung, Nojun Kwak, Woontack Woo  

**Link**: [PDF](https://arxiv.org/pdf/2505.04185)  

**Abstract**: Generating high-quality 3D models from 2D sketches is a challenging task due to the inherent ambiguity and sparsity of sketch data. In this paper, we present S3D, a novel framework that converts simple hand-drawn sketches into detailed 3D models. Our method utilizes a U-Net-based encoder-decoder architecture to convert sketches into face segmentation masks, which are then used to generate a 3D representation that can be rendered from novel views. To ensure robust consistency between the sketch domain and the 3D output, we introduce a novel style-alignment loss that aligns the U-Net bottleneck features with the initial encoder outputs of the 3D generation module, significantly enhancing reconstruction fidelity. To further enhance the network's robustness, we apply augmentation techniques to the sketch dataset. This streamlined framework demonstrates the effectiveness of S3D in generating high-quality 3D models from sketch inputs. The source code for this project is publicly available at this https URL. 

**Abstract (ZH)**: 从2D草图生成高质量3D模型是一项具有挑战性的工作，因为草图数据存在固有的模糊性和稀疏性。本文提出了一种名为S3D的新框架，能够将简单的手绘草图转化为详细的3D模型。我们的方法利用基于U-Net的编码-解码架构将草图转换为面部分割掩码，然后利用这些掩码生成可以从新视角渲染的3D表示。为确保草图域与3D输出之间的稳健一致性，我们引入了一种新颖的风格对齐损失，将U-Net瓶颈特征与3D生成模块的初始编码输出对齐，显著提高了重建保真度。为了进一步增强网络的稳健性，我们对草图数据集应用了增强技术。这一精简框架证明了S3D在从草图输入生成高质量3D模型方面的有效性。该项目的源代码在此公开获取：这个 https URL。 

---
# DOTA: Deformable Optimized Transformer Architecture for End-to-End Text Recognition with Retrieval-Augmented Generation 

**Title (ZH)**: DOTA：用于检索增强生成的端到端文本识别可变形优化变压器架构 

**Authors**: Naphat Nithisopa, Teerapong Panboonyuen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04175)  

**Abstract**: Text recognition in natural images remains a challenging yet essential task, with broad applications spanning computer vision and natural language processing. This paper introduces a novel end-to-end framework that combines ResNet and Vision Transformer backbones with advanced methodologies, including Deformable Convolutions, Retrieval-Augmented Generation, and Conditional Random Fields (CRF). These innovations collectively enhance feature representation and improve Optical Character Recognition (OCR) performance. Specifically, the framework substitutes standard convolution layers in the third and fourth blocks with Deformable Convolutions, leverages adaptive dropout for regularization, and incorporates CRF for more refined sequence modeling. Extensive experiments conducted on six benchmark datasets IC13, IC15, SVT, IIIT5K, SVTP, and CUTE80 validate the proposed method's efficacy, achieving notable accuracies: 97.32% on IC13, 58.26% on IC15, 88.10% on SVT, 74.13% on IIIT5K, 82.17% on SVTP, and 66.67% on CUTE80, resulting in an average accuracy of 77.77%. These results establish a new state-of-the-art for text recognition, demonstrating the robustness of the approach across diverse and challenging datasets. 

**Abstract (ZH)**: 自然图像中的文本识别依然是一个具有广阔应用前景但又颇具挑战性的任务，涉及计算机视觉和自然语言处理领域。本文介绍了一种结合ResNet和Vision Transformer骨干网络，并采用变形卷积、检索增强生成、条件随机场等先进方法的端到端框架。这些创新共同提升了特征表示能力，提高了光学字符识别（OCR）的性能。具体而言，该框架在第三和第四块中替代了标准卷积层，采用自适应dropout进行正则化，并引入条件随机场进行更精细的序列建模。在IC13、IC15、SVT、IIIT5K、SVTP和CUTE80六个基准数据集上进行的广泛实验验证了所提方法的有效性，取得了显著的准确性：IC13为97.32%，IC15为58.26%，SVT为88.10%，IIIT5K为74.13%，SVTP为82.17%，CUTE80为66.67%，平均准确率为77.77%。这些结果表明，该方法在各类复杂数据集上具有鲁棒性，建立了新的文本识别状态-of-艺术水平。 

---
# On-Device LLM for Context-Aware Wi-Fi Roaming 

**Title (ZH)**: 基于设备的上下文感知Wi-Fi漫游大语言模型 

**Authors**: Ju-Hyung Lee, Yanqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04174)  

**Abstract**: Wireless roaming is a critical yet challenging task for maintaining seamless connectivity in dynamic mobile environments. Conventional threshold-based or heuristic schemes often fail, leading to either sticky or excessive handovers. We introduce the first cross-layer use of an on-device large language model (LLM): high-level reasoning in the application layer that issues real-time actions executed in the PHY/MAC stack. The LLM addresses two tasks: (i) context-aware AP selection, where structured prompts fuse environmental cues (e.g., location, time) to choose the best BSSID; and (ii) dynamic threshold adjustment, where the model adaptively decides when to roam. To satisfy the tight latency and resource budgets of edge hardware, we apply a suite of optimizations-chain-of-thought prompting, parameter-efficient fine-tuning, and quantization. Experiments on indoor and outdoor datasets show that our approach surpasses legacy heuristics and DRL baselines, achieving a strong balance between roaming stability and signal quality. These findings underscore the promise of application-layer LLM reasoning for lower-layer wireless control in future edge systems. 

**Abstract (ZH)**: 设备上的大语言模型在跨层中的无线漫游应用：基于高阶推理的实时动作优化 

---
# TS-SNN: Temporal Shift Module for Spiking Neural Networks 

**Title (ZH)**: TS-SNN: Temporal Shift 模块 for Spiking Neural Networks 

**Authors**: Kairong Yu, Tianqing Zhang, Qi Xu, Gang Pan, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04165)  

**Abstract**: Spiking Neural Networks (SNNs) are increasingly recognized for their biological plausibility and energy efficiency, positioning them as strong alternatives to Artificial Neural Networks (ANNs) in neuromorphic computing applications. SNNs inherently process temporal information by leveraging the precise timing of spikes, but balancing temporal feature utilization with low energy consumption remains a challenge. In this work, we introduce Temporal Shift module for Spiking Neural Networks (TS-SNN), which incorporates a novel Temporal Shift (TS) module to integrate past, present, and future spike features within a single timestep via a simple yet effective shift operation. A residual combination method prevents information loss by integrating shifted and original features. The TS module is lightweight, requiring only one additional learnable parameter, and can be seamlessly integrated into existing architectures with minimal additional computational cost. TS-SNN achieves state-of-the-art performance on benchmarks like CIFAR-10 (96.72\%), CIFAR-100 (80.28\%), and ImageNet (70.61\%) with fewer timesteps, while maintaining low energy consumption. This work marks a significant step forward in developing efficient and accurate SNN architectures. 

**Abstract (ZH)**: 基于时空移位模块的脉冲神经网络（TS-SNN）：高效且精确的时间和能量管理 

---
# R^3-VQA: "Read the Room" by Video Social Reasoning 

**Title (ZH)**: R^3-VQA: "读清环境"的视频社会推理 

**Authors**: Lixing Niu, Jiapeng Li, Xingping Yu, Shu Wang, Ruining Feng, Bo Wu, Ping Wei, Yisen Wang, Lifeng Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04147)  

**Abstract**: "Read the room" is a significant social reasoning capability in human daily life. Humans can infer others' mental states from subtle social cues. Previous social reasoning tasks and datasets lack complexity (e.g., simple scenes, basic interactions, incomplete mental state variables, single-step reasoning, etc.) and fall far short of the challenges present in real-life social interactions. In this paper, we contribute a valuable, high-quality, and comprehensive video dataset named R^3-VQA with precise and fine-grained annotations of social events and mental states (i.e., belief, intent, desire, and emotion) as well as corresponding social causal chains in complex social scenarios. Moreover, we include human-annotated and model-generated QAs. Our task R^3-VQA includes three aspects: Social Event Understanding, Mental State Estimation, and Social Causal Reasoning. As a benchmark, we comprehensively evaluate the social reasoning capabilities and consistencies of current state-of-the-art large vision-language models (LVLMs). Comprehensive experiments show that (i) LVLMs are still far from human-level consistent social reasoning in complex social scenarios; (ii) Theory of Mind (ToM) prompting can help LVLMs perform better on social reasoning tasks. We provide some of our dataset and codes in supplementary material and will release our full dataset and codes upon acceptance. 

**Abstract (ZH)**: "读厅内情"是人类日常生活中的一个重要社会推理能力。人类可以从微妙的社会线索中推断出他人的心理状态。以往的社会推理任务和数据集缺乏复杂性（例如，简单场景、基本互动、不完整的心理状态变量、单步推理等），远不能满足现实生活社会互动中的挑战。在本文中，我们贡献了一个名为R^3-VQA的高质量、全面的视频数据集，该数据集对社会事件和心理状态（即信念、意图、欲望和情绪）进行了精确和细致的标注，并包含了相应的社会因果链，在复杂的社会场景中。此外，我们还包含了人工标注和模型生成的问答。我们的任务R^3-VQA包括三个方面：社会事件理解、心理状态估计和社会因果推理。作为基准，我们全面评估了当前最先进的大规模视觉-语言模型（LVLMs）的社会推理能力和一致性。全面的实验结果显示：（i）在复杂社会场景中，LVLMs的社会推理一致性仍然远未达到人类水平；（ii）心智理论（ToM）提示可以促进LVLMs在社会推理任务中的表现。我们提供了一部分数据集和代码作为补充材料，并将在论文被接受后公布完整数据集和代码。 

---
# Unmasking the Canvas: A Dynamic Benchmark for Image Generation Jailbreaking and LLM Content Safety 

**Title (ZH)**: 揭开画布：图像生成解锁与LLM内容安全的动态基准 

**Authors**: Variath Madhupal Gautham Nair, Vishal Varma Dantuluri  

**Link**: [PDF](https://arxiv.org/pdf/2505.04146)  

**Abstract**: Existing large language models (LLMs) are advancing rapidly and produce outstanding results in image generation tasks, yet their content safety checks remain vulnerable to prompt-based jailbreaks. Through preliminary testing on platforms such as ChatGPT, MetaAI, and Grok, we observed that even short, natural prompts could lead to the generation of compromising images ranging from realistic depictions of forged documents to manipulated images of public figures.
We introduce Unmasking the Canvas (UTC Benchmark; UTCB), a dynamic and scalable benchmark dataset to evaluate LLM vulnerability in image generation. Our methodology combines structured prompt engineering, multilingual obfuscation (e.g., Zulu, Gaelic, Base64), and evaluation using Groq-hosted LLaMA-3. The pipeline supports both zero-shot and fallback prompting strategies, risk scoring, and automated tagging. All generations are stored with rich metadata and curated into Bronze (non-verified), Silver (LLM-aided verification), and Gold (manually verified) tiers. UTCB is designed to evolve over time with new data sources, prompt templates, and model behaviors.
Warning: This paper includes visual examples of adversarial inputs designed to test model safety. All outputs have been redacted to ensure responsible disclosure. 

**Abstract (ZH)**: 揭示画布：评估大语言模型在图像生成中的漏洞基准（UTC Benchmark；UTCB） 

---
# Bringing legal knowledge to the public by constructing a legal question bank using large-scale pre-trained language model 

**Title (ZH)**: 通过构建大规模预训练语言模型法律问题库将法律知识普及于公众 

**Authors**: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu, Michael M. K. Cheung, Henry W. H. Chan, Anne S. Y. Cheung, Felix W. H. Chan, Yongxi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04132)  

**Abstract**: Access to legal information is fundamental to access to justice. Yet accessibility refers not only to making legal documents available to the public, but also rendering legal information comprehensible to them. A vexing problem in bringing legal information to the public is how to turn formal legal documents such as legislation and judgments, which are often highly technical, to easily navigable and comprehensible knowledge to those without legal education. In this study, we formulate a three-step approach for bringing legal knowledge to laypersons, tackling the issues of navigability and comprehensibility. First, we translate selected sections of the law into snippets (called CLIC-pages), each being a small piece of article that focuses on explaining certain technical legal concept in layperson's terms. Second, we construct a Legal Question Bank (LQB), which is a collection of legal questions whose answers can be found in the CLIC-pages. Third, we design an interactive CLIC Recommender (CRec). Given a user's verbal description of a legal situation that requires a legal solution, CRec interprets the user's input and shortlists questions from the question bank that are most likely relevant to the given legal situation and recommends their corresponding CLIC pages where relevant legal knowledge can be found. In this paper we focus on the technical aspects of creating an LQB. We show how large-scale pre-trained language models, such as GPT-3, can be used to generate legal questions. We compare machine-generated questions (MGQs) against human-composed questions (HCQs) and find that MGQs are more scalable, cost-effective, and more diversified, while HCQs are more precise. We also show a prototype of CRec and illustrate through an example how our 3-step approach effectively brings relevant legal knowledge to the public. 

**Abstract (ZH)**: 获取法律信息是获得正义的基础。然而，可访问性不仅指将法律文件提供给公众，还指使法律信息对公众易于理解和运用。将正式的法律文件，如立法和判决，转化为非法律背景人员易于导航和理解的知识是一个棘手的问题。在本研究中，我们提出了一种三步方法来将法律知识带给普通民众，解决可导航性和可理解性的问题。首先，我们将法律中的选定部分翻译成短片段（称为CLIC页面），每个片段专注于用非法律术语解释某一特定的技术法律概念。其次，我们构建了一个法律问题银行（LQB），这是一个集合了一系列法律问题的库，其答案可以在CLIC页面中找到。第三，我们设计了一个交互式的CLIC推荐系统（CRec）。根据用户对需要法律解决方案的法律情景的口头描述，CRec解释用户的输入，并从问题库中筛选出最有可能与给定法律情景相关的提问，并推荐相应的CLIC页面，以找到相关的法律知识。本文我们专注于法律问题银行的技术方面。我们展示如何使用大规模预训练语言模型（如GPT-3）生成法律问题。我们将机器生成的问题（MGQs）与人类编写的提问（HCQs）进行了比较，发现MGQs更具可扩展性、成本效益和多样性，而HCQs则更为精确。我们也展示了CRec的原型，并通过一个示例说明了我们三步方法如何有效将相关法律知识带给公众。 

---
# LLMs' Suitability for Network Security: A Case Study of STRIDE Threat Modeling 

**Title (ZH)**: LLMs在网络安全领域的适用性：基于STRIDE威胁建模的案例研究 

**Authors**: AbdulAziz AbdulGhaffar, Ashraf Matrawy  

**Link**: [PDF](https://arxiv.org/pdf/2505.04101)  

**Abstract**: Artificial Intelligence (AI) is expected to be an integral part of next-generation AI-native 6G networks. With the prevalence of AI, researchers have identified numerous use cases of AI in network security. However, there are almost nonexistent studies that analyze the suitability of Large Language Models (LLMs) in network security. To fill this gap, we examine the suitability of LLMs in network security, particularly with the case study of STRIDE threat modeling. We utilize four prompting techniques with five LLMs to perform STRIDE classification of 5G threats. From our evaluation results, we point out key findings and detailed insights along with the explanation of the possible underlying factors influencing the behavior of LLMs in the modeling of certain threats. The numerical results and the insights support the necessity for adjusting and fine-tuning LLMs for network security use cases. 

**Abstract (ZH)**: 人工智能（AI）有望成为下一代AI原生6G网络不可或缺的一部分。随着AI的普及，研究人员已经识别出许多AI在网络安全性方面的应用案例。然而，几乎不存在分析大型语言模型（LLMs）在网络安全性方面适用性的研究。为填补这一空白，我们检查了LLMs在网络安全性方面的适用性，特别是通过STRIDE威胁建模案例研究。我们利用四种提示技术与五种LLM对5G威胁进行STRIDE分类。从我们的评估结果中，我们指出了关键发现和详细见解，并解释了可能影响LLMs在某些威胁建模中行为的潜在因素。数值结果和见解支持调整和微调LLMs以适应网络安全性应用场景的必要性。 

---
# An Empirical Study of OpenAI API Discussions on Stack Overflow 

**Title (ZH)**: OpenAI API讨论在Stack Overflow上的实证研究 

**Authors**: Xiang Chen, Jibin Wang, Chaoyang Gao, Xiaolin Ju, Zhanqi Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04084)  

**Abstract**: The rapid advancement of large language models (LLMs), represented by OpenAI's GPT series, has significantly impacted various domains such as natural language processing, software development, education, healthcare, finance, and scientific research. However, OpenAI APIs introduce unique challenges that differ from traditional APIs, such as the complexities of prompt engineering, token-based cost management, non-deterministic outputs, and operation as black boxes. To the best of our knowledge, the challenges developers encounter when using OpenAI APIs have not been explored in previous empirical studies. To fill this gap, we conduct the first comprehensive empirical study by analyzing 2,874 OpenAI API-related discussions from the popular Q&A forum Stack Overflow. We first examine the popularity and difficulty of these posts. After manually categorizing them into nine OpenAI API-related categories, we identify specific challenges associated with each category through topic modeling analysis. Based on our empirical findings, we finally propose actionable implications for developers, LLM vendors, and researchers. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进展，以OpenAI的GPT系列为代表，对自然语言处理、软件开发、教育、医疗、金融和科学研究等领域产生了显著影响。然而，OpenAI APIs引入了与传统APIs不同的独特挑战，如提示工程的复杂性、基于令牌的成本管理、非确定性输出以及以黑盒方式运行。据我们所知，之前的经验研究尚未探讨开发者在使用OpenAI APIs时遇到的挑战。为了填补这一空白，我们首次通过分析来自受欢迎的问答论坛Stack Overflow的2,874条OpenAI API相关讨论，进行全面的经验研究。我们首先分析这些帖子的流行程度和难度，然后通过主题建模分析将它们手动归类为九个OpenAI API相关类别，并识别每个类别特有的挑战。基于我们的实证发现，我们最终为开发者、LLM供应商和研究人员提出了可操作的建议。 

---
# Plexus: Taming Billion-edge Graphs with 3D Parallel GNN Training 

**Title (ZH)**: Plexus: 三维并行GNN训练驯化亿级边图 

**Authors**: Aditya K. Ranjan, Siddharth Singh, Cunyang Wei, Abhinav Bhatele  

**Link**: [PDF](https://arxiv.org/pdf/2505.04083)  

**Abstract**: Graph neural networks have emerged as a potent class of neural networks capable of leveraging the connectivity and structure of real-world graphs to learn intricate properties and relationships between nodes. Many real-world graphs exceed the memory capacity of a GPU due to their sheer size, and using GNNs on them requires techniques such as mini-batch sampling to scale. However, this can lead to reduced accuracy in some cases, and sampling and data transfer from the CPU to the GPU can also slow down training. On the other hand, distributed full-graph training suffers from high communication overhead and load imbalance due to the irregular structure of graphs. We propose Plexus, a three-dimensional (3D) parallel approach for full-graph training that tackles these issues and scales to billion-edge graphs. Additionally, we introduce optimizations such as a permutation scheme for load balancing, and a performance model to predict the optimal 3D configuration. We evaluate Plexus on several graph datasets and show scaling results for up to 2048 GPUs on Perlmutter, which is 33% of the machine, and 2048 GCDs on Frontier. Plexus achieves unprecedented speedups of 2.3x-12.5x over existing methods and a reduction in the time to solution by 5.2-8.7x on Perlmutter and 7-54.2x on Frontier. 

**Abstract (ZH)**: Plexus：一种三维并行方法，用于处理十亿边图的全图训练 

---
# LLM-e Guess: Can LLMs Capabilities Advance Without Hardware Progress? 

**Title (ZH)**: LLM-e 猜想：在硬件进步之外，LLM 能力能否得以提升？ 

**Authors**: Teddy Foley, Spencer Guo, Henry Josephson, Anqi Qu, Jack Sanderson  

**Link**: [PDF](https://arxiv.org/pdf/2505.04075)  

**Abstract**: This paper examines whether large language model (LLM) capabilities can continue to advance without additional compute by analyzing the development and role of algorithms used in state-of-the-art LLMs. Motivated by regulatory efforts that have largely focused on restricting access to high-performance hardware, we ask: Can LLMs progress in a compute-constrained environment, and how do algorithmic innovations perform under such conditions?
To address these questions, we introduce a novel classification framework that distinguishes between compute-dependent innovations -- which yield disproportionate benefits at high compute levels (e.g., the Transformer architecture and mixture-of-experts models) and compute-independent innovations, which improve efficiency across all compute scales (e.g., rotary positional encoding, FlashAttention, or layer normalization). We quantify these contributions using a metric called compute-equivalent gain (CEG), which estimates the additional compute that would be required to achieve similar improvements without these algorithmic advancements.
To validate this framework, we conduct small-scale training experiments with a scaled-down GPT-2 model. Our results confirm that compute-independent advancements yield meaningful performance gains even in resource-constrained settings, with a CEG of up to $3.5\times$ over a baseline model. By contrast, compute-dependent advancements provided little benefit or even degraded performance at the small scale, reinforcing the importance of compute availability for certain algorithmic gains. 

**Abstract (ZH)**: 本文通过分析最先进的大型语言模型中使用的算法发展和作用，探讨大型语言模型的能力是否能够在无需额外计算资源的情况下继续进步。鉴于监管努力主要集中在限制高性能硬件的访问上，我们提出的问题是：在计算资源受限的环境中，大型语言模型能否进步，以及在这些条件下算法创新的表现如何？

为了回答这些问题，我们引入了一种新的分类框架，区分计算依赖型创新（在高计算水平下可获得不成比例的好处，例如Transformer架构和expert混合模型）和计算独立型创新（在所有计算规模下提高效率，例如旋转位置编码、FlashAttention或层规范化）。我们使用计算等效增益（CEG）这一度量来量化这些贡献，CEG估计了在没有这些算法进步的情况下实现相似改进所需的额外计算资源。

为了验证这一框架，我们在一个缩放后的GPT-2模型上进行了小规模训练实验。结果表明，即使在资源受限的环境中，计算独立型进步也能带来显著的性能提升，CEG最高达到3.5倍的基础模型水平。相比之下，在小规模计算环境下，计算依赖型进步几乎没有益处甚至降低了性能，这进一步强调了特定算法进步的计算资源重要性。 

---
# Advancing and Benchmarking Personalized Tool Invocation for LLMs 

**Title (ZH)**: 个性化工具调用 advancement 和基准测试 for LLMs 

**Authors**: Xu Huang, Yuefeng Huang, Weiwen Liu, Xingshan Zeng, Yasheng Wang, Ruiming Tang, Hong Xie, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04072)  

**Abstract**: Tool invocation is a crucial mechanism for extending the capabilities of Large Language Models (LLMs) and has recently garnered significant attention. It enables LLMs to solve complex problems through tool calls while accessing up-to-date world knowledge. However, existing work primarily focuses on the fundamental ability of LLMs to invoke tools for problem-solving, without considering personalized constraints in tool invocation. In this work, we introduce the concept of Personalized Tool Invocation and define two key tasks: Tool Preference and Profile-dependent Query. Tool Preference addresses user preferences when selecting among functionally similar tools, while Profile-dependent Query considers cases where a user query lacks certain tool parameters, requiring the model to infer them from the user profile. To tackle these challenges, we propose PTool, a data synthesis framework designed for personalized tool invocation. Additionally, we construct \textbf{PTBench}, the first benchmark for evaluating personalized tool invocation. We then fine-tune various open-source models, demonstrating the effectiveness of our framework and providing valuable insights. Our benchmark is public at this https URL. 

**Abstract (ZH)**: 个性化工具调用是大型语言模型（LLMs）能力扩展的关键机制，近期引发了广泛关注。它使LLMs能够通过工具调用解决复杂问题，同时访问最新的世界知识。然而，现有研究主要集中在LLMs的基本工具调用能力上，忽略了工具调用中的个性化约束。在本文中，我们引入了个性化工具调用的概念，并定义了两个关键任务：工具偏好和基于个人资料的查询。工具偏好解决在选择功能相似的工具时用户偏好问题，而基于个人资料的查询考虑了用户查询缺少某些工具参数的情况，要求模型从用户个人资料中推断这些参数。为应对这些挑战，我们提出了PTool，一种为个性化工具调用设计的数据合成框架。此外，我们构建了PTBench，这是第一个评估个性化工具调用的基准。随后，我们对各种开源模型进行了微调，展示了该框架的有效性并提供了宝贵见解。我们的基准可以在此网址访问：this https URL。 

---
# Izhikevich-Inspired Temporal Dynamics for Enhancing Privacy, Efficiency, and Transferability in Spiking Neural Networks 

**Title (ZH)**: 受Izhikevich启发的时序动态以增强脉冲神经网络中的隐私性、效率和可迁移性 

**Authors**: Ayana Moshruba, Hamed Poursiami, Maryam Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2505.04034)  

**Abstract**: Biological neurons exhibit diverse temporal spike patterns, which are believed to support efficient, robust, and adaptive neural information processing. While models such as Izhikevich can replicate a wide range of these firing dynamics, their complexity poses challenges for directly integrating them into scalable spiking neural networks (SNN) training pipelines. In this work, we propose two probabilistically driven, input-level temporal spike transformations: Poisson-Burst and Delayed-Burst that introduce biologically inspired temporal variability directly into standard Leaky Integrate-and-Fire (LIF) neurons. This enables scalable training and systematic evaluation of how spike timing dynamics affect privacy, generalization, and learning performance. Poisson-Burst modulates burst occurrence based on input intensity, while Delayed-Burst encodes input strength through burst onset timing. Through extensive experiments across multiple benchmarks, we demonstrate that Poisson-Burst maintains competitive accuracy and lower resource overhead while exhibiting enhanced privacy robustness against membership inference attacks, whereas Delayed-Burst provides stronger privacy protection at a modest accuracy trade-off. These findings highlight the potential of biologically grounded temporal spike dynamics in improving the privacy, generalization and biological plausibility of neuromorphic learning systems. 

**Abstract (ZH)**: 生物神经元表现出多样化的时空放电模式，这些模式被认为支持高效、稳健和适应性的神经信息处理。虽然Izhikevich等模型可以复制这些放电动态的广泛范围，但它们的复杂性为直接将它们集成到可扩展的脉冲神经网络（SNN）训练管道中带来了挑战。在本工作中，我们提出了两种基于概率的输入级时空放电变换：Poisson-Burst和Delayed-Burst，直接将生物启发的时空变异性引入标准的泄漏型积分-发放（LIF）神经元中。这使得可以实现可扩展的训练并系统地评估放电时间动态如何影响隐私、泛化和学习性能。Poisson-Burst基于输入强度调节爆发的发生，而Delayed-Burst通过爆发起始时间编码输入强度。通过在多个基准上的广泛实验，我们证明Poisson-Burst在保持竞争力的同时降低了资源开销，且在对抗成员推理攻击方面的隐私鲁棒性有所提升，而Delayed-Burst在轻微的准确性折衷下提供了更强的隐私保护。这些发现强调了基于生物过程的时间放电动态在提高神经形态学习系统中的隐私性、泛化能力和生物可行性方面的潜在价值。 

---
# Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving 

**Title (ZH)**: 棱镜： unleashing GPU sharing for cost-efficient multi-LLM serving 

**Authors**: Shan Yu, Jiarong Xing, Yifan Qiao, Mingyuan Ma, Yangmin Li, Yang Wang, Shuo Yang, Zhiqiang Xie, Shiyi Cao, Ke Bao, Ion Stoica, Harry Xu, Ying Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04021)  

**Abstract**: Serving large language models (LLMs) is expensive, especially for providers hosting many models, making cost reduction essential. The unique workload patterns of serving multiple LLMs (i.e., multi-LLM serving) create new opportunities and challenges for this task. The long-tail popularity of models and their long idle periods present opportunities to improve utilization through GPU sharing. However, existing GPU sharing systems lack the ability to adjust their resource allocation and sharing policies at runtime, making them ineffective at meeting latency service-level objectives (SLOs) under rapidly fluctuating workloads.
This paper presents Prism, a multi-LLM serving system that unleashes the full potential of GPU sharing to achieve both cost efficiency and SLO attainment. At its core, Prism tackles a key limitation of existing systems$\unicode{x2014}$the lack of $\textit{cross-model memory coordination}$, which is essential for flexibly sharing GPU memory across models under dynamic workloads. Prism achieves this with two key designs. First, it supports on-demand memory allocation by dynamically mapping physical to virtual memory pages, allowing flexible memory redistribution among models that space- and time-share a GPU. Second, it improves memory efficiency through a two-level scheduling policy that dynamically adjusts sharing strategies based on models' runtime demands. Evaluations on real-world traces show that Prism achieves more than $2\times$ cost savings and $3.3\times$ SLO attainment compared to state-of-the-art systems. 

**Abstract (ZH)**: Prism：一种实现多大型语言模型高效服务和SLO达成的GPU共享系统 

---
# SLOT: Structuring the Output of Large Language Models 

**Title (ZH)**: SLOT：结构化大型语言模型的输出 

**Authors**: Darren Yow-Bang Wang, Zhengyuan Shen, Soumya Smruti Mishra, Zhichao Xu, Yifei Teng, Haibo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.04016)  

**Abstract**: Structured outputs are essential for large language models (LLMs) in critical applications like agents and information extraction. Despite their capabilities, LLMs often generate outputs that deviate from predefined schemas, significantly hampering reliable application development. We present SLOT (Structured LLM Output Transformer), a model-agnostic approach that transforms unstructured LLM outputs into precise structured formats. While existing solutions predominantly rely on constrained decoding techniques or are tightly coupled with specific models, SLOT employs a fine-tuned lightweight language model as a post-processing layer, achieving flexibility across various LLMs and schema specifications. We introduce a systematic pipeline for data curation and synthesis alongside a formal evaluation methodology that quantifies both schema accuracy and content fidelity. Our results demonstrate that fine-tuned Mistral-7B model with constrained decoding achieves near perfect schema accuracy (99.5%) and content similarity (94.0%), outperforming Claude-3.5-Sonnet by substantial margins (+25 and +20 percentage points, respectively). Notably, even compact models like Llama-3.2-1B can match or exceed the structured output capabilities of much larger proprietary models when equipped with SLOT, enabling reliable structured generation in resource-constrained environments. 

**Abstract (ZH)**: 结构化输出对于大型语言模型在智能代理和信息提取等关键应用中的应用至关重要。尽管具备强大的能力，大型语言模型（LLMs）经常生成不符合预定义模式的输出，严重影响了可靠应用的开发。我们提出了SLOT（结构化LLM输出转换器），这是一种模型无关的方法，将不结构化的LLM输出转换为精确的结构化格式。现有解决方案主要依赖受限解码技术或与特定模型紧密结合，而SLOT采用微调的轻量级语言模型作为后处理层，实现了对各种LLMs和模式规范的高度灵活性。我们介绍了一种系统化的数据采集和合成管道，以及一种正式的评估方法，该方法量化了模式准确性和内容保真度。实验结果表明，使用受限解码技术微调的Mistral-7B模型达到了近完美的模式准确率（99.5%）和内容相似度（94.0%），显著优于Claude-3.5-Sonnet（分别高出25和20个百分点）。值得注意的是，即使像Llama-3.2-1B这样的紧凑型模型，配备了SLOT后，也能达到甚至超过许多更大且专门为特定任务设计的模型的结构化输出能力，从而在资源受限的环境中实现可靠的结构化生成。 

---
# MergeGuard: Efficient Thwarting of Trojan Attacks in Machine Learning Models 

**Title (ZH)**: MergeGuard: 在机器学习模型中高效对抗木马攻击的方法 

**Authors**: Soheil Zibakhsh Shabgahi, Yaman Jandali, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2505.04015)  

**Abstract**: This paper proposes MergeGuard, a novel methodology for mitigation of AI Trojan attacks. Trojan attacks on AI models cause inputs embedded with triggers to be misclassified to an adversary's target class, posing a significant threat to model usability trained by an untrusted third party. The core of MergeGuard is a new post-training methodology for linearizing and merging fully connected layers which we show simultaneously improves model generalizability and performance. Our Proof of Concept evaluation on Transformer models demonstrates that MergeGuard maintains model accuracy while decreasing trojan attack success rate, outperforming commonly used (post-training) Trojan mitigation by fine-tuning methodologies. 

**Abstract (ZH)**: MergeGuard：一种新型的AI木马攻击缓解方法 

---
# PARC: Physics-based Augmentation with Reinforcement Learning for Character Controllers 

**Title (ZH)**: PARC：基于物理的增强学习字符控制器辅助方法 

**Authors**: Michael Xu, Yi Shi, KangKang Yin, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04002)  

**Abstract**: Humans excel in navigating diverse, complex environments with agile motor skills, exemplified by parkour practitioners performing dynamic maneuvers, such as climbing up walls and jumping across gaps. Reproducing these agile movements with simulated characters remains challenging, in part due to the scarcity of motion capture data for agile terrain traversal behaviors and the high cost of acquiring such data. In this work, we introduce PARC (Physics-based Augmentation with Reinforcement Learning for Character Controllers), a framework that leverages machine learning and physics-based simulation to iteratively augment motion datasets and expand the capabilities of terrain traversal controllers. PARC begins by training a motion generator on a small dataset consisting of core terrain traversal skills. The motion generator is then used to produce synthetic data for traversing new terrains. However, these generated motions often exhibit artifacts, such as incorrect contacts or discontinuities. To correct these artifacts, we train a physics-based tracking controller to imitate the motions in simulation. The corrected motions are then added to the dataset, which is used to continue training the motion generator in the next iteration. PARC's iterative process jointly expands the capabilities of the motion generator and tracker, creating agile and versatile models for interacting with complex environments. PARC provides an effective approach to develop controllers for agile terrain traversal, which bridges the gap between the scarcity of motion data and the need for versatile character controllers. 

**Abstract (ZH)**: 基于物理增强与强化学习的地形导航角色控制器框架 

---
# Can Large Language Models Predict Parallel Code Performance? 

**Title (ZH)**: 大型语言模型能否预测并行代码性能？ 

**Authors**: Gregory Bolet, Giorgis Georgakoudis, Harshitha Menon, Konstantinos Parasyris, Niranjan Hasabnis, Hayden Estes, Kirk W. Cameron, Gal Oren  

**Link**: [PDF](https://arxiv.org/pdf/2505.03988)  

**Abstract**: Accurate determination of the performance of parallel GPU code typically requires execution-time profiling on target hardware -- an increasingly prohibitive step due to limited access to high-end GPUs. This paper explores whether Large Language Models (LLMs) can offer an alternative approach for GPU performance prediction without relying on hardware. We frame the problem as a roofline classification task: given the source code of a GPU kernel and the hardware specifications of a target GPU, can an LLM predict whether the GPU kernel is compute-bound or bandwidth-bound?
For this study, we build a balanced dataset of 340 GPU kernels, obtained from HeCBench benchmark and written in CUDA and OpenMP, along with their ground-truth labels obtained via empirical GPU profiling. We evaluate LLMs across four scenarios: (1) with access to profiling data of the kernel source, (2) zero-shot with source code only, (3) few-shot with code and label pairs, and (4) fine-tuned on a small custom dataset.
Our results show that state-of-the-art LLMs have a strong understanding of the Roofline model, achieving 100% classification accuracy when provided with explicit profiling data. We also find that reasoning-capable LLMs significantly outperform standard LLMs in zero- and few-shot settings, achieving up to 64% accuracy on GPU source codes, without profiling information. Lastly, we find that LLM fine-tuning will require much more data than what we currently have available.
This work is among the first to use LLMs for source-level roofline performance prediction via classification, and illustrates their potential to guide optimization efforts when runtime profiling is infeasible. Our findings suggest that with better datasets and prompt strategies, LLMs could become practical tools for HPC performance analysis and performance portability. 

**Abstract (ZH)**: 利用大规模语言模型进行GPU性能预测的研究 

---
# Diffusion Models are Secretly Exchangeable: Parallelizing DDPMs via Autospeculation 

**Title (ZH)**: 扩散模型实际上具有交换性：通过自动推测并行化DDPMs 

**Authors**: Hengyuan Hu, Aniket Das, Dorsa Sadigh, Nima Anari  

**Link**: [PDF](https://arxiv.org/pdf/2505.03983)  

**Abstract**: Denoising Diffusion Probabilistic Models (DDPMs) have emerged as powerful tools for generative modeling. However, their sequential computation requirements lead to significant inference-time bottlenecks. In this work, we utilize the connection between DDPMs and Stochastic Localization to prove that, under an appropriate reparametrization, the increments of DDPM satisfy an exchangeability property. This general insight enables near-black-box adaptation of various performance optimization techniques from autoregressive models to the diffusion setting. To demonstrate this, we introduce \emph{Autospeculative Decoding} (ASD), an extension of the widely used speculative decoding algorithm to DDPMs that does not require any auxiliary draft models. Our theoretical analysis shows that ASD achieves a $\tilde{O} (K^{\frac{1}{3}})$ parallel runtime speedup over the $K$ step sequential DDPM. We also demonstrate that a practical implementation of autospeculative decoding accelerates DDPM inference significantly in various domains. 

**Abstract (ZH)**: 去噪扩散概率模型（DDPMs）已经成为生成建模的强大工具。然而，它们的顺序计算要求导致了推断时的重大瓶颈。在本文中，我们利用DDPMs与随机局部化的连接，证明在适当的重构参数下，DDPM的增量满足可交换性。这一一般见解使得可以将各种性能优化技术从自回归模型直接适应到扩散设置中。为了证明这一点，我们引入了\emph{自推测解码}（ASD），这是一种将广泛使用的推测解码算法扩展到DDPMs的方法，无需任何辅助草图模型。我们的理论分析表明，ASD在$K$步顺序DDPM上实现了$\tilde{O}(K^{\frac{1}{3}})$的并行运行时间加速。我们还展示了自推测解码的实用实现可以显著加速DDPM在各种领域的推断过程。 

---
# X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains 

**Title (ZH)**: X-推理器：跨模态与领域的可泛化推理 

**Authors**: Qianchu Liu, Sheng Zhang, Guanghui Qin, Timothy Ossowski, Yu Gu, Ying Jin, Sid Kiblawi, Sam Preston, Mu Wei, Paul Vozila, Tristan Naumann, Hoifung Poon  

**Link**: [PDF](https://arxiv.org/pdf/2505.03981)  

**Abstract**: Recent proprietary models (e.g., o3) have begun to demonstrate strong multimodal reasoning capabilities. Yet, most existing open-source research concentrates on training text-only reasoning models, with evaluations limited to mainly mathematical and general-domain tasks. Therefore, it remains unclear how to effectively extend reasoning capabilities beyond text input and general domains. This paper explores a fundamental research question: Is reasoning generalizable across modalities and domains? Our findings support an affirmative answer: General-domain text-based post-training can enable such strong generalizable reasoning. Leveraging this finding, we introduce X-Reasoner, a vision-language model post-trained solely on general-domain text for generalizable reasoning, using a two-stage approach: an initial supervised fine-tuning phase with distilled long chain-of-thoughts, followed by reinforcement learning with verifiable rewards. Experiments show that X-Reasoner successfully transfers reasoning capabilities to both multimodal and out-of-domain settings, outperforming existing state-of-the-art models trained with in-domain and multimodal data across various general and medical benchmarks (Figure 1). Additionally, we find that X-Reasoner's performance in specialized domains can be further enhanced through continued training on domain-specific text-only data. Building upon this, we introduce X-Reasoner-Med, a medical-specialized variant that achieves new state of the art on numerous text-only and multimodal medical benchmarks. 

**Abstract (ZH)**: Recent Proprietary Models (e.g., o3) Have Begun to Demonstrate Strong Multimodal Reasoning Capabilities: Is Reasoning Generalizable Across Modalities and Domains? X-Reasoner Enables Such Generalizable Reasoning Through Post-Training on General-Domain Text 

---
# Deep Learning Framework for Infrastructure Maintenance: Crack Detection and High-Resolution Imaging of Infrastructure Surfaces 

**Title (ZH)**: 基于深度学习的基础设施维护框架：裂缝检测与基础设施表面高分辨率成像 

**Authors**: Nikhil M. Pawar, Jorge A. Prozzi, Feng Hong, Surya Sarat Chandra Congress  

**Link**: [PDF](https://arxiv.org/pdf/2505.03974)  

**Abstract**: Recently, there has been an impetus for the application of cutting-edge data collection platforms such as drones mounted with camera sensors for infrastructure asset management. However, the sensor characteristics, proximity to the structure, hard-to-reach access, and environmental conditions often limit the resolution of the datasets. A few studies used super-resolution techniques to address the problem of low-resolution images. Nevertheless, these techniques were observed to increase computational cost and false alarms of distress detection due to the consideration of all the infrastructure images i.e., positive and negative distress classes. In order to address the pre-processing of false alarm and achieve efficient super-resolution, this study developed a framework consisting of convolutional neural network (CNN) and efficient sub-pixel convolutional neural network (ESPCNN). CNN accurately classified both the classes. ESPCNN, which is the lightweight super-resolution technique, generated high-resolution infrastructure image of positive distress obtained from CNN. The ESPCNN outperformed bicubic interpolation in all the evaluation metrics for super-resolution. Based on the performance metrics, the combination of CNN and ESPCNN was observed to be effective in preprocessing the infrastructure images with negative distress, reducing the computational cost and false alarms in the next step of super-resolution. The visual inspection showed that EPSCNN is able to capture crack propagation, complex geometry of even minor cracks. The proposed framework is expected to help the highway agencies in accurately performing distress detection and assist in efficient asset management practices. 

**Abstract (ZH)**: 近年来，搭载摄像头传感器的无人机等前沿数据采集平台在基础设施资产管理中的应用得到了推动。然而，传感器特性、结构附近的接近性、难以到达的访问条件以及环境状况往往限制了数据集的分辨率。少数研究使用超分辨率技术来解决低分辨率图像的问题。尽管如此，这些技术由于考虑了所有基础设施图像（包括正负应力类别）而被观察到增加了计算成本和应力检测的假报警率。为了预先处理假报警并实现有效的超分辨率，本研究开发了一个框架，该框架由卷积神经网络（CNN）和高效子像素卷积神经网络（ESPCNN）组成。CNN精确地分类了两类。ESPCNN，作为轻量级的超分辨率技术，生成了由CNN获取的正应力类的高分辨率基础设施图像。在所有超分辨率评价指标中，ESPCNN的表现优于双立方插值。根据性能指标，CNN和ESPCNN的组合被观察到在预处理包含负应力的基础设施图像、降低后续超分辨率步骤中的计算成本和假报警率方面是有效的。视觉检查表明，ESPCNN能够捕捉裂缝传播、复杂几何结构，甚至细微裂缝。所提出框架有望帮助高速公路管理部门准确进行损伤检测，并辅助高效的资产管理实践。 

---
# Decentralized Distributed Proximal Policy Optimization (DD-PPO) for High Performance Computing Scheduling on Multi-User Systems 

**Title (ZH)**: 多用户系统中高性能计算调度的去中心化分布式近端策略优化（DD-PPO） 

**Authors**: Matthew Sgambati, Aleksandar Vakanski, Matthew Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2505.03946)  

**Abstract**: Resource allocation in High Performance Computing (HPC) environments presents a complex and multifaceted challenge for job scheduling algorithms. Beyond the efficient allocation of system resources, schedulers must account for and optimize multiple performance metrics, including job wait time and system utilization. While traditional rule-based scheduling algorithms dominate the current deployments of HPC systems, the increasing heterogeneity and scale of those systems is expected to challenge the efficiency and flexibility of those algorithms in minimizing job wait time and maximizing utilization. Recent research efforts have focused on leveraging advancements in Reinforcement Learning (RL) to develop more adaptable and intelligent scheduling strategies. Recent RL-based scheduling approaches have explored a range of algorithms, from Deep Q-Networks (DQN) to Proximal Policy Optimization (PPO), and more recently, hybrid methods that integrate Graph Neural Networks with RL techniques. However, a common limitation across these methods is their reliance on relatively small datasets, and these methods face scalability issues when using large datasets. This study introduces a novel RL-based scheduler utilizing the Decentralized Distributed Proximal Policy Optimization (DD-PPO) algorithm, which supports large-scale distributed training across multiple workers without requiring parameter synchronization at every step. By eliminating reliance on centralized updates to a shared policy, the DD-PPO scheduler enhances scalability, training efficiency, and sample utilization. The validation dataset leveraged over 11.5 million real HPC job traces for comparing DD-PPO performance between traditional and advanced scheduling approaches, and the experimental results demonstrate improved scheduling performance in comparison to both rule-based schedulers and existing RL-based scheduling algorithms. 

**Abstract (ZH)**: 高性能计算（HPC）环境中资源分配为作业调度算法提出了一个复杂且多面的挑战。除了有效地分配系统资源外，调度器必须考虑和优化多个性能指标，包括作业等待时间和系统利用率。虽然传统的基于规则的调度算法目前占据了HPC系统的部署，但那些系统的日益异构性和规模预计将对这些算法在最小化作业等待时间和最大化利用率方面的效率和灵活性提出挑战。最近的研究工作集中在利用强化学习（RL）的最新进展来开发更加适应性和智能的调度策略。最近基于RL的调度方法探索了从深度Q网络（DQN）到正则化策略优化（PPO）等一系列算法，并且最近还开发了将图神经网络与RL技术结合的混合方法。然而，这些方法的一个常见局限是它们依赖于相对较小的数据集，并且在使用大数据集时面临可扩展性问题。本研究提出了一种新型的基于RL的调度器，采用分散分布式正则化策略优化（DD-PPO）算法，在多个工人之间实现大规模分布式训练而无需在每一步都进行参数同步。通过消除对集中更新共享策略的依赖，DD-PPO调度器增强了可扩展性、训练效率和样本利用。实验利用超过1150万个实际HPC作业轨迹的数据集来对比DD-PPO与其他传统和高级调度方法的性能，实验结果表明，与基于规则的调度器和现有的基于RL的调度算法相比，DD-PPO调度器在调度性能上取得了改进。 

---
# AI-Driven Security in Cloud Computing: Enhancing Threat Detection, Automated Response, and Cyber Resilience 

**Title (ZH)**: AI驱动的云 computing 安全：提升威胁检测、自动化应对和网络安全韧性 

**Authors**: Shamnad Mohamed Shaffi, Sunish Vengathattil, Jezeena Nikarthil Sidhick, Resmi Vijayan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03945)  

**Abstract**: Cloud security concerns have been greatly realized in recent years due to the increase of complicated threats in the computing world. Many traditional solutions do not work well in real-time to detect or prevent more complex threats. Artificial intelligence is today regarded as a revolution in determining a protection plan for cloud data architecture through machine learning, statistical visualization of computing infrastructure, and detection of security breaches followed by counteraction. These AI-enabled systems make work easier as more network activities are scrutinized, and any anomalous behavior that might be a precursor to a more serious breach is prevented. This paper examines ways AI can enhance cloud security by applying predictive analytics, behavior-based security threat detection, and AI-stirring encryption. It also outlines the problems of the previous security models and how AI overcomes them. For a similar reason, issues like data privacy, biases in the AI model, and regulatory compliance are also covered. So, AI improves the protection of cloud computing contexts; however, more efforts are needed in the subsequent phases to extend the technology's reliability, modularity, and ethical aspects. This means that AI can be blended with other new computing technologies, including blockchain, to improve security frameworks further. The paper discusses the current trends in securing cloud data architecture using AI and presents further research and application directions. 

**Abstract (ZH)**: 近年来，由于计算世界中复杂威胁的增加，云安全问题得到了广泛关注。许多传统解决方案无法在实时检测或预防更复杂的威胁中发挥良好作用。人工智能被认为是通过机器学习、计算基础设施的统计可视化、检测安全违规并采取相应措施来为云数据架构制定保护计划的一种革命。这些基于人工智能的系统通过审查更多的网络活动使工作变得更加容易，并防止了可能导致更严重违规的任何异常行为。本文探讨了人工智能如何通过预测分析、基于行为的安全威胁检测以及人工智能驱动的加密来增强云安全。它还概述了之前安全模型的问题以及人工智能如何克服这些问题。由于类似原因，数据隐私、人工智能模型中的偏差和监管合规性等问题也得到了讨论。因此，人工智能提高了云计算环境的保护水平；然而，在后续阶段仍需更多努力来扩展该技术的可靠性、模块性和伦理方面。这意味着人工智能可以与其他新兴计算技术，包括区块链，相结合，以进一步改善安全框架。本文讨论了使用人工智能保护云数据架构的当前趋势，并提出了进一步的研究和应用方向。 

---
# A Graphical Global Optimization Framework for Parameter Estimation of Statistical Models with Nonconvex Regularization Functions 

**Title (ZH)**: 非凸正则化函数下统计模型参数估计的图形全局优化框架 

**Authors**: Danial Davarnia, Mohammadreza Kiaghadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03899)  

**Abstract**: Optimization problems with norm-bounding constraints arise in a variety of applications, including portfolio optimization, machine learning, and feature selection. A common approach to these problems involves relaxing the norm constraint via Lagrangian relaxation, transforming it into a regularization term in the objective function. A particularly challenging class includes the zero-norm function, which promotes sparsity in statistical parameter estimation. Most existing exact methods for solving these problems introduce binary variables and artificial bounds to reformulate them as higher-dimensional mixed-integer programs, solvable by standard solvers. Other exact approaches exploit specific structural properties of the objective, making them difficult to generalize across different problem types. Alternative methods employ nonconvex penalties with favorable statistical characteristics, but these are typically addressed using heuristic or local optimization techniques due to their structural complexity. In this paper, we propose a novel graph-based method to globally solve optimization problems involving generalized norm-bounding constraints. Our approach encompasses standard $\ell_p$-norms for $p \in [0, \infty)$ and nonconvex penalties such as SCAD and MCP. We leverage decision diagrams to construct strong convex relaxations directly in the original variable space, eliminating the need for auxiliary variables or artificial bounds. Integrated into a spatial branch-and-cut framework, our method guarantees convergence to the global optimum. We demonstrate its effectiveness through preliminary computational experiments on benchmark sparse linear regression problems involving complex nonconvex penalties, which are not tractable using existing global optimization techniques. 

**Abstract (ZH)**: 基于图的方法求解涉及广义范数约束的优化问题 

---
# Novel Extraction of Discriminative Fine-Grained Feature to Improve Retinal Vessel Segmentation 

**Title (ZH)**: 新型提取具有判别性的细粒度特征以提高视网膜血管分割 

**Authors**: Shuang Zeng, Chee Hong Lee, Micky C Nnamdi, Wenqi Shi, J Ben Tamo, Lei Zhu, Hangzhou He, Xinliang Zhang, Qian Chen, May D. Wang, Yanye Lu, Qiushi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.03896)  

**Abstract**: Retinal vessel segmentation is a vital early detection method for several severe ocular diseases. Despite significant progress in retinal vessel segmentation with the advancement of Neural Networks, there are still challenges to overcome. Specifically, retinal vessel segmentation aims to predict the class label for every pixel within a fundus image, with a primary focus on intra-image discrimination, making it vital for models to extract more discriminative features. Nevertheless, existing methods primarily focus on minimizing the difference between the output from the decoder and the label, but ignore fully using feature-level fine-grained representations from the encoder. To address these issues, we propose a novel Attention U-shaped Kolmogorov-Arnold Network named AttUKAN along with a novel Label-guided Pixel-wise Contrastive Loss for retinal vessel segmentation. Specifically, we implement Attention Gates into Kolmogorov-Arnold Networks to enhance model sensitivity by suppressing irrelevant feature activations and model interpretability by non-linear modeling of KAN blocks. Additionally, we also design a novel Label-guided Pixel-wise Contrastive Loss to supervise our proposed AttUKAN to extract more discriminative features by distinguishing between foreground vessel-pixel pairs and background pairs. Experiments are conducted across four public datasets including DRIVE, STARE, CHASE_DB1, HRF and our private dataset. AttUKAN achieves F1 scores of 82.50%, 81.14%, 81.34%, 80.21% and 80.09%, along with MIoU scores of 70.24%, 68.64%, 68.59%, 67.21% and 66.94% in the above datasets, which are the highest compared to 11 networks for retinal vessel segmentation. Quantitative and qualitative results show that our AttUKAN achieves state-of-the-art performance and outperforms existing retinal vessel segmentation methods. Our code will be available at this https URL. 

**Abstract (ZH)**: 视网膜血管分割是早期检测多种严重眼病的重要方法。尽管随着神经网络的发展，视网膜血管分割取得了显著进步，但仍存在挑战。具体而言，视网膜血管分割旨在预测fundus图像中每个像素的类别标签，主要关注图像内的区分，因此模型需要提取更具区分性的特征。然而，现有方法主要集中在最小化解码器输出与标签之间的差异，而忽视了充分使用编码器的特征级细粒度表示。为应对这些问题，我们提出了一种新颖的注意力U型柯尔莫哥洛夫-阿诺尔德网络（AttUKAN）及其新颖的标签引导像素级对比损失，以应用于视网膜血管分割。实验在包括DRIVE、STARE、CHASE_DB1、HRF和我们私人数据集在内的四个公共数据集上进行。AttUKAN在上述数据集上的F1分数分别为82.50%、81.14%、81.34%、80.21%和80.09%，MIoU分数分别为70.24%、68.64%、68.59%、67.21%和66.94%，其性能优于11种视网膜血管分割网络。定量和定性结果表明，我们的AttUKAN达到最先进的性能，并优于现有视网膜血管分割方法。我们的代码将在此处提供。 

---
# Scratch Copilot: Supporting Youth Creative Coding with AI 

**Title (ZH)**: Scratch Copilot: 以AI支持青少年创意编程 

**Authors**: Stefania Druga, Amy J. Ko  

**Link**: [PDF](https://arxiv.org/pdf/2505.03867)  

**Abstract**: Creative coding platforms like Scratch have democratized programming for children, yet translating imaginative ideas into functional code remains a significant hurdle for many young learners. While AI copilots assist adult programmers, few tools target children in block-based environments. Building on prior research \cite{druga_how_2021,druga2023ai, druga2023scratch}, we present Cognimates Scratch Copilot: an AI-powered assistant integrated into a Scratch-like environment, providing real-time support for ideation, code generation, debugging, and asset creation. This paper details the system architecture and findings from an exploratory qualitative evaluation with 18 international children (ages 7--12). Our analysis reveals how the AI Copilot supported key creative coding processes, particularly aiding ideation and debugging. Crucially, it also highlights how children actively negotiated the use of AI, demonstrating strong agency by adapting or rejecting suggestions to maintain creative control. Interactions surfaced design tensions between providing helpful scaffolding and fostering independent problem-solving, as well as learning opportunities arising from navigating AI limitations and errors. Findings indicate Cognimates Scratch Copilot's potential to enhance creative self-efficacy and engagement. Based on these insights, we propose initial design guidelines for AI coding assistants that prioritize youth agency and critical interaction alongside supportive scaffolding. 

**Abstract (ZH)**: 创意编程平台如Scratch虽然已使编程 democratized 于儿童，但将富有想象力的想法转化为可运行的代码仍然是许多年轻学习者的一大障碍。虽然 AI 共同飞行员辅助成人编程，但很少有工具针对基于积木的环境中的儿童。基于先前的研究 \cite{druga_how_2021,druga2023ai, druga2023scratch}，我们介绍了 Cognimates Scratch 共同飞行员：一种集成于类似 Scratch 的环境中的人工智能辅助工具，提供实时支持，包括创意思维、代码生成、调试和资产创建。本文详细介绍了该系统架构及其对 18 名国际儿童（年龄 7-12 岁）进行探索性定性评估的发现。我们的分析揭示了人工智能共同飞行员如何支持关键的创意编程过程，特别是帮助创新点的开发和调试。重要的是，它还突显了儿童如何积极协商人工智能的使用，表现出强大的自主权，通过适应或拒绝建议来保持创意控制。互动体现出在提供有帮助的教学支架和培养独立问题解决能力之间的设计张力，同时也揭示了导航人工智能限制和错误所带来的学习机会。研究结果表明，Cognimates Scratch 共同飞行员有可能增强创意自我效能感和参与度。基于这些见解，我们提出了一些建议的设计指南，强调青少年的自主权和批判性交互并重，以提供支持性的教学支架。 

---
# From Glue-Code to Protocols: A Critical Analysis of A2A and MCP Integration for Scalable Agent Systems 

**Title (ZH)**: 从粘合代码到协议：面向可扩展代理系统的A2A和MCP集成关键分析 

**Authors**: Qiaomu Li, Ying Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.03864)  

**Abstract**: Artificial intelligence is rapidly evolving towards multi-agent systems where numerous AI agents collaborate and interact with external tools. Two key open standards, Google's Agent to Agent (A2A) protocol for inter-agent communication and Anthropic's Model Context Protocol (MCP) for standardized tool access, promise to overcome the limitations of fragmented, custom integration approaches. While their potential synergy is significant, this paper argues that effectively integrating A2A and MCP presents unique, emergent challenges at their intersection, particularly concerning semantic interoperability between agent tasks and tool capabilities, the compounded security risks arising from combined discovery and execution, and the practical governance required for the envisioned "Agent Economy". This work provides a critical analysis, moving beyond a survey to evaluate the practical implications and inherent difficulties of combining these horizontal and vertical integration standards. We examine the benefits (e.g., specialization, scalability) while critically assessing their dependencies and trade-offs in an integrated context. We identify key challenges increased by the integration, including novel security vulnerabilities, privacy complexities, debugging difficulties across protocols, and the need for robust semantic negotiation mechanisms. In summary, A2A+MCP offers a vital architectural foundation, but fully realizing its potential requires substantial advancements to manage the complexities of their combined operation. 

**Abstract (ZH)**: 人工智能正快速向多-agent系统发展，其中众多AI代理协作并与外部工具交互。谷歌的代理到代理（A2A）协议和Anthropic的模型上下文协议（MCP）作为两个关键的开放标准，有望克服模块化、定制集成方法的局限性。虽然它们的协同潜力巨大，但本文认为，在A2A和MCP的交汇处有效集成这两项标准带来了独特且新兴的挑战，特别是代理任务和工具能力之间的语义互操作性问题、综合发现和执行带来的复合安全风险以及为设想中的“代理经济”所需的实践治理。本文提供了关键分析，超越了单纯的调研，评估组合这些横向和纵向集成标准的实践影响和固有困难。我们探讨了其益处（如专业化、可扩展性），同时批判性地评估这些标准在集成环境下的依赖性和权衡。我们指出了集成带来的关键挑战，包括新型安全漏洞、隐私复杂性、跨协议调试困难以及对稳健的语义谈判机制的需要。总之，A2A+MCP提供了重要的架构基础，但要充分发挥其潜力，还需要在管理其联合运营的复杂性方面取得显著进展。 

---
# Data-Driven Falsification of Cyber-Physical Systems 

**Title (ZH)**: 数据驱动的网络物理系统反驳方法 

**Authors**: Atanu Kundu, Sauvik Gon, Rajarshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2505.03863)  

**Abstract**: Cyber-Physical Systems (CPS) are abundant in safety-critical domains such as healthcare, avionics, and autonomous vehicles. Formal verification of their operational safety is, therefore, of utmost importance. In this paper, we address the falsification problem, where the focus is on searching for an unsafe execution in the system instead of proving their absence. The contribution of this paper is a framework that (a) connects the falsification of CPS with the falsification of deep neural networks (DNNs) and (b) leverages the inherent interpretability of Decision Trees for faster falsification of CPS. This is achieved by: (1) building a surrogate model of the CPS under test, either as a DNN model or a Decision Tree, (2) application of various DNN falsification tools to falsify CPS, and (3) a novel falsification algorithm guided by the explanations of safety violations of the CPS model extracted from its Decision Tree surrogate. The proposed framework has the potential to exploit a repertoire of \emph{adversarial attack} algorithms designed to falsify robustness properties of DNNs, as well as state-of-the-art falsification algorithms for DNNs. Although the presented methodology is applicable to systems that can be executed/simulated in general, we demonstrate its effectiveness, particularly in CPS. We show that our framework, implemented as a tool \textsc{FlexiFal}, can detect hard-to-find counterexamples in CPS that have linear and non-linear dynamics. Decision tree-guided falsification shows promising results in efficiently finding multiple counterexamples in the ARCH-COMP 2024 falsification benchmarks~\cite{khandait2024arch}. 

**Abstract (ZH)**: 基于物理系统的网络安全物理系统(CPS)在医疗、航空和自动驾驶等安全关键领域中广泛应用。确保其操作安全性因此变得尤为重要。本文针对反证问题进行探讨，重点在于寻找系统的不安全执行，而非证明其不存在。本文的贡献在于提出了一种框架，该框架将CPS的反证与其深度神经网络(DNNs)的反证关联起来，并利用决策树的内在可解释性加速CPS的反证过程。这一目标通过以下方式实现：(1) 构建待测CPS的代理模型，该模型可以是DNN模型或决策树；(2) 应用多种DNN反证工具对CPS进行反证；(3) 通过从CPS模型的决策树代理中提取的安全违规解释指导的一种新型反证算法。所提出的框架有望利用专门设计用于反证DNNs的鲁棒性性质的对抗攻击算法，以及最先进的DNN反证算法。尽管所展示的方法适用于一般可执行/模拟的系统，但我们特别展示了其在CPS中的有效性。我们证明，作为工具\textsc{FlexiFal}实现的该框架能够检测具有线性和非线性动力学的CPS中的难找反例。基于决策树的反证在ARCH-COMP 2024反证基准测试中显示出高效发现多个反例的前景。 

---
# Deepfakes on Demand: the rise of accessible non-consensual deepfake image generators 

**Title (ZH)**: 按需生成虚假影像：可访问的非同意Deepfake图像生成器 

**Authors**: Will Hawkins, Chris Russell, Brent Mittelstadt  

**Link**: [PDF](https://arxiv.org/pdf/2505.03859)  

**Abstract**: Advances in multimodal machine learning have made text-to-image (T2I) models increasingly accessible and popular. However, T2I models introduce risks such as the generation of non-consensual depictions of identifiable individuals, otherwise known as deepfakes. This paper presents an empirical study exploring the accessibility of deepfake model variants online. Through a metadata analysis of thousands of publicly downloadable model variants on two popular repositories, Hugging Face and Civitai, we demonstrate a huge rise in easily accessible deepfake models. Almost 35,000 examples of publicly downloadable deepfake model variants are identified, primarily hosted on Civitai. These deepfake models have been downloaded almost 15 million times since November 2022, with the models targeting a range of individuals from global celebrities to Instagram users with under 10,000 followers. Both Stable Diffusion and Flux models are used for the creation of deepfake models, with 96% of these targeting women and many signalling intent to generate non-consensual intimate imagery (NCII). Deepfake model variants are often created via the parameter-efficient fine-tuning technique known as low rank adaptation (LoRA), requiring as few as 20 images, 24GB VRAM, and 15 minutes of time, making this process widely accessible via consumer-grade computers. Despite these models violating the Terms of Service of hosting platforms, and regulation seeking to prevent dissemination, these results emphasise the pressing need for greater action to be taken against the creation of deepfakes and NCII. 

**Abstract (ZH)**: 多模态机器学习的进步使得文本到图像（T2I）模型越来越普及和易于获取，但同时也带来了如非同意生成可识别个体的深度假图像等风险。本文通过分析两个流行仓库Hugging Face和Civitai上数千个可公开下载的深度假图像模型变体的元数据，探索了深度假图像模型变体的可获取性。研究结果显示，大量易于获取的深度假图像模型变体在网络上广泛存在。超过35,000个可公开下载的深度假图像模型变体主要托管在Civitai上。自2022年11月以来，这些模型几乎被下载了1500多万次，目标个体范围从全球名人到拥有不到10,000名粉丝的Instagram用户。用于创建深度假图像模型的Stable Diffusion和Flux模型中，有96%的目标为女性，且许多模型表明意图生成非同意的亲密图像。深度假图像模型变体通常通过低秩适应（LoRA）参数高效微调技术创建，只需20张图片、24GB VRAM和15分钟的时间，使其通过消费级计算机即可广泛获取。尽管这些模型违反了托管平台的服务条款，并存在监管措施以防止传播，但研究结果强调了采取更大力度行动以应对深度假图像和非同意的亲密图像的迫切需求。 

---
# An Active Inference Model of Covert and Overt Visual Attention 

**Title (ZH)**: 隐蔽与外显视觉注意的主动推断模型 

**Authors**: Tin Mišić, Karlo Koledić, Fabio Bonsignorio, Ivan Petrović, Ivan Marković  

**Link**: [PDF](https://arxiv.org/pdf/2505.03856)  

**Abstract**: The ability to selectively attend to relevant stimuli while filtering out distractions is essential for agents that process complex, high-dimensional sensory input. This paper introduces a model of covert and overt visual attention through the framework of active inference, utilizing dynamic optimization of sensory precisions to minimize free-energy. The model determines visual sensory precisions based on both current environmental beliefs and sensory input, influencing attentional allocation in both covert and overt modalities. To test the effectiveness of the model, we analyze its behavior in the Posner cueing task and a simple target focus task using two-dimensional(2D) visual data. Reaction times are measured to investigate the interplay between exogenous and endogenous attention, as well as valid and invalid cueing. The results show that exogenous and valid cues generally lead to faster reaction times compared to endogenous and invalid cues. Furthermore, the model exhibits behavior similar to inhibition of return, where previously attended locations become suppressed after a specific cue-target onset asynchrony interval. Lastly, we investigate different aspects of overt attention and show that involuntary, reflexive saccades occur faster than intentional ones, but at the expense of adaptability. 

**Abstract (ZH)**: 处理复杂高维度感官输入的代理需要具备选择性关注相关刺激并过滤干扰的能力。本文通过主动推断框架引入了一种视觉注意模型，利用传感器精度的动态优化以最小化自由能。该模型根据当前环境信念和感官输入来确定视觉感官精度，影响知觉和外显模态中的注意分配。为了测试该模型的有效性，我们使用二维(2D)视觉数据分析了其在Posner指导任务和简单目标聚焦任务中的行为，并测量反应时间来探究外源性与内源性注意以及有效和无效引导之间的相互作用。结果显示，外源性与有效引导通常会导致更快的反应时间，而内源性与无效引导则不然。此外，模型表现出类似抑制回返的行为，即在特定的刺激-目标出现间隔后，先前注意的区域会受到抑制。最后，我们探讨了外显注意的不同方面，并发现不自主的反射性眨眼比有意的眨眼更快发生，但牺牲了适应性。 

---
# GRAPE: Heterogeneous Graph Representation Learning for Genetic Perturbation with Coding and Non-Coding Biotype 

**Title (ZH)**: GRAPE: 异构图表示学习在编码和非编码生物类型遗传 Perturbation 中的应用 

**Authors**: Changxi Chi, Jun Xia, Jingbo Zhou, Jiabei Cheng, Chang Yu, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03853)  

**Abstract**: Predicting genetic perturbations enables the identification of potentially crucial genes prior to wet-lab experiments, significantly improving overall experimental efficiency. Since genes are the foundation of cellular life, building gene regulatory networks (GRN) is essential to understand and predict the effects of genetic perturbations. However, current methods fail to fully leverage gene-related information, and solely rely on simple evaluation metrics to construct coarse-grained GRN. More importantly, they ignore functional differences between biotypes, limiting the ability to capture potential gene interactions. In this work, we leverage pre-trained large language model and DNA sequence model to extract features from gene descriptions and DNA sequence data, respectively, which serve as the initialization for gene representations. Additionally, we introduce gene biotype information for the first time in genetic perturbation, simulating the distinct roles of genes with different biotypes in regulating cellular processes, while capturing implicit gene relationships through graph structure learning (GSL). We propose GRAPE, a heterogeneous graph neural network (HGNN) that leverages gene representations initialized with features from descriptions and sequences, models the distinct roles of genes with different biotypes, and dynamically refines the GRN through GSL. The results on publicly available datasets show that our method achieves state-of-the-art performance. 

**Abstract (ZH)**: 基于大型语言模型和DNA序列模型的基因表示学习与基因调控网络构建以预测基因扰动并捕获潜在基因交互作用 

---
# Impact Analysis of Inference Time Attack of Perception Sensors on Autonomous Vehicles 

**Title (ZH)**: 感知传感器推理时间攻击的影响分析对自动驾驶车辆的影响分析 

**Authors**: Hanlin Chen, Simin Chen, Wenyu Li, Wei Yang, Yiheng Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03850)  

**Abstract**: As a safety-critical cyber-physical system, cybersecurity and related safety issues for Autonomous Vehicles (AVs) have been important research topics for a while. Among all the modules on AVs, perception is one of the most accessible attack surfaces, as drivers and AVs have no control over the outside environment. Most current work targeting perception security for AVs focuses on perception correctness. In this work, we propose an impact analysis based on inference time attacks for autonomous vehicles. We demonstrate in a simulation system that such inference time attacks can also threaten the safety of both the ego vehicle and other traffic participants. 

**Abstract (ZH)**: 作为安全关键的计算物理系统，自动驾驶车辆（AVs）的网络安全及相关安全问题一直是重要的研究课题。在所有AV模块中，感知是最易遭受攻击的模块之一，因为驾驶者和AV无法控制外部环境。目前大多数针对AV感知安全的研究集中在感知准确性上。在本工作中，我们提出了基于推理时间攻击的影响分析方法。我们通过仿真系统展示，这样的推理时间攻击也会影响AV自身和其它交通参与者的安全。 

---
# Advanced Clustering Framework for Semiconductor Image Analytics Integrating Deep TDA with Self-Supervised and Transfer Learning Techniques 

**Title (ZH)**: 基于深度拓扑数据分析的自监督与迁移学习集成的半导体图像分析高级聚类框架 

**Authors**: Janhavi Giri, Attila Lengyel, Don Kent, Edward Kibardin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03848)  

**Abstract**: Semiconductor manufacturing generates vast amounts of image data, crucial for defect identification and yield optimization, yet often exceeds manual inspection capabilities. Traditional clustering techniques struggle with high-dimensional, unlabeled data, limiting their effectiveness in capturing nuanced patterns. This paper introduces an advanced clustering framework that integrates deep Topological Data Analysis (TDA) with self-supervised and transfer learning techniques, offering a novel approach to unsupervised image clustering. TDA captures intrinsic topological features, while self-supervised learning extracts meaningful representations from unlabeled data, reducing reliance on labeled datasets. Transfer learning enhances the framework's adaptability and scalability, allowing fine-tuning to new datasets without retraining from scratch. Validated on synthetic and open-source semiconductor image datasets, the framework successfully identifies clusters aligned with defect patterns and process variations. This study highlights the transformative potential of combining TDA, self-supervised learning, and transfer learning, providing a scalable solution for proactive process monitoring and quality control in semiconductor manufacturing and other domains with large-scale image datasets. 

**Abstract (ZH)**: 一种将深度拓扑数据分析、自监督学习和迁移学习相结合的高级聚类框架：在半导体制造中的应用 

---
# GAME: Learning Multimodal Interactions via Graph Structures for Personality Trait Estimation 

**Title (ZH)**: GAME: 通过图结构学习多模态交互以估计人格特质 

**Authors**: Kangsheng Wang, Yuhang Li, Chengwei Ye, Yufei Lin, Huanzhen Zhang, Bohan Hu, Linuo Xu, Shuyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03846)  

**Abstract**: Apparent personality analysis from short videos poses significant chal-lenges due to the complex interplay of visual, auditory, and textual cues. In this paper, we propose GAME, a Graph-Augmented Multimodal Encoder designed to robustly model and fuse multi-source features for automatic personality prediction. For the visual stream, we construct a facial graph and introduce a dual-branch Geo Two-Stream Network, which combines Graph Convolutional Networks (GCNs) and Convolutional Neural Net-works (CNNs) with attention mechanisms to capture both structural and appearance-based facial cues. Complementing this, global context and iden-tity features are extracted using pretrained ResNet18 and VGGFace back-bones. To capture temporal dynamics, frame-level features are processed by a BiGRU enhanced with temporal attention modules. Meanwhile, audio representations are derived from the VGGish network, and linguistic se-mantics are captured via the XLM-Roberta transformer. To achieve effective multimodal integration, we propose a Channel Attention-based Fusion module, followed by a Multi-Layer Perceptron (MLP) regression head for predicting personality traits. Extensive experiments show that GAME con-sistently outperforms existing methods across multiple benchmarks, vali-dating its effectiveness and generalizability. 

**Abstract (ZH)**: 从短视频中分析表观人格存在显著挑战，由于其中视觉、听觉和文本线索的复杂交互。本文提出GAME，一种图增强多模态编码器，旨在稳健地建模和融合多源特征以实现自动人格预测。对于视觉流，构建面部图并引入双支Geo Two-Stream网络，结合图卷积网络（GCNs）和卷积神经网络（CNNs）以及注意机制来捕获结构和基于外观的面部特征。为了捕捉时间动态，帧级特征通过增强时序注意模块的双向GRU进行处理。同时，音频表示由VGGish网络提取，语言语义通过XLM-Roberta变换器捕获。为了实现有效的多模态融合，提出基于通道注意的融合模块，随后是用于预测人格特质的多层感知机（MLP）回归头部。广泛实验表明，GAME在多个基准测试中一致优于现有方法，验证了其有效性和泛化能力。 

---
# A Deep Learning approach for Depressive Symptoms assessment in Parkinson's disease patients using facial videos 

**Title (ZH)**: 使用面部视频评估帕金森病患者抑郁症状的深度学习方法 

**Authors**: Ioannis Kyprakis, Vasileios Skaramagkas, Iro Boura, Georgios Karamanis, Dimitrios I. Fotiadis, Zinovia Kefalopoulou, Cleanthe Spanaki, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.03845)  

**Abstract**: Parkinson's disease (PD) is a neurodegenerative disorder, manifesting with motor and non-motor symptoms. Depressive symptoms are prevalent in PD, affecting up to 45% of patients. They are often underdiagnosed due to overlapping motor features, such as hypomimia. This study explores deep learning (DL) models-ViViT, Video Swin Tiny, and 3D CNN-LSTM with attention layers-to assess the presence and severity of depressive symptoms, as detected by the Geriatric Depression Scale (GDS), in PD patients through facial video analysis. The same parameters were assessed in a secondary analysis taking into account whether patients were one hour after (ON-medication state) or 12 hours without (OFF-medication state) dopaminergic medication. Using a dataset of 1,875 videos from 178 patients, the Video Swin Tiny model achieved the highest performance, with up to 94% accuracy and 93.7% F1-score in binary classification (presence of absence of depressive symptoms), and 87.1% accuracy with an 85.4% F1-score in multiclass tasks (absence or mild or severe depressive symptoms). 

**Abstract (ZH)**: 帕金森病患者抑郁症状的深学习模型评估：基于面部视频分析的Geriatric Depression Scale评分 

---
# From Spaceborn to Airborn: SAR Image Synthesis Using Foundation Models for Multi-Scale Adaptation 

**Title (ZH)**: 从星载到空载：用于多尺度适应的foundation模型SAR图像合成 

**Authors**: Solène Debuysère, Nicolas Trouvé, Nathan Letheule, Olivier Lévêque, Elise Colin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03844)  

**Abstract**: The availability of Synthetic Aperture Radar (SAR) satellite imagery has increased considerably in recent years, with datasets commercially available. However, the acquisition of high-resolution SAR images in airborne configurations, remains costly and limited. Thus, the lack of open source, well-labeled, or easily exploitable SAR text-image datasets is a barrier to the use of existing foundation models in remote sensing applications. In this context, synthetic image generation is a promising solution to augment this scarce data, enabling a broader range of applications. Leveraging over 15 years of ONERA's extensive archival airborn data from acquisition campaigns, we created a comprehensive training dataset of 110 thousands SAR images to exploit a 3.5 billion parameters pre-trained latent diffusion model. In this work, we present a novel approach utilizing spatial conditioning techniques within a foundation model to transform satellite SAR imagery into airborne SAR representations. Additionally, we demonstrate that our pipeline is effective for bridging the realism of simulated images generated by ONERA's physics-based simulator EMPRISE. Our method explores a key application of AI in advancing SAR imaging technology. To the best of our knowledge, we are the first to introduce this approach in the literature. 

**Abstract (ZH)**: 合成孔径雷达(SAR)卫星图像的可用性近年来显著增加，商用数据集已广泛可用。然而，航空配置下的高分辨率SAR图像获取依然成本高昂且有限。因此，开源、标注良好或易于利用的SAR图文数据集的缺乏是限制现有基础模型在遥感应用中使用的一个障碍。在此背景下，合成图像生成是一种有前景的解决方案，以补充这种稀缺的数据，从而实现更广泛的应用。利用ONERA近15年丰富的航空数据档案，我们创建了一个包含11万张SAR图像的综合训练数据集，以利用一个预训练的35亿参数潜在扩散模型。在这项工作中，我们提出了一种利用空间条件化技术的基础模型新方法，将卫星SAR图像转化为航空SAR表示。此外，我们还展示了我们的管道在弥合ONERA物理仿真器EMPRISE生成的模拟图像的现实感方面具有有效性。我们的方法探索了AI在推进SAR成像技术中的关键应用。据我们所知，这是首次在文献中引入这种方法。 

---
# CoCoB: Adaptive Collaborative Combinatorial Bandits for Online Recommendation 

**Title (ZH)**: CoCoB: 自适应协作组合臂赛选算法的在线推荐 

**Authors**: Cairong Yan, Jinyi Han, Jin Ju, Yanting Zhang, Zijian Wang, Xuan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03840)  

**Abstract**: Clustering bandits have gained significant attention in recommender systems by leveraging collaborative information from neighboring users to better capture target user preferences. However, these methods often lack a clear definition of similar users and face challenges when users with unique preferences lack appropriate neighbors. In such cases, relying on divergent preferences of misidentified neighbors can degrade recommendation quality. To address these limitations, this paper proposes an adaptive Collaborative Combinatorial Bandits algorithm (CoCoB). CoCoB employs an innovative two-sided bandit architecture, applying bandit principles to both the user and item sides. The user-bandit employs an enhanced Bayesian model to explore user similarity, identifying neighbors based on a similarity probability threshold. The item-bandit treats items as arms, generating diverse recommendations informed by the user-bandit's output. CoCoB dynamically adapts, leveraging neighbor preferences when available or focusing solely on the target user otherwise. Regret analysis under a linear contextual bandit setting and experiments on three real-world datasets demonstrate CoCoB's effectiveness, achieving an average 2.4% improvement in F1 score over state-of-the-art methods. 

**Abstract (ZH)**: 基于聚类的组合臂协同过滤算法在推荐系统中的应用：CoCoB算法的研究 

---
# IntelliCardiac: An Intelligent Platform for Cardiac Image Segmentation and Classification 

**Title (ZH)**: IntelliCardiac：一种心脏图像分割与分类的智能平台 

**Authors**: Ting Yu Tsai, An Yu, Meghana Spurthi Maadugundu, Ishrat Jahan Mohima, Umme Habiba Barsha, Mei-Hwa F. Chen, Balakrishnan Prabhakaran, Ming-Ching Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03838)  

**Abstract**: Precise and effective processing of cardiac imaging data is critical for the identification and management of the cardiovascular diseases. We introduce IntelliCardiac, a comprehensive, web-based medical image processing platform for the automatic segmentation of 4D cardiac images and disease classification, utilizing an AI model trained on the publicly accessible ACDC dataset. The system, intended for patients, cardiologists, and healthcare professionals, offers an intuitive interface and uses deep learning models to identify essential heart structures and categorize cardiac diseases. The system supports analysis of both the right and left ventricles as well as myocardium, and then classifies patient's cardiac images into five diagnostic categories: dilated cardiomyopathy, myocardial infarction, hypertrophic cardiomyopathy, right ventricular abnormality, and no disease. IntelliCardiac combines a deep learning-based segmentation model with a two-step classification pipeline. The segmentation module gains an overall accuracy of 92.6\%. The classification module, trained on characteristics taken from segmented heart structures, achieves 98\% accuracy in five categories. These results exceed the performance of the existing state-of-the-art methods that integrate both segmentation and classification models. IntelliCardiac, which supports real-time visualization, workflow integration, and AI-assisted diagnostics, has great potential as a scalable, accurate tool for clinical decision assistance in cardiac imaging and diagnosis. 

**Abstract (ZH)**: 基于AI的心脏影像精准自动分割与疾病分类综合平台IntelliCardiac 

---
# Explainable Face Recognition via Improved Localization 

**Title (ZH)**: 基于改进定位的可解释人脸识别 

**Authors**: Rashik Shadman, Daqing Hou, Faraz Hussain, M G Sarwar Murshed  

**Link**: [PDF](https://arxiv.org/pdf/2505.03837)  

**Abstract**: Biometric authentication has become one of the most widely used tools in the current technological era to authenticate users and to distinguish between genuine users and imposters. Face is the most common form of biometric modality that has proven effective. Deep learning-based face recognition systems are now commonly used across different domains. However, these systems usually operate like black-box models that do not provide necessary explanations or justifications for their decisions. This is a major disadvantage because users cannot trust such artificial intelligence-based biometric systems and may not feel comfortable using them when clear explanations or justifications are not provided. This paper addresses this problem by applying an efficient method for explainable face recognition systems. We use a Class Activation Mapping (CAM)-based discriminative localization (very narrow/specific localization) technique called Scaled Directed Divergence (SDD) to visually explain the results of deep learning-based face recognition systems. We perform fine localization of the face features relevant to the deep learning model for its prediction/decision. Our experiments show that the SDD Class Activation Map (CAM) highlights the relevant face features very specifically compared to the traditional CAM and very accurately. The provided visual explanations with narrow localization of relevant features can ensure much-needed transparency and trust for deep learning-based face recognition systems. 

**Abstract (ZH)**: 基于生物特征的身份验证已成为当前技术时代中最广泛使用的工具之一，用于用户认证和区分真实用户与冒充者。面部是最常见的生物特征模态，证明了其有效性。基于深度学习的面部识别系统现在在不同领域中普遍使用。然而，这些系统通常像黑箱模型一样运作，不提供其决策必要的解释或合理性。这是一大劣势，因为用户无法信任基于人工智能的生物特征系统，并且在没有提供清晰解释或合理性的情况下可能不会感到舒适使用它们。本文通过应用一种高效的方法解决了这一问题，用于可解释的面部识别系统。我们使用基于Class Activation Mapping (CAM)的具有区分性定位（非常狭窄/具体的定位）技术——Scaled Directed Divergence (SDD)——来可视化地解释基于深度学习的面部识别系统的结果。我们对与深度学习模型预测/决策相关的面部特征进行了精细定位。我们的实验证明，与传统的CAM相比，SDD Class Activation Map (CAM)能够非常具体地突出显示相关的面部特征，并且非常准确。提供的视觉解释，结合对相关特征的狭窄定位，可以确保基于深度学习的面部识别系统所需的透明度和信任。 

---
# OBD-Finder: Explainable Coarse-to-Fine Text-Centric Oracle Bone Duplicates Discovery 

**Title (ZH)**: OBD-Finder: 可解释的从粗到细基于文字的甲骨文重复发现 

**Authors**: Chongsheng Zhang, Shuwen Wu, Yingqi Chen, Matthias Aßenmacher, Christian Heumann, Yi Men, Gaojuan Fan, João Gama  

**Link**: [PDF](https://arxiv.org/pdf/2505.03836)  

**Abstract**: Oracle Bone Inscription (OBI) is the earliest systematic writing system in China, while the identification of Oracle Bone (OB) duplicates is a fundamental issue in OBI research. In this work, we design a progressive OB duplicate discovery framework that combines unsupervised low-level keypoints matching with high-level text-centric content-based matching to refine and rank the candidate OB duplicates with semantic awareness and interpretability. We compare our approach with state-of-the-art content-based image retrieval and image matching methods, showing that our approach yields comparable recall performance and the highest simplified mean reciprocal rank scores for both Top-5 and Top-15 retrieval results, and with significantly accelerated computation efficiency. We have discovered over 60 pairs of new OB duplicates in real-world deployment, which were missed by OBI researchers for decades. The models, video illustration and demonstration of this work are available at: this https URL. 

**Abstract (ZH)**: 殷墟甲骨文（OBI）是最早的 systematic 记写系统，而甲骨文（OB）复制件的识别是 OBI 研究中的基本问题。本文设计了一个渐进的 OB 复制品发现框架，结合了无监督的低层级关键点匹配和高层级文本中心的内容匹配，以语义意识和可解释性来细化和排名候选 OB 复制品。我们将我们的方法与最先进的基于内容的图像检索和图像匹配方法进行比较，结果显示，我们的方法在 Top-5 和 Top-15 检索结果中达到了可比拟的召回率性能，并且具有显著加速的计算效率。我们已经在实际部署中发现了超过 60 组新的 OB 复制品，这些复制品被 OBI 研究者漏过了数十年之久。本文的模型、视频演示和示例可访问此链接: this https URL。 

---
# The Shift Towards Preprints in AI Policy Research: A Comparative Study of Preprint Trends in the U.S., Europe, and South Korea 

**Title (ZH)**: AI政策研究中预印本的转变：美国、欧洲和韩国预印本趋势的比较研究 

**Authors**: Simon Suh, Jihyuk Bang, Ji Woo Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.03835)  

**Abstract**: The adoption of open science has quickly changed how artificial intelligence (AI) policy research is distributed globally. This study examines the regional trends in the citation of preprints, specifically focusing on the impact of two major disruptive events: the COVID-19 pandemic and the release of ChatGPT, on research dissemination patterns in the United States, Europe, and South Korea from 2015 to 2024. Using bibliometrics data from the Web of Science, this study tracks how global disruptive events influenced the adoption of preprints in AI policy research and how such shifts vary by region. By marking the timing of these disruptive events, the analysis reveals that while all regions experienced growth in preprint citations, the magnitude and trajectory of change varied significantly. The United States exhibited sharp, event-driven increases; Europe demonstrated institutional growth; and South Korea maintained consistent, linear growth in preprint adoption. These findings suggest that global disruptions may have accelerated preprint adoption, but the extent and trajectory are shaped by local research cultures, policy environments, and levels of open science maturity. This paper emphasizes the need for future AI governance strategies to consider regional variability in research dissemination and highlights opportunities for further longitudinal and comparative research to deepen our understanding of open-access adoption in AI policy development. 

**Abstract (ZH)**: 开放科学的采纳迅速改变了人工智能政策研究的全球分布方式。本文研究了预印本引用的区域趋势，特别是 COVID-19 疫情和 ChatGPT 的发布这两大破坏性事件对 2015 年至 2024 年期间美国、欧洲和韩国人工智能政策研究传播模式的影响。利用 Web of Science 的引文数据，本文追踪全球破坏性事件如何影响人工智能政策研究中预印本的采纳，并分析这些变化在不同地区的差异性。通过标注这些破坏性事件的时间点，分析揭示，尽管各地区都经历了预印本引用数量的增长，但增长的幅度和轨迹存在显著差异。美国表现出事件驱动的大幅增长；欧洲则表现出机构增长；而韩国则保持着平稳、线性的预印本采纳增长。这些发现表明，全球性中断可能加速了预印本的采纳，但其程度和轨迹由当地的研究文化、政策环境和开放科学成熟度所塑造。本文强调了未来人工智能治理策略需要考虑研究传播的区域差异，并指出了进一步纵向和比较研究以深化对人工智能政策发展中的开放访问采纳理解的机会。 

---
# PointExplainer: Towards Transparent Parkinson's Disease Diagnosis 

**Title (ZH)**: PointExplainer: 向往透明的帕金森病诊断 

**Authors**: Xuechao Wang, Sven Nomm, Junqing Huang, Kadri Medijainen, Aaro Toomela, Michael Ruzhansky  

**Link**: [PDF](https://arxiv.org/pdf/2505.03833)  

**Abstract**: Deep neural networks have shown potential in analyzing digitized hand-drawn signals for early diagnosis of Parkinson's disease. However, the lack of clear interpretability in existing diagnostic methods presents a challenge to clinical trust. In this paper, we propose PointExplainer, an explainable diagnostic strategy to identify hand-drawn regions that drive model diagnosis. Specifically, PointExplainer assigns discrete attribution values to hand-drawn segments, explicitly quantifying their relative contributions to the model's decision. Its key components include: (i) a diagnosis module, which encodes hand-drawn signals into 3D point clouds to represent hand-drawn trajectories, and (ii) an explanation module, which trains an interpretable surrogate model to approximate the local behavior of the black-box diagnostic model. We also introduce consistency measures to further address the issue of faithfulness in explanations. Extensive experiments on two benchmark datasets and a newly constructed dataset show that PointExplainer can provide intuitive explanations with no diagnostic performance degradation. The source code is available at this https URL. 

**Abstract (ZH)**: 深神经网络在分析数字化手绘信号以实现帕金森病早期诊断方面显示出潜力。然而，现有诊断方法缺乏清晰的可解释性，这对临床信任构成挑战。本文提出了一种可解释的诊断策略PointExplainer，以识别驱动模型诊断的手绘区域。具体而言，PointExplainer为手绘段分配离散的归因值，明确量化其对模型决策的相对贡献。其主要组件包括：(i) 诊断模块，将手绘信号编码为3D点云以表示手绘轨迹，(ii) 解释模块，训练一个可解释的替代模型来逼近黑盒诊断模型的局部行为。我们还引入了一致性度量以进一步解决解释忠实性的问题。在两个基准数据集和一个新构建的数据集上进行的广泛实验表明，PointExplainer可以在不牺牲诊断性能的情况下提供直观的解释。源代码可在以下链接获得：this https URL。 

---
# Video Forgery Detection for Surveillance Cameras: A Review 

**Title (ZH)**: 监控摄像头中视频伪造检测：一个综述 

**Authors**: Noor B. Tayfor, Tarik A. Rashid, Shko M. Qader, Bryar A. Hassan, Mohammed H. Abdalla, Jafar Majidpour, Aram M. Ahmed, Hussein M. Ali, Aso M. Aladdin, Abdulhady A. Abdullah, Ahmed S. Shamsaldin, Haval M. Sidqi, Abdulrahman Salih, Zaher M. Yaseen, Azad A. Ameen, Janmenjoy Nayak, Mahmood Yashar Hamza  

**Link**: [PDF](https://arxiv.org/pdf/2505.03832)  

**Abstract**: The widespread availability of video recording through smartphones and digital devices has made video-based evidence more accessible than ever. Surveillance footage plays a crucial role in security, law enforcement, and judicial processes. However, with the rise of advanced video editing tools, tampering with digital recordings has become increasingly easy, raising concerns about their authenticity. Ensuring the integrity of surveillance videos is essential, as manipulated footage can lead to misinformation and undermine judicial decisions. This paper provides a comprehensive review of existing forensic techniques used to detect video forgery, focusing on their effectiveness in verifying the authenticity of surveillance recordings. Various methods, including compression-based analysis, frame duplication detection, and machine learning-based approaches, are explored. The findings highlight the growing necessity for more robust forensic techniques to counteract evolving forgery methods. Strengthening video forensic capabilities will ensure that surveillance recordings remain credible and admissible as legal evidence. 

**Abstract (ZH)**: 通过智能手机和数字化设备的广泛视频录制功能，基于视频的证据比以往任何时候都更加容易获取。监视视频在安全、执法和司法程序中发挥着重要作用。然而，随着高级视频编辑工具的兴起，篡改数字记录变得越来越容易，这引发了对其真实性的担忧。确保监视视频的完整性至关重要，因为篡改的视频可能导致误导信息并削弱司法裁决。本文对现有的用于检测视频伪造的法医技术进行了全面回顾，重点关注这些技术在验证监视记录的真实性方面的有效性。文中探讨了包括基于压缩的分析、帧复制检测以及基于机器学习的方法等各种方法。研究结果强调了开发更 robust 法医技术以应对不断演变的伪造方法的必要性。增强视频法医能力将确保监视记录继续具有可信度并可作为法律证据使用。 

---
# VideoLLM Benchmarks and Evaluation: A Survey 

**Title (ZH)**: VideoLLM基准与评估：一个综述 

**Authors**: Yogesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03829)  

**Abstract**: The rapid development of Large Language Models (LLMs) has catalyzed significant advancements in video understanding technologies. This survey provides a comprehensive analysis of benchmarks and evaluation methodologies specifically designed or used for Video Large Language Models (VideoLLMs). We examine the current landscape of video understanding benchmarks, discussing their characteristics, evaluation protocols, and limitations. The paper analyzes various evaluation methodologies, including closed-set, open-set, and specialized evaluations for temporal and spatiotemporal understanding tasks. We highlight the performance trends of state-of-the-art VideoLLMs across these benchmarks and identify key challenges in current evaluation frameworks. Additionally, we propose future research directions to enhance benchmark design, evaluation metrics, and protocols, including the need for more diverse, multimodal, and interpretability-focused benchmarks. This survey aims to equip researchers with a structured understanding of how to effectively evaluate VideoLLMs and identify promising avenues for advancing the field of video understanding with large language models. 

**Abstract (ZH)**: 大型语言模型的迅速发展促进了视频理解技术的重大进步。本文综述了专为视频大型语言模型（VideoLLMs）设计或使用的基准测试和评估方法。我们全面分析了当前视频理解基准测试的现状，讨论了它们的特性、评估协议及其局限性。文章分析了包括封闭集、开放集以及针对时序和空时理解任务的特殊评估方法在内的各种评估方法。我们展示了最先进的VideoLLMs在这些基准测试中的性能趋势，并指出了现有评估框架中的关键挑战。此外，我们提出了增强基准测试设计、评估指标和协议的未来研究方向，包括需要更多样化、多模态和可解释性的基准测试。本文旨在帮助研究人员系统地了解如何有效评估VideoLLMs，并识别出推进视频理解领域发展的有希望的研究方向。 

---
# Sentiment-Aware Recommendation Systems in E-Commerce: A Review from a Natural Language Processing Perspective 

**Title (ZH)**: 电商领域具有情感意识的推荐系统：自然语言处理视角的综述 

**Authors**: Yogesh Gajula  

**Link**: [PDF](https://arxiv.org/pdf/2505.03828)  

**Abstract**: E-commerce platforms generate vast volumes of user feedback, such as star ratings, written reviews, and comments. However, most recommendation engines rely primarily on numerical scores, often overlooking the nuanced opinions embedded in free text. This paper comprehensively reviews sentiment-aware recommendation systems from a natural language processing perspective, covering advancements from 2023 to early 2025. It highlights the benefits of integrating sentiment analysis into e-commerce recommenders to enhance prediction accuracy and explainability through detailed opinion extraction. Our survey categorizes recent work into four main approaches: deep learning classifiers that combine sentiment embeddings with user item interactions, transformer based methods for nuanced feature extraction, graph neural networks that propagate sentiment signals, and conversational recommenders that adapt in real time to user feedback. We summarize model architectures and demonstrate how sentiment flows through recommendation pipelines, impacting dialogue-based suggestions. Key challenges include handling noisy or sarcastic text, dynamic user preferences, and bias mitigation. Finally, we outline research gaps and provide a roadmap for developing smarter, fairer, and more user-centric recommendation tools. 

**Abstract (ZH)**: 电子商务平台生成了大量的用户反馈，包括星级评价、书面评论和评论。然而，大多数推荐引擎主要依赖数值评分，往往忽视了嵌入在自由文本中的细腻意见。本文从自然语言处理的角度全面回顾了自2023年初至2025年初的情感感知推荐系统的发展，强调将情感分析整合到电子商务推荐系统中以提高预测准确性和可解释性，通过详细的情感意见提取实现。我们总结了最近的工作分为四种主要方法：结合情感嵌入和用户项目交互的深度学习分类器、基于变换器的方法以提取细腻特征、传播情感信号的图神经网络以及适应用户反馈的对话推荐器。我们概述了模型架构，并展示了情感如何在推荐管道中流动，影响基于对话的建议。主要挑战包括处理嘈杂或讽刺文本、动态用户偏好以及偏见缓解。最后，我们概述了研究空白，并提供了开发更智能、公平和以用户为中心的推荐工具的道路图。 

---
# MISE: Meta-knowledge Inheritance for Social Media-Based Stressor Estimation 

**Title (ZH)**: MISE：基于元知识继承的社交媒体压力源估计算法 

**Authors**: Xin Wang, Ling Feng, Huijun Zhang, Lei Cao, Kaisheng Zeng, Qi Li, Yang Ding, Yi Dai, David Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2505.03827)  

**Abstract**: Stress haunts people in modern society, which may cause severe health issues if left unattended. With social media becoming an integral part of daily life, leveraging social media to detect stress has gained increasing attention. While the majority of the work focuses on classifying stress states and stress categories, this study introduce a new task aimed at estimating more specific stressors (like exam, writing paper, etc.) through users' posts on social media. Unfortunately, the diversity of stressors with many different classes but a few examples per class, combined with the consistent arising of new stressors over time, hinders the machine understanding of stressors. To this end, we cast the stressor estimation problem within a practical scenario few-shot learning setting, and propose a novel meta-learning based stressor estimation framework that is enhanced by a meta-knowledge inheritance mechanism. This model can not only learn generic stressor context through meta-learning, but also has a good generalization ability to estimate new stressors with little labeled data. A fundamental breakthrough in our approach lies in the inclusion of the meta-knowledge inheritance mechanism, which equips our model with the ability to prevent catastrophic forgetting when adapting to new stressors. The experimental results show that our model achieves state-of-the-art performance compared with the baselines. Additionally, we construct a social media-based stressor estimation dataset that can help train artificial intelligence models to facilitate human well-being. The dataset is now public at \href{this https URL}{\underline{Kaggle}} and \href{this https URL}{\underline{Hugging Face}}. 

**Abstract (ZH)**: 社会媒体中特定压力源的少样本学习估算：一种元学习框架 

---
# In-situ and Non-contact Etch Depth Prediction in Plasma Etching via Machine Learning (ANN & BNN) and Digital Image Colorimetry 

**Title (ZH)**: 基于机器学习（ANN & BNN）和数字图像颜色测量的等离子体刻蚀原位无接触刻蚀深度预测 

**Authors**: Minji Kang, Seongho Kim, Eunseo Go, Donghyeon Paek, Geon Lim, Muyoung Kim, Soyeun Kim, Sung Kyu Jang, Min Sup Choi, Woo Seok Kang, Jaehyun Kim, Jaekwang Kim, Hyeong-U Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.03826)  

**Abstract**: Precise monitoring of etch depth and the thickness of insulating materials, such as Silicon dioxide and silicon nitride, is critical to ensuring device performance and yield in semiconductor manufacturing. While conventional ex-situ analysis methods are accurate, they are constrained by time delays and contamination risks. To address these limitations, this study proposes a non-contact, in-situ etch depth prediction framework based on machine learning (ML) techniques. Two scenarios are explored. In the first scenario, an artificial neural network (ANN) is trained to predict average etch depth from process parameters, achieving a significantly lower mean squared error (MSE) compared to a linear baseline model. The approach is then extended to incorporate variability from repeated measurements using a Bayesian Neural Network (BNN) to capture both aleatoric and epistemic uncertainty. Coverage analysis confirms the BNN's capability to provide reliable uncertainty estimates. In the second scenario, we demonstrate the feasibility of using RGB data from digital image colorimetry (DIC) as input for etch depth prediction, achieving strong performance even in the absence of explicit process parameters. These results suggest that the integration of DIC and ML offers a viable, cost-effective alternative for real-time, in-situ, and non-invasive monitoring in plasma etching processes, contributing to enhanced process stability, and manufacturing efficiency. 

**Abstract (ZH)**: 精确监测硅氧化物和硅氮化物等绝缘材料的蚀刻深度对于确保半导体制造中的器件性能和产量至关重要。尽管传统的离线分析方法准确，但它们受到时间延迟和污染风险的限制。为了解决这些限制，本研究提出了一种基于机器学习技术的非接触式原位蚀刻深度预测框架。两种场景被探索。在第一种场景中，人工神经网络（ANN）被训练从工艺参数中预测平均蚀刻深度，其均方误差（MSE）显著低于线性基线模型。随后的方法扩展了重复测量的变异性，使用贝叶斯神经网络（BNN）捕捉Aleatoric和Epistemic不确定性。覆盖分析证实了BNN提供可靠不确定性估计的能力。在第二种场景中，我们展示了使用数字图像颜色度量（DIC）的RGB数据作为蚀刻深度预测输入的可能性，即使没有显式的工艺参数也取得了优异的性能。这些结果表明，将DIC与机器学习集成提供了一种可行且成本效益高的替代方法，用于等离子蚀刻过程中的实时、原位和非侵入性监测，有助于增强工艺稳定性和制造效率。 

---
# Intelligently Augmented Contrastive Tensor Factorization: Empowering Multi-dimensional Time Series Classification in Low-Data Environments 

**Title (ZH)**: 智能增强对比张量因子分解：在数据稀少环境中提升多维时间序列分类 

**Authors**: Anushiya Arunan, Yan Qin, Xiaoli Li, Yuen Chau  

**Link**: [PDF](https://arxiv.org/pdf/2505.03825)  

**Abstract**: Classification of multi-dimensional time series from real-world systems require fine-grained learning of complex features such as cross-dimensional dependencies and intra-class variations-all under the practical challenge of low training data availability. However, standard deep learning (DL) struggles to learn generalizable features in low-data environments due to model overfitting. We propose a versatile yet data-efficient framework, Intelligently Augmented Contrastive Tensor Factorization (ITA-CTF), to learn effective representations from multi-dimensional time series. The CTF module learns core explanatory components of the time series (e.g., sensor factors, temporal factors), and importantly, their joint dependencies. Notably, unlike standard tensor factorization (TF), the CTF module incorporates a new contrastive loss optimization to induce similarity learning and class-awareness into the learnt representations for better classification performance. To strengthen this contrastive learning, the preceding ITA module generates targeted but informative augmentations that highlight realistic intra-class patterns in the original data, while preserving class-wise properties. This is achieved by dynamically sampling a "soft" class prototype to guide the warping of each query data sample, which results in an augmentation that is intelligently pattern-mixed between the "soft" class prototype and the query sample. These augmentations enable the CTF module to recognize complex intra-class variations despite the limited original training data, and seek out invariant class-wise properties for accurate classification performance. The proposed method is comprehensively evaluated on five different classification tasks. Compared to standard TF and several DL benchmarks, notable performance improvements up to 18.7% were achieved. 

**Abstract (ZH)**: 面向实际系统的多维时间序列分类需要在训练数据有限的情况下细粒度学习复杂的特征，如跨维度依赖关系和类内变异性。然而，标准深度学习在低数据环境中由于模型过拟合难以学习可泛化的特征。我们提出了一种既灵活又高效的方法——智能增强对比张量分解（ITA-CTF），用于从多维时间序列中学习有效的表示。CTF模块学习时间序列的核心解释组件（如传感器因子、时域因子），以及它们的联合依赖关系。值得注意的是，与标准张量分解不同，CTF模块引入了一种新的对比损失优化，以在学习的表示中诱导相似性学习和类意识，从而提高分类性能。为了增强这种对比学习，先前的ITA模块生成了有针对性但信息丰富的增强，突出显示了原始数据中的现实类内模式，同时保留类属性。这通过动态采样一个“软”类原型来引导每个查询数据样本的形变来实现，从而产生一种在“软”类原型和查询样本之间智能模式混合的增强。这些增强使得CTF模块即使在有限的原始训练数据下也能识别复杂的类内变异性，并寻求不变的类属性以获得准确的分类性能。该方法在五个不同的分类任务上进行了全面评估，与标准张量分解和几种深度学习基准方法相比，实现了高达18.7%的性能提升。 

---
# Memory Assisted LLM for Personalized Recommendation System 

**Title (ZH)**: 基于内存辅助的大语言模型个性化推荐系统 

**Authors**: Jiarui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.03824)  

**Abstract**: Large language models (LLMs) have demonstrated significant potential in solving recommendation tasks. With proven capabilities in understanding user preferences, LLM personalization has emerged as a critical area for providing tailored responses to individuals. Current studies explore personalization through prompt design and fine-tuning, paving the way for further research in personalized LLMs. However, existing approaches are either costly and inefficient in capturing diverse user preferences or fail to account for timely updates to user history. To address these gaps, we propose the Memory-Assisted Personalized LLM (MAP). Through user interactions, we first create a history profile for each user, capturing their preferences, such as ratings for historical items. During recommendation, we extract relevant memory based on similarity, which is then incorporated into the prompts to enhance personalized recommendations. In our experiments, we evaluate MAP using a sequential rating prediction task under two scenarios: single domain, where memory and tasks are from the same category (e.g., movies), and cross-domain (e.g., memory from movies and recommendation tasks in books). The results show that MAP outperforms regular LLM-based recommenders that integrate user history directly through prompt design. Moreover, as user history grows, MAP's advantage increases in both scenarios, making it more suitable for addressing successive personalized user requests. 

**Abstract (ZH)**: 大型语言模型（LLMs）在解决推荐任务方面的潜力显著。通过在用户偏好理解方面的 proven 能力，LLM 个性化已成为提供个性化响应的关键领域。当前研究通过提示设计和微调探索个性化，为个性化 LLM 的进一步研究铺平了道路。然而，现有方法要么在捕捉多样化用户偏好方面代价高昂且效率低下，要么无法及时更新用户历史。为解决这些不足，我们提出了一种基于记忆的个性化 LLM（MAP）。通过用户交互，我们首先为每位用户创建历史档案，记录他们对历史项目的偏好，如历史项目评分。在推荐过程中，我们基于相似性提取相关记忆，并将其融入提示中以增强个性化推荐。在我们的实验中，我们使用序列评分预测任务在两种场景下评估 MAP：单一领域场景，其中记忆和任务属于同一类别（例如，电影），以及跨领域场景（例如，电影记忆和书籍推荐任务）。结果表明，MAP 在直接通过提示设计整合用户历史的常规 LLM 推荐器中表现更优。此外，随着用户历史记录的增长，MAP 在两种场景中的优势更加显著，使其更适用于解决连续的个性化用户请求。 

---
# DRSLF: Double Regularized Second-Order Low-Rank Representation for Web Service QoS Prediction 

**Title (ZH)**: DRSLF：双正则化二阶低秩表示的Web服务QoS预测 

**Authors**: Hao Wu, Jialiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03822)  

**Abstract**: Quality-of-Service (QoS) data plays a crucial role in cloud service selection. Since users cannot access all services, QoS can be represented by a high-dimensional and incomplete (HDI) matrix. Latent factor analysis (LFA) models have been proven effective as low-rank representation techniques for addressing this issue. However, most LFA models rely on first-order optimizers and use L2-norm regularization, which can lead to lower QoS prediction accuracy. To address this issue, this paper proposes a double regularized second-order latent factor (DRSLF) model with two key ideas: a) integrating L1-norm and L2-norm regularization terms to enhance the low-rank representation performance; b) incorporating second-order information by calculating the Hessian-vector product in each conjugate gradient step. Experimental results on two real-world response-time QoS datasets demonstrate that DRSLF has a higher low-rank representation capability than two baselines. 

**Abstract (ZH)**: 服务质量（QoS）数据在云服务选择中起着至关重要的作用。由于用户无法访问所有服务，QoS可以表示为高维不完全（HDI）矩阵。潜在因子分析（LFA）模型已被证明是低秩表示的有效技术，用于解决这个问题。然而，大多数LFA模型依赖于一阶优化器，并使用L2范数正则化，这可能导致较低的QoS预测准确性。为了解决这一问题，本文提出了一种双正则化二次潜在因子（DRSLF）模型，该模型包含两个关键思想：a) 结合L1范数和L2范数正则化项以提高低秩表示性能；b) 通过在每个共轭梯度步骤中计算海森berg向量乘积来引入二次信息。在两个实际响应时间QoS数据集上的实验结果显示，DRSLF在低秩表示能力上优于两个基线模型。 

---
# Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models 

**Title (ZH)**: 超越识别：评估视觉视角换位在视觉语言模型中的表现 

**Authors**: Gracjan Góral, Alicja Ziarko, Piotr Miłoś, Michał Nauman, Maciej Wołczyk, Michał Kosiński  

**Link**: [PDF](https://arxiv.org/pdf/2505.03821)  

**Abstract**: We investigate the ability of Vision Language Models (VLMs) to perform visual perspective taking using a novel set of visual tasks inspired by established human tests. Our approach leverages carefully controlled scenes, in which a single humanoid minifigure is paired with a single object. By systematically varying spatial configurations - such as object position relative to the humanoid minifigure and the humanoid minifigure's orientation - and using both bird's-eye and surface-level views, we created 144 unique visual tasks. Each visual task is paired with a series of 7 diagnostic questions designed to assess three levels of visual cognition: scene understanding, spatial reasoning, and visual perspective taking. Our evaluation of several state-of-the-art models, including GPT-4-Turbo, GPT-4o, Llama-3.2-11B-Vision-Instruct, and variants of Claude Sonnet, reveals that while they excel in scene understanding, the performance declines significantly on spatial reasoning and further deteriorates on perspective-taking. Our analysis suggests a gap between surface-level object recognition and the deeper spatial and perspective reasoning required for complex visual tasks, pointing to the need for integrating explicit geometric representations and tailored training protocols in future VLM development. 

**Abstract (ZH)**: 我们利用一套新颖的视觉任务探究视觉语言模型（VLMs）进行视觉换位思考的能力，这些任务灵感来源于已有的人类测试。通过使用谨慎控制的场景，每个场景中仅包含一个 humanoid 小型人偶和一个对象，我们系统地变化空间配置——如对象相对于 humanoid 小型人偶的位置和小型人偶的朝向——并结合鸟瞰视角和表面视图，生成了 144 个独特的视觉任务。每个视觉任务配有一系列 7 个诊断性问题，旨在评估场景理解、空间推理和视觉换位思考三个层面的认知能力。对包括 GPT-4-Turbo、GPT-4o、Llama-3.2-11B-Vision-Instruct 及 Claude Sonnet 变种在内的几种最先进的模型的评估揭示，虽然他们在场景理解方面表现出色，但在空间推理方面性能大幅下降，并且在换位思考方面更是表现不佳。我们的分析表明，在表面级物体识别和对复杂视觉任务所需的深层次空间和视角推理之间存在差距，这指出了未来 VLM 发展中需要结合明确的几何表示和定制化训练协议的需求。 

---
# Focus on the Likely: Test-time Instance-based Uncertainty Removal 

**Title (ZH)**: 关注可能的：测试时基于实例的不确定性去除 

**Authors**: Johannes Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2505.03819)  

**Abstract**: We propose two novel test-time fine-tuning methods to improve uncertain model predictions. Our methods require no auxiliary data and use the given test instance only. Instead of performing a greedy selection of the most likely class to make a prediction, we introduce an additional focus on the likely classes step during inference. By applying a single-step gradient descent, we refine predictions when an initial forward pass indicates high uncertainty. This aligns predictions more closely with the ideal of assigning zero probability to less plausible outcomes. Our theoretical discussion provides a deeper understanding highlighting the impact on shared and non-shared features among (focus) classes. The experimental evaluation highlights accuracy gains on samples exhibiting high decision uncertainty for a diverse set of models from both the text and image domain using the same hyperparameters. 

**Abstract (ZH)**: 我们提出两种新的测试时微调方法以提高不确定模型的预测质量。这两种方法不需要辅助数据，仅使用给定的测试实例。我们引入了在推理过程中关注可能类别的额外步骤，而不是进行贪婪选择最可能的类别来做出预测。通过应用单步梯度下降，我们在初始前向传播指示高不确定性时 refining 预测。这使预测更接近于将较少可能的结果分配零概率的理想状态。我们的理论讨论深化了对（关注）类别之间共享和非共享特征影响的理解。实验证明，在采用相同超参数的情况下，这些方法在文本和图像领域的一系列模型中对表现出高决策不确定性的样本展示了准确性的提升。 

---
# Program Semantic Inequivalence Game with Large Language Models 

**Title (ZH)**: 大型语言模型下的程序语义不等价性博弈 

**Authors**: Antonio Valerio Miceli-Barone, Vaishak Belle, Ali Payani  

**Link**: [PDF](https://arxiv.org/pdf/2505.03818)  

**Abstract**: Large Language Models (LLMs) can achieve strong performance on everyday coding tasks, but they can fail on complex tasks that require non-trivial reasoning about program semantics. Finding training examples to teach LLMs to solve these tasks can be challenging.
In this work, we explore a method to synthetically generate code reasoning training data based on a semantic inequivalence game SInQ: a generator agent creates program variants that are semantically distinct, derived from a dataset of real-world programming tasks, while an evaluator agent has to identify input examples that cause the original programs and the generated variants to diverge in their behaviour, with the agents training each other semi-adversarially. We prove that this setup enables theoretically unlimited improvement through self-play in the limit of infinite computational resources.
We evaluated our approach on multiple code generation and understanding benchmarks, including cross-language vulnerability detection (Lu et al., 2021), where our method improves vulnerability detection in C/C++ code despite being trained exclusively on Python code, and the challenging Python builtin identifier swap benchmark (Miceli-Barone et al., 2023), showing that whereas modern LLMs still struggle with this benchmark, our approach yields substantial improvements.
We release the code needed to replicate the experiments, as well as the generated synthetic data, which can be used to fine-tune LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以在日常编码任务中取得强大表现，但在执行需要对程序语义进行非平凡推理的复杂任务时可能会失败。找到用于训练LLMs解决这些任务的训练示例可能是具有挑战性的。

在本工作中，我们探索了一种基于语义不等价游戏SInQ的合成生成代码推理训练数据的方法：生成器代理创建与原始现实编程任务语义上不同的程序变体，而评估器代理需要识别导致原始程序和生成变体在行为上出现分歧的输入示例，代理之间以半对抗的方式进行训练。我们证明了这种设置可以通过无限计算资源下的自我对弈实现理论上无限的改进。

我们通过多个代码生成和理解基准测试评估了我们的方法，包括跨语言漏洞检测（Lu et al., 2021），在仅使用Python代码进行训练的情况下，该方法在C/C++代码中提高了漏洞检测能力；以及具有挑战性的Python内置标识符互换基准测试（Miceli-Barone et al., 2023），结果显示尽管现代LLMs在这一基准测试中仍然存在困难，但我们的方法取得了显著的改进。

我们发布了用于复现实验所需的代码，以及生成的合成数据，这些数据可以用于微调LLMs。 

---
# Modeling Behavioral Preferences of Cyber Adversaries Using Inverse Reinforcement Learning 

**Title (ZH)**: 使用逆强化学习建模网络对手的行为偏好 

**Authors**: Aditya Shinde, Prashant Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03817)  

**Abstract**: This paper presents a holistic approach to attacker preference modeling from system-level audit logs using inverse reinforcement learning (IRL). Adversary modeling is an important capability in cybersecurity that lets defenders characterize behaviors of potential attackers, which enables attribution to known cyber adversary groups. Existing approaches rely on documenting an ever-evolving set of attacker tools and techniques to track known threat actors. Although attacks evolve constantly, attacker behavioral preferences are intrinsic and less volatile. Our approach learns the behavioral preferences of cyber adversaries from forensics data on their tools and techniques. We model the attacker as an expert decision-making agent with unknown behavioral preferences situated in a computer host. We leverage attack provenance graphs of audit logs to derive a state-action trajectory of the attack. We test our approach on open datasets of audit logs containing real attack data. Our results demonstrate for the first time that low-level forensics data can automatically reveal an adversary's subjective preferences, which serves as an additional dimension to modeling and documenting cyber adversaries. Attackers' preferences tend to be invariant despite their different tools and indicate predispositions that are inherent to the attacker. As such, these inferred preferences can potentially serve as unique behavioral signatures of attackers and improve threat attribution. 

**Abstract (ZH)**: 基于系统审计日志的反向强化学习的攻击者偏好建模整体方法 

---
# Geospatial and Temporal Trends in Urban Transportation: A Study of NYC Taxis and Pathao Food Deliveries 

**Title (ZH)**: 城市交通的地理空间和时间趋势：基于纽约出租车和Pathao食品配送的研究 

**Authors**: Bidyarthi Paul, Fariha Tasnim Chowdhury, Dipta Biswas, Meherin Sultana  

**Link**: [PDF](https://arxiv.org/pdf/2505.03816)  

**Abstract**: Urban transportation plays a vital role in modern city life, affecting how efficiently people and goods move around. This study analyzes transportation patterns using two datasets: the NYC Taxi Trip dataset from New York City and the Pathao Food Trip dataset from Dhaka, Bangladesh. Our goal is to identify key trends in demand, peak times, and important geographical hotspots. We start with Exploratory Data Analysis (EDA) to understand the basic characteristics of the datasets. Next, we perform geospatial analysis to map out high-demand and low-demand regions. We use the SARIMAX model for time series analysis to forecast demand patterns, capturing seasonal and weekly variations. Lastly, we apply clustering techniques to identify significant areas of high and low demand. Our findings provide valuable insights for optimizing fleet management and resource allocation in both passenger transport and food delivery services. These insights can help improve service efficiency, better meet customer needs, and enhance urban transportation systems in diverse urban environments. 

**Abstract (ZH)**: 城市交通在现代城市生活中发挥着至关重要的作用，影响着人们和货物的移动效率。本研究使用两个数据集对交通模式进行分析：纽约市的NYC Taxi Trip数据集和达卡的Pathao Food Trip数据集。我们的目标是识别需求的关键趋势、高峰时段以及重要的地理热点。我们首先进行探索性数据分析(EDA)以了解数据集的基本特征。接着，我们进行空间地理分析以绘制高需求和低需求区域的分布。我们使用SARIMAX模型进行时间序列分析以预测需求模式，捕捉季节性和周日变化。最后，我们应用聚类技术以识别高需求和低需求的显著区域。我们的发现为优化客运和服务资源分配提供了有价值的信息，有助于提高服务效率，更好地满足客户需求，并强化多样城市环境中的城市交通系统。 

---
# Cer-Eval: Certifiable and Cost-Efficient Evaluation Framework for LLMs 

**Title (ZH)**: Cer-Eval: 可认证和成本效益评估框架 for LLMs 

**Authors**: Ganghua Wang, Zhaorun Chen, Bo Li, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03814)  

**Abstract**: As foundation models continue to scale, the size of trained models grows exponentially, presenting significant challenges for their evaluation. Current evaluation practices involve curating increasingly large datasets to assess the performance of large language models (LLMs). However, there is a lack of systematic analysis and guidance on determining the sufficiency of test data or selecting informative samples for evaluation. This paper introduces a certifiable and cost-efficient evaluation framework for LLMs. Our framework adapts to different evaluation objectives and outputs confidence intervals that contain true values with high probability. We use ``test sample complexity'' to quantify the number of test points needed for a certifiable evaluation and derive tight bounds on test sample complexity. Based on the developed theory, we develop a partition-based algorithm, named Cer-Eval, that adaptively selects test points to minimize the cost of LLM evaluation. Real-world experiments demonstrate that Cer-Eval can save 20% to 40% test points across various benchmarks, while maintaining an estimation error level comparable to the current evaluation process and providing a 95% confidence guarantee. 

**Abstract (ZH)**: 基础模型继续扩大规模，训练模型的大小呈指数增长，这为模型评估带来了重大挑战。当前评估实践涉及收集越来越大的数据集来评估大型语言模型（LLMs）的性能。然而，缺乏系统分析和指导来确定测试数据的充足性或选择具有信息量的样本进行评估。本文介绍了用于LLMs的可验证且成本效益高的评估框架。我们的框架适应不同的评估目标，并输出高概率包含真实值的信任区间。我们使用“测试样本复杂性”来量化进行可验证评估所需的测试点数量，并推导出测试样本复杂性的紧致界。基于开发的理论，我们开发了一种基于分区的算法，名为Cer-Eval，该算法能够自适应地选择测试点以最小化LLM评估的成本。实验证明，Cer-Eval可以在各种基准测试中节约20%至40%的测试点，同时维持与当前评估过程相当的估计误差水平，并提供95%的信心保证。 

---
# ScarceGAN: Discriminative Classification Framework for Rare Class Identification for Longitudinal Data with Weak Prior 

**Title (ZH)**: ScarceGAN：用于纵向数据中稀有类别识别的辨别性分类框架（基于弱先验） 

**Authors**: Surajit Chakrabarty, Rukma Talwadker, Tridib Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.03811)  

**Abstract**: This paper introduces ScarceGAN which focuses on identification of extremely rare or scarce samples from multi-dimensional longitudinal telemetry data with small and weak label prior. We specifically address: (i) severe scarcity in positive class, stemming from both underlying organic skew in the data, as well as extremely limited labels; (ii) multi-class nature of the negative samples, with uneven density distributions and partially overlapping feature distributions; and (iii) massively unlabelled data leading to tiny and weak prior on both positive and negative classes, and possibility of unseen or unknown behavior in the unlabelled set, especially in the negative class. Although related to PU learning problems, we contend that knowledge (or lack of it) on the negative class can be leveraged to learn the compliment of it (i.e., the positive class) better in a semi-supervised manner. To this effect, ScarceGAN re-formulates semi-supervised GAN by accommodating weakly labelled multi-class negative samples and the available positive samples. It relaxes the supervised discriminator's constraint on exact differentiation between negative samples by introducing a 'leeway' term for samples with noisy prior. We propose modifications to the cost objectives of discriminator, in supervised and unsupervised path as well as that of the generator. For identifying risky players in skill gaming, this formulation in whole gives us a recall of over 85% (~60% jump over vanilla semi-supervised GAN) on our scarce class with very minimal verbosity in the unknown space. Further ScarceGAN outperforms the recall benchmarks established by recent GAN based specialized models for the positive imbalanced class identification and establishes a new benchmark in identifying one of rare attack classes (0.09%) in the intrusion dataset from the KDDCUP99 challenge. 

**Abstract (ZH)**: ScarceGAN：处理极端稀少样本的半监督生成对抗网络 

---
# Grouped Sequency-arranged Rotation: Optimizing Rotation Transformation for Quantization for Free 

**Title (ZH)**: 分组序排列旋转：无需代价优化旋转变换以实现量化 

**Authors**: Euntae Choi, Sumin Song, Woosang Lim, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03810)  

**Abstract**: Large Language Models (LLMs) face deployment challenges due to high computational costs, and while Post-Training Quantization (PTQ) offers a solution, existing rotation-based methods struggle at very low bit-widths like 2-bit. We introduce a novel, training-free approach to construct an improved rotation matrix, addressing the limitations of current methods. The key contributions include leveraging the Walsh-Hadamard transform with sequency ordering, which clusters similar frequency components to reduce quantization error compared to standard Hadamard matrices, significantly improving performance. Furthermore, we propose a Grouped Sequency-arranged Rotation (GSR) using block-diagonal matrices with smaller Walsh blocks, effectively isolating outlier impacts and achieving performance comparable to optimization-based methods without requiring any training. Our method demonstrates robust performance on reasoning tasks and Perplexity (PPL) score on WikiText-2. Our method also enhances results even when applied over existing learned rotation techniques. 

**Abstract (ZH)**: 一种新型无训练旋转矩阵构建方法：基于沃尔什-哈达玛变换的分组顺序排列旋转（GSR）以应对低位宽量化挑战 

---
# When Dynamic Data Selection Meets Data Augmentation 

**Title (ZH)**: 当动态数据选择遇上了数据增强 

**Authors**: Suorong Yang, Peng Ye, Furao Shen, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.03809)  

**Abstract**: Dynamic data selection aims to accelerate training with lossless performance. However, reducing training data inherently limits data diversity, potentially hindering generalization. While data augmentation is widely used to enhance diversity, it is typically not optimized in conjunction with selection. As a result, directly combining these techniques fails to fully exploit their synergies. To tackle the challenge, we propose a novel online data training framework that, for the first time, unifies dynamic data selection and augmentation, achieving both training efficiency and enhanced performance. Our method estimates each sample's joint distribution of local density and multimodal semantic consistency, allowing for the targeted selection of augmentation-suitable samples while suppressing the inclusion of noisy or ambiguous data. This enables a more significant reduction in dataset size without sacrificing model generalization. Experimental results demonstrate that our method outperforms existing state-of-the-art approaches on various benchmark datasets and architectures, e.g., reducing 50\% training costs on ImageNet-1k with lossless performance. Furthermore, our approach enhances noise resistance and improves model robustness, reinforcing its practical utility in real-world scenarios. 

**Abstract (ZH)**: 动态数据选择旨在通过无损性能加速训练。然而，减少训练数据固有地限制了数据多样性，可能妨碍泛化能力。尽管数据增强广泛用于提高多样性，但通常未与选择优化结合使用。因此，直接将这两种技术结合起来未能充分利用它们的协同效应。为解决这一挑战，我们提出了一种新颖的在线数据训练框架，首次将动态数据选择和增强统一起来，实现训练效率和增强性能。我们的方法估计每个样本的局部密度和多模态语义一致性联合分布，从而可以目标选择适合增强的样本，同时抑制噪声或模糊数据的包含。这使得在不牺牲模型泛化能力的情况下显著减少数据集大小。实验结果表明，我们的方法在各种基准数据集和架构上优于现有最先进的方法，例如，在ImageNet-1k上将训练成本降低50%的同时保持无损性能。此外，我们的方法增强了对噪声的抵抗力和提高了模型的稳健性，强化了其实用性在实际场景中的应用。 

---
# Facilitating Video Story Interaction with Multi-Agent Collaborative System 

**Title (ZH)**: 促进基于多Agent协作系统的视频故事互动 

**Authors**: Yiwen Zhang, Jianing Hao, Zhan Wang, Hongling Sheng, Wei Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03807)  

**Abstract**: Video story interaction enables viewers to engage with and explore narrative content for personalized experiences. However, existing methods are limited to user selection, specially designed narratives, and lack customization. To address this, we propose an interactive system based on user intent. Our system uses a Vision Language Model (VLM) to enable machines to understand video stories, combining Retrieval-Augmented Generation (RAG) and a Multi-Agent System (MAS) to create evolving characters and scene experiences. It includes three stages: 1) Video story processing, utilizing VLM and prior knowledge to simulate human understanding of stories across three modalities. 2) Multi-space chat, creating growth-oriented characters through MAS interactions based on user queries and story stages. 3) Scene customization, expanding and visualizing various story scenes mentioned in dialogue. Applied to the Harry Potter series, our study shows the system effectively portrays emergent character social behavior and growth, enhancing the interactive experience in the video story world. 

**Abstract (ZH)**: 视频故事交互使得观众能够参与并探索叙事内容以获得个性化体验。然而，现有方法局限于用户选择、特别设计的叙事，并缺乏个性化定制。为了解决这一问题，我们提出了一种基于用户意图的交互系统。该系统利用视觉语言模型（VLM）使机器理解视频故事，并结合检索增强生成（RAG）和多代理系统（MAS）来创建不断发展的角色和场景体验。它包括三个阶段：1）视频故事处理，利用VLM和先验知识在三个模态中模拟人类对故事的理解。2）多空间聊天，通过MAS交互根据用户查询和故事阶段生成增长导向的角色。3）场景定制，扩展和可视化对话中提到的各种故事场景。应用于哈利·波特系列，我们的研究表明该系统有效地展示了角色社会行为的涌现性和增长性，增强了视频故事世界中的互动体验。 

---
# Perception-Informed Neural Networks: Beyond Physics-Informed Neural Networks 

**Title (ZH)**: 感知驱动的神经网络：超越物理驱动的神经网络 

**Authors**: Mehran Mazandarani, Marzieh Najariyan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03806)  

**Abstract**: This article introduces Perception-Informed Neural Networks (PrINNs), a framework designed to incorporate perception-based information into neural networks, addressing both systems with known and unknown physics laws or differential equations. Moreover, PrINNs extend the concept of Physics-Informed Neural Networks (PINNs) and their variants, offering a platform for the integration of diverse forms of perception precisiation, including singular, probability distribution, possibility distribution, interval, and fuzzy graph. In fact, PrINNs allow neural networks to model dynamical systems by integrating expert knowledge and perception-based information through loss functions, enabling the creation of modern data-driven models. Some of the key contributions include Mixture of Experts Informed Neural Networks (MOEINNs), which combine heterogeneous expert knowledge into the network, and Transformed-Knowledge Informed Neural Networks (TKINNs), which facilitate the incorporation of meta-information for enhanced model performance. Additionally, Fuzzy-Informed Neural Networks (FINNs) as a modern class of fuzzy deep neural networks leverage fuzzy logic constraints within a deep learning architecture, allowing online training without pre-training and eliminating the need for defuzzification. PrINNs represent a significant step forward in bridging the gap between traditional physics-based modeling and modern data-driven approaches, enabling neural networks to learn from both structured physics laws and flexible perception-based rules. This approach empowers neural networks to operate in uncertain environments, model complex systems, and discover new forms of differential equations, making PrINNs a powerful tool for advancing computational science and engineering. 

**Abstract (ZH)**: 感知驱动的神经网络（PrINNs）：一种结合感知信息的框架 

---
# MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance 

**Title (ZH)**: MoEQuant: 基于专家平衡采样和亲和力引导的混合专家大型语言模型量化增强 

**Authors**: Xing Hu, Zhixuan Chen, Dawei Yang, Zukang Xu, Chen Xu, Zhihang Yuan, Sifan Zhou, Jiangyong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03804)  

**Abstract**: Mixture-of-Experts (MoE) large language models (LLMs), which leverage dynamic routing and sparse activation to enhance efficiency and scalability, have achieved higher performance while reducing computational costs. However, these models face significant memory overheads, limiting their practical deployment and broader adoption. Post-training quantization (PTQ), a widely used method for compressing LLMs, encounters severe accuracy degradation and diminished generalization performance when applied to MoE models. This paper investigates the impact of MoE's sparse and dynamic characteristics on quantization and identifies two primary challenges: (1) Inter-expert imbalance, referring to the uneven distribution of samples across experts, which leads to insufficient and biased calibration for less frequently utilized experts; (2) Intra-expert imbalance, arising from MoE's unique aggregation mechanism, which leads to varying degrees of correlation between different samples and their assigned experts. To address these challenges, we propose MoEQuant, a novel quantization framework tailored for MoE LLMs. MoE-Quant includes two novel techniques: 1) Expert-Balanced Self-Sampling (EBSS) is an efficient sampling method that efficiently constructs a calibration set with balanced expert distributions by leveraging the cumulative probabilities of tokens and expert balance metrics as guiding factors. 2) Affinity-Guided Quantization (AGQ), which incorporates affinities between experts and samples into the quantization process, thereby accurately assessing the impact of individual samples on different experts within the MoE layer. Experiments demonstrate that MoEQuant achieves substantial performance gains (more than 10 points accuracy gain in the HumanEval for DeepSeekMoE-16B under 4-bit quantization) and boosts efficiency. 

**Abstract (ZH)**: MoE模型的高效量化框架：MoEQuant 

---
# RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Scalar and Vector Quantization 

**Title (ZH)**: RWKVQuant：基于代理引导混合标量和向量量化的方法量化RWKV家族模型 

**Authors**: Chen Xu, Yuxuan Yue, Zukang Xu, Xing Hu, Jiangyong Yu, Zhixuan Chen, Sifan Zhou, Zhihang Yuan, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03803)  

**Abstract**: RWKV is a modern RNN architecture with comparable performance to Transformer, but still faces challenges when deployed to resource-constrained devices. Post Training Quantization (PTQ), which is a an essential technique to reduce model size and inference latency, has been widely used in Transformer models. However, it suffers significant degradation of performance when applied to RWKV. This paper investigates and identifies two key constraints inherent in the properties of RWKV: (1) Non-linear operators hinder the parameter-fusion of both smooth- and rotation-based quantization, introducing extra computation overhead. (2) The larger amount of uniformly distributed weights poses challenges for cluster-based quantization, leading to reduced accuracy. To this end, we propose RWKVQuant, a PTQ framework tailored for RWKV models, consisting of two novel techniques: (1) a coarse-to-fine proxy capable of adaptively selecting different quantization approaches by assessing the uniformity and identifying outliers in the weights, and (2) a codebook optimization algorithm that enhances the performance of cluster-based quantization methods for element-wise multiplication in RWKV. Experiments show that RWKVQuant can quantize RWKV-6-14B into about 3-bit with less than 1% accuracy loss and 2.14x speed up. 

**Abstract (ZH)**: RWKVQuant：一种针对RWKV模型的后训练量化框架 

---
# Efficient Fine-Tuning of Quantized Models via Adaptive Rank and Bitwidth 

**Title (ZH)**: 基于自适应秩和位宽的量化模型高效微调方法 

**Authors**: Changhai Zhou, Yuhua Zhou, Qian Qiao, Weizhong Zhang, Cheng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03802)  

**Abstract**: QLoRA effectively combines low-bit quantization and LoRA to achieve memory-friendly fine-tuning for large language models (LLM). Recently, methods based on SVD for continuous update iterations to initialize LoRA matrices to accommodate quantization errors have generally failed to consistently improve performance. Dynamic mixed precision is a natural idea for continuously improving the fine-tuning performance of quantized models, but previous methods often optimize low-rank subspaces or quantization components separately, without considering their synergy. To address this, we propose \textbf{QR-Adaptor}, a unified, gradient-free strategy that uses partial calibration data to jointly search the quantization components and the rank of low-rank spaces for each layer, thereby continuously improving model performance. QR-Adaptor does not minimize quantization error but treats precision and rank allocation as a discrete optimization problem guided by actual downstream performance and memory usage. Compared to state-of-the-art (SOTA) quantized LoRA fine-tuning methods, our approach achieves a 4.89\% accuracy improvement on GSM8K, and in some cases even outperforms the 16-bit fine-tuned model while maintaining the memory footprint of the 4-bit setting. 

**Abstract (ZH)**: QLoRA有效地结合低比特量化和LoRA，实现大语言模型的内存友好型微调 

---
# Large Language Model Compression with Global Rank and Sparsity Optimization 

**Title (ZH)**: 全球秩和稀疏性优化的大语言模型压缩 

**Authors**: Changhai Zhou, Qian Qiao, Weizhong Zhang, Cheng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03801)  

**Abstract**: Low-rank and sparse composite approximation is a natural idea to compress Large Language Models (LLMs). However, such an idea faces two primary challenges that adversely affect the performance of existing methods. The first challenge relates to the interaction and cooperation between low-rank and sparse matrices, while the second involves determining weight allocation across different layers, as redundancy varies considerably among them. To address these challenges, we propose a novel two-stage LLM compression method with the capability of global rank and sparsity optimization. It is noteworthy that the overall optimization space is vast, making comprehensive optimization computationally prohibitive. Therefore, to reduce the optimization space, our first stage utilizes robust principal component analysis to decompose the weight matrices of LLMs into low-rank and sparse components, which span the low dimensional and sparse spaces containing the resultant low-rank and sparse matrices, respectively. In the second stage, we propose a probabilistic global optimization technique to jointly identify the low-rank and sparse structures within the above two spaces. The appealing feature of our approach is its ability to automatically detect the redundancy across different layers and to manage the interaction between the sparse and low-rank components. Extensive experimental results indicate that our method significantly surpasses state-of-the-art techniques for sparsification and composite approximation. 

**Abstract (ZH)**: 低秩和稀疏复合近似是压缩大型语言模型（LLMs）的自然想法。然而，这种想法面临着两个主要挑战，这些挑战会严重影响现有方法的性能。第一个挑战涉及低秩和稀疏矩阵之间的交互和协作，而第二个挑战则在于确定不同层之间的权重分配，因为这些层中的冗余程度差异很大。为了解决这些挑战，我们提出了一种具有全局秩和稀疏优化能力的新型两阶段LLM压缩方法。值得注意的是，整体优化空间非常庞大，使得全面优化在计算上是不可行的。因此，为了减少优化空间，我们第一阶段使用鲁棒主成分分析将LLM的权重矩阵分解为低秩和稀疏组件，这些组件分别占据低维和稀疏空间，其中包含相应的低秩和稀疏矩阵。在第二阶段，我们提出了一种概率全局优化技术，用于联合识别上述两个空间中的低秩和稀疏结构。我们方法的迷人之处在于其能够自动检测不同层之间的冗余，并管理稀疏和低秩组件之间的交互。广泛的研究结果表明，我们的方法在稀疏化和复合近似方面显著优于现有最先进的技术。 

---
# Scalability Matters: Overcoming Challenges in InstructGLM with Similarity-Degree-Based Sampling 

**Title (ZH)**: 可扩展性至关重要：基于相似度级别采样的InstructGLM挑战克服策略 

**Authors**: Hyun Lee, Chris Yi, Maminur Islam, B.D.S. Aritra  

**Link**: [PDF](https://arxiv.org/pdf/2505.03799)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in various natural language processing tasks; however, their application to graph-related problems remains limited, primarily due to scalability constraints and the absence of dedicated mechanisms for processing graph structures. Existing approaches predominantly integrate LLMs with Graph Neural Networks (GNNs), using GNNs as feature encoders or auxiliary components. However, directly encoding graph structures within LLMs has been underexplored, particularly in the context of large-scale graphs where token limitations hinder effective representation. To address these challenges, we propose SDM-InstructGLM, a novel instruction-tuned Graph Language Model (InstructGLM) framework that enhances scalability and efficiency without relying on GNNs. Our method introduces a similarity-degree-based biased random walk mechanism, which selectively samples and encodes graph information based on node-feature similarity and degree centrality, ensuring an adaptive and structured representation within the LLM. This approach significantly improves token efficiency, mitigates information loss due to random sampling, and enhances performance on graph-based tasks such as node classification and link prediction. Furthermore, our results demonstrate the feasibility of LLM-only graph processing, enabling scalable and interpretable Graph Language Models (GLMs) optimized through instruction-based fine-tuning. This work paves the way for GNN-free approaches to graph learning, leveraging LLMs as standalone graph reasoning models. Our source code is available on GitHub. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种自然语言处理任务中展现了强大的能力；然而，其在图相关问题中的应用受限于可扩展性限制和缺乏专门处理图结构的机制。现有方法主要将LLMs与图神经网络（GNNs）集成，使用GNNs作为特征编码器或辅助组件。然而，在大规模图中直接在LLMs中编码图结构尚未得到充分探索，尤其是当词汇量限制影响有效表示时。为解决这些问题，我们提出了一种新的指令调优图语言模型（InstructGLM）框架——SDM-InstructGLM，该框架通过不依赖于GNNs来增强可扩展性和效率。我们的方法引入了一种基于相似度-度量偏差随机游走机制，该机制根据节点特征相似性和度中心性有选择地采样和编码图信息，确保在LLM中实现适应性和结构化的表示。此方法显著提高了词汇效率，减轻了由于随机采样导致的信息损失，并提升了基于图的任务（如节点分类和链接预测）上的性能。此外，我们的结果证明了仅使用LLMs进行图处理的可行性，通过基于指令的微调实现可扩展且可解释的图语言模型（GLMs）。本工作为进一步使用LLMs作为独立图推理模型进行图学习铺平了道路。我们的源代码可在GitHub上获取。 

---
# Position: Foundation Models Need Digital Twin Representations 

**Title (ZH)**: 位置：基础模型需要数字孪生表示。 

**Authors**: Yiqing Shen, Hao Ding, Lalithkumar Seenivasan, Tianmin Shu, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2505.03798)  

**Abstract**: Current foundation models (FMs) rely on token representations that directly fragment continuous real-world multimodal data into discrete tokens. They limit FMs to learning real-world knowledge and relationships purely through statistical correlation rather than leveraging explicit domain knowledge. Consequently, current FMs struggle with maintaining semantic coherence across modalities, capturing fine-grained spatial-temporal dynamics, and performing causal reasoning. These limitations cannot be overcome by simply scaling up model size or expanding datasets. This position paper argues that the machine learning community should consider digital twin (DT) representations, which are outcome-driven digital representations that serve as building blocks for creating virtual replicas of physical processes, as an alternative to the token representation for building FMs. Finally, we discuss how DT representations can address these challenges by providing physically grounded representations that explicitly encode domain knowledge and preserve the continuous nature of real-world processes. 

**Abstract (ZH)**: 当前的基础模型依赖于标记表示，直接将连续的多模态现实世界数据分解为离散标记。它们限制基础模型仅通过统计相关性来学习现实世界的知识和关系，而非利用显性的领域知识。因此，当前的基础模型在维护跨模态的语义连贯性、捕捉精细的空间-时间动态以及执行因果推理方面存在困难。这些限制仅靠扩大模型规模或增加数据集无法克服。本文认为，机器学习社区应考虑以结果驱动的数字孪生（DT）表示作为构建基础模型的替代方案，数字孪生表示是创建物理过程虚拟副本的构建块。最后，我们讨论了数字孪生表示如何通过提供物理上 ground 的表示和显性编码领域知识来解决这些挑战，同时保留现实世界过程的连续性。 

---
# AI-Driven IRM: Transforming insider risk management with adaptive scoring and LLM-based threat detection 

**Title (ZH)**: AI驱动的IRM：通过适应性评分和基于LLM的威胁检测转型内部风险管理工作 

**Authors**: Lokesh Koli, Shubham Kalra, Rohan Thakur, Anas Saifi, Karanpreet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.03796)  

**Abstract**: Insider threats pose a significant challenge to organizational security, often evading traditional rule-based detection systems due to their subtlety and contextual nature. This paper presents an AI-powered Insider Risk Management (IRM) system that integrates behavioral analytics, dynamic risk scoring, and real-time policy enforcement to detect and mitigate insider threats with high accuracy and adaptability. We introduce a hybrid scoring mechanism - transitioning from the static PRISM model to an adaptive AI-based model utilizing an autoencoder neural network trained on expert-annotated user activity data. Through iterative feedback loops and continuous learning, the system reduces false positives by 59% and improves true positive detection rates by 30%, demonstrating substantial gains in detection precision. Additionally, the platform scales efficiently, processing up to 10 million log events daily with sub-300ms query latency, and supports automated enforcement actions for policy violations, reducing manual intervention. The IRM system's deployment resulted in a 47% reduction in incident response times, highlighting its operational impact. Future enhancements include integrating explainable AI, federated learning, graph-based anomaly detection, and alignment with Zero Trust principles to further elevate its adaptability, transparency, and compliance-readiness. This work establishes a scalable and proactive framework for mitigating emerging insider risks in both on-premises and hybrid environments. 

**Abstract (ZH)**: 内部威胁对组织安全构成重大挑战，往往由于其隐蔽性和情境性而规避传统的基于规则的检测系统。本文提出了一种AI驱动的内部风险管理系统（IRM），该系统整合了行为分析、动态风险评分和实时政策执行，以实现高准确性和适应性的内部威胁检测与缓解。我们引入了一种混合评分机制——从静态的PRISM模型过渡到利用专家标注用户活动数据训练的自动编码神经网络的适应性AI模型。通过迭代反馈循环和持续学习，该系统将假阳性率降低59%，并提高了30%的真实阳性检测率，展示了在检测精度方面的显著提升。此外，该平台高效扩展，每日处理多达1000万条日志事件，并具有亚300毫秒的查询延迟，支持对政策违规行为的自动化执行措施，减少人工干预。IRM系统的部署使事件响应时间减少了47%，突显了其操作影响。未来增强包括集成可解释的AI、联邦学习、基于图的异常检测以及与零信任原则的对齐，以进一步提高其适应性、透明度和合规性。这项工作为在企业级和混合环境中缓解新兴内部风险构建了可扩展和前瞻性的框架。 

---
# Modeling Human Behavior in a Strategic Network Game with Complex Group Dynamics 

**Title (ZH)**: 具有复杂群体动态的战略网络游戏中的人类行为建模 

**Authors**: Jacob W. Crandall, Jonathan Skaggs  

**Link**: [PDF](https://arxiv.org/pdf/2505.03795)  

**Abstract**: Human networks greatly impact important societal outcomes, including wealth and health inequality, poverty, and bullying. As such, understanding human networks is critical to learning how to promote favorable societal outcomes. As a step toward better understanding human networks, we compare and contrast several methods for learning models of human behavior in a strategic network game called the Junior High Game (JHG). These modeling methods differ with respect to the assumptions they use to parameterize human behavior (behavior vs. community-aware behavior) and the statistical moments they model (mean vs. distribution). Results show that the highest-performing method models the population's distribution rather than the mean and assumes humans use community-aware behavior rather than behavior matching. When applied to small societies (6-11 individuals), this learned model, called hCAB, closely mirrors the population dynamics of human groups (with some differences). Additionally, a user study reveals that human participants were unable to distinguish hCAB agents from other humans, thus illustrating that individual hCAB behavior plausibly mirrors human behavior in this strategic network game. 

**Abstract (ZH)**: 人类网络极大地影响着财富和健康不平等、贫困和欺凌等重要社会成果。因此，理解人类网络对于学习如何促进有利的社会成果至关重要。为更好地理解人类网络，我们比较了几种在名为初中游戏（JHG）的战略网络游戏中学习人类行为模型的方法。这些建模方法在参数化人类行为（行为 vs. 社区意识行为）和建模的统计时刻（均值 vs. 分布）的假设上有所不同。结果表明，表现最高的方法建模的是人群的分布而非均值，并假设人类使用社区意识行为而非匹配行为。当应用于小社会（6-11 人）时，这种学习到的模型（称为hCAB）在一定程度上反映了人类群体的动态特性。此外，一项用户研究显示，人类参与者无法区分hCAB代理与其他人类，从而说明了个体hCAB行为在这一战略网络游戏中可能真实地反映了人类行为。 

---
# LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection 

**Title (ZH)**: LENSLLM: 揭示大型语言模型选择的微调动态 

**Authors**: Xinyue Zeng, Haohui Wang, Junhong Lin, Jun Wu, Tyler Cody, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.03793)  

**Abstract**: The proliferation of open-sourced Large Language Models (LLMs) and diverse downstream tasks necessitates efficient model selection, given the impracticality of fine-tuning all candidates due to computational constraints. Despite the recent advances in LLM selection, a fundamental research question largely remains nascent: how can we model the dynamic behaviors of LLMs during fine-tuning, thereby enhancing our understanding of their generalization performance across diverse downstream tasks? In this work, we propose a novel theoretical framework that provides a proper lens to assess the generalization capabilities of LLMs, thereby enabling accurate and efficient LLM selection for downstream applications. In particular, we first derive a Hessian-based PAC-Bayes generalization bound that unveils fine-tuning dynamics of LLMs and then introduce LENSLLM, a Neural Tangent Kernel(NTK)-based Rectified Scaling Model that enables accurate performance predictions across diverse tasks while maintaining computational efficiency. Extensive empirical results on 3 large-scale benchmarks demonstrate that our model achieves up to 91.1% accuracy and reduces up to 88.5% computational cost in LLM selection, outperforming 5 state-of-the-art methods. We open-source our proposed LENSLLM model and corresponding results at the Github link: this https URL. 

**Abstract (ZH)**: 开源大型语言模型（LLMs）的迅速增长及其多样的下游任务要求高效的选择模型，鉴于精细调整所有候选模型在计算上的不可行性。尽管在LLM选择方面取得了最近的进展，但仍有一个基本的研究问题尚未得到充分探索：如何在精细调整过程中建模LLM的动力学行为，从而提升其在多样的下游任务中表现的泛化理解？在本文中，我们提出了一种新的理论框架，以评估LLM的泛化能力，并由此实现对下游应用中高效且准确的LLM选择。特别是，我们首先推导出基于Hessian的PAC-Bayes泛化边界，揭示了LLM的精细调整动力学，然后引入了基于神经摆动核（NTK）的反向放大规模模型LENSLLM，该模型能够在保持计算效率的同时，实现对多样化任务的准确性能预测。在三个大规模基准上的广泛实验证明，我们的模型在LLM选择中达到了高达91.1%的准确率并降低了高达88.5%的计算成本，超过了5种最先进的方法。我们已在GitHub链接处开源了提出的LENSLLM模型及其结果：this https URL。 

---
# Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning 

**Title (ZH)**: 通过反事实软强化学习实现高效的在线VLM代理调优 

**Authors**: Lang Feng, Weihao Tan, Zhiyi Lyu, Longtao Zheng, Haiyang Xu, Ming Yan, Fei Huang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.03792)  

**Abstract**: Online fine-tuning vision-language model (VLM) agents with reinforcement learning (RL) has shown promise for equipping agents with multi-step, goal-oriented capabilities in dynamic environments. However, their open-ended textual action space and non-end-to-end nature of action generation present significant challenges to effective online exploration in RL, e.g., explosion of the exploration space. We propose a novel online fine-tuning method, Counterfactual Soft Reinforcement Learning (CoSo), better suited to the textual output space of VLM agents. Compared to prior methods that assign uniform uncertainty to all tokens, CoSo leverages counterfactual reasoning to dynamically assess the causal influence of individual tokens on post-processed actions. By prioritizing the exploration of action-critical tokens while reducing the impact of semantically redundant or low-impact tokens, CoSo enables a more targeted and efficient online rollout process. We provide theoretical analysis proving CoSo's convergence and policy improvement guarantees, and extensive empirical evaluations supporting CoSo's effectiveness. Our results across a diverse set of agent tasks, including Android device control, card gaming, and embodied AI, highlight its remarkable ability to enhance exploration efficiency and deliver consistent performance gains. The code is available at this https URL. 

**Abstract (ZH)**: 基于反事实软强化学习的在线微调视觉语言模型智能体方法在动态环境中的多步目标导向能力研究 

---
# Practical Boolean Backpropagation 

**Title (ZH)**: 实用布尔反向传播 

**Authors**: Simon Golbert  

**Link**: [PDF](https://arxiv.org/pdf/2505.03791)  

**Abstract**: Boolean neural networks offer hardware-efficient alternatives to real-valued models. While quantization is common, purely Boolean training remains underexplored. We present a practical method for purely Boolean backpropagation for networks based on a single specific gate we chose, operating directly in Boolean algebra involving no numerics. Initial experiments confirm its feasibility. 

**Abstract (ZH)**: 纯布尔训练为基于单一特定门电路的网络提供硬件高效的替代方案：无需数值运算的纯布尔反向传播仍然未被充分探索。初步实验证实了其可行性。 

---
# A Time-Series Data Augmentation Model through Diffusion and Transformer Integration 

**Title (ZH)**: 通过扩散和变换器集成的时间序列数据增强模型 

**Authors**: Yuren Zhang, Zhongnan Pu, Lei Jing  

**Link**: [PDF](https://arxiv.org/pdf/2505.03790)  

**Abstract**: With the development of Artificial Intelligence, numerous real-world tasks have been accomplished using technology integrated with deep learning. To achieve optimal performance, deep neural networks typically require large volumes of data for training. Although advances in data augmentation have facilitated the acquisition of vast datasets, most of this data is concentrated in domains like images and speech. However, there has been relatively less focus on augmenting time-series data. To address this gap and generate a substantial amount of time-series data, we propose a simple and effective method that combines the Diffusion and Transformer models. By utilizing an adjusted diffusion denoising model to generate a large volume of initial time-step action data, followed by employing a Transformer model to predict subsequent actions, and incorporating a weighted loss function to achieve convergence, the method demonstrates its effectiveness. Using the performance improvement of the model after applying augmented data as a benchmark, and comparing the results with those obtained without data augmentation or using traditional data augmentation methods, this approach shows its capability to produce high-quality augmented data. 

**Abstract (ZH)**: 随着人工智能的发展，深度学习技术被广泛应用于许多实际任务。为了实现最优性能，深度神经网络通常需要大量的数据进行训练。尽管数据增强技术的进步促进了大量数据的获取，但大多数据集中在图像和语音领域。然而，对时间序列数据的数据增强研究相对较少。为填补这一空白并生成大量时间序列数据，我们提出了一种简单而有效的方法，该方法结合了扩散模型和变压器模型。通过使用调整后的扩散去噪模型生成大量初始时间步动作数据，然后利用变压器模型预测后续动作，并结合加权损失函数以实现模型收敛，该方法展示了其有效性。通过将模型在应用增强数据后的性能提升作为基准，并将其结果与未使用数据增强或使用传统数据增强方法的结果进行比较，该方法证明了其生成高质量增强数据的能力。 

---
# Calibrating Uncertainty Quantification of Multi-Modal LLMs using Grounding 

**Title (ZH)**: 多模态大语言模型不确定性量化校准研究 

**Authors**: Trilok Padhi, Ramneet Kaur, Adam D. Cobb, Manoj Acharya, Anirban Roy, Colin Samplawski, Brian Matejek, Alexander M. Berenbeim, Nathaniel D. Bastian, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2505.03788)  

**Abstract**: We introduce a novel approach for calibrating uncertainty quantification (UQ) tailored for multi-modal large language models (LLMs). Existing state-of-the-art UQ methods rely on consistency among multiple responses generated by the LLM on an input query under diverse settings. However, these approaches often report higher confidence in scenarios where the LLM is consistently incorrect. This leads to a poorly calibrated confidence with respect to accuracy. To address this, we leverage cross-modal consistency in addition to self-consistency to improve the calibration of the multi-modal models. Specifically, we ground the textual responses to the visual inputs. The confidence from the grounding model is used to calibrate the overall confidence. Given that using a grounding model adds its own uncertainty in the pipeline, we apply temperature scaling - a widely accepted parametric calibration technique - to calibrate the grounding model's confidence in the accuracy of generated responses. We evaluate the proposed approach across multiple multi-modal tasks, such as medical question answering (Slake) and visual question answering (VQAv2), considering multi-modal models such as LLaVA-Med and LLaVA. The experiments demonstrate that the proposed framework achieves significantly improved calibration on both tasks. 

**Abstract (ZH)**: 我们提出了一种针对多模态大型语言模型的新型不确定性量化校准方法。现有的先进不确定性量化方法依赖于大型语言模型在多种设置下对输入查询生成的多个响应之间的一致性。然而，这些方法往往在大型语言模型一致错误的情况下报告更高的置信度，导致与准确度不匹配的校准置信度。为了解决这一问题，我们除了利用自我一致性之外，还利用跨模态一致性来提高多模态模型的校准。具体而言，我们将文本响应与视觉输入对接。对接模型的置信度用于校准整体置信度。由于使用对接模型会在管道中增加自身的不确定性，我们应用温度缩放——一种广泛接受的参数校准技术——来校准对接模型对未来响应准确性的置信度。我们在医疗问答（Slake）和视觉问答（VQAv2）等多个多模态任务上评估了所提出的方法，涉及多模态模型如LLaVA-Med和LLaVA。实验结果表明，所提出的框架在这两个任务上实现了显著改进的校准。 

---
# ArrhythmiaVision: Resource-Conscious Deep Learning Models with Visual Explanations for ECG Arrhythmia Classification 

**Title (ZH)**: 心律失常视觉：考虑资源的深学习模型及其心电图心律失常分类的可视化解释 

**Authors**: Zuraiz Baig, Sidra Nasir, Rizwan Ahmed Khan, Muhammad Zeeshan Ul Haque  

**Link**: [PDF](https://arxiv.org/pdf/2505.03787)  

**Abstract**: Cardiac arrhythmias are a leading cause of life-threatening cardiac events, highlighting the urgent need for accurate and timely detection. Electrocardiography (ECG) remains the clinical gold standard for arrhythmia diagnosis; however, manual interpretation is time-consuming, dependent on clinical expertise, and prone to human error. Although deep learning has advanced automated ECG analysis, many existing models abstract away the signal's intrinsic temporal and morphological features, lack interpretability, and are computationally intensive-hindering their deployment on resource-constrained platforms. In this work, we propose two novel lightweight 1D convolutional neural networks, ArrhythmiNet V1 and V2, optimized for efficient, real-time arrhythmia classification on edge devices. Inspired by MobileNet's depthwise separable convolutional design, these models maintain memory footprints of just 302.18 KB and 157.76 KB, respectively, while achieving classification accuracies of 0.99 (V1) and 0.98 (V2) on the MIT-BIH Arrhythmia Dataset across five classes: Normal Sinus Rhythm, Left Bundle Branch Block, Right Bundle Branch Block, Atrial Premature Contraction, and Premature Ventricular Contraction. In order to ensure clinical transparency and relevance, we integrate Shapley Additive Explanations and Gradient-weighted Class Activation Mapping, enabling both local and global interpretability. These techniques highlight physiologically meaningful patterns such as the QRS complex and T-wave that contribute to the model's predictions. We also discuss performance-efficiency trade-offs and address current limitations related to dataset diversity and generalizability. Overall, our findings demonstrate the feasibility of combining interpretability, predictive accuracy, and computational efficiency in practical, wearable, and embedded ECG monitoring systems. 

**Abstract (ZH)**: 心脏病律失常是导致危及生命的心脏事件的主要原因，凸显了对准确及时检测的迫切需求。心电图（ECG）仍然是心律失常诊断的临床金标准；然而，人工解释耗时且依赖临床专业知识，容易出错。尽管深度学习促进了自动ECG分析的进步，但许多现有模型忽略了信号的内在时域和形态特征，缺乏可解释性，计算成本高，阻碍了它们在资源受限平台上部署。在本文中，我们提出了两个新型的轻量级一维卷积神经网络模型——ArrhythmiNet V1和V2，旨在实现边缘设备上的高效实时心律失常分类。受MobileNet深度可分离卷积设计的启发，这些模型的内存占用仅为302.18 KB和157.76 KB，而在MIT-BIH心律失常数据库中，分别在五个类别（正常窦性心律、左束支传导阻滞、右束支传导阻滞、房性期前收缩和室性期前收缩）上实现了0.99（V1）和0.98（V2）的分类精度。为确保临床透明度和相关性，我们整合了Shapley加权解释和梯度加权分类激活映射，实现了局部和全局可解释性。这些技术突显了如QRS波群和T波等生理学上有意义的模式，这些模式对模型预测有贡献。我们还讨论了性能效率权衡，并解决了与数据集多样性及泛化能力相关的问题。总体而言，我们的研究结果表明，在实际、可穿戴和嵌入式ECG监测系统中结合可解释性、预测准确性和计算效率的可能性。 

---
# GPU Performance Portability needs Autotuning 

**Title (ZH)**: GPU 性能移植需要自动调优 

**Authors**: Burkhard Ringlein, Thomas Parnell, Radu Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2505.03780)  

**Abstract**: As LLMs grow in complexity, achieving state-of-the-art performance requires tight co-design across algorithms, software, and hardware. Today's reliance on a single dominant platform limits portability, creates vendor lock-in, and raises barriers for new AI hardware. In this work, we make the case for combining just-in-time (JIT) compilation with kernel parameter autotuning to enable portable, state-of-the-art performance LLM execution without code changes. Focusing on flash attention -- a widespread performance-critical LLM kernel -- we demonstrate that this approach explores up to 15x more kernel parameter configurations, produces significantly more diverse code across multiple dimensions, and even outperforms vendor-optimized implementations by up to 230%, all while reducing kernel code size by 70x and eliminating manual code optimizations. Our results highlight autotuning as a promising path to unlocking model portability across GPU vendors. 

**Abstract (ZH)**: 随着LLMs日益复杂，实现最优性能需要在算法、软件和硬件之间进行紧密协同设计。当前对单一主导平台的依赖限制了灵活性， creates vendor lock-in，并提高了新AI硬件的进入门槛。在本工作中，我们提出了结合即时编译（JIT）与内核参数自调优，以实现无需修改代码的便携式最优性能LLM执行。我们专注于闪注意力机制——一种广泛应用的关键性能内核——证明了这种方法探索了多达15倍更多的内核参数配置，产生了在多个维度上显著更多样化的代码，并且在某些情况下甚至比供应商优化的实现提高了230%，同时将内核代码大小减少了70倍，并消除了手动代码优化。我们的结果突显了自调优在解锁不同GPU供应商间模型便携性方面的潜力。 

---
# The Influence of Text Variation on User Engagement in Cross-Platform Content Sharing 

**Title (ZH)**: 跨平台内容共享中文本变异对用户参与度的影响 

**Authors**: Yibo Hu, Yiqiao Jin, Meng Ye, Ajay Divakaran, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03769)  

**Abstract**: In today's cross-platform social media landscape, understanding factors that drive engagement for multimodal content, especially text paired with visuals, remains complex. This study investigates how rewriting Reddit post titles adapted from YouTube video titles affects user engagement. First, we build and analyze a large dataset of Reddit posts sharing YouTube videos, revealing that 21% of post titles are minimally modified. Statistical analysis demonstrates that title rewrites measurably improve engagement. Second, we design a controlled, multi-phase experiment to rigorously isolate the effects of textual variations by neutralizing confounding factors like video popularity, timing, and community norms. Comprehensive statistical tests reveal that effective title rewrites tend to feature emotional resonance, lexical richness, and alignment with community-specific norms. Lastly, pairwise ranking prediction experiments using a fine-tuned BERT classifier achieves 74% accuracy, significantly outperforming near-random baselines, including GPT-4o. These results validate that our controlled dataset effectively minimizes confounding effects, allowing advanced models to both learn and demonstrate the impact of textual features on engagement. By bridging quantitative rigor with qualitative insights, this study uncovers engagement dynamics and offers a robust framework for future cross-platform, multimodal content strategies. 

**Abstract (ZH)**: 在当今跨平台社交媒体 landscape 中，理解推动多媒体内容（尤其是配有视觉的文本）参与度的因素依然复杂。本研究探讨如何修改源自 YouTube 视频标题的 Reddit 文章标题以影响用户参与度。首先，我们构建并分析了一个大规模的 Reddit 分享 YouTube 视频的文章数据集，发现 21% 的文章标题仅进行了轻微修改。统计分析表明，标题修改显著提高了参与度。其次，我们设计了一项严密控制的多阶段实验，通过消除如视频受欢迎程度、时间因素和社区规范等混淆变量，严格隔离文本变化的影响。全面的统计测试表明，有效的标题修改通常具有情感共鸣、词汇丰富性和与特定社区规范的契合。最后，使用微调后的 BERT 分类器进行成对排名预测实验实现了 74% 的准确率，显著优于近随机基线，包括 GPT-4o。这些结果验证了我们控制的数据集有效地最小化了混淆效应，使高级模型能够学习并展示文本特征对参与度的影响。通过结合定量严谨性和定性洞察，本研究揭示了参与动态，并为未来的跨平台多媒体内容策略提供了稳健框架。 

---
# Ultra-Low-Power Spiking Neurons in 7 nm FinFET Technology: A Comparative Analysis of Leaky Integrate-and-Fire, Morris-Lecar, and Axon-Hillock Architectures 

**Title (ZH)**: 7 nm FinFET 技术下的超低功耗脉冲神经元：Leaky Integrate-and-Fire、Morris-Lecar 和 Axon-Hillock 架构的比较分析 

**Authors**: Logan Larsh, Raiyan Siddique, Sarah Sharif Yaser Mike Banad  

**Link**: [PDF](https://arxiv.org/pdf/2505.03764)  

**Abstract**: Neuromorphic computing aims to replicate the brain's remarkable energy efficiency and parallel processing capabilities for large-scale artificial intelligence applications. In this work, we present a comprehensive comparative study of three spiking neuron circuit architectures-Leaky-Integrate-and-Fire (LIF), Morris-Lecar (ML), and Axon-Hillock (AH)-implemented in a 7 nm FinFET technology. Through extensive SPICE simulations, we explore the optimization of spiking frequency, energy per spike, and static power consumption. Our results show that the AH design achieves the highest throughput, demonstrating multi-gigahertz firing rates (up to 3 GHz) with attojoule energy costs. By contrast, the ML architecture excels in subthreshold to near-threshold regimes, offering robust low-power operation (as low as 0.385 aJ/spike) and biological bursting behavior. Although LIF benefits from a decoupled current mirror for high-frequency operation, it exhibits slightly higher static leakage compared to ML and AH at elevated supply voltages. Comparisons with previous node implementations (22 nm planar, 28 nm) reveal that 7 nm FinFETs can drastically boost energy efficiency and speed albeit at the cost of increased subthreshold leakage in deep subthreshold regions. By quantifying design trade-offs for each neuron architecture, our work provides a roadmap for optimizing spiking neuron circuits in advanced nanoscale technologies to deliver neuromorphic hardware capable of both ultra-low-power operation and high computational throughput. 

**Abstract (ZH)**: 神经形态计算旨在复制大脑的卓越能效和并行处理能力，以应对大规模人工智能应用。在本工作中，我们全面比较了三种尖针神经元电路架构——Leaky-Integrate-and-Fire (LIF)、Morris-Lecar (ML) 和 Axon-Hillock (AH)，这些架构在7 nm FinFET技术中实现。通过广泛的SPICE仿真，我们探索了尖针频率、每尖针能耗和静态功耗的优化。我们的结果显示，AH设计实现了最高的吞吐量，其多吉赫兹发射率（高达3 GHz）伴随阿拓焦耳级的能量成本。相比之下，ML架构在亚阈值到近阈值区域内表现出色，提供了稳健的低功耗操作（低至0.385 aJ/尖针）和生物学上的爆发行为。尽管LIF得益于用于高频操作的解耦电流镜，但在高供电电压下，其静态漏电流略高于ML和AH。与之前的节点实现（22 nm 平面，28 nm）的比较揭示，7 nm FinFET可以大幅提高能效和速度，尽管代价是深亚阈值区域内的亚阈值泄漏增加。通过量化每种神经元架构的设计权衡，我们的工作为在先进的纳米尺度技术中优化尖针神经元电路提供了路线图，以实现既能进行超低功耗操作又能提供高计算吞吐量的神经形态硬件。 

---
# Splitwiser: Efficient LM inference with constrained resources 

**Title (ZH)**: Splitwiser：在受限资源下高效的LM推理 

**Authors**: Asad Aali, Adney Cardoza, Melissa Capo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03763)  

**Abstract**: Efficient inference of LLMs remains a crucial challenge, with two main phases: a compute-intensive prompt computation and a memory-intensive token generation. Despite existing batching and scheduling techniques, token generation phases fail to fully utilize compute resources, especially when compared to prompt computation phases. To address these challenges, we propose Splitwiser, a methodology that splits the two phases of an LLM inference request onto the same GPU, thereby reducing overhead and improving memory access and cache utilization. By eliminating the need to transfer data across devices, Splitwiser aims to minimize network-related overheads. In this report, we describe the basic structure of our proposed pipeline while sharing preliminary results and analysis. We implement our proposed multiprocessing design on two widely-used and independent LLM architectures: Huggingface and vLLM. We open-source our code for the respective implementations: 1) Huggingface (this https URL), and 2) vLLM (this https URL). 

**Abstract (ZH)**: 高效的大型语言模型推理仍然是一项关键挑战，主要包括两个主要阶段：密集的提示计算和密集的标记生成。尽管存在现有的批处理和调度技术，但标记生成阶段未能充分利用计算资源，尤其是在与提示计算阶段相比时。为应对这些挑战，我们提出了一种名为Splitwiser的方法，该方法将大型语言模型推理请求的两个阶段分配到同一个GPU上，从而减少开销并提高内存访问和缓存利用率。通过消除在不同设备之间传输数据的需要，Splitwiser旨在最小化与网络相关的时间开销。在本报告中，我们描述了我们提出的管道的基本结构，并分享了初步结果和分析。我们将在两个广泛使用的独立大型语言模型架构Huggingface和vLLM上实现我们提出的多处理设计。我们开源了相应的实现代码：1) Huggingface (点击此链接), 2) vLLM (点击此链接)。 

---
# Deep Reinforcement Learning for Investor-Specific Portfolio Optimization: A Volatility-Guided Asset Selection Approach 

**Title (ZH)**: 基于波动率引导的资产选择方法的投资者特定投资组合优化的深度强化学习研究 

**Authors**: Arishi Orra, Aryan Bhambu, Himanshu Choudhary, Manoj Thakur, Selvaraju Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03760)  

**Abstract**: Portfolio optimization requires dynamic allocation of funds by balancing the risk and return tradeoff under dynamic market conditions. With the recent advancements in AI, Deep Reinforcement Learning (DRL) has gained prominence in providing adaptive and scalable strategies for portfolio optimization. However, the success of these strategies depends not only on their ability to adapt to market dynamics but also on the careful pre-selection of assets that influence overall portfolio performance. Incorporating the investor's preference in pre-selecting assets for a portfolio is essential in refining their investment strategies. This study proposes a volatility-guided DRL-based portfolio optimization framework that dynamically constructs portfolios based on investors' risk profiles. The Generalized Autoregressive Conditional Heteroscedasticity (GARCH) model is utilized for volatility forecasting of stocks and categorizes them based on their volatility as aggressive, moderate, and conservative. The DRL agent is then employed to learn an optimal investment policy by interacting with the historical market data. The efficacy of the proposed methodology is established using stocks from the Dow $30$ index. The proposed investor-specific DRL-based portfolios outperformed the baseline strategies by generating consistent risk-adjusted returns. 

**Abstract (ZH)**: 基于波动率引导的DRL动态投资组合优化框架：根据投资者风险偏好构建投资组合，并在道琼斯30指数股票上验证其优势 

---
# Improving the Serving Performance of Multi-LoRA Large Language Models via Efficient LoRA and KV Cache Management 

**Title (ZH)**: 通过高效LoRA和KV缓存管理提高多LoRA大型语言模型的服务性能 

**Authors**: Hang Zhang, Jiuchen Shi, Yixiao Wang, Quan Chen, Yizhou Shan, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03756)  

**Abstract**: Multiple Low-Rank Adapters (Multi-LoRAs) are gaining popularity for task-specific Large Language Model (LLM) applications. For multi-LoRA serving, caching hot KV caches and LoRA adapters in high bandwidth memory of accelerations can improve inference performance. However, existing Multi-LoRA inference systems fail to optimize serving performance like Time-To-First-Toke (TTFT), neglecting usage dependencies when caching LoRAs and KVs. We therefore propose FASTLIBRA, a Multi-LoRA caching system to optimize the serving performance. FASTLIBRA comprises a dependency-aware cache manager and a performance-driven cache swapper. The cache manager maintains the usage dependencies between LoRAs and KV caches during the inference with a unified caching pool. The cache swapper determines the swap-in or out of LoRAs and KV caches based on a unified cost model, when the HBM is idle or busy, respectively. Experimental results show that ELORA reduces the TTFT by 63.4% on average, compared to state-of-the-art works. 

**Abstract (ZH)**: 多低秩适配器（Multi-LoRAs）缓存系统FASTLIBRA：优化任务特定大型语言模型推理性能 

---
# AI-Powered Agile Analog Circuit Design and Optimization 

**Title (ZH)**: 基于AI的敏捷模拟电路设计与优化 

**Authors**: Jinhai Hu, Wang Ling Goh, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03750)  

**Abstract**: Artificial intelligence (AI) techniques are transforming analog circuit design by automating device-level tuning and enabling system-level co-optimization. This paper integrates two approaches: (1) AI-assisted transistor sizing using Multi-Objective Bayesian Optimization (MOBO) for direct circuit parameter optimization, demonstrated on a linearly tunable transconductor; and (2) AI-integrated circuit transfer function modeling for system-level optimization in a keyword spotting (KWS) application, demonstrated by optimizing an analog bandpass filter within a machine learning training loop. The combined insights highlight how AI can improve analog performance, reduce design iteration effort, and jointly optimize analog components and application-level metrics. 

**Abstract (ZH)**: 人工智能技术正在通过自动化器件级调谐和实现系统级协同优化来变革模拟电路设计。本文结合了两种方法：（1）使用多目标贝叶斯优化（MOBO）的人工智能辅助晶体管尺寸优化，用于直接电路参数优化，以线性可调跨导为例；（2）将人工智能集成到电路传递函数建模中，在关键词识别（KWS）应用中实现系统级优化，通过在机器学习训练回路中优化一个模拟带通滤波器来演示。结合这些见解突显了人工智能如何提高模拟性能、减少设计迭代努力，并实现模拟组件和应用级指标的联合优化。 

---
# APSQ: Additive Partial Sum Quantization with Algorithm-Hardware Co-Design 

**Title (ZH)**: APSQ: 增量部分和量化结合算法-硬件协同设计 

**Authors**: Yonghao Tan, Pingcheng Dong, Yongkun Wu, Yu Liu, Xuejiao Liu, Peng Luo, Shih-Yang Liu, Xijie Huang, Dong Zhang, Luhong Liang, Kwang-Ting Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03748)  

**Abstract**: DNN accelerators, significantly advanced by model compression and specialized dataflow techniques, have marked considerable progress. However, the frequent access of high-precision partial sums (PSUMs) leads to excessive memory demands in architectures utilizing input/weight stationary dataflows. Traditional compression strategies have typically overlooked PSUM quantization, which may account for 69% of power consumption. This study introduces a novel Additive Partial Sum Quantization (APSQ) method, seamlessly integrating PSUM accumulation into the quantization framework. A grouping strategy that combines APSQ with PSUM quantization enhanced by a reconfigurable architecture is further proposed. The APSQ performs nearly lossless on NLP and CV tasks across BERT, Segformer, and EfficientViT models while compressing PSUMs to INT8. This leads to a notable reduction in energy costs by 28-87%. Extended experiments on LLaMA2-7B demonstrate the potential of APSQ for large language models. Code is available at this https URL. 

**Abstract (ZH)**: DNN加速器在模型压缩和专业数据流技术的推动下取得了显著进步，然而，在采用输入/权重固定数据流架构中，高精度部分和（PSUM）的频繁访问导致了过高的内存需求。传统压缩策略通常忽略了PSUM量化，而这可能占用了69%的功耗。本研究提出了一种新颖的加性部分和量化（APSQ）方法，将PSUM累加无缝集成到量化框架中。进一步提出了结合APSQ与增强型重配置架构的PSUM量化组合策略。APSQ在BERT、Segformer和EfficientViT模型的NLP和CV任务中几乎无损压缩PSUM至INT8，从而减少高达28-87%的能耗。扩展实验表明，APSQ对大型语言模型LLaMA2-7B具有潜在优势。相关代码可访问此链接。 

---
# The Evolution of Rough Sets 1970s-1981 

**Title (ZH)**: 粗糙集的发展1970年代至1981年 

**Authors**: Viktor Marek, Ewa Orłowska, Ivo Düntsch  

**Link**: [PDF](https://arxiv.org/pdf/2505.03747)  

**Abstract**: In this note research and publications by Zdzisław Pawlak and his collaborators from 1970s and 1981 are recalled. Focus is placed on the sources of inspiration which one can identify on the basis of those publications. Finally, developments from 1981 related to rough sets and information systems are outlined. 

**Abstract (ZH)**: 20世纪70年代和1981年Zdzisław Pawlak及其合作者的研究与出版回顾：基于这些出版物的灵感来源及其后的研究进展概述 

---
# Promoting Security and Trust on Social Networks: Explainable Cyberbullying Detection Using Large Language Models in a Stream-Based Machine Learning Framework 

**Title (ZH)**: 基于流式机器学习框架的可解释网络欺凌检测：使用大型语言模型促进社交网络的安全与信任 

**Authors**: Silvia García-Méndez, Francisco De Arriba-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2505.03746)  

**Abstract**: Social media platforms enable instant and ubiquitous connectivity and are essential to social interaction and communication in our technological society. Apart from its advantages, these platforms have given rise to negative behaviors in the online community, the so-called cyberbullying. Despite the many works involving generative Artificial Intelligence (AI) in the literature lately, there remain opportunities to study its performance apart from zero/few-shot learning strategies. Accordingly, we propose an innovative and real-time solution for cyberbullying detection that leverages stream-based Machine Learning (ML) models able to process the incoming samples incrementally and Large Language Models (LLMS) for feature engineering to address the evolving nature of abusive and hate speech online. An explainability dashboard is provided to promote the system's trustworthiness, reliability, and accountability. Results on experimental data report promising performance close to 90 % in all evaluation metrics and surpassing those obtained by competing works in the literature. Ultimately, our proposal contributes to the safety of online communities by timely detecting abusive behavior to prevent long-lasting harassment and reduce the negative consequences in society. 

**Abstract (ZH)**: 社交媒体平台 enables 即时和普遍的连接性，并在我们的技术社会中对于社会互动和交流是必不可少的。除了其优势，这些平台还催生了在线社区中的负面行为，即网络欺凌。尽管近期文献中涉及生成人工智能（AI）的工作很多，但仍有机会研究其性能，不仅限于零/少-shot 学习策略。因此，我们提出了一个创新且实时的网络欺凌检测解决方案，该方案利用基于流的机器学习（ML）模型以逐增量处理传入样本，并运用大型语言模型（LLMs）进行特征工程，以应对在线恶意和仇恨言论的演变性质。提供了可解释性仪表板以促进系统的可信度、可靠性和问责制。实验数据上的结果报告显示，在所有评估指标上接近90%的性能表现，并且超过文献中竞争工作的结果。最终，我们的提案通过及时检测恶意行为来 contributeto 在线社区的安全，从而预防持久性骚扰并减少社会的负面影响。 

---
# AccLLM: Accelerating Long-Context LLM Inference Via Algorithm-Hardware Co-Design 

**Title (ZH)**: AccLLM: 通过算法-硬件协同设计加速长上下文LLM推理 

**Authors**: Yanbiao Liang, Huihong Shi, Haikuo Shao, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03745)  

**Abstract**: Recently, large language models (LLMs) have achieved huge success in the natural language processing (NLP) field, driving a growing demand to extend their deployment from the cloud to edge devices. However, deploying LLMs on resource-constrained edge devices poses significant challenges, including (1) intensive computations and huge model sizes, (2) great memory and bandwidth demands introduced by the autoregressive generation process, and (3) limited scalability for handling long sequences. To address these challenges, we propose AccLLM, a comprehensive acceleration framework that enables efficient and fast long-context LLM inference through algorithm and hardware co-design. At the algorithmic level, we integrate (1) pruning, (2) {\Lambda}-shaped attention, and (3) an innovative W2A8KV4 (2-bit weights, 8-bit activations, and 4-bit KV cache) quantization scheme, thus effectively reducing memory and bandwidth requirements while facilitating LLMs' long-sequence generation. At the hardware level, we design a dedicated FPGA-based accelerator with a reconfigurable computing engine to effectively and flexibly accommodate diverse operations arising from our compression algorithm, thereby fully translating the algorithmic innovations into tangible hardware efficiency. We validate AccLLM on the Xilinx Alveo U280 FPGA, demonstrating a 4.07x energy efficiency and a 2.98x throughput compared to the state-of-the-art work FlightLLM. 

**Abstract (ZH)**: 近期，大规模语言模型（LLMs）在自然语言处理（NLP）领域取得了巨大成功，推动了它们从云端向边缘设备部署的需求增长。然而，在资源受限的边缘设备上部署LLMs带来了显著挑战，包括（1）密集的计算和庞大的模型规模，（2）自回归生成过程中引入的巨大内存和带宽需求，以及（3）处理长序列时的有限扩展性。为应对这些挑战，我们提出了一种全面的加速框架AccLLM，通过算法和硬件协同设计实现了高效且快速的长上下文LLM推理。在算法层面，我们整合了（1）剪枝，（2）Λ形注意机制，以及（3）一种创新的W2A8KV4（2位权重，8位激活和4位KV缓存）量化方案，从而有效地降低了内存和带宽需求，促进了LLMs长序列生成能力的增强。在硬件层面，我们设计了一种专用的基于FPGA的加速器，配备可重构计算引擎，能够有效且灵活地适应来自我们压缩算法的各种操作，从而全面将算法创新转化为实际的硬件效率。我们在Xilinx Alveo U280 FPGA上验证了AccLLM，相比于当前最先进的工作FlightLLM，展示了4.07倍的能量效率和2.98倍的吞吐量。 

---
# Beyond Misinformation: A Conceptual Framework for Studying AI Hallucinations in (Science) Communication 

**Title (ZH)**: 超越 misinformation：研究科学传播中 AI 幻觉的 conceptual framework 

**Authors**: Anqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.13777)  

**Abstract**: This paper proposes a conceptual framework for understanding AI hallucinations as a distinct form of misinformation. While misinformation scholarship has traditionally focused on human intent, generative AI systems now produce false yet plausible outputs absent of such intent. I argue that these AI hallucinations should not be treated merely as technical failures but as communication phenomena with social consequences. Drawing on a supply-and-demand model and the concept of distributed agency, the framework outlines how hallucinations differ from human-generated misinformation in production, perception, and institutional response. I conclude by outlining a research agenda for communication scholars to investigate the emergence, dissemination, and audience reception of hallucinated content, with attention to macro (institutional), meso (group), and micro (individual) levels. This work urges communication researchers to rethink the boundaries of misinformation theory in light of probabilistic, non-human actors increasingly embedded in knowledge production. 

**Abstract (ZH)**: 本文提出了一种概念框架，用于理解AI幻觉作为 misinformation 的一种独特形式。传统 misinformation 研究侧重于人类意图，而生成型 AI 系统现在可以产生缺乏此类意图的虚假但可信的输出。我认为，这些 AI 幻觉不应仅仅被视为技术故障，而应视为具有社会后果的沟通现象。基于供需模型和分散式代理的概念，该框架阐述了幻觉在生产、感知和机构应对方面的差异。最后，本文提出了一个研究议程，建议沟通学者调查幻觉内容的产生、传播和受众接受情况，关注宏观（机构）、中观（群体）和微观（个体）三个层面。本文呼吁沟通研究者重新思考 misinformation 理论的边界，以应对日益嵌入知识生产过程中的概率性非人类行为体。 

---
