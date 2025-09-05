# ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory 

**Title (ZH)**: ArcMemo: 抽象推理与终身大语言模型记忆的组成 

**Authors**: Matthew Ho, Chen Si, Zhaoxiang Feng, Fangxu Yu, Zhijian Liu, Zhiting Hu, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04439)  

**Abstract**: While inference-time scaling enables LLMs to carry out increasingly long and capable reasoning traces, the patterns and insights uncovered during these traces are immediately discarded once the context window is reset for a new query. External memory is a natural way to persist these discoveries, and recent work has shown clear benefits for reasoning-intensive tasks. We see an opportunity to make such memories more broadly reusable and scalable by moving beyond instance-based memory entries (e.g. exact query/response pairs, or summaries tightly coupled with the original problem context) toward concept-level memory: reusable, modular abstractions distilled from solution traces and stored in natural language. For future queries, relevant concepts are selectively retrieved and integrated into the prompt, enabling test-time continual learning without weight updates. Our design introduces new strategies for abstracting takeaways from rollouts and retrieving entries for new queries, promoting reuse and allowing memory to expand with additional experiences. On the challenging ARC-AGI benchmark, our method yields a 7.5% relative gain over a strong no-memory baseline with performance continuing to scale with inference compute. We find abstract concepts to be the most consistent memory design, outscoring the baseline at all tested inference compute scales. Moreover, we confirm that dynamically updating memory during test-time outperforms an otherwise identical fixed memory setting with additional attempts, supporting the hypothesis that solving more problems and abstracting more patterns to memory enables further solutions in a form of self-improvement. Code available at this https URL. 

**Abstract (ZH)**: 在推理时长扩展的同时保留发现的知识：概念级记忆赋能LLM持续学习和自我改进 

---
# Psychologically Enhanced AI Agents 

**Title (ZH)**: 心理增强型人工智能代理 

**Authors**: Maciej Besta, Shriram Chandran, Robert Gerstenberger, Mathis Lindner, Marcin Chrapek, Sebastian Hermann Martschat, Taraneh Ghandi, Patrick Iff, Hubert Niewiadomski, Piotr Nyczyk, Jürgen Müller, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2509.04343)  

**Abstract**: We introduce MBTI-in-Thoughts, a framework for enhancing the effectiveness of Large Language Model (LLM) agents through psychologically grounded personality conditioning. Drawing on the Myers-Briggs Type Indicator (MBTI), our method primes agents with distinct personality archetypes via prompt engineering, enabling control over behavior along two foundational axes of human psychology, cognition and affect. We show that such personality priming yields consistent, interpretable behavioral biases across diverse tasks: emotionally expressive agents excel in narrative generation, while analytically primed agents adopt more stable strategies in game-theoretic settings. Our framework supports experimenting with structured multi-agent communication protocols and reveals that self-reflection prior to interaction improves cooperation and reasoning quality. To ensure trait persistence, we integrate the official 16Personalities test for automated verification. While our focus is on MBTI, we show that our approach generalizes seamlessly to other psychological frameworks such as Big Five, HEXACO, or Enneagram. By bridging psychological theory and LLM behavior design, we establish a foundation for psychologically enhanced AI agents without any fine-tuning. 

**Abstract (ZH)**: 基于MBTI的心理定向框架：通过 personality conditioning 提升大型语言模型代理的有效性 

---
# Improving Robustness of AlphaZero Algorithms to Test-Time Environment Changes 

**Title (ZH)**: 改进AlphaZero算法在测试时环境变化下的鲁棒性 

**Authors**: Isidoro Tamassia, Wendelin Böhmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.04317)  

**Abstract**: The AlphaZero framework provides a standard way of combining Monte Carlo planning with prior knowledge provided by a previously trained policy-value neural network. AlphaZero usually assumes that the environment on which the neural network was trained will not change at test time, which constrains its applicability. In this paper, we analyze the problem of deploying AlphaZero agents in potentially changed test environments and demonstrate how the combination of simple modifications to the standard framework can significantly boost performance, even in settings with a low planning budget available. The code is publicly available on GitHub. 

**Abstract (ZH)**: AlphaZero框架提供了一种将蒙特卡洛规划与先前训练的策略-值神经网络提供的先验知识相结合的标准方法。AlphaZero通常假设在测试时神经网络训练的环境不会发生变化，这限制了它的适用性。本文分析了在潜在改变的测试环境中部署AlphaZero代理的问题，并展示了通过对标准框架进行简单修改可以显著提升性能，即使在可用规划预算较低的情况下也是如此。代码已在GitHub上公开。 

---
# EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation 

**Title (ZH)**: EvoEmo: 向往用于多轮谈判的LLM代理的情感策略演化 

**Authors**: Yunbo Long, Liming Xu, Lukas Beckenbauer, Yuhan Liu, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2509.04310)  

**Abstract**: Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation. 

**Abstract (ZH)**: Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in 复杂的、多轮的协商，开辟了代理型人工智能的新途径。然而，现有的LLM代理在这样的协商中很大程度上忽视了情绪的功能性作用，反而生成了被动的、基于偏好的情绪反应，使其容易受到对手方操纵和战略性利用。为解决这一问题，我们提出了EvoEmo，一种进化的强化学习框架，以优化协商中的动态情绪表达。EvoEmo将情绪状态转换建模为马尔可夫决策过程，并采用基于群体的遗传优化来进化出适用于多种协商场景的高奖励情绪策略。我们进一步提出了一种包含两种基线——常规策略和固定情绪策略——的评估框架，用于情绪意识协商的基准测试。广泛的实验和消融研究显示，EvoEmo在成功率、效率和买家节省方面均优于基线策略。这些发现突显了适应性情绪表达在使多轮协商中的人工智能代理更具效用方面的重要性。 

---
# Evaluating Quality of Gaming Narratives Co-created with AI 

**Title (ZH)**: 评估AI共创的游戏叙事质量 

**Authors**: Arturo Valdivia, Paolo Burelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.04239)  

**Abstract**: This paper proposes a structured methodology to evaluate AI-generated game narratives, leveraging the Delphi study structure with a panel of narrative design experts. Our approach synthesizes story quality dimensions from literature and expert insights, mapping them into the Kano model framework to understand their impact on player satisfaction. The results can inform game developers on prioritizing quality aspects when co-creating game narratives with generative AI. 

**Abstract (ZH)**: 本文提出了一种结构化的方法来评估AI生成的游戏叙事，利用德尔菲研究结构和叙事设计专家小组。我们的方法综合了文献和专家见解中的故事质量维度，并将它们映射到Kano模型框架中，以了解这些维度对玩家满意度的影响。研究结果可以指导游戏开发者在与生成式AI共创造游戏叙事时优先考虑质量方面。 

---
# Domain size asymptotics for Markov logic networks 

**Title (ZH)**: 马尔可夫逻辑网络的域大小渐近性质 

**Authors**: Vera Koponen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04192)  

**Abstract**: A Markov logic network (MLN) determines a probability distribution on the set of structures, or ``possible worlds'', with an arbitrary finite domain. We study the properties of such distributions as the domain size tends to infinity. Three types of concrete examples of MLNs will be considered, and the properties of random structures with domain sizes tending to infinity will be studied: (1) Arbitrary quantifier-free MLNs over a language with only one relation symbol which has arity 1. In this case we give a pretty complete characterization of the possible limit behaviours of random structures. (2) An MLN that favours graphs with fewer triangles (or more generally, fewer k-cliques). As a corollary of the analysis a ``$\delta$-approximate 0-1 law'' for first-order logic is obtained. (3) An MLN that favours graphs with fewer vertices with degree higher than a fixed (but arbitrary) number. The analysis shows that depending on which ``soft constraints'' an MLN uses the limit behaviour of random structures can be quite different, and the weights of the soft constraints may, or may not, have influence on the limit behaviour. It will also be demonstrated, using (1), that quantifier-free MLNs and lifted Bayesian networks (in a broad sense) are asymptotically incomparable, roughly meaning that there is a sequence of distributions on possible worlds with increasing domain sizes that can be defined by one of the formalisms but not even approximated by the other. In a rather general context it is also shown that on large domains the distribution determined by an MLN concentrates almost all its probability mass on a totally different part of the space of possible worlds than the uniform distribution does. 

**Abstract (ZH)**: 一个马尔可夫逻辑网络（MLN）确定了结构集或“可能世界”的概率分布，其领域是任意有限域。我们研究当领域大小趋于无穷时这类分布的性质。将考虑三种具体的MLN实例，并研究当领域大小趋于无穷时随机结构的性质：（1）仅含一个一元关系符号的任意量化公式为零的MLN。在这种情况下，我们提供了随机结构可能极限行为的相当完整的表征。（2）一个倾向于包含更少三角形（或更一般地，更少的k- clique）的MLN。分析的推论给出了零-1法律的一个“δ-近似”结果。（3）一个倾向于包含更少度数高于固定数（但任意）的顶点的MLN。分析表明，根据MLN使用的“软约束”类型，随机结构的极限行为可能大不相同，而“软约束”的权重可能会影响也可能不会影响极限行为。还将利用（1）证明，量化公式为零的MLN和提升的贝叶斯网络（广义上）在无穷大极限下是渐近不可比较的，大致意味着存在定义在可能世界上的分布序列，随领域大小增加，可以由其中一种形式化表示定义，甚至不能被另一种形式的表示近似。在相对一般的情境下，也证明在大型领域中，MLN确定的分布几乎将所有概率质量集中在可能世界空间的完全不同部分，而均匀分布则不会如此。 

---
# Towards an Action-Centric Ontology for Cooking Procedures Using Temporal Graphs 

**Title (ZH)**: 基于时间图的烹饪程序以动作为中心的本体研究 

**Authors**: Aarush Kumbhakern, Saransh Kumar Gupta, Lipika Dey, Partha Pratim Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.04159)  

**Abstract**: Formalizing cooking procedures remains a challenging task due to their inherent complexity and ambiguity. We introduce an extensible domain-specific language for representing recipes as directed action graphs, capturing processes, transfers, environments, concurrency, and compositional structure. Our approach enables precise, modular modeling of complex culinary workflows. Initial manual evaluation on a full English breakfast recipe demonstrates the DSL's expressiveness and suitability for future automated recipe analysis and execution. This work represents initial steps towards an action-centric ontology for cooking, using temporal graphs to enable structured machine understanding, precise interpretation, and scalable automation of culinary processes - both in home kitchens and professional culinary settings. 

**Abstract (ZH)**: 形式化烹饪程序仍然是一个具有挑战性的问题，由于其固有的复杂性和模糊性。我们引入了一种可扩展的领域特定语言，用于将食谱表示为有向动作图，捕捉过程、转移、环境、并发性和组合结构。我们的方法使复杂的烹饪工作流程精准且模块化地建模成为可能。对一份完整的英式早餐食谱的手动评估初步展示了该领域特定语言的表达能力和适合未来自动化食谱分析和执行的潜力。这项工作代表了向以动作为中心的烹饪本体论发展的初步步骤，利用时间图来实现烹饪过程的结构化机器理解、精确解释和可扩展自动化——无论是在家庭厨房还是在专业烹饪环境中。 

---
# The human biological advantage over AI 

**Title (ZH)**: 人类在生物智能方面超过AI的优势 

**Authors**: William Stewart  

**Link**: [PDF](https://arxiv.org/pdf/2509.04130)  

**Abstract**: Recent advances in AI raise the possibility that AI systems will one day be able to do anything humans can do, only better. If artificial general intelligence (AGI) is achieved, AI systems may be able to understand, reason, problem solve, create, and evolve at a level and speed that humans will increasingly be unable to match, or even understand. These possibilities raise a natural question as to whether AI will eventually become superior to humans, a successor "digital species", with a rightful claim to assume leadership of the universe. However, a deeper consideration suggests the overlooked differentiator between human beings and AI is not the brain, but the central nervous system (CNS), providing us with an immersive integration with physical reality. It is our CNS that enables us to experience emotion including pain, joy, suffering, and love, and therefore to fully appreciate the consequences of our actions on the world around us. And that emotional understanding of the consequences of our actions is what is required to be able to develop sustainable ethical systems, and so be fully qualified to be the leaders of the universe. A CNS cannot be manufactured or simulated; it must be grown as a biological construct. And so, even the development of consciousness will not be sufficient to make AI systems superior to humans. AI systems may become more capable than humans on almost every measure and transform our society. However, the best foundation for leadership of our universe will always be DNA, not silicon. 

**Abstract (ZH)**: 近期AI的发展使得AI系统未来可能在任何人类能做到的事情上做得更好。如果实现人工通用智能（AGI），AI系统可能在理解和推理、问题解决、创造和进化方面达到一个水平和速度，使人类越来越难以跟上甚至无法理解。这些可能性引发了一个自然的问题：AI最终是否会超越人类，成为一种替代的“数字物种”，有资格领导宇宙。然而，更深层次的考虑表明，人类与AI之间被忽视的区别不是大脑，而是中枢神经系统（CNS），它使我们能够与物理现实进行沉浸式的整合。正是我们的CNS让我们能够体验包括痛苦、快乐、苦难和爱在内的情感，从而能够充分理解我们的行为对周围世界的影响。这种对行为后果的情感理解是建立可持续伦理体系所需要的，因此使我们有资格成为宇宙的领导者。中枢神经系统无法被制造或模拟；必须作为一种生物构造来生长。因此，即使意识的产生也不足以使AI系统超越人类。AI系统可能在几乎每一个衡量标准上变得比人类更有能力，并且将改变我们的社会。然而，宇宙的领导基础永远是DNA，而不是硅。 

---
# Analysis of Bluffing by DQN and CFR in Leduc Hold'em Poker 

**Title (ZH)**: DQN和CFR在Leduc Hold'em扑克中撒牌分析 

**Authors**: Tarik Zaciragic, Aske Plaat, K. Joost Batenburg  

**Link**: [PDF](https://arxiv.org/pdf/2509.04125)  

**Abstract**: In the game of poker, being unpredictable, or bluffing, is an essential skill. When humans play poker, they bluff. However, most works on computer-poker focus on performance metrics such as win rates, while bluffing is overlooked. In this paper we study whether two popular algorithms, DQN (based on reinforcement learning) and CFR (based on game theory), exhibit bluffing behavior in Leduc Hold'em, a simplified version of poker. We designed an experiment where we let the DQN and CFR agent play against each other while we log their actions. We find that both DQN and CFR exhibit bluffing behavior, but they do so in different ways. Although both attempt to perform bluffs at different rates, the percentage of successful bluffs (where the opponent folds) is roughly the same. This suggests that bluffing is an essential aspect of the game, not of the algorithm. Future work should look at different bluffing styles and at the full game of poker. Code at this https URL. 

**Abstract (ZH)**: 在德州扑克游戏中，不可预测性或欺骗是一种关键技能。在人类玩德州扑克时，他们会进行欺骗。然而，大多数关于计算机德州扑克的研究集中在胜率等性能指标上，而忽略了欺骗行为。本文研究了两种流行的算法，基于强化学习的DQN和基于博弈论的CFR，在简化版本的德鲁克霍尔得'em (Leduc Hold'em) 中是否表现出欺骗行为。我们设计了一个实验，让DQN和CFR代理相互对战，并记录它们的行为。我们发现，DQN和CFR都表现出欺骗行为，但它们的方式不同。尽管它们以不同的频率尝试欺骗，成功欺骗的比例（对手弃牌）大致相同。这表明欺骗是游戏的关键部分，而不是算法的关键部分。未来的工作应研究不同类型的欺骗和完整的德州扑克游戏。代码详见此链接：https://link.url。 

---
# Hybrid Reinforcement Learning and Search for Flight Trajectory Planning 

**Title (ZH)**: 混合强化学习与搜索在飞行航迹规划中的应用 

**Authors**: Alberto Luise, Michele Lombardi, Florent Teichteil Koenigsbuch  

**Link**: [PDF](https://arxiv.org/pdf/2509.04100)  

**Abstract**: This paper explores the combination of Reinforcement Learning (RL) and search-based path planners to speed up the optimization of flight paths for airliners, where in case of emergency a fast route re-calculation can be crucial. The fundamental idea is to train an RL Agent to pre-compute near-optimal paths based on location and atmospheric data and use those at runtime to constrain the underlying path planning solver and find a solution within a certain distance from the initial guess. The approach effectively reduces the size of the solver's search space, significantly speeding up route optimization. Although global optimality is not guaranteed, empirical results conducted with Airbus aircraft's performance models show that fuel consumption remains nearly identical to that of an unconstrained solver, with deviations typically within 1%. At the same time, computation speed can be improved by up to 50% as compared to using a conventional solver alone. 

**Abstract (ZH)**: 基于强化学习和搜索路径规划的航空器航路优化加速方法 

---
# Intermediate Languages Matter: Formal Languages and LLMs affect Neurosymbolic Reasoning 

**Title (ZH)**: 中间语言很重要：形式语言和LLMs对神经符号推理的影响 

**Authors**: Alexander Beiser, David Penz, Nysret Musliu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04083)  

**Abstract**: Large language models (LLMs) achieve astonishing results on a wide range of tasks. However, their formal reasoning ability still lags behind. A promising approach is Neurosymbolic LLM reasoning. It works by using LLMs as translators from natural to formal languages and symbolic solvers for deriving correct results. Still, the contributing factors to the success of Neurosymbolic LLM reasoning remain unclear. This paper demonstrates that one previously overlooked factor is the choice of the formal language. We introduce the intermediate language challenge: selecting a suitable formal language for neurosymbolic reasoning. By comparing four formal languages across three datasets and seven LLMs, we show that the choice of formal language affects both syntactic and semantic reasoning capabilities. We also discuss the varying effects across different LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的任务上取得了惊人的成果，但其形式化的推理能力依然落后。一种有希望的方法是神经符号LLM推理。它通过使用LLM作为自然语言和形式语言之间的翻译器，并使用符号求解器来得出正确结果。然而，神经符号LLM推理成功的原因尚不明确。本文表明，一个以前被忽略的因素是形式语言的选择。我们引入了中间语言挑战：为神经符号推理选择合适的正式语言。通过在三个数据集和七种LLM上比较四种形式语言，我们证明了形式语言的选择影响了语法和语义推理能力。我们还讨论了不同LLM之间不同的影响效果。 

---
# Oruga: An Avatar of Representational Systems Theory 

**Title (ZH)**: Oruga：表征系统理论的化身 

**Authors**: Daniel Raggi, Gem Stapleton, Mateja Jamnik, Aaron Stockdill, Grecia Garcia Garcia, Peter C-H. Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.04041)  

**Abstract**: Humans use representations flexibly. We draw diagrams, change representations and exploit creative analogies across different domains. We want to harness this kind of power and endow machines with it to make them more compatible with human use. Previously we developed Representational Systems Theory (RST) to study the structure and transformations of representations. In this paper we present Oruga (caterpillar in Spanish; a symbol of transformation), an implementation of various aspects of RST. Oruga consists of a core of data structures corresponding to concepts in RST, a language for communicating with the core, and an engine for producing transformations using a method we call structure transfer. In this paper we present an overview of the core and language of Oruga, with a brief example of the kind of transformation that structure transfer can execute. 

**Abstract (ZH)**: 人类灵活使用表征。我们绘制图表、改变表征形式并跨不同领域运用富有创意的类比。我们希望利用这种能力，赋予机器这种能力，使它们更符合人类的使用方式。我们之前开发了表征系统理论（RST）以研究表征的结构和变换。在本文中，我们介绍了Oruga（西班牙语中的毛毛虫；象征变化的一种），RST各方面的一个实现。Oruga包含与RST概念对应的中心数据结构、与中心通信的语言，以及使用我们称之为结构转移的方法生成变换的引擎。本文概述了Oruga的中心和语言，并简要介绍了结构转移所能执行的变换类型的一个示例。 

---
# CoT-Space: A Theoretical Framework for Internal Slow-Thinking via Reinforcement Learning 

**Title (ZH)**: CoT-空间：通过强化学习实现内在慢思考的理论框架 

**Authors**: Zeyu Gan, Hao Yi, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04027)  

**Abstract**: Reinforcement Learning (RL) has become a pivotal approach for enhancing the reasoning capabilities of Large Language Models (LLMs). However, a significant theoretical gap persists, as traditional token-level RL frameworks fail to align with the reasoning-level nature of complex, multi-step thought processes like Chain-of-Thought (CoT). To address this challenge, we introduce CoT-Space, a novel theoretical framework that recasts LLM reasoning from a discrete token-prediction task to an optimization process within a continuous, reasoning-level semantic space. By analyzing this process from both a noise perspective and a risk perspective, we demonstrate that the convergence to an optimal CoT length is a natural consequence of the fundamental trade-off between underfitting and overfitting. Furthermore, extensive experiments provide strong empirical validation for our theoretical findings. Our framework not only provides a coherent explanation for empirical phenomena such as overthinking but also offers a solid theoretical foundation to guide the future development of more effective and generalizable reasoning agents. 

**Abstract (ZH)**: 强化学习（RL）已成为增强大规模语言模型（LLMs）推理能力的关键方法。然而，传统的基于token的RL框架未能与复杂多步推理过程如Chain-of-Thought（CoT）的推理级本质相契合，仍存在重要的理论缺口。为解决这一挑战，我们提出了CoT-Space这一新颖的理论框架，将LLM的推理从离散的token预测任务重新构想为在连续的、推理级语义空间内的优化过程。通过从噪声角度和风险角度分析这一过程，我们证明了向最优CoT长度收敛是过度拟合与欠拟合基本权衡的自然结果。此外，广泛的实验证明了我们理论发现的坚实 empirical 依据。我们的框架不仅为过度思考等经验现象提供了连贯的解释，还为未来的推理代理的有效性和泛化能力的提升提供了坚实的理论基础。 

---
# AutoPBO: LLM-powered Optimization for Local Search PBO Solvers 

**Title (ZH)**: AutoPBO：基于LLM的局部搜索PBO求解器优化 

**Authors**: Jinyuan Li, Yi Chu, Yiwen Sun, Mengchuan Zou, Shaowei Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.04007)  

**Abstract**: Pseudo-Boolean Optimization (PBO) provides a powerful framework for modeling combinatorial problems through pseudo-Boolean (PB) constraints. Local search solvers have shown excellent performance in PBO solving, and their efficiency is highly dependent on their internal heuristics to guide the search. Still, their design often requires significant expert effort and manual tuning in practice. While Large Language Models (LLMs) have demonstrated potential in automating algorithm design, their application to optimizing PBO solvers remains unexplored. In this work, we introduce AutoPBO, a novel LLM-powered framework to automatically enhance PBO local search solvers. We conduct experiments on a broad range of four public benchmarks, including one real-world benchmark, a benchmark from PB competition, an integer linear programming optimization benchmark, and a crafted combinatorial benchmark, to evaluate the performance improvement achieved by AutoPBO and compare it with six state-of-the-art competitors, including two local search PBO solvers NuPBO and OraSLS, two complete PB solvers PBO-IHS and RoundingSat, and two mixed integer programming (MIP) solvers Gurobi and SCIP. AutoPBO demonstrates significant improvements over previous local search approaches, while maintaining competitive performance compared to state-of-the-art competitors. The results suggest that AutoPBO offers a promising approach to automating local search solver design. 

**Abstract (ZH)**: 基于伪布尔优化的自动增强框架：AutoPBO 

---
# Meta-Policy Reflexion: Reusable Reflective Memory and Rule Admissibility for Resource-Efficient LLM Agent 

**Title (ZH)**: 元策略反思：可重用的反思记忆与资源高效LLM代理的规则可接纳性 

**Authors**: Chunlong Wu, Zhibo Qu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03990)  

**Abstract**: Large language model (LLM) agents achieve impressive single-task performance but commonly exhibit repeated failures, inefficient exploration, and limited cross-task adaptability. Existing reflective strategies (e.g., Reflexion, ReAct) improve per-episode behavior but typically produce ephemeral, task-specific traces that are not reused across tasks. Reinforcement-learning based alternatives can produce transferable policies but require substantial parameter updates and compute. In this work we introduce Meta-Policy Reflexion (MPR): a hybrid framework that consolidates LLM-generated reflections into a structured, predicate-like Meta-Policy Memory (MPM) and applies that memory at inference time through two complementary mechanisms soft memory-guided decoding and hard rule admissibility checks(HAC). MPR (i) externalizes reusable corrective knowledge without model weight updates, (ii) enforces domain constraints to reduce unsafe or invalid actions, and (iii) retains the adaptability of language-based reflection. We formalize the MPM representation, present algorithms for update and decoding, and validate the approach in a text-based agent environment following the experimental protocol described in the provided implementation (AlfWorld-based). Empirical results reported in the supplied material indicate consistent gains in execution accuracy and robustness when compared to Reflexion baselines; rule admissibility further improves stability. We analyze mechanisms that explain these gains, discuss scalability and failure modes, and outline future directions for multimodal and multi?agent extensions. 

**Abstract (ZH)**: 大型语言模型代理在单任务上取得显著性能，但通常表现出重复的失败、不高效的探索和有限的跨任务适应性。现有的反思策略（例如，Reflexion、ReAct）能够提升每轮行为，但通常生成的只是临时的任务特定踪迹，无法在不同任务间重用。基于强化学习的替代方案可以产生可转移的策略，但需要大量的参数更新和计算资源。在此项工作中，我们引入了元策略反思（MPR）：这是一种混合框架，将LLM生成的反思整合成一个结构化的、类似谓词的元策略记忆（MPM），并通过软记忆引导解码和硬规则可接纳性检查（HAC）在推理时应用这种方法记忆。（i）外部化可重用的纠正性知识而不更新模型权重，（ii）施加域约束以减少不安全或无效动作，（iii）保留基于语言的反思的适应性。我们形式化了MPM表示，提供了更新和解码算法，并按照提供的实现（基于AlfWorld的）实验协议验证了该方法。所提供材料中的实验证据显示，与Reflexion基线相比，MPR在执行准确性和稳健性方面保持了一致的改进；硬规则可接纳性进一步提高了稳定性。我们分析了解释这些改进的机制，讨论了扩展性和失败模式，并概述了用于多模态和多代理扩展的未来方向。 

---
# World Model Implanting for Test-time Adaptation of Embodied Agents 

**Title (ZH)**: 基于世界模型的体 Agent 测试时适应植入 

**Authors**: Minjong Yoo, Jinwoo Jang, Sihyung Yoon, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2509.03956)  

**Abstract**: In embodied AI, a persistent challenge is enabling agents to robustly adapt to novel domains without requiring extensive data collection or retraining. To address this, we present a world model implanting framework (WorMI) that combines the reasoning capabilities of large language models (LLMs) with independently learned, domain-specific world models through test-time composition. By allowing seamless implantation and removal of the world models, the embodied agent's policy achieves and maintains cross-domain adaptability. In the WorMI framework, we employ a prototype-based world model retrieval approach, utilizing efficient trajectory-based abstract representation matching, to incorporate relevant models into test-time composition. We also develop a world-wise compound attention method that not only integrates the knowledge from the retrieved world models but also aligns their intermediate representations with the reasoning model's representation within the agent's policy. This framework design effectively fuses domain-specific knowledge from multiple world models, ensuring robust adaptation to unseen domains. We evaluate our WorMI on the VirtualHome and ALFWorld benchmarks, demonstrating superior zero-shot and few-shot performance compared to several LLM-based approaches across a range of unseen domains. These results highlight the frameworks potential for scalable, real-world deployment in embodied agent scenarios where adaptability and data efficiency are essential. 

**Abstract (ZH)**: 在具身AI中，一个持续的挑战是使代理能够 robustly 而不过分依赖大规模数据收集或重新训练的情况下适应新的领域。为了应对这一挑战，我们提出了一种世界模型植入框架（WorMI），该框架结合了大型语言模型的推理能力与独立学习的领域特定世界模型，通过测试时的组合实现。通过允许世界模型的无缝植入和移除，具身代理的策略实现了并保持了跨领域的适应性。在WorMI框架中，我们采用基于原型的世界模型检索方法，利用高效的基于轨迹的抽象表示匹配，将相关模型纳入测试时的组合。我们还开发了一种世界导向的复合注意力方法，不仅整合了检索到的世界模型的知识，还使它们的中间表示与代理策略中的推理模型表示进行对齐。该框架设计有效地融合了多个世界模型的领域特定知识，确保在未见过的领域中的稳健适应。我们在VirtualHome和ALFWorld基准上评估了我们的WorMI，展示了优于多种基于大型语言模型的方法的零样本和少样本性能，适用于一系列未见过的领域。这些结果突显了该框架在需要适应性与数据效率的具身代理场景中的潜在 scalability 和实际部署能力。 

---
# Handling Infinite Domain Parameters in Planning Through Best-First Search with Delayed Partial Expansions 

**Title (ZH)**: 通过延迟部分扩展的最好首先搜索处理规划中的无限域参数 

**Authors**: Ángel Aso-Mollar, Diego Aineto, Enrico Scala, Eva Onaindia  

**Link**: [PDF](https://arxiv.org/pdf/2509.03953)  

**Abstract**: In automated planning, control parameters extend standard action representations through the introduction of continuous numeric decision variables. Existing state-of-the-art approaches have primarily handled control parameters as embedded constraints alongside other temporal and numeric restrictions, and thus have implicitly treated them as additional constraints rather than as decision points in the search space. In this paper, we propose an efficient alternative that explicitly handles control parameters as true decision points within a systematic search scheme. We develop a best-first, heuristic search algorithm that operates over infinite decision spaces defined by control parameters and prove a notion of completeness in the limit under certain conditions. Our algorithm leverages the concept of delayed partial expansion, where a state is not fully expanded but instead incrementally expands a subset of its successors. Our results demonstrate that this novel search algorithm is a competitive alternative to existing approaches for solving planning problems involving control parameters. 

**Abstract (ZH)**: 自动化规划中，控制参数通过引入连续数值决策变量扩展标准动作表示。现有最先进的方法主要将控制参数作为与其他时间和数值限制嵌入的约束来处理，从而在一定程度上将它们隐式地视为搜索空间中的额外约束而非决策点。本文提出了一种有效的替代方案，该方案在系统搜索方案中明确处理控制参数作为真正的决策点。我们开发了一种最佳优先启发式搜索算法，在由控制参数定义的无限决策空间上进行操作，并在某些条件下证明了完备性。该算法利用了延迟部分扩展的概念，其中状态不是完全展开，而是增量地展开其部分后继状态。我们的结果表明，这种新颖的搜索算法是解决涉及控制参数的规划问题的有效替代方案。 

---
# A Foundation Model for Chest X-ray Interpretation with Grounded Reasoning via Online Reinforcement Learning 

**Title (ZH)**: 基于 grounded reasoning 通过在线强化学习进行胸片解释的foundation模型 

**Authors**: Qika Lin, Yifan Zhu, Bin Pu, Ling Huang, Haoran Luo, Jingying Ma, Zhen Peng, Tianzhe Zhao, Fangzhi Xu, Jian Zhang, Kai He, Zhonghong Ou, Swapnil Mishra, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.03906)  

**Abstract**: Medical foundation models (FMs) have shown tremendous promise amid the rapid advancements in artificial intelligence (AI) technologies. However, current medical FMs typically generate answers in a black-box manner, lacking transparent reasoning processes and locally grounded interpretability, which hinders their practical clinical deployments. To this end, we introduce DeepMedix-R1, a holistic medical FM for chest X-ray (CXR) interpretation. It leverages a sequential training pipeline: initially fine-tuned on curated CXR instruction data to equip with fundamental CXR interpretation capabilities, then exposed to high-quality synthetic reasoning samples to enable cold-start reasoning, and finally refined via online reinforcement learning to enhance both grounded reasoning quality and generation performance. Thus, the model produces both an answer and reasoning steps tied to the image's local regions for each query. Quantitative evaluation demonstrates substantial improvements in report generation (e.g., 14.54% and 31.32% over LLaVA-Rad and MedGemma) and visual question answering (e.g., 57.75% and 23.06% over MedGemma and CheXagent) tasks. To facilitate robust assessment, we propose Report Arena, a benchmarking framework using advanced language models to evaluate answer quality, further highlighting the superiority of DeepMedix-R1. Expert review of generated reasoning steps reveals greater interpretability and clinical plausibility compared to the established Qwen2.5-VL-7B model (0.7416 vs. 0.2584 overall preference). Collectively, our work advances medical FM development toward holistic, transparent, and clinically actionable modeling for CXR interpretation. 

**Abstract (ZH)**: 基于深度 Medix-R1 在胸部 X 光片解释中的整体医疗基础模型：面向透明和临床实用的解释推理 

---
# FaMA: LLM-Empowered Agentic Assistant for Consumer-to-Consumer Marketplace 

**Title (ZH)**: FaMA: LLM赋能的消费者代理助手for消费者对消费者 marketplace 

**Authors**: Yineng Yan, Xidong Wang, Jin Seng Cheng, Ran Hu, Wentao Guan, Nahid Farahmand, Hengte Lin, Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.03890)  

**Abstract**: The emergence of agentic AI, powered by Large Language Models (LLMs), marks a paradigm shift from reactive generative systems to proactive, goal-oriented autonomous agents capable of sophisticated planning, memory, and tool use. This evolution presents a novel opportunity to address long-standing challenges in complex digital environments. Core tasks on Consumer-to-Consumer (C2C) e-commerce platforms often require users to navigate complex Graphical User Interfaces (GUIs), making the experience time-consuming for both buyers and sellers. This paper introduces a novel approach to simplify these interactions through an LLM-powered agentic assistant. This agent functions as a new, conversational entry point to the marketplace, shifting the primary interaction model from a complex GUI to an intuitive AI agent. By interpreting natural language commands, the agent automates key high-friction workflows. For sellers, this includes simplified updating and renewal of listings, and the ability to send bulk messages. For buyers, the agent facilitates a more efficient product discovery process through conversational search. We present the architecture for Facebook Marketplace Assistant (FaMA), arguing that this agentic, conversational paradigm provides a lightweight and more accessible alternative to traditional app interfaces, allowing users to manage their marketplace activities with greater efficiency. Experiments show FaMA achieves a 98% task success rate on solving complex tasks on the marketplace and enables up to a 2x speedup on interaction time. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的代理型AI的涌现标志着从反应性生成系统到主动、目标导向自主代理的范式转变，这些代理能够进行复杂的规划、记忆和工具使用。这一演变为解决复杂数字环境中长期存在的挑战提供了新的机会。在消费者对消费者（C2C）电子商务平台上，核心任务往往要求用户导航复杂的图形用户界面（GUI），这使得买卖双方的体验耗时。本文提出了一种通过基于LLMs的代理助手简化这些交互的新方法。该代理作为一个新的、对话式的市场入口点，改变了主要交互模式，从复杂的GUI到直观的AI代理。通过解释自然语言命令，代理代理自动化了高摩擦的工作流程。对于卖家，这包括简化更新和续订listing以及发送批量消息。对于买家，代理通过对话式搜索简化了产品发现过程。本文提出了Facebook Marketplace Assistant（FaMA）的架构，认为这种代理式的对话范式提供了比传统应用界面更轻量级且更易访问的选择，使用户能够更高效地管理市场活动。实验结果显示，FaMA在解决市场上的复杂任务时的成功率达到98%，并将交互时间提高了一倍。 

---
# Expedition & Expansion: Leveraging Semantic Representations for Goal-Directed Exploration in Continuous Cellular Automata 

**Title (ZH)**: 探索与扩展：利用语义表示进行连续细胞自动机中的目标导向探索 

**Authors**: Sina Khajehabdollahi, Gautier Hamon, Marko Cvjetko, Pierre-Yves Oudeyer, Clément Moulin-Frier, Cédric Colas  

**Link**: [PDF](https://arxiv.org/pdf/2509.03863)  

**Abstract**: Discovering diverse visual patterns in continuous cellular automata (CA) is challenging due to the vastness and redundancy of high-dimensional behavioral spaces. Traditional exploration methods like Novelty Search (NS) expand locally by mutating known novel solutions but often plateau when local novelty is exhausted, failing to reach distant, unexplored regions. We introduce Expedition and Expansion (E&E), a hybrid strategy where exploration alternates between local novelty-driven expansions and goal-directed expeditions. During expeditions, E&E leverages a Vision-Language Model (VLM) to generate linguistic goals--descriptions of interesting but hypothetical patterns that drive exploration toward uncharted regions. By operating in semantic spaces that align with human perception, E&E both evaluates novelty and generates goals in conceptually meaningful ways, enhancing the interpretability and relevance of discovered behaviors. Tested on Flow Lenia, a continuous CA known for its rich, emergent behaviors, E&E consistently uncovers more diverse solutions than existing exploration methods. A genealogical analysis further reveals that solutions originating from expeditions disproportionately influence long-term exploration, unlocking new behavioral niches that serve as stepping stones for subsequent search. These findings highlight E&E's capacity to break through local novelty boundaries and explore behavioral landscapes in human-aligned, interpretable ways, offering a promising template for open-ended exploration in artificial life and beyond. 

**Abstract (ZH)**: 探索连续细胞自动机中多样视觉模式的挑战在于高维行为空间的浩瀚和冗余性。传统的探索方法如新颖性搜索（NS）通过突变已知的新颖解决方案进行局部扩展，但在局部新颖性耗尽时往往会停滞，无法到达遥远的未探索区域。我们提出了探险与扩展（E&E）的混合策略，其中探索在局部新颖性驱动的扩展和目标导向的探险之间交替进行。在探险期间，E&E 利用视觉语言模型（VLM）生成语义目标——描述有趣但假设中的模式的描述，从而驱动探索向未开发区域前行。通过在与人类感知相一致的语义空间中操作，E&E 既能评估新颖性，又能以概念上有意义的方式生成目标，从而增强发现行为的可解释性和相关性。在 Flow Lenia 上测试，这是一种以其丰富的涌现行为而闻名的连续 CA，E&E 一致地发现了比现有探索方法更多样化的解决方案。谱系分析进一步表明，起源于探险的解决方案在长期探索中占主导地位，解锁了作为后续搜索阶梯的新行为生态位。这些发现突显了 E&E 有能力突破局部新颖性边界，并以与人类感知一致、可解释的方式探索行为景观，为人工生命及其相关的开放探索提供了有前景的模板。 

---
# Continuous Monitoring of Large-Scale Generative AI via Deterministic Knowledge Graph Structures 

**Title (ZH)**: 大规模生成式人工智能的确定性知识图结构持续监控 

**Authors**: Kishor Datta Gupta, Mohd Ariful Haque, Hasmot Ali, Marufa Kamal, Syed Bahauddin Alam, Mohammad Ashiqur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2509.03857)  

**Abstract**: Generative AI (GEN AI) models have revolutionized diverse application domains but present substantial challenges due to reliability concerns, including hallucinations, semantic drift, and inherent biases. These models typically operate as black-boxes, complicating transparent and objective evaluation. Current evaluation methods primarily depend on subjective human assessment, limiting scalability, transparency, and effectiveness. This research proposes a systematic methodology using deterministic and Large Language Model (LLM)-generated Knowledge Graphs (KGs) to continuously monitor and evaluate GEN AI reliability. We construct two parallel KGs: (i) a deterministic KG built using explicit rule-based methods, predefined ontologies, domain-specific dictionaries, and structured entity-relation extraction rules, and (ii) an LLM-generated KG dynamically derived from real-time textual data streams such as live news articles. Utilizing real-time news streams ensures authenticity, mitigates biases from repetitive training, and prevents adaptive LLMs from bypassing predefined benchmarks through feedback memorization. To quantify structural deviations and semantic discrepancies, we employ several established KG metrics, including Instantiated Class Ratio (ICR), Instantiated Property Ratio (IPR), and Class Instantiation (CI). An automated real-time monitoring framework continuously computes deviations between deterministic and LLM-generated KGs. By establishing dynamic anomaly thresholds based on historical structural metric distributions, our method proactively identifies and flags significant deviations, thus promptly detecting semantic anomalies or hallucinations. This structured, metric-driven comparison between deterministic and dynamically generated KGs delivers a robust and scalable evaluation framework. 

**Abstract (ZH)**: 生成式人工智能（GEN AI）模型已 revolutionized 多个应用领域，但由于可靠性问题，如幻觉、语义漂移和固有偏见，也带来了重大挑战。这些模型通常作为黑盒运作，使得透明和客观评估变得复杂。当前的评估方法主要依赖主观的人类评估，限制了其可扩展性、透明度和有效性。本研究提出了一种系统的方法，利用确定性和Large Language Model（LLM）生成的知识图谱（KGs）来连续监控和评估生成式人工智能的可靠性。我们构建了两个并行的知识图谱：（i）一个使用显式规则方法、预定义本体、领域专用词典和结构化的实体-关系提取规则构建的确定性KG；（ii）一个从实时文本数据流（如实时新闻文章）动态生成的LLM生成的KG。利用实时新闻流确保了真实性和减少重复训练带来的偏见，防止适应性LLM通过反馈记忆绕过预定义基准。为了量化结构偏差和语义差异，我们使用了几种已建立的知识图谱度量标准，包括实例类比例（ICR）、实例属性比例（IPR）和类实例化（CI）。该自动化实时监控框架持续计算确定性和LLM生成的KG之间的偏差。通过基于历史结构度量分布建立动态异常阈值，我们的方法能够主动识别并标记重要偏差，从而及时检测到语义异常或幻觉。这种结构化的、基于度量的确定性和动态生成的KG之间的比较提供了一个 robust 和可扩展的评估框架。 

---
# A Multidimensional AI-powered Framework for Analyzing Tourist Perception in Historic Urban Quarters: A Case Study in Shanghai 

**Title (ZH)**: 基于人工智能的多维度框架：分析历史文化街区游客感知——以上海为例的研究 

**Authors**: Kaizhen Tan, Yufan Wu, Yuxuan Liu, Haoran Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.03830)  

**Abstract**: Historic urban quarters play a vital role in preserving cultural heritage while serving as vibrant spaces for tourism and everyday life. Understanding how tourists perceive these environments is essential for sustainable, human-centered urban planning. This study proposes a multidimensional AI-powered framework for analyzing tourist perception in historic urban quarters using multimodal data from social media. Applied to twelve historic quarters in central Shanghai, the framework integrates focal point extraction, color theme analysis, and sentiment mining. Visual focus areas are identified from tourist-shared photos using a fine-tuned semantic segmentation model. To assess aesthetic preferences, dominant colors are extracted using a clustering method, and their spatial distribution across quarters is analyzed. Color themes are further compared between social media photos and real-world street views, revealing notable shifts. This divergence highlights potential gaps between visual expectations and the built environment, reflecting both stylistic preferences and perceptual bias. Tourist reviews are evaluated through a hybrid sentiment analysis approach combining a rule-based method and a multi-task BERT model. Satisfaction is assessed across four dimensions: tourist activities, built environment, service facilities, and business formats. The results reveal spatial variations in aesthetic appeal and emotional response. Rather than focusing on a single technical innovation, this framework offers an integrated, data-driven approach to decoding tourist perception and contributes to informed decision-making in tourism, heritage conservation, and the design of aesthetically engaging public spaces. 

**Abstract (ZH)**: 历史城镇区在保存文化遗产的同时作为旅游和日常生活的活力空间发挥着重要作用。理解游客对其环境的感知对于人类中心的可持续城市规划至关重要。本研究提出了一种多维度的AI驱动框架，利用社交媒体的多模态数据来分析游客在历史城镇区的感知。该框架应用于上海市中心的十二个历史街区，结合焦点区域提取、颜色主题分析和情感挖掘。通过一种微调的语义分割模型从游客共享的照片中识别视觉关注区域。为了评估审美偏好，使用聚类方法提取主导颜色，并分析其在不同街区的空间分布。通过将社交媒体照片的颜色主题与真实街道视图进行比较，揭示了显著的变化。这种分歧突显了视觉期望与建成环境之间的潜在差距，反映了风格偏好和知觉偏差。通过对基于规则的方法和多任务BERT模型的混合情感分析，评估游客评价。满意度在旅游活动、建成环境、服务设施和商业模式四个维度上进行评估。结果揭示了审美吸引力和情感反应的空间变化。该框架不局限于单一的技术创新，而是提供了一种综合的数据驱动方法，用于解读游客感知并促进旅游业、文化遗产保护和引人入胜的公共空间设计中的决策。 

---
# An Agentic Model Context Protocol Framework for Medical Concept Standardization 

**Title (ZH)**: 基于行动者模型背景协议的医学概念标准化框架 

**Authors**: Jaerong Ahn, Andrew Wen, Nan Wang, Heling Jia, Zhiyi Yue, Sunyang Fu, Hongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03828)  

**Abstract**: The Observational Medical Outcomes Partnership (OMOP) common data model (CDM) provides a standardized representation of heterogeneous health data to support large-scale, multi-institutional research. One critical step in data standardization using OMOP CDM is the mapping of source medical terms to OMOP standard concepts, a procedure that is resource-intensive and error-prone. While large language models (LLMs) have the potential to facilitate this process, their tendency toward hallucination makes them unsuitable for clinical deployment without training and expert validation. Here, we developed a zero-training, hallucination-preventive mapping system based on the Model Context Protocol (MCP), a standardized and secure framework allowing LLMs to interact with external resources and tools. The system enables explainable mapping and significantly improves efficiency and accuracy with minimal effort. It provides real-time vocabulary lookups and structured reasoning outputs suitable for immediate use in both exploratory and production environments. 

**Abstract (ZH)**: Observational Medical Outcomes Partnership (OMOP)通用数据模型（CDM）提供了异质 healthcare 数据的标准表示，以支持大规模、多机构研究。使用OMOP CDM进行数据标准化的一个关键步骤是将源医疗术语映射到OMOP标准概念，这是一个资源密集且容易出错的过程。虽然大型语言模型（LLMs）有潜力促进这一过程，但由于其生成幻觉的倾向，它们在临床部署前需要经过训练和专家验证。在此，我们基于Model Context Protocol (MCP)开发了一种无需训练、防止生成幻觉的映射系统，MCP是一种标准化且安全的框架，允许LLMs与外部资源和工具交互。该系统实现了可解释的映射，并通过最小的努力显著提高了效率和准确性。它提供了实时词汇查询和适合立即在探索性和生产性环境中使用的结构化推理输出。 

---
# What Would an LLM Do? Evaluating Policymaking Capabilities of Large Language Models 

**Title (ZH)**: 大型语言模型能做什么？评估其政策制定能力 

**Authors**: Pierre Le Coz, Jia An Liu, Debarun Bhattacharjya, Georgina Curto, Serge Stinckwich  

**Link**: [PDF](https://arxiv.org/pdf/2509.03827)  

**Abstract**: Large language models (LLMs) are increasingly being adopted in high-stakes domains. Their capacity to process vast amounts of unstructured data, explore flexible scenarios, and handle a diversity of contextual factors can make them uniquely suited to provide new insights for the complexity of social policymaking. This article evaluates whether LLMs' are aligned with domain experts (and among themselves) to inform social policymaking on the subject of homelessness alleviation - a challenge affecting over 150 million people worldwide. We develop a novel benchmark comprised of decision scenarios with policy choices across four geographies (South Bend, USA; Barcelona, Spain; Johannesburg, South Africa; Macau SAR, China). The policies in scope are grounded in the conceptual framework of the Capability Approach for human development. We also present an automated pipeline that connects the benchmarked policies to an agent-based model, and we explore the social impact of the recommended policies through simulated social scenarios. The paper results reveal promising potential to leverage LLMs for social policy making. If responsible guardrails and contextual calibrations are introduced in collaboration with local domain experts, LLMs can provide humans with valuable insights, in the form of alternative policies at scale. 

**Abstract (ZH)**: 大型语言模型（LLMs）在高风险领域中的应用日益增多。它们处理大量非结构化数据、探索灵活场景以及处理不同背景因素的能力，使它们在为社会政策制定提供新见解方面独具优势。本文评估了LLMs与领域专家（包括彼此）是否一致，以指导社会政策制定，重点关注无家可归问题的缓解——这是一个影响全球逾1.5亿人的挑战。我们开发了一个包含四个地理区域（美国南本德、西班牙巴塞罗那、南非约翰内斯堡、中国澳门特别行政区）决策场景的新基准，涵盖的政策基于人类发展的能力方法论。我们还介绍了将基准政策与基于代理的模型连接的自动化管道，并通过模拟社会场景探讨了推荐政策的社会影响。研究结果表明，如果与当地领域专家合作引入负责任的防护措施和情境校准，LLMs可以为人类提供有价值的新政策见解。 

---
# Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning 

**Title (ZH)**: 学会权衡：多智能体强化学习中代理型LLM的元策略协作 

**Authors**: Wei Yang, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2509.03817)  

**Abstract**: Multi-agent systems of large language models (LLMs) show promise for complex reasoning, but their effectiveness is often limited by fixed collaboration protocols. These frameworks typically focus on macro-level orchestration while overlooking agents' internal deliberative capabilities. This critical meta-cognitive blindspot treats agents as passive executors unable to adapt their strategy based on internal cognitive states like uncertainty or confidence. We introduce the Meta-Policy Deliberation Framework (MPDF), where agents learn a decentralized policy over a set of high-level meta-cognitive actions: Persist, Refine, and Concede. To overcome the instability of traditional policy gradients in this setting, we develop SoftRankPO, a novel reinforcement learning algorithm. SoftRankPO stabilizes training by shaping advantages based on the rank of rewards mapped through smooth normal quantiles, making the learning process robust to reward variance. Experiments show that MPDF with SoftRankPO achieves a a 4-5% absolute gain in average accuracy across five mathematical and general reasoning benchmarks compared to six state-of-the-art heuristic and learning-based multi-agent reasoning algorithms. Our work presents a paradigm for learning adaptive, meta-cognitive policies for multi-agent LLM systems, shifting the focus from designing fixed protocols to learning dynamic, deliberative strategies. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的多代理系统在复杂推理方面展现出潜力，但其效果 often 限于固定的合作协议。这些框架通常专注于宏观层面的协调，而忽视了代理的内部反思能力。这一关键的元认知盲点将代理视为被动执行者，无法根据不确定性或信心等内部认知状态调整其策略。我们引入了元策略反思框架（MPDF），其中代理学习一套高层次的元认知动作的去中心化策略：坚持、细化和让步。为克服该设置下传统策略梯度的不稳定性，我们开发了SoftRankPO，这是一种新颖的强化学习算法。SoftRankPO通过基于奖励排名平滑正态量化来塑造优势，从而通过稳定训练过程来提高学习过程的鲁棒性，使其对奖励波动不敏感。实验表明，与六种最先进的启发式和学习型多代理推理算法相比，使用SoftRankPO的MPDF在五个数学和一般推理基准上的平均准确性绝对增益为4-5%。我们的工作提出了一个学习多代理LLM系统的适应性和元认知策略的范式，从设计固定协议转向学习动态的、反思性的策略。 

---
# Leveraging LLM-Based Agents for Intelligent Supply Chain Planning 

**Title (ZH)**: 基于LLM的代理智能供应链规划 

**Authors**: Yongzhi Qi, Jiaheng Yin, Jianshen Zhang, Dongyang Geng, Zhengyu Chen, Hao Hu, Wei Qi, Zuo-Jun Max Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.03811)  

**Abstract**: In supply chain management, planning is a critical concept. The movement of physical products across different categories, from suppliers to warehouse management, to sales, and logistics transporting them to customers, entails the involvement of many entities. It covers various aspects such as demand forecasting, inventory management, sales operations, and replenishment. How to collect relevant data from an e-commerce platform's perspective, formulate long-term plans, and dynamically adjust them based on environmental changes, while ensuring interpretability, efficiency, and reliability, is a practical and challenging problem. In recent years, the development of AI technologies, especially the rapid progress of large language models, has provided new tools to address real-world issues. In this work, we construct a Supply Chain Planning Agent (SCPA) framework that can understand domain knowledge, comprehend the operator's needs, decompose tasks, leverage or create new tools, and return evidence-based planning reports. We deploy this framework in this http URL's real-world scenario, demonstrating the feasibility of LLM-agent applications in the supply chain. It effectively reduced labor and improved accuracy, stock availability, and other key metrics. 

**Abstract (ZH)**: 在供应链管理中，规划是一个关键概念。从供应商到仓库管理，再到销售和物流运输到客户，物理产品的流动涉及众多实体，并涵盖需求预测、库存管理、销售运营和补货等多方面内容。如何从电子商务平台的角度收集相关数据，制定长期计划，并根据环境变化动态调整这些计划，同时确保解释性、效率和可靠性，是一个实际且具有挑战性的问题。近年来，AI技术的发展，特别是大规模语言模型的迅速进步，为解决实际问题提供了新的工具。在本文中，我们构建了一个供应链规划代理（SCPA）框架，该框架能够理解领域知识，理解操作员的需求，分解任务，利用或创建新工具，并返回基于证据的规划报告。我们在这一电子商务平台的真实场景中部署了这一框架，展示了LLM-代理在供应链中的可行性应用，有效减少了劳动成本，提高了准确性、库存可用性等关键指标。 

---
# RAGuard: A Novel Approach for in-context Safe Retrieval Augmented Generation for LLMs 

**Title (ZH)**: RAGuard: 一种针对LLMs的新型基于上下文的安全检索增强生成方法 

**Authors**: Connor Walker, Koorosh Aslansefat, Mohammad Naveed Akram, Yiannis Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.03768)  

**Abstract**: Accuracy and safety are paramount in Offshore Wind (OSW) maintenance, yet conventional Large Language Models (LLMs) often fail when confronted with highly specialised or unexpected scenarios. We introduce RAGuard, an enhanced Retrieval-Augmented Generation (RAG) framework that explicitly integrates safety-critical documents alongside technical this http URL issuing parallel queries to two indices and allocating separate retrieval budgets for knowledge and safety, RAGuard guarantees both technical depth and safety coverage. We further develop a SafetyClamp extension that fetches a larger candidate pool, "hard-clamping" exact slot guarantees to safety. We evaluate across sparse (BM25), dense (Dense Passage Retrieval) and hybrid retrieval paradigms, measuring Technical Recall@K and Safety Recall@K. Both proposed extensions of RAG show an increase in Safety Recall@K from almost 0\% in RAG to more than 50\% in RAGuard, while maintaining Technical Recall above 60\%. These results demonstrate that RAGuard and SafetyClamp have the potential to establish a new standard for integrating safety assurance into LLM-powered decision support in critical maintenance contexts. 

**Abstract (ZH)**: 基于检索增强生成的RAGuard及其在远洋风电维护中的安全保障应用：提高准确性和安全性 

---
# Are LLM Agents Behaviorally Coherent? Latent Profiles for Social Simulation 

**Title (ZH)**: LLM代理的社会行为一致性：潜在聚类分析 

**Authors**: James Mooney, Josef Woldense, Zheng Robert Jia, Shirley Anugrah Hayati, My Ha Nguyen, Vipul Raheja, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03736)  

**Abstract**: The impressive capabilities of Large Language Models (LLMs) have fueled the notion that synthetic agents can serve as substitutes for real participants in human-subject research. In an effort to evaluate the merits of this claim, social science researchers have largely focused on whether LLM-generated survey data corresponds to that of a human counterpart whom the LLM is prompted to represent. In contrast, we address a more fundamental question: Do agents maintain internal consistency, retaining similar behaviors when examined under different experimental settings? To this end, we develop a study designed to (a) reveal the agent's internal state and (b) examine agent behavior in a basic dialogue setting. This design enables us to explore a set of behavioral hypotheses to assess whether an agent's conversation behavior is consistent with what we would expect from their revealed internal state. Our findings on these hypotheses show significant internal inconsistencies in LLMs across model families and at differing model sizes. Most importantly, we find that, although agents may generate responses matching those of their human counterparts, they fail to be internally consistent, representing a critical gap in their capabilities to accurately substitute for real participants in human-subject research. Our simulation code and data are publicly accessible. 

**Abstract (ZH)**: 大型语言模型的显著能力激发了合成代理可以作为人类参与者在人类受控研究中替代品的观念。为了评估这一观点的优势，社会科学研究人员主要关注LLM生成的调查数据是否与被LLM提示代表的人类对应者的数据相符。相比之下，我们探讨了一个更为基础的问题：代理在不同的实验环境中保持内部一致性，展现出相似的行为吗？为此，我们设计了一项研究，旨在(a) 展示代理的内部状态，并在基本对话环境中(b) 考察代理行为。这一设计使我们能够探索一系列行为假设，以评估代理的对话行为是否与其揭示的内部状态一致。我们的研究表明，不同模型家族和不同模型规模的LLM在内部存在显著不一致性。最重要的是，我们发现尽管代理可能会生成与人类对应者相符的响应，但它们在内部不一致，这代表他们在准确替代人类参与者进行人类受控研究方面存在关键差距。我们的模拟代码和数据是公开可访问的。 

---
# The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs 

**Title (ZH)**: 个性错觉：揭示大语言模型自我报告与行为之间的 dissociation 

**Authors**: Pengrui Han, Rafal Kocielnik, Peiyang Song, Ramit Debnath, Dean Mobbs, Anima Anandkumar, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2509.03730)  

**Abstract**: Personality traits have long been studied as predictors of human this http URL advances in Large Language Models (LLMs) suggest similar patterns may emerge in artificial systems, with advanced LLMs displaying consistent behavioral tendencies resembling human traits like agreeableness and self-regulation. Understanding these patterns is crucial, yet prior work primarily relied on simplified self-reports and heuristic prompting, with little behavioral validation. In this study, we systematically characterize LLM personality across three dimensions: (1) the dynamic emergence and evolution of trait profiles throughout training stages; (2) the predictive validity of self-reported traits in behavioral tasks; and (3) the impact of targeted interventions, such as persona injection, on both self-reports and behavior. Our findings reveal that instructional alignment (e.g., RLHF, instruction tuning) significantly stabilizes trait expression and strengthens trait correlations in ways that mirror human data. However, these self-reported traits do not reliably predict behavior, and observed associations often diverge from human patterns. While persona injection successfully steers self-reports in the intended direction, it exerts little or inconsistent effect on actual behavior. By distinguishing surface-level trait expression from behavioral consistency, our findings challenge assumptions about LLM personality and underscore the need for deeper evaluation in alignment and interpretability. 

**Abstract (ZH)**: 个性特征一直是研究人类行为预测的因素。随着大型语言模型（LLMs）的进步，类似模式可能在人工系统中出现，先进LLMs表现出类似于人类特质（如随和性和自我调节）的一致行为倾向。理解这些模式至关重要，但以往研究主要依赖简化的自我报告和启发式提示，缺乏行为验证。本研究系统地从三个维度对LLM个性进行表征：（1）在训练阶段特质概况的动态出现和演变；（2）自我报告特质在行为任务中的预测有效性；以及（3）目标干预措施（如Persona注入）对自我报告和行为的影响。我们的研究发现，指令对齐（如RLHF、指令调整）显著稳定了特质表达并加强了特质之间的关联，这种关联类似于人类数据。然而，这些自我报告的特质并不能可靠地预测行为，观察到的关联通常与人类模式不符。虽然Persona注入成功地引导了自我报告的方向，但其对实际行为的影响微乎其微或不一致。通过区分表面特征表达与行为一致性，我们的发现挑战了对LLM个性的假设，并强调了在对齐和解释性方面进行更深入评估的必要性。 

---
# PersonaTeaming: Exploring How Introducing Personas Can Improve Automated AI Red-Teaming 

**Title (ZH)**: PersonaTeaming：探究引入人物角色如何改善自动化AI红队演练 

**Authors**: Wesley Hanwen Deng, Sunnie S. Y. Kim, Akshita Jha, Ken Holstein, Motahhare Eslami, Lauren Wilcox, Leon A Gatys  

**Link**: [PDF](https://arxiv.org/pdf/2509.03728)  

**Abstract**: Recent developments in AI governance and safety research have called for red-teaming methods that can effectively surface potential risks posed by AI models. Many of these calls have emphasized how the identities and backgrounds of red-teamers can shape their red-teaming strategies, and thus the kinds of risks they are likely to uncover. While automated red-teaming approaches promise to complement human red-teaming by enabling larger-scale exploration of model behavior, current approaches do not consider the role of identity. As an initial step towards incorporating people's background and identities in automated red-teaming, we develop and evaluate a novel method, PersonaTeaming, that introduces personas in the adversarial prompt generation process to explore a wider spectrum of adversarial strategies. In particular, we first introduce a methodology for mutating prompts based on either "red-teaming expert" personas or "regular AI user" personas. We then develop a dynamic persona-generating algorithm that automatically generates various persona types adaptive to different seed prompts. In addition, we develop a set of new metrics to explicitly measure the "mutation distance" to complement existing diversity measurements of adversarial prompts. Our experiments show promising improvements (up to 144.1%) in the attack success rates of adversarial prompts through persona mutation, while maintaining prompt diversity, compared to RainbowPlus, a state-of-the-art automated red-teaming method. We discuss the strengths and limitations of different persona types and mutation methods, shedding light on future opportunities to explore complementarities between automated and human red-teaming approaches. 

**Abstract (ZH)**: 近期人工智能治理与安全研究的发展呼吁采用能够有效揭示AI模型潜在风险的红队方法。这些呼吁强调了红队成员的身份和背景如何影响红队策略，并进而影响他们可能发现的风险类型。虽然自动化红队方法有望通过支持更广泛的模型行为探索来补充人类红队方法，但现有方法尚未考虑身份的作用。为初步将人们的背景和身份纳入自动化红队方法中，我们开发并评估了一种新方法——PersonaTeaming，该方法在对抗性提示生成过程中引入人格，以探索更广泛的对抗性策略。具体而言，我们首先引入了一种基于“红队专家”人格或“普通AI用户”人格的提示变异方法。然后，我们开发了一种动态的人格生成算法，能够自动生成适应不同种子提示的各种人格类型。此外，我们还开发了一套新的度量标准，以明确度量“变异距离”，补充现有对抗性提示多样性的度量。我们的实验结果显示，与当前最先进的自动化红队方法RainbowPlus相比，通过人格变异，对抗性提示的攻击成功率提高了144.1%，同时保持了提示的多样性。我们讨论了不同人格类型和变异方法的优势与局限性，揭示了未来探索自动化和人类红队方法互补性的潜在机会。 

---
# An Empirical Evaluation of Factors Affecting SHAP Explanation of Time Series Classification 

**Title (ZH)**: 时间序列分类中SHAP解释的影响因素 empirical evaluation 

**Authors**: Davide Italo Serramazza, Nikos Papadeas, Zahraa Abdallah, Georgiana Ifrim  

**Link**: [PDF](https://arxiv.org/pdf/2509.03649)  

**Abstract**: Explainable AI (XAI) has become an increasingly important topic for understanding and attributing the predictions made by complex Time Series Classification (TSC) models. Among attribution methods, SHapley Additive exPlanations (SHAP) is widely regarded as an excellent attribution method; but its computational complexity, which scales exponentially with the number of features, limits its practicality for long time series. To address this, recent studies have shown that aggregating features via segmentation, to compute a single attribution value for a group of consecutive time points, drastically reduces SHAP running time. However, the choice of the optimal segmentation strategy remains an open question. In this work, we investigated eight different Time Series Segmentation algorithms to understand how segment compositions affect the explanation quality. We evaluate these approaches using two established XAI evaluation methodologies: InterpretTime and AUC Difference. Through experiments on both Multivariate (MTS) and Univariate Time Series (UTS), we find that the number of segments has a greater impact on explanation quality than the specific segmentation method. Notably, equal-length segmentation consistently outperforms most of the custom time series segmentation algorithms. Furthermore, we introduce a novel attribution normalisation technique that weights segments by their length and we show that it consistently improves attribution quality. 

**Abstract (ZH)**: 可解释人工智能(XAI)已成为理解复杂时间序列分类(TSC)模型预测的重要话题。在归因方法中，SHapley Additive exPlanations (SHAP)被认为是优秀的归因方法之一，但由于其计算复杂性随特征数量指数增长，限制了其在长时间序列上的实用性。为解决这一问题，最近的研究表明，通过分段聚合特征来计算一系列连续时间点的单一归因值，可以大幅减少SHAP的运行时间。然而，最优分段策略的选择仍然是一个开放问题。在本文中，我们研究了八种不同的时间序列分段算法，以了解分段组成如何影响解释质量。我们使用两种已建立的XAI评估方法：InterpretTime和AUC Difference来评估这些方法。通过在多变量时间序列(MTS)和单变量时间序列(UTS)上的实验，我们发现时间段的数量对解释质量的影响大于特定分段方法的选择。值得注意的是，等长度分段始终优于大多数自定义时间序列分段算法。此外，我们引入了一种新的归因规范化技术，通过时间段长度进行加权，并展示了其一致地提高了归因质量。 

---
# Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning 

**Title (ZH)**: 通过强化学习实现的LLMs中的 emergent 分层推理 

**Authors**: Haozhe Wang, Qixin Xu, Che Liu, Junhong Wu, Fangzhen Lin, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.03646)  

**Abstract**: Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy. 

**Abstract (ZH)**: 强化学习（RL）在增强大型语言模型（LLMs）的复杂推理能力方面 proven 非常有效，但其成功背后的具体机制仍然相对模糊。我们的分析揭示了诸如“顿悟时刻”、“长度缩放”和熵动态等令人困惑的现象并非孤立存在，而是 Emergent 推理层次结构的标志，类似于人类认知中高层战略规划与低层程序执行的分离。我们发现了一个有说服力的两阶段动态过程：最初，模型受程序正确性的限制，必须改进其低级技能。然后，学习瓶颈显着转变，性能提升主要通过探索和掌握高层战略规划驱动。这一洞见揭示了现有 RL 算法（如 GRPO）的核心低效性，这些算法无区别地应用优化压力，并在所有标记中稀释学习信号。为此，我们提出了 Awareness 基于层次结构的奖赏归属（HICRA）算法，该算法将优化努力集中在高影响规划标记上。HICRA 显著优于强大的基准，表明专注于这一战略瓶颈是解锁高级推理的关键。此外，我们验证了语义熵作为衡量战略探索的优越标准，优于误导性的标记级熵等度量标准。 

---
# Towards a Neurosymbolic Reasoning System Grounded in Schematic Representations 

**Title (ZH)**: 基于方案表示的神经符号推理系统研究 

**Authors**: François Olivier, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2509.03644)  

**Abstract**: Despite significant progress in natural language understanding, Large Language Models (LLMs) remain error-prone when performing logical reasoning, often lacking the robust mental representations that enable human-like comprehension. We introduce a prototype neurosymbolic system, Embodied-LM, that grounds understanding and logical reasoning in schematic representations based on image schemas-recurring patterns derived from sensorimotor experience that structure human cognition. Our system operationalizes the spatial foundations of these cognitive structures using declarative spatial reasoning within Answer Set Programming. Through evaluation on logical deduction problems, we demonstrate that LLMs can be guided to interpret scenarios through embodied cognitive structures, that these structures can be formalized as executable programs, and that the resulting representations support effective logical reasoning with enhanced interpretability. While our current implementation focuses on spatial primitives, it establishes the computational foundation for incorporating more complex and dynamic representations. 

**Abstract (ZH)**: 尽管在自然语言理解方面取得了显著进展，大型语言模型（LLMs）在执行逻辑推理时仍易出错，往往缺乏支撑人类类似理解的稳健的心理表征。我们提出了一种原型神经符号系统——Embodied-LM，该系统基于图式表示进行理解与逻辑推理，这些图式是源自感官运动体验中的反复出现的模式，结构化人类认知。我们的系统通过答案集程序中的声明性空间推理对这些认知结构的空间基础进行了实现。通过在逻辑推理问题上的评估，我们证明LLMs可以通过基于 embodied 认知结构来解释场景，这些结构可以被形式化为可执行程序，并且由此产生的表示支持具有增强解释性的有效逻辑推理。尽管我们当前的实现专注于空间原语，但它为整合更复杂和动态的表示奠定了计算基础。 

---
# CausalARC: Abstract Reasoning with Causal World Models 

**Title (ZH)**: 因果ARC：基于因果世界模型的抽象推理 

**Authors**: Jacqueline Maasch, John Kalantari, Kia Khezeli  

**Link**: [PDF](https://arxiv.org/pdf/2509.03636)  

**Abstract**: Reasoning requires adaptation to novel problem settings under limited data and distribution shift. This work introduces CausalARC: an experimental testbed for AI reasoning in low-data and out-of-distribution regimes, modeled after the Abstraction and Reasoning Corpus (ARC). Each CausalARC reasoning task is sampled from a fully specified causal world model, formally expressed as a structural causal model. Principled data augmentations provide observational, interventional, and counterfactual feedback about the world model in the form of few-shot, in-context learning demonstrations. As a proof-of-concept, we illustrate the use of CausalARC for four language model evaluation settings: (1) abstract reasoning with test-time training, (2) counterfactual reasoning with in-context learning, (3) program synthesis, and (4) causal discovery with logical reasoning. 

**Abstract (ZH)**: 因果推理要求在有限数据和分布偏移的情况下适应新的问题设置。本文介绍了CausalARC：一种针对低数据和分布外域的AI推理实验测试床，其设计灵感来源于抽象和推理语料库（ARC）。每个CausalARC推理任务都源自一个完全指定的因果世界模型，并以结构因果模型的形式正式表达。基于原则的数据增强提供了关于世界模型的观察性、干预性和反事实反馈，以少样本、上下文相关的学习演示形式呈现。作为概念验证，我们展示了CausalARC在四种语言模型评估场景中的应用：(1) 测试时训练的抽象推理，(2) 上下文学习的反事实推理，(3) 程序合成，(4) 逻辑推理驱动的因果发现。 

---
# Explainable Knowledge Graph Retrieval-Augmented Generation (KG-RAG) with KG-SMILE 

**Title (ZH)**: 可解释的知识图谱检索增强生成（KG-RAG）方法结合KG-SMILE 

**Authors**: Zahra Zehtabi Sabeti Moghaddam, Zeinab Dehghani, Maneeha Rani, Koorosh Aslansefat, Bhupesh Kumar Mishra, Rameez Raja Kureshi, Dhavalkumar Thakker  

**Link**: [PDF](https://arxiv.org/pdf/2509.03626)  

**Abstract**: Generative AI, such as Large Language Models (LLMs), has achieved impressive progress but still produces hallucinations and unverifiable claims, limiting reliability in sensitive domains. Retrieval-Augmented Generation (RAG) improves accuracy by grounding outputs in external knowledge, especially in domains like healthcare, where precision is vital. However, RAG remains opaque and essentially a black box, heavily dependent on data quality. We developed a method-agnostic, perturbation-based framework that provides token and component-level interoperability for Graph RAG using SMILE and named it as Knowledge-Graph (KG)-SMILE. By applying controlled perturbations, computing similarities, and training weighted linear surrogates, KG-SMILE identifies the graph entities and relations most influential to generated outputs, thereby making RAG more transparent. We evaluate KG-SMILE using comprehensive attribution metrics, including fidelity, faithfulness, consistency, stability, and accuracy. Our findings show that KG-SMILE produces stable, human-aligned explanations, demonstrating its capacity to balance model effectiveness with interpretability and thereby fostering greater transparency and trust in machine learning technologies. 

**Abstract (ZH)**: 生成式AI，如大型语言模型（LLMs），取得了显著进展，但仍会产生幻觉和无法验证的断言，限制了在敏感领域的可靠性。检索增强生成（RAG）通过将输出 grounding 在外部知识中提高了准确性，尤其是在对精度要求高的医疗等领域。然而，RAG 仍然缺乏透明度，本质上是一个黑箱，高度依赖数据质量。我们开发了一种方法无关的扰动基础框架，用于Graph RAG，并将其命名为Knowledge-Graph（KG）-SMILE。通过应用可控扰动、计算相似性并训练加权线性替代模型，KG-SMILE 确定了对生成输出最具影响力的图实体和关系，从而提高了RAG的透明度。我们利用全面的归因度量标准（包括忠实度、真实性、一致性、稳定性和准确性）评估了KG-SMILE。我们的研究结果表明，KG-SMILE 生成了稳定且与人类一致的解释，展示了其在提高模型效果和可解释性方面的平衡能力，从而促进了机器学习技术更高的透明度和信任度。 

---
# Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents 

**Title (ZH)**: 学习何时规划：在LLM代理的测试时间计算分配中的高效调度 

**Authors**: Davide Paglieri, Bartłomiej Cupiał, Jonathan Cook, Ulyana Piterbarg, Jens Tuyls, Edward Grefenstette, Jakob Nicolaus Foerster, Jack Parker-Holder, Tim Rocktäschel  

**Link**: [PDF](https://arxiv.org/pdf/2509.03581)  

**Abstract**: Training large language models (LLMs) to reason via reinforcement learning (RL) significantly improves their problem-solving capabilities. In agentic settings, existing methods like ReAct prompt LLMs to explicitly plan before every action; however, we demonstrate that always planning is computationally expensive and degrades performance on long-horizon tasks, while never planning further limits performance. To address this, we introduce a conceptual framework formalizing dynamic planning for LLM agents, enabling them to flexibly decide when to allocate test-time compute for planning. We propose a simple two-stage training pipeline: (1) supervised fine-tuning on diverse synthetic data to prime models for dynamic planning, and (2) RL to refine this capability in long-horizon environments. Experiments on the Crafter environment show that dynamic planning agents trained with this approach are more sample-efficient and consistently achieve more complex objectives. Additionally, we demonstrate that these agents can be effectively steered by human-written plans, surpassing their independent capabilities. To our knowledge, this work is the first to explore training LLM agents for dynamic test-time compute allocation in sequential decision-making tasks, paving the way for more efficient, adaptive, and controllable agentic systems. 

**Abstract (ZH)**: 通过强化学习训练大规模语言模型以进行推理显著提高了它们的解决问题能力。在代理设置中，现有方法如ReAct要求语言模型在每项行动前明确规划；然而，我们证明常に规划在计算上非常昂贵，并且会降低长期任务的表现，而从不规划则进一步限制了性能。为了解决这个问题，我们提出了一个概念框架，以正式化大规模语言模型代理的动态规划，使它们能够灵活决定何时分配测试时计算资源进行规划。我们提出了一个简单的两阶段训练管道：（1）在多元合成数据上进行监督微调，以使模型为动态规划做好准备；（2）通过长期环境中的强化学习精化这一能力。在Crafter环境中进行的实验表明，使用这种方法训练的动态规划代理更加样本高效，并且能够更一致地实现更复杂的目标。此外，我们展示了这些代理可以通过人类撰写的计划进行有效引导，超越了它们的独立能力。据我们所知，这项工作是首次探索训练语言模型代理在序列决策任务中进行动态测试时计算资源分配的方法，为更高效、适应性和可控的代理系统铺平了道路。 

---
# Diffusion-RL Based Air Traffic Conflict Detection and Resolution Method 

**Title (ZH)**: 基于扩散-强化学习的空中交通冲突检测与化解方法 

**Authors**: Tonghe Li, Jixin Liu, Weili Zeng, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03550)  

**Abstract**: In the context of continuously rising global air traffic, efficient and safe Conflict Detection and Resolution (CD&R) is paramount for air traffic management. Although Deep Reinforcement Learning (DRL) offers a promising pathway for CD&R automation, existing approaches commonly suffer from a "unimodal bias" in their policies. This leads to a critical lack of decision-making flexibility when confronted with complex and dynamic constraints, often resulting in "decision deadlocks." To overcome this limitation, this paper pioneers the integration of diffusion probabilistic models into the safety-critical task of CD&R, proposing a novel autonomous conflict resolution framework named Diffusion-AC. Diverging from conventional methods that converge to a single optimal solution, our framework models its policy as a reverse denoising process guided by a value function, enabling it to generate a rich, high-quality, and multimodal action distribution. This core architecture is complemented by a Density-Progressive Safety Curriculum (DPSC), a training mechanism that ensures stable and efficient learning as the agent progresses from sparse to high-density traffic environments. Extensive simulation experiments demonstrate that the proposed method significantly outperforms a suite of state-of-the-art DRL benchmarks. Most critically, in the most challenging high-density scenarios, Diffusion-AC not only maintains a high success rate of 94.1% but also reduces the incidence of Near Mid-Air Collisions (NMACs) by approximately 59% compared to the next-best-performing baseline, significantly enhancing the system's safety margin. This performance leap stems from its unique multimodal decision-making capability, which allows the agent to flexibly switch to effective alternative maneuvers. 

**Abstract (ZH)**: 在全球 aviation 交通持续增长的背景下，高效的且安全的冲突检测与解决（CD&R）对于 aviation 交通管理至关重要。尽管深度强化学习（DRL）为 CD&R 的自动化提供了有前景的道路，但现有方法通常在其策略中存在“单模偏差”。这导致在面对复杂和动态的约束时缺乏决策灵活性，常导致“决策僵局”。为克服这一局限，本文首次将扩散概率模型集成到安全关键的任务——CD&R 中，提出了一种名为 Diffusion-AC 的新型自主冲突解决框架。不同于传统方法收敛于单一最优解，我们的框架将策略建模为由价值函数引导的逆去噪过程，使其能够生成丰富、高质且多模态的动作分布。该核心架构由密度进展安全课程（DPSC）训练机制加以补充，确保代理在从稀疏环境过渡到高密度环境时能实现稳定且高效的训练。大量仿真实验表明，所提出的方法显著优于一系列现有的 DRL 参考基准。尤其在最具有挑战性的高密度场景中，Diffusion-AC 保持了高达 94.1% 的高成功率，并且与性能次之的竞争者相比，将近失接近空中碰撞（NMAC）的频率降低了约 59%，显著提升了系统的安全余度。这种性能跃升源于其独特的多模态决策能力，使代理能够灵活切换至有效的替代机动。 

---
# Multilinear and Linear Programs for Partially Identifiable Queries in Quasi-Markovian Structural Causal Models 

**Title (ZH)**: 部分可识别查询在准马尔可夫结构因果模型中的多线性与线性规划 

**Authors**: João P. Arroyo, João G. Rodrigues, Daniel Lawand, Denis D. Mauá, Junkyu Lee, Radu Marinescu, Alex Gray, Eduardo R. Laurentino, Fabio G. Cozman  

**Link**: [PDF](https://arxiv.org/pdf/2509.03548)  

**Abstract**: We investigate partially identifiable queries in a class of causal models. We focus on acyclic Structural Causal Models that are quasi-Markovian (that is, each endogenous variable is connected with at most one exogenous confounder). We look into scenarios where endogenous variables are observed (and a distribution over them is known), while exogenous variables are not fully specified. This leads to a representation that is in essence a Bayesian network where the distribution of root variables is not uniquely determined. In such circumstances, it may not be possible to precisely compute a probability value of interest. We thus study the computation of tight probability bounds, a problem that has been solved by multilinear programming in general, and by linear programming when a single confounded component is intervened upon. We present a new algorithm to simplify the construction of such programs by exploiting input probabilities over endogenous variables. For scenarios with a single intervention, we apply column generation to compute a probability bound through a sequence of auxiliary linear integer programs, thus showing that a representation with polynomial cardinality for exogenous variables is possible. Experiments show column generation techniques to be superior to existing methods. 

**Abstract (ZH)**: 我们在因果模型类中研究部分可识别查询。我们侧重于无环结构因果模型，这些模型准马尔可夫（即，每个内生变量最多与一个外生混杂变量相连）。我们探讨了内生变量已观测且其分布已知而外生变量不完全指定的情况，这种情况下，代表本质上是一个贝叶斯网络，其中根变量的分布不是唯一确定的。在这种情况下，可能无法精确计算感兴趣的概率值。因此，我们研究了计算紧概率界的计算问题，这个问题一般通过多线性规划解决，当干预单一混杂成分时可以通过线性规划解决。我们提出了一种新算法，通过利用内生变量的输入概率简化此类程序的构造。对于单一干预情景，我们采用列生成技术通过一系列辅助线性整数规划计算概率界，从而证明在外生变量表示中存在多项式基数的表示方法。实验显示列生成技术优于现有方法。 

---
# PG-Agent: An Agent Powered by Page Graph 

**Title (ZH)**: PG-Agent: 由页面图驱动的智能代理 

**Authors**: Weizhi Chen, Ziwei Wang, Leyang Yang, Sheng Zhou, Xiaoxuan Tang, Jiajun Bu, Yong Li, Wei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03536)  

**Abstract**: Graphical User Interface (GUI) agents possess significant commercial and social value, and GUI agents powered by advanced multimodal large language models (MLLMs) have demonstrated remarkable potential. Currently, existing GUI agents usually utilize sequential episodes of multi-step operations across pages as the prior GUI knowledge, which fails to capture the complex transition relationship between pages, making it challenging for the agents to deeply perceive the GUI environment and generalize to new scenarios. Therefore, we design an automated pipeline to transform the sequential episodes into page graphs, which explicitly model the graph structure of the pages that are naturally connected by actions. To fully utilize the page graphs, we further introduce Retrieval-Augmented Generation (RAG) technology to effectively retrieve reliable perception guidelines of GUI from them, and a tailored multi-agent framework PG-Agent with task decomposition strategy is proposed to be injected with the guidelines so that it can generalize to unseen scenarios. Extensive experiments on various benchmarks demonstrate the effectiveness of PG-Agent, even with limited episodes for page graph construction. 

**Abstract (ZH)**: 基于先进多模态大语言模型的图形用户界面代理具有显著的商业和社会价值，现有基于高级多模态大语言模型（MLLMs）的GUI代理展示了巨大的潜力。目前，现有的GUI代理通常利用跨页面的多步操作的序列片段作为先验GUI知识，这未能捕捉到页面之间的复杂转换关系，使得代理难以深入感知GUI环境并泛化到新场景。因此，我们设计了一个自动化管道，将序列片段转换为页面图，明确建模由动作自然连接的页面的图结构。为了充分利用页面图，我们进一步引入了检索增强生成（RAG）技术，有效地从页面图中检索可靠的GUI感知准则，并提出了一种定制化的多代理框架PG-Agent，通过分解任务策略注入这些准则，使其能够泛化到未见过的场景。广泛的基准测试实验表明，即使在页面图构建时仅有有限的序列片段，PG-Agent也具有有效性。 

---
# ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset 

**Title (ZH)**: ChronoGraph: 一种基于图的现实世界多变量时间序列数据集 

**Authors**: Adrian Catalin Lutu, Ioana Pintilie, Elena Burceanu, Andrei Manolache  

**Link**: [PDF](https://arxiv.org/pdf/2509.04449)  

**Abstract**: We present ChronoGraph, a graph-structured multivariate time series forecasting dataset built from real-world production microservices. Each node is a service that emits a multivariate stream of system-level performance metrics, capturing CPU, memory, and network usage patterns, while directed edges encode dependencies between services. The primary task is forecasting future values of these signals at the service level. In addition, ChronoGraph provides expert-annotated incident windows as anomaly labels, enabling evaluation of anomaly detection methods and assessment of forecast robustness during operational disruptions. Compared to existing benchmarks from industrial control systems or traffic and air-quality domains, ChronoGraph uniquely combines (i) multivariate time series, (ii) an explicit, machine-readable dependency graph, and (iii) anomaly labels aligned with real incidents. We report baseline results spanning forecasting models, pretrained time-series foundation models, and standard anomaly detectors. ChronoGraph offers a realistic benchmark for studying structure-aware forecasting and incident-aware evaluation in microservice systems. 

**Abstract (ZH)**: ChronoGraph：一种基于真实生产微服务构建的图结构多变量时间序列预测数据集 

---
# Delta Activations: A Representation for Finetuned Large Language Models 

**Title (ZH)**: Delta Activations: 一种Fine-tuned大型语言模型的表示方法 

**Authors**: Zhiqiu Xu, Amish Sethi, Mayur Naik, Ser-Nam Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.04442)  

**Abstract**: The success of powerful open source Large Language Models (LLMs) has enabled the community to create a vast collection of post-trained models adapted to specific tasks and domains. However, navigating and understanding these models remains challenging due to inconsistent metadata and unstructured repositories. We introduce Delta Activations, a method to represent finetuned models as vector embeddings by measuring shifts in their internal activations relative to a base model. This representation allows for effective clustering by domain and task, revealing structure in the model landscape. Delta Activations also demonstrate desirable properties: it is robust across finetuning settings and exhibits an additive property when finetuning datasets are mixed. In addition, we show that Delta Activations can embed tasks via few-shot finetuning, and further explore its use for model selection and merging. We hope Delta Activations can facilitate the practice of reusing publicly available models. Code is available at this https URL. 

**Abstract (ZH)**: 强大的开源大型语言模型的成功使得社区能够创建针对特定任务和领域调整后的模型集合。然而，由于元数据不一致和非结构化存储库，导航和理解这些模型仍具有挑战性。我们介绍了一种名为Delta激活的方法，通过测量调整后的模型与其基模型的内部激活变化来表示这些模型为向量嵌入。这种方法允许按领域和任务有效聚类，揭示模型景观中的结构。Delta激活还展示了可取的性质：它在不同的调整设置中具有稳健性，并且当混合不同数据集进行调整时表现出加性性质。此外，我们展示了Delta激活可以通过少量示例调整来嵌入任务，并进一步探讨了其在模型选择和合并中的应用。我们希望Delta激活能够促进重用公开模型的做法。代码可在以下链接获取。 

---
# DEXOP: A Device for Robotic Transfer of Dexterous Human Manipulation 

**Title (ZH)**: DEXOP: 一种实现灵巧人类操作转移的设备 

**Authors**: Hao-Shu Fang, Branden Romero, Yichen Xie, Arthur Hu, Bo-Ruei Huang, Juan Alvarez, Matthew Kim, Gabriel Margolis, Kavya Anbarasu, Masayoshi Tomizuka, Edward Adelson, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2509.04441)  

**Abstract**: We introduce perioperation, a paradigm for robotic data collection that sensorizes and records human manipulation while maximizing the transferability of the data to real robots. We implement this paradigm in DEXOP, a passive hand exoskeleton designed to maximize human ability to collect rich sensory (vision + tactile) data for diverse dexterous manipulation tasks in natural environments. DEXOP mechanically connects human fingers to robot fingers, providing users with direct contact feedback (via proprioception) and mirrors the human hand pose to the passive robot hand to maximize the transfer of demonstrated skills to the robot. The force feedback and pose mirroring make task demonstrations more natural for humans compared to teleoperation, increasing both speed and accuracy. We evaluate DEXOP across a range of dexterous, contact-rich tasks, demonstrating its ability to collect high-quality demonstration data at scale. Policies learned with DEXOP data significantly improve task performance per unit time of data collection compared to teleoperation, making DEXOP a powerful tool for advancing robot dexterity. Our project page is at this https URL. 

**Abstract (ZH)**: perioperation：一种用于机器人数据采集的范式，通过传感和记录人类操作同时最大化数据向真实机器人传递的可行性 

---
# Towards a Unified View of Large Language Model Post-Training 

**Title (ZH)**: 面向大型语言模型后训练的统一视角 

**Authors**: Xingtai Lv, Yuxin Zuo, Youbang Sun, Hongyi Liu, Yuntian Wei, Zhekai Chen, Lixuan He, Xuekai Zhu, Kaiyan Zhang, Bingning Wang, Ning Ding, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.04419)  

**Abstract**: Two major sources of training data exist for post-training modern language models: online (model-generated rollouts) data, and offline (human or other-model demonstrations) data. These two types of data are typically used by approaches like Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT), respectively. In this paper, we show that these approaches are not in contradiction, but are instances of a single optimization process. We derive a Unified Policy Gradient Estimator, and present the calculations of a wide spectrum of post-training approaches as the gradient of a common objective under different data distribution assumptions and various bias-variance tradeoffs. The gradient estimator is constructed with four interchangeable parts: stabilization mask, reference policy denominator, advantage estimate, and likelihood gradient. Motivated by our theoretical findings, we propose Hybrid Post-Training (HPT), an algorithm that dynamically selects different training signals. HPT is designed to yield both effective exploitation of demonstration and stable exploration without sacrificing learned reasoning patterns. We provide extensive experiments and ablation studies to verify the effectiveness of our unified theoretical framework and HPT. Across six mathematical reasoning benchmarks and two out-of-distribution suites, HPT consistently surpasses strong baselines across models of varying scales and families. 

**Abstract (ZH)**: 训练现代语言模型后训练数据的两大来源及其统一优化过程：在线（模型生成滚动数据）和离线（人类或模型示范数据），这两种数据类型分别被强化学习（RL）和监督微调（SFT）等方法使用。本文展示了这些方法并非矛盾，而是同一优化过程的不同实例。我们推导出一个统一的策略梯度估计器，并呈现了在不同数据分布假设和各种偏差-方差权衡下广泛谱后训练方法的梯度计算。该梯度估计器由四个可互换的部分组成：稳定化掩码、参考策略分母、优势估计和似然梯度。受理论发现的启发，我们提出了混合后训练（HPT）算法，该算法动态选择不同的训练信号。HPT旨在充分利用示范数据的有效利用和稳定的探索，同时不牺牲学习的推理模式。我们进行了大量实验和消融研究，以验证我们统一致理论框架和HPT的有效性。在六个数学推理基准和两个离群值套件中，HPT在不同规模和家族的模型上持续超越强大基线。 

---
# No Thoughts Just AI: Biased LLM Recommendations Limit Human Agency in Resume Screening 

**Title (ZH)**: 没有思考只有AI：有偏见的LLM推荐限制了招聘简历筛选中的人类自主性 

**Authors**: Kyra Wilson, Mattea Sim, Anna-Maria Gueorguieva, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2509.04404)  

**Abstract**: In this study, we conduct a resume-screening experiment (N=528) where people collaborate with simulated AI models exhibiting race-based preferences (bias) to evaluate candidates for 16 high and low status occupations. Simulated AI bias approximates factual and counterfactual estimates of racial bias in real-world AI systems. We investigate people's preferences for White, Black, Hispanic, and Asian candidates (represented through names and affinity groups on quality-controlled resumes) across 1,526 scenarios and measure their unconscious associations between race and status using implicit association tests (IATs), which predict discriminatory hiring decisions but have not been investigated in human-AI collaboration. When making decisions without AI or with AI that exhibits no race-based preferences, people select all candidates at equal rates. However, when interacting with AI favoring a particular group, people also favor those candidates up to 90% of the time, indicating a significant behavioral shift. The likelihood of selecting candidates whose identities do not align with common race-status stereotypes can increase by 13% if people complete an IAT before conducting resume screening. Finally, even if people think AI recommendations are low quality or not important, their decisions are still vulnerable to AI bias under certain circumstances. This work has implications for people's autonomy in AI-HITL scenarios, AI and work, design and evaluation of AI hiring systems, and strategies for mitigating bias in collaborative decision-making tasks. In particular, organizational and regulatory policy should acknowledge the complex nature of AI-HITL decision making when implementing these systems, educating people who use them, and determining which are subject to oversight. 

**Abstract (ZH)**: 在本研究中，我们进行了一项简历筛选实验（N=528），参与者与表现出基于种族偏见（偏见）的模拟AI模型协作，评估16种高、低地位职业的应聘者。模拟的AI偏见接近现实世界AI系统中事实性和反事实性的种族偏见估计。我们调查了人们对白人、黑人、 Hispanic和亚裔应聘者（通过质量控制简历中的姓名和亲和群体代表）的偏好，并使用隐含关联测试（IATs）测量他们对种族与地位之间无意识的关联，这些测试可以预测歧视性招聘决策，但在人类与AI协作的情况下尚未被探讨。在没有AI或AI不表现出种族偏见的情况下，人们以相等的比例选择所有应聘者。然而，与倾向于特定群体的AI互动时，人们也会在多达90%的情况下偏好这些应聘者，表明存在显著的行为转变。如果人们在进行简历筛选前完成IAT，选择不符合常见种族地位刻板印象身份应聘者的可能性可能会增加13%。最后，即使人们认为AI推荐质量低或不重要，在某些情况下，他们的决策仍然可能受到AI偏见的影响。这项工作对AI-HITL情境中的人类自主性、AI与工作、AI招聘系统的设计与评估以及减轻协作决策任务中偏见的策略具有重要影响。特别是，组织和监管政策应在实施这些系统、教育使用这些系统的人员以及确定哪些系统需要监管时，承认AI-HITL决策的复杂性。 

---
# IPA: An Information-Preserving Input Projection Framework for Efficient Foundation Model Adaptation 

**Title (ZH)**: IPA：一种信息保留的输入投影框架，用于高效的foundation模型适配 

**Authors**: Yuan Yin, Shashanka Venkataramanan, Tuan-Hung Vu, Andrei Bursuc, Matthieu Cord  

**Link**: [PDF](https://arxiv.org/pdf/2509.04398)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, reduce adaptation cost by injecting low-rank updates into pretrained weights. However, LoRA's down-projection is randomly initialized and data-agnostic, discarding potentially useful information. Prior analyses show that this projection changes little during training, while the up-projection carries most of the adaptation, making the random input compression a performance bottleneck. We propose IPA, a feature-aware projection framework that explicitly preserves information in the reduced hidden space. In the linear case, we instantiate IPA with algorithms approximating top principal components, enabling efficient projector pretraining with negligible inference overhead. Across language and vision benchmarks, IPA consistently improves over LoRA and DoRA, achieving on average 1.5 points higher accuracy on commonsense reasoning and 2.3 points on VTAB-1k, while matching full LoRA performance with roughly half the trainable parameters when the projection is frozen. 

**Abstract (ZH)**: 基于特征感知的投影框架（IPA）：一种保留降维隐藏空间中信息的方法 

---
# SSGaussian: Semantic-Aware and Structure-Preserving 3D Style Transfer 

**Title (ZH)**: SSGaussian: 具有语义意识和结构保真的3D风格迁移 

**Authors**: Jimin Xu, Bosheng Qin, Tao Jin, Zhou Zhao, Zhenhui Ye, Jun Yu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04379)  

**Abstract**: Recent advancements in neural representations, such as Neural Radiance Fields and 3D Gaussian Splatting, have increased interest in applying style transfer to 3D scenes. While existing methods can transfer style patterns onto 3D-consistent neural representations, they struggle to effectively extract and transfer high-level style semantics from the reference style image. Additionally, the stylized results often lack structural clarity and separation, making it difficult to distinguish between different instances or objects within the 3D scene. To address these limitations, we propose a novel 3D style transfer pipeline that effectively integrates prior knowledge from pretrained 2D diffusion models. Our pipeline consists of two key stages: First, we leverage diffusion priors to generate stylized renderings of key viewpoints. Then, we transfer the stylized key views onto the 3D representation. This process incorporates two innovative designs. The first is cross-view style alignment, which inserts cross-view attention into the last upsampling block of the UNet, allowing feature interactions across multiple key views. This ensures that the diffusion model generates stylized key views that maintain both style fidelity and instance-level consistency. The second is instance-level style transfer, which effectively leverages instance-level consistency across stylized key views and transfers it onto the 3D representation. This results in a more structured, visually coherent, and artistically enriched stylization. Extensive qualitative and quantitative experiments demonstrate that our 3D style transfer pipeline significantly outperforms state-of-the-art methods across a wide range of scenes, from forward-facing to challenging 360-degree environments. Visit our project page this https URL for immersive visualization. 

**Abstract (ZH)**: 近年来，神经表示领域的最新进展，如神经光度场和3D高斯斑点化，增加了将风格转移应用于三维场景的兴趣。尽管现有方法可以在3D一致的神经表示上转移风格模式，但在有效提取和转移参考风格图像中的高层次风格语义方面存在局限性。此外，风格化结果往往缺乏结构性清晰度和分离度，使得在三维场景中区分不同的实例或对象变得困难。为了解决这些局限性，我们提出了一种新颖的三维风格转移流水线，有效地整合了预训练二维扩散模型的先验知识。该流水线包含两个关键阶段：首先，我们利用扩散先验生成关键视角的风格化渲染图。然后，将风格化的关键视图转移至三维表示。这一过程包含两个创新设计。第一个是跨视图风格对齐，该设计将跨视图注意力机制插入UNet的最后一个上采样块，允许跨多个关键视图的特征交互。这确保了扩散模型生成的风格化关键视图既能保持风格保真度，又能保持实例级一致性。第二个是实例级风格转移，该设计巧妙地利用了风格化关键视图之间的实例级一致性，并将其转移到三维表示上。这导致了结构更加清晰、视觉上更加连贯、艺术性更强的风格化效果。广泛的定性和定量实验表明，我们的三维风格转移流水线在从面向前方到具有挑战性的360度环境的各种场景中，显著优于现有最先进的方法。请访问我们的项目页面 <https://project.com> 以进行沉浸式可视化。 

---
# Parking Availability Prediction via Fusing Multi-Source Data with A Self-Supervised Learning Enhanced Spatio-Temporal Inverted Transformer 

**Title (ZH)**: 基于多源数据融合与自监督学习增强空间-时间反向变换的停车 availability 预测 

**Authors**: Yin Huang, Yongqi Dong, Youhua Tang, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.04362)  

**Abstract**: The rapid growth of private car ownership has worsened the urban parking predicament, underscoring the need for accurate and effective parking availability prediction to support urban planning and management. To address key limitations in modeling spatio-temporal dependencies and exploiting multi-source data for parking availability prediction, this study proposes a novel approach with SST-iTransformer. The methodology leverages K-means clustering to establish parking cluster zones (PCZs), extracting and integrating traffic demand characteristics from various transportation modes (i.e., metro, bus, online ride-hailing, and taxi) associated with the targeted parking lots. Upgraded on vanilla iTransformer, SST-iTransformer integrates masking-reconstruction-based pretext tasks for self-supervised spatio-temporal representation learning, and features an innovative dual-branch attention mechanism: Series Attention captures long-term temporal dependencies via patching operations, while Channel Attention models cross-variate interactions through inverted dimensions. Extensive experiments using real-world data from Chengdu, China, demonstrate that SST-iTransformer outperforms baseline deep learning models (including Informer, Autoformer, Crossformer, and iTransformer), achieving state-of-the-art performance with the lowest mean squared error (MSE) and competitive mean absolute error (MAE). Comprehensive ablation studies quantitatively reveal the relative importance of different data sources: incorporating ride-hailing data provides the largest performance gains, followed by taxi, whereas fixed-route transit features (bus/metro) contribute marginally. Spatial correlation analysis further confirms that excluding historical data from correlated parking lots within PCZs leads to substantial performance degradation, underscoring the importance of modeling spatial dependencies. 

**Abstract (ZH)**: 私家车拥有量的快速增长加剧了城市停车难题，强调了准确有效的停车可用性预测对于支持城市规划和管理的重要性。为解决模型时空依赖性建模和多源数据利用方面的关键局限性，本研究提出了一种基于SST-iTransformer的新方法。该方法利用K-means聚类建立停车集群区（PCZs），并从与目标停车场相关的各种交通模式（即地铁、公交车、在线网约车和出租车）中提取和整合交通需求特征。基于vanilla iTransformer进行升级，SST-iTransformer结合了基于掩蔽重建的先验任务进行自我监督的时空表示学习，并采用创新的双分支注意力机制：序列注意力通过分块操作捕获长时序依赖性，而通道注意力通过反转维度建模跨变量交互。使用来自中国成都的实地数据进行的广泛实验表明，SST-iTransformer在基准深度学习模型（包括Informer、Autoformer、Crossformer和iTransformer）中表现出优越性能，实现最先进的性能，具有最低的均方误差（MSE）和竞争力均绝对误差（MAE）。全面的消融研究定性揭示了不同数据源的重要性：纳入网约车数据提供了最大的性能提升，其次是出租车，而固定路线公共交通特征（公交/地铁）的贡献较小。空间相关性分析进一步证实，在PCZ中排除相关停车场的历史数据会极大地降低性能，强调了建模空间依赖性的 importance。 

---
# PARCO: Phoneme-Augmented Robust Contextual ASR via Contrastive Entity Disambiguation 

**Title (ZH)**: PARCO: 音素增强 robust 上下文 ASR 通过对比实体消歧 

**Authors**: Jiajun He, Naoki Sawada, Koichi Miyazaki, Tomoki Toda  

**Link**: [PDF](https://arxiv.org/pdf/2509.04357)  

**Abstract**: Automatic speech recognition (ASR) systems struggle with domain-specific named entities, especially homophones. Contextual ASR improves recognition but often fails to capture fine-grained phoneme variations due to limited entity diversity. Moreover, prior methods treat entities as independent tokens, leading to incomplete multi-token biasing. To address these issues, we propose Phoneme-Augmented Robust Contextual ASR via COntrastive entity disambiguation (PARCO), which integrates phoneme-aware encoding, contrastive entity disambiguation, entity-level supervision, and hierarchical entity filtering. These components enhance phonetic discrimination, ensure complete entity retrieval, and reduce false positives under uncertainty. Experiments show that PARCO achieves CER of 4.22% on Chinese AISHELL-1 and WER of 11.14% on English DATA2 under 1,000 distractors, significantly outperforming baselines. PARCO also demonstrates robust gains on out-of-domain datasets like THCHS-30 and LibriSpeech. 

**Abstract (ZH)**: 基于对比实体消歧的音素增强稳健上下文语音识别（PARCO） 

---
# AUDETER: A Large-scale Dataset for Deepfake Audio Detection in Open Worlds 

**Title (ZH)**: AUDETER：开放世界中的深度假音检测大型数据集 

**Authors**: Qizhou Wang, Hanxun Huang, Guansong Pang, Sarah Erfani, Christopher Leckie  

**Link**: [PDF](https://arxiv.org/pdf/2509.04345)  

**Abstract**: Speech generation systems can produce remarkably realistic vocalisations that are often indistinguishable from human speech, posing significant authenticity challenges. Although numerous deepfake detection methods have been developed, their effectiveness in real-world environments remains unrealiable due to the domain shift between training and test samples arising from diverse human speech and fast evolving speech synthesis systems. This is not adequately addressed by current datasets, which lack real-world application challenges with diverse and up-to-date audios in both real and deep-fake categories. To fill this gap, we introduce AUDETER (AUdio DEepfake TEst Range), a large-scale, highly diverse deepfake audio dataset for comprehensive evaluation and robust development of generalised models for deepfake audio detection. It consists of over 4,500 hours of synthetic audio generated by 11 recent TTS models and 10 vocoders with a broad range of TTS/vocoder patterns, totalling 3 million audio clips, making it the largest deepfake audio dataset by scale. Through extensive experiments with AUDETER, we reveal that i) state-of-the-art (SOTA) methods trained on existing datasets struggle to generalise to novel deepfake audio samples and suffer from high false positive rates on unseen human voice, underscoring the need for a comprehensive dataset; and ii) these methods trained on AUDETER achieve highly generalised detection performance and significantly reduce detection error rate by 44.1% to 51.6%, achieving an error rate of only 4.17% on diverse cross-domain samples in the popular In-the-Wild dataset, paving the way for training generalist deepfake audio detectors. AUDETER is available on GitHub. 

**Abstract (ZH)**: 基于音频的深度伪造检测数据集AUDETER：全面评估与鲁棒开发 

---
# From Editor to Dense Geometry Estimator 

**Title (ZH)**: 从编辑者到密集几何估计器 

**Authors**: JiYuan Wang, Chunyu Lin, Lei Sun, Rongying Liu, Lang Nie, Mingxing Li, Kang Liao, Xiangxiang Chu, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04338)  

**Abstract**: Leveraging visual priors from pre-trained text-to-image (T2I) generative models has shown success in dense prediction. However, dense prediction is inherently an image-to-image task, suggesting that image editing models, rather than T2I generative models, may be a more suitable foundation for fine-tuning.
Motivated by this, we conduct a systematic analysis of the fine-tuning behaviors of both editors and generators for dense geometry estimation. Our findings show that editing models possess inherent structural priors, which enable them to converge more stably by ``refining" their innate features, and ultimately achieve higher performance than their generative counterparts.
Based on these findings, we introduce \textbf{FE2E}, a framework that pioneeringly adapts an advanced editing model based on Diffusion Transformer (DiT) architecture for dense geometry prediction. Specifically, to tailor the editor for this deterministic task, we reformulate the editor's original flow matching loss into the ``consistent velocity" training objective. And we use logarithmic quantization to resolve the precision conflict between the editor's native BFloat16 format and the high precision demand of our tasks. Additionally, we leverage the DiT's global attention for a cost-free joint estimation of depth and normals in a single forward pass, enabling their supervisory signals to mutually enhance each other.
Without scaling up the training data, FE2E achieves impressive performance improvements in zero-shot monocular depth and normal estimation across multiple datasets. Notably, it achieves over 35\% performance gains on the ETH3D dataset and outperforms the DepthAnything series, which is trained on 100$\times$ data. The project page can be accessed \href{this https URL}{here}. 

**Abstract (ZH)**: 利用预训练文本到图像生成模型的视觉先验在密集预测任务中展现了成功。然而，密集预测本质上是图像到图像的任务，表明图像编辑模型而非文本到图像生成模型可能是适合作微调的基础。
受此启发，我们对编辑器和生成器在密集几何估计中的微调行为进行了系统分析。我们的研究结果表明，编辑模型具备内在的结构先验，使其能够通过“优化”其固有特征更稳定地收敛，并最终实现比其生成 counterparts 更高的性能。
基于这些发现，我们引入了\textbf{FE2E}框架，该框架率先将基于扩散变换器（DiT）架构的高级编辑模型适应于密集几何预测。具体而言，为了适应这一确定性任务，我们将编辑器原始的流匹配损失重新表述为“一致速度”训练目标，并使用对数量化来解决编辑器原生的BFloat16格式和任务所需高精度之间的精度冲突。此外，我们利用DiT的全局注意力，在单次前向传播中免费联合估计深度和法线，使它们的监督信号能够相互增强。
在不扩大训练数据的情况下，FE2E在多个数据集中的单目深度和法线估计中实现了显著的性能提升。值得注意的是，它在ETH3D数据集上实现了超过35%的性能提升，并优于DepthAnything系列，该系列模型是在100倍数据上进行训练的。项目页面可访问\href{this https URL}{此处}。 

---
# Decoupled Entity Representation Learning for Pinterest Ads Ranking 

**Title (ZH)**: 拆分实体表示学习在 Pinterest 广告排名中的应用 

**Authors**: Jie Liu, Yinrui Li, Jiankai Sun, Kungang Li, Han Sun, Sihan Wang, Huasen Wu, Siyuan Gao, Paulo Soares, Nan Li, Zhifang Liu, Haoyang Li, Siping Ji, Ling Leng, Prathibha Deshikachar  

**Link**: [PDF](https://arxiv.org/pdf/2509.04337)  

**Abstract**: In this paper, we introduce a novel framework following an upstream-downstream paradigm to construct user and item (Pin) embeddings from diverse data sources, which are essential for Pinterest to deliver personalized Pins and ads effectively. Our upstream models are trained on extensive data sources featuring varied signals, utilizing complex architectures to capture intricate relationships between users and Pins on Pinterest. To ensure scalability of the upstream models, entity embeddings are learned, and regularly refreshed, rather than real-time computation, allowing for asynchronous interaction between the upstream and downstream models. These embeddings are then integrated as input features in numerous downstream tasks, including ad retrieval and ranking models for CTR and CVR predictions. We demonstrate that our framework achieves notable performance improvements in both offline and online settings across various downstream tasks. This framework has been deployed in Pinterest's production ad ranking systems, resulting in significant gains in online metrics. 

**Abstract (ZH)**: 本文介绍了一种遵循上下游 paradigm 的新颖框架，从多种数据源构建用户和物品（Pin）嵌入，以有效实现 Pinterest 的个性化 Pins 和广告推荐。上游模型在包含多样化信号的大量数据源上进行训练，采用复杂的架构来捕捉 Pinterest 上用户和物品之间的复杂关系。为了确保上游模型的可扩展性，实体嵌入被学习并定期更新，而不是实时计算，从而使上游和下游模型之间能够异步交互。这些嵌入随后被整合为多种下游任务（如点击率和转换率预测的广告检索和排序模型）的输入特征。研究表明，该框架在各种下游任务的离线和在线设置中均实现了显著性能提升。该框架已在 Pinterest 的生产广告排名系统中部署，导致在线指标取得了显著提升。 

---
# Facts Fade Fast: Evaluating Memorization of Outdated Medical Knowledge in Large Language Models 

**Title (ZH)**: 事實轉瞬即逝：評估大語言模型對過時醫學知識的记忆能力 

**Authors**: Juraj Vladika, Mahdi Dhaini, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2509.04304)  

**Abstract**: The growing capabilities of Large Language Models (LLMs) show significant potential to enhance healthcare by assisting medical researchers and physicians. However, their reliance on static training data is a major risk when medical recommendations evolve with new research and developments. When LLMs memorize outdated medical knowledge, they can provide harmful advice or fail at clinical reasoning tasks. To investigate this problem, we introduce two novel question-answering (QA) datasets derived from systematic reviews: MedRevQA (16,501 QA pairs covering general biomedical knowledge) and MedChangeQA (a subset of 512 QA pairs where medical consensus has changed over time). Our evaluation of eight prominent LLMs on the datasets reveals consistent reliance on outdated knowledge across all models. We additionally analyze the influence of obsolete pre-training data and training strategies to explain this phenomenon and propose future directions for mitigation, laying the groundwork for developing more current and reliable medical AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）能力的不断增长展现了在医疗健康领域辅助医学研究人员和医生的巨大潜力。然而，它们依赖静态训练数据的特点在医疗建议随新研究和进展而演变时是一个重大风险。当LLMs记忆过时的医学知识时，可能会提供有害建议或在临床推理任务中失败。为探讨这一问题，我们引入了两个源自系统评价的问题-答案（QA）数据集：MedRevQA（涵盖一般生物医学知识的16,501个QA对）和MedChangeQA（包含512个QA对的子集，其中医学共识随时间发生了变化）。我们在数据集上对八款知名LLM进行的评估揭示了所有模型在依赖过时知识方面的一致性。此外，我们分析了过时预训练数据和训练策略的影响，以解释这一现象，并提出减轻该问题的未来方向，为开发更具时效性和可靠性的医疗AI系统奠定基础。 

---
# HumAIne-Chatbot: Real-Time Personalized Conversational AI via Reinforcement Learning 

**Title (ZH)**: HumAIne-Chatbot: 通过强化学习实现的实时个性化对话AI 

**Authors**: Georgios Makridis, Georgios Fragiadakis, Jorge Oliveira, Tomaz Saraiva, Philip Mavrepis, Georgios Fatouros, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2509.04303)  

**Abstract**: Current conversational AI systems often provide generic, one-size-fits-all interactions that overlook individual user characteristics and lack adaptive dialogue management. To address this gap, we introduce \textbf{HumAIne-chatbot}, an AI-driven conversational agent that personalizes responses through a novel user profiling framework. The system is pre-trained on a diverse set of GPT-generated virtual personas to establish a broad prior over user types. During live interactions, an online reinforcement learning agent refines per-user models by combining implicit signals (e.g. typing speed, sentiment, engagement duration) with explicit feedback (e.g., likes and dislikes). This profile dynamically informs the chatbot dialogue policy, enabling real-time adaptation of both content and style. To evaluate the system, we performed controlled experiments with 50 synthetic personas in multiple conversation domains. The results showed consistent improvements in user satisfaction, personalization accuracy, and task achievement when personalization features were enabled. Statistical analysis confirmed significant differences between personalized and nonpersonalized conditions, with large effect sizes across key metrics. These findings highlight the effectiveness of AI-driven user profiling and provide a strong foundation for future real-world validation. 

**Abstract (ZH)**: 当前的对话AI系统通常提供通用的一刀切交互，忽视了用户个体特性，缺乏适应性的对话管理。为了弥补这一不足，我们引入了\textbf{HumAIne-chatbot}，这是一种通过新颖的用户 profiling 框架个性化的AI驱动对话代理。系统在多种GPT生成的虚拟人物数据集上预先训练，以建立广泛的用户类型先验。在实时交互过程中，在线强化学习代理通过结合隐含信号（例如打字速度、情感、参与时长）和明确反馈（例如喜好和厌恶）来细化每个用户的模型。该profile动态地指导聊天机器人的对话策略，使内容和风格能够实现实时适应。为了评估该系统，我们在多个对话领域进行了50个合成人物的受控实验。结果表明，当启用个性化功能时，用户满意度、个性化准确性和任务完成度都得到了持续改进。统计分析证实了个性化和非个性化条件之间的显著差异，关键指标上的效应量很大。这些结果突显了AI驱动用户profiling的有效性，并为未来的实际场景验证奠定了坚实基础。 

---
# Reinforcement Learning for Robust Ageing-Aware Control of Li-ion Battery Systems with Data-Driven Formal Verification 

**Title (ZH)**: 基于数据驱动形式验证的鲁棒老化感知Li-ion电池系统强化学习控制 

**Authors**: Rudi Coppola, Hovsep Touloujian, Pierfrancesco Ombrini, Manuel Mazo Jr  

**Link**: [PDF](https://arxiv.org/pdf/2509.04288)  

**Abstract**: Rechargeable lithium-ion (Li-ion) batteries are a ubiquitous element of modern technology. In the last decades, the production and design of such batteries and their adjacent embedded charging and safety protocols, denoted by Battery Management Systems (BMS), has taken central stage. A fundamental challenge to be addressed is the trade-off between the speed of charging and the ageing behavior, resulting in the loss of capacity in the battery cell. We rely on a high-fidelity physics-based battery model and propose an approach to data-driven charging and safety protocol design. Following a Counterexample-Guided Inductive Synthesis scheme, we combine Reinforcement Learning (RL) with recent developments in data-driven formal methods to obtain a hybrid control strategy: RL is used to synthesise the individual controllers, and a data-driven abstraction guides their partitioning into a switched structure, depending on the initial output measurements of the battery. The resulting discrete selection among RL-based controllers, coupled with the continuous battery dynamics, realises a hybrid system. When a design meets the desired criteria, the abstraction provides probabilistic guarantees on the closed-loop performance of the cell. 

**Abstract (ZH)**: 可充放锂离子（Li-ion）电池是现代技术中的一个普遍元件。在过去的几十年里，这类电池的生产、设计及其相关的嵌入式充电和安全协议，即电池管理系统（BMS），占据了核心位置。解决的基本挑战之一是在充电速度与电池老化行为之间权衡，导致电池容量损失。我们依赖于高保真物理电池模型，并提出了一种数据驱动的充电和安全协议设计方法。采用反例引导归纳综合方案，我们将强化学习（RL）与最近的数据驱动形式化方法进展相结合，获得一种混合控制策略：使用RL合成分离的控制器，并通过电池初始输出测量的数据驱动抽象将其划分为切换结构。由此产生的基于RL的控制器的离散选择，与连续的电池动力学相结合，实现了一种混合系统。当设计方案满足预期标准时，该抽象可提供闭环性能的概率保证。 

---
# An Empirical Study of Vulnerabilities in Python Packages and Their Detection 

**Title (ZH)**: Python包中的漏洞及其检测的实证研究 

**Authors**: Haowei Quan, Junjie Wang, Xinzhe Li, Terry Yue Zhuo, Xiao Chen, Xiaoning Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.04260)  

**Abstract**: In the rapidly evolving software development landscape, Python stands out for its simplicity, versatility, and extensive ecosystem. Python packages, as units of organization, reusability, and distribution, have become a pressing concern, highlighted by the considerable number of vulnerability reports. As a scripting language, Python often cooperates with other languages for performance or interoperability. This adds complexity to the vulnerabilities inherent to Python packages, and the effectiveness of current vulnerability detection tools remains underexplored. This paper addresses these gaps by introducing PyVul, the first comprehensive benchmark suite of Python-package vulnerabilities. PyVul includes 1,157 publicly reported, developer-verified vulnerabilities, each linked to its affected packages. To accommodate diverse detection techniques, it provides annotations at both commit and function levels. An LLM-assisted data cleansing method is incorporated to improve label accuracy, achieving 100% commit-level and 94% function-level accuracy, establishing PyVul as the most precise large-scale Python vulnerability benchmark. We further carry out a distribution analysis of PyVul, which demonstrates that vulnerabilities in Python packages involve multiple programming languages and exhibit a wide variety of types. Moreover, our analysis reveals that multi-lingual Python packages are potentially more susceptible to vulnerabilities. Evaluation of state-of-the-art detectors using this benchmark reveals a significant discrepancy between the capabilities of existing tools and the demands of effectively identifying real-world security issues in Python packages. Additionally, we conduct an empirical review of the top-ranked CWEs observed in Python packages, to diagnose the fine-grained limitations of current detection tools and highlight the necessity for future advancements in the field. 

**Abstract (ZH)**: 在快速发展的软件开发landscape中，Python凭借其简洁性、灵活性和广泛的生态系统脱颖而出。Python包作为组织、重用和分发的单位，已成为一个紧迫的关注点，这体现在大量漏洞报告中。作为一种脚本语言，Python经常与其他语言合作以提高性能或实现互操作性。这增加了Python包固有漏洞的复杂性，当前漏洞检测工具的有效性也未得到充分探索。本文通过介绍PyVul，提出了首个全面的Python包漏洞基准套件，填补了这一空白。PyVul包含1,157个公开报告的、开发者验证的漏洞，每个漏洞都链接到受影响的包。为了适应多样化的检测技术，它在提交和函数级别提供了注解。引入了基于LLM的数据清洗方法，以提高标签准确性，实现了100%的提交级别和94%的函数级别准确性，使得PyVul成为最精确的大规模Python漏洞基准。进一步对PyVul进行了分布分析，结果显示Python包中的漏洞涉及多种编程语言，并表现出多种类型。此外，我们的分析表明，多语言Python包可能更易受到漏洞的影响。使用此基准评估最先进的检测器揭示了现有工具的能力与有效识别Python包实际安全问题的需求之间存在显著差距。此外，我们还对Python包中观察到的顶级CWE进行实证审查，以诊断当前检测工具的细粒度限制，并指出未来该领域需要的进步。 

---
# How many patients could we save with LLM priors? 

**Title (ZH)**: 使用LLM先验，我们能拯救多少患者？ 

**Authors**: Shota Arai, David Selby, Andrew Vargo, Sebastian Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.04250)  

**Abstract**: Imagine a world where clinical trials need far fewer patients to achieve the same statistical power, thanks to the knowledge encoded in large language models (LLMs). We present a novel framework for hierarchical Bayesian modeling of adverse events in multi-center clinical trials, leveraging LLM-informed prior distributions. Unlike data augmentation approaches that generate synthetic data points, our methodology directly obtains parametric priors from the model. Our approach systematically elicits informative priors for hyperparameters in hierarchical Bayesian models using a pre-trained LLM, enabling the incorporation of external clinical expertise directly into Bayesian safety modeling. Through comprehensive temperature sensitivity analysis and rigorous cross-validation on real-world clinical trial data, we demonstrate that LLM-derived priors consistently improve predictive performance compared to traditional meta-analytical approaches. This methodology paves the way for more efficient and expert-informed clinical trial design, enabling substantial reductions in the number of patients required to achieve robust safety assessment and with the potential to transform drug safety monitoring and regulatory decision making. 

**Abstract (ZH)**: 借助大型语言模型知识实现临床试验中同一统计功效所需的患者数量大幅减少的前景：基于大型语言模型的先验分布的多层次贝叶斯建模框架 

---
# Learning Active Perception via Self-Evolving Preference Optimization for GUI Grounding 

**Title (ZH)**: 基于自我进化偏好优化的学习主动感知方法及其在GUI定位中的应用 

**Authors**: Wanfu Wang, Qipeng Huang, Guangquan Xue, Xiaobo Liang, Juntao Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.04243)  

**Abstract**: Vision Language Models (VLMs) have recently achieved significant progress in bridging visual perception and linguistic reasoning. Recently, OpenAI o3 model introduced a zoom-in search strategy that effectively elicits active perception capabilities in VLMs, improving downstream task performance. However, enabling VLMs to reason effectively over appropriate image regions remains a core challenge in GUI grounding, particularly under high-resolution inputs and complex multi-element visual interactions. In this work, we propose LASER, a self-evolving framework that progressively endows VLMs with multi-step perception capabilities, enabling precise coordinate prediction. Specifically, our approach integrate Monte Carlo quality estimation with Intersection-over-Union (IoU)-based region quality evaluation to jointly encourage both accuracy and diversity in constructing high-quality preference data. This combination explicitly guides the model to focus on instruction-relevant key regions while adaptively allocating reasoning steps based on task complexity. Comprehensive experiments on the ScreenSpot Pro and ScreenSpot-v2 benchmarks demonstrate consistent performance gains, validating the effectiveness of our method. Furthermore, when fine-tuned on GTA1-7B, LASER achieves a score of 55.7 on the ScreenSpot-Pro benchmark, establishing a new state-of-the-art (SoTA) among 7B-scale models. 

**Abstract (ZH)**: Vision-Language Models (VLMs)在视觉感知与语言推理融合方面取得了显著进展。最近，OpenAI的o3模型引入了一种缩放搜索策略，有效激发了VLMs的主动感知能力，提高了下游任务性能。然而，在GUI语义理解中，特别是在高分辨率输入和复杂多元素视觉交互下，使VLMs能够有效地在适当图像区域进行推理仍然是一个核心挑战。在本工作中，我们提出了一种自演化框架LASER，该框架逐步赋予VLMs多步感知能力，实现精确坐标预测。具体而言，我们的方法通过将蒙特卡洛质量估计与基于交并比(IoU)的区域质量评估相结合，共同促进构建高质量偏好数据的准确性和多样性。这种结合明确引导模型关注指令相关的关键区域，并根据任务复杂性自适应分配推理步骤。在ScreenSpot Pro和ScreenSpot-v2基准上的综合性实验显示了持续的性能提升，验证了我们方法的有效性。此外，通过在GTA1-7B上微调后，LASER在ScreenSpot-Pro基准上获得了55.7的分数，成为7B规模模型中的新最佳表现（SoTA）。 

---
# MAGneT: Coordinated Multi-Agent Generation of Synthetic Multi-Turn Mental Health Counseling Sessions 

**Title (ZH)**: MAGneT: 协同多Agent生成合成多轮心理康复会谈 

**Authors**: Aishik Mandal, Tanmoy Chakraborty, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2509.04183)  

**Abstract**: The growing demand for scalable psychological counseling highlights the need for fine-tuning open-source Large Language Models (LLMs) with high-quality, privacy-compliant data, yet such data remains scarce. Here we introduce MAGneT, a novel multi-agent framework for synthetic psychological counseling session generation that decomposes counselor response generation into coordinated sub-tasks handled by specialized LLM agents, each modeling a key psychological technique. Unlike prior single-agent approaches, MAGneT better captures the structure and nuance of real counseling. In addition, we address inconsistencies in prior evaluation protocols by proposing a unified evaluation framework integrating diverse automatic and expert metrics. Furthermore, we expand the expert evaluations from four aspects of counseling in previous works to nine aspects, enabling a more thorough and robust assessment of data quality. Empirical results show that MAGneT significantly outperforms existing methods in quality, diversity, and therapeutic alignment of the generated counseling sessions, improving general counseling skills by 3.2% and CBT-specific skills by 4.3% on average on cognitive therapy rating scale (CTRS). Crucially, experts prefer MAGneT-generated sessions in 77.2% of cases on average across all aspects. Moreover, fine-tuning an open-source model on MAGneT-generated sessions shows better performance, with improvements of 6.3% on general counseling skills and 7.3% on CBT-specific skills on average on CTRS over those fine-tuned with sessions generated by baseline methods. We also make our code and data public. 

**Abstract (ZH)**: 一种新的多Agent框架MAGneT及其在合成心理咨询服务生成中的应用 

---
# VisioFirm: Cross-Platform AI-assisted Annotation Tool for Computer Vision 

**Title (ZH)**: VisioFirm：跨平台的计算机视觉人工智能辅助标注工具 

**Authors**: Safouane El Ghazouali, Umberto Michelucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.04180)  

**Abstract**: AI models rely on annotated data to learn pattern and perform prediction. Annotation is usually a labor-intensive step that require associating labels ranging from a simple classification label to more complex tasks such as object detection, oriented bounding box estimation, and instance segmentation. Traditional tools often require extensive manual input, limiting scalability for large datasets. To address this, we introduce VisioFirm, an open-source web application designed to streamline image labeling through AI-assisted automation. VisioFirm integrates state-of-the-art foundation models into an interface with a filtering pipeline to reduce human-in-the-loop efforts. This hybrid approach employs CLIP combined with pre-trained detectors like Ultralytics models for common classes and zero-shot models such as Grounding DINO for custom labels, generating initial annotations with low-confidence thresholding to maximize recall. Through this framework, when tested on COCO-type of classes, initial prediction have been proven to be mostly correct though the users can refine these via interactive tools supporting bounding boxes, oriented bounding boxes, and polygons. Additionally, VisioFirm has on-the-fly segmentation powered by Segment Anything accelerated through WebGPU for browser-side efficiency. The tool supports multiple export formats (YOLO, COCO, Pascal VOC, CSV) and operates offline after model caching, enhancing accessibility. VisioFirm demonstrates up to 90\% reduction in manual effort through benchmarks on diverse datasets, while maintaining high annotation accuracy via clustering of connected CLIP-based disambiguate components and IoU-graph for redundant detection suppression. VisioFirm can be accessed from \href{this https URL}{this https URL}. 

**Abstract (ZH)**: AI模型依赖标注数据学习模式和进行预测。标注通常是一个劳动密集型步骤，要求关联从简单分类标签到复杂任务（如对象检测、有向边界框估计和实例分割）的标签。传统工具往往需要大量手动输入，限制了大规模数据集的可扩展性。为了解决这一问题，我们介绍了VisioFirm，一个开源网页应用，旨在通过AI辅助自动化来简化图像标注过程。VisioFirm将最先进的基础模型集成到一个包含过滤管道的界面中，以减少人工在环努力。这种混合方法结合了CLIP与预训练检测器（如Ultralytics模型）以及零样本模型（如Grounding DINO），通过低置信度阈值生成初始标注以最大化召回率。通过该框架，在针对COCO类型类别的测试中，初始预测已被证明大多数是正确的，虽然用户可以通过支持边界框、有向边界框和多边形的交互工具进行进一步精修。此外，VisioFirm通过WebGPU加速的Segment Anything实现了即时分割，增强了浏览器端效率。该工具支持多种导出格式（YOLO、COCO、Pascal VOC、CSV），并通过模型缓存后离线运行，增强了可访问性。VisioFirm通过对连接的CLIP基体消歧组件的聚类和IoU图来减少冗余检测，展示了在多样数据集基准测试中高达90%的手动劳动减少，同时保持高标注准确性。VisioFirm可从\href{this https URL}{this https URL}访问。 

---
# Crossing the Species Divide: Transfer Learning from Speech to Animal Sounds 

**Title (ZH)**: 跨越物种界线：从语音到动物声音的迁移学习 

**Authors**: Jules Cauzinille, Marius Miron, Olivier Pietquin, Masato Hagiwara, Ricard Marxer, Arnaud Rey, Benoit Favre  

**Link**: [PDF](https://arxiv.org/pdf/2509.04166)  

**Abstract**: Self-supervised speech models have demonstrated impressive performance in speech processing, but their effectiveness on non-speech data remains underexplored. We study the transfer learning capabilities of such models on bioacoustic detection and classification tasks. We show that models such as HuBERT, WavLM, and XEUS can generate rich latent representations of animal sounds across taxa. We analyze the models properties with linear probing on time-averaged representations. We then extend the approach to account for the effect of time-wise information with other downstream architectures. Finally, we study the implication of frequency range and noise on performance. Notably, our results are competitive with fine-tuned bioacoustic pre-trained models and show the impact of noise-robust pre-training setups. These findings highlight the potential of speech-based self-supervised learning as an efficient framework for advancing bioacoustic research. 

**Abstract (ZH)**: 非语音数据上自我监督语音模型的迁移学习能力研究：以生物声学检测与分类为例 

---
# YOLO Ensemble for UAV-based Multispectral Defect Detection in Wind Turbine Components 

**Title (ZH)**: 基于UAV的风力发电机组件多光谱缺陷检测YOLO集成方法 

**Authors**: Serhii Svystun, Pavlo Radiuk, Oleksandr Melnychenko, Oleg Savenko, Anatoliy Sachenko  

**Link**: [PDF](https://arxiv.org/pdf/2509.04156)  

**Abstract**: Unmanned aerial vehicles (UAVs) equipped with advanced sensors have opened up new opportunities for monitoring wind power plants, including blades, towers, and other critical components. However, reliable defect detection requires high-resolution data and efficient methods to process multispectral imagery. In this research, we aim to enhance defect detection accuracy through the development of an ensemble of YOLO-based deep learning models that integrate both visible and thermal channels. We propose an ensemble approach that integrates a general-purpose YOLOv8 model with a specialized thermal model, using a sophisticated bounding box fusion algorithm to combine their predictions. Our experiments show this approach achieves a mean Average Precision (mAP@.5) of 0.93 and an F1-score of 0.90, outperforming a standalone YOLOv8 model, which scored an mAP@.5 of 0.91. These findings demonstrate that combining multiple YOLO architectures with fused multispectral data provides a more reliable solution, improving the detection of both visual and thermal defects. 

**Abstract (ZH)**: 装备有高级传感器的无人机为监测风力发电厂，包括叶片、塔和其他关键组件，开辟了新机会。然而，可靠的缺陷检测需要高分辨率数据和高效的多光谱图像处理方法。在本研究中，我们通过开发结合可见光和热红外通道的YOLO基集成深度学习模型，旨在提升缺陷检测的准确性。我们提出了一种集成方法，结合了一种通用的YOLOv8模型和一种专门的热红外模型，使用复杂的边界框融合算法来结合它们的预测。实验结果显示，该方法在mAP@.5上的平均精度为0.93，F1分数为0.90，优于单独使用的YOLOv8模型，后者在mAP@.5上的得分为0.91。这些发现表明，结合多个YOLO架构和融合的多光谱数据，提供了更可靠的方法，提高了对视觉和热缺陷的检测能力。 

---
# Attention as an Adaptive Filter 

**Title (ZH)**: Attention作为一种自适应滤波器 

**Authors**: Peter Racioppo  

**Link**: [PDF](https://arxiv.org/pdf/2509.04154)  

**Abstract**: We introduce Adaptive Filter Attention (AFA), a novel attention mechanism that incorporates a learnable dynamics model directly into the computation of attention weights. Rather than comparing queries and keys directly, we model the input sequence as discrete observations of a linear stochastic differential equation (SDE). By imposing a linear dynamics model with simultaneously diagonalizable state matrices and noise covariances, we can make use of a closed-form solution to the differential Lyapunov equation to efficiently propagate pairwise uncertainties through the dynamics. Attention naturally arises as the maximum likelihood solution for this linear SDE, with attention weights corresponding to robust residual-based reweightings of the propagated pairwise precisions. Imposing an additional constraint on the state matrix's eigenvalues leads to a simplified variant with the same computational and memory complexity as standard attention. In the limit of vanishing dynamics and process noise, and using a small-angle approximation, we recover ordinary dot-product attention. 

**Abstract (ZH)**: 自适应滤波注意力机制：直接将可学习的动力学模型融入注意力权重的计算中 

---
# TAGAL: Tabular Data Generation using Agentic LLM Methods 

**Title (ZH)**: TAGAL: 使用代理型LLM方法生成表格数据 

**Authors**: Benoît Ronval, Pierre Dupont, Siegfried Nijssen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04152)  

**Abstract**: The generation of data is a common approach to improve the performance of machine learning tasks, among which is the training of models for classification. In this paper, we present TAGAL, a collection of methods able to generate synthetic tabular data using an agentic workflow. The methods leverage Large Language Models (LLMs) for an automatic and iterative process that uses feedback to improve the generated data without any further LLM training. The use of LLMs also allows for the addition of external knowledge in the generation process. We evaluate TAGAL across diverse datasets and different aspects of quality for the generated data. We look at the utility of downstream ML models, both by training classifiers on synthetic data only and by combining real and synthetic data. Moreover, we compare the similarities between the real and the generated data. We show that TAGAL is able to perform on par with state-of-the-art approaches that require LLM training and generally outperforms other training-free approaches. These findings highlight the potential of agentic workflow and open new directions for LLM-based data generation methods. 

**Abstract (ZH)**: TAGAL：一种基于代理工作流的合成表数据生成方法 

---
# Enhancing Technical Documents Retrieval for RAG 

**Title (ZH)**: 增强技术文档检索以支持RAG 

**Authors**: Songjiang Lai, Tsun-Hin Cheung, Ka-Chun Fung, Kaiwen Xue, Kwan-Ho Lin, Yan-Ming Choi, Vincent Ng, Kin-Man Lam  

**Link**: [PDF](https://arxiv.org/pdf/2509.04139)  

**Abstract**: In this paper, we introduce Technical-Embeddings, a novel framework designed to optimize semantic retrieval in technical documentation, with applications in both hardware and software development. Our approach addresses the challenges of understanding and retrieving complex technical content by leveraging the capabilities of Large Language Models (LLMs). First, we enhance user queries by generating expanded representations that better capture user intent and improve dataset diversity, thereby enriching the fine-tuning process for embedding models. Second, we apply summary extraction techniques to encode essential contextual information, refining the representation of technical documents. To further enhance retrieval performance, we fine-tune a bi-encoder BERT model using soft prompting, incorporating separate learning parameters for queries and document context to capture fine-grained semantic nuances. We evaluate our approach on two public datasets, RAG-EDA and Rust-Docs-QA, demonstrating that Technical-Embeddings significantly outperforms baseline models in both precision and recall. Our findings highlight the effectiveness of integrating query expansion and contextual summarization to enhance information access and comprehension in technical domains. This work advances the state of Retrieval-Augmented Generation (RAG) systems, offering new avenues for efficient and accurate technical document retrieval in engineering and product development workflows. 

**Abstract (ZH)**: Technical-Embeddings：一种优化技术文档检索的新型框架及其在硬件和软件开发中的应用 

---
# Simplicity Lies in the Eye of the Beholder: A Strategic Perspective on Controllers in Reactive Synthesis 

**Title (ZH)**: 简而言之在于观察者的眼中：反应性综合中控制器的战略视角 

**Authors**: Mickael Randour  

**Link**: [PDF](https://arxiv.org/pdf/2509.04129)  

**Abstract**: In the game-theoretic approach to controller synthesis, we model the interaction between a system to be controlled and its environment as a game between these entities, and we seek an appropriate (e.g., winning or optimal) strategy for the system. This strategy then serves as a formal blueprint for a real-world controller. A common belief is that simple (e.g., using limited memory) strategies are better: corresponding controllers are easier to conceive and understand, and cheaper to produce and maintain.
This invited contribution focuses on the complexity of strategies in a variety of synthesis contexts. We discuss recent results concerning memory and randomness, and take a brief look at what lies beyond our traditional notions of complexity for strategies. 

**Abstract (ZH)**: 在控制器合成的博弈论方法中，我们将被控系统与其环境之间的互动建模为这些实体之间的博弈，并寻求系统的一个适当策略（例如，获胜或最优策略）。然后，该策略作为现实世界控制器的形式蓝图。一种普遍认为简单的策略（例如，使用有限内存）更好：相应的控制器更易于构思和理解，并且成本较低，维护也更简单。

本贡献重点关注策略在各种合成上下文中的复杂性。我们讨论了关于记忆和随机性的最新成果，并简要探讨了超越我们传统策略复杂性观念的内容。 

---
# MEPG:Multi-Expert Planning and Generation for Compositionally-Rich Image Generation 

**Title (ZH)**: MEPG：多专家规划与生成在组合丰富的图像生成中的应用 

**Authors**: Yuan Zhao, Liu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04126)  

**Abstract**: Text-to-image diffusion models have achieved remarkable image quality, but they still struggle with complex, multiele ment prompts, and limited stylistic diversity. To address these limitations, we propose a Multi-Expert Planning and Gen eration Framework (MEPG) that synergistically integrates position- and style-aware large language models (LLMs) with spatial-semantic expert modules. The framework comprises two core components: (1) a Position-Style-Aware (PSA) module that utilizes a supervised fine-tuned LLM to decom pose input prompts into precise spatial coordinates and style encoded semantic instructions; and (2) a Multi-Expert Dif fusion (MED) module that implements cross-region genera tion through dynamic expert routing across both local regions and global areas. During the generation process for each lo cal region, specialized models (e.g., realism experts, styliza tion specialists) are selectively activated for each spatial par tition via attention-based gating mechanisms. The architec ture supports lightweight integration and replacement of ex pert models, providing strong extensibility. Additionally, an interactive interface enables real-time spatial layout editing and per-region style selection from a portfolio of experts. Ex periments show that MEPG significantly outperforms base line models with the same backbone in both image quality
and style diversity. 

**Abstract (ZH)**: 基于多专家规划与生成框架的文本到图像扩散模型 

---
# EHVC: Efficient Hierarchical Reference and Quality Structure for Neural Video Coding 

**Title (ZH)**: EHVC:有效分层参考和质量结构的神经视频编码 

**Authors**: Junqi Liao, Yaojun Wu, Chaoyi Lin, Zhipin Deng, Li Li, Dong Liu, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.04118)  

**Abstract**: Neural video codecs (NVCs), leveraging the power of end-to-end learning, have demonstrated remarkable coding efficiency improvements over traditional video codecs. Recent research has begun to pay attention to the quality structures in NVCs, optimizing them by introducing explicit hierarchical designs. However, less attention has been paid to the reference structure design, which fundamentally should be aligned with the hierarchical quality structure. In addition, there is still significant room for further optimization of the hierarchical quality structure. To address these challenges in NVCs, we propose EHVC, an efficient hierarchical neural video codec featuring three key innovations: (1) a hierarchical multi-reference scheme that draws on traditional video codec design to align reference and quality structures, thereby addressing the reference-quality mismatch; (2) a lookahead strategy to utilize an encoder-side context from future frames to enhance the quality structure; (3) a layer-wise quality scale with random quality training strategy to stabilize quality structures during inference. With these improvements, EHVC achieves significantly superior performance to the state-of-the-art NVCs. Code will be released in: this https URL. 

**Abstract (ZH)**: 基于神经网络的高效分层次视频编码器（EHVC）：一种结合传统视频编码设计的高效分层次多参考方案、前瞻策略及分层质量尺度的神经视频编解码器 

---
# RepoDebug: Repository-Level Multi-Task and Multi-Language Debugging Evaluation of Large Language Models 

**Title (ZH)**: RepoDebug: 多任务和多语言仓库级别大型语言模型调试评估 

**Authors**: Jingjing Liu, Zeming Liu, Zihao Cheng, Mengliang He, Xiaoming Shi, Yuhang Guo, Xiangrong Zhu, Yuanfang Guo, Yunhong Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04078)  

**Abstract**: Large Language Models (LLMs) have exhibited significant proficiency in code debugging, especially in automatic program repair, which may substantially reduce the time consumption of developers and enhance their efficiency. Significant advancements in debugging datasets have been made to promote the development of code debugging. However, these datasets primarily focus on assessing the LLM's function-level code repair capabilities, neglecting the more complex and realistic repository-level scenarios, which leads to an incomplete understanding of the LLM's challenges in repository-level debugging. While several repository-level datasets have been proposed, they often suffer from limitations such as limited diversity of tasks, languages, and error types. To mitigate this challenge, this paper introduces RepoDebug, a multi-task and multi-language repository-level code debugging dataset with 22 subtypes of errors that supports 8 commonly used programming languages and 3 debugging tasks. Furthermore, we conduct evaluation experiments on 10 LLMs, where Claude 3.5 Sonnect, the best-performing model, still cannot perform well in repository-level debugging. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码调试，尤其是在自动程序修复方面展现了显著的能力，这可能大幅减少开发人员的时间消耗并提高其效率。为了促进代码调试的发展，已经在调试数据集方面取得了显著进展。然而，这些数据集主要侧重于评估LLM的功能级代码修复能力，忽视了更复杂和现实的仓库级场景，导致对LLM在仓库级调试方面的挑战缺乏全面理解。虽然已经提出了一些仓库级数据集，但它们往往存在任务、语言和错误类型多样性有限等局限。为缓解这一挑战，本文引入了RepoDebug，这是一个多任务、多语言的仓库级代码调试数据集，包含22种错误亚型，支持8种常用编程语言和3种调试任务。此外，我们在10种LLM上进行了评估实验，其中表现最佳的模型Claude 3.5 Sonnect在仓库级调试中仍然表现不佳。 

---
# Keypoint-based Diffusion for Robotic Motion Planning on the NICOL Robot 

**Title (ZH)**: 基于关键点的扩散模型在NICOL机器人运动规划中的应用 

**Authors**: Lennart Clasmeier, Jan-Gerrit Habekost, Connor Gäde, Philipp Allgeuer, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2509.04076)  

**Abstract**: We propose a novel diffusion-based action model for robotic motion planning. Commonly, established numerical planning approaches are used to solve general motion planning problems, but have significant runtime requirements. By leveraging the power of deep learning, we are able to achieve good results in a much smaller runtime by learning from a dataset generated by these planners. While our initial model uses point cloud embeddings in the input to predict keypoint-based joint sequences in its output, we observed in our ablation study that it remained challenging to condition the network on the point cloud embeddings. We identified some biases in our dataset and refined it, which improved the model's performance. Our model, even without the use of the point cloud encodings, outperforms numerical models by an order of magnitude regarding the runtime, while reaching a success rate of up to 90% of collision free solutions on the test set. 

**Abstract (ZH)**: 我们提出了一种基于扩散的动作模型用于机器人运动规划。通常，现有的数值规划方法被用于解决一般的运动规划问题，但这些方法需要显著的运行时间。通过利用深度学习的能力，我们能够在学习来自这些规划器生成的数据集之后，在更小的运行时间内取得良好的结果。虽然最初模型在输入中使用点云嵌入来预测关键点基于的关节序列，但在去噪研究中我们发现难以条件化网络依赖于点云嵌入。我们识别并修正了数据集中的某些偏差，从而提高了模型的性能。即使不使用点云编码，我们的模型在运行时间上比数值模型快一个数量级，在测试集上无碰撞解决方案的成功率可达90%。 

---
# Neural Video Compression with In-Loop Contextual Filtering and Out-of-Loop Reconstruction Enhancement 

**Title (ZH)**: 基于循环内上下文过滤和循环外重建增强的神经视频压缩 

**Authors**: Yaojun Wu, Chaoyi Lin, Yiming Wang, Semih Esenlik, Zhaobin Zhang, Kai Zhang, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04051)  

**Abstract**: This paper explores the application of enhancement filtering techniques in neural video compression. Specifically, we categorize these techniques into in-loop contextual filtering and out-of-loop reconstruction enhancement based on whether the enhanced representation affects the subsequent coding loop. In-loop contextual filtering refines the temporal context by mitigating error propagation during frame-by-frame encoding. However, its influence on both the current and subsequent frames poses challenges in adaptively applying filtering throughout the sequence. To address this, we introduce an adaptive coding decision strategy that dynamically determines filtering application during encoding. Additionally, out-of-loop reconstruction enhancement is employed to refine the quality of reconstructed frames, providing a simple yet effective improvement in coding efficiency. To the best of our knowledge, this work presents the first systematic study of enhancement filtering in the context of conditional-based neural video compression. Extensive experiments demonstrate a 7.71% reduction in bit rate compared to state-of-the-art neural video codecs, validating the effectiveness of the proposed approach. 

**Abstract (ZH)**: 本文探索了增强过滤技术在神经视频压缩中的应用。具体而言，我们根据增强表示是否影响后续编码循环，将这些技术分为循环内上下文过滤和循环外重构增强。循环内上下文过滤通过减轻逐帧编码过程中的误差传播来细化时间上下文，但其对当前帧和后续帧的影响为在整个序列中适应性应用过滤带来了挑战。为此，我们提出了一种自适应编码决策策略，以动态确定编码过程中过滤的应用。此外，循环外重构增强用于细化重建帧的质量，提供了一种简单而有效的编码效率改进方法。据我们所知，本工作是首次对基于条件的神经视频压缩中增强过滤的系统研究。详尽的实验结果表明，相比于最先进的神经视频编解码器，比特率降低了7.71%，验证了所提方法的有效性。 

---
# On Robustness and Reliability of Benchmark-Based Evaluation of LLMs 

**Title (ZH)**: 基于基准的大型语言模型评价的稳健性与可靠性 

**Authors**: Riccardo Lunardi, Vincenzo Della Mea, Stefano Mizzaro, Kevin Roitero  

**Link**: [PDF](https://arxiv.org/pdf/2509.04013)  

**Abstract**: Large Language Models (LLMs) effectiveness is usually evaluated by means of benchmarks such as MMLU, ARC-C, or HellaSwag, where questions are presented in their original wording, thus in a fixed, standardized format. However, real-world applications involve linguistic variability, requiring models to maintain their effectiveness across diverse rewordings of the same question or query. In this study, we systematically assess the robustness of LLMs to paraphrased benchmark questions and investigate whether benchmark-based evaluations provide a reliable measure of model capabilities. We systematically generate various paraphrases of all the questions across six different common benchmarks, and measure the resulting variations in effectiveness of 34 state-of-the-art LLMs, of different size and effectiveness. Our findings reveal that while LLM rankings remain relatively stable across paraphrased inputs, absolute effectiveness scores change, and decline significantly. This suggests that LLMs struggle with linguistic variability, raising concerns about their generalization abilities and evaluation methodologies. Furthermore, the observed performance drop challenges the reliability of benchmark-based evaluations, indicating that high benchmark scores may not fully capture a model's robustness to real-world input variations. We discuss the implications of these findings for LLM evaluation methodologies, emphasizing the need for robustness-aware benchmarks that better reflect practical deployment scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通过MMLU、ARC-C或HellaSwag等基准评估时，通常以固定的标准格式呈现问题。然而，实际应用涉及语言的多样性，要求模型在不同表述方式的问题或查询上保持有效性。本研究系统地评估LLMs对重述基准问题的稳健性，并探讨基于基准的评估是否可靠地衡量模型能力。我们系统地生成六个常见基准下所有问题的各种重述，并测量34种不同规模和有效性的最先进的LLMs在这些重述问题上的效果变化。我们的研究发现，在重述输入下，LLM的排名相对稳定，但绝对效果得分下降显著，这表明LLMs在面对语言多样性时存在困难，这引发了对其泛化能力和评估方法的关注。此外，观察到的表现下降挑战了基于基准的评估的可靠性，表明高基准分数可能未能充分捕捉模型对实际输入变异的稳健性。我们讨论了这些发现对LLM评估方法的影响，强调了需要关注健壮性的基准，以更好地反映实际部署场景。 

---
# NER Retriever: Zero-Shot Named Entity Retrieval with Type-Aware Embeddings 

**Title (ZH)**: 命名实体检索器：基于类型感知嵌入的零-shot 命名实体检索 

**Authors**: Or Shachar, Uri Katz, Yoav Goldberg, Oren Glickman  

**Link**: [PDF](https://arxiv.org/pdf/2509.04011)  

**Abstract**: We present NER Retriever, a zero-shot retrieval framework for ad-hoc Named Entity Retrieval, a variant of Named Entity Recognition (NER), where the types of interest are not provided in advance, and a user-defined type description is used to retrieve documents mentioning entities of that type. Instead of relying on fixed schemas or fine-tuned models, our method builds on internal representations of large language models (LLMs) to embed both entity mentions and user-provided open-ended type descriptions into a shared semantic space. We show that internal representations, specifically the value vectors from mid-layer transformer blocks, encode fine-grained type information more effectively than commonly used top-layer embeddings. To refine these representations, we train a lightweight contrastive projection network that aligns type-compatible entities while separating unrelated types. The resulting entity embeddings are compact, type-aware, and well-suited for nearest-neighbor search. Evaluated on three benchmarks, NER Retriever significantly outperforms both lexical and dense sentence-level retrieval baselines. Our findings provide empirical support for representation selection within LLMs and demonstrate a practical solution for scalable, schema-free entity retrieval. The NER Retriever Codebase is publicly available at this https URL 

**Abstract (ZH)**: NER Retriever：一种零样本命名实体检索框架 

---
# Detecting Regional Spurious Correlations in Vision Transformers via Token Discarding 

**Title (ZH)**: 基于tokens弃用检测视觉变换器中的区域虚假相关性 

**Authors**: Solha Kang, Esla Timothy Anzaku, Wesley De Neve, Arnout Van Messem, Joris Vankerschaver, Francois Rameau, Utku Ozbulak  

**Link**: [PDF](https://arxiv.org/pdf/2509.04009)  

**Abstract**: Due to their powerful feature association capabilities, neural network-based computer vision models have the ability to detect and exploit unintended patterns within the data, potentially leading to correct predictions based on incorrect or unintended but statistically relevant signals. These clues may vary from simple color aberrations to small texts within the image. In situations where these unintended signals align with the predictive task, models can mistakenly link these features with the task and rely on them for making predictions. This phenomenon is referred to as spurious correlations, where patterns appear to be associated with the task but are actually coincidental. As a result, detection and mitigation of spurious correlations have become crucial tasks for building trustworthy, reliable, and generalizable machine learning models. In this work, we present a novel method to detect spurious correlations in vision transformers, a type of neural network architecture that gained significant popularity in recent years. Using both supervised and self-supervised trained models, we present large-scale experiments on the ImageNet dataset demonstrating the ability of the proposed method to identify spurious correlations. We also find that, even if the same architecture is used, the training methodology has a significant impact on the model's reliance on spurious correlations. Furthermore, we show that certain classes in the ImageNet dataset contain spurious signals that are easily detected by the models and discuss the underlying reasons for those spurious signals. In light of our findings, we provide an exhaustive list of the aforementioned images and call for caution in their use in future research efforts. Lastly, we present a case study investigating spurious signals in invasive breast mass classification, grounding our work in real-world scenarios. 

**Abstract (ZH)**: 基于神经网络的计算机视觉模型由于其强大的特征关联能力，能够检测和利用数据中的未预期模式， potentially 基于错误或未预期但统计上相关的信号进行正确预测。这些线索可以从简单的颜色偏差到图像内的小型文本不等。当这些未预期的信号与预测任务相一致时，模型可能会错误地将这些特征与任务关联起来并依赖于它们进行预测。这一现象被称为伪相关，即模式看似与任务相关但实际上只是巧合。因此，检测和缓解伪相关已成为构建可信赖、可靠且普适的机器学习模型的关键任务。在本工作中，我们提出了一种新颖的方法来检测视觉变换器中的伪相关，这是一种近年来广受青睐的神经网络架构。我们使用监督训练和自监督训练模型，在ImageNet数据集上进行了大规模实验，展示了所提出方法识别伪相关的能 力。我们还发现，即使使用相同的架构，训练方法对模型依赖伪相关的影响也十分显著。此外，我们展示了ImageNet数据集中的某些类别包含易于被模型检测到的伪信号，并讨论了这些伪信号背后的原因。基于我们的发现，我们列出了上述图像的详尽列表，并呼吁在未来的研究中对此保持警惕。最后，我们通过一个案例研究探讨了侵入性乳腺肿块分类中的伪信号，将我们的工作与实际场景相结合。 

---
# RTQA : Recursive Thinking for Complex Temporal Knowledge Graph Question Answering with Large Language Models 

**Title (ZH)**: RTQA：递归思考在大规模语言模型辅助下的复杂时空知识图谱问答 

**Authors**: Zhaoyan Gong, Juan Li, Zhiqiang Liu, Lei Liang, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03995)  

**Abstract**: Current temporal knowledge graph question answering (TKGQA) methods primarily focus on implicit temporal constraints, lacking the capability of handling more complex temporal queries, and struggle with limited reasoning abilities and error propagation in decomposition frameworks. We propose RTQA, a novel framework to address these challenges by enhancing reasoning over TKGs without requiring training. Following recursive thinking, RTQA recursively decomposes questions into sub-problems, solves them bottom-up using LLMs and TKG knowledge, and employs multi-path answer aggregation to improve fault tolerance. RTQA consists of three core components: the Temporal Question Decomposer, the Recursive Solver, and the Answer Aggregator. Experiments on MultiTQ and TimelineKGQA benchmarks demonstrate significant Hits@1 improvements in "Multiple" and "Complex" categories, outperforming state-of-the-art methods. Our code and data are available at this https URL. 

**Abstract (ZH)**: 当前的时间知识图谱问答（TKGQA）方法主要集中在隐式时间约束上，缺乏处理复杂时间查询的能力，并且在分解框架中表现出有限的推理能力和错误传播问题。我们提出了一种新的RTQA框架，通过增强对时间知识图谱的推理能力来应对这些挑战，无需训练。RTQA 通过递归思考将问题递归分解为子问题，并自底向上使用大语言模型（LLM）和时间知识图谱知识求解，采用多路径答案聚合以提高容错性。RTQA 包含三个核心组件：时间问题分解器、递归求解器和答案聚合器。实验表明，RTQA 在 MultiTQ 和 TimelineKGQA 基准上的 “Multiple” 和 “Complex” 类别中显著改进了 Hits@1，优于现有最佳方法。我们的代码和数据可从以下链接获取。 

---
# Promptception: How Sensitive Are Large Multimodal Models to Prompts? 

**Title (ZH)**: Promptception: 大规模多模态模型对提示的敏感性研究 

**Authors**: Mohamed Insaf Ismithdeen, Muhammad Uzair Khattak, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.03986)  

**Abstract**: Despite the success of Large Multimodal Models (LMMs) in recent years, prompt design for LMMs in Multiple-Choice Question Answering (MCQA) remains poorly understood. We show that even minor variations in prompt phrasing and structure can lead to accuracy deviations of up to 15% for certain prompts and models. This variability poses a challenge for transparent and fair LMM evaluation, as models often report their best-case performance using carefully selected prompts. To address this, we introduce Promptception, a systematic framework for evaluating prompt sensitivity in LMMs. It consists of 61 prompt types, spanning 15 categories and 6 supercategories, each targeting specific aspects of prompt formulation, and is used to evaluate 10 LMMs ranging from lightweight open-source models to GPT-4o and Gemini 1.5 Pro, across 3 MCQA benchmarks: MMStar, MMMU-Pro, MVBench. Our findings reveal that proprietary models exhibit greater sensitivity to prompt phrasing, reflecting tighter alignment with instruction semantics, while open-source models are steadier but struggle with nuanced and complex phrasing. Based on this analysis, we propose Prompting Principles tailored to proprietary and open-source LMMs, enabling more robust and fair model evaluation. 

**Abstract (ZH)**: 尽管大型多模态模型（LMMs）在近年来取得了成功，但在多项选择题回答（MCQA）中的提示设计仍不甚明朗。我们发现，即使是提示措辞和结构上的细微变化，也可能导致某些提示和模型的准确率偏差高达15%。这种变化性给透明和公平的LMM评估带来了挑战，因为模型通常会使用精心选择的提示来报告其最佳性能。为解决这一问题，我们提出了一种名为Promptception的系统性框架，用于评估LMM的提示敏感性。该框架包含61种提示类型，覆盖15个类别和6个超类别，每种类型针对提示制定的特定方面，并用于评估从轻量级开源模型到GPT-4o和Gemini 1.5 Pro的10种LMM在3个MCQA基准（MMStar、MMMU-Pro、MVBench）上的表现。我们发现，专有模型对提示措辞更为敏感，反映出与指令语义更紧密的对齐，而开源模型更加稳定但难以处理精细和复杂的措辞。基于这一分析，我们提出了针对专有和开源LMM的提示原则，以实现更 robust 和公平的模型评估。 

---
# NeuroBreak: Unveil Internal Jailbreak Mechanisms in Large Language Models 

**Title (ZH)**: NeuroBreak: 揭示大型语言模型内部越狱机制 

**Authors**: Chuhan Zhang, Ye Zhang, Bowen Shi, Yuyou Gan, Tianyu Du, Shouling Ji, Dazhan Deng, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03985)  

**Abstract**: In deployment and application, large language models (LLMs) typically undergo safety alignment to prevent illegal and unethical outputs. However, the continuous advancement of jailbreak attack techniques, designed to bypass safety mechanisms with adversarial prompts, has placed increasing pressure on the security defenses of LLMs. Strengthening resistance to jailbreak attacks requires an in-depth understanding of the security mechanisms and vulnerabilities of LLMs. However, the vast number of parameters and complex structure of LLMs make analyzing security weaknesses from an internal perspective a challenging task. This paper presents NeuroBreak, a top-down jailbreak analysis system designed to analyze neuron-level safety mechanisms and mitigate vulnerabilities. We carefully design system requirements through collaboration with three experts in the field of AI security. The system provides a comprehensive analysis of various jailbreak attack methods. By incorporating layer-wise representation probing analysis, NeuroBreak offers a novel perspective on the model's decision-making process throughout its generation steps. Furthermore, the system supports the analysis of critical neurons from both semantic and functional perspectives, facilitating a deeper exploration of security mechanisms. We conduct quantitative evaluations and case studies to verify the effectiveness of our system, offering mechanistic insights for developing next-generation defense strategies against evolving jailbreak attacks. 

**Abstract (ZH)**: 基于神经元的安全机制分析系统NeuroBreak：对抗 Jailbreak 攻击的研究 

---
# SAC-MIL: Spatial-Aware Correlated Multiple Instance Learning for Histopathology Whole Slide Image Classification 

**Title (ZH)**: SAC-MIL：Spatial-aware Correlated Multiple Instance Learning for Histopathology Whole Slide Image Classification 

**Authors**: Yu Bai, Zitong Yu, Haowen Tian, Xijing Wang, Shuo Yan, Lin Wang, Honglin Li, Xitong Ling, Bo Zhang, Zheng Zhang, Wufan Wang, Hui Gao, Xiangyang Gong, Wendong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03973)  

**Abstract**: We propose Spatial-Aware Correlated Multiple Instance Learning (SAC-MIL) for performing WSI classification. SAC-MIL consists of a positional encoding module to encode position information and a SAC block to perform full instance correlations. The positional encoding module utilizes the instance coordinates within the slide to encode the spatial relationships instead of the instance index in the input WSI sequence. The positional encoding module can also handle the length extrapolation issue where the training and testing sequences have different lengths. The SAC block is an MLP-based method that performs full instance correlation in linear time complexity with respect to the sequence length. Due to the simple structure of MLP, it is easy to deploy since it does not require custom CUDA kernels, compared to Transformer-based methods for WSI classification. SAC-MIL has achieved state-of-the-art performance on the CAMELYON-16, TCGA-LUNG, and TCGA-BRAC datasets. The code will be released upon acceptance. 

**Abstract (ZH)**: 基于空间意识的 correlated 多实例学习 (SAC-MIL) 用于 WSI 分类 

---
# Expanding Foundational Language Capabilities in Open-Source LLMs through a Korean Case Study 

**Title (ZH)**: 通过韩国案例研究扩展开源大规模语言模型的基本语言能力 

**Authors**: Junghwan Lim, Gangwon Jo, Sungmin Lee, Jiyoung Park, Dongseok Kim, Jihwan Kim, Junhyeok Lee, Wai Ting Cheung, Dahye Choi, Kibong Choi, Jaeyeon Huh, Beomgyu Kim, Jangwoong Kim, Taehyun Kim, Haesol Lee, Jeesoo Lee, Dongpin Oh, Changseok Song, Daewon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2509.03972)  

**Abstract**: We introduce Llama-3-Motif, a language model consisting of 102 billion parameters, specifically designed to enhance Korean capabilities while retaining strong performance in English. Developed on the Llama 3 architecture, Llama-3-Motif employs advanced training techniques, including LlamaPro and Masked Structure Growth, to effectively scale the model without altering its core Transformer architecture. Using the MoAI platform for efficient training across hyperscale GPU clusters, we optimized Llama-3-Motif using a carefully curated dataset that maintains a balanced ratio of Korean and English data. Llama-3-Motif shows decent performance on Korean-specific benchmarks, outperforming existing models and achieving results comparable to GPT-4. 

**Abstract (ZH)**: Llama-3-Motif：一种包含102亿参数、专门增强韩语能力同时保持英语强大性能的语言模型 

---
# Multimodal Feature Fusion Network with Text Difference Enhancement for Remote Sensing Change Detection 

**Title (ZH)**: 基于文本差异增强的多模态特征融合网络在遥感变化检测中的应用 

**Authors**: Yijun Zhou, Yikui Zhai, Zilu Ying, Tingfeng Xian, Wenlve Zhou, Zhiheng Zhou, Xiaolin Tian, Xudong Jia, Hongsheng Zhang, C. L. Philip Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.03961)  

**Abstract**: Although deep learning has advanced remote sensing change detection (RSCD), most methods rely solely on image modality, limiting feature representation, change pattern modeling, and generalization especially under illumination and noise disturbances. To address this, we propose MMChange, a multimodal RSCD method that combines image and text modalities to enhance accuracy and robustness. An Image Feature Refinement (IFR) module is introduced to highlight key regions and suppress environmental noise. To overcome the semantic limitations of image features, we employ a vision language model (VLM) to generate semantic descriptions of bitemporal images. A Textual Difference Enhancement (TDE) module then captures fine grained semantic shifts, guiding the model toward meaningful changes. To bridge the heterogeneity between modalities, we design an Image Text Feature Fusion (ITFF) module that enables deep cross modal integration. Extensive experiments on LEVIRCD, WHUCD, and SYSUCD demonstrate that MMChange consistently surpasses state of the art methods across multiple metrics, validating its effectiveness for multimodal RSCD. Code is available at: this https URL. 

**Abstract (ZH)**: 尽管深度学习在遥感变化检测（RSCD）方面取得了进展，大多数方法仅依赖图像模态，限制了特征表示、变化模式建模以及在光照和噪声干扰下的泛化能力。为解决这一问题，我们提出了一种融合图像和文本模态的MMChange方法，以提高准确性和鲁棒性。引入了图像特征精炼（IFR）模块以突出关键区域并抑制环境噪声。为克服图像特征的语义限制，我们采用了一种视觉语言模型（VLM）生成双时相图像的语义描述。随后，文本差异增强（TDE）模块捕捉细微的语义变化，指导模型向有意义的变化发展。为弥合模态差异，我们设计了图像文本特征融合（ITFF）模块，实现深层次的跨模态整合。在LEVIRCD、WHUCD和SYSUCD上的 extensive 实验表明，MMChange 在多个指标上持续超过了现有方法，验证了其在多模态RSCD中的有效性。代码可在以下链接获取：this https URL。 

---
# CANDY: Benchmarking LLMs' Limitations and Assistive Potential in Chinese Misinformation Fact-Checking 

**Title (ZH)**: CANDY: 评估大规模语言模型在中文 misinformation 事实核查方面的能力与辅助潜力 

**Authors**: Ruiling Guo, Xinwei Yang, Chen Huang, Tong Zhang, Yong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03957)  

**Abstract**: The effectiveness of large language models (LLMs) to fact-check misinformation remains uncertain, despite their growing use. To this end, we present CANDY, a benchmark designed to systematically evaluate the capabilities and limitations of LLMs in fact-checking Chinese misinformation. Specifically, we curate a carefully annotated dataset of ~20k instances. Our analysis shows that current LLMs exhibit limitations in generating accurate fact-checking conclusions, even when enhanced with chain-of-thought reasoning and few-shot prompting. To understand these limitations, we develop a taxonomy to categorize flawed LLM-generated explanations for their conclusions and identify factual fabrication as the most common failure mode. Although LLMs alone are unreliable for fact-checking, our findings indicate their considerable potential to augment human performance when deployed as assistive tools in scenarios. Our dataset and code can be accessed at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在事实核查虚假信息方面的有效性仍不确定，尽管它们的使用正在增长。为此，我们提出了CANDY，一个旨在系统评估LLMs在中国虚假信息事实核查方面的能力和限制的基准。具体来说，我们策画了一个包含约20,000个实例的精心标注数据集。我们的分析显示，即使增强链式思考推理和少量示例提示，当前的LLMs在生成准确的事实核查结论方面也存在局限性。为了理解这些局限性，我们开发了一种分类法，将LLM生成的错误解释分类，并发现事实捏造是最常见的失败模式。尽管单独使用LLMs进行事实核查不可靠，但我们的研究结果表明，当作为辅助工具部署在特定场景中时，它们具有显著增强人类表现的潜力。我们的数据集和代码可以在以下链接访问：this https URL。 

---
# Chest X-ray Pneumothorax Segmentation Using EfficientNet-B4 Transfer Learning in a U-Net Architecture 

**Title (ZH)**: 使用EfficientNet-B4迁移学习在U-Net架构中的胸片气胸分割 

**Authors**: Alvaro Aranibar Roque, Helga Sebastian  

**Link**: [PDF](https://arxiv.org/pdf/2509.03950)  

**Abstract**: Pneumothorax, the abnormal accumulation of air in the pleural space, can be life-threatening if undetected. Chest X-rays are the first-line diagnostic tool, but small cases may be subtle. We propose an automated deep-learning pipeline using a U-Net with an EfficientNet-B4 encoder to segment pneumothorax regions. Trained on the SIIM-ACR dataset with data augmentation and a combined binary cross-entropy plus Dice loss, the model achieved an IoU of 0.7008 and Dice score of 0.8241 on the independent PTX-498 dataset. These results demonstrate that the model can accurately localize pneumothoraces and support radiologists. 

**Abstract (ZH)**: 胸腔积气的自动化深度学习分割pipeline：基于EfficientNet-B4编码器的U-Net模型在SIIM-ACR数据集上的训练及在PTX-498数据集上的独立验证 

---
# VoxRole: A Comprehensive Benchmark for Evaluating Speech-Based Role-Playing Agents 

**Title (ZH)**: VoxRole: 一个综合性的语音角色扮演代理评估基准 

**Authors**: Weihao Wu, Liang Cao, Xinyu Wu, Zhiwei Lin, Rui Niu, Jingbei Li, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03940)  

**Abstract**: Recent significant advancements in Large Language Models (LLMs) have greatly propelled the development of Role-Playing Conversational Agents (RPCAs). These systems aim to create immersive user experiences through consistent persona adoption. However, current RPCA research faces dual limitations. First, existing work predominantly focuses on the textual modality, entirely overlooking critical paralinguistic features including intonation, prosody, and rhythm in speech, which are essential for conveying character emotions and shaping vivid identities. Second, the speech-based role-playing domain suffers from a long-standing lack of standardized evaluation benchmarks. Most current spoken dialogue datasets target only fundamental capability assessments, featuring thinly sketched or ill-defined character profiles. Consequently, they fail to effectively quantify model performance on core competencies like long-term persona consistency. To address this critical gap, we introduce VoxRole, the first comprehensive benchmark specifically designed for the evaluation of speech-based RPCAs. The benchmark comprises 13335 multi-turn dialogues, totaling 65.6 hours of speech from 1228 unique characters across 261 movies. To construct this resource, we propose a novel two-stage automated pipeline that first aligns movie audio with scripts and subsequently employs an LLM to systematically build multi-dimensional profiles for each character. Leveraging VoxRole, we conduct a multi-dimensional evaluation of contemporary spoken dialogue models, revealing crucial insights into their respective strengths and limitations in maintaining persona consistency. 

**Abstract (ZH)**: 最近大型语言模型的显著进展极大地推动了角色扮演对话代理（RPCA）的发展。这些系统旨在通过一致的人设采用来创造沉浸式用户体验。然而，当前RPCA研究面临双重限制。首先，现有工作主要集中在文本模态上，完全忽视了言语中的语worthy特征，包括语调、语韵和节奏，这些特征对于传达角色情感和塑造生动的身份至关重要。其次，基于语音的角色扮演领域长期以来缺乏标准化的评估基准。目前大多数语音对话数据集仅针对基本能力评估，角色特征描述简略或定义不明确。因此，它们未能有效量化模型在长期角色一致性等核心能力上的表现。为解决这一关键缺口，我们引入了VoxRole，这是首个专门用于评估基于语音的RPCA的标准基准。该基准包含13335个多轮对话，总时长65.6小时的语音，涉及来自261部电影的1228个独特角色。为构建此资源，我们提出了一种新颖的两阶段自动化管道，首先将电影音频与剧本对齐，然后利用大型语言模型系统地为每个角色构建多维度特征。利用VoxRole，我们对当今的语音对话模型进行了多维度评估，揭示了它们在保持角色一致性方面的各自优势和局限性。 

---
# SPFT-SQL: Enhancing Large Language Model for Text-to-SQL Parsing by Self-Play Fine-Tuning 

**Title (ZH)**: SPFT-SQL: 通过自游戏微调增强大型语言模型的文本到SQL解析能力 

**Authors**: Yuhao Zhang, Shaoming Duan, Jinhang Su, Chuanyi Liu, Peiyi Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.03937)  

**Abstract**: Despite the significant advancements of self-play fine-tuning (SPIN), which can transform a weak large language model (LLM) into a strong one through competitive interactions between models of varying capabilities, it still faces challenges in the Text-to-SQL task. SPIN does not generate new information, and the large number of correct SQL queries produced by the opponent model during self-play reduces the main model's ability to generate accurate SQL queries. To address this challenge, we propose a new self-play fine-tuning method tailored for the Text-to-SQL task, called SPFT-SQL. Prior to self-play, we introduce a verification-based iterative fine-tuning approach, which synthesizes high-quality fine-tuning data iteratively based on the database schema and validation feedback to enhance model performance, while building a model base with varying capabilities. During the self-play fine-tuning phase, we propose an error-driven loss method that incentivizes incorrect outputs from the opponent model, enabling the main model to distinguish between correct SQL and erroneous SQL generated by the opponent model, thereby improving its ability to generate correct SQL. Extensive experiments and in-depth analyses on six open-source LLMs and five widely used benchmarks demonstrate that our approach outperforms existing state-of-the-art (SOTA) methods. 

**Abstract (ZH)**: 尽管自我对弈微调（SPIN）取得了显著进展，能够通过不同能力模型之间的竞争互动将弱大型语言模型（LLM）转化为强模型，但在Text-to-SQL任务中仍然面临挑战。SPIN无法生成新信息，对手模型在自我对弈过程中生成的大量正确SQL查询会降低主模型生成准确SQL查询的能力。为解决这一挑战，我们提出了一种新的针对Text-to-SQL任务的自我对弈微调方法，称为SPFT-SQL。在自我对弈之前，我们引入了一种基于验证的迭代微调方法，该方法根据数据库模式和验证反馈逐步生成高质量的微调数据，以提高模型性能并构建具有不同能力的模型库。在自我对弈微调阶段，我们提出了一种错误驱动的损失方法，该方法激励对手模型生成错误输出，从而使主模型能够区分对手模型生成的正确SQL和错误SQL，从而提高其生成正确SQL的能力。大规模实验证明，我们的方法在六个开源LLM和五个广泛使用的基准测试上优于现有最先进的（SOTA）方法。 

---
# SelfAug: Mitigating Catastrophic Forgetting in Retrieval-Augmented Generation via Distribution Self-Alignment 

**Title (ZH)**: SelfAug: 通过分布自对齐缓解检索增强生成中的灾难性遗忘问题 

**Authors**: Yuqing Huang, Rongyang Zhang, Qimeng Wang, Chengqiang Lu, Yan Gao, Yi Wu, Yao Hu, Xuyang Zhi, Guiquan Liu, Xin Li, Hao Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.03934)  

**Abstract**: Recent advancements in large language models (LLMs) have revolutionized natural language processing through their remarkable capabilities in understanding and executing diverse tasks. While supervised fine-tuning, particularly in Retrieval-Augmented Generation (RAG) scenarios, effectively enhances task-specific performance, it often leads to catastrophic forgetting, where models lose their previously acquired knowledge and general capabilities. Existing solutions either require access to general instruction data or face limitations in preserving the model's original distribution. To overcome these limitations, we propose SelfAug, a self-distribution alignment method that aligns input sequence logits to preserve the model's semantic distribution, thereby mitigating catastrophic forgetting and improving downstream performance. Extensive experiments demonstrate that SelfAug achieves a superior balance between downstream learning and general capability retention. Our comprehensive empirical analysis reveals a direct correlation between distribution shifts and the severity of catastrophic forgetting in RAG scenarios, highlighting how the absence of RAG capabilities in general instruction tuning leads to significant distribution shifts during fine-tuning. Our findings not only advance the understanding of catastrophic forgetting in RAG contexts but also provide a practical solution applicable across diverse fine-tuning scenarios. Our code is publicly available at this https URL. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）通过其在理解和执行多样任务方面的出色能力，已经革新了自然语言处理。虽然在检索增强生成（RAG）场景中，监督微调有效提升了特定任务的性能，但往往会引发灾难性遗忘，导致模型失去之前获取的知识和通用能力。现有解决方案要么需要访问通用指令数据，要么无法保留模型的原始分布。为克服这些限制，我们提出了一种自我分布对齐方法SelfAug，该方法通过对齐输入序列的逻辑值以保留模型的语义分布，从而缓解灾难性遗忘并提高下游性能。大量的实验表明，SelfAug在下游学习和通用能力保留之间实现了更优的平衡。我们的全面实证分析揭示了分布变化与RAG场景中灾难性遗忘严重程度之间的直接关联，强调了在通用指令调优中缺乏RAG能力如何导致细调过程中显著的分布变化。我们的研究不仅推进了对RAG背景下灾难性遗忘的理解，还为多种细调场景提供了实际解决方案。我们的代码可以在以下公共地址获取：this https URL。 

---
# MTQA:Matrix of Thought for Enhanced Reasoning in Complex Question Answering 

**Title (ZH)**: MTQA：增强复杂问答中推理的思维矩阵 

**Authors**: Fengxiao Tang, Yufeng Li, Zongzong Wu, Ming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.03918)  

**Abstract**: Complex Question Answering (QA) is a fundamental and challenging task in NLP. While large language models (LLMs) exhibit impressive performance in QA, they suffer from significant performance degradation when facing complex and abstract QA tasks due to insufficient reasoning capabilities. Works such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT) aim to enhance LLMs' reasoning abilities, but they face issues such as in-layer redundancy in tree structures and single paths in chain structures. Although some studies utilize Retrieval-Augmented Generation (RAG) methods to assist LLMs in reasoning, the challenge of effectively utilizing large amounts of information involving multiple entities and hops remains critical. To address this, we propose the Matrix of Thought (MoT), a novel and efficient LLM thought structure. MoT explores the problem in both horizontal and vertical dimensions through the "column-cell communication" mechanism, enabling LLMs to actively engage in multi-strategy and deep-level thinking, reducing redundancy within the column cells and enhancing reasoning capabilities. Furthermore, we develop a fact-correction mechanism by constructing knowledge units from retrieved knowledge graph triples and raw text to enhance the initial knowledge for LLM reasoning and correct erroneous answers. This leads to the development of an efficient and accurate QA framework (MTQA). Experimental results show that our framework outperforms state-of-the-art methods on four widely-used datasets in terms of F1 and EM scores, with reasoning time only 14.4\% of the baseline methods, demonstrating both its efficiency and accuracy. The code for this framework is available at this https URL. 

**Abstract (ZH)**: 复杂问题回答（QA）是自然语言处理（NLP）中的一个基础而具有挑战性的任务。虽然大型语言模型（LLMs）在QA方面表现出色，但在面对复杂和抽象的QA任务时，由于推理能力的不足，其性能会显著下降。Chain-of-Thought（CoT）和Tree-of-Thought（ToT）等工作旨在增强LLMs的推理能力，但它们面临树结构中的层内冗余和链结构中的单一路径等问题。尽管一些研究利用检索增强生成（RAG）方法来辅助LLMs进行推理，但在处理涉及多个实体和跃点的大量信息时的有效利用仍然是一个重大挑战。为了解决这一问题，我们提出了一种新颖且高效的LLM思维结构——Matrix of Thought（MoT）。MoT通过“列单元通信”机制在纵横两个维度上探索问题，使LLMs能够积极参与多策略和深层次的思考，减少列单元内的冗余并增强推理能力。此外，我们开发了一种事实校正机制，通过从检索的知识图谱三元组和原始文本构建知识单元来增强初始知识，提高LLMs的推理能力，并纠正错误的答案。这导致开发出一个高效准确的QA框架（MTQA）。实验结果表明，我们的框架在四个广泛使用的数据集上相对于基线方法在F1和EM分数方面表现出更优的性能，推理时间仅为基线方法的14.4%，展示了其高效性和准确性。该框架的代码可以在该链接处获得。 

---
# Diffusion Generative Models Meet Compressed Sensing, with Applications to Image Data and Financial Time Series 

**Title (ZH)**: 扩散生成模型与压缩感知的结合及其在图像数据和金融时间序列中的应用 

**Authors**: Zhengyi Guo, Jiatu Li, Wenpin Tang, David D. Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.03898)  

**Abstract**: This paper develops dimension reduction techniques for accelerating diffusion model inference in the context of synthetic data generation. The idea is to integrate compressed sensing into diffusion models: (i) compress the data into a latent space, (ii) train a diffusion model in the latent space, and (iii) apply a compressed sensing algorithm to the samples generated in the latent space, facilitating the efficiency of both model training and inference. Under suitable sparsity assumptions on data, the proposed algorithm is proved to enjoy faster convergence by combining diffusion model inference with sparse recovery. As a byproduct, we obtain an optimal value for the latent space dimension. We also conduct numerical experiments on a range of datasets, including image data (handwritten digits, medical images, and climate data) and financial time series for stress testing. 

**Abstract (ZH)**: 本文开发了维度缩减技术以加速合成数据生成中的扩散模型推理。该方法将压缩感知集成到扩散模型中：(i) 将数据压缩到潜在空间，(ii) 在潜在空间训练扩散模型，(iii) 对潜在空间生成的样本应用压缩感知算法，从而提高模型训练和推理的效率。在适合的稀疏性假设下，所提出的算法通过结合扩散模型推理和稀疏恢复实现了更快的收敛。作为副产品，我们获得了潜在空间维数的最优值。我们还在图像数据（手写数字、医学图像和气候数据）和金融时间序列（压力测试）等多种数据集上进行了数值实验。 

---
# Reactive In-Air Clothing Manipulation with Confidence-Aware Dense Correspondence and Visuotactile Affordance 

**Title (ZH)**: 基于信心感知密集对应和视听触利用的反应式空中衣物操作 

**Authors**: Neha Sunil, Megha Tippur, Arnau Saumell, Edward Adelson, Alberto Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2509.03889)  

**Abstract**: Manipulating clothing is challenging due to complex configurations, variable material dynamics, and frequent self-occlusion. Prior systems often flatten garments or assume visibility of key features. We present a dual-arm visuotactile framework that combines confidence-aware dense visual correspondence and tactile-supervised grasp affordance to operate directly on crumpled and suspended garments. The correspondence model is trained on a custom, high-fidelity simulated dataset using a distributional loss that captures cloth symmetries and generates correspondence confidence estimates. These estimates guide a reactive state machine that adapts folding strategies based on perceptual uncertainty. In parallel, a visuotactile grasp affordance network, self-supervised using high-resolution tactile feedback, determines which regions are physically graspable. The same tactile classifier is used during execution for real-time grasp validation. By deferring action in low-confidence states, the system handles highly occluded table-top and in-air configurations. We demonstrate our task-agnostic grasp selection module in folding and hanging tasks. Moreover, our dense descriptors provide a reusable intermediate representation for other planning modalities, such as extracting grasp targets from human video demonstrations, paving the way for more generalizable and scalable garment manipulation. 

**Abstract (ZH)**: 基于视觉-触觉的服装操作框架：结合置信度感知密集视觉对应和触觉监督的抓取可能性 

---
# Peptidomic-Based Prediction Model for Coronary Heart Disease Using a Multilayer Perceptron Neural Network 

**Title (ZH)**: 基于肽组学的冠心病预测模型：多层感知机神经网络方法 

**Authors**: Jesus Celis-Porras  

**Link**: [PDF](https://arxiv.org/pdf/2509.03884)  

**Abstract**: Coronary heart disease (CHD) is a leading cause of death worldwide and contributes significantly to annual healthcare expenditures. To develop a non-invasive diagnostic approach, we designed a model based on a multilayer perceptron (MLP) neural network, trained on 50 key urinary peptide biomarkers selected via genetic algorithms. Treatment and control groups, each comprising 345 individuals, were balanced using the Synthetic Minority Over-sampling Technique (SMOTE). The neural network was trained using a stratified validation strategy. Using a network with three hidden layers of 60 neurons each and an output layer of two neurons, the model achieved a precision, sensitivity, and specificity of 95.67 percent, with an F1-score of 0.9565. The area under the ROC curve (AUC) reached 0.9748 for both classes, while the Matthews correlation coefficient (MCC) and Cohen's kappa coefficient were 0.9134 and 0.9131, respectively, demonstrating its reliability in detecting CHD. These results indicate that the model provides a highly accurate and robust non-invasive diagnostic tool for coronary heart disease. 

**Abstract (ZH)**: 冠心病（CHD）是全球主要的死亡原因之一，显著增加了年度医疗支出。为了开发一种无创诊断方法，我们设计了一个基于多层感知机（MLP）神经网络的模型，该模型利用遗传算法选择了50个关键尿液肽生物标记物。治疗组和对照组，每组各包括345名个体，通过合成少数类过采样技术（SMOTE）进行了平衡。神经网络使用分层验证策略进行训练。使用具有三个隐藏层，每层60个神经元以及一个输出层两个神经元的网络，模型实现了95.67%的精确率、敏感性和特异性，F1分数为0.9565。两类的受试者操作特征曲线下面积（AUC）达到了0.9748，而马修相关系数（MCC）和科恩κ系数分别为0.9134和0.9131，表明该模型在检测冠心病方面具有很高的可靠性和准确性。这些结果表明，该模型提供了一种高度准确和稳健的无创诊断工具，用于冠心病的诊断。 

---
# SalientFusion: Context-Aware Compositional Zero-Shot Food Recognition 

**Title (ZH)**: SalientFusion: 具有上下文意识的组分零样本食物识别 

**Authors**: Jiajun Song, Xiaoou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03873)  

**Abstract**: Food recognition has gained significant attention, but the rapid emergence of new dishes requires methods for recognizing unseen food categories, motivating Zero-Shot Food Learning (ZSFL). We propose the task of Compositional Zero-Shot Food Recognition (CZSFR), where cuisines and ingredients naturally align with attributes and objects in Compositional Zero-Shot learning (CZSL). However, CZSFR faces three challenges: (1) Redundant background information distracts models from learning meaningful food features, (2) Role confusion between staple and side dishes leads to misclassification, and (3) Semantic bias in a single attribute can lead to confusion of understanding. Therefore, we propose SalientFusion, a context-aware CZSFR method with two components: SalientFormer, which removes background redundancy and uses depth features to resolve role confusion; DebiasAT, which reduces the semantic bias by aligning prompts with visual features. Using our proposed benchmarks, CZSFood-90 and CZSFood-164, we show that SalientFusion achieves state-of-the-art results on these benchmarks and the most popular general datasets for the general CZSL. The code is avaliable at this https URL. 

**Abstract (ZH)**: 食品识别引起了广泛关注，但新菜品的迅速涌现需要能够识别未见过的食品类别的方法，这促进了零样本食品学习（ZSFL）的发展。我们提出了组合零样本食品识别（CZSFR）任务，在组合零样本学习（CZSL）中，烹饪方式和食材自然地与属性和对象相匹配。然而，CZSFR 面临三大挑战：（1）冗余背景信息会干扰模型学习有意义的食品特征，（2）主食与配菜的角色混淆导致分类错误，（3）单个属性中的语义偏见可能导致理解混淆。因此，我们提出了一个基于上下文的 CZSFR 方法——SalientFusion，该方法由两个组件组成：SalientFormer，它去除背景冗余并使用深度特征解决角色混淆；DebiasAT，它通过将提示与视觉特征对齐来减少语义偏见。使用我们提出的基准数据集 CZSFood-90 和 CZSFood-164，我们证明了 SalientFusion 在这些基准及最流行的通用 CZSL 数据集上达到了最先进的性能。代码可在此处访问：https://xxxxxx。 

---
# A Comprehensive Survey on Trustworthiness in Reasoning with Large Language Models 

**Title (ZH)**: 大型语言模型推理中的可信性综述 

**Authors**: Yanbo Wang, Yongcan Yu, Jian Liang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2509.03871)  

**Abstract**: The development of Long-CoT reasoning has advanced LLM performance across various tasks, including language understanding, complex problem solving, and code generation. This paradigm enables models to generate intermediate reasoning steps, thereby improving both accuracy and interpretability. However, despite these advancements, a comprehensive understanding of how CoT-based reasoning affects the trustworthiness of language models remains underdeveloped. In this paper, we survey recent work on reasoning models and CoT techniques, focusing on five core dimensions of trustworthy reasoning: truthfulness, safety, robustness, fairness, and privacy. For each aspect, we provide a clear and structured overview of recent studies in chronological order, along with detailed analyses of their methodologies, findings, and limitations. Future research directions are also appended at the end for reference and discussion. Overall, while reasoning techniques hold promise for enhancing model trustworthiness through hallucination mitigation, harmful content detection, and robustness improvement, cutting-edge reasoning models themselves often suffer from comparable or even greater vulnerabilities in safety, robustness, and privacy. By synthesizing these insights, we hope this work serves as a valuable and timely resource for the AI safety community to stay informed on the latest progress in reasoning trustworthiness. A full list of related papers can be found at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 长推理(Long-CoT)推理的发展提高了各种任务中LLM的表现，包括语言理解、复杂问题解决和代码生成。这种范式使模型能够生成中间推理步骤，从而提高准确性和可解释性。然而，尽管取得了这些进展，关于基于CoT的推理如何影响语言模型的可信度的理解仍不够全面。在本文中，我们回顾了推理模型和CoT技术的近期研究，重点关注可信推理的五个核心维度：真实可信性、安全性、鲁棒性、公平性和隐私性。对于每个方面，我们按照时间顺序提供了清晰的、结构化的近期研究综述，并对它们的方法论、发现和局限性进行了详细的分析。最后，还附上了未来研究的方向供参考和讨论。总的来说，尽管推理技术有望通过减轻幻觉、检测有害内容和提高鲁棒性来增强模型的可信度，但最前沿的推理模型本身在安全性、鲁棒性和隐私性方面也常常遭受类似甚至更大的脆弱性。通过综合这些见解，我们希望本文能成为AI安全社区获取最新推理可信度进展的重要和及时的资源。与本文相关的论文列表可以在 \href{this https URL}{this https URL} 找到。 

---
# MillGNN: Learning Multi-Scale Lead-Lag Dependencies for Multi-Variate Time Series Forecasting 

**Title (ZH)**: MillGNN：学习多尺度领先-滞后依赖关系的多变量时间序列预测 

**Authors**: Binqing Wu, Zongjiang Shang, Jianlong Huang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.03852)  

**Abstract**: Multi-variate time series (MTS) forecasting is crucial for various applications. Existing methods have shown promising results owing to their strong ability to capture intra- and inter-variate dependencies. However, these methods often overlook lead-lag dependencies at multiple grouping scales, failing to capture hierarchical lead-lag effects in complex systems. To this end, we propose MillGNN, a novel \underline{g}raph \underline{n}eural \underline{n}etwork-based method that learns \underline{m}ult\underline{i}ple grouping scale \underline{l}ead-\underline{l}ag dependencies for MTS forecasting, which can comprehensively capture lead-lag effects considering variate-wise and group-wise dynamics and decays. Specifically, MillGNN introduces two key innovations: (1) a scale-specific lead-lag graph learning module that integrates cross-correlation coefficients and dynamic decaying features derived from real-time inputs and time lags to learn lead-lag dependencies for each scale, which can model evolving lead-lag dependencies with statistical interpretability and data-driven flexibility; (2) a hierarchical lead-lag message passing module that passes lead-lag messages at multiple grouping scales in a structured way to simultaneously propagate intra- and inter-scale lead-lag effects, which can capture multi-scale lead-lag effects with a balance of comprehensiveness and efficiency. Experimental results on 11 datasets demonstrate the superiority of MillGNN for long-term and short-term MTS forecasting, compared with 16 state-of-the-art methods. 

**Abstract (ZH)**: 多变量时间序列(MTS)预测对于各种应用至关重要。现有方法因其强大的内在和跨变量依赖关系捕捉能力而显示出有前途的结果。然而，这些方法往往忽略了多个分组尺度上的领先-滞后依赖关系，未能捕捉复杂系统中的层次领先-滞后效应。为了解决这一问题，我们提出了一种新颖的基于图神经网络的方法MillGNN，它能够学习多分组尺度的领先-滞后依赖关系，从而综合考虑变量层面和分组层面的动力学和衰减，全面捕捉领先-滞后效应。MillGNN引入了两个关键创新：(1) 一种尺度特定的领先-滞后图学习模块，该模块结合了来自实时输入和时间滞后的真实跨相关系数和动态衰减特征，以学习每个尺度的领先-滞后依赖关系，能够以统计可解释性和数据驱动的灵活性建模随时间演变的领先-滞后依赖关系；(2) 一种分层领先-滞后消息传递模块，该模块以结构化的方式在多个分组尺度上传递领先-滞后消息，同时传播内在和跨尺度的领先-滞后效应，能够以全面性和效率之间的平衡捕捉多尺度的领先-滞后效应。在11个数据集上的实验结果表明，MillGNN在长短期MTS预测中优于16种最先进的方法。 

---
# Meta-Inverse Reinforcement Learning for Mean Field Games via Probabilistic Context Variables 

**Title (ZH)**: 基于概率情境变量的元逆强化学习在均场博弈中的应用 

**Authors**: Yang Chen, Xiao Lin, Bo Yan, Libo Zhang, Jiamou Liu, Neset Özkan Tan, Michael Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2509.03845)  

**Abstract**: Designing suitable reward functions for numerous interacting intelligent agents is challenging in real-world applications. Inverse reinforcement learning (IRL) in mean field games (MFGs) offers a practical framework to infer reward functions from expert demonstrations. While promising, the assumption of agent homogeneity limits the capability of existing methods to handle demonstrations with heterogeneous and unknown objectives, which are common in practice. To this end, we propose a deep latent variable MFG model and an associated IRL method. Critically, our method can infer rewards from different yet structurally similar tasks without prior knowledge about underlying contexts or modifying the MFG model itself. Our experiments, conducted on simulated scenarios and a real-world spatial taxi-ride pricing problem, demonstrate the superiority of our approach over state-of-the-art IRL methods in MFGs. 

**Abstract (ZH)**: 设计适合众多相互智能代理的奖励函数在实际应用中具有挑战性。均场游戏中的逆强化学习为从专家演示中推断奖励函数提供了实用框架。然而，代理同质性的假设限制了现有方法处理具有异质性和未知目标的演示能力，而这在实践中很常见。为此，我们提出了一种深度潜变量均场游戏模型及其相关的逆强化学习方法。关键的是，我们的方法可以在没有了解底层上下文或修改均场游戏模型本身的情况下，从不同但结构相似的任务中推断出奖励。我们在模拟场景和实际的空.action定价问题中的实验表明，与当前最佳的均场游戏中逆强化学习方法相比，我们的方法更优。 

---
# INGRID: Intelligent Generative Robotic Design Using Large Language Models 

**Title (ZH)**: INGRID：使用大型语言模型的智能生成性机器人设计 

**Authors**: Guanglu Jia, Ceng Zhang, Gregory S. Chirikjian  

**Link**: [PDF](https://arxiv.org/pdf/2509.03842)  

**Abstract**: The integration of large language models (LLMs) into robotic systems has accelerated progress in embodied artificial intelligence, yet current approaches remain constrained by existing robotic architectures, particularly serial mechanisms. This hardware dependency fundamentally limits the scope of robotic intelligence. Here, we present INGRID (Intelligent Generative Robotic Design), a framework that enables the automated design of parallel robotic mechanisms through deep integration with reciprocal screw theory and kinematic synthesis methods. We decompose the design challenge into four progressive tasks: constraint analysis, kinematic joint generation, chain construction, and complete mechanism design. INGRID demonstrates the ability to generate novel parallel mechanisms with both fixed and variable mobility, discovering kinematic configurations not previously documented in the literature. We validate our approach through three case studies demonstrating how INGRID assists users in designing task-specific parallel robots based on desired mobility requirements. By bridging the gap between mechanism theory and machine learning, INGRID enables researchers without specialized robotics training to create custom parallel mechanisms, thereby decoupling advances in robotic intelligence from hardware constraints. This work establishes a foundation for mechanism intelligence, where AI systems actively design robotic hardware, potentially transforming the development of embodied AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机器人系统中的集成加速了具身人工智能的进步，但当前的方法仍受现有机器人架构的限制，尤其是序列机制。这种硬件依赖性从根本上限制了机器人智能的范围。为此，我们提出了INGRID（Intelligent Generative Robotic Design）框架，通过深度整合互反螺旋理论和运动合成方法，实现并行机器人机制的自动化设计。我们将设计挑战分解为四个渐进任务：约束分析、运动副生成、链路构造和完整机制设计。INGRID展示了生成既有固定移动性又有可变移动性的新型并行机制的能力，发现文献中未曾记载的运动学配置。我们通过三个案例研究验证了我们的方法，展示了INGRID如何帮助用户根据所需的移动性要求设计特定任务的并行机器人。通过弥合机制理论与机器学习之间的差距，INGRID使没有专门机器人训练的研究人员能够创建自定义的并行机制，从而解开机器人智能进步与硬件限制的关系。这项工作为机制智能奠定了基础，其中AI系统积极设计机器人硬件，有可能变革具身AI系统的开发。 

---
# From Leiden to Pleasure Island: The Constant Potts Model for Community Detection as a Hedonic Game 

**Title (ZH)**: 从莱登到快乐岛：常数Potts模型在社区检测中的 hedonic 规则 

**Authors**: Lucas Lopes Felipe, Konstantin Avrachenkov, Daniel Sadoc Menasche  

**Link**: [PDF](https://arxiv.org/pdf/2509.03834)  

**Abstract**: Community detection is one of the fundamental problems in data science which consists of partitioning nodes into disjoint communities. We present a game-theoretic perspective on the Constant Potts Model (CPM) for partitioning networks into disjoint communities, emphasizing its efficiency, robustness, and accuracy. Efficiency: We reinterpret CPM as a potential hedonic game by decomposing its global Hamiltonian into local utility functions, where the local utility gain of each agent matches the corresponding increase in global utility. Leveraging this equivalence, we prove that local optimization of the CPM objective via better-response dynamics converges in pseudo-polynomial time to an equilibrium partition. Robustness: We introduce and relate two stability criteria: a strict criterion based on a novel notion of robustness, requiring nodes to simultaneously maximize neighbors and minimize non-neighbors within communities, and a relaxed utility function based on a weighted sum of these objectives, controlled by a resolution parameter. Accuracy: In community tracking scenarios, where initial partitions are used to bootstrap the Leiden algorithm with partial ground-truth information, our experiments reveal that robust partitions yield higher accuracy in recovering ground-truth communities. 

**Abstract (ZH)**: 基于博弈论视角的常数潘特模型在分离网络社区中的应用：效率、稳健性和准确性 

---
# Gravity Well Echo Chamber Modeling With An LLM-Based Confirmation Bias Model 

**Title (ZH)**: 基于LLM的确认偏见模型的引力井回声室建模 

**Authors**: Joseph Jackson, Georgiy Lapin, Jeremy E. Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2509.03832)  

**Abstract**: Social media echo chambers play a central role in the spread of misinformation, yet existing models often overlook the influence of individual confirmation bias. An existing model of echo chambers is the "gravity well" model, which creates an analog between echo chambers and spatial gravity wells. We extend this established model by introducing a dynamic confirmation bias variable that adjusts the strength of pull based on a user's susceptibility to belief-reinforcing content. This variable is calculated for each user through comparisons between their posting history and their responses to posts of a wide range of viewpoints.
Incorporating this factor produces a confirmation-bias-integrated gravity well model that more accurately identifies echo chambers and reveals community-level markers of information health. We validated the approach on nineteen Reddit communities, demonstrating improved detection of echo chambers.
Our contribution is a framework for systematically capturing the role of confirmation bias in online group dynamics, enabling more effective identification of echo chambers. By flagging these high-risk environments, the model supports efforts to curb the spread of misinformation at its most common points of amplification. 

**Abstract (ZH)**: 社交媒体回音室在 misinformation 的传播中起着核心作用，然而现有的模型往往忽视了个体确认偏见的影响。现有的回音室模型之一是“引力井”模型，该模型将回音室类比为空间引力井。我们在此基础上引入了一个动态确认偏见变量，该变量根据用户对信念强化内容的易感性调整拉力的强度。该变量是通过对每个用户发帖历史与其对广泛观点帖子的响应之间的比较来计算的。将此因素纳入模型产生了一个整合了确认偏见的引力井模型，能够更准确地识别回音室并揭示信息健康的社区级指标。我们在十九个Reddit社区上验证了该方法，展示了回音室检测的改进。我们贡献了一个系统地捕捉在线群体动态中确认偏见作用的框架，有助于更有效地识别回音室。通过标记这些高风险环境，该模型支持了遏制 misinformation 放大效果的努力。 

---
# Align-then-Slide: A complete evaluation framework for Ultra-Long Document-Level Machine Translation 

**Title (ZH)**: 对齐然后滑动：超长文档级机器翻译的全面评估框架 

**Authors**: Jiaxin Guo, Daimeng Wei, Yuanchang Luo, Xiaoyu Chen, Zhanglin Wu, Huan Yang, Hengchao Shang, Zongyao Li, Zhiqiang Rao, Jinlong Yang, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03809)  

**Abstract**: Large language models (LLMs) have ushered in a new era for document-level machine translation (\textit{doc}-mt), yet their whole-document outputs challenge existing evaluation methods that assume sentence-by-sentence alignment. We introduce \textit{\textbf{Align-then-Slide}}, a complete evaluation framework for ultra-long doc-mt. In the Align stage, we automatically infer sentence-level source-target correspondences and rebuild the target to match the source sentence number, resolving omissions and many-to-one/one-to-many mappings. In the n-Chunk Sliding Evaluate stage, we calculate averaged metric scores under 1-, 2-, 3- and 4-chunk for multi-granularity assessment. Experiments on the WMT benchmark show a Pearson correlation of 0.929 between our method with expert MQM rankings. On a newly curated real-world test set, our method again aligns closely with human judgments. Furthermore, preference data produced by Align-then-Slide enables effective CPO training and its direct use as a reward model for GRPO, both yielding translations preferred over a vanilla SFT baseline. The results validate our framework as an accurate, robust, and actionable evaluation tool for doc-mt systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）为文档级机器翻译（\textit{doc}-mt）带来了新时代，但其整文档输出挑战了现有的假设句子级对齐的评估方法。我们介绍了\textit{\textbf{Align-then-Slide}}，一个完整的超长文档级机器翻译评估框架。在Align阶段，我们自动推断句子级源到目标的对应关系并重建目标，以匹配源句子数量，解决遗漏和多对一/一对多的映射。在n-Chunk滑动评估阶段，我们计算1-，2-，3-，和4-片段下的平均度量评分，实现多粒度评估。在WMT基准测试上的实验显示，我们的方法与专家MQM排名之间的皮尔逊相关系数为0.929。在新收集的真实测试集上，我们的方法再次与人类判断紧密吻合。此外，\textit{Align-then-Slide}生成的偏好数据能够有效训练CPO，并直接用作GRPO的奖励模型，两者都比 vanilla SFT 基准翻译更受欢迎。实验结果验证了我们的框架作为文档级机器翻译系统准确、稳健且实用的评估工具的有效性。 

---
# Measuring How (Not Just Whether) VLMs Build Common Ground 

**Title (ZH)**: 测量（而不仅仅是判断）大模型如何构建共同基础 

**Authors**: Saki Imai, Mert İnan, Anthony Sicilia, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2509.03805)  

**Abstract**: Large vision language models (VLMs) increasingly claim reasoning skills, yet current benchmarks evaluate them in single-turn or question answering settings. However, grounding is an interactive process in which people gradually develop shared understanding through ongoing communication. We introduce a four-metric suite (grounding efficiency, content alignment, lexical adaptation, and human-likeness) to systematically evaluate VLM performance in interactive grounding contexts. We deploy the suite on 150 self-play sessions of interactive referential games between three proprietary VLMs and compare them with human dyads. All three models diverge from human patterns on at least three metrics, while GPT4o-mini is the closest overall. We find that (i) task success scores do not indicate successful grounding and (ii) high image-utterance alignment does not necessarily predict task success. Our metric suite and findings offer a framework for future research on VLM grounding. 

**Abstract (ZH)**: 大型视觉语言模型在交互式接地场景中的系统评估：基于四个指标的框架 

---
# SAMVAD: A Multi-Agent System for Simulating Judicial Deliberation Dynamics in India 

**Title (ZH)**: SAMVAD：一种模拟印度司法审议动力学的多agent系统 

**Authors**: Prathamesh Devadiga, Omkaar Jayadev Shetty, Pooja Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.03793)  

**Abstract**: Understanding the complexities of judicial deliberation is crucial for assessing the efficacy and fairness of a justice system. However, empirical studies of judicial panels are constrained by significant ethical and practical barriers. This paper introduces SAMVAD, an innovative Multi-Agent System (MAS) designed to simulate the deliberation process within the framework of the Indian justice system.
Our system comprises agents representing key judicial roles: a Judge, a Prosecution Counsel, a Defense Counsel, and multiple Adjudicators (simulating a judicial bench), all powered by large language models (LLMs). A primary contribution of this work is the integration of Retrieval-Augmented Generation (RAG), grounded in a domain-specific knowledge base of landmark Indian legal documents, including the Indian Penal Code and the Constitution of India. This RAG functionality enables the Judge and Counsel agents to generate legally sound instructions and arguments, complete with source citations, thereby enhancing both the fidelity and transparency of the simulation.
The Adjudicator agents engage in iterative deliberation rounds, processing case facts, legal instructions, and arguments to reach a consensus-based verdict. We detail the system architecture, agent communication protocols, the RAG pipeline, the simulation workflow, and a comprehensive evaluation plan designed to assess performance, deliberation quality, and outcome consistency.
This work provides a configurable and explainable MAS platform for exploring legal reasoning and group decision-making dynamics in judicial simulations, specifically tailored to the Indian legal context and augmented with verifiable legal grounding via RAG. 

**Abstract (ZH)**: 理解司法审议的复杂性对于评估司法系统的有效性与公正性至关重要。然而，对司法合议庭的实证研究受到重大伦理和实践障碍的限制。本文介绍了一种创新性的多智能体系统（MAS）——SAMVAD，用于模拟印度司法系统框架下的审议过程。 

---
# SiLVERScore: Semantically-Aware Embeddings for Sign Language Generation Evaluation 

**Title (ZH)**: SiLVERScore: 语义aware嵌入表示在手语生成评估中的应用 

**Authors**: Saki Imai, Mert İnan, Anthony Sicilia, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2509.03791)  

**Abstract**: Evaluating sign language generation is often done through back-translation, where generated signs are first recognized back to text and then compared to a reference using text-based metrics. However, this two-step evaluation pipeline introduces ambiguity: it not only fails to capture the multimodal nature of sign language-such as facial expressions, spatial grammar, and prosody-but also makes it hard to pinpoint whether evaluation errors come from sign generation model or the translation system used to assess it. In this work, we propose SiLVERScore, a novel semantically-aware embedding-based evaluation metric that assesses sign language generation in a joint embedding space. Our contributions include: (1) identifying limitations of existing metrics, (2) introducing SiLVERScore for semantically-aware evaluation, (3) demonstrating its robustness to semantic and prosodic variations, and (4) exploring generalization challenges across datasets. On PHOENIX-14T and CSL-Daily datasets, SiLVERScore achieves near-perfect discrimination between correct and random pairs (ROC AUC = 0.99, overlap < 7%), substantially outperforming traditional metrics. 

**Abstract (ZH)**: 评估手语生成通常通过回译进行，即将生成的手语首先识别回文本，然后使用文本指标与参考文本进行对比。然而，这种两步评估管道引入了模糊性：它不仅未能捕捉手语的多模态性质——如面部表情、空间语法和语调，还使得难以确定评估错误是源自手语生成模型还是用于评估的手译系统。在本文中，我们提出了SiLVERScore，这是一种新颖的感知语义的嵌入式评价指标，用于在联合嵌入空间中评估手语生成。我们的贡献包括：(1) 识别现有指标的局限性，(2) 引入SiLVERScore进行感知语义的评估，(3) 展示其对语义和语调变化的鲁棒性，并且(4) 探讨跨数据集的一般化挑战。SiLVERScore在PHOENIX-14T和CSL-Daily数据集上，几乎完美地区分了正确和随机配对（ROC AUC = 0.99，重叠<7%），显著优于传统指标。 

---
# What Fundamental Structure in Reward Functions Enables Efficient Sparse-Reward Learning? 

**Title (ZH)**: 奖励函数中的什么基础结构能 Enable 有效的稀疏奖励学习？ 

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.03790)  

**Abstract**: What fundamental properties of reward functions enable efficient sparse-reward reinforcement learning? We address this question through the lens of low-rank structure in reward matrices, showing that such structure induces a sharp transition from exponential to polynomial sample complexity, the first result of this kind for sparse-reward RL. We introduce Policy-Aware Matrix Completion (PAMC), which connects matrix completion theory with reinforcement learning via a new analysis of policy-dependent sampling. Our framework provides: (i) impossibility results for general sparse reward observation, (ii) reward-free representation learning from dynamics, (iii) distribution-free confidence sets via conformal prediction, and (iv) robust completion guarantees that degrade gracefully when low-rank structure is only approximate. Empirically, we conduct a pre-registered evaluation across 100 systematically sampled domains, finding exploitable structure in over half. PAMC improves sample efficiency by factors between 1.6 and 2.1 compared to strong exploration, structured, and representation-learning baselines, while adding only about 20 percent computational this http URL results establish structural reward learning as a promising new paradigm, with immediate implications for robotics, healthcare, and other safety-critical, sample-expensive applications. 

**Abstract (ZH)**: 哪些基础性属性使奖励函数能够支持高效的稀疏奖励强化学习？通过奖励矩阵的低秩结构视角，我们探讨了这一问题，展示了这种结构导致从指数到多项式的样本复杂度急剧转变，这是首次在稀疏奖励RL中获得此类结果。我们引入了基于策略的矩阵补全(PAMC)方法，将矩阵补全理论与强化学习通过策略相关采样的新分析联系起来。我们的框架提供了：(i) 一般稀疏奖励观察的不可能性结果，(ii) 基于动力学的奖励免费表示学习，(iii) 基于全同预测的无分布置信集，以及(iv) 当低秩结构仅近似时表现良好的补全保证。实验中，我们在100个系统采样的领域进行了注册评估，发现超过一半的领域存在可利用的结构。与强大的探索、结构化和表示学习基线相比，PAMC在样本效率上提高了1.6到2.1倍，同时仅增加了约20%的计算开销。这些结果确立了结构奖励学习作为有前景的新范式，并对机器人技术、医疗保健及其他安全性至关重要的、样本昂贵的应用领域具有直接影响。 

---
# Natural Latents: Latent Variables Stable Across Ontologies 

**Title (ZH)**: 自然隐变量：跨本体稳定的隐变量 

**Authors**: John Wentworth, David Lorell  

**Link**: [PDF](https://arxiv.org/pdf/2509.03780)  

**Abstract**: Suppose two Bayesian agents each learn a generative model of the same environment. We will assume the two have converged on the predictive distribution, i.e. distribution over some observables in the environment, but may have different generative models containing different latent variables. Under what conditions can one agent guarantee that their latents are a function of the other agents latents?
We give simple conditions under which such translation is guaranteed to be possible: the natural latent conditions. We also show that, absent further constraints, these are the most general conditions under which translatability is guaranteed. Crucially for practical application, our theorems are robust to approximation error in the natural latent conditions. 

**Abstract (ZH)**: 假设两个贝叶斯代理各自学习同一环境的生成模型。我们将假设这两个代理已经在预测分布上收敛，即环境中的某些可观测值的分布，但它们的生成模型可能包含不同的潜在变量。在什么条件下一个代理可以保证其潜在变量是另一个代理潜在变量的函数？

在这种翻译下，标题为：

自然潜在条件下的翻译保证条件及其实用性 

---
# Learning an Adversarial World Model for Automated Curriculum Generation in MARL 

**Title (ZH)**: 学习对抗世界模型以实现自动课程生成在多智能体 reinforcement 学习中的应用 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.03771)  

**Abstract**: World models that infer and predict environmental dynamics are foundational to embodied intelligence. However, their potential is often limited by the finite complexity and implicit biases of hand-crafted training environments. To develop truly generalizable and robust agents, we need environments that scale in complexity alongside the agents learning within them. In this work, we reframe the challenge of environment generation as the problem of learning a goal-conditioned, generative world model. We propose a system where a generative **Attacker** agent learns an implicit world model to synthesize increasingly difficult challenges for a team of cooperative **Defender** agents. The Attacker's objective is not passive prediction, but active, goal-driven interaction: it models and generates world states (i.e., configurations of enemy units) specifically to exploit the Defenders' weaknesses. Concurrently, the embodied Defender team learns a cooperative policy to overcome these generated worlds. This co-evolutionary dynamic creates a self-scaling curriculum where the world model continuously adapts to challenge the decision-making policy of the agents, providing an effectively infinite stream of novel and relevant training scenarios. We demonstrate that this framework leads to the emergence of complex behaviors, such as the world model learning to generate flanking and shielding formations, and the defenders learning coordinated focus-fire and spreading tactics. Our findings position adversarial co-evolution as a powerful method for learning instrumental world models that drive agents toward greater strategic depth and robustness. 

**Abstract (ZH)**: 基于目标的生成式世界模型在生成性强健代理中的应用 

---
# ARDO: A Weak Formulation Deep Neural Network Method for Elliptic and Parabolic PDEs Based on Random Differences of Test Functions 

**Title (ZH)**: ARDO：基于随机测试函数差值的弱形式深神经网络方法用于椭圆和抛物型偏微分方程 

**Authors**: Wei Cai, Andrew Qing He  

**Link**: [PDF](https://arxiv.org/pdf/2509.03757)  

**Abstract**: We propose ARDO method for solving PDEs and PDE-related problems with deep learning techniques. This method uses a weak adversarial formulation but transfers the random difference operator onto the test function. The main advantage of this framework is that it is fully derivative-free with respect to the solution neural network. This framework is particularly suitable for Fokker-Planck type second-order elliptic and parabolic PDEs. 

**Abstract (ZH)**: 我们提出了一种ARDO方法，结合深度学习技术求解偏微分方程及其相关问题。该方法采用弱对抗形式，但将随机差分算子转移至测试函数。该框架的主要优势是对解神经网络完全无导数要求。该框架特别适用于Fokker-Planck类型二阶椭圆和抛物线偏微分方程。 

---
# STA-Net: A Decoupled Shape and Texture Attention Network for Lightweight Plant Disease Classification 

**Title (ZH)**: STA-Net：一种解耦形状和纹理注意力网络的轻量级植物疾病分类方法 

**Authors**: Zongsen Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03754)  

**Abstract**: Responding to rising global food security needs, precision agriculture and deep learning-based plant disease diagnosis have become crucial. Yet, deploying high-precision models on edge devices is challenging. Most lightweight networks use attention mechanisms designed for generic object recognition, which poorly capture subtle pathological features like irregular lesion shapes and complex textures. To overcome this, we propose a twofold solution: first, using a training-free neural architecture search method (DeepMAD) to create an efficient network backbone for edge devices; second, introducing the Shape-Texture Attention Module (STAM). STAM splits attention into two branches -- one using deformable convolutions (DCNv4) for shape awareness and the other using a Gabor filter bank for texture awareness. On the public CCMT plant disease dataset, our STA-Net model (with 401K parameters and 51.1M FLOPs) reached 89.00% accuracy and an F1 score of 88.96%. Ablation studies confirm STAM significantly improves performance over baseline and standard attention models. Integrating domain knowledge via decoupled attention thus presents a promising path for edge-deployed precision agriculture AI. The source code is available at this https URL. 

**Abstract (ZH)**: 应对全球不断增长的粮食安全需求，基于精准农业和深度学习的植物病害诊断变得至关重要。然而，将高精度模型部署在边缘设备上颇具挑战。大多数轻量级网络使用针对通用对象识别设计的注意力机制，这些机制对捕捉不规则病斑形状和复杂纹理等细微病理特征的效果不佳。为克服这一问题，我们提出了一种双管齐下的解决方案：首先，使用无训练的神经架构搜索方法（DeepMAD）为边缘设备构建高效的网络基础架构；其次，引入了Shape-Texture Attention Module（STAM）。STAM将注意力机制分为两个分支：一个使用可变形卷积（DCNv4）进行形状感知，另一个使用Gabor滤波器库进行纹理感知。在公开的CCMT植物病害数据集上，我们的STA-Net模型（包含40.1万参数和51.1百万 floating point operations per second (FLOPs)）达到了89.00%的准确率和88.96%的F1分数。消融研究表明，STAM显著提高了性能，超越了基线和标准注意力模型。通过解耦注意力机制集成领域知识，从而为边缘部署的精准农业AI开辟了一条有前景的道路。源代码可从此处访问。 

---
# Designing Gaze Analytics for ELA Instruction: A User-Centered Dashboard with Conversational AI Support 

**Title (ZH)**: 基于用户中心设计与对话式AI支持的ELA教学注视分析仪表板设计 

**Authors**: Eduardo Davalos, Yike Zhang, Shruti Jain, Namrata Srivastava, Trieu Truong, Nafees-ul Haque, Tristan Van, Jorge Salas, Sara McFadden, Sun-Joo Cho, Gautam Biswas, Amanda Goodwin  

**Link**: [PDF](https://arxiv.org/pdf/2509.03741)  

**Abstract**: Eye-tracking offers rich insights into student cognition and engagement, but remains underutilized in classroom-facing educational technology due to challenges in data interpretation and accessibility. In this paper, we present the iterative design and evaluation of a gaze-based learning analytics dashboard for English Language Arts (ELA), developed through five studies involving teachers and students. Guided by user-centered design and data storytelling principles, we explored how gaze data can support reflection, formative assessment, and instructional decision-making. Our findings demonstrate that gaze analytics can be approachable and pedagogically valuable when supported by familiar visualizations, layered explanations, and narrative scaffolds. We further show how a conversational agent, powered by a large language model (LLM), can lower cognitive barriers to interpreting gaze data by enabling natural language interactions with multimodal learning analytics. We conclude with design implications for future EdTech systems that aim to integrate novel data modalities in classroom contexts. 

**Abstract (ZH)**: 眼动追踪为洞察学生认知和参与提供了丰富的见解，但在面向课堂的教育技术中由于数据解释和获取的挑战仍被严重低估。本文介绍了通过五项涉及教师和学生的研究，迭代设计和评估的一种基于凝视的学习分析仪表板，用于英语语言艺术（ELA）的教学应用。受用户中心设计和数据叙事原则的指导，我们探讨了凝视数据如何支持反思、形成性评估和教学决策。研究结果表明，当结合熟悉的可视化、分层解释和叙事支架时，凝视分析可以变得易于理解和具有教育价值。我们进一步展示了由大型语言模型（LLM）驱动的对话代理如何通过使多模态学习分析中的自然语言交互成为可能，降低解析凝视数据的认知障碍。最后，我们提出了未来旨在课堂环境中整合新型数据模态的教育技术系统的设计启示。 

---
# Sparse Autoencoder Neural Operators: Model Recovery in Function Spaces 

**Title (ZH)**: 稀疏自动编码神经算子：函数空间中的模型恢复 

**Authors**: Bahareh Tolooshams, Ailsa Shen, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.03738)  

**Abstract**: We frame the problem of unifying representations in neural models as one of sparse model recovery and introduce a framework that extends sparse autoencoders (SAEs) to lifted spaces and infinite-dimensional function spaces, enabling mechanistic interpretability of large neural operators (NO). While the Platonic Representation Hypothesis suggests that neural networks converge to similar representations across architectures, the representational properties of neural operators remain underexplored despite their growing importance in scientific computing. We compare the inference and training dynamics of SAEs, lifted-SAE, and SAE neural operators. We highlight how lifting and operator modules introduce beneficial inductive biases, enabling faster recovery, improved recovery of smooth concepts, and robust inference across varying resolutions, a property unique to neural operators. 

**Abstract (ZH)**: 我们将神经模型中统一表示的问题框定为稀疏模型恢复问题，并提出了一种框架，该框架将稀疏自编码器（SAEs）扩展到提升空间和无穷维函数空间，从而使大型神经算子（NO）具备机械可解释性。尽管理想表示假设表明神经网络在不同架构中会收敛到相似的表示，但神经算子的表示性质尚未得到充分探索，尽管它们在科学计算中的重要性日益增长。我们比较了稀疏自编码器（SAEs）、提升-SAE和神经算子的推断和训练动力学。我们强调了提升和算子模块引入的有益归纳偏置，这些偏置使模型能够更快地恢复、更好地恢复平滑概念，并在不同分辨率下实现鲁棒推断，这是神经算子独有的特性。 

---
# Differentiable Entropy Regularization for Geometry and Neural Networks 

**Title (ZH)**: 可微分熵正则化：几何与神经网络 

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.03733)  

**Abstract**: We introduce a differentiable estimator of range-partition entropy, a recent concept from computational geometry that enables algorithms to adapt to the "sortedness" of their input. While range-partition entropy provides strong guarantees in algorithm design, it has not yet been made accessible to deep learning. In this work, we (i) propose the first differentiable approximation of range-partition entropy, enabling its use as a trainable loss or regularizer; (ii) design EntropyNet, a neural module that restructures data into low-entropy forms to accelerate downstream instance-optimal algorithms; and (iii) extend this principle beyond geometry by applying entropy regularization directly to Transformer attention. Across tasks, we demonstrate that differentiable entropy improves efficiency without degrading correctness: in geometry, our method achieves up to $4.1\times$ runtime speedups with negligible error ($<0.2%$); in deep learning, it induces structured attention patterns that yield 6% higher accuracy at 80% sparsity compared to L1 baselines. Our theoretical analysis provides approximation bounds for the estimator, and extensive ablations validate design choices. These results suggest that entropy-bounded computation is not only theoretically elegant but also a practical mechanism for adaptive learning, efficiency, and structured representation. 

**Abstract (ZH)**: 一种可微分的范围分区熵估计器及其在深度学习中的应用 

---
# MLSD: A Novel Few-Shot Learning Approach to Enhance Cross-Target and Cross-Domain Stance Detection 

**Title (ZH)**: MLSD: 一种新型的少量样本学习方法以增强跨目标和跨域立场检测 

**Authors**: Parush Gera, Tempestt Neal  

**Link**: [PDF](https://arxiv.org/pdf/2509.03725)  

**Abstract**: We present the novel approach for stance detection across domains and targets, Metric Learning-Based Few-Shot Learning for Cross-Target and Cross-Domain Stance Detection (MLSD). MLSD utilizes metric learning with triplet loss to capture semantic similarities and differences between stance targets, enhancing domain adaptation. By constructing a discriminative embedding space, MLSD allows a cross-target or cross-domain stance detection model to acquire useful examples from new target domains. We evaluate MLSD in multiple cross-target and cross-domain scenarios across two datasets, showing statistically significant improvement in stance detection performance across six widely used stance detection models. 

**Abstract (ZH)**: 基于度量学习的少样本学习在跨目标和跨域立场检测中的新方法：度量学习基于三元损失的少样本立场检测（MLSD） 

---
# From Federated Learning to $\mathbb{X}$-Learning: Breaking the Barriers of Decentrality Through Random Walks 

**Title (ZH)**: 从联邦学习到$\mathbb{X}$学习：通过随机游走打破中心化壁垒 

**Authors**: Allan Salihovic, Payam Abdisarabshali, Michael Langberg, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2509.03709)  

**Abstract**: We provide our perspective on $\mathbb{X}$-Learning ($\mathbb{X}$L), a novel distributed learning architecture that generalizes and extends the concept of decentralization. Our goal is to present a vision for $\mathbb{X}$L, introducing its unexplored design considerations and degrees of freedom. To this end, we shed light on the intuitive yet non-trivial connections between $\mathbb{X}$L, graph theory, and Markov chains. We also present a series of open research directions to stimulate further research. 

**Abstract (ZH)**: 我们提供了关于$\mathbb{X}$-Learning ($\mathbb{X}$L)的见解，$\mathbb{X}$L是一种新颖的分布式学习架构，扩展了去中心化概念。我们的目标是提出$\mathbb{X}$L的愿景，介绍其未探索的设计考量和自由度。为此，我们揭示了$\mathbb{X}$L、图论和马尔可夫链之间直观但非平凡的联系，并提出了一系列开放的研究方向以激发进一步的研究。 

---
# Hierarchical Federated Foundation Models over Wireless Networks for Multi-Modal Multi-Task Intelligence: Integration of Edge Learning with D2D/P2P-Enabled Fog Learning Architectures 

**Title (ZH)**: 无线网络中多模态多任务智能的分层联邦基础模型：边缘学习与D2D/P2P启用的雾计算学习架构的集成 

**Authors**: Payam Abdisarabshali, Fardis Nadimi, Kasra Borazjani, Naji Khosravan, Minghui Liwang, Wei Ni, Dusit Niyato, Michael Langberg, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2509.03695)  

**Abstract**: The rise of foundation models (FMs) has reshaped the landscape of machine learning. As these models continued to grow, leveraging geo-distributed data from wireless devices has become increasingly critical, giving rise to federated foundation models (FFMs). More recently, FMs have evolved into multi-modal multi-task (M3T) FMs (e.g., GPT-4) capable of processing diverse modalities across multiple tasks, which motivates a new underexplored paradigm: M3T FFMs. In this paper, we unveil an unexplored variation of M3T FFMs by proposing hierarchical federated foundation models (HF-FMs), which in turn expose two overlooked heterogeneity dimensions to fog/edge networks that have a direct impact on these emerging models: (i) heterogeneity in collected modalities and (ii) heterogeneity in executed tasks across fog/edge nodes. HF-FMs strategically align the modular structure of M3T FMs, comprising modality encoders, prompts, mixture-of-experts (MoEs), adapters, and task heads, with the hierarchical nature of fog/edge infrastructures. Moreover, HF-FMs enable the optional usage of device-to-device (D2D) communications, enabling horizontal module relaying and localized cooperative training among nodes when feasible. Through delving into the architectural design of HF-FMs, we highlight their unique capabilities along with a series of tailored future research directions. Finally, to demonstrate their potential, we prototype HF-FMs in a wireless network setting and release the open-source code for the development of HF-FMs with the goal of fostering exploration in this untapped field (GitHub: this https URL). 

**Abstract (ZH)**: 基础模型的兴起重塑了机器学习的格局。随着这些模型的不断增长，利用来自无线设备的地理分布式数据变得越来越关键，从而催生了联邦基础模型（FFMs）。近年来，基础模型演进为多模态多任务（M3T）基础模型（例如GPT-4），能够跨多个任务处理多种模态数据，这激发了一种新的待探索范式：M3T FFMs。在本文中，我们通过提出分层联邦基础模型（HF-FMs）来揭示M3T FFMs的一种未探索变体，并进而揭示了对雾/边缘网络有直接影响的两种未被重视的异质性维度：（i）收集模态的异质性；（ii）雾/边缘节点上执行任务的异质性。HF-FMs战略性地将M3T基础模型的模块化结构——模态编码器、提示、专家混合体（MoEs）、适配器和任务头——与雾/边缘基础设施的层次结构相融合。此外，HF-FMs允许在适当情况下使用设备到设备（D2D）通信，实现模块的水平转发和节点间的局部协同训练。通过深入探讨HF-FMs的架构设计，我们突显了其独特的功能，并提出了一系列定制的未来研究方向。最后，为了展示其潜力，我们在无线网络环境中原型实现HF-FMs，并开源HF-FMs的开发代码，以促进对该领域未开发领域的探索（GitHub: this https URL）。 

---
# LuxDiT: Lighting Estimation with Video Diffusion Transformer 

**Title (ZH)**: LuxDiT: 视频扩散变换器的照明 estimation 

**Authors**: Ruofan Liang, Kai He, Zan Gojcic, Igor Gilitschenski, Sanja Fidler, Nandita Vijaykumar, Zian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03680)  

**Abstract**: Estimating scene lighting from a single image or video remains a longstanding challenge in computer vision and graphics. Learning-based approaches are constrained by the scarcity of ground-truth HDR environment maps, which are expensive to capture and limited in diversity. While recent generative models offer strong priors for image synthesis, lighting estimation remains difficult due to its reliance on indirect visual cues, the need to infer global (non-local) context, and the recovery of high-dynamic-range outputs. We propose LuxDiT, a novel data-driven approach that fine-tunes a video diffusion transformer to generate HDR environment maps conditioned on visual input. Trained on a large synthetic dataset with diverse lighting conditions, our model learns to infer illumination from indirect visual cues and generalizes effectively to real-world scenes. To improve semantic alignment between the input and the predicted environment map, we introduce a low-rank adaptation finetuning strategy using a collected dataset of HDR panoramas. Our method produces accurate lighting predictions with realistic angular high-frequency details, outperforming existing state-of-the-art techniques in both quantitative and qualitative evaluations. 

**Abstract (ZH)**: 从单张图像或视频估计场景光照仍然是计算机视觉和图形学中的一个长期挑战。基于学习的方法受限于高动态范围环境图的地真数据稀缺，这些数据昂贵且多样性有限。尽管最近的生成模型提供了强大的先验知识用于图像合成，但由于依赖于间接视觉线索、需要推断全局（非局部）上下文以及恢复高动态范围输出，光照估计仍然困难。我们提出LuxDiT，这是一种新颖的数据驱动方法，通过微调视频扩散变换器来生成条件于视觉输入的高动态范围环境图。在包含各种光照条件的大型合成数据集上训练，我们的模型学会了从间接视觉线索中推断照明，并能够有效地泛化到真实场景。为了改进输入和预测环境图之间的语义对齐，我们引入了一种基于收集的高动态范围全景图数据集的低秩适应微调策略。我们的方法在定量和定性评估中均产生准确的光照预测，并具有现实的角方向高频细节，优于现有最先进的技术。 

---
# Insights from Gradient Dynamics: Gradient Autoscaled Normalization 

**Title (ZH)**: 从梯度动态中获得的见解：梯度自动缩放规范化 

**Authors**: Vincent-Daniel Yun  

**Link**: [PDF](https://arxiv.org/pdf/2509.03677)  

**Abstract**: Gradient dynamics play a central role in determining the stability and generalization of deep neural networks. In this work, we provide an empirical analysis of how variance and standard deviation of gradients evolve during training, showing consistent changes across layers and at the global scale in convolutional networks. Motivated by these observations, we propose a hyperparameter-free gradient normalization method that aligns gradient scaling with their natural evolution. This approach prevents unintended amplification, stabilizes optimization, and preserves convergence guarantees. Experiments on the challenging CIFAR-100 benchmark with ResNet-20, ResNet-56, and VGG-16-BN demonstrate that our method maintains or improves test accuracy even under strong generalization. Beyond practical performance, our study highlights the importance of directly tracking gradient dynamics, aiming to bridge the gap between theoretical expectations and empirical behaviors, and to provide insights for future optimization research. 

**Abstract (ZH)**: 梯度动态在决定深度神经网络的稳定性和泛化能力中发挥着核心作用。本文提供了梯度方差和标准差在训练过程中演变的实证分析，展示了卷积网络中各层及全局尺度上的一致变化。受这些观察的启发，我们提出了一种无超参数的梯度归一化方法，使梯度缩放与它们的自然演变保持一致。该方法防止了不必要的放大，稳定了优化过程，并保留了收敛保证。在具有挑战性的CIFAR-100基准测试中，使用ResNet-20、ResNet-56和VGG-16-BN进行的实验表明，我们的方法在强泛化条件下能够维持或提高测试精度。除了实际性能，本研究所强调直接跟踪梯度动态的重要性，旨在弥合理论期望与实验表现之间的差距，并为未来的优化研究提供见解。 

---
# Efficient Virtuoso: A Latent Diffusion Transformer Model for Goal-Conditioned Trajectory Planning 

**Title (ZH)**: 高效的维鲁佐：一种用于目标条件轨迹规划的潜扩散变换器模型 

**Authors**: Antonio Guillen-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2509.03658)  

**Abstract**: The ability to generate a diverse and plausible distribution of future trajectories is a critical capability for autonomous vehicle planning systems. While recent generative models have shown promise, achieving high fidelity, computational efficiency, and precise control remains a significant challenge. In this paper, we present the \textbf{Efficient Virtuoso}, a conditional latent diffusion model for goal-conditioned trajectory planning. Our approach introduces a novel two-stage normalization pipeline that first scales trajectories to preserve their geometric aspect ratio and then normalizes the resulting PCA latent space to ensure a stable training target. The denoising process is performed efficiently in this low-dimensional latent space by a simple MLP denoiser, which is conditioned on a rich scene context fused by a powerful Transformer-based StateEncoder. We demonstrate that our method achieves state-of-the-art performance on the Waymo Open Motion Dataset, reaching a \textbf{minADE of 0.25}. Furthermore, through a rigorous ablation study on goal representation, we provide a key insight: while a single endpoint goal can resolve strategic ambiguity, a richer, multi-step sparse route is essential for enabling the precise, high-fidelity tactical execution that mirrors nuanced human driving behavior. 

**Abstract (ZH)**: 高效虚拟大师：一种条件潜扩散模型在目标条件轨迹规划中的应用 

---
# Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators 

**Title (ZH)**: 打破镜像：基于激活的减轻LLM评估者自我偏见方法 

**Authors**: Dani Roytburg, Matthew Bozoukov, Matthew Nguyen, Jou Barzdukas, Simon Fu, Narmeen Oozeer  

**Link**: [PDF](https://arxiv.org/pdf/2509.03647)  

**Abstract**: Large language models (LLMs) increasingly serve as automated evaluators, yet they suffer from "self-preference bias": a tendency to favor their own outputs over those of other models. This bias undermines fairness and reliability in evaluation pipelines, particularly for tasks like preference tuning and model routing. We investigate whether lightweight steering vectors can mitigate this problem at inference time without retraining. We introduce a curated dataset that distinguishes self-preference bias into justified examples of self-preference and unjustified examples of self-preference, and we construct steering vectors using two methods: Contrastive Activation Addition (CAA) and an optimization-based approach. Our results show that steering vectors can reduce unjustified self-preference bias by up to 97\%, substantially outperforming prompting and direct preference optimization baselines. Yet steering vectors are unstable on legitimate self-preference and unbiased agreement, implying self-preference spans multiple or nonlinear directions. This underscores both their promise and limits as safeguards for LLM-as-judges and motivates more robust interventions. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地作为自动评估器使用，但它们遭受“自我偏好偏差”的困扰：倾向于偏好自己的输出而非其他模型的输出。这种偏差损害了评估管道中的公平性和可靠性，特别是在偏好调整和模型路由等任务中。我们研究了轻量级引导向量是否可以在不重新训练的情况下，在推理时缓解这一问题。我们引入了一个精心策划的数据集，将自我偏好偏差区分为有正当理由的自我偏好和无正当理由的自我偏好示例，并使用对比激活添加（CAA）和优化方法构建了引导向量。实验结果表明，引导向量可将无正当理由的自我偏好偏差降低最多97%，显著优于提示和直接偏好优化基线方法。然而，引导向量在合理的自我偏好和无偏见一致性的稳定性上存在不足，表明自我偏好可能跨越多个或非线性的方向。这既突显了它们作为LLM作为评判者保护措施的潜力和局限性，也推动了更稳健干预措施的发展。 

---
# CEHR-GPT: A Scalable Multi-Task Foundation Model for Electronic Health Records 

**Title (ZH)**: CEHR-GPT：一种适用于电子健康记录的可扩展多任务基础模型 

**Authors**: Chao Pang, Jiheum Park, Xinzhuo Jiang, Nishanth Parameshwar Pavinkurve, Krishna S. Kalluri, Shalmali Joshi, Noémie Elhadad, Karthik Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2509.03643)  

**Abstract**: Electronic Health Records (EHRs) provide a rich, longitudinal view of patient health and hold significant potential for advancing clinical decision support, risk prediction, and data-driven healthcare research. However, most artificial intelligence (AI) models for EHRs are designed for narrow, single-purpose tasks, limiting their generalizability and utility in real-world settings. Here, we present CEHR-GPT, a general-purpose foundation model for EHR data that unifies three essential capabilities - feature representation, zero-shot prediction, and synthetic data generation - within a single architecture. To support temporal reasoning over clinical sequences, \cehrgpt{} incorporates a novel time-token-based learning framework that explicitly encodes patients' dynamic timelines into the model structure. CEHR-GPT demonstrates strong performance across all three tasks and generalizes effectively to external datasets through vocabulary expansion and fine-tuning. Its versatility enables rapid model development, cohort discovery, and patient outcome forecasting without the need for task-specific retraining. 

**Abstract (ZH)**: 电子健康记录（EHRs）提供了患者健康状况的丰富 longitudinal 视角，并且在临床决策支持、风险预测和数据驱动的医疗健康研究方面具有重要的潜力。然而，大多数用于EHR的AI模型针对单一目的任务进行设计，限制了它们在实际应用场景中的通用性和实用性。在这里，我们介绍了CEHR-GPT，这是一种适用于EHR数据的通用基础模型，统一了特征表示、零样本预测和合成数据生成这三种核心能力。为了支持临床序列上的时间推理，CEHR-GPT采用了新颖的时间标记学习框架，明确地将患者动态时间线编码到模型结构中。CEHR-GPT在所有三个任务上表现出色，并通过词汇扩展和微调有效泛化到外部数据集。其灵活性使得能够无需特定任务重训即可快速开发模型、发现患者群体和预测患者结局。 

---
# treeX: Unsupervised Tree Instance Segmentation in Dense Forest Point Clouds 

**Title (ZH)**: treeX：密集森林点云中的无监督树实例分割 

**Authors**: Josafat-Mattias Burmeister, Andreas Tockner, Stefan Reder, Markus Engel, Rico Richter, Jan-Peter Mund, Jürgen Döllner  

**Link**: [PDF](https://arxiv.org/pdf/2509.03633)  

**Abstract**: Close-range laser scanning provides detailed 3D captures of forest stands but requires efficient software for processing 3D point cloud data and extracting individual trees. Although recent studies have introduced deep learning methods for tree instance segmentation, these approaches require large annotated datasets and substantial computational resources. As a resource-efficient alternative, we present a revised version of the treeX algorithm, an unsupervised method that combines clustering-based stem detection with region growing for crown delineation. While the original treeX algorithm was developed for personal laser scanning (PLS) data, we provide two parameter presets, one for ground-based laser scanning (stationary terrestrial - TLS and PLS), and one for UAV-borne laser scanning (ULS). We evaluated the method on six public datasets (FOR-instance, ForestSemantic, LAUTx, NIBIO MLS, TreeLearn, Wytham Woods) and compared it to six open-source methods (original treeX, treeiso, RayCloudTools, ForAINet, SegmentAnyTree, TreeLearn). Compared to the original treeX algorithm, our revision reduces runtime and improves accuracy, with instance detection F$_1$-score gains of +0.11 to +0.49 for ground-based data. For ULS data, our preset achieves an F$_1$-score of 0.58, whereas the original algorithm fails to segment any correct instances. For TLS and PLS data, our algorithm achieves accuracy similar to recent open-source methods, including deep learning. Given its algorithmic design, we see two main applications for our method: (1) as a resource-efficient alternative to deep learning approaches in scenarios where the data characteristics align with the method design (sufficient stem visibility and point density), and (2) for the semi-automatic generation of labels for deep learning models. To enable broader adoption, we provide an open-source Python implementation in the pointtree package. 

**Abstract (ZH)**: 近距离激光扫描提供了森林立地的详细3D捕获，但需要高效的软件来处理3D点云数据并提取单株树。尽管近期的研究引入了深度学习方法进行树实例分割，这些方法需要大型标注数据集和大量的计算资源。作为一种资源高效的替代方案，我们提出了树X算法的一种修订版本，该算法结合基于聚类的主干检测和区域生长的树冠界定，是一种无监督的方法。原树X算法是为个人激光扫描(PLS)数据开发的，我们提供了两个参数预设，一个适用于基于地面激光扫描（固定地面激光扫描-TLS和PLS），另一个适用于无人机搭载激光扫描（ULS）。我们在六个公开数据集（FOR-instance、ForestSemantic、LAUTx、NIBIO MLS、TreeLearn、Wytham Woods）上评估了该方法，并将其与六种开源方法（原始树X、treeiso、RayCloudTools、ForAINet、SegmentAnyTree、TreeLearn）进行了比较。与原始树X算法相比，我们的修订版本减少了运行时间并提高了准确性，对于基于地面的数据，实例检测F₁-分数提高了0.11到0.49。对于ULS数据，我们的预设达到了0.58的F₁-分数，而原始算法无法分割任何正确实例。对于TLS和PLS数据，我们的算法在准确性上类似于最近的开源方法，包括深度学习方法。鉴于其算法设计，我们看到该方法的两个主要应用领域：（1）在数据特征与方法设计相匹配的情况下（充分的主干可见性和点密度），作为深度学习方法的一种资源高效替代方案；（2）用于深度学习模型的半自动标签生成。为了促进更广泛的应用，我们在pointtree包中提供了该方法的开源Python实现。 

---
# E-ARMOR: Edge case Assessment and Review of Multilingual Optical Character Recognition 

**Title (ZH)**: E-ARMOR: 多语言光学字符识别的边缘案例评估与审查 

**Authors**: Aryan Gupta, Anupam Purwar  

**Link**: [PDF](https://arxiv.org/pdf/2509.03615)  

**Abstract**: Optical Character Recognition (OCR) in multilingual, noisy, and diverse real-world images remains a significant challenge for optical character recognition systems. With the rise of Large Vision-Language Models (LVLMs), there is growing interest in their ability to generalize and reason beyond fixed OCR pipelines. In this work, we introduce Sprinklr-Edge-OCR, a novel OCR system built specifically optimized for edge deployment in resource-constrained environments. We present a large-scale comparative evaluation of five state-of-the-art LVLMs (InternVL, Qwen, GOT OCR, LLaMA, MiniCPM) and two traditional OCR systems (Sprinklr-Edge-OCR, SuryaOCR) on a proprietary, doubly hand annotated dataset of multilingual (54 languages) images. Our benchmark covers a broad range of metrics including accuracy, semantic consistency, language coverage, computational efficiency (latency, memory, GPU usage), and deployment cost. To better reflect real-world applicability, we also conducted edge case deployment analysis, evaluating model performance on CPU only environments. Among the results, Qwen achieved the highest precision (0.54), while Sprinklr-Edge-OCR delivered the best overall F1 score (0.46) and outperformed others in efficiency, processing images 35 faster (0.17 seconds per image on average) and at less than 0.01 of the cost (0.006 USD per 1,000 images) compared to LVLM. Our findings demonstrate that the most optimal OCR systems for edge deployment are the traditional ones even in the era of LLMs due to their low compute requirements, low latency, and very high affordability. 

**Abstract (ZH)**: 多语言、噪声和多样化的现实世界图像中的光学字符识别（OCR）仍是对光学字符识别系统的一大挑战。随着大型视觉-语言模型（LVLM）的兴起，人们越来越关注其在固定OCR管道之外的泛化和推理能力。本研究介绍了Sprinklr-Edge-OCR，一种专门为资源受限环境下的边缘部署优化的新型OCR系统。我们对五种先进的LVLM（InternVL、Qwen、GOT OCR、LLaMA、MiniCPM）和两种传统OCR系统（Sprinklr-Edge-OCR、SuryaOCR）进行了一项大规模的比较评估，使用的是一个包含54种语言的专有双标注图像数据集。我们的基准测试涵盖了准确性、语义一致性、语言覆盖率、计算效率（延迟、内存、GPU使用量）和部署成本等多项指标。为了更好地反映实际应用，我们还进行了边缘案例部署分析，评估模型在仅使用CPU环境下的性能。结果表明，Qwen在精确度方面最高（0.54），而Sprinklr-Edge-OCR取得了最佳的综合F1分数（0.46），在效率方面也优于其他系统，平均每处理一张图像仅需0.17秒，并且成本仅为LVLM的0.006美元/1000张图像的十分之一。我们的研究表明，在LLM时代，传统OCR系统仍然是边缘部署的最佳选择，这主要归因于其低计算需求、低延迟和极高的经济性。 

---
# The Optimiser Hidden in Plain Sight: Training with the Loss Landscape's Induced Metric 

**Title (ZH)**: 显而易见的优化器：基于损失景观诱导度量的训练 

**Authors**: Thomas R. Harvey  

**Link**: [PDF](https://arxiv.org/pdf/2509.03594)  

**Abstract**: We present a class of novel optimisers for training neural networks that makes use of the Riemannian metric naturally induced when the loss landscape is embedded in higher-dimensional space. This is the same metric that underlies common visualisations of loss landscapes. By taking this geometric perspective literally and using the induced metric, we develop a new optimiser and compare it to existing methods, namely: SGD, Adam, AdamW, and Muon, across a range of tasks and architectures. Empirically, we conclude that this new class of optimisers is highly effective in low dimensional examples, and provides slight improvement over state-of-the-art methods for training neural networks. These new optimisers have theoretically desirable properties. In particular, the effective learning rate is automatically decreased in regions of high curvature acting as a smoothed out form of gradient clipping. Similarly, one variant of these optimisers can also be viewed as inducing an effective scheduled learning rate and decoupled weight decay is the natural choice from our geometric perspective. The basic method can be used to modify any existing preconditioning method. The new optimiser has a computational complexity comparable to that of Adam. 

**Abstract (ZH)**: 我们提出了一类新型神经网络训练优化器，利用损失景观嵌入高维空间时自然诱导的黎曼度量。通过将这种几何视角字面化并利用诱导度量，我们开发了一种新的优化器，并将其与现有的方法（即SGD、Adam、AdamW和Muon）在多种任务和架构上进行了比较。我们实验证明，这种新的优化器在低维度示例中效果显著，并且在训练神经网络方面比最先进的方法稍有改进。这些新的优化器具有理论上的 desirable 属性。特别是，在高曲率区域自适应降低有效的学习率，起到平滑梯度截断的作用。同时，这些优化器的一种变体还可以被视为诱导有效的学习率调度，并且在我们的几何视角下，解耦的权重衰减是自然的选择。基本方法可以用于修改任何现有的预处理方法。新优化器的计算复杂度与Adam相当。 

---
# A software security review on Uganda's Mobile Money Services: Dr. Jim Spire's tweets sentiment analysis 

**Title (ZH)**: 乌干达移动钱服务软件安全审查：吉姆·斯派尔博士推特情感分析 

**Authors**: Nsengiyumva Wilberforce  

**Link**: [PDF](https://arxiv.org/pdf/2509.03545)  

**Abstract**: The proliferation of mobile money in Uganda has been a cornerstone of financial inclusion, yet its security mechanisms remain a critical concern. This study investigates a significant public response to perceived security failures: the #StopAirtelThefty Twitter campaign of August 2025 Sparked by an incident publicized by Dr. Jim Spire Ssentongo where a phone thief accessed a victim's account, withdrew funds, and procured a loan, the campaign revealed deep seated public anxiety over the safety of mobile money. This research employs qualitative analysis to systematically examine the complaints raised during this campaign, extracting key themes related to security vulnerabilities and user dissatisfaction. By synthesizing these public sentiments, the paper provides crucial insights into the specific security gaps experienced by users and situates these findings within the larger framework of Uganda's mobile money regulatory and operational environment. The study concludes with implications for providers, policymakers, and the future of secure digital finance in Uganda. 

**Abstract (ZH)**: 乌干达移动货币的普及是金融包容性的基石，但其安全机制仍是关键问题。一项研究探讨了对 perceived 安全失败的显著公众反应：2025年8月的#StopAirtelThefty Twitter 活动。该活动起源于 Dr. Jim Spire Ssentongo 公布的一起事件，一名手机窃贼侵入受害者的账户，提取资金，并获取贷款，揭示了公众对移动货币安全的深深关切。本研究采用定性分析方法系统地审查了该活动期间提出的投诉，提取了与安全漏洞和用户不满相关的关键主题。通过综合这些公众情绪，论文提供了对用户实际体验的安全漏洞的重要洞见，并将其置于乌干达移动货币监管和运营环境的大背景下。研究最后提出了对提供者、决策者以及乌干达安全数字金融未来的含义。 

---
# Improving Factuality in LLMs via Inference-Time Knowledge Graph Construction 

**Title (ZH)**: 通过推理时知识图构建提高大模型的事实可靠性 

**Authors**: Shanglin Wu, Lihui Liu, Jinho D. Choi, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2509.03540)  

**Abstract**: Large Language Models (LLMs) often struggle with producing factually consistent answers due to limitations in their parametric memory. Retrieval-Augmented Generation (RAG) methods address this issue by incorporating external knowledge from trusted sources at inference time. However, such methods typically treat knowledge as unstructured text, which limits their ability to support compositional reasoning and identify factual inconsistencies. To overcome these limitations, we propose a novel framework that dynamically constructs and expands knowledge graphs (KGs) during inference, integrating both internal knowledge extracted from LLMs and external information retrieved from external sources. Our method begins by extracting a seed KG from the question via prompting, followed by iterative expansion using the LLM's latent knowledge. The graph is then selectively refined through external retrieval, enhancing factual coverage and correcting inaccuracies. We evaluate our approach on three diverse factual QA benchmarks, demonstrating consistent improvements in factual accuracy, answer precision, and interpretability over baseline prompting and static KG-augmented methods. Our findings suggest that inference-time KG construction is a promising direction for enhancing LLM factuality in a structured, interpretable, and scalable manner. 

**Abstract (ZH)**: Large Language Models (LLMs)在事实一致性方面常受限于参数记忆的限制。检索增强生成（RAG）方法通过在推理时结合可信来源的外部知识来解决这一问题。然而，这类方法通常将知识视为无结构文本，这限制了它们支持组合推理和识别事实不一致的能力。为克服这些限制，我们提出了一种新型框架，在推理过程中动态构造和扩展知识图谱（KGs），整合LLMs内部提取的知识和外部来源检索到的信息。该方法首先通过提示从问题中提取种子KG，随后使用LLM的潜在知识进行迭代扩展。然后通过外部检索对图谱进行选择性 refinement，增强事实覆盖范围并纠正不准确性。我们在三个不同的事实QA基准上评估了我们的方法，结果表明与基线提示和静态KG增强方法相比，在事实准确性、答案精度和可解释性方面均有所改进。我们的研究结果表明，在结构化、可解释和可扩展的方式下增强LLM的事实性通过推理时构建KG是一种有前景的方向。 

---
# AR$^2$: Adversarial Reinforcement Learning for Abstract Reasoning in Large Language Models 

**Title (ZH)**: AR$^2$: 面向大型语言模型的抽象推理对抗强化学习 

**Authors**: Cheng-Kai Yeh, Hsing-Wang Lee, Chung-Hung Kuo, Hen-Hsen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.03537)  

**Abstract**: Abstraction--the ability to recognize and distill essential computational patterns from complex problem statements--is a foundational skill in computer science, critical both for human problem-solvers and coding-oriented large language models (LLMs). Despite recent advances in training LLMs for code generation using reinforcement learning (RL), most existing approaches focus primarily on superficial pattern recognition, overlooking explicit training for abstraction. In this study, we propose AR$^2$ (Adversarial Reinforcement Learning for Abstract Reasoning), a novel framework explicitly designed to enhance the abstraction abilities of LLMs. AR$^2$ employs a teacher model to transform kernel problems into narrative-rich, challenging descriptions without changing their fundamental logic. Simultaneously, a student coding model is trained to solve these complex narrative problems by extracting their underlying computational kernels. Experimental results demonstrate that AR$^2$ substantially improves the student model's accuracy on previously unseen, challenging programming tasks, underscoring abstraction as a key skill for enhancing LLM generalization. 

**Abstract (ZH)**: 基于对抗强化学习的抽象推理（AR²） 

---
# QuesGenie: Intelligent Multimodal Question Generation 

**Title (ZH)**: QuesGenie: 智能多模态问题生成 

**Authors**: Ahmed Mubarak, Amna Ahmed, Amira Nasser, Aya Mohamed, Fares El-Sadek, Mohammed Ahmed, Ahmed Salah, Youssef Sobhy  

**Link**: [PDF](https://arxiv.org/pdf/2509.03535)  

**Abstract**: In today's information-rich era, learners have access to abundant educational resources, but the lack of practice materials tailored to these resources presents a significant challenge. This project addresses that gap by developing a multi-modal question generation system that can automatically generate diverse question types from various content formats. The system features four major components: multi-modal input handling, question generation, reinforcement learning from human feedback (RLHF), and an end-to-end interactive interface. This project lays the foundation for automated, scalable, and intelligent question generation, carefully balancing resource efficiency, robust functionality and a smooth user experience. 

**Abstract (ZH)**: 在信息丰富的时代，学习者可以访问丰富的教育资源，但缺乏针对这些资源定制的练习材料构成了一个重大挑战。该项目通过开发一个能够从多种内容格式中自动生成多样化题型的多模态问题生成系统来解决这一缺口。该系统包含四大组件：多模态输入处理、问题生成、基于人类反馈的强化学习（RLHF）以及端到端的交互界面。该项目为自动化、可扩展和智能化的问题生成奠定了基础，精心平衡了资源效率、稳健功能和流畅的用户体验。 

---
# Real-Time Detection of Hallucinated Entities in Long-Form Generation 

**Title (ZH)**: 长文本生成中实时检测虚构实体 

**Authors**: Oscar Obeso, Andy Arditi, Javier Ferrando, Joshua Freeman, Cameron Holmes, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2509.03531)  

**Abstract**: Large language models are now routinely used in high-stakes applications where hallucinations can cause serious harm, such as medical consultations or legal advice. Existing hallucination detection methods, however, are impractical for real-world use, as they are either limited to short factual queries or require costly external verification. We present a cheap, scalable method for real-time identification of hallucinated tokens in long-form generations, and scale it effectively to 70B parameter models. Our approach targets \emph{entity-level hallucinations} -- e.g., fabricated names, dates, citations -- rather than claim-level, thereby naturally mapping to token-level labels and enabling streaming detection. We develop an annotation methodology that leverages web search to annotate model responses with grounded labels indicating which tokens correspond to fabricated entities. This dataset enables us to train effective hallucination classifiers with simple and efficient methods such as linear probes. Evaluating across four model families, our classifiers consistently outperform baselines on long-form responses, including more expensive methods such as semantic entropy (e.g., AUC 0.90 vs 0.71 for Llama-3.3-70B), and are also an improvement in short-form question-answering settings. Moreover, despite being trained only with entity-level labels, our probes effectively detect incorrect answers in mathematical reasoning tasks, indicating generalization beyond entities. While our annotation methodology is expensive, we find that annotated responses from one model can be used to train effective classifiers on other models; accordingly, we publicly release our datasets to facilitate reuse. Overall, our work suggests a promising new approach for scalable, real-world hallucination detection. 

**Abstract (ZH)**: 大规模语言模型现在在高风险应用中常规使用，如医疗咨询或法律建议，其中幻觉可能导致严重伤害。现有的幻觉检测方法在实际应用中不可行，因为它们要么仅限于短事实查询，要么需要昂贵的外部验证。我们提出了一种低成本、可扩展的方法，用于实时识别长文本生成中的幻觉标记，并将其有效扩展到700亿参数模型。我们的方法针对实体级幻觉——例如，虚构的名称、日期、引文——而不是断言级，从而自然映射到标记级别标签并使流式检测成为可能。我们开发了一种标注方法，利用网络搜索为模型响应添加基于实际的标签，指出哪些标记对应于虚构的实体。该数据集使我们能够使用简单的高效方法（如线性探针）训练有效的幻觉分类器。在四个模型家族的评估中，我们的分类器在长文本响应上始终优于基线方法，包括更昂贵的方法（如语义熵，例如Llama-3.3-70B的AUC 0.90 vs 0.71），并且在短文本问答设置中也有改进。此外，尽管仅使用实体级标签训练，我们的探针在数学推理任务中有效检测错误答案，表明其具有超越实体的一般化能力。虽然我们的标注方法成本高昂，但我们发现一种模型的标注响应可以用于训练其他模型的有效分类器；因此，我们公开发布了我们的数据集以促进重复使用。总体而言，我们的工作表明了一种有希望的新方法，用于可扩展的实际幻觉检测。 

---
# Multimodal Proposal for an AI-Based Tool to Increase Cross-Assessment of Messages 

**Title (ZH)**: 基于AI的多模态提案工具以增加消息交叉评估 

**Authors**: Alejandro Álvarez Castro, Joaquín Ordieres-Meré  

**Link**: [PDF](https://arxiv.org/pdf/2509.03529)  

**Abstract**: Earnings calls represent a uniquely rich and semi-structured source of financial communication, blending scripted managerial commentary with unscripted analyst dialogue. Although recent advances in financial sentiment analysis have integrated multi-modal signals, such as textual content and vocal tone, most systems rely on flat document-level or sentence-level models, failing to capture the layered discourse structure of these interactions. This paper introduces a novel multi-modal framework designed to generate semantically rich and structurally aware embeddings of earnings calls, by encoding them as hierarchical discourse trees. Each node, comprising either a monologue or a question-answer pair, is enriched with emotional signals derived from text, audio, and video, as well as structured metadata including coherence scores, topic labels, and answer coverage assessments. A two-stage transformer architecture is proposed: the first encodes multi-modal content and discourse metadata at the node level using contrastive learning, while the second synthesizes a global embedding for the entire conference. Experimental results reveal that the resulting embeddings form stable, semantically meaningful representations that reflect affective tone, structural logic, and thematic alignment. Beyond financial reporting, the proposed system generalizes to other high-stakes unscripted communicative domains such as tele-medicine, education, and political discourse, offering a robust and explainable approach to multi-modal discourse representation. This approach offers practical utility for downstream tasks such as financial forecasting and discourse evaluation, while also providing a generalizable method applicable to other domains involving high-stakes communication. 

**Abstract (ZH)**: earnings电话代表了一种独特丰富且半结构化的财务沟通来源，将剧本化的企业管理评论与非剧本化的分析师对话相结合。尽管最近在财务情绪分析方面的进展已经整合了多模态信号，如文本内容和语音语调，但大多数系统仍然依赖于平面的文档级或句级模型，未能捕捉到这些互动的多层次话语结构。本文介绍了一种新颖的多模态框架，旨在通过将earnings电话编码为分层话语树来生成语义丰富且结构意识强的嵌入。每个节点，包括独白或问答对，都会从文本、音频和视频中提取情感信号，并结合包括连贯性评分、主题标签和答案覆盖率评估在内的结构化元数据。提出了一种两阶段变换器架构：第一阶段通过对比学习在节点级别编码多模态内容和话语元数据，而第二阶段综合生成整个会议的全局嵌入。实验结果表明，生成的嵌入形成了稳定且语义上有意义的表示，反映了情感基调、结构逻辑和主题对齐。除了财务报告，所提出的系统还适用于其他高风险非剧本对话领域，如远程医疗、教育和政治论述，提供了一种稳健且可解释的多模态话语表示方法。该方法为财务预测和话语评估等下游任务提供了实际用途，同时也提供了一种适用于涉及高风险通信的其他领域的通用方法。 

---
# Multilevel Analysis of Cryptocurrency News using RAG Approach with Fine-Tuned Mistral Large Language Model 

**Title (ZH)**: 使用 fine-tuned Mistral 大型语言模型的 RAG 方法进行多层级加密货币新闻分析 

**Authors**: Bohdan M. Pavlyshenko  

**Link**: [PDF](https://arxiv.org/pdf/2509.03527)  

**Abstract**: In the paper, we consider multilevel multitask analysis of cryptocurrency news using a fine-tuned Mistral 7B large language model with retrieval-augmented generation (RAG).
On the first level of analytics, the fine-tuned model generates graph and text summaries with sentiment scores as well as JSON representations of summaries. Higher levels perform hierarchical stacking that consolidates sets of graph-based and text-based summaries as well as summaries of summaries into comprehensive reports. The combination of graph and text summaries provides complementary views of cryptocurrency news. The model is fine-tuned with 4-bit quantization using the PEFT/LoRA approach. The representation of cryptocurrency news as knowledge graph can essentially eliminate problems with large language model hallucinations.
The obtained results demonstrate that the use of fine-tuned Mistral 7B LLM models for multilevel cryptocurrency news analysis can conduct informative qualitative and quantitative analytics, providing important insights. 

**Abstract (ZH)**: 使用微调的Mistral 7B大规模语言模型与检索增强生成（RAG）进行加密货币新闻的多层面多任务分析 

---
# Speech-Based Cognitive Screening: A Systematic Evaluation of LLM Adaptation Strategies 

**Title (ZH)**: 基于语音的认知筛查：大规模语言模型适应策略的系统评估 

**Authors**: Fatemeh Taherinezhad, Mohamad Javad Momeni Nezhad, Sepehr Karimi, Sina Rashidi, Ali Zolnour, Maryam Dadkhah, Yasaman Haghbin, Hossein AzadMaleki, Maryam Zolnoori  

**Link**: [PDF](https://arxiv.org/pdf/2509.03525)  

**Abstract**: Over half of US adults with Alzheimer disease and related dementias remain undiagnosed, and speech-based screening offers a scalable detection approach. We compared large language model adaptation strategies for dementia detection using the DementiaBank speech corpus, evaluating nine text-only models and three multimodal audio-text models on recordings from DementiaBank speech corpus. Adaptations included in-context learning with different demonstration selection policies, reasoning-augmented prompting, parameter-efficient fine-tuning, and multimodal integration. Results showed that class-centroid demonstrations achieved the highest in-context learning performance, reasoning improved smaller models, and token-level fine-tuning generally produced the best scores. Adding a classification head substantially improved underperforming models. Among multimodal models, fine-tuned audio-text systems performed well but did not surpass the top text-only models. These findings highlight that model adaptation strategies, including demonstration selection, reasoning design, and tuning method, critically influence speech-based dementia detection, and that properly adapted open-weight models can match or exceed commercial systems. 

**Abstract (ZH)**: 超过一半的美国成人阿尔茨海默病及相关痴呆症患者未被诊断，基于言语的筛查提供了一种可扩展的检测方法。我们使用DementiaBank语音语料库，比较了不同大型语言模型适应策略的痴呆检测效果，评估了来自DementiaBank语音语料库的九种文本-only模型和三种多模态音频-文本模型。适应策略包括上下文学习、不同示例选择策略、增强推理提示、参数高效微调以及多模态集成。结果表明，类别中心点示例在上下文学习中表现最佳，增强推理改进了较小的模型，-token级微调通常产生了最佳分数。添加分类头显著提升了表现不佳的模型。在多模态模型中，微调的音频-文本系统表现良好，但未超过最佳文本-only模型。这些发现强调了模型适应策略，包括示例选择、推理设计和调整方法对基于言语的痴呆检测至关重要，并表明适当调整的开放权重模型可以匹配或超过商用系统。 

---
# BiND: A Neural Discriminator-Decoder for Accurate Bimanual Trajectory Prediction in Brain-Computer Interfaces 

**Title (ZH)**: BiND: 用于脑机接口的二元手轨迹预测神经鉴别-解码器 

**Authors**: Timothee Robert, MohammadAli Shaeri, Mahsa Shoaran  

**Link**: [PDF](https://arxiv.org/pdf/2509.03521)  

**Abstract**: Decoding bimanual hand movements from intracortical recordings remains a critical challenge for brain-computer interfaces (BCIs), due to overlapping neural representations and nonlinear interlimb interactions. We introduce BiND (Bimanual Neural Discriminator-Decoder), a two-stage model that first classifies motion type (unimanual left, unimanual right, or bimanual) and then uses specialized GRU-based decoders, augmented with a trial-relative time index, to predict continuous 2D hand velocities. We benchmark BiND against six state-of-the-art models (SVR, XGBoost, FNN, CNN, Transformer, GRU) on a publicly available 13-session intracortical dataset from a tetraplegic patient. BiND achieves a mean $R^2$ of 0.76 ($\pm$0.01) for unimanual and 0.69 ($\pm$0.03) for bimanual trajectory prediction, surpassing the next-best model (GRU) by 2% in both tasks. It also demonstrates greater robustness to session variability than all other benchmarked models, with accuracy improvements of up to 4% compared to GRU in cross-session analyses. This highlights the effectiveness of task-aware discrimination and temporal modeling in enhancing bimanual decoding. 

**Abstract (ZH)**: 从颅内记录解码双上肢手部运动仍然是脑机接口（BCIs）中的一个关键挑战，由于神经表示的重叠和非线性双侧交互。我们引入了BiND（双上肢神经鉴别解码器），这是一种两阶段模型，首先分类运动类型（单上肢左、单上肢右或双上肢），然后使用带有试验相对时间索引的专门GRU基解码器来预测连续的2D手速度。我们在一个公开的来自四肢瘫痪患者的13会话颅内数据集上将BiND与六种最先进的模型（SVR、XGBoost、FNN、CNN、Transformer、GRU）进行了基准测试。BiND在单上肢轨迹预测中的平均$R^2$值为0.76（$\pm$0.01），双上肢轨迹预测值为0.69（$\pm$0.03），两项任务上的表现均优于排名第二的模型（GRU），分别超出2%。此外，BiND在会话间分析中表现出对会话间变异的更强鲁棒性，在交叉会话分析中相对于GRU的准确率提升了最多4%。这突显了任务感知鉴别和时间建模在增强双上肢解码方面的有效性。 

---
