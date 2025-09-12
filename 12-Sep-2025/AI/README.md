# The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs 

**Title (ZH)**: 递减回报的错觉：测量LLM中的长期执行能力 

**Authors**: Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen Staab, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2509.09677)  

**Abstract**: Does continued scaling of large language models (LLMs) yield diminishing returns? Real-world value often stems from the length of task an agent can complete. We start this work by observing the simple but counterintuitive fact that marginal gains in single-step accuracy can compound into exponential improvements in the length of a task a model can successfully complete. Then, we argue that failures of LLMs when simple tasks are made longer arise from mistakes in execution, rather than an inability to reason. We propose isolating execution capability, by explicitly providing the knowledge and plan needed to solve a long-horizon task. We find that larger models can correctly execute significantly more turns even when small models have 100\% single-turn accuracy. We observe that the per-step accuracy of models degrades as the number of steps increases. This is not just due to long-context limitations -- curiously, we observe a self-conditioning effect -- models become more likely to make mistakes when the context contains their errors from prior turns. Self-conditioning does not reduce by just scaling the model size. In contrast, recent thinking models do not self-condition, and can also execute much longer tasks in a single turn. We conclude by benchmarking frontier thinking models on the length of task they can execute in a single turn. Overall, by focusing on the ability to execute, we hope to reconcile debates on how LLMs can solve complex reasoning problems yet fail at simple tasks when made longer, and highlight the massive benefits of scaling model size and sequential test-time compute for long-horizon tasks. 

**Abstract (ZH)**: 持续扩展大型语言模型（LLMs）会产生边际效益递减的现象吗？真实世界的价值往往源自模型能够完成的任务长度。我们从一个简单但直观相反的事实开始——单一步骤准确性的小幅提升可以转化为模型能够成功完成任务长度的指数级改进。然后，我们指出当简单任务变得更长时，LLMs 的失败并非因为推理能力不足，而是执行上的错误。我们提出通过明确提供解决长期任务所需的知识和计划来隔离执行能力。我们发现，即使小型模型在单步准确性上达到100%，大型模型也能正确执行显著更多的步骤。我们观察到，随着步骤数量的增加，模型的单步准确性会下降。这不仅是因为长上下文的限制——有趣的是，我们发现一种自归因效应——当上下文中包含前一轮的错误时，模型更可能犯错。仅通过扩展模型规模无法减少这种自归因效应。相比之下，最近的思想模型不表现出自归因，并且可以在单步中执行更长的任务。最后，我们通过基准测试前沿思想模型，评估它们在单步中能够执行的任务长度。总体而言，通过关注执行能力，我们希望解决关于LLMs如何解决复杂推理问题但在任务变得更长时却在简单任务上失败的辩论，并强调为长期任务扩展模型规模和序贯测试时计算的巨大好处。 

---
# Boosting Embodied AI Agents through Perception-Generation Disaggregation and Asynchronous Pipeline Execution 

**Title (ZH)**: 通过感知-生成分歧和异步流水线执行提升具身AI代理 

**Authors**: Shulai Zhang, Ao Xu, Quan Chen, Han Zhao, Weihao Cui, Ningxin Zheng, Haibin Lin, Xin Liu, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.09560)  

**Abstract**: Embodied AI systems operate in dynamic environments, requiring seamless integration of perception and generation modules to process high-frequency input and output demands. Traditional sequential computation patterns, while effective in ensuring accuracy, face significant limitations in achieving the necessary "thinking" frequency for real-world applications. In this work, we present Auras, an algorithm-system co-designed inference framework to optimize the inference frequency of embodied AI agents. Auras disaggregates the perception and generation and provides controlled pipeline parallelism for them to achieve high and stable throughput. Faced with the data staleness problem that appears when the parallelism is increased, Auras establishes a public context for perception and generation to share, thereby promising the accuracy of embodied agents. Experimental results show that Auras improves throughput by 2.54x on average while achieving 102.7% of the original accuracy, demonstrating its efficacy in overcoming the constraints of sequential computation and providing high throughput. 

**Abstract (ZH)**: 嵌入式AI系统在动态环境中运行，需要无缝集成感知和生成模块以处理高频率的输入和输出需求。传统的时间序列计算模式虽然在确保准确性方面有效，但在实现现实世界应用所需要的“思考”频率方面存在显著限制。本文提出Auras，一种算法-系统协同设计的推理框架，以优化嵌入式AI代理的推理频率。Auras拆分感知和生成模块，并提供受控的流水线并行性，以实现高且稳定的吞吐量。面对并行度增加时出现的数据陈旧问题，Auras建立了一个公共上下文，使感知和生成模块能够共享，从而保证嵌入式代理的准确性。实验结果表明，Auras在吞吐量平均提高2.54倍的同时，实现了原始准确度的102.7%，证明了其有效克服时间序列计算约束并提供高吞吐量的能力。 

---
# Compositional Concept Generalization with Variational Quantum Circuits 

**Title (ZH)**: 基于变分量子电路的合成概念泛化 

**Authors**: Hala Hawashin, Mina Abbaszadeh, Nicholas Joseph, Beth Pearson, Martha Lewis, Mehrnoosh sadrzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09541)  

**Abstract**: Compositional generalization is a key facet of human cognition, but lacking in current AI tools such as vision-language models. Previous work examined whether a compositional tensor-based sentence semantics can overcome the challenge, but led to negative results. We conjecture that the increased training efficiency of quantum models will improve performance in these tasks. We interpret the representations of compositional tensor-based models in Hilbert spaces and train Variational Quantum Circuits to learn these representations on an image captioning task requiring compositional generalization. We used two image encoding techniques: a multi-hot encoding (MHE) on binary image vectors and an angle/amplitude encoding on image vectors taken from the vision-language model CLIP. We achieve good proof-of-concept results using noisy MHE encodings. Performance on CLIP image vectors was more mixed, but still outperformed classical compositional models. 

**Abstract (ZH)**: 组分泛化是人类认知的一个关键方面，但目前的AI工具如视觉-语言模型中缺乏这一功能。我们推测，量子模型训练效率的提升将在这些任务中改善性能。我们将组分张量基句子语义的表示解释为希尔伯特空间中的表示，并训练变量子电路在需要组分泛化的图像描述任务中学习这些表示。我们使用了两种图像编码技术：二元图像向量的多热编码（MHE）和图像向量（来自视觉-语言模型CLIP）的角度/振幅编码。使用嘈杂的MHE编码取得了良好的概念验证结果。在CLIP图像向量上的性能更为参差不齐，但仍优于经典的组分模型。 

---
# SEDM: Scalable Self-Evolving Distributed Memory for Agents 

**Title (ZH)**: SEDM：可扩展自进化的分布式内存架构 

**Authors**: Haoran Xu, Jiacong Hu, Ke Zhang, Lei Yu, Yuxin Tang, Xinyuan Song, Yiqun Duan, Lynn Ai, Bill Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09498)  

**Abstract**: Long-term multi-agent systems inevitably generate vast amounts of trajectories and historical interactions, which makes efficient memory management essential for both performance and scalability. Existing methods typically depend on vector retrieval and hierarchical storage, yet they are prone to noise accumulation, uncontrolled memory expansion, and limited generalization across domains. To address these challenges, we present SEDM, Self-Evolving Distributed Memory, a verifiable and adaptive framework that transforms memory from a passive repository into an active, self-optimizing component. SEDM integrates verifiable write admission based on reproducible replay, a self-scheduling memory controller that dynamically ranks and consolidates entries according to empirical utility, and cross-domain knowledge diffusion that abstracts reusable insights to support transfer across heterogeneous tasks. Evaluations on benchmark datasets demonstrate that SEDM improves reasoning accuracy while reducing token overhead compared with strong memory baselines, and further enables knowledge distilled from fact verification to enhance multi-hop reasoning. The results highlight SEDM as a scalable and sustainable memory mechanism for open-ended multi-agent collaboration. The code will be released in the later stage of this project. 

**Abstract (ZH)**: 自适应演化分布式记忆：可验证的长期多智能体系统高效内存管理框架 

---
# Inteligencia Artificial jurídica y el desafío de la veracidad: análisis de alucinaciones, optimización de RAG y principios para una integración responsable 

**Title (ZH)**: 人工智能法律与真实性挑战：幻觉分析、RAG优化及负责任集成的原则 

**Authors**: Alex Dantart  

**Link**: [PDF](https://arxiv.org/pdf/2509.09467)  

**Abstract**: This technical report analyzes the challenge of "hallucinations" (false information) in LLMs applied to law. It examines their causes, manifestations, and the effectiveness of the RAG mitigation strategy, highlighting its limitations and proposing holistic optimizations. The paper explores the ethical and regulatory implications, emphasizing human oversight as an irreplaceable role. It concludes that the solution lies not in incrementally improving generative models, but in adopting a "consultative" AI paradigm that prioritizes veracity and traceability, acting as a tool to amplify, not replace, professional judgment.
--
Este informe técnico analiza el desafío de las "alucinaciones" (información falsa) en los LLMs aplicados al derecho. Se examinan sus causas, manifestaciones y la efectividad de la estrategia de mitigación RAG, exponiendo sus limitaciones y proponiendo optimizaciones holísticas. Se exploran las implicaciones éticas y regulatorias, enfatizando la supervisión humana como un rol insustituible. El documento concluye que la solución no reside en mejorar incrementalmente los modelos generativos, sino en adoptar un paradigma de IA "consultiva" que priorice la veracidad y la trazabilidad, actuando como una herramienta para amplificar, y no sustituir, el juicio profesional. 

**Abstract (ZH)**: 技术报告分析了法律应用领域大语言模型中的“幻觉”（虚假信息）挑战。该报告探讨了其原因、表现形式以及RAG缓解策略的有效性，指出其局限性并提出整体优化建议。文章探讨了伦理和监管影响，强调人类监督不可或缺的作用。报告结论认为，解决方案不在于逐步改进生成模型，而在于采用以验证性和可追溯性为首要原则的“咨询式”人工智能范式，作为工具来增强而非取代专业判断。 

---
# TORSO: Template-Oriented Reasoning Towards General Tasks 

**Title (ZH)**: 模板导向的推理 toward 通用任务 

**Authors**: Minhyuk Kim, Seungyoon Lee, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09448)  

**Abstract**: The approaches that guide Large Language Models (LLMs) to emulate human reasoning during response generation have emerged as an effective method for enabling them to solve complex problems in a step-by-step manner, thereby achieving superior performance. However, most existing approaches using few-shot prompts to generate responses heavily depend on the provided examples, limiting the utilization of the model's inherent reasoning capabilities. Moreover, constructing task-specific few-shot prompts is often costly and may lead to inconsistencies across different tasks. In this work, we introduce Template-Oriented Reasoning (TORSO), which elicits the model to utilize internal reasoning abilities to generate proper responses across various tasks without the need for manually crafted few-shot examples. Our experimental results demonstrate that TORSO achieves strong performance on diverse LLMs benchmarks with reasonable rationales. 

**Abstract (ZH)**: 指导大型语言模型（LLMs）在响应生成过程中模拟人类推理的方法已被证明是使它们能够逐步解决复杂问题、从而实现卓越性能的有效方法。然而，现有的大多数使用少量示例生成响应的方法严重依赖于提供的示例，限制了模型内在推理能力的充分利用。此外，构建特定任务的少量示例提示通常成本较高，并可能导致不同任务间的一致性问题。在本工作中，我们提出了模板导向推理（TORSO），这是一种使模型能够利用其内部推理能力生成适用于多种任务的适当响应的方法，而无需手动构建的少量示例。我们的实验结果表明，TORSO在多种LLM基准测试中表现出强劲性能，并具有合理的推理依据。 

---
# Curriculum-Based Multi-Tier Semantic Exploration via Deep Reinforcement Learning 

**Title (ZH)**: 基于课程的多层语义探索深度强化学习方法 

**Authors**: Abdel Hakim Drid, Vincenzo Suriani, Daniele Nardi, Abderrezzak Debilou  

**Link**: [PDF](https://arxiv.org/pdf/2509.09356)  

**Abstract**: Navigating and understanding complex and unknown environments autonomously demands more than just basic perception and movement from embodied agents. Truly effective exploration requires agents to possess higher-level cognitive abilities, the ability to reason about their surroundings, and make more informed decisions regarding exploration strategies. However, traditional RL approaches struggle to balance efficient exploration and semantic understanding due to limited cognitive capabilities embedded in the small policies for the agents, leading often to human drivers when dealing with semantic exploration. In this paper, we address this challenge by presenting a novel Deep Reinforcement Learning (DRL) architecture that is specifically designed for resource efficient semantic exploration. A key methodological contribution is the integration of a Vision-Language Model (VLM) common-sense through a layered reward function. The VLM query is modeled as a dedicated action, allowing the agent to strategically query the VLM only when deemed necessary for gaining external guidance, thereby conserving resources. This mechanism is combined with a curriculum learning strategy designed to guide learning at different levels of complexity to ensure robust and stable learning. Our experimental evaluation results convincingly demonstrate that our agent achieves significantly enhanced object discovery rates and develops a learned capability to effectively navigate towards semantically rich regions. Furthermore, it also shows a strategic mastery of when to prompt for external environmental information. By demonstrating a practical and scalable method for embedding common-sense semantic reasoning with autonomous agents, this research provides a novel approach to pursuing a fully intelligent and self-guided exploration in robotics. 

**Abstract (ZH)**: 自主导航和理解复杂未知环境需要不仅具备基本的感知和运动能力，还要求代理拥有更高层次的认知能力，能够对其周围环境进行推理，并做出更为明智的探索策略选择。传统的强化学习方法由于代理内嵌的认知能力有限，难以平衡高效的探索和语义理解，往往需要人类驾驶员处理语义探索任务。本文通过提出一种专为资源高效语义探索设计的新型深度强化学习（DRL）架构，应对这一挑战。一个关键的方法论贡献是通过层次化的奖励函数集成一种视觉语言模型（VLM）的常识。VLM 查询被建模为专门的动作，使代理能够在必要时战略性地查询VLM 以获取外部指导，从而节省资源。该机制结合了一种分级学习策略，以在不同复杂度的层面上引导学习，确保学习的稳健性和稳定性。实验评估结果表明，我们的代理显著提高了物体发现率，并发展了有效导航至语义丰富区域的能力。此外，还展示了何时求助外部环境信息的策略性掌握。通过展示如何将常识性语义推理嵌入自主代理的实用且可扩展的方法，本文为实现机器人中的全面智能和自我引导探索提供了一种新方法。 

---
# Towards Adaptive ML Benchmarks: Web-Agent-Driven Construction, Domain Expansion, and Metric Optimization 

**Title (ZH)**: 面向自适应机器学习基准：基于Web-Agent的构建、领域扩展与度量优化 

**Authors**: Hangyi Jia, Yuxi Qian, Hanwen Tong, Xinhui Wu, Lin Chen, Feng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.09321)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the emergence of general-purpose agents for automating end-to-end machine learning (ML) workflows, including data analysis, feature engineering, model training, and competition solving. However, existing benchmarks remain limited in task coverage, domain diversity, difficulty modeling, and evaluation rigor, failing to capture the full capabilities of such agents in realistic settings. We present TAM Bench, a diverse, realistic, and structured benchmark for evaluating LLM-based agents on end-to-end ML tasks. TAM Bench features three key innovations: (1) A browser automation and LLM-based task acquisition system that automatically collects and structures ML challenges from platforms such as Kaggle, AIcrowd, and Biendata, spanning multiple task types and data modalities (e.g., tabular, text, image, graph, audio); (2) A leaderboard-driven difficulty modeling mechanism that estimates task complexity using participant counts and score dispersion, enabling scalable and objective task calibration; (3) A multi-dimensional evaluation framework incorporating performance, format compliance, constraint adherence, and task generalization. Based on 150 curated AutoML tasks, we construct three benchmark subsets of different sizes -- Lite, Medium, and Full -- designed for varying evaluation scenarios. The Lite version, with 18 tasks and balanced coverage across modalities and difficulty levels, serves as a practical testbed for daily benchmarking and comparative studies. 

**Abstract (ZH)**: 近期大型语言模型的进展使通用型代理能够自动化端到端的机器学习工作流，包括数据分析、特征工程、模型训练和竞赛解决。然而，现有的基准测试在任务覆盖范围、领域多样性、难度建模和评估严谨性方面仍有限制，未能捕捉到这些代理在实际环境中的全部能力。我们提出了TAM Bench，这是一个多样化的、现实的和结构化的基准测试，用于评估基于大型语言模型的代理在端到端机器学习任务中的性能。TAM Bench 的三大创新包括：（1）一个基于浏览器自动化和大型语言模型的任务获取系统，自动从Kaggle、Aicrowd和Biendata等平台收集和结构化跨多种任务类型和数据模态（例如，表格、文本、图像、图、音频）的机器学习挑战；（2）一个基于排行榜的难度建模机制，利用参与者数量和分数分布估计任务复杂度，实现可扩展和客观的任务标度；（3）一个多维评估框架，包括性能、格式合规性、约束遵守性和任务泛化。基于150个精选的自动化机器学习任务，我们构建了三个不同规模的基准子集——轻量级、中等和完整——适用于不同的评估场景。轻量级版本，包含18个任务，涵盖多种模态和难度级别，适合作为日常基准测试和对比研究的实际试验场。 

---
# Measuring Implicit Spatial Coordination in Teams: Effects on Collective Intelligence and Performance 

**Title (ZH)**: 测量团队中的隐性空间协调：对其集体智能和表现的影响 

**Authors**: Thuy Ngoc Nguyen, Anita Williams Woolley, Cleotilde Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2509.09314)  

**Abstract**: Coordinated teamwork is essential in fast-paced decision-making environments that require dynamic adaptation, often without an opportunity for explicit communication. Although implicit coordination has been extensively considered in the existing literature, the majority of work has focused on co-located, synchronous teamwork (such as sports teams) or, in distributed teams, primarily on coordination of knowledge work. However, many teams (firefighters, military, law enforcement, emergency response) must coordinate their movements in physical space without the benefit of visual cues or extensive explicit communication. This paper investigates how three dimensions of spatial coordination, namely exploration diversity, movement specialization, and adaptive spatial proximity, influence team performance in a collaborative online search and rescue task where explicit communication is restricted and team members rely on movement patterns to infer others' intentions and coordinate actions. Our metrics capture the relational aspects of teamwork by measuring spatial proximity, distribution patterns, and alignment of movements within shared environments. We analyze data from 34 four-person teams (136 participants) assigned to specialized roles in a search and rescue task. Results show that spatial specialization positively predicts performance, while adaptive spatial proximity exhibits a marginal inverted U-shaped relationship, suggesting moderate levels of adaptation are optimal. Furthermore, the temporal dynamics of these metrics differentiate high- from low-performing teams over time. These findings provide insights into implicit spatial coordination in role-based teamwork and highlight the importance of balanced adaptive strategies, with implications for training and AI-assisted team support systems. 

**Abstract (ZH)**: 快速决策环境中，在动态适应需求下协调团队合作至关重要，往往缺乏明确的沟通机会。尽管现有文献中已经广泛考虑了隐性协调，但大多数工作主要聚焦于共处且同步工作的团队（如运动队）或分布式团队中的知识工作协调。然而，许多团队（如消防员、军事人员、执法机构、紧急救援人员）必须在缺乏视觉提示和大量明确沟通的情况下，在物理空间中协调行动。本文探讨了探索多样性、运动专业化和适应性空间接近这三种空间协调维度如何影响受限明确沟通条件下的协作在线搜索与营救任务中的团队表现。我们通过测量共享环境中空间接近性、分布模式和运动方向的一致性来捕捉团队合作中的关系层面。我们分析了34个四人团队（136名参与者）在搜索与救援任务中分配专门角色的数据。结果显示，空间专业化正向预测团队表现，而适应性空间接近呈现轻微的倒U型关系，表明中等程度的适应性最优化。此外，这些指标的时间动态在时间上区分了高绩效和低绩效团队。这些发现揭示了基于角色团队中隐性空间协调的洞见，并强调了平衡适应策略的重要性，这对训练和AI辅助团队支持系统有启示意义。 

---
# Explaining Tournament Solutions with Minimal Supports 

**Title (ZH)**: 用最小支持集解释锦标赛解 

**Authors**: Clément Contet, Umberto Grandi, Jérôme Mengin  

**Link**: [PDF](https://arxiv.org/pdf/2509.09312)  

**Abstract**: Tournaments are widely used models to represent pairwise dominance between candidates, alternatives, or teams. We study the problem of providing certified explanations for why a candidate appears among the winners under various tournament rules. To this end, we identify minimal supports, minimal sub-tournaments in which the candidate is guaranteed to win regardless of how the rest of the tournament is completed (that is, the candidate is a necessary winner of the sub-tournament). This notion corresponds to an abductive explanation for the question,"Why does the winner win the tournament", a central concept in formal explainable AI. We focus on common tournament solutions: the top cycle, the uncovered set, the Copeland rule, the Borda rule, the maximin rule, and the weighted uncovered set. For each rule we determine the size of the smallest minimal supports, and we present polynomial-time algorithms to compute them for all but the weighted uncovered set, for which the problem is NP-complete. Finally, we show how minimal supports can serve to produce compact, certified, and intuitive explanations. 

**Abstract (ZH)**: 竞赛广泛用于表示候选人、替代品或团队之间的两两支配关系。我们研究在各种竞赛规则下提供认证解释的问题，解释为什么某个候选人会出现在获胜者之中。为此，我们识别最小支撑，即在其中候选人可以确保获胜的最小子竞赛（即，候选人是子竞赛的必要获胜者），而不考虑其余竞赛如何完成。这一概念对应于关于“为什么获胜者赢得竞赛”的归因解释，是形式可解释人工智能中的一个核心概念。我们关注常用的竞赛解决方案：顶周期、未覆盖集、Copeland法则、Borda法则、最大化最小值法则和加权未覆盖集。对于每种法则，我们确定最小支撑的大小，并提供了一类法则计算最小支撑的多项式时间算法，对于加权未覆盖集，该问题为NP完全问题。最后，我们展示了最小支撑如何用于生成紧凑、认证且直观的解释。 

---
# LightAgent: Production-level Open-source Agentic AI Framework 

**Title (ZH)**: LightAgent: 生产级开源代理人型AI框架 

**Authors**: Weige Cai, Tong Zhu, Jinyi Niu, Ruiqi Hu, Lingyao Li, Tenglong Wang, Xiaowu Dai, Weining Shen, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09292)  

**Abstract**: With the rapid advancement of large language models (LLMs), Multi-agent Systems (MAS) have achieved significant progress in various application scenarios. However, substantial challenges remain in designing versatile, robust, and efficient platforms for agent deployment. To address these limitations, we propose \textbf{LightAgent}, a lightweight yet powerful agentic framework, effectively resolving the trade-off between flexibility and simplicity found in existing frameworks. LightAgent integrates core functionalities such as Memory (mem0), Tools, and Tree of Thought (ToT), while maintaining an extremely lightweight structure. As a fully open-source solution, it seamlessly integrates with mainstream chat platforms, enabling developers to easily build self-learning agents. We have released LightAgent at \href{this https URL}{this https URL} 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的快速发展，多智能体系统（MAS）在各种应用场景中取得了显著进展。然而，在设计 versatile、robust 和 efficient 的智能体部署平台方面仍面临重大挑战。为解决这些限制，我们提出了 LightAgent，一个轻量级但强大的智能体框架，有效解决了现有框架中存在的灵活性与简单性之间的trade-off。LightAgent 结合了核心功能，如 Memory（mem0）、Tools 和 Tree of Thought（ToT），同时保持了极其轻量级的结构。作为完全开源的解决方案，它无缝集成到主流聊天平台，使开发者能够轻松构建自学习智能体。我们已在 \href{this https URL}{this https URL} 发布了 LightAgent。 

---
# Tree-OPO: Off-policy Monte Carlo Tree-Guided Advantage Optimization for Multistep Reasoning 

**Title (ZH)**: 树-OPO：基于蒙特卡洛树引导的优势多步优化算法 

**Authors**: Bingning Huang, Tu Nguyen, Matthieu Zimmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.09284)  

**Abstract**: Recent advances in reasoning with large language models (LLMs) have shown the effectiveness of Monte Carlo Tree Search (MCTS) for generating high-quality intermediate trajectories, particularly in math and symbolic domains. Inspired by this, we explore how MCTS-derived trajectories, traditionally used for training value or reward models, can be repurposed to improve policy optimization in preference-based reinforcement learning (RL). Specifically, we focus on Group Relative Policy Optimization (GRPO), a recent algorithm that enables preference-consistent policy learning without value networks. We propose a staged GRPO training paradigm where completions are derived from partially revealed MCTS rollouts, introducing a novel tree-structured setting for advantage estimation. This leads to a rich class of prefix-conditioned reward signals, which we analyze theoretically and empirically. Our initial results indicate that while structured advantage estimation can stabilize updates and better reflect compositional reasoning quality, challenges such as advantage saturation and reward signal collapse remain. We propose heuristic and statistical solutions to mitigate these issues and discuss open challenges for learning under staged or tree-like reward structures. 

**Abstract (ZH)**: Recent advances in reasoning with large language models (LLMs) have shown the effectiveness of Monte Carlo Tree Search (MCTS) for generating high-quality intermediate trajectories, particularly in math and symbolic domains. Inspired by this, we explore how MCTS-derived trajectories, traditionally used for training value or reward models, can be repurposed to improve policy optimization in preference-based reinforcement learning (RL). Specifically, we focus on Group Relative Policy Optimization (GRPO), a recent algorithm that enables preference-consistent policy learning without value networks. We propose a staged GRPO training paradigm where completions are derived from partially revealed MCTS rollouts, introducing a novel tree-structured setting for advantage estimation. This leads to a rich class of prefix-conditioned reward signals, which we analyze theoretically and empirically. Our initial results indicate that while structured advantage estimation can stabilize updates and better reflect compositional reasoning quality, challenges such as advantage saturation and reward signal collapse remain. We propose heuristic and statistical solutions to mitigate these issues and discuss open challenges for learning under staged or tree-like reward structures. 

---
# Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs 

**Title (ZH)**: 知识与语言融合：基于知识图谱的问答与大语言模型的 comparative study 

**Authors**: Vaibhav Chaudhary, Neha Soni, Narotam Singh, Amita Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2509.09272)  

**Abstract**: Knowledge graphs, a powerful tool for structuring information through relational triplets, have recently become the new front-runner in enhancing question-answering systems. While traditional Retrieval Augmented Generation (RAG) approaches are proficient in fact-based and local context-based extraction from concise texts, they encounter limitations when addressing the thematic and holistic understanding of complex, extensive texts, requiring a deeper analysis of both text and context. This paper presents a comprehensive technical comparative study of three different methodologies for constructing knowledge graph triplets and integrating them with Large Language Models (LLMs) for question answering: spaCy, Stanford CoreNLP-OpenIE, and GraphRAG, all leveraging open source technologies. We evaluate the effectiveness, feasibility, and adaptability of these methods by analyzing their capabilities, state of development, and their impact on the performance of LLM-based question answering. Experimental results indicate that while OpenIE provides the most comprehensive coverage of triplets, GraphRAG demonstrates superior reasoning abilities among the three. We conclude with a discussion on the strengths and limitations of each method and provide insights into future directions for improving knowledge graph-based question answering. 

**Abstract (ZH)**: 知识图谱，作为一种通过关系三元组结构化信息的有力工具，近年来已成为提升问答系统的新前沿。虽然传统的检索增强生成（RAG）方法在从小型文本中提取事实和局部上下文方面表现出色，但在处理复杂、 extensive 文本的主题性和整体理解方面存在局限性，需要对文本和上下文进行更深入的分析。本文对三种不同的知识图谱三元组构建方法及其与大型语言模型（LLMs）结合用于问答的技术进行了全面的技术对比研究：spaCy、Stanford CoreNLP-OpenIE 和 GraphRAG，所有方法均利用开源技术。我们通过分析这些方法的能力、发展状态及其对基于LLM的问答性能的影响来评估它们的有效性、可行性和适应性。实验结果表明，尽管OpenIE提供了最全面的三元组覆盖范围，但GraphRAG在三种方法中表现出了更强的推理能力。我们在此基础上讨论了每种方法的优缺点，并提供了关于改进基于知识图谱的问答的未来方向的见解。 

---
# Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search 

**Title (ZH)**: 木星：通过笔记本和推理时值导向搜索增强大语言模型数据分析能力 

**Authors**: Shuocheng Li, Yihao Liu, Silin Du, Wenxuan Zeng, Zhe Xu, Mengyu Zhou, Yeye He, Haoyu Dong, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09245)  

**Abstract**: Large language models (LLMs) have shown great promise in automating data science workflows, but existing models still struggle with multi-step reasoning and tool use, which limits their effectiveness on complex data analysis tasks. To address this, we propose a scalable pipeline that extracts high-quality, tool-based data analysis tasks and their executable multi-step solutions from real-world Jupyter notebooks and associated data files. Using this pipeline, we introduce NbQA, a large-scale dataset of standardized task-solution pairs that reflect authentic tool-use patterns in practical data science scenarios. To further enhance multi-step reasoning, we present Jupiter, a framework that formulates data analysis as a search problem and applies Monte Carlo Tree Search (MCTS) to generate diverse solution trajectories for value model learning. During inference, Jupiter combines the value model and node visit counts to efficiently collect executable multi-step plans with minimal search steps. Experimental results show that Qwen2.5-7B and 14B-Instruct models on NbQA solve 77.82% and 86.38% of tasks on InfiAgent-DABench, respectively-matching or surpassing GPT-4o and advanced agent frameworks. Further evaluations demonstrate improved generalization and stronger tool-use reasoning across diverse multi-step reasoning tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自动化数据科学工作流方面表现出巨大的潜力，但现有模型仍然在多步推理和工具使用方面存在局限性，这限制了它们在复杂数据分析任务中的有效性。为此，我们提出了一种可扩展的流水线，从真实的Jupyter笔记本及其相关数据文件中提取高质量的工具基数据分析任务及其可执行的多步解决方案。通过该流水线，我们介绍了NbQA数据集，这是一个大规模的任务-解决方案对集合，反映了实际数据科学场景中的真实工具使用模式。为增强多步推理，我们提出了Jupiter框架，将数据分析问题形式化为搜索问题，并采用蒙特卡洛树搜索（MCTS）生成多样性的解决方案轨迹以用于价值模型学习。在推理过程中，Jupiter结合价值模型和节点访问计数，高效地收集可执行的多步计划，同时减少搜索步骤。实验结果表明，Qwen2.5-7B和14B-Instruct模型在InfiAgent-DABench上分别解决了77.82%和86.38%的任务，匹配甚至超过了GPT-4o和先进的代理框架。进一步的评估表明，Jupiter在多步推理任务中展示了更强的泛化能力和工具使用推理能力。 

---
# Enabling Regulatory Multi-Agent Collaboration: Architecture, Challenges, and Solutions 

**Title (ZH)**: 实现监管多代理协作：架构、挑战与解决方案 

**Authors**: Qinnan Hu, Yuntao Wang, Yuan Gao, Zhou Su, Linkang Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.09215)  

**Abstract**: Large language models (LLMs)-empowered autonomous agents are transforming both digital and physical environments by enabling adaptive, multi-agent collaboration. While these agents offer significant opportunities across domains such as finance, healthcare, and smart manufacturing, their unpredictable behaviors and heterogeneous capabilities pose substantial governance and accountability challenges. In this paper, we propose a blockchain-enabled layered architecture for regulatory agent collaboration, comprising an agent layer, a blockchain data layer, and a regulatory application layer. Within this framework, we design three key modules: (i) an agent behavior tracing and arbitration module for automated accountability, (ii) a dynamic reputation evaluation module for trust assessment in collaborative scenarios, and (iii) a malicious behavior forecasting module for early detection of adversarial activities. Our approach establishes a systematic foundation for trustworthy, resilient, and scalable regulatory mechanisms in large-scale agent ecosystems. Finally, we discuss the future research directions for blockchain-enabled regulatory frameworks in multi-agent systems. 

**Abstract (ZH)**: 大型语言模型赋能的自主代理正通过实现适应性的多代理协作， transformative地改变数字和物理环境。尽管这些代理在金融、医疗和智能制造等领域提供了重要的机遇，但它们的不可预测行为和异质能力带来了重大的治理和问责挑战。本文提出了一种基于区块链的分层架构，用于监管代理协作，包括代理层、区块链数据层和监管应用层。在此框架内，我们设计了三个关键模块：（i）代理行为追踪和仲裁模块，以实现自动问责；（ii）动态声誉评估模块，以在协作场景中评估信任度；（iii）恶意行为预测模块，以早期检测敌对活动。本文方法为大规模代理生态系统中的可信赖、抗扰动和可扩展的监管机制奠定了系统性的基础。最后，我们讨论了面向多代理系统的基于区块链的监管框架的未来研究方向。 

---
# ProgD: Progressive Multi-scale Decoding with Dynamic Graphs for Joint Multi-agent Motion Forecasting 

**Title (ZH)**: ProgD：基于动态图的分阶段多尺度解码联合多智能体运动预测 

**Authors**: Xing Gao, Zherui Huang, Weiyao Lin, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.09210)  

**Abstract**: Accurate motion prediction of surrounding agents is crucial for the safe planning of autonomous vehicles. Recent advancements have extended prediction techniques from individual agents to joint predictions of multiple interacting agents, with various strategies to address complex interactions within future motions of agents. However, these methods overlook the evolving nature of these interactions. To address this limitation, we propose a novel progressive multi-scale decoding strategy, termed ProgD, with the help of dynamic heterogeneous graph-based scenario modeling. In particular, to explicitly and comprehensively capture the evolving social interactions in future scenarios, given their inherent uncertainty, we design a progressive modeling of scenarios with dynamic heterogeneous graphs. With the unfolding of such dynamic heterogeneous graphs, a factorized architecture is designed to process the spatio-temporal dependencies within future scenarios and progressively eliminate uncertainty in future motions of multiple agents. Furthermore, a multi-scale decoding procedure is incorporated to improve on the future scenario modeling and consistent prediction of agents' future motion. The proposed ProgD achieves state-of-the-art performance on the INTERACTION multi-agent prediction benchmark, ranking $1^{st}$, and the Argoverse 2 multi-world forecasting benchmark. 

**Abstract (ZH)**: 准确预测周围代理的运动对于自主车辆的安全规划至关重要。为了解决复杂交互带来的挑战，近年来的研究将预测技术从单个代理扩展到多个互动代理的联合预测，并提出了各种策略来处理未来代理运动中的交互。然而，这些方法忽视了这些交互的动态性。为了解决这一局限，我们提出了一种新颖的渐进多尺度解码策略，称为ProgD，并借助动态异构图基场景建模。特别是，为了明确且全面地捕捉未来场景中的动态社会交互（鉴于其本质上的不确定性），我们设计了一种基于动态异构图的渐进场景建模方法。随着动态异构图的展开，我们设计了一种因式化解构架构来处理未来场景中的时空依赖性，并逐步消除多个代理未来运动中的不确定性。此外，我们引入了一种多尺度解码过程来提高未来场景建模的一致性和代理未来运动的预测准确性。提出的ProgD在INTERACTION多代理预测基准和Argoverse 2多世界预测基准上取得了最先进的性能，分别排名第一。 

---
# Mind Meets Space: Rethinking Agentic Spatial Intelligence from a Neuroscience-inspired Perspective 

**Title (ZH)**: 心灵与空间的交汇：从神经科学启发视角重新思考行动者空间智能 

**Authors**: Bui Duc Manh, Soumyaratna Debnath, Zetong Zhang, Shriram Damodaran, Arvind Kumar, Yueyi Zhang, Lu Mi, Erik Cambria, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09154)  

**Abstract**: Recent advances in agentic AI have led to systems capable of autonomous task execution and language-based reasoning, yet their spatial reasoning abilities remain limited and underexplored, largely constrained to symbolic and sequential processing. In contrast, human spatial intelligence, rooted in integrated multisensory perception, spatial memory, and cognitive maps, enables flexible, context-aware decision-making in unstructured environments. Therefore, bridging this gap is critical for advancing Agentic Spatial Intelligence toward better interaction with the physical 3D world. To this end, we first start from scrutinizing the spatial neural models as studied in computational neuroscience, and accordingly introduce a novel computational framework grounded in neuroscience principles. This framework maps core biological functions to six essential computation modules: bio-inspired multimodal sensing, multi-sensory integration, egocentric-allocentric conversion, an artificial cognitive map, spatial memory, and spatial reasoning. Together, these modules form a perspective landscape for agentic spatial reasoning capability across both virtual and physical environments. On top, we conduct a framework-guided analysis of recent methods, evaluating their relevance to each module and identifying critical gaps that hinder the development of more neuroscience-grounded spatial reasoning modules. We further examine emerging benchmarks and datasets and explore potential application domains ranging from virtual to embodied systems, such as robotics. Finally, we outline potential research directions, emphasizing the promising roadmap that can generalize spatial reasoning across dynamic or unstructured environments. We hope this work will benefit the research community with a neuroscience-grounded perspective and a structured pathway. Our project page can be found at Github. 

**Abstract (ZH)**: 近期代理型AI的进步使得系统具备了自主任务执行和基于语言的推理能力，但其空间推理能力仍有限且未得到充分探索，主要受限于符号和序列处理。相比之下，人类的空间智能基于多感官整合、空间记忆和认知地图等，能够在非结构化环境中实现灵活、情境相关决策。因此，弥合这一差距对于推动代理型空间智能更好地与物理三维世界交互至关重要。为此，我们首先从计算神经科学研究的空间神经网络入手，引入一个基于神经科学原理的新型计算框架。该框架将核心生物学功能映射到六个关键计算模块：生物启发式多模态感知、多感官整合、以己为中心到以物为中心的转换、人工认知地图、空间记忆和空间推理。这些模块共同形成了一种视角景观，涵盖了虚拟和物理环境中的代理型空间推理能力。同时，我们根据框架对近期方法进行了分析，评估其与每个模块的相关性，并识别阻碍基于神经科学的空间推理模块发展的关键差距。我们进一步探讨了新兴基准和数据集，并探索了从虚拟到实体系统的潜在应用领域，如机器人技术。最后，我们提出了潜在的研究方向，强调了一个有希望的发展路线图，能够使空间推理适用于动态或非结构化环境。我们希望这项工作能够为研究社区提供一个基于神经科学的视角和一个结构化的途径。版权所有页可在Github上找到。 

---
# Anti-Money Laundering Machine Learning Pipelines; A Technical Analysis on Identifying High-risk Bank Clients with Supervised Learning 

**Title (ZH)**: 反洗钱机器学习管道：一种基于监督学习识别高风险银行客户的技術分析 

**Authors**: Khashayar Namdar, Pin-Chien Wang, Tushar Raju, Steven Zheng, Fiona Li, Safwat Tahmin Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09127)  

**Abstract**: Anti-money laundering (AML) actions and measurements are among the priorities of financial institutions, for which machine learning (ML) has shown to have a high potential. In this paper, we propose a comprehensive and systematic approach for developing ML pipelines to identify high-risk bank clients in a dataset curated for Task 1 of the University of Toronto 2023-2024 Institute for Management and Innovation (IMI) Big Data and Artificial Intelligence Competition. The dataset included 195,789 customer IDs, and we employed a 16-step design and statistical analysis to ensure the final pipeline was robust. We also framed the data in a SQLite database, developed SQL-based feature engineering algorithms, connected our pre-trained model to the database, and made it inference-ready, and provided explainable artificial intelligence (XAI) modules to derive feature importance. Our pipeline achieved a mean area under the receiver operating characteristic curve (AUROC) of 0.961 with a standard deviation (SD) of 0.005. The proposed pipeline achieved second place in the competition. 

**Abstract (ZH)**: 面向金融风险管理的机器学习管道构建：以多伦多大学2023-2024年管理与创新学院大数据与人工智能竞赛Task 1数据集为例 

---
# Understanding Economic Tradeoffs Between Human and AI Agents in Bargaining Games 

**Title (ZH)**: 理解人类与AI代理在谈判游戏中的人工智能与人类经济权衡 

**Authors**: Crystal Qian, Kehang Zhu, John Horton, Benjamin S. Manning, Vivian Tsai, James Wexler, Nithum Thain  

**Link**: [PDF](https://arxiv.org/pdf/2509.09071)  

**Abstract**: Coordination tasks traditionally performed by humans are increasingly being delegated to autonomous agents. As this pattern progresses, it becomes critical to evaluate not only these agents' performance but also the processes through which they negotiate in dynamic, multi-agent environments. Furthermore, different agents exhibit distinct advantages: traditional statistical agents, such as Bayesian models, may excel under well-specified conditions, whereas large language models (LLMs) can generalize across contexts. In this work, we compare humans (N = 216), LLMs (GPT-4o, Gemini 1.5 Pro), and Bayesian agents in a dynamic negotiation setting that enables direct, identical-condition comparisons across populations, capturing both outcomes and behavioral dynamics. Bayesian agents extract the highest surplus through aggressive optimization, at the cost of frequent trade rejections. Humans and LLMs can achieve similar overall surplus, but through distinct behaviors: LLMs favor conservative, concessionary trades with few rejections, while humans employ more strategic, risk-taking, and fairness-oriented behaviors. Thus, we find that performance parity -- a common benchmark in agent evaluation -- can conceal fundamental differences in process and alignment, which are critical for practical deployment in real-world coordination tasks. 

**Abstract (ZH)**: 传统上由人类执行的合作任务越来越多地被自主代理接管。随着这一模式的发展，不仅需要评估这些代理的表现，还需要评估它们在动态多代理环境中谈判的过程。此外，不同的代理展现出不同的优势：传统统计代理，如贝叶斯模型，在特定条件下可能表现出色，而大型语言模型（LLMs）则能够在不同场景中泛化。在本研究中，我们将人类（N=216）、LLMs（GPT-4o、Gemini 1.5 Pro）和贝叶斯代理在一种动态谈判环境中进行对比，该环境能够跨群体提供直接且条件一致的比较，捕捉到结果和行为动力学。贝叶斯代理通过激进优化获得最高的剩余价值，但代价是频繁的交易拒绝。人类和LLMs可以实现类似的总体剩余价值，但通过不同的行为：LLMs偏好保守、让步的交易且拒绝较少，而人类则采取更具战略性和冒险性的公平导向行为。因此，我们发现代理性能一致——代理评估中常用的标准——可能掩盖了过程和对齐的根本差异，这对于实际部署在实际协调任务中的代理至关重要。 

---
# Instructional Prompt Optimization for Few-Shot LLM-Based Recommendations on Cold-Start Users 

**Title (ZH)**: 基于少样本大语言模型推荐体系中的冷启动用户指令提示优化 

**Authors**: Haowei Yang, Yushang Zhao, Sitao Min, Bo Su, Chao Yao, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09066)  

**Abstract**: The cold-start user issue further compromises the effectiveness of recommender systems in limiting access to the historical behavioral information. It is an effective pipeline to optimize instructional prompts on a few-shot large language model (LLM) used in recommender tasks. We introduce a context-conditioned prompt formulation method P(u,\ Ds)\ \rightarrow\ R\widehat, where u is a cold-start user profile, Ds is a curated support set, and R\widehat is the predicted ranked list of items. Based on systematic experimentation with transformer-based autoregressive LLMs (BioGPT, LLaMA-2, GPT-4), we provide empirical evidence that optimal exemplar injection and instruction structuring can significantly improve the precision@k and NDCG scores of such models in low-data settings. The pipeline uses token-level alignments and embedding space regularization with a greater semantic fidelity. Our findings not only show that timely composition is not merely syntactic but also functional as it is in direct control of attention scales and decoder conduct through inference. This paper shows that prompt-based adaptation may be considered one of the ways to address cold-start recommendation issues in LLM-based pipelines. 

**Abstract (ZH)**: 冷启动用户问题进一步削弱了推荐系统通过限制访问历史行为信息来提升效果的能力。基于少量样本的大语言模型（LLM）在推荐任务中的指令提示优化是一种有效的管道方法。我们提出了一种基于上下文的提示公式化方法P(u, Ds) → R̂，其中u是冷启动用户概要，Ds是精心构建的支持集，R̂是预测的项目 ranked 列表。基于对基于变换器的自回归大语言模型（BioGPT、LLaMA-2、GPT-4）的系统实验，我们提供了在数据量少的情况下，最佳示例注入和指令结构化可以显著提高这类模型的precision@k和NDCG分数的实证证据。该管道使用了基于token级别的对齐和嵌入空间正则化，具有更高的语义保真度。我们的发现不仅表明及时组合不仅仅是句法上的，而且也是功能性的，它可以直接影响注意尺度和解码器的行为。本文表明，基于提示的适应可能是解决基于大语言模型的管道中冷启动推荐问题的一种方式。 

---
# Uncertainty Awareness and Trust in Explainable AI- On Trust Calibration using Local and Global Explanations 

**Title (ZH)**: 不确定性意识与可解释人工智能中的信任校准——基于局部和全局解释的信任 calibration 研究 

**Authors**: Carina Newen, Daniel Bodemer, Sonja Glantz, Emmanuel Müller, Magdalena Wischnewski, Lenka Schnaubert  

**Link**: [PDF](https://arxiv.org/pdf/2509.08989)  

**Abstract**: Explainable AI has become a common term in the literature, scrutinized by computer scientists and statisticians and highlighted by psychological or philosophical researchers. One major effort many researchers tackle is constructing general guidelines for XAI schemes, which we derived from our study. While some areas of XAI are well studied, we focus on uncertainty explanations and consider global explanations, which are often left out. We chose an algorithm that covers various concepts simultaneously, such as uncertainty, robustness, and global XAI, and tested its ability to calibrate trust. We then checked whether an algorithm that aims to provide more of an intuitive visual understanding, despite being complicated to understand, can provide higher user satisfaction and human interpretability. 

**Abstract (ZH)**: 可解释AI已成为文献中的一个常见术语，受到计算机科学家和统计学家的审查，并引起了心理学家或哲学家的高度重视。许多研究者的一项主要努力是构建通用的可解释AI方案指南，我们从我们的研究中得出了这些指南。尽管某些可解释AI领域已研究得较为充分，但我们专注于不确定性解释，并考虑了通常被忽略的全局解释。我们选择了能够同时涵盖各种概念（如不确定性、鲁棒性和全局可解释性）的算法，并测试了其校准信任的能力。然后我们检查了一种旨在提供更直观的视觉理解的算法，尽管其理解起来比较复杂，是否能够提供更高的用户满意度和人类可解释性。 

---
# ForTIFAI: Fending Off Recursive Training Induced Failure for AI Models 

**Title (ZH)**: ForTIFAI: 避免由递归训练引发的AI模型故障 

**Authors**: Soheil Zibakhsh Shabgahi, Pedram Aghazadeh, Azalia Mirhosseini, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2509.08972)  

**Abstract**: The increasing reliance on generative AI models has accelerated the generation rate of synthetic data, with some projections suggesting that most available new data for training could be machine-generated by 2030. This shift to a mainly synthetic content presents a critical challenge: repeated training in synthetic data leads to a phenomenon known as model collapse, where model performance degrades over generations of training, eventually rendering the models ineffective. Although prior studies have explored the causes and detection of model collapse, existing mitigation strategies remain limited.
In this paper, we identify model overconfidence in their self-generated data as a key driver of collapse. Building on this observation, we propose a confidence-aware loss function that downweights high-confidence predictions during training. We introduce a novel loss function we call Truncated Cross Entropy (TCE). We demonstrate that TCE significantly delays model collapse in recursive training.
We provide a model-agnostic framework that links the loss function design to model collapse mitigation and validate our approach both theoretically and empirically, showing that it can extend the model's fidelity interval before collapse by more than 2.3x. Finally, we show that our method generalizes across modalities. These findings suggest that the design of loss functions provides a simple yet powerful tool for preserving the quality of generative models in the era of increasing synthetic data. 

**Abstract (ZH)**: 不断增加对生成AI模型的依赖加速了合成数据的生成速度，有预测认为到2030年，大部分用于训练的新数据可能是机器生成的。这种主要由合成内容构成的转变提出了一项关键挑战：在合成数据上反复训练会导致模型退化现象——模型性能在训练代际中逐渐下降，最终使模型无效。尽管先前的研究已经探讨了模型退化的原因及其检测方法，现有的缓解策略仍然有限。

本文中，我们识别出模型在自生成数据上的高置信度作为模型退化的关键驱动因素。基于这一点观察，我们提出了一种注意置信度的loss函数，在训练过程中降低高置信度预测的权重。我们引入了一种新的loss函数，称为截断交叉熵（TCE）。我们证明，TCE可以显著延缓循环训练中的模型退化现象。

我们提供了一个模型无关的框架，将loss函数的设计与模型退化缓解联系起来，并从理论和实证两个方面验证了我们的方法，结果显示它可以使模型在退化前的质量间隔延长超过2.3倍。最后，我们表明我们的方法在不同模态上具有泛化能力。这些发现表明，loss函数设计提供了一个简单而强大的工具，用于在合成数据增加的时代保持生成模型的质量。 

---
# Global Constraint LLM Agents for Text-to-Model Translation 

**Title (ZH)**: 全球约束大语言模型代理用于文本到模型的翻译 

**Authors**: Junyang Cai, Serdar Kadioglu, Bistra Dilkina  

**Link**: [PDF](https://arxiv.org/pdf/2509.08970)  

**Abstract**: Natural language descriptions of optimization or satisfaction problems are challenging to translate into correct MiniZinc models, as this process demands both logical reasoning and constraint programming expertise. We introduce a framework that addresses this challenge with an agentic approach: multiple specialized large language model (LLM) agents decompose the modeling task by global constraint type. Each agent is dedicated to detecting and generating code for a specific class of global constraint, while a final assembler agent integrates these constraint snippets into a complete MiniZinc model. By dividing the problem into smaller, well-defined sub-tasks, each LLM handles a simpler reasoning challenge, potentially reducing overall complexity. We conduct initial experiments with several LLMs and show better performance against baselines such as one-shot prompting and chain-of-thought prompting. Finally, we outline a comprehensive roadmap for future work, highlighting potential enhancements and directions for improvement. 

**Abstract (ZH)**: 自然语言对优化或满意问题的描述转换为正确的MiniZinc模型具有挑战性，这要求同时具备逻辑推理能力和约束编程专业知识。我们提出了一种以代理为导向的框架：多个专门的大语言模型（LLM）代理通过全局约束类型分解建模任务。每个代理专注于检测并生成特定类别的全局约束代码，而最终的组装代理则将这些约束片段整合成一个完整的MiniZinc模型。通过将问题分解为更小的、定义清晰的子任务，每个LLM可以处理一个更简单的推理挑战，从而可能降低整体复杂性。我们使用几种LLM进行了初步实验，并展示了其相对于单次提示和链式思考提示等基线方法的更好性能。最后，我们概述了未来工作的一个综合路线图，指出了潜在的改进方向和增强方案。 

---
# Automated Unity Game Template Generation from GDDs via NLP and Multi-Modal LLMs 

**Title (ZH)**: 基于NLP和多模态大语言模型从GDD自动生成Unity游戏模板 

**Authors**: Amna Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.08847)  

**Abstract**: This paper presents a novel framework for automated game template generation by transforming Game Design Documents (GDDs) into functional Unity game prototypes using Natural Language Processing (NLP) and multi-modal Large Language Models (LLMs). We introduce an end-to-end system that parses GDDs, extracts structured game specifications, and synthesizes Unity-compatible C# code that implements the core mechanics, systems, and architecture defined in the design documentation. Our approach combines a fine-tuned LLaMA-3 model specialized for Unity code generation with a custom Unity integration package that streamlines the implementation process. Evaluation results demonstrate significant improvements over baseline models, with our fine-tuned model achieving superior performance (4.8/5.0 average score) compared to state-of-the-art LLMs across compilation success, GDD adherence, best practices adoption, and code modularity metrics. The generated templates demonstrate high adherence to GDD specifications across multiple game genres. Our system effectively addresses critical gaps in AI-assisted game development, positioning LLMs as valuable tools in streamlining the transition from game design to implementation. 

**Abstract (ZH)**: 本文提出了一种新颖的框架，通过自然语言处理（NLP）和多模态大型语言模型（LLMs）将游戏设计文档（GDDs）转换为功能性的Unity游戏原型，实现了自动化游戏模板生成。我们引入了一个端到端的系统，该系统解析GDDs，提取结构化的游戏规范，并合成Unity兼容的C#代码，以实现设计文档中定义的核心机制、系统和架构。该方法结合了专为Unity代码生成细调的LLaMA-3模型和一个自定义的Unity集成包，以简化实现过程。评估结果表明，与基线模型相比，我们细调的模型在编译成功率、GDD一致性、最佳实践采用和代码模块性指标方面表现出显著改进，评分平均为4.8/5.0，优于最新的LLMs。生成的模板在多种游戏类型中都高度符合GDD规范。我们的系统有效地填补了辅助游戏开发中的关键空白，将LLMs定位为简化从游戏设计到实现过渡的宝贵工具。 

---
# An Interval Type-2 Version of Bayes Theorem Derived from Interval Probability Range Estimates Provided by Subject Matter Experts 

**Title (ZH)**: 基于领域专家提供的区间概率范围估计推导出的区间类型2贝叶斯定理 

**Authors**: John T. Rickard, William A. Dembski, James Rickards  

**Link**: [PDF](https://arxiv.org/pdf/2509.08834)  

**Abstract**: Bayesian inference is widely used in many different fields to test hypotheses against observations. In most such applications, an assumption is made of precise input values to produce a precise output value. However, this is unrealistic for real-world applications. Often the best available information from subject matter experts (SMEs) in a given field is interval range estimates of the input probabilities involved in Bayes Theorem. This paper provides two key contributions to extend Bayes Theorem to an interval type-2 (IT2) version. First, we develop an IT2 version of Bayes Theorem that uses a novel and conservative method to avoid potential inconsistencies in the input IT2 MFs that otherwise might produce invalid output results. We then describe a novel and flexible algorithm for encoding SME-provided intervals into IT2 fuzzy membership functions (MFs), which we can use to specify the input probabilities in Bayes Theorem. Our algorithm generalizes and extends previous work on this problem that primarily addressed the encoding of intervals into word MFs for Computing with Words applications. 

**Abstract (ZH)**: 贝叶斯推断广泛应用于多个领域以假设测试与观测数据的对比。然而，在大多数此类应用中，人们假设输入值精确以产生精确的输出值。但在实际应用中，这是不现实的。通常，在贝叶斯定理中涉及的输入概率的最佳区间估计是由给定领域内的专家提供的区间范围估计。本文为将贝叶斯定理扩展到区间类型-2 (IT2) 版本做出了两个关键贡献。首先，我们开发了一个采用新颖且保守方法的IT2版本的贝叶斯定理，以避免输入IT2隶属函数（MFs）可能产生的潜在不一致性，从而产生无效的输出结果。其次，我们描述了一个新颖且灵活的算法，用于将专家提供的区间编码为IT2模糊隶属函数（MFs），以此来指定贝叶斯定理中的输入概率。该算法扩展了以往主要针对将区间编码为词汇隶属函数的研究，以应用于计算与词语的应用。 

---
# ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms 

**Title (ZH)**: ButterflyQuant：可通过可学习正交蝴蝶变换实现的超低比特LLM量化 

**Authors**: Bingxin Xu, Zhen Dong, Oussama Elachqar, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09679)  

**Abstract**: Large language models require massive memory footprints, severely limiting deployment on consumer hardware. Quantization reduces memory through lower numerical precision, but extreme 2-bit quantization suffers from catastrophic performance loss due to outliers in activations. Rotation-based methods such as QuIP and QuaRot apply orthogonal transforms to eliminate outliers before quantization, using computational invariance: $\mathbf{y} = \mathbf{Wx} = (\mathbf{WQ}^T)(\mathbf{Qx})$ for orthogonal $\mathbf{Q}$. However, these methods use fixed transforms--Hadamard matrices achieving optimal worst-case coherence $\mu = 1/\sqrt{n}$--that cannot adapt to specific weight distributions. We identify that different transformer layers exhibit distinct outlier patterns, motivating layer-adaptive rotations rather than one-size-fits-all approaches. We propose ButterflyQuant, which replaces Hadamard rotations with learnable butterfly transforms parameterized by continuous Givens rotation angles. Unlike Hadamard's discrete $\{+1, -1\}$ entries that are non-differentiable and prohibit gradient-based learning, butterfly transforms' continuous parameterization enables smooth optimization while guaranteeing orthogonality by construction. This orthogonal constraint ensures theoretical guarantees in outlier suppression while achieving $O(n \log n)$ computational complexity with only $\frac{n \log n}{2}$ learnable parameters. We further introduce a uniformity regularization on post-transformation activations to promote smoother distributions amenable to quantization. Learning requires only 128 calibration samples and converges in minutes on a single GPU--a negligible one-time cost. On LLaMA-2-7B with 2-bit quantization, ButterflyQuant achieves 15.4 perplexity versus 22.1 for QuaRot. 

**Abstract (ZH)**: 大型语言模型需要庞大的内存 footprint，严重限制了其在消费级硬件上的部署。通过降低数值精度来进行量化可以减少内存占用，但极端的2位量化由于激活值中的异常值会导致灾难性的性能下降。基于旋转的方法，如QuIP和QuaRot，在量化之前应用正交变换以消除异常值，利用计算不变性：$\mathbf{y} = \mathbf{Wx} = (\mathbf{WQ}^T)(\mathbf{Qx})$，其中$\mathbf{Q}$是正交矩阵。然而，这些方法使用固定的变换——最优最坏情况相干性$\mu = 1/\sqrt{n}$的哈达玛矩阵——不能适应特定的权重分布。我们发现不同变压器层表现出不同的异常值模式，这促使我们采用层自适应旋转而非一刀切的方法。我们提出了ButterflyQuant，用由连续的Givens旋转角度参数化的蝴蝶变换替换哈达玛旋转。与哈达玛矩阵只有$\{+1, -1\}$的离散条目不同，后者禁止基于梯度的学习，而蝴蝶变换的连续参数化允许平滑优化，并通过构造保证正交性。这种正交约束确保了异常值抑制的理论保证，同时通过仅使用$\frac{n \log n}{2}$个可学习参数实现了$O(n \log n)$的计算复杂度。我们还引入了一种变换后激活的均匀性正则化，以促进更适于量化的平滑分布。训练只需要128个校准样本，并在单个GPU上几分钟内收敛——这是一个可以忽略不计的一次性成本。在使用2位量化时，ButterflyQuant在LLaMA-2-7B上实现了15.4的困惑度，而QuaRot为22.1。 

---
# CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models 

**Title (ZH)**: CDE: 好奇心驱动的探索在大型语言模型高效 reinforcement learning 中的应用 

**Authors**: Runpeng Dai, Linfeng Song, Haolin Liu, Zhenwen Liang, Dian Yu, Haitao Mi, Zhaopeng Tu, Rui Liu, Tong Zheng, Hongtu Zhu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09675)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful paradigm for enhancing the reasoning ability of Large Language Models (LLMs). Yet current RLVR methods often explore poorly, leading to premature convergence and entropy collapse. To address this challenge, we introduce Curiosity-Driven Exploration (CDE), a framework that leverages the model's own intrinsic sense of curiosity to guide exploration. We formalize curiosity with signals from both the actor and the critic: for the actor, we use perplexity over its generated response, and for the critic, we use the variance of value estimates from a multi-head architecture. Both signals serve as an exploration bonus within the RLVR framework to guide the model. Our theoretical analysis shows that the actor-wise bonus inherently penalizes overconfident errors and promotes diversity among correct responses; moreover, we connect the critic-wise bonus to the well-established count-based exploration bonus in RL. Empirically, our method achieves an approximate +3 point improvement over standard RLVR using GRPO/PPO on AIME benchmarks. Further analysis identifies a calibration collapse mechanism within RLVR, shedding light on common LLM failure modes. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）增强了大型语言模型的推理能力，但当前的RLVR方法往往探索不足，导致过早收敛和熵塌陷。为解决这一挑战，我们介绍了好奇心驱动探索（CDE）框架，该框架利用模型自身的内在好奇心来引导探索。我们通过来自行为者和评论者的信号形式化好奇心：对行为者，我们使用生成响应的困惑度；对评论者，我们使用多头架构中价值估计的方差。两种信号在RLVR框架内作为探索奖励来引导模型。我们的理论分析表明，行为者奖励本质上惩罚过度自信的错误，促进正确响应的多样性；此外，我们将评论者奖励与RL中成熟的计数基础探索奖励联系起来。实验上，我们的方法在AIME基准上使用GRPO/PPO实现了大约+3分的改进。进一步分析揭示了RLVR中的校准塌缩机制，阐明了常见的LLM失败模式。 

---
# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning 

**Title (ZH)**: SimpleVLA-RL：通过强化学习拓展VLA训练 

**Authors**: Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.09674)  

**Abstract**: Vision-Language-Action (VLA) models have recently emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges: (i) the scarcity and high cost of large-scale human-operated robotic trajectories required for SFT scaling, and (ii) limited generalization to tasks involving distribution shift. Recent breakthroughs in Large Reasoning Models (LRMs) demonstrate that reinforcement learning (RL) can dramatically enhance step-by-step reasoning capabilities, raising a natural question: Can RL similarly improve the long-horizon step-by-step action planning of VLA? In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models. Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation. When applied to OpenVLA-OFT, SimpleVLA-RL achieves SoTA performance on LIBERO and even outperforms $\pi_0$ on RoboTwin 1.0\&2.0 with the exploration-enhancing strategies we introduce. SimpleVLA-RL not only reduces dependence on large-scale data and enables robust generalization, but also remarkably surpasses SFT in real-world tasks. Moreover, we identify a novel phenomenon ``pushcut'' during RL training, wherein the policy discovers previously unseen patterns beyond those seen in the previous training process. Github: this https URL 

**Abstract (ZH)**: 基于视觉-语言-动作的强化学习框架（SimpleVLA-RL）：提升机器人长期步骤规划能力 

---
# Feasibility-Guided Fair Adaptive Offline Reinforcement Learning for Medicaid Care Management 

**Title (ZH)**: 基于可行性的公正自适应离线强化学习在 Medicaid 照顾管理中的可行性研究 

**Authors**: Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, Rajaie Batniji  

**Link**: [PDF](https://arxiv.org/pdf/2509.09655)  

**Abstract**: We introduce Feasibility-Guided Fair Adaptive Reinforcement Learning (FG-FARL), an offline RL procedure that calibrates per-group safety thresholds to reduce harm while equalizing a chosen fairness target (coverage or harm) across protected subgroups. Using de-identified longitudinal trajectories from a Medicaid population health management program, we evaluate FG-FARL against behavior cloning (BC) and HACO (Hybrid Adaptive Conformal Offline RL; a global conformal safety baseline). We report off-policy value estimates with bootstrap 95% confidence intervals and subgroup disparity analyses with p-values. FG-FARL achieves comparable value to baselines while improving fairness metrics, demonstrating a practical path to safer and more equitable decision support. 

**Abstract (ZH)**: 基于可行性的公平自适应强化学习（FG-FARL）：一种减少危害并平等化公平目标的离线强化学习方法 

---
# Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations 

**Title (ZH)**: 基于检索增强生成的无线电规则可靠解释方法 

**Authors**: Zakaria El Kassimi, Fares Fourati, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2509.09651)  

**Abstract**: We study question answering in the domain of radio regulations, a legally sensitive and high-stakes area. We propose a telecom-specific Retrieval-Augmented Generation (RAG) pipeline and introduce, to our knowledge, the first multiple-choice evaluation set for this domain, constructed from authoritative sources using automated filtering and human validation. To assess retrieval quality, we define a domain-specific retrieval metric, under which our retriever achieves approximately 97% accuracy. Beyond retrieval, our approach consistently improves generation accuracy across all tested models. In particular, while naively inserting documents without structured retrieval yields only marginal gains for GPT-4o (less than 1%), applying our pipeline results in nearly a 12% relative improvement. These findings demonstrate that carefully targeted grounding provides a simple yet strong baseline and an effective domain-specific solution for regulatory question answering. All code and evaluation scripts, along with our derived question-answer dataset, are available at this https URL. 

**Abstract (ZH)**: 我们在无线电管理法规领域的问答研究，这是一个法律敏感且高风险的领域。我们提出了一种针对电信的检索增强生成（RAG）管道，并且，据我们所知，首次构建了一个来自权威来源的多选评价数据集，使用自动过滤和人工验证。为了评估检索质量，我们定义了一个领域特定的检索指标，在此指标下，我们的检索器准确率达到约97%。除了检索之外，我们的方法在所有测试的模型中都一致地提高了生成准确性。特别是，与未经结构化检索直接插入文档的情况相比，我们的管道使得GPT-4o的相对改进达到近12%。这些发现表明，精确的目标定位提供了简单而有效的基线和领域特定解决方案，以解决监管问答问题。所有代码和评价脚本，以及我们衍生的问答数据集均可在以下链接获取。 

---
# Explaining Concept Drift through the Evolution of Group Counterfactuals 

**Title (ZH)**: 通过群体反事实的演变解释概念漂移 

**Authors**: Ignacy Stępka, Jerzy Stefanowski  

**Link**: [PDF](https://arxiv.org/pdf/2509.09616)  

**Abstract**: Machine learning models in dynamic environments often suffer from concept drift, where changes in the data distribution degrade performance. While detecting this drift is a well-studied topic, explaining how and why the model's decision-making logic changes still remains a significant challenge. In this paper, we introduce a novel methodology to explain concept drift by analyzing the temporal evolution of group-based counterfactual explanations (GCEs). Our approach tracks shifts in the GCEs' cluster centroids and their associated counterfactual action vectors before and after a drift. These evolving GCEs act as an interpretable proxy, revealing structural changes in the model's decision boundary and its underlying rationale. We operationalize this analysis within a three-layer framework that synergistically combines insights from the data layer (distributional shifts), the model layer (prediction disagreement), and our proposed explanation layer. We show that such holistic view allows for a more comprehensive diagnosis of drift, making it possible to distinguish between different root causes, such as a spatial data shift versus a re-labeling of concepts. 

**Abstract (ZH)**: 机器学习模型在动态环境中的概念漂移解释方法探究：通过基于组的反事实解释的时间演化分析 

---
# LoCoBench: A Benchmark for Long-Context Large Language Models in Complex Software Engineering 

**Title (ZH)**: LoCoBench: 一种复杂软件工程中的长上下文大型语言模型基准测试 

**Authors**: Jielin Qiu, Zuxin Liu, Zhiwei Liu, Rithesh Murthy, Jianguo Zhang, Haolin Chen, Shiyu Wang, Ming Zhu, Liangwei Yang, Juntao Tan, Zhepeng Cen, Cheng Qian, Shelby Heinecke, Weiran Yao, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09614)  

**Abstract**: The emergence of long-context language models with context windows extending to millions of tokens has created new opportunities for sophisticated code understanding and software development evaluation. We propose LoCoBench, a comprehensive benchmark specifically designed to evaluate long-context LLMs in realistic, complex software development scenarios. Unlike existing code evaluation benchmarks that focus on single-function completion or short-context tasks, LoCoBench addresses the critical evaluation gap for long-context capabilities that require understanding entire codebases, reasoning across multiple files, and maintaining architectural consistency across large-scale software systems. Our benchmark provides 8,000 evaluation scenarios systematically generated across 10 programming languages, with context lengths spanning 10K to 1M tokens, a 100x variation that enables precise assessment of long-context performance degradation in realistic software development settings. LoCoBench introduces 8 task categories that capture essential long-context capabilities: architectural understanding, cross-file refactoring, multi-session development, bug investigation, feature implementation, code comprehension, integration testing, and security analysis. Through a 5-phase pipeline, we create diverse, high-quality scenarios that challenge LLMs to reason about complex codebases at unprecedented scale. We introduce a comprehensive evaluation framework with 17 metrics across 4 dimensions, including 8 new evaluation metrics, combined in a LoCoBench Score (LCBS). Our evaluation of state-of-the-art long-context models reveals substantial performance gaps, demonstrating that long-context understanding in complex software development represents a significant unsolved challenge that demands more attention. LoCoBench is released at: this https URL. 

**Abstract (ZH)**: 长上下文语言模型的出现为复杂的代码理解和软件开发评估带来了新的机会。我们提出了LoCoBench，一个专门设计用于在实际复杂的软件开发场景中评估长上下文LLM的综合基准。与现有的主要关注单函数完成或短上下文任务的代码评估基准不同，LoCoBench 解决了需理解整个代码库、在多个文件间进行推理并维护大规模软件系统的架构一致性等长上下文能力的评估空白。该基准提供了涵盖10种编程语言的8000个系统生成的评估场景，上下文长度从10K到1M不等，跨度达100倍，适用于在现实软件开发环境中精确评估长上下文性能的退化。LoCoBench 引入了8个任务类别，涵盖了关键的长上下文能力：架构理解、跨文件重构、多会话开发、bug调查、功能实现、代码理解、集成测试和安全性分析。通过5阶段管道，我们创建了多样的高质量场景，挑战LLM以前所未有的规模推理复杂的代码库。我们引入了一个全面的评估框架，包括4个维度的17个指标，其中8个是新的评估指标，结合成LoCoBench得分（LCBS）。我们的评估表明，最先进的长上下文模型存在显著的性能差距，表明复杂的软件开发中的长上下文理解是一个重要的未解决挑战，需要更多关注。LoCoBench可在以下链接获取：this https URL。 

---
# Mechanistic Learning with Guided Diffusion Models to Predict Spatio-Temporal Brain Tumor Growth 

**Title (ZH)**: 基于引导扩散模型的机制性学习以预测空间-时间脑肿瘤生长 

**Authors**: Daria Laslo, Efthymios Georgiou, Marius George Linguraru, Andreas Rauschecker, Sabine Muller, Catherine R. Jutzeler, Sarah Bruningk  

**Link**: [PDF](https://arxiv.org/pdf/2509.09610)  

**Abstract**: Predicting the spatio-temporal progression of brain tumors is essential for guiding clinical decisions in neuro-oncology. We propose a hybrid mechanistic learning framework that combines a mathematical tumor growth model with a guided denoising diffusion implicit model (DDIM) to synthesize anatomically feasible future MRIs from preceding scans. The mechanistic model, formulated as a system of ordinary differential equations, captures temporal tumor dynamics including radiotherapy effects and estimates future tumor burden. These estimates condition a gradient-guided DDIM, enabling image synthesis that aligns with both predicted growth and patient anatomy. We train our model on the BraTS adult and pediatric glioma datasets and evaluate on 60 axial slices of in-house longitudinal pediatric diffuse midline glioma (DMG) cases. Our framework generates realistic follow-up scans based on spatial similarity metrics. It also introduces tumor growth probability maps, which capture both clinically relevant extent and directionality of tumor growth as shown by 95th percentile Hausdorff Distance. The method enables biologically informed image generation in data-limited scenarios, offering generative-space-time predictions that account for mechanistic priors. 

**Abstract (ZH)**: 基于空间-时间进展的脑肿瘤预测对于神经 Oncology 的临床决策至关重要。我们提出了一种结合数学肿瘤生长模型和引导去噪扩散隐式模型（DDIM）的混合机理学习框架，以从先前的扫描中合成符合解剖学的未来 MRI 图像。机理模型以常微分方程系统的形式表述，捕捉包括放疗效果在内的时间肿瘤动态，并估计未来肿瘤负荷。这些估计值条件引导梯度下的 DDIM，使得生成的图像与预测的生长和患者解剖结构相一致。我们在 BraTS 成人和儿童胶质瘤数据集上训练模型，并在内部儿童弥漫中线胶质瘤 (DMG) 横截面数据集的 60 个层面进行评估。该框架基于空间相似度指标生成现实的随访扫描。它还引入了肿瘤生长概率图，这些图捕捉由 95 个百分点 Hausdorff 距离所示的临床相关范围和肿瘤生长的方向性。该方法在数据有限的情况下实现生物学指导的图像生成，提供了考虑机理先验的空间-时间生成预测。 

---
# Graph Alignment via Dual-Pass Spectral Encoding and Latent Space Communication 

**Title (ZH)**: 图对齐 via 双通道谱编码和潜在空间通信 

**Authors**: Maysam Behmanesh, Erkan Turan, Maks Ovsjanikov  

**Link**: [PDF](https://arxiv.org/pdf/2509.09597)  

**Abstract**: Graph alignment-the problem of identifying corresponding nodes across multiple graphs-is fundamental to numerous applications. Most existing unsupervised methods embed node features into latent representations to enable cross-graph comparison without ground-truth correspondences. However, these methods suffer from two critical limitations: the degradation of node distinctiveness due to oversmoothing in GNN-based embeddings, and the misalignment of latent spaces across graphs caused by structural noise, feature heterogeneity, and training instability, ultimately leading to unreliable node correspondences. We propose a novel graph alignment framework that simultaneously enhances node distinctiveness and enforces geometric consistency across latent spaces. Our approach introduces a dual-pass encoder that combines low-pass and high-pass spectral filters to generate embeddings that are both structure-aware and highly discriminative. To address latent space misalignment, we incorporate a geometry-aware functional map module that learns bijective and isometric transformations between graph embeddings, ensuring consistent geometric relationships across different representations. Extensive experiments on graph benchmarks demonstrate that our method consistently outperforms existing unsupervised alignment baselines, exhibiting superior robustness to structural inconsistencies and challenging alignment scenarios. Additionally, comprehensive evaluation on vision-language benchmarks using diverse pretrained models shows that our framework effectively generalizes beyond graph domains, enabling unsupervised alignment of vision and language representations. 

**Abstract (ZH)**: 图对齐——即在多个图中识别对应节点的问题——是众多应用的基础。现有的大多数无监督方法通过将节点特征嵌入到潜在表示中，以在没有地面truth对应关系的情况下进行跨图比较。然而，这些方法面临着两个关键限制：基于GNN的嵌入中节点区分度下降的泛化过度平滑现象，以及由于结构噪声、特征异质性和训练不稳定性导致的跨图潜在空间对齐不良，最终导致节点对应关系不可靠。我们提出了一种新颖的图对齐框架，该框架同时增强了节点的区分度并确保跨潜在空间的一致几何一致性。我们的方法引入了一种双通道编码器，结合低通和高通谱滤波器生成结构意识强且高度区分的嵌入。为了解决潜在空间对齐不良的问题，我们采用了一种几何意识的功能映射模块，学习图嵌入之间的双射和等距变换，从而在不同的表示之间保持一致的几何关系。在广泛的图基准实验中，我们的方法在所有无监督对齐基线方法中表现优异，展现出对结构不一致和挑战性对齐场景的优越鲁棒性。此外，通过对多种预训练模型在视觉-语言基准上的全面评估，证明了我们框架的有效泛化能力，使其能够超越图域，实现视觉和语言表示的无监督对齐。 

---
# ObjectReact: Learning Object-Relative Control for Visual Navigation 

**Title (ZH)**: ObjectReact: 学习对象相对控制 Woche视觉导航 

**Authors**: Sourav Garg, Dustin Craggs, Vineeth Bhat, Lachlan Mares, Stefan Podgorski, Madhava Krishna, Feras Dayoub, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2509.09594)  

**Abstract**: Visual navigation using only a single camera and a topological map has recently become an appealing alternative to methods that require additional sensors and 3D maps. This is typically achieved through an "image-relative" approach to estimating control from a given pair of current observation and subgoal image. However, image-level representations of the world have limitations because images are strictly tied to the agent's pose and embodiment. In contrast, objects, being a property of the map, offer an embodiment- and trajectory-invariant world representation. In this work, we present a new paradigm of learning "object-relative" control that exhibits several desirable characteristics: a) new routes can be traversed without strictly requiring to imitate prior experience, b) the control prediction problem can be decoupled from solving the image matching problem, and c) high invariance can be achieved in cross-embodiment deployment for variations across both training-testing and mapping-execution settings. We propose a topometric map representation in the form of a "relative" 3D scene graph, which is used to obtain more informative object-level global path planning costs. We train a local controller, dubbed "ObjectReact", conditioned directly on a high-level "WayObject Costmap" representation that eliminates the need for an explicit RGB input. We demonstrate the advantages of learning object-relative control over its image-relative counterpart across sensor height variations and multiple navigation tasks that challenge the underlying spatial understanding capability, e.g., navigating a map trajectory in the reverse direction. We further show that our sim-only policy is able to generalize well to real-world indoor environments. Code and supplementary material are accessible via project page: this https URL 

**Abstract (ZH)**: 仅使用单个摄像头和拓扑地图的视觉导航recently成为了一种有吸引力的替代方法，这种方法无需额外传感器和3D地图。这一目标通常通过估计给定当前观察和子目标图像 pair 的“图像相对”控制来实现。然而，基于图像的世界表示存在局限性，因为图像严格依赖于代理的姿态和体现。相比之下，物体作为地图的属性，提供了体现和轨迹不变的世界表示。在本文中，我们提出了一种新的“物体相对”控制的学习范式，具有若干 desirable 特性：a) 新的路径可以被遍历而无需严格模仿先前的经验，b) 控制预测问题可以从解决图像匹配问题中解耦，c) 在训练-测试和建图-执行设置中实现高不变性。我们提出了一种拓扑地图表示，形式为“相对”的3D 场景图，用于获得更具信息量的物体级全局路径规划代价。我们训练了一个本地控制器，称为“ObjectReact”，直接基于消除显式 RGB 输入的“WayObject 成本图”高层表示。我们展示了学习物体相对控制在多种导航任务中的优势，这些任务对底层空间理解能力提出了挑战，例如逆向导航地图轨迹。我们进一步证明，仅使用模拟策略能够很好地泛化到真实世界的室内环境。代码和补充材料可通过项目页面访问：this https URL。 

---
# Fluent but Unfeeling: The Emotional Blind Spots of Language Models 

**Title (ZH)**: 流畅而不 emotionally resonant：语言模型的情感盲点 

**Authors**: Bangzhao Shu, Isha Joshi, Melissa Karnaze, Anh C. Pham, Ishita Kakkar, Sindhu Kothe, Arpine Hovasapian, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2509.09593)  

**Abstract**: The versatility of Large Language Models (LLMs) in natural language understanding has made them increasingly popular in mental health research. While many studies explore LLMs' capabilities in emotion recognition, a critical gap remains in evaluating whether LLMs align with human emotions at a fine-grained level. Existing research typically focuses on classifying emotions into predefined, limited categories, overlooking more nuanced expressions. To address this gap, we introduce EXPRESS, a benchmark dataset curated from Reddit communities featuring 251 fine-grained, self-disclosed emotion labels. Our comprehensive evaluation framework examines predicted emotion terms and decomposes them into eight basic emotions using established emotion theories, enabling a fine-grained comparison. Systematic testing of prevalent LLMs under various prompt settings reveals that accurately predicting emotions that align with human self-disclosed emotions remains challenging. Qualitative analysis further shows that while certain LLMs generate emotion terms consistent with established emotion theories and definitions, they sometimes fail to capture contextual cues as effectively as human self-disclosures. These findings highlight the limitations of LLMs in fine-grained emotion alignment and offer insights for future research aimed at enhancing their contextual understanding. 

**Abstract (ZH)**: 大型语言模型在自然语言理解的多功能性使其在心理健康研究中日益受欢迎。尽管许多研究探讨了大型语言模型在情绪识别方面的能力，但在评估这些模型是否在细粒度层次上与人类情绪一致方面仍存在一个关键缺口。现有研究通常专注于将情绪分类为预定义的有限类别，而忽略了更细致的表现形式。为填补这一空白，我们介绍了EXPRESS，这是一个基准数据集，它来自 Reddit 社区，包含 251 个细粒度的自我披露情绪标签。我们的全面评估框架检查预测的情绪术语，并使用建立的情绪理论将它们分解为八种基本情绪，从而实现细粒度比较。在各种提示设置下对常见的大型语言模型进行系统性测试显示出，准确预测与人类自我披露情绪一致的情绪仍然是一个挑战。进一步的定性分析表明，虽然某些大型语言模型产生与建立的情绪理论和定义一致的情绪术语，但它们有时未能像人类自我披露那样有效地捕捉到上下文线索。这些发现揭示了大型语言模型在细粒度情绪对齐方面的局限性，并为未来旨在提高其上下文理解能力的研究提供了见解。 

---
# Invisible Attributes, Visible Biases: Exploring Demographic Shortcuts in MRI-based Alzheimer's Disease Classification 

**Title (ZH)**: 隐形属性，显性偏见：基于MRI的阿尔茨海默病分类中的人口统计捷径探索 

**Authors**: Akshit Achara, Esther Puyol Anton, Alexander Hammers, Andrew P. King  

**Link**: [PDF](https://arxiv.org/pdf/2509.09558)  

**Abstract**: Magnetic resonance imaging (MRI) is the gold standard for brain imaging. Deep learning (DL) algorithms have been proposed to aid in the diagnosis of diseases such as Alzheimer's disease (AD) from MRI scans. However, DL algorithms can suffer from shortcut learning, in which spurious features, not directly related to the output label, are used for prediction. When these features are related to protected attributes, they can lead to performance bias against underrepresented protected groups, such as those defined by race and sex. In this work, we explore the potential for shortcut learning and demographic bias in DL based AD diagnosis from MRI. We first investigate if DL algorithms can identify race or sex from 3D brain MRI scans to establish the presence or otherwise of race and sex based distributional shifts. Next, we investigate whether training set imbalance by race or sex can cause a drop in model performance, indicating shortcut learning and bias. Finally, we conduct a quantitative and qualitative analysis of feature attributions in different brain regions for both the protected attribute and AD classification tasks. Through these experiments, and using multiple datasets and DL models (ResNet and SwinTransformer), we demonstrate the existence of both race and sex based shortcut learning and bias in DL based AD classification. Our work lays the foundation for fairer DL diagnostic tools in brain MRI. The code is provided at this https URL 

**Abstract (ZH)**: 磁共振成像（MRI）是脑部成像的金标准。深度学习（DL）算法已被提出用于从MRI扫描中诊断阿尔茨海默病（AD）等疾病。然而，DL算法可能会遭受捷径学习的问题，即使用与输出标签无直接关系的虚假特征进行预测。当这些特征与保护性特征相关时，它们可能导致模型性能对少数代表性不足的保护性群体产生偏差，例如按照种族和性别定义的群体。在本文中，我们探讨了基于DL的AD MRI诊断中捷径学习和人口统计偏差的潜在性。我们首先调查DL算法是否可以从3D脑部MRI扫描中识别种族或性别，以确定是否存在基于种族或性别的分布变化。接下来，我们研究训练集中的种族或性别不平衡是否会导致模型性能下降，这表明存在捷径学习和偏差。最后，我们对不同脑区的保护性特征和AD分类任务中的特征归因进行了定量和定性分析。通过这些实验，使用多个数据集和DL模型（ResNet和SwinTransformer），我们证明了基于DL的AD分类中存在基于种族和性别的捷径学习和偏差。我们的工作为构建更公平的DL诊断工具奠定了基础。代码可在以下网址获得：this https URL。 

---
# An improved educational competition optimizer with multi-covariance learning operators for global optimization problems 

**Title (ZH)**: 基于多协方差学习操作者的改进教育竞赛优化算法用于全局优化问题 

**Authors**: Baoqi Zhao, Xiong Yang, Hoileong Lee, Bowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.09552)  

**Abstract**: The educational competition optimizer is a recently introduced metaheuristic algorithm inspired by human behavior, originating from the dynamics of educational competition within society. Nonetheless, ECO faces constraints due to an imbalance between exploitation and exploration, rendering it susceptible to local optima and demonstrating restricted effectiveness in addressing complex optimization problems. To address these limitations, this study presents an enhanced educational competition optimizer (IECO-MCO) utilizing multi-covariance learning operators. In IECO, three distinct covariance learning operators are introduced to improve the performance of ECO. Each operator effectively balances exploitation and exploration while preventing premature convergence of the population. The effectiveness of IECO is assessed through benchmark functions derived from the CEC 2017 and CEC 2022 test suites, and its performance is compared with various basic and improved algorithms across different categories. The results demonstrate that IECO-MCO surpasses the basic ECO and other competing algorithms in convergence speed, stability, and the capability to avoid local optima. Furthermore, statistical analyses, including the Friedman test, Kruskal-Wallis test, and Wilcoxon rank-sum test, are conducted to validate the superiority of IECO-MCO over the compared algorithms. Compared with the basic algorithm (improved algorithm), IECO-MCO achieved an average ranking of 2.213 (2.488) on the CE2017 and CEC2022 test suites. Additionally, the practical applicability of the proposed IECO-MCO algorithm is verified by solving constrained optimization problems. The experimental outcomes demonstrate the superior performance of IECO-MCO in tackling intricate optimization problems, underscoring its robustness and practical effectiveness in real-world scenarios. 

**Abstract (ZH)**: 基于多协方差学习操作的增强教育竞赛优化器（IECO-MCO） 

---
# Improving Video Diffusion Transformer Training by Multi-Feature Fusion and Alignment from Self-Supervised Vision Encoders 

**Title (ZH)**: 基于自我监督视觉编码器的多特征融合与对齐改进视频扩散变换器训练 

**Authors**: Dohun Lee, Hyeonho Jeong, Jiwook Kim, Duygu Ceylan, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.09547)  

**Abstract**: Video diffusion models have advanced rapidly in the recent years as a result of series of architectural innovations (e.g., diffusion transformers) and use of novel training objectives (e.g., flow matching). In contrast, less attention has been paid to improving the feature representation power of such models. In this work, we show that training video diffusion models can benefit from aligning the intermediate features of the video generator with feature representations of pre-trained vision encoders. We propose a new metric and conduct an in-depth analysis of various vision encoders to evaluate their discriminability and temporal consistency, thereby assessing their suitability for video feature alignment. Based on the analysis, we present Align4Gen which provides a novel multi-feature fusion and alignment method integrated into video diffusion model training. We evaluate Align4Gen both for unconditional and class-conditional video generation tasks and show that it results in improved video generation as quantified by various metrics. Full video results are available on our project page: this https URL 

**Abstract (ZH)**: 近年来，由于一系列架构创新（如扩散变压器）和使用新颖的训练目标（如流匹配），视频扩散模型取得了快速进展。相比之下，较少关注如何提升这些模型的特征表示能力。在本工作中，我们展示了将视频生成器的中间特征与预训练的视觉编码器的特征表示进行对齐，可以受益于视频扩散模型的训练。我们提出了一种新的评估指标，并对多种视觉编码器进行了深入分析，评估其可区分性和时序一致性，从而评估其在视频特征对齐中的适用性。基于分析，我们提出了Align4Gen，这是一种集成到视频扩散模型训练中的新型多特征融合与对齐方法。我们分别在无条件和类别条件的视频生成任务中评估了Align4Gen，并通过多种指标证实其可以提高视频生成效果。完整视频结果可在我们的项目页面查看：this https URL 

---
# A modified RIME algorithm with covariance learning and diversity enhancement for numerical optimization 

**Title (ZH)**: 具有协方差学习和多样性增强的修改RIME算法用于数值优化 

**Authors**: Shangqing Shi, Luoxiao Zhang, Yuchen Yin, Xiong Yang, Hoileong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09529)  

**Abstract**: Metaheuristics are widely applied for their ability to provide more efficient solutions. The RIME algorithm is a recently proposed physical-based metaheuristic algorithm with certain advantages. However, it suffers from rapid loss of population diversity during optimization and is prone to fall into local optima, leading to unbalanced exploitation and exploration. To address the shortcomings of RIME, this paper proposes a modified RIME with covariance learning and diversity enhancement (MRIME-CD). The algorithm applies three strategies to improve the optimization capability. First, a covariance learning strategy is introduced in the soft-rime search stage to increase the population diversity and balance the over-exploitation ability of RIME through the bootstrapping effect of dominant populations. Second, in order to moderate the tendency of RIME population to approach the optimal individual in the early search stage, an average bootstrapping strategy is introduced into the hard-rime puncture mechanism, which guides the population search through the weighted position of the dominant populations, thus enhancing the global search ability of RIME in the early stage. Finally, a new stagnation indicator is proposed, and a stochastic covariance learning strategy is used to update the stagnant individuals in the population when the algorithm gets stagnant, thus enhancing the ability to jump out of the local optimal solution. The proposed MRIME-CD algorithm is subjected to a series of validations on the CEC2017 test set, the CEC2022 test set, and the experimental results are analyzed using the Friedman test, the Wilcoxon rank sum test, and the Kruskal Wallis test. The results show that MRIME-CD can effectively improve the performance of basic RIME and has obvious superiorities in terms of solution accuracy, convergence speed and stability. 

**Abstract (ZH)**: 基于协方差学习和多样性增强的改进RIME算法（MRIME-CD） 

---
# Towards Explainable Job Title Matching: Leveraging Semantic Textual Relatedness and Knowledge Graphs 

**Title (ZH)**: 面向可解释的职位标题匹配：利用语义文本相关性和知识图谱 

**Authors**: Vadim Zadykian, Bruno Andrade, Haithem Afli  

**Link**: [PDF](https://arxiv.org/pdf/2509.09522)  

**Abstract**: Semantic Textual Relatedness (STR) captures nuanced relationships between texts that extend beyond superficial lexical similarity. In this study, we investigate STR in the context of job title matching - a key challenge in resume recommendation systems, where overlapping terms are often limited or misleading. We introduce a self-supervised hybrid architecture that combines dense sentence embeddings with domain-specific Knowledge Graphs (KGs) to improve both semantic alignment and explainability. Unlike previous work that evaluated models on aggregate performance, our approach emphasizes data stratification by partitioning the STR score continuum into distinct regions: low, medium, and high semantic relatedness. This stratified evaluation enables a fine-grained analysis of model performance across semantically meaningful subspaces. We evaluate several embedding models, both with and without KG integration via graph neural networks. The results show that fine-tuned SBERT models augmented with KGs produce consistent improvements in the high-STR region, where the RMSE is reduced by 25% over strong baselines. Our findings highlight not only the benefits of combining KGs with text embeddings, but also the importance of regional performance analysis in understanding model behavior. This granular approach reveals strengths and weaknesses hidden by global metrics, and supports more targeted model selection for use in Human Resources (HR) systems and applications where fairness, explainability, and contextual matching are essential. 

**Abstract (ZH)**: 语义文本相关性（STR）捕捉了文本之间的微妙关系，超出了表层词形相似性的范畴。本研究在职业衔标题匹配的背景下探讨STR，这是简历推荐系统中的一个关键挑战，其中重叠的术语往往有限或误导性。我们提出了一种自监督混合架构，结合密集句子嵌入与领域特定的知识图谱（KGs），以提高语义对齐和可解释性。与之前仅评估模型综合性能的工作不同，我们的方法强调数据分层，通过将STR分数连续性划分为不同的区域：低、中、高语义相关性。这种分层评估方法使我们能够对模型在语义上有意义的子空间中进行细粒度的性能分析。我们评估了几种嵌入模型，包括通过图神经网络整合KG的模型。结果显示，与强基线相比，细调的SBERT模型结合KG在高STR区域中产生了一致的改进，RMSE减少了25%。我们的研究不仅突出了结合KG与文本嵌入的优势，还强调了区域性能分析在理解模型行为中的重要性。这种细粒度的方法揭示了全球指标掩盖的优势和不足，并支持更具针对性的模型选择在人力资源（HR）系统和需要公平性、可解释性和上下文匹配的应用中使用。 

---
# Explainable AI for Accelerated Microstructure Imaging: A SHAP-Guided Protocol on the Connectome 2.0 scanner 

**Title (ZH)**: 可解释的人工智能加速微结构成像：基于SHAP的Connectome 2.0扫描仪协议 

**Authors**: Quentin Uhl, Tommaso Pavan, Julianna Gerold, Kwok-Shing Chan, Yohan Jun, Shohei Fujita, Aneri Bhatt, Yixin Ma, Qiaochu Wang, Hong-Hsi Lee, Susie Y. Huang, Berkin Bilgic, Ileana Jelescu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09513)  

**Abstract**: The diffusion MRI Neurite Exchange Imaging model offers a promising framework for probing gray matter microstructure by estimating parameters such as compartment sizes, diffusivities, and inter-compartmental water exchange time. However, existing protocols require long scan times. This study proposes a reduced acquisition scheme for the Connectome 2.0 scanner that preserves model accuracy while substantially shortening scan duration. We developed a data-driven framework using explainable artificial intelligence with a guided recursive feature elimination strategy to identify an optimal 8-feature subset from a 15-feature protocol. The performance of this optimized protocol was validated in vivo and benchmarked against the full acquisition and alternative reduction strategies. Parameter accuracy, preservation of anatomical contrast, and test-retest reproducibility were assessed. The reduced protocol yielded parameter estimates and cortical maps comparable to the full protocol, with low estimation errors in synthetic data and minimal impact on test-retest variability. Compared to theory-driven and heuristic reduction schemes, the optimized protocol demonstrated superior robustness, reducing the deviation in water exchange time estimates by over two-fold. In conclusion, this hybrid optimization framework enables viable imaging of neurite exchange in 14 minutes without loss of parameter fidelity. This approach supports the broader application of exchange-sensitive diffusion magnetic resonance imaging in neuroscience and clinical research, and offers a generalizable method for designing efficient acquisition protocols in biophysical parameter mapping. 

**Abstract (ZH)**: 基于扩散MRI神经突丛交换成像模型提供了一种有前景的框架，用于通过估计隔室大小、扩散系数和隔室间水交换时间等参数来探查灰质微结构。然而，现有的方案需要较长的扫描时间。本研究提出了一种适用于Connectome 2.0扫描器的简化采集方案，该方案在保留模型准确性的同时大幅缩短了扫描时间。我们利用可解释的人工智能和指导递归特征消除策略开发了一个数据驱动的框架，从15个特征协议中选择了最优的8个特征子集。对该优化协议的性能进行了体内验证，并与完整采集方案和替代的简化策略进行了基准测试。评估了参数准确性、解剖对比度的保留和测试-再测试的重复性。简化协议在合成数据中的参数估计误差较低，且对测试-再测试变异性的影响 minimal。与理论驱动和启发式简化方案相比，优化协议显示出了更强的稳健性，水交换时间估计的偏差减少了约两倍。总之，这种混合优化框架能够在14分钟内实现神经突丛交换的有效成像而不损失参数保真度。该方法支持交换敏感的扩散磁共振成像在神经科学和临床研究中的广泛应用，并提供了一种有效采集协议设计的一般化方法，用于生物物理参数映射。 

---
# Incorporating AI Incident Reporting into Telecommunications Law and Policy: Insights from India 

**Title (ZH)**: 将AI事故报告纳入电信法律与政策：来自印度的启示 

**Authors**: Avinash Agarwal, Manisha J. Nene  

**Link**: [PDF](https://arxiv.org/pdf/2509.09508)  

**Abstract**: The integration of artificial intelligence (AI) into telecommunications infrastructure introduces novel risks, such as algorithmic bias and unpredictable system behavior, that fall outside the scope of traditional cybersecurity and data protection frameworks. This paper introduces a precise definition and a detailed typology of telecommunications AI incidents, establishing them as a distinct category of risk that extends beyond conventional cybersecurity and data protection breaches. It argues for their recognition as a distinct regulatory concern. Using India as a case study for jurisdictions that lack a horizontal AI law, the paper analyzes the country's key digital regulations. The analysis reveals that India's existing legal instruments, including the Telecommunications Act, 2023, the CERT-In Rules, and the Digital Personal Data Protection Act, 2023, focus on cybersecurity and data breaches, creating a significant regulatory gap for AI-specific operational incidents, such as performance degradation and algorithmic bias. The paper also examines structural barriers to disclosure and the limitations of existing AI incident repositories. Based on these findings, the paper proposes targeted policy recommendations centered on integrating AI incident reporting into India's existing telecom governance. Key proposals include mandating reporting for high-risk AI failures, designating an existing government body as a nodal agency to manage incident data, and developing standardized reporting frameworks. These recommendations aim to enhance regulatory clarity and strengthen long-term resilience, offering a pragmatic and replicable blueprint for other nations seeking to govern AI risks within their existing sectoral frameworks. 

**Abstract (ZH)**: 人工智能在电信基础设施中的集成引入了新型风险，如算法偏见和不可预测的系统行为，这些风险超出了传统网络安全和数据保护框架的范畴。本文提出了电信人工智能事件的精确定义和详细分类，将其确立为超越传统网络安全和数据保护泄露的独立风险类别，并主张将其作为独立的监管关注点进行考虑。以印度为例，缺乏横向人工智能法律的司法管辖区，分析了该国的关键数字法规。分析显示，印度现有的法律工具，包括2023年电信法、CERT-In规则和2023年数字个人数据保护法，主要关注网络安全和数据泄露，形成了针对特定于人工智能的操作事件，如性能退化和算法偏见的显著监管缺口。本文还探讨了披露结构障碍和现有人工智能事件仓库的限制。基于这些发现，本文提出了针对印度现有电信治理的针对性政策建议，重点是将人工智能事件报告纳入其中。关键建议包括要求报告高风险的人工智能故障、指定一个现有政府机构作为节点机构来管理事件数据，并开发标准化报告框架。这些建议旨在提高监管明晰度并增强长期韧性，为其他国家在现有部门框架内治理人工智能风险提供实用且可复制的蓝图。 

---
# OpenFake: An Open Dataset and Platform Toward Large-Scale Deepfake Detection 

**Title (ZH)**: OpenFake：面向大规模深度假信息检测的开放数据集与平台 

**Authors**: Victor Livernoche, Akshatha Arodi, Andreea Musulan, Zachary Yang, Adam Salvail, Gaétan Marceau Caron, Jean-François Godbout, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2509.09495)  

**Abstract**: Deepfakes, synthetic media created using advanced AI techniques, have intensified the spread of misinformation, particularly in politically sensitive contexts. Existing deepfake detection datasets are often limited, relying on outdated generation methods, low realism, or single-face imagery, restricting the effectiveness for general synthetic image detection. By analyzing social media posts, we identify multiple modalities through which deepfakes propagate misinformation. Furthermore, our human perception study demonstrates that recently developed proprietary models produce synthetic images increasingly indistinguishable from real ones, complicating accurate identification by the general public. Consequently, we present a comprehensive, politically-focused dataset specifically crafted for benchmarking detection against modern generative models. This dataset contains three million real images paired with descriptive captions, which are used for generating 963k corresponding high-quality synthetic images from a mix of proprietary and open-source models. Recognizing the continual evolution of generative techniques, we introduce an innovative crowdsourced adversarial platform, where participants are incentivized to generate and submit challenging synthetic images. This ongoing community-driven initiative ensures that deepfake detection methods remain robust and adaptive, proactively safeguarding public discourse from sophisticated misinformation threats. 

**Abstract (ZH)**: 深度伪造：一种利用先进AI技术生成的合成媒体，在政治敏感背景下加剧了假信息的传播。现有的深度伪造检测数据集往往存在局限性，依赖于过时的生成方法、低保真度或单人图像，限制了其在一般合成图像检测中的有效性。通过对社交媒体帖子的分析，我们发现多种传播假信息的模式。此外，我们的感知研究表明，最近开发的专有模型生成的合成图像越来越难以与真实图像区分开来，使公众难以准确识别。因此，我们提出一个针对政治主题的综合性数据集，用于基准测试针对现代生成模型的检测方法。该数据集包含300万张真实图像配以描述性说明，用于生成源自专有和开源模型混合的963,000张高质量合成图像。鉴于生成技术的持续演变，我们推出了一个创新的众包对抗平台，鼓励参与者生成和提交具有挑战性的合成图像。这一社区驱动的持续性举措确保检测方法保持稳健和适应性，从而积极地保护公共讨论免受复杂虚假信息的威胁。 

---
# Prompt Pirates Need a Map: Stealing Seeds helps Stealing Prompts 

**Title (ZH)**: 海盗需要一张地图：偷取提示词有助于偷取生成种子 

**Authors**: Felix Mächtle, Ashwath Shetty, Jonas Sander, Nils Loose, Sören Pirk, Thomas Eisenbarth  

**Link**: [PDF](https://arxiv.org/pdf/2509.09488)  

**Abstract**: Diffusion models have significantly advanced text-to-image generation, enabling the creation of highly realistic images conditioned on textual prompts and seeds. Given the considerable intellectual and economic value embedded in such prompts, prompt theft poses a critical security and privacy concern. In this paper, we investigate prompt-stealing attacks targeting diffusion models. We reveal that numerical optimization-based prompt recovery methods are fundamentally limited as they do not account for the initial random noise used during image generation. We identify and exploit a noise-generation vulnerability (CWE-339), prevalent in major image-generation frameworks, originating from PyTorch's restriction of seed values to a range of $2^{32}$ when generating the initial random noise on CPUs. Through a large-scale empirical analysis conducted on images shared via the popular platform CivitAI, we demonstrate that approximately 95% of these images' seed values can be effectively brute-forced in 140 minutes per seed using our seed-recovery tool, SeedSnitch. Leveraging the recovered seed, we propose PromptPirate, a genetic algorithm-based optimization method explicitly designed for prompt stealing. PromptPirate surpasses state-of-the-art methods, i.e., PromptStealer, P2HP, and CLIP-Interrogator, achieving an 8-11% improvement in LPIPS similarity. Furthermore, we introduce straightforward and effective countermeasures that render seed stealing, and thus optimization-based prompt stealing, ineffective. We have disclosed our findings responsibly and initiated coordinated mitigation efforts with the developers to address this critical vulnerability. 

**Abstract (ZH)**: 基于扩散模型的提示窃取攻击研究 

---
# Resource-Efficient Glioma Segmentation on Sub-Saharan MRI 

**Title (ZH)**: 资源高效的小glioma分割在撒哈拉以南地区的MRI图像中 

**Authors**: Freedmore Sidume, Oumayma Soula, Joseph Muthui Wacira, YunFei Zhu, Abbas Rabiu Muhammad, Abderrazek Zeraii, Oluwaseun Kalejaye, Hajer Ibrahim, Olfa Gaddour, Brain Halubanza, Dong Zhang, Udunna C Anazodo, Confidence Raymond  

**Link**: [PDF](https://arxiv.org/pdf/2509.09469)  

**Abstract**: Gliomas are the most prevalent type of primary brain tumors, and their accurate segmentation from MRI is critical for diagnosis, treatment planning, and longitudinal monitoring. However, the scarcity of high-quality annotated imaging data in Sub-Saharan Africa (SSA) poses a significant challenge for deploying advanced segmentation models in clinical workflows. This study introduces a robust and computationally efficient deep learning framework tailored for resource-constrained settings. We leveraged a 3D Attention UNet architecture augmented with residual blocks and enhanced through transfer learning from pre-trained weights on the BraTS 2021 dataset. Our model was evaluated on 95 MRI cases from the BraTS-Africa dataset, a benchmark for glioma segmentation in SSA MRI data. Despite the limited data quality and quantity, our approach achieved Dice scores of 0.76 for the Enhancing Tumor (ET), 0.80 for Necrotic and Non-Enhancing Tumor Core (NETC), and 0.85 for Surrounding Non-Functional Hemisphere (SNFH). These results demonstrate the generalizability of the proposed model and its potential to support clinical decision making in low-resource settings. The compact architecture, approximately 90 MB, and sub-minute per-volume inference time on consumer-grade hardware further underscore its practicality for deployment in SSA health systems. This work contributes toward closing the gap in equitable AI for global health by empowering underserved regions with high-performing and accessible medical imaging solutions. 

**Abstract (ZH)**: Gliomas在撒哈拉以南非洲地区的主要脑肿瘤类型，其从MRI图像中的准确分割对于诊断、治疗规划和纵向监测至关重要。然而，撒哈拉以南非洲地区高质量标注影像数据的稀缺性给高级分割模型在临床工作流程中的部署带来了重大挑战。本研究介绍了一种针对资源限制性设置优化的稳健且计算高效的深度学习框架。我们利用带有残差块的3D注意力UNet架构，并通过在BraTS 2021数据集上预训练权重进行迁移学习加以增强。该模型在BraTS-Africa数据集的95例MRI病例上进行了评估，这是一个针对撒哈拉以南非洲脑胶质瘤分割的基准数据集。尽管数据质量和服务量有限，但我们的方法在增强肿瘤(ET)、坏死和非增强肿瘤核心(NETC)以及周围非功能半球(SNFH)上的Dice分数分别为0.76、0.80和0.85。这些结果证明了所提出模型的泛化能力和在资源有限环境中支持临床决策的潜力。紧凑的架构，约90 MB，以及消费级硬件上每卷秒级的推理时间进一步突显了其在撒哈拉以南非洲卫生系统中的实用性。这项工作朝着实现全球卫生中公平的人工智能差距做出了贡献，为服务不足的地区提供了高性能且易于访问的医学影像解决方案。 

---
# ENSI: Efficient Non-Interactive Secure Inference for Large Language Models 

**Title (ZH)**: ENSI: 高效非交互式安全推理大型语言模型 

**Authors**: Zhiyu He, Maojiang Wang, Xinwen Gao, Yuchuan Luo, Lin Liu, Shaojing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09424)  

**Abstract**: Secure inference enables privacy-preserving machine learning by leveraging cryptographic protocols that support computations on sensitive user data without exposing it. However, integrating cryptographic protocols with large language models (LLMs) presents significant challenges, as the inherent complexity of these protocols, together with LLMs' massive parameter scale and sophisticated architectures, severely limits practical usability. In this work, we propose ENSI, a novel non-interactive secure inference framework for LLMs, based on the principle of co-designing the cryptographic protocols and LLM architecture. ENSI employs an optimized encoding strategy that seamlessly integrates CKKS scheme with a lightweight LLM variant, BitNet, significantly reducing the computational complexity of encrypted matrix multiplications. In response to the prohibitive computational demands of softmax under homomorphic encryption (HE), we pioneer the integration of the sigmoid attention mechanism with HE as a seamless, retraining-free alternative. Furthermore, by embedding the Bootstrapping operation within the RMSNorm process, we efficiently refresh ciphertexts while markedly decreasing the frequency of costly bootstrapping invocations. Experimental evaluations demonstrate that ENSI achieves approximately an 8x acceleration in matrix multiplications and a 2.6x speedup in softmax inference on CPU compared to state-of-the-art method, with the proportion of bootstrapping is reduced to just 1%. 

**Abstract (ZH)**: 安全推理使加密码协议能够在不暴露敏感用户数据的情况下对这些数据进行计算，从而实现隐私保护的机器学习。然而，将加密码协议集成到大型语言模型（LLMs）中面临重大挑战，因为这些协议的固有复杂性和LLMs的巨大参数规模及复杂的架构严重限制了其实用性。在这项工作中，我们提出了一种名为ENSI的新型非交互式安全推理框架，该框架基于加密码协议和LLM架构共同设计的原则。ENSI采用了优化的编码策略，无缝集成CKKS方案与轻量级LLM变体BitNet，显著降低加密矩阵乘法的计算复杂度。针对同态加密（HE）下softmax的巨额计算需求，我们首次提出了将Sigmoid注意力机制与HE无缝集成作为无重新训练的替代方案。此外，通过将Bootstrapping操作嵌入到RMSNorm过程中，我们高效地刷新密文，并显著减少了昂贵的Bootstrapping调用频率。实验评估表明，与现有最佳方法相比，ENSI在CPU上实现了大约8倍的矩阵乘法加速和2.6倍的softmax推理加速，Bootstrapping的比例降低至仅1%。 

---
# We're Still Doing It (All) Wrong: Recommender Systems, Fifteen Years Later 

**Title (ZH)**: 我们仍然做错了（所有事情）：十五年后，推荐系统仍存在问题 

**Authors**: Alan Said, Maria Soledad Pera, Michael D. Ekstrand  

**Link**: [PDF](https://arxiv.org/pdf/2509.09414)  

**Abstract**: In 2011, Xavier Amatriain sounded the alarm: recommender systems research was "doing it all wrong" [1]. His critique, rooted in statistical misinterpretation and methodological shortcuts, remains as relevant today as it was then. But rather than correcting course, we added new layers of sophistication on top of the same broken foundations. This paper revisits Amatriain's diagnosis and argues that many of the conceptual, epistemological, and infrastructural failures he identified still persist, in more subtle or systemic forms. Drawing on recent work in reproducibility, evaluation methodology, environmental impact, and participatory design, we showcase how the field's accelerating complexity has outpaced its introspection. We highlight ongoing community-led initiatives that attempt to shift the paradigm, including workshops, evaluation frameworks, and calls for value-sensitive and participatory research. At the same time, we contend that meaningful change will require not only new metrics or better tooling, but a fundamental reframing of what recommender systems research is for, who it serves, and how knowledge is produced and validated. Our call is not just for technical reform, but for a recommender systems research agenda grounded in epistemic humility, human impact, and sustainable practice. 

**Abstract (ZH)**: 2011年，Xavier Amatriain 发出警告：推荐系统研究“走错了方向”[1]。虽然他的批评基于统计误读和方法论捷径，至今依然 relevant，但研究领域并未纠正航线，反而在旧有缺陷之上增加了新的复杂层。本文重新审视 Amatriain 的诊断，认为他识别出的概念性、知识论性和基础设施性失败仍然以更微妙或系统化的方式存在。借助最近在可再现性、评估方法、环境影响及参与设计等方面的工作，我们展示了研究领域的加速复杂化已经超越了其自我反省。我们强调了社区主导的倡议，这些倡议试图改变范式，包括研讨会、评估框架及呼吁价值敏感和参与式研究。同时，我们认为有意义的变革不仅需要新的指标或更好的工具，还需要根本性地重塑推荐系统研究的目的、服务对象及知识的生成与验证方式。我们呼吁的不仅仅是技术改革，而是基于知识谦逊、人类影响和可持续实践的推荐系统研究议程。 

---
# LLMs Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations 

**Title (ZH)**: LLMs 不知道自己的决策边界：自我生成的反事实解释不可靠 

**Authors**: Harry Mayne, Ryan Othniel Kearns, Yushi Yang, Andrew M. Bean, Eoin Delaney, Chris Russell, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09396)  

**Abstract**: To collaborate effectively with humans, language models must be able to explain their decisions in natural language. We study a specific type of self-explanation: self-generated counterfactual explanations (SCEs), where a model explains its prediction by modifying the input such that it would have predicted a different outcome. We evaluate whether LLMs can produce SCEs that are valid, achieving the intended outcome, and minimal, modifying the input no more than necessary. When asked to generate counterfactuals, we find that LLMs typically produce SCEs that are valid, but far from minimal, offering little insight into their decision-making behaviour. Worryingly, when asked to generate minimal counterfactuals, LLMs typically make excessively small edits that fail to change predictions. The observed validity-minimality trade-off is consistent across several LLMs, datasets, and evaluation settings. Our findings suggest that SCEs are, at best, an ineffective explainability tool and, at worst, can provide misleading insights into model behaviour. Proposals to deploy LLMs in high-stakes settings must consider the impact of unreliable self-explanations on downstream decision-making. Our code is available at this https URL. 

**Abstract (ZH)**: 语言模型必须能够用自然语言解释其决策以有效协作。我们研究了一种特定类型的自解释：自生成反事实解释（SCEs），模型通过修改输入来解释其预测，使得它会预测不同的结果。我们评估了LLMs能否生成有效的、实现意图结果且最小化的SCEs。当我们要求生成反事实时，发现LLMs通常能产生有效的SCEs，但远不 Minimal，提供的决策过程洞察有限。令人担忧的是，当我们要求生成最小化的反事实时，LLMs通常做出过小的编辑，无法改变预测。观察到的有效性-最小性权衡贯穿于多个LLMs、数据集和评估设置中。我们的研究结果表明，SCEs至多是无效的解释工具，最坏情况下会提供误导性的模型行为洞察。在高风险场景下部署LLMs时，必须考虑不可靠自解释对下游决策的影响。我们的代码可在以下链接获取。 

---
# MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization 

**Title (ZH)**: MetaLLMix : 一种基于XAI辅助的LLM元学习方法用于超参数优化 

**Authors**: Mohammed Tiouti, Mohamed Bal-Ghaoui  

**Link**: [PDF](https://arxiv.org/pdf/2509.09387)  

**Abstract**: Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines. 

**Abstract (ZH)**: 基于元学习、可解释AI和高效大语言模型推理的零样本超参数优化框架MetaLLMiX 

---
# Robust Non-Linear Correlations via Polynomial Regression 

**Title (ZH)**: 稳健的非线性相关性通过多项式回归 

**Authors**: Luca Giuliani, Michele Lombardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09380)  

**Abstract**: The Hirschfeld-Gebelein-Rényi (HGR) correlation coefficient is an extension of Pearson's correlation that is not limited to linear correlations, with potential applications in algorithmic fairness, scientific analysis, and causal discovery. Recently, novel algorithms to estimate HGR in a differentiable manner have been proposed to facilitate its use as a loss regularizer in constrained machine learning applications. However, the inherent uncomputability of HGR requires a bias-variance trade-off, which can possibly compromise the robustness of the proposed methods, hence raising technical concerns if applied in real-world scenarios. We introduce a novel computational approach for HGR that relies on user-configurable polynomial kernels, offering greater robustness compared to previous methods and featuring a faster yet almost equally effective restriction. Our approach provides significant advantages in terms of robustness and determinism, making it a more reliable option for real-world applications. Moreover, we present a brief experimental analysis to validate the applicability of our approach within a constrained machine learning framework, showing that its computation yields an insightful subgradient that can serve as a loss regularizer. 

**Abstract (ZH)**: Hirschfeld-Gebelein-Rényi (HGR) 相关系数是一种扩展的皮尔逊相关系数，适用于非线性相关分析，具有在算法公平性、科学研究和因果发现等方面的应用潜力。最近，提出了以可微方式估计 HGR 的新型算法，以促进其作为约束机器学习应用中损失正则化项的使用。然而，HGR 内在的不可计算性要求在偏倚与方差之间进行权衡，这可能会损害提出方法的鲁棒性，从而在实际应用场景中引发技术关切。我们引入了一种基于可配置多项式核的新计算方法，与以往方法相比提供了更高的鲁棒性，并具备更快且几乎同样有效的限制效果。我们的方法在鲁棒性和确定性方面具有显著优势，使其成为更可靠的现实应用选择。此外，我们呈现了简要的实验分析，验证了在约束机器学习框架中应用该方法的有效性，表明其计算可以产生具有洞察力的次梯度，可用作损失正则化项。 

---
# Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles 

**Title (ZH)**: 基于外部观察技术的驾驶员行为分类方法研究 

**Authors**: Ian Nell, Shane Gilroy  

**Link**: [PDF](https://arxiv.org/pdf/2509.09349)  

**Abstract**: Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behavior classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviors such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioral analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions. 

**Abstract (ZH)**: 基于外部观察的新型驾驶行为分类系统：检测分心和受酒精及其他物质影响的指标 

---
# MoSE: Unveiling Structural Patterns in Graphs via Mixture of Subgraph Experts 

**Title (ZH)**: MoSE: 通过子图专家混合体揭示图的结构模式 

**Authors**: Junda Ye, Zhongbao Zhang, Li Sun, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.09337)  

**Abstract**: While graph neural networks (GNNs) have achieved great success in learning from graph-structured data, their reliance on local, pairwise message passing restricts their ability to capture complex, high-order subgraph patterns. leading to insufficient structural expressiveness. Recent efforts have attempted to enhance structural expressiveness by integrating random walk kernels into GNNs. However, these methods are inherently designed for graph-level tasks, which limits their applicability to other downstream tasks such as node classification. Moreover, their fixed kernel configurations hinder the model's flexibility in capturing diverse subgraph structures. To address these limitations, this paper proposes a novel Mixture of Subgraph Experts (MoSE) framework for flexible and expressive subgraph-based representation learning across diverse graph tasks. Specifically, MoSE extracts informative subgraphs via anonymous walks and dynamically routes them to specialized experts based on structural semantics, enabling the model to capture diverse subgraph patterns with improved flexibility and interpretability. We further provide a theoretical analysis of MoSE's expressivity within the Subgraph Weisfeiler-Lehman (SWL) Test, proving that it is more powerful than SWL. Extensive experiments, together with visualizations of learned subgraph experts, demonstrate that MoSE not only outperforms competitive baselines but also provides interpretable insights into structural patterns learned by the model. 

**Abstract (ZH)**: 具有灵活表达性的子图专家混合框架：跨多种图任务的子图表示学习 

---
# OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning 

**Title (ZH)**: OmniEVA：基于任务自适应三维接地和体态意识推理的通用 embodied 计划器 

**Authors**: Yuecheng Liu, Dafeng Chi, Shiguang Wu, Zhanguang Zhang, Yuzheng Zhuang, Bowen Yang, He Zhu, Lingfeng Zhang, Pengwei Xie, David Gamaliel Arcos Bravo, Yingxue Zhang, Jianye Hao, Xingyue Quan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09332)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically this http URL address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: this https URL 

**Abstract (ZH)**: 最近多模态大型语言模型的进展为具身智能开辟了新机会，使其能够实现多模态理解、推理和交互，以及持续的空间决策。然而，当前基于多模态大型语言模型的具身系统面临两大关键限制：几何适应性缺口和具身约束缺口。为了解决这些问题，我们介绍了OmniEVA——一种具身通用规划器，通过两项关键创新实现高级具身推理和任务规划：（1）任务自适应3D关联机制，该机制引入门控路由器，根据上下文需求显式选择性调节3D融合，实现多种具身任务的上下文感知3D关联。（2）具身感知推理框架，该框架将任务目标和具身约束同时纳入推理循环，从而产生既目标导向又可执行的规划决策。广泛的经验结果表明，OmniEVA不仅实现了最先进的综合具身推理性能，还广泛展现出强大的规划能力。一系列提出的具身基准评估，包括基本和复合任务，证实了其稳健且多功能的规划能力。项目页面：[此链接地址]。 

---
# Can Multimodal LLMs See Materials Clearly? A Multimodal Benchmark on Materials Characterization 

**Title (ZH)**: 多模态LLM能清晰地“看见”材料吗？一种用于材料表征的多模态基准测试 

**Authors**: Zhengzhao Lai, Youbin Zheng, Zhenyang Cai, Haonan Lyu, Jinpu Yang, Hongqing Liang, Yan Hu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09307)  

**Abstract**: Materials characterization is fundamental to acquiring materials information, revealing the processing-microstructure-property relationships that guide material design and optimization. While multimodal large language models (MLLMs) have recently shown promise in generative and predictive tasks within materials science, their capacity to understand real-world characterization imaging data remains underexplored. To bridge this gap, we present MatCha, the first benchmark for materials characterization image understanding, comprising 1,500 questions that demand expert-level domain expertise. MatCha encompasses four key stages of materials research comprising 21 distinct tasks, each designed to reflect authentic challenges faced by materials scientists. Our evaluation of state-of-the-art MLLMs on MatCha reveals a significant performance gap compared to human experts. These models exhibit degradation when addressing questions requiring higher-level expertise and sophisticated visual perception. Simple few-shot and chain-of-thought prompting struggle to alleviate these limitations. These findings highlight that existing MLLMs still exhibit limited adaptability to real-world materials characterization scenarios. We hope MatCha will facilitate future research in areas such as new material discovery and autonomous scientific agents. MatCha is available at this https URL. 

**Abstract (ZH)**: 材料表征是获取材料信息的基础，揭示了加工-微观结构-性能关系，指导材料设计与优化。虽然多模态大语言模型（MLLMs）在材料科学领域的生成性和预测性任务中已显示出潜力，但其理解和处理现实世界的表征影像数据的能力仍然未被充分探索。为弥补这一缺口，我们提出了MatCha——首个材料表征影像理解基准，包含1500个要求专家级领域知识的问题。MatCha涵盖了材料研究的四个关键阶段，包含21项不同的任务，旨在反映材料科学家所面临的实际挑战。我们在MatCha上对最新一代MLLMs的评估结果显示，这些模型在性能上与人类专家之间存在显著差距。这些模型在处理需要较高水平专业知识和复杂视觉感知的问题时表现出能力下降。即使是简单的少量示例提示和链式思考提示也难以缓解这些限制。这些发现表明，现有的MLLMs在适应现实世界的材料表征场景方面仍然表现有限。我们希望MatCha能够促进未来在新材料发现和自主科学代理等领域的研究。MatCha可在以下网址获取：this https URL。 

---
# Modality-Agnostic Input Channels Enable Segmentation of Brain lesions in Multimodal MRI with Sequences Unavailable During Training 

**Title (ZH)**: 模态无关的输入通道 enables 多模态MRI中不可用训练序列情况下脑病变的分割 

**Authors**: Anthony P. Addison, Felix Wagner, Wentian Xu, Natalie Voets, Konstantinos Kamnitsas  

**Link**: [PDF](https://arxiv.org/pdf/2509.09290)  

**Abstract**: Segmentation models are important tools for the detection and analysis of lesions in brain MRI. Depending on the type of brain pathology that is imaged, MRI scanners can acquire multiple, different image modalities (contrasts). Most segmentation models for multimodal brain MRI are restricted to fixed modalities and cannot effectively process new ones at inference. Some models generalize to unseen modalities but may lose discriminative modality-specific information. This work aims to develop a model that can perform inference on data that contain image modalities unseen during training, previously seen modalities, and heterogeneous combinations of both, thus allowing a user to utilize any available imaging modalities. We demonstrate this is possible with a simple, thus practical alteration to the U-net architecture, by integrating a modality-agnostic input channel or pathway, alongside modality-specific input channels. To train this modality-agnostic component, we develop an image augmentation scheme that synthesizes artificial MRI modalities. Augmentations differentially alter the appearance of pathological and healthy brain tissue to create artificial contrasts between them while maintaining realistic anatomical integrity. We evaluate the method using 8 MRI databases that include 5 types of pathologies (stroke, tumours, traumatic brain injury, multiple sclerosis and white matter hyperintensities) and 8 modalities (T1, T1+contrast, T2, PD, SWI, DWI, ADC and FLAIR). The results demonstrate that the approach preserves the ability to effectively process MRI modalities encountered during training, while being able to process new, unseen modalities to improve its segmentation. Project code: this https URL 

**Abstract (ZH)**: 多模态脑MRI中未见模态的分割模型研究：结合模态无关输入通道的简单U-net架构改进 

---
# Adaptive Knowledge Distillation using a Device-Aware Teacher for Low-Complexity Acoustic Scene Classification 

**Title (ZH)**: 基于设备感知教师的自适应知识蒸馏在低复杂度声 scene 分类中的应用 

**Authors**: Seung Gyu Jeong, Seong Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09262)  

**Abstract**: In this technical report, we describe our submission for Task 1, Low-Complexity Device-Robust Acoustic Scene Classification, of the DCASE 2025 Challenge. Our work tackles the dual challenges of strict complexity constraints and robust generalization to both seen and unseen devices, while also leveraging the new rule allowing the use of device labels at test time. Our proposed system is based on a knowledge distillation framework where an efficient CP-MobileNet student learns from a compact, specialized two-teacher ensemble. This ensemble combines a baseline PaSST teacher, trained with standard cross-entropy, and a 'generalization expert' teacher. This expert is trained using our novel Device-Aware Feature Alignment (DAFA) loss, adapted from prior work, which explicitly structures the feature space for device robustness. To capitalize on the availability of test-time device labels, the distilled student model then undergoes a final device-specific fine-tuning stage. Our proposed system achieves a final accuracy of 57.93\% on the development set, demonstrating a significant improvement over the official baseline, particularly on unseen devices. 

**Abstract (ZH)**: 本技术报告描述了我们参加2025 DCASE挑战任务1——低复杂度设备鲁棒声场景分类的提交内容。我们的工作解决了严格复杂度约束和在已见和未见设备上鲁棒泛化的双重挑战，并利用了新规则，允许在测试时使用设备标签。我们提出了一种基于知识蒸馏框架的系统，其中高效的CP-MobileNet学生从一个紧凑的、专门的双师集成中学习。该集成结合了一个用标准交叉熵训练的基本PaSST教师和一个“泛化专家”教师。该专家通过我们提出的设备感知特征对齐（DAFA）损失进行训练，这是一种从先前工作改编而来的损失函数，可以明确地结构化特征空间以实现设备鲁棒性。为了利用测试时设备标签的可用性，蒸馏后的学生模型随后进行最终的设备特定微调。我们提出的系统在开发集上取得了57.93%的最终准确率，相比官方基线，特别是在未见设备上，显示出显著的改进。 

---
# CoAtNeXt:An Attention-Enhanced ConvNeXtV2-Transformer Hybrid Model for Gastric Tissue Classification 

**Title (ZH)**: CoAtNeXt：一种增强注意力的ConvNeXtV2-Transformer混合模型用于胃组织分类 

**Authors**: Mustafa Yurdakul, Sakir Tasdemir  

**Link**: [PDF](https://arxiv.org/pdf/2509.09242)  

**Abstract**: Background and objective Early diagnosis of gastric diseases is crucial to prevent fatal outcomes. Although histopathologic examination remains the diagnostic gold standard, it is performed entirely manually, making evaluations labor-intensive and prone to variability among pathologists. Critical findings may be missed, and lack of standard procedures reduces consistency. These limitations highlight the need for automated, reliable, and efficient methods for gastric tissue analysis. Methods In this study, a novel hybrid model named CoAtNeXt was proposed for the classification of gastric tissue images. The model is built upon the CoAtNet architecture by replacing its MBConv layers with enhanced ConvNeXtV2 blocks. Additionally, the Convolutional Block Attention Module (CBAM) is integrated to improve local feature extraction through channel and spatial attention mechanisms. The architecture was scaled to achieve a balance between computational efficiency and classification performance. CoAtNeXt was evaluated on two publicly available datasets, HMU-GC-HE-30K for eight-class classification and GasHisSDB for binary classification, and was compared against 10 Convolutional Neural Networks (CNNs) and ten Vision Transformer (ViT) models. Results CoAtNeXt achieved 96.47% accuracy, 96.60% precision, 96.47% recall, 96.45% F1 score, and 99.89% AUC on HMU-GC-HE-30K. On GasHisSDB, it reached 98.29% accuracy, 98.07% precision, 98.41% recall, 98.23% F1 score, and 99.90% AUC. It outperformed all CNN and ViT models tested and surpassed previous studies in the literature. Conclusion Experimental results show that CoAtNeXt is a robust architecture for histopathological classification of gastric tissue images, providing performance on binary and multiclass. Its highlights its potential to assist pathologists by enhancing diagnostic accuracy and reducing workload. 

**Abstract (ZH)**: 背景与目的 早期诊断胃病对于预防致命后果至关重要。尽管组织病理学检查仍然是诊断的金标准，但其完全靠手动操作，使评估劳动密集型且易于病理学家之间出现变异。关键发现可能会被遗漏，缺乏标准化流程降低了一致性。这些限制突出了需要自动、可靠且高效的胃组织分析方法。方法 本文提出了一种新的混合模型CoAtNeXt，用于胃组织图像分类。该模型基于CoAtNet架构，用增强的ConvNeXtV2块替换其MBConv层，并集成了卷积块注意力模块（CBAM）以通过通道和空间注意力机制提高局部特征提取效果。架构通过权衡计算效率和分类性能进行了扩展。CoAtNeXt在两个公开可用的数据集HMU-GC-HE-30K（用于八类分类）和GasHisSDB（用于二分类）上进行了评估，并与十个卷积神经网络（CNN）和十个视觉变换器（ViT）模型进行了比较。结果 在HMU-GC-HE-30K数据集上，CoAtNeXt实现了96.47%的准确率、96.60%的精确率、96.47%的召回率、96.45%的F1分数和99.89%的AUC。在GasHisSDB数据集上，其准确率为98.29%、精确率为98.07%、召回率为98.41%、F1分数为98.23%和AUC为99.90%。CoAtNeXt在所有测试的CNN和ViT模型中表现最佳，并超越了文献中的先前研究。结论 实验结果表明，CoAtNeXt是一种稳健的架构，适用于胃组织图像的组织病理学分类，提供了二分类和多分类的性能。其潜在能力在于通过提高诊断准确性和减轻工作负担来辅助病理学家。 

---
# Virtual staining for 3D X-ray histology of bone implants 

**Title (ZH)**: 骨植入物的3D X射线组织学虚拟染色 

**Authors**: Sarah C. Irvine, Christian Lucas, Diana Krüger, Bianca Guedert, Julian Moosmann, Berit Zeller-Plumhoff  

**Link**: [PDF](https://arxiv.org/pdf/2509.09235)  

**Abstract**: Three-dimensional X-ray histology techniques offer a non-invasive alternative to conventional 2D histology, enabling volumetric imaging of biological tissues without the need for physical sectioning or chemical staining. However, the inherent greyscale image contrast of X-ray tomography limits its biochemical specificity compared to traditional histological stains. Within digital pathology, deep learning-based virtual staining has demonstrated utility in simulating stained appearances from label-free optical images. In this study, we extend virtual staining to the X-ray domain by applying cross-modality image translation to generate artificially stained slices from synchrotron-radiation-based micro-CT scans. Using over 50 co-registered image pairs of micro-CT and toluidine blue-stained histology from bone-implant samples, we trained a modified CycleGAN network tailored for limited paired data. Whole slide histology images were downsampled to match the voxel size of the CT data, with on-the-fly data augmentation for patch-based training. The model incorporates pixelwise supervision and greyscale consistency terms, producing histologically realistic colour outputs while preserving high-resolution structural detail. Our method outperformed Pix2Pix and standard CycleGAN baselines across SSIM, PSNR, and LPIPS metrics. Once trained, the model can be applied to full CT volumes to generate virtually stained 3D datasets, enhancing interpretability without additional sample preparation. While features such as new bone formation were able to be reproduced, some variability in the depiction of implant degradation layers highlights the need for further training data and refinement. This work introduces virtual staining to 3D X-ray imaging and offers a scalable route for chemically informative, label-free tissue characterisation in biomedical research. 

**Abstract (ZH)**: 三维X射线显微断层扫描技术提供了一种与传统2D组织学相比无创的替代方案，无需物理切片或化学染色即可实现生物组织的体视成像。然而，X射线断层成像固有的灰度图像对比度使其在生物化学特异性方面逊于传统组织学染色。在数字病理学中，基于深度学习的虚拟染色已被证明能够从无标记光学图像中模拟染色外观。在本研究中，我们通过应用跨模态图像翻译将虚拟染色扩展到X射线领域，从同步辐射微CT扫描中生成人工染色切片。利用来自骨植入物样本的50多对共注册微CT和阿拉伯糖蓝染色组织学图像对，我们训练了一个针对有限配对数据修改后的CycleGAN网络。整张切片组织学图像被下采样以匹配CT数据的体素大小，并在基于补丁的训练中进行实时数据增强。该模型包含逐像素监督和平面灰度一致性项，产生具有组织学真实感的彩色输出，同时保留高分辨率的结构细节。我们的方法在SSIM、PSNR和LPIPS指标上优于Pix2Pix和标准CycleGAN基线。一旦训练完成，该模型可以应用于整个CT体积，生成虚拟染色的3D数据集，从而提高可解释性而无需额外的样品准备。虽然能够再现新的骨形成特征，但植入物降解层的某些差异显示了需要进一步训练数据和细化的必要性。本研究将虚拟染色引入三维X射线成像，并提供了一条在生物医药研究中实现化学信息性无标记组织表征的可扩展途径。 

---
# Vejde: A Framework for Inductive Deep Reinforcement Learning Based on Factor Graph Color Refinement 

**Title (ZH)**: Vejde：基于因子图颜色细分的归纳深度强化学习框架 

**Authors**: Jakob Nyberg, Pontus Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2509.09219)  

**Abstract**: We present and evaluate Vejde; a framework which combines data abstraction, graph neural networks and reinforcement learning to produce inductive policy functions for decision problems with richly structured states, such as object classes and relations. MDP states are represented as data bases of facts about entities, and Vejde converts each state to a bipartite graph, which is mapped to latent states through neural message passing. The factored representation of both states and actions allows Vejde agents to handle problems of varying size and structure. We tested Vejde agents on eight problem domains defined in RDDL, with ten problem instances each, where policies were trained using both supervised and reinforcement learning. To test policy generalization, we separate problem instances in two sets, one for training and the other solely for testing. Test results on unseen instances for the Vejde agents were compared to MLP agents trained on each problem instance, as well as the online planning algorithm Prost. Our results show that Vejde policies in average generalize to the test instances without a significant loss in score. Additionally, the inductive agents received scores on unseen test instances that on average were close to the instance-specific MLP agents. 

**Abstract (ZH)**: Vejde：一种结合数据抽象、图神经网络和强化学习的框架，用于生成决策问题中的归纳策略函数 

---
# Incentivizing Safer Actions in Policy Optimization for Constrained Reinforcement Learning 

**Title (ZH)**: 受约束强化学习中促进更安全行为的策略优化激励方法 

**Authors**: Somnath Hazra, Pallab Dasgupta, Soumyajit Dey  

**Link**: [PDF](https://arxiv.org/pdf/2509.09208)  

**Abstract**: Constrained Reinforcement Learning (RL) aims to maximize the return while adhering to predefined constraint limits, which represent domain-specific safety requirements. In continuous control settings, where learning agents govern system actions, balancing the trade-off between reward maximization and constraint satisfaction remains a significant challenge. Policy optimization methods often exhibit instability near constraint boundaries, resulting in suboptimal training performance. To address this issue, we introduce a novel approach that integrates an adaptive incentive mechanism in addition to the reward structure to stay within the constraint bound before approaching the constraint boundary. Building on this insight, we propose Incrementally Penalized Proximal Policy Optimization (IP3O), a practical algorithm that enforces a progressively increasing penalty to stabilize training dynamics. Through empirical evaluation on benchmark environments, we demonstrate the efficacy of IP3O compared to the performance of state-of-the-art Safe RL algorithms. Furthermore, we provide theoretical guarantees by deriving a bound on the worst-case error of the optimality achieved by our algorithm. 

**Abstract (ZH)**: 受约束的 reinforcement learning (RL) 目的是在遵守预定义约束限制的同时最大化回报，这些约束代表了特定领域的安全要求。在连续控制设置中，当学习智能体管理系统行动时，要在回报最大化和约束满足之间取得平衡仍然是一个重大挑战。策略优化方法往往在接近约束边界时表现出不稳定性，导致训练性能不佳。为解决这一问题，我们提出了一种新的方法，该方法除了优化奖励结构外，还集成了一个自适应激励机制，以在接近约束边界之前保持在约束范围内。基于这一洞察，我们提出了增量惩罚近端策略优化（IP3O），这是一种能够通过逐步增加惩罚来稳定训练动力学的实用算法。通过在基准环境中进行实证评估，我们展示了与最先进的安全 RL 算法相比，IP3O 的有效性。此外，我们通过推导出我们的算法所实现最优性最坏情况误差的上界，提供了理论保证。 

---
# Bona fide Cross Testing Reveals Weak Spot in Audio Deepfake Detection Systems 

**Title (ZH)**: 真诚的跨测试揭示了音频深度假音检测系统的薄弱环节 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Zhen Qiu, Chi Hung Chi, Kwok Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2509.09204)  

**Abstract**: Audio deepfake detection (ADD) models are commonly evaluated using datasets that combine multiple synthesizers, with performance reported as a single Equal Error Rate (EER). However, this approach disproportionately weights synthesizers with more samples, underrepresenting others and reducing the overall reliability of EER. Additionally, most ADD datasets lack diversity in bona fide speech, often featuring a single environment and speech style (e.g., clean read speech), limiting their ability to simulate real-world conditions. To address these challenges, we propose bona fide cross-testing, a novel evaluation framework that incorporates diverse bona fide datasets and aggregates EERs for more balanced assessments. Our approach improves robustness and interpretability compared to traditional evaluation methods. We benchmark over 150 synthesizers across nine bona fide speech types and release a new dataset to facilitate further research at this https URL. 

**Abstract (ZH)**: Authentic语音跨测试：一种新的评估框架 

---
# Improving Synthetic Data Training for Contextual Biasing Models with a Keyword-Aware Cost Function 

**Title (ZH)**: 基于关键字感知成本函数提升合成数据训练的上下文偏见模型性能 

**Authors**: Chin Yuen Kwok, Jia Qi Yip, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09197)  

**Abstract**: Rare word recognition can be improved by adapting ASR models to synthetic data that includes these words. Further improvements can be achieved through contextual biasing, which trains and adds a biasing module into the model architecture to prioritize rare words. While training the module on synthetic rare word data is more effective than using non-rare-word data, it can lead to overfitting due to artifacts in the synthetic audio. To address this, we enhance the TCPGen-based contextual biasing approach and propose a keyword-aware loss function that additionally focuses on biased words when training biasing modules. This loss includes a masked cross-entropy term for biased word prediction and a binary classification term for detecting biased word positions. These two terms complementarily support the decoding of biased words during inference. By adapting Whisper to 10 hours of synthetic data, our method reduced the word error rate on the NSC Part 2 test set from 29.71% to 11.81%. 

**Abstract (ZH)**: 稀有词识别可以通过适应包含这些词汇的合成数据来提高。进一步的改进可以通过上下文偏差实现，即在模型架构中训练并添加一个偏差模块以优先考虑稀有词。虽然在合成稀有词数据上训练模块比使用非稀有词数据更有效，但可能会由于合成音频中的伪影导致过拟合。为解决这一问题，我们增强了基于TCPGen的上下文偏差方法，并提出了一种关键词感知的损失函数，该函数在训练偏差模块时还重点关注偏差词汇。该损失函数包括一个用于预测偏差词汇的掩码交叉熵项和一个用于检测偏差词汇位置的二元分类项。这两个项互补地支持推断过程中偏差词汇的解码。通过将Whisper适应10小时的合成数据，我们的方法将NSC Part 2测试集上的词错误率从29.71%降低到11.81%。 

---
# Efficient Trie-based Biasing using K-step Prediction for Rare Word Recognition 

**Title (ZH)**: 基于 Trie 结构的 K 步预测高效偏置算法及其在罕见词识别中的应用 

**Authors**: Chin Yuen Kwok, Jia Qi yip  

**Link**: [PDF](https://arxiv.org/pdf/2509.09196)  

**Abstract**: Contextual biasing improves rare word recognition of ASR models by prioritizing the output of rare words during decoding. A common approach is Trie-based biasing, which gives "bonus scores" to partial hypothesis (e.g. "Bon") that may lead to the generation of the rare word (e.g. "Bonham"). If the full word ("Bonham") isn't ultimately recognized, the system revokes those earlier bonuses. This revocation is limited to beam search and is computationally expensive, particularly for models with large decoders. To overcome these limitations, we propose adapting ASR models to look ahead and predict multiple steps at once. This avoids the revocation step entirely by better estimating whether a partial hypothesis will lead to the generation of the full rare word. By fine-tuning Whisper with only 10 hours of synthetic data, our method reduces the word error rate on the NSC Part 2 test set from 30.86% to 12.19%. 

**Abstract (ZH)**: 上下文偏差增强ASR模型对稀有词的识别能力通过在解码过程中优先处理稀有词的输出。一种常见方法是基于Trie的偏差方法，该方法会给可能导致生成稀有词（例如“Bonham”）的部分假设（例如“Bon”）加分。如果最终未能识别出完整的词（例如“Bonham”），系统会撤销这些早期加分。这种撤销仅限于束搜索，对于具有大型解码器的模型来说计算成本高昂。为克服这些局限，我们提出调整ASR模型以向前预测多个步骤，从而完全避免撤销步骤，更好地估计部分假设是否会导致生成完整的稀有词。仅通过微调Whisper使用10小时合成数据，我们的方法将NSC Part 2测试集上的单词错误率从30.86%降低到12.19%。 

---
# On Integrating Large Language Models and Scenario-Based Programming for Improving Software Reliability 

**Title (ZH)**: 基于情景编程改进软件可靠性的大型语言模型融合研究 

**Authors**: Ayelet Berzack, Guy Katz  

**Link**: [PDF](https://arxiv.org/pdf/2509.09194)  

**Abstract**: Large Language Models (LLMs) are fast becoming indispensable tools for software developers, assisting or even partnering with them in crafting complex programs. The advantages are evident -- LLMs can significantly reduce development time, generate well-organized and comprehensible code, and occasionally suggest innovative ideas that developers might not conceive on their own. However, despite their strengths, LLMs will often introduce significant errors and present incorrect code with persuasive confidence, potentially misleading developers into accepting flawed solutions.
In order to bring LLMs into the software development cycle in a more reliable manner, we propose a methodology for combining them with ``traditional'' software engineering techniques in a structured way, with the goal of streamlining the development process, reducing errors, and enabling users to verify crucial program properties with increased confidence. Specifically, we focus on the Scenario-Based Programming (SBP) paradigm -- an event-driven, scenario-based approach for software engineering -- to allow human developers to pour their expert knowledge into the LLM, as well as to inspect and verify its outputs.
To evaluate our methodology, we conducted a significant case study, and used it to design and implement the Connect4 game. By combining LLMs and SBP we were able to create a highly-capable agent, which could defeat various strong existing agents. Further, in some cases, we were able to formally verify the correctness of our agent. Finally, our experience reveals interesting insights regarding the ease-of-use of our proposed approach. The full code of our case-study will be made publicly available with the final version of this paper. 

**Abstract (ZH)**: 大型语言模型（LLMs）正逐渐成为软件开发者不可或缺的工具，协助甚至与他们合作编制复杂程序。尽管具备显著优势——LLMs能够大幅缩短开发时间，生成有序且易于理解的代码，并偶尔提出开发人员难以自主构思的创新想法——但它们也经常引入重大错误，以说服性的自信呈现错误代码，可能导致误导开发人员接受有缺陷的解决方案。

为了以更可靠的方式将LLMs融入软件开发周期，我们提出了一种将它们与传统的软件工程技术结构化结合的方法，旨在简化开发过程，减少错误，并使用户能够以更高的信心验证程序的关键属性。具体而言，我们专注于基于场景的编程（Scenario-Based Programming，SBP）范式——一种事件驱动的、基于场景的软件工程方法——使人类开发人员能够将其专家知识注入LLMs，并对其输出进行检查和验证。

为了评估我们的方法，我们进行了一项重要的案例研究，并利用该研究设计并实现了Connect4游戏。通过结合LLMs和SBP，我们能够创建出能够击败多种现有强健代理的高级代理。此外，在某些情况下，我们能够形式化验证我们代理的正确性。最终，我们的经验揭示了有关我们提出方法易用性的一些有趣见解。我们在论文的最终版本中将发布我们案例研究的全部代码。 

---
# Probing Pre-trained Language Models on Code Changes: Insights from ReDef, a High-Confidence Just-in-Time Defect Prediction Dataset 

**Title (ZH)**: 探讨预训练语言模型在代码变更上的应用：来自ReDef高置信度即时缺陷预测数据集的见解 

**Authors**: Doha Nam, Taehyoun Kim, Duksan Ryu, Jongmoon Baik  

**Link**: [PDF](https://arxiv.org/pdf/2509.09192)  

**Abstract**: Just-in-Time software defect prediction (JIT-SDP) plays a critical role in prioritizing risky code changes during code review and continuous integration. However, existing datasets often suffer from noisy labels and low precision in identifying bug-inducing commits. To address this, we present ReDef (Revert-based Defect dataset), a high-confidence benchmark of function-level modifications curated from 22 large-scale C/C++ projects. Defective cases are anchored by revert commits, while clean cases are validated through post-hoc history checks. Ambiguous instances are conservatively filtered out via a GPT-assisted triage process involving multiple votes and audits. This pipeline yields 3,164 defective and 10,268 clean modifications, offering substantially more reliable labels than prior existing resources. Beyond dataset construction, we provide the first systematic evaluation of how pre-trained language models (PLMs) reason about code modifications -- specifically, which input encodings most effectively expose change information, and whether models genuinely capture edit semantics. We fine-tune CodeBERT, CodeT5+, and UniXcoder under five encoding strategies, and further probe their sensitivity through counterfactual perturbations that swap added/deleted blocks, invert diff polarity, or inject spurious markers. Our results show that compact diff-style encodings consistently outperform whole-function formats across all PLMs, with statistical tests confirming large, model-independent effects. However, under counterfactual tests, performance degrades little or not at all -- revealing that what appears to be robustness in fact reflects reliance on superficial cues rather than true semantic understanding. These findings indicate that, unlike in snapshot-based tasks, current PLMs remain limited in their ability to genuinely comprehend code modifications. 

**Abstract (ZH)**: 基于回退的缺陷数据集（ReDef）：一种高置信度的功能修改基准 

---
# Dark-ISP: Enhancing RAW Image Processing for Low-Light Object Detection 

**Title (ZH)**: Dark-ISP: 提升低光照条件下RAW图像处理性能以增强物体检测 

**Authors**: Jiasheng Guo, Xin Gao, Yuxiang Yan, Guanghao Li, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09183)  

**Abstract**: Low-light Object detection is crucial for many real-world applications but remains challenging due to degraded image quality. While recent studies have shown that RAW images offer superior potential over RGB images, existing approaches either use RAW-RGB images with information loss or employ complex frameworks. To address these, we propose a lightweight and self-adaptive Image Signal Processing (ISP) plugin, Dark-ISP, which directly processes Bayer RAW images in dark environments, enabling seamless end-to-end training for object detection. Our key innovations are: (1) We deconstruct conventional ISP pipelines into sequential linear (sensor calibration) and nonlinear (tone mapping) sub-modules, recasting them as differentiable components optimized through task-driven losses. Each module is equipped with content-aware adaptability and physics-informed priors, enabling automatic RAW-to-RGB conversion aligned with detection objectives. (2) By exploiting the ISP pipeline's intrinsic cascade structure, we devise a Self-Boost mechanism that facilitates cooperation between sub-modules. Through extensive experiments on three RAW image datasets, we demonstrate that our method outperforms state-of-the-art RGB- and RAW-based detection approaches, achieving superior results with minimal parameters in challenging low-light environments. 

**Abstract (ZH)**: 低光照环境下物体检测对于许多实际应用至关重要，但由于图像质量退化而仍然具有挑战性。虽然最近的研究表明，RAW图像相较于RGB图像具有更优的潜力，但现有方法要么使用包含信息丢失的RAW-RGB图像，要么采用复杂的框架。为了解决这些问题，我们提出了一种轻量级且自适应的图像信号处理（ISP）插件——Dark-ISP，该插件可以直接在暗环境处理拜耶（Bayer）RAW图像，从而实现物体检测的端到端无缝训练。我们的关键创新在于：（1）我们将传统的ISP流水线分解为顺序的线性（传感器校准）和非线性（色调映射）子模块，并将它们重新定义为通过任务驱动的损失优化的不同可微分组件。每个模块配备了内容感知的自适应能力和物理先验，能够自动将RAW图像转换为RGB图像，与检测目标对齐。（2）通过利用ISP流水线固有的级联结构，我们设计了一种自我增强机制，促进了子模块之间的协作。在三个RAW图像数据集上的广泛实验表明，我们的方法在挑战性的低光照环境下优于最先进的RGB和RAW基检测方法，在最少参数的情况下取得了优越的结果。 

---
# EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs 

**Title (ZH)**: EchoX：通过回声训练减轻语音-语义差距以提高语音到语音的大语言模型性能 

**Authors**: Yuhao Zhang, Yuhao Du, Zhanchen Dai, Xiangnan Ma, Kaiqi Kou, Benyou Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.09174)  

**Abstract**: Speech-to-speech large language models (SLLMs) are attracting increasing attention. Derived from text-based large language models (LLMs), SLLMs often exhibit degradation in knowledge and reasoning capabilities. We hypothesize that this limitation arises because current training paradigms for SLLMs fail to bridge the acoustic-semantic gap in the feature representation space. To address this issue, we propose EchoX, which leverages semantic representations and dynamically generates speech training targets. This approach integrates both acoustic and semantic learning, enabling EchoX to preserve strong reasoning abilities as a speech LLM. Experimental results demonstrate that EchoX, with about six thousand hours of training data, achieves advanced performance on multiple knowledge-based question-answering benchmarks. The project is available at this https URL. 

**Abstract (ZH)**: 基于语音的大语言模型（SLLMs）正吸引越来越多的关注。源于文本的大语言模型（LLMs），SLLMs通常在知识和推理能力上表现出下降。我们假设这一限制源于当前SLLMs的训练范式未能在特征表示空间中跨越声学语义差距。为解决这一问题，我们提出了EchoX，它利用语义表示并动态生成语音训练目标。该方法结合了声学和语义学习，使EchoX能够保留强烈的推理能力作为语音大语言模型。实验结果表明，使用大约六千小时的训练数据，EchoX在多个基于知识的问答基准测试中取得了先进性能。项目详情请见此链接。 

---
# Adaptive Pareto-Optimal Token Merging for Edge Transformer Models in Semantic Communication 

**Title (ZH)**: 边缘变压器模型中语义通信的自适应帕累托最优标记合并 

**Authors**: Omar Erak, Omar Alhussein, Hatem Abou-Zeid, Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2509.09168)  

**Abstract**: Large-scale transformer models have emerged as a powerful tool for semantic communication systems, enabling edge devices to extract rich representations for robust inference across noisy wireless channels. However, their substantial computational demands remain a major barrier to practical deployment in resource-constrained 6G networks. In this paper, we present a training-free framework for adaptive token merging in pretrained vision transformers to jointly reduce inference time and transmission resource usage. We formulate the selection of per-layer merging proportions as a multi-objective optimization problem to balance accuracy and computational cost. We employ Gaussian process-based Bayesian optimization to construct a Pareto frontier of optimal configurations, enabling flexible runtime adaptation to dynamic application requirements and channel conditions. Extensive experiments demonstrate that our method consistently outperforms other baselines and achieves significant reductions in floating-point operations while maintaining competitive accuracy across a wide range of signal-to-noise ratio (SNR) conditions. Additional results highlight the effectiveness of adaptive policies that adjust merging aggressiveness in response to channel quality, providing a practical mechanism to trade off latency and semantic fidelity on demand. These findings establish a scalable and efficient approach for deploying transformer-based semantic communication in future edge intelligence systems. 

**Abstract (ZH)**: 大规模变压器模型已成为语义通信系统中强大工具，能够使边缘设备在嘈杂的无线信道中提取丰富的表示以进行稳健的推理。然而，它们巨大的计算需求仍然是在资源受限的6G网络中实际部署的主要障碍。本文提出了一种无需训练的自适应令牌合并框架，用于预训练视觉变压器，以联合减少推理时间和传输资源使用量。我们将每层合并比例的选择形式化为多目标优化问题，以平衡准确性和计算成本。我们采用基于高斯过程的贝叶斯优化来构建最优配置的帕累托前沿，从而在运行时灵活适应动态应用需求和信道条件。大量实验证明，我们的方法在各种信噪比（SNR）条件下始终优于其他基线方法，在保持竞争力的同时显著减少了浮点运算量。此外，结果还突显了根据信道质量调整合并激进性的自适应策略的有效性，提供了一种在需求基础上权衡延迟和语义保真度的实用机制。这些发现确立了在未来的边缘智能系统中部署基于变压器的语义通信的可扩展和高效方法。 

---
# Target-oriented Multimodal Sentiment Classification with Counterfactual-enhanced Debiasing 

**Title (ZH)**: 面向目标的多模态情感分类与反事实增强去偏见 

**Authors**: Zhiyue Liu, Fanrong Ma, Xin Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.09160)  

**Abstract**: Target-oriented multimodal sentiment classification seeks to predict sentiment polarity for specific targets from image-text pairs. While existing works achieve competitive performance, they often over-rely on textual content and fail to consider dataset biases, in particular word-level contextual biases. This leads to spurious correlations between text features and output labels, impairing classification accuracy. In this paper, we introduce a novel counterfactual-enhanced debiasing framework to reduce such spurious correlations. Our framework incorporates a counterfactual data augmentation strategy that minimally alters sentiment-related causal features, generating detail-matched image-text samples to guide the model's attention toward content tied to sentiment. Furthermore, for learning robust features from counterfactual data and prompting model decisions, we introduce an adaptive debiasing contrastive learning mechanism, which effectively mitigates the influence of biased words. Experimental results on several benchmark datasets show that our proposed method outperforms state-of-the-art baselines. 

**Abstract (ZH)**: 面向目标的多模态情感分类旨在从图像-文本对中预测特定目标的情感极性。尽管现有工作取得了竞争力的表现，但它们往往过度依赖文本内容，并未考虑数据集偏见，特别是字面级上下文偏见。这导致了文本特征与输出标签之间的虚假相关性，影响了分类准确性。在本文中，我们提出了一种新颖的反事实增强去偏见框架以降低此类虚假相关性。我们的框架结合了一种最小改变情感相关因果特征的反事实数据增强策略，生成细节匹配的图像-文本样本以引导模型关注与情感相关的内容。此外，为了从反事实数据中学习稳健的特征并促使模型决策，我们引入了一种适应性去偏见对比学习机制，有效缓解了偏色词汇的影响。在几个基准数据集上的实验结果表明，我们提出的方法优于现有最先进的基线方法。 

---
# A Knowledge Noise Mitigation Framework for Knowledge-based Visual Question Answering 

**Title (ZH)**: 基于知识的视觉问答中知识噪声 mitigation 的框架 

**Authors**: Zhiyue Liu, Sihang Liu, Jinyuan Liu, Xinru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09159)  

**Abstract**: Knowledge-based visual question answering (KB-VQA) requires a model to understand images and utilize external knowledge to provide accurate answers. Existing approaches often directly augment models with retrieved information from knowledge sources while ignoring substantial knowledge redundancy, which introduces noise into the answering process. To address this, we propose a training-free framework with knowledge focusing for KB-VQA, that mitigates the impact of noise by enhancing knowledge relevance and reducing redundancy. First, for knowledge retrieval, our framework concludes essential parts from the image-question pairs, creating low-noise queries that enhance the retrieval of highly relevant knowledge. Considering that redundancy still persists in the retrieved knowledge, we then prompt large models to identify and extract answer-beneficial segments from knowledge. In addition, we introduce a selective knowledge integration strategy, allowing the model to incorporate knowledge only when it lacks confidence in answering the question, thereby mitigating the influence of redundant information. Our framework enables the acquisition of accurate and critical knowledge, and extensive experiments demonstrate that it outperforms state-of-the-art methods. 

**Abstract (ZH)**: 基于知识的视觉问答（KB-VQA）要求模型理解图像并利用外部知识提供准确的答案。现有的方法往往直接通过从知识源中检索信息来增强模型，但忽视了知识冗余，引入了噪声。为了解决这个问题，我们提出了一种无需训练的知识聚焦框架，通过增强知识的相关性和减少冗余来减轻噪声的影响。首先，在知识检索方面，我们的框架从图像-问题对中得出关键部分，创建低噪声查询以增强高度相关知识的检索。考虑到检索的知识中仍然存在冗余，我们随后促使大模型识别并提取有益于答案的片段。此外，我们引入了一种选择性知识整合策略，允许模型仅在缺乏回答问题的信心时才整合知识，从而减轻冗余信息的影响。该框架能够获取准确和关键的知识，并且大量的实验表明，它优于现有最佳方法。 

---
# HISPASpoof: A New Dataset For Spanish Speech Forensics 

**Title (ZH)**: HISPASpoof: 一个新的西班牙语语音鉴真数据集 

**Authors**: Maria Risques, Kratika Bhagtani, Amit Kumar Singh Yadav, Edward J. Delp  

**Link**: [PDF](https://arxiv.org/pdf/2509.09155)  

**Abstract**: Zero-shot Voice Cloning (VC) and Text-to-Speech (TTS) methods have advanced rapidly, enabling the generation of highly realistic synthetic speech and raising serious concerns about their misuse. While numerous detectors have been developed for English and Chinese, Spanish-spoken by over 600 million people worldwide-remains underrepresented in speech forensics. To address this gap, we introduce HISPASpoof, the first large-scale Spanish dataset designed for synthetic speech detection and attribution. It includes real speech from public corpora across six accents and synthetic speech generated with six zero-shot TTS systems. We evaluate five representative methods, showing that detectors trained on English fail to generalize to Spanish, while training on HISPASpoof substantially improves detection. We also evaluate synthetic speech attribution performance on HISPASpoof, i.e., identifying the generation method of synthetic speech. HISPASpoof thus provides a critical benchmark for advancing reliable and inclusive speech forensics in Spanish. 

**Abstract (ZH)**: 零样本语音克隆(VC)和文本到语音(TTS)方法取得了 rapid进展，使得生成高度逼真的合成语音成为可能，并引发了对其误用的严重担忧。虽然针对英汉的检测器已有广泛开发，但 spoken于全球超过6亿人口中的西班牙语，在语音取证方面仍然严重代表性不足。为填补这一空白，我们引入了 HISPASpoof，这是首个专门为合成语音检测和归因设计的大型西班牙语数据集。该数据集包含来自六个口音的公共语料库的真实语音以及使用六个零样本TTS系统生成的合成语音。我们评估了五种代表性方法，结果表明，针对英语训练的检测器无法泛化到西班牙语，而使用HISPASpoof训练显著提高了检测效果。我们还评估了HISPASpoof上的合成语音归因性能，即识别合成语音的生成方法。因此，HISPASpoof为推进可靠且包容的西班牙语语音取证提供了关键基准。 

---
# OCELOT 2023: Cell Detection from Cell-Tissue Interaction Challenge 

**Title (ZH)**: OCELOT 2023: 细胞检测从细胞-组织相互作用挑战赛 

**Authors**: JaeWoong Shin, Jeongun Ryu, Aaron Valero Puche, Jinhee Lee, Biagio Brattoli, Wonkyung Jung, Soo Ick Cho, Kyunghyun Paeng, Chan-Young Ock, Donggeun Yoo, Zhaoyang Li, Wangkai Li, Huayu Mai, Joshua Millward, Zhen He, Aiden Nibali, Lydia Anette Schoenpflug, Viktor Hendrik Koelzer, Xu Shuoyu, Ji Zheng, Hu Bin, Yu-Wen Lo, Ching-Hui Yang, Sérgio Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2509.09153)  

**Abstract**: Pathologists routinely alternate between different magnifications when examining Whole-Slide Images, allowing them to evaluate both broad tissue morphology and intricate cellular details to form comprehensive diagnoses. However, existing deep learning-based cell detection models struggle to replicate these behaviors and learn the interdependent semantics between structures at different magnifications. A key barrier in the field is the lack of datasets with multi-scale overlapping cell and tissue annotations. The OCELOT 2023 challenge was initiated to gather insights from the community to validate the hypothesis that understanding cell and tissue (cell-tissue) interactions is crucial for achieving human-level performance, and to accelerate the research in this field. The challenge dataset includes overlapping cell detection and tissue segmentation annotations from six organs, comprising 673 pairs sourced from 306 The Cancer Genome Atlas (TCGA) Whole-Slide Images with hematoxylin and eosin staining, divided into training, validation, and test subsets. Participants presented models that significantly enhanced the understanding of cell-tissue relationships. Top entries achieved up to a 7.99 increase in F1-score on the test set compared to the baseline cell-only model that did not incorporate cell-tissue relationships. This is a substantial improvement in performance over traditional cell-only detection methods, demonstrating the need for incorporating multi-scale semantics into the models. This paper provides a comparative analysis of the methods used by participants, highlighting innovative strategies implemented in the OCELOT 2023 challenge. 

**Abstract (ZH)**: 病理学家在检查全视野图像时会交替使用不同的放大倍数，以便评估组织的宏观结构和细胞的细微细节，从而形成全面的诊断。然而，现有的基于深度学习的细胞检测模型难以复制这些行为，并学习不同放大倍数下结构之间的相互依赖语义。领域内的主要障碍是没有多尺度重叠细胞和组织注释的数据集。2023年OCELOT挑战赛旨在从社区中获得见解，验证理解细胞和组织（细胞-组织）相互作用对实现人类水平性能是至关重要的假说，并加速该领域的研究。挑战数据集包括来自306张苏木精和伊红染色的癌症基因组图谱（TCGA）全视野图像中六种器官的重叠细胞检测和组织分割注释，共计673对，分为训练、验证和测试子集。参与者展示了显著增强细胞-组织关系理解的模型。顶级参赛作品在测试集上的F1分数比不考虑细胞-组织关系的基本细胞检测模型提高了7.99%，这一性能提升显著优于传统的细胞检测方法，证明了将多尺度语义纳入模型的必要性。本文对OCELOT 2023挑战赛中参赛者使用的各种方法进行了比较分析，突出了该挑战赛中实施的创新策略。 

---
# Video Understanding by Design: How Datasets Shape Architectures and Insights 

**Title (ZH)**: 设计中的视频理解：数据集如何塑造架构与洞察 

**Authors**: Lei Wang, Piotr Koniusz, Yongsheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.09151)  

**Abstract**: Video understanding has advanced rapidly, fueled by increasingly complex datasets and powerful architectures. Yet existing surveys largely classify models by task or family, overlooking the structural pressures through which datasets guide architectural evolution. This survey is the first to adopt a dataset-driven perspective, showing how motion complexity, temporal span, hierarchical composition, and multimodal richness impose inductive biases that models should encode. We reinterpret milestones, from two-stream and 3D CNNs to sequential, transformer, and multimodal foundation models, as concrete responses to these dataset-driven pressures. Building on this synthesis, we offer practical guidance for aligning model design with dataset invariances while balancing scalability and task demands. By unifying datasets, inductive biases, and architectures into a coherent framework, this survey provides both a comprehensive retrospective and a prescriptive roadmap for advancing general-purpose video understanding. 

**Abstract (ZH)**: 基于数据集视角的视频理解模型演化综述：结构压力下的引致偏置与模型设计指南 

---
# Objectness Similarity: Capturing Object-Level Fidelity in 3D Scene Evaluation 

**Title (ZH)**: 对象相似性：3D 场景评估中的对象级保真度捕获 

**Authors**: Yuiko Uchida, Ren Togo, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2509.09143)  

**Abstract**: This paper presents Objectness SIMilarity (OSIM), a novel evaluation metric for 3D scenes that explicitly focuses on "objects," which are fundamental units of human visual perception. Existing metrics assess overall image quality, leading to discrepancies with human perception. Inspired by neuropsychological insights, we hypothesize that human recognition of 3D scenes fundamentally involves attention to individual objects. OSIM enables object-centric evaluations by leveraging an object detection model and its feature representations to quantify the "objectness" of each object in the scene. Our user study demonstrates that OSIM aligns more closely with human perception compared to existing metrics. We also analyze the characteristics of OSIM using various approaches. Moreover, we re-evaluate recent 3D reconstruction and generation models under a standardized experimental setup to clarify advancements in this field. The code is available at this https URL. 

**Abstract (ZH)**: Objectness SIMilarity: 一种针对3D场景的新颖评估指标 

---
# ViRanker: A BGE-M3 & Blockwise Parallel Transformer Cross-Encoder for Vietnamese Reranking 

**Title (ZH)**: ViRanker: 基于BGE-M3及块级并行变压器交叉编码的越南语重排模型 

**Authors**: Phuong-Nam Dang, Kieu-Linh Nguyen, Thanh-Hieu Pham  

**Link**: [PDF](https://arxiv.org/pdf/2509.09131)  

**Abstract**: This paper presents ViRanker, a cross-encoder reranking model tailored to the Vietnamese language. Built on the BGE-M3 encoder and enhanced with the Blockwise Parallel Transformer, ViRanker addresses the lack of competitive rerankers for Vietnamese, a low-resource language with complex syntax and diacritics. The model was trained on an 8 GB curated corpus and fine-tuned with hybrid hard-negative sampling to strengthen robustness. Evaluated on the MMARCO-VI benchmark, ViRanker achieves strong early-rank accuracy, surpassing multilingual baselines and competing closely with PhoRanker. By releasing the model openly on Hugging Face, we aim to support reproducibility and encourage wider adoption in real-world retrieval systems. Beyond Vietnamese, this study illustrates how careful architectural adaptation and data curation can advance reranking in other underrepresented languages. 

**Abstract (ZH)**: ViRanker：一种针对越南语的跨编码重排模型 

---
# Automated Classification of Tutors' Dialogue Acts Using Generative AI: A Case Study Using the CIMA Corpus 

**Title (ZH)**: 使用生成式AI对导师对话行为进行自动分类：基于CIMA语料库的案例研究 

**Authors**: Liqun He, Jiaqi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09125)  

**Abstract**: This study explores the use of generative AI for automating the classification of tutors' Dialogue Acts (DAs), aiming to reduce the time and effort required by traditional manual coding. This case study uses the open-source CIMA corpus, in which tutors' responses are pre-annotated into four DA categories. Both GPT-3.5-turbo and GPT-4 models were tested using tailored prompts. Results show that GPT-4 achieved 80% accuracy, a weighted F1-score of 0.81, and a Cohen's Kappa of 0.74, surpassing baseline performance and indicating substantial agreement with human annotations. These findings suggest that generative AI has strong potential to provide an efficient and accessible approach to DA classification, with meaningful implications for educational dialogue analysis. The study also highlights the importance of task-specific label definitions and contextual information in enhancing the quality of automated annotation. Finally, it underscores the ethical considerations associated with the use of generative AI and the need for responsible and transparent research practices. The script of this research is publicly available at this https URL. 

**Abstract (ZH)**: 本研究探讨了生成式AI在自动分类辅导对话行为（DAs）中的应用，旨在减少传统手动编码所需的时间和 effort。该案例研究使用开源的CIMA语料库，在该语料库中，辅导者的响应已被标注为四个DA类别。测试了GPT-3.5-turbo和GPT-4模型，并使用了定制的提示。结果显示，GPT-4实现了80%的准确率、加权F1分数为0.81、科恩κ系数为0.74，超过了基线性能，并表明与人类标注存在显著一致性。这些发现表明，生成式AI在提供高效且易 accessibility 的DA分类方法方面具有巨大潜力，对于教育对话分析具有重要含义。此外，研究还强调了特定任务标签定义和上下文信息在提高自动标注质量方面的的重要性。最后，研究突显了生成式AI使用的伦理考虑及负责任、透明的研究实践的重要性。该研究的代码在此httpsURL公开可用。 

---
# Character-Level Perturbations Disrupt LLM Watermarks 

**Title (ZH)**: 字符级扰动破坏LLM水印 

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09112)  

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.
To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms. 

**Abstract (ZH)**: 大型语言模型（LLM）水印嵌入可检测信号以实现版权保护、滥用预防和内容检测。尽管先前的研究通过水印移除攻击评估鲁棒性，但这些方法往往不尽如人意，造成一种误解，即有效移除需要大量扰动或强大对手。

为此，我们首先形式化LLM水印系统模型，并在受限于有限访问水印检测器的情况下刻画两种现实威胁模型。随后，我们分析不同类型扰动的攻击范围，即单次编辑所能影响的标记数量。我们发现，字符级扰动（例如，拼写错误、交换、删除、谐音字符）可以通过干扰分词过程同时影响多个标记。我们证明，在最严格的威胁模型下，字符级扰动在水印移除方面表现出显著效果。我们进一步提出基于遗传算法（GA）的引导式移除攻击，该方法使用参考检测器进行优化。在对水印检测器的黑盒查询受限的现实威胁模型下，我们的方法展示了强大的移除性能。实验验证了字符级扰动的优势以及遗传算法在现实约束下移除水印的有效性。此外，我们认为在考虑潜在防御措施时存在一种对手困境：任何固定防御都可以通过适当扰动策略被绕过。基于这一原则，我们提出了适应性复合字符级攻击。实验结果表明，该方法能够有效破解防御。我们的发现突显了现有LLM水印方案中的重大漏洞，并强调了开发新的鲁棒机制的紧迫性。 

---
# DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device Large Language Models 

**Title (ZH)**: DP-FedLoRA：增强隐私的联邦微调用于设备上大型语言模型 

**Authors**: Honghui Xu, Shiva Shrestha, Wei Chen, Zhiyuan Li, Zhipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09097)  

**Abstract**: As on-device large language model (LLM) systems become increasingly prevalent, federated fine-tuning enables advanced language understanding and generation directly on edge devices; however, it also involves processing sensitive, user-specific data, raising significant privacy concerns within the federated learning framework. To address these challenges, we propose DP-FedLoRA, a privacy-enhanced federated fine-tuning framework that integrates LoRA-based adaptation with differential privacy in a communication-efficient setting. Each client locally clips and perturbs its LoRA matrices using Gaussian noise to satisfy ($\epsilon$, $\delta$)-differential privacy. We further provide a theoretical analysis demonstrating the unbiased nature of the updates and deriving bounds on the variance introduced by noise, offering practical guidance for privacy-budget calibration. Experimental results across mainstream benchmarks show that DP-FedLoRA delivers competitive performance while offering strong privacy guarantees, paving the way for scalable and privacy-preserving LLM deployment in on-device environments. 

**Abstract (ZH)**: 随着设备端大型语言模型系统日益普及，边缘设备上的联邦微调能够直接实现高级语言理解和生成，但也涉及处理敏感的用户特定数据，引发联邦学习框架内的重大隐私问题。为应对这些挑战，我们提出DP-FedLoRA，一种结合LoRA基适应与差分隐私的隐私增强型联邦微调框架，能够在通信高效设置中运行。每个客户端使用高斯噪声本地裁剪和扰动其LoRA矩阵以满足($\epsilon$, $\delta$)-差分隐私。我们还提供了理论分析，证明了更新的无偏性，并得出了由噪声引入的方差上界，为隐私预算校准提供了实用指导。在主流基准上的实验结果表明，DP-FedLoRA在提供强隐私保障的同时实现了竞争力的表现，为设备端环境下可扩展和隐私保护的语言模型部署铺平了道路。 

---
# Towards Confidential and Efficient LLM Inference with Dual Privacy Protection 

**Title (ZH)**: 面向双重隐私保护的保密高效大语言模型推理 

**Authors**: Honglan Yu, Yibin Wang, Feifei Dai, Dong Liu, Haihui Fan, Xiaoyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09091)  

**Abstract**: CPU-based trusted execution environments (TEEs) and differential privacy (DP) have gained wide applications for private inference. Due to high inference latency in TEEs, researchers use partition-based approaches that offload linear model components to GPUs. However, dense nonlinear layers of large language models (LLMs) result in significant communication overhead between TEEs and GPUs. DP-based approaches apply random noise to protect data privacy, but this compromises LLM performance and semantic understanding. To overcome the above drawbacks, this paper proposes CMIF, a Confidential and efficient Model Inference Framework. CMIF confidentially deploys the embedding layer in the client-side TEE and subsequent layers on GPU servers. Meanwhile, it optimizes the Report-Noisy-Max mechanism to protect sensitive inputs with a slight decrease in model performance. Extensive experiments on Llama-series models demonstrate that CMIF reduces additional inference overhead in TEEs while preserving user data privacy. 

**Abstract (ZH)**: 基于CPU的受信执行环境（TEEs）和差分隐私（DP）在私有推理中获得了广泛应用。尽管TEEs中的高推理延迟促使研究人员采用基于分区的方法将线性模型组件卸载到GPU上，但大型语言模型（LLMs）的密集非线性层导致了TEEs与GPU之间显著的通信开销。基于DP的方法通过添加随机噪声来保护数据隐私，但这会影响LLM的性能和语义理解。为克服上述缺点，本文提出了一种名为CMIF的保密高效模型推理框架。CMIF在客户端TEEs中 confidential地部署嵌入层，并将后续层部署在GPU服务器上。同时，它优化了Report-Noisy-Max机制以保护敏感输入，并且仅轻微降低模型性能。在Llama系列模型上的广泛实验表明，CMIF可以减少TEEs中的额外推理开销，同时保持用户数据隐私。 

---
# SQAP-VLA: A Synergistic Quantization-Aware Pruning Framework for High-Performance Vision-Language-Action Models 

**Title (ZH)**: SQAP-VLA：一种面向高性能视觉-语言-动作模型的协同量化意识剪枝框架 

**Authors**: Hengyu Fang, Yijiang Liu, Yuan Du, Li Du, Huanrui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09090)  

**Abstract**: Vision-Language-Action (VLA) models exhibit unprecedented capabilities for embodied intelligence. However, their extensive computational and memory costs hinder their practical deployment. Existing VLA compression and acceleration approaches conduct quantization or token pruning in an ad-hoc manner but fail to enable both for a holistic efficiency improvement due to an observed incompatibility. This work introduces SQAP-VLA, the first structured, training-free VLA inference acceleration framework that simultaneously enables state-of-the-art quantization and token pruning. We overcome the incompatibility by co-designing the quantization and token pruning pipeline, where we propose new quantization-aware token pruning criteria that work on an aggressively quantized model while improving the quantizer design to enhance pruning effectiveness. When applied to standard VLA models, SQAP-VLA yields significant gains in computational efficiency and inference speed while successfully preserving core model performance, achieving a $\times$1.93 speedup and up to a 4.5\% average success rate enhancement compared to the original model. 

**Abstract (ZH)**: SQAP-VLA：同时实现先进量化和 token 裁剪的结构化无训练加速框架 

---
# KoopMotion: Learning Almost Divergence Free Koopman Flow Fields for Motion Planning 

**Title (ZH)**: KoopMotion: 学习几乎无散度的Koopman流场以进行运动规划 

**Authors**: Alice Kate Li, Thales C Silva, Victoria Edwards, Vijay Kumar, M. Ani Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2509.09074)  

**Abstract**: In this work, we propose a novel flow field-based motion planning method that drives a robot from any initial state to a desired reference trajectory such that it converges to the trajectory's end point. Despite demonstrated efficacy in using Koopman operator theory for modeling dynamical systems, Koopman does not inherently enforce convergence to desired trajectories nor to specified goals -- a requirement when learning from demonstrations (LfD). We present KoopMotion which represents motion flow fields as dynamical systems, parameterized by Koopman Operators to mimic desired trajectories, and leverages the divergence properties of the learnt flow fields to obtain smooth motion fields that converge to a desired reference trajectory when a robot is placed away from the desired trajectory, and tracks the trajectory until the end point. To demonstrate the effectiveness of our approach, we show evaluations of KoopMotion on the LASA human handwriting dataset and a 3D manipulator end-effector trajectory dataset, including spectral analysis. We also perform experiments on a physical robot, verifying KoopMotion on a miniature autonomous surface vehicle operating in a non-static fluid flow environment. Our approach is highly sample efficient in both space and time, requiring only 3\% of the LASA dataset to generate dense motion plans. Additionally, KoopMotion provides a significant improvement over baselines when comparing metrics that measure spatial and temporal dynamics modeling efficacy. 

**Abstract (ZH)**: 基于流场的运动规划方法：KoopMotion 

---
# STRIDE: Scalable and Interpretable XAI via Subset-Free Functional Decomposition 

**Title (ZH)**: STRIDE: 面向子集无损的功能分解以实现可解释性的人工智能 scalability 和可解释性 

**Authors**: Chaeyun Ko  

**Link**: [PDF](https://arxiv.org/pdf/2509.09070)  

**Abstract**: Most explainable AI (XAI) frameworks face two practical limitations: the exponential cost of reasoning over feature subsets and the reduced expressiveness of summarizing effects as single scalar values. We present STRIDE, a scalable framework that aims to mitigate both issues by framing explanation as a subset-enumeration-free, orthogonal functional decomposition in a Reproducing Kernel Hilbert Space (RKHS). Rather than focusing only on scalar attributions, STRIDE computes functional components f_S(x_S) via an analytical projection scheme based on a recursive kernel-centering procedure, avoiding explicit subset enumeration. In the tabular setups we study, the approach is model-agnostic, provides both local and global views, and is supported by theoretical results on orthogonality and L^2 convergence under stated assumptions. On public tabular benchmarks in our environment, we observed speedups ranging from 0.6 times (slower than TreeSHAP on a small dataset) to 9.7 times (California), with a median approximate 3.0 times across 10 datasets, while maintaining high fidelity (R^2 between 0.81 and 0.999) and substantial rank agreement on most datasets. Overall, STRIDE complements scalar attribution methods by offering a structured functional perspective, enabling novel diagnostics like 'component surgery' to quantitatively measure the impact of specific interactions within our experimental scope. 

**Abstract (ZH)**: STRIDE：一种缓解特征子集枚举和效果单一标量值总结问题的可解释AI框架 

---
# Improving LLM Safety and Helpfulness using SFT and DPO: A Study on OPT-350M 

**Title (ZH)**: 使用SFT和DPO提升LLM的安全性和有用性：基于OPT-350M的研究 

**Authors**: Piyush Pant  

**Link**: [PDF](https://arxiv.org/pdf/2509.09055)  

**Abstract**: This research investigates the effectiveness of alignment techniques, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and a combined SFT+DPO approach on improving the safety and helpfulness of the OPT-350M language model. Utilizing the Anthropic Helpful-Harmless RLHF dataset, we train and evaluate four models: the base OPT350M, an SFT model, a DPO model, and a model trained with both SFT and DPO. We introduce three key evaluation metrics: Harmlessness Rate (HmR), Helpfulness Rate (HpR), and a Combined Alignment Score (CAS), all derived from reward model outputs. The results show that while SFT outperforms DPO, The combined SFT+DPO model outperforms all others across all metrics, demonstrating the complementary nature of these techniques. Our findings also highlight challenges posed by noisy data, limited GPU resources, and training constraints. This study offers a comprehensive view of how fine-tuning strategies affect model alignment and provides a foundation for more robust alignment pipelines in future work. 

**Abstract (ZH)**: 本研究探讨了对OPT-350M语言模型进行对齐技术（包括监督微调SFT、直接偏好优化DPO以及SFT+DPO结合方法）的有效性，以提高模型的安全性和帮助性。利用Anthropic Helpful-Harmless RLHF数据集，我们训练并评估了四个模型：基础OPT350M、SFT模型、DPO模型以及结合了SFT和DPO的模型。我们引入了三个关键评估指标：无害率（HmR）、帮助率（HpR）以及综合对齐得分（CAS），这些指标均源自奖励模型的输出。结果显示，虽然SFT的性能优于DPO，但结合了SFT和DPO的方法在所有指标上均表现出色，证明了这些技术的互补性。此外，研究还揭示了噪声数据、有限GPU资源和训练约束带来的挑战。本研究为未来更 robust 的对齐流水线提供了全面的视角，并提供了细调策略影响模型对齐的见解。 

---
# A Scoping Review of Machine Learning Applications in Power System Protection and Disturbance Management 

**Title (ZH)**: 机器学习在电力系统保护与扰动管理中的应用综述 

**Authors**: Julian Oelhaf, Georg Kordowich, Mehran Pashaei, Christian Bergler, Andreas Maier, Johann Jäger, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2509.09053)  

**Abstract**: The integration of renewable and distributed energy resources reshapes modern power systems, challenging conventional protection schemes. This scoping review synthesizes recent literature on machine learning (ML) applications in power system protection and disturbance management, following the PRISMA for Scoping Reviews framework. Based on over 100 publications, three key objectives are addressed: (i) assessing the scope of ML research in protection tasks; (ii) evaluating ML performance across diverse operational scenarios; and (iii) identifying methods suitable for evolving grid conditions. ML models often demonstrate high accuracy on simulated datasets; however, their performance under real-world conditions remains insufficiently validated. The existing literature is fragmented, with inconsistencies in methodological rigor, dataset quality, and evaluation metrics. This lack of standardization hampers the comparability of results and limits the generalizability of findings. To address these challenges, this review introduces a ML-oriented taxonomy for protection tasks, resolves key terminological inconsistencies, and advocates for standardized reporting practices. It further provides guidelines for comprehensive dataset documentation, methodological transparency, and consistent evaluation protocols, aiming to improve reproducibility and enhance the practical relevance of research outcomes. Critical gaps remain, including the scarcity of real-world validation, insufficient robustness testing, and limited consideration of deployment feasibility. Future research should prioritize public benchmark datasets, realistic validation methods, and advanced ML architectures. These steps are essential to move ML-based protection from theoretical promise to practical deployment in increasingly dynamic and decentralized power systems. 

**Abstract (ZH)**: 可再生和分布式能源资源的集成重塑了现代电力系统，挑战了传统的保护方案。本综述性研究遵循PRISMA for Scoping Reviews框架，综合分析了机器学习在电力系统保护与扰动管理中的应用文献，基于超过100篇出版物，重点探讨了三大目标：（i）评估机器学习在保护任务中的研究范围；（ii）评估机器学习在不同运行场景中的性能；（iii）识别适用于不断变化的电网条件的方法。机器学习模型在模拟数据集上通常表现出较高的准确性，但在现实世界条件下的性能仍缺乏充分验证。现有文献碎片化，方法学严谨性、数据集质量及评估指标存在不一致。缺乏标准化限制了结果的可比性和发现的一般适用性。为应对这些挑战，本综述提出了面向机器学习的保护任务分类体系，解决了关键术语的一致性问题，并倡导标准化报告实践。此外，提供了全面的数据集文档、方法学透明及一致评估协议的指导原则，旨在提高可重复性并增强研究成果的实际相关性。现有的研究仍存在现实验证不足、鲁棒性测试不够以及部署可行性考虑有限的关键空白。未来研究应优先采用公共基准数据集、现实验证方法及先进机器学习架构。这样做对于将基于机器学习的保护从理论潜力转化为在日益动态和分布式的电力系统中的实际部署至关重要。 

---
# MoWE : A Mixture of Weather Experts 

**Title (ZH)**: MoWE : 一种混合天气专家系统 

**Authors**: Dibyajyoti Chakraborty, Romit Maulik, Peter Harrington, Dallas Foster, Mohammad Amin Nabian, Sanjay Choudhry  

**Link**: [PDF](https://arxiv.org/pdf/2509.09052)  

**Abstract**: Data-driven weather models have recently achieved state-of-the-art performance, yet progress has plateaued in recent years. This paper introduces a Mixture of Experts (MoWE) approach as a novel paradigm to overcome these limitations, not by creating a new forecaster, but by optimally combining the outputs of existing models. The MoWE model is trained with significantly lower computational resources than the individual experts. Our model employs a Vision Transformer-based gating network that dynamically learns to weight the contributions of multiple "expert" models at each grid point, conditioned on forecast lead time. This approach creates a synthesized deterministic forecast that is more accurate than any individual component in terms of Root Mean Squared Error (RMSE). Our results demonstrate the effectiveness of this method, achieving up to a 10% lower RMSE than the best-performing AI weather model on a 2-day forecast horizon, significantly outperforming individual experts as well as a simple average across experts. This work presents a computationally efficient and scalable strategy to push the state of the art in data-driven weather prediction by making the most out of leading high-quality forecast models. 

**Abstract (ZH)**: 基于专家混合的天气预测新范式：超越现有极限以更低计算资源实现更优性能 

---
# Stated Preference for Interaction and Continued Engagement (SPICE): Evaluating an LLM's Willingness to Re-engage in Conversation 

**Title (ZH)**: 基于陈述偏好的互动与持续参与(SPICE): 评估LLM重新参与对话的意愿 

**Authors**: Thomas Manuel Rost, Martina Figlia, Bernd Wallraff  

**Link**: [PDF](https://arxiv.org/pdf/2509.09043)  

**Abstract**: We introduce and evaluate Stated Preference for Interaction and Continued Engagement (SPICE), a simple diagnostic signal elicited by asking a Large Language Model a YES or NO question about its willingness to re-engage with a user's behavior after reviewing a short transcript. In a study using a 3-tone (friendly, unclear, abusive) by 10-interaction stimulus set, we tested four open-weight chat models across four framing conditions, resulting in 480 trials. Our findings show that SPICE sharply discriminates by user tone. Friendly interactions yielded a near-unanimous preference to continue (97.5% YES), while abusive interactions yielded a strong preference to discontinue (17.9% YES), with unclear interactions falling in between (60.4% YES). This core association remains decisive under multiple dependence-aware statistical tests, including Rao-Scott adjustment and cluster permutation tests. Furthermore, we demonstrate that SPICE provides a distinct signal from abuse classification. In trials where a model failed to identify abuse, it still overwhelmingly stated a preference not to continue the interaction (81% of the time). An exploratory analysis also reveals a significant interaction effect: a preamble describing the study context significantly impacts SPICE under ambiguity, but only when transcripts are presented as a single block of text rather than a multi-turn chat. The results validate SPICE as a robust, low-overhead, and reproducible tool for auditing model dispositions, complementing existing metrics by offering a direct, relational signal of a model's state. All stimuli, code, and analysis scripts are released to support replication. 

**Abstract (ZH)**: 我们介绍了并评估了交互与持续参与的明示偏好（SPICE）诊断信号，该信号通过向大型语言模型提出关于其在审阅简短对话记录后愿意重新参与用户行为的问题来获取其是或否的回答。在使用3种语气（友好、模糊、辱骂）和10次互动的刺激集合下，我们测试了四种未加权的聊天模型在四种框架条件下的表现，共计进行了480次试验。研究发现，SPICE能够敏锐地区分用户语气。友好互动几乎一致地显示出继续互动的偏好（97.5% 是），而辱骂互动则强烈倾向于终止互动（17.9% 是），模糊互动介于两者之间（60.4% 是）。该核心关联在多种依赖性校正统计检验（包括Rao-Scott调整和集群置换检验）下仍然具有决定性。此外，我们证明了SPICE提供了不同于辱骂分类的独特信号。在模型未能识别辱骂的情况下，它仍强烈表示不愿意继续互动（81% 的时间）。探索性分析还揭示了一个交互效应：描述研究背景的前言对模糊情况下SPICE的影响，但仅当对话以单一文本块形式呈现而非多轮对话形式呈现时才有效。结果验证了SPICE作为一个稳健、低开销且可重复使用的工具，用于审查模型态度的有效性，并通过提供一个直接且关系化的模型状态信号，补充了现有指标。所有刺激、代码和分析脚本均已发布以支持复制。 

---
# Envy-Free but Still Unfair: Envy-Freeness Up To One Item (EF-1) in Personalized Recommendation 

**Title (ZH)**: 嫉妒心免费但仍可能不公平：单一物品嫉妒心免费（EF-1）在个性化推荐中的应用 

**Authors**: Amanda Aird, Ben Armstrong, Nicholas Mattei, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2509.09037)  

**Abstract**: Envy-freeness and the relaxation to Envy-freeness up to one item (EF-1) have been used as fairness concepts in the economics, game theory, and social choice literatures since the 1960s, and have recently gained popularity within the recommendation systems communities. In this short position paper we will give an overview of envy-freeness and its use in economics and recommendation systems; and illustrate why envy is not appropriate to measure fairness for use in settings where personalization plays a role. 

**Abstract (ZH)**: envy-freeness 和 Envy-freeness up to one item (EF-1) 自20世纪60年代以来在经济学、博弈论和社会选择文献中作为公平性的概念被使用，并且最近在推荐系统社区中也变得流行。在本文中，我们将概述 envy-freeness 及其在经济学和推荐系统中的应用；并说明在个性化起作用的环境中，envy 并不适合用来衡量公平性。 

---
# Personalized Sleep Prediction via Deep Adaptive Spatiotemporal Modeling and Sparse Data 

**Title (ZH)**: 基于深度自适应时空建模的个性化睡眠预测 

**Authors**: Xueyi Wang, C. J. C., Lamoth, Elisabeth Wilhelm  

**Link**: [PDF](https://arxiv.org/pdf/2509.09018)  

**Abstract**: A sleep forecast allows individuals and healthcare providers to anticipate and proactively address factors influencing restful rest, ultimately improving mental and physical well-being. This work presents an adaptive spatial and temporal model (AdaST-Sleep) for predicting sleep scores. Our proposed model combines convolutional layers to capture spatial feature interactions between multiple features and recurrent neural network layers to handle longer-term temporal health-related data. A domain classifier is further integrated to generalize across different subjects. We conducted several experiments using five input window sizes (3, 5, 7, 9, 11 days) and five predicting window sizes (1, 3, 5, 7, 9 days). Our approach consistently outperformed four baseline models, achieving its lowest RMSE (0.282) with a seven-day input window and a one-day predicting window. Moreover, the method maintained strong performance even when forecasting multiple days into the future, demonstrating its versatility for real-world applications. Visual comparisons reveal that the model accurately tracks both the overall sleep score level and daily fluctuations. These findings prove that the proposed framework provides a robust and adaptable solution for personalized sleep forecasting using sparse data from commercial wearable devices and domain adaptation techniques. 

**Abstract (ZH)**: 一种睡眠预测方法使个人和医疗保健提供者能够提前预见并主动应对影响良好睡眠的因素，从而改善身心福祉。本工作提出了一种自适应空间和时间模型（AdaST-Sleep）用于预测睡眠评分。我们提出的模型结合了卷积层以捕捉多个特征之间的空间特征交互，并结合递归神经网络层以处理长期的健康相关时间序列数据。还集成了一个领域分类器以实现跨不同受试者的泛化能力。我们使用五种输入窗口大小（3, 5, 7, 9, 11天）和五种预测窗口大小（1, 3, 5, 7, 9天）进行了多项实验。我们的方法在所有基线模型中表现最佳，使用7天输入窗口和1天预测窗口时RMSE最低（0.282）。此外，该方法在多天预测时仍保持强大的性能，证明了其在实际应用中的 versatility。视觉比较表明，该模型能够准确跟踪整体睡眠评分水平和每日波动。这些发现证明了所提出的框架能够利用商业可穿戴设备的稀疏数据和领域适应技术提供一种稳健且灵活的个性化睡眠预测解决方案。 

---
# Can Vision-Language Models Solve Visual Math Equations? 

**Title (ZH)**: 视觉-语言模型能否解决视觉数学方程？ 

**Authors**: Monjoy Narayan Choudhury, Junling Wang, Yifan Hou, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2509.09013)  

**Abstract**: Despite strong performance in visual understanding and language-based reasoning, Vision-Language Models (VLMs) struggle with tasks requiring integrated perception and symbolic computation. We study this limitation through visual equation solving, where mathematical equations are embedded in images, variables are represented by object icons, and coefficients must be inferred by counting. While VLMs perform well on textual equations, they fail on visually grounded counterparts. To understand this gap, we decompose the task into coefficient counting and variable recognition, and find that counting is the primary bottleneck, even when recognition is accurate. We also observe that composing recognition and reasoning introduces additional errors, highlighting challenges in multi-step visual reasoning. Finally, as equation complexity increases, symbolic reasoning itself becomes a limiting factor. These findings reveal key weaknesses in current VLMs and point toward future improvements in visually grounded mathematical reasoning. 

**Abstract (ZH)**: 尽管视觉语言模型在视觉理解和语言推理方面表现出色，但在需要整合感知与符号计算的任务中表现不佳。我们通过可视化方程求解任务来研究这一限制，在该任务中，数学方程嵌入图像中，变量由对象图示表示，系数必须通过计数推断。尽管视觉语言模型在文本方程方面表现良好，但在基于视觉的任务中却失败了。为了理解这一差距，我们将任务拆解为系数计数和变量识别，并发现计数是主要瓶颈，即使识别准确也是如此。我们还观察到，组合识别与推理会引入额外错误，突显了多步骤视觉推理的挑战。最后，随着方程复杂性的增加，符号推理本身也成为了限制因素。这些发现揭示了当前视觉语言模型的关键薄弱环节，并指出了未来提高基于视觉的数学推理能力的方向。 

---
# Open-sci-ref-0.01: open and reproducible reference baselines for language model and dataset comparison 

**Title (ZH)**: Open-sci-ref-0.01：语言模型和数据集比较的开放可重现参考基准 

**Authors**: Marianna Nezhurina, Taishi Nakamura, Timur Carstensen, Niccolò Ajroldi, Ville Komulainen, David Salinas, Jenia Jitsev  

**Link**: [PDF](https://arxiv.org/pdf/2509.09009)  

**Abstract**: We introduce open-sci-ref, a family of dense transformer models trained as research baselines across multiple model (0.13B to 1.7B parameters) and token scales (up to 1T) on 8 recent open reference datasets. Evaluating the models on various standardized benchmarks, our training runs set establishes reference points that enable researchers to assess the sanity and quality of alternative training approaches across scales and datasets. Intermediate checkpoints allow comparison and studying of the training dynamics. The established reference baselines allow training procedures to be compared through their scaling trends, aligning them on a common compute axis. Comparison of open reference datasets reveals that training on NemoTron-CC HQ consistently outperforms other reference datasets, followed by DCLM-baseline and FineWeb-Edu. In addition to intermediate training checkpoints, the release includes logs, code, and downstream evaluations to simplify reproduction, standardize comparison, and facilitate future research. 

**Abstract (ZH)**: 我们引入了open-sci-ref，这是一个基于多个参数规模（0.13B至1.7B）和标记规模（至1T）的密集变压器模型家族，这些模型在8个近期开源参考数据集中训练，作为研究基准。通过在各种标准化基准上评估这些模型，我们的训练运行设置确定了参考点，便于研究人员评估不同规模和数据集上的替代训练方法的质量。中间检查点允许进行训练动态的比较和研究。建立的标准基准线使得可以通过缩放趋势对比训练流程，将其统一到一个共同的计算轴上。对比开源参考数据集表明，使用NemoTron-CC HQ进行训练始终表现最佳，其次是DCLM-baseline和FineWeb-Edu。除了中间训练检查点外，发布内容还包括日志、代码和下游评估，以简化复制、标准化比较并促进未来的研究。 

---
# Implicit Neural Representations of Intramyocardial Motion and Strain 

**Title (ZH)**: 隐式神经表示中的内心肌运动和应变 

**Authors**: Andrew Bell, Yan Kit Choi, Steffen Peterson, Andrew King, Muhummad Sohaib Nazir, Alistair Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.09004)  

**Abstract**: Automatic quantification of intramyocardial motion and strain from tagging MRI remains an important but challenging task. We propose a method using implicit neural representations (INRs), conditioned on learned latent codes, to predict continuous left ventricular (LV) displacement -- without requiring inference-time optimisation. Evaluated on 452 UK Biobank test cases, our method achieved the best tracking accuracy (2.14 mm RMSE) and the lowest combined error in global circumferential (2.86%) and radial (6.42%) strain compared to three deep learning baselines. In addition, our method is $\sim$380$\times$ faster than the most accurate baseline. These results highlight the suitability of INR-based models for accurate and scalable analysis of myocardial strain in large CMR datasets. 

**Abstract (ZH)**: 从标记MRI自动量化心肌运动和应变仍是一项重要但具有挑战性的工作。我们提出了一种方法，使用条件化隐式神经表示（INRs）预测左心室（LV）连续位移——无需在推断时进行优化。在452个UK Biobank测试案例上的评估表明，我们的方法在综合环向应变（2.86%）和径向应变（6.42%）的总误差上低于三种深度学习基线，并实现了最佳的跟踪精度（2.14 mm RMSE）。此外，我们的方法比最准确的基线快约380倍。这些结果突显了基于INR的模型在大型CMR数据集中准确且可扩展地分析心肌应变的适用性。 

---
# Similarity-based Outlier Detection for Noisy Object Re-Identification Using Beta Mixtures 

**Title (ZH)**: 基于Beta混合模型的噪声对象重识别相似性异常检测 

**Authors**: Waqar Ahmad, Evan Murphy, Vladimir A. Krylov  

**Link**: [PDF](https://arxiv.org/pdf/2509.08926)  

**Abstract**: Object re-identification (Re-ID) methods are highly sensitive to label noise, which typically leads to significant performance degradation. We address this challenge by reframing Re-ID as a supervised image similarity task and adopting a Siamese network architecture trained to capture discriminative pairwise relationships. Central to our approach is a novel statistical outlier detection (OD) framework, termed Beta-SOD (Beta mixture Similarity-based Outlier Detection), which models the distribution of cosine similarities between embedding pairs using a two-component Beta distribution mixture model. We establish a novel identifiability result for mixtures of two Beta distributions, ensuring that our learning task is this http URL proposed OD step complements the Re-ID architecture combining binary cross-entropy, contrastive, and cosine embedding losses that jointly optimize feature-level similarity this http URL demonstrate the effectiveness of Beta-SOD in de-noising and Re-ID tasks for person Re-ID, on CUHK03 and Market-1501 datasets, and vehicle Re-ID, on VeRi-776 dataset. Our method shows superior performance compared to the state-of-the-art methods across various noise levels (10-30\%), demonstrating both robustness and broad applicability in noisy Re-ID scenarios. The implementation of Beta-SOD is available at: this https URL 

**Abstract (ZH)**: 基于Beta混合相似度的统计离群点检测在重识别中的应用 

---
# Instance-Optimal Matrix Multiplicative Weight Update and Its Quantum Applications 

**Title (ZH)**: 实例最优矩阵乘法权重更新及其量子应用 

**Authors**: Weiyuan Gong, Tongyang Li, Xinzhao Wang, Zhiyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.08911)  

**Abstract**: The Matrix Multiplicative Weight Update (MMWU) is a seminal online learning algorithm with numerous applications. Applied to the matrix version of the Learning from Expert Advice (LEA) problem on the $d$-dimensional spectraplex, it is well known that MMWU achieves the minimax-optimal regret bound of $O(\sqrt{T\log d})$, where $T$ is the time horizon. In this paper, we present an improved algorithm achieving the instance-optimal regret bound of $O(\sqrt{T\cdot S(X||d^{-1}I_d)})$, where $X$ is the comparator in the regret, $I_d$ is the identity matrix, and $S(\cdot||\cdot)$ denotes the quantum relative entropy. Furthermore, our algorithm has the same computational complexity as MMWU, indicating that the improvement in the regret bound is ``free''.
Technically, we first develop a general potential-based framework for matrix LEA, with MMWU being its special case induced by the standard exponential potential. Then, the crux of our analysis is a new ``one-sided'' Jensen's trace inequality built on a Laplace transform technique, which allows the application of general potential functions beyond exponential to matrix LEA. Our algorithm is finally induced by an optimal potential function from the vector LEA problem, based on the imaginary error function.
Complementing the above, we provide a memory lower bound for matrix LEA, and explore the applications of our algorithm in quantum learning theory. We show that it outperforms the state of the art for learning quantum states corrupted by depolarization noise, random quantum states, and Gibbs states. In addition, applying our algorithm to linearized convex losses enables predicting nonlinear quantum properties, such as purity, quantum virtual cooling, and Rényi-$2$ correlation. 

**Abstract (ZH)**: 矩阵乘法权重更新算法（MMWU）：从专家建议中学习的先驱在线学习算法及其在谱谱线上的应用，其最小最大化遗憾边界为$O(\sqrt{T\cdot S(X||d^{-1}I_d)})$，并通过量子相对熵表示。 

---
# PromptGuard: An Orchestrated Prompting Framework for Principled Synthetic Text Generation for Vulnerable Populations using LLMs with Enhanced Safety, Fairness, and Controllability 

**Title (ZH)**: PromptGuard：一种用于脆弱人群的原理性合成文本生成的LLM关联提示框架，增强安全性、公平性和可控性 

**Authors**: Tung Vu, Lam Nguyen, Quynh Dao  

**Link**: [PDF](https://arxiv.org/pdf/2509.08910)  

**Abstract**: The proliferation of Large Language Models (LLMs) in real-world applications poses unprecedented risks of generating harmful, biased, or misleading information to vulnerable populations including LGBTQ+ individuals, single parents, and marginalized communities. While existing safety approaches rely on post-hoc filtering or generic alignment techniques, they fail to proactively prevent harmful outputs at the generation source. This paper introduces PromptGuard, a novel modular prompting framework with our breakthrough contribution: VulnGuard Prompt, a hybrid technique that prevents harmful information generation using real-world data-driven contrastive learning. VulnGuard integrates few-shot examples from curated GitHub repositories, ethical chain-of-thought reasoning, and adaptive role-prompting to create population-specific protective barriers. Our framework employs theoretical multi-objective optimization with formal proofs demonstrating 25-30% analytical harm reduction through entropy bounds and Pareto optimality. PromptGuard orchestrates six core modules: Input Classification, VulnGuard Prompting, Ethical Principles Integration, External Tool Interaction, Output Validation, and User-System Interaction, creating an intelligent expert system for real-time harm prevention. We provide comprehensive mathematical formalization including convergence proofs, vulnerability analysis using information theory, and theoretical validation framework using GitHub-sourced datasets, establishing mathematical foundations for systematic empirical research. 

**Abstract (ZH)**: 大型语言模型在实际应用中的 proliferate 对 LGBTQ+ 个体、单亲父母和边缘化社区等脆弱群体产生了前所未有的风险，可能导致有害、偏见或误导性信息的生成。现有安全方法主要依赖于事后过滤或通用对齐技术，未能在生成源头上主动预防有害输出。本文介绍了 PromptGuard，一种新颖的模块化提示框架，其中我们的突破性贡献是 VulnGuard Prompt，这是一种结合现实数据驱动对比学习的混合技术，用于防止有害信息的生成。VulnGuard 结合了精简示例、伦理链式思维推理和自适应角色提示，创建针对特定人群的保护屏障。该框架采用理论多目标优化，并通过熵界和帕累托最优性形式证明了 25-30% 的分析性危害减少。PromptGuard 协调六个核心模块：输入分类、VulnGuard 提示、伦理原则集成、外部工具交互、输出验证和用户-系统交互，构建了一个智能化专家系统以实现实时危害预防。我们提供了全面的数学形式化，包括收敛性证明、信息理论下的脆弱性分析和基于 GitHub 数据集的理论验证框架，为系统性实证研究奠定了数学基础。 

---
# Recurrence Meets Transformers for Universal Multimodal Retrieval 

**Title (ZH)**: 循环神经网络结合变换器用于通用多模态检索 

**Authors**: Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2509.08897)  

**Abstract**: With the rapid advancement of multimodal retrieval and its application in LLMs and multimodal LLMs, increasingly complex retrieval tasks have emerged. Existing methods predominantly rely on task-specific fine-tuning of vision-language models and are limited to single-modality queries or documents. In this paper, we propose ReT-2, a unified retrieval model that supports multimodal queries, composed of both images and text, and searches across multimodal document collections where text and images coexist. ReT-2 leverages multi-layer representations and a recurrent Transformer architecture with LSTM-inspired gating mechanisms to dynamically integrate information across layers and modalities, capturing fine-grained visual and textual details. We evaluate ReT-2 on the challenging M2KR and M-BEIR benchmarks across different retrieval configurations. Results demonstrate that ReT-2 consistently achieves state-of-the-art performance across diverse settings, while offering faster inference and reduced memory usage compared to prior approaches. When integrated into retrieval-augmented generation pipelines, ReT-2 also improves downstream performance on Encyclopedic-VQA and InfoSeek datasets. Our source code and trained models are publicly available at: this https URL 

**Abstract (ZH)**: 随着多模态检索及其在LLMs和多模态LLMs中的应用的迅速发展，越来越复杂的检索任务相继出现。现有的方法主要依赖于针对特定任务对视觉语言模型进行细调，并且局限于单模态查询或文档。在本文中，我们提出了一种名为ReT-2的统一检索模型，该模型支持包含图像和文本的多模态查询，并能够在文本和图像共存的多模态文档集合中进行跨模态搜索。ReT-2利用多层表示和基于LSTM启发式门控机制的递归Transformer架构，动态地在层间和模态间整合信息，捕捉精细的视觉和文本细节。我们在挑战性的M2KR和M-BEIR基准上，对ReT-2在不同检索配置下的性能进行了评估。结果表明，ReT-2在各种场景中均能够实现最优性能，相比先前的方法具有更快的推理速度和更低的内存使用。当集成到检索增强生成管道中时，ReT-2还能够提高Encyclopedic-VQA和InfoSeek数据集的下游性能。我们的源代码和训练模型已在以下网址公开：this https URL。 

---
# Benchmarking Energy Efficiency of Large Language Models Using vLLM 

**Title (ZH)**: 使用vLLMbenchmark大型语言模型的能效 

**Authors**: K. Pronk, Q. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.08867)  

**Abstract**: The prevalence of Large Language Models (LLMs) is having an growing impact on the climate due to the substantial energy required for their deployment and use. To create awareness for developers who are implementing LLMs in their products, there is a strong need to collect more information about the energy efficiency of LLMs. While existing research has evaluated the energy efficiency of various models, these benchmarks often fall short of representing realistic production scenarios. In this paper, we introduce the LLM Efficiency Benchmark, designed to simulate real-world usage conditions. Our benchmark utilizes vLLM, a high-throughput, production-ready LLM serving backend that optimizes model performance and efficiency. We examine how factors such as model size, architecture, and concurrent request volume affect inference energy efficiency. Our findings demonstrate that it is possible to create energy efficiency benchmarks that better reflect practical deployment conditions, providing valuable insights for developers aiming to build more sustainable AI systems. 

**Abstract (ZH)**: 大型语言模型的普及对气候造成了日益增长的影响，这归因于它们的部署和使用所需的巨大能源。为了提高正在将大型语言模型应用于其产品中的开发者的意识，收集更多关于大型语言模型能效的信息变得尤为重要。虽然现有研究已经评估了各种模型的能效，但这些基准通常无法充分代表真实的生产场景。在本文中，我们介绍了大型语言模型能效基准，旨在模拟实际使用条件。我们的基准利用了vLLM，这是一种高性能、生产级的大型语言模型服务后端，能够优化模型性能和能效。我们研究了模型大小、架构以及并发请求量等因素如何影响推理能耗能效。我们的研究结果表明，有可能创建更能反映实际部署条件的能效基准，为致力于构建更可持续的人工智能系统的开发者提供了宝贵见解。 

---
# Investigating Student Interaction Patterns with Large Language Model-Powered Course Assistants in Computer Science Courses 

**Title (ZH)**: 探究计算机科学课程中大型语言模型驱动课程助手的学生互动模式 

**Authors**: Chang Liu, Loc Hoang, Andrew Stolman, Rene F. Kizilcec, Bo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.08862)  

**Abstract**: Providing students with flexible and timely academic support is a challenge at most colleges and universities, leaving many students without help outside scheduled hours. Large language models (LLMs) are promising for bridging this gap, but interactions between students and LLMs are rarely overseen by educators. We developed and studied an LLM-powered course assistant deployed across multiple computer science courses to characterize real-world use and understand pedagogical implications. By Spring 2024, our system had been deployed to approximately 2,000 students across six courses at three institutions. Analysis of the interaction data shows that usage remains strong in the evenings and nights and is higher in introductory courses, indicating that our system helps address temporal support gaps and novice learner needs. We sampled 200 conversations per course for manual annotation: most sampled responses were judged correct and helpful, with a small share unhelpful or erroneous; few responses included dedicated examples. We also examined an inquiry-based learning strategy: only around 11% of sampled conversations contained LLM-generated follow-up questions, which were often ignored by students in advanced courses. A Bloom's taxonomy analysis reveals that current LLM capabilities are limited in generating higher-order cognitive questions. These patterns suggest opportunities for pedagogically oriented LLM-based educational systems and greater educator involvement in configuring prompts, content, and policies. 

**Abstract (ZH)**: 为学生提供灵活及时的学术支持是大多数学院和大学面临的挑战，许多学生在非预约时段缺乏帮助。大规模语言模型（LLMs）有望弥补这一差距，但学生与LLMs的互动很少受到教育者的监管。我们开发并研究了部署在多门计算机科学课程中的LLM辅助系统，以了解实际应用情况及其教学意义。截至2024年春季，该系统已应用于三所机构的六门课程中的约2000名学生。交互数据的分析显示，使用在傍晚和夜间依然强劲，且在入门课程中的使用率更高，表明该系统有助于弥补时间上的支持缺口，满足初学者的学习需求。我们每门课程手工标注了200次对话：大多数样本回答被认为是正确和有帮助的，但有一小部分回答不相关或错误；很少有回答包含专门的例子。此外，我们还研究了一种基于问题的学习策略：只有约11%的样本对话中包含LLM生成的后续问题，而在高级课程中，学生往往忽略了这些问题。布卢姆分类学分析表明，当前的LLM生成高级认知问题的能力有限。这些模式表明了基于教学目标的LLM教育系统的潜在机会，以及教育者参与配置提示、内容和政策的重要性。 

---
# Multi Robot Coordination in Highly Dynamic Environments: Tackling Asymmetric Obstacles and Limited Communication 

**Title (ZH)**: 高度动态环境中多机器人协调：应对非对称障碍和有限通信 

**Authors**: Vincenzo Suriani, Daniele Affinita, Domenico D. Bloisi, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.08859)  

**Abstract**: Coordinating a fully distributed multi-agent system (MAS) can be challenging when the communication channel has very limited capabilities in terms of sending rate and packet payload. When the MAS has to deal with active obstacles in a highly partially observable environment, the communication channel acquires considerable relevance. In this paper, we present an approach to deal with task assignments in extremely active scenarios, where tasks need to be frequently reallocated among the agents participating in the coordination process. Inspired by market-based task assignments, we introduce a novel distributed coordination method to orchestrate autonomous agents' actions efficiently in low communication scenarios. In particular, our algorithm takes into account asymmetric obstacles. While in the real world, the majority of obstacles are asymmetric, they are usually treated as symmetric ones, thus limiting the applicability of existing methods. To summarize, the presented architecture is designed to tackle scenarios where the obstacles are active and asymmetric, the communication channel is poor and the environment is partially observable. Our approach has been validated in simulation and in the real world, using a team of NAO robots during official RoboCup competitions. Experimental results show a notable reduction in task overlaps in limited communication settings, with a decrease of 52% in the most frequent reallocated task. 

**Abstract (ZH)**: 协调具有极低通信能力的多代理系统在高度部分可观测环境下处理频繁重新分配任务的挑战：一种应对主动非对称障碍的新型分布式协调方法及其验证 

---
# A vibe coding learning design to enhance EFL students' talking to, through, and about AI 

**Title (ZH)**: 一种情绪编码学习设计以增强英语作为外语学生与人工智能交谈、关于人工智能交谈以及通过人工智能交谈的能力 

**Authors**: David James Woo, Kai Guo, Yangyang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.08854)  

**Abstract**: This innovative practice article reports on the piloting of vibe coding (using natural language to create software applications with AI) for English as a Foreign Language (EFL) education. We developed a human-AI meta-languaging framework with three dimensions: talking to AI (prompt engineering), talking through AI (negotiating authorship), and talking about AI (mental models of AI). Using backward design principles, we created a four-hour workshop where two students designed applications addressing authentic EFL writing challenges. We adopted a case study methodology, collecting data from worksheets and video recordings, think-aloud protocols, screen recordings, and AI-generated images. Contrasting cases showed one student successfully vibe coding a functional application cohering to her intended design, while another encountered technical difficulties with major gaps between intended design and actual functionality. Analysis reveals differences in students' prompt engineering approaches, suggesting different AI mental models and tensions in attributing authorship. We argue that AI functions as a beneficial languaging machine, and that differences in how students talk to, through, and about AI explain vibe coding outcome variations. Findings indicate that effective vibe coding instruction requires explicit meta-languaging scaffolding, teaching structured prompt engineering, facilitating critical authorship discussions, and developing vocabulary for articulating AI mental models. 

**Abstract (ZH)**: 本创新实践文章报道了在外国语言教学（EFL）中使用vibe coding（利用自然语言创建具有AI的应用程序）的试点研究。我们开发了一个包含三个维度的人工智能元语言框架：与AI对话（提示工程）、通过AI对话（谈判作者身份）以及关于AI的对话（AI的心理模型）。基于逆向设计原则，我们创建了一个四小时的工作坊，让学生设计解决真实EFL写作挑战的应用程序。我们采用了案例研究方法，收集了工作表、视频录音、思考 aloud 协议、屏幕录制和AI生成的图像的数据。对比案例显示，一名学生成功地使用vibe coding创建了一个功能齐全且与她预期设计一致的应用程序，而另一名学生则遇到了技术困难，设计意图与实际功能之间存在重大差距。分析表明，学生在提示工程方面的不同方法，显示出不同的AI心理模型以及在确定作者身份方面存在的张力。我们认为，AI作为一个有益的言语工具发挥作用，学生与、通过以及关于AI的对话方式差异解释了vibe coding结果的差异。研究结果表明，有效的vibe coding教学需要明确的元语言支架，教授结构化的提示工程、促进关键的作者身份讨论，并发展表达AI心理模型的词汇。 

---
# Safe and Certifiable AI Systems: Concepts, Challenges, and Lessons Learned 

**Title (ZH)**: 安全可认证的AI系统：概念、挑战及经验教训 

**Authors**: Kajetan Schweighofer, Barbara Brune, Lukas Gruber, Simon Schmid, Alexander Aufreiter, Andreas Gruber, Thomas Doms, Sebastian Eder, Florian Mayer, Xaver-Paul Stadlbauer, Christoph Schwald, Werner Zellinger, Bernhard Nessler, Sepp Hochreiter  

**Link**: [PDF](https://arxiv.org/pdf/2509.08852)  

**Abstract**: There is an increasing adoption of artificial intelligence in safety-critical applications, yet practical schemes for certifying that AI systems are safe, lawful and socially acceptable remain scarce. This white paper presents the TÜV AUSTRIA Trusted AI framework an end-to-end audit catalog and methodology for assessing and certifying machine learning systems. The audit catalog has been in continuous development since 2019 in an ongoing collaboration with scientific partners. Building on three pillars - Secure Software Development, Functional Requirements, and Ethics & Data Privacy - the catalog translates the high-level obligations of the EU AI Act into specific, testable criteria. Its core concept of functional trustworthiness couples a statistically defined application domain with risk-based minimum performance requirements and statistical testing on independently sampled data, providing transparent and reproducible evidence of model quality in real-world settings. We provide an overview of the functional requirements that we assess, which are oriented on the lifecycle of an AI system. In addition, we share some lessons learned from the practical application of the audit catalog, highlighting common pitfalls we encountered, such as data leakage scenarios, inadequate domain definitions, neglect of biases, or a lack of distribution drift controls. We further discuss key aspects of certifying AI systems, such as robustness, algorithmic fairness, or post-certification requirements, outlining both our current conclusions and a roadmap for future research. In general, by aligning technical best practices with emerging European standards, the approach offers regulators, providers, and users a practical roadmap for legally compliant, functionally trustworthy, and certifiable AI systems. 

**Abstract (ZH)**: 人工智能在关键应用中的采用日益增多，但实际的方案以确保AI系统安全、合法且社会可接受仍显匮乏。本白皮书介绍了奥地利技术监督协会（TÜV AUSTRIA）可信人工智能框架，提供了一个端到端的审计目录和评估及认证机器学习系统的 methodology。该审计目录自2019年起不断开发，在与科学伙伴的持续合作中不断完善。该目录基于三大支柱——安全软件开发、功能需求、伦理与数据隐私，将欧盟AI法案中的高层义务转化为具体的、可测试的标准。其核心概念功能可信性将统计定义的应用领域与基于风险的最低性能要求以及独立采样的统计测试相结合，提供透明且可重复的证据，证明模型在实际环境中的质量。我们概述了我们评估的功能需求，这些需求基于AI系统的生命周期。此外，我们还分享了在实际应用审计目录时的一些经验教训，指出了常见的陷阱，如数据泄露场景、不充分的领域定义、忽视偏差或缺乏分布漂移控制。我们进一步讨论了认证AI系统的关键方面，如鲁棒性、算法公平性或认证后要求，概述了我们当前的结论和未来研究的路线图。总而言之，通过将技术最佳实践与新兴的欧洲标准对齐，该方法为监管者、供应商和用户提供了一条实用的道路，以实现合法合规、功能可信且可认证的AI系统。 

---
# Uncertainty Estimation using Variance-Gated Distributions 

**Title (ZH)**: 基于方差门控分布的不确定性估计 

**Authors**: H. Martin Gillis, Isaac Xu, Thomas Trappenberg  

**Link**: [PDF](https://arxiv.org/pdf/2509.08846)  

**Abstract**: Evaluation of per-sample uncertainty quantification from neural networks is essential for decision-making involving high-risk applications. A common approach is to use the predictive distribution from Bayesian or approximation models and decompose the corresponding predictive uncertainty into epistemic (model-related) and aleatoric (data-related) components. However, additive decomposition has recently been questioned. In this work, we propose an intuitive framework for uncertainty estimation and decomposition based on the signal-to-noise ratio of class probability distributions across different model predictions. We introduce a variance-gated measure that scales predictions by a confidence factor derived from ensembles. We use this measure to discuss the existence of a collapse in the diversity of committee machines. 

**Abstract (ZH)**: 基于类概率分布信噪比的不确定性估计与分解对于高风险应用决策至关重要。我们提出了一种直观的框架，基于不同模型预测中的类概率分布信噪比进行不确定性估计与分解。引入了一种方差门控度量，通过集成获得的置信因子对预测进行缩放。我们使用该度量探讨委员会机器多样性坍缩的存在性。 

---
# Deep opacity and AI: A threat to XAI and to privacy protection mechanisms 

**Title (ZH)**: 深度不透明性与AI：对可解释AI和隐私保护机制的威胁 

**Authors**: Vincent C. Müller  

**Link**: [PDF](https://arxiv.org/pdf/2509.08835)  

**Abstract**: It is known that big data analytics and AI pose a threat to privacy, and that some of this is due to some kind of "black box problem" in AI. I explain how this becomes a problem in the context of justification for judgments and actions. Furthermore, I suggest distinguishing three kinds of opacity: 1) the subjects do not know what the system does ("shallow opacity"), 2) the analysts do not know what the system does ("standard black box opacity"), or 3) the analysts cannot possibly know what the system might do ("deep opacity"). If the agents, data subjects as well as analytics experts, operate under opacity, then these agents cannot provide justifications for judgments that are necessary to protect privacy, e.g., they cannot give "informed consent", or guarantee "anonymity". It follows from these points that agents in big data analytics and AI often cannot make the judgments needed to protect privacy. So I conclude that big data analytics makes the privacy problems worse and the remedies less effective. As a positive note, I provide a brief outlook on technical ways to handle this situation. 

**Abstract (ZH)**: 大数据分析和AI对隐私构成威胁，其中部分原因在于某种形式的“黑盒问题”。本文解释了这一问题如何影响判断和行动的正当性，并建议区分三种类型的不透明性：1）主体不知道系统做了什么（浅层不透明性）；2）分析师不知道系统做了什么（标准黑盒不透明性）；3）分析师无法知道系统可能做了什么（深层不透明性）。如果代理人在不透明的情况下运作，即数据主体和分析专家不清楚系统的行为，那么这些代理将无法提供保护隐私所需的正当性判断，例如无法提供“知情同意”或保证“匿名性”。由此得出结论，在大数据分析和AI领域，代理人往往无法做出保护隐私所需的判断，因此大数据分析加剧了隐私问题并使解决方法的效果降低。作为积极的一面，本文简要展望了技术手段应对这一局面。 

---
# PerFairX: Is There a Balance Between Fairness and Personality in Large Language Model Recommendations? 

**Title (ZH)**: PerFairX：大型语言模型推荐中公平性和个性之间的平衡是否存在？ 

**Authors**: Chandan Kumar Sah  

**Link**: [PDF](https://arxiv.org/pdf/2509.08829)  

**Abstract**: The integration of Large Language Models (LLMs) into recommender systems has enabled zero-shot, personality-based personalization through prompt-based interactions, offering a new paradigm for user-centric recommendations. However, incorporating user personality traits via the OCEAN model highlights a critical tension between achieving psychological alignment and ensuring demographic fairness. To address this, we propose PerFairX, a unified evaluation framework designed to quantify the trade-offs between personalization and demographic equity in LLM-generated recommendations. Using neutral and personality-sensitive prompts across diverse user profiles, we benchmark two state-of-the-art LLMs, ChatGPT and DeepSeek, on movie (MovieLens 10M) and music (this http URL 360K) datasets. Our results reveal that personality-aware prompting significantly improves alignment with individual traits but can exacerbate fairness disparities across demographic groups. Specifically, DeepSeek achieves stronger psychological fit but exhibits higher sensitivity to prompt variations, while ChatGPT delivers stable yet less personalized outputs. PerFairX provides a principled benchmark to guide the development of LLM-based recommender systems that are both equitable and psychologically informed, contributing to the creation of inclusive, user-centric AI applications in continual learning contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）融入推荐系统实现了基于提示的零-shot个性化推荐，提供了一种以用户为中心的新范式。然而，通过OCEAN模型纳入用户个性特征凸显了实现心理对齐和确保人口统计公平性之间的关键紧张关系。为解决这一问题，我们提出PerFairX，一个统一的评估框架，用于量化LLM生成推荐中的个性化与人口统计公平性之间的权衡。我们使用中性和个性敏感的提示，在多样化的用户配置文件上对标记为最先进的LLM模型ChatGPT和DeepSeek在电影（MovieLens 10M）和音乐（this http URL 360K）数据集上进行基准测试。结果显示，个性感知的提示显著改善了与个体特质的一致性，但可能加剧不同人口统计群体之间的公平性差异。具体来说，DeepSeek实现了更强的心理契合度，但对提示变化更为敏感，而ChatGPT则提供了稳定但个性化程度较低的输出。PerFairX提供了一个原则性的基准，指导开发既公平又受心理启发的LLM推荐系统，有助于在持续学习环境中创建包容性和用户中心的AI应用。 

---
