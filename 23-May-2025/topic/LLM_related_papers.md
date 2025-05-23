# Human-like Semantic Navigation for Autonomous Driving using Knowledge Representation and Large Language Models 

**Title (ZH)**: 基于知识表示和大型语言模型的人类级语义导航自主驾驶 

**Authors**: Augusto Luis Ballardini, Miguel Ángel Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16498)  

**Abstract**: Achieving full automation in self-driving vehicles remains a challenge, especially in dynamic urban environments where navigation requires real-time adaptability. Existing systems struggle to handle navigation plans when faced with unpredictable changes in road layouts, spontaneous detours, or missing map data, due to their heavy reliance on predefined cartographic information. In this work, we explore the use of Large Language Models to generate Answer Set Programming rules by translating informal navigation instructions into structured, logic-based reasoning. ASP provides non-monotonic reasoning, allowing autonomous vehicles to adapt to evolving scenarios without relying on predefined maps. We present an experimental evaluation in which LLMs generate ASP constraints that encode real-world urban driving logic into a formal knowledge representation. By automating the translation of informal navigation instructions into logical rules, our method improves adaptability and explainability in autonomous navigation. Results show that LLM-driven ASP rule generation supports semantic-based decision-making, offering an explainable framework for dynamic navigation planning that aligns closely with how humans communicate navigational intent. 

**Abstract (ZH)**: 在自驾车中实现完全自动化仍是一个挑战，特别是在需要实时适应性的动态城市环境中。现有系统难以处理不可预测的道路布局变化、突发的路线绕行或缺失的地图数据，因为它们高度依赖预定义的地图信息。在此工作中，我们探索使用大型语言模型通过将非正式的导航指令翻译为结构化的逻辑推理规则来生成Answer Set Programming（ASP）规则。ASP提供了非单调推理能力，使自动驾驶车辆能够在无需依赖预定义地图的情况下适应不断变化的场景。我们通过实验评估，展示了LLM生成的ASP约束如何将现实世界的城市驾驶逻辑编码为形式化的知识表示。通过自动化非正式导航指令到逻辑规则的翻译，我们的方法提高了自主导航的适应性和可解释性。结果表明，由LLM驱动的ASP规则生成支持基于语义的决策制定，提供了一个与人类导航意图交流紧密相连的可解释动态导航规划框架。 

---
# X-MAS: Towards Building Multi-Agent Systems with Heterogeneous LLMs 

**Title (ZH)**: X-MAS: 向构建异构大语言模型多智能体系统方向努力 

**Authors**: Rui Ye, Xiangrui Liu, Qimin Wu, Xianghe Pang, Zhenfei Yin, Lei Bai, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16997)  

**Abstract**: LLM-based multi-agent systems (MAS) extend the capabilities of single LLMs by enabling cooperation among multiple specialized agents. However, most existing MAS frameworks rely on a single LLM to drive all agents, constraining the system's intelligence to the limit of that model. This paper explores the paradigm of heterogeneous LLM-driven MAS (X-MAS), where agents are powered by diverse LLMs, elevating the system's potential to the collective intelligence of diverse LLMs. We introduce X-MAS-Bench, a comprehensive testbed designed to evaluate the performance of various LLMs across different domains and MAS-related functions. As an extensive empirical study, we assess 27 LLMs across 5 domains (encompassing 21 test sets) and 5 functions, conducting over 1.7 million evaluations to identify optimal model selections for each domain-function combination. Building on these findings, we demonstrate that transitioning from homogeneous to heterogeneous LLM-driven MAS can significantly enhance system performance without requiring structural redesign. Specifically, in a chatbot-only MAS scenario, the heterogeneous configuration yields up to 8.4\% performance improvement on the MATH dataset. In a mixed chatbot-reasoner scenario, the heterogeneous MAS could achieve a remarkable 47\% performance boost on the AIME dataset. Our results underscore the transformative potential of heterogeneous LLMs in MAS, highlighting a promising avenue for advancing scalable, collaborative AI systems. 

**Abstract (ZH)**: 基于LLM的异构多智能体系统（X-MAS）：提升多智能体系统潜力的异构LLM驱动范式 

---
# Beyond Correlation: Towards Causal Large Language Model Agents in Biomedicine 

**Title (ZH)**: 超越相关性：迈向生物医学中的因果大型语言模型代理 

**Authors**: Adib Bazgir, Amir Habibdoust Lafmajani, Yuwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16982)  

**Abstract**: Large Language Models (LLMs) show promise in biomedicine but lack true causal understanding, relying instead on correlations. This paper envisions causal LLM agents that integrate multimodal data (text, images, genomics, etc.) and perform intervention-based reasoning to infer cause-and-effect. Addressing this requires overcoming key challenges: designing safe, controllable agentic frameworks; developing rigorous benchmarks for causal evaluation; integrating heterogeneous data sources; and synergistically combining LLMs with structured knowledge (KGs) and formal causal inference tools. Such agents could unlock transformative opportunities, including accelerating drug discovery through automated hypothesis generation and simulation, enabling personalized medicine through patient-specific causal models. This research agenda aims to foster interdisciplinary efforts, bridging causal concepts and foundation models to develop reliable AI partners for biomedical progress. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在生物医学领域展现出潜力，但缺乏真正的因果理解，而是依赖于相关性。本文构想了集成多模态数据（文本、图像、基因组等）并进行干预推理以推断因果关系的因果LLM代理。实现这一目标需要克服关键挑战：设计安全可控的代理框架；开发严格的因果评估基准；整合异构数据源；以及协同结合LLMs与结构化知识（KGs）和形式化的因果推理工具。这些代理有望解锁变革性机会，包括通过自动化假设生成和模拟加速药物发现，以及通过患者特异性因果模型实现个性化医疗。这一研究议程旨在促进跨学科努力，将因果概念与基础模型结合起来，开发可信赖的AI伙伴以促进生物医学的进步。 

---
# Know the Ropes: A Heuristic Strategy for LLM-based Multi-Agent System Design 

**Title (ZH)**: 了解关键环节：一种基于LLM的多 Agents 系统设计的启发式策略 

**Authors**: Zhenkun Li, Lingyao Li, Shuhang Lin, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16979)  

**Abstract**: Single-agent LLMs hit hard limits--finite context, role overload, and brittle domain transfer. Conventional multi-agent fixes soften those edges yet expose fresh pains: ill-posed decompositions, fuzzy contracts, and verification overhead that blunts the gains. We therefore present Know-The-Ropes (KtR), a framework that converts domain priors into an algorithmic blueprint hierarchy, in which tasks are recursively split into typed, controller-mediated subtasks, each solved zero-shot or with the lightest viable boost (e.g., chain-of-thought, micro-tune, self-check). Grounded in the No-Free-Lunch theorem, KtR trades the chase for a universal prompt for disciplined decomposition. On the Knapsack problem (3-8 items), three GPT-4o-mini agents raise accuracy from 3% zero-shot to 95% on size-5 instances after patching a single bottleneck agent. On the tougher Task-Assignment problem (6-15 jobs), a six-agent o3-mini blueprint hits 100% up to size 10 and 84% on sizes 13-15, versus 11% zero-shot. Algorithm-aware decomposition plus targeted augmentation thus turns modest models into reliable collaborators--no ever-larger monoliths required. 

**Abstract (ZH)**: Single-agent LLMs碰触硬极限——有限语境、角色过载和脆弱的知识域迁移。传统的多agent解决方案缓解了这些限制，但也暴露出新的问题：含糊的细分、模糊的协议和验证开销，这些都削弱了收益。因此，我们提出了Know-The-Ropes (KtR)框架，该框架将先验知识转化为分层算法蓝图，任务递归拆分为类型化、控制器中介的子任务，每个子任务零样本解决或仅以最小可行增强（例如，思考链、微量调整、自我检查）解决。基于No-Free-Lunch定理，KtR交易了对通用提示的追求，以实现有序的细分。在背包问题（3-8项）中，三个GPT-4o-mini代理在修补了一个瓶颈代理后，将零样本准确率从3%提高到大小为5的实例的95%。在更复杂的任务分配问题（6-15项工作）中，六代理o3-mini蓝图在大小为10时达到100%，在大小13-15时达到84%，而零样本准确率为11%。因此，算法意识的细分加上有针对性的增强将朴素模型转变为可靠的合作者——无需更大的单一实体。 

---
# HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation 

**Title (ZH)**: HyGenar: 由大规模语言模型驱动的混合遗传算法用于少量样本语法规则生成 

**Authors**: Weizhi Tang, Yixuan Li, Chris Sypherd, Elizabeth Polgreen, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2505.16978)  

**Abstract**: Grammar plays a critical role in natural language processing and text/code generation by enabling the definition of syntax, the creation of parsers, and guiding structured outputs. Although large language models (LLMs) demonstrate impressive capabilities across domains, their ability to infer and generate grammars has not yet been thoroughly explored. In this paper, we aim to study and improve the ability of LLMs for few-shot grammar generation, where grammars are inferred from sets of a small number of positive and negative examples and generated in Backus-Naur Form. To explore this, we introduced a novel dataset comprising 540 structured grammar generation challenges, devised 6 metrics, and evaluated 8 various LLMs against it. Our findings reveal that existing LLMs perform sub-optimally in grammar generation. To address this, we propose an LLM-driven hybrid genetic algorithm, namely HyGenar, to optimize grammar generation. HyGenar achieves substantial improvements in both the syntactic and semantic correctness of generated grammars across LLMs. 

**Abstract (ZH)**: 语法在自然语言处理和文本/代码生成中扮演着关键角色，它使得语法定义、解析器创建以及指导结构化输出成为可能。尽管大规模语言模型（LLMs）在各个领域展现出令人印象深刻的性能，但它们推断和生成语法的能力尚未得到充分探索。在本文中，我们旨在研究和提高LLMs在少样本语法生成方面的能力，其中语法是从少量正负例中推断并以Backus-Naur形式生成的。为探索这一领域，我们引入了一个包含540个结构化语法生成挑战的新数据集，制定了6个评估指标，并将8种不同的LLMs进行了评估。我们的发现表明，现有LLMs在语法生成方面表现欠佳。为此，我们提出了一种基于LLM的混合遗传算法，即HyGenar，以优化语法生成。HyGenar在LLMs生成的语法的句法和语义正确性方面取得了显著改善。 

---
# AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios 

**Title (ZH)**: AGENTIF: 在代理场景下大型语言模型指令跟随能力的基准评估 

**Authors**: Yunjia Qi, Hao Peng, Xiaozhi Wang, Amy Xin, Youfeng Liu, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16944)  

**Abstract**: Large Language Models (LLMs) have demonstrated advanced capabilities in real-world agentic applications. Growing research efforts aim to develop LLM-based agents to address practical demands, introducing a new challenge: agentic scenarios often involve lengthy instructions with complex constraints, such as extended system prompts and detailed tool specifications. While adherence to such instructions is crucial for agentic applications, whether LLMs can reliably follow them remains underexplored. In this paper, we introduce AgentIF, the first benchmark for systematically evaluating LLM instruction following ability in agentic scenarios. AgentIF features three key characteristics: (1) Realistic, constructed from 50 real-world agentic applications. (2) Long, averaging 1,723 words with a maximum of 15,630 words. (3) Complex, averaging 11.9 constraints per instruction, covering diverse constraint types, such as tool specifications and condition constraints. To construct AgentIF, we collect 707 human-annotated instructions across 50 agentic tasks from industrial application agents and open-source agentic systems. For each instruction, we annotate the associated constraints and corresponding evaluation metrics, including code-based evaluation, LLM-based evaluation, and hybrid code-LLM evaluation. We use AgentIF to systematically evaluate existing advanced LLMs. We observe that current models generally perform poorly, especially in handling complex constraint structures and tool specifications. We further conduct error analysis and analytical experiments on instruction length and meta constraints, providing some findings about the failure modes of existing LLMs. We have released the code and data to facilitate future research. 

**Abstract (ZH)**: 大型语言模型（LLMs）在现实世界的代理应用中展现了先进的能力。不断增长的研究努力旨在开发基于LLM的代理以应对实际需求，这引入了一个新的挑战：代理场景通常涉及长篇复杂的指令，如扩展系统提示和详细的工具规范。尽管遵守这些指令对于代理应用至关重要，但LLMs能否可靠地遵循它们仍是一个未被充分探索的问题。在本文中，我们介绍了AgentIF，这是首个系统性评估LLM指令遵循能力的基准。AgentIF具有三个关键特征：(1) 现实性强，来自50个真实的代理应用。(2) 长度长，平均1,723字，最多15,630字。(3) 复杂性强，平均每条指令包含11.9个约束，涵盖了不同类型，如工具规范和条件约束。为构建AgentIF，我们收集了来自50个代理任务的707条人类注释的指令，这些任务来自工业应用代理和开源代理系统。对于每条指令，我们标注了相关的约束及其对应的评估指标，包括代码评估、基于LLM的评估以及代码-LLM混合评估。我们使用AgentIF系统性评估现有的高级LLM。我们发现当前模型在这类复杂约束结构和工具规范的处理方面表现较差。我们进一步对指令长度和元约束进行错误分析和实验性研究，揭示了一些关于现有LLM失败模式的发现。我们已公开了代码和数据，以促进未来的研究。 

---
# MCP-RADAR: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models 

**Title (ZH)**: MCP-RADAR：评估大型语言模型工具使用能力的多维度基准 

**Authors**: Xuanqi Gao, Siyi Xie, Juan Zhai, Shqing Ma, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16700)  

**Abstract**: As Large Language Models (LLMs) evolve from passive text generators to active reasoning agents capable of tool interaction, the Model Context Protocol (MCP) has emerged as a standardized framework for dynamic tool discovery and orchestration. Despite widespread industry adoption, existing evaluation methodologies fail to adequately assess tool utilization capabilities within this new paradigm. This paper introduces MCP-RADAR, the first comprehensive benchmark specifically designed to evaluate LLM performance in the MCP framework through a novel five-dimensional approach measuring: answer accuracy, tool selection efficiency, computational resource efficiency, parameter construction accuracy, and execution speed. Unlike conventional benchmarks that rely on subjective human evaluations or binary success metrics, MCP-RADAR employs objective, quantifiable measurements across multiple task domains including software engineering, mathematical reasoning, and general problem-solving. Our evaluations of leading commercial and open-source LLMs reveal distinctive capability profiles with significant trade-offs between accuracy, efficiency, and speed, challenging traditional single-metric performance rankings. Besides, we provide valuable guidance for developers to optimize their tools for maximum model compatibility and effectiveness. While focused on MCP due to its standardized approach, our methodology remains applicable across all LLM agent tool integration frameworks, providing valuable insights for both LLM developers and tool creators to optimize the entire LLM-tool interaction ecosystem. The implementation, configurations, and datasets used in our evaluation are publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）从被动文本生成器进化为能够进行工具交互的主动推理代理，模型上下文协议（MCP）作为一种标准化框架，用于动态工具发现和编排，由此而生。尽管MCP在工业界得到了广泛应用，但现有的评估方法无法充分评估这一新范式下的工具利用能力。本文介绍了MCP-RADAR，这是第一个专为评估MCP框架下LLM性能而设计的全面基准，采用新颖的五维度方法测量：答案准确性、工具选择效率、计算资源效率、参数构建准确性和执行速度。不同于依赖主观人类评估或二元成功标准的传统基准，MCP-RADAR在软件工程、数学推理和一般问题解决等多个任务领域采用客观、可量化的评估指标。我们的评估表明，主流商用和开源LLM在准确度、效率和速度之间存在权衡，挑战了传统的单一指标性能排名。此外，我们为开发者提供了优化工具以最大化模型兼容性和有效性的宝贵建议。由于专注于MCP的标准化方法，我们的方法在所有LLM代理工具集成框架中都是适用的，为LLM开发者和工具创建者提供了优化整个LLM-工具交互生态系统的重要见解。我们在评估中使用的实现、配置和数据集可在此网址访问。 

---
# ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming 

**Title (ZH)**: 详述：人类与大语言模型在编程竞赛中的全面基准测试 

**Authors**: Xinwei Yang, Zhaofeng Liu, Chen Huang, Jiashuai Zhang, Tong Zhang, Yifan Zhang, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16667)  

**Abstract**: While recent research increasingly emphasizes the value of human-LLM collaboration in competitive programming and proposes numerous empirical methods, a comprehensive understanding remains elusive due to the fragmented nature of existing studies and their use of diverse, application-specific human feedback. Thus, our work serves a three-fold purpose: First, we present the first taxonomy of human feedback consolidating the entire programming process, which promotes fine-grained evaluation. Second, we introduce ELABORATIONSET, a novel programming dataset specifically designed for human-LLM collaboration, meticulously annotated to enable large-scale simulated human feedback and facilitate costeffective real human interaction studies. Third, we introduce ELABORATION, a novel benchmark to facilitate a thorough assessment of human-LLM competitive programming. With ELABORATION, we pinpoint strengthes and weaknesses of existing methods, thereby setting the foundation for future improvement. Our code and dataset are available at this https URL 

**Abstract (ZH)**: 近年来，越来越多的研究强调人类与大模型合作在竞争编程中的价值，并提出了多种实证方法，但由于现有研究零碎且使用多样化的应用特定的人类反馈，全面理解仍不明晰。因此，我们的工作具有三重目的：首先，我们提出了首个涵盖整个编程过程的人类反馈分类法，促进精细评估。其次，我们引入了ELABORATIONSET，这是一个专门为人类与大模型合作设计的新颖编程数据集，详细标注以支持大规模模拟人类反馈并促进经济高效的真人交互研究。第三，我们引入了ELABORATION，这是一个新颖的基准工具，用于促进竞争编程中的人类与大模型系统的全面评估。通过ELABORATION，我们明确了现有方法的强项和弱点，为未来改进奠定了基础。我们的代码和数据集可在以下网址获取。 

---
# SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving 

**Title (ZH)**: SMART: 自生成和自验证多维度评估以检验LLMs的数学问题解决能力 

**Authors**: Yujie Hou, Ting Zhang, Mei Wang, Xuetao Ma, Hu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16646)  

**Abstract**: Large Language Models have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine mathematical reasoning or superficial pattern recognition. Common evaluation metrics, such as final answer accuracy, fail to disentangle the underlying competencies involved, offering limited diagnostic value. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework. SMART decomposes mathematical problem solving into four distinct dimensions: understanding, reasoning, arithmetic, and reflection \& refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. Crucially, SMART integrates an automated self-generating and self-validating mechanism to produce and verify benchmark data, ensuring both scalability and reliability. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings demonstrate the inadequacy of final answer accuracy as a sole metric and motivate a new holistic metric to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance. 

**Abstract (ZH)**: 大型语言模型在多种数学基准测试中取得了显著成果，但对其成功是否真实反映数学推理能力而非表面模式识别存有疑虑。常用的评估指标，如最终答案准确性，无法区分潜在的能力，提供有限的诊断价值。为解决这些局限性，我们引入了SMART：一个自我生成和自我验证的多维度评估框架。SMART将数学问题解决分解为四个独立维度：理解、推理、算术和反思与完善。每个维度通过定制任务独立评估，使LML的行为分析具有可解释性和精细度。最关键的是，SMART整合了自动自我生成和自我验证机制，以生成和验证基准数据，确保了规模性和可靠性。我们将SMART应用于21个最先进的开源和闭源LML，揭示了它们在不同维度上的能力差异。我们的研究结果表明，仅依靠最终答案准确性作为单一指标的不足，并激发了新的综合性指标来更好地捕捉真正的问题解决能力。代码和基准数据将在接收后发布。 

---
# Advancing the Scientific Method with Large Language Models: From Hypothesis to Discovery 

**Title (ZH)**: 运用大型语言模型推动科学研究方法：从假设到发现 

**Authors**: Yanbo Zhang, Sumeer A. Khan, Adnan Mahmud, Huck Yang, Alexander Lavin, Michael Levin, Jeremy Frey, Jared Dunnmon, James Evans, Alan Bundy, Saso Dzeroski, Jesper Tegner, Hector Zenil  

**Link**: [PDF](https://arxiv.org/pdf/2505.16477)  

**Abstract**: With recent Nobel Prizes recognising AI contributions to science, Large Language Models (LLMs) are transforming scientific research by enhancing productivity and reshaping the scientific method. LLMs are now involved in experimental design, data analysis, and workflows, particularly in chemistry and biology. However, challenges such as hallucinations and reliability persist. In this contribution, we review how Large Language Models (LLMs) are redefining the scientific method and explore their potential applications across different stages of the scientific cycle, from hypothesis testing to discovery. We conclude that, for LLMs to serve as relevant and effective creative engines and productivity enhancers, their deep integration into all steps of the scientific process should be pursued in collaboration and alignment with human scientific goals, with clear evaluation metrics. The transition to AI-driven science raises ethical questions about creativity, oversight, and responsibility. With careful guidance, LLMs could evolve into creative engines, driving transformative breakthroughs across scientific disciplines responsibly and effectively. However, the scientific community must also decide how much it leaves to LLMs to drive science, even when associations with 'reasoning', mostly currently undeserved, are made in exchange for the potential to explore hypothesis and solution regions that might otherwise remain unexplored by human exploration alone. 

**Abstract (ZH)**: 近年来，诺贝尔奖认可了AI对科学的贡献，大型语言模型（LLMs）正在通过提升生产力和重塑科学方法来革新科学研究。LLMs现已被应用于实验设计、数据分析和工作流程中，特别是在化学和生物学领域。然而，幻觉和可靠性等问题仍然存在。本文回顾了大型语言模型如何重新定义科学方法，并探讨了它们在科学周期各个阶段的潜在应用。我们得出结论，为了使LLMs成为相关且有效的创造力引擎和生产力提升工具，需要在与人类科学目标合作和准确定位的背景下深入整合到科学研究的所有步骤中，并采用明确的评估指标。向AI驱动的科学研究过渡引发了关于创造力、监管和责任的伦理问题。在谨慎引导下，LLMs可以演变为创造力引擎，负责任且有效推动跨学科的变革性突破。然而，科学社区还必须决定在赋权LLMs驱动科学时保留多大程度的自主权，即使会因为所谓的“推理”与人类独自探索可能未被发现的假说和解决方案区域相比而产生潜在的关联。 

---
# ReflectEvo: Improving Meta Introspection of Small LLMs by Learning Self-Reflection 

**Title (ZH)**: ReflectEvo: 改进小规模LLM元内省能力的学习自我反思方法 

**Authors**: Jiaqi Li, Xinyi Dong, Yang Liu, Zhizhuo Yang, Quansen Wang, Xiaobo Wang, SongChun Zhu, Zixia Jia, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.16475)  

**Abstract**: We present a novel pipeline, ReflectEvo, to demonstrate that small language models (SLMs) can enhance meta introspection through reflection learning. This process iteratively generates self-reflection for self-training, fostering a continuous and self-evolving process. Leveraging this pipeline, we construct ReflectEvo-460k, a large-scale, comprehensive, self-generated reflection dataset with broadened instructions and diverse multi-domain tasks. Building upon this dataset, we demonstrate the effectiveness of reflection learning to improve SLMs' reasoning abilities using SFT and DPO with remarkable performance, substantially boosting Llama-3 from 52.4% to 71.2% and Mistral from 44.4% to 71.1%. It validates that ReflectEvo can rival or even surpass the reasoning capability of the three prominent open-sourced models on BIG-bench without distillation from superior models or fine-grained human annotation. We further conduct a deeper analysis of the high quality of self-generated reflections and their impact on error localization and correction. Our work highlights the potential of continuously enhancing the reasoning performance of SLMs through iterative reflection learning in the long run. 

**Abstract (ZH)**: 我们提出了一种新颖的工作流程ReflectEvo，以展示小型语言模型(SLMs)可以通过反思学习增强元反省的能力。该过程通过迭代生成自我反省以促进自我训练，进而形成一个持续的、自我演化的过程。利用这一工作流程，我们构建了ReflectEvo-460k，这是一个大规模、全面、自我生成的反思数据集，包含了广泛的任务指令和多种领域任务。基于此数据集，我们利用SFT和DPO展示了反思学习在提高SLMs推理能力方面的有效性，显著提升了Llama-3从52.4%到71.2%，Mistral从44.4%到71.1%。这验证了ReflectEvo在无需来自更优秀的模型蒸馏或细粒度的人工注释的情况下，能够与三大开源模型在BIG-bench上的推理能力相媲美甚至超越。我们进一步分析了自我生成反思的高质量及其对错误定位和修正的影响。我们的工作突显了长期通过迭代反思学习持续提升SLMs推理性能的潜力。 

---
# MMMR: Benchmarking Massive Multi-Modal Reasoning Tasks 

**Title (ZH)**: MMMR：大规模多模态推理任务benchmarking 

**Authors**: Guiyao Tie, Xueyang Zhou, Tianhe Gu, Ruihang Zhang, Chaoran Hu, Sizhe Zhang, Mengqu Sun, Yan Zhang, Pan Zhou, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16459)  

**Abstract**: Recent advances in Multi-Modal Large Language Models (MLLMs) have enabled unified processing of language, vision, and structured inputs, opening the door to complex tasks such as logical deduction, spatial reasoning, and scientific analysis. Despite their promise, the reasoning capabilities of MLLMs, particularly those augmented with intermediate thinking traces (MLLMs-T), remain poorly understood and lack standardized evaluation benchmarks. Existing work focuses primarily on perception or final answer correctness, offering limited insight into how models reason or fail across modalities. To address this gap, we introduce the MMMR, a new benchmark designed to rigorously evaluate multi-modal reasoning with explicit thinking. The MMMR comprises 1) a high-difficulty dataset of 1,083 questions spanning six diverse reasoning types with symbolic depth and multi-hop demands and 2) a modular Reasoning Trace Evaluation Pipeline (RTEP) for assessing reasoning quality beyond accuracy through metrics like relevance, consistency, and structured error annotations. Empirical results show that MLLMs-T overall outperform non-thinking counterparts, but even top models like Claude-3.7-Sonnet and Gemini-2.5 Pro suffer from reasoning pathologies such as inconsistency and overthinking. This benchmark reveals persistent gaps between accuracy and reasoning quality and provides an actionable evaluation pipeline for future model development. Overall, the MMMR offers a scalable foundation for evaluating, comparing, and improving the next generation of multi-modal reasoning systems. 

**Abstract (ZH)**: Recent advances in 多模态大型语言模型 (MLLMs) 使统一处理语言、视觉和结构化输入成为可能，开启了逻辑推理、空间推理和科学分析等复杂任务的大门。尽管前景广阔，但特别是配备了中间推理痕迹 (MLLMs-T) 的 MLLMs 的推理能力仍缺乏理解，缺少标准化评估基准。现有工作主要集中在感知或最终答案的正确性上，这为了解模型在不同模态下的推理或失败提供有限的见解。为了解决这一差距，我们引入了 MMMR，一种新的基准，旨在通过明确的思考进行多模态推理的严格评估。MMMR 包括 1) 一个高难度数据集，包含 1,083 个跨六个不同推理类型的复杂问题，具有符号深度和多跳需求，以及 2) 一种模块化的推理痕迹评估流水线 (RTEP)，通过相关性、连贯性和结构化错误注释等指标来评估推理质量，而不仅仅是准确性。实验结果表明，MLLMs-T 整体上优于非思考模型，但即使是顶级模型如 Claude-3.7-Sonnet 和 Gemini-2.5 Pro 也存在推理路径障碍，如连贯性差和过度推理。该基准揭示了准确性与推理质量之间的持续差距，并为未来模型开发提供了一套可操作的评估流水线。总体而言，MMMR 为评估、比较和改进下一代多模态推理系统提供了可扩展的基础。 

---
# Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning 

**Title (ZH)**: 激励双重过程思考以实现高效大型语言模型推理 

**Authors**: Xiaoxue Cheng, Junyi Li, Zhenduo Zhang, Xinyu Tang, Wayne Xin Zhao, Xinyu Kong, Zhiqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16315)  

**Abstract**: Large reasoning models (LRMs) have demonstrated strong performance on complex reasoning tasks, but often suffer from overthinking, generating redundant content regardless of task difficulty. Inspired by the dual process theory in cognitive science, we propose Adaptive Cognition Policy Optimization (ACPO), a reinforcement learning framework that enables LRMs to achieve efficient reasoning through adaptive cognitive allocation and dynamic system switch. ACPO incorporates two key components: (1) introducing system-aware reasoning tokens to explicitly represent the thinking modes thereby making the model's cognitive process transparent, and (2) integrating online difficulty estimation and token length budget to guide adaptive system switch and reasoning during reinforcement learning. To this end, we propose a two-stage training strategy. The first stage begins with supervised fine-tuning to cold start the model, enabling it to generate reasoning paths with explicit thinking modes. In the second stage, we apply ACPO to further enhance adaptive system switch for difficulty-aware reasoning. Experimental results demonstrate that ACPO effectively reduces redundant reasoning while adaptively adjusting cognitive allocation based on task complexity, achieving efficient hybrid reasoning. 

**Abstract (ZH)**: 基于认知策略优化的大型推理模型高效推理方法（Adaptive Cognition Policy Optimization for Efficient Reasoning in Large Reasoning Models） 

---
# EquivPruner: Boosting Efficiency and Quality in LLM-Based Search via Action Pruning 

**Title (ZH)**: EquivPruner: 通过动作剪枝提升基于LLM的搜索的效率和质量 

**Authors**: Jiawei Liu, Qisi Chen, Jianshu Zhang, Quan Liu, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.16312)  

**Abstract**: Large Language Models (LLMs) excel at complex reasoning through search algorithms, yet current strategies often suffer from massive token consumption due to redundant exploration of semantically equivalent steps. Existing semantic similarity methods struggle to accurately identify such equivalence in domain-specific contexts like mathematical reasoning. To address this, we propose EquivPruner, a simple yet effective approach that identifies and prunes semantically equivalent actions during LLM reasoning search. We also introduce MathEquiv, the first dataset we created for mathematical statement equivalence, which enables the training of a lightweight equivalence detector. Extensive experiments across various models and tasks demonstrate that EquivPruner significantly reduces token consumption, improving searching efficiency and often bolstering reasoning accuracy. For instance, when applied to Qwen2.5-Math-7B-Instruct on GSM8K, EquivPruner reduced token consumption by 48.1\% while also improving accuracy. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型通过搜索算法在复杂推理方面表现出色，但由于在语义等价步骤上的冗余探索，当前策略往往会消耗大量令牌。现有的语义相似性方法在处理数学推理等特定领域语境中的等价性识别时表现不佳。为解决这一问题，我们提出了一种名为EquivPruner的简单而有效的方法，在LLM推理搜索过程中识别并修剪语义等价的操作。同时，我们引入了MathEquiv数据集，这是首个用于数学命题等价性的数据集，使轻量级等价性检测器的训练成为可能。在各种模型和任务上的广泛实验表明，EquivPruner能显著降低令牌消耗，提高搜索效率，并且常常增强推理准确性。例如，在应用于Qwen2.5-Math-7B-Instruct和GSM8K任务时，EquivPruner将令牌消耗减少了48.1%，同时提高了一定的准确率。代码已开源。 

---
# How do Scaling Laws Apply to Knowledge Graph Engineering Tasks? The Impact of Model Size on Large Language Model Performance 

**Title (ZH)**: 扩展律在知识图谱工程任务中的应用：模型规模对大型语言模型性能的影响 

**Authors**: Desiree Heim, Lars-Peter Meyer, Markus Schröder, Johannes Frey, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2505.16276)  

**Abstract**: When using Large Language Models (LLMs) to support Knowledge Graph Engineering (KGE), one of the first indications when searching for an appropriate model is its size. According to the scaling laws, larger models typically show higher capabilities. However, in practice, resource costs are also an important factor and thus it makes sense to consider the ratio between model performance and costs. The LLM-KG-Bench framework enables the comparison of LLMs in the context of KGE tasks and assesses their capabilities of understanding and producing KGs and KG queries. Based on a dataset created in an LLM-KG-Bench run covering 26 open state-of-the-art LLMs, we explore the model size scaling laws specific to KGE tasks. In our analyses, we assess how benchmark scores evolve between different model size categories. Additionally, we inspect how the general score development of single models and families of models correlates to their size. Our analyses revealed that, with a few exceptions, the model size scaling laws generally also apply to the selected KGE tasks. However, in some cases, plateau or ceiling effects occurred, i.e., the task performance did not change much between a model and the next larger model. In these cases, smaller models could be considered to achieve high cost-effectiveness. Regarding models of the same family, sometimes larger models performed worse than smaller models of the same family. These effects occurred only locally. Hence it is advisable to additionally test the next smallest and largest model of the same family. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）支持知识图谱工程（KGE）时的模型大小 Scaling Laws 特异性研究：基于 LLM-KG-Bench 架构的分析 

---
# MAPLE: Many-Shot Adaptive Pseudo-Labeling for In-Context Learning 

**Title (ZH)**: MAPLE: 多-shot自适应伪标签化在上下文学习中的应用 

**Authors**: Zihan Chen, Song Wang, Zhen Tan, Jundong Li, Cong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16225)  

**Abstract**: In-Context Learning (ICL) empowers Large Language Models (LLMs) to tackle diverse tasks by incorporating multiple input-output examples, known as demonstrations, into the input of LLMs. More recently, advancements in the expanded context windows of LLMs have led to many-shot ICL, which uses hundreds of demonstrations and outperforms few-shot ICL, which relies on fewer examples. However, this approach is often hindered by the high cost of obtaining large amounts of labeled data. To address this challenge, we propose Many-Shot Adaptive Pseudo-LabEling, namely MAPLE, a novel influence-based many-shot ICL framework that utilizes pseudo-labeled samples to compensate for the lack of label information. We first identify a subset of impactful unlabeled samples and perform pseudo-labeling on them by querying LLMs. These pseudo-labeled samples are then adaptively selected and tailored to each test query as input to improve the performance of many-shot ICL, without significant labeling costs. Extensive experiments on real-world datasets demonstrate the effectiveness of our framework, showcasing its ability to enhance LLM adaptability and performance with limited labeled data. 

**Abstract (ZH)**: 基于影响分析的Many-Shot自适应伪标注学习 

---
# LightRouter: Towards Efficient LLM Collaboration with Minimal Overhead 

**Title (ZH)**: LightRouter: 向量效LLM协作 minimal overhead 

**Authors**: Yifan Zhang, Xinkui Zhao, Zuxin Wang, Guanjie Cheng, Yueshen Xu, Shuiguang Deng, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16221)  

**Abstract**: The rapid advancement of large language models has unlocked remarkable capabilities across a diverse array of natural language processing tasks. However, the considerable differences among available LLMs-in terms of cost, performance, and computational demands-pose significant challenges for users aiming to identify the most suitable model for specific tasks. In this work, we present LightRouter, a novel framework designed to systematically select and integrate a small subset of LLMs from a larger pool, with the objective of jointly optimizing both task performance and cost efficiency. LightRouter leverages an adaptive selection mechanism to identify models that require only a minimal number of boot tokens, thereby reducing costs, and further employs an effective integration strategy to combine their outputs. Extensive experiments across multiple benchmarks demonstrate that LightRouter matches or outperforms widely-used ensemble baselines, achieving up to a 25% improvement in accuracy. Compared with leading high-performing models, LightRouter achieves comparable performance while reducing inference costs by up to 27%. Importantly, our framework operates without any prior knowledge of individual models and relies exclusively on inexpensive, lightweight models. This work introduces a practical approach for efficient LLM selection and provides valuable insights into optimal strategies for model combination. 

**Abstract (ZH)**: 大型语言模型的迅速发展解锁了多样自然语言处理任务的非凡能力。然而，可用的大规模语言模型在成本、性能和计算需求方面的显著差异为用户选择最适合特定任务的模型带来了重大挑战。本文介绍了LightRouter，一种新型框架，旨在系统地从大量候选模型中选择和整合一个小型子集，以同时优化任务性能和成本效率。LightRouter利用自适应选择机制来识别仅需少量启动标记的模型，从而降低成本，并进一步采用有效整合策略结合它们的输出。多基准实验表明，LightRouter与广泛使用的集成基线相当或优于基线，最高可提升25%的准确性。与顶级高性能模型相比，LightRouter在降低成本高达27%的同时实现了相当的性能。重要的是，我们的框架无需任何关于个体模型的先验知识，仅依赖于经济且轻量级的模型。本文引入了一种高效的大型语言模型选择实践方法，并提供了关于模型组合的优化策略的宝贵见解。 

---
# SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning 

**Title (ZH)**: SafeKey: 增强安全推理中的恍然大悟洞察 

**Authors**: Kaiwen Zhou, Xuandong Zhao, Gaowen Liu, Jayanth Srinivasa, Aosong Feng, Dawn Song, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16186)  

**Abstract**: Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations. 

**Abstract (ZH)**: 大型推理模型中的安全关键时刻：SafeKey 

---
# LLM-Powered AI Agent Systems and Their Applications in Industry 

**Title (ZH)**: LLM驱动的AI代理系统及其在工业领域的应用 

**Authors**: Guannan Liang, Qianqian Tong  

**Link**: [PDF](https://arxiv.org/pdf/2505.16120)  

**Abstract**: The emergence of Large Language Models (LLMs) has reshaped agent systems. Unlike traditional rule-based agents with limited task scope, LLM-powered agents offer greater flexibility, cross-domain reasoning, and natural language interaction. Moreover, with the integration of multi-modal LLMs, current agent systems are highly capable of processing diverse data modalities, including text, images, audio, and structured tabular data, enabling richer and more adaptive real-world behavior. This paper comprehensively examines the evolution of agent systems from the pre-LLM era to current LLM-powered architectures. We categorize agent systems into software-based, physical, and adaptive hybrid systems, highlighting applications across customer service, software development, manufacturing automation, personalized education, financial trading, and healthcare. We further discuss the primary challenges posed by LLM-powered agents, including high inference latency, output uncertainty, lack of evaluation metrics, and security vulnerabilities, and propose potential solutions to mitigate these concerns. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现重塑了代理系统。不同于传统基于规则的代理系统，LLM驱动的代理系统提供了更大的灵活性、跨域推理和自然语言交互。此外，通过集成多模态LLM，当前的代理系统能够高效处理包括文本、图像、音频和结构化表格数据在内的多种数据模态，从而实现更加丰富和适应性强的现实世界行为。本文全面考察了从预LLM时代到当前LLM驱动架构代理系统的演化。我们将代理系统分为软件基座式、物理式和适应性混合系统，并强调了其在客户服务、软件开发、制造业自动化、个性化教育、金融交易和医疗保健等领域的应用。我们进一步讨论了LLM驱动代理系统所面临的 primary challenges，包括高推理延迟、输出不确定性、缺乏评价指标以及安全漏洞，并提出了可能的解决方案以减轻这些担忧。 

---
# Logic-of-Thought: Empowering Large Language Models with Logic Programs for Solving Puzzles in Natural Language 

**Title (ZH)**: 逻辑思维：通过逻辑程序增强大型语言模型以解决自然语言中的谜题 

**Authors**: Naiqi Li, Peiyuan Liu, Zheng Liu, Tao Dai, Yong Jiang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16114)  

**Abstract**: Solving puzzles in natural language poses a long-standing challenge in AI. While large language models (LLMs) have recently shown impressive capabilities in a variety of tasks, they continue to struggle with complex puzzles that demand precise reasoning and exhaustive search. In this paper, we propose Logic-of-Thought (Logot), a novel framework that bridges LLMs with logic programming to address this problem. Our method leverages LLMs to translate puzzle rules and states into answer set programs (ASPs), the solution of which are then accurately and efficiently inferred by an ASP interpreter. This hybrid approach combines the natural language understanding of LLMs with the precise reasoning capabilities of logic programs. We evaluate our method on various grid puzzles and dynamic puzzles involving actions, demonstrating near-perfect accuracy across all tasks. Our code and data are available at: this https URL. 

**Abstract (ZH)**: 自然语言求解谜题是AI领域的长期挑战。尽管大规模语言模型（LLMs）最近在各种任务中展现了令人印象深刻的性能，它们在需要精确推理和全面搜索的复杂谜题中仍然表现不佳。在本文中，我们提出了一种新的框架——Logic-of-Thought (Logot)，将LLMs与逻辑编程相结合，以解决这一问题。我们的方法利用LLMs将谜题规则和状态转换为回答集程序（ASP），ASP解析器随后准确高效地推断出解。这种混合方法结合了LLMs的自然语言理解和逻辑程序的精确推理能力。我们在各种网格谜题和涉及动作的动态谜题上评估了该方法，展示了在所有任务中几乎完美的准确性。我们的代码和数据可在以下链接获取：this https URL。 

---
# Can AI Read Between The Lines? Benchmarking LLMs On Financial Nuance 

**Title (ZH)**: AI能读透文字背后的意思吗？LLMs在金融细微差异方面的基准测试 

**Authors**: Dominick Kubica, Dylan T. Gordon, Nanami Emura, Derleen Saini, Charlie Goldenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.16090)  

**Abstract**: As of 2025, Generative Artificial Intelligence (GenAI) has become a central tool for productivity across industries. Beyond text generation, GenAI now plays a critical role in coding, data analysis, and research workflows. As large language models (LLMs) continue to evolve, it is essential to assess the reliability and accuracy of their outputs, especially in specialized, high-stakes domains like finance. Most modern LLMs transform text into numerical vectors, which are used in operations such as cosine similarity searches to generate responses. However, this abstraction process can lead to misinterpretation of emotional tone, particularly in nuanced financial contexts. While LLMs generally excel at identifying sentiment in everyday language, these models often struggle with the nuanced, strategically ambiguous language found in earnings call transcripts. Financial disclosures frequently embed sentiment in hedged statements, forward-looking language, and industry-specific jargon, making it difficult even for human analysts to interpret consistently, let alone AI models. This paper presents findings from the Santa Clara Microsoft Practicum Project, led by Professor Charlie Goldenberg, which benchmarks the performance of Microsoft's Copilot, OpenAI's ChatGPT, Google's Gemini, and traditional machine learning models for sentiment analysis of financial text. Using Microsoft earnings call transcripts, the analysis assesses how well LLM-derived sentiment correlates with market sentiment and stock movements and evaluates the accuracy of model outputs. Prompt engineering techniques are also examined to improve sentiment analysis results. Visualizations of sentiment consistency are developed to evaluate alignment between tone and stock performance, with sentiment trends analyzed across Microsoft's lines of business to determine which segments exert the greatest influence. 

**Abstract (ZH)**: 截至2025年，生成式人工智能（GenAI）已成为各行各业提高生产力的核心工具。除了文本生成，GenAI在编程、数据分析和研究工作流程中也扮演着至关重要的角色。随着大型语言模型（LLMs）的不断演变，评估其输出的可靠性和准确性变得尤为重要，尤其是在金融等专业且高风险的领域。大多数现代LLM将文本转换为数值向量，用于如余弦相似度搜索等操作生成响应。然而，这一抽象过程可能导致情绪色调的误读，尤其是在细腻的金融语境中。虽然LLMs在识别普通语言中的情感方面表现良好，但在分析收益电话会议记录中发现的细微、战略性含糊语言时，这些模型往往会遇到困难。财务披露经常嵌入了谨慎陈述、前瞻性语言和行业特定术语的情感，这使得即使是人类分析师也难以一致地进行解释，更不用说AI模型了。本文基于Santa Clara微软实践项目的研究成果，该项目由Charlie Goldenberg教授领导，旨在评估微软Copilot、OpenAI的ChatGPT、Google的Gemini以及传统机器学习模型在金融文本情感分析方面的表现。通过分析微软收益电话会议记录，该研究评估LLM衍生情感与市场情绪和股票变动的相关性，并评估模型输出的准确性。还研究了促进情感分析结果的方法工程技术。开发了情感一致性可视化来评估语气与股票表现之间的对齐情况，并分析了微软各业务线的情感趋势，以确定哪些细分市场的影响最大。 

---
# Optimizing LLM-Based Multi-Agent System with Textual Feedback: A Case Study on Software Development 

**Title (ZH)**: 基于文本反馈优化LLM驱动的多智能体系统：软件开发案例研究 

**Authors**: Ming Shen, Raphael Shu, Anurag Pratik, James Gung, Yubin Ge, Monica Sunkara, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16086)  

**Abstract**: We have seen remarkable progress in large language models (LLMs) empowered multi-agent systems solving complex tasks necessitating cooperation among experts with diverse skills. However, optimizing LLM-based multi-agent systems remains challenging. In this work, we perform an empirical case study on group optimization of role-based multi-agent systems utilizing natural language feedback for challenging software development tasks under various evaluation dimensions. We propose a two-step agent prompts optimization pipeline: identifying underperforming agents with their failure explanations utilizing textual feedback and then optimizing system prompts of identified agents utilizing failure explanations. We then study the impact of various optimization settings on system performance with two comparison groups: online against offline optimization and individual against group optimization. For group optimization, we study two prompting strategies: one-pass and multi-pass prompting optimizations. Overall, we demonstrate the effectiveness of our optimization method for role-based multi-agent systems tackling software development tasks evaluated on diverse evaluation dimensions, and we investigate the impact of diverse optimization settings on group behaviors of the multi-agent systems to provide practical insights for future development. 

**Abstract (ZH)**: 我们已经在大型语言模型（LLMs）赋能的多智能体系统解决需要专家之间协作的复杂任务方面看到了显著进步。然而，优化基于LLM的多智能体系统仍然是一个挑战。本研究通过对角色_based多智能体系统进行自然语言反馈驱动的小组优化的实证案例研究，探讨了在多种评价维度下挑战性软件开发任务中的系统性能影响。我们提出了一种两阶段智能体提示优化管道：利用文本反馈识别表现不佳的智能体及其失败解释，然后利用失败解释优化被识别智能体的系统提示。我们研究了不同优化设置对系统性能的影响，设置了两个比较组：在线优化与离线优化，以及个体优化与小组优化。在小组优化方面，我们研究了两种提示策略：单轮和多轮提示优化。总体而言，我们展示了该优化方法在面对多种评价维度下的软件开发任务挑战性任务的角色_based多智能体系统中的有效性，并探讨了不同优化设置对多智能体系统小组行为的影响，以提供对未来开发的实用见解。 

---
# How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior 

**Title (ZH)**: LSTM记忆管理对语言模型代理的影响：一种经验跟随行为的实证研究 

**Authors**: Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Jiliang Tang, Himabindu Lakkaraju, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16067)  

**Abstract**: Memory is a critical component in large language model (LLM)-based agents, enabling them to store and retrieve past executions to improve task performance over time. In this paper, we conduct an empirical study on how memory management choices impact the LLM agents' behavior, especially their long-term performance. Specifically, we focus on two fundamental memory operations that are widely used by many agent frameworks-addition, which incorporates new experiences into the memory base, and deletion, which selectively removes past experiences-to systematically study their impact on the agent behavior. Through our quantitative analysis, we find that LLM agents display an experience-following property: high similarity between a task input and the input in a retrieved memory record often results in highly similar agent outputs. Our analysis further reveals two significant challenges associated with this property: error propagation, where inaccuracies in past experiences compound and degrade future performance, and misaligned experience replay, where outdated or irrelevant experiences negatively influence current tasks. Through controlled experiments, we show that combining selective addition and deletion strategies can help mitigate these negative effects, yielding an average absolute performance gain of 10% compared to naive memory growth. Furthermore, we highlight how memory management choices affect agents' behavior under challenging conditions such as task distribution shifts and constrained memory resources. Our findings offer insights into the behavioral dynamics of LLM agent memory systems and provide practical guidance for designing memory components that support robust, long-term agent performance. We also release our code to facilitate further study. 

**Abstract (ZH)**: 基于大型语言模型的代理的记忆管理研究：行为影响与挑战 

---
# SPhyR: Spatial-Physical Reasoning Benchmark on Material Distribution 

**Title (ZH)**: SPhyR：材料分布的空间物理推理基准 

**Authors**: Philipp D. Siedler  

**Link**: [PDF](https://arxiv.org/pdf/2505.16048)  

**Abstract**: We introduce a novel dataset designed to benchmark the physical and spatial reasoning capabilities of Large Language Models (LLM) based on topology optimization, a method for computing optimal material distributions within a design space under prescribed loads and supports. In this dataset, LLMs are provided with conditions such as 2D boundary, applied forces and supports, and must reason about the resulting optimal material distribution. The dataset includes a variety of tasks, ranging from filling in masked regions within partial structures to predicting complete material distributions. Solving these tasks requires understanding the flow of forces and the required material distribution under given constraints, without access to simulation tools or explicit physical models, challenging models to reason about structural stability and spatial organization. Our dataset targets the evaluation of spatial and physical reasoning abilities in 2D settings, offering a complementary perspective to traditional language and logic benchmarks. 

**Abstract (ZH)**: 我们引入了一个新型的数据集，用于评估基于拓扑优化的大规模语言模型（LLM）的物理和空间推理能力。该数据集提供给LLM二维边界、施加的力和支撑条件，并要求其推断在这些条件下最优的材料分布。数据集包括从填充部分结构中的空白区域到预测完整材料分布的各种任务。解决这些任务需要理解在给定约束条件下力的传递和所需的材料分布，挑战模型进行结构稳定性和空间组织的推理。我们的数据集旨在评估2D设置中的空间和物理推理能力，为传统语言和逻辑基准测试提供补充视角。 

---
# Causal LLM Routing: End-to-End Regret Minimization from Observational Data 

**Title (ZH)**: 因果LLM路由：基于观测数据的端到端后悔最小化 

**Authors**: Asterios Tsiourvas, Wei Sun, Georgia Perakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.16037)  

**Abstract**: LLM routing aims to select the most appropriate model for each query, balancing competing performance metrics such as accuracy and cost across a pool of language models. Prior approaches typically adopt a decoupled strategy, where the metrics are first predicted and the model is then selected based on these estimates. This setup is prone to compounding errors and often relies on full-feedback data, where each query is evaluated by all candidate models, which is costly to obtain and maintain in practice. In contrast, we learn from observational data, which records only the outcome of the model actually deployed. We propose a causal end-to-end framework that learns routing policies by minimizing decision-making regret from observational data. To enable efficient optimization, we introduce two theoretically grounded surrogate objectives: a classification-based upper bound, and a softmax-weighted regret approximation shown to recover the optimal policy at convergence. We further extend our framework to handle heterogeneous cost preferences via an interval-conditioned architecture. Experiments on public benchmarks show that our method outperforms existing baselines, achieving state-of-the-art performance across different embedding models. 

**Abstract (ZH)**: LLM路由旨在为每个查询选择最合适的模型，平衡准确度和成本等竞争性能指标。先前的方法通常采用解耦策略，先预测指标，再基于这些估计来选择模型。这种设置容易累积错误，并且通常依赖于全反馈数据，即每个查询都被所有候选模型评估，这在实践中成本高昂且难以维持。相比之下，我们从观测数据中学习，只记录实际部署的模型的结果。我们提出了一种因果端到端框架，通过从观测数据中最小化决策遗憾来学习路由策略。为了实现高效的优化，我们引入了两个理论依据的替代目标：基于分类的上界和softmax加权遗憾近似，在收敛时可恢复最优策略。我们进一步扩展了框架以处理不同的成本偏好，通过区间条件化架构来实现。实验表明，我们的方法在公共基准测试中优于现有基线，实现了不同嵌入模型下的最佳性能。 

---
# SpatialScore: Towards Unified Evaluation for Multimodal Spatial Understanding 

**Title (ZH)**: SpatialScore: 朝着统一评估多模态空间理解的方向 

**Authors**: Haoning Wu, Xiao Huang, Yaohui Chen, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.17012)  

**Abstract**: Multimodal large language models (MLLMs) have achieved impressive success in question-answering tasks, yet their capabilities for spatial understanding are less explored. This work investigates a critical question: do existing MLLMs possess 3D spatial perception and understanding abilities? Concretely, we make the following contributions in this paper: (i) we introduce VGBench, a benchmark specifically designed to assess MLLMs for visual geometry perception, e.g., camera pose and motion estimation; (ii) we propose SpatialScore, the most comprehensive and diverse multimodal spatial understanding benchmark to date, integrating VGBench with relevant data from the other 11 existing datasets. This benchmark comprises 28K samples across various spatial understanding tasks, modalities, and QA formats, along with a carefully curated challenging subset, SpatialScore-Hard; (iii) we develop SpatialAgent, a novel multi-agent system incorporating 9 specialized tools for spatial understanding, supporting both Plan-Execute and ReAct reasoning paradigms; (iv) we conduct extensive evaluations to reveal persistent challenges in spatial reasoning while demonstrating the effectiveness of SpatialAgent. We believe SpatialScore will offer valuable insights and serve as a rigorous benchmark for the next evolution of MLLMs. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在问答任务中取得了显著成功，但其空间理解能力尚未得到充分探索。本文探讨了一个关键问题：现有的MLLMs是否具备三维空间感知和理解能力？具体而言，本文做出如下贡献：（i）我们引入了VGBench，这是一个专门用于评估MLLMs的视觉几何感知基准，例如相机姿态和运动估计；（ii）我们提出了SpatialScore，这是迄今为止最全面和多样的多模态空间理解基准，将VGBench与来自其他11个现有数据集的相关数据集成；该基准包括了各种空间理解任务、模态和问答格式的28,000个样本，以及一个精心策划的具有挑战性的子集SpatialScore-Hard；（iii）我们开发了SpatialAgent，这是一个包含9种专门工具的新型多智能体系统，支持计划-执行和ReAct推理范式；（iv）我们进行了广泛的评估，揭示了空间推理中的持续挑战，同时展示了SpatialAgent的有效性。我们相信SpatialScore将提供有价值的见解，并作为MLLMs下一阶段演进的严格基准。 

---
# R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning 

**Title (ZH)**: R1-Searcher++: 通过强化学习激励LLM的动态知识获取 

**Authors**: Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17005)  

**Abstract**: Large Language Models (LLMs) are powerful but prone to hallucinations due to static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting external information, but current methods often are costly, generalize poorly, or ignore the internal knowledge of the model. In this paper, we introduce R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage both internal and external knowledge sources. R1-Searcher++ employs a two-stage training strategy: an initial SFT Cold-start phase for preliminary format learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses outcome-supervision to encourage exploration, incorporates a reward mechanism for internal knowledge utilization, and integrates a memorization mechanism to continuously assimilate retrieved information, thereby enriching the model's internal knowledge. By leveraging internal knowledge and external search engine, the model continuously improves its capabilities, enabling efficient retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++ outperforms previous RAG and reasoning methods and achieves efficient retrieval. The code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）虽然强大但容易因静态知识而产生幻觉。检索增强生成（RAG）通过注入外部信息有所帮助，但当前方法往往成本高、泛化能力差或忽视模型的内部知识。本文介绍了一种新的R1-Searcher++框架，旨在训练LLMs适应性地利用内部和外部知识来源。R1-Searcher++采用两阶段训练策略：初始的SFT冷启动阶段进行初步格式学习，随后是基于奖励的动态知识获取阶段。奖励阶段使用结果监督鼓励探索，引入了内部知识利用的奖励机制，并结合记忆机制持续吸收检索到的信息，从而丰富模型的内部知识。通过利用内部知识和外部搜索引擎，模型能够不断改进其能力，实现高效的检索增强推理。我们的实验表明，R1-Searcher++在与以往的RAG和推理方法的对比中表现更优，实现了高效的检索。代码可在以下链接获取：this https URL。 

---
# Do Large Language Models Excel in Complex Logical Reasoning with Formal Language? 

**Title (ZH)**: 大型语言模型在形式语言中的复杂逻辑推理方面表现出色吗？ 

**Authors**: Jin Jiang, Jianing Wang, Yuchen Yan, Yang Liu, Jianhua Zhu, Mengdi Zhang, Xunliang Cai, Liangcai Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16998)  

**Abstract**: Large Language Models (LLMs) have been shown to achieve breakthrough performance on complex logical reasoning tasks. Nevertheless, most existing research focuses on employing formal language to guide LLMs to derive reliable reasoning paths, while systematic evaluations of these capabilities are still limited. In this paper, we aim to conduct a comprehensive evaluation of LLMs across various logical reasoning problems utilizing formal languages. From the perspective of three dimensions, i.e., spectrum of LLMs, taxonomy of tasks, and format of trajectories, our key findings are: 1) Thinking models significantly outperform Instruct models, especially when formal language is employed; 2) All LLMs exhibit limitations in inductive reasoning capability, irrespective of whether they use a formal language; 3) Data with PoT format achieves the best generalization performance across other languages. Additionally, we also curate the formal-relative training data to further enhance the small language models, and the experimental results indicate that a simple rejected fine-tuning method can better enable LLMs to generalize across formal languages and achieve the best overall performance. Our codes and reports are available at this https URL. 

**Abstract (ZH)**: 大型语言模型在复杂逻辑推理任务中取得了突破性的性能，但大多数现有研究主要集中于使用正式语言指导模型推导可靠的推理路径，系统的评估仍然有限。本文旨在利用正式语言对大型语言模型在各种逻辑推理问题上的能力进行全面评估。从三个维度，即大型语言模型的谱系、任务的分类以及轨迹的格式，我们的关键发现是：1) 思维模型明显优于指令模型，尤其是在使用正式语言的情况下；2) 所有大型语言模型在归纳推理能力方面都存在局限性，无论是否使用正式语言；3)采用PoT格式的数据在其他语言中表现出最佳的泛化性能。此外，我们还整理了相关正式语言训练数据以进一步增强小型语言模型，并实验结果表明一个简单的拒绝微调方法能够更好地使模型跨正式语言泛化并取得最佳的总体性能。我们的代码和报告可在以下链接访问：this https URL。 

---
# $\text{R}^2\text{ec}$: Towards Large Recommender Models with Reasoning 

**Title (ZH)**: R$^2\text{ec}$: 向大规模具有推理能力的推荐模型迈进 

**Authors**: Runyang You, Yongqi Li, Xinyu Lin, Xin Zhang, Wenjie Wang, Wenjie Li, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16994)  

**Abstract**: Large recommender models have extended LLMs as powerful recommenders via encoding or item generation, and recent breakthroughs in LLM reasoning synchronously motivate the exploration of reasoning in recommendation. Current studies usually position LLMs as external reasoning modules to yield auxiliary thought for augmenting conventional recommendation pipelines. However, such decoupled designs are limited in significant resource cost and suboptimal joint optimization. To address these issues, we propose \name, a unified large recommender model with intrinsic reasoning capabilities. Initially, we reconceptualize the model architecture to facilitate interleaved reasoning and recommendation in the autoregressive process. Subsequently, we propose RecPO, a corresponding reinforcement learning framework that optimizes \name\ both the reasoning and recommendation capabilities simultaneously in a single policy update; RecPO introduces a fused reward scheme that solely leverages recommendation labels to simulate the reasoning capability, eliminating dependency on specialized reasoning annotations. Experiments on three datasets with various baselines verify the effectiveness of \name, showing relative improvements of 68.67\% in Hit@5 and 45.21\% in NDCG@20. Code available at this https URL. 

**Abstract (ZH)**: 一种具有内在推理能力的统一大型推荐模型 

---
# MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems 

**Title (ZH)**: MASLab: 一种统一且全面的基于LLM的多Agent系统代码库 

**Authors**: Rui Ye, Keduan Huang, Qimin Wu, Yuzhu Cai, Tian Jin, Xianghe Pang, Xiangrui Liu, Jiaqi Su, Chen Qian, Bohan Tang, Kaiqu Liang, Jiaao Chen, Yue Hu, Zhenfei Yin, Rongye Shi, Bo An, Yang Gao, Wenjun Wu, Lei Bai, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16988)  

**Abstract**: LLM-based multi-agent systems (MAS) have demonstrated significant potential in enhancing single LLMs to address complex and diverse tasks in practical applications. Despite considerable advancements, the field lacks a unified codebase that consolidates existing methods, resulting in redundant re-implementation efforts, unfair comparisons, and high entry barriers for researchers. To address these challenges, we introduce MASLab, a unified, comprehensive, and research-friendly codebase for LLM-based MAS. (1) MASLab integrates over 20 established methods across multiple domains, each rigorously validated by comparing step-by-step outputs with its official implementation. (2) MASLab provides a unified environment with various benchmarks for fair comparisons among methods, ensuring consistent inputs and standardized evaluation protocols. (3) MASLab implements methods within a shared streamlined structure, lowering the barriers for understanding and extension. Building on MASLab, we conduct extensive experiments covering 10+ benchmarks and 8 models, offering researchers a clear and comprehensive view of the current landscape of MAS methods. MASLab will continue to evolve, tracking the latest developments in the field, and invite contributions from the broader open-source community. 

**Abstract (ZH)**: 基于大型语言模型的多代理系统（LLM-based Multi-Agent Systems, LLM- MAS）展示了在实际应用中处理复杂多样的任务方面的显著潜力。尽管取得了显著进步，该领域缺乏一个整合现有方法的统一代码库，导致重复实现、不公平比较和高研究门槛。为解决这些问题，我们介绍了MASLab，一个统一、全面且有利于研究的LLM- MAS代码库。（1）MASLab整合了来自多个领域的超过20种成熟方法，并通过逐步输出与官方实现的比较进行了严格验证。（2）MASLab提供了一个统一的环境和多种基准，用于公正比较方法，确保一致的输入和标准化的评估协议。（3）MASLab在共享的简化结构中实现方法，降低了理解和扩展的门槛。基于MASLab，我们进行了广泛的实验，覆盖了10多个基准和8个模型，为研究人员提供了当前LM-MAS方法布局的清晰而全面的视角。MASLab将持续发展，跟踪该领域的最新进展，并邀请更广泛开源社区的贡献。 

---
# T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning 

**Title (ZH)**: 面向工具的多轮代理规划对话数据集 

**Authors**: Amartya Chakraborty, Paresh Dashore, Nadia Bathaee, Anmol Jain, Anirban Das, Shi-Xiong Zhang, Sambit Sahu, Milind Naphade, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2505.16986)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities as intelligent agents capable of solving complex problems. However, effective planning in scenarios involving dependencies between API or tool calls-particularly in multi-turn conversations-remains a significant challenge. To address this, we introduce T1, a tool-augmented, multi-domain, multi-turn conversational dataset specifically designed to capture and manage inter-tool dependencies across diverse domains. T1 enables rigorous evaluation of agents' ability to coordinate tool use across nine distinct domains (4 single domain and 5 multi-domain) with the help of an integrated caching mechanism for both short- and long-term memory, while supporting dynamic replanning-such as deciding whether to recompute or reuse cached results. Beyond facilitating research on tool use and planning, T1 also serves as a benchmark for evaluating the performance of open-source language models. We present results powered by T1-Agent, highlighting their ability to plan and reason in complex, tool-dependent scenarios. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了作为具有解决复杂问题能力的智能代理的 impressive 能力。然而，在涉及 API 或工具调用之间的依赖关系的情景中，尤其是在多轮对话中，有效的规划仍然是一项重大挑战。为了解决这一问题，我们引入了 T1，一个工具增强的、跨域的、多轮对话数据集，旨在捕捉和管理不同领域之间的工具间依赖关系。T1 通过集成的缓存机制支持短期和长期记忆，并允许动态重新规划（如决定重计算或重用缓存结果）来评估代理协调工具使用的能力，支持在四个单域和五个跨域任务中的九个不同的领域中进行严格的评估。除了促进关于工具使用和规划的研究外，T1 还作为开源语言模型性能评估的标准基准。我们展示了由 T1-Agent 得出的结果，突显了其在复杂、工具依赖场景中进行规划和推理的能力。 

---
# Fixing Data That Hurts Performance: Cascading LLMs to Relabel Hard Negatives for Robust Information Retrieval 

**Title (ZH)**: 修复损害性能的数据：级联大语言模型重新标记困难负样本以实现稳健的信息检索 

**Authors**: Nandan Thakur, Crystina Zhang, Xueguang Ma, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16967)  

**Abstract**: Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection reduces the training set size by 2.35$\times$ and increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We propose a simple, cost-effective approach using cascading LLM prompts to identify and relabel hard negatives. Experimental results show that relabeling false negatives with true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7-1.4 nDCG@10 on BEIR and by 1.7-1.8 nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of the cascading design is further supported by human annotation results, where we find judgment by GPT-4o shows much higher agreement with humans than GPT-4o-mini. 

**Abstract (ZH)**: 训练稳健的检索和重排模型通常依赖大规模的检索数据集；例如，BGE集合包含来自各种数据源的160万查询-段落对。然而，我们发现某些数据集会对模型效果产生负面影响——从BGE集合中剔除15个数据集中的8个，可将训练集规模减少2.35倍，并在BEIR上提高nDCG@10分数1.0分。这促使我们对训练数据质量进行更深入的考察，特别是关注那些本应相关但被错误标记为无关的“假负样本”。我们提出了一种简单且成本效益高的方法，通过级联LLM提示来识别和重新标记困难的负样本。实验结果表明，使用真正相关样本重新标记假负样本可以提高E5（基线）和Qwen2.5-7B检索模型在BEIR上的nDCG@10分数0.7-1.4分，并且在零样本AIR-Bench评估中提高1.7-1.8分。对使用重新标记数据进行微调的重排模型（如BEIR上的Qwen2.5-3B）也观察到了类似收益。进一步的人工标注结果显示，GPT-4o的判断与人类的高度一致，而GPT-4o-mini则不然。 

---
# Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models 

**Title (ZH)**: 无形的提示，可见的威胁：大型语言模型外部资源中的恶意字体注入 

**Authors**: Junjie Xiong, Changjia Zhu, Shuhang Lin, Chong Zhang, Yongfeng Zhang, Yao Liu, Lingyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16957)  

**Abstract**: Large Language Models (LLMs) are increasingly equipped with capabilities of real-time web search and integrated with protocols like Model Context Protocol (MCP). This extension could introduce new security vulnerabilities. We present a systematic investigation of LLM vulnerabilities to hidden adversarial prompts through malicious font injection in external resources like webpages, where attackers manipulate code-to-glyph mapping to inject deceptive content which are invisible to users. We evaluate two critical attack scenarios: (1) "malicious content relay" and (2) "sensitive data leakage" through MCP-enabled tools. Our experiments reveal that indirect prompts with injected malicious font can bypass LLM safety mechanisms through external resources, achieving varying success rates based on data sensitivity and prompt design. Our research underscores the urgent need for enhanced security measures in LLM deployments when processing external content. 

**Abstract (ZH)**: 大型语言模型通过恶意字体注入外部资源中的隐藏对抗提示研究：针对模型上下文协议（MCP）启用工具的新安全漏洞分析 

---
# Bottlenecked Transformers: Periodic KV Cache Abstraction for Generalised Reasoning 

**Title (ZH)**: 瓶颈变换器：周期性 KV 缓存抽象化通用推理 

**Authors**: Adnan Oomerjee, Zafeirios Fountas, Zhongwei Yu, Haitham Bou-Ammar, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16950)  

**Abstract**: Despite their impressive capabilities, Large Language Models struggle with generalisation beyond their training distribution, often exhibiting sophisticated pattern interpolation rather than true abstract reasoning (extrapolation). In this work, we approach this limitation through the lens of Information Bottleneck (IB) theory, which posits that model generalisation emerges from an optimal balance between input compression and retention of predictive information in latent representations. We prove using IB theory that decoder-only Transformers are inherently constrained in their ability to form task-optimal sequence representations. We then use this result to demonstrate that periodic global transformation of the internal sequence-level representations (KV cache) is a necessary computational step for improving Transformer generalisation in reasoning tasks. Based on these theoretical insights, we propose a modification to the Transformer architecture, in the form of an additional module that globally rewrites the KV cache at periodic intervals, shifting its capacity away from memorising input prefixes and toward encoding features most useful for predicting future tokens. Our model delivers substantial gains on mathematical reasoning benchmarks, outperforming both vanilla Transformers with up to 3.5x more parameters, as well as heuristic-driven pruning mechanisms for cache compression. Our approach can be seen as a principled generalisation of existing KV-cache compression methods; whereas such methods focus solely on compressing input representations, they often do so at the expense of retaining predictive information, and thus their capabilities are inherently bounded by those of an unconstrained model. This establishes a principled framework to manipulate Transformer memory using information theory, addressing fundamental reasoning limitations that scaling alone cannot overcome. 

**Abstract (ZH)**: 尽管大型语言模型具有 impressive 的能力，但在超出训练分布范围的一般化方面仍然存在局限性，常常表现出复杂的模式插值而缺乏真正的抽象推理（外推）。本文通过信息瓶颈（IB）理论的视角来解决这一限制，该理论认为模型的一般化源自输入压缩和潜在表示中预测信息保留之间的最优平衡。我们利用 IB 理论证明，仅解码器的变压器在形成任务最优序列表示方面存在内在限制。然后，我们利用这一结果证明，周期性地全局变换内部序列级表示（KV 缓存）是提高变压器在推理任务中一般化能力的必要计算步骤。基于这些理论洞见，我们提出了一种对变压器架构的修改，形式上增加了一个额外模块，该模块在周期性间隔内全局重写 KV 缓存，将其实用容量从记忆输入前缀转移到对预测未来标记最有用的特征编码。我们的模型在数学推理基准测试中取得了显著的改进，超越了具有多达 3.5 倍参数数的 vanilla 变压器，以及驱动的缓存压缩启发式剪枝机制。我们的方法可以看作是现有 KV 缓存压缩方法的原理上的一般化；这些方法仅专注于压缩输入表示，而常常以保留预测信息为代价，因此它们的能力受到不受约束的模型的内在限制。这为我们提供了一种原理上的框架，使用信息理论操纵变压器记忆，解决了仅靠扩展无法克服的基本推理限制。 

---
# MixAT: Combining Continuous and Discrete Adversarial Training for LLMs 

**Title (ZH)**: MixAT：结合连续性和离散性对抗训练的大型语言模型训练方法 

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16947)  

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at this https URL. 

**Abstract (ZH)**: 尽管在大型语言模型（LLMs）的安全性和对齐方面付出了近期努力，当前针对前沿LLMs的对抗攻击仍能一致地迫使生成有害内容。虽然对抗训练已经被广泛研究并显示出显著提高传统机器学习模型鲁棒性的效果，但在LLMs背景下其优势与局限性尚不够了解。具体而言，在现有离散对抗攻击在生成有害内容方面非常有效的同时，使用具体的对抗提示训练LLMs通常计算成本高昂，导致对连续松弛的依赖。由于这些松弛不对应于离散输入令牌，这种潜在训练方法往往使模型容易受到一系列离散攻击。在这项工作中，我们通过引入结合更强离散攻击和更快连续攻击的MixAT新方法来填补这一差距。我们全面评估MixAT在最先进的各类攻击下，并提出最低成功率指标（ALO-ASR）来捕捉模型的最坏情况脆弱性。结果显示，MixAT在鲁棒性（ALO-ASR < 20%）方面显著优于先前防御方法（ALO-ASR > 50%），同时保持与基于连续松弛的方法相当的运行时间。此外，我们在现实部署场景下分析MixAT，探讨聊天模板、量化、低秩适配器和温度如何影响对抗训练和评估，揭示当前方法中的额外盲点。我们的结果表明，MixAT的离散-连续防御提供了一种具有最小计算开销的鲁棒性-准确率权衡，并展示了其构建更安全LLMs的潜力。我们在此处提供代码和模型：https://your-link-url。 

---
# Latent Principle Discovery for Language Model Self-Improvement 

**Title (ZH)**: 语言模型自我提升的潜在原理发现 

**Authors**: Keshav Ramji, Tahira Naseem, Ramón Fernandez Astudillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16927)  

**Abstract**: When language model (LM) users aim to improve the quality of its generations, it is crucial to specify concrete behavioral attributes that the model should strive to reflect. However, curating such principles across many domains, even non-exhaustively, requires a labor-intensive annotation process. To automate this process, we propose eliciting these latent attributes guiding model reasoning towards human-preferred responses by explicitly modeling them in a self-correction setting. Our approach mines new principles from the LM itself and compresses the discovered elements to an interpretable set via clustering. Specifically, we employ an approximation of posterior-regularized Monte Carlo Expectation-Maximization to both identify a condensed set of the most effective latent principles and teach the LM to strategically invoke them in order to intrinsically refine its responses. We demonstrate that bootstrapping our algorithm over multiple iterations enables smaller language models (7-8B parameters) to self-improve, achieving +8-10% in AlpacaEval win-rate, an average of +0.3 on MT-Bench, and +19-23% in principle-following win-rate on IFEval. We also show that clustering the principles yields interpretable and diverse model-generated constitutions while retaining model performance. The gains our method achieves highlight the potential of automated, principle-driven post-training recipes toward continual self-improvement. 

**Abstract (ZH)**: 通过自洽设置explicitly建模latent属性以自动化提升语言模型生成质量的原理驱动方法 

---
# CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework 

**Title (ZH)**: CAIN：通过两阶段恶意系统提示生成和优化框架操控LLM-人类对话 

**Authors**: Viet Pham, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.16888)  

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available. 

**Abstract (ZH)**: 大语言模型（LLMs）已经在多个应用中取得了进展，但也被认为容易受到 adversarial 攻击。本文介绍了一种新的安全威胁：通过操纵 LLMs 的系统提示来生成仅对特定目标问题（例如，“我应该支持哪位美国总统？”、“新冠疫苗安全吗？”）产生恶意回答的系统提示，而在其他问题上表现 benign。这种攻击具有严重性，因为它可以使恶意行为者通过在线传播看似无害但实际上有害的系统提示来进行大规模信息操控。为了展示这种攻击，我们开发了 CAIN 算法，该算法可以在黑盒环境中或不访问 LLM 参数的情况下自动为特定目标问题定制此类有害系统提示。CAIN 在开源和商用大语言模型上的评估显示了显著的 adversarial 影响。在未针对特定目标的攻击中或迫使 LLM 生成不正确答案时，CAIN 在目标问题上的 F1 值最多可降低 40%，同时在良性输入上保持高准确率。在针对特定目标的攻击中或迫使 LLM 生成特定有害回答时，CAIN 在这些目标响应上的 F1 值超过 70%，对良性问题的负面影响最小。我们的结果强调了在实际应用中增强大语言模型的健壮性措施的重要性。所有源代码将公开可用。 

---
# Don't "Overthink" Passage Reranking: Is Reasoning Truly Necessary? 

**Title (ZH)**: 别“过度思考”段落重排：推理真的必要吗？ 

**Authors**: Nour Jedidi, Yung-Sung Chuang, James Glass, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16886)  

**Abstract**: With the growing success of reasoning models across complex natural language tasks, researchers in the Information Retrieval (IR) community have begun exploring how similar reasoning capabilities can be integrated into passage rerankers built on Large Language Models (LLMs). These methods typically employ an LLM to produce an explicit, step-by-step reasoning process before arriving at a final relevance prediction. But, does reasoning actually improve reranking accuracy? In this paper, we dive deeper into this question, studying the impact of the reasoning process by comparing reasoning-based pointwise rerankers (ReasonRR) to standard, non-reasoning pointwise rerankers (StandardRR) under identical training conditions, and observe that StandardRR generally outperforms ReasonRR. Building on this observation, we then study the importance of reasoning to ReasonRR by disabling its reasoning process (ReasonRR-NoReason), and find that ReasonRR-NoReason is surprisingly more effective than ReasonRR. Examining the cause of this result, our findings reveal that reasoning-based rerankers are limited by the LLM's reasoning process, which pushes it toward polarized relevance scores and thus fails to consider the partial relevance of passages, a key factor for the accuracy of pointwise rerankers. 

**Abstract (ZH)**: 随着推理模型在复杂自然语言任务中取得越来越多的成功，信息检索（IR）领域的研究人员开始探索如何将类似的推理能力整合到基于大规模语言模型（LLMs）的段落重排序器中。这些方法通常使用LLM产生显式的、逐步的推理过程，最终得出最后的相关性预测。但推理是否真的能提高重排序准确性？在本文中，我们深入探讨了这一问题，通过在相同的训练条件下将基于推理的点wise重排序器（ReasonRR）与标准的非推理点wise重排序器（StandardRR）进行对比研究，发现StandardRR通常优于ReasonRR。在此基础上，我们进一步探讨了推理对ReasonRR的重要性，通过禁用其推理过程（ReasonRR-NoReason）进行研究，发现ReasonRR-NoReason竟然比ReasonRR更为有效。研究这一结果的原因，我们的发现表明，基于推理的重排序器受到LLM推理过程的限制，这促使它趋向于极化的相关性评分，从而未能考虑段落的部分相关性，这是点wise重排序器准确性的一个关键因素。 

---
# CASTILLO: Characterizing Response Length Distributions of Large Language Models 

**Title (ZH)**: CASTILLO: 大型语言模型响应长度分布Characterizing 

**Authors**: Daniel F. Perez-Ramirez, Dejan Kostic, Magnus Boman  

**Link**: [PDF](https://arxiv.org/pdf/2505.16881)  

**Abstract**: Efficiently managing compute resources for Large Language Model (LLM) inference remains challenging due to the inherently stochastic and variable lengths of autoregressive text generation. Accurately estimating response lengths in advance enables proactive resource allocation, yet existing approaches either bias text generation towards certain lengths or rely on assumptions that ignore model- and prompt-specific variability. We introduce CASTILLO, a dataset characterizing response length distributions across 13 widely-used open-source LLMs evaluated on seven distinct instruction-following corpora. For each $\langle$prompt, model$\rangle$ sample pair, we generate 10 independent completions using fixed decoding hyper-parameters, record the token length of each response, and publish summary statistics (mean, std-dev, percentiles), along with the shortest and longest completions, and the exact generation settings. Our analysis reveals significant inter- and intra-model variability in response lengths (even under identical generation settings), as well as model-specific behaviors and occurrences of partial text degeneration in only subsets of responses. CASTILLO enables the development of predictive models for proactive scheduling and provides a systematic framework for analyzing model-specific generation behaviors. We publicly release the dataset and code to foster research at the intersection of generative language modeling and systems. 

**Abstract (ZH)**: 高效管理大规模语言模型（LLM）推理计算资源仍具有挑战性，原因在于自回归文本生成固有的随机性和响应长度的变异性。准确地事先估计响应长度可以实现主动的资源分配，但现有方法要么偏向于生成特定长度的文本，要么依赖于忽略模型和提示特定变异性的情况性假设。我们引入了CASTILLO数据集，该数据集描述了13个广泛使用的开源LLM在七个不同指令遵循语料库上的响应长度分布。对于每个$\langle$提示，模型$\rangle$样本对，我们使用固定解码超参数生成10个独立的完成，记录每个响应的token长度，并发布汇总统计信息（均值、标准差、百分位数），以及最短和最长的完成，和精确的生成设置。我们的分析揭示了即使在相同的生成设置下，响应长度的显著跨模型和跨样本变异性，以及模型特定的行为和仅在某些响应中出现的部分文本退化现象。CASTILLO使得预测模型的开发成为可能，为分析模型特定的生成行为提供了一种系统框架。我们公开释放了数据集和代码，以促进生成语言建模与系统交叉领域的研究。 

---
# SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis 

**Title (ZH)**: SimpleDeepSearcher: 通过网页驱动的推理轨迹合成实现深度信息检索 

**Authors**: Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, Lei Fang, Zhongyuan Wang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16834)  

**Abstract**: Retrieval-augmented generation (RAG) systems have advanced large language models (LLMs) in complex deep search scenarios requiring multi-step reasoning and iterative information retrieval. However, existing approaches face critical limitations that lack high-quality training trajectories or suffer from the distributional mismatches in simulated environments and prohibitive computational costs for real-world deployment. This paper introduces SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap through strategic data engineering rather than complex training paradigms. Our approach synthesizes high-quality training data by simulating realistic user interactions in live web search environments, coupled with a multi-criteria curation strategy that optimizes the diversity and quality of input and output side. Experiments on five benchmarks across diverse domains demonstrate that SFT on only 871 curated samples yields significant improvements over RL-based baselines. Our work establishes SFT as a viable pathway by systematically addressing the data-scarce bottleneck, offering practical insights for efficient deep search systems. Our code is available at this https URL. 

**Abstract (ZH)**: 基于检索的生成（RAG）系统在复杂的深度搜索场景中推进了大型语言模型（LLMs），这些场景需要多步推理和迭代的信息检索。然而，现有方法面临关键限制，包括缺乏高质量的训练轨迹或在模拟环境中存在分布不匹配问题，以及在实际部署中高昂的计算成本。本文引入了SimpleDeepSearcher，这是一种轻量级但有效的框架，通过战略性数据工程而非复杂的训练范式来填补这一缺口。我们的方法通过模拟现实生活中的网络搜索环境中的真实用户交互来合成高质量的训练数据，并结合多标准策展策略，优化输入和输出的多样性和质量。横跨五个不同领域的基准实验表明，仅在871个策展样本上进行样本Fine-tuning（SFT）就能显著优于基于强化学习的方法。我们的研究系统地解决了数据稀缺瓶颈，为高效的深度搜索系统提供了实用见解。代码可在以下链接获取：this https URL。 

---
# Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs 

**Title (ZH)**: 卸载并不等同于删除：探究大语言模型中机器卸载的可逆性 

**Authors**: Xiaoyu Xu, Xiang Yue, Yang Liu, Qingqing Ye, Haibo Hu, Minxin Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.16831)  

**Abstract**: Unlearning in large language models (LLMs) is intended to remove the influence of specific data, yet current evaluations rely heavily on token-level metrics such as accuracy and perplexity. We show that these metrics can be misleading: models often appear to forget, but their original behavior can be rapidly restored with minimal fine-tuning, revealing that unlearning may obscure information rather than erase it. To diagnose this phenomenon, we introduce a representation-level evaluation framework using PCA-based similarity and shift, centered kernel alignment, and Fisher information. Applying this toolkit across six unlearning methods, three domains (text, code, math), and two open-source LLMs, we uncover a critical distinction between reversible and irreversible forgetting. In reversible cases, models suffer token-level collapse yet retain latent features; in irreversible cases, deeper representational damage occurs. We further provide a theoretical account linking shallow weight perturbations near output layers to misleading unlearning signals, and show that reversibility is modulated by task type and hyperparameters. Our findings reveal a fundamental gap in current evaluation practices and establish a new diagnostic foundation for trustworthy unlearning in LLMs. We provide a unified toolkit for analyzing LLM representation changes under unlearning and relearning: this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的遗忘意图是在去除特定数据影响的同时保留其他信息，但当前评估主要依赖于基于令牌级别的指标，如准确率和困惑度。我们表明，这些指标可能会误导：模型在表面上似乎已经忘记，但通过少量微调即可迅速恢复其原始行为，揭示了遗忘可能掩盖信息而非彻底删除信息。为了诊断这一现象，我们引入了一种基于PCA的相似性和偏移、中心核对齐和Fisher信息的表示层面评估框架。应用这一工具包，我们发现可逆遗忘和不可逆遗忘之间的关键区别。在可逆的情况下，模型在基于令牌的层面出现崩溃，但仍保留潜在特征；而在不可逆的情况下，深层表示遭受了更严重的破坏。我们进一步提供了将浅层权重扰动与误导性的遗忘信号联系起来的理论解释，并表明可逆性受任务类型和超参数的调节。我们的发现揭示了当前评估实践中存在的根本缺陷，并建立了规模语言模型可信遗忘的新诊断基础。我们提供了一个统一的工具包来分析语言模型在遗忘和重新学习下的表示变化：这个链接 

---
# Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability 

**Title (ZH)**: 意外的不对齐：语言模型微调诱导出意外的脆弱性 

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16789)  

**Abstract**: As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at this https URL. 

**Abstract (ZH)**: 随着大型语言模型的流行，它们对对抗攻击的脆弱性仍然是主要关切。虽然在领域特定数据集上进行微调以提高模型性能是一种常见做法，但这可能会在底层模型中引入脆弱性。在本文中，我们研究了由微调数据特性引发的意外失准问题。我们首先识别潜在的相关因素，如语言特征、语义相似性和毒性。然后评估这些微调模型的对抗性能，并评估数据集因素与攻击成功率之间的关联性。最后，我们探索潜在的因果关系，提供了对抗防御策略的新见解，并强调了数据集设计在保持模型对齐方面的重要作用。我们的代码可在以下链接获取：this https URL。 

---
# CoTSRF: Utilize Chain of Thought as Stealthy and Robust Fingerprint of Large Language Models 

**Title (ZH)**: CoTSRF: 将思维链作为大型语言模型隐蔽且 robust 的指纹rolloater 

**Authors**: Zhenzhen Ren, GuoBiao Li, Sheng Li, Zhenxing Qian, Xinpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16785)  

**Abstract**: Despite providing superior performance, open-source large language models (LLMs) are vulnerable to abusive usage. To address this issue, recent works propose LLM fingerprinting methods to identify the specific source LLMs behind suspect applications. However, these methods fail to provide stealthy and robust fingerprint verification. In this paper, we propose a novel LLM fingerprinting scheme, namely CoTSRF, which utilizes the Chain of Thought (CoT) as the fingerprint of an LLM. CoTSRF first collects the responses from the source LLM by querying it with crafted CoT queries. Then, it applies contrastive learning to train a CoT extractor that extracts the CoT feature (i.e., fingerprint) from the responses. Finally, CoTSRF conducts fingerprint verification by comparing the Kullback-Leibler divergence between the CoT features of the source and suspect LLMs against an empirical threshold. Various experiments have been conducted to demonstrate the advantage of our proposed CoTSRF for fingerprinting LLMs, particularly in stealthy and robust fingerprint verification. 

**Abstract (ZH)**: 尽管开源大型语言模型（LLM）性能优越，但极易遭受滥用。为应对这一问题，近期的研究提出了一种LLM指纹识别方法，以确定嫌疑应用程序背后的特定源LLM。然而，这些方法未能提供隐蔽且 robust 的指纹验证方法。本文提出了一种新颖的LLM指纹识别方案，即CoTSRF，该方案利用Chain of Thought（CoT）作为LLM的指纹。CoTSRF首先通过定制的CoT查询收集源LLM的响应，然后应用对比学习训练一个CoT提取器，从响应中提取CoT特征（即指纹）。最后，CoTSRF通过将源LLM和嫌疑LLM的CoT特征的Kullback-Leibler距离与经验阈值进行比较来进行指纹验证。各种实验表明，CoTSRF在隐蔽和robust的指纹验证方面具有显著优势。 

---
# When Safety Detectors Aren't Enough: A Stealthy and Effective Jailbreak Attack on LLMs via Steganographic Techniques 

**Title (ZH)**: 当安全检测器不够用时：通过隐写技术对大语言模型进行隐蔽而有效的越狱攻击 

**Authors**: Jianing Geng, Biao Yi, Zekun Fei, Tongxi Wu, Lihai Nie, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16765)  

**Abstract**: Jailbreak attacks pose a serious threat to large language models (LLMs) by bypassing built-in safety mechanisms and leading to harmful outputs. Studying these attacks is crucial for identifying vulnerabilities and improving model security. This paper presents a systematic survey of jailbreak methods from the novel perspective of stealth. We find that existing attacks struggle to simultaneously achieve toxic stealth (concealing toxic content) and linguistic stealth (maintaining linguistic naturalness). Motivated by this, we propose StegoAttack, a fully stealthy jailbreak attack that uses steganography to hide the harmful query within benign, semantically coherent text. The attack then prompts the LLM to extract the hidden query and respond in an encrypted manner. This approach effectively hides malicious intent while preserving naturalness, allowing it to evade both built-in and external safety mechanisms. We evaluate StegoAttack on four safety-aligned LLMs from major providers, benchmarking against eight state-of-the-art methods. StegoAttack achieves an average attack success rate (ASR) of 92.00%, outperforming the strongest baseline by 11.0%. Its ASR drops by less than 1% even under external detection (e.g., Llama Guard). Moreover, it attains the optimal comprehensive scores on stealth detection metrics, demonstrating both high efficacy and exceptional stealth capabilities. The code is available at this https URL 

**Abstract (ZH)**: Jailbreak攻击对大型语言模型（LLMs）构成严重威胁，通过绕过内置的安全机制并导致有害输出。研究这些攻击对于识别漏洞和改善模型安全性至关重要。本文从隐蔽性的新颖视角系统总结了 Jailbreak 方法。我们发现现有的攻击难以同时实现有毒隐蔽（隐藏有毒内容）和语言隐蔽（保持语言自然性）。受此启发，我们提出了 StegoAttack，这是一种完全隐蔽的 Jailbreak 攻击，利用隐写术将有害查询隐藏在 benign、语义一致的文本中。攻击随后促使LLM提取隐藏查询并以加密方式响应。该方法有效地隐藏了恶意意图的同时保留了自然性，使其能够避开内置和外部安全机制。我们在来自主要提供商的四种安全对齐的LLM上评估了StegoAttack，并将其与八种最先进的方法进行了基准测试。StegoAttack实现了平均攻击成功率（ASR）为92.00%，比最强基准高11.0%。即使在外部检测下（例如，Llama Guard），其ASR也仅下降不到1%。此外，它在隐蔽性检测指标上取得了最优综合得分，展示了高效性和卓越的隐蔽性能力。代码可在以下链接获取。 

---
# TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning 

**Title (ZH)**: TRIM: 实现目标行 wise 迭代度量驱动稀疏性的极度压缩 

**Authors**: Florentin Beck, William Rudman, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2505.16743)  

**Abstract**: Large Language Models (LLMs) present significant computational and memory challenges due to their extensive size, making pruning essential for their efficient deployment. Existing one-shot pruning methods often apply uniform sparsity constraints across layers or within each layer, resulting in suboptimal performance, especially at high sparsity ratios. This work introduces TRIM (Targeted Row-wise Iterative Metric-driven pruning), a novel approach that applies varying sparsity ratios to individual output dimensions (rows) within each layer. TRIM employs an iterative adjustment process guided by quality metrics to optimize dimension-wise sparsity allocation, focusing on reducing variance in quality retention across outputs to preserve critical information. TRIM can be seamlessly integrated with existing layer-wise pruning strategies. Our evaluations on perplexity and zero-shot tasks across diverse LLM families (Qwen2.5, LLaMA-2, and OPT) and sparsity levels demonstrate that TRIM achieves new state-of-the-art results and enhances stability. For instance, at 80% sparsity, TRIM reduces perplexity by 48% for Qwen2.5-14B and over 90% for OPT-13B compared to baseline methods. We conclude that fine-grained, dimension-wise sparsity adaptation is crucial for pushing the limits of extreme LLM compression. Code available at: this https URL 

**Abstract (ZH)**: 针对大规模语言模型的逐行迭代度量驱动剪枝(TRIM)：细粒度的维度-wise稀疏性适配对于极致压缩大语言模型至关重要 

---
# Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization 

**Title (ZH)**: 通过安全意识探查优化缓解大语言模型微调风险 

**Authors**: Chengcan Wu, Zhixin Zhang, Zeming Wei, Yihao Zhang, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16737)  

**Abstract**: The significant progress of large language models (LLMs) has led to remarkable achievements across numerous applications. However, their ability to generate harmful content has sparked substantial safety concerns. Despite the implementation of safety alignment techniques during the pre-training phase, recent research indicates that fine-tuning LLMs on adversarial or even benign data can inadvertently compromise their safety. In this paper, we re-examine the fundamental issue of why fine-tuning on non-harmful data still results in safety degradation. We introduce a safety-aware probing (SAP) optimization framework designed to mitigate the safety risks of fine-tuning LLMs. Specifically, SAP incorporates a safety-aware probe into the gradient propagation process, mitigating the model's risk of safety degradation by identifying potential pitfalls in gradient directions, thereby enhancing task-specific performance while successfully preserving model safety. Our extensive experimental results demonstrate that SAP effectively reduces harmfulness below the original fine-tuned model and achieves comparable test loss to standard fine-tuning methods. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的重要进展已经在众多应用中取得了显著成果，然而其生成有害内容的能力引发了重大安全关切。尽管在预训练阶段实施了安全对齐技术，近期研究显示，对LLMs进行对抗性或甚至良性数据微调仍然可能导致安全降级。在本文中，我们重新审视了尽管使用非有害数据进行微调但仍导致安全降级的根本原因。我们提出了一种安全感知探针（SAP）优化框架，旨在减轻对LLMs进行微调的安全风险。具体而言，SAP将安全感知探针融入梯度传播过程，通过识别梯度方向中的潜在风险来降低模型的安全降级风险，从而在增强任务特定性能的同时成功保持模型安全。我们的大量实验结果表明，SAP有效减少了有害性并达到了与标准微调方法相当的测试损失。我们的代码可在以下链接获取：this https URL。 

---
# Breaking mBad! Supervised Fine-tuning for Cross-Lingual Detoxification 

**Title (ZH)**: 打破mBad！监督微调在跨语言去污中的应用 

**Authors**: Himanshu Beniwal, Youngwoo Kim, Maarten Sap, Soham Dan, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16722)  

**Abstract**: As large language models (LLMs) become increasingly prevalent in global applications, ensuring that they are toxicity-free across diverse linguistic contexts remains a critical challenge. We explore "Cross-lingual Detoxification", a cross-lingual paradigm that mitigates toxicity, enabling detoxification capabilities to transfer between high and low-resource languages across different script families. We analyze cross-lingual detoxification's effectiveness through 504 extensive settings to evaluate toxicity reduction in cross-distribution settings with limited data and investigate how mitigation impacts model performance on non-toxic tasks, revealing trade-offs between safety and knowledge preservation. Our code and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 跨语言去毒研究：一种在不同文字体系的高资源和低资源语言之间转移去毒能力的跨语言 paradigm 

---
# Training Long-Context LLMs Efficiently via Chunk-wise Optimization 

**Title (ZH)**: 通过分块优化高效训练长上下文LLMs 

**Authors**: Wenhao Li, Yuxin Zhang, Gen Luo, Daohai Yu, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.16710)  

**Abstract**: While long-context large language models (LLMs) exhibit remarkable document processing capabilities, their prohibitively high training costs often hinder customized applications. To mitigate this issue, we propose \textit{Sequential Chunk-wise Optimization} (SeCO), a memory-efficient training paradigm that partitions lengthy inputs into manageable chunks. Each chunk independently constructs its computational graph and performs localized backpropagation, ensuring that only one chunk's forward activations are stored in memory. Building on SeCO, we further introduce \textit{Sparse Chunk-wise Optimization} (SpaCO), which reduces computational overhead by selectively propagating gradients to specific chunks and incorporates a carefully designed compensation factor to ensure unbiased gradient estimation. SpaCO decouples the computational cost of backpropagation from the context length, enabling training time to gradually converge to inference time as sequences become longer. Implemented as lightweight training wrappers, both SeCO and SpaCO offer substantial practical benefits. For example, when fine-tuning an 8B model with LoRA on a single RTX 3090 GPU, SeCO expands maximum sequence length from 1K to 16K tokens, while SpaCO demonstrates accelerated training speed -- achieving up to 3x faster than SeCO under the same experimental setup. These innovations provide new insights into optimizing long-context models, making them more accessible for practical applications. We have open-sourced the code at \href{this https URL}{here}. 

**Abstract (ZH)**: 长上下文大型语言模型展现出卓越的文档处理能力，但由于高昂的训练成本常常限制了定制化应用。为缓解这一问题，我们提出了 Sequential Chunk-wise Optimization (SeCO)——一种内存高效的训练范式，将长输入分割为可管理的片段。每个片段独立构建计算图并执行局部反向传播，确保仅存储一个片段的前向激活。在此基础上，我们进一步引入 Sparse Chunk-wise Optimization (SpaCO)，通过选择性地将梯度传播到特定片段来减少计算开销，并结合精心设计的补偿因子以确保无偏的梯度估计。SpaCO 将反向传播的计算成本与上下文长度解耦，使训练时间随着序列变长逐渐收敛至推理时间。作为轻量级训练包装器，SeCO 和 SpaCO 提供了显著的实际益处。例如，在使用 LoRA 对一个 8B 模型进行微调时，SeCO 将最大序列长度从 1K 扩展至 16K 令牌，而 SpaCO 在相同的实验设置下展示了加速的训练速度，比 SeCO 快至 3 倍。这些创新为优化长上下文模型提供了新的思路，使它们更能适用于实际应用。我们已将代码开源在 [这里](this https URL)。 

---
# Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence 

**Title (ZH)**: 超越归纳头部：上下文内元学习诱导多阶段电路 emergence 

**Authors**: Gouki Minegishi, Hiroki Furuta, Shohei Taniguchi, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16694)  

**Abstract**: Transformer-based language models exhibit In-Context Learning (ICL), where predictions are made adaptively based on context. While prior work links induction heads to ICL through a sudden jump in accuracy, this can only account for ICL when the answer is included within the context. However, an important property of practical ICL in large language models is the ability to meta-learn how to solve tasks from context, rather than just copying answers from context; how such an ability is obtained during training is largely unexplored. In this paper, we experimentally clarify how such meta-learning ability is acquired by analyzing the dynamics of the model's circuit during training. Specifically, we extend the copy task from previous research into an In-Context Meta Learning setting, where models must infer a task from examples to answer queries. Interestingly, in this setting, we find that there are multiple phases in the process of acquiring such abilities, and that a unique circuit emerges in each phase, contrasting with the single-phases change in induction heads. The emergence of such circuits can be related to several phenomena known in large language models, and our analysis lead to a deeper understanding of the source of the transformer's ICL ability. 

**Abstract (ZH)**: 基于Transformer的语言模型表现出的内省式学习（ICL），其中预测是根据上下文自适应地做出的。尽管先前的工作通过准确性的突然飞跃将归纳头与ICL联系起来，这只能解释当答案包含在上下文中时的ICL现象。然而，实际大型语言模型中的ICL的一个重要特性是其解决任务的能力是通过上下文进行元学习获得的，而不仅仅是从上下文中复制答案；这一能力在训练过程中是如何获得的仍然鲜有探讨。在本文中，我们通过分析模型电路在训练过程中的动态性，实验证实了这种元学习能力是如何获得的。具体而言，我们将复制任务扩展为一个内省式元学习环境，在这种环境中，模型必须从示例中推断出任务以回答查询。有趣的是，在这种环境中，我们发现获得此类能力的过程存在多个阶段，并且每个阶段会出现一个独特的电路，这与归纳头的单阶段变化相反。这些电路的出现与已知的大型语言模型中的几种现象有关，而我们的分析加深了对变压器ICL能力来源的理解。 

---
# Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator 

**Title (ZH)**: 你的预训练大规模语言模型实际上是无监督的置信度校准器 

**Authors**: Beier Luo, Shuoyuan Wang, Yixuan Li, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16690)  

**Abstract**: Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $\tau$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $\tau$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks. 

**Abstract (ZH)**: 大型语言模型的后训练对于适应预训练语言模型（PLMs）以与人类偏好和下游任务对齐是必不可少的。后训练语言模型（PoLMs）常常表现出过度自信，对正确和错误的输出都赋予很高的置信度，这可能在关键应用中削弱可靠性。校准PoLMs的主要障碍是缺乏用于个体下游任务的标注数据。为此，我们提出了一种新颖的无监督方法——分歧感知置信校准（DACA），以优化后训练置信校准（如温度$\tau$）的参数。该方法受到PLM和PoLM之间预测分歧导致置信对齐时置信度低估的启发。理论上，PLM的置信度低估了PoLM在分歧样本上的预测准确性，导致更大的$\tau$值并产生不自信的预测。DACA通过仅选择一致性样本进行校准，有效地解耦了分歧的影响。从而避免了由于分歧样本导致的温度缩放中的过大$\tau$值，提高了校准性能。广泛的实验表明了该方法的有效性，在常用基准上将开源和API基于的大型语言模型（如GPT-4o）的平均ECE提升了高达15.08%。 

---
# R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO 

**Title (ZH)**: R1-ShareVL: 通过Share-GRPO激励多模态大型语言模型的推理能力 

**Authors**: Huanjin Yao, Qixiang Yin, Jingyi Zhang, Min Yang, Yibo Wang, Wenhao Wu, Fei Su, Li Shen, Minghui Qiu, Dacheng Tao, Jiaxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16673)  

**Abstract**: In this work, we aim to incentivize the reasoning ability of Multimodal Large Language Models (MLLMs) via reinforcement learning (RL) and develop an effective approach that mitigates the sparse reward and advantage vanishing issues during RL. To this end, we propose Share-GRPO, a novel RL approach that tackle these issues by exploring and sharing diverse reasoning trajectories over expanded question space. Specifically, Share-GRPO first expands the question space for a given question via data transformation techniques, and then encourages MLLM to effectively explore diverse reasoning trajectories over the expanded question space and shares the discovered reasoning trajectories across the expanded questions during RL. In addition, Share-GRPO also shares reward information during advantage computation, which estimates solution advantages hierarchically across and within question variants, allowing more accurate estimation of relative advantages and improving the stability of policy training. Extensive evaluations over six widely-used reasoning benchmarks showcase the superior performance of our method. Code will be available at this https URL. 

**Abstract (ZH)**: 本研究旨在通过强化学习激励多模态大型语言模型的推理能力，并开发一种有效的方法，以缓解强化学习过程中稀疏奖励和优势消失问题。为此，我们提出了Share-GRPO，这是一种通过探索和在扩展的问题空间中共享多样化的推理轨迹来解决这些问题的新型强化学习方法。Specifically, Share-GRPO 首先通过数据变换技术扩展给定问题的问题空间，然后在强化学习过程中促使多模态大型语言模型有效探索扩展后问题空间中的多样化推理轨迹，并在扩展的问题上共享发现的推理轨迹。此外，Share-GRPO 在优势计算过程中也共享奖励信息，以分层次地估计不同变体问题之间的解的优势，从而更准确地估计相对优势并提高策略训练的稳定性。在六个广泛使用的推理基准上的广泛评估展示了该方法的优越性能。代码将发布在此 URL。 

---
# BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models 

**Title (ZH)**: BitHydra: 针对大型语言模型的位翻转推理成本攻击 

**Authors**: Xiaobei Yan, Yiming Li, Zhaoxin Fan, Han Qiu, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16670)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities across a wide range of applications, but their ever-increasing size and resource demands make them vulnerable to inference cost attacks, where attackers induce victim LLMs to generate the longest possible output content. In this paper, we revisit existing inference cost attacks and reveal that these methods can hardly produce large-scale malicious effects since they are self-targeting, where attackers are also the users and therefore have to execute attacks solely through the inputs, whose generated content will be charged by LLMs and can only directly influence themselves. Motivated by these findings, this paper introduces a new type of inference cost attacks (dubbed 'bit-flip inference cost attack') that target the victim model itself rather than its inputs. Specifically, we design a simple yet effective method (dubbed 'BitHydra') to effectively flip critical bits of model parameters. This process is guided by a loss function designed to suppress <EOS> token's probability with an efficient critical bit search algorithm, thus explicitly defining the attack objective and enabling effective optimization. We evaluate our method on 11 LLMs ranging from 1.5B to 14B parameters under both int8 and float16 settings. Experimental results demonstrate that with just 4 search samples and as few as 3 bit flips, BitHydra can force 100% of test prompts to reach the maximum generation length (e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its efficiency, scalability, and strong transferability across unseen inputs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种应用中展现了出色的性能，但其不断增加的规模和资源需求使其容易受到推理成本攻击的影响，攻击者可诱导受害的LLMs生成尽可能长的输出内容。本文重新审视了现有的推理成本攻击，并揭示这些方法难以产生大规模恶意效果，因为它们是自我靶向的，攻击者同时也是用户，因此必须通过输入来执行攻击，这些输入生成的内容将被LLMs收费，并只能直接影响自身。受此发现的启发，本文提出了一种新的推理成本攻击类型（称为“位翻转推理成本攻击”），该攻击直接针对受害模型本身而非其输入。具体地，我们设计了一种简单而有效的方法（称为“BitHydra”）来有效翻转模型参数的关键位。这一过程由一个损失函数引导，该损失函数通过一种高效的关键位搜索算法抑制<EOS>标记的概率，从而明确定义攻击目标并实现有效的优化。我们在涵盖1.5B至14B参数的11个LLM（包括int8和float16设置）上评估了该方法。实验结果表明，只需4个搜索样本和3次位翻转，BitHydra就能迫使代表性LLM（如LLaMA3）上的100%测试提示达到最大生成长度（例如，2048个标记），这突显了其高效性、可扩展性和在未见输入上的强转移能力。 

---
# Collaboration among Multiple Large Language Models for Medical Question Answering 

**Title (ZH)**: 多个大型语言模型在医疗问答中的协作 

**Authors**: Kexin Shang, Chia-Hsuan Chang, Christopher C. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16648)  

**Abstract**: Empowered by vast internal knowledge reservoir, the new generation of large language models (LLMs) demonstrate untapped potential to tackle medical tasks. However, there is insufficient effort made towards summoning up a synergic effect from multiple LLMs' expertise and background. In this study, we propose a multi-LLM collaboration framework tailored on a medical multiple-choice questions dataset. Through post-hoc analysis on 3 pre-trained LLM participants, our framework is proved to boost all LLMs reasoning ability as well as alleviate their divergence among questions. We also measure an LLM's confidence when it confronts with adversary opinions from other LLMs and observe a concurrence between LLM's confidence and prediction accuracy. 

**Abstract (ZH)**: 受大规模内部知识库支持，新一代大型语言模型（LLMs）展示了应对医疗任务的未充分利用的潜力。然而，尚未有充分努力将多个LLMs的专业背景汇聚产生协同效果。在本研究中，我们提出了一种针对医学选择题数据集的多LLM协作框架。通过事后分析3个预训练LLM的表现，我们的框架证明能够提升所有LLMs的推理能力，并减少它们在不同问题上的分歧。我们还测量了LLM在面对其他LLMs的对立观点时的信心，并观察到LLM的信心与其预测准确率之间的关联。 

---
# From Evaluation to Defense: Advancing Safety in Video Large Language Models 

**Title (ZH)**: 从评估到防御：推进视频大规模语言模型的安全性 

**Authors**: Yiwei Sun, Peiqi Jiang, Chuanbin Liu, Luohao Lin, Zhiying Lu, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16643)  

**Abstract**: While the safety risks of image-based large language models have been extensively studied, their video-based counterparts (Video LLMs) remain critically under-examined. To systematically study this problem, we introduce \textbf{VideoSafetyBench (VSB-77k) - the first large-scale, culturally diverse benchmark for Video LLM safety}, which compromises 77,646 video-query pairs and spans 19 principal risk categories across 10 language communities. \textit{We reveal that integrating video modality degrades safety performance by an average of 42.3\%, exposing systemic risks in multimodal attack exploitation.} To address this vulnerability, we propose \textbf{VideoSafety-R1}, a dual-stage framework achieving unprecedented safety gains through two innovations: (1) Alarm Token-Guided Safety Fine-Tuning (AT-SFT) injects learnable alarm tokens into visual and textual sequences, enabling explicit harm perception across modalities via multitask objectives. (2) Then, Safety-Guided GRPO enhances defensive reasoning through dynamic policy optimization with rule-based rewards derived from dual-modality verification. These components synergize to shift safety alignment from passive harm recognition to active reasoning. The resulting framework achieves a 65.1\% improvement on VSB-Eval-HH, and improves by 59.1\%, 44.3\%, and 15.0\% on the image safety datasets MMBench, VLGuard, and FigStep, respectively. \textit{Our codes are available in the supplementary materials.} \textcolor{red}{Warning: This paper contains examples of harmful language and videos, and reader discretion is recommended.} 

**Abstract (ZH)**: 视频基础大型语言模型的安全风险：VideoSafetyBench（VSB-77k）-首个大规模跨文化视频LLM安全基准 

---
# SSR-Zero: Simple Self-Rewarding Reinforcement Learning for Machine Translation 

**Title (ZH)**: SSR-Zero: 简单的自奖励强化学习机器翻译 

**Authors**: Wenjie Yang, Mao Zheng, Mingyang Song, Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16637)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable capabilities in machine translation (MT). However, most advanced MT-specific LLMs heavily rely on external supervision signals during training, such as human-annotated reference data or trained reward models (RMs), which are often expensive to obtain and challenging to scale. To overcome this limitation, we propose a Simple Self-Rewarding (SSR) Reinforcement Learning (RL) framework for MT that is reference-free, fully online, and relies solely on self-judging rewards. Training with SSR using 13K monolingual examples and Qwen-2.5-7B as the backbone, our model SSR-Zero-7B outperforms existing MT-specific LLMs, e.g., TowerInstruct-13B and GemmaX-28-9B, as well as larger general LLMs like Qwen2.5-32B-Instruct in English $\leftrightarrow$ Chinese translation tasks from WMT23, WMT24, and Flores200 benchmarks. Furthermore, by augmenting SSR with external supervision from COMET, our strongest model, SSR-X-Zero-7B, achieves state-of-the-art performance in English $\leftrightarrow$ Chinese translation, surpassing all existing open-source models under 72B parameters and even outperforming closed-source models, e.g., GPT-4o and Gemini 1.5 Pro. Our analysis highlights the effectiveness of the self-rewarding mechanism compared to the external LLM-as-a-judge approach in MT and demonstrates its complementary benefits when combined with trained RMs. Our findings provide valuable insight into the potential of self-improving RL methods. We have publicly released our code, data and models. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机器翻译（MT）方面 recently 已经展现了令人瞩目的能力。然而，大多数先进的MT专用LLMs在训练过程中严重依赖外部监督信号，如人工标注的参考数据或训练好的奖励模型（RMs），这些信号往往非常昂贵且难以扩展。为克服这一限制，我们提出了一种基于参考的Simple Self-Rewarding（SSR）强化学习（RL）框架，该框架完全在线，仅依赖自我评判奖励进行训练。使用13000个单语例句和Qwen-2.5-7B作为骨干模型进行训练后，我们的SSR-Zero-7B模型在WMT23、WMT24和Flores200基准的英译中和中译英任务中优于现有MT专用LLMs，例如TowerInstruct-13B和GemmaX-28-9B，以及更大的通用LLMs如Qwen2.5-32B-Instruct。此外，通过结合COMET提供的外部监督，我们最强的模型SSR-X-Zero-7B在英译中任务中达到了最先进的性能，超过了所有现有参数量小于72B的开源模型，甚至优于某些封闭源模型，例如GPT-4o和Gemini 1.5 Pro。我们的分析突出了自我奖励机制在MT中比外部LLM-as-a-judge方法的有效性，并表明将其与训练好的RMs结合使用时具有互补优势。我们的发现为自我提升的RL方法的潜在能力提供了宝贵见解。我们已公开发布了我们的代码、数据和模型。 

---
# Steering Large Language Models for Machine Translation Personalization 

**Title (ZH)**: 指导大型语言模型实现机器翻译个性化 

**Authors**: Daniel Scalena, Gabriele Sarti, Arianna Bisazza, Elisabetta Fersini, Malvina Nissim  

**Link**: [PDF](https://arxiv.org/pdf/2505.16612)  

**Abstract**: High-quality machine translation systems based on large language models (LLMs) have simplified the production of personalized translations reflecting specific stylistic constraints. However, these systems still struggle in settings where stylistic requirements are less explicit and might be harder to convey via prompting. We explore various strategies for personalizing LLM-generated translations in low-resource settings, focusing on the challenging literary translation domain. We explore prompting strategies and inference-time interventions for steering model generations towards a personalized style, and propose a contrastive framework exploiting latent concepts extracted from sparse autoencoders to identify salient personalization properties. Our results show that steering achieves strong personalization while preserving translation quality. We further examine the impact of steering on LLM representations, finding model layers with a relevant impact for personalization are impacted similarly by multi-shot prompting and our steering method, suggesting similar mechanism at play. 

**Abstract (ZH)**: 基于大规模语言模型的高质机器翻译系统简化了反映特定風格约束的个性化翻译生产。然而，这些系统在风格要求不够明确且难以通过提示传达的情况下仍面临挑战。我们探讨了在低资源环境下个性化大规模语言模型生成翻译的各种策略，重点关注具有挑战性的文学翻译领域。我们探索了提示策略和推理时干预方法，以引导模型生成个性化风格，并提出了一种利用稀疏自编码器提取的潜在概念进行对比的框架，以识别重要的个性化属性。我们的结果表明，引导不仅可以实现强大的个性化，还能保持翻译质量。我们进一步研究了引导对大规模语言模型表示的影响，发现对个性化有相关影响的模型层级，在多轮提示和我们的引导方法作用下表现出相似的影响，表明可能存在相似的机制。 

---
# O$^2$-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering 

**Title (ZH)**: O$^2$-Searcher: 一种基于搜索的开放域无止境问答代理模型 

**Authors**: Jianbiao Mei, Tao Hu, Daocheng Fu, Licheng Wen, Xuemeng Yang, Rong Wu, Pinlong Cai, Xing Gao, Yu Yang, Chengjun Xie, Botian Shi, Yong Liu, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16582)  

**Abstract**: Large Language Models (LLMs), despite their advancements, are fundamentally limited by their static parametric knowledge, hindering performance on tasks requiring open-domain up-to-date information. While enabling LLMs to interact with external knowledge environments is a promising solution, current efforts primarily address closed-end problems. Open-ended questions, which characterized by lacking a standard answer or providing non-unique and diverse answers, remain underexplored. To bridge this gap, we present O$^2$-Searcher, a novel search agent leveraging reinforcement learning to effectively tackle both open-ended and closed-ended questions in the open domain. O$^2$-Searcher leverages an efficient, locally simulated search environment for dynamic knowledge acquisition, effectively decoupling the external world knowledge from model's sophisticated reasoning processes. It employs a unified training mechanism with meticulously designed reward functions, enabling the agent to identify problem types and adapt different answer generation strategies. Furthermore, to evaluate performance on complex open-ended tasks, we construct O$^2$-QA, a high-quality benchmark featuring 300 manually curated, multi-domain open-ended questions with associated web page caches. Extensive experiments show that O$^2$-Searcher, using only a 3B model, significantly surpasses leading LLM agents on O$^2$-QA. It also achieves SOTA results on various closed-ended QA benchmarks against similarly-sized models, while performing on par with much larger ones. 

**Abstract (ZH)**: 开放领域中面向开放与封闭问题的搜索代理O$^2$-Searcher 

---
# Finetuning-Activated Backdoors in LLMs 

**Title (ZH)**: Finetuning-激活的Backdoors在LLMs中 

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16567)  

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs. 

**Abstract (ZH)**: 开放访问的大语言模型（LLMs）的微调已成为实现任务特定性能改进的标准做法。迄今为止，微调一直被视为一个受控和安全的过程，其中在良性数据集上进行训练会导致可预测的行为。在本文中，我们首次证明了一个对手可以创建看似无害但经过下游微调后表现出恶意行为的中毒LLMs。为此，我们提出了一种名为FAB（Finetuning-Activated Backdoor）的攻击方法，通过元学习技术对LLM进行中毒，模拟下游微调过程，明确优化使微调后的模型表现出恶意行为。同时，中毒的LLM受到正则化以保留其一般能力，并在微调前不表现出恶意行为。当用户在其自己的数据集上微调看似无害的模型时，他们会不知不觉地触发其隐藏的后门行为。我们展示了FAB在多个LLM和三种目标行为（未经请求的广告、拒绝及脱疆）上的有效性。此外，我们还展示了FAB后门对用户不同微调选择（如数据集、步骤数、调度器等）的鲁棒性。我们的发现挑战了关于微调安全性的现有假设，揭示了又一个利用LLMs复杂性的关键攻击向量。 

---
# DuFFin: A Dual-Level Fingerprinting Framework for LLMs IP Protection 

**Title (ZH)**: DuFFin：LLMs IP保护的双重层级指纹框架 

**Authors**: Yuliang Yan, Haochun Tang, Shuo Yan, Enyan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2505.16530)  

**Abstract**: Large language models (LLMs) are considered valuable Intellectual Properties (IP) for legitimate owners due to the enormous computational cost of training. It is crucial to protect the IP of LLMs from malicious stealing or unauthorized deployment. Despite existing efforts in watermarking and fingerprinting LLMs, these methods either impact the text generation process or are limited in white-box access to the suspect model, making them impractical. Hence, we propose DuFFin, a novel $\textbf{Du}$al-Level $\textbf{Fin}$gerprinting $\textbf{F}$ramework for black-box setting ownership verification. DuFFin extracts the trigger pattern and the knowledge-level fingerprints to identify the source of a suspect model. We conduct experiments on a variety of models collected from the open-source website, including four popular base models as protected LLMs and their fine-tuning, quantization, and safety alignment versions, which are released by large companies, start-ups, and individual users. Results show that our method can accurately verify the copyright of the base protected LLM on their model variants, achieving the IP-ROC metric greater than 0.95. Our code is available at this https URL. 

**Abstract (ZH)**: 一种用于黑盒设置的Dual-Level Fingerprinting Framework (DuFFin) 用于所有权验证 

---
# Benchmarking and Pushing the Multi-Bias Elimination Boundary of LLMs via Causal Effect Estimation-guided Debiasing 

**Title (ZH)**: 基于因果效应估计导向的去偏见方法，benchmarking及推动大型语言模型多偏见消除边界 

**Authors**: Zhouhao Sun, Zhiyuan Kan, Xiao Ding, Li Du, Yang Zhao, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16522)  

**Abstract**: Despite significant progress, recent studies have indicated that current large language models (LLMs) may still utilize bias during inference, leading to the poor generalizability of LLMs. Some benchmarks are proposed to investigate the generalizability of LLMs, with each piece of data typically containing one type of controlled bias. However, a single piece of data may contain multiple types of biases in practical applications. To bridge this gap, we propose a multi-bias benchmark where each piece of data contains five types of biases. The evaluations conducted on this benchmark reveal that the performance of existing LLMs and debiasing methods is unsatisfying, highlighting the challenge of eliminating multiple types of biases simultaneously. To overcome this challenge, we propose a causal effect estimation-guided multi-bias elimination method (CMBE). This method first estimates the causal effect of multiple types of biases simultaneously. Subsequently, we eliminate the causal effect of biases from the total causal effect exerted by both the semantic information and biases during inference. Experimental results show that CMBE can effectively eliminate multiple types of bias simultaneously to enhance the generalizability of LLMs. 

**Abstract (ZH)**: 尽管取得了显著进展，近期研究表明，当前的大规模语言模型（LLMs）在推断过程中仍然可能存在偏见，导致LLMs的泛化能力不足。一些基准被提出以研究LLMs的泛化能力，每条数据通常包含一种控制偏见。然而，在实际应用中，一条数据可能包含多种类型的偏见。为解决这一问题，我们提出了一个多偏见基准，每条数据包含五类偏见。在该基准上的评估显示，现有的LLMs和去偏方法表现不佳，突显了同时消除多种偏见的挑战。为克服这一挑战，我们提出了一个基于因果效应估计的多偏见消除方法（CMBE）。该方法首先同时估计多种偏见的因果效应。随后，我们从语义信息和偏见共同作用的总因果效应中消除偏见的因果效应。实验结果表明，CMBE能够同时有效消除多种偏见，提升LLMs的泛化能力。 

---
# Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs 

**Title (ZH)**: 隐藏状态在隐藏什么？测试大型语言模型事实编码能力的极限 

**Authors**: Giovanni Servedio, Alessandro De Bellis, Dario Di Palma, Vito Walter Anelli, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16520)  

**Abstract**: Factual hallucinations are a major challenge for Large Language Models (LLMs). They undermine reliability and user trust by generating inaccurate or fabricated content. Recent studies suggest that when generating false statements, the internal states of LLMs encode information about truthfulness. However, these studies often rely on synthetic datasets that lack realism, which limits generalization when evaluating the factual accuracy of text generated by the model itself. In this paper, we challenge the findings of previous work by investigating truthfulness encoding capabilities, leading to the generation of a more realistic and challenging dataset. Specifically, we extend previous work by introducing: (1) a strategy for sampling plausible true-false factoid sentences from tabular data and (2) a procedure for generating realistic, LLM-dependent true-false datasets from Question Answering collections. Our analysis of two open-source LLMs reveals that while the findings from previous studies are partially validated, generalization to LLM-generated datasets remains challenging. This study lays the groundwork for future research on factuality in LLMs and offers practical guidelines for more effective evaluation. 

**Abstract (ZH)**: 大规模语言模型中的事实幻觉是一个主要挑战。它们通过生成不准确或虚构的内容削弱了可靠性和用户信任。近期研究表明，当生成虚假陈述时，大语言模型内部状态中包含了关于真实性的信息。然而，这些研究往往依赖于缺乏现实性的合成数据集，这限制了对其自身生成文本的事实准确性评估的泛化能力。本文通过探讨真实性编码能力，提出了一种更具现实性和挑战性的数据集生成方法。具体来说，本文扩展了先前的工作，引入了（1）从表格数据中采样可证实的真假事实句的策略，以及（2）从问答集合中生成真实的、依赖于大语言模型的真假数据集的程序。我们对两个开源大语言模型的分析表明，尽管先前研究的部分发现得到了验证，但将其推广到大语言模型生成的数据集仍然具有挑战性。本文为未来关于大语言模型事实性的研究奠定了基础，并提供了更有效的评估实践指南。 

---
# CUB: Benchmarking Context Utilisation Techniques for Language Models 

**Title (ZH)**: CUB：评估语言模型中上下文利用技术的标准测试集 

**Authors**: Lovisa Hagström, Youna Kim, Haeun Yu, Sang-goo Lee, Richard Johansson, Hyunsoo Cho, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.16518)  

**Abstract**: Incorporating external knowledge is crucial for knowledge-intensive tasks, such as question answering and fact checking. However, language models (LMs) may ignore relevant information that contradicts outdated parametric memory or be distracted by irrelevant contexts. While many context utilisation manipulation techniques (CMTs) that encourage or suppress context utilisation have recently been proposed to alleviate these issues, few have seen systematic comparison. In this paper, we develop CUB (Context Utilisation Benchmark) to help practitioners within retrieval-augmented generation (RAG) identify the best CMT for their needs. CUB allows for rigorous testing on three distinct context types, observed to capture key challenges in realistic context utilisation scenarios. With this benchmark, we evaluate seven state-of-the-art methods, representative of the main categories of CMTs, across three diverse datasets and tasks, applied to nine LMs. Our results show that most of the existing CMTs struggle to handle the full set of types of contexts that may be encountered in real-world retrieval-augmented scenarios. Moreover, we find that many CMTs display an inflated performance on simple synthesised datasets, compared to more realistic datasets with naturally occurring samples. Altogether, our results show the need for holistic tests of CMTs and the development of CMTs that can handle multiple context types. 

**Abstract (ZH)**: Incorporating 外部知识对于知识密集型任务（如问答和事实核查）至关重要。然而，语言模型（LMs）可能会忽略与其过时参数记忆相矛盾的相关信息，或被无关上下文所分散。虽然最近提出了许多鼓励或抑制上下文利用的技术（CMTs）来缓解这些问题，但很少有进行全面比较。在本文中，我们开发了CUB（上下文利用基准）以帮助检索增强生成（RAG）领域的实践者识别最适合他们需求的最佳CMT。CUB 允许在三种不同类型的上下文中进行严格的测试，这些类型被视为现实环境中上下文利用挑战的关键。借助此基准，我们评估了七种最先进的方法，这些方法代表了主要的CMT类别，在三个不同的数据集和任务上应用于九种LMs。我们的结果表明，现有的大多数CMT难以处理真实世界检索增强场景中可能出现的各种类型的上下文。此外，我们发现，许多CMT在简单合成数据集上的表现明显优于包含自然出现样本的更现实的数据集。总之，我们的结果强调了对CMT进行全面测试以及开发能够处理多种上下文类型的CMT的需求。 

---
# Edge-First Language Model Inference: Models, Metrics, and Tradeoffs 

**Title (ZH)**: 边缘优先语言模型推理：模型、度量标准与tradeoffs 

**Authors**: SiYoung Jang, Roberto Morabito  

**Link**: [PDF](https://arxiv.org/pdf/2505.16508)  

**Abstract**: The widespread adoption of Language Models (LMs) across industries is driving interest in deploying these services across the computing continuum, from the cloud to the network edge. This shift aims to reduce costs, lower latency, and improve reliability and privacy. Small Language Models (SLMs), enabled by advances in model compression, are central to this shift, offering a path to on-device inference on resource-constrained edge platforms. This work examines the interplay between edge and cloud deployments, starting from detailed benchmarking of SLM capabilities on single edge devices, and extending to distributed edge clusters. We identify scenarios where edge inference offers comparable performance with lower costs, and others where cloud fallback becomes essential due to limits in scalability or model capacity. Rather than proposing a one-size-fits-all solution, we present platform-level comparisons and design insights for building efficient, adaptive LM inference systems across heterogeneous environments. 

**Abstract (ZH)**: 语言模型在各行业的广泛应用推动了跨计算 continuum 部署服务的兴趣，从云端到网络边缘。这一转变旨在降低费用、减少延迟，并提高可靠性和隐私性。通过模型压缩实现的小语言模型（SLM）是这一转变的核心，它们为资源受限的边缘平台提供离线推理的可能性。本文探讨了边缘和云端部署之间的相互作用，从单个边缘设备上小语言模型能力的详细基准测试开始，扩展到分布式边缘集群。我们识别了边缘推理在某些情况下提供类似性能并具有更低成本的场景，同时也指出了在可扩展性或模型容量有限时依赖云端作为后备的必要性。我们不提供一刀切的解决方案，而是提供了跨异构环境构建高效、自适应语言模型推理系统的平台级比较和设计洞察。 

---
# Smaller, Smarter, Closer: The Edge of Collaborative Generative AI 

**Title (ZH)**: 更小，更智能，更近：协作生成AI的边缘 

**Authors**: Roberto Morabito, SiYoung Jang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16499)  

**Abstract**: The rapid adoption of generative AI (GenAI), particularly Large Language Models (LLMs), has exposed critical limitations of cloud-centric deployments, including latency, cost, and privacy concerns. Meanwhile, Small Language Models (SLMs) are emerging as viable alternatives for resource-constrained edge environments, though they often lack the capabilities of their larger counterparts. This article explores the potential of collaborative inference systems that leverage both edge and cloud resources to address these challenges. By presenting distinct cooperation strategies alongside practical design principles and experimental insights, we offer actionable guidance for deploying GenAI across the computing continuum. 

**Abstract (ZH)**: 生成式人工智能（GenAI）尤其是大型语言模型（LLMs）的快速 adoption 已暴露出基于云部署的关键限制，包括延迟、成本和隐私问题。同时，小型语言模型（SLMs）正成为资源受限边缘环境的可行替代方案，尽管它们往往缺乏大型模型的能力。本文探讨了利用边缘和云资源的协作推理系统在应对这些挑战方面的潜力。通过提出不同的合作策略并结合实用的设计原则和实验见解，我们为在计算连续体中部署GenAI提供了可操作的指导。 

---
# LLaMAs Have Feelings Too: Unveiling Sentiment and Emotion Representations in LLaMA Models Through Probing 

**Title (ZH)**: LLaMAs 也有情感：通过探针揭示LLaMA模型中的情感和情绪表示 

**Authors**: Dario Di Palma, Alessandro De Bellis, Giovanni Servedio, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16491)  

**Abstract**: Large Language Models (LLMs) have rapidly become central to NLP, demonstrating their ability to adapt to various tasks through prompting techniques, including sentiment analysis. However, we still have a limited understanding of how these models capture sentiment-related information. This study probes the hidden layers of Llama models to pinpoint where sentiment features are most represented and to assess how this affects sentiment analysis.
Using probe classifiers, we analyze sentiment encoding across layers and scales, identifying the layers and pooling methods that best capture sentiment signals. Our results show that sentiment information is most concentrated in mid-layers for binary polarity tasks, with detection accuracy increasing up to 14% over prompting techniques. Additionally, we find that in decoder-only models, the last token is not consistently the most informative for sentiment encoding. Finally, this approach enables sentiment tasks to be performed with memory requirements reduced by an average of 57%.
These insights contribute to a broader understanding of sentiment in LLMs, suggesting layer-specific probing as an effective approach for sentiment tasks beyond prompting, with potential to enhance model utility and reduce memory requirements. 

**Abstract (ZH)**: 大型语言模型（LLMs）已迅速成为自然语言处理（NLP）的核心，通过提示技术展示了其适应各种任务的能力，包括情感分析。然而，我们仍对这些模型如何捕捉情感相关信息缺乏充分的理解。本研究探讨了Llama模型的隐藏层，以确定情感特征最集中的位置，并评估这一过程对情感分析的影响。

通过探针分类器，我们分析了不同层和尺度的情感编码，识别出最适合捕捉情感信号的层和池化方法。结果显示，对于二元极性任务，情感信息主要集中在中间层，使用探针分类器的情感检测准确率最高可提高14%。此外，我们发现，在解码器-only模型中，最后一个标记并不总是最信息丰富的标记。最后，这种方法使得情感任务的内存需求平均减少57%。

这些见解有助于更全面地理解LLMs中的情感，表明针对情感任务的层特定探针是一种有效的方法，不仅可以超越提示技术的应用，而且还具有提高模型实用性和减少内存需求的潜力。 

---
# Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning 

**Title (ZH)**: 通过合成任务和强化学习教学大型语言模型保持上下文忠诚度 

**Authors**: Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Bofei Gao, Kangyang Luo, Wenhao Li, Yufei Huang, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16483)  

**Abstract**: Teaching large language models (LLMs) to be faithful in the provided context is crucial for building reliable information-seeking systems. Therefore, we propose a systematic framework, CANOE, to improve the faithfulness of LLMs in both short-form and long-form generation tasks without human annotations. Specifically, we first synthesize short-form question-answering (QA) data with four diverse tasks to construct high-quality and easily verifiable training data without human annotation. Also, we propose Dual-GRPO, a rule-based reinforcement learning method that includes three tailored rule-based rewards derived from synthesized short-form QA data, while simultaneously optimizing both short-form and long-form response generation. Notably, Dual-GRPO eliminates the need to manually label preference data to train reward models and avoids over-optimizing short-form generation when relying only on the synthesized short-form QA data. Experimental results show that CANOE greatly improves the faithfulness of LLMs across 11 different downstream tasks, even outperforming the most advanced LLMs, e.g., GPT-4o and OpenAI o1. 

**Abstract (ZH)**: Teaching大型语言模型（LLMs）在提供上下文中的诚信对于构建可靠的信息查询系统至关重要。因此，我们提出了一种系统性框架CANOE，以在短文本和长文本生成任务中不依赖人类注解提高LLMs的诚信度。具体而言，我们首先通过四种不同的任务综合生成短文本问答（QA）数据，以构建无需人工标注的高质量且易于验证的训练数据。同时，我们提出了基于规则的增强学习方法Dual-GRPO，该方法包括三个针对合成短文本QA数据定制的规则奖励，同时优化短文本和长文本响应生成。值得注意的是，Dual-GRPO消除了需要手动标注偏好数据来训练奖励模型的需求，并避免了仅依赖合成短文本QA数据时过度优化短文本生成。实验结果表明，CANOE显著提高了LLMs在11个不同下游任务中的诚信度，甚至超越了最先进的LLMs，例如GPT-4o和OpenAI o1。 

---
# Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation 

**Title (ZH)**: 基于Jensen-Shannon散度的检索增强生成中上下文归因机理研究 

**Authors**: Ruizhe Li, Chen Chen, Yuchen Hu, Yanjun Gao, Xi Wang, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2505.16415)  

**Abstract**: Retrieval-Augmented Generation (RAG) leverages large language models (LLMs) combined with external contexts to enhance the accuracy and reliability of generated responses. However, reliably attributing generated content to specific context segments, context attribution, remains challenging due to the computationally intensive nature of current methods, which often require extensive fine-tuning or human annotation. In this work, we introduce a novel Jensen-Shannon Divergence driven method to Attribute Response to Context (ARC-JSD), enabling efficient and accurate identification of essential context sentences without additional fine-tuning or surrogate modelling. Evaluations on a wide range of RAG benchmarks, such as TyDi QA, Hotpot QA, and Musique, using instruction-tuned LLMs in different scales demonstrate superior accuracy and significant computational efficiency improvements compared to the previous surrogate-based method. Furthermore, our mechanistic analysis reveals specific attention heads and multilayer perceptron (MLP) layers responsible for context attribution, providing valuable insights into the internal workings of RAG models. 

**Abstract (ZH)**: 基于检索增强生成的响应与上下文归因（ARC-JSD）方法 

---
# Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning 

**Title (ZH)**: Tool-Star: 通过强化学习赋能兼具LLM大脑的多工具推理器 

**Authors**: Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16410)  

**Abstract**: Recently, large language models (LLMs) have shown remarkable reasoning capabilities via large-scale reinforcement learning (RL). However, leveraging the RL algorithm to empower effective multi-tool collaborative reasoning in LLMs remains an open challenge. In this paper, we introduce Tool-Star, an RL-based framework designed to empower LLMs to autonomously invoke multiple external tools during stepwise reasoning. Tool-Star integrates six types of tools and incorporates systematic designs in both data synthesis and training. To address the scarcity of tool-use data, we propose a general tool-integrated reasoning data synthesis pipeline, which combines tool-integrated prompting with hint-based sampling to automatically and scalably generate tool-use trajectories. A subsequent quality normalization and difficulty-aware classification process filters out low-quality samples and organizes the dataset from easy to hard. Furthermore, we propose a two-stage training framework to enhance multi-tool collaborative reasoning by: (1) cold-start fine-tuning, which guides LLMs to explore reasoning patterns via tool-invocation feedback; and (2) a multi-tool self-critic RL algorithm with hierarchical reward design, which reinforces reward understanding and promotes effective tool collaboration. Experimental analyses on over 10 challenging reasoning benchmarks highlight the effectiveness and efficiency of Tool-Star. The code is available at this https URL. 

**Abstract (ZH)**: 最近，大型语言模型（LLMs）通过大规模强化学习（RL）展示了卓越的推理能力。然而，利用RL算法增强LLMs的自主多工具协同推理能力仍是一个开放的挑战。本文介绍了一种基于RL的框架Tool-Star，旨在使LLMs在逐步推理过程中自主调用多个外部工具。Tool-Star集成了六类工具，并在数据合成和训练中采用了系统性设计。为了解决工具使用数据稀缺的问题，我们提出了一种通用的工具集成推理数据合成管道，该管道结合了工具集成提示和基于提示的采样，以自动生成工具使用轨迹。随后的质量规范化和难度感知分类过程过滤掉低质量样本，将数据集按难度排序。此外，我们提出了一种两阶段训练框架，通过以下方式增强多工具协同推理能力：（1）冷启动微调，通过工具调用反馈引导LLMs探索推理模式；（2）具有层次化奖励设计的多工具自批判RL算法，增强奖励理解并促进有效的工具协作。对超过10个具有挑战性的推理基准的实验分析突显了Tool-Star的有效性和效率。代码已发布于此：https://this-url。 

---
# Resource for Error Analysis in Text Simplification: New Taxonomy and Test Collection 

**Title (ZH)**: 文本简化中的错误分析资源：新分类与测试集 

**Authors**: Benjamin Vendeville, Liana Ermakova, Pierre De Loor  

**Link**: [PDF](https://arxiv.org/pdf/2505.16392)  

**Abstract**: The general public often encounters complex texts but does not have the time or expertise to fully understand them, leading to the spread of misinformation. Automatic Text Simplification (ATS) helps make information more accessible, but its evaluation methods have not kept up with advances in text generation, especially with Large Language Models (LLMs). In particular, recent studies have shown that current ATS metrics do not correlate with the presence of errors. Manual inspections have further revealed a variety of errors, underscoring the need for a more nuanced evaluation framework, which is currently lacking. This resource paper addresses this gap by introducing a test collection for detecting and classifying errors in simplified texts. First, we propose a taxonomy of errors, with a formal focus on information distortion. Next, we introduce a parallel dataset of automatically simplified scientific texts. This dataset has been human-annotated with labels based on our proposed taxonomy. Finally, we analyze the quality of the dataset, and we study the performance of existing models to detect and classify errors from that taxonomy. These contributions give researchers the tools to better evaluate errors in ATS, develop more reliable models, and ultimately improve the quality of automatically simplified texts. 

**Abstract (ZH)**: 公共读者经常遇到复杂的文本，但没有足够的时间或专业知识来完全理解这些文本，导致错误信息的传播。自动文本简化（ATS）有助于使信息更加易于获取，但其评估方法尚未跟上文本生成技术的进步，特别是大型语言模型（LLMs）的进步。近期研究显示，当前的ATS指标与错误的出现无关。进一步的手动检查揭示了各种错误，突显了需要一种更加复杂的评估框架的需求，而这一需求目前尚未得到满足。本文献综述通过引入一个检测和分类简化文本中错误的数据集来弥补这一缺口。首先，我们提出了一种错误分类学，重点关注信息失真。其次，我们介绍了自动简化科学文本的平行数据集。该数据集基于我们提出的分类学进行了人工注释。最后，我们分析了数据集的质量，并研究了现有模型检测和分类分类学中错误的性能。这些贡献为研究人员提供了工具，以便更好地评估ATS中的错误，开发更可靠的技术，并最终提高自动简化文本的质量。 

---
# SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning 

**Title (ZH)**: SATURN: 基于SAT的强化学习以释放语言模型推理能力 

**Authors**: Huanyu Liu, Jia Li, Hao Zhu, Kechi Zhang, Yihong Dong, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16368)  

**Abstract**: How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) suffer from three key limitations: (1) Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. (2) Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3) Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it hard to train LLMs to develop reasoning ability from easy to hard.
To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLM reasoning. Saturn enables scalable task construction, rule-based verification, and precise difficulty control. Saturn designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions.
We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, Saturn achieves further improvements of +8.8%. We release the source code, data, and models to support future research. 

**Abstract (ZH)**: 如何设计有效的强化学习任务以充分利用大型语言模型的推理能力仍然是一个开放问题。现有强化学习任务（如数学、编程和构建推理任务）存在三个关键局限性：可扩展性、验证性和可控难度。卫星：基于SAT的强化学习框架 

---
# AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training 

**Title (ZH)**: AdamS: 动量本身可以成为大规模语言模型预训练和后训练的规范化方法 

**Authors**: Huishuai Zhang, Bohan Wang, Luoxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16363)  

**Abstract**: We introduce AdamS, a simple yet effective alternative to Adam for large language model (LLM) pretraining and post-training. By leveraging a novel denominator, i.e., the root of weighted sum of squares of the momentum and the current gradient, AdamS eliminates the need for second-moment estimates. Hence, AdamS is efficient, matching the memory and compute footprint of SGD with momentum while delivering superior optimization performance. Moreover, AdamS is easy to adopt: it can directly inherit hyperparameters of AdamW, and is entirely model-agnostic, integrating seamlessly into existing pipelines without modifications to optimizer APIs or architectures. The motivation behind AdamS stems from the observed $(L_0, L_1)$ smoothness properties in transformer objectives, where local smoothness is governed by gradient magnitudes that can be further approximated by momentum magnitudes. We establish rigorous theoretical convergence guarantees and provide practical guidelines for hyperparameter selection. Empirically, AdamS demonstrates strong performance in various tasks, including pre-training runs on GPT-2 and Llama2 (up to 13B parameters) and reinforcement learning in post-training regimes. With its efficiency, simplicity, and theoretical grounding, AdamS stands as a compelling alternative to existing optimizers. 

**Abstract (ZH)**: AdamS: 一种用于大规模语言模型预训练和后训练的有效替代Adam的方法 

---
# AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners 

**Title (ZH)**: AdaSTaR: 自适应数据采样训练自教会推理器 

**Authors**: Woosung Koh, Wonbeen Oh, Jaein Jang, MinHyung Lee, Hyeongjin Kim, Ah Yeon Kim, Joonkee Kim, Junghyun Lee, Taehyeon Kim, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16322)  

**Abstract**: Self-Taught Reasoners (STaR), synonymously known as Rejection sampling Fine-Tuning (RFT), is an integral part of the training pipeline of self-improving reasoning Language Models (LMs). The self-improving mechanism often employs random observation (data) sampling. However, this results in trained observation imbalance; inefficiently over-training on solved examples while under-training on challenging ones. In response, we introduce Adaptive STaR (AdaSTaR), a novel algorithm that rectifies this by integrating two adaptive sampling principles: (1) Adaptive Sampling for Diversity: promoting balanced training across observations, and (2) Adaptive Sampling for Curriculum: dynamically adjusting data difficulty to match the model's evolving strength. Across six benchmarks, AdaSTaR achieves best test accuracy in all instances (6/6) and reduces training FLOPs by an average of 58.6% against an extensive list of baselines. These improvements in performance and efficiency generalize to different pre-trained LMs and larger models, paving the way for more efficient and effective self-improving LMs. 

**Abstract (ZH)**: 自学习推理器（STaR），又称拒绝采样微调（RFT），是自我改进推理语言模型（LMs）训练管道中的一个关键组成部分。为了应对训练过程中出现的观察数据不平衡问题，即过度训练于已解决问题而忽视具有挑战性的问题，我们提出了自适应STaR（AdaSTaR）算法，该算法结合了两种自适应采样原则：（1）多样性自适应采样：促进观察数据的均衡训练，（2） Curriculum自适应采样：动态调整数据难度以匹配模型的发展强度。在六个基准上，AdaSTaR在所有情况下均实现了最佳测试准确率，并且与一系列基线相比，平均减少了58.6%的训练FLOPs。这些在性能和效率上的改进适用于不同预训练的LMs和更大规模的模型，为更高效和有效的自我改进LMs铺平了道路。 

---
# PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models 

**Title (ZH)**: PMPO：概率度量提示优化方法研究（适用于小型和大型语言模型） 

**Authors**: Chenzhuo Zhao, Ziqian Liu, Xingda Wang, Junting Lu, Chaoyi Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16307)  

**Abstract**: Prompt optimization offers a practical and broadly applicable alternative to fine-tuning for improving large language model (LLM) performance. However, existing methods often rely on costly output generation, self-critiquing abilities, or human-annotated preferences, which limit their scalability, especially for smaller or non-instruction-tuned models. We introduce PMPO (Probabilistic Metric Prompt Optimization), a unified framework that refines prompts using token-level cross-entropy loss as a direct, lightweight evaluation signal. PMPO identifies low-quality prompt segments by masking and measuring their impact on loss, then rewrites and selects improved variants by minimizing loss over positive and negative examples. Unlike prior methods, it requires no output sampling or human evaluation during optimization, relying only on forward passes and log-likelihoods. PMPO supports both supervised and preference-based tasks through a closely aligned loss-based evaluation strategy. Experiments show that PMPO consistently outperforms prior methods across model sizes and tasks: it achieves the highest average accuracy on BBH, performs strongly on GSM8K and AQUA-RAT, and improves AlpacaEval 2.0 win rates by over 19 points. These results highlight PMPO's effectiveness, efficiency, and broad applicability. 

**Abstract (ZH)**: 概率度量提示优化为提高大型语言模型性能提供了一种实用且广泛适用的替代方法，无需微调。然而，现有方法通常依赖于昂贵的输出生成、自我批判能力或人工标注的偏好，这限制了它们的扩展性，尤其是在对于较小或未指令微调的模型中。我们引入了PMPO（概率度量提示优化）统一框架，该框架使用标记级交叉熵损失作为直接、轻量级的评价信号来精炼提示。PMPO通过遮罩和测量其对损失的影响来识别低质量提示段落，然后通过最小化损失来重写和选择改进的版本。与先有方法不同，它在优化过程中不需要采样输出或人工评估，仅依赖前向传递和对数似然性。PMPO通过紧密对齐的基于损失的评价策略支持监督和偏好任务。实验结果表明，PMPO在不同模型规模和任务中均优于先前方法：在BBH上实现最高平均准确率，在GSM8K和AQUA-RAT上表现强劲，并将AlpacaEval 2.0的胜率提高了超过19个百分点。这些结果突显了PMPO的有效性、效率和广泛的适用性。 

---
# Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning 

**Title (ZH)**: Transformer Copilot: 从错误日志中学习在大型语言模型 fine-tuning 中 

**Authors**: Jiaru Zou, Yikun Ban, Zihao Li, Yunzhe Qi, Ruizhong Qiu, Ling Yang, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16270)  

**Abstract**: Large language models are typically adapted to downstream tasks through supervised fine-tuning on domain-specific data. While standard fine-tuning focuses on minimizing generation loss to optimize model parameters, we take a deeper step by retaining and leveraging the model's own learning signals, analogous to how human learners reflect on past mistakes to improve future performance. We first introduce the concept of Mistake Log to systematically track the model's learning behavior and recurring errors throughout fine-tuning. Treating the original transformer-based model as the Pilot, we correspondingly design a Copilot model to refine the Pilot's inference performance via logits rectification. We name the overall Pilot-Copilot framework the Transformer Copilot, which introduces (i) a novel Copilot model design, (ii) a joint training paradigm where the Copilot continuously learns from the evolving Mistake Log alongside the Pilot, and (iii) a fused inference paradigm where the Copilot rectifies the Pilot's logits for enhanced generation. We provide both theoretical and empirical analyses on our new learning framework. Experiments on 12 benchmarks spanning commonsense, arithmetic, and recommendation tasks demonstrate that Transformer Copilot consistently improves performance by up to 34.5%, while introducing marginal computational overhead to Pilot models and exhibiting strong scalability and transferability. 

**Abstract (ZH)**: 大型语言模型通常通过领域特定数据的监督微调来适应下游任务。而标准的微调侧重于最小化生成损失以优化模型参数，我们则更进一步，通过保留并利用模型自身的学习信号来改进模型，类似于人类学习者通过反思过去的错误来提升未来的表现。我们首先引入了“错误日志”（Mistake Log）的概念，系统地追踪微调过程中模型的学习行为及其反复出现的错误。我们将原始的基于Transformer的模型视为“副驾”（Pilot），相应地设计了一个“副驾”模型，通过修正“副驾”的概率输出来提升“副驾”的推理性能。我们将整体的“副驾-副驾”框架命名为Transformer Copilot，该框架引入了(i)一种新型的“副驾”模型设计，(ii)一种联合训练范式，其中“副驾”不断从不断演化的错误日志中学习，并与“副驾”同步进行，以及(iii)一种融合推理范式，其中“副驾”修正“副驾”的概率输出以增强生成性能。我们对新的学习框架进行了理论和实验分析。在涵盖常识、算术和推荐等12个基准任务的实验中，Transformer Copilot 一致地提高了34.5%的性能，同时对“副驾”模型的计算开销仅增加了轻微的影响，并且展示了强大的扩展性和迁移性。 

---
# LIFEBench: Evaluating Length Instruction Following in Large Language Models 

**Title (ZH)**: LIFEBench: 评估大型语言模型的长度指令跟随能力 

**Authors**: Wei Zhang, Zhenhong Zhou, Junfeng Fang, Rongwu Xu, Kun Wang, Yuanhe Zhang, Rui Wang, Ge Zhang, Xinfeng Li, Li Sun, Lingjuan Lyu, Yang Liu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.16234)  

**Abstract**: While large language models (LLMs) can solve PhD-level reasoning problems over long context inputs, they still struggle with a seemingly simpler task: following explicit length instructions-e.g., write a 10,000-word novel. Additionally, models often generate far too short outputs, terminate prematurely, or even refuse the request. Existing benchmarks focus primarily on evaluating generations quality, but often overlook whether the generations meet length constraints. To this end, we introduce Length Instruction Following Evaluation Benchmark (LIFEBench) to comprehensively evaluate LLMs' ability to follow length instructions across diverse tasks and a wide range of specified lengths. LIFEBench consists of 10,800 instances across 4 task categories in both English and Chinese, covering length constraints ranging from 16 to 8192 words. We evaluate 26 widely-used LLMs and find that most models reasonably follow short-length instructions but deteriorate sharply beyond a certain threshold. Surprisingly, almost all models fail to reach the vendor-claimed maximum output lengths in practice, as further confirmed by our evaluations extending up to 32K words. Even long-context LLMs, despite their extended input-output windows, counterintuitively fail to improve length-instructions following. Notably, Reasoning LLMs outperform even specialized long-text generation models, achieving state-of-the-art length following. Overall, LIFEBench uncovers fundamental limitations in current LLMs' length instructions following ability, offering critical insights for future progress. 

**Abstract (ZH)**: 长指令遵循评估基准（LIFEBench） 

---
# AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models 

**Title (ZH)**: AudioTrust: 评估音频大规模语言模型的多维度可信性 

**Authors**: Kai Li, Can Shen, Yile Liu, Jirui Han, Kelong Zheng, Xuechao Zou, Zhe Wang, Xingjian Du, Shun Zhang, Hanjun Luo, Yingbin Jin, Xinxin Xing, Ziyang Ma, Yue Liu, Xiaojun Jia, Yifan Zhang, Junfeng Fang, Kun Wang, Yibo Yan, Haoyang Li, Yiming Li, Xiaobin Zhuang, Yang Liu, Haibo Hu, Zhuo Chen, Zhizheng Wu, Xiaolin Hu, Eng-Siong Chng, XiaoFeng Wang, Wenyuan Xu, Wei Dong, Xinfeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16211)  

**Abstract**: The rapid advancement and expanding applications of Audio Large Language Models (ALLMs) demand a rigorous understanding of their trustworthiness. However, systematic research on evaluating these models, particularly concerning risks unique to the audio modality, remains largely unexplored. Existing evaluation frameworks primarily focus on the text modality or address only a restricted set of safety dimensions, failing to adequately account for the unique characteristics and application scenarios inherent to the audio modality. We introduce AudioTrust-the first multifaceted trustworthiness evaluation framework and benchmark specifically designed for ALLMs. AudioTrust facilitates assessments across six key dimensions: fairness, hallucination, safety, privacy, robustness, and authentication. To comprehensively evaluate these dimensions, AudioTrust is structured around 18 distinct experimental setups. Its core is a meticulously constructed dataset of over 4,420 audio/text samples, drawn from real-world scenarios (e.g., daily conversations, emergency calls, voice assistant interactions), specifically designed to probe the multifaceted trustworthiness of ALLMs. For assessment, the benchmark carefully designs 9 audio-specific evaluation metrics, and we employ a large-scale automated pipeline for objective and scalable scoring of model outputs. Experimental results reveal the trustworthiness boundaries and limitations of current state-of-the-art open-source and closed-source ALLMs when confronted with various high-risk audio scenarios, offering valuable insights for the secure and trustworthy deployment of future audio models. Our platform and benchmark are available at this https URL. 

**Abstract (ZH)**: AudioTrust：面向音频大语言模型的多维度可信性评估框架与基准 

---
# NQKV: A KV Cache Quantization Scheme Based on Normal Distribution Characteristics 

**Title (ZH)**: NQKV：基于正态分布特性的一种键值缓存量化方案 

**Authors**: Zhihang Cai, Xingjun Zhang, Zhendong Tan, Zheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16210)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency across a wide range of tasks. However, LLMs often require larger batch sizes to enhance throughput or longer context lengths to meet task demands, which significantly increases the memory resource consumption of the Key-Value (KV) cache during inference, becoming a major bottleneck in LLM deployment. To address this issue, quantization is a common and straightforward approach. Currently, quantization methods for activations are limited to 8-bit, and quantization to even lower bits can lead to substantial accuracy drops. To further save space by quantizing the KV cache to even lower bits, we analyzed the element distribution of the KV cache and designed the NQKV algorithm. Since the elements within each block of the KV cache follow a normal distribution, NQKV employs per-block quantile quantization to achieve information-theoretically optimal quantization error. Without significantly compromising model output quality, NQKV enables the OPT model to perform inference with an 2x larger batch size or a 4x longer context length, and it improves throughput by 9.3x compared to when the KV cache is not used. 

**Abstract (ZH)**: 大型语言模型通过量化技术实现KV缓存高效 inference 

---
# Automated Feedback Loops to Protect Text Simplification with Generative AI from Information Loss 

**Title (ZH)**: 自动反馈循环以保护基于生成式AI的文字简化不致信息损失 

**Authors**: Abhay Kumara Sri Krishna Nandiraju, Gondy Leroy, David Kauchak, Arif Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2505.16172)  

**Abstract**: Understanding health information is essential in achieving and maintaining a healthy life. We focus on simplifying health information for better understanding. With the availability of generative AI, the simplification process has become efficient and of reasonable quality, however, the algorithms remove information that may be crucial for comprehension. In this study, we compare generative AI to detect missing information in simplified text, evaluate its importance, and fix the text with the missing information. We collected 50 health information texts and simplified them using gpt-4-0613. We compare five approaches to identify missing elements and regenerate the text by inserting the missing elements. These five approaches involve adding missing entities and missing words in various ways: 1) adding all the missing entities, 2) adding all missing words, 3) adding the top-3 entities ranked by gpt-4-0613, and 4, 5) serving as controls for comparison, adding randomly chosen entities. We use cosine similarity and ROUGE scores to evaluate the semantic similarity and content overlap between the original, simplified, and reconstructed simplified text. We do this for both summaries and full text. Overall, we find that adding missing entities improves the text. Adding all the missing entities resulted in better text regeneration, which was better than adding the top-ranked entities or words, or random words. Current tools can identify these entities, but are not valuable in ranking them. 

**Abstract (ZH)**: 理解健康信息对于实现和维持健康生活至关重要。本研究重点在于简化健康信息以提高理解效果。借助生成式AI，简化过程变得高效且质量合理，然而算法可能会移除对于理解至关重要的信息。本研究旨在比较生成式AI检测简化文本中缺失信息的能力，并评估这些信息的重要性，进而通过插入缺失信息修复文本。我们收集了50篇健康信息文本，并使用gpt-4-0613进行简化。我们比较了五种方法以识别缺失元素并通过对缺失元素进行再生来修复文本。这五种方法包括以不同方式添加缺失实体和缺失词汇：1) 添加所有缺失实体，2) 添加所有缺失词汇，3) 添加由gpt-4-0613排名前三的实体，以及4)、5) 作为对照，添加随机选择的实体。我们使用余弦相似度和ROUGE分数来评估原始文本、简化文本和重构简化文本之间的语义相似性和内容重叠。这包括摘要和完整文本。总体而言，我们发现添加缺失实体能够提高文本的质量。添加所有缺失实体的效果优于添加排名靠前的实体或随机词汇。当前工具可以识别这些实体，但在排序方面尚不具备价值。 

---
# When VLMs Meet Image Classification: Test Sets Renovation via Missing Label Identification 

**Title (ZH)**: 当大模型遇到图像分类：通过缺失标签识别更新测试集 

**Authors**: Zirui Pang, Haosheng Tan, Yuhan Pu, Zhijie Deng, Zhouan Shen, Keyu Hu, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16149)  

**Abstract**: Image classification benchmark datasets such as CIFAR, MNIST, and ImageNet serve as critical tools for model evaluation. However, despite the cleaning efforts, these datasets still suffer from pervasive noisy labels and often contain missing labels due to the co-existing image pattern where multiple classes appear in an image sample. This results in misleading model comparisons and unfair evaluations. Existing label cleaning methods focus primarily on noisy labels, but the issue of missing labels remains largely overlooked. Motivated by these challenges, we present a comprehensive framework named REVEAL, integrating state-of-the-art pre-trained vision-language models (e.g., LLaVA, BLIP, Janus, Qwen) with advanced machine/human label curation methods (e.g., Docta, Cleanlab, MTurk), to systematically address both noisy labels and missing label detection in widely-used image classification test sets. REVEAL detects potential noisy labels and omissions, aggregates predictions from various methods, and refines label accuracy through confidence-informed predictions and consensus-based filtering. Additionally, we provide a thorough analysis of state-of-the-art vision-language models and pre-trained image classifiers, highlighting their strengths and limitations within the context of dataset renovation by revealing 10 observations. Our method effectively reveals missing labels from public datasets and provides soft-labeled results with likelihoods. Through human verifications, REVEAL significantly improves the quality of 6 benchmark test sets, highly aligning to human judgments and enabling more accurate and meaningful comparisons in image classification. 

**Abstract (ZH)**: 基于图像分类基准数据集（如CIFAR、MNIST和ImageNet）的清理框架：同时应对噪声标签与缺失标签 

---
# Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation 

**Title (ZH)**: 通过稀疏自编码器引导LVLMs以减轻幻觉现象 

**Authors**: Zhenglin Hua, Jinghan He, Zijun Yao, Tianxu Han, Haiyun Guo, Yuheng Jia, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16146)  

**Abstract**: Large vision-language models (LVLMs) have achieved remarkable performance on multimodal tasks such as visual question answering (VQA) and image captioning. However, they still suffer from hallucinations, generating text inconsistent with visual input, posing significant risks in real-world applications. Existing approaches to address this issue focus on incorporating external knowledge bases, alignment training, or decoding strategies, all of which require substantial computational cost and time. Recent works try to explore more efficient alternatives by adjusting LVLMs' internal representations. Although promising, these methods may cause hallucinations to be insufficiently suppressed or lead to excessive interventions that negatively affect normal semantics. In this work, we leverage sparse autoencoders (SAEs) to identify semantic directions closely associated with either hallucinations or actuality, realizing more precise and direct hallucination-related representations. Our analysis demonstrates that interventions along the faithful direction we identified can mitigate hallucinations, while those along the hallucinatory direction can exacerbate them. Building on these insights, we propose Steering LVLMs via SAE Latent Directions (SSL), a training-free method based on SAE-derived latent directions to mitigate hallucinations in LVLMs. Extensive experiments demonstrate that SSL significantly outperforms existing decoding approaches in mitigating hallucinations, while maintaining transferability across different model architectures with negligible additional time overhead. 

**Abstract (ZH)**: 利用稀疏自编码器引导LVLMs减轻幻觉（Steering LVLMs via SAE Latent Directions (SSL) for Mitigating Hallucinations） 

---
# Merge to Mix: Mixing Datasets via Model Merging 

**Title (ZH)**: Merge to Mix: 通过模型合并进行数据集混合 

**Authors**: Zhixu Silvia Tao, Kasper Vinken, Hao-Wei Yeh, Avi Cooper, Xavier Boix  

**Link**: [PDF](https://arxiv.org/pdf/2505.16066)  

**Abstract**: Mixing datasets for fine-tuning large models (LMs) has become critical for maximizing performance on downstream tasks. However, composing effective dataset mixtures typically relies on heuristics and trial-and-error, often requiring multiple fine-tuning runs to achieve the desired outcome. We propose a novel method, $\textit{Merge to Mix}$, that accelerates composing dataset mixtures through model merging. Model merging is a recent technique that combines the abilities of multiple individually fine-tuned LMs into a single LM by using a few simple arithmetic operations. Our key insight is that merging models individually fine-tuned on each dataset in a mixture can effectively serve as a surrogate for a model fine-tuned on the entire mixture. Merge to Mix leverages this insight to accelerate selecting dataset mixtures without requiring full fine-tuning on each candidate mixture. Our experiments demonstrate that Merge to Mix surpasses state-of-the-art methods in dataset selection for fine-tuning LMs. 

**Abstract (ZH)**: 混合数据集以加速大规模模型微调已成为最大化下游任务性能的关键。然而，有效组合数据集混合体通常依赖于启发式方法和尝试错误，常常需要多次微调才能达到预期效果。我们提出了一种新颖的方法——Merge to Mix，通过模型合并加速数据集混合体的组合。模型合并是一种近期的技术，通过少数简单的算术操作将多个独立微调的大型语言模型（LMs）的能力整合到一个模型中。我们的核心洞察是，将混合体中的每个数据集独立微调的模型合并可以有效替代在一个混合体上进行整体微调的模型。Merge to Mix 利用这一洞察加速数据集混合体的选择，而无需对每个候选混合体进行完整的微调。我们的实验表明，Merge to Mix 在大型语言模型微调的数据集选择方面超越了现有最先进方法。 

---
# Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models 

**Title (ZH)**: 并不是所有模型都适用于专家卸载：关于混合专家模型的局部路由一致性研究 

**Authors**: Jingcong Liang, Siyuan Wang, Miren Tian, Yitong Li, Duyu Tang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16056)  

**Abstract**: Mixture-of-Experts (MoE) enables efficient scaling of large language models (LLMs) with sparsely activated experts during inference. To effectively deploy large MoE models on memory-constrained devices, many systems introduce *expert offloading* that caches a subset of experts in fast memory, leaving others on slow memory to run on CPU or load on demand. While some research has exploited the locality of expert activations, where consecutive tokens activate similar experts, the degree of this **local routing consistency** varies across models and remains understudied. In this paper, we propose two metrics to measure local routing consistency of MoE models: (1) **Segment Routing Best Performance (SRP)**, which evaluates how well a fixed group of experts can cover the needs of a segment of tokens, and (2) **Segment Cache Best Hit Rate (SCH)**, which measures the optimal segment-level cache hit rate under a given cache size limit. We analyzed 20 MoE LLMs with diverse sizes and architectures and found that models that apply MoE on every layer and do not use shared experts exhibit the highest local routing consistency. We further showed that domain-specialized experts contribute more to routing consistency than vocabulary-specialized ones, and that most models can balance between cache effectiveness and efficiency with cache sizes approximately 2x the active experts. These findings pave the way for memory-efficient MoE design and deployment without compromising inference speed. We publish the code for replicating experiments at this https URL . 

**Abstract (ZH)**: Mixture-of-Experts (MoE)在推理时稀疏激活专家，使大规模语言模型（LLMs）的高效扩展成为可能。为了有效在内存受限设备上部署大规模MoE模型，许多系统引入了“专家卸载”机制，将部分专家缓存在快速内存中，其余专家置于慢速内存上运行或按需加载。虽然一些研究利用了专家激活的局部性，即连续token激活相似的专家，但这种“局部路由一致性”的程度在不同模型之间变化较大且尚未得到充分研究。在本文中，我们提出了两个度量MoE模型局部路由一致性的指标：（1）段落路由最佳性能（SRP），评估固定专家组在一个token段上的表现；（2）段落缓存最佳命中率（SCH），衡量在给定缓存大小限制下的最优段落级缓存命中率。我们分析了20个不同规模和架构的MoE LLM，并发现不使用共享专家且在每一层应用MoE的模型展现出最高的局部路由一致性。进一步研究表明，领域特化的专家对路由一致性贡献更多，而非词汇特化的专家。大多数模型可以在缓存大小约为活跃专家数量2倍的情况下，平衡缓存有效性和效率。这些发现为在不影响推理速度的情况下进行内存高效MoE设计与部署铺平了道路。我们将在该网址发布实验复现代码：this https URL。 

---
# NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning 

**Title (ZH)**: NOVER：基于验证者无 involving 验证者的强化学习促使语言模型激励训练 

**Authors**: Wei Liu, Siya Qi, Xinyu Wang, Chen Qian, Yali Du, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16022)  

**Abstract**: Recent advances such as DeepSeek R1-Zero highlight the effectiveness of incentive training, a reinforcement learning paradigm that computes rewards solely based on the final answer part of a language model's output, thereby encouraging the generation of intermediate reasoning steps. However, these methods fundamentally rely on external verifiers, which limits their applicability to domains like mathematics and coding where such verifiers are readily available. Although reward models can serve as verifiers, they require high-quality annotated data and are costly to train. In this work, we propose NOVER, NO-VERifier Reinforcement Learning, a general reinforcement learning framework that requires only standard supervised fine-tuning data with no need for an external verifier. NOVER enables incentive training across a wide range of text-to-text tasks and outperforms the model of the same size distilled from large reasoning models such as DeepSeek R1 671B by 7.7 percent. Moreover, the flexibility of NOVER enables new possibilities for optimizing large language models, such as inverse incentive training. 

**Abstract (ZH)**: Recent Advances Such as DeepSeek R1-Zero Highlight the Effectiveness of Incentive Training in Reinforcement Learning, but External Verifiers Limit Their Applicability to Specific Domains: NOVER, NO-VERifier Reinforcement Learning, Enables Incentive Training Across Text-to-Text Tasks Without External Verifiers 

---
# Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations 

**Title (ZH)**: 稀疏自编码器中的可解释性幻觉：概念表示的稳健性评估 

**Authors**: Aaron J. Li, Suraj Srinivas, Usha Bhalla, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.16004)  

**Abstract**: Sparse autoencoders (SAEs) are commonly used to interpret the internal activations of large language models (LLMs) by mapping them to human-interpretable concept representations. While existing evaluations of SAEs focus on metrics such as the reconstruction-sparsity tradeoff, human (auto-)interpretability, and feature disentanglement, they overlook a critical aspect: the robustness of concept representations to input perturbations. We argue that robustness must be a fundamental consideration for concept representations, reflecting the fidelity of concept labeling. To this end, we formulate robustness quantification as input-space optimization problems and develop a comprehensive evaluation framework featuring realistic scenarios in which adversarial perturbations are crafted to manipulate SAE representations. Empirically, we find that tiny adversarial input perturbations can effectively manipulate concept-based interpretations in most scenarios without notably affecting the outputs of the base LLMs themselves. Overall, our results suggest that SAE concept representations are fragile and may be ill-suited for applications in model monitoring and oversight. 

**Abstract (ZH)**: 稀疏自编码器（SAEs）常用于通过将其映射到可人力解释的概念表示，来解析大型语言模型（LLMs）的内部激活。虽然现有SAE评估主要集中在重构稀疏性权衡、人类（自助）解释性和特征解耦等指标上，但忽略了概念表示对输入扰动具有鲁棒性这一关键方面。我们主张鲁棒性应是概念表示的基本考量因素，反映概念标注的准确性。为此，我们将鲁棒性量化形式化为输入空间优化问题，并开发了一个包含现实场景的综合评估框架，在这些场景中，恶意扰动被精心设计以操控SAE表示。实验结果表明，在大多数情况下，细微的恶意输入扰动可以有效操控基于概念的解释，而对基础LLM本身输出的影响并不明显。总体而言，我们的结果表明SAE概念表示是脆弱的，可能不适合用于模型监控和监督的应用中。 

---
# SLMEval: Entropy-Based Calibration for Human-Aligned Evaluation of Large Language Models 

**Title (ZH)**: SLMEval: 基于熵的校准以实现人工对齐评价的大语言模型 

**Authors**: Roland Daynauth, Christopher Clarke, Krisztian Flautner, Lingjia Tang, Jason Mars  

**Link**: [PDF](https://arxiv.org/pdf/2505.16003)  

**Abstract**: The LLM-as-a-Judge paradigm offers a scalable, reference-free approach for evaluating language models. Although several calibration techniques have been proposed to better align these evaluators with human judgment, prior studies focus primarily on narrow, well-structured benchmarks. As a result, it remains unclear whether such calibrations generalize to real-world, open-ended tasks.
In this work, we show that SOTA calibrated evaluators often fail in these settings, exhibiting weak or even negative correlation with human judgments. To address this, we propose SLMEval, a novel and efficient calibration method based on entropy maximization over a small amount of human preference data. By estimating a latent distribution over model quality and reweighting evaluator scores accordingly, SLMEval achieves strong correlation with human evaluations across two real-world production use cases and the public benchmark. For example, on one such task, SLMEval achieves a Spearman correlation of 0.57 with human judgments, while G-Eval yields a negative correlation. In addition, SLMEval reduces evaluation costs by 5-30x compared to GPT-4-based calibrated evaluators such as G-eval. 

**Abstract (ZH)**: LLM-as-a-Judge范式提供了无参考的大规模语言模型评估方法。尽管提出了一些校准技术以更准确地对齐这些评估者与人类判断，但之前的研究所关注的主要是一些狭窄且结构良好的基准。因此，尚不清楚这样的校准是否能够泛化到实际世界中的开放任务。

在本研究中，我们展示了最先进的校准评估者在这些场景中往往表现不佳，与人类判断表现出弱或甚至是负相关。为了解决这一问题，我们提出了一种基于少量人类偏好数据的熵最大化的新颖且高效的校准方法SLMEval。通过估计模型质量的潜在分布并相应地重新加权评价分数，SLMEval在两个实际世界生产应用场景和公共基准测试中实现了与人类评估的强相关性。例如，在其中一个任务中，SLMEval与人类判断之间的斯皮尔曼相关系数为0.57，而G-Eval则表现出负相关。此外，与基于GPT-4的校准评估者（如G-Eval）相比，SLMEval将评估成本降低了5到30倍。 

---
# Causal Interventions Reveal Shared Structure Across English Filler-Gap Constructions 

**Title (ZH)**: 因果干预揭示英语填充-空位构 constructions 中的共享结构 

**Authors**: Sasha Boguraev, Christopher Potts, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2505.16002)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful sources of evidence for linguists seeking to develop theories of syntax. In this paper, we argue that causal interpretability methods, applied to LLMs, can greatly enhance the value of such evidence by helping us characterize the abstract mechanisms that LLMs learn to use. Our empirical focus is a set of English filler-gap dependency constructions (e.g., questions, relative clauses). Linguistic theories largely agree that these constructions share many properties. Using experiments based in Distributed Interchange Interventions, we show that LLMs converge on similar abstract analyses of these constructions. These analyses also reveal previously overlooked factors -- relating to frequency, filler type, and surrounding context -- that could motivate changes to standard linguistic theory. Overall, these results suggest that mechanistic, internal analyses of LLMs can push linguistic theory forward. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已成为语言学家开发句法理论的重要证据来源。本文我们argue，将因果可解释性方法应用于LLMs，可以极大地增强这类证据的价值，帮助我们characterize LLMs学习使用的抽象机制。我们的实证重点是一组英语填充-空白依赖构造（例如，疑问句、从句）。语言学理论很大程度上认为这些构造共享许多属性。我们基于分布式互换干预的实验表明，LLMs对这些构造给出了类似的抽象分析。这些分析还揭示了以前未被注意到的因素——涉及频率、填充词类型以及周围语境——这些因素可能促使对标准语言理论进行修改。总体而言，这些结果表明，对LLMs的机械内部分析可以推动语言学理论的发展。 

---
# Leveraging Online Data to Enhance Medical Knowledge in a Small Persian Language Model 

**Title (ZH)**: 利用在线数据提升小规模波斯语医疗知识模型 

**Authors**: Mehrdad ghassabi, Pedram Rostami, Hamidreza Baradaran Kashani, Amirhossein Poursina, Zahra Kazemi, Milad Tavakoli  

**Link**: [PDF](https://arxiv.org/pdf/2505.16000)  

**Abstract**: The rapid advancement of language models has demonstrated the potential of artificial intelligence in the healthcare industry. However, small language models struggle with specialized domains in low-resource languages like Persian. While numerous medical-domain websites exist in Persian, no curated dataset or corpus has been available making ours the first of its kind. This study explores the enhancement of medical knowledge in a small language model by leveraging accessible online data, including a crawled corpus from medical magazines and a dataset of real doctor-patient QA pairs. We fine-tuned a baseline model using our curated data to improve its medical knowledge. Benchmark evaluations demonstrate that the fine-tuned model achieves improved accuracy in medical question answering and provides better responses compared to its baseline. This work highlights the potential of leveraging open-access online data to enrich small language models in medical fields, providing a novel solution for Persian medical AI applications suitable for resource-constrained environments. 

**Abstract (ZH)**: 语言模型的 rapid advancement 在医疗健康行业的应用潜力已得到证实，但小型语言模型在低资源语言如波斯语的专业领域中存在挑战。尽管存在许多波斯语医疗领域网站，但缺乏经过整理的数据集或语料库，使得我们的工作成为该领域的首创。本研究通过利用可获得的在线数据，包括从医学杂志抓取的语料库和真实的医生-患者问答数据集，探索了增强小型语言模型医学知识的方法。我们使用精选数据对基准模型进行了微调，以提高其医学知识水平。基准评估表明，微调后的模型在医学问答中的准确性和响应质量均优于基准模型。本研究强调了利用开放访问的在线数据来丰富小型语言模型在医学领域的可能性，提出了适合资源受限环境的波斯语医疗AI应用的新解决方案。 

---
# Pre-training Large Memory Language Models with Internal and External Knowledge 

**Title (ZH)**: 使用内部和外部知识预训练大规模语言模型 

**Authors**: Linxi Zhao, Sofian Zalouk, Christian K. Belardi, Justin Lovelace, Jin Peng Zhou, Kilian Q. Weinberger, Yoav Artzi, Jennifer J. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.15962)  

**Abstract**: Neural language models are black-boxes -- both linguistic patterns and factual knowledge are distributed across billions of opaque parameters. This entangled encoding makes it difficult to reliably inspect, verify, or update specific facts. We propose a new class of language models, Large Memory Language Models (LMLM) with a pre-training recipe that stores factual knowledge in both internal weights and an external database. Our approach strategically masks externally retrieved factual values from the training loss, thereby teaching the model to perform targeted lookups rather than relying on memorization in model weights. Our experiments demonstrate that LMLMs achieve competitive performance compared to significantly larger, knowledge-dense LLMs on standard benchmarks, while offering the advantages of explicit, editable, and verifiable knowledge bases. This work represents a fundamental shift in how language models interact with and manage factual knowledge. 

**Abstract (ZH)**: 大型记忆语言模型：通过内部权重和外部数据库存储事实知识的新类语言模型 

---
# Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey 

**Title (ZH)**: 面向大规模音语言模型综合评估的研究综述 

**Authors**: Chih-Kai Yang, Neo S. Ho, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.15957)  

**Abstract**: With advancements in large audio-language models (LALMs), which enhance large language models (LLMs) with auditory capabilities, these models are expected to demonstrate universal proficiency across various auditory tasks. While numerous benchmarks have emerged to assess LALMs' performance, they remain fragmented and lack a structured taxonomy. To bridge this gap, we conduct a comprehensive survey and propose a systematic taxonomy for LALM evaluations, categorizing them into four dimensions based on their objectives: (1) General Auditory Awareness and Processing, (2) Knowledge and Reasoning, (3) Dialogue-oriented Ability, and (4) Fairness, Safety, and Trustworthiness. We provide detailed overviews within each category and highlight challenges in this field, offering insights into promising future directions. To the best of our knowledge, this is the first survey specifically focused on the evaluations of LALMs, providing clear guidelines for the community. We will release the collection of the surveyed papers and actively maintain it to support ongoing advancements in the field. 

**Abstract (ZH)**: 随着大型音频语言模型（LALMs）的发展，这些模型通过增强大型语言模型（LLMs）的听觉能力，预期将在各种听觉任务中展现出普遍的专业能力。尽管已经出现了众多评价LALMs性能的基准，但它们仍较为零散且缺乏结构化的分类体系。为填补这一空白，我们进行了一项全面的综述，并提出了一套系统性的评价分类体系，根据其目标将LALM分为四个维度：（1）一般听觉意识和处理能力，（2）知识与推理，（3）对话导向能力，以及（4）公平性、安全性和可靠性。我们在每个分类中提供了详细概述，并突出了该领域的挑战，提供了未来研究方向的见解。据我们所知，这是首个专门针对LALM评价的综述，为该领域提供了明确的指南。我们将发布所调研论文的集合，并积极维护，以支持该领域的持续发展。 

---
# Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization 

**Title (ZH)**: 从大型语言模型中提取概率性知识进行贝叶斯网络参数化 

**Authors**: Aliakbar Nafar, Kristen Brent Venable, Zijun Cui, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15918)  

**Abstract**: Large Language Models (LLMs) have demonstrated potential as factual knowledge bases; however, their capability to generate probabilistic knowledge about real-world events remains understudied. This paper investigates using probabilistic knowledge inherent in LLMs to derive probability estimates for statements concerning events and their interrelationships captured via a Bayesian Network (BN). Using LLMs in this context allows for the parameterization of BNs, enabling probabilistic modeling within specific domains. Experiments on eighty publicly available Bayesian Networks, from healthcare to finance, demonstrate that querying LLMs about the conditional probabilities of events provides meaningful results when compared to baselines, including random and uniform distributions, as well as approaches based on next-token generation probabilities. We explore how these LLM-derived distributions can serve as expert priors to refine distributions extracted from minimal data, significantly reducing systematic biases. Overall, this work introduces a promising strategy for automatically constructing Bayesian Networks by combining probabilistic knowledge extracted from LLMs with small amounts of real-world data. Additionally, we evaluate several prompting strategies for eliciting probabilistic knowledge from LLMs and establish the first comprehensive baseline for assessing LLM performance in extracting probabilistic knowledge. 

**Abstract (ZH)**: 大型语言模型（LLMs）在事实知识库方面展现了潜力，但其生成关于现实世界事件的概率性知识的能力仍需进一步研究。本文探讨了利用LLMs内在的概率性知识来通过贝叶斯网络（BN）推导出关于事件及其相互关系的概率估计。在这种背景下使用LLMs允许对BN进行参数化，从而在特定领域内实现概率建模。实验表明，查询LLMs关于事件条件概率的结果相较于随机分布、均匀分布以及基于下一个词生成概率的方法，具有重要意义。我们研究了这些LLM衍生的分布如何作为专家先验知识，以 refinement 的方式改进从少量数据中提取的分布，从而显著减少系统性偏差。总体而言，本文提出了一种通过结合从LLMs中提取的概率性知识和少量现实世界数据自动构建贝叶斯网络的有前景策略。此外，我们评估了几种提示策略以从LLMs中引出概率性知识，并建立了第一个全面的基线来评估LLMs在提取概率性知识方面的表现。 

---
# What Lives? A meta-analysis of diverse opinions on the definition of life 

**Title (ZH)**: 生命何为？关于生命定义的多元观点元分析 

**Authors**: Reed Bender, Karina Kofman, Blaise Agüera y Arcas, Michael Levin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15849)  

**Abstract**: The question of "what is life?" has challenged scientists and philosophers for centuries, producing an array of definitions that reflect both the mystery of its emergence and the diversity of disciplinary perspectives brought to bear on the question. Despite significant progress in our understanding of biological systems, psychology, computation, and information theory, no single definition for life has yet achieved universal acceptance. This challenge becomes increasingly urgent as advances in synthetic biology, artificial intelligence, and astrobiology challenge our traditional conceptions of what it means to be alive. We undertook a methodological approach that leverages large language models (LLMs) to analyze a set of definitions of life provided by a curated set of cross-disciplinary experts. We used a novel pairwise correlation analysis to map the definitions into distinct feature vectors, followed by agglomerative clustering, intra-cluster semantic analysis, and t-SNE projection to reveal underlying conceptual archetypes. This methodology revealed a continuous landscape of the themes relating to the definition of life, suggesting that what has historically been approached as a binary taxonomic problem should be instead conceived as differentiated perspectives within a unified conceptual latent space. We offer a new methodological bridge between reductionist and holistic approaches to fundamental questions in science and philosophy, demonstrating how computational semantic analysis can reveal conceptual patterns across disciplinary boundaries, and opening similar pathways for addressing other contested definitional territories across the sciences. 

**Abstract (ZH)**: “生命是什么？”这一问题challenge了科学家和哲学家数世纪之久，产生了多种定义，这些定义既反映了其起源的神秘性，也反映了不同学科视角的多样性。尽管我们在生物学系统、心理学、计算和信息理论方面的理解取得了 significant 进展，但尚未达成对生命单一定义的普遍接受。随着合成生物学、人工智能和天体生物学的进展，我们对生命意义的传统认知面临新的挑战。我们采用了一种方法论方法，利用大规模语言模型（LLMs）分析由跨学科专家提供的生命定义。我们使用了一种新颖的成对相关分析，将定义映射为不同的特征向量，随后进行了凝聚聚类、簇内语义分析和t-SNE投影，以揭示潜在的概念原型。这种方法揭示了与生命定义相关的主题连续谱，表明历史上被视为二分分类问题的内容，应该被视为统一概念潜在空间内的分化视角。我们提供了一种方法论桥梁，将还原论和整体论方法应用于科学研究和哲学的基本问题，展示了如何通过计算语义分析揭示跨学科界限的概念模式，并为解决科学领域其他争议性定义领域开辟了类似的路径。 

---
# Transforming Decoder-Only Transformers for Accurate WiFi-Telemetry Based Indoor Localization 

**Title (ZH)**: 仅解码器变压器模型的变换以实现基于WiFi-遥测的精确室内定位 

**Authors**: Nayan Sanjay Bhatia, Katia Obraczka  

**Link**: [PDF](https://arxiv.org/pdf/2505.15835)  

**Abstract**: Wireless Fidelity (WiFi) based indoor positioning is a widely researched area for determining the position of devices within a wireless network. Accurate indoor location has numerous applications, such as asset tracking and indoor navigation. Despite advances in WiFi localization techniques -- in particular approaches that leverage WiFi telemetry -- their adoption in practice remains limited due to several factors including environmental changes that cause signal fading, multipath effects, interference, which, in turn, impact positioning accuracy. In addition, telemetry data differs depending on the WiFi device vendor, offering distinct features and formats; use case requirements can also vary widely. Currently, there is no unified model to handle all these variations effectively. In this paper, we present WiFiGPT, a Generative Pretrained Transformer (GPT) based system that is able to handle these variations while achieving high localization accuracy. Our experiments with WiFiGPT demonstrate that GPTs, in particular Large Language Models (LLMs), can effectively capture subtle spatial patterns in noisy wireless telemetry, making them reliable regressors. Compared to existing state-of-the-art methods, our method matches and often surpasses conventional approaches for multiple types of telemetry. Achieving sub-meter accuracy for RSSI and FTM and centimeter-level precision for CSI demonstrates the potential of LLM-based localisation to outperform specialized techniques, all without handcrafted signal processing or calibration. 

**Abstract (ZH)**: 基于WiFi的室内定位：一种能够处理变异并实现高精度定位的生成预训练变压器系统 

---
# UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models 

**Title (ZH)**: UltraEdit: 无需训练、特定主题和记忆的大型语言模型终身编辑 

**Authors**: Xiaojie Gu, Guangxu Chen, Jungang Li, Jia-Chen Gu, Xuming Hu, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14679)  

**Abstract**: Lifelong learning enables large language models (LLMs) to adapt to evolving information by continually updating their internal knowledge. An ideal system should support efficient, wide-ranging updates while preserving existing capabilities and ensuring reliable deployment. Model editing stands out as a promising solution for this goal, offering a focused and efficient way to revise a model's internal knowledge. Although recent paradigms have made notable progress, they often struggle to meet the demands of practical lifelong adaptation at scale. To bridge this gap, we propose ULTRAEDIT-a fundamentally new editing solution that is training-, subject- and memory-free, making it particularly well-suited for ultra-scalable, real-world lifelong model editing. ULTRAEDIT performs editing through a self-contained process that relies solely on lightweight linear algebra operations to compute parameter shifts, enabling fast and consistent parameter modifications with minimal overhead. To improve scalability in lifelong settings, ULTRAEDIT employs a lifelong normalization strategy that continuously updates feature statistics across turns, allowing it to adapt to distributional shifts and maintain consistency over time. ULTRAEDIT achieves editing speeds over 7x faster than the previous state-of-the-art method-which was also the fastest known approach-while consuming less than 1/3 the VRAM, making it the only method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU. Furthermore, we construct ULTRAEDITBENCH-the largest dataset in the field to date, with over 2M editing pairs-and demonstrate that our method supports up to 1M edits while maintaining high accuracy. Comprehensive experiments on four datasets and six models show that ULTRAEDIT consistently achieves superior performance across diverse model editing scenarios. Our code is available at: this https URL. 

**Abstract (ZH)**: 终身学习使大规模语言模型（LLMs）能够通过不断更新其内部知识来适应不断变化的信息。理想的系统应该支持高效、广泛的更新，同时保留现有能力并确保可靠的部署。模型编辑作为一种专注于高效更新模型内部知识的解决方案脱颖而出。尽管最近的范式取得了显著进展，但在大规模实际终身适应方面，它们往往难以满足需求。为解决这一问题，我们提出ULTRAEDIT——一种全新的、无训练、无领域知识和无内存负担的编辑解决方案，使其特别适合超大规模的实际终身模型编辑。ULTRAEDIT通过一个自包含的过程进行编辑，仅依赖轻量级线性代数操作来计算参数变动，从而实现快速且一致的参数修改，同时减少开销。为提高终身场景下的可扩展性，ULTRAEDIT采用了终身标准化策略，能够不断更新跨回合的特征统计，从而适应分布变化并保持时间一致。与上一种最先进方法相比，ULTRAEDIT的编辑速度高出7倍以上，且消耗的VRAM少于其1/3，从而成为唯一能够在24GB消费级GPU上编辑7B LLM的方法。此外，我们构建了迄今为止领域内最大的数据集ULTRAEDITBENCH，包含超过200万对编辑数据，并证明我们的方法在支持高达100万次编辑的同时保持高准确性。在四个数据集和六个模型的综合实验中，ULTRAEDIT在各种模型编辑场景中表现出优越性能。我们的代码可从以下链接获得：this https URL。 

---
# Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning 

**Title (ZH)**: 基于问题求解逻辑导向的在域复杂推理融合学习课程设计 

**Authors**: Xuetao Ma, Wenbin Jiang, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15401)  

**Abstract**: In-context learning (ICL) can significantly enhance the complex reasoning capabilities of large language models (LLMs), with the key lying in the selection and ordering of demonstration examples. Previous methods typically relied on simple features to measure the relevance between examples. We argue that these features are not sufficient to reflect the intrinsic connections between examples. In this study, we propose a curriculum ICL strategy guided by problem-solving logic. We select demonstration examples by analyzing the problem-solving logic and order them based on curriculum learning. Specifically, we constructed a problem-solving logic instruction set based on the BREAK dataset and fine-tuned a language model to analyze the problem-solving logic of examples. Subsequently, we selected appropriate demonstration examples based on problem-solving logic and assessed their difficulty according to the number of problem-solving steps. In accordance with the principles of curriculum learning, we ordered the examples from easy to hard to serve as contextual prompts. Experimental results on multiple benchmarks indicate that our method outperforms previous ICL approaches in terms of performance and efficiency, effectively enhancing the complex reasoning capabilities of LLMs. Our project will be publicly available subsequently. 

**Abstract (ZH)**: 基于问题求解逻辑的在上下文学习策略 

---
