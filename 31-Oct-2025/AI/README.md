# LLMs Process Lists With General Filter Heads 

**Title (ZH)**: LLMs处理列表的通用过滤头 

**Authors**: Arnab Sen Sharma, Giordano Rogers, Natalie Shapira, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2510.26784)  

**Abstract**: We investigate the mechanisms underlying a range of list-processing tasks in LLMs, and we find that LLMs have learned to encode a compact, causal representation of a general filtering operation that mirrors the generic "filter" function of functional programming. Using causal mediation analysis on a diverse set of list-processing tasks, we find that a small number of attention heads, which we dub filter heads, encode a compact representation of the filtering predicate in their query states at certain tokens. We demonstrate that this predicate representation is general and portable: it can be extracted and reapplied to execute the same filtering operation on different collections, presented in different formats, languages, or even in tasks. However, we also identify situations where transformer LMs can exploit a different strategy for filtering: eagerly evaluating if an item satisfies the predicate and storing this intermediate result as a flag directly in the item representations. Our results reveal that transformer LMs can develop human-interpretable implementations of abstract computational operations that generalize in ways that are surprisingly similar to strategies used in traditional functional programming patterns. 

**Abstract (ZH)**: 我们调查了大规模语言模型中一系列列表处理任务的机制，并发现这些模型已经学习到了一种紧凑的、具有因果性的过滤操作表示，这种表示类似于函数编程中的通用“过滤”函数。通过使用因果中介分析对一系列不同的列表处理任务进行分析，我们发现少量的注意力头（我们称之为过滤头），在某些标记的查询状态中，编码了过滤谓词的紧凑表示。我们证明了这种谓词表示是通用且可移植的：它可以被提取并应用于在不同的集合、不同格式、不同语言甚至不同任务中执行相同的过滤操作。然而，我们还发现变长模型在某些情况下会利用不同的过滤策略：提前评估项目是否满足谓词，并直接将这个中间结果作为标志存储在项目表示中。我们的结果揭示了变长模型可以发展出人类可解释的抽象计算操作的实现方式，并且这些操作的泛化方式出乎意料地类似于传统函数编程模式所使用的方法。 

---
# The Oversight Game: Learning to Cooperatively Balance an AI Agent's Safety and Autonomy 

**Title (ZH)**: 监管博弈：学习协同平衡AI代理的安全与自主性 

**Authors**: William Overman, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2510.26752)  

**Abstract**: As increasingly capable agents are deployed, a central safety question is how to retain meaningful human control without modifying the underlying system. We study a minimal control interface where an agent chooses whether to act autonomously (play) or defer (ask), while a human simultaneously chooses whether to be permissive (trust) or to engage in oversight (oversee). If the agent defers, the human's choice determines the outcome, potentially leading to a corrective action or a system shutdown. We model this interaction as a two-player Markov Game. Our analysis focuses on cases where this game qualifies as a Markov Potential Game (MPG), a class of games where we can provide an alignment guarantee: under a structural assumption on the human's value function, any decision by the agent to act more autonomously that benefits itself cannot harm the human's value. We also analyze extensions to this MPG framework. Theoretically, this perspective provides conditions for a specific form of intrinsic alignment. If the reward structures of the human-agent game meet these conditions, we have a formal guarantee that the agent improving its own outcome will not harm the human's. Practically, this model motivates a transparent control layer with predictable incentives where the agent learns to defer when risky and act when safe, while its pretrained policy and the environment's reward structure remain untouched. Our gridworld simulation shows that through independent learning, the agent and human discover their optimal oversight roles. The agent learns to ask when uncertain and the human learns when to oversee, leading to an emergent collaboration that avoids safety violations introduced post-training. This demonstrates a practical method for making misaligned models safer after deployment. 

**Abstract (ZH)**: 随着日益 capable 的代理被部署，一个核心的安全问题是如何在不修改底层系统的情况下保留有意义的人类控制。我们研究了一个最小控制接口，在该接口中，代理选择是否自主行动（play）或推迟行动（ask），同时人类同时选择是否给予信任（trust）或进行监督（oversee）。如果代理推迟行动，人类的选择将决定结果，可能会导致纠正措施或系统关闭。我们将这种交互建模为一个两玩家马尔可夫博弈。我们的分析集中在当该游戏符合马尔可夫潜力博弈（MPG）类标准时的情况，在这种情况下，可以在人类价值函数的结构假设下提供对齐保证：任何有利于代理自身但不损害人类价值的自主行动决策都是安全的。我们还分析了MPG框架的扩展。理论上，这种视角提供了特定形式内在对齐的条件。如果人类-代理游戏的奖励结构满足这些条件，我们有正式保证，代理改善自身结果不会损害人类。实践上，这种模型激励了一个透明的控制层，具有可预测的激励机制，其中代理在冒险时学会推迟行动，在安全时学会自主行动，而代理的预训练策略和环境的奖励结构保持不变。我们的网格世界模拟显示，通过独立学习，代理和人类发现了它们的最优监督角色。代理在不确定时学会请求，人类在需要时学会监督，从而形成了一种新兴的合作模式，避免了训练后引入的安全违规行为。这证明了一种在部署后使未对齐模型更安全的实用方法。 

---
# Cross-Platform Evaluation of Reasoning Capabilities in Foundation Models 

**Title (ZH)**: 跨平台评估基础模型的推理能力 

**Authors**: J. de Curtò, I. de Zarzà, Pablo García, Jordi Cabot  

**Link**: [PDF](https://arxiv.org/pdf/2510.26732)  

**Abstract**: This paper presents a comprehensive cross-platform evaluation of reasoning capabilities in contemporary foundation models, establishing an infrastructure-agnostic benchmark across three computational paradigms: HPC supercomputing (MareNostrum 5), cloud platforms (Nebius AI Studio), and university clusters (a node with eight H200 GPUs).
We evaluate 15 foundation models across 79 problems spanning eight academic domains (Physics, Mathematics, Chemistry, Economics, Biology, Statistics, Calculus, and Optimization) through three experimental phases: (1) Baseline establishment: Six models (Mixtral-8x7B, Phi-3, LLaMA 3.1-8B, Gemma-2-9b, Mistral-7B, OLMo-7B) evaluated on 19 problems using MareNostrum 5, establishing methodology and reference performance; (2) Infrastructure validation: The 19-problem benchmark repeated on university cluster (seven models including Falcon-Mamba state-space architecture) and Nebius AI Studio (nine state-of-the-art models: Hermes-4 70B/405B, LLaMA 3.1-405B/3.3-70B, Qwen3 30B/235B, DeepSeek-R1, GPT-OSS 20B/120B) to confirm infrastructure-agnostic reproducibility; (3) Extended evaluation: Full 79-problem assessment on both university cluster and Nebius platforms, probing generalization at scale across architectural diversity.
The findings challenge conventional scaling assumptions, establish training data quality as more critical than model size, and provide actionable guidelines for model selection across educational, production, and research contexts. The tri-infrastructure methodology and 79-problem benchmark enable longitudinal tracking of reasoning capabilities as foundation models evolve. 

**Abstract (ZH)**: 本文提出了一种跨平台评估当代基础模型推理能力的全面方法，建立了在三种计算范式下的基础设施无关基准：高性能计算超级计算机（MareNostrum 5）、云平台（Nebius AI Studio）和大学集群（包含8个H200 GPU的节点）。

我们通过三个实验阶段评估了15个基础模型在79个问题上的表现，涵盖八个学术领域（物理、数学、化学、经济学、生物学、统计学、微积分和优化）：（1）基线建立：使用MareNostrum 5评估六种模型（Mixtral-8x7B、Phi-3、LLaMA 3.1-8B、Gemma-2-9b、Mistral-7B、OLMo-7B），建立方法和参考性能；（2）基础设施验证：在大学集群和Nebius AI Studio上重复19个问题基准测试（包括Falcon-Mamba状态空间架构的七种模型和九种顶尖模型：Hermes-4 70B/405B、LLaMA 3.1-405B/3.3-70B、Qwen3 30B/235B、DeepSeek-R1、GPT-OSS 20B/120B），验证基础设施无关的可再现性；（3）扩展评估：在大学集群和Nebius平台对全部79个问题进行评估，探索广泛的架构多样性下的泛化能力。

研究结果挑战了传统的扩展假设，确立了训练数据质量比模型规模更为关键，并提供了适用于教育、生产和研究环境的模型选择指南。三平台方法和79个问题基准测试可实现基础模型随时间推进的纵向追踪能力。 

---
# Unveiling Intrinsic Text Bias in Multimodal Large Language Models through Attention Key-Space Analysis 

**Title (ZH)**: 通过注意键空间分析揭示多模态大型语言模型中的固有文本偏见 

**Authors**: Xinhan Zheng, Huyu Wu, Xueting Wang, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26721)  

**Abstract**: Multimodal large language models (MLLMs) exhibit a pronounced preference for textual inputs when processing vision-language data, limiting their ability to reason effectively from visual evidence. Unlike prior studies that attribute this text bias to external factors such as data imbalance or instruction tuning, we propose that the bias originates from the model's internal architecture. Specifically, we hypothesize that visual key vectors (Visual Keys) are out-of-distribution (OOD) relative to the text key space learned during language-only pretraining. Consequently, these visual keys receive systematically lower similarity scores during attention computation, leading to their under-utilization in the context representation. To validate this hypothesis, we extract key vectors from LLaVA and Qwen2.5-VL and analyze their distributional structures using qualitative (t-SNE) and quantitative (Jensen-Shannon divergence) methods. The results provide direct evidence that visual and textual keys occupy markedly distinct subspaces within the attention space. The inter-modal divergence is statistically significant, exceeding intra-modal variation by several orders of magnitude. These findings reveal that text bias arises from an intrinsic misalignment within the attention key space rather than solely from external data factors. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在处理视觉语言数据时表现出对文本输入的明显偏好，限制了其从视觉证据中有效推理的能力。不同于以往将这种文本偏见归因于外部因素如数据不平衡或指令微调的研究，我们认为这种偏见源自模型内部架构。具体来说，我们假设视觉关键向量（Visual Keys）在仅语言预训练过程中学习到的文本关键空间中是离群的。因此，这些视觉关键向量在注意力计算中获得系统地较低的相似度分数，导致它们在上下文表示中的利用率较低。为了验证这一假设，我们从LLaVA和Qwen2.5-VL中提取关键向量，并使用定性（t-SNE）和定量（Jensen-Shannon距离）方法分析其分布结构。结果直接证明了视觉和文本关键向量在注意力空间中占据明显不同的子空间。跨模态的差异在统计上是显著的，远超过同模态变异性。这些发现揭示了文本偏见源于注意力关键空间中的内在不匹配，而不仅仅是外部数据因素。 

---
# Delegated Authorization for Agents Constrained to Semantic Task-to-Scope Matching 

**Title (ZH)**: 代理的委托授权，受限于语义任务到范围匹配 

**Authors**: Majed El Helou, Chiara Troiani, Benjamin Ryder, Jean Diaconu, Hervé Muyal, Marcelo Yannuzzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.26702)  

**Abstract**: Authorizing Large Language Model driven agents to dynamically invoke tools and access protected resources introduces significant risks, since current methods for delegating authorization grant overly broad permissions and give access to tools allowing agents to operate beyond the intended task scope. We introduce and assess a delegated authorization model enabling authorization servers to semantically inspect access requests to protected resources, and issue access tokens constrained to the minimal set of scopes necessary for the agents' assigned tasks. Given the unavailability of datasets centered on delegated authorization flows, particularly including both semantically appropriate and inappropriate scope requests for a given task, we introduce ASTRA, a dataset and data generation pipeline for benchmarking semantic matching between tasks and scopes. Our experiments show both the potential and current limitations of model-based matching, particularly as the number of scopes needed for task completion increases. Our results highlight the need for further research into semantic matching techniques enabling intent-aware authorization for multi-agent and tool-augmented applications, including fine-grained control, such as Task-Based Access Control (TBAC). 

**Abstract (ZH)**: 授权大型语言模型驱动的代理动态调用工具和访问受保护资源引入了重大风险，因为当前的方法在委派授权时授予了过于广泛的权限，并提供了允许代理超出预定任务范围操作的工具访问。我们提出并评估了一种委派授权模型，使授权服务器能够语义检查对受保护资源的访问请求，并发放仅限于代理分配任务所需最小权限集的访问令牌。由于缺乏专注于委派授权流程的数据集，特别是包括针对给定任务的语义合适和不合适的作用范围请求的数据集，我们介绍了ASTRA数据集和数据生成管道，用于基准测试任务和作用范围之间的语义匹配。我们的实验结果显示了基于模型的匹配的潜力及其当前限制，特别是在所需作用范围的数量增加时。我们的结果强调了需要进一步研究语义匹配技术的重要性，这些技术能够为多代理和工具增强应用提供意图感知的授权，包括细粒度控制，例如基于任务的访问控制（TBAC）。 

---
# The Era of Agentic Organization: Learning to Organize with Language Models 

**Title (ZH)**: 代_agents_组织：学会与语言模型组织 

**Authors**: Zewen Chi, Li Dong, Qingxiu Dong, Yaru Hao, Xun Wu, Shaohan Huang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.26658)  

**Abstract**: We envision a new era of AI, termed agentic organization, where agents solve complex problems by working collaboratively and concurrently, enabling outcomes beyond individual intelligence. To realize this vision, we introduce asynchronous thinking (AsyncThink) as a new paradigm of reasoning with large language models, which organizes the internal thinking process into concurrently executable structures. Specifically, we propose a thinking protocol where an organizer dynamically assigns sub-queries to workers, merges intermediate knowledge, and produces coherent solutions. More importantly, the thinking structure in this protocol can be further optimized through reinforcement learning. Experiments demonstrate that AsyncThink achieves 28% lower inference latency compared to parallel thinking while improving accuracy on mathematical reasoning. Moreover, AsyncThink generalizes its learned asynchronous thinking capabilities, effectively tackling unseen tasks without additional training. 

**Abstract (ZH)**: 我们设想了一个新时代的AI，称为自主组织时代，在这个时代中，代理通过协作和并发解决问题，从而实现超越个体智能的结果。为了实现这一愿景，我们引入了异步思考（AsyncThink）作为一种新的大规模语言模型推理范式，将内部思考过程组织成为可并发执行的结构。具体而言，我们提出了一种思考协议，其中组织者动态分配子查询、合并中间知识并生成连贯的解决方案。更重要的是，该协议中的思考结构可以通过强化学习进一步优化。实验表明，与并行思考相比，AsyncThink 的推理延迟降低了 28%，并且在数学推理方面提高了准确性。此外，AsyncThink 能够泛化其学习到的异步思考能力，有效地处理未见过的任务而无需额外训练。 

---
# Normative Reasoning in Large Language Models: A Comparative Benchmark from Logical and Modal Perspectives 

**Title (ZH)**: 大型语言模型中的规范推理：从逻辑和模态视角的比较基准 

**Authors**: Kentaro Ozeki, Risako Ando, Takanobu Morishita, Hirohiko Abe, Koji Mineshima, Mitsuhiro Okada  

**Link**: [PDF](https://arxiv.org/pdf/2510.26606)  

**Abstract**: Normative reasoning is a type of reasoning that involves normative or deontic modality, such as obligation and permission. While large language models (LLMs) have demonstrated remarkable performance across various reasoning tasks, their ability to handle normative reasoning remains underexplored. In this paper, we systematically evaluate LLMs' reasoning capabilities in the normative domain from both logical and modal perspectives. Specifically, to assess how well LLMs reason with normative modals, we make a comparison between their reasoning with normative modals and their reasoning with epistemic modals, which share a common formal structure. To this end, we introduce a new dataset covering a wide range of formal patterns of reasoning in both normative and epistemic domains, while also incorporating non-formal cognitive factors that influence human reasoning. Our results indicate that, although LLMs generally adhere to valid reasoning patterns, they exhibit notable inconsistencies in specific types of normative reasoning and display cognitive biases similar to those observed in psychological studies of human reasoning. These findings highlight challenges in achieving logical consistency in LLMs' normative reasoning and provide insights for enhancing their reliability. All data and code are released publicly at this https URL. 

**Abstract (ZH)**: 规范推理是一种涉及义务和许可等规范或义理模态的推理类型。尽管大规模语言模型（LLMs）在各种推理任务中展现了卓越的表现，但它们处理规范推理的能力仍鲜有研究。本文从逻辑和模态 perspective 出发，系统评估 LLMs 在规范领域内的推理能力。为此，我们通过将 LLMs 的规范模态推理与表征模态进行比较，来评估它们的规范模态推理能力，这两者共享类似的正式结构。为此，我们引入了一个新的数据集，涵盖了规范和表征领域广泛形式推理模式的同时，也融入了影响人类推理的非形式认知因素。我们的结果显示，尽管 LLMs 通常遵循有效的推理模式，但在特定类型的规范推理中表现出明显的不一致性，并展示了与人类推理心理学研究中观察到的认知偏差相似的现象。这些发现突出了在 LLMs 的规范推理中实现逻辑一致性的挑战，并为增强其可靠性提供了见解。所有数据和代码已在 https://github.com/alibaba/Qwen-RLM 公开发布。 

---
# Agentic AI Home Energy Management System: A Large Language Model Framework for Residential Load Scheduling 

**Title (ZH)**: 代理型AI家庭能源管理系统：面向居民负荷调度的大语言模型框架 

**Authors**: Reda El Makroum, Sebastian Zwickl-Bernhard, Lukas Kranzl  

**Link**: [PDF](https://arxiv.org/pdf/2510.26603)  

**Abstract**: The electricity sector transition requires substantial increases in residential demand response capacity, yet Home Energy Management Systems (HEMS) adoption remains limited by user interaction barriers requiring translation of everyday preferences into technical parameters. While large language models have been applied to energy systems as code generators and parameter extractors, no existing implementation deploys LLMs as autonomous coordinators managing the complete workflow from natural language input to multi-appliance scheduling. This paper presents an agentic AI HEMS where LLMs autonomously coordinate multi-appliance scheduling from natural language requests to device control, achieving optimal scheduling without example demonstrations. A hierarchical architecture combining one orchestrator with three specialist agents uses the ReAct pattern for iterative reasoning, enabling dynamic coordination without hardcoded workflows while integrating Google Calendar for context-aware deadline extraction. Evaluation across three open-source models using real Austrian day-ahead electricity prices reveals substantial capability differences. Llama-3.3-70B successfully coordinates all appliances across all scenarios to match cost-optimal benchmarks computed via mixed-integer linear programming, while other models achieve perfect single-appliance performance but struggle to coordinate all appliances simultaneously. Progressive prompt engineering experiments demonstrate that analytical query handling without explicit guidance remains unreliable despite models' general reasoning capabilities. We open-source the complete system including orchestration logic, agent prompts, tools, and web interfaces to enable reproducibility, extension, and future research. 

**Abstract (ZH)**: 电力部门转型需要显著增加住宅需求响应能力，然而家庭能源管理系统（HEMS）的采用受限于用户交互障碍，即需要将日常偏好转化为技术参数。虽然大型语言模型已被应用于能源系统中作为代码生成器和参数抽取器，但目前没有任何实现将LLM部署为自主协调者，以管理从自然语言输入到多设备调度的完整工作流程。本文介绍了一种自主型AI HEMS，其中LLM自主协调从自然语言请求到设备控制的多设备调度，实现最优调度而无需示例演示。该系统采用分层架构结合一个协调器和三个专家代理，使用ReAct模式进行迭代推理，实现动态协调而不需硬编码的工作流程，并集成Google日历以实现上下文感知的截止日期提取。使用三个开源模型对实际奥地利次日电价进行评估，结果显示不同模型的能力存在显著差异。Llama-3.3-70B成功协调所有设备以匹配通过混合整数线性规划计算的成本最优基准，而其他模型在单设备上表现完美但在同时协调所有设备方面面临困难。渐进式提示工程技术实验表明，即使在利用模型的普遍推理能力的情况下，无具体指导的分析查询处理仍不可靠。我们开源了整个系统，包括协调逻辑、代理提示、工具和Web界面，以实现可重复性、扩展性和未来研究。 

---
# EdgeRunner 20B: Military Task Parity with GPT-5 while Running on the Edge 

**Title (ZH)**: EdgeRunner 20B: 在边缘设备上运行与GPT-5军事任务性能相当 

**Authors**: Jack FitzGerald, Aristotelis Lazaridis, Dylan Bates, Aman Sharma, Jonnathan Castillo, Yousif Azami, Sean Bailey, Jeremy Cao, Peter Damianov, Kevin de Haan, Luke Kerbs, Vincent Lu, Joseph Madigan, Jeremy McLaurin, Jonathan Tainer, Dave Anderson, Jonathan Beck, Jamie Cuticello, Colton Malkerson, Tyler Saltsman  

**Link**: [PDF](https://arxiv.org/pdf/2510.26550)  

**Abstract**: We present EdgeRunner 20B, a fine-tuned version of gpt-oss-20b optimized for military tasks. EdgeRunner 20B was trained on 1.6M high-quality records curated from military documentation and websites. We also present four new tests sets: (a) combat arms, (b) combat medic, (c) cyber operations, and (d) mil-bench-5k (general military knowledge). On these military test sets, EdgeRunner 20B matches or exceeds GPT-5 task performance with 95%+ statistical significance, except for the high reasoning setting on the combat medic test set and the low reasoning setting on the mil-bench-5k test set. Versus gpt-oss-20b, there is no statistically-significant regression on general-purpose benchmarks like ARC-C, GPQA Diamond, GSM8k, IFEval, MMLU Pro, or TruthfulQA, except for GSM8k in the low reasoning setting. We also present analyses on hyperparameter settings, cost, and throughput. These findings show that small, locally-hosted models are ideal solutions for data-sensitive operations such as in the military domain, allowing for deployment in air-gapped edge devices. 

**Abstract (ZH)**: EdgeRunner 20B： toward optimized edge computing for military tasks 

---
# Human-AI Complementarity: A Goal for Amplified Oversight 

**Title (ZH)**: 人机互补：增强监督的目标 

**Authors**: Rishub Jain, Sophie Bridgers, Lili Janzer, Rory Greig, Tian Huey Teh, Vladimir Mikulik  

**Link**: [PDF](https://arxiv.org/pdf/2510.26518)  

**Abstract**: Human feedback is critical for aligning AI systems to human values. As AI capabilities improve and AI is used to tackle more challenging tasks, verifying quality and safety becomes increasingly challenging. This paper explores how we can leverage AI to improve the quality of human oversight. We focus on an important safety problem that is already challenging for humans: fact-verification of AI outputs. We find that combining AI ratings and human ratings based on AI rater confidence is better than relying on either alone. Giving humans an AI fact-verification assistant further improves their accuracy, but the type of assistance matters. Displaying AI explanation, confidence, and labels leads to over-reliance, but just showing search results and evidence fosters more appropriate trust. These results have implications for Amplified Oversight -- the challenge of combining humans and AI to supervise AI systems even as they surpass human expert performance. 

**Abstract (ZH)**: 人类反馈对于对齐AI系统与人类价值观至关重要。随着AI能力的提升和AI被用于应对更具挑战性的任务，验证质量和安全性变得越来越有挑战性。本文探讨了我们如何利用AI来提高人类监督的质量。我们专注于一个已经对人类构成挑战的重要安全问题：AI输出的事实核查。我们发现，基于AI评分者信心结合AI评分和人类评分优于单独依赖其中任何一种。给人类提供AI事实核查助手将进一步提高其准确性，但协助方式很重要。显示AI解释、信心和标签会导致过度依赖，而仅仅展示搜索结果和证据则能促进更适当的信任。这些结果对于放大监督——即使AI超越人类专家性能，人类和AI结合监督AI系统的挑战——具有重要意义。 

---
# Context Engineering 2.0: The Context of Context Engineering 

**Title (ZH)**: Context 工程 2.0：Context 工程 的上下文 

**Authors**: Qishuo Hua, Lyumanshan Ye, Dayuan Fu, Yang Xiao, Xiaojie Cai, Yunze Wu, Jifan Lin, Junfei Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26493)  

**Abstract**: Karl Marx once wrote that ``the human essence is the ensemble of social relations'', suggesting that individuals are not isolated entities but are fundamentally shaped by their interactions with other entities, within which contexts play a constitutive and essential role. With the advent of computers and artificial intelligence, these contexts are no longer limited to purely human--human interactions: human--machine interactions are included as well. Then a central question emerges: How can machines better understand our situations and purposes? To address this challenge, researchers have recently introduced the concept of context engineering. Although it is often regarded as a recent innovation of the agent era, we argue that related practices can be traced back more than twenty years. Since the early 1990s, the field has evolved through distinct historical phases, each shaped by the intelligence level of machines: from early human--computer interaction frameworks built around primitive computers, to today's human--agent interaction paradigms driven by intelligent agents, and potentially to human--level or superhuman intelligence in the future. In this paper, we situate context engineering, provide a systematic definition, outline its historical and conceptual landscape, and examine key design considerations for practice. By addressing these questions, we aim to offer a conceptual foundation for context engineering and sketch its promising future. This paper is a stepping stone for a broader community effort toward systematic context engineering in AI systems. 

**Abstract (ZH)**: 卡尔·马克思曾经写道，“人的本质是一切社会关系的总和”，暗示个人不是孤立的存在体，而是从根本上被与其他存在体的互动所塑造，在这些互动中，背景扮演着构成性和本质性的角色。随着计算机和人工智能的发展，这些背景不仅限于人与人的互动：人与机器的互动也被包括在内。随之而来的一个核心问题是：机器如何更好地理解我们的处境和目的？为了解决这一挑战，研究人员最近引入了“情境工程”这一概念。尽管它通常被视为智能代理时代的一项近期创新，我们认为相关的实践可以追溯到二十多年前。自20世纪90年代初以来，该领域经历了不同的历史阶段，每个阶段都由机器的智能水平所塑造：从早期围绕原始计算机构建的人机交互框架到今天由智能代理驱动的人机互动范式，并可能迈向类似人类或超人类的智能未来。本文将情境工程置于具体背景中，提供系统性定义，概述其历史和概念框架，并探讨实践中的关键设计考虑。通过这些问题的回答，我们旨在为情境工程提供概念基础，并勾勒其充满希望的未来。本文是促进AI系统中系统化情境工程更广泛社区努力的基石。 

---
# LINK-KG: LLM-Driven Coreference-Resolved Knowledge Graphs for Human Smuggling Networks 

**Title (ZH)**: LINK-KG: 由大语言模型驱动且消除了指代冲突的知识图谱用于人口走私网络 

**Authors**: Dipak Meher, Carlotta Domeniconi, Guadalupe Correa-Cabrera  

**Link**: [PDF](https://arxiv.org/pdf/2510.26486)  

**Abstract**: Human smuggling networks are complex and constantly evolving, making them difficult to analyze comprehensively. Legal case documents offer rich factual and procedural insights into these networks but are often long, unstructured, and filled with ambiguous or shifting references, posing significant challenges for automated knowledge graph (KG) construction. Existing methods either overlook coreference resolution or fail to scale beyond short text spans, leading to fragmented graphs and inconsistent entity linking. We propose LINK-KG, a modular framework that integrates a three-stage, LLM-guided coreference resolution pipeline with downstream KG extraction. At the core of our approach is a type-specific Prompt Cache, which consistently tracks and resolves references across document chunks, enabling clean and disambiguated narratives for structured knowledge graph construction from both short and long legal texts. LINK-KG reduces average node duplication by 45.21% and noisy nodes by 32.22% compared to baseline methods, resulting in cleaner and more coherent graph structures. These improvements establish LINK-KG as a strong foundation for analyzing complex criminal networks. 

**Abstract (ZH)**: 人类偷渡网络复杂且不断演变，难以进行全面分析。法律案例文件提供了丰富的事实和程序性见解，但往往是长篇、无结构的，并且充满了模糊或不断变化的引用，给自动知识图谱（KG）构建带来了重大挑战。现有方法要么忽略了共指消解，要么无法将处理范围扩大到短文本片段，导致知识图谱碎片化和实体链接不一致。我们提出了一种模块化框架LINK-KG，该框架集成了一个三阶段、基于大语言模型（LLM）的共指消解管道，并在下游KG提取中加以应用。我们的方法核心是一个特定类型的提示缓存，能够跨文档片段一致地跟踪和解决引用，从而为从短文本和长文本法律文件中构建结构化知识图谱提供清晰和去模糊化的叙述。与基线方法相比，LINK-KG将平均节点重复率降低了45.21%，噪声节点降低了32.22%，形成了更为清洁和连贯的图结构。这些改进奠定了LINK-KG分析复杂犯罪网络的坚实基础。 

---
# Who Has The Final Say? Conformity Dynamics in ChatGPT's Selections 

**Title (ZH)**: 谁拥有最终决定权？ChatGPT选择中的从众动态 

**Authors**: Clarissa Sabrina Arlinghaus, Tristan Kenneweg, Barbara Hammer, Günter W. Maier  

**Link**: [PDF](https://arxiv.org/pdf/2510.26481)  

**Abstract**: Large language models (LLMs) such as ChatGPT are increasingly integrated into high-stakes decision-making, yet little is known about their susceptibility to social influence. We conducted three preregistered conformity experiments with GPT-4o in a hiring context. In a baseline study, GPT consistently favored the same candidate (Profile C), reported moderate expertise (M = 3.01) and high certainty (M = 3.89), and rarely changed its choice. In Study 1 (GPT + 8), GPT faced unanimous opposition from eight simulated partners and almost always conformed (99.9%), reporting lower certainty and significantly elevated self-reported informational and normative conformity (p < .001). In Study 2 (GPT + 1), GPT interacted with a single partner and still conformed in 40.2% of disagreement trials, reporting less certainty and more normative conformity. Across studies, results demonstrate that GPT does not act as an independent observer but adapts to perceived social consensus. These findings highlight risks of treating LLMs as neutral decision aids and underline the need to elicit AI judgments prior to exposing them to human opinions. 

**Abstract (ZH)**: 大型语言模型（LLMs）如ChatGPT在高风险决策中的社会影响 susceptibility 分析：三项预注册实验探究 

---
# Chain-of-Thought Hijacking 

**Title (ZH)**: 连锁思考窃取 

**Authors**: Jianli Zhao, Tingchen Fu, Rylan Schaeffer, Mrinank Sharma, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2510.26418)  

**Abstract**: Large reasoning models (LRMs) achieve higher task performance by allocating more inference-time compute, and prior works suggest this scaled reasoning may also strengthen safety by improving refusal. Yet we find the opposite: the same reasoning can be used to bypass safeguards. We introduce Chain-of-Thought Hijacking, a jailbreak attack on reasoning models. The attack pads harmful requests with long sequences of harmless puzzle reasoning. Across HarmBench, CoT Hijacking reaches a 99%, 94%, 100%, and 94% attack success rate (ASR) on Gemini 2.5 Pro, GPT o4 mini, Grok 3 mini, and Claude 4 Sonnet, respectively - far exceeding prior jailbreak methods for LRMs. To understand the effectiveness of our attack, we turn to a mechanistic analysis, which shows that mid layers encode the strength of safety checking, while late layers encode the verification outcome. Long benign CoT dilutes both signals by shifting attention away from harmful tokens. Targeted ablations of attention heads identified by this analysis causally decrease refusal, confirming their role in a safety subnetwork. These results show that the most interpretable form of reasoning - explicit CoT - can itself become a jailbreak vector when combined with final-answer cues. We release prompts, outputs, and judge decisions to facilitate replication. 

**Abstract (ZH)**: 大型推理模型中的思维链条劫持：一种针对推理模型的 Jailbreak 攻击 

---
# MedSAE: Dissecting MedCLIP Representations with Sparse Autoencoders 

**Title (ZH)**: MedSAE: 用稀疏自编码器剖析 MedCLIP 表征 

**Authors**: Riccardo Renzulli, Colas Lepoutre, Enrico Cassano, Marco Grangetto  

**Link**: [PDF](https://arxiv.org/pdf/2510.26411)  

**Abstract**: Artificial intelligence in healthcare requires models that are accurate and interpretable. We advance mechanistic interpretability in medical vision by applying Medical Sparse Autoencoders (MedSAEs) to the latent space of MedCLIP, a vision-language model trained on chest radiographs and reports. To quantify interpretability, we propose an evaluation framework that combines correlation metrics, entropy analyzes, and automated neuron naming via the MedGEMMA foundation model. Experiments on the CheXpert dataset show that MedSAE neurons achieve higher monosemanticity and interpretability than raw MedCLIP features. Our findings bridge high-performing medical AI and transparency, offering a scalable step toward clinically reliable representations. 

**Abstract (ZH)**: 医疗保健中的人工智能需要准确可解释的模型。我们通过将Medical Sparse Autoencoders (MedSAEs) 应用于MedCLIP（一种在胸片和报告上训练的vision-language模型）的潜在空间，推进了医疗视觉的机械可解释性。为了量化可解释性，我们提出了一种结合相关性度量、熵分析和通过MedGEMMA基础模型自动神经元命名的评价框架。在CheXpert数据集上的实验表明，MedSAE神经元的单义性和可解释性高于原始的MedCLIP特征。我们的研究结果将高性能的医疗AI与透明性相结合，提供了一条走向临床可靠表示的可扩展步骤。 

---
# Autograder+: A Multi-Faceted AI Framework for Rich Pedagogical Feedback in Programming Education 

**Title (ZH)**: Autograder+: 一个多方面的人工智能框架，用于编程教育中的丰富教学反馈 

**Authors**: Vikrant Sahu, Gagan Raj Gupta, Raghav Borikar, Nitin Mane  

**Link**: [PDF](https://arxiv.org/pdf/2510.26402)  

**Abstract**: The rapid growth of programming education has outpaced traditional assessment tools, leaving faculty with limited means to provide meaningful, scalable feedback. Conventional autograders, while efficient, act as black-box systems that simply return pass/fail results, offering little insight into student thinking or learning needs.
Autograder+ is designed to shift autograding from a purely summative process to a formative learning experience. It introduces two key capabilities: automated feedback generation using a fine-tuned Large Language Model, and visualization of student code submissions to uncover learning patterns. The model is fine-tuned on curated student code and expert feedback to ensure pedagogically aligned, context-aware guidance.
In evaluation across 600 student submissions from multiple programming tasks, the system produced feedback with strong semantic alignment to instructor comments. For visualization, contrastively learned code embeddings trained on 1,000 annotated submissions enable grouping solutions into meaningful clusters based on functionality and approach. The system also supports prompt-pooling, allowing instructors to guide feedback style through selected prompt templates.
By integrating AI-driven feedback, semantic clustering, and interactive visualization, Autograder+ reduces instructor workload while supporting targeted instruction and promoting stronger learning outcomes. 

**Abstract (ZH)**: 编程教育的迅速发展超出了传统评估工具的能力，使得教师在提供有意义且可扩展的反馈方面手段有限。传统的自动评分系统虽然高效，但作为黑盒系统，仅能返回通过或未通过的结果，无法提供关于学生思考或学习需求的洞察。

Autograder+旨在将自动评分从单纯的总结性过程转变为形成性学习体验。它引入了两项关键功能：使用微调后的大型语言模型自动生成反馈和可视化学生代码提交，以揭示学习模式。该模型通过精选的学生代码和专家反馈进行微调，以确保教育导向性的、情境意识的指导。

在来自多个编程任务的600份学生提交的评估中，系统生成的反馈与教师评论在语义上有很强的一致性。对于可视化而言，基于1000份标注提交训练的对比学习代码嵌入能够根据功能和方法将解决方案分组为有意义的集群。系统还支持提示池功能，允许教师通过选定的提示模板指导反馈风格。

通过集成AI驱动的反馈、语义聚类和互动可视化，Autograder+减少了教师的工作负担，支持更有针对性的指导并促进更强的学习成果。 

---
# A Pragmatic View of AI Personhood 

**Title (ZH)**: 一种实用视角下的AI人格论 

**Authors**: Joel Z. Leibo, Alexander Sasha Vezhnevets, William A. Cunningham, Stanley M. Bileschi  

**Link**: [PDF](https://arxiv.org/pdf/2510.26396)  

**Abstract**: The emergence of agentic Artificial Intelligence (AI) is set to trigger a "Cambrian explosion" of new kinds of personhood. This paper proposes a pragmatic framework for navigating this diversification by treating personhood not as a metaphysical property to be discovered, but as a flexible bundle of obligations (rights and responsibilities) that societies confer upon entities for a variety of reasons, especially to solve concrete governance problems. We argue that this traditional bundle can be unbundled, creating bespoke solutions for different contexts. This will allow for the creation of practical tools -- such as facilitating AI contracting by creating a target "individual" that can be sanctioned -- without needing to resolve intractable debates about an AI's consciousness or rationality. We explore how individuals fit in to social roles and discuss the use of decentralized digital identity technology, examining both "personhood as a problem", where design choices can create "dark patterns" that exploit human social heuristics, and "personhood as a solution", where conferring a bundle of obligations is necessary to ensure accountability or prevent conflict. By rejecting foundationalist quests for a single, essential definition of personhood, this paper offers a more pragmatic and flexible way to think about integrating AI agents into our society. 

**Abstract (ZH)**: 代理型人工智能的兴起将触发新型主体性的“寒武纪爆炸”：一种实用框架的提出 

---
# Scales++: Compute Efficient Evaluation Subset Selection with Cognitive Scales Embeddings 

**Title (ZH)**: Scales++: 具有认知尺度嵌入的计算高效评估子集选择 

**Authors**: Andrew M. Bean, Nabeel Seedat, Shengzhuang Chen, Jonathan Richard Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2510.26384)  

**Abstract**: The prohibitive cost of evaluating large language models (LLMs) on comprehensive benchmarks necessitates the creation of small yet representative data subsets (i.e., tiny benchmarks) that enable efficient assessment while retaining predictive fidelity. Current methods for this task operate under a model-centric paradigm, selecting benchmarking items based on the collective performance of existing models. Such approaches are limited by large upfront costs, an inability to immediately handle new benchmarks (`cold-start'), and the fragile assumption that future models will share the failure patterns of their predecessors. In this work, we challenge this paradigm and propose a item-centric approach to benchmark subset selection, arguing that selection should be based on the intrinsic properties of the task items themselves, rather than on model-specific failure patterns. We instantiate this item-centric efficient benchmarking approach via a novel method, Scales++, where data selection is based on the cognitive demands of the benchmark samples. Empirically, we show Scales++ reduces the upfront selection cost by over 18x while achieving competitive predictive fidelity. On the Open LLM Leaderboard, using just a 0.5\% data subset, we predict full benchmark scores with a 2.9% mean absolute error. We demonstrate that this item-centric approach enables more efficient model evaluation without significant fidelity degradation, while also providing better cold-start performance and more interpretable benchmarking. 

**Abstract (ZH)**: prohibitive 成本在评估大规模语言模型 (LLMs) 时，全面基准测试的数据子集（即小型基准测试）的创建是必要的，以便实现高效的评估同时保持预测保真度。当前的方法在模型为中心的范式下运行，基于现有模型的综合表现选择基准测试项目。这些方法受到大额前期成本、无法立即处理新基准（冷启动）以及未来模型将与其前任共享故障模式的脆弱假设的限制。在本工作中，我们挑战了这一范式，并提出了一种基于项目的基准测试子集选择方法，认为选择应基于任务项目的固有属性，而不是基于特定模型的故障模式。我们通过一种新颖的方法 Scales++ 实现了这种基于项目的高效基准测试方法，其中数据选择基于基准样本的认知需求。实证结果表明，Scales++ 将前期选择成本减少了超过 18 倍，同时实现了可竞争的预测保真度。在 Open LLM 领导板上，使用仅 0.5% 的数据子集，我们以 2.9% 的均方误差预测了完整的基准分数。我们证明，这种基于项目的接近方法可以在不显著牺牲保真度的情况下实现更高效的模型评估，同时提供更好的冷启动性能和更具可解释性的基准测试。 

---
# AI Mathematician as a Partner in Advancing Mathematical Discovery - A Case Study in Homogenization Theory 

**Title (ZH)**: AI数学家在推进数学发现中的合作者角色——以 homogenization 理论为例 

**Authors**: Yuanhang Liu, Beichen Wang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26380)  

**Abstract**: Artificial intelligence (AI) has demonstrated impressive progress in mathematical reasoning, yet its integration into the practice of mathematical research remains limited. In this study, we investigate how the AI Mathematician (AIM) system can operate as a research partner rather than a mere problem solver. Focusing on a challenging problem in homogenization theory, we analyze the autonomous reasoning trajectories of AIM and incorporate targeted human interventions to structure the discovery process. Through iterative decomposition of the problem into tractable subgoals, selection of appropriate analytical methods, and validation of intermediate results, we reveal how human intuition and machine computation can complement one another. This collaborative paradigm enhances the reliability, transparency, and interpretability of the resulting proofs, while retaining human oversight for formal rigor and correctness. The approach leads to a complete and verifiable proof, and more broadly, demonstrates how systematic human-AI co-reasoning can advance the frontier of mathematical discovery. 

**Abstract (ZH)**: 人工智能（AI）在数学推理方面展现了令人印象深刻的进步，但在数学研究实践中的整合仍受到限制。本研究探讨了AI数学家（AIM）系统如何作为研究伙伴而非 merely 问题解决者发挥作用。聚焦于调和理论中的一个具有挑战性的问题，我们分析了AIM的自主推理轨迹，并结合有针对性的人类干预来结构发现过程。通过将问题迭代分解为可处理的子目标、选择合适的分析方法以及验证中间结果，我们揭示了人类直觉与机器计算之间的互补作用。这种协作范式增强了所得证明的可靠性和透明度，并保持了形式严谨性和正确性的人类监督。该方法导致了一个完整且可验证的证明，更广泛地说，证明了系统的人工智能与人类共同推理如何推进数学发现的前沿。 

---
# BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning 

**Title (ZH)**: BOTS：贝叶斯在线任务选择的统一框架在LLM强化微调中 

**Authors**: Qianli Shen, Daoyuan Chen, Yilun Huang, Zhenqing Ling, Yaliang Li, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.26374)  

**Abstract**: Reinforcement finetuning (RFT) is a key technique for aligning Large Language Models (LLMs) with human preferences and enhancing reasoning, yet its effectiveness is highly sensitive to which tasks are explored during training. Uniform task sampling is inefficient, wasting computation on tasks that are either trivial or unsolvable, while existing task selection methods often suffer from high rollout costs, poor adaptivity, or incomplete evidence. We introduce \textbf{BOTS}, a unified framework for \textbf{B}ayesian \textbf{O}nline \textbf{T}ask \textbf{S}election in LLM reinforcement finetuning. Grounded in Bayesian inference, BOTS adaptively maintains posterior estimates of task difficulty as the model evolves. It jointly incorporates \emph{explicit evidence} from direct evaluations of selected tasks and \emph{implicit evidence} inferred from these evaluations for unselected tasks, with Thompson sampling ensuring a principled balance between exploration and exploitation. To make implicit evidence practical, we instantiate it with an ultra-light interpolation-based plug-in that estimates difficulties of unevaluated tasks without extra rollouts, adding negligible overhead. Empirically, across diverse domains and LLM scales, BOTS consistently improves data efficiency and performance over baselines and ablations, providing a practical and extensible solution for dynamic task selection in RFT. 

**Abstract (ZH)**: Bayesian Online Task Selection for Reinforcement Fine-tuning of Large Language Models 

---
# Discovering State Equivalences in UCT Search Trees By Action Pruning 

**Title (ZH)**: 在UCT搜索树中通过动作裁剪发现状态等价性 

**Authors**: Robin Schmöcker, Alexander Dockhorn, Bodo Rosenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2510.26346)  

**Abstract**: One approach to enhance Monte Carlo Tree Search (MCTS) is to improve its sample efficiency by grouping/abstracting states or state-action pairs and sharing statistics within a group. Though state-action pair abstractions are mostly easy to find in algorithms such as On the Go Abstractions in Upper Confidence bounds applied to Trees (OGA-UCT), nearly no state abstractions are found in either noisy or large action space settings due to constraining conditions. We provide theoretical and empirical evidence for this claim, and we slightly alleviate this state abstraction problem by proposing a weaker state abstraction condition that trades a minor loss in accuracy for finding many more abstractions. We name this technique Ideal Pruning Abstractions in UCT (IPA-UCT), which outperforms OGA-UCT (and any of its derivatives) across a large range of test domains and iteration budgets as experimentally validated. IPA-UCT uses a different abstraction framework from Abstraction of State-Action Pairs (ASAP) which is the one used by OGA-UCT, which we name IPA. Furthermore, we show that both IPA and ASAP are special cases of a more general framework that we call p-ASAP which itself is a special case of the ASASAP framework. 

**Abstract (ZH)**: 一种提高蒙特卡洛树搜索(MCTS)采样效率的方法是通过分组/抽象状态或状态-动作对并在组内共享统计信息来实现，我们通过更弱的状态抽象条件提出了一种技术，该条件在准确性略有损失的情况下，可以找到更多的抽象，该技术名为理想剪枝抽象在UCT中的应用（IPA-UCT），其在多个测试领域和迭代预算范围内表现优于OGA-UCT及其衍生算法。此外，我们展示了IPA和ASAP都是更一般框架p-ASAP的特例，而p-ASAP又是ASASAP框架的特例。 

---
# GraphCompliance: Aligning Policy and Context Graphs for LLM-Based Regulatory Compliance 

**Title (ZH)**: GraphCompliance：规管合规的政策与上下文图谱对齐 

**Authors**: Jiseong Chung, Ronny Ko, Wonchul Yoo, Makoto Onizuka, Sungmok Kim, Tae-Wan Kim, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.26309)  

**Abstract**: Compliance at web scale poses practical challenges: each request may require a regulatory assessment. Regulatory texts (e.g., the General Data Protection Regulation, GDPR) are cross-referential and normative, while runtime contexts are expressed in unstructured natural language. This setting motivates us to align semantic information in unstructured text with the structured, normative elements of regulations. To this end, we introduce GraphCompliance, a framework that represents regulatory texts as a Policy Graph and runtime contexts as a Context Graph, and aligns them. In this formulation, the policy graph encodes normative structure and cross-references, whereas the context graph formalizes events as subject-action-object (SAO) and entity-relation triples. This alignment anchors the reasoning of a judge large language model (LLM) in structured information and helps reduce the burden of regulatory interpretation and event parsing, enabling a focus on the core reasoning step. In experiments on 300 GDPR-derived real-world scenarios spanning five evaluation tasks, GraphCompliance yields 4.1-7.2 percentage points (pp) higher micro-F1 than LLM-only and RAG baselines, with fewer under- and over-predictions, resulting in higher recall and lower false positive rates. Ablation studies indicate contributions from each graph component, suggesting that structured representations and a judge LLM are complementary for normative reasoning. 

**Abstract (ZH)**: 面向网页规模的合规性面临实践挑战：每次请求可能需要进行监管评估。监管文本（例如通用数据保护条例，GDPR）是相互引用和规范性的，而运行时上下文则以非结构化自然语言表达。这种设置促使我们通过图谱对齐非结构化文本中的语义信息与监管中的结构化规范元素进行对齐。为此，我们引入了GraphCompliance框架，该框架将监管文本表示为策略图，将运行时上下文表示为情境图，并对齐它们。在这一表示中，策略图编码规范结构和相互引用，而情境图形式化事件为主体-动作-客体（SAO）和实体-关系三元组。这种对齐为大型语言模型（LLM）裁判的推理锚定了结构化信息，有助于减少监管解释和事件解析的负担，使重点放在核心推理步骤上。在涵盖五个评估任务的300个源自GDPR的真实世界场景实验中，GraphCompliance在LLM仅模型和RAG基线上分别获得了4.1-7.2个百分点更高的微F1值，预测更少的误报和漏报，从而提高了召回率并降低了假阳性率。消融研究表明，每个图组件的贡献，表明结构化表示和裁判LLM对规范推理具有互补性。 

---
# Graph-Enhanced Policy Optimization in LLM Agent Training 

**Title (ZH)**: 图增强策略优化在大语言模型代理训练中 

**Authors**: Jiazhen Yuan, Wei Zhao, Zhengbiao Bai  

**Link**: [PDF](https://arxiv.org/pdf/2510.26270)  

**Abstract**: Group based reinforcement learning (RL) has shown impressive results on complex reasoning and mathematical tasks. Yet, when applied to train multi-turn, interactive LLM agents, these methods often suffer from structural blindness-the inability to exploit the underlying connectivity of the environment. This manifests in three critical challenges: (1) inefficient, unguided exploration, (2) imprecise credit assignment due to overlooking pivotal states, and (3) myopic planning caused by static reward discounting. We address these issues with Graph-Enhanced Policy Optimization (GEPO), which dynamically constructs a state-transition graph from agent experience and employs graph-theoretic centrality to provide three synergistic learning signals: (1)structured intrinsic rewards that guide exploration toward high-impact states, (2) a graph-enhanced advantage function for topology-aware credit assignment, and (3) a dynamic discount factor adapted to each state's strategic value. On the ALFWorld, WebShop, and a proprietary Workbench benchmarks, GEPO demonstrates strong performance, achieving absolute success rate gains of +4.1%, +5.3%, and +10.9% over competitive baselines. These results highlight that explicitly modeling environmental structure is a robust, generalizable strategy for advancing LLM agent training. 

**Abstract (ZH)**: 基于图增强的策略优化在多轮交互大模型代理训练中的应用 

---
# Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles 

**Title (ZH)**: 基于检索增强生成的分布式大规模语言模型代理用于具有应急车辆的通用交通信号控制 

**Authors**: Xinhang Li, Qing Guo, Junyu Chen, Zheng Guo, Shengzhe Xu, Lei Li, Lin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26242)  

**Abstract**: With increasing urban traffic complexity, Traffic Signal Control (TSC) is essential for optimizing traffic flow and improving road safety. Large Language Models (LLMs) emerge as promising approaches for TSC. However, they are prone to hallucinations in emergencies, leading to unreliable decisions that may cause substantial delays for emergency vehicles. Moreover, diverse intersection types present substantial challenges for traffic state encoding and cross-intersection training, limiting generalization across heterogeneous intersections. Therefore, this paper proposes Retrieval Augmented Generation (RAG)-enhanced distributed LLM agents with Emergency response for Generalizable TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning framework, which dynamically adjusts reasoning depth based on the emergency scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to distill specific knowledge and guidance from historical cases, enhancing the reliability and rationality of agents' emergency decisions. Secondly, this paper designs a type-agnostic traffic representation and proposes a Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3 adaptively samples training experience from diverse intersections with environment feedback-based priority and fine-tunes LLM agents with a designed reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies across heterogeneous intersections. On three real-world road networks with 17 to 177 heterogeneous intersections, extensive experiments show that REG-TSC reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle waiting time by 83.16%, outperforming other state-of-the-art methods. 

**Abstract (ZH)**: 基于检索增强生成的面向应急响应的通用交通信号控制（REG-TSC） 

---
# Questionnaire meets LLM: A Benchmark and Empirical Study of Structural Skills for Understanding Questions and Responses 

**Title (ZH)**: 问卷调查遇见LLM：对理解问题和回应的结构性技能的基准研究与实证分析 

**Authors**: Duc-Hai Nguyen, Vijayakumar Nanjappan, Barry O'Sullivan, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26238)  

**Abstract**: Millions of people take surveys every day, from market polls and academic studies to medical questionnaires and customer feedback forms. These datasets capture valuable insights, but their scale and structure present a unique challenge for large language models (LLMs), which otherwise excel at few-shot reasoning over open-ended text. Yet, their ability to process questionnaire data or lists of questions crossed with hundreds of respondent rows remains underexplored. Current retrieval and survey analysis tools (e.g., Qualtrics, SPSS, REDCap) are typically designed for humans in the workflow, limiting such data integration with LLM and AI-empowered automation. This gap leaves scientists, surveyors, and everyday users without evidence-based guidance on how to best represent questionnaires for LLM consumption. We address this by introducing QASU (Questionnaire Analysis and Structural Understanding), a benchmark that probes six structural skills, including answer lookup, respondent count, and multi-hop inference, across six serialization formats and multiple prompt strategies. Experiments on contemporary LLMs show that choosing an effective format and prompt combination can improve accuracy by up to 8.8% points compared to suboptimal formats. For specific tasks, carefully adding a lightweight structural hint through self-augmented prompting can yield further improvements of 3-4% points on average. By systematically isolating format and prompting effects, our open source benchmark offers a simple yet versatile foundation for advancing both research and real-world practice in LLM-based questionnaire analysis. 

**Abstract (ZH)**: 大规模语言模型在问卷分析与结构理解中的挑战与机遇：QASU基准评测 

---
# One Model to Critique Them All: Rewarding Agentic Tool-Use via Efficient Reasoning 

**Title (ZH)**: 一个模型 critique 所有模型：通过高效推理奖励自主工具使用 

**Authors**: Renhao Li, Jianhong Tu, Yang Su, Hamid Alinejad-Rokny, Derek F. Wong, Junyang Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26167)  

**Abstract**: Reward models (RMs) play a critical role in aligning large language models (LLMs) with human preferences. Yet in the domain of tool learning, the lack of RMs specifically designed for function-calling tasks has limited progress toward more capable agentic AI. We introduce ToolRM, a family of lightweight generative RMs tailored for general tool-use scenarios. To build these models, we propose a novel pipeline that constructs pairwise preference data using rule-based scoring and multidimensional sampling. This yields ToolPref-Pairwise-30K, a diverse, balanced, and challenging dataset of critique tasks that supports reinforcement learning with verifiable feedback. To evaluate tool-use RMs, we also introduce TRBench$_{BFCL}$, a benchmark built on the agentic evaluation suite BFCL. Trained on our constructed data, models from the Qwen3-4B/8B series achieve up to 14.28% higher accuracy, substantially outperforming frontier models such as Claude 4 and OpenAI o3 in pairwise reward judgments. Beyond training objectives, ToolRM generalizes to broader critique tasks, including Best-of-N sampling and self-correction. Experiments on ACEBench highlight its effectiveness and efficiency, enabling inference-time scaling and reducing output token usage by over 66%. We release data and model checkpoints to facilitate future research. 

**Abstract (ZH)**: 工具RMs在工具学习领域中对于对齐大型语言模型与人类偏好发挥着关键作用。然而，在工具学习领域缺乏专门设计的功能调用任务RMs限制了更强大自主AI的发展。我们引入了ToolRM，这是一种针对通用工具使用场景定制的轻量级生成性RMs。为了构建这些模型，我们提出了一个新的流水线，使用基于规则的评分和多维采样来构建成对偏好数据。这产生了ToolPref-Pairwise-30K，一个多样、平衡且具有挑战性的批判任务数据集，支持具有可验证反馈的强化学习。为了评估工具使用RMs，我们还引入了基于BFCL自主评估套件构建的TRBench$_{BFCL}$基准测试。在我们构建的数据上训练的Qwen3-4B/8B系列模型在成对奖励判断中最高准确率提高了14.28%，显著优于诸如Claude 4和OpenAI o3等前沿模型。除了训练目标外，ToolRM还能泛化到更广泛的批判任务，包括Best-of-N采样和自我纠正。ACEBench上的实验突显了其有效性和效率，使其能够在推理时扩展并减少输出标记使用量超过66%。我们发布了数据和模型检查点，以促进未来的研究。 

---
# The FM Agent 

**Title (ZH)**: FM代理 

**Authors**: Annan Li, Chufan Wu, Zengle Ge, Yee Hin Chong, Zhinan Hou, Lizhe Cao, Cheng Ju, Jianmin Wu, Huaiming Li, Haobo Zhang, Shenghao Feng, Mo Zhao, Fengzhi Qiu, Rui Yang, Mengmeng Zhang, Wenyi Zhu, Yingying Sun, Quan Sun, Shunhao Yan, Danyu Liu, Dawei Yin, Dou Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26144)  

**Abstract**: Large language models (LLMs) are catalyzing the development of autonomous AI research agents for scientific and engineering discovery. We present FM Agent, a novel and general-purpose multi-agent framework that leverages a synergistic combination of LLM-based reasoning and large-scale evolutionary search to address complex real-world challenges. The core of FM Agent integrates several key innovations: 1) a cold-start initialization phase incorporating expert guidance, 2) a novel evolutionary sampling strategy for iterative optimization, 3) domain-specific evaluators that combine correctness, effectiveness, and LLM-supervised feedback, and 4) a distributed, asynchronous execution infrastructure built on Ray. Demonstrating broad applicability, our system has been evaluated across diverse domains, including operations research, machine learning, GPU kernel optimization, and classical mathematical problems. FM Agent reaches state-of-the-art results autonomously, without human interpretation or tuning -- 1976.3 on ALE-Bench (+5.2\%), 43.56\% on MLE-Bench (+4.0pp), up to 20x speedups on KernelBench, and establishes new state-of-the-art(SOTA) results on several classical mathematical problems. Beyond academic benchmarks, FM Agent shows considerable promise for both large-scale enterprise R\&D workflows and fundamental scientific research, where it can accelerate innovation, automate complex discovery processes, and deliver substantial engineering and scientific advances with broader societal impact. 

**Abstract (ZH)**: 大型语言模型（LLMs）正催化自主AI研究代理在科学与工程发现中的发展。我们提出FM代理，这是一种新颖的一般用途多代理框架，结合了基于LLM的推理和大规模进化搜索，以解决复杂的真实世界挑战。FM代理的核心集成了多项关键技术创新：1）带有专家指导的冷启动初始化阶段，2）一种新型的进化采样策略用于迭代优化，3）特定领域的评估器结合了正确性、有效性以及LLM监督反馈，以及4）基于Ray构建的分布式异步执行基础设施。展示出广泛适用性，我们的系统已在包括运筹学、机器学习、GPU内核优化和经典数学问题等多个领域进行了评估。FM代理能够自主达到最先进的结果，无需人类解释或调整——在ALE-Bench上得分为1976.3（+5.2%），MLE-Bench得分为43.56%（性能提升4.0个百分点），在KernelBench上最快可达20倍的加速，并在若干经典数学问题上建立了新的最先进的结果。除了学术基准，FM代理在大规模企业研发工作流和基础科学研究中展现出巨大的潜力，能够加速创新、自动化复杂发现过程，并带来广泛的工程和科学进步，具有更广泛的社会效益。 

---
# Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math 

**Title (ZH)**: 数学推理课程：从数学中bootstrap广域LLM推理 

**Authors**: Bo Pang, Deqian Kong, Silvio Savarese, Caiming Xiong, Yingbo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.26143)  

**Abstract**: Reinforcement learning (RL) can elicit strong reasoning in large language models (LLMs), yet most open efforts focus on math and code. We propose Reasoning Curriculum, a simple two-stage curriculum that first elicits reasoning skills in pretraining-aligned domains such as math, then adapts and refines these skills across other domains via joint RL. Stage 1 performs a brief cold start and then math-only RL with verifiable rewards to develop reasoning skills. Stage 2 runs joint RL on mixed-domain data to transfer and consolidate these skills. The curriculum is minimal and backbone-agnostic, requiring no specialized reward models beyond standard verifiability checks. Evaluated on Qwen3-4B and Llama-3.1-8B over a multi-domain suite, reasoning curriculum yields consistent gains. Ablations and a cognitive-skill analysis indicate that both stages are necessary and that math-first elicitation increases cognitive behaviors important for solving complex problems. Reasoning Curriculum provides a compact, easy-to-adopt recipe for general reasoning. 

**Abstract (ZH)**: 强化学习（RL）可以在大型语言模型（LLMs）中激发强烈的推理能力，但大多数开源努力主要集中在数学和代码上。我们提出了一种简单的两阶段课程：首先在与预训练对齐的领域（如数学）中激发推理技能，然后通过联合RL跨其他领域进行适应和精炼。第一阶段进行简短的冷启动和仅数学的RL，使用可验证的奖励来发展推理技能。第二阶段在混合领域数据上运行联合RL以转移和巩固这些技能。该课程简单且与基础模型无关，只需要标准的可验证性检查之外的无特殊奖励模型。在Qwen3-4B和Llama-3.1-8B上对多领域套件进行评估表明，推理课程可以带来一致的收益。消融实验和认知能力分析表明，两个阶段都是必要的，数学优先的激发可以增加解决复杂问题所需的重要认知行为。推理课程提供了一种紧凑且易于采用的一般推理方法。 

---
# Beyond Benchmarks: The Economics of AI Inference 

**Title (ZH)**: 超越基准：AI 推理的经济学 

**Authors**: Boqin Zhuang, Jiacheng Qiao, Mingqian Liu, Mingxing Yu, Ping Hong, Rui Li, Xiaoxia Song, Xiangjun Xu, Xu Chen, Yaoyao Ma, Yujie Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.26136)  

**Abstract**: The inference cost of Large Language Models (LLMs) has become a critical factor in determining their commercial viability and widespread adoption. This paper introduces a quantitative ``economics of inference'' framework, treating the LLM inference process as a compute-driven intelligent production activity. We analyze its marginal cost, economies of scale, and quality of output under various performance configurations. Based on empirical data from WiNEval-3.0, we construct the first ``LLM Inference Production Frontier,'' revealing three principles: diminishing marginal cost, diminishing returns to scale, and an optimal cost-effectiveness zone. This paper not only provides an economic basis for model deployment decisions but also lays an empirical foundation for the future market-based pricing and optimization of AI inference resources. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理成本已成为决定其商业可行性和广泛采用的关键因素。本文引入了一个定量的“推理经济学”框架，将LLM的推理过程视为基于计算的智能生产活动。我们分析了在各种性能配置下其边际成本、规模经济以及输出质量。基于WiNEval-3.0的实证数据，我们构建了首个“LLM推理生产前沿”，揭示了三个原则：边际成本递减、规模报酬递减以及最优成本效益区。本文不仅为模型部署决策提供了经济基础，还为未来基于市场的AI推理资源定价与优化奠定了实证基础。 

---
# GUI Knowledge Bench: Revealing the Knowledge Gap Behind VLM Failures in GUI Tasks 

**Title (ZH)**: GUI Knowledge Bench: 揭示GUI任务中VLM失败背后的知识差距 

**Authors**: Chenrui Shi, Zedong Yu, Zhi Gao, Ruining Feng, Enqi Liu, Yuwei Wu, Yunde Jia, Liuyu Xiang, Zhaofeng He, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.26098)  

**Abstract**: Large vision language models (VLMs) have advanced graphical user interface (GUI) task automation but still lag behind humans. We hypothesize this gap stems from missing core GUI knowledge, which existing training schemes (such as supervised fine tuning and reinforcement learning) alone cannot fully address. By analyzing common failure patterns in GUI task execution, we distill GUI knowledge into three dimensions: (1) interface perception, knowledge about recognizing widgets and system states; (2) interaction prediction, knowledge about reasoning action state transitions; and (3) instruction understanding, knowledge about planning, verifying, and assessing task completion progress. We further introduce GUI Knowledge Bench, a benchmark with multiple choice and yes/no questions across six platforms (Web, Android, MacOS, Windows, Linux, IOS) and 292 applications. Our evaluation shows that current VLMs identify widget functions but struggle with perceiving system states, predicting actions, and verifying task completion. Experiments on real world GUI tasks further validate the close link between GUI knowledge and task success. By providing a structured framework for assessing GUI knowledge, our work supports the selection of VLMs with greater potential prior to downstream training and provides insights for building more capable GUI agents. 

**Abstract (ZH)**: 大型视觉语言模型（VLMs）在自动化图形用户界面（GUI）任务方面取得了进展，但仍落后于人类。我们认为这一差距源于缺失的核心GUI知识，现有的训练方案（如监督微调和强化学习）无法充分解决这一问题。通过分析GUI任务执行中的常见失败模式，我们将GUI知识归纳为三个维度：（1）界面感知，关于识别控件和系统状态的知识；（2）交互预测，关于推理动作状态转换的知识；以及（3）指令理解，关于规划、验证和评估任务完成进度的知识。我们进一步引入了GUI Knowledge Bench基准测试，该基准测试包含跨六种平台（Web、Android、MacOS、Windows、Linux、iOS）和292个应用程序的多项选择和是非问题。我们的评估结果显示，当前的VLMs能够识别控件功能，但在感知系统状态、预测动作和验证任务完成方面存在困难。针对真实世界GUI任务的实验进一步验证了GUI知识与任务成功之间的密切联系。通过为评估GUI知识提供结构化框架，我们的工作支持在下游训练前选择具有更大潜力的VLMs，并为构建更强大的GUI代理提供洞见。 

---
# Lean4Physics: Comprehensive Reasoning Framework for College-level Physics in Lean4 

**Title (ZH)**: Lean4Physics：面向大学水平物理学的Lean4综合推理框架 

**Authors**: Yuxin Li, Minghao Liu, Ruida Wang, Wenzhao Ji, Zhitao He, Rui Pan, Junming Huang, Tong Zhang, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2510.26094)  

**Abstract**: We present **Lean4PHYS**, a comprehensive reasoning framework for college-level physics problems in Lean4. **Lean4PHYS** includes *LeanPhysBench*, a college-level benchmark for formal physics reasoning in Lean4, which contains 200 hand-crafted and peer-reviewed statements derived from university textbooks and physics competition problems. To establish a solid foundation for formal reasoning in physics, we also introduce *PhysLib*, a community-driven repository containing fundamental unit systems and theorems essential for formal physics reasoning. Based on the benchmark and Lean4 repository we composed in **Lean4PHYS**, we report baseline results using major expert Math Lean4 provers and state-of-the-art closed-source models, with the best performance of DeepSeek-Prover-V2-7B achieving only 16% and Claude-Sonnet-4 achieving 35%. We also conduct a detailed analysis showing that our *PhysLib* can achieve an average improvement of 11.75% in model performance. This demonstrates the challenging nature of our *LeanPhysBench* and the effectiveness of *PhysLib*. To the best of our knowledge, this is the first study to provide a physics benchmark in Lean4. 

**Abstract (ZH)**: Lean4PHYS：Lean4中针对大学物理问题的综合推理框架 

---
# Can AI be Accountable? 

**Title (ZH)**: AI能负责任吗？ 

**Authors**: Andrew L. Kun  

**Link**: [PDF](https://arxiv.org/pdf/2510.26057)  

**Abstract**: The AI we use is powerful, and its power is increasing rapidly. If this powerful AI is to serve the needs of consumers, voters, and decision makers, then it is imperative that the AI is accountable. In general, an agent is accountable to a forum if the forum can request information from the agent about its actions, if the forum and the agent can discuss this information, and if the forum can sanction the agent. Unfortunately, in too many cases today's AI is not accountable -- we cannot question it, enter into a discussion with it, let alone sanction it. In this chapter we relate the general definition of accountability to AI, we illustrate what it means for AI to be accountable and unaccountable, and we explore approaches that can improve our chances of living in a world where all AI is accountable to those who are affected by it. 

**Abstract (ZH)**: 我们使用的AI非常强大，其能力正在快速增加。如果要让这种强大的AI服务于消费者、选民和决策者的需求，那么AI必须具有问责性。一般来说，如果一个代理可以向某个论坛提供其行为的信息，该论坛可以与代理讨论这些信息，并可以对代理进行制裁，那么该代理就是对这个论坛问责的。不幸的是，在今天太多情况下，AI并不具有问责性——我们不能质疑它，甚至不能与其进行讨论，更不用说对其进行制裁了。在本章中，我们将问责性的一般定义应用于AI，阐述AI具有和不具有问责性的含义，并探讨可以提高所有受影响的AI都对其问责的机会的方法。 

---
# Large Language Model-assisted Autonomous Vehicle Recovery from Immobilization 

**Title (ZH)**: 大型语言模型辅助的自动驾驶车辆脱困技术 

**Authors**: Zhipeng Bao, Qianwen Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.26023)  

**Abstract**: Despite significant advancements in recent decades, autonomous vehicles (AVs) continue to face challenges in navigating certain traffic scenarios where human drivers excel. In such situations, AVs often become immobilized, disrupting overall traffic flow. Current recovery solutions, such as remote intervention (which is costly and inefficient) and manual takeover (which excludes non-drivers and limits AV accessibility), are inadequate. This paper introduces StuckSolver, a novel Large Language Model (LLM) driven recovery framework that enables AVs to resolve immobilization scenarios through self-reasoning and/or passenger-guided decision-making. StuckSolver is designed as a plug-in add-on module that operates on top of the AV's existing perception-planning-control stack, requiring no modification to its internal architecture. Instead, it interfaces with standard sensor data streams to detect immobilization states, interpret environmental context, and generate high-level recovery commands that can be executed by the AV's native planner. We evaluate StuckSolver on the Bench2Drive benchmark and in custom-designed uncertainty scenarios. Results show that StuckSolver achieves near-state-of-the-art performance through autonomous self-reasoning alone and exhibits further improvements when passenger guidance is incorporated. 

**Abstract (ZH)**: 尽管在近几十年取得了显著进展，自动驾驶车辆（AVs）在某些交通场景中仍面临挑战，这些场景是人类驾驶员所擅长的。在这种情况下，AVs经常变得无法行动，从而破坏整体交通流动。当前的恢复解决方案，如远程干预（成本高且效率低）和手动接管（排除非驾驶员且限制AV的可达性），都不足。本文提出了一种名为StuckSolver的新型大型语言模型（LLM）驱动恢复框架，使AV能够通过自我推理和/or乘客引导决策来解决无法行动的场景。StuckSolver设计为插件附加模块，运行在AV现有感知-规划-控制栈之上，不需要修改其内部架构。相反，它与标准传感器数据流接口，以检测无法行动状态，解释环境上下文，并生成可由AV内置规划器执行的高层恢复命令。我们在Bench2Drive基准测试和自定义设计的不确定性场景中评估了StuckSolver。结果显示，StuckSolver仅通过自主自我推理即可实现接近最先进的性能，并且在结合乘客引导时表现出进一步的改进。 

---
# AutoSurvey2: Empowering Researchers with Next Level Automated Literature Surveys 

**Title (ZH)**: AutoSurvey2：为研究人员提供高级自动化文献综述支持 

**Authors**: Siyi Wu, Chiaxin Liang, Ziqian Bi, Leyi Zhao, Tianyang Wang, Junhao Song, Yichao Zhang, Keyu Chen, Xinyuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.26012)  

**Abstract**: The rapid growth of research literature, particularly in large language models (LLMs), has made producing comprehensive and current survey papers increasingly difficult. This paper introduces autosurvey2, a multi-stage pipeline that automates survey generation through retrieval-augmented synthesis and structured evaluation. The system integrates parallel section generation, iterative refinement, and real-time retrieval of recent publications to ensure both topical completeness and factual accuracy. Quality is assessed using a multi-LLM evaluation framework that measures coverage, structure, and relevance in alignment with expert review standards. Experimental results demonstrate that autosurvey2 consistently outperforms existing retrieval-based and automated baselines, achieving higher scores in structural coherence and topical relevance while maintaining strong citation fidelity. By combining retrieval, reasoning, and automated evaluation into a unified framework, autosurvey2 provides a scalable and reproducible solution for generating long-form academic surveys and contributes a solid foundation for future research on automated scholarly writing. All code and resources are available at this https URL. 

**Abstract (ZH)**: The rapid增长的Research文献，尤其是在大规模语言模型(LLMs)领域的快速增长，使得撰写综合性和时效性强的综述论文越来越具挑战性。本文介绍了autosurvey2，这是一种多阶段流水线，通过检索增强的合成和结构化评估自动化综述生成。该系统结合了并行部分生成、迭代 refinement 和实时检索近期出版物，以确保主题完整性和事实准确性。质量通过一个基于多大规模语言模型的评估框架来评估，该框架在专家评审标准的指导下衡量覆盖面、结构和相关性。实验结果表明，autosurvey2一致地优于现有的基于检索和自动化的基线方法，在结构连贯性和主题相关性方面取得更高的分数，同时保持较强的引文准确性。通过将检索、推理和自动化评估统一到一个框架中，autosurvey2提供了生成长格式学术综述的可扩展和可重复解决方案，并为未来自动学术写作研究奠定了坚实基础。所有代码和资源均可在该网址获取。 

---
# From Queries to Insights: Agentic LLM Pipelines for Spatio-Temporal Text-to-SQL 

**Title (ZH)**: 从查询到洞察：用于空时文本到SQL的能动LLM流水线 

**Authors**: Manu Redd, Tao Zhe, Dongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.25997)  

**Abstract**: Natural-language-to-SQL (NL-to-SQL) systems hold promise for democratizing access to structured data, allowing users to query databases without learning SQL. Yet existing systems struggle with realistic spatio-temporal queries, where success requires aligning vague user phrasing with schema-specific categories, handling temporal reasoning, and choosing appropriate outputs. We present an agentic pipeline that extends a naive text-to-SQL baseline (llama-3-sqlcoder-8b) with orchestration by a Mistral-based ReAct agent. The agent can plan, decompose, and adapt queries through schema inspection, SQL generation, execution, and visualization tools. We evaluate on 35 natural-language queries over the NYC and Tokyo check-in dataset, covering spatial, temporal, and multi-dataset reasoning. The agent achieves substantially higher accuracy than the naive baseline 91.4% vs. 28.6% and enhances usability through maps, plots, and structured natural-language summaries. Crucially, our design enables more natural human-database interaction, supporting users who lack SQL expertise, detailed schema knowledge, or prompting skill. We conclude that agentic orchestration, rather than stronger SQL generators alone, is a promising foundation for interactive geospatial assistants. 

**Abstract (ZH)**: 基于文本到SQL的代理管道在自然语言到SQL系统中的应用：增强地理空间查询能力 

---
# Estimating cognitive biases with attention-aware inverse planning 

**Title (ZH)**: 基于注意力感知逆规划的认知偏差估计 

**Authors**: Sounak Banerjee, Daphne Cornelisse, Deepak Gopinath, Emily Sumner, Jonathan DeCastro, Guy Rosman, Eugene Vinitsky, Mark K. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2510.25951)  

**Abstract**: People's goal-directed behaviors are influenced by their cognitive biases, and autonomous systems that interact with people should be aware of this. For example, people's attention to objects in their environment will be biased in a way that systematically affects how they perform everyday tasks such as driving to work. Here, building on recent work in computational cognitive science, we formally articulate the attention-aware inverse planning problem, in which the goal is to estimate a person's attentional biases from their actions. We demonstrate how attention-aware inverse planning systematically differs from standard inverse reinforcement learning and how cognitive biases can be inferred from behavior. Finally, we present an approach to attention-aware inverse planning that combines deep reinforcement learning with computational cognitive modeling. We use this approach to infer the attentional strategies of RL agents in real-life driving scenarios selected from the Waymo Open Dataset, demonstrating the scalability of estimating cognitive biases with attention-aware inverse planning. 

**Abstract (ZH)**: 人们的定向行为受到认知偏差的影响，与人的交互自主系统应予关注。例如，人们的环境注意力分配会系统性地影响他们日常工作如开车上班的表现。在此基础上，我们借鉴计算认知科学的最新成果，正式阐明了注意力感知逆规划问题，目标是从人的行为中估计其注意力偏差。我们展示了注意力感知逆规划如何系统性地不同于标准逆强化学习，并说明可以通过行为推断认知偏差。最后，我们提出了一种结合深度强化学习与计算认知建模的注意力感知逆规划方法。我们使用此方法从Waymo开放数据集中选择的真实驾驶场景中推断强化学习代理的注意力策略，展示了注意力感知逆规划估计认知偏差的可扩展性。 

---
# Humains-Junior: A 3.8B Language Model Achieving GPT-4o-Level Factual Accuracy by Directed Exoskeleton Reasoning 

**Title (ZH)**: Humains-Junior: 通过定向外骨骼推理实现GPT-4级事实准确性的380亿参数语言模型 

**Authors**: Nissan Yaron, Dan Bystritsky, Ben-Etzion Yaron  

**Link**: [PDF](https://arxiv.org/pdf/2510.25933)  

**Abstract**: We introduce Humans-Junior, a 3.8B model that matches GPT-4o on the FACTS Grounding public subset within a $\pm 5$ pp equivalence margin.
Results. On Q1--Q500 under identical judges, GPT-4o scores 73.5% (95% CI 69.5--77.2) and Humans-Junior 72.7% (95% CI 68.7--76.5); the paired difference is 0.8 pp (bootstrap 95% CI $-3.1$ to $+4.7$; permutation $p = 0.72$; Cohen's $d = 0.023$). TOST establishes equivalence at $\pm 5$ pp (not at $\pm 3$ pp). When purchased as managed APIs, Humans-Junior's base model (Phi-3.5-mini-instruct) is $\approx 19\times$ less expensive than GPT-4o on Microsoft AI Foundry pricing; self-hosted or edge deployments can drive incremental inference cost toward zero. Measured vs estimated pricing sources are tabulated in Appendix E.
Method. Our approach combines minimal directed "Exoskeleton Reasoning" scaffolds with behavioral fine-tuning that teaches protocol compliance (epistemic discipline) rather than domain answers. Fine-tuning alone adds little; combined, they synergize (+17.7 pp, $p < 0.001$) and reduce variance ($\approx 25\%$). In prompt-only settings on frontier models (Q1--Q100; non-comparable), directed reasoning improved GPT-4o by +11.8 pp to 85.3% and Gemini-2.5-Pro by +5.0 pp to 93.3% (baseline 88.3%, $n = 100$); see Section~5.
TL;DR. A 3.8B model achieves GPT-4o-level FACTS accuracy (equivalent within $\pm 5$ pp on Q1--Q500). Cloud pricing shows $\approx 19\times$ lower cost versus GPT-4o, and self-hosted/edge deployments can approach zero marginal cost. Pricing sources are listed in Appendix E. Frontier prompt-only gains (Q1--Q100; non-comparable) and optimized-prompt exploratory results under earlier judges are summarized in Appendix F.
Keywords: Small Language Models, Factual Grounding, Directed Reasoning, Fine-Tuning, Model Alignment, Cost-Efficient AI 

**Abstract (ZH)**: 人类Junior：一个38亿参数模型，在FACTS Grounding公共子集上的表现与GPT-4o相媲美，误差在±5个百分点以内。 

---
# FinOps Agent -- A Use-Case for IT Infrastructure and Cost Optimization 

**Title (ZH)**: FinOps代理——IT基础设施和成本优化的应用案例 

**Authors**: Ngoc Phuoc An Vo, Manish Kesarwani, Ruchi Mahindru, Chandrasekhar Narayanaswami  

**Link**: [PDF](https://arxiv.org/pdf/2510.25914)  

**Abstract**: FinOps (Finance + Operations) represents an operational framework and cultural practice which maximizes cloud business value through collaborative financial accountability across engineering, finance, and business teams. FinOps practitioners face a fundamental challenge: billing data arrives in heterogeneous formats, taxonomies, and metrics from multiple cloud providers and internal systems which eventually lead to synthesizing actionable insights, and making time-sensitive decisions. To address this challenge, we propose leveraging autonomous, goal-driven AI agents for FinOps automation. In this paper, we built a FinOps agent for a typical use-case for IT infrastructure and cost optimization. We built a system simulating a realistic end-to-end industry process starting with retrieving data from various sources to consolidating and analyzing the data to generate recommendations for optimization. We defined a set of metrics to evaluate our agent using several open-source and close-source language models and it shows that the agent was able to understand, plan, and execute tasks as well as an actual FinOps practitioner. 

**Abstract (ZH)**: FinOps（ finance + operations）代表一种通过跨工程、财务和业务团队协作财务问责制来最大化云业务价值的操作框架和文化实践。FinOps从业者面临一项基本挑战：来自多个云提供商和内部系统的计费数据以异构格式、分类法和指标形式到达，最终导致综合可操作的洞察并作出及时决策。为应对这一挑战，我们提出利用自主的目标驱动AI代理进行FinOps自动化。在本文中，我们构建了一个用于典型IT基础设施和成本优化的FinOps代理。我们构建了一个模拟真实端到端行业过程的系统，从从各种来源获取数据开始，到集中和分析数据以生成优化建议。我们定义了一组指标来评估我们的代理，使用多个开源和闭源语言模型表明，该代理能够理解、规划和执行任务，与实际的FinOps从业者相当。 

---
# SciTrust 2.0: A Comprehensive Framework for Evaluating Trustworthiness of Large Language Models in Scientific Applications 

**Title (ZH)**: SciTrust 2.0：科学应用中大型语言模型可信性评估的综合框架 

**Authors**: Emily Herron, Junqi Yin, Feiyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.25908)  

**Abstract**: Large language models (LLMs) have demonstrated transformative potential in scientific research, yet their deployment in high-stakes contexts raises significant trustworthiness concerns. Here, we introduce SciTrust 2.0, a comprehensive framework for evaluating LLM trustworthiness in scientific applications across four dimensions: truthfulness, adversarial robustness, scientific safety, and scientific ethics. Our framework incorporates novel, open-ended truthfulness benchmarks developed through a verified reflection-tuning pipeline and expert validation, alongside a novel ethics benchmark for scientific research contexts covering eight subcategories including dual-use research and bias. We evaluated seven prominent LLMs, including four science-specialized models and three general-purpose industry models, using multiple evaluation metrics including accuracy, semantic similarity measures, and LLM-based scoring. General-purpose industry models overall outperformed science-specialized models across each trustworthiness dimension, with GPT-o4-mini demonstrating superior performance in truthfulness assessments and adversarial robustness. Science-specialized models showed significant deficiencies in logical and ethical reasoning capabilities, along with concerning vulnerabilities in safety evaluations, particularly in high-risk domains such as biosecurity and chemical weapons. By open-sourcing our framework, we provide a foundation for developing more trustworthy AI systems and advancing research on model safety and ethics in scientific contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科学研究中展现了变革性的潜力，但在高风险情境中的部署引发了重大可信度关切。在此背景下，我们介绍了SciTrust 2.0，一个全面的框架，用于评估科学应用中LLMs的可信度，涉及四个维度：事实准确性、对抗鲁棒性、科学安全性和科学伦理。该框架结合了通过验证的反思调优管道和专家验证开发的新型开放性事实准确性基准，以及涵盖双用途研究和偏见等八个子类别的新型伦理基准，适用于科学研究情境。我们使用多个评估指标，包括准确性、语义相似度度量和基于LLM的评分，对七种 prominent LLMs 进行了评估，包括四种专门针对科学的模型和三种通用行业模型。通用行业模型在每个可信度维度上总体上优于专门针对科学的模型，其中GPT-o4-mini在事实准确性评估和对抗鲁棒性方面表现出优越性能。专门针对科学的模型在逻辑和伦理推理能力方面显示出了显著的不足，并在安全性评估中显示出了令人担忧的脆弱性，特别是在生物安全和化学武器等高风险领域。通过开源我们的框架，我们为开发更可信的AI系统并推进科学情境下模型安全性和伦理性的研究奠定了基础。 

---
# Approximating Human Preferences Using a Multi-Judge Learned System 

**Title (ZH)**: 使用多评审员学习系统逼近人类偏好 

**Authors**: Eitán Sprejer, Fernando Avalos, Augusto Bernardi, Jose Pedro Brito de Azevedo Faustino, Jacob Haimes, Narmeen Fatimah Oozeer  

**Link**: [PDF](https://arxiv.org/pdf/2510.25884)  

**Abstract**: Aligning LLM-based judges with human preferences is a significant challenge, as they are difficult to calibrate and often suffer from rubric sensitivity, bias, and instability. Overcoming this challenge advances key applications, such as creating reliable reward models for Reinforcement Learning from Human Feedback (RLHF) and building effective routing systems that select the best-suited model for a given user query. In this work, we propose a framework for modeling diverse, persona-based preferences by learning to aggregate outputs from multiple rubric-conditioned judges. We investigate the performance of this approach against naive baselines and assess its robustness through case studies on both human and LLM-judges biases. Our primary contributions include a persona-based method for synthesizing preference labels at scale and two distinct implementations of our aggregator: Generalized Additive Model (GAM) and a Multi-Layer Perceptron (MLP). 

**Abstract (ZH)**: 基于LLM的法官与人类偏好对齐是一个重大挑战，它们难以校准且常受评分标准敏感性、偏见和不稳定性的困扰。克服这一挑战促进了关键应用的发展，如为人类反馈强化学习（RLHF）创建可靠的奖励模型以及构建有效的路由系统以选择最适合用户查询的模型。在本工作中，我们提出了一种通过学习聚合多个评分标准条件下的法官输出来建模多元人设偏好的框架。我们该方法与天真baseline的性能进行了对比，并通过针对人类和LLM法官偏见的案例研究评估了其稳健性。我们的主要贡献包括一种基于人设的大规模合成偏好标签的方法以及我们聚合器的两种实现：广义加性模型（GAM）和多层感知机（MLP）。 

---
# The Information-Theoretic Imperative: Compression and the Epistemic Foundations of Intelligence 

**Title (ZH)**: 信息论的 imperative ：压缩与智能的认识论基础 

**Authors**: Christian Dittrich, Jennifer Flygare Kinne  

**Link**: [PDF](https://arxiv.org/pdf/2510.25883)  

**Abstract**: Existing frameworks converge on the centrality of compression to intelligence but leave underspecified why this process enforces the discovery of causal structure rather than superficial statistical patterns. We introduce a two-level framework to address this gap. The Information-Theoretic Imperative (ITI) establishes that any system persisting in uncertain environments must minimize epistemic entropy through predictive compression: this is the evolutionary "why" linking survival pressure to information-processing demands. The Compression Efficiency Principle (CEP) specifies how efficient compression mechanically selects for generative, causal models through exception-accumulation dynamics, making reality alignment a consequence rather than a contingent achievement. Together, ITI and CEP define a causal chain: from survival pressure to prediction necessity, compression requirement, efficiency optimization, generative structure discovery, and ultimately reality alignment. Each link follows from physical, information-theoretic, or evolutionary constraints, implying that intelligence is the mechanically necessary outcome of persistence in structured environments. This framework yields empirically testable predictions: compression efficiency, measured as approach to the rate-distortion frontier, correlates with out-of-distribution generalization; exception-accumulation rates differentiate causal from correlational models; hierarchical systems exhibit increasing efficiency across abstraction layers; and biological systems demonstrate metabolic costs that track representational complexity. ITI and CEP thereby provide a unified account of convergence across biological, artificial, and multi-scale systems, addressing the epistemic and functional dimensions of intelligence without invoking assumptions about consciousness or subjective experience. 

**Abstract (ZH)**: 基于信息论的必要性与压缩效率原则：弥合智能中压缩与因果结构发现之间的鸿沟 

---
# Through the Judge's Eyes: Inferred Thinking Traces Improve Reliability of LLM Raters 

**Title (ZH)**: 从法官的角度看：推断出的思考痕迹提高大模型评价的可靠性 

**Authors**: Xingjian Zhang, Tianhong Gao, Suliang Jin, Tianhao Wang, Teng Ye, Eytan Adar, Qiaozhu Mei  

**Link**: [PDF](https://arxiv.org/pdf/2510.25860)  

**Abstract**: Large language models (LLMs) are increasingly used as raters for evaluation tasks. However, their reliability is often limited for subjective tasks, when human judgments involve subtle reasoning beyond annotation labels. Thinking traces, the reasoning behind a judgment, are highly informative but challenging to collect and curate. We present a human-LLM collaborative framework to infer thinking traces from label-only annotations. The proposed framework uses a simple and effective rejection sampling method to reconstruct these traces at scale. These inferred thinking traces are applied to two complementary tasks: (1) fine-tuning open LLM raters; and (2) synthesizing clearer annotation guidelines for proprietary LLM raters. Across multiple datasets, our methods lead to significantly improved LLM-human agreement. Additionally, the refined annotation guidelines increase agreement among different LLM models. These results suggest that LLMs can serve as practical proxies for otherwise unrevealed human thinking traces, enabling label-only corpora to be extended into thinking-trace-augmented resources that enhance the reliability of LLM raters. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地被用作评价任务的评分者。然而，在人类判断涉及超越标注标签的细微推理时，它们的可靠性往往有限。判断背后的推理思路虽高度信息丰富，但收集和整理起来极具挑战。我们提出了一种人-LLM协作框架，用于从仅标注的评分中推断推理思路。该框架采用简单的有效的拒绝抽样方法，在大规模范围内重构这些推理思路。推断出的这些推理思路应用于两个互补任务：（1）微调开放的LLM评分者；（2）为专有的LLM评分者合成更清晰的标注指南。在多个数据集上，我们的方法显著提高了LLM-人类的一致性。此外，改进的标注指南增加了不同LLM模型之间的共识。这些结果表明，LLMs可以作为实用的人类未揭示推理思路的代理，使仅标注的数据集扩充为包含推理思路增强资源，从而提高LLM评分者可靠性。 

---
# Symbolically Scaffolded Play: Designing Role-Sensitive Prompts for Generative NPC Dialogue 

**Title (ZH)**: 符号支撑的玩耍：设计角色敏感的生成NPC对话提示 

**Authors**: Vanessa Figueiredo, David Elumeze  

**Link**: [PDF](https://arxiv.org/pdf/2510.25820)  

**Abstract**: Large Language Models (LLMs) promise to transform interactive games by enabling non-player characters (NPCs) to sustain unscripted dialogue. Yet it remains unclear whether constrained prompts actually improve player experience. We investigate this question through The Interview, a voice-based detective game powered by GPT-4o. A within-subjects usability study ($N=10$) compared high-constraint (HCP) and low-constraint (LCP) prompts, revealing no reliable experiential differences beyond sensitivity to technical breakdowns. Guided by these findings, we redesigned the HCP into a hybrid JSON+RAG scaffold and conducted a synthetic evaluation with an LLM judge, positioned as an early-stage complement to usability testing. Results uncovered a novel pattern: scaffolding effects were role-dependent: the Interviewer (quest-giver NPC) gained stability, while suspect NPCs lost improvisational believability. These findings overturn the assumption that tighter constraints inherently enhance play. Extending fuzzy-symbolic scaffolding, we introduce \textit{Symbolically Scaffolded Play}, a framework in which symbolic structures are expressed as fuzzy, numerical boundaries that stabilize coherence where needed while preserving improvisation where surprise sustains engagement. 

**Abstract (ZH)**: 大型语言模型（LLMs）有望通过使非玩家角色（NPCs）能够维持非剧本对话来转变互动游戏。然而，尚不清楚受限提示是否实际上能改善玩家体验。我们通过由GPT-4驱动的声音推理游戏《访谈》来探讨这一问题。一项单被试使用性研究（$N=10$）将高约束（HCP）和低约束（LCP）提示进行了对比，结果显示除了技术故障的敏感性外，没有发现可靠的游戏体验差异。基于这些发现，我们重新设计了HCP为混合JSON+RAG支架，并进行了一项合成评估，以LLM裁判定位，作为使用性测试的早期补充。结果揭示了一种新型模式：支架效应因角色依赖：访谈者（任务提供者NPC）获得了稳定性，而嫌疑犯NPC失去了即兴可信度。这些发现推翻了更紧约束条件必然增强游戏体验的假设。扩展模糊符号支架，我们引入了《符号支架化游戏》框架，在该框架中，符号结构以模糊的、数值的边界形式表达，以便在需要时稳定连贯性，同时在惊喜保持游戏参与的地方保留即兴创作。 

---
# An Agentic Framework for Rapid Deployment of Edge AI Solutions in Industry 5.0 

**Title (ZH)**: 面向 Industry 5.0 的边缘人工智能解决方案快速部署能力框架 

**Authors**: Jorge Martinez-Gil, Mario Pichler, Nefeli Bountouni, Sotiris Koussouris, Marielena Márquez Barreiro, Sergio Gusmeroli  

**Link**: [PDF](https://arxiv.org/pdf/2510.25813)  

**Abstract**: We present a novel framework for Industry 5.0 that simplifies the deployment of AI models on edge devices in various industrial settings. The design reduces latency and avoids external data transfer by enabling local inference and real-time processing. Our implementation is agent-based, which means that individual agents, whether human, algorithmic, or collaborative, are responsible for well-defined tasks, enabling flexibility and simplifying integration. Moreover, our framework supports modular integration and maintains low resource requirements. Preliminary evaluations concerning the food industry in real scenarios indicate improved deployment time and system adaptability performance. The source code is publicly available at this https URL. 

**Abstract (ZH)**: 我们提出了一种面向Industry 5.0的新型框架，该框架简化了各种工业环境中AI模型在边缘设备上的部署。该设计通过实现局部推理和实时处理来减少延迟并避免外部数据传输。我们的实现基于代理，这意味着无论代理是人力、算法还是协作形式，都负责明确的任务，从而实现了灵活性并简化了集成。此外，该框架支持模块化集成并保持低资源要求。初步评估显示，在实际食品行业场景中的部署时间与系统适应性性能有所改善。源代码可在以下网址公开访问：this https URL。 

---
# Towards Piece-by-Piece Explanations for Chess Positions with SHAP 

**Title (ZH)**: 基于SHAP的象棋局面逐块解释方法 

**Authors**: Francesco Spinnato  

**Link**: [PDF](https://arxiv.org/pdf/2510.25775)  

**Abstract**: Contemporary chess engines offer precise yet opaque evaluations, typically expressed as centipawn scores. While effective for decision-making, these outputs obscure the underlying contributions of individual pieces or patterns. In this paper, we explore adapting SHAP (SHapley Additive exPlanations) to the domain of chess analysis, aiming to attribute a chess engines evaluation to specific pieces on the board. By treating pieces as features and systematically ablating them, we compute additive, per-piece contributions that explain the engines output in a locally faithful and human-interpretable manner. This method draws inspiration from classical chess pedagogy, where players assess positions by mentally removing pieces, and grounds it in modern explainable AI techniques. Our approach opens new possibilities for visualization, human training, and engine comparison. We release accompanying code and data to foster future research in interpretable chess AI. 

**Abstract (ZH)**: 当代国际象棋引擎提供精确但不透明的评估，通常以百子分数的形式表达。虽然这些输出对决策非常有效，但它们掩盖了每枚棋子或模式的底层贡献。本文探讨将SHAP（SHapley Additive exPlanations）方法应用于国际象棋分析领域，旨在将引擎的评估归因于棋盘上的特定棋子。通过将棋子视为特征并系统地去除它们，我们计算出每个棋子的增量贡献，从而以局部忠实且易于人类理解的方式解释引擎的输出。该方法借鉴了经典的国际象棋教学方法，其中棋手通过心理移除棋子来评估局面，并将其置于现代可解释AI技术的基础之上。本文的方法为可视化、人类训练和引擎比较开辟了新的可能性。我们发布了配套的代码和数据，以促进可解释国际象棋AI的未来研究。 

---
# Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-CoF Benchmark 

**Title (ZH)**: 视频模型准备好作为零样本推理器了吗？基于MME-CoF基准的实证研究 

**Authors**: Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2510.26802)  

**Abstract**: Recent video generation models can produce high-fidelity, temporally coherent videos, indicating that they may encode substantial world knowledge. Beyond realistic synthesis, they also exhibit emerging behaviors indicative of visual perception, modeling, and manipulation. Yet, an important question still remains: Are video models ready to serve as zero-shot reasoners in challenging visual reasoning scenarios? In this work, we conduct an empirical study to comprehensively investigate this question, focusing on the leading and popular Veo-3. We evaluate its reasoning behavior across 12 dimensions, including spatial, geometric, physical, temporal, and embodied logic, systematically characterizing both its strengths and failure modes. To standardize this study, we curate the evaluation data into MME-CoF, a compact benchmark that enables in-depth and thorough assessment of Chain-of-Frame (CoF) reasoning. Our findings reveal that while current video models demonstrate promising reasoning patterns on short-horizon spatial coherence, fine-grained grounding, and locally consistent dynamics, they remain limited in long-horizon causal reasoning, strict geometric constraints, and abstract logic. Overall, they are not yet reliable as standalone zero-shot reasoners, but exhibit encouraging signs as complementary visual engines alongside dedicated reasoning models. Project page: this https URL 

**Abstract (ZH)**: 最近的视频生成模型可以生成高保真、时序上一致的视频，表明它们可能编码了大量的世界知识。除了现实合成之外，它们还表现出一些视觉感知、建模和操作的新兴行为。然而，一个重要的问题仍然存在：视频模型是否准备好在具有挑战性的视觉推理场景中作为零-shot推理引擎使用？在本文中，我们进行了一项实证研究，全面探讨这一问题，重点关注领先的流行模型Veо-3。我们从12个维度评估其推理行为，包括空间、几何、物理、时间和实体逻辑，系统地刻画其优势和失败模式。为了标准化这一研究，我们将评估数据整理成MME-CoF，这是一个紧凑的基准测试，能够深入和全面评估Chain-of-Frame（CoF）推理。我们的发现表明，尽管当前的视频模型在短时效空间一致性、精细的空间相关性和局部一致的动力学方面表现出有希望的推理模式，但在长时效因果推理、严格的几何约束和抽象逻辑方面依然有限。总体而言，它们尚未可靠地作为独立的零-shot推理引擎使用，但作为与专门推理模型互补的视觉引擎时展现出令人鼓舞的迹象。 

---
# Gistify! Codebase-Level Understanding via Runtime Execution 

**Title (ZH)**: Gistify！通过运行时执行实现代码库级理解 

**Authors**: Hyunji Lee, Minseon Kim, Chinmay Singh, Matheus Pereira, Atharv Sonwane, Isadora White, Elias Stengel-Eskin, Mohit Bansal, Zhengyan Shi, Alessandro Sordoni, Marc-Alexandre Côté, Xingdi Yuan, Lucas Caccia  

**Link**: [PDF](https://arxiv.org/pdf/2510.26790)  

**Abstract**: As coding agents are increasingly deployed in large codebases, the need to automatically design challenging, codebase-level evaluation is central. We propose Gistify, a task where a coding LLM must create a single, minimal, self-contained file that can reproduce a specific functionality of a codebase. The coding LLM is given full access to a codebase along with a specific entrypoint (e.g., a python command), and the generated file must replicate the output of the same command ran under the full codebase, while containing only the essential components necessary to execute the provided command. Success on Gistify requires both structural understanding of the codebase, accurate modeling of its execution flow as well as the ability to produce potentially large code patches. Our findings show that current state-of-the-art models struggle to reliably solve Gistify tasks, especially ones with long executions traces. 

**Abstract (ZH)**: 随着编码代理在大规模代码库中的应用越来越广泛，自动设计代码库级别的挑战性评估变得至关重要。我们提出了一项名为Gistify的任务，其中编码LLM必须创建一个单一的、最小的、自包含的文件，以重现代码库中的特定功能。编码LLM被给予访问整个代码库以及一个特定的入口点（例如，一个Python命令），生成的文件必须在执行相同的命令时重现整个代码库的输出，同时仅包含执行所提供命令所需的必要组成部分。Gistify任务的成功要求对代码库的结构有深刻的理解，准确地模型化其执行流程，以及能够生成潜在的大规模代码补丁。我们的研究发现，当前最先进的模型在解决Gistify任务时表现不佳，尤其是在执行轨迹较长的任务方面。 

---
# Defeating the Training-Inference Mismatch via FP16 

**Title (ZH)**: 通过FP16消除训练-推理不匹配 

**Authors**: Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.26788)  

**Abstract**: Reinforcement learning (RL) fine-tuning of large language models (LLMs) often suffers from instability due to the numerical mismatch between the training and inference policies. While prior work has attempted to mitigate this issue through algorithmic corrections or engineering alignments, we show that its root cause lies in the floating point precision itself. The widely adopted BF16, despite its large dynamic range, introduces large rounding errors that breaks the consistency between training and inference. In this work, we demonstrate that simply reverting to \textbf{FP16} effectively eliminates this mismatch. The change is simple, fully supported by modern frameworks with only a few lines of code change, and requires no modification to the model architecture or learning algorithm. Our results suggest that using FP16 uniformly yields more stable optimization, faster convergence, and stronger performance across diverse tasks, algorithms and frameworks. We hope these findings motivate a broader reconsideration of precision trade-offs in RL fine-tuning. 

**Abstract (ZH)**: 使用FP16有效消除大型语言模型强化学习微调中的数值不匹配问题 

---
# Remote Labor Index: Measuring AI Automation of Remote Work 

**Title (ZH)**: 远程劳动指数：衡量人工智能对远程工作的自动化程度 

**Authors**: Mantas Mazeika, Alice Gatti, Cristina Menghini, Udari Madhushani Sehwag, Shivam Singhal, Yury Orlovskiy, Steven Basart, Manasi Sharma, Denis Peskoff, Elaine Lau, Jaehyuk Lim, Lachlan Carroll, Alice Blair, Vinaya Sivakumar, Sumana Basu, Brad Kenstler, Yuntao Ma, Julian Michael, Xiaoke Li, Oliver Ingebretsen, Aditya Mehta, Jean Mottola, John Teichmann, Kevin Yu, Zaina Shaik, Adam Khoja, Richard Ren, Jason Hausenloy, Long Phan, Ye Htet, Ankit Aich, Tahseen Rabbani, Vivswan Shah, Andriy Novykov, Felix Binder, Kirill Chugunov, Luis Ramirez, Matias Geralnik, Hernán Mesura, Dean Lee, Ed-Yeremai Hernandez Cardona, Annette Diamond, Summer Yue, Alexandr Wang, Bing Liu, Ernesto Hernandez, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2510.26787)  

**Abstract**: AIs have made rapid progress on research-oriented benchmarks of knowledge and reasoning, but it remains unclear how these gains translate into economic value and automation. To measure this, we introduce the Remote Labor Index (RLI), a broadly multi-sector benchmark comprising real-world, economically valuable projects designed to evaluate end-to-end agent performance in practical settings. AI agents perform near the floor on RLI, with the highest-performing agent achieving an automation rate of 2.5%. These results help ground discussions of AI automation in empirical evidence, setting a common basis for tracking AI impacts and enabling stakeholders to proactively navigate AI-driven labor automation. 

**Abstract (ZH)**: AIs在知识和推理的研究导向基准上取得了 rapid 进展，但这些进展如何转化为经济价值和自动化尚不明确。为了衡量这一点，我们引入了远程劳动力指数（RLI），这是一个广泛覆盖多个行业的基准，包含实际的、具有经济价值的项目，旨在评估代理在实际环境中的端到端表现。AI代理在RLI上的表现接近最低水平，最高性能的代理实现了2.5%的自动化率。这些结果有助于基于实证证据讨论AI自动化，为跟踪AI影响设定共同基准，并使利益相关者能够主动应对由AI驱动的劳动力自动化。 

---
# Clone Deterministic 3D Worlds with Geometrically-Regularized World Models 

**Title (ZH)**: 克隆确定性3D世界：带有几何正则化的世界模型 

**Authors**: Zaishuo Xia, Yukuan Lu, Xinyi Li, Yifan Xu, Yubei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26782)  

**Abstract**: A world model is an internal model that simulates how the world evolves. Given past observations and actions, it predicts the future of both the embodied agent and its environment. Accurate world models are essential for enabling agents to think, plan, and reason effectively in complex, dynamic settings. Despite rapid progress, current world models remain brittle and degrade over long horizons. We argue that a central cause is representation quality: exteroceptive inputs (e.g., images) are high-dimensional, and lossy or entangled latents make dynamics learning unnecessarily hard. We therefore ask whether improving representation learning alone can substantially improve world-model performance. In this work, we take a step toward building a truly accurate world model by addressing a fundamental yet open problem: constructing a model that can fully clone and overfit to a deterministic 3D world. We propose Geometrically-Regularized World Models (GRWM), which enforces that consecutive points along a natural sensory trajectory remain close in latent representation space. This approach yields significantly improved latent representations that align closely with the true topology of the environment. GRWM is plug-and-play, requires only minimal architectural modification, scales with trajectory length, and is compatible with diverse latent generative backbones. Across deterministic 3D settings and long-horizon prediction tasks, GRWM significantly increases rollout fidelity and stability. Analyses show that its benefits stem from learning a latent manifold with superior geometric structure. These findings support a clear takeaway: improving representation learning is a direct and useful path to robust world models, delivering reliable long-horizon predictions without enlarging the dynamics module. 

**Abstract (ZH)**: 几何正则化世界模型：构建真正准确的世界模型 

---
# Faithful and Fast Influence Function via Advanced Sampling 

**Title (ZH)**: 忠实且高效的影响力函数通过高级采样 

**Authors**: Jungyeon Koh, Hyeonsu Lyu, Jonggyu Jang, Hyun Jong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26776)  

**Abstract**: How can we explain the influence of training data on black-box models? Influence functions (IFs) offer a post-hoc solution by utilizing gradients and Hessians. However, computing the Hessian for an entire dataset is resource-intensive, necessitating a feasible alternative. A common approach involves randomly sampling a small subset of the training data, but this method often results in highly inconsistent IF estimates due to the high variance in sample configurations. To address this, we propose two advanced sampling techniques based on features and logits. These samplers select a small yet representative subset of the entire dataset by considering the stochastic distribution of features or logits, thereby enhancing the accuracy of IF estimations. We validate our approach through class removal experiments, a typical application of IFs, using the F1-score to measure how effectively the model forgets the removed class while maintaining inference consistency on the remaining classes. Our method reduces computation time by 30.1% and memory usage by 42.2%, or improves the F1-score by 2.5% compared to the baseline. 

**Abstract (ZH)**: 如何解释训练数据对黑盒模型的影响？特征和logits导向的高级采样技术提供了一种替代方案，以提高影响函数估计的准确性。 

---
# STaMP: Sequence Transformation and Mixed Precision for Low-Precision Activation Quantization 

**Title (ZH)**: STaMP: 序列转换和混合精度用于低精度激活量化 

**Authors**: Marco Federici, Riccardo Del Chiaro, Boris van Breugel, Paul Whatmough, Markus Nagel  

**Link**: [PDF](https://arxiv.org/pdf/2510.26771)  

**Abstract**: Quantization is the key method for reducing inference latency, power and memory footprint of generative AI models. However, accuracy often degrades sharply when activations are quantized below eight bits. Recent work suggests that invertible linear transformations (e.g. rotations) can aid quantization, by reparameterizing feature channels and weights. In this paper, we propose \textit{Sequence Transformation and Mixed Precision} (STaMP) quantization, a novel strategy that applies linear transformations along the \textit{sequence} dimension to exploit the strong local correlation in language and visual data. By keeping a small number of tokens in each intermediate activation at higher precision, we can maintain model accuracy at lower (average) activations bit-widths. We evaluate STaMP on recent LVM and LLM architectures, demonstrating that it significantly improves low bit width activation quantization and complements established activation and weight quantization methods including recent feature transformations. 

**Abstract (ZH)**: 序列变换和混合精度量化（STaMP）：利用序列维度上的线性变换提高低比特激活量化性能 

---
# AMO-Bench: Large Language Models Still Struggle in High School Math Competitions 

**Title (ZH)**: AMO-Bench: 大型语言模型仍在高中数学竞赛中挣扎 

**Authors**: Shengnan An, Xunliang Cai, Xuezhi Cao, Xiaoyu Li, Yehao Lin, Junlin Liu, Xinxuan Lv, Dan Ma, Xuanlin Wang, Ziwen Wang, Shuang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.26768)  

**Abstract**: We present AMO-Bench, an Advanced Mathematical reasoning benchmark with Olympiad level or even higher difficulty, comprising 50 human-crafted problems. Existing benchmarks have widely leveraged high school math competitions for evaluating mathematical reasoning capabilities of large language models (LLMs). However, many existing math competitions are becoming less effective for assessing top-tier LLMs due to performance saturation (e.g., AIME24/25). To address this, AMO-Bench introduces more rigorous challenges by ensuring all 50 problems are (1) cross-validated by experts to meet at least the International Mathematical Olympiad (IMO) difficulty standards, and (2) entirely original problems to prevent potential performance leakages from data memorization. Moreover, each problem in AMO-Bench requires only a final answer rather than a proof, enabling automatic and robust grading for evaluation. Experimental results across 26 LLMs on AMO-Bench show that even the best-performing model achieves only 52.4% accuracy on AMO-Bench, with most LLMs scoring below 40%. Beyond these poor performances, our further analysis reveals a promising scaling trend with increasing test-time compute on AMO-Bench. These results highlight the significant room for improving the mathematical reasoning in current LLMs. We release AMO-Bench to facilitate further research into advancing the reasoning abilities of language models. this https URL 

**Abstract (ZH)**: AMO-Bench：一个高级数学推理基准，包含奥林匹克级别甚至更高的难度，共包含50个人工制作的问题 

---
# Deep sequence models tend to memorize geometrically; it is unclear why 

**Title (ZH)**: 深度序列模型倾向于记忆几何特征；尚不清楚其中原因。 

**Authors**: Shahriar Noroozizadeh, Vaishnavh Nagarajan, Elan Rosenfeld, Sanjiv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.26745)  

**Abstract**: In sequence modeling, the parametric memory of atomic facts has been predominantly abstracted as a brute-force lookup of co-occurrences between entities. We contrast this associative view against a geometric view of how memory is stored. We begin by isolating a clean and analyzable instance of Transformer reasoning that is incompatible with memory as strictly a storage of the local co-occurrences specified during training. Instead, the model must have somehow synthesized its own geometry of atomic facts, encoding global relationships between all entities, including non-co-occurring ones. This in turn has simplified a hard reasoning task involving an $\ell$-fold composition into an easy-to-learn 1-step geometric task.
From this phenomenon, we extract fundamental aspects of neural embedding geometries that are hard to explain. We argue that the rise of such a geometry, despite optimizing over mere local associations, cannot be straightforwardly attributed to typical architectural or optimizational pressures. Counterintuitively, an elegant geometry is learned even when it is not more succinct than a brute-force lookup of associations.
Then, by analyzing a connection to Node2Vec, we demonstrate how the geometry stems from a spectral bias that -- in contrast to prevailing theories -- indeed arises naturally despite the lack of various pressures. This analysis also points to practitioners a visible headroom to make Transformer memory more strongly geometric. We hope the geometric view of parametric memory encourages revisiting the default intuitions that guide researchers in areas like knowledge acquisition, capacity, discovery and unlearning. 

**Abstract (ZH)**: 在序列建模中，原子事实的参数化记忆主要被抽象为实体共现的简单查找。我们将这种关联视角与记忆的几何存储方式进行了对比。我们首先隔离了一个与严格存储训练期间指定的局部共现无兼容性的Transformer推理实例。模型必须通过某种方式合成自身的原子事实几何结构，编码所有实体之间的全局关系，包括非共现实体之间的关系。这进而将一个艰难的推理任务简化为一个易于学习的一步几何任务。

从这一现象中，我们提取出难以用典型架构或优化压力解释的基本神经嵌入几何特性。我们认为，尽管优化仅涉及局部关联，如此几何结构的兴起仍不能简单归因于这些典型压力。令人Unexpected地，即使几何结构并不是比关联查找更简洁，它也能自然地被学习到。

通过分析Node2Vec的连接关系，我们展示了这种几何结构源自光谱偏差——与现有理论相反，即使缺乏各种压力，光谱偏差确实自然地出现。这种分析也为实践者指出了增强Transformer记忆几何特性的可见空间。我们希望对参数化记忆的几何视角能促进研究人员重新审视知识获取、容量、发现和遗忘等领域的默认直觉。 

---
# A General Incentives-Based Framework for Fairness in Multi-agent Resource Allocation 

**Title (ZH)**: 基于激励的多代理资源分配公平性通用框架 

**Authors**: Ashwin Kumar, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2510.26740)  

**Abstract**: We introduce the General Incentives-based Framework for Fairness (GIFF), a novel approach for fair multi-agent resource allocation that infers fair decision-making from standard value functions. In resource-constrained settings, agents optimizing for efficiency often create inequitable outcomes. Our approach leverages the action-value (Q-)function to balance efficiency and fairness without requiring additional training. Specifically, our method computes a local fairness gain for each action and introduces a counterfactual advantage correction term to discourage over-allocation to already well-off agents. This approach is formalized within a centralized control setting, where an arbitrator uses the GIFF-modified Q-values to solve an allocation problem.
Empirical evaluations across diverse domains, including dynamic ridesharing, homelessness prevention, and a complex job allocation task-demonstrate that our framework consistently outperforms strong baselines and can discover far-sighted, equitable policies. The framework's effectiveness is supported by a theoretical foundation; we prove its fairness surrogate is a principled lower bound on the true fairness improvement and that its trade-off parameter offers monotonic tuning. Our findings establish GIFF as a robust and principled framework for leveraging standard reinforcement learning components to achieve more equitable outcomes in complex multi-agent systems. 

**Abstract (ZH)**: 基于激励的一致性框架以实现公平性（GIFF）：一种新的公平多智能体资源分配方法 

---
# ExpertFlow: Adaptive Expert Scheduling and Memory Coordination for Efficient MoE Inference 

**Title (ZH)**: ExpertFlow：自适应专家调度与内存协调以实现高效MoE推理 

**Authors**: Zixu Shen, Kexin Chu, Yifan Zhang, Dawei Xiang, Runxin Wu, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26730)  

**Abstract**: The expansion of large language models is increasingly limited by the constrained memory capacity of modern GPUs. To mitigate this, Mixture-of-Experts (MoE) architectures activate only a small portion of parameters during inference, significantly lowering both memory demand and computational overhead. However, conventional MoE inference approaches, which select active experts independently at each layer, often introduce considerable latency because of frequent parameter transfers between host and GPU memory. In addition, current cross-layer prediction strategies, which are typically based on fixed steps, lack adaptability across different hardware platforms and workloads, thereby reducing their robustness and effectiveness.
To address these challenges, we present ExpertFlow, a runtime system for MoE inference that combines adaptive expert prefetching and cache-aware routing. ExpertFlow continuously adjusts its prediction horizon for expert activation by leveraging runtime statistics such as transfer bandwidth, parameter dimensionality, and model feedback signals. Furthermore, it incorporates a hybrid cross-layer prediction scheme that fuses pregating information with intermediate computational states to anticipate future expert needs. By adaptively refining prefetching decisions and aligning them with actual usage behavior, ExpertFlow effectively decreases cache misses and removes latency caused by expert swap-ins. Our evaluation demonstrates that ExpertFlow reduces model stall time to less than 0.1% of the baseline, highlighting its capability to optimize MoE inference under stringent memory constraints. 

**Abstract (ZH)**: 基于专家流的MoE推理运行时系统：结合自适应专家预取和缓存感知路由 

---
# Non-Convex Over-the-Air Heterogeneous Federated Learning: A Bias-Variance Trade-off 

**Title (ZH)**: 非凸空中异构联邦学习：偏差-方差权衡 

**Authors**: Muhammad Faraz Ul Abrar, Nicolò Michelusi  

**Link**: [PDF](https://arxiv.org/pdf/2510.26722)  

**Abstract**: Over-the-air (OTA) federated learning (FL) has been well recognized as a scalable paradigm that exploits the waveform superposition of the wireless multiple-access channel to aggregate model updates in a single use. Existing OTA-FL designs largely enforce zero-bias model updates by either assuming \emph{homogeneous} wireless conditions (equal path loss across devices) or forcing zero-bias updates to guarantee convergence. Under \emph{heterogeneous} wireless scenarios, however, such designs are constrained by the weakest device and inflate the update variance. Moreover, prior analyses of biased OTA-FL largely address convex objectives, while most modern AI models are highly non-convex. Motivated by these gaps, we study OTA-FL with stochastic gradient descent (SGD) for general smooth non-convex objectives under wireless heterogeneity. We develop novel OTA-FL SGD updates that allow a structured, time-invariant model bias while facilitating reduced variance updates. We derive a finite-time stationarity bound (expected time average squared gradient norm) that explicitly reveals a bias-variance trade-off. To optimize this trade-off, we pose a non-convex joint OTA power-control design and develop an efficient successive convex approximation (SCA) algorithm that requires only statistical CSI at the base station. Experiments on a non-convex image classification task validate the approach: the SCA-based design accelerates convergence via an optimized bias and improves generalization over prior OTA-FL baselines. 

**Abstract (ZH)**: 空中 federated learning 与随机梯度下降在无线异构环境下的建模与优化 

---
# On the limitation of evaluating machine unlearning using only a single training seed 

**Title (ZH)**: 关于仅使用单个训练种子评估机器遗忘限制的局限性 

**Authors**: Jamie Lanyon, Axel Finke, Petros Andreou, Georgina Cosma  

**Link**: [PDF](https://arxiv.org/pdf/2510.26714)  

**Abstract**: Machine unlearning (MU) aims to remove the influence of certain data points from a trained model without costly retraining. Most practical MU algorithms are only approximate and their performance can only be assessed empirically. Care must therefore be taken to make empirical comparisons as representative as possible. A common practice is to run the MU algorithm multiple times independently starting from the same trained model. In this work, we demonstrate that this practice can give highly non-representative results because -- even for the same architecture and same dataset -- some MU methods can be highly sensitive to the choice of random number seed used for model training. We therefore recommend that empirical comphttps://info.arxiv.org/help/prep#commentsarisons of MU algorithms should also reflect the variability across different model training seeds. 

**Abstract (ZH)**: 机器遗忘（MU）的目标是在无需昂贵重新训练的情况下从训练模型中去除某些数据点的影响。大多数实际的MU算法只是近似的，其性能只能通过经验来评估。因此，在进行经验比较时必须尽可能使其具有代表性。一种常见的做法是从同一个训练模型开始，独立地多次运行MU算法。然而，在这项工作中，我们证明了这种做法可能会导致高度非代表性的结果，即使在同一架构和同一数据集下，某些MU方法也可能对用于模型训练的随机数种子的选择高度敏感。因此，我们建议在评估MU算法时也应反映出不同模型训练种子之间的变化性。 

---
# The End of Manual Decoding: Towards Truly End-to-End Language Models 

**Title (ZH)**: 手动解码的终结：迈向真正意义上的端到端语言模型 

**Authors**: Zhichao Wang, Dongyang Ma, Xinting Huang, Deng Cai, Tian Lan, Jiahao Xu, Haitao Mi, Xiaoying Tang, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26697)  

**Abstract**: The "end-to-end" label for LLMs is a misnomer. In practice, they depend on a non-differentiable decoding process that requires laborious, hand-tuning of hyperparameters like temperature and top-p. This paper introduces AutoDeco, a novel architecture that enables truly "end-to-end" generation by learning to control its own decoding strategy. We augment the standard transformer with lightweight heads that, at each step, dynamically predict context-specific temperature and top-p values alongside the next-token logits. This approach transforms decoding into a parametric, token-level process, allowing the model to self-regulate its sampling strategy within a single forward pass.
Through extensive experiments on eight benchmarks, we demonstrate that AutoDeco not only significantly outperforms default decoding strategies but also achieves performance comparable to an oracle-tuned baseline derived from "hacking the test set"-a practical upper bound for any static method. Crucially, we uncover an emergent capability for instruction-based decoding control: the model learns to interpret natural language commands (e.g., "generate with low randomness") and adjusts its predicted temperature and top-p on a token-by-token basis, opening a new paradigm for steerable and interactive LLM decoding. 

**Abstract (ZH)**: LLMs的“端到端”标签是一个误解。实际上，它们依赖于一个非可微解码过程，需要手动调整温度和top-p等超参数。本文介绍了AutoDeco，一种新型架构，能够通过学习控制自身的解码策略从而实现真正意义上的“端到端”生成。我们通过在轻量级头部中动态预测上下文相关的温度和top-p值，并将其与下一个标记的logits结合，将解码过程转变为参数化和标记级别的过程，允许模型在单向前传中自我调节其采样策略。通过在八个基准上进行广泛的实验，我们证明AutoDeco不仅显著优于默认解码策略，还能够在性能上与通过“破解测试集”得到的oracle调优基准相当——这是任何静态方法的实用上限。更重要的是，我们发现了一种新兴的能力：指令导向的解码控制。模型能够理解自然语言命令（例如，“生成具有低随机性”），并在标记级别上调整其预测的温度和top-p，从而开启一种可控和交互的LLM解码新范式。 

---
# Process Integrated Computer Vision for Real-Time Failure Prediction in Steel Rolling Mill 

**Title (ZH)**: 实时预测钢铁轧制厂故障的工艺集成计算机视觉技术 

**Authors**: Vaibhav Kurrey, Sivakalyan Pujari, Gagan Raj Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.26684)  

**Abstract**: We present a long-term deployment study of a machine vision-based anomaly detection system for failure prediction in a steel rolling mill. The system integrates industrial cameras to monitor equipment operation, alignment, and hot bar motion in real time along the process line. Live video streams are processed on a centralized video server using deep learning models, enabling early prediction of equipment failures and process interruptions, thereby reducing unplanned breakdown costs. Server-based inference minimizes the computational load on industrial process control systems (PLCs), supporting scalable deployment across production lines with minimal additional resources. By jointly analyzing sensor data from data acquisition systems and visual inputs, the system identifies the location and probable root causes of failures, providing actionable insights for proactive maintenance. This integrated approach enhances operational reliability, productivity, and profitability in industrial manufacturing environments. 

**Abstract (ZH)**: 基于机器视觉的钢轧机故障预测异常检测系统长周期部署研究 

---
# Evontree: Ontology Rule-Guided Self-Evolution of Large Language Models 

**Title (ZH)**: Evontree: 基于本体规则的大语言模型自我进化方法 

**Authors**: Mingchen Tu, Zhiqiang Liu, Juan Li, Liangyurui Liu, Junjie Wang, Lei Liang, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26683)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional capabilities across multiple domains by leveraging massive pre-training and curated fine-tuning data. However, in data-sensitive fields such as healthcare, the lack of high-quality, domain-specific training corpus hinders LLMs' adaptation for specialized applications. Meanwhile, domain experts have distilled domain wisdom into ontology rules, which formalize relationships among concepts and ensure the integrity of knowledge management repositories. Viewing LLMs as implicit repositories of human knowledge, we propose Evontree, a novel framework that leverages a small set of high-quality ontology rules to systematically extract, validate, and enhance domain knowledge within LLMs, without requiring extensive external datasets. Specifically, Evontree extracts domain ontology from raw models, detects inconsistencies using two core ontology rules, and reinforces the refined knowledge via self-distilled fine-tuning. Extensive experiments on medical QA benchmarks with Llama3-8B-Instruct and Med42-v2 demonstrate consistent outperformance over both unmodified models and leading supervised baselines, achieving up to a 3.7% improvement in accuracy. These results confirm the effectiveness, efficiency, and robustness of our approach for low-resource domain adaptation of LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过利用大量的预先训练数据和精心策划的微调数据，在多个领域展现了卓越的能力。然而，在医疗保健等数据敏感领域，缺乏高质量的领域特定训练语料库阻碍了LLMs的应用专业化。同时，领域专家将领域智慧提炼为本体规则，这些规则正式化了概念之间的关系，并确保了知识管理库的一致性。将LLMs视作人类知识的隐式存储库，我们提出了一种名为Evontree的新框架，该框架利用少量高质量的本体规则系统地提取、验证和增强LLMs中的领域知识，而无需大量外部数据集。具体而言，Evontree从原始模型中提取领域本体，使用两种核心本体规则检测不一致性，并通过自我提炼微调强化精炼知识。在使用Llama3-8B-Instruct和Med42-v2进行的医疗问答基准测试中，与未经修改的模型和领先的监督基线相比，该框架表现出显著的优势，准确率提高了多达3.7%。这些结果证实了我们的方法在LLMs低资源领域适应性方面的有效性和稳健性。 

---
# Hybrid DQN-TD3 Reinforcement Learning for Autonomous Navigation in Dynamic Environments 

**Title (ZH)**: 混合DQN-TD3强化学习在动态环境中的自主导航 

**Authors**: Xiaoyi He, Danggui Chen, Zhenshuo Zhang, Zimeng Bai  

**Link**: [PDF](https://arxiv.org/pdf/2510.26646)  

**Abstract**: This paper presents a hierarchical path-planning and control framework that combines a high-level Deep Q-Network (DQN) for discrete sub-goal selection with a low-level Twin Delayed Deep Deterministic Policy Gradient (TD3) controller for continuous actuation. The high-level module selects behaviors and sub-goals; the low-level module executes smooth velocity commands. We design a practical reward shaping scheme (direction, distance, obstacle avoidance, action smoothness, collision penalty, time penalty, and progress), together with a LiDAR-based safety gate that prevents unsafe motions. The system is implemented in ROS + Gazebo (TurtleBot3) and evaluated with PathBench metrics, including success rate, collision rate, path efficiency, and re-planning efficiency, in dynamic and partially observable environments. Experiments show improved success rate and sample efficiency over single-algorithm baselines (DQN or TD3 alone) and rule-based planners, with better generalization to unseen obstacle configurations and reduced abrupt control changes. Code and evaluation scripts are available at the project repository. 

**Abstract (ZH)**: 本文提出了一种分级路径规划与控制框架，该框架结合了高层次的深度Q网络（DQN）进行离散子目标选择和低层次的双延迟深度确定性策略梯度（TD3）控制器进行连续动作执行。高层次模块选择行为和子目标；低层次模块执行平滑的速度命令。我们设计了一个实用的奖励塑造方案（方向、距离、障碍物避免、动作平滑性、碰撞惩罚、时间惩罚和进程度），并配备了一个基于LiDAR的安全门，防止不安全的运动。该系统在ROS + Gazebo（TurtleBot3）上实现，并使用PathBench指标在动态和部分可观测环境中进行评估，包括成功率、碰撞率、路径效率和重规划效率。实验结果显示，该框架在单算法基线（单独的DQN或TD3）和基于规则的规划者上提高了成功率和样本效率，并且能够更好地泛化到未见过的障碍配置，并减少突然的控制变化。代码和评估脚本可在项目仓库中获得。 

---
# Aeolus: A Multi-structural Flight Delay Dataset 

**Title (ZH)**: Aeolus: 多结构航班延误数据集 

**Authors**: Lin Xu, Xinyun Yuan, Yuxuan Liang, Suwan Yin, Yuankai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26616)  

**Abstract**: We introduce Aeolus, a large-scale Multi-modal Flight Delay Dataset designed to advance research on flight delay prediction and support the development of foundation models for tabular data. Existing datasets in this domain are typically limited to flat tabular structures and fail to capture the spatiotemporal dynamics inherent in delay propagation. Aeolus addresses this limitation by providing three aligned modalities: (i) a tabular dataset with rich operational, meteorological, and airportlevel features for over 50 million flights; (ii) a flight chain module that models delay propagation along sequential flight legs, capturing upstream and downstream dependencies; and (iii) a flight network graph that encodes shared aircraft, crew, and airport resource connections, enabling cross-flight relational reasoning. The dataset is carefully constructed with temporal splits, comprehensive features, and strict leakage prevention to support realistic and reproducible machine learning evaluation. Aeolus supports a broad range of tasks, including regression, classification, temporal structure modeling, and graph learning, serving as a unified benchmark across tabular, sequential, and graph modalities. We release baseline experiments and preprocessing tools to facilitate adoption. Aeolus fills a key gap for both domain-specific modeling and general-purpose structured data this http URL source code and data can be accessed at this https URL 

**Abstract (ZH)**: Aeolus：一种大规模多模态飞行延误数据集，用于推进飞行延误预测研究并支持表格数据基础模型的发展 

---
# ResMatching: Noise-Resilient Computational Super-Resolution via Guided Conditional Flow Matching 

**Title (ZH)**: ResMatching: 噪声鲁棒的指导条件流匹配计算超分辨率 

**Authors**: Anirban Ray, Vera Galinova, Florian Jug  

**Link**: [PDF](https://arxiv.org/pdf/2510.26601)  

**Abstract**: Computational Super-Resolution (CSR) in fluorescence microscopy has, despite being an ill-posed problem, a long history. At its very core, CSR is about finding a prior that can be used to extrapolate frequencies in a micrograph that have never been imaged by the image-generating microscope. It stands to reason that, with the advent of better data-driven machine learning techniques, stronger prior can be learned and hence CSR can lead to better results. Here, we present ResMatching, a novel CSR method that uses guided conditional flow matching to learn such improved data-priors. We evaluate ResMatching on 4 diverse biological structures from the BioSR dataset and compare its results against 7 baselines. ResMatching consistently achieves competitive results, demonstrating in all cases the best trade-off between data fidelity and perceptual realism. We observe that CSR using ResMatching is particularly effective in cases where a strong prior is hard to learn, e.g. when the given low-resolution images contain a lot of noise. Additionally, we show that ResMatching can be used to sample from an implicitly learned posterior distribution and that this distribution is calibrated for all tested use-cases, enabling our method to deliver a pixel-wise data-uncertainty term that can guide future users to reject uncertain predictions. 

**Abstract (ZH)**: 荧光显微镜中超分辨率计算（Computational Super-Resolution in Fluorescence Microscopy: A Long History Despite Being an Ill-Posed Problem） 

---
# Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems 

**Title (ZH)**: 有效利用令牌：面向高效的运行时多智能体系统 

**Authors**: Fulin Lin, Shaowen Chen, Ruishan Fang, Hongwei Wang, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.26585)  

**Abstract**: While Multi-Agent Systems (MAS) excel at complex tasks, their growing autonomy with operational complexity often leads to critical inefficiencies, such as excessive token consumption and failures arising from misinformation. Existing methods primarily focus on post-hoc failure attribution, lacking proactive, real-time interventions to enhance robustness and efficiency. To this end, we introduce SupervisorAgent, a lightweight and modular framework for runtime, adaptive supervision that operates without altering the base agent's architecture. Triggered by an LLM-free adaptive filter, SupervisorAgent intervenes at critical junctures to proactively correct errors, guide inefficient behaviors, and purify observations. On the challenging GAIA benchmark, SupervisorAgent reduces the token consumption of the Smolagent framework by an average of 29.45% without compromising its success rate. Extensive experiments across five additional benchmarks (math reasoning, code generation, and question answering) and various SoTA foundation models validate the broad applicability and robustness of our approach. The code is available at this https URL. 

**Abstract (ZH)**: 多代理系统（MAS）在复杂任务中表现出色，但随着运行复杂性的增加，其自主性往往会引发关键的低效问题，如令牌消耗过多和因错误信息引起的功能失效。现有方法主要侧重于事后故障归因，缺乏前瞻性的、实时的干预措施来提高稳健性和效率。为此，我们提出了SupervisorAgent，一个轻量级且模块化的运行时自适应监督框架，无需修改基础代理架构。SupervisorAgent通过一个无LLM的自适应滤波器触发，在关键节点上主动纠正错误、引导不当行为并净化观测结果。在具有挑战性的GAIA基准测试中，SupervisorAgent在不牺牲成功率的情况下，平均减少了Smolagent框架的令牌消耗29.45%。广泛实验表明，我们的方法具有广泛的适用性和鲁棒性，在五个额外的基准测试（数学推理、代码生成和问答）以及多种最先进的基础模型上得到了验证。代码可在此处获取：this https URL。 

---
# InfoFlow: Reinforcing Search Agent Via Reward Density Optimization 

**Title (ZH)**: InfoFlow：通过奖励密度优化强化搜索代理 

**Authors**: Kun Luo, Hongjin Qian, Zheng Liu, Ziyi Xia, Shitao Xiao, Siqi Bao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26575)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a promising approach for enhancing agentic deep search. However, its application is often hindered by low \textbf{Reward Density} in deep search scenarios, where agents expend significant exploratory costs for infrequent and often null final rewards. In this paper, we formalize this challenge as the \textbf{Reward Density Optimization} problem, which aims to improve the reward obtained per unit of exploration cost. This paper introduce \textbf{InfoFlow}, a systematic framework that tackles this problem from three aspects. 1) \textbf{Subproblem decomposition}: breaking down long-range tasks to assign process rewards, thereby providing denser learning signals. 2) \textbf{Failure-guided hints}: injecting corrective guidance into stalled trajectories to increase the probability of successful outcomes. 3) \textbf{Dual-agent refinement}: employing a dual-agent architecture to offload the cognitive burden of deep exploration. A refiner agent synthesizes the search history, which effectively compresses the researcher's perceived trajectory, thereby reducing exploration cost and increasing the overall reward density. We evaluate InfoFlow on multiple agentic search benchmarks, where it significantly outperforms strong baselines, enabling lightweight LLMs to achieve performance comparable to advanced proprietary LLMs. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）：提升代理深度搜索的有效方法及其应用中的奖励密度优化（InfoFlow框架） 

---
# Multiclass Local Calibration With the Jensen-Shannon Distance 

**Title (ZH)**: 多类局部校准与詹森-沙伦距离 

**Authors**: Cesare Barbera, Lorenzo Perini, Giovanni De Toni, Andrea Passerini, Andrea Pugnana  

**Link**: [PDF](https://arxiv.org/pdf/2510.26566)  

**Abstract**: Developing trustworthy Machine Learning (ML) models requires their predicted probabilities to be well-calibrated, meaning they should reflect true-class frequencies. Among calibration notions in multiclass classification, strong calibration is the most stringent, as it requires all predicted probabilities to be simultaneously calibrated across all classes. However, existing approaches to multiclass calibration lack a notion of distance among inputs, which makes them vulnerable to proximity bias: predictions in sparse regions of the feature space are systematically miscalibrated. This is especially relevant in high-stakes settings, such as healthcare, where the sparse instances are exactly those most at risk of biased treatment. In this work, we address this main shortcoming by introducing a local perspective on multiclass calibration. First, we formally define multiclass local calibration and establish its relationship with strong calibration. Second, we theoretically analyze the pitfalls of existing evaluation metrics when applied to multiclass local calibration. Third, we propose a practical method for enhancing local calibration in Neural Networks, which enforces alignment between predicted probabilities and local estimates of class frequencies using the Jensen-Shannon distance. Finally, we empirically validate our approach against existing multiclass calibration techniques. 

**Abstract (ZH)**: 开发值得信赖的机器学习（ML）模型要求其预测概率准确反映真实类频率。在多类分类的校准概念中，强校准是最严格的，因为它要求所有预测概率在所有类中同时得到校准。然而，现有的多类校准方法缺乏输入之间的距离观念，这使它们容易受到邻近偏差的影响：特征空间中稀疏区域的预测是系统地失校准的。这一点在诸如医疗保健这样高风险的场景中尤为重要，因为在这些场景中，稀疏实例正是那些最有可能受到偏差对待的实例。在本文中，我们通过引入多类校准的局部视角来解决这一主要不足。首先，我们正式定义多类局部校准并建立其与强校准的关系。其次，我们理论上分析现有评估指标在应用于多类局部校准时的缺陷。第三，我们提出了一种实用的方法来增强神经网络中的局部校准，该方法使用Jensen-Shannon距离强制预测概率与局部类频率估计之间的对齐。最后，我们实证验证了我们的方法相对于现有多类校准技术的有效性。 

---
# Adaptive Inverse Kinematics Framework for Learning Variable-Length Tool Manipulation in Robotics 

**Title (ZH)**: 适应性逆运动学框架：学习机器人中可变长度工具操作 

**Authors**: Prathamesh Kothavale, Sravani Boddepalli  

**Link**: [PDF](https://arxiv.org/pdf/2510.26551)  

**Abstract**: Conventional robots possess a limited understanding of their kinematics and are confined to preprogrammed tasks, hindering their ability to leverage tools efficiently. Driven by the essential components of tool usage - grasping the desired outcome, selecting the most suitable tool, determining optimal tool orientation, and executing precise manipulations - we introduce a pioneering framework. Our novel approach expands the capabilities of the robot's inverse kinematics solver, empowering it to acquire a sequential repertoire of actions using tools of varying lengths. By integrating a simulation-learned action trajectory with the tool, we showcase the practicality of transferring acquired skills from simulation to real-world scenarios through comprehensive experimentation. Remarkably, our extended inverse kinematics solver demonstrates an impressive error rate of less than 1 cm. Furthermore, our trained policy achieves a mean error of 8 cm in simulation. Noteworthy, our model achieves virtually indistinguishable performance when employing two distinct tools of different lengths. This research provides an indication of potential advances in the exploration of all four fundamental aspects of tool usage, enabling robots to master the intricate art of tool manipulation across diverse tasks. 

**Abstract (ZH)**: 传统机器人对自身运动学的理解有限，只能执行预编程任务，限制了它们高效利用工具的能力。基于工具使用的核心组成部分——抓取所需的成果、选择最适合的工具、确定最佳工具姿态以及执行精确操作——我们提出了一种开创性框架。我们的新型方法扩展了机器人逆运动学解算器的功能，使其能够使用不同长度的工具获取一系列连续的操作。通过将模拟学习的动作轨迹与工具结合，我们展示了将从模拟中获得的技能转移到实际场景中的可行性，通过全面的实验进行了证明。值得注意的是，我们的扩展逆运动学解算器表现出小于1 cm的显著低误差率。此外，我们的训练策略在模拟中的平均误差为8 cm。特别地，当使用两种不同长度的工具时，我们的模型实现了几乎不可区分的性能。这项研究为探索工具使用的所有四个基本方面提供了可能的进步方向，使机器人能够掌握不同任务中复杂的工具操作艺术。 

---
# The Structure of Relation Decoding Linear Operators in Large Language Models 

**Title (ZH)**: 大型语言模型中关系解码线性运算子的结构 

**Authors**: Miranda Anna Christ, Adrián Csiszárik, Gergely Becsó, Dániel Varga  

**Link**: [PDF](https://arxiv.org/pdf/2510.26543)  

**Abstract**: This paper investigates the structure of linear operators introduced in Hernandez et al. [2023] that decode specific relational facts in transformer language models. We extend their single-relation findings to a collection of relations and systematically chart their organization. We show that such collections of relation decoders can be highly compressed by simple order-3 tensor networks without significant loss in decoding accuracy. To explain this surprising redundancy, we develop a cross-evaluation protocol, in which we apply each linear decoder operator to the subjects of every other relation. Our results reveal that these linear maps do not encode distinct relations, but extract recurring, coarse-grained semantic properties (e.g., country of capital city and country of food are both in the country-of-X property). This property-centric structure clarifies both the operators' compressibility and highlights why they generalize only to new relations that are semantically close. Our findings thus interpret linear relational decoding in transformer language models as primarily property-based, rather than relation-specific. 

**Abstract (ZH)**: 本文探讨了Hernandez等人在[2023]中引入的线性算子结构，这些算子用于解码变压器语言模型中的特定关系事实。我们扩展了他们关于单关系的研究，系统地梳理了多关系的组织结构。我们展示了这样的关系解码集合可以通过简单的阶数为3的张量网络进行高度压缩，而不会显著损失解码精度。为了解释这种令人惊讶的冗余性，我们开发了一种跨评估协议，在该协议中，我们对每个关系的应用主体使用每种线性解码算子。研究结果表明，这些线性映射并未编码独立的关系，而是抽取了反复出现的、粗粒度的语义特性（例如，首都所在的国家与食物所在的国家都归属于X所在的国家这一语义属性）。这种以特性为中心的结构不仅阐明了操作符的压缩性，还突显了它们仅能泛化到语义上接近的新关系的原因。因此，本文的研究结果将变压器语言模型中的线性关系解码主要解释为特性基于的，而非特定关系的。 

---
# Inside CORE-KG: Evaluating Structured Prompting and Coreference Resolution for Knowledge Graphs 

**Title (ZH)**: CORE-KG内部：结构化提示和共指消解在知识图谱评估中的应用 

**Authors**: Dipak Meher, Carlotta Domeniconi  

**Link**: [PDF](https://arxiv.org/pdf/2510.26512)  

**Abstract**: Human smuggling networks are increasingly adaptive and difficult to analyze. Legal case documents offer critical insights but are often unstructured, lexically dense, and filled with ambiguous or shifting references, which pose significant challenges for automated knowledge graph (KG) construction. While recent LLM-based approaches improve over static templates, they still generate noisy, fragmented graphs with duplicate nodes due to the absence of guided extraction and coreference resolution. The recently proposed CORE-KG framework addresses these limitations by integrating a type-aware coreference module and domain-guided structured prompts, significantly reducing node duplication and legal noise. In this work, we present a systematic ablation study of CORE-KG to quantify the individual contributions of its two key components. Our results show that removing coreference resolution results in a 28.32% increase in node duplication and a 4.32% increase in noisy nodes, while removing structured prompts leads to a 4.34% increase in node duplication and a 73.33% increase in noisy nodes. These findings offer empirical insights for designing robust LLM-based pipelines for extracting structured representations from complex legal texts. 

**Abstract (ZH)**: 人类走私网络日益具有适应性且难以分析。法律案件文件提供了关键见解，但这些文件往往结构不一、词汇密集，并且充满了模糊或变化的引用，这对自动知识图谱（KG）构建构成了重大挑战。尽管基于近年来大语言模型（LLM）的方法在静态模板上有所改进，但由于缺乏引导式提取和共指解析，它们仍会产生嘈杂、碎片化的图谱，其中包含重复节点。最近提出的CORE-KG框架通过整合类型意识的共指模块和领域导向的结构化提示，显著减少了节点重复和法律噪声。在本文中，我们进行了一项系统的消融研究，以量化其两个关键组件的单独贡献。我们的结果表明，去除共指解析会导致节点重复增加28.32%，噪音节点增加4.32%，而去除结构化提示会导致节点重复增加4.34%，噪音节点增加73.33%。这些发现提供了实证见解，用于设计从复杂法律文本中提取结构化表示的健壮的大语言模型（LLM）管道。 

---
# Simulating and Experimenting with Social Media Mobilization Using LLM Agents 

**Title (ZH)**: 使用大规模语言模型代理模拟和实验社交媒体动员 

**Authors**: Sadegh Shirani, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2510.26494)  

**Abstract**: Online social networks have transformed the ways in which political mobilization messages are disseminated, raising new questions about how peer influence operates at scale. Building on the landmark 61-million-person Facebook experiment \citep{bond201261}, we develop an agent-based simulation framework that integrates real U.S. Census demographic distributions, authentic Twitter network topology, and heterogeneous large language model (LLM) agents to examine the effect of mobilization messages on voter turnout. Each simulated agent is assigned demographic attributes, a personal political stance, and an LLM variant (\texttt{GPT-4.1}, \texttt{GPT-4.1-Mini}, or \texttt{GPT-4.1-Nano}) reflecting its political sophistication. Agents interact over realistic social network structures, receiving personalized feeds and dynamically updating their engagement behaviors and voting intentions. Experimental conditions replicate the informational and social mobilization treatments of the original Facebook study. Across scenarios, the simulator reproduces qualitative patterns observed in field experiments, including stronger mobilization effects under social message treatments and measurable peer spillovers. Our framework provides a controlled, reproducible environment for testing counterfactual designs and sensitivity analyses in political mobilization research, offering a bridge between high-validity field experiments and flexible computational modeling.\footnote{Code and data available at this https URL} 

**Abstract (ZH)**: 在线社交网络已改变了政治动员信息的传播方式，提出了关于大规模环境下同伴影响如何运作的新问题。基于具有里程碑意义的6100万用户Facebook实验\[1\]，我们构建了一个基于代理的模拟框架，该框架集成了真实的美国人口普查人口统计分布、真实的Twitter网络拓扑以及异质大型语言模型（LLM）代理，以考察动员信息对投票率的影响。每个模拟代理被赋予人口统计属性、个人政治立场，并且具有反映其政治精明程度的不同版本的LLM（\texttt{GPT-4.1}、\texttt{GPT-4.1-Mini}或\texttt{GPT-4.1-Nano}）。代理人在现实的社会网络结构上相互作用，接收个性化的内容并动态更新他们的参与行为和投票意向。实验条件复制了原始Facebook研究中的信息和社交动员干预措施。在不同的情景下，模拟器再现了实地实验中的定性模式，包括在社交信息干预下更强的动员效果以及可测量的同伴溢出效应。该框架为政治动员研究中测试反事实设计和敏感性分析提供了一个受控且可再现的环境，架起了高有效性的实地实验与灵活的计算建模之间的桥梁。\footnote{代码和数据可从此链接获取}。 

---
# Bayesian Network Fusion of Large Language Models for Sentiment Analysis 

**Title (ZH)**: 大规模语言模型的贝叶斯网络融合情感分析 

**Authors**: Rasoul Amirzadeh, Dhananjay Thiruvady, Fatemeh Shiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.26484)  

**Abstract**: Large language models (LLMs) continue to advance, with an increasing number of domain-specific variants tailored for specialised tasks. However, these models often lack transparency and explainability, can be costly to fine-tune, require substantial prompt engineering, yield inconsistent results across domains, and impose significant adverse environmental impact due to their high computational demands. To address these challenges, we propose the Bayesian network LLM fusion (BNLF) framework, which integrates predictions from three LLMs, including FinBERT, RoBERTa, and BERTweet, through a probabilistic mechanism for sentiment analysis. BNLF performs late fusion by modelling the sentiment predictions from multiple LLMs as probabilistic nodes within a Bayesian network. Evaluated across three human-annotated financial corpora with distinct linguistic and contextual characteristics, BNLF demonstrates consistent gains of about six percent in accuracy over the baseline LLMs, underscoring its robustness to dataset variability and the effectiveness of probabilistic fusion for interpretable sentiment classification. 

**Abstract (ZH)**: 大型语言模型（LLMs）不断发展，出现了越来越多针对特定领域的变体以适应特定任务。然而，这些模型往往缺乏透明度和解释性，调优成本高，需要大量的提示工程，跨领域的结果一致性差，并且由于其高度的计算需求对环境造成显著的负面影响。为了解决这些挑战，我们提出了一种贝叶斯网络LLM融合（BNLF）框架，该框架通过概率机制集成来自FinBERT、RoBERTa和BERTweet等三个LLM的情感分析预测。BNLF通过将多个LLM的情感预测建模为贝叶斯网络中的概率节点来进行晚期融合。在三项人工标注的金融语料库上进行评估，这些语料库具有不同的语言和上下文特征，BNLF在准确率上比基线LLM平均提高了约六个百分点，这表明其对数据集变异性的鲁棒性和概率融合在具有解释性的情感分类中的有效性。 

---
# Counteracting Matthew Effect in Self-Improvement of LVLMs through Head-Tail Re-balancing 

**Title (ZH)**: 通过头部-尾部重新平衡对抗LVLMs的马太效应在其自我改进中的影响 

**Authors**: Xin Guo, Zhiheng Xi, Yiwen Ding, Yitao Zhai, Xiaowei Shi, Xunliang Cai, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26474)  

**Abstract**: Self-improvement has emerged as a mainstream paradigm for advancing the reasoning capabilities of large vision-language models (LVLMs), where models explore and learn from successful trajectories iteratively. However, we identify a critical issue during this process: the model excels at generating high-quality trajectories for simple queries (i.e., head data) but struggles with more complex ones (i.e., tail data). This leads to an imbalanced optimization that drives the model to prioritize simple reasoning skills, while hindering its ability to tackle more complex reasoning tasks. Over iterations, this imbalance becomes increasingly pronounced--a dynamic we term the "Matthew effect"--which ultimately hinders further model improvement and leads to performance bottlenecks. To counteract this challenge, we introduce four efficient strategies from two perspectives: distribution-reshaping and trajectory-resampling, to achieve head-tail re-balancing during the exploration-and-learning self-improvement process. Extensive experiments on Qwen2-VL-7B-Instruct and InternVL2.5-4B models across visual reasoning tasks demonstrate that our methods consistently improve visual reasoning capabilities, outperforming vanilla self-improvement by 3.86 points on average. 

**Abstract (ZH)**: 自提高已成为提升大型视觉-语言模型推理能力的主要范式，其中模型通过迭代探索和学习成功的轨迹。然而，在这一过程中我们发现一个关键问题：模型在生成简单查询（即头部数据）的高质量轨迹方面表现出色，但在处理更复杂的查询（即尾部数据）方面却力不从心。这导致优化不平衡，促使模型优先发展简单的推理技能，而削弱其应对更复杂推理任务的能力。随着迭代次数的增加，这种不平衡变得越来越显著——我们称这一现象为“马太效应”——最终阻碍了模型进一步改进并导致性能瓶颈。为应对这一挑战，我们从两个角度引入了四种有效的策略：分布重塑和轨迹重采样，以在探索和学习的自提高过程中实现头部与尾部的平衡。在Qwen2-VL-7B-Instruct和InternVL2.5-4B模型上针对视觉推理任务进行的大量实验表明，我们的方法在视觉推理能力上始终表现出改进，平均优于 vanilla 自提高3.86分。 

---
# SecureReviewer: Enhancing Large Language Models for Secure Code Review through Secure-aware Fine-tuning 

**Title (ZH)**: SecureReviewer：通过安全意识微调提升大型语言模型的代码安全审查能力 

**Authors**: Fang Liu, Simiao Liu, Yinghao Zhu, Xiaoli Lian, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26457)  

**Abstract**: Identifying and addressing security issues during the early phase of the development lifecycle is critical for mitigating the long-term negative impacts on software systems. Code review serves as an effective practice that enables developers to check their teammates' code before integration into the codebase. To streamline the generation of review comments, various automated code review approaches have been proposed, where LLM-based methods have significantly advanced the capabilities of automated review generation. However, existing models primarily focus on general-purpose code review, their effectiveness in identifying and addressing security-related issues remains underexplored. Moreover, adapting existing code review approaches to target security issues faces substantial challenges, including data scarcity and inadequate evaluation metrics. To address these limitations, we propose SecureReviewer, a new approach designed for enhancing LLMs' ability to identify and resolve security-related issues during code review. Specifically, we first construct a dataset tailored for training and evaluating secure code review capabilities. Leveraging this dataset, we fine-tune LLMs to generate code review comments that can effectively identify security issues and provide fix suggestions with our proposed secure-aware fine-tuning strategy. To mitigate hallucination in LLMs and enhance the reliability of their outputs, we integrate the RAG technique, which grounds the generated comments in domain-specific security knowledge. Additionally, we introduce SecureBLEU, a new evaluation metric designed to assess the effectiveness of review comments in addressing security issues. Experimental results demonstrate that SecureReviewer outperforms state-of-the-art baselines in both security issue detection accuracy and the overall quality and practical utility of generated review comments. 

**Abstract (ZH)**: 在开发生命周期早期阶段识别和解决安全问题对于减轻长期负面影响至关重要。代码审查作为一种有效实践，使开发人员能够在代码集成到代码库之前检查队友的代码。为了简化审查评论的生成，已经提出了各种自动化代码审查方法，其中基于LLM的方法显著提升了自动化审查生成的能力。然而，现有模型主要集中在通用代码审查上，它们在识别和解决与安全相关问题方面的有效性尚待探索。此外，将现有代码审查方法调整为专门针对安全问题面临巨大挑战，包括数据稀缺性和不够完善的评估指标。为了解决这些局限性，我们提出了一种名为SecureReviewer的新方法，旨在增强LLM在代码审查过程中识别和解决安全问题的能力。具体来说，我们首先构建了一个专门用于训练和评估安全代码审查能力的数据集。利用该数据集，我们通过我们提出的安全意识微调策略对LLM进行微调，使其能够生成有效识别安全问题并提供修复建议的审查评论。为了减轻LLM的幻觉现象并提高其输出的可靠性，我们集成了一种RAG技术，使生成的评论基于特定领域的安全知识。此外，我们引入了一种新的评估指标SecureBLEU，旨在评估审查评论在解决安全问题方面的有效性。实验结果表明，SecureReviewer在安全问题检测精度以及生成的审查评论的整体质量与实用价值方面均优于现有先进baseline方法。 

---
# Robust Graph Condensation via Classification Complexity Mitigation 

**Title (ZH)**: 分类复杂性减轻下的鲁棒图凝聚 

**Authors**: Jiayi Luo, Qingyun Sun, Beining Yang, Haonan Yuan, Xingcheng Fu, Yanbiao Ma, Jianxin Li, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26451)  

**Abstract**: Graph condensation (GC) has gained significant attention for its ability to synthesize smaller yet informative graphs. However, existing studies often overlook the robustness of GC in scenarios where the original graph is corrupted. In such cases, we observe that the performance of GC deteriorates significantly, while existing robust graph learning technologies offer only limited effectiveness. Through both empirical investigation and theoretical analysis, we reveal that GC is inherently an intrinsic-dimension-reducing process, synthesizing a condensed graph with lower classification complexity. Although this property is critical for effective GC performance, it remains highly vulnerable to adversarial perturbations. To tackle this vulnerability and improve GC robustness, we adopt the geometry perspective of graph data manifold and propose a novel Manifold-constrained Robust Graph Condensation framework named MRGC. Specifically, we introduce three graph data manifold learning modules that guide the condensed graph to lie within a smooth, low-dimensional manifold with minimal class ambiguity, thereby preserving the classification complexity reduction capability of GC and ensuring robust performance under universal adversarial attacks. Extensive experiments demonstrate the robustness of \ModelName\ across diverse attack scenarios. 

**Abstract (ZH)**: 图凝缩（GC）由于其生成更小但具有信息性的图形的能力而得到了广泛关注。然而，现有研究往往忽视了原图被破坏时GC的鲁棒性。在这种情况下，我们观察到GC的性能显著下降，而现有的鲁棒图学习技术则只能提供有限的有效性。通过实证研究和理论分析，我们揭示了GC本质上是一个固有维数降低的过程，生成了一个分类复杂性较低的凝缩图。尽管这一特性对于有效GC性能至关重要，但它仍然对对抗性扰动高度脆弱。为了应对这种脆弱性并提高GC的鲁棒性，我们从图数据流形的几何视角出发，提出了一种新的流形约束鲁棒图凝缩框架，命名为MRGC。具体而言，我们引入了三个图数据流形学习模块，引导凝缩图位于一个光滑的低维流形上，具有最小的类别模糊性，从而保留了GC的分类复杂性降低能力，并在普遍对抗性攻击下保证了稳健的性能。广泛的实验表明，\ModelName\在各种攻击场景下具有鲁棒性。 

---
# Personalized Treatment Outcome Prediction from Scarce Data via Dual-Channel Knowledge Distillation and Adaptive Fusion 

**Title (ZH)**: 基于双通道知识蒸馏和自适应融合的稀缺数据个性化治疗效果预测 

**Authors**: Wenjie Chen, Li Zhuang, Ziying Luo, Yu Liu, Jiahao Wu, Shengcai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26444)  

**Abstract**: Personalized treatment outcome prediction based on trial data for small-sample and rare patient groups is critical in precision medicine. However, the costly trial data limit the prediction performance. To address this issue, we propose a cross-fidelity knowledge distillation and adaptive fusion network (CFKD-AFN), which leverages abundant but low-fidelity simulation data to enhance predictions on scarce but high-fidelity trial data. CFKD-AFN incorporates a dual-channel knowledge distillation module to extract complementary knowledge from the low-fidelity model, along with an attention-guided fusion module to dynamically integrate multi-source information. Experiments on treatment outcome prediction for the chronic obstructive pulmonary disease demonstrates significant improvements of CFKD-AFN over state-of-the-art methods in prediction accuracy, ranging from 6.67\% to 74.55\%, and strong robustness to varying high-fidelity dataset sizes. Furthermore, we extend CFKD-AFN to an interpretable variant, enabling the exploration of latent medical semantics to support clinical decision-making. 

**Abstract (ZH)**: 基于试验数据的个性化治疗效果预测对于小样本和稀有患者群体至关重要，是精准医学中的关键任务。然而，昂贵的试验数据限制了预测性能。为此，我们提出了一种跨保真度知识蒸馏和自适应融合网络（CFKD-AFN），利用丰富的但保真度较低的模拟数据来增强对稀少但保真度较高的试验数据的预测。CFKD-AFN 包含一个双通道知识蒸馏模块，用于从低保真度模型中提取互补知识，以及一个基于注意力的融合模块，用于动态整合多源信息。慢性阻塞性肺病治疗效果预测实验展示了 CFKD-AFN 在预测准确性方面比现有方法显著提高，范围从 6.67% 到 74.55%，并且在不同大小的高保真度数据集上表现出强大的稳健性。此外，我们将 CFKD-AFN 扩展为一种可解释的变体，以探索潜在的医学语义来支持临床决策。 

---
# SSCL-BW: Sample-Specific Clean-Label Backdoor Watermarking for Dataset Ownership Verification 

**Title (ZH)**: SSCL-BW: 样本特定的干净标签后门水印方法用于数据集所有权验证 

**Authors**: Yingjia Wang, Ting Qiao, Xing Liu, Chongzuo Li, Sixing Wu, Jianbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.26420)  

**Abstract**: The rapid advancement of deep neural networks (DNNs) heavily relies on large-scale, high-quality datasets. However, unauthorized commercial use of these datasets severely violates the intellectual property rights of dataset owners. Existing backdoor-based dataset ownership verification methods suffer from inherent limitations: poison-label watermarks are easily detectable due to label inconsistencies, while clean-label watermarks face high technical complexity and failure on high-resolution images. Moreover, both approaches employ static watermark patterns that are vulnerable to detection and removal. To address these issues, this paper proposes a sample-specific clean-label backdoor watermarking (i.e., SSCL-BW). By training a U-Net-based watermarked sample generator, this method generates unique watermarks for each sample, fundamentally overcoming the vulnerability of static watermark patterns. The core innovation lies in designing a composite loss function with three components: target sample loss ensures watermark effectiveness, non-target sample loss guarantees trigger reliability, and perceptual similarity loss maintains visual imperceptibility. During ownership verification, black-box testing is employed to check whether suspicious models exhibit predefined backdoor behaviors. Extensive experiments on benchmark datasets demonstrate the effectiveness of the proposed method and its robustness against potential watermark removal attacks. 

**Abstract (ZH)**: 基于样本特定的干净标签后门水印方法（即SSCL-BW） 

---
# LoCoT2V-Bench: A Benchmark for Long-Form and Complex Text-to-Video Generation 

**Title (ZH)**: LoCoT2V-Bench: 一个长文本和复杂文本生成视频的基准 

**Authors**: Xiangqing Zheng, Chengyue Wu, Kehai Chen, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26412)  

**Abstract**: Recently text-to-video generation has made impressive progress in producing short, high-quality clips, but evaluating long-form outputs remains a major challenge especially when processing complex prompts. Existing benchmarks mostly rely on simplified prompts and focus on low-level metrics, overlooking fine-grained alignment with prompts and abstract dimensions such as narrative coherence and thematic expression. To address these gaps, we propose LoCoT2V-Bench, a benchmark specifically designed for long video generation (LVG) under complex input conditions. Based on various real-world videos, LoCoT2V-Bench introduces a suite of realistic and complex prompts incorporating elements like scene transitions and event dynamics. Moreover, it constructs a multi-dimensional evaluation framework that includes our newly proposed metrics such as event-level alignment, fine-grained temporal consistency, content clarity, and the Human Expectation Realization Degree (HERD) that focuses on more abstract attributes like narrative flow, emotional response, and character development. Using this framework, we conduct a comprehensive evaluation of nine representative LVG models, finding that while current methods perform well on basic visual and temporal aspects, they struggle with inter-event consistency, fine-grained alignment, and high-level thematic adherence, etc. Overall, LoCoT2V-Bench provides a comprehensive and reliable platform for evaluating long-form complex text-to-video generation and highlights critical directions for future method improvement. 

**Abstract (ZH)**: LoCoT2V-Bench：一种专门针对复杂输入条件下长视频生成的基准 

---
# Human-in-the-loop Online Rejection Sampling for Robotic Manipulation 

**Title (ZH)**: 带有人类在环的在线拒绝采样用于机器人操作 

**Authors**: Guanxing Lu, Rui Zhao, Haitao Lin, He Zhang, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26406)  

**Abstract**: Reinforcement learning (RL) is widely used to produce robust robotic manipulation policies, but fine-tuning vision-language-action (VLA) models with RL can be unstable due to inaccurate value estimates and sparse supervision at intermediate steps. In contrast, imitation learning (IL) is easy to train but often underperforms due to its offline nature. In this paper, we propose Hi-ORS, a simple yet effective post-training method that utilizes rejection sampling to achieve both training stability and high robustness. Hi-ORS stabilizes value estimation by filtering out negatively rewarded samples during online fine-tuning, and adopts a reward-weighted supervised training objective to provide dense intermediate-step supervision. For systematic study, we develop an asynchronous inference-training framework that supports flexible online human-in-the-loop corrections, which serve as explicit guidance for learning error-recovery behaviors. Across three real-world tasks and two embodiments, Hi-ORS fine-tunes a pi-base policy to master contact-rich manipulation in just 1.5 hours of real-world training, outperforming RL and IL baselines by a substantial margin in both effectiveness and efficiency. Notably, the fine-tuned policy exhibits strong test-time scalability by reliably executing complex error-recovery behaviors to achieve better performance. 

**Abstract (ZH)**: 基于拒绝采样的稳健后训练方法Hi-ORS：实现稳定性和鲁棒性的平衡 

---
# SPG-CDENet: Spatial Prior-Guided Cross Dual Encoder Network for Multi-Organ Segmentation 

**Title (ZH)**: SPG-CDENet：空间先验引导的跨模态双编码器网络用于多器官分割 

**Authors**: Xizhi Tian, Changjun Zhou, Yulin. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26390)  

**Abstract**: Multi-organ segmentation is a critical task in computer-aided diagnosis. While recent deep learning methods have achieved remarkable success in image segmentation, huge variations in organ size and shape challenge their effectiveness in multi-organ segmentation. To address these challenges, we propose a Spatial Prior-Guided Cross Dual Encoder Network (SPG-CDENet), a novel two-stage segmentation paradigm designed to improve multi-organ segmentation accuracy. Our SPG-CDENet consists of two key components: a spatial prior network and a cross dual encoder network. The prior network generates coarse localization maps that delineate the approximate ROI, serving as spatial guidance for the dual encoder network. The cross dual encoder network comprises four essential components: a global encoder, a local encoder, a symmetric cross-attention module, and a flow-based decoder. The global encoder captures global semantic features from the entire image, while the local encoder focuses on features from the prior network. To enhance the interaction between the global and local encoders, a symmetric cross-attention module is proposed across all layers of the encoders to fuse and refine features. Furthermore, the flow-based decoder directly propagates high-level semantic features from the final encoder layer to all decoder layers, maximizing feature preservation and utilization. Extensive qualitative and quantitative experiments on two public datasets demonstrate the superior performance of SPG-CDENet compared to existing segmentation methods. Furthermore, ablation studies further validate the effectiveness of the proposed modules in improving segmentation accuracy. 

**Abstract (ZH)**: 多器官分割是计算机辅助诊断中的关键任务。尽管近年来深度学习方法在图像分割方面取得了显著成功，但器官大小和形状的巨大变化挑战了其在多器官分割中的有效性。为了解决这些挑战，我们提出了一种空间先验引导交叉双编码器网络（SPG-CDENet），这是一种新颖的两阶段分割范式，旨在提高多器官分割的准确性。 

---
# The Geometry of Dialogue: Graphing Language Models to Reveal Synergistic Teams for Multi-Agent Collaboration 

**Title (ZH)**: 对话的几何学：绘制语言模型以揭示多代理协作的协同团队 

**Authors**: Kotaro Furuya, Yuichi Kitagawa  

**Link**: [PDF](https://arxiv.org/pdf/2510.26352)  

**Abstract**: While a multi-agent approach based on large language models (LLMs) represents a promising strategy to surpass the capabilities of single models, its success is critically dependent on synergistic team composition. However, forming optimal teams is a significant challenge, as the inherent opacity of most models obscures the internal characteristics necessary for effective collaboration. In this paper, we propose an interaction-centric framework for automatic team composition that does not require any prior knowledge including their internal architectures, training data, or task performances. Our method constructs a "language model graph" that maps relationships between models from the semantic coherence of pairwise conversations, and then applies community detection to identify synergistic model clusters. Our experiments with diverse LLMs demonstrate that the proposed method discovers functionally coherent groups that reflect their latent specializations. Priming conversations with specific topics identified synergistic teams which outperform random baselines on downstream benchmarks and achieve comparable accuracy to that of manually-curated teams based on known model specializations. Our findings provide a new basis for the automated design of collaborative multi-agent LLM teams. 

**Abstract (ZH)**: 基于大语言模型的多agent方法中自动生成协同团队的交互中心框架 

---
# Reinforcement Learning for Pollution Detection in a Randomized, Sparse and Nonstationary Environment with an Autonomous Underwater Vehicle 

**Title (ZH)**: 自主水下车辆在随机、稀疏且非稳定环境中的污染检测强化学习 

**Authors**: Sebastian Zieglmeier, Niklas Erdmann, Narada D. Warakagoda  

**Link**: [PDF](https://arxiv.org/pdf/2510.26347)  

**Abstract**: Reinforcement learning (RL) algorithms are designed to optimize problem-solving by learning actions that maximize rewards, a task that becomes particularly challenging in random and nonstationary environments. Even advanced RL algorithms are often limited in their ability to solve problems in these conditions. In applications such as searching for underwater pollution clouds with autonomous underwater vehicles (AUVs), RL algorithms must navigate reward-sparse environments, where actions frequently result in a zero reward. This paper aims to address these challenges by revisiting and modifying classical RL approaches to efficiently operate in sparse, randomized, and nonstationary environments. We systematically study a large number of modifications, including hierarchical algorithm changes, multigoal learning, and the integration of a location memory as an external output filter to prevent state revisits. Our results demonstrate that a modified Monte Carlo-based approach significantly outperforms traditional Q-learning and two exhaustive search patterns, illustrating its potential in adapting RL to complex environments. These findings suggest that reinforcement learning approaches can be effectively adapted for use in random, nonstationary, and reward-sparse environments. 

**Abstract (ZH)**: 强化学习（RL）算法旨在通过学习最大化回报的动作来优化问题解决，在随机和非平稳环境下这一任务变得尤为挑战。即使是最先进的RL算法，在这些条件下也往往难以解决问题。在使用自主水下车辆（AUVs）搜索水下污染云这样的应用中，RL算法必须在回报稀疏的环境中导航，其中频繁执行的动作导致零回报。本文旨在通过重新审视和修改经典RL方法，以更有效地在稀疏、随机和非平稳环境中操作。我们系统地研究了大量修改方法，包括层次算法的改变、多目标学习以及将位置记忆作为外部输出滤波器集成以防止状态重访。我们的结果表明，修改后的基于蒙特卡洛的方法显著优于传统的Q学习和两种详尽搜索模式，展示了其适应复杂环境的潜力。这些发现表明，强化学习方法可以有效适应随机、非平稳和回报稀疏的环境。 

---
# MisSynth: Improving MISSCI Logical Fallacies Classification with Synthetic Data 

**Title (ZH)**: MisSynth: 通过合成数据提高MISSCI逻辑谬误分类效果 

**Authors**: Mykhailo Poliakov, Nadiya Shvai  

**Link**: [PDF](https://arxiv.org/pdf/2510.26345)  

**Abstract**: Health-related misinformation is very prevalent and potentially harmful. It is difficult to identify, especially when claims distort or misinterpret scientific findings. We investigate the impact of synthetic data generation and lightweight fine-tuning techniques on the ability of large language models (LLMs) to recognize fallacious arguments using the MISSCI dataset and framework. In this work, we propose MisSynth, a pipeline that applies retrieval-augmented generation (RAG) to produce synthetic fallacy samples, which are then used to fine-tune an LLM model. Our results show substantial accuracy gains with fine-tuned models compared to vanilla baselines. For instance, the LLaMA 3.1 8B fine-tuned model achieved an over 35% F1-score absolute improvement on the MISSCI test split over its vanilla baseline. We demonstrate that introducing synthetic fallacy data to augment limited annotated resources can significantly enhance zero-shot LLM classification performance on real-world scientific misinformation tasks, even with limited computational resources. The code and synthetic dataset are available on this https URL. 

**Abstract (ZH)**: 健康相关的 misinformation 非常普遍且可能具有危害性。由于断言可能歪曲或误解科学研究成果，因此很难识别。我们利用MISSCI数据集和框架，研究合成数据生成和轻量级微调技术对大型语言模型（LLMs）识别谬误论点能力的影响。在这项工作中，我们提出了一种名为MisSynth的管道，该管道利用检索增强生成（RAG）技术生成合成谬误样本，然后将这些样本用于微调LLM模型。结果显示，微调后的模型相对于传统的基线模型在准确率上有了显著提升。例如，LLaMA 3.1 8B微调模型在MISSCI测试集上的F1分数绝对提高了超过35%。我们证明，即使在计算资源有限的情况下，引入合成谬误数据以补充有限的标注资源，也能显著提升零样本LLM在实际科学 misinformation任务中的分类性能。相关代码和合成数据集可从此链接获取。 

---
# Linear Causal Discovery with Interventional Constraints 

**Title (ZH)**: 具有干预约束的线性因果发现 

**Authors**: Zhigao Guo, Feng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.26342)  

**Abstract**: Incorporating causal knowledge and mechanisms is essential for refining causal models and improving downstream tasks such as designing new treatments. In this paper, we introduce a novel concept in causal discovery, termed interventional constraints, which differs fundamentally from interventional data. While interventional data require direct perturbations of variables, interventional constraints encode high-level causal knowledge in the form of inequality constraints on causal effects. For instance, in the Sachs dataset (Sachs et al.\ 2005), Akt has been shown to be activated by PIP3, meaning PIP3 exerts a positive causal effect on Akt. Existing causal discovery methods allow enforcing structural constraints (for example, requiring a causal path from PIP3 to Akt), but they may still produce incorrect causal conclusions such as learning that "PIP3 inhibits Akt". Interventional constraints bridge this gap by explicitly constraining the total causal effect between variable pairs, ensuring learned models respect known causal influences. To formalize interventional constraints, we propose a metric to quantify total causal effects for linear causal models and formulate the problem as a constrained optimization task, solved using a two-stage constrained optimization method. We evaluate our approach on real-world datasets and demonstrate that integrating interventional constraints not only improves model accuracy and ensures consistency with established findings, making models more explainable, but also facilitates the discovery of new causal relationships that would otherwise be costly to identify. 

**Abstract (ZH)**: 将干预因果知识和机制融入因果模型对于细化因果模型并改善下游任务（如设计新治疗方法）至关重要。本文引入了一种新颖的因果发现概念，称为干预约束，这与干预数据从根本上不同。干预数据需要对变量进行直接扰动，而干预约束则通过不等式约束形式编码高级别因果知识。例如，在Sachs数据集（Sachs et al. 2005）中，Akt已被证明被PIP3激活，这意味着PIP3对Akt具有正向因果效应。现有因果发现方法允许施加结构约束（例如，要求从PIP3到Akt的因果路径），但仍可能得出错误的因果结论，如“PIP3抑制Akt”。干预约束通过明确限制变量对之间的总因果效应，确保学习的模型遵守已知的因果影响。为形式化干预约束，我们提出了一种度量标准来量化线性因果模型中的总因果效应，并将问题表述为受约束的优化任务，使用两阶段受约束优化方法求解。我们在现实世界数据集上评估了我们的方法，并展示了整合干预约束不仅提高了模型的准确性并确保与现有发现的一致性，使模型更具可解释性，还促进了新因果关系的发现，这原本可能成本极高。 

---
# GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-guided Latent Diffusion Model? 

**Title (ZH)**: GLYPH-SR：通过VLM引导的潜在扩散模型，我们能否同时实现高质量图像超分辨和高保真文本恢复？ 

**Authors**: Mingyu Sung, Seungjae Ham, Kangwoo Kim, Yeokyoung Yoon, Sangseok Yun, Il-Min Kim, Jae-Mo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26339)  

**Abstract**: Image super-resolution(SR) is fundamental to many vision system-from surveillance and autonomy to document analysis and retail analytics-because recovering high-frequency details, especially scene-text, enables reliable downstream perception. Scene-text, i.e., text embedded in natural images such as signs, product labels, and storefronts, often carries the most actionable information; when characters are blurred or hallucinated, optical character recognition(OCR) and subsequent decisions fail even if the rest of the image appears sharp. Yet previous SR research has often been tuned to distortion (PSNR/SSIM) or learned perceptual metrics (LIPIS, MANIQA, CLIP-IQA, MUSIQ) that are largely insensitive to character-level errors. Furthermore, studies that do address text SR often focus on simplified benchmarks with isolated characters, overlooking the challenges of text within complex natural scenes. As a result, scene-text is effectively treated as generic texture. For SR to be effective in practical deployments, it is therefore essential to explicitly optimize for both text legibility and perceptual quality. We present GLYPH-SR, a vision-language-guided diffusion framework that aims to achieve both objectives jointly. GLYPH-SR utilizes a Text-SR Fusion ControlNet(TS-ControlNet) guided by OCR data, and a ping-pong scheduler that alternates between text- and scene-centric guidance. To enable targeted text restoration, we train these components on a synthetic corpus while keeping the main SR branch frozen. Across SVT, SCUT-CTW1500, and CUTE80 at x4, and x8, GLYPH-SR improves OCR F1 by up to +15.18 percentage points over diffusion/GAN baseline (SVT x8, OpenOCR) while maintaining competitive MANIQA, CLIP-IQA, and MUSIQ. GLYPH-SR is designed to satisfy both objectives simultaneously-high readability and high visual realism-delivering SR that looks right and reds right. 

**Abstract (ZH)**: 基于视觉语言指导的扩散框架：同时提升场景文本可读性和视觉真实感 

---
# From Amateur to Master: Infusing Knowledge into LLMs via Automated Curriculum Learning 

**Title (ZH)**: 从业余到大师：通过自动化课程学习向LLMs注入知识 

**Authors**: Nishit Neema, Srinjoy Mukherjee, Sapan Shah, Gokul Ramakrishnan, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2510.26336)  

**Abstract**: Large Language Models (LLMs) excel at general tasks but underperform in specialized domains like economics and psychology, which require deep, principled understanding. To address this, we introduce ACER (Automated Curriculum-Enhanced Regimen) that transforms generalist models into domain experts without sacrificing their broad capabilities. ACER first synthesizes a comprehensive, textbook-style curriculum by generating a table of contents for a subject and then creating question-answer (QA) pairs guided by Bloom's taxonomy. This ensures systematic topic coverage and progressively increasing difficulty. The resulting synthetic corpus is used for continual pretraining with an interleaved curriculum schedule, aligning learning across both content and cognitive dimensions.
Experiments with Llama 3.2 (1B and 3B) show significant gains in specialized MMLU subsets. In challenging domains like microeconomics, where baselines struggle, ACER boosts accuracy by 5 percentage points. Across all target domains, we observe a consistent macro-average improvement of 3 percentage points. Notably, ACER not only prevents catastrophic forgetting but also facilitates positive cross-domain knowledge transfer, improving performance on non-target domains by 0.7 points. Beyond MMLU, ACER enhances performance on knowledge-intensive benchmarks like ARC and GPQA by over 2 absolute points, while maintaining stable performance on general reasoning tasks. Our results demonstrate that ACER offers a scalable and effective recipe for closing critical domain gaps in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通用任务上表现出色，但在经济学和心理学等专门领域表现不佳，这些领域需要深入的专业理解。为了解决这个问题，我们引入了ACER（Automated Curriculum-Enhanced Regimen），它能够将通用模型转换为领域专家，同时保留其广泛的通用能力。ACER 首先生成一个主题的大纲，并据此创建由布卢姆分类学引导的问题-答案（QA）对，从而合成一个全面的、教材风格的课程，确保系统地覆盖各个主题并逐步增加难度。生成的合成语料库用于持续预训练，并采用交错的课程时间表，从而使学习在内容和认知维度上保持一致。实验表明，ACER 在 Llama 3.2（1B 和 3B）以及经济学等具有挑战性的专业领域中带来了显著改进。在所有目标领域，我们观察到一致的宏观平均改进为 3 个百分点。值得注意的是，ACER 不仅防止了灾难性遗忘，还促进了跨领域的积极知识转移，在非目标领域上提升了 0.7 个百分点的表现。此外，ACER 在知识密集型基准测试，如ARC和GPQA上提高了 2 个百分点以上的性能，同时在通用推理任务上保持了稳定的表现。我们的结果表明，ACER 为缩小LLMs中的关键领域差距提供了一种可扩展且有效的方案。 

---
# Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics 

**Title (ZH)**: 结合退火朗格文动力学与扩散模型的后验采样 

**Authors**: Zhiyang Xun, Shivam Gupta, Eric Price  

**Link**: [PDF](https://arxiv.org/pdf/2510.26324)  

**Abstract**: Given a noisy linear measurement $y = Ax + \xi$ of a distribution $p(x)$, and a good approximation to the prior $p(x)$, when can we sample from the posterior $p(x \mid y)$? Posterior sampling provides an accurate and fair framework for tasks such as inpainting, deblurring, and MRI reconstruction, and several heuristics attempt to approximate it. Unfortunately, approximate posterior sampling is computationally intractable in general.
To sidestep this hardness, we focus on (local or global) log-concave distributions $p(x)$. In this regime, Langevin dynamics yields posterior samples when the exact scores of $p(x)$ are available, but it is brittle to score--estimation error, requiring an MGF bound (sub-exponential error). By contrast, in the unconditional setting, diffusion models succeed with only an $L^2$ bound on the score error. We prove that combining diffusion models with an annealed variant of Langevin dynamics achieves conditional sampling in polynomial time using merely an $L^4$ bound on the score error. 

**Abstract (ZH)**: 给定一个分布$p(x)$的带噪线性测量$y = Ax + \xi$及其良好的先验近似$p(x)$，在什么情况下可以从后验$p(x \mid y)$中采样？后验采样为填图、去模糊和MRI重建等任务提供了一个准确且公平的框架，但有几个启发式方法试图近似它。然而，近似后验采样通常在一般情况下是计算上不可行的。

为避免这一困难，我们关注局部或全局的对数凹分布$p(x)$。在这种情况下，当可以获取$p(x)$的确切分数时，Langevin动力学可以生成后验样本，但这种动力学对分数估计误差很敏感，需要一个MGF界（亚指数误差）。相反，在无条件情况下，扩散模型仅需分数误差的$L^2$界就能成功。我们证明，将扩散模型与Langevin动力学的退火变体结合可以仅使用分数误差的$L^4$界在多项式时间内实现有条件采样。 

---
# Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime 

**Title (ZH)**: 分离数据上每样本Adam的隐性偏差：超出全批量范式 

**Authors**: Beomhan Baek, Minhak Song, Chulhee Yun  

**Link**: [PDF](https://arxiv.org/pdf/2510.26303)  

**Abstract**: Adam [Kingma and Ba, 2015] is the de facto optimizer in deep learning, yet its theoretical understanding remains limited. Prior analyses show that Adam favors solutions aligned with $\ell_\infty$-geometry, but these results are restricted to the full-batch regime. In this work, we study the implicit bias of incremental Adam (using one sample per step) for logistic regression on linearly separable data, and we show that its bias can deviate from the full-batch behavior. To illustrate this, we construct a class of structured datasets where incremental Adam provably converges to the $\ell_2$-max-margin classifier, in contrast to the $\ell_\infty$-max-margin bias of full-batch Adam. For general datasets, we develop a proxy algorithm that captures the limiting behavior of incremental Adam as $\beta_2 \to 1$ and we characterize its convergence direction via a data-dependent dual fixed-point formulation. Finally, we prove that, unlike Adam, Signum [Bernstein et al., 2018] converges to the $\ell_\infty$-max-margin classifier for any batch size by taking $\beta$ close enough to 1. Overall, our results highlight that the implicit bias of Adam crucially depends on both the batching scheme and the dataset, while Signum remains invariant. 

**Abstract (ZH)**: Adam [Kingma和Ba, 2015]是深度学习中的事实上的优化器，但对其理论理解仍然有限。以往的分析表明，Adam偏好与$\ell_\infty$几何对齐的解决方案，但这些结果仅限于全批量情况。在这项工作中，我们研究了增量Adam（每步使用一个样本）在线性可分数据上的逻辑回归中的隐式偏见，并证明其偏见可以偏离全批量行为。为了说明这一点，我们构造了一类结构化数据集，在这类数据集上，增量Adam可以证明收敛到$\ell_2$-最大边际分类器，而全批量Adam偏好$\ell_\infty$-最大边际分类器。对于一般的数据集，我们开发了一个代理算法来捕捉增量Adam随$\beta_2 \to 1$变化的极限行为，并通过数据依赖的双重不动点形式来表征其收敛方向。最后，我们证明，与Adam不同，Signum [Bernstein等人, 2018]可以通过将$\beta$调整得足够接近1，对任何批量大小都收敛到$\ell_\infty$-最大边际分类器。总体而言，我们的结果强调了Adam的隐式偏见不仅取决于批量方案，还取决于数据集，而Signum保持不变。 

---
# Understanding Hardness of Vision-Language Compositionality from A Token-level Causal Lens 

**Title (ZH)**: 从token层面因果视角理解视觉-语言组合性的难度 

**Authors**: Ziliang Chen, Tianang Xiao, Jusheng Zhang, Yongsen Zheng, Xipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26302)  

**Abstract**: Contrastive Language-Image Pre-training (CLIP) delivers strong cross modal generalization by aligning images and texts in a shared embedding space, yet it persistently fails at compositional reasoning over objects, attributes, and relations often behaving like a bag-of-words matcher. Prior causal accounts typically model text as a single vector, obscuring token-level structure and leaving core phenomena-such as prompt sensitivity and failures on hard negatives unexplained. We address this gap with a token-aware causal representation learning (CRL) framework grounded in a sequential, language-token SCM. Our theory extends block identifiability to tokenized text, proving that CLIP's contrastive objective can recover the modal-invariant latent variable under both sentence-level and token-level SCMs. Crucially, token granularity yields the first principled explanation of CLIP's compositional brittleness: composition nonidentifiability. We show the existence of pseudo-optimal text encoders that achieve perfect modal-invariant alignment yet are provably insensitive to SWAP, REPLACE, and ADD operations over atomic concepts, thereby failing to distinguish correct captions from hard negatives despite optimizing the same training objective as true-optimal encoders. The analysis further links language-side nonidentifiability to visual-side failures via the modality gap and shows how iterated composition operators compound hardness, motivating improved negative mining strategies. 

**Abstract (ZH)**: 基于因果表示学习的.token感知对比预训练 

---
# Can Agent Conquer Web? Exploring the Frontiers of ChatGPT Atlas Agent in Web Games 

**Title (ZH)**: 智能体能否 conquer 互联网？探究 ChatGPT 图Athlon 代理在网页游戏中的边界 

**Authors**: Jingran Zhang, Ning Li, Justin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2510.26298)  

**Abstract**: OpenAI's ChatGPT Atlas introduces new capabilities for web interaction, enabling the model to analyze webpages, process user intents, and execute cursor and keyboard inputs directly within the browser. While its capacity for information retrieval tasks has been demonstrated, its performance in dynamic, interactive environments remains less explored. In this study, we conduct an early evaluation of Atlas's web interaction capabilities using browser-based games as test scenarios, including Google's T-Rex Runner, Sudoku, Flappy Bird, and this http URL. We employ in-game performance scores as quantitative metrics to assess performance across different task types. Our results show that Atlas performs strongly in logical reasoning tasks like Sudoku, completing puzzles significantly faster than human baselines, but struggles substantially in real-time games requiring precise timing and motor control, often failing to progress beyond initial obstacles. These findings suggest that while Atlas demonstrates capable analytical processing, there remain notable limitations in dynamic web environments requiring real-time interaction. The website of our project can be found at this https URL. 

**Abstract (ZH)**: OpenAI的ChatGPT Atlas引入了新的网页交互能力，使其能够分析网页、处理用户意图并在浏览器中直接执行鼠标和键盘输入。尽管其在信息检索任务上的能力已得到验证，但在动态、交互式环境中的性能尚未得到充分探索。本研究使用基于浏览器的游戏作为测试场景，评估Atlas的网页交互能力，包括Google的T-Rex Runner、数独、Flappy Bird等游戏。我们采用游戏内的性能评分作为定量指标，评估不同任务类型的表现。结果显示，Atlas在数独等逻辑推理任务中表现出色，完成谜题的速度远超人类基准，但在需要精确时间控制和运动控制的实时游戏中却表现不佳，往往无法克服初始障碍。这些发现表明，虽然Atlas展示了强大的分析处理能力，但在要求实时交互的动态网络环境中仍存在显著限制。我们的项目网站可访问此处：这个 https URL。 

---
# Unravelling the Mechanisms of Manipulating Numbers in Language Models 

**Title (ZH)**: 探究操控语言模型中数字机制的方法 

**Authors**: Michal Štefánik, Timothee Mickus, Marek Kadlčík, Bertram Højer, Michal Spiegel, Raúl Vázquez, Aman Sinha, Josef Kuchař, Philipp Mondorf  

**Link**: [PDF](https://arxiv.org/pdf/2510.26285)  

**Abstract**: Recent work has shown that different large language models (LLMs) converge to similar and accurate input embedding representations for numbers. These findings conflict with the documented propensity of LLMs to produce erroneous outputs when dealing with numeric information. In this work, we aim to explain this conflict by exploring how language models manipulate numbers and quantify the lower bounds of accuracy of these mechanisms. We find that despite surfacing errors, different language models learn interchangeable representations of numbers that are systematic, highly accurate and universal across their hidden states and the types of input contexts. This allows us to create universal probes for each LLM and to trace information -- including the causes of output errors -- to specific layers. Our results lay a fundamental understanding of how pre-trained LLMs manipulate numbers and outline the potential of more accurate probing techniques in addressed refinements of LLMs' architectures. 

**Abstract (ZH)**: 近期研究表明，不同的大型语言模型（LLMs）对数字的输入嵌入表示趋于收敛且准确。这些发现与现有文献中关于LLMs处理数字信息时产生错误输出的倾向相矛盾。在这项工作中，我们旨在通过探索语言模型如何处理数字以及量化这些机制的下限准确性来解释这一矛盾。我们发现，尽管表面上存在错误，不同语言模型学习到的数字表示是系统性的、高度准确且在隐藏层和不同类型输入上下文中具有普适性。这使我们能够为每种LLM创建通用探针，并追踪信息——包括输出错误的原因——到特定层面。我们的结果为预训练LLMs如何处理数字提供了基本理解，并指出了更准确探针技术在LLM架构改进方面潜在的价值。 

---
# Distributional Multi-objective Black-box Optimization for Diffusion-model Inference-time Multi-Target Generation 

**Title (ZH)**: 分布式的多目标黑箱优化在扩散模型推理时的多目标生成 

**Authors**: Kim Yong Tan, Yueming Lyu, Ivor Tsang, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2510.26278)  

**Abstract**: Diffusion models have been successful in learning complex data distributions. This capability has driven their application to high-dimensional multi-objective black-box optimization problem. Existing approaches often employ an external optimization loop, such as an evolutionary algorithm, to the diffusion model. However, these approaches treat the diffusion model as a black-box refiner, which overlooks the internal distribution transition of the diffusion generation process, limiting their efficiency. To address these challenges, we propose the Inference-time Multi-target Generation (IMG) algorithm, which optimizes the diffusion process at inference-time to generate samples that simultaneously satisfy multiple objectives. Specifically, our IMG performs weighted resampling during the diffusion generation process according to the expected aggregated multi-objective values. This weighted resampling strategy ensures the diffusion-generated samples are distributed according to our desired multi-target Boltzmann distribution. We further derive that the multi-target Boltzmann distribution has an interesting log-likelihood interpretation, where it is the optimal solution to the distributional multi-objective optimization problem. We implemented IMG for a multi-objective molecule generation task. Experiments show that IMG, requiring only a single generation pass, achieves a significantly higher hypervolume than baseline optimization algorithms that often require hundreds of diffusion generations. Notably, our algorithm can be viewed as an optimized diffusion process and can be integrated into existing methods to further improve their performance. 

**Abstract (ZH)**: 基于推断时多目标生成的扩散模型算法 

---
# A Research Roadmap for Augmenting Software Engineering Processes and Software Products with Generative AI 

**Title (ZH)**: 增强软件工程过程和软件产品生成式AI的研究路线图 

**Authors**: Domenico Amalfitano, Andreas Metzger, Marco Autili, Tommaso Fulcini, Tobias Hey, Jan Keim, Patrizio Pelliccione, Vincenzo Scotti, Anne Koziolek, Raffaela Mirandola, Andreas Vogelsang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26275)  

**Abstract**: Generative AI (GenAI) is rapidly transforming software engineering (SE) practices, influencing how SE processes are executed, as well as how software systems are developed, operated, and evolved. This paper applies design science research to build a roadmap for GenAI-augmented SE. The process consists of three cycles that incrementally integrate multiple sources of evidence, including collaborative discussions from the FSE 2025 "Software Engineering 2030" workshop, rapid literature reviews, and external feedback sessions involving peers. McLuhan's tetrads were used as a conceptual instrument to systematically capture the transforming effects of GenAI on SE processes and software this http URL resulting roadmap identifies four fundamental forms of GenAI augmentation in SE and systematically characterizes their related research challenges and opportunities. These insights are then consolidated into a set of future research directions. By grounding the roadmap in a rigorous multi-cycle process and cross-validating it among independent author teams and peers, the study provides a transparent and reproducible foundation for analyzing how GenAI affects SE processes, methods and tools, and for framing future research within this rapidly evolving area. Based on these findings, the article finally makes ten predictions for SE in the year 2030. 

**Abstract (ZH)**: 生成式人工智能（GenAI）正在快速重塑软件工程（SE）实践，影响SE过程的执行方式，以及软件系统的开发、运行和演化。本文采用设计科学方法构建了一个GenAI增强SE的路线图。该过程包括三个循环，逐步整合多种证据来源，包括2025年FSE“软件工程2030”研讨会的协作讨论、快速文献综述以及来自同行的外部反馈会。利用麦卢汉的四象限作为概念工具，系统地捕获GenAI对SE过程和软件系统的转变效应。由此产生的路线图识别了四种基本形式的GenAI在SE中的增强，并系统地对其相关研究挑战和机遇进行了分类。这些见解随后被整合为一套未来研究方向。通过在一个严格的多循环过程基础上构建路线图，并在独立作者团队和同行之间进行交叉验证，研究为分析GenAI对SE过程、方法和工具的影响提供了透明且可重现的基础，并为将未来研究置于这一快速发展的领域内奠定框架。基于这些发现，文章最终提出了十个关于2030年软件工程的预测。 

---
# Angular Steering: Behavior Control via Rotation in Activation Space 

**Title (ZH)**: 角度转向：通过激活空间旋转进行行为控制 

**Authors**: Hieu M. Vu, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26243)  

**Abstract**: Controlling specific behaviors in large language models while preserving their general capabilities is a central challenge for safe and reliable artificial intelligence deployment. Current steering methods, such as vector addition and directional ablation, are constrained within a two-dimensional subspace defined by the activation and feature direction, making them sensitive to chosen parameters and potentially affecting unrelated features due to unintended interactions in activation space. We introduce Angular Steering, a novel and flexible method for behavior modulation that operates by rotating activations within a fixed two-dimensional subspace. By formulating steering as a geometric rotation toward or away from a target behavior direction, Angular Steering provides continuous, fine-grained control over behaviors such as refusal and compliance. We demonstrate this method using refusal steering emotion steering as use cases. Additionally, we propose Adaptive Angular Steering, a selective variant that rotates only activations aligned with the target feature, further enhancing stability and coherence. Angular Steering generalizes existing addition and orthogonalization techniques under a unified geometric rotation framework, simplifying parameter selection and maintaining model stability across a broader range of adjustments. Experiments across multiple model families and sizes show that Angular Steering achieves robust behavioral control while maintaining general language modeling performance, underscoring its flexibility, generalization, and robustness compared to prior approaches. Code and artifacts are available at this https URL. 

**Abstract (ZH)**: 在保留大型语言模型通用能力的同时控制特定行为是实现安全可靠人工智能部署的核心挑战。当前的调控方法，如向量加法和方向消融，受限于由激活和特征方向定义的二维子空间，这使得它们对选定参数敏感，并且由于激活空间中的意外交互可能影响不相关的特征。我们提出了角度调控，这是一种新颖且灵活的行为调控方法，通过在固定的二维子空间内旋转激活来实现。通过将调控形式化为几何旋转朝向或远离目标行为方向，角度调控提供了对拒绝和遵从等行为的连续、精细控制。我们通过拒绝调控和情绪调控案例展示了这一方法。此外，我们提出了适应性角度调控，这是一种选择性变体，仅旋转与目标特征对齐的激活，进一步增强了稳定性和一致性。角度调控在统一的几何旋转框架下推广了现有的加法和技术祛除技术，简化了参数选择并保持模型在更广泛调整范围内的稳定性。多模型家族和规模下的实验表明，角度调控在保持通用语言建模性能的同时实现了稳健的行为控制，突显了其灵活性、泛化能力和稳健性，优于先前的方法。代码和相关材料可从该网址获取。 

---
# MPRU: Modular Projection-Redistribution Unlearning as Output Filter for Classification Pipelines 

**Title (ZH)**: MPRU: 模块化投影-重分布遗忘作为分类流水线的输出过滤器 

**Authors**: Minyi Peng, Darian Gunamardi, Ivan Tjuawinata, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2510.26230)  

**Abstract**: As a new and promising approach, existing machine unlearning (MU) works typically emphasize theoretical formulations or optimization objectives to achieve knowledge removal. However, when deployed in real-world scenarios, such solutions typically face scalability issues and have to address practical requirements such as full access to original datasets and model. In contrast to the existing approaches, we regard classification training as a sequential process where classes are learned sequentially, which we call \emph{inductive approach}. Unlearning can then be done by reversing the last training sequence. This is implemented by appending a projection-redistribution layer in the end of the model. Such an approach does not require full access to the original dataset or the model, addressing the challenges of existing methods. This enables modular and model-agnostic deployment as an output filter into existing classification pipelines with minimal alterations. We conducted multiple experiments across multiple datasets including image (CIFAR-10/100 using CNN-based model) and tabular datasets (Covertype using tree-based model). Experiment results show consistently similar output to a fully retrained model with a high computational cost reduction. This demonstrates the applicability, scalability, and system compatibility of our solution while maintaining the performance of the output in a more practical setting. 

**Abstract (ZH)**: 作为一种新的有前途的方法，现有的机器遗忘（MU）工作通常侧重于理论建模或优化目标以实现知识删除。然而，在实际应用中，这些解决方案通常面临可扩展性问题，并且必须满足诸如原始数据集和模型的完全访问等实际需求。与现有方法不同，我们将分类训练视为一个顺序过程，即类别按顺序学习，我们称之为归纳方法。然后可以通过逆转最后一个训练序列来进行遗忘。这通过在模型末尾添加一个投影-重新分配层来实现。这种 approach 不需要访问原始数据集或模型，从而解决了现有方法的挑战。这使得我们的解决方案能够以最少的修改作为输出过滤器集成到现有的分类流水线中，实现模块化和模型无关的部署。我们在多个数据集上进行了多次实验，包括使用基于 CNN 的模型的图像数据集（CIFAR-10/100）和使用基于树的模型的表数据集（Covertype）。实验结果表明，输出与高计算成本下重新训练的模型具有相似性，这证明了我们解决方案的适用性、可扩展性和系统兼容性，同时保持了输出的性能在更实际的环境中。 

---
# Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit space 

**Title (ZH)**: Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit Space 

**Authors**: Sekitoshi Kanai, Tsukasa Yoshida, Hiroshi Takahashi, Haru Kuroki, Kazumune Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2510.26219)  

**Abstract**: Test-time alignment of large language models (LLMs) attracts attention because fine-tuning LLMs requires high computational costs. In this paper, we propose a new test-time alignment method called adaptive importance sampling on pre-logits (AISP) on the basis of the sampling-based model predictive control with the stochastic control input. AISP applies the Gaussian perturbation into pre-logits, which are outputs of the penultimate layer, so as to maximize expected rewards with respect to the mean of the perturbation. We demonstrate that the optimal mean is obtained by importance sampling with sampled rewards. AISP outperforms best-of-n sampling in terms of rewards over the number of used samples and achieves higher rewards than other reward-based test-time alignment methods. 

**Abstract (ZH)**: 基于采样模型预测控制和随机控制输入的预输出自适应重要性采样测试时对齐方法 

---
# Hybrid LLM and Higher-Order Quantum Approximate Optimization for CSA Collateral Management 

**Title (ZH)**: 混合大语言模型和高阶量子近似优化在 CSA 净额结算担保管理中的应用 

**Authors**: Tao Jin, Stuart Florescu, Heyu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26217)  

**Abstract**: We address finance-native collateral optimization under ISDA Credit Support Annexes (CSAs), where integer lots, Schedule A haircuts, RA/MTA gating, and issuer/currency/class caps create rugged, legally bounded search spaces. We introduce a certifiable hybrid pipeline purpose-built for this domain: (i) an evidence-gated LLM that extracts CSA terms to a normalized JSON (abstain-by-default, span-cited); (ii) a quantum-inspired explorer that interleaves simulated annealing with micro higher order QAOA (HO-QAOA) on binding sub-QUBOs (subset size n <= 16, order k <= 4) to coordinate multi-asset moves across caps and RA-induced discreteness; (iii) a weighted risk-aware objective (Movement, CVaR, funding-priced overshoot) with an explicit coverage window U <= Reff+B; and (iv) CP-SAT as single arbiter to certify feasibility and gaps, including a U-cap pre-check that reports the minimal feasible buffer B*. Encoding caps/rounding as higher-order terms lets HO-QAOA target the domain couplings that defeat local swaps. On government bond datasets and multi-CSA inputs, the hybrid improves a strong classical baseline (BL-3) by 9.1%, 9.6%, and 10.7% across representative harnesses, delivering better cost-movement-tail frontiers under governance settings. We release governance grade artifacts-span citations, valuation matrix audit, weight provenance, QUBO manifests, and CP-SAT traces-to make results auditable and reproducible. 

**Abstract (ZH)**: 面向ISDA信贷支持附则的金融原生抵押品优化：一种可验证的混合管道方法 

---
# Towards Global Retrieval Augmented Generation: A Benchmark for Corpus-Level Reasoning 

**Title (ZH)**: 面向全局检索增强生成：一个语料库级推理基准 

**Authors**: Qi Luo, Xiaonan Li, Tingshuo Fan, Xinchi Chen, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26205)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a leading approach to reducing hallucinations in large language models (LLMs). Current RAG evaluation benchmarks primarily focus on what we call local RAG: retrieving relevant chunks from a small subset of documents to answer queries that require only localized understanding within specific text chunks. However, many real-world applications require a fundamentally different capability -- global RAG -- which involves aggregating and analyzing information across entire document collections to derive corpus-level insights (for example, "What are the top 10 most cited papers in 2023?"). In this paper, we introduce GlobalQA -- the first benchmark specifically designed to evaluate global RAG capabilities, covering four core task types: counting, extremum queries, sorting, and top-k extraction. Through systematic evaluation across different models and baselines, we find that existing RAG methods perform poorly on global tasks, with the strongest baseline achieving only 1.51 F1 score. To address these challenges, we propose GlobalRAG, a multi-tool collaborative framework that preserves structural coherence through chunk-level retrieval, incorporates LLM-driven intelligent filters to eliminate noisy documents, and integrates aggregation modules for precise symbolic computation. On the Qwen2.5-14B model, GlobalRAG achieves 6.63 F1 compared to the strongest baseline's 1.51 F1, validating the effectiveness of our method. 

**Abstract (ZH)**: 全局增强生成（GlobalRAG）：一种评估全局RAG能力的新基准 

---
# What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data 

**Title (ZH)**: 我的人类反馈中包含了什么？学习可解释的偏好数据描述 

**Authors**: Rajiv Movva, Smitha Milli, Sewon Min, Emma Pierson  

**Link**: [PDF](https://arxiv.org/pdf/2510.26202)  

**Abstract**: Human feedback can alter language models in unpredictable and undesirable ways, as practitioners lack a clear understanding of what feedback data encodes. While prior work studies preferences over certain attributes (e.g., length or sycophancy), automatically extracting relevant features without pre-specifying hypotheses remains challenging. We introduce What's In My Human Feedback? (WIMHF), a method to explain feedback data using sparse autoencoders. WIMHF characterizes both (1) the preferences a dataset is capable of measuring and (2) the preferences that the annotators actually express. Across 7 datasets, WIMHF identifies a small number of human-interpretable features that account for the majority of the preference prediction signal achieved by black-box models. These features reveal a wide diversity in what humans prefer, and the role of dataset-level context: for example, users on Reddit prefer informality and jokes, while annotators in HH-RLHF and PRISM disprefer them. WIMHF also surfaces potentially unsafe preferences, such as that LMArena users tend to vote against refusals, often in favor of toxic content. The learned features enable effective data curation: re-labeling the harmful examples in Arena yields large safety gains (+37%) with no cost to general performance. They also allow fine-grained personalization: on the Community Alignment dataset, we learn annotator-specific weights over subjective features that improve preference prediction. WIMHF provides a human-centered analysis method for practitioners to better understand and use preference data. 

**Abstract (ZH)**: 人类反馈可能以不可预测且不 desirable 的方式改变语言模型，因为实践者缺乏对反馈数据编码内容的清晰理解。尽管先前的工作研究了某些属性的偏好（例如长度或逢迎），但在无需预先假设的情况下自动提取相关特征仍然是一个挑战。我们引入了“What’s In My Human Feedback?”（WIMHF）方法，使用稀疏自编码器解释反馈数据。WIMHF 描述了（1）数据集能够测量的偏好以及（2）实际表达的偏好。在 7 个数据集中，WIMHF 识别出少量可由人类解释的特征，这些特征解释了黑盒模型实现的大多数偏好预测信号。这些特征揭示了人类偏好的广泛多样性，以及数据集级别上下文的作用：例如，Reddit 上的用户更喜欢非正式性和幽默，而 HH-RLHF 和 PRISM 的注释员则不偏好这些。WIMHF 还揭示了潜在的不安全偏好，例如 LMArena 用户倾向于反对拒绝，并经常有利有毒内容。学习到的特征能够有效进行数据管理：重新标记竞技场中的有害示例可带来显著的安全收益（+37%）且不会影响总体性能。它们还允许细粒度的个性化：在社区对齐数据集中，我们学习了注释器特定的权重，这些权重改善了偏好预测。WIMHF 为实践者提供了一种以人为中心的分析方法，帮助他们更好地理解并利用偏好数据。 

---
# Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation 

**Title (ZH)**: 不要让它褪色：通过令牌时间步分配保存扩散语言模型中的编辑 

**Authors**: Woojin Kim, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2510.26200)  

**Abstract**: While diffusion language models (DLMs) enable fine-grained refinement, their practical controllability remains fragile. We identify and formally characterize a central failure mode called update forgetting, in which uniform and context agnostic updates induce token level fluctuations across timesteps, erasing earlier semantic edits and disrupting the cumulative refinement process, thereby degrading fluency and coherence. As this failure originates in uniform and context agnostic updates, effective control demands explicit token ordering. We propose Token Timestep Allocation (TTA), which realizes soft and semantic token ordering via per token timestep schedules: critical tokens are frozen early, while uncertain tokens receive continued refinement. This timestep based ordering can be instantiated as either a fixed policy or an adaptive policy driven by task signals, thereby supporting a broad spectrum of refinement strategies. Because it operates purely at inference time, it applies uniformly across various DLMs and naturally extends to diverse supervision sources. Empirically, TTA improves controllability and fluency: on sentiment control, it yields more than 20 percent higher accuracy and nearly halves perplexity using less than one fifth the steps; in detoxification, it lowers maximum toxicity (12.2 versus 14.5) and perplexity (26.0 versus 32.0). Together, these results demonstrate that softened ordering via timestep allocation is the critical lever for mitigating update forgetting and achieving stable and controllable diffusion text generation. 

**Abstract (ZH)**: 基于时间步分配的令牌排序在弥合更新遗忘以实现可控的扩散语言生成中的作用 

---
# Predicting All-Cause Hospital Readmissions from Medical Claims Data of Hospitalised Patients 

**Title (ZH)**: 基于住院患者医疗索赔数据预测所有原因再住院 

**Authors**: Avinash Kadimisetty, Arun Rajagopalan, Vijendra SK  

**Link**: [PDF](https://arxiv.org/pdf/2510.26188)  

**Abstract**: Reducing preventable hospital readmissions is a national priority for payers, providers, and policymakers seeking to improve health care and lower costs. The rate of readmission is being used as a benchmark to determine the quality of healthcare provided by the hospitals. In thisproject, we have used machine learning techniques like Logistic Regression, Random Forest and Support Vector Machines to analyze the health claims data and identify demographic and medical factors that play a crucial role in predicting all-cause readmissions. As the health claims data is high dimensional, we have used Principal Component Analysis as a dimension reduction technique and used the results for building regression models. We compared and evaluated these models based on the Area Under Curve (AUC) metric. Random Forest model gave the highest performance followed by Logistic Regression and Support Vector Machine models. These models can be used to identify the crucial factors causing readmissions and help identify patients to focus on to reduce the chances of readmission, ultimately bringing down the cost and increasing the quality of healthcare provided to the patients. 

**Abstract (ZH)**: 降低可预防的医院再入院率是支付方、提供方和政策制定者提高医疗护理质量和降低成本的国家优先事项。再入院率被用作衡量医院提供医疗服务质量的基准。在本项目中，我们使用了逻辑回归、随机森林和支持向量机等机器学习技术来分析健康索赔数据，并识别预测所有原因再入院的关键人口统计和医学因素。由于健康索赔数据维度高，我们使用主成分分析作为降维技术，并使用其结果构建回归模型。我们根据曲线下面积（AUC）指标比较和评估了这些模型。随机森林模型的性能最高，其次是逻辑回归模型和支持向量机模型。这些模型可以用来识别导致再入院的关键因素，并帮助识别需要重点关注的患者，从而降低再入院率，最终降低医疗成本并提高医疗服务的质量。 

---
# ConceptScope: Characterizing Dataset Bias via Disentangled Visual Concepts 

**Title (ZH)**: ConceptScope: 通过解耦视觉概念表征数据集偏差 

**Authors**: Jinho Choi, Hyesu Lim, Steffen Schneider, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2510.26186)  

**Abstract**: Dataset bias, where data points are skewed to certain concepts, is ubiquitous in machine learning datasets. Yet, systematically identifying these biases is challenging without costly, fine-grained attribute annotations. We present ConceptScope, a scalable and automated framework for analyzing visual datasets by discovering and quantifying human-interpretable concepts using Sparse Autoencoders trained on representations from vision foundation models. ConceptScope categorizes concepts into target, context, and bias types based on their semantic relevance and statistical correlation to class labels, enabling class-level dataset characterization, bias identification, and robustness evaluation through concept-based subgrouping. We validate that ConceptScope captures a wide range of visual concepts, including objects, textures, backgrounds, facial attributes, emotions, and actions, through comparisons with annotated datasets. Furthermore, we show that concept activations produce spatial attributions that align with semantically meaningful image regions. ConceptScope reliably detects known biases (e.g., background bias in Waterbirds) and uncovers previously unannotated ones (e.g, co-occurring objects in ImageNet), offering a practical tool for dataset auditing and model diagnostics. 

**Abstract (ZH)**: 基于概念范围的视觉数据集可扩展自动分析框架：识别和量化概念偏差 

---
# Accumulative SGD Influence Estimation for Data Attribution 

**Title (ZH)**: 累积SGD影响估计用于数据归属 

**Authors**: Yunxiao Shi, Shuo Yang, Yixin Su, Rui Zhang, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26185)  

**Abstract**: Modern data-centric AI needs precise per-sample influence. Standard SGD-IE approximates leave-one-out effects by summing per-epoch surrogates and ignores cross-epoch compounding, which misranks critical examples. We propose ACC-SGD-IE, a trajectory-aware estimator that propagates the leave-one-out perturbation across training and updates an accumulative influence state at each step. In smooth strongly convex settings it achieves geometric error contraction and, in smooth non-convex regimes, it tightens error bounds; larger mini-batches further reduce constants. Empirically, on Adult, 20 Newsgroups, and MNIST under clean and corrupted data and both convex and non-convex training, ACC-SGD-IE yields more accurate influence estimates, especially over long epochs. For downstream data cleansing it more reliably flags noisy samples, producing models trained on ACC-SGD-IE cleaned data that outperform those cleaned with SGD-IE. 

**Abstract (ZH)**: 现代以数据为中心的AI需要精确的单样本影响分析。标准SGD-IE通过累加每个epoch的替代效应来近似leave-one-out效果，并忽略了跨epoch的影响累积，从而错误地排序关键样本。我们提出了一种轨迹感知估计器ACC-SGD-IE，它在训练过程中传播leave-one-out扰动，并在每一步更新累积影响状态。在光滑强凸设置中，它实现了几何误差收缩；在光滑非凸设置中，它紧化了误差界限；较大的小批量进一步减少了常数。实验结果表明，在成人数据集、20个新闻组数据集和MNIST数据集上，在干净和污染的数据以及凸和非凸训练下，ACC-SGD-IE提供了更准确的影响估计，尤其是在长时间训练中。对于下游数据清理，ACC-SGD-IE更可靠地标记出噪声样本，使用ACC-SGD-IE清理后的数据训练的模型优于使用SGD-IE清理后的数据训练的模型。 

---
# Linking Heterogeneous Data with Coordinated Agent Flows for Social Media Analysis 

**Title (ZH)**: 基于协调智能体流动的异构数据链接与社会媒体分析 

**Authors**: Shifu Chen, Dazhen Deng, Zhihong Xu, Sijia Xu, Tai-Quan Peng, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26172)  

**Abstract**: Social media platforms generate massive volumes of heterogeneous data, capturing user behaviors, textual content, temporal dynamics, and network structures. Analyzing such data is crucial for understanding phenomena such as opinion dynamics, community formation, and information diffusion. However, discovering insights from this complex landscape is exploratory, conceptually challenging, and requires expertise in social media mining and visualization. Existing automated approaches, though increasingly leveraging large language models (LLMs), remain largely confined to structured tabular data and cannot adequately address the heterogeneity of social media analysis. We present SIA (Social Insight Agents), an LLM agent system that links heterogeneous multi-modal data -- including raw inputs (e.g., text, network, and behavioral data), intermediate outputs, mined analytical results, and visualization artifacts -- through coordinated agent flows. Guided by a bottom-up taxonomy that connects insight types with suitable mining and visualization techniques, SIA enables agents to plan and execute coherent analysis strategies. To ensure multi-modal integration, it incorporates a data coordinator that unifies tabular, textual, and network data into a consistent flow. Its interactive interface provides a transparent workflow where users can trace, validate, and refine the agent's reasoning, supporting both adaptability and trustworthiness. Through expert-centered case studies and quantitative evaluation, we show that SIA effectively discovers diverse and meaningful insights from social media while supporting human-agent collaboration in complex analytical tasks. 

**Abstract (ZH)**: 社交媒体平台生成大量的异构数据，捕捉用户行为、文本内容、时间动态和网络结构。分析此类数据对于理解意见动态、社区形成和信息扩散等现象至关重要。然而，从这一复杂景观中发现见解是探索性的、概念上具有挑战性的，并且需要具备社交媒体挖掘和可视化方面的专业知识。尽管现有自动化方法越来越多地利用大型语言模型（LLMs），它们仍然主要局限于结构化表格数据，无法充分应对社交媒体分析的异质性。我们提出了一种名为SIA（Social Insight Agents）的LLM代理系统，该系统通过协调的代理流将包括原始输入（如文本、网络和行为数据）、中间输出、挖掘分析结果和可视化成果在内的异构多模态数据联系起来。SIA 通过连接见解类型与合适的挖掘和可视化技术的自底向上的分类体系，使代理能够规划和执行一致的分析策略。为确保多模态集成，它包含一个数据协调器，该协调器将表格、文本和网络数据统一成一个一致的流程。交互式界面提供了透明的工作流，用户可以追溯、验证和精炼代理的推理过程，支持适应性和可靠性。通过对专家中心的案例研究和定量评估，我们展示了SIA 在复杂分析任务中有效发现多样且有意义的社交媒体见解，同时支持人类与代理的合作。 

---
# Learning to Manage Investment Portfolios beyond Simple Utility Functions 

**Title (ZH)**: 超越简单效用函数的学习投资组合管理 

**Authors**: Maarten P. Scholl, Mahmoud Mahfouz, Anisoara Calinescu, J. Doyne Farmer  

**Link**: [PDF](https://arxiv.org/pdf/2510.26165)  

**Abstract**: While investment funds publicly disclose their objectives in broad terms, their managers optimize for complex combinations of competing goals that go beyond simple risk-return trade-offs. Traditional approaches attempt to model this through multi-objective utility functions, but face fundamental challenges in specification and parameterization. We propose a generative framework that learns latent representations of fund manager strategies without requiring explicit utility specification.
Our approach directly models the conditional probability of a fund's portfolio weights, given stock characteristics, historical returns, previous weights, and a latent variable representing the fund's strategy. Unlike methods based on reinforcement learning or imitation learning, which require specified rewards or labeled expert objectives, our GAN-based architecture learns directly from the joint distribution of observed holdings and market data.
We validate our framework on a dataset of 1436 U.S. equity mutual funds. The learned representations successfully capture known investment styles, such as "growth" and "value," while also revealing implicit manager objectives. For instance, we find that while many funds exhibit characteristics of Markowitz-like optimization, they do so with heterogeneous realizations for turnover, concentration, and latent factors.
To analyze and interpret the end-to-end model, we develop a series of tests that explain the model, and we show that the benchmark's expert labeling are contained in our model's encoding in a linear interpretable way.
Our framework provides a data-driven approach for characterizing investment strategies for applications in market simulation, strategy attribution, and regulatory oversight. 

**Abstract (ZH)**: 尽管投资基金公开披露其总体目标，但其管理者会优化超越简单风险-收益权衡的复杂目标组合。传统方法尝试通过多目标效用函数来建模这一过程，但在效用函数的设定和参数化方面面临根本性的挑战。我们提出了一种生成式框架，该框架无需明确指定效用函数即可学习基金管理者策略的潜在表示。

我们的方法直接建模给定股票特征、历史回报、先前权重以及表示基金策略的潜在变量时，基金组合权重的条件概率。与基于强化学习或模仿学习的方法不同，后者需要指定奖励或标记专家目标，我们的基于生成对抗网络（GAN）的架构直接从观察持有量和市场数据的联合分布中学习。

我们在包含1436只美国共同基金的数据集上验证了该框架。学习到的表示成功捕捉了已知的投资风格，例如“成长型”和“价值型”，同时揭示了隐式的管理者目标。例如，我们发现尽管许多基金表现出马柯维茨式的优化特征，但它们在周转率、集中度和潜在因素方面具有异质表现。

为了分析和解释端到端模型，我们开发了一系列测试方法来解释模型，并展示了基准的专家标签以线性可解释的方式包含在我们的模型编码中。

我们的框架提供了数据驱动的方法，用于在市场模拟、策略归因和监管审查等应用中 characterizing 投资策略。 

---
# Segmentation over Complexity: Evaluating Ensemble and Hybrid Approaches for Anomaly Detection in Industrial Time Series 

**Title (ZH)**: 复杂性上的分割评估：工业时序异常检测的集成与混合方法评价 

**Authors**: Emilio Mastriani, Alessandro Costa, Federico Incardona, Kevin Munari, Sebastiano Spinello  

**Link**: [PDF](https://arxiv.org/pdf/2510.26159)  

**Abstract**: In this study, we investigate the effectiveness of advanced feature engineering and hybrid model architectures for anomaly detection in a multivariate industrial time series, focusing on a steam turbine system. We evaluate the impact of change point-derived statistical features, clustering-based substructure representations, and hybrid learning strategies on detection performance. Despite their theoretical appeal, these complex approaches consistently underperformed compared to a simple Random Forest + XGBoost ensemble trained on segmented data. The ensemble achieved an AUC-ROC of 0.976, F1-score of 0.41, and 100% early detection within the defined time window. Our findings highlight that, in scenarios with highly imbalanced and temporally uncertain data, model simplicity combined with optimized segmentation can outperform more sophisticated architectures, offering greater robustness, interpretability, and operational utility. 

**Abstract (ZH)**: 本研究探讨了高级特征工程和混合模型架构在多变量工业时间序列异常检测中的有效性，重点研究了蒸汽涡轮系统。我们评估了变化点衍生统计特征、基于聚类的子结构表示以及混合学习策略对检测性能的影响。尽管这些复杂的方法在理论上具有吸引力，但在所有情况下，基于分割数据训练的简单随机森林+极端随机树 ensemble 的表现始终优于这些复杂方法。该 ensemble 达到了 0.976 的 AUC-ROC、0.41 的 F1 分数，并在定义的时间窗口内实现了 100% 的早期检测率。我们的研究结果表明，在高度不平衡且时间上具有不确定性数据的情景下，模型的简单性结合优化的分割可以优于更复杂的架构，提供更高的稳健性、可解释性和操作实用性。 

---
# Bridging the Gap Between Molecule and Textual Descriptions via Substructure-aware Alignment 

**Title (ZH)**: 通过子结构感知对齐弥合分子与文本描述之间的差距 

**Authors**: Hyuntae Park, Yeachan Kim, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.26157)  

**Abstract**: Molecule and text representation learning has gained increasing interest due to its potential for enhancing the understanding of chemical information. However, existing models often struggle to capture subtle differences between molecules and their descriptions, as they lack the ability to learn fine-grained alignments between molecular substructures and chemical phrases. To address this limitation, we introduce MolBridge, a novel molecule-text learning framework based on substructure-aware alignments. Specifically, we augment the original molecule-description pairs with additional alignment signals derived from molecular substructures and chemical phrases. To effectively learn from these enriched alignments, MolBridge employs substructure-aware contrastive learning, coupled with a self-refinement mechanism that filters out noisy alignment signals. Experimental results show that MolBridge effectively captures fine-grained correspondences and outperforms state-of-the-art baselines on a wide range of molecular benchmarks, highlighting the significance of substructure-aware alignment in molecule-text learning. 

**Abstract (ZH)**: 分子和文本表示学习由于能够增强对化学信息的理解而日益受到关注。然而，现有模型往往难以捕捉分子及其描述之间的细微差异，因为它们缺乏学习分子亚结构和化学短语之间细粒度对齐的能力。为了解决这一局限性，我们介绍了MolBridge，一种基于亚结构感知对齐的新颖分子-文本学习框架。具体来说，MolBridge 通过增加源自分子亚结构和化学短语的额外对齐信号来扩充原始的分子-描述配对。为了有效从这些增强的对齐信号中学习，MolBridge 使用了亚结构感知对比学习，并结合了一种自我精炼机制来筛选掉噪音对齐信号。实验结果表明，MolBridge 能够有效捕捉细粒度对应关系，并在多种分子基准测试上优于最先进的基线，突出了亚结构感知对齐在分子-文本学习中的重要性。 

---
# MV-MLM: Bridging Multi-View Mammography and Language for Breast Cancer Diagnosis and Risk Prediction 

**Title (ZH)**: MV-MLM：桥梁多视图乳腺X线影像与语言在乳腺癌诊断和风险预测中的应用 

**Authors**: Shunjie-Fabian Zheng, Hyeonjun Lee, Thijs Kooi, Ali Diba  

**Link**: [PDF](https://arxiv.org/pdf/2510.26151)  

**Abstract**: Large annotated datasets are essential for training robust Computer-Aided Diagnosis (CAD) models for breast cancer detection or risk prediction. However, acquiring such datasets with fine-detailed annotation is both costly and time-consuming. Vision-Language Models (VLMs), such as CLIP, which are pre-trained on large image-text pairs, offer a promising solution by enhancing robustness and data efficiency in medical imaging tasks. This paper introduces a novel Multi-View Mammography and Language Model for breast cancer classification and risk prediction, trained on a dataset of paired mammogram images and synthetic radiology reports. Our MV-MLM leverages multi-view supervision to learn rich representations from extensive radiology data by employing cross-modal self-supervision across image-text pairs. This includes multiple views and the corresponding pseudo-radiology reports. We propose a novel joint visual-textual learning strategy to enhance generalization and accuracy performance over different data types and tasks to distinguish breast tissues or cancer characteristics(calcification, mass) and utilize these patterns to understand mammography images and predict cancer risk. We evaluated our method on both private and publicly available datasets, demonstrating that the proposed model achieves state-of-the-art performance in three classification tasks: (1) malignancy classification, (2) subtype classification, and (3) image-based cancer risk prediction. Furthermore, the model exhibits strong data efficiency, outperforming existing fully supervised or VLM baselines while trained on synthetic text reports and without the need for actual radiology reports. 

**Abstract (ZH)**: 大型注释数据集是训练用于乳腺癌检测或风险预测的鲁棒计算机辅助诊断(CAD)模型的关键。然而，获取具有细致注释的数据集既昂贵又耗时。视觉-语言模型(VLMs)，如CLIP，这些模型是在大规模图像-文本对上预训练的，通过在医学成像任务中增强鲁棒性和数据效率提供了有希望的解决方案。本文介绍了一种用于乳腺癌分类和风险预测的新型多视图乳腺X线摄影和语言模型，该模型基于配对的乳腺X线摄影图像和合成放射学报告数据集进行训练。我们的MV-MLM利用多视图监督，通过在图像-文本对之间进行跨模态自监督学习来从大量放射学数据中学习丰富的表示，包括多个视图和相应的伪放射学报告。我们提出了一种新的联合视觉-文本学习策略，以增强在不同数据类型和任务上的泛化能力和准确性能，区分乳腺组织或癌症特征（钙化、肿块），利用这些模式来理解乳腺X线摄影图像并预测癌症风险。我们在私人和公开可用的两个数据集上评估了该方法，结果表明所提模型在三项分类任务中达到了最先进的性能：(1) 恶性程度分类，(2) 亚型分类，(3) 图像基癌症风险预测。此外，该模型表现出强大的数据效率，在使用合成文本报告进行训练而无需实际放射学报告的情况下，优于现有完全监督或VLM基线模型。 

---
# Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation 

**Title (ZH)**: 超越合成基准：评估大语言模型在实际世界类级代码生成中的性能 

**Authors**: Musfiqur Rahman, SayedHassan Khatoonabadi, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2510.26130)  

**Abstract**: Large language models (LLMs) have advanced code generation at the function level, yet their ability to produce correct class-level implementations in authentic software projects remains poorly understood. This work introduces a novel benchmark derived from open-source repositories, comprising real-world classes divided into seen and unseen partitions to evaluate generalization under practical conditions. The evaluation examines multiple LLMs under varied input specifications, retrieval-augmented configurations, and documentation completeness levels.
Results reveal a stark performance disparity: LLMs achieve 84% to 89% correctness on established synthetic benchmarks but only 25% to 34% on real-world class tasks, with negligible differences between familiar and novel codebases. Comprehensive docstrings yield modest gains of 1% to 3% in functional accuracy, though statistical significance is rare. Retrieval-augmented generation proves most effective with partial documentation, improving correctness by 4% to 7% by supplying concrete implementation patterns absent from specifications. Error profiling identifies AttributeError, TypeError, and AssertionError as dominant failure modes (84% of cases), with synthetic tests overemphasizing assertion issues and real-world scenarios highlighting type and attribute mismatches. Retrieval augmentation reduces logical flaws but can introduce dependency conflicts.
The benchmark and analysis expose critical limitations in current LLM capabilities for class-level engineering, offering actionable insights for enhancing context modelling, documentation strategies, and retrieval integration in production code assistance tools. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在函数级代码生成方面取得了进展，但在真实软件项目中生成正确的类级实现的能力尚不完全理解。本研究引入了一个源自开源仓库的新基准，将实际类划分为已见和未见部分，以在实际条件下评估泛化能力。评估在各种输入规范、检索增强配置和文档完整性水平下对多个LLM进行测试。

结果显示，性能差距明显：LLM在成熟的合成基准测试中实现84%到89%的正确性，而在真实世界的类任务中仅实现25%到34%的正确性，熟悉和新颖代码库之间的差别不大。全面的文档字符串在功能准确性方面仅带来1%到3%的提升，尽管统计显著性罕见。检索增强生成在部分文档情况下最有效，通过提供规范中缺失的具体实现模式，正确性提高4%到7%。错误剖析确定了AttributeError、TypeError和AssertionError为主要失败模式（占84%的情况），合成测试过度强调断言问题，而真实世界场景则突出了类型和属性不符。检索增强减少了逻辑错误，但可能引入依赖冲突。

该基准和分析揭示了当前LLM在类级工程方面的重要局限性，为增强上下文建模、文档策略和检索集成提供可操作的见解，以改进生产代码辅助工具。 

---
# WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios 

**Title (ZH)**: Waymo 开放数据集：面向具有挑战性的长尾场景的端到端驾驶 

**Authors**: Runsheng Xu, Hubert Lin, Wonseok Jeon, Hao Feng, Yuliang Zou, Liting Sun, John Gorman, Kate Tolstaya, Sarah Tang, Brandyn White, Ben Sapp, Mingxing Tan, Jyh-Jing Hwang, Drago Anguelov  

**Link**: [PDF](https://arxiv.org/pdf/2510.26125)  

**Abstract**: Vision-based end-to-end (E2E) driving has garnered significant interest in the research community due to its scalability and synergy with multimodal large language models (MLLMs). However, current E2E driving benchmarks primarily feature nominal scenarios, failing to adequately test the true potential of these systems. Furthermore, existing open-loop evaluation metrics often fall short in capturing the multi-modal nature of driving or effectively evaluating performance in long-tail scenarios. To address these gaps, we introduce the Waymo Open Dataset for End-to-End Driving (WOD-E2E). WOD-E2E contains 4,021 driving segments (approximately 12 hours), specifically curated for challenging long-tail scenarios that that are rare in daily life with an occurring frequency of less than 0.03%. Concretely, each segment in WOD-E2E includes the high-level routing information, ego states, and 360-degree camera views from 8 surrounding cameras. To evaluate the E2E driving performance on these long-tail situations, we propose a novel open-loop evaluation metric: Rater Feedback Score (RFS). Unlike conventional metrics that measure the distance between predicted way points and the logs, RFS measures how closely the predicted trajectory matches rater-annotated trajectory preference labels. We have released rater preference labels for all WOD-E2E validation set segments, while the held out test set labels have been used for the 2025 WOD-E2E Challenge. Through our work, we aim to foster state of the art research into generalizable, robust, and safe end-to-end autonomous driving agents capable of handling complex real-world situations. 

**Abstract (ZH)**: 基于视觉的端到端驾驶因其实现的 scalability 和与多模态大语言模型的协同效应而在研究界引起了广泛关注。然而，当前的端到端驾驶基准主要包含常规场景，未能充分测试这些系统的真正潜力。此外，现有的开环评估指标往往无法捕捉驾驶的多模态本质或有效评估长尾场景中的性能。为解决这些问题，我们介绍了Waymo 开放数据集用于端到端驾驶 (WOD-E2E)。WOD-E2E 包含 4,021 个驾驶段（约 12 小时），特别为日常生活中罕见的、发生频率低于 0.03% 的挑战性长尾场景精心策划。具体而言，每个WOD-E2E 的段落包括高级路线信息、ego 状态以及来自 8 个周围摄像头的全景视图。为评估这些长尾情况下的端到端驾驶性能，我们提出了一种新的开环评估指标：评分反馈评分 (RFS)。不同于传统的衡量预测路径点与日志之间距离的指标，RFS 通过预测轨迹与评分标注的路径偏好标签的吻合程度来衡量。我们为所有 WOD-E2E 验证集段落提供了评分偏好标签，而被保留的测试集标签则用于 2025 年 WOD-E2E 挑战赛。通过我们的工作，我们旨在促进对通用、鲁棒和安全的端到端自主驾驶代理的研究，这些代理能够处理复杂的现实世界情况。 

---
# EgoExo-Con: Exploring View-Invariant Video Temporal Understanding 

**Title (ZH)**: egoExo-Con: 探索视角不变的视频时间理解 

**Authors**: Minjoon Jung, Junbin Xiao, Junghyun Kim, Byoung-Tak Zhang, Angela Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.26113)  

**Abstract**: Can Video-LLMs achieve consistent temporal understanding when videos capture the same event from different viewpoints? To study this, we introduce EgoExo-Con (Consistency), a benchmark of comprehensively synchronized egocentric and exocentric video pairs with human-refined queries in natural language. EgoExo-Con emphasizes two temporal understanding tasks: Temporal Verification and Temporal Grounding. It evaluates not only correctness but consistency across viewpoints. Our analysis reveals two critical limitations of existing Video-LLMs: (1) models often fail to maintain consistency, with results far worse than their single-view performances. (2) When naively finetuned with synchronized videos of both viewpoints, the models show improved consistency but often underperform those trained on a single view. For improvements, we propose View-GRPO, a novel reinforcement learning framework that effectively strengthens view-specific temporal reasoning while encouraging consistent comprehension across viewpoints. Our method demonstrates its superiority over naive SFT and GRPO, especially for improving cross-view consistency. All resources will be made publicly available. 

**Abstract (ZH)**: 视频LLMs在捕捉相同事件不同视角的视频时能否实现一致的时间理解？为研究这一问题，我们引入了EgoExo-Con（一致度），这是一个全面同步的第一人称和第三人称视频对基准，包含人类精炼的自然语言查询。EgoExo-Con 强调两项时间理解任务：时间验证和时间定位。它不仅评估正确性，还评估不同视角之间的一致性。我们的分析揭示了现有视频LLMs的两个关键局限性：(1) 模型往往无法保持一致性，其结果远逊于单视角的性能。(2) 当使用同步视角的视频进行简单的微调时，模型的一致性有所提高，但通常在跨视角一致性上逊于仅在一个视角上训练的模型。为改进这些问题，我们提出了View-GRPO，一个新颖的强化学习框架，能够在增强特定视角的时间推理的同时促进跨视角的一致理解。我们的方法在改善跨视角一致性方面优于简单的SFT和GRPO。所有资源将公开提供。 

---
# Security Risk of Misalignment between Text and Image in Multi-modal Model 

**Title (ZH)**: 多模态模型中文本与图像不一致的security风险 

**Authors**: Xiaosen Wang, Zhijin Ge, Shaokang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26105)  

**Abstract**: Despite the notable advancements and versatility of multi-modal diffusion models, such as text-to-image models, their susceptibility to adversarial inputs remains underexplored. Contrary to expectations, our investigations reveal that the alignment between textual and Image modalities in existing diffusion models is inadequate. This misalignment presents significant risks, especially in the generation of inappropriate or Not-Safe-For-Work (NSFW) content. To this end, we propose a novel attack called Prompt-Restricted Multi-modal Attack (PReMA) to manipulate the generated content by modifying the input image in conjunction with any specified prompt, without altering the prompt itself. PReMA is the first attack that manipulates model outputs by solely creating adversarial images, distinguishing itself from prior methods that primarily generate adversarial prompts to produce NSFW content. Consequently, PReMA poses a novel threat to the integrity of multi-modal diffusion models, particularly in image-editing applications that operate with fixed prompts. Comprehensive evaluations conducted on image inpainting and style transfer tasks across various models confirm the potent efficacy of PReMA. 

**Abstract (ZH)**: 尽管多模态扩散模型，如文本到图像模型，在进展和灵活性方面取得了显著成就，但这些模型对对抗输入的敏感性依然研究不足。与预期相反，我们的研究表明，现有扩散模型中文本模态和图像模态之间的对齐是不充分的。这种不对齐在生成不适当或不适合工作（NSFW）内容时带来了重大风险。为此，我们提出了一种名为Prompt-Restricted Multi-modal Attack（PReMA）的新型攻击方法，通过修改输入图像并与任何指定的提示相结合来操纵生成的内容，而不改变提示本身。PReMA是第一个仅通过创建对抗图像来操纵模型输出的攻击方法，区别于先前主要生成对抗提示的方法。因此，PReMA对使用固定提示的图像编辑应用程序中的多模态扩散模型完整性构成了新的威胁。在各种模型上进行的图像填补和风格迁移任务中的全面评估证实了PReMA的强大效果。 

---
# SAFE: A Novel Approach to AI Weather Evaluation through Stratified Assessments of Forecasts over Earth 

**Title (ZH)**: SAFE：一种通过分层评估地球预报的新方法以进行AI气象评价 

**Authors**: Nick Masi, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2510.26099)  

**Abstract**: The dominant paradigm in machine learning is to assess model performance based on average loss across all samples in some test set. This amounts to averaging performance geospatially across the Earth in weather and climate settings, failing to account for the non-uniform distribution of human development and geography. We introduce Stratified Assessments of Forecasts over Earth (SAFE), a package for elucidating the stratified performance of a set of predictions made over Earth. SAFE integrates various data domains to stratify by different attributes associated with geospatial gridpoints: territory (usually country), global subregion, income, and landcover (land or water). This allows us to examine the performance of models for each individual stratum of the different attributes (e.g., the accuracy in every individual country). To demonstrate its importance, we utilize SAFE to benchmark a zoo of state-of-the-art AI-based weather prediction models, finding that they all exhibit disparities in forecasting skill across every attribute. We use this to seed a benchmark of model forecast fairness through stratification at different lead times for various climatic variables. By moving beyond globally-averaged metrics, we for the first time ask: where do models perform best or worst, and which models are most fair? To support further work in this direction, the SAFE package is open source and available at this https URL 

**Abstract (ZH)**: 地球上的分层预测评估：一种揭示地球预测集性能的包（Stratified Assessments of Forecasts over Earth: A Package for Elucidating the Stratified Performance of Predictions Made over Earth） 

---
# Network-Constrained Policy Optimization for Adaptive Multi-agent Vehicle Routing 

**Title (ZH)**: 网络约束的政策优化方法用于自适应多agent车辆路径规划 

**Authors**: Fazel Arasteh, Arian Haghparast, Manos Papagelis  

**Link**: [PDF](https://arxiv.org/pdf/2510.26089)  

**Abstract**: Traffic congestion in urban road networks leads to longer trip times and higher emissions, especially during peak periods. While the Shortest Path First (SPF) algorithm is optimal for a single vehicle in a static network, it performs poorly in dynamic, multi-vehicle settings, often worsening congestion by routing all vehicles along identical paths. We address dynamic vehicle routing through a multi-agent reinforcement learning (MARL) framework for coordinated, network-aware fleet navigation. We first propose Adaptive Navigation (AN), a decentralized MARL model where each intersection agent provides routing guidance based on (i) local traffic and (ii) neighborhood state modeled using Graph Attention Networks (GAT). To improve scalability in large networks, we further propose Hierarchical Hub-based Adaptive Navigation (HHAN), an extension of AN that assigns agents only to key intersections (hubs). Vehicles are routed hub-to-hub under agent control, while SPF handles micro-routing within each hub region. For hub coordination, HHAN adopts centralized training with decentralized execution (CTDE) under the Attentive Q-Mixing (A-QMIX) framework, which aggregates asynchronous vehicle decisions via attention. Hub agents use flow-aware state features that combine local congestion and predictive dynamics for proactive routing. Experiments on synthetic grids and real urban maps (Toronto, Manhattan) show that AN reduces average travel time versus SPF and learning baselines, maintaining 100% routing success. HHAN scales to networks with hundreds of intersections, achieving up to 15.9% improvement under heavy traffic. These findings highlight the potential of network-constrained MARL for scalable, coordinated, and congestion-aware routing in intelligent transportation systems. 

**Abstract (ZH)**: 动态交通网络中的交通拥堵导致旅行时间增加和排放量上升，特别是在高峰时段。虽然最短路径优先（SPF）算法在静态网络中对单一车辆是最优的，但在动态、多辆车的环境中表现不佳，往往会通过将所有车辆导向相同的路径来加剧拥堵。我们通过多代理 reinforcement 学习（MARL）框架解决动态车辆路由问题，以实现网络感知的车队协同导航。我们首先提出自适应导航（AN），这是一种去中心化的 MARL 模型，其中每个交叉口代理基于 （i）局部交通和 （ii）使用图注意网络（GAT）建模的邻域状态提供路径指导。为了在大网络中提高可扩展性，我们进一步提出分层中心节点自适应导航（HHAN），这是 AN 的扩展，仅将代理分配给关键交叉口（枢纽）。车辆在代理控制下枢纽到枢纽地行驶，而 SPF 负责每个枢纽区域内的微路径规划。对于枢纽协调，HHAN 在注意力增强 Q 混合（A-QMIX）框架下采用集中训练与去中心化执行（CTDE），通过注意力机制聚合异步车辆决策。枢纽代理使用流量感知状态特征，结合局部拥堵和预测动态实现主动导航。在合成网格和实际城市地图（多伦多、曼哈顿）上的实验显示，AN 在平均旅行时间方面优于 SPF 和学习基准，并保持100%的导航成功率。HHAN 可扩展到包含数百个交叉口的网络，在重交通情况下可实现高达 15.9% 的改进。这些发现突显了受网络约束的 MARL 在智能交通系统中实现可扩展、协同和拥堵感知路由的潜力。 

---
# Nirvana: A Specialized Generalist Model With Task-Aware Memory Mechanism 

**Title (ZH)**: 涅槃：一种具有任务意识记忆机制的专门通用模型 

**Authors**: Yuhua Jiang, Shuang Cheng, Yihao Liu, Ermo Hua, Che Jiang, Weigao Sun, Yu Cheng, Feifei Gao, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.26083)  

**Abstract**: Specialized Generalist Models (SGMs) aim to preserve broad capabilities while achieving expert-level performance in target domains. However, traditional LLM structures including Transformer, Linear Attention, and hybrid models do not employ specialized memory mechanism guided by task information. In this paper, we present Nirvana, an SGM with specialized memory mechanism, linear time complexity, and test-time task information extraction. Besides, we propose the Task-Aware Memory Trigger ($\textit{Trigger}$) that flexibly adjusts memory mechanism based on the current task's requirements. In Trigger, each incoming sample is treated as a self-supervised fine-tuning task, enabling Nirvana to adapt its task-related parameters on the fly to domain shifts. We also design the Specialized Memory Updater ($\textit{Updater}$) that dynamically memorizes the context guided by Trigger. We conduct experiments on both general language tasks and specialized medical tasks. On a variety of natural language modeling benchmarks, Nirvana achieves competitive or superior results compared to the existing LLM structures. To prove the effectiveness of Trigger on specialized tasks, we test Nirvana's performance on a challenging medical task, i.e., Magnetic Resonance Imaging (MRI). We post-train frozen Nirvana backbone with lightweight codecs on paired electromagnetic signals and MRI images. Despite the frozen Nirvana backbone, Trigger guides the model to adapt to the MRI domain with the change of task-related parameters. Nirvana achieves higher-quality MRI reconstruction compared to conventional MRI models as well as the models with traditional LLMs' backbone, and can also generate accurate preliminary clinical reports accordingly. 

**Abstract (ZH)**: 特殊通用模型（SGMs）旨在保留广泛的能力同时在目标领域达到专家级别的性能。然而，传统的LLM结构，包括Transformer、线性注意力和混合模型，不采用基于任务信息的专门记忆机制。本文提出了一种名为Nirvana的SGM，该模型具有专门的记忆机制、线性时间复杂度和测试时任务信息提取能力。此外，我们提出了任务感知记忆触发器（$\textit{Trigger}$），该机制可根据当前任务的需求灵活调整记忆机制。在$\textit{Trigger}$中，每个输入样本被视为一个自我监督的微调任务，使Nirvana能够根据领域转移实时调整与任务相关参数。我们还设计了专门的记忆更新器（$\textit{Updater}$），该更新器根据$\textit{Trigger}$动态地记忆上下文。我们在一般语言任务和专门的医疗任务上进行了实验。在多种自然语言建模基准测试中，Nirvana取得了与现有LLM结构相当或更优的结果。为了证明$\textit{Trigger}$在专门任务上的有效性，我们在配对的电磁信号和MRI图像上对冻结的Nirvana骨干进行后训练，并使用轻量级编解码器。即使冻结的Nirvana骨干，$\textit{Trigger}$也能通过调整与任务相关参数指导模型适应MRI领域。与传统的MRI模型以及以传统LLM架构为基础的模型相比，Nirvana在MRI重建质量上表现出更高水平，并能够生成准确的初步临床报告。 

---
# Learning Geometry: A Framework for Building Adaptive Manifold Models through Metric Optimization 

**Title (ZH)**: 学习几何：通过度量优化构建自适应流形模型的框架 

**Authors**: Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26068)  

**Abstract**: This paper proposes a novel paradigm for machine learning that moves beyond traditional parameter optimization. Unlike conventional approaches that search for optimal parameters within a fixed geometric space, our core idea is to treat the model itself as a malleable geometric entity. Specifically, we optimize the metric tensor field on a manifold with a predefined topology, thereby dynamically shaping the geometric structure of the model space. To achieve this, we construct a variational framework whose loss function carefully balances data fidelity against the intrinsic geometric complexity of the manifold. The former ensures the model effectively explains observed data, while the latter acts as a regularizer, penalizing overly curved or irregular geometries to encourage simpler models and prevent overfitting. To address the computational challenges of this infinite-dimensional optimization problem, we introduce a practical method based on discrete differential geometry: the continuous manifold is discretized into a triangular mesh, and the metric tensor is parameterized by edge lengths, enabling efficient optimization using automatic differentiation tools. Theoretical analysis reveals a profound analogy between our framework and the Einstein-Hilbert action in general relativity, providing an elegant physical interpretation for the concept of "data-driven geometry". We further argue that even with fixed topology, metric optimization offers significantly greater expressive power than models with fixed geometry. This work lays a solid foundation for constructing fully dynamic "meta-learners" capable of autonomously evolving their geometry and topology, and it points to broad application prospects in areas such as scientific model discovery and robust representation learning. 

**Abstract (ZH)**: 本文提出了一种超越传统参数优化的新范式，将模型本身视为可塑的几何实体。具体地，我们优化具有预定义拓扑结构的流形上的度量张量场，从而动态塑造模型空间的几何结构。为此，我们构建了一个变分框架，其损失函数平衡了数据保真度与流形内在几何复杂度。前者确保模型能有效解释观测数据，而后者作为正则化项，惩罚过于弯曲或不规则的几何结构，以鼓励更简单的模型并防止过拟合。为解决这一无限维优化问题的计算挑战，我们引入了一种基于离散微分几何的实用方法：将连续流形离散化为三角网，度量张量通过边长参数化，从而可以利用自动微分工具进行高效优化。理论分析揭示了我们框架与广义相对论中的爱因斯坦-希尔伯트作用之间的深刻类比，为“数据驱动几何”的概念提供了优雅的物理解释。我们进一步认为，即使拓扑固定，度量优化比固定几何结构的模型提供了显著更大的表达能力。本文为构建能够自主进化其几何结构和拓扑结构的完全动态“元学习器”奠定了坚实基础，并指出了其在科学模型发现和稳健表示学习等领域的广泛应用前景。 

---
# Data-driven Projection Generation for Efficiently Solving Heterogeneous Quadratic Programming Problems 

**Title (ZH)**: 基于数据驱动的投影生成方法用于高效求解异构二次规划问题 

**Authors**: Tomoharu Iwata, Futoshi Futami  

**Link**: [PDF](https://arxiv.org/pdf/2510.26061)  

**Abstract**: We propose a data-driven framework for efficiently solving quadratic programming (QP) problems by reducing the number of variables in high-dimensional QPs using instance-specific projection. A graph neural network-based model is designed to generate projections tailored to each QP instance, enabling us to produce high-quality solutions even for previously unseen problems. The model is trained on heterogeneous QPs to minimize the expected objective value evaluated on the projected solutions. This is formulated as a bilevel optimization problem; the inner optimization solves the QP under a given projection using a QP solver, while the outer optimization updates the model parameters. We develop an efficient algorithm to solve this bilevel optimization problem, which computes parameter gradients without backpropagating through the solver. We provide a theoretical analysis of the generalization ability of solving QPs with projection matrices generated by neural networks. Experimental results demonstrate that our method produces high-quality feasible solutions with reduced computation time, outperforming existing methods. 

**Abstract (ZH)**: 基于实例特定投影的数据驱动框架：通过减少高维QP中的变量数高效求解二次规划问题 

---
# Dynamic VLM-Guided Negative Prompting for Diffusion Models 

**Title (ZH)**: 动态VLM引导的负 Lawyers' Prompting for 扩散模型 

**Authors**: Hoyeon Chang, Seungjin Kim, Yoonseok Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.26052)  

**Abstract**: We propose a novel approach for dynamic negative prompting in diffusion models that leverages Vision-Language Models (VLMs) to adaptively generate negative prompts during the denoising process. Unlike traditional Negative Prompting methods that use fixed negative prompts, our method generates intermediate image predictions at specific denoising steps and queries a VLM to produce contextually appropriate negative prompts. We evaluate our approach on various benchmark datasets and demonstrate the trade-offs between negative guidance strength and text-image alignment. 

**Abstract (ZH)**: 我们提出了一种利用视觉语言模型在去噪过程中自适应生成动态负提示的新方法。我们在多个基准数据集上评估了该方法，并展示了负引导强度与文本图像对齐之间的权衡。 

---
# Do Students Debias Like Teachers? On the Distillability of Bias Mitigation Methods 

**Title (ZH)**: 学生像教师一样去偏差吗？关于偏差缓解方法的可提炼性研究 

**Authors**: Jiali Cheng, Chirag Agarwal, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.26038)  

**Abstract**: Knowledge distillation (KD) is an effective method for model compression and transferring knowledge between models. However, its effect on model's robustness against spurious correlations that degrade performance on out-of-distribution data remains underexplored. This study investigates the effect of knowledge distillation on the transferability of ``debiasing'' capabilities from teacher models to student models on natural language inference (NLI) and image classification tasks. Through extensive experiments, we illustrate several key findings: (i) overall the debiasing capability of a model is undermined post-KD; (ii) training a debiased model does not benefit from injecting teacher knowledge; (iii) although the overall robustness of a model may remain stable post-distillation, significant variations can occur across different types of biases; and (iv) we pin-point the internal attention pattern and circuit that causes the distinct behavior post-KD. Given the above findings, we propose three effective solutions to improve the distillability of debiasing methods: developing high quality data for augmentation, implementing iterative knowledge distillation, and initializing student models with weights obtained from teacher models. To the best of our knowledge, this is the first study on the effect of KD on debiasing and its interenal mechanism at scale. Our findings provide understandings on how KD works and how to design better debiasing methods. 

**Abstract (ZH)**: 知识蒸馏对模型去偏差能力迁移的影响研究 

---
# SIRAJ: Diverse and Efficient Red-Teaming for LLM Agents via Distilled Structured Reasoning 

**Title (ZH)**: SIRAJ：通过提炼结构化推理实现的多元高效红队测试方法for LLM代理 

**Authors**: Kaiwen Zhou, Ahmed Elgohary, A S M Iftekhar, Amin Saied  

**Link**: [PDF](https://arxiv.org/pdf/2510.26037)  

**Abstract**: The ability of LLM agents to plan and invoke tools exposes them to new safety risks, making a comprehensive red-teaming system crucial for discovering vulnerabilities and ensuring their safe deployment. We present SIRAJ: a generic red-teaming framework for arbitrary black-box LLM agents. We employ a dynamic two-step process that starts with an agent definition and generates diverse seed test cases that cover various risk outcomes, tool-use trajectories, and risk sources. Then, it iteratively constructs and refines model-based adversarial attacks based on the execution trajectories of former attempts. To optimize the red-teaming cost, we present a model distillation approach that leverages structured forms of a teacher model's reasoning to train smaller models that are equally effective. Across diverse evaluation agent settings, our seed test case generation approach yields 2 -- 2.5x boost to the coverage of risk outcomes and tool-calling trajectories. Our distilled 8B red-teamer model improves attack success rate by 100%, surpassing the 671B Deepseek-R1 model. Our ablations and analyses validate the effectiveness of the iterative framework, structured reasoning, and the generalization of our red-teamer models. 

**Abstract (ZH)**: LLM代理规划和调用工具的能力使其面临新的安全风险，因此需要全面的红队系统来发现漏洞并确保其安全部署。我们提出了SIRAJ：任意黑盒LLM代理的一般红队框架。我们采用了一个动态的两步过程，首先定义代理，并生成涵盖各种风险结果、工具使用轨迹和风险源的多样化的种子测试案例。然后，根据前次尝试的执行轨迹迭代构建和优化基于模型的对抗攻击。为了优化红队的成本，我们提出了一种模型蒸馏方法，利用教师模型推理的结构化形式来训练更小但同样有效的模型。在多种评估代理设置下，我们的种子测试案例生成方法使风险结果和工具调用轨迹的覆盖范围提高了2-2.5倍。我们蒸馏的8B红队模型将攻击成功率提高了100%，超过了671B的Deepseek-R1模型。我们的消融实验和分析验证了迭代框架、结构化推理和我们红队模型泛化的有效性。 

---
# Artificial Intelligence-Enabled Analysis of Radiology Reports: Epidemiology and Consequences of Incidental Thyroid Findings 

**Title (ZH)**: 人工智能赋能的放射学报告分析：偶发性甲状腺发现的流行病学和后果 

**Authors**: Felipe Larios, Mariana Borras-Osorio, Yuqi Wu, Ana Gabriela Claros, David Toro-Tobon, Esteban Cabezas, Ricardo Loor-Torres, Maria Mateo Chavez, Kerly Guevara Maldonado, Luis Vilatuna Andrango, Maria Lizarazo Jimenez, Ivan Mateo Alzamora, Misk Al Zahidy, Marcelo Montero, Ana Cristina Proano, Cristian Soto Jacome, Jungwei W. Fan, Oscar J. Ponce-Ponte, Megan E. Branda, Naykky Singh Ospina, Juan P. Brito  

**Link**: [PDF](https://arxiv.org/pdf/2510.26032)  

**Abstract**: Importance Incidental thyroid findings (ITFs) are increasingly detected on imaging performed for non-thyroid indications. Their prevalence, features, and clinical consequences remain undefined. Objective To develop, validate, and deploy a natural language processing (NLP) pipeline to identify ITFs in radiology reports and assess their prevalence, features, and clinical outcomes. Design, Setting, and Participants Retrospective cohort of adults without prior thyroid disease undergoing thyroid-capturing imaging at Mayo Clinic sites from July 1, 2017, to September 30, 2023. A transformer-based NLP pipeline identified ITFs and extracted nodule characteristics from image reports from multiple modalities and body regions. Main Outcomes and Measures Prevalence of ITFs, downstream thyroid ultrasound, biopsy, thyroidectomy, and thyroid cancer diagnosis. Logistic regression identified demographic and imaging-related factors. Results Among 115,683 patients (mean age, 56.8 [SD 17.2] years; 52.9% women), 9,077 (7.8%) had an ITF, of which 92.9% were nodules. ITFs were more likely in women, older adults, those with higher BMI, and when imaging was ordered by oncology or internal medicine. Compared with chest CT, ITFs were more likely via neck CT, PET, and nuclear medicine scans. Nodule characteristics were poorly documented, with size reported in 44% and other features in fewer than 15% (e.g. calcifications). Compared with patients without ITFs, those with ITFs had higher odds of thyroid nodule diagnosis, biopsy, thyroidectomy and thyroid cancer diagnosis. Most cancers were papillary, and larger when detected after ITFs vs no ITF. Conclusions ITFs were common and strongly associated with cascades leading to the detection of small, low-risk cancers. These findings underscore the role of ITFs in thyroid cancer overdiagnosis and the need for standardized reporting and more selective follow-up. 

**Abstract (ZH)**: 重要偶然性甲状腺发现（ITFs）在非甲状腺指征的影像检查中越来越常见。其流行率、特征和临床后果尚未明确定义。目的：开发、验证并部署自然语言处理（NLP）管道以识别放射学报告中的ITFs及其流行率、特征和临床结果。设计、地点和参与者：回顾性成人队列，无甲状腺疾病史，在Mayo Clinic站点从2017年7月1日至2023年9月30日期间接受甲状腺捕获影像检查。基于变压器的NLP管道识别ITFs并从多种影像模态和身体区域的影像报告中提取结节特征。主要结局和测量指标：ITFs的流行率、后续甲状腺超声检查、活检、甲状腺切除术和甲状腺癌诊断。逐步回归分析确定了与人口统计学和影像学相关因素。结果：在115,683名患者（平均年龄56.8岁[标准差17.2]；52.9%为女性）中，9,077名（7.8%）有ITF，其中92.9%为结节。ITFs在女性、老年人、BMI较高者及由肿瘤科或内科申请影像检查的人群中更为常见。与胸部CT相比，颈部CT、PET和放射性核素扫描更多发现ITFs。结节特征记录不佳，尺寸在44%的报告中记录，而其他特征在不到15%的报告中记录（例如，钙化）。与无ITFs的患者相比，有ITFs的患者甲状腺结节诊断、活检、甲状腺切除术和甲状腺癌诊断的可能性更高。大多数癌症为乳头状，且在通过ITFs检测时比未检测时更大。结论：ITFs常见，与检测微小、低风险癌症的级联过程密切相关。这些发现强调了ITFs在甲状腺癌过度诊断中的作用，并突出了标准化报告和更具选择性的随访的必要性。 

---
# Rethinking Cross-lingual Alignment: Balancing Transfer and Cultural Erasure in Multilingual LLMs 

**Title (ZH)**: 重新思考跨语言对齐：多语言LLM中转移与文化消除的平衡 

**Authors**: HyoJung Han, Sweta Agrawal, Eleftheria Briakou  

**Link**: [PDF](https://arxiv.org/pdf/2510.26024)  

**Abstract**: Cross-lingual alignment (CLA) aims to align multilingual representations, enabling Large Language Models (LLMs) to seamlessly transfer knowledge across languages. While intuitive, we hypothesize, this pursuit of representational convergence can inadvertently cause "cultural erasure", the functional loss of providing culturally-situated responses that should diverge based on the query language. In this work, we systematically analyze this trade-off by introducing a holistic evaluation framework, the transfer-localization plane, which quantifies both desirable knowledge transfer and undesirable cultural erasure. Using this framework, we re-evaluate recent CLA approaches and find that they consistently improve factual transfer at the direct cost of cultural localization across all six languages studied. Our investigation into the internal representations of these models reveals a key insight: universal factual transfer and culturally-specific knowledge are optimally steerable at different model layers. Based on this finding, we propose Surgical Steering, a novel inference-time method that disentangles these two objectives. By applying targeted activation steering to distinct layers, our approach achieves a better balance between the two competing dimensions, effectively overcoming the limitations of current alignment techniques. 

**Abstract (ZH)**: 跨语言对齐 (CLA) 的目标是使多语言表示对齐，从而使大型语言模型（LLMs）能够无缝地在语言之间转移知识。虽然直观上如此，但我们推测，这种表征收敛的追求可能会无意中导致“文化抹除”，即功能性地丧失提供基于查询语言应有所差异的文化情境响应的能力。在本工作中，我们通过引入一个综合评估框架——转移-本地化平面，系统地分析了这一权衡，该框架量化了期望的知识转移和不利的文化抹除。使用此框架，我们重新评估了近期的 CLA 方法，并发现它们在所有六种研究语言中都以一致的方式在直接牺牲文化本地化的同时提高了事实知识转移。我们对这些模型内部表示的研究揭示了一个关键见解：泛化的事实知识转移和文化特异性知识在不同的模型层上最优化地可调节。基于这一发现，我们提出了手术调节（Surgical Steering）这一新颖的推理时方法，以分离这两项目标。通过针对不同层应用有针对性的激活调节，我们的方法能够在两个竞争维度之间取得更好的平衡，有效地克服了当前对齐技术的局限性。 

---
# PORTool: Tool-Use LLM Training with Rewarded Tree 

**Title (ZH)**: PORTool: 带有奖励树的工具使用大型语言模型训练 

**Authors**: Feijie Wu, Weiwu Zhu, Yuxiang Zhang, Soumya Chatterjee, Jiarong Zhu, Fan Mo, Rodin Luo, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.26020)  

**Abstract**: Current tool-use large language models (LLMs) are trained on static datasets, enabling them to interact with external tools and perform multi-step, tool-integrated reasoning, which produces tool-call trajectories. However, these models imitate how a query is resolved in a generic tool-call routine, thereby failing to explore possible solutions and demonstrating limited performance in an evolved, dynamic tool-call environment. In this work, we propose PORTool, a reinforcement learning (RL) method that encourages a tool-use LLM to explore various trajectories yielding the correct answer. Specifically, this method starts with generating multiple rollouts for a given query, and some of them share the first few tool-call steps, thereby forming a tree-like structure. Next, we assign rewards to each step, based on its ability to produce a correct answer and make successful tool calls. A shared step across different trajectories receives the same reward, while different steps under the same fork receive different rewards. Finally, these step-wise rewards are used to calculate fork-relative advantages, blended with trajectory-relative advantages, to train the LLM for tool use. The experiments utilize 17 tools to address user queries, covering both time-sensitive and time-invariant topics. We conduct ablation studies to systematically justify the necessity and the design robustness of step-wise rewards. Furthermore, we compare the proposed PORTool with other training approaches and demonstrate significant improvements in final accuracy and the number of tool-call steps. 

**Abstract (ZH)**: 当前使用的大型语言模型（LLMs）是在静态数据集上训练的，使它们能够与外部工具交互并执行多步骤、工具集成推理，从而产生工具调用轨迹。然而，这些模型模仿了一种通用工具调用常规处理查询的方式，未能探索可能的解决方案，并且在不断进化的动态工具调用环境中表现有限。在本文中，我们提出了PORTool，这是一种强化学习（RL）方法，鼓励工具使用LLM探索各种产生正确答案的轨迹。具体而言，该方法从对给定查询生成多个展开开始，其中一些展开共享前几步的工具调用，从而形成一种树状结构。接下来，我们根据每一步产生正确答案和成功调用工具的能力为其分配奖励。在不同轨迹中共享的步骤获得相同的奖励，而同一分支下的不同步骤获得不同的奖励。最后，这些步骤奖励用于计算分支相对优势，并与轨迹相对优势混合，以训练LLM用于工具使用。实验利用17种工具处理用户查询，涵盖了时间敏感和时间不变的话题。我们进行了消融研究，系统地验证了步骤奖励的必要性和设计稳健性。此外，我们将提出的PORTool与其他训练方法进行比较，并展示了最终准确性和工具调用步数上的显著改进。 

---
# RADRON: Cooperative Localization of Ionizing Radiation Sources by MAVs with Compton Cameras 

**Title (ZH)**: RADRON：使用Compton相机的MAVs协作定位电离辐射源 

**Authors**: Petr Stibinger, Tomas Baca, Daniela Doubravova, Jan Rusnak, Jaroslav Solc, Jan Jakubek, Petr Stepan, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2510.26018)  

**Abstract**: We present a novel approach to localizing radioactive material by cooperating Micro Aerial Vehicles (MAVs). Our approach utilizes a state-of-the-art single-detector Compton camera as a highly sensitive, yet miniature detector of ionizing radiation. The detector's exceptionally low weight (40 g) opens up new possibilities of radiation detection by a team of cooperating agile MAVs. We propose a new fundamental concept of fusing the Compton camera measurements to estimate the position of the radiation source in real time even from extremely sparse measurements. The data readout and processing are performed directly onboard and the results are used in a dynamic feedback to drive the motion of the vehicles. The MAVs are stabilized in a tightly cooperating swarm to maximize the information gained by the Compton cameras, rapidly locate the radiation source, and even track a moving radiation source. 

**Abstract (ZH)**: 我们提出了一种由微型飞行器协作实现放射性物质定位的新方法。该方法利用最先进的单探测器克勒普顿相机作为离子辐射的高灵敏度微型探测器。探测器的极低重量（40克）为合作灵活的微型飞行器团队提供了新的辐射探测可能性。我们提出了一种新的基本概念，即融合克勒普顿相机测量数据以实时从极少的测量数据中估计放射源的位置。数据读取和处理直接在飞行器上进行，并将结果用于动态反馈以驱动飞行器的运动。微型飞行器在紧密合作的集群中稳定，以最大化克勒普顿相机获得的信息，快速定位放射源，并能追踪移动的放射源。 

---
# Climate Adaptation-Aware Flood Prediction for Coastal Cities Using Deep Learning 

**Title (ZH)**: 沿海城市基于深度学习的气候适应性洪水预测 

**Authors**: Bilal Hassan, Areg Karapetyan, Aaron Chung Hin Chow, Samer Madanat  

**Link**: [PDF](https://arxiv.org/pdf/2510.26017)  

**Abstract**: Climate change and sea-level rise (SLR) pose escalating threats to coastal cities, intensifying the need for efficient and accurate methods to predict potential flood hazards. Traditional physics-based hydrodynamic simulators, although precise, are computationally expensive and impractical for city-scale coastal planning applications. Deep Learning (DL) techniques offer promising alternatives, however, they are often constrained by challenges such as data scarcity and high-dimensional output requirements. Leveraging a recently proposed vision-based, low-resource DL framework, we develop a novel, lightweight Convolutional Neural Network (CNN)-based model designed to predict coastal flooding under variable SLR projections and shoreline adaptation scenarios. Furthermore, we demonstrate the ability of the model to generalize across diverse geographical contexts by utilizing datasets from two distinct regions: Abu Dhabi and San Francisco. Our findings demonstrate that the proposed model significantly outperforms state-of-the-art methods, reducing the mean absolute error (MAE) in predicted flood depth maps on average by nearly 20%. These results highlight the potential of our approach to serve as a scalable and practical tool for coastal flood management, empowering decision-makers to develop effective mitigation strategies in response to the growing impacts of climate change. Project Page: this https URL 

**Abstract (ZH)**: 气候变化和海平面上升对沿海城市的威胁日益加剧，强化了对有效和精确的洪水灾害预测方法的需求。传统的物理学基础水动力模拟器虽然精确，但对于城市规模的沿海规划应用来说计算成本高且不实用。深度学习技术提供了有前景的替代方案，但经常会受到数据稀缺性和高维输出要求的限制。利用一种最近提出的基于视觉、低资源消耗的深度学习框架，我们开发了一种新的轻量级卷积神经网络（CNN）基预测模型，用于预测在不同的海平面上升和海岸线适应场景下的沿海洪水。此外，通过使用来自阿布扎比和旧金山两个不同地区的数据集，我们展示了该模型在不同地理背景下泛化的能力。我们的研究发现，所提议的模型显著优于现有最佳方法，平均降低了预测洪水深度图的绝对均方误差（MAE）近20%。这些结果强调了我们方法在沿海洪水管理中的潜在作用，能够为应对气候变化日益加剧的影响制定有效的缓解策略提供支持。项目页面：这个 https URL 

---
# Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis 

**Title (ZH)**: 离散时间生存分析的双混合专家框架 

**Authors**: Hyeonjun Lee, Hyungseob Shin, Gunhee Nam, Hyeonsoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.26014)  

**Abstract**: Survival analysis is a task to model the time until an event of interest occurs, widely used in clinical and biomedical research. A key challenge is to model patient heterogeneity while also adapting risk predictions to both individual characteristics and temporal dynamics. We propose a dual mixture-of-experts (MoE) framework for discrete-time survival analysis. Our approach combines a feature-encoder MoE for subgroup-aware representation learning with a hazard MoE that leverages patient features and time embeddings to capture temporal dynamics. This dual-MoE design flexibly integrates with existing deep learning based survival pipelines. On METABRIC and GBSG breast cancer datasets, our method consistently improves performance, boosting the time-dependent C-index up to 0.04 on the test sets, and yields further gains when incorporated into the Consurv framework. 

**Abstract (ZH)**: 生存分析是建模感兴趣事件发生时间的任务，广泛应用于临床和生物医学研究。主要挑战在于同时建模患者异质性并根据个体特征和时间动态调整风险预测。我们提出了一种双重混合专家（MoE）框架用于离散时间生存分析。该方法结合了特征编码MoE进行亚组意识的表现学习，以及利用患者特征和时间嵌入的危险MoE来捕捉时间动态。这种双重MoE设计灵活地与现有的基于深度学习的生存分析管道集成。在METABRIC和GBSG乳腺癌数据集中，我们的方法一致提高了性能，在测试集上将时间依赖的C指数提升至0.04，并在与Consurv框架结合时进一步提高了性能。 

---
# The Quest for Reliable Metrics of Responsible AI 

**Title (ZH)**: 负责任人工智能可靠指标的追求 

**Authors**: Theresia Veronika Rampisela, Maria Maistro, Tuukka Ruotsalo, Christina Lioma  

**Link**: [PDF](https://arxiv.org/pdf/2510.26007)  

**Abstract**: The development of Artificial Intelligence (AI), including AI in Science (AIS), should be done following the principles of responsible AI. Progress in responsible AI is often quantified through evaluation metrics, yet there has been less work on assessing the robustness and reliability of the metrics themselves. We reflect on prior work that examines the robustness of fairness metrics for recommender systems as a type of AI application and summarise their key takeaways into a set of non-exhaustive guidelines for developing reliable metrics of responsible AI. Our guidelines apply to a broad spectrum of AI applications, including AIS. 

**Abstract (ZH)**: 负责人工智能原则下的人工智能（AI）及其科学应用（AIS）的发展应当通过负责任的AI评估指标来量化进展，但对这些评估指标自身的稳健性和可靠性进行评估的工作相对较少。我们借鉴了先前研究中关于推荐系统公平性指标稳健性的相关工作，并将其关键经验总结为一套非详尽的指导原则，以发展可靠的负责任AI评估指标。这些指导原则适用于广泛的人工智能应用，包括科学应用（AIS）。 

---
# DARTS: A Drone-Based AI-Powered Real-Time Traffic Incident Detection System 

**Title (ZH)**: 基于无人机的AI驱动实时交通事件检测系统 

**Authors**: Bai Li, Achilleas Kourtellis, Rong Cao, Joseph Post, Brian Porter, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26004)  

**Abstract**: Rapid and reliable incident detection is critical for reducing crash-related fatalities, injuries, and congestion. However, conventional methods, such as closed-circuit television, dashcam footage, and sensor-based detection, separate detection from verification, suffer from limited flexibility, and require dense infrastructure or high penetration rates, restricting adaptability and scalability to shifting incident hotspots. To overcome these challenges, we developed DARTS, a drone-based, AI-powered real-time traffic incident detection system. DARTS integrates drones' high mobility and aerial perspective for adaptive surveillance, thermal imaging for better low-visibility performance and privacy protection, and a lightweight deep learning framework for real-time vehicle trajectory extraction and incident detection. The system achieved 99% detection accuracy on a self-collected dataset and supports simultaneous online visual verification, severity assessment, and incident-induced congestion propagation monitoring via a web-based interface. In a field test on Interstate 75 in Florida, DARTS detected and verified a rear-end collision 12 minutes earlier than the local transportation management center and monitored incident-induced congestion propagation, suggesting potential to support faster emergency response and enable proactive traffic control to reduce congestion and secondary crash risk. Crucially, DARTS's flexible deployment architecture reduces dependence on frequent physical patrols, indicating potential scalability and cost-effectiveness for use in remote areas and resource-constrained settings. This study presents a promising step toward a more flexible and integrated real-time traffic incident detection system, with significant implications for the operational efficiency and responsiveness of modern transportation management. 

**Abstract (ZH)**: 基于无人机的AI驱动实时交通事件检测系统DARTS 

---
# Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning 

**Title (ZH)**: supervised reinforcement learning：从专家轨迹到逐步推理 

**Authors**: Yihe Deng, I-Hung Hsu, Jun Yan, Zifeng Wang, Rujun Han, Gufeng Zhang, Yanfei Chen, Wei Wang, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.25992)  

**Abstract**: Large Language Models (LLMs) often struggle with problems that require multi-step reasoning. For small-scale open-source models, Reinforcement Learning with Verifiable Rewards (RLVR) fails when correct solutions are rarely sampled even after many attempts, while Supervised Fine-Tuning (SFT) tends to overfit long demonstrations through rigid token-by-token imitation. To address this gap, we propose Supervised Reinforcement Learning (SRL), a framework that reformulates problem solving as generating a sequence of logical "actions". SRL trains the model to generate an internal reasoning monologue before committing to each action. It provides smoother rewards based on the similarity between the model's actions and expert actions extracted from the SFT dataset in a step-wise manner. This supervision offers richer learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. As a result, SRL enables small models to learn challenging problems previously unlearnable by SFT or RLVR. Moreover, initializing training with SRL before refining with RLVR yields the strongest overall performance. Beyond reasoning benchmarks, SRL generalizes effectively to agentic software engineering tasks, establishing it as a robust and versatile training framework for reasoning-oriented LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）往往难以解决需要多步推理的问题。对于小型开源模型，可验证奖励强化学习（RLVR）在即使经过多次尝试后也难以采样到正确解，而监督微调（SFT）容易通过严格的逐 token 仿本来过度拟合长示例。为了解决这一差距，我们提出了监督强化学习（SRL）框架，将问题解决重新定义为生成一系列逻辑“动作”序列。SRL 训练模型在执行每个动作之前生成内部推理独白。奖励根据模型的动作与从 SFT 数据集中提取的专家动作的相似度逐步提供。这种监督即使在所有展开结果均不正确的情况下也能提供更丰富的学习信号，并鼓励受专家示范指导的灵活推理。因此，SRL 使小模型能够学习以前由 SFT 或 RLVR 无法解决的具有挑战性的问题。此外，使用 SRL 初始化训练再用 RLVR 进一步优化可获得最佳整体性能。除了推理基准测试之外，SRL 在代理软件工程任务中表现出有效的泛化能力，确立了其作为面向推理的 LLM 的强大且通用训练框架的地位。 

---
# Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer 

**Title (ZH)**: Brain-IT：通过脑交互变换器从fMRI进行图像重建 

**Authors**: Roman Beliy, Amit Zalcher, Jonathan Kogman, Navve Wasserman, Michal Irani  

**Link**: [PDF](https://arxiv.org/pdf/2510.25976)  

**Abstract**: Reconstructing images seen by people from their fMRI brain recordings provides a non-invasive window into the human brain. Despite recent progress enabled by diffusion models, current methods often lack faithfulness to the actual seen images. We present "Brain-IT", a brain-inspired approach that addresses this challenge through a Brain Interaction Transformer (BIT), allowing effective interactions between clusters of functionally-similar brain-voxels. These functional-clusters are shared by all subjects, serving as building blocks for integrating information both within and across brains. All model components are shared by all clusters & subjects, allowing efficient training with a limited amount of data. To guide the image reconstruction, BIT predicts two complementary localized patch-level image features: (i)high-level semantic features which steer the diffusion model toward the correct semantic content of the image; and (ii)low-level structural features which help to initialize the diffusion process with the correct coarse layout of the image. BIT's design enables direct flow of information from brain-voxel clusters to localized image features. Through these principles, our method achieves image reconstructions from fMRI that faithfully reconstruct the seen images, and surpass current SotA approaches both visually and by standard objective metrics. Moreover, with only 1-hour of fMRI data from a new subject, we achieve results comparable to current methods trained on full 40-hour recordings. 

**Abstract (ZH)**: 从fMRI脑成像重建人的视觉图像：一种脑启发的方法 

---
# WaveVerif: Acoustic Side-Channel based Verification of Robotic Workflows 

**Title (ZH)**: WaveVerif：基于声学侧信道的机器人工作流程验证 

**Authors**: Zeynep Yasemin Erdogan, Shishir Nagaraja, Chuadhry Mujeeb Ahmed, Ryan Shah  

**Link**: [PDF](https://arxiv.org/pdf/2510.25960)  

**Abstract**: In this paper, we present a framework that uses acoustic side- channel analysis (ASCA) to monitor and verify whether a robot correctly executes its intended commands. We develop and evaluate a machine-learning-based workflow verification system that uses acoustic emissions generated by robotic movements. The system can determine whether real-time behavior is consistent with expected commands. The evaluation takes into account movement speed, direction, and microphone distance. The results show that individual robot movements can be validated with over 80% accuracy under baseline conditions using four different classifiers: Support Vector Machine (SVM), Deep Neural Network (DNN), Recurrent Neural Network (RNN), and Convolutional Neural Network (CNN). Additionally, workflows such as pick-and-place and packing could be identified with similarly high confidence. Our findings demonstrate that acoustic signals can support real-time, low-cost, passive verification in sensitive robotic environments without requiring hardware modifications. 

**Abstract (ZH)**: 本文提出了一种利用声学侧通道分析（ASCA）框架，监测和验证机器人是否正确执行其预期命令的方法。该文开发并评估了一种基于机器学习的工作流验证系统，该系统利用机器人运动产生的声学发射信号。该系统能够确定实时行为是否与预期命令一致。评估考虑了运动速度、方向和麦克风距离等因素。结果表明，在基线条件下，使用四种不同的分类器（支持向量机（SVM）、深度神经网络（DNN）、递归神经网络（RNN）和卷积神经网络（CNN）），单一机器人动作的验证准确率超过80%。此外，诸如拿起放下和打包等工作流程也能够以高置信度被识别。研究发现表明，声学信号可以在不需硬件修改的情况下，支持敏感机器人环境中实时、低成本的被动验证。 

---
# Application and Validation of Geospatial Foundation Model Data for the Prediction of Health Facility Programmatic Outputs -- A Case Study in Malawi 

**Title (ZH)**: 地理空间基础模型数据在预测卫生设施项目产出的应用与验证——以马拉维为例 

**Authors**: Lynn Metz, Rachel Haggard, Michael Moszczynski, Samer Asbah, Chris Mwase, Patricia Khomani, Tyler Smith, Hannah Cooper, Annie Mwale, Arbaaz Muslim, Gautam Prasad, Mimi Sun, Tomer Shekel, Joydeep Paul, Anna Carter, Shravya Shetty, Dylan Green  

**Link**: [PDF](https://arxiv.org/pdf/2510.25954)  

**Abstract**: The reliability of routine health data in low and middle-income countries (LMICs) is often constrained by reporting delays and incomplete coverage, necessitating the exploration of novel data sources and analytics. Geospatial Foundation Models (GeoFMs) offer a promising avenue by synthesizing diverse spatial, temporal, and behavioral data into mathematical embeddings that can be efficiently used for downstream prediction tasks. This study evaluated the predictive performance of three GeoFM embedding sources - Google Population Dynamics Foundation Model (PDFM), Google AlphaEarth (derived from satellite imagery), and mobile phone call detail records (CDR) - for modeling 15 routine health programmatic outputs in Malawi, and compared their utility to traditional geospatial interpolation methods. We used XGBoost models on data from 552 health catchment areas (January 2021-May 2023), assessing performance with R2, and using an 80/20 training and test data split with 5-fold cross-validation used in training. While predictive performance was mixed, the embedding-based approaches improved upon baseline geostatistical methods in 13 of 15 (87%) indicators tested. A Multi-GeoFM model integrating all three embedding sources produced the most robust predictions, achieving average 5-fold cross validated R2 values for indicators like population density (0.63), new HIV cases (0.57), and child vaccinations (0.47) and test set R2 of 0.64, 0.68, and 0.55, respectively. Prediction was poor for prediction targets with low primary data availability, such as TB and malnutrition cases. These results demonstrate that GeoFM embeddings imbue a modest predictive improvement for select health and demographic outcomes in an LMIC context. We conclude that the integration of multiple GeoFM sources is an efficient and valuable tool for supplementing and strengthening constrained routine health information systems. 

**Abstract (ZH)**: 低收入和中等收入国家常规健康数据的可靠性经常受限于报告延迟和不完整覆盖，急需探索新型数据源和分析方法。空间基础模型（GeoFMs）通过合成多元空间、时间和行为数据为下游预测任务提供了有希望的途径。本研究评估了三种GeoFM嵌入源——谷歌人口动态基础模型（PDFM）、Google AlphaEarth（基于卫星影像）和移动电话通信详细记录（CDR）——在马拉维建模15项常规健康项目输出的预测性能，并将其与传统地理插值方法的实用性进行了比较。我们使用552个卫生保健区的数据（2021年1月-2023年5月），以R2作为性能评估指标，并采用80/20的训练和测试数据分割以及五折交叉验证进行训练。虽然预测性能参差不齐，但在15个测试指标中的13个（87%）中，基于嵌入的方法优于基线地理统计方法。三重GeoFM模型整合了三种嵌入源，产生了最稳健的预测结果，平均五折交叉验证R2值分别为人口密度（0.63）、新发HIV病例（0.57）和儿童疫苗接种（0.47），以及测试集的R2分别为0.64、0.68和0.55。对于主要数据获取难度大的结核病和营养不良病例的预测效果较差。这些结果表明，在低收入和中等收入国家的背景下，GeoFM嵌入为某些健康和人口统计结果提供了适度的预测改进。我们得出结论，多源GeoFM的整合是一种高效的工具，可补充并加强受限的常规健康信息系统。 

---
# Revisiting Multilingual Data Mixtures in Language Model Pretraining 

**Title (ZH)**: 重访语言模型预训练中的多语言数据混合 

**Authors**: Negar Foroutan, Paul Teiletche, Ayush Kumar Tarun, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2510.25947)  

**Abstract**: The impact of different multilingual data mixtures in pretraining large language models (LLMs) has been a topic of ongoing debate, often raising concerns about potential trade-offs between language coverage and model performance (i.e., the curse of multilinguality). In this work, we investigate these assumptions by training 1.1B and 3B parameter LLMs on diverse multilingual corpora, varying the number of languages from 25 to 400. Our study challenges common beliefs surrounding multilingual training. First, we find that combining English and multilingual data does not necessarily degrade the in-language performance of either group, provided that languages have a sufficient number of tokens included in the pretraining corpus. Second, we observe that using English as a pivot language (i.e., a high-resource language that serves as a catalyst for multilingual generalization) yields benefits across language families, and contrary to expectations, selecting a pivot language from within a specific family does not consistently improve performance for languages within that family. Lastly, we do not observe a significant "curse of multilinguality" as the number of training languages increases in models at this scale. Our findings suggest that multilingual data, when balanced appropriately, can enhance language model capabilities without compromising performance, even in low-resource settings 

**Abstract (ZH)**: 不同多语数据混合对大规模语言模型预训练的影响：挑战多语训练的常见 belief，并发现多语数据在适当平衡时可以增强语言模型能力，而不会牺牲性能，即使在低资源环境下也是如此。 

---
# A Process Mining-Based System For The Analysis and Prediction of Software Development Workflows 

**Title (ZH)**: 基于过程挖掘的软件开发工作流分析与预测系统 

**Authors**: Antía Dorado, Iván Folgueira, Sofía Martín, Gonzalo Martín, Álvaro Porto, Alejandro Ramos, John Wallace  

**Link**: [PDF](https://arxiv.org/pdf/2510.25935)  

**Abstract**: CodeSight is an end-to-end system designed to anticipate deadline compliance in software development workflows. It captures development and deployment data directly from GitHub, transforming it into process mining logs for detailed analysis. From these logs, the system generates metrics and dashboards that provide actionable insights into PR activity patterns and workflow efficiency. Building on this structured representation, CodeSight employs an LSTM model that predicts remaining PR resolution times based on sequential activity traces and static features, enabling early identification of potential deadline breaches. In tests, the system demonstrates high precision and F1 scores in predicting deadline compliance, illustrating the value of integrating process mining with machine learning for proactive software project management. 

**Abstract (ZH)**: CodeSight：一种用于软件开发工作流中截止日期合规性预测的端到端系统 

---
# Multi-Agent Reinforcement Learning for Market Making: Competition without Collusion 

**Title (ZH)**: 多智能体强化学习在市场制作中的应用：无串通的竞争 

**Authors**: Ziyi Wang, Carmine Ventre, Maria Polukarov  

**Link**: [PDF](https://arxiv.org/pdf/2510.25929)  

**Abstract**: Algorithmic collusion has emerged as a central question in AI: Will the interaction between different AI agents deployed in markets lead to collusion? More generally, understanding how emergent behavior, be it a cartel or market dominance from more advanced bots, affects the market overall is an important research question.
We propose a hierarchical multi-agent reinforcement learning framework to study algorithmic collusion in market making. The framework includes a self-interested market maker (Agent~A), which is trained in an uncertain environment shaped by an adversary, and three bottom-layer competitors: the self-interested Agent~B1 (whose objective is to maximize its own PnL), the competitive Agent~B2 (whose objective is to minimize the PnL of its opponent), and the hybrid Agent~B$^\star$, which can modulate between the behavior of the other two. To analyze how these agents shape the behavior of each other and affect market outcomes, we propose interaction-level metrics that quantify behavioral asymmetry and system-level dynamics, while providing signals potentially indicative of emergent interaction patterns.
Experimental results show that Agent~B2 secures dominant performance in a zero-sum setting against B1, aggressively capturing order flow while tightening average spreads, thus improving market execution efficiency. In contrast, Agent~B$^\star$ exhibits a self-interested inclination when co-existing with other profit-seeking agents, securing dominant market share through adaptive quoting, yet exerting a milder adverse impact on the rewards of Agents~A and B1 compared to B2. These findings suggest that adaptive incentive control supports more sustainable strategic co-existence in heterogeneous agent environments and offers a structured lens for evaluating behavioral design in algorithmic trading systems. 

**Abstract (ZH)**: 算法共谋已成为AI领域的核心问题：不同的AI代理在市场中交互是否会引发共谋？更广泛地说，理解诸如卡特尔或由更先进机器人主导市场等 emergent 行为如何影响整体市场是一个重要的研究问题。 

---
# Transferring Causal Effects using Proxies 

**Title (ZH)**: 使用代理变量转移因果效果 

**Authors**: Manuel Iglesias-Alonso, Felix Schur, Julius von Kügelgen, Jonas Peters  

**Link**: [PDF](https://arxiv.org/pdf/2510.25924)  

**Abstract**: We consider the problem of estimating a causal effect in a multi-domain setting. The causal effect of interest is confounded by an unobserved confounder and can change between the different domains. We assume that we have access to a proxy of the hidden confounder and that all variables are discrete or categorical. We propose methodology to estimate the causal effect in the target domain, where we assume to observe only the proxy variable. Under these conditions, we prove identifiability (even when treatment and response variables are continuous). We introduce two estimation techniques, prove consistency, and derive confidence intervals. The theoretical results are supported by simulation studies and a real-world example studying the causal effect of website rankings on consumer choices. 

**Abstract (ZH)**: 多域环境中未观察到共因干扰下的因果效应估计方法 

---
# Evaluating the Impact of LLM-Assisted Annotation in a Perspectivized Setting: the Case of FrameNet Annotation 

**Title (ZH)**: 在视角化设置中评估LLM辅助注释的影响：FrameNet注释案例研究 

**Authors**: Frederico Belcavello, Ely Matos, Arthur Lorenzi, Lisandra Bonoto, Lívia Ruiz, Luiz Fernando Pereira, Victor Herbst, Yulla Navarro, Helen de Andrade Abreu, Lívia Dutra, Tiago Timponi Torrent  

**Link**: [PDF](https://arxiv.org/pdf/2510.25904)  

**Abstract**: The use of LLM-based applications as a means to accelerate and/or substitute human labor in the creation of language resources and dataset is a reality. Nonetheless, despite the potential of such tools for linguistic research, comprehensive evaluation of their performance and impact on the creation of annotated datasets, especially under a perspectivized approach to NLP, is still missing. This paper contributes to reduction of this gap by reporting on an extensive evaluation of the (semi-)automatization of FrameNet-like semantic annotation by the use of an LLM-based semantic role labeler. The methodology employed compares annotation time, coverage and diversity in three experimental settings: manual, automatic and semi-automatic annotation. Results show that the hybrid, semi-automatic annotation setting leads to increased frame diversity and similar annotation coverage, when compared to the human-only setting, while the automatic setting performs considerably worse in all metrics, except for annotation time. 

**Abstract (ZH)**: 基于LLM的应用在语言资源和数据集创建中的使用是现实存在的。尽管此类工具在语言研究中具有潜在价值，但对其性能和对注解数据集创建影响的全面评估，尤其是从视角化的NLP视角来看，仍缺乏。本文通过报告一种基于LLM的语义角色标注器对类似FrameNet的语义标注进行半自动化处理的广泛评估， contributes to缩小这一差距。研究方法在三种实验设置下比较了注释时间、覆盖面和多样性：手工注释、自动注释和半自动注释。结果表明，混合的半自动注释设置在框架多样性方面优于仅有人工注释的设置，且在所有指标上具有相似的覆盖范围，而全自动设置在所有指标上表现较差，仅在注释时间上有优势。 

---
# PRISM: Proof-Carrying Artifact Generation through LLM x MDE Synergy and Stratified Constraints 

**Title (ZH)**: PRISM: 通过LLM与MDE协同及分层约束生成证明承载的制品 

**Authors**: Tong Ma, Hui Lai, Hui Wang, Zhenhu Tian, Jizhou Wang, Haichao Wu, Yongfan Gao, Chaochao Li, Fengjie Xu, Ling Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.25890)  

**Abstract**: PRISM unifies Large Language Models with Model-Driven Engineering to generate regulator-ready artifacts and machine-checkable evidence for safety- and compliance-critical domains. PRISM integrates three pillars: a Unified Meta-Model (UMM) reconciles heterogeneous schemas and regulatory text into a single semantic space; an Integrated Constraint Model (ICM) compiles structural and semantic requirements into enforcement artifacts including generation-time automata (GBNF, DFA) and post-generation validators (e.g., SHACL, SMT); and Constraint-Guided Verifiable Generation (CVG) applies these through two-layer enforcement - structural constraints drive prefix-safe decoding while semantic/logical validation produces machine-checkable certificates. When violations occur, PRISM performs audit-guided repair and records generation traces for compliance review. We evaluate PRISM in automotive software engineering (AUTOSAR) and cross-border legal jurisdiction (Brussels I bis). PRISM produces structurally valid, auditable artifacts that integrate with existing tooling and substantially reduce manual remediation effort, providing a practical path toward automated artifact generation with built-in assurance. 

**Abstract (ZH)**: PRISM将大型语言模型与模型驱动工程相结合，生成监管-ready的 artifacts 和可机器验证的证据，应用于安全和合规关键领域。 

---
# AAGATE: A NIST AI RMF-Aligned Governance Platform for Agentic AI 

**Title (ZH)**: AAGATE: 一个符合NIST AI RMF规范的自主人工智能治理平台 

**Authors**: Ken Huang, Jerry Huang, Yasir Mehmood, Hammad Atta, Muhammad Zeeshan Baig, Muhammad Aziz Ul Haq  

**Link**: [PDF](https://arxiv.org/pdf/2510.25863)  

**Abstract**: This paper introduces the Agentic AI Governance Assurance & Trust Engine (AAGATE), a Kubernetes-native control plane designed to address the unique security and governance challenges posed by autonomous, language-model-driven agents in production. Recognizing the limitations of traditional Application Security (AppSec) tooling for improvisational, machine-speed systems, AAGATE operationalizes the NIST AI Risk Management Framework (AI RMF). It integrates specialized security frameworks for each RMF function: the Agentic AI Threat Modeling MAESTRO framework for Map, a hybrid of OWASP's AIVSS and SEI's SSVC for Measure, and the Cloud Security Alliance's Agentic AI Red Teaming Guide for Manage. By incorporating a zero-trust service mesh, an explainable policy engine, behavioral analytics, and decentralized accountability hooks, AAGATE provides a continuous, verifiable governance solution for agentic AI, enabling safe, accountable, and scalable deployment. The framework is further extended with DIRF for digital identity rights, LPCI defenses for logic-layer injection, and QSAF monitors for cognitive degradation, ensuring governance spans systemic, adversarial, and ethical risks. 

**Abstract (ZH)**: 基于Kubernetes的代理AI治理保障与信任引擎（AAGATE）：一种应对生产环境中自主语言模型驱动代理独特安全与治理挑战的解决方案 

---
# Identity Management for Agentic AI: The new frontier of authorization, authentication, and security for an AI agent world 

**Title (ZH)**: 代理人工智能的身份管理：AI代理世界中的授权、认证与安全新前沿 

**Authors**: Tobin South, Subramanya Nagabhushanaradhya, Ayesha Dissanayaka, Sarah Cecchetti, George Fletcher, Victor Lu, Aldo Pietropaolo, Dean H. Saxe, Jeff Lombardo, Abhishek Maligehalli Shivalingaiah, Stan Bounev, Alex Keisner, Andor Kesselman, Zack Proser, Ginny Fahs, Andrew Bunyea, Ben Moskowitz, Atul Tulshibagwale, Dazza Greenwood, Jiaxin Pei, Alex Pentland  

**Link**: [PDF](https://arxiv.org/pdf/2510.25819)  

**Abstract**: The rapid rise of AI agents presents urgent challenges in authentication, authorization, and identity management. Current agent-centric protocols (like MCP) highlight the demand for clarified best practices in authentication and authorization. Looking ahead, ambitions for highly autonomous agents raise complex long-term questions regarding scalable access control, agent-centric identities, AI workload differentiation, and delegated authority. This OpenID Foundation whitepaper is for stakeholders at the intersection of AI agents and access management. It outlines the resources already available for securing today's agents and presents a strategic agenda to address the foundational authentication, authorization, and identity problems pivotal for tomorrow's widespread autonomous systems. 

**Abstract (ZH)**: AI代理的迅速崛起对身份验证、授权和身份管理提出了迫切挑战。当前以代理为中心的协议（如MCP）突显了明确最佳实践的需求。展望未来，对高度自主代理的期望引发了关于可扩展访问控制、以代理为中心的身份、AI工作负载差异化和授权委托的复杂长期问题。本OpenID基金会白皮书面向AI代理和访问管理交叉领域的利益相关者，概述了当前可用的安全资源，并提出了一个战略议程，解决对未来广泛自主系统至关重要的身份验证、授权和身份基础问题。 

---
# ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion 

**Title (ZH)**: ScaleDiff: 通过高效且模型无关的扩散实现高分辨率图像合成 

**Authors**: Sungho Koh, SeungJu Cha, Hyunwoo Oh, Kwanyoung Lee, Dong-Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.25818)  

**Abstract**: Text-to-image diffusion models often exhibit degraded performance when generating images beyond their training resolution. Recent training-free methods can mitigate this limitation, but they often require substantial computation or are incompatible with recent Diffusion Transformer models. In this paper, we propose ScaleDiff, a model-agnostic and highly efficient framework for extending the resolution of pretrained diffusion models without any additional training. A core component of our framework is Neighborhood Patch Attention (NPA), an efficient mechanism that reduces computational redundancy in the self-attention layer with non-overlapping patches. We integrate NPA into an SDEdit pipeline and introduce Latent Frequency Mixing (LFM) to better generate fine details. Furthermore, we apply Structure Guidance to enhance global structure during the denoising process. Experimental results demonstrate that ScaleDiff achieves state-of-the-art performance among training-free methods in terms of both image quality and inference speed on both U-Net and Diffusion Transformer architectures. 

**Abstract (ZH)**: 无训练扩展：一种适用于预训练扩散模型的模型agnostic且高效的框架 

---
# Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start 

**Title (ZH)**: Metis-SPECS: 通过自distilled偏好引导的冷启动解耦多模态学习 

**Authors**: Kun Chen, Peng Shi, Haibo Qiu, Zhixiong Zeng, Siqi Yang, Wenji Mao, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.25801)  

**Abstract**: Reinforcement learning (RL) with verifiable rewards has recently catalyzed a wave of "MLLM-r1" approaches that bring RL to vision language models. Most representative paradigms begin with a cold start, typically employing supervised fine-tuning (SFT), to initialize the policy before RL. However, SFT-based cold start adopts the reasoning paradigm intertwined with task solution and output format, which may induce instruction-style overfitting, weakens out-of-distribution generalization, and ultimately affects downstream RL. We revisit the cold start along two views, its training method and data construction, and introduce the Generalization Factor (GF) coefficient to quantify the generalization capability under different methods. Our empirical study finds that preference-based training methods (e.g. DPO) generalizes better than SFT-based methods in cold start. Motivated by this, we propose SPECS-a Self-distilled, Preference-based Cold Start framework that decouples multimodal learning: (1) generates introspective preference data pairs via self-distillation, avoiding reliance on larger teachers or manual annotation; (2) performs preference-based training to learn, focusing on shallow, transferable surface-form criteria (format, structure, style) rather than memorizing content; and (3) hands off to RL with verifiable rewards for deep reasoning results. Experimental results across multiple multimodal benchmarks show that our decoupling learning framework yields consistent performance gains over strong baselines, improving MEGA-Bench by 4.1% and MathVista by 12.2%. Additional experiments indicate that SPECS contributes to reducing in-distribution "stuckness," improving exploration, stabilizing training, and raising the performance ceiling. 

**Abstract (ZH)**: 可验证奖励的强化学习与视觉语言模型的“MLLM-r1”方法recently催化了一波新的 approaches，使RL应用到视觉语言模型中。大多数最具代表性的范式从冷启动开始，通常使用监督微调（SFT）来初始化策略，然后进行RL。然而，基于SFT的冷启动结合了解题推理与输出格式，可能导致指令式过拟合，减弱离分布的泛化能力，最终影响下游的RL任务。我们从训练方法和数据构造两个视角重新审视冷启动，并引入了泛化因子（GF）系数来量化不同方法下的泛化能力。我们的实证研究表明，基于偏好训练的方法（例如DPO）在冷启动时的泛化能力优于基于SFT的方法。受此启发，我们提出了一种自我蒸馏、基于偏好的冷启动框架SPECS，以解耦多模态学习：(1) 通过自我蒸馏生成反思偏好数据对，避免依赖于更大的教师模型或手动标注；(2) 进行基于偏好的训练，专注于浅层、可迁移的表象形式标准（格式、结构、风格），而不是记忆内容；(3) 将可验证奖励的RL与之结合，用于深入推理结果。跨多个多模态基准的实验结果表明，我们的解耦学习框架在强基线之上表现出一致的性能提升，分别在MEGA-Bench和MathVista上提高了4.1%和12.2%。额外的实验表明，SPECS有助于减少分布内“停滞”，提高探索性，稳定训练，并提高性能上限。 

---
# MemEIC: A Step Toward Continual and Compositional Knowledge Editing 

**Title (ZH)**: MemEIC：走向持续且组合式知识编辑 

**Authors**: Jin Seong, Jiyun Park, Wencke Liermann, Hongseok Choi, Yoonji Nam, Hyun Kim, Soojong Lim, Namhoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.25798)  

**Abstract**: The dynamic nature of information necessitates continuously updating large vision-language models (LVLMs). While recent knowledge editing techniques hint at promising directions, they often focus on editing a single modality (vision or language) in isolation. This prevalent practice neglects the inherent multimodality of LVLMs and the continuous nature of knowledge updates, potentially leading to suboptimal editing outcomes when considering the interplay between modalities and the need for ongoing knowledge refinement. To address these limitations, we propose MemEIC, a novel method for Continual and Compositional Knowledge Editing (CCKE) in LVLMs. MemEIC enables compositional editing of both visual and textual knowledge sequentially. Our approach employs a hybrid external-internal editor featuring a dual external memory for cross-modal evidence retrieval and dual LoRA adapters that facilitate disentangled parameter updates for each modality. A key component is a brain-inspired knowledge connector, activated selectively for compositional reasoning, that integrates information across different modalities. Experiments demonstrate that MemEIC significantly improves performance on complex multimodal questions and effectively preserves prior edits, setting a new benchmark for CCKE in LVLMs. 

**Abstract (ZH)**: MemEIC：视觉语言大模型中连续与组成性知识编辑的新方法 

---
# Non-myopic Matching and Rebalancing in Large-Scale On-Demand Ride-Pooling Systems Using Simulation-Informed Reinforcement Learning 

**Title (ZH)**: 使用基于仿真增强学习的大规模按需拼车系统非短视配对与再平衡 

**Authors**: Farnoosh Namdarpour, Joseph Y. J. Chow  

**Link**: [PDF](https://arxiv.org/pdf/2510.25796)  

**Abstract**: Ride-pooling, also known as ride-sharing, shared ride-hailing, or microtransit, is a service wherein passengers share rides. This service can reduce costs for both passengers and operators and reduce congestion and environmental impacts. A key limitation, however, is its myopic decision-making, which overlooks long-term effects of dispatch decisions. To address this, we propose a simulation-informed reinforcement learning (RL) approach. While RL has been widely studied in the context of ride-hailing systems, its application in ride-pooling systems has been less explored. In this study, we extend the learning and planning framework of Xu et al. (2018) from ride-hailing to ride-pooling by embedding a ride-pooling simulation within the learning mechanism to enable non-myopic decision-making. In addition, we propose a complementary policy for rebalancing idle vehicles. By employing n-step temporal difference learning on simulated experiences, we derive spatiotemporal state values and subsequently evaluate the effectiveness of the non-myopic policy using NYC taxi request data. Results demonstrate that the non-myopic policy for matching can increase the service rate by up to 8.4% versus a myopic policy while reducing both in-vehicle and wait times for passengers. Furthermore, the proposed non-myopic policy can decrease fleet size by over 25% compared to a myopic policy, while maintaining the same level of performance, thereby offering significant cost savings for operators. Incorporating rebalancing operations into the proposed framework cuts wait time by up to 27.3%, in-vehicle time by 12.5%, and raises service rate by 15.1% compared to using the framework for matching decisions alone at the cost of increased vehicle minutes traveled per passenger. 

**Abstract (ZH)**: 拼车，也称为共乘、共享出行或微交通，是一种乘客共乘的服务。这种服务可以减少乘客和运营者的成本，减轻交通拥堵和环境影响。然而，其关键限制在于短期内的调度决策忽视了长期效果。为解决这一问题，我们提出了一种基于仿真的强化学习方法。虽然强化学习在出行叫车系统中得到了广泛研究，但在拼车系统中的应用相对较少。在本研究中，我们将 Xu 等人（2018）为出行叫车系统开发的学习和规划框架扩展到拼车系统，通过在学习机制中嵌入拼车仿真以实现非短视决策。此外，我们还提出了一种为闲置车辆再平衡的补充策略。通过在模拟经验上应用 n 步时差学习，我们推导出时空状态值，并使用纽约市出租车请求数据评估非短视策略的有效性。结果显示，与短视策略相比，非短视匹配策略可以将服务率提高多达 8.4%，同时减少乘客的乘车时间和等待时间。此外，与短视策略相比，提出的非短视策略可以将车队规模减少超过 25%，同时保持相同的性能水平，从而为运营者提供显著的成本节省。将再平衡操作纳入所提出的框架可以将等待时间减少多达 27.3%，乘车时间减少 12.5%，服务率提高 15.1%，但成本是每乘客增加的行驶时间。 

---
# The Kinetics of Reasoning: How Chain-of-Thought Shapes Learning in Transformers? 

**Title (ZH)**: 推理的动力学：链式思考如何塑造 Transformers 中的learning？ 

**Authors**: Zihan Pengmei, Costas Mavromatis, Zhengyuan Shen, Yunyi Zhang, Vassilis N. Ioannidis, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2510.25791)  

**Abstract**: Chain-of-thought (CoT) supervision can substantially improve transformer performance, yet the mechanisms by which models learn to follow and benefit from CoT remain poorly understood. We investigate these learning dynamics through the lens of grokking by pretraining transformers on symbolic reasoning tasks with tunable algorithmic complexity and controllable data composition to study their generalization. Models were trained under two settings: (i) producing only final answers, and (ii) emitting explicit CoT traces before answering. Our results show that while CoT generally improves task performance, its benefits depend on task complexity. To quantify these effects, we model the accuracy of the logarithmic training steps with a three-parameter logistic curve, revealing how the learning speed and shape vary with task complexity, data distribution, and the presence of CoT supervision. We also uncover a transient trace unfaithfulness phase: early in training, models often produce correct answers while skipping or contradicting CoT steps, before later aligning their reasoning traces with answers. Empirically, we (1) demonstrate that CoT accelerates generalization but does not overcome tasks with higher algorithmic complexity, such as finding list intersections; (2) introduce a kinetic modeling framework for understanding transformer learning; (3) characterize trace faithfulness as a dynamic property that emerges over training; and (4) show CoT alters internal transformer computation mechanistically. 

**Abstract (ZH)**: Chain-of-Thought 监督可以显著提高变压器的表现，但模型学习遵循和受益于 Chain-of-Thought 的机制尚不完全理解。我们通过预训练变压器完成可调算法复杂性和可控数据组合的符号推理任务，从grokking的角度探究其泛化学习动力学。模型在两种设置下进行训练：（i）仅仅生成最终答案；（ii）在回答前发出明确的 Chain-of-Thought 跟踪。结果显示，尽管 Chain-of-Thought 通常提高任务性能，其益处取决于任务复杂度。为了量化这些效果，我们用三参数逻辑曲线建模对数训练步长的准确度，揭示学习速度和形状如何随任务复杂度、数据分布以及是否存在 Chain-of-Thought 监督而变化。我们还发现一个暂态跟踪不忠实相：在训练初期，模型通常生成正确答案的同时跳过或违背 Chain-of-Thought 步骤，之后再与答案对齐其推理跟踪。实验上，我们（1）证明 Chain-of-Thought 加速了泛化，但不能克服更高算法复杂度的任务，例如查找列表交集；（2）引入了一个动力学建模框架来理解变压器学习；（3）将跟踪忠实性表征为一种动态属性，随训练而浮现；（4）展示 Chain-of-Thought 从机制上改变了变压器的内部计算。 

---
# Unsupervised local learning based on voltage-dependent synaptic plasticity for resistive and ferroelectric synapses 

**Title (ZH)**: 基于电压依赖突触可塑性的无监督局部学习方法及其在阻变和铁电突触中的应用 

**Authors**: Nikhil Garg, Ismael Balafrej, Joao Henrique Quintino Palhares, Laura Bégon-Lours, Davide Florini, Donato Francesco Falcone, Tommaso Stecconi, Valeria Bragaglia, Bert Jan Offrein, Jean-Michel Portal, Damien Querlioz, Yann Beilliard, Dominique Drouin, Fabien Alibart  

**Link**: [PDF](https://arxiv.org/pdf/2510.25787)  

**Abstract**: The deployment of AI on edge computing devices faces significant challenges related to energy consumption and functionality. These devices could greatly benefit from brain-inspired learning mechanisms, allowing for real-time adaptation while using low-power. In-memory computing with nanoscale resistive memories may play a crucial role in enabling the execution of AI workloads on these edge devices. In this study, we introduce voltage-dependent synaptic plasticity (VDSP) as an efficient approach for unsupervised and local learning in memristive synapses based on Hebbian principles. This method enables online learning without requiring complex pulse-shaping circuits typically necessary for spike-timing-dependent plasticity (STDP). We show how VDSP can be advantageously adapted to three types of memristive devices (TiO$_2$, HfO$_2$-based metal-oxide filamentary synapses, and HfZrO$_4$-based ferroelectric tunnel junctions (FTJ)) with disctinctive switching characteristics. System-level simulations of spiking neural networks incorporating these devices were conducted to validate unsupervised learning on MNIST-based pattern recognition tasks, achieving state-of-the-art performance. The results demonstrated over 83% accuracy across all devices using 200 neurons. Additionally, we assessed the impact of device variability, such as switching thresholds and ratios between high and low resistance state levels, and proposed mitigation strategies to enhance robustness. 

**Abstract (ZH)**: 基于忆阻器的电压依赖性突触可塑性在边缘计算设备上的无监督和局部学习研究 

---
# BlackboxNLP-2025 MIB Shared Task: Improving Circuit Faithfulness via Better Edge Selection 

**Title (ZH)**: BlackboxNLP-2025 MIB 共享任务：通过更好的边选择提高电路忠实度 

**Authors**: Yaniv Nikankin, Dana Arad, Itay Itzhak, Anja Reusch, Adi Simhi, Gal Kesten-Pomeranz, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.25786)  

**Abstract**: One of the main challenges in mechanistic interpretability is circuit discovery, determining which parts of a model perform a given task. We build on the Mechanistic Interpretability Benchmark (MIB) and propose three key improvements to circuit discovery. First, we use bootstrapping to identify edges with consistent attribution scores. Second, we introduce a simple ratio-based selection strategy to prioritize strong positive-scoring edges, balancing performance and faithfulness. Third, we replace the standard greedy selection with an integer linear programming formulation. Our methods yield more faithful circuits and outperform prior approaches across multiple MIB tasks and models. Our code is available at: this https URL. 

**Abstract (ZH)**: 机制可解释性中的主要挑战之一是电路发现，即确定模型中哪一部分执行特定任务。我们基于机制可解释性基准（MIB）并提出三项关键改进以促进电路发现。首先，我们使用自助法识别具有一致归因分数的边。第二，我们引入了一种简单的比率选择策略，以优先考虑高得分的边，平衡性能与忠实度。第三，我们用整数线性规划形式取代了标准的贪婪选择。我们的方法生成了更忠实的电路，并在多个MIB任务和模型中优于先前的方法。代码可在此处获取：this https URL。 

---
# HiMAE: Hierarchical Masked Autoencoders Discover Resolution-Specific Structure in Wearable Time Series 

**Title (ZH)**: HiMAE：层次遮蔽自编码器发现可穿戴时间序列的分辨率特定结构 

**Authors**: Simon A. Lee, Cyrus Tanade, Hao Zhou, Juhyeon Lee, Megha Thukral, Minji Han, Rachel Choi, Md Sazzad Hissain Khan, Baiying Lu, Migyeong Gwak, Mehrab Bin Morshed, Viswam Nathan, Md Mahbubur Rahman, Li Zhu, Subramaniam Venkatraman, Sharanya Arcot Desai  

**Link**: [PDF](https://arxiv.org/pdf/2510.25785)  

**Abstract**: Wearable sensors provide abundant physiological time series, yet the principles governing their predictive utility remain unclear. We hypothesize that temporal resolution is a fundamental axis of representation learning, with different clinical and behavioral outcomes relying on structure at distinct scales. To test this resolution hypothesis, we introduce HiMAE (Hierarchical Masked Autoencoder), a self supervised framework that combines masked autoencoding with a hierarchical convolutional encoder decoder. HiMAE produces multi resolution embeddings that enable systematic evaluation of which temporal scales carry predictive signal, transforming resolution from a hyperparameter into a probe for interpretability. Across classification, regression, and generative benchmarks, HiMAE consistently outperforms state of the art foundation models that collapse scale, while being orders of magnitude smaller. HiMAE is an efficient representation learner compact enough to run entirely on watch, achieving sub millisecond inference on smartwatch class CPUs for true edge inference. Together, these contributions position HiMAE as both an efficient self supervised learning method and a discovery tool for scale sensitive structure in wearable health. 

**Abstract (ZH)**: 可穿戴传感器提供了丰富的生理时间序列数据，但其预测效用的基本原理仍不清楚。我们假设时间分辨率是表示学习的基本轴线，不同的临床和行为结果依赖于不同尺度的结构。为了检验这一分辨率假设，我们引入了HiMAE（分层掩盖自编码器），这是一种结合了掩蔽自编码和分层卷积编码解码器的自我监督框架。HiMAE生成多分辨率嵌入，使系统性评估哪些时间尺度携带预测信号成为可能，将分辨率从超参数转变为解释性的探针。在分类、回归和生成基准测试中，HiMAE在不牺牲性能的情况下，模型大小比压缩尺度的状态-of-the-art基础模型小了数量级。HiMAE是一种高效的表示学习者，其大小足够小，可以在手表上完全运行，实现了智能手表级别CPU上的亚毫秒级推理，进行真正的边缘推理。这些贡献使HiMAE既是一种高效的自我监督学习方法，也是一种探索可穿戴健康中尺度敏感结构的发现工具。 

---
# zFLoRA: Zero-Latency Fused Low-Rank Adapters 

**Title (ZH)**: 零延迟融合低秩适配器：zFLoRA 

**Authors**: Dhananjaya Gowda, Seoha Song, Harshith Goka, Junhyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.25784)  

**Abstract**: Large language models (LLMs) are increasingly deployed with task-specific adapters catering to multiple downstream applications. In such a scenario, the additional compute associated with these apparently insignificant number of adapter parameters (typically less than 1% of the base model) turns out to be disproportionately significant during inference time (upto 2.5x times that of the base model). In this paper, we propose a new zero-latency fused low-rank adapter (zFLoRA) that introduces zero or negligible latency overhead on top of the base model. Experimental results on LLMs of size 1B, 3B and 7B show that zFLoRA compares favorably against the popular supervised fine-tuning benchmarks including low-rank adapters (LoRA) as well as full fine-tuning (FFT). Experiments are conducted on 18 different tasks across three different categories namely commonsense reasoning, math reasoning and summary-dialogue. Latency measurements made on NPU (Samsung Galaxy S25+) as well as GPU (NVIDIA H100) platforms show that the proposed zFLoRA adapters introduce zero to negligible latency overhead. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地部署了针对多种下游应用的任务特定适配器。在这种情况下，这些看似不重要的适配器参数（通常少于基模型的1%）在推理过程中所产生的附加计算量出人意料地显著（最多达到基模型的2.5倍）。本文提出了一种新型零延迟融合低秩适配器（zFLoRA），其在基模型基础上引入了零或可忽略的延迟开销。实验结果表明，在1B、3B和7B规模的LLM上，zFLoRA相对于包括低秩适配器（LoRA）和全量微调（FFT）在内的流行监督微调基准具有竞争力。实验在三大类别（常识推理、数学推理和摘要对话）的18个不同的任务上进行。在NPU（三星Galaxy S25+）和GPU（NVIDIA H100）平台上进行的延迟测量表明，所提出的zFLoRA适配器引入了零或可忽略的延迟开销。 

---
# LASTIST: LArge-Scale Target-Independent STance dataset 

**Title (ZH)**: LASTIST: 大规模目标无关立场数据集 

**Authors**: DongJae Kim, Yaejin Lee, Minsu Park, Eunil Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.25783)  

**Abstract**: Stance detection has emerged as an area of research in the field of artificial intelligence. However, most research is currently centered on the target-dependent stance detection task, which is based on a person's stance in favor of or against a specific target. Furthermore, most benchmark datasets are based on English, making it difficult to develop models in low-resource languages such as Korean, especially for an emerging field such as stance detection. This study proposes the LArge-Scale Target-Independent STance (LASTIST) dataset to fill this research gap. Collected from the press releases of both parties on Korean political parties, the LASTIST dataset uses 563,299 labeled Korean sentences. We provide a detailed description of how we collected and constructed the dataset and trained state-of-the-art deep learning and stance detection models. Our LASTIST dataset is designed for various tasks in stance detection, including target-independent stance detection and diachronic evolution stance detection. We deploy our dataset on this https URL. 

**Abstract (ZH)**: 针对大规模无目标依赖立场检测的数据集（LASTIST） 

---
# A Practitioner's Guide to Kolmogorov-Arnold Networks 

**Title (ZH)**: Kolmogorov-Arnold网络使用指南 

**Authors**: Amir Noorizadegan, Sifan Wang, Leevan Ling  

**Link**: [PDF](https://arxiv.org/pdf/2510.25781)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently emerged as a promising alternative to traditional Multilayer Perceptrons (MLPs), inspired by the Kolmogorov-Arnold representation theorem. Unlike MLPs, which use fixed activation functions on nodes, KANs employ learnable univariate basis functions on edges, offering enhanced expressivity and interpretability. This review provides a systematic and comprehensive overview of the rapidly expanding KAN landscape, moving beyond simple performance comparisons to offer a structured synthesis of theoretical foundations, architectural variants, and practical implementation strategies. By collecting and categorizing a vast array of open-source implementations, we map the vibrant ecosystem supporting KAN development. We begin by bridging the conceptual gap between KANs and MLPs, establishing their formal equivalence and highlighting the superior parameter efficiency of the KAN formulation. A central theme of our review is the critical role of the basis function; we survey a wide array of choices, including B-splines, Chebyshev and Jacobi polynomials, ReLU compositions, Gaussian RBFs, and Fourier series, and analyze their respective trade-offs in terms of smoothness, locality, and computational cost. We then categorize recent advancements into a clear roadmap, covering techniques for improving accuracy, efficiency, and regularization. Key topics include physics-informed loss design, adaptive sampling, domain decomposition, hybrid architectures, and specialized methods for handling discontinuities. Finally, we provide a practical "Choose-Your-KAN" guide to help practitioners select appropriate architectures, and we conclude by identifying current research gaps. The associated GitHub repository this https URL complements this paper and serves as a structured reference for ongoing KAN research. 

**Abstract (ZH)**: Kolmogorov-Arnold网络（KANs）：一种基于柯尔莫戈罗夫-阿诺尔德表示定理的有前途的多层感知机替代方案 

---
# Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets 

**Title (ZH)**: 磁性市场：研究代理市场的开源环境 

**Authors**: Gagan Bansal, Wenyue Hua, Zezhou Huang, Adam Fourney, Amanda Swearngin, Will Epperson, Tyler Payne, Jake M. Hofman, Brendan Lucier, Chinmay Singh, Markus Mobius, Akshay Nambi, Archana Yadav, Kevin Gao, David M. Rothschild, Aleksandrs Slivkins, Daniel G. Goldstein, Hussein Mozannar, Nicole Immorlica, Maya Murad, Matthew Vogel, Subbarao Kambhampati, Eric Horvitz, Saleema Amershi  

**Link**: [PDF](https://arxiv.org/pdf/2510.25779)  

**Abstract**: As LLM agents advance, they are increasingly mediating economic decisions, ranging from product discovery to transactions, on behalf of users. Such applications promise benefits but also raise many questions about agent accountability and value for users. Addressing these questions requires understanding how agents behave in realistic market conditions. However, previous research has largely evaluated agents in constrained settings, such as single-task marketplaces (e.g., negotiation) or structured two-agent interactions. Real-world markets are fundamentally different: they require agents to handle diverse economic activities and coordinate within large, dynamic ecosystems where multiple agents with opaque behaviors may engage in open-ended dialogues. To bridge this gap, we investigate two-sided agentic marketplaces where Assistant agents represent consumers and Service agents represent competing businesses. To study these interactions safely, we develop Magentic-Marketplace-- a simulated environment where Assistants and Services can operate. This environment enables us to study key market dynamics: the utility agents achieve, behavioral biases, vulnerability to manipulation, and how search mechanisms shape market outcomes. Our experiments show that frontier models can approach optimal welfare-- but only under ideal search conditions. Performance degrades sharply with scale, and all models exhibit severe first-proposal bias, creating 10-30x advantages for response speed over quality. These findings reveal how behaviors emerge across market conditions, informing the design of fair and efficient agentic marketplaces. 

**Abstract (ZH)**: 随着大规模语言模型代理的发展，它们在产品发现到交易等一系列经济决策中代表用户进行调解。这类应用带来了诸多好处，但也引发了关于代理问责制和用户价值等方面的众多问题。解决这些问题需要理解代理在现实市场条件下的行为。然而，以往的研究大多是通过受限环境来评估代理，例如单一任务市场（例如谈判）或结构化的两代理交互。现实世界市场本质上是不同的：它们要求代理处理多样化的经济活动，并在包含多个具有不透明行为的代理的大型动态生态系统中进行协调。为了弥合这一差距，我们研究了代理双边市场，其中助手代理代表消费者，服务代理代表竞争企业。为安全地研究这些交互，我们开发了Magnetic-Marketplace——一个模拟环境，允许助手和服务代理运营。该环境使我们能够研究关键市场动态：代理实现的效用、行为偏差、易受操纵性以及搜索机制如何影响市场结果。我们的实验表明，前沿模型在理想搜索条件下可以接近最优福利——但随着规模的扩大，性能急剧下降，所有模型都表现出严重的初次提案偏差，这使响应速度比质量高出10-30倍。这些发现揭示了在不同市场条件下行为是如何演变的，从而有助于设计公平高效的代理市场。 

---
# DINO-YOLO: Self-Supervised Pre-training for Data-Efficient Object Detection in Civil Engineering Applications 

**Title (ZH)**: DINO-YOLO：基于自我监督预训练的高效数据物体检测在土木工程应用中 

**Authors**: Malaisree P, Youwai S, Kitkobsin T, Janrungautai S, Amorndechaphon D, Rojanavasu P  

**Link**: [PDF](https://arxiv.org/pdf/2510.25140)  

**Abstract**: Object detection in civil engineering applications is constrained by limited annotated data in specialized domains. We introduce DINO-YOLO, a hybrid architecture combining YOLOv12 with DINOv3 self-supervised vision transformers for data-efficient detection. DINOv3 features are strategically integrated at two locations: input preprocessing (P0) and mid-backbone enhancement (P3). Experimental validation demonstrates substantial improvements: Tunnel Segment Crack detection (648 images) achieves 12.4% improvement, Construction PPE (1K images) gains 13.7%, and KITTI (7K images) shows 88.6% improvement, while maintaining real-time inference (30-47 FPS). Systematic ablation across five YOLO scales and nine DINOv3 variants reveals that Medium-scale architectures achieve optimal performance with DualP0P3 integration (55.77% mAP@0.5), while Small-scale requires Triple Integration (53.63%). The 2-4x inference overhead (21-33ms versus 8-16ms baseline) remains acceptable for field deployment on NVIDIA RTX 5090. DINO-YOLO establishes state-of-the-art performance for civil engineering datasets (<10K images) while preserving computational efficiency, providing practical solutions for construction safety monitoring and infrastructure inspection in data-constrained environments. 

**Abstract (ZH)**: 在特殊领域中基于有限标注数据的 civil engineering 应用中的物体检测受到限制。我们引入了 DINO-YOLO，这是一种将 YOLOv12 与 DINOv3 自监督视觉变换器结合的混合架构，用于高效数据检测。DINOv3 特征在输入预处理（P0）和中间骨干增强（P3）两个位置战略性集成。实验验证显示了显著改进：隧道段裂缝检测（648 张图像）提高了 12.4%，建筑工程个人防护装备（1000 张图像）提高了 13.7%，KITTI（7000 张图像）提高了 88.6%，同时保持实时推断（30-47 FPS）。系统性消融研究覆盖五个 YOLO 模型尺度和九种 DINOv3 变体，表明中等规模架构通过 DualP0P3 集成实现最佳性能（55.77% mAP@0.5），而小型架构需要三重集成（53.63%）。2-4 倍的推断开销（21-33ms 对比 8-16ms 基线）在 NVIDIA RTX 5090 的现场部署中是可以接受的。DINO-YOLO 为包含少于 10000 张图像的 civil engineering 数据集设定了最先进的性能，同时保持了计算效率，为数据受限环境中的建筑安全监测和基础设施检查提供了实用解决方案。 

---
