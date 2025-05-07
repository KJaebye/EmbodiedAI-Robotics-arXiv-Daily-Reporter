# Capability-Driven Skill Generation with LLMs: A RAG-Based Approach for Reusing Existing Libraries and Interfaces 

**Title (ZH)**: 基于RAG的方法：面向能力的技能生成——重用现有库和接口的LLM驱动 approach 

**Authors**: Luis Miguel Vieira da Silva, Aljosha Köcher, Nicolas König, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2505.03295)  

**Abstract**: Modern automation systems increasingly rely on modular architectures, with capabilities and skills as one solution approach. Capabilities define the functions of resources in a machine-readable form and skills provide the concrete implementations that realize those capabilities. However, the development of a skill implementation conforming to a corresponding capability remains a time-consuming and challenging task. In this paper, we present a method that treats capabilities as contracts for skill implementations and leverages large language models to generate executable code based on natural language user input. A key feature of our approach is the integration of existing software libraries and interface technologies, enabling the generation of skill implementations across different target languages. We introduce a framework that allows users to incorporate their own libraries and resource interfaces into the code generation process through a retrieval-augmented generation architecture. The proposed method is evaluated using an autonomous mobile robot controlled via Python and ROS 2, demonstrating the feasibility and flexibility of the approach. 

**Abstract (ZH)**: 现代自动化系统越来越多地依赖模块化架构，能力与技能为其解决方案之一。能力以机器可读的形式定义资源的功能，而技能则提供实现这些能力的具体实现。然而，根据相应能力开发符合规范的技能实现仍然是一个耗时且具有挑战性的工作。本文提出了一种方法，将能力视为技能实现的契约，并利用大型语言模型根据自然语言用户输入生成可执行代码。我们方法的关键特征是整合现有的软件库和接口技术，实现不同目标语言中的技能实现生成。我们引入了一个框架，允许用户通过检索增强生成架构将其自己的库和资源接口纳入代码生成过程。所提出的方法通过一个自主移动机器人（通过Python和ROS 2 控制）进行评估，展示了该方法的可行性和灵活性。 

---
# Graph Drawing for LLMs: An Empirical Evaluation 

**Title (ZH)**: LLMs的图形绘制：一项实证评价 

**Authors**: Walter Didimo, Fabrizio Montecchiani, Tommaso Piselli  

**Link**: [PDF](https://arxiv.org/pdf/2505.03678)  

**Abstract**: Our work contributes to the fast-growing literature on the use of Large Language Models (LLMs) to perform graph-related tasks. In particular, we focus on usage scenarios that rely on the visual modality, feeding the model with a drawing of the graph under analysis. We investigate how the model's performance is affected by the chosen layout paradigm, the aesthetics of the drawing, and the prompting technique used for the queries. We formulate three corresponding research questions and present the results of a thorough experimental analysis. Our findings reveal that choosing the right layout paradigm and optimizing the readability of the input drawing from a human perspective can significantly improve the performance of the model on the given task. Moreover, selecting the most effective prompting technique is a challenging yet crucial task for achieving optimal performance. 

**Abstract (ZH)**: 我们的工作为大型语言模型（LLMs）在图相关任务中的应用这一快速发展的研究领域做出了贡献。特别地，我们关注依赖视觉模态的应用场景，通过向模型提供要分析的图的绘制来实现。我们研究了所选布局范式、绘制的美学以及用于查询的提示技术对模型性能的影响。我们提出了三个相应的研究问题，并呈现了详细实验分析的结果。我们的发现表明，从人类视角优化输入绘制的可读性和选择最有效的提示技术可以显著提高模型在给定任务上的性能。此外，选择最有效的提示技术是实现最佳性能的一项具有挑战但至关重要的任务。 

---
# A Hashgraph-Inspired Consensus Mechanism for Reliable Multi-Model Reasoning 

**Title (ZH)**: 基于哈希图启发的可靠多模型推理共识机制 

**Authors**: Kolawole E. Ogunsina, Morayo A. Ogunsina  

**Link**: [PDF](https://arxiv.org/pdf/2505.03553)  

**Abstract**: Inconsistent outputs and hallucinations from large language models (LLMs) are major obstacles to reliable AI systems. When different proprietary reasoning models (RMs), such as those by OpenAI, Google, Anthropic, DeepSeek, and xAI, are given the same complex request, they often produce divergent results due to variations in training and inference. This paper proposes a novel consensus mechanism, inspired by distributed ledger technology, to validate and converge these outputs, treating each RM as a black-box peer. Building on the Hashgraph consensus algorithm, our approach employs gossip-about-gossip communication and virtual voting to achieve agreement among an ensemble of RMs. We present an architectural design for a prototype system in which RMs iteratively exchange and update their answers, using information from each round to improve accuracy and confidence in subsequent rounds. This approach goes beyond simple majority voting by incorporating the knowledge and cross-verification content of every model. We justify the feasibility of this Hashgraph-inspired consensus for AI ensembles and outline its advantages over traditional ensembling techniques in reducing nonfactual outputs. Preliminary considerations for implementation, evaluation criteria for convergence and accuracy, and potential challenges are discussed. The proposed mechanism demonstrates a promising direction for multi-agent AI systems to self-validate and deliver high-fidelity responses in complex tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）的不一致输出和幻觉是可靠AI系统的主要障碍。本论文提出了一种受分布式账本技术启发的新型共识机制，用于验证和收敛这些输出，将每个推理模型（RM）视为黑盒 peer。基于 Hashgraph 共识算法，我们的方法采用辗转相告通信和虚拟投票来达成一组 RMs 之间的共识。我们提出了一种原型系统的架构设计，在该设计中，RMs 逐步交换和更新答案，并利用每轮的反馈信息改进后续轮次的准确性和信心。这种方法通过整合每个模型的知识和交叉验证内容，超越了简单的多数表决。我们论证了这种受 Hashgraph 启发的共识机制在AI集成系统中的可行性，并概述了它在减少非事实输出方面相对于传统集成技术的优势。讨论了实施的初步考虑、收敛和准确性的评估标准，以及潜在挑战。所提机制为多Agent AI系统在复杂任务中自我验证并提供高保真响应指明了一条有希望的方向。 

---
# STORY2GAME: Generating (Almost) Everything in an Interactive Fiction Game 

**Title (ZH)**: STORY2GAME：生成交互式 fiction 游戏中几乎一切内容 

**Authors**: Eric Zhou, Shreyas Basavatia, Moontashir Siam, Zexin Chen, Mark O. Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2505.03547)  

**Abstract**: We introduce STORY2GAME, a novel approach to using Large Language Models to generate text-based interactive fiction games that starts by generating a story, populates the world, and builds the code for actions in a game engine that enables the story to play out interactively. Whereas a given set of hard-coded actions can artificially constrain story generation, the ability to generate actions means the story generation process can be more open-ended but still allow for experiences that are grounded in a game state. The key to successful action generation is to use LLM-generated preconditions and effects of actions in the stories as guides for what aspects of the game state must be tracked and changed by the game engine when a player performs an action. We also introduce a technique for dynamically generating new actions to accommodate the player's desire to perform actions that they think of that are not part of the story. Dynamic action generation may require on-the-fly updates to the game engine's state representation and revision of previously generated actions. We evaluate the success rate of action code generation with respect to whether a player can interactively play through the entire generated story. 

**Abstract (ZH)**: 我们介绍了一种名为STORY2GAME的新型方法，该方法利用大型语言模型生成基于文本的交互式小说游戏，首先生成一个故事，填充世界，并在游戏引擎中构建代码以实现故事的互动播放。与硬编码的一组固定动作可能人为地限制故事生成不同，能够生成动作意味着故事生成过程可以更加开放，但仍能确保体验与游戏状态紧密结合。成功生成动作的关键在于使用大语言模型生成的动作的前提条件和效果作为游戏中必须跟踪和改变状态的指南。我们还介绍了一种动态生成新动作的技术，以适应玩家希望执行故事中未包含的动作的需求。动态生成动作可能需要游戏引擎状态表示的实时更新以及之前生成的动作的修订。我们以玩家能否完整互动地游玩整个生成的故事来评估动作代码生成的成功率。 

---
# am-ELO: A Stable Framework for Arena-based LLM Evaluation 

**Title (ZH)**: Arena-based LLM评估的稳定框架：am-ELO 

**Authors**: Zirui Liu, Jiatong Li, Yan Zhuang, Qi Liu, Shuanghong Shen, Jie Ouyang, Mingyue Cheng, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03475)  

**Abstract**: Arena-based evaluation is a fundamental yet significant evaluation paradigm for modern AI models, especially large language models (LLMs). Existing framework based on ELO rating system suffers from the inevitable instability problem due to ranking inconsistency and the lack of attention to the varying abilities of annotators. In this paper, we introduce a novel stable arena framework to address these issues by enhancing the ELO Rating System. Specifically, we replace the iterative update method with a Maximum Likelihood Estimation (MLE) approach, m-ELO, and provide theoretical proof of the consistency and stability of the MLE approach for model ranking. Additionally, we proposed the am-ELO, which modify the Elo Rating's probability function to incorporate annotator abilities, enabling the simultaneous estimation of model scores and annotator reliability. Experiments demonstrate that this method ensures stability, proving that this framework offers a more robust, accurate, and stable evaluation method for LLMs. 

**Abstract (ZH)**: 基于竞技场的评估是一种对于现代AI模型，尤其是大型语言模型（LLMs）的基本而重要的评估范式。基于ELO评分系统的现有框架由于排名不一致性和忽视注释员能力变化而不可避免地存在稳定性问题。在本文中，我们引入了一种新颖的稳定竞技场框架，通过增强ELO评分系统来解决这些问题。具体地，我们用最大似然估计（MLE）方法替代迭代更新方法，并提供了MLE方法在模型排名中一致性和稳定性的理论证明。此外，我们提出了am-ELO，通过修改ELO评分的概率函数来整合注释员的能力，使得可以同时估计模型得分和注释员的可靠性。实验表明，该方法确保了稳定性，证明了该框架提供了一种更稳健、准确且稳定的LLM评估方法。 

---
# The Steganographic Potentials of Language Models 

**Title (ZH)**: 语言模型的隐写 potentials 翻译为：语言模型的隐写潜力 

**Authors**: Artem Karpov, Tinuade Adeleke, Seong Hah Cho, Natalia Perez-Campanero  

**Link**: [PDF](https://arxiv.org/pdf/2505.03439)  

**Abstract**: The potential for large language models (LLMs) to hide messages within plain text (steganography) poses a challenge to detection and thwarting of unaligned AI agents, and undermines faithfulness of LLMs reasoning. We explore the steganographic capabilities of LLMs fine-tuned via reinforcement learning (RL) to: (1) develop covert encoding schemes, (2) engage in steganography when prompted, and (3) utilize steganography in realistic scenarios where hidden reasoning is likely, but not prompted. In these scenarios, we detect the intention of LLMs to hide their reasoning as well as their steganography performance. Our findings in the fine-tuning experiments as well as in behavioral non fine-tuning evaluations reveal that while current models exhibit rudimentary steganographic abilities in terms of security and capacity, explicit algorithmic guidance markedly enhances their capacity for information concealment. 

**Abstract (ZH)**: 大型语言模型在明文中隐藏信息（隐写）的潜在能力对检测和抵御未对齐的AI代理构成挑战，并损害了大型语言模型推理的可信度。我们探讨了通过强化学习（RL）微调的大型语言模型的隐写能力：（1）开发隐蔽编码方案，（2）在提示时进行隐写，以及（3）在隐藏推理很可能但未提示的现实场景中利用隐写。在这些场景中，我们检测到大型语言模型隐藏推理的意图及其隐写性能。在微调实验以及行为非微调评估中的发现表明，尽管当前模型在安全性与容量方面表现出初步的隐写能力，但明确的算法指导显著增强了信息隐藏的能力。 

---
# Procedural Memory Is Not All You Need: Bridging Cognitive Gaps in LLM-Based Agents 

**Title (ZH)**: 程序记忆并非足矣：连接基于LLM的代理的认知缺口 

**Authors**: Schaun Wheeler, Olivier Jeunen  

**Link**: [PDF](https://arxiv.org/pdf/2505.03434)  

**Abstract**: Large Language Models (LLMs) represent a landmark achievement in Artificial Intelligence (AI), demonstrating unprecedented proficiency in procedural tasks such as text generation, code completion, and conversational coherence. These capabilities stem from their architecture, which mirrors human procedural memory -- the brain's ability to automate repetitive, pattern-driven tasks through practice. However, as LLMs are increasingly deployed in real-world applications, it becomes impossible to ignore their limitations operating in complex, unpredictable environments. This paper argues that LLMs, while transformative, are fundamentally constrained by their reliance on procedural memory. To create agents capable of navigating ``wicked'' learning environments -- where rules shift, feedback is ambiguous, and novelty is the norm -- we must augment LLMs with semantic memory and associative learning systems. By adopting a modular architecture that decouples these cognitive functions, we can bridge the gap between narrow procedural expertise and the adaptive intelligence required for real-world problem-solving. 

**Abstract (ZH)**: 大规模语言模型（LLMs）代表了人工智能（AI）领域的里程碑成就，展示了在文本生成、代码补全和对话连贯性等程序性任务上的无precedent的能效。这些能力源自于它们的架构，这种架构模仿了人类程序性记忆——大脑通过练习自动化重复的、基于模式的任务的能力。然而，随着LLMs在真实世界应用中的日益部署，它们在复杂、不可预测环境中的局限性变得无法忽视。本文认为，尽管LLMs具有革命性意义，但它们从根本上依赖于程序性记忆而受到限制。要创建能够导航“棘手”学习环境中的代理——在这些环境中规则会变化、反馈具有模糊性且新颖性是常态——我们必须增强LLMs以具备语义记忆和关联学习系统。通过采用模块化架构解耦这些认知功能，我们可以弥合狭隘的程序性专长相对于适应性智能在现实世界问题解决中的需求之间的差距。 

---
# Validating the Effectiveness of a Large Language Model-based Approach for Identifying Children's Development across Various Free Play Settings in Kindergarten 

**Title (ZH)**: 基于大型语言模型的方法在幼儿园不同自由玩耍情境下识别儿童发展的有效性验证 

**Authors**: Yuanyuan Yang, Yuan Shen, Tianchen Sun, Yangbin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.03369)  

**Abstract**: Free play is a fundamental aspect of early childhood education, supporting children's cognitive, social, emotional, and motor development. However, assessing children's development during free play poses significant challenges due to the unstructured and spontaneous nature of the activity. Traditional assessment methods often rely on direct observations by teachers, parents, or researchers, which may fail to capture comprehensive insights from free play and provide timely feedback to educators. This study proposes an innovative approach combining Large Language Models (LLMs) with learning analytics to analyze children's self-narratives of their play experiences. The LLM identifies developmental abilities, while performance scores across different play settings are calculated using learning analytics techniques. We collected 2,224 play narratives from 29 children in a kindergarten, covering four distinct play areas over one semester. According to the evaluation results from eight professionals, the LLM-based approach achieved high accuracy in identifying cognitive, motor, and social abilities, with accuracy exceeding 90% in most domains. Moreover, significant differences in developmental outcomes were observed across play settings, highlighting each area's unique contributions to specific abilities. These findings confirm that the proposed approach is effective in identifying children's development across various free play settings. This study demonstrates the potential of integrating LLMs and learning analytics to provide child-centered insights into developmental trajectories, offering educators valuable data to support personalized learning and enhance early childhood education practices. 

**Abstract (ZH)**: 自由游戏是幼儿教育的基本方面，支持儿童的认知、社会、情感和运动发展。然而，评估儿童在自由游戏中的发展由于活动的无结构和自发性而面临重大挑战。传统评估方法通常依赖于教师、家长或研究者的直接观察，这可能无法全面捕捉自由游戏的洞察并为教育者提供及时反馈。本研究提出了一种结合大型语言模型（LLMs）与学习分析的方法，以分析儿童对游戏体验的自我叙述。LLM识别发展能力，不同游戏环境下的表现分数则通过学习分析技术进行计算。我们在一所 kindergarten 收集了29名儿童在四个不同游戏区域为期一学期的2224份游戏叙述。根据八名专业人士的评估结果，基于LLM的方法在识别认知、运动和社会能力方面达到了较高的准确性，大多数领域超过90%的准确率。此外，还观察到不同游戏环境下的发展结果存在显著差异，突显了每个区域对特定能力的独特贡献。这些发现证实该方法有效识别不同自由游戏环境中的儿童发展。本研究展示了将LLMs与学习分析结合的潜力，以提供以儿童为中心的发展轨迹洞察，为教育者提供有价值的数据支持个性化学习并提升幼儿教育实践。 

---
# AI-Driven Scholarly Peer Review via Persistent Workflow Prompting, Meta-Prompting, and Meta-Reasoning 

**Title (ZH)**: 基于持续工作流提示、元提示和元推理的AI驱动学术同行评审 

**Authors**: Evgeny Markhasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03332)  

**Abstract**: Critical peer review of scientific manuscripts presents a significant challenge for Large Language Models (LLMs), partly due to data limitations and the complexity of expert reasoning. This report introduces Persistent Workflow Prompting (PWP), a potentially broadly applicable prompt engineering methodology designed to bridge this gap using standard LLM chat interfaces (zero-code, no APIs). We present a proof-of-concept PWP prompt for the critical analysis of experimental chemistry manuscripts, featuring a hierarchical, modular architecture (structured via Markdown) that defines detailed analysis workflows. We develop this PWP prompt through iterative application of meta-prompting techniques and meta-reasoning aimed at systematically codifying expert review workflows, including tacit knowledge. Submitted once at the start of a session, this PWP prompt equips the LLM with persistent workflows triggered by subsequent queries, guiding modern reasoning LLMs through systematic, multimodal evaluations. Demonstrations show the PWP-guided LLM identifying major methodological flaws in a test case while mitigating LLM input bias and performing complex tasks, including distinguishing claims from evidence, integrating text/photo/figure analysis to infer parameters, executing quantitative feasibility checks, comparing estimates against claims, and assessing a priori plausibility. To ensure transparency and facilitate replication, we provide full prompts, detailed demonstration analyses, and logs of interactive chats as supplementary resources. Beyond the specific application, this work offers insights into the meta-development process itself, highlighting the potential of PWP, informed by detailed workflow formalization, to enable sophisticated analysis using readily available LLMs for complex scientific tasks. 

**Abstract (ZH)**: 持久工作流提示：一种用于大型语言模型的批判性同行评审方法 

---
# RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation 

**Title (ZH)**: RAG-MCP：通过检索增强生成减轻大模型工具选择中的提示膨胀问题 

**Authors**: Tiantian Gan, Qiyao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.03275)  

**Abstract**: Large language models (LLMs) struggle to effectively utilize a growing number of external tools, such as those defined by the Model Context Protocol (MCP)\cite{IntroducingMCP}, due to prompt bloat and selection complexity. We introduce RAG-MCP, a Retrieval-Augmented Generation framework that overcomes this challenge by offloading tool discovery. RAG-MCP uses semantic retrieval to identify the most relevant MCP(s) for a given query from an external index before engaging the LLM. Only the selected tool descriptions are passed to the model, drastically reducing prompt size and simplifying decision-making. Experiments, including an MCP stress test, demonstrate RAG-MCP significantly cuts prompt tokens (e.g., by over 50%) and more than triples tool selection accuracy (43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables scalable and accurate tool integration for LLMs. 

**Abstract (ZH)**: Large语言模型（LLMs）难以有效利用不断增长的外部工具，例如由模型上下文协议（MCP）定义的工具，由于提示膨胀和选择复杂性。我们引入了RAG-MCP，这是一种检索增强生成框架，通过卸载工具发现来克服这一挑战。RAG-MCP使用语义检索，从外部索引中识别与给定查询最相关的MCP(s)，然后才让LLM参与。仅将选定的工具描述传递给模型，极大地减少了提示大小并简化了决策过程。实验，包括MCP压力测试，表明RAG-MCP显著减少了提示令牌数量（例如，超过50%）并在基准任务上将工具选择准确性提高了三倍多（43.13% vs 13.62%基线）。RAG-MCP使LLM的可扩展且准确的工具集成成为可能。 

---
# Patterns and Mechanisms of Contrastive Activation Engineering 

**Title (ZH)**: 对比激活工程的模式与机制 

**Authors**: Yixiong Hao, Ayush Panda, Stepan Shabalin, Sheikh Abdur Raheem Ali  

**Link**: [PDF](https://arxiv.org/pdf/2505.03189)  

**Abstract**: Controlling the behavior of Large Language Models (LLMs) remains a significant challenge due to their inherent complexity and opacity. While techniques like fine-tuning can modify model behavior, they typically require extensive computational resources. Recent work has introduced a class of contrastive activation engineering (CAE) techniques as promising approaches for steering LLM outputs through targeted modifications to their internal representations. Applied at inference-time with zero cost, CAE has the potential to introduce a new paradigm of flexible, task-specific LLM behavior tuning. We analyze the performance of CAE in in-distribution, out-of-distribution settings, evaluate drawbacks, and begin to develop comprehensive guidelines for its effective deployment. We find that 1. CAE is only reliably effective when applied to in-distribution contexts. 2. Increasing the number of samples used to generate steering vectors has diminishing returns at around 80 samples. 3. Steering vectors are susceptible to adversarial inputs that reverses the behavior that is steered for. 4. Steering vectors harm the overall model perplexity. 5. Larger models are more resistant to steering-induced degradation. 

**Abstract (ZH)**: 控制大型语言模型的行为仍然是一个重大挑战，由于它们固有的复杂性和不透明性。对比激活工程（CAE）技术作为通过目标修改其内部表示来引导LLM输出的有希望的方法，已在推理时零成本下引入，具有引入灵活、任务特定的LLM行为调优新范式的潜力。我们分析了CAE在分布内和分布外设置中的性能，评估了其缺点，并开始制定全面的指南以促进其有效部署。我们发现：1. CAE仅在应用于分布内上下文中时才可靠有效。2. 用于生成引导向量的样本数量增加到约80个时，其效果递减。3. 引导向量对对抗输入敏感，这些输入会使引导行为反转。4. 引导向量损害了整体模型的困惑度。5. 较大的模型对引导引起的退化更具抵抗力。 

---
# CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics 

**Title (ZH)**: CombiBench: 组合数学领域大语言模型能力基准评测 

**Authors**: Junqi Liu, Xiaohan Lin, Jonas Bayer, Yael Dillies, Weijie Jiang, Xiaodan Liang, Roman Soletskyi, Haiming Wang, Yunzhou Xie, Beibei Xiong, Zhengfeng Yang, Jujian Zhang, Lihong Zhi, Jia Li, Zhengying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03171)  

**Abstract**: Neurosymbolic approaches integrating large language models with formal reasoning have recently achieved human-level performance on mathematics competition problems in algebra, geometry and number theory. In comparison, combinatorics remains a challenging domain, characterized by a lack of appropriate benchmarks and theorem libraries. To address this gap, we introduce CombiBench, a comprehensive benchmark comprising 100 combinatorial problems, each formalized in Lean~4 and paired with its corresponding informal statement. The problem set covers a wide spectrum of difficulty levels, ranging from middle school to IMO and university level, and span over ten combinatorial topics. CombiBench is suitable for testing IMO solving capabilities since it includes all IMO combinatorial problems since 2000 (except IMO 2004 P3 as its statement contain an images). Furthermore, we provide a comprehensive and standardized evaluation framework, dubbed Fine-Eval (for $\textbf{F}$ill-in-the-blank $\textbf{in}$ L$\textbf{e}$an Evaluation), for formal mathematics. It accommodates not only proof-based problems but also, for the first time, the evaluation of fill-in-the-blank questions. Using Fine-Eval as the evaluation method and Kimina Lean Server as the backend, we benchmark several LLMs on CombiBench and observe that their capabilities for formally solving combinatorial problems remain limited. Among all models tested (none of which has been trained for this particular task), Kimina-Prover attains the best results, solving 7 problems (out of 100) under both ``with solution'' and ``without solution'' scenarios. We open source the benchmark dataset alongside with the code of the proposed evaluation method at this https URL. 

**Abstract (ZH)**: 神经符号方法结合大型语言模型与形式推理在组合数学竞赛问题上已达到人类水平，在代数、几何和数论方面取得了进展。相比之下，组合数学仍是具有挑战性的领域，特征为缺乏合适的基准和定理库。为应对这一挑战，我们引入了CombiBench，这是一个包含100个组合数学问题的综合基准，每个问题均用Lean 4形式化，并配以相应的非形式化陈述。问题集涵盖了从小学到国际数学奥林匹克（IMO）和大学水平的广泛难度级别，涵盖十个组合数学主题。CombiBench 适合测试IMO解题能力，因为它包括了2000年以来的所有IMO组合数学问题（除了2004年IMO P3，因其陈述包含图片）。此外，我们还提供了一种全面且标准化的评估框架，名为Fine-Eval（用于Lean的填空评估），用于正式数学的评估。Fine-Eval 不仅适用于证明问题，还首次适用于填空题的评估。使用Fine-Eval 作为评估方法和Kimina Lean Server 作为后端，我们在CombiBench 上测试了几种语言模型，并观察到它们在形式化解决组合数学问题方面的能力仍然有限。在所有测试的模型（其中没有一个专门为此任务训练）中，Kimina-Prover 获得了最佳结果，在“有解”和“无解”两种场景下分别解决了7个问题（总计100个）。我们已经开源了基准数据集及其所提出的评估方法的代码，可通过以下链接访问：this https URL。 

---
# Holmes: Automated Fact Check with Large Language Models 

**Title (ZH)**: Holmes: 使用大规模语言模型的自动事实核查 

**Authors**: Haoran Ou, Gelei Deng, Xingshuo Han, Jie Zhang, Xinlei He, Han Qiu, Shangwei Guo, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03135)  

**Abstract**: The rise of Internet connectivity has accelerated the spread of disinformation, threatening societal trust, decision-making, and national security. Disinformation has evolved from simple text to complex multimodal forms combining images and text, challenging existing detection methods. Traditional deep learning models struggle to capture the complexity of multimodal disinformation. Inspired by advances in AI, this study explores using Large Language Models (LLMs) for automated disinformation detection. The empirical study shows that (1) LLMs alone cannot reliably assess the truthfulness of claims; (2) providing relevant evidence significantly improves their performance; (3) however, LLMs cannot autonomously search for accurate evidence. To address this, we propose Holmes, an end-to-end framework featuring a novel evidence retrieval method that assists LLMs in collecting high-quality evidence. Our approach uses (1) LLM-powered summarization to extract key information from open sources and (2) a new algorithm and metrics to evaluate evidence quality. Holmes enables LLMs to verify claims and generate justifications effectively. Experiments show Holmes achieves 88.3% accuracy on two open-source datasets and 90.2% in real-time verification tasks. Notably, our improved evidence retrieval boosts fact-checking accuracy by 30.8% over existing methods 

**Abstract (ZH)**: 互联网连接的普及加速了虚假信息的传播，威胁着社会信任、决策和国家安全。虚假信息从简单的文本发展为结合图像和文本的复杂多媒体形式，挑战现有的检测方法。传统的深度学习模型难以捕获多媒体虚假信息的复杂性。受人工智能advance的启发，本研究探讨了使用大型语言模型（LLMs）进行自动化虚假信息检测的可能性。实证研究显示：（1）LLMs单独使用不能可靠地评估声明的真实性；（2）提供相关证据能显著提高其性能；（3）然而，LLMs不能自主搜索准确的证据。为此，我们提出了Holmes框架，这是一种端到端框架，具有新颖的证据检索方法，协助LLMs收集高质量的证据。我们的方法包括：（1）利用LLM进行总结，从开源资源中提取关键信息；（2）提出新的算法和指标来评估证据质量。Holmes使LLMs能够有效地验证声明并生成说明。实验结果显示，Holmes在两个开源数据集上的准确率为88.3%，在实时验证任务中的准确率为90.2%。值得注意的是，我们改进的证据检索方法使事实核查的准确性提高了30.8%。 

---
# BLAB: Brutally Long Audio Bench 

**Title (ZH)**: BRAB: 剧烈长音频基准 

**Authors**: Orevaoghene Ahia, Martijn Bartelds, Kabir Ahuja, Hila Gonen, Valentin Hofmann, Siddhant Arora, Shuyue Stella Li, Vishal Puttagunta, Mofetoluwa Adeyemi, Charishma Buchireddy, Ben Walls, Noah Bennett, Shinji Watanabe, Noah A. Smith, Yulia Tsvetkov, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03054)  

**Abstract**: Developing large audio language models (LMs) capable of understanding diverse spoken interactions is essential for accommodating the multimodal nature of human communication and can increase the accessibility of language technologies across different user populations. Recent work on audio LMs has primarily evaluated their performance on short audio segments, typically under 30 seconds, with limited exploration of long-form conversational speech segments that more closely reflect natural user interactions with these models. We introduce Brutally Long Audio Bench (BLAB), a challenging long-form audio benchmark that evaluates audio LMs on localization, duration estimation, emotion, and counting tasks using audio segments averaging 51 minutes in length. BLAB consists of 833+ hours of diverse, full-length audio clips, each paired with human-annotated, text-based natural language questions and answers. Our audio data were collected from permissively licensed sources and underwent a human-assisted filtering process to ensure task compliance. We evaluate six open-source and proprietary audio LMs on BLAB and find that all of them, including advanced models such as Gemini 2.0 Pro and GPT-4o, struggle with the tasks in BLAB. Our comprehensive analysis reveals key insights into the trade-offs between task difficulty and audio duration. In general, we find that audio LMs struggle with long-form speech, with performance declining as duration increases. They perform poorly on localization, temporal reasoning, counting, and struggle to understand non-phonemic information, relying more on prompts than audio content. BLAB serves as a challenging evaluation framework to develop audio LMs with robust long-form audio understanding capabilities. 

**Abstract (ZH)**: 开发能够在长时间音频对话中理解多样口语交互的大规模音频语言模型对于适应人类多模态通信的性质并提高语言技术在不同用户群体中的 accessibility 至关重要。近年来，关于音频语言模型的研究主要评估其在短音频片段上的性能，通常少于30秒，对更长的对话音频片段的研究有限，这类音频片段更接近于自然用户与这些模型的互动。我们引入了Brutally Long Audio Bench (BLAB) 挑战性长格式音频基准，该基准使用平均时长为51分钟的音频片段评估音频语言模型在定位、时长估计、情绪识别和计数任务上的表现。BLAB 包含833+小时的多样化全长度音频剪辑，每个剪辑配有人工标注的基于文本的自然语言问题和答案。我们从许可来源收集了音频数据，并通过人工辅助的筛选过程确保任务合规。我们在 BLAB 上评估了六个开源和专有音频语言模型，发现所有模型，包括高级模型如 Gemini 2.0 Pro 和 GPT-4o，在 BLAB 上的这些任务中都表现出困难。我们的全面分析揭示了任务难度和音频时长之间的关键权衡。总体而言，我们发现音频语言模型在处理长格式语音时面临困难，随着时长增加，其性能下降。它们在定位、时间推理和计数方面表现不佳，难以理解非语音信息，更多依赖于提示而非音频内容。BLAB 作为一项具有挑战性的评估框架，旨在开发具有稳健长格式音频理解能力的音频语言模型。 

---
# Evaluating the Impact of AI-Powered Audiovisual Personalization on Learner Emotion, Focus, and Learning Outcomes 

**Title (ZH)**: 评估人工智能驱动的音视频个性化对学习者情绪、专注度和学习成果的影响 

**Authors**: George Xi Wang, Jingying Deng, Safinah Ali  

**Link**: [PDF](https://arxiv.org/pdf/2505.03033)  

**Abstract**: Independent learners often struggle with sustaining focus and emotional regulation in unstructured or distracting settings. Although some rely on ambient aids such as music, ASMR, or visual backgrounds to support concentration, these tools are rarely integrated into cohesive, learner-centered systems. Moreover, existing educational technologies focus primarily on content adaptation and feedback, overlooking the emotional and sensory context in which learning takes place. Large language models have demonstrated powerful multimodal capabilities including the ability to generate and adapt text, audio, and visual content. Educational research has yet to fully explore their potential in creating personalized audiovisual learning environments. To address this gap, we introduce an AI-powered system that uses LLMs to generate personalized multisensory study environments. Users select or generate customized visual themes (e.g., abstract vs. realistic, static vs. animated) and auditory elements (e.g., white noise, ambient ASMR, familiar vs. novel sounds) to create immersive settings aimed at reducing distraction and enhancing emotional stability. Our primary research question investigates how combinations of personalized audiovisual elements affect learner cognitive load and engagement. Using a mixed-methods design that incorporates biometric measures and performance outcomes, this study evaluates the effectiveness of LLM-driven sensory personalization. The findings aim to advance emotionally responsive educational technologies and extend the application of multimodal LLMs into the sensory dimension of self-directed learning. 

**Abstract (ZH)**: 独立学习者往往在无结构或有干扰的环境中难以维持专注力和情绪调节。尽管有些人依赖环境辅助工具，如音乐、ASMR或视觉背景来支持集中注意力，但这些工具很少被整合到以学习者为中心的系统中。此外，现有的教育技术主要集中在内容适应和反馈上，忽略了学习过程中所处的情感和感官背景。大规模语言模型展示了强大的多模态能力，包括生成和适应文本、音频和视觉内容的能力。教育研究尚未充分探索在创建个性化的视听学习环境方面的潜力。为填补这一空白，我们介绍了一个基于AI的系统，该系统使用LLM生成个性化的多感官学习环境。用户可以选择或生成定制的视觉主题（例如，抽象 vs. 现实，静态 vs. 动画）和听觉元素（例如，白噪音、环境ASMR、熟悉 vs. 新奇声音），以创建减少干扰并增强情绪稳定的沉浸式环境。我们主要的研究问题是探讨个性化视听元素的不同组合如何影响学习者的认知负荷和参与度。本研究采用结合生物测量和绩效结果的混合方法设计，评估LLM驱动的感官个性化效果。研究结果旨在推动具有情感响应性的教育技术，并将多模态LLM的应用扩展到自主学习的感官维度。 

---
# VITA-Audio: Fast Interleaved Cross-Modal Token Generation for Efficient Large Speech-Language Model 

**Title (ZH)**: VITA-音频：快速交错跨模态令牌生成以实现高效大型语音-语言模型 

**Authors**: Zuwei Long, Yunhang Shen, Chaoyou Fu, Heting Gao, Lijiang Li, Peixian Chen, Mengdan Zhang, Hang Shao, Jian Li, Jinlong Peng, Haoyu Cao, Ke Li, Rongrong Ji, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.03739)  

**Abstract**: With the growing requirement for natural human-computer interaction, speech-based systems receive increasing attention as speech is one of the most common forms of daily communication. However, the existing speech models still experience high latency when generating the first audio token during streaming, which poses a significant bottleneck for deployment. To address this issue, we propose VITA-Audio, an end-to-end large speech model with fast audio-text token generation. Specifically, we introduce a lightweight Multiple Cross-modal Token Prediction (MCTP) module that efficiently generates multiple audio tokens within a single model forward pass, which not only accelerates the inference but also significantly reduces the latency for generating the first audio in streaming scenarios. In addition, a four-stage progressive training strategy is explored to achieve model acceleration with minimal loss of speech quality. To our knowledge, VITA-Audio is the first multi-modal large language model capable of generating audio output during the first forward pass, enabling real-time conversational capabilities with minimal latency. VITA-Audio is fully reproducible and is trained on open-source data only. Experimental results demonstrate that our model achieves an inference speedup of 3~5x at the 7B parameter scale, but also significantly outperforms open-source models of similar model size on multiple benchmarks for automatic speech recognition (ASR), text-to-speech (TTS), and spoken question answering (SQA) tasks. 

**Abstract (ZH)**: 基于语音的端到端大型语音模型VITA-Audio：高效音频文本令牌生成 

---
# ReGraP-LLaVA: Reasoning enabled Graph-based Personalized Large Language and Vision Assistant 

**Title (ZH)**: ReGraP-LLaVA: 基于图的推理增强个性化大型语言和视觉助手 

**Authors**: Yifan Xiang, Zhenxi Zhang, Bin Li, Yixuan Weng, Shoujun Zhou, Yangfan He, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03654)  

**Abstract**: Recent advances in personalized MLLMs enable effective capture of user-specific concepts, supporting both recognition of personalized concepts and contextual captioning. However, humans typically explore and reason over relations among objects and individuals, transcending surface-level information to achieve more personalized and contextual understanding. To this end, existing methods may face three main limitations: Their training data lacks multi-object sets in which relations among objects are learnable. Building on the limited training data, their models overlook the relations between different personalized concepts and fail to reason over them. Their experiments mainly focus on a single personalized concept, where evaluations are limited to recognition and captioning tasks. To address the limitations, we present a new dataset named ReGraP, consisting of 120 sets of personalized knowledge. Each set includes images, KGs, and CoT QA pairs derived from the KGs, enabling more structured and sophisticated reasoning pathways. We propose ReGraP-LLaVA, an MLLM trained with the corresponding KGs and CoT QA pairs, where soft and hard graph prompting methods are designed to align KGs within the model's semantic space. We establish the ReGraP Benchmark, which contains diverse task types: multiple-choice, fill-in-the-blank, True/False, and descriptive questions in both open- and closed-ended settings. The proposed benchmark is designed to evaluate the relational reasoning and knowledge-connection capability of personalized MLLMs. We conduct experiments on the proposed ReGraP-LLaVA and other competitive MLLMs. Results show that the proposed model not only learns personalized knowledge but also performs relational reasoning in responses, achieving the SoTA performance compared with the competitive methods. All the codes and datasets are released at: this https URL. 

**Abstract (ZH)**: Recent advances in个性化MLLMs使用户特定概念的捕捉更加有效，支持个性化概念的识别和上下文描述。然而，人类通常探索和推理对象及个体之间的关系，超越表面信息以达成更加个性化和上下文的理解。为此，现有方法可能面临三大限制：其训练数据缺乏可学习对象间关系的多对象集合。基于有限的训练数据，其模型忽视了不同个性化概念之间的关系并无法推理它们。其实验主要集中在单一个性化概念上，评估主要限于识别和描述任务。为解决这些问题，我们提出一个新的名为ReGraP的数据集，包含120个个性化知识集合。每个集合包括图像、知识图谱（KG）和基于KG的CoT QA对，从而提供更结构化和复杂的推理路径。我们提出ReGraP-LLaVA，该模型使用相应的KG和CoT QA对进行训练，并设计了软性和硬性图结构提示方法，使其KGs在模型的语义空间内对齐。我们建立了ReGraP基准，包含多种任务类型：多项选择题、填充题、真/假判断及开放式和封闭式的描述性问题。该提出的基准旨在评估个性化MLLMs的关系推理和知识连接能力。我们在提出的ReGraP-LLaVA和其它竞争性MLLMs上进行了实验。结果显示，提出的模型不仅学习了个性化知识，还在响应中进行了关系推理，性能优于其他竞争方法。所有代码和数据集发布在：https://your-link-url.com。 

---
# LlamaFirewall: An open source guardrail system for building secure AI agents 

**Title (ZH)**: LlamaFirewall: 一个开源的安全护栏系统，用于构建安全的AI代理 

**Authors**: Sahana Chennabasappa, Cyrus Nikolaidis, Daniel Song, David Molnar, Stephanie Ding, Shengye Wan, Spencer Whitman, Lauren Deason, Nicholas Doucette, Abraham Montilla, Alekhya Gampa, Beto de Paola, Dominik Gabi, James Crnkovich, Jean-Christophe Testud, Kat He, Rashnil Chaturvedi, Wu Zhou, Joshua Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2505.03574)  

**Abstract**: Large language models (LLMs) have evolved from simple chatbots into autonomous agents capable of performing complex tasks such as editing production code, orchestrating workflows, and taking higher-stakes actions based on untrusted inputs like webpages and emails. These capabilities introduce new security risks that existing security measures, such as model fine-tuning or chatbot-focused guardrails, do not fully address. Given the higher stakes and the absence of deterministic solutions to mitigate these risks, there is a critical need for a real-time guardrail monitor to serve as a final layer of defense, and support system level, use case specific safety policy definition and enforcement. We introduce LlamaFirewall, an open-source security focused guardrail framework designed to serve as a final layer of defense against security risks associated with AI Agents. Our framework mitigates risks such as prompt injection, agent misalignment, and insecure code risks through three powerful guardrails: PromptGuard 2, a universal jailbreak detector that demonstrates clear state of the art performance; Agent Alignment Checks, a chain-of-thought auditor that inspects agent reasoning for prompt injection and goal misalignment, which, while still experimental, shows stronger efficacy at preventing indirect injections in general scenarios than previously proposed approaches; and CodeShield, an online static analysis engine that is both fast and extensible, aimed at preventing the generation of insecure or dangerous code by coding agents. Additionally, we include easy-to-use customizable scanners that make it possible for any developer who can write a regular expression or an LLM prompt to quickly update an agent's security guardrails. 

**Abstract (ZH)**: 大型语言模型（LLMs）已从简单的聊天机器人发展成为能够执行复杂任务（如编辑生产代码、协调工作流和基于网页和电子邮件等不可信输入采取高风险行动）的自主代理。这些能力引入了现有安全措施（如模型微调或聊天机器人专用的防护栏）无法充分解决的新安全风险。鉴于高风险且缺乏确定性的解决方案，急需一个实时防护栏监控系统作为最终的防御层，并支持针对特定应用场景的安全政策定义和执行。我们介绍了LlamaFirewall，一个开源的安全防护栏框架，旨在作为AI代理相关安全风险的最终防御层。我们的框架通过三种强有力的防护栏来降低风险，包括PromptGuard 2，一种通用的脱域检测器，展示出明显处于前沿的技术性能；Agent Alignment Checks，一种思维链审计器，检查代理推理以防止提示注入和目标偏移，尽管仍处于实验阶段，但在一般场景中显示了比之前提出的方法更强的间接注入预防效果；以及CodeShield，一个既快速又可扩展的在线静态分析引擎，旨在防止编码代理生成不安全或危险的代码。此外，我们还提供了易于使用的可定制扫描器，使任何能够编写正则表达式或LLM提示的开发人员都能够快速更新代理的安全防护栏。 

---
# MedArabiQ: Benchmarking Large Language Models on Arabic Medical Tasks 

**Title (ZH)**: MedArabiQ: 评估大型语言模型在阿拉伯医学任务中的性能 

**Authors**: Mouath Abu Daoud, Chaimae Abouzahir, Leen Kharouf, Walid Al-Eisawi, Nizar Habash, Farah E. Shamout  

**Link**: [PDF](https://arxiv.org/pdf/2505.03427)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant promise for various applications in healthcare. However, their efficacy in the Arabic medical domain remains unexplored due to the lack of high-quality domain-specific datasets and benchmarks. This study introduces MedArabiQ, a novel benchmark dataset consisting of seven Arabic medical tasks, covering multiple specialties and including multiple choice questions, fill-in-the-blank, and patient-doctor question answering. We first constructed the dataset using past medical exams and publicly available datasets. We then introduced different modifications to evaluate various LLM capabilities, including bias mitigation. We conducted an extensive evaluation with five state-of-the-art open-source and proprietary LLMs, including GPT-4o, Claude 3.5-Sonnet, and Gemini 1.5. Our findings highlight the need for the creation of new high-quality benchmarks that span different languages to ensure fair deployment and scalability of LLMs in healthcare. By establishing this benchmark and releasing the dataset, we provide a foundation for future research aimed at evaluating and enhancing the multilingual capabilities of LLMs for the equitable use of generative AI in healthcare. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗健康领域的应用展现了显著的潜力，然而它们在阿拉伯医学领域的效果尚未被探索，主要原因是缺乏高质量的专业领域数据集和基准。本研究 introduces MedArabiQ，一个包含七个阿拉伯医学任务的新颖基准数据集，覆盖多个专科，并包括多项选择题、填空题和病人-医生问答。我们首先使用过去的医学考试和公开可用的数据集构建了数据集。然后，我们引入了不同的修改来评估各种LLM的能力，包括偏见缓解。我们使用五种最新开源和专有LLM进行了广泛评估，包括GPT-4o、Claude 3.5-Sonnet和Gemini 1.5。我们的发现强调了创建跨越不同语言的新高质量基准的必要性，以确保LLMs在医疗健康领域中的公平部署和扩展。通过建立这一基准并发布数据集，我们为未来旨在评估和提升LLMs多语言能力的研究提供了基础，以促进生成性AI在医疗健康领域的公平使用。 

---
# Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation 

**Title (ZH)**: 基于QLoRA微调的大模型和检索增强生成的轻量级临床决策支持系统 

**Authors**: Mohammad Shoaib Ansari, Mohd Sohail Ali Khan, Shubham Revankar, Aditya Varma, Anil S. Mokhade  

**Link**: [PDF](https://arxiv.org/pdf/2505.03406)  

**Abstract**: This research paper investigates the application of Large Language Models (LLMs) in healthcare, specifically focusing on enhancing medical decision support through Retrieval-Augmented Generation (RAG) integrated with hospital-specific data and fine-tuning using Quantized Low-Rank Adaptation (QLoRA). The system utilizes Llama 3.2-3B-Instruct as its foundation model. By embedding and retrieving context-relevant healthcare information, the system significantly improves response accuracy. QLoRA facilitates notable parameter efficiency and memory optimization, preserving the integrity of medical information through specialized quantization techniques. Our research also shows that our model performs relatively well on various medical benchmarks, indicating that it can be used to make basic medical suggestions. This paper details the system's technical components, including its architecture, quantization methods, and key healthcare applications such as enhanced disease prediction from patient symptoms and medical history, treatment suggestions, and efficient summarization of complex medical reports. We touch on the ethical considerations-patient privacy, data security, and the need for rigorous clinical validation-as well as the practical challenges of integrating such systems into real-world healthcare workflows. Furthermore, the lightweight quantized weights ensure scalability and ease of deployment even in low-resource hospital environments. Finally, the paper concludes with an analysis of the broader impact of LLMs on healthcare and outlines future directions for LLMs in medical settings. 

**Abstract (ZH)**: 本研究论文探讨了大型语言模型（LLMs）在医疗领域的应用，特别关注通过结合医院特定数据和量化低秩适应（QLoRA）技术的检索增强生成（RAG）来提升医疗决策支持系统。该系统以Llama 3.2-3B-Instruct作为基础模型。通过嵌入和检索相关医疗信息，系统显著提高了响应准确性。QLoRA促进参数效率和内存优化，并通过专门的量化技术保持了医疗信息的完整性。研究表明，该模型在多种医疗基准测试中表现出色，表明其可用于提供基本的医疗建议。本文详细介绍了系统的技术组件，包括其架构、量化方法以及增强疾病预测、治疗建议和复杂医疗报告的高效总结等关键医疗应用。论文还讨论了伦理考虑（如患者隐私、数据安全和严格的临床验证需求）以及将此类系统整合到实际医疗工作流程中的实际挑战。轻量化的量化权重确保即使在资源有限的医院环境中，系统也具有可扩展性和易于部署的特点。最后，论文分析了大型语言模型在医疗领域的广泛影响，并概述了未来医疗环境中大型语言模型的发展方向。 

---
# Automatic Calibration for Membership Inference Attack on Large Language Models 

**Title (ZH)**: 针对大规模语言模型的成员推断攻击的自动校准 

**Authors**: Saleh Zare Zade, Yao Qiang, Xiangyu Zhou, Hui Zhu, Mohammad Amin Roshani, Prashant Khanduri, Dongxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03392)  

**Abstract**: Membership Inference Attacks (MIAs) have recently been employed to determine whether a specific text was part of the pre-training data of Large Language Models (LLMs). However, existing methods often misinfer non-members as members, leading to a high false positive rate, or depend on additional reference models for probability calibration, which limits their practicality. To overcome these challenges, we introduce a novel framework called Automatic Calibration Membership Inference Attack (ACMIA), which utilizes a tunable temperature to calibrate output probabilities effectively. This approach is inspired by our theoretical insights into maximum likelihood estimation during the pre-training of LLMs. We introduce ACMIA in three configurations designed to accommodate different levels of model access and increase the probability gap between members and non-members, improving the reliability and robustness of membership inference. Extensive experiments on various open-source LLMs demonstrate that our proposed attack is highly effective, robust, and generalizable, surpassing state-of-the-art baselines across three widely used benchmarks. Our code is available at: \href{this https URL}{\textcolor{blue}{Github}}. 

**Abstract (ZH)**: 自动校准会员推断攻击 (ACMIA)：一种利用可调温度有效校准输出概率的新框架 

---
# SPAP: Structured Pruning via Alternating Optimization and Penalty Methods 

**Title (ZH)**: SPAP：交替优化和惩罚方法引导的结构化剪枝 

**Authors**: Hanyu Hu, Xiaoming Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03373)  

**Abstract**: The deployment of large language models (LLMs) is often constrained by their substantial computational and memory demands. While structured pruning presents a viable approach by eliminating entire network components, existing methods suffer from performance degradation, reliance on heuristic metrics, or expensive finetuning. To address these challenges, we propose SPAP (Structured Pruning via Alternating Optimization and Penalty Methods), a novel and efficient structured pruning framework for LLMs grounded in optimization theory. SPAP formulates the pruning problem through a mixed-integer optimization model, employs a penalty method that effectively makes pruning decisions to minimize pruning errors, and introduces an alternating minimization algorithm tailored to the splittable problem structure for efficient weight updates and performance recovery. Extensive experiments on OPT, LLaMA-3/3.1/3.2, and Qwen2.5 models demonstrate SPAP's superiority over state-of-the-art methods, delivering linear inference speedups (1.29$\times$ at 30% sparsity) and proportional memory reductions. Our work offers a practical, optimization-driven solution for pruning LLMs while preserving model performance. 

**Abstract (ZH)**: 大型语言模型（LLM）的部署常常受限于其巨大的计算和内存需求。尽管结构化剪枝提供了一种可行的方法通过消除整个网络组件来缓解这一问题，但现有方法存在性能下降、依赖于启发式指标或昂贵的微调等问题。为了解决这些挑战，我们提出了一种基于优化理论的新颖且高效的结构化剪枝框架SPAP（交替优化和惩罚方法的结构化剪枝）。SPAP通过混合整数优化模型来形式化剪枝问题，采用惩罚方法有效地做出剪枝决策以最小化剪枝误差，并引入了一种适合拆分问题结构的交替最小化算法，以实现高效权重更新和性能恢复。在OPT、LLaMA-3/3.1/3.2和Qwen2.5模型上的广泛实验表明，SPAP在性能上优于最先进的方法，提供线性推理加速（在30%稀疏性下为1.29倍）和按比例减少的内存占用。我们的工作提供了一种实用的、基于优化的解决方案，可在保持模型性能的同时进行LLM剪枝。 

---
# Avoid Recommending Out-of-Domain Items: Constrained Generative Recommendation with LLMs 

**Title (ZH)**: 避免推荐领域外项目：受约束的生成式推荐方法（使用大语言模型） 

**Authors**: Hao Liao, Wensheng Lu, Jianxun Lian, Mingqi Wu, Shuo Wang, Yong Zhang, Yitian Huang, Mingyang Zhou, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.03336)  

**Abstract**: Large Language Models (LLMs) have shown promise for generative recommender systems due to their transformative capabilities in user interaction. However, ensuring they do not recommend out-of-domain (OOD) items remains a challenge. We study two distinct methods to address this issue: RecLM-ret, a retrieval-based method, and RecLM-cgen, a constrained generation method. Both methods integrate seamlessly with existing LLMs to ensure in-domain recommendations. Comprehensive experiments on three recommendation datasets demonstrate that RecLM-cgen consistently outperforms RecLM-ret and existing LLM-based recommender models in accuracy while eliminating OOD recommendations, making it the preferred method for adoption. Additionally, RecLM-cgen maintains strong generalist capabilities and is a lightweight plug-and-play module for easy integration into LLMs, offering valuable practical benefits for the community. Source code is available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成推荐系统中的应用展现了潜力，由于其在用户交互方面的变革性能力。然而，确保它们不会推荐领域外（OOD）的项目仍然是一项挑战。我们研究了两种不同的方法来解决这一问题：RecLM-ret，一种检索基方法，和RecLM-cgen，一种受限生成方法。这两种方法能无缝集成到现有的LLM中，以确保推荐的领域相关性。在三个推荐数据集上的全面实验表明，RecLM-cgen在准确性上始终优于RecLM-ret和现有的基于LLM的推荐模型，并且能够消除领域外推荐，使其成为首选的采用方法。此外，RecLM-cgen保持了强大的通识能力，并且是一个轻量级即插即用模块，易于集成到LLM中，为社区提供重要的实际益处。源代码可在此处访问：this https URL。 

---
# Absolute Zero: Reinforced Self-play Reasoning with Zero Data 

**Title (ZH)**: 绝对零度：零数据强化自我博弈推理 

**Authors**: Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03335)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown promise in enhancing the reasoning capabilities of large language models by learning directly from outcome-based rewards. Recent RLVR works that operate under the zero setting avoid supervision in labeling the reasoning process, but still depend on manually curated collections of questions and answers for training. The scarcity of high-quality, human-produced examples raises concerns about the long-term scalability of relying on human supervision, a challenge already evident in the domain of language model pretraining. Furthermore, in a hypothetical future where AI surpasses human intelligence, tasks provided by humans may offer limited learning potential for a superintelligent system. To address these concerns, we propose a new RLVR paradigm called Absolute Zero, in which a single model learns to propose tasks that maximize its own learning progress and improves reasoning by solving them, without relying on any external data. Under this paradigm, we introduce the Absolute Zero Reasoner (AZR), a system that self-evolves its training curriculum and reasoning ability by using a code executor to both validate proposed code reasoning tasks and verify answers, serving as an unified source of verifiable reward to guide open-ended yet grounded learning. Despite being trained entirely without external data, AZR achieves overall SOTA performance on coding and mathematical reasoning tasks, outperforming existing zero-setting models that rely on tens of thousands of in-domain human-curated examples. Furthermore, we demonstrate that AZR can be effectively applied across different model scales and is compatible with various model classes. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）通过直接从基于结果的奖励中学习，在增强大型语言模型的推理能力方面展现了潜力。Recent RLVR工作在零设定环境中运行，避免了标注推理过程的监督，但仍依赖于手动整理的问题和答案集合进行训练。高质量的人工生成示例的稀缺性引发了对长期依赖人工监督可行性的担忧，这一挑战在语言模型预训练领域已经显现。此外，在人工智能超越人类智能的假设未来中，人类提供的任务可能对超智能系统的学习潜力有限。为应对这些担忧，我们提出了一种新的RLVR paradigm，称为Absolute Zero，在该 paradigm中，单个模型学会提出最大化自身学习进度的任务，并通过解决这些任务来提高推理能力，而不依赖任何外部数据。在这一paradigm中，我们引入了Absolute Zero Reasoner（AZR）系统，该系统利用代码执行器验证提议的代码推理任务并验证答案，作为统一的验证奖励来源，引导开放而具体的learnings。尽管完全不依赖外部数据进行训练，AZR在编程和数学推理任务上的整体性能仍达到了SOTA水平，超越了依赖数万个人工整理示例的现有零设定模型。此外，我们证明了AZR可以在不同的模型规模上有效应用，并与多种模型类兼容。 

---
# A Trustworthy Multi-LLM Network: Challenges,Solutions, and A Use Case 

**Title (ZH)**: 可信赖的多大型语言模型网络：挑战、解决方案及一项应用案例 

**Authors**: Haoxiang Luo, Gang Sun, Yinqiu Liu, Dusit Niyato, Hongfang Yu, Mohammed Atiquzzaman, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03196)  

**Abstract**: Large Language Models (LLMs) demonstrate strong potential across a variety of tasks in communications and networking due to their advanced reasoning capabilities. However, because different LLMs have different model structures and are trained using distinct corpora and methods, they may offer varying optimization strategies for the same network issues. Moreover, the limitations of an individual LLM's training data, aggravated by the potential maliciousness of its hosting device, can result in responses with low confidence or even bias. To address these challenges, we propose a blockchain-enabled collaborative framework that connects multiple LLMs into a Trustworthy Multi-LLM Network (MultiLLMN). This architecture enables the cooperative evaluation and selection of the most reliable and high-quality responses to complex network optimization problems. Specifically, we begin by reviewing related work and highlighting the limitations of existing LLMs in collaboration and trust, emphasizing the need for trustworthiness in LLM-based systems. We then introduce the workflow and design of the proposed Trustworthy MultiLLMN framework. Given the severity of False Base Station (FBS) attacks in B5G and 6G communication systems and the difficulty of addressing such threats through traditional modeling techniques, we present FBS defense as a case study to empirically validate the effectiveness of our approach. Finally, we outline promising future research directions in this emerging area. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通信和网络领域各种任务中展现出强大的潜力，得益于其高级推理能力。然而，由于不同的LLMs具有不同的模型结构，并且采用不同的数据集和训练方法进行训练，它们可能在解决相同网络问题时提供不同的优化策略。此外，个体LLMs训练数据的限制，加上宿主设备潜在的恶意行为，可能导致低置信度或有偏见的响应。为了应对这些挑战，我们提出了一种基于区块链的合作框架，将多个LLMs连接成一个可信的多LLMs网络（MultiLLMN）。该架构允许多个LLMs协同评估和选择最可靠和高质量的复杂网络优化问题解决方案。具体来说，我们首先回顾相关工作，强调现有LLMs在合作和信任方面存在的局限性，突出LLM为基础的系统需要具备可信性的重要性和紧迫性。然后，我们介绍了所提出的可信多LLMs网络框架的工作流程和设计。鉴于5G和6G通信系统中虚假基站（FBS）攻击的严重性以及通过传统建模技术难以应对这类威胁的事实，我们以FBS防御为例，实证验证了我们方法的有效性。最后，我们概述了在这一新兴领域未来研究的有前景的方向。 

---
# Soft Best-of-n Sampling for Model Alignment 

**Title (ZH)**: 软的最佳n抽样模型对齐 

**Authors**: Claudio Mayrink Verdun, Alex Oesterling, Himabindu Lakkaraju, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2505.03156)  

**Abstract**: Best-of-$n$ (BoN) sampling is a practical approach for aligning language model outputs with human preferences without expensive fine-tuning. BoN sampling is performed by generating $n$ responses to a prompt and then selecting the sample that maximizes a reward function. BoN yields high reward values in practice at a distortion cost, as measured by the KL-divergence between the sampled and original distribution. This distortion is coarsely controlled by varying the number of samples: larger $n$ yields a higher reward at a higher distortion cost. We introduce Soft Best-of-$n$ sampling, a generalization of BoN that allows for smooth interpolation between the original distribution and reward-maximizing distribution through a temperature parameter $\lambda$. We establish theoretical guarantees showing that Soft Best-of-$n$ sampling converges sharply to the optimal tilted distribution at a rate of $O(1/n)$ in KL and the expected (relative) reward. For sequences of discrete outputs, we analyze an additive reward model that reveals the fundamental limitations of blockwise sampling. 

**Abstract (ZH)**: Best-of-$n$ (BoN) 抽样是一种实用的方法，用于通过生成$n$个响应并选择最大化奖励函数的样本来使语言模型输出与人类偏好保持一致，而无需昂贵的精细调整。BoN抽样通过生成$n$个响应并选择最大化奖励函数的样本来实现，所得到的奖励值在实际中会伴随着KL散度度量的失真成本。通过改变样本数量来粗略控制这种失真：较大的$n$会产生更高的奖励但伴随更高的失真成本。我们引入了Soft Best-of-$n$抽样，这是一种BoN的推广，通过温度参数$\lambda$可以在原始分布和奖励最大化分布之间实现平滑插值。我们建立了理论保证，证明Soft Best-of-$n$抽样以$O(1/n)$的速率在KL和期望（相对）奖励方面收敛到最优倾斜分布。对于离散输出的序列，我们分析了一个加性奖励模型，揭示了块状抽样的基本局限性。 

---
# Assessing and Enhancing the Robustness of LLM-based Multi-Agent Systems Through Chaos Engineering 

**Title (ZH)**: 通过混沌工程评估并增强基于LLM的多代理系统 robustness 

**Authors**: Joshua Owotogbe  

**Link**: [PDF](https://arxiv.org/pdf/2505.03096)  

**Abstract**: This study explores the application of chaos engineering to enhance the robustness of Large Language Model-Based Multi-Agent Systems (LLM-MAS) in production-like environments under real-world conditions. LLM-MAS can potentially improve a wide range of tasks, from answering questions and generating content to automating customer support and improving decision-making processes. However, LLM-MAS in production or preproduction environments can be vulnerable to emergent errors or disruptions, such as hallucinations, agent failures, and agent communication failures. This study proposes a chaos engineering framework to proactively identify such vulnerabilities in LLM-MAS, assess and build resilience against them, and ensure reliable performance in critical applications. 

**Abstract (ZH)**: 本研究探讨了将混沌工程应用于提高类生产环境中大规模语言模型基于的多智能体系统（LLM-MAS）鲁棒性的应用，以应对实际条件下的潜在错误或中断。LLM-MAS可以从回答问题、生成内容到自动化的客户服务和决策过程改进等多种任务中受益。然而，在生产或准生产环境中，LLM-MAS可能易受幻觉、智能体故障及智能体间通信故障等新兴错误或中断的影响。本研究提出了一种混沌工程框架，旨在前瞻性地识别LLM-MAS中的这些脆弱性，评估并构建对其的抗御能力，以确保其在关键应用中可靠的表现。 

---
# Developing A Framework to Support Human Evaluation of Bias in Generated Free Response Text 

**Title (ZH)**: 开发一个框架以支持对生成自由响应文本中的偏见进行人工评估 

**Authors**: Jennifer Healey, Laurie Byrum, Md Nadeem Akhtar, Surabhi Bhargava, Moumita Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2505.03053)  

**Abstract**: LLM evaluation is challenging even the case of base models. In real world deployments, evaluation is further complicated by the interplay of task specific prompts and experiential context. At scale, bias evaluation is often based on short context, fixed choice benchmarks that can be rapidly evaluated, however, these can lose validity when the LLMs' deployed context differs. Large scale human evaluation is often seen as too intractable and costly. Here we present our journey towards developing a semi-automated bias evaluation framework for free text responses that has human insights at its core. We discuss how we developed an operational definition of bias that helped us automate our pipeline and a methodology for classifying bias beyond multiple choice. We additionally comment on how human evaluation helped us uncover problematic templates in a bias benchmark. 

**Abstract (ZH)**: LLM偏见评估即使对于基础模型也具有挑战性。在实际部署中，任务特定提示和经验性上下文的交互进一步复杂化了评估过程。在大规模部署中，偏见评估通常基于短上下文和固定选择的基准，这些基准可以快速评估，然而，当部署的上下文与LLM不符时，这些基准的有效性会降低。大规模人工评估通常被视为难以实现且成本高昂。我们提出了一个以人类洞察为核心、具有半自动化偏见评估框架的发展历程。我们讨论了我们如何制定可操作的偏见定义，以帮助自动化评估管道，以及如何超越选择题分类偏见的方法论。此外，我们还提到人类评估如何帮助我们发现偏见基准中的问题模板。 

---
# Memorization or Interpolation ? Detecting LLM Memorization through Input Perturbation Analysis 

**Title (ZH)**: 记忆还是内插？通过输入扰动分析检测大模型的记忆现象 

**Authors**: Albérick Euraste Djiré, Abdoul Kader Kaboré, Earl T. Barr, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2505.03019)  

**Abstract**: While Large Language Models (LLMs) achieve remarkable performance through training on massive datasets, they can exhibit concerning behaviors such as verbatim reproduction of training data rather than true generalization. This memorization phenomenon raises significant concerns about data privacy, intellectual property rights, and the reliability of model evaluations. This paper introduces PEARL, a novel approach for detecting memorization in LLMs. PEARL assesses how sensitive an LLM's performance is to input perturbations, enabling memorization detection without requiring access to the model's internals. We investigate how input perturbations affect the consistency of outputs, enabling us to distinguish between true generalization and memorization. Our findings, following extensive experiments on the Pythia open model, provide a robust framework for identifying when the model simply regurgitates learned information. Applied on the GPT 4o models, the PEARL framework not only identified cases of memorization of classic texts from the Bible or common code from HumanEval but also demonstrated that it can provide supporting evidence that some data, such as from the New York Times news articles, were likely part of the training data of a given model. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）通过大规模数据训练实现了出色的性能，但它们可能会表现出诸如机械重复训练数据而不是真正泛化的令人关注的行为。这种记忆现象引发了关于数据隐私、知识产权以及模型评估可靠性的重大关注。本文介绍了PEARL，一种用于检测LLMs记忆现象的新方法。PEARL评估LLM对输入扰动的性能敏感性，能够在无需访问模型内部结构的情况下检测记忆现象。我们研究了输入扰动如何影响输出的一致性，使我们能够区分真正的泛化和记忆现象。我们的研究结果，通过对Pythia开源模型的大量实验，提供了一种 robust 的框架，用于识别模型仅仅是重复已学习信息的情况。将PEARL框架应用于GPT 4o模型后，不仅识别出了对《圣经》经典文本或HumanEval中的通用代码的记忆现象，还展示了该框架可以提供支持证据，证明某些数据，如《纽约时报》的文章内容，很可能包含在某个模型的训练数据中。 

---
# RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale 

**Title (ZH)**: RADLADS: 快速注意力萃取以实现大规模线性注意力解码器 

**Authors**: Daniel Goldstein, Eric Alcaide, Janna Lu, Eugene Cheah  

**Link**: [PDF](https://arxiv.org/pdf/2505.03005)  

**Abstract**: We present Rapid Attention Distillation to Linear Attention Decoders at Scale (RADLADS), a protocol for rapidly converting softmax attention transformers into linear attention decoder models, along with two new RWKV-variant architectures, and models converted from popular Qwen2.5 open source models in 7B, 32B, and 72B sizes. Our conversion process requires only 350-700M tokens, less than 0.005% of the token count used to train the original teacher models. Converting to our 72B linear attention model costs less than \$2,000 USD at today's prices, yet quality at inference remains close to the original transformer. These models achieve state-of-the-art downstream performance across a set of standard benchmarks for linear attention models of their size. We release all our models on HuggingFace under the Apache 2.0 license, with the exception of our 72B models which are also governed by the Qwen License Agreement.
Models at this https URL Training Code at this https URL 

**Abstract (ZH)**: 快速注意力蒸馏到大规模线性注意力解码器（RADLADS）：一种将softmax注意力变换器快速转换为线性注意力解码器模型的协议，及其两种新的RWKV-变体架构和从流行开源Qwen2.5模型转换而来的7B、32B和72B规模的模型。 

---
# When Your Own Output Becomes Your Training Data: Noise-to-Meaning Loops and a Formal RSI Trigger 

**Title (ZH)**: 当你的输出成为你的训练数据：噪声到意义的循环及正式的RSI触发器 

**Authors**: Rintaro Ando  

**Link**: [PDF](https://arxiv.org/pdf/2505.02888)  

**Abstract**: We present Noise-to-Meaning Recursive Self-Improvement (N2M-RSI), a minimal formal model showing that once an AI agent feeds its own outputs back as inputs and crosses an explicit information-integration threshold, its internal complexity will grow without bound under our assumptions. The framework unifies earlier ideas on self-prompting large language models, Gödelian self-reference, and AutoML, yet remains implementation-agnostic. The model furthermore scales naturally to interacting swarms of agents, hinting at super-linear effects once communication among instances is permitted. For safety reasons, we omit system-specific implementation details and release only a brief, model-agnostic toy prototype in Appendix C. 

**Abstract (ZH)**: 噪声到含义递归自我完善（N2M-RSI）：一种最小形式模型 

---
# Unlearning vs. Obfuscation: Are We Truly Removing Knowledge? 

**Title (ZH)**: 卸学习 vs. 模糊化：我们真正删除知识了吗？ 

**Authors**: Guangzhi Sun, Potsawee Manakul, Xiao Zhan, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2505.02884)  

**Abstract**: Unlearning has emerged as a critical capability for large language models (LLMs) to support data privacy, regulatory compliance, and ethical AI deployment. Recent techniques often rely on obfuscation by injecting incorrect or irrelevant information to suppress knowledge. Such methods effectively constitute knowledge addition rather than true removal, often leaving models vulnerable to probing. In this paper, we formally distinguish unlearning from obfuscation and introduce a probing-based evaluation framework to assess whether existing approaches genuinely remove targeted information. Moreover, we propose DF-MCQ, a novel unlearning method that flattens the model predictive distribution over automatically generated multiple-choice questions using KL-divergence, effectively removing knowledge about target individuals and triggering appropriate refusal behaviour. Experimental results demonstrate that DF-MCQ achieves unlearning with over 90% refusal rate and a random choice-level uncertainty that is much higher than obfuscation on probing questions. 

**Abstract (ZH)**: 卸载已成为大型语言模型（LLMs）支持数据隐私、合规性和伦理AI部署的关键能力。近期技术通常依赖于混淆方法，通过注入错误或不相关信息来抑制知识。这些方法实际上构成了知识的增加而非真正的删除，往往使模型仍然容易受到探查。在本文中，我们正式区分了卸载与混淆，并引入了一种基于探查的评估框架，以评估现有方法是否真正删除了目标信息。此外，我们提出了DF-MCQ，这是一种新颖的卸载方法，通过使用KL散度对生成的多项选择题的模型预测分布进行扁平化处理，有效地删除了关于目标个体的知识，并触发适当的拒绝行为。实验结果表明，DF-MCQ的拒绝率达到90%以上，在探查问题上的随机选择级不确定性远高于混淆方法。 

---
# Rewriting Pre-Training Data Boosts LLM Performance in Math and Code 

**Title (ZH)**: 预训练数据重写提升大语言模型在数学和代码上的性能 

**Authors**: Kazuki Fujii, Yukito Tajima, Sakae Mizuki, Hinari Shimada, Taihei Shiotani, Koshiro Saito, Masanari Ohi, Masaki Kawamura, Taishi Nakamura, Takumi Okamoto, Shigeki Ishida, Kakeru Hattori, Youmi Ma, Hiroya Takamura, Rio Yokota, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.02881)  

**Abstract**: The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains. 

**Abstract (ZH)**: 大型语言模型（LLMs）在程序合成和数学推理中的性能从根本上受限于其预训练语料的质量。我们引入了两个开源数据集，通过Llama 3.3社区许可发布，显著提升了LLM的性能，通过系统地重写公开数据。SwallowCode（约161亿词元）通过一个新颖的四阶段管道细化《The-Stack-v2》中的Python片段：语法验证、基于pylint的风格过滤，以及两阶段LLM重写过程，以确保风格一致并transform片段为自包含、算法高效示例。SwallowMath（约23亿词元）通过清除样板代码、恢复上下文和重新格式化解决方案为简洁、分步解释来增强Finemath-4+。在固定50亿词元的预训练预算内，持续使用SwallowCode预训练Llama-3.1-8B在HumanEval和HumanEval+上的pass@1分别提升了17.0%和17.7%，超越了基准模型的代码生成能力。类似地，替换为SwallowMath在GSM8K上的准确率提升了12.4%，在MATH上的准确率提升了7.6%。消融研究证实，每个管道阶段都贡献了增量提升，重写带来了最大的收益。所有数据集、提示和检查点都公开可用，支持可再现研究并推动LLM预训练向专业化领域的进步。 

---
# Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading 

**Title (ZH)**: 从阅读的眼动中解码开放式的信息寻求目标 

**Authors**: Cfir Avraham Hadar, Omer Shubi, Yoav Meiri, Yevgeni Berzak  

**Link**: [PDF](https://arxiv.org/pdf/2505.02872)  

**Abstract**: When reading, we often have specific information that interests us in a text. For example, you might be reading this paper because you are curious about LLMs for eye movements in reading, the experimental design, or perhaps you only care about the question ``but does it work?''. More broadly, in daily life, people approach texts with any number of text-specific goals that guide their reading behavior. In this work, we ask, for the first time, whether open-ended reading goals can be automatically decoded from eye movements in reading. To address this question, we introduce goal classification and goal reconstruction tasks and evaluation frameworks, and use large-scale eye tracking for reading data in English with hundreds of text-specific information seeking tasks. We develop and compare several discriminative and generative multimodal LLMs that combine eye movements and text for goal classification and goal reconstruction. Our experiments show considerable success on both tasks, suggesting that LLMs can extract valuable information about the readers' text-specific goals from eye movements. 

**Abstract (ZH)**: 当阅读时，我们往往对文本中有特定的兴趣信息。例如，你可能因为对阅读中的眼动与LLMs的关系、实验设计感兴趣，或者只关心“有效果吗？”的问题。更广泛地说，在日常生活中，人们带着各种与文本相关的目标阅读，这些目标指引着他们的阅读行为。在本研究中，我们首次探讨是否可以自动解码开放性的阅读目标从眼动中。为此，我们引入了目标分类和目标重建任务及评估框架，并使用包含数百个文本特定信息寻求任务的大量英语阅读眼动追踪数据。我们开发并比较了几种结合眼动与文本的判别性和生成性多模态LLM，用于目标分类和目标重建。我们的实验在这两方面均取得了显著成果，表明LLM可以从眼动中提取有价值的读者文本特定目标信息。 

---
# Accelerating Large Language Model Reasoning via Speculative Search 

**Title (ZH)**: 通过推测性搜索加速大规模语言模型推理 

**Authors**: Zhihai Wang, Jie Wang, Jilai Pan, Xilin Xia, Huiling Zhen, Mingxuan Yuan, Jianye Hao, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02865)  

**Abstract**: Tree-search-based reasoning methods have significantly enhanced the reasoning capability of large language models (LLMs) by facilitating the exploration of multiple intermediate reasoning steps, i.e., thoughts. However, these methods suffer from substantial inference latency, as they have to generate numerous reasoning thoughts, severely limiting LLM applicability. To address this challenge, we propose a novel Speculative Search (SpecSearch) framework that significantly accelerates LLM reasoning by optimizing thought generation. Specifically, SpecSearch utilizes a small model to strategically collaborate with a large model at both thought and token levels, efficiently generating high-quality reasoning thoughts. The major pillar of SpecSearch is a novel quality-preserving rejection mechanism, which effectively filters out thoughts whose quality falls below that of the large model's outputs. Moreover, we show that SpecSearch preserves comparable reasoning quality to the large model. Experiments on both the Qwen and Llama models demonstrate that SpecSearch significantly outperforms state-of-the-art approaches, achieving up to 2.12$\times$ speedup with comparable reasoning quality. 

**Abstract (ZH)**: 基于树搜索的推理方法显著增强了大型语言模型的推理能力，通过促进多种中间推理步骤（即思维）的探索。然而，这些方法面临着显著的推理延迟问题，因为它们需要生成大量的推理思维，极大地限制了大型语言模型的应用。为应对这一挑战，我们提出了一种新颖的猜测搜索（SpecSearch）框架，通过优化思维生成大幅加速大型语言模型的推理。具体而言，SpecSearch 在思维和标记层面利用一个小模型战略性地与大模型合作，高效生成高质量的推理思维。SpecSearch 的主要支柱是一种新颖的质量保留拒绝机制，能够有效过滤掉质量低于大模型输出的思维。此外，我们证明 SpecSearch 保留了与大模型相当的推理质量。实验结果显示，SpecSearch 在 Qwen 和 Llama 模型上均显著优于现有最佳方法，实现了最高达 2.12 倍的加速，同时保持了类似的推理质量。 

---
# Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs 

**Title (ZH)**: 无法只见树木而不见森林：调用启发式和偏差以揭示LLMs的非理性选择 

**Authors**: Haoming Yang, Ke Ma, Xiaojun Jia, Yingfei Sun, Qianqian Xu, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02862)  

**Abstract**: Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）表现出色，但仍容易受到 Jailbreak 攻击，这会危及它们的安全机制。现有研究常依赖于暴力优化或手动设计，未能在真实场景中揭示潜在风险。为应对这一挑战，我们提出了一种新的 Jailbreak 攻击框架 ICRT，该框架受人类认知中的启发和偏差启发。利用简单效应，我们采用认知分解来减少恶意提示的复杂性。同时，利用相关性偏差重组提示，增强语义对齐并有效诱导有害输出。此外，我们引入了一种基于排名的有害性评估指标，该指标通过使用 Elo、HodgeRank 和 Rank Centrality 等排名聚合方法超越传统的成功或失败二元范式，全面量化生成内容的有害性。实验结果表明，我们的方法能一致地绕过主流 LLMs 的安全机制并生成高风险内容，为 Jailbreak 攻击风险提供了见解，并助力更强的安全防御策略。 

---
# Enhancing ML Model Interpretability: Leveraging Fine-Tuned Large Language Models for Better Understanding of AI 

**Title (ZH)**: 提升机器学习模型可解释性：利用 fine-tuned 大型语言模型以更好地理解 AI 

**Authors**: Jonas Bokstaller, Julia Altheimer, Julian Dormehl, Alina Buss, Jasper Wiltfang, Johannes Schneider, Maximilian Röglinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.02859)  

**Abstract**: Across various sectors applications of eXplainableAI (XAI) gained momentum as the increasing black-boxedness of prevailing Machine Learning (ML) models became apparent. In parallel, Large Language Models (LLMs) significantly developed in their abilities to understand human language and complex patterns. By combining both, this paper presents a novel reference architecture for the interpretation of XAI through an interactive chatbot powered by a fine-tuned LLM. We instantiate the reference architecture in the context of State-of-Health (SoH) prediction for batteries and validate its design in multiple evaluation and demonstration rounds. The evaluation indicates that the implemented prototype enhances the human interpretability of ML, especially for users with less experience with XAI. 

**Abstract (ZH)**: 跨领域应用的可解释人工智能(XAI)随着现有机器学习(ML)模型的黑盒性质日益明显而逐渐获得动力。与此同时，大规模语言模型(LLMs)在理解和识别复杂模式方面的能力有了显著提升。通过结合两者，本文提出了一种新的参考架构，利用微调后的LLM驱动交互式聊天机器人来解释XAI。我们将该参考架构应用于电池的状态健康(SoH)预测，并在多次评估和演示环节中验证了其设计。评估结果显示，实现的原型提高了机器学习的人类可解释性，尤其是对于XAI经验较少的用户。 

---
# 30DayGen: Leveraging LLMs to Create a Content Corpus for Habit Formation 

**Title (ZH)**: 30DayGen: 利用大语言模型创建习惯形成内容 corpus 

**Authors**: Franklin Zhang, Sonya Zhang, Alon Halevy  

**Link**: [PDF](https://arxiv.org/pdf/2505.02851)  

**Abstract**: In this paper, we present 30 Day Me, a habit formation application that leverages Large Language Models (LLMs) to help users break down their goals into manageable, actionable steps and track their progress. Central to the app is the 30DAYGEN system, which generates 3,531 unique 30-day challenges sourced from over 15K webpages, and enables runtime search of challenge ideas aligned with user-defined goals. We showcase how LLMs can be harnessed to rapidly construct domain specific content corpora for behavioral and educational purposes, and propose a practical pipeline that incorporates effective LLM enhanced approaches for content generation and semantic deduplication. 

**Abstract (ZH)**: 本文介绍了30 Day Me，这一利用大规模语言模型（LLMs）帮助用户将目标分解为可管理的行动步骤并跟踪进度的应用程序。应用程序的核心是30DAYGEN系统，该系统从超过15000个网页中生成了3531个独特的30天挑战，并能够根据用户定义的目标进行运行时挑战构思搜索。我们展示了如何利用LLMs快速构建用于行为和教育目的的专业领域内容集合，并提出了一个结合有效LLM增强方法的内容生成和语义去重的实际管道。 

---
# Harnessing Structured Knowledge: A Concept Map-Based Approach for High-Quality Multiple Choice Question Generation with Effective Distractors 

**Title (ZH)**: 基于概念图的方法：一种用于生成高质量多项选择题的有效干扰项的概念图基方法 

**Authors**: Nicy Scaria, Silvester John Joseph Kennedy, Diksha Seth, Ananya Thakur, Deepak Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2505.02850)  

**Abstract**: Generating high-quality MCQs, especially those targeting diverse cognitive levels and incorporating common misconceptions into distractor design, is time-consuming and expertise-intensive, making manual creation impractical at scale. Current automated approaches typically generate questions at lower cognitive levels and fail to incorporate domain-specific misconceptions. This paper presents a hierarchical concept map-based framework that provides structured knowledge to guide LLMs in generating MCQs with distractors. We chose high-school physics as our test domain and began by developing a hierarchical concept map covering major Physics topics and their interconnections with an efficient database design. Next, through an automated pipeline, topic-relevant sections of these concept maps are retrieved to serve as a structured context for the LLM to generate questions and distractors that specifically target common misconceptions. Lastly, an automated validation is completed to ensure that the generated MCQs meet the requirements provided. We evaluate our framework against two baseline approaches: a base LLM and a RAG-based generation. We conducted expert evaluations and student assessments of the generated MCQs. Expert evaluation shows that our method significantly outperforms the baseline approaches, achieving a success rate of 75.20% in meeting all quality criteria compared to approximately 37% for both baseline methods. Student assessment data reveal that our concept map-driven approach achieved a significantly lower guess success rate of 28.05% compared to 37.10% for the baselines, indicating a more effective assessment of conceptual understanding. The results demonstrate that our concept map-based approach enables robust assessment across cognitive levels and instant identification of conceptual gaps, facilitating faster feedback loops and targeted interventions at scale. 

**Abstract (ZH)**: 基于层次概念图的生成高质量选择题框架：指导LLM生成针对认知层次和常见误解的选择题 

---
# Enhancing tutoring systems by leveraging tailored promptings and domain knowledge with Large Language Models 

**Title (ZH)**: 利用大型语言模型定制提示和领域知识增强辅导系统 

**Authors**: Mohsen Balavar, Wenli Yang, David Herbert, Soonja Yeom  

**Link**: [PDF](https://arxiv.org/pdf/2505.02849)  

**Abstract**: Recent advancements in artificial intelligence (AI) and machine learning have reignited interest in their impact on Computer-based Learning (CBL). AI-driven tools like ChatGPT and Intelligent Tutoring Systems (ITS) have enhanced learning experiences through personalisation and flexibility. ITSs can adapt to individual learning needs and provide customised feedback based on a student's performance, cognitive state, and learning path. Despite these advances, challenges remain in accommodating diverse learning styles and delivering real-time, context-aware feedback. Our research aims to address these gaps by integrating skill-aligned feedback via Retrieval Augmented Generation (RAG) into prompt engineering for Large Language Models (LLMs) and developing an application to enhance learning through personalised tutoring in a computer science programming context. The pilot study evaluated a proposed system using three quantitative metrics: readability score, response time, and feedback depth, across three programming tasks of varying complexity. The system successfully sorted simulated students into three skill-level categories and provided context-aware feedback. This targeted approach demonstrated better effectiveness and adaptability compared to general methods. 

**Abstract (ZH)**: 近期人工智能和机器学习的进展重新引发了其对基于计算机的学习（CBL）影响的兴趣。通过个性化和灵活性提升学习体验的人工智能驱动工具如ChatGPT和智能辅导系统（ITS）已得到增强。ITS可以根据学生的表现、认知状态和学习路径适应个性化需求并提供定制反馈。尽管取得了这些进展，但在适应多种学习风格和提供实时、情境感知反馈方面仍存在挑战。我们的研究旨在通过将技能对齐反馈结合检索增强生成（RAG）技术融入大语言模型（LLMs）的提示工程，并开发一个应用程序，以在计算机科学编程背景下通过个性化辅导提升学习效果来弥补这些差距。试点研究使用三种定量指标：可读性分数、响应时间和反馈深度，评估了一个拟议系统的性能，涉及三个不同复杂程度的编程任务。该系统成功地将模拟学生分类为三个技能级别，并提供了情境感知反馈。这种有针对性的方法在效果和适应性方面优于通用方法。 

---
# Aligning Large Language Models with Healthcare Stakeholders: A Pathway to Trustworthy AI Integration 

**Title (ZH)**: 将大型语言模型与医疗健康利益相关者对齐：通往可信赖AI集成的道路 

**Authors**: Kexin Ding, Mu Zhou, Akshay Chaudhari, Shaoting Zhang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2505.02848)  

**Abstract**: The wide exploration of large language models (LLMs) raises the awareness of alignment between healthcare stakeholder preferences and model outputs. This alignment becomes a crucial foundation to empower the healthcare workflow effectively, safely, and responsibly. Yet the varying behaviors of LLMs may not always match with healthcare stakeholders' knowledge, demands, and values. To enable a human-AI alignment, healthcare stakeholders will need to perform essential roles in guiding and enhancing the performance of LLMs. Human professionals must participate in the entire life cycle of adopting LLM in healthcare, including training data curation, model training, and inference. In this review, we discuss the approaches, tools, and applications of alignments between healthcare stakeholders and LLMs. We demonstrate that LLMs can better follow human values by properly enhancing healthcare knowledge integration, task understanding, and human guidance. We provide outlooks on enhancing the alignment between humans and LLMs to build trustworthy real-world healthcare applications. 

**Abstract (ZH)**: 大型语言模型在医疗健康领域的广泛应用提高了对医疗健康利益相关者偏好与模型输出之间一致性的认识。这种一致性成为有效、安全和负责任地赋能医疗工作流程的基础。然而，大型语言模型的行为可能并不始终与医疗健康利益相关者的知识、需求和价值观相符。为了实现人机一致，医疗健康利益相关者需要在引导和提升大型语言模型性能方面发挥关键作用。专业人员必须参与将大型语言模型应用于医疗保健的整个生命周期，包括训练数据的整理、模型训练和推理。在本文综述中，我们探讨了医疗健康利益相关者与大型语言模型之间一致性的方法、工具和应用。我们表明，通过适当增强医疗健康知识整合、任务理解和人为引导，大型语言模型能够更好地体现人类价值观。我们还展望了增强人类与大型语言模型之间一致性的前景，以构建可信赖的医疗健康实际应用。 

---
# Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in Large Language Models 

**Title (ZH)**: 有感知能力的代理作为法官：评估大型语言模型的高阶社交认知 

**Authors**: Bang Zhang, Ruotian Ma, Qingxuan Jiang, Peisong Wang, Jiaqi Chen, Zheng Xie, Xingyu Chen, Yue Wang, Fanghua Ye, Jian Li, Yifan Yang, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02847)  

**Abstract**: Assessing how well a large language model (LLM) understands human, rather than merely text, remains an open challenge. To bridge the gap, we introduce Sentient Agent as a Judge (SAGE), an automated evaluation framework that measures an LLM's higher-order social cognition. SAGE instantiates a Sentient Agent that simulates human-like emotional changes and inner thoughts during interaction, providing a more realistic evaluation of the tested model in multi-turn conversations. At every turn, the agent reasons about (i) how its emotion changes, (ii) how it feels, and (iii) how it should reply, yielding a numerical emotion trajectory and interpretable inner thoughts. Experiments on 100 supportive-dialogue scenarios show that the final Sentient emotion score correlates strongly with Barrett-Lennard Relationship Inventory (BLRI) ratings and utterance-level empathy metrics, validating psychological fidelity. We also build a public Sentient Leaderboard covering 18 commercial and open-source models that uncovers substantial gaps (up to 4x) between frontier systems (GPT-4o-Latest, Gemini2.5-Pro) and earlier baselines, gaps not reflected in conventional leaderboards (e.g., Arena). SAGE thus provides a principled, scalable and interpretable tool for tracking progress toward genuinely empathetic and socially adept language agents. 

**Abstract (ZH)**: 评估大型语言模型在理解人类而非仅仅文本方面的能力依然是一项开放的挑战。为了弥合这一差距，我们引入了觉知代理作为评判者（SAGE），这是一种自动化评估框架，用于测量大型语言模型的高层次社会认知能力。SAGE 实现了一个模拟人类情感变化和交互中内心想法的觉知代理，从而为多轮对话中的测试模型提供更为现实的评估。在每次交互中，代理会推理其情感的变化、感觉以及应如何回应，产生一个数值化的情感轨迹和可解释的内心想法。在100个支持性对话场景上的实验表明，最终的觉知情感得分与Barrett-Lennard关系量表（BLRI）评分和语句层面的共情指标高度相关，验证了心理真实性的存在。我们还构建了一个公开的觉知排行榜，涵盖了18个商用和开源模型，揭示了前沿系统（如GPT-4o-Latest、Gemini2.5-Pro）与早期基线之间存在的显著差距（最多4倍），这些差距在传统的排行榜（如Arena）中并未体现。因此，SAGE 提供了一个原则性、可扩展且可解释的工具，用于追踪朝着真正具有共情和社会适应语言代理方向的进步。 

---
