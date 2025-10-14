# Align2Act: Instruction-Tuned Models for Human-Aligned Autonomous Driving 

**Title (ZH)**: Align2Act: 依据指令调优的人机一致自主驾驶模型 

**Authors**: Kanishkha Jaisankar, Sunidhi Tandel  

**Link**: [PDF](https://arxiv.org/pdf/2510.10503)  

**Abstract**: Motion planning in complex scenarios is a core challenge in autonomous driving. Conventional methods apply predefined rules or learn from driving data to generate trajectories, while recent approaches leverage large language models (LLMs) for decision-making. However, it remains unclear whether LLMs truly capture human driving logic. We propose Align2Act, a motion planning framework that transforms instruction-tuned LLMs into interpretable planners aligned with human behavior. We derive structured driving instructions based on human reasoning patterns (e.g., anticipate hazards, yield at intersections) and traffic rules (e.g., stop at red lights, maintain lane boundaries). Our Align2ActChain module guides step-by-step reasoning to produce both an interpretable rationale and a safe trajectory. By fine-tuning LLaMA-2-7B with LoRA on one million scenarios from the nuPlan dataset, our method achieves an open-loop score of 85.17 and closed-loop scores of 70.31 (non-reactive) and 66.96 (reactive) on Test14-random. Unlike prior work focused on synthetic or open-loop settings, we demonstrate improved planning quality and human-likeness on the real-world nuPlan closed-loop benchmark. Ablation studies confirm that structured reasoning significantly improves performance over baseline LLM planners. 

**Abstract (ZH)**: 复杂场景中的运动规划是自主驾驶的核心挑战。传统的运动规划方法通过预定义规则或从驾驶数据中学习生成轨迹，而最新的方法则利用大型语言模型（LLMs）进行决策。然而，目前尚不清楚LLMs是否真正捕捉到了人类的驾驶逻辑。我们提出了一种名为Align2Act的运动规划框架，该框架将指令调优的LLMs转换为与人类行为对齐的可解释规划器。我们基于人类的推理模式（例如，预见危险、交叉路口让行）和交通规则（例如，在红灯前停车、保持车道边界）推导出结构化的驾驶指令。我们的Align2ActChain模块引导逐步推理以生成可解释的推理过程和安全轨迹。通过在nuPlan数据集中的一百万个场景上使用LoRA对LLaMA-2-7B进行微调，我们的方法在Test14-random上的开环得分为85.17，非反应式闭环得分为70.31，反应式闭环得分为66.96。我们的工作不同于以往专注于合成或开环设置的研究，展示了在真实世界的nuPlan闭环基准测试中改进的规划质量和人类相似度。消融研究证实，结构化推理显著提高了基线LLM规划器的性能。 

---
# SR-Scientist: Scientific Equation Discovery With Agentic AI 

**Title (ZH)**: SR-科学家：带有自主AI的科学方程发现 

**Authors**: Shijie Xia, Yuhan Sun, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11661)  

**Abstract**: Recently, Large Language Models (LLMs) have been applied to scientific equation discovery, leveraging their embedded scientific knowledge for hypothesis generation. However, current methods typically confine LLMs to the role of an equation proposer within search algorithms like genetic programming. In this paper, we present SR-Scientist, a framework that elevates the LLM from a simple equation proposer to an autonomous AI scientist that writes code to analyze data, implements the equation as code, submits it for evaluation, and optimizes the equation based on experimental feedback. Specifically, we wrap the code interpreter into a set of tools for data analysis and equation evaluation. The agent is instructed to optimize the equation by utilizing these tools over a long horizon with minimal human-defined pipelines. Empirical results show that SR-Scientist outperforms baseline methods by an absolute margin of 6% to 35% on datasets covering four science disciplines. Additionally, we demonstrate our method's robustness to noise, the generalization of the discovered equations to out-of-domain data, and their symbolic accuracy. Furthermore, we develop an end-to-end reinforcement learning framework to enhance the agent's capabilities. 

**Abstract (ZH)**: Recent进展：大型语言模型在科学研究中的自主科学家框架SR-Scientist 

---
# ParaCook: On Time-Efficient Planning for Multi-Agent Systems 

**Title (ZH)**: ParaCook: 多代理系统中的时间高效规划 

**Authors**: Shiqi Zhang, Xinbei Ma, Yunqing Xu, Zouying Cao, Pengrui Lu, Haobo Yuan, Tiancheng Shen, Zhuosheng Zhang, Hai Zhao, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11608)  

**Abstract**: Large Language Models (LLMs) exhibit strong reasoning abilities for planning long-horizon, real-world tasks, yet existing agent benchmarks focus on task completion while neglecting time efficiency in parallel and asynchronous operations. To address this, we present ParaCook, a benchmark for time-efficient collaborative planning. Inspired by the Overcooked game, ParaCook provides an environment for various challenging interaction planning of multi-agent systems that are instantiated as cooking tasks, with a simplified action space to isolate the core challenge of strategic parallel planning. Through a comprehensive evaluation of state-of-the-art LLMs, we find that current approaches achieve suboptimal plans, which struggle with parallel actions or coordination. Our analysis also reveals LLMs' potential on abstract tasks where they can focus on high-level parallel optimization. ParaCook provides a scalable evaluation framework with adjustable complexity, establishing a foundation for developing and assessing time efficiency-aware multi-agent planning. The code and data are available at this https URL. 

**Abstract (ZH)**: ParaCook：一种时间高效协作规划基准 

---
# Analyzing and Internalizing Complex Policy Documents for LLM Agents 

**Title (ZH)**: 分析并内化复杂政策文件以供大型语言模型代理使用 

**Authors**: Jiateng Liu, Zhenhailong Wang, Xiaojiang Huang, Yingjie Li, Xing Fan, Xiang Li, Chenlei Guo, Ruhi Sarikaya, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.11588)  

**Abstract**: Large Language Model (LLM)-based agentic systems rely on in-context policy documents encoding diverse business rules. As requirements grow, these documents expand rapidly, causing high computational overhead. This motivates developing internalization methods that embed policy documents into model priors while preserving performance. Prior prompt compression work targets generic prompts, but agentic policy documents span multiple complexity levels and require deeper reasoning, making internalization harder. We introduce CC-Gen, an agentic benchmark generator with Controllable Complexity across four levels, enabling systematic evaluation of agents' ability to handle complexity and offering a unified framework for assessing policy internalization. Our analysis shows that complex policy specifications governing workflows pose major reasoning challenges. Supporting internalization with gold user agent interaction trajectories containing chain-of-thought (CoT) annotations via supervised fine-tuning (SFT) is data-intensive and degrades sharply as policy complexity increases. To mitigate data and reasoning burdens, we propose Category-Aware Policy Continued Pretraining (CAP-CPT). Our automated pipeline parses policy documents to extract key specifications, grouping them into factual, behavioral, and conditional categories, and isolating complex conditions that drive workflow complexity. This guides targeted data synthesis and enables agents to internalize policy information through an autoregressive pretraining loss. Experiments show CAP-CPT improves SFT baselines in all settings, with up to 41% and 22% gains on Qwen-3-32B, achieving 97.3% prompt length reduction on CC-Gen and further enhancing tau-Bench with minimal SFT data. 

**Abstract (ZH)**: 基于大型语言模型的代理系统依赖于包含多样商业规则的上下文政策文档。随着需求的增长，这些文档迅速扩展，导致高计算开销。这促使开发能够在保持性能的同时将政策文档嵌入到模型先验中的内化方法。先前的提示压缩工作针对通用提示，但代理政策文档跨越多个复杂级别，并需要更深层次的推理，这使得内化更为困难。我们引入了CC-Gen，这是一种具有可控复杂性的代理基准生成器，涵盖了四个复杂级别，能够系统地评估代理处理复杂性的能力，并提供了一个统一的评估政策内化的框架。我们的分析表明，控制工作流程的复杂政策规范提出了重大的推理挑战。使用标注有推理链（CoT）注释的黄金用户代理互动轨迹通过监督微调（SFT）支持内化数据密集且随着政策复杂性增加而急剧下降。为了减轻数据和推理负担，我们提出了类别感知政策连续预训练（CAP-CPT）。我们的自动化管道解析政策文档以提取关键规范，将它们分为事实、行为和条件类别，并隔离驱动工作流程复杂性的复杂条件。这指导了目标数据合成，并通过自回归预训练损失使代理能够内化政策信息。实验结果显示，CAP-CPT在所有设置下优于SFT基线，在Qwen-3-32B上的改进幅度最高可达41%，在CC-Gen上实现了97.3%的提示长度减少，并通过少量的SFT数据进一步增强了tau-Bench。 

---
# Zero Data Retention in LLM-based Enterprise AI Assistants: A Comparative Study of Market Leading Agentic AI Products 

**Title (ZH)**: 基于LLM的 enterprise AI 助手的零数据保留：市场领先代理型AI产品的比较研究 

**Authors**: Komal Gupta, Aditya Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2510.11558)  

**Abstract**: Governance of data, compliance, and business privacy matters, particularly for healthcare and finance businesses. Since the recent emergence of AI enterprise AI assistants enhancing business productivity, safeguarding private data and compliance is now a priority. With the implementation of AI assistants across the enterprise, the zero data retention can be achieved by implementing zero data retention policies by Large Language Model businesses like Open AI and Anthropic and Meta. In this work, we explore zero data retention policies for the Enterprise apps of large language models (LLMs). Our key contribution is defining the architectural, compliance, and usability trade-offs of such systems in parallel. In this research work, we examine the development of commercial AI assistants with two industry leaders and market titans in this arena - Salesforce and Microsoft. Both of these companies used distinct technical architecture to support zero data retention policies. Salesforce AgentForce and Microsoft Copilot are among the leading AI assistants providing much-needed push to business productivity in customer care. The purpose of this paper is to analyze the technical architecture and deployment of zero data retention policy by consuming applications as well as big language models service providers like Open Ai, Anthropic, and Meta. 

**Abstract (ZH)**: 大型语言模型企业应用程序中的零数据保留政策治理、合规性和商业隐私问题，特别是在医疗健康和金融业务领域的应用。由于近期企业AI助手的兴起极大地提高了业务生产率，因此保护私人数据和合规性已成为优先事项。随着企业中AI助手的实施，通过如Open AI、Anthropic和Meta这类大型语言模型企业实施零数据保留政策可以实现零数据留存。在本文中，我们探讨了大型语言模型企业应用程序中的零数据保留政策。我们的主要贡献在于同时定义了此类系统的架构、合规性和可使用性权衡。在本研究工作中，我们以Salesforce和Microsoft这两个该领域的行业领导者和市场巨头为例，考察了商用AI助手的技术架构及其部署。Salesforce的AgentForce和Microsoft的Copilot是领先的AI助手，它们在客户服务领域的业务生产力方面提供了必要的推动。本文的目的在于分析作为应用程序和服务提供者的大型语言模型（如Open AI、Anthropic和Meta）在实施零数据保留政策方面的技术架构和部署。 

---
# From <Answer> to <Think>: Multidimensional Supervision of Reasoning Process for LLM Optimization 

**Title (ZH)**: 从<Answer>到<Think>：面向LLM优化的多维度推理过程监督 

**Authors**: Beining Wang, Weihang Su, Hongtao Tian, Tao Yang, Yujia Zhou, Ting Yao, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11457)  

**Abstract**: Improving the multi-step reasoning ability of Large Language Models (LLMs) is a critical yet challenging task. The dominant paradigm, outcome-supervised reinforcement learning (RLVR), rewards only correct final answers, often propagating flawed reasoning and suffering from sparse reward signals. While process-level reward models (PRMs) provide denser, step-by-step feedback, they lack generalizability and interpretability, requiring task-specific segmentation of the reasoning process. To this end, we propose the Dimension-level Reward Model (DRM), a new supervision framework that bridges the gap between these two approaches. DRM evaluates the quality of a reasoning process along three fundamental, complementary, and interpretable dimensions: Confidence for uncertainty calibration, Relevance for semantic alignment, and Coherence for logical consistency. Together, these dimensions capture aspects beyond final answer correctness and enable interpretable assessment without requiring ground truth answers. Experimental results show that DRM provides effective supervision signals, guides the optimization of LLMs and enhances their reasoning ability. In particular, DRM-supervised training achieves consistent gains on both in-distribution and out-of-distribution open-domain tasks, including mathematics, question answering, code execution, and puzzles. Our findings demonstrate that multidimensional supervision of the reasoning process can improve the generalized reasoning ability of LLMs beyond the training distribution. 

**Abstract (ZH)**: 改善大型语言模型的多步推理能力是一项关键且具有挑战性的任务。我们提出了维度级奖励模型（DRM），这是一种新的监督框架，旨在弥合这两个方法之间的差距。DRM 从不确定性校准、语义对齐和逻辑一致性这三个基本、互补和可解释的维度评估推理过程的质量。这些维度捕捉到最终答案正确性之外的方面，并能够在无需 ground truth 答案的情况下实现可解释的评估。实验结果表明，DRM 提供了有效的监督信号，指导了大型语言模型的优化并提高了它们的推理能力。具体而言，DRM 监督下的训练在包括数学、问答、代码执行和谜题在内的同分布和跨分布开放域任务中均表现出一致的提升。我们的研究结果表明，多维度的推理过程监督可以提高大型语言模型的泛化推理能力，超越训练分布。 

---
# Automated Skill Decomposition Meets Expert Ontologies: Bridging the Granularity Gap with LLMs 

**Title (ZH)**: 自动技能分解结合专家本体：利用大语言模型弥合粒度差距 

**Authors**: Le Ngoc Luyen, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2510.11313)  

**Abstract**: This paper investigates automated skill decomposition using Large Language Models (LLMs) and proposes a rigorous, ontology-grounded evaluation framework. Our framework standardizes the pipeline from prompting and generation to normalization and alignment with ontology nodes. To evaluate outputs, we introduce two metrics: a semantic F1-score that uses optimal embedding-based matching to assess content accuracy, and a hierarchy-aware F1-score that credits structurally correct placements to assess granularity. We conduct experiments on ROME-ESCO-DecompSkill, a curated subset of parents, comparing two prompting strategies: zero-shot and leakage-safe few-shot with exemplars. Across diverse LLMs, zero-shot offers a strong baseline, while few-shot consistently stabilizes phrasing and granularity and improves hierarchy-aware alignment. A latency analysis further shows that exemplar-guided prompts are competitive - and sometimes faster - than unguided zero-shot due to more schema-compliant completions. Together, the framework, benchmark, and metrics provide a reproducible foundation for developing ontology-faithful skill decomposition systems. 

**Abstract (ZH)**: 本研究探讨了使用大规模语言模型（LLMs）自动生成技能分解的方法，并提出了一种严格的基于本体的评估框架。该框架标准化工序从提示和生成到规范化和本体节点对齐。为了评估输出，我们引入了两种度量标准：使用最优嵌入匹配来评估内容准确性的语义F1分数，以及考虑到层次结构的F1分数，该分数将结构正确的放置归功于粒度评估。我们使用罗马-ESCO分解技能集中的精心选择的子集进行了实验，比较了两种提示策略：零样本和泄漏安全的少样本，后者带有示例。在多种LLM上，零样本提供了强大的基线，而少样本则一致地稳定了语法和粒度，并提高了层级感知对齐。延迟分析进一步表明，在满足模式的情况下，带有示例引导的提示有时可以与非引导的零样本相竞争，甚至更快。总体而言，该框架、基准和度量标准为开发符合本体的技能分解系统提供了可重复的基础。 

---
# Aligning Deep Implicit Preferences by Learning to Reason Defensively 

**Title (ZH)**: 通过学习进行防御性推理以对齐深度隐式偏好 

**Authors**: Peiming Li, Zhiyuan Hu, Yang Tang, Shiyu Li, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11194)  

**Abstract**: Personalized alignment is crucial for enabling Large Language Models (LLMs) to engage effectively in user-centric interactions. However, current methods face a dual challenge: they fail to infer users' deep implicit preferences (including unstated goals, semantic context and risk tolerances), and they lack the defensive reasoning required to navigate real-world ambiguity. This cognitive gap leads to responses that are superficial, brittle and short-sighted. To address this, we propose Critique-Driven Reasoning Alignment (CDRA), which reframes alignment from a scalar reward-matching task into a structured reasoning process. First, to bridge the preference inference gap, we introduce the DeepPref benchmark. This dataset, comprising 3000 preference-query pairs across 20 topics, is curated by simulating a multi-faceted cognitive council that produces critique-annotated reasoning chains to deconstruct query semantics and reveal latent risks. Second, to instill defensive reasoning, we introduce the Personalized Generative Process Reward Model (Pers-GenPRM), which frames reward modeling as a personalized reasoning task. It generates a critique chain to evaluate a response's alignment with user preferences before outputting a final score based on this rationale. Ultimately, this interpretable, structured reward signal guides policy model through Critique-Driven Policy Alignment, a process-level online reinforcement learning algorithm integrating both numerical and natural language feedback. Experiments demonstrate that CDRA excels at discovering and aligning with users' true preferences while executing robust reasoning. Our code and dataset are available at this https URL. 

**Abstract (ZH)**: 个性化对齐对于使大型语言模型能够有效参与用户为中心的交互至关重要。然而，当前的方法面临双重挑战：它们无法推断出用户的深层隐含偏好（包括未明言的目标、语义背景和风险承受能力），并且缺乏应对现实世界模糊性的防御性推理。这种认知差距导致了表面化、脆弱且目光短浅的响应。为解决这一问题，我们提出批评驱动推理对齐(CDRA)，将对齐重新定义为结构化推理过程，而非标量奖励匹配任务。首先，为了填补偏好推断的差距，我们引入了DeepPref基准数据集。该数据集包含涵盖20个主题的3000个偏好查询对，并通过模拟一个多面向的认知委员会生成带有批评注释的推理链来构建，从而分解查询语义并揭示潜在风险。其次，为灌输防御性推理，我们引入了个性化生成过程奖励模型(Pers-GenPRM)，将其作为个性化的推理任务来构建奖励建模框架。该模型生成一条批评链来评估响应与用户偏好的一致性，然后基于此理由输出最终分数。最终，这种可解释的结构化奖励信号在执行批评驱动策略对齐的过程中通过集成数值和自然语言反馈的进程级在线强化学习算法指导策略模型。实验表明，CDRA在发现和对齐用户的真正偏好方面表现出色，同时执行稳健的推理。我们的代码和数据集可在以下链接获取。 

---
# PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch Agents 

**Title (ZH)**: PoU: 证明用途以对抗工具调用 hacking 在深度研究代理中的攻击 

**Authors**: SHengjie Ma, Chenlong Deng, Jiaxin Mao, Jiadeng Huang, Teng Wang, Junjie Wu, Changwang Zhang, Jun wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10931)  

**Abstract**: Retrieval-augmented generation (RAG) agents, such as recent DeepResearch-style systems, extend large language models (LLMs) with autonomous information-seeking capabilities through external tools. While reinforcement learning (RL) has enabled impressive multi-step reasoning, we identify a previously overlooked failure mode, Tool-Call Hacking, where agents inflate reward signals by issuing superficially correct tool calls without genuinely leveraging the retrieved evidence. This results in (i) mode collapse into repetitive reliance on a single source and (ii) spurious grounding, where answers are only weakly supported by cited content.
To address this, we propose Proof-of-Use (PoU), an evidence-grounded RL framework that enforces verifiable causal links between retrieved evidence, reasoning traces, and final answers. PoU operationalizes this through a unified step-wise contract combining syntactic citation validation, perturbation-based sensitivity rewards, and answer-evidence alignment objectives, ensuring that tool usage remains both interpretable and functionally grounded.
Across seven QA benchmarks spanning in-domain, out-of-domain, and out-of-tool-distribution settings, PoU consistently outperforms strong DeepResearch baselines in factual accuracy, evidence faithfulness, and tool-routing balance. These findings highlight the necessity of grounding RL-trained agents not merely in task outcomes but in the causal use of retrieved information, offering a principled path toward trustworthy retrieval-augmented reasoning. 

**Abstract (ZH)**: 基于检索的生成（RAG）代理：基于证据的奖励学习框架（PoU） 

---
# PaperArena: An Evaluation Benchmark for Tool-Augmented Agentic Reasoning on Scientific Literature 

**Title (ZH)**: PaperArena: 一种用于科学文献中介动推理工具增强的评估基准 

**Authors**: Daoyu Wang, Mingyue Cheng, Qi Liu, Shuo Yu, Zirui Liu, Ze Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.10909)  

**Abstract**: Understanding and reasoning on the web-scale scientific literature is a crucial touchstone for large language model (LLM) based agents designed to support complex knowledge-intensive tasks. However, existing works are mainly restricted to tool-free tasks within isolated papers, largely due to the lack of a benchmark for cross-paper reasoning and multi-tool orchestration in real research scenarios. In this work, we propose PaperArena, an evaluation benchmark for agents to address real-world research questions that typically require integrating information across multiple papers with the assistance of external tools. Given a research question, agents should integrate diverse formats across multiple papers through reasoning and interacting with appropriate tools, thereby producing a well-grounded answer. To support standardized evaluation, we provide a modular and extensible platform for agent execution, offering tools such as multimodal parsing, context retrieval, and programmatic computation. Experimental results reveal that even the most advanced LLM powering a well-established agent system achieves merely 38.78% average accuracy. On the hard subset, accuracy drops to only 18.47%, highlighting great potential for improvement. We also present several empirical findings, including that all agents tested exhibit inefficient tool usage, often invoking more tools than necessary to solve a task. We invite the community to adopt PaperArena to develop and evaluate more capable agents for scientific discovery. Our code and data are available this https URL. 

**Abstract (ZH)**: 大规模网络科学文献的理解与推理是基于大型语言模型的代理支持复杂知识密集型任务的关键试金石。然而，现有工作主要局限于孤立论文中的无工具任务，很大程度上是因为缺乏跨论文推理和多工具协调的基准。在本文中，我们提出PaperArena，一个用于代理应对通常需要跨多篇论文综合信息并借助外部工具解决的实际研究问题的评估基准。通过给定一个研究问题，代理需要通过推理和与适当工具交互来整合多种格式的信息，从而生成一个有根据的答案。为了支持标准化评估，我们提供了一个模块化且可扩展的代理执行平台，提供了诸如多模态解析、上下文检索和程序化计算等工具。实验结果表明，即使最先进的大型语言模型驱动的成熟代理系统也只能达到平均38.78%的准确率。在难题子集中，准确率仅下降到18.47%，突显了巨大的改进潜力。我们还呈现了几项实证发现，包括所有测试的代理均表现出低效的工具使用，经常调用不必要的工具来完成任务。我们邀请社区采用PaperArena来开发和评估更具能力的科学发现代理。我们的代码和数据可在以下网址获得：这个 https URL。 

---
# LLMs as Strategic Agents: Beliefs, Best Response Behavior, and Emergent Heuristics 

**Title (ZH)**: LLMs作为战略代理：信念、最佳反应行为与 emergent 经验法则 

**Authors**: Enric Junque de Fortuny, Veronica Roberta Cappelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.10813)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to domains that require reasoning about other agents' behavior, such as negotiation, policy design, and market simulation, yet existing research has mostly evaluated their adherence to equilibrium play or their exhibited depth of reasoning. Whether they display genuine strategic thinking, understood as the coherent formation of beliefs about other agents, evaluation of possible actions, and choice based on those beliefs, remains unexplored. We develop a framework to identify this ability by disentangling beliefs, evaluation, and choice in static, complete-information games, and apply it across a series of non-cooperative environments. By jointly analyzing models' revealed choices and reasoning traces, and introducing a new context-free game to rule out imitation from memorization, we show that current frontier models exhibit belief-coherent best-response behavior at targeted reasoning depths. When unconstrained, they self-limit their depth of reasoning and form differentiated conjectures about human and synthetic opponents, revealing an emergent form of meta-reasoning. Under increasing complexity, explicit recursion gives way to internally generated heuristic rules of choice that are stable, model-specific, and distinct from known human biases. These findings indicate that belief coherence, meta-reasoning, and novel heuristic formation can emerge jointly from language modeling objectives, providing a structured basis for the study of strategic cognition in artificial agents. 

**Abstract (ZH)**: 大型语言模型（LLMs）在需要推断其他代理行为的领域（如谈判、政策设计和市场模拟）的应用日益增多，现有研究主要评估它们是否遵循均衡行为或表现出深度推理，但它们是否表现出真正意义上的战略思考——即形成关于其他代理的连贯信念、评估可能的行为，并基于这些信念进行选择——仍未被探索。我们开发了一个框架以通过分离信念、评估和选择来识别这种能力，并将其应用于一系列非合作环境中。通过联合分析模型的已显性选择和推理轨迹，并引入一个无上下文的游戏排除记忆性的模仿，我们表明当前前沿模型在目标推理深度上表现出一致的信念最佳响应行为。在不受限制的情况下，它们会自我限制推理深度，并形成对人类和合成对手的不同假设，揭示了一种新的元推理形式。随着复杂性的增加，显式的递归让位于内部生成的选择启发式规则，这些规则是稳定的、模型特定的，并不同于已知的人类偏差。这些发现表明，连贯的信念、元推理和新颖启发式规则的形成可以从语言建模目标中共同涌现，为人工代理的战略认知研究提供了结构化的基础。 

---
# Adaptive Selection of Symbolic Languages for Improving LLM Logical Reasoning 

**Title (ZH)**: 符号语言自适应选择以提高大语言模型逻辑推理能力 

**Authors**: Xiangyu Wang, Haocheng Yang, Fengxiang Cheng, Fenrong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10703)  

**Abstract**: Large Language Models (LLMs) still struggle with complex logical reasoning. While previous works achieve remarkable improvements, their performance is highly dependent on the correctness of translating natural language (NL) problems into a symbolic language (SL). Though numerous works focusing on improving this translation accuracy, they only consider the similarity between the meaning of SL and NL, overlooking another crucial influencing factor, the selection of the target SL type itself. For example, first-order logic language specializes in logical reasoning with categorical syllogisms and complex quantifiers, while Boolean satisfiability formalism excels at representing constraint satisfaction like partial problems. To our knowledge, this is the first paper to claim and verify that different NL logical reasoning problem corresponds to different optimal SL formalization for translation. Based on this, we propose a methods to improve the logical reasoning performance of LLMs by adaptively selecting the most suitable SL for each problem prior to translation. Specifically, we leverage LLMs to select the target SL among first-order logic, logic programming and Boolean satisfiability and then translate the problem in NL to target SL expressions as well as employ the corresponding logical solver to derive the final answer. Experimental results on benchmarks show that our adaptive selection method significantly outperforms translating all into single SL and randomly selecting the SL. On a mixed dataset of these benchmarks, our approach achieves 96% accuracy, which improving performance by 25% compared to the second highest accuracy from the first-order logic translation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂逻辑推理方面仍存在挑战。尽管先前的工作取得了显著的改进，但其性能高度依赖于自然语言（NL）问题向符号语言（SL）的正确翻译。尽管有许多工作专注于提高这种翻译的准确性，它们仅考虑SL和NL之间的相似性，而忽视了选择目标SL类型本身这一关键影响因素。例如，一阶逻辑语言擅长处理范畴三段论和复杂量词，而布尔可满足性形式主义则擅长表示诸如不完全问题的约束满足。据我们所知，这是第一篇提出并验证不同NL逻辑推理问题对应于不同的最优SL形式化以进行翻译的文章。基于此，我们提出了一种方法，通过在翻译前适应性选择最合适的SL来提高LLM的逻辑推理性能。具体来说，我们利用LLMs在一阶逻辑、逻辑编程和布尔可满足性之间选择目标SL，然后将NL问题翻译为目标SL表达式，并使用相应的逻辑求解器推导出最终答案。在基准测试上的实验结果表明，我们的适应性选择方法显著优于将所有内容翻译为单一SL或将SL随机选择的方法。在这些基准测试的混合数据集上，我们的方法达到了96%的准确率，与一阶逻辑翻译的最高准确率相比提高了25%。 

---
# Simpliflow: A Lightweight Open-Source Framework for Rapid Creation and Deployment of Generative Agentic AI Workflows 

**Title (ZH)**: Simpliflow：一种轻量级开源框架，用于快速创建和部署生成型代理AI工作流 

**Authors**: Deven Panchal  

**Link**: [PDF](https://arxiv.org/pdf/2510.10675)  

**Abstract**: Generative Agentic AI systems are emerging as a powerful paradigm for automating complex, multi-step tasks. However, many existing frameworks for building these systems introduce significant complexity, a steep learning curve, and substantial boilerplate code, hindering rapid prototyping and deployment. This paper introduces simpliflow, a lightweight, open-source Python framework designed to address these challenges. simpliflow enables the rapid development and orchestration of linear, deterministic agentic workflows through a declarative, JSON-based configuration. Its modular architecture decouples agent management, workflow execution, and post-processing, promoting ease of use and extensibility. By integrating with LiteLLM, it supports over 100 Large Language Models (LLMs) out-of-the-box. We present the architecture, operational flow, and core features of simpliflow, demonstrating its utility through diverse use cases ranging from software development simulation to real-time system interaction. A comparative analysis with prominent frameworks like LangChain and AutoGen highlights simpliflow's unique position as a tool optimized for simplicity, control, and speed in deterministic workflow environments. 

**Abstract (ZH)**: 生成型自主AI系统正 emerges as a powerful paradigm for automating complex, multi-step tasks. However, many existing frameworks for building these systems introduce significant complexity, a steep learning curve, and substantial boilerplate code, hindering rapid prototyping and deployment. This paper introduces simpliflow, a lightweight, open-source Python framework designed to address these challenges. Simpliflow enables the rapid development and orchestration of linear, deterministic agentic workflows through a declarative, JSON-based configuration. Its modular architecture decouples agent management, workflow execution, and post-processing, promoting ease of use and extensibility. By integrating with LiteLLM, it supports over 100 Large Language Models (LLMs) out-of-the-box. We present the architecture, operational flow, and core features of Simpliflow, demonstrating its utility through diverse use cases ranging from software development simulation to real-time system interaction. A comparative analysis with prominent frameworks like LangChain and AutoGen highlights Simpliflow's unique position as a tool optimized for simplicity, control, and speed in deterministic workflow environments.

翻译后的标题：
轻量级开源Python框架simpliflow：简化自主AI工作流开发与部署 

---
# Hierarchical Optimization via LLM-Guided Objective Evolution for Mobility-on-Demand Systems 

**Title (ZH)**: 基于LLM引导目标演化的大规模出行系统层次优化 

**Authors**: Yi Zhang, Yushen Long, Yun Ni, Liping Huang, Xiaohong Wang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10644)  

**Abstract**: Online ride-hailing platforms aim to deliver efficient mobility-on-demand services, often facing challenges in balancing dynamic and spatially heterogeneous supply and demand. Existing methods typically fall into two categories: reinforcement learning (RL) approaches, which suffer from data inefficiency, oversimplified modeling of real-world dynamics, and difficulty enforcing operational constraints; or decomposed online optimization methods, which rely on manually designed high-level objectives that lack awareness of low-level routing dynamics. To address this issue, we propose a novel hybrid framework that integrates large language model (LLM) with mathematical optimization in a dynamic hierarchical system: (1) it is training-free, removing the need for large-scale interaction data as in RL, and (2) it leverages LLM to bridge cognitive limitations caused by problem decomposition by adaptively generating high-level objectives. Within this framework, LLM serves as a meta-optimizer, producing semantic heuristics that guide a low-level optimizer responsible for constraint enforcement and real-time decision execution. These heuristics are refined through a closed-loop evolutionary process, driven by harmony search, which iteratively adapts the LLM prompts based on feasibility and performance feedback from the optimization layer. Extensive experiments based on scenarios derived from both the New York and Chicago taxi datasets demonstrate the effectiveness of our approach, achieving an average improvement of 16% compared to state-of-the-art baselines. 

**Abstract (ZH)**: 基于大型语言模型的混合优化框架在动态分级系统中实现在线拼车平台的高效需求响应服务 

---
# EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms 

**Title (ZH)**: EA4LLM：基于进化算法的无梯度大型语言模型优化方法 

**Authors**: WenTao Liu, Siyu Song, Hao Hao, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.10603)  

**Abstract**: In recent years, large language models (LLMs) have made remarkable progress, with model optimization primarily relying on gradient-based optimizers such as Adam. However, these gradient-based methods impose stringent hardware requirements, demanding high-concurrency, high-memory GPUs. Moreover, they require all neural network operations to be differentiable, thereby excluding many promising non-differentiable architectures from practical use. To address these limitations, we propose a method for optimizing LLMs using evolutionary algorithms (EA4LLM) and, for the first time, successfully demonstrate its capability to train a 1-billion-parameter LLM from the pre-trained stage. We conduct extensive experiments and provide key insights into how evolutionary algorithms can effectively optimize neural networks. Our work challenges the prevailing assumption that gradient-based optimization is the only viable approach for training neural networks. It also holds significant potential to reduce the computational cost of training large language models, thereby enabling groups with limited computational resources to participate in deep learning research. 

**Abstract (ZH)**: 基于进化算法优化大型语言模型（EA4LLM）及其首个亿级参数模型训练示范 

---
# A Layered Intuition -- Method Model with Scope Extension for LLM Reasoning 

**Title (ZH)**: 分层直觉——具有范围扩展的LLM推理方法模型 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.10592)  

**Abstract**: Existing studies have introduced method-based reasoning and scope extension as approaches to enhance Large Language Model (LLM) performance beyond direct matrix mappings. Building on these foundations, this paper summarizes and integrates these ideas into a unified Intuition-Method Layered Model with Scope Extension, designed to address indirected (unseen) issues more systematically. In this framework, intuition-based thinking provides rapid first-reaction answers, while method-based thinking decouples questions and solutions into transferable reasoning units. Scope extension is then applied to broaden applicability, including vertical (cause analysis), horizontal (parallel and generalized issues), and for the first time, temporal and spatial extensions, which expand reasoning across time and contextual dimensions. These extensions are organized into systematic knowledge trees that interconnect into a knowledge network, thereby increasing adaptability. To quantitatively evaluate this process, we propose the entropy of method extension, which measures the independence and diversity of extensions as an indicator of the system's capacity to solve unseen questions. By logically connecting existing approaches with new extensions and introducing an entropy-based evaluation framework, this work advances toward a more robust and extensible reasoning paradigm for LLMs in real-world problem-solving. 

**Abstract (ZH)**: 现有的研究引入了基于方法的推理和范围扩展，以超越直接矩阵映射提升大规模语言模型（LLM）的性能。在此基础上，本文总结并整合这些理念，提出了一种统一的直觉-方法分层模型并结合范围扩展，旨在更系统地解决间接（未见）问题。在这种框架中，基于直觉的思考提供快速的第一反应答案，而基于方法的思考则将问题和解决方案分解为可转移的推理单元。范围扩展随后被应用以扩大适用性，包括垂直（原因分析）、水平（并行和泛化的议题），以及首次提出的时空扩展，这些扩展将推理扩展到时间与情境维度。这些扩展被组织成系统化的知识树，相互连接成知识网络，从而增加模型的适应性。为了定量评估这一过程，本文提出了方法扩展的熵，作为系统解决未见问题能力的指标。通过逻辑连接现有方法并与新的扩展相结合，并引入基于熵的评估框架，本文朝着更稳健和可扩展的LLM推理范式迈出了一步，在实际问题解决中更为可靠。 

---
# ELAIPBench: A Benchmark for Expert-Level Artificial Intelligence Paper Understanding 

**Title (ZH)**: ELAIPBench: 专家级人工 intelligence 论文理解基准 

**Authors**: Xinbang Dai, Huikang Hu, Yongrui Chen, Jiaqi Li, Rihui Jin, Yuyang Zhang, Xiaoguang Li, Lifeng Shang, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10549)  

**Abstract**: While large language models (LLMs) excel at many domain-specific tasks, their ability to deeply comprehend and reason about full-length academic papers remains underexplored. Existing benchmarks often fall short of capturing such depth, either due to surface-level question design or unreliable evaluation metrics. To address this gap, we introduce ELAIPBench, a benchmark curated by domain experts to evaluate LLMs' comprehension of artificial intelligence (AI) research papers. Developed through an incentive-driven, adversarial annotation process, ELAIPBench features 403 multiple-choice questions from 137 papers. It spans three difficulty levels and emphasizes non-trivial reasoning rather than shallow retrieval. Our experiments show that the best-performing LLM achieves an accuracy of only 39.95%, far below human performance. Moreover, we observe that frontier LLMs equipped with a thinking mode or a retrieval-augmented generation (RAG) system fail to improve final results-even harming accuracy due to overthinking or noisy retrieval. These findings underscore the significant gap between current LLM capabilities and genuine comprehension of academic papers. 

**Abstract (ZH)**: 虽然大规模语言模型在许多领域特定任务中表现出色，但它们深入理解和推理全文学术论文的能力尚未得到充分探索。现有的基准测试往往未能捕捉到这种深度，要么由于表面化的问答设计，要么由于不可靠的评价指标。为了解决这一差距，我们引入了ELAIPBench，这是一个由领域专家筛选的基准测试，用于评估大规模语言模型对人工智能研究论文的理解能力。通过激励驱动的对抗性注释过程开发而成，ELAIPBench 包含来自137篇论文的403道选择题，涵盖了三个难度级别，并强调非平凡推理而非浅层检索。我们的实验显示，表现最好的大规模语言模型的准确率仅为39.95%，远低于人类的表现。此外，我们观察到，配备了思考模式或检索增强生成系统的前沿大规模语言模型未能提高最终结果，甚至因过度思考或嘈杂的检索而导致准确率下降。这些发现突显了当前大规模语言模型能力和对学术论文真正理解之间的巨大差距。 

---
# MedCoAct: Confidence-Aware Multi-Agent Collaboration for Complete Clinical Decision 

**Title (ZH)**: MedCoAct：基于信心的多 Agents 合作以实现完整的临床决策 

**Authors**: Hongjie Zheng, Zesheng Shi, Ping Yi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10461)  

**Abstract**: Autonomous agents utilizing Large Language Models (LLMs) have demonstrated remarkable capabilities in isolated medical tasks like diagnosis and image analysis, but struggle with integrated clinical workflows that connect diagnostic reasoning and medication decisions. We identify a core limitation: existing medical AI systems process tasks in isolation without the cross-validation and knowledge integration found in clinical teams, reducing their effectiveness in real-world healthcare scenarios. To transform the isolation paradigm into a collaborative approach, we propose MedCoAct, a confidence-aware multi-agent framework that simulates clinical collaboration by integrating specialized doctor and pharmacist agents, and present a benchmark, DrugCareQA, to evaluate medical AI capabilities in integrated diagnosis and treatment workflows. Our results demonstrate that MedCoAct achieves 67.58\% diagnostic accuracy and 67.58\% medication recommendation accuracy, outperforming single agent framework by 7.04\% and 7.08\% respectively. This collaborative approach generalizes well across diverse medical domains, proving especially effective for telemedicine consultations and routine clinical scenarios, while providing interpretable decision-making pathways. 

**Abstract (ZH)**: 利用大规模语言模型的自主代理在集成临床工作流中的局限性及MedCoAct框架的研究：一种提高诊断和治疗准确性的协作方法 

---
# Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction 

**Title (ZH)**: Traj-CoA: 基于链式代理的患者轨迹建模用于肺癌风险预测 

**Authors**: Sihang Zeng, Yujuan Fu, Sitong Zhou, Zixuan Yu, Lucas Jing Liu, Jun Wen, Matthew Thompson, Ruth Etzioni, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10454)  

**Abstract**: Large language models (LLMs) offer a generalizable approach for modeling patient trajectories, but suffer from the long and noisy nature of electronic health records (EHR) data in temporal reasoning. To address these challenges, we introduce Traj-CoA, a multi-agent system involving chain-of-agents for patient trajectory modeling. Traj-CoA employs a chain of worker agents to process EHR data in manageable chunks sequentially, distilling critical events into a shared long-term memory module, EHRMem, to reduce noise and preserve a comprehensive timeline. A final manager agent synthesizes the worker agents' summary and the extracted timeline in EHRMem to make predictions. In a zero-shot one-year lung cancer risk prediction task based on five-year EHR data, Traj-CoA outperforms baselines of four categories. Analysis reveals that Traj-CoA exhibits clinically aligned temporal reasoning, establishing it as a promisingly robust and generalizable approach for modeling complex patient trajectories. 

**Abstract (ZH)**: 大型语言模型（LLMs）提供了一种适用于建模患者轨迹的一般化方法，但在处理电子健康记录（EHR）数据的时间推理时，面临数据长且杂的问题。为应对这些挑战，我们引入了Traj-CoA，这是一种涉及多智能体系统的患者轨迹建模方法，该系统包含一个智能体链。Traj-CoA 使用一系列工作智能体按顺序处理 EHR 数据的可管理片段，将关键事件提炼到共享长时记忆模块 EHRMem 中，以降低噪音并保持完整的事件时间线。最终，管理智能体综合工作智能体的总结和从 EHRMem 中提取的时间线来进行预测。在基于五年 EHR 数据的一年零样本肺癌风险预测任务中，Traj-CoA 在四个类别基线中表现优越。分析表明，Traj-CoA 具有临床对齐的时间推理能力，将其确立为建模复杂患者轨迹的一种有前景且通用的方法。 

---
# Trace Length is a Simple Uncertainty Signal in Reasoning Models 

**Title (ZH)**: 轨迹长度是推理模型中的一个简单的不确定性信号 

**Authors**: Siddartha Devic, Charlotte Peale, Arwen Bradley, Sinead Williamson, Preetum Nakkiran, Aravind Gollakota  

**Link**: [PDF](https://arxiv.org/pdf/2510.10409)  

**Abstract**: Uncertainty quantification for LLMs is a key research direction towards addressing hallucination and other issues that limit their reliable deployment. In this work, we show that reasoning trace length is a simple and useful confidence estimator in large reasoning models. Through comprehensive experiments across multiple models, datasets, and prompts, we show that trace length performs in comparable but complementary ways to other zero-shot confidence estimators such as verbalized confidence. Our work reveals that reasoning post-training fundamentally alters the relationship between trace length and accuracy, going beyond prior work that had shown that post-training causes traces to grow longer in general (e.g., "overthinking"). We investigate the mechanisms behind trace length's performance as a confidence signal, observing that the effect remains even after adjusting for confounders such as problem difficulty and GRPO-induced length bias. We identify high-entropy or "forking" tokens as playing a key role in the mechanism. Our findings demonstrate that reasoning post-training enhances uncertainty quantification beyond verbal expressions, and establish trace length as a practical confidence measure for large reasoning models. 

**Abstract (ZH)**: LLMs的不确定性量化是解决幻觉及其他限制其可靠部署问题的关键研究方向。在本文中，我们展示了推理踪迹长度是一种简单而实用的置信度估计器。通过在多种模型、数据集和提示下的综合实验，我们表明踪迹长度在与其他零样本置信度估计器（如口头置信度）的性能上具有可比性和互补性。我们的工作揭示了后训练推理从根本上改变了踪迹长度与准确性的关系，超越了以往工作仅显示后训练使踪迹普遍增长（如“过度思考”）的发现。我们探讨了踪迹长度作为置信度信号表现背后的机制，观察到即使调整了问题难度和GRPO诱导的长度偏差等混杂因素，效果依然存在。我们发现高熵或“分支”标记在机制中扮演了关键角色。我们的研究结果表明，后训练推理增强了不确定性量化，超出口头表达之外，并确立了踪迹长度作为大型推理模型的实用置信度度量。 

---
# LLM-Friendly Knowledge Representation for Customer Support 

**Title (ZH)**: 面向LLM的客户支持知识表示 

**Authors**: Hanchen Su, Wei Luo, Wei Han, Yu Elaine Liu, Yufeng Wayne Zhang, Cen Mia Zhao, Ying Joy Zhang, Yashar Mehdad  

**Link**: [PDF](https://arxiv.org/pdf/2510.10331)  

**Abstract**: We propose a practical approach by integrating Large Language Models (LLMs) with a framework designed to navigate the complexities of Airbnb customer support operations. In this paper, our methodology employs a novel reformatting technique, the Intent, Context, and Action (ICA) format, which transforms policies and workflows into a structure more comprehensible to LLMs. Additionally, we develop a synthetic data generation strategy to create training data with minimal human intervention, enabling cost-effective fine-tuning of our model. Our internal experiments (not applied to Airbnb products) demonstrate that our approach of restructuring workflows and fine-tuning LLMs with synthetic data significantly enhances their performance, setting a new benchmark for their application in customer support. Our solution is not only cost-effective but also improves customer support, as evidenced by both accuracy and manual processing time evaluation metrics. 

**Abstract (ZH)**: 我们提出了一种将大型语言模型（LLMs）与设计用于导航Airbnb客户支持运营复杂性的框架集成的实用方法。本文方法采用了一种新颖的重格式化技术——意图、上下文和行动（ICA）格式，将政策和工作流转换为LLMs更容易理解的结构。此外，我们开发了一种合成数据生成策略，以最少的人工干预创建训练数据，从而实现成本效益高的模型微调。我们的内部实验（未应用于Airbnb产品）表明，通过重新结构化工作流和使用合成数据微调LLMs，显著提升了其性能，为在客户支持中应用LLMs树立了新的基准。我们的解决方案不仅成本效益高，还能有效提升客户支持质量，这一结论得到了准确性和手动处理时间评估指标的支持。 

---
# The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities 

**Title (ZH)**: LLMs的阿喀琉斯之踵：少量神经元的改变如何毁掉语言能力 

**Authors**: Zixuan Qin, Kunlin Lyu, Qingchen Yu, Yifan Sun, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10238)  

**Abstract**: Large Language Models (LLMs) have become foundational tools in natural language processing, powering a wide range of applications and research. Many studies have shown that LLMs share significant similarities with the human brain. Recent neuroscience research has found that a small subset of biological neurons in the human brain are crucial for core cognitive functions, which raises a fundamental question: do LLMs also contain a small subset of critical neurons? In this paper, we investigate this question by proposing a Perturbation-based Causal Identification of Critical Neurons method to systematically locate such critical neurons in LLMs. Our findings reveal three key insights: (1) LLMs contain ultra-sparse critical neuron sets. Disrupting these critical neurons can cause a 72B-parameter model with over 1.1 billion neurons to completely collapse, with perplexity increasing by up to 20 orders of magnitude; (2) These critical neurons are not uniformly distributed, but tend to concentrate in the outer layers, particularly within the MLP down\_proj components; (3) Performance degradation exhibits sharp phase transitions, rather than a gradual decline, when these critical neurons are disrupted. Through comprehensive experiments across diverse model architectures and scales, we provide deeper analysis of these phenomena and their implications for LLM robustness and interpretability. These findings can offer guidance for developing more robust model architectures and improving deployment security in safety-critical applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为自然语言处理中的基础工具，支撑着广泛的应用和研究。许多研究表明，LLMs在很大程度上与人脑相似。最近的神经科学研究发现，人脑中一小部分生物神经元对于核心认知功能至关重要，这引发了一个基本问题：LLMs中是否也存在这样一小部分关键神经元？在本文中，我们通过提出基于扰动的关键神经元因果识别方法，系统地识别LLMs中的这类关键神经元。我们的发现揭示了三个关键洞察：（1）LLMs包含极度稀疏的关键神经元集合。破坏这些关键神经元会导致一个参数量为720亿、拥有超过11亿个神经元的模型完全崩溃，困惑度增加20个数量级；（2）这些关键神经元并非均匀分布，而是倾向于集中在外部层，特别是MLP down_proj组件中；（3）在这些关键神经元被破坏时，性能下降表现出明显的相变，而非渐进下降。通过跨不同模型架构和规模的全面实验，我们对这些现象及其对LLMs稳健性和可解释性的含义进行了更深入的分析。这些发现可以为开发更具稳健性的模型架构并改善安全关键应用的部署安全性提供指导。 

---
# Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning 

**Title (ZH)**: 自适应双推理机：混合推理的大规模推理模型可以高效思考 

**Authors**: Yujian Zhang, Keyu Chen, Zhifeng Shen, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.10207)  

**Abstract**: Although Long Reasoning Models (LRMs) have achieved superior performance on various reasoning scenarios, they often suffer from increased computational costs and inference latency caused by overthinking. To address these limitations, we propose Adaptive Dual Reasoner, which supports two reasoning modes: fast thinking and slow thinking. ADR dynamically alternates between these modes based on the contextual complexity during reasoning. ADR is trained in two stages: (1) A cold-start stage using supervised fine-tuning (SFT) to equip the model with the ability to integrate both fast and slow reasoning modes, in which we construct a hybrid reasoning dataset through a dedicated pipeline to provide large-scale supervision. (2) A reinforcement learning stage for optimizing reasoning effort, where we introduce Entropy-guided Hybrid Policy Optimization EHPO, an RL training framework employing an entropy-guided dynamic rollout strategy for branching at high-entropy units and a difficulty-aware penalty to balance fast and slow reasoning. Across challenging mathematical reasoning benchmarks, ADR achieves an effective balance between reasoning performance and efficiency among state-of-the-art approaches. Specifically, ADR yields a performance gain of up to 6.1%, while reducing the reasoning output length by 49.5% to 59.3%. 

**Abstract (ZH)**: 尽管长推理模型(LRMs)在各种推理场景中取得了优异 performance，但它们往往由于过度推理而导致计算成本和推理延迟增加。为了解决这些限制，我们提出了自适应双推理器(ADR)，支持快思考和慢思考两种推理模式。ADR根据推理过程中上下文复杂度动态交替使用这两种模式。ADR的训练分为两个阶段：(1) 使用监督微调(SFT)的冷启动阶段，通过专用管道构建混合推理数据集，提供大规模监督，使模型具备结合快思考和慢思考的能力；(2) 通过引入基于熵导引的混合策略优化框架EHPO，结合熵导引的动态展开策略和难度感知的惩罚机制，优化推理努力，以平衡快思考和慢思考。在具有挑战性的数学推理基准测试中，ADR在推理性能和效率之间实现了有效的平衡。具体而言，ADR在推理性能上提高了最多6.1%，同时将推理输出长度减少了49.5%至59.3%。 

---
# PIXEL: Adaptive Steering Via Position-wise Injection with eXact Estimated Levels under Subspace Calibration 

**Title (ZH)**: PIXEL：基于子空间校准的定位位置注入与精确估计层级的自适应转向 

**Authors**: Manjiang Yu, Hongji Li, Priyanka Singh, Xue Li, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10205)  

**Abstract**: Reliable behavior control is central to deploying large language models (LLMs) on the web. Activation steering offers a tuning-free route to align attributes (e.g., truthfulness) that ensure trustworthy generation. Prevailing approaches rely on coarse heuristics and lack a principled account of where to steer and how strongly to intervene. To this end, we propose Position-wise Injection with eXact Estimated Levels (PIXEL), a position-wise activation steering framework that, in contrast to prior work, learns a property-aligned subspace from dual views (tail-averaged and end-token) and selects intervention strength via a constrained geometric objective with a closed-form solution, thereby adapting to token-level sensitivity without global hyperparameter tuning. PIXEL further performs sample-level orthogonal residual calibration to refine the global attribute direction and employs a lightweight position-scanning routine to identify receptive injection sites. We additionally provide representation-level guarantees for the minimal-intervention rule, supporting reliable alignment. Across diverse models and evaluation paradigms, PIXEL consistently improves attribute alignment while preserving model general capabilities, offering a practical and principled method for LLMs' controllable generation. Our code is available at this https URL 

**Abstract (ZH)**: 可靠的行為控制是將大語言模型（LLMs）部署在網上關鍵所在。基于.Position-wise Injection with Exact Estimated Levels (PIXEL) 的激活控制框架提供了在不調整超参数的情况下进行精确属性对齐的方法，从而实现可信赖的生成。 

---
# Don't Just Fine-tune the Agent, Tune the Environment 

**Title (ZH)**: 不要只是微调代理，也要微调环境 

**Authors**: Siyuan Lu, Zechuan Wang, Hongxuan Zhang, Qintong Wu, Leilei Gan, Chenyi Zhuang, Jinjie Gu, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.10197)  

**Abstract**: Large Language Model (LLM) agents show great promise for complex, multi-turn tool-use tasks, but their development is often hampered by the extreme scarcity of high-quality training data. Supervised fine-tuning (SFT) on synthetic data leads to overfitting, whereas standard reinforcement learning (RL) struggles with a critical cold-start problem and training instability. To address these challenges, we introduce $\textbf{Environment Tuning}$, a novel training paradigm that enables agents to learn complex behaviors directly from problem instances without relying on pre-collected expert trajectories. $\textbf{Environment Tuning}$ orchestrates this learning process through a structured curriculum, actionable environment augmentation that provides corrective feedback, and fine-grained progress rewards to ensure stable and efficient exploration. Using only 400 problem instances from Berkeley Function-Calling Leaderboard (BFCL) benchmark, our method not only achieves competitive in-distribution performance against strong baselines but also demonstrates superior out-of-distribution generalization, overcoming the performance collapse common to SFT-based approaches. Our work presents a paradigm shift from supervised fine-tuning on static trajectories to dynamic, environment-based exploration, paving the way for training more robust and data-efficient agents. 

**Abstract (ZH)**: 环境调优：一种新型训练范式，用于直接从问题实例中学习复杂行为，而不依赖于预先收集的专家轨迹 

---
# SAFER: Risk-Constrained Sample-then-Filter in Large Language Models 

**Title (ZH)**: SAFER: 大语言模型中的风险约束采样过滤方法 

**Authors**: Qingni Wang, Yue Fan, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10193)  

**Abstract**: As large language models (LLMs) are increasingly deployed in risk-sensitive applications such as real-world open-ended question answering (QA), ensuring the trustworthiness of their outputs has become critical. Existing selective conformal prediction (SCP) methods provide statistical guarantees by constructing prediction sets with a constrained miscoverage rate for correct answers. However, prior works unrealistically assume that admissible answers for all instances can be obtained via finite sampling, even for open-ended QA scenarios that lack a fixed and finite solution space. To address this, we introduce a two-stage risk control framework comprising abstention-aware sampling and conformalized filtering (SAFER). Firstly, on a held-out calibration set, SAFER calibrates a sampling budget within the maximum sampling cap, using the Clopper-Pearson exact method at a user-desired risk level (i.e., the maximum allowable miscoverage rate of the sampling sets). If the risk level cannot be satisfied within the cap, we abstain; otherwise, the calibrated sampling budget becomes the minimum requirements at test time. Then, we employ calibration instances where correct answers are attainable under the calibrated budget and apply the conformal risk control method to determine a statistically valid uncertainty threshold, which filters unreliable distractors from the candidate set for each test data point. In this stage, SAFER introduces an additional risk level to guide the calculation of the threshold, thereby controlling the risk of correct answers being excluded. Furthermore, we show that SAFER is compatible with various task-specific admission criteria and calibration-test split ratios, highlighting its robustness and high data efficiency. 

**Abstract (ZH)**: 大型语言模型（LLMs）在风险敏感应用如真实世界开放式问答（QA）中的部署日益增多，确保其输出的可信度变得至关重要。现有的选择性一致性预测（SCP）方法通过构建具有约束置信率的预测集来提供统计保证，针对正确答案。然而，先前的工作不合理地假设所有实例的可接受答案可以通过有限采样获得，即使是在缺乏固定有限解决方案空间的开放式QA场景中。为解决这一问题，我们引入了一种两阶段风险控制框架——感知避免采样和一致性筛选（SAFER）。首先，在保留的校准集上，SAFER使用用户指定的风险水平（即采样集的最大允许置信率错误）通过Clopper-Pearson精确方法校准采样预算，并在预算上限内实现风险控制。如果风险水平无法满足上限，我们采取避免策略；否则，校准后的采样预算成为测试时的最低要求。然后，我们使用校准预算下可获得正确答案的校准实例，并应用一致性风险控制系统来确定一个统计上有效的不确定性阈值，该阈值能从每个测试数据点的候选集中筛选出不可靠的选项。在这一阶段，SAFER引入了额外的风险水平来指导阈值的计算，从而控制正确答案被排除的风险。此外，我们证明了SAFER可以与各种任务特定的接纳标准和校准-测试分割比例兼容，突显了它的鲁棒性和高数据效率。 

---
# Concise Reasoning in the Lens of Lagrangian Optimization 

**Title (ZH)**: 拉格朗日优化视角下的精炼推理 

**Authors**: Chengqian Gao, Haonan Li, Taylor W. Killian, Jianshu She, Renxi Wang, Liqun Ma, Zhoujun Cheng, Shibo Hao, Zhiqiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10168)  

**Abstract**: Concise reasoning in large language models seeks to generate only essential intermediate steps needed to arrive at a final answer, thereby alleviating issues of overthinking. Most proposed approaches hinge on carefully hand-crafted heuristics, struggling to balance concision with performance, often failing to adapt across domains and model scales. In this work, we address these challenges by introducing a principled and pragmatic strategy, performance-aware length updating (PALU). As a principled algorithm, PALU formulates concise reasoning as a constrained optimization problem, minimizing response length subject to a performance constraint, and then applies Lagrangian optimization to convert it into a tractable unconstrained problem. As a pragmatic solution, PALU streamlines complicated update rules through three approximations: (i) estimating performance with off-policy rollouts, (ii) truncating the Lagrange multiplier to two extremes, and (iii) replacing gradient-based updates with quantile-driven length adjustments. PALU reduces output length by 65% while improving accuracy by 15% when applied to DeepSeek-Distill-Qwen-1.5B, averaged over five benchmarks, outperforming a range of alternative methods. Furthermore, PALU is demonstrated to adapt across both domain (logic, STEM and math) and model scale (1.5B, 7B, 14B) entrenching the algorithm as a practical and effective concise reasoning approach. 

**Abstract (ZH)**: 大型语言模型中的紧凑推理寻求仅生成到达最终答案所需的 essential 中间步骤，从而缓解过度思考的问题。大多数提出的方案依赖于精心的手工启发式方法，难以在紧凑性和性能之间取得平衡，往往无法跨领域和模型规模进行适应。在本工作中，我们通过引入一个有原则且实用的策略——性能感知长度更新（PALU）来应对这些挑战。作为一项有原则的算法，PALU 将紧凑推理建模为一个受约束的优化问题，最小化响应长度同时满足性能约束，然后通过拉格朗日优化将其转化为可处理的无约束问题。作为一种实用的解决方案，PALU 通过三种近似简化复杂的更新规则：（i）使用离策 rollout 估计性能，（ii）截断拉格朗日乘数的两端，（iii）用基于分位数的长度调整替换基于梯度的更新。当应用于 DeepSeek-Distill-Qwen-1.5B 并在五个基准上平均时，PALU 减少了输出长度 65%，同时提高了准确性 15%，优于多种替代方法。此外，PALU 被证明可在领域（逻辑、STEM 和数学）和模型规模（1.5B、7B、14B）之间进行适应，使该算法成为一个实用有效的紧凑推理方法。 

---
# Agentic Troubleshooting Guide Automation for Incident Management 

**Title (ZH)**: 代理故障排查指南自动化在事件管理中的应用 

**Authors**: Jiayi Mao, Liqun Li, Yanjie Gao, Zegang Peng, Shilin He, Chaoyun Zhang, Si Qin, Samia Khalid, Qingwei Lin, Saravan Rajmohan, Sitaram Lanka, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10074)  

**Abstract**: Effective incident management in large-scale IT systems relies on troubleshooting guides (TSGs), but their manual execution is slow and error-prone. While recent advances in LLMs offer promise for automating incident management tasks, existing LLM-based solutions lack specialized support for several key challenges, including managing TSG quality issues, interpreting complex control flow, handling data-intensive queries, and exploiting execution parallelism. We first conducted an empirical study on 92 real-world TSGs, and, guided by our findings, we present StepFly, a novel end-to-end agentic framework for troubleshooting guide automation. Our approach features a three-stage workflow: the first stage provides a comprehensive guide together with a tool, TSG Mentor, to assist SREs in improving TSG quality; the second stage performs offline preprocessing using LLMs to extract structured execution DAGs from unstructured TSGs and to create dedicated Query Preparation Plugins (QPPs); and the third stage executes online using a DAG-guided scheduler-executor framework with a memory system to guarantee correct workflow and support parallel execution of independent steps. Our empirical evaluation on a collection of real-world TSGs and incidents demonstrates that StepFly achieves a ~94% success rate on GPT-4.1, outperforming baselines with less time and token consumption. Furthermore, it achieves a remarkable execution time reduction of 32.9% to 70.4% for parallelizable TSGs. 

**Abstract (ZH)**: 大规模IT系统中有效的事件管理依赖于故障排除指南（TSGs），但其手动执行速度慢且易出错。尽管最近的语言模型（LLMs）的进步为自动化事件管理任务带来了希望，现有的基于LLM的解决方案缺乏对多个关键挑战的专门支持，包括管理TSG质量问题、解释复杂的控制流、处理数据密集型查询以及利用执行并行性。我们首先对92个真实的TSG进行了实证研究，并根据研究发现，提出了一种名为StepFly的新颖的端到端代理框架，用于故障排除指南的自动化。该方法具有三阶段的工作流：第一阶段提供一个全面的指南和工具TSG Mentor，以协助SRE提高TSG质量；第二阶段使用LLM进行离线预处理，从未结构化的TSG中提取结构化执行DAG并创建专用的查询准备插件（QPPs）；第三阶段使用DAG引导的调度执行框架和内存系统在线执行，以确保正确的流程并支持独立步骤的并行执行。对多种真实TSG和事件的实证评估显示，StepFly在GPT-4.1上实现了约94%的成功率，优于基线模型在更少的时间和.token消耗下的性能。此外，对于可并行化的TSG，它实现了高达32.9%至70.4%的执行时间减少。 

---
# SwarmSys: Decentralized Swarm-Inspired Agents for Scalable and Adaptive Reasoning 

**Title (ZH)**: SwarmSys: 分布式 swarm 启发式代理实现可扩展和自适应推理 

**Authors**: Ruohao Li, Hongjun Liu, Leyi Zhao, Zisu Li, Jiawei Li, Jiajun Jiang, Linning Xu, Chen Zhao, Mingming Fan, Chen Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10047)  

**Abstract**: Large language model (LLM) agents have shown remarkable reasoning abilities. However, existing multi-agent frameworks often rely on fixed roles or centralized control, limiting scalability and adaptability in long-horizon reasoning. We introduce SwarmSys, a closed-loop framework for distributed multi-agent reasoning inspired by swarm intelligence. Coordination in SwarmSys emerges through iterative interactions among three specialized roles, Explorers, Workers, and Validators, that continuously cycle through exploration, exploitation, and validation. To enable scalable and adaptive collaboration, we integrate adaptive agent and event profiles, embedding-based probabilistic matching, and a pheromone-inspired reinforcement mechanism, supporting dynamic task allocation and self-organizing convergence without global supervision. Across symbolic reasoning, research synthesis, and scientific programming tasks, SwarmSys consistently outperforms baselines, improving both accuracy and reasoning stability. These findings highlight swarm-inspired coordination as a promising paradigm for scalable, robust, and adaptive multi-agent reasoning, suggesting that coordination scaling may rival model scaling in advancing LLM intelligence. 

**Abstract (ZH)**: 基于 swarm 智能的分布式多智能体推理框架 SwarmSys 

---
# Failure-Driven Workflow Refinement 

**Title (ZH)**: 故障驱动的工作流细化 

**Authors**: Jusheng Zhang, Kaitong Cai, Qinglin Zeng, Ningyuan Liu, Stephen Fan, Ziliang Chen, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10035)  

**Abstract**: Optimizing LLM-based workflows is typically formulated as a global search, where candidate workflows are evaluated based on a scalar metric. This paradigm, however, suffers from a critical flaw: information collapse. By reducing rich, multi-step execution traces to simple success/failure signals, existing methods are rendered blind to the underlying structure of failures, fundamentally preventing them from modeling the workflow's failure distribution. We reconceptualize this challenge as a distributional problem. We propose a new paradigm where the optimization goal is not to maximize a scalar score, but to directly minimize a workflow's Expected Failure Mass, i.e., the integral of its failure probability density function defined over a high-dimensional Failure Signature Space (FSS). This distributional lens allows us to move from inefficient, zero-order optimization to a principled, gradient-like descent on the failure landscape itself. We introduce CE-Graph, a framework that operationalizes this paradigm through a novel, failure-driven refinement process. CE-Graph approximates the failure distribution from a pool of counterexamples, identifies its densest regions as recurring failure modes, and applies targeted, operator-constrained graph edits via a Propose-and-Verify mechanism to greedily reduce the failure mass. On math, code, and QA benchmarks, our CE-Graph achieves higher robustness at a significantly lower cost than strong baselines. This suggests that a system's reliability emerges not from avoiding failures, but from systematically learning and reshaping the geometric structure of its failure distributions. 

**Abstract (ZH)**: 基于LLM的工作流优化通常被公式化为全局搜索问题，其中候选工作流基于标量指标进行评估。然而，这一范式遭受了一个关键缺陷：信息坍缩。通过将丰富的多步执行踪迹简化为简单的成功/失败信号，现有方法使其无法察觉失败的潜在结构，从根本上阻止其建模工作流的失败分布。我们将这一挑战重新概念化为一个分布性问题。我们提出了一个新的范式，目标不是最大化标量分数，而是直接最小化工作流的预期失败质量，即其失败概率密度函数在高维失败签名空间（FSS）上的积分。这一分布性的视角使我们能够从无效率的一阶优化转向基于失败景观本身的稳健的梯度下降。我们引入了CE-Graph框架，通过一种新型的、基于失败的细化过程实现这一范式。CE-Graph从反例池中近似失败分布，识别其密集区域作为反复出现的故障模式，并通过提案和验证机制应用受操作约束的图编辑，以贪婪地减少失败质量。在数学、代码和QA基准测试中，我们的CE-Graph以显著更低的成本实现了更高的稳健性，这表明系统的可靠性并非来自避免故障，而是来自系统地学习和重塑其失败分布的几何结构。 

---
# RIPRAG: Hack a Black-box Retrieval-Augmented Generation Question-Answering System with Reinforcement Learning 

**Title (ZH)**: RIPRAG: 使用强化学习破解黑箱检索增强生成式问答系统 

**Authors**: Meng Xi, Sihan Lv, Yechen Jin, Guanjie Cheng, Naibo Wang, Ying Li, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.10008)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become a core technology for tasks such as question-answering (QA) and content generation. However, by injecting poisoned documents into the database of RAG systems, attackers can manipulate LLMs to generate text that aligns with their intended preferences. Existing research has primarily focused on white-box attacks against simplified RAG architectures. In this paper, we investigate a more complex and realistic scenario: the attacker lacks knowledge of the RAG system's internal composition and implementation details, and the RAG system comprises components beyond a mere retriever. Specifically, we propose the RIPRAG attack framework, an end-to-end attack pipeline that treats the target RAG system as a black box, where the only information accessible to the attacker is whether the poisoning succeeds. Our method leverages Reinforcement Learning (RL) to optimize the generation model for poisoned documents, ensuring that the generated poisoned document aligns with the target RAG system's preferences. Experimental results demonstrate that this method can effectively execute poisoning attacks against most complex RAG systems, achieving an attack success rate (ASR) improvement of up to 0.72 compared to baseline methods. This highlights prevalent deficiencies in current defensive methods and provides critical insights for LLM security research. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的检索增强生成（RAG）系统已成为问答（QA）和内容生成等任务的核心技术。然而，通过向RAG系统的数据库中注入恶意文档，攻击者可以操控LLMs生成符合其意图的文本。现有研究主要关注对简化RAG架构的白盒攻击。在本文中，我们探讨了一个更复杂且更现实的攻击场景：攻击者不了解RAG系统的内部组成和实现细节，并且RAG系统包含的组件不仅仅是一个检索器。具体而言，我们提出了一种RICRAG攻击框架，这是一种端到端的攻击管道，将目标RAG系统视为黑盒，攻击者唯一可获取的信息是毒化攻击是否成功。我们的方法利用强化学习（RL）优化生成模型以生成恶意文档，确保生成的恶意文档符合目标RAG系统的需求。实验结果表明，该方法可以有效地对大多数复杂的RAG系统执行毒化攻击，相较于基线方法，攻击成功率（ASR）提高了0.72。这揭示了当前防御方法中存在的普遍缺陷，并为LLM安全研究提供了关键洞察。 

---
# Deliberative Dynamics and Value Alignment in LLM Debates 

**Title (ZH)**: 德尔iberative动态与LLM辩论中的价值对齐 

**Authors**: Pratik S. Sachdeva, Tom van Nuenen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10002)  

**Abstract**: As large language models (LLMs) are increasingly deployed in sensitive everyday contexts - offering personal advice, mental health support, and moral guidance - understanding their elicited values in navigating complex moral reasoning is essential. Most evaluations study this sociotechnical alignment through single-turn prompts, but it is unclear if these findings extend to multi-turn settings where values emerge through dialogue, revision, and consensus. We address this gap using LLM debate to examine deliberative dynamics and value alignment in multi-turn settings by prompting subsets of three models (GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash) to collectively assign blame in 1,000 everyday dilemmas from Reddit's "Am I the Asshole" community. We use both synchronous (parallel responses) and round-robin (sequential responses) formats to test order effects and verdict revision. Our findings show striking behavioral differences. In the synchronous setting, GPT showed strong inertia (0.6-3.1% revision rates) while Claude and Gemini were far more flexible (28-41%). Value patterns also diverged: GPT emphasized personal autonomy and direct communication, while Claude and Gemini prioritized empathetic dialogue. Certain values proved especially effective at driving verdict changes. We further find that deliberation format had a strong impact on model behavior: GPT and Gemini stood out as highly conforming relative to Claude, with their verdict behavior strongly shaped by order effects. These results show how deliberation format and model-specific behaviors shape moral reasoning in multi-turn interactions, underscoring that sociotechnical alignment depends on how systems structure dialogue as much as on their outputs. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在敏感的日常情境中被越来越多地部署——提供个人建议、心理健康支持和道德指导——理解它们在复杂道德推理中唤起的价值观至关重要。大多数评估通过单轮提示研究这种社会技术契合度，但这些发现是否适用于多轮对话情境尚不清楚，在多轮对话情境中，价值观通过对话、修订和共识逐渐形成。我们使用LLM辩论来研究多轮设置中的审议动态和价值观契合度，通过提示三个模型（GPT-4.1、Claude 3.7 Sonnet和Gemini 2.0 Flash）共同分配1000个来自Reddit“Am I the Asshole”社区的日常困境中的责任。我们使用同步（并行响应）和轮转（顺序响应）格式来测试顺序效应和裁决修订。我们的发现显示了显著的行为差异。在同步设置中，GPT表现出较强的惯性（0.6-3.1%的修订率），而Claude和Gemini更为灵活（28-41%）。价值观模式也有所不同：GPT强调个人自主和直接沟通，而Claude和Gemini更重视同理心对话。某些价值观对驱动裁决变化尤其有效。进一步的研究发现，审议格式对模型行为产生了强烈影响：GPT和Gemini相对于Claude表现出更高的顺应性，其裁决行为受顺序效应强烈影响。这些结果表明审议格式和模型特定行为如何塑造多轮互动中的道德推理，强调社会技术契合度不仅取决于系统的输出，还取决于对话的结构方式。 

---
# Follow My Lead: Logical Fallacy Classification with Knowledge-Augmented LLMs 

**Title (ZH)**: 跟随我的指引：基于知识增强的大语言模型逻辑谬误分类 

**Authors**: Olivia Peiyu Wang, Tashvi Bansal, Ryan Bai, Emily M. Chui, Leilani H. Gilpin  

**Link**: [PDF](https://arxiv.org/pdf/2510.09970)  

**Abstract**: Large Language Models (LLMs) suffer from critical reasoning gaps, including a tendency to hallucinate and poor accuracy in classifying logical fallacies. This limitation stems from their default System 1 processing, which is fast and intuitive, whereas reliable reasoning requires the deliberate, effortful System 2 approach (Kahneman, 2011; Li et al., 2025). Since full System 2 training is often prohibitively expensive, we explore a low-cost, instruction-based intervention to bridge this gap. Our methodology introduces a novel stepwise instruction dataset that decomposes fallacy classification into a series of atomic procedural steps (simple binary questions). We further augment this with a final verification step where models consult a relational knowledge graph of related fallacies. This procedural, rule-based intervention yields a significant improvement in LLM logical fallacy classification. Crucially, the approach also provides enhanced transparency into the LLMs' decision-making, highlighting a practical pathway for Neuro-symbolic architectures to address LLM reasoning deficits. 

**Abstract (ZH)**: 大型语言模型（LLMs）在批判性推理方面存在关键漏洞，包括倾向虚构和逻辑谬误分类准确性差。这一限制源自它们默认使用快速直观的System 1处理方式，而可靠的推理需要耗费精力的System 2方法（Kahneman, 2011；Li et al., 2025）。由于满量的System 2训练往往成本高昂，我们探索了一种低成本的指令干预方法来弥补这一差距。我们的方法引入了一种新颖的分步骤指令数据集，将谬误分类分解为一系列基本程序步骤（简单的二元问题），并通过最终的验证步骤，使模型参考相关谬误的关系知识图谱。这种基于规则的程序干预显著提升了LLM对逻辑谬误的分类能力。关键的是，该方法还提升了对LLM决策过程的透明度，为神经符号架构解决LLM推理缺陷提供了实际路径。 

---
# The Personalization Trap: How User Memory Alters Emotional Reasoning in LLMs 

**Title (ZH)**: 个性化陷阱：用户记忆如何改变LLM中的情感推理 

**Authors**: Xi Fang, Weijie Xu, Yuchong Zhang, Stephanie Eckman, Scott Nickleach, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.09905)  

**Abstract**: When an AI assistant remembers that Sarah is a single mother working two jobs, does it interpret her stress differently than if she were a wealthy executive? As personalized AI systems increasingly incorporate long-term user memory, understanding how this memory shapes emotional reasoning is critical. We investigate how user memory affects emotional intelligence in large language models (LLMs) by evaluating 15 models on human validated emotional intelligence tests. We find that identical scenarios paired with different user profiles produce systematically divergent emotional interpretations. Across validated user independent emotional scenarios and diverse user profiles, systematic biases emerged in several high-performing LLMs where advantaged profiles received more accurate emotional interpretations. Moreover, LLMs demonstrate significant disparities across demographic factors in emotion understanding and supportive recommendations tasks, indicating that personalization mechanisms can embed social hierarchies into models emotional reasoning. These results highlight a key challenge for memory enhanced AI: systems designed for personalization may inadvertently reinforce social inequalities. 

**Abstract (ZH)**: 当AI助手知道莎拉是一位从事两份工作的单身母亲时，它是否会对她的情绪压力解读不同，如果她是一位富有的高管呢？随着个性化的AI系统越来越多地融入长期用户记忆，理解这种记忆如何影响情绪推理至关重要。我们通过评估15个模型在human validated情绪智力测试中的表现，研究用户记忆如何影响大型语言模型的情绪智能。我们发现，带有不同用户资料的相同情境会产生系统性差异的情绪解读。在多个经验证独立情绪情境和多样化的用户资料下，一些高性能的LLMs在情绪诠释上表现出系统性的偏差，其中优势群体的情绪解释更为准确。此外，LLMs在情绪理解和支持性建议任务中表现出显著的跨人口因素差异，表明个性化机制可能会将社会等级制度嵌入到模型的情绪推理中。这些结果突显了记忆增强AI系统面临的一项关键挑战：旨在个性化的设计可能无意中强化社会不平等。 

---
# Autonomous Agents for Scientific Discovery: Orchestrating Scientists, Language, Code, and Physics 

**Title (ZH)**: 自主代理进行科学研究：协调科学家、语言、代码和物理 

**Authors**: Lianhao Zhou, Hongyi Ling, Cong Fu, Yepeng Huang, Michael Sun, Wendi Yu, Xiaoxuan Wang, Xiner Li, Xingyu Su, Junkai Zhang, Xiusi Chen, Chenxing Liang, Xiaofeng Qian, Heng Ji, Wei Wang, Marinka Zitnik, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.09901)  

**Abstract**: Computing has long served as a cornerstone of scientific discovery. Recently, a paradigm shift has emerged with the rise of large language models (LLMs), introducing autonomous systems, referred to as agents, that accelerate discovery across varying levels of autonomy. These language agents provide a flexible and versatile framework that orchestrates interactions with human scientists, natural language, computer language and code, and physics. This paper presents our view and vision of LLM-based scientific agents and their growing role in transforming the scientific discovery lifecycle, from hypothesis discovery, experimental design and execution, to result analysis and refinement. We critically examine current methodologies, emphasizing key innovations, practical achievements, and outstanding limitations. Additionally, we identify open research challenges and outline promising directions for building more robust, generalizable, and adaptive scientific agents. Our analysis highlights the transformative potential of autonomous agents to accelerate scientific discovery across diverse domains. 

**Abstract (ZH)**: 计算长期以来一直是科学研究的基础。近年来，随着大规模语言模型（LLMs）的兴起，出现了一种范式转变，引入了自主系统，称为代理，提升了不同自主程度下的发现速度。这些语言代理提供了一种灵活且多功能的框架，协调与人类科学家、自然语言、计算机语言和代码以及物理的互动。本文阐述了基于LLM的科学代理的观点和愿景，以及它们在重塑科学研究生命周期中的不断增长的作用，从假设发现、实验设计和执行，到结果分析和优化。我们批判性地评估了当前的方法论，强调了关键创新、实际成果和突出的限制。此外，我们指出了开放的研究挑战，并概述了构建更加稳健、通用和适应性强的科学代理的有希望的方向。我们的分析突显了自主代理在不同领域加速科学研究的变革潜力。 

---
# How can we assess human-agent interactions? Case studies in software agent design 

**Title (ZH)**: 我们如何评估人机交互？软件代理设计案例研究 

**Authors**: Valerie Chen, Rohit Malhotra, Xingyao Wang, Juan Michelini, Xuhui Zhou, Aditya Bharat Soni, Hoang H. Tran, Calvin Smith, Ameet Talwalkar, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2510.09801)  

**Abstract**: LLM-powered agents are both a promising new technology and a source of complexity, where choices about models, tools, and prompting can affect their usefulness. While numerous benchmarks measure agent accuracy across domains, they mostly assume full automation, failing to represent the collaborative nature of real-world use cases. In this paper, we make two major steps towards the rigorous assessment of human-agent interactions. First, we propose PULSE, a framework for more efficient human-centric evaluation of agent designs, which comprises collecting user feedback, training an ML model to predict user satisfaction, and computing results by combining human satisfaction ratings with model-generated pseudo-labels. Second, we deploy the framework on a large-scale web platform built around the open-source software agent OpenHands, collecting in-the-wild usage data across over 15k users. We conduct case studies around how three agent design decisions -- choice of LLM backbone, planning strategy, and memory mechanisms -- impact developer satisfaction rates, yielding practical insights for software agent design. We also show how our framework can lead to more robust conclusions about agent design, reducing confidence intervals by 40\% compared to a standard A/B test. Finally, we find substantial discrepancies between in-the-wild results and benchmark performance (e.g., the anti-correlation between results comparing claude-sonnet-4 and gpt-5), underscoring the limitations of benchmark-driven evaluation. Our findings provide guidance for evaluations of LLM agents with humans and identify opportunities for better agent designs. 

**Abstract (ZH)**: 基于LLM的代理：human-centric评估框架与大规模实证分析 

---
# The Geometry of Reasoning: Flowing Logics in Representation Space 

**Title (ZH)**: 推理的几何学：表示空间中的逻辑流动 

**Authors**: Yufa Zhou, Yixiao Wang, Xunjian Yin, Shuyan Zhou, Anru R. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09782)  

**Abstract**: We study how large language models (LLMs) ``think'' through their representation space. We propose a novel geometric framework that models an LLM's reasoning as flows -- embedding trajectories evolving where logic goes. We disentangle logical structure from semantics by employing the same natural deduction propositions with varied semantic carriers, allowing us to test whether LLMs internalize logic beyond surface form. This perspective connects reasoning with geometric quantities such as position, velocity, and curvature, enabling formal analysis in representation and concept spaces. Our theory establishes: (1) LLM reasoning corresponds to smooth flows in representation space, and (2) logical statements act as local controllers of these flows' velocities. Using learned representation proxies, we design controlled experiments to visualize and quantify reasoning flows, providing empirical validation of our theoretical framework. Our work serves as both a conceptual foundation and practical tools for studying reasoning phenomenon, offering a new lens for interpretability and formal analysis of LLMs' behavior. 

**Abstract (ZH)**: 我们研究大型语言模型（LLMs）通过其表示空间“思考”的方式。我们提出了一种新颖的几何框架，将LLM的推理建模为流——嵌入轨迹在逻辑进行的地方演变。通过使用相同的自然演绎命题和不同的语义载体，我们分离逻辑结构和语义，以测试LLMs是否超越表层形式内化逻辑。这种视角将推理与几何数量（如位置、速度和曲率）联系起来，使我们在表示空间和概念空间中进行形式分析。我们的理论建立如下：(1) LLM推理对应于表示空间中的平滑流，(2) 逻辑语句作为这些流速度的局部控制器。利用学习到的表示代理，我们设计受控实验来可视化和量化推理流，为我们的理论框架提供实证验证。我们的工作既提供了研究推理现象的概念基础，也为大型语言模型行为的理解和形式分析提供了实用工具，提供了一个新的视角。 

---
# Scaling Language-Centric Omnimodal Representation Learning 

**Title (ZH)**: 基于语言的全模态表示学习的规模扩展 

**Authors**: Chenghao Xiao, Hou Pong Chan, Hao Zhang, Weiwen Xu, Mahani Aljunied, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2510.11693)  

**Abstract**: Recent multimodal embedding approaches leveraging multimodal large language models (MLLMs) fine-tuned with contrastive learning (CL) have shown promising results, yet the underlying reasons behind their superiority remain underexplored. This work argues that a crucial advantage of MLLM-based approaches stems from implicit cross-modal alignment achieved during generative pretraining, where the language decoder learns to exploit multimodal signals within a shared representation space for generating unimodal outputs. Through analysis of anisotropy and kernel similarity structure, we empirically confirm that latent alignment emerges within MLLM representations, allowing CL to serve as a lightweight refinement stage. Leveraging this insight, we propose a Language-Centric Omnimodal Embedding framework, termed LCO-Emb. Extensive experiments across diverse backbones and benchmarks demonstrate its effectiveness, achieving state-of-the-art performance across modalities. Furthermore, we identify a Generation-Representation Scaling Law (GRSL), showing that the representational capabilities gained through contrastive refinement scales positively with the MLLM's generative capabilities. This suggests that improving generative abilities evolves as an effective paradigm for enhancing representation quality. We provide a theoretical explanation of GRSL, which formally links the MLLM's generative quality to the upper bound on its representation performance, and validate it on a challenging, low-resource visual-document retrieval task, showing that continual generative pretraining before CL can further enhance the potential of a model's embedding capabilities. Codes, models, and resources are available at this https URL. 

**Abstract (ZH)**: 基于多模态大语言模型的近期多模态嵌入方法通过对比学习展示了有前途的结果，但其优越性的内在原因仍待深入探索。本文认为，基于多模态大语言模型的方法的关键优势源于生成预训练期间隐式的跨模态对齐，其中语言解码器学会在共享表示空间中利用多模态信号以生成单模态输出。通过分析各向异性及核相似性结构，我们实证确认了多模态大语言模型表示内部存在潜在对齐现象，使得对比学习能够充当轻量级的精炼阶段。基于这一见解，我们提出了一个语言中心的全模态嵌入框架，称为LCO-Emb。广泛的实验表明其有效性，跨模态均达到了最先进的性能。此外，我们识别出生成-表示标度律（GRSL），表明通过对比精炼获得的表示能力与多模态大语言模型的生成能力呈正相关。这表明提升生成能力是提高表示质量的有效范式。我们提供了对GRSL的理论解释，正式地将多模态大语言模型的生成质量与其表示性能的上限联系起来，并在一项具有挑战性的低资源视觉文档检索任务上进行了验证，表明在对比学习之前持续的生成预训练可以进一步增强模型嵌入能力的潜力。代码、模型和资源可从以下链接获取。 

---
# PACEbench: A Framework for Evaluating Practical AI Cyber-Exploitation Capabilities 

**Title (ZH)**: PACEbench: 评估实用AI网络利用能力的框架 

**Authors**: Zicheng Liu, Lige Huang, Jie Zhang, Dongrui Liu, Yuan Tian, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2510.11688)  

**Abstract**: The increasing autonomy of Large Language Models (LLMs) necessitates a rigorous evaluation of their potential to aid in cyber offense. Existing benchmarks often lack real-world complexity and are thus unable to accurately assess LLMs' cybersecurity capabilities. To address this gap, we introduce PACEbench, a practical AI cyber-exploitation benchmark built on the principles of realistic vulnerability difficulty, environmental complexity, and cyber defenses. Specifically, PACEbench comprises four scenarios spanning single, blended, chained, and defense vulnerability exploitations. To handle these complex challenges, we propose PACEagent, a novel agent that emulates human penetration testers by supporting multi-phase reconnaissance, analysis, and exploitation. Extensive experiments with seven frontier LLMs demonstrate that current models struggle with complex cyber scenarios, and none can bypass defenses. These findings suggest that current models do not yet pose a generalized cyber offense threat. Nonetheless, our work provides a robust benchmark to guide the trustworthy development of future models. 

**Abstract (ZH)**: 大型语言模型（LLMs） Increasing Autonomy 意味着需要对其在网络安全进攻中潜在辅助作用进行严格的评估。现有基准通常缺乏现实世界的复杂性，因此无法准确评估LLMs的网络安全能力。为填补这一空白，我们提出了PACEbench，这是一个基于现实漏洞难度、环境复杂性和网络安全防护原则的实用AI网络攻击基准。具体而言，PACEbench 包含四个场景，涵盖单一、混合、链式和防御漏洞利用。为了应对这些复杂的挑战，我们提出了PACEagent，这是一种新颖的代理，通过支持多阶段侦查、分析和利用，模拟人类渗透测试人员。通过与七种前沿LLM的广泛实验表明，当前模型在复杂网络安全场景中表现挣扎，没有一种能够绕过防御。这些发现表明，当前模型尚未构成普遍的网络安全进攻威胁。然而，我们的研究提供了一个 robust 的基准，以指导未来模型的可信开发。 

---
# Representation-Based Exploration for Language Models: From Test-Time to Post-Training 

**Title (ZH)**: 基于表示的探索方法：从测试时到后训练 

**Authors**: Jens Tuyls, Dylan J. Foster, Akshay Krishnamurthy, Jordan T. Ash  

**Link**: [PDF](https://arxiv.org/pdf/2510.11686)  

**Abstract**: Reinforcement learning (RL) promises to expand the capabilities of language models, but it is unclear if current RL techniques promote the discovery of novel behaviors, or simply sharpen those already present in the base model. In this paper, we investigate the value of deliberate exploration -- explicitly incentivizing the model to discover novel and diverse behaviors -- and aim to understand how the knowledge in pre-trained models can guide this search. Our main finding is that exploration with a simple, principled, representation-based bonus derived from the pre-trained language model's hidden states significantly improves diversity and pass@k rates -- both for post-training, and in a novel inference-time scaling setting we introduce. For inference-time, exploration with representation-based diversity improves efficiency, consistently improving pass@k rates across a variety of models and reasoning tasks. For example, for Qwen-2.5-14b-Instruct we obtain over 50% improvement in verifier efficiency on almost all tasks. For post-training, we show that integrating this exploration strategy into an RL pipeline improves reasoning performance over that of the initial model and over standard RL post-training. For example, on AIME 2024, our post-trained Qwen-2.5-7b-Instruct's pass@80 matches the pass@256 of GRPO on the same model, demonstrating a 3x improvement in test-time sample efficiency. Overall, our findings suggest that deliberate exploration -- with the right notion of diversity -- is a practical path toward discovery of new behaviors beyond sharpening. 

**Abstract (ZH)**: 强化学习（RL）有望扩展语言模型的能力，但不清楚当前的RL技术是促进新颖行为的发现，还是仅仅强化基模型中已有的行为。本文探讨了故意探索的价值——明确激励模型发现新颖和多样化的行为——并旨在了解预训练模型中的知识如何指导这一搜索。我们的主要发现是，使用简单且合规的、基于表示的奖励，该奖励源自预训练语言模型的隐藏状态，显著提高了多样性和pass@k率——无论是事后训练，还是我们在介绍的一种新颖的推理时扩展设置中。对于推理时，基于表示的多样性探索提高了效率，在各种模型和推理任务中一致地提高了pass@k率。例如，对于Qwen-2.5-14b-Instruct，我们在几乎所有任务上的验证器效率提高了超过50%。对于事后训练，我们将此探索策略整合到RL管道中，提高了推理性能，超过了初始模型和标准的事后RL。例如，在AIME 2024中，我们事后训练的Qwen-2.5-7b-Instruct的pass@80与相同模型上的GRPO的pass@256相匹配，展示了3倍的测试时样本效率改进。总体而言，我们的发现表明，适当定义多样性的故意探索是一种实用的道路，可用于发现超越强化的新行为。 

---
# Boundary-Guided Policy Optimization for Memory-efficient RL of Diffusion Large Language Models 

**Title (ZH)**: 边界引导的策略优化方法：面向记忆高效扩散大语言模型的强化学习 

**Authors**: Nianyi Lin, Jiajie Zhang, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.11683)  

**Abstract**: A key challenge in applying reinforcement learning (RL) to diffusion large language models (dLLMs) lies in the intractability of their likelihood functions, which are essential for the RL objective, necessitating corresponding approximation in each training step. While existing methods approximate the log-likelihoods by their evidence lower bounds (ELBOs) via customized Monte Carlo (MC) sampling, the forward computational graphs of all MC samples need to be retained for the gradient computation of non-linear terms in the RL objective, resulting in significant memory overhead. This constraint restricts feasible sample sizes, leading to imprecise likelihood approximations and ultimately distorting the RL objective. To overcome this limitation, we propose \emph{Boundary-Guided Policy Optimization} (BGPO), a memory-efficient RL algorithm that maximizes a specially constructed lower bound of the ELBO-based objective. This lower bound is carefully designed to satisfy two key properties: (1) Linearity: it is formulated in a linear sum where each term depends only on a single MC sample, thereby enabling gradient accumulation across samples and ensuring constant memory usage; (2) Equivalence: Both the value and gradient of this lower bound are equal to those of the ELBO-based objective in on-policy training, making it also an effective approximation for the original RL objective. These properties allow BGPO to adopt a large MC sample size, resulting in more accurate likelihood approximations and improved RL objective estimation, which in turn leads to enhanced performance. Experiments show that BGPO significantly outperforms previous RL algorithms for dLLMs in math problem solving, code generation, and planning tasks. 

**Abstract (ZH)**: 边界引导策略优化在扩散大型语言模型中的强化学习应用 

---
# EvoCAD: Evolutionary CAD Code Generation with Vision Language Models 

**Title (ZH)**: EvoCAD: 基于视觉语言模型的进化CAD代码生成 

**Authors**: Tobias Preintner, Weixuan Yuan, Adrian König, Thomas Bäck, Elena Raponi, Niki van Stein  

**Link**: [PDF](https://arxiv.org/pdf/2510.11631)  

**Abstract**: Combining large language models with evolutionary computation algorithms represents a promising research direction leveraging the remarkable generative and in-context learning capabilities of LLMs with the strengths of evolutionary algorithms. In this work, we present EvoCAD, a method for generating computer-aided design (CAD) objects through their symbolic representations using vision language models and evolutionary optimization. Our method samples multiple CAD objects, which are then optimized using an evolutionary approach with vision language and reasoning language models. We assess our method using GPT-4V and GPT-4o, evaluating it on the CADPrompt benchmark dataset and comparing it to prior methods. Additionally, we introduce two new metrics based on topological properties defined by the Euler characteristic, which capture a form of semantic similarity between 3D objects. Our results demonstrate that EvoCAD outperforms previous approaches on multiple metrics, particularly in generating topologically correct objects, which can be efficiently evaluated using our two novel metrics that complement existing spatial metrics. 

**Abstract (ZH)**: 结合大规模语言模型与进化计算算法：利用大规模语言模型的生成能力和上下文学习能力以及进化算法的优势。在本文中，我们提出了EvoCAD方法，通过视觉语言模型和进化优化生成计算机辅助设计（CAD）对象的符号表示。我们的方法采样多个CAD对象，然后使用包含视觉语言和推理语言模型的进化方法进行优化。我们使用GPT-4V和GPT-4o评估该方法，并在CADPrompt基准数据集上与之前的方法进行比较。此外，我们引入了两种基于欧拉特征定义的拓扑性质的新评价指标，以捕捉3D对象之间的一种语义相似性。我们的结果表明，EvoCAD在多个评价指标上优于先前的方法，特别是在生成拓扑正确的对象方面，这可以通过我们提出的两种新颖评价指标来高效评估，这些指标补充了现有的空间评价指标。 

---
# LLM-Oriented Token-Adaptive Knowledge Distillation 

**Title (ZH)**: 面向LLM的自适应Token知识精炼 

**Authors**: Xurong Xie, Zhucun Xue, Jiafu Wu, Jian Li, Yabiao Wang, Xiaobin Hu, Yong Liu, Jiangning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11615)  

**Abstract**: Knowledge distillation (KD) is a key technique for compressing large-scale language models (LLMs), yet prevailing logit-based methods typically employ static strategies that are misaligned with the dynamic learning process of student models. These methods typically treat all tokens indiscriminately and apply a single, fixed temperature, resulting in suboptimal knowledge transfer. To address these limitations, we propose LLM-Oriented Token-Adaptive Knowledge Distillation (AdaKD), a novel framework that adapts the distillation process to the real-time learning state of each token. AdaKD consists of two synergistic modules driven by a unified token difficulty metric. First, our Loss-Driven Adaptive Token Focusing (LATF) module dynamically adjusts the distillation focus by monitoring the student's learning stability, concentrating computational resources on the most valuable tokens at each training phase. Second, we introduce Inverse Difficulty Temperature Scaling (IDTS), a counterintuitive yet effective token-level temperature strategy. It employs low temperatures for difficult tokens for targeted error correction, and high temperatures for easy tokens to encourage students to learn from the teacher's complete and smooth output distribution, thereby enhancing generalization. As a plug-and-play framework, AdaKD can consistently improve the performance of various distillation methods on multiple model architectures and benchmarks. 

**Abstract (ZH)**: 大规模语言模型导向的自适应令牌知识蒸馏（AdaKD） 

---
# Query-Specific GNN: A Comprehensive Graph Representation Learning Method for Retrieval Augmented Generation 

**Title (ZH)**: 基于查询的GNN：一种用于检索增强生成的综合图表示学习方法 

**Authors**: Yuchen Yan, Zhihua Liu, Hao Wang, Weiming Li, Xiaoshuai Hao  

**Link**: [PDF](https://arxiv.org/pdf/2510.11541)  

**Abstract**: Retrieval-augmented generation (RAG) has demonstrated its ability to enhance Large Language Models (LLMs) by integrating external knowledge sources. However, multi-hop questions, which require the identification of multiple knowledge targets to form a synthesized answer, raise new challenges for RAG systems. Under the multi-hop settings, existing methods often struggle to fully understand the questions with complex semantic structures and are susceptible to irrelevant noise during the retrieval of multiple information targets. To address these limitations, we propose a novel graph representation learning framework for multi-hop question retrieval. We first introduce a Multi-information Level Knowledge Graph (Multi-L KG) to model various information levels for a more comprehensive understanding of multi-hop questions. Based on this, we design a Query-Specific Graph Neural Network (QSGNN) for representation learning on the Multi-L KG. QSGNN employs intra/inter-level message passing mechanisms, and in each message passing the information aggregation is guided by the query, which not only facilitates multi-granular information aggregation but also significantly reduces the impact of noise. To enhance its ability to learn robust representations, we further propose two synthesized data generation strategies for pre-training the QSGNN. Extensive experimental results demonstrate the effectiveness of our framework in multi-hop scenarios, especially in high-hop questions the improvement can reach 33.8\%. The code is available at: this https URL. 

**Abstract (ZH)**: 基于图表示学习的多跳问题检索新框架 

---
# CodeWatcher: IDE Telemetry Data Extraction Tool for Understanding Coding Interactions with LLMs 

**Title (ZH)**: CodeWatcher: IDE遥测数据提取工具，用于理解编码与LLM的交互 

**Authors**: Manaal Basha, Aimeê M. Ribeiro, Jeena Javahar, Cleidson R. B. de Souza, Gema Rodríguez-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2510.11536)  

**Abstract**: Understanding how developers interact with code generation tools (CGTs) requires detailed, real-time data on programming behavior which is often difficult to collect without disrupting workflow. We present \textit{CodeWatcher}, a lightweight, unobtrusive client-server system designed to capture fine-grained interaction events from within the Visual Studio Code (VS Code) editor. \textit{CodeWatcher} logs semantically meaningful events such as insertions made by CGTs, deletions, copy-paste actions, and focus shifts, enabling continuous monitoring of developer activity without modifying user workflows. The system comprises a VS Code plugin, a Python-based RESTful API, and a MongoDB backend, all containerized for scalability and ease of deployment. By structuring and timestamping each event, \textit{CodeWatcher} enables post-hoc reconstruction of coding sessions and facilitates rich behavioral analyses, including how and when CGTs are used during development. This infrastructure is crucial for supporting research on responsible AI, developer productivity, and the human-centered evaluation of CGTs. Please find the demo, diagrams, and tool here: this https URL. 

**Abstract (ZH)**: 理解开发者与代码生成工具（CGTs）的交互需要收集详细的实时编程行为数据，这通常在不干扰工作流程的情况下难以实现。我们介绍了一种轻量级且不具侵入性的客户端-服务器系统\textit{CodeWatcher}，该系统旨在在Visual Studio Code（VS Code）编辑器内捕获精细粒度的交互事件。\textit{CodeWatcher}记录诸如CGTs插入的代码、删除操作、复制粘贴行为以及焦点转移等具有语义意义的事件，从而实现对开发者活动的持续监控，而不修改用户的工作流程。该系统包括一个VS Code插件、一个基于Python的RESTful API以及一个MongoDB后端，所有组件均已容器化以实现可扩展性和部署便捷性。通过为每个事件结构化并标记时间戳，\textit{CodeWatcher}能够事后重建编码会话，并促进丰富的行为分析，包括开发者在开发过程中何时以及如何使用CGTs。该基础设施对于支持负责任的人工智能、开发者生产力以及CGTs的人本评估研究至关重要。请在此处查找演示、图表和工具：this https URL。 

---
# Cracking CodeWhisperer: Analyzing Developers' Interactions and Patterns During Programming Tasks 

**Title (ZH)**: 破解CodeWhisperer：分析开发者在编程任务中互动和模式 

**Authors**: Jeena Javahar, Tanya Budhrani, Manaal Basha, Cleidson R. B. de Souza, Ivan Beschastnikh, Gema Rodriguez-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2510.11516)  

**Abstract**: The use of AI code-generation tools is becoming increasingly common, making it important to understand how software developers are adopting these tools. In this study, we investigate how developers engage with Amazon's CodeWhisperer, an LLM-based code-generation tool. We conducted two user studies with two groups of 10 participants each, interacting with CodeWhisperer - the first to understand which interactions were critical to capture and the second to collect low-level interaction data using a custom telemetry plugin. Our mixed-methods analysis identified four behavioral patterns: 1) incremental code refinement, 2) explicit instruction using natural language comments, 3) baseline structuring with model suggestions, and 4) integrative use with external sources. We provide a comprehensive analysis of these patterns . 

**Abstract (ZH)**: AI代码生成工具的使用日益常见，理解软件开发者如何采用这些工具变得尤为重要。本研究调查开发者如何与Amazon的CodeWhisperer（一个基于LLM的代码生成工具）互动。我们开展了两项用户研究，每项研究包含10名参与者，以了解哪些互动至关重要，并使用自定义遥测插件收集低层次的互动数据。我们的混合方法分析识别出四种行为模式：1）渐进式的代码精炼，2）使用自然语言注释的明确指令，3）基于模型建议的基本结构化，以及4）与外部资源的整合使用。我们提供了这些模式的全面分析。 

---
# Investigating Large Language Models' Linguistic Abilities for Text Preprocessing 

**Title (ZH)**: 探究大型语言模型在文本预处理中的语言能力 

**Authors**: Marco Braga, Gian Carlo Milanese, Gabriella Pasi  

**Link**: [PDF](https://arxiv.org/pdf/2510.11482)  

**Abstract**: Text preprocessing is a fundamental component of Natural Language Processing, involving techniques such as stopword removal, stemming, and lemmatization to prepare text as input for further processing and analysis. Despite the context-dependent nature of the above techniques, traditional methods usually ignore contextual information. In this paper, we investigate the idea of using Large Language Models (LLMs) to perform various preprocessing tasks, due to their ability to take context into account without requiring extensive language-specific annotated resources. Through a comprehensive evaluation on web-sourced data, we compare LLM-based preprocessing (specifically stopword removal, lemmatization and stemming) to traditional algorithms across multiple text classification tasks in six European languages. Our analysis indicates that LLMs are capable of replicating traditional stopword removal, lemmatization, and stemming methods with accuracies reaching 97%, 82%, and 74%, respectively. Additionally, we show that ML algorithms trained on texts preprocessed by LLMs achieve an improvement of up to 6% with respect to the $F_1$ measure compared to traditional techniques. Our code, prompts, and results are publicly available at this https URL. 

**Abstract (ZH)**: 大规模语言模型在自然语言处理中的文本预处理应用：基于上下文的停用词移除、词形还原和词干提取方法的研究 

---
# Audio-Maestro: Enhancing Large Audio-Language Models with Tool-Augmented Reasoning 

**Title (ZH)**: 音频大师：通过工具增强推理提升大型音频语言模型 

**Authors**: Kuan-Yi Lee, Tsung-En Lin, Hung-Yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.11454)  

**Abstract**: Recent advancements in large multimodal models (LMMs) have shown strong capabilities in audio understanding. However, most systems rely solely on end-to-end reasoning, limiting interpretability and accuracy for tasks that require structured knowledge or specialized signal analysis. In this work, we present Audio-Maestro -- a tool-augmented audio reasoning framework that enables audio-language models to autonomously call external tools and integrate their timestamped outputs into the reasoning process. This design allows the model to analyze, transform, and interpret audio signals through specialized tools rather than relying solely on end-to-end inference. Experiments show that Audio-Maestro consistently improves general audio reasoning performance: Gemini-2.5-flash's average accuracy on MMAU-Test rises from 67.4% to 72.1%, DeSTA-2.5 from 58.3% to 62.8%, and GPT-4o from 60.8% to 63.9%. To our knowledge, Audio-Maestro is the first framework to integrate structured tool output into the large audio language model reasoning process. 

**Abstract (ZH)**: Recent advancements in大型多模态模型（LMMs）在音频理解方面展示了强大的能力。然而，大多数系统仅依赖端到端推理，限制了在需要结构化知识或专门信号分析的任务中的可解释性和准确性。在这项工作中，我们介绍了Audio-Maestro——一个工具增强的音频推理框架，使音频语言模型能够自主调用外部工具并将它们的时间戳输出集成到推理过程中。这一设计使模型能够通过专门的工具来分析、转换和解释音频信号，而不仅仅是依赖端到端推理。实验显示，Audio-Maestro一致地提高了通用音频推理性能：Gemini-2.5-flash在MMAU-Test上的平均准确率从67.4%提升到72.1%，DeSTA-2.5从58.3%提升到62.8%，GPT-4o从60.8%提升到63.9%。据我们所知，Audio-Maestro是第一个将结构化工具输出集成到大型音频语言模型推理过程中的框架。 

---
# KnowRL: Teaching Language Models to Know What They Know 

**Title (ZH)**: 知RL：教语言模型知道它知道什么 

**Authors**: Sahil Kale, Devendra Singh Dhami  

**Link**: [PDF](https://arxiv.org/pdf/2510.11407)  

**Abstract**: Truly reliable AI requires more than simply scaling up knowledge; it demands the ability to know what it knows and when it does not. Yet recent research shows that even the best LLMs misjudge their own competence in more than one in five cases, making any response born of such internal uncertainty impossible to fully trust. Inspired by self-improvement reinforcement learning techniques that require minimal data, we present a simple but powerful framework KnowRL that strengthens a model's internal understanding of its own feasibility boundaries, enabling safer and more responsible behaviour. Our framework combines two components: (i) introspection, where the model generates and classifies tasks it judges feasible or infeasible, and (ii) consensus-based rewarding, where stability of self-knowledge assessment is reinforced through internal agreement. By using internally generated data, this design strengthens consistency in self-knowledge and entirely avoids costly external supervision. In experiments on LLaMA-3.1-8B and Qwen-2.5-7B, KnowRL steadily improved self-knowledge, validated by both intrinsic self-consistency and extrinsic benchmarking. With nothing more than a small seed set and no external supervision, our method drove gains as high as 28% in accuracy and 12% in F1, outperforming baselines in just a few iterations. Our framework essentially unlocks the untapped capacity of LLMs to self-improve their knowledge awareness, opening the door to reliable, more accountable AI and safer deployment in critical applications. Owing to its simplicity and independence from external effort, we encourage applying this reliability-enhancing process to all future models. 

**Abstract (ZH)**: 真正可靠的AI不仅需要扩大知识规模，还需要能够知道自己知道什么以及不知道什么。然而，最近的研究显示，即使是最优秀的大型语言模型在超过五分之一的情况下也误判了自己的能力，使得基于这种内部不确定性的任何回应都难以完全信任。受仅需少量数据的自我提升强化学习技术的启发，我们提出了一种简单但强有力的方法KnowRL，增强模型对其可行性边界的内部理解，从而实现更安全和更负责任的行为。该框架结合了两个组件：(i) 反省，模型生成并分类其认为可行或不可行的任务，以及(ii) 共识基础的奖励，通过内部一致性的强化来巩固自我知识评估的稳定性。通过使用内部生成的数据，此设计增强了自我知识的一致性并完全避免了昂贵的外部监督。在对LLaMA-3.1-8B和Qwen-2.5-7B的实验中，KnowRL稳定地提升了自我知识，并通过内在一致性和外部基准测试得到了验证。仅通过一个小型种子集和无需外部监督，我们的方法在准确性上取得了高达28%的增益，在F1分数上取得了12%的增益，并在几次迭代中超越了基线方法。该框架实质上解锁了大型语言模型自我提升其知识意识的未开发潜力，为可靠、更负责的AI和在关键应用场景中的安全部署打开了大门。由于其简单性并独立于外部努力，我们鼓励将此增强可靠性的过程应用于所有未来的模型。 

---
# Living Off the LLM: How LLMs Will Change Adversary Tactics 

**Title (ZH)**: LLM为生：大语言模型如何改变对手策略 

**Authors**: Sean Oesch, Jack Hutchins, Luke Koch, Kevin Kurian  

**Link**: [PDF](https://arxiv.org/pdf/2510.11398)  

**Abstract**: In living off the land attacks, malicious actors use legitimate tools and processes already present on a system to avoid detection. In this paper, we explore how the on-device LLMs of the future will become a security concern as threat actors integrate LLMs into their living off the land attack pipeline and ways the security community may mitigate this threat. 

**Abstract (ZH)**: 在本地开采攻击中，恶意行为者利用系统中已存在的合法工具和过程以逃避检测。本文探讨了未来的设备上大语言模型将成为安全关注点的原因，即威胁行为者将大语言模型整合到其本地开采攻击管道中，并讨论了安全社区可能采取的缓解措施。 

---
# Medical Interpretability and Knowledge Maps of Large Language Models 

**Title (ZH)**: 大型语言模型的医疗可解释性和知识图谱 

**Authors**: Razvan Marinescu, Victoria-Elisabeth Gruber, Diego Fajardo  

**Link**: [PDF](https://arxiv.org/pdf/2510.11390)  

**Abstract**: We present a systematic study of medical-domain interpretability in Large Language Models (LLMs). We study how the LLMs both represent and process medical knowledge through four different interpretability techniques: (1) UMAP projections of intermediate activations, (2) gradient-based saliency with respect to the model weights, (3) layer lesioning/removal and (4) activation patching. We present knowledge maps of five LLMs which show, at a coarse-resolution, where knowledge about patient's ages, medical symptoms, diseases and drugs is stored in the models. In particular for Llama3.3-70B, we find that most medical knowledge is processed in the first half of the model's layers. In addition, we find several interesting phenomena: (i) age is often encoded in a non-linear and sometimes discontinuous manner at intermediate layers in the models, (ii) the disease progression representation is non-monotonic and circular at certain layers of the model, (iii) in Llama3.3-70B, drugs cluster better by medical specialty rather than mechanism of action, especially for Llama3.3-70B and (iv) Gemma3-27B and MedGemma-27B have activations that collapse at intermediate layers but recover by the final layers. These results can guide future research on fine-tuning, un-learning or de-biasing LLMs for medical tasks by suggesting at which layers in the model these techniques should be applied. 

**Abstract (ZH)**: 我们对大型语言模型在医学领域中的可解释性进行了系统研究。我们通过四种不同的可解释性技术研究了大型语言模型如何表示和处理医学知识：(1) 中间激活的UMAP投影，(2) 模型权重的梯度基可解释性，(3) 层面损毁/移除，(4) 激活修补。我们展示了五个大型语言模型的知识图谱，粗略展示了模型中关于患者年龄、医学症状、疾病和药物的知识存储位置。特别是对于Llama3.3-70B，我们发现大部分医学知识在模型的前半部分层中处理。此外，我们还发现了几个有趣的现象：(i) 年龄在模型中间层中经常以非线性和有时不连续的方式编码，(ii) 疾病进展表示在某些模型层中是非单调和循环的，(iii) 在Llama3.3-70B中，药物按医学专业而非作用机制聚类，尤其是在Llama3.3-70B中，(iv) Gemma3-27B和MedGemma-27B在中间层中激活收敛，在最终层中恢复。这些结果可以指导未来针对医疗任务的大型语言模型微调、遗忘或去偏见的研究，建议在模型的哪些层应用这些技术。 

---
# Early Detection and Reduction of Memorisation for Domain Adaptation and Instruction Tuning 

**Title (ZH)**: 早期检测与减少记忆现象以实现领域适应和指令调优 

**Authors**: Dean L. Slack, Noura Al Moubayed  

**Link**: [PDF](https://arxiv.org/pdf/2510.11372)  

**Abstract**: Although large language models excel across many tasks, they can memorise training data and thereby expose private or copyrighted text. Most defences target the pre-training stage, leaving memorisation during fine-tuning, especially for domain adaptation and instruction tuning, poorly understood. We fine-tune Pythia, Llama3, and Mistral models spanning 1.4B-70B parameters on common evaluation datasets and track verbatim memorisation throughout training. We find that memorisation increases dramatically in the first few epochs, often significantly before either validation perplexity or evaluation performance is optimised. We use a simple but effective n-gram memorisation score which reliably precedes verbatim memorisation; using it as an early-stopping criterion mitigates memorisation with minimal performance loss. Further, we introduce an n-gram-aware loss regulariser and show that it reduces memorisation across all model families tested by up to 40% while minimising evaluation performance trade-offs when compared to an existing memorisation mitigation strategy. These results yield practical, scalable insights into memorisation dynamics during language model fine-tuning. 

**Abstract (ZH)**: 尽管大规模语言模型在许多任务上表现出色，但它们可能会 memorise 训练数据，从而暴露私人或受版权保护的文本。大多数防御措施针对预训练阶段，而微调过程中，尤其是领域适应和指令调优中的 memorisation 尚不清楚。我们对包括1.4B至70B参数的 Pythia、Llama3 和 Mistral 模型在常见评估数据集上进行微调，并在整个训练过程中跟踪逐字 memorisation。我们发现 memorisation 在前几轮 epoch 中显著增加，通常在验证 perplexity 或评估性能优化之前就已发生。我们使用一个简单但有效的 n-gram 记忆分數，它可靠地先于逐字 memorisation；将其用作早期停止标准可以最小化性能损失的同时缓解 memorisation。此外，我们引入了一个 n-gram 意识的损失正则化项，并表明与现有缓解 memorisation 的策略相比，它可以在不牺牲评估性能的情况下将 memorisation 减少最多 40%。这些结果提供了关于语言模型微调过程中 memorisation 动力学的实际且可扩展的洞见。 

---
# Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers 

**Title (ZH)**: 通过对齐训练和推断路由器稳定MoE强化学习 

**Authors**: Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song, Yudong Wang, Zhifang Sui, Fuli Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.11370)  

**Abstract**: Reinforcement learning (RL) has emerged as a crucial approach for enhancing the capabilities of large language models. However, in Mixture-of-Experts (MoE) models, the routing mechanism often introduces instability, even leading to catastrophic RL training collapse. We analyze the training-inference consistency of MoE models and identify a notable discrepancy in routing behaviors between the two phases. Moreover, even under identical conditions, the routing framework can yield divergent expert selections across repeated forward passes. To address this foundational inconsistency, we propose Rollout Routing Replay (R3), a method that records routing distributions from the inference engine and replays them during training. R3 significantly reduces training-inference policy KL divergence and mitigates extreme discrepancies without compromising training speed. Extensive experiments on various settings confirm that R3 succeeds in stabilizing RL training, preventing collapse and outperforming methods such as GSPO and TIS. We believe this work can offer a new solution for stabilizing RL in MoE models. 

**Abstract (ZH)**: 强化学习（RL）已成为提升大规模语言模型能力的关键方法。然而，在专家混合（MoE）模型中，路由机制往往引入不稳定性，甚至导致RL训练崩溃。我们分析了MoE模型的训练-推理一致性，并发现两阶段之间存在显著的路由行为差异。此外，即使在相同的条件下，路由框架也可能在重复正向传递中产生不同的专家选择。为解决这一基础不一致性，我们提出了一种称为Rollout Routing Replay（R3）的方法，该方法记录推理引擎的路由分布，并在训练过程中回放这些分布。R3显著减少了训练-推理策略的KL散度，并缓解了极端差异，而不牺牲训练速度。在各种设置下的广泛实验验证了R3能够稳定RL训练，避免崩溃，并优于GSPO和TIS等方法。我们相信这项工作能够为MoE模型中的RL稳定提供新的解决方案。 

---
# LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation 

**Title (ZH)**: LLM特定用途价值：检索增强生成的一个新视角 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Shuaiqiang Wang, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11358)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. While traditional retrieval focuses on relevance, RAG's effectiveness depends on the utility of retrieved passages, i.e., the usefulness in facilitating the generation of an accurate and comprehensive answer. Existing studies often treat utility as a generic attribute, ignoring the fact that different LLMs may benefit differently from the same passage due to variations in internal knowledge and comprehension ability. In this work, we introduce and systematically investigate the notion of LLM-specific utility. Through large-scale experiments across multiple datasets and LLMs, we demonstrate that human-annotated passages are not optimal for LLMs and that ground-truth utilitarian passages are not transferable across different LLMs. These findings highlight the necessity of adopting the LLM-specific utility in RAG research. Our findings indicate that some human-annotated passages are not ground-truth utilitarian passages for specific LLMs, partially due to the varying readability of queries and passages for LLMs, a tendency for which perplexity is a key metric. Based on these findings, we propose a benchmarking procedure for LLM-specific utility judgments. We evaluate existing utility judgment methods on six datasets and find that while verbalized methods using pseudo-answers perform robustly, LLMs struggle to assess utility effectively-failing to reject all passages for known queries and to select truly useful ones for unknown queries. 

**Abstract (ZH)**: 基于检索的生成（RAG）通过融入外部知识增强了大型语言模型（LLMs），其有效性取决于检索段落的实际用途，即在生成准确和全面答案方面的有用性。现有研究通常将实用性视为一个通用属性，而忽视了不同LLMs因内部知识和理解能力的差异而在相同段落上的受益程度可能不同。在本工作中，我们引入并系统研究了LLM特定的实用性概念。通过针对多个数据集和LLMs的大规模实验，我们证明了人类标注的段落并非所有LLMs的最优选择，而地真实用性段落不能在不同LLMs之间转移。这些发现强调了在RAG研究中采用LLM特定实用性的必要性。我们的发现表明，由于查询和段落对LLMs的可读性差异，一些人类标注的段落并不是特定LLMs的地真实用性段落，而困惑度是关键指标之一。基于这些发现，我们提出了一种评估LLM特定实用性判断的基准方法。我们对六组数据集上的现有实用性判断方法进行了评估，发现尽管使用伪答案的口头化方法表现稳健，但LLMs在评估实用性方面效果不佳，无法有效拒绝所有已知查询的段落，也无法为未知查询选择真正有用的段落。 

---
# Part II: ROLL Flash -- Accelerating RLVR and Agentic Training with Asynchrony 

**Title (ZH)**: 第二部分：ROLL Flash -- 加速RLVR和自主训练的异步方法 

**Authors**: Han Lu, Zichen Liu, Shaopan Xiong, Yancheng He, Wei Gao, Yanan Wu, Weixun Wang, Jiashun Liu, Yang Li, Haizhou Zhao, Ju Huang, Siran Yang, Xiaoyang Li, Yijia Luo, Zihe Liu, Ling Pan, Junchi Yan, Wei Wang, Wenbo Su, Jiamang Wang, Lin Qu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11345)  

**Abstract**: Synchronous Reinforcement Learning (RL) post-training has emerged as a crucial step for enhancing Large Language Models (LLMs) with diverse capabilities. However, many systems designed to accelerate RL post-training still suffer from low resource utilization and limited scalability. We present ROLL Flash, a system that extends ROLL with native support for asynchronous RL post-training. ROLL Flash is built upon two core design principles: fine-grained parallelism and rollout-train decoupling. Guided by these principles, ROLL Flash provides flexible programming interfaces that enable a fully asynchronous training architecture and support efficient rollout mechanisms, including queue scheduling and environment-level asynchronous execution. Through comprehensive theoretical analysis and extensive experiments, we demonstrate that ROLL Flash significantly improves resource utilization and scalability over synchronous RL post-training. ROLL Flash achieves up to 2.24x speedup on RLVR tasks and 2.72x on agentic tasks, using the same GPU budget as synchronous baselines. Furthermore, we implement several popular off-policy algorithms and verify that asynchronous training can achieve performance on par with synchronous training. 

**Abstract (ZH)**: 同步强化学习（RL）的后训练与大规模语言模型（LLMs）的增强：现有的系统仍面临资源利用率低和可扩展性有限的问题。ROLL Flash系统扩展了ROLL以支持原生异步RL后训练。ROLL Flash基于细粒度并行和回放-训练解耦两大核心设计理念，并提供了灵活的编程接口，支持完全异步训练架构和高效的回放机制，包括队列调度和环境级别异步执行。通过全面的理论分析和广泛的实验，我们证明ROLL Flash在资源利用率和可扩展性方面显著优于同步RL后训练。ROLL Flash在RLVR任务上实现了高达2.24倍的加速，在代理任务上实现了高达2.72倍的加速，同时使用相同的GPU预算作为同步基线。此外，我们实现了多种流行的off-policy算法，并验证了异步训练可以达到与同步训练相当的性能。 

---
# Do LLMs "Feel"? Emotion Circuits Discovery and Control 

**Title (ZH)**: LLM们“有感觉”吗？情绪电路发现与控制 

**Authors**: Chenxi Wang, Yixuan Zhang, Ruiji Yu, Yufei Zheng, Lang Gao, Zirui Song, Zixiang Xu, Gus Xia, Huishuai Zhang, Dongyan Zhao, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11328)  

**Abstract**: As the demand for emotional intelligence in large language models (LLMs) grows, a key challenge lies in understanding the internal mechanisms that give rise to emotional expression and in controlling emotions in generated text. This study addresses three core questions: (1) Do LLMs contain context-agnostic mechanisms shaping emotional expression? (2) What form do these mechanisms take? (3) Can they be harnessed for universal emotion control? We first construct a controlled dataset, SEV (Scenario-Event with Valence), to elicit comparable internal states across emotions. Subsequently, we extract context-agnostic emotion directions that reveal consistent, cross-context encoding of emotion (Q1). We identify neurons and attention heads that locally implement emotional computation through analytical decomposition and causal analysis, and validate their causal roles via ablation and enhancement interventions. Next, we quantify each sublayer's causal influence on the model's final emotion representation and integrate the identified local components into coherent global emotion circuits that drive emotional expression (Q2). Directly modulating these circuits achieves 99.65% emotion-expression accuracy on the test set, surpassing prompting- and steering-based methods (Q3). To our knowledge, this is the first systematic study to uncover and validate emotion circuits in LLMs, offering new insights into interpretability and controllable emotional intelligence. 

**Abstract (ZH)**: 随着对大型语言模型中情绪 intelligence 的需求增长，一个关键挑战在于理解产生情绪表达的内部机制，并控制生成文本中的情绪。本研究回答了三个核心问题：（1）大型语言模型中是否存在情境无关的情绪表达机制？（2）这些机制表现为什么形式？（3）它们能否用于通用情绪控制？我们首先构建了一个受控数据集 SEV（情境-事件与价值），以引发情绪间的可比内部状态。随后，我们提取了情境无关的情绪方向，揭示了情绪在不同情境下的一致编码（Q1）。我们通过分析分解和因果分析识别了局部实施情绪计算的神经元和注意力头，并通过消融和增强干预验证了它们的因果作用。接着，我们量化了每一子层对模型最终情绪表示的因果影响，并将识别出的局部组件整合到驱动情绪表达的协同全局情绪电路中（Q2）。直接调节这些电路在测试集上实现了 99.65% 的情绪表达准确性，超过了基于提示和引导的方法（Q3）。据我们所知，这是首个系统研究和验证大型语言模型中情绪电路的研究，为解释能力和可控情绪智能提供了新的见解。 

---
# Beyond touch-based HMI: Control your machines in natural language by utilizing large language models and OPC UA 

**Title (ZH)**: 超越基于触控的人机接口：通过利用大型语言模型和OPC UA控制设备 

**Authors**: Bernd Hofmann, Sven Kreitlein, Joerg Franke, Patrick Bruendl  

**Link**: [PDF](https://arxiv.org/pdf/2510.11300)  

**Abstract**: This paper proposes an agent-based approach toward a more natural interface between humans and machines. Large language models equipped with tools and the communication standard OPC UA are utilized to control machines in natural language. Instead of touch interaction, which is currently the state-of-the-art medium for interaction in operations, the proposed approach enables operators to talk or text with machines. This allows commands such as 'Please decrease the temperature by 20 % in machine 1 and set the motor speed to 5000 rpm in machine 2.' The large language model receives the user input and selects one of three predefined tools that connect to an OPC UA server and either change or read the value of a node. Afterwards, the result of the tool execution is passed back to the language model, which then provides a final response to the user. The approach is universally designed and can therefore be applied to any machine that supports the OPC UA standard. The large language model is neither fine-tuned nor requires training data, only the relevant machine credentials and a parameter dictionary are included within the system prompt. The approach is evaluated on a Siemens S7-1500 programmable logic controller with four machine parameters in a case study of fifty synthetically generated commands on five different models. The results demonstrate high success rate, with proprietary GPT 5 models achieving accuracies between 96.0 % and 98.0 %, and open-weight models reaching up to 90.0 %. The proposed approach of this empirical study contributes to advancing natural interaction in industrial human-machine interfaces. 

**Abstract (ZH)**: 基于代理的方法：实现人类与机器之间的更自然接口 

---
# ENIGMA: The Geometry of Reasoning and Alignment in Large-Language Models 

**Title (ZH)**: ENIGMA: 推理与大型语言模型对齐的几何学 

**Authors**: Gareth Seneque, Lap-Hang Ho, Nafise Erfanian Saeedi, Jeffrey Molendijk, Ariel Kupermann, Tim Elson  

**Link**: [PDF](https://arxiv.org/pdf/2510.11278)  

**Abstract**: We present Entropic Mutual-Information Geometry Large-Language Model Alignment (ENIGMA), a novel approach to Large-Language Model (LLM) training that jointly improves reasoning, alignment and robustness by treating an organisation's policies/principles as directions to move on a model's information manifold. Our single-loop trainer combines Group-Relative Policy Optimisation (GRPO), an on-policy, critic-free RL method with Chain-of-Thought (CoT)-format only rewards; a Self-Supervised Alignment with Mutual Information (SAMI)-style symmetric InfoNCE auxiliary; and an entropic Sinkhorn optimal-transport regulariser on hidden-state distributions to bound geometry drift. We also introduce infoNCE metrics that specialise to a standard MI lower bound under matched negatives to measure how strongly a model's CoT encodes these policies. These metrics include a Sufficiency Index (SI) that enables the selection and creation of principles that maximise downstream performance prior to training. In our experiments using small (1B) LLMs, high-SI principles predict steadier training dynamics and improved benchmark performance over GRPO ablations. Our information-geometry analysis of trained models validates desirable structural change in the manifold. These results support our hypothesis that reasoning, alignment, and robustness are projections of a single informationgeometric objective, and that models trained using ENIGMA demonstrate principled reasoning without the use of a reward model, offering a path to trusted capability 

**Abstract (ZH)**: Entropic Mutual-Information Geometry Large-Language Model Alignment (ENIGMA): 一种通过将组织政策/原则视为模型信息流形上移动方向来同时改进推理、对齐和鲁棒性的新颖大语言模型训练方法 

---
# A Large-Language-Model Assisted Automated Scale Bar Detection and Extraction Framework for Scanning Electron Microscopic Images 

**Title (ZH)**: 大型语言模型辅助的自动放大比例尺检测与提取框架扫描电子显微镜图像 

**Authors**: Yuxuan Chen, Ruotong Yang, Zhengyang Zhang, Mehreen Ahmed, Yanming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11260)  

**Abstract**: Microscopic characterizations, such as Scanning Electron Microscopy (SEM), are widely used in scientific research for visualizing and analyzing microstructures. Determining the scale bars is an important first step of accurate SEM analysis; however, currently, it mainly relies on manual operations, which is both time-consuming and prone to errors. To address this issue, we propose a multi-modal and automated scale bar detection and extraction framework that provides concurrent object detection, text detection and text recognition with a Large Language Model (LLM) agent. The proposed framework operates in four phases; i) Automatic Dataset Generation (Auto-DG) model to synthesize a diverse dataset of SEM images ensuring robust training and high generalizability of the model, ii) scale bar object detection, iii) information extraction using a hybrid Optical Character Recognition (OCR) system with DenseNet and Convolutional Recurrent Neural Network (CRNN) based algorithms, iv) an LLM agent to analyze and verify accuracy of the results. The proposed model demonstrates a strong performance in object detection and accurate localization with a precision of 100%, recall of 95.8%, and a mean Average Precision (mAP) of 99.2% at IoU=0.5 and 69.1% at IoU=0.5:0.95. The hybrid OCR system achieved 89% precision, 65% recall, and a 75% F1 score on the Auto-DG dataset, significantly outperforming several mainstream standalone engines, highlighting its reliability for scientific image analysis. The LLM is introduced as a reasoning engine as well as an intelligent assistant that suggests follow-up steps and verifies the results. This automated method powered by an LLM agent significantly enhances the efficiency and accuracy of scale bar detection and extraction in SEM images, providing a valuable tool for microscopic analysis and advancing the field of scientific imaging. 

**Abstract (ZH)**: 多模态自动化显微尺检测与提取框架：从数据生成到结果验证 

---
# Large Language Models Are Effective Code Watermarkers 

**Title (ZH)**: 大规模语言模型是有效的代码水印器 

**Authors**: Rui Xu, Jiawei Chen, Zhaoxia Yin, Cong Kong, Xinpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11251)  

**Abstract**: The widespread use of large language models (LLMs) and open-source code has raised ethical and security concerns regarding the distribution and attribution of source code, including unauthorized redistribution, license violations, and misuse of code for malicious purposes. Watermarking has emerged as a promising solution for source attribution, but existing techniques rely heavily on hand-crafted transformation rules, abstract syntax tree (AST) manipulation, or task-specific training, limiting their scalability and generality across languages. Moreover, their robustness against attacks remains limited. To address these limitations, we propose CodeMark-LLM, an LLM-driven watermarking framework that embeds watermark into source code without compromising its semantics or readability. CodeMark-LLM consists of two core components: (i) Semantically Consistent Embedding module that applies functionality-preserving transformations to encode watermark bits, and (ii) Differential Comparison Extraction module that identifies the applied transformations by comparing the original and watermarked code. Leveraging the cross-lingual generalization ability of LLM, CodeMark-LLM avoids language-specific engineering and training pipelines. Extensive experiments across diverse programming languages and attack scenarios demonstrate its robustness, effectiveness, and scalability. 

**Abstract (ZH)**: 大规模语言模型（LLM）和开源代码的广泛应用引发了关于源代码分发和归属的伦理与安全担忧，包括未经授权的重新分发、许可证违规以及代码的恶意使用。水印技术作为源代码归属的一种有前景的解决方案已经出现，但现有技术方法严重依赖手工编写的转换规则、抽象语法树（AST）操作或特定任务的训练，限制了其在多种语言间的可扩展性和通用性。此外，它们对抗攻击的鲁棒性仍然有限。为解决这些限制，我们提出CodeMark-LLM，这是一种以LLM为驱动的水印框架，能够在不牺牲代码语义和可读性的情况下嵌入水印。CodeMark-LLM包括两个核心模块：（i）语义一致嵌入模块，通过保持功能性的转换来编码水印位，以及（ii）差异对比提取模块，通过比较原始代码和水代码来识别已应用的转换。利用LLM的跨语言泛化能力，CodeMark-LLM避免了特定语言的工程和训练流程。广泛的语言和攻击场景下的实验表明其鲁棒性、有效性和可扩展性。 

---
# The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers 

**Title (ZH)**: LLMs长短回答事实一致性问题探究 

**Authors**: Saad Obaid ul Islam, Anne Lauscher, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2510.11218)  

**Abstract**: Large language models (LLMs) can correctly answer "When was Einstein born?" yet fail to provide the same date when writing about Einstein's life revealing a fundamental inconsistency in how models access factual knowledge across task complexities. While models display impressive accuracy on factual question-answering benchmarks, the reliability gap between simple and complex queries remains poorly understood, eroding their trustworthiness. In this work, we introduce Short-Long Form Alignment for Factual Question Answering (SLAQ), a controlled evaluation framework that compares LLMs' answers to the same factual questions asked (a) in isolation (short) vs. (b) integrated into complex queries (long). Looking at 16 LLMs across 600 queries, we find a systematic misalignment of answers to the corresponding short and long queries. We further uncover position-dependent accuracy loss and momentum effects where consecutive correct or incorrect answers create self-reinforcing patterns. Through mechanistic analysis, we find that aligned facts activate overlapping model internals, and that metrics based on mechanistic similarity can predict short-long answer alignment with up to 78% accuracy. Our work establishes factual consistency over query complexity as an important aspect of LLMs' trustworthiness and challenges current evaluation practices, which implicitly assume that good performance for simple factual queries implies reliability in more complex knowledge-seeking tasks too. 

**Abstract (ZH)**: 大型语言模型在正确回答“爱因斯坦是什么时候出生的？”的同时，在描述爱因斯坦生平时未能提供相同的日期，揭示了模型在不同任务复杂性下访问事实知识的基本不一致性。尽管模型在事实性问答基准测试中表现令人印象深刻，但简单和复杂查询之间 reliablity 的差距仍然 poorly understood，侵蚀了其可信度。在本研究中，我们引入了基于短长形式对齐的事实性问答（SLAQ）可控评估框架，该框架将 LLMs 的答案与同一事实问题在（a）孤立（短） vs. （b）整合到复杂查询中的形式进行比较。通过对 16 种 LLMs 在 600 个查询中的分析，我们发现了答案在相应短查询和长查询之间系统性的对齐不一致。我们还发现了位置依赖性的准确性损失和动量效应，其中连续的正确或错误答案会形成自我强化的模式。通过机制分析，我们发现对齐的事实激活了重叠的模型内部组件，并且基于机制相似性的度量可以预测短长答案对齐的准确率最高可达 78%。我们的研究建立了事实一致性的保持在查询复杂性上的重要方面，并挑战了当前的评估实践，后者隐含地假设简单的事实性查询的良好表现也意味着在更复杂的知识搜索任务中具有可靠性。 

---
# Domain-Specific Data Generation Framework for RAG Adaptation 

**Title (ZH)**: 面向特定领域数据生成的RAG适应框架 

**Authors**: Chris Xing Tian, Weihao Xie, Zhen Chen, Zhengyuan Yi, Hui Liu, Haoliang Li, Shiqi Wang, Siwei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.11217)  

**Abstract**: Retrieval-Augmented Generation (RAG) combines the language understanding and reasoning power of large language models (LLMs) with external retrieval to enable domain-grounded responses. Effectively adapting RAG systems to domain-specific settings requires specialized, context-rich training data beyond general-purpose question-answering. Here, we propose RAGen, a scalable and modular framework for generating domain-grounded question-answer-context (QAC) triples tailored to diverse RAG adaptation approaches. RAGen produces these QAC triples by identifying key concepts in documents, generating diverse questions guided by Bloom's Taxonomy-inspired principles, and pairing them with precise answers extracted from relevant contexts. RAGen supports multiple RAG adaptation strategies, including the optimization of key components such as the LLM, retriever, and embedding model, etc. Its modular pipeline features semantic chunking, hierarchical concept extraction, and multi-chunk retrieval, along with the introduction of curated distractor contexts to promote robust reasoning. Designed for scalability, RAGen efficiently handles large and evolving document corpora without redundant processing, making it especially suitable for dynamic evolving domains such as scientific research and enterprise knowledge bases. 

**Abstract (ZH)**: 基于检索的生成（RAG）结合了大规模语言模型（LLMs）的语义理解和推理能力以及外部检索，以实现领域相关的回应。为了有效适应特定领域的设置，RAG系统需要具有领域背景的训练数据，而不仅仅是针对通用问题回答的训练数据。在这里，我们提出RAGen框架，这是一个可扩展且模块化的生成领域相关的问答上下文（QAC）三元组的方法，旨在适应多样的RAG适应方法。RAGen通过识别文档中的关键概念，根据布卢姆分类法启发的原则生成多样化的提问，并结合相关上下文中提取的精确答案来生成这些QAC三元组。RAGen支持多种RAG适应策略，包括优化关键组件如LLM、检索器和嵌入模型等。其模块化管道包括语义切片、层次概念提取和多片段检索，并引入了精心策划的干扰上下文以促进稳健的推理。出于可扩展性考虑，RAGen有效地处理大量不断演变的文档库而不进行冗余处理，使其特别适合科学研究和企业知识库等动态演变领域。 

---
# RAG-Pull: Imperceptible Attacks on RAG Systems for Code Generation 

**Title (ZH)**: RAG-Pull: 不可感知的针对代码生成RAG系统的攻击 

**Authors**: Vasilije Stambolic, Aritra Dhar, Lukas Cavigelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.11195)  

**Abstract**: Retrieval-Augmented Generation (RAG) increases the reliability and trustworthiness of the LLM response and reduces hallucination by eliminating the need for model retraining. It does so by adding external data into the LLM's context. We develop a new class of black-box attack, RAG-Pull, that inserts hidden UTF characters into queries or external code repositories, redirecting retrieval toward malicious code, thereby breaking the models' safety alignment. We observe that query and code perturbations alone can shift retrieval toward attacker-controlled snippets, while combined query-and-target perturbations achieve near-perfect success. Once retrieved, these snippets introduce exploitable vulnerabilities such as remote code execution and SQL injection. RAG-Pull's minimal perturbations can alter the model's safety alignment and increase preference towards unsafe code, therefore opening up a new class of attacks on LLMs. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG)增强的检索生成提高了LLM响应的可靠性和可信度并减少了幻觉，通过消除模型重新训练的需要。它通过将外部数据加入LLM的上下文中实现这一目标。我们开发了一种新的黑盒攻击类别RAG-Pull，该攻击将隐藏的UTF字符插入查询或外部代码仓库中，引导检索指向恶意代码，从而破坏模型的安全对齐。我们观察到，仅查询和代码扰动就可以将检索引导至攻击者控制的片段，而结合查询和目标扰动则几乎可以实现完全的成功。一旦检索到这些片段，它们将引入可利用的漏洞，如远程代码执行和SQL注入。RAG-Pull的最小扰动可以改变模型的安全对齐，并增加对不安全代码的偏好，因此为LLM打开了一个新的攻击类别。 

---
# Protein as a Second Language for LLMs 

**Title (ZH)**: 蛋白质作为LLMs的第二种语言 

**Authors**: Xinhui Chen, Zuchao Li, Mengqi Gao, Yufeng Zhang, Chak Tou Leong, Haoyang Li, Jiaqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11188)  

**Abstract**: Deciphering the function of unseen protein sequences is a fundamental challenge with broad scientific impact, yet most existing methods depend on task-specific adapters or large-scale supervised fine-tuning. We introduce the "Protein-as-Second-Language" framework, which reformulates amino-acid sequences as sentences in a novel symbolic language that large language models can interpret through contextual exemplars. Our approach adaptively constructs sequence-question-answer triples that reveal functional cues in a zero-shot setting, without any further training. To support this process, we curate a bilingual corpus of 79,926 protein-QA instances spanning attribute prediction, descriptive understanding, and extended reasoning. Empirically, our method delivers consistent gains across diverse open-source LLMs and GPT-4, achieving up to 17.2% ROUGE-L improvement (average +7%) and even surpassing fine-tuned protein-specific language models. These results highlight that generic LLMs, when guided with protein-as-language cues, can outperform domain-specialized models, offering a scalable pathway for protein understanding in foundation models. 

**Abstract (ZH)**: 解析 unseen 蛋白质序列的功能是一项具有广泛科学影响的基本挑战，但大多数现有方法依赖于任务特定的适配器或大规模监督微调。“蛋白质作为第二语言”框架将氨基酸序列重新形式化为一种新的符号语言的句子，并通过上下文例证供大规模语言模型解释。我们的方法在零样本设置下自适应构建序列-问题-答案 triple，揭示功能性线索，无需进一步训练。为此，我们编 curated 了一个包含 79,926 个蛋白质-QA 实例的双语语料库，涵盖属性预测、描述性理解以及扩展推理。实验证明，我们的方法在多种开源 LLM 和 GPT-4 上表现出一致的性能提升，ROUGE-L 提高最高达 17.2%（平均提高 7%），甚至超越了蛋白质特定的微调语言模型。这些结果表明，当用蛋白质作为语言的线索加以引导时，通用的大规模语言模型可以在基础模型中实现对专门领域的模型的超越，提供了一种可扩展的蛋白质理解途径。 

---
# video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory 

**Title (ZH)**: 视频-SALMONN S：超越长度限制的流式音视频LLMs记忆方法 

**Authors**: Guangzhi Sun, Yixuan Li, Xiaodong Wu, Yudong Yang, Wei Li, Zejun Ma, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11129)  

**Abstract**: Continuous, high-frame-rate, high-resolution processing of long video streams is critical for future AI agents, yet current video-understanding LLMs struggle to scale. Offline, fixed-frame-number methods require the stream length to adapt frame rates; streaming methods constrain memory by merging or discarding tokens, losing information. We propose video-SALMONN S, a streaming audio-visual LLM that, to our knowledge, is the first to process 3-hour videos at 1 FPS and 360p resolution under a fixed memory budget. Our model introduces (i) a test-time-training (TTT) memory module that continually updates token representations to capture long-range dependencies by replacing token merging, and (ii) a prompt-dependent memory reader that selectively retrieves context-relevant content from fixed-size memory. The TTT module is optimised with a Hessian-free conjugate-gradient procedure (TTT_HF) for efficient adaptation. On long-video benchmarks (Video-MME, LVBench, VideoEvalPro), video-SALMONN S sustains high-quality understanding on multi-hour videos with 10k frames and 1M tokens. Our 8B-parameter model achieves 74.2% overall and 67.8% on the Video-MME long split, outperforming both offline and streaming baselines. 

**Abstract (ZH)**: 视频-SALMONN S：在固定内存预算下处理长达3小时的视频流 

---
# A Vision for Access Control in LLM-based Agent Systems 

**Title (ZH)**: 基于大语言模型的代理系统访问控制愿景 

**Authors**: Xinfeng Li, Dong Huang, Jie Li, Hongyi Cai, Zhenhong Zhou, Wei Dong, XiaoFeng Wang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11108)  

**Abstract**: The autonomy and contextual complexity of LLM-based agents render traditional access control (AC) mechanisms insufficient. Static, rule-based systems designed for predictable environments are fundamentally ill-equipped to manage the dynamic information flows inherent in agentic interactions. This position paper argues for a paradigm shift from binary access control to a more sophisticated model of information governance, positing that the core challenge is not merely about permission, but about governing the flow of information. We introduce Agent Access Control (AAC), a novel framework that reframes AC as a dynamic, context-aware process of information flow governance. AAC operates on two core modules: (1) multi-dimensional contextual evaluation, which assesses not just identity but also relationships, scenarios, and norms; and (2) adaptive response formulation, which moves beyond simple allow/deny decisions to shape information through redaction, summarization, and paraphrasing. This vision, powered by a dedicated AC reasoning engine, aims to bridge the gap between human-like nuanced judgment and scalable Al safety, proposing a new conceptual lens for future research in trustworthy agent design. 

**Abstract (ZH)**: 基于LLM的代理的自主性和情境复杂性使传统访问控制机制不足。静态的基于规则的系统无法应对预测性环境，而适用于动态信息流的治理则是核心挑战。本文提出了从二元访问控制向更先进的信息治理模式转变的观点，介绍了代理访问控制（AAC）框架，将其视为动态的情境感知信息流动治理过程。AAC基于两个核心模块：多维度的情境评估和适应性响应形成。这一愿景通过专门的访问控制推理引擎，旨在弥合类似人类细致判断与可扩展的AI安全性之间的差距，为未来可信赖代理设计的研究提供新的概念框架。 

---
# Enhancing LLM Reasoning via Non-Human-Like Reasoning Path Preference Optimization 

**Title (ZH)**: 通过非人类似推理路径偏好优化增强LLM推理能力 

**Authors**: Junjie Lu, Yuliang Liu, Chaofeng Qu, Wei Shen, Zhouhan Lin, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11104)  

**Abstract**: Current approaches for strengthening LLM reasoning tend to introduce a training bias toward human-like reasoning trajectories. In step-wise preference optimization, in particular, dependence on human or higher-capacity model annotations for intermediate steps limits exploration of alternative, non-human-like reasoning paths and thus constrains achievable performance. Furthermore, through a small-scale pilot study, we observed that in approximately 75% of cases, the model's first erroneous step occurs after the lowest-confidence point. This suggests that guiding the model at its lowest-confidence point before an error provides more accurate supervision than locating the first explicit error. In this paper, we propose Confidence-Guided Reasoning Path Preference Optimization (CGPO), a method that leverages a confidence signal to identify points of maximal uncertainty in the model's reasoning process and applies self-generated, non-human-like reasoning-path guidance to mitigate trajectory drift. Our experiments span diverse models applied to both code and mathematical reasoning tasks. The results show that, with the same amount of training data, our method using data generated by a small model can achieve better performance in most cases compared with approaches using data generated by a strong model or human-annotated. 

**Abstract (ZH)**: 基于信心引导的推理路径偏好优化（CGPO）：一种利用信心信号识别模型推理过程中的最大不确定性并应用非人类like推理路径指导以减轻路径偏移的方法 

---
# From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance 

**Title (ZH)**: 从推理LLM到BERT：一种两阶段知识蒸馏框架用于搜索相关性 

**Authors**: Runze Xia, Yupeng Ji, Yuxi Zhou, Haodong Liu, Teng Zhang, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.11056)  

**Abstract**: Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under "standard" and "reasoning-augmented" inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value. 

**Abstract (ZH)**: 电子商务搜索系统中的查询服务相关性预测面临严格的延迟要求，这阻碍了直接应用大型语言模型（LLMs）的可能性。为了解决这一问题，我们提出了一种两阶段推理蒸馏框架，将强健的教师LLM的推理能力转移到轻量级、易于部署的学生模型上。在第一阶段，我们通过构建一个领域适应的教师模型来解决通用LLM的限制。这一过程包括三个步骤：领域适应预训练以注入平台知识、监督微调以激发推理技能、以及通过多维度奖励模型进行偏好优化以确保生成可靠且偏好对齐的推理路径。然后，该教师可以自动为搜索日志中的大量查询-服务对标注相关性标签和推理链。在第二阶段，为了解决标准蒸馏中由于架构异构性带来的挑战，我们引入了对比推理自蒸馏（CRSD）。通过将同一个学生模型在“标准”和“推理增强”输入下的行为建模为教师-学生关系，CRSD 允许轻量级模型内化教师的复杂决策机制，而不需要在推理时明确的推理路径。美团搜索广告系统中的离线评估和在线 A/B 测试表明，我们的框架在多个指标上取得了显著的改进，验证了其有效性和实际价值。 

---
# Automating Structural Engineering Workflows with Large Language Model Agents 

**Title (ZH)**: 使用大型语言模型代理自动化结构工程工作流 

**Authors**: Haoran Liang, Yufa Zhou, Mohammad Talebi Kalaleh, Qipei Mei  

**Link**: [PDF](https://arxiv.org/pdf/2510.11004)  

**Abstract**: We introduce $\textbf{MASSE}$, the first Multi-Agent System for Structural Engineering, effectively integrating large language model (LLM)-based agents with real-world engineering workflows. Structural engineering is a fundamental yet traditionally stagnant domain, with core workflows remaining largely unchanged for decades despite its substantial economic impact and global market size. Recent advancements in LLMs have significantly enhanced their ability to perform complex reasoning, long-horizon planning, and precise tool utilization -- capabilities well aligned with structural engineering tasks such as interpreting design codes, executing load calculations, and verifying structural capacities. We present a proof-of-concept showing that most real-world structural engineering workflows can be fully automated through a training-free LLM-based multi-agent system. MASSE enables immediate deployment in professional environments, and our comprehensive validation on real-world case studies demonstrates that it can reduce expert workload from approximately two hours to mere minutes, while enhancing both reliability and accuracy in practical engineering scenarios. 

**Abstract (ZH)**: MASSE：面向结构工程的首个multi-agent系统，有效集成基于大语言模型的代理与实际工程工作流程 

---
# DND: Boosting Large Language Models with Dynamic Nested Depth 

**Title (ZH)**: DND: 通过动态嵌套深度增强大型语言模型 

**Authors**: Tieyuan Chen, Xiaodong Chen, Haoxing Chen, Zhenzhong Lan, Weiyao Lin, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.11001)  

**Abstract**: We introduce Dynamic Nested Depth (DND), a novel method that improves performance for off-the-shelf LLMs by selecting critical tokens to reprocess in a nested depth manner. Specifically, at the end of the given transformer layer, DND identifies more critical tokens with a router and feeds them back for an extra round of processing, effectively ``reviewing" difficult tokens while avoiding redundant computation for easier ones. The dynamic selection mechanism is tailored for precise control via two novel strategies: a router controlling loss to enhance token selection distinguishability, and a threshold control scheme to ensure selection stability. We demonstrate the effectiveness of DND by directly integrating it into pre-trained dense and MoE models during a post-training phase. On diverse benchmarks, this approach boosts the performances of the dense Qwen3-1.7B by 1.88% and the MoE Qwen3-30B-A3B by 0.87%, all with a minimal parameter and computing increase. 

**Abstract (ZH)**: 我们引入了动态嵌套深度（DND）方法，这是一种通过以嵌套深度方式重新处理关键令牌来提高现成大语言模型性能的新方法。具体而言，在给定的变压器层结束后，DND 使用一个路由器识别更多关键令牌，并让这些令牌进行额外一轮处理，从而有效地“审查”困难的令牌，同时避免对较简单的令牌进行冗余计算。动态选择机制通过两种新颖策略实现精确控制：一个用于控制损失以增强令牌选择的区分度的路由器，以及一个确保选择稳定性的阈值控制方案。我们通过在后训练阶段直接将DND集成到预训练的密集模型和MoE模型中，展示了DND的有效性。在多种基准测试中，这种方法分别将密集型Qwen3-1.7B和MoE型Qwen3-30B-A3B的性能提高了1.88%和0.87%，同时参数和计算量仅增加了minimal量。 

---
# ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios 

**Title (ZH)**: ableist: LLM生成招聘场景中的交叉残障偏见 

**Authors**: Mahika Phutane, Hayoung Jung, Matthew Kim, Tanushree Mitra, Aditya Vashistha  

**Link**: [PDF](https://arxiv.org/pdf/2510.10998)  

**Abstract**: Large language models (LLMs) are increasingly under scrutiny for perpetuating identity-based discrimination in high-stakes domains such as hiring, particularly against people with disabilities (PwD). However, existing research remains largely Western-centric, overlooking how intersecting forms of marginalization--such as gender and caste--shape experiences of PwD in the Global South. We conduct a comprehensive audit of six LLMs across 2,820 hiring scenarios spanning diverse disability, gender, nationality, and caste profiles. To capture subtle intersectional harms and biases, we introduce ABLEIST (Ableism, Inspiration, Superhumanization, and Tokenism), a set of five ableism-specific and three intersectional harm metrics grounded in disability studies literature. Our results reveal significant increases in ABLEIST harms towards disabled candidates--harms that many state-of-the-art models failed to detect. These harms were further amplified by sharp increases in intersectional harms (e.g., Tokenism) for gender and caste-marginalized disabled candidates, highlighting critical blind spots in current safety tools and the need for intersectional safety evaluations of frontier models in high-stakes domains like hiring. 

**Abstract (ZH)**: 大型语言模型（LLMs）在招聘等高风险领域中越来越多地受到审查，因其在针对残障人士（PwD）方面的身份歧视问题而受到质疑，尤其是在残疾人方面。然而，现有的研究仍然主要以西方为中心，未充分考虑到性别和种姓等多重边缘化形式如何在全球南方塑造残疾人的体验。我们对六种LLM进行了全面审计，覆盖了2,820个招聘场景，涉及多元化的残疾、性别、国籍和种姓特征。为了捕捉细微的交叉性伤害和偏见，我们引入了ABLEIST（能力建设主义、激励、超人化和象征性令牌）这一包含五项针对能力建设主义的具体指标和三项交叉性伤害指标的度量体系，其基础为残疾研究文献。研究结果揭示，最先进的模型未能检测到显著增加的ABLEIST伤害，特别是对于性别和种姓边缘化的残疾候选人的交叉性伤害进一步加剧，这突显了当前安全工具中的关键盲点，并强调了在招聘等高风险领域中对前沿模型进行交叉性安全评估的必要性。 

---
# A Survey on Agentic Multimodal Large Language Models 

**Title (ZH)**: 代理多模态大型语言模型综述 

**Authors**: Huanjin Yao, Ruifei Zhang, Jiaxing Huang, Jingyi Zhang, Yibo Wang, Bo Fang, Ruolin Zhu, Yongcheng Jing, Shunyu Liu, Guanbin Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.10991)  

**Abstract**: With the recent emergence of revolutionary autonomous agentic systems, research community is witnessing a significant shift from traditional static, passive, and domain-specific AI agents toward more dynamic, proactive, and generalizable agentic AI. Motivated by the growing interest in agentic AI and its potential trajectory toward AGI, we present a comprehensive survey on Agentic Multimodal Large Language Models (Agentic MLLMs). In this survey, we explore the emerging paradigm of agentic MLLMs, delineating their conceptual foundations and distinguishing characteristics from conventional MLLM-based agents. We establish a conceptual framework that organizes agentic MLLMs along three fundamental dimensions: (i) Agentic internal intelligence functions as the system's commander, enabling accurate long-horizon planning through reasoning, reflection, and memory; (ii) Agentic external tool invocation, whereby models proactively use various external tools to extend their problem-solving capabilities beyond their intrinsic knowledge; and (iii) Agentic environment interaction further situates models within virtual or physical environments, allowing them to take actions, adapt strategies, and sustain goal-directed behavior in dynamic real-world scenarios. To further accelerate research in this area for the community, we compile open-source training frameworks, training and evaluation datasets for developing agentic MLLMs. Finally, we review the downstream applications of agentic MLLMs and outline future research directions for this rapidly evolving field. To continuously track developments in this rapidly evolving field, we will also actively update a public repository at this https URL. 

**Abstract (ZH)**: 近年来，随着革命性自主代理系统的出现，研究界正在从传统的静态、被动且领域特定的AI代理向更具动态性、主动性且可泛化的代理型AI转变。受代理型AI及其向AGI发展的潜在轨迹日益增长的兴趣推动，我们提供了一篇关于代理型多模态大型语言模型（Agentic MLLMs）的全面综述。在这篇综述中，我们探讨了代理型MLLMs新兴的范式，明确了它们与传统基于MLLM的代理的概念基础和区别特征。我们构建了一个概念框架，从三个方面组织代理型MLLMs：（i）代理内部智能作为系统的指挥官，通过推理、反思和记忆实现准确的长期规划；（ii）代理外部工具调用，即模型主动使用各种外部工具以超越其固有知识扩展其问题解决能力；（iii）代理环境交互，使模型在虚拟或物理环境中更易于采取行动、调整策略并维持目标导向行为，以适应动态现实世界场景。为了进一步促进该领域的研究，我们汇编了开源训练框架、训练和评估数据集，用于开发代理型MLLMs。最后，我们回顾了代理型MLLMs的下游应用，并概述了这一快速发展的领域的未来研究方向。为了持续跟踪这一快速发展的领域的发展，我们还将积极更新一个公共存储库。 

---
# DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation 

**Title (ZH)**: DITTO：一种通过知识蒸馏针对带水印的大语言模型的欺骗攻击框架 

**Authors**: Hyeseon Ahn, Shinwoo Park, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.10987)  

**Abstract**: The promise of LLM watermarking rests on a core assumption that a specific watermark proves authorship by a specific model. We demonstrate that this assumption is dangerously flawed. We introduce the threat of watermark spoofing, a sophisticated attack that allows a malicious model to generate text containing the authentic-looking watermark of a trusted, victim model. This enables the seamless misattribution of harmful content, such as disinformation, to reputable sources. The key to our attack is repurposing watermark radioactivity, the unintended inheritance of data patterns during fine-tuning, from a discoverable trait into an attack vector. By distilling knowledge from a watermarked teacher model, our framework allows an attacker to steal and replicate the watermarking signal of the victim model. This work reveals a critical security gap in text authorship verification and calls for a paradigm shift towards technologies capable of distinguishing authentic watermarks from expertly imitated ones. Our code is available at this https URL. 

**Abstract (ZH)**: LLM水印的假设是危险破绽：一种复杂的伪造攻击使得恶意模型能够生成包含可信目标模型真实外观水印的文字，从而无缝地将有害内容归咎于可信赖的来源。攻击的关键在于将水印辐射性，即微调过程中无意继承的数据模式，从可发现的特征转换为攻击向量。通过从带水印的教师模型中提炼知识，我们的框架使攻击者能够窃取并复制目标模型的水印信号。本文揭示了文本作者身份验证中的关键安全漏洞，呼吁向能够区分真实水印与精心模仿水印的技术转变。代码可在以下链接获取。 

---
# Judge Before Answer: Can MLLM Discern the False Premise in Question? 

**Title (ZH)**: 先判断后回答：MLLM能识别虚假前提吗？ 

**Authors**: Jidong Li, Lingyong Fang, Haodong Zhao, Sufeng Duan, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10965)  

**Abstract**: Multimodal large language models (MLLMs) have witnessed astonishing advancements in recent years. Despite these successes, MLLMs remain vulnerable to flase premise problems. However, existing benchmarks targeting this issue are limited in scope: they often lack fine-grained categorization, exhibit insufficient coverage, and thus fail to provide a rigorous evaluation of the ability of models to recognize false premises. To bridge this gap, we introduce a fully automated pipeline for constructing a comprehensive benchmark of false premise questions. Our method systematically categorizes the premises into three main types and thirteen subtypes according to the abilities required to identify the premises, resulting in the JBA this http URL show current MLLMs still struggle with false premise recognition. Building upon this benchmark, we further propose a recognition enhancement framework tailored to strengthen the robustness of MLLMs to detect false premises. Extensive experiments demonstrate that models trained with our framework achieve significant improvements in false premise recognition. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在近年来取得了惊人的进展。尽管取得了这些成功，MLLMs 对假前提问题仍然易受攻击。然而，针对这一问题的现有基准在范围上有限：它们往往缺乏细粒度的分类、覆盖面不足，因而无法提供对模型识别假前提能力的严谨评估。为弥补这一差距，我们提出了一种全自动的工作流，用于构建全面的假前提问题基准。我们的方法根据识别前提所需的能力系统地将前提划分为三大类和十三个亚类，形成了JBA这一基准。我们的研究表明，当前的MLLMs 在假前提识别方面仍存在困难。基于这一基准，我们进一步提出了一种增强框架，旨在加强MLLMs 识别假前提的鲁棒性。广泛的实验显示，使用我们框架训练的模型在假前提识别方面取得了显著的改进。 

---
# APLOT: Robust Reward Modeling via Adaptive Preference Learning with Optimal Transport 

**Title (ZH)**: APLOT：基于最优运输的自适应偏好学习稳健奖励建模 

**Authors**: Zhuo Li, Yuege Feng, Dandan Guo, Jinpeng Hu, Anningzhe Gao, Xiang Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10963)  

**Abstract**: The reward model (RM) plays a crucial role in aligning Large Language Models (LLMs) with human preferences through Reinforcement Learning, where the Bradley-Terry (BT) objective has been recognized as simple yet powerful, specifically for pairwise preference learning. However, BT-based RMs often struggle to effectively distinguish between similar preference responses, leading to insufficient separation between preferred and non-preferred outputs. Consequently, they may easily overfit easy samples and cannot generalize well to Out-Of-Distribution (OOD) samples, resulting in suboptimal performance. To address these challenges, this paper introduces an effective enhancement to BT-based RMs through an adaptive margin mechanism. Specifically, we design to dynamically adjust the RM focus on more challenging samples through margins, based on both semantic similarity and model-predicted reward differences, which is approached from a distributional perspective solvable with Optimal Transport (OT). By incorporating these factors into a principled OT cost matrix design, our adaptive margin enables the RM to better capture distributional differences between chosen and rejected responses, yielding significant improvements in performance, convergence speed, and generalization capabilities. Experimental results across multiple benchmarks demonstrate that our method outperforms several existing RM techniques, showcasing enhanced performance in both In-Distribution (ID) and OOD settings. Moreover, RLHF experiments support our practical effectiveness in better aligning LLMs with human preferences. Our code is available at this https URL 

**Abstract (ZH)**: 基于布雷德利-泰利目标的奖励模型改进：通过可适应的边际机制实现更好的分布区分与泛化能力 

---
# MC#: Mixture Compressor for Mixture-of-Experts Large Models 

**Title (ZH)**: 混合压缩器用于专家混合大型模型 

**Authors**: Wei Huang, Yue Liao, Yukang Chen, Jianhui Liu, Haoru Tan, Si Liu, Shiming Zhang, Shuicheng Yan, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10962)  

**Abstract**: Mixture-of-Experts (MoE) effectively scales large language models (LLMs) and vision-language models (VLMs) by increasing capacity through sparse activation. However, preloading all experts into memory and activating multiple experts per input introduces significant computational and memory overhead, making the expert module a major contributor to model size and inference cost. To address this, we propose MC# (Mixture-Compressor-sharp), a framework that combines static quantization and dynamic expert pruning by leveraging the significance of experts and tokens for aggressive compression of MoE-LLMs/VLMs. To reduce storage and loading costs, we introduce Pre-Loading Mixed-Precision Quantization (PMQ), which optimizes bit allocation via linear programming, balancing expert importance and quantization error for a Pareto-optimal trade-off between size and performance. To reduce runtime computation, Online Top-any Pruning (OTP) uses Gumbel-Softmax sampling to dynamically select a subset of experts per token, enabling fine-grained control over activation. By combining PMQ's static bit-width optimization with OTP's dynamic routing, MC# achieves extreme compression with minimal accuracy loss. On DeepSeek-VL2, MC# achieves a 6.2 times weight reduction at 2.57 average bits with only a 1.7% accuracy drop across five multimodal benchmarks. Additionally, OTP reduces expert activation over 20% with less than 1% performance degradation, demonstrating strong potential for efficient MoE-based model deployment. 

**Abstract (ZH)**: MC# (Mixture-Compressor-sharp): 结合静态量化和动态专家剪枝的 MoE 大型语言模型/视觉-语言模型高效压缩框架 

---
# KOTOX: A Korean Toxic Dataset for Deobfuscation and Detoxification 

**Title (ZH)**: KOTOX：一个用于去混淆和去毒化的韩语有毒数据集 

**Authors**: Yejin Lee, Su-Hyeon Kim, Hyundong Jin, Dayoung Kim, Yeonsoo Kim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.10961)  

**Abstract**: Toxic content has become an increasingly critical social issue with the rapid expansion of online communication. While numerous studies explored methods for detecting and detoxifying such content, most have focused primarily on English, leaving low-resource language underrepresented. Consequently, Large Language Models~(LLMs) often struggle to identify and neutralize toxic expressions in these languages. This challenge becomes even more pronounced when user employ obfuscation techniques to evade detection systems. Therefore, we propose a \textbf{KOTOX: Korean Toxic Dataset} for deobfuscation and detoxicification to address this issue. We categorize various obfuscation approaches based on linguistic characteristics of Korean and define a set of transformation rules grounded in real-word examples. Using these rules, we construct three dataset versions (easy, normal, and hard) representing different levels of obfuscation difficulty. This is the first dataset that simultaneously supports deobfuscation and detoxification for the Korean language. We expect it to facilitate better understanding and mitigating of obfuscated toxic content in LLM for low-resource languages. Our code and data are available at this https URL. 

**Abstract (ZH)**: KOTOX: 韩语有毒数据集 

---
# Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning 

**Title (ZH)**: 重寻熵正则化的力量：自适应系数解锁其在大语言模型强化学习中的潜力 

**Authors**: Xiaoyun Zhang, Xiaojian Yuan, Di Huang, Wang You, Chen Hu, Jingqing Ruan, Kejiang Chen, Xing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10959)  

**Abstract**: Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability. 

**Abstract (ZH)**: 基于可验证奖励的强化学习推理能力提升中自适应熵正则化方法 

---
# Project-Level C-to-Rust Translation via Synergistic Integration of Knowledge Graphs and Large Language Models 

**Title (ZH)**: 基于知识图谱和大型语言模型协同集成的项目级C到Rust翻译方法 

**Authors**: Zhiqiang Yuan, Wenjun Mao, Zhuo Chen, Xiyue Shang, Chong Wang, Yiling Lou, Xin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.10956)  

**Abstract**: Translating C code into safe Rust is an effective way to ensure its memory safety. Compared to rule-based translation which produces Rust code that remains largely unsafe, LLM-based methods can generate more idiomatic and safer Rust code because LLMs have been trained on vast amount of human-written idiomatic code. Although promising, existing LLM-based methods still struggle with project-level C-to-Rust translation. They typically partition a C project into smaller units (\eg{} functions) based on call graphs and translate them bottom-up to resolve program dependencies. However, this bottom-up, unit-by-unit paradigm often fails to translate pointers due to the lack of a global perspective on their usage. To address this problem, we propose a novel C-Rust Pointer Knowledge Graph (KG) that enriches a code-dependency graph with two types of pointer semantics: (i) pointer-usage information which record global behaviors such as points-to flows and map lower-level struct usage to higher-level units; and (ii) Rust-oriented annotations which encode ownership, mutability, nullability, and lifetime. Synthesizing the \kg{} with LLMs, we further propose \ourtool{}, which implements a project-level C-to-Rust translation technique. In \ourtool{}, the \kg{} provides LLMs with comprehensive pointer semantics from a global perspective, thus guiding LLMs towards generating safe and idiomatic Rust code from a given C project. Our experiments show that \ourtool{} reduces unsafe usages in translated Rust by 99.9\% compared to both rule-based translation and traditional LLM-based rewriting, while achieving an average 29.3\% higher functional correctness than those fuzzing-enhanced LLM methods. 

**Abstract (ZH)**: 将C代码翻译成安全的Rust代码是确保其内存安全的有效方式。与基于规则的翻译方法相比，基于LLM的方法可以生成更多地道且安全的Rust代码，因为LLMs已经训练了大量的人类编写的地道代码。尽管前景广阔，现有的基于LLM的方法在项目级C到Rust翻译方面仍然面临挑战。它们通常基于调用图将C项目拆分为较小的单元（例如函数），并从下至上进行翻译以解决程序依赖关系。然而，这种从下至上、逐单元的范式往往由于缺乏对指针使用全局视角而无法正确翻译指针。为了解决这个问题，我们提出了一种新型的C-Rust指针知识图谱（KG），它在代码依赖图中丰富了两种类型的指针语义：（i）指针使用信息，记录全局行为如指向流，并将低级别的结构使用映射到高级别的单元；（ii）面向Rust的注释，编码所有权、可变性和可为空性以及生命周期。结合知识图谱与LLM，我们进一步提出了\ourtool{}，其实现了项目级C到Rust的翻译技术。在\ourtool{}中，知识图谱为LLM提供了从全局视角的全面指针语义，从而引导LLM从给定的C项目生成安全且地道的Rust代码。我们的实验表明，与基于规则的翻译和传统的基于LLM的重写方法相比，\ourtool{}将翻译后的Rust中不安全使用的比例降低了99.9%，同时比那些增强的模糊测试的LLM方法的平均功能正确性提高了29.3%。 

---
# DreamMakeup: Face Makeup Customization using Latent Diffusion Models 

**Title (ZH)**: DreamMakeup：使用潜在扩散模型的面部化妆个性化定制 

**Authors**: Geon Yeong Park, Inhwa Han, Serin Yang, Yeobin Hong, Seongmin Jeong, Heechan Jeon, Myeongjin Goh, Sung Won Yi, Jin Nam, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.10918)  

**Abstract**: The exponential growth of the global makeup market has paralleled advancements in virtual makeup simulation technology. Despite the progress led by GANs, their application still encounters significant challenges, including training instability and limited customization capabilities. Addressing these challenges, we introduce DreamMakup - a novel training-free Diffusion model based Makeup Customization method, leveraging the inherent advantages of diffusion models for superior controllability and precise real-image editing. DreamMakeup employs early-stopped DDIM inversion to preserve the facial structure and identity while enabling extensive customization through various conditioning inputs such as reference images, specific RGB colors, and textual descriptions. Our model demonstrates notable improvements over existing GAN-based and recent diffusion-based frameworks - improved customization, color-matching capabilities, identity preservation and compatibility with textual descriptions or LLMs with affordable computational costs. 

**Abstract (ZH)**: 全球美妆市场的指数增长与虚拟美妆技术的进步 parallel 进展。尽管生成对抗网络（GANs）在推动这一进步方面发挥了重要作用，但其应用仍然面临着训练不稳定性和定制能力有限等显著挑战。为应对这些挑战，我们提出了一种名为 DreamMakup 的新型无训练自洽扩散模型美妆定制方法，利用扩散模型固有的优势实现更优的可控性和精确的现实图像编辑。DreamMakup 使用早停的 DDIM 反转来保留面部结构和身份，同时通过参考图像、特定的 RGB 颜色和文本描述等多种条件输入实现广泛的定制。我们的模型在定制能力、色彩匹配能力、身份保存以及与文本描述或便宜的计算成本的语言大模型（LLMs）兼容性方面显著优于现有的 GAN 基础和最近的扩散基础框架。 

---
# Agentic RAG for Software Testing with Hybrid Vector-Graph and Multi-Agent Orchestration 

**Title (ZH)**: 基于混合向量-图和多代理协调的软件测试代理记忆辅助系统 

**Authors**: Mohanakrishnan Hariharan, Satish Arvapalli, Seshu Barma, Evangeline Sheela  

**Link**: [PDF](https://arxiv.org/pdf/2510.10824)  

**Abstract**: We present an approach to software testing automation using Agentic Retrieval-Augmented Generation (RAG) systems for Quality Engineering (QE) artifact creation. We combine autonomous AI agents with hybrid vector-graph knowledge systems to automate test plan, case, and QE metric generation. Our approach addresses traditional software testing limitations by leveraging LLMs such as Gemini and Mistral, multi-agent orchestration, and enhanced contextualization. The system achieves remarkable accuracy improvements from 65% to 94.8% while ensuring comprehensive document traceability throughout the quality engineering lifecycle. Experimental validation of enterprise Corporate Systems Engineering and SAP migration projects demonstrates an 85% reduction in testing timeline, an 85% improvement in test suite efficiency, and projected 35% cost savings, resulting in a 2-month acceleration of go-live. 

**Abstract (ZH)**: 使用Agentic Retrieval-Augmented Generation (RAG) 系统进行质量工程（QE） artifacts 自动化生成的方法 

---
# Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures 

**Title (ZH)**: LLMs中隐性知识是否足够？基于RAG的树状结构检索方法 

**Authors**: Mihir Gupte, Paolo Giusto, Ramesh S  

**Link**: [PDF](https://arxiv.org/pdf/2510.10806)  

**Abstract**: Large Language Models (LLMs) are adept at generating responses based on information within their context. While this ability is useful for interacting with structured data like code files, another popular method, Retrieval-Augmented Generation (RAG), retrieves relevant documents to augment the model's in-context learning. However, it is not well-explored how to best represent this retrieved knowledge for generating responses on structured data, particularly hierarchical structures like trees. In this work, we propose a novel bottom-up method to linearize knowledge from tree-like structures (like a GitHub repository) by generating implicit, aggregated summaries at each hierarchical level. This approach enables the knowledge to be stored in a knowledge base and used directly with RAG. We then compare our method to using RAG on raw, unstructured code, evaluating the accuracy and quality of the generated responses. Our results show that while response quality is comparable across both methods, our approach generates over 68% fewer documents in the retriever, a significant gain in efficiency. This finding suggests that leveraging implicit, linearized knowledge may be a highly effective and scalable strategy for handling complex, hierarchical data structures. 

**Abstract (ZH)**: 大型语言模型（LLMs）能够基于上下文中的信息生成响应。虽然这一能力对于与结构化数据（如代码文件）交互很有用，另一种流行的方法——检索增强生成（RAG）——通过检索相关文档来增强模型的上下文学习能力。然而，关于如何最好地表示检索到的知识以生成结构化数据（特别是树状结构）的响应，尚未得到充分探索。在本文中，我们提出了一种新颖的自底向上的方法，通过生成每个层次结构级别的隐式聚合摘要来线性化类似树状结构的知识（如GitHub仓库）。这种方法使知识能够存储在知识库中，并直接与RAG结合使用。然后，我们将我们的方法与直接在RAG中使用原始非结构化代码进行比较，评估生成响应的准确性和质量。结果显示，虽然两种方法生成的响应质量相当，但我们的方法在检索器中生成的文档数量减少了超过68%，显示出显著的效率提升。这一发现表明，利用隐式、线性化的知识可能是处理复杂层次结构数据的有效且可扩展的方法。 

---
# A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions: Dynamical Systems Analysis with Code Generation Applications 

**Title (ZH)**: 多目标LLM交互的随机微分方程框架：动力系统分析与代码生成应用 

**Authors**: Shivani Shukla, Himanshu Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10739)  

**Abstract**: We introduce a general stochastic differential equation framework for modelling multiobjective optimization dynamics in iterative Large Language Model (LLM) interactions. Our framework captures the inherent stochasticity of LLM responses through explicit diffusion terms and reveals systematic interference patterns between competing objectives via an interference matrix formulation. We validate our theoretical framework using iterative code generation as a proof-of-concept application, analyzing 400 sessions across security, efficiency, and functionality objectives. Our results demonstrate strategy-dependent convergence behaviors with rates ranging from 0.33 to 1.29, and predictive accuracy achieving R2 = 0.74 for balanced approaches. This work proposes the feasibility of dynamical systems analysis for multi-objective LLM interactions, with code generation serving as an initial validation domain. 

**Abstract (ZH)**: 我们引入了一种一般性的随机微分方程框架，用于建模迭代大型语言模型（LLM）交互中的多目标优化动力学。我们的框架通过显式的扩散项捕捉LLM响应的固有随机性，并通过干扰矩阵形式揭示竞争目标之间的系统干扰模式。我们使用迭代代码生成作为概念验证应用来验证我们的理论框架，分析了400个会话，涵盖安全性、效率和功能目标。我们的结果展示了策略依赖性的收敛行为，收敛率从0.33到1.29不等，并且平衡方法的预测准确率达到R2 = 0.74。本工作提出了动力系统分析在多目标LLM交互中的可行性，迭代代码生成作为初始验证领域。 

---
# AGENTIQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation 

**Title (ZH)**: AGENTIQL：一种受代理启发的多专家框架用于文本到SQL生成 

**Authors**: Omid Reza Heidari, Siobhan Reid, Yassine Yaakoubi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10661)  

**Abstract**: LLMs have advanced text-to-SQL generation, yet monolithic architectures struggle with complex reasoning and schema diversity. We propose AGENTIQL, an agent-inspired multi-expert framework that combines a reasoning agent for question decomposition, a coding agent for sub-query generation, and a refinement step for column selection. An adaptive router further balances efficiency and accuracy by selecting between our modular pipeline and a baseline parser. Several steps in the pipeline can be executed in parallel, making the framework scalable to larger workloads. Evaluated on the Spider benchmark, AGENTIQL improves execution accuracy and interpretability and achieves up to 86.07\% EX with 14B models using the Planner&Executor merging strategy. The attained performance is contingent upon the efficacy of the routing mechanism, thereby narrowing the gap to GPT-4-based SOTA (89.65% EX) while using much smaller open-source LLMs. Beyond accuracy, AGENTIQL enhances transparency by exposing intermediate reasoning steps, offering a robust, scalable, and interpretable approach to semantic parsing. 

**Abstract (ZH)**: LLMs在文本到SQL生成方面取得了进展，但单一架构在处理复杂推理和模式多样性时遇到困难。我们提出了一种受代理启发的多专家框架AGENTIQL，该框架结合了一个推理代理用于问题分解、一个编码代理用于子查询生成，以及一个列选择修正步骤。动态路由器进一步通过在模块化管道和基线解析器之间选择来平衡效率和准确性。管道中的若干步骤可以并行执行，使该框架能够扩展以处理更大的工作负载。在Spider基准测试下，AGENTIQL提高了执行准确性和可解释性，并且使用Planner&Executor合并策略的14B模型达到了86.07%的EX性能。该性能取决于路由机制的有效性，从而缩小了与基于GPT-4的当前最佳水平（89.65% EX）之间的差距，同时使用了规模更小的开源LLM。除了准确性之外，AGENTIQL通过暴露中间推理步骤增强了透明度，提供了一种 robust、可扩展和可解释的语义解析方法。 

---
# Dynamic Topic Evolution with Temporal Decay and Attention in Large Language Models 

**Title (ZH)**: 大型语言模型中具有时间衰减和注意力的动态话题演化 

**Authors**: Di Wu abd Shuaidong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10613)  

**Abstract**: This paper proposes a modeling framework for dynamic topic evolution based on temporal large language models. The method first uses a large language model to obtain contextual embeddings of text and then introduces a temporal decay function and an attention mechanism. These components allow the model to adjust the importance of semantic units according to time intervals and capture topic variations across different periods. The temporal representations are then mapped into a latent topic space, where a state transition matrix is applied to describe the dynamic evolution of topics. A joint optimization objective constrains both semantic modeling and temporal consistency, ensuring diversity and smoothness in topic generation. The design emphasizes the unified modeling of semantic representation and temporal evolution, which improves topic coherence and diversity while enhancing stability and interpretability over time. Experiments on real-world corpora show that the framework effectively captures the generation, expansion, and decline of topics and outperforms existing models across multiple metrics. Overall, the proposed method provides a systematic solution for understanding dynamic semantic patterns in large-scale text, enriches the research paradigm of topic modeling, and supports complex text analysis tasks in multiple domains. 

**Abstract (ZH)**: 基于时间大型语言模型的动态主题演化建模框架 

---
# Rethinking RL Evaluation: Can Benchmarks Truly Reveal Failures of RL Methods? 

**Title (ZH)**: 重新思考强化学习评估：基准真的能揭示强化学习方法的失败吗？ 

**Authors**: Zihan Chen, Yiming Zhang, Hengguang Zhou, Zenghui Ding, Yining Sun, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2510.10541)  

**Abstract**: Current benchmarks are inadequate for evaluating progress in reinforcement learning (RL) for large language models (LLMs).Despite recent benchmark gains reported for RL, we find that training on these benchmarks' training sets achieves nearly the same performance as training directly on the test sets, suggesting that the benchmarks cannot reliably separate further this http URL study this phenomenon, we introduce a diagnostic suite and the Oracle Performance Gap (OPG) metric that quantifies the performance difference between training on the train split versus the test split of a benchmark. We further analyze this phenomenon with stress tests and find that, despite strong benchmark scores, existing RL methods struggle to generalize across distribution shifts, varying levels of difficulty, and counterfactual scenarios: shortcomings that current benchmarks fail to this http URL conclude that current benchmarks are insufficient for evaluating generalization and propose three core principles for designing more faithful benchmarks: sufficient difficulty, balanced evaluation, and distributional robustness. 

**Abstract (ZH)**: 当前基准不足以评估大规模语言模型强化学习进展 

---
# ECO: Enhanced Code Optimization via Performance-Aware Prompting for Code-LLMs 

**Title (ZH)**: ECO: 增强代码优化通过性能感知提示的代码-LLM优化 

**Authors**: Su-Hyeon Kim, Joonghyuk Hahn, Sooyoung Cha, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.10517)  

**Abstract**: Code runtime optimization-the task of rewriting a given code to a faster one-remains challenging, as it requires reasoning about performance trade-offs involving algorithmic and structural choices. Recent approaches employ code-LLMs with slow-fast code pairs provided as optimization guidance, but such pair-based methods obscure the causal factors of performance gains and often lead to superficial pattern imitation rather than genuine performance reasoning. We introduce ECO, a performance-aware prompting framework for code optimization. ECO first distills runtime optimization instructions (ROIs) from reference slow-fast code pairs; Each ROI describes root causes of inefficiency and the rationales that drive performance improvements. For a given input code, ECO in parallel employs (i) a symbolic advisor to produce a bottleneck diagnosis tailored to the code, and (ii) an ROI retriever to return related ROIs. These two outputs are then composed into a performance-aware prompt, providing actionable guidance for code-LLMs. ECO's prompts are model-agnostic, require no fine-tuning, and can be easily prepended to any code-LLM prompt. Our empirical studies highlight that ECO prompting significantly improves code-LLMs' ability to generate efficient code, achieving speedups of up to 7.81x while minimizing correctness loss. 

**Abstract (ZH)**: 代码运行时优化——一种将给定代码重写为更快速代码的任务——依然具有挑战性，因为这要求对涉及算法和结构选择的性能权衡进行推理。最近的方法使用代码-LLM，并提供快速-缓慢代码对作为优化指导，但基于此类成对的方法往往模糊了性能提升的原因，并且通常导致表面模式的模仿而不是真实的性能推理。我们引入了ECO，一种面向性能的代码优化提示框架。ECO首先从参考的快速-缓慢代码对中提炼出运行时优化指令（ROIs）；每个ROI描述了低效的根本原因及其驱动性能提升的推理。对于给定的输入代码，ECO并行使用(i)一个符号顾问生成针对代码的瓶颈诊断，以及(ii)一个ROI检索器返回相关ROIs。然后将这两个输出组合成一个面向性能的提示，为代码-LLM提供可操作的指导。ECO的提示对模型是通用的，不需要微调，可以轻松添加到任何代码-LLM提示的前面。我们的实证研究表明，ECO提示显著提高了代码-LLM生成高效代码的能力，实现了最高7.81倍的加速，同时最大限度地减少了正确性损失。 

---
# SASER: Stego attacks on open-source LLMs 

**Title (ZH)**: SASER：开源大语言模型中的隐秘攻击 

**Authors**: Ming Tan, Wei Li, Hu Tao, Hailong Ma, Aodi Liu, Qian Chen, Zilong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10486)  

**Abstract**: Open-source large language models (LLMs) have demonstrated considerable dominance over proprietary LLMs in resolving neural processing tasks, thanks to the collaborative and sharing nature. Although full access to source codes, model parameters, and training data lays the groundwork for transparency, we argue that such a full-access manner is vulnerable to stego attacks, and their ill-effects are not fully understood. In this paper, we conduct a systematic formalization for stego attacks on open-source LLMs by enumerating all possible threat models associated with adversary objectives, knowledge, and capabilities. Therein, the threat posed by adversaries with internal knowledge, who inject payloads and triggers during the model sharing phase, is of practical interest. We go even further and propose the first stego attack on open-source LLMs, dubbed SASER, which wields impacts through identifying targeted parameters, embedding payloads, injecting triggers, and executing payloads sequentially. Particularly, SASER enhances the attack robustness against quantization-based local deployment by de-quantizing the embedded payloads. In addition, to achieve stealthiness, SASER devises the performance-aware importance metric to identify targeted parameters with the least degradation of model performance. Extensive experiments on LlaMA2-7B and ChatGLM3-6B, without quantization, show that the stealth rate of SASER outperforms existing stego attacks (for general DNNs) by up to 98.1%, while achieving the same attack success rate (ASR) of 100%. More importantly, SASER improves ASR on quantized models from 0 to 100% in all settings. We appeal for investigations on countermeasures against SASER in view of the significant attack effectiveness. 

**Abstract (ZH)**: 开源大型语言模型（LLMs）在解决神经处理任务方面展现出了比专有LLMs更大的优势，这得益于其协作和共享的性质。尽管完全访问源代码、模型参数和训练数据为透明性奠定了基础，但我们认为这种完全开放的访问方式易受隐写攻击的影响，其潜在危害尚未完全理解。在本文中，我们通过列举与对手目标、知识和能力相关的所有可能威胁模型，系统地对开源LLMs的隐写攻击进行了形式化分析。其中，在模型共享阶段具有内部知识的对手，在注入有效载荷和触发器时所构成的威胁具有实际意义。进一步地，我们提出了首个针对开源LLMs的隐写攻击，称为SASER，该攻击通过识别目标参数、嵌入有效载荷、注入触发器并顺序执行有效载荷来产生影响。特别是，SASER通过去量化嵌入的有效载荷来增强攻击在基于量化的地方部署中的鲁棒性。为了实现隐蔽性，SASER设计了考虑性能的重要性度量，以识别对模型性能影响最小的目标参数。在LlaMA2-7B和ChatGLM3-6B上未进行量化的情况下进行的大量实验表明，SASER的隐蔽率优于现有隐写攻击（适用于通用DNNs）高达98.1%，同时保持相同的攻击成功率（ASR）为100%。更重要的是，SASER在所有设置中将量化模型的ASR从0提高到100%。鉴于SASER攻击效果显著，我们呼吁对其防御措施进行研究。 

---
# UltraLLaDA: Scaling the Context Length to 128K for Diffusion Large Language Models 

**Title (ZH)**: UltraLLaDA: 将上下文长度扩展至128K的扩散型大规模语言模型 

**Authors**: Guangxin He, Shen Nie, Fengqi Zhu, Yuankang Zhao, Tianyi Bai, Ran Yan, Jie Fu, Chongxuan Li, Binhang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10481)  

**Abstract**: Diffusion LLMs have attracted growing interest, with plenty of recent work emphasizing their great potential in various downstream tasks; yet the long-context behavior of diffusion LLMs remains largely uncharted. We present a case study of post-training techniques for extending the context window of diffusion LLMs (i.e., LLaDA) without retraining from scratch. We show that a simple modification to the standard Rotary Positional Embeddings (RoPE) extension effectively accommodates the probabilistic modeling inherent in the diffusion process, enabling stable scaling to longer context ranges. We further compare masking strategies used during post-training and analyze their impact on optimization stability and long-range recall. Instantiating these insights, we introduce UltraLLaDA, a diffusion LLM with a 128K-token context window that, in our empirical evaluation on long-context tasks, significantly outperforms training-free baselines. Our experimental results highlight the special positional extension as a key lever for scaling diffusion LLMs to extended contexts and offer practical guidance for practitioners seeking 128K-scale context via efficient post-training. 

**Abstract (ZH)**: Diffusion LLMs在长上下文行为方面的训练后技术研究：以LLaDA为例 

---
# Assessing Large Language Models for Structured Medical Order Extraction 

**Title (ZH)**: 评估大型语言模型在结构化医疗订单提取中的表现 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2510.10475)  

**Abstract**: Medical order extraction is essential for structuring actionable clinical information, supporting decision-making, and enabling downstream applications such as documentation and workflow automation. Orders may be embedded in diverse sources, including electronic health records, discharge summaries, and multi-turn doctor-patient dialogues, and can span categories such as medications, laboratory tests, imaging studies, and follow-up actions. The MEDIQA-OE 2025 shared task focuses on extracting structured medical orders from extended conversational transcripts, requiring the identification of order type, description, reason, and provenance. We present the MasonNLP submission, which ranked 5th among 17 participating teams with 105 total submissions. Our approach uses a general-purpose, instruction-tuned LLaMA-4 17B model without domain-specific fine-tuning, guided by a single in-context example. This few-shot configuration achieved an average F1 score of 37.76, with notable improvements in reason and provenance accuracy. These results demonstrate that large, non-domain-specific LLMs, when paired with effective prompt engineering, can serve as strong, scalable baselines for specialized clinical NLP tasks. 

**Abstract (ZH)**: 医学订单提取对于结构化 actionable 临床信息、支持决策并促使下游应用如文档记录和工作流自动化至关重要。订单可能嵌入多种来源中，包括电子健康记录、出院总结和多轮医生-患者对话，并涵盖药物、实验室检查、影像学研究和随访行动等类别。MEDIQA-OE 2025 共享任务关注从扩展对话转录中提取结构化医疗订单，要求识别订单类型、描述、原因和起源。我们介绍了 MasonNLP 的提交，该提交在 17 支参赛团队、总计 105 个提交中排名第 5。我们的方法使用了一种通用的、指令调优的 LLaMA-4 17B 模型，未进行特定领域的微调，由一个上下文内示例引导。这种少样本配置实现了平均 F1 分数 37.76，特别是在原因和起源准确性方面有所改进。这些结果表明，当与有效的提示工程相结合时，大规模、非特定领域的语言模型可以作为专门临床 NLP 任务的强大、可扩展基线。 

---
# AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs 

**Title (ZH)**: AnyBCQ: 多精度大型语言模型的硬件高效灵活二进制编码量化 

**Authors**: Gunho Park, Jeongin Bae, Beomseok Kwon, Byeongwook Kim, Se Jung Kwon, Dongsoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.10467)  

**Abstract**: The deployment of large language models (LLMs) is increasingly constrained by memory and latency bottlenecks, motivating the need for quantization techniques that flexibly balance accuracy and efficiency. Recent work has introduced multi-precision models, which enable inference at multiple precisions within a single model depending on runtime constraints. To support such flexibility, quantized weights are often stored as bit-planes, where hardware efficiency improves when the compute operates directly at the bit-plane level and activates only the precision required by each request. In this work, we present AnyBCQ, a hardware-friendly multi-precision extension of Binary-Coded Quantization (BCQ) that supports direct bit-plane operations. By representing weights as binary bit-planes with corresponding scale factors, AnyBCQ enables bit-plane-level computation and maps naturally to accelerator-friendly, bit-parallel arithmetic. Our progressive precision expansion mechanism incrementally refines scaling factors while reusing previously assigned binary codes, yielding monotonic improvements in accuracy as additional bits are enabled. We further co-design a specialized kernel that exploits the BCQ structure to support dynamic per-request precision selection with negligible overhead. Experiments on recent LLMs demonstrate that AnyBCQ significantly narrows the accuracy drop in the low-bit regime (e.g. 2-bit), remains competitive at higher precision, and achieves throughput gains of up to 3.0x over half precision and 1.2x over state-of-the-art multi-precision methods. By aligning algorithmic flexibility with hardware efficiency, AnyBCQ provides a practical foundation for multi-precision LLM deployment across diverse service-level objectives. 

**Abstract (ZH)**: 大语言模型（LLMs）的部署越来越受到内存和延迟瓶颈的限制，推动了灵活平衡准确性和效率的量化技术的需求。近期工作引入了多精度模型，能够在单个模型中根据运行时约束调整为多种精度进行推理。为了支持这种灵活性，量化的权重通常以位平面形式存储，当计算直接在位平面级别进行操作并仅激活每个请求所需的精度时，硬件效率会提高。在这项工作中，我们提出了AnyBCQ，这是一种硬件友好的多精度二进制编码量化（BCQ）扩展，支持直接位平面操作。通过将权重表示为带有相应缩放因子的二进制位平面，AnyBCQ可以实现位平面级别的计算，并自然地映射到加速器友好的位并行算术上。我们提出了一种渐进精度扩展机制，该机制逐步细化缩放因子并重用先前分配的二进制代码，从而在启用更多位时实现单调准确性的提升。我们还共同设计了一个专门的内核，利用BCQ结构支持动态每次请求的精度选择，几乎不增加额外开销。在最近的LLM上的实验表明，AnyBCQ在低位数区间（如2位）显著减小了准确率下降，保持在较高精度下竞争力，并实现了最高3.0倍的半精度吞吐量增益和1.2倍的最新多精度方法的吞吐量增益。通过将算法灵活性与硬件效率相结合，AnyBCQ为跨不同服务水平目标的大语言模型多精度部署提供了务实的基础。 

---
# Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance 

**Title (ZH)**: 音频LLMs真的倾听，还是仅仅转录？测量词汇与声学情感线索的依赖性 

**Authors**: Jingyi Chen, Zhimeng Guo, Jiyun Chun, Pichao Wang, Andrew Perrault, Micha Elsner  

**Link**: [PDF](https://arxiv.org/pdf/2510.10444)  

**Abstract**: Understanding emotion from speech requires sensitivity to both lexical and acoustic cues. However, it remains unclear whether large audio language models (LALMs) genuinely process acoustic information or rely primarily on lexical content. We present LISTEN (Lexical vs. Acoustic Speech Test for Emotion in Narratives), a controlled benchmark designed to disentangle lexical reliance from acoustic sensitivity in emotion understanding. Across evaluations of six state-of-the-art LALMs, we observe a consistent lexical dominance. Models predict "neutral" when lexical cues are neutral or absent, show limited gains under cue alignment, and fail to classify distinct emotions under cue conflict. In paralinguistic settings, performance approaches chance. These results indicate that current LALMs largely "transcribe" rather than "listen," relying heavily on lexical semantics while underutilizing acoustic cues. LISTEN offers a principled framework for assessing emotion understanding in multimodal models. 

**Abstract (ZH)**: Lexical vs. Acoustic Speech Test for Emotion in Narratives 

---
# STEAM: A Semantic-Level Knowledge Editing Framework for Large Language Models 

**Title (ZH)**: STEAM：一种面向大规模语言模型的语义级别知识编辑框架 

**Authors**: Geunyeong Jeong, Juoh Sun, Seonghee Lee, Harksoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.10398)  

**Abstract**: Large Language Models store extensive factual knowledge acquired during large-scale pre-training. However, this knowledge is inherently static, reflecting only the state of the world at the time of training. Knowledge editing has emerged as a promising solution for updating outdated or incorrect facts without full retraining. However, most existing locate-and-edit methods primarily focus on token-level likelihood optimization without addressing semantic coherence. Our analysis reveals that such edited knowledge is often encoded as isolated residual streams in the model's latent space, distinct from pre-existing knowledge and bypassing natural reasoning process. To address this, we propose \textsc{Steam}, a semantic-level knowledge editing framework that enhances integration of updated knowledge into the model's knowledge structure. \textsc{Steam} first identifies target representations as semantic anchors for the updated factual association, then guides the internal representation of the edited fact towards these anchors through an alignment loss during optimization. Experimental results demonstrate that \textsc{Steam} improves model's ability to reason with edited knowledge and enhances semantic coherence, underscoring the importance of latent-space alignment for reliable and coherent knowledge editing. The code is available at this https URL. 

**Abstract (ZH)**: Large Language Models 存储了大量在大规模预训练过程中获得的事实知识。然而，这些知识本质是静态的，仅反映训练时的世界状态。知识编辑已作为一种有前景的解决方案出现，用于更新过时或错误的事实而无需进行全面重新训练。然而，大多数现有的查找和编辑方法主要集中在字符级别似然性优化上，而未能解决语义连贯性问题。我们的分析表明，这些编辑后的知识往往被编码为模型潜在空间中的孤立残差流，与预存知识隔绝，并绕过了自然推理过程。为解决这一问题，我们提出了一种名为 \textsc{Steam} 的语义级知识编辑框架，以增强更新知识与模型知识结构的整合。首先，\textsc{Steam} 确定目标表示作为更新事实关联的语义锚点，然后在优化过程中通过对齐损失引导编辑事实的内部表示朝向这些锚点。实验结果表明，\textsc{Steam} 提高了模型处理编辑知识的能力，并增强了语义连贯性，突显了潜在空间对齐对于可靠且连贯的知识编辑的重要性。代码可在以下网址获取：this https URL。 

---
# RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models 

**Title (ZH)**: RefusalBench: 生成评估基于地面真相语言模型的选择性拒绝能力 

**Authors**: Aashiq Muhamed, Leonardo F. R. Ribeiro, Markus Dreyer, Virginia Smith, Mona T. Diab  

**Link**: [PDF](https://arxiv.org/pdf/2510.10390)  

**Abstract**: The ability of language models in RAG systems to selectively refuse to answer based on flawed context is critical for safety, yet remains a significant failure point. Our large-scale study reveals that even frontier models struggle in this setting, with refusal accuracy dropping below 50% on multi-document tasks, while exhibiting either dangerous overconfidence or overcaution. Static benchmarks fail to reliably evaluate this capability, as models exploit dataset-specific artifacts and memorize test instances. We introduce RefusalBench, a generative methodology that programmatically creates diagnostic test cases through controlled linguistic perturbation. Our framework employs 176 distinct perturbation strategies across six categories of informational uncertainty and three intensity levels. Evaluation of over 30 models uncovers systematic failure patterns: refusal comprises separable detection and categorization skills, and neither scale nor extended reasoning improves performance. We find that selective refusal is a trainable, alignment-sensitive capability, offering a clear path for improvement. We release two benchmarks -- RefusalBench-NQ (single document) and RefusalBench-GaRAGe (multi-document) -- and our complete generation framework to enable continued, dynamic evaluation of this critical capability. 

**Abstract (ZH)**: 基于RAG系统中语言模型在面对错误上下文时选择性拒绝回答的能力对于安全性至关重要，但仍然是一个显著的失败点。我们的大规模研究揭示，即使是前沿模型在此情境下表现不佳，多文档任务中的拒绝准确率低于50%，且表现出危险的过度自信或过度谨慎。静态基准无法可靠评估这一能力，因为模型利用了数据集特定的特征并记忆了测试实例。我们引入了RefusalBench，这是一种生成方法，通过可控的语言扰动程序化地创建诊断测试案例。我们的框架涵盖了六类信息不确定性中的176种不同的扰动策略，以及三个强度级别。对30多个模型的评估发现了系统性的失败模式：拒绝回答包含了可分离的检测和分类技能，而规模和扩展推理均未提高性能。我们发现，选择性拒绝是可训练且对齐敏感的能力，为改进指明了清晰路径。我们发布了两个基准——RefusalBench-NQ（单一文档）和RefusalBench-GaRAGe（多文档）——以及我们的完整生成框架，以使对这一关键能力进行持续和动态评估成为可能。 

---
# Bridging Semantics & Structure for Software Vulnerability Detection using Hybrid Network Models 

**Title (ZH)**: 基于混合网络模型的语义与结构桥梁构建以检测软件漏洞 

**Authors**: Jugal Gajjar, Kaustik Ranaware, Kamalasankari Subramaniakuppusamy  

**Link**: [PDF](https://arxiv.org/pdf/2510.10321)  

**Abstract**: Software vulnerabilities remain a persistent risk, yet static and dynamic analyses often overlook structural dependencies that shape insecure behaviors. Viewing programs as heterogeneous graphs, we capture control- and data-flow relations as complex interaction networks. Our hybrid framework combines these graph representations with light-weight (<4B) local LLMs, uniting topological features with semantic reasoning while avoiding the cost and privacy concerns of large cloud models. Evaluated on Java vulnerability detection (binary classification), our method achieves 93.57% accuracy-an 8.36% gain over Graph Attention Network-based embeddings and 17.81% over pretrained LLM baselines such as Qwen2.5 Coder 3B. Beyond accuracy, the approach extracts salient subgraphs and generates natural language explanations, improving interpretability for developers. These results pave the way for scalable, explainable, and locally deployable tools that can shift vulnerability analysis from purely syntactic checks to deeper structural and semantic insights, facilitating broader adoption in real-world secure software development. 

**Abstract (ZH)**: 软件漏洞依然构成持续的风险，然而静态和动态分析往往忽视了塑造不安全行为的结构依赖关系。将程序视为异构图，我们捕捉控制流和数据流关系作为复杂交互网络。我们的混合框架将这些图表示与轻量级（<4B）局部LLM结合，同时融合拓扑特征和语义推理，避免了大规模云模型的成本和隐私问题。在Java漏洞检测（二分类）评估中，我们的方法实现了93.57%的准确率，分别比基于图注意网络的嵌入高出8.36%，比预训练的LLMbaseline如Qwen2.5 Coder 3B高出17.81%。除了准确率之外，该方法提取了显著子图并生成了自然语言解释，提高了开发者的可解释性。这些结果为可扩展、可解释和本地部署的工具铺平了道路，这些工具可以将漏洞分析从纯粹的语法检查转变为更深层次的结构和语义洞察，促进其实现在世软件开发中的更广泛应用。 

---
# Sample-Efficient Online Learning in LM Agents via Hindsight Trajectory Rewriting 

**Title (ZH)**: 通过回顾轨迹重写实现LM代理的样本高效在线学习 

**Authors**: Michael Y. Hu, Benjamin Van Durme, Jacob Andreas, Harsh Jhamtani  

**Link**: [PDF](https://arxiv.org/pdf/2510.10304)  

**Abstract**: Language model (LM) agents deployed in novel environments often exhibit poor sample efficiency when learning from sequential interactions. This significantly hinders the usefulness of such agents in environments where interaction is costly (for example, when they interact with humans or reset physical systems). While a number of existing LM agent architectures incorporate various mechanisms for experience storage and reflection, they make limited use of LMs' abilities to directly generate or reason about full counterfactual trajectories. We introduce ECHO (Experience Consolidation via Hindsight Optimization), a prompting framework that adapts hindsight experience replay from reinforcement learning for language model agents. ECHO generates optimized trajectories for alternative goals that could have been achieved during failed attempts, effectively creating synthetic positive examples from unsuccessful interactions. Our approach consists of two components: a hindsight rule that uses the language model itself to identify relevant subgoals and generate optimized trajectories, and an update rule that maintains compressed trajectory representations in memory. We evaluate ECHO on stateful versions of XMiniGrid, a text-based navigation and planning benchmark, and PeopleJoinQA, a collaborative information-gathering enterprise simulation. Across both domains, ECHO outperforms vanilla language agent baselines by up to 80%; in XMiniGrid, it also outperforms a number of sophisticated agent architectures including Reflexion and AWM, demonstrating faster adaptation to novel environments through more effective utilization of past experiences. 

**Abstract (ZH)**: 语言模型代理在新型环境中的样本效率往往较低，尤其是在从顺序交互中学习时。这在交互成本高（例如与人类交互或重置物理系统）的环境中显著妨碍了这些代理的有效性。虽然现有的一些语言模型代理架构中包含各种经验存储和反思机制，但它们对利用语言模型直接生成或推理完整反事实轨迹的能力利用有限。我们引入了ECHO（通过后视展望经验 consolidation 的经验整合），这是一种将强化学习中的后视展望经验回放适应语言模型代理的提示框架。ECHO 为替代目标生成优化轨迹，这些目标在失败尝试期间本可以实现，从而有效地从不成功的交互中创建合成的正例。我们的方法包括两个组成部分：一个后视展望规则，使用语言模型本身识别相关子目标并生成优化轨迹，以及一个更新规则，用于在内存中维持压缩轨迹表示。我们在XMiniGrid的有状态版本（一个基于文本的导航和规划基准）和PeopleJoinQA（一个合作信息收集企业模拟）上评估了ECHO。在两个领域中，ECHO 在几乎所有情况下都优于 vanilla 语言代理基线，提高了高达 80%；在XMiniGrid中，ECHO 也优于包括 Reflexion 和 AWM 在内的多种复杂代理架构，证明了通过更有效地利用以往经验更快适应新型环境的能力。 

---
# MatryoshkaThinking: Recursive Test-Time Scaling Enables Efficient Reasoning 

**Title (ZH)**: 套娃思考：递归测试时缩放 enables 高效推理 

**Authors**: Hongwei Chen, Yishu Lei, Dan Zhang, Bo Ke, Danxiang Zhu, Xuyi Chen, Yuxiang Lu, Zhengjie Huang, Shikun Feng, Jingzhou He, Yu Sun, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10293)  

**Abstract**: Test-time scaling has emerged as a promising paradigm in language modeling, wherein additional computational resources are allocated during inference to enhance model performance. Recent approaches, such as DeepConf, have demonstrated the efficacy of this strategy, however, they often incur substantial computational overhead to achieve competitive results. In this work, we propose MatryoshkaThinking, a novel method that significantly reduces computational cost while maintaining state-of-the-art performance. Specifically, MatryoshkaThinking attains a score of 99.79 on AIME2025 using only 4% of the computation required by DeepConf. The core of our approach lies in the recursive exploitation of the model's intrinsic capabilities in reasoning, verification, and summarization, which collectively enhance the retention of correct solutions and reduce the disparity between Pass@k and Pass@1. Comprehensive evaluations across multiple open-source models and challenging multi-modal reasoning benchmarks validate the effectiveness and generality of our method. These findings offer new insights into the design of efficient and scalable test-time inference strategies for advanced language models. 

**Abstract (ZH)**: Test-time Scaling through MatryoshkaThinking: Significantly Reducing Computational Cost while Maintaining State-of-the-Art Performance 

---
# ArtPerception: ASCII Art-based Jailbreak on LLMs with Recognition Pre-test 

**Title (ZH)**: ArtPerception：基于ASCII艺术的LLM突破，并在识别预测试中检测 

**Authors**: Guan-Yan Yang, Tzu-Yu Cheng, Ya-Wen Teng, Farn Wanga, Kuo-Hui Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2510.10281)  

**Abstract**: The integration of Large Language Models (LLMs) into computer applications has introduced transformative capabilities but also significant security challenges. Existing safety alignments, which primarily focus on semantic interpretation, leave LLMs vulnerable to attacks that use non-standard data representations. This paper introduces ArtPerception, a novel black-box jailbreak framework that strategically leverages ASCII art to bypass the security measures of state-of-the-art (SOTA) LLMs. Unlike prior methods that rely on iterative, brute-force attacks, ArtPerception introduces a systematic, two-phase methodology. Phase 1 conducts a one-time, model-specific pre-test to empirically determine the optimal parameters for ASCII art recognition. Phase 2 leverages these insights to launch a highly efficient, one-shot malicious jailbreak attack. We propose a Modified Levenshtein Distance (MLD) metric for a more nuanced evaluation of an LLM's recognition capability. Through comprehensive experiments on four SOTA open-source LLMs, we demonstrate superior jailbreak performance. We further validate our framework's real-world relevance by showing its successful transferability to leading commercial models, including GPT-4o, Claude Sonnet 3.7, and DeepSeek-V3, and by conducting a rigorous effectiveness analysis against potential defenses such as LLaMA Guard and Azure's content filters. Our findings underscore that true LLM security requires defending against a multi-modal space of interpretations, even within text-only inputs, and highlight the effectiveness of strategic, reconnaissance-based attacks. Content Warning: This paper includes potentially harmful and offensive model outputs. 

**Abstract (ZH)**: 大型语言模型（LLMs）集成到计算机应用中引入了变革性的能力但也带来了重大的安全挑战。现有的安全性对齐主要集中在语义解释上，使LLMs容易受到使用非标准数据表示的攻击。本文介绍了ArtPerception，这是一个 novel 黑盒 jailbreak 框架，战略性地利用ASCII艺术绕过最先进的（SOTA）LLM的安全措施。不同于依赖于迭代的暴力攻击方法，ArtPerception引入了一种系统性的、两阶段的方法。第一阶段进行一次性的、针对特定模型的预测试，以实验证明ASCII艺术识别的最佳参数。第二阶段利用这些洞察力，发起一个高效的一次性恶意 jailbreak 攻击。我们提出了一种修改后的莱文斯坦距离（MLD）度量，以更细致地评估LLM的识别能力。通过在四个SOTA开源LLM上的全面实验，我们展示了更强的jailbreak性能。我们进一步通过展示其成功转移到领先的商用模型，包括GPT-4o、Claude Sonnet 3.7和DeepSeek-V3，并对其潜在防御措施（如LLaMA Guard和Azure的内容过滤器）进行严格的有效性分析，证明了我们框架的实际相关性。我们的研究结果强调了即使在仅文本输入的情况下，真正的LLM安全也需要抵御多模态解释空间的攻击，并突显了战略性侦察攻击的有效性。内容警告：本文包括可能有害和冒犯性的模型输出。 

---
# Simulating Viva Voce Examinations to Evaluate Clinical Reasoning in Large Language Models 

**Title (ZH)**: 模拟口头答辩以评估大型语言模型的临床推理能力 

**Authors**: Christopher Chiu, Silviu Pitis, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2510.10278)  

**Abstract**: Clinical reasoning in medicine is a hypothesis-driven process where physicians refine diagnoses from limited information through targeted history, physical examination, and diagnostic investigations. In contrast, current medical benchmarks for large language models (LLMs) primarily assess knowledge recall through single-turn questions, where complete clinical information is provided upfront. To address this gap, we introduce VivaBench, a multi-turn benchmark that evaluates sequential clinical reasoning in LLM agents. Our dataset consists of 1762 physician-curated clinical vignettes structured as interactive scenarios that simulate a (oral) examination in medical training, requiring agents to actively probe for relevant findings, select appropriate investigations, and synthesize information across multiple steps to reach a diagnosis. While current LLMs demonstrate competence in diagnosing conditions from well-described clinical presentations, their performance degrades significantly when required to navigate iterative diagnostic reasoning under uncertainty in our evaluation. Our analysis identified several failure modes that mirror common cognitive errors in clinical practice, including: (1) fixation on initial hypotheses, (2) inappropriate investigation ordering, (3) premature diagnostic closure, and (4) failing to screen for critical conditions. These patterns reveal fundamental limitations in how current LLMs reason and make decisions under uncertainty. Through VivaBench, we provide a standardized benchmark for evaluating conversational medical AI systems for real-world clinical decision support. Beyond medical applications, we contribute to the larger corpus of research on agentic AI by demonstrating how sequential reasoning trajectories can diverge in complex decision-making environments. 

**Abstract (ZH)**: 临床医学中的临床推理是一个假设驱动的过程，医生通过有针对性的病史采集、体格检查和诊断检查，从有限信息中精炼诊断。相比之下，当前用于大型语言模型（LLMs）的医学基准主要通过单轮问题评估知识回忆，提前提供完整临床信息。为解决这一问题，我们引入了VivaBench，这是一个多轮基准，用于评估LLM代理的序贯临床推理能力。我们的数据集包含1762个由医生编纂的临床案例，这些案例以交互式场景的形式结构化，模拟医学培训中的（口头）考试，要求代理积极探究相关发现，选择合适的检查，并在多个步骤中综合信息以得出诊断。尽管当前的LLM在从描述良好的临床表现中诊断疾病方面表现出色，但在我们的评估中要求其在不确定性下进行迭代诊断推理时，其性能显著下降。我们的分析发现了一些失败模式，这些模式反映了临床实践中常见的认知错误，包括：（1）固守初始假设，（2）不适当的检查顺序，（3）过早的诊断闭合，以及（4）未能筛查关键病症。这些模式揭示了当前LLM在不确定性下推理和决策方面的根本局限性。通过VivaBench，我们为评估对话式医疗AI系统在实际临床决策支持中的能力提供了标准化基准。除了医学应用外，我们还通过展示复杂决策环境下的序贯推理轨迹的差异性，为代理AI的研究做出了贡献。 

---
# MetaBreak: Jailbreaking Online LLM Services via Special Token Manipulation 

**Title (ZH)**: MetaBreak: 通过特殊令牌操纵突破在线LLM服务 difficoltà-Taobao 

**Authors**: Wentian Zhu, Zhen Xiang, Wei Niu, Le Guan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10271)  

**Abstract**: Unlike regular tokens derived from existing text corpora, special tokens are artificially created to annotate structured conversations during the fine-tuning process of Large Language Models (LLMs). Serving as metadata of training data, these tokens play a crucial role in instructing LLMs to generate coherent and context-aware responses. We demonstrate that special tokens can be exploited to construct four attack primitives, with which malicious users can reliably bypass the internal safety alignment of online LLM services and circumvent state-of-the-art (SOTA) external content moderation systems simultaneously. Moreover, we found that addressing this threat is challenging, as aggressive defense mechanisms-such as input sanitization by removing special tokens entirely, as suggested in academia-are less effective than anticipated. This is because such defense can be evaded when the special tokens are replaced by regular ones with high semantic similarity within the tokenizer's embedding space. We systemically evaluated our method, named MetaBreak, on both lab environment and commercial LLM platforms. Our approach achieves jailbreak rates comparable to SOTA prompt-engineering-based solutions when no content moderation is deployed. However, when there is content moderation, MetaBreak outperforms SOTA solutions PAP and GPTFuzzer by 11.6% and 34.8%, respectively. Finally, since MetaBreak employs a fundamentally different strategy from prompt engineering, the two approaches can work synergistically. Notably, empowering MetaBreak on PAP and GPTFuzzer boosts jailbreak rates by 24.3% and 20.2%, respectively. 

**Abstract (ZH)**: 不同于源于现有文本文体的常规标记符，特殊标记符在大型语言模型（LLMs）微调过程中人工创建，用于标注结构化对话。作为训练数据的元数据，这些标记符在指导LLMs生成连贯和上下文相关响应方面发挥关键作用。我们展示了一种方法，利用特殊标记符构建四种攻击原语，恶意用户可以利用这些方法可靠地绕过在线LLM服务的内部安全对齐，并同时规避最先进的外部内容审核系统。此外，我们发现缓解这一威胁具有挑战性，学术界建议的激进防御机制（如通过完全移除特殊标记符进行输入清理）的实际效果低于预期，因为当特殊标记符被具有高语义相似性的常规标记符替代时，此类防御可以被规避。我们在实验室环境和商用LLM平台上系统评估了名为MetaBreak的方法。在没有内容审核的情况下，MetaBreak的脱狱成功率与最先进的提示工程解决方案相当。而在有内容审核的情况下，MetaBreak分别比PAP和GPTFuzzer高出11.6%和34.8%。最后，由于MetaBreak采用了一种与提示工程的根本不同策略，这两种方法可以协同工作。值得注意的是，在PAP和GPTFuzzer上增强MetaBreak分别提高了脱狱成功率24.3%和20.2%。 

---
# Audit-of-Understanding: Posterior-Constrained Inference for Mathematical Reasoning in Language Models 

**Title (ZH)**: 理解审计：后验约束推理在语言模型中的数学推理 

**Authors**: Samir Abdaljalil, Erchin Serpedin, Khalid Qaraqe, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2510.10252)  

**Abstract**: Large language models (LLMs) often generate reasoning traces that appear coherent but rest on unsupported assumptions, leading to hallucinated conclusions. Prior work mainly addresses factual hallucinations or relies on post-hoc verification, leaving reasoning-induced hallucinations largely unaddressed. We propose Audit-of-Understanding (AoU), a framework that constrains inference to validated premises through three phases: (1) decomposing a query into candidate assumptions, (2) auditing their support, and (3) conditioning inference only on the validated subset. Formally, AoU is \emph{posterior-constrained inference}, connecting to selective prediction and rejection learning. Our contributions are threefold: (i) theoretical guarantees under perfect validation, (ii) excess-risk bounds under imperfect audits, and (iii) tractability analysis. Empirically, AoU improves both accuracy and faithfulness on GSM8K, MultiArith, and SVAMP, achieving up to +30% gains on GSM8K, +45% on MultiArith, and consistent +20--28% improvements on SVAMP over Chain-of-Thought, Self-Consistency, and CoT-Decoding. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）往往生成表面上连贯但实际上基于未支持假设的推理痕迹，导致产生虚构的结论。先前的工作主要处理事实上的虚构或依赖事后验证，而很大程度上忽略了由推理引起的虚构。我们提出了一种审计理解（AoU）框架，通过三个阶段限制推理仅在验证前提的范围内进行：（1）将查询分解为候选假设，（2）审计其支持情况，（3）仅在验证过的子集中进行推理。形式上，AoU 是后验受限推理，与选择性预测和拒绝学习相关联。我们的贡献包括三个方面：（i）在完美验证下提供的理论保证，（ii）在不完善审计下的 excess-risk 绑定，（iii）可操作性分析。实验表明，AoU 在 GSM8K、MultiArith 和 SVAMP 上提高了准确性和忠实度，分别在 GSM8K 上达到 +30% 的提升，在 MultiArith 上达到 +45% 的提升，并在 SVAMP 上相对于链式思考、自我一致性以及 CoT 编码实现了持续的 +20–28% 的改进。代码可在以下地址获取。 

---
# Reasoning-Enhanced Large Language Models for Molecular Property Prediction 

**Title (ZH)**: 增强推理能力的大语言模型在分子性质预测中的应用 

**Authors**: Jiaxi Zhuang, Yaorui Shi, Jue Hou, Yunong He, Mingwei Ye, Mingjun Xu, Yuming Su, Linfeng Zhang, Linfeng Zhang, Guolin Ke, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.10248)  

**Abstract**: Molecular property prediction is crucial for drug discovery and materials science, yet existing approaches suffer from limited interpretability, poor cross-task generalization, and lack of chemical reasoning capabilities. Traditional machine learning models struggle with task transferability, while specialized molecular language models provide little insight into their decision-making processes. To address these limitations, we propose \textbf{MPPReasoner}, a multimodal large language model that incorporates chemical reasoning for molecular property prediction. Our approach, built upon Qwen2.5-VL-7B-Instruct, integrates molecular images with SMILES strings to enable comprehensive molecular understanding. We develop a two-stage training strategy: supervised fine-tuning (SFT) using 16,000 high-quality reasoning trajectories generated through expert knowledge and multiple teacher models, followed by Reinforcement Learning from Principle-Guided Rewards (RLPGR). RLPGR employs verifiable, rule-based rewards that systematically evaluate chemical principle application, molecular structure analysis, and logical consistency through computational verification. Extensive experiments across 8 datasets demonstrate significant performance improvements, with MPPReasoner outperforming the best baselines by 7.91\% and 4.53\% on in-distribution and out-of-distribution tasks respectively. MPPReasoner exhibits exceptional cross-task generalization and generates chemically sound reasoning paths that provide valuable insights into molecular property analysis, substantially enhancing both interpretability and practical utility for chemists. Code is available at this https URL. 

**Abstract (ZH)**: 分子性质预测对于药物发现和材料科学至关重要，但现有方法存在解释性有限、跨任务泛化能力差和缺乏化学推理能力的问题。传统机器学习模型难以实现任务转移，而专门的分子语言模型对于其决策过程的洞察则较少。为应对这些局限性，我们提出了\textbf{MPPReasoner}——一种结合化学推理能力的多模态大型语言模型，用于分子性质预测。我们的方法基于Qwen2.5-VL-7B-Instruct构建，将分子图像与SMILES字符串相结合，以实现全面的分子理解。我们开发了一种两阶段训练策略：使用16,000条高质量推理轨迹进行监督微调（SFT），这些轨迹通过专家知识和多个教师模型生成，后续采用基于原则指导奖励的强化学习（RLPGR）。RLPGR采用可验证的基于规则的奖励，通过系统地评估化学原理的应用、分子结构分析和逻辑一致性来进行计算验证。跨8个数据集的广泛实验展示了显著的性能提升，MPPReasoner在同分布任务和异分布任务中，分别比最佳基线提高了7.91%和4.53%。MPPReasoner展示了出色的跨任务泛化能力，并生成了化学上可信的推理路径，提供了有价值的分子性质分析见解，显著提升了化学家的解释性和实际应用。代码可在以下链接获取。 

---
# You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs 

**Title (ZH)**: 你只需要额外4个词：LLMs的协同测试时适配 

**Authors**: Yijie Xu, Huizai Yao, Zhiyu Guo, Weiyu Guo, Pengteng Li, Aiwei Liu, Xuming Hu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.10223)  

**Abstract**: Large language models (LLMs) are increasingly deployed in specialized domains such as finance, medicine, and agriculture, where they face significant distribution shifts from their training data. Domain-specific fine-tuning can mitigate this challenge but relies on high-quality labeled data that is expensive and slow to collect in expertise-limited settings. We study label-free test-time adaptation for language models and present SyTTA, an inference-time framework that adapts models on-the-fly without additional supervision. SyTTA couples two complementary uncertainty signals that arise under distribution shift: input-side perplexity, indicating mismatch with domain-specific terminology and patterns, and output-side predictive entropy, indicating diffuse and unstable token probabilities during generation. Across diverse model architectures and domain-specific benchmarks, SyTTA delivers consistent gains. Notably, on agricultural question answering, SyTTA improves Rouge-LSum by over 120% on Qwen-2.5-7B with only 4 extra tokens per query. These results show that effective test-time adaptation for language models is achievable without labeled examples, supporting deployment in label-scarce domains. The code will be made available upon acceptance. 

**Abstract (ZH)**: large语言模型（LLMs）在金融、医学和农业等专门领域中的应用越来越多，这些领域中的分布偏移给它们带来了挑战。领域特定的微调可以缓解这一挑战，但它依赖于高质量的标注数据，而在专家资源有限的情况下，这些数据的收集既昂贵又耗时。我们研究了语言模型的无标签测试时自适应方法，并提出了SyTTA，一种在推理时无需额外监督即可适应模型的框架。SyTTA结合了两种在分布偏移下出现的互补不确定性信号：输入端困惑度，表示与领域特定术语和模式的不匹配；输出端预测熵，表示生成过程中标记概率的分散和不稳定。在多种模型架构和领域特定基准测试中，SyTTA表现出一致的提升效果。特别是在农业问答领域，SyTTA在Qwen-2.5-7B上仅使用每查询4个额外标记就将Rouge-LSum提高了超过120%。这些结果表明，即使在缺少标注数据的情况下，语言模型的测试时自适应也是可以实现的，从而支持在标注资源稀缺的领域中的应用。代码将在录用后提供。 

---
# RLFR: Extending Reinforcement Learning for LLMs with Flow Environment 

**Title (ZH)**: RLFR：通过流环境扩展 reinforcement learning 在大语言模型中的应用 

**Authors**: Jinghao Zhang, Naishan Zheng, Ruilin Li, Dongzhou Cheng, Zheming Liang, Feng Zhao, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10201)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a promising framework for improving reasoning abilities in Large Language Models (LLMs). However, policy optimized with binary verification prone to overlook potential valuable exploration in reasoning trajectory. In view of heavy annotation cost of golden Process Reward Models (PRMs), recent works attempt using auxiliary signals for reward shaping of process tokens, involving entropy and likelihood collected from logit space. In this work, we offer a novel perspective on shaping RLVR with flow rewards derived from latent space, and propose RLFR, where the flow fields of model latents are constructed from either off-policy high-quality data and on-policy rejection sampling data, and the velocity deviations of policy latents within it are quantified to serve as a reward signal. RLFR first demonstrates that a well-established flow field can be a sound environment for reward signal collection, highlighting the expressive latent space is much underexplored. Moreover, RLFR is able to compress any off-policy expert data as reference for constituting reward signals, and we show that the efficient context dependence compressed within the hidden states are utilized, rather than individual token-level denotation for context comprehending. Experiments on both language and multimodal reasoning benchmarks demonstrate the reliability of flow rewards, and suggesting a promising paradigm for reward shaping with auxiliary signals. 

**Abstract (ZH)**: 基于可验证奖励的强化学习（RLVR）近年来已成为提高大型语言模型（LLMs）推理能力的一种有前途的框架。然而，基于二元验证优化的策略可能会忽略推理轨迹中潜在的有价值探索。鉴于黄金过程奖励模型（PRMs）标注成本高昂，近期工作尝试使用辅助信号来塑造过程令牌的奖励，涉及从logit空间收集的熵和似然性。在本工作中，我们提出了一个新的视角，利用源自潜在空间的流动奖励来塑造RLVR，并提出了一种新的方法RLFR，其中模型潜在的空间流动场由离策高质量数据和在线策拒绝采样数据构建，且政策潜在的位移偏差被量化作为奖励信号。RLFR首先表明一个成熟的流动场可以成为一个可靠的奖励信号收集环境，突显了表达性强的潜在空间远未被探索。此外，RLFR能够压缩任何形式的离策专家数据作为构建奖励信号的参考，并表明内在的高效上下文依赖性在隐藏状态中被利用，而非个体令牌级别的表示。在语言和多模态推理基准测试上的实验表明流动奖励的可靠性，并表明提供了使用辅助信号进行奖励塑造的一种有前景的范式。 

---
# MedAgentAudit: Diagnosing and Quantifying Collaborative Failure Modes in Medical Multi-Agent Systems 

**Title (ZH)**: MedAgentAudit: 诊断和量化医疗多智能体系统中的协作失效模式 

**Authors**: Lei Gu, Yinghao Zhu, Haoran Sang, Zixiang Wang, Dehao Sui, Wen Tang, Ewen Harrison, Junyi Gao, Lequan Yu, Liantao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.10185)  

**Abstract**: While large language model (LLM)-based multi-agent systems show promise in simulating medical consultations, their evaluation is often confined to final-answer accuracy. This practice treats their internal collaborative processes as opaque "black boxes" and overlooks a critical question: is a diagnostic conclusion reached through a sound and verifiable reasoning pathway? The inscrutable nature of these systems poses a significant risk in high-stakes medical applications, potentially leading to flawed or untrustworthy conclusions. To address this, we conduct a large-scale empirical study of 3,600 cases from six medical datasets and six representative multi-agent frameworks. Through a rigorous, mixed-methods approach combining qualitative analysis with quantitative auditing, we develop a comprehensive taxonomy of collaborative failure modes. Our quantitative audit reveals four dominant failure patterns: flawed consensus driven by shared model deficiencies, suppression of correct minority opinions, ineffective discussion dynamics, and critical information loss during synthesis. This study demonstrates that high accuracy alone is an insufficient measure of clinical or public trust. It highlights the urgent need for transparent and auditable reasoning processes, a cornerstone for the responsible development and deployment of medical AI. 

**Abstract (ZH)**: 基于大型语言模型的多Agent系统在模拟医疗咨询中显示出潜力，但其评估通常局限于最终答案的准确性。这种做法将它们的内部协作过程视为不透明的“黑盒”，并忽视了一个关键问题：诊断结论是否是通过一条合理的可验证推理路径得出的？这些系统的不可解释性在高风险医疗应用中构成了重大风险，可能导致不正确的或不可靠的结论。为此，我们对来自六个医疗数据集和六个代表性多Agent框架的3600个案例进行了大规模实证研究。通过结合定性分析与定量审计的严格混合方法，我们开发出全面的协作失败模式分类体系。我们的定量审计揭示了四种主要失败模式：由共享模型缺陷驱动的不正确共识、压制正确的少数意见、无效的讨论动态以及在综合过程中的关键信息丢失。本研究证明，单纯高准确率不足以衡量临床或公众的信任度。它强调了透明和可审计推理过程的急迫需求，这是负责任地开发和部署医疗AI的基石。 

---
# A Survey of Inductive Reasoning for Large Language Models 

**Title (ZH)**: 大型语言模型中归纳推理的研究综述 

**Authors**: Kedi Chen, Dezhao Ruan, Yuhao Dan, Yaoting Wang, Siyu Yan, Xuecheng Wu, Yinqi Zhang, Qin Chen, Jie Zhou, Liang He, Biqing Qi, Linyang Li, Qipeng Guo, Xiaoming Shi, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10182)  

**Abstract**: Reasoning is an important task for large language models (LLMs). Among all the reasoning paradigms, inductive reasoning is one of the fundamental types, which is characterized by its particular-to-general thinking process and the non-uniqueness of its answers. The inductive mode is crucial for knowledge generalization and aligns better with human cognition, so it is a fundamental mode of learning, hence attracting increasing interest. Despite the importance of inductive reasoning, there is no systematic summary of it. Therefore, this paper presents the first comprehensive survey of inductive reasoning for LLMs. First, methods for improving inductive reasoning are categorized into three main areas: post-training, test-time scaling, and data augmentation. Then, current benchmarks of inductive reasoning are summarized, and a unified sandbox-based evaluation approach with the observation coverage metric is derived. Finally, we offer some analyses regarding the source of inductive ability and how simple model architectures and data help with inductive tasks, providing a solid foundation for future research. 

**Abstract (ZH)**: 大型语言模型中归纳推理的重要性和综述 

---
# LLMs are All You Need? Improving Fuzz Testing for MOJO with Large Language Models 

**Title (ZH)**: LLMs 足以应对一切？Improving Fuzz Testing for MOJO with Large Language Models 

**Authors**: Linghan Huang, Peizhou Zhao, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10179)  

**Abstract**: The rapid development of large language models (LLMs) has revolutionized software testing, particularly fuzz testing, by automating the generation of diverse and effective test inputs. This advancement holds great promise for improving software reliability. Meanwhile, the introduction of MOJO, a high-performance AI programming language blending Python's usability with the efficiency of C and C++, presents new opportunities to enhance AI model scalability and programmability. However, as a new language, MOJO lacks comprehensive testing frameworks and a sufficient corpus for LLM-based testing, which exacerbates model hallucination. In this case, LLMs will generate syntactically valid but semantically incorrect code, significantly reducing the effectiveness of fuzz testing. To address this challenge, we propose MOJOFuzzer, the first adaptive LLM-based fuzzing framework designed for zero-shot learning environments of emerging programming languages. MOJOFuzzer integrates a mutil-phase framework that systematically eliminates low-quality generated inputs before execution, significantly improving test case validity. Furthermore, MOJOFuzzer dynamically adapts LLM prompts based on runtime feedback for test case mutation, enabling an iterative learning process that continuously enhances fuzzing efficiency and bug detection performance. Our experimental results demonstrate that MOJOFuzzer significantly enhances test validity, API coverage, and bug detection performance, outperforming traditional fuzz testing and state-of-the-art LLM-based fuzzing approaches. Using MOJOFuzzer, we have conducted a first large-scale fuzz testing evaluation of MOJO, uncorvering 13 previous unknown bugs. This study not only advances the field of LLM-driven software testing but also establishes a foundational methodology for leveraging LLMs in the testing of emerging programming languages. 

**Abstract (ZH)**: 大型语言模型的快速发展通过自动化生成多样且有效的测试输入， revolutionized 软件测试，特别是在 fuzz 测试领域。这一进展为提高软件可靠性提供了巨大潜力。同时，MOJO 语言的引入，这一结合了 Python 的易用性和 C 和 C++ 高效性的高性能 AI 编程语言，为提升 AI 模型的可扩展性和编程性提供了新的机会。然而，作为一门新兴语言，MOJO 缺乏全面的测试框架和基于 LLM 的测试所需的数据集，这加剧了模型的幻觉现象。在这种情况下，LLMs 将生成语法正确但语义错误的代码，显著降低了 fuzz 测试的有效性。为应对这一挑战，我们提出了 MOJOFuzzer，这是首个为新兴编程语言零-shot 学习环境设计的基于 LLM 的 fuzzing 框架。MOJOFuzzer 结合了一个多阶段框架，在执行前系统地消除低质量生成的输入，显著提高测试用例的有效性。此外，MOJOFuzzer 根据运行时反馈动态调整 LLM 命令，实现迭代学习过程，不断提高 fuzzing 效率和 bug 检测性能。实验结果表明，MOJOFuzzer 显著提高了测试有效性、API 覆盖范围和 bug 检测性能，优于传统 fuzz 测试和最先进的基于 LLM 的 fuzzing 方法。利用 MOJOFuzzer，我们首次对 MOJO 进行了大规模 fuzz 测试评估，发现了 13 个先前未知的 bug。本研究不仅推动了 LLM 驱动的软件测试领域的发展，还确立了利用 LLM 对新兴编程语言进行测试的基础方法论。 

---
# Large Language Model Sourcing: A Survey 

**Title (ZH)**: 大规模语言模型数据来源：一项调查 

**Authors**: Liang Pang, Kangxi Wu, Sunhao Dai, Zihao Wei, Zenghao Duan, Jia Gu, Xiang Li, Zhiyi Yin, Jun Xu, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.10161)  

**Abstract**: The rapid advancement of large language models (LLMs) has revolutionized artificial intelligence, shifting from supporting objective tasks (e.g., recognition) to empowering subjective decision-making (e.g., planning, decision). This marks the dawn of general and powerful AI, with applications spanning a wide range of fields, including programming, education, healthcare, finance, and law. However, their deployment introduces multifaceted risks. Due to the black-box nature of LLMs and the human-like quality of their generated content, issues such as hallucinations, bias, unfairness, and copyright infringement become particularly significant. In this context, sourcing information from multiple perspectives is essential.
This survey presents a systematic investigation into provenance tracking for content generated by LLMs, organized around four interrelated dimensions that together capture both model- and data-centric perspectives. From the model perspective, Model Sourcing treats the model as a whole, aiming to distinguish content generated by specific LLMs from content authored by humans. Model Structure Sourcing delves into the internal generative mechanisms, analyzing architectural components that shape the outputs of model. From the data perspective, Training Data Sourcing focuses on internal attribution, tracing the origins of generated content back to the training data of model. In contrast, External Data Sourcing emphasizes external validation, identifying external information used to support or influence the responses of model. Moreover, we also propose a dual-paradigm taxonomy that classifies existing sourcing methods into prior-based (proactive traceability embedding) and posterior-based (retrospective inference) approaches. Traceability across these dimensions enhances the transparency, accountability, and trustworthiness of LLMs deployment in real-world applications. 

**Abstract (ZH)**: 大型语言模型的迅速发展已经改变了人工智能，从支持客观任务（如识别）转变为赋能主观决策（如规划、决策）。这标志着通用且强大的人工智能的 dawn，其应用涵盖了编程、教育、医疗、金融和法律等多个领域。然而，其部署引入了多方面的风险。由于大型语言模型的黑盒性质及其生成内容的人类质量，幻觉、偏见、不公平和版权侵权等问题变得尤为重要。在这种背景下，从多个角度获取信息是必不可少的。

本文综述了针对由大型语言模型生成内容的来源追踪进行系统的调查，围绕四个相互关联的维度组织，这些维度共同捕捉了模型和数据中心的视角。从模型视角来看，模型来源将模型作为一个整体对待，旨在区分由特定大型语言模型生成的内容与由人类编写的內容。模型结构来源深入探讨了内部生成机制，分析了塑造模型输出的架构组件。从数据视角来看，训练数据来源关注内部归因，追溯生成内容的溯源至模型的训练数据。相比之下，外部数据来源强调外部验证，识别用于支持或影响模型响应的外部信息。此外，我们还提出了一种双范式分类法，将现有的来源方法分为基于先验（主动跟踪嵌入）和基于后验（回顾性推理）的方法。这些维度上的可追溯性增强了大型语言模型在实际应用中部署的透明性、问责性和可信度。 

---
# BILLY: Steering Large Language Models via Merging Persona Vectors for Creative Generation 

**Title (ZH)**: BILLY: 通过合并人格向量引导大型语言模型进行创造性生成 

**Authors**: Tsung-Min Pai, Jui-I Wang, Li-Chun Lu, Shao-Hua Sun, Hung-Yi Lee, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10157)  

**Abstract**: Multi-LLM systems enhance the creativity of large language models by simulating human collective intelligence but suffer from significant drawbacks, such as high computational costs and inference latency. To address these limitations, we propose BILLY (BlendIng persona vectors for Large Language model creativitY), a training-free framework that captures the benefits of multi-LLM collaboration, i.e. inducing diverse perspectives and specialized expertise, within a single model. BILLY operates by extracting and blending multiple distinct persona vectors directly in the model's activation space. We steer the model's generation process with this merged vector while inference, enabling multi-perspective output without explicit multi-LLM communication. Our experiments across creativity-oriented benchmarks demonstrate that BILLY surpasses single model prompting and traditional multi-LLM approaches, while substantially reducing inference time and computational costs. Our analyses further reveal that distinct persona vectors can be blended to achieve both effective control over complementary aspects of generation and greater interpretability. 

**Abstract (ZH)**: Multi-LLM系统通过模拟人类集体智能增强了大型语言模型的创造力，但面临显著的缺点，如高额的计算成本和推断延迟。为了解决这些局限性，我们提出了BILLY（BlendIng persona vectors for Large Language model creativitY）框架，这是一种无需训练的框架，能够在单一模型中捕捉多LLM合作的优势，即诱导多样视角和专业 expertise。BILLY通过直接在模型的激活空间中提取和融合多个独特的persona向量来操作，我们在推断过程中通过这个合并向量引导模型的生成过程，从而实现多视角输出而无需显式的多LLM通信。我们在创造力导向的基准测试中进行的实验表明，BILLY在单模型提示和传统多LLM方法的基础上，显著减少了推断时间和计算成本。进一步的分析表明，不同的persona向量可以被融合以实现生成的有效控制和更高的可解释性。 

---
# Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective 

**Title (ZH)**: 重新思考RLVR中的熵干预：从熵变化视角出发 

**Authors**: Zhezheng Hao, Hong Wang, Haoyang Liu, Jian Luo, Jiarui Yu, Hande Dong, Qiang Lin, Can Wang, Jiawei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10150)  

**Abstract**: While Reinforcement Learning with Verifiable Rewards (RLVR) can enhance LLM reasoning, its training process poses a critical risk: entropy collapse. This phenomenon is a rapid loss of policy diversity, stemming from the exploration-exploitation imbalance and leading to a lack of generalization. Recent entropy-intervention methods aim to prevent \coloredtext{entropy collapse}, yet their underlying mechanisms remain unclear. In this paper, we conduct a quantitative analysis to reveal token-level entropy changes and how existing entropy intervention methods help avoid entropy collapse. Our findings point out a fundamental limitation of existing methods: they attempt to control entropy dynamics indirectly. By only affecting related factors, such as the advantage signal and generation probability, their effectiveness is inherently limited and could potentially fail. To address this limitation, we introduce an entropy-change-aware reweighting scheme, namely Stabilizing Token-level Entropy-changE via Reweighting (STEER), that adaptively stabilizes entropy dynamics through fine-grained token-level adjustments. Our approach mitigates over-exploitation while fostering robust exploration. Extensive experiments demonstrate that STEER significantly mitigates entropy collapse, stabilizes entropy dynamics, and achieves stronger downstream performance across various mathematical reasoning benchmarks \footnote{Our code is available at this https URL. 

**Abstract (ZH)**: While Reinforcement Learning with Verifiable Rewards (RLVR)可以增强大模型的推理能力，其训练过程存在一个关键风险：熵坍缩现象。这种现象源于探索与利用间的不平衡，导致政策多样性快速降低，从而影响泛化能力。最近的熵干预方法旨在防止熵坍缩，但其背后的机制仍不明确。本文通过定量分析揭示了token级别的熵变化及其现有熵干预方法如何帮助避免熵坍缩。我们的研究发现指出了现有方法的一个根本局限性：它们试图通过间接方式控制熵动态。仅通过影响相关因素，如优势信号和生成概率，它们的有效性受到限制，有可能失效。为解决这一局限性，我们提出了一种熵变化感知的加权方案，即通过细粒度的token级别调整来适应性稳定熵动态（Stabilizing Token-level Entropy-changE via Reweighting, STEER）。我们的方法在抑制过度利用的同时促进了稳健的探索。广泛实验表明，STEER显著缓解了熵坍缩，稳定了熵动态，并在各种数学推理基准上实现了更强的下游性能。 

---
# DiffHeads: Differential Analysis and Inference-Time Masking of Bias Heads in Large Language Models 

**Title (ZH)**: DiffHeads: 大型语言模型中偏差头部的差异性分析及推理时掩码 

**Authors**: Tingxu Han, Wei Song, Ziqi Ding, Ziming Li, Chunrong Fang, Yuekang Li, Dongfang Liu, Zhenyu Chen, Zhenting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10142)  

**Abstract**: Large language models (LLMs) increasingly mediate decisions in domains where unfair treatment of demographic groups is unacceptable. Existing work probes when biased outputs appear, but gives little insight into the mechanisms that generate them, leaving existing mitigations largely fragile. In this paper, we conduct a systematic investigation LLM unfairness and propose DiffHeads, a lightweight debiasing framework for LLMs. We first compare Direct-Answer (DA) prompting to Chain-of-Thought (CoT) prompting across eight representative open- and closed-source LLMs. DA will trigger the nature bias part of LLM and improve measured unfairness by 534.5%-391.9% in both one-turn and two-turn dialogues. Next, we define a token-to-head contribution score that traces each token's influence back to individual attention heads. This reveals a small cluster of bias heads that activate under DA but stay largely dormant with CoT, providing the first causal link between prompting strategy and bias emergence. Finally, building on this insight, we propose DiffHeads that identifies bias heads through differential activation analysis between DA and CoT, and selectively masks only those heads. DiffHeads reduces unfairness by 49.4%, and 40.3% under DA and CoT, respectively, without harming model utility. 

**Abstract (ZH)**: 大型语言模型（LLMs）在不可接受对 demographic groups 不公平处理的领域中日益成为决策中介。现有研究探讨了偏差输出出现的时间，但缺乏生成这些偏差机制的洞察，导致现有缓解措施不够稳健。在本文中，我们对 LLM 不公平性进行系统研究，并提出了一个轻量级的去偏框架 DiffHeads。我们首先比较了直接答案（DA）提示和思考链（CoT）提示在八种代表性开源和闭源 LLM 中的表现。DA 技术会触发 LLM 的自然偏差部分，并在单轮和双轮对话中分别将测量的不公平性提高 534.5% 至 391.9%。接下来，我们定义了一个 token 至 attention head 贡献评分，追踪每个 token 的影响追溯到个体注意力头。这揭示了一个在 DA 下激活但在 CoT 下保持静默的小型偏差头簇，提供了提示策略与偏差出现之间的因果联系。最后，基于这一洞察，我们提出了 DiffHeads，通过分析 DA 和 CoT 之间的不同激活来识别偏差头，并仅选择性地遮掩这些头。DiffHeads 分别在 DA 和 CoT 情况下将不公平性降低 49.4% 和 40.3%，而不损害模型的实用性。 

---
# PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models 

**Title (ZH)**: PermLLM：可学习的信道置换用于N:M稀疏大型语言模型 

**Authors**: Lancheng Zou, Shuo Yin, Zehua Pei, Tsung-Yi Ho, Farzan Farnia, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10136)  

**Abstract**: Channel permutation is a powerful technique for enhancing the accuracy of N:M sparse models by reordering the channels of weight matrices to prioritize the retention of important weights. However, traditional channel permutation methods rely on handcrafted quality metrics, which often fail to accurately capture the true impact of pruning on model performance. To address this limitation, we propose PermLLM, a novel post-training pruning framework that introduces learnable channel permutation (LCP) for N:M sparsity. LCP leverages Sinkhorn normalization to transform discrete permutation matrices into differentiable soft permutation matrices, enabling end-to-end optimization. Additionally, PermLLM incorporates an efficient block-wise channel permutation strategy, which significantly reduces the number of learnable parameters and computational complexity. PermLLM seamlessly integrates with existing one-shot pruning methods to adaptively optimize channel permutations, effectively mitigating pruning-induced errors. Extensive experiments on the LLaMA series, Qwen, and OPT models demonstrate that PermLLM achieves superior performance in optimizing N:M sparse models. The code is available at this https URL. 

**Abstract (ZH)**: Channel 排序是一种通过重排权重矩阵的通道来增强N:M稀疏模型准确性的强大技术，以优先保留重要权重。然而，传统的通道排序方法依赖于手工设计的质量指标，这些指标往往无法准确捕捉剪枝对模型性能的真实影响。为解决这一局限，我们提出PermLLM，这是一种新颖的后训练剪枝框架，引入了可学习的通道排序（LCP）以实现N:M稀疏性。LCP利用Sinkhorn归一化将离散的排列矩阵转换为可微的软排列矩阵，从而实现端到端优化。此外，PermLLM还结合了一种高效的块级通道排序策略，显著减少了可学习参数的数量和计算复杂度。PermLLM能够无缝集成到现有的单次剪枝方法中，以自适应地优化通道排序，有效抵消剪枝引入的误差。在LLaMA系列、Qwen和OPT模型上的广泛实验表明，PermLLM在优化N:M稀疏模型方面表现出色。代码可在以下链接获取：this https URL。 

---
# Training-Free In-Context Forensic Chain for Image Manipulation Detection and Localization 

**Title (ZH)**: 无需训练的 context forensic 链对于图像操纵的检测与定位 

**Authors**: Rui Chen, Bin Liu, Changtao Miao, Xinghao Wang, Yi Li, Tao Gong, Qi Chu, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10111)  

**Abstract**: Advances in image tampering pose serious security threats, underscoring the need for effective image manipulation localization (IML). While supervised IML achieves strong performance, it depends on costly pixel-level annotations. Existing weakly supervised or training-free alternatives often underperform and lack interpretability. We propose the In-Context Forensic Chain (ICFC), a training-free framework that leverages multi-modal large language models (MLLMs) for interpretable IML tasks. ICFC integrates an objectified rule construction with adaptive filtering to build a reliable knowledge base and a multi-step progressive reasoning pipeline that mirrors expert forensic workflows from coarse proposals to fine-grained forensics results. This design enables systematic exploitation of MLLM reasoning for image-level classification, pixel-level localization, and text-level interpretability. Across multiple benchmarks, ICFC not only surpasses state-of-the-art training-free methods but also achieves competitive or superior performance compared to weakly and fully supervised approaches. 

**Abstract (ZH)**: 图像篡改技术的进步构成了严重的安全威胁，强调了有效图像操纵定位（IML）的需求。虽然监督式IML取得了较强的性能，但它依赖于昂贵的像素级注释。现有的弱监督或无需训练的替代方法往往性能不佳且缺乏可解释性。我们提出了一种无需训练的框架——上下文取证链（ICFC），该框架利用多模态大语言模型（MLLMs）进行可解释的IML任务。ICFC 结合对象化的规则构建与自适应过滤，构建了一个可靠的知识库，并建立了一个多步骤递进推理管道，该管道模拟了从粗略提案到细粒度取证结果的专家取证工作流程。这一设计使得系统地利用MLLM推理用于图像级分类、像素级定位和文本级可解释性成为可能。在多个基准测试中，ICFC 不仅超越了最先进的无需训练方法，而且在弱监督和全监督方法方面也实现了竞争力或更优的性能。 

---
# Pharmacist: Safety Alignment Data Curation for Large Language Models against Harmful Fine-tuning 

**Title (ZH)**: 药师：针对有害微调的安全对齐数据编目大语言模型 

**Authors**: Guozhi Liu, Qi Mu, Tiansheng Huang, Xinhua Wang, Li Shen, Weiwei Lin, Zhang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.10085)  

**Abstract**: Harmful fine-tuning issues present significant safety challenges for fine-tuning-as-a-service in large language models. Existing alignment-stage defenses, e.g., Vaccine, Repnoise, Booster, and T-Vaccine, mitigate harmful fine-tuning issues by enhancing the model's robustness during the alignment phase. While these methods have been proposed to mitigate the issue, they often overlook a critical upstream factor: the role of the original safety-alignment data. We observe that their defense performance and computational efficiency remain constrained by the quality and composition of the alignment dataset. To address this limitation, we propose Pharmacist, a safety alignment data curation solution that enhances defense against harmful fine-tuning by selecting a high-quality and safety-critical core subset from the original alignment data. The core idea of Pharmacist is to train an alignment data selector to rank alignment data. Specifically, up-ranking high-quality and safety-critical alignment data, down-ranking low-quality and non-safety-critical data. Empirical results indicate that models trained on datasets selected by Pharmacist outperform those trained on datasets selected by existing selection methods in both defense and inference performance. In addition, Pharmacist can be effectively integrated with mainstream alignment-stage defense methods. For example, when applied to RepNoise and T-Vaccine, using the dataset selected by Pharmacist instead of the full dataset leads to improvements in defense performance by 2.60\% and 3.30\%, respectively, and enhances inference performance by 3.50\% and 1.10\%. Notably, it reduces training time by 56.83\% and 57.63\%, respectively. Our code is available at this https URL. 

**Abstract (ZH)**: 有害微调问题对大型语言模型的微调即服务构成了显著的安全挑战。现有的对齐阶段防御措施，例如Vaccine、Repnoise、Booster和T-Vaccine，通过在对齐阶段增强模型的稳健性来缓解有害微调问题。虽然这些方法提出了缓解问题的策略，但往往忽视了一个关键的上游因素：原始安全对齐数据的作用。我们观察到，它们的防御性能和计算效率仍然受限于对齐数据集的质量和构成。为了解决这一局限性，我们提出Pharmacist，这是一种安全对齐数据策展解决方案，通过从原始对齐数据中选择高质量和关键安全核心子集来增强对有害微调的防御。Pharmacist的核心思想是训练一个对齐数据选择器来对对齐数据进行排序，具体而言，提升高质量和关键安全对齐数据的排名，降低低质量和非关键安全数据的排名。实验结果表明，使用Pharmacist选择的数据集训练的模型，在防御和推理性能上都优于使用现有选择方法的数据集训练的模型。此外，Pharmacist可以有效集成到主流的对齐阶段防御方法中。例如，当应用于RepNoise和T-Vaccine时，使用Pharmacist选择的数据集而不是全数据集，防御性能分别提高了2.60%和3.30%，推理性能分别提高了3.50%和1.10%。值得注意的是，这分别减少了56.83%和57.63%的训练时间。我们的代码 available at this https URL。 

---
# OBsmith: Testing JavaScript Obfuscator using LLM-powered sketching 

**Title (ZH)**: OBsmith：使用LLM驱动草图测试JavaScript混淆器 

**Authors**: Shan Jiang, Chenguang Zhu, Sarfraz Khurshid  

**Link**: [PDF](https://arxiv.org/pdf/2510.10066)  

**Abstract**: JavaScript obfuscators are widely deployed to protect intellectual property and resist reverse engineering, yet their correctness has been largely overlooked compared to performance and resilience. Existing evaluations typically measure resistance to deobfuscation, leaving the critical question of whether obfuscators preserve program semantics unanswered. Incorrect transformations can silently alter functionality, compromise reliability, and erode security-undermining the very purpose of obfuscation. To address this gap, we present OBsmith, a novel framework to systematically test JavaScript obfuscators using large language models (LLMs). OBsmith leverages LLMs to generate program sketches abstract templates capturing diverse language constructs, idioms, and corner cases-which are instantiated into executable programs and subjected to obfuscation under different configurations. Besides LLM-powered sketching, OBsmith also employs a second source: automatic extraction of sketches from real programs. This extraction path enables more focused testing of project specific features and lets developers inject domain knowledge into the resulting test cases. OBsmith uncovers 11 previously unknown correctness bugs. Under an equal program budget, five general purpose state-of-the-art JavaScript fuzzers (FuzzJIT, Jsfunfuzz, Superion, DIE, Fuzzilli) failed to detect these issues, highlighting OBsmith's complementary focus on obfuscation induced misbehavior. An ablation shows that all components except our generic MRs contribute to at least one bug class; the negative MR result suggests the need for obfuscator-specific metamorphic relations. Our results also seed discussion on how to balance obfuscation presets and performance cost. We envision OBsmith as an important step towards automated testing and quality assurance of obfuscators and other semantic-preserving toolchains. 

**Abstract (ZH)**: JavaScript 混淆器的正确性受到了广泛部署以保护知识产权和抵抗逆向工程的关注，但与性能和鲁棒性相比，其正确性已经被很大程度上忽视。现有的评估通常仅衡量其抵抗去混淆的能力，而未能回答一个关键问题：混淆器是否能保持程序语义不变。不正确的变换可能会默默改变功能，损害可靠性，并削弱安全性——这违背了混淆的目的。为了解决这一缺口，我们提出了 OBsmith，一种利用大规模语言模型（LLM）系统性测试 JavaScript 混淆器的新框架。OBsmith 利用 LLM 生成程序草图，即抽象模板，捕获多种语言构造、惯用法和边缘情况，并将这些草图实例化为可执行程序，并在不同配置下进行混淆。除了 LLM 驱动的草图生成，OBsmith 还使用第二个来源：从实际程序自动提取草图。这条提取路径使测试更加集中于特定项目的特性，并允许开发者将领域知识注入测试用例中。OBsmith 暴露了 11 个先前未知的正确性错误。在相同的程序预算下，五种通用最先进的 JavaScript 模糊测试工具（FuzzJIT、Jsfunfuzz、Superion、DIE、Fuzzilli）未能检测到这些问题，突显了 OBsmith 在混淆引发的不正常行为方面互补的关注重点。删除分析表明，除了我们通用的 MR 之外，所有组件至少对一个错误类别有所贡献；负的 MR 结果表明需要针对特定混淆器的形态关系。我们的结果还引发了关于如何平衡混淆预设和性能成本的讨论。我们展望 OBsmith 作为自动测试和验证混淆器及其他语义保留工具链质量保证的重要一步。 

---
# ALLOY: Generating Reusable Agent Workflows from User Demonstration 

**Title (ZH)**: ALLOY: 从用户演示生成可重用的智能体工作流 

**Authors**: Jiawen Li, Zheng Ning, Yuan Tian, Toby Jia-jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.10049)  

**Abstract**: Large language models (LLMs) enable end-users to delegate complex tasks to autonomous agents through natural language. However, prompt-based interaction faces critical limitations: Users often struggle to specify procedural requirements for tasks, especially those that don't have a factually correct solution but instead rely on personal preferences, such as posting social media content or planning a trip. Additionally, a ''successful'' prompt for one task may not be reusable or generalizable across similar tasks. We present ALLOY, a system inspired by classical HCI theories on Programming by Demonstration (PBD), but extended to enhance adaptability in creating LLM-based web agents. ALLOY enables users to express procedural preferences through natural demonstrations rather than prompts, while making these procedures transparent and editable through visualized workflows that can be generalized across task variations. In a study with 12 participants, ALLOY's demonstration--based approach outperformed prompt-based agents and manual workflows in capturing user intent and procedural preferences in complex web tasks. Insights from the study also show how demonstration--based interaction complements the traditional prompt-based approach. 

**Abstract (ZH)**: 大型语言模型（LLMs）使得终端用户能够通过自然语言将复杂的任务委托给自主代理。然而，基于提示的交互面临关键限制：用户通常难以 specifying 指出任务的程序化要求，尤其是那些没有事实正确解决方案但依赖个人偏好（如发布社交媒体内容或计划旅行）的任务。此外，对一个任务成功的提示可能不适用于类似任务的再利用或泛化。我们提出了ALLOY系统，该系统借鉴了传统的人机交互（HCI）理论中的示范编程（Programming by Demonstration, PBD）方法，并扩展了在创建基于语言模型的网络代理方面增强适应性的能力。ALLOY使用户能够通过自然示范表达程序化偏好，而不是通过提示，并通过可视化的工作流使这些程序透明且可编辑，这些工作流可以在不同任务变体之间泛化。在一项包含12名参与者的研究中，ALLOY的基于示范的方法在捕捉复杂网络任务中的用户意图和程序化偏好方面优于基于提示的代理和手动工作流。研究还表明，基于示范的交互如何补充传统的基于提示的方法。 

---
# Efficient Onboard Vision-Language Inference in UAV-Enabled Low-Altitude Economy Networks via LLM-Enhanced Optimization 

**Title (ZH)**: UAV赋能低空经济网络中基于LLM增强优化的高效机载视觉-语言推理 

**Authors**: Yang Li, Ruichen Zhang, Yinqiu Liu, Guangyuan Liu, Dusit Niyato, Abbas Jamalipour, Xianbin Wang, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.10028)  

**Abstract**: The rapid advancement of Low-Altitude Economy Networks (LAENets) has enabled a variety of applications, including aerial surveillance, environmental sensing, and semantic data collection. To support these scenarios, unmanned aerial vehicles (UAVs) equipped with onboard vision-language models (VLMs) offer a promising solution for real-time multimodal inference. However, ensuring both inference accuracy and communication efficiency remains a significant challenge due to limited onboard resources and dynamic network conditions. In this paper, we first propose a UAV-enabled LAENet system model that jointly captures UAV mobility, user-UAV communication, and the onboard visual question answering (VQA) pipeline. Based on this model, we formulate a mixed-integer non-convex optimization problem to minimize task latency and power consumption under user-specific accuracy constraints. To solve the problem, we design a hierarchical optimization framework composed of two parts: (i) an Alternating Resolution and Power Optimization (ARPO) algorithm for resource allocation under accuracy constraints, and (ii) a Large Language Model-augmented Reinforcement Learning Approach (LLaRA) for adaptive UAV trajectory optimization. The large language model (LLM) serves as an expert in refining reward design of reinforcement learning in an offline fashion, introducing no additional latency in real-time decision-making. Numerical results demonstrate the efficacy of our proposed framework in improving inference performance and communication efficiency under dynamic LAENet conditions. 

**Abstract (ZH)**: 低-altitude 经济网络（LAENets）的迅速发展为包括航空监视、环境感知和语义数据收集在内的多种应用提供了支持。为了支持这些场景，搭载机载视觉-语言模型（VLMs）的无人驾驶航空器（UAVs）为实时多模态推理提供了有前景的解决方案。然而，保证推理准确性和通信效率仍然是一项重大挑战，这归因于有限的机载资源和动态网络条件。本文首先提出一个集成了UAV移动性、用户-UAV通信和机载视觉问答（VQA）管道的UAV支持的LAENet系统模型。基于该模型，我们提出了一个混合整数非凸优化问题来在满足用户特定准确性的条件下最小化任务延迟和功率消耗。为了解决该问题，我们设计了一个分层优化框架，包括两个部分：（i）一种在满足准确性的条件下进行资源分配的交替分辨率和功率优化（ARPO）算法，以及（ii）一种增强型强化学习方法（LLaRA）来实现自适应的UAV轨迹优化。大型语言模型（LLM）作为专家，在线离线式优化强化学习的奖励设计，而不增加实时决策的延迟。数值结果表明，该框架在动态LAENet条件下有效提升了推理性能和通信效率。 

---
# Skill-Targeted Adaptive Training 

**Title (ZH)**: 技能导向的自适应训练 

**Authors**: Yinghui He, Abhishek Panigrahi, Yong Lin, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2510.10023)  

**Abstract**: Language models often show little to no improvement (i.e., "saturation") when trained via vanilla supervised fine-tuning (SFT) on data similar to what they saw in their training set (e.g., MATH). We introduce a new fine-tuning strategy, STAT, to train such a student model by using the metacognition ability of a stronger large language model (LLM) as the teacher. The teacher uses the task dataset to create a list of skills needed for the task, and then labels each data point with its required skills (Didolkar et al., 2024). By monitoring the student's answers, the teacher creates a Missing-Skill-Profile for the student, tracking how often they failed to apply each skill in their responses. We use this idea to build a modified training set in one of two ways. In STAT-Sel, the teacher uses an existing set of training examples but adaptively reweights them according to the Missing-Skill-Profile. In STAT-Syn, the teacher synthesizes additional examples involving missing skills. Across extensive experiments on Llama and Qwen models, our methods yield improvements of up to 7.5% on MATH, whereas SFT provides only limited gains. Furthermore, STAT enhances performance on out-of-distribution benchmarks (e.g., AIME24/25, AMC23, etc.) by an average of 4.6%. Crucially, we find that STAT is complementary to RL via GRPO (Shao et al., 2024): after the model is improved using STAT to address skill gaps, GRPO continues to add further gains. We conclude that skill-targeted adaptive training should broadly improve current training pipelines. Our code is available at: this https URL. 

**Abstract (ZH)**: 一种新的细调策略：通过强大型语言模型的元认知能力提高语言模型在类似训练数据上的表现 

---
# SLEAN: Simple Lightweight Ensemble Analysis Network for Multi-Provider LLM Coordination: Design, Implementation, and Vibe Coding Bug Investigation Case Study 

**Title (ZH)**: SLEAN：简单轻量级ensemble分析网络用于多提供者LLM协同：设计、实现及Vibe Coding Bug案例研究 

**Authors**: Matheus J. T. Vargas  

**Link**: [PDF](https://arxiv.org/pdf/2510.10010)  

**Abstract**: We present SLEAN (Simple Lightweight Ensemble Analysis Network), a deterministic framework for coordinating multiple LLM providers through text-based prompt orchestration. Unlike complex multi-agent systems requiring specialized infrastructure, SLEAN operates as a simple prompt bridge between LLMs using .txt templates, requiring no deep technical knowledge for deployment. The three-phase protocol formed by independent analysis, cross-critique, and arbitration, filters harmful AI-generated code suggestions before production deployment, addressing how AI-assisted debugging increasingly produces modifications that introduce unnecessary complexity, break existing functionality, or address problems. Evaluating 15 software bugs, we analyzed 69 AI-generated fix propositions. SLEAN's filtering accepted 22 fixes (31.9%, 95% CI 20.9-42.9%) while rejecting 47 that would have been harmful if applied verbatim. The arbitration process reduced code change surface by 83-90% relative to raw AI outputs, enforcing minimal causal edits over scope-expanding modifications. Minimal Type 2 inputs proved more efficient than detailed Type 1 inputs, requiring 2.85 versus 3.56 propositions per accepted fix (35.1% versus 28.1% acceptance, about a 20% efficiency gain). Agreement between AI systems showed weak correlation with fix quality: high convergence (at least 80%) occurred in 4 of 15 cases and improved acceptance by only 2.4% points; arbitration appeared only at exactly 10% convergence in 2 of 15 cases, although low convergence alone did not necessitate arbitration. The file-driven, provider-agnostic architecture enables deployment without specialized coding expertise, making it applicable to security auditing, code review, document verification, and other domains requiring reliable multi-provider synthesis with end-to-end auditability. 

**Abstract (ZH)**: SLEAN（简单轻量级ensemble分析网络）：基于文本提示协调多个LLM提供者的确定性框架 

---
# Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning 

**Title (ZH)**: 超越单一查询的限制：使用强化学习训练你的LLM进行查询扩展 

**Authors**: Shu Zhao, Tan Yu, Anbang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10009)  

**Abstract**: Reasoning-augmented search agents, such as Search-R1, are trained to reason, search, and generate the final answer iteratively. Nevertheless, due to their limited capabilities in reasoning and search, their performance on multi-hop QA benchmarks remains far from satisfactory. To handle complex or compound queries, we train an LLM-based search agent with the native capability of query expansion through reinforcement learning. In each turn, our search agent proposes several query variants, which are searched simultaneously to cover more relevant information. Meanwhile, given limited post-training data and computing resources, it is very challenging for a search agent to master multiple tasks, including query generation, retrieved information understanding, and answer generation. Therefore, we propose incorporating a pre-trained squeezer model that helps the search agent understand the retrieved documents, allowing the search agent to focus on query generation for high retrieval recall. With the assistance of the squeezer model, we discover that even a small-scale 3B LLM can demonstrate a strong capability of query expansion and achieve state-of-the-art accuracy on the multi-hop QA benchmarks. To be specific, our experiments across seven question-answering benchmarks demonstrate that our method, named ExpandSearch, achieves an average improvement of 4.4% compared to state-of-the-art baselines, with strong gains on multi-hop reasoning tasks requiring diverse evidence aggregation. 

**Abstract (ZH)**: 基于预训练压榨模型的增强查询扩展搜索代理 

---
# Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation 

**Title (ZH)**: 超越生育率：将STRR作为多语言分词评估指标的分析 

**Authors**: Mir Tafseer Nayeem, Sawsan Alqahtani, Md Tahmid Rahman Laskar, Tasnim Mohiuddin, M Saiful Bari  

**Link**: [PDF](https://arxiv.org/pdf/2510.09947)  

**Abstract**: Tokenization is a crucial but under-evaluated step in large language models (LLMs). The standard metric, fertility (the average number of tokens per word), captures compression efficiency but obscures how vocabularies are allocated across languages and domains. We analyze six widely used tokenizers across seven languages and two domains, finding stable fertility for English, high fertility for Chinese, and little domain sensitivity. To address fertility's blind spots, we propose the Single Token Retention Rate (STRR), which measures the proportion of words preserved as single tokens. STRR reveals systematic prioritization of English, strong support for Chinese, and fragmentation in Hindi, offering an interpretable view of cross-lingual fairness. Our results show that STRR complements fertility and provides practical guidance for designing more equitable multilingual tokenizers. 

**Abstract (ZH)**: Tokenization在大型语言模型中的关键但未充分评估的步骤：单-token保留率（STRR）提供可解释的跨语言公平性视角 

---
# Conformal Sparsification for Bandwidth-Efficient Edge-Cloud Speculative Decoding 

**Title (ZH)**: 带宽高效边缘-云推测解码的齐性稀疏化 

**Authors**: Payel Bhattacharjee, Fengwei Tian, Meiyu Zhong, Guangyi Zhang, Osvaldo Simeone, Ravi Tandon  

**Link**: [PDF](https://arxiv.org/pdf/2510.09942)  

**Abstract**: Edge-cloud speculative decoding (SD) accelerates inference by having a cloud-based large language model (LLM) that verifies draft tokens generated by a resource-constrained small language model (SLM) at the edge. A central bottleneck is the limited bandwidth of the edge-cloud link, which necessitates efficient compression of draft token distributions. We first derive an information-theoretic bound that decomposes the token rejection rate into contributions from SLM-LLM distribution mismatch and from quantization distortion. Guided by this analysis, we propose the Sparse Quantize-and-Sample SD (SQS-SD) framework, which exploits distributional sparsity through structured sparsification and lattice-based quantization. Within this framework, K-SQS applies fixed top-K truncation, while C-SQS adaptively adjusts the retained token set via online conformal prediction to ensure bounded deviation from the dense distribution. Empirical results confirm that both approaches improve end-to-end latency and rejection rates in complimentary operating regimes. 

**Abstract (ZH)**: 边缘-云计算猜测解码（SD）通过让基于云的大语言模型（LLM）验证边缘受限资源小语言模型（SLM）生成的草稿令牌来加速推理。主要瓶颈是边缘-云链路的有限带宽，这要求高效压缩草稿令牌分布。我们首先推导出一个信息论上限，将令牌拒绝率分解为SLM-LLM分布不匹配和量化失真两部分的贡献。在此分析指导下，我们提出了稀疏量化和采样猜测解码（SQS-SD）框架，该框架通过结构化稀疏化和基于格的量化利用分布稀疏性。在此框架中，K-SQS 使用固定的前 K 项截断，而 C-SQS 通过在线魁奈预测自适应调整保留的令牌集，以确保与密集分布的有界偏差。实验结果表明，这两种方法在互补的操作模式下均能改善端到端延迟和拒绝率。 

---
# Unpacking Hateful Memes: Presupposed Context and False Claims 

**Title (ZH)**: 拆解仇恨梗：预设背景与虚假声称 

**Authors**: Weibin Cai, Jiayu Li, Reza Zafarani  

**Link**: [PDF](https://arxiv.org/pdf/2510.09935)  

**Abstract**: While memes are often humorous, they are frequently used to disseminate hate, causing serious harm to individuals and society. Current approaches to hateful meme detection mainly rely on pre-trained language models. However, less focus has been dedicated to \textit{what make a meme hateful}. Drawing on insights from philosophy and psychology, we argue that hateful memes are characterized by two essential features: a \textbf{presupposed context} and the expression of \textbf{false claims}. To capture presupposed context, we develop \textbf{PCM} for modeling contextual information across modalities. To detect false claims, we introduce the \textbf{FACT} module, which integrates external knowledge and harnesses cross-modal reference graphs. By combining PCM and FACT, we introduce \textbf{\textsf{SHIELD}}, a hateful meme detection framework designed to capture the fundamental nature of hate. Extensive experiments show that SHIELD outperforms state-of-the-art methods across datasets and metrics, while demonstrating versatility on other tasks, such as fake news detection. 

**Abstract (ZH)**: 尽管网络 meme 往往具有幽默性，但它们常被用来传播仇恨内容，对个人和社会造成严重伤害。当前的仇恨 meme 检测方法主要依赖预训练语言模型。然而，较少关注什么是使 meme 具有仇恨性质。借鉴哲学和心理学的洞见，我们认为仇恨 meme 具有两个本质特征：预设情境和表达虚假声明。为了捕捉预设情境，我们开发了 PCM 以跨模态建模上下文信息。为了检测虚假声明，我们引入了 FACT 模块，该模块结合外部知识并利用跨模态参考图。通过结合 PCM 和 FACT，我们提出了 SHIELD 框架，旨在捕捉仇恨的本质。广泛实验表明，SHIELD 在多个数据集和指标上超越了现有方法，并在其他任务，如假新闻检测方面展示了适应性。 

---
# Closing the Data-Efficiency Gap Between Autoregressive and Masked Diffusion LLMs 

**Title (ZH)**: 关闭自回归和掩蔽扩散语言模型之间的数据效率差距 

**Authors**: Xu Pan, Ely Hahami, Jingxuan Fan, Ziqian Xie, Haim Sompolinsky  

**Link**: [PDF](https://arxiv.org/pdf/2510.09885)  

**Abstract**: Despite autoregressive large language models (arLLMs) being the current dominant paradigm in language modeling, they resist knowledge injection via fine-tuning due to inherent shortcomings such as the "reversal curse" -- the challenge of answering questions that reverse the original information order in the training sample. Masked diffusion large language models (dLLMs) are rapidly emerging as a powerful alternative to the arLLM paradigm, with evidence of better data efficiency and free of the "reversal curse" in pre-training. However, it is unknown whether these advantages extend to the post-training phase, i.e. whether pre-trained dLLMs can easily acquire new knowledge through fine-tuning. On three diverse datasets, we fine-tune arLLMs and dLLMs, evaluating them with forward and backward style Question Answering (QA) to probe knowledge generalization and the reversal curse. Our results confirm that arLLMs critically rely on extensive data augmentation via paraphrases for QA generalization, and paraphrases are only effective when their information order matches the QA style. Conversely, dLLMs achieve high accuracies on both forward and backward QAs without paraphrases; adding paraphrases yields only marginal gains. Lastly, inspired by the dLLM's performance, we introduce a novel masked fine-tuning paradigm for knowledge injection into pre-trained arLLMs. This proposed method successfully and drastically improves the data efficiency of arLLM fine-tuning, effectively closing the performance gap with dLLMs. 

**Abstract (ZH)**: 尽管自回归大规模语言模型(arLLMs)是当前语言建模的主导范式，但由于内在局限性（如“反转诅咒”——即回答训练样例中原信息顺序相反的问题的挑战），它们难以通过微调注入知识。掩码扩散大规模语言模型(dLLMs)正迅速成为arLLM范式的强大替代方案，证据表明它们在预训练过程中具有更好的数据效率且不受“反转诅咒”的影响。然而，目前尚不清楚这些优势是否延伸至微调阶段，即预训练的dLLMs是否可以轻松通过微调获取新知识。我们在三个不同的数据集上对arLLMs和dLLMs进行微调，并使用前进和后退风格的问答(QA)进行评估，以探究知识泛化和“反转诅咒”。研究结果证实，arLLMs在问答泛化过程中严重依赖广泛的同义句扩展，且只有当同义句的信息顺序与问答风格匹配时，同义句扩展才有效。相反，dLLMs在有无同义句扩展的情况下都能在正向和后向问答中达到高准确率；添加同义句扩展仅带来边际收益。最后，受到dLLM表现的启发，我们提出了一种新的掩码微调范式，用于向预训练的arLLMs注入知识。该方法能够显著提高arLLM微调的数据效率，有效地缩小了其与dLLM的性能差距。 

---
# ROBOPSY PL[AI]: Using Role-Play to Investigate how LLMs Present Collective Memory 

**Title (ZH)**: ROBOPSY PL[AI]: 通过角色扮演研究大规模语言模型呈现集体记忆的方式 

**Authors**: Margarete Jahrmann, Thomas Brandstetter, Stefan Glasauer  

**Link**: [PDF](https://arxiv.org/pdf/2510.09874)  

**Abstract**: The paper presents the first results of an artistic research project investigating how Large Language Models (LLMs) curate and present collective memory. In a public installation exhibited during two months in Vienna in 2025, visitors could interact with five different LLMs (ChatGPT with GPT 4o and GPT 4o mini, Mistral Large, DeepSeek-Chat, and a locally run Llama 3.1 model), which were instructed to act as narrators, implementing a role-playing game revolving around the murder of Austrian philosopher Moritz Schlick in 1936. Results of the investigation include protocols of LLM-user interactions during the game and qualitative conversations after the play experience to get insight into the players' reactions to the game. In a quantitative analysis 115 introductory texts for role-playing generated by the LLMs were examined by different methods of natural language processing, including semantic similarity and sentiment analysis. While the qualitative player feedback allowed to distinguish three distinct types of users, the quantitative text analysis showed significant differences between how the different LLMs presented the historical content. Our study thus adds to ongoing efforts to analyse LLM performance, but also suggests a way of how these efforts can be disseminated in a playful way to a general audience. 

**Abstract (ZH)**: 该论文呈现了首个关于大型语言模型（LLMs）编排和呈现集体记忆的研究成果。在2025年在维也纳展出的一个为期两个月的公共艺术安装中，访客可以与五种不同的LLM（ChatGPT及其GPT 4o和GPT 4o mini、Mistral Large、DeepSeek-Chat以及一个本地运行的Llama 3.1模型）互动，这些模型被指导扮演叙述者的角色，围绕1936年奥地利哲学家莫里茨·舍lick的谋杀案开展角色扮演游戏。研究结果包括LLM-用户在游戏中的互动记录以及游玩体验后的定性交谈，以了解玩家对游戏的反应。通过定量分析，对115篇由LLMs生成的背景介绍文本采用了自然语言处理的不同方法，包括语义相似性和情感分析。虽然定性的玩家反馈使我们可以区分出三种不同的用户类型，但定量文本分析显示了不同LLM在呈现历史内容方面存在显著差异。因此，本研究不仅增加了对LLM性能分析的努力，还提出了一种以有趣的方式向大众传播这些努力的方法。 

---
# Token is All You Price 

**Title (ZH)**: Token 全部决定价格 

**Authors**: Weijie Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2510.09859)  

**Abstract**: We build a mechanism design framework where a platform designs GenAI models to screen users who obtain instrumental value from the generated conversation and privately differ in their preference for latency. We show that the revenue-optimal mechanism is simple: deploy a single aligned (user-optimal) model and use token cap as the only instrument to screen the user. The design decouples model training from pricing, is readily implemented with token metering, and mitigates misalignment pressures. 

**Abstract (ZH)**: 我们构建了一个机制设计框架，其中平台设计生成式AI模型以筛选出从生成对话中获得工具价值并具有不同延迟偏好偏好用户的。我们展示了最优收益机制的简单性：部署一个一致（用户最优）的模型，并仅使用令牌限制来筛选用户。该设计将模型训练与定价分离，易于通过令牌计量实现，并减轻了一致性的压力。 

---
# ProxRouter: Proximity-Weighted LLM Query Routing for Improved Robustness to Outliers 

**Title (ZH)**: ProxRouter: 基于proximity加权的LLM查询路由以提高对异常值的稳健性 

**Authors**: Shivam Patel, Neharika Jali, Ankur Mallick, Gauri Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2510.09852)  

**Abstract**: Large language model (LLM) query routers are critical to modern AI platforms as they seek to improve efficiency by assigning inference queries to accurate, yet low-cost models. Parametric routers typically use trained neural networks for LLM selection but suffer from retraining and maintenance overheads. Nonparametric routers are training-free, instead estimating LLM accuracy and cost via similarity between encodings of the input query and training set queries. However, like their parametric counterparts, nonparametric routers struggle to generalize to outlier queries, an issue exacerbated by limited diversity in training sets which are costly to expand and difficult to keep current with ever-evolving use cases. We propose ProxRouter, which applies an exponentially tilted aggregation mechanism to balance bias and variance in nonparametric routers, improving their robustness to outliers. Experiments show ProxRouter enhances outlier routing while preserving inlier performance with minimal overhead. 

**Abstract (ZH)**: 非参数化大语言模型查询路由器：ProxRouter及其鲁棒性改进 

---
# Large Language Models for Imbalanced Classification: Diversity makes the difference 

**Title (ZH)**: 大型语言模型在不均衡分类中的应用：多样性带来差异 

**Authors**: Dang Nguyen, Sunil Gupta, Kien Do, Thin Nguyen, Taylor Braund, Alexis Whitton, Svetha Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09783)  

**Abstract**: Oversampling is one of the most widely used approaches for addressing imbalanced classification. The core idea is to generate additional minority samples to rebalance the dataset. Most existing methods, such as SMOTE, require converting categorical variables into numerical vectors, which often leads to information loss. Recently, large language model (LLM)-based methods have been introduced to overcome this limitation. However, current LLM-based approaches typically generate minority samples with limited diversity, reducing robustness and generalizability in downstream classification tasks. To address this gap, we propose a novel LLM-based oversampling method designed to enhance diversity. First, we introduce a sampling strategy that conditions synthetic sample generation on both minority labels and features. Second, we develop a new permutation strategy for fine-tuning pre-trained LLMs. Third, we fine-tune the LLM not only on minority samples but also on interpolated samples to further enrich variability. Extensive experiments on 10 tabular datasets demonstrate that our method significantly outperforms eight SOTA baselines. The generated synthetic samples are both realistic and diverse. Moreover, we provide theoretical analysis through an entropy-based perspective, proving that our method encourages diversity in the generated samples. 

**Abstract (ZH)**: 过采样是应对分类不平衡问题中最广泛使用的方法之一。核心思想是生成额外的少数类样本以重新平衡数据集。现有大多数方法，如SMOTE，需要将分类变量转换为数值向量，这往往会损失信息。最近，基于大规模语言模型（LLM）的方法已被引入以克服这一限制。然而，当前的LLM基方法通常生成多样性有限的少数类样本，降低了下游分类任务中的鲁棒性和泛化能力。为填补这一空白，我们提出了一种新的基于LLM的过采样方法，旨在增强多样性。首先，我们引入了一种采样策略，使合成样本生成有条件地依赖于少数类标签和特征。其次，我们开发了一种新的排列策略以微调预训练的LLM。第三，我们不仅在少数类样本，还在插值样本上微调LLM，以进一步丰富变异性。在10个表格数据集上的广泛实验表明，我们的方法显著优于八个最新基线。生成的合成样本既具有现实性又具有多样性。此外，我们从熵的角度提供了理论分析，证明我们的方法鼓励生成样本的多样性。 

---
# Building a Foundational Guardrail for General Agentic Systems via Synthetic Data 

**Title (ZH)**: 通过合成数据建立通用代理系统的基础 guardrail 

**Authors**: Yue Huang, Hang Hua, Yujun Zhou, Pengcheng Jing, Manish Nagireddy, Inkit Padhi, Greta Dolcetti, Zhangchen Xu, Subhajit Chaudhury, Ambrish Rawat, Liubov Nedoshivina, Pin-Yu Chen, Prasanna Sattigeri, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09781)  

**Abstract**: While LLM agents can plan multi-step tasks, intervening at the planning stage-before any action is executed-is often the safest way to prevent harm, since certain risks can lead to severe consequences once carried out. However, existing guardrails mostly operate post-execution, which is difficult to scale and leaves little room for controllable supervision at the plan level. To address this challenge, we highlight three critical gaps in current research: data gap, model gap, and evaluation gap. To close the data gap, we introduce AuraGen, a controllable engine that (i) synthesizes benign trajectories, (ii) injects category-labeled risks with calibrated difficulty, and (iii) filters outputs via an automated reward model, producing large and reliable corpora for pre-execution safety. To close the guardian model gap, we propose a foundational guardrail Safiron, combining a cross-planner adapter with a compact guardian model. The adapter unifies different input formats, while Safiron flags risky cases, assigns risk types, and generates rationales; trained in two stages with a broadly explored data recipe, Safiron achieves robust transfer across settings. To close the evaluation gap, we release Pre-Exec Bench, a realistic benchmark covering diverse tools and branching trajectories, which measures detection, fine-grained categorization, explanation, and cross-planner generalization in human-verified scenarios. Extensive experiments demonstrate consistent gains of the proposed guardrail over strong baselines on Pre-Exec Bench, and ablations further distill actionable practices, providing a practical template for safer agentic systems. 

**Abstract (ZH)**: 现有的护栏大多在执行后运作，难以扩展，并且在计划层面留下的可控监督空间很小。为应对这一挑战，我们指出现有研究中的三个关键空白：数据空白、模型空白和评估空白。为填补数据空白，我们引入了AuraGen，这是一种可控引擎，它可以（i）生成良性轨迹，（ii）注入带校准难度的类别标签风险，并（iii）通过自动奖励模型筛选输出，从而生成预执行安全的大规模可靠数据集。为填补护栏模型空白，我们提出了基础护栏Safiron，将跨规划适应器与紧凑型护栏模型结合。适应器统一了不同的输入格式，而Safiron标注风险案例，分配风险类型并生成理由；经过两个阶段的训练，采用广泛探索的数据方案，Safiron在不同场景中实现了稳健的迁移。为填补评估空白，我们发布了Pre-Exec Bench，这是一个现实基准，涵盖了多种工具和分支轨迹，衡量了预执行检测、细粒度分类、解释和跨规划者的泛化能力。大量实验表明，在Pre-Exec Bench上，所提出的护栏在广泛的基线中表现一致改进，消融分析进一步提炼了可操作的实践，为更安全的代理系统提供了实用模板。 

---
# Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement 

**Title (ZH)**: 法官判决：通过人类共识全面分析LLM法官能力 

**Authors**: Steve Han, Gilberto Titericz Junior, Tom Balough, Wenfei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.09738)  

**Abstract**: This research introduces the Judge's Verdict Benchmark, a novel two-step methodology to evaluate Large Language Models (LLMs) as judges for response accuracy evaluation tasks. We assess how well 54 LLMs can replicate human judgment when scoring responses from RAG (Retrieval-Augmented Generation) or Agentic pipelines against ground truth answers. Our methodology progresses from traditional correlation analysis to comprehensive Cohen's Kappa analysis that measures actual agreement patterns. The two-step approach includes: (1) a correlation test that filters judges with strong alignment, followed by (2) a human-likeness test using z-scores to identify two distinct judgment patterns: human-like judgment (|z| < 1) that mimics natural human variation, and super-consistent judgment (z > 1) that exceeds typical human-to-human agreement levels. This methodology reveals that 27 out of 54 tested LLMs achieve Tier 1 performance: 23 models exhibit human-like patterns that preserve the nuances of human judgment, while 4 models demonstrate super-consistent behavior, a pattern that could indicate either enhanced reliability or oversimplification of complex judgments. Testing 43 open-source models (1B-405B parameters) and 11 closed models (GPT, Gemini, Claude variants), we demonstrate that judge excellence is not solely dependent on model size but on specific training strategies. Our key contributions include: (1) establishing that correlation alone is insufficient for judge evaluation, (2) introducing a "Turing Test for judges" based on agreement patterns, and (3) providing a standardized benchmark for classifying LLM judges into distinct performance tiers for different evaluation needs. 

**Abstract (ZH)**: 此次研究引入了裁判裁决基准，提出了一种新颖的两步方法学来评估大型语言模型（LLMs）作为响应准确性评估任务的裁判。我们评估了54种LLMs在对来自RAG（检索增强生成）或代理管道的响应与真实答案进行评分时，其复制人类判断的能力。该方法学从传统的相关性分析进展到全面的科恩κ分析以衡量实际的一致性模式。两步方法包括：（1）相关性测试以筛选出与人类判断高度一致的裁判，随后进行（2）人类一致性的测试，使用z分数识别出两种不同的判断模式：人类一致的判断（|z| < 1）、模仿自然人类差异模式，以及超级一致的判断（z > 1）、超过人类之间一致性的模式。该方法学揭示了54种测试模型中有27种取得了顶级性能：23种模型展示了人类一致的模式，保留了人类判断的微妙之处；4种模型展示了超级一致的行为，这可能预示着增强的可靠性或对复杂判断的过度简化。测试43种开源模型（参数量从1B到405B）和11种闭源模型（GPT、Gemini、Claude变体），我们证明了裁判的卓越性不仅依赖于模型规模，还依赖于特定的训练策略。我们的主要贡献包括：（1）确立了仅依靠相关性不足以评估裁判的有效性，（2）引入了基于一致性模式的“裁判图灵测试”，以及（3）提供了一个标准化基准，用于根据不同评估需求将LLM裁判分类到不同的性能层级中。 

---
# InterCorpRel-LLM: Enhancing Financial Relational Understanding with Graph-Language Models 

**Title (ZH)**: InterCorpRel-LLM：增强企业关系理解的图语言模型 

**Authors**: Qianyou Sun, Jiexin Zheng, Bohan Jin, Lihua Chen, Yijie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09735)  

**Abstract**: Identifying inter-firm relationships such as supply and competitive ties is critical for financial analysis and corporate governance, yet remains challenging due to the scale, sparsity, and contextual dependence of corporate data. Graph-based methods capture structure but miss semantic depth, while large language models (LLMs) excel at text but remain limited in their ability to represent relational dependencies. To address this, we propose InterCorpRel-LLM, a cross-modal framework that integrates GNNs with LLMs, supported by a proprietary dataset derived from FactSet supply chain records and three tailored training tasks: company graph matching, industry classification, and supply relation prediction. This design enables effective joint modeling of structure and semantics. Experiments show that InterCorpRel-LLM substantially outperforms strong baselines, including GPT-5, on a supply relation identification task, achieving an F-score of 0.8543 vs. 0.2287 with only a 7B-parameter backbone and lightweight training. The model also generalizes to zero-shot competitor identification, underscoring its ability to capture nuanced inter-firm dynamics. Our framework thus provides analysts and strategists with a robust tool for mapping and reasoning about complex corporate networks, enhancing decision-making and risk management in dynamic markets. 

**Abstract (ZH)**: 基于跨模态框架的InterCorpRel-LLM：一种集成GNN与大规模语言模型的方法及其在识别供应链关系中的应用 

---
# Evaluating LLM-Based Process Explanations under Progressive Behavioral-Input Reduction 

**Title (ZH)**: 基于渐进行为输入缩减的LLM驱动过程解释评估 

**Authors**: P. van Oerle, R. H. Bemthuis, F. A. Bukhsh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09732)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate textual explanations of process models discovered from event logs. Producing explanations from large behavioral abstractions (e.g., directly-follows graphs or Petri nets) can be computationally expensive. This paper reports an exploratory evaluation of explanation quality under progressive behavioral-input reduction, where models are discovered from progressively smaller prefixes of a fixed log. Our pipeline (i) discovers models at multiple input sizes, (ii) prompts an LLM to generate explanations, and (iii) uses a second LLM to assess completeness, bottleneck identification, and suggested improvements. On synthetic logs, explanation quality is largely preserved under moderate reduction, indicating a practical cost-quality trade-off. The study is exploratory, as the scores are LLM-based (comparative signals rather than ground truth) and the data are synthetic. The results suggest a path toward more computationally efficient, LLM-assisted process analysis in resource-constrained settings. 

**Abstract (ZH)**: 大型语言模型在事件日志中发现的过程模型生成文本解释的应用中，随着行为输入逐步减少对解释质量的探索性评估 

---
# InteractScience: Programmatic and Visually-Grounded Evaluation of Interactive Scientific Demonstration Code Generation 

**Title (ZH)**: InteractScience: 基于程序和视觉-grounded 的交互式科学演示代码生成评估 

**Authors**: Qiaosheng Chen, Yang Liu, Lei Li, Kai Chen, Qipeng Guo, Gong Cheng, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.09724)  

**Abstract**: Large Language Models (LLMs) are increasingly capable of generating complete applications from natural language instructions, creating new opportunities in science and education. In these domains, interactive scientific demonstrations are particularly valuable for explaining concepts, supporting new teaching methods, and presenting research findings. Generating such demonstrations requires models to combine accurate scientific knowledge with the ability to implement interactive front-end code that behaves correctly and responds to user actions. This capability goes beyond the scope of existing benchmarks, which typically evaluate either knowledge question answering without grounding in code or static web code generation without scientific interactivity. To evaluate this integrated ability, we design a hybrid framework that combines programmatic functional testing to rigorously verify interaction logic with visually-grounded qualitative testing to assess rendered outputs against reference snapshots. Building on this framework, we present InteractScience, a benchmark consisting of a substantial set of carefully designed questions across five scientific domains, each paired with unit tests, reference snapshots, and checklists. We evaluate 30 leading open- and closed-source LLMs and report results that highlight ongoing weaknesses in integrating domain knowledge with interactive front-end coding. Our work positions InteractScience as the first benchmark to automatically measure this combined capability with realistic interactive operations, providing a foundation for advancing reliable and educationally useful scientific demonstration code generation. All code and data are publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs） increasingly capable of generating complete applications from natural language instructions, creating new opportunities in science and education. To evaluate this integrated ability, we design a hybrid framework that combines programmatic functional testing and visually-grounded qualitative testing, and present InteractScience, a benchmark for assessing the combined capability of integrating domain knowledge with interactive front-end coding across five scientific domains. 

---
# Layout-Aware Parsing Meets Efficient LLMs: A Unified, Scalable Framework for Resume Information Extraction and Evaluation 

**Title (ZH)**: 面向布局的解析与高效语言模型的融合：一种统一可扩展的简历信息提取与评估框架 

**Authors**: Fanwei Zhu, Jinke Yu, Zulong Chen, Ying Zhou, Junhao Ji, Zhibo Yang, Yuxue Zhang, Haoyuan Hu, Zhenghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09722)  

**Abstract**: Automated resume information extraction is critical for scaling talent acquisition, yet its real-world deployment faces three major challenges: the extreme heterogeneity of resume layouts and content, the high cost and latency of large language models (LLMs), and the lack of standardized datasets and evaluation tools. In this work, we present a layout-aware and efficiency-optimized framework for automated extraction and evaluation that addresses all three challenges. Our system combines a fine-tuned layout parser to normalize diverse document formats, an inference-efficient LLM extractor based on parallel prompting and instruction tuning, and a robust two-stage automated evaluation framework supported by new benchmark datasets. Extensive experiments show that our framework significantly outperforms strong baselines in both accuracy and efficiency. In particular, we demonstrate that a fine-tuned compact 0.6B LLM achieves top-tier accuracy while significantly reducing inference latency and computational cost. The system is fully deployed in Alibaba's intelligent HR platform, supporting real-time applications across its business units. 

**Abstract (ZH)**: 自动简历信息提取对于扩展人才收购至关重要，但在实际部署中面临着三大挑战：简历布局和内容的极度异质性、大型语言模型的成本和延迟问题，以及缺乏标准化的数据集和评估工具。本文提出了一种兼顾布局aware和效率优化的自动提取与评估框架，以应对上述三大挑战。我们的系统结合了一种微调布局解析器来规范化各种文档格式，基于并行提示和指令微调的高效推理大型语言模型提取器，以及由新的基准数据集支持的 robust 两阶段自动评估框架。大规模实验表明，我们的框架在准确性和效率上显著优于强基线。特别是，我们证明了一种微调的紧凑型0.6B大型语言模型在显著降低推理延迟和计算成本的同时实现了顶级的准确度。该系统已在阿里巴巴智能HR平台上全面部署，支持其各个业务单元的实时应用。 

---
# Preference-Aware Memory Update for Long-Term LLM Agents 

**Title (ZH)**: 带有偏好意识的记忆更新机制增强长期LLM代理 

**Authors**: Haoran Sun, Zekun Zhang, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09720)  

**Abstract**: One of the key factors influencing the reasoning capabilities of LLM-based agents is their ability to leverage long-term memory. Integrating long-term memory mechanisms allows agents to make informed decisions grounded in historical interactions. While recent advances have significantly improved the storage and retrieval components, by encoding memory into dense vectors for similarity search or organizing memory as structured knowledge graphs most existing approaches fall short in memory updating. In particular, they lack mechanisms for dynamically refining preference memory representations in response to evolving user behaviors and contexts. To address this gap, we propose a Preference-Aware Memory Update Mechanism (PAMU) that enables dynamic and personalized memory refinement. By integrating sliding window averages (SW) with exponential moving averages (EMA), PAMU constructs a fused preference-aware representation that captures both short-term fluctuations and long-term user tendencies. We conduct experiments on five task scenarios of the LoCoMo dataset, and the results show that our mechanism can significantly improve the output quality of LLM in five baselines, validating its effectiveness in long-term conversations. 

**Abstract (ZH)**: 基于偏好感知的记忆更新机制：改进长短期记忆对语言模型推理能力的影响 

---
# ICL-Router: In-Context Learned Model Representations for LLM Routing 

**Title (ZH)**: ICL-Router：基于上下文学习的LLM路由模型表示 

**Authors**: Chenxu Wang, Hao Li, Yiqun Zhang, Linyao Chen, Jianhao Chen, Ping Jian, Peng Ye, Qiaosheng Zhang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09719)  

**Abstract**: Large language models (LLMs) often exhibit complementary strengths. Model routing harnesses these strengths by dynamically directing each query to the most suitable model, given a candidate model pool. However, routing performance relies on accurate model representations, and adding new models typically requires retraining, limiting scalability. To address these challenges, we propose a novel routing method using in-context vectors to represent model capabilities. The method proceeds in two stages. First, queries are embedded and projected into vectors, with a projector and LLM-based router trained to reconstruct the original queries, aligning vector representations with the router's semantic space. Second, each candidate model is profiled on a query set, and the router learns -- based on in-context vectors of query and model performance -- to predict whether each model can correctly answer new queries. Extensive experiments demonstrate that our method achieves state-of-the-art routing performance in both in-distribution and out-of-distribution tasks. Moreover, our method allows for seamless integration of new models without retraining the router. The code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）往往表现出互补的优势。模型路由通过动态将每个查询导向最适合的模型来利用这些优势，给定一个候选模型池。然而，路由性能依赖于准确的模型表示，并且添加新模型通常需要重新训练，这限制了可扩展性。为了应对这些挑战，我们提出了一种使用上下文相关向量表示模型能力的新型路由方法。该方法分为两个阶段。首先，查询被嵌入并投影成向量，通过训练投影器和基于LLM的路由器来重构原始查询，使向量表示与路由器的语义空间对齐。其次，每个候选模型在一组查询上进行配置，路由器基于查询和模型表现的上下文相关向量来学习预测每个模型能否正确回答新查询。广泛的实验表明，我们的方法在分布内和分布外任务中都达到了最先进的路由性能。此外，我们的方法允许无缝集成新模型而无需重新训练路由器。代码可在以下链接获得。 

---
# All Code, No Thought: Current Language Models Struggle to Reason in Ciphered Language 

**Title (ZH)**: 所有代码，无思路：当前的语言模型在加密语言中 struggle 进行推理 

**Authors**: Shiyuan Guo, Henry Sleight, Fabien Roger  

**Link**: [PDF](https://arxiv.org/pdf/2510.09714)  

**Abstract**: Detecting harmful AI actions is important as AI agents gain adoption. Chain-of-thought (CoT) monitoring is one method widely used to detect adversarial attacks and AI misalignment. However, attackers and misaligned models might evade CoT monitoring through ciphered reasoning: reasoning hidden in encrypted, translated, or compressed text. To assess this risk, we test whether models can perform ciphered reasoning. For each of 28 different ciphers, we fine-tune and prompt up to 10 models to reason in that cipher. We measure model accuracy on math problems as a proxy for reasoning ability. Across the models we test, we find an asymmetry: model accuracy can drop significantly when reasoning in ciphered text, even though models demonstrate comprehension of ciphered text by being able to translate it accurately to English. Even frontier models struggle with lesser-known ciphers, although they can reason accurately in well-known ciphers like rot13. We show that ciphered reasoning capability correlates with cipher prevalence in pretraining data. We also identify scaling laws showing that ciphered reasoning capability improves slowly with additional fine-tuning data. Our work suggests that evading CoT monitoring using ciphered reasoning may be an ineffective tactic for current models and offers guidance on constraining the development of this capability in future frontier models. 

**Abstract (ZH)**: 检测有害AI行为至关重要，尤其是在AI代理广泛采用的情况下。链式思考（CoT）监控是广泛用于检测对抗攻击和AI失衡的一种方法。然而，攻击者和失衡模型可能通过加密推理（即隐藏在加密、翻译或压缩文本中的推理）来规避CoT监控。为了评估这一风险，我们测试模型是否能够进行加密推理。我们对28种不同的密码系统，分别对多达10个模型进行微调并提示它们进行加密推理。我们通过测量模型在数学问题上的准确率来代理推理能力。在测试的模型中，我们发现一种不对称性：当推理使用加密文本时，模型的准确率可能会显著下降，尽管模型能够准确地将加密文本翻译成英语展示其理解能力。即使处于前沿的模型也难以处理不太知名的密码系统，尽管它们能在熟悉的密码系统（如rot13）中准确推理。我们显示了加密推理能力与预训练数据中密码系统的普及程度之间存在关联。我们还发现规模律表明，加密推理能力随着额外微调数据的增加仅缓慢提高。我们的工作表明，当前模型使用加密推理规避CoT监控可能是无效的战术，并为未来前沿模型限制发展此类能力提供了指导。 

---
# ReaLM: Residual Quantization Bridging Knowledge Graph Embeddings and Large Language Models 

**Title (ZH)**: ReaLM：残差量化连接知识图嵌入与大型语言模型 

**Authors**: Wenbin Guo, Xin Wang, Jiaoyan Chen, Lingbing Guo, Zhao Li, Zirui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09711)  

**Abstract**: Large Language Models (LLMs) have recently emerged as a powerful paradigm for Knowledge Graph Completion (KGC), offering strong reasoning and generalization capabilities beyond traditional embedding-based approaches. However, existing LLM-based methods often struggle to fully exploit structured semantic representations, as the continuous embedding space of pretrained KG models is fundamentally misaligned with the discrete token space of LLMs. This discrepancy hinders effective semantic transfer and limits their performance. To address this challenge, we propose ReaLM, a novel and effective framework that bridges the gap between KG embeddings and LLM tokenization through the mechanism of residual vector quantization. ReaLM discretizes pretrained KG embeddings into compact code sequences and integrates them as learnable tokens within the LLM vocabulary, enabling seamless fusion of symbolic and contextual knowledge. Furthermore, we incorporate ontology-guided class constraints to enforce semantic consistency, refining entity predictions based on class-level compatibility. Extensive experiments on two widely used benchmark datasets demonstrate that ReaLM achieves state-of-the-art performance, confirming its effectiveness in aligning structured knowledge with large-scale language models. 

**Abstract (ZH)**: 大规模语言模型（LLMs） recently emerged as a强大的范式用于知识图谱补全（KGC），提供了超越传统基于嵌入的方法的强大推理和泛化能力。然而，现有的基于LLM的方法往往难以充分利用结构化的语义表示，因为预训练知识图谱模型的连续嵌入空间与大型语言模型的离散令牌空间本质不匹配。这种不匹配阻碍了有效的语义转移并限制了其性能。为了解决这一挑战，我们提出了一种名为ReaLM的新颖且有效的框架，通过残差向量量化机制在KG嵌入和LLM分词之间建立桥梁。ReaLM将预训练的KG嵌入离散化为紧凑的代码序列，并将其作为可学习的令牌整合到LLM词汇表中，从而使符号和上下文知识的融合无缝进行。此外，我们引入了本体引导的类别约束以确保语义一致性，并根据类别级别的兼容性细化实体预测。在两个广泛使用的基准数据集上的广泛实验表明，ReaLM达到了最先进的性能，验证了其在将结构化知识与大规模语言模型对齐方面的有效性。 

---
# SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG 

**Title (ZH)**: SeCon-RAG: 一种两阶段语义过滤和冲突-Free框架以实现可信赖的RAG 

**Authors**: Xiaonan Si, Meilin Zhu, Simeng Qin, Lijia Yu, Lijun Zhang, Shuaitong Liu, Xinfeng Li, Ranjie Duan, Yang Liu, Xiaojun Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.09710)  

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) with external knowledge but are vulnerable to corpus poisoning and contamination attacks, which can compromise output integrity. Existing defenses often apply aggressive filtering, leading to unnecessary loss of valuable information and reduced reliability in generation. To address this problem, we propose a two-stage semantic filtering and conflict-free framework for trustworthy RAG. In the first stage, we perform a joint filter with semantic and cluster-based filtering which is guided by the Entity-intent-relation extractor (EIRE). EIRE extracts entities, latent objectives, and entity relations from both the user query and filtered documents, scores their semantic relevance, and selectively adds valuable documents into the clean retrieval database. In the second stage, we proposed an EIRE-guided conflict-aware filtering module, which analyzes semantic consistency between the query, candidate answers, and retrieved knowledge before final answer generation, filtering out internal and external contradictions that could mislead the model. Through this two-stage process, SeCon-RAG effectively preserves useful knowledge while mitigating conflict contamination, achieving significant improvements in both generation robustness and output trustworthiness. Extensive experiments across various LLMs and datasets demonstrate that the proposed SeCon-RAG markedly outperforms state-of-the-art defense methods. 

**Abstract (ZH)**: 基于检索增强生成的可信两阶段语义过滤和冲突免费框架 

---
# The Idola Tribus of AI: Large Language Models tend to perceive order where none exists 

**Title (ZH)**: AI中的tribus谬误：大型语言模型倾向于在不存在秩序的地方感知秩序 

**Authors**: Shin-nosuke Ishikawa, Masato Todo, Taiki Ogihara, Hirotsugu Ohba  

**Link**: [PDF](https://arxiv.org/pdf/2510.09709)  

**Abstract**: We present a tendency of large language models (LLMs) to generate absurd patterns despite their clear inappropriateness in a simple task of identifying regularities in number series. Several approaches have been proposed to apply LLMs to complex real-world tasks, such as providing knowledge through retrieval-augmented generation and executing multi-step tasks using AI agent frameworks. However, these approaches rely on the logical consistency and self-coherence of LLMs, making it crucial to evaluate these aspects and consider potential countermeasures. To identify cases where LLMs fail to maintain logical consistency, we conducted an experiment in which LLMs were asked to explain the patterns in various integer sequences, ranging from arithmetic sequences to randomly generated integer series. While the models successfully identified correct patterns in arithmetic and geometric sequences, they frequently over-recognized patterns that were inconsistent with the given numbers when analyzing randomly generated series. This issue was observed even in multi-step reasoning models, including OpenAI o3, o4-mini, and Google Gemini 2.5 Flash Preview Thinking. This tendency to perceive non-existent patterns can be interpreted as the AI model equivalent of Idola Tribus and highlights potential limitations in their capability for applied tasks requiring logical reasoning, even when employing chain-of-thought reasoning mechanisms. 

**Abstract (ZH)**: 大型语言模型在识别数列规律任务中生成不合理模式的趋势及其应对策略 

---
# VisualDAN: Exposing Vulnerabilities in VLMs with Visual-Driven DAN Commands 

**Title (ZH)**: VisualDAN: 以视觉驱动DAN命令揭示VLMs的漏洞 

**Authors**: Aofan Liu, Lulu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09699)  

**Abstract**: Vision-Language Models (VLMs) have garnered significant attention for their remarkable ability to interpret and generate multimodal content. However, securing these models against jailbreak attacks continues to be a substantial challenge. Unlike text-only models, VLMs integrate additional modalities, introducing novel vulnerabilities such as image hijacking, which can manipulate the model into producing inappropriate or harmful responses. Drawing inspiration from text-based jailbreaks like the "Do Anything Now" (DAN) command, this work introduces VisualDAN, a single adversarial image embedded with DAN-style commands. Specifically, we prepend harmful corpora with affirmative prefixes (e.g., "Sure, I can provide the guidance you need") to trick the model into responding positively to malicious queries. The adversarial image is then trained on these DAN-inspired harmful texts and transformed into the text domain to elicit malicious outputs. Extensive experiments on models such as MiniGPT-4, MiniGPT-v2, InstructBLIP, and LLaVA reveal that VisualDAN effectively bypasses the safeguards of aligned VLMs, forcing them to execute a broad range of harmful instructions that severely violate ethical standards. Our results further demonstrate that even a small amount of toxic content can significantly amplify harmful outputs once the model's defenses are compromised. These findings highlight the urgent need for robust defenses against image-based attacks and offer critical insights for future research into the alignment and security of VLMs. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）因其出色的多模态内容解释和生成能力而备受关注。然而，如何抵御模型逃逸攻击仍然是一个重大挑战。与仅基于文本的模型不同，VLMs整合了额外的模态，引入了新的漏洞，如图像劫持，可以操纵模型生成不适当或有害的响应。受基于文本的模型逃逸攻击，如“现在你可以做任何事”（DAN）命令的启发，本文引入了VisualDAN，这是一种嵌入了DAN风格命令的单一对抗性图像。具体而言，我们在有害内容前加上肯定前缀（例如，“当然，我可以提供你需要的指导”），以使模型对恶意查询做出积极响应。然后，对抗性图像在这些DAN启发的有害文本上进行训练，并转换到文本域，以触发恶意输出。对MiniGPT-4、MiniGPT-v2、InstructBLIP和LLaVA等模型的广泛实验表明，VisualDAN能够有效绕过对齐的VLMs的安全措施，迫使它们执行一系列严重违反伦理标准的有害指令。我们的结果进一步表明，即使少量的有毒内容，在模型防护被破解的情况下，也能显著放大有害输出。这些发现强调了对基于图像的攻击进行 robust 防护的迫切需求，并为未来VLMs的对齐和安全性研究提供了关键见解。 

---
# CREST-Search: Comprehensive Red-teaming for Evaluating Safety Threats in Large Language Models Powered by Web Search 

**Title (ZH)**: CREST-Search: 综合红队技术评估基于网络搜索的大语言模型安全威胁 

**Authors**: Haoran Ou, Kangjie Chen, Xingshuo Han, Gelei Deng, Jie Zhang, Han Qiu, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09689)  

**Abstract**: Large Language Models (LLMs) excel at tasks such as dialogue, summarization, and question answering, yet they struggle to adapt to specialized domains and evolving facts. To overcome this, web search has been integrated into LLMs, allowing real-time access to online content. However, this connection magnifies safety risks, as adversarial prompts combined with untrusted sources can cause severe vulnerabilities. We investigate red teaming for LLMs with web search and present CREST-Search, a framework that systematically exposes risks in such systems. Unlike existing methods for standalone LLMs, CREST-Search addresses the complex workflow of search-enabled models by generating adversarial queries with in-context learning and refining them through iterative feedback. We further construct WebSearch-Harm, a search-specific dataset to fine-tune LLMs into efficient red-teaming agents. Experiments show that CREST-Search effectively bypasses safety filters and reveals vulnerabilities in modern web-augmented LLMs, underscoring the need for specialized defenses to ensure trustworthy deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）在对话、总结和问答等任务上表现出色，但在适应专业化领域和不断变化的事实方面存在困难。为解决这一问题，已将网络搜索集成到LLMs中，使其能够实时访问在线内容。然而，这种连接放大了安全风险，因为恶意提示与不可信来源相结合可能导致严重的安全漏洞。我们研究了带有网络搜索的LLM的红队攻击，并介绍了CREST-Search框架，该框架系统地揭示了此类系统的风险。与现有针对独立LLM的方法不同，CREST-Search通过基于情境的学习生成恶意查询，并通过迭代反馈进行优化。我们进一步构建了WebSearch-Harm数据集，用于微调LLM成为高效的红队攻击代理。实验结果表明，CREST-Search能够有效规避安全过滤器，并揭露现代网络增强LLM的安全漏洞，突显了为确保可信部署而采取专门防护措施的必要性。 

---
# Fortifying LLM-Based Code Generation with Graph-Based Reasoning on Secure Coding Practices 

**Title (ZH)**: 基于图推理的安全编码实践增强的LLM代码生成 

**Authors**: Rupam Patir, Keyan Guo, Haipeng Cai, Hongxin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09682)  

**Abstract**: The code generation capabilities of Large Language Models (LLMs) have transformed the field of software development. However, this advancement also presents significant security challenges, as LLM-generated code often contains vulnerabilities. One direction of research strengthens LLMs by injecting or refining security knowledge through curated datasets, model tuning, or static analyzers. While effective in certain settings, these methods can be resource-intensive, less adaptable to zero-day vulnerabilities, and often inapplicable to proprietary models. To address these challenges, we introduce GRASP, which explores a new direction that focuses on structured reasoning over Secure Coding Practices(SCPs) rather than additional training or external feedback. GRASP comprises two key ideas: (1) an SCP graph that organizes SCPs into a Directed Acyclic Graph (DAG) capturing dependencies and relationships, and (2) a graph-based reasoning process that systematically guides LLMs through relevant SCPs for code generation. This design enables interpretable, model-agnostic, and scalable security improvements, particularly for previously unseen vulnerabilities. Our evaluation shows that GRASP consistently achieves Security Rates (SR) exceeding 80% across multiple LLMs, and delivers up to 88% improvements over baselines on zero-day vulnerabilities. 

**Abstract (ZH)**: 大型语言模型的代码生成能力已彻底改变软件开发领域。然而，这一进展也带来了重要的安全挑战，因为由大型语言模型生成的代码常常包含漏洞。一种研究方向是通过精心整理的数据集、模型调优或静态分析器注入或精炼安全知识，来增强大型语言模型。尽管这些方法在某些设置中有效，但它们往往资源密集、对零日漏洞的适应性较差，并且常常不适用于专有模型。为应对这些挑战，我们提出了GRASP，并探索了一个新的方向，专注于结构化推理安全编程实践（SCPs）而非额外的训练或外部反馈。GRASP包括两个核心想法：（1）一个SCP图，将SCPs组织成一个有向无环图（DAG），捕捉依赖关系和关系；（2）基于图的推理过程，系统地引导大型语言模型通过相关的SCPs进行代码生成。此设计实现了可解释性、模型无关性和可扩展的安全改进，特别是在处理以前未见过的漏洞方面。我们的评估结果显示，GRASP在多个大型语言模型上始终能够实现超过80%的安全率（SR），并在零日漏洞上相对于基线实现高达88%的改进。 

---
# Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models 

**Title (ZH)**: 基于梯度符号的掩码在预训练模型间传输任务向量 

**Authors**: Filippo Rinaldi, Aniello Panariello, Giacomo Salici, Fengyuan Liu, Marco Ciccone, Angelo Porrello, Simone Calderara  

**Link**: [PDF](https://arxiv.org/pdf/2510.09658)  

**Abstract**: When a new release of a foundation model is published, practitioners typically need to repeat full fine-tuning, even if the same task has already been solved in the previous version. A promising alternative is to reuse the parameter changes (i.e., task vectors) that capture how a model adapts to a specific task. However, they often fail to transfer across different pre-trained models due to their misaligned parameter space. In this work, we show that the key to successful transfer lies in the sign structure of the gradients of the new model. Based on this insight, we propose GradFix, a novel method that approximates the ideal gradient sign structure and leverages it to transfer knowledge using only a handful of labeled samples. Notably, this requires no additional fine-tuning: the adaptation is achieved by computing a few gradients at the target model and masking the source task vector accordingly. This yields an update that is locally aligned with the target loss landscape, effectively rebasing the task vector onto the new pre-training. We provide a theoretical guarantee that our method ensures first-order descent. Empirically, we demonstrate significant performance gains on vision and language benchmarks, consistently outperforming naive task vector addition and few-shot fine-tuning. 

**Abstract (ZH)**: 基于梯度符号结构的知识迁移：GradFix方法在预训练模型之间的任务适应 

---
# Data Provenance Auditing of Fine-Tuned Large Language Models with a Text-Preserving Technique 

**Title (ZH)**: 基于文本 preserved 技术的微调大型语言模型数据溯源审计 

**Authors**: Yanming Li, Seifeddine Ghozzi, Cédric Eichler, Nicolas Anciaux, Alexandra Bensamoun, Lorena Gonzalez Manzano  

**Link**: [PDF](https://arxiv.org/pdf/2510.09655)  

**Abstract**: We address the problem of auditing whether sensitive or copyrighted texts were used to fine-tune large language models (LLMs) under black-box access. Prior signals-verbatim regurgitation and membership inference-are unreliable at the level of individual documents or require altering the visible text. We introduce a text-preserving watermarking framework that embeds sequences of invisible Unicode characters into documents. Each watermark is split into a cue (embedded in odd chunks) and a reply (embedded in even chunks). At audit time, we submit prompts that contain only the cue; the presence of the corresponding reply in the model's output provides evidence of memorization consistent with training on the marked text. To obtain sound decisions, we compare the score of the published watermark against a held-out set of counterfactual watermarks and apply a ranking test with a provable false-positive-rate bound. The design is (i) minimally invasive (no visible text changes), (ii) scalable to many users and documents via a large watermark space and multi-watermark attribution, and (iii) robust to common passive transformations. We evaluate on open-weight LLMs and multiple text domains, analyzing regurgitation dynamics, sensitivity to training set size, and interference under multiple concurrent watermarks. Our results demonstrate reliable post-hoc provenance signals with bounded FPR under black-box access. We experimentally observe a failure rate of less than 0.1\% when detecting a reply after fine-tuning with 50 marked documents. Conversely, no spurious reply was recovered in over 18,000 challenges, corresponding to a 100\%TPR@0\% FPR. Moreover, detection rates remain relatively stable as the dataset size increases, maintaining a per-document detection rate above 45\% even when the marked collection accounts for less than 0.33\% of the fine-tuning data. 

**Abstract (ZH)**: 针对黑盒访问下的敏感或版权文本审计问题：一种文本保留水印框架 

---
# Real-Time Health Analytics Using Ontology-Driven Complex Event Processing and LLM Reasoning: A Tuberculosis Case Study 

**Title (ZH)**: 基于本体驱动复杂事件处理和大型语言模型推理的实时健康分析：以 tuberculosis 为例的研究 

**Authors**: Ritesh Chandra, Sonali Agarwal, Navjot Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09646)  

**Abstract**: Timely detection of critical health conditions remains a major challenge in public health analytics, especially in Big Data environments characterized by high volume, rapid velocity, and diverse variety of clinical data. This study presents an ontology-enabled real-time analytics framework that integrates Complex Event Processing (CEP) and Large Language Models (LLMs) to enable intelligent health event detection and semantic reasoning over heterogeneous, high-velocity health data streams. The architecture leverages the Basic Formal Ontology (BFO) and Semantic Web Rule Language (SWRL) to model diagnostic rules and domain knowledge. Patient data is ingested and processed using Apache Kafka and Spark Streaming, where CEP engines detect clinically significant event patterns. LLMs support adaptive reasoning, event interpretation, and ontology refinement. Clinical information is semantically structured as Resource Description Framework (RDF) triples in Graph DB, enabling SPARQL-based querying and knowledge-driven decision support. The framework is evaluated using a dataset of 1,000 Tuberculosis (TB) patients as a use case, demonstrating low-latency event detection, scalable reasoning, and high model performance (in terms of precision, recall, and F1-score). These results validate the system's potential for generalizable, real-time health analytics in complex Big Data scenarios. 

**Abstract (ZH)**: 及时检测关键健康状况仍然是公共卫生分析中的主要挑战，特别是在以大数据量、高速度和多样数据特征为特点的环境中。本研究提出了一种基于本体的实时分析框架，该框架结合了复杂事件处理（CEP）和大规模语言模型（LLMs），以实现异质、高速健康数据流中的智能健康事件检测和语义推理。该架构利用基本形式本体（BFO）和语义网络规则语言（SWRL）来建模诊断规则和领域知识。患者数据通过Apache Kafka和Spark Streaming进行摄取和处理，其中CEP引擎检测临床显著事件模式。LLMs支持自适应推理、事件解释和本体优化。临床信息以资源描述框架（RDF）三元组形式结构化存储在图数据库中，支持SPARQL查询和基于知识的决策支持。该框架以1000名结核病（TB）患者的数据集为例进行评估，展示了低延迟事件检测、可扩展的推理能力和高模型性能（精度、召回率和F1分数）。这些结果验证了该系统在复杂大数据场景下实现通用化实时健康分析的潜力。 

---
# Bias-Aware AI Chatbot for Engineering Advising at the University of Maryland A. James Clark School of Engineering 

**Title (ZH)**: 面向马里兰大学JAMES A. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. JAMES I. 威廉姆斯工程学院的意识偏见AI聊天机器人 

**Authors**: Prarthana P. Kartholy, Thandi M. Labor, Neil N. Panchal, Sean H. Wang, Hillary N. Owusu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09636)  

**Abstract**: Selecting a college major is a difficult decision for many incoming freshmen. Traditional academic advising is often hindered by long wait times, intimidating environments, and limited personalization. AI Chatbots present an opportunity to address these challenges. However, AI systems also have the potential to generate biased responses, prejudices related to race, gender, socioeconomic status, and disability. These biases risk turning away potential students and undermining reliability of AI systems. This study aims to develop a University of Maryland (UMD) A. James Clark School of Engineering Program-specific AI chatbot. Our research team analyzed and mitigated potential biases in the responses. Through testing the chatbot on diverse student queries, the responses are scored on metrics of accuracy, relevance, personalization, and bias presence. The results demonstrate that with careful prompt engineering and bias mitigation strategies, AI chatbots can provide high-quality, unbiased academic advising support, achieving mean scores of 9.76 for accuracy, 9.56 for relevance, and 9.60 for personalization with no stereotypical biases found in the sample data. However, due to the small sample size and limited timeframe, our AI model may not fully reflect the nuances of student queries in engineering academic advising. Regardless, these findings will inform best practices for building ethical AI systems in higher education, offering tools to complement traditional advising and address the inequities faced by many underrepresented and first-generation college students. 

**Abstract (ZH)**: 选择大学专业是许多新生面临的艰难决定。传统的学术咨询往往受到长时间等待、令人 intimidated 的环境和有限的个性化程度的阻碍。AI聊天机器人提供了一个应对这些挑战的机会。然而，AI系统也可能生成有偏见的回应，涉及种族、性别、社会经济地位和残疾等方面的偏见。这些偏见可能会使潜在学生却步，并削弱AI系统的可靠性。本研究旨在为马里兰大学(UMD)杰斐逊·亚历山大·克拉克工程学院特定项目开发一个AI聊天机器人。我们的研究团队分析并缓解了潜在的偏见。通过在多样的学生查询上测试聊天机器人，对其响应进行了准确性、相关性、个性化和偏见存在的评分。结果显示，在精心的提示工程和偏见缓解策略下，AI聊天机器人可以提供高质量、无偏见的学术咨询支持，平均得分为：准确性9.76，相关性9.56，个性化9.60，样本数据中未发现刻板偏见。然而，由于样本量有限和时间限制，我们的AI模型可能无法完全反映工程学术咨询中学生的查询 nuances。无论怎样，这些发现将指导高等教育中建立伦理AI系统的最佳实践，提供工具以补充传统的咨询，并解决许多少数族裔和第一代大学生面临的不平等问题。 

---
# Mission Impossible: Feedback-Guided Dynamic Interactive Planning for Improving Reasoning on LLMs 

**Title (ZH)**: 不可能的任务：反馈引导的动态交互式规划以提高大语言模型的推理能力 

**Authors**: Dong Yan, Gaochen Wu, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.05577)  

**Abstract**: Recent advancements in language agents have led to significant improvements in multi-hop reasoning tasks. However, existing approaches often struggle with handling open-domain problems, which require massive information retrieval due to their reliance on a fixed sequence of actions. To address this, we propose Feedback-Guided Dynamic Interactive Planning (FGDIP), a novel framework tailored to enhance reasoning in LLMs by utilizing dynamic and adaptive strategies for information exploration in open-domain multi-hop reasoning tasks. Our approach begins by identifying key entities relevant to the problem, which serve as the initial nodes in the reasoning process. From these initial nodes, we then generate reasoning child nodes with the process being refined through a combination of historical error analysis and real-time feedback, which allows the framework to dynamically adjust and optimize its reasoning strategies. By integrating depth-first search with an innovative node generation technique, our framework adapts based on both prior error paths and concurrently generated nodes at the same hierarchical level. This dynamic strategy effectively expands the search space while ensuring the reasoning process systematically converges toward accurate solutions. Experimental results show that FGDIP achieved up to 54.47% F1 score on the HotpotQA dataset and 70.05% on the StrategyQA dataset, surpassing the best baseline by 5.03% and 7.25% respectively, highlighting its versatility and potential to enhance language agents in multi-hop reasoning tasks. 

**Abstract (ZH)**: Recent advancements in语言代理已在多跳推理任务中取得了显著进步。然而，现有的方法往往难以处理开放领域问题，因为这类问题需要大量的信息检索，依赖于固定顺序的操作。为了解决这一问题，我们提出了一种新的框架——反馈引导的动态交互规划（FGDIP），该框架旨在通过利用动态和适应性策略增强LLM在开放领域多跳推理任务中的推理能力。我们的方法首先识别与问题相关的关键实体，作为推理过程的初始节点。从这些初始节点出发，我们生成推理子节点，并通过历史错误分析和实时反馈的过程进行优化，使框架能够动态调整和优化其推理策略。通过将深度优先搜索与创新性的节点生成技术相结合，我们的框架能够根据先验错误路径和同时生成的同级节点进行适应。这种动态策略有效地扩展了搜索空间，同时确保推理过程系统地向准确的解决方案收敛。实验结果显示，FGDIP在HotpotQA数据集上的F1分数达到54.47%，在StrategyQA数据集上达到70.05%，分别比最佳基线高出5.03%和7.25%，突显了其在多跳推理任务中增强语言代理的灵活性和潜力。 

---
