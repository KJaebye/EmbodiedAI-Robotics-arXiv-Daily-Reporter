# LLM-based ambiguity detection in natural language instructions for collaborative surgical robots 

**Title (ZH)**: 基于LLM的自然语言指令中模糊性的检测方法用于协作手术机器人 

**Authors**: Ana Davila, Jacinto Colan, Yasuhisa Hasegawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.11525)  

**Abstract**: Ambiguity in natural language instructions poses significant risks in safety-critical human-robot interaction, particularly in domains such as surgery. To address this, we propose a framework that uses Large Language Models (LLMs) for ambiguity detection specifically designed for collaborative surgical scenarios. Our method employs an ensemble of LLM evaluators, each configured with distinct prompting techniques to identify linguistic, contextual, procedural, and critical ambiguities. A chain-of-thought evaluator is included to systematically analyze instruction structure for potential issues. Individual evaluator assessments are synthesized through conformal prediction, which yields non-conformity scores based on comparison to a labeled calibration dataset. Evaluating Llama 3.2 11B and Gemma 3 12B, we observed classification accuracy exceeding 60% in differentiating ambiguous from unambiguous surgical instructions. Our approach improves the safety and reliability of human-robot collaboration in surgery by offering a mechanism to identify potentially ambiguous instructions before robot action. 

**Abstract (ZH)**: 自然语言指令的歧义性在安全关键的人机交互中尤其是在手术领域中构成了重大风险。为此，我们提出了一种框架，利用大型语言模型（LLMs）进行专门针对协作手术场景的歧义检测。该方法采用由具有不同提示技术的LLM评估器组成的集成系统，以识别语言、上下文、程序和关键性歧义。包含一种推理链评估器，用于系统地分析指令结构以识别潜在问题。通过确立性预测综合各个评估器的评估结果，基于与标记校准数据集的比较，生成非协合得分。在对Llama 3.2 11B和Gemma 3 12B进行评估后，我们发现它们在区分手术指令的歧义性和非歧义性方面达到了超过60%的分类准确率。该方法通过在机器人行动前识别潜在的歧义指令，提高了手术中的人机协作的安全性和可靠性。 

---
# How Many Instructions Can LLMs Follow at Once? 

**Title (ZH)**: 大规模语言模型一次能遵循多少指令？ 

**Authors**: Daniel Jaroslawicz, Brendan Whiting, Parth Shah, Karime Maamari  

**Link**: [PDF](https://arxiv.org/pdf/2507.11538)  

**Abstract**: Production-grade LLM systems require robust adherence to dozens or even hundreds of instructions simultaneously. However, the instruction-following capabilities of LLMs at high instruction densities have not yet been characterized, as existing benchmarks only evaluate models on tasks with a single or few instructions. We introduce IFScale, a simple benchmark of 500 keyword-inclusion instructions for a business report writing task to measure how instruction-following performance degrades as instruction density increases. We evaluate 20 state-of-the-art models across seven major providers and find that even the best frontier models only achieve 68% accuracy at the max density of 500 instructions. Our analysis reveals model size and reasoning capability to correlate with 3 distinct performance degradation patterns, bias towards earlier instructions, and distinct categories of instruction-following errors. Our insights can help inform design of instruction-dense prompts in real-world applications and highlight important performance-latency tradeoffs. We open-source the benchmark and all results for further analysis at this https URL. 

**Abstract (ZH)**: 生产级LLM系统需要严格遵守成百上千条指令。然而，现有基准仅评估模型在单一或少量指令任务上的表现，尚未描述大模型在高指令密度下的指令遵循能力。为此，我们提出了IFScale基准测试，包含500条关键词包含指令，以评估指令密度增加时的指令遵循性能下降情况。我们对来自七家主要提供商的20个领先模型进行了评估，发现即使最好的前沿模型在最大密度500条指令下的准确率也只有68%。我们的分析揭示了模型大小和推理能力与三种不同的性能下降模式、偏向早期指令的趋势及不同类别指令遵循错误之间的关联。我们的洞见有助于指导实际应用中复杂指令提示的设计，并突出性能与延迟之间的关键权衡。我们开源了基准测试及所有结果以供进一步分析。 

---
# DrafterBench: Benchmarking Large Language Models for Tasks Automation in Civil Engineering 

**Title (ZH)**: DrafterBench: 大型语言模型在土木工程任务自动化中的基准测试 

**Authors**: Yinsheng Li, Zhen Dong, Yi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11527)  

**Abstract**: Large Language Model (LLM) agents have shown great potential for solving real-world problems and promise to be a solution for tasks automation in industry. However, more benchmarks are needed to systematically evaluate automation agents from an industrial perspective, for example, in Civil Engineering. Therefore, we propose DrafterBench for the comprehensive evaluation of LLM agents in the context of technical drawing revision, a representation task in civil engineering. DrafterBench contains twelve types of tasks summarized from real-world drawing files, with 46 customized functions/tools and 1920 tasks in total. DrafterBench is an open-source benchmark to rigorously test AI agents' proficiency in interpreting intricate and long-context instructions, leveraging prior knowledge, and adapting to dynamic instruction quality via implicit policy awareness. The toolkit comprehensively assesses distinct capabilities in structured data comprehension, function execution, instruction following, and critical reasoning. DrafterBench offers detailed analysis of task accuracy and error statistics, aiming to provide deeper insight into agent capabilities and identify improvement targets for integrating LLMs in engineering applications. Our benchmark is available at this https URL, with the test set hosted at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLM）代理在技术和制图修订任务中的综合评估：DrafterBench 

---
# Modeling Code: Is Text All You Need? 

**Title (ZH)**: 代码建模：文本足够吗？ 

**Authors**: Daniel Nichols, Konstantinos Parasyris, Harshitha Menon, Brian R. Bartoldson, Giorgis Georgakoudis, Tal Ben-Nun, Abhinav Bhatele  

**Link**: [PDF](https://arxiv.org/pdf/2507.11467)  

**Abstract**: Code LLMs have become extremely popular recently for modeling source code across a variety of tasks, such as generation, translation, and summarization. However, transformer-based models are limited in their capabilities to reason through structured, analytical properties of code, such as control and data flow. Previous work has explored the modeling of these properties with structured data and graph neural networks. However, these approaches lack the generative capabilities and scale of modern LLMs. In this work, we introduce a novel approach to combine the strengths of modeling both code as text and more structured forms. 

**Abstract (ZH)**: 代码LLM在多种任务中的应用近年来十分流行，涵盖了生成、翻译和摘要等任务。然而，基于变换器的模型在推理代码的结构化和分析性质（如控制流和数据流）方面能力有限。先前的研究探索了使用结构化数据和图神经网络来建模这些性质，但这些方法缺乏现代LLM的生成能力和规模。在本文中，我们提出了一种新颖的方法，结合了将代码建模为文本和更结构化形式的优势。 

---
# Foundation Models for Logistics: Toward Certifiable, Conversational Planning Interfaces 

**Title (ZH)**: 物流领域的基础模型：迈向可验证、对话式规划接口 

**Authors**: Yunhao Yang, Neel P. Bhatt, Christian Ellis, Alvaro Velasquez, Zhangyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11352)  

**Abstract**: Logistics operators, from battlefield coordinators rerouting airlifts ahead of a storm to warehouse managers juggling late trucks, often face life-critical decisions that demand both domain expertise and rapid and continuous replanning. While popular methods like integer programming yield logistics plans that satisfy user-defined logical constraints, they are slow and assume an idealized mathematical model of the environment that does not account for uncertainty. On the other hand, large language models (LLMs) can handle uncertainty and promise to accelerate replanning while lowering the barrier to entry by translating free-form utterances into executable plans, yet they remain prone to misinterpretations and hallucinations that jeopardize safety and cost. We introduce a neurosymbolic framework that pairs the accessibility of natural-language dialogue with verifiable guarantees on goal interpretation. It converts user requests into structured planning specifications, quantifies its own uncertainty at the field and token level, and invokes an interactive clarification loop whenever confidence falls below an adaptive threshold. A lightweight model, fine-tuned on just 100 uncertainty-filtered examples, surpasses the zero-shot performance of GPT-4.1 while cutting inference latency by nearly 50%. These preliminary results highlight a practical path toward certifiable, real-time, and user-aligned decision-making for complex logistics. 

**Abstract (ZH)**: 神经符号框架在复杂物流中的可认证实时和用户对齐决策 

---
# Opus: A Prompt Intention Framework for Complex Workflow Generation 

**Title (ZH)**: Opus：复杂工作流生成的提示意图框架 

**Authors**: Théo Fagnoni, Mahsun Altin, Chia En Chung, Phillip Kingston, Alan Tuning, Dana O. Mohamed, Inès Adnani  

**Link**: [PDF](https://arxiv.org/pdf/2507.11288)  

**Abstract**: This paper introduces the Opus Prompt Intention Framework, designed to improve complex Workflow Generation with instruction-tuned Large Language Models (LLMs). We propose an intermediate Intention Capture layer between user queries and Workflow Generation, implementing the Opus Workflow Intention Framework, which consists of extracting Workflow Signals from user queries, interpreting them into structured Workflow Intention objects, and generating Workflows based on these Intentions. Our results show that this layer enables LLMs to produce logical and meaningful outputs that scale reliably as query complexity increases. On a synthetic benchmark of 1,000 multi-intent query-Workflow(s) pairs, applying the Opus Prompt Intention Framework to Workflow Generation yields consistent improvements in semantic Workflow similarity metrics. In this paper, we introduce the Opus Prompt Intention Framework by applying the concepts of Workflow Signal and Workflow Intention to LLM-driven Workflow Generation. We present a reproducible, customizable LLM-based Intention Capture system to extract Workflow Signals and Workflow Intentions from user queries. Finally, we provide empirical evidence that the proposed system significantly improves Workflow Generation quality compared to direct generation from user queries, particularly in cases of Mixed Intention Elicitation. 

**Abstract (ZH)**: Opus Prompt Intention Framework: 一种用于改进指令调优大规模语言模型驱动的工作流生成的框架 

---
# Taming Uncertainty via Automation: Observing, Analyzing, and Optimizing Agentic AI Systems 

**Title (ZH)**: 通过自动化管理不确定性：观察、分析与优化代理人工智能系统 

**Authors**: Dany Moshkovich, Sergey Zeltyn  

**Link**: [PDF](https://arxiv.org/pdf/2507.11277)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed within agentic systems-collections of interacting, LLM-powered agents that execute complex, adaptive workflows using memory, tools, and dynamic planning. While enabling powerful new capabilities, these systems also introduce unique forms of uncertainty stemming from probabilistic reasoning, evolving memory states, and fluid execution paths. Traditional software observability and operations practices fall short in addressing these challenges.
This paper introduces AgentOps: a comprehensive framework for observing, analyzing, optimizing, and automating operation of agentic AI systems. We identify distinct needs across four key roles-developers, testers, site reliability engineers (SREs), and business users-each of whom engages with the system at different points in its lifecycle. We present the AgentOps Automation Pipeline, a six-stage process encompassing behavior observation, metric collection, issue detection, root cause analysis, optimized recommendations, and runtime automation. Throughout, we emphasize the critical role of automation in managing uncertainty and enabling self-improving AI systems-not by eliminating uncertainty, but by taming it to ensure safe, adaptive, and effective operation. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益被部署在智能代理系统中，这些系统是由配备语言模型的强大代理组成的集合，通过记忆、工具和动态规划执行复杂的适应性工作流。虽然这些系统提供了强大的新能力，但也引入了概率推理、不断演化的记忆状态和动态执行路径带来的独特不确定性形式。传统的软件可观测性和运营实践在解决这些挑战时力不从心。

本文介绍了智能代理运营（AgentOps）：一个全面的框架，用于观察、分析、优化和自动化智能代理AI系统的运营。我们确定了四个关键角色——开发者、测试员、站点可靠性工程师（SRE）和业务用户在系统生命周期不同阶段对系统的参与需求。我们提出了智能代理运营自动化流水线，一个六阶段过程，涵盖行为观察、指标收集、问题检测、根本原因分析、优化建议和运行时自动化。在整个过程中，我们强调自动化在管理不确定性并实现自我改进的AI系统中的关键作用——不是通过消除不确定性，而是通过驾驭不确定性以确保安全、适应性和有效的运行。 

---
# Function-to-Style Guidance of LLMs for Code Translation 

**Title (ZH)**: LLM代码翻译的功能与风格指导 

**Authors**: Longhui Zhang, Bin Wang, Jiahao Wang, Xiaofeng Zhao, Min Zhang, Hao Yang, Meishan Zhang, Yu Li, Jing Li, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11083)  

**Abstract**: Large language models (LLMs) have made significant strides in code translation tasks. However, ensuring both the correctness and readability of translated code remains a challenge, limiting their effective adoption in real-world software development. In this work, we propose F2STrans, a function-to-style guiding paradigm designed to progressively improve the performance of LLMs in code translation. Our approach comprises two key stages: (1) Functional learning, which optimizes translation correctness using high-quality source-target code pairs mined from online programming platforms, and (2) Style learning, which improves translation readability by incorporating both positive and negative style examples. Additionally, we introduce a novel code translation benchmark that includes up-to-date source code, extensive test cases, and manually annotated ground-truth translations, enabling comprehensive functional and stylistic evaluations. Experiments on both our new benchmark and existing datasets demonstrate that our approach significantly improves code translation performance. Notably, our approach enables Qwen-1.5B to outperform prompt-enhanced Qwen-32B and GPT-4 on average across 20 diverse code translation scenarios. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在代码翻译任务中取得了显著进展。然而，确保翻译代码的正确性和可读性仍然是一个挑战，限制了其在实际软件开发中的有效应用。在本工作中，我们提出了一种基于功能导向的代码翻译范式F2STrans，旨在逐步提升LLMs在代码翻译中的性能。我们的方法包括两个关键阶段：（1）功能学习，通过从在线编程平台中挖掘高质量的源-目标代码对来优化翻译正确性；（2）风格学习，通过结合正反面的风格示例来提升翻译可读性。此外，我们引入了一个新的代码翻译基准，其中包括最新的源代码、广泛的测试案例以及人工标注的真实翻译，从而能够进行全面的功能和风格评估。在我们新的基准和现有数据集上的实验表明，我们的方法显著提升了代码翻译性能。值得注意的是，我们的方法使得Qwen-1.5B在20个不同的代码翻译场景中平均性能优于增强提示的Qwen-32B和GPT-4。 

---
# Lessons Learned from Evaluation of LLM based Multi-agents in Safer Therapy Recommendation 

**Title (ZH)**: 基于LLM的多代理在安全疗法推荐评估中获得的教训 

**Authors**: Yicong Wu, Ting Chen, Irit Hochberg, Zhoujian Sun, Ruth Edry, Zhengxing Huang, Mor Peleg  

**Link**: [PDF](https://arxiv.org/pdf/2507.10911)  

**Abstract**: Therapy recommendation for chronic patients with multimorbidity is challenging due to risks of treatment conflicts. Existing decision support systems face scalability limitations. Inspired by the way in which general practitioners (GP) manage multimorbidity patients, occasionally convening multidisciplinary team (MDT) collaboration, this study investigated the feasibility and value of using a Large Language Model (LLM)-based multi-agent system (MAS) for safer therapy recommendations. We designed a single agent and a MAS framework simulating MDT decision-making by enabling discussion among LLM agents to resolve medical conflicts. The systems were evaluated on therapy planning tasks for multimorbidity patients using benchmark cases. We compared MAS performance with single-agent approaches and real-world benchmarks. An important contribution of our study is the definition of evaluation metrics that go beyond the technical precision and recall and allow the inspection of clinical goals met and medication burden of the proposed advices to a gold standard benchmark. Our results show that with current LLMs, a single agent GP performs as well as MDTs. The best-scoring models provide correct recommendations that address all clinical goals, yet the advices are incomplete. Some models also present unnecessary medications, resulting in unnecessary conflicts between medication and conditions or drug-drug interactions. 

**Abstract (ZH)**: 基于大规模语言模型的多Agent系统在慢性多病患者治疗推荐中的可行性和价值研究 

---
# Automated Thematic Analyses Using LLMs: Xylazine Wound Management Social Media Chatter Use Case 

**Title (ZH)**: 使用大语言模型自动主题分析：Xylazine伤口管理社交媒体讨论案例研究 

**Authors**: JaMor Hairston, Ritvik Ranjan, Sahithi Lakamana, Anthony Spadaro, Selen Bozkurt, Jeanmarie Perrone, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2507.10803)  

**Abstract**: Background Large language models (LLMs) face challenges in inductive thematic analysis, a task requiring deep interpretive and domain-specific expertise. We evaluated the feasibility of using LLMs to replicate expert-driven thematic analysis of social media data. Methods Using two temporally non-intersecting Reddit datasets on xylazine (n=286 and n=686, for model optimization and validation, respectively) with twelve expert-derived themes, we evaluated five LLMs against expert coding. We modeled the task as a series of binary classifications, rather than a single, multi-label classification, employing zero-, single-, and few-shot prompting strategies and measuring performance via accuracy, precision, recall, and F1-score. Results On the validation set, GPT-4o with two-shot prompting performed best (accuracy: 90.9%; F1-score: 0.71). For high-prevalence themes, model-derived thematic distributions closely mirrored expert classifications (e.g., xylazine use: 13.6% vs. 17.8%; MOUD use: 16.5% vs. 17.8%). Conclusions Our findings suggest that few-shot LLM-based approaches can automate thematic analyses, offering a scalable supplement for qualitative research. Keywords: thematic analysis, large language models, natural language processing, qualitative analysis, social media, prompt engineering, public health 

**Abstract (ZH)**: 背景：大规模语言模型在归纳主题分析任务中面临挑战，该任务需要深入的解释性和领域特定的专业知识。我们评估了使用大规模语言模型复制专家驱动的社交媒体数据主题分析可行性的方法。方法：使用两个时间上不重叠的关于xylazine的Reddit数据集（分别为n=286和n=686，用于模型优化和验证），包含十二个专家提取的主题，我们对五个大规模语言模型进行了评估，以与专家编码进行比较。我们将任务模型化为一系列二分类任务，而不是一个单一的多标签分类任务，采用了零样本、单样本和少样本提示策略，并通过准确率、精确率、召回率和F1分数来衡量性能。结果：在验证集上，使用两样本提示的GPT-4o性能最佳（准确率：90.9%；F1分数：0.71）。对于高频率主题，模型提取的主题分布与专家分类高度一致（例如，xylazine使用情况：13.6% vs. 17.8%；莫顿类使用情况：16.5% vs. 17.8%）。结论：我们的研究结果表明，少样本的大规模语言模型方法可以自动化主题分析，为定量研究提供可扩展的补充。关键词：主题分析，大规模语言模型，自然语言处理，定量分析，社交媒体，提示工程，公共卫生 

---
# Enhancing the Capabilities of Large Language Models for API calls through Knowledge Graphs 

**Title (ZH)**: 通过知识图谱增强大型语言模型的API调用能力 

**Authors**: Ye Yang, Xue Xiao, Ping Yin, Taotao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.10630)  

**Abstract**: API calls by large language models (LLMs) offer a cutting-edge approach for data analysis. However, their ability to effectively utilize tools via API calls remains underexplored in knowledge-intensive domains like meteorology. This paper introduces KG2data, a system that integrates knowledge graphs, LLMs, ReAct agents, and tool-use technologies to enable intelligent data acquisition and query handling in the meteorological field. Using a virtual API, we evaluate API call accuracy across three metrics: name recognition failure, hallucination failure, and call correctness. KG2data achieves superior performance (1.43%, 0%, 88.57%) compared to RAG2data (16%, 10%, 72.14%) and chat2data (7.14%, 8.57%, 71.43%). KG2data differs from typical LLM-based systems by addressing their limited access to domain-specific knowledge, which hampers performance on complex or terminology-rich queries. By using a knowledge graph as persistent memory, our system enhances content retrieval, complex query handling, domain-specific reasoning, semantic relationship resolution, and heterogeneous data integration. It also mitigates the high cost of fine-tuning LLMs, making the system more adaptable to evolving domain knowledge and API structures. In summary, KG2data provides a novel solution for intelligent, knowledge-based question answering and data analysis in domains with high knowledge demands. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的API调用为数据解析提供了前沿的方法，但在气象等知识密集型领域，它们通过API调用有效利用工具的能力仍待探索。本文介绍了一种KG2data系统，该系统结合了知识图谱、LLMs、ReAct代理和工具使用技术，以在气象领域实现智能数据采集和查询处理。通过虚拟API，我们根据命名识别失败、幻觉失败和调用正确性三个指标评估API调用的准确性。KG2data在三个指标上的性能分别为1.43%、0%、88.57%，显著优于RAG2data（16%，10%，72.14%）和chat2data（7.14%，8.57%，71.43%）。与典型的基于LLM的系统相比，KG2data通过解决LLMs对领域特定知识访问有限的问题，提高了处理复杂或术语丰富的查询的能力。利用知识图谱作为持久化内存，我们的系统增强了内容检索、复杂查询处理、领域特定推理、语义关系解析和异构数据集成，并降低了对LLM微调的高成本，使系统更具适应性，能够更好地应对领域知识和API结构的演变。总之，KG2data为高知识需求领域提供了智能、基于知识的问题回答和数据分析的创新解决方案。 

---
# Comprehension Without Competence: Architectural Limits of LLMs in Symbolic Computation and Reasoning 

**Title (ZH)**: 理解而无能力：大语言模型在符号计算与推理方面的架构限制 

**Authors**: Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10624)  

**Abstract**: Large Language Models (LLMs) display striking surface fluency yet systematically fail at tasks requiring symbolic reasoning, arithmetic accuracy, and logical consistency. This paper offers a structural diagnosis of such failures, revealing a persistent gap between \textit{comprehension} and \textit{competence}. Through controlled experiments and architectural analysis, we demonstrate that LLMs often articulate correct principles without reliably applying them--a failure rooted not in knowledge access, but in computational execution. We term this phenomenon the computational \textit{split-brain syndrome}, where instruction and action pathways are geometrically and functionally dissociated. This core limitation recurs across domains, from mathematical operations to relational inferences, and explains why model behavior remains brittle even under idealized prompting. We argue that LLMs function as powerful pattern completion engines, but lack the architectural scaffolding for principled, compositional reasoning. Our findings delineate the boundary of current LLM capabilities and motivate future models with metacognitive control, principle lifting, and structurally grounded execution. This diagnosis also clarifies why mechanistic interpretability findings may reflect training-specific pattern coordination rather than universal computational principles, and why the geometric separation between instruction and execution pathways suggests limitations in neural introspection and mechanistic analysis. 

**Abstract (ZH)**: 大型语言模型（LLMs）表现出色的表面流畅性，但在需要符号推理、算术准确性和逻辑一致性的任务上系统性地失败。本文提供了一种结构性诊断，揭示了理解与能力之间持续存在的差距。通过受控实验和架构分析，我们证明LLMs往往能够表述正确的原则，但无法可靠地应用这些原则——这种失败并非源于知识访问，而是计算执行的不足。我们称之为计算分裂脑综合症，其中指令路径和行动路径在空间上和功能上是分离的。这一核心限制在各个领域都存在，从数学运算到关系推理，并解释了即使在理想化的提示下，模型行为仍然脆弱的原因。我们认为LLMs充当强大的模式补全引擎，但缺乏用于原理化、组合推理的架构支撑。我们的发现界定了当前LLM能力的边界，并促使未来的模型具备元认知控制、原理提升和结构支撑的执行能力。此外，这一诊断也阐明了机制可解释性研究可能反映出训练特定的模式协调，而非普适的计算原理，并解释了指令路径和执行路径的几何分离如何表明神经内省和机制分析的局限性。 

---
# AirLLM: Diffusion Policy-based Adaptive LoRA for Remote Fine-Tuning of LLM over the Air 

**Title (ZH)**: AirLLM：基于扩散策略的适配LoRA远端微调空中语言模型 

**Authors**: Shiyi Yang, Xiaoxue Yu, Rongpeng Li, Jianhang Zhu, Zhifeng Zhao, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.11515)  

**Abstract**: Operating Large Language Models (LLMs) on edge devices is increasingly challenged by limited communication bandwidth and strained computational and memory costs. Thus, cloud-assisted remote fine-tuning becomes indispensable. Nevertheless, existing Low-Rank Adaptation (LoRA) approaches typically employ fixed or heuristic rank configurations, and the subsequent over-the-air transmission of all LoRA parameters could be rather inefficient. To address this limitation, we develop AirLLM, a hierarchical diffusion policy framework for communication-aware LoRA adaptation. Specifically, AirLLM models the rank configuration as a structured action vector that spans all LoRA-inserted projections. To solve the underlying high-dimensional sequential decision-making problem, a Proximal Policy Optimization (PPO) agent generates coarse-grained decisions by jointly observing wireless states and linguistic complexity, which are then refined via Denoising Diffusion Implicit Models (DDIM) to produce high-resolution, task- and channel-adaptive rank vectors. The two modules are optimized alternatively, with the DDIM trained under the Classifier-Free Guidance (CFG) paradigm to maintain alignment with PPO rewards. Experiments under varying signal-to-noise ratios demonstrate that AirLLM consistently enhances fine-tuning performance while significantly reducing transmission costs, highlighting the effectiveness of reinforcement-driven, diffusion-refined rank adaptation for scalable and efficient remote fine-tuning over the air. 

**Abstract (ZH)**: 基于通信感知的层级扩散洛拉适应框架（AirLLM）：强化学习驱动的远程细调优化 

---
# KisMATH: Do LLMs Have Knowledge of Implicit Structures in Mathematical Reasoning? 

**Title (ZH)**: KisMATH：LLM们是否具备数学推理中隐含结构的知识？ 

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Saptarshi Saha, Utpal Garain, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2507.11408)  

**Abstract**: Chain-of-thought traces have been shown to improve performance of large language models in a plethora of reasoning tasks, yet there is no consensus on the mechanism through which this performance boost is achieved. To shed more light on this, we introduce Causal CoT Graphs (CCGs), which are directed acyclic graphs automatically extracted from reasoning traces that model fine-grained causal dependencies in the language model output. A collection of $1671$ mathematical reasoning problems from MATH500, GSM8K and AIME, and their associated CCGs are compiled into our dataset -- \textbf{KisMATH}. Our detailed empirical analysis with 15 open-weight LLMs shows that (i) reasoning nodes in the CCG are mediators for the final answer, a condition necessary for reasoning; and (ii) LLMs emphasise reasoning paths given by the CCG, indicating that models internally realise structures akin to our graphs. KisMATH enables controlled, graph-aligned interventions and opens up avenues for further investigation into the role of chain-of-thought in LLM reasoning. 

**Abstract (ZH)**: Chain-of-Thought Traces in Large Language Models: An Analysis via Causal CoT Graphs 

---
# EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes 

**Title (ZH)**: EXAONE 4.0: 统一的大语言模型整合非推理与推理模式 

**Authors**: LG AI Research, Kyunghoon Bae, Eunbi Choi, Kibong Choi, Stanley Jungkyu Choi, Yemuk Choi, Kyubeen Han, Seokhee Hong, Junwon Hwang, Taewan Hwang, Joonwon Jang, Hyojin Jeon, Kijeong Jeon, Gerrard Jeongwon Jo, Hyunjik Jo, Jiyeon Jung, Euisoon Kim, Hyosang Kim, Jihoon Kim, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Youchul Kim, Edward Hwayoung Lee, Gwangho Lee, Haeju Lee, Honglak Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Young Min Paik, Yongmin Park, Youngyong Park, Sanghyun Seo, Sihoon Yang, Heuiyeen Yeen, Sihyuk Yi, Hyeongu Yun  

**Link**: [PDF](https://arxiv.org/pdf/2507.11407)  

**Abstract**: This technical report introduces EXAONE 4.0, which integrates a Non-reasoning mode and a Reasoning mode to achieve both the excellent usability of EXAONE 3.5 and the advanced reasoning abilities of EXAONE Deep. To pave the way for the agentic AI era, EXAONE 4.0 incorporates essential features such as agentic tool use, and its multilingual capabilities are extended to support Spanish in addition to English and Korean. The EXAONE 4.0 model series consists of two sizes: a mid-size 32B model optimized for high performance, and a small-size 1.2B model designed for on-device applications. The EXAONE 4.0 demonstrates superior performance compared to open-weight models in its class and remains competitive even against frontier-class models. The models are publicly available for research purposes and can be easily downloaded via this https URL. 

**Abstract (ZH)**: 本技术报告介绍了EXAONE 4.0，该版本整合了非推理模式和推理模式，既保持了EXAONE 3.5的优秀易用性，又具备了EXAONE Deep的高级推理能力。为了为有意识的AI时代铺平道路，EXAONE 4.0增加了有意识工具使用等关键功能，其多语言能力扩展支持西班牙语、英语和韩语。EXAONE 4.0模型系列包括两种尺寸：一个中型32B模型，优化高性能使用，以及一个小型1.2B模型，专门为设备端应用设计。与其他同类开放权重模型相比，EXAONE 4.0表现出色，并且即使与最先进的模型相比也能保持竞争力。这些模型可供研究使用，并可通过此链接直接下载。 

---
# Automated Novelty Evaluation of Academic Paper: A Collaborative Approach Integrating Human and Large Language Model Knowledge 

**Title (ZH)**: 学术论文新颖性评价的协作方法：融合人类与大规模语言模型知识 

**Authors**: Wenqing Wu, Chengzhi Zhang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11330)  

**Abstract**: Novelty is a crucial criterion in the peer review process for evaluating academic papers. Traditionally, it's judged by experts or measure by unique reference combinations. Both methods have limitations: experts have limited knowledge, and the effectiveness of the combination method is uncertain. Moreover, it's unclear if unique citations truly measure novelty. The large language model (LLM) possesses a wealth of knowledge, while human experts possess judgment abilities that the LLM does not possess. Therefore, our research integrates the knowledge and abilities of LLM and human experts to address the limitations of novelty assessment. The most common novelty in academic papers is the introduction of new methods. In this paper, we propose leveraging human knowledge and LLM to assist pretrained language models (PLMs, e.g. BERT etc.) in predicting the method novelty of papers. Specifically, we extract sentences related to the novelty of the academic paper from peer review reports and use LLM to summarize the methodology section of the academic paper, which are then used to fine-tune PLMs. In addition, we have designed a text-guided fusion module with novel Sparse-Attention to better integrate human and LLM knowledge. We compared the method we proposed with a large number of baselines. Extensive experiments demonstrate that our method achieves superior performance. 

**Abstract (ZH)**: 新颖性是学术论文同行评审过程中一个关键的评价标准。传统上，新颖性由专家判断或通过独特的参考组合进行度量。这两种方法都有局限性：专家的知识有限，组合方法的有效性也不确定。此外，独特的引用是否能真正度量新颖性也存疑。大规模语言模型（LLM）拥有丰富的知识，而人类专家则具有LLM不具备的判断能力。因此，我们的研究将LLM的知识和能力与人类专家的知识相结合，以克服新颖性评估的局限性。学术论文中最常见的新颖性在于引入新的方法。本文提出利用人类知识和LLM辅助预训练语言模型（PLMs，如BERT等）预测论文方法的新颖性。具体而言，我们从同行评审报告中提取与论文新颖性相关的句子，并使用LLM总结学术论文的方法部分，然后用于微调PLMs。此外，我们还设计了一个文本引导的融合模块，采用新颖的稀疏注意机制，更好地整合人类和LLM的知识。我们将提出的方法与多种基线进行了比较。广泛的经验表明，我们的方法取得了更好的性能。 

---
# Internal Value Alignment in Large Language Models through Controlled Value Vector Activation 

**Title (ZH)**: 通过控制价值向量激活实现大型语言模型内部价值对齐 

**Authors**: Haoran Jin, Meng Li, Xiting Wang, Zhihao Xu, Minlie Huang, Yantao Jia, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2507.11316)  

**Abstract**: Aligning Large Language Models (LLMs) with human values has attracted increasing attention since it provides clarity, transparency, and the ability to adapt to evolving scenarios. In this paper, we introduce a Controlled Value Vector Activation (ConVA) method that directly aligns the internal values of LLMs by interpreting how a value is encoded in their latent representations and modifies relevant activations to ensure consistent values in LLMs. To ensure an accurate and unbiased interpretation, we propose a context-controlled value vector identification method. To consistently control values without sacrificing model performance, we introduce a gated value vector activation method for effective and minimum degree of value control. Experiments show that our method achieves the highest control success rate across 10 basic values without hurting LLM performance and fluency, and ensures target values even with opposite and potentially malicious input prompts. Source code and data are available at~ this https URL. 

**Abstract (ZH)**: 控制值向量激活（ConVA）方法：通过直接调整大型语言模型的内部价值实现与人类价值观的对齐 

---
# An Agentic Flow for Finite State Machine Extraction using Prompt Chaining 

**Title (ZH)**: 基于提示链的有限状态机提取代理流程 

**Authors**: Fares Wael, Youssef Maklad, Ali Hamdi, Wael Elsersy  

**Link**: [PDF](https://arxiv.org/pdf/2507.11222)  

**Abstract**: Finite-State Machines (FSMs) are critical for modeling the operational logic of network protocols, enabling verification, analysis, and vulnerability discovery. However, existing FSM extraction techniques face limitations such as scalability, incomplete coverage, and ambiguity in natural language specifications. In this paper, we propose FlowFSM, a novel agentic framework that leverages Large Language Models (LLMs) combined with prompt chaining and chain-of-thought reasoning to extract accurate FSMs from raw RFC documents. FlowFSM systematically processes protocol specifications, identifies state transitions, and constructs structured rule-books by chaining agent outputs. Experimental evaluation across FTP and RTSP protocols demonstrates that FlowFSM achieves high extraction precision while minimizing hallucinated transitions, showing promising results. Our findings highlight the potential of agent-based LLM systems in the advancement of protocol analysis and FSM inference for cybersecurity and reverse engineering applications. 

**Abstract (ZH)**: 基于大型语言模型的agentic框架FlowFSM：从原始RFC文档中精准提取有限状态机 

---
# Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias 

**Title (ZH)**: 基于角色扮演的LLM驱动多-agent支持框架：检测与解决家庭沟通偏见 

**Authors**: Rushia Harada, Yuken Kimura, Keito Inoshita  

**Link**: [PDF](https://arxiv.org/pdf/2507.11210)  

**Abstract**: Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions. 

**Abstract (ZH)**: 家庭环境中的心灵福祉涉及常规度量常常忽略的微妙心理动态。特别是，被称为理想家长偏见的无意识父母期望会抑制儿童的情感表达和自主性。这种抑制通常源于旨在良好的但具有价值导向的沟通，从家庭外部难以察觉或解决。聚焦于这些潜在动态，本研究探索基于大规模语言模型（LLM）的心理安全家庭沟通支持。我们构建了一个包含30个场景的日语亲子对话语料库，每个场景都标注了理想家长偏见和抑制情感的元数据。基于此语料库，我们开发了一种基于角色扮演的大规模语言模型多代理对话支持框架，该框架分析对话并生成反馈。专业代理检测抑制情感，描述父母话语中的隐含理想家长偏见，并推断出如儿童的年龄和背景等上下文属性。元代理将这些输出汇总为结构化的报告，然后传递给五个选定的专家代理。这些代理通过结构化的四步讨论过程协作生成同理心和可操作的反馈。实验显示，该系统可以在中等准确度下检测抑制情感的类别，并生成具有高同理心和实际性的反馈。此外，结合此反馈的模拟后续对话显示出情绪表达和相互理解的改进迹象，表明该框架在支持家庭互动的积极转变方面具有潜力。 

---
# Temperature and Persona Shape LLM Agent Consensus With Minimal Accuracy Gains in Qualitative Coding 

**Title (ZH)**: 温度和人格塑造LLM代理共识， minimal 准确度提升在定性编码中。 

**Authors**: Conrad Borchers, Bahar Shahrokhian, Francesco Balzan, Elham Tajik, Sreecharan Sankaranarayanan, Sebastian Simon  

**Link**: [PDF](https://arxiv.org/pdf/2507.11198)  

**Abstract**: Large Language Models (LLMs) enable new possibilities for qualitative research at scale, including coding and data annotation. While multi-agent systems (MAS) can emulate human coding workflows, their benefits over single-agent coding remain poorly understood. We conducted an experimental study of how agent persona and temperature shape consensus-building and coding accuracy of dialog segments based on a codebook with 8 codes. Our open-source MAS mirrors deductive human coding through structured agent discussion and consensus arbitration. Using six open-source LLMs (with 3 to 32 billion parameters) and 18 experimental configurations, we analyze over 77,000 coding decisions against a gold-standard dataset of human-annotated transcripts from online math tutoring sessions. Temperature significantly impacted whether and when consensus was reached across all six LLMs. MAS with multiple personas (including neutral, assertive, or empathetic), significantly delayed consensus in four out of six LLMs compared to uniform personas. In three of those LLMs, higher temperatures significantly diminished the effects of multiple personas on consensus. However, neither temperature nor persona pairing lead to robust improvements in coding accuracy. Single agents matched or outperformed MAS consensus in most conditions. Only one model (OpenHermesV2:7B) and code category showed above-chance gains from MAS deliberation when temperature was 0.5 or lower and especially when the agents included at least one assertive persona. Qualitative analysis of MAS collaboration for these configurations suggests that MAS may nonetheless aid in narrowing ambiguous code applications that could improve codebooks and human-AI coding. We contribute new insight into the limits of LLM-based qualitative methods, challenging the notion that diverse MAS personas lead to better outcomes. We open-source our MAS and experimentation code. 

**Abstract (ZH)**: 大型语言模型（LLMs）为大规模定性研究提供了新的可能性，包括编码和数据注释。尽管多智能体系统（MAS）可以模拟人类编码工作流，但它们在多智能体编码方面的优势仍然 poorly understood（缺乏充分理解）。我们通过一个包含8个代码的代码本，研究了代理角色和温度如何影响基于对话片段的共识构建和编码准确性。我们的开源MAS通过对结构化代理讨论和共识仲裁来模拟演绎性的人类编码。使用六种开源LLM（参数量在3亿到32亿之间）和18种实验配置，我们对超过77,000个编码决策进行了分析，这些数据来源于在线数学辅导会话的人类标注转录数据集。温度显著影响了所有六种LLM中是否以及何时达成共识。包含多个角色（包括中立、自信或 Empathetic）的MAS，在四种情况下显著延迟了在六种LLM中的共识，相比统一的角色。在三种LLM中，较高的温度显著削弱了多个角色对共识的影响。然而，温度和角色配对均未导致编码准确性的稳定提升。单个代理在大多数情况下与MAS的共识相匹配甚至表现更好。仅有一个模型（OpenHermesV2:7B）和代码类别，在温度为0.5或更低时，以及尤其是在代理中至少包含一个自信角色时，从MAS讨论中获得了超出随机猜测的收益。对于这些配置下的MAS合作定性分析表明，MAS可能有助于缩小模糊代码应用的范围，从而改善代码本和人类-AI编码。我们为LLM基础的定性方法提供了新的见解，挑战了多样化的MAS角色会导致更好结果的观点。我们开源了我们的MAS和实验代码。 

---
# Mixture of Experts in Large Language Models 

**Title (ZH)**: 大型语言模型中的专家混合模型 

**Authors**: Danyang Zhang, Junhao Song, Ziqian Bi, Yingfang Yuan, Tianyang Wang, Joe Yeong, Junfeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11181)  

**Abstract**: This paper presents a comprehensive review of the Mixture-of-Experts (MoE) architecture in large language models, highlighting its ability to significantly enhance model performance while maintaining minimal computational overhead. Through a systematic analysis spanning theoretical foundations, core architectural designs, and large language model (LLM) applications, we examine expert gating and routing mechanisms, hierarchical and sparse MoE configurations, meta-learning approaches, multimodal and multitask learning scenarios, real-world deployment cases, and recent advances and challenges in deep learning. Our analysis identifies key advantages of MoE, including superior model capacity compared to equivalent Bayesian approaches, improved task-specific performance, and the ability to scale model capacity efficiently. We also underscore the importance of ensuring expert diversity, accurate calibration, and reliable inference aggregation, as these are essential for maximizing the effectiveness of MoE architectures. Finally, this review outlines current research limitations, open challenges, and promising future directions, providing a foundation for continued innovation in MoE architecture and its applications. 

**Abstract (ZH)**: 这篇论文对大型语言模型中的Mixture-of-Experts（MoE）架构进行了全面回顾，强调了其在显著提升模型性能的同时保持了较小的计算开销。通过系统分析理论基础、核心架构设计、以及大型语言模型（LLM）应用，我们考察了专家门控和路由机制、层级和稀疏MoE配置、元学习方法、多模态和多任务学习场景、真实世界部署案例以及深度学习中的最新进展和挑战。我们的分析指出了MoE的关键优势，包括与等效的贝叶斯方法相比更强大的模型容量、更好的任务特定性能以及高效扩展模型容量的能力。我们还强调了确保专家多样性、准确校准和可靠的推理聚合的重要性，这些对于最大化MoE架构的有效性至关重要。最后，本文概述了当前研究限制、开放挑战以及有希望的未来发展方向，为MoE架构及其应用的持续创新奠定了基础。 

---
# LogTinyLLM: Tiny Large Language Models Based Contextual Log Anomaly Detection 

**Title (ZH)**: LogTinyLLM：基于上下文的日志异常检测小型大型语言模型 

**Authors**: Isaiah Thompson Ocansey, Ritwik Bhattacharya, Tanmay Sen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11071)  

**Abstract**: Log anomaly detection using traditional rule based or deep learning based methods is often challenging due to the large volume and highly complex nature of log sequence. So effective way of detection of anomalous sequence of logs is crucial for system maintenance and development. This paper proposes parameter efficient finetuning specifically low rank adaptation (LoRA) and adapter based approaches for finding contextual anomalies in sequence of logs in large log data set. It compares different tiny large language models (LLMs) on the Thunderbird dataset. The results show that LoRA based finetuning provides substantial performance improvements of 18 to 19 percentage over LogBert based full finetuning approach, achieving accuracy scores between 97.76% and 98.83% compared to 79.37%. 

**Abstract (ZH)**: 使用传统基于规则或深度学习的方法进行日志异常检测往往由于日志序列的大量和高度复杂性而具有挑战性。因此，有效的日志异常序列检测方法对于系统维护和开发至关重要。本文提出了一种参数高效的微调方法，特别是低秩适应（LoRA）和适配器基方法，以在大数据日志集中发现日志序列中的上下文异常。该研究在Thunderbird数据集上比较了不同的小型大型语言模型（LLMs）。结果表明，基于LoRA的微调在日志BERT完全微调方法上提供了18%到19%的显著性能提升，准确率达到了97.76%至98.83%，而后者仅为79.37%。 

---
# SWE-MERA: A Dynamic Benchmark for Agenticly Evaluating Large Language Models on Software Engineering Tasks 

**Title (ZH)**: SWE-MERA：一种动态基准，用于自主评估大型语言模型在软件工程任务中的表现 

**Authors**: Pavel Adamenko, Mikhail Ivanov, Aidar Valeev, Rodion Levichev, Pavel Zadorozhny, Ivan Lopatin, Dmitry Babayev, Alena Fenogenova, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2507.11059)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) in software engineering has revealed critical limitations in existing benchmarks, particularly the widely used SWE-bench dataset. Recent studies have uncovered severe data contamination issues, e.g. SWE-bench reports 32.67% of successful patches involve direct solution leakage and 31.08\% pass due to inadequate test cases. We introduce SWE-MERA, a dynamic, continuously updated benchmark designed to address these fundamental challenges through an automated collection of real-world GitHub issues and rigorous quality validation. Our approach implements a reliable pipeline that ensures quality while minimizing contamination risks, resulting in approximately 10,000 potential tasks with 300 samples currently available. Evaluation using the Aider coding agent demonstrates strong discriminative power in state-of-the-art models. We report performance across a dozen recent LLMs evaluated on tasks collected between September 2024 and June 2025. 

**Abstract (ZH)**: 大语言模型在软件工程中的迅速发展揭示了现有基准的严重局限性，特别是广泛使用的SWE-bench数据集。近期研究发现了严重的数据污染问题，例如，SWE-bench报告32.67%的成功补丁涉及直接解题泄露，31.08%通过了由于不足的测试案例。我们引入SWE-MERA，这是一种动态的、不断更新的基准，旨在通过自动收集GitHub实际问题并进行严格的品质验证来解决这些根本性挑战。我们的方法实施了一种可靠的流水线，确保品质同时最大限度地减少污染风险，目前已收集大约10,000个潜在任务，其中300个样本可用。使用Aider编码代理进行评估展示了最先进的模型的强大区分能力。我们在2024年9月至2025年6月期间收集的任务上评估了十几种近期的大语言模型。 

---
# LLM-Augmented Symptom Analysis for Cardiovascular Disease Risk Prediction: A Clinical NLP 

**Title (ZH)**: LLM增强症状分析在心血管疾病风险预测中的临床NLP研究 

**Authors**: Haowei Yang, Ziyu Shen, Junli Shao, Luyao Men, Xinyue Han, Jing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.11052)  

**Abstract**: Timely identification and accurate risk stratification of cardiovascular disease (CVD) remain essential for reducing global mortality. While existing prediction models primarily leverage structured data, unstructured clinical notes contain valuable early indicators. This study introduces a novel LLM-augmented clinical NLP pipeline that employs domain-adapted large language models for symptom extraction, contextual reasoning, and correlation from free-text reports. Our approach integrates cardiovascular-specific fine-tuning, prompt-based inference, and entity-aware reasoning. Evaluations on MIMIC-III and CARDIO-NLP datasets demonstrate improved performance in precision, recall, F1-score, and AUROC, with high clinical relevance (kappa = 0.82) assessed by cardiologists. Challenges such as contextual hallucination, which occurs when plausible information contracts with provided source, and temporal ambiguity, which is related with models struggling with chronological ordering of events are addressed using prompt engineering and hybrid rule-based verification. This work underscores the potential of LLMs in clinical decision support systems (CDSS), advancing early warning systems and enhancing the translation of patient narratives into actionable risk assessments. 

**Abstract (ZH)**: 及时识别和准确分层心血管疾病（CVD）风险对于降低全球死亡率仍然至关重要。尽管现有的预测模型主要利用结构化数据，但未结构化的临床笔记包含有价值的早期指标。本研究引入了一种新型的LLM增强临床NLP管道，采用领域适应的大语言模型进行症状提取、上下文推理和自由文本报告中的相关性分析。我们的方法结合了心血管特定的微调、基于提示的推断和实体感知推理。在MIMIC-III和CARDIO-NLP数据集上的评估结果显示，在精确度、召回率、F1分数和AUROC方面均有所提升，并且经过心脏病专家评估具有高度的临床相关性（κ=0.82）。通过提示工程和混合规则验证应对了上下文幻觉和时间模糊性等挑战。本工作突显了LLM在临床决策支持系统（CDSS）中的潜在价值，推进了早期预警系统的改进，并增强了患者叙述转化为可操作的风险评估的转化。 

---
# First-Order Error Matters: Accurate Compensation for Quantized Large Language Models 

**Title (ZH)**: 一阶误差很重要：准确补偿量化大型语言模型 

**Authors**: Xingyu Zheng, Haotong Qin, Yuye Li, Jiakai Wang, Jinyang Guo, Michele Magno, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11017)  

**Abstract**: Post-training quantization (PTQ) offers an efficient approach to compressing large language models (LLMs), significantly reducing memory access and computational costs. Existing compensation-based weight calibration methods often rely on a second-order Taylor expansion to model quantization error, under the assumption that the first-order term is negligible in well-trained full-precision models. However, we reveal that the progressive compensation process introduces accumulated first-order deviations between latent weights and their full-precision counterparts, making this assumption fundamentally flawed. To address this, we propose FOEM, a novel PTQ method that explicitly incorporates first-order gradient terms to improve quantization error compensation. FOEM approximates gradients by directly computing the difference between latent and full-precision weights, avoiding the high cost and limited generalization of backpropagation-based gradient computation. This approach introduces minimal additional computational overhead. Moreover, FOEM leverages precomputed Cholesky factors to efficiently recover the inverse of Hessian submatrices in real time. Extensive experiments across a wide range of models and benchmarks demonstrate that FOEM consistently outperforms the classical GPTQ method. In 3-bit weight-only quantization, FOEM reduces the perplexity of Llama3-8B by 89.6%, and improves the 5-shot MMLU accuracy of Llama3-70B from 51.7% to 74.9%, approaching the full-precision performance of 78.6%. Furthermore, FOEM can be seamlessly integrated with advanced techniques such as GPTAQ and SpinQuant, yielding additional improvements under the challenging W4A4KV4 setting, and further narrowing the accuracy gap with full-precision baselines beyond what current state-of-the-art methods achieve. The code is available at this https URL. 

**Abstract (ZH)**: .POST-TRAINING 量化 (PTQ) 提供了一种高效的方法来压缩大规模语言模型 (LLMs)，显著减少了内存访问和计算成本。现有的基于补偿的权重标定方法往往依赖于二阶泰勒展开来建模量化误差，并假设在充分训练的全精度模型中一阶项是可忽略的。然而，我们揭示了渐进补偿过程在潜在权重与其全精度对应值之间引入了一阶偏差累计，使这一假设从根本上变为错误。为此，我们提出了FOEM，这是一种新颖的PTQ方法，明确引入了一阶梯度项以改进量化误差补偿。FOEM 通过直接计算潜在权重和全精度权重之间的差异来近似梯度，避免了基于反向传播的梯度计算高成本和有限泛化能力。这种做法引入了最小的额外计算开销。此外，FOEM 利用预计算的 Cholesky 因子以高效方式实时恢复海森矩阵的逆。广泛的实验表明，FOEM 在多种模型和基准测试中表现始终优于经典的GPTQ方法。在3比特权重量化中，FOEM 将Llama3-8B的困惑度降低了89.6%，并将Llama3-70B的5-shot MMLU准确性从51.7%提高到74.9%，接近全精度性能的78.6%。此外，FOEM 可以无缝集成到如GPTAQ和SpinQuant等高级技术中，在具有挑战性的W4A4KV4设置下提供额外改进，并进一步缩小与全精度基线之间的准确性差距，超越当前最先进的方法的实现。代码可在此处获取。 

---
# Modeling Understanding of Story-Based Analogies Using Large Language Models 

**Title (ZH)**: 基于故事类类比的理解建模研究 

**Authors**: Kalit Inani, Keshav Kabra, Vijay Marupudi, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2507.10957)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have brought them closer to matching human cognition across a variety of tasks. How well do these models align with human performance in detecting and mapping analogies? Prior research has shown that LLMs can extract similarities from analogy problems but lack robust human-like reasoning. Building on Webb, Holyoak, and Lu (2023), the current study focused on a story-based analogical mapping task and conducted a fine-grained evaluation of LLM reasoning abilities compared to human performance. First, it explored the semantic representation of analogies in LLMs, using sentence embeddings to assess whether they capture the similarity between the source and target texts of an analogy, and the dissimilarity between the source and distractor texts. Second, it investigated the effectiveness of explicitly prompting LLMs to explain analogies. Throughout, we examine whether LLMs exhibit similar performance profiles to those observed in humans by evaluating their reasoning at the level of individual analogies, and not just at the level of overall accuracy (as prior studies have done). Our experiments include evaluating the impact of model size (8B vs. 70B parameters) and performance variation across state-of-the-art model architectures such as GPT-4 and LLaMA3. This work advances our understanding of the analogical reasoning abilities of LLMs and their potential as models of human reasoning. 

**Abstract (ZH)**: 最近大型语言模型（LLMs）的发展使它们在各种任务上更接近人类认知。这些模型在检测和映射类比方面与人类表现的匹配程度如何？先前的研究表明，LLMs可以从类比问题中提取相似性，但在稳健的人类-like推理方面存在不足。在此基础上，本研究专注于基于故事的类比映射任务，并对LLMs的推理能力与人类表现进行了细致评估。首先，研究探索了LLMs中类比的语义表示，使用句子嵌入评估它们是否捕捉了源文本和目标文本之间的相似性，以及源文本和干扰文本之间的不相似性。其次，研究调查了显式提示LLMs解释类比的有效性。在整个过程中，我们通过评估个体类比的推理能力，而不是仅评估总体准确性（如先前研究所做的），考察LLMs是否表现出与人类相似的表现模式。我们的实验还包括评估模型大小（8B vs. 70B参数）及在当前先进模型架构如GPT-4和LLaMA3上的性能差异。这项工作推进了我们对LLMs类比推理能力和其作为人类推理模型的潜力的理解。 

---
# Artificial Finance: How AI Thinks About Money 

**Title (ZH)**: 人工智能金融：AI如何思考金钱 

**Authors**: Orhan Erdem, Ragavi Pobbathi Ashok  

**Link**: [PDF](https://arxiv.org/pdf/2507.10933)  

**Abstract**: In this paper, we explore how large language models (LLMs) approach financial decision-making by systematically comparing their responses to those of human participants across the globe. We posed a set of commonly used financial decision-making questions to seven leading LLMs, including five models from the GPT series(GPT-4o, GPT-4.5, o1, o3-mini), Gemini 2.0 Flash, and DeepSeek R1. We then compared their outputs to human responses drawn from a dataset covering 53 nations. Our analysis reveals three main results. First, LLMs generally exhibit a risk-neutral decision-making pattern, favoring choices aligned with expected value calculations when faced with lottery-type questions. Second, when evaluating trade-offs between present and future, LLMs occasionally produce responses that appear inconsistent with normative reasoning. Third, when we examine cross-national similarities, we find that the LLMs' aggregate responses most closely resemble those of participants from Tanzania. These findings contribute to the understanding of how LLMs emulate human-like decision behaviors and highlight potential cultural and training influences embedded within their outputs. 

**Abstract (ZH)**: 本文通过系统比较七种领先的大型语言模型（包括GPT系列的GPT-4o、GPT-4.5、o1、o3-mini，Gemini 2.0 Flash和DeepSeek R1）和来自覆盖53个国家的人类参与者数据集的响应，探讨了大型语言模型在金融决策中的方法。研究发现三个主要结果：首先，大型语言模型通常表现出风险中立的决策模式，倾向于在彩票类型问题中选择符合预期价值计算的结果；其次，在评估现在和未来之间的权衡时，大型语言模型偶尔会产生与规范推理不符的回应；第三，当考察跨境相似性时，发现大型语言模型的综合回应最接近坦桑尼亚参与者的回应。这些发现有助于理解大型语言模型如何模拟类似人类的决策行为，并突出其输出中嵌入的文化和训练影响。 

---
# HanjaBridge: Resolving Semantic Ambiguity in Korean LLMs via Hanja-Augmented Pre-Training 

**Title (ZH)**: HanjaBridge：通过 Hanja 增强预训练解决韩语大规模语言模型的语义歧义 

**Authors**: Seungho Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10920)  

**Abstract**: Large language models (LLMs) often show poor performance in low-resource languages like Korean, partly due to unique linguistic challenges such as homophonous Sino-Korean words that are indistinguishable in Hangul script. To address this semantic ambiguity, we propose HanjaBridge, a novel meaning-injection technique integrated into a continual pre-training (CPT) framework. Instead of deterministically mapping a word to a single Hanja (Chinese character), HanjaBridge presents the model with all possible Hanja candidates for a given homograph, encouraging the model to learn contextual disambiguation. This process is paired with token-level knowledge distillation to prevent catastrophic forgetting. Experimental results show that HanjaBridge significantly improves Korean language understanding, achieving a 21\% relative improvement on the KoBALT benchmark. Notably, by reinforcing semantic alignment between Korean and Chinese through shared Hanja, we observe a strong positive cross-lingual transfer. Furthermore, these gains persist even when Hanja augmentation is omitted at inference time, ensuring practical efficiency with no additional run-time cost. 

**Abstract (ZH)**: Large语言模型（LLMs）在韩语等低资源语言上常表现出较差的效果，部分原因是由于韩语中特有的语义挑战，如同音异义的汉语借词，在hangul字母中无法区分。为解决这一语义歧义问题，我们提出了一种名为HanjaBridge的新颖意义注入技术，将其集成到持续预训练（CPT）框架中。不同于将一个词确定性地映射到单个汉字，HanjaBridge为给定的同形词展示所有可能的汉字候选，促使模型学习上下文消歧。该过程配有标记级的知识蒸馏，以防止灾难性遗忘。实验结果表明，HanjaBridge显著提高了韩语理解能力，在KoBALT基准测试上取得了21%的相对改进。值得注意的是，通过共享汉字强化韩语和汉语之间的语义对齐，我们观察到强烈的跨语言迁移效果。此外，即使在推理时不使用汉字扩充，这些增益仍然存在，确保了实际中的高效性且无额外运行时成本。 

---
# Semantic Context for Tool Orchestration 

**Title (ZH)**: 工具编排的语义上下文 

**Authors**: Robert Müller  

**Link**: [PDF](https://arxiv.org/pdf/2507.10820)  

**Abstract**: This paper demonstrates that Semantic Context (SC), leveraging descriptive tool information, is a foundational component for robust tool orchestration. Our contributions are threefold. First, we provide a theoretical foundation using contextual bandits, introducing SC-LinUCB and proving it achieves lower regret and adapts favourably in dynamic action spaces. Second, we provide parallel empirical validation with Large Language Models, showing that SC is critical for successful in-context learning in both static (efficient learning) and non-stationary (robust adaptation) settings. Third, we propose the FiReAct pipeline, and demonstrate on a benchmark with over 10,000 tools that SC-based retrieval enables an LLM to effectively orchestrate over a large action space. These findings provide a comprehensive guide to building more sample-efficient, adaptive, and scalable orchestration agents. 

**Abstract (ZH)**: 本文展示了语义上下文（SC），利用描述性工具信息，是稳健工具编排的基础组件。我们的贡献包括三个方面。首先，我们使用上下文臂赛选引入SC-LinUCB，并提供理论基础，证明它能实现更低的遗憾并以动态动作空间中表现出良好的适应性。其次，我们通过大规模语言模型并行实验证明，SC 对于成功进行上下文内学习（在静态情况下实现高效学习和在非平稳情况下实现稳健适应）至关重要。第三，我们提出了FiReAct 管道，并在一个包含超过10,000个工具的基准测试中证明，基于SC 的检索使大语言模型能够有效地在大动作空间中编排。这些发现为构建更高效、适应性强和可扩展的编排代理提供了全面指南。 

---
# Warehouse Spatial Question Answering with LLM Agent 

**Title (ZH)**: 基于LLM代理的仓库空间问答 

**Authors**: Hsiang-Wei Huang, Jen-Hao Cheng, Kuang-Ming Chen, Cheng-Yen Yang, Bahaa Alattar, Yi-Ru Lin, Pyongkun Kim, Sangwon Kim, Kwangju Kim, Chung-I Huang, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10778)  

**Abstract**: Spatial understanding has been a challenging task for existing Multi-modal Large Language Models~(MLLMs). Previous methods leverage large-scale MLLM finetuning to enhance MLLM's spatial understanding ability. In this paper, we present a data-efficient approach. We propose a LLM agent system with strong and advanced spatial reasoning ability, which can be used to solve the challenging spatial question answering task in complex indoor warehouse scenarios. Our system integrates multiple tools that allow the LLM agent to conduct spatial reasoning and API tools interaction to answer the given complicated spatial question. Extensive evaluations on the 2025 AI City Challenge Physical AI Spatial Intelligence Warehouse dataset demonstrate that our system achieves high accuracy and efficiency in tasks such as object retrieval, counting, and distance estimation. The code is available at: this https URL 

**Abstract (ZH)**: 现有的多模态大规模语言模型在空间理解方面存在挑战。先前的方法通过大规模多模态大型语言模型微调来增强其空间理解能力。本文提出了一种数据高效的方法。我们提出了一种具有强大和先进空间推理能力的LLM代理系统，该系统可以用于解决复杂室内仓库场景中的空间问答任务。我们的系统整合了多种工具，使LLM代理能够进行空间推理和API工具交互以回答给定的复杂空间问题。在2025 AI City Challenge Physical AI Spatial Intelligence Warehouse数据集上的广泛应用表明，我们的系统在物体检索、计数和距离估计等任务中实现了高准确性和高效性。代码available at: this https URL。 

---
# Exploring User Security and Privacy Attitudes and Concerns Toward the Use of General-Purpose LLM Chatbots for Mental Health 

**Title (ZH)**: 探索用户在使用通用型大语言模型聊天机器人进行心理健康管理方面的安全和隐私态度与关切 

**Authors**: Jabari Kwesi, Jiaxun Cao, Riya Manchanda, Pardis Emami-Naeini  

**Link**: [PDF](https://arxiv.org/pdf/2507.10695)  

**Abstract**: Individuals are increasingly relying on large language model (LLM)-enabled conversational agents for emotional support. While prior research has examined privacy and security issues in chatbots specifically designed for mental health purposes, these chatbots are overwhelmingly "rule-based" offerings that do not leverage generative AI. Little empirical research currently measures users' privacy and security concerns, attitudes, and expectations when using general-purpose LLM-enabled chatbots to manage and improve mental health. Through 21 semi-structured interviews with U.S. participants, we identified critical misconceptions and a general lack of risk awareness. Participants conflated the human-like empathy exhibited by LLMs with human-like accountability and mistakenly believed that their interactions with these chatbots were safeguarded by the same regulations (e.g., HIPAA) as disclosures with a licensed therapist. We introduce the concept of "intangible vulnerability," where emotional or psychological disclosures are undervalued compared to more tangible forms of information (e.g., financial or location-based data). To address this, we propose recommendations to safeguard user mental health disclosures with general-purpose LLM-enabled chatbots more effectively. 

**Abstract (ZH)**: 个体日益依赖大型语言模型（LLM）驱动的对话代理寻求情感支持。尽管先前的研究已经探讨了专门设计用于心理健康目的的聊天机器人的隐私和安全问题，但这些聊天机器人大多是基于规则的解决方案，并未利用生成式AI。目前，很少有实证研究测量用户在使用通用型LLM驱动的聊天机器人管理并改善心理健康时的隐私和安全担忧、态度和期望。通过与美国参与者进行21次半结构化访谈，我们识别出了关键的误解和一般的风险意识缺乏问题。参与者将LLMs展现的人类同理心与人类责任感混为一谈，并错误地认为他们与这些聊天机器人的互动受与受过许可训练的心理治疗师相同的法规（如HIPAA）保护。我们提出了“无形脆弱性”的概念，指的是情感或心理披露相比有形信息（如财务或位置数据）被低估。为了解决这一问题，我们提出了建议，以更有效地保护用户在通用型LLM驱动的聊天机器人中的心理健康披露。 

---
# A Code Comprehension Benchmark for Large Language Models for Code 

**Title (ZH)**: 大型语言模型理解代码的基准数据集 

**Authors**: Jayant Havare, Saurav Chaudhary, Ganesh Ramakrishnan, Kaushik Maharajan, Srikanth Tamilselvam  

**Link**: [PDF](https://arxiv.org/pdf/2507.10641)  

**Abstract**: Large Language Models have shown impressive capabilities in coding tasks like code generation and code completion, as they have been trained on a large amount of code data. Also, since one of the core pretraining objectives is Next Token Prediction, these models tends to learn surface-level syntactic patterns in code. However, this does not guarantee code comprehension ability i.e. the ability to capture the semantics of the code. In our opinion, this is the reason why these models often underperform on tasks that require deeper semantic understanding, such as code debugging and code optimization. To address this, we propose fine-tuning these models specifically for code comprehension tasks using large-scale datasets, enabling them to develop a more robust understanding of code semantics. We evaluate three code models of varying sizes on a suite of code comprehension tasks designed to assess semantic understanding beyond surface-level syntactic pattern matching. In particular, we analyze performance on the Subjectivity Grading Task and observe that model performance improves after fine-tuning on relevant downstream tasks. The most significant improvement is seen in the QWQ-32B model, where accuracy increases from 70% to 83.47%. A similar or explainable trend is observed across other models, clearly indicating an enhancement in code comprehension ability. Among the models studied, the DPO-fine-tuned Codestral-22B achieves the highest micro-accuracy of 87.66% on the Subjectivity Grading Task. 

**Abstract (ZH)**: 大型语言模型在编码任务如代码生成和代码补全方面展示了令人印象深刻的 capability，因为它们是在大量的代码数据上进行训练的。由于其中一个核心预训练目标是下一个 token 预测，这些模型倾向于学习代码的表层句法模式。然而，这并不能保证代码理解能力，即捕获代码语义的能力。依我们之见，这是这些模型在需要深度语义理解的任务（如代码调试和代码优化）中常常表现不佳的原因。为了解决这个问题，我们建议使用大规模数据集对这些模型进行细调，以专门针对代码理解任务，使它们能够更牢固地理解代码语义。我们在一系列设计用于评估超越表层句法模式匹配的语义理解的代码理解任务上评估了三种不同规模的代码模型。特别地，我们分析了主观性评分任务上的表现，并观察到在相关下游任务上进行细调后，模型性能有所提升。QWQ-32B 模型表现尤为显著，准确率从 70% 提高到 83.47%。其他模型中也观察到相似的或可解释的趋势，明确表明代码理解能力有所增强。在研究的模型中，DPO-Fine-tuned Codestral-22B 在主观性评分任务上实现了最高的微准确率 87.66%。 

---
# SPICEAssistant: LLM using SPICE Simulation Tools for Schematic Design of Switched-Mode Power Supplies 

**Title (ZH)**: SPICEAssistant：使用SPICE仿真工具进行开关模式电源电路设计的大型语言模型 

**Authors**: Simon Nau, Jan Krummenauer, André Zimmermann  

**Link**: [PDF](https://arxiv.org/pdf/2507.10639)  

**Abstract**: State-of-the-art large language models (LLMs) show high performance across a wide range of tasks in many domains of science. In the field of electronic design automation (EDA), it is yet to be determined to what extent they are capable to understand, adapt, and dimension electronic circuits. This paper focuses on the application of LLMs to switched-mode power supply (SMPS) design on printed circuit boards (PCBs). Particular challenges for LLMs in this context include their limited ability to interpret results from key simulation tools like SPICE and the multi-step design process. To address these challenges, we suggest SPICEAssistant, a framework that provides a broad selection of tools to an LLM. The tools serve as an interface to SPICE, allowing the LLM to interact flexibly with the simulator to estimate the impact of its modifications to the circuit. To evaluate the performance of SPICEAssistant, we defined a benchmark consisting of 256 questions testing the ability to adapt circuit netlists to fulfil different SMPS design tasks. The benchmarking results show that simulation feedback effectively improves SMPS design capabilities of LLMs. An increasing number of simulation iterations leads to enhanced performance. The SPICEAssistant framework significantly outperforms the standalone LLM GPT-4o on the benchmark by approximately 38%. 

**Abstract (ZH)**: 最先进的大型语言模型（LLMs）在科学多个领域的广泛任务中表现出高度性能。在电子设计自动化（EDA）领域，尚未明确LLMs在理解、适应和优化电子电路方面的能力。本文专注于将LLMs应用于印制电路板（PCBs）上的开关模式电源（SMPS）设计。在这一背景下，LLMs面临的特定挑战包括其解释关键仿真工具（如SPICE）结果的能力有限，以及多步骤的设计过程。为解决这些问题，我们建议采用SPICEAssistant框架，该框架为LLMs提供了一系列工具，作为与SPICE交互的接口，使LLMs能够灵活地与仿真器互动以评估其对电路修改的影响。为了评估SPICEAssistant的性能，我们定义了一个基准测试，其中包括256个问题，测试LLMs调整电路网表以完成不同SMPS设计任务的能力。基准测试结果表明，仿真反馈有效提升了LLMs的SMPS设计能力。逐步增加的仿真迭代次数提高了性能。SPICEAssistant框架在基准测试中显著优于独立的LLM GPT-4o，约38%。 

---
# GHPO: Adaptive Guidance for Stable and Efficient LLM Reinforcement Learning 

**Title (ZH)**: GHPO: 自适应指导以实现稳定高效的LLM强化学习 

**Authors**: Ziru Liu, Cheng Gong, Xinyu Fu, Yaofang Liu, Ran Chen, Shoubo Hu, Suiyun Zhang, Rui Liu, Qingfu Zhang, Dandan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10628)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for facilitating the self-improvement of large language models (LLMs), particularly in the domain of complex reasoning tasks. However, prevailing on-policy RL methods often contend with significant training instability and inefficiency. This is primarily due to a capacity-difficulty mismatch, where the complexity of training data frequently outpaces the model's current capabilities, leading to critically sparse reward signals and stalled learning progress. This challenge is particularly acute for smaller, more resource-efficient LLMs. To overcome this, we introduce the Guided Hybrid Policy Optimization (GHPO), a novel difficulty-aware reinforcement learning framework. GHPO dynamically calibrates task difficulty by employing adaptive prompt refinement to provide targeted guidance. This unique approach adaptively balances direct imitation learning for problems currently beyond the model's reach with exploration-based reinforcement learning for more manageable tasks, effectively creating a smooth and optimized learning curriculum. Extensive experiments demonstrate that GHPO achieves an average performance gain of approximately 5% across six challenging mathematics benchmarks, consistently outperforming strong on-policy reinforcement learning and curriculum learning baselines. Further analysis confirms that our framework significantly enhances both training stability and final reasoning performance, thus offering a scalable and efficient solution for developing powerful and robust reasoning models. 

**Abstract (ZH)**: 可验证奖励增强学习（RLVR） recently emerged as a powerful paradigm for facilitating the self-improvement of large language models (LLMs), particularly in the domain of complex reasoning tasks. However, prevailing on-policy RL methods often contend with significant training instability and inefficiency. This is primarily due to a capacity-difficulty mismatch, where the complexity of training data frequently outpaces the model's current capabilities, leading to critically sparse reward signals and stalled learning progress. To overcome this, we introduce the Guided Hybrid Policy Optimization (GHPO), a novel difficulty-aware reinforcement learning framework. 

---
# LLMs Meet Cross-Modal Time Series Analytics: Overview and Directions 

**Title (ZH)**: LLMs融入跨模态时间序列分析：概览与发展方向 

**Authors**: Chenxi Liu, Hao Miao, Cheng Long, Yan Zhao, Ziyue Li, Panos Kalnis  

**Link**: [PDF](https://arxiv.org/pdf/2507.10620)  

**Abstract**: Large Language Models (LLMs) have emerged as a promising paradigm for time series analytics, leveraging their massive parameters and the shared sequential nature of textual and time series data. However, a cross-modality gap exists between time series and textual data, as LLMs are pre-trained on textual corpora and are not inherently optimized for time series. In this tutorial, we provide an up-to-date overview of LLM-based cross-modal time series analytics. We introduce a taxonomy that classifies existing approaches into three groups based on cross-modal modeling strategies, e.g., conversion, alignment, and fusion, and then discuss their applications across a range of downstream tasks. In addition, we summarize several open challenges. This tutorial aims to expand the practical application of LLMs in solving real-world problems in cross-modal time series analytics while balancing effectiveness and efficiency. Participants will gain a thorough understanding of current advancements, methodologies, and future research directions in cross-modal time series analytics. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在时间序列分析领域展现出 promising 的范式，通过利用其庞大的参数和文本数据与时间序列数据共享的序列特性。然而，时间序列数据与文本数据之间存在跨模态差距，因为LLMs是在文本语料库上进行预训练的，并未固有地优化时间序列分析。在本教程中，我们将提供基于LLM的时间序列跨模态分析的最新综述。我们介绍了基于跨模态建模策略的分类体系，将其划分为转换、对齐和融合三种类别，并讨论了它们在一系列下游任务中的应用。此外，我们总结了几项-open-challenges。本教程旨在平衡效用与效率，扩大LLMs在解决跨模态时间序列分析中实际问题的应用范围。参与者将深入了解跨模态时间序列分析的最新进展、方法论及未来研究方向。 

---
# Scalpel vs. Hammer: GRPO Amplifies Existing Capabilities, SFT Replaces Them 

**Title (ZH)**: 刀片 vs. 卤头：GRPO 强化现有能力，SFT 取而代之 

**Authors**: Neel Rajani, Aryo Pradipta Gema, Seraphina Goldfarb-Tarrant, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2507.10616)  

**Abstract**: Training large language models (LLMs) for reasoning via maths and code datasets has become a major new focus in LLM post-training. Two particularly popular approaches are reinforcement learning (RL) and supervised fine-tuning (SFT), but their training dynamics are poorly understood. We present a comparative analysis of RL and SFT on the same maths problems with the same model and similar hyperparameters. We find that RL yields minor in-domain gains on maths and slight degradation on knowledge-intensive benchmarks like MMLU, while both trends are more pronounced in SFT. We also analyse model parameters across checkpoints, observing that both algorithms modify query and key weights the most. Meanwhile, SFT exhibits greater updates and also affects mid-layer MLPs more, leading us to hypothesise that this may have caused the out-of-domain degradation. We therefore investigate whether freezing parts of the model during training can mitigate the reduced performance on knowledge-intensive benchmarks. However, our results are inconclusive, with benefits on GPQA:Diamond and degradation on other benchmarks. Taken together, our observations provide a preliminary indication for why RL amplifies existing capabilities, while SFT replaces old skills with new ones. 

**Abstract (ZH)**: 通过数学和代码数据集进行逻辑推理训练的大语言模型（LLMs）已成为LLM后训练中的一个主要新重点。强化学习（RL）和监督微调（SFT）是两种特别流行的方法，但它们的训练动态尚不完全理解。我们对同一数学问题、同一模型和相似超参数条件下，RL和SFT进行了比较分析。我们发现，RL在数学领域内仅带来轻微提升，但在知识密集型基准测试如MMLU上表现出轻微下降；而这种趋势在SFT中更为明显。我们还分析了检查点中的模型参数，观察到两种算法中最常修改的是查询和键权重。同时，SFT表现出更大的更新，并且更多地影响中间层的MLP，使我们假设这可能是导致领域外下降的原因。因此，我们探究在训练过程中冻结部分模型是否能缓解在知识密集型基准测试中的性能下降。然而，我们的结果不够明确，对GPQA:Diamond有利，但在其他基准测试中表现出下降。综合我们的观察，这为解释为什么RL放大现有能力、而SFT用新技能取代旧技能提供了初步线索。 

---
# Fine-tuning Large Language Model for Automated Algorithm Design 

**Title (ZH)**: 大规模语言模型的微调以实现自动化算法设计 

**Authors**: Fei Liu, Rui Zhang, Xi Lin, Zhichao Lu, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10614)  

**Abstract**: The integration of large language models (LLMs) into automated algorithm design has shown promising potential. A prevalent approach embeds LLMs within search routines to iteratively generate and refine candidate algorithms. However, most existing methods rely on off-the-shelf LLMs trained for general coding tasks,leaving a key question open: Do we need LLMs specifically tailored for algorithm design? If so, how can such LLMs be effectively obtained and how well can they generalize across different algorithm design tasks? In this paper, we take a first step toward answering these questions by exploring fine-tuning of LLMs for algorithm design. We introduce a Diversity-Aware Rank based (DAR) sampling strategy to balance training data diversity and quality, then we leverage direct preference optimization to efficiently align LLM outputs with task objectives. Our experiments, conducted on Llama-3.2-1B-Instruct and Llama- 3.1-8B-Instruct, span three distinct algorithm design tasks. Results suggest that finetuned LLMs can significantly outperform their off-the-shelf counterparts with the smaller Llama-3.2-1B-Instruct and match the larger Llama-3.1-8B-Instruct on the admissible set problem. Moreover, we observe promising generalization: LLMs finetuned on specific algorithm design tasks also improve performance on related tasks with varying settings. These findings highlight the value of task-specific adaptation for LLMs in algorithm design and open new avenues for future research. 

**Abstract (ZH)**: 大型语言模型在自动化算法设计中的整合展示了潜在的应用前景。一种常见的方法是将大型语言模型嵌入搜索过程中，以迭代地生成和优化候选算法。然而，现有方法大多依赖于通用编码任务训练的标准大型语言模型，这留下了一个关键问题：是否需要专门针对算法设计的大型语言模型？如果是的话，如何有效获取这样的大型语言模型，以及它们在不同算法设计任务上的泛化能力如何？在本文中，我们通过探索针对算法设计的大型语言模型微调，迈出了解答这些问题的第一步。我们引入了一种旨在平衡训练数据多样性和质量的多样性感知排名（DAR）采样策略，然后利用直接偏好优化对大型语言模型的输出进行高效调整，使其与任务目标相匹配。实验在Llama-3.2-1B-Instruct和Llama-3.1-8B-Instruct上进行，涵盖三个不同的算法设计任务。结果表明，微调后的大型语言模型可以显著优于标准大型语言模型，尤其是在可接受集合问题上，小型的Llama-3.2-1B-Instruct的表现甚至可以匹纯洁大型的Llama-3.1-8B-Instruct。此外，我们观察到泛化能力：专门针对特定算法设计任务进行微调的大型语言模型，也可以在设置不同的相关任务中提高性能。这些发现突显了针对算法设计任务对大型语言模型进行任务特定适应的价值，并为未来的研究开辟了新的方向。 

---
# Sub-Scaling Laws: On the Role of Data Density and Training Strategies in LLMs 

**Title (ZH)**: 亚线性律：数据密度和训练策略在大规模语言模型中的作用 

**Authors**: Zhengyu Chen, Siqi Wang, Teng Xiao, Yudong Wang, Shiqi Chen, Xunliang Cai, Junxian He, Jingang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10613)  

**Abstract**: Traditional scaling laws in natural language processing suggest that increasing model size and training data enhances performance. However, recent studies reveal deviations, particularly in large language models, where performance improvements decelerate, which is a phenomenon known as sub-scaling. This paper revisits these scaling laws by examining the impact of data quality and training strategies on model performance. Through extensive empirical analysis of over 400 models, we identify high data density and non-optimal resource allocation as key factors contributing to sub-scaling. High data density leads to diminishing returns due to redundant information, while optimal resource allocation is crucial for sustained performance improvements. We propose a sub-optimal scaling law that better predicts performance in sub-scaling regimes, highlighting the importance of data quality and diversity. 

**Abstract (ZH)**: 传统自然语言处理中的缩放定律表明，增加模型规模和训练数据可以提升性能。然而，近期的研究揭示了偏差，特别是在大型语言模型中，性能提升趋于减缓，这一现象被称为亚缩放。本文通过研究数据质量和训练策略对模型性能的影响，重新审视这些缩放定律。通过对超过400个模型的广泛实证分析，我们发现高数据密度和非最优资源配置是导致亚缩放的关键因素。高数据密度导致因冗余信息而产生边际效益递减，而最优资源配置对于持续性能提升至关重要。我们提出了一种亚最优缩放定律，更好地预测亚缩放区域的性能，强调了数据质量和多样性的重要性。 

---
# LaSM: Layer-wise Scaling Mechanism for Defending Pop-up Attack on GUI Agents 

**Title (ZH)**: LaSM：面向GUI代理弹窗攻击的层级缩放机制 

**Authors**: Zihe Yan, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10610)  

**Abstract**: Graphical user interface (GUI) agents built on multimodal large language models (MLLMs) have recently demonstrated strong decision-making abilities in screen-based interaction tasks. However, they remain highly vulnerable to pop-up-based environmental injection attacks, where malicious visual elements divert model attention and lead to unsafe or incorrect actions. Existing defense methods either require costly retraining or perform poorly under inductive interference. In this work, we systematically study how such attacks alter the attention behavior of GUI agents and uncover a layer-wise attention divergence pattern between correct and incorrect outputs. Based on this insight, we propose \textbf{LaSM}, a \textit{Layer-wise Scaling Mechanism} that selectively amplifies attention and MLP modules in critical layers. LaSM improves the alignment between model saliency and task-relevant regions without additional training. Extensive experiments across 12 types of pop-up perturbations and 4 different model backbones show that LaSM consistently enhances the defense success rate. When combined with prompt-level alerts, LaSM achieves over 98\% robustness even under strong inductive attacks. Our findings reveal that attention misalignment is a core vulnerability in MLLM agents and can be effectively addressed through selective layer-wise modulation. 

**Abstract (ZH)**: 基于多模态大型语言模型的图形用户界面（GUI）代理在屏幕交互任务中展示了强大的决策能力，但仍然高度易受基于弹出窗口的环境注入攻击的影响，这些恶意视觉元素会转移模型的注意力并导致不安全或错误的操作。现有防护方法要么需要昂贵的重新训练，要么在归纳干扰下表现不佳。在本工作中，我们系统地研究了此类攻击如何改变GUI代理的注意力行为，并发现正确和错误输出之间存在逐层注意力偏差模式。基于这一洞察，我们提出了一种名为LaSM的逐层放大机制，该机制选择性地放大关键层中的注意力和MLP模块。LaSM在无需额外训练的情况下提高了模型显著性和任务相关区域之间的对齐程度。我们在包括12种不同类型弹出窗口扰动和4种不同模型架构的广泛实验中显示，LaSM始终提高了防护成功率。结合提示级别警报时，LaSM即使在强归纳攻击下也能实现超过98%的鲁棒性。我们的研究发现，注意力偏差是多模态大型语言模型代理的核心脆弱性，并可以通过选择性的逐层调节有效解决。 

---
# RedOne: Revealing Domain-specific LLM Post-Training in Social Networking Services 

**Title (ZH)**: RedOne: 揭示社交媒体服务中的领域特定LLM后训练 

**Authors**: Fei Zhao, Chonggang Lu, Yue Wang, Zheyong Xie, Ziyan Liu, Haofu Qian, JianZhao Huang, Fangcheng Shi, Zijie Meng, Hongcheng Guo, Mingqian He, Xinze Lyu, Yiming Lu, Ziyang Xiang, Zheyu Ye, Chengqiang Lu, Zhe Xu, Yi Wu, Yao Hu, Yan Gao, Jun Fan, Xiaolong Jiang, Weiting Liu, Boyang Wang, Shaosheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.10605)  

**Abstract**: As a primary medium for modern information dissemination, social networking services (SNS) have experienced rapid growth, which has proposed significant challenges for platform content management and interaction quality improvement. Recently, the development of large language models (LLMs) has offered potential solutions but existing studies focus on isolated tasks, which not only encounter diminishing benefit from the data scaling within individual scenarios but also fail to flexibly adapt to diverse real-world context. To address these challenges, we introduce RedOne, a domain-specific LLM designed to break the performance bottleneck of single-task baselines and establish a comprehensive foundation for the SNS. RedOne was developed through a three-stage training strategy consisting of continue pretraining, supervised fine-tuning, and preference optimization, using a large-scale real-world dataset. Through extensive experiments, RedOne maintains strong general capabilities, and achieves an average improvement up to 14.02% across 8 major SNS tasks and 7.56% in SNS bilingual evaluation benchmark, compared with base models. Furthermore, through online testing, RedOne reduced the exposure rate in harmful content detection by 11.23% and improved the click page rate in post-view search by 14.95% compared with single-tasks finetuned baseline models. These results establish RedOne as a robust domain-specific LLM for SNS, demonstrating excellent generalization across various tasks and promising applicability in real-world scenarios. 

**Abstract (ZH)**: 作为现代信息传播的主要媒介，社交网络服务（SNS）经历了 rapid growth，这对平台内容管理和互动质量的提升提出了重大挑战。近年来，大型语言模型（LLMs）的发展提供了潜在的解决方案，但现有研究主要集中在孤立的任务上，不仅在单一场景中面临数据规模增大的边际效益递减问题，还无法灵活适应多样的现实世界情境。为应对这些挑战，我们引入了 RedOne，这是一种领域特定的 LLM，旨在打破单任务基线的性能瓶颈，并为 SNS 建立全面的基础。RedOne 通过包含持续预训练、监督微调和偏好优化的三阶段训练策略进行了开发，使用的是大规模的真实世界数据集。通过广泛的实验，RedOne 维持了强大的一般能力，并在8项主要 SNS 任务中平均提高了14.02%，在 SNS 双语评估基准测试中提高了7.56%。此外，在线测试结果显示，与单任务微调基线模型相比，RedOne 将有害内容检测的曝光率降低了11.23%，提高了张贴查看后的点击页率14.95%。这些结果确立了 RedOne 作为一个在 SNS 中表现出色的领域特定 LLM 的地位，展示了其在各种任务上的卓越泛化能力和在实际场景中的广泛应用潜力。 

---
# Emergence of Hierarchical Emotion Organization in Large Language Models 

**Title (ZH)**: 大型语言模型中层级情绪组织的涌现 

**Authors**: Bo Zhao, Maya Okawa, Eric J. Bigelow, Rose Yu, Tomer Ullman, Ekdeep Singh Lubana, Hidenori Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2507.10599)  

**Abstract**: As large language models (LLMs) increasingly power conversational agents, understanding how they model users' emotional states is critical for ethical deployment. Inspired by emotion wheels -- a psychological framework that argues emotions organize hierarchically -- we analyze probabilistic dependencies between emotional states in model outputs. We find that LLMs naturally form hierarchical emotion trees that align with human psychological models, and larger models develop more complex hierarchies. We also uncover systematic biases in emotion recognition across socioeconomic personas, with compounding misclassifications for intersectional, underrepresented groups. Human studies reveal striking parallels, suggesting that LLMs internalize aspects of social perception. Beyond highlighting emergent emotional reasoning in LLMs, our results hint at the potential of using cognitively-grounded theories for developing better model evaluations. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）越来越多地驱动对话代理，了解它们如何建模用户的情感状态对于伦理应用至关重要。受情绪轮盘这一心理学框架的启发，该框架认为情绪是层次化的，我们分析了模型输出中情感状态的概率依赖性。我们发现，LLMs自然形成了与人类心理学模型相一致的层级情绪树，且更大的模型发展出更复杂的层级结构。我们还发现了跨社会经济人物的情绪识别系统性偏差，而交叉影响的边缘群体出现了累积的分类错误。人类研究揭示了令人震惊的相似之处，表明LLMs内化了社会知觉的某些方面。除了突显LLMs中涌现的情感推理之外，我们的结果暗示了使用认知驱动的理论来开发更好模型评估的潜力。 

---
# PLEX: Perturbation-free Local Explanations for LLM-Based Text Classification 

**Title (ZH)**: PLEX：基于LLM的文本分类的无扰动局部解释 

**Authors**: Yogachandran Rahulamathavan, Misbah Farooq, Varuna De Silva  

**Link**: [PDF](https://arxiv.org/pdf/2507.10596)  

**Abstract**: Large Language Models (LLMs) excel in text classification, but their complexity hinders interpretability, making it difficult to understand the reasoning behind their predictions. Explainable AI (XAI) methods like LIME and SHAP offer local explanations by identifying influential words, but they rely on computationally expensive perturbations. These methods typically generate thousands of perturbed sentences and perform inferences on each, incurring a substantial computational burden, especially with LLMs. To address this, we propose \underline{P}erturbation-free \underline{L}ocal \underline{Ex}planation (PLEX), a novel method that leverages the contextual embeddings extracted from the LLM and a ``Siamese network" style neural network trained to align with feature importance scores. This one-off training eliminates the need for subsequent perturbations, enabling efficient explanations for any new sentence. We demonstrate PLEX's effectiveness on four different classification tasks (sentiment, fake news, fake COVID-19 news and depression), showing more than 92\% agreement with LIME and SHAP. Our evaluation using a ``stress test" reveals that PLEX accurately identifies influential words, leading to a similar decline in classification accuracy as observed with LIME and SHAP when these words are removed. Notably, in some cases, PLEX demonstrates superior performance in capturing the impact of key features. PLEX dramatically accelerates explanation, reducing time and computational overhead by two and four orders of magnitude, respectively. This work offers a promising solution for explainable LLM-based text classification. 

**Abstract (ZH)**: Perturbation-free Local Explanation (PLEX) for Explainable Large Language Models-based Text Classification 

---
# ToolRegistry: A Protocol-Agnostic Tool Management Library for Function-Calling LLMs 

**Title (ZH)**: ToolRegistry：一种面向函数调用LLM的协议agnostic工具管理库 

**Authors**: Peng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.10593)  

**Abstract**: Large Language Model (LLM) applications are increasingly relying on external tools to extend their capabilities beyond text generation. However, current tool integration approaches suffer from fragmentation, protocol limitations, and implementation complexity, leading to substantial development overhead. This paper presents Toolregistry, a protocol-agnostic tool management library that simplifies tool registration, representation, execution, and lifecycle management via a unified interface. Our evaluation demonstrates that \toolregistry achieves 60-80% reduction in tool integration code, up to 3.1x performance improvements through concurrent execution, and 100% compatibility with OpenAI function calling standards. Real-world case studies show significant improvements in development efficiency and code maintainability across diverse integration scenarios. \toolregistry is open-source and available at this https URL, with comprehensive documentation at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLM）的应用越来越多地依赖外部工具来扩展其文本生成之外的能力。然而，当前的工具集成方法存在碎片化、协议限制和实现复杂性的问题，导致了大量的开发开销。本文介绍了Toolregistry，一种协议无关的工具管理库，通过统一接口简化了工具注册、表示、执行和生命周期管理。我们的评估显示，Toolregistry在工具集成代码上实现了60-80%的减少，并通过并行执行实现了高达3.1倍的性能提升，同时完全兼容OpenAI函数调用标准。实际案例研究显示，在各种集成场景中，Toolregistry显著提高了开发效率和代码可维护性。Toolregistry是开源的，并可在以下网址获取：this https URL，详细文档请参见以下网址：this https URL。 

---
# Repairing Language Model Pipelines by Meta Self-Refining Competing Constraints at Runtime 

**Title (ZH)**: 运行时元自完善竞争约束修正语言模型管道 

**Authors**: Mojtaba Eshghie  

**Link**: [PDF](https://arxiv.org/pdf/2507.10590)  

**Abstract**: Language Model (LM) pipelines can dynamically refine their outputs against programmatic constraints. However, their effectiveness collapses when faced with competing soft constraints, leading to inefficient backtracking loops where satisfying one constraint violates another. We introduce Meta Self-Refining, a framework that equips LM pipelines with a meta-corrective layer to repair these competitions at runtime/inference-time. Our approach monitors the pipeline's execution history to detect oscillatory failures. Upon detection, it invokes a meta-repairer LM that analyzes the holistic state of the backtracking attempts and synthesizes a strategic instruction to balance the competing requirements. This self-repair instruction guides the original LM out of a failing refining loop towards a successful output. Our results show Meta Self-Refining can successfully repair these loops, leading to more efficient LM programs. 

**Abstract (ZH)**: 语言模型（LM）管道可以动态地在其输出中应用程序化约束的修正。然而，当面对竞争性的软约束时，其效果会失效，导致无效的回溯循环，满足一个约束会违反另一个约束。我们引入了Meta Self-Refining框架，该框架为LM管道配备了元校正层，在运行时/推理时修复这些竞争。我们的方法监控管道的执行历史以检测振荡失败。检测到后，它会调用一个元修复器LM，分析回溯尝试的总体状态并合成一个战略指令以平衡竞争需求。该自我修复指令引导原始LM从失败的修正循环中走出，生成成功的输出。我们的结果表明，Meta Self-Refining能够成功修复这些循环，使LM程序更加高效。 

---
# Anthropomimetic Uncertainty: What Verbalized Uncertainty in Language Models is Missing 

**Title (ZH)**: 拟人类不确定性：语言模型中口头化不确定性的缺失要素 

**Authors**: Dennis Ulmer, Alexandra Lorson, Ivan Titov, Christian Hardmeier  

**Link**: [PDF](https://arxiv.org/pdf/2507.10587)  

**Abstract**: Human users increasingly rely on natural language interactions with large language models (LLMs) in order to receive help on a large variety of tasks and problems. However, the trustworthiness and perceived legitimacy of LLMs is undermined by the fact that their output is frequently stated in very confident terms, even when its accuracy is questionable. Therefore, there is a need to signal the confidence of the language model to a user in order to reap the benefits of human-machine collaboration and mitigate potential harms. Verbalized uncertainty is the expression of confidence with linguistic means, an approach that integrates perfectly into language-based interfaces. Nevertheless, most recent research in natural language processing (NLP) overlooks the nuances surrounding human uncertainty communication and the data biases that influence machine uncertainty communication. We argue for anthropomimetic uncertainty, meaning that intuitive and trustworthy uncertainty communication requires a degree of linguistic authenticity and personalization to the user, which could be achieved by emulating human communication. We present a thorough overview over the research in human uncertainty communication, survey ongoing research, and perform additional analyses to demonstrate so-far overlooked biases in verbalized uncertainty. We conclude by pointing out unique factors in human-machine communication of uncertainty and deconstruct anthropomimetic uncertainty into future research directions for NLP. 

**Abstract (ZH)**: 人类用户越来越多地依赖大型语言模型（LLMs）的自然语言交互以获得各种任务和问题的帮助，但模型输出经常以极其自信的语气表述，即使其准确性存疑，这损害了人们对LLMs的可信度和合法性感知。因此，需要向用户提供模型信心信号，以充分利用人机协作的好处并减轻潜在危害。口头化的不确定性是通过语言手段表达信心的方法，这一方法与基于语言的接口完美契合。然而，自然语言处理（NLP）领域的最新研究大多忽视了人类不确定性沟通的细微之处以及影响机器不确定性沟通的数据偏见。我们主张仿人类不确定性，即直观且可信的不确定性沟通需要一定程度的语言 authenticity和个性化，可以通过模仿人类沟通来实现。我们对人类不确定性沟通的研究进行了全面回顾，调查正在进行中的研究，并进行额外分析以展示口头化不确定性中的未被注意到的偏见。最后，我们指出了人机不确定性沟通中的独特因素，并将仿人类不确定性分解为未来NLP研究方向。 

---
# AutoRAG-LoRA: Hallucination-Triggered Knowledge Retuning via Lightweight Adapters 

**Title (ZH)**: AutoRAG-LoRA：由幻觉触发的知识轻量级适调 

**Authors**: Kaushik Dwivedi, Padmanabh Patanjali Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2507.10586)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable fluency across a range of natural language tasks, yet remain vulnerable to hallucinations - factual inaccuracies that undermine trust in real world deployment. We present AutoRAG-LoRA, a modular framework for Retrieval-Augmented Generation (RAG) that tackles hallucination in large language models through lightweight LoRA-based adapters and KL-regularized training. Our pipeline integrates automated prompt rewriting, hybrid retrieval, and low-rank adapter tuning to ground responses in retrieved evidence. A hallucination detection module, using both classifier-based and self-evaluation techniques, assigns confidence scores to generated outputs, triggering an optional feedback correction loop. This loop enforces factual alignment via contrastive KL loss and adapter fine tuning. We demonstrate that AutoRAG-LoRA significantly reduces the factual drift while preserving the efficiency and modularity of the model. 

**Abstract (ZH)**: Large Language Models (LLMs) 在自然语言任务中表现出色，但易产生幻觉——事实不准确的陈述，这削弱了实际应用中的可信度。我们提出了AutoRAG-LoRA，一种通过轻量级LoRA适配器和KL正则化训练来解决大型语言模型幻觉问题的模块化框架。该框架整合了自动提示重写、混合检索和低秩适配器调优，以使生成的回应基于检索到的证据。一个幻觉检测模块，结合分类器和自我评估技术，为生成的输出打分，并触发可选的反馈纠正循环。该循环通过对比KL损失和适配器微调来强制执行事实一致性。我们证明，AutoRAG-LoRA 显著减少了事实偏离，同时保持了模型的高效性和模块性。 

---
# A Taxonomy for Design and Evaluation of Prompt-Based Natural Language Explanations 

**Title (ZH)**: 基于提示的自然语言解释设计与评估分类框架 

**Authors**: Isar Nejadgholi, Mona Omidyeganeh, Marc-Antoine Drouin, Jonathan Boisvert  

**Link**: [PDF](https://arxiv.org/pdf/2507.10585)  

**Abstract**: Effective AI governance requires structured approaches for stakeholders to access and verify AI system behavior. With the rise of large language models, Natural Language Explanations (NLEs) are now key to articulating model behavior, which necessitates a focused examination of their characteristics and governance implications. We draw on Explainable AI (XAI) literature to create an updated XAI taxonomy, adapted to prompt-based NLEs, across three dimensions: (1) Context, including task, data, audience, and goals; (2) Generation and Presentation, covering generation methods, inputs, interactivity, outputs, and forms; and (3) Evaluation, focusing on content, presentation, and user-centered properties, as well as the setting of the evaluation. This taxonomy provides a framework for researchers, auditors, and policymakers to characterize, design, and enhance NLEs for transparent AI systems. 

**Abstract (ZH)**: 有效的AI治理需要结构化的途径以使各方能够访问和验证AI系统的行为。随着大规模语言模型的兴起，自然语言解释（NLEs）已成为说明模型行为的关键，这需要对它们的特性和治理影响进行聚焦研究。我们借鉴可解释AI（XAI）文献，创建了一个适应基于提示的NLE的更新版XAI分类框架，涵盖三个维度：（1）背景，包括任务、数据、受众和目标；（2）生成与呈现，涵盖生成方法、输入、交互性、输出和形式；以及（3）评估，侧重于内容、呈现、以用户为中心的特性，以及评估环境。该分类框架为研究人员、审计员和政策制定者提供了一个框架，用于表征、设计和优化透明AI系统的NLE。 

---
# ARPaCCino: An Agentic-RAG for Policy as Code Compliance 

**Title (ZH)**: ARPaCCino: 一种代理驱动的RAG政策即代码合规性助手 

**Authors**: Francesco Romeo, Luigi Arena, Francesco Blefari, Francesco Aurelio Pironti, Matteo Lupinacci, Angelo Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.10584)  

**Abstract**: Policy as Code (PaC) is a paradigm that encodes security and compliance policies into machine-readable formats, enabling automated enforcement in Infrastructure as Code (IaC) environments. However, its adoption is hindered by the complexity of policy languages and the risk of misconfigurations. In this work, we present ARPaCCino, an agentic system that combines Large Language Models (LLMs), Retrieval-Augmented-Generation (RAG), and tool-based validation to automate the generation and verification of PaC rules. Given natural language descriptions of the desired policies, ARPaCCino generates formal Rego rules, assesses IaC compliance, and iteratively refines the IaC configurations to ensure conformance. Thanks to its modular agentic architecture and integration with external tools and knowledge bases, ARPaCCino supports policy validation across a wide range of technologies, including niche or emerging IaC frameworks. Experimental evaluation involving a Terraform-based case study demonstrates ARPaCCino's effectiveness in generating syntactically and semantically correct policies, identifying non-compliant infrastructures, and applying corrective modifications, even when using smaller, open-weight LLMs. Our results highlight the potential of agentic RAG architectures to enhance the automation, reliability, and accessibility of PaC workflows. 

**Abstract (ZH)**: 基于代理的Policy as Code (PaC)的自动化生成与验证：结合大语言模型、检索增强生成和工具验证的ARPaCCino系统 

---
# An Offline Mobile Conversational Agent for Mental Health Support: Learning from Emotional Dialogues and Psychological Texts with Student-Centered Evaluation 

**Title (ZH)**: 面向心理健康的离线移动对话代理：基于学生中心评估的情感对话与心理文本学习 

**Authors**: Vimaleswar A, Prabhu Nandan Sahu, Nilesh Kumar Sahu, Haroon R Lone  

**Link**: [PDF](https://arxiv.org/pdf/2507.10580)  

**Abstract**: Mental health plays a crucial role in the overall well-being of an individual. In recent years, digital platforms have been increasingly used to expand mental health and emotional support. However, there are persistent challenges related to limited user accessibility, internet connectivity, and data privacy, which highlight the need for an offline, smartphone-based solution. To address these challenges, we propose EmoSApp (Emotional Support App): an entirely offline, smartphone-based conversational app designed for mental health and emotional support. The system leverages Large Language Models (LLMs), specifically fine-tuned, quantized and deployed using Torchtune and Executorch for resource-constrained devices, allowing all inferences to occur on the smartphone. To equip EmoSApp with robust domain expertise, we fine-tuned the LLaMA-3.2-1B-Instruct model on our custom curated ``Knowledge dataset'' of 14,582 mental-health QA pairs, along with the multi-turn conversational data.
Through qualitative human evaluation with the student population, we demonstrate that EmoSApp has the ability to respond coherently, empathetically, maintain interactive dialogue, and provide relevant suggestions to user's mental health problems. Additionally, quantitative evaluations on nine standard commonsense and reasoning benchmarks demonstrate the efficacy of our fine-tuned, quantized model in low-resource settings. By prioritizing on-device deployment and specialized domain adaptation, EmoSApp serves as a blueprint for future innovations in portable, secure, and highly tailored AI-driven mental health solutions. 

**Abstract (ZH)**: 情感健康在个体的整体福祉中扮演着关键角色。近年来，数字平台被越来越多地用于扩展心理健康和情感支持。然而，有限的用户访问性、互联网连接性和数据隐私等问题依然存在，突显了需要一种离线的智能手机基于解决方案的必要性。为应对这些挑战，我们提出EmoSApp（情感支持应用）：一种完全离线的，基于智能手机的对话应用，旨在提供情感支持和心理健康服务。该系统利用了大型语言模型（LLMs），特别是通过Torchtune和Executorch进行细调和部署，适用于资源受限的设备，使得所有推理都在智能手机上进行。为使EmoSApp具备强大的领域专业知识，我们在一个包含14,582个心理健康问答对的自定义知识数据集上对LLaMA-3.2-1B-Instruct模型进行了细调，并结合了多轮对话数据。

通过针对学生群体的定性人类评估，我们证明EmoSApp能够一贯地、富有同情心地响应，保持互动对话，并向用户的情感健康问题提供相关建议。此外，在九个标准常识和推理基准上的定量评估表明，我们的细调和量化模型在资源受限环境中具有有效性。通过优先考虑设备上部署和专门的领域适应，EmoSApp为未来便携、安全和高度定制的AI驱动心理健康解决方案提供了蓝图。 

---
# Findings of the BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-powered Tutors 

**Title (ZH)**: BEA 2025 共享任务关于人工智能辅导工具教学能力评估的成果 

**Authors**: Ekaterina Kochmar, Kaushal Kumar Maurya, Kseniia Petukhova, KV Aditya Srivatsa, Anaïs Tack, Justin Vasselli  

**Link**: [PDF](https://arxiv.org/pdf/2507.10579)  

**Abstract**: This shared task has aimed to assess pedagogical abilities of AI tutors powered by large language models (LLMs), focusing on evaluating the quality of tutor responses aimed at student's mistake remediation within educational dialogues. The task consisted of five tracks designed to automatically evaluate the AI tutor's performance across key dimensions of mistake identification, precise location of the mistake, providing guidance, and feedback actionability, grounded in learning science principles that define good and effective tutor responses, as well as the track focusing on detection of the tutor identity. The task attracted over 50 international teams across all tracks. The submitted models were evaluated against gold-standard human annotations, and the results, while promising, show that there is still significant room for improvement in this domain: the best results for the four pedagogical ability assessment tracks range between macro F1 scores of 58.34 (for providing guidance) and 71.81 (for mistake identification) on three-class problems, with the best F1 score in the tutor identification track reaching 96.98 on a 9-class task. In this paper, we overview the main findings of the shared task, discuss the approaches taken by the teams, and analyze their performance. All resources associated with this task are made publicly available to support future research in this critical domain. 

**Abstract (ZH)**: 这项共享任务旨在评估由大型语言模型（LLMs）驱动的人工智能辅导系统的教学能力，重点在于评估辅导系统在教育对话中纠正学生错误时响应质量。该任务包括五个赛道，旨在从错误识别、错误位置精确性、提供指导以及反馈执行力等方面自动评估人工智能辅导员的性能，这些维度基于学习科学原则，定义了良好的有效辅导响应标准，并且还包括辅导身份检测的赛道。来自全球的50多支团队参加了所有赛道的比赛。提交的模型接受了金标准人类注释的评估，结果虽然令人振奋，但仍显示在此领域还有很大的改进空间：四个人工智能辅导系统教学能力评估赛道的最佳结果分别涵盖三类问题中的宏F1分数为58.34（提供指导）和71.81（错误识别），而辅导身份检测赛道的最佳F1分数为96.98，涉及九类任务。本文概述了共享任务的主要发现，讨论了参赛团队所采用的方法，并分析了他们的表现。所有与该任务相关的资源均向公众开放，以支持该关键领域的未来研究。 

---
# Can Large Language Models Understand As Well As Apply Patent Regulations to Pass a Hands-On Patent Attorney Test? 

**Title (ZH)**: 大型语言模型在通过实际专利律师考试时，能否既理解和应用专利法规？ 

**Authors**: Bhakti Khera, Rezvan Alamian, Pascal A. Scherz, Stephan M. Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.10576)  

**Abstract**: The legal field already uses various large language models (LLMs) in actual applications, but their quantitative performance and reasons for it are underexplored. We evaluated several open-source and proprietary LLMs -- including GPT-series, Anthropic, Deepseek and Llama-3, variants -- on parts of the European Qualifying Examination (EQE) for future European Patent Attorneys. OpenAI o1 led with 0.82 accuracy and 0.81 F1 score, whereas (Amazon Web Services) AWS Llama 3.1 8B lagged at 0.50 accuracy, and a Python-deployed Llama 3.1 8B scored 0.55. The latter two are within the range of mere guessing for the two-answer forced-choice design. None of the evaluated models could have passed the examination fully, as accuracy never exceeded the average threshold of 0.90 required for professional-level standards -- also not models that are regularly promoted for their assumed beyond-PhD- and bar-admitted-lawyer-level performance. GPT-4o excelled at integrating text and graphics, while Claude 3 Opus often lost formatting coherence. Human patent experts evaluated the textual justifications and uncovered various critical shortcomings of each model. They valued clarity and legal rationale over the raw correctness of the answers, which revealed misalignment between automatic metrics and expert judgment. Model outputs were sensitive to modest temperature changes and prompt wording, which underscores the remaining necessity of expert oversight. Future work should target logical consistency, robust multimodality, and adaptive prompting to approach human-level patent proficiency. In summary, despite the outstanding performance of recent large models, the general public might overestimate their performance. The field has a long way to go to develop a virtual patent attorney. This paper wants to point out several specific limitations that need solutions. 

**Abstract (ZH)**: 现有的法律领域已经在实际应用中使用了各种大型语言模型（LLMs），但其量化性能及原因尚未充分探索。我们评估了几种开源和专有LLM——包括GPT系列、Anthropic、Deepseek和Llama-3变体——在欧洲资格考试（EQE）的部分试题上，以供未来欧洲专利代理人使用。OpenAI o1以0.82的准确率和0.81的F1分数领先，而(Amazon Web Services) AWS Llama 3.1 8B仅为0.50的准确率，一个部署在Python中的Llama 3.1 8B得分为0.55。后两种在两选项强制选择设计中仅略高于纯粹猜测的范围。没有任何评估模型能够全面通过考试，即使是对被认为超越博士和通过律师考试的模型也是如此，其准确率从未超过专业水平所需的平均阈值0.90。人类专利专家评估了文本解释，并发现了每个模型的各种关键不足。他们更重视清晰度和法律依据，而非答案的纯粹正确性，这揭示了自动指标与专家判断之间的不一致。模型输出对温度变化和提示措辞的变化非常敏感，这强调了专业监督的必要性。未来的工作应专注于逻辑一致性、稳健的多媒体能力和适应性提示，以接近人类水平的专利熟练度。总的来说，尽管最近的大规模模型表现出色，但公众可能对其性能存在过度估计。该领域还需走很长一段路以发展出虚拟专利代理人。本文旨在指出需要解决的若干具体限制。 

---
# NLP Meets the World: Toward Improving Conversations With the Public About Natural Language Processing Research 

**Title (ZH)**: NLP遇见世界：关于自然语言处理研究面向公众对话的改进之路 

**Authors**: Shomir Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2507.10559)  

**Abstract**: Recent developments in large language models (LLMs) have been accompanied by rapidly growing public interest in natural language processing (NLP). This attention is reflected by major news venues, which sometimes invite NLP researchers to share their knowledge and views with a wide audience. Recognizing the opportunities of the present, for both the research field and for individual researchers, this paper shares recommendations for communicating with a general audience about LLMs' capabilities and limitations. These recommendations cover three themes: vague terminology as an obstacle to public understanding, unreasonable expectations as obstacles to sustainable growth, and ethical failures as obstacles to continued support. Published NLP research and popular news coverage are cited to illustrate these themes with examples. The recommendations promote effective, transparent communication with the general public about NLP, in order to strengthen public understanding and encourage support for research. 

**Abstract (ZH)**: Recent developments in大型语言模型（LLMs）引发了公众对自然语言处理（NLP）的浓厚兴趣。这种关注在主流媒体中有所体现，有时会邀请NLP研究人员与广大读者分享他们的知识和观点。本论文旨在利用当前的机遇，针对研究领域和个人研究人员，分享关于沟通LLMs的能力和限制的建议。这些建议涵盖了三个主题：模糊术语是公众理解的障碍、不合理的期望是可持续增长的障碍、伦理失败是持续支持的障碍。文章通过引用已发表的NLP研究和流行新闻报道中的例子来阐述这些主题，旨在促进与公众的有效、透明沟通，以增强公众的理解并鼓励对研究的支持。 

---
