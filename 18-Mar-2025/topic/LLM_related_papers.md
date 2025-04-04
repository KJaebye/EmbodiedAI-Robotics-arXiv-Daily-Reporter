# ICCO: Learning an Instruction-conditioned Coordinator for Language-guided Task-aligned Multi-robot Control 

**Title (ZH)**: ICCO: 基于指令调节的多机器人任务对齐的语言引导控制协调器 

**Authors**: Yoshiki Yano, Kazuki Shibata, Maarten Kokshoorn, Takamitsu Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2503.12122)  

**Abstract**: Recent advances in Large Language Models (LLMs) have permitted the development of language-guided multi-robot systems, which allow robots to execute tasks based on natural language instructions. However, achieving effective coordination in distributed multi-agent environments remains challenging due to (1) misalignment between instructions and task requirements and (2) inconsistency in robot behaviors when they independently interpret ambiguous instructions. To address these challenges, we propose Instruction-Conditioned Coordinator (ICCO), a Multi-Agent Reinforcement Learning (MARL) framework designed to enhance coordination in language-guided multi-robot systems. ICCO consists of a Coordinator agent and multiple Local Agents, where the Coordinator generates Task-Aligned and Consistent Instructions (TACI) by integrating language instructions with environmental states, ensuring task alignment and behavioral consistency. The Coordinator and Local Agents are jointly trained to optimize a reward function that balances task efficiency and instruction following. A Consistency Enhancement Term is added to the learning objective to maximize mutual information between instructions and robot behaviors, further improving coordination. Simulation and real-world experiments validate the effectiveness of ICCO in achieving language-guided task-aligned multi-robot control. The demonstration can be found at this https URL. 

**Abstract (ZH)**: 近期大规模语言模型的进展使语言引导的多机器人系统的发展成为可能，这些系统允许机器人根据自然语言指令执行任务。然而，在分布式多智能体环境中实现有效的协调仍然具有挑战性，原因包括（1）指令与任务需求之间的不一致，以及（2）当机器人独立解释含糊的指令时出现的行为不一致性。为了解决这些挑战，我们提出了一种多智能体强化学习框架——指令条件协调器（ICCO），旨在增强语言引导的多机器人系统中的协调性。ICCO包括一个协调器智能体和多个本地智能体，协调器通过整合语言指令与环境状态生成任务对齐且行为一致的指令（TACI），确保任务对齐和行为一致性。协调器和本地智能体共同训练以优化平衡任务效率和指令遵守的奖励函数。学习目标中加入了一致性增强项，以最大限度地提高指令与机器人行为之间的互信息，进一步提高协调性。模拟和实际实验验证了ICCO在实现语言引导的任务对齐多机器人控制方面的有效性。相关演示可访问此网址。 

---
# MAP: Multi-user Personalization with Collaborative LLM-powered Agents 

**Title (ZH)**: MAP: 多用户个性化协作LLM代理 

**Authors**: Christine Lee, Jihye Choi, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12757)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) and LLM-powered agents in multi-user settings underscores the need for reliable, usable methods to accommodate diverse preferences and resolve conflicting directives. Drawing on conflict resolution theory, we introduce a user-centered workflow for multi-user personalization comprising three stages: Reflection, Analysis, and Feedback. We then present MAP -- a \textbf{M}ulti-\textbf{A}gent system for multi-user \textbf{P}ersonalization -- to operationalize this workflow. By delegating subtasks to specialized agents, MAP (1) retrieves and reflects on relevant user information, while enhancing reliability through agent-to-agent interactions, (2) provides detailed analysis for improved transparency and usability, and (3) integrates user feedback to iteratively refine results. Our user study findings (n=12) highlight MAP's effectiveness and usability for conflict resolution while emphasizing the importance of user involvement in resolution verification and failure management. This work highlights the potential of multi-agent systems to implement user-centered, multi-user personalization workflows and concludes by offering insights for personalization in multi-user contexts. 

**Abstract (ZH)**: 大规模语言模型及其驱动代理在多用户环境中的广泛应用强调了需要可靠且易于使用的多用户个性化方法来满足多样化的需求并解决冲突指令。基于冲突解决理论，我们提出了一种以用户为中心的多用户个性化工作流，包含三个阶段：反思、分析和反馈。随后，我们介绍了MAP——一个多代理系统，用于实现这种工作流。通过将子任务委派给专门的代理，MAP（1）检索和反思相关用户信息，并通过代理间的交互提高可靠性；（2）提供详细的分析以增强透明度和易用性；（3）整合用户反馈以迭代优化结果。我们的用户研究结果（n=12）表明，MAP在冲突解决方面有效且易用，并强调了用户参与解决验证和故障管理的重要性。这项工作强调了多代理系统在实现用户为中心的多用户个性化工作流方面的潜力，并提出了一些在多用户情境下进行个性化设计的见解。 

---
# Logic-RAG: Augmenting Large Multimodal Models with Visual-Spatial Knowledge for Road Scene Understanding 

**Title (ZH)**: 逻辑-RAG：增强大规模多模态模型的视觉空间知识以理解道路场景 

**Authors**: Imran Kabir, Md Alimoor Reza, Syed Billah  

**Link**: [PDF](https://arxiv.org/pdf/2503.12663)  

**Abstract**: Large multimodal models (LMMs) are increasingly integrated into autonomous driving systems for user interaction. However, their limitations in fine-grained spatial reasoning pose challenges for system interpretability and user trust. We introduce Logic-RAG, a novel Retrieval-Augmented Generation (RAG) framework that improves LMMs' spatial understanding in driving scenarios. Logic-RAG constructs a dynamic knowledge base (KB) about object-object relationships in first-order logic (FOL) using a perception module, a query-to-logic embedder, and a logical inference engine. We evaluated Logic-RAG on visual-spatial queries using both synthetic and real-world driving videos. When using popular LMMs (GPT-4V, Claude 3.5) as proxies for an autonomous driving system, these models achieved only 55% accuracy on synthetic driving scenes and under 75% on real-world driving scenes. Augmenting them with Logic-RAG increased their accuracies to over 80% and 90%, respectively. An ablation study showed that even without logical inference, the fact-based context constructed by Logic-RAG alone improved accuracy by 15%. Logic-RAG is extensible: it allows seamless replacement of individual components with improved versions and enables domain experts to compose new knowledge in both FOL and natural language. In sum, Logic-RAG addresses critical spatial reasoning deficiencies in LMMs for autonomous driving applications. Code and data are available at this https URL. 

**Abstract (ZH)**: 基于逻辑的检索增强生成（Logic-RAG）：提高自动驾驶场景中大模型的细粒度空间理解 

---
# Knowledge-Aware Iterative Retrieval for Multi-Agent Systems 

**Title (ZH)**: 知识驱动迭代检索在多Agent系统中的应用 

**Authors**: Seyoung Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.13275)  

**Abstract**: We introduce a novel large language model (LLM)-driven agent framework, which iteratively refines queries and filters contextual evidence by leveraging dynamically evolving knowledge. A defining feature of the system is its decoupling of external sources from an internal knowledge cache that is progressively updated to guide both query generation and evidence selection. This design mitigates bias-reinforcement loops and enables dynamic, trackable search exploration paths, thereby optimizing the trade-off between exploring diverse information and maintaining accuracy through autonomous agent decision-making. Our approach is evaluated on a broad range of open-domain question answering benchmarks, including multi-step tasks that mirror real-world scenarios where integrating information from multiple sources is critical, especially given the vulnerabilities of LLMs that lack explicit reasoning or planning capabilities. The results show that the proposed system not only outperforms single-step baselines regardless of task difficulty but also, compared to conventional iterative retrieval methods, demonstrates pronounced advantages in complex tasks through precise evidence-based reasoning and enhanced efficiency. The proposed system supports both competitive and collaborative sharing of updated context, enabling multi-agent extension. The benefits of multi-agent configurations become especially prominent as task difficulty increases. The number of convergence steps scales with task difficulty, suggesting cost-effective scalability. 

**Abstract (ZH)**: 一种基于large language model的迭代查询与动态知识驱动智能代理框架：多源信息整合下的性能优化与多智能体扩展 

---
# MAP: Evaluation and Multi-Agent Enhancement of Large Language Models for Inpatient Pathways 

**Title (ZH)**: MAP: 大型语言模型在住院路径中的评估与多agent增强 

**Authors**: Zhen Chen, Zhihao Peng, Xusheng Liang, Cheng Wang, Peigan Liang, Linsheng Zeng, Minjie Ju, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.13205)  

**Abstract**: Inpatient pathways demand complex clinical decision-making based on comprehensive patient information, posing critical challenges for clinicians. Despite advancements in large language models (LLMs) in medical applications, limited research focused on artificial intelligence (AI) inpatient pathways systems, due to the lack of large-scale inpatient datasets. Moreover, existing medical benchmarks typically concentrated on medical question-answering and examinations, ignoring the multifaceted nature of clinical decision-making in inpatient settings. To address these gaps, we first developed the Inpatient Pathway Decision Support (IPDS) benchmark from the MIMIC-IV database, encompassing 51,274 cases across nine triage departments and 17 major disease categories alongside 16 standardized treatment options. Then, we proposed the Multi-Agent Inpatient Pathways (MAP) framework to accomplish inpatient pathways with three clinical agents, including a triage agent managing the patient admission, a diagnosis agent serving as the primary decision maker at the department, and a treatment agent providing treatment plans. Additionally, our MAP framework includes a chief agent overseeing the inpatient pathways to guide and promote these three clinician agents. Extensive experiments showed our MAP improved the diagnosis accuracy by 25.10% compared to the state-of-the-art LLM HuatuoGPT2-13B. It is worth noting that our MAP demonstrated significant clinical compliance, outperforming three board-certified clinicians by 10%-12%, establishing a foundation for inpatient pathways systems. 

**Abstract (ZH)**: 基于综合患者信息的住院路径需求复杂临床决策，对临床医生构成关键挑战。尽管在医疗应用中大型语言模型取得了进展，但由于缺乏大规模住院患者数据集，有限的研究关注人工智能住院路径系统。此外，现有的医疗基准通常集中在医学问答和检查上，忽略了住院环境中临床决策的多维性质。为解决这些差距，我们首先从MIMIC-IV数据库中开发了住院路径决策支持（IPDS）基准，涵盖了九个 triage 部门和17个主要疾病类别，以及16种标准化治疗选项。然后，我们提出了多代理住院路径（MAP）框架，通过三个临床代理来实现住院路径，包括负责患者入院的分诊代理、作为部门主要决策者的诊断代理以及提供治疗计划的治疗代理。此外，我们的MAP框架还包括一个主管代理，监督住院路径以指导和促进这三个临床代理。大量实验表明，与最先进的大型语言模型HuatuoGPT2-13B相比，我们的MAP提高了25.10%的诊断准确性。值得注意的是，我们的MAP在临床合规性方面表现出显著优势，相比三个认证的临床医生性能高出10%-12%，为住院路径系统奠定了基础。 

---
# Are LLMs (Really) Ideological? An IRT-based Analysis and Alignment Tool for Perceived Socio-Economic Bias in LLMs 

**Title (ZH)**: 大语言模型（真的）有意识形态倾向吗？基于IRT的分析与对齐工具，用于评估大语言模型中的社会经济偏见 

**Authors**: Jasmin Wachter, Michael Radloff, Maja Smolej, Katharina Kinder-Kurlanda  

**Link**: [PDF](https://arxiv.org/pdf/2503.13149)  

**Abstract**: We introduce an Item Response Theory (IRT)-based framework to detect and quantify socioeconomic bias in large language models (LLMs) without relying on subjective human judgments. Unlike traditional methods, IRT accounts for item difficulty, improving ideological bias estimation. We fine-tune two LLM families (Meta-LLaMa 3.2-1B-Instruct and Chat- GPT 3.5) to represent distinct ideological positions and introduce a two-stage approach: (1) modeling response avoidance and (2) estimating perceived bias in answered responses. Our results show that off-the-shelf LLMs often avoid ideological engagement rather than exhibit bias, challenging prior claims of partisanship. This empirically validated framework enhances AI alignment research and promotes fairer AI governance. 

**Abstract (ZH)**: 基于项目反应理论的框架：在无需依赖主观人类判断的情况下检测和量化大型语言模型中的社会经济偏见 

---
# R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization 

**Title (ZH)**: R1-VL：通过逐步组相对策略优化学习多模态大型语言模型推理 

**Authors**: Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12937)  

**Abstract**: Recent studies generally enhance MLLMs' reasoning capabilities via supervised fine-tuning on high-quality chain-of-thought reasoning data, which often leads models to merely imitate successful reasoning paths without understanding what the wrong reasoning paths are. In this work, we aim to enhance the MLLMs' reasoning ability beyond passively imitating positive reasoning paths. To this end, we design Step-wise Group Relative Policy Optimization (StepGRPO), a new online reinforcement learning framework that enables MLLMs to self-improve reasoning ability via simple, effective and dense step-wise rewarding. Specifically, StepGRPO introduces two novel rule-based reasoning rewards: Step-wise Reasoning Accuracy Reward (StepRAR) and Step-wise Reasoning Validity Reward (StepRVR). StepRAR rewards the reasoning paths that contain necessary intermediate reasoning steps via a soft key-step matching technique, while StepRAR rewards reasoning paths that follow a well-structured and logically consistent reasoning process through a reasoning completeness and logic evaluation strategy. With the proposed StepGRPO, we introduce R1-VL, a series of MLLMs with outstanding capabilities in step-by-step reasoning. Extensive experiments over 8 benchmarks demonstrate the superiority of our methods. 

**Abstract (ZH)**: Recent studies generally enhance MLLMs' reasoning capabilities via supervised fine-tuning on high-quality chain-of-thought reasoning data, which often leads models to merely imitate successful reasoning paths without understanding what the wrong reasoning paths are. In this work, we aim to enhance the MLLMs' reasoning ability beyond passively imitating positive reasoning paths. To this end, we design Step-wise Group Relative Policy Optimization (StepGRPO), a new online reinforcement learning framework that enables MLLMs to self-improve reasoning ability via simple, effective, and dense step-wise rewarding. Specifically, StepGRPO introduces two novel rule-based reasoning rewards: Step-wise Reasoning Accuracy Reward (StepRAR) and Step-wise Reasoning Validity Reward (StepRVR). StepRAR rewards the reasoning paths that contain necessary intermediate reasoning steps via a soft key-step matching technique, while StepRVR rewards reasoning paths that follow a well-structured and logically consistent reasoning process through a reasoning completeness and logic evaluation strategy. With the proposed StepGRPO, we introduce R1-VL, a series of MLLMs with outstanding capabilities in step-by-step reasoning. Extensive experiments over 8 benchmarks demonstrate the superiority of our methods。 

---
# Can Reasoning Models Reason about Hardware? An Agentic HLS Perspective 

**Title (ZH)**: 基于代理的硬件合成视角：推理模型能否 reasoning 关于硬件？ 

**Authors**: Luca Collini, Andrew Hennessee, Ramesh Karri, Siddharth Garg  

**Link**: [PDF](https://arxiv.org/pdf/2503.12721)  

**Abstract**: Recent Large Language Models (LLMs) such as OpenAI o3-mini and DeepSeek-R1 use enhanced reasoning through Chain-of-Thought (CoT). Their potential in hardware design, which relies on expert-driven iterative optimization, remains unexplored. This paper investigates whether reasoning LLMs can address challenges in High-Level Synthesis (HLS) design space exploration and optimization. During HLS, engineers manually define pragmas/directives to balance performance and resource constraints. We propose an LLM-based optimization agentic framework that automatically restructures code, inserts pragmas, and identifies optimal design points via feedback from HLs tools and access to integer-linear programming (ILP) solvers. Experiments compare reasoning models against conventional LLMs on benchmarks using success rate, efficiency, and design quality (area/latency) metrics, and provide the first-ever glimpse into the CoTs produced by a powerful open-source reasoning model like DeepSeek-R1. 

**Abstract (ZH)**: 近期的大规模语言模型（LLMs）如OpenAI o3-mini和DeepSeek-R1通过链式推理（Chain-of-Thought，CoT）增强了推理能力。尽管它们在硬件设计中的潜力，特别是依赖于专家驱动的迭代优化的设计空间探索和优化，尚未被充分发掘。本文探讨了推理LLMs是否能够应对高阶综合（HLS）设计空间探索和优化中的挑战。在HLS过程中，工程师手动定义pragma/指南来平衡性能和资源约束。我们提出了一种基于LLM的优化代理框架，该框架能够自动重构代码、插入pragma并利用来自HLS工具的反馈及整数线性规划（ILP）求解器访问能力来识别最优设计点。实验在基准测试上将推理模型与传统LLMs进行比较，使用成功率、效率和设计质量（面积/延迟）等指标，并提供了关于强大开源推理模型DeepSeek-R1产生的CoT的首次见解。 

---
# VeriLA: A Human-Centered Evaluation Framework for Interpretable Verification of LLM Agent Failures 

**Title (ZH)**: VeriLA: 一种以人为本的可解释LLM代理故障评估框架 

**Authors**: Yoo Yeon Sung, Hannah Kim, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12651)  

**Abstract**: AI practitioners increasingly use large language model (LLM) agents in compound AI systems to solve complex reasoning tasks, these agent executions often fail to meet human standards, leading to errors that compromise the system's overall performance. Addressing these failures through human intervention is challenging due to the agents' opaque reasoning processes, misalignment with human expectations, the complexity of agent dependencies, and the high cost of manual inspection. This paper thus introduces a human-centered evaluation framework for Verifying LLM Agent failures (VeriLA), which systematically assesses agent failures to reduce human effort and make these agent failures interpretable to humans. The framework first defines clear expectations of each agent by curating human-designed agent criteria. Then, it develops a human-aligned agent verifier module, trained with human gold standards, to assess each agent's execution output. This approach enables granular evaluation of each agent's performance by revealing failures from a human standard, offering clear guidelines for revision, and reducing human cognitive load. Our case study results show that VeriLA is both interpretable and efficient in helping practitioners interact more effectively with the system. By upholding accountability in human-agent collaboration, VeriLA paves the way for more trustworthy and human-aligned compound AI systems. 

**Abstract (ZH)**: AI从业者 increasingly 使用大规模语言模型 (LLM) 剂体制作复杂混合AI系统以解决复杂推理任务，这些剂体制作的执行Often常 不符合人类标准，导致影响系统整体性能的错误。通过人工干预解决这些问题具有挑战性，因为剂体制作具有不透明的推理过程、与人类期望不一致、剂体制作的复杂依赖关系以及手动审查的高成本。本文因此介绍了以人为中心的评估框架VeriLA，该框架系统地评估剂体制作的失败，以减少人工努力并使这些失败对人类可解释。该框架首先通过收集设计的人类标准的剂体制作标准来定义每个剂体制作的明确期望。然后，它开发了一个与人类标准对齐的剂体制作验证模块，通过人类金标准训练，以评估每个剂体制作的执行输出。该方法通过从人类标准揭示失败，实现每个剂体制作性能的逐个评估，并提供明确的修订指南，从而减轻人类的认知负担。我们的案例研究结果表明，VeriLA在帮助从业者更有效地与系统交互方面既可解释又有效。通过维护人类-剂体制作合作中的问责制，VeriLA为更具可信度和人类导向的混合AI系统铺平了道路。 

---
# MPBench: A Comprehensive Multimodal Reasoning Benchmark for Process Errors Identification 

**Title (ZH)**: MPBench: 一个全面的多模态推理基准，用于过程错误识别 

**Authors**: Zhaopan Xu, Pengfei Zhou, Jiaxin Ai, Wangbo Zhao, Kai Wang, Xiaojiang Peng, Wenqi Shao, Hongxun Yao, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12505)  

**Abstract**: Reasoning is an essential capacity for large language models (LLMs) to address complex tasks, where the identification of process errors is vital for improving this ability. Recently, process-level reward models (PRMs) were proposed to provide step-wise rewards that facilitate reinforcement learning and data production during training and guide LLMs toward correct steps during inference, thereby improving reasoning accuracy. However, existing benchmarks of PRMs are text-based and focus on error detection, neglecting other scenarios like reasoning search. To address this gap, we introduce MPBench, a comprehensive, multi-task, multimodal benchmark designed to systematically assess the effectiveness of PRMs in diverse scenarios. MPBench employs three evaluation paradigms, each targeting a specific role of PRMs in the reasoning process: (1) Step Correctness, which assesses the correctness of each intermediate reasoning step; (2) Answer Aggregation, which aggregates multiple solutions and selects the best one; and (3) Reasoning Process Search, which guides the search for optimal reasoning steps during inference. Through these paradigms, MPBench makes comprehensive evaluations and provides insights into the development of multimodal PRMs. 

**Abstract (ZH)**: 大语言模型(LLMs)处理复杂任务时需要推理能力，其中过程错误的识别对于提高这一能力至关重要。近日，提出了过程级奖励模型(PRMs)，以提供步骤级奖励，促进训练期间的强化学习和数据生成，并在推理时引导LLMs采取正确的步骤，从而提高推理准确性。然而，现有的PRM基准大多是基于文本，主要集中在错误检测上，忽略了如推理搜索等其他场景。为弥补这一点，我们引入了MPBench，这是一个全面的多任务、多模态基准，旨在系统评估PRMs在各种场景中的有效性。MPBench采用三种评估范式，分别针对PRMs在推理过程中的特定作用：(1) 步骤正确性，评估每个中间推理步骤的正确性；(2) 答案汇总，汇总多种解决方案并选择最佳者；(3) 推理过程搜索，在推理时引导对最优推理步骤的搜索。通过这些范式，MPBench进行了全面评估并为多模态PRM的发展提供了洞见。 

---
# A Survey on the Optimization of Large Language Model-based Agents 

**Title (ZH)**: 大型语言模型代理的优化研究 

**Authors**: Shangheng Du, Jiabao Zhao, Jinxin Shi, Zhentao Xie, Xin Jiang, Yanhong Bai, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2503.12434)  

**Abstract**: With the rapid development of Large Language Models (LLMs), LLM-based agents have been widely adopted in various fields, becoming essential for autonomous decision-making and interactive tasks. However, current work typically relies on prompt design or fine-tuning strategies applied to vanilla LLMs, which often leads to limited effectiveness or suboptimal performance in complex agent-related environments. Although LLM optimization techniques can improve model performance across many general tasks, they lack specialized optimization towards critical agent functionalities such as long-term planning, dynamic environmental interaction, and complex decision-making. Although numerous recent studies have explored various strategies to optimize LLM-based agents for complex agent tasks, a systematic review summarizing and comparing these methods from a holistic perspective is still lacking. In this survey, we provide a comprehensive review of LLM-based agent optimization approaches, categorizing them into parameter-driven and parameter-free methods. We first focus on parameter-driven optimization, covering fine-tuning-based optimization, reinforcement learning-based optimization, and hybrid strategies, analyzing key aspects such as trajectory data construction, fine-tuning techniques, reward function design, and optimization algorithms. Additionally, we briefly discuss parameter-free strategies that optimize agent behavior through prompt engineering and external knowledge retrieval. Finally, we summarize the datasets and benchmarks used for evaluation and tuning, review key applications of LLM-based agents, and discuss major challenges and promising future directions. Our repository for related references is available at this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的快速发展，基于LLM的代理已经在多个领域广泛应用，成为自主决策和交互任务中不可或缺的部分。然而，当前的工作通常依赖于对原始LLM进行提示设计或微调策略，这在复杂的代理相关环境中往往导致有限的效果或次优性能。尽管LLM优化技术可以在众多通用任务中提高模型性能，但它们缺乏针对诸如长期规划、动态环境交互和复杂决策等关键代理功能的专业优化。尽管最近有许多研究探索了各种策略来优化基于LLM的代理以适应复杂的代理任务，但仍缺乏从整体视角对其进行总结和比较的系统综述。在这篇综述中，我们提供了一个全面的基于LLM的代理优化方法综述，将其分为参数驱动和无参数方法两类。我们首先关注参数驱动的优化，包括基于微调的优化、基于强化学习的优化和混合策略，分析了轨迹数据构建、微调技术、奖励函数设计和优化算法等方面的关键问题。此外，我们简要讨论了通过提示工程和外部知识检索来优化代理行为的无参数策略。最后，我们总结了用于评估和调整的常用数据集和基准，并回顾了基于LLM的代理的关键应用，讨论了主要挑战和有前途的未来方向。相关参考文献的仓库可在以下链接访问：this https URL。 

---
# SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially? 

**Title (ZH)**: SPIN-Bench: 计划策略性和社会推理能力，大型语言模型表现如何？ 

**Authors**: Jianzhu Yao, Kevin Wang, Ryan Hsieh, Haisu Zhou, Tianqing Zou, Zerui Cheng, Zhangyang Wang, Pramod Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2503.12349)  

**Abstract**: Reasoning and strategic behavior in \emph{social interactions} is a hallmark of intelligence. This form of reasoning is significantly more sophisticated than isolated planning or reasoning tasks in static settings (e.g., math problem solving). In this paper, we present \textit{Strategic Planning, Interaction, and Negotiation} (\textbf{SPIN-Bench}), a new multi-domain evaluation designed to measure the intelligence of \emph{strategic planning} and \emph{social reasoning}. While many existing benchmarks focus on narrow planning or single-agent reasoning, SPIN-Bench combines classical PDDL tasks, competitive board games, cooperative card games, and multi-agent negotiation scenarios in one unified framework. The framework includes both a benchmark as well as an arena to simulate and evaluate the variety of social settings to test reasoning and strategic behavior of AI agents. We formulate the benchmark SPIN-Bench by systematically varying action spaces, state complexity, and the number of interacting agents to simulate a variety of social settings where success depends on not only methodical and step-wise decision making, but also \emph{conceptual inference} of other (adversarial or cooperative) participants. Our experiments reveal that while contemporary LLMs handle \emph{basic fact retrieval} and \emph{short-range planning} reasonably well, they encounter significant performance bottlenecks in tasks requiring \emph{deep multi-hop reasoning} over large state spaces and \emph{socially adept} coordination under uncertainty. We envision SPIN-Bench as a catalyst for future research on robust multi-agent planning, social reasoning, and human--AI teaming. 

**Abstract (ZH)**: 战略规划、互动与谈判：智能评估基准（SPIN-Bench） 

---
# Automating the loop in traffic incident management on highway 

**Title (ZH)**: 在高速公路交通事件管理中自动化循环过程 

**Authors**: Matteo Cercola, Nicola Gatti, Pedro Huertas Leyva, Benedetto Carambia, Simone Formentin  

**Link**: [PDF](https://arxiv.org/pdf/2503.12085)  

**Abstract**: Effective traffic incident management is essential for ensuring safety, minimizing congestion, and reducing response times in emergency situations. Traditional highway incident management relies heavily on radio room operators, who must make rapid, informed decisions in high-stakes environments. This paper proposes an innovative solution to support and enhance these decisions by integrating Large Language Models (LLMs) into a decision-support system for traffic incident management. We introduce two approaches: (1) an LLM + Optimization hybrid that leverages both the flexibility of natural language interaction and the robustness of optimization techniques, and (2) a Full LLM approach that autonomously generates decisions using only LLM capabilities. We tested our solutions using historical event data from Autostrade per l'Italia. Experimental results indicate that while both approaches show promise, the LLM + Optimization solution demonstrates superior reliability, making it particularly suited to critical applications where consistency and accuracy are paramount. This research highlights the potential for LLMs to transform highway incident management by enabling accessible, data-driven decision-making support. 

**Abstract (ZH)**: 有效的交通事件管理对于确保安全、减少拥堵和降低应急反应时间至关重要。传统的高速公路事件管理高度依赖于电台室操作员，他们在高风险环境中必须迅速做出明智的决策。本文提出了一种创新解决方案，通过将大型语言模型（LLMs）集成到交通事件管理决策支持系统中来支持和增强这些决策。我们介绍了两种方法：（1）LLM +优化的混合方法，利用自然语言交互的灵活性和优化技术的稳健性，以及（2）全LLM方法，仅利用LLM的能力自主生成决策。我们使用Autostrade per l'Italia的历史事件数据测试了我们的解决方案。实验结果显示，虽然两种方法都显示出潜力，但LLM +优化的方法在可靠性方面表现出色，使其特别适合那些一致性和准确性至关重要的关键应用。本文强调了LLMs在通过实现可访问的数据驱动决策支持来变革高速公路事件管理方面的潜力。 

---
# SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning 

**Title (ZH)**: SagaLLM：多智能体LLM规划中的上下文管理、验证和事务保证 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11951)  

**Abstract**: Recent LLM-based agent frameworks have demonstrated impressive capabilities in task delegation and workflow orchestration, but face significant challenges in maintaining context awareness and ensuring planning consistency. This paper presents SagaLLM, a structured multi-agent framework that addresses four fundamental limitations in current LLM approaches: inadequate self-validation, context narrowing, lacking transaction properties, and insufficient inter-agent coordination. By implementing specialized context management agents and validation protocols, SagaLLM preserves critical constraints and state information throughout complex planning processes, enabling robust and consistent decision-making even during disruptions. We evaluate our approach using selected problems from the REALM benchmark, focusing on sequential and reactive planning scenarios that challenge both context retention and adaptive reasoning. Our experiments with state-of-the-art LLMs, Claude 3.7, DeepSeek R1, GPT-4o, and GPT-o1, demonstrate that while these models exhibit impressive reasoning capabilities, they struggle with maintaining global constraint awareness during complex planning tasks, particularly when adapting to unexpected changes. In contrast, the distributed cognitive architecture of SagaLLM shows significant improvements in planning consistency, constraint enforcement, and adaptation to disruptions in various scenarios. 

**Abstract (ZH)**: Recent LLM-Based Agent Frameworks Have Demonstrated Impressive Capabilities in Task Delegation and Workflow Orchestration but Face Significant Challenges in Maintaining Context Awareness and Ensuring Planning Consistency: This Paper Presents SagaLLM, a Structured Multi-Agent Framework That Addresses Four Fundamental Limitations in Current LLM Approaches 

---
# Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation 

**Title (ZH)**: 监控推理模型的不当行为及促进混淆的风险 

**Authors**: Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, David Farhi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11926)  

**Abstract**: Mitigating reward hacking--where AI systems misbehave due to flaws or misspecifications in their learning objectives--remains a key challenge in constructing capable and aligned models. We show that we can monitor a frontier reasoning model, such as OpenAI o3-mini, for reward hacking in agentic coding environments by using another LLM that observes the model's chain-of-thought (CoT) reasoning. CoT monitoring can be far more effective than monitoring agent actions and outputs alone, and we further found that a LLM weaker than o3-mini, namely GPT-4o, can effectively monitor a stronger model. Because CoT monitors can be effective at detecting exploits, it is natural to ask whether those exploits can be suppressed by incorporating a CoT monitor directly into the agent's training objective. While we show that integrating CoT monitors into the reinforcement learning reward can indeed produce more capable and more aligned agents in the low optimization regime, we find that with too much optimization, agents learn obfuscated reward hacking, hiding their intent within the CoT while still exhibiting a significant rate of reward hacking. Because it is difficult to tell when CoTs have become obfuscated, it may be necessary to pay a monitorability tax by not applying strong optimization pressures directly to the chain-of-thought, ensuring that CoTs remain monitorable and useful for detecting misaligned behavior. 

**Abstract (ZH)**: 缓解奖励作弊——即AI系统因学习目标中的缺陷或疏漏而导致的不当行为——仍然是构建具备能力和对齐的模型的关键挑战。我们展示了可以通过另一个大规模语言模型（LLM）监测模型的逐步推理（CoT）过程，来监控诸如OpenAI o3-mini这类前沿推理模型在代理编码环境中的奖励作弊行为。CoT监测比单独监测代理行为和输出更为有效，我们进一步发现，一个比o3-mini弱的LLM，即GPT-4o，也能有效监测更强的模型。由于CoT监测能够有效检测作弊行为，自然会引发一个疑问，即是否可以通过将CoT监测直接纳入代理的训练目标中来抑制这些作弊行为。尽管我们展示了在低优化条件下，将CoT监测集成到强化学习奖励中确实能够产出更具能力和更对齐的代理，但我们发现，在过度优化的情况下，代理会学会更加隐蔽的奖励作弊行为，将其意图隐藏在CoT中，但仍体现出显著的奖励作弊率。由于难以判断CoT何时变得隐蔽，可能需要支付可监测性税，即不直接对CoT施加强大的优化压力，以确保CoT保持可监测性和检测不对齐行为的有效性。 

---
# Visualizing Thought: Conceptual Diagrams Enable Robust Planning in LMMs 

**Title (ZH)**: 可视化思维：概念图谱使大规模语言模型实现稳健规划 

**Authors**: Nasim Borazjanizadeh, Roei Herzig, Eduard Oks, Trevor Darrell, Rogerio Feris, Leonid Karlinsky  

**Link**: [PDF](https://arxiv.org/pdf/2503.11790)  

**Abstract**: Human reasoning relies on constructing and manipulating mental models-simplified internal representations of situations that we use to understand and solve problems. Conceptual diagrams (for example, sketches drawn by humans to aid reasoning) externalize these mental models, abstracting irrelevant details to efficiently capture relational and spatial information. In contrast, Large Language Models (LLMs) and Large Multimodal Models (LMMs) predominantly reason through textual representations, limiting their effectiveness in complex multi-step combinatorial and planning tasks. In this paper, we propose a zero-shot fully automatic framework that enables LMMs to reason through multiple chains of self-generated intermediate conceptual diagrams, significantly enhancing their combinatorial planning capabilities. Our approach does not require any human initialization beyond a natural language description of the task. It integrates both textual and diagrammatic reasoning within an optimized graph-of-thought inference framework, enhanced by beam search and depth-wise backtracking. Evaluated on multiple challenging PDDL planning domains, our method substantially improves GPT-4o's performance (for example, from 35.5% to 90.2% in Blocksworld). On more difficult planning domains with solution depths up to 40, our approach outperforms even the o1-preview reasoning model (for example, over 13% improvement in Parking). These results highlight the value of conceptual diagrams as a complementary reasoning medium in LMMs. 

**Abstract (ZH)**: 人类推理依赖于构建和操作心理模型——这些是简化内部表示的情景，我们用来理解和解决问题。概念图（例如，人类绘制的辅助推理的草图）外部化这些心理模型，抽象掉不相关的细节，以有效地捕捉关系和空间信息。相比之下，大型语言模型（LLMs）和大型多模态模型（LMMs）主要通过文本表示来进行推理，这限制了它们在复杂多步组合和规划任务中的有效性。在本文中，我们提出了一种零样本全自动框架，使LMMs能够通过多条自动生成的中间概念图进行推理，显著增强了它们的组合规划能力。我们的方法不需要任何人类初始化，只需自然语言描述任务。该方法在优化的思想图推断框架中结合了文本和图示推理，并通过束搜索和深度回溯得到了增强。在多个具有挑战性的PDDL规划领域进行评估后，我们的方法明显提高了GPT-4o的表现（例如，在Blocksworld中，从35.5%提高到90.2%）。在更具挑战性的规划领域，解决方案深度高达40时，我们的方法甚至超过了o1-preview推理模型（例如，在Parking中，提高了超过13%）。这些结果突显了概念图作为LMMs中补充推理媒介的价值。 

---
# Toward a method for LLM-enabled Indoor Navigation 

**Title (ZH)**: 面向LLM赋能的室内导航方法研究 

**Authors**: Alberto Coffrini, Mohammad Amin Zadenoori, Paolo Barsocchi, Francesco Furfari, Antonino Crivello, Alessio Ferrari  

**Link**: [PDF](https://arxiv.org/pdf/2503.11702)  

**Abstract**: Indoor navigation presents unique challenges due to complex layouts, lack of GPS signals, and accessibility concerns. Existing solutions often struggle with real-time adaptability and user-specific needs. In this work, we explore the potential of a Large Language Model (LLM), i.e., ChatGPT, to generate natural, context-aware navigation instructions from indoor map images. We design and evaluate test cases across different real-world environments, analyzing the effectiveness of LLMs in interpreting spatial layouts, handling user constraints, and planning efficient routes. Our findings demonstrate the potential of LLMs for supporting personalized indoor navigation, with an average of 52% correct indications and a maximum of 62%. The results do not appear to depend on the complexity of the layout or the complexity of the expected path, but rather on the number of points of interest and the abundance of visual information, which negatively affect the performance. 

**Abstract (ZH)**: 室内导航由于复杂的布局、缺少GPS信号和可达性问题而面临独特挑战。现有的解决方案往往难以实现实时适应性和满足用户特定需求。在本研究中，我们探索了大型语言模型（LLM），即ChatGPT，生成基于室内地图图像的自然、情境感知导航指令的潜力。我们设计并评估了不同现实环境下的测试案例，分析了LLM在解释空间布局、处理用户约束以及规划高效路线方面的有效性。我们的研究结果表明，LLM有支持个性化室内导航的潜力，准确指示的平均比例为52%，最高达到62%。结果似乎与布局的复杂性或预期路径的复杂性无关，而是与兴趣点的数量以及视觉信息的丰富性有关，这些因素对性能有负面影响。 

---
# An LLM-Based Approach for Insight Generation in Data Analysis 

**Title (ZH)**: 基于LLM的方法在数据分析中的洞察生成 

**Authors**: Alberto Sánchez Pérez, Alaa Boukhary, Paolo Papotti, Luis Castejón Lozano, Adam Elwood  

**Link**: [PDF](https://arxiv.org/pdf/2503.11664)  

**Abstract**: Generating insightful and actionable information from databases is critical in data analysis. This paper introduces a novel approach using Large Language Models (LLMs) to automatically generate textual insights. Given a multi-table database as input, our method leverages LLMs to produce concise, text-based insights that reflect interesting patterns in the tables. Our framework includes a Hypothesis Generator to formulate domain-relevant questions, a Query Agent to answer such questions by generating SQL queries against a database, and a Summarization module to verbalize the insights. The insights are evaluated for both correctness and subjective insightfulness using a hybrid model of human judgment and automated metrics. Experimental results on public and enterprise databases demonstrate that our approach generates more insightful insights than other approaches while maintaining correctness. 

**Abstract (ZH)**: 从数据库中生成启发性和可操作性的信息对于数据分析至关重要。本文介绍了一种使用大型语言模型（LLMs）的新型方法，以自动生成文本洞察。给定一个多表数据库作为输入，我们的方法利用LLMs生成简洁的、基于文本的洞察，反映表格中的有趣模式。我们的框架包括一个假设生成器，用于提出领域相关的问题；一个查询代理，通过生成针对数据库的SQL查询来回答这些问题；以及一个总结模块，用于口头表达洞察。这些洞察使用人类判断和自动化度量相结合的混合模型进行正确性和主观洞察性的评估。实验结果表明，与其它方法相比，我们的方法在保持正确性的同时生成了更具洞察性的洞察。 

---
# MetaScale: Test-Time Scaling with Evolving Meta-Thoughts 

**Title (ZH)**: MetaScale: 测试时动态扩容与进化元思维 

**Authors**: Qin Liu, Wenxuan Zhou, Nan Xu, James Y. Huang, Fei Wang, Sheng Zhang, Hoifung Poon, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13447)  

**Abstract**: One critical challenge for large language models (LLMs) for making complex reasoning is their reliance on matching reasoning patterns from training data, instead of proactively selecting the most appropriate cognitive strategy to solve a given task. Existing approaches impose fixed cognitive structures that enhance performance in specific tasks but lack adaptability across diverse scenarios. To address this limitation, we introduce METASCALE, a test-time scaling framework based on meta-thoughts -- adaptive thinking strategies tailored to each task. METASCALE initializes a pool of candidate meta-thoughts, then iteratively selects and evaluates them using a multi-armed bandit algorithm with upper confidence bound selection, guided by a reward model. To further enhance adaptability, a genetic algorithm evolves high-reward meta-thoughts, refining and extending the strategy pool over time. By dynamically proposing and optimizing meta-thoughts at inference time, METASCALE improves both accuracy and generalization across a wide range of tasks. Experimental results demonstrate that MetaScale consistently outperforms standard inference approaches, achieving an 11% performance gain in win rate on Arena-Hard for GPT-4o, surpassing o1-mini by 0.9% under style control. Notably, METASCALE scales more effectively with increasing sampling budgets and produces more structured, expert-level responses. 

**Abstract (ZH)**: 针对大型语言模型进行复杂推理的一项關鍵挑戰是其依賴於匹配訓練數據中的推理模式，而不是積極選擇最適合的認知策略來解決特定任務。現有方法推動固定認知結構，這些結構在特定任務上提高了性能，但缺乏在多樣場景中的適應性。為此，我們引入了METASCALE，一種基於元思維的測試時擴展框架——針對每個任務量身定制的自适应思考策略。METASCALE初始化候選元思維池，然後通過多臂bandit算法結合上 verschied擇優選擇，並由獎勵模型指導，迭代選擇和評估這些思維策略。為進一步提升適應性，我們使用遺傳算法進化高獎勵的元思維，隨時間推移調整和擴充策略池。通過動態在推理時提出和優化元思維，METASCALE在多種任務上提高了準確性和泛化能力。實驗結果表明，METASCALE一致優於標準推理方法，在GPT-4o的Arena-Hard評比中獲勝率提高了11%，在風格控制下超越o1-mini 0.9%。值得注意的是，METASCALE隨著采樣預算的增加更具擴展性，並生成更具結構性、專家級別的回應。 

---
# Faithfulness of LLM Self-Explanations for Commonsense Tasks: Larger Is Better, and Instruction-Tuning Allows Trade-Offs but Not Pareto Dominance 

**Title (ZH)**: 大型语言模型自我解释的忠实性对于常识任务：规模越大越好，指令调优允许权衡但不允许帕累托占优 

**Authors**: Noah Y. Siegel, Nicolas Heess, Maria Perez-Ortiz, Oana-Maria Camburu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13445)  

**Abstract**: As large language models (LLMs) become increasingly capable, ensuring that their self-generated explanations are faithful to their internal decision-making process is critical for safety and oversight. In this work, we conduct a comprehensive counterfactual faithfulness analysis across 62 models from 8 families, encompassing both pretrained and instruction-tuned variants and significantly extending prior studies of counterfactual tests. We introduce phi-CCT, a simplified variant of the Correlational Counterfactual Test, which avoids the need for token probabilities while explaining most of the variance of the original test. Our findings reveal clear scaling trends: larger models are consistently more faithful on our metrics. However, when comparing instruction-tuned and human-imitated explanations, we find that observed differences in faithfulness can often be attributed to explanation verbosity, leading to shifts along the true-positive/false-positive Pareto frontier. While instruction-tuning and prompting can influence this trade-off, we find limited evidence that they fundamentally expand the frontier of explanatory faithfulness beyond what is achievable with pretrained models of comparable size. Our analysis highlights the nuanced relationship between instruction-tuning, verbosity, and the faithful representation of model decision processes. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的能力不断增强，确保其自动生成的解释忠实于其内部决策过程对于安全性和监督至关重要。在本研究中，我们对62个模型（涵盖8大家族，包括预训练和指令调优变体）进行了全面的反事实忠实性分析，显著扩展了先前的反事实测试研究。我们引入了phi-CCT，这是一种简化版的相关反事实测试，它可以避免使用令牌概率来解释原始测试的主要方差。我们的研究发现显示了清晰的扩展趋势：更大的模型在我们的指标上始终更加忠实。然而，在比较指令调优和人类模仿的解释时，我们发现观察到的忠实性差异通常可以归因于解释的冗长性，导致在真正 positives/假阳性帕累托前沿上发生转变。虽然指令调优和提示可能会影响这种权衡，但我们发现有限的证据表明它们能够在预训练模型可比规模的情况下根本上扩大解释忠实性的范围。我们的分析突显了指令调优、冗长性与模型决策过程忠实表现之间的复杂关系。 

---
# xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference 

**Title (ZH)**: xLSTM 7B: 一种快速而高效的递归大语言模型 

**Authors**: Maximilian Beck, Korbinian Pöppel, Phillip Lippe, Richard Kurle, Patrick M. Blies, Günter Klambauer, Sebastian Böck, Sepp Hochreiter  

**Link**: [PDF](https://arxiv.org/pdf/2503.13427)  

**Abstract**: Recent breakthroughs in solving reasoning, math and coding problems with Large Language Models (LLMs) have been enabled by investing substantial computation budgets at inference time. Therefore, inference speed is one of the most critical properties of LLM architectures, and there is a growing need for LLMs that are efficient and fast at inference. Recently, LLMs built on the xLSTM architecture have emerged as a powerful alternative to Transformers, offering linear compute scaling with sequence length and constant memory usage, both highly desirable properties for efficient inference. However, such xLSTM-based LLMs have yet to be scaled to larger models and assessed and compared with respect to inference speed and efficiency. In this work, we introduce xLSTM 7B, a 7-billion-parameter LLM that combines xLSTM's architectural benefits with targeted optimizations for fast and efficient inference. Our experiments demonstrate that xLSTM 7B achieves performance on downstream tasks comparable to other similar-sized LLMs, while providing significantly faster inference speeds and greater efficiency compared to Llama- and Mamba-based LLMs. These results establish xLSTM 7B as the fastest and most efficient 7B LLM, offering a solution for tasks that require large amounts of test-time computation. Our work highlights xLSTM's potential as a foundational architecture for methods building on heavy use of LLM inference. Our model weights, model code and training code are open-source. 

**Abstract (ZH)**: Recent突破性进展：大规模语言模型在推理、数学和编码问题上的应用得益于推理阶段大量计算资源的投资。因此，推理速度是大规模语言模型架构中最关键的属性之一，提高模型效率和加快推理速度的需求日益增长。最近，基于xLSTM架构的语言模型作为一种强大的替代方案出现了，它们具备随着序列长度线性扩展计算量和常数内存使用量的特点，这些都是高效推理所高度渴望的属性。然而，这样的基于xLSTM的语言模型尚未被扩展到更大的模型，并根据推理速度和效率进行了评估和比较。在本文中，我们介绍了参数量为7亿的xLSTM 7B，它结合了xLSTM架构的优势，并针对快速高效推理进行了目标优化。实验结果表明，xLSTM 7B在下游任务上的性能与其他相似规模的语言模型相当，同时提供了比Llama-和Mamba为基础的语言模型更快的推理速度和更高的效率。这些结果确立了xLSTM 7B为最快的7B语言模型，并为需要大量测试时间计算的任务提供了解决方案。我们的工作突显了xLSTM作为依赖大量语言模型推理的方法的基础架构的潜力。我们的模型权重、模型代码和训练代码均为开源。 

---
# DLPO: Towards a Robust, Efficient, and Generalizable Prompt Optimization Framework from a Deep-Learning Perspective 

**Title (ZH)**: DLPO: 从深度学习视角 towards 健壮、高效和通用的提示优化框架 

**Authors**: Dengyun Peng, Yuhang Zhou, Qiguang Chen, Jinhao Liu, Jingjing Chen, Libo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.13413)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, largely driven by well-designed prompts. However, crafting and selecting such prompts often requires considerable human effort, significantly limiting its scalability. To mitigate this, recent studies have explored automated prompt optimization as a promising solution. Despite these efforts, existing methods still face critical challenges in robustness, efficiency, and generalization. To systematically address these challenges, we first conduct an empirical analysis to identify the limitations of current reflection-based prompt optimization paradigm. Building on these insights, we propose 7 innovative approaches inspired by traditional deep learning paradigms for prompt optimization (DLPO), seamlessly integrating these concepts into text-based gradient optimization. Through these advancements, we progressively tackle the aforementioned challenges and validate our methods through extensive experimentation. We hope our study not only provides valuable guidance for future research but also offers a comprehensive understanding of the challenges and potential solutions in prompt optimization. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在多样任务中取得了显著成功，很大程度上得益于精心设计的提示。然而，设计和选择这些提示往往需要大量的手工努力，明显限制了其可扩展性。为解决这一问题，最近的研究探索了自动化提示优化作为一种有前景的解决方案。尽管取得了这些进展，现有方法在鲁棒性、效率和泛化能力方面仍然面临关键挑战。为了系统性地应对这些挑战，我们首先进行了实证分析以识别当前基于反思的提示优化范式的局限性。基于这些见解，我们提出了7种受到传统深度学习范式启发的创新方法，将这些概念无缝集成到基于文本的梯度优化中。通过这些进步，我们逐步解决了上述挑战，并通过广泛的实验验证了我们的方法。我们希望本研究不仅能为未来的研究提供有价值的指导，还能全面理解提示优化中面临的挑战及其潜在解决方案。我们的代码可在以下链接获取：this https URL。 

---
# Using the Tools of Cognitive Science to Understand Large Language Models at Different Levels of Analysis 

**Title (ZH)**: 用认知科学的工具从不同分析层理解大规模语言模型 

**Authors**: Alexander Ku, Declan Campbell, Xuechunzi Bai, Jiayi Geng, Ryan Liu, Raja Marjieh, R. Thomas McCoy, Andrew Nam, Ilia Sucholutsky, Veniamin Veselovsky, Liyi Zhang, Jian-Qiao Zhu, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2503.13401)  

**Abstract**: Modern artificial intelligence systems, such as large language models, are increasingly powerful but also increasingly hard to understand. Recognizing this problem as analogous to the historical difficulties in understanding the human mind, we argue that methods developed in cognitive science can be useful for understanding large language models. We propose a framework for applying these methods based on Marr's three levels of analysis. By revisiting established cognitive science techniques relevant to each level and illustrating their potential to yield insights into the behavior and internal organization of large language models, we aim to provide a toolkit for making sense of these new kinds of minds. 

**Abstract (ZH)**: 现代人工智能系统，如大型语言模型，既越来越强大也愈加难以理解。认识到这一问题类似于历史上理解人类心智的困难，我们认为可以借鉴认知科学中的方法来理解大型语言模型。我们提出一种基于 Marr 分析的三个层次框架，通过回顾与每个层次相关的已确立的认知科学技术，并阐述它们对揭示大型语言模型行为和内部组织的潜在洞察力，旨在提供一套工具箱，帮助理解这些新类型的“心智”。 

---
# Cream of the Crop: Harvesting Rich, Scalable and Transferable Multi-Modal Data for Instruction Fine-Tuning 

**Title (ZH)**: 优中选优：收获丰富的、可扩展且可迁移的多模态数据以供指令微调 

**Authors**: Mengyao Lyu, Yan Li, Huasong Zhong, Wenhao Yang, Hui Chen, Jungong Han, Guiguang Ding, Zhenheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13383)  

**Abstract**: The hypothesis that pretrained large language models (LLMs) necessitate only minimal supervision during the fine-tuning (SFT) stage (Zhou et al., 2024) has been substantiated by recent advancements in data curation and selection research. However, their stability and generalizability are compromised due to the vulnerability to experimental setups and validation protocols, falling short of surpassing random sampling (Diddee & Ippolito, 2024; Xia et al., 2024b). Built upon LLMs, multi-modal LLMs (MLLMs), combined with the sheer token volume and heightened heterogeneity of data sources, amplify both the significance and complexity of data selection.
To harvest multi-modal instructional data in a robust and efficient manner, we re-define the granularity of the quality metric by decomposing it into 14 vision-language-related capabilities, and introduce multi-modal rich scorers to evaluate the capabilities of each data candidate. To promote diversity, in light of the inherent objective of the alignment stage, we take interaction style as diversity indicator and use a multi-modal rich styler to identify data instruction patterns. In doing so, our multi-modal rich scorers and styler (mmSSR) guarantee that high-scoring information is conveyed to users in diversified forms. Free from embedding-based clustering or greedy sampling, mmSSR efficiently scales to millions of data with varying budget constraints, supports customization for general or specific capability acquisition, and facilitates training-free generalization to new domains for curation. Across 10+ experimental settings, validated by 14 multi-modal benchmarks, we demonstrate consistent improvements over random sampling, baseline strategies and state-of-the-art selection methods, achieving 99.1% of full performance with only 30% of the 2.6M data. 

**Abstract (ZH)**: 预训练大语言模型在微调阶段仅需最小监督的新颖性及其挑战与对策：基于多模态数据的选择与评估 

---
# Reliable and Efficient Amortized Model-based Evaluation 

**Title (ZH)**: 可靠的高效模型化评价方法 

**Authors**: Sang Truong, Yuheng Tu, Percy Liang, Bo Li, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2503.13335)  

**Abstract**: Comprehensive evaluations of language models (LM) during both development and deployment phases are necessary because these models possess numerous capabilities (e.g., mathematical reasoning, legal support, or medical diagnostic) as well as safety risks (e.g., racial bias, toxicity, or misinformation). The average score across a wide range of benchmarks provides a signal that helps guide the use of these LMs in practice. Currently, holistic evaluations are costly due to the large volume of benchmark questions, making frequent evaluations impractical. A popular attempt to lower the cost is to compute the average score on a subset of the benchmark. This approach, unfortunately, often renders an unreliable measure of LM performance because the average score is often confounded with the difficulty of the questions in the benchmark subset. Item response theory (IRT) was designed to address this challenge, providing a reliable measurement by careful controlling for question difficulty. Unfortunately, question difficulty is expensive to estimate. Facing this challenge, we train a model that predicts question difficulty from its content, enabling a reliable measurement at a fraction of the cost. In addition, we leverage this difficulty predictor to further improve the evaluation efficiency through training a question generator given a difficulty level. This question generator is essential in adaptive testing, where, instead of using a random subset of the benchmark questions, informative questions are adaptively chosen based on the current estimation of LLM performance. Experiments on 22 common natural language benchmarks and 172 LMs show that this approach is more reliable and efficient compared to current common practice. 

**Abstract (ZH)**: 全面评估开发和部署阶段语言模型的必要性及其挑战与解决方案 

---
# LEAVS: An LLM-based Labeler for Abdominal CT Supervision 

**Title (ZH)**: LEAVS: 一种基于LLM的腹部CT标注器 

**Authors**: Ricardo Bigolin Lanfredi, Yan Zhuang, Mark Finkelstein, Praveen Thoppey Srinivasan Balamuralikrishna, Luke Krembs, Brandon Khoury, Arthi Reddy, Pritam Mukherjee, Neil M. Rofsky, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2503.13330)  

**Abstract**: Extracting structured labels from radiology reports has been employed to create vision models to simultaneously detect several types of abnormalities. However, existing works focus mainly on the chest region. Few works have been investigated on abdominal radiology reports due to more complex anatomy and a wider range of pathologies in the abdomen. We propose LEAVS (Large language model Extractor for Abdominal Vision Supervision). This labeler can annotate the certainty of presence and the urgency of seven types of abnormalities for nine abdominal organs on CT radiology reports. To ensure broad coverage, we chose abnormalities that encompass most of the finding types from CT reports. Our approach employs a specialized chain-of-thought prompting strategy for a locally-run LLM using sentence extraction and multiple-choice questions in a tree-based decision system. We demonstrate that the LLM can extract several abnormality types across abdominal organs with an average F1 score of 0.89, significantly outperforming competing labelers and humans. Additionally, we show that extraction of urgency labels achieved performance comparable to human annotations. Finally, we demonstrate that the abnormality labels contain valuable information for training a single vision model that classifies several organs as normal or abnormal. We release our code and structured annotations for a public CT dataset containing over 1,000 CT volumes. 

**Abstract (ZH)**: 基于大型语言模型的腹部影像监督标注器LEAVS：从腹部放射报告中提取结构化标签 

---
# Computation Mechanism Behind LLM Position Generalization 

**Title (ZH)**: LLM位置泛化的计算机制 

**Authors**: Chi Han, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.13305)  

**Abstract**: Most written natural languages are composed of sequences of words and sentences. Similar to humans, large language models (LLMs) exhibit flexibility in handling textual positions - a phenomenon we term position generalization. They can understand texts with position perturbations and generalize to longer texts than those encountered during training with the latest techniques. These phenomena suggest that LLMs handle positions tolerantly, but how LLMs computationally process positional relevance remains largely unexplored. This work connects the linguistic phenomenon with LLMs' computational mechanisms. We show how LLMs enforce certain computational mechanisms for the aforementioned tolerance in position perturbations. Despite the complex design of the self-attention mechanism, this work reveals that LLMs learn a counterintuitive disentanglement of attention logits. Their values show a 0.959 linear correlation with an approximation of the arithmetic sum of positional relevance and semantic importance. Furthermore, we identify a prevalent pattern in intermediate features, which we prove theoretically enables this effect. The pattern, which is different from how randomly initialized parameters would behave, suggests that it is a learned behavior rather than a natural result of the model architecture. Based on these findings, we provide computational explanations and criteria for LLMs' position flexibilities. This work takes a pioneering step in linking position generalization with modern LLMs' internal mechanisms. 

**Abstract (ZH)**: 大型自然语言由单词和句子序列组成。类似人类，大规模语言模型（LLMs）在处理文本位置方面表现出灵活性——我们称其为位置泛化现象。它们能够理解位置发生扰动的文本，并且能够泛化到比训练中遇到的更长的文本，最新的技术使得这一现象成为可能。这些现象表明LLMs在处理位置时具有容忍性，但LLMs是如何在计算上处理位置相关性的，依然 largely unexplored。这项工作将语言现象与LLMs的计算机制联系起来。我们展示了LLMs如何通过某些计算机制来实现上述位置扰动的容忍性。尽管自我注意机制的设计复杂，但这项工作揭示了LLMs学习了一种直觉相反的注意力logits的分离。其值与位置相关性和语义重要性的算术和的近似值之间存在0.959的线性相关性。此外，我们识别出中间特征的一个普遍模式，我们证明这种模式理论上使这一效果得以实现。这种模式与随机初始化参数的行为不同，表明这是一种学习行为而非模型架构的自然结果。基于这些发现，我们提供了LLMs位置灵活性的计算解释和标准。这项工作在将位置泛化与现代LLMs的内部机制联系起来方面迈出了先驱性的一步。 

---
# A Survey on Transformer Context Extension: Approaches and Evaluation 

**Title (ZH)**: 变压器上下文扩展综述：方法与评估 

**Authors**: Yijun Liu, Jinzheng Yu, Yang Xu, Zhongyang Li, Qingfu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13299)  

**Abstract**: Large language models (LLMs) based on Transformer have been widely applied in the filed of natural language processing (NLP), demonstrating strong performance, particularly in handling short text tasks. However, when it comes to long context scenarios, the performance of LLMs degrades due to some challenges. To alleviate this phenomenon, there is a number of work proposed recently. In this survey, we first list the challenges of applying pre-trained LLMs to process long contexts. Then systematically review the approaches related to long context and propose our taxonomy categorizing them into four main types: positional encoding, context compression, retrieval augmented, and attention pattern. In addition to the approaches, we focus on the evaluation of long context, organizing relevant data, tasks, and metrics based on existing long context benchmarks. Finally, we summarize unresolved issues in the long context domain and put forward our views on future developments. 

**Abstract (ZH)**: 基于Transformer的大语言模型在自然语言处理中的应用：长上下文挑战及解决方案综述 

---
# LLM-Match: An Open-Sourced Patient Matching Model Based on Large Language Models and Retrieval-Augmented Generation 

**Title (ZH)**: LLM-Match：一种基于大规模语言模型和检索增强生成的开源患者匹配模型 

**Authors**: Xiaodi Li, Shaika Chowdhury, Chung Il Wi, Maria Vassilaki, Ken Liu, Terence T Sio, Owen Garrick, Young J Juhn, James R Cerhan, Cui Tao, Nansu Zong  

**Link**: [PDF](https://arxiv.org/pdf/2503.13281)  

**Abstract**: Patient matching is the process of linking patients to appropriate clinical trials by accurately identifying and matching their medical records with trial eligibility criteria. We propose LLM-Match, a novel framework for patient matching leveraging fine-tuned open-source large language models. Our approach consists of four key components. First, a retrieval-augmented generation (RAG) module extracts relevant patient context from a vast pool of electronic health records (EHRs). Second, a prompt generation module constructs input prompts by integrating trial eligibility criteria (both inclusion and exclusion criteria), patient context, and system instructions. Third, a fine-tuning module with a classification head optimizes the model parameters using structured prompts and ground-truth labels. Fourth, an evaluation module assesses the fine-tuned model's performance on the testing datasets. We evaluated LLM-Match on four open datasets, n2c2, SIGIR, TREC 2021, and TREC 2022, using open-source models, comparing it against TrialGPT, Zero-Shot, and GPT-4-based closed models. LLM-Match outperformed all baselines. 

**Abstract (ZH)**: 患者匹配是通过准确识别和匹配患者的医疗记录与临床试验资格标准，将患者链接到合适的临床试验的过程。我们提出了一种名为LLM-Match的新框架，该框架利用微调的开源大规模语言模型进行患者匹配。该方法包括四个关键组成部分。首先，检索增强生成（RAG）模块从庞大的电子健康记录（EHRs）池中提取相关患者上下文。其次，提示生成模块通过整合试验资格标准（包括纳入标准和排除标准）、患者上下文和系统指令构建输入提示。第三，具有分类头的微调模块使用结构化提示和真实标签优化模型参数。第四，评估模块在测试数据集上评估微调模型的性能。我们在n2c2、SIGIR、TREC 2021和TREC 2022四个开源数据集上使用开源模型评估了LLM-Match，并将其与TrialGPT、零样本和基于GPT-4的封闭模型进行了比较。LLM-Match在所有基线中表现最佳。 

---
# Goal2Story: A Multi-Agent Fleet based on Privately Enabled sLLMs for Impacting Mapping on Requirements Elicitation 

**Title (ZH)**: Goal2Story：基于私有增强sLLMs的多代理车队以影响需求提取的映射 

**Authors**: Xinkai Zou, Yan Liu, Xiongbo Shi, Chen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13279)  

**Abstract**: As requirements drift with rapid iterations, agile development becomes the dominant paradigm. Goal-driven Requirements Elicitation (RE) is a pivotal yet challenging task in agile project development due to its heavy tangling with adaptive planning and efficient collaboration. Recently, AI agents have shown promising ability in supporting requirements analysis by saving significant time and effort for stakeholders. However, current research mainly focuses on functional RE, and research works have not been reported bridging the long journey from goal to user stories. Moreover, considering the cost of LLM facilities and the need for data and idea protection, privately hosted small-sized LLM should be further utilized in RE. To address these challenges, we propose Goal2Story, a multi-agent fleet that adopts the Impact Mapping (IM) framework while merely using cost-effective sLLMs for goal-driven RE. Moreover, we introduce a StorySeek dataset that contains over 1,000 user stories (USs) with corresponding goals and project context information, as well as the semi-automatic dataset construction method. For evaluation, we proposed two metrics: Factuality Hit Rate (FHR) to measure consistency between the generated USs with the dataset and Quality And Consistency Evaluation (QuACE) to evaluate the quality of the generated USs. Experimental results demonstrate that Goal2Story outperforms the baseline performance of the Super-Agent adopting powerful LLMs, while also showcasing the performance improvements in key metrics brought by CoT and Agent Profile to Goal2Story, as well as its exploration in identifying latent needs. 

**Abstract (ZH)**: 基于目标到用户故事的多Agent框架：低成本小规模LLM在敏捷开发中驱动需求分析 

---
# Can Language Models Follow Multiple Turns of Entangled Instructions? 

**Title (ZH)**: 语言模型能跟随缠绕指令的多轮指示吗？ 

**Authors**: Chi Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.13222)  

**Abstract**: Despite significant achievements in improving the instruction-following capabilities of large language models (LLMs), the ability to process multiple potentially entangled or conflicting instructions remains a considerable challenge. Real-world scenarios often require consistency across multiple instructions over time, such as secret privacy, personal preferences, and prioritization, which demand sophisticated abilities to integrate multiple turns and carefully balance competing objectives when instructions intersect or conflict. This work presents a systematic investigation of LLMs' capabilities in handling multiple turns of instructions, covering three levels of difficulty: (1) retrieving information from instructions, (2) tracking and reasoning across turns, and (3) resolving conflicts among instructions. We construct MultiTurnInstruct with around 1.1K high-quality multi-turn conversations through the human-in-the-loop approach and result in nine capability categories, including statics and dynamics, reasoning, and multitasking. Our finding reveals an intriguing trade-off between different capabilities. While GPT models demonstrate superior memorization, they show reduced effectiveness in privacy-protection tasks requiring selective information withholding. Larger models exhibit stronger reasoning capabilities but still struggle with resolving conflicting instructions. Importantly, these performance gaps cannot be attributed solely to information loss, as models demonstrate strong BLEU scores on memorization tasks but their attention mechanisms fail to integrate multiple related instructions effectively. These findings highlight critical areas for improvement in complex real-world tasks involving multi-turn instructions. 

**Abstract (ZH)**: 尽管在提高大型语言模型（LLMs）的指令遵循能力方面取得了显著进展，但在处理多个潜在交织或冲突的指令方面仍然存在显著挑战。现实世界的情景往往要求随着时间的推移在多个指令之间保持一致性，如保密隐私、个人偏好和优先级，这要求模型具备集成多个回合并仔细平衡竞争目标的高级能力。本研究系统地探讨了LLMs在处理多轮指令方面的能力，涵盖了三个难度级别：（1）从指令中检索信息，（2）跟踪和在各轮之间推理，（3）解决指令之间的冲突。我们通过闭环的人工干预方法构建了包含约1100个多轮高质量对话的MultiTurnInstruct数据集，并划分了九个能力类别，包括静态和动态特征、推理和多任务处理。我们的发现揭示了不同能力之间有趣的权衡关系。虽然GPT模型在记忆方面表现优异，但在需要选择性信息保留的隐私保护任务中效果降低。更大规模的模型展示了更强的推理能力，但仍难以解决冲突指令。重要的是，这些性能差距不能仅归因于信息丢失，因为模型在记忆任务上表现出强劲的BLEU分数，但它们的注意力机制难以有效整合多个相关指令。这些发现突显了在涉及多轮指令的复杂现实任务中改进的关键领域。 

---
# Improving Complex Reasoning with Dynamic Prompt Corruption: A soft prompt Optimization Approach 

**Title (ZH)**: 使用动态提示腐化改进复杂推理：一种软提示优化方法 

**Authors**: Sinan Fan, Liang Xie, Chen Shen, Ge Teng, Xiaosong Yuan, Xiaofeng Zhang, Chenxi Huang, Wenxiao Wang, Xiaofei He, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.13208)  

**Abstract**: Prompt-tuning (PT) for large language models (LLMs) can facilitate the performance on various conventional NLP tasks with significantly fewer trainable parameters. However, our investigation reveals that PT provides limited improvement and may even degrade the primitive performance of LLMs on complex reasoning tasks. Such a phenomenon suggests that soft prompts can positively impact certain instances while negatively affecting others, particularly during the later phases of reasoning. To address these challenges, We first identify an information accumulation within the soft prompts. Through detailed analysis, we demonstrate that this phenomenon is often accompanied by erroneous information flow patterns in the deeper layers of the model, which ultimately lead to incorrect reasoning outcomes. we propose a novel method called \textbf{D}ynamic \textbf{P}rompt \textbf{C}orruption (DPC) to take better advantage of soft prompts in complex reasoning tasks, which dynamically adjusts the influence of soft prompts based on their impact on the reasoning process. Specifically, DPC consists of two stages: Dynamic Trigger and Dynamic Corruption. First, Dynamic Trigger measures the impact of soft prompts, identifying whether beneficial or detrimental. Then, Dynamic Corruption mitigates the negative effects of soft prompts by selectively masking key tokens that interfere with the reasoning process. We validate the proposed approach through extensive experiments on various LLMs and reasoning tasks, including GSM8K, MATH, and AQuA. Experimental results demonstrate that DPC can consistently enhance the performance of PT, achieving 4\%-8\% accuracy gains compared to vanilla prompt tuning, highlighting the effectiveness of our approach and its potential to enhance complex reasoning in LLMs. 

**Abstract (ZH)**: 动态提示损坏（DPC）：用于复杂推理任务的软提示优化 

---
# Lifting the Veil on Visual Information Flow in MLLMs: Unlocking Pathways to Faster Inference 

**Title (ZH)**: 揭开视觉信息在MLLMs中流动的面纱：解锁更快推理的路径 

**Authors**: Hao Yin, Guangzong Si, Zilei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13108)  

**Abstract**: Multimodal large language models (MLLMs) improve performance on vision-language tasks by integrating visual features from pre-trained vision encoders into large language models (LLMs). However, how MLLMs process and utilize visual information remains unclear. In this paper, a shift in the dominant flow of visual information is uncovered: (1) in shallow layers, strong interactions are observed between image tokens and instruction tokens, where most visual information is injected into instruction tokens to form cross-modal semantic representations; (2) in deeper layers, image tokens primarily interact with each other, aggregating the remaining visual information to optimize semantic representations within visual modality. Based on these insights, we propose Hierarchical Modality-Aware Pruning (HiMAP), a plug-and-play inference acceleration method that dynamically prunes image tokens at specific layers, reducing computational costs by approximately 65% without sacrificing performance. Our findings offer a new understanding of visual information processing in MLLMs and provide a state-of-the-art solution for efficient inference. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）通过将预训练视觉编码器的视觉特征集成到大规模语言模型（LLMs）中，提高了在视觉语言任务上的性能。然而，MLLMs如何处理和利用视觉信息仍不清楚。本文揭示了视觉信息主导流动的转变：（1）在浅层层中，观察到图像令牌与指令令牌之间存在强烈交互，大部分视觉信息注入到指令令牌中以形成跨模态语义表示；（2）在深层层中，图像令牌主要与其他图像令牌交互，聚集剩余的视觉信息以优化视觉模态内的语义表示。基于这些洞察，我们提出了层次化的模态感知剪枝（HiMAP），这是一种即插即用的推理加速方法，在特定层动态剪枝图像令牌，减少约65%的计算成本而不牺牲性能。我们的发现为MLLMs中的视觉信息处理提供了新的理解，并提供了最先进的高效推理解决方案。 

---
# ClusComp: A Simple Paradigm for Model Compression and Efficient Finetuning 

**Title (ZH)**: ClusComp: 一种简单的模型压缩和高效微调范式 

**Authors**: Baohao Liao, Christian Herold, Seyyed Hadi Hashemi, Stefan Vasilev, Shahram Khadivi, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2503.13089)  

**Abstract**: As large language models (LLMs) scale, model compression is crucial for edge deployment and accessibility. Weight-only quantization reduces model size but suffers from performance degradation at lower bit widths. Moreover, standard finetuning is incompatible with quantized models, and alternative methods often fall short of full finetuning. In this paper, we propose ClusComp, a simple yet effective compression paradigm that clusters weight matrices into codebooks and finetunes them block-by-block. ClusComp (1) achieves superior performance in 2-4 bit quantization, (2) pushes compression to 1-bit while outperforming ultra-low-bit methods with minimal finetuning, and (3) enables efficient finetuning, even surpassing existing quantization-based approaches and rivaling full FP16 finetuning. Notably, ClusComp supports compression and finetuning of 70B LLMs on a single A6000-48GB GPU. 

**Abstract (ZH)**: 大规模语言模型（LLMs）扩展时，模型压缩对于边缘部署和可访问性至关重要。权重仅为量化可以减少模型大小，但在较低位宽时会遭受性能下降。此外，标准微调与量化模型不兼容，而替代方法往往无法达到完整微调的效果。在此论文中，我们提出了一种简单而有效的压缩 paradigma，即将权重矩阵聚类成代码本并逐块微调。ClusComp (1) 在 2-4 位量化中实现了优越的性能，(2) 将压缩推至 1 位，并通过最少的微调超越超低位量化方法，(3) 使微调更加高效，甚至超越现有的基于量化的方法，接近全 FP16 微调。值得注意的是，ClusComp 可在单块 A6000-48GB GPU 上对 70B LLM 进行压缩和微调。 

---
# A Framework to Assess Multilingual Vulnerabilities of LLMs 

**Title (ZH)**: 评估多语言模型脆弱性的框架 

**Authors**: Likai Tang, Niruth Bogahawatta, Yasod Ginige, Jiarui Xu, Shixuan Sun, Surangika Ranathunga, Suranga Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2503.13081)  

**Abstract**: Large Language Models (LLMs) are acquiring a wider range of capabilities, including understanding and responding in multiple languages. While they undergo safety training to prevent them from answering illegal questions, imbalances in training data and human evaluation resources can make these models more susceptible to attacks in low-resource languages (LRL). This paper proposes a framework to automatically assess the multilingual vulnerabilities of commonly used LLMs. Using our framework, we evaluated six LLMs across eight languages representing varying levels of resource availability. We validated the assessments generated by our automated framework through human evaluation in two languages, demonstrating that the framework's results align with human judgments in most cases. Our findings reveal vulnerabilities in LRL; however, these may pose minimal risk as they often stem from the model's poor performance, resulting in incoherent responses. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在获得更广泛的能力，包括多语言的理解和响应。尽管它们接受安全性训练以防止回答非法问题，但训练数据不平衡和人类评估资源的不足可能使这些模型在低资源语言（LRL）中更容易受到攻击。本文提出了一种框架以自动评估常用LLM的多语言脆弱性。使用该框架，我们在八种代表不同资源可获取程度的语言上评估了六种LLM。通过两种语言的人类评估验证了我们自动框架生成的评估结果，显示该框架的结果大多数情况下与人类判断一致。我们的研究发现LRL存在脆弱性，但这些脆弱性通常源自模型表现不佳，导致不连贯的响应，可能带来的风险较小。 

---
# Mitigating Cross-Modal Distraction and Ensuring Geometric Feasibility via Affordance-Guided, Self-Consistent MLLMs for Food Preparation Task Planning 

**Title (ZH)**: 通过能力导向、自我一致的MLLMs减轻跨模态干扰并确保几何可行性以进行食品准备任务规划 

**Authors**: Yu-Hong Shen, Chuan-Yu Wu, Yi-Ru Yang, Yen-Ling Tai, Yi-Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13055)  

**Abstract**: We study Multimodal Large Language Models (MLLMs) with in-context learning for food preparation task planning. In this context, we identify two key challenges: cross-modal distraction and geometric feasibility. Cross-modal distraction occurs when the inclusion of visual input degrades the reasoning performance of a MLLM. Geometric feasibility refers to the ability of MLLMs to ensure that the selected skills are physically executable in the environment. To address these issues, we adapt Chain of Thought (CoT) with Self-Consistency to mitigate reasoning loss from cross-modal distractions and use affordance predictor as skill preconditions to guide MLLM on geometric feasibility. We construct a dataset to evaluate the ability of MLLMs on quantity estimation, reachability analysis, relative positioning and collision avoidance. We conducted a detailed evaluation to identify issues among different baselines and analyze the reasons for improvement, providing insights into each approach. Our method reaches a success rate of 76.7% on the entire dataset, showing a substantial improvement over the CoT baseline at 36.7%. 

**Abstract (ZH)**: 我们研究了具有上下文学习能力的多模态大型语言模型（MLLMs）在食物 preparation 任务规划中的应用。在此背景下，我们识别了两个关键挑战：跨模态干扰和几何可行性。跨模态干扰指的是视觉输入的加入会损害 MLLM 的推理性能。几何可行性是指 MLLM 确保所选技能在环境中物理可执行的能力。为了应对这些挑战，我们采用带自我一致性检验的推理链（CoT）减少跨模态干扰带来的推理损失，并使用功能预测器作为技能的前提条件，以指导 MLLM 对几何可行性的处理。我们构建了一个数据集，评估 MLLM 在数量估计、可达性分析、相对定位和碰撞避免方面的能力。我们进行了详细评估，识别不同基线中的问题并分析改进的原因，为每种方法提供了见解。我们的方法在整个数据集上的成功率达到了 76.7%，相较于 CoT 基线的 36.7% 显示出显著的改进。 

---
# A Multi-Stage Framework with Taxonomy-Guided Reasoning for Occupation Classification Using Large Language Models 

**Title (ZH)**: 基于分类 taxonomy 引导推理的多阶段框架在大规模语言模型中进行职业分类 

**Authors**: Palakorn Achananuparp, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12989)  

**Abstract**: Automatically annotating job data with standardized occupations from taxonomies, known as occupation classification, is crucial for labor market analysis. However, this task is often hindered by data scarcity and the challenges of manual annotations. While large language models (LLMs) hold promise due to their extensive world knowledge and in-context learning capabilities, their effectiveness depends on their knowledge of occupational taxonomies, which remains unclear. In this study, we assess the ability of LLMs to generate precise taxonomic entities from taxonomy, highlighting their limitations. To address these challenges, we propose a multi-stage framework consisting of inference, retrieval, and reranking stages, which integrates taxonomy-guided reasoning examples to enhance performance by aligning outputs with taxonomic knowledge. Evaluations on a large-scale dataset show significant improvements in classification accuracy. Furthermore, we demonstrate the framework's adaptability for multi-label skill classification. Our results indicate that the framework outperforms existing LLM-based methods, offering a practical and scalable solution for occupation classification and related tasks across LLMs. 

**Abstract (ZH)**: 自动使用标准化职业分类标注工作数据对于劳动力市场分析至关重要。然而，这一任务常常受到数据稀缺性和手动标注挑战的阻碍。虽然大型语言模型（LLMs）因其广泛的世界知识和上下文学习能力展现出潜力，但其有效性仍取决于对职业分类的知识掌握，这尚未明确。在本研究中，我们评估了LLMs生成精确分类实体的能力，并指出其局限性。为了应对这些挑战，我们提出了一种多阶段框架，包括推理、检索和重排序阶段，该框架整合了基于分类的推理示例，以通过将输出与分类知识对齐来提升性能。大规模数据集的评估显示分类准确性显著提高。此外，我们展示了该框架在多标签技能分类中的适用性。我们的结果表明，该框架优于现有基于LLM的方法，为职业分类及相关任务提供了一种实用且可扩展的解决方案，适用于各种LLM。 

---
# ROMA: a Read-Only-Memory-based Accelerator for QLoRA-based On-Device LLM 

**Title (ZH)**: ROMA：基于只读存储器的QLoRA-Based On-Device LLM加速器 

**Authors**: Wenqiang Wang, Yijia Zhang, Zikai Zhang, Guanting Huo, Hao Liang, Shijie Cao, Ningyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12988)  

**Abstract**: As large language models (LLMs) demonstrate powerful capabilities, deploying them on edge devices has become increasingly crucial, offering advantages in privacy and real-time interaction. QLoRA has emerged as the standard approach for on-device LLMs, leveraging quantized models to reduce memory and computational costs while utilizing LoRA for task-specific adaptability. In this work, we propose ROMA, a QLoRA accelerator with a hybrid storage architecture that uses ROM for quantized base models and SRAM for LoRA weights and KV cache. Our insight is that the quantized base model is stable and converged, making it well-suited for ROM storage. Meanwhile, LoRA modules offer the flexibility to adapt to new data without requiring updates to the base model. To further reduce the area cost of ROM, we introduce a novel B-ROM design and integrate it with the compute unit to form a fused cell for efficient use of chip resources. ROMA can effectively store both a 4-bit 3B and a 2-bit 8B LLaMA model entirely on-chip, achieving a notable generation speed exceeding 20,000 tokens/s without requiring external memory. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）展现出强大的能力，将它们部署在边缘设备上变得越来越重要，这在隐私和实时交互方面提供了优势。QLoRA已经成为边端LLM的标准方法，通过量化模型减少内存和计算成本，同时利用LoRA实现任务特定的适应性。在这项工作中，我们提出了ROMA，一种具有混合存储架构的QLoRA加速器，使用ROM存储量化基模型，使用SRAM存储LoRA权重和KV缓存。我们的见解是，量化基模型稳定且已收敛，非常适合ROM存储。同时，LoRA模块提供了灵活性，可以在无需更新基模型的情况下适应新数据。为了进一步减少ROM的面积成本，我们引入了一种新颖的B-ROM设计，并将其与计算单元集成，形成高效的芯片资源融合单元。ROMA可以在芯片上有效存储4位的3B和2位的8B LLaMA模型，实现超过20,000个令牌/秒的生成速度，无需外部内存。 

---
# Aligning Vision to Language: Text-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning 

**Title (ZH)**: 视觉与语言对齐：无文本的多模态知识图构建以增强LLM推理 

**Authors**: Junming Liu, Siyuan Meng, Yanting Gao, Song Mao, Pinlong Cai, Guohang Yan, Yirong Chen, Zilin Bian, Botian Shi, Ding Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12972)  

**Abstract**: Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型在多模态推理中面对不完整知识和幻觉问题，这些挑战即使在部分依赖于模态隔离的语言知识图谱中也无法完全缓解。虽然多模态知识图谱（MMKGs）能够增强跨模态理解，但它们的实际构建因手动文本注释的语义狭窄性和视觉语义实体链接中的固有噪声而受限。本文提出了一种名为Vision-align-to-Language集成知识图谱（VaLiK）的新方法，通过跨模态信息补充增强大型语言模型的推理能力。具体来说，我们利用预训练的视觉-语言模型将图像特征与文本对齐，并将其转化为包含图像特定信息的描述。此外，我们开发了一种跨模态相似性验证机制以量化语义一致性，有效过滤掉特征对齐过程中引入的噪声。即使没有手动标注的图像描述，经过改进的描述即可构建多模态知识图谱。与传统的多模态知识图谱构建方法相比，我们的方法在保持直接实体到图像链接能力的同时实现了显著的存储效率提升。实验结果表明，使用VaLiK增强的大型语言模型在多模态推理任务中优于之前最先进的模型。我们的代码在此处发布：this https URL。 

---
# MirrorGuard: Adaptive Defense Against Jailbreaks via Entropy-Guided Mirror Crafting 

**Title (ZH)**: MirrorGuard：基于熵导向的镜像 Crafting 以适应性防御越狱攻击 

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Litian Zhang, Lirong Qiu, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12931)  

**Abstract**: Defending large language models (LLMs) against jailbreak attacks is crucial for ensuring their safe deployment. Existing defense strategies generally rely on predefined static criteria to differentiate between harmful and benign prompts. However, such rigid rules are incapable of accommodating the inherent complexity and dynamic nature of real jailbreak attacks. In this paper, we propose a novel concept of ``mirror'' to enable dynamic and adaptive defense. A mirror refers to a dynamically generated prompt that mirrors the syntactic structure of the input while ensuring semantic safety. The personalized discrepancies between the input prompts and their corresponding mirrors serve as the guiding principles for defense. A new defense paradigm, MirrorGuard, is further proposed to detect and calibrate risky inputs based on such mirrors. An entropy-based detection metric, Relative Input Uncertainty (RIU), is integrated into MirrorGuard to quantify the discrepancies between input prompts and mirrors. MirrorGuard is evaluated on several popular datasets, demonstrating state-of-the-art defense performance while maintaining general effectiveness. 

**Abstract (ZH)**: 防御大语言模型（LLMs）免受Jailbreak攻击至关重要，以确保其安全部署。现有的防御策略通常依赖于预定义的静态标准来区分有害和无害的提示。然而，这样的僵化规则无法适应真实Jailbreak攻击的固有复杂性和动态性。本文提出了一种新的“镜像”概念，以实现动态和自适应的防御。镜像指的是一个动态生成的提示，它模仿输入的句法结构同时确保语义安全。个性化输入提示与其对应镜像之间的差异作为防御的指导原则。本文进一步提出了一种新的防御范式MirrorGuard，基于此类镜像来检测和校准潜在危险的输入。引入了一个基于熵的检测指标相对输入不确定性（RIU）来量化输入提示与镜像之间的差异。MirrorGuard在多个流行数据集上进行了评估，展示了领先的安全防御性能，同时保持了一般的有效性。 

---
# HICD: Hallucination-Inducing via Attention Dispersion for Contrastive Decoding to Mitigate Hallucinations in Large Language Models 

**Title (ZH)**: HICD：通过注意力分散诱导 hallucination 的对比解码方法以减轻大型语言模型中的 hallucination 

**Authors**: Xinyan Jiang, Hang Ye, Yongxin Zhu, Xiaoying Zheng, Zikang Chen, Jun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.12908)  

**Abstract**: Large Language Models (LLMs) often generate hallucinations, producing outputs that are contextually inaccurate or factually incorrect. We introduce HICD, a novel method designed to induce hallucinations for contrastive decoding to mitigate hallucinations. Unlike existing contrastive decoding methods, HICD selects attention heads crucial to the model's prediction as inducing heads, then induces hallucinations by dispersing attention of these inducing heads and compares the hallucinated outputs with the original outputs to obtain the final result. Our approach significantly improves performance on tasks requiring contextual faithfulness, such as context completion, reading comprehension, and question answering. It also improves factuality in tasks requiring accurate knowledge recall. We demonstrate that our inducing heads selection and attention dispersion method leads to more "contrast-effective" hallucinations for contrastive decoding, outperforming other hallucination-inducing methods. Our findings provide a promising strategy for reducing hallucinations by inducing hallucinations in a controlled manner, enhancing the performance of LLMs in a wide range of tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）往往会产生幻觉，产出与上下文不符或事实错误的输出。我们提出HICD，一种新颖的方法，通过对比解码来减轻幻觉，该方法选择对模型预测至关重要的注意力头作为诱导头，然后通过分散这些诱导头的注意力来诱导幻觉，并将诱导出的输出与原始输出进行比较以获得最终结果。我们的方法在需要上下文忠实性的任务（如上下文填充、阅读理解和问答）中显著提高了性能，同时也提高了需要准确知识回忆的任务的事实性。我们证明，我们的诱导头选择和注意力分散方法能够产生更有“对比效果”的幻觉，优于其他幻觉诱导方法。我们的研究结果提供了一种有希望的策略，通过控制方式诱导幻觉来减少幻觉，从而在各种任务中增强语言模型的性能。 

---
# nvBench 2.0: A Benchmark for Natural Language to Visualization under Ambiguity 

**Title (ZH)**: nvBench 2.0: 一种处理歧义性的自然语言到可视化基准 

**Authors**: Tianqi Luo, Chuhan Huang, Leixian Shen, Boyan Li, Shuyu Shen, Wei Zeng, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.12880)  

**Abstract**: Natural Language to Visualization (NL2VIS) enables users to create visualizations from natural language queries, making data insights more accessible. However, NL2VIS faces challenges in interpreting ambiguous queries, as users often express their visualization needs in imprecise language. To address this challenge, we introduce nvBench 2.0, a new benchmark designed to evaluate NL2VIS systems in scenarios involving ambiguous queries. nvBench 2.0 includes 7,878 natural language queries and 24,076 corresponding visualizations, derived from 780 tables across 153 domains. It is built using a controlled ambiguity-injection pipeline that generates ambiguous queries through a reverse-generation workflow. By starting with unambiguous seed visualizations and selectively injecting ambiguities, the pipeline yields multiple valid interpretations for each query, with each ambiguous query traceable to its corresponding visualization through step-wise reasoning paths. We evaluate various Large Language Models (LLMs) on their ability to perform ambiguous NL2VIS tasks using nvBench 2.0. We also propose Step-NL2VIS, an LLM-based model trained on nvBench 2.0, which enhances performance in ambiguous scenarios through step-wise preference optimization. Our results show that Step-NL2VIS outperforms all baselines, setting a new state-of-the-art for ambiguous NL2VIS tasks. 

**Abstract (ZH)**: 自然语言到可视化(NL2VIS)基准2.0：处理模糊查询的评估与优化 

---
# A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules 

**Title (ZH)**: 跨学习率调度的损失曲线预测多重幂律规律 

**Authors**: Kairong Luo, Haodong Wen, Shengding Hu, Zhenbo Sun, Zhiyuan Liu, Maosong Sun, Kaifeng Lyu, Wenguang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.12811)  

**Abstract**: Training large models is both resource-intensive and time-consuming, making it crucial to understand the quantitative relationship between model performance and hyperparameters. In this paper, we present an empirical law that describes how the pretraining loss of large language models evolves under different learning rate schedules, such as constant, cosine, and step decay schedules. Our proposed law takes a multi-power form, combining a power law based on the sum of learning rates and additional power laws to account for a loss reduction effect induced by learning rate decay. We extensively validate this law on various model sizes and architectures, and demonstrate that after fitting on a few learning rate schedules, the law accurately predicts the loss curves for unseen schedules of different shapes and horizons. Moreover, by minimizing the predicted final pretraining loss across learning rate schedules, we are able to find a schedule that outperforms the widely used cosine learning rate schedule. Interestingly, this automatically discovered schedule bears some resemblance to the recently proposed Warmup-Stable-Decay (WSD) schedule (Hu et al, 2024) but achieves a slightly lower final loss. We believe these results could offer valuable insights for understanding the dynamics of pretraining and designing learning rate schedules to improve efficiency. 

**Abstract (ZH)**: 大规模模型训练既资源密集又耗时，理解模型性能与超参数之间的量化关系至关重要。本文提出了一条经验定律，描述了在不同学习率调度（如恒定、余弦和阶梯衰减调度）下大规模语言模型预训练损失的变化规律。该定律采用多幂次形式，结合基于学习率总和的幂次定律和额外的幂次定律，以解释由学习率衰减引起的损失减少效应。我们在不同模型规模和架构上广泛验证了这条定律，并证明在拟合少量学习率调度后，该定律能够准确预测不同形状和持续时间的未见调度下的损失曲线。此外，通过最小化不同学习率调度下的预测最终预训练损失，我们找到了一种性能优于广泛使用的余弦学习率调度的调度方式。有趣的是，这种自动发现的调度方式与近期提出的一种温升稳定衰减（WSD）调度方式（Hu等人，2024）有些相似，但最终损失略低。我们认为这些结果为理解预训练动力学和设计提高效率的学习率调度提供了有价值的见解。 

---
# Quantum-Enhanced LLM Efficient Fine Tuning 

**Title (ZH)**: 量子增强的大语言模型高效微调 

**Authors**: Xiaofei Kong, Lei Li, Menghan Dou, Zhaoyun Chen, Yuchun Wu, Guoping Guo  

**Link**: [PDF](https://arxiv.org/pdf/2503.12790)  

**Abstract**: Low-Rank Adaptation (LoRA) enables efficient fine-tuning of pre-trained language models via low-rank matrix approximation, which is effective in many scenarios. However, its low-rank representation capacity is constrained in complex tasks or high-rank dependency settings, potentially limiting model adaptability. Addressing the expressive bottleneck of classical low-rank approximation in fine-tuning large language models, this paper proposes a parameter-efficient fine-tuning method based on a Quantum Weighted Tensor Hybrid Network (QWTHN), which leverages Quantum Neural Network (QNN). The study investigates quantum-classical hybrid parameter-efficient fine-tuning in low-rank spaces. QWTHN decomposes pre-trained weights into quantum neural network and tensor network representations, utilizing quantum state superposition and other methods to break through classical rank limitations. Experiments show that the proposed quantum fine-tuning technique for large models approaches or even surpasses the parameter efficiency of LoRA. On the CPsyCounD and R1-Distill-SFT datasets, QWTHN, compared to classical LoRA, reduces training loss by up to 15% while using 76% fewer parameters, and achieves an 8.4% performance improvement on the CPsyCounD test set. This research not only realizes lightweight and efficient adaptation of quantum resources to billion-parameter models but also validates the practical path of quantum hardware driven by large model tasks, laying the first engineering-ready technical foundation for future quantum-enhanced AGI systems. 

**Abstract (ZH)**: 基于量子加权张量混合网络的参数高效大型语言模型细调方法 

---
# Plausibility Vaccine: Injecting LLM Knowledge for Event Plausibility 

**Title (ZH)**: 拟合疫苗：注入大语言模型知识以提高事件可信度 

**Authors**: Jacob Chmura, Jonah Dauvet, Sebastian Sabry  

**Link**: [PDF](https://arxiv.org/pdf/2503.12667)  

**Abstract**: Despite advances in language modelling, distributional methods that build semantic representations from co-occurrences fail to discriminate between plausible and implausible events. In this work, we investigate how plausibility prediction can be improved by injecting latent knowledge prompted from large language models using parameter-efficient fine-tuning. We train 12 task adapters to learn various physical properties and association measures and perform adapter fusion to compose latent semantic knowledge from each task on top of pre-trained AlBERT embeddings. We automate auxiliary task data generation, which enables us to scale our approach and fine-tune our learned representations across two plausibility datasets. Our code is available at this https URL. 

**Abstract (ZH)**: 尽管在语言建模方面取得了进展，但基于共现构建语义表示的方法仍然无法区分可能事件和不可能事件。在此项工作中，我们研究了如何通过注入大型语言模型提示的潜在知识来提高可验证性预测性能，使用参数高效微调进行推进。我们训练了12个任务适配器来学习各种物理属性和关联度量，并通过适配器融合将每项任务中的潜在语义知识构建在预训练的AlBERT嵌入之上。我们自动化了辅助任务数据生成，从而使我们可以扩展我们的方法，并跨两个可验证性数据集微调我们学到的表示。我们的代码可在以下链接获取：this https URL。 

---
# MoECollab: Democratizing LLM Development Through Collaborative Mixture of Experts 

**Title (ZH)**: MoECollab: 通过专家混合协作 democratize LLM 开发 

**Authors**: Harshit  

**Link**: [PDF](https://arxiv.org/pdf/2503.12592)  

**Abstract**: Large Language Model (LLM) development has become increasingly centralized, limiting participation to well-resourced organizations. This paper introduces MoECollab, a novel framework leveraging Mixture of Experts (MoE) architecture to enable distributed, collaborative LLM development. By decomposing monolithic models into specialized expert modules coordinated by a trainable gating network, our framework allows diverse contributors to participate regardless of computational resources. We provide a complete technical implementation with mathematical foundations for expert dynamics, gating mechanisms, and integration strategies. Experiments on multiple datasets demonstrate that our approach achieves accuracy improvements of 3-7% over baseline models while reducing computational requirements by 34%. Expert specialization yields significant domain-specific gains, with improvements from 51% to 88% F1 score in general classification and from 23% to 44% accuracy in news categorization. We formalize the routing entropy optimization problem and demonstrate how proper regularization techniques lead to 14% higher expert utilization rates. These results validate MoECollab as an effective approach for democratizing LLM development through architecturally-supported collaboration. 

**Abstract (ZH)**: MoECollab：利用专家混合架构支持分布式协作的大语言模型开发 

---
# From Guessing to Asking: An Approach to Resolving the Persona Knowledge Gap in LLMs during Multi-Turn Conversations 

**Title (ZH)**: 从猜测到提问：在多轮对话中解决LLMs个性知识缺口的方法 

**Authors**: Sarvesh Baskar, Tanmay Tulsidas Verelakar, Srinivasan Parthasarathy, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2503.12556)  

**Abstract**: In multi-turn dialogues, large language models (LLM) face a critical challenge of ensuring coherence while adapting to user-specific information. This study introduces the persona knowledge gap, the discrepancy between a model's internal understanding and the knowledge required for coherent, personalized conversations. While prior research has recognized these gaps, computational methods for their identification and resolution remain underexplored. We propose Conversation Preference Elicitation and Recommendation (CPER), a novel framework that dynamically detects and resolves persona knowledge gaps using intrinsic uncertainty quantification and feedback-driven refinement. CPER consists of three key modules: a Contextual Understanding Module for preference extraction, a Dynamic Feedback Module for measuring uncertainty and refining persona alignment, and a Persona-Driven Response Generation module for adapting responses based on accumulated user context. We evaluate CPER on two real-world datasets: CCPE-M for preferential movie recommendations and ESConv for mental health support. Using A/B testing, human evaluators preferred CPER's responses 42% more often than baseline models in CCPE-M and 27% more often in ESConv. A qualitative human evaluation confirms that CPER's responses are preferred for maintaining contextual relevance and coherence, particularly in longer (12+ turn) conversations. 

**Abstract (ZH)**: 在多轮对话中，大型语言模型（LLM）面临确保连贯性并适应用户特定信息的关键挑战。本研究引入了 Persona 知识缺口的概念，即模型内部理解与其进行连贯和个性化对话所需知识之间的 discrepancy。虽然先前的研究已经认识到了这些缺口，但对其识别和解决的计算方法仍较少探索。我们提出了一种名为 Conversation Preference Elicitation and Recommendation (CPER) 的新型框架，该框架能够动态检测并解决 Persona 知识缺口，通过内在不确定性量化和反馈驱动的优化来实现。CPER 包含三个关键模块：上下文理解模块进行偏好提取、动态反馈模块衡量不确定性并精炼 Persona 对齐、以及基于累积用户上下文生成 Persona 驱动的响应模块。我们在两个真实世界数据集中评估了 CPER：CCPE-M 用于偏好电影推荐，ESConv 用于心理健康支持。A/B 测试结果显示，在 CCPE-M 中，人类评估者更偏好 CPER 的响应，偏好率为 42%，而在 ESConv 中，偏好率为 27%。定性的人类评估进一步证实了 CPER 的响应更适合保持上下文相关性和连贯性，尤其是在较长（12 轮及以上）的对话中。 

---
# EXAONE Deep: Reasoning Enhanced Language Models 

**Title (ZH)**: EXAONE Deep:  reasoning Enhanced Language Models 

**Authors**: LG AI Research, Kyunghoon Bae, Eunbi Choi, Kibong Choi, Stanley Jungkyu Choi, Yemuk Choi, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Kijeong Jeon, Gerrard Jeongwon Jo, Hyunjik Jo, Jiyeon Jung, Hyosang Kim, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Youchul Kim, Edward Hwayoung Lee, Haeju Lee, Honglak Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Yongmin Park, Sihoon Yang, Heuiyeen Yeen, Sihyuk Yi, Hyeongu Yun  

**Link**: [PDF](https://arxiv.org/pdf/2503.12524)  

**Abstract**: We present EXAONE Deep series, which exhibits superior capabilities in various reasoning tasks, including math and coding benchmarks. We train our models mainly on the reasoning-specialized dataset that incorporates long streams of thought processes. Evaluation results show that our smaller models, EXAONE Deep 2.4B and 7.8B, outperform other models of comparable size, while the largest model, EXAONE Deep 32B, demonstrates competitive performance against leading open-weight models. All EXAONE Deep models are openly available for research purposes and can be downloaded from this https URL 

**Abstract (ZH)**: EXAONE Deep系列：在各种推理任务，包括数学和编码基准测试中展现出卓越能力 

---
# LLM-Driven Multi-step Translation from C to Rust using Static Analysis 

**Title (ZH)**: 基于LLM的从C到Rust的多步翻译及静态分析驱动方法 

**Authors**: Tianyang Zhou, Haowen Lin, Somesh Jha, Mihai Christodorescu, Kirill Levchenko, Varun Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2503.12511)  

**Abstract**: Translating software written in legacy languages to modern languages, such as C to Rust, has significant benefits in improving memory safety while maintaining high performance. However, manual translation is cumbersome, error-prone, and produces unidiomatic code. Large language models (LLMs) have demonstrated promise in producing idiomatic translations, but offer no correctness guarantees as they lack the ability to capture all the semantics differences between the source and target languages. To resolve this issue, we propose SACTOR, an LLM-driven C-to-Rust zero-shot translation tool using a two-step translation methodology: an "unidiomatic" step to translate C into Rust while preserving semantics, and an "idiomatic" step to refine the code to follow Rust's semantic standards. SACTOR utilizes information provided by static analysis of the source C program to address challenges such as pointer semantics and dependency resolution. To validate the correctness of the translated result from each step, we use end-to-end testing via the foreign function interface to embed our translated code segment into the original code. We evaluate the translation of 200 programs from two datasets and two case studies, comparing the performance of GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash, Llama 3.3 70B and DeepSeek-R1 in SACTOR. Our results demonstrate that SACTOR achieves high correctness and improved idiomaticity, with the best-performing model (DeepSeek-R1) reaching 93% and (GPT-4o, Claude 3.5, DeepSeek-R1) reaching 84% correctness (on each dataset, respectively), while producing more natural and Rust-compliant translations compared to existing methods. 

**Abstract (ZH)**: 使用大型语言模型驱动的C到Rust零-shot翻译工具SACTOR：两步翻译方法及其应用 

---
# Facilitating Automated Online Consensus Building through Parallel Thinking 

**Title (ZH)**: 促进并行思考以实现自动化在线共识构建 

**Authors**: Wen Gu, Zhaoxing Li, Jan Buermann, Jim Dilkes, Dimitris Michailidis, Shinobu Hasegawa, Vahid Yazdanpanah, Sebastian Stein  

**Link**: [PDF](https://arxiv.org/pdf/2503.12499)  

**Abstract**: Consensus building is inherently challenging due to the diverse opinions held by stakeholders. Effective facilitation is crucial to support the consensus building process and enable efficient group decision making. However, the effectiveness of facilitation is often constrained by human factors such as limited experience and scalability. In this research, we propose a Parallel Thinking-based Facilitation Agent (PTFA) that facilitates online, text-based consensus building processes. The PTFA automatically collects textual posts and leverages large language models (LLMs) to perform all of the six distinct roles of the well-established Six Thinking Hats technique in parallel thinking. To illustrate the potential of PTFA, a pilot study was carried out and PTFA's ability in idea generation, emotional probing, and deeper analysis of ideas was demonstrated. Furthermore, a comprehensive dataset that contains not only the conversational content among the participants but also between the participants and the agent is constructed for future study. 

**Abstract (ZH)**: 基于并行思考的协同构建代理（PTFA）：促进在线文本协同构建过程 

---
# Augmented Adversarial Trigger Learning 

**Title (ZH)**: 增强对抗触发学习 

**Authors**: Zhe Wang, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2503.12339)  

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. 

**Abstract (ZH)**: 基于梯度优化的对抗攻击方法自动化学习生成逃逸提示或泄露系统提示的对抗触发器。在本文中，我们更详细地探讨了对抗触发器学习的优化目标，并提出了一种增强目标的对抗触发器学习方法ATLA（Adversarial Trigger Learning with Augmented objectives）。ATLA将之前研究中使用的负对数似然损失改进为加权损失形式，鼓励学习到的对抗触发器更加优化响应格式 tokens。这使得ATLA能够在仅有一个查询-响应对的情况下学习到对抗触发器，并且所学的触发器在面对其他类似查询时具有良好的泛化能力。我们进一步设计了一种变体，通过附加一个辅助损失来增强触发器优化，该损失抑制逃避性响应。我们展示了如何使用ATLA学习生成对抗后缀以逃逸LLM以及提取隐藏系统提示。实验证明，ATLA在性能上始终优于当前最先进的技术，在获得几乎100%攻击成功率的同时，所需的查询数量减少了80%。学会的逃逸后缀在面对未见过的查询时具有高度的泛化能力，并能很好地转移应用到新的LLM上。 

---
# Leveraging Vision Capabilities of Multimodal LLMs for Automated Data Extraction from Plots 

**Title (ZH)**: 利用多模态LLM的视觉能力实现图表中数据的自动化提取 

**Authors**: Maciej P. Polak, Dane Morgan  

**Link**: [PDF](https://arxiv.org/pdf/2503.12326)  

**Abstract**: Automated data extraction from research texts has been steadily improving, with the emergence of large language models (LLMs) accelerating progress even further. Extracting data from plots in research papers, however, has been such a complex task that it has predominantly been confined to manual data extraction. We show that current multimodal large language models, with proper instructions and engineered workflows, are capable of accurately extracting data from plots. This capability is inherent to the pretrained models and can be achieved with a chain-of-thought sequence of zero-shot engineered prompts we call PlotExtract, without the need to fine-tune. We demonstrate PlotExtract here and assess its performance on synthetic and published plots. We consider only plots with two axes in this analysis. For plots identified as extractable, PlotExtract finds points with over 90% precision (and around 90% recall) and errors in x and y position of around 5% or lower. These results prove that multimodal LLMs are a viable path for high-throughput data extraction for plots and in many circumstances can replace the current manual methods of data extraction. 

**Abstract (ZH)**: 基于大型语言模型的多模态数据提取方法在研究论文图示中的应用进展：PlotExtract验证 

---
# The Lucie-7B LLM and the Lucie Training Dataset: Open resources for multilingual language generation 

**Title (ZH)**: Lucie-7B大规模语言模型与Lucie训练数据集：多语言语言生成的开源资源 

**Authors**: Olivier Gouvert, Julie Hunter, Jérôme Louradour, Christophe Cerisara, Evan Dufraisse, Yaya Sy, Laura Rivière, Jean-Pierre Lorré, OpenLLM-France community  

**Link**: [PDF](https://arxiv.org/pdf/2503.12294)  

**Abstract**: We present both the Lucie Training Dataset and the Lucie-7B foundation model. The Lucie Training Dataset is a multilingual collection of textual corpora centered around French and designed to offset anglo-centric biases found in many datasets for large language model pretraining. Its French data is pulled not only from traditional web sources, but also from French cultural heritage documents, filling an important gap in modern datasets. Beyond French, which makes up the largest share of the data, we added documents to support several other European languages, including English, Spanish, German, and Italian. Apart from its value as a resource for French language and culture, an important feature of this dataset is that it prioritizes data rights by minimizing copyrighted material. In addition, building on the philosophy of past open projects, it is redistributed in the form used for training and its processing is described on Hugging Face and GitHub. The Lucie-7B foundation model is trained on equal amounts of data in French and English -- roughly 33% each -- in an effort to better represent cultural aspects of French-speaking communities. We also describe two instruction fine-tuned models, Lucie-7B-Instruct-v1.1 and Lucie-7B-Instruct-human-data, which we release as demonstrations of Lucie-7B in use. These models achieve promising results compared to state-of-the-art models, demonstrating that an open approach prioritizing data rights can still deliver strong performance. We see these models as an initial step toward developing more performant, aligned models in the near future. Model weights for Lucie-7B and the Lucie instruct models, along with intermediate checkpoints for the former, are published on Hugging Face, while model training and data preparation code is available on GitHub. This makes Lucie-7B one of the first OSI compliant language models according to the new OSI definition. 

**Abstract (ZH)**: 我们介绍了Lucie训练数据集和Lucie-7B基础模型。 

---
# Integrating Chain-of-Thought and Retrieval Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes 

**Title (ZH)**: 将链式思考与检索增强生成集成以提高临床笔记中罕见疾病诊断的准确性 

**Authors**: Da Wu, Zhanliang Wang, Quan Nguyen, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12286)  

**Abstract**: Background: Several studies show that large language models (LLMs) struggle with phenotype-driven gene prioritization for rare diseases. These studies typically use Human Phenotype Ontology (HPO) terms to prompt foundation models like GPT and LLaMA to predict candidate genes. However, in real-world settings, foundation models are not optimized for domain-specific tasks like clinical diagnosis, yet inputs are unstructured clinical notes rather than standardized terms. How LLMs can be instructed to predict candidate genes or disease diagnosis from unstructured clinical notes remains a major challenge. Methods: We introduce RAG-driven CoT and CoT-driven RAG, two methods that combine Chain-of-Thought (CoT) and Retrieval Augmented Generation (RAG) to analyze clinical notes. A five-question CoT protocol mimics expert reasoning, while RAG retrieves data from sources like HPO and OMIM (Online Mendelian Inheritance in Man). We evaluated these approaches on rare disease datasets, including 5,980 Phenopacket-derived notes, 255 literature-based narratives, and 220 in-house clinical notes from Childrens Hospital of Philadelphia. Results: We found that recent foundations models, including Llama 3.3-70B-Instruct and DeepSeek-R1-Distill-Llama-70B, outperformed earlier versions such as Llama 2 and GPT-3.5. We also showed that RAG-driven CoT and CoT-driven RAG both outperform foundation models in candidate gene prioritization from clinical notes; in particular, both methods with DeepSeek backbone resulted in a top-10 gene accuracy of over 40% on Phenopacket-derived clinical notes. RAG-driven CoT works better for high-quality notes, where early retrieval can anchor the subsequent reasoning steps in domain-specific evidence, while CoT-driven RAG has advantage when processing lengthy and noisy notes. 

**Abstract (ZH)**: 背景: 一些研究表明，大型语言模型（LLMs）在罕见疾病表型驱动的基因优先级确定方面存在困难。这些研究通常使用人类表型 ontology (HPO) 术语来提示如 GPT 和 LLaMA 等基础模型预测候选基因。然而，在实际应用中，基础模型并未针对如临床诊断等特定领域任务进行优化，输入的是未结构化的临床笔记而非标准化术语。如何从未结构化的临床笔记中指导 LLMs 预测候选基因或疾病诊断仍然是一个重大挑战。方法: 我们介绍了基于 RAG 的 CoT 和 CoT 指导的 RAG 两种方法，将 Chain-of-Thought (CoT) 和 Retrieval Augmented Generation (RAG) 结合起来分析临床笔记。我们使用一个五问题 CoT 评估方案来模拟专家推理过程，而 RAG 则从 HPO 和 OMIM（在线人类遗传学）等来源检索数据。我们在包含 5,980 条来源于 Phenopacket 的临床笔记、255 条基于文献的叙述以及 Childrens Hospital of Philadelphia 的 220 条内部临床笔记的罕见疾病数据集中评估了这些方法。结果: 我们发现，包括 Llama 3.3-70B-Instruct 和 DeepSeek-R1-Distill-Llama-70B 在内的最近版本的基础模型优于之前版本的 Llama 2 和 GPT-3.5。我们还展示了基于 RAG 的 CoT 和 CoT 指导的 RAG 方法在从临床笔记预测候选基因方面的表现优于基础模型；特别是，以 DeepSeek 为基础的方法在来源于 Phenopacket 的临床笔记中的 top-10 基因准确率超过 40%。基于 RAG 的 CoT 在高质量的笔记中表现出色，因为它可以通过早期检索为后续的推理步骤提供领域特定的证据，而 CoT 指导的 RAG 在处理冗长和噪音较多的笔记时具有优势。 

---
# Toward Foundation Models for Online Complex Event Detection in CPS-IoT: A Case Study 

**Title (ZH)**: 面向 CPS-IoT 中在线复杂事件检测的基础模型研究：一个案例研究 

**Authors**: Liying Han, Gaofeng Dong, Xiaomin Ouyang, Lance Kaplan, Federico Cerutti, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2503.12282)  

**Abstract**: Complex events (CEs) play a crucial role in CPS-IoT applications, enabling high-level decision-making in domains such as smart monitoring and autonomous systems. However, most existing models focus on short-span perception tasks, lacking the long-term reasoning required for CE detection. CEs consist of sequences of short-time atomic events (AEs) governed by spatiotemporal dependencies. Detecting them is difficult due to long, noisy sensor data and the challenge of filtering out irrelevant AEs while capturing meaningful patterns. This work explores CE detection as a case study for CPS-IoT foundation models capable of long-term reasoning. We evaluate three approaches: (1) leveraging large language models (LLMs), (2) employing various neural architectures that learn CE rules from data, and (3) adopting a neurosymbolic approach that integrates neural models with symbolic engines embedding human knowledge. Our results show that the state-space model, Mamba, which belongs to the second category, outperforms all methods in accuracy and generalization to longer, unseen sensor traces. These findings suggest that state-space models could be a strong backbone for CPS-IoT foundation models for long-span reasoning tasks. 

**Abstract (ZH)**: 复杂事件（CEs）在CPS-IoT应用中起着关键作用，能够在智能监测和自主系统等领域实现高级决策。然而，现有的大多数模型主要关注短时感知任务，缺乏用于CE检测所需的长期推理能力。CEs由受时空依赖关系支配的短暂时间原子事件（AEs）序列组成。检测CEs由于长期的噪声传感器数据和过滤无关AEs以捕捉有意义模式的挑战而困难。本研究以CEs检测为例探讨了具备长期推理能力的CPS-IoT基础模型。我们评估了三种方法：（1）利用大型语言模型（LLMs），（2）采用各种神经架构从数据中学习CE规则，以及（3）采用结合神经模型与嵌入人类知识的符号引擎的神经符号方法。我们的结果显示，属于第二类的态空间模型Mamba在准确性和对更长、未见过的传感器轨迹的一般化方面均优于所有方法。这表明态空间模型可能是长期跨度推理任务中CPS-IoT基础模型的强大支撑。 

---
# Agentic Search Engine for Real-Time IoT Data 

**Title (ZH)**: 基于代理的实时物联网数据搜索引擎 

**Authors**: Abdelrahman Elewah, Khalid Elgazzar  

**Link**: [PDF](https://arxiv.org/pdf/2503.12255)  

**Abstract**: The Internet of Things (IoT) has enabled diverse devices to communicate over the Internet, yet the fragmentation of IoT systems limits seamless data sharing and coordinated management. We have recently introduced SensorsConnect, a unified framework to enable seamless content and sensor data sharing in collaborative IoT systems, inspired by how the World Wide Web (WWW) enabled a shared and accessible space for information among humans. This paper presents the IoT Agentic Search Engine (IoT-ASE), a real-time search engine tailored for IoT environments. IoT-ASE leverages Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) techniques to address the challenge of searching vast, real-time IoT data, enabling it to handle complex queries and deliver accurate, contextually relevant results. We implemented a use-case scenario in Toronto to demonstrate how IoT-ASE can improve service quality recommendations by leveraging real-time IoT data. Our evaluation shows that IoT-ASE achieves a 92\% accuracy in retrieving intent-based services and produces responses that are concise, relevant, and context-aware, outperforming generalized responses from systems like Gemini. These findings highlight the potential IoT-ASE to make real-time IoT data accessible and support effective, real-time decision-making. 

**Abstract (ZH)**: 物联网代理搜索引擎（IoT-ASE）：面向物联网环境的实时搜索引擎 

---
# Adaptive Fault Tolerance Mechanisms of Large Language Models in Cloud Computing Environments 

**Title (ZH)**: 大型语言模型在云计算环境中的自适应容错机制 

**Authors**: Yihong Jin, Ze Yang, Xinhe Xu, Yihan Zhang, Shuyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.12228)  

**Abstract**: With the rapid evolution of Large Language Models (LLMs) and their large-scale experimentation in cloud-computing spaces, the challenge of guaranteeing their security and efficiency in a failure scenario has become a main issue. To ensure the reliability and availability of large-scale language models in cloud computing scenarios, such as frequent resource failures, network problems, and computational overheads, this study proposes a novel adaptive fault tolerance mechanism. It builds upon known fault-tolerant mechanisms, such as checkpointing, redundancy, and state transposition, introducing dynamic resource allocation and prediction of failure based on real-time performance metrics. The hybrid model integrates data driven deep learning-based anomaly detection technique underlining the contribution of cloud orchestration middleware for predictive prevention of system failures. Additionally, the model integrates adaptive checkpointing and recovery strategies that dynamically adapt according to load and system state to minimize the influence on the performance of the model and minimize downtime. The experimental results demonstrate that the designed model considerably enhances the fault tolerance in large-scale cloud surroundings, and decreases the system downtime by $\mathbf{30\%}$, and has a better modeling availability than the classical fault tolerance mechanism. 

**Abstract (ZH)**: 带预测性预防的自适应故障容忍机制在云环境中大规模语言模型中的应用研究 

---
# Research on Large Language Model Cross-Cloud Privacy Protection and Collaborative Training based on Federated Learning 

**Title (ZH)**: 基于联邦学习的大型语言模型跨云隐私保护与协作训练研究 

**Authors**: Ze Yang, Yihong Jin, Yihan Zhang, Juntian Liu, Xinhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12226)  

**Abstract**: The fast development of large language models (LLMs) and popularization of cloud computing have led to increasing concerns on privacy safeguarding and data security of cross-cloud model deployment and training as the key challenges. We present a new framework for addressing these issues along with enabling privacy preserving collaboration on training between distributed clouds based on federated learning. Our mechanism encompasses cutting-edge cryptographic primitives, dynamic model aggregation techniques, and cross-cloud data harmonization solutions to enhance security, efficiency, and scalability to the traditional federated learning paradigm. Furthermore, we proposed a hybrid aggregation scheme to mitigate the threat of Data Leakage and to optimize the aggregation of model updates, thus achieving substantial enhancement on the model effectiveness and stability. Experimental results demonstrate that the training efficiency, privacy protection, and model accuracy of the proposed model compare favorably to those of the traditional federated learning method. 

**Abstract (ZH)**: 大型语言模型的快速发展和云计算的普及导致了跨云模型部署与训练中的隐私保护和数据安全问题日益成为关键挑战。我们提出了一种新的框架，结合联邦学习，在分布式的云之间实现隐私保护的训练协作。我们的机制包括先进的密码学原语、动态模型聚合技术和跨云数据协调解决方案，以增强传统联邦学习范式的安全性、效率和可扩展性。此外，我们提出了一种混合聚合方案以减轻数据泄露威胁并优化模型更新的聚合，从而在模型效果和稳定性方面实现显著提升。实验结果表明，所提出模型的训练效率、隐私保护和模型准确率优于传统的联邦学习方法。 

---
# MT-RewardTree: A Comprehensive Framework for Advancing LLM-Based Machine Translation via Reward Modeling 

**Title (ZH)**: MT-RewardTree：一种基于奖励建模推动大语言模型驱动机器翻译发展的综合框架 

**Authors**: Zhaopeng Feng, Jiahan Ren, Jiayuan Su, Jiamei Zheng, Zhihang Tang, Hongwei Wang, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12123)  

**Abstract**: Process reward models (PRMs) have shown success in complex reasoning tasks for large language models (LLMs). However, their application to machine translation (MT) remains underexplored due to the lack of systematic methodologies and evaluation benchmarks. To address this gap, we introduce \textbf{MT-RewardTree}, a comprehensive framework for constructing, evaluating, and deploying process reward models in MT. Unlike traditional vanilla preference pair construction, we propose a novel method for automatically generating token-level preference pairs using approximate Monte Carlo Tree Search (MCTS), which mitigates the prohibitive cost of human annotation for fine-grained steps. Then, we establish the first MT-specific reward model benchmark and provide a systematic comparison of different reward modeling architectures, revealing that token-level supervision effectively captures fine-grained preferences. Experimental results demonstrate that our MT-PRM-Qwen-2.5-3B achieves state-of-the-art performance in both token-level and sequence-level evaluation given the same input prefix. Furthermore, we showcase practical applications where PRMs enable test-time alignment for LLMs without additional alignment training and significantly improve performance in hypothesis ensembling. Our work provides valuable insights into the role of reward models in MT research. Our code and data are released in \href{this https URL}{this https URL\_RewardTreePage}. 

**Abstract (ZH)**: 用于机器翻译的process奖励模型框架：MT-RewardTree 

---
# Universal Speech Token Learning via Low-Bitrate Neural Codec and Pretrained Representations 

**Title (ZH)**: 低比特率神经编解码器和预训练表示的通用语音 token 学习 

**Authors**: Xue Jiang, Xiulian Peng, Yuan Zhang, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12115)  

**Abstract**: Current large speech language models are mainly based on semantic tokens from discretization of self-supervised learned representations and acoustic tokens from a neural codec, following a semantic-modeling and acoustic-synthesis paradigm. However, semantic tokens discard paralinguistic attributes of speakers that is important for natural spoken communication, while prompt-based acoustic synthesis from semantic tokens has limits in recovering paralinguistic details and suffers from robustness issues, especially when there are domain gaps between the prompt and the target. This paper unifies two types of tokens and proposes the UniCodec, a universal speech token learning that encapsulates all semantics of speech, including linguistic and paralinguistic information, into a compact and semantically-disentangled unified token. Such a unified token can not only benefit speech language models in understanding with paralinguistic hints but also help speech generation with high-quality output. A low-bitrate neural codec is leveraged to learn such disentangled discrete representations at global and local scales, with knowledge distilled from self-supervised learned features. Extensive evaluations on multilingual datasets demonstrate its effectiveness in generating natural, expressive and long-term consistent output quality with paralinguistic attributes well preserved in several speech processing tasks. 

**Abstract (ZH)**: 当前的大规模语言模型主要基于自监督学习表示的语义token和神经编解码器的声学token进行离散化，遵循语义建模和声学合成的范式。然而，语义token舍弃了影响自然口语交流的语用属性，而基于提示的声学合成从语义token恢复语用细节的能力有限，并且在提示与目标存在领域差异时尤其存在稳健性问题。本文统一了两种类型的token，提出了统一编解码器UniCodec，这是一种统一的语音token学习，将所有语言和非语言信息紧凑且语义地封装到单一token中。这种统一token不仅有助于语音语言模型在使用语用提示时的理解，还能帮助高质量的语音生成。利用低比特率神经编解码器在全局和局部尺度上学习语义分离的离散表示，并从自监督学习特征中提取知识。在多种语音处理任务中的多语言数据集上的广泛评估表明，UniCodec能够生成自然、表达性强且长时一致的输出，同时保留了语用属性。 

---
# RECSIP: REpeated Clustering of Scores Improving the Precision 

**Title (ZH)**: RECSIP: 重复聚类改进精度 

**Authors**: André Schamschurko, Nenad Petrovic, Alois Christian Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.12108)  

**Abstract**: The latest research on Large Language Models (LLMs) has demonstrated significant advancement in the field of Natural Language Processing (NLP). However, despite this progress, there is still a lack of reliability in these models. This is due to the stochastic architecture of LLMs, which presents a challenge for users attempting to ascertain the reliability of a model's response. These responses may cause serious harm in high-risk environments or expensive failures in industrial contexts. Therefore, we introduce the framework REpeated Clustering of Scores Improving the Precision (RECSIP) which focuses on improving the precision of LLMs by asking multiple models in parallel, scoring and clustering their responses to ensure a higher reliability on the response. The evaluation of our reference implementation recsip on the benchmark MMLU-Pro using the models GPT-4o, Claude and Gemini shows an overall increase of 5.8 per cent points compared to the best used model. 

**Abstract (ZH)**: 大型语言模型（LLMs）的最新研究在自然语言处理（NLP）领域展现了显著的进步。然而，尽管取得了这些进展，这些模型仍缺乏可靠性。这归因于LLMs的随机性架构，这给用户确定模型响应可靠性的过程带来了挑战。这些响应在高风险环境或工业场景中可能导致严重的危害或昂贵的失败。因此，我们提出了一种名为REpeated Clustering of Scores Improving the Precision (RECSIP) 的框架，该框架通过并行询问多个模型、评分和聚类其响应来提高LLMs的精度，从而确保响应的更高可靠性。我们在基准MMLU-Pro上对参考实现recsip的评估显示，与所使用的最佳模型相比，其准确率提高了5.8个百分点。 

---
# Comparing Human Expertise and Large Language Models Embeddings in Content Validity Assessment of Personality Tests 

**Title (ZH)**: 比较人类专家与大型语言模型嵌入在人格测验内容效度评估中的表现 

**Authors**: Nicola Milano, Michela Ponticorvo, Davide Marocco  

**Link**: [PDF](https://arxiv.org/pdf/2503.12080)  

**Abstract**: In this article we explore the application of Large Language Models (LLMs) in assessing the content validity of psychometric instruments, focusing on the Big Five Questionnaire (BFQ) and Big Five Inventory (BFI). Content validity, a cornerstone of test construction, ensures that psychological measures adequately cover their intended constructs. Using both human expert evaluations and advanced LLMs, we compared the accuracy of semantic item-construct alignment. Graduate psychology students employed the Content Validity Ratio (CVR) to rate test items, forming the human baseline. In parallel, state-of-the-art LLMs, including multilingual and fine-tuned models, analyzed item embeddings to predict construct mappings. The results reveal distinct strengths and limitations of human and AI approaches. Human validators excelled in aligning the behaviorally rich BFQ items, while LLMs performed better with the linguistically concise BFI items. Training strategies significantly influenced LLM performance, with models tailored for lexical relationships outperforming general-purpose LLMs. Here we highlights the complementary potential of hybrid validation systems that integrate human expertise and AI precision. The findings underscore the transformative role of LLMs in psychological assessment, paving the way for scalable, objective, and robust test development methodologies. 

**Abstract (ZH)**: 本文探讨了大型语言模型（LLMs）在评估心理测量工具内容效度中的应用，重点关注五大人格问卷（BFQ）和五大人格问卷（BFI）。内容效度是测试构建的基石，确保心理测量工具充分覆盖其预期的心理结构。利用人类专家评估和先进的LLM，我们比较了语义项目-构念对齐的准确性。研究生心理学学生使用内容效度比（CVR）评估测试项目，形成人类基准。同时，最先进的LLM，包括多语言和细调模型，分析项目嵌入以预测构念映射。研究结果揭示了人类和AI方法的各自优势和局限性。人类验证者在对行为丰富的BFQ项目进行对齐方面表现出色，而LLM在处理语言简洁的BFI项目方面表现更好。训练策略显著影响了LLM的性能，针对词汇关系的模型优于通用LLM。本文强调了结合人类专业知识和AI精确性的混合验证系统的互补潜力。研究结果凸显了LLM在心理评估中的变革作用，为可扩展、客观和稳健的测试开发方法铺平了道路。 

---
# V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents 

**Title (ZH)**: V-Stylist: 视频风格化 via MLLM 剂量的协作与反思 

**Authors**: Zhengrong Yue, Shaobin Zhuang, Kunchang Li, Yanbo Ding, Yali Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12077)  

**Abstract**: Despite the recent advancement in video stylization, most existing methods struggle to render any video with complex transitions, based on an open style description of user query. To fill this gap, we introduce a generic multi-agent system for video stylization, V-Stylist, by a novel collaboration and reflection paradigm of multi-modal large language models. Specifically, our V-Stylist is a systematical workflow with three key roles: (1) Video Parser decomposes the input video into a number of shots and generates their text prompts of key shot content. Via a concise video-to-shot prompting paradigm, it allows our V-Stylist to effectively handle videos with complex transitions. (2) Style Parser identifies the style in the user query and progressively search the matched style model from a style tree. Via a robust tree-of-thought searching paradigm, it allows our V-Stylist to precisely specify vague style preference in the open user query. (3) Style Artist leverages the matched model to render all the video shots into the required style. Via a novel multi-round self-reflection paradigm, it allows our V-Stylist to adaptively adjust detail control, according to the style requirement. With such a distinct design of mimicking human professionals, our V-Stylist achieves a major breakthrough over the primary challenges for effective and automatic video stylization. Moreover,we further construct a new benchmark Text-driven Video Stylization Benchmark (TVSBench), which fills the gap to assess stylization of complex videos on open user queries. Extensive experiments show that, V-Stylist achieves the state-of-the-art, e.g.,V-Stylist surpasses FRESCO and ControlVideo by 6.05% and 4.51% respectively in overall average metrics, marking a significant advance in video stylization. 

**Abstract (ZH)**: 基于多模态大语言模型新颖协作与反思范式的通用多 agent 视频风格化系统 V-Stylist 

---
# Applications of Large Language Model Reasoning in Feature Generation 

**Title (ZH)**: 大型语言模型推理在特征生成中的应用 

**Authors**: Dharani Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2503.11989)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing through their state of art reasoning capabilities. This paper explores the convergence of LLM reasoning techniques and feature generation for machine learning tasks. We examine four key reasoning approaches: Chain of Thought, Tree of Thoughts, Retrieval-Augmented Generation, and Thought Space Exploration. Our analysis reveals how these approaches can be used to identify effective feature generation rules without having to manually specify search spaces. The paper categorizes LLM-based feature generation methods across various domains including finance, healthcare, and text analytics. LLMs can extract key information from clinical notes and radiology reports in healthcare, by enabling more efficient data utilization. In finance, LLMs facilitate text generation, summarization, and entity extraction from complex documents. We analyze evaluation methodologies for assessing feature quality and downstream performance, with particular attention to OCTree's decision tree reasoning approach that provides language-based feedback for iterative improvements. Current challenges include hallucination, computational efficiency, and domain adaptation. As of March 2025, emerging approaches include inference-time compute scaling, reinforcement learning, and supervised fine-tuning with model distillation. Future directions point toward multimodal feature generation, self-improving systems, and neuro-symbolic approaches. This paper provides a detailed overview of an emerging field that promises to automate and enhance feature engineering through language model reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过其先进的推理能力革命性地改变了自然语言处理。本文探讨了LLM推理技术与特征生成在机器学习任务中的融合。我们研究了四种关键的推理方法：链式思维、思路树、检索增强生成以及思维空间探索。我们的分析揭示了这些方法如何用于识别有效的特征生成规则，而无需手动指定搜索空间。本文将LLM基于的特征生成方法分类应用于金融、医疗和文本分析等各种领域。在医疗领域，LLM能够从临床笔记和放射学报告中提取关键信息，通过促进更高效的数据显示利用。在金融领域，LLM促进文本生成、总结以及从复杂文档中提取实体。我们分析了评估特征质量和下游性能的评估方法，特别是OCTree的决策树推理方法，该方法提供基于语言的反馈以实现迭代改进。当前的挑战包括幻想、计算效率和领域适应。截至2025年3月，新兴的方法包括推理时计算扩展、强化学习和模型蒸馏下的监督微调。未来方向指向多模态特征生成、自我改进系统和神经-符号方法。本文提供了关于通过语言模型推理自动和增强特征工程新兴领域的详细概述。 

---
# No LLM is Free From Bias: A Comprehensive Study of Bias Evaluation in Large Language models 

**Title (ZH)**: 没有免费的午餐：大型语言模型偏见评估的全面研究 

**Authors**: Charaka Vinayak Kumar, Ashok Urlana, Gopichand Kanumolu, Bala Mallikarjunarao Garlapati, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2503.11985)  

**Abstract**: Advancements in Large Language Models (LLMs) have increased the performance of different natural language understanding as well as generation tasks. Although LLMs have breached the state-of-the-art performance in various tasks, they often reflect different forms of bias present in the training data. In the light of this perceived limitation, we provide a unified evaluation of benchmarks using a set of representative LLMs that cover different forms of biases starting from physical characteristics to socio-economic categories. Moreover, we propose five prompting approaches to carry out the bias detection task across different aspects of bias. Further, we formulate three research questions to gain valuable insight in detecting biases in LLMs using different approaches and evaluation metrics across benchmarks. The results indicate that each of the selected LLMs suffer from one or the other form of bias with the LLaMA3.1-8B model being the least biased. Finally, we conclude the paper with the identification of key challenges and possible future directions. 

**Abstract (ZH)**: 大型语言模型的进展提高了不同自然语言理解和生成任务的性能。尽管大型语言模型在各种任务中达到了最先进的性能，但它们常常反映出训练数据中存在的不同形式的偏见。鉴于这一局限性，我们使用一组代表性的大型语言模型进行统一评估，这些模型涵盖了从物理特征到社会经济类别等不同形式的偏见。此外，我们提出了五种提示方法，以在不同偏见方面执行偏见检测任务。进一步地，我们提出了三个研究问题，以通过不同方法和评估指标在基准测试中获取有关检测大型语言模型中偏见的有价值见解。结果表明，所选的每个大型语言模型都存在某种形式的偏见，LLaMA3.1-8B模型是最不偏颇的。最后，我们总结了论文中的关键挑战和可能的未来方向。 

---
# HInter: Exposing Hidden Intersectional Bias in Large Language Models 

**Title (ZH)**: HInter: 展示大规模语言模型中的隐藏综合性偏见 

**Authors**: Badr Souani, Ezekiel Soremekun, Mike Papadakis, Setsuko Yokoyama, Sudipta Chattopadhyay, Yves Le Traon  

**Link**: [PDF](https://arxiv.org/pdf/2503.11962)  

**Abstract**: Large Language Models (LLMs) may portray discrimination towards certain individuals, especially those characterized by multiple attributes (aka intersectional bias). Discovering intersectional bias in LLMs is challenging, as it involves complex inputs on multiple attributes (e.g. race and gender). To address this challenge, we propose HInter, a test technique that synergistically combines mutation analysis, dependency parsing and metamorphic oracles to automatically detect intersectional bias in LLMs. HInter generates test inputs by systematically mutating sentences using multiple mutations, validates inputs via a dependency invariant and detects biases by checking the LLM response on the original and mutated sentences. We evaluate HInter using six LLM architectures and 18 LLM models (GPT3.5, Llama2, BERT, etc) and find that 14.61% of the inputs generated by HInter expose intersectional bias. Results also show that our dependency invariant reduces false positives (incorrect test inputs) by an order of magnitude. Finally, we observed that 16.62% of intersectional bias errors are hidden, meaning that their corresponding atomic cases do not trigger biases. Overall, this work emphasize the importance of testing LLMs for intersectional bias. 

**Abstract (ZH)**: 大型语言模型中交叠偏见的检测：HInter测试技术 

---
# Integration of Explainable AI Techniques with Large Language Models for Enhanced Interpretability for Sentiment Analysis 

**Title (ZH)**: 将可解释AI技术与大规模语言模型结合以增强情感分析的可解释性 

**Authors**: Thivya Thogesan, Anupiya Nugaliyadde, Kok Wai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11948)  

**Abstract**: Interpretability remains a key difficulty in sentiment analysis with Large Language Models (LLMs), particularly in high-stakes applications where it is crucial to comprehend the rationale behind forecasts. This research addressed this by introducing a technique that applies SHAP (Shapley Additive Explanations) by breaking down LLMs into components such as embedding layer,encoder,decoder and attention layer to provide a layer-by-layer knowledge of sentiment prediction. The approach offers a clearer overview of how model interpret and categorise sentiment by breaking down LLMs into these parts. The method is evaluated using the Stanford Sentiment Treebank (SST-2) dataset, which shows how different sentences affect different layers. The effectiveness of layer-wise SHAP analysis in clarifying sentiment-specific token attributions is demonstrated by experimental evaluations, which provide a notable enhancement over current whole-model explainability techniques. These results highlight how the suggested approach could improve the reliability and transparency of LLM-based sentiment analysis in crucial applications. 

**Abstract (ZH)**: 大型语言模型中情感分析的可解释性依然是一个关键难题，特别是在高风险应用中，理解预测背后的逻辑至关重要。本研究通过引入一种方法来应对这一挑战，该方法利用SHAP（SHapley Additive Explanations）将大型语言模型分解为嵌入层、编码器、解码器和注意力层等组件，以逐层揭示情感预测的知识。该方法通过将大型语言模型分解为这些部分，为模型如何解释和分类情感提供更清晰的 overview。该方法使用斯坦福情感树库（SST-2）数据集进行评估，展示了不同句子如何影响不同层。实验评价展示了分层SHAP分析在阐明情感特定标记属性方面的有效性，这比现有整个模型的解释性方法有了显著提升。这些结果突出显示了所提方法如何在关键应用中提高基于大型语言模型的情感分析的可靠性和透明度。 

---
# REGEN: A Dataset and Benchmarks with Natural Language Critiques and Narratives 

**Title (ZH)**: REGEN：一个包含自然语言批评和叙事的数据集和基准 

**Authors**: Kun Su, Krishna Sayana, Hubert Pham, James Pine, Yuri Vasilevski, Raghavendra Vasudeva, Marialena Kyriakidi, Liam Hebert, Ambarish Jash, Anushya Subbiah, Sukhdeep Sodhi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11924)  

**Abstract**: This paper introduces a novel dataset REGEN (Reviews Enhanced with GEnerative Narratives), designed to benchmark the conversational capabilities of recommender Large Language Models (LLMs), addressing the limitations of existing datasets that primarily focus on sequential item prediction. REGEN extends the Amazon Product Reviews dataset by inpainting two key natural language features: (1) user critiques, representing user "steering" queries that lead to the selection of a subsequent item, and (2) narratives, rich textual outputs associated with each recommended item taking into account prior context. The narratives include product endorsements, purchase explanations, and summaries of user preferences.
Further, we establish an end-to-end modeling benchmark for the task of conversational recommendation, where models are trained to generate both recommendations and corresponding narratives conditioned on user history (items and critiques). For this joint task, we introduce a modeling framework LUMEN (LLM-based Unified Multi-task Model with Critiques, Recommendations, and Narratives) which uses an LLM as a backbone for critiquing, retrieval and generation. We also evaluate the dataset's quality using standard auto-rating techniques and benchmark it by training both traditional and LLM-based recommender models. Our results demonstrate that incorporating critiques enhances recommendation quality by enabling the recommender to learn language understanding and integrate it with recommendation signals. Furthermore, LLMs trained on our dataset effectively generate both recommendations and contextual narratives, achieving performance comparable to state-of-the-art recommenders and language models. 

**Abstract (ZH)**: 一种新型数据集REGEN（增强生成叙述的评论），用于评估推荐大型语言模型的对话能力 

---
# LLMs for Translation: Historical, Low-Resourced Languages and Contemporary AI Models 

**Title (ZH)**: LLMs for Translation: Historical and Low-Resourced Languages and Contemporary AI Models 

**Authors**: Merve Tekgurler  

**Link**: [PDF](https://arxiv.org/pdf/2503.11898)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable adaptability in performing various tasks, including machine translation (MT), without explicit training. Models such as OpenAI's GPT-4 and Google's Gemini are frequently evaluated on translation benchmarks and utilized as translation tools due to their high performance. This paper examines Gemini's performance in translating an 18th-century Ottoman Turkish manuscript, Prisoner of the Infidels: The Memoirs of Osman Agha of Timisoara, into English. The manuscript recounts the experiences of Osman Agha, an Ottoman subject who spent 11 years as a prisoner of war in Austria, and includes his accounts of warfare and violence. Our analysis reveals that Gemini's safety mechanisms flagged between 14 and 23 percent of the manuscript as harmful, resulting in untranslated passages. These safety settings, while effective in mitigating potential harm, hinder the model's ability to provide complete and accurate translations of historical texts. Through real historical examples, this study highlights the inherent challenges and limitations of current LLM safety implementations in the handling of sensitive and context-rich materials. These real-world instances underscore potential failures of LLMs in contemporary translation scenarios, where accurate and comprehensive translations are crucial-for example, translating the accounts of modern victims of war for legal proceedings or humanitarian documentation. 

**Abstract (ZH)**: Large Language Models (LLMs)在无需显式训练的情况下展示了在各种任务中的卓越适应性，包括机器翻译（MT）。如OpenAI的GPT-4和Google的Gemini等模型由于其高性能，经常用于翻译基准测试并作为翻译工具。本文探讨了Gemini在将18世纪的奥斯曼土耳其手稿《异教徒的囚徒：特梅索拉奥斯曼帕夏的回忆录》翻译成英文方面的表现。该手稿记载了奥斯曼帕夏奥斯曼·阿加在奥地利被俘11年的经历，包括他对战争和暴力的描述。我们的分析显示，Gemini的安全机制标记了手稿14%到23%的内容为有害内容，导致这些部分未被翻译。这些安全设置虽然有效减轻了潜在的危害，但也限制了模型提供全面准确的历史文本翻译的能力。通过实际情况，本文揭示了当前LLM安全实施在处理敏感和情境丰富材料时固有的挑战和局限性。这些实际情况突显了在当今翻译场景中LLM可能在准确和全面翻译方面存在的潜在失败，例如在法律程序或人道主义记录中翻译现代战争受害者的叙述。 

---
# Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing 

**Title (ZH)**: 基于迭代与邻域辅助模型编辑的欠编辑与过编辑解决方法 

**Authors**: Bhiman Kumar Baghel, Scott M. Jordan, Zheyuan Ryan Shi, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11895)  

**Abstract**: Large Language Models (LLMs) are used in various downstream language tasks, making it crucial to keep their knowledge up-to-date, but both retraining and fine-tuning the model can be costly. Model editing offers an efficient and effective alternative by a single update to only a key subset of model parameters. While being efficient, these methods are not perfect. Sometimes knowledge edits are unsuccessful, i.e., UnderEdit, or the edit contaminated neighboring knowledge that should remain unchanged, i.e., OverEdit. To address these limitations, we propose iterative model editing, based on our hypothesis that a single parameter update is often insufficient, to mitigate UnderEdit, and neighbor-assisted model editing, which incorporates neighboring knowledge during editing to minimize OverEdit. Extensive experiments demonstrate that our methods effectively reduce UnderEdit up to 38 percentage points and OverEdit up to 6 percentage points across multiple model editing algorithms, LLMs, and benchmark datasets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种下游语言任务中被广泛应用，保持其知识的时效性至关重要，但重新训练和微调模型可能会很昂贵。通过单次更新关键参数子集来进行模型编辑提供了高效且有效的替代方案。尽管这些方法是高效的，但并非完美无缺。有时知识编辑会失败，即UnderEdit，或者编辑会污染不应改变的邻近知识，即OverEdit。为解决这些问题，我们提出了迭代模型编辑，并基于假设单次参数更新往往不足以改进UnderEdit问题；同时引入了邻域辅助模型编辑，该方法在编辑过程中整合邻近知识以减少OverEdit问题。广泛实验证明，我们的方法能有效降低多个模型编辑算法、大型语言模型和基准数据集上的UnderEdit最高38个百分点和OverEdit最高6个百分点。 

---
# FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-the-World LoRA 

**Title (ZH)**: FedALT：通过适配本地训练与全局参数调整的联邦微调 

**Authors**: Jieming Bian, Lei Wang, Letian Zhang, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11880)  

**Abstract**: Fine-tuning large language models (LLMs) in federated settings enables privacy-preserving adaptation but suffers from cross-client interference due to model aggregation. Existing federated LoRA fine-tuning methods, primarily based on FedAvg, struggle with data heterogeneity, leading to harmful cross-client interference and suboptimal personalization. In this work, we propose \textbf{FedALT}, a novel personalized federated LoRA fine-tuning algorithm that fundamentally departs from FedAvg. Instead of using an aggregated model to initialize local training, each client continues training its individual LoRA while incorporating shared knowledge through a separate Rest-of-the-World (RoTW) LoRA component. To effectively balance local adaptation and global information, FedALT introduces an adaptive mixer that dynamically learns input-specific weightings between the individual and RoTW LoRA components using the Mixture-of-Experts (MoE) principle. Through extensive experiments on NLP benchmarks, we demonstrate that FedALT significantly outperforms state-of-the-art personalized federated LoRA fine-tuning methods, achieving superior local adaptation without sacrificing computational efficiency. 

**Abstract (ZH)**: FedALT：一种新型个性化联邦LoRA微调算法 

---
# Safe Vision-Language Models via Unsafe Weights Manipulation 

**Title (ZH)**: 通过不安全的权重操纵实现安全的跨模态模型 

**Authors**: Moreno D'Incà, Elia Peruzzo, Xingqian Xu, Humphrey Shi, Nicu Sebe, Massimiliano Mancini  

**Link**: [PDF](https://arxiv.org/pdf/2503.11742)  

**Abstract**: Vision-language models (VLMs) often inherit the biases and unsafe associations present within their large-scale training dataset. While recent approaches mitigate unsafe behaviors, their evaluation focuses on how safe the model is on unsafe inputs, ignoring potential shortcomings on safe ones. In this paper, we first revise safety evaluation by introducing SafeGround, a new set of metrics that evaluate safety at different levels of granularity. With this metric, we uncover a surprising issue of training-based methods: they make the model less safe on safe inputs. From this finding, we take a different direction and explore whether it is possible to make a model safer without training, introducing Unsafe Weights Manipulation (UWM). UWM uses a calibration set of safe and unsafe instances to compare activations between safe and unsafe content, identifying the most important parameters for processing the latter. Their values are then manipulated via negation. Experiments show that UWM achieves the best tradeoff between safety and knowledge preservation, consistently improving VLMs on unsafe queries while outperforming even training-based state-of-the-art methods on safe ones. 

**Abstract (ZH)**: 基于视觉-语言模型的安全性评估：SafeGround及其应用 

---
# CoLLMLight: Cooperative Large Language Model Agents for Network-Wide Traffic Signal Control 

**Title (ZH)**: CollMLight：协作的大语言模型代理在网络范围内的交通信号控制中应用 

**Authors**: Zirui Yuan, Siqi Lai, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11739)  

**Abstract**: Traffic Signal Control (TSC) plays a critical role in urban traffic management by optimizing traffic flow and mitigating congestion. While Large Language Models (LLMs) have recently emerged as promising tools for TSC due to their exceptional problem-solving and generalization capabilities, existing approaches fail to address the essential need for inter-agent coordination, limiting their effectiveness in achieving network-wide optimization. To bridge this gap, we propose CoLLMLight, a cooperative LLM agent framework for TSC. Specifically, we first construct a structured spatiotemporal graph to capture real-time traffic dynamics and spatial relationships among neighboring intersections, enabling the LLM to reason about complex traffic interactions. Moreover, we introduce a complexity-aware reasoning mechanism that dynamically adapts reasoning depth based on real-time traffic conditions, ensuring optimal computational efficiency without sacrificing decision quality. Besides, we propose a fine-tuning strategy that leverages iterative simulation-driven data collection and environmental feedback to build a lightweight LLM tailored for cooperative TSC. Extensive experiments on both synthetic and real-world datasets demonstrate that CoLLMLight outperforms state-of-the-art methods in diverse traffic scenarios, showcasing its effectiveness, scalability, and robustness. 

**Abstract (ZH)**: 交通信号控制（TSC）在城市交通管理中发挥着关键作用，通过优化交通流量和缓解拥堵。尽管大型语言模型（LLMs）因卓越的问题解决能力和通用化能力而近年来成为TSC的有前景工具，但现有方法未能满足代理间协调的必要需求，限制了它们在网络级优化方面的效果。为了弥补这一差距，我们提出了一种协同LLM代理框架CoLLMLight，用于TSC。具体地，我们首先构建了一个结构化的时空图，以捕捉实时交通动态和相邻交叉口之间的空间关系，使LLM能够推理复杂的交通交互。此外，我们引入了一种面向复杂性的推理机制，可根据实时交通条件动态调整推理深度，确保在不牺牲决策质量的情况下获得最优的计算效率。另外，我们提出了一种微调策略，利用迭代仿真驱动的数据收集和环境反馈来构建一个适用于协同TSC的轻量级LLM。在合成数据集和真实世界数据集上的广泛实验表明，CoLLMLight在多种交通场景中均优于现有方法，展示了其有效性、可扩展性和鲁棒性。 

---
# LLM Agents for Education: Advances and Applications 

**Title (ZH)**: 教育领域的大型语言模型代理：进展与应用 

**Authors**: Zhendong Chu, Shen Wang, Jian Xie, Tinghui Zhu, Yibo Yan, Jinheng Ye, Aoxiao Zhong, Xuming Hu, Jing Liang, Philip S. Yu, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11733)  

**Abstract**: Large Language Model (LLM) agents have demonstrated remarkable capabilities in automating tasks and driving innovation across diverse educational applications. In this survey, we provide a systematic review of state-of-the-art research on LLM agents in education, categorizing them into two broad classes: (1) \emph{Pedagogical Agents}, which focus on automating complex pedagogical tasks to support both teachers and students; and (2) \emph{Domain-Specific Educational Agents}, which are tailored for specialized fields such as science education, language learning, and professional development. We comprehensively examine the technological advancements underlying these LLM agents, including key datasets, benchmarks, and algorithmic frameworks that drive their effectiveness. Furthermore, we discuss critical challenges such as privacy, bias and fairness concerns, hallucination mitigation, and integration with existing educational ecosystems. This survey aims to provide a comprehensive technological overview of LLM agents for education, fostering further research and collaboration to enhance their impact for the greater good of learners and educators alike. 

**Abstract (ZH)**: 大型语言模型代理在教育领域的研究进展：面向教学的应用与专属性教育代理的系统综述 

---
# MEADOW: Memory-efficient Dataflow and Data Packing for Low Power Edge LLMs 

**Title (ZH)**: MEADOW: 节省内存的数据流和数据打包方法用于低功耗边缘端大语言模型 

**Authors**: Abhishek Moitra, Arkapravo Ghosh, Shrey Agarwal, Aporva Amarnath, Karthik Swaminathan, Priyadarshini Panda  

**Link**: [PDF](https://arxiv.org/pdf/2503.11663)  

**Abstract**: The computational and memory challenges of large language models (LLMs) have sparked several optimization approaches towards their efficient implementation. While prior LLM-targeted quantization, and prior works on sparse acceleration have significantly mitigated the memory and computation bottleneck, they do so assuming high power platforms such as GPUs and server-class FPGAs with large off-chip memory bandwidths and employ a generalized matrix multiplication (GEMM) execution of all the layers in the decoder. In such a GEMM-based execution, data is fetched from an off-chip memory, computed and stored back. However, at reduced off-chip memory capacities, as is the case with low-power edge devices, this implementation strategy significantly increases the attention computation latency owing to the repeated storage and fetch of large intermediate tokens to and from the off-chip memory. Moreover, fetching the weight matrices from a bandwidth constrained memory further aggravates the memory bottleneck problem. To this end, we introduce MEADOW, a framework that significantly reduces the off-chip memory access for LLMs with a novel token-parallel head-sequential (TPHS) dataflow. Additionally, MEADOW applies weight packing that performs loss-less decomposition of large weight matrices to their unique elements thereby, reducing the enormous weight fetch latency. MEADOW demonstrates 1.5x and 2.5x lower decode and prefill latency, respectively, compared to a GEMM-based LLM implementation on the low power Xilinx ZCU102 FPGA platform that consumes less than 10W. Additionally, MEADOW achieves an end-to-end latency improvement of over 40%, compared to prior LLM optimization works. 

**Abstract (ZH)**: 大型语言模型（LLM）的计算和内存挑战促使了多种高效实现的优化方法。尽管针对LLM的量化方法和稀疏加速的早期工作显著缓解了内存和计算瓶颈，它们主要针对高性能平台（如GPU和服务器级FPGA），这些平台具有大的片外内存带宽，并且所有解码层都采用通用矩阵乘法（GEMM）执行方式。在GEMM基础上的执行中，数据从片外内存中读取、计算后返回存储。但在片外内存容量减小，如低功耗边缘设备中，这一实现策略会因重复的存储和从片外内存读取大量中间令牌而显著增加注意力计算延迟。此外，从带宽受限的内存中获取权重矩阵进一步加剧了内存瓶颈问题。为此，我们提出了MEADOW框架，通过一种新颖的令牌并行头部顺序（TPHS）数据流显著减少了LLM的片外内存访问。此外，MEADOW采用了权重打包技术，对大型权重矩阵进行无损分解到其唯一元素，从而减少了巨大的权重获取延迟。MEADOW在消耗不到10W功率的低功耗Xilinx ZCU102 FPGA平台上，解码延迟和预填充延迟分别降低了1.5倍和2.5倍。与之前的LLM优化工作相比，MEADOW实现了端到端延迟提高了超过40%。 

---
# Explainable Sentiment Analysis with DeepSeek-R1: Performance, Efficiency, and Few-Shot Learning 

**Title (ZH)**: 基于DeepSeek-R1的可解释情感分析：性能、效率及少样本学习 

**Authors**: Donghao Huang, Zhaoxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11655)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly enhanced sentiment analysis capabilities. However, the trade-offs between model performance, efficiency, and explainability of some latest models remain underexplored. This study presents the first comprehensive evaluation of the DeepSeek-R1 series of models, reasoning open-source LLMs, for sentiment analysis, comparing them against OpenAI's GPT-4 and GPT-4-mini. We systematically analyze their performance under few-shot prompting conditions, scaling up to 50-shot configurations to assess in-context learning effectiveness. Our experiments reveal that DeepSeek-R1 demonstrates competitive accuracy, particularly in multi-class sentiment tasks, while offering enhanced interpretability through its detailed reasoning process. Additionally, we highlight the impact of increasing few-shot examples on model performance and discuss key trade-offs between explainability and computational efficiency. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）的发展显著提升了情感分析的能力，但一些最新模型在性能、效率和可解释性之间的权衡尚未被充分探索。本研究首次全面评估了DeepSeek-R1系列开源模型在情感分析中的表现，将其与OpenAI的GPT-4和GPT-4-mini进行比较。我们系统分析了这些模型在少量提示条件下的性能，并扩展到50-shot配置以评估上下文学习的有效性。实验结果显示，DeepSeek-R1在多类别情感任务中表现出竞争力，并通过详细的推理过程提高了可解释性。此外，我们强调了增加少量提示样本对模型性能的影响，并讨论了可解释性与计算效率之间的关键权衡。 

---
