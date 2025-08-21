# Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination 

**Title (ZH)**: LLM代理能否解决协作任务？一种基于紧迫性感知的规划与协调研究 

**Authors**: João Vitor de Carvalho Silva, Douglas G. Macharet  

**Link**: [PDF](https://arxiv.org/pdf/2508.14635)  

**Abstract**: The ability to coordinate actions across multiple agents is critical for solving complex, real-world problems. Large Language Models (LLMs) have shown strong capabilities in communication, planning, and reasoning, raising the question of whether they can also support effective collaboration in multi-agent settings. In this work, we investigate the use of LLM agents to solve a structured victim rescue task that requires division of labor, prioritization, and cooperative planning. Agents operate in a fully known graph-based environment and must allocate resources to victims with varying needs and urgency levels. We systematically evaluate their performance using a suite of coordination-sensitive metrics, including task success rate, redundant actions, room conflicts, and urgency-weighted efficiency. This study offers new insights into the strengths and failure modes of LLMs in physically grounded multi-agent collaboration tasks, contributing to future benchmarks and architectural improvements. 

**Abstract (ZH)**: 跨多个代理协调行动的能力对于解决复杂的真实世界问题是至关重要的。大规模语言模型（LLMs）在通信、规划和推理方面表现出强大的能力，引起了一个问题，即它们是否也能在多代理环境中支持有效的合作。在本研究中，我们探讨了使用LLM代理解决一个需要分工、优先处理和协同规划的结构化救援任务的可能性。代理在完全已知的图基环境中操作，并必须根据受害者的不同需求和紧迫性级别分配资源。我们使用一系列敏感于协调的指标系统性地评估其性能，包括任务成功率、冗余行动、房间冲突和紧迫性加权效率。这项研究为物理环境中多代理合作任务中LLMs的优势和失效模式提供了新的见解，贡献于未来的基准测试和架构改进。 

---
# DEXTER-LLM: Dynamic and Explainable Coordination of Multi-Robot Systems in Unknown Environments via Large Language Models 

**Title (ZH)**: DEXTER-LLM：通过大型语言模型在未知环境中的多机器人系统动态可解释协调 

**Authors**: Yuxiao Zhu, Junfeng Chen, Xintong Zhang, Meng Guo, Zhongkui Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14387)  

**Abstract**: Online coordination of multi-robot systems in open and unknown environments faces significant challenges, particularly when semantic features detected during operation dynamically trigger new tasks. Recent large language model (LLMs)-based approaches for scene reasoning and planning primarily focus on one-shot, end-to-end solutions in known environments, lacking both dynamic adaptation capabilities for online operation and explainability in the processes of planning. To address these issues, a novel framework (DEXTER-LLM) for dynamic task planning in unknown environments, integrates four modules: (i) a mission comprehension module that resolves partial ordering of tasks specified by natural languages or linear temporal logic formulas (LTL); (ii) an online subtask generator based on LLMs that improves the accuracy and explainability of task decomposition via multi-stage reasoning; (iii) an optimal subtask assigner and scheduler that allocates subtasks to robots via search-based optimization; and (iv) a dynamic adaptation and human-in-the-loop verification module that implements multi-rate, event-based updates for both subtasks and their assignments, to cope with new features and tasks detected online. The framework effectively combines LLMs' open-world reasoning capabilities with the optimality of model-based assignment methods, simultaneously addressing the critical issue of online adaptability and explainability. Experimental evaluations demonstrate exceptional performances, with 100% success rates across all scenarios, 160 tasks and 480 subtasks completed on average (3 times the baselines), 62% less queries to LLMs during adaptation, and superior plan quality (2 times higher) for compound tasks. Project page at this https URL 

**Abstract (ZH)**: 在未知环境中的多机器人系统在线协调面临显著挑战，特别是在操作过程中检测到的语义特征动态触发新任务时。基于大型语言模型（LLMs）的场景推理和规划方法主要集中在已知环境的一次性端到端解决方案上，缺乏在线操作的动态适应能力和规划过程中的可解释性。为了解决这些问题，提出了一种新的框架（DEXTER-LLM），用于未知环境中的动态任务规划，该框架整合了四个模块：（i）任务理解模块，解决由自然语言或线性时序逻辑公式（LTL）指定的任务的部分顺序；（ii）基于LLMs的在线子任务生成器，通过多阶段推理提高任务分解的准确性和可解释性；（iii）最优子任务分配和调度模块，通过基于搜索的优化将子任务分配给机器人；（iv）动态适应和人类在环验证模块，通过多速率、事件驱动更新同时处理在线检测到的新特征和任务。该框架有效结合了LLMs的开放世界推理能力和基于模型的分配方法的最优化，同时解决了在线适应性和可解释性这两个关键问题。实验评估显示，该框架在所有场景下均取得100%的成功率，平均完成160个任务和480个子任务，适应过程中LLMs查询量减少62%，多任务规划质量提高2倍。项目页面详见此链接。 

---
# Privileged Self-Access Matters for Introspection in AI 

**Title (ZH)**: 特权自我访问对于AI的反省至关重要 

**Authors**: Siyuan Song, Harvey Lederman, Jennifer Hu, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2508.14802)  

**Abstract**: Whether AI models can introspect is an increasingly important practical question. But there is no consensus on how introspection is to be defined. Beginning from a recently proposed ''lightweight'' definition, we argue instead for a thicker one. According to our proposal, introspection in AI is any process which yields information about internal states through a process more reliable than one with equal or lower computational cost available to a third party. Using experiments where LLMs reason about their internal temperature parameters, we show they can appear to have lightweight introspection while failing to meaningfully introspect per our proposed definition. 

**Abstract (ZH)**: AI模型能否内省：一个日益重要的实践问题及其定义探讨 

---
# MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers 

**Title (ZH)**: MCP-宇宙：基于真实模型上下文协议服务器评估大型语言模型 

**Authors**: Ziyang Luo, Zhiqi Shen, Wenzhuo Yang, Zirui Zhao, Prathyusha Jwalapuram, Amrita Saha, Doyen Sahoo, Silvio Savarese, Caiming Xiong, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14704)  

**Abstract**: The Model Context Protocol has emerged as a transformative standard for connecting large language models to external data sources and tools, rapidly gaining adoption across major AI providers and development platforms. However, existing benchmarks are overly simplistic and fail to capture real application challenges such as long-horizon reasoning and large, unfamiliar tool spaces. To address this critical gap, we introduce MCP-Universe, the first comprehensive benchmark specifically designed to evaluate LLMs in realistic and hard tasks through interaction with real-world MCP servers. Our benchmark encompasses 6 core domains spanning 11 different MCP servers: Location Navigation, Repository Management, Financial Analysis, 3D Design, Browser Automation, and Web Searching. To ensure rigorous evaluation, we implement execution-based evaluators, including format evaluators for agent format compliance, static evaluators for time-invariant content matching, and dynamic evaluators that automatically retrieve real-time ground truth for temporally sensitive tasks. Through extensive evaluation of leading LLMs, we find that even SOTA models such as GPT-5 (43.72%), Grok-4 (33.33%) and Claude-4.0-Sonnet (29.44%) exhibit significant performance limitations. In addition, our benchmark poses a significant long-context challenge for LLM agents, as the number of input tokens increases rapidly with the number of interaction steps. Moreover, it introduces an unknown-tools challenge, as LLM agents often lack familiarity with the precise usage of the MCP servers. Notably, enterprise-level agents like Cursor cannot achieve better performance than standard ReAct frameworks. Beyond evaluation, we open-source our extensible evaluation framework with UI support, enabling researchers and practitioners to seamlessly integrate new agents and MCP servers while fostering innovation in the rapidly evolving MCP ecosystem. 

**Abstract (ZH)**: MCP-Universe：首个全面评估大规模语言模型在现实和复杂任务中与真实MCP服务器交互的基准 

---
# Entropy-Constrained Strategy Optimization in Urban Floods: A Multi-Agent Framework with LLM and Knowledge Graph Integration 

**Title (ZH)**: 基于LLM和知识图谱集成的Urban Floods熵约束策略优化多Agent框架 

**Authors**: Peilin Ji, Xiao Xue, Simeng Wang, Wenhao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14654)  

**Abstract**: In recent years, the increasing frequency of extreme urban rainfall events has posed significant challenges to emergency scheduling systems. Urban flooding often leads to severe traffic congestion and service disruptions, threatening public safety and mobility. However, effective decision making remains hindered by three key challenges: (1) managing trade-offs among competing goals (e.g., traffic flow, task completion, and risk mitigation) requires dynamic, context-aware strategies; (2) rapidly evolving environmental conditions render static rules inadequate; and (3) LLM-generated strategies frequently suffer from semantic instability and execution inconsistency. Existing methods fail to align perception, global optimization, and multi-agent coordination within a unified framework. To tackle these challenges, we introduce H-J, a hierarchical multi-agent framework that integrates knowledge-guided prompting, entropy-constrained generation, and feedback-driven optimization. The framework establishes a closed-loop pipeline spanning from multi-source perception to strategic execution and continuous refinement. We evaluate H-J on real-world urban topology and rainfall data under three representative conditions: extreme rainfall, intermittent bursts, and daily light rain. Experiments show that H-J outperforms rule-based and reinforcement-learning baselines in traffic smoothness, task success rate, and system robustness. These findings highlight the promise of uncertainty-aware, knowledge-constrained LLM-based approaches for enhancing resilience in urban flood response. 

**Abstract (ZH)**: 近年来，极端城市降雨事件频率的增加对应急调度系统提出了重大挑战。城市洪涝常常导致严重的交通拥堵和服务中断，威胁公共安全和流动性。然而，有效的决策仍受到三项关键挑战的阻碍：（1）在处理互斥目标（如交通流量、任务完成和风险缓解）之间的权衡时，需要动态且情境感知的策略；（2）快速变化的环境条件使静态规则变得不适用；（3）由大型语言模型生成的策略经常遭受语义不稳定和执行不一致的问题。现有方法未能在统一框架内对齐感知、全局优化和多智能体协调。为应对这些挑战，我们提出了一种层次化的多智能体框架H-J，该框架整合了基于知识的提示、 entropy约束生成和反馈驱动的优化。该框架建立了一个从多源感知到战略执行并持续改进的闭环管道。我们使用真实的都市拓扑和降雨数据，在三种代表性条件下（极端降雨、间歇性突发降雨和日常小雨）评估了H-J。实验结果显示，H-J在交通顺畅性、任务成功率和系统鲁棒性方面均优于基于规则和强化学习的基准方法。这些发现突显了不确定性意识和知识约束的大规模语言模型方法在增强城市洪涝应对韧性方面的潜力。 

---
# Who Sees What? Structured Thought-Action Sequences for Epistemic Reasoning in LLMs 

**Title (ZH)**: 见所未见？面向LLM的epistemic推理的结构化思维-行动序列 

**Authors**: Luca Annese, Sabrina Patania, Silvia Serino, Tom Foulsham, Silvia Rossi, Azzurra Ruggeri, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2508.14564)  

**Abstract**: Recent advances in large language models (LLMs) and reasoning frameworks have opened new possibilities for improving the perspective -taking capabilities of autonomous agents. However, tasks that involve active perception, collaborative reasoning, and perspective taking (understanding what another agent can see or knows) pose persistent challenges for current LLM-based systems. This study investigates the potential of structured examples derived from transformed solution graphs generated by the Fast Downward planner to improve the performance of LLM-based agents within a ReAct framework. We propose a structured solution-processing pipeline that generates three distinct categories of examples: optimal goal paths (G-type), informative node paths (E-type), and step-by-step optimal decision sequences contrasting alternative actions (L-type). These solutions are further converted into ``thought-action'' examples by prompting an LLM to explicitly articulate the reasoning behind each decision. While L-type examples slightly reduce clarification requests and overall action steps, they do not yield consistent improvements. Agents are successful in tasks requiring basic attentional filtering but struggle in scenarios that required mentalising about occluded spaces or weighing the costs of epistemic actions. These findings suggest that structured examples alone are insufficient for robust perspective-taking, underscoring the need for explicit belief tracking, cost modelling, and richer environments to enable socially grounded collaboration in LLM-based agents. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）和推理框架为提高自主代理的视角转换能力开拓了新可能性。然而，涉及主动感知、协作推理和视角转换（理解另一代理所能见或所知的内容）的任务仍给当前基于LLM的系统带来持续挑战。本研究探讨了从Fast Downward规划器生成的变换解图谱中提取的结构化示例的潜力，以改善基于LLM的代理在ReAct框架内的性能。我们提出了一种结构化解决方案处理流水线，生成三种不同类型示例：最优目标路径（G类）、有信息节点路径（E类），以及对比替代行动的逐步最优决策序列（L类）。进一步通过提示LLM明确阐述每个决策背后的推理，将这些解决方案转换为“思考-行动”示例。虽然L类示例略微减少了澄清请求和总体行动步骤，但并未带来一致的性能改进。代理在需要基本注意力筛选的任务中表现良好，但在涉及预测遮挡空间或权衡认知行动成本的场景中则表现不佳。这些发现表明，仅依靠结构化示例不足以实现稳健的视角转换，突显了显式信念跟踪、成本建模以及更丰富的环境在使基于LLM的代理实现社会性合作方面的重要性。 

---
# Automated Optimization Modeling through Expert-Guided Large Language Model Reasoning 

**Title (ZH)**: 通过专家引导的大语言模型推理实现自动化优化建模 

**Authors**: Beinuo Yang, Qishen Zhou, Junyi Li, Xingchen Su, Simon Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14410)  

**Abstract**: Optimization Modeling (OM) is essential for solving complex decision-making problems. However, the process remains time-consuming and error-prone, heavily relying on domain experts. While Large Language Models (LLMs) show promise in addressing these challenges through their natural language understanding and reasoning capabilities, current approaches face three critical limitations: high benchmark labeling error rates reaching up to 42\%, narrow evaluation scope that only considers optimal values, and computational inefficiency due to heavy reliance on multi-agent systems or model fine-tuning. In this work, we first enhance existing datasets through systematic error correction and more comprehensive annotation. Additionally, we introduce LogiOR, a new optimization modeling benchmark from the logistics domain, containing more complex problems with standardized annotations. Furthermore, we present ORThought, a novel framework that leverages expert-level optimization modeling principles through chain-of-thought reasoning to automate the OM process. Through extensive empirical evaluation, we demonstrate that ORThought outperforms existing approaches, including multi-agent frameworks, with particularly significant advantages on complex optimization problems. Finally, we provide a systematic analysis of our method, identifying critical success factors and failure modes, providing valuable insights for future research on LLM-based optimization modeling. 

**Abstract (ZH)**: 优化建模（OM）是解决复杂决策问题的关键。然而，这一过程依然耗时且易出错，严重依赖领域专家。尽管大型语言模型（LLMs）凭借其自然语言理解和推理能力有望解决这些挑战，当前的方法仍面临三大关键局限：基准标注错误率高达42%，评估范围狭窄仅考虑最优值，以及由于过度依赖多智能体系统或模型微调而导致的计算效率低下。在本项研究中，我们首先通过系统性的错误修正和更加全面的标注来提升现有数据集。此外，我们引入了LogiOR，一个源自物流领域的新型优化建模基准，包含更多复杂且标准化的问题。进一步地，我们提出了ORThought框架，该框架利用链式推理实现专家级优化建模原则，以自动化OM过程。通过广泛的实证评估，我们证明ORThought在复杂的优化问题上优于现有方法，特别是在复杂优化问题上显示出显著优势。最后，我们对我们的方法进行了系统的分析，明确了关键成功因素和失败模式，为基于LLM的优化建模未来研究提供了有价值见解。 

---
# Explaining Hitori Puzzles: Neurosymbolic Proof Staging for Sequential Decisions 

**Title (ZH)**: 解释Hitori数独：神经符号证明舞台化用于序列决策 

**Authors**: Maria Leonor Pacheco, Fabio Somenzi, Dananjay Srinivas, Ashutosh Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14294)  

**Abstract**: We propose a neurosymbolic approach to the explanation of complex sequences of decisions that combines the strengths of decision procedures and Large Language Models (LLMs). We demonstrate this approach by producing explanations for the solutions of Hitori puzzles. The rules of Hitori include local constraints that are effectively explained by short resolution proofs. However, they also include a connectivity constraint that is more suitable for visual explanations. Hence, Hitori provides an excellent testing ground for a flexible combination of SAT solvers and LLMs. We have implemented a tool that assists humans in solving Hitori puzzles, and we present experimental evidence of its effectiveness. 

**Abstract (ZH)**: 我们提出一种结合决策过程和大型语言模型优势的神经符号方法来解释复杂的决策序列，并通过生成九宮格谜题的解决方案解释来演示该方法。九宮格谜题的规则包括有效的局部约束，这些约束可以通过简短的归结证明来解释，但也包括一个更适合视觉解释的连通性约束。因此，九宮格谜题为灵活结合SAT求解器和大型语言模型提供了一个优秀的测试平台。我们实现了一个辅助人类解决九宮格谜题的工具，并展示了其有效性的实验证据。 

---
# Large Language Models are Highly Aligned with Human Ratings of Emotional Stimuli 

**Title (ZH)**: 大型语言模型与人类对情绪刺激的评价高度一致。 

**Authors**: Mattson Ogg, Chace Ashcraft, Ritwik Bose, Raphael Norman-Tenazas, Michael Wolmetz  

**Link**: [PDF](https://arxiv.org/pdf/2508.14214)  

**Abstract**: Emotions exert an immense influence over human behavior and cognition in both commonplace and high-stress tasks. Discussions of whether or how to integrate large language models (LLMs) into everyday life (e.g., acting as proxies for, or interacting with, human agents), should be informed by an understanding of how these tools evaluate emotionally loaded stimuli or situations. A model's alignment with human behavior in these cases can inform the effectiveness of LLMs for certain roles or interactions. To help build this understanding, we elicited ratings from multiple popular LLMs for datasets of words and images that were previously rated for their emotional content by humans. We found that when performing the same rating tasks, GPT-4o responded very similarly to human participants across modalities, stimuli and most rating scales (r = 0.9 or higher in many cases). However, arousal ratings were less well aligned between human and LLM raters, while happiness ratings were most highly aligned. Overall LLMs aligned better within a five-category (happiness, anger, sadness, fear, disgust) emotion framework than within a two-dimensional (arousal and valence) organization. Finally, LLM ratings were substantially more homogenous than human ratings. Together these results begin to describe how LLM agents interpret emotional stimuli and highlight similarities and differences among biological and artificial intelligence in key behavioral domains. 

**Abstract (ZH)**: 情绪对人类行为和认知在这方面和高压任务中均产生巨大影响。关于是否以及如何将大型语言模型（LLMs）融入日常生活（例如作为人类代理的代理或与其交互）的讨论，应基于对其如何评估情绪化刺激或情况的理解。模型在这些情况下的行为一致性可以影响LLMs在特定角色或交互中的有效性。为了帮助构建这种理解，我们对之前由人类评定情绪内容的词汇和图像数据集进行了多个流行LLMs的评分评估。我们发现，当执行相同评分任务时，GPT-4在不同模态、刺激物和大多数评分尺度上的响应与人类参与者非常相似（许多情况下相关系数r为0.9或更高）。然而，唤醒度评分的人机一致性较差，而快乐度评分一致性最高。总体而言，LLMs在五类（快乐、愤怒、悲伤、恐惧、厌恶）情绪框架下的表现一致性较好，而在二维（唤醒度和价值）组织下则较差。最后，LLMs的评分显著比人类评分更加一致。这些结果开始描述了LLM代理如何解释情绪化刺激，并突显了生物和人工智之间在关键行为领域中的相似性和差异性。 

---
# Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs 

**Title (ZH)**: 量化邂逅dLLMs：Diffusion大语言模型后训练量化系统的研究 

**Authors**: Haokun Lin, Haobo Xu, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14896)  

**Abstract**: Recent advances in diffusion large language models (dLLMs) have introduced a promising alternative to autoregressive (AR) LLMs for natural language generation tasks, leveraging full attention and denoising-based decoding strategies. However, the deployment of these models on edge devices remains challenging due to their massive parameter scale and high resource demands. While post-training quantization (PTQ) has emerged as a widely adopted technique for compressing AR LLMs, its applicability to dLLMs remains largely unexplored. In this work, we present the first systematic study on quantizing diffusion-based language models. We begin by identifying the presence of activation outliers, characterized by abnormally large activation values that dominate the dynamic range. These outliers pose a key challenge to low-bit quantization, as they make it difficult to preserve precision for the majority of values. More importantly, we implement state-of-the-art PTQ methods and conduct a comprehensive evaluation across multiple task types and model variants. Our analysis is structured along four key dimensions: bit-width, quantization method, task category, and model type. Through this multi-perspective evaluation, we offer practical insights into the quantization behavior of dLLMs under different configurations. We hope our findings provide a foundation for future research in efficient dLLM deployment. All codes and experimental setups will be released to support the community. 

**Abstract (ZH)**: 近期在扩散大语言模型（dLLMs）方面取得的进展为自然语言生成任务提供了autoregressive（AR）大语言模型的有前途的替代方案，利用了全注意力和去噪解码策略。然而，由于这些模型参数量庞大和高资源需求，它们在边缘设备上的部署仍然具有挑战性。尽管后训练量化（PTQ）已成为压缩AR大语言模型的广泛应用技术，但其在dLLMs上的适用性仍鲜有研究。在这项工作中，我们进行了首个系统性研究，以量化扩散基础语言模型。我们首先识别出异常激活值的存在，这些值异常大且主导了动态范围。这些异常值对低比特量化构成关键挑战，因为它们使得很难保留大多数值的精度。更重要的是，我们实现了最先进的PTQ方法，并对多种任务类型和模型变体进行了全面评估。我们的分析沿四个关键维度进行：比特宽度、量化方法、任务类别和模型类型。通过这种多视角评估，我们提供了关于不同配置下dLLMs量化行为的实用见解。我们希望我们的研究结果为未来高效dLLM部署的研究奠定基础。所有代码和实验设置将向社区公开。 

---
# Long Chain-of-Thought Reasoning Across Languages 

**Title (ZH)**: 跨语言长链推理 

**Authors**: Josh Barua, Seun Eisape, Kayo Yin, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2508.14828)  

**Abstract**: Scaling inference through long chains-of-thought (CoTs) has unlocked impressive reasoning capabilities in large language models (LLMs), yet the reasoning process remains almost exclusively English-centric. We construct translated versions of two popular English reasoning datasets, fine-tune Qwen 2.5 (7B) and Qwen 3 (8B) models, and present a systematic study of long CoT generation across French, Japanese, Latvian, and Swahili. Our experiments reveal three key findings. First, the efficacy of using English as a pivot language varies by language: it provides no benefit for French, improves performance when used as the reasoning language for Japanese and Latvian, and proves insufficient for Swahili where both task comprehension and reasoning remain poor. Second, extensive multilingual pretraining in Qwen 3 narrows but does not eliminate the cross-lingual performance gap. A lightweight fine-tune using only 1k traces still improves performance by over 30\% in Swahili. Third, data quality versus scale trade-offs are language dependent: small, carefully curated datasets suffice for English and French, whereas larger but noisier corpora prove more effective for Swahili and Latvian. Together, these results clarify when and why long CoTs transfer across languages and provide translated datasets to foster equitable multilingual reasoning research. 

**Abstract (ZH)**: 通过长链思考（CoTs）扩展推理缩放能力已经在大型语言模型（LLMs）中解锁了令人印象深刻的推理能力，但推理过程几乎完全以英语为中心。我们构建了两个流行英语推理数据集的翻译版本，对Qwen 2.5（7B）和Qwen 3（8B）模型进行微调，并对法语、日语、拉脱维亚语和斯瓦希里语的长CoT生成进行了系统性研究。我们的实验揭示了三个关键发现。首先，使用英语作为中介语言的有效性因语言而异：它对法语没有益处，在使用英语作为推理语言时可以提高日语和拉脱维亚语的表现，但在斯瓦希里语中则不足，因为任务理解和推理表现都较差。其次，Qwen 3中的 extensive 多语言预训练缩小了但未能消除跨语言性能差距。仅使用1k轨迹的轻量级微调在斯瓦希里语中仍然能将性能提高超过30%。第三，数据质量与规模之间的权衡因语言而异：对英语和法语而言，少量但精挑细选的数据集就足够，而对斯瓦希里语和拉脱维亚语而言，虽然数据集更大但噪声更多，效果更佳。总之，这些结果阐明了长CoTs在不同语言间转移的时间和原因，并提供了翻译数据集以促进公平的多语言推理研究。 

---
# Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs 

**Title (ZH)**: 评价检索增强生成与长上下文输入在电子健康记录临床推理中的性能对比 

**Authors**: Skatje Myers, Dmitriy Dligach, Timothy A. Miller, Samantha Barr, Yanjun Gao, Matthew Churpek, Anoop Mayampurath, Majid Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2508.14817)  

**Abstract**: Electronic health records (EHRs) are long, noisy, and often redundant, posing a major challenge for the clinicians who must navigate them. Large language models (LLMs) offer a promising solution for extracting and reasoning over this unstructured text, but the length of clinical notes often exceeds even state-of-the-art models' extended context windows. Retrieval-augmented generation (RAG) offers an alternative by retrieving task-relevant passages from across the entire EHR, potentially reducing the amount of required input tokens. In this work, we propose three clinical tasks designed to be replicable across health systems with minimal effort: 1) extracting imaging procedures, 2) generating timelines of antibiotic use, and 3) identifying key diagnoses. Using EHRs from actual hospitalized patients, we test three state-of-the-art LLMs with varying amounts of provided context, using either targeted text retrieval or the most recent clinical notes. We find that RAG closely matches or exceeds the performance of using recent notes, and approaches the performance of using the models' full context while requiring drastically fewer input tokens. Our results suggest that RAG remains a competitive and efficient approach even as newer models become capable of handling increasingly longer amounts of text. 

**Abstract (ZH)**: 电子健康记录中的图像检查提取、抗生素使用时间线生成及关键诊断识别：基于检索增强生成的方法研究 

---
# TransLLM: A Unified Multi-Task Foundation Framework for Urban Transportation via Learnable Prompting 

**Title (ZH)**: TransLLM: 一种基于可学习提示的统一多任务基础框架城市交通 

**Authors**: Jiaming Leng, Yunying Bi, Chuan Qin, Bing Yin, Yanyong Zhang, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14782)  

**Abstract**: Urban transportation systems encounter diverse challenges across multiple tasks, such as traffic forecasting, electric vehicle (EV) charging demand prediction, and taxi dispatch. Existing approaches suffer from two key limitations: small-scale deep learning models are task-specific and data-hungry, limiting their generalizability across diverse scenarios, while large language models (LLMs), despite offering flexibility through natural language interfaces, struggle with structured spatiotemporal data and numerical reasoning in transportation domains. To address these limitations, we propose TransLLM, a unified foundation framework that integrates spatiotemporal modeling with large language models through learnable prompt composition. Our approach features a lightweight spatiotemporal encoder that captures complex dependencies via dilated temporal convolutions and dual-adjacency graph attention networks, seamlessly interfacing with LLMs through structured embeddings. A novel instance-level prompt routing mechanism, trained via reinforcement learning, dynamically personalizes prompts based on input characteristics, moving beyond fixed task-specific templates. The framework operates by encoding spatiotemporal patterns into contextual representations, dynamically composing personalized prompts to guide LLM reasoning, and projecting the resulting representations through specialized output layers to generate task-specific predictions. Experiments across seven datasets and three tasks demonstrate the exceptional effectiveness of TransLLM in both supervised and zero-shot settings. Compared to ten baseline models, it delivers competitive performance on both regression and planning problems, showing strong generalization and cross-task adaptability. Our code is available at this https URL. 

**Abstract (ZH)**: 城市交通系统在交通预测、电动车辆充电需求预测和出租车调度等多个任务中面临多样化的挑战。现有方法存在两个关键限制：小规模的深度学习模型任务特定且数据需求高，限制了其在多样化场景中的泛化能力；而大型语言模型虽然通过自然语言界面提供了灵活性，但在交通领域的结构化时空数据和数值推理方面仍存在困难。为解决这些限制，我们提出了一种名为TransLLM的统一基础框架，该框架通过可学习的提示组合将时空建模与大型语言模型相结合。该方法的特点是一个轻量级的时空编码器，通过膨胀时间卷积和双重邻接图注意力网络捕捉复杂的依赖关系，并通过结构化嵌入无缝地与大型语言模型接口。一个新颖的实例级提示路由机制，通过强化学习训练，根据输入特征动态个性化提示，超越了固定的任务特定模板。该框架通过编码时空模式为上下文表示，动态组合个性化的提示以引导大型语言模型推理，并通过专业化输出层投影结果表示以生成特定任务的预测。跨七个数据集和三个任务的实验在监督和零-shot设置中均证明了TransLLM的卓越效果。与十个基线模型相比，它在回归和规划问题上表现竞争力强，显示了强大的泛化能力和跨任务适应性。代码可在以下链接获取：this https URL。 

---
# PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning 

**Title (ZH)**: PepThink-R1：具有解释性循环肽优化的CoT SFT和强化学习大语言模型 

**Authors**: Ruheng Wang, Hang Zhang, Trieu Nguyen, Shasha Feng, Hao-Wei Pang, Xiang Yu, Li Xiao, Peter Zhiping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14765)  

**Abstract**: Designing therapeutic peptides with tailored properties is hindered by the vastness of sequence space, limited experimental data, and poor interpretability of current generative models. To address these challenges, we introduce PepThink-R1, a generative framework that integrates large language models (LLMs) with chain-of-thought (CoT) supervised fine-tuning and reinforcement learning (RL). Unlike prior approaches, PepThink-R1 explicitly reasons about monomer-level modifications during sequence generation, enabling interpretable design choices while optimizing for multiple pharmacological properties. Guided by a tailored reward function balancing chemical validity and property improvements, the model autonomously explores diverse sequence variants. We demonstrate that PepThink-R1 generates cyclic peptides with significantly enhanced lipophilicity, stability, and exposure, outperforming existing general LLMs (e.g., GPT-5) and domain-specific baseline in both optimization success and interpretability. To our knowledge, this is the first LLM-based peptide design framework that combines explicit reasoning with RL-driven property control, marking a step toward reliable and transparent peptide optimization for therapeutic discovery. 

**Abstract (ZH)**: 设计具有定制属性的治疗肽受到序列空间 vastness、有限的实验数据以及当前生成模型解释性差的阻碍。为应对这些挑战，我们引入了PepThink-R1，这是一种结合了大型语言模型（LLMs）、链式思考（CoT）监督微调和强化学习（RL）的生成框架。与之前的方法不同，PepThink-R1 在序列生成过程中明确地考虑单体级修改，使设计选择具有可解释性并优化多种药理学属性。在平衡化学有效性和属性改进的定制奖励函数引导下，该模型自主探索多种序列变体。我们证明，与现有的通用LLM（如GPT-5）和特定领域基线相比，PepThink-R1 在优化成功率和可解释性方面表现更优。据我们所知，这是第一个结合显式推理和基于RL的属性控制的LLM肽设计框架，朝着可靠和透明的治疗肽优化迈出了重要一步。 

---
# Evaluating Multilingual and Code-Switched Alignment in LLMs via Synthetic Natural Language Inference 

**Title (ZH)**: 评价多语言和代码切换对齐在大语言模型中的合成自然语言推理评估 

**Authors**: Samir Abdaljalil, Erchin Serpedin, Khalid Qaraqe, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2508.14735)  

**Abstract**: Large language models (LLMs) are increasingly applied in multilingual contexts, yet their capacity for consistent, logically grounded alignment across languages remains underexplored. We present a controlled evaluation framework for multilingual natural language inference (NLI) that generates synthetic, logic-based premise-hypothesis pairs and translates them into a typologically diverse set of languages. This design enables precise control over semantic relations and allows testing in both monolingual and mixed-language (code-switched) conditions. Surprisingly, code-switching does not degrade, and can even improve, performance, suggesting that translation-induced lexical variation may serve as a regularization signal. We validate semantic preservation through embedding-based similarity analyses and cross-lingual alignment visualizations, confirming the fidelity of translated pairs. Our findings expose both the potential and the brittleness of current LLM cross-lingual reasoning, and identify code-switching as a promising lever for improving multilingual robustness. Code available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在多语言场景中的应用日益增多，但其在不同语言中保持一致性和逻辑一致性的能力尚未得到充分探索。我们提出了一种控制性评估框架，用于多语言自然语言推理（NLI），生成合成的、基于逻辑的前提-假设对，并将其翻译成类型多样的一系列语言。这一设计允许对语义关系进行精确控制，并可以在单一语言和混合语言（代码转换）条件下进行测试。令人惊讶的是，代码转换不会降低性能，甚至可能提升性能，这表明翻译引起的词汇变化可能作为正则化信号发挥作用。我们通过嵌入式相似性分析和跨语言对齐可视化验证了语义保存的可靠性，证实了翻译后对齐的准确性。我们的研究揭示了当前LLM跨语言推理的潜力和脆弱性，并指出代码转换是提高多语言鲁棒性的有前景的杠杆。代码可在以下地址获得：this https URL。 

---
# Transplant Then Regenerate: A New Paradigm for Text Data Augmentation 

**Title (ZH)**: 移植然后再生：一种新的文本数据扩增范式 

**Authors**: Guangzhan Wang, Hongyu Zhang, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14723)  

**Abstract**: Data augmentation is a critical technique in deep learning. Traditional methods like Back-translation typically focus on lexical-level rephrasing, which primarily produces variations with the same semantics. While large language models (LLMs) have enhanced text augmentation by their "knowledge emergence" capability, controlling the style and structure of these outputs remains challenging and requires meticulous prompt engineering. In this paper, we propose LMTransplant, a novel text augmentation paradigm leveraging LLMs. The core idea of LMTransplant is transplant-then-regenerate: incorporating seed text into a context expanded by LLM, and asking the LLM to regenerate a variant based on the expanded context. This strategy allows the model to create more diverse and creative content-level variants by fully leveraging the knowledge embedded in LLMs, while preserving the core attributes of the original text. We evaluate LMTransplant across various text-related tasks, demonstrating its superior performance over existing text augmentation methods. Moreover, LMTransplant demonstrates exceptional scalability as the size of augmented data grows. 

**Abstract (ZH)**: 数据扩充是深度学习中的一种关键技术。传统方法如反向翻译通常专注于词汇层面的重写，主要产生具有相同语义的变化。虽然大规模语言模型（LLMs）通过其“知识涌现”能力增强了文本扩充能力，但在控制这些输出的风格和结构方面仍具有挑战性，需要精细的提示工程。本文提出了一种名为LMTransplant的新型文本扩充范式，利用LLMs。LMTransplant的核心思想是移植-再生：将种子文本融入由LLM扩大的上下文中，要求LLM基于扩大的上下文生成一个变体。这种策略允许模型充分利用LLMs中嵌入的知识来创建更多样化和创造性的内容级变体，同时保留原始文本的核心属性。我们对LMTransplant在各种文本相关任务中进行了评估，展示了其相对于现有文本扩充方法的优越性能。此外，LMTransplant在扩充数据量增大时表现出色。 

---
# Towards LLM-generated explanations for Component-based Knowledge Graph Question Answering Systems 

**Title (ZH)**: 面向组件知识图问答系统生成的LLM解释研究 

**Authors**: Dennis Schiese, Aleksandr Perevalov, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2508.14553)  

**Abstract**: Over time, software systems have reached a level of complexity that makes it difficult for their developers and users to explain particular decisions made by them. In this paper, we focus on the explainability of component-based systems for Question Answering (QA). These components often conduct processes driven by AI methods, in which behavior and decisions cannot be clearly explained or justified, s.t., even for QA experts interpreting the executed process and its results is hard. To address this challenge, we present an approach that considers the components' input and output data flows as a source for representing the behavior and provide explanations for the components, enabling users to comprehend what happened. In the QA framework used here, the data flows of the components are represented as SPARQL queries (inputs) and RDF triples (outputs). Hence, we are also providing valuable insights on verbalization regarding these data types. In our experiments, the approach generates explanations while following template-based settings (baseline) or via the use of Large Language Models (LLMs) with different configurations (automatic generation). Our evaluation shows that the explanations generated via LLMs achieve high quality and mostly outperform template-based approaches according to the users' ratings. Therefore, it enables us to automatically explain the behavior and decisions of QA components to humans while using RDF and SPARQL as a context for explanations. 

**Abstract (ZH)**: 基于组件的问答系统解释性研究 

---
# Adaptively Robust LLM Inference Optimization under Prediction Uncertainty 

**Title (ZH)**: 自适应鲁棒的大语言模型推理优化在预测不确定性下 

**Authors**: Zixi Chen, Yinyu Ye, Zijie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.14544)  

**Abstract**: We study the problem of optimizing Large Language Model (LLM) inference scheduling to minimize total latency. LLM inference is an online and multi-task service process and also heavily energy consuming by which a pre-trained LLM processes input requests and generates output tokens sequentially. Therefore, it is vital to improve its scheduling efficiency and reduce the power consumption while a great amount of prompt requests are arriving. A key challenge in LLM inference scheduling is that while the prompt length is known upon arrival, the output length, which critically impacts memory usage and processing time, is unknown. To address this uncertainty, we propose algorithms that leverage machine learning to predict output lengths, assuming the prediction provides an interval classification (min-max range) for each request.
We first design a conservative algorithm, $\mathcal{A}_{\max}$, which schedules requests based on the upper bound of predicted output lengths to prevent memory overflow. However, this approach is overly conservative: as prediction accuracy decreases, performance degrades significantly due to potential overestimation. To overcome this limitation, we propose $\mathcal{A}_{\min}$, an adaptive algorithm that initially treats the predicted lower bound as the output length and dynamically refines this estimate during inferencing. We prove that $\mathcal{A}_{\min}$ achieves a log-scale competitive ratio. Through numerical simulations, we demonstrate that $\mathcal{A}_{\min}$ often performs nearly as well as the hindsight scheduler, highlighting both its efficiency and robustness in practical scenarios. Moreover, $\mathcal{A}_{\min}$ relies solely on the lower bound of the prediction interval--an advantageous design choice since upper bounds on output length are typically more challenging to predict accurately. 

**Abstract (ZH)**: 我们研究了大型语言模型（LLM）推理调度优化问题，以最小化总延迟。我们探讨了一种在线且多任务的服务过程，通过预训练的LLM顺序处理输入请求并生成输出标记，这一过程高度能耗。因此，在大量提示请求到达时，提高调度效率和降低能耗至关重要。LLM推理调度的一个关键挑战在于，在请求到达时已知提示长度，但关键的输出长度未知，这直接影响内存使用和处理时间。为应对这种不确定性，我们提出了利用机器学习预测输出长度的算法，假设预测提供每个请求的区间分类（最小值-最大值范围）。

我们首先设计了一个保守算法$\mathcal{A}_{\max}$，该算法基于预测输出长度的上限进行调度，以防止内存溢出。然而，这种方法过于保守：随着预测准确性的下降，由于潜在的过度估计，性能会显著下降。为克服这一限制，我们提出了一种自适应算法$\mathcal{A}_{\min}$，该算法初始时将预测的下界作为输出长度，并在推理过程中动态修正这一估计。我们证明了$\mathcal{A}_{\min}$在对数尺度下具有竞争力的比例。通过数值仿真，我们证明了$\mathcal{A}_{\min}$通常能接近事后调度的表现，突显了其在实际场景中的高效性和稳健性。此外，$\mathcal{A}_{\min}$仅依赖预测区间的下界——这是一个有利的设计选择，因为输出长度的上限通常更难以准确预测。 

---
# Post-hoc LLM-Supported Debugging of Distributed Processes 

**Title (ZH)**: Post-hoc LLM-Supported Debugging of Distributed Processes 

**Authors**: Dennis Schiese, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2508.14540)  

**Abstract**: In this paper, we address the problem of manual debugging, which nowadays remains resource-intensive and in some parts archaic. This problem is especially evident in increasingly complex and distributed software systems. Therefore, our objective of this work is to introduce an approach that can possibly be applied to any system, at both the macro- and micro-level, to ease this debugging process. This approach utilizes a system's process data, in conjunction with generative AI, to generate natural-language explanations. These explanations are generated from the actual process data, interface information, and documentation to guide the developers more efficiently to understand the behavior and possible errors of a process and its sub-processes. Here, we present a demonstrator that employs this approach on a component-based Java system. However, our approach is language-agnostic. Ideally, the generated explanations will provide a good understanding of the process, even if developers are not familiar with all the details of the considered system. Our demonstrator is provided as an open-source web application that is freely accessible to all users. 

**Abstract (ZH)**: 本文address了手动调试这一现今仍然资源密集且在某些方面过时的问题，特别是在日益复杂和分布式的软件系统中尤为明显。因此，本文的工作目标是介绍一种可以在宏观和微观层面上应用于任何系统的办法，以简化这一调试过程。该方法利用系统的过程数据和生成式AI，生成自然语言解释。这些解释基于实际的过程数据、接口信息和文档，旨在更有效地指导开发者理解过程及其子过程的行为和可能的错误。文中以基于组件的Java系统为例展示了一种实现该方法的示例。然而，该方法在语言上是通用的。理想情况下，生成的解释将提供对过程的良好理解，即使开发者对所考虑的系统的一些细节不熟悉。本文的示例提供了一个开源的网络应用，供所有用户免费访问。 

---
# In2x at WMT25 Translation Task 

**Title (ZH)**: In2x在WMT25翻译任务中 

**Authors**: Lei Pang, Hanyi Mao, Quanjia Xiao, HaiXiao Liu, Xiangyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14472)  

**Abstract**: This paper presents the open-system submission by the In2x research team for the WMT25 General Machine Translation Shared Task. Our submission focuses on Japanese-related translation tasks, aiming to explore a generalizable paradigm for extending large language models (LLMs) to other languages. This paradigm encompasses aspects such as data construction methods and reward model design. The ultimate goal is to enable large language model systems to achieve exceptional performance in low-resource or less commonly spoken languages. 

**Abstract (ZH)**: 本论文介绍了In2x研究团队为WMT25通用机器翻译共享任务的开放系统提交内容。我们的提交主要关注日语相关的翻译任务，旨在探索一种可推广的范式，将大型语言模型（LLMs）扩展到其他语言。该范式涵盖了数据构建方法和奖励模型设计等方面。最终目标是使大型语言模型系统在低资源或较少使用的语言中实现出色的表现。 

---
# NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer Reasoning Model 

**Title (ZH)**: NVIDIA Nemotron Nano 2：一种精确高效的混合Mamba-Transformer推理模型 

**Authors**: NVIDIA, Aarti Basant, Abhijit Khairnar, Abhijit Paithankar, Abhinav Khattar, Adi Renduchintala, Adithya Renduchintala, Aditya Malte, Akhiad Bercovich, Akshay Hazare, Alejandra Rico, Aleksander Ficek, Alex Kondratenko, Alex Shaposhnikov, Ali Taghibakhshi, Amelia Barton, Ameya Sunil Mahabaleshwarkar, Amy Shen, Andrew Tao, Ann Guan, Anna Shors, Anubhav Mandarwal, Arham Mehta, Arun Venkatesan, Ashton Sharabiani, Ashwath Aithal, Ashwin Poojary, Ayush Dattagupta, Balaram Buddharaju, Banghua Zhu, Barnaby Simkin, Bilal Kartal, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Brandon Norick, Brian Yu, Bryan Catanzaro, Charles Wang, Charlie Truong, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Christian Munley, Christopher Parisien, Dan Su, Daniel Afrimi, Daniel Korzekwa, Daniel Rohrer, Daria Gitman, David Mosallanezhad, Deepak Narayanan, Dima Rekesh, Dina Yared, Dmytro Pykhtar, Dong Ahn, Duncan Riach, Eileen Long, Elliott Ning, Eric Chung, Erick Galinkin, Evelina Bakhturina, Gargi Prasad, Gerald Shen, Haim Elisha, Harsh Sharma, Hayley Ross, Helen Ngo, Herman Sahota, Hexin Wang, Hoo Chang Shin, Hua Huang, Iain Cunningham, Igor Gitman, Ivan Moshkov, Jaehun Jung, Jan Kautz, Jane Polak Scowcroft, Jared Casper, Jimmy Zhang, Jinze Xue, Jocelyn Huang, Joey Conway, John Kamalu, Jonathan Cohen, Joseph Jennings, Julien Veron Vialard, Junkeun Yi, Jupinder Parmar, Kari Briski, Katherine Cheung, Katherine Luna, Keith Wyss, Keshav Santhanam, Kezhi Kong, Krzysztof Pawelec, Kumar Anik, Kunlun Li, Kushan Ahmadian, Lawrence McAfee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14444)  

**Abstract**: We introduce Nemotron-Nano-9B-v2, a hybrid Mamba-Transformer language model designed to increase throughput for reasoning workloads while achieving state-of-the-art accuracy compared to similarly-sized models. Nemotron-Nano-9B-v2 builds on the Nemotron-H architecture, in which the majority of the self-attention layers in the common Transformer architecture are replaced with Mamba-2 layers, to achieve improved inference speed when generating the long thinking traces needed for reasoning. We create Nemotron-Nano-9B-v2 by first pre-training a 12-billion-parameter model (Nemotron-Nano-12B-v2-Base) on 20 trillion tokens using an FP8 training recipe. After aligning Nemotron-Nano-12B-v2-Base, we employ the Minitron strategy to compress and distill the model with the goal of enabling inference on up to 128k tokens on a single NVIDIA A10G GPU (22GiB of memory, bfloat16 precision). Compared to existing similarly-sized models (e.g., Qwen3-8B), we show that Nemotron-Nano-9B-v2 achieves on-par or better accuracy on reasoning benchmarks while achieving up to 6x higher inference throughput in reasoning settings like 8k input and 16k output tokens. We are releasing Nemotron-Nano-9B-v2, Nemotron-Nano12B-v2-Base, and Nemotron-Nano-9B-v2-Base checkpoints along with the majority of our pre- and post-training datasets on Hugging Face. 

**Abstract (ZH)**: Nemotron-Nano-9B-v2：一种用于提高推理工作负载吞吐量的混合Mamba-Transformer语言模型 

---
# Cognitive Surgery: The Awakening of Implicit Territorial Awareness in LLMs 

**Title (ZH)**: 认知手术：LLMs中隐含领土意识的觉醒 

**Authors**: Yinghan Zhou, Weifeng Zhu, Juan Wen, Wanli Peng, Zhengxian Wu, Yiming Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.14408)  

**Abstract**: Large language models (LLMs) have been shown to possess a degree of self-recognition capability-the ability to identify whether a given text was generated by themselves. Prior work has demonstrated that this capability is reliably expressed under the Pair Presentation Paradigm (PPP), where the model is presented with two texts and asked to choose which one it authored. However, performance deteriorates sharply under the Individual Presentation Paradigm (IPP), where the model is given a single text to judge authorship. Although this phenomenon has been observed, its underlying causes have not been systematically analyzed. In this paper, we first replicate existing findings to confirm that LLMs struggle to distinguish self- from other-generated text under IPP. We then investigate the reasons for this failure and attribute it to a phenomenon we term Implicit Territorial Awareness (ITA)-the model's latent ability to distinguish self- and other-texts in representational space, which remains unexpressed in its output behavior. To awaken the ITA of LLMs, we propose Cognitive Surgery (CoSur), a novel framework comprising four main modules: representation extraction, territory construction, authorship discrimination and cognitive editing. Experimental results demonstrate that our proposed method improves the performance of three different LLMs in the IPP scenario, achieving average accuracies of 83.25%, 66.19%, and 88.01%, respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）具有一定程度的自我识别能力——识别给定文本是否由自己生成的能力。prior工作表明，这一能力在Pair Presentation Paradigm（PPP）下可靠地表达出来，即模型在面对两个文本并选择哪个是自己生成时表现良好。然而，在Individual Presentation Paradigm（IPP）下，模型仅需判断单个文本的作者身份时，其表现显著下降。尽管已经观察到这一现象，但其背后的原因尚未系统分析。在本文中，我们首先重现现有发现，验证LLMs在IPP下难以区分自我生成与其他生成的文本。然后，我们探讨这种失败的原因，并将其归因于我们称为隐含领地意识（ITA）的现象——模型在表示空间中区分自我与其他文本的潜在能力，但在其输出行为中未被表达。为了唤醒LLMs的ITA，我们提出了一种名为Cognitive Surgery（CoSur）的新型框架，包括四个主要模块：表示提取、领地构建、作者识别和认知编辑。实验结果表明，我们的方法在IPP场景下提高了三种不同LLMs的表现，分别达到83.25%、66.19%和88.01%的平均准确率。 

---
# DEPTH: Hallucination-Free Relation Extraction via Dependency-Aware Sentence Simplification and Two-tiered Hierarchical Refinement 

**Title (ZH)**: 深度：基于依赖意识句子简化和两层层次 refinement 的关系提取 

**Authors**: Yupei Yang, Fan Feng, Lin Yang, Wanxi Deng, Lin Qu, Biwei Huang, Shikui Tu, Lei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14391)  

**Abstract**: Relation extraction enables the construction of structured knowledge for many downstream applications. While large language models (LLMs) have shown great promise in this domain, most existing methods concentrate on relation classification, which predicts the semantic relation type between a related entity pair. However, we observe that LLMs often struggle to reliably determine whether a relation exists, especially in cases involving complex sentence structures or intricate semantics, which leads to spurious predictions. Such hallucinations can introduce noisy edges in knowledge graphs, compromising the integrity of structured knowledge and downstream reliability. To address these challenges, we propose DEPTH, a framework that integrates Dependency-aware sEntence simPlification and Two-tiered Hierarchical refinement into the relation extraction pipeline. Given a sentence and its candidate entity pairs, DEPTH operates in two stages: (1) the Grounding module extracts relations for each pair by leveraging their shortest dependency path, distilling the sentence into a minimal yet coherent relational context that reduces syntactic noise while preserving key semantics; (2) the Refinement module aggregates all local predictions and revises them based on a holistic understanding of the sentence, correcting omissions and inconsistencies. We further introduce a causality-driven reward model that mitigates reward hacking by disentangling spurious correlations, enabling robust fine-tuning via reinforcement learning with human feedback. Experiments on six benchmarks demonstrate that DEPTH reduces the average hallucination rate to 7.0\% while achieving a 17.2\% improvement in average F1 score over state-of-the-art baselines. 

**Abstract (ZH)**: 依赖感知的句子简化和两级层次细化结合的Relation Extraction框架：DEPTH 

---
# Credence Calibration Game? Calibrating Large Language Models through Structured Play 

**Title (ZH)**: 信任校准博弈？通过结构化游戏校准大型语言模型 

**Authors**: Ke Fang, Tianyi Zhao, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.14390)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in decision-critical domains, it becomes essential to ensure that their confidence estimates faithfully correspond to their actual correctness. Existing calibration methods have primarily focused on post-hoc adjustments or auxiliary model training; however, many of these approaches necessitate additional supervision or parameter updates. In this work, we propose a novel prompt-based calibration framework inspired by the Credence Calibration Game. Our method establishes a structured interaction loop wherein LLMs receive feedback based on the alignment of their predicted confidence with correctness. Through feedback-driven prompting and natural language summaries of prior performance, our framework dynamically improves model calibration. Extensive experiments across models and game configurations demonstrate consistent improvements in evaluation metrics. Our results highlight the potential of game-based prompting as an effective strategy for LLM calibration. Code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在决策关键领域应用日益增多，确保其置信估计准确反映其实际正确性变得至关重要。现有校准方法主要侧重于事后调整或辅助模型训练；然而，许多这些方法需要额外的监督或参数更新。在本文中，我们提出了一种受可信度校准游戏启发的新型提示基校准框架。该方法通过结构化交互循环，使LLMs根据预测的置信度与正确性的一致性接收反馈。通过基于反馈的提示和自然语言总结的先前性能，我们的框架动态提高模型校准。跨模型和游戏配置的广泛实验证明了在评估指标上的持续改进。我们的结果强调基于游戏的提示作为LLM校准有效策略的潜力。代码和数据可在该网址获取。 

---
# ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities 

**Title (ZH)**: ZPD-SCA: 揭示LLMs在评估学生认知能力方面存在的盲点 

**Authors**: Wenhan Dong, Zhen Sun, Yuemeng Zhao, Zifan Peng, Jun Wu, Jingyi Zheng, Yule Liu, Xinlei He, Yu Wang, Ruiming Wang, Xinyi Huang, Lei Mo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14377)  

**Abstract**: Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在教育应用中展现了潜力，但在准确评估阅读材料与学生发育阶段的认知一致性方面的能力尚待充分探索。这一缺口尤其关键，因为临近发展区（ZPD）的基本教育原则强调学习资源应与学生的认知能力（SCA）相匹配。尽管这一匹配的重要性不言而喻，但在不同学生年龄组阅读理解难度评估方面，针对大规模语言模型（LLMs）进行的全面研究依然匮乏，特别是在汉语语言教育的背景下。为了填补这一缺口，我们提出了ZPD-SCA这一新型基准，旨在评估阶段性的汉语阅读理解难度。该基准由60名特级教师进行标注，这代表了全国在职教师的前0.15%。实验结果显示，大规模语言模型在零样本学习场景中表现不佳，甚至低于随机猜测的概率。当提供上下文示例时，模型的性能显著提升，某些模型的准确率几乎提高了其零样本基线的一倍。这些结果揭示了大规模语言模型评估阅读难度的初步能力，同时也暴露了其当前训练在教育相关判断方面存在的局限性。值得注意的是，即使是表现最佳的模型也显示出系统的方向性偏差，这表明在准确匹配材料难度与认知能力方面存在困难。此外，不同体裁之间模型性能的显著差异凸显了该任务的复杂性。我们设想，ZPD-SCA能够为评估和改善大型语言模型在认知一致的教育应用中的表现提供基础。 

---
# Organ-Agents: Virtual Human Physiology Simulator via LLMs 

**Title (ZH)**: 器官代理：通过大语言模型的虚拟人体生理模拟器 

**Authors**: Rihao Chang, He Jiao, Weizhi Nie, Honglin Guo, Keliang Xie, Zhenhua Wu, Lina Zhao, Yunpeng Bai, Yongtao Ma, Lanjun Wang, Yuting Su, Xi Gao, Weijie Wang, Nicu Sebe, Bruno Lepri, Bingwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14357)  

**Abstract**: Recent advances in large language models (LLMs) have enabled new possibilities in simulating complex physiological systems. We introduce Organ-Agents, a multi-agent framework that simulates human physiology via LLM-driven agents. Each Simulator models a specific system (e.g., cardiovascular, renal, immune). Training consists of supervised fine-tuning on system-specific time-series data, followed by reinforcement-guided coordination using dynamic reference selection and error correction. We curated data from 7,134 sepsis patients and 7,895 controls, generating high-resolution trajectories across 9 systems and 125 variables. Organ-Agents achieved high simulation accuracy on 4,509 held-out patients, with per-system MSEs <0.16 and robustness across SOFA-based severity strata. External validation on 22,689 ICU patients from two hospitals showed moderate degradation under distribution shifts with stable simulation. Organ-Agents faithfully reproduces critical multi-system events (e.g., hypotension, hyperlactatemia, hypoxemia) with coherent timing and phase progression. Evaluation by 15 critical care physicians confirmed realism and physiological plausibility (mean Likert ratings 3.9 and 3.7). Organ-Agents also enables counterfactual simulations under alternative sepsis treatment strategies, generating trajectories and APACHE II scores aligned with matched real-world patients. In downstream early warning tasks, classifiers trained on synthetic data showed minimal AUROC drops (<0.04), indicating preserved decision-relevant patterns. These results position Organ-Agents as a credible, interpretable, and generalizable digital twin for precision diagnosis, treatment simulation, and hypothesis testing in critical care. 

**Abstract (ZH)**: Recent Advances in大型语言模型（LLMs）近期在大型语言模型（LLMs）方面的进展开启了模拟复杂生理系统的新可能性。我们引入了Organ-Agents，一种通过LLM驱动的代理模拟人类生理的多代理框架。每个Simulator模拟特定的系统（例如，心血管系统、肾系统、免疫系统）。训练过程包括基于特定时间序列数据的监督调优，随后是基于强化学习的协调，使用动态参考选择和错误校正。我们从7,134例脓毒症患者和7,895例对照者中收集了数据，生成了跨越9个系统和125个变量的高分辨率轨迹。Organ-Agents在4,509例留出患者中实现了高水平的模拟准确性，每个系统的均方误差<0.16，并且在SOFA基严重程度分层中表现出鲁棒性。两个医院的22,689例ICU患者的外部验证显示，在分布转移下有适度退化但模拟稳定。Organ-Agents真实地再现了关键的多系统事件（如低血压、高乳酸血症、低氧血症），具有连贯的时间进程和相位进展。由15名重症医学专家评估确认了其现实性和生理学合理性（平均Likert评分3.9和3.7）。Organ-Agents还能够模拟在不同脓毒症治疗策略下的事实替代场景，生成与匹配的现实世界患者一致的轨迹和APACHE II评分。在下游早期预警任务中，基于合成数据训练的分类器的AUROC下降幅度最小（<0.04），表明保留了决策相关的模式。这些结果将Organ-Agents定位为精准诊断、治疗模拟和重症医学中假设测试的一种可信、可解释和可泛化的数字孪生。 

---
# Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency 

**Title (ZH)**: 零知识大语言模型幻想检测与缓解通过细粒度跨模型一致性 

**Authors**: Aman Goel, Daniel Schwartz, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14314)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but they remain susceptible to hallucinations--generating content that appears plausible but contains factual inaccuracies. We present Finch-Zk, a black-box framework that leverages FINe-grained Cross-model consistency to detect and mitigate Hallucinations in LLM outputs without requiring external knowledge sources. Finch-Zk introduces two key innovations: 1) a cross-model consistency checking strategy that reveals fine-grained inaccuracies by comparing responses generated by diverse models from semantically-equivalent prompts, and 2) a targeted mitigation technique that applies precise corrections to problematic segments while preserving accurate content. Experiments on the FELM dataset show Finch-Zk improves hallucination detection F1 scores by 6-39\% compared to existing approaches. For mitigation, Finch-Zk achieves 7-8 absolute percentage points improvement in answer accuracy on the GPQA-diamond dataset when applied to state-of-the-art models like Llama 4 Maverick and Claude 4 Sonnet. Extensive evaluation across multiple models demonstrates that Finch-Zk provides a practical, deployment-ready safeguard for enhancing factual reliability in production LLM systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务中展现出了令人印象深刻的能力，但仍易产生幻觉——生成看似合理但实际上包含事实错误的内容。我们提出了Finch-Zk，这是一种黑箱框架，利用细粒度跨模型一致性来检测和减轻LLM输出的幻觉，无需外部知识源。Finch-Zk 引入了两项关键技术：1) 一种跨模型一致性检查策略，通过比较来自语义等价提示的多样化模型生成的响应来揭示细粒度的不准确之处；2) 一种针对性的缓解技术，对有问题的部分进行精确修正，同时保留准确的内容。实验结果显示，Finch-Zk 在FELM数据集上的幻觉检测F1分数比现有方法提高了6%-39%。在缓解方面，当应用于如Llama 4 Maverick和Claude 4 Sonnet等最先进模型时，Finch-Zk 在GPQA-diamond数据集上实现了7%-8%的绝对准确率提升。广泛的评估表明，Finch-Zk 提供了一种实用且可部署的方法，用于增强生产中LLM系统的事实可靠性。 

---
# Your Reward Function for RL is Your Best PRM for Search: Unifying RL and Search-Based TTS 

**Title (ZH)**: 你的奖励函数用于强化学习即是搜索最佳PRM：统一基于强化学习和基于搜索的TTS 

**Authors**: Can Jin, Yang Zhou, Qixin Zhang, Hongwu Peng, Di Zhang, Marco Pavone, Ligong Han, Zhang-Wei Hong, Tong Che, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2508.14313)  

**Abstract**: Test-time scaling (TTS) for large language models (LLMs) has thus far fallen into two largely separate paradigms: (1) reinforcement learning (RL) methods that optimize sparse outcome-based rewards, yet suffer from instability and low sample efficiency; and (2) search-based techniques guided by independently trained, static process reward models (PRMs), which require expensive human- or LLM-generated labels and often degrade under distribution shifts. In this paper, we introduce AIRL-S, the first natural unification of RL-based and search-based TTS. Central to AIRL-S is the insight that the reward function learned during RL training inherently represents the ideal PRM for guiding downstream search. Specifically, we leverage adversarial inverse reinforcement learning (AIRL) combined with group relative policy optimization (GRPO) to learn a dense, dynamic PRM directly from correct reasoning traces, entirely eliminating the need for labeled intermediate process data. At inference, the resulting PRM simultaneously serves as the critic for RL rollouts and as a heuristic to effectively guide search procedures, facilitating robust reasoning chain extension, mitigating reward hacking, and enhancing cross-task generalization. Experimental results across eight benchmarks, including mathematics, scientific reasoning, and code generation, demonstrate that our unified approach improves performance by 9 % on average over the base model, matching GPT-4o. Furthermore, when integrated into multiple search algorithms, our PRM consistently outperforms all baseline PRMs trained with labeled data. These results underscore that, indeed, your reward function for RL is your best PRM for search, providing a robust and cost-effective solution to complex reasoning tasks in LLMs. 

**Abstract (ZH)**: Test-time Scaling for Large Language Models via Adversarial Inverse Reinforcement Learning and Group Relative Policy Optimization 

---
# GLASS: Test-Time Acceleration for LLMs via Global-Local Neural Importance Aggregation 

**Title (ZH)**: GLASS: 通过全局-局部神经重要性聚合实现大语言模型测试时加速 

**Authors**: Amirmohsen Sattarifard, Sepehr Lavasani, Ehsan Imani, Kunlin Zhang, Hanlin Xu, Fengyu Sun, Negar Hassanpour, Chao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14302)  

**Abstract**: Deploying Large Language Models (LLMs) on edge hardware demands aggressive, prompt-aware dynamic pruning to reduce computation without degrading quality. Static or predictor-based schemes either lock in a single sparsity pattern or incur extra runtime overhead, and recent zero-shot methods that rely on statistics from a single prompt fail on short prompt and/or long generation scenarios. We introduce A/I-GLASS: Activation- and Impact-based Global-Local neural importance Aggregation for feed-forward network SparSification, two training-free methods that dynamically select FFN units using a rank-aggregation of prompt local and model-intrinsic global neuron statistics. Empirical results across multiple LLMs and benchmarks demonstrate that GLASS significantly outperforms prior training-free methods, particularly in challenging long-form generation scenarios, without relying on auxiliary predictors or adding any inference overhead. 

**Abstract (ZH)**: 基于激活和影响的全局-局部神经重要性聚合的前向网络稀疏化：无训练动态选择方法 

---
# Amortized Bayesian Meta-Learning for Low-Rank Adaptation of Large Language Models 

**Title (ZH)**: 低秩适应的大语言模型的拟似然贝叶斯元学习 

**Authors**: Liyi Zhang, Jake Snell, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2508.14285)  

**Abstract**: Fine-tuning large language models (LLMs) with low-rank adaptaion (LoRA) is a cost-effective way to incorporate information from a specific dataset. However, it is often unclear how well the fine-tuned LLM will generalize, i.e., how well it will perform on unseen datasets. Methods have been proposed to improve generalization by optimizing with in-context prompts, or by using meta-learning to fine-tune LLMs. However, these methods are expensive in memory and computation, requiring either long-context prompts or saving copies of parameters and using second-order gradient updates. To address these challenges, we propose Amortized Bayesian Meta-Learning for LoRA (ABMLL). This method builds on amortized Bayesian meta-learning for smaller models, adapting this approach to LLMs while maintaining its computational efficiency. We reframe task-specific and global parameters in the context of LoRA and use a set of new hyperparameters to balance reconstruction accuracy and the fidelity of task-specific parameters to the global ones. ABMLL provides effective generalization and scales to large models such as Llama3-8B. Furthermore, as a result of using a Bayesian framework, ABMLL provides improved uncertainty quantification. We test ABMLL on Unified-QA and CrossFit datasets and find that it outperforms existing methods on these benchmarks in terms of both accuracy and expected calibration error. 

**Abstract (ZH)**: 使用Amortized Bayesian Meta-Learning for LoRA实现大型语言模型的有效泛化 

---
# Disentangling concept semantics via multilingual averaging in Sparse Autoencoders 

**Title (ZH)**: 通过稀疏自编码器中的多语言平均分离概念语义 

**Authors**: Cliff O'Reilly, Ernesto Jimenez-Ruiz, Tillman Weyde  

**Link**: [PDF](https://arxiv.org/pdf/2508.14275)  

**Abstract**: Connecting LLMs with formal knowledge representation and reasoning is a promising approach to address their shortcomings. Embeddings and sparse autoencoders are widely used to represent textual content, but the semantics are entangled with syntactic and language-specific information. We propose a method that isolates concept semantics in Large Langue Models by averaging concept activations derived via Sparse Autoencoders. We create English text representations from OWL ontology classes, translate the English into French and Chinese and then pass these texts as prompts to the Gemma 2B LLM. Using the open source Gemma Scope suite of Sparse Autoencoders, we obtain concept activations for each class and language version. We average the different language activations to derive a conceptual average. We then correlate the conceptual averages with a ground truth mapping between ontology classes. Our results give a strong indication that the conceptual average aligns to the true relationship between classes when compared with a single language by itself. The result hints at a new technique which enables mechanistic interpretation of internal network states with higher accuracy. 

**Abstract (ZH)**: 将大型语言模型与形式化的知识表示和推理相连是一种有希望的方法，以解决其不足之处。通过稀疏自编码器提取的概念激活进行平均，以隔离大型语言模型中的概念语义。从OWL本体类创建英文文本表示，将其翻译成法语和中文，然后将这些文本作为提示传递给Gemma 2B大型语言模型。使用开源的Gemma Scope套件中的稀疏自编码器，我们获得了每种类别和语言版本的概念激活。我们对不同的语言激活进行平均，得出概念平均值。然后我们将概念平均值与本体类之间的 ground truth 映射进行相关分析。我们的结果强烈表明，当与单一语言相比时，概念平均值能够更好地对类别之间的真正关系进行对齐。这一结果暗示了一种新的技术，该技术能够以更高的准确性对内部网络状态进行机械性解释。 

---
# CCFC: Core & Core-Full-Core Dual-Track Defense for LLM Jailbreak Protection 

**Title (ZH)**: CCFC: 核心与核心-全核心双轨防御技术用于LLM脱戒保护 

**Authors**: Jiaming Hu, Haoyu Wang, Debarghya Mukherjee, Ioannis Ch. Paschalidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.14128)  

**Abstract**: Jailbreak attacks pose a serious challenge to the safe deployment of large language models (LLMs). We introduce CCFC (Core & Core-Full-Core), a dual-track, prompt-level defense framework designed to mitigate LLMs' vulnerabilities from prompt injection and structure-aware jailbreak attacks. CCFC operates by first isolating the semantic core of a user query via few-shot prompting, and then evaluating the query using two complementary tracks: a core-only track to ignore adversarial distractions (e.g., toxic suffixes or prefix injections), and a core-full-core (CFC) track to disrupt the structural patterns exploited by gradient-based or edit-based attacks. The final response is selected based on a safety consistency check across both tracks, ensuring robustness without compromising on response quality. We demonstrate that CCFC cuts attack success rates by 50-75% versus state-of-the-art defenses against strong adversaries (e.g., DeepInception, GCG), without sacrificing fidelity on benign queries. Our method consistently outperforms state-of-the-art prompt-level defenses, offering a practical and effective solution for safer LLM deployment. 

**Abstract (ZH)**: Jailbreak攻击对大型语言模型的安全部署构成严重挑战。我们提出了CCFC（核心与全核心）双轨制提示级别防御框架，旨在减轻由提示注入和结构感知型 jailbreak 攻击引起的大型语言模型的脆弱性。CCFC 通过少量提示隔离用户查询的语义核心，并使用两条互补的轨道进行评估：核心仅轨道忽略对抗性干扰（例如，有害的后缀或前缀注入），以及核心全核心（CFC）轨道以破坏基于梯度或基于编辑的攻击所利用的结构模式。最终响应基于两轨道的安全一致性检查进行选择，确保其稳健性而不牺牲响应质量。我们证明，与强对手（例如，DeepInception、GCG）的最新防御措施相比，CCFC 可将攻击成功率削减 50-75%，并且在良性查询上不牺牲准确度。我们的方法持续优于最新的提示级别防御措施，提供了一种实用而有效的解决方案，以实现更安全的大规模语言模型部署。 

---
# Hard Examples Are All You Need: Maximizing GRPO Post-Training Under Annotation Budgets 

**Title (ZH)**: 硬样本即所需：在标注预算下的GRPO训练后优化最大化 

**Authors**: Benjamin Pikus, Pratyush Ranjan Tiwari, Burton Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.14094)  

**Abstract**: Collecting high-quality training examples for language model fine-tuning is expensive, with practical budgets limiting the amount of data that can be procured. We investigate a critical question for resource-constrained alignment: under a fixed acquisition budget, should practitioners prioritize examples that are easy, medium, hard, or of random difficulty? We study Group Relative Policy Optimization (GRPO) fine-tuning across different model sizes and families, comparing four subset selection policies chosen from the same unlabeled pool using base-model difficulty estimates obtained via multi-sample evaluation. Our experiments reveal that training on the hardest examples yields the largest performance gains, up to 47%, while training on easy examples yield the smallest gains. Analysis reveals that this effect arises from harder examples providing more learnable opportunities during GRPO training. These findings provide practical guidance for budget-constrained post-training: prioritizing hard examples yields substantial performance gains on reasoning tasks when using GRPO. 

**Abstract (ZH)**: 在资源受限的对齐中，在固定获取预算下，语言模型微调时应优先选择难度最容易、中等、困难还是随机难度的训练示例？我们的研究发现，在最困难的示例上进行训练可获得最大的性能提升，高达47%，而在最容易的示例上进行训练则获得最小的性能提升。分析表明，这种效果源于最难的示例在GRPO训练过程中提供了更多的可学习机会。这些发现为预算受限的后续训练提供了实用指导：在使用GRPO时，优先选择困难的示例可以在推理任务中获得显著的性能提升。 

---
# DLLMQuant: Quantizing Diffusion-based Large Language Models 

**Title (ZH)**: DLLMQuant: 基于扩散的大语言模型量化 

**Authors**: Chen Xu, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14090)  

**Abstract**: Diffusion-based large language models (DLLMs) have shown promise for non-autoregressive text generation, but their deployment is constrained by large model sizes and heavy computational costs. Post-training quantization (PTQ), a widely used method for compressing and accelerating Large Language Models (LLMs), suffers from severe accuracy degradation and reduced generalization performance when directly applied to DLLMs (e.g., AWQ suffers a 16% accuracy drop on LLADA under W4A4). This paper explores how DLLMs' key mechanisms - dynamic masking, iterative generation, bidirectional attention - clash with quantization. We identify three core issues: 1) Iterative generation and dynamic masking ratios lead to distinct token distributions across decoding steps, which are not adequately captured by existing PTQ calibration methods; 2) Quantization errors are accumulated and amplified progressively during iteration in DLLMs, causing quantized models to perform worse as decoding steps progress; 3) Unmasked tokens stabilize while masked remain probabilistic, making overall feature distribution incompatible with existing PTQ methods. To address these issues, we propose DLLMQuant, a PTQ framework tailored for DLLMs, which incorporates three novel techniques: 1) Temporal-Mask Adaptive Sampling (TMAS), a calibration method that accounts for both time and mask factors, with the capacity to capture distributions across timesteps. 2) Interaction-Aware Activation Quantization (IA-AQ), which utilizes bidirectional attention's interaction signals to dynamically allocate quantization resources. 3) Certainty-Guided Quantization (CGQ), which integrates mask status and token scores as key weighting criteria into error compensation, making weight quantization more suitable for DLLMs. Experiments show that DLLMQuant achieves significant performance gains while enhancing efficiency. 

**Abstract (ZH)**: 基于扩散的大语言模型（DLLMs）在非自回归文本生成方面表现出潜力，但其部署受限于庞大的模型规模和高昂的计算成本。针对大语言模型（LLMs）压缩和加速的后训练量化（PTQ）方法在直接应用于DLLMs（例如，AWQ在LLADA下的准确率下降了16%）时，会导致严重的准确率下降和泛化性能降低。本文探讨了DLLMs的关键机制——动态掩码、迭代生成、双向注意——与量化之间的冲突。我们识别出三个核心问题：1）迭代生成和动态掩码比例导致解码步骤中不同的令牌分布，现有的PTQ校准方法难以充分捕捉；2）在DLLMs中，量化误差在迭代过程中逐渐累积和放大，导致量化模型随着解码步骤的增加而表现更差；3）未掩码的令牌趋于稳定，而掩码的则保持概率性，使得整体特征分布与现有的PTQ方法不兼容。为解决这些问题，我们提出了DLLMQuant，这是一种针对DLLMs的PTQ框架，结合了三种新颖的技术：1）时间-掩码自适应采样（TMAS），一种同时考虑时间和掩码因素的校准方法，能够捕捉跨时间步的分布；2）交互感知激活量化（IA-AQ），利用双向注意的交互信号动态分配量化资源；3）基于确定性量化（CGQ），将掩码状态和令牌得分作为关键权重指标纳入误差补偿，使权重量化更适合DLLMs。实验表明，DLLMQuant在提高性能的同时也增强了效率。 

---
# PersRM-R1: Enhance Personalized Reward Modeling with Reinforcement Learning 

**Title (ZH)**: PersRM-R1: 通过强化学习提升个性化奖励建模 

**Authors**: Mengdi Li, Guanqiao Chen, Xufeng Zhao, Haochen Wen, Shu Yang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14076)  

**Abstract**: Reward models (RMs), which are central to existing post-training methods, aim to align LLM outputs with human values by providing feedback signals during fine-tuning. However, existing RMs struggle to capture nuanced, user-specific preferences, especially under limited data and across diverse domains. Thus, we introduce PersRM-R1, the first reasoning-based reward modeling framework specifically designed to identify and represent personal factors from only one or a few personal exemplars. To address challenges including limited data availability and the requirement for robust generalization, our approach combines synthetic data generation with a two-stage training pipeline consisting of supervised fine-tuning followed by reinforcement fine-tuning. Experimental results demonstrate that PersRM-R1 outperforms existing models of similar size and matches the performance of much larger models in both accuracy and generalizability, paving the way for more effective personalized LLMs. 

**Abstract (ZH)**: 基于推理的个性化奖励模型（PersRM-R1）：仅通过少量个人示例识别和表示个性化因素 

---
# Special-Character Adversarial Attacks on Open-Source Language Model 

**Title (ZH)**: 开源语言模型中的特殊字符对抗攻击 

**Authors**: Ephraiem Sarabamoun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14070)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across diverse natural language processing tasks, yet their vulnerability to character-level adversarial manipulations presents significant security challenges for real-world deployments. 

**Abstract (ZH)**: 大规模语言模型在多项自然语言处理任务中取得了显著性能，但对其字符级 adversarial 操纵的脆弱性为实际部署带来了重大安全挑战。 

---
# Retrieval-Augmented Generation in Industry: An Interview Study on Use Cases, Requirements, Challenges, and Evaluation 

**Title (ZH)**: 工业领域的检索增强生成：使用案例、需求、挑战及评估的访谈研究 

**Authors**: Lorenz Brehme, Benedikt Dornauer, Thomas Ströhle, Maximilian Ehrhart, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14066)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a well-established and rapidly evolving field within AI that enhances the outputs of large language models by integrating relevant information retrieved from external knowledge sources. While industry adoption of RAG is now beginning, there is a significant lack of research on its practical application in industrial contexts. To address this gap, we conducted a semistructured interview study with 13 industry practitioners to explore the current state of RAG adoption in real-world settings. Our study investigates how companies apply RAG in practice, providing (1) an overview of industry use cases, (2) a consolidated list of system requirements, (3) key challenges and lessons learned from practical experiences, and (4) an analysis of current industry evaluation methods. Our main findings show that current RAG applications are mostly limited to domain-specific QA tasks, with systems still in prototype stages; industry requirements focus primarily on data protection, security, and quality, while issues such as ethics, bias, and scalability receive less attention; data preprocessing remains a key challenge, and system evaluation is predominantly conducted by humans rather than automated methods. 

**Abstract (ZH)**: 检索增强生成（RAG）是AI中一个已建立并迅速发展的领域，通过整合从外部知识源检索的相关信息来增强大型语言模型的输出。尽管RAG在工业领域的应用现在正在起步，但对其实际应用的研究仍存在显著不足。为弥补这一空白，我们对13名行业实践者进行了半结构化访谈研究，以探讨RAG在实际应用场景中的现状。研究内容包括（1）行业应用案例概览，（2）系统需求汇总，（3）实用经验中的关键挑战和教训，以及（4）当前工业评估方法的分析。主要发现表明，当前的RAG应用主要限于特定领域的问答任务，系统仍处于原型阶段；行业需求主要集中在数据保护、安全性和质量方面，而伦理、偏差和可扩展性等问题关注较少；数据预处理仍然是关键挑战，系统评估主要由人工完成而非自动化方法。 

---
# An automatic patent literature retrieval system based on LLM-RAG 

**Title (ZH)**: 基于LLM-RAG的自动专利文献检索系统 

**Authors**: Yao Ding, Yuqing Wu, Ziyang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.14064)  

**Abstract**: With the acceleration of technological innovation efficient retrieval and classification of patent literature have become essential for intellectual property management and enterprise RD Traditional keyword and rulebased retrieval methods often fail to address complex query intents or capture semantic associations across technical domains resulting in incomplete and lowrelevance results This study presents an automated patent retrieval framework integrating Large Language Models LLMs with RetrievalAugmented Generation RAG technology The system comprises three components: 1) a preprocessing module for patent data standardization, 2) a highefficiency vector retrieval engine leveraging LLMgenerated embeddings, and 3) a RAGenhanced query module that combines external document retrieval with contextaware response generation Evaluations were conducted on the Google Patents dataset 20062024 containing millions of global patent records with metadata such as filing date domain and status The proposed gpt35turbo0125RAG configuration achieved 805 semantic matching accuracy and 92.1% recall surpassing baseline LLM methods by 28 percentage points The framework also demonstrated strong generalization in crossdomain classification and semantic clustering tasks These results validate the effectiveness of LLMRAG integration for intelligent patent retrieval providing a foundation for nextgeneration AIdriven intellectual property analysis platforms 

**Abstract (ZH)**: 基于大型语言模型的检索增强生成专利检索框架：面向知识产权管理的技术创新efficient专利文献的高效检索与分类对于知识产权管理和企业研发至关重要。传统的基于关键词和规则的检索方法往往无法处理复杂的查询意图或捕捉跨技术领域的语义关联，导致检索结果不完整且相关性低。本研究提出了一种结合大型语言模型（LLM）和检索增强生成（RAG）技术的自动化专利检索框架。该系统包含三个组件：1）专利数据预处理模块，进行数据标准化；2）高效率的向量检索引擎，利用LLM生成的嵌入；3）结合外部文档检索与语境感知响应生成的RAG增强查询模块。评估在包含数百万条全球专利记录及元数据（如申请日期、领域和状态）的Google Patents 2006-2024数据集上进行。提出的gpt35turbo0125RAG配置实现了80.5%的语义匹配准确率和92.1%的召回率，比基线LLM方法高出28个百分点。该框架还展示了在跨领域分类和语义聚类任务中的强泛化能力。这些结果验证了LLMRAG集成在智能专利检索中的有效性，为其下一代AI驱动的知识产权分析平台奠定了基础。 

---
# A Multi-Agent Approach to Neurological Clinical Reasoning 

**Title (ZH)**: 多智能体方法在神经病临床推理中的应用 

**Authors**: Moran Sorka, Alon Gorenshtein, Dvir Aran, Shahar Shelly  

**Link**: [PDF](https://arxiv.org/pdf/2508.14063)  

**Abstract**: Large language models (LLMs) have shown promise in medical domains, but their ability to handle specialized neurological reasoning requires systematic evaluation. We developed a comprehensive benchmark using 305 questions from Israeli Board Certification Exams in Neurology, classified along three complexity dimensions: factual knowledge depth, clinical concept integration, and reasoning complexity. We evaluated ten LLMs using base models, retrieval-augmented generation (RAG), and a novel multi-agent system. Results showed significant performance variation. OpenAI-o1 achieved the highest base performance (90.9% accuracy), while specialized medical models performed poorly (52.9% for Meditron-70B). RAG provided modest benefits but limited effectiveness on complex reasoning questions. In contrast, our multi-agent framework, decomposing neurological reasoning into specialized cognitive functions including question analysis, knowledge retrieval, answer synthesis, and validation, achieved dramatic improvements, especially for mid-range models. The LLaMA 3.3-70B-based agentic system reached 89.2% accuracy versus 69.5% for its base model, with substantial gains on level 3 complexity questions. The multi-agent approach transformed inconsistent subspecialty performance into uniform excellence, addressing neurological reasoning challenges that persisted with RAG enhancement. We validated our approach using an independent dataset of 155 neurological cases from MedQA. Results confirm that structured multi-agent approaches designed to emulate specialized cognitive processes significantly enhance complex medical reasoning, offering promising directions for AI assistance in challenging clinical contexts. 

**Abstract (ZH)**: 大型语言模型在医疗领域展现出了潜力，但其处理专门化神经学推理的能力需要系统性评估。我们使用来源于以色列神经学认证考试的305个问题开发了一个综合基准，这些问题按三个复杂性维度分类：事实知识深度、临床概念集成和推理复杂度。我们评估了十种大型语言模型，包括基本模型、检索增强生成（RAG）以及一种新的多agent系统。结果表明，性能存在显著差异。OpenAI-o1在基模型评估中取得了最高性能（准确率90.9%），而专业医疗模型表现不佳（Meditron-70B准确率为52.9%）。RAG在复杂推理问题上提供了适度的好处，但在复杂性上效果有限。相比之下，我们提出的多agent框架将神经学推理分解为专门的认知功能，包括问题分析、知识检索、答案合成和验证，实现了显著的改进，尤其是在中等复杂度的问题上。基于LLaMA 3.3-70B的多agent系统达到了89.2%的准确率，而其基模型的准确率为69.5%，在复杂性层级3的问题上有了显著提升。多agent方法将专业亚专科表现的不一致性转变为了统一的卓越表现，解决了RAG增强后依然存在的神经学推理挑战。我们使用来自MedQA的155例独立神经学病例数据集验证了该方法，结果表明，设计用于模拟专门认知过程的结构化多agent方法显著增强了复杂医学推理能力，为AI在复杂临床环境中的辅助提供了有前景的方向。 

---
# Assessing and Mitigating Data Memorization Risks in Fine-Tuned Large Language Models 

**Title (ZH)**: 评估及缓解 Fine-Tuned 大型语言模型的数据记忆风险 

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji  

**Link**: [PDF](https://arxiv.org/pdf/2508.14062)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse natural language processing tasks, but their tendency to memorize training data poses significant privacy risks, particularly during fine-tuning processes. This paper presents a comprehensive empirical analysis of data memorization in fine-tuned LLMs and introduces a novel multi-layered privacy protection framework. Through controlled experiments on modern LLM architectures including GPT-2, Phi-3, and Gemma-2, we demonstrate that fine-tuning with repeated sensitive data increases privacy leakage rates from baseline levels of 0-5% to 60-75%, representing a 64.2% average increase across tested models. We propose and rigorously evaluate four complementary privacy protection methods: semantic data deduplication, differential privacy during generation, entropy-based filtering, and pattern-based content filtering. Our experimental results show that these techniques can reduce data leakage to 0% while maintaining 94.7% of original model utility. 

**Abstract (ZH)**: 大型语言模型在各种自然语言处理任务中展现了卓越的能力，但它们在微调过程中记忆训练数据的趋势构成了重大的隐私风险。本文对微调大型语言模型中的数据记忆现象进行了全面的经验分析，并提出了一种新颖的多层隐私保护框架。通过在包括GPT-2、Phi-3和Gemma-2在内的现代大型语言模型架构上进行受控实验，我们证明，使用重复的敏感数据微调会导致从基线水平（0-5%）的数据泄漏率增加到60-75%，测试模型平均增加了64.2%的数据泄漏率。我们提出了并严格评估了四种互补的隐私保护方法：语义数据去重、生成过程中的差分隐私、基于熵的过滤以及基于模式的内容过滤。实验结果表明，这些技术可以在维持原始模型94.7%的功能的同时将数据泄漏减少到0%。 

---
# T-REX: Table -- Refute or Entail eXplainer 

**Title (ZH)**: T-REX: 表格反驳或蕴含解释器 

**Authors**: Tim Luka Horstmann, Baptiste Geisenberger, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2508.14055)  

**Abstract**: Verifying textual claims against structured tabular data is a critical yet challenging task in Natural Language Processing with broad real-world impact. While recent advances in Large Language Models (LLMs) have enabled significant progress in table fact-checking, current solutions remain inaccessible to non-experts. We introduce T-REX (T-REX: Table -- Refute or Entail eXplainer), the first live, interactive tool for claim verification over multimodal, multilingual tables using state-of-the-art instruction-tuned reasoning LLMs. Designed for accuracy and transparency, T-REX empowers non-experts by providing access to advanced fact-checking technology. The system is openly available online. 

**Abstract (ZH)**: 基于结构化表格数据验证文本断言是自然语言处理中的一个重要而具有挑战性的问题，具有广泛的实际影响。虽然大型语言模型的 Recent 进展促进了表格事实核查的重要进展，但当前的解决方案仍难以为非专家所用。我们介绍了 T-REX（T-REX：表——驳斥或蕴含解释器），这是第一个使用最先进的指令微调推理大语言模型进行多模态、多语言表格断言验证的实时交互工具。T-REX 旨在提高准确性和透明度，为非专家提供高级事实核查技术的访问权限。该系统已对外开放在线使用。 

---
# MAHL: Multi-Agent LLM-Guided Hierarchical Chiplet Design with Adaptive Debugging 

**Title (ZH)**: MAHL: 多-agent LLM引导的分层芯片块设计及自适应调试 

**Authors**: Jinwei Tang, Jiayin Qin, Nuo Xu, Pragnya Sudershan Nalla, Yu Cao, Yang, Zhao, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.14053)  

**Abstract**: As program workloads (e.g., AI) increase in size and algorithmic complexity, the primary challenge lies in their high dimensionality, encompassing computing cores, array sizes, and memory hierarchies. To overcome these obstacles, innovative approaches are required. Agile chip design has already benefited from machine learning integration at various stages, including logic synthesis, placement, and routing. With Large Language Models (LLMs) recently demonstrating impressive proficiency in Hardware Description Language (HDL) generation, it is promising to extend their abilities to 2.5D integration, an advanced technique that saves area overhead and development costs. However, LLM-driven chiplet design faces challenges such as flatten design, high validation cost and imprecise parameter optimization, which limit its chiplet design capability. To address this, we propose MAHL, a hierarchical LLM-based chiplet design generation framework that features six agents which collaboratively enable AI algorithm-hardware mapping, including hierarchical description generation, retrieval-augmented code generation, diverseflow-based validation, and multi-granularity design space exploration. These components together enhance the efficient generation of chiplet design with optimized Power, Performance and Area (PPA). Experiments show that MAHL not only significantly improves the generation accuracy of simple RTL design, but also increases the generation accuracy of real-world chiplet design, evaluated by Pass@5, from 0 to 0.72 compared to conventional LLMs under the best-case scenario. Compared to state-of-the-art CLARIE (expert-based), MAHL achieves comparable or even superior PPA results under certain optimization objectives. 

**Abstract (ZH)**: 基于大型语言模型的层次化Chiplet设计生成框架MAHL 

---
# The Hidden Cost of Readability: How Code Formatting Silently Consumes Your LLM Budget 

**Title (ZH)**: 隐藏的可读性成本：代码格式化悄无声息地消耗你的LLM预算 

**Authors**: Dangfeng Pan, Zhensu Sun, Cenyuan Zhang, David Lo, Xiaoning Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.13666)  

**Abstract**: Source code is usually formatted with elements like indentation and newlines to improve readability for human developers. However, these visual aids do not seem to be beneficial for large language models (LLMs) in the same way since the code is processed as a linear sequence of tokens. Furthermore, these additional tokens can lead to increased computational costs and longer response times for LLMs. If such formatting elements are non-essential to LLMs, we can reduce such costs by removing them from the code. To figure out the role played by formatting elements, we conduct a comprehensive empirical study to evaluate the impact of code formatting on LLM performance and efficiency. Through large-scale experiments on Fill-in-the-Middle Code Completion tasks across four programming languages (Java, Python, C++, C\#) and ten LLMs-including both commercial and open-source models-we systematically analyze token count and performance when formatting elements are removed. Key findings indicate that LLMs can maintain performance across formatted code and unformatted code, achieving an average input token reduction of 24.5\% with negligible output token reductions. This makes code format removal a practical optimization strategy for improving LLM efficiency. Further exploration reveals that both prompting and fine-tuning LLMs can lead to significant reductions (up to 36.1\%) in output code length without compromising correctness. To facilitate practical applications, we develop a bidirectional code transformation tool for format processing, which can be seamlessly integrated into existing LLM inference workflows, ensuring both human readability and LLM efficiency. 

**Abstract (ZH)**: 源代码通常通过缩进和换行等元素格式化以提高人类开发者的可读性。然而，这些视觉辅助元素似乎并不像对人类开发者那样对大型语言模型（LLMs）有益，因为代码被处理为一系列线性标记序列。此外，这些额外的标记可能会导致LLMs的计算成本增加和响应时间变长。如果这些格式化元素对LLMs来说是非必要的，我们可以通过移除它们来减少这些成本。为了搞清楚格式化元素的作用，我们进行了一项全面的经验研究，评估代码格式化对LLM性能和效率的影响。通过针对四种编程语言（Java、Python、C++、C#）和十种LLM（包括商业和开源模型）的大规模实验，在填写中间代码完成任务中，系统地分析移除格式化元素后的标记数量和性能。主要发现表明，LLMs在格式化代码和非格式化代码之间可以保持相同性能，平均每减少24.5%的输入标记，输出标记几乎无减少。这使得代码格式去除成为一个实用的优化策略，以提高LLM效率。进一步探索显示，无论是提示还是微调LLMs，都可以显著减少多达36.1%的输出代码长度，而不牺牲正确性。为了便于实际应用，我们开发了一个双向代码转换工具，用于格式化处理，可以无缝集成到现有的LLM推理工作流中，确保同时保持人类可读性和LLM效率。 

---
