# Privileged Self-Access Matters for Introspection in AI 

**Title (ZH)**: 特权自我访问对于AI的反省至关重要 

**Authors**: Siyuan Song, Harvey Lederman, Jennifer Hu, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2508.14802)  

**Abstract**: Whether AI models can introspect is an increasingly important practical question. But there is no consensus on how introspection is to be defined. Beginning from a recently proposed ''lightweight'' definition, we argue instead for a thicker one. According to our proposal, introspection in AI is any process which yields information about internal states through a process more reliable than one with equal or lower computational cost available to a third party. Using experiments where LLMs reason about their internal temperature parameters, we show they can appear to have lightweight introspection while failing to meaningfully introspect per our proposed definition. 

**Abstract (ZH)**: AI模型能否内省：一个日益重要的实践问题及其定义探讨 

---
# Data-Driven Probabilistic Evaluation of Logic Properties with PAC-Confidence on Mealy Machines 

**Title (ZH)**: 基于PAC置信度的Mealy机的数据驱动概率逻辑性质评估 

**Authors**: Swantje Plambeck, Ali Salamati, Eyke Huellermeier, Goerschwin Fey  

**Link**: [PDF](https://arxiv.org/pdf/2508.14710)  

**Abstract**: Cyber-Physical Systems (CPS) are complex systems that require powerful models for tasks like verification, diagnosis, or debugging. Often, suitable models are not available and manual extraction is difficult. Data-driven approaches then provide a solution to, e.g., diagnosis tasks and verification problems based on data collected from the system. In this paper, we consider CPS with a discrete abstraction in the form of a Mealy machine. We propose a data-driven approach to determine the safety probability of the system on a finite horizon of n time steps. The approach is based on the Probably Approximately Correct (PAC) learning paradigm. Thus, we elaborate a connection between discrete logic and probabilistic reachability analysis of systems, especially providing an additional confidence on the determined probability. The learning process follows an active learning paradigm, where new learning data is sampled in a guided way after an initial learning set is collected. We validate the approach with a case study on an automated lane-keeping system. 

**Abstract (ZH)**: 基于数据驱动的方法确定离散抽象CPS在有限时间段内的安全性概率 

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
# LeanGeo: Formalizing Competitional Geometry problems in Lean 

**Title (ZH)**: LeanGeo: 正式化竞标几何问题的Lean方法 

**Authors**: Chendong Song, Zihan Wang, Frederick Pu, Haiming Wang, Xiaohan Lin, Junqi Liu, Jia Li, Zhengying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14644)  

**Abstract**: Geometry problems are a crucial testbed for AI reasoning capabilities. Most existing geometry solving systems cannot express problems within a unified framework, thus are difficult to integrate with other mathematical fields. Besides, since most geometric proofs rely on intuitive diagrams, verifying geometry problems is particularly challenging. To address these gaps, we introduce LeanGeo, a unified formal system for formalizing and solving competition-level geometry problems within the Lean 4 theorem prover. LeanGeo features a comprehensive library of high-level geometric theorems with Lean's foundational logic, enabling rigorous proof verification and seamless integration with Mathlib. We also present LeanGeo-Bench, a formal geometry benchmark in LeanGeo, comprising problems from the International Mathematical Olympiad (IMO) and other advanced sources. Our evaluation demonstrates the capabilities and limitations of state-of-the-art Large Language Models on this benchmark, highlighting the need for further advancements in automated geometric reasoning. We open source the theorem library and the benchmark of LeanGeo at this https URL. 

**Abstract (ZH)**: 几何问题是一个重要的AI推理能力测试平台。现有的大多数几何解题系统无法在统一框架中表达问题，因此难以与其他数学领域集成。此外，由于大多数几何证明依赖直观图形，验证几何问题尤为困难。为解决这些问题，我们引入了LeanGeo，一个在Lean 4定理证明器中用于形式化和解决竞赛级别几何问题的统一形式系统。LeanGeo具备Lean基础逻辑的全面高阶几何定理库，支持严格的证明验证并将Mathlib无缝集成。我们还介绍了LeanGeo-Bench，一个LeanGeo中的形式几何基准测试，包含国际数学奥林匹克（IMO）和其他高级来源的问题。我们的评估展示了当前最先进的大型语言模型在这一基准测试中的能力和局限性，突显了自动化几何推理进一步发展的需求。我们已在https://github.com/LeanGeo/LeanGeo开放了定理库和基准测试代码。 

---
# Who Sees What? Structured Thought-Action Sequences for Epistemic Reasoning in LLMs 

**Title (ZH)**: 见所未见？面向LLM的epistemic推理的结构化思维-行动序列 

**Authors**: Luca Annese, Sabrina Patania, Silvia Serino, Tom Foulsham, Silvia Rossi, Azzurra Ruggeri, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2508.14564)  

**Abstract**: Recent advances in large language models (LLMs) and reasoning frameworks have opened new possibilities for improving the perspective -taking capabilities of autonomous agents. However, tasks that involve active perception, collaborative reasoning, and perspective taking (understanding what another agent can see or knows) pose persistent challenges for current LLM-based systems. This study investigates the potential of structured examples derived from transformed solution graphs generated by the Fast Downward planner to improve the performance of LLM-based agents within a ReAct framework. We propose a structured solution-processing pipeline that generates three distinct categories of examples: optimal goal paths (G-type), informative node paths (E-type), and step-by-step optimal decision sequences contrasting alternative actions (L-type). These solutions are further converted into ``thought-action'' examples by prompting an LLM to explicitly articulate the reasoning behind each decision. While L-type examples slightly reduce clarification requests and overall action steps, they do not yield consistent improvements. Agents are successful in tasks requiring basic attentional filtering but struggle in scenarios that required mentalising about occluded spaces or weighing the costs of epistemic actions. These findings suggest that structured examples alone are insufficient for robust perspective-taking, underscoring the need for explicit belief tracking, cost modelling, and richer environments to enable socially grounded collaboration in LLM-based agents. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）和推理框架为提高自主代理的视角转换能力开拓了新可能性。然而，涉及主动感知、协作推理和视角转换（理解另一代理所能见或所知的内容）的任务仍给当前基于LLM的系统带来持续挑战。本研究探讨了从Fast Downward规划器生成的变换解图谱中提取的结构化示例的潜力，以改善基于LLM的代理在ReAct框架内的性能。我们提出了一种结构化解决方案处理流水线，生成三种不同类型示例：最优目标路径（G类）、有信息节点路径（E类），以及对比替代行动的逐步最优决策序列（L类）。进一步通过提示LLM明确阐述每个决策背后的推理，将这些解决方案转换为“思考-行动”示例。虽然L类示例略微减少了澄清请求和总体行动步骤，但并未带来一致的性能改进。代理在需要基本注意力筛选的任务中表现良好，但在涉及预测遮挡空间或权衡认知行动成本的场景中则表现不佳。这些发现表明，仅依靠结构化示例不足以实现稳健的视角转换，突显了显式信念跟踪、成本建模以及更丰富的环境在使基于LLM的代理实现社会性合作方面的重要性。 

---
# The Agent Behavior: Model, Governance and Challenges in the AI Digital Age 

**Title (ZH)**: 智能时代的代理行为：模型、治理与挑战 

**Authors**: Qiang Zhang, Pei Yan, Yijia Xu, Chuanpo Fu, Yong Fang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14415)  

**Abstract**: Advancements in AI have led to agents in networked environments increasingly mirroring human behavior, thereby blurring the boundary between artificial and human actors in specific contexts. This shift brings about significant challenges in trust, responsibility, ethics, security and etc. The difficulty in supervising of agent behaviors may lead to issues such as data contamination and unclear accountability. To address these challenges, this paper proposes the "Network Behavior Lifecycle" model, which divides network behavior into 6 stages and systematically analyzes the behavioral differences between humans and agents at each stage. Based on these insights, the paper further introduces the "Agent for Agent (A4A)" paradigm and the "Human-Agent Behavioral Disparity (HABD)" model, which examine the fundamental distinctions between human and agent behaviors across 5 dimensions: decision mechanism, execution efficiency, intention-behavior consistency, behavioral inertia, and irrational patterns. The effectiveness of the model is verified through real-world cases such as red team penetration and blue team defense. Finally, the paper discusses future research directions in dynamic cognitive governance architecture, behavioral disparity quantification, and meta-governance protocol stacks, aiming to provide a theoretical foundation and technical roadmap for secure and trustworthy human-agent collaboration. 

**Abstract (ZH)**: 人工智能的进步使得网络环境中的人工智能代理越来越接近人类行为，从而在特定情境下模糊了人工与人类行为之间的界限。这一转变带来了信任、责任、伦理和安全等方面的重大挑战。由于监管代理行为的难度，可能会导致数据污染和责任不清等问题。为应对这些挑战，本文提出了“网络行为生命周期”模型，将网络行为分为六个阶段，并系统分析每个阶段人类和代理的行为差异。基于这些见解，本文进一步介绍了“代理对代理（A4A）”范式和“人类-代理行为差异（HABD）”模型，该模型从五个维度（决策机制、执行效率、意图与行为一致性、行为惯性和非理性模式）考察人类和代理行为的差异。该模型通过实际案例（如红队渗透和蓝队防御）的有效性得到了验证。最后，本文探讨了动态认知治理架构、行为差异量化和元治理协议栈等未来研究方向，旨在为安全可信的人机协作提供理论基础和技术路线图。 

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
# Graph Structure Learning with Temporal Graph Information Bottleneck for Inductive Representation Learning 

**Title (ZH)**: 基于时间图信息瓶颈的图结构学习与归纳表示学习 

**Authors**: Jiafeng Xiong, Rizos Sakellariou  

**Link**: [PDF](https://arxiv.org/pdf/2508.14859)  

**Abstract**: Temporal graph learning is crucial for dynamic networks where nodes and edges evolve over time and new nodes continuously join the system. Inductive representation learning in such settings faces two major challenges: effectively representing unseen nodes and mitigating noisy or redundant graph information. We propose GTGIB, a versatile framework that integrates Graph Structure Learning (GSL) with Temporal Graph Information Bottleneck (TGIB). We design a novel two-step GSL-based structural enhancer to enrich and optimize node neighborhoods and demonstrate its effectiveness and efficiency through theoretical proofs and experiments. The TGIB refines the optimized graph by extending the information bottleneck principle to temporal graphs, regularizing both edges and features based on our derived tractable TGIB objective function via variational approximation, enabling stable and efficient optimization. GTGIB-based models are evaluated to predict links on four real-world datasets; they outperform existing methods in all datasets under the inductive setting, with significant and consistent improvement in the transductive setting. 

**Abstract (ZH)**: 时序图学习对于节点和边随时间演化并不断有新节点加入的动态网络至关重要。在这样的环境中，归纳表示学习面临两大挑战：有效表示未见过的节点以及减轻噪声或冗余的图信息。我们提出了GTGIB，这是一种将图结构学习（GSL）与时序图信息瓶颈（TGIB）相结合的通用框架。我们设计了一种新颖的两步GSL基结构增强器，用于丰富和优化节点邻域，并通过理论证明和实验展示了其有效性和效率。TGIB通过将信息瓶颈原则扩展到时序图中并基于我们推导出的可处理的TGIB目标函数通过变分近似来同时正则化边和特征，使优化更加稳定和高效。基于GTGIB的模型在四个真实数据集上用于预测链接；在归纳设置中，它们在所有数据集中都优于现有方法，并且在自举设置中表现出显著且一致的改进。 

---
# $TIME[t] \subseteq SPACE[O(\sqrt{t})]$ via Tree Height Compression 

**Title (ZH)**: $TIME[t] \subseteq SPACE[O(\sqrt{t})]$ 通过树高压缩实现 

**Authors**: Logan Nye  

**Link**: [PDF](https://arxiv.org/pdf/2508.14831)  

**Abstract**: We prove a square-root space simulation for deterministic multitape Turing machines, showing $\TIME[t] \subseteq \SPACE[O(\sqrt{t})]$. The key step is a Height Compression Theorem that uniformly (and in logspace) reshapes the canonical left-deep succinct computation tree for a block-respecting run into a binary tree whose evaluation-stack depth along any DFS path is $O(\log T)$ for $T = \lceil t/b \rceil$, while preserving $O(b)$ work at leaves, $O(1)$ at internal nodes, and edges that are logspace-checkable; semantic correctness across merges is witnessed by an exact $O(b)$ window replay at the unique interface. The proof uses midpoint (balanced) recursion, a per-path potential that bounds simultaneously active interfaces by $O(\log T)$, and an indegree-capping replacement of multiway merges by balanced binary combiners. Algorithmically, an Algebraic Replay Engine with constant-degree maps over a constant-size field, together with pointerless DFS and index-free streaming, ensures constant-size per-level tokens and eliminates wide counters, yielding the additive tradeoff $S(b)=O(b + \log(t/b))$ for block sizes $b \ge b_0$ with $b_0 = \Theta(\log t)$, which at the canonical choice $b = \Theta(\sqrt{t})$ gives $O(\sqrt{t})$ space; the $b_0$ threshold rules out degenerate blocks where addressing scratch would dominate the window footprint. The construction is uniform, relativizes, and is robust to standard model choices. Consequences include branching-program upper bounds $2^{O(\sqrt{s})}$ for size-$s$ bounded-fan-in circuits, tightened quadratic-time lower bounds for $\SPACE[n]$-complete problems via the standard hierarchy argument, and $O(\sqrt{t})$-space certifying interpreters; under explicit locality assumptions, the framework extends to geometric $d$-dimensional models. 

**Abstract (ZH)**: 确定多带图灵机的平方根空间模拟：$\TIME[t] \subseteq \SPACE[O(\sqrt{t})]$ 

---
# Long Chain-of-Thought Reasoning Across Languages 

**Title (ZH)**: 跨语言长链推理 

**Authors**: Josh Barua, Seun Eisape, Kayo Yin, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2508.14828)  

**Abstract**: Scaling inference through long chains-of-thought (CoTs) has unlocked impressive reasoning capabilities in large language models (LLMs), yet the reasoning process remains almost exclusively English-centric. We construct translated versions of two popular English reasoning datasets, fine-tune Qwen 2.5 (7B) and Qwen 3 (8B) models, and present a systematic study of long CoT generation across French, Japanese, Latvian, and Swahili. Our experiments reveal three key findings. First, the efficacy of using English as a pivot language varies by language: it provides no benefit for French, improves performance when used as the reasoning language for Japanese and Latvian, and proves insufficient for Swahili where both task comprehension and reasoning remain poor. Second, extensive multilingual pretraining in Qwen 3 narrows but does not eliminate the cross-lingual performance gap. A lightweight fine-tune using only 1k traces still improves performance by over 30\% in Swahili. Third, data quality versus scale trade-offs are language dependent: small, carefully curated datasets suffice for English and French, whereas larger but noisier corpora prove more effective for Swahili and Latvian. Together, these results clarify when and why long CoTs transfer across languages and provide translated datasets to foster equitable multilingual reasoning research. 

**Abstract (ZH)**: 通过长链思考（CoTs）扩展推理缩放能力已经在大型语言模型（LLMs）中解锁了令人印象深刻的推理能力，但推理过程几乎完全以英语为中心。我们构建了两个流行英语推理数据集的翻译版本，对Qwen 2.5（7B）和Qwen 3（8B）模型进行微调，并对法语、日语、拉脱维亚语和斯瓦希里语的长CoT生成进行了系统性研究。我们的实验揭示了三个关键发现。首先，使用英语作为中介语言的有效性因语言而异：它对法语没有益处，在使用英语作为推理语言时可以提高日语和拉脱维亚语的表现，但在斯瓦希里语中则不足，因为任务理解和推理表现都较差。其次，Qwen 3中的 extensive 多语言预训练缩小了但未能消除跨语言性能差距。仅使用1k轨迹的轻量级微调在斯瓦希里语中仍然能将性能提高超过30%。第三，数据质量与规模之间的权衡因语言而异：对英语和法语而言，少量但精挑细选的数据集就足够，而对斯瓦希里语和拉脱维亚语而言，虽然数据集更大但噪声更多，效果更佳。总之，这些结果阐明了长CoTs在不同语言间转移的时间和原因，并提供了翻译数据集以促进公平的多语言推理研究。 

---
# From Passive Tool to Socio-cognitive Teammate: A Conceptual Framework for Agentic AI in Human-AI Collaborative Learning 

**Title (ZH)**: 从被动工具到社会认知同伴：人类与AI协作学习中赋能AI的概念框架 

**Authors**: Lixiang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14825)  

**Abstract**: The role of Artificial Intelligence (AI) in education is undergoing a rapid transformation, moving beyond its historical function as an instructional tool towards a new potential as an active participant in the learning process. This shift is driven by the emergence of agentic AI, autonomous systems capable of proactive, goal-directed action. However, the field lacks a robust conceptual framework to understand, design, and evaluate this new paradigm of human-AI interaction in learning. This paper addresses this gap by proposing a novel conceptual framework (the APCP framework) that charts the transition from AI as a tool to AI as a collaborative partner. We present a four-level model of escalating AI agency within human-AI collaborative learning: (1) the AI as an Adaptive Instrument, (2) the AI as a Proactive Assistant, (3) the AI as a Co-Learner, and (4) the AI as a Peer Collaborator. Grounded in sociocultural theories of learning and Computer-Supported Collaborative Learning (CSCL), this framework provides a structured vocabulary for analysing the shifting roles and responsibilities between human and AI agents. The paper further engages in a critical discussion of the philosophical underpinnings of collaboration, examining whether an AI, lacking genuine consciousness or shared intentionality, can be considered a true collaborator. We conclude that while AI may not achieve authentic phenomenological partnership, it can be designed as a highly effective functional collaborator. This distinction has significant implications for pedagogy, instructional design, and the future research agenda for AI in education, urging a shift in focus towards creating learning environments that harness the complementary strengths of both human and AI. 

**Abstract (ZH)**: 人工智能在教育中的角色正在经历快速转型，从传统的教学工具转变为学习过程中的主动参与者。这一转变由有能力进行预见性、目标导向行动的代理人工智能的出现推动。然而，该领域缺乏一套坚实的理论框架来理解、设计和评估这种新的人类-人工智能交互范式。本文通过提出一个新颖的概念框架（APCP框架）来填补这一空白，该框架描绘了从人工智能作为工具到作为协作伙伴的过渡过程。我们提出了一种四层模型，描述了人类-人工智能协作学习中人工智能代理级别的逐步升级：(1) 适应性工具，(2) 预见性助手，(3) 共同学习者，和(4) 平等合作者。该框架基于社会文化学习理论和计算机支持的协作学习，提供了分析人类和人工智能代理之间角色和责任转变的结构化语言。本文进一步探讨了合作的哲学基础，探讨了一个缺少真实意识或共享意图的人工智能是否可以被视为真正的合作者。我们得出结论，虽然人工智能可能无法实现真正现象学上的伙伴关系，但它可以被设计成非常有效的功能性合作者。这种区分对教育方法、教学设计以及人工智能在教育中的未来研究议程具有重要意义，敦促转向利用人类和人工智能互补优势的学习环境。 

---
# Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs 

**Title (ZH)**: 评价检索增强生成与长上下文输入在电子健康记录临床推理中的性能对比 

**Authors**: Skatje Myers, Dmitriy Dligach, Timothy A. Miller, Samantha Barr, Yanjun Gao, Matthew Churpek, Anoop Mayampurath, Majid Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2508.14817)  

**Abstract**: Electronic health records (EHRs) are long, noisy, and often redundant, posing a major challenge for the clinicians who must navigate them. Large language models (LLMs) offer a promising solution for extracting and reasoning over this unstructured text, but the length of clinical notes often exceeds even state-of-the-art models' extended context windows. Retrieval-augmented generation (RAG) offers an alternative by retrieving task-relevant passages from across the entire EHR, potentially reducing the amount of required input tokens. In this work, we propose three clinical tasks designed to be replicable across health systems with minimal effort: 1) extracting imaging procedures, 2) generating timelines of antibiotic use, and 3) identifying key diagnoses. Using EHRs from actual hospitalized patients, we test three state-of-the-art LLMs with varying amounts of provided context, using either targeted text retrieval or the most recent clinical notes. We find that RAG closely matches or exceeds the performance of using recent notes, and approaches the performance of using the models' full context while requiring drastically fewer input tokens. Our results suggest that RAG remains a competitive and efficient approach even as newer models become capable of handling increasingly longer amounts of text. 

**Abstract (ZH)**: 电子健康记录中的图像检查提取、抗生素使用时间线生成及关键诊断识别：基于检索增强生成的方法研究 

---
# TransLight: Image-Guided Customized Lighting Control with Generative Decoupling 

**Title (ZH)**: TransLight: 基于图像引导的生成解耦定制照明控制 

**Authors**: Zongming Li, Lianghui Zhu, Haocheng Shen, Longjin Ran, Wenyu Liu, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14814)  

**Abstract**: Most existing illumination-editing approaches fail to simultaneously provide customized control of light effects and preserve content integrity. This makes them less effective for practical lighting stylization requirements, especially in the challenging task of transferring complex light effects from a reference image to a user-specified target image. To address this problem, we propose TransLight, a novel framework that enables high-fidelity and high-freedom transfer of light effects. Extracting the light effect from the reference image is the most critical and challenging step in our method. The difficulty lies in the complex geometric structure features embedded in light effects that are highly coupled with content in real-world scenarios. To achieve this, we first present Generative Decoupling, where two fine-tuned diffusion models are used to accurately separate image content and light effects, generating a newly curated, million-scale dataset of image-content-light triplets. Then, we employ IC-Light as the generative model and train our model with our triplets, injecting the reference lighting image as an additional conditioning signal. The resulting TransLight model enables customized and natural transfer of diverse light effects. Notably, by thoroughly disentangling light effects from reference images, our generative decoupling strategy endows TransLight with highly flexible illumination control. Experimental results establish TransLight as the first method to successfully transfer light effects across disparate images, delivering more customized illumination control than existing techniques and charting new directions for research in illumination harmonization and editing. 

**Abstract (ZH)**: TransLight：高保真高自由度的照明效果转移框架 

---
# DINOv3 with Test-Time Training for Medical Image Registration 

**Title (ZH)**: DINOv3在测试时训练的医学图像配准 

**Authors**: Shansong Wang, Mojtaba Safari, Mingzhe Hu, Qiang Li, Chih-Wei Chang, Richard LJ Qiu, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14809)  

**Abstract**: Prior medical image registration approaches, particularly learning-based methods, often require large amounts of training data, which constrains clinical adoption. To overcome this limitation, we propose a training-free pipeline that relies on a frozen DINOv3 encoder and test-time optimization of the deformation field in feature space. Across two representative benchmarks, the method is accurate and yields regular deformations. On Abdomen MR-CT, it attained the best mean Dice score (DSC) of 0.790 together with the lowest 95th percentile Hausdorff Distance (HD95) of 4.9+-5.0 and the lowest standard deviation of Log-Jacobian (SDLogJ) of 0.08+-0.02. On ACDC cardiac MRI, it improves mean DSC to 0.769 and reduces SDLogJ to 0.11 and HD95 to 4.8, a marked gain over the initial alignment. The results indicate that operating in a compact foundation feature space at test time offers a practical and general solution for clinical registration without additional training. 

**Abstract (ZH)**: 无需额外训练的基于冻结DINOv3编码器和特征空间中的变形场测试时优化的无训练管道在医学图像注册中的应用 

---
# MF-LPR$^2$: Multi-Frame License Plate Image Restoration and Recognition using Optical Flow 

**Title (ZH)**: MF-LPR$^2$: 多帧车牌图像恢复与识别利用光流技术 

**Authors**: Kihyun Na, Junseok Oh, Youngkwan Cho, Bumjin Kim, Sungmin Cho, Jinyoung Choi, Injung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.14797)  

**Abstract**: License plate recognition (LPR) is important for traffic law enforcement, crime investigation, and surveillance. However, license plate areas in dash cam images often suffer from low resolution, motion blur, and glare, which make accurate recognition challenging. Existing generative models that rely on pretrained priors cannot reliably restore such poor-quality images, frequently introducing severe artifacts and distortions. To address this issue, we propose a novel multi-frame license plate restoration and recognition framework, MF-LPR$^2$, which addresses ambiguities in poor-quality images by aligning and aggregating neighboring frames instead of relying on pretrained knowledge. To achieve accurate frame alignment, we employ a state-of-the-art optical flow estimator in conjunction with carefully designed algorithms that detect and correct erroneous optical flow estimations by leveraging the spatio-temporal consistency inherent in license plate image sequences. Our approach enhances both image quality and recognition accuracy while preserving the evidential content of the input images. In addition, we constructed a novel Realistic LPR (RLPR) dataset to evaluate MF-LPR$^2$. The RLPR dataset contains 200 pairs of low-quality license plate image sequences and high-quality pseudo ground-truth images, reflecting the complexities of real-world scenarios. In experiments, MF-LPR$^2$ outperformed eight recent restoration models in terms of PSNR, SSIM, and LPIPS by significant margins. In recognition, MF-LPR$^2$ achieved an accuracy of 86.44%, outperforming both the best single-frame LPR (14.04%) and the multi-frame LPR (82.55%) among the eleven baseline models. The results of ablation studies confirm that our filtering and refinement algorithms significantly contribute to these improvements. 

**Abstract (ZH)**: 基于多帧的 License Plate 修复与识别框架 (MF-LPR$^2$) 

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
# Reliable generation of isomorphic physics problems using ChatGPT with prompt-chaining and tool use 

**Title (ZH)**: 使用ChatGPT、提示链和工具使用生成可靠的同构物理问题 

**Authors**: Zhongzhou Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14755)  

**Abstract**: We present a method for generating large numbers of isomorphic physics problems using ChatGPT through prompt chaining and tool use. This approach enables precise control over structural variations-such as numeric values and spatial relations-while supporting diverse contextual variations in the problem body. By utilizing the Python code interpreter, the method supports automatic solution validation and simple diagram generation, addressing key limitations in existing LLM-based methods. We generated two example isomorphic problem banks and compared the outcome against simpler prompt-based approaches. Results show that prompt-chaining produces significantly higher quality and more consistent outputs than simpler, non-chaining prompts. This work demonstrates a promising method for efficient problem creation accessible to the average instructor, which opens new possibilities for personalized adaptive testing and automated content development. 

**Abstract (ZH)**: 使用ChatGPT通过提示链和工具使用生成大量同构物理问题的方法 

---
# Cross-Modality Controlled Molecule Generation with Diffusion Language Model 

**Title (ZH)**: 跨模态控制的分子生成差分语言模型 

**Authors**: Yunzhe Zhang, Yifei Wang, Khanh Vinh Nguyen, Pengyu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.14748)  

**Abstract**: Current SMILES-based diffusion models for molecule generation typically support only unimodal constraint. They inject conditioning signals at the start of the training process and require retraining a new model from scratch whenever the constraint changes. However, real-world applications often involve multiple constraints across different modalities, and additional constraints may emerge over the course of a study. This raises a challenge: how to extend a pre-trained diffusion model not only to support cross-modality constraints but also to incorporate new ones without retraining. To tackle this problem, we propose the Cross-Modality Controlled Molecule Generation with Diffusion Language Model (CMCM-DLM), demonstrated by two distinct cross modalities: molecular structure and chemical properties. Our approach builds upon a pre-trained diffusion model, incorporating two trainable modules, the Structure Control Module (SCM) and the Property Control Module (PCM), and operates in two distinct phases during the generation process. In Phase I, we employs the SCM to inject structural constraints during the early diffusion steps, effectively anchoring the molecular backbone. Phase II builds on this by further introducing PCM to guide the later stages of inference to refine the generated molecules, ensuring their chemical properties match the specified targets. Experimental results on multiple datasets demonstrate the efficiency and adaptability of our approach, highlighting CMCM-DLM's significant advancement in molecular generation for drug discovery applications. 

**Abstract (ZH)**: 跨模态控制分子生成的扩散语言模型（CMCM-DLM） 

---
# Evaluating Multilingual and Code-Switched Alignment in LLMs via Synthetic Natural Language Inference 

**Title (ZH)**: 评价多语言和代码切换对齐在大语言模型中的合成自然语言推理评估 

**Authors**: Samir Abdaljalil, Erchin Serpedin, Khalid Qaraqe, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2508.14735)  

**Abstract**: Large language models (LLMs) are increasingly applied in multilingual contexts, yet their capacity for consistent, logically grounded alignment across languages remains underexplored. We present a controlled evaluation framework for multilingual natural language inference (NLI) that generates synthetic, logic-based premise-hypothesis pairs and translates them into a typologically diverse set of languages. This design enables precise control over semantic relations and allows testing in both monolingual and mixed-language (code-switched) conditions. Surprisingly, code-switching does not degrade, and can even improve, performance, suggesting that translation-induced lexical variation may serve as a regularization signal. We validate semantic preservation through embedding-based similarity analyses and cross-lingual alignment visualizations, confirming the fidelity of translated pairs. Our findings expose both the potential and the brittleness of current LLM cross-lingual reasoning, and identify code-switching as a promising lever for improving multilingual robustness. Code available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在多语言场景中的应用日益增多，但其在不同语言中保持一致性和逻辑一致性的能力尚未得到充分探索。我们提出了一种控制性评估框架，用于多语言自然语言推理（NLI），生成合成的、基于逻辑的前提-假设对，并将其翻译成类型多样的一系列语言。这一设计允许对语义关系进行精确控制，并可以在单一语言和混合语言（代码转换）条件下进行测试。令人惊讶的是，代码转换不会降低性能，甚至可能提升性能，这表明翻译引起的词汇变化可能作为正则化信号发挥作用。我们通过嵌入式相似性分析和跨语言对齐可视化验证了语义保存的可靠性，证实了翻译后对齐的准确性。我们的研究揭示了当前LLM跨语言推理的潜力和脆弱性，并指出代码转换是提高多语言鲁棒性的有前景的杠杆。代码可在以下地址获得：this https URL。 

---
# AFABench: A Generic Framework for Benchmarking Active Feature Acquisition 

**Title (ZH)**: AFABench：动态特征获取的通用benchmark框架 

**Authors**: Valter Schütz, Han Wu, Reza Rezvan, Linus Aronsson, Morteza Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2508.14734)  

**Abstract**: In many real-world scenarios, acquiring all features of a data instance can be expensive or impractical due to monetary cost, latency, or privacy concerns. Active Feature Acquisition (AFA) addresses this challenge by dynamically selecting a subset of informative features for each data instance, trading predictive performance against acquisition cost. While numerous methods have been proposed for AFA, ranging from greedy information-theoretic strategies to non-myopic reinforcement learning approaches, fair and systematic evaluation of these methods has been hindered by the lack of standardized benchmarks. In this paper, we introduce AFABench, the first benchmark framework for AFA. Our benchmark includes a diverse set of synthetic and real-world datasets, supports a wide range of acquisition policies, and provides a modular design that enables easy integration of new methods and tasks. We implement and evaluate representative algorithms from all major categories, including static, greedy, and reinforcement learning-based approaches. To test the lookahead capabilities of AFA policies, we introduce a novel synthetic dataset, AFAContext, designed to expose the limitations of greedy selection. Our results highlight key trade-offs between different AFA strategies and provide actionable insights for future research. The benchmark code is available at: this https URL. 

**Abstract (ZH)**: 在许多实际场景中，获取数据实例的所有特征可能由于成本、延迟或隐私原因而昂贵或 impractical。主动特征获取（AFA）通过为每个数据实例动态选择一组有信息量的特征，权衡预测性能与获取成本。尽管已经提出了各种 AFA 方法，从贪婪的信息论策略到非短视的强化学习方法，但由于缺乏标准化基准，公正和系统的方法评估受到限制。在本文中，我们引入了 AFABench，这是第一个用于 AFA 的基准框架。我们的基准包括一系列合成和真实世界的数据集，支持广泛的获取策略，并提供模块化设计，便于新方法和任务的集成。我们实现了并评估了所有主要类别中的代表性算法，包括静态、贪婪和基于强化学习的方法。为了测试 AFA 策略的前瞻性能力，我们引入了一个新颖的合成数据集 AFAContext，旨在暴露贪婪选择的局限性。我们的结果突显了不同 AFA 策略之间的重要权衡，并为未来的研究提供了可操作的见解。基准代码可在以下链接访问：this https URL。 

---
# Emerson-Lei and Manna-Pnueli Games for LTLf+ and PPLTL+ Synthesis 

**Title (ZH)**: Emerson-Lei和Manna-Pnueli游戏在LTLf+和PPLTL+合成中的应用 

**Authors**: Daniel Hausmann, Shufang Zhu, Gianmarco Parretti, Christoph Weinhuber, Giuseppe De Giacomo, Nir Piterman  

**Link**: [PDF](https://arxiv.org/pdf/2508.14725)  

**Abstract**: Recently, the Manna-Pnueli Hierarchy has been used to define the temporal logics LTLfp and PPLTLp, which allow to use finite-trace LTLf/PPLTL techniques in infinite-trace settings while achieving the expressiveness of full LTL. In this paper, we present the first actual solvers for reactive synthesis in these logics. These are based on games on graphs that leverage DFA-based techniques from LTLf/PPLTL to construct the game arena. We start with a symbolic solver based on Emerson-Lei games, which reduces lower-class properties (guarantee, safety) to higher ones (recurrence, persistence) before solving the game. We then introduce Manna-Pnueli games, which natively embed Manna-Pnueli objectives into the arena. These games are solved by composing solutions to a DAG of simpler Emerson-Lei games, resulting in a provably more efficient approach. We implemented the solvers and practically evaluated their performance on a range of representative formulas. The results show that Manna-Pnueli games often offer significant advantages, though not universally, indicating that combining both approaches could further enhance practical performance. 

**Abstract (ZH)**: 最近，Manna-Pnueli 层次结构被用于定义时空逻辑LTLfp和PPLTLp，这些逻辑允许在无限轨迹设置中使用有限轨迹LTLf/PPLTL技术，同时保持完整LTL的表达能力。本文提出了这些逻辑中反应合成的第一个实际求解器。这些求解器基于图上的博弈，利用LTLf/PPLTL中的DFA基技术来构建博弈场地。我们首先基于Emerson-Lei博弈的符号求解器，将其较低级别的属性（保证、安全性）转化为高级属性（循环性、持久性）后再求解游戏。然后引入了Manna-Pnueli博弈，这些博弈自然地将Manna-Pnueli目标嵌入到博弈场地中。这些博弈通过组合简单Emerson-Lei博弈的解来求解，从而提供了一个可证明更高效的方法。我们实现并实际评估了这些求解器在一系列代表性公式的性能。结果表明，Manna-Pnueli博弈通常提供了显著的优势，尽管不是普遍适用于所有情况，表明结合两种方法可能进一步提高实际性能。 

---
# Transplant Then Regenerate: A New Paradigm for Text Data Augmentation 

**Title (ZH)**: 移植然后再生：一种新的文本数据扩增范式 

**Authors**: Guangzhan Wang, Hongyu Zhang, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14723)  

**Abstract**: Data augmentation is a critical technique in deep learning. Traditional methods like Back-translation typically focus on lexical-level rephrasing, which primarily produces variations with the same semantics. While large language models (LLMs) have enhanced text augmentation by their "knowledge emergence" capability, controlling the style and structure of these outputs remains challenging and requires meticulous prompt engineering. In this paper, we propose LMTransplant, a novel text augmentation paradigm leveraging LLMs. The core idea of LMTransplant is transplant-then-regenerate: incorporating seed text into a context expanded by LLM, and asking the LLM to regenerate a variant based on the expanded context. This strategy allows the model to create more diverse and creative content-level variants by fully leveraging the knowledge embedded in LLMs, while preserving the core attributes of the original text. We evaluate LMTransplant across various text-related tasks, demonstrating its superior performance over existing text augmentation methods. Moreover, LMTransplant demonstrates exceptional scalability as the size of augmented data grows. 

**Abstract (ZH)**: 数据扩充是深度学习中的一种关键技术。传统方法如反向翻译通常专注于词汇层面的重写，主要产生具有相同语义的变化。虽然大规模语言模型（LLMs）通过其“知识涌现”能力增强了文本扩充能力，但在控制这些输出的风格和结构方面仍具有挑战性，需要精细的提示工程。本文提出了一种名为LMTransplant的新型文本扩充范式，利用LLMs。LMTransplant的核心思想是移植-再生：将种子文本融入由LLM扩大的上下文中，要求LLM基于扩大的上下文生成一个变体。这种策略允许模型充分利用LLMs中嵌入的知识来创建更多样化和创造性的内容级变体，同时保留原始文本的核心属性。我们对LMTransplant在各种文本相关任务中进行了评估，展示了其相对于现有文本扩充方法的优越性能。此外，LMTransplant在扩充数据量增大时表现出色。 

---
# ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine 

**Title (ZH)**: .shizhenGPT:向量跨模态中医药大型语言模型 

**Authors**: Junying Chen, Zhenyang Cai, Zhiheng Liu, Yunjin Yang, Rongsheng Wang, Qingying Xiao, Xiangyi Feng, Zhan Su, Jing Guo, Xiang Wan, Guangjun Yu, Haizhou Li, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14706)  

**Abstract**: Despite the success of large language models (LLMs) in various domains, their potential in Traditional Chinese Medicine (TCM) remains largely underexplored due to two critical barriers: (1) the scarcity of high-quality TCM data and (2) the inherently multimodal nature of TCM diagnostics, which involve looking, listening, smelling, and pulse-taking. These sensory-rich modalities are beyond the scope of conventional LLMs. To address these challenges, we present ShizhenGPT, the first multimodal LLM tailored for TCM. To overcome data scarcity, we curate the largest TCM dataset to date, comprising 100GB+ of text and 200GB+ of multimodal data, including 1.2M images, 200 hours of audio, and physiological signals. ShizhenGPT is pretrained and instruction-tuned to achieve deep TCM knowledge and multimodal reasoning. For evaluation, we collect recent national TCM qualification exams and build a visual benchmark for Medicinal Recognition and Visual Diagnosis. Experiments demonstrate that ShizhenGPT outperforms comparable-scale LLMs and competes with larger proprietary models. Moreover, it leads in TCM visual understanding among existing multimodal LLMs and demonstrates unified perception across modalities like sound, pulse, smell, and vision, paving the way toward holistic multimodal perception and diagnosis in TCM. Datasets, models, and code are publicly available. We hope this work will inspire further exploration in this field. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各个领域取得了成功，但在传统中医（TCM）领域的潜力仍因两大关键障碍而未得到充分探索：（1）高质量TCM数据稀缺；（2）TCM诊断的固有多模态性质，涉及到看、听、闻和切脉。这些富含感官信息的模态超出了传统LLMs的功能范围。为应对这些挑战，我们提出了ShizhenGPT，这是首个专门为TCM设计的多模态LLM。为克服数据稀缺问题，我们编纂了迄今为止最大规模的TCM数据集，包含超过100GB的文本和超过200GB的多模态数据，其中包括120万张图像、200小时的音频和生理信号。ShizhenGPT进行预训练和指令调整，以实现深厚的TCM知识和多模态推理。为评估效果，我们收集了最新的国家级TCM资格考试，并建立了一个视觉基准用于药用识别和视觉诊断。实验表明，ShizhenGPT优于同等规模的LLMs，并能与更大的专属模型竞争。此外，ShizhenGPT在现有多模态LLMs中领先于TCM视觉理解，并展示了横跨听觉、脉搏、嗅觉和视觉等模态的统一感知能力，为TCM的全貌多模态感知和诊断铺平了道路。数据集、模型和代码已公开。我们希望这项工作能激发更多对该领域的探索。 

---
# Learning in Repeated Multi-Objective Stackelberg Games with Payoff Manipulation 

**Title (ZH)**: 重复多目标斯塔克尔贝格博弈中的收益操纵学习 

**Authors**: Phurinut Srisawad, Juergen Branke, Long Tran-Thanh  

**Link**: [PDF](https://arxiv.org/pdf/2508.14705)  

**Abstract**: We study payoff manipulation in repeated multi-objective Stackelberg games, where a leader may strategically influence a follower's deterministic best response, e.g., by offering a share of their own payoff. We assume that the follower's utility function, representing preferences over multiple objectives, is unknown but linear, and its weight parameter must be inferred through interaction. This introduces a sequential decision-making challenge for the leader, who must balance preference elicitation with immediate utility maximisation. We formalise this problem and propose manipulation policies based on expected utility (EU) and long-term expected utility (longEU), which guide the leader in selecting actions and offering incentives that trade off short-term gains with long-term impact. We prove that under infinite repeated interactions, longEU converges to the optimal manipulation. Empirical results across benchmark environments demonstrate that our approach improves cumulative leader utility while promoting mutually beneficial outcomes, all without requiring explicit negotiation or prior knowledge of the follower's utility function. 

**Abstract (ZH)**: 我们在重复的多目标Stackelberg游戏中研究收益操纵问题，其中领导者可以通过提供自身收益的一部分等方式战略性地影响跟随者的确定性最佳响应。我们假设跟随者的效用函数，代表其在多个目标上的偏好，是未知的但线性的，其权重参数必须通过互动进行推断。这为领导者引入了一个 sequenced 决策挑战：领导者必须在偏好诱导与即时效用最大化之间寻求平衡。我们对该问题进行形式化，并提出基于期望效用(EU)和长期期望效用(longEU)的操纵策略，这些策略指导领导者在权衡短期利益与长期影响之间做出行动选择和激励措施。我们证明，在无限重复互动下，longEU 收敛于最优操纵。我们的方法在基准环境中的实证结果表明，我们的方法能够提高领导者的累积效用，促进互惠互利的结果，而无需进行显式的谈判或了解跟随者的效用函数。 

---
# Foe for Fraud: Transferable Adversarial Attacks in Credit Card Fraud Detection 

**Title (ZH)**: 为欺诈者设圈套：信用卡欺诈检测中的可转移 adversarial 攻击 

**Authors**: Jan Lum Fok, Qingwen Zeng, Shiping Chen, Oscar Fawkes, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14699)  

**Abstract**: Credit card fraud detection (CCFD) is a critical application of Machine Learning (ML) in the financial sector, where accurately identifying fraudulent transactions is essential for mitigating financial losses. ML models have demonstrated their effectiveness in fraud detection task, in particular with the tabular dataset. While adversarial attacks have been extensively studied in computer vision and deep learning, their impacts on the ML models, particularly those trained on CCFD tabular datasets, remains largely unexplored. These latent vulnerabilities pose significant threats to the security and stability of the financial industry, especially in high-value transactions where losses could be substantial. To address this gap, in this paper, we present a holistic framework that investigate the robustness of CCFD ML model against adversarial perturbations under different circumstances. Specifically, the gradient-based attack methods are incorporated into the tabular credit card transaction data in both black- and white-box adversarial attacks settings. Our findings confirm that tabular data is also susceptible to subtle perturbations, highlighting the need for heightened awareness among financial technology practitioners regarding ML model security and trustworthiness. Furthermore, the experiments by transferring adversarial samples from gradient-based attack method to non-gradient-based models also verify our findings. Our results demonstrate that such attacks remain effective, emphasizing the necessity of developing robust defenses for CCFD algorithms. 

**Abstract (ZH)**: 信用卡欺诈检测中的对抗攻击研究：基于机器学习的全面框架 

---
# ECHO: Frequency-aware Hierarchical Encoding for Variable-length Signal 

**Title (ZH)**: ECHO：频域aware分层编码用于变长信号 

**Authors**: Yucong Zhang, Juan Liu, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14689)  

**Abstract**: Pre-trained foundation models have demonstrated remarkable success in vision and language, yet their potential for general machine signal modeling-covering acoustic, vibration, and other industrial sensor data-remains under-explored. Existing approach using sub-band-based encoders has achieved competitive results but are limited by fixed input lengths, and the absence of explicit frequency positional encoding. In this work, we propose a novel foundation model that integrates an advanced band-split architecture with relative frequency positional embeddings, enabling precise spectral localization across arbitrary sampling configurations. The model supports inputs of arbitrary length without padding or segmentation, producing a concise embedding that retains both temporal and spectral fidelity. We evaluate our method on SIREN (this https URL), a newly introduced large-scale benchmark for machine signal encoding that unifies multiple datasets, including all DCASE task 2 challenges (2020-2025) and widely-used industrial signal corpora. Experimental results demonstrate consistent state-of-the-art performance in anomaly detection and fault identification, confirming the effectiveness and generalization capability of the proposed model. We open-sourced ECHO on this https URL. 

**Abstract (ZH)**: 预训练基础模型在视觉和语言领域取得了显著成功，但在声学、振动及其他工业传感器数据的一般机器信号建模方面尚存在巨大的探索空间。现有基于子带的编码器方法取得了竞争性的结果，但受到固定输入长度和缺乏显式频率位置编码的限制。本文提出了一种新型基础模型，结合了先进的带分段架构和相对频率位置嵌入，使得模型能够在任意采样配置下实现精确的频谱定位。该模型支持任意长度的输入，无需填充或切分，生成一个简洁的嵌入，保留了时间域和频谱的保真度。我们在SIREN（this https URL）上对该方法进行了评估，SIREN是一个新的大规模基准测试，用于机器信号编码，整合了多个数据集，包括所有DCASE任务2挑战（2020-2025）和广泛使用的工业信号数据集。实验结果表明，该方法在异常检测和故障识别方面保持了领先性能，证实了所提出模型的有效性和泛化能力。我们在该链接（this https URL）上开源了ECHO。 

---
# ELATE: Evolutionary Language model for Automated Time-series Engineering 

**Title (ZH)**: ELATE：演化语言模型驱动的时间序列工程自动化 

**Authors**: Andrew Murray, Danial Dervovic, Michael Cashmore  

**Link**: [PDF](https://arxiv.org/pdf/2508.14667)  

**Abstract**: Time-series prediction involves forecasting future values using machine learning models. Feature engineering, whereby existing features are transformed to make new ones, is critical for enhancing model performance, but is often manual and time-intensive. Existing automation attempts rely on exhaustive enumeration, which can be computationally costly and lacks domain-specific insights. We introduce ELATE (Evolutionary Language model for Automated Time-series Engineering), which leverages a language model within an evolutionary framework to automate feature engineering for time-series data. ELATE employs time-series statistical measures and feature importance metrics to guide and prune features, while the language model proposes new, contextually relevant feature transformations. Our experiments demonstrate that ELATE improves forecasting accuracy by an average of 8.4% across various domains. 

**Abstract (ZH)**: 时间序列预测涉及使用机器学习模型进行未来值的预测。特征工程，即通过转换现有特征来创建新特征，对于提升模型性能至关重要，但通常需要手动进行且耗时。现有自动化尝试依赖于穷尽枚举，这可能计算成本高且缺乏特定领域的洞见。我们介绍了ELATE（进化语言模型驱动的时间序列自动化工程），它在进化框架中利用语言模型来自动进行时间序列数据的特征工程。ELATE利用时间序列统计量和特征重要性指标来指导和修剪特征，同时语言模型提出与上下文相关的新特征转换。我们的实验表明，ELATE在各种领域中平均提高了8.4%的预测准确性。 

---
# OneLoc: Geo-Aware Generative Recommender Systems for Local Life Service 

**Title (ZH)**: OneLoc：基于地理位置的生成型推荐系统本地生活服务 

**Authors**: Zhipeng Wei, Kuo Cai, Junda She, Jie Chen, Minghao Chen, Yang Zeng, Qiang Luo, Wencong Zeng, Ruiming Tang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.14646)  

**Abstract**: Local life service is a vital scenario in Kuaishou App, where video recommendation is intrinsically linked with store's location information. Thus, recommendation in our scenario is challenging because we should take into account user's interest and real-time location at the same time. In the face of such complex scenarios, end-to-end generative recommendation has emerged as a new paradigm, such as OneRec in the short video scenario, OneSug in the search scenario, and EGA in the advertising scenario. However, in local life service, an end-to-end generative recommendation model has not yet been developed as there are some key challenges to be solved. The first challenge is how to make full use of geographic information. The second challenge is how to balance multiple objectives, including user interests, the distance between user and stores, and some other business objectives. To address the challenges, we propose OneLoc. Specifically, we leverage geographic information from different perspectives: (1) geo-aware semantic ID incorporates both video and geographic information for tokenization, (2) geo-aware self-attention in the encoder leverages both video location similarity and user's real-time location, and (3) neighbor-aware prompt captures rich context information surrounding users for generation. To balance multiple objectives, we use reinforcement learning and propose two reward functions, i.e., geographic reward and GMV reward. With the above design, OneLoc achieves outstanding offline and online performance. In fact, OneLoc has been deployed in local life service of Kuaishou App. It serves 400 million active users daily, achieving 21.016% and 17.891% improvements in terms of gross merchandise value (GMV) and orders numbers. 

**Abstract (ZH)**: 本地生活服务中的端到端生成推荐：OneLoc方法 

---
# Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination 

**Title (ZH)**: LLM代理能否解决协作任务？一种基于紧迫性感知的规划与协调研究 

**Authors**: João Vitor de Carvalho Silva, Douglas G. Macharet  

**Link**: [PDF](https://arxiv.org/pdf/2508.14635)  

**Abstract**: The ability to coordinate actions across multiple agents is critical for solving complex, real-world problems. Large Language Models (LLMs) have shown strong capabilities in communication, planning, and reasoning, raising the question of whether they can also support effective collaboration in multi-agent settings. In this work, we investigate the use of LLM agents to solve a structured victim rescue task that requires division of labor, prioritization, and cooperative planning. Agents operate in a fully known graph-based environment and must allocate resources to victims with varying needs and urgency levels. We systematically evaluate their performance using a suite of coordination-sensitive metrics, including task success rate, redundant actions, room conflicts, and urgency-weighted efficiency. This study offers new insights into the strengths and failure modes of LLMs in physically grounded multi-agent collaboration tasks, contributing to future benchmarks and architectural improvements. 

**Abstract (ZH)**: 跨多个代理协调行动的能力对于解决复杂的真实世界问题是至关重要的。大规模语言模型（LLMs）在通信、规划和推理方面表现出强大的能力，引起了一个问题，即它们是否也能在多代理环境中支持有效的合作。在本研究中，我们探讨了使用LLM代理解决一个需要分工、优先处理和协同规划的结构化救援任务的可能性。代理在完全已知的图基环境中操作，并必须根据受害者的不同需求和紧迫性级别分配资源。我们使用一系列敏感于协调的指标系统性地评估其性能，包括任务成功率、冗余行动、房间冲突和紧迫性加权效率。这项研究为物理环境中多代理合作任务中LLMs的优势和失效模式提供了新的见解，贡献于未来的基准测试和架构改进。 

---
# A Study of the Scale Invariant Signal to Distortion Ratio in Speech Separation with Noisy References 

**Title (ZH)**: 噪声参考下语音分离中不变尺度信噪比的研究 

**Authors**: Simon Dahl Jepsen, Mads Græsbøll Christensen, Jesper Rindom Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14623)  

**Abstract**: This paper examines the implications of using the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) as both evaluation and training objective in supervised speech separation, when the training references contain noise, as is the case with the de facto benchmark WSJ0-2Mix. A derivation of the SI-SDR with noisy references reveals that noise limits the achievable SI-SDR, or leads to undesired noise in the separated outputs. To address this, a method is proposed to enhance references and augment the mixtures with WHAM!, aiming to train models that avoid learning noisy references. Two models trained on these enhanced datasets are evaluated with the non-intrusive NISQA.v2 metric. Results show reduced noise in separated speech but suggest that processing references may introduce artefacts, limiting overall quality gains. Negative correlation is found between SI-SDR and perceived noisiness across models on the WSJ0-2Mix and Libri2Mix test sets, underlining the conclusion from the derivation. 

**Abstract (ZH)**: 使用带有噪声参考的尺度不变信干比（SI-SDR）作为监督语音分离中的评估和训练目标的影响研究：增强参考并结合WHAM!进行混合以避免学习噪声参考 

---
# UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling 

**Title (ZH)**: UST-SSM：统一时空状态空间模型及其在点云视频建模中的应用 

**Authors**: Peiming Li, Ziyi Wang, Yulin Yuan, Hong Liu, Xiangming Meng, Junsong Yuan, Mengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14604)  

**Abstract**: Point cloud videos capture dynamic 3D motion while reducing the effects of lighting and viewpoint variations, making them highly effective for recognizing subtle and continuous human actions. Although Selective State Space Models (SSMs) have shown good performance in sequence modeling with linear complexity, the spatio-temporal disorder of point cloud videos hinders their unidirectional modeling when directly unfolding the point cloud video into a 1D sequence through temporally sequential scanning. To address this challenge, we propose the Unified Spatio-Temporal State Space Model (UST-SSM), which extends the latest advancements in SSMs to point cloud videos. Specifically, we introduce Spatial-Temporal Selection Scanning (STSS), which reorganizes unordered points into semantic-aware sequences through prompt-guided clustering, thereby enabling the effective utilization of points that are spatially and temporally distant yet similar within the sequence. For missing 4D geometric and motion details, Spatio-Temporal Structure Aggregation (STSA) aggregates spatio-temporal features and compensates. To improve temporal interaction within the sampled sequence, Temporal Interaction Sampling (TIS) enhances fine-grained temporal dependencies through non-anchor frame utilization and expanded receptive fields. Experimental results on the MSR-Action3D, NTU RGB+D, and Synthia 4D datasets validate the effectiveness of our method. Our code is available at this https URL. 

**Abstract (ZH)**: 点云视频捕捉动态三维运动同时减少光照和视角变化的影响，使其适用于识别细微且连贯的人体动作。尽管选择性状态空间模型（SSMs）在序列建模中表现出良好的性能且具有线性复杂度，但点云视频的时空杂乱阻碍了其通过时间序扫描直接将点云视频展开为一维序列的一维建模。为解决这一挑战，我们提出了一体化时空状态空间模型（UST-SSM），该模型将最新的SSM进展扩展应用于点云视频。具体来说，我们引入了时空选择扫描（STSS），通过提示引导聚类重新组织无序点为语义感知序列，从而有效利用在序列中时空上相距但相似的点。对于缺失的4D几何和运动细节，时空结构聚合（STSA）聚合时空特征并进行补偿。为了改善采样序列内的时域交互，时域交互采样（TIS）通过利用非锚帧和扩展的感受野增强细粒度时域依赖性。在 MSR-Action3D、NTU RGB+D 和 Synthia 4D 数据集上的实验结果验证了我们方法的有效性。我们的代码可在以下网址获取：this https URL。 

---
# An Open-Source HW-SW Co-Development Framework Enabling Efficient Multi-Accelerator Systems 

**Title (ZH)**: 开源硬件-软件协同开发框架，用于高效多加速器系统 

**Authors**: Ryan Albert Antonio, Joren Dumoulin, Xiaoling Yi, Josse Van Delm, Yunhao Deng, Guilherme Paim, Marian Verhelst  

**Link**: [PDF](https://arxiv.org/pdf/2508.14582)  

**Abstract**: Heterogeneous accelerator-centric compute clusters are emerging as efficient solutions for diverse AI workloads. However, current integration strategies often compromise data movement efficiency and encounter compatibility issues in hardware and software. This prevents a unified approach that balances performance and ease of use. To this end, we present SNAX, an open-source integrated HW-SW framework enabling efficient multi-accelerator platforms through a novel hybrid-coupling scheme, consisting of loosely coupled asynchronous control and tightly coupled data access. SNAX brings reusable hardware modules designed to enhance compute accelerator utilization, and its customizable MLIR-based compiler to automate key system management tasks, jointly enabling rapid development and deployment of customized multi-accelerator compute clusters. Through extensive experimentation, we demonstrate SNAX's efficiency and flexibility in a low-power heterogeneous SoC. Accelerators can easily be integrated and programmed to achieve > 10x improvement in neural network performance compared to other accelerator systems while maintaining accelerator utilization of > 90% in full system operation. 

**Abstract (ZH)**: 异构加速器为中心的计算集群正 emerged as efficient solutions for diverse AI工作负载。然而，当前的集成策略往往牺牲了数据移动效率，并在硬件和软件兼容性上遇到问题。这阻碍了兼顾性能和使用便捷性的统一方法。为此，我们提出了SNAX，这是一种开源的软硬件整合框架，通过一种新颖的混合耦合方案——松耦合异步控制和紧耦合数据访问，实现高效的多加速器平台。SNAX集成了可重用的硬件模块以增强计算加速器利用率，并配备了基于MLIR的可定制编译器以自动化关键系统管理任务，共同支持定制多加速器计算集群的快速开发和部署。通过广泛实验，我们展示了SNAX在低功耗异构SoC上的效率和灵活性。加速器可以轻松集成和编程，相比其他加速器系统， neural network性能提升超过10倍，同时在整个系统运行中保持加速器利用率高于90%。 

---
# Mamba2 Meets Silence: Robust Vocal Source Separation for Sparse Regions 

**Title (ZH)**: Mamba2 遇上 Silence：稀疏区域稳健的语音源分离 

**Authors**: Euiyeon Kim, Yong-Hoon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14556)  

**Abstract**: We introduce a new music source separation model tailored for accurate vocal isolation. Unlike Transformer-based approaches, which often fail to capture intermittently occurring vocals, our model leverages Mamba2, a recent state space model, to better capture long-range temporal dependencies. To handle long input sequences efficiently, we combine a band-splitting strategy with a dual-path architecture. Experiments show that our approach outperforms recent state-of-the-art models, achieving a cSDR of 11.03 dB-the best reported to date-and delivering substantial gains in uSDR. Moreover, the model exhibits stable and consistent performance across varying input lengths and vocal occurrence patterns. These results demonstrate the effectiveness of Mamba-based models for high-resolution audio processing and open up new directions for broader applications in audio research. 

**Abstract (ZH)**: 我们提出了一种新的音乐源分离模型，专门用于准确的声乐隔离。与經常出现捕捉不力的变换器基方法不同，我们的模型利用了Mamba2这一近期的状态空间模型，更好地捕捉了长期时间依赖关系。为了高效处理长输入序列，我们结合了带分割策略和双路径架构。实验结果显示，我们的方法优于最近的最先进的模型，实现了迄今为止报道的最佳cSDR值11.03 dB，并在uSDR上取得了显著的提升。此外，该模型在不同输入长度和声乐出现模式下表现出稳定且一致的性能。这些结果证明了基于Mamba的模型在高分辨率音频处理中的有效性，并为更广泛的声音研究应用打开了新的方向。 

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
# Beyond ReLU: Chebyshev-DQN for Enhanced Deep Q-Networks 

**Title (ZH)**: 超越ReLU：Chebyshev-DQN 以增强深度Q网络 

**Authors**: Saman Yazdannik, Morteza Tayefi, Shamim Sanisales  

**Link**: [PDF](https://arxiv.org/pdf/2508.14536)  

**Abstract**: The performance of Deep Q-Networks (DQN) is critically dependent on the ability of its underlying neural network to accurately approximate the action-value function. Standard function approximators, such as multi-layer perceptrons, may struggle to efficiently represent the complex value landscapes inherent in many reinforcement learning problems. This paper introduces a novel architecture, the Chebyshev-DQN (Ch-DQN), which integrates a Chebyshev polynomial basis into the DQN framework to create a more effective feature representation. By leveraging the powerful function approximation properties of Chebyshev polynomials, we hypothesize that the Ch-DQN can learn more efficiently and achieve higher performance. We evaluate our proposed model on the CartPole-v1 benchmark and compare it against a standard DQN with a comparable number of parameters. Our results demonstrate that the Ch-DQN with a moderate polynomial degree (N=4) achieves significantly better asymptotic performance, outperforming the baseline by approximately 39\%. However, we also find that the choice of polynomial degree is a critical hyperparameter, as a high degree (N=8) can be detrimental to learning. This work validates the potential of using orthogonal polynomial bases in deep reinforcement learning while also highlighting the trade-offs involved in model complexity. 

**Abstract (ZH)**: Chebyshev-DQN：将切比雪夫多项式基集成到DQN框架中以提高价值函数表示效果 

---
# EffiFusion-GAN: Efficient Fusion Generative Adversarial Network for Speech Enhancement 

**Title (ZH)**: EffiFusion-GAN: 高效融合生成对抗网络用于语音增强 

**Authors**: Bin Wen, Tien-Ping Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14525)  

**Abstract**: We introduce EffiFusion-GAN (Efficient Fusion Generative Adversarial Network), a lightweight yet powerful model for speech enhancement. The model integrates depthwise separable convolutions within a multi-scale block to capture diverse acoustic features efficiently. An enhanced attention mechanism with dual normalization and residual refinement further improves training stability and convergence. Additionally, dynamic pruning is applied to reduce model size while maintaining performance, making the framework suitable for resource-constrained environments. Experimental evaluation on the public VoiceBank+DEMAND dataset shows that EffiFusion-GAN achieves a PESQ score of 3.45, outperforming existing models under the same parameter settings. 

**Abstract (ZH)**: EffiFusion-GAN：一种轻量而强大的语音增强生成对抗网络 

---
# MISS: Multi-Modal Tree Indexing and Searching with Lifelong Sequential Behavior for Retrieval Recommendation 

**Title (ZH)**: MISS：基于终身序列行为的多模态树索引和检索推荐 

**Authors**: Chengcheng Guo, Junda She, Kuo Cai, Shiyao Wang, Qigen Hu, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.14515)  

**Abstract**: Large-scale industrial recommendation systems typically employ a two-stage paradigm of retrieval and ranking to handle huge amounts of information. Recent research focuses on improving the performance of retrieval model. A promising way is to introduce extensive information about users and items. On one hand, lifelong sequential behavior is valuable. Existing lifelong behavior modeling methods in ranking stage focus on the interaction of lifelong behavior and candidate items from retrieval stage. In retrieval stage, it is difficult to utilize lifelong behavior because of a large corpus of candidate items. On the other hand, existing retrieval methods mostly relay on interaction information, potentially disregarding valuable multi-modal information. To solve these problems, we represent the pioneering exploration of leveraging multi-modal information and lifelong sequence model within the advanced tree-based retrieval model. We propose Multi-modal Indexing and Searching with lifelong Sequence (MISS), which contains a multi-modal index tree and a multi-modal lifelong sequence modeling module. Specifically, for better index structure, we propose multi-modal index tree, which is built using the multi-modal embedding to precisely represent item similarity. To precisely capture diverse user interests in user lifelong sequence, we propose collaborative general search unit (Co-GSU) and multi-modal general search unit (MM-GSU) for multi-perspective interests searching. 

**Abstract (ZH)**: 大规模工业推荐系统通常采用检索和排名的两阶段范式来处理大量信息。近期研究集中在提高检索模型的性能上。一种有前景的方法是引入大量的用户和项目信息。一方面，终生顺序行为很有价值。排名阶段现有的终生行为建模方法主要关注终生行为与检索阶段候选项目的交互。在检索阶段，难以利用终生行为，因为候选项的规模很大。另一方面，现有的检索方法主要依赖交互信息，可能会忽略有价值的多模态信息。为了解决这些问题，我们开展了利用多模态信息和终生序列模型的开创性探索，将其置于先进的树状检索模型中。我们提出了多模态终生序列模型索引与搜索（MISS），该模型包含多模态索引树和多模态终生序列建模模块。具体而言，为了获得更好的索引结构，我们提出了基于多模态嵌入的多模态索引树，以精确表示项目相似性。为了精确捕捉用户终生序列中的多样兴趣，我们提出协作通用搜索单元（Co-GSU）和多模态通用搜索单元（MM-GSU）进行多视角兴趣搜索。 

---
# PB-IAD: Utilizing multimodal foundation models for semantic industrial anomaly detection in dynamic manufacturing environments 

**Title (ZH)**: PB-IAD：利用多模态基础模型在动态制造环境中进行语义工业异常检测 

**Authors**: Bernd Hofmann, Albert Scheck, Joerg Franke, Patrick Bruendl  

**Link**: [PDF](https://arxiv.org/pdf/2508.14504)  

**Abstract**: The detection of anomalies in manufacturing processes is crucial to ensure product quality and identify process deviations. Statistical and data-driven approaches remain the standard in industrial anomaly detection, yet their adaptability and usability are constrained by the dependence on extensive annotated datasets and limited flexibility under dynamic production conditions. Recent advances in the perception capabilities of foundation models provide promising opportunities for their adaptation to this downstream task. This paper presents PB-IAD (Prompt-based Industrial Anomaly Detection), a novel framework that leverages the multimodal and reasoning capabilities of foundation models for industrial anomaly detection. Specifically, PB-IAD addresses three key requirements of dynamic production environments: data sparsity, agile adaptability, and domain user centricity. In addition to the anomaly detection, the framework includes a prompt template that is specifically designed for iteratively implementing domain-specific process knowledge, as well as a pre-processing module that translates domain user inputs into effective system prompts. This user-centric design allows domain experts to customise the system flexibly without requiring data science expertise. The proposed framework is evaluated by utilizing GPT-4.1 across three distinct manufacturing scenarios, two data modalities, and an ablation study to systematically assess the contribution of semantic instructions. Furthermore, PB-IAD is benchmarked to state-of-the-art methods for anomaly detection such as PatchCore. The results demonstrate superior performance, particularly in data-sparse scenarios and low-shot settings, achieved solely through semantic instructions. 

**Abstract (ZH)**: 基于提示的工业异常检测（Prompt-based Industrial Anomaly Detection） 

---
# Exact Shapley Attributions in Quadratic-time for FANOVA Gaussian Processes 

**Title (ZH)**: 精确的Shapley归因在 Quadratic 时间内的FANOVA Gaussian 过程 

**Authors**: Majid Mohammadi, Krikamol Muandet, Ilaria Tiddi, Annette Ten Teije, Siu Lun Chau  

**Link**: [PDF](https://arxiv.org/pdf/2508.14499)  

**Abstract**: Shapley values are widely recognized as a principled method for attributing importance to input features in machine learning. However, the exact computation of Shapley values scales exponentially with the number of features, severely limiting the practical application of this powerful approach. The challenge is further compounded when the predictive model is probabilistic - as in Gaussian processes (GPs) - where the outputs are random variables rather than point estimates, necessitating additional computational effort in modeling higher-order moments. In this work, we demonstrate that for an important class of GPs known as FANOVA GP, which explicitly models all main effects and interactions, *exact* Shapley attributions for both local and global explanations can be computed in *quadratic time*. For local, instance-wise explanations, we define a stochastic cooperative game over function components and compute the exact stochastic Shapley value in quadratic time only, capturing both the expected contribution and uncertainty. For global explanations, we introduce a deterministic, variance-based value function and compute exact Shapley values that quantify each feature's contribution to the model's overall sensitivity. Our methods leverage a closed-form (stochastic) Möbius representation of the FANOVA decomposition and introduce recursive algorithms, inspired by Newton's identities, to efficiently compute the mean and variance of Shapley values. Our work enhances the utility of explainable AI, as demonstrated by empirical studies, by providing more scalable, axiomatically sound, and uncertainty-aware explanations for predictions generated by structured probabilistic models. 

**Abstract (ZH)**: FANOVA高斯过程的精确Shapley归属：基于阶层时刻的高效计算 

---
# Synaptic bundle theory for spike-driven sensor-motor system: More than eight independent synaptic bundles collapse reward-STDP learning 

**Title (ZH)**: 突触束理论在由尖峰驱动的感觉-运动系统中的应用：超过八条独立的突触束塌缩奖励-STDP学习 

**Authors**: Takeshi Kobayashi, Shogo Yonekura, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14492)  

**Abstract**: Neuronal spikes directly drive muscles and endow animals with agile movements, but applying the spike-based control signals to actuators in artificial sensor-motor systems inevitably causes a collapse of learning. We developed a system that can vary \emph{the number of independent synaptic bundles} in sensor-to-motor connections. This paper demonstrates the following four findings: (i) Learning collapses once the number of motor neurons or the number of independent synaptic bundles exceeds a critical limit. (ii) The probability of learning failure is increased by a smaller number of motor neurons, while (iii) if learning succeeds, a smaller number of motor neurons leads to faster learning. (iv) The number of weight updates that move in the opposite direction of the optimal weight can quantitatively explain these results. The functions of spikes remain largely unknown. Identifying the parameter range in which learning systems using spikes can be constructed will make it possible to study the functions of spikes that were previously inaccessible due to the difficulty of learning. 

**Abstract (ZH)**: 神经元尖峰直接驱动肌肉并赋予动物敏捷运动，但将基于尖峰的控制信号应用于人工传感器-效应器系统的执行器会导致学习崩溃。我们开发了一个可以变动传感器-效应器连接中独立突触束数量的系统。本文展示了以下四项发现：（i）一旦神经元数量或独立突触束数量超过某一临界限度，学习就会崩溃。（ii）较小数量的效应器神经元会增加学习失败的概率，而（iii）如果学习成功，较小数量的效应器神经元会使得学习更快。（iv）反向于最优权值更新的数量可以定量解释这些结果。尖峰的功能仍然 largely unknown。确定可以构建尖峰学习系统的参数范围将使得之前因学习难度而无法研究的功能变得可研究。 

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
# Detecting Reading-Induced Confusion Using EEG and Eye Tracking 

**Title (ZH)**: 基于EEG和眼动跟踪检测阅读诱导的混淆 

**Authors**: Haojun Zhuang, Dünya Baradari, Nataliya Kosmyna, Arnav Balyan, Constanze Albrecht, Stephanie Chen, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2508.14442)  

**Abstract**: Humans regularly navigate an overwhelming amount of information via text media, whether reading articles, browsing social media, or interacting with chatbots. Confusion naturally arises when new information conflicts with or exceeds a reader's comprehension or prior knowledge, posing a challenge for learning. In this study, we present a multimodal investigation of reading-induced confusion using EEG and eye tracking. We collected neural and gaze data from 11 adult participants as they read short paragraphs sampled from diverse, real-world sources. By isolating the N400 event-related potential (ERP), a well-established neural marker of semantic incongruence, and integrating behavioral markers from eye tracking, we provide a detailed analysis of the neural and behavioral correlates of confusion during naturalistic reading. Using machine learning, we show that multimodal (EEG + eye tracking) models improve classification accuracy by 4-22% over unimodal baselines, reaching an average weighted participant accuracy of 77.3% and a best accuracy of 89.6%. Our results highlight the dominance of the brain's temporal regions in these neural signatures of confusion, suggesting avenues for wearable, low-electrode brain-computer interfaces (BCI) for real-time monitoring. These findings lay the foundation for developing adaptive systems that dynamically detect and respond to user confusion, with potential applications in personalized learning, human-computer interaction, and accessibility. 

**Abstract (ZH)**: 人类通过文本媒体阅读文章、浏览社交媒体或与聊天机器人互动时，会定期处理大量的信息。当新信息与读者的理解或先前知识冲突时，会产生困惑，这对学习形成挑战。本研究通过脑电图（EEG）和眼动追踪技术，对阅读引起的困惑进行了多模态调查。我们收集了11名成人阅读来自多样化真实来源短段落的神经和注视数据。通过分离N400事件相关电位（ERP），一种已建立的语言语义不匹配的神经标志物，并结合眼动追踪的行为标志物，我们对自然阅读过程中困惑的神经和行为相关性进行了详细分析。通过机器学习，我们展示了多模态（EEG + 眼动追踪）模型相比单一模态基线模型，分类准确率提高了4-22%，平均加权参与者准确率为77.3%，最高准确率为89.6%。我们的研究结果突显了大脑颞区在这些困惑神经特征中的主导地位，为可穿戴低电极脑-机接口（BCI）实时监测提供了方向。这些发现为开发能够动态检测和回应用户困惑的自适应系统奠定了基础，潜在应用包括个性化学习、人机交互和无障碍技术。 

---
# Cognitive Surgery: The Awakening of Implicit Territorial Awareness in LLMs 

**Title (ZH)**: 认知手术：LLMs中隐含领土意识的觉醒 

**Authors**: Yinghan Zhou, Weifeng Zhu, Juan Wen, Wanli Peng, Zhengxian Wu, Yiming Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.14408)  

**Abstract**: Large language models (LLMs) have been shown to possess a degree of self-recognition capability-the ability to identify whether a given text was generated by themselves. Prior work has demonstrated that this capability is reliably expressed under the Pair Presentation Paradigm (PPP), where the model is presented with two texts and asked to choose which one it authored. However, performance deteriorates sharply under the Individual Presentation Paradigm (IPP), where the model is given a single text to judge authorship. Although this phenomenon has been observed, its underlying causes have not been systematically analyzed. In this paper, we first replicate existing findings to confirm that LLMs struggle to distinguish self- from other-generated text under IPP. We then investigate the reasons for this failure and attribute it to a phenomenon we term Implicit Territorial Awareness (ITA)-the model's latent ability to distinguish self- and other-texts in representational space, which remains unexpressed in its output behavior. To awaken the ITA of LLMs, we propose Cognitive Surgery (CoSur), a novel framework comprising four main modules: representation extraction, territory construction, authorship discrimination and cognitive editing. Experimental results demonstrate that our proposed method improves the performance of three different LLMs in the IPP scenario, achieving average accuracies of 83.25%, 66.19%, and 88.01%, respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）具有一定程度的自我识别能力——识别给定文本是否由自己生成的能力。prior工作表明，这一能力在Pair Presentation Paradigm（PPP）下可靠地表达出来，即模型在面对两个文本并选择哪个是自己生成时表现良好。然而，在Individual Presentation Paradigm（IPP）下，模型仅需判断单个文本的作者身份时，其表现显著下降。尽管已经观察到这一现象，但其背后的原因尚未系统分析。在本文中，我们首先重现现有发现，验证LLMs在IPP下难以区分自我生成与其他生成的文本。然后，我们探讨这种失败的原因，并将其归因于我们称为隐含领地意识（ITA）的现象——模型在表示空间中区分自我与其他文本的潜在能力，但在其输出行为中未被表达。为了唤醒LLMs的ITA，我们提出了一种名为Cognitive Surgery（CoSur）的新型框架，包括四个主要模块：表示提取、领地构建、作者识别和认知编辑。实验结果表明，我们的方法在IPP场景下提高了三种不同LLMs的表现，分别达到83.25%、66.19%和88.01%的平均准确率。 

---
# NoteIt: A System Converting Instructional Videos to Interactable Notes Through Multimodal Video Understanding 

**Title (ZH)**: NoteIt: 一种通过多模态视频理解将 instructional videos 转换为可交互笔记的系统 

**Authors**: Running Zhao, Zhihan Jiang, Xinchen Zhang, Chirui Chang, Handi Chen, Weipeng Deng, Luyao Jin, Xiaojuan Qi, Xun Qian, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2508.14395)  

**Abstract**: Users often take notes for instructional videos to access key knowledge later without revisiting long videos. Automated note generation tools enable users to obtain informative notes efficiently. However, notes generated by existing research or off-the-shelf tools fail to preserve the information conveyed in the original videos comprehensively, nor can they satisfy users' expectations for diverse presentation formats and interactive features when using notes digitally. In this work, we present NoteIt, a system, which automatically converts instructional videos to interactable notes using a novel pipeline that faithfully extracts hierarchical structure and multimodal key information from videos. With NoteIt's interface, users can interact with the system to further customize the content and presentation formats of the notes according to their preferences. We conducted both a technical evaluation and a comparison user study (N=36). The solid performance in objective metrics and the positive user feedback demonstrated the effectiveness of the pipeline and the overall usability of NoteIt. Project website: this https URL 

**Abstract (ZH)**: 用户常常为 instructional videos 做笔记以在后续访问关键知识时不再观看长视频。自动笔记生成工具使用户能够高效地获取有信息量的笔记。然而，现有研究或即用型工具生成的笔记未能全面保留原始视频中的信息，也无法满足用户在数字环境下对多样化呈现格式和交互功能的需求。在本工作中，我们提出了 NoteIt 系统，该系统使用一种新颖的工作流程，自动将 instructional videos 转换为可交互的笔记，并忠实提取视频中的层次结构和多元模态关键信息。通过 NoteIt 的界面，用户可以根据自己的偏好进一步自定义笔记的内容和呈现格式。我们进行了技术评估和比较用户研究（N=36）。客观指标的良好表现和积极的用户反馈证明了该工作流程的有效性和 NoteIt 的整体易用性。项目网址：this https URL。 

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
# Online Incident Response Planning under Model Misspecification through Bayesian Learning and Belief Quantization 

**Title (ZH)**: 基于贝叶斯学习和信念量化在模型错指情况下的在线事件响应规划 

**Authors**: Kim Hammar, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14385)  

**Abstract**: Effective responses to cyberattacks require fast decisions, even when information about the attack is incomplete or inaccurate. However, most decision-support frameworks for incident response rely on a detailed system model that describes the incident, which restricts their practical utility. In this paper, we address this limitation and present an online method for incident response planning under model misspecification, which we call MOBAL: Misspecified Online Bayesian Learning. MOBAL iteratively refines a conjecture about the model through Bayesian learning as new information becomes available, which facilitates model adaptation as the incident unfolds. To determine effective responses online, we quantize the conjectured model into a finite Markov model, which enables efficient response planning through dynamic programming. We prove that Bayesian learning is asymptotically consistent with respect to the information feedback. Additionally, we establish bounds on misspecification and quantization errors. Experiments on the CAGE-2 benchmark show that MOBAL outperforms the state of the art in terms of adaptability and robustness to model misspecification. 

**Abstract (ZH)**: 基于模型错构的在线贝叶斯学习响应规划 

---
# ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities 

**Title (ZH)**: ZPD-SCA: 揭示LLMs在评估学生认知能力方面存在的盲点 

**Authors**: Wenhan Dong, Zhen Sun, Yuemeng Zhao, Zifan Peng, Jun Wu, Jingyi Zheng, Yule Liu, Xinlei He, Yu Wang, Ruiming Wang, Xinyi Huang, Lei Mo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14377)  

**Abstract**: Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在教育应用中展现了潜力，但在准确评估阅读材料与学生发育阶段的认知一致性方面的能力尚待充分探索。这一缺口尤其关键，因为临近发展区（ZPD）的基本教育原则强调学习资源应与学生的认知能力（SCA）相匹配。尽管这一匹配的重要性不言而喻，但在不同学生年龄组阅读理解难度评估方面，针对大规模语言模型（LLMs）进行的全面研究依然匮乏，特别是在汉语语言教育的背景下。为了填补这一缺口，我们提出了ZPD-SCA这一新型基准，旨在评估阶段性的汉语阅读理解难度。该基准由60名特级教师进行标注，这代表了全国在职教师的前0.15%。实验结果显示，大规模语言模型在零样本学习场景中表现不佳，甚至低于随机猜测的概率。当提供上下文示例时，模型的性能显著提升，某些模型的准确率几乎提高了其零样本基线的一倍。这些结果揭示了大规模语言模型评估阅读难度的初步能力，同时也暴露了其当前训练在教育相关判断方面存在的局限性。值得注意的是，即使是表现最佳的模型也显示出系统的方向性偏差，这表明在准确匹配材料难度与认知能力方面存在困难。此外，不同体裁之间模型性能的显著差异凸显了该任务的复杂性。我们设想，ZPD-SCA能够为评估和改善大型语言模型在认知一致的教育应用中的表现提供基础。 

---
# Computing-In-Memory Dataflow for Minimal Buffer Traffic 

**Title (ZH)**: 计算在内存的数据流传输技术Minimal Buffer Traffic 

**Authors**: Choongseok Song, Doo Seok Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2508.14375)  

**Abstract**: Computing-In-Memory (CIM) offers a potential solution to the memory wall issue and can achieve high energy efficiency by minimizing data movement, making it a promising architecture for edge AI devices. Lightweight models like MobileNet and EfficientNet, which utilize depthwise convolution for feature extraction, have been developed for these devices. However, CIM macros often face challenges in accelerating depthwise convolution, including underutilization of CIM memory and heavy buffer traffic. The latter, in particular, has been overlooked despite its significant impact on latency and energy consumption. To address this, we introduce a novel CIM dataflow that significantly reduces buffer traffic by maximizing data reuse and improving memory utilization during depthwise convolution. The proposed dataflow is grounded in solid theoretical principles, fully demonstrated in this paper. When applied to MobileNet and EfficientNet models, our dataflow reduces buffer traffic by 77.4-87.0%, leading to a total reduction in data traffic energy and latency by 10.1-17.9% and 15.6-27.8%, respectively, compared to the baseline (conventional weight-stationary dataflow). 

**Abstract (ZH)**: 计算在内存中（CIM）的方法为解决内存墙问题提供了潜在的解决方案，并可通过最小化数据移动实现高效能耗，使其成为边缘AI设备的有前途的架构。针对这些设备开发的轻量级模型，如MobileNet和EfficientNet，使用深度卷积进行特征提取。然而，CIM宏在加速深度卷积时常常面临挑战，包括CIM内存的利用不足和严重的缓存流量。后者虽然对延迟和能耗有重大影响，但常常被忽视。为解决这一问题，我们提出了一种新型CIM数据流，该数据流通过最大化数据重用和提高深度卷积期间的内存利用率，显著减少了缓存流量。本研究基于坚实的理论基础，全面展示了该数据流的应用效果。将该数据流应用于MobileNet和EfficientNet模型时，缓存流量减少了77.4%-87.0%，相比于基线（传统的权重静止数据流），总的数据传输能耗和延迟分别减少了10.1%-17.9%和15.6%-27.8%。 

---
# Learning Point Cloud Representations with Pose Continuity for Depth-Based Category-Level 6D Object Pose Estimation 

**Title (ZH)**: 基于姿态连续性的点云表示学习用于深度导向的类别级6D物体姿态估计 

**Authors**: Zhujun Li, Shuo Zhang, Ioannis Stamos  

**Link**: [PDF](https://arxiv.org/pdf/2508.14358)  

**Abstract**: Category-level object pose estimation aims to predict the 6D pose and 3D size of objects within given categories. Existing approaches for this task rely solely on 6D poses as supervisory signals without explicitly capturing the intrinsic continuity of poses, leading to inconsistencies in predictions and reduced generalization to unseen poses. To address this limitation, we propose HRC-Pose, a novel depth-only framework for category-level object pose estimation, which leverages contrastive learning to learn point cloud representations that preserve the continuity of 6D poses. HRC-Pose decouples object pose into rotation and translation components, which are separately encoded and leveraged throughout the network. Specifically, we introduce a contrastive learning strategy for multi-task, multi-category scenarios based on our 6D pose-aware hierarchical ranking scheme, which contrasts point clouds from multiple categories by considering rotational and translational differences as well as categorical information. We further design pose estimation modules that separately process the learned rotation-aware and translation-aware embeddings. Our experiments demonstrate that HRC-Pose successfully learns continuous feature spaces. Results on REAL275 and CAMERA25 benchmarks show that our method consistently outperforms existing depth-only state-of-the-art methods and runs in real-time, demonstrating its effectiveness and potential for real-world applications. Our code is at this https URL. 

**Abstract (ZH)**: 类别级别物体姿态估计旨在预测给定类别内物体的6D姿态和3D尺寸。现有的方法仅依赖于6D姿态作为监督信号，而未明确捕捉姿态的内在连续性，导致预测不一致且对未见过的姿态泛化能力降低。为此，我们提出了一种新颖的仅深度框架HRC-Pose，该框架利用对比学习学习保留6D姿态连续性的点云表示。HRC-Pose将物体姿态分解为旋转和平移分量，并分别进行编码和利用。具体而言，我们基于六维姿态感知层次排名方案引入一种多任务、多类别场景下的对比学习策略，通过考虑旋转和平移差异以及类别信息对来自多个类别的点云进行对比。我们还设计了分别处理所学习的旋转感知和位移感知嵌入的姿态估计模块。我们的实验表明HRC-Pose成功学习了连续的特征空间。在REAL275和CAMERA25基准上的结果表明，我们的方法一致地优于现有的仅深度最先进的方法，并且能够实时运行，证明了其有效性和在实际应用中的潜力。代码参见此链接：https://github.com/alexisliujh/HRC-Pose。 

---
# Organ-Agents: Virtual Human Physiology Simulator via LLMs 

**Title (ZH)**: 器官代理：通过大语言模型的虚拟人体生理模拟器 

**Authors**: Rihao Chang, He Jiao, Weizhi Nie, Honglin Guo, Keliang Xie, Zhenhua Wu, Lina Zhao, Yunpeng Bai, Yongtao Ma, Lanjun Wang, Yuting Su, Xi Gao, Weijie Wang, Nicu Sebe, Bruno Lepri, Bingwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14357)  

**Abstract**: Recent advances in large language models (LLMs) have enabled new possibilities in simulating complex physiological systems. We introduce Organ-Agents, a multi-agent framework that simulates human physiology via LLM-driven agents. Each Simulator models a specific system (e.g., cardiovascular, renal, immune). Training consists of supervised fine-tuning on system-specific time-series data, followed by reinforcement-guided coordination using dynamic reference selection and error correction. We curated data from 7,134 sepsis patients and 7,895 controls, generating high-resolution trajectories across 9 systems and 125 variables. Organ-Agents achieved high simulation accuracy on 4,509 held-out patients, with per-system MSEs <0.16 and robustness across SOFA-based severity strata. External validation on 22,689 ICU patients from two hospitals showed moderate degradation under distribution shifts with stable simulation. Organ-Agents faithfully reproduces critical multi-system events (e.g., hypotension, hyperlactatemia, hypoxemia) with coherent timing and phase progression. Evaluation by 15 critical care physicians confirmed realism and physiological plausibility (mean Likert ratings 3.9 and 3.7). Organ-Agents also enables counterfactual simulations under alternative sepsis treatment strategies, generating trajectories and APACHE II scores aligned with matched real-world patients. In downstream early warning tasks, classifiers trained on synthetic data showed minimal AUROC drops (<0.04), indicating preserved decision-relevant patterns. These results position Organ-Agents as a credible, interpretable, and generalizable digital twin for precision diagnosis, treatment simulation, and hypothesis testing in critical care. 

**Abstract (ZH)**: Recent Advances in大型语言模型（LLMs）近期在大型语言模型（LLMs）方面的进展开启了模拟复杂生理系统的新可能性。我们引入了Organ-Agents，一种通过LLM驱动的代理模拟人类生理的多代理框架。每个Simulator模拟特定的系统（例如，心血管系统、肾系统、免疫系统）。训练过程包括基于特定时间序列数据的监督调优，随后是基于强化学习的协调，使用动态参考选择和错误校正。我们从7,134例脓毒症患者和7,895例对照者中收集了数据，生成了跨越9个系统和125个变量的高分辨率轨迹。Organ-Agents在4,509例留出患者中实现了高水平的模拟准确性，每个系统的均方误差<0.16，并且在SOFA基严重程度分层中表现出鲁棒性。两个医院的22,689例ICU患者的外部验证显示，在分布转移下有适度退化但模拟稳定。Organ-Agents真实地再现了关键的多系统事件（如低血压、高乳酸血症、低氧血症），具有连贯的时间进程和相位进展。由15名重症医学专家评估确认了其现实性和生理学合理性（平均Likert评分3.9和3.7）。Organ-Agents还能够模拟在不同脓毒症治疗策略下的事实替代场景，生成与匹配的现实世界患者一致的轨迹和APACHE II评分。在下游早期预警任务中，基于合成数据训练的分类器的AUROC下降幅度最小（<0.04），表明保留了决策相关的模式。这些结果将Organ-Agents定位为精准诊断、治疗模拟和重症医学中假设测试的一种可信、可解释和可泛化的数字孪生。 

---
# Inter-Class Relational Loss for Small Object Detection: A Case Study on License Plates 

**Title (ZH)**: 类内关系损失在小目标检测中的应用：车牌检测案例研究 

**Authors**: Dian Ning, Dong Seog Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.14343)  

**Abstract**: In one-stage multi-object detection tasks, various intersection over union (IoU)-based solutions aim at smooth and stable convergence near the targets during training. However, IoU-based losses fail to correctly update the gradient of small objects due to an extremely flat gradient. During the update of multiple objects, the learning of small objects' gradients suffers more because of insufficient gradient updates. Therefore, we propose an inter-class relational loss to efficiently update the gradient of small objects while not sacrificing the learning efficiency of other objects based on the simple fact that an object has a spatial relationship to another object (e.g., a car plate is attached to a car in a similar position). When the predicted car plate's bounding box is not within its car, a loss punishment is added to guide the learning, which is inversely proportional to the overlapped area of the car's and predicted car plate's bounding box. By leveraging the spatial relationship at the inter-class level, the loss guides small object predictions using larger objects and enhances latent information in deeper feature maps. In this paper, we present twofold contributions using license plate detection as a case study: (1) a new small vehicle multi-license plate dataset (SVMLP), featuring diverse real-world scenarios with high-quality annotations; and (2) a novel inter-class relational loss function designed to promote effective detection performance. We highlight the proposed ICR loss penalty can be easily added to existing IoU-based losses and enhance the performance. These contributions improve the standard mean Average Precision (mAP) metric, achieving gains of 10.3% and 1.6% in mAP$^{\text{test}}_{50}$ for YOLOv12-T and UAV-DETR, respectively, without any additional hyperparameter tuning. Code and dataset will be available soon. 

**Abstract (ZH)**: 在一阶段多对象检测任务中，各种基于交并比(IoU)的解决方案旨在在训练过程中使模型平滑稳定地收敛至目标。然而，基于IoU的损失函数由于梯度极其平坦而无法正确更新小对象的梯度。在更新多个对象时，小对象梯度的学习因梯度更新不足而受到更大影响。因此，我们提出了一种跨类别关系损失，旨在在不牺牲其他对象学习效率的前提下高效地更新小对象的梯度，基于这样一个简单事实：一个对象与另一个对象在空间上存在关联（例如，车牌通常位于汽车的相似位置）。当预测的车牌边界框不在汽车边界框内时，会增加损失惩罚以引导学习，该惩罚与汽车边界框和预测车牌边界框重叠区域的大小成反比。通过在跨类别层面利用空间关系，损失可以利用更大对象来引导小对象预测并增强深层特征图中的隐含信息。在本文中，我们以车牌检测为例介绍了两项贡献：（1）一个新的人车多车牌数据集（SVMLP），该数据集包含多种真实世界的场景和高质量标注；（2）一种新颖的跨类别关系损失函数，旨在促进有效的检测性能。我们强调，提出的ICR损失惩罚可以轻松集成到现有的基于IoU的损失函数中并提高性能。这些贡献在不进行任何额外超参数调整的情况下，分别提高了YOLOv12-T和UAV-DETR的标准平均精度（mAP）指标10.3%和1.6%。代码和数据集将在不久的将来公开。 

---
# Generative AI Against Poaching: Latent Composite Flow Matching for Wildlife Conservation 

**Title (ZH)**: 生成式AI对抗偷猎：潜在复合流匹配在野生动物保护中的应用 

**Authors**: Lingkai Kong, Haichuan Wang, Charles A. Emogor, Vincent Börsch-Supan, Lily Xu, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2508.14342)  

**Abstract**: Poaching poses significant threats to wildlife and biodiversity. A valuable step in reducing poaching is to forecast poacher behavior, which can inform patrol planning and other conservation interventions. Existing poaching prediction methods based on linear models or decision trees lack the expressivity to capture complex, nonlinear spatiotemporal patterns. Recent advances in generative modeling, particularly flow matching, offer a more flexible alternative. However, training such models on real-world poaching data faces two central obstacles: imperfect detection of poaching events and limited data. To address imperfect detection, we integrate flow matching with an occupancy-based detection model and train the flow in latent space to infer the underlying occupancy state. To mitigate data scarcity, we adopt a composite flow initialized from a linear-model prediction rather than random noise which is the standard in diffusion models, injecting prior knowledge and improving generalization. Evaluations on datasets from two national parks in Uganda show consistent gains in predictive accuracy. 

**Abstract (ZH)**: 基于流匹配的占用模型在野生动物偷猎预测中的应用：克服检测不完美和数据稀缺难题 

---
# A Comparative Evaluation of Teacher-Guided Reinforcement Learning Techniques for Autonomous Cyber Operations 

**Title (ZH)**: 教师引导的强化学习技术在自主网络运营中的比较评估 

**Authors**: Konur Tholl, Mariam El Mezouar, Ranwa Al Mallah  

**Link**: [PDF](https://arxiv.org/pdf/2508.14340)  

**Abstract**: Autonomous Cyber Operations (ACO) rely on Reinforcement Learning (RL) to train agents to make effective decisions in the cybersecurity domain. However, existing ACO applications require agents to learn from scratch, leading to slow convergence and poor early-stage performance. While teacher-guided techniques have demonstrated promise in other domains, they have not yet been applied to ACO. In this study, we implement four distinct teacher-guided techniques in the simulated CybORG environment and conduct a comparative evaluation. Our results demonstrate that teacher integration can significantly improve training efficiency in terms of early policy performance and convergence speed, highlighting its potential benefits for autonomous cybersecurity. 

**Abstract (ZH)**: 自主网络操作（ACO）依赖强化学习（RL）训练代理在网络安全领域做出有效的决策。然而，现有的ACO应用需要代理从头学习，导致收敛速度慢且早期性能差。尽管教师引导技术在其他领域显示出潜力，但尚未应用于ACO。在本研究中，我们在模拟的CybORG环境中实施了四种不同的教师引导技术，并进行了比较评估。我们的结果表明，教师集成可以显著提高早期策略性能和收敛速度的训练效率，突显了其在自主网络安全方面的潜在益处。 

---
# Power Stabilization for AI Training Datacenters 

**Title (ZH)**: AI训练数据中心的功率稳定化 

**Authors**: Esha Choukse, Brijesh Warrier, Scot Heath, Luz Belmont, April Zhao, Hassan Ali Khan, Brian Harry, Matthew Kappel, Russell J. Hewett, Kushal Datta, Yu Pei, Caroline Lichtenberger, John Siegler, David Lukofsky, Zaid Kahn, Gurpreet Sahota, Andy Sullivan, Charles Frederick, Hien Thai, Rebecca Naughton, Daniel Jurnove, Justin Harp, Reid Carper, Nithish Mahalingam, Srini Varkala, Alok Gautam Kumbhare, Satyajit Desai, Venkatesh Ramamurthy, Praneeth Gottumukkala, Girish Bhatia, Kelsey Wildstone, Laurentiu Olariu, Mohammed Ayna, Mike Kendrick, Ricardo Bianchini, Aaron Hurst, Reza Zamani, Xin Li, Gene Oden, Rory Carmichael, Tom Li, Apoorv Gupta, Nilesh Dattani, Lawrence Marwong, Rob Nertney, Jeff Liott, Miro Enev, Divya Ramakrishnan, Ian Buck, Jonah Alben  

**Link**: [PDF](https://arxiv.org/pdf/2508.14318)  

**Abstract**: Large Artificial Intelligence (AI) training workloads spanning several tens of thousands of GPUs present unique power management challenges. These arise due to the high variability in power consumption during the training. Given the synchronous nature of these jobs, during every iteration there is a computation-heavy phase, where each GPU works on the local data, and a communication-heavy phase where all the GPUs synchronize on the data. Because compute-heavy phases require much more power than communication phases, large power swings occur. The amplitude of these power swings is ever increasing with the increase in the size of training jobs. An even bigger challenge arises from the frequency spectrum of these power swings which, if harmonized with critical frequencies of utilities, can cause physical damage to the power grid infrastructure. Therefore, to continue scaling AI training workloads safely, we need to stabilize the power of such workloads. This paper introduces the challenge with production data and explores innovative solutions across the stack: software, GPU hardware, and datacenter infrastructure. We present the pros and cons of each of these approaches and finally present a multi-pronged approach to solving the challenge. The proposed solutions are rigorously tested using a combination of real hardware and Microsoft's in-house cloud power simulator, providing critical insights into the efficacy of these interventions under real-world conditions. 

**Abstract (ZH)**: 大规模人工智能（AI）训练工作负载跨越数千个GPU，带来独特的电源管理挑战。这些挑战源于训练过程中电力消耗的高变异性。由于这些任务的同步性质，在每一轮迭代中，都有一个计算密集型阶段，每个GPU处理本地数据，以及一个通信密集型阶段，所有GPU同步数据。因为计算密集型阶段需要更多的电力，而通信密集型阶段需要较少的电力，所以电力波动幅度较大。随着训练任务规模的增加，这种电力波动的幅度也在增加。更大的挑战来自于这些电力波动的频率谱，如果与电力设施的关键频率谐振，可能会对电力网络基础设施造成物理损害。因此，为了安全地扩大AI训练工作负载的规模，我们需要稳定此类工作负载的电力。本文使用生产数据介绍该挑战，并在软件、GPU硬件和数据中心基础设施等多个层面探索创新解决方案。我们分析了每种方法的优缺点，并最终提出一种多管齐下的解决方案。提出的解决方案通过结合实际硬件和微软内部云电源模拟器进行了严格测试，为这些干预措施在实际条件下的有效性提供了关键见解。 

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
# Learning Time-Varying Convexifications of Multiple Fairness Measures 

**Title (ZH)**: 学习多公平性指标的时间依赖凸化 

**Authors**: Quan Zhou, Jakub Marecek, Robert Shorten  

**Link**: [PDF](https://arxiv.org/pdf/2508.14311)  

**Abstract**: There is an increasing appreciation that one may need to consider multiple measures of fairness, e.g., considering multiple group and individual fairness notions. The relative weights of the fairness regularisers are a priori unknown, may be time varying, and need to be learned on the fly. We consider the learning of time-varying convexifications of multiple fairness measures with limited graph-structured feedback. 

**Abstract (ZH)**: 考虑多种公平性标准的动态凸包学习及其有限图结构反馈 

---
# GLASS: Test-Time Acceleration for LLMs via Global-Local Neural Importance Aggregation 

**Title (ZH)**: GLASS: 通过全局-局部神经重要性聚合实现大语言模型测试时加速 

**Authors**: Amirmohsen Sattarifard, Sepehr Lavasani, Ehsan Imani, Kunlin Zhang, Hanlin Xu, Fengyu Sun, Negar Hassanpour, Chao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14302)  

**Abstract**: Deploying Large Language Models (LLMs) on edge hardware demands aggressive, prompt-aware dynamic pruning to reduce computation without degrading quality. Static or predictor-based schemes either lock in a single sparsity pattern or incur extra runtime overhead, and recent zero-shot methods that rely on statistics from a single prompt fail on short prompt and/or long generation scenarios. We introduce A/I-GLASS: Activation- and Impact-based Global-Local neural importance Aggregation for feed-forward network SparSification, two training-free methods that dynamically select FFN units using a rank-aggregation of prompt local and model-intrinsic global neuron statistics. Empirical results across multiple LLMs and benchmarks demonstrate that GLASS significantly outperforms prior training-free methods, particularly in challenging long-form generation scenarios, without relying on auxiliary predictors or adding any inference overhead. 

**Abstract (ZH)**: 基于激活和影响的全局-局部神经重要性聚合的前向网络稀疏化：无训练动态选择方法 

---
# Pixels to Play: A Foundation Model for 3D Gameplay 

**Title (ZH)**: 从像素到游玩：三维游戏的基础模型 

**Authors**: Yuguang Yue, Chris Green, Samuel Hunt, Irakli Salia, Wenzhe Shi, Jonathan J Hunt  

**Link**: [PDF](https://arxiv.org/pdf/2508.14295)  

**Abstract**: We introduce Pixels2Play-0.1 (P2P0.1), a foundation model that learns to play a wide range of 3D video games with recognizable human-like behavior. Motivated by emerging consumer and developer use cases - AI teammates, controllable NPCs, personalized live-streamers, assistive testers - we argue that an agent must rely on the same pixel stream available to players and generalize to new titles with minimal game-specific engineering. P2P0.1 is trained end-to-end with behavior cloning: labeled demonstrations collected from instrumented human game-play are complemented by unlabeled public videos, to which we impute actions via an inverse-dynamics model. A decoder-only transformer with auto-regressive action output handles the large action space while remaining latency-friendly on a single consumer GPU. We report qualitative results showing competent play across simple Roblox and classic MS-DOS titles, ablations on unlabeled data, and outline the scaling and evaluation steps required to reach expert-level, text-conditioned control. 

**Abstract (ZH)**: Pixels2Play-0.1：一种学习广泛范围3D视频游戏并表现出可识别的人类行为的基础模型 

---
# OccluNet: Spatio-Temporal Deep Learning for Occlusion Detection on DSA 

**Title (ZH)**: OccluNet：基于DSA的时空深度学习遮挡检测 

**Authors**: Anushka A. Kore, Frank G. te Nijenhuis, Matthijs van der Sluijs, Wim van Zwam, Charles Majoie, Geert Lycklama à Nijeholt, Danny Ruijters, Frans Vos, Sandra Cornelissen, Ruisheng Su, Theo van Walsum  

**Link**: [PDF](https://arxiv.org/pdf/2508.14286)  

**Abstract**: Accurate detection of vascular occlusions during endovascular thrombectomy (EVT) is critical in acute ischemic stroke (AIS). Interpretation of digital subtraction angiography (DSA) sequences poses challenges due to anatomical complexity and time constraints. This work proposes OccluNet, a spatio-temporal deep learning model that integrates YOLOX, a single-stage object detector, with transformer-based temporal attention mechanisms to automate occlusion detection in DSA sequences. We compared OccluNet with a YOLOv11 baseline trained on either individual DSA frames or minimum intensity projections. Two spatio-temporal variants were explored for OccluNet: pure temporal attention and divided space-time attention. Evaluation on DSA images from the MR CLEAN Registry revealed the model's capability to capture temporally consistent features, achieving precision and recall of 89.02% and 74.87%, respectively. OccluNet significantly outperformed the baseline models, and both attention variants attained similar performance. Source code is available at this https URL 

**Abstract (ZH)**: 准确检测内血管闭塞对于急性缺血性卒中的内血管溶栓(EVT)至关重要。由于解剖复杂性和时间限制，数字减影血管造影(DSA)序列的解释具有挑战性。本文提出OccluNet，这是一种时空深度学习模型，结合了单一阶段物体检测器YOLOX和基于变压器的时序注意机制，以自动在DSA序列中检测闭塞。我们将OccluNet与YOLOv1基准模型进行了比较，该基准模型是基于个体DSA帧或最小强度投影进行训练的。OccluNet探索了两种时空变体：纯时序注意和空间-时序分割注意。在MR CLEAN登记册的DSA图像上的评估显示，该模型能够捕捉到一致的时空特征，精确度和召回率分别为89.02%和74.87%。OccluNet显著优于基线模型，两种注意机制的性能相似。源代码可在以下网址获取。 

---
# Amortized Bayesian Meta-Learning for Low-Rank Adaptation of Large Language Models 

**Title (ZH)**: 低秩适应的大语言模型的拟似然贝叶斯元学习 

**Authors**: Liyi Zhang, Jake Snell, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2508.14285)  

**Abstract**: Fine-tuning large language models (LLMs) with low-rank adaptaion (LoRA) is a cost-effective way to incorporate information from a specific dataset. However, it is often unclear how well the fine-tuned LLM will generalize, i.e., how well it will perform on unseen datasets. Methods have been proposed to improve generalization by optimizing with in-context prompts, or by using meta-learning to fine-tune LLMs. However, these methods are expensive in memory and computation, requiring either long-context prompts or saving copies of parameters and using second-order gradient updates. To address these challenges, we propose Amortized Bayesian Meta-Learning for LoRA (ABMLL). This method builds on amortized Bayesian meta-learning for smaller models, adapting this approach to LLMs while maintaining its computational efficiency. We reframe task-specific and global parameters in the context of LoRA and use a set of new hyperparameters to balance reconstruction accuracy and the fidelity of task-specific parameters to the global ones. ABMLL provides effective generalization and scales to large models such as Llama3-8B. Furthermore, as a result of using a Bayesian framework, ABMLL provides improved uncertainty quantification. We test ABMLL on Unified-QA and CrossFit datasets and find that it outperforms existing methods on these benchmarks in terms of both accuracy and expected calibration error. 

**Abstract (ZH)**: 使用Amortized Bayesian Meta-Learning for LoRA实现大型语言模型的有效泛化 

---
# Tooth-Diffusion: Guided 3D CBCT Synthesis with Fine-Grained Tooth Conditioning 

**Title (ZH)**: 牙指导的细粒度牙科3D CBCT合成 

**Authors**: Said Djafar Said, Torkan Gholamalizadeh, Mostafa Mehdipour Ghazi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14276)  

**Abstract**: Despite the growing importance of dental CBCT scans for diagnosis and treatment planning, generating anatomically realistic scans with fine-grained control remains a challenge in medical image synthesis. In this work, we propose a novel conditional diffusion framework for 3D dental volume generation, guided by tooth-level binary attributes that allow precise control over tooth presence and configuration. Our approach integrates wavelet-based denoising diffusion, FiLM conditioning, and masked loss functions to focus learning on relevant anatomical structures. We evaluate the model across diverse tasks, such as tooth addition, removal, and full dentition synthesis, using both paired and distributional similarity metrics. Results show strong fidelity and generalization with low FID scores, robust inpainting performance, and SSIM values above 0.91 even on unseen scans. By enabling realistic, localized modification of dentition without rescanning, this work opens opportunities for surgical planning, patient communication, and targeted data augmentation in dental AI workflows. The codes are available at: this https URL. 

**Abstract (ZH)**: 尽管牙科CBCT扫描在诊断和治疗规划中的重要性日益增加，但在医学图像合成中生成具有精细控制的解剖学上逼真的扫描仍然具有挑战性。在这项工作中，我们提出了一种新的基于条件扩散的三维牙科体素生成框架，该框架由牙齿级二元属性引导，允许对牙齿的存在和配置进行精确控制。我们的方法结合了小波去噪扩散、FiLM条件和掩码损失函数，以专注于相关解剖结构的学习。我们在牙齿添加、移除以及全口牙齿合成等多种任务上评估了该模型，使用配对和分布相似性度量进行评估。结果显示高保真度和泛化能力，低FID得分，稳健的 inpainting 表现，以及在未见扫描上 SSIM 值高于 0.91。通过实现无需重新扫描即可对牙齿进行逼真的局部修改，本工作为手术规划、患者沟通和牙科AI工作流程中的针对性数据增强提供了机会。代码可在以下网址获取：this https URL。 

---
# Disentangling concept semantics via multilingual averaging in Sparse Autoencoders 

**Title (ZH)**: 通过稀疏自编码器中的多语言平均分离概念语义 

**Authors**: Cliff O'Reilly, Ernesto Jimenez-Ruiz, Tillman Weyde  

**Link**: [PDF](https://arxiv.org/pdf/2508.14275)  

**Abstract**: Connecting LLMs with formal knowledge representation and reasoning is a promising approach to address their shortcomings. Embeddings and sparse autoencoders are widely used to represent textual content, but the semantics are entangled with syntactic and language-specific information. We propose a method that isolates concept semantics in Large Langue Models by averaging concept activations derived via Sparse Autoencoders. We create English text representations from OWL ontology classes, translate the English into French and Chinese and then pass these texts as prompts to the Gemma 2B LLM. Using the open source Gemma Scope suite of Sparse Autoencoders, we obtain concept activations for each class and language version. We average the different language activations to derive a conceptual average. We then correlate the conceptual averages with a ground truth mapping between ontology classes. Our results give a strong indication that the conceptual average aligns to the true relationship between classes when compared with a single language by itself. The result hints at a new technique which enables mechanistic interpretation of internal network states with higher accuracy. 

**Abstract (ZH)**: 将大型语言模型与形式化的知识表示和推理相连是一种有希望的方法，以解决其不足之处。通过稀疏自编码器提取的概念激活进行平均，以隔离大型语言模型中的概念语义。从OWL本体类创建英文文本表示，将其翻译成法语和中文，然后将这些文本作为提示传递给Gemma 2B大型语言模型。使用开源的Gemma Scope套件中的稀疏自编码器，我们获得了每种类别和语言版本的概念激活。我们对不同的语言激活进行平均，得出概念平均值。然后我们将概念平均值与本体类之间的 ground truth 映射进行相关分析。我们的结果强烈表明，当与单一语言相比时，概念平均值能够更好地对类别之间的真正关系进行对齐。这一结果暗示了一种新的技术，该技术能够以更高的准确性对内部网络状态进行机械性解释。 

---
# Effect of Data Augmentation on Conformal Prediction for Diabetic Retinopathy 

**Title (ZH)**: 数据增强对糖尿病视网膜病变置信预测的影响 

**Authors**: Rizwan Ahamed, Annahita Amireskandari, Joel Palko, Carol Laxson, Binod Bhattarai, Prashnna Gyawali  

**Link**: [PDF](https://arxiv.org/pdf/2508.14266)  

**Abstract**: The clinical deployment of deep learning models for high-stakes tasks such as diabetic retinopathy (DR) grading requires demonstrable reliability. While models achieve high accuracy, their clinical utility is limited by a lack of robust uncertainty quantification. Conformal prediction (CP) offers a distribution-free framework to generate prediction sets with statistical guarantees of coverage. However, the interaction between standard training practices like data augmentation and the validity of these guarantees is not well understood. In this study, we systematically investigate how different data augmentation strategies affect the performance of conformal predictors for DR grading. Using the DDR dataset, we evaluate two backbone architectures -- ResNet-50 and a Co-Scale Conv-Attentional Transformer (CoaT) -- trained under five augmentation regimes: no augmentation, standard geometric transforms, CLAHE, Mixup, and CutMix. We analyze the downstream effects on conformal metrics, including empirical coverage, average prediction set size, and correct efficiency. Our results demonstrate that sample-mixing strategies like Mixup and CutMix not only improve predictive accuracy but also yield more reliable and efficient uncertainty estimates. Conversely, methods like CLAHE can negatively impact model certainty. These findings highlight the need to co-design augmentation strategies with downstream uncertainty quantification in mind to build genuinely trustworthy AI systems for medical imaging. 

**Abstract (ZH)**: 高风险任务如糖尿病视网膜病变(DR)分级中基于深度学习的临床部署需要可验证的可靠性。本文系统探究不同数据增强策略对DR分级中校准预测器性能的影响。使用DDR数据集，评估两种骨干架构——ResNet-50和Co-Scale Conv-Attentional Transformer (CoaT)——在五种增强制度下的性能：无增强、标准几何变换、CLAHE、Mixup和CutMix。分析下游校准指标，包括经验覆盖率、平均预测集大小和正确效率。研究结果表明，样本混合法如Mixup和CutMix不仅能提高预测准确性，还能提供更可靠和高效的不确定性估计。相反，方法如CLAHE可能会负面影响模型确定性。这些发现强调了在构建真正可信赖的医疗影像AI系统时需综合设计增强策略和下游不确定性量化的重要性。 

---
# Incident Analysis for AI Agents 

**Title (ZH)**: AI代理的事件分析 

**Authors**: Carson Ezell, Xavier Roberts-Gaal, Alan Chan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14231)  

**Abstract**: As AI agents become more widely deployed, we are likely to see an increasing number of incidents: events involving AI agent use that directly or indirectly cause harm. For example, agents could be prompt-injected to exfiltrate private information or make unauthorized purchases. Structured information about such incidents (e.g., user prompts) can help us understand their causes and prevent future occurrences. However, existing incident reporting processes are not sufficient for understanding agent incidents. In particular, such processes are largely based on publicly available data, which excludes useful, but potentially sensitive, information such as an agent's chain of thought or browser history. To inform the development of new, emerging incident reporting processes, we propose an incident analysis framework for agents. Drawing on systems safety approaches, our framework proposes three types of factors that can cause incidents: system-related (e.g., CBRN training data), contextual (e.g., prompt injections), and cognitive (e.g., misunderstanding a user request). We also identify specific information that could help clarify which factors are relevant to a given incident: activity logs, system documentation and access, and information about the tools an agent uses. We provide recommendations for 1) what information incident reports should include and 2) what information developers and deployers should retain and make available to incident investigators upon request. As we transition to a world with more agents, understanding agent incidents will become increasingly crucial for managing risks. 

**Abstract (ZH)**: 随着AI代理的广泛应用，我们可能会看到越来越多的事件：涉及AI代理使用且直接或间接导致损害的事件。例如，代理可能会被注入提示以窃取私人信息或进行未经授权的购买。关于此类事件的结构化信息（如用户提示）可以帮助我们了解其原因并防止未来事件的发生。然而，现有的事件报告流程不足以理解代理事件。特别是，这些流程主要基于公开可用的数据，这排除了有用但可能敏感的信息，例如代理的思维链或浏览器历史。为了引导新的、新兴的事件报告流程的发展，我们提出了一种代理事件分析框架。借鉴系统安全方法，我们的框架提出了可能导致事件的三种因素类型：系统相关的（如CBRN训练数据）、情境相关的（如提示注入）和认知相关的（如误解用户请求）。我们还确定了特定的信息，这些信息有助于阐明哪些因素与特定事件相关：活动日志、系统文档和访问信息以及代理使用的工具信息。我们提供了关于1）事件报告应包含什么信息以及2）开发人员和部署者应保留并在要求时提供给事件调查人员的什么信息的建议。随着我们进入拥有更多代理的世界，理解代理事件对于管理风险将变得越来越重要。 

---
# New Insights into Automatic Treatment Planning for Cancer Radiotherapy Using Explainable Artificial Intelligence 

**Title (ZH)**: 使用可解释人工智能的新见解自动规划癌症放射治疗 

**Authors**: Md Mainul Abrar, Xun Jia, Yujie Chi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14229)  

**Abstract**: Objective: This study aims to uncover the opaque decision-making process of an artificial intelligence (AI) agent for automatic treatment planning.
Approach: We examined a previously developed AI agent based on the Actor-Critic with Experience Replay (ACER) network, which automatically tunes treatment planning parameters (TPPs) for inverse planning in prostate cancer intensity modulated radiotherapy. We selected multiple checkpoint ACER agents from different stages of training and applied an explainable AI (EXAI) method to analyze the attribution from dose-volume histogram (DVH) inputs to TPP-tuning decisions. We then assessed each agent's planning efficacy and efficiency and evaluated their policy and final TPP tuning spaces. Combining these analyses, we systematically examined how ACER agents generated high-quality treatment plans in response to different DVH inputs.
Results: Attribution analysis revealed that ACER agents progressively learned to identify dose-violation regions from DVH inputs and promote appropriate TPP-tuning actions to mitigate them. Organ-wise similarities between DVH attributions and dose-violation reductions ranged from 0.25 to 0.5 across tested agents. Agents with stronger attribution-violation similarity required fewer tuning steps (~12-13 vs. 22), exhibited a more concentrated TPP-tuning space with lower entropy (~0.3 vs. 0.6), converged on adjusting only a few TPPs, and showed smaller discrepancies between practical and theoretical tuning steps. Putting together, these findings indicate that high-performing ACER agents can effectively identify dose violations from DVH inputs and employ a global tuning strategy to achieve high-quality treatment planning, much like skilled human planners.
Significance: Better interpretability of the agent's decision-making process may enhance clinician trust and inspire new strategies for automatic treatment planning. 

**Abstract (ZH)**: 研究目标：本研究旨在揭示人工智能（AI）代理在自动治疗计划中的不透明决策过程。
方法：我们研究了一个基于Actor-Critic with Experience Replay（ACER）网络的先前开发的AI代理，该代理自动调整前列腺癌强度调制放射治疗逆向计划中的治疗计划参数（TPPs）。我们选择了不同训练阶段的多个检查点ACER代理，并应用可解释的人工智能（EXAI）方法，分析剂量体积直方图（DVH）输入到TPP调整决策的归因。然后，我们评估了每个代理的计划效果和效率，并评估了它们的策略和最终TPP调整空间。结合这些分析，我们系统地探讨了ACER代理如何根据不同DVH输入生成高质量的治疗计划。
结果：归因分析表明，ACER代理逐步学会从DVH输入中识别剂量违规区域，并促进适当的TPP调整行动以减轻这些违规。受试代理之间的器官内DVH归因与剂量违规减少之间的相似性范围为0.25至0.5。具有更强的归因-违规相似性的代理所需的调整步骤更少（约为12-13步 vs. 22步），具有更集中的TPP调整空间和更低的熵（约为0.3 vs. 0.6），只调整少数几个TPP，并且实际调整步骤与理论调整步骤的差异更小。综合来看，这些发现表明高性能的ACER代理可以有效地从DVH输入中识别出剂量违规，并采用全局调整策略以实现高质量的治疗计划，类似于熟练的人类计划者。
意义：改善代理决策过程的可解释性可能提升临床医生的信任，并激发新的自动治疗计划策略。 

---
# A Survey on Video Anomaly Detection via Deep Learning: Human, Vehicle, and Environment 

**Title (ZH)**: 深度学习视角下的视频异常检测综述：人类、车辆及环境 

**Authors**: Ghazal Alinezhad Noghre, Armin Danesh Pazho, Hamed Tabkhi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14203)  

**Abstract**: Video Anomaly Detection (VAD) has emerged as a pivotal task in computer vision, with broad relevance across multiple fields. Recent advances in deep learning have driven significant progress in this area, yet the field remains fragmented across domains and learning paradigms. This survey offers a comprehensive perspective on VAD, systematically organizing the literature across various supervision levels, as well as adaptive learning methods such as online, active, and continual learning. We examine the state of VAD across three major application categories: human-centric, vehicle-centric, and environment-centric scenarios, each with distinct challenges and design considerations. In doing so, we identify fundamental contributions and limitations of current methodologies. By consolidating insights from subfields, we aim to provide the community with a structured foundation for advancing both theoretical understanding and real-world applicability of VAD systems. This survey aims to support researchers by providing a useful reference, while also drawing attention to the broader set of open challenges in anomaly detection, including both fundamental research questions and practical obstacles to real-world deployment. 

**Abstract (ZH)**: 视频异常检测（VAD）已成为计算机视觉中的关键任务，广泛应用于多个领域。深度学习的 Recent 进展推动了该领域的显著进步，然而该领域仍然在不同领域和学习范式之间碎片化。本文综述为 VAD 提供了一个全面的视角，系统地整理了各种监督级别以及在线学习、主动学习和持续学习等自适应学习方法的文献。我们从三大主要应用类别——以人为核心的、以车辆为核心的和以环境为核心的场景——出发，审视了 VAD 的现状，每个场景都具有独特的挑战和设计考虑。通过识别当前方法论的基本贡献和局限性，我们旨在为社区提供一个结构化的基础，以促进对 VAD 系统的理论理解和实际应用。本文综述旨在为研究人员提供一个有用的参考，同时还将人们的注意力引向异常检测领域更广泛的一系列开放挑战，包括基础研究问题和实际部署中的障碍。 

---
# RynnEC: Bringing MLLMs into Embodied World 

**Title (ZH)**: RynnEC: 将MLLMs带入实体世界 

**Authors**: Ronghao Dang, Yuqian Yuan, Yunxuan Mao, Kehan Li, Jiangpin Liu, Zhikai Wang, Xin Li, Fan Wang, Deli Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14160)  

**Abstract**: We introduce RynnEC, a video multimodal large language model designed for embodied cognition. Built upon a general-purpose vision-language foundation model, RynnEC incorporates a region encoder and a mask decoder, enabling flexible region-level video interaction. Despite its compact architecture, RynnEC achieves state-of-the-art performance in object property understanding, object segmentation, and spatial reasoning. Conceptually, it offers a region-centric video paradigm for the brain of embodied agents, providing fine-grained perception of the physical world and enabling more precise interactions. To mitigate the scarcity of annotated 3D datasets, we propose an egocentric video based pipeline for generating embodied cognition data. Furthermore, we introduce RynnEC-Bench, a region-centered benchmark for evaluating embodied cognitive capabilities. We anticipate that RynnEC will advance the development of general-purpose cognitive cores for embodied agents and facilitate generalization across diverse embodied tasks. The code, model checkpoints, and benchmark are available at: this https URL 

**Abstract (ZH)**: 我们引入了RynnEC，一个为 embodiled 认知设计的视频多模态大语言模型。基于通用的视觉-语言基础模型，RynnEC 结合了区域编码器和掩码解码器，实现了灵活的区域级视频交互。尽管具有紧凑的架构，RynnEC 在对象属性理解、对象分割和空间推理方面均取得了最先进的性能。从概念上讲，它为 embodied 代理的大脑提供了一种以区域为中心的视频范式，提供了对物理世界的精细感知并促进了更精确的交互。为缓解注释的 3D 数据集稀缺性，我们提出了一种以第一人称视频为基础的数据生成流水线，用于生成 embodiled 认知数据。此外，我们介绍了 RynnEC-Bench，一种以区域为中心的基准，用于评估 embodiled 认知能力。我们预计 RynnEC 将推动通用认知核心的发展，并促进跨多种 embodied 任务的泛化。源代码、模型检查点和基准均可在以下网址获得：this https URL。 

---
# LENS: Learning to Segment Anything with Unified Reinforced Reasoning 

**Title (ZH)**: LENS: 学习分割一切——统一强化推理方法 

**Authors**: Lianghui Zhu, Bin Ouyang, Yuxuan Zhang, Tianheng Cheng, Rui Hu, Haocheng Shen, Longjin Ran, Xiaoxin Chen, Li Yu, Wenyu Liu, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14153)  

**Abstract**: Text-prompted image segmentation enables fine-grained visual understanding and is critical for applications such as human-computer interaction and robotics. However, existing supervised fine-tuning methods typically ignore explicit chain-of-thought (CoT) reasoning at test time, which limits their ability to generalize to unseen prompts and domains. To address this issue, we introduce LENS, a scalable reinforcement-learning framework that jointly optimizes the reasoning process and segmentation in an end-to-end manner. We propose unified reinforcement-learning rewards that span sentence-, box-, and segment-level cues, encouraging the model to generate informative CoT rationales while refining mask quality. Using a publicly available 3-billion-parameter vision-language model, i.e., Qwen2.5-VL-3B-Instruct, LENS achieves an average cIoU of 81.2% on the RefCOCO, RefCOCO+, and RefCOCOg benchmarks, outperforming the strong fine-tuned method, i.e., GLaMM, by up to 5.6%. These results demonstrate that RL-driven CoT reasoning serves as a robust prior for text-prompted segmentation and offers a practical path toward more generalizable Segment Anything models. Code is available at this https URL. 

**Abstract (ZH)**: 文本提示的图像分割能够实现细粒度的视觉理解，并且对于人机交互和机器人等领域具有关键作用。然而，现有的监督微调方法通常在测试时忽略显式的链式思考（CoT）推理，这限制了它们对未见提示和领域进行泛化的能力。为了解决这一问题，我们引入了LENS，这是一种可扩展的强化学习框架，能够以端到端的方式同时优化推理过程和分割。我们提出了统一的强化学习奖励，涵盖了句子级、框级和区域级的线索，鼓励模型生成具有信息性的CoT推理并改进掩码质量。使用一个公开可用的30亿参数视觉语言模型，即Qwen2.5-VL-3B-Instruct，LENS在RefCOCO、RefCOCO+和RefCOCOg基准上的平均cIoU达到了81.2%，相比强微调方法GLaMM最高提升了5.6%。这些结果表明，基于RL的CoT推理为文本提示的分割提供了一个稳健的先验，并为更泛化的Segment Anything模型提供了一条实用的道路。相关代码可在以下链接获取。 

---
# A Systematic Study of Deep Learning Models and xAI Methods for Region-of-Interest Detection in MRI Scans 

**Title (ZH)**: 深度学习模型和xAI方法在MRI扫描区域-of-兴趣检测中的系统性研究 

**Authors**: Justin Yiu, Kushank Arora, Daniel Steinberg, Rohit Ghiya  

**Link**: [PDF](https://arxiv.org/pdf/2508.14151)  

**Abstract**: Magnetic Resonance Imaging (MRI) is an essential diagnostic tool for assessing knee injuries. However, manual interpretation of MRI slices remains time-consuming and prone to inter-observer variability. This study presents a systematic evaluation of various deep learning architectures combined with explainable AI (xAI) techniques for automated region of interest (ROI) detection in knee MRI scans. We investigate both supervised and self-supervised approaches, including ResNet50, InceptionV3, Vision Transformers (ViT), and multiple U-Net variants augmented with multi-layer perceptron (MLP) classifiers. To enhance interpretability and clinical relevance, we integrate xAI methods such as Grad-CAM and Saliency Maps. Model performance is assessed using AUC for classification and PSNR/SSIM for reconstruction quality, along with qualitative ROI visualizations. Our results demonstrate that ResNet50 consistently excels in classification and ROI identification, outperforming transformer-based models under the constraints of the MRNet dataset. While hybrid U-Net + MLP approaches show potential for leveraging spatial features in reconstruction and interpretability, their classification performance remains lower. Grad-CAM consistently provided the most clinically meaningful explanations across architectures. Overall, CNN-based transfer learning emerges as the most effective approach for this dataset, while future work with larger-scale pretraining may better unlock the potential of transformer models. 

**Abstract (ZH)**: 磁共振成像（MRI）是评估膝关节损伤的重要诊断工具。然而，手动interpretation of MRI切片仍然耗时且容易产生观察者间差异。本研究系统评估了各种深度学习架构与可解释人工智能（xAI）技术相结合的方法，用于膝关节MRI扫描的自动感兴趣区域（ROI）检测。我们探讨了监督和半监督方法，包括ResNet50、InceptionV3、视觉变换器（ViT）以及多种带有深层感知机（MLP）分类器的U-Net变体。为了增强可解释性和临床相关性，我们整合了Grad-CAM和显著性图等xAI方法。模型性能通过分类的AUC和重建质量的PSNR/SSIM进行评估，并辅以质性的ROI可视化。研究结果表明，在MRNet数据集中，ResNet50在分类和ROI识别方面表现出色，优于基于变压器的模型。虽然混合U-Net + MLP方法在重建和解释性方面显示出潜力，但在分类性能上仍然较低。Grad-CAM在各架构中提供了最具有临床意义的解释。总体而言，基于CNN的迁移学习在该数据集中是最有效的，而未来使用更大规模预训练的工作可能会更好地释放变压器模型的潜力。 

---
# Neuro-inspired Ensemble-to-Ensemble Communication Primitives for Sparse and Efficient ANNs 

**Title (ZH)**: 受神经启发的稀疏高效ANNs的群集间通信基元 

**Authors**: Orestis Konstantaropoulos, Stelios Manolis Smirnakis, Maria Papadopouli  

**Link**: [PDF](https://arxiv.org/pdf/2508.14140)  

**Abstract**: The structure of biological neural circuits-modular, hierarchical, and sparsely interconnected-reflects an efficient trade-off between wiring cost, functional specialization, and robustness. These principles offer valuable insights for artificial neural network (ANN) design, especially as networks grow in depth and scale. Sparsity, in particular, has been widely explored for reducing memory and computation, improving speed, and enhancing generalization. Motivated by systems neuroscience findings, we explore how patterns of functional connectivity in the mouse visual cortex-specifically, ensemble-to-ensemble communication, can inform ANN design. We introduce G2GNet, a novel architecture that imposes sparse, modular connectivity across feedforward layers. Despite having significantly fewer parameters than fully connected models, G2GNet achieves superior accuracy on standard vision benchmarks. To our knowledge, this is the first architecture to incorporate biologically observed functional connectivity patterns as a structural bias in ANN design. We complement this static bias with a dynamic sparse training (DST) mechanism that prunes and regrows edges during training. We also propose a Hebbian-inspired rewiring rule based on activation correlations, drawing on principles of biological plasticity. G2GNet achieves up to 75% sparsity while improving accuracy by up to 4.3% on benchmarks, including Fashion-MNIST, CIFAR-10, and CIFAR-100, outperforming dense baselines with far fewer computations. 

**Abstract (ZH)**: 生物神经回路的结构——模块化、分层和稀疏连接——反映了在布线成本、功能专业化和鲁棒性之间的高效权衡。这些原理为人工神经网络(ANN)设计提供了宝贵的见解，尤其是在网络加深和扩大时。稀疏性尤其广泛探索，以减少内存和计算量，提高速度并增强泛化能力。受系统神经科学发现的启发，我们探讨了小鼠初级视觉皮层中功能连接模式，特别是群体到群体的通信，如何指导ANN设计。我们提出了G2GNet，这是一种新颖的架构，在前馈层间施加稀疏且模块化的连接。尽管参数数量远少于全连接模型，G2GNet在标准视觉基准测试中仍能达到更好的准确性。据我们所知，这是第一个在ANN设计中将生物观察到的功能连接模式作为结构偏倚纳入的架构。我们还通过训练中剪枝和再生长边的动态稀疏训练(DST)机制来补充这种静态偏倚。此外，我们提出了一种基于激活相关性的Hebbian启发式重连规则，利用生物可塑性原理。G2GNet在包括Fashion-MNIST、CIFAR-10和CIFAR-100等基准测试中实现了高达75%的稀疏性，同时提高了高达4.3%的准确性，计算量远少于密集基准模型。 

---
# The Statistical Validation of Innovation Lens 

**Title (ZH)**: 创新棱镜的统计验证 

**Authors**: Giacomo Radaelli, Jonah Lynch  

**Link**: [PDF](https://arxiv.org/pdf/2508.14139)  

**Abstract**: Information overload and the rapid pace of scientific advancement make it increasingly difficult to evaluate and allocate resources to new research proposals. Is there a structure to scientific discovery that could inform such decisions? We present statistical evidence for such structure, by training a classifier that successfully predicts high-citation research papers between 2010-2024 in the Computer Science, Physics, and PubMed domains. 

**Abstract (ZH)**: 信息过载和科学进步的快速节奏使得评估和分配资源给新的研究提案越来越困难。科学发现是否有结构可以帮助做出这样的决策？我们通过训练分类器，成功预测了2010-2024年间计算机科学、物理和PubMed领域高引用的研究论文，提供了统计证据。 

---
# STAS: Spatio-Temporal Adaptive Computation Time for Spiking Transformers 

**Title (ZH)**: STAS: 基于时空自适应计算时间的脉冲变压器 

**Authors**: Donghwa Kang, Doohyun Kim, Sang-Ki Ko, Jinkyu Lee, Brent ByungHoon Kang, Hyeongboo Baek  

**Link**: [PDF](https://arxiv.org/pdf/2508.14138)  

**Abstract**: Spiking neural networks (SNNs) offer energy efficiency over artificial neural networks (ANNs) but suffer from high latency and computational overhead due to their multi-timestep operational nature. While various dynamic computation methods have been developed to mitigate this by targeting spatial, temporal, or architecture-specific redundancies, they remain fragmented. While the principles of adaptive computation time (ACT) offer a robust foundation for a unified approach, its application to SNN-based vision Transformers (ViTs) is hindered by two core issues: the violation of its temporal similarity prerequisite and a static architecture fundamentally unsuited for its principles. To address these challenges, we propose STAS (Spatio-Temporal Adaptive computation time for Spiking transformers), a framework that co-designs the static architecture and dynamic computation policy. STAS introduces an integrated spike patch splitting (I-SPS) module to establish temporal stability by creating a unified input representation, thereby solving the architectural problem of temporal dissimilarity. This stability, in turn, allows our adaptive spiking self-attention (A-SSA) module to perform two-dimensional token pruning across both spatial and temporal axes. Implemented on spiking Transformer architectures and validated on CIFAR-10, CIFAR-100, and ImageNet, STAS reduces energy consumption by up to 45.9%, 43.8%, and 30.1%, respectively, while simultaneously improving accuracy over SOTA models. 

**Abstract (ZH)**: 基于时空自适应计算时间的突触神经网络视觉变换器(STAS) 

---
# ERIS: An Energy-Guided Feature Disentanglement Framework for Out-of-Distribution Time Series Classification 

**Title (ZH)**: ERIS：一种能量导向的特征解缠框架用于分布外时间序列分类 

**Authors**: Xin Wu, Fei Teng, Ji Zhang, Xingwang Li, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14134)  

**Abstract**: An ideal time series classification (TSC) should be able to capture invariant representations, but achieving reliable performance on out-of-distribution (OOD) data remains a core obstacle. This obstacle arises from the way models inherently entangle domain-specific and label-relevant features, resulting in spurious correlations. While feature disentanglement aims to solve this, current methods are largely unguided, lacking the semantic direction required to isolate truly universal features. To address this, we propose an end-to-end Energy-Regularized Information for Shift-Robustness (\textbf{ERIS}) framework to enable guided and reliable feature disentanglement. The core idea is that effective disentanglement requires not only mathematical constraints but also semantic guidance to anchor the separation process. ERIS incorporates three key mechanisms to achieve this goal. Specifically, we first introduce an energy-guided calibration mechanism, which provides crucial semantic guidance for the separation, enabling the model to self-calibrate. Additionally, a weight-level orthogonality strategy enforces structural independence between domain-specific and label-relevant features, thereby mitigating their interference. Moreover, an auxiliary adversarial training mechanism enhances robustness by injecting structured perturbations. Experiments demonstrate that ERIS improves upon state-of-the-art baselines by an average of 4.04% accuracy across four benchmarks. 

**Abstract (ZH)**: 一种理想的时序分类（TSC）应该能够捕获不变表示，但在处理分布外（OOD）数据时实现可靠性能仍然是一个核心障碍。这一障碍源于模型固有地将领域特定和标签相关特征纠缠在一起，导致虚假相关性。虽然特征去纠缠旨在解决这一问题，但当前的方法大多缺乏方向性指导，无法分离真正通用的特征。为此，我们提出了一种端到端的能量正则化信息对移不变鲁棒性（\textbf{ERIS}）框架，以实现指导性和可靠性的特征去纠缠。核心思想是，有效的去纠缠不仅需要数学约束，还需要语义指导以锚定分离过程。ERIS 通过三种关键机制实现了这一目标。具体来说，我们首先引入了一种能量导向的校准机制，为分离提供重要的语义指导，使模型能够自我校准。此外，一种权重级别的正交策略强制执行领域特定和标签相关特征之间的结构独立性，从而减少它们之间的干扰。同时，一种辅助对抗训练机制通过注入结构性扰动增强了鲁棒性。实验表明，ERIS 在四个基准上的准确率平均提高了 4.04%。 

---
# Automated surgical planning with nnU-Net: delineation of the anatomy in hepatobiliary phase MRI 

**Title (ZH)**: 基于nnU-Net的自动手术规划：肝胆期MRI中的人体解剖轮廓化 

**Authors**: Karin A. Olthof, Matteo Fusagli, Bianca Güttner, Tiziano Natali, Bram Westerink, Stefanie Speidel, Theo J.M. Ruers, Koert F.D. Kuhlmann, Andrey Zhylka  

**Link**: [PDF](https://arxiv.org/pdf/2508.14133)  

**Abstract**: Background: The aim of this study was to develop and evaluate a deep learning-based automated segmentation method for hepatic anatomy (i.e., parenchyma, tumors, portal vein, hepatic vein and biliary tree) from the hepatobiliary phase of gadoxetic acid-enhanced MRI. This method should ease the clinical workflow of preoperative planning.
Methods: Manual segmentation was performed on hepatobiliary phase MRI scans from 90 consecutive patients who underwent liver surgery between January 2020 and October 2023. A deep learning network (nnU-Net v1) was trained on 72 patients with an extra focus on thin structures and topography preservation. Performance was evaluated on an 18-patient test set by comparing automated and manual segmentations using Dice similarity coefficient (DSC). Following clinical integration, 10 segmentations (assessment dataset) were generated using the network and manually refined for clinical use to quantify required adjustments using DSC.
Results: In the test set, DSCs were 0.97+/-0.01 for liver parenchyma, 0.80+/-0.04 for hepatic vein, 0.79+/-0.07 for biliary tree, 0.77+/-0.17 for tumors, and 0.74+/-0.06 for portal vein. Average tumor detection rate was 76.6+/-24.1%, with a median of one false-positive per patient. The assessment dataset showed minor adjustments were required for clinical use of the 3D models, with high DSCs for parenchyma (1.00+/-0.00), portal vein (0.98+/-0.01) and hepatic vein (0.95+/-0.07). Tumor segmentation exhibited greater variability (DSC 0.80+/-0.27). During prospective clinical use, the model detected three additional tumors initially missed by radiologists.
Conclusions: The proposed nnU-Net-based segmentation method enables accurate and automated delineation of hepatic anatomy. This enables 3D planning to be applied efficiently as a standard-of-care for every patient undergoing liver surgery. 

**Abstract (ZH)**: 背景: 本研究旨在开发并评估一种基于深度学习的自动化分割方法，用于从钆贝酸二钠增强MRI的肝胆期扫描中分割肝脏解剖结构（即肝实质、肿瘤、门静脉、肝静脉和胆管树），以简化术前规划的临床工作流程。 

---
# An Improved Multi-Agent Algorithm for Cooperative and Competitive Environments by Identifying and Encouraging Cooperation among Agents 

**Title (ZH)**: 一种通过识别和鼓励智能体之间的合作来改进的多智能体算法及其在竞争与合作环境中的应用 

**Authors**: Junjie Qi, Siqi Mao, Tianyi Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14131)  

**Abstract**: We propose an improved algorithm by identifying and encouraging cooperative behavior in multi-agent environments. First, we analyze the shortcomings of existing algorithms in addressing multi-agent reinforcement learning problems. Then, based on the existing algorithm MADDPG, we introduce a new parameter to increase the reward that an agent can obtain when cooperative behavior among agents is identified. Finally, we compare our improved algorithm with MADDPG in environments from PettingZoo. The results show that the new algorithm helps agents achieve both higher team rewards and individual rewards. 

**Abstract (ZH)**: 我们提出一种改进算法，通过识别和鼓励多智能体环境中的协作行为。首先，我们分析现有算法在解决多智能体 reinforcement learning 问题中的不足。然后，基于现有的算法 MADDPG，我们引入一个新的参数，以提高智能体在识别到协作行为时所能获得的奖励。最后，我们在来自 PettingZoo 的环境中将改进后的算法与 MADDPG 进行比较。结果表明，新的算法有助于智能体实现更高的团队奖励和个人奖励。 

---
# Fracture Detection and Localisation in Wrist and Hand Radiographs using Detection Transformer Variants 

**Title (ZH)**: 使用检测变换器变体在腕部和手部放射影像中进行骨折检测与定位 

**Authors**: Aditya Bagri, Vasanthakumar Venugopal, Anandakumar D, Revathi Ezhumalai, Kalyan Sivasailam, Bargava Subramanian, VarshiniPriya, Meenakumari K S, Abi M, Renita S  

**Link**: [PDF](https://arxiv.org/pdf/2508.14129)  

**Abstract**: Background: Accurate diagnosis of wrist and hand fractures using radiographs is essential in emergency care, but manual interpretation is slow and prone to errors. Transformer-based models show promise in improving medical image analysis, but their application to extremity fractures is limited. This study addresses this gap by applying object detection transformers to wrist and hand X-rays.
Methods: We fine-tuned the RT-DETR and Co-DETR models, pre-trained on COCO, using over 26,000 annotated X-rays from a proprietary clinical dataset. Each image was labeled for fracture presence with bounding boxes. A ResNet-50 classifier was trained on cropped regions to refine abnormality classification. Supervised contrastive learning was used to enhance embedding quality. Performance was evaluated using AP@50, precision, and recall metrics, with additional testing on real-world X-rays.
Results: RT-DETR showed moderate results (AP@50 = 0.39), while Co-DETR outperformed it with an AP@50 of 0.615 and faster convergence. The integrated pipeline achieved 83.1% accuracy, 85.1% precision, and 96.4% recall on real-world X-rays, demonstrating strong generalization across 13 fracture types. Visual inspection confirmed accurate localization.
Conclusion: Our Co-DETR-based pipeline demonstrated high accuracy and clinical relevance in wrist and hand fracture detection, offering reliable localization and differentiation of fracture types. It is scalable, efficient, and suitable for real-time deployment in hospital workflows, improving diagnostic speed and reliability in musculoskeletal radiology. 

**Abstract (ZH)**: 背景：准确诊断腕部和手部骨折对于急诊护理至关重要，但手动解释影像资料速度慢且容易出错。基于 Transformer 的模型在改善医学影像分析方面展现出潜力，但在四肢骨折的应用方面有限。本研究通过将对象检测 Transformer 应用于腕部和手部 X 光片来填补这一空白。

方法：我们使用超过 26,000 张带有注释的私有临床数据集中的 X 光片对 RT-DETR 和 Co-DETR 模型进行了微调，这些模型预先在 COCO 上进行了训练。每张图像都标注了骨折的存在情况，并用边界框进行标记。ResNet-50 分类器在裁剪区域上进行了训练以细化异常分类。使用监督对比学习来增强嵌入质量。使用 AP@50、精确度和召回率指标评估性能，并通过实际 X 光片进行了进一步测试。

结果：RT-DETR 的表现为中等 (AP@50 = 0.39)，而 Co-DETR 在 AP@50 方面表现出色 (0.615)，收敛速度更快。集成管道在实际 X 光片上的准确率为 83.1%，精确率为 85.1%，召回率为 96.4%，展示了在 13 种骨折类型上的强大泛化能力。视觉检查证实了定位的准确性。

结论：我们的基于 Co-DETR 的管道在腕部和手部骨折检测中展示了高精度和临床相关性，提供了可靠的位置检测和骨折类型的区分能力。该方案具有可扩展性、高效性，并适合实时部署在医院工作流程中，提高了骨科放射学诊断的速度和可靠性。 

---
# CCFC: Core & Core-Full-Core Dual-Track Defense for LLM Jailbreak Protection 

**Title (ZH)**: CCFC: 核心与核心-全核心双轨防御技术用于LLM脱戒保护 

**Authors**: Jiaming Hu, Haoyu Wang, Debarghya Mukherjee, Ioannis Ch. Paschalidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.14128)  

**Abstract**: Jailbreak attacks pose a serious challenge to the safe deployment of large language models (LLMs). We introduce CCFC (Core & Core-Full-Core), a dual-track, prompt-level defense framework designed to mitigate LLMs' vulnerabilities from prompt injection and structure-aware jailbreak attacks. CCFC operates by first isolating the semantic core of a user query via few-shot prompting, and then evaluating the query using two complementary tracks: a core-only track to ignore adversarial distractions (e.g., toxic suffixes or prefix injections), and a core-full-core (CFC) track to disrupt the structural patterns exploited by gradient-based or edit-based attacks. The final response is selected based on a safety consistency check across both tracks, ensuring robustness without compromising on response quality. We demonstrate that CCFC cuts attack success rates by 50-75% versus state-of-the-art defenses against strong adversaries (e.g., DeepInception, GCG), without sacrificing fidelity on benign queries. Our method consistently outperforms state-of-the-art prompt-level defenses, offering a practical and effective solution for safer LLM deployment. 

**Abstract (ZH)**: Jailbreak攻击对大型语言模型的安全部署构成严重挑战。我们提出了CCFC（核心与全核心）双轨制提示级别防御框架，旨在减轻由提示注入和结构感知型 jailbreak 攻击引起的大型语言模型的脆弱性。CCFC 通过少量提示隔离用户查询的语义核心，并使用两条互补的轨道进行评估：核心仅轨道忽略对抗性干扰（例如，有害的后缀或前缀注入），以及核心全核心（CFC）轨道以破坏基于梯度或基于编辑的攻击所利用的结构模式。最终响应基于两轨道的安全一致性检查进行选择，确保其稳健性而不牺牲响应质量。我们证明，与强对手（例如，DeepInception、GCG）的最新防御措施相比，CCFC 可将攻击成功率削减 50-75%，并且在良性查询上不牺牲准确度。我们的方法持续优于最新的提示级别防御措施，提供了一种实用而有效的解决方案，以实现更安全的大规模语言模型部署。 

---
# A Cost-Effective Framework for Predicting Parking Availability Using Geospatial Data and Machine Learning 

**Title (ZH)**: 基于地理空间数据和机器学习的低成本预测停车位可用性框架 

**Authors**: Madyan Bagosher, Tala Mustafa, Mohammad Alsmirat, Amal Al-Ali, Isam Mashhour Al Jawarneh  

**Link**: [PDF](https://arxiv.org/pdf/2508.14125)  

**Abstract**: As urban populations continue to grow, cities face numerous challenges in managing parking and determining occupancy. This issue is particularly pronounced in university campuses, where students need to find vacant parking spots quickly and conveniently during class timings. The limited availability of parking spaces on campuses underscores the necessity of implementing efficient systems to allocate vacant parking spots effectively. We propose a smart framework that integrates multiple data sources, including street maps, mobility, and meteorological data, through a spatial join operation to capture parking behavior and vehicle movement patterns over the span of 3 consecutive days with an hourly duration between 7AM till 3PM. The system will not require any sensing tools to be installed in the street or in the parking area to provide its services since all the data needed will be collected using location services. The framework will use the expected parking entrance and time to specify a suitable parking area. Several forecasting models, namely, Linear Regression, Support Vector Regression (SVR), Random Forest Regression (RFR), and Long Short-Term Memory (LSTM), are evaluated. Hyperparameter tuning was employed using grid search, and model performance is assessed using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) and Coefficient of Determination (R2). Random Forest Regression achieved the lowest RMSE of 0.142 and highest R2 of 0.582. However, given the time-series nature of the task, an LSTM model may perform better with additional data and longer timesteps. 

**Abstract (ZH)**: 随着城市人口的持续增长，城市在管理停车和确定停车位占用情况方面面临着诸多挑战。这一问题在大学校园中尤为突出，学生在上课期间需要快速便捷地找到空闲停车位。校园内有限的停车位凸显了实施高效系统以有效分配空闲停车位的必要性。我们提出了一种智能框架，通过空间连接操作整合街道地图、交通流动和气象数据等多种数据源，以每天3小时为单位，在上午7点至下午3点之间的时间段内捕捉3天的停车行为和车辆移动模式。该系统无需在街道或停车场安装任何传感设备即可提供服务，所有所需数据将通过位置服务收集。该框架将根据预计的停车入口和时间指定合适的停车区域。评估了多种预测模型，包括线性回归、支持向量回归（SVR）、随机森林回归（RFR）和长短期记忆网络（LSTM）。使用网格搜索进行超参数调整，并使用均方根误差（RMSE）、平均绝对误差（MAE）和决定系数（R2）评估模型性能。随机森林回归实现了最低的RMSE（0.142）和最高的R2（0.582）。然而，鉴于任务的时间序列性质，在有更多数据和更长的时间步长情况下，LSTM模型可能会表现更好。 

---
# AI Agents for Photonic Integrated Circuit Design Automation 

**Title (ZH)**: AI代理在光子集成电路设计自动化中的应用 

**Authors**: Ankita Sharma, YuQi Fu, Vahid Ansari, Rishabh Iyer, Fiona Kuang, Kashish Mistry, Raisa Islam Aishy, Sara Ahmad, Joaquin Matres, Dirk R. Englund, Joyce K.S. Poon  

**Link**: [PDF](https://arxiv.org/pdf/2508.14123)  

**Abstract**: We present Photonics Intelligent Design and Optimization (PhIDO), a multi-agent framework that converts natural-language photonic integrated circuit (PIC) design requests into layout mask files. We compare 7 reasoning large language models for PhIDO using a testbench of 102 design descriptions that ranged from single devices to 112-component PICs. The success rate for single-device designs was up to 91%. For design queries with less than or equal to 15 components, o1, Gemini-2.5-pro, and Claude Opus 4 achieved the highest end-to-end pass@5 success rates of approximately 57%, with Gemini-2.5-pro requiring the fewest output tokens and lowest cost. The next steps toward autonomous PIC development include standardized knowledge representations, expanded datasets, extended verification, and robotic automation. 

**Abstract (ZH)**: 光电智能设计与优化(PhIDO)：一种多代理框架，将自然语言的光子集成电路(PIC)设计请求转换为布局掩膜文件 

---
# SimGenHOI: Physically Realistic Whole-Body Humanoid-Object Interaction via Generative Modeling and Reinforcement Learning 

**Title (ZH)**: SimGenHOI: 通过生成建模和强化学习实现的物理真实人体形态人偶与物体交互 

**Authors**: Yuhang Lin, Yijia Xie, Jiahong Xie, Yuehao Huang, Ruoyu Wang, Jiajun Lv, Yukai Ma, Xingxing Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14120)  

**Abstract**: Generating physically realistic humanoid-object interactions (HOI) is a fundamental challenge in robotics. Existing HOI generation approaches, such as diffusion-based models, often suffer from artifacts such as implausible contacts, penetrations, and unrealistic whole-body actions, which hinder successful execution in physical environments. To address these challenges, we introduce SimGenHOI, a unified framework that combines the strengths of generative modeling and reinforcement learning to produce controllable and physically plausible HOI. Our HOI generative model, based on Diffusion Transformers (DiT), predicts a set of key actions conditioned on text prompts, object geometry, sparse object waypoints, and the initial humanoid pose. These key actions capture essential interaction dynamics and are interpolated into smooth motion trajectories, naturally supporting long-horizon generation. To ensure physical realism, we design a contact-aware whole-body control policy trained with reinforcement learning, which tracks the generated motions while correcting artifacts such as penetration and foot sliding. Furthermore, we introduce a mutual fine-tuning strategy, where the generative model and the control policy iteratively refine each other, improving both motion realism and tracking robustness. Extensive experiments demonstrate that SimGenHOI generates realistic, diverse, and physically plausible humanoid-object interactions, achieving significantly higher tracking success rates in simulation and enabling long-horizon manipulation tasks. Code will be released upon acceptance on our project page: this https URL. 

**Abstract (ZH)**: 生成物理上真实的类人物体交互（HOI）是机器人学中的一个基础挑战。现有的HOI生成方法，如基于扩散的模型，常常受到不合理接触、穿透和不现实的整体动作等瑕疵的影响，这阻碍了在物理环境中成功的执行。为解决这些挑战，我们引入了SimGenHOI，这是一种结合生成建模和强化学习优势的统一框架，用于生成可控且物理上合理的HOI。我们的HOI生成模型基于扩散变换器（DiT），在文本提示、物体几何、稀疏物体航点以及初始类人姿态的条件下预测一系列关键动作。这些关键动作捕捉了关键的交互动态，并被插值为平滑的运动轨迹，自然支持长时间段的生成。为确保物理真实性，我们设计了一种基于接触感知的全身控制策略，该策略通过强化学习训练，跟踪生成的运动并修正诸如穿透和脚滑等瑕疵。此外，我们提出了一个互训策略，生成模型和控制策略在生成和跟踪中迭代地相互提高，增强运动的真实性和跟踪鲁棒性。广泛的实验表明，SimGenHOI能够生成逼真、多样且物理上合理的类人物体交互，在模拟中显著提高跟踪成功率，并能够执行长时间段的操作任务。代码将在接受后在我们的项目页面发布：this https URL。 

---
# Documenting Deployment with Fabric: A Repository of Real-World AI Governance 

**Title (ZH)**: 基于Fabric的部署文档：真实世界AI治理案例库 

**Authors**: Mackenzie Jorgensen, Kendall Brogle, Katherine M. Collins, Lujain Ibrahim, Arina Shah, Petra Ivanovic, Noah Broestl, Gabriel Piles, Paul Dongha, Hatim Abdulhussein, Adrian Weller, Jillian Powers, Umang Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2508.14119)  

**Abstract**: Artificial intelligence (AI) is increasingly integrated into society, from financial services and traffic management to creative writing. Academic literature on the deployment of AI has mostly focused on the risks and harms that result from the use of AI. We introduce Fabric, a publicly available repository of deployed AI use cases to outline their governance mechanisms. Through semi-structured interviews with practitioners, we collect an initial set of 20 AI use cases. In addition, we co-design diagrams of the AI workflow with the practitioners. We discuss the oversight mechanisms and guardrails used in practice to safeguard AI use. The Fabric repository includes visual diagrams of AI use cases and descriptions of the deployed systems. Using the repository, we surface gaps in governance and find common patterns in human oversight of deployed AI systems. We intend for Fabric to serve as an extendable, evolving tool for researchers to study the effectiveness of AI governance. 

**Abstract (ZH)**: 人工智能（AI）日益融入社会，从金融服务和交通管理到创意写作。关于AI部署的相关学术文献主要集中在AI使用过程中带来的风险和危害。我们介绍了Fabric，一个公开的AI使用案例仓库，用于阐述其治理机制。通过与实践者的半结构化访谈，我们收集了20个初始AI使用案例。此外，我们与实践者共同设计了AI工作流程图。我们讨论了实践中使用的监督机制和护栏，以保障AI使用。Fabric仓库包括AI使用案例的可视化图表和部署系统的说明。利用该仓库，我们揭示了治理中的差距，并发现了在部署AI系统中常见的监管模式。我们旨在让Fabric成为一个可扩展、不断进化的工具，供研究者研究AI治理的有效性。 

---
# Enriching Moral Perspectives on AI: Concepts of Trust amongst Africans 

**Title (ZH)**: 丰富人工智能的道德视角：非洲人的信任概念 

**Authors**: Lameck Mbangula Amugongo, Nicola J Bidwell, Joseph Mwatukange  

**Link**: [PDF](https://arxiv.org/pdf/2508.14116)  

**Abstract**: The trustworthiness of AI is considered essential to the adoption and application of AI systems. However, the meaning of trust varies across industry, research and policy spaces. Studies suggest that professionals who develop and use AI regard an AI system as trustworthy based on their personal experiences and social relations at work. Studies about trust in AI and the constructs that aim to operationalise trust in AI (e.g., consistency, reliability, explainability and accountability). However, the majority of existing studies about trust in AI are situated in Western, Educated, Industrialised, Rich and Democratic (WEIRD) societies. The few studies about trust and AI in Africa do not include the views of people who develop, study or use AI in their work. In this study, we surveyed 157 people with professional and/or educational interests in AI from 25 African countries, to explore how they conceptualised trust in AI. Most respondents had links with workshops about trust and AI in Africa in Namibia and Ghana. Respondents' educational background, transnational mobility, and country of origin influenced their concerns about AI systems. These factors also affected their levels of distrust in certain AI applications and their emphasis on specific principles designed to foster trust. Respondents often expressed that their values are guided by the communities in which they grew up and emphasised communal relations over individual freedoms. They described trust in many ways, including applying nuances of Afro-relationalism to constructs in international discourse, such as reliability and reliance. Thus, our exploratory study motivates more empirical research about the ways trust is practically enacted and experienced in African social realities of AI design, use and governance. 

**Abstract (ZH)**: 人工智能的可信性被认为是其采纳和应用的关键。然而，信任的意义在不同行业、研究和政策空间中有所差异。研究表明，开发和使用人工智能的专业人士基于其工作中的个人经验和社会关系认为人工智能系统可信。关于人工智能中的信任以及旨在实现人工智能信任的构建维度（如一致性、可靠性、可解释性和可问责性）的研究已有所成果。然而，大多数关于人工智能信任的研究都集中在西方、受教育、工业化、富裕和民主（WEIRD）社会。关于非洲人工智能信任和人工智能的研究中，很少包含开发、研究或在工作中使用人工智能人员的观点。在本研究中，我们对来自25个非洲国家的157名具有人工智能专业和/或教育兴趣的人进行了调查，以了解他们如何概念化人工智能中的信任。大多数受访者与纳米比亚和加纳的信任与人工智能研讨会有关联。受访者的教育背景、跨国流动性以及原籍国影响了他们对人工智能系统的顾虑。这些因素也影响了他们对某些人工智能应用的信任程度以及他们对某些促进信任的原则的重视程度。受访者往往表示，他们的价值观受到成长环境的影响，强调社群关系而非个人自由。他们以多种方式描述信任，包括将非洲关系主义的细微差别应用于国际讨论中的可靠性等概念。因此，我们的探索性研究促使更多实证研究关注人工智能设计、使用和治理中非洲社会现实中的信任实践与体验。 

---
# Towards Low-Latency Tracking of Multiple Speakers With Short-Context Speaker Embeddings 

**Title (ZH)**: 面向短上下文 speaker 嵌入的低延迟多说话人跟踪方法 

**Authors**: Taous Iatariene, Alexandre Guérin, Romain Serizel  

**Link**: [PDF](https://arxiv.org/pdf/2508.14115)  

**Abstract**: Speaker embeddings are promising identity-related features that can enhance the identity assignment performance of a tracking system by leveraging its spatial predictions, i.e, by performing identity reassignment. Common speaker embedding extractors usually struggle with short temporal contexts and overlapping speech, which imposes long-term identity reassignment to exploit longer temporal contexts. However, this increases the probability of tracking system errors, which in turn impacts negatively on identity reassignment. To address this, we propose a Knowledge Distillation (KD) based training approach for short context speaker embedding extraction from two speaker mixtures. We leverage the spatial information of the speaker of interest using beamforming to reduce overlap. We study the feasibility of performing identity reassignment over blocks of fixed size, i.e., blockwise identity reassignment, to go towards a low-latency speaker embedding based tracking system. Results demonstrate that our distilled models are effective at short-context embedding extraction and more robust to overlap. Although, blockwise reassignment results indicate that further work is needed to handle simultaneous speech more effectively. 

**Abstract (ZH)**: 基于知识蒸馏的短时上下文说话人嵌入提取方法及块级身份重新指派研究 

---
# Ambiguity Resolution with Human Feedback for Code Writing Tasks 

**Title (ZH)**: 基于人类反馈的代码编写任务中的歧义解析 

**Authors**: Aditey Nandan, Viraj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.14114)  

**Abstract**: Specifications for code writing tasks are usually expressed in natural language and may be ambiguous. Programmers must therefore develop the ability to recognize ambiguities in task specifications and resolve them by asking clarifying questions. We present and evaluate a prototype system, based on a novel technique (ARHF: Ambiguity Resolution with Human Feedback), that (1) suggests specific inputs on which a given task specification may be ambiguous, (2) seeks limited human feedback about the code's desired behavior on those inputs, and (3) uses this feedback to generate code that resolves these ambiguities. We evaluate the efficacy of our prototype, and we discuss the implications of such assistive systems on Computer Science education. 

**Abstract (ZH)**: 代码编写任务的规定通常用自然语言表达并且可能存在歧义。程序员必须具备识别任务规定中歧义并借助澄清问题的方式解决这些歧义的能力。我们提出并评估了一个基于新型技术（ARHF：Ambiguity Resolution with Human Feedback）的原型系统，该系统能够（1）建议特定的输入，在这些输入上给定的任务规定可能存在歧义；（2）寻求对于这些输入代码期望行为的有限人力反馈；以及（3）利用这些反馈生成解决这些歧义的代码。我们评估了该原型系统的有效性，并讨论了此类辅助系统对计算机科学教育的影响。 

---
# Federated Action Recognition for Smart Worker Assistance Using FastPose 

**Title (ZH)**: 基于FastPose的联邦动作识别及其在智能工人辅助中的应用 

**Authors**: Vinit Hegiste, Vidit Goyal, Tatjana Legler, Martin Ruskowski  

**Link**: [PDF](https://arxiv.org/pdf/2508.14113)  

**Abstract**: In smart manufacturing environments, accurate and real-time recognition of worker actions is essential for productivity, safety, and human-machine collaboration. While skeleton-based human activity recognition (HAR) offers robustness to lighting, viewpoint, and background variations, most existing approaches rely on centralized datasets, which are impractical in privacy-sensitive industrial scenarios. This paper presents a federated learning (FL) framework for pose-based HAR using a custom skeletal dataset of eight industrially relevant upper-body gestures, captured from five participants and processed using a modified FastPose model. Two temporal backbones, an LSTM and a Transformer encoder, are trained and evaluated under four paradigms: centralized, local (per-client), FL with weighted federated averaging (FedAvg), and federated ensemble learning (FedEnsemble). On the global test set, the FL Transformer improves over centralized training by +12.4 percentage points, with FedEnsemble delivering a +16.3 percentage points gain. On an unseen external client, FL and FedEnsemble exceed centralized accuracy by +52.6 and +58.3 percentage points, respectively. These results demonstrate that FL not only preserves privacy but also substantially enhances cross-user generalization, establishing it as a practical solution for scalable, privacy-aware HAR in heterogeneous industrial settings. 

**Abstract (ZH)**: 在智能制造环境中，基于姿态的人机协作实时识别对于提高生产效率、确保安全性和促进人机协作至关重要。尽管基于骨架的动作识别（HAR）能够应对光照、视角和背景的变化，但大多数现有方法依赖于集中化的数据集，在涉及个人隐私的工业场景中并不实用。本文提出了一种联邦学习（FL）框架，用于基于姿态的动作识别，该框架利用了一个包含八种工业相关上肢动作的自定义骨架数据集，数据来自于五名参与者并通过修改后的FastPose模型进行处理。两种时间上卷积结构，LSTM和Transformer编码器，分别在集中式、本地（按客户端）、加权联邦平均（FedAvg）和联邦集成学习（FedEnsemble）四种范式下进行训练和评估。在全局测试集上，FL Transformer相较于集中式训练提高了12.4个百分点，而FedEnsemble则提高了16.3个百分点。在一个未见过的外部客户端上，FL和FedEnsemble的准确率分别提高了52.6和58.3个百分点。这些结果表明，联邦学习不仅能保护隐私，还能显著提升跨用户的泛化能力，从而为异构工业环境下的可扩展、隐私保护的动作识别提供了实用的解决方案。 

---
# Surya: Foundation Model for Heliophysics 

**Title (ZH)**: Surya：太阳物理领域的基础模型 

**Authors**: Sujit Roy, Johannes Schmude, Rohit Lal, Vishal Gaur, Marcus Freitag, Julian Kuehnert, Theodore van Kessel, Dinesha V. Hegde, Andrés Muñoz-Jaramillo, Johannes Jakubik, Etienne Vos, Kshitiz Mandal, Ata Akbari Asanjan, Joao Lucas de Sousa Almeida, Amy Lin, Talwinder Singh, Kang Yang, Chetraj Pandey, Jinsu Hong, Berkay Aydin, Thorsten Kurth, Ryan McGranaghan, Spiridon Kasapis, Vishal Upendran, Shah Bahauddin, Daniel da Silva, Nikolai V. Pogorelov, Campbell Watson, Manil Maskey, Madhulika Guhathakurta, Juan Bernabe-Moreno, Rahul Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2508.14112)  

**Abstract**: Heliophysics is central to understanding and forecasting space weather events and solar activity. Despite decades of high-resolution observations from the Solar Dynamics Observatory (SDO), most models remain task-specific and constrained by scarce labeled data, limiting their capacity to generalize across solar phenomena. We introduce Surya, a 366M parameter foundation model for heliophysics designed to learn general-purpose solar representations from multi-instrument SDO observations, including eight Atmospheric Imaging Assembly (AIA) channels and five Helioseismic and Magnetic Imager (HMI) products. Surya employs a spatiotemporal transformer architecture with spectral gating and long--short range attention, pretrained on high-resolution solar image forecasting tasks and further optimized through autoregressive rollout tuning. Zero-shot evaluations demonstrate its ability to forecast solar dynamics and flare events, while downstream fine-tuning with parameter-efficient Low-Rank Adaptation (LoRA) shows strong performance on solar wind forecasting, active region segmentation, solar flare forecasting, and EUV spectra. Surya is the first foundation model in heliophysics that uses time advancement as a pretext task on full-resolution SDO data. Its novel architecture and performance suggest that the model is able to learn the underlying physics behind solar evolution. 

**Abstract (ZH)**: 太阳物理是理解与预报空间天气事件和太阳活动的关键。尽管太阳动力学观测卫星（SDO）提供了高分辨率的长期观测数据，大多数模型仍然具有特定任务性和不足的标记数据限制，这限制了它们在太阳现象上的泛化能力。我们提出了Surya，一个366M参数的基础模型，旨在从太阳动力学观测卫星（SDO）的多仪器观测数据中学习通用的太阳表示，包括八个大气成像组合（AIA）通道和五个太阳地震和磁场成像（HMI）产品。Surya采用时空变换器架构，包含频谱门控和长短期注意机制，并在高分辨率太阳图像预测任务上进行预训练，进一步通过自回归滚动调优以优化性能。零样本评估展示了其预报太阳动力学和耀斑事件的能力，而基于参数高效的低秩适应（LoRA）的下游微调在日冕物质抛射预报、活跃区域分割、耀斑预报和极端紫外线光谱方面表现出色。Surya是太阳物理学中第一个使用时间推进作为前置任务的基础模型，其新颖的架构和性能表明该模型能够学习太阳演化的基本物理机制。 

---
# PAPPL: Personalized AI-Powered Progressive Learning Platform 

**Title (ZH)**: 个性化AI驱动渐进学习平台 

**Authors**: Shayan Bafandkar, Sungyong Chung, Homa Khosravian, Alireza Talebpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.14109)  

**Abstract**: Engineering education has historically been constrained by rigid, standardized frameworks, often neglecting students' diverse learning needs and interests. While significant advancements have been made in online and personalized education within K-12 and foundational sciences, engineering education at both undergraduate and graduate levels continues to lag in adopting similar innovations. Traditional evaluation methods, such as exams and homework assignments, frequently overlook individual student requirements, impeding personalized educational experiences. To address these limitations, this paper introduces the Personalized AI-Powered Progressive Learning (PAPPL) platform, an advanced Intelligent Tutoring System (ITS) designed specifically for engineering education. It highlights the development of a scalable, data-driven tutoring environment leveraging cutting-edge AI technology to enhance personalized learning across diverse academic disciplines, particularly in STEM fields. PAPPL integrates core ITS components including the expert module, student module, tutor module, and user interface, and utilizes GPT-4o, a sophisticated large language model (LLM), to deliver context-sensitive and pedagogically sound hints based on students' interactions. The system uniquely records student attempts, detects recurring misconceptions, and generates progressively targeted feedback, providing personalized assistance that adapts dynamically to each student's learning profile. Additionally, PAPPL offers instructors detailed analytics, empowering evidence-based adjustments to teaching strategies. This study provides a fundamental framework for the progression of Generative ITSs scalable to all education levels, delivering important perspectives on personalized progressive learning and the wider possibilities of Generative AI in the field of education. 

**Abstract (ZH)**: 工程教育 historically 一直受到僵化的标准化框架的限制，往往忽视了学生的多元学习需求和兴趣。虽然K-12教育和基础科学领域的在线和个性化教育已经取得了显著进展，但工程教育在本科学位和研究生学位层面仍然未能采用类似的创新。传统的评估方法，如考试和作业，经常忽视个别学生的需求，阻碍了个性化的学习体验。为了应对这些局限性，本文介绍了个性化AI驱动渐进式学习（PAPPL）平台，这是一个专门为工程教育设计的高级智能辅导系统（ITS）。该平台利用前沿的AI技术开发了一个可扩展、数据驱动的辅导环境，旨在跨多个学术学科尤其是STEM领域增强个性化学习。PAPPL集成了专家模块、学生模块、辅导员模块和用户界面，并利用GPT-4o这一复杂的大规模语言模型（LLM），根据学生的互动提供上下文相关且符合教育学的提示。该系统记录学生的尝试，检测重复的概念错误，并生成渐进式有针对性的反馈，提供个性化帮助，动态适应每个学生的学习特征。此外，PAPPL还为教师提供了详细的分析，使他们能够根据数据来调整教学策略。本研究为生成性ITS的发展提供了一个基本框架，使其可以扩展到所有教育层次，提供建立在个性化渐进式学习和教育领域生成式AI更广泛可能性的基础观点。 

---
# SuryaBench: Benchmark Dataset for Advancing Machine Learning in Heliophysics and Space Weather Prediction 

**Title (ZH)**: SuryaBench: 太阳物理学与空间天气预测中机器学习的发展基准数据集 

**Authors**: Sujit Roy, Dinesha V. Hegde, Johannes Schmude, Amy Lin, Vishal Gaur, Rohit Lal, Kshitiz Mandal, Talwinder Singh, Andrés Muñoz-Jaramillo, Kang Yang, Chetraj Pandey, Jinsu Hong, Berkay Aydin, Ryan McGranaghan, Spiridon Kasapis, Vishal Upendran, Shah Bahauddin, Daniel da Silva, Marcus Freitag, Iksha Gurung, Nikolai Pogorelov, Campbell Watson, Manil Maskey, Juan Bernabe-Moreno, Rahul Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2508.14107)  

**Abstract**: This paper introduces a high resolution, machine learning-ready heliophysics dataset derived from NASA's Solar Dynamics Observatory (SDO), specifically designed to advance machine learning (ML) applications in solar physics and space weather forecasting. The dataset includes processed imagery from the Atmospheric Imaging Assembly (AIA) and Helioseismic and Magnetic Imager (HMI), spanning a solar cycle from May 2010 to July 2024. To ensure suitability for ML tasks, the data has been preprocessed, including correction of spacecraft roll angles, orbital adjustments, exposure normalization, and degradation compensation. We also provide auxiliary application benchmark datasets complementing the core SDO dataset. These provide benchmark applications for central heliophysics and space weather tasks such as active region segmentation, active region emergence forecasting, coronal field extrapolation, solar flare prediction, solar EUV spectra prediction, and solar wind speed estimation. By establishing a unified, standardized data collection, this dataset aims to facilitate benchmarking, enhance reproducibility, and accelerate the development of AI-driven models for critical space weather prediction tasks, bridging gaps between solar physics, machine learning, and operational forecasting. 

**Abstract (ZH)**: 基于NASA太阳动力学观测台(SDO)的高分辨率机器学习_ready太阳物理数据集：用于太阳物理和空间天气预报的机器学习应用研究 

---
# High-Throughput Low-Cost Segmentation of Brightfield Microscopy Live Cell Images 

**Title (ZH)**: 高通量低成本的明场显微镜活细胞图像分割 

**Authors**: Surajit Das, Gourav Roy, Pavel Zun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14106)  

**Abstract**: Live cell culture is crucial in biomedical studies for analyzing cell properties and dynamics in vitro. This study focuses on segmenting unstained live cells imaged with bright-field microscopy. While many segmentation approaches exist for microscopic images, none consistently address the challenges of bright-field live-cell imaging with high throughput, where temporal phenotype changes, low contrast, noise, and motion-induced blur from cellular movement remain major obstacles. We developed a low-cost CNN-based pipeline incorporating comparative analysis of frozen encoders within a unified U-Net architecture enhanced with attention mechanisms, instance-aware systems, adaptive loss functions, hard instance retraining, dynamic learning rates, progressive mechanisms to mitigate overfitting, and an ensemble technique. The model was validated on a public dataset featuring diverse live cell variants, showing consistent competitiveness with state-of-the-art methods, achieving 93% test accuracy and an average F1-score of 89% (std. 0.07) on low-contrast, noisy, and blurry images. Notably, the model was trained primarily on bright-field images with limited exposure to phase-contrast microscopy (<10%), yet it generalized effectively to the phase-contrast LIVECell dataset, demonstrating modality, robustness and strong performance. This highlights its potential for real-world laboratory deployment across imaging conditions. The model requires minimal compute power and is adaptable using basic deep learning setups such as Google Colab, making it practical for training on other cell variants. Our pipeline outperforms existing methods in robustness and precision for bright-field microscopy segmentation. The code and dataset are available for reproducibility 

**Abstract (ZH)**: 活细胞培养是生物医药研究中分析细胞特性与动力学的重要手段。本研究聚焦于通过明场显微镜成像的未染色活细胞分割。尽管存在多种针对显微镜图像的分割方法，但在高速处理、缺乏染色、对比度低、噪声干扰以及细胞运动导致的模糊等问题上，现有方法尚未能够一致解决。我们开发了一种低成本的基于卷积神经网络（CNN）的分割流水线，整合了在统一U-Net架构内的冻干编码器比较分析，并结合了注意力机制、实例感知系统、自适应损失函数、困难实例重训、动态学习率以及渐进机制以减轻过拟合，以及集成技术。该模型在公共数据集上进行了验证，该数据集包含多种活细胞变体，显示出与最新方法的一致竞争力，基于低对比度、噪声和模糊图像，测试准确率达到93%，平均F1分数为89%（标准差0.07）。值得注意的是，该模型主要基于明场图像进行训练，并且很少接触相差显微镜(<10%)，但它在相差显微镜LIVECell数据集上表现出良好的泛化能力，展示了其模态适应性、稳健性和出色表现。这突显了其在各种成像条件下实际实验室部署的潜力。该模型所需的计算能力较低，并可使用基本的深度学习设置如Google Colab进行调整，使之适用于其他细胞类型的训练。与现有方法相比，我们的流水线在明场显微镜分割的稳健性和精确性方面表现出色。代码和数据集可供复现使用。 

---
# You Don't Know Until You Click:Automated GUI Testing for Production-Ready Software Evaluation 

**Title (ZH)**: 直到点击才知道：面向生产级软件评估的自动GUI测试 

**Authors**: Yutong Bian, Xianhao Lin, Yupeng Xie, Tianyang Liu, Mingchen Zhuge, Siyuan Lu, Haoming Tang, Jinlin Wang, Jiayi Zhang, Jiaqi Chen, Xiangru Tang, Yongxin Ni, Sirui Hong, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14104)  

**Abstract**: Large Language Models (LLMs) and code agents in software development are rapidly evolving from generating isolated code snippets to producing full-fledged software applications with graphical interfaces, interactive logic, and dynamic behaviors. However, current benchmarks fall short in evaluating such production-ready software, as they often rely on static checks or binary pass/fail scripts, failing to capture the interactive behaviors and runtime dynamics that define real-world usability - qualities that only emerge when an application is actively used. This is the blind spot of current evaluation: you don't know if an app works until you click through it, interact with it, and observe how it responds. To bridge this gap, we introduce RealDevWorld, a novel evaluation framework for automated end-to-end assessment of LLMs' ability to generate production-ready repositories from scratch. It features two key components: (1) RealDevBench, a diverse collection of 194 open-ended software engineering tasks across multiple domains, incorporating multimodal elements to reflect real-world complexity; and (2) AppEvalPilot, a new agent-as-a-judge evaluation system that simulates realistic, GUI-based user interactions to automatically and holistically assess software functional correctness, visual fidelity, and runtime behavior. The framework delivers fine-grained, task-specific diagnostic feedback, supporting nuanced evaluation beyond simple success/failure judgments. Empirical results show that RealDevWorld delivers effective, automatic, and human-aligned evaluations, achieving an accuracy of 0.92 and a correlation of 0.85 with expert human assessments, while significantly reducing the reliance on manual review. This enables scalable, human-aligned assessment of production-level software generated by LLMs. Our code is available on GitHub. 

**Abstract (ZH)**: 大型语言模型（LLMs）和代码代理在软件开发中的快速进化从生成孤立的代码片段转变为生产全程软件应用，这些应用具有图形界面、交互逻辑和动态行为。然而，当前的基准测试在评估此类可部署软件方面存在不足，因为它们往往依赖静态检查或二进制通过/失败脚本，无法捕捉由实际使用定义的互动行为和运行时动态。这就是当前评估的盲点：你不知道一个应用能否正常工作，除非亲自点击测试、与之交互并观察其反应。为弥补这一差距，我们提出了RealDevWorld，这是一种新颖的评估框架，用于自动化评估LLMs从零开始生成可部署软件仓库的能力。该框架包含两个关键组件：（1）RealDevBench，包含跨多个领域的194个开放性软件工程任务的多样化集合，并结合多模态元素以反映现实世界的复杂性；（2）AppEvalPilot，一种新型代理作为评委评估系统，模拟真实的、基于图形界面的用户交互，以自动和全面评估软件的功能正确性、视觉保真度和运行时行为。该框架提供了精细粒度的任务特定诊断反馈，支持超越简单成功/失败判断的深入评估。实验证据表明，RealDevWorld提供了有效、自动且与专家评估一致的评估，准确率达到0.92，相关性达到0.85，同时显著减少了对人工审核的依赖，从而能够对由LLMs生成的生产级软件进行可扩展的、与人类一致的评估。我们的代码已在GitHub上开源。 

---
# Implicit Hypergraph Neural Network 

**Title (ZH)**: 隐式超图神经网络 

**Authors**: Akash Choudhuri, Yongjian Zhong, Bijaya Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2508.14101)  

**Abstract**: Hypergraphs offer a generalized framework for capturing high-order relationships between entities and have been widely applied in various domains, including healthcare, social networks, and bioinformatics. Hypergraph neural networks, which rely on message-passing between nodes over hyperedges to learn latent representations, have emerged as the method of choice for predictive tasks in many of these domains. These approaches typically perform only a small number of message-passing rounds to learn the representations, which they then utilize for predictions. The small number of message-passing rounds comes at a cost, as the representations only capture local information and forego long-range high-order dependencies. However, as we demonstrate, blindly increasing the message-passing rounds to capture long-range dependency also degrades the performance of hyper-graph neural networks.
Recent works have demonstrated that implicit graph neural networks capture long-range dependencies in standard graphs while maintaining performance. Despite their popularity, prior work has not studied long-range dependency issues on hypergraph neural networks. Here, we first demonstrate that existing hypergraph neural networks lose predictive power when aggregating more information to capture long-range dependency. We then propose Implicit Hypergraph Neural Network (IHNN), a novel framework that jointly learns fixed-point representations for both nodes and hyperedges in an end-to-end manner to alleviate this issue. Leveraging implicit differentiation, we introduce a tractable projected gradient descent approach to train the model efficiently. Extensive experiments on real-world hypergraphs for node classification demonstrate that IHNN outperforms the closest prior works in most settings, establishing a new state-of-the-art in hypergraph learning. 

**Abstract (ZH)**: Hyper图提供了捕获实体之间高阶关系的通用框架，并在医疗保健、社交网络和生物信息学等多个领域得到了广泛的应用。依赖于超边上的节点间消息传递来学习潜在表示的超图神经网络，已成为这些领域中许多预测任务的首选方法。这些方法通常只进行少量的消息传递轮次来学习表示，然后使用这些表示进行预测。虽然少量的消息传递轮次可以捕获局部信息，但会忽略长距离的高阶依赖关系。然而，我们证明，盲目增加消息传递轮次以捕获长距离依赖也会损害超图神经网络的性能。最近的研究表明，隐式图神经网络可以在保持性能的同时捕获标准图中的长距离依赖关系。尽管隐式图神经网络广受欢迎，但先前的工作尚未研究超图神经网络中的长距离依赖问题。我们首先证明，现有的超图神经网络在聚合更多信息以捕获长距离依赖时会失去预测能力。然后，我们提出了一个新的隐式超图神经网络框架(IHNN)，该框架以端到端的方式联合学习节点和超边的固定点表示，以解决这一问题。利用隐式微分，我们引入了一种可计算的投影梯度下降方法，以高效地训练模型。实验结果表明，在实际超图上的节点分类任务中，IHNN在大多数情况下优于最近的工作，建立了超图学习的新前沿。 

---
# Domain Translation of a Soft Robotic Arm using Conditional Cycle Generative Adversarial Network 

**Title (ZH)**: 基于条件周期生成对抗网络的软质机械臂域翻译 

**Authors**: Nilay Kushawaha, Carlo Alessi, Lorenzo Fruzzetti, Egidio Falotico  

**Link**: [PDF](https://arxiv.org/pdf/2508.14100)  

**Abstract**: Deep learning provides a powerful method for modeling the dynamics of soft robots, offering advantages over traditional analytical approaches that require precise knowledge of the robot's structure, material properties, and other physical characteristics. Given the inherent complexity and non-linearity of these systems, extracting such details can be challenging. The mappings learned in one domain cannot be directly transferred to another domain with different physical properties. This challenge is particularly relevant for soft robots, as their materials gradually degrade over time. In this paper, we introduce a domain translation framework based on a conditional cycle generative adversarial network (CCGAN) to enable knowledge transfer from a source domain to a target domain. Specifically, we employ a dynamic learning approach to adapt a pose controller trained in a standard simulation environment to a domain with tenfold increased viscosity. Our model learns from input pressure signals conditioned on corresponding end-effector positions and orientations in both domains. We evaluate our approach through trajectory-tracking experiments across five distinct shapes and further assess its robustness under noise perturbations and periodicity tests. The results demonstrate that CCGAN-GP effectively facilitates cross-domain skill transfer, paving the way for more adaptable and generalizable soft robotic controllers. 

**Abstract (ZH)**: 基于条件周期生成对抗网络的领域转移框架：软体机器人跨域技能迁移 

---
# No More Marching: Learning Humanoid Locomotion for Short-Range SE(2) Targets 

**Title (ZH)**: 不再僵硬行进：学习面向短距离SE(2)目标的人形机器人运动学 

**Authors**: Pranay Dugar, Mohitvishnu S. Gadde, Jonah Siekmann, Yesh Godse, Aayam Shrestha, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2508.14098)  

**Abstract**: Humanoids operating in real-world workspaces must frequently execute task-driven, short-range movements to SE(2) target poses. To be practical, these transitions must be fast, robust, and energy efficient. While learning-based locomotion has made significant progress, most existing methods optimize for velocity-tracking rather than direct pose reaching, resulting in inefficient, marching-style behavior when applied to short-range tasks. In this work, we develop a reinforcement learning approach that directly optimizes humanoid locomotion for SE(2) targets. Central to this approach is a new constellation-based reward function that encourages natural and efficient target-oriented movement. To evaluate performance, we introduce a benchmarking framework that measures energy consumption, time-to-target, and footstep count on a distribution of SE(2) goals. Our results show that the proposed approach consistently outperforms standard methods and enables successful transfer from simulation to hardware, highlighting the importance of targeted reward design for practical short-range humanoid locomotion. 

**Abstract (ZH)**: 真实世界工作空间中的人形机器人必须频繁执行任务驱动的短距离SE(2)目标位姿移动。为了实用，这些转换必须快速、稳健且能效高。尽管基于学习的移动已取得显著进展，但现有大多数方法优化的是速度跟踪而非直接位姿达到，导致应用于短距离任务时行为效率低下且步伐僵硬。在这项工作中，我们开发了一种强化学习方法，直接优化人形机器人移动以达到SE(2)目标。该方法的核心是一种新的星座基奖励函数，鼓励自然且高效的目标导向移动。为了评估性能，我们引入了一种基准框架，该框架在一系列SE(2)目标上衡量能耗、达到目标时间和步数。实验结果表明，所提出的方法始终优于标准方法，并成功实现了从模拟到硬件的迁移，突显了针对实际短距离人形移动的奖励设计的重要性。 

---
# Non-Dissipative Graph Propagation for Non-Local Community Detection 

**Title (ZH)**: 非耗散图传播用于非局部社区检测 

**Authors**: William Leeney, Alessio Gravina, Davide Bacciu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14097)  

**Abstract**: Community detection in graphs aims to cluster nodes into meaningful groups, a task particularly challenging in heterophilic graphs, where nodes sharing similarities and membership to the same community are typically distantly connected. This is particularly evident when this task is tackled by graph neural networks, since they rely on an inherently local message passing scheme to learn the node representations that serve to cluster nodes into communities. In this work, we argue that the ability to propagate long-range information during message passing is key to effectively perform community detection in heterophilic graphs. To this end, we introduce the Unsupervised Antisymmetric Graph Neural Network (uAGNN), a novel unsupervised community detection approach leveraging non-dissipative dynamical systems to ensure stability and to propagate long-range information effectively. By employing antisymmetric weight matrices, uAGNN captures both local and global graph structures, overcoming the limitations posed by heterophilic scenarios. Extensive experiments across ten datasets demonstrate uAGNN's superior performance in high and medium heterophilic settings, where traditional methods fail to exploit long-range dependencies. These results highlight uAGNN's potential as a powerful tool for unsupervised community detection in diverse graph environments. 

**Abstract (ZH)**: 在图中检测社区旨在将节点聚类到有意义的小组中，而在异ophilic图中，这一任务尤为具有挑战性，因为共享相似性和属于同一社区的节点通常相差甚远。特别是在图神经网络处理此任务时，由于它们依赖于固有的局部消息传递方案来学习用于聚类节点的节点表示，因此这一挑战尤为明显。在本文中，我们认为在消息传递过程中传播长程信息的能力对于有效执行异ophilic图中的社区检测至关重要。为此，我们引入了基于非耗散动力系统的无监督反对称图神经网络（uAGNN），这是一种新颖的无监督社区检测方法，能够确保稳定性和有效地传播长程信息。通过使用反反对称权重矩阵，uAGNN捕捉到了局部和全局图结构，克服了异ophilic场景带来的限制。在十个数据集上的广泛实验表明，uAGNN在高和中等异ophilic设置中表现出色，而在传统方法无法利用长程依赖性的情况下，uAGNN表现出优越性能。这些结果突显了uAGNN作为多种图环境中无监督社区检测强大工具的潜力。 

---
# Hard Examples Are All You Need: Maximizing GRPO Post-Training Under Annotation Budgets 

**Title (ZH)**: 硬样本即所需：在标注预算下的GRPO训练后优化最大化 

**Authors**: Benjamin Pikus, Pratyush Ranjan Tiwari, Burton Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.14094)  

**Abstract**: Collecting high-quality training examples for language model fine-tuning is expensive, with practical budgets limiting the amount of data that can be procured. We investigate a critical question for resource-constrained alignment: under a fixed acquisition budget, should practitioners prioritize examples that are easy, medium, hard, or of random difficulty? We study Group Relative Policy Optimization (GRPO) fine-tuning across different model sizes and families, comparing four subset selection policies chosen from the same unlabeled pool using base-model difficulty estimates obtained via multi-sample evaluation. Our experiments reveal that training on the hardest examples yields the largest performance gains, up to 47%, while training on easy examples yield the smallest gains. Analysis reveals that this effect arises from harder examples providing more learnable opportunities during GRPO training. These findings provide practical guidance for budget-constrained post-training: prioritizing hard examples yields substantial performance gains on reasoning tasks when using GRPO. 

**Abstract (ZH)**: 在资源受限的对齐中，在固定获取预算下，语言模型微调时应优先选择难度最容易、中等、困难还是随机难度的训练示例？我们的研究发现，在最困难的示例上进行训练可获得最大的性能提升，高达47%，而在最容易的示例上进行训练则获得最小的性能提升。分析表明，这种效果源于最难的示例在GRPO训练过程中提供了更多的可学习机会。这些发现为预算受限的后续训练提供了实用指导：在使用GRPO时，优先选择困难的示例可以在推理任务中获得显著的性能提升。 

---
# Logical Expressivity and Explanations for Monotonic GNNs with Scoring Functions 

**Title (ZH)**: 基于评分函数的单调GNN的逻辑表达能力和解释性 

**Authors**: Matthew Morris, David J. Tena Cucala, Bernardo Cuenca Grau  

**Link**: [PDF](https://arxiv.org/pdf/2508.14091)  

**Abstract**: Graph neural networks (GNNs) are often used for the task of link prediction: predicting missing binary facts in knowledge graphs (KGs). To address the lack of explainability of GNNs on KGs, recent works extract Datalog rules from GNNs with provable correspondence guarantees. The extracted rules can be used to explain the GNN's predictions; furthermore, they can help characterise the expressive power of various GNN models. However, these works address only a form of link prediction based on a restricted, low-expressivity graph encoding/decoding method. In this paper, we consider a more general and popular approach for link prediction where a scoring function is used to decode the GNN output into fact predictions. We show how GNNs and scoring functions can be adapted to be monotonic, use the monotonicity to extract sound rules for explaining predictions, and leverage existing results about the kind of rules that scoring functions can capture. We also define procedures for obtaining equivalent Datalog programs for certain classes of monotonic GNNs with scoring functions. Our experiments show that, on link prediction benchmarks, monotonic GNNs and scoring functions perform well in practice and yield many sound rules. 

**Abstract (ZH)**: 图神经网络（GNNs）常用于链接预测任务：在知识图谱（KGs）中预测缺失的二元事实。为了应对GNNs在KGs上的可解释性不足问题，近期工作通过具有可证明对应保证的方式从GNNs中提取Datalog规则。提取出的规则可以用于解释GNN的预测；此外，它们还可以帮助表征各种GNN模型的表达能力。然而，这些工作仅关注一种受限的低表达能力图编码/解码方法下的链接预测形式。在本文中，我们考虑了一个更通用且流行的方法，该方法使用评分函数将GNN输出解码为事实预测。我们展示了如何对GNNs和评分函数进行调整以使其单调，并利用单调性抽取用于解释预测的可靠规则，同时借鉴评分函数能够捕获的规则类型的相关现有结果。我们还定义了获得特定单调类GNN与评分函数等价Datalog程序的过程。我们的实验表明，在链接预测基准测试中，单调GNNs和评分函数在实践中表现良好，并生成了大量可靠的规则。 

---
# DLLMQuant: Quantizing Diffusion-based Large Language Models 

**Title (ZH)**: DLLMQuant: 基于扩散的大语言模型量化 

**Authors**: Chen Xu, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14090)  

**Abstract**: Diffusion-based large language models (DLLMs) have shown promise for non-autoregressive text generation, but their deployment is constrained by large model sizes and heavy computational costs. Post-training quantization (PTQ), a widely used method for compressing and accelerating Large Language Models (LLMs), suffers from severe accuracy degradation and reduced generalization performance when directly applied to DLLMs (e.g., AWQ suffers a 16% accuracy drop on LLADA under W4A4). This paper explores how DLLMs' key mechanisms - dynamic masking, iterative generation, bidirectional attention - clash with quantization. We identify three core issues: 1) Iterative generation and dynamic masking ratios lead to distinct token distributions across decoding steps, which are not adequately captured by existing PTQ calibration methods; 2) Quantization errors are accumulated and amplified progressively during iteration in DLLMs, causing quantized models to perform worse as decoding steps progress; 3) Unmasked tokens stabilize while masked remain probabilistic, making overall feature distribution incompatible with existing PTQ methods. To address these issues, we propose DLLMQuant, a PTQ framework tailored for DLLMs, which incorporates three novel techniques: 1) Temporal-Mask Adaptive Sampling (TMAS), a calibration method that accounts for both time and mask factors, with the capacity to capture distributions across timesteps. 2) Interaction-Aware Activation Quantization (IA-AQ), which utilizes bidirectional attention's interaction signals to dynamically allocate quantization resources. 3) Certainty-Guided Quantization (CGQ), which integrates mask status and token scores as key weighting criteria into error compensation, making weight quantization more suitable for DLLMs. Experiments show that DLLMQuant achieves significant performance gains while enhancing efficiency. 

**Abstract (ZH)**: 基于扩散的大语言模型（DLLMs）在非自回归文本生成方面表现出潜力，但其部署受限于庞大的模型规模和高昂的计算成本。针对大语言模型（LLMs）压缩和加速的后训练量化（PTQ）方法在直接应用于DLLMs（例如，AWQ在LLADA下的准确率下降了16%）时，会导致严重的准确率下降和泛化性能降低。本文探讨了DLLMs的关键机制——动态掩码、迭代生成、双向注意——与量化之间的冲突。我们识别出三个核心问题：1）迭代生成和动态掩码比例导致解码步骤中不同的令牌分布，现有的PTQ校准方法难以充分捕捉；2）在DLLMs中，量化误差在迭代过程中逐渐累积和放大，导致量化模型随着解码步骤的增加而表现更差；3）未掩码的令牌趋于稳定，而掩码的则保持概率性，使得整体特征分布与现有的PTQ方法不兼容。为解决这些问题，我们提出了DLLMQuant，这是一种针对DLLMs的PTQ框架，结合了三种新颖的技术：1）时间-掩码自适应采样（TMAS），一种同时考虑时间和掩码因素的校准方法，能够捕捉跨时间步的分布；2）交互感知激活量化（IA-AQ），利用双向注意的交互信号动态分配量化资源；3）基于确定性量化（CGQ），将掩码状态和令牌得分作为关键权重指标纳入误差补偿，使权重量化更适合DLLMs。实验表明，DLLMQuant在提高性能的同时也增强了效率。 

---
# CoBAD: Modeling Collective Behaviors for Human Mobility Anomaly Detection 

**Title (ZH)**: CoBAD: 模型集体行为以人类移动性异常检测 

**Authors**: Haomin Wen, Shurui Cao, Leman Akoglu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14088)  

**Abstract**: Detecting anomalies in human mobility is essential for applications such as public safety and urban planning. While traditional anomaly detection methods primarily focus on individual movement patterns (e.g., a child should stay at home at night), collective anomaly detection aims to identify irregularities in collective mobility behaviors across individuals (e.g., a child is at home alone while the parents are elsewhere) and remains an underexplored challenge. Unlike individual anomalies, collective anomalies require modeling spatiotemporal dependencies between individuals, introducing additional complexity. To address this gap, we propose CoBAD, a novel model designed to capture Collective Behaviors for human mobility Anomaly Detection. We first formulate the problem as unsupervised learning over Collective Event Sequences (CES) with a co-occurrence event graph, where CES represents the event sequences of related individuals. CoBAD then employs a two-stage attention mechanism to model both the individual mobility patterns and the interactions across multiple individuals. Pre-trained on large-scale collective behavior data through masked event and link reconstruction tasks, CoBAD is able to detect two types of collective anomalies: unexpected co-occurrence anomalies and absence anomalies, the latter of which has been largely overlooked in prior work. Extensive experiments on large-scale mobility datasets demonstrate that CoBAD significantly outperforms existing anomaly detection baselines, achieving an improvement of 13%-18% in AUCROC and 19%-70% in AUCPR. All source code is available at this https URL. 

**Abstract (ZH)**: 集体行为检测在人类移动中的异常对于公共安全和城市规划等应用至关重要。与传统的基于个体移动模式的异常检测方法不同，集体异常检测旨在识别跨个体的集体移动行为的不规律性（例如，一个孩子独自在家而父母在其他地方），这仍是一个未被充分探索的挑战。CoBAD提出了一种新型模型，旨在捕捉集体行为以进行人类移动异常检测。该模型首先通过共现事件图将问题形式化为集体事件序列（CES）的无监督学习，其中CES表示相关个体的事件序列。随后，CoBAD采用两阶段注意力机制来建模个体移动模式及其跨多个个体的交互。通过大规模集体行为数据的掩蔽事件和链接重建任务进行预训练，CoBAD能够检测两种类型的集体异常：意外共现异常和缺位异常，其中后者在以往工作中被严重忽视。在大规模移动数据集上的广泛实验表明，CoBAD显著优于现有异常检测基线，AUCROC提升13%-18%，AUCPR提升19%-70%。所有源代码可在以下链接获取。 

---
# FM4NPP: A Scaling Foundation Model for Nuclear and Particle Physics 

**Title (ZH)**: FM4NPP：核物理与粒子物理的扩展基础模型 

**Authors**: David Park, Shuhang Li, Yi Huang, Xihaier Luo, Haiwang Yu, Yeonju Go, Christopher Pinkenburg, Yuewei Lin, Shinjae Yoo, Joseph Osborn, Jin Huang, Yihui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.14087)  

**Abstract**: Large language models have revolutionized artificial intelligence by enabling large, generalizable models trained through self-supervision. This paradigm has inspired the development of scientific foundation models (FMs). However, applying this capability to experimental particle physics is challenging due to the sparse, spatially distributed nature of detector data, which differs dramatically from natural language. This work addresses if an FM for particle physics can scale and generalize across diverse tasks. We introduce a new dataset with more than 11 million particle collision events and a suite of downstream tasks and labeled data for evaluation. We propose a novel self-supervised training method for detector data and demonstrate its neural scalability with models that feature up to 188 million parameters. With frozen weights and task-specific adapters, this FM consistently outperforms baseline models across all downstream tasks. The performance also exhibits robust data-efficient adaptation. Further analysis reveals that the representations extracted by the FM are task-agnostic but can be specialized via a single linear mapping for different downstream tasks. 

**Abstract (ZH)**: 大型语言模型通过自监督训练实现了人工智能的革命，从而开发了通用的大规模模型。这一范式激发了科学基础模型（FMs）的发展。然而，将这一能力应用于实验粒子物理学存在挑战，因为检测器数据稀疏且空间分布，与自然语言截然不同。本文探讨了粒子物理学中的FM是否能够跨多种任务进行扩展和通用化。我们介绍了包含超过1100万次粒子碰撞事件的新数据集，并提出了一套下游任务和标注数据以供评估。我们提出了一种新颖的自监督训练方法用于检测器数据，并用参数量高达1.88亿的模型演示了其神经可扩展性。通过固定权重和任务特定适配器，该FM在所有下游任务上均优于基线模型，并展现出稳健的数据效率适应性。进一步的分析表明，FM提取的表示在任务无关，但可以通过单一的线性映射进行专门化以适应不同的下游任务。 

---
# GeoMAE: Masking Representation Learning for Spatio-Temporal Graph Forecasting with Missing Values 

**Title (ZH)**: GeoMAE：处理缺失值的时空图预测掩蔽表示学习 

**Authors**: Songyu Ke, Chenyu Wu, Yuxuan Liang, Xiuwen Yi, Yanping Sun, Junbo Zhang, Yu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.14083)  

**Abstract**: Accurate acquisition of crowd flow at Points of Interest (POIs) is pivotal for effective traffic management, public service, and urban planning. Despite this importance, due to the limitations of urban sensing techniques, the data quality from most sources is inadequate for monitoring crowd flow at each POI. This renders the inference of accurate crowd flow from low-quality data a critical and challenging task. The complexity is heightened by three key factors: 1) \emph{The scarcity and rarity of labeled data}, 2) \emph{The intricate spatio-temporal dependencies among POIs}, and 3) \emph{The myriad correlations between precise crowd flow and GPS reports}.
To address these challenges, we recast the crowd flow inference problem as a self-supervised attributed graph representation learning task and introduce a novel \underline{C}ontrastive \underline{S}elf-learning framework for \underline{S}patio-\underline{T}emporal data (\model). Our approach initiates with the construction of a spatial adjacency graph founded on the POIs and their respective distances. We then employ a contrastive learning technique to exploit large volumes of unlabeled spatio-temporal data. We adopt a swapped prediction approach to anticipate the representation of the target subgraph from similar instances. Following the pre-training phase, the model is fine-tuned with accurate crowd flow data. Our experiments, conducted on two real-world datasets, demonstrate that the \model pre-trained on extensive noisy data consistently outperforms models trained from scratch. 

**Abstract (ZH)**: 准确获取兴趣点处的人流流是一个有效的交通管理、公共服务和城市规划的关键。尽管如此，由于城市感知技术的限制，大多数数据源的质量不足以监测每个兴趣点的人流。这使得从低质量数据推断准确的人流成为一个关键而具有挑战性的任务。复杂性由以下三个关键因素加剧：1) 标签数据的稀缺性和稀有性，2) 兴趣点之间的复杂时空相关性，3) 精确的人流与GPS报告之间的多元关联。为了解决这些挑战，我们重新定义了人流推断问题为自监督属性图表示学习任务，并引入了一种新颖的时空数据对比自学习框架（\model）。该方法首先基于兴趣点及其距离构建空间邻接图，然后利用对比学习技术利用大量未标记的时空数据。采用交换预测方法，从相似实例预测目标子图的表示。在预训练阶段之后，模型使用准确的人流数据进行微调。我们在两个真实世界数据集上的实验表明，使用大量嘈杂数据预训练的\model相较于从零开始训练的模型表现更优。 

---
# Label Smoothing is a Pragmatic Information Bottleneck 

**Title (ZH)**: 标签平滑是实用的信息瓶颈 

**Authors**: Sota Kudo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14077)  

**Abstract**: This study revisits label smoothing via a form of information bottleneck. Under the assumption of sufficient model flexibility and no conflicting labels for the same input, we theoretically and experimentally demonstrate that the model output obtained through label smoothing explores the optimal solution of the information bottleneck. Based on this, label smoothing can be interpreted as a practical approach to the information bottleneck, enabling simple implementation. As an information bottleneck method, we experimentally show that label smoothing also exhibits the property of being insensitive to factors that do not contain information about the target, or to factors that provide no additional information about it when conditioned on another variable. 

**Abstract (ZH)**: 本文通过信息瓶颈的一种形式重新审视标签平滑。在模型足够灵活且输入不存在冲突标签的前提下，我们从理论上和实验上证明，通过标签平滑得到的模型输出探索了信息瓶颈的最优解。基于此，标签平滑可以被解释为信息瓶颈的一种实用方法，使其易于实现。作为信息瓶颈方法，实验表明标签平滑对不包含目标信息的因素或在给定另一个变量时提供无额外信息的因素表现出不敏感的特性。 

---
# PersRM-R1: Enhance Personalized Reward Modeling with Reinforcement Learning 

**Title (ZH)**: PersRM-R1: 通过强化学习提升个性化奖励建模 

**Authors**: Mengdi Li, Guanqiao Chen, Xufeng Zhao, Haochen Wen, Shu Yang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14076)  

**Abstract**: Reward models (RMs), which are central to existing post-training methods, aim to align LLM outputs with human values by providing feedback signals during fine-tuning. However, existing RMs struggle to capture nuanced, user-specific preferences, especially under limited data and across diverse domains. Thus, we introduce PersRM-R1, the first reasoning-based reward modeling framework specifically designed to identify and represent personal factors from only one or a few personal exemplars. To address challenges including limited data availability and the requirement for robust generalization, our approach combines synthetic data generation with a two-stage training pipeline consisting of supervised fine-tuning followed by reinforcement fine-tuning. Experimental results demonstrate that PersRM-R1 outperforms existing models of similar size and matches the performance of much larger models in both accuracy and generalizability, paving the way for more effective personalized LLMs. 

**Abstract (ZH)**: 基于推理的个性化奖励模型（PersRM-R1）：仅通过少量个人示例识别和表示个性化因素 

---
# Explainable Graph Spectral Clustering For Text Embeddings 

**Title (ZH)**: 可解释的图谱聚类用于文本嵌入 

**Authors**: Mieczysław A. Kłopotek, Sławomir T. Wierzchoń, Bartłomiej Starosta, Piotr Borkowski, Dariusz Czerski, Eryk Laskowski  

**Link**: [PDF](https://arxiv.org/pdf/2508.14075)  

**Abstract**: In a previous paper, we proposed an introduction to the explainability of Graph Spectral Clustering results for textual documents, given that document similarity is computed as cosine similarity in term vector space.
In this paper, we generalize this idea by considering other embeddings of documents, in particular, based on the GloVe embedding idea. 

**Abstract (ZH)**: 在前一篇论文中，我们提出了一种关于文本文档的图谱聚类结果可解释性的介绍，基于词向量空间中文档相似性通过余弦相似性计算。
在本文中，我们通过考虑基于GloVe嵌入理念的其他文档嵌入，进一步推广了这一想法。 

---
# GEPD:GAN-Enhanced Generalizable Model for EEG-Based Detection of Parkinson's Disease 

**Title (ZH)**: GAN增强的用于基于EEG的帕金森病检测的可迁移模型 

**Authors**: Qian Zhang, Ruilin Zhang, Biaokai Zhu, Xun Han, Jun Xiao, Yifan Liu, Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14074)  

**Abstract**: Electroencephalography has been established as an effective method for detecting Parkinson's disease, typically diagnosed this http URL Parkinson's disease detection methods have shown significant success within individual datasets, however, the variability in detection methods across different EEG datasets and the small size of each dataset pose challenges for training a generalizable model for cross-dataset scenarios. To address these issues, this paper proposes a GAN-enhanced generalizable model, named GEPD, specifically for EEG-based cross-dataset classification of Parkinson's this http URL, we design a generative network that creates fusion EEG data by controlling the distribution similarity between generated data and real this http URL addition, an EEG signal quality assessment model is designed to ensure the quality of generated data this http URL, we design a classification network that utilizes a combination of multiple convolutional neural networks to effectively capture the time-frequency characteristics of EEG signals, while maintaining a generalizable structure and ensuring easy this http URL work is dedicated to utilizing intelligent methods to study pathological manifestations, aiming to facilitate the diagnosis and monitoring of neurological this http URL evaluation results demonstrate that our model performs comparably to state-of-the-art models in cross-dataset settings, achieving an accuracy of 84.3% and an F1-score of 84.0%, showcasing the generalizability of the proposed model. 

**Abstract (ZH)**: 基于EEG跨数据集通用模型的GAN增强方法：GEPD在帕金森病检测中的应用 

---
# MCLPD:Multi-view Contrastive Learning for EEG-based PD Detection Across Datasets 

**Title (ZH)**: MCLPD：基于多视图对比学习的跨数据集的EEG Parkinson's Disease检测 

**Authors**: Qian Zhanga, Ruilin Zhang, Jun Xiao, Yifan Liu, Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14073)  

**Abstract**: Electroencephalography has been validated as an effective technique for detecting Parkinson's disease,particularly in its early this http URL,the high cost of EEG data annotation often results in limited dataset size and considerable discrepancies across datasets,including differences in acquisition protocols and subject demographics,significantly hinder the robustness and generalizability of models in cross-dataset detection this http URL address such challenges,this paper proposes a semi-supervised learning framework named MCLPD,which integrates multi-view contrastive pre-training with lightweight supervised fine-tuning to enhance cross-dataset PD detection this http URL pre-training,MCLPD uses self-supervised learning on the unlabeled UNM this http URL build contrastive pairs,it applies dual augmentations in both time and frequency domains,which enrich the data and naturally fuse time-frequency this http URL the fine-tuning phase,only a small proportion of labeled data from another two datasets (UI and UC)is used for supervised this http URL results show that MCLPD achieves F1 scores of 0.91 on UI and 0.81 on UC using only 1%of labeled data,which further improve to 0.97 and 0.87,respectively,when 5%of labeled data is this http URL to existing methods,MCLPD substantially improves cross-dataset generalization while reducing the dependency on labeled data,demonstrating the effectiveness of the proposed framework. 

**Abstract (ZH)**: 基于多视图对比预训练的半监督Parkinson's病检测框架 

---
# Edge-Selector Model Applied for Local Search Neighborhood for Solving Vehicle Routing Problems 

**Title (ZH)**: 基于边缘选择模型的局部搜索邻域在解决车辆路线问题中的应用 

**Authors**: Bachtiar Herdianto, Romain Billot, Flavien Lucas, Marc Sevaux, Daniele Vigo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14071)  

**Abstract**: This research proposes a hybrid Machine Learning and metaheuristic mechanism that is designed to solve Vehicle Routing Problems (VRPs). The main of our method is an edge solution selector model, which classifies solution edges to identify prohibited moves during the local search, hence guiding the search process within metaheuristic baselines. Two learning-based mechanisms are used to develop the edge selector: a simple tabular binary classifier and a Graph Neural Network (GNN). The tabular classifier employs Gradient Boosting Trees and Feedforward Neural Network as the baseline algorithms. Adjustments to the decision threshold are also applied to handle the class imbalance in the problem instance. An alternative mechanism employs the GNN to utilize graph structure for direct solution edge prediction, with the objective of guiding local search by predicting prohibited moves. These hybrid mechanisms are then applied in state-fo-the-art metaheuristic baselines. Our method demonstrates both scalability and generalizability, achieving performance improvements across different baseline metaheuristics, various problem sizes and variants, including the Capacitated Vehicle Routing Problem (CVRP) and CVRP with Time Windows (CVRPTW). Experimental evaluations on benchmark datasets up to 30,000 customer nodes, supported by pair-wise statistical analysis, verify the observed improvements. 

**Abstract (ZH)**: 一种混合机器学习和元启发式机制及其在车辆路线问题中的应用 

---
# Special-Character Adversarial Attacks on Open-Source Language Model 

**Title (ZH)**: 开源语言模型中的特殊字符对抗攻击 

**Authors**: Ephraiem Sarabamoun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14070)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across diverse natural language processing tasks, yet their vulnerability to character-level adversarial manipulations presents significant security challenges for real-world deployments. 

**Abstract (ZH)**: 大规模语言模型在多项自然语言处理任务中取得了显著性能，但对其字符级 adversarial 操纵的脆弱性为实际部署带来了重大安全挑战。 

---
# Revisit Choice Network for Synthesis and Technology Mapping 

**Title (ZH)**: 重访选择网络用于综合与技术映射 

**Authors**: Chen Chen, Jiaqi Yin, Cunxi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14068)  

**Abstract**: Choice network construction is a critical technique for alleviating structural bias issues in Boolean optimization, equivalence checking, and technology mapping. Previous works on lossless synthesis utilize independent optimization to generate multiple snapshots, and use simulation and SAT solvers to identify functionally equivalent nodes. These nodes are then merged into a subject graph with choice nodes. However, such methods often neglect the quality of these choices, raising the question of whether they truly contribute to effective technology mapping.
This paper introduces Cristal, a novel methodology and framework for constructing Boolean choice networks. Specifically, Cristal introduces a new flow of choice network-based synthesis and mapping, including representative logic cone search, structural mutation for generating diverse choice structures via equality saturation, and priority-ranking choice selection along with choice network construction and validation. Through these techniques, Cristal constructs fewer but higher-quality choices.
Our experimental results demonstrate that Cristal outperforms the state-of-the-art Boolean choice network construction implemented in ABC in the post-mapping stage, achieving average reductions of 3.85%/8.35% (area/delay) in delay-oriented mode, 0.11%/2.74% in area-oriented mode, and a 63.77% runtime reduction on large-scale cases across a diverse set of combinational circuits from the IWLS 2005, ISCAS'89, and EPFL benchmark suites. 

**Abstract (ZH)**: 选择网络构建是缓解布尔优化、等价检查和技术映射中结构偏见问题的关键技术。先前的无损综合工作利用独立优化生成多个快照，并使用仿真和SAT求解器来识别功能等价节点，然后将这些节点合并到主题图中。然而，这些方法往往忽略这些选择的质量，引发了它们是否真正有助于有效技术映射的疑问。

本文提出Cristal，一种用于构造布尔选择网络的新颖方法和技术框架。具体而言，Cristal引入了一种基于选择网络的综合和映射的新流程，包括代表性逻辑锥搜索、结构突变以通过等价饱和生成多样化的选择结构、优先级排名选择以及选择网络构建与验证。通过这些技术，Cristal constructs fewer but higher-quality choices。

实验结果表明，Cristal在ABC实现的无损选择网络构建后映射阶段表现更优，平均延时模式下的延时降低3.85%/8.35%（面积/延迟），面积模式下的面积降低0.11%/2.74%，并在大量来自IWLS 2005、ISCAS'89和EPFL基准套件的组合电路中实现了63.77%的运行时间减少。 

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
# Dual-Phase Playtime-guided Recommendation: Interest Intensity Exploration and Multimodal Random Walks 

**Title (ZH)**: 双阶段玩乐时间指导推荐：兴趣强度探索与多模态随机游走 

**Authors**: Jingmao Zhang, Zhiting Zhao, Yunqi Lin, Jianghong Ma, Tianjun Wei, Haijun Zhang, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14058)  

**Abstract**: The explosive growth of the video game industry has created an urgent need for recommendation systems that can scale with expanding catalogs and maintain user engagement. While prior work has explored accuracy and diversity in recommendations, existing models underutilize playtime, a rich behavioral signal unique to gaming platforms, and overlook the potential of multimodal information to enhance diversity. In this paper, we propose DP2Rec, a novel Dual-Phase Playtime-guided Recommendation model designed to jointly optimize accuracy and diversity. First, we introduce a playtime-guided interest intensity exploration module that separates strong and weak preferences via dual-beta modeling, enabling fine-grained user profiling and more accurate recommendations. Second, we present a playtime-guided multimodal random walks module that simulates player exploration using transitions guided by both playtime-derived interest similarity and multimodal semantic similarity. This mechanism preserves core preferences while promoting cross-category discovery through latent semantic associations and adaptive category balancing. Extensive experiments on a real-world game dataset show that DP2Rec outperforms existing methods in both recommendation accuracy and diversity. 

**Abstract (ZH)**: 视频游戏行业的爆炸性增长创造了对能够适应扩展目录并维持用户参与度的推荐系统的需求。虽然先前的工作已经探索了推荐的准确性和多样性，但现有的模型未能充分利用专属于游戏平台的玩时数据，并且忽视了多模态信息增强多样性的潜力。在本文中，我们提出了一种新的双重阶段基于玩时的推荐模型DP2Rec，旨在同时优化准确性和多样性。首先，我们引入了一个基于玩时的兴趣强度探索模块，通过双贝塔建模分离强偏好和弱偏好，实现细粒度的用户画像和更准确的推荐。其次，我们提出了一个基于玩时的多模态随机游走模块，通过结合由玩时衍生的兴趣相似性和多模态语义相似性引导的转换来模拟玩家探索。该机制在保持核心偏好的同时，通过潜在语义关联和适应性类别平衡促进跨类别发现。在现实游戏数据集上的广泛实验表明，DP2Rec 在推荐准确性和多样性方面均优于现有方法。 

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
# FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering 

**Title (ZH)**: FinAgentBench: 金融问答中代理检索的基准数据集 

**Authors**: Chanyeol Choi, Jihoon Kwon, Alejandro Lopez-Lira, Chaewoon Kim, Minjae Kim, Juneha Hwang, Jaeseon Ha, Hojun Choi, Suyeol Yun, Yongjin Kim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14052)  

**Abstract**: Accurate information retrieval (IR) is critical in the financial domain, where investors must identify relevant information from large collections of documents. Traditional IR methods-whether sparse or dense-often fall short in retrieval accuracy, as it requires not only capturing semantic similarity but also performing fine-grained reasoning over document structure and domain-specific knowledge. Recent advances in large language models (LLMs) have opened up new opportunities for retrieval with multi-step reasoning, where the model ranks passages through iterative reasoning about which information is most relevant to a given query. However, there exists no benchmark to evaluate such capabilities in the financial domain. To address this gap, we introduce FinAgentBench, the first large-scale benchmark for evaluating retrieval with multi-step reasoning in finance -- a setting we term agentic retrieval. The benchmark consists of 3,429 expert-annotated examples on S&P-100 listed firms and assesses whether LLM agents can (1) identify the most relevant document type among candidates, and (2) pinpoint the key passage within the selected document. Our evaluation framework explicitly separates these two reasoning steps to address context limitations. This design enables to provide a quantitative basis for understanding retrieval-centric LLM behavior in finance. We evaluate a suite of state-of-the-art models and further demonstrated how targeted fine-tuning can significantly improve agentic retrieval performance. Our benchmark provides a foundation for studying retrieval-centric LLM behavior in complex, domain-specific tasks for finance. We will release the dataset publicly upon acceptance of the paper and plan to expand and share dataset for the full S&P 500 and beyond. 

**Abstract (ZH)**: 金融领域的准确信息检索对于投资者从大量文档中识别相关信息至关重要。传统的稀疏或稠密信息检索方法在检索准确性上经常不够理想，因为它不仅要求捕获语义相似性，还需要进行文档结构和领域特定知识的精细推理。近期大型语言模型（LLMs）的进步为多步推理的信息检索提供了新的机会，在这种检索模式中，模型通过对哪些信息最相关于给定查询的迭代推理来排序段落。然而，在金融领域中尚无用于评估这些能力的基准。为了填补这一空白，我们提出了FinAgentBench，这是第一个评估金融领域中多步推理信息检索的大规模基准——我们称之为代理检索。基准数据集包含3,429个专家标注的样本，针对S&P-100上市公司的文档，评估LLM代理能否（1）在候选文档中识别最相关的文档类型，以及（2）定位选定文档中的关键段落。我们的评估框架明确地将这两个推理步骤区分开来，以解决上下文限制问题。该设计为理解金融领域中以检索为中心的LLM行为提供了定量基础。我们评估了一系列最先进的模型，并展示了针对性微调如何显著提高代理检索性能。基准数据集为研究金融领域复杂、特定领域的检索中心LLM行为提供了基础。论文被接收后，我们将公开发布该数据集，并计划扩展和共享全S&P 500及更广泛的数据集。 

---
# The Hidden Cost of Readability: How Code Formatting Silently Consumes Your LLM Budget 

**Title (ZH)**: 隐藏的可读性成本：代码格式化悄无声息地消耗你的LLM预算 

**Authors**: Dangfeng Pan, Zhensu Sun, Cenyuan Zhang, David Lo, Xiaoning Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.13666)  

**Abstract**: Source code is usually formatted with elements like indentation and newlines to improve readability for human developers. However, these visual aids do not seem to be beneficial for large language models (LLMs) in the same way since the code is processed as a linear sequence of tokens. Furthermore, these additional tokens can lead to increased computational costs and longer response times for LLMs. If such formatting elements are non-essential to LLMs, we can reduce such costs by removing them from the code. To figure out the role played by formatting elements, we conduct a comprehensive empirical study to evaluate the impact of code formatting on LLM performance and efficiency. Through large-scale experiments on Fill-in-the-Middle Code Completion tasks across four programming languages (Java, Python, C++, C\#) and ten LLMs-including both commercial and open-source models-we systematically analyze token count and performance when formatting elements are removed. Key findings indicate that LLMs can maintain performance across formatted code and unformatted code, achieving an average input token reduction of 24.5\% with negligible output token reductions. This makes code format removal a practical optimization strategy for improving LLM efficiency. Further exploration reveals that both prompting and fine-tuning LLMs can lead to significant reductions (up to 36.1\%) in output code length without compromising correctness. To facilitate practical applications, we develop a bidirectional code transformation tool for format processing, which can be seamlessly integrated into existing LLM inference workflows, ensuring both human readability and LLM efficiency. 

**Abstract (ZH)**: 源代码通常通过缩进和换行等元素格式化以提高人类开发者的可读性。然而，这些视觉辅助元素似乎并不像对人类开发者那样对大型语言模型（LLMs）有益，因为代码被处理为一系列线性标记序列。此外，这些额外的标记可能会导致LLMs的计算成本增加和响应时间变长。如果这些格式化元素对LLMs来说是非必要的，我们可以通过移除它们来减少这些成本。为了搞清楚格式化元素的作用，我们进行了一项全面的经验研究，评估代码格式化对LLM性能和效率的影响。通过针对四种编程语言（Java、Python、C++、C#）和十种LLM（包括商业和开源模型）的大规模实验，在填写中间代码完成任务中，系统地分析移除格式化元素后的标记数量和性能。主要发现表明，LLMs在格式化代码和非格式化代码之间可以保持相同性能，平均每减少24.5%的输入标记，输出标记几乎无减少。这使得代码格式去除成为一个实用的优化策略，以提高LLM效率。进一步探索显示，无论是提示还是微调LLMs，都可以显著减少多达36.1%的输出代码长度，而不牺牲正确性。为了便于实际应用，我们开发了一个双向代码转换工具，用于格式化处理，可以无缝集成到现有的LLM推理工作流中，确保同时保持人类可读性和LLM效率。 

---
