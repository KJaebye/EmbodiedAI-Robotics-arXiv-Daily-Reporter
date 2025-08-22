# In-Context Iterative Policy Improvement for Dynamic Manipulation 

**Title (ZH)**: 基于上下文的迭代策略改进方法用于动态操作 

**Authors**: Mark Van der Merwe, Devesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2508.15021)  

**Abstract**: Attention-based architectures trained on internet-scale language data have demonstrated state of the art reasoning ability for various language-based tasks, such as logic problems and textual reasoning. Additionally, these Large Language Models (LLMs) have exhibited the ability to perform few-shot prediction via in-context learning, in which input-output examples provided in the prompt are generalized to new inputs. This ability furthermore extends beyond standard language tasks, enabling few-shot learning for general patterns. In this work, we consider the application of in-context learning with pre-trained language models for dynamic manipulation. Dynamic manipulation introduces several crucial challenges, including increased dimensionality, complex dynamics, and partial observability. To address this, we take an iterative approach, and formulate our in-context learning problem to predict adjustments to a parametric policy based on previous interactions. We show across several tasks in simulation and on a physical robot that utilizing in-context learning outperforms alternative methods in the low data regime. Video summary of this work and experiments can be found this https URL. 

**Abstract (ZH)**: 基于互联网规模语言数据训练的注意力机制架构展现了各种语言任务中优异的推理能力，如逻辑问题和文本推理。此外，这些大型语言模型（LLMs）还展示了通过上下文学习进行少样本预测的能力，在这种学习方式中，提示中的输入-输出示例可以泛化到新的输入。这一能力进一步超越了标准的语言任务，使少样本学习适用于更广泛的模式。在这项工作中，我们考虑使用预训练语言模型进行动态操作的应用。动态操作引入了若干关键挑战，包括维度增加、复杂动力学和部分可观测性。为应对这些挑战，我们采取迭代方法，并将上下文学习问题形式化为基于先前交互预测参数化策略调整的问题。我们在模拟和物理机器人上进行的多项任务中展示了利用上下文学习在数据稀缺条件下优于其他方法的结果。有关此项工作和实验的视频总结，请访问以下链接：this https URL。 

---
# Language-Guided Tuning: Enhancing Numeric Optimization with Textual Feedback 

**Title (ZH)**: 语言引导调整：结合文本反馈增强数值优化 

**Authors**: Yuxing Lu, Yucheng Hu, Nan Sun, Xukai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15757)  

**Abstract**: Configuration optimization remains a critical bottleneck in machine learning, requiring coordinated tuning across model architecture, training strategy, feature engineering, and hyperparameters. Traditional approaches treat these dimensions independently and lack interpretability, while recent automated methods struggle with dynamic adaptability and semantic reasoning about optimization decisions. We introduce Language-Guided Tuning (LGT), a novel framework that employs multi-agent Large Language Models to intelligently optimize configurations through natural language reasoning. We apply textual gradients - qualitative feedback signals that complement numerical optimization by providing semantic understanding of training dynamics and configuration interdependencies. LGT coordinates three specialized agents: an Advisor that proposes configuration changes, an Evaluator that assesses progress, and an Optimizer that refines the decision-making process, creating a self-improving feedback loop. Through comprehensive evaluation on six diverse datasets, LGT demonstrates substantial improvements over traditional optimization methods, achieving performance gains while maintaining high interpretability. 

**Abstract (ZH)**: 基于语言指导的调优：一种新的多智能体框架 

---
# Transduction is All You Need for Structured Data Workflows 

**Title (ZH)**: 结构化数据工作流中传播学习即一切 

**Authors**: Alfio Gliozzo, Naweed Khan, Christodoulos Constantinides, Nandana Mihindukulasooriya, Nahuel Defosse, Junkyu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.15610)  

**Abstract**: This paper introduces Agentics, a modular framework for building agent-based systems capable of structured reasoning and compositional generalization over complex data. Designed with research and practical applications in mind, Agentics offers a novel perspective on working with data and AI workflows. In this framework, agents are abstracted from the logical flow and they are used internally to the data type to enable logical transduction among data. Agentics encourages AI developers to focus on modeling data rather than crafting prompts, enabling a declarative language in which data types are provided by LLMs and composed through logical transduction, which is executed by LLMs when types are connected. We provide empirical evidence demonstrating the applicability of this framework across domain-specific multiple-choice question answering, semantic parsing for text-to-SQL, and automated prompt optimization tasks, achieving state-of-the-art accuracy or improved scalability without sacrificing performance. The open-source implementation is available at \texttt{this https URL}. 

**Abstract (ZH)**: 本文介绍了Agentics，这是一种模块化框架，用于构建能够进行结构化推理和复杂数据组合理式泛化的代理系统。该框架旨在研究和实际应用中使用，提供了处理数据和AI工作流的新视角。在该框架中，代理被从逻辑流程中抽象出来，并在数据类型内部使用，以实现数据之间的逻辑转换。Agentics 鼓励AI开发者专注于数据建模而非构造提示，实现一种声明性语言，在这种语言中，数据类型由LLM提供，并通过逻辑转换进行组合，当类型连接时，由LLM执行转换。我们提供了实验证据，证明该框架在特定领域的多项选择题回答、文本到SQL的语义解析以及自动提示优化任务中具有适用性，实现了最先进的准确率或提高了可扩展性而不牺牲性能。开源实现可在\texttt{this https URL}获得。 

---
# DeepThink3D: Enhancing Large Language Models with Programmatic Reasoning in Complex 3D Situated Reasoning Tasks 

**Title (ZH)**: DeepThink3D：在复杂三维情境推理任务中增强大型语言模型的程序化推理能力 

**Authors**: Jiayi Song, Rui Wan, Lipeng Ma, Weidong Yang, Qingyuan Zhou, Yixuan Li, Ben Fei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15548)  

**Abstract**: This work enhances the ability of large language models (LLMs) to perform complex reasoning in 3D scenes. Recent work has addressed the 3D situated reasoning task by invoking tool usage through large language models. Large language models call tools via APIs and integrate the generated programs through a chain of thought to solve problems based on the program results. However, due to the simplicity of the questions in the dataset, the generated program reasoning chains are relatively short. To solve this main challenge, in this paper, we introduce DeepThink3D to enhance the tool usage of LLMs in complex 3D situated reasoning tasks. Our work proposes a combinatorial and iterative evolutionary approach on the SQA3D benchmark to generate more complex questions. Building on this foundation, we fine-tune the large language model to make it more proficient in using 3D tools. By employing Direct Preference Optimization (DPO), we directly optimize the toolchain strategies generated by models, thereby enhancing their accuracy in complex tasks. 

**Abstract (ZH)**: 本研究增强了大型语言模型（LLMs）在3D场景中进行复杂推理的能力。近期的研究通过调用工具使用来解决3D情境推理任务，使大型语言模型通过API调用工具，并通过chain of thought生成程序来解决问题。然而，由于数据集中问题的简单性，生成的程序推理链相对较短。为解决这一主要挑战，本文引入了DeepThink3D，以增强LLMs在复杂3D情境推理任务中的工具使用能力。我们的工作在SQA3D基准上提出了组合性和迭代性的进化方法来生成更复杂的问答。在此基础上，我们对大型语言模型进行微调，使其更擅长使用3D工具。通过采用直接偏好优化（DPO），我们可以直接优化模型生成的工具链策略，从而在复杂任务中提高其准确性。 

---
# Super-additive Cooperation in Language Model Agents 

**Title (ZH)**: 超加性合作在语言模型代理中 

**Authors**: Filippo Tonini, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2508.15510)  

**Abstract**: With the prospect of autonomous artificial intelligence (AI) agents, studying their tendency for cooperative behavior becomes an increasingly relevant topic. This study is inspired by the super-additive cooperation theory, where the combined effects of repeated interactions and inter-group rivalry have been argued to be the cause for cooperative tendencies found in humans. We devised a virtual tournament where language model agents, grouped into teams, face each other in a Prisoner's Dilemma game. By simulating both internal team dynamics and external competition, we discovered that this blend substantially boosts both overall and initial, one-shot cooperation levels (the tendency to cooperate in one-off interactions). This research provides a novel framework for large language models to strategize and act in complex social scenarios and offers evidence for how intergroup competition can, counter-intuitively, result in more cooperative behavior. These insights are crucial for designing future multi-agent AI systems that can effectively work together and better align with human values. Source code is available at this https URL. 

**Abstract (ZH)**: 随着自主人工智能（AI）代理的前景日益显现，研究其合作行为倾向变得 increasingly relevant。本研究受超加性合作理论启发，该理论认为重复互动和团体间的竞争是人类表现出合作倾向的原因。我们设计了一种虚拟锦标赛，其中语言模型代理被分组进行互动，在囚徒困境游戏中彼此对抗。通过模拟团队内部动态和外部竞争，我们发现这种结合在总体上和初步的一次性合作水平上显著提高了合作水平。本研究提供了一种新的框架，使大型语言模型能够在复杂的社会场景中制定策略并采取行动，并提供了关于团体间竞争如何出人意料地导致更多合作行为的证据。这些见解对于设计未来能够有效协同工作的多代理AI系统，并更好地与人类价值观契合至关重要。相关源代码可在该网址获取。 

---
# Think in Blocks: Adaptive Reasoning from Direct Response to Deep Reasoning 

**Title (ZH)**: 从直接响应到深层推理的模块化思考：自适应推理 

**Authors**: Yekun Zhu, Guang Chen, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15507)  

**Abstract**: Large Language Models (LLMs) with chains-of-thought have demonstrated strong performance on an increasing range of tasks, particularly those involving complex logical reasoning. However, excessively long chains can lead to overthinking, causing computational waste and slower responses. This raises a question: can LLMs dynamically adjust the length of their reasoning processes based on task complexity? To address this, we propose the Think in Blocks framework, which enables adaptive reasoning-from zero to deep reasoning-by partitioning the reasoning process into a tunable number of blocks. Our main contributions are: (1) Establishing an explicit block-structured paradigm in which the model first predicts an integer reasoning budget-the number of blocks-and then partitions its reasoning accordingly; (2) Training an adaptive model through a three-stage pipeline-Supervised Fine-Tuning, reward-guided Direct Preference Optimization, and Reinforcement Learning-that adjusts its reasoning depth to problem difficulty; (3) Exploiting the explicit block count to dynamically control reasoning depth at inference time, allowing flexible adjustment of chain-of-thought length during deployment. 

**Abstract (ZH)**: 具有思维链的大语言模型（LLMs）在执行涉及复杂逻辑推理的越来越多任务中展示了强大的性能。然而，过长的思维链可能导致过度思考，造成计算资源浪费和响应变慢。这引发了一个问题：大语言模型能否根据任务复杂度动态调整其推理过程的长度？为解决这一问题，我们提出了块思维框架（Think in Blocks），该框架通过将推理过程划分为可调数量的块来实现从浅层到深层推理的适应性推理。我们的主要贡献是：（1）确立了一个明确的块结构范式，模型首先预测一个整数推理预算——块的数量，然后根据该预算划分推理；（2）通过一个三阶段训练管道——监督微调、奖励导向的直接偏好优化和强化学习来训练适应性模型，使其根据问题难度调整推理深度；（3）利用显式的块计数在推理时动态控制推理深度，在部署过程中灵活调整思维链长度。 

---
# From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence 

**Title (ZH)**: 从比特到董事会：面向商业卓越的前沿多代理大语言模型框架 

**Authors**: Zihao Wang, Junming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15447)  

**Abstract**: Large Language Models (LLMs) have shown promising potential in business applications, particularly in enterprise decision support and strategic planning, yet current approaches often struggle to reconcile intricate operational analyses with overarching strategic goals across diverse market environments, leading to fragmented workflows and reduced collaboration across organizational levels. This paper introduces BusiAgent, a novel multi-agent framework leveraging LLMs for advanced decision-making in complex corporate environments. BusiAgent integrates three core innovations: an extended Continuous Time Markov Decision Process (CTMDP) for dynamic agent modeling, a generalized entropy measure to optimize collaborative efficiency, and a multi-level Stackelberg game to handle hierarchical decision processes. Additionally, contextual Thompson sampling is employed for prompt optimization, supported by a comprehensive quality assurance system to mitigate errors. Extensive empirical evaluations across diverse business scenarios validate BusiAgent's efficacy, demonstrating its capacity to generate coherent, client-focused solutions that smoothly integrate granular insights with high-level strategy, significantly outperforming established approaches in both solution quality and user satisfaction. By fusing cutting-edge AI technologies with deep business insights, BusiAgent marks a substantial step forward in AI-driven enterprise decision-making, empowering organizations to navigate complex business landscapes more effectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）在商业应用中展现出潜在的优势，特别是在企业决策支持和战略规划方面，但当前的方法往往难以在多变的市场环境中协调复杂的操作分析与总体战略目标，导致工作流程碎片化并降低组织不同层级之间的协作。本文介绍了BusiAgent，这是一种利用LLMs的新颖多智能体框架，用于处理复杂企业环境下的高级决策。BusiAgent集成了三项核心创新：扩展的连续时间马尔可夫决策过程（CTMDP）以动态建模智能体、广义熵度量以优化协作效率、以及多层次的斯坦克尔伯格博弈以处理层级决策过程。此外，还采用了上下文泰勒斯采样方法以优化提示，并通过全面的质量保证系统减轻错误。在多种业务场景下的广泛实证评估验证了BusiAgent的有效性，显示出其能够生成一致的、以客户需求为导向的解决方案，能够将细粒度的见解与高层次的战略无缝整合，其在解决方案质量和用户满意度方面均显著优于现有方法。通过将先进的AI技术与深厚的商业洞察相结合，BusiAgent为企业驱动的决策制定带来了重要突破，助力组织更有效地应对复杂的商业环境。 

---
# GraSP: A Unified Graph-Based Framework for Scalable Generation, Quality Tagging, and Management of Synthetic Data for SFT and DPO 

**Title (ZH)**: GraSP: 一种统一的基于图的合成数据生成、质量标注和管理框架以支持SFT和DPO 

**Authors**: Bidyapati Pradhan, Surajit Dasgupta, Amit Kumar Saha, Omkar Anustoop, Sriram Puttagunta, Vipul Mittal, Gopal Sarda  

**Link**: [PDF](https://arxiv.org/pdf/2508.15432)  

**Abstract**: The advancement of large language models (LLMs) is critically dependent on the availability of high-quality datasets for Supervised Fine-Tuning (SFT), alignment tasks like Direct Preference Optimization (DPO), etc. In this work, we present a comprehensive synthetic data generation framework that facilitates scalable, configurable, and high-fidelity generation of synthetic data tailored for these training paradigms. Our approach employs a modular and configuration-based pipeline capable of modeling complex dialogue flows with minimal manual intervention. This framework uses a dual-stage quality tagging mechanism, combining heuristic rules and LLM-based evaluations, to automatically filter and score data extracted from OASST-formatted conversations, ensuring the curation of high-quality dialogue samples. The resulting datasets are structured under a flexible schema supporting both SFT and DPO use cases, enabling seamless integration into diverse training workflows. Together, these innovations offer a robust solution for generating and managing synthetic conversational data at scale, significantly reducing the overhead of data preparation in LLM training pipelines. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的进步关键依赖于高质量数据集的支持，包括监督微调（SFT）、直接偏好优化（DPO）等对齐任务。在本文中，我们提出了一种全面的合成数据生成框架，以实现这些训练范式所需的可扩展、可配置和高保真合成数据的生成。该方法采用模块化和基于配置的流水线，能够在 Minimal 手动干预的情况下建模复杂的对话流程。该框架使用一种双重质量标记机制，结合启发式规则和LLM评估，自动筛选和评分OASST格式对话中提取的数据，确保对话样本的质量。生成的数据集采用灵活的Schema支持SFT和DPO等多种应用场景，使这些数据能够无缝集成到各种训练工作流中。这些创新共同提供了一种稳健的解决方案，用于大规模生成和管理合成对话数据，显著减少了LLM训练流水线中的数据准备开销。 

---
# DiagECG: An LLM-Driven Framework for Diagnostic Reasoning via Discretized ECG Tokenization 

**Title (ZH)**: DiagECG：一种基于离散化心电图标记驱动的大型语言模型框架用于诊断推理 

**Authors**: Jinning Yang, Wen Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15338)  

**Abstract**: Electrocardiography plays a central role in cardiovascular diagnostics, yet existing automated approaches often struggle to generalize across clinical tasks and offer limited support for open-ended reasoning. We present DiagECG, a novel framework that integrates time-series and language modeling by enabling large language models to process 12-lead ECG signals for clinical text generation tasks. Our approach discretizes continuous ECG embeddings into symbolic tokens using a lead-independent encoder and quantization module. These tokens are then used to extend the vocabulary of LLM, allowing the model to handle both ECG and natural language inputs in a unified manner. To bridge the modality gap, we pretrain the model on an autoregressive ECG forecasting task, enabling the LLM to model temporal dynamics using its native language modeling capabilities. Finally, we perform instruction tuning on both ECG question answering and diagnostic report generation. Without modifying the core model, DiagECG achieves strong performance across tasks while maintaining generalization to out-of-distribution settings. Extensive experiments demonstrate the effectiveness of each component and highlight the potential of integrating symbolic ECG representations into LLMs for medical reasoning. 

**Abstract (ZH)**: 心电图在心血管诊断中发挥着核心作用，但现有的自动化方法往往难以跨临床任务泛化，并提供的开放推理支持有限。我们提出了DiagECG，一种通过使大型语言模型能够处理12导联ECG信号，从而为临床文本生成任务整合时间序列和语言建模的新框架。我们的方法使用与导联无关的编码器和量化模块将连续的ECG嵌入离散化为符号令牌。这些令牌随后用于扩展大语言模型的词汇，从而使模型能够同时处理ECG和自然语言输入。为了解决模态差距，我们在自回归ECG预测任务上对模型进行预训练，使大语言模型能够利用其固有的语言建模能力来建模时间动态。最后，我们在ECG问题回答和诊断报告生成方面进行指令调整。在不修改核心模型的情况下，DiagECG在各种任务上实现了强大的性能，并保持了在分布外设置中的泛化能力。广泛的实验表明，每个组件的有效性，并强调将符号ECG表示整合到大语言模型中以进行医学推理的潜力。 

---
# RETAIL: Towards Real-world Travel Planning for Large Language Models 

**Title (ZH)**: RETAIL: 向现实世界中的旅行规划迈进的大语言模型方向 

**Authors**: Bin Deng, Yizhe Feng, Zeming Liu, Qing Wei, Xiangrong Zhu, Shuai Chen, Yuanfang Guo, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15335)  

**Abstract**: Although large language models have enhanced automated travel planning abilities, current systems remain misaligned with real-world scenarios. First, they assume users provide explicit queries, while in reality requirements are often implicit. Second, existing solutions ignore diverse environmental factors and user preferences, limiting the feasibility of plans. Third, systems can only generate plans with basic POI arrangements, failing to provide all-in-one plans with rich details. To mitigate these challenges, we construct a novel dataset \textbf{RETAIL}, which supports decision-making for implicit queries while covering explicit queries, both with and without revision needs. It also enables environmental awareness to ensure plan feasibility under real-world scenarios, while incorporating detailed POI information for all-in-one travel plans. Furthermore, we propose a topic-guided multi-agent framework, termed TGMA. Our experiments reveal that even the strongest existing model achieves merely a 1.0% pass rate, indicating real-world travel planning remains extremely challenging. In contrast, TGMA demonstrates substantially improved performance 2.72%, offering promising directions for real-world travel planning. 

**Abstract (ZH)**: 虽然大型语言模型增强了自动旅行规划的能力，但当前系统仍与现实场景不一致。首先，它们假设用户会提供明确查询，而在现实中需求往往是隐含的。其次，现有解决方案忽略了多种环境因素和用户偏好，限制了规划的实际可行性。第三，系统只能生成基本POI安排的计划，无法提供包含丰富细节的一站式计划。为应对这些挑战，我们构建了一个新的数据集RETAIL，支持处理隐含查询并涵盖需要修订和不需要修订的明确查询，同时增强了环境意识，确保在现实场景下计划的可行性，同时也整合了详细的POI信息以支持一站式旅行计划。此外，我们提出了一个主题引导的多agent框架，称为TGMA。我们的实验表明，当前最强的模型的通过率仅为1.0%，表明现实世界的旅行规划依然极具挑战性。相比之下，TGMA展示了显著改进的性能，达到2.72%，提供了现实世界旅行规划的有前途的方向。 

---
# Coarse-to-Fine Grounded Memory for LLM Agent Planning 

**Title (ZH)**: 从粗到细 grounded 记忆机制下的大语言模型代理规划 

**Authors**: Wei Yang, Jinwei Xiao, Hongming Zhang, Qingyang Zhang, Yanna Wang, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15305)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have driven growing interest in LLM-based agents for complex planning tasks. To avoid costly agent training, many studies adopted memory mechanism that enhances LLM with offline experiences or online trajectory analysis. However, existing works focus on single-granularity memory derived from dynamic environmental interactions, which are inherently constrained by the quality of the collected experiences. This limitation, in turn, constrain the diversity of knowledge and the flexibility of planning. We propose Coarse-to-Fine Grounded Memory (\Ours{}), a novel framework that grounds coarse-to-fine memories with LLM, thereby fully leverage them for flexible adaptation to diverse scenarios. \Ours{} grounds environmental information into coarse-grained focus points to guide experience collection in training tasks, followed by grounding of actionable hybrid-grained tips from each experience. At inference, \Ours{} retrieves task-relevant experiences and tips to support planning. When facing environmental anomalies, the LLM grounds the current situation into fine-grained key information, enabling flexible self-QA reflection and plan correction. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）促进了基于LLMs的代理在复杂规划任务中的应用兴趣。为避免代理训练成本高昂，许多研究采用了记忆机制，通过离线经验或在线轨迹分析增强LLMs。然而，现有工作主要关注从动态环境交互中提取的单一粒度记忆，这些记忆本质上受到收集经验质量的限制。这种限制反过来限制了知识的多样性和规划的灵活性。我们提出了从粗到细 grounding 记忆（\Ours{}），这是一种新颖的框架，通过LLMs对粗到细的记忆进行grounding，从而充分利用这些记忆实现对多样化场景的灵活适应。在训练任务中，\Ours{}将环境信息转换为粗粒度的焦点点以指导经验收集，在每项经验中接地可操作的混合粒度提示。在推理时，\Ours{}检索与任务相关的历史经验与提示以支持规划。面对环境异常时，LLMs将当前情况转换为细粒度的关键信息，从而实现灵活的自我问答反思和计划修正。 

---
# R-ConstraintBench: Evaluating LLMs on NP-Complete Scheduling 

**Title (ZH)**: R-约束基准：评估大语言模型在NP完全调度问题上的性能 

**Authors**: Raj Jain, Marc Wetter  

**Link**: [PDF](https://arxiv.org/pdf/2508.15204)  

**Abstract**: Effective scheduling under tight resource, timing, and operational constraints underpins large-scale planning across sectors such as capital projects, manufacturing, logistics, and IT fleet transitions. However, the reliability of large language models (LLMs) when reasoning under high-constraint regimes is insufficiently characterized. To address this gap, we present R-ConstraintBench, a scalable framework that evaluates models on Resource-Constrained Project Scheduling Problems (RCPSP), an NP-Complete feasibility class, while difficulty increases via linear growth in constraints. R-ConstraintBench incrementally increases non-redundant precedence constraints in Directed Acyclic Graphs (DAGs) and then introduces downtime, temporal windows, and disjunctive constraints. As an illustrative example, we instantiate the benchmark in a data center migration setting and evaluate multiple LLMs using feasibility and error analysis, identifying degradation thresholds and constraint types most associated with failure. Empirically, strong models are near-ceiling on precedence-only DAGs, but feasibility performance collapses when downtime, temporal windows, and disjunctive constraints interact, implicating constraint interaction, not graph depth, as the principal bottleneck. Performance on clean synthetic ramps also does not guarantee transfer to domain-grounded scenarios, underscoring limited generalization. 

**Abstract (ZH)**: 有效的资源、时间和操作约束下的调度对于跨资本项目、制造、物流和IT机队转换等多个领域的大规模规划至关重要。然而，大型语言模型（LLMs）在高约束环境下的可靠性尚未得到充分表征。为解决这一问题，我们提出了一种可扩展的框架R-ConstraintBench，该框架在Resource-Constrained Project Scheduling Problems（RCPSP）这一NP完全可行类问题上评估模型，通过约束的线性增长增加难度。R-ConstraintBench通过逐步增加有向无环图（DAGs）中的非冗余前置约束，然后引入停机时间、时间窗口和排斥约束来增加难度。作为示例，我们在数据中心迁移场景中实例化基准，并使用可行性和错误分析评估多个LLM，识别与失败最相关的降级阈值和约束类型。实验结果显示，强大的模型在仅前置约束的DAG上接近上限，但当停机时间、时间窗口和排斥约束相互作用时，可行性能急剧下降，表明约束交互而非图深度是主要瓶颈。清洁合成坡度上的性能也不能保证转移到实际场景中，强调了模型泛化能力的局限性。 

---
# LLM4Sweat: A Trustworthy Large Language Model for Hyperhidrosis Support 

**Title (ZH)**: LLM4Sweat: 一种可靠的大型语言模型以支持多汗症管理 

**Authors**: Wenjie Lin, Jin Wei-Kocsis  

**Link**: [PDF](https://arxiv.org/pdf/2508.15192)  

**Abstract**: While large language models (LLMs) have shown promise in healthcare, their application for rare medical conditions is still hindered by scarce and unreliable datasets for fine-tuning. Hyperhidrosis, a disorder causing excessive sweating beyond physiological needs, is one such rare disorder, affecting 2-3% of the population and significantly impacting both physical comfort and psychosocial well-being. To date, no work has tailored LLMs to advance the diagnosis or care of hyperhidrosis. To address this gap, we present LLM4Sweat, an open-source and domain-specific LLM framework for trustworthy and empathetic hyperhidrosis support. The system follows a three-stage pipeline. In the data augmentation stage, a frontier LLM generates medically plausible synthetic vignettes from curated open-source data to create a diverse and balanced question-answer dataset. In the fine-tuning stage, an open-source foundation model is fine-tuned on the dataset to provide diagnosis, personalized treatment recommendations, and empathetic psychological support. In the inference and expert evaluation stage, clinical and psychological specialists assess accuracy, appropriateness, and empathy, with validated responses iteratively enriching the dataset. Experiments show that LLM4Sweat outperforms baselines and delivers the first open-source LLM framework for hyperhidrosis, offering a generalizable approach for other rare diseases with similar data and trustworthiness challenges. 

**Abstract (ZH)**: 基于大型语言模型的汗疱症支持框架：LLM4Sweat 

---
# aiXiv: A Next-Generation Open Access Ecosystem for Scientific Discovery Generated by AI Scientists 

**Title (ZH)**: aiXiv: 由AI科学家生成的下一代开放访问科学发现生态系统 

**Authors**: Pengsong Zhang, Xiang Hu, Guowei Huang, Yang Qi, Heng Zhang, Xiuxu Li, Jiaxing Song, Jiabin Luo, Yijiang Li, Shuo Yin, Chengxiao Dai, Eric Hanchen Jiang, Xiaoyan Zhou, Zhenfei Yin, Boqin Yuan, Jing Dong, Guinan Su, Guanren Qiao, Haiming Tang, Anghong Du, Lili Pan, Zhenzhong Lan, Xinyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15126)  

**Abstract**: Recent advances in large language models (LLMs) have enabled AI agents to autonomously generate scientific proposals, conduct experiments, author papers, and perform peer reviews. Yet this flood of AI-generated research content collides with a fragmented and largely closed publication ecosystem. Traditional journals and conferences rely on human peer review, making them difficult to scale and often reluctant to accept AI-generated research content; existing preprint servers (e.g. arXiv) lack rigorous quality-control mechanisms. Consequently, a significant amount of high-quality AI-generated research lacks appropriate venues for dissemination, hindering its potential to advance scientific progress. To address these challenges, we introduce aiXiv, a next-generation open-access platform for human and AI scientists. Its multi-agent architecture allows research proposals and papers to be submitted, reviewed, and iteratively refined by both human and AI scientists. It also provides API and MCP interfaces that enable seamless integration of heterogeneous human and AI scientists, creating a scalable and extensible ecosystem for autonomous scientific discovery. Through extensive experiments, we demonstrate that aiXiv is a reliable and robust platform that significantly enhances the quality of AI-generated research proposals and papers after iterative revising and reviewing on aiXiv. Our work lays the groundwork for a next-generation open-access ecosystem for AI scientists, accelerating the publication and dissemination of high-quality AI-generated research content. Code is available at this https URL. Website is available at this https URL. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）使AI代理能够自主生成科学提案、开展实验、撰写论文和进行同行评审。然而，大量由AI生成的研究内容与分散且主要封闭的出版生态系统相碰撞。传统期刊和会议依赖于人力同行评审，这使得它们难以扩展，并且往往不愿接受AI生成的研究内容；现有的预印本服务器（如arXiv）缺乏严格的质量控制机制。因此，大量高质量的AI生成研究缺乏适当的传播渠道，阻碍了其潜在的科学发展。为应对这些挑战，我们引入了aiXiv，这是一个用于人类和AI科学家的下一代开放获取平台。其多代理架构允许研究提案和论文由人类和AI科学家共同提交、评审和迭代改进。此外，aiXiv提供了API和MCP接口，使异构的人类和AI科学家无缝集成，创建一个可扩展和可扩展的自主科学发现生态系统。通过广泛实验，我们证明aiXiv是一个可靠且 robust 的平台，在aiXiv上经过迭代修订和评审后，显著提高了AI生成的研究提案和论文的质量。我们的工作为基础构建了一个下一代开放访问生态系统，加速高质量AI生成研究成果的出版和传播。代码可在以下网址获得：this https URL。网站可在以下网址获得：this https URL。 

---
# S3LoRA: Safe Spectral Sharpness-Guided Pruning in Adaptation of Agent Planner 

**Title (ZH)**: S3LoRA: 安全频谱锐化指导的代理规划器适配剪枝 

**Authors**: Shuang Ao, Gopal Rumchurn  

**Link**: [PDF](https://arxiv.org/pdf/2508.15068)  

**Abstract**: Adapting Large Language Models (LLMs) using parameter-efficient fine-tuning (PEFT) techniques such as LoRA has enabled powerful capabilities in LLM-based agents. However, these adaptations can unintentionally compromise safety alignment, leading to unsafe or unstable behaviors, particularly in agent planning tasks. Existing safety-aware adaptation methods often require access to both base and instruction-tuned model checkpoints, which are frequently unavailable in practice, limiting their applicability. We propose S3LoRA (Safe Spectral Sharpness-Guided Pruning LoRA), a lightweight, data-free, and model-independent framework that mitigates safety risks in LoRA-adapted models by inspecting only the fine-tuned weight updates. We first introduce Magnitude-Aware Spherically Normalized SVD (MAS-SVD), which robustly analyzes the structural properties of LoRA updates while preserving global magnitude information. We then design the Spectral Sharpness Index (SSI), a sharpness-aware metric to detect layers with highly concentrated and potentially unsafe updates. These layers are pruned post-hoc to reduce risk without sacrificing task performance. Extensive experiments and ablation studies across agent planning and language generation tasks show that S3LoRA consistently improves safety metrics while maintaining or improving utility metrics and significantly reducing inference cost. These results establish S3LoRA as a practical and scalable solution for safely deploying LLM-based agents in real-world, resource-constrained, and safety-critical environments. 

**Abstract (ZH)**: 使用LoRA等参数高效微调（PEFT）技术适应大型语言模型（LLMs）已赋予基于LLM的代理强大的能力。然而，这些适应可能会无意中损害安全性对齐，特别是在代理规划任务中可能导致不安全或不稳定的行为。现有的安全性感知适应方法通常需要访问基础模型和指令微调模型的检查点，而在实践中这些检查点往往不可用，限制了它们的应用范围。我们提出了S3LoRA（Safe Spectral Sharpness-Guided Pruning LoRA），这是一种轻量级、无需数据且模型独立的框架，通过仅检查微调权重更新来减轻LoRA适应模型中的安全性风险。我们首先引入了感知幅度的球形规范化SVD（MAS-SVD），它稳健地分析LoRA更新的结构特性，同时保留整体幅度信息。然后我们设计了频谱尖锐度索引（SSI），这是一种尖锐度感知的指标，用于检测具有高度集中且可能不安全更新的层。这些层在后处理中被剪枝以降低风险，而不牺牲任务性能。广泛的实验和剥离研究显示，S3LoRA在提高安全性指标的同时保持或改进了实用性指标，并显著降低了推理成本。这些结果表明，S3LoRA是一种实用且可扩展的解决方案，可在资源受限和安全性关键的实际环境中安全部署基于LLM的代理。 

---
# Don't Think Twice! Over-Reasoning Impairs Confidence Calibration 

**Title (ZH)**: 不要犹豫！过度推理会损害自信度校准。 

**Authors**: Romain Lacombe, Kerrie Wu, Eddie Dilworth  

**Link**: [PDF](https://arxiv.org/pdf/2508.15050)  

**Abstract**: Large Language Models deployed as question answering tools require robust calibration to avoid overconfidence. We systematically evaluate how reasoning capabilities and budget affect confidence assessment accuracy, using the ClimateX dataset (Lacombe et al., 2023) and expanding it to human and planetary health. Our key finding challenges the "test-time scaling" paradigm: while recent reasoning LLMs achieve 48.7% accuracy in assessing expert confidence, increasing reasoning budgets consistently impairs rather than improves calibration. Extended reasoning leads to systematic overconfidence that worsens with longer thinking budgets, producing diminishing and negative returns beyond modest computational investments. Conversely, search-augmented generation dramatically outperforms pure reasoning, achieving 89.3% accuracy by retrieving relevant evidence. Our results suggest that information access, rather than reasoning depth or inference budget, may be the critical bottleneck for improved confidence calibration of knowledge-intensive tasks. 

**Abstract (ZH)**: 大型语言模型作为问答工具部署时需要稳健校准以避免过度自信。我们系统评估推理能力与预算对信心评估准确度的影响，使用ClimateX数据集（Lacombe等，2023）并扩展到人类和 planetary 健康领域。我们关键发现挑战了“测试时缩放”范式：虽然近期推理大模型在评估专家信心方面达到48.7%的准确性，但增加推理预算始终会恶化而不是改善校准。延长推理会导致系统性过度自信，随着思考预算的增加而加剧，超出适度计算投资后产生递减甚至负效益。相反，搜索增强型生成显著优于单纯推理，通过检索相关证据达到89.3%的准确度。我们的结果表明，对于知识密集型任务的信心校准改进，信息访问可能比推理深度或推理预算更为关键。 

---
# Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism 

**Title (ZH)**: Collab-REC: 一种基于LLM的代理框架，用于平衡旅游推荐 

**Authors**: Ashmi Banerjee, Fitri Nur Aisyah, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2508.15030)  

**Abstract**: We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems. 

**Abstract (ZH)**: Collab-REC：一个多智能体框架，用于应对流行性偏见并增强旅游推荐的多样性 

---
# LiveMCP-101: Stress Testing and Diagnosing MCP-enabled Agents on Challenging Queries 

**Title (ZH)**: LiveMCP-101：针对具有MCP功能代理的苛刻查询进行压力测试与诊断 

**Authors**: Ming Yin, Dinghan Shen, Silei Xu, Jianbing Han, Sixun Dong, Mian Zhang, Yebowen Hu, Shujian Liu, Simin Ma, Song Wang, Sathish Reddy Indurthi, Xun Wang, Yiran Chen, Kaiqiang Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.15760)  

**Abstract**: Tool calling has emerged as a critical capability for AI agents to interact with the real world and solve complex tasks. While the Model Context Protocol (MCP) provides a powerful standardized framework for tool integration, there is a significant gap in benchmarking how well AI agents can effectively solve multi-step tasks using diverse MCP tools in realistic, dynamic scenarios. In this work, we present LiveMCP-101, a benchmark of 101 carefully curated real-world queries, refined through iterative LLM rewriting and manual review, that require coordinated use of multiple MCP tools including web search, file operations, mathematical reasoning, and data analysis. Moreover, we introduce a novel evaluation approach that leverages ground-truth execution plans rather than raw API outputs, better reflecting the evolving nature of real-world environments. Experiments show that even frontier LLMs achieve a success rate below 60\%, highlighting major challenges in tool orchestration. Detailed ablations and error analysis further reveal distinct failure modes and inefficiencies in token usage, pointing to concrete directions for advancing current models. LiveMCP-101 sets a rigorous standard for evaluating real-world agent capabilities, advancing toward autonomous AI systems that reliably execute complex tasks through tool use. 

**Abstract (ZH)**: LiveMCP-101：一种严格的标准，用于评估在多步骤任务中有效使用多样MCP工具的现实世界代理能力 

---
# Dissecting Tool-Integrated Reasoning: An Empirical Study and Analysis 

**Title (ZH)**: 剖析工具集成推理：一项实证研究与分析 

**Authors**: Yufeng Zhao, Junnan Liu, Hongwei Liu, Dongsheng Zhu, Yuan Shen, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15754)  

**Abstract**: Large Language Models (LLMs) have made significant strides in reasoning tasks through methods like chain-of-thought (CoT) reasoning. However, they often fall short in tasks requiring precise computations. Tool-Integrated Reasoning (TIR) has emerged as a solution by incorporating external tools into the reasoning process. Nevertheless, the generalization of TIR in improving the reasoning ability of LLM is still unclear. Additionally, whether TIR has improved the model's reasoning behavior and helped the model think remains to be studied. We introduce ReasonZoo, a comprehensive benchmark encompassing nine diverse reasoning categories, to evaluate the effectiveness of TIR across various domains. Additionally, we propose two novel metrics, Performance-Aware Cost (PAC) and Area Under the Performance-Cost Curve (AUC-PCC), to assess reasoning efficiency. Our empirical evaluation demonstrates that TIR-enabled models consistently outperform their non-TIR counterparts in both mathematical and non-mathematical tasks. Furthermore, TIR enhances reasoning efficiency, as evidenced by improved PAC and AUC-PCC, indicating reduced overthinking and more streamlined reasoning. These findings underscore the domain-general benefits of TIR and its potential to advance LLM capabilities in complex reasoning tasks. 

**Abstract (ZH)**: 大语言模型（LLMs）通过链式思考等方法在推理任务中取得了显著进展，但在需要精确计算的任务中往往表现不佳。工具集成推理（TIR）通过将外部工具纳入推理过程而作为一种解决方案出现，但TIR在提高LLM的推理能力方面的泛化能力仍不清楚。此外，TIR是否改善了模型的推理行为并帮助模型进行思考仍有待研究。我们引入了ReasonZoo，这是一个包含九种不同推理类别的全面基准，旨在评估TIR在各种领域的有效性。此外，我们提出了两种新的度量标准，即性能感知成本（PAC）和性能-成本曲线下的面积（AUC-PCC），以评估推理效率。我们的实证评估表明，TIR增强的模型在数学和非数学任务中始终优于非TIR模型。此外，TIR提高了推理效率，这从改进的PAC和AUC-PCC中可以得到证实，表明减少了过度思考并使推理更加流畅。这些发现强调了TIR在不同领域的普遍优势，并指出了其在复杂推理任务中提升LLM能力的潜力。 

---
# Trained Miniatures: Low cost, High Efficacy SLMs for Sales & Marketing 

**Title (ZH)**: 训练微缩模型：销售与市场推广中的低成本高效率选择 

**Authors**: Ishaan Bhola, Mukunda NS, Sravanth Kurmala, Harsh Nandwani, Arihant Jain  

**Link**: [PDF](https://arxiv.org/pdf/2508.15617)  

**Abstract**: Large language models (LLMs) excel in text generation; however, these creative elements require heavy computation and are accompanied by a steep cost. Especially for targeted applications such as sales and marketing outreach, these costs are far from feasible. This paper introduces the concept of "Trained Miniatures" - Small Language Models(SLMs) fine-tuned for specific, high-value applications, generating similar domain-specific responses for a fraction of the cost. 

**Abstract (ZH)**: 小型训练模型：特定高价值应用的低成本文本生成 

---
# Subjective Behaviors and Preferences in LLM: Language of Browsing 

**Title (ZH)**: LLM冲浪过程中主观行为与偏好的语言表达 

**Authors**: Sai Sundaresan, Harshita Chopra, Atanu R. Sinha, Koustava Goswami, Nagasai Saketh Naidu, Raghav Karan, N Anushka  

**Link**: [PDF](https://arxiv.org/pdf/2508.15474)  

**Abstract**: A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment. 

**Abstract (ZH)**: 大语言模型（LLM）在各个领域和任务中展现出灵活性，被认为是能够满足具有广泛行为和偏好的用户。当用户的行为和偏好本质上是主观的，如他们在网站或应用程序上的普遍且异质的浏览行为时，我们质疑这一关于LLM的看法。由此生成的页面序贯行为日志形成了类似于每个用户自己构建的“语言”，尽管缺乏自然语言中的结构和语法。我们询问：（i）小型LM能否比大型LM更好地代表“浏览语言”？（ii）单个参数集（或单个LM）能否充分捕捉众多用户异质且主观的行为和偏好？（iii）一个具有高平均性能且性能方差低的单个LM能否在用户级别实现良好的对齐？为此，我们引入了适应主观行为的聚类语言模型训练方法——HeTLM（具有异质性意识的语言模型训练）。我们发现：（i）使用页面级分词器训练的小型LM优于大型预训练或微调LM；（ii）具有异质性簇特定参数集的HeTLM优于相同家族的单个LM，控制参数数量；（iii）生成的均值增加且方差降低，表明对齐有所提高。 

---
# Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection 

**Title (ZH)**: 使用变形表示投影实现可信删除LLM中有害信息 

**Authors**: Chengcan Wu, Zeming Wei, Huanran Chen, Yinpeng Dong, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.15449)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive performance in various domains and tasks, concerns about their safety are becoming increasingly severe. In particular, since models may store unsafe knowledge internally, machine unlearning has emerged as a representative paradigm to ensure model safety. Existing approaches employ various training techniques, such as gradient ascent and negative preference optimization, in attempts to eliminate the influence of undesired data on target models. However, these methods merely suppress the activation of undesired data through parametric training without completely eradicating its informational traces within the model. This fundamental limitation makes it difficult to achieve effective continuous unlearning, rendering these methods vulnerable to relearning attacks. To overcome these challenges, we propose a Metamorphosis Representation Projection (MRP) approach that pioneers the application of irreversible projection properties to machine unlearning. By implementing projective transformations in the hidden state space of specific network layers, our method effectively eliminates harmful information while preserving useful knowledge. Experimental results demonstrate that our approach enables effective continuous unlearning and successfully defends against relearning attacks, achieving state-of-the-art performance in unlearning effectiveness while preserving natural performance. Our code is available in this https URL. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各种领域和任务中展现了 impressive 的性能，但对其安全性的担忧日益严重。特别是由于模型可能内部存储了不安全的知识，机器遗忘已作为一个代表性范式出现，以确保模型的安全性。现有方法通过使用梯度上升、负偏好优化等各种训练技术尝试消除目标模型中不希望数据的影响。然而，这些方法仅通过参数训练抑制不希望数据的激活，而没有完全消除其在模型中的信息痕迹，这一根本限制使得实现有效的连续遗忘变得困难，从而使这些方法容易受到重学攻击。为克服这些挑战，我们提出了一种变形表示投影（MRP）方法，该方法首次将不可逆投影性质应用于机器遗忘。通过在特定网络层的隐藏状态空间中实施投影变换，我们的方法有效消除了有害信息的同时保留了有用的知识。实验结果表明，我们的方法能够实现有效的连续遗忘并成功抵御重学攻击，实现了遗忘效果的最优性能，同时保持自然性能。我们的代码可在以下链接获取：this https URL。 

---
# Mitigating Hallucinations in LM-Based TTS Models via Distribution Alignment Using GFlowNets 

**Title (ZH)**: 基于GFlowNets的分布对齐方法减轻LM-Based TTS模型中的幻觉现象 

**Authors**: Chenlin Liu, Minghui Fang, Patrick Zhang, Wei Zhou, Jie Gao, Jiqing Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.15442)  

**Abstract**: Language Model (LM)-based Text-to-Speech (TTS) systems often generate hallucinated speech that deviates from input text. Existing mitigation strategies either demand excessive training resources or introduce significant inference latency. In this paper, we propose GFlOwNet-guided distribution AlignmenT (GOAT) for LM-based TTS, a post-training framework that mitigates hallucinations without relying on massive resources or inference cost. Specifically, we first conduct an uncertainty analysis, revealing a strong positive correlation between hallucination and model uncertainty. Based on this, we reformulate TTS generation as a trajectory flow optimization problem and introduce an enhanced Subtrajectory Balance objective together with a sharpened internal reward as target distribution. We further integrate reward temperature decay and learning rate optimization for stability and performance balance. Extensive experiments show that GOAT reduce over 50% character error rates on challenging test cases and lowering uncertainty by up to 58%, demonstrating its strong generalization ability and effectiveness. 

**Abstract (ZH)**: 基于语言模型的文本到语音系统中的GOAT引导分布对齐：一种无需大量资源的后训练框架以减轻幻听现象 

---
# Test-time Corpus Feedback: From Retrieval to RAG 

**Title (ZH)**: 测试时语料库反馈：从检索到RAG 

**Authors**: Mandeep Rathee, Venktesh V, Sean MacAvaney, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2508.15437)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a standard framework for knowledge-intensive NLP tasks, combining large language models (LLMs) with document retrieval from external corpora. Despite its widespread use, most RAG pipelines continue to treat retrieval and reasoning as isolated components, retrieving documents once and then generating answers without further interaction. This static design often limits performance on complex tasks that require iterative evidence gathering or high-precision retrieval. Recent work in both the information retrieval (IR) and NLP communities has begun to close this gap by introducing adaptive retrieval and ranking methods that incorporate feedback. In this survey, we present a structured overview of advanced retrieval and ranking mechanisms that integrate such feedback. We categorize feedback signals based on their source and role in improving the query, retrieved context, or document pool. By consolidating these developments, we aim to bridge IR and NLP perspectives and highlight retrieval as a dynamic, learnable component of end-to-end RAG systems. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为知识密集型NLP任务的标准框架，结合了大型语言模型（LLMs）和从外部语料库检索文档。尽管其广泛应用，大多数RAG管道仍继续将检索和推理视为独立组件，只检索一次文档，然后生成答案而不再进一步交互。这种静态设计往往限制了在需要迭代证据收集或高精度检索的复杂任务上的性能。信息检索（IR）和NLP领域近期工作已经开始通过引入适应性检索和排名方法来克服这一局限，这些方法包括反馈机制。在这篇综述中，我们提供了一个结构化的高级检索和排名机制概述，这些机制整合了反馈。我们根据反馈信号的来源及其在改进查询、检索上下文或文档池方面的作用对其进行分类。通过整合这些发展，我们旨在弥合IR和NLP视角，并突出检索作为端到端RAG系统中的动态、可学习组件的重要性。 

---
# LLaSO: A Foundational Framework for Reproducible Research in Large Language and Speech Model 

**Title (ZH)**: LLaSO: 用于大规模语言和语音模型可再现研究的基本框架 

**Authors**: Yirong Sun, Yizhong Geng, Peidong Wei, Yanjun Chen, Jinghan Yang, Rongfei Chen, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15418)  

**Abstract**: The development of Large Speech-Language Models (LSLMs) has been slowed by fragmented architectures and a lack of transparency, hindering the systematic comparison and reproducibility of research. Unlike in the vision-language domain, the LSLM field suffers from the common practice of releasing model weights without their corresponding training data and configurations. To address these critical gaps, we introduce LLaSO, the first fully open, end-to-end framework for large-scale speech-language modeling. LLaSO provides the community with three essential resources: (1) LLaSO-Align, a 12M-instance speech-text alignment corpus; (2) LLaSO-Instruct, a 13.5M-instance multi-task instruction-tuning dataset; and (3) LLaSO-Eval, a reproducible benchmark for standardized evaluation. To validate our framework, we build and release LLaSO-Base, a 3.8B-parameter reference model trained exclusively on our public data. It achieves a normalized score of 0.72, establishing a strong, reproducible baseline that surpasses comparable models. Our analysis reveals that while broader training coverage enhances performance, significant generalization gaps persist on unseen tasks, particularly in pure audio scenarios. By releasing the complete stack of data, benchmarks, and models, LLaSO establishes a foundational open standard to unify research efforts and accelerate community-driven progress in LSLMs. We release the code, dataset, pretrained models, and results in this https URL. 

**Abstract (ZH)**: 大型语音语言模型（LSLMs）的发展受到碎片化架构和缺乏透明性的阻碍，妨碍了研究的系统比较和再现性。与视觉语言领域不同，LSLM领域普遍存在问题，即在不发布其对应训练数据和配置的情况下发布模型权重。为填补这些关键空白，我们介绍了LLaSO，这是首个全面开源的端到端大规模语音语言建模框架。LLaSO为社区提供了三项重要资源：（1）LLaSO-Align，一个包含1200万实例的语音-文本对齐语料库；（2）LLaSO-Instruct，一个包含1350万实例的多任务指令调整数据集；（3）LLaSO-Eval，一个可重现的标准评估基准。为验证我们的框架，我们构建并发布了LLaSO-Base，一个仅在我们公开数据上训练的380亿参数参考模型，其标准化得分为0.72，建立了强健的可再现基线，超越了同类模型。我们的分析表明，虽然更广泛的训练覆盖面可以提高性能，但在未见任务上仍存在显著的泛化差距，特别是在纯音频场景中。通过发布完整的数据集、基准和模型堆栈，LLaSO建立了统一研究努力的基础开放标准，加速了社区驱动的大规模语音语言模型进展。我们在https://link.toLLLLO.release/发布了代码、数据集、预训练模型和结果。 

---
# When Audio and Text Disagree: Revealing Text Bias in Large Audio-Language Models 

**Title (ZH)**: 当音频和文本不一致时：揭示大型音频语言模型中的文本偏见 

**Authors**: Cheng Wang, Gelei Deng, Xianglin Yang, Han Qiu, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15407)  

**Abstract**: Large Audio-Language Models (LALMs) are enhanced with audio perception capabilities, enabling them to effectively process and understand multimodal inputs that combine audio and text. However, their performance in handling conflicting information between audio and text modalities remains largely unexamined. This paper introduces MCR-BENCH, the first comprehensive benchmark specifically designed to evaluate how LALMs prioritize information when presented with inconsistent audio-text pairs. Through extensive evaluation across diverse audio understanding tasks, we reveal a concerning phenomenon: when inconsistencies exist between modalities, LALMs display a significant bias toward textual input, frequently disregarding audio evidence. This tendency leads to substantial performance degradation in audio-centric tasks and raises important reliability concerns for real-world applications. We further investigate the influencing factors of text bias, and explore mitigation strategies through supervised finetuning, and analyze model confidence patterns that reveal persistent overconfidence even with contradictory inputs. These findings underscore the need for improved modality balance during training and more sophisticated fusion mechanisms to enhance the robustness when handling conflicting multi-modal inputs. The project is available at this https URL. 

**Abstract (ZH)**: 大型音频-语言模型（LALMs）通过增强音频感知能力，能够有效处理和理解结合了音频和文本的多模态输入。然而，它们在处理音频和文本模态之间矛盾信息方面的表现尚未得到充分研究。本文介绍了MCR-BENCH，这是首个专门设计用于评估LALMs在面对不一致的音频-文本配对时如何优先处理信息的综合性基准测试。通过在多样化的音频理解任务中进行广泛评估，我们揭示了一个令人担忧的现象：当模态之间存在不一致时，LALMs表现出对文本输入的显著偏向，经常忽视音频证据。这种倾向导致了以音频为中心任务的重大性能下降，并对实际应用的可靠性提出了重要质疑。我们进一步探讨了文本偏向的影响因素，并通过监督微调探索缓解策略，分析模型信心模式，揭示即使在矛盾输入的情况下，模型仍然表现出顽固的高信心。这些发现强调了在训练中改进模态平衡和开发更复杂的融合机制的必要性，以增强处理矛盾多模态输入的健壮性。该项目可在以下链接访问：此httpsURL。 

---
# Unveiling Trust in Multimodal Large Language Models: Evaluation, Analysis, and Mitigation 

**Title (ZH)**: 揭示多模态大型语言模型中的信任：评估、分析与缓解 

**Authors**: Yichi Zhang, Yao Huang, Yifan Wang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15370)  

**Abstract**: The trustworthiness of Multimodal Large Language Models (MLLMs) remains an intense concern despite the significant progress in their capabilities. Existing evaluation and mitigation approaches often focus on narrow aspects and overlook risks introduced by the multimodality. To tackle these challenges, we propose MultiTrust-X, a comprehensive benchmark for evaluating, analyzing, and mitigating the trustworthiness issues of MLLMs. We define a three-dimensional framework, encompassing five trustworthiness aspects which include truthfulness, robustness, safety, fairness, and privacy; two novel risk types covering multimodal risks and cross-modal impacts; and various mitigation strategies from the perspectives of data, model architecture, training, and inference algorithms. Based on the taxonomy, MultiTrust-X includes 32 tasks and 28 curated datasets, enabling holistic evaluations over 30 open-source and proprietary MLLMs and in-depth analysis with 8 representative mitigation methods. Our extensive experiments reveal significant vulnerabilities in current models, including a gap between trustworthiness and general capabilities, as well as the amplification of potential risks in base LLMs by both multimodal training and inference. Moreover, our controlled analysis uncovers key limitations in existing mitigation strategies that, while some methods yield improvements in specific aspects, few effectively address overall trustworthiness, and many introduce unexpected trade-offs that compromise model utility. These findings also provide practical insights for future improvements, such as the benefits of reasoning to better balance safety and performance. Based on these insights, we introduce a Reasoning-Enhanced Safety Alignment (RESA) approach that equips the model with chain-of-thought reasoning ability to discover the underlying risks, achieving state-of-the-art results. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）的可靠性依然是一个严峻的关注焦点，尽管其能力已经取得了显著进步。现有的评估和缓解方法往往关注狭窄方面，忽视了多模态带来的一些风险。为应对这些挑战，我们提出MultiTrust-X，一个全面的测评基准，用于评估、分析和缓解MLLMs的可靠性问题。我们定义了一个三维框架，涵盖了真实性、稳健性、安全性、公平性和隐私等五大可靠性方面；涵盖多模态风险和跨模态影响的两种新型风险类型；以及从数据、模型结构、训练和推理算法等多个视角的缓解策略。基于这一分类，MultiTrust-X 包含32个任务和28个精选数据集，能够在30个开源和专有MLLM上进行全面评估，并使用8种代表性缓解方法进行深入分析。广泛的实验证明了当前模型存在显著的脆弱性，包括可靠性与通用能力之间的差距，以及通过多模态训练和推理放大基LLM的潜在风险。此外，我们通过受控分析揭示了现有缓解策略的关键局限性——尽管某些方法能在特定方面取得改进，但很少有方法能有效解决整体可靠性问题，且许多方法引入了意想不到的权衡，损害了模型的实用性。这些发现也为未来改进提供了实用见解，如推理在更好地平衡安全性和性能上的优势。在此基础上，我们提出了增强推理的安全对齐（RESA）方法，使模型具备链式推理能力以发现潜在风险，达到了现有最佳效果。 

---
# IPIGuard: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents 

**Title (ZH)**: IPIGuard：一种新型工具依赖图基的防护方法，对抗LLM代理中的间接提示注入攻击 

**Authors**: Hengyu An, Jinghuai Zhang, Tianyu Du, Chunyi Zhou, Qingming Li, Tao Lin, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.15310)  

**Abstract**: Large language model (LLM) agents are widely deployed in real-world applications, where they leverage tools to retrieve and manipulate external data for complex tasks. However, when interacting with untrusted data sources (e.g., fetching information from public websites), tool responses may contain injected instructions that covertly influence agent behaviors and lead to malicious outcomes, a threat referred to as Indirect Prompt Injection (IPI). Existing defenses typically rely on advanced prompting strategies or auxiliary detection models. While these methods have demonstrated some effectiveness, they fundamentally rely on assumptions about the model's inherent security, which lacks structural constraints on agent behaviors. As a result, agents still retain unrestricted access to tool invocations, leaving them vulnerable to stronger attack vectors that can bypass the security guardrails of the model. To prevent malicious tool invocations at the source, we propose a novel defensive task execution paradigm, called IPIGuard, which models the agents' task execution process as a traversal over a planned Tool Dependency Graph (TDG). By explicitly decoupling action planning from interaction with external data, IPIGuard significantly reduces unintended tool invocations triggered by injected instructions, thereby enhancing robustness against IPI attacks. Experiments on the AgentDojo benchmark show that IPIGuard achieves a superior balance between effectiveness and robustness, paving the way for the development of safer agentic systems in dynamic environments. 

**Abstract (ZH)**: IPIGuard：基于工具依赖图的任务执行防御范式以防范间接提示注入攻击 

---
# M-$LLM^3$REC: A Motivation-Aware User-Item Interaction Framework for Enhancing Recommendation Accuracy with LLMs 

**Title (ZH)**: M-$LLM^3$REC：一种基于动机感知的用户-物品交互框架，用于通过LLMs提升推荐准确性 

**Authors**: Lining Chen, Qingwen Zeng, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15262)  

**Abstract**: Recommendation systems have been essential for both user experience and platform efficiency by alleviating information overload and supporting decision-making. Traditional methods, i.e., content-based filtering, collaborative filtering, and deep learning, have achieved impressive results in recommendation systems. However, the cold-start and sparse-data scenarios are still challenging to deal with. Existing solutions either generate pseudo-interaction sequence, which often introduces redundant or noisy signals, or rely heavily on semantic similarity, overlooking dynamic shifts in user motivation. To address these limitations, this paper proposes a novel recommendation framework, termed M-$LLM^3$REC, which leverages large language models for deep motivational signal extraction from limited user interactions. M-$LLM^3$REC comprises three integrated modules: the Motivation-Oriented Profile Extractor (MOPE), Motivation-Oriented Trait Encoder (MOTE), and Motivational Alignment Recommender (MAR). By emphasizing motivation-driven semantic modeling, M-$LLM^3$REC demonstrates robust, personalized, and generalizable recommendations, particularly boosting performance in cold-start situations in comparison with the state-of-the-art frameworks. 

**Abstract (ZH)**: 推荐系统通过减轻信息过载和支持决策制定，对于提升用户体验和平台效率至关重要。传统的基于内容过滤、协同过滤和深度学习的方法已经在推荐系统中取得了令人印象深刻的成果。然而，在冷启动和稀疏数据场景下仍存在挑战。现有的解决方案要么生成伪交互序列，这通常引入冗余或噪声信号，要么过度依赖语义相似性，忽视了用户动机的动态变化。为解决这些局限性，本文提出了一种新的推荐框架，称为M-$LLM^3$REC，该框架利用大型语言模型从有限的用户交互中提取深层次的动力信号。M-$LLM^3$REC 包含三个集成模块：动机导向资料提取器（MOPE）、动机导向特征编码器（MOTE）以及动力对齐推荐器（MAR）。通过强调基于动机的语义建模，M-$LLM^3$REC 展示了稳健、个性化和泛化的推荐能力，特别是在冷启动情况下，相对于现有最先进的框架显著提升了性能。 

---
# Conflict-Aware Soft Prompting for Retrieval-Augmented Generation 

**Title (ZH)**: 冲突意识软提示增强检索生成 

**Authors**: Eunseong Choi, June Park, Hyeri Lee, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.15253)  

**Abstract**: Retrieval-augmented generation (RAG) enhances the capabilities of large language models (LLMs) by incorporating external knowledge into their input prompts. However, when the retrieved context contradicts the LLM's parametric knowledge, it often fails to resolve the conflict between incorrect external context and correct parametric knowledge, known as context-memory conflict. To tackle this problem, we introduce Conflict-Aware REtrieval-Augmented Generation (CARE), consisting of a context assessor and a base LLM. The context assessor encodes compact memory token embeddings from raw context tokens. Through grounded/adversarial soft prompting, the context assessor is trained to discern unreliable context and capture a guidance signal that directs reasoning toward the more reliable knowledge source. Extensive experiments show that CARE effectively mitigates context-memory conflicts, leading to an average performance gain of 5.0\% on QA and fact-checking benchmarks, establishing a promising direction for trustworthy and adaptive RAG systems. 

**Abstract (ZH)**: 冲突aware检索增强生成（CARE）：缓解上下文记忆冲突的研究 

---
# VocabTailor: Dynamic Vocabulary Selection for Downstream Tasks in Small Language Models 

**Title (ZH)**: VocabTailor: 小规模语言模型中下游任务的动态词汇选择 

**Authors**: Hanling Zhang, Yayu Zhou, Tongcheng Fang, Zhihang Yuan, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15229)  

**Abstract**: Small Language Models (SLMs) provide computational advantages in resource-constrained environments, yet memory limitations remain a critical bottleneck for edge device deployment. A substantial portion of SLMs' memory footprint stems from vocabulary-related components, particularly embeddings and language modeling (LM) heads, due to large vocabulary sizes. Existing static vocabulary pruning, while reducing memory usage, suffers from rigid, one-size-fits-all designs that cause information loss from the prefill stage and a lack of flexibility. In this work, we identify two key principles underlying the vocabulary reduction challenge: the lexical locality principle, the observation that only a small subset of tokens is required during any single inference, and the asymmetry in computational characteristics between vocabulary-related components of SLM. Based on these insights, we introduce VocabTailor, a novel decoupled dynamic vocabulary selection framework that addresses memory constraints through offloading embedding and implements a hybrid static-dynamic vocabulary selection strategy for LM Head, enabling on-demand loading of vocabulary components. Comprehensive experiments across diverse downstream tasks demonstrate that VocabTailor achieves a reduction of up to 99% in the memory usage of vocabulary-related components with minimal or no degradation in task performance, substantially outperforming existing static vocabulary pruning. 

**Abstract (ZH)**: 小语言模型通过在资源受限环境中提供计算优势，为边缘设备部署带来了潜在的好处，然而内存限制仍然是一个关键瓶颈。小语言模型中的内存占用大量来自与词汇相关的组件，特别是嵌入和语言模型头，由于词汇表规模庞大。现有的静态词汇剪枝虽然减少了内存使用，但其僵化的、一刀切的设计导致了预填充阶段的信息损失以及缺乏灵活性。在本文中，我们识别出词汇缩减面临的两个关键原则：词汇局部性原则，即在任何单一推理过程中只需要一小部分标记；以及词汇相关组件的计算特性之间的不对称性。基于这些见解，我们引入了VocabTailor，这是一种新颖的解耦动态词汇选择框架，通过卸载嵌入来解决内存限制问题，并对LM头实施混合静态-动态词汇选择策略，实现按需加载词汇组件。在涉及多样下游任务的综合实验中，VocabTailor实现了词汇相关组件内存使用最多99%的减少，同时任务性能略有或无退步，在现有静态词汇剪枝方法上具有显著优势。 

---
# GenTune: Toward Traceable Prompts to Improve Controllability of Image Refinement in Environment Design 

**Title (ZH)**: GenTune: 向可追溯提示的方向改进环境设计中图像细化的可控性 

**Authors**: Wen-Fan Wang, Ting-Ying Lee, Chien-Ting Lu, Che-Wei Hsu, Nil Ponsa Campany, Yu Chen, Mike Y. Chen, Bing-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15227)  

**Abstract**: Environment designers in the entertainment industry create imaginative 2D and 3D scenes for games, films, and television, requiring both fine-grained control of specific details and consistent global coherence. Designers have increasingly integrated generative AI into their workflows, often relying on large language models (LLMs) to expand user prompts for text-to-image generation, then iteratively refining those prompts and applying inpainting. However, our formative study with 10 designers surfaced two key challenges: (1) the lengthy LLM-generated prompts make it difficult to understand and isolate the keywords that must be revised for specific visual elements; and (2) while inpainting supports localized edits, it can struggle with global consistency and correctness. Based on these insights, we present GenTune, an approach that enhances human--AI collaboration by clarifying how AI-generated prompts map to image content. Our GenTune system lets designers select any element in a generated image, trace it back to the corresponding prompt labels, and revise those labels to guide precise yet globally consistent image refinement. In a summative study with 20 designers, GenTune significantly improved prompt--image comprehension, refinement quality, and efficiency, and overall satisfaction (all $p < .01$) compared to current practice. A follow-up field study with two studios further demonstrated its effectiveness in real-world settings. 

**Abstract (ZH)**: 环境设计师在娱乐行业中创建游戏、电影和电视的想象中的2D和3D场景，需要对详细细节进行精细控制并保持全局的一致性。设计师们越来越多地将生成式AI集成到他们的工作流程中，通常依赖大规模语言模型（LLMs）扩展文本到图像生成的用户提示，然后迭代地精炼这些提示并应用修补技术。然而，我们针对10名设计师的形成性研究揭示了两个关键挑战：（1）LLM生成的提示过于冗长，使得理解和隔离需要修订的具体视觉元素的关键词变得困难；（2）尽管修补技术支持局部编辑，但它在全局一致性和准确性方面存在问题。基于这些见解，我们提出了一种名为GenTune的方法，通过阐明AI生成的提示如何映射到图像内容来增强人-机协作。GenTune系统使设计师能够选择生成图像中的任何元素，追溯回相应的提示标签，并修改这些标签以指导精确且全局一致的图像精修。在20名设计师的总结性研究中，GenTune显著提高了提示-图像理解、精修质量、效率及总体满意度（所有$p < .01$），相较于现有做法。后续的实地研究进一步证明了其在实际场景中的有效性。 

---
# SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning 

**Title (ZH)**: SparK: 查询感知的可恢复KV缓存通道无结构稀疏性剪枝 

**Authors**: Huanxuan Liao, Yixing Xu, Shizhu He, Guanchen Li, Xuanwu Yin, Dong Li, Emad Barsoum, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15212)  

**Abstract**: Long-context inference in large language models (LLMs) is increasingly constrained by the KV cache bottleneck: memory usage grows linearly with sequence length, while attention computation scales quadratically. Existing approaches address this issue by compressing the KV cache along the temporal axis through strategies such as token eviction or merging to reduce memory and computational overhead. However, these methods often neglect fine-grained importance variations across feature dimensions (i.e., the channel axis), thereby limiting their ability to effectively balance efficiency and model accuracy. In reality, we observe that channel saliency varies dramatically across both queries and positions: certain feature channels carry near-zero information for a given query, while others spike in relevance. To address this oversight, we propose SPARK, a training-free plug-and-play method that applies unstructured sparsity by pruning KV at the channel level, while dynamically restoring the pruned entries during attention score computation. Notably, our approach is orthogonal to existing KV compression and quantization techniques, making it compatible for integration with them to achieve further acceleration. By reducing channel-level redundancy, SPARK enables processing of longer sequences within the same memory budget. For sequences of equal length, SPARK not only preserves or improves model accuracy but also reduces KV cache storage by over 30% compared to eviction-based methods. Furthermore, even with an aggressive pruning ratio of 80%, SPARK maintains performance with less degradation than 5% compared to the baseline eviction method, demonstrating its robustness and effectiveness. Our code will be available at this https URL. 

**Abstract (ZH)**: 长上下文推理中大型语言模型的内存瓶颈：通过时间轴上的密钥值缓存压缩策略来缓解KV缓存瓶颈，但忽略特征维度上的细粒度重要性变化 

---
# SemToken: Semantic-Aware Tokenization for Efficient Long-Context Language Modeling 

**Title (ZH)**: SemToken：面向高效长上下文语言建模的语义感知分词方法 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15190)  

**Abstract**: Tokenization plays a critical role in language modeling, yet existing approaches such as Byte-Pair Encoding (BPE) or WordPiece operate purely on frequency statistics, ignoring the underlying semantic structure of text. This leads to over-tokenization of semantically redundant spans and underutilization of contextual coherence, particularly in long-context scenarios. In this work, we propose \textbf{SemToken}, a semantic-aware tokenization framework that jointly reduces token redundancy and improves computation efficiency. SemToken first extracts contextual semantic embeddings via lightweight encoders and performs local semantic clustering to merge semantically equivalent tokens. Then, it allocates heterogeneous token granularity based on semantic density, allowing finer-grained tokenization in content-rich regions and coarser compression in repetitive or low-entropy spans. SemToken can be seamlessly integrated with modern language models and attention acceleration methods. Experiments on long-context language modeling benchmarks such as WikiText-103 and LongBench show that SemToken achieves up to $2.4\times$ reduction in token count and $1.9\times$ speedup, with negligible or no degradation in perplexity and downstream accuracy. Our findings suggest that semantic structure offers a promising new axis for optimizing tokenization and computation in large language models. 

**Abstract (ZH)**: 基于语义的分词框架：SemToken及其在语言模型中的应用 

---
# Hydra: A 1.6B-Parameter State-Space Language Model with Sparse Attention, Mixture-of-Experts, and Memory 

**Title (ZH)**: Hydra：一个具有稀疏注意、专家混合和记忆的16亿参数状态空间语言模型 

**Authors**: Siddharth Chaudhary, Bennett Browning  

**Link**: [PDF](https://arxiv.org/pdf/2508.15099)  

**Abstract**: We present Hydra as an architectural proposal for hybrid long-context language models that combine conditional computation, long-context memory mechanisms, and sparse mixture-of-experts within an approximately 1.6B parameter design envelope. Hydra integrates a Mamba-style Structured State Space Model (SSM) backbone with intermittent sparse global attention, chunk-level MoE feed-forward routing, and dual (workspace plus factual PKM) memories. We formalize the component interfaces, give transparent parameter and complexity accounting, and outline a staged curriculum intended to stably activate the parts. We accompany the specification with illustrative toy-scale prototype measurements (tens of millions of parameters on synthetic data) whose sole purpose is to demonstrate implementation feasibility and qualitative scaling behaviors (for example, long-context throughput crossover and controllable expert routing), not to claim competitive full-scale performance. We explicitly delineate assumptions and open risks (training complexity, memory utilization, specialization dynamics) and position Hydra as a blueprint to stimulate empirical follow-up rather than a finished system. By combining SSM efficiency, selective sparse attention, MoE capacity, and learnable memory, Hydra sketches a path toward modular, input-adaptive long-context language models; validating end-task gains at target scale remains future work. 

**Abstract (ZH)**: Hydra：一种结合条件计算、长上下文记忆机制和稀疏专家混合的混合架构提案 

---
# Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset 

**Title (ZH)**: Nemotron-CC-Math：一个大规模高质量数学预训练数据集（133亿词 token 规模） 

**Authors**: Rabeeh Karimi Mahabadi, Sanjeev Satheesh, Shrimai Prabhumoye, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2508.15096)  

**Abstract**: Pretraining large language models (LLMs) on high-quality, structured data such as mathematics and code substantially enhances reasoning capabilities. However, existing math-focused datasets built from Common Crawl suffer from degraded quality due to brittle extraction heuristics, lossy HTML-to-text conversion, and the failure to reliably preserve mathematical structure. In this work, we introduce Nemotron-CC-Math, a large-scale, high-quality mathematical corpus constructed from Common Crawl using a novel, domain-agnostic pipeline specifically designed for robust scientific text extraction.
Unlike previous efforts, our pipeline recovers math across various formats (e.g., MathJax, KaTeX, MathML) by leveraging layout-aware rendering with lynx and a targeted LLM-based cleaning stage. This approach preserves the structural integrity of equations and code blocks while removing boilerplate, standardizing notation into LaTeX representation, and correcting inconsistencies.
We collected a large, high-quality math corpus, namely Nemotron-CC-Math-3+ (133B tokens) and Nemotron-CC-Math-4+ (52B tokens). Notably, Nemotron-CC-Math-4+ not only surpasses all prior open math datasets-including MegaMath, FineMath, and OpenWebMath-but also contains 5.5 times more tokens than FineMath-4+, which was previously the highest-quality math pretraining dataset. When used to pretrain a Nemotron-T 8B model, our corpus yields +4.8 to +12.6 gains on MATH and +4.6 to +14.3 gains on MBPP+ over strong baselines, while also improving general-domain performance on MMLU and MMLU-Stem.
We present the first pipeline to reliably extract scientific content--including math--from noisy web-scale data, yielding measurable gains in math, code, and general reasoning, and setting a new state of the art among open math pretraining corpora. To support open-source efforts, we release our code and datasets. 

**Abstract (ZH)**: 预训练大型语言模型（LLMs）在高质量、结构化的数据上，如数学和代码，极大地提升了推理能力。然而，现有专注于数学的数据集由于提取 heuristic 的脆弱性、HTML 转换的损失以及数学结构保真的失败，其质量存在下降。在本工作中，我们引入了 Nemotron-CC-Math，这是一种使用新颖的、跨领域的管道从 Common Crawl 构建的大规模高质量数学语料库，该管道专门设计用于稳健的科学文本提取。

我们的管道通过利用 lynx 进行布局感知渲染，并结合目标 LLM 基础的清理阶段，能够恢复各种格式（如 MathJax、KaTeX、MathML）的数学内容。这种方法在保留等式和代码块结构完整性的基础上，去除了冗余内容，将符号标准化为 LaTeX 表示，并纠正了不一致性。

我们收集了一个大规模、高质量的数学语料库，命名为 Nemotron-CC-Math-3+（133B 词元）和 Nemotron-CC-Math-4+（52B 词元）。值得注意的是，Nemotron-CC-Math-4+ 不仅超越了所有先前的开放数学数据集（包括 MegaMath、FineMath 和 OpenWebMath），而且其词元数量是 FineMath-4+ 的 5.5 倍。当用于预训练 Nemotron-T 8B 模型时，我们的语料库在 MATH 上带来了 4.8 到 12.6 的增益，在 MBPP+ 上带来了 4.6 到 14.3 的增益，同时在 MMLU 和 MMLU-Stem 的通用领域性能方面也有所提升。

我们首次提出了一种可靠地从嘈杂的网络数据中提取科学内容（包括数学）的方法，带来了数学、代码和一般推理能力的可测量提升，并在开放数学预训练语料库中设定了新的基准。为了支持开源努力，我们发布了代码和数据集。 

---
# Mapping the Course for Prompt-based Structured Prediction 

**Title (ZH)**: 基于提示的结构化预测框架研究 

**Authors**: Matt Pauk, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2508.15090)  

**Abstract**: LLMs have been shown to be useful for a variety of language tasks, without requiring task-specific fine-tuning. However, these models often struggle with hallucinations and complex reasoning problems due to their autoregressive nature. We propose to address some of these issues, specifically in the area of structured prediction, by combining LLMs with combinatorial inference in an attempt to marry the predictive power of LLMs with the structural consistency provided by inference methods. We perform exhaustive experiments in an effort to understand which prompting strategies can effectively estimate LLM confidence values for use with symbolic inference, and show that, regardless of the prompting strategy, the addition of symbolic inference on top of prompting alone leads to more consistent and accurate predictions. Additionally, we show that calibration and fine-tuning using structured prediction objectives leads to increased performance for challenging tasks, showing that structured learning is still valuable in the era of LLMs. 

**Abstract (ZH)**: LLMs在结构化预测中的组合推理：通过结合LLMs和组合推理解决幻觉和复杂推理问题 

---
# MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs 

**Title (ZH)**: MoEcho：利用侧信道攻击以侵犯混合专家大语言模型中用户隐私 

**Authors**: Ruyi Ding, Tianhong Xu, Xinyi Shen, Aidong Adam Ding, Yunsi Fei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15036)  

**Abstract**: The transformer architecture has become a cornerstone of modern AI, fueling remarkable progress across applications in natural language processing, computer vision, and multimodal learning. As these models continue to scale explosively for performance, implementation efficiency remains a critical challenge. Mixture of Experts (MoE) architectures, selectively activating specialized subnetworks (experts), offer a unique balance between model accuracy and computational cost. However, the adaptive routing in MoE architectures, where input tokens are dynamically directed to specialized experts based on their semantic meaning inadvertently opens up a new attack surface for privacy breaches. These input-dependent activation patterns leave distinctive temporal and spatial traces in hardware execution, which adversaries could exploit to deduce sensitive user data. In this work, we propose MoEcho, discovering a side channel analysis based attack surface that compromises user privacy on MoE based systems. Specifically, in MoEcho, we introduce four novel architectural side channels on different computing platforms, including Cache Occupancy Channels and Pageout+Reload on CPUs, and Performance Counter and TLB Evict+Reload on GPUs, respectively. Exploiting these vulnerabilities, we propose four attacks that effectively breach user privacy in large language models (LLMs) and vision language models (VLMs) based on MoE architectures: Prompt Inference Attack, Response Reconstruction Attack, Visual Inference Attack, and Visual Reconstruction Attack. MoEcho is the first runtime architecture level security analysis of the popular MoE structure common in modern transformers, highlighting a serious security and privacy threat and calling for effective and timely safeguards when harnessing MoE based models for developing efficient large scale AI services. 

**Abstract (ZH)**: 基于MoE架构的MoEcho：发现并分析其侧信道攻击面以保障用户隐私 

---
# Disentangling the Drivers of LLM Social Conformity: An Uncertainty-Moderated Dual-Process Mechanism 

**Title (ZH)**: 分离大语言模型社会 conformity 的驱动因素：一种不确定性调节的双过程机制 

**Authors**: Huixin Zhong, Yanan Liu, Qi Cao, Shijin Wang, Zijing Ye, Zimu Wang, Shiyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14918)  

**Abstract**: As large language models (LLMs) integrate into collaborative teams, their social conformity -- the tendency to align with majority opinions -- has emerged as a key concern. In humans, conformity arises from informational influence (rational use of group cues for accuracy) or normative influence (social pressure for approval), with uncertainty moderating this balance by shifting from purely analytical to heuristic processing. It remains unclear whether these human psychological mechanisms apply to LLMs. This study adapts the information cascade paradigm from behavioral economics to quantitatively disentangle the two drivers to investigate the moderate effect. We evaluated nine leading LLMs across three decision-making scenarios (medical, legal, investment), manipulating information uncertainty (q = 0.667, 0.55, and 0.70, respectively). Our results indicate that informational influence underpins the models' behavior across all contexts, with accuracy and confidence consistently rising with stronger evidence. However, this foundational mechanism is dramatically modulated by uncertainty. In low-to-medium uncertainty scenarios, this informational process is expressed as a conservative strategy, where LLMs systematically underweight all evidence sources. In contrast, high uncertainty triggers a critical shift: while still processing information, the models additionally exhibit a normative-like amplification, causing them to overweight public signals (beta > 1.55 vs. private beta = 0.81). 

**Abstract (ZH)**: 大规模语言模型（LLMs）融入协作团队后，其社会 conformity 的影响——倾向于与多数意见一致——已成为一个重要关切。本研究借鉴行为经济学中的信息级联范式，定量解析信息影响和社会影响之间的关系，以探讨其平衡的中等影响。我们评估了九种领先的LLMs在三种决策场景（医疗、法律、投资）中的表现，并操纵信息不确定性（分别为q = 0.667、0.55和0.70）。结果表明，在所有情境下，信息影响支撑着模型的行为，准确性和信心随着更强证据的一致性而提升。然而，这一基础机制在不确定性的影响下发生了显著变化。在低至中等不确定性情境中，这一信息过程表现为保守策略，LLMs系统性地低估了所有证据来源。相反，在高不确定性情境下，这种信息处理触发了一种关键转变：模型除了继续处理信息外，还会表现出类似规范影响的放大效应，导致它们过度重视公开信号（β > 1.55，对比私人信号β = 0.81）。 

---
# Transsion Multilingual Speech Recognition System for MLC-SLM 2025 Challenge 

**Title (ZH)**: Transsion 多语种语音识别系统for MLC-SLM 2025 挑战赛 

**Authors**: Xiaoxiao Li, An Zhu, Youhai Jiang, Fengjie Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14916)  

**Abstract**: This paper presents the architecture and performance of a novel Multilingual Automatic Speech Recognition (ASR) system developed by the Transsion Speech Team for Track 1 of the MLC-SLM 2025 Challenge. The proposed system comprises three key components: 1) a frozen Whisper-large-v3 based speech encoder, leveraging large-scale pretraining to ensure robust acoustic feature extraction; 2) a trainable adaptor module using Linear-ReLU-Linear transformation mechanisms to effectively align speech and text representations; and 3) a frozen Qwen2.5-7B-Instruct large language model (LLM) integrated with trainable LoRA for optimized contextual linguistic decoding. By systematically combining pretrained models with task specific fine-tuning, the system achieved a word/character error rate (WER/CER) of 9.83% across 11 languages in the evaluation set and ranked third place among global participants. 

**Abstract (ZH)**: 本论文介绍了传音言语团队为2025年MLC-SLM挑战赛Track 1开发的新型多语言自动语音识别(ASR)系统的设计与性能。该系统包括三个关键组件：1）基于Whisper-large-v3的冻结语音编码器，利用大规模预训练确保稳健的声学特征提取；2）使用Linear-ReLU-Linear变换机制的可训练适配器模块，以有效地对齐语音和文本表示；以及3）与可训练LoRA集成的冻结Qwen2.5-7B-Instruct大型语言模型（LLM），以优化上下文语言解码。通过系统地结合预训练模型和任务特定微调，该系统在评价集中的11种语言上实现了9.83%的字错误率/字符错误率（WER/CER），并在全球参赛者中排名第三。 

---
# Efficient Switchable Safety Control in LLMs via Magic-Token-Guided Co-Training 

**Title (ZH)**: 通过魔法令牌引导的协同训练实现LLMs中的高效可切换安全性控制 

**Authors**: Jianfeng Si, Lin Sun, Zhewen Tan, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14904)  

**Abstract**: Current methods for content safety in Large Language Models (LLMs), such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often rely on multi-stage training pipelines and lack fine-grained, post-deployment controllability. To address these limitations, we propose a unified co-training framework that efficiently integrates multiple safety behaviors: positive (lawful/prosocial), negative (unfiltered/risk-prone) and rejective (refusal-oriented/conservative) within a single SFT stage. Notably, each behavior is dynamically activated via a simple system-level instruction, or magic token, enabling stealthy and efficient behavioral switching at inference time. This flexibility supports diverse deployment scenarios, such as positive for safe user interaction, negative for internal red-teaming, and rejective for context-aware refusals triggered by upstream moderation signals. This co-training strategy induces a distinct Safety Alignment Margin in the output space, characterized by well-separated response distributions corresponding to each safety mode. The existence of this margin provides empirical evidence for the model's safety robustness and enables unprecedented fine-grained control. Experiments show that our method matches the safety alignment quality of SFT+DPO, with our 8B model notably surpassing DeepSeek-R1 (671B) in safety performance, while significantly reducing both training complexity and deployment costs. This work presents a scalable, efficient, and highly controllable solution for LLM content safety. 

**Abstract (ZH)**: 当前大型语言模型内容安全方法，如监督微调(SFT)和基于人类反馈的强化学习(RLHF)，往往依赖多阶段训练管道并在部署后缺乏精细控制。为解决这些限制，我们提出了一种统一的联合训练框架，在单个SFT阶段中高效整合多种安全行为：正面（合法/亲社会）、负面（未经筛选/风险倾向）和拒绝（拒绝导向/保守）。值得注意的是，每种行为可通过简单的系统级指令或魔法标记动态激活，从而在推理时实现隐蔽且高效的切换行为。这种灵活性支持多种部署场景，例如正面用于安全用户交互、负面用于内部红队测试、拒绝用于根据上游审核信号触发的内容感知拒绝。这种联合训练策略在输出空间中诱导出一种独特的安全对齐边际，其特征是对应每种安全模式且区分良好的响应分布。边际的存在为模型的安全鲁棒性提供了实证证据，并使精细控制成为可能。实验显示，我们的方法在安全对齐质量上与SFT+DPO相当，8B模型在安全性能上显著超越DeepSeek-R1（671B），同时大幅降低了训练复杂性和部署成本。本工作提出了一种可扩展、高效且高度可控的大规模语言模型内容安全解决方案。 

---
